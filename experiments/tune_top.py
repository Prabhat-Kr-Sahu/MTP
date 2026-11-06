#!/usr/bin/env python3
"""Hypertune the best models, retrain each on its best params, report progress.

Per model: an Optuna TPE study over the training hyperparameters (val RMSE,
25 % of the train split per trial, median pruning), then one full-data
retrain with the best trial's params, evaluated on the test split and saved
as results/<model>_tuned.json -- the baseline results/<model>.json is never
touched. Models come from the notebooks via score_risk.load_notebook_model,
so nothing is re-implemented here.

Everything resumes:
  * the study lives in tuning/<model>/optuna.db; a rerun tops up to --trials
    finished (complete + pruned) trials;
  * a trial interrupted mid-run (kill, Ctrl-C, reboot) is re-queued with the
    same params on the next start and resumes from trial_<n>/last_checkpoint.pt;
  * the final retrain resumes from checkpoints/<model>_tuned/last_checkpoint.pt;
  * a model whose results/<model>_tuned.json exists is skipped.

Usage:
    python experiments/tune_top.py --list
    python experiments/tune_top.py --set all            # the real run
    python experiments/tune_top.py --status             # safe while it runs
    python experiments/tune_top.py miat --debug --trials 3 --fraction 0.05
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.benchmark import RESULTS_DIR, RISK_DIR, is_loose, save_result  # noqa: E402
from experiments.score_risk import EXPERIMENTS, MODEL_CELL, load_notebook_model  # noqa: E402
from src.config import (  # noqa: E402
    DEBUG_MAX_ROWS,
    DEVICE,
    INTENTIONS_FILE,
    NGSIM_LOCATIONS,
    NORMALIZATION_FILE,
    SEED,
)
from src.data.loader import NGSIMDataLoader, create_dataloader  # noqa: E402
from src.data.normalization import load_stats, normalize_samples  # noqa: E402
from src.evaluation.evaluate import evaluate_trajectory  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.pipeline import test_groups  # noqa: E402
from src.train import fit, prepare_data  # noqa: E402

logger = get_logger()

# Top 5 by mean pred-source ROC-AUC over the six headline measures, and top 5
# by test rmse_avg -- both read off the committed stage-1/2 results.
SETS = {
    "risk": ["inatran", "cognitive", "dacr_amtp", "miat", "mha_lstm"],
    "rmse": ["miat", "traj_mamba", "x_track", "dacr_amtp", "stdan"],
}
# Cheapest first (baseline full-run minutes 17 ... 189), so one slow model
# cannot hold the others back.
ORDER = ["dacr_amtp", "stdan", "inatran", "cognitive", "mha_lstm", "miat",
         "x_track", "traj_mamba"]
SEARCH_SPACE = {
    "learning_rate": "1e-4 .. 2e-3, log",
    "lr_decay": "0.6 .. 0.97",
    "batch_size": "{64, 128, 256}",
    "weight_decay": "1e-6 .. 1e-3, log",
    "num_epochs": "12 .. 30",
    "dropout": "0 .. 0.3 (only models whose notebook uses DROPOUT)",
}


def roots(debug):
    """(tuning, checkpoints, results) roots; debug runs never touch the real ones."""
    if debug:
        base = EXPERIMENTS / "tuning_debug"
        return base, base / "checkpoints", base / "results"
    return EXPERIMENTS / "tuning", EXPERIMENTS / "checkpoints", RESULTS_DIR


def uses_dropout(name):
    """Only 3 of the 8 notebooks read DROPOUT; tuning it elsewhere would be a
    dimension that does nothing but dilute the TPE search."""
    cells = json.loads((EXPERIMENTS / f"{name}.ipynb").read_text(encoding="utf-8"))["cells"]
    return "DROPOUT" in "".join(cells[MODEL_CELL]["source"])


def suggest(trial, dropout):
    """(fit kwargs, namespace overrides) for one trial."""
    hp = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        "lr_decay": trial.suggest_float("lr_decay", 0.6, 0.97),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 12, 30),
    }
    overrides = {"DROPOUT": trial.suggest_float("dropout", 0.0, 0.3)} if dropout else {}
    return hp, overrides


def split_params(params):
    """Inverse of suggest() for a finished trial's params dict."""
    hp = {k: v for k, v in params.items() if k != "dropout"}
    return hp, ({"DROPOUT": params["dropout"]} if "dropout" in params else {})


def storage_url(tune_dir):
    return f"sqlite:///{(tune_dir / 'optuna.db').as_posix()}"


def open_study(name, tune_dir):
    import optuna

    tune_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize", study_name=name, storage=storage_url(tune_dir),
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    # A fresh TPESampler(seed=SEED) replays the same random start-up draws, so a
    # restarted study would repeat its first trials; offsetting by the trials
    # already stored keeps it deterministic without repeats.
    study.sampler = optuna.samplers.TPESampler(seed=SEED + len(study.trials))
    return study


def finished(study):
    from optuna.trial import TrialState
    return [t for t in study.get_trials(deepcopy=False)
            if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]


def trial_dir(tune_dir, trial):
    """A resumed trial reuses the directory of the trial it resumes, which is
    what lets it pick up that trial's last_checkpoint.pt."""
    return tune_dir / f"trial_{trial.user_attrs.get('resume_of', trial.number)}"


def requeue_interrupted(study, tune_dir):
    """Re-queue, with the same params, every trial a previous run left behind.

    ponytail: one process per model is assumed (the queue runs models in
    turn), so at start-up any RUNNING trial is a dead one; two processes on
    the same model would steal each other's trials. A trial left FAIL with
    its checkpoint still on disk was interrupted by Ctrl-C (an OOM or a
    prune deletes the directory). Returns the number re-queued.
    """
    import optuna
    from optuna.trial import TrialState

    latest = {}                    # resume chain root -> its newest trial
    for t in study.get_trials(deepcopy=False):
        latest[t.user_attrs.get("resume_of", t.number)] = t
    count = 0
    for root, t in latest.items():
        interrupted = t.state == TrialState.RUNNING or (
            t.state == TrialState.FAIL
            and (tune_dir / f"trial_{root}" / "last_checkpoint.pt").exists())
        if not interrupted:
            continue
        if t.state == TrialState.RUNNING:
            study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
        study.add_trial(optuna.trial.create_trial(
            state=TrialState.WAITING, params=t.params, distributions=t.distributions,
            user_attrs={"resume_of": root}))
        count += 1
    return count


def tune(name, data, args):
    """Run the study for one model up to args.trials finished trials."""
    import optuna
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState
    from tqdm.auto import tqdm

    tune_root, _, _ = roots(args.debug)
    tune_dir = tune_root / name
    study = open_study(name, tune_dir)
    requeued = requeue_interrupted(study, tune_dir)
    if requeued:
        logger.info("[%s] re-queued %d interrupted trial(s)", name, requeued)
    done = len(finished(study))
    if done >= args.trials:
        logger.info("[%s] %d trials already finished", name, done)
        return study

    dropout = uses_dropout(name)
    train_s, val_s, centers, norm_stats, use_risk_loss = (
        data["train_sub"], data["val"], data["centers"], data["norm_stats"],
        data["use_risk_loss"][name])

    def objective(trial):
        hp, overrides = suggest(trial, dropout)
        directory = trial_dir(tune_dir, trial)
        resume = directory / "last_checkpoint.pt"
        torch.manual_seed(SEED)
        torch.cuda.empty_cache()            # free memory left by an OOM trial
        model, _ = load_notebook_model(name, device=DEVICE, load_weights=False,
                                       stats=norm_stats, overrides=overrides)

        def report(epoch, entry):
            trial.report(entry["val_rmse"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            log = fit(model, train_s, val_s, centers, save_dir=str(directory),
                      resume_from=str(resume) if resume.exists() else None,
                      checkpoint_every=0, on_epoch=report,
                      use_risk_loss=use_risk_loss, **hp)
        except (optuna.TrialPruned, torch.OutOfMemoryError):
            shutil.rmtree(directory, ignore_errors=True)
            raise
        # A crash or kill never reaches here, so the checkpoint survives for the retry.
        trial.set_user_attr("training_log", log)
        shutil.rmtree(directory, ignore_errors=True)
        return min(entry["val_rmse"] for entry in log)

    bar = tqdm(total=args.trials, initial=done, desc=f"{name} trials", mininterval=0)

    def tick(study, trial):
        if trial.state in (TrialState.COMPLETE, TrialState.PRUNED):
            bar.update(1)
        if any(t.state == TrialState.COMPLETE for t in study.get_trials(deepcopy=False)):
            bar.set_postfix(best=f"{study.best_value:.4f}")

    logger.info("[%s] tuning: %d/%d trials done, dropout tuned: %s", name, done,
                args.trials, dropout)
    study.optimize(
        objective, n_trials=2 * args.trials,      # hard cap if trials keep failing
        callbacks=[MaxTrialsCallback(args.trials, states=(TrialState.COMPLETE,
                                                          TrialState.PRUNED)), tick],
        catch=(torch.OutOfMemoryError,),
    )
    bar.close()
    export(study, tune_dir)
    return study


def export(study, tune_dir):
    """best_params.json and trials.csv: the committable record of a study."""
    frame = study.trials_dataframe(attrs=("number", "value", "params", "state",
                                          "duration", "user_attrs"))
    frame.drop(columns=[c for c in frame if c.startswith("user_attrs")]).to_csv(
        tune_dir / "trials.csv", index=False)
    best = study.best_trial
    (tune_dir / "best_params.json").write_text(json.dumps(
        {"trial": best.number, "val_rmse": best.value, "params": best.params}, indent=2))


def retrain(name, study, data, args):
    """Full-data fit on the best params, test RMSE, results/<name>_tuned.json."""
    _, ckpt_root, results_dir = roots(args.debug)
    tuned = f"{name}_tuned"
    if (results_dir / f"{tuned}.json").exists():
        logger.info("[%s] %s.json exists, skipping retrain", name, tuned)
        return
    best = study.best_trial
    hp, overrides = split_params(best.params)
    ckpt = ckpt_root / tuned
    ckpt.mkdir(parents=True, exist_ok=True)
    for file in (NORMALIZATION_FILE, INTENTIONS_FILE):   # score_risk reads them here
        shutil.copy(data["dir"] / file, ckpt / file)

    torch.manual_seed(SEED)
    model, namespace = load_notebook_model(name, device=DEVICE, load_weights=False,
                                           stats=data["norm_stats"], overrides=overrides)
    resume = ckpt / "last_checkpoint.pt"
    logger.info("[%s] retrain on full train split with trial %d: %s", name,
                best.number, best.params)
    started = time.time()
    log = fit(model, data["train"], data["val"], data["centers"], save_dir=str(ckpt),
              resume_from=str(resume) if resume.exists() else None,
              use_risk_loss=data["use_risk_loss"][name], **hp)
    minutes = (time.time() - started) / 60

    model.load_state_dict(torch.load(ckpt / "best_model.pt", map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    test_metrics = {}
    for label, samples in test_groups(data["test"]).items():
        loader = create_dataloader(samples, batch_size=32, shuffle=False)
        test_metrics[label] = {**evaluate_trajectory(model, loader, DEVICE), "n": len(samples)}

    trials = study.get_trials(deepcopy=False)
    save_result({
        "name": tuned,
        "description": f"{namespace['DESCRIPTION']} [tuned]",
        "model_class": type(model).__name__,
        "n_params": sum(p.numel() for p in model.parameters()),
        "hparams": {**hp, "use_risk_loss": data["use_risk_loss"][name]},
        "overrides": overrides,
        "tuning": {
            "base_model": name, "objective": "min val RMSE over epochs",
            "best_trial": best.number, "best_val_rmse_on_subset": best.value,
            "n_trials": len(trials),
            "states": {s: sum(t.state.name == s for t in trials)
                       for s in ("COMPLETE", "PRUNED", "FAIL")},
            "train_fraction": args.fraction, "search_space": SEARCH_SPACE,
        },
        "data": {"locations": data["locations"], "debug": args.debug},
        "train_minutes": minutes,          # this session only if the retrain resumed
        "training_log": log,
        "test": test_metrics,
        "collisions": None,
    }, results_dir=results_dir)


def load_data(models, args):
    """Split, normalise and fit the codebook once; every model and trial shares it."""
    tune_root, _, _ = roots(args.debug)
    locations = ["us-101"] if args.debug else NGSIM_LOCATIONS
    loader = NGSIMDataLoader(locations=locations,
                             max_rows=DEBUG_MAX_ROWS if args.debug else None)
    data_dir = tune_root / "_data"
    train_s, val_s, centers = prepare_data(loader, save_dir=str(data_dir))
    _, _, test_raw = loader.get_splits()
    norm_stats = load_stats(str(data_dir / NORMALIZATION_FILE))
    k = max(1, int(len(train_s) * args.fraction))
    logger.info("[tune] %d of %d train samples per trial, %d val", k, len(train_s), len(val_s))
    use_risk_loss = {}
    for name in models:          # each notebook's own loss choice (S3)
        _, namespace = load_notebook_model(name, load_weights=False, stats=norm_stats)
        use_risk_loss[name] = bool(namespace["HPARAMS"]["use_risk_loss"])
    return {
        "dir": data_dir, "locations": locations, "norm_stats": norm_stats,
        "train": train_s, "train_sub": random.Random(SEED).sample(train_s, k),
        "val": val_s, "centers": centers,
        "test": normalize_samples(test_raw, norm_stats), "use_risk_loss": use_risk_loss,
    }


# --------------------------------------------------------------------------- #
#  Progress
# --------------------------------------------------------------------------- #
def baseline_minutes(name):
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text())["train_minutes"] if path.exists() else float("nan")


def status_row(name, args):
    """One --status row. Read-only, so it is safe while the batch runs."""
    import optuna
    from optuna.trial import TrialState

    tune_root, ckpt_root, results_dir = roots(args.debug)
    row = {"model": name, "trials": f"0/{args.trials}", "pruned": 0, "failed": 0,
           "running": "-", "best_val_rmse": "-", "retrain": "-",
           "tuned_json": (results_dir / f"{name}_tuned.json").exists(),
           "risk_v2": ",".join(sorted(json.loads(p.read_text())["variants"]))
           if (p := RISK_DIR.parent / "risk_v2" / f"{name}_tuned.json").exists() else "-"}
    db = tune_root / name / "optuna.db"
    # Unstarted: ~fraction x mean 21 epochs / 12 x baseline, ~0.6 left by pruning.
    eta = args.trials * baseline_minutes(name) * args.fraction * 21 / 12 * 0.6
    if db.exists():
        study = optuna.load_study(study_name=name, storage=storage_url(tune_root / name))
        trials = study.get_trials(deepcopy=False)
        done = [t for t in trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]
        running = [t for t in trials if t.state == TrialState.RUNNING]
        row.update(trials=f"{len(done)}/{args.trials}",
                   pruned=sum(t.state == TrialState.PRUNED for t in trials),
                   failed=sum(t.state == TrialState.FAIL for t in trials))
        if running:
            t = running[0]
            row["running"] = f"#{t.number} ep {len(t.intermediate_values)}/{t.params.get('num_epochs', '?')}"
        if any(t.state == TrialState.COMPLETE for t in trials):
            row["best_val_rmse"] = f"{study.best_value:.4f}"
        durations = [(t.datetime_complete - t.datetime_start).total_seconds() / 60
                     for t in done if t.datetime_complete]
        if durations:
            eta = max(0, args.trials - len(done)) * sum(durations) / len(durations)
        last = ckpt_root / f"{name}_tuned" / "last_checkpoint.pt"
        if last.exists() and any(t.state == TrialState.COMPLETE for t in trials):
            epoch = torch.load(last, map_location="cpu", weights_only=True)["epoch"]
            row["retrain"] = f"ep {epoch}/{study.best_params['num_epochs']}"
    if row["tuned_json"]:
        eta = 0.0
    row["eta_h"] = round(eta / 60, 1)
    return row


def print_status(models, args):
    import pandas as pd
    frame = pd.DataFrame([status_row(m, args) for m in models]).set_index("model")
    print(frame.to_string())
    print(f"\ntuning left: ~{frame['eta_h'].sum():.1f} h (rows with no finished "
          "trial are baseline-scaled estimates; final retrains not included)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="notebook stems; default --set")
    parser.add_argument("--set", choices=("risk", "rmse", "all"), default="all")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--fraction", type=float, default=0.25,
                        help="share of the train split each trial trains on")
    parser.add_argument("--debug", action="store_true",
                        help="us-101 capped rows, outputs under experiments/tuning_debug/")
    parser.add_argument("--no-retrain", action="store_true", help="tune only")
    parser.add_argument("--status", action="store_true", help="progress table and exit")
    parser.add_argument("--list", action="store_true", help="show the queue and exit")
    args = parser.parse_args(argv)

    chosen = args.models or (SETS["risk"] + SETS["rmse"] if args.set == "all"
                             else SETS[args.set])
    models = [m for m in ORDER if m in chosen] + [m for m in chosen if m not in ORDER]
    models = list(dict.fromkeys(models))
    if args.list:
        for i, name in enumerate(models, 1):
            sets = "+".join(s for s, members in SETS.items() if name in members) or "-"
            print(f"{i:3}. {name:18} sets={sets:9} dropout={uses_dropout(name)!s:5}"
                  f" baseline={baseline_minutes(name):6.1f} min"
                  f"{'  LOOSE' if is_loose(name) else ''}")
        return 0
    if args.status:
        print_status(models, args)
        return 0

    data = load_data(models, args)
    statuses = []
    for name in models:
        started = time.time()
        try:
            study = tune(name, data, args)
            if not args.no_retrain:
                retrain(name, study, data, args)
            status = "ok"
        except Exception as error:                      # keep the batch going
            logger.exception("%s failed", name)
            status = f"FAILED: {type(error).__name__}: {error}"
        statuses.append((name, status, (time.time() - started) / 60))

    print(f"\n{'model':14} {'status':10} minutes")
    for name, status, minutes in statuses:
        print(f"{name:14} {status:10} {minutes:7.1f}")
    return 0 if all(s == "ok" for _, s, _ in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
