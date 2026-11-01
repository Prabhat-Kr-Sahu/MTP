#!/usr/bin/env python3
"""Score trained notebook models with the full risk-measure suite.

Why a script and not a notebook rerun: notebook cell 16 calls fit()
unconditionally, so re-executing the ten notebooks would retrain for ~25 h to
re-derive checkpoints that already exist, and cell 25 would overwrite the 39
committed result JSONs. This writes to experiments/results/risk/ instead and
never touches results/<name>.json.

Each notebook defines its model class and build_model() inside cell 13, so
neither can be imported. They are reachable as data, though: the cells are
addressable in the .ipynb JSON, and build_model()'s only dependency on the
expensive data cell is norm_stats, which comes from the checkpoint directory.
So this execs cells 4 (imports), 7 (config) and 13 (model) and skips cell 10.

Usage:
    python experiments/score_risk.py --top 10 --sources cv,oracle
    python experiments/score_risk.py miat --sources pred --sample-stride 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiments.benchmark import RISK_DIR, fidelities, is_loose  # noqa: E402
from src.config import (  # noqa: E402
    DEBUG_MAX_ROWS,
    DEVICE,
    NGSIM_LOCATIONS,
    NORMALIZATION_FILE,
    SEED,
)
from src.data.loader import NGSIMDataLoader  # noqa: E402
from src.data.normalization import load_stats  # noqa: E402
from src.evaluation.risk_metrics import EARLINESS_METRICS, SCALE_CONSTANTS  # noqa: E402
from src.evaluation.risk_scoring import (  # noqa: E402
    NEIGHBOUR_SOURCES,
    build_report,
    score_split,
)
from src.logger import get_logger  # noqa: E402

EXPERIMENTS = ROOT / "experiments"
COMPARISON_CSV = EXPERIMENTS / "comparison_rmse.csv"
IMPORT_CELL, CONFIG_CELL, MODEL_CELL = 4, 7, 13
logger = get_logger()


def top_models(n=10):
    """The n best models by rmse_avg, from the committed comparison table.

    Read from the CSV rather than hard-coded, so the selection stays honest if
    the benchmark is rerun.
    """
    frame = pd.read_csv(COMPARISON_CSV)
    return frame.sort_values("rmse_avg")["model"].head(n).tolist()


def load_notebook_model(name, device=DEVICE, load_weights=True, stats=None):
    """(model, namespace) by exec'ing a notebook's import, config and model cells.

    Cell 10 (the NGSIM load, minutes) is skipped: build_model() needs only
    norm_stats from it, which is read straight from the checkpoint directory.

    `stats` overrides that read, so the loader can be exercised without a
    trained checkpoint -- the checkpoints live only on the GPU box.
    """
    path = EXPERIMENTS / f"{name}.ipynb"
    if not path.exists():
        raise FileNotFoundError(path)
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    source = {i: "".join(cells[i]["source"]) for i in (IMPORT_CELL, CONFIG_CELL, MODEL_CELL)}
    if len(code) != 9:
        raise ValueError(f"{name}: expected 9 code cells, found {len(code)}")

    namespace = {"__name__": "__notebook__", "display": lambda *a, **k: None,
                 "get_ipython": lambda: None}
    for index in (IMPORT_CELL, CONFIG_CELL):
        exec(compile(source[index], f"{name}:cell{index}", "exec"), namespace)
    namespace["DEBUG"] = False
    namespace["LOCATIONS"] = NGSIM_LOCATIONS

    checkpoint_dir = Path(namespace["CKPT_DIR"])
    if stats is None:
        stats_path = checkpoint_dir / NORMALIZATION_FILE
        if not stats_path.exists():
            raise FileNotFoundError(f"{name}: {stats_path} missing (train it first)")
        stats = load_stats(str(stats_path))
    namespace["norm_stats"] = stats
    exec(compile(source[MODEL_CELL], f"{name}:cell{MODEL_CELL}", "exec"), namespace)

    model = namespace["model"]
    if load_weights:
        weights = checkpoint_dir / "best_model.pt"
        if not weights.exists():
            raise FileNotFoundError(f"{name}: {weights} missing (train it first)")
        model.load_state_dict(
            torch.load(weights, map_location=device, weights_only=True)
        )
    return model.eval().to(device), namespace


def score_one(name, loader, test_samples, args, oracle=None, oracle_report=None):
    """Score one model across the requested sources; returns the result dict."""
    model, namespace = load_notebook_model(
        name, device=args.device, load_weights=not args.random_weights,
        stats=args.stats_override,
    )
    norm_stats = namespace["norm_stats"]
    variants = {}
    for source in args.sources:
        if source == "oracle" and oracle_report is not None:
            # Identical for every model, so reuse it rather than repeating a
            # full AUC sweep over every pair and measure ten times over.
            variants[source] = oracle_report
            continue
        started = time.time()
        scored = score_split(
            model, loader, test_samples, norm_stats, threshold=args.threshold,
            source=source, batch_size=args.batch_size,
            sample_stride=1 if source in ("cv", "oracle") else args.sample_stride,
            device=args.device,
            prefixes=EARLINESS_METRICS if source == "cv" else (),
        )
        variants[source] = build_report(scored, oracle=oracle)
        variants[source]["minutes"] = round((time.time() - started) / 60, 2)
        logger.info("%s / %s: %d pairs, %d collisions, %.1f min", name, source,
                    scored.n_pairs, scored.n_collisions,
                    variants[source]["minutes"])
    return {
        "name": name,
        "fidelity": fidelities().get(name, "unknown"),
        "loose_reimplementation": is_loose(name),
        "threshold": args.threshold,
        "label_criterion": f"true min centre distance < {args.threshold} m",
        "threshold_rule": "roc_nearest_01",
        "scale_constants": SCALE_CONSTANTS,
        "seed": SEED,
        "random_weights": bool(args.random_weights),
        "debug": bool(args.debug),
        "variants": variants,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="notebook stems; default --top")
    parser.add_argument("--top", type=int, default=10,
                        help="best N by rmse_avg when no models are named")
    parser.add_argument("--sources", default="cv,oracle",
                        help=f"comma-separated, from {','.join(NEIGHBOUR_SOURCES)}")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-stride", type=int, default=1,
                        help="subsample the split for the costly scene sources")
    parser.add_argument("--out", type=Path, default=RISK_DIR)
    parser.add_argument("--debug", action="store_true",
                        help="us-101 only, capped rows -- for smoke tests")
    parser.add_argument("--random-weights", action="store_true",
                        help="skip loading checkpoints; exercises the data path "
                             "only, so the SCORES ARE MEANINGLESS")
    parser.add_argument("--force", action="store_true", help="rescore existing results")
    parser.add_argument("--list", action="store_true", help="show the selection and exit")
    args = parser.parse_args(argv)

    args.device = DEVICE
    args.sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(args.sources) - set(NEIGHBOUR_SOURCES)
    if unknown:
        parser.error(f"unknown sources {sorted(unknown)}; pick from {NEIGHBOUR_SOURCES}")

    args.stats_override = None
    if args.random_weights:
        import torch as _torch
        from src.config import STATE_DIM
        args.stats_override = {"mean": _torch.zeros(STATE_DIM),
                               "std": _torch.ones(STATE_DIM)}
        logger.warning("--random-weights: checkpoints are NOT loaded, so the "
                       "scores in this run are meaningless. Smoke test only.")

    models = args.models or top_models(args.top)
    if args.list:
        for i, name in enumerate(models, 1):
            flag = "  LOOSE" if is_loose(name) else ""
            print(f"{i:3}. {name}{flag}")
        print(f"\nsources: {args.sources}   device: {args.device}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    pending = [m for m in models
               if args.force or not (args.out / f"{m}.json").exists()]
    if not pending:
        print("nothing to do; pass --force to rescore")
        return 0

    torch.manual_seed(SEED)
    loader = NGSIMDataLoader(
        locations=["us-101"] if args.debug else NGSIM_LOCATIONS,
        max_rows=DEBUG_MAX_ROWS if args.debug else None,
    )
    _, _, test_samples = loader.get_splits()
    logger.info("test split: %d samples", len(test_samples))

    # The oracle source reads true futures on both sides, so its scores do not
    # depend on the model at all: score it once and reuse it for every model,
    # both as a variant and as the regression target for the others.
    oracle = oracle_report = None
    if "oracle" in args.sources:
        model, namespace = load_notebook_model(
            pending[0], device=args.device,
            load_weights=not args.random_weights, stats=args.stats_override,
        )
        oracle = score_split(
            model, loader, test_samples, namespace["norm_stats"],
            threshold=args.threshold, source="oracle",
            batch_size=args.batch_size, device=args.device,
        )
        oracle_report = build_report(oracle, oracle=oracle)
        (args.out / "_oracle.json").write_text(
            json.dumps(oracle_report, indent=2), encoding="utf-8"
        )
        logger.info("oracle: %d pairs, %d collisions", oracle.n_pairs,
                    oracle.n_collisions)

    statuses = []
    for name in pending:
        started = time.time()
        try:
            result = score_one(name, loader, test_samples, args, oracle=oracle,
                               oracle_report=oracle_report)
            (args.out / f"{name}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            status = "ok"
        except Exception as error:                      # keep the batch going
            logger.exception("%s failed", name)
            status = f"FAILED: {type(error).__name__}: {error}"
        statuses.append((name, status, round((time.time() - started) / 60, 1)))

    print(f"\n{'model':22} {'status':14} minutes")
    for name, status, minutes in statuses:
        print(f"{name:22} {status:14} {minutes:7.1f}")
    print(f"total {sum(m for _, _, m in statuses):.1f} min")
    return 0 if all(s == "ok" for _, s, _ in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
