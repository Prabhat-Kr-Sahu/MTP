#!/usr/bin/env python3
"""Baseline vs tuned: RMSE, risk AUCs with event-clustered CIs, ROC curves.

Reads results/<m>.json and results/<m>_tuned.json (RMSE), and the risk_v2
scoring pass: results/risk_v2/<name>.json (AUCs, ROC points) plus the
<name>.<source>.npz sidecars (per-positive AUC placements and event keys).

Every CI resamples whole vehicle-pair events (~100 at 2 m), not pairs, and is
paired: baseline and tuned share the pair set within a source, so one event
draw serves both and their difference. Writes experiments/comparison_tuned_*.csv
and experiments/figures/*.png, and prints the tables RISK_TUNING_RESULTS.md quotes.

    python experiments/compare_tuned.py
"""
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.benchmark import HORIZONS, OVERALL, RESULTS_DIR, is_loose  # noqa: E402
from experiments.tune_top import ORDER  # noqa: E402
from src.evaluation.collision import event_bootstrap  # noqa: E402
from src.evaluation.risk_metrics import PROBABILITY_METRICS  # noqa: E402
from src.evaluation.risk_scoring import HEADLINE_METRICS  # noqa: E402

EXPERIMENTS = ROOT / "experiments"
SOURCES = ("cv", "cv_scene", "pred")           # oracle is model-free: own rows
LABELS = ("2", "3")
VERSIONS = ("baseline", "tuned")
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def tuned(name):
    return f"{name}_tuned"


def entry(model, version):
    return model if version == "baseline" else tuned(model)


@lru_cache(maxsize=None)
def risk_json(risk_dir, name):
    path = Path(risk_dir) / ("_oracle.json" if name == "_oracle" else f"{name}.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@lru_cache(maxsize=None)
def sidecar(risk_dir, name, source):
    path = Path(risk_dir) / ("_oracle.npz" if source == "oracle" else f"{name}.{source}.npz")
    return dict(np.load(path)) if path.exists() else None


def variant(risk_dir, name, source):
    if source == "oracle":
        return risk_json(risk_dir, "_oracle")
    payload = risk_json(risk_dir, name)
    return payload and payload["variants"].get(source)


def aucs(payload, label, metric):
    """(roc, pr) for one metric at one label threshold from a variant report."""
    if label == "2":
        row = next(r for r in payload["rows"] if r["metric"] == metric)
    else:
        row = payload["by_label_threshold"][label]["metrics"][metric]
    return row["roc_auc"], row["pr_auc"]


def placements(risk_dir, name, source, label, metric):
    """(pair index, event keys, placements) or None if not stored."""
    z = sidecar(risk_dir, name, source)
    key = f"placement_{label}__{metric}"
    if z is None or key not in z:
        return None
    return z[f"index_{label}"], z[f"events_{label}"], z[key].astype(float)


def check(name, source, label, metric, values, roc):
    """The placements must reproduce the stored AUC, or the CI is of something else."""
    if np.isfinite(roc) and abs(values.mean() - roc) > 1e-4:
        raise AssertionError(f"{name}/{source}/{label} m/{metric}: placement mean "
                             f"{values.mean():.6f} != stored roc_auc {roc:.6f}")


# --------------------------------------------------------------------------- #
#  Tables
# --------------------------------------------------------------------------- #
def rmse_table(models):
    rows = []
    for m in models:
        paths = [RESULTS_DIR / f"{entry(m, v)}.json" for v in VERSIONS]
        if not all(p.exists() for p in paths):
            continue
        base, tune = (json.loads(p.read_text()) for p in paths)
        row = {"model": m, "loose": is_loose(m)}
        for h in HORIZONS:
            b, t = base["test"][OVERALL][h], tune["test"][OVERALL][h]
            row.update({f"{h}_baseline": b, f"{h}_tuned": t, f"{h}_delta": t - b})
        row.update({
            "val_rmse_baseline": min(e["val_rmse"] for e in base["training_log"]),
            "val_rmse_tuned": min(e["val_rmse"] for e in tune["training_log"]),
            "epochs_tuned": tune["hparams"]["num_epochs"],
            "best_epoch_tuned": min(tune["training_log"], key=lambda e: e["val_loss"])["epoch"],
            "train_minutes_tuned": tune["train_minutes"],
            **{f"hp_{k}": v for k, v in tune["hparams"].items() if k != "num_epochs"},
            "hp_dropout": tune.get("overrides", {}).get("DROPOUT"),
            "tuning_trials": tune["tuning"]["n_trials"],
            "tuning_pruned": tune["tuning"]["states"]["PRUNED"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def risk_table(models, risk_dir, n_boot):
    """One row per (model, source, label, metric): both AUCs, three paired CIs."""
    rows = []
    for m in models:
        for source in SOURCES:
            pay_b, pay_t = variant(risk_dir, m, source), variant(risk_dir, tuned(m), source)
            if pay_b is None or pay_t is None:
                continue
            for label in LABELS:
                for metric in HEADLINE_METRICS if label != "2" else [
                        r["metric"] for r in pay_b["rows"]]:
                    pb = placements(risk_dir, m, source, label, metric)
                    pt = placements(risk_dir, tuned(m), source, label, metric)
                    (roc_b, pr_b), (roc_t, pr_t) = (aucs(pay_b, label, metric),
                                                    aucs(pay_t, label, metric))
                    row = {"model": m, "loose": is_loose(m), "source": source,
                           "label_m": label, "metric": metric,
                           "roc_baseline": roc_b, "roc_tuned": roc_t,
                           "roc_delta": roc_t - roc_b,
                           "pr_baseline": pr_b, "pr_tuned": pr_t, "pr_delta": pr_t - pr_b}
                    if pb is not None and pt is not None:
                        if not np.array_equal(pb[0], pt[0]):
                            raise AssertionError(f"{m}/{source}: pair sets differ")
                        check(m, source, label, metric, pb[2], roc_b)
                        check(tuned(m), source, label, metric, pt[2], roc_t)
                        ci = event_bootstrap(np.stack([pb[2], pt[2], pt[2] - pb[2]]),
                                             pb[1], n_boot=n_boot)
                        row.update({
                            "n_positives": len(pb[0]), "n_events": len(np.unique(pb[1])),
                            "ci_baseline_lo": ci[0, 0], "ci_baseline_hi": ci[0, 1],
                            "ci_tuned_lo": ci[1, 0], "ci_tuned_hi": ci[1, 1],
                            "ci_delta_lo": ci[2, 0], "ci_delta_hi": ci[2, 1],
                            "delta_significant": bool(ci[2, 0] > 0 or ci[2, 1] < 0),
                        })
                    rows.append(row)
    return pd.DataFrame(rows)


def oracle_table(risk_dir, n_boot):
    payload = risk_json(risk_dir, "_oracle")
    if payload is None:
        return pd.DataFrame()
    rows = []
    for label in LABELS:
        for metric in HEADLINE_METRICS:
            roc, pr = aucs(payload, label, metric)
            row = {"label_m": label, "metric": metric, "roc": roc, "pr": pr}
            p = placements(risk_dir, None, "oracle", label, metric)
            if p is not None:
                check("_oracle", "oracle", label, metric, p[2], roc)
                (lo, hi), = event_bootstrap(p[2], p[1], n_boot=n_boot)
                row.update(ci_lo=lo, ci_hi=hi)
            rows.append(row)
    return pd.DataFrame(rows)


def headline_mean(risk_dir, name, source, label):
    """Per-positive mean placement over the headline six: its mean is their mean AUC."""
    parts = [placements(risk_dir, name, source, label, m) for m in HEADLINE_METRICS]
    if any(p is None for p in parts):
        return None
    return parts[0][0], parts[0][1], np.mean([p[2] for p in parts], axis=0)


def ranking_table(models, risk_dir, rmse, n_boot):
    """All 16 entries under pred, ranked by mean headline ROC-AUC with CIs."""
    rows = []
    for label in LABELS:
        for m in models:
            for version in VERSIONS:
                name = entry(m, version)
                hm = headline_mean(risk_dir, name, "pred", label)
                if hm is None:
                    continue
                (lo, hi), = event_bootstrap(hm[2], hm[1], n_boot=n_boot)
                rm = rmse.set_index("model") if len(rmse) else None
                rows.append({
                    "label_m": label, "entry": name, "model": m, "version": version,
                    "loose": is_loose(m), "headline_roc": hm[2].mean(),
                    "ci_lo": lo, "ci_hi": hi,
                    "rmse_avg": rm.loc[m, f"rmse_avg_{version}"]
                    if rm is not None and m in rm.index else np.nan,
                })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["rank"] = frame.groupby("label_m")["headline_roc"].rank(ascending=False)
    return frame


def spearman(ranking):
    """Spearman(risk rank, RMSE rank) within each version, per label."""
    out = {}
    for (label, version), sub in ranking.groupby(["label_m", "version"]):
        if sub["rmse_avg"].notna().sum() > 2:
            out[f"{version} @ {label} m"] = round(
                sub["headline_roc"].corr(-sub["rmse_avg"], method="spearman"), 3)
    return out


def pred_gain_table(models, risk_dir, n_boot):
    """pred - cv_scene per model and version, paired on the shared scene pair set."""
    rows = []
    for label in LABELS:
        for m in models:
            for version in VERSIONS:
                name = entry(m, version)
                p, c = (headline_mean(risk_dir, name, s, label) for s in ("pred", "cv_scene"))
                if p is None or c is None:
                    continue
                if not np.array_equal(p[0], c[0]):
                    raise AssertionError(f"{name}: pred and cv_scene pair sets differ")
                ci = event_bootstrap(np.stack([p[2] - c[2]]), p[1], n_boot=n_boot)
                rows.append({"label_m": label, "model": m, "version": version,
                             "pred": p[2].mean(), "cv_scene": c[2].mean(),
                             "gain": (p[2] - c[2]).mean(),
                             "ci_lo": ci[0, 0], "ci_hi": ci[0, 1]})
    return pd.DataFrame(rows)


def stratified_table(models, risk_dir):
    rows = []
    for m in models:
        for version in VERSIONS:
            payload = variant(risk_dir, entry(m, version), "pred")
            if payload is None:
                continue
            for bucket, block in payload["stratified"].items():
                for row in block["rows"]:
                    if row["metric"] in HEADLINE_METRICS:
                        rows.append({"model": m, "version": version, "bucket": bucket,
                                     "metric": row["metric"], "roc": row["roc_auc"],
                                     "n_positives": block["n_pairs"]})
    return pd.DataFrame(rows)


def calibration_table(models, risk_dir):
    rows = []
    for m in models:
        for version in VERSIONS:
            for source in ("cv", "pred"):
                payload = variant(risk_dir, entry(m, version), source)
                for metric, c in (payload or {}).get("calibration", {}).items():
                    if metric in PROBABILITY_METRICS:
                        rows.append({"model": m, "version": version, "source": source,
                                     "metric": metric, "brier": c["brier"],
                                     "bss": 1 - c["brier"] / c["uncertainty"]
                                     if c["uncertainty"] else np.nan,
                                     "reliability": c["reliability"],
                                     "resolution": c["resolution"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def _axes_style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def roc_figures(models, risk_dir, out_dir, label_m="2"):
    """One figure per (source, headline measure) at one label: a panel per model,
    baseline dashed blue, tuned solid orange, FPR on a log axis because the
    operating region at these base rates is FPR << 1 %."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for source in SOURCES:
        for metric in HEADLINE_METRICS:
            present = [m for m in models
                       if variant(risk_dir, m, source) and variant(risk_dir, tuned(m), source)]
            if not present:
                continue
            cols = 4
            rows = -(-len(present) // cols)
            fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.9 * rows),
                                     sharex=True, sharey=True, squeeze=False)
            for ax, m in zip(axes.flat, present):
                _axes_style(ax)
                label = ""
                for version, color, style in (("baseline", BLUE, "--"), ("tuned", ORANGE, "-")):
                    payload = variant(risk_dir, entry(m, version), source)
                    curve = payload.get("roc_curves", {}).get(label_m, {}).get(metric)
                    if not isinstance(curve, dict):
                        continue
                    fpr = np.maximum(curve["fpr"], 1e-5)
                    ax.step(fpr, curve["tpr"], where="post", color=color, linestyle=style,
                            linewidth=2, label=version)
                    auc = aucs(payload, label_m, metric)[0]
                    label += f"{auc:.3f} " if label == "" else f"-> {auc:.3f}"
                ax.plot([1e-5, 1], [1e-5, 1], color=MUTED, linewidth=0.8, linestyle=":")
                ax.set_xscale("log")
                ax.set_xlim(1e-5, 1)
                ax.set_ylim(0, 1.02)
                flag = " (LOOSE)" if is_loose(m) else ""
                ax.set_title(f"{m}{flag}\nAUC {label}", fontsize=9, color=INK, loc="left")
            for ax in axes.flat[len(present):]:
                ax.axis("off")
            for ax in axes[-1]:
                ax.set_xlabel("false-positive rate (log)", fontsize=8, color=MUTED)
            for ax in axes[:, 0]:
                ax.set_ylabel("true-positive rate", fontsize=8, color=MUTED)
            if axes.flat[0].get_legend_handles_labels()[0]:
                axes.flat[0].legend(fontsize=8, frameon=False, loc="lower right")
            fig.suptitle(f"ROC, {metric}, {source} neighbours, label < {label_m} m: "
                         "baseline vs tuned", fontsize=11, color=INK, x=0.01, ha="left")
            fig.tight_layout()
            path = out_dir / f"roc_{source}_{metric}_{label_m}m.png"
            fig.savefig(path, dpi=130, facecolor="#fcfcfb")
            plt.close(fig)
            written.append(path)
    return written


def forest_figure(risk, out_dir, source="pred", label="2"):
    """Paired ROC-AUC change (tuned - baseline) with its event-bootstrap CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = risk[(risk.source == source) & (risk.label_m == label)
               & risk.metric.isin(HEADLINE_METRICS)]
    if sub.empty or "ci_delta_lo" not in sub:
        return None
    models = list(dict.fromkeys(sub.model))
    fig, axes = plt.subplots(2, 3, figsize=(12, 1.0 + 0.42 * len(models) * 2), sharex=True)
    for ax, metric in zip(axes.flat, HEADLINE_METRICS):
        _axes_style(ax)
        rows = sub[sub.metric == metric].set_index("model").reindex(models)
        y = np.arange(len(models))[::-1]
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.hlines(y, rows.ci_delta_lo, rows.ci_delta_hi, color=ORANGE, linewidth=2)
        ax.plot(rows.roc_delta, y, "o", color=ORANGE, markersize=8,
                markeredgecolor="#fcfcfb", markeredgewidth=2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{m}{'*' if is_loose(m) else ''}" for m in models], fontsize=8)
        ax.set_title(metric, fontsize=10, color=INK, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("ROC-AUC change, tuned - baseline", fontsize=8, color=MUTED)
    fig.suptitle(f"Tuning effect on risk ROC-AUC ({source}, label < {label} m), "
                 "95 % event-bootstrap CI   (* LOOSE)", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    path = out_dir / f"forest_{source}_{label}m.png"
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def fmt(frame):
    return frame.to_string(float_format=lambda v: f"{v:.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--risk-dir", type=Path, default=RESULTS_DIR / "risk_v2")
    parser.add_argument("--out", type=Path, default=EXPERIMENTS)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    risk_dir = str(args.risk_dir)
    models = ORDER

    rmse = rmse_table(models)
    risk = risk_table(models, risk_dir, args.n_boot)
    oracle = oracle_table(risk_dir, args.n_boot)
    ranking = ranking_table(models, risk_dir, rmse, args.n_boot)
    gain = pred_gain_table(models, risk_dir, args.n_boot)
    strat = stratified_table(models, risk_dir)
    cal = calibration_table(models, risk_dir)

    args.out.mkdir(parents=True, exist_ok=True)
    for stem, frame in (("rmse", rmse), ("risk", risk), ("oracle", oracle),
                        ("ranking", ranking), ("pred_gain", gain),
                        ("stratified", strat), ("calibration", cal)):
        frame.to_csv(args.out / f"comparison_tuned_{stem}.csv", index=False)

    pd.set_option("display.width", 200)
    if len(rmse):
        print("\n## RMSE (test, overall), baseline vs tuned")
        print(fmt(rmse[["model", "loose", "rmse_avg_baseline", "rmse_avg_tuned",
                        "rmse_avg_delta", "rmse_5s_baseline", "rmse_5s_tuned",
                        "epochs_tuned", "best_epoch_tuned"]]))
    if len(risk):
        head = risk[risk.metric.isin(HEADLINE_METRICS)]
        cols = ["model", "metric", "roc_baseline", "roc_tuned", "roc_delta"] + (
            ["ci_delta_lo", "ci_delta_hi", "delta_significant"] if "ci_delta_lo" in risk else [])
        for source in SOURCES:
            for label in LABELS:
                sub = head[(head.source == source) & (head.label_m == label)]
                if len(sub):
                    print(f"\n## {source}, label < {label} m: ROC-AUC baseline -> tuned")
                    print(fmt(sub[cols]))
        print("\n## Mean ROC-AUC change over models, all measures, label < 2 m")
        two = risk[risk.label_m == "2"]
        print(fmt(two.pivot_table(index="metric", columns="source", values="roc_delta",
                                  aggfunc="mean")))
    if len(oracle):
        print("\n## oracle (model-free ceiling)")
        print(fmt(oracle))
    if len(ranking):
        for label in LABELS:
            sub = ranking[ranking.label_m == label].sort_values("rank")
            if len(sub):
                print(f"\n## pred ranking, label < {label} m, mean headline ROC-AUC")
                print(fmt(sub[["rank", "entry", "loose", "headline_roc", "ci_lo", "ci_hi",
                               "rmse_avg"]]))
        print("\nSpearman(risk, -RMSE):", spearman(ranking))
    if len(gain):
        print("\n## pred - cv_scene (headline mean), baseline vs tuned")
        print(fmt(gain))
    if len(strat):
        print("\n## pred, horizon-stratified ROC-AUC, mean over models")
        print(fmt(strat.pivot_table(index=["metric", "version"], columns="bucket",
                                    values="roc", aggfunc="mean")))
    if len(cal):
        print("\n## Brier skill score, mean over models")
        print(fmt(cal.pivot_table(index=["metric", "source"], columns="version",
                                  values="bss", aggfunc="mean")))

    if not args.no_figures:
        figures = args.out / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        written = []
        for label in LABELS:
            written += roc_figures(models, risk_dir, figures, label)
            path = forest_figure(risk, figures, label=label) if len(risk) else None
            written += [path] if path else []
        print(f"\nwrote {len(written)} figures to {figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
