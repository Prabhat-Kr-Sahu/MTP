"""Print the tables behind RISK_PREDICTION_RESULTS.md section 5.

Reads only the committed experiments/results/risk/*.json and
experiments/comparison_rmse.csv, so the section can be regenerated anywhere.

    python experiments/summarise_risk.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.benchmark import RISK_DIR, is_loose, load_risk_results  # noqa: E402

VARIANTS = ("oracle", "cv", "cv_scene", "pred")
HEADLINE = ("p_coll_gauss", "p_coll_mc", "s_field_cal", "min_dist", "ttc_true", "tet")


def long_table(results):
    """(model, variant, metric, label) -> roc_auc, pr_auc, base_rate."""
    rows = []
    for r in results:
        for variant, payload in r["variants"].items():
            for row in payload["rows"]:
                rows.append({"model": r["name"], "variant": variant, "metric": row["metric"],
                             "label": "2", "roc": row["roc_auc"], "pr": row["pr_auc"],
                             "base": payload["base_rate"]})
            for label, block in payload.get("by_label_threshold", {}).items():
                if label == "2":
                    continue
                for metric, m in block["metrics"].items():
                    rows.append({"model": r["name"], "variant": variant, "metric": metric,
                                 "label": label, "roc": m["roc_auc"], "pr": m["pr_auc"],
                                 "base": block["base_rate"]})
    return pd.DataFrame(rows)


def fmt(frame):
    return frame.to_string(float_format=lambda v: f"{v:.3f}")


def main():
    results = load_risk_results(RISK_DIR)
    t = long_table(results)
    rmse = pd.read_csv(ROOT / "experiments/comparison_rmse.csv").set_index("model")["rmse_avg"]
    models = [r["name"] for r in sorted(results, key=lambda r: rmse[r["name"]])]

    for label in ("2", "3"):
        sub = t[t.label == label]
        print(f"\n## Measures, mean ROC-AUC over the 10 models, label < {label} m "
              f"(oracle is model-free)")
        table = sub.pivot_table(index="metric", columns="variant", values="roc", aggfunc="mean")
        table = table[list(VARIANTS)].sort_values("pred", ascending=False)
        spread = sub[sub.variant == "pred"].groupby("metric")["roc"].agg(lambda v: v.max() - v.min())
        table["pred_spread"] = spread
        table["pred-cv_scene"] = table["pred"] - table["cv_scene"]
        print(fmt(table))
        print("base rates:", sub.groupby("variant")["base"].first().round(5).to_dict())

    for label in ("2", "3"):
        sub = t[(t.label == label) & t.metric.isin(HEADLINE)]
        for variant in ("pred", "cv"):
            print(f"\n## Per model ({variant}), label < {label} m: ROC-AUC")
            table = sub[sub.variant == variant].pivot(index="model", columns="metric", values="roc")
            table = table.loc[models, list(HEADLINE)]
            table.insert(0, "rmse_avg", rmse[models])
            table.insert(1, "loose", [is_loose(m) for m in models])
            print(fmt(table))
            ranks = table[list(HEADLINE)].rank(ascending=False)
            rho = {m: ranks[m].corr(table["rmse_avg"].rank(), method="spearman")
                   for m in HEADLINE}
            print("Spearman(risk rank, RMSE rank):", {k: round(v, 2) for k, v in rho.items()})
            print("PR-AUC:")
            pr = sub[sub.variant == variant].pivot(index="model", columns="metric", values="pr")
            print(fmt(pr.loc[models, list(HEADLINE)]))

    print("\n## Calibration (pred variant), mean over models")
    cal = []
    for r in results:
        for metric, c in r["variants"]["pred"]["calibration"].items():
            cal.append({"model": r["name"], "metric": metric, **{k: c[k] for k in (
                "brier", "reliability", "resolution", "uncertainty", "log_loss")}})
    cal = pd.DataFrame(cal)
    print(fmt(cal.groupby("metric").mean(numeric_only=True)))
    gauss = next(r for r in results if r["name"] == models[0])["variants"]["pred"]
    bins = gauss["calibration"]["p_coll_gauss"]["reliability_bins"]
    print(f"\nreliability bins, p_coll_gauss, {models[0]} (pred):")
    for lo, n, pm, obs in zip(bins["bin_edges"], bins["bin_count"],
                              bins["mean_predicted"], bins["observed_frequency"]):
        print(f"  [{lo:.1f}, {lo + 0.1:.1f})  n={n:>7}  predicted={pm:.3f}  observed={obs:.4f}")

    print("\n## Earliness (cv), mean over models")
    early = []
    for r in results:
        for metric, e in r["variants"]["cv"]["earliness"].items():
            early.append({"metric": metric, "detected": e["detected_fraction"],
                          "mean_lead_s": e["mean_lead_time"],
                          **{f"recall@{k}": v for k, v in e["recall_by_prefix"].items()}})
    print(fmt(pd.DataFrame(early).groupby("metric").mean()))

    print("\n## Horizon-stratified ROC-AUC (pred), mean / spread over models")
    rows = []
    for r in results:
        for bucket, block in r["variants"]["pred"]["stratified"].items():
            for row in block["rows"]:
                if row["metric"] in HEADLINE:
                    rows.append({"bucket": bucket, "metric": row["metric"],
                                 "model": r["name"], "roc": row["roc_auc"],
                                 "n": block["n_pairs"]})
    s = pd.DataFrame(rows)
    print("pairs per bucket:", s.groupby("bucket")["n"].first().to_dict())
    print(fmt(s.pivot_table(index="metric", columns="bucket", values="roc", aggfunc="mean")))
    print("spread (max-min over models):")
    print(fmt(s.pivot_table(index="metric", columns="bucket", values="roc",
                            aggfunc=lambda v: np.nanmax(v) - np.nanmin(v))))


if __name__ == "__main__":
    main()
