"""Save one result JSON per experiment notebook and compare every saved result.

Each notebook in experiments/ calls save_result() once and compare() at the end,
so the comparison section always reflects every model run so far.
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
HORIZONS = ["rmse_1s", "rmse_2s", "rmse_3s", "rmse_4s", "rmse_5s", "rmse_avg"]
OVERALL = "NGSIM Test Split"


def save_result(result: dict, results_dir: Path = RESULTS_DIR) -> Path:
    """Write results/<name>.json; rerunning a notebook overwrites its own entry."""
    results_dir.mkdir(parents=True, exist_ok=True)
    result = {"timestamp": datetime.now().isoformat(timespec="seconds"), **result}
    path = results_dir / f"{result['name']}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"Saved {path}")
    return path


def load_results(results_dir: Path = RESULTS_DIR) -> list:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def rmse_table(results: list) -> pd.DataFrame:
    """Rows = models sorted by overall rmse_avg; cols = horizons + per-group avg."""
    rows = []
    for r in results:
        data = r.get("data", {})
        row = {"model": r["name"], "description": r.get("description", ""),
               # ponytail: runs on different data are listed side by side, not filtered; check this column
               "data": ("debug " if data.get("debug") else "") + ",".join(data.get("locations", [])),
               "epochs": len(r.get("training_log", [])), "n_params": r.get("n_params")}
        row.update({h: r["test"][OVERALL][h] for h in HORIZONS})
        row.update({f"{g} avg": m["rmse_avg"] for g, m in r["test"].items() if g != OVERALL})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").sort_values("rmse_avg")


def collision_table(results: list) -> pd.DataFrame:
    """Long table (model, metric) -> F1/precision/recall/AUCs; models without collisions skipped."""
    rows = [
        {"model": r["name"], **{k: row[k] for k in
         ("metric", "f1", "precision", "recall", "pr_auc", "roc_auc")}}
        for r in results if r.get("collisions") for row in r["collisions"]["rows"]
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["metric", "model"]).sort_index()


def compare(results_dir: Path = RESULTS_DIR, plot: bool = True):
    """Benchmark every saved result. Returns (rmse_df, collision_df); writes comparison CSVs."""
    results = load_results(results_dir)
    if not results:
        print(f"No results in {results_dir} yet.")
        return pd.DataFrame(), pd.DataFrame()
    rmse = rmse_table(results)
    coll = collision_table(results)
    rmse.to_csv(results_dir.parent / "comparison_rmse.csv")
    if not coll.empty:
        coll.to_csv(results_dir.parent / "comparison_collisions.csv")

    if plot:
        from IPython.display import display
        display(rmse.style.highlight_min(subset=HORIZONS, color="lightgreen").format(precision=4, na_rep="-"))

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        for r in results:
            axes[0].plot(range(1, 6), [r["test"][OVERALL][h] for h in HORIZONS[:5]],
                         marker="o", label=r["name"])
            log = pd.DataFrame(r.get("training_log", []))
            if not log.empty:
                axes[1].plot(log["epoch"], log["val_rmse"], marker=".", label=r["name"])
                axes[2].plot(log["epoch"], log["val_loss"], marker=".", label=r["name"])
        for ax, title, xlabel in zip(axes, ["Test RMSE vs horizon", "Val RMSE", "Val loss"],
                                     ["horizon (s)", "epoch", "epoch"]):
            ax.set(title=title, xlabel=xlabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("RMSE (m)")
        plt.tight_layout()
        plt.show()

        if not coll.empty:
            display(coll.unstack("model").style.format(precision=4, na_rep="-"))
    return rmse, coll
