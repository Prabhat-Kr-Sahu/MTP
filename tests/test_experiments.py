from experiments.benchmark import compare, save_result

HORIZONS = ["rmse_1s", "rmse_2s", "rmse_3s", "rmse_4s", "rmse_5s"]


def _result(name, avg, collisions=None):
    test = {h: avg for h in HORIZONS} | {"rmse_avg": avg, "n": 10}
    return {"name": name, "description": "", "training_log": [{"epoch": 1, "val_rmse": avg, "val_loss": 1.0}],
            "test": {"NGSIM Test Split": test, "site us-101": test}, "collisions": collisions}


def test_compare_ranks_every_saved_result_and_skips_missing_collisions(tmp_path):
    results_dir = tmp_path / "results"
    row = {"metric": "min_dist", "f1": 0.5, "precision": 0.5, "recall": 0.5, "pr_auc": 0.4, "roc_auc": 0.6}
    save_result(_result("worse", 3.0, {"threshold": 2.0, "rows": [row]}), results_dir)
    save_result(_result("better", 1.0), results_dir)

    rmse, coll = compare(results_dir, plot=False)

    assert list(rmse.index) == ["better", "worse"]
    assert rmse.loc["better", "site us-101 avg"] == 1.0
    assert list(coll.index) == [("min_dist", "worse")]
    assert (tmp_path / "comparison_rmse.csv").exists()
