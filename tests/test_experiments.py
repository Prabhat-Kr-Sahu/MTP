import json

from experiments.benchmark import (
    compare,
    compare_risk,
    fidelities,
    is_loose,
    risk_stratified_table,
    risk_table,
    save_result,
)

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


# --------------------------------------------------------------------------- #
#  Fidelity flag and the risk comparison tables
# --------------------------------------------------------------------------- #
def test_fidelities_parse_the_extended_set_table():
    parsed = fidelities()
    assert parsed, "expected rows from MODEL_ALTERNATIVES.md"
    assert is_loose("inatran") and is_loose("dacr_amtp")
    assert not is_loose("miat") and not is_loose("mha_lstm")
    assert not is_loose("not_a_model_at_all")


def _risk_result(name, pr_auc, loose=False):
    row = {
        "metric": "min_dist", "higher_is_riskier": False, "threshold": 2.0,
        "default_threshold": 2.0, "threshold_rule": "roc_nearest_01",
        "non_finite_fraction": 0.0, "precision": 0.4, "recall": 0.6,
        "f1": 0.48, "false_alarm_rate": 0.02,
        "recall_at_precision_50": 0.3, "pr_auc": pr_auc, "roc_auc": 0.8,
    }
    return {
        "name": name, "loose_reimplementation": loose,
        "variants": {
            "cv": {"rows": [row], "n_pairs": 100,
                   "stratified": {"collision in 1s": {"n_pairs": 7, "rows": [row]}}},
        },
    }


def test_risk_table_is_indexed_by_metric_variant_model(tmp_path):
    risk_dir = tmp_path / "results" / "risk"
    risk_dir.mkdir(parents=True)
    (risk_dir / "aaa.json").write_text(json.dumps(_risk_result("aaa", 0.3)))
    (risk_dir / "bbb.json").write_text(json.dumps(_risk_result("bbb", 0.7, loose=True)))
    # A leading underscore marks the shared oracle run, which is not a model.
    (risk_dir / "_oracle.json").write_text(json.dumps(_risk_result("_oracle", 0.9)))

    table = risk_table(risk_dir)
    assert list(table.index) == [("min_dist", "cv", "aaa"), ("min_dist", "cv", "bbb")]
    assert table.loc[("min_dist", "cv", "bbb"), "loose_reimplementation"]
    assert not table.loc[("min_dist", "cv", "aaa"), "loose_reimplementation"]
    assert table.loc[("min_dist", "cv", "bbb"), "pr_auc"] == 0.7


def test_risk_stratified_table_carries_buckets(tmp_path):
    risk_dir = tmp_path / "results" / "risk"
    risk_dir.mkdir(parents=True)
    (risk_dir / "aaa.json").write_text(json.dumps(_risk_result("aaa", 0.3)))
    table = risk_stratified_table(risk_dir)
    assert list(table.index) == [("collision in 1s", "min_dist", "cv", "aaa")]
    assert table.iloc[0]["n_pairs"] == 7


def test_compare_risk_writes_committable_csv_names(tmp_path):
    risk_dir = tmp_path / "results" / "risk"
    risk_dir.mkdir(parents=True)
    (risk_dir / "aaa.json").write_text(json.dumps(_risk_result("aaa", 0.3)))
    flat, stratified = compare_risk(risk_dir)
    assert not flat.empty and not stratified.empty
    # Must be comparison_*.csv to survive the *.csv rule in .gitignore.
    assert (tmp_path / "comparison_risk.csv").exists()
    assert (tmp_path / "comparison_risk_stratified.csv").exists()


def test_risk_tables_are_empty_without_results(tmp_path):
    assert risk_table(tmp_path / "nope").empty
    assert risk_stratified_table(tmp_path / "nope").empty
