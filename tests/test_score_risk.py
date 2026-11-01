"""The notebook-model loader and the risk report builder."""
import numpy as np
import pandas as pd
import pytest
import torch

from experiments.benchmark import is_loose
from experiments.score_risk import load_notebook_model, top_models
from src.config import STATE_DIM, TF_STEPS, TH_STEPS
from src.evaluation.risk_metrics import METRICS, PROBABILITY_METRICS
from src.evaluation.risk_scoring import build_report, metric_report

STATS = {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)}


def test_top_models_reads_the_committed_comparison_table():
    models = top_models(10)
    assert len(models) == 10
    frame = pd.read_csv("experiments/comparison_rmse.csv").sort_values("rmse_avg")
    assert models == frame["model"].head(10).tolist()
    assert models[0] == frame["model"].iloc[0]


@pytest.mark.parametrize("name", top_models(10))
def test_every_top_notebook_model_loads_and_honours_the_forward_contract(name):
    """exec of cells 4/7/13 must reconstruct the exact trained architecture.

    Checkpoints live only on the GPU box, so weights are skipped and stats are
    injected; the parameter count is compared against the committed benchmark,
    which is what proves the reconstruction is the same model that was trained.
    """
    model, namespace = load_notebook_model(
        name, device=torch.device("cpu"), load_weights=False, stats=STATS
    )
    assert namespace["EXP_NAME"] == name

    batch, neighbours = 2, 4
    states = torch.randn(batch, TH_STEPS, neighbours, STATE_DIM)
    mask = torch.ones(batch, neighbours, dtype=torch.bool)
    dims = torch.tensor([4.5, 1.8]).expand(batch, neighbours, 2).clone()
    with torch.no_grad():
        traj_dist, goals, _ = model(states, mask, dims)

    assert traj_dist.shape == (batch, TF_STEPS, 5)
    assert goals is None or goals.shape == (batch, neighbours - 1, 4)
    assert (traj_dist[..., 2:4] > 0).all(), "sigma must be positive"
    assert (traj_dist[..., 4].abs() < 1.0).all(), "rho must lie in (-1, 1)"

    expected = pd.read_csv("experiments/comparison_rmse.csv").set_index("model")
    assert sum(p.numel() for p in model.parameters()) == int(
        expected.loc[name, "n_params"]
    )


def test_missing_notebook_is_reported_clearly():
    with pytest.raises(FileNotFoundError):
        load_notebook_model("definitely_not_a_notebook", stats=STATS)


def test_loose_flag_matches_the_documented_fidelity():
    assert is_loose("dacr_amtp") and is_loose("inatran")
    assert not is_loose("miat") and not is_loose("traj_mamba")


class _Result:
    """Minimal stand-in for RiskScores, so the report builder is tested alone."""

    def __init__(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        self.gt = (rng.uniform(size=n) < 0.2).astype(np.int8)
        self.gt_time = rng.uniform(0.0, 5.0, n).astype(np.float32)
        self.gt_distance = rng.uniform(0.0, 20.0, n).astype(np.float32)
        self.pairs = np.stack([np.arange(n), np.arange(n) + 1], axis=-1)
        # Risky pairs score lower on distance-like measures, higher on the rest.
        self.scores = {}
        for name, (_, higher, _) in METRICS.items():
            base = rng.uniform(size=n)
            signal = self.gt * (0.5 if higher else -0.5)
            value = np.clip(base + signal, 0.0, 1.0)
            self.scores[name] = value.astype(np.float32)
        self.prefix_scores = {
            "min_dist": np.tile(self.scores["min_dist"][:, None], (1, 5))
        }
        self.n_samples, self.n_filled = n, 0
        self.source, self.threshold = "cv", 2.0

    @property
    def n_pairs(self):
        return int(self.gt.size)

    @property
    def n_collisions(self):
        return int(self.gt.sum())

    def summary(self):
        return {"source": self.source, "threshold": self.threshold,
                "n_samples": self.n_samples, "n_pairs": self.n_pairs,
                "n_gt_collisions": self.n_collisions,
                "base_rate": float(self.gt.mean()),
                "n_future_steps_filled": self.n_filled}


def test_build_report_covers_every_metric_and_section():
    result = _Result()
    report = build_report(result, oracle=result)
    assert {r["metric"] for r in report["rows"]} == set(METRICS)
    assert set(report["sanity"]) == set(METRICS)
    assert set(report["regression_vs_oracle"]) == set(METRICS)
    assert set(report["calibration"]) <= PROBABILITY_METRICS
    assert "min_dist" in report["earliness"]
    assert report["n_pairs"] == result.n_pairs


def test_build_report_is_json_serialisable():
    import json
    json.dumps(build_report(_Result(), oracle=_Result()))


def test_report_rows_carry_the_shared_threshold_rule():
    for row in build_report(_Result())["rows"]:
        assert row["threshold_rule"] in {"roc_nearest_01", "default"}
        assert 0.0 <= row["recall_at_precision_50"] <= 1.0


def test_sanity_block_separates_collision_from_non_collision():
    """Doc section C4: the first thing to check. The synthetic scores embed a
    deliberate signal, so separation must appear in the right direction."""
    report = build_report(_Result())
    for name, (_, higher, _) in METRICS.items():
        block = report["sanity"][name]
        if block["mean_collision"] is None or block["mean_non_collision"] is None:
            continue
        if higher:
            assert block["mean_collision"] > block["mean_non_collision"], name
        else:
            assert block["mean_collision"] < block["mean_non_collision"], name


def test_stratified_buckets_are_reported():
    report = build_report(_Result())
    assert report["stratified"], "expected at least one populated bucket"
    for bucket, payload in report["stratified"].items():
        assert payload["n_pairs"] > 0
        assert {r["metric"] for r in payload["rows"]} == set(METRICS)


def test_metric_report_falls_back_when_a_label_class_is_missing():
    scores = np.linspace(0.0, 1.0, 50)
    row = metric_report(scores, np.zeros(50, dtype=int), "min_dist", False, 2.0)
    assert row["threshold_rule"] == "default"
    assert np.isnan(row["pr_auc"]) and np.isnan(row["roc_auc"])
