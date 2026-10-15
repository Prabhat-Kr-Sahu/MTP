"""Tests for MTP extension modules (intentions, collision, normalization)."""
import numpy as np
import torch


from src.intentions.kmeans import fit_intention_codebook
from src.evaluation.collision import (
    collision_metrics_from_labels,
    compute_pr_auc,
    compute_roc_auc,
)
from src.data.normalization import compute_normalization_stats, apply_normalization


def test_kmeans_codebook_shape():
    endpoints = np.random.randn(500, 4).astype(np.float32) * 10
    centers = fit_intention_codebook(endpoints, k=100, seed=0)
    assert centers.shape == (100, 4)


def test_collision_metrics():
    m = collision_metrics_from_labels([1, 0, 1, 0], [1, 0, 0, 0])
    assert 0 <= m["precision"] <= 1
    assert 0 <= m["recall"] <= 1
    assert 0 <= m["f1"] <= 1
    assert abs(m["false_alarm_rate"] - 1 / 3) < 1e-6


def test_auc_respects_metric_risk_direction():
    labels = [1, 1, 0, 0]
    lower_is_riskier = [0.1, 0.2, 0.8, 0.9]
    higher_is_riskier = [0.9, 0.8, 0.2, 0.1]

    for scores, higher in (
        (lower_is_riskier, False),
        (higher_is_riskier, True),
    ):
        assert abs(compute_pr_auc(scores, labels, higher) - 1.0) < 1e-6
        assert abs(compute_roc_auc(scores, labels, higher) - 1.0) < 1e-6


def test_auc_handles_infinite_no_event_scores():
    scores = [0.1, 0.2, float("inf"), float("inf")]
    labels = [1, 1, 0, 0]
    assert abs(compute_pr_auc(scores, labels) - 1.0) < 1e-6
    assert abs(compute_roc_auc(scores, labels) - 1.0) < 1e-6


def test_normalization_roundtrip():
    states = [torch.randn(1, 30, 5, 12) * 5 + 2 for _ in range(4)]
    states[0][0, 0, 0, 0] = float("nan")
    stats = compute_normalization_stats(states)
    normed = apply_normalization(states[0], stats)
    assert normed.shape == states[0].shape
    assert torch.isfinite(normed).all()
    assert normed[0, 0, 0, 0] == 0
