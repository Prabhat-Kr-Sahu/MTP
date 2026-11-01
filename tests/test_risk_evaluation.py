"""Evaluation axes added for the risk comparison (doc sections B2-B6)."""
import numpy as np
import pytest

from src.evaluation.collision import (
    brier_score,
    compute_pr_auc,
    earliness,
    lead_time_at_recall,
    log_loss,
    murphy_decomposition,
    recall_at_precision,
    reliability_bins,
    risk_regression,
    roc_curve,
    select_threshold,
    stratify,
)
from src.evaluation.risk_metrics import is_collision

# Separable: the 20 positives sit at low scores, as for a distance metric.
LOW = np.r_[np.linspace(0.1, 1.0, 20), np.linspace(5.0, 9.0, 80)]
HIGH = np.r_[np.linspace(50.0, 80.0, 20), np.linspace(0.1, 5.0, 80)]
LABELS = np.r_[np.ones(20), np.zeros(80)]


def _calibrated(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, n)
    return p, (rng.uniform(0.0, 1.0, n) < p).astype(int)


def test_pr_auc_unchanged_by_the_shared_curve_refactor():
    """compute_pr_auc now routes through the shared _pr_curve sweep.

    The expected values were produced by the pre-refactor implementation at
    commit 09b1500 and matched bit-for-bit afterwards, so this pins behaviour
    rather than merely restating the current output.
    """
    mixed = np.r_[np.linspace(0.1, 6.0, 20), np.linspace(1.0, 9.0, 80)]
    noisy = np.random.default_rng(7).uniform(0.0, 10.0, 100)
    for scores, higher, expected in [
        (LOW, False, 1.0000000000000002),
        (HIGH, True, 1.0000000000000002),
        (mixed, False, 0.4573535882315947),
        (noisy, False, 0.16259272944182016),
    ]:
        assert compute_pr_auc(scores, LABELS, higher) == pytest.approx(expected, rel=1e-12)


def test_roc_curve_starts_at_the_origin_and_ends_at_one():
    fpr, tpr, thresholds = roc_curve(LOW, LABELS, higher_is_riskier=False)
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
    assert np.isinf(thresholds[0])


def test_select_threshold_sign_and_direction():
    """The returned threshold must classify correctly through is_collision for
    BOTH directions -- the negation _prepare_auc_inputs applies for
    lower-is-riskier metrics is the easiest thing here to get backwards."""
    low_threshold = select_threshold(LOW, LABELS, higher_is_riskier=False)
    predicted = [is_collision("min_dist", v, low_threshold) for v in LOW]
    assert np.array_equal(np.array(predicted, dtype=int), LABELS)

    high_threshold = select_threshold(HIGH, LABELS, higher_is_riskier=True)
    predicted = [is_collision("drac", v, high_threshold) for v in HIGH]
    assert np.array_equal(np.array(predicted, dtype=int), LABELS)


def test_recall_at_precision():
    assert recall_at_precision(LOW, LABELS, False, 0.5) == pytest.approx(1.0)
    assert recall_at_precision(LOW, LABELS, False, 1.01) == 0.0


def test_brier_and_log_loss_at_the_extremes():
    y = np.r_[np.ones(10), np.zeros(10)]
    assert brier_score(y, y) == pytest.approx(0.0)
    assert brier_score(1.0 - y, y) == pytest.approx(1.0)
    assert log_loss(y, y) < 1e-9
    assert log_loss(np.full(20, 0.5), y) == pytest.approx(np.log(2.0))


def test_murphy_decomposition_closes_exactly():
    p, y = _calibrated()
    for n_bins in (10, 50, 200):
        d = murphy_decomposition(p, y, n_bins=n_bins)
        assert d["brier"] == pytest.approx(
            d["reliability"] - d["resolution"] + d["uncertainty"] + d["within_bin"],
            abs=1e-12,
        )


def test_calibrated_forecast_has_near_zero_reliability():
    p, y = _calibrated()
    assert murphy_decomposition(p, y)["reliability"] < 1e-3


def test_constant_forecast_has_zero_resolution():
    _, y = _calibrated()
    constant = np.full(y.size, y.mean())
    assert murphy_decomposition(constant, y)["resolution"] == pytest.approx(0.0, abs=1e-12)


def test_reliability_bins_count_every_sample():
    p, y = _calibrated(n=500)
    bins = reliability_bins(p, y, n_bins=5)
    assert sum(bins["bin_count"]) == 500
    assert len(bins["bin_edges"]) == 6


def test_probabilities_outside_the_unit_interval_are_rejected():
    with pytest.raises(ValueError):
        brier_score(np.array([0.5, 1.5]), np.array([0, 1]))


def test_earliness_scores_a_missed_positive_as_zero_lead():
    prefix = np.array([
        [9.0, 8.0, 3.0, 1.0, 1.0],    # fires at 4 s, event 4.5 s -> lead 0.5
        [9.0, 9.0, 9.0, 9.0, 0.5],    # fires at 5 s, event 4.5 s -> lead 0
        [9.0, 9.0, 9.0, 9.0, 9.0],    # never fires -> miss, lead 0
        [9.0, 9.0, 9.0, 9.0, 9.0],    # negative pair, ignored
    ])
    labels = np.array([1, 1, 1, 0])
    events = np.array([4.5, 4.5, 2.0, 0.0])
    out = earliness(prefix, labels, events, 2.0, higher_is_riskier=False)
    assert out["n_positive"] == 3
    assert out["detected_fraction"] == pytest.approx(2 / 3)
    assert out["mean_lead_time"] == pytest.approx(0.5 / 3)
    assert out["recall_by_prefix"]["5s"] == pytest.approx(2 / 3)


def test_lead_time_at_recall_is_bounded_by_the_horizon():
    prefix = np.array([
        [9.0, 8.0, 3.0, 1.0, 1.0],
        [9.0, 9.0, 9.0, 9.0, 0.5],
        [9.0, 9.0, 9.0, 9.0, 9.0],
        [9.0, 9.0, 9.0, 9.0, 9.0],
    ])
    labels = np.array([1, 1, 1, 0])
    events = np.array([4.5, 4.5, 2.0, 0.0])
    got = lead_time_at_recall(prefix, labels, events, higher_is_riskier=False)
    assert 0.0 <= got <= 5.0


def test_earliness_rejects_a_shape_mismatch():
    with pytest.raises(ValueError):
        earliness(np.zeros((3, 5)), np.array([1, 0]), np.array([1.0, 2.0]), 1.0)


def test_risk_regression_excludes_non_finite_pairs():
    out = risk_regression(np.array([1.0, 2.0, np.inf, 4.0]),
                          np.array([1.1, 2.5, 3.0, np.inf]))
    assert out["n"] == 2 and out["n_excluded"] == 2
    assert out["mr"] == pytest.approx(0.5)      # 0.09 within 0.15, 0.20 outside


def test_risk_regression_absolute_mode():
    out = risk_regression(np.array([1.0, 2.0]), np.array([1.1, 5.0]),
                          mr_threshold=1.0, normalise=False)
    assert out["mr"] == pytest.approx(0.5)


def test_stratify_buckets_are_disjoint_and_cover_every_pair():
    times = np.array([0.0, 0.5, 1.5, 2.5, 4.0, 3.0])
    labels = np.array([1, 1, 1, 1, 1, 0])
    masks = stratify(times, labels)
    stacked = np.stack(list(masks.values()))
    assert stacked.sum(axis=0).tolist() == [1, 1, 1, 1, 1, 1]      # covering
    assert stacked.sum() == labels.size                            # disjoint
    # A closest approach at t = 0 belongs to the most imminent bucket.
    assert masks["collision in 1s"][0]


def test_stratify_keeps_negatives_out_of_the_collision_buckets():
    masks = stratify(np.array([0.5, 0.5]), np.array([1, 0]))
    assert masks["collision in 1s"].tolist() == [True, False]
    assert masks["non-collision"].tolist() == [False, True]
