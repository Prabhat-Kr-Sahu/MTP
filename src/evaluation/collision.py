"""MTP pairwise collision-risk extension (plan.md sections 48-56).

STRAP predicts target trajectories; this module converts predicted
trajectories into pairwise interaction / collision-risk outputs.
Ground-truth future is used ONLY for label generation (evaluation).
"""
import numpy as np


def collision_metrics_from_labels(pred, gt):
    """Precision/recall/F1 from binary prediction/label tensors."""
    pred = np.asarray(pred).astype(float)
    gt = np.asarray(gt).astype(float)
    tp = float(((pred == 1) & (gt == 1)).sum())
    fp = float(((pred == 1) & (gt == 0)).sum())
    fn = float(((pred == 0) & (gt == 1)).sum())
    tn = float(((pred == 0) & (gt == 0)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    false_alarm_rate = fp / (fp + tn + 1e-9)  # FPR
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": acc,
        "false_alarm_rate": false_alarm_rate,
    }


def _prepare_auc_inputs(pred_scores, gt_labels, higher_is_riskier):
    scores = np.asarray(pred_scores, dtype=float).reshape(-1)
    labels = np.asarray(gt_labels).reshape(-1)
    if scores.size != labels.size:
        raise ValueError("pred_scores and gt_labels must have the same length")
    if np.isnan(scores).any():
        raise ValueError("pred_scores must not contain NaN")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("gt_labels must contain only 0 and 1")

    scores = scores if higher_is_riskier else -scores
    finite = scores[np.isfinite(scores)]
    if not finite.size:
        raise ValueError("pred_scores must contain at least one finite score")
    lower = np.nextafter(finite.min(), -np.inf)
    upper = np.nextafter(finite.max(), np.inf)
    scores = np.nan_to_num(scores, posinf=upper, neginf=lower)
    return scores, labels.astype(int)


def _sweep(pred_scores, gt_labels, higher_is_riskier):
    """Cumulative TP/FP at every distinct score, descending, ties deduped.

    Returns (true_positives, false_positives, thresholds, n_positive, n_negative)
    in the internal space where larger always means riskier.
    """
    scores, labels = _prepare_auc_inputs(pred_scores, gt_labels, higher_is_riskier)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    false_positives = np.cumsum(1 - sorted_labels)
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    positive_count = int(labels.sum())
    return (
        true_positives[ends], false_positives[ends], sorted_scores[ends],
        positive_count, len(labels) - positive_count,
    )


def _pr_curve(pred_scores, gt_labels, higher_is_riskier=False):
    """(precision, recall, thresholds) at every distinct score, descending."""
    tp, fp, thresholds, positive_count, _ = _sweep(
        pred_scores, gt_labels, higher_is_riskier
    )
    if positive_count == 0:
        raise ValueError("PR-AUC requires at least one positive label")
    return tp / (tp + fp), tp / positive_count, thresholds


def compute_pr_auc(pred_scores, gt_labels, higher_is_riskier=False):
    """Compute Precision-Recall AUC from continuous risk scores.

    Args:
        pred_scores: Continuous risk scores for each pair.
        gt_labels: Binary ground truth labels (1 = collision)
        higher_is_riskier: Whether larger input scores mean higher risk.

    Returns:
        PR-AUC value
    """
    precision, recall, _ = _pr_curve(pred_scores, gt_labels, higher_is_riskier)
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    return float(np.sum(
        (recall[:-1] - recall[1:]) * (precision[:-1] + precision[1:]) / 2
    ))


def compute_roc_auc(pred_scores, gt_labels, higher_is_riskier=False):
    """Compute ROC-AUC from continuous risk scores."""
    pred_scores, gt_labels = _prepare_auc_inputs(
        pred_scores, gt_labels, higher_is_riskier
    )
    positive_count = int(gt_labels.sum())
    negative_count = len(gt_labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both positive and negative labels")

    order = np.argsort(pred_scores, kind="mergesort")
    sorted_scores = pred_scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(sorted_scores)]
    average_ranks = (starts + 1 + ends) / 2
    ranks = np.empty(len(sorted_scores), dtype=float)
    ranks[order] = np.repeat(average_ranks, ends - starts)
    positive_rank_sum = ranks[gt_labels == 1].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


# --------------------------------------------------------------------------- #
#  B2 -- operating points and the shared threshold rule
# --------------------------------------------------------------------------- #
def roc_curve(pred_scores, gt_labels, higher_is_riskier=False):
    """(fpr, tpr, thresholds) at every distinct score, with (0, 0) prepended.

    Thresholds are in the internal space where larger always means riskier;
    use select_threshold to get one back in the metric's own units.
    """
    tp, fp, thresholds, positive_count, negative_count = _sweep(
        pred_scores, gt_labels, higher_is_riskier
    )
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC requires both positive and negative labels")
    return (
        np.r_[0.0, fp / negative_count],
        np.r_[0.0, tp / positive_count],
        np.r_[np.inf, thresholds],
    )


def roc_points(pred_scores, gt_labels, higher_is_riskier=False, n_points=200):
    """ROC curve resampled onto a fixed FPR grid: {"fpr": [...], "tpr": [...]}.

    The grid is half log-spaced from 1e-5, because at base rates of 0.05-2 %
    the operating region is low FPR and a linear grid would spend every point
    where no alarm system lives. tpr is the step-function value at each fpr.
    """
    fpr, tpr, _ = roc_curve(pred_scores, gt_labels, higher_is_riskier)
    grid = np.unique(np.r_[0.0, np.logspace(-5, 0, n_points // 2),
                           np.linspace(0, 1, n_points - n_points // 2)])
    at = np.searchsorted(fpr, grid, side="right") - 1
    return {"fpr": grid.round(6).tolist(), "tpr": tpr[at].round(6).tolist()}


def auc_placements(pred_scores, gt_labels, higher_is_riskier=False):
    """Per positive: the fraction of negatives it out-ranks, ties counted half.

    Their mean is exactly compute_roc_auc. Holding the negatives fixed turns
    an AUC bootstrap over positives into a mean over these values, which is
    what makes the event-clustered bootstrap cheap and pairable across models.
    """
    scores, labels = _prepare_auc_inputs(pred_scores, gt_labels, higher_is_riskier)
    negatives = np.sort(scores[labels == 0])
    if not negatives.size or not labels.any():
        raise ValueError("placements require both positive and negative labels")
    positives = scores[labels == 1]
    below = np.searchsorted(negatives, positives, side="left")
    tied = np.searchsorted(negatives, positives, side="right") - below
    return (below + 0.5 * tied) / negatives.size


def event_bootstrap(placements, events, n_boot=2000, seed=42, alpha=0.05):
    """(lo, hi) percentile CI for ROC-AUC resampling whole events, not pairs.

    `placements` is one array (P,) or several stacked (K, P) on the same
    positives, e.g. baseline and tuned; the same event draw is applied to all,
    so a CI on their difference is paired. Returns arrays of shape (K, 2).
    Negatives are held fixed: they number ~1e6, so their sampling variance is
    negligible next to ~100 positive events.
    """
    placements = np.atleast_2d(np.asarray(placements, dtype=float))
    _, inverse = np.unique(events, return_inverse=True)
    n_events = inverse.max() + 1
    sums = np.stack([np.bincount(inverse, weights=p, minlength=n_events)
                     for p in placements])                      # (K, E)
    counts = np.bincount(inverse, minlength=n_events)           # (E,)
    draws = np.random.default_rng(seed).integers(0, n_events, (n_boot, n_events))
    weight = np.stack([np.bincount(d, minlength=n_events) for d in draws])  # (B, E)
    boot = (weight @ sums.T) / (weight @ counts)[:, None]       # (B, K)
    return np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0).T


def recall_at_precision(pred_scores, gt_labels, higher_is_riskier=False,
                        min_precision=0.5):
    """Highest recall whose precision is at least min_precision; 0.0 if unreachable.

    A fixed-precision operating point is the fairest single-number comparison
    for a safety detector, because PR-AUC alone is base-rate dependent.
    """
    precision, recall, _ = _pr_curve(pred_scores, gt_labels, higher_is_riskier)
    feasible = precision >= min_precision
    return float(recall[feasible].max()) if feasible.any() else 0.0


def select_threshold(pred_scores, gt_labels, higher_is_riskier=False):
    """Threshold at the ROC point nearest (fpr=0, tpr=1), in the metric's own units.

    One metric-agnostic rule applied identically to every measure, so no method
    gets a hand-tuned operating point (Jiao et al. 2024). The result is ready to
    pass straight to risk_metrics.is_collision(name, value, threshold).
    """
    fpr, tpr, thresholds = roc_curve(pred_scores, gt_labels, higher_is_riskier)
    best = thresholds[int(np.argmin(fpr ** 2 + (1 - tpr) ** 2))]
    if not np.isfinite(best):                      # degenerate: everything positive
        finite = thresholds[np.isfinite(thresholds)]
        best = finite[0] if finite.size else 0.0
    # is_collision uses a strict comparison, so nudge the boundary to keep the
    # chosen ROC point (scores >= best) on the positive side. The sign undoes
    # the negation _prepare_auc_inputs applies for lower-is-riskier metrics.
    if higher_is_riskier:
        return float(np.nextafter(best, -np.inf))
    return float(np.nextafter(-best, np.inf))


# --------------------------------------------------------------------------- #
#  B3 -- calibration (probability-valued metrics only)
# --------------------------------------------------------------------------- #
def _check_probabilities(probabilities, gt_labels):
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(gt_labels).reshape(-1)
    if p.size != y.size:
        raise ValueError("probabilities and gt_labels must have the same length")
    if not np.isfinite(p).all():
        raise ValueError("probabilities must be finite")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("gt_labels must contain only 0 and 1")
    return p, y.astype(float)


def brier_score(probabilities, gt_labels):
    """Mean squared error of a probability forecast; a strictly proper score."""
    p, y = _check_probabilities(probabilities, gt_labels)
    return float(np.mean((p - y) ** 2))


def log_loss(probabilities, gt_labels, eps=1e-12):
    """Negative log likelihood of a probability forecast; strictly proper."""
    p, y = _check_probabilities(probabilities, gt_labels)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _bins(probabilities, n_bins):
    """Equal-width bin index on [0, 1], with the top edge inclusive."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)
    return index, edges


def murphy_decomposition(probabilities, gt_labels, n_bins=10):
    """Split the Brier score into reliability, resolution and uncertainty.

    Report the split, not just the total: a respectable Brier score can hide
    poor reliability. A useful forecast has a Brier below its uncertainty term.

    Murphy's identity (brier = reliability - resolution + uncertainty) is exact
    only for forecasts that are constant within each bin. For continuous
    forecasts the remainder is the within-bin spread, which binning cannot
    attribute, so it is returned explicitly as `within_bin` and

        brier == reliability - resolution + uncertainty + within_bin

    holds exactly. A large `within_bin` relative to the other terms means the
    decomposition is bin-limited -- raise n_bins before reading much into it.
    """
    p, y = _check_probabilities(probabilities, gt_labels)
    index, _ = _bins(p, n_bins)
    base_rate = y.mean()
    reliability = resolution = 0.0
    for k in range(n_bins):
        in_bin = index == k
        count = int(in_bin.sum())
        if not count:
            continue
        mean_forecast = p[in_bin].mean()
        observed = y[in_bin].mean()
        reliability += count * (mean_forecast - observed) ** 2
        resolution += count * (observed - base_rate) ** 2
    total = len(y)
    brier = brier_score(p, y)
    reliability /= total
    resolution /= total
    uncertainty = base_rate * (1.0 - base_rate)
    return {
        "brier": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "within_bin": float(brier - (reliability - resolution + uncertainty)),
    }


def reliability_bins(probabilities, gt_labels, n_bins=10):
    """Reliability-diagram data; a diagonal observed-vs-forecast curve is calibrated."""
    p, y = _check_probabilities(probabilities, gt_labels)
    index, edges = _bins(p, n_bins)
    counts, mean_forecast, observed = [], [], []
    for k in range(n_bins):
        in_bin = index == k
        count = int(in_bin.sum())
        counts.append(count)
        mean_forecast.append(float(p[in_bin].mean()) if count else None)
        observed.append(float(y[in_bin].mean()) if count else None)
    return {
        "bin_edges": edges.tolist(), "bin_count": counts,
        "mean_predicted": mean_forecast, "observed_frequency": observed,
    }


# --------------------------------------------------------------------------- #
#  B4 -- earliness
# --------------------------------------------------------------------------- #
def earliness(prefix_scores, gt_labels, event_times, thresholds,
              higher_is_riskier=False, prefix_seconds=(1, 2, 3, 4, 5)):
    """Anticipation from horizon prefixes: how early does the measure fire?

    `prefix_scores` is (P, K): the metric evaluated on the first
    prefix_seconds[k] seconds of the predicted horizon. `event_times` is the
    ground-truth time of closest approach per pair, in seconds.

    The lead time is event_time minus the earliest prefix that crosses the
    threshold, floored at 0; pairs that never cross score 0, the standard
    convention for a missed anticipation.

    This is anticipation over the PREDICTION HORIZON, not the dashcam TTA that
    scans an observation-time score series: the metric contract yields one
    scalar per pair, so a per-timestep series over observation time does not
    exist here. Named and documented as a distinct quantity rather than
    presented as the TTA of the accident-anticipation literature.
    """
    scores = np.asarray(prefix_scores, dtype=float)
    labels = np.asarray(gt_labels).reshape(-1).astype(int)
    events = np.asarray(event_times, dtype=float).reshape(-1)
    seconds = np.asarray(prefix_seconds, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != labels.size:
        raise ValueError("prefix_scores must be (n_pairs, n_prefixes)")
    if scores.shape[1] != seconds.size:
        raise ValueError("prefix_seconds must match prefix_scores columns")
    if not labels.any():
        raise ValueError("earliness requires at least one positive label")

    thresholds = np.broadcast_to(
        np.asarray(thresholds, dtype=float), (scores.shape[1],)
    )
    fired = (scores > thresholds) if higher_is_riskier else (scores < thresholds)
    ever = fired.any(axis=1)
    first = np.where(ever, seconds[np.argmax(fired, axis=1)], np.inf)

    positive = labels == 1
    lead = np.where(
        ever[positive], np.maximum(0.0, events[positive] - first[positive]), 0.0
    )
    return {
        "mean_lead_time": float(lead.mean()),
        "median_lead_time": float(np.median(lead)),
        "detected_fraction": float(ever[positive].mean()),
        "recall_by_prefix": {
            "{:g}s".format(s): float(fired[positive, k].mean())
            for k, s in enumerate(seconds)
        },
        "n_positive": int(positive.sum()),
    }


def lead_time_at_recall(prefix_scores, gt_labels, event_times,
                        higher_is_riskier=False, target_recall=0.8,
                        prefix_seconds=(1, 2, 3, 4, 5)):
    """Mean lead time at the threshold whose full-horizon recall first reaches
    target_recall, so earliness cannot be inflated by lowering the threshold."""
    scores = np.asarray(prefix_scores, dtype=float)
    _, recall, thresholds = _pr_curve(scores[:, -1], gt_labels, higher_is_riskier)
    reached = np.flatnonzero(recall >= target_recall)
    if not reached.size:
        return 0.0
    internal = thresholds[reached[0]]
    own_units = (np.nextafter(internal, -np.inf) if higher_is_riskier
                 else np.nextafter(-internal, np.inf))
    return earliness(
        scores, gt_labels, event_times, own_units,
        higher_is_riskier=higher_is_riskier, prefix_seconds=prefix_seconds,
    )["mean_lead_time"]


# --------------------------------------------------------------------------- #
#  B5 -- risk as regression (against the oracle run)
# --------------------------------------------------------------------------- #
def risk_regression(pred_values, gt_values, mr_threshold=0.15, normalise=True):
    """Agreement between a measure on predicted futures and on true futures.

    MR is the fraction of pairs whose error exceeds mr_threshold -- relative
    when `normalise`, else absolute (Wang et al. use 0.15 normalised risk,
    2.5 m/s collision velocity, 0.25 s collision time).

    Pairs where either side is non-finite are excluded and counted, because
    inf means "no event" rather than a numeric prediction.
    """
    pred = np.asarray(pred_values, dtype=float).reshape(-1)
    gt = np.asarray(gt_values, dtype=float).reshape(-1)
    if pred.size != gt.size:
        raise ValueError("pred_values and gt_values must have the same length")
    usable = np.isfinite(pred) & np.isfinite(gt)
    if not usable.any():
        return {"mse": float("nan"), "mae": float("nan"), "mr": float("nan"),
                "n": 0, "n_excluded": int(pred.size)}
    error = pred[usable] - gt[usable]
    scale = np.abs(gt[usable]) + 1e-9 if normalise else 1.0
    return {
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))),
        "mr": float(np.mean(np.abs(error) / scale > mr_threshold)),
        "n": int(usable.sum()),
        "n_excluded": int((~usable).sum()),
    }


# --------------------------------------------------------------------------- #
#  B6 -- stratification by how imminent the collision is
# --------------------------------------------------------------------------- #
def stratify(gt_time, gt_labels, buckets=(1.0, 2.0, 3.0, 5.0)):
    """Disjoint boolean masks over pairs, by ground-truth time to closest approach.

    Mirrors STRAP's Table II (plan.md section 55). That table is the reason this
    exists: STRAP-B and STRAP-R share an identical average RMSE and separate
    only under this stratification, so an aggregate number cannot see the
    difference the comparison is trying to measure.
    """
    times = np.asarray(gt_time, dtype=float).reshape(-1)
    labels = np.asarray(gt_labels).reshape(-1).astype(int)
    if times.size != labels.size:
        raise ValueError("gt_time and gt_labels must have the same length")

    positive = labels == 1
    # Start below zero so a pair whose closest approach is at t = 0 -- which
    # happens whenever the two are already closest at the first predicted step
    # -- lands in the first bucket instead of in none of them.
    masks, lower = {}, -np.inf
    for upper in buckets:
        masks["collision in {:g}s".format(upper)] = (
            positive & (times > lower) & (times <= upper)
        )
        lower = upper
    masks["non-collision"] = ~positive
    return masks
