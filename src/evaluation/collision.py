"""MTP pairwise collision-risk extension (plan.md sections 48-56).

STRAP predicts target trajectories; this module converts predicted
trajectories into pairwise interaction / collision-risk outputs.
Ground-truth future is used ONLY for label generation (evaluation).
"""
import torch
import numpy as np


def pairwise_min_distance(traj_i, traj_j):
    """Minimum center distance over the horizon.

    Args:
        traj_i, traj_j: (..., T_f, 2) predicted or GT positions.
    Returns:
        d_min, t_min_idx.
    """
    dist = torch.norm(traj_i - traj_j, dim=-1)  # (..., T_f)
    d_min, t_idx = dist.min(dim=-1)
    return d_min, t_idx


def pairwise_min_distance_horizon(traj_i, traj_j, horizon_steps):
    """Min distance restricted to first `horizon_steps` timesteps."""
    dist = torch.norm(traj_i[..., :horizon_steps, :] - traj_j[..., :horizon_steps, :], dim=-1)
    return dist.min(dim=-1).values


def collision_label_from_trajectories(traj_i, traj_j, threshold=2.0):
    """Binary GT collision label (evaluation only)."""
    d_min, _ = pairwise_min_distance(traj_i, traj_j)
    return (d_min < threshold).float()


def predicted_collision(traj_i_pred, traj_j_pred, threshold=2.0):
    """Binary predicted collision from PREDICTED trajectories (no GT)."""
    d_min, t_idx = pairwise_min_distance(traj_i_pred, traj_j_pred)
    return (d_min < threshold).float(), d_min, t_idx


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


def compute_pr_auc(pred_scores, gt_labels, higher_is_riskier=False):
    """Compute Precision-Recall AUC from continuous risk scores.

    Args:
        pred_scores: Continuous risk scores for each pair.
        gt_labels: Binary ground truth labels (1 = collision)
        higher_is_riskier: Whether larger input scores mean higher risk.

    Returns:
        PR-AUC value
    """
    pred_scores, gt_labels = _prepare_auc_inputs(
        pred_scores, gt_labels, higher_is_riskier
    )
    positive_count = int(gt_labels.sum())
    if positive_count == 0:
        raise ValueError("PR-AUC requires at least one positive label")

    order = np.argsort(-pred_scores, kind="mergesort")
    sorted_scores = pred_scores[order]
    sorted_labels = gt_labels[order]
    true_positives = np.cumsum(sorted_labels)
    false_positives = np.cumsum(1 - sorted_labels)
    end_indices = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    precision = true_positives[end_indices] / (end_indices + 1)
    recall = true_positives[end_indices] / positive_count
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


def evaluate_horizons(pred_traj, gt_traj, other_traj, horizons=(10, 20, 30, 40, 50), threshold=2.0):
    """Collision detection performance at 1s..5s horizons (10Hz).

    Args:
        pred_traj: (B, T_f, 2) predicted target trajectory means.
        gt_traj: (B, T_f, 2) GT target trajectories (labels only).
        other_traj: (B, T_f, 2) other vehicle GT/predicted positions.
    """
    out = {}
    labels = ["1s", "2s", "3s", "4s", "5s"]
    for label, h in zip(labels, horizons):
        h = min(h, pred_traj.shape[1])
        p, _, _ = predicted_collision(
            pred_traj[:, :h], other_traj[:, :h], threshold
        )
        g = collision_label_from_trajectories(
            gt_traj[:, :h], other_traj[:, :h], threshold
        )
        m = collision_metrics_from_labels(
            p.detach().cpu().numpy(), g.detach().cpu().numpy()
        )
        out[label] = m
    return out
