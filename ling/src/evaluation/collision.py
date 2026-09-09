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
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": acc,
    }


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
