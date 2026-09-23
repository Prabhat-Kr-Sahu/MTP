"""Evaluation metrics for STRAP trajectory prediction."""
from src.logger import get_logger
logger = get_logger()
import torch
import numpy as np
from typing import Dict, List


def compute_rmse(pred_dist: torch.Tensor, gt_positions: torch.Tensor) -> Dict[str, float]:
    """
    Compute RMSE at each prediction horizon.

    Args:
        pred_dist: (B, T_f, 5) predicted distribution [mu_x, mu_y, ...]
        gt_positions: (B, T_f, 2) ground truth positions

    Returns:
        dict with RMSE at 1s, 2s, 3s, 4s, 5s and average
    """
    pred_mu = pred_dist[:, :, :2]  # (B, T_f, 2)
    errors = torch.norm(pred_mu - gt_positions, dim=-1)  # (B, T_f)

    results = {}
    # NGSIM is 10Hz, so timesteps correspond to:
    # 1s = 10 steps, 2s = 20 steps, etc.
    horizon_indices = [9, 19, 29, 39, 49]  # 0-indexed
    horizon_labels = ["1s", "2s", "3s", "4s", "5s"]

    for label, idx in zip(horizon_labels, horizon_indices):
        if idx < errors.shape[1]:
            results[f"rmse_{label}"] = errors[:, idx].mean().item()

    results["rmse_avg"] = errors.mean().item()
    return results


def compute_collision_metrics(
    pred_dist: torch.Tensor,
    gt_positions: torch.Tensor,
    target_positions: torch.Tensor,
    collision_threshold: float = 2.0,
) -> Dict[str, float]:
    """
    Compute collision-related metrics.

    Args:
        pred_dist: predicted trajectory distribution
        gt_positions: ground truth positions
        target_positions: positions of other vehicles
        collision_threshold: distance threshold for collision

    Returns:
        dict with collision metrics
    """
    pred_mu = pred_dist[:, :, :2]
    # Distance between predicted target and other vehicles
    # Simplified: just compute min distance to target
    distances = torch.norm(pred_mu - target_positions.unsqueeze(1), dim=-1)
    min_dist, _ = distances.min(dim=-1)

    predicted_collision = (min_dist < collision_threshold).float()
    actual_collision = (torch.norm(gt_positions - target_positions.unsqueeze(1), dim=-1).min(dim=-1).values < collision_threshold).float()

    tp = (predicted_collision * actual_collision).sum().item()
    fp = (predicted_collision * (1 - actual_collision)).sum().item()
    fn = ((1 - predicted_collision) * actual_collision).sum().item()
    tn = ((1 - predicted_collision) * (1 - actual_collision)).sum().item()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "predicted_collision_rate": predicted_collision.mean().item(),
        "actual_collision_rate": actual_collision.mean().item(),
    }


def evaluate_trajectory(model, dataloader, device) -> Dict[str, float]:
    """Full trajectory evaluation."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                states, gt_positions, _, mask = batch
                mask = mask.to(device)
            else:
                states, gt_positions, _ = batch
                mask = None
            
            states = states.to(device)
            gt_positions = gt_positions.to(device)

            traj_dist, _, _ = model(states, mask)
            metrics = compute_rmse(traj_dist, gt_positions)
            all_metrics.append(metrics)

    # Average across batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_metrics


def print_metrics(metrics: Dict[str, float], label: str = ""):
    """Pretty print evaluation metrics."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Metrics {label}")
    logger.info(f"{'='*50}")
    for key, val in metrics.items():
        logger.info(f"  {key}: {val:.4f}")
    logger.info(f"{'='*50}\n")

