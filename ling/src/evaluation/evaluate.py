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
    squared_displacement = (pred_mu - gt_positions).square().sum(dim=-1)

    results = {}
    # NGSIM is 10Hz, so timesteps correspond to:
    # 1s = 10 steps, 2s = 20 steps, etc.
    horizon_indices = [9, 19, 29, 39, 49]  # 0-indexed
    horizon_labels = ["1s", "2s", "3s", "4s", "5s"]

    for label, idx in zip(horizon_labels, horizon_indices):
        if idx < squared_displacement.shape[1]:
            results[f"rmse_{label}"] = squared_displacement[:, idx].mean().sqrt().item()

    results["rmse_avg"] = float(np.mean(list(results.values())))
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
    squared_error_sum = None
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 5:
                states, gt_positions, _, mask, _ = batch
                mask = mask.to(device)
            elif len(batch) == 4:
                states, gt_positions, _, mask = batch
                mask = mask.to(device)
            else:
                states, gt_positions, _ = batch
                mask = None
            
            states = states.to(device)
            gt_positions = gt_positions.to(device)

            traj_dist, _, _ = model(states, mask)
            squared_displacement = (
                traj_dist[:, :, :2] - gt_positions
            ).square().sum(dim=-1)
            batch_sum = squared_displacement.sum(dim=0)
            squared_error_sum = (
                batch_sum if squared_error_sum is None
                else squared_error_sum + batch_sum
            )
            sample_count += gt_positions.shape[0]

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty test dataset")

    horizon_rmse = (squared_error_sum / sample_count).sqrt()
    horizon_indices = [9, 19, 29, 39, 49]
    horizon_labels = ["1s", "2s", "3s", "4s", "5s"]
    metrics = {
        f"rmse_{label}": horizon_rmse[index].item()
        for label, index in zip(horizon_labels, horizon_indices)
        if index < len(horizon_rmse)
    }
    metrics["rmse_avg"] = float(np.mean(list(metrics.values())))
    return metrics


def print_metrics(metrics: Dict[str, float], label: str = ""):
    """Pretty print evaluation metrics."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Metrics {label}")
    logger.info(f"{'='*50}")
    for key, val in metrics.items():
        logger.info(f"  {key}: {val:.4f}")
    logger.info(f"{'='*50}\n")

