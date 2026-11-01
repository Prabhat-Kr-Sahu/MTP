"""Evaluation metrics for STRAP trajectory prediction."""
from src.logger import get_logger
logger = get_logger()
import torch
import numpy as np
from typing import Dict


HORIZON_STEPS = [9, 19, 29, 39, 49]  # 1s..5s at 10 Hz (0-indexed)


def horizon_rmse(squared_error_sum, sample_count):
    """Per-horizon RMSE from summed squared displacement (T_f,) over sample_count samples."""
    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty test dataset")
    rmse = (squared_error_sum / sample_count).sqrt()
    metrics = {
        f"rmse_{s // 10 + 1}s": rmse[s].item() for s in HORIZON_STEPS if s < len(rmse)
    }
    metrics["rmse_avg"] = float(np.mean(list(metrics.values())))
    return metrics


def evaluate_trajectory(model, dataloader, device) -> Dict[str, float]:
    """Full trajectory evaluation."""
    model.eval()
    squared_error_sum = None
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            states, gt_positions, _, mask = batch[:4]
            states, gt_positions = states.to(device), gt_positions.to(device)
            mask = mask.to(device) if mask is not None else None
            dims = batch[5].to(device) if len(batch) > 5 else None

            traj_dist, _, _ = model(states, mask, dims)
            squared_displacement = (
                traj_dist[:, :, :2] - gt_positions
            ).square().sum(dim=-1)
            batch_sum = squared_displacement.sum(dim=0)
            squared_error_sum = (
                batch_sum if squared_error_sum is None
                else squared_error_sum + batch_sum
            )
            sample_count += gt_positions.shape[0]

    return horizon_rmse(squared_error_sum, sample_count)


def print_metrics(metrics: Dict[str, float], label: str = ""):
    """Pretty print evaluation metrics."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Metrics {label}")
    logger.info(f"{'='*50}")
    for key, val in metrics.items():
        logger.info(f"  {key}: {val:.4f}")
    logger.info(f"{'='*50}\n")
