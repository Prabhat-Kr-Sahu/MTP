"""Train-only feature normalization (plan.md sections 6, 15, 57)."""
import json
import os
import torch


def compute_normalization_stats(states_list):
    """Compute per-feature mean/std from TRAINING states only.

    Args:
        states_list: list of (1, T_h, N, F) tensors.
    Returns:
        dict with mean/std lists.
    """
    all_states = torch.cat(states_list, dim=0)  # (N_samples, T_h, N, F)
    mean = all_states.mean(dim=(0, 1, 2))
    std = all_states.std(dim=(0, 1, 2)).clamp(min=1e-6)
    return {"mean": mean.tolist(), "std": std.tolist(), "feature_dim": mean.numel()}


def apply_normalization(states, stats):
    mean = torch.tensor(stats["mean"], dtype=states.dtype, device=states.device)
    std = torch.tensor(stats["std"], dtype=states.dtype, device=states.device)
    return (states - mean) / std


def save_stats(stats, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path):
    with open(path) as f:
        return json.load(f)
