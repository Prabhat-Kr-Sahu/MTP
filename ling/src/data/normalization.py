"""Train-only feature normalization (plan.md sections 6, 15, 57)."""
import json
import os
import torch


def compute_normalization_stats(states_list):
    """Compute per-feature mean/std from TRAINING states only.

    Args:
        states_list: list of (1, T_h, N_i, F) tensors (N_i may vary).
    Returns:
        dict with mean/std lists.
    """
    # Flatten all states to (total_entries, F) ignoring variable N
    flat = []
    for s in states_list:
        # s: (1, T_h, N_i, F) → reshape to (-1, F)
        flat.append(s.reshape(-1, s.shape[-1]))
    all_flat = torch.cat(flat, dim=0)  # (total, F)
    mean = all_flat.mean(dim=0)
    std = all_flat.std(dim=0).clamp(min=1e-6)
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
