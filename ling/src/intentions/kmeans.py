"""K-means intention modes for STRAP (plan.md Phase 9 / section 17).

Builds K=100 spatial intention modes from TRAINING ground-truth future
endpoints only. Each intention is [x, y, vx, vy].
"""
import os
import pickle
import numpy as np
import torch


def fit_intention_codebook(endpoints, k=100, seed=42):
    """Fit k-means codebook on training endpoints.

    Args:
        endpoints: (N, 4) array of [x, y, vx, vy] training GT goals.
        k: number of intention modes.
        seed: random seed.
    Returns:
        centers: (K, 4) cluster centers.
    """
    endpoints = np.asarray(endpoints, dtype=np.float64)
    n = len(endpoints)
    k = min(k, n)
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(endpoints)
        centers = km.cluster_centers_.astype(np.float32)
    except Exception:
        # Fallback: torch-only k-means (env sklearn/psutil broken).
        centers = _torch_kmeans(
            torch.from_numpy(endpoints.astype(np.float32)), k=k, seed=seed
        ).numpy()
    if len(centers) < k:
        # Pad by repeating (should not happen with enough training data).
        pad = np.repeat(centers[-1:], k - len(centers), axis=0)
        centers = np.concatenate([centers, pad], axis=0)
    return centers


def _torch_kmeans(data, k, seed=42, iters=50):
    """Minimal k-means with torch (CPU). data: (N, D)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(data.shape[0], generator=g)[:k]
    centers = data[idx].clone()
    for _ in range(iters):
        dist = torch.cdist(data, centers)  # (N, K)
        assign = dist.argmin(dim=1)
        new_centers = centers.clone()
        for c in range(k):
            pts = data[assign == c]
            if len(pts) > 0:
                new_centers[c] = pts.mean(dim=0)
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers


def save_codebook(centers, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"centers": np.asarray(centers, dtype=np.float32)}, f)


def load_codebook(path, k=100):
    with open(path, "rb") as f:
        data = pickle.load(f)
    centers = np.asarray(data["centers"], dtype=np.float32)
    assert centers.shape == (k, 4), f"Expected ({k},4), got {centers.shape}"
    return centers


def build_codebook_from_samples(samples, k=100, seed=42):
    """Build codebook from training samples.

    Each sample is (states, gt_positions, gt_goals); uses target GT
    endpoint (final position) + final velocity estimate.
    """
    endpoints = []
    for _, gt_pos, _ in samples:
        gt = gt_pos[0] if isinstance(gt_pos, torch.Tensor) and gt_pos.dim() == 3 else gt_pos
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        gt = np.asarray(gt)
        end_pos = gt[-1, :2]
        vel = (gt[-1, :2] - gt[max(0, len(gt) - 10), :2]) / 1.0  # ~1s velocity proxy
        endpoints.append(np.concatenate([end_pos, vel]))
    return fit_intention_codebook(np.stack(endpoints), k=k, seed=seed)
