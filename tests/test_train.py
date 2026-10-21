import math

import numpy as np
import torch

from src.config import DEVICE, K
from src.models.strap import STRAP
from src.train import fit


def _sample(n_vehicles=3):
    """Synthetic sample in the create_sample tuple layout (leading batch dim 1)."""
    return (
        torch.randn(1, 30, n_vehicles, 11),                          # states
        torch.cumsum(torch.full((1, 50, 2), 0.1), dim=1),            # gt_pos
        torch.randn(1, n_vehicles - 1, 4),                           # gt_goals
        torch.ones(1, n_vehicles, dtype=torch.bool),                 # mask
        list(range(n_vehicles)),                                     # vehicle_ids
        torch.ones(1, n_vehicles - 1, dtype=torch.bool),             # goal_mask
        torch.zeros(1, 2),                                           # target_origin
        torch.tensor([[[4.5, 1.8]] * n_vehicles]),                   # dims
    )


def test_fit_uses_tuned_hparams_reports_each_epoch_and_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    seen = []
    centers = np.random.default_rng(0).normal(size=(K, 4)).astype(np.float32)

    log = fit(
        STRAP(dropout=0.2).to(DEVICE), [_sample() for _ in range(4)], [_sample() for _ in range(2)],
        centers, num_epochs=2, learning_rate=1e-3, batch_size=2, lr_decay=0.5,
        weight_decay=0.0, beta=1.0, save_dir=None,
        on_epoch=lambda epoch, entry: seen.append((epoch, entry["val_rmse"])),
    )

    assert [epoch for epoch, _ in seen] == [1, 2]
    assert all(math.isfinite(rmse) for _, rmse in seen)
    assert len(log) == 2
    assert math.isclose(log[-1]["lr"], 1e-3 * 0.5 ** 2)
    assert not any(tmp_path.iterdir())  # save_dir=None writes no files
