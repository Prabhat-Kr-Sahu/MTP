import numpy as np
import pytest
import torch

from src.evaluation.evaluate import evaluate_trajectory


class _ZeroPredictor:
    def eval(self):
        return self

    def __call__(self, states, mask, dims=None):
        return torch.zeros(states.shape[0], 50, 5), None, None


def test_evaluate_trajectory_weights_uneven_batches_by_sample():
    states_a = torch.zeros(2, 1, 1, 11)
    target_a = torch.zeros(2, 50, 2)
    target_a[0, 9, 0] = 5.0
    states_b = torch.zeros(1, 1, 1, 11)
    target_b = torch.zeros(1, 50, 2)
    mask_a = torch.ones(2, 1, dtype=torch.bool)
    mask_b = torch.ones(1, 1, dtype=torch.bool)
    batches = [
        (states_a, target_a, None, mask_a),
        (states_b, target_b, None, mask_b),
    ]

    metrics = evaluate_trajectory(_ZeroPredictor(), batches, torch.device("cpu"))

    assert metrics["rmse_1s"] == pytest.approx(np.sqrt(25 / 3))