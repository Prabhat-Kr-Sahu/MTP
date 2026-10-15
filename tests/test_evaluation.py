import numpy as np
import pytest
import torch

from src.evaluation.evaluate import compute_rmse, evaluate_trajectory


def test_compute_rmse_uses_euclidean_squared_error_and_horizon_average():
    prediction = torch.zeros(2, 50, 5)
    target = torch.zeros(2, 50, 2)
    target[0, 9] = torch.tensor([3.0, 4.0])
    target[0, 19] = torch.tensor([6.0, 8.0])

    metrics = compute_rmse(prediction, target)

    expected = [np.sqrt(25 / 2), np.sqrt(100 / 2), 0.0, 0.0, 0.0]
    assert metrics["rmse_1s"] == pytest.approx(expected[0])
    assert metrics["rmse_2s"] == pytest.approx(expected[1])
    assert metrics["rmse_avg"] == pytest.approx(np.mean(expected))


class _ZeroPredictor:
    def eval(self):
        return self

    def __call__(self, states, mask):
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