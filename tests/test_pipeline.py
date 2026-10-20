import torch

from src.pipeline import constant_velocity, predict_target


class _StandStill(torch.nn.Module):
    """Predicts the target stays at its last observed position (relative 0,0)."""
    def forward(self, states, mask, dims=None):
        return torch.zeros(states.shape[0], 50, 5), None, None


def test_target_and_candidates_share_the_target_relative_frame():
    states = torch.zeros(1, 30, 2, 11)          # raw, un-normalized
    sample = (states, None, None, torch.ones(1, 2, dtype=torch.bool), [1, 2], None,
              torch.tensor([[500.0, 7.0]]), torch.ones(1, 2, 2))  # absolute target origin
    norm_stats = {"mean": [5.0] * 11, "std": [2.0] * 11}

    target = predict_target(_StandStill(), sample, norm_stats).cpu()  # DEVICE may be cuda
    neighbors = constant_velocity([[10.0, 0.0]], [[3.0, 0.0]], 50).cpu()  # 10 m ahead, 3 m/s

    torch.testing.assert_close(target[0, 0], torch.zeros(2))             # relative frame
    assert neighbors.shape == (1, 50, 1, 2)
    torch.testing.assert_close(neighbors[0, 0, 0], torch.tensor([10.3, 0.0]))  # 10 + 3*0.1
    torch.testing.assert_close(neighbors[0, -1, 0], torch.tensor([25.0, 0.0])) # 10 + 3*5.0
