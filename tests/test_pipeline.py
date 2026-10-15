import torch

from src.pipeline import rollout_pairs


class _StandStill(torch.nn.Module):
    """Predicts the target stays at its last observed position (relative 0,0)."""
    def forward(self, states, mask):
        return torch.zeros(states.shape[0], 50, 5), None, None


def test_rollout_pairs_share_the_target_relative_frame():
    states = torch.zeros(1, 30, 2, 11)          # raw, un-normalized
    states[0, -1, 1, :2] = torch.tensor([10.0, 0.0])  # neighbour 10 m ahead of target
    states[0, -1, 1, 2] = 3.0 / 30.0              # neighbour speed 3 m/s, /30 as in create_sample
    sample = (states, None, None, torch.ones(1, 2, dtype=torch.bool), [1, 2], None,
              torch.tensor([[500.0, 7.0]]))        # absolute target origin
    norm_stats = {"mean": [5.0] * 11, "std": [2.0] * 11}

    target, neighbors = rollout_pairs(_StandStill(), sample, norm_stats)

    target, neighbors = target.cpu(), neighbors.cpu()   # DEVICE may be cuda
    torch.testing.assert_close(target[0, 0], torch.zeros(2))             # relative frame
    torch.testing.assert_close(neighbors[0, 0, 0], torch.tensor([10.3, 0.0]))  # 10 + 3*0.1
    torch.testing.assert_close(neighbors[0, -1, 0], torch.tensor([25.0, 0.0])) # 10 + 3*5.0
