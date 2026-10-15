"""Unit tests for risk field computation."""
import torch
import pytest

from src.risk.s_field import compute_s_field
from src.risk.o_field import compute_o_field, closest_approach


def test_s_field_decreases_with_distance():
    """S-field should decrease as vehicles move farther away."""
    delta_x = torch.tensor([0.0, 1.0, 5.0, 10.0])
    delta_y = torch.tensor([0.0, 1.0, 5.0, 10.0])
    risk = compute_s_field(delta_x, delta_y)

    assert risk[0] > risk[1] > risk[2] > risk[3]
    assert risk[0].item() == pytest.approx(1.0, abs=1e-6)  # self-distance = 0


def test_s_field_bounds():
    """S-field values should be in [0, 1]."""
    delta_x = torch.randn(10) * 5
    delta_y = torch.randn(10) * 5
    risk = compute_s_field(delta_x, delta_y)
    # NOTE: exp(-large) underflows to exactly 0.0 in float32 for far vehicles,
    # so allow >= 0 (mathematically (0, 1]).
    assert (risk >= 0).all()
    assert (risk <= 1.0 + 1e-6).all()  # Allow tiny numerical error


def test_o_field_decreases_with_distance():
    """O-field should decrease as predicted minimum distance increases."""
    d_pred = torch.tensor([1.0, 5.0, 10.0, 20.0])
    t_pred = torch.tensor([1.0, 2.0, 5.0, 10.0])
    risk = compute_o_field(d_pred, t_pred)

    assert risk[0] > risk[1] > risk[2] > risk[3]


def test_o_field_bounds():
    """O-field values should be in (0, 1]."""
    d_pred = torch.rand(10) * 20 + 0.1
    t_pred = torch.rand(10) * 10 + 0.1
    risk = compute_o_field(d_pred, t_pred, d_star=5.0, t_star=2.0)
    assert (risk > 0).all()
    assert (risk <= 1.0).all()


def test_closest_approach_distinguishes_approaching_and_receding():
    relative_position = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    relative_velocity = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])

    min_distance, time_to_closest = closest_approach(
        relative_position, relative_velocity
    )

    torch.testing.assert_close(min_distance, torch.tensor([0.0, 10.0]))
    torch.testing.assert_close(time_to_closest, torch.tensor([5.0, 0.0]))
    risk = compute_o_field(min_distance, time_to_closest)
    torch.testing.assert_close(
        risk,
        torch.tensor([torch.exp(torch.tensor(-2.5)), torch.exp(torch.tensor(-2.0))]),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
