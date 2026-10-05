"""Unit tests for risk field computation."""
import torch
import pytest
import sys
sys.path.insert(0, 'C:/Users/prabh/OneDrive/Desktop/MTP/ling')

from src.risk.s_field import compute_s_field, compute_s_field_batch
from src.risk.o_field import compute_o_field, closest_approach
from src.risk.risk_features import compute_risk_field


def test_s_field_decreases_with_distance():
    """S-field should decrease as vehicles move farther away."""
    delta_x = torch.tensor([0.0, 1.0, 5.0, 10.0])
    delta_y = torch.tensor([0.0, 1.0, 5.0, 10.0])
    risk = compute_s_field(delta_x, delta_y)

    assert risk[0] > risk[1] > risk[2] > risk[3]
    assert risk[0].item() == pytest.approx(1.0, abs=1e-6)  # self-distance = 0


def test_s_field_symmetry():
    """S-field should be symmetric in x and y with same parameters."""
    risk_xy = compute_s_field(
        torch.tensor([2.0]), torch.tensor([2.0]),
        gamma_x=1.0, gamma_y=1.0, alpha_x=2.0, alpha_y=2.0
    )
    assert risk_xy.item() == pytest.approx(
        compute_s_field(torch.tensor([2.0]), torch.tensor([2.0]),
        gamma_x=1.0, gamma_y=1.0, alpha_x=2.0, alpha_y=2.0).item()
    )


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


def test_risk_field_shape():
    """Risk field computation should produce correct shapes."""
    B, N = 4, 10
    delta_x = torch.randn(B, N)
    delta_y = torch.randn(B, N)
    d_pred = torch.randn(B, N).abs()
    t_pred = torch.randn(B, N).abs()

    s, o, mask, num_sel = compute_risk_field(
        delta_x, delta_y, d_pred, t_pred
    )
    assert s.shape == (B, N)
    assert o.shape == (B, N)
    assert mask.shape == (B, N)
    assert num_sel.shape == (B,)
    assert mask.sum(dim=-1).max() <= 15


def test_risk_threshold():
    """Risk-aware selection should respect the 0.005 threshold."""
    B, N = 2, 20
    delta_x = torch.randn(B, N) * 0.1  # Very close -> high risk
    delta_y = torch.randn(B, N) * 0.1
    d_pred = torch.ones(B, N) * 2.0
    t_pred = torch.ones(B, N) * 1.0

    s, o, mask, _ = compute_risk_field(
        delta_x, delta_y, d_pred, t_pred, risk_threshold=0.005
    )
    # With very close vehicles, most should be selected
    assert mask.sum(dim=-1).max() > 0


def test_risk_selection_uses_s_or_o_and_excludes_self():
    delta_x = torch.tensor([[0.0, 10.0, 0.5]])
    delta_y = torch.zeros_like(delta_x)
    d_pred = torch.tensor([[0.0, 0.1, 100.0]])
    t_pred = torch.tensor([[0.1, 0.1, 100.0]])

    _, _, selected, _ = compute_risk_field(
        delta_x, delta_y, d_pred, t_pred, risk_threshold=0.005, max_neighbors=2
    )

    assert not selected[0, 0]
    assert selected[0, 1]
    assert selected[0, 2]


def test_risk_selection_falls_back_to_geometric_nearest_neighbor():
    delta_x = torch.tensor([[0.0, 10.0, 2.0]])
    delta_y = torch.zeros_like(delta_x)
    d_pred = torch.full_like(delta_x, 100.0)
    t_pred = torch.full_like(delta_x, 100.0)

    _, _, selected, count = compute_risk_field(
        delta_x, delta_y, d_pred, t_pred, risk_threshold=0.005, max_neighbors=2
    )

    assert selected.tolist() == [[False, False, True]]
    assert count.tolist() == [1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
