"""Unit tests for STRAP model architecture."""
import torch
import pytest
import sys
sys.path.insert(0, 'C:/Users/prabh/OneDrive/Desktop/MTP/ling')

from src.models.motion_encoder import MotionEncoder
from src.models.spatial_encoder import SpatialEncoder
from src.models.temporal_encoder import TemporalEncoder
from src.models.goal_predictor import GoalPredictor
from src.models.risk_decoder import RiskAttentiveDecoder
from src.models.strap import STRAP


def test_motion_encoder_shape():
    """Motion encoder should produce (B, T_h, N, D) output."""
    B, T_h, N, F_in, D = 2, 30, 16, 11, 64
    model = MotionEncoder(F_in, D)
    x = torch.randn(B, T_h, N, F_in)
    out = model(x)
    assert out.shape == (B, T_h, N, D)


def test_spatial_encoder_shape():
    """Spatial encoder should preserve shape."""
    B, T_h, N, D = 2, 5, 16, 64
    model = SpatialEncoder(D, num_heads=4)
    x = torch.randn(B, T_h, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    out = model(x, mask)
    assert out.shape == (B, T_h, N, D)


def test_temporal_encoder_shape():
    """Temporal encoder should preserve shape."""
    B, T_h, N, D = 2, 5, 16, 64
    model = TemporalEncoder(D, num_heads=4)
    x = torch.randn(B, T_h, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    out = model(x, mask)
    assert out.shape == (B, T_h, N, D)


def test_goal_predictor_shape():
    """Goal predictor should output (B, N_v, 4)."""
    B, T_h, N_v, D = 2, 30, 15, 64
    model = GoalPredictor(D)
    x = torch.randn(B, T_h, N_v, D)
    out = model(x)
    assert out.shape == (B, N_v, 4)


def test_risk_decoder_output_shape():
    """Risk decoder should output (B, T_f, 5) and (B, K, 2)."""
    B, T_h, D, K = 2, 5, 64, 100
    model = RiskAttentiveDecoder(D, num_heads=4, k_intentions=K, num_decoder_layers=2)
    target_enc = torch.randn(B, T_h, D)
    neighbor_goals = torch.randn(B, 15, 4)
    neighbor_risk = torch.randn(B, 15, 2)
    mask = torch.ones(B, 15, dtype=torch.bool)

    traj_dist, risk_field = model(target_enc, neighbor_goals, neighbor_risk, mask)
    assert traj_dist.shape == (B, 50, 5)  # (B, T_f=50, 5)
    assert risk_field.shape == (B, K, 2)


def test_risk_decoder_ignores_padded_neighbor_goals():
    torch.manual_seed(7)
    model = RiskAttentiveDecoder(
        d_model=64, num_heads=4, k_intentions=8, num_decoder_layers=1, dropout=0
    ).eval()
    target_enc = torch.randn(1, 4, 64)
    neighbor_goals = torch.randn(1, 2, 4)
    neighbor_risk = torch.randn(1, 2, 2)
    mask = torch.tensor([[True, False]])

    baseline_traj, baseline_risk = model(
        target_enc, neighbor_goals, neighbor_risk, mask
    )
    changed_goals = neighbor_goals.clone()
    changed_goals[:, 1] = 10000
    changed_traj, changed_risk = model(
        target_enc, changed_goals, neighbor_risk, mask
    )

    torch.testing.assert_close(baseline_risk, changed_risk)
    torch.testing.assert_close(baseline_traj, changed_traj)


def test_strap_full_forward():
    """Full STRAP model forward pass."""
    B, T_h, N, F_in = 2, 30, 16, 11
    model = STRAP(input_dim=F_in)
    states = torch.randn(B, 30, N, F_in)
    mask = torch.ones(B, N, dtype=torch.bool)

    traj_dist, goals, risk = model(states, mask)
    assert traj_dist.shape == (B, 50, 5)
    assert goals.shape == (B, 15, 4)
    assert risk.shape == (B, 100, 2)


def test_gaussian_output_constraints():
    """Output distribution should satisfy sigma > 0 and |rho| < 1."""
    B, T_f = 2, 50
    model = STRAP(input_dim=11)
    states = torch.randn(B, 30, 16, 11)
    mask = torch.ones(B, 16, dtype=torch.bool)

    with torch.no_grad():
        traj_dist, _, _ = model(states, mask)

    sigma_x = traj_dist[:, :, 2]
    sigma_y = traj_dist[:, :, 3]
    rho = traj_dist[:, :, 4]

    assert (sigma_x > 0).all()
    assert (sigma_y > 0).all()
    assert (rho > -1).all() and (rho < 1).all()


def test_loss_computation():
    """Loss functions should produce finite values."""
    from src.losses.loss import goal_loss, trajectory_loss, basic_loss

    B, T_f = 2, 50
    pred_goals = torch.randn(B, 15, 4)
    gt_goals = torch.randn(B, 15, 4)
    pred_dist = torch.randn(B, T_f, 5)
    pred_dist[:, :, 2] = torch.abs(pred_dist[:, :, 2]) + 0.1  # sigma > 0
    pred_dist[:, :, 3] = torch.abs(pred_dist[:, :, 3]) + 0.1
    pred_dist[:, :, 4] = torch.tanh(pred_dist[:, :, 4])  # rho in (-1, 1)
    gt_positions = torch.randn(B, T_f, 2)

    loss, gl, tl = basic_loss(pred_goals, gt_goals, pred_dist, gt_positions)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_risk_scaled_loss():
    """Risk-scaled loss should produce finite values."""
    from src.losses.loss import risk_scaled_loss

    B = 2
    pred_goals = torch.randn(B, 15, 4)
    gt_goals = torch.randn(B, 15, 4)
    pred_dist = torch.randn(B, 50, 5)
    pred_dist[:, :, 2] = torch.abs(pred_dist[:, :, 2]) + 0.1
    pred_dist[:, :, 3] = torch.abs(pred_dist[:, :, 3]) + 0.1
    pred_dist[:, :, 4] = torch.tanh(pred_dist[:, :, 4])
    gt_positions = torch.randn(B, 50, 2)
    risk_field = torch.randn(B, 100, 2)

    loss, gamma, gl, tl = risk_scaled_loss(
        pred_goals, gt_goals, pred_dist, gt_positions, risk_field
    )
    assert torch.isfinite(loss)
    assert gamma >= 1.0


def test_goal_loss_ignores_masked_neighbors():
    from src.losses.loss import goal_loss

    target = torch.zeros(1, 2, 4)
    mask = torch.tensor([[True, False]])
    prediction = torch.ones(1, 2, 4)
    baseline = goal_loss(prediction, target, goal_mask=mask)
    prediction[:, 1] = 1000
    torch.testing.assert_close(goal_loss(prediction, target, goal_mask=mask), baseline)


def test_risk_scaled_loss_uses_per_sample_weight():
    from src.losses.loss import basic_loss, risk_scaled_loss

    pred_goals = torch.ones(2, 1, 4)
    gt_goals = torch.zeros_like(pred_goals)
    pred_dist = torch.zeros(2, 50, 5)
    pred_dist[:, :, 2:4] = 1.0
    gt_positions = torch.zeros(2, 50, 2)
    goal_mask = torch.ones(2, 1, dtype=torch.bool)
    risk = torch.tensor([[[0.0, 0.0]], [[torch.log(torch.tensor(2.0)), 0.0]]])

    base, _, _ = basic_loss(
        pred_goals, gt_goals, pred_dist, gt_positions, goal_mask=goal_mask
    )
    weighted, gamma_mean, _, _ = risk_scaled_loss(
        pred_goals, gt_goals, pred_dist, gt_positions, risk, goal_mask=goal_mask
    )

    torch.testing.assert_close(gamma_mean, torch.tensor(1.5).item())
    torch.testing.assert_close(weighted, 1.5 * base)


def test_risk_scaled_loss_no_overflow_for_large_risk():
    """risk_scaled_loss must produce finite gamma even when risk_sum >> 88."""
    from src.losses.loss import risk_scaled_loss

    B = 2
    pred_goals = torch.randn(B, 1, 4)
    gt_goals = torch.zeros_like(pred_goals)
    pred_dist = torch.zeros(B, 50, 5)
    pred_dist[:, :, 2:4] = 1.0
    gt_positions = torch.zeros(B, 50, 2)
    # Very large risk values that would overflow exp() without the fix
    risk = torch.tensor([[[100.0, 0.0]], [[500.0, 0.0]]])

    loss, gamma, _, _ = risk_scaled_loss(
        pred_goals, gt_goals, pred_dist, gt_positions, risk
    )
    assert float(loss) < float('inf'), f"Loss is not finite: {loss}"
    assert float(gamma) >= 1.0, f"Gamma should be >= 1.0, got {gamma}"
    assert float(gamma) <= 1e4, f"Gamma should be capped at 1e4, got {gamma}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
