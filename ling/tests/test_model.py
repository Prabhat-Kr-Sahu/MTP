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
    B, T_h, N, F_in, D = 2, 30, 16, 10, 64
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
    B, T_h, N_v, D = 2, 5, 15, 64
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


def test_strap_full_forward():
    """Full STRAP model forward pass."""
    B, T_h, N, F_in = 2, 5, 16, 10
    model = STRAP(input_dim=F_in)
    states = torch.randn(B, T_h, N, F_in)
    mask = torch.ones(B, N, dtype=torch.bool)

    traj_dist, goals, risk = model(states, mask)
    assert traj_dist.shape == (B, 50, 5)
    assert goals.shape == (B, 15, 4)
    assert risk.shape == (B, 100, 2)


def test_gaussian_output_constraints():
    """Output distribution should satisfy sigma > 0 and |rho| < 1."""
    B, T_f = 2, 50
    model = STRAP(input_dim=10)
    states = torch.randn(B, 5, 16, 10)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
