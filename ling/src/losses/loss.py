"""Loss functions for STRAP: goal loss, trajectory loss, risk-scaled loss."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_nll(
    mu: torch.Tensor,          # (B, T_f, 2) predicted mean
    sigma_x: torch.Tensor,     # (B, T_f) x std
    sigma_y: torch.Tensor,     # (B, T_f) y std
    rho: torch.Tensor,         # (B, T_f) correlation
    y_gt: torch.Tensor,        # (B, T_f, 2) ground truth positions
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Negative log-likelihood of ground truth under predicted bivariate Gaussian.

    NLL = log(2*pi*sigma_x*sigma_y*sqrt(1-rho^2)) + (1/(2*(1-rho^2))) *
          [((x-mu_x)/sigma_x)^2 + ((y-mu_y)/sigma_y)^2 - 2*rho*((x-mu_x)/sigma_x)*((y-mu_y)/sigma_y)]
    """
    # Clamp for numerical stability
    sigma_x = sigma_x.clamp(min=eps)
    sigma_y = sigma_y.clamp(min=eps)
    rho = rho.clamp(-0.999, 0.999)

    dx = (mu[:, :, 0] - y_gt[:, :, 0]) / sigma_x
    dy = (mu[:, :, 1] - y_gt[:, :, 1]) / sigma_y

    # Determinant term
    z = 1 - rho ** 2
    log_det = math.log(2 * math.pi) + torch.log(sigma_x) + torch.log(sigma_y) + 0.5 * torch.log(z)

    # Quadratic term
    quad = (dx ** 2 + dy ** 2 - 2 * rho * dx * dy) / (2 * z.clamp(min=eps) + eps)

    nll = (log_det + quad).mean()
    return nll


def goal_loss(pred_goals: torch.Tensor, gt_goals: torch.Tensor) -> torch.Tensor:
    """MSE loss for surrounding vehicle goals."""
    return F.mse_loss(pred_goals, gt_goals)


def trajectory_loss(
    pred_dist: torch.Tensor,      # (B, T_f, 5) [mu_x, mu_y, sigma_x, sigma_y, rho]
    gt_positions: torch.Tensor,    # (B, T_f, 2)
) -> tuple:
    """
    Trajectory loss = MSE position + NLL.

    Returns:
        total_loss, mse_part, nll_part
    """
    pred_mu = pred_dist[:, :, :2]  # (B, T_f, 2)
    pred_sigma_x = pred_dist[:, :, 2].clamp(min=1e-6)
    pred_sigma_y = pred_dist[:, :, 3].clamp(min=1e-6)
    pred_rho = pred_dist[:, :, 4].clamp(-0.999, 0.999)

    mse = F.mse_loss(pred_mu, gt_positions)
    nll = gaussian_nll(pred_mu, pred_sigma_x, pred_sigma_y, pred_rho, gt_positions)

    return mse + nll, mse, nll


def risk_scaled_loss(
    pred_goals: torch.Tensor,
    gt_goals: torch.Tensor,
    pred_dist: torch.Tensor,
    gt_positions: torch.Tensor,
    risk_values: torch.Tensor,    # (B,) or (B, K) risk values
    beta: float = 0.0,
) -> tuple:
    """
    Risk-scaled total loss.

    gamma_risk = max(exp(R_s + R_o) - beta, 1)
    L_total = gamma_risk * (L_goal + L_traj)

    Returns:
        total_loss, gamma_risk, goal_l, traj_l
    """
    # Compute base losses
    goal_l = goal_loss(pred_goals, gt_goals)
    traj_l, _, _ = trajectory_loss(pred_dist, gt_positions)

    # Risk scaling (plan.md section 32).
    # risk_values is (B, K, 2); per-intention risk = mean over K of (Rs+Ro)/N
    # would be ideal, but risk_field stores summed risks. Normalize by
    # number of intentions and clamp for numerical stability: without
    # clamping, exp(sum) overflows (e.g. exp(38) ~ 3e16) on synthetic data.
    if risk_values.dim() > 2:
        # (B, K, 2) -> mean risk per intention, averaged over batch
        risk_sum = risk_values.sum(dim=-1).mean()
        # Normalize by typical neighbor count to keep gamma in sane range
        risk_sum = risk_sum / max(1, risk_values.shape[1] // 10)
    elif risk_values.dim() > 1:
        # Average risk across intentions
        risk_sum = risk_values.sum(dim=-1).mean()  # scalar
    else:
        risk_sum = risk_values.mean()

    risk_sum = risk_sum.clamp(max=5.0)
    gamma_risk = torch.clamp(
        torch.exp(risk_sum) - beta, min=1.0, max=10.0
    )

    total_loss = gamma_risk * (goal_l + traj_l)

    return total_loss, gamma_risk.item(), goal_l.item(), traj_l.item()


def basic_loss(
    pred_goals: torch.Tensor,
    gt_goals: torch.Tensor,
    pred_dist: torch.Tensor,
    gt_positions: torch.Tensor,
) -> tuple:
    """STRAP-B loss (no risk scaling)."""
    goal_l = goal_loss(pred_goals, gt_goals)
    traj_l, _, _ = trajectory_loss(pred_dist, gt_positions)
    return goal_l + traj_l, goal_l.item(), traj_l.item()
