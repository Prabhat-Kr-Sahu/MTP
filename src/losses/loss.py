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
    reduction: str = "mean",
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

    nll = (log_det + quad).mean(dim=-1)
    return nll.mean() if reduction == "mean" else nll


def goal_loss(
    pred_goals: torch.Tensor,
    gt_goals: torch.Tensor,
    goal_mask: torch.Tensor = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE loss for surrounding vehicle goals."""
    squared_error = (pred_goals - gt_goals).square()
    if goal_mask is None:
        per_sample = squared_error.mean(dim=(1, 2))
    else:
        weights = goal_mask.to(dtype=squared_error.dtype).unsqueeze(-1)
        denominator = (weights.sum(dim=(1, 2)) * pred_goals.shape[-1]).clamp(min=1)
        per_sample = (squared_error * weights).sum(dim=(1, 2)) / denominator
    return per_sample.mean() if reduction == "mean" else per_sample


def trajectory_loss(
    pred_dist: torch.Tensor,      # (B, T_f, 5) [mu_x, mu_y, sigma_x, sigma_y, rho]
    gt_positions: torch.Tensor,    # (B, T_f, 2)
    reduction: str = "mean",
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

    mse = (pred_mu - gt_positions).square().sum(dim=-1).mean(dim=-1)
    nll = gaussian_nll(
        pred_mu, pred_sigma_x, pred_sigma_y, pred_rho, gt_positions,
        reduction="none",
    )

    total = mse + nll
    if reduction == "mean":
        return total.mean(), mse.mean(), nll.mean()
    return total, mse, nll


def risk_scaled_loss(
    pred_goals: torch.Tensor,
    gt_goals: torch.Tensor,
    pred_dist: torch.Tensor,
    gt_positions: torch.Tensor,
    risk_values: torch.Tensor,    # (B,) or (B, K) risk values
    beta: float = 0.0,
    goal_mask: torch.Tensor = None,
) -> tuple:
    """
    Risk-scaled total loss.

    gamma_risk = max(exp(R_s + R_o) - beta, 1)
    L_total = gamma_risk * (L_goal + L_traj)

    Returns:
        total_loss, gamma_risk, goal_l, traj_l
    """
    # Compute base losses
    goal_per_sample = goal_loss(
        pred_goals, gt_goals, goal_mask=goal_mask, reduction="none"
    )
    traj_per_sample, _, _ = trajectory_loss(
        pred_dist, gt_positions, reduction="none"
    )

    # Per-sample gamma = max(exp(Rs + Ro) - beta, 1).
    # Clamp the exponent BEFORE exp() to avoid float32 overflow:
    # exp(88.7) ≈ 1.4e38 < 3.4e38 (float32 max), exp(89) would overflow.
    # Also clamp the final gamma_risk to prevent inf*0=NaN when base loss is 0.
    if risk_values.dim() > 2:
        risk_sum = risk_values.sum(dim=-1).mean(dim=-1)
    elif risk_values.dim() > 1:
        risk_sum = risk_values.sum(dim=-1)
    else:
        risk_sum = risk_values
    exp_arg = torch.clamp(risk_sum - beta, max=88.7)
    gamma_risk = torch.exp(exp_arg)
    gamma_risk = torch.clamp(gamma_risk, min=1.0, max=1e4)

    total_loss = (gamma_risk * (goal_per_sample + traj_per_sample)).mean()

    return (
        total_loss,
        gamma_risk.mean().item(),
        goal_per_sample.mean().item(),
        traj_per_sample.mean().item(),
    )


def basic_loss(
    pred_goals: torch.Tensor,
    gt_goals: torch.Tensor,
    pred_dist: torch.Tensor,
    gt_positions: torch.Tensor,
    goal_mask: torch.Tensor = None,
) -> tuple:
    """STRAP-B loss (no risk scaling)."""
    goal_l = goal_loss(pred_goals, gt_goals, goal_mask=goal_mask)
    traj_l, _, _ = trajectory_loss(pred_dist, gt_positions)
    return goal_l + traj_l, goal_l.item(), traj_l.item()
