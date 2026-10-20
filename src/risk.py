"""STRAP risk fields (paper Eq. 1-2) and pairwise trajectory distance."""
import torch

from src.config import (
    ALPHA_X_POLY, ALPHA_Y, BETA_1, BETA_2, DEFAULT_LENGTH, DEFAULT_WIDTH, DT,
    GAMMA_X_POLY, GAMMA_Y, T_STAR,
)


def compute_s_field(
    delta_x: torch.Tensor,
    delta_y: torch.Tensor,
    gamma_x: float = 1.0,
    gamma_y: float = 1.0,
    alpha_x: float = 2.0,
    alpha_y: float = 2.0,
) -> torch.Tensor:
    """
    Compute the S-field (subjective spatial proximity risk) between
    a target vehicle and surrounding vehicles.

    Equation (1) from STRAP paper:
        r_ij^s = exp(-|dx/gamma_x|^alpha_x - |dy/gamma_y|^alpha_y)

    Args:
        delta_x: (..., N) relative longitudinal distances
        delta_y: (..., N) relative lateral distances
        gamma_x: longitudinal scaling factor (>1)
        gamma_y: lateral scaling factor (>1)
        alpha_x: longitudinal shape factor (>=2)
        alpha_y: lateral shape factor (>=2)

    Returns:
        S-field risk values of same shape as delta_x
    """
    term_x = torch.abs(delta_x / gamma_x) ** alpha_x
    term_y = torch.abs(delta_y / gamma_y) ** alpha_y
    return torch.exp(-(term_x + term_y))


def closest_approach(
    relative_position: torch.Tensor,
    relative_velocity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return minimum future distance and its time for constant relative velocity."""
    speed_squared = relative_velocity.square().sum(dim=-1)
    time_to_closest = -(
        relative_position * relative_velocity
    ).sum(dim=-1) / speed_squared.clamp(min=1e-12)
    time_to_closest = time_to_closest.clamp(min=0.0)
    closest_position = relative_position + time_to_closest.unsqueeze(-1) * relative_velocity
    return torch.linalg.vector_norm(closest_position, dim=-1), time_to_closest


def compute_o_field(
    d_pred: torch.Tensor,
    t_pred: torch.Tensor,
    d_star: float = 5.0,
    t_star: float = 2.0,
    beta_1: float = 1.0,
    beta_2: float = 1.0,
) -> torch.Tensor:
    """
    Compute the O-field (objective future collision-related risk).

    Equation (2) from STRAP paper:
        r_ij^o = exp[-(d_m/d*)^beta_1] * exp[-(t_m/t*)^beta_2]

    Args:
        d_pred: (..., N) predicted future minimum distance between vehicles
        t_pred: (..., N) time to closest approach / gap narrowing time
        d_star: distance scaling factor
        t_star: time scaling factor
        beta_1: distance shape factor
        beta_2: time shape factor

    Returns:
        O-field risk values
    """
    d_term = torch.exp(-((d_pred / d_star) ** beta_1))
    t_term = torch.exp(-((t_pred / t_star) ** beta_2))
    return d_term * t_term


def _polyval(coefficients, x):
    out = torch.zeros_like(x)
    for c in coefficients:
        out = out * x + c
    return out


def pair_risk(
    relative_position: torch.Tensor,  # (..., 2) other - ego, [longitudinal, lateral]
    relative_velocity: torch.Tensor,  # (..., 2) other - ego
    ego_speed: torch.Tensor,          # broadcastable to (...)
    ego_dims: torch.Tensor,           # (..., 2) [length, width], broadcastable
    other_dims: torch.Tensor,         # (..., 2) [length, width]
) -> tuple[torch.Tensor, torch.Tensor]:
    """S-field and O-field with the [22] calibration (SOLUTIONS problem 4).

    S: bumper-to-bumper gaps, gamma_x(v) / alpha_x(v) cubic in ego speed.
    O: perpendicular miss distance d_m and t_m = -D.V / V.V, d* = 0.5 (w_i + w_j),
       zero when the pair is not closing.
    """
    # ponytail: ego speed (not relative speed) drives gamma_x(v); [22] wording to re-check.
    ego_speed = torch.as_tensor(ego_speed, dtype=relative_position.dtype)
    gap_x = (relative_position[..., 0].abs()
             - 0.5 * (ego_dims[..., 0] + other_dims[..., 0])).clamp(min=0)
    gap_y = (relative_position[..., 1].abs()
             - 0.5 * (ego_dims[..., 1] + other_dims[..., 1])).clamp(min=0)
    s = compute_s_field(
        gap_x, gap_y, _polyval(GAMMA_X_POLY, ego_speed), GAMMA_Y,
        _polyval(ALPHA_X_POLY, ego_speed), ALPHA_Y,
    )

    speed_squared = relative_velocity.square().sum(dim=-1)
    moving = speed_squared > 1e-12
    safe_speed_squared = torch.where(moving, speed_squared, torch.ones_like(speed_squared))
    t_m = -(relative_position * relative_velocity).sum(dim=-1) / safe_speed_squared
    cross = (relative_position[..., 0] * relative_velocity[..., 1]
             - relative_position[..., 1] * relative_velocity[..., 0])
    d_m = cross.abs() / safe_speed_squared.sqrt()
    d_star = 0.5 * (ego_dims[..., 1] + other_dims[..., 1])
    o = compute_o_field(d_m, t_m.clamp(min=0), d_star, T_STAR, BETA_1, BETA_2)
    o = torch.where(moving & (t_m > 0), o, torch.zeros_like(o))
    return s, o


def default_dims(*shape, device=None) -> torch.Tensor:
    """(*shape, 2) tensor of the fallback [length, width]."""
    return torch.tensor([DEFAULT_LENGTH, DEFAULT_WIDTH], device=device).expand(*shape, 2)


def endpoint_risk(
    ego: torch.Tensor,         # (..., 4) [x, y, vx, vy]
    others: torch.Tensor,      # (..., N, 4)
    ego_dims: torch.Tensor,    # (..., 2)
    other_dims: torch.Tensor,  # (..., N, 2)
    mask: torch.Tensor = None, # (..., N) bool
) -> torch.Tensor:
    """Summed [R_s, R_o] of one ego end point against N others -> (..., 2).

    Shared by the decoder (ego = each intention mode, others = predicted goals)
    and the loss weight (ego = true target end point, others = true goals).
    """
    s, o = pair_risk(
        others[..., :2] - ego[..., None, :2],
        others[..., 2:4] - ego[..., None, 2:4],
        ego[..., 2:4].norm(dim=-1, keepdim=True),
        ego_dims[..., None, :],
        other_dims,
    )
    if mask is not None:
        s = s.masked_fill(~mask.bool(), 0.0)
        o = o.masked_fill(~mask.bool(), 0.0)
    return torch.stack([s.sum(dim=-1), o.sum(dim=-1)], dim=-1)


def end_velocity(trajectory: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """Velocity over the last 1 s of a (..., T, 2) trajectory: (p[-1] - p[-11]) / 1 s."""
    steps = round(1.0 / dt)
    return (trajectory[..., -1, :] - trajectory[..., -1 - steps, :]) / (steps * dt)


def ground_truth_risk(gt_positions, gt_goals, goal_mask, dims) -> torch.Tensor:
    """Risk at the TRUE 5 s end point against the TRUE neighbour goals -> (B, 2).

    Labels only, so the risk-scaled loss weight has no gradient path (problem 2).
    dims: (B, N_v+1, 2), row 0 = target.
    """
    ego = torch.cat([gt_positions[:, -1], end_velocity(gt_positions)], dim=-1)
    return endpoint_risk(ego, gt_goals, dims[:, 0], dims[:, 1:], goal_mask)


def pairwise_distance(target_traj: torch.Tensor, neighbor_trajs: torch.Tensor) -> torch.Tensor:
    """Distance between target (B, T, 2) and each neighbour (B, T, N, 2) -> (B, T, N)."""
    return torch.norm(target_traj.unsqueeze(2) - neighbor_trajs, dim=-1)
