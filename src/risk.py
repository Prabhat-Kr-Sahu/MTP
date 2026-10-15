"""STRAP risk fields (paper Eq. 1-2) and pairwise trajectory distance."""
import torch


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


def pairwise_distance(target_traj: torch.Tensor, neighbor_trajs: torch.Tensor) -> torch.Tensor:
    """Distance between target (B, T, 2) and each neighbour (B, T, N, 2) -> (B, T, N)."""
    return torch.norm(target_traj.unsqueeze(2) - neighbor_trajs, dim=-1)
