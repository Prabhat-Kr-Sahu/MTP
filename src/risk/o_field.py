"""O-field: Objective future collision-related risk."""
import torch


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
