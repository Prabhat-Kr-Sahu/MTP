"""S-field: Subjective spatial proximity risk."""
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
