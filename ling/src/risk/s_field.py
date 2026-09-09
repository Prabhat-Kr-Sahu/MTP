"""S-field: Subjective spatial proximity risk."""
import torch
import torch.nn.functional as F


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


def compute_s_field_batch(
    positions: torch.Tensor,  # (B, T_h, N_v+1, 2) relative positions
    gamma_x: float = 1.0,
    gamma_y: float = 1.0,
    alpha_x: float = 2.0,
    alpha_y: float = 2.0,
) -> torch.Tensor:
    """
    Compute S-field for all vehicle pairs in a batch.

    Args:
        positions: relative (x, y) positions, shape (B, T_h, N_v+1, 2)
        Returns:
            s_field: (B, T_h, N_v+1) risk from target to each neighbor
    """
    # positions[:, :, 0] is the target itself -> zero risk for self
    # Compute pairwise delta_x, delta_y between target (index 0) and all others
    # positions shape: (B, T_h, N_v+1, 2)
    target_pos = positions[:, :, 0:1, :]  # (B, T_h, 1, 2)
    delta = positions - target_pos  # (B, T_h, N_v+1, 2)
    delta_x = delta[..., 0]  # (B, T_h, N_v+1)
    delta_y = delta[..., 1]  # (B, T_h, N_v+1)
    return compute_s_field(delta_x, delta_y, gamma_x, gamma_y, alpha_x, alpha_y)


class SFIELD(torch.nn.Module):
    """Learnable S-field module."""

    def __init__(self, gamma_x=1.0, gamma_y=1.0, alpha_x=2.0, alpha_y=2.0):
        super().__init__()
        self.gamma_x = torch.nn.Parameter(torch.tensor(gamma_x, dtype=torch.float32))
        self.gamma_y = torch.nn.Parameter(torch.tensor(gamma_y, dtype=torch.float32))
        self.alpha_x = torch.nn.Parameter(torch.tensor(alpha_x, dtype=torch.float32))
        self.alpha_y = torch.nn.Parameter(torch.tensor(alpha_y, dtype=torch.float32))

    def forward(self, delta_x, delta_y):
        return compute_s_field(
            delta_x, delta_y,
            float(self.gamma_x), float(self.gamma_y),
            float(self.alpha_x), float(self.alpha_y),
        )
