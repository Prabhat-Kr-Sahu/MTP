"""O-field: Objective future collision-related risk."""
import torch
import torch.nn.functional as F


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


def compute_o_field_from_trajectories(
    target_traj: torch.Tensor,    # (B, T_f, 2) predicted target positions
    neighbor_traj: torch.Tensor,  # (B, T_f, N_v, 2) predicted neighbor positions
    d_star: float = 5.0,
    t_star: float = 2.0,
    beta_1: float = 1.0,
    beta_2: float = 1.0,
) -> torch.Tensor:
    """
    Compute O-field from predicted future trajectories.

    Args:
        target_traj: (B, T_f, 2) target vehicle future positions
        neighbor_traj: (B, T_f, N_v, 2) neighbor vehicle future positions
        Returns:
            o_field: (B, T_f, N_v) O-field risk per future timestep per neighbor
    """
    B, T_f, _ = target_traj.shape
    N_v = neighbor_traj.shape[2]

    # target: (B, T_f, 2) -> (B, T_f, 1, 2); neighbor: (B, T_f, N_v, 2)
    # Broadcast directly: distance per timestep per neighbor.
    target_expanded = target_traj.unsqueeze(2)  # (B, T_f, 1, 2)

    # Distance at each future timestep
    dist = torch.norm(target_expanded - neighbor_traj, dim=-1)  # (B, T_f, N_v)

    # Find minimum distance and its index (time of closest approach)
    d_min, _ = torch.min(dist, dim=1)  # (B, N_v)
    t_min_idx = torch.argmin(dist, dim=1).float()  # (B, N_v)

    # Normalize time index to [0, T_f-1] -> approximate time to closest approach
    t_min = t_min_idx * (T_f * 0.1)  # scale to seconds (assuming DT=0.1)

    return compute_o_field(d_min, t_min, d_star, t_star, beta_1, beta_2)


class OFIELD(torch.nn.Module):
    """Learnable O-field module."""

    def __init__(self, d_star=5.0, t_star=2.0, beta_1=1.0, beta_2=1.0):
        super().__init__()
        self.d_star = torch.nn.Parameter(torch.tensor(d_star, dtype=torch.float32))
        self.t_star = torch.nn.Parameter(torch.tensor(t_star, dtype=torch.float32))
        self.beta_1 = torch.nn.Parameter(torch.tensor(beta_1, dtype=torch.float32))
        self.beta_2 = torch.nn.Parameter(torch.tensor(beta_2, dtype=torch.float32))

    def forward(self, d_pred, t_pred):
        return compute_o_field(
            d_pred, t_pred,
            float(self.d_star), float(self.t_star),
            float(self.beta_1), float(self.beta_2),
        )
