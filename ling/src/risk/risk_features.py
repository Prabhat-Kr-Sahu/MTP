"""Risk feature computation and neighborhood selection."""
import torch
from src.risk.s_field import compute_s_field
from src.risk.o_field import compute_o_field, closest_approach


def compute_risk_field(
    delta_x: torch.Tensor,
    delta_y: torch.Tensor,
    d_pred: torch.Tensor,
    t_pred: torch.Tensor,
    risk_threshold: float = 0.005,
    max_neighbors: int = 15,
    s_gamma_x: float = 1.0,
    s_gamma_y: float = 1.0,
    s_alpha_x: float = 2.0,
    s_alpha_y: float = 2.0,
    o_d_star: float = 5.0,
    o_t_star: float = 2.0,
    o_beta_1: float = 1.0,
    o_beta_2: float = 1.0,
) -> tuple:
    """
    Compute S-field, O-field, and perform risk-aware neighborhood selection.

    Note: delta_x corresponds to frames_data index 0 (LONGITUDINAL / local_y),
    delta_y corresponds to index 1 (LATERAL / local_x). See _build_frame_matrix.

    Args:
        delta_x: (B, N) relative LONGITUDINAL distances to target (frames_data[:, :, 0])
        delta_y: (B, N) relative LATERAL distances to target (frames_data[:, :, 1])
        d_pred: (B, N) predicted minimum distances
        t_pred: (B, N) predicted times to closest approach
        risk_threshold: vehicles with S or O risk > this are considered interacting
        max_neighbors: maximum number of neighbors to select

    Returns:
        s_field: (B, N) subjective risk
        o_field: (B, N) objective risk
        selected_mask: (B, N) boolean mask of selected neighbors
        num_selected: (B,) number of selected neighbors per sample
    """
    s_field = compute_s_field(delta_x, delta_y, s_gamma_x, s_gamma_y, s_alpha_x, s_alpha_y)
    o_field = compute_o_field(d_pred, t_pred, o_d_star, o_t_star, o_beta_1, o_beta_2)

    interacting = (s_field > risk_threshold) | (o_field > risk_threshold)
    interacting[:, 0] = False
    combined_risk = (s_field + o_field).masked_fill(~interacting, -torch.inf)
    B, N = delta_x.shape
    selected_mask = torch.zeros(B, N, dtype=torch.bool, device=delta_x.device)
    num_select = min(max_neighbors, max(0, N - 1))
    if num_select:
        selected_values, selected_idx = torch.topk(combined_risk, k=num_select, dim=-1)
        selected_mask.scatter_(1, selected_idx, torch.isfinite(selected_values))
        distances = torch.sqrt(delta_x[:, 1:].square() + delta_y[:, 1:].square())
        for batch_idx in range(B):
            if not selected_mask[batch_idx].any():
                selected_mask[batch_idx, distances[batch_idx].argmin() + 1] = True

    num_selected = selected_mask.sum(dim=-1)

    return s_field, o_field, selected_mask, num_selected


def build_risk_features(
    positions: torch.Tensor,  # (B, T_h, N_v+1, 2) in frames_data layout: index 0 = longitudinal (local_y), index 1 = lateral (local_x)
    velocities: torch.Tensor,  # (B, T_h, N_v+1, 2) [v_long, v_lat]
    accelerations: torch.Tensor,  # (B, T_h, N_v+1, 2) [a_long, a_lat]
    dimensions: torch.Tensor,  # (B, T_h, N_v+1, 2) length, width
    vehicle_types: torch.Tensor,  # (B, T_h, N_v+1)
    lane_ids: torch.Tensor,  # (B, T_h, N_v+1)
    s_gamma_x: float = 1.0,
    s_gamma_y: float = 1.0,
    s_alpha_x: float = 2.0,
    s_alpha_y: float = 2.0,
) -> tuple:
    """
    Build complete risk features for all vehicles in the scene.

    Note: This function is currently unused (dead code). Positions are expected
    in frames_data layout: index 0 = longitudinal (local_y), index 1 = lateral (local_x).

    Returns:
        risk_features: (B, T_h, N_v+1, 2) S-field and O-field per vehicle
        s_field: (B, T_h, N_v+1) subjective risk
        o_field: (B, T_h, N_v+1) objective risk
    """
    # Compute S-field using target-relative positions
    # Returns (B, T_h, N_v+1) with self-risk = 1 at index 0; zero it out.
    B, T_h = positions.shape[0], positions.shape[1]
    s_field_all = compute_s_field_batch(positions, s_gamma_x, s_gamma_y, s_alpha_x, s_alpha_y)
    s_field_all[:, :, 0] = 0.0

    rel_vel = velocities[:, :, 1:, :] - velocities[:, :, 0:1, :]  # (B, T_h, N_v, 2)
    rel_pos = positions[:, :, 1:, :] - positions[:, :, 0:1, :]  # (B, T_h, N_v, 2)
    min_distance, time_to_closest = closest_approach(rel_pos, rel_vel)
    o_field_neigh = compute_o_field(min_distance, time_to_closest.clamp(min=0.1))

    # Assemble full (B, T_h, N_v+1) tensors with zero self-risk.
    s_field_padded = s_field_all
    o_field_padded = torch.cat(
        [torch.zeros(B, T_h, 1, device=positions.device), o_field_neigh], dim=-1
    )

    risk_features = torch.stack([s_field_padded, o_field_padded], dim=-1)  # (B, T_h, N_v+1, 2)

    return risk_features, s_field_padded, o_field_padded


def compute_s_field_batch(positions, gamma_x, gamma_y, alpha_x, alpha_y):
    """Batch S-field computation."""
    delta = positions - positions[:, :, 0:1, :]  # relative to target
    delta_x = delta[..., 0]
    delta_y = delta[..., 1]
    return compute_s_field(delta_x, delta_y, gamma_x, gamma_y, alpha_x, alpha_y)
