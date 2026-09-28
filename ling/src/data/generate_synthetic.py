"""Synthetic data generator for testing STRAP without NGSIM data."""
import torch
import numpy as np
from src.risk.s_field import compute_s_field
from src.risk.o_field import compute_o_field, closest_approach
from src.config import (
    GAMMA_X, GAMMA_Y, ALPHA_X, ALPHA_Y,
    D_STAR, T_STAR, BETA_1, BETA_2,
)


def generate_synthetic_trajectory(
    num_vehicles: int = 20,
    num_frames: int = 100,
    scene_length: float = 100.0,
    dt: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Generate a synthetic traffic scene resembling NGSIM data.

    Returns:
        dict with:
            vehicle_ids: (num_vehicles,)
            frames: (num_frames, num_vehicles, 5) [longitudinal, lateral, v_long, v_lat, a_long]
            vehicle_types: (num_vehicles,)
            vehicle_lengths: (num_vehicles,)
            vehicle_widths: (num_vehicles,)
            lane_ids: (num_vehicles,)
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    vehicle_ids = np.arange(num_vehicles)
    vehicle_types = rng.choice([0, 1, 2], size=num_vehicles)  # car, truck, bus
    vehicle_lengths = rng.uniform(3.0, 7.0, size=num_vehicles)
    vehicle_widths = rng.uniform(1.5, 2.5, size=num_vehicles)
    lane_ids = rng.choice([0, 1, 2], size=num_vehicles)

    # Generate trajectories with correlated motion
    frames = np.zeros((num_frames, num_vehicles, 5))

    # Initial positions
    x = rng.uniform(0, scene_length, size=num_vehicles)
    y = rng.uniform(-10, 10, size=num_vehicles)
    vx = rng.uniform(5, 30, size=num_vehicles)  # m/s
    vy = rng.uniform(-2, 2, size=num_vehicles)

    for t in range(num_frames):
        # Update with some correlation and noise
        ax = rng.normal(0, 0.5, size=num_vehicles)
        ay = rng.normal(0, 0.3, size=num_vehicles)
        vx += ax * dt
        vy += ay * dt
        vx = np.clip(vx, 0, 40)
        vy = np.clip(vy, -5, 5)
        x += vx * dt
        y += vy * dt

        # Wrap around scene
        x = x % scene_length

        frames[t] = np.stack([x, y, vx, vy, ax], axis=-1)

    return {
        "vehicle_ids": vehicle_ids,
        "frames": frames,
        "vehicle_types": vehicle_types,
        "vehicle_lengths": vehicle_lengths,
        "vehicle_widths": vehicle_widths,
        "lane_ids": lane_ids,
        "lane_history": np.broadcast_to(lane_ids, (num_frames, num_vehicles)).copy(),
    }


def create_sample(scene: dict, target_idx: int = 0, T_h: int = 30, T_f: int = 50):
    """
    Create a training sample from a synthetic scene.

    Returns:
        states: (T_h, N_v+1, F) state features
        target_traj: (T_f, 2) ground truth future positions
        neighbor_goals: (N_v, 4) ground truth neighbor goals
        mask: (N_v+1,) boolean mask
    """
    num_vehicles = len(scene["vehicle_ids"])
    frames = scene["frames"]

    # Target vehicle history and future
    target_hist = frames[:T_h, target_idx]  # (T_h, 4)
    target_future = frames[T_h:T_h + T_f, target_idx]  # (T_f, 4)
    target_origin = torch.from_numpy(target_hist[-1, :2].copy()).float().unsqueeze(0)

    # Relative coordinates: subtract target position from all vehicles
    all_hist = frames[:T_h]  # (T_h, num_vehicles, 4)
    target_pos = all_hist[:, target_idx:target_idx + 1, :2]  # (T_h, 1, 2)

    # Relative positions
    rel_pos = all_hist[:, :, :2] - target_pos  # (T_h, N, 2)

    # Build state features
    states = []
    for v in range(num_vehicles):
        v_rel_pos = torch.from_numpy(rel_pos[:, v]).float()  # (T_h, 2)
        v_vel = torch.from_numpy(all_hist[:, v, 2:4]).float()  # (T_h, 2)
        v_type_f = float(scene["vehicle_types"][v]) / 2.0
        lane_history = scene.get("lane_history")
        if lane_history is None:
            v_lane = np.full(T_h, scene["lane_ids"][v], dtype=np.float32)
        else:
            v_lane = np.asarray(lane_history[:T_h, v], dtype=np.float32)
        v_acc = all_hist[:, v, 4:5] if all_hist.shape[-1] > 4 else np.zeros((T_h, 1), dtype=np.float32)
        v_length_f = float(scene["vehicle_lengths"][v]) / 10.0
        v_width_f = float(scene["vehicle_widths"][v]) / 3.0

        # Simple state: relative pos + abs vel + type + lane + dimensions
        # Normalize by scene length
        feat = torch.cat([
            v_rel_pos,
            v_vel / 30.0,  # normalize velocity
            torch.from_numpy(v_acc / 10.0).float(),
            torch.full((T_h, 1), v_type_f),
            torch.from_numpy((v_lane / 2.0).reshape(T_h, 1)).float(),
            torch.full((T_h, 1), v_length_f),
            torch.full((T_h, 1), v_width_f),
        ], dim=-1)  # (T_h, 11) including the two risk features
        states.append(feat)

    # Stack: (T_h, N_v+1, F)
    states = torch.stack(states, dim=1)  # (T_h, N_v+1, F)

    # ------ Compute real S-field and O-field risk features ------
    # S-field: spatial proximity risk at each timestep
    rel_pos_t = torch.from_numpy(rel_pos).float()  # (T_h, N, 2)
    delta_x = rel_pos_t[:, :, 0]  # (T_h, N)
    delta_y = rel_pos_t[:, :, 1]  # (T_h, N)
    s_risk = compute_s_field(delta_x, delta_y, GAMMA_X, GAMMA_Y, ALPHA_X, ALPHA_Y)  # (T_h, N)
    s_risk[:, target_idx] = 0.0  # self-risk = 0

    # O-field: future collision risk approximated via velocity-based TTC
    vel_all = torch.from_numpy(all_hist[:, :, 2:4]).float()  # (T_h, N, 2)
    target_vel = vel_all[:, target_idx:target_idx + 1, :]  # (T_h, 1, 2)
    rel_vel = vel_all - target_vel  # (T_h, N, 2)
    dist = torch.norm(rel_pos_t, dim=-1).clamp(min=1e-6)  # (T_h, N)
    min_distance, time_to_closest = closest_approach(rel_pos_t, rel_vel)
    o_risk = compute_o_field(
        min_distance, time_to_closest.clamp(min=0.1),
        D_STAR, T_STAR, BETA_1, BETA_2,
    )
    o_risk[:, target_idx] = 0.0  # self-risk = 0

    risk_features = torch.stack([s_risk, o_risk], dim=-1)  # (T_h, N, 2)
    states = torch.cat([states, risk_features], dim=-1)  # (T_h, N_v+1, F+2)

    # Add batch dimension
    states = states.unsqueeze(0)  # (1, T_h, N_v+1, F)

    # Future labels share a fixed origin at the last observed target position.
    target_traj = torch.tensor(target_future[:, :2], dtype=torch.float32) - target_origin
    target_traj = target_traj.unsqueeze(0)  # (1, T_f, 2)

    # Neighbor goals (position + velocity at end of prediction horizon)
    neighbor_goals = []
    goal_mask = []
    future_goal_mask = scene.get("future_goal_mask")
    for v in range(num_vehicles):
        if v == target_idx:
            continue
        has_future_goal = (
            bool(future_goal_mask[v]) if future_goal_mask is not None
            else np.isfinite(frames[T_h + T_f - 1, v, :4]).all()
        )
        if has_future_goal:
            future_pos = frames[T_h + T_f - 1, v, :2]
            future_vel = frames[T_h + T_f - 1, v, 2:4]
            goal = torch.cat([
                torch.from_numpy(future_pos.copy()).float() - target_origin[0],
                torch.from_numpy(future_vel.copy()).float(),
            ])
        else:
            goal = torch.zeros(4, dtype=torch.float32)
        neighbor_goals.append(goal)
        goal_mask.append(has_future_goal)
    if neighbor_goals:
        neighbor_goals = torch.stack(neighbor_goals).unsqueeze(0)
        goal_mask = torch.tensor(
            [bool(valid) for valid in goal_mask], dtype=torch.bool
        ).unsqueeze(0)
    else:
        neighbor_goals = torch.zeros(1, 0, 4, dtype=torch.float32)
        goal_mask = torch.zeros(1, 0, dtype=torch.bool)

    # Mask: (1, num_vehicles) — target + neighbors present in this sample
    mask = torch.ones(1, num_vehicles, dtype=torch.bool)
    
    # Keep track of vehicle IDs for logging
    vehicle_ids = scene["vehicle_ids"]

    return states, target_traj, neighbor_goals, mask, vehicle_ids, goal_mask, target_origin
