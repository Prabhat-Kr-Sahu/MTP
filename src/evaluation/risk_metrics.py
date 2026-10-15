"""Trajectory-based risk metrics for collision evaluation.

Each metric maps a target trajectory (B, T_f, 2) and neighbour trajectories
(B, T_f, N_v, 2) to one value per neighbour, shape (B, N_v).
"""
import torch

from src.risk import compute_o_field, compute_s_field, pairwise_distance


def _velocity(traj, dt):
    """Finite-difference velocity along dim 1; the first step is zero."""
    vel = torch.zeros_like(traj)
    vel[:, 1:] = (traj[:, 1:] - traj[:, :-1]) / dt
    return vel


def min_dist(target, neighbors, dt=0.1, threshold=2.0):
    """Minimum Distance (MD) over the horizon."""
    return pairwise_distance(target, neighbors).min(dim=1).values


def ttc(target, neighbors, dt=0.1, threshold=2.0):
    """Time-To-Collision: first time distance < threshold; inf if never."""
    below = pairwise_distance(target, neighbors) < threshold
    first = below.float().argmax(dim=1).float() * dt
    return torch.where(below.any(dim=1), first, torch.full_like(first, float("inf")))


def pet(target, neighbors, dt=0.1, threshold=2.0, spatial_threshold=2.0):
    """Post-Encroachment Time: smallest time gap between the neighbour being at
    a point and the target passing within spatial_threshold of it; inf if never."""
    # d[b, n, t_n, t_t] = |neighbour(t_n) - target(t_t)|
    d = torch.norm(
        neighbors.permute(0, 2, 1, 3).unsqueeze(3) - target[:, None, None], dim=-1
    )
    d_min, t_t = d.min(dim=3)
    t_n = torch.arange(d.shape[2], device=d.device)
    gap = (t_n - t_t).abs().float() * dt
    gap = torch.where(d_min < spatial_threshold, gap, torch.full_like(gap, float("inf")))
    return gap.min(dim=2).values


def thw(target, neighbors, dt=0.1, threshold=2.0):
    """Time Headway: min over horizon of distance / target speed."""
    vel = _velocity(target, dt)
    vel[:, 0] = vel[:, 1]
    speed = vel.norm(dim=-1).clamp(min=0.1)
    return (pairwise_distance(target, neighbors) / speed.unsqueeze(2)).min(dim=1).values


def tet(target, neighbors, dt=0.1, threshold=2.0):
    """Time Exposed TTC: total time spent closer than threshold."""
    return (pairwise_distance(target, neighbors) < threshold).float().sum(dim=1) * dt


def tit(target, neighbors, dt=0.1, threshold=2.0):
    """Time Integrated TTC: integral of (threshold - distance) while below it."""
    return (threshold - pairwise_distance(target, neighbors)).clamp(min=0).sum(dim=1) * dt


def drac(target, neighbors, dt=0.1, threshold=2.0):
    """Deceleration Rate to Avoid Collision: max of closing_speed^2 / (2 d)."""
    dist = pairwise_distance(target, neighbors).clamp(min=0.01)
    rel_vel = _velocity(target, dt).unsqueeze(2) - _velocity(neighbors, dt)
    direction = (target.unsqueeze(2) - neighbors) / dist.unsqueeze(-1)
    closing = (-(rel_vel * direction).sum(dim=-1)).clamp(min=0)
    return (closing ** 2 / (2 * dist)).max(dim=1).values


def rd(target, neighbors, dt=0.1, threshold=2.0):
    """Required Deceleration: max of target_speed^2 / (2 d)."""
    dist = pairwise_distance(target, neighbors).clamp(min=0.01)
    speed = _velocity(target, dt).norm(dim=-1)
    return (speed.unsqueeze(2) ** 2 / (2 * dist)).max(dim=1).values


def tmd(target, neighbors, dt=0.1, threshold=2.0):
    """Time to Minimum Distance."""
    return pairwise_distance(target, neighbors).min(dim=1).indices.float() * dt


def s_field(target, neighbors, dt=0.1, threshold=2.0):
    """Max STRAP S-field over the horizon (default gamma=1, alpha=2)."""
    delta = neighbors - target.unsqueeze(2)
    return compute_s_field(delta[..., 0], delta[..., 1]).max(dim=1).values


def o_field(target, neighbors, dt=0.1, threshold=2.0):
    """STRAP O-field from min distance and its time (default d*=5, t*=2)."""
    d_min, t_idx = pairwise_distance(target, neighbors).min(dim=1)
    return compute_o_field(d_min, t_idx.float() * dt)


# name -> (function, higher_is_riskier, default_threshold); order = report row order
METRICS = {
    "min_dist": (min_dist, False, 2.0),
    "ttc": (ttc, False, 2.0),
    "pet": (pet, False, 2.0),
    "thw": (thw, False, 2.0),
    "tet": (tet, True, 0.5),
    "tit": (tit, True, 0.5),
    "drac": (drac, True, 3.0),
    "rd": (rd, True, 3.0),
    "tmd": (tmd, False, 2.0),
    "s_field": (s_field, True, 0.005),
    "o_field": (o_field, True, 0.005),
}


def is_collision(name, value, threshold=None):
    """True if `value` of metric `name` crosses its (or the given) threshold."""
    _, higher_is_riskier, default = METRICS[name]
    t = default if threshold is None else threshold
    return value > t if higher_is_riskier else value < t
