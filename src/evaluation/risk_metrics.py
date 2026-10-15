"""Extensible risk metric evaluation framework.

Supported metrics:
  - min_dist: Minimum Distance (MD)
  - ttc: Time to Collision (TTC)
  - pet: Post-Encroachment Time (PET)
  - thw: Time Headway (THW)
  - tet: Time Exposed TTC (TET)
  - tit: Time Integrated TTC (TIT)
  - drac: Deceleration Rate to Avoid Collision (DRAC)
  - rd: Required Deceleration (RD)
  - tmd: Time to Minimum Distance (TMD)
  - s_field: STRAP Spatial Risk Field
  - o_field: STRAP Objective Risk Field
"""
import torch
import math
from abc import ABC, abstractmethod


class RiskMetricRegistry:
    """Registry for all risk metrics."""
    _metrics = {}

    @classmethod
    def register(cls, name):
        def wrapper(metric_class):
            cls._metrics[name] = metric_class
            return metric_class
        return wrapper

    @classmethod
    def get(cls, name):
        if name not in cls._metrics:
            available = list(cls._metrics.keys())
            raise ValueError(f"Risk metric '{name}' not found. Available: {available}")
        return cls._metrics[name]()

    @classmethod
    def all_names(cls):
        return list(cls._metrics.keys())


class RiskMetric(ABC):
    """Abstract base class for trajectory-based risk metrics."""

    # Subclasses should set these:
    #   higher_is_riskier = True  -> collision if value > threshold
    #   higher_is_riskier = False -> collision if value < threshold
    higher_is_riskier = False
    default_threshold = 2.0

    @abstractmethod
    def compute(self, target_traj, neighbor_trajs, dt=0.1, **kwargs):
        """
        Compute the risk metric.

        Args:
            target_traj: (B, T_f, 2) predicted target trajectory
            neighbor_trajs: (B, T_f, N_v, 2) predicted neighbor trajectories
            dt: Timestep duration (seconds)

        Returns:
            risk_values: (B, N_v) Tensor of computed risk values per neighbor
        """
        pass

    def is_collision(self, value, threshold=None):
        """Check if a risk value indicates a collision."""
        t = threshold if threshold is not None else self.default_threshold
        if self.higher_is_riskier:
            return value > t
        else:
            return value < t


# ---------------------------------------------------------------------------
# Helper: compute pairwise distances over time
# ---------------------------------------------------------------------------

def _pairwise_dist(target_traj, neighbor_trajs):
    """Compute (B, T_f, N_v) pairwise distance between target and neighbors."""
    target_exp = target_traj.unsqueeze(2)  # (B, T_f, 1, 2)
    return torch.norm(target_exp - neighbor_trajs, dim=-1)  # (B, T_f, N_v)


# ============================= METRICS =====================================


@RiskMetricRegistry.register("min_dist")
class MinimumDistanceMetric(RiskMetric):
    """Minimum Distance (MD): absolute minimum distance over the horizon."""
    higher_is_riskier = False
    default_threshold = 2.0

    def compute(self, target_traj, neighbor_trajs, dt=0.1, **kwargs):
        dist = _pairwise_dist(target_traj, neighbor_trajs)
        min_dist, _ = dist.min(dim=1)
        return min_dist


@RiskMetricRegistry.register("ttc")
class TimeToCollisionMetric(RiskMetric):
    """Time-To-Collision (TTC): time at which distance first drops below
    collision_threshold. Returns inf if no collision."""
    higher_is_riskier = False
    default_threshold = 2.0

    def compute(self, target_traj, neighbor_trajs, dt=0.1, collision_threshold=2.0, **kwargs):
        dist = _pairwise_dist(target_traj, neighbor_trajs)
        # Find the first timestep where distance < collision_threshold
        below = dist < collision_threshold  # (B, T_f, N_v)
        # Use argmax on the boolean to find first True; if none, argmax returns 0
        any_below = below.any(dim=1)  # (B, N_v)
        first_idx = below.float().argmax(dim=1).float()  # (B, N_v)
        ttc = torch.where(
            any_below,
            first_idx * dt,
            torch.full_like(first_idx, float('inf'))
        )
        return ttc


@RiskMetricRegistry.register("pet")
class PostEncroachmentTimeMetric(RiskMetric):
    """Post-Encroachment Time (PET): time difference between target leaving
    and neighbor arriving at the same spatial point.

    Approximated as: for each neighbor, find the timestep where the target
    is closest to the neighbor's position at any timestep, then measure the
    time gap. Lower PET = higher risk."""
    higher_is_riskier = False
    default_threshold = 2.0

    def compute(self, target_traj, neighbor_trajs, dt=0.1, spatial_threshold=2.0, **kwargs):
        B, T_f, N_v, _ = neighbor_trajs.shape

        pet_values = torch.full((B, N_v), float('inf'))

        for b in range(B):
            for n in range(N_v):
                min_pet = float('inf')
                # For each timestep t_n where the neighbor is at a position,
                # find the closest timestep t_t where the target is near that position
                for t_n in range(T_f):
                    n_pos = neighbor_trajs[b, t_n, n]  # (2,)
                    # Distance from target at all timesteps to this neighbor position
                    d = torch.norm(target_traj[b, :, :] - n_pos.unsqueeze(0), dim=-1)  # (T_f,)
                    d_min, t_t = d.min(dim=0)
                    if d_min.item() < spatial_threshold:
                        time_gap = abs(t_n - t_t.item()) * dt
                        if time_gap < min_pet:
                            min_pet = time_gap
                pet_values[b, n] = min_pet

        return pet_values


@RiskMetricRegistry.register("thw")
class TimeHeadwayMetric(RiskMetric):
    """Time Headway (THW): longitudinal distance / speed of following vehicle.
    Computed at each timestep, returns the minimum over the horizon.
    Lower THW = higher risk."""
    higher_is_riskier = False
    default_threshold = 2.0

    def compute(self, target_traj, neighbor_trajs, dt=0.1, **kwargs):
        B, T_f, N_v, _ = neighbor_trajs.shape

        # Compute longitudinal distance (y-component) and velocity
        # Target velocity approximated from consecutive positions
        target_vel = torch.zeros_like(target_traj)
        target_vel[:, 1:] = (target_traj[:, 1:] - target_traj[:, :-1]) / dt
        target_vel[:, 0] = target_vel[:, 1]  # copy first
        target_speed = torch.norm(target_vel, dim=-1).clamp(min=0.1)  # (B, T_f)

        dist = _pairwise_dist(target_traj, neighbor_trajs)  # (B, T_f, N_v)
        thw = dist / target_speed.unsqueeze(2)  # (B, T_f, N_v)
        min_thw, _ = thw.min(dim=1)
        return min_thw


@RiskMetricRegistry.register("tet")
class TimeExposedTTCMetric(RiskMetric):
    """Time Exposed TTC (TET): total time during which distance is below
    collision_threshold. Higher TET = more time spent in danger.
    Higher is riskier."""
    higher_is_riskier = True
    default_threshold = 0.5

    def compute(self, target_traj, neighbor_trajs, dt=0.1, collision_threshold=2.0, **kwargs):
        dist = _pairwise_dist(target_traj, neighbor_trajs)
        below = (dist < collision_threshold).float()  # (B, T_f, N_v)
        tet = below.sum(dim=1) * dt  # (B, N_v) total exposure time in seconds
        return tet


@RiskMetricRegistry.register("tit")
class TimeIntegratedTTCMetric(RiskMetric):
    """Time Integrated TTC (TIT): integral of (threshold - distance)
    over timesteps where distance < threshold. Higher TIT = riskier.
    Captures both duration and severity of exposure."""
    higher_is_riskier = True
    default_threshold = 0.5

    def compute(self, target_traj, neighbor_trajs, dt=0.1, collision_threshold=2.0, **kwargs):
        dist = _pairwise_dist(target_traj, neighbor_trajs)
        # deficit = max(0, threshold - distance)
        deficit = (collision_threshold - dist).clamp(min=0)  # (B, T_f, N_v)
        tit = deficit.sum(dim=1) * dt  # (B, N_v)
        return tit


@RiskMetricRegistry.register("drac")
class DRACMetric(RiskMetric):
    """Deceleration Rate to Avoid Collision (DRAC):
    DRAC = v_rel^2 / (2 * d), where v_rel is relative closing speed.
    Returns max DRAC over the horizon per neighbor. Higher = riskier."""
    higher_is_riskier = True
    default_threshold = 3.0

    def compute(self, target_traj, neighbor_trajs, dt=0.1, **kwargs):
        B, T_f, N_v, _ = neighbor_trajs.shape
        target_exp = target_traj.unsqueeze(2)  # (B, T_f, 1, 2)

        dist = _pairwise_dist(target_traj, neighbor_trajs).clamp(min=0.01)  # (B, T_f, N_v)

        # Relative velocity: difference in consecutive displacements
        target_vel = torch.zeros_like(target_exp)
        target_vel[:, 1:] = (target_exp[:, 1:] - target_exp[:, :-1]) / dt
        neighbor_vel = torch.zeros_like(neighbor_trajs)
        neighbor_vel[:, 1:] = (neighbor_trajs[:, 1:] - neighbor_trajs[:, :-1]) / dt

        rel_vel = target_vel - neighbor_vel  # (B, T_f, N_v, 2)
        # Relative closing speed: projection of rel_vel onto the line connecting them
        rel_pos = target_exp - neighbor_trajs  # (B, T_f, N_v, 2)
        rel_pos_norm = rel_pos / (dist.unsqueeze(-1).clamp(min=0.01))
        closing_speed = -(rel_vel * rel_pos_norm).sum(dim=-1)  # (B, T_f, N_v)
        closing_speed = closing_speed.clamp(min=0)  # only approaching

        drac = closing_speed ** 2 / (2 * dist)  # (B, T_f, N_v)
        max_drac, _ = drac.max(dim=1)  # (B, N_v)
        return max_drac


@RiskMetricRegistry.register("rd")
class RequiredDecelerationMetric(RiskMetric):
    """Required Deceleration (RD): v^2 / (2 * d), using absolute target speed
    and pairwise distance. Returns max RD over the horizon. Higher = riskier."""
    higher_is_riskier = True
    default_threshold = 3.0

    def compute(self, target_traj, neighbor_trajs, dt=0.1, **kwargs):
        dist = _pairwise_dist(target_traj, neighbor_trajs).clamp(min=0.01)

        target_vel = torch.zeros_like(target_traj)
        target_vel[:, 1:] = (target_traj[:, 1:] - target_traj[:, :-1]) / dt
        speed = torch.norm(target_vel, dim=-1)  # (B, T_f)

        rd = speed.unsqueeze(2) ** 2 / (2 * dist)  # (B, T_f, N_v)
        max_rd, _ = rd.max(dim=1)
        return max_rd


@RiskMetricRegistry.register("tmd")
class TimeToMinDistMetric(RiskMetric):
    """Time to Minimum Distance (TMD): the time at which the minimum
    distance to a neighbor occurs. Lower TMD = less time to react."""
    higher_is_riskier = False
    default_threshold = 2.0

    def compute(self, target_traj, neighbor_trajs, dt=0.1, **kwargs):
        dist = _pairwise_dist(target_traj, neighbor_trajs)
        _, min_idx = dist.min(dim=1)  # (B, N_v)
        tmd = min_idx.float() * dt
        return tmd


@RiskMetricRegistry.register("s_field")
class SFieldMetric(RiskMetric):
    """STRAP Spatial Risk Field (S-field):
    r_ij^s = exp(-(|dx/γx|^αx + |dy/γy|^αy)).
    Max S-field over future timesteps. Higher = riskier."""
    higher_is_riskier = True
    default_threshold = 0.005

    def compute(self, target_traj, neighbor_trajs, dt=0.1,
                gamma_x=1.0, gamma_y=1.0, alpha_x=2.0, alpha_y=2.0, **kwargs):
        target_exp = target_traj.unsqueeze(2)  # (B, T_f, 1, 2)
        delta = neighbor_trajs - target_exp  # (B, T_f, N_v, 2)
        dx = delta[..., 0]
        dy = delta[..., 1]
        s_risk = torch.exp(
            -(torch.abs(dx / gamma_x) ** alpha_x +
              torch.abs(dy / gamma_y) ** alpha_y)
        )  # (B, T_f, N_v)
        max_s, _ = s_risk.max(dim=1)  # (B, N_v)
        return max_s


@RiskMetricRegistry.register("o_field")
class OFieldMetric(RiskMetric):
    """STRAP Objective Risk Field (O-field):
    r_ij^o = exp[-(d_m/d*)^β1] * exp[-(t_m/t*)^β2].
    Uses future min distance and time to closest approach. Higher = riskier."""
    higher_is_riskier = True
    default_threshold = 0.005

    def compute(self, target_traj, neighbor_trajs, dt=0.1,
                d_star=5.0, t_star=2.0, beta_1=1.0, beta_2=1.0, **kwargs):
        dist = _pairwise_dist(target_traj, neighbor_trajs)
        d_min, t_idx = dist.min(dim=1)  # (B, N_v)
        t_min = t_idx.float() * dt

        d_term = torch.exp(-((d_min / d_star) ** beta_1))
        t_term = torch.exp(-((t_min / t_star) ** beta_2))
        return d_term * t_term
