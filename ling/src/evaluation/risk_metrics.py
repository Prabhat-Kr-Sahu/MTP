"""Extensible risk metric evaluation framework."""
import torch
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


class RiskMetric(ABC):
    """Abstract base class for trajectory-based risk metrics."""

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


@RiskMetricRegistry.register("ttc")
class TimeToCollisionMetric(RiskMetric):
    """Time-To-Collision based on future trajectory Euclidean distance."""

    def compute(self, target_traj, neighbor_trajs, dt=0.1, collision_threshold=2.0, **kwargs):
        B, T_f, N_v, _ = neighbor_trajs.shape

        # Broadcast target prediction to match neighbors: (B, T_f, 1, 2)
        target_exp = target_traj.unsqueeze(2)

        # Compute Euclidean distance at every future timestep: (B, T_f, N_v)
        dist = torch.norm(target_exp - neighbor_trajs, dim=-1)

        # Find minimum distance and the time index it occurs
        min_dist, min_time_idx = dist.min(dim=1)  # Both are (B, N_v)

        # A collision happens if the minimum distance is below our threshold
        is_collision = min_dist < collision_threshold

        # TTC is the time index * dt. If no collision, TTC is infinity.
        ttc = torch.where(
            is_collision,
            min_time_idx.float() * dt,
            torch.full_like(min_dist, float('inf'))
        )

        return ttc


@RiskMetricRegistry.register("min_dist")
class MinimumDistanceMetric(RiskMetric):
    """Absolute minimum distance over the predicted horizon."""

    def compute(self, target_traj, neighbor_trajs, dt=0.1, **kwargs):
        # Broadcast target prediction to match neighbors: (B, T_f, 1, 2)
        target_exp = target_traj.unsqueeze(2)

        # Compute Euclidean distance at every future timestep: (B, T_f, N_v)
        dist = torch.norm(target_exp - neighbor_trajs, dim=-1)

        # Find minimum distance over time
        min_dist, _ = dist.min(dim=1)  # (B, N_v)

        return min_dist
