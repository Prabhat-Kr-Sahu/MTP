import torch

from src.evaluation.risk_metrics import METRICS, is_collision, pet


def _pet_reference(target, neighbors, dt=0.1, spatial_threshold=2.0):
    B, T, N, _ = neighbors.shape
    out = torch.full((B, N), float("inf"))
    for b in range(B):
        for n in range(N):
            for t_n in range(T):
                d = torch.norm(target[b] - neighbors[b, t_n, n], dim=-1)
                d_min, t_t = d.min(dim=0)
                if d_min.item() < spatial_threshold:
                    out[b, n] = min(out[b, n].item(), abs(t_n - t_t.item()) * dt)
    return out


def test_every_metric_returns_one_value_per_neighbour():
    target = torch.cumsum(torch.randn(2, 50, 2), dim=1)
    neighbors = target.unsqueeze(2) + torch.randn(2, 50, 4, 2)
    for name, (fn, _, _) in METRICS.items():
        assert fn(target, neighbors).shape == (2, 4), name


def test_vectorised_pet_matches_loop_reference():
    torch.manual_seed(1)
    target = torch.cumsum(torch.randn(2, 20, 2), dim=1)
    neighbors = target.unsqueeze(2) + torch.randn(2, 20, 3, 2) * 2
    torch.testing.assert_close(pet(target, neighbors), _pet_reference(target, neighbors))


def test_is_collision_respects_direction():
    assert is_collision("min_dist", 1.0) and not is_collision("min_dist", 3.0)
    assert is_collision("drac", 4.0) and not is_collision("drac", 1.0)
