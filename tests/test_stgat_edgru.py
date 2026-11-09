"""STGAT-EDGRU model contract and the Collision Risk Index maths."""
import math

import torch

from src.evaluation.risk_metrics import (
    METRICS,
    collision_radius,
    cri_aggregate,
    cri_fuse,
    cri_steps,
)
from src.models.stgat_edgru import STGATEDGRU


def _inputs(B=2, T=6, N=4, F=6):
    torch.manual_seed(0)
    x = torch.randn(B, T, N, F)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[-1, -1] = B == 1   # with B > 1 the last scene has a padded vehicle
    return x, mask


def test_shapes_and_valid_gaussian():
    x, mask = _inputs()
    model = STGATEDGRU(in_dim=6, t_f=5, d=16, heads=4).eval()
    traj, goals, extra = model(x, mask, None)
    assert traj.shape == (2, 5, 5) and goals.shape == (2, 3, 4) and extra is None
    assert (traj[..., 2:4] > 0).all() and (traj[..., 4].abs() < 1).all()
    assert model.forward_all(x, mask).shape == (2, 4, 5, 5)


def test_forward_matches_forward_all_target():
    x, mask = _inputs()
    model = STGATEDGRU(in_dim=6, t_f=5, d=16, heads=4).eval()
    assert torch.allclose(model(x, mask)[0], model.forward_all(x, mask)[:, 0], atol=1e-5)


def test_far_vehicles_do_not_influence_target():
    x, mask = _inputs(B=1)
    x[..., 1:, 0] += 100.0                  # every neighbour > 50 m away
    model = STGATEDGRU(in_dim=6, t_f=5, d=16, heads=4).eval()
    base = model(x, mask)[0]
    x2 = x.clone()
    x2[..., 1:, 2:] += 5.0                  # change neighbours' other features
    assert torch.allclose(base, model(x2, mask)[0], atol=1e-5)
    x3 = x.clone()
    x3[..., 1, 0] -= 95.0                   # bring one neighbour within range
    x3[..., 1, 2:] += 5.0
    assert not torch.allclose(base, model(x3, mask)[0], atol=1e-5)


def test_teacher_forcing_changes_decoding():
    x, mask = _inputs()
    model = STGATEDGRU(in_dim=6, t_f=5, d=16, heads=4).eval()
    gt = torch.randn(2, 4, 5, 2) * 10
    assert not torch.allclose(model.forward_all(x, mask), model.forward_all(x, mask, gt))


def test_ablations_run():
    x, mask = _inputs()
    for flags in ({"use_transformer": False}, {"use_attention": False}, {"use_gmu": False}):
        traj, _, _ = STGATEDGRU(in_dim=6, t_f=5, d=16, heads=4, **flags).eval()(x, mask)
        assert traj.shape == (2, 5, 5) and torch.isfinite(traj).all()


def _pair(offset, sigma_i=0.5, sigma_j=0.0, rho_i=0.0, rho_j=0.0, speed_j=0.0, T=4):
    t = torch.arange(1, T + 1, dtype=torch.float32) * 0.1
    mu_i = torch.stack([10.0 * t, torch.zeros(T)], -1).view(1, T, 1, 2)
    mu_j = torch.stack([offset + speed_j * t, torch.zeros(T)], -1).view(1, T, 1, 2)
    s_i, s_j = torch.full((1, T, 1, 2), sigma_i), torch.full((1, T, 1, 2), sigma_j)
    r_i, r_j = torch.full((1, T, 1), rho_i), torch.full((1, T, 1), rho_j)
    return mu_i, s_i, r_i, mu_j, s_j, r_j


def test_collision_radius_eq30():
    r = collision_radius(torch.tensor([[4.0, 2.0]]), torch.tensor([[[4.0, 2.0]]]))
    assert torch.allclose(r, torch.tensor([[0.5 * math.sqrt(64 + 16)]]))


def test_probability_near_one_when_overlapping_and_zero_when_far():
    radius = torch.tensor(2.0)
    f, p, _ = cri_steps(*_pair(offset=0.0, sigma_i=0.05, speed_j=10.0), radius)
    assert (p > 0.99).all()
    f, p, _ = cri_steps(*_pair(offset=100.0), radius)
    assert (p == 0).all() and (f == 0).all()


def test_probability_matches_normal_cdf_in_1d_limit():
    # Very wide lateral sigma is impossible to isolate in 2D, so check a known
    # 2D case instead: centred isotropic Gaussian inside radius r has
    # P = 1 - exp(-r^2 / (2 s^2)).
    T = 1
    mu = torch.zeros(1, T, 1, 2)
    s = torch.full((1, T, 1, 2), 1.0)
    z = torch.zeros(1, T, 1)
    _, p, _ = cri_steps(mu, s, z, mu, torch.zeros_like(s), z, torch.tensor(1.5), fixed=True)
    assert abs(p.item() - (1 - math.exp(-1.5 ** 2 / 2))) < 0.04


def test_paper_vs_fixed_variance_and_intensity():
    radius = torch.tensor(2.0)
    args = _pair(offset=0.0, sigma_i=1.0, sigma_j=1.0, speed_j=10.0)
    _, p_paper, e_paper = cri_steps(*args, radius, fixed=False)
    _, p_fixed, e_fixed = cri_steps(*args, radius, fixed=True)
    # Eq. 29 adds sigmas (2.0) instead of variances (sqrt 2): wider -> lower P.
    assert (p_paper < p_fixed).all()
    # Same speed: paper intensity = exp(0) = 1, fixed = 0.
    assert torch.allclose(e_paper, torch.ones_like(e_paper))
    assert torch.allclose(e_fixed, torch.zeros_like(e_fixed))
    # Paper intensity falls as relative speed grows; fixed rises.
    slow = cri_steps(*_pair(offset=0.0, speed_j=8.0), radius, fixed=False)[2]
    fast = cri_steps(*_pair(offset=0.0, speed_j=0.0), radius, fixed=False)[2]
    assert (fast < slow).all()
    slow = cri_steps(*_pair(offset=0.0, speed_j=8.0), radius, fixed=True)[2]
    fast = cri_steps(*_pair(offset=0.0, speed_j=0.0), radius, fixed=True)[2]
    assert (fast > slow).all()


def test_fuse_and_aggregate():
    f = torch.tensor([[[0.2], [0.6]]])           # (B=1, T=2, N=1)
    assert torch.allclose(cri_fuse(f), torch.tensor([[0.5 * 0.4 + 0.5 * 0.6]]))
    f = torch.tensor([[[0.4, 0.1]]])             # (B=1, T=1, N=2)
    p = torch.tensor([[[0.8, 0.2]]])
    assert torch.allclose(cri_aggregate(f, p), torch.tensor([[0.8 * 0.4 + 0.2 * 0.1]]))
    assert cri_aggregate(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)).item() == 0.0


def test_registered_metrics_bounded():
    target = torch.stack([torch.linspace(0.1, 5, 50), torch.zeros(50)], -1).unsqueeze(0)
    neighbours = torch.stack([torch.full((50,), 3.0), torch.full((50,), 50.0)], -1)
    neighbours = torch.stack([target[0], neighbours], 1).unsqueeze(0)   # (1,50,2,2)
    neighbours[:, :, 0] += torch.tensor([2.0, 0.0])
    for name in ("cri_paper", "cri_fixed"):
        out = METRICS[name][0](target, neighbours, sigma=torch.full((1, 50, 2), 0.5))
        assert out.shape == (1, 2) and ((out >= 0) & (out <= 1)).all()
        assert out[0, 1] == 0
