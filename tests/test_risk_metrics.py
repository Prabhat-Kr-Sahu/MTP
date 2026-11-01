import torch

from src.evaluation import risk_metrics
from src.evaluation.risk_metrics import (
    METRICS,
    PROBABILITY_METRICS,
    btn,
    dce,
    is_collision,
    p_coll_gauss,
    pet,
    ttce,
)


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


def _closing_pair(gap=400.0, ego=30.0, other=25.0, T=50):
    """One pair closing longitudinally: (target (1,T,2), neighbors (1,T,1,2))."""
    steps = torch.arange(T, dtype=torch.float32)
    target = torch.stack([steps * ego * 0.1, torch.zeros(T)], -1).view(1, T, 2)
    neighbors = torch.stack([gap + steps * other * 0.1, torch.zeros(T)], -1).view(1, T, 1, 2)
    return target, neighbors


def test_padding_a_neighbour_does_not_change_other_pairs():
    """The invariant batched scoring depends on: ragged candidate counts are
    padded to the batch maximum, so a padded far-away neighbour must not alter
    any real pair's score. Covers every present and future METRICS entry."""
    torch.manual_seed(2)
    target = torch.cumsum(torch.randn(3, 50, 2), dim=1) * 0.5
    narrow = target.unsqueeze(2) + torch.randn(3, 50, 4, 2) + 3.0
    wide = torch.cat([narrow, torch.full((3, 50, 1, 2), 1e4)], dim=2)
    for name, (fn, _, _) in METRICS.items():
        a = fn(target, narrow).nan_to_num(nan=1e30)
        b = fn(target, wide)[:, :4].nan_to_num(nan=1e30)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5, msg=name)


def test_no_metric_returns_nan_when_a_neighbour_future_is_absent():
    """Ground-truth neighbour futures carry NaN where a vehicle leaves the
    section; torch.min/max propagate NaN, which would poison every score on
    the oracle and pred_gt variants."""
    torch.manual_seed(3)
    target = torch.cumsum(torch.randn(2, 50, 2), dim=1) * 0.5
    neighbors = target.unsqueeze(2) + torch.randn(2, 50, 3, 2) + 2.0
    neighbors[:, 25:, 0, :] = float("nan")       # neighbour 0 absent half-way
    neighbors[:, :, 1, :] = float("nan")         # neighbour 1 never present
    for name, (fn, _, _) in METRICS.items():
        assert not torch.isnan(fn(target, neighbors)).any(), name


def test_probability_metrics_are_bounded():
    torch.manual_seed(4)
    target = torch.cumsum(torch.randn(2, 50, 2), dim=1) * 0.5
    neighbors = target.unsqueeze(2) + torch.randn(2, 50, 3, 2)
    for name in sorted(PROBABILITY_METRICS):
        out = METRICS[name][0](target, neighbors)
        assert (out >= 0).all() and (out <= 1).all(), name


def test_ctx_sigma_changes_the_gaussian_collision_probability():
    """Exercises the ctx path the base contract test cannot reach."""
    target = torch.zeros(1, 5, 2)
    neighbors = torch.full((1, 5, 1, 2), 6.0)
    ctx = {"ego_dims": torch.tensor([[4.5, 1.8]]),
           "other_dims": torch.tensor([[[4.5, 1.8]]]),
           "rho": torch.zeros(1, 5)}
    tight = p_coll_gauss(target, neighbors, sigma=torch.full((1, 5, 2), 0.1), **ctx)
    loose = p_coll_gauss(target, neighbors, sigma=torch.full((1, 5, 2), 5.0), **ctx)
    assert tight.item() < loose.item()


def test_dst_reduces_to_required_longitudinal_acceleration_at_zero_safety_time():
    """The identity the source states: DST with t_s = 0 is a_long,req."""
    for gap, other in [(60.0, 1.0), (400.0, 25.0)]:
        target, neighbors = _closing_pair(gap=gap, other=other)
        saved, risk_metrics.T_SAFETY = risk_metrics.T_SAFETY, 0.0
        try:
            got = risk_metrics.dst(target, neighbors).item()
            want = btn(target, neighbors).item() * risk_metrics.A_MAX_BRAKE
        finally:
            risk_metrics.T_SAFETY = saved
        assert abs(got - want) < 1e-3, (gap, got, want)


def test_monte_carlo_matches_the_analytic_integral_at_zero_correlation():
    """p_coll_gauss factorises the rectangle integral across axes, ignoring rho.
    At rho = 0 the factorisation is exact, so the rho-exact Monte-Carlo
    estimate must agree -- this is the test for that approximation."""
    target = torch.zeros(1, 1, 2)
    neighbors = torch.tensor([[[[4.0, 0.6]]]])
    ctx = {"sigma": torch.full((1, 1, 2), 2.0), "rho": torch.zeros(1, 1),
           "ego_dims": torch.tensor([[4.5, 1.8]]),
           "other_dims": torch.tensor([[[4.5, 1.8]]])}
    saved, risk_metrics.MC_SAMPLES = risk_metrics.MC_SAMPLES, 16384
    try:
        analytic = p_coll_gauss(target, neighbors, **ctx).item()
        sampled = risk_metrics.p_coll_mc(target, neighbors, **ctx).item()
    finally:
        risk_metrics.MC_SAMPLES = saved
    assert abs(analytic - sampled) < 0.02, (analytic, sampled)


def test_closer_neighbours_are_riskier_for_every_metric():
    """Direction sanity: moving the neighbour nearer must not make any metric
    report less risk, once higher_is_riskier is accounted for."""
    target, near = _closing_pair(gap=30.0)
    _, far = _closing_pair(gap=300.0)
    for name, (fn, higher_is_riskier, _) in METRICS.items():
        if name in {"tmd", "ttce", "pret"}:
            continue          # timing metrics, not monotone in separation
        n = fn(target, near).item()
        f = fn(target, far).item()
        assert (n >= f) if higher_is_riskier else (n <= f), (name, n, f)


def test_dce_and_ttce_agree_with_closest_approach():
    """Head-on pair with a hand-computable closest approach."""
    T = 10
    target = torch.zeros(1, T, 2)
    neighbors = torch.stack(
        [10.0 - torch.arange(T, dtype=torch.float32), torch.full((T,), 3.0)], -1
    ).view(1, T, 1, 2)                      # closes 10 m/s longitudinally, 3 m offset
    assert abs(dce(target, neighbors).item() - 3.0) < 1e-4
    assert ttce(target, neighbors).item() >= 0.0
