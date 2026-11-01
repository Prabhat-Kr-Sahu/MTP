"""Trajectory-based risk metrics for collision evaluation.

Each metric maps a target trajectory (B, T_f, 2) and neighbour trajectories
(B, T_f, N_v, 2) to one value per neighbour, shape (B, N_v).

The contract is

    fn(target, neighbors, dt=DT, threshold=2.0, **ctx) -> (B, N_v)

`threshold` always means the collision radius / contact distance in metres; a
metric's own scale constants live in this module, never in `threshold`.

`ctx` carries the optional extra inputs some families need (the predicted
Gaussian, vehicle dimensions). Every metric is *total* on the base contract:
`_context` fills anything missing from what is derivable from `target` and
`neighbors`, so `fn(target, neighbors)` always works. A derived sigma is not a
real prediction -- `risk_scoring.score_split` always supplies the model's own.

Families refer to RISK_PREDICTION_METHODOLOGIES.md:
  A1 deterministic surrogates, A2 collision probability, A3 risk fields,
  A4 severity-weighted, A6 formal/set-based.
"""
import math

import torch

from src.config import DT
from src.risk import (
    closest_approach,
    compute_o_field,
    compute_s_field,
    default_dims,
    pair_risk,
    pairwise_distance,
)

# --------------------------------------------------------------------------- #
#  Scale constants. Reported into every result JSON so the comparison is
#  reproducible and nothing here can be quietly tuned on the test split.
# --------------------------------------------------------------------------- #
A_MAX_BRAKE = 8.0      # m/s^2, emergency longitudinal deceleration
A_MAX_LAT = 4.0        # m/s^2, available lateral acceleration
T_SAFETY = 1.5         # s, safety time headway for DST
RHO_VEHICLE = 200.0    # kg/m^2 of footprint; 4.5 x 1.8 m -> 1620 kg
MC_SAMPLES = 64        # Monte-Carlo draws for p_coll_mc
MC_CHUNK = 16          # draws per chunk, to bound peak memory
GAMMA_CI = 1.0         # 1/s, conflict-index calibration (country/geometry dependent)
DELTA_CI = 1.0         # fraction of kinetic energy reaching occupants
JOKSCH_V = 31.74       # m/s, Joksch fatality-probability scale
RSS_RHO = 0.5          # s, response time
RSS_A_MAX = 3.0        # m/s^2, max acceleration during the response
RSS_B_MIN = 4.0        # m/s^2, min braking of the rear vehicle
RSS_B_MAX = 8.0        # m/s^2, max braking of the front vehicle
RSS_MU_LAT = 0.5       # m, lateral fluctuation margin
EPS = 1e-6


# --------------------------------------------------------------------------- #
#  Shared helpers
# --------------------------------------------------------------------------- #
def _velocity(traj, dt):
    """Finite-difference velocity along dim 1; the first step is zero."""
    vel = torch.zeros_like(traj)
    vel[:, 1:] = (traj[:, 1:] - traj[:, :-1]) / dt
    return vel


def _velocity_n(trajs, dt):
    """As _velocity, for neighbour trajectories (B, T, N, 2)."""
    vel = torch.zeros_like(trajs)
    vel[:, 1:] = (trajs[:, 1:] - trajs[:, :-1]) / dt
    return vel


def _min_over_time(values):
    """Min over the time axis, treating NaN as "no interaction at this step".

    Ground-truth neighbour futures carry NaN where a vehicle is absent, and
    torch.min propagates NaN -- which would silently poison every score on the
    oracle and pred_gt variants. All-NaN pairs yield +inf, i.e. no event.
    """
    return values.nan_to_num(nan=float("inf")).min(dim=1).values


def _max_over_time(values, empty=0.0):
    """Max over the time axis, treating NaN as absent. All-NaN pairs give `empty`."""
    filled = values.nan_to_num(nan=float("-inf")).max(dim=1).values
    return torch.where(torch.isfinite(filled), filled, torch.full_like(filled, empty))


def _sum_over_time(values):
    """Sum over the time axis, treating NaN as zero contribution."""
    return values.nan_to_num(nan=0.0).sum(dim=1)


def _relative(target, neighbors, dt):
    """(delta, relative_velocity, distance): other - ego, shapes (B,T,N,2/2/-)."""
    delta = neighbors - target.unsqueeze(2)
    rel_vel = _velocity_n(neighbors, dt) - _velocity(target, dt).unsqueeze(2)
    return delta, rel_vel, torch.linalg.vector_norm(delta, dim=-1)


def _context(target, neighbors, dt, ctx):
    """Fill the optional extra inputs from whatever the caller did not supply.

    Returns (sigma (B,T,2), rho (B,T), ego_dims (B,2), other_dims (B,N,2),
    ego_speed (B,T)). Defaults: dims from src.risk.default_dims, ego speed from
    the predicted trajectory itself (exact -- the trajectory *is* the motion),
    and an isotropic 1 m sigma with zero correlation, which only ever applies
    when a caller asks for a probability metric without a predicted Gaussian.
    """
    B, T, _ = target.shape
    N = neighbors.shape[2]
    device = target.device

    sigma = ctx.get("sigma")
    if sigma is None:
        sigma = torch.ones(B, T, 2, device=device)
    rho = ctx.get("rho")
    if rho is None:
        rho = torch.zeros(B, T, device=device)

    ego_dims = ctx.get("ego_dims")
    if ego_dims is None:
        ego_dims = default_dims(B, device=device).clone()
    other_dims = ctx.get("other_dims")
    if other_dims is None:
        other_dims = default_dims(B, N, device=device).clone()

    ego_speed = ctx.get("ego_speed")
    if ego_speed is None:
        ego_speed = _velocity(target, dt).norm(dim=-1)
    elif ego_speed.dim() == 1:
        ego_speed = ego_speed.unsqueeze(1).expand(B, T)

    return sigma, rho, ego_dims, other_dims, ego_speed


def _half_extents(ego_dims, other_dims):
    """Summed half length/width per pair: (B, N) longitudinal, (B, N) lateral."""
    half = 0.5 * (ego_dims.unsqueeze(1) + other_dims)   # (B, N, 2)
    return half[..., 0], half[..., 1]


def _masses(ego_dims, other_dims):
    """Mass proxy from footprint area. NGSIM has no masses -- see the doc."""
    ego = RHO_VEHICLE * ego_dims[:, 0] * ego_dims[:, 1]            # (B,)
    other = RHO_VEHICLE * other_dims[..., 0] * other_dims[..., 1]  # (B, N)
    return ego.unsqueeze(1).expand_as(other), other


def _box_probability(delta, sigma, rho, half_long, half_lat):
    """P(predicted target centre lies in the neighbour's box) per step and pair.

    delta (B,T,N,2) is other - ego, so the target offset relative to the
    neighbour is -delta and the per-axis interval is [-h, h].

    ponytail: the rectangle integral is factorised across axes, which ignores
    rho. |rho| is typically small here; p_coll_mc is the rho-exact reference and
    the two are asserted to agree at rho = 0.
    """
    s = sigma.clamp(min=EPS).unsqueeze(2)                 # (B,T,1,2)
    h = torch.stack([half_long, half_lat], dim=-1).unsqueeze(1)   # (B,1,N,2)
    upper = torch.special.ndtr((h + delta) / s)
    lower = torch.special.ndtr((-h + delta) / s)
    return (upper - lower).clamp(min=0.0).prod(dim=-1)    # (B,T,N)


def _a_req(delta, rel_vel, half_long, half_lat):
    """Required longitudinal and lateral acceleration to avoid contact.

    Returns two (B, T, N) tensors. Longitudinal uses the bumper-to-bumper gap
    and the closing speed along the longitudinal axis; lateral asks what lateral
    acceleration clears the remaining width inside the longitudinal closure time.
    """
    gap_long = (delta[..., 0].abs() - half_long.unsqueeze(1)).clamp(min=0.1)
    sign = torch.sign(delta[..., 0])
    closing = (-rel_vel[..., 0] * sign).clamp(min=0.0)
    a_long = closing.square() / (2.0 * gap_long)

    overlap = (half_lat.unsqueeze(1) - delta[..., 1].abs()).clamp(min=0.0)
    closure_time = (gap_long / closing.clamp(min=EPS)).clamp(min=EPS)
    a_lat = torch.where(
        overlap > 0, 2.0 * overlap / closure_time.square(), torch.zeros_like(overlap)
    )
    return a_long, a_lat


# --------------------------------------------------------------------------- #
#  A1 -- deterministic surrogates (the original eleven)
# --------------------------------------------------------------------------- #
def min_dist(target, neighbors, dt=DT, threshold=2.0, **_):
    """Minimum Distance (MD) over the horizon."""
    return _min_over_time(pairwise_distance(target, neighbors))


def ttc(target, neighbors, dt=DT, threshold=2.0, **_):
    """Distance-crossing onset time: first time distance < threshold; inf if never.

    Kept under its original name because 39 committed result JSONs and the
    --collision_metric default refer to it. `ttc_true` is the real TTC.
    """
    below = pairwise_distance(target, neighbors) < threshold
    first = below.float().argmax(dim=1).float() * dt
    return torch.where(below.any(dim=1), first, torch.full_like(first, float("inf")))


def pet(target, neighbors, dt=DT, threshold=2.0, spatial_threshold=2.0, **_):
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


def thw(target, neighbors, dt=DT, threshold=2.0, **_):
    """Time Headway: min over horizon of distance / target speed."""
    vel = _velocity(target, dt)
    vel[:, 0] = vel[:, 1]
    speed = vel.norm(dim=-1).clamp(min=0.1)
    return _min_over_time(pairwise_distance(target, neighbors) / speed.unsqueeze(2))


def tet(target, neighbors, dt=DT, threshold=2.0, **_):
    """Time Exposed TTC: total time spent closer than threshold."""
    return _sum_over_time(
        (pairwise_distance(target, neighbors) < threshold).float()
    ) * dt


def tit(target, neighbors, dt=DT, threshold=2.0, **_):
    """Time Integrated TTC: integral of (threshold - distance) while below it."""
    return _sum_over_time(
        (threshold - pairwise_distance(target, neighbors)).clamp(min=0)
    ) * dt


def drac(target, neighbors, dt=DT, threshold=2.0, **_):
    """Deceleration Rate to Avoid Collision: max of closing_speed^2 / (2 d)."""
    delta, rel_vel, dist = _relative(target, neighbors, dt)
    dist = dist.clamp(min=0.01)
    # Radial component of the relative velocity, positive when closing.
    closing = (-(rel_vel * delta).sum(dim=-1) / dist).clamp(min=0)
    return _max_over_time(closing ** 2 / (2 * dist))


def rd(target, neighbors, dt=DT, threshold=2.0, **_):
    """Required Deceleration: max of target_speed^2 / (2 d)."""
    dist = pairwise_distance(target, neighbors).clamp(min=0.01)
    speed = _velocity(target, dt).norm(dim=-1)
    return _max_over_time(speed.unsqueeze(2) ** 2 / (2 * dist))


def tmd(target, neighbors, dt=DT, threshold=2.0, **_):
    """Time to Minimum Distance."""
    dist = pairwise_distance(target, neighbors).nan_to_num(nan=float("inf"))
    return dist.min(dim=1).indices.float() * dt


def s_field(target, neighbors, dt=DT, threshold=2.0, **_):
    """Max uncalibrated STRAP S-field over the horizon (gamma=1, alpha=2)."""
    delta = neighbors - target.unsqueeze(2)
    return _max_over_time(compute_s_field(delta[..., 0], delta[..., 1]))


def o_field(target, neighbors, dt=DT, threshold=2.0, **_):
    """Uncalibrated STRAP O-field from min distance and its time (d*=5, t*=2)."""
    dist = pairwise_distance(target, neighbors).nan_to_num(nan=float("inf"))
    d_min, t_idx = dist.min(dim=1)
    return compute_o_field(d_min, t_idx.float() * dt)


# --------------------------------------------------------------------------- #
#  A1 -- added surrogates
# --------------------------------------------------------------------------- #
def ttc_true(target, neighbors, dt=DT, threshold=2.0, **_):
    """True Time-To-Collision: earliest contact time under the predicted motion.

    Per step, solve ||D + tau V|| = threshold for the smallest tau >= 0, then
    take the minimum of (step time + tau). inf when contact never occurs.
    """
    delta, rel_vel, _ = _relative(target, neighbors, dt)
    a = rel_vel.square().sum(dim=-1)
    b = 2.0 * (delta * rel_vel).sum(dim=-1)
    c = delta.square().sum(dim=-1) - threshold ** 2
    disc = b * b - 4.0 * a * c
    tau = (-b - disc.clamp(min=0.0).sqrt()) / (2.0 * a).clamp(min=EPS)
    touching = c <= 0                                  # already within contact distance
    reachable = (disc >= 0) & (a > EPS) & (tau >= 0)
    steps = torch.arange(delta.shape[1], device=delta.device).view(1, -1, 1) * dt
    time = torch.where(touching, steps, steps + tau)
    return _min_over_time(
        torch.where(touching | reachable, time, torch.full_like(time, float("inf")))
    )


def _cpa(target, neighbors, dt):
    """(dce, ttce) from the step whose constant-velocity CPA is smallest."""
    delta, rel_vel, _ = _relative(target, neighbors, dt)
    cpa_dist, cpa_time = closest_approach(delta, rel_vel)
    cpa_dist = cpa_dist.nan_to_num(nan=float("inf"))
    idx = cpa_dist.argmin(dim=1, keepdim=True)
    dce = cpa_dist.gather(1, idx).squeeze(1)
    offset = cpa_time.nan_to_num(nan=0.0).gather(1, idx).squeeze(1)
    return dce, idx.squeeze(1).float() * dt + offset


def dce(target, neighbors, dt=DT, threshold=2.0, **_):
    """Distance of Closest Encounter (reuses src.risk.closest_approach)."""
    return _cpa(target, neighbors, dt)[0]


def ttce(target, neighbors, dt=DT, threshold=2.0, **_):
    """Time to Closest Encounter."""
    return _cpa(target, neighbors, dt)[1]


def psd(target, neighbors, dt=DT, threshold=2.0, **_):
    """Proportion of Stopping Distance: min of distance / minimum stopping distance."""
    dist = pairwise_distance(target, neighbors)
    speed = _velocity(target, dt).norm(dim=-1)
    msd = (speed.square() / (2.0 * A_MAX_BRAKE)).clamp(min=0.01)
    return _min_over_time(dist / msd.unsqueeze(2))


def btn(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """Brake Threat Number: required deceleration / available deceleration.

    >= 1 means braking alone cannot avoid contact, which makes it comparable
    across vehicles in a way raw DRAC is not.
    """
    delta, rel_vel, _ = _relative(target, neighbors, dt)
    _, _, ego_dims, other_dims, _ = _context(target, neighbors, dt, ctx)
    half_long, half_lat = _half_extents(ego_dims, other_dims)
    a_long, _ = _a_req(delta, rel_vel, half_long, half_lat)
    return _max_over_time(a_long / A_MAX_BRAKE)


def stn(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """Steer Threat Number: required lateral acceleration / available lateral."""
    delta, rel_vel, _ = _relative(target, neighbors, dt)
    _, _, ego_dims, other_dims, _ = _context(target, neighbors, dt, ctx)
    half_long, half_lat = _half_extents(ego_dims, other_dims)
    _, a_lat = _a_req(delta, rel_vel, half_long, half_lat)
    return _max_over_time(a_lat / A_MAX_LAT)


def a_lat_req(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """Required Lateral Acceleration to evade, max over the horizon."""
    delta, rel_vel, _ = _relative(target, neighbors, dt)
    _, _, ego_dims, other_dims, _ = _context(target, neighbors, dt, ctx)
    half_long, half_lat = _half_extents(ego_dims, other_dims)
    _, a_lat = _a_req(delta, rel_vel, half_long, half_lat)
    return _max_over_time(a_lat)


def dst(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """Deceleration to Safety Time: deceleration needed to hold T_SAFETY seconds.

    Reduces to the constant-acceleration DRAC at T_SAFETY = 0.
    """
    delta, rel_vel, _ = _relative(target, neighbors, dt)
    _, _, ego_dims, other_dims, _ = _context(target, neighbors, dt, ctx)
    half_long, _ = _half_extents(ego_dims, other_dims)
    # Same gap floor as _a_req, so DST reduces exactly to a_long,req at T_SAFETY = 0.
    gap = (delta[..., 0].abs() - half_long.unsqueeze(1)).clamp(min=0.1)
    other_speed = _velocity_n(neighbors, dt)[..., 0].abs()
    sign = torch.sign(delta[..., 0])
    closing = (-rel_vel[..., 0] * sign).clamp(min=0.0)
    denom = 2.0 * (gap - T_SAFETY * other_speed).clamp(min=0.1)
    return _max_over_time(closing.square() / denom)


def pret(target, neighbors, dt=DT, threshold=2.0, **_):
    """Predictive Encroachment Time at the predicted path intersection.

    Constant-velocity rays from the first predicted step; |t1 - t2| at their
    crossing, inf when the rays are parallel or cross outside the horizon.

    Expect this to be sparse on NGSIM: collision_candidates reports neighbour
    velocity as [speed, 0], so in the cv variant the rays are near-parallel for
    most pairs and PrET is inf. The inf fraction is itself reportable.
    """
    horizon = target.shape[1] * dt
    p_t = target[:, 0].unsqueeze(1)                           # (B,1,2)
    v_t = _velocity(target, dt)[:, 1].unsqueeze(1)            # (B,1,2)
    p_n = neighbors[:, 0]                                     # (B,N,2)
    v_n = _velocity_n(neighbors, dt)[:, 1]                    # (B,N,2)

    det = v_n[..., 0] * v_t[..., 1] - v_t[..., 0] * v_n[..., 1]
    rhs = p_n - p_t
    t1 = (rhs[..., 0] * -v_n[..., 1] + v_n[..., 0] * rhs[..., 1]) / det.where(
        det.abs() > EPS, torch.ones_like(det)
    )
    t2 = (v_t[..., 0] * rhs[..., 1] - rhs[..., 0] * v_t[..., 1]) / det.where(
        det.abs() > EPS, torch.ones_like(det)
    )
    gap = (t1 - t2).abs()
    valid = (det.abs() > EPS) & (t1 >= 0) & (t2 >= 0) & (t1 <= horizon) & (t2 <= horizon)
    return torch.where(valid, gap, torch.full_like(gap, float("inf")))


# --------------------------------------------------------------------------- #
#  A2 -- collision probability from the predicted Gaussian
# --------------------------------------------------------------------------- #
def p_coll_gauss(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """Collision *state* probability: max over the horizon of the per-step
    overlap integral of the predicted bivariate Gaussian with the neighbour box.

    This is the form P-PDRF uses. p_coll_mc is the collision *event*
    probability; the doc flags that the two are routinely conflated.
    """
    delta = neighbors - target.unsqueeze(2)
    sigma, rho, ego_dims, other_dims, _ = _context(target, neighbors, dt, ctx)
    half_long, half_lat = _half_extents(ego_dims, other_dims)
    return _max_over_time(_box_probability(delta, sigma, rho, half_long, half_lat))


def p_coll_mc(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """Collision *event* probability by Monte Carlo over the predicted Gaussian.

    Draws MC_SAMPLES trajectories via the Cholesky factor of each step's
    covariance (so rho is exact, unlike p_coll_gauss), and counts the fraction
    that overlap the neighbour box at any step. Seeded for reproducibility.
    """
    sigma, rho, ego_dims, other_dims, _ = _context(target, neighbors, dt, ctx)
    half_long, half_lat = _half_extents(ego_dims, other_dims)
    h = torch.stack([half_long, half_lat], dim=-1).unsqueeze(1)   # (B,1,N,2)

    sx = sigma[..., 0].clamp(min=EPS).unsqueeze(-1)   # (B,T,1)
    sy = sigma[..., 1].clamp(min=EPS).unsqueeze(-1)
    r = rho.clamp(-0.999, 0.999).unsqueeze(-1)
    generator = torch.Generator(device="cpu").manual_seed(0)

    hits = torch.zeros(target.shape[0], neighbors.shape[2], device=target.device)
    drawn = 0
    while drawn < MC_SAMPLES:
        n = min(MC_CHUNK, MC_SAMPLES - drawn)
        eps = torch.randn(
            n, *target.shape, generator=generator, dtype=target.dtype
        ).to(target.device)                                      # (n,B,T,2)
        # Cholesky of [[sx^2, r sx sy], [r sx sy, sy^2]]
        dx = sx * eps[..., 0:1]
        dy = sy * (r * eps[..., 0:1] + (1 - r * r).clamp(min=0).sqrt() * eps[..., 1:2])
        sample = target.unsqueeze(0) + torch.cat([dx, dy], dim=-1)   # (n,B,T,2)
        offset = (neighbors.unsqueeze(0) - sample.unsqueeze(3)).abs()  # (n,B,T,N,2)
        # NaN <= h is False, so absent neighbour steps never count as a hit.
        inside = (offset <= h.unsqueeze(0)).all(dim=-1)              # (n,B,T,N)
        hits = hits + inside.any(dim=2).float().sum(dim=0)
        drawn += n
    return hits / MC_SAMPLES


# --------------------------------------------------------------------------- #
#  A3 -- calibrated risk fields (the [22] calibration, via src.risk.pair_risk)
# --------------------------------------------------------------------------- #
def _pair_fields(target, neighbors, dt, ctx):
    """Per-step calibrated (S, O) fields, each (B, T, N)."""
    delta, rel_vel, _ = _relative(target, neighbors, dt)
    _, _, ego_dims, other_dims, ego_speed = _context(target, neighbors, dt, ctx)
    B, T, N, _ = delta.shape
    return pair_risk(
        delta,
        rel_vel,
        ego_speed.unsqueeze(2).expand(B, T, N),
        ego_dims.view(B, 1, 1, 2).expand(B, T, N, 2),
        other_dims.view(B, 1, N, 2).expand(B, T, N, 2),
    )


def s_field_cal(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """Calibrated S-field: bumper-to-bumper gaps, speed-dependent gamma_x/alpha_x."""
    return _max_over_time(_pair_fields(target, neighbors, dt, ctx)[0])


def o_field_cal(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """Calibrated O-field: perpendicular miss distance, d* = half summed widths."""
    return _max_over_time(_pair_fields(target, neighbors, dt, ctx)[1])


def so_field_cal(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """R_s + R_o -- the quantity the risk-scaled loss and neighbour selection use."""
    s, o = _pair_fields(target, neighbors, dt, ctx)
    return _max_over_time(s + o)


# --------------------------------------------------------------------------- #
#  A4 -- severity-weighted risk
# --------------------------------------------------------------------------- #
def _delta_v_series(target, neighbors, dt, ctx):
    """Per-step mass-weighted speed change (B, T, N), plus the mass tensors."""
    _, rel_vel, _ = _relative(target, neighbors, dt)
    _, _, ego_dims, other_dims, _ = _context(target, neighbors, dt, ctx)
    m_ego, m_other = _masses(ego_dims, other_dims)
    ratio = (m_other / (m_ego + m_other)).unsqueeze(1)          # (B,1,N)
    return ratio * torch.linalg.vector_norm(rel_vel, dim=-1), m_ego, m_other


def delta_v(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """Delta-v at the step of minimum predicted distance.

    Mass is proxied as RHO_VEHICLE x length x width -- NGSIM carries no masses.
    """
    series, _, _ = _delta_v_series(target, neighbors, dt, ctx)
    dist = pairwise_distance(target, neighbors).nan_to_num(nan=float("inf"))
    idx = dist.argmin(dim=1, keepdim=True)
    return series.nan_to_num(nan=0.0).gather(1, idx).squeeze(1)


def conflict_index(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """Conflict Index: severity (released kinetic energy) x a PET-based
    collision-probability proxy, delta * dKE * exp(-gamma * PET).

    The review doc renders this as a division by exp(-gamma*PET), which makes
    risk grow with PET. We use the product, which matches the source's own
    description ("PET enhanced with a collision probability estimation as well
    as a severity factor") and is what makes it a probability x severity form.
    PET = inf therefore gives zero risk.
    """
    series, m_ego, m_other = _delta_v_series(target, neighbors, dt, ctx)
    dv = delta_v(target, neighbors, dt, threshold, **ctx)
    m_eff = m_ego * m_other / (m_ego + m_other)
    energy = DELTA_CI * 0.5 * m_eff * dv.square()
    encroachment = pet(target, neighbors, dt, threshold)
    return energy * torch.exp(-GAMMA_CI * encroachment)


def p_pdrf(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """P-PDRF form: max over the horizon of P(collision) x P(fatality | impact).

    Severity is Joksch's (delta_v / 31.74)^4. The only genuine
    Risk = probability x severity measure in the suite.
    """
    delta = neighbors - target.unsqueeze(2)
    sigma, rho, ego_dims, other_dims, _ = _context(target, neighbors, dt, ctx)
    half_long, half_lat = _half_extents(ego_dims, other_dims)
    probability = _box_probability(delta, sigma, rho, half_long, half_lat)
    series, _, _ = _delta_v_series(target, neighbors, dt, ctx)
    severity = (series / JOKSCH_V).clamp(min=0.0, max=1.0) ** 4
    return _max_over_time(probability * severity)


# --------------------------------------------------------------------------- #
#  A6 -- formal / set-based
# --------------------------------------------------------------------------- #
def _rss(target, neighbors, dt, ctx):
    """(longitudinal intrusion, lateral intrusion) per step, each (B, T, N)."""
    delta, _, _ = _relative(target, neighbors, dt)
    _, _, ego_dims, other_dims, ego_speed = _context(target, neighbors, dt, ctx)
    half_long, half_lat = _half_extents(ego_dims, other_dims)

    other_speed = _velocity_n(neighbors, dt)[..., 0].abs()      # (B,T,N)
    ego = ego_speed.unsqueeze(2).expand_as(other_speed)
    # The vehicle behind is the one that must keep the distance.
    ahead = delta[..., 0] > 0
    v_rear = torch.where(ahead, ego, other_speed)
    v_front = torch.where(ahead, other_speed, ego)

    d_min = (
        v_rear * RSS_RHO
        + 0.5 * RSS_A_MAX * RSS_RHO ** 2
        + (v_rear + RSS_A_MAX * RSS_RHO) ** 2 / (2 * RSS_B_MIN)
        - v_front.square() / (2 * RSS_B_MAX)
    ).clamp(min=0.0)
    lat_min = half_lat.unsqueeze(1) + RSS_MU_LAT
    return d_min - delta[..., 0].abs(), lat_min - delta[..., 1].abs()


def rss_ds(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """RSS Dangerous Situation: 1 if both safe distances are ever violated.

    Nominal scale, so its ROC has two points and its AUC collapses to
    (1 + TPR - FPR)/2 -- not comparable with the continuous rows. rss_margin is
    the gradable version of the same quantity.
    """
    long_intrusion, lat_intrusion = _rss(target, neighbors, dt, ctx)
    unsafe = (long_intrusion > 0) & (lat_intrusion > 0)
    return _max_over_time(unsafe.float())


def rss_margin(target, neighbors, dt=DT, threshold=2.0, **ctx):
    """Worst longitudinal RSS intrusion over the horizon (metres, >0 is unsafe)."""
    long_intrusion, _ = _rss(target, neighbors, dt, ctx)
    return _max_over_time(long_intrusion, empty=float("-inf"))


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
# name -> (function, higher_is_riskier, default_threshold); order = report row order
METRICS = {
    # A1 -- original eleven
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
    # A1 -- added
    "ttc_true": (ttc_true, False, 3.0),
    "ttce": (ttce, False, 3.0),
    "dce": (dce, False, 2.0),
    "psd": (psd, False, 1.0),
    "btn": (btn, True, 1.0),
    "stn": (stn, True, 1.0),
    "a_lat_req": (a_lat_req, True, 2.0),
    "dst": (dst, True, 3.0),
    "pret": (pret, False, 1.5),
    # A2 -- collision probability
    "p_coll_gauss": (p_coll_gauss, True, 0.05),
    "p_coll_mc": (p_coll_mc, True, 0.05),
    # A3 -- calibrated fields
    "s_field_cal": (s_field_cal, True, 0.005),
    "o_field_cal": (o_field_cal, True, 0.005),
    "so_field_cal": (so_field_cal, True, 0.01),
    # A4 -- severity-weighted
    "delta_v": (delta_v, True, 5.0),
    "conflict_index": (conflict_index, True, 1e4),
    "p_pdrf": (p_pdrf, True, 1e-4),
    # A6 -- formal
    "rss_ds": (rss_ds, True, 0.5),
    "rss_margin": (rss_margin, True, 0.0),
}

# Metrics whose output is a probability in [0, 1], so calibration applies.
PROBABILITY_METRICS = frozenset({"p_coll_gauss", "p_coll_mc", "p_pdrf", "rss_ds"})

# Metrics evaluated on horizon prefixes for the earliness axis. Restricted to
# keep the 5x scoring cost bounded; see RISK_PREDICTION_METHODOLOGIES.md B4.
EARLINESS_METRICS = ("min_dist", "ttc_true", "drac", "btn", "p_coll_gauss", "so_field_cal")

# Every scale constant, recorded into each result JSON so a comparison can be
# reproduced and no parameter can be quietly tuned on the test split.
SCALE_CONSTANTS = {
    "A_MAX_BRAKE": A_MAX_BRAKE, "A_MAX_LAT": A_MAX_LAT, "T_SAFETY": T_SAFETY,
    "RHO_VEHICLE": RHO_VEHICLE, "MC_SAMPLES": MC_SAMPLES, "GAMMA_CI": GAMMA_CI,
    "DELTA_CI": DELTA_CI, "JOKSCH_V": JOKSCH_V, "RSS_RHO": RSS_RHO,
    "RSS_A_MAX": RSS_A_MAX, "RSS_B_MIN": RSS_B_MIN, "RSS_B_MAX": RSS_B_MAX,
    "RSS_MU_LAT": RSS_MU_LAT,
}


def is_collision(name, value, threshold=None):
    """True if `value` of metric `name` crosses its (or the given) threshold."""
    _, higher_is_riskier, default = METRICS[name]
    t = default if threshold is None else threshold
    return value > t if higher_is_riskier else value < t
