# Solutions: Paper vs Implementation

Companion to [PAPER_VS_IMPLEMENTATION.md](PAPER_VS_IMPLEMENTATION.md). That document lists the gaps between the STRAP paper and our
code. This one records, problem by problem, what we decided to do about each gap, the alternatives we rejected, and why.

As of 2026-10-20.

**Status legend**

| Status | Meaning |
| --- | --- |
| Decided | Solution agreed; not yet implemented |
| Implemented | Solution in the code and tested |
| Open | Not yet discussed; placeholder below |
| Deprioritised | Analysed; judged not important enough to change now |

**Summary**

| # | Problem | Status | Chosen solution |
| --- | --- | --- | --- |
| 1 | NGSIM recording periods are overlaid; no unique key | Implemented | Key = (location, period, frame_id) for time and (location, period, vehicle_id) for vehicles |
| 1b | Absolute lane id means different things per site | Implemented | Lane feature = target lane − vehicle lane at the same timestep (left +, right −) |
| 2 | γ_risk lets the model cheat the risk-scaled loss | Implemented | γ_risk from ground-truth future risk (no gradient path) |
| 3 | Intention codebook is trainable, and eval silently resets it | Implemented | Freeze the codebook as a buffer |
| 4 | Risk-field parameters are placeholders | Implemented | Full calibration from reference [22] (gap-based S-field, collision-only O-field); loss bias β = 0 |
| 5 | Validation and test are I-80 only | Implemented | Option B: scenes per recording, each recording split by time 70/10/20 with 80-frame guard gaps, then pool and shuffle each split |
| 6 | Encoder stacking order vs Fig. 2 | Implemented | Interleave: (spatial → temporal) block stacked N = 3 times, S, T, S, T, S, T |
| 7 | Target history positions are always 0 | Deprioritised | Keep per-timestep origin; lost lateral history changes 5 s lateral RMSE by ≈ 0.02–0.03 m |
| 8 | Lateral velocity is always 0 | Deprioritised | Keep vy = 0; same small lateral gain as problem 7 |
| 9 | Nearest-15 fallback / neighbour count | Implemented | Remove the fallback (0 neighbours if none passes 0.005); collision ground truth from all vehicles present, not only selected neighbours |
| 10 | Table II risk-level breakdown missing | Deprioritised | Not implemented for now; overall and per-site RMSE are reported instead |
| D | How us-101 and i-80 are used together | Implemented | One pooled model on all 6 recordings, shuffled; report overall + per-site RMSE |
| N | Novel objective: risk-appetite-constrained prediction | Open (partly scoped) | — |

**Implementation notes (2026-10-20)**

Problems 1, 1b, 2, 3, 4, 5, 6, 9 and D are in the code. Each has a check in `tests/test_solutions.py`, and the
`--debug` pipeline (prepare → train → eval → collisions) runs end to end on both sites. Choices the decisions left open:

- **Problem 4, speed in γx(v), αx(v):** the **ego (target) speed** is used, in `pair_risk` (`src/risk.py`). This is still to be checked against [22].
- **Problem 4, dimensions in the decoder:** each sample now carries raw `[length, width]` per vehicle (sample index 7, batch index 5).
  `STRAP(states, mask, dims)` passes them to the decoder's `endpoint_risk`. When dims are missing, a 4.5 × 1.8 m car is assumed.
- **Problem 2:** `src.risk.ground_truth_risk` gives the loss weight. `risk_scaled_loss` also detaches its risk input as a safeguard.
- **Problem 9, collision candidates:** the **nearest 15** vehicles present at frame 30 (`NGSIMDataLoader.collision_candidates`), by centre distance.
  Ground truth is the minimum true distance over the frames where the candidate is still recorded. Candidates with no future frame are dropped.
- **Paths:** checkpoints and logs moved to `strap_reproduction_v3`, so the old v2 results are kept and not overwritten.
- **Open issue found during testing (not one of the problems above):** in eval mode the predicted σ collapses to about 1e-5, so the
  validation NLL, and with it the validation loss, reaches 1e6–1e9 while validation RMSE is normal. The v2 run already showed this
  (val loss 8.6e9 → 8.8e6). `best_model.pt` is chosen by validation loss, so the selected epoch is unreliable until this is fixed.

---

## Problem 1: NGSIM recording periods are overlaid; no unique key

**Status:** Implemented

### Description
Each NGSIM site was recorded in three separate 15-minute periods. Every period restarts `frame_id` and reuses the same
`vehicle_id` values. Our loader identifies rows only by `(vehicle_id, frame_id)` (`NGSIMDataLoader.build_scenes`), so it stacks all three
periods on top of each other. It also groups vehicles by `vehicle_id` across both sites, so a us-101 car and an i-80 car with the same id are
merged into one record. Today this is harmless only because the two sites' frame ranges are offset and never overlap.

Measured on the cached data:

| Check (us-101) | Result |
| --- | --- |
| Recording periods (`global_time // 100 − frame_id`) | exactly 3 (1.18M / 2.11M / 1.52M rows; sums to the total) |
| Exact duplicate rows (same vehicle, same time) | 704,000 (14.7%) |
| Vehicle ids used by different cars in different periods | 2,178 of 2,847 |
| `(vehicle_id, frame_id)` keys holding 2+ different cars | 478,611 |
| Frames mixing periods | 9,945 of 9,996 |

Example: vehicle 434 at frame 2376 is two cars, one in lane 4 at 181 m (period 3) and one in lane 5 at 508 m (period 2).
That frame holds 445 cars, about 3× the real density.

**Consequences**
1. **Phantom neighbours.** Targets see cars from other periods. About 65% of candidates in the example frame were never on the road with the target.
2. **Lost samples.** Rows of two cars that share an id interleave, giving NaN histories that `build_samples` silently drops.
3. **Fake collisions.** The ground-truth minimum distance can be measured to a phantom car.

### Solution
Give every time step and every vehicle a unique key that includes the location **and** the recording period:

- **Time key** = (location, period, frame_id); **vehicle key** = (location, period, vehicle_id).
- A helper `_separate_recordings(df, location_number)` in `src/data/loader.py`, applied per location in `fetch_ngsim_multi`:
  1. `period` = dense rank of `global_time // 100 − frame_id` (exact; no guessing).
  2. Drop exact duplicates on (`vehicle_id`, `global_time`), first logging how many duplicate groups differ in any other column.
  3. Encode `vehicle_id = location_number·1,000,000 + period·100,000 + original_id` and keep `orig_vehicle_id`
     (e.g. 1200434 = us-101, period 2, vehicle 434).
  4. Offset `frame_id` per (location, period), with an 80-frame (one window) gap between segments, so no window straddles two recordings.
- `NGSIMDataLoader.fetch` always routes through `fetch_ngsim_multi`, so single-location and `--debug` runs get the same keys.
- The raw CSV cache is untouched, so no re-download is needed. All models must be retrained afterwards.

### Alternatives rejected

| Alternative | Why rejected |
| --- | --- |
| **(location, frame_id) only** | Separates the two sites but not the three periods inside each site, so the phantom cars, interleaved rows and fake collisions all remain. |
| **(location, global_time) as the clock** | Consecutive periods overlap in real time (us-101: 90 s and 84 s; i-80: 180 s between periods 2 and 3; none between 1 and 2), so plain `global_time` ranges cannot assign the overlap rows to a period. Re-measured 2026-10-20: only 3–10 cars per boundary come within 1 m of a car from the other period, and 0–7 pairs stay within 1 m for ≥ 3 s. The median nearest-car distance across periods is 140–190 m, so the overlap mostly covers different stretches of road. (An earlier figure of 808 / 6,768 "twin" cars was wrong.) Any twin that is merged into one scene still gives S-field ≈ 1 and a fake 0 m collision. Also, global_time spans April (i-80) to June (us-101), about 5×10⁷ tenths of a second, too large for the frame-mask array in `build_scenes`, and it would flip the temporal split. |
| **Keep only one period per site** | Fixes the problem but keeps only about 33–45% of the data, biases training towards one traffic state (congestion builds over the recording), and removes high-risk events. |
| **Re-download the original per-period NGSIM files** | A large rewrite of the download code. The original files also restart frame and vehicle ids, so the same key logic would still be needed. Its only advantage, exact period boundaries, we already get from `global_time` without error. |

### Why this solution
It removes all three consequences at the source (every key maps to exactly one car), keeps 100% of the non-duplicate data,
is a small change in one file that extends the per-location offsetting that already exists, and leaves all downstream code unchanged.

### Known costs
- A car still driving at a period boundary is split into two tracks, losing at most one 8 s window per boundary.
- The timeline has about 6 segments (≈ 55–60k frames); `build_scenes` peak memory rises to about 250 MB.
- The temporal split now runs over 6 segments; see problems 5 and D.
- The real-time overlap between periods is **kept** (user decision, 2026-10-20). Only ≤ 10 cars per boundary are recorded twice.
  Under problem 5 option B this is a tiny train/test leak (≤ ~30 cars in total); option B′ removes it by keeping those minutes out of train.
- Problem 1 is also what makes the split's 80-frame guard gap valid: before it, one `frame_id` stands for three moments, so the
  temporal cut cannot keep train and test apart in real time (see problem 5, "Current pipeline order").

---

## Problem 1b: The absolute lane id means different things at each site

**Status:** Implemented

### Description
Input feature 6 is the absolute NGSIM `lane_id` ÷ 2. The numbering differs per site: us-101 has lanes 1–8 (1–5 main, 6 auxiliary, 7 on-ramp,
8 off-ramp) and i-80 has lanes 1–7 (1 is HOV, 7 is on-ramp). Lane 5 is therefore not the same place on both roads, and the model has to learn
each site's numbering instead of the useful fact "same lane / one lane over".

### Solution
Make the lane feature relative to the target, at the same timestep:

```text
relative_lane(v, t) = target_lane(t) − vehicle_lane(v, t)
```

- NGSIM lane 1 is the leftmost lane, so a vehicle on the **left is positive** and on the **right is negative**: +2 = two lanes left, −1 = one lane right, 0 = same lane.
- The **target is always 0** (reference = its own lane at each timestep).
- **Raw subtraction**: ramp lanes are not remapped.
- No ÷2 scaling (z-scoring handles scale). The feature count stays 11. Implemented in `create_sample` (`src/data/loader.py`).

### Alternatives rejected

| Alternative | Why rejected |
| --- | --- |
| Reference = target lane at the **last observed frame** | Would keep the target's own lane change visible (−1 → 0), but the user chose a target that is 0 at every step for simplicity. Revisit with problems 7/8. |
| **Side only** (+1 any left lane, −1 any right lane) | Loses how far away the lane is; the signed gap is more informative. |
| Map ramp lanes to their lateral slot (us-101 6/7/8 → one slot) | More faithful, but needs per-site lane maps. Raw subtraction was preferred for simplicity. |
| Derive the lane gap from lateral position, round(Δy / 3.6 m) | Ignores `lane_id` entirely and duplicates feature 1 (Δy). |
| Keep the absolute lane for the target row only / add a lanes-to-left/right feature | Keeps the "where on the road" cue, but adds a target-specific encoding. Not chosen. |

### Why this solution
It is site-independent, directly encodes the relation that matters for interaction (same lane, adjacent, two over),
and is a one-line change with no new feature dimensions.

### Known limitations (accepted)
1. **The target's own lane change is invisible.** Target lane = 0, target relative position = 0 (problem 7) and lateral velocity = 0 (problem 8)
   together leave no input showing the target changing lanes. To be revisited with problems 7 and 8.
2. **Ramp gaps are overstated on us-101.** Lanes 6, 7 and 8 all sit at the same lateral position (x ≈ 19.4–20.6 m) on different
   stretches of road, so "8 − 5 = 3 lanes" is physically 1 lane. i-80 lane 7 is physically next to lane 6, so it is correct.
3. **The "where on the road" cue is lost.** Being in the leftmost/HOV lane or on an on-ramp that must merge is a strong lane-change cue.

---

## Problem 2: γ_risk lets the model cheat the risk-scaled loss

**Status:** Implemented

### Description
The risk-scaled loss (paper Eq. 13–14) multiplies each sample's loss by γ = exp(R_s + R_o). Our code takes R from the decoder's
`risk_field`, which is computed from the model's **own predicted** neighbour goals and the **trainable** intention codebook,
and never detaches it. The model can therefore lower its loss by predicting a scene that *looks* less risky, instead of predicting more accurately.

Evidence (untrained model, random inputs; it shows the gradient path exists, not a training result):
- The gradient on the goal predictor's output through γ was about 30× the gradient from prediction error (norm 192 vs 6.5).
- The gradient on the intention codebook was about 4,000× larger with γ attached (324 vs 0.08).
- Validation sets γ = 1, so this never shows in the validation loss.
- The code computes `exp(R − β)` while the paper says `max(exp(R) − β, 1)`. They agree only at β = 0.

### Solution
Compute γ from **ground-truth future risk**: the paper's own §III-E formula (S-field + O-field summed over neighbours), evaluated at the
**true** target end point (position and 1 s velocity at t = T + 5 s) against the **true** neighbour end points (`gt_goals`, masked by `goal_mask`).

- New helper `endpoint_risk(...)` in `src/risk.py`. The decoder's inline risk computation moves onto it, so the decoder (100 modes) and the loss (ground truth) share one formula.
- `risk_scaled_loss` uses the paper's form: γ = clamp(max(exp(R) − β, 1), max = 1e4).
- γ depends only on labels, so there is no gradient path and nothing to cheat. Validation uses `basic_loss`.

### Alternatives rejected

| Alternative | Why rejected |
| --- | --- |
| **A. Detach the predicted risk** (`risk_field.detach()`) | Removes the cheat with one line, but γ is still built from the model's own guesses: random early in training, changing every epoch, and averaged over 100 hypothetical end points (general crowding, not this sample's danger). |
| **B. Observed risk at the last history frame** | Stable and matches "risk perceived by the target", but only sees danger that exists *now*; cut-ins and braking during the next 5 s get no extra weight. Also needs raw (un-normalised) risk passed through the sample format. |

### Why this solution
It is the only option that removes the cheat, gives each sample a fixed weight, **and** weights the scenes whose real future is dangerous,
which is what the paper's Table II and the MTP collision objective care about. It needs no change to the data format,
because the true futures are already in every batch. Using labels in the loss is fine; γ is not needed at prediction time.

### Known limitation
Risk is measured only at the 5 s end point (as in the decoder). A near-miss at 2 s that resolves by 5 s is not weighted.
Weighting it would need neighbours' full future trajectories in the batch.

---

## Problem 3: The intention codebook is trainable, and evaluation silently resets it

**Status:** Implemented

### Description
The paper fixes its K = 100 intention modes as k-means centres of training end points (§III-E). Our `intention_means` is an `nn.Parameter`,
so Adam (and weight decay) keeps moving the modes during training. Then `load_trained` overwrites the trained modes with the original centres
from `intention_clusters.pkl`, so the model is evaluated with different modes from the ones it was trained with.

Evidence from saved checkpoints:
- `strap_reproduction_v2`: trained modes are on average 4.4 m (max 6.9 m) from their k-means centres.
- `ling_legacy`: average drift 321 m (max 782 m) against centres about 41 m from the origin. Either problem 2's cheat pushed the modes away, or the `.pkl` belongs to another run; either way, the evaluation was inconsistent.
- Related: the codebook's velocity is `(p[49] − p[40]) / 1.0`, which spans 0.9 s, so speed is understated by about 10%.

### Solution
- Store the modes with `register_buffer` instead of `nn.Parameter`. The state_dict key is unchanged, so old checkpoints still load.
- `load_trained` only checks that the `.pkl` matches the checkpoint's buffer and warns on mismatch; it never overwrites silently.
- Fix the codebook velocity to `(p[−1] − p[−11]) / 1.0 s`, the same helper as problem 2.

### Alternatives rejected

| Alternative | Why rejected |
| --- | --- |
| **B. Keep trainable, make train/eval consistent** (stop the overwrite, no weight decay) | Fixes the mismatch but not the drift. Not in the paper; nothing keeps modes near real end points, so the risk field can end up measuring danger at places nobody drives to. |
| **C. Fixed centres + small learned offset per mode** | More parameters and code with no evidence it helps; not in the paper. |

### Why this solution
It is the paper's design, fixes both the drift and the train/eval mismatch, keeps the risk field meaningful, and is a tiny change.

---

## Problem 4: Risk-field parameters are placeholders

**Status:** Implemented (2026-10-20): option A, full [22] calibration.

### Description
The paper gives **no values** for any risk-field parameter. What it does give (§III-B, §IV-A):

| Parameter | Paper |
| --- | --- |
| γx, γy (S-field scale) | only "> 1" |
| αx, αy (S-field shape) | only "≥ 2" |
| d\*, t\* (O-field scale) | "scaling factors", no value or limit |
| β1, β2 (O-field shape) | "shape factors", no value or limit |
| β (loss bias, Eq. 13) | "a pre-defined bias term", no value |
| Neighbour threshold | 0.005 (given) |
| Neighbours per vehicle | 6 on average, maximum Nv = 15 (given) |

(An earlier version of this section listed "β ≥ 1" as a paper limit; the paper does not state it.) We used placeholders: γx = γy = 1.5 m, α = 2, d* = 5 m, t* = 2 s,
β1 = β2 = 1, on centre-to-centre distances. Our O-field also gives risk to cars that are moving apart. These values drive input features 9–10,
neighbour selection, the decoder's risk field and γ_risk.

Measured on us-101 (1.33M target–car pairs within 150 m):
- **S-field is effectively dead:** above 0.005 in only 0.14% of pairs.
- **O-field measures closeness, not collision:** 15.6% of pairs light up, including cars moving apart.

### Source values: reference [22]
Zuo, Bian, Zuo, Ozbay 2025, *Composite Safety Potential Field for Highway Driving Risk Assessment* (arXiv 2504.21158), the field STRAP describes:

- **S-field:** distances are bumper-to-bumper **gaps**.
  - γx(v) = 5.1053e-4·v³ − 3.7051e-2·v² + 1.0621·v + 1.2925 (5.7 m at 5 m/s → 12.7 m at 25 m/s).
  - βx(v) = 2.2214e-5·v³ − 1.4834e-3·v² + 9.6673e-3·v + 3.2589 (≈ 3.3 → 2.8).
  - γy = 1.4310, βy = 4.9956.
- **O-field:**
  - d̂ₘ = perpendicular miss distance; t̂ₘ = −DᵀV / VᵀV.
  - d* = 0.5·(wᵢ + wⱼ), βp = 10, t* = 7.5 s, βt = 2.
  - Risk is 0 when the cars are moving apart.

### How the parameters affect the risk
- **Scales (γx, γy, d*, t*):** raising them raises risk everywhere; the danger zone grows.
- **Shapes (αx, αy, β1, β2):** raising them raises risk *inside* the scale and lowers it *outside*; the edge gets sharper. At distance = scale every shape gives e⁻¹ ≈ 0.37.
- S multiplies its longitudinal and lateral terms, so a car is risky only when close in both directions.
- Risk enters three places: summed as input features, compared with 0.005 for neighbour selection, and exponentiated in γ_risk.
  The exponential makes the loss weight very sensitive to the scales.

### Sensitivity sweep
us-101 period 1: 1.67M pairs, of which 770 are real conflicts (lateral overlap and bumper gap < 2 m within 5 s). One parameter is varied at a time around [22].
AUC measures how well the risk ranks the real conflicts.

| Setting | Avg neighbours | AUC | Mean loss weight |
| --- | --- | --- | --- |
| γx ×0.1 / ×0.3 / **×1** / ×3 / ×10 | 1.5 / 1.7 / **3.2** / 7.7 / 14.0 | 0.840 / 0.956 / **0.985** / 0.989 / 0.978 | 1.7 / 1.8 / **3.0** / 39 / 5,838 |
| d* 0.5 / **1.9** / 5 / 10 m | 2.9 / **3.2** / 4.8 / 7.4 | 0.981 / **0.985** / 0.975 / 0.960 | 2.7 / **3.4** / 35 / 266 |
| t* 1 / 2 / 4 / **7.5** / 15 s | 2.5 / 2.5 / 2.6 / **3.2** / 4.2 | 0.980 / 0.982 / 0.984 / **0.985** / 0.982 | 2.4 / 2.4 / 2.4 / **3.0** / 21 |
| αx 1 / 2 / **3** / 5 / 10 | 7.4 / 4.0 / **3.3** / 2.8 / 2.5 | 0.987 / 0.987 / **0.985** / 0.982 / 0.967 | ≈ 3 |
| Current values | 8.5 | 0.910 | 2.3 |
| Plain 1/(1 + distance) | 15 | 0.968 | — |
| Paper (§IV-A) | 6 | — | — |

**Reading:** neither extreme is "safe". Too small makes the risk blind (few neighbours, AUC falls). Too large makes everything look risky
(neighbours saturate at 15, the loss weight explodes, AUC falls). The [22] values sit at or near the best AUC for every parameter.
The current values rank real conflicts worse than plain distance.

### Options under consideration
| Option | Notes |
| --- | --- |
| A. Full [22] calibration | Cited source; S comes alive; O becomes true collision risk; 3.2 neighbours. |
| A, then re-tune γx after problem 8 | Same as A, with a check against the paper's 6 neighbours once lateral velocity exists. |
| [22] with γx ×3 | AUC 0.989 and 7.7 neighbours, but loss weights about 13× larger and no source for the factor. |
| B. [22] constants on centre distances | Rejected in analysis: gap-calibrated γy on centre distances makes adjacent lanes vanish. |
| C. Tune to 6 neighbours | One number cannot pin down eight parameters; arbitrary. |
| D. Keep current values | S stays dead; O stays a proximity measure. |

### Solution
**Option A: use the full calibration from [22], and keep the loss bias β = 0.**
- **S-field:** bumper-to-bumper gaps (from `v_length`, `v_width`), γx(v) and αx(v) as the speed-dependent polynomials above, γy = 1.4310, αy = 4.9956.
- **O-field:** perpendicular miss distance d̂ₘ, t̂ₘ = −DᵀV / VᵀV, d\* = 0.5·(wᵢ + wⱼ), β1 = 10, t\* = 7.5 s, β2 = 2, and risk = 0 when the cars are moving apart.
- **Thresholds kept from the paper:** 0.005 for neighbour selection, Nv = 15.
- **β (loss bias) = 0.** This is the neutral choice; with problem 2's paper formula, γ = max(exp(R) − β, 1). It can be swept later as an ablation.
- Apply it in one place (`src/risk.py`), so input features 9–10, neighbour selection, the decoder's endpoint risk and γ_risk all use the same field.

**How the values were decided** (the paper is silent, so this is our assumption, stated as such):
1. **Inherit from the source of the field.** [22] defines this exact S-field / O-field pair and calibrates it on highway data.
   Two of its authors (Zilin Bian, Dachuan Zuo) are also STRAP authors, so [22]'s values are the most likely ones STRAP used.
2. **Consistent with the paper's limits:** γx = 5.7–12.7 m and γy = 1.43 are > 1; the shapes 2.8–3.3 (longitudinal) and 5.0 (lateral) are ≥ 2.
3. **Checked against the paper's indirect target:** the 0.005 threshold gives 3.2 neighbours on average, against the paper's 6.
   Reported as a known gap, not tuned away (see the costs below).
4. **Validated on training data only, never on test:** the sensitivity sweep above puts [22] at or near the best AUC for every parameter,
   while the current placeholders rank real conflicts worse than plain distance (0.910 vs 0.968). The sweep is reported as a sensitivity analysis.

**Citation note:** §III-B says "we adopt the safety potential field model … from [21]" (Wang et al. 2022), but the S-field / O-field names and both
equations match [22]; [21] is a different, prediction-based risk metric. This is treated as a probable citation slip, and [22] is cited for the values.

### Alternatives rejected
- **A, then re-tune γx after problem 8:** no longer applies, because problem 8 is deprioritised (vy stays 0).
- **[22] with γx ×3:** the AUC gain is only 0.004, loss weights become about 13× larger, and the factor has no source.
- **B (centre distances):** gap-calibrated γy on centre distances makes adjacent lanes vanish.
- **C (tune to 6 neighbours):** one number cannot pin down eight parameters, so the result would be arbitrary.
- **D (keep current values):** the S-field stays dead and the O-field stays a proximity measure.

### Why this solution
It is the only option with a source: the field's own calibration, by overlapping authors. It also has the best or near-best conflict ranking
in our sweep and moderate loss weights (mean γ ≈ 3.0), and it needs no tuning against a target we cannot pin down.

### Known costs and checks (accepted)
- **3.2 neighbours vs the paper's 6.** Log it under problem 9, together with how often the nearest-15 fallback fires.
- **Gaps need vehicle dimensions everywhere risk is computed.** The decoder's endpoint risk uses predicted goals, which carry no dimensions,
  so the target's and neighbours' length and width must be passed to it.
- **Check which speed γx(v) and αx(v) use** in [22] (ego speed or relative speed) before coding it.
- **With vy = 0 (problem 8), the O-field only sees same-lane closing.** Accepted under problem 8.
- **Retrain afterwards:** risk features, neighbour selection and γ all change. Do it together with problems 1, 2, 3 and 5, so there is one retrain.
- Linked to the novel objective (N), where the appetite may scale these parameters.

---

## Problem 5: Validation and test are I-80 only

**Status:** Implemented (2026-10-20): option B.

### Description
The two sites are chained on one timeline and split 70/10/20 by time, so validation and test come entirely from I-80.
Everything below assumes problem 1's unique keys. The timeline then has 6 (location, period) segments in this order:
us-101 p1, p2, p3, then i-80 p1, p2, p3, with 80-frame gaps between segments.

Measured on the cached data (unique frames per period):

| Site | p1 | p2 | p3 |
| --- | --- | --- | --- |
| us-101 | 9,529 (07:50–08:06) | 9,991 (08:04–08:21) | 9,948 (08:19–08:36) |
| i-80 | 10,013 (15:59–16:16) | 9,787 (16:59–17:16) | 11,685 (17:13–17:32) |

Total: 60,953 frames, about 49% us-101 and 51% i-80. Real-time overlaps between consecutive periods: us-101 90 s and 84 s,
i-80 180 s (p2→p3) and none between p1 and p2.

### Current pipeline order, and whether shuffling leaks
1. `build_scenes` cuts the 8 s windows, and `build_samples` turns each window into one sample.
2. `get_splits` makes the temporal cut and drops every window within 80 frames (one window) of each cut, **for all vehicles**.
3. Normalisation statistics and the k-means codebook are fitted on **train only**.
4. **The training set alone is shuffled**, every epoch (`train.py`, `shuffle=True`). Val and test are not shuffled, which is correct because order does not affect RMSE.

**Shuffling does not leak.**
- The model is stateless between samples: its input is only that window's 30 history frames, and no hidden state carries over between batches.
- Shuffling reorders whole samples, not the frames inside them, so each window keeps its time order.
- Inside train, the model does see other windows' futures: window A's future is window B's history, and a target's future is a neighbour-goal label in another target's scene. That is allowed, because all of it is training data. Leakage only means information crossing from train into val or test.
- The real side effect of overlapping windows (stride 1 s, length 8 s) is **correlation**: the effective sample count is lower than the number of windows. Shuffling helps here, because without it a batch would be near-copies of one car.
- The shuffle order is not reproducible: `SEED` only reaches k-means, and there is no `torch.manual_seed`.

**The leak risk is at the split boundary.** The 80-frame guard gap means no raw frame of any car is in two splits, so a test
target's future cannot appear in training as another car's history or neighbour goal. This holds only if a `frame_id` means one moment.
- **Before problem 1** it does not. The three periods share `frame_id`s, so the cut puts the start of every i-80 period in train and the end of every period in test.
  The end of i-80 p2 (test) runs in the same real time as the start of p3 (train) for 180 s. Only about 10 cars are recorded in both,
  but the traffic state on both sides of the split is the same.
- **After problem 1** each period has its own frame range, so the guard gap works again. What is left depends on the option below.

### Option A: one global temporal cut (current `get_splits`, unchanged)
- The cuts fall at 70% (frame 42,667) and 80% (frame 48,762) of the timeline. Both land inside i-80 p2, at about 33% and 95% of it.
- **Train** = all of us-101 + i-80 p1 + the first ~5 min of i-80 p2 (i-80 is about 31% of train).
  **Val** = ~10 min of i-80 p2 (about 17:05–17:15).
  **Test** = the last ~50 s of i-80 p2 + all of i-80 p3 (17:13–17:32, the PM peak).
- **Leakage after problem 1:**
  - **Train vs test: none.** Train ends around 17:04 and test starts around 17:14.
  - Val (the end of p2) shares about 2 min of real time with the start of test (p3). Val only picks the best epoch, so this is minor.
- **Pros:** no code change. Test is strictly later than train. It is a hard, honest test of transfer to a new period and a partly new site.
- **Cons:**
  - Val and test are i-80 only, and test is a single congested period.
  - RMSE mixes site shift with congestion and is not comparable to the paper's NGSIM-wide number.
  - Model selection (val) is tuned to i-80, and us-101 is never evaluated.

### Option B: cut each recording 70 / 10 / 20
- Each of the 6 segments is cut at 70% and 80% of its own frames, with the existing 80-frame guard gap after each cut.
  **Train** = the first ~10.5 min of every period, **val** = the next ~1.5 min, **test** = the last ~3 min.
- This is close to the standard NGSIM protocol: as recalled, CS-LSTM splits each recording 70/10/20 by vehicle id, which STDAN reuses.
  Verify this in CS-LSTM's `preprocess_data.m` before citing it.
- **Leakage after problem 1:** the end of period k (test) runs in the same real time as the start of period k+1 (train), for 84–180 s per boundary.
  At most ~10 cars per boundary (≤ ~30 in total) appear in both, plus the same traffic state at that moment.
- **Pros:**
  - Test covers both sites (about 49% / 51%) and all 6 periods.
  - Per-site RMSE can be reported directly.
  - Train, val and test come from the same mix of sites and traffic states.
  - Closes TODO #5 in [PAPER_VS_IMPLEMENTATION.md](PAPER_VS_IMPLEMENTATION.md) §10.
- **Cons:**
  - Test follows train by only ~1.5 min inside the same period. Traffic states (e.g. one congestion wave) are correlated, so the score is optimistic.
  - The guard gaps grow from 2 to 12, losing about 1.6% of frames instead of 0.3%.
  - `get_splits` must cut per segment, using `location` + the `period` column that problem 1 keeps.

### Option B′: option B, with the real-time overlap kept out of train
- As B, but the first 84–180 s of each period k+1 that run at the same real time as period k are **excluded from train**.
  Windows in those minutes are dropped if they would fall in train, and kept only if they fall in val or test.
- **Leakage after problem 1: none** between train and test. No moment of real time is on both sides.
- **Cost:** about 1–3 min of training data per site (≈ 3% of train), plus a few lines in `get_splits` to mark overlap frames
  (compare each period's `global_time` range with the previous period's).
- It refines problem 1's "keep the overlap" decision: the overlap is kept, but not in train.

### Option C: random split of the built scenes (rejected)
Question: after the scenes are built, does it matter if train / val / test are drawn as a random sample? **Yes. It leaks the answer.**
- **Same-target overlap.** A window is 80 frames and the stride is 10, so each window shares frames with up to 7 earlier and
  7 later windows of the same target. With a random 70/10/20 draw, the chance that none of those 14 is in train is about
  0.3¹⁴ ≈ 5×10⁻⁸. In practice every test window has a near-copy in train.
- **Labels become inputs.** The window that starts 30 frames later uses this window's future frames 30–59 as its history.
  So test labels are literally training inputs.
- **Cross-vehicle leak.** A test target also appears as a neighbour in other targets' training windows at the same time,
  and its 5 s end point is a training label there (neighbour goal Ĝ).
- **Effect:** test RMSE measures memorisation, not prediction. It is optimistic and not comparable to a held-out score.
  This is the reason given in PAPER_VS_IMPLEMENTATION.md §1 ("Split method").
- This is different from shuffling **after** the split, which the code already does and which is safe (see above).

**Proposal considered (2026-10-20): build scenes per recording, pool them, shuffle, then split randomly.**
- **Part 1 is good.** Building scenes independently for each (location, period) is problem 1's fix. It removes phantom
  neighbours and id collisions, and should be kept.
- **Part 2 leaks.** A random split of the pooled scenes is option C. The leak is not *between* recordings but between windows
  that overlap in time *inside* one recording, so building scenes per recording does not remove it. All three leaks above still apply:
  near-copies of the same car, test labels as training inputs, and test cars as neighbours with goal labels in training scenes.
- **Fixes that do not work:**
  - Non-overlapping windows (stride 80) remove the near-copies, but the cross-vehicle leak stays: another car's window covers the same seconds as this car's future.
  - A random split by vehicle removes the same-car leaks, but test cars still appear as neighbours in training scenes.
- **Correct version of the idea:**
  1. Build scenes per recording.
  2. Split each recording **by time**, 70/10/20, with the 80-frame guard gap (option B, or B′).
  3. Pool the 6 train parts and shuffle them, and pool val and test the same way.

  This keeps the goal of the proposal (every split contains both sites and all periods, mixed together) without leakage.
  Random shuffling is safe only **after** the time-based split, never instead of it.

Weaker random variants:

| Variant | Verdict |
| --- | --- |
| Random split **by target vehicle** (all windows of a car in one set) | Removes the same-target overlap, but keeps the cross-vehicle leak (a test car is a neighbour with a goal label in training scenes) and the shared traffic state at the same instant. Still leaky. |
| Random **time blocks** (e.g. 1-minute chunks with 80-frame guard gaps) | Leak-free and covers all recordings, but it is option B with many more gaps (more lost frames, train and test even closer in time). |

### Comparison

| | A: global cut | B: per recording | B′: B without overlap in train | C: random |
| --- | --- | --- | --- | --- |
| Test sites | i-80 only | us-101 + i-80 | us-101 + i-80 | both |
| Test periods | i-80 p3 (+ end of p2) | all 6 | all 6 | all 6 |
| Code change | none | `get_splits` per segment | B + overlap mask | small |
| Train–test leakage (after problem 1) | none (val–test ~2 min, minor) | ≤ ~30 overlap cars | none | severe (near-copies, labels as inputs) |
| Training data lost | ~0.3% (2 gaps) | ~1.6% (12 gaps) | ~1.6% + ~3% overlap | — |
| Score | hardest (time + site shift) | optimistic (same period) | optimistic (same period) | invalid (memorisation) |
| Comparable to paper's NGSIM number | no | closer (standard protocol) | closer | no (inflated) |

### Solution
**Decision: option B.** Problem 1 must be implemented first.

1. **Build scenes per recording.** Each (location, period) gets its own scenes, using problem 1's unique keys.
2. **Split each recording by time, 70 / 10 / 20.** Cut each of the 6 segments at 70% and 80% of its own frames, and drop
   every window within 80 frames (one window) of a cut, for all vehicles.
3. **Pool and shuffle.** Pool the 6 train parts into one training set, and pool val and test the same way.
   Train is shuffled every epoch (already the case in `train.py`). Val and test order does not matter.
4. Fit normalisation statistics and the k-means codebook on the pooled train set only (unchanged).
5. Report RMSE overall **and per site** (see D).

Code change: `get_splits` cuts per segment (grouped by `location` + `period`) instead of over one global timeline.
The 80-frame guard gap logic is reused unchanged inside each segment.

### Alternatives rejected

| Alternative | Why rejected |
| --- | --- |
| A: one global temporal cut | Val and test are i-80 only and test is one congested period, so RMSE is an i-80 PM-peak number, not comparable to the paper. |
| B′: B with the real-time overlap kept out of train | Removes the ≤ ~30-car overlap leak, but costs ~3% of train and extra code. Not chosen, in line with problem 1's "keep the overlap" decision; it remains the upgrade path if the leak matters. |
| C: random split of pooled scenes | Leaks the answer (near-copies, labels as inputs, cross-vehicle goals). See above. |

### Why this solution
- Every split contains both sites and all 6 periods, so the test score covers all of NGSIM, as the paper's does.
- It is close to the standard NGSIM protocol (CS-LSTM / STDAN split each recording 70/10/20; to verify before citing), so results are comparable with the baselines.
- Splitting by time keeps the 80-frame guard gap valid, so there is no train/test leak except the accepted ≤ ~30 overlap cars.
- Shuffling after the split gives mixed batches with no leakage.

### Known costs (accepted)
- Test follows train by only ~1.5 min inside each period, so traffic states are correlated and the score is somewhat optimistic.
- 12 guard gaps instead of 2: about 1.6% of frames lost instead of 0.3%.
- ≤ ~10 cars per period boundary (≤ ~30 in total) can appear in both train and test, because of the kept real-time overlap.

---

## Problem 6: Encoder stacking order vs Fig. 2

**Status:** Implemented (2026-10-20)

### Description
We run 3 spatial layers, then 3 temporal layers (S, S, S, T, T, T) in `STRAP.forward` (`src/models/strap.py`).

The paper never says "interleaved" or "alternating", but its text describes one spatial-temporal block that is repeated (§III-D):
- The temporal encoder's input is the **spatial encoder's output**: H̄s = Hs + PE (Eq. 5).
- The temporal output C is "the final output of the spatial-temporal encoder" (Eq. 7), so one spatial + one temporal layer form one encoder.
- *"The spatial and temporal encoder layers can be stacked N times … where N represents the number of encoding layers."*
- §IV-A: *"The model has 3 spatial and temporal encoder layers"*, so N = 3.

Stacking that block 3 times gives S → T → S → T → S → T. Our order contradicts Eq. 5: the 2nd and 3rd temporal layers receive a temporal output, not a spatial one.

**Strength of evidence:** implied by Eqs. 5–7 and the stacking sentence, not stated explicitly.
Fig. 2 may show the arrangement directly, but it is an image and was not checked here. Look at it before calling this confirmed.

### Solution
**Interleave the layers.** One loop over i = 0…2: `H = spatial_encoders[i](H, mask)`, then `H = temporal_encoders[i](H, mask)`.
- The parameter count is unchanged (same 3 + 3 layers), and both `ModuleList`s and their state_dict keys stay as they are.
- Each temporal layer adds the sinusoidal PE to its input (`temporal_encoder.py`), which now happens once per block. That matches Eq. 5 applied per stacked block.
- Update the module docstring in `strap.py`, which currently describes "(x3)" per encoder.

### Alternatives rejected

| Alternative | Why rejected |
| --- | --- |
| Keep S, S, S, T, T, T | Contradicts Eq. 5 for the stacked layers, and is not what "the spatial and temporal encoder layers stacked N times" describes. |
| Add PE only once, before the first temporal layer | Eq. 5 is written per temporal encoder. Adding PE again to an input that already carries it is harmless and simpler to keep as is. |

### Why this solution
It is the most natural reading of the paper's equations and stacking sentence, it is a small change in one function, and it costs no extra parameters.
Interleaving also lets later spatial layers attend over features that already contain temporal context, which is the stated aim ("more hierarchical and abstract representations").

### Known costs
- Changes the model: old checkpoints load, but they were trained with the other order, so their results do not carry over. Retrain together with problems 1–5.
- Update `PAPER_VS_IMPLEMENTATION.md` §6 (the "3 S, then 3 T" note) and register item #6 once it is implemented.

---

## Problem 7: Target history positions are always 0

**Status:** Deprioritised (user decision, 2026-10-20)

### Description
Positions are relative to the target at each timestep, so the target row is always (0, 0) and its own path history is lost.
With problem 1b, the target's lane is also always 0.

What is actually lost:

| Target information | Still visible? |
| --- | --- |
| Longitudinal motion | **Yes.** Speed and acceleration are inputs at all 30 history frames, so the along-road path is effectively kept. |
| Lateral motion (drift, lane change in progress) | **No.** Δy = 0, vy = 0 (problem 8) and relative lane = 0 (problem 1b). Only indirect: all neighbours' Δy shift together. |

### Measurement
Raw data, every 8 s window (stride 1 s), separated by recording (problem 1 keys). "With history" = best linear extrapolation of the target's last 1 s of lateral movement.

| | us-101 | i-80 |
| --- | --- | --- |
| Windows | 364,412 | 414,305 |
| Target changes lane in history / in future | 1.9% / 2.7% | 1.8% / 2.9% |
| History lateral move > 0.5 m | 9.3% | 11.6% |
| Correlation between history and future lateral move | 0.23 | 0.32 |
| Lateral RMSE at 5 s, all windows: without → with history | 0.56 → 0.54 m | 0.59 → 0.56 m |
| Lateral RMSE at 5 s, future lane-change windows: without → with | 2.46 → 2.30 m | 2.48 → 2.23 m |
| Longitudinal RMSE at 5 s from speed alone (for scale) | 6.94 m | 6.19 m |

### Solution
**Keep the per-timestep origin.** The lost lateral history changes the overall 5 s error by only about 0.02–0.03 m, against
about 6 m of longitudinal error, so it barely affects RMSE.

### Alternatives rejected

| Alternative | Why rejected |
| --- | --- |
| Fixed origin (target at frame 30) for the history | Recovers the target's lateral path, but the measured gain is too small to be worth a feature change and retraining now. |

### Known cost (accepted)
- Lane-change windows (≈ 3%) lose 7–10% of lateral accuracy at 5 s. These cut-ins are the high-risk cases that Table II and the collision evaluation focus on.
- The numbers above are a lower bound: linear extrapolation is crude and NGSIM `local_x` is noisy, so the network might use the history better.
- Revisit if lane-change or Table II results turn out weak.

---

## Problem 8: Lateral velocity is always 0

**Status:** Deprioritised (user decision, 2026-10-20)

### Description
NGSIM's `v_vel` is a scalar speed along the lane, so feature 3 (vy) is always 0 and the O-field only sees same-lane closing.

**The dataset does not provide v_y.** NGSIM has no lateral-velocity column: `v_vel` (speed) and `v_acc` (acceleration) are
scalars along the direction of travel. So lateral velocity cannot be taken from the data as given. The only way to get it would be
to derive it from positions (Δ`local_x` / Δt), which needs smoothing because NGSIM positions are noisy (see the alternative below).

### Solution
**Keep vy = 0.** For the target, a real lateral velocity would recover the same lateral signal measured in problem 7.
That gain is small: about 0.02–0.03 m of 5 s lateral RMSE overall, and 7–10% only on the ≈ 3% of windows with a lane change.

### Alternatives rejected

| Alternative | Why rejected |
| --- | --- |
| vy = Δ`local_x` / Δt (smoothed), for all vehicles | Same small prediction gain as problem 7; not worth a feature change and retraining now. |

### Known cost (accepted)
- The O-field still has no lateral relative velocity, so a neighbour cutting in from the next lane only registers through
  S-field proximity, not as a closing conflict. This affects neighbour selection and the risk features, not only the target's lateral input.
- The decoder's predicted neighbour goals keep vy ≈ 0 (ground truth vy is always 0).
- Revisit together with problem 7 if lane-change, Table II or collision results turn out weak, and before re-tuning problem 4's risk parameters.

---

## Problem 9: Nearest-15 fallback / neighbour count

**Status:** Implemented (2026-10-20)

### Description
When no vehicle passes the 0.005 risk threshold, the loader falls back to the 15 nearest vehicles (`build_scenes`, `src/data/loader.py`).
The paper's rule would reject these vehicles: *"We consider the vehicles with either S-field risk or O-field risk greater than 0.005 as interacting
with the target vehicle and select them as its neighborhood vehicles"* (§IV-A). The fallback also inflates the average neighbour count we compare with the paper's 6.

### Solution
**Remove the fallback.** If no vehicle passes the threshold, the scene has **0 neighbours** (target only). Otherwise keep the risk-ranked selection, capped at Nv = 15.

The model already handles target-only scenes. They occur today when no candidate is present (`selected_ids = []`), so no new code path is introduced:
- Spatial attention sees only the target itself (one valid key, no all-masked rows).
- Goal loss is 0 for that sample (mask sum 0, denominator clamped to 1 in `goal_loss`).
- Decoder risk is 0, so the mode weights are uniform and γ = 1, which is correct for a risk-free scene.

**Collision evaluation fix (required with this change).** Collision pairs and their ground-truth minimum distance are currently built only
from the selected neighbours (`rollout_pairs`, `gt_min_dist` in `src/pipeline.py`). Without the fallback, a car whose risk is ≤ 0.005 at frame 30
but which later comes dangerously close would never be evaluated, and the test would only check conflicts the risk field already flagged.
- Build the collision candidate set and its ground truth from **all vehicles present at frame 30** (or the nearest 15), independently of the model's neighbour selection.
- Keep the model's inputs unchanged (risk-selected neighbours only).

**Checks to add:**
1. Log the share of samples with 0 neighbours and the average neighbour count, against the paper's 6 (≈ 3.2 expected with problem 4's [22] values).
2. Report test RMSE separately for zero-neighbour samples.
3. One small test: a batch in which every sample has 0 neighbours (`max_N = 1`, goal tensors `(B, 0, 4)`) passes through `model()` and `risk_scaled_loss` without errors or NaN.
   This is unlikely at batch size 128, but possible with `--debug`.

### Alternatives rejected

| Alternative | Why rejected |
| --- | --- |
| Keep the nearest-15 fallback | Adds vehicles the paper's rule rejects, inflates the neighbour count, and feeds non-interacting cars into the risk features and the decoder's risk field. |
| Fallback to fewer nearest vehicles (e.g. nearest 1–3) | Still not the paper's rule; any k is arbitrary. |
| Lower the threshold until every scene has a neighbour | Changes the paper's given 0.005 and makes every scene look interacting. |

### Why this solution
It is exactly the paper's selection rule, it needs only the fallback branch deleted, and the model already supports target-only scenes.
Decoupling the collision ground truth from neighbour selection keeps the collision evaluation honest: it measures all real conflicts,
not only those the risk field already flagged.

### Known costs (accepted)
- Less context in calm scenes: a leader far ahead in free flow is no longer an input. This is the paper's design, and constant velocity already does well in such scenes.
  The zero-neighbour RMSE check above shows whether it hurts.
- The collision evaluation needs its own candidate set, a small change in `src/pipeline.py`.

---

## Problem 10: Table II risk-level breakdown missing

**Status:** Deprioritised (user decision, 2026-10-20)

### Description
The paper reports RMSE for "collision in 1 s / 2 s / 3 s / 5 s" and non-collision scenes, but does not define how a scene is labelled.

### Solution
**Not implemented for now.** The main results are overall and per-site RMSE (problems 5 and D), plus the collision evaluation (problem 9).

### Known cost (accepted)
- No comparison with the paper's Table II, the table that shows STRAP-R's gain in high-risk scenes.
- Revisit if the risk-scaled loss (STRAP-R vs STRAP-B) needs to be justified on high-risk scenes. Doing so first needs our own labelling
  definition of "collision in N s" (e.g. ground-truth gap below a threshold within N s), stated as an assumption.

---

## D: How us-101 and i-80 are used together

**Status:** Implemented (2026-10-20)

### Description
How to train and evaluate with two freeway sites: pooled or per site, in what order, and how results are reported.
How the 6 (location, period) segments are split is problem 5.

Current code: both sites are concatenated into one table (`fetch_ngsim_multi`, with frame ids offset per site), one model is
trained on the mix, train batches are shuffled, normalisation is fitted on train only, and a single overall RMSE is reported.

### Solution
**Pooled training: one model on all 6 recordings of both sites.**
1. Combine the 6 per-recording train parts from problem 5 into one training set. Shuffle it every epoch (already the case).
2. Fit normalisation statistics and the k-means intention codebook on the pooled train set (already the case).
3. Do not balance the sites and do not add a site-id feature. The sites are already about 49% / 51% by frames, and
   problem 1b's relative lane removes the main site-specific input.
4. Report test RMSE **overall and per site** (us-101, i-80), and optionally per period.
5. Optional extra experiment, kept separate from the main result: cross-site generalisation (train on us-101, test on i-80, and vice versa).

### Alternatives rejected

| Alternative | Why rejected |
| --- | --- |
| Per-site models | Halves the training data, and the paper reports one NGSIM model and number. |
| Sequential training (one site, then fine-tune on the other) | Not standard for NGSIM. Fine-tuning drifts the model towards the last site and partly forgets the first (catastrophic forgetting). This only suits adapting to one target domain (e.g. HighD → NGSIM), which is not our goal. |
| Site balancing / site-id input | Not needed with a ~49/51 mix; a site id would also let the model learn site-specific shortcuts. |

### Why this solution
Pooled training is the NGSIM standard (CS-LSTM and the models that reuse its preprocessing, including STDAN, which STRAP compares against).
It uses all the data, matches what the paper reports, and needs no code change beyond problems 1 and 5.
Per-site RMSE is a small addition (`evaluate_trajectory` grouped by location) that shows whether one site is harder.

---

## N: Novel objective, a risk-appetite-constrained prediction model

**Status:** Open (partly scoped)

### Description
After a faithful STRAP reproduction, build a **risk-constrained** model whose predictions follow a driver's **risk appetite**,
carrying over the method of the user's IEEE TAI 2026 paper (*A CVaR-Constrained Safe RL Framework with Action Repair for Practical
Portfolio Optimization*, DOI 10.1109/TAI.2026.3686783): a CVaR cost ≤ d (appetite), an augmented Lagrangian (λ, ρ updates) and action repair.

### Decided so far
- The appetite is **given as a user input**.
- The appetite is a **continuous scalar** (e.g. a ∈ [0, 1]).

### Working mapping (not final)
| Portfolio paper | Trajectory prediction |
| --- | --- |
| Assets | The 100 intention modes (candidate destinations) |
| Portfolio weights (≥ 0, sum to 1) | Mode weights (same simplex) |
| Return | Prediction accuracy (MSE + NLL) |
| CVaR of loss | CVaR of S/O risk over the weighted mode mixture |
| d = investor appetite | d(a) = driver appetite |
| Augmented Lagrangian on PPO | Augmented Lagrangian on the supervised loss (risk is differentiable) |
| Action repair | Mode repair: top-K, weight bounds, renormalise; risk ≤ d(a) |

### Open questions
1. Whose appetite: the target driver (exact analogue: the trajectory is the driver's "action") or the AV using the predictions (a cautious setting must make predictions *more* pessimistic)?
2. Supervised training + augmented Lagrangian, or RL (PPO with actor–critic–cost) as in the paper?
3. Does the appetite change only the threshold d, only the risk-field parameters θ, or both?
4. One model conditioned on a, one model per a (as in the paper's Fig. 4), or repair only at inference?
5. Mode weights are currently softmax(−risk) with no likelihood term; a learned per-mode probability head is likely needed.

### Solution
_To be decided._

### Alternatives rejected
_To be decided._

### Why this solution
_To be decided._
