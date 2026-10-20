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

**Summary**

| # | Problem | Status | Chosen solution |
| --- | --- | --- | --- |
| 1 | NGSIM recording periods are overlaid; no unique key | Decided | Key = (location, period, frame_id) for time and (location, period, vehicle_id) for vehicles |
| 1b | Absolute lane id means different things per site | Decided | Lane feature = target lane − vehicle lane at the same timestep (left +, right −) |
| 2 | γ_risk lets the model cheat the risk-scaled loss | Decided | γ_risk from ground-truth future risk (no gradient path) |
| 3 | Intention codebook is trainable, and eval silently resets it | Decided | Freeze the codebook as a buffer |
| 4 | Risk-field parameters are placeholders | Open (analysis done) | — |
| 5 | Validation and test are I-80 only | Open (options written up) | A: one global cut, or B: cut each recording 70/10/20; random split rejected |
| 6 | Encoder stacking order vs Fig. 2 | Open | — |
| 7 | Target history positions are always 0 | Open | — |
| 8 | Lateral velocity is always 0 | Open | — |
| 9 | Nearest-15 fallback / neighbour count | Open | — |
| 10 | Table II risk-level breakdown missing | Open | — |
| D | How us-101 and i-80 are used together | Open | — |
| N | Novel objective: risk-appetite-constrained prediction | Open (partly scoped) | — |

---

## Problem 1: NGSIM recording periods are overlaid; no unique key

**Status:** Decided

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
  Under problem 5 option B this is a tiny train/test leak (≤ ~30 cars in total).

---

## Problem 1b: The absolute lane id means different things at each site

**Status:** Decided

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

**Status:** Decided

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

**Status:** Decided

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

**Status:** Open. Analysis done, decision pending.

### Description
The paper only gives limits (γx, γy > 1; αx, αy ≥ 2; β ≥ 1), not values. We used placeholders: γx = γy = 1.5 m, α = 2, d* = 5 m, t* = 2 s,
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
| A. Full [22] calibration | Cited source; S comes alive; O becomes true collision risk; 3.2 neighbours now (expected to rise after problem 8). |
| A, then re-tune γx after problem 8 | Same as A, with a check against the paper's 6 neighbours once lateral velocity exists. |
| [22] with γx ×3 | AUC 0.989 and 7.7 neighbours, but loss weights about 13× larger and no source for the factor. |
| B. [22] constants on centre distances | Rejected in analysis: gap-calibrated γy on centre distances makes adjacent lanes vanish. |
| C. Tune to 6 neighbours | One number cannot pin down eight parameters; arbitrary. |
| D. Keep current values | S stays dead; O stays a proximity measure. |

**Decision:** _pending_. Linked to the novel objective (N), where the appetite may scale these parameters.

---

## Problem 5: Validation and test are I-80 only

**Status:** Open. Options A and B written up, decision pending; option C (random split) rejected.

### Description
The two sites are chained on one timeline and split 70/10/20 by time, so validation and test come entirely from I-80.
Everything below assumes problem 1's unique keys. The timeline then has 6 (location, period) segments in this order:
us-101 p1, p2, p3, then i-80 p1, p2, p3, with 80-frame gaps between segments.

Measured on the cached data (unique frames per period):

| Site | p1 | p2 | p3 |
| --- | --- | --- | --- |
| us-101 | 9,529 (07:50–08:06) | 9,991 (08:04–08:21) | 9,948 (08:19–08:36) |
| i-80 | 10,013 (15:59–16:16) | 9,787 (16:59–17:16) | 11,685 (17:13–17:32) |

Total: 60,953 frames, about 49% us-101 and 51% i-80.

### Option A: one global temporal cut (current `get_splits`, unchanged)
- The cuts fall at 70% (frame 42,667) and 80% (frame 48,762) of the timeline. Both land inside i-80 p2, at about 33% and 95% of it.
- **Train** = all of us-101 + i-80 p1 + the first ~5 min of i-80 p2 (i-80 is about 31% of train).
  **Val** = ~10 min of i-80 p2 (about 17:05–17:15).
  **Test** = the last ~50 s of i-80 p2 + all of i-80 p3 (17:13–17:32, the PM peak).
- **Pros:** no code change. Test is strictly later than train. It is a hard, honest test of transfer to a new period and a partly new site.
  No cut lies at a period boundary, so the cars recorded twice in the overlaps cannot cross splits.
- **Cons:**
  - Val and test are i-80 only, and test is a single congested period.
  - RMSE mixes site shift with congestion and is not comparable to the paper's NGSIM-wide number.
  - Model selection (val) is tuned to i-80, and us-101 is never evaluated.

### Option B: cut each recording 70 / 10 / 20
- Each of the 6 segments is cut at 70% and 80% of its own frames, with the existing 80-frame guard gap after each cut.
  **Train** = the first ~10.5 min of every recording, **val** = the next ~1.5 min, **test** = the last ~3 min.
- **Pros:**
  - Test covers both sites (about 49% / 51%) and all 6 periods.
  - Per-site RMSE can be reported directly.
  - Train, val and test come from the same mix of sites and traffic states.
  - Closes TODO #5 in [PAPER_VS_IMPLEMENTATION.md](PAPER_VS_IMPLEMENTATION.md) §10.
- **Cons:**
  - Test follows train by only ~1.5 min inside the same recording. Traffic states (e.g. one congestion wave) are correlated, so the score is optimistic.
  - The guard gaps grow from 2 to 12, losing about 1.6% of frames instead of 0.3%.
  - `get_splits` must cut per segment, using `location` + the `period` column that problem 1 keeps.
  - **Overlap leak:** the end of period k (test) shares real time with the start of period k+1 (train). At most ~10 cars per boundary
    (≤ ~30 in total) can appear in both. Accept this, or drop windows of those cars.

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

Weaker random variants:

| Variant | Verdict |
| --- | --- |
| Random split **by target vehicle** (all windows of a car in one set) | Removes the same-target overlap, but keeps the cross-vehicle leak (a test car is a neighbour with a goal label in training scenes) and the shared traffic state at the same instant. Still leaky. |
| Random **time blocks** (e.g. 1-minute chunks with 80-frame guard gaps) | Leak-free and covers all recordings, but it is option B with many more gaps (more lost frames, train and test even closer in time). |

### Comparison

| | A: global cut | B: per recording | C: random |
| --- | --- | --- | --- |
| Test sites | i-80 only | us-101 + i-80 | both |
| Test periods | i-80 p3 (+ end of p2) | all 6 | all 6 |
| Code change | none | `get_splits` per segment | small |
| Leakage | none | ≤ ~30 overlap cars | severe (near-copies, labels as inputs) |
| Score | hardest (time + site shift) | optimistic (same recording) | invalid (memorisation) |
| Comparable to paper's NGSIM number | no | closer | no (inflated) |

### Solution
**Decision:** _pending_, between A and B. C is rejected.

### Why this solution
_To be decided._

---

## Problem 6: Encoder stacking order vs Fig. 2

**Status:** Open

### Description
_Placeholder._ We run 3 spatial layers, then 3 temporal layers; Fig. 2 suggests interleaving (S, T, S, T, S, T).

### Solution
_To be decided._

### Alternatives rejected
_To be decided._

### Why this solution
_To be decided._

---

## Problem 7: Target history positions are always 0

**Status:** Open

### Description
_Placeholder._ Positions are relative to the target at each timestep, so the target row is always (0, 0) and its own path history is lost.
With problem 1b, the target's lane is also always 0.

### Solution
_To be decided._

### Alternatives rejected
_To be decided._

### Why this solution
_To be decided._

---

## Problem 8: Lateral velocity is always 0

**Status:** Open

### Description
_Placeholder._ NGSIM's `v_vel` is a scalar speed along the lane, so feature 3 (vy) is always 0 and the O-field only sees same-lane closing.

### Solution
_To be decided._

### Alternatives rejected
_To be decided._

### Why this solution
_To be decided._

---

## Problem 9: Nearest-15 fallback / neighbour count

**Status:** Open

### Description
_Placeholder._ When no vehicle passes the 0.005 risk threshold, the loader falls back to the 15 nearest vehicles, which the paper's rule would reject.
The average neighbour count should be compared with the paper's 6.

### Solution
_To be decided._

### Alternatives rejected
_To be decided._

### Why this solution
_To be decided._

---

## Problem 10: Table II risk-level breakdown missing

**Status:** Open

### Description
_Placeholder._ The paper reports RMSE for "collision in 1 s / 2 s / 3 s / 5 s" and non-collision scenes, but does not define how a scene is labelled.

### Solution
_To be decided._

### Alternatives rejected
_To be decided._

### Why this solution
_To be decided._

---

## D: How us-101 and i-80 are used together

**Status:** Open (requested for discussion after problem 10)

### Description
_Placeholder._ Pooled vs per-site training and testing; how the 6 (location, period) segments from problem 1 are split; whether results are reported per site.

### Solution
_To be decided._

### Alternatives rejected
_To be decided._

### Why this solution
_To be decided._

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
