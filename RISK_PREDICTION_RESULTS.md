# Risk Prediction from Trajectory Predictions — Protocol and Results

Companion to [RISK_PREDICTION_METHODOLOGIES.md](RISK_PREDICTION_METHODOLOGIES.md), which
catalogues the methods. This document records what was actually implemented, the protocol it
was measured under, and the results.

Family letters (A1–A6) and axis letters (B1–B8) refer to that document.

**Follow-up:** [RISK_TUNING_RESULTS.md](RISK_TUNING_RESULTS.md) tunes seven of these models.
Tuning improves RMSE by 2–5 % but lowers `pred` risk ROC-AUC for all seven. It also adds
event-clustered and paired bootstrap CIs, which make the §5.3 ranking statistically
supported, and it closes caveat 13 below.

---

## 1. What was measured, and why this axis

The 39-model benchmark in [experiments/comparison_rmse.csv](experiments/comparison_rmse.csv) is
**RMSE-only**: every one of the 39 result JSONs carries `"collisions": null`, because the 42.8 h
batch ran with `--no-collisions`. No risk prediction had ever been scored in this project.

That gap matters more than it looks. The ten best models span
**2.0716 → 2.1300 m average RMSE — a 5.8 % spread**, which
[experiments/MODEL_ALTERNATIVES.md](experiments/MODEL_ALTERNATIVES.md) itself describes as within
noise for numbers copied between papers. Displacement error cannot rank these models. The
methodology doc's §B7 gives the reason: in a closed-loop study, static ADE showed *no significant
correlation* with driving performance, and the dynamics gap accounted for 77 % of the disparity.

So the question this document answers is not "which model predicts trajectories best" — that is
already a near-tie — but **"which model's predictions support a better collision-risk judgement"**.

**Short answer (§5).** When risk is computed with constant-velocity neighbours, the ten models
are indistinguishable (ROC-AUC spread ≤ 0.022). When each model also predicts the neighbours,
they separate sharply (spread up to 0.28), with an ordering unrelated to RMSE (Spearman ≈ 0):
inatran and cognitive lead and physics_informed trails on every headline measure, with
dacr_amtp third on five of six.
Model-predicted neighbours are nonetheless *worse* than constant velocity for 56 of 60
model × measure cases. No probability output is calibrated. The labels behind all of this are
~100 independent events, a third with physically overlapping vehicles (§2.1b), so the ranking is
a strong signal, not a settled result.

### The models

The ten best by `rmse_avg`, read from the committed comparison table rather than hard-coded:

| # | model | rmse_avg | 5 s | params | fidelity |
|---|---|---|---|---|---|
| 1 | `miat` | 2.0716 | 4.0551 | 194,986 | Paper description |
| 2 | `traj_mamba` | 2.0860 | 4.0936 | 121,207 | Paper description |
| 3 | `x_track` | 2.0903 | 4.0912 | 166,405 | Paper description |
| 4 | `dacr_amtp` | 2.1055 | 4.1144 | 254,150 | **LOOSE** |
| 5 | `stdan` | 2.1136 | 4.1365 | 217,483 | Paper baseline |
| 6 | `mfp` | 2.1139 | 4.1388 | 77,786 | Paper description, simplified |
| 7 | `cognitive` | 2.1139 | 4.1366 | 227,979 | Interpretation |
| 8 | `inatran` | 2.1176 | 4.1289 | 173,381 | **LOOSE** |
| 9 | `physics_informed` | 2.1240 | 4.1371 | 212,684 | Interpretation |
| 10 | `mha_lstm` | 2.1300 | 4.1597 | 146,825 | Paper description |

**LOOSE** means the notebook was built from a paper name and a one-line description;
`MODEL_ALTERNATIVES.md` states plainly that such results do not represent the published model.
They are kept in the ranking — excluding models after seeing their scores is a worse bias than
including them — and every output row carries a `loose_reimplementation` flag so
`dacr_amtp` can never be cited as the published DACR-AMTP.

---

## 2. Protocol

### 2.1 The label rule, and why it is swept rather than chosen

**NGSIM contains essentially no real collisions.** Every "collision" label here is therefore
derived by applying a distance criterion to *ground-truth future* trajectories. What is being
measured is:

> does risk computed from the **predicted** future agree with risk computed from the **actual**
> future?

That is a well-defined and useful question — it isolates predictor quality from the
surrogate-definition question — but it is **not crash prediction**, and no number in this
document should be read as one.

The threshold is a live question, not a formality: a 250 k-row debug subset produced
**6,201 pairs and zero collisions at 2.0 m**. Rather than guess, `score_split` retains each
pair's true minimum separation, so `build_report` re-derives labels at **1, 2, 3 and 5 m** and
reports both AUCs and the base rate at each, alongside the percentiles of the true separation.
`plan.md` §50 already required this threshold to be settled experimentally.

### 2.1a How a label is built

Model-free: computed from true trajectories only, so all ten models score against identical
labels.

1. **Sample** — one target vehicle, 3 s observed history (30 frames at 10 Hz) and 5 s
   (50 frames) of future; windows start every 10 frames.
2. **Split** — the test split is the last 20 % of each recording in time, with a full-window
   guard gap after the validation cut so no frame occurs in two splits.
3. **Candidates** (`cv`, `oracle`) — at the last observed frame, the 15 vehicles nearest the
   target, drawn from the global frame index rather than the model's risk-selected neighbour
   set (`loader.collision_candidates`). A candidate with no future position at all is dropped.
   The scene sources (`cv_scene`, `pred`) instead use the scene's own neighbours.
4. **Label** — over the 50 future frames, the true **centre-to-centre** distance between target
   and candidate; *d_min* is its minimum (absent frames skipped) and *t_min* when it occurs.
   **Positive iff d_min < 2.0 m.**

Centre distance, not box overlap: for two 4.5 × 1.8 m cars, 2 m apart means the boxes
overlap if one is directly behind the other, and ~0.2 m of clearance if they are side by side.

### 2.1b What the labelled set looks like

Full NGSIM test split, `cv` candidate set. Produced by a model-free profiling pass over the
same `_candidates` / `_labels` code the scoring run uses.

| | |
|---|---|
| Test samples | 121,130, from **2,162** distinct target vehicles (US-101 1,169 · I-80 993) |
| Target–neighbour pairs | 1,814,838 (~15 per sample; US-101 931,728 · I-80 883,110) |
| Positives at 1 / **2** / 3 / 5 m | 261 / **977** / 9,829 / 177,769 |
| Base rate at 1 / **2** / 3 / 5 m | 0.014 % / **0.054 %** / 0.54 % / 9.8 % |
| True minimum separation, percentiles | p0.1 2.47 m · p1 3.17 · p5 3.83 · p10 5.09 · p50 13.4 m |
| Scene-neighbour set (`cv_scene`, `pred`) | 440,276 pairs, 912 positives (0.21 %) |

**Positives are few independent events, each counted many times.** Overlapping windows
(stride 10 frames ≪ 80-frame window), and each vehicle appearing in turn as the target, count
one encounter repeatedly:

| threshold | positive pairs | **distinct vehicle-pair events** | windows per event (median / max) | top-10 events hold | events with boxes overlapping |
|---|---|---|---|---|---|
| 1 m | 261 | **13** | 14 / 86 | 236 pairs (90 %) | 13 (100 %) |
| 2 m | 977 | **101** | 9 / 90 | 277 pairs (28 %) | 36 (36 %) |
| 3 m | 9,829 | **914** | 10 / 93 | 492 pairs (5 %) | 54 (6 %) |

**Geometry of the 2 m positives** (at the moment of closest approach):

| | value |
|---|---|
| longitudinal offset, median (p10–p90) | **0.16 m** (0.02–0.85) |
| lateral offset, median (p10–p90) | **1.50 m** (0.11–1.86) |
| side-by-side (longitudinal < lateral offset) | 807 of 977 (83 %) |
| bounding boxes physically overlapping | 500 of 977 pairs (51 %), 36 of 101 events |
| already < 2 m at the last observed frame | 210 (21.5 %) |
| distance at last observed frame, median | 7.47 m (negatives: 15.9 m) |
| closest approach in 0 / ≤1 / 1–2 / 2–3 / 3–5 s | 88 / 209 / 195 / 176 / 309 |
| target speed, median | 6.1 m/s (all pairs: 6.8 m/s) |
| relative speed, median (p90) | 3.5 m/s (9.5) |
| heavy vehicle (> 8 m) involved | 0 |
| events by location | US-101 64 · I-80 37 |

Reproduce with `python3 experiments/profile_collision_labels.py`; the run behind these tables
is `experiments/runs/collision_profile.out`.

What follows from this:

- **The effective sample size is ~101, not 977.** Positives within an event are strongly
  correlated, so AUC confidence intervals are far wider than a 977-positive count suggests.
  ROC-AUC differences of order 0.01 between models should be treated as within noise. The
  scoring run keeps only aggregates, so an event-clustered bootstrap is not yet computed.
- **A large share of positives are data artefacts.** NGSIM contains no crashes, yet 36 of the
  101 events at 2 m have the two vehicle boxes overlapping. These are tracking errors, which
  are documented for NGSIM (Montanino & Punzo 2015; Coifman & Li 2017). At 1 m it is every
  event: 13 events, ~0.1–0.2 m apart at crawling speed (median 2.9 m/s), consistent with one
  physical vehicle tracked twice. **The 1 m sweep point should not be used.**
- **The 2 m label is a side-by-side label, not a rear-end label.** Positives sit almost exactly
  abreast (0.16 m longitudinal) with the lateral gap at or below one vehicle width. Rear-end
  conflicts, the scenario TTC/DRAC/PSD are built for, are nearly absent; this is one reason the
  longitudinal surrogates rank low here.
- **3 m is the cleaner, milder label**: 914 events, only 6 % with overlapping boxes, mostly
  adjacent-lane passing (median lateral 2.75 m). It is the better-supported operating point for
  ranking, though it measures close proximity rather than conflict.
- **About a fifth of 2 m positives are already within 2 m when prediction starts**, so they are
  near-trivial from the current state. The horizon-stratified view (§4) separates them out.

### 2.2 The four neighbour-future sources

"Risk from predicted trajectories" has no single meaning, and the contrast between readings is
the most informative output here.

| source | target future | neighbour future | candidate set | what it isolates |
|---|---|---|---|---|
| `cv` | model | constant velocity | 15 nearest | continuity with the original 11-measure definition |
| `cv_scene` | model | constant velocity | scene's own | matched control for `pred` |
| `pred` | model | model, as its own target | scene's own | **predictor contribution**: `pred − cv_scene` |
| `oracle` | ground truth | ground truth | 15 nearest | **ceiling**: `oracle − cv` separates a weak predictor from a weak label rule |

`cv_scene` and `pred` share a pair set, so their difference is attributable to the predictor
rather than to a change of population. `cv` and `oracle` likewise.

**Why `pred` uses the scene's own neighbours.** `loader.collision_candidates` draws the 15
nearest vehicles from the *global* frame index, so most candidates have no column on the scene's
vehicle axis and hence no `target_idx` from which to build a sample. The scene's own neighbours
do. Using the 15-nearest set instead would require rebuilding roughly 1.8 M scenes.

The origin shift between a neighbour's own prediction frame and the target's turns out to equal
the neighbour's observed relative position exactly, which makes it directly testable: a
zero-mean predictor must park every neighbour on that position, and a test asserts it.

**`oracle` is model-free**, so it is scored once and reused for every model — both as a variant
and as the regression target for the others. Its probability measures are fed
`ORACLE_SIGMA = 1e-3`, which collapses the Gaussian box integral to the exact geometric overlap
indicator (verified crisp at the box boundary) rather than skipping them or supplying a
fabricated spread that would read as a real prediction.

### 2.3 One threshold rule for all 30 measures

Each measure's decision threshold is the score at the ROC point nearest `(fpr=0, tpr=1)`,
computed identically for every measure. No measure gets a hand-tuned operating point. This is
the metric-agnostic rule from Jiao et al. (2024), and it also resolves a pre-existing
inconsistency in which scores were computed with one threshold and decided with another.

### 2.4 Frozen parameters

Every scale constant is recorded in each result JSON under `scale_constants`, so a comparison
can be reproduced and nothing can be quietly tuned on the test split:

```
A_MAX_BRAKE 8.0   A_MAX_LAT 4.0   T_SAFETY 1.5   RHO_VEHICLE 200.0
MC_SAMPLES 64     GAMMA_CI 1.0    DELTA_CI 1.0   JOKSCH_V 31.74
RSS_RHO 0.5       RSS_A_MAX 3.0   RSS_B_MIN 4.0  RSS_B_MAX 8.0   RSS_MU_LAT 0.5
```

The risk-field calibration (`GAMMA_X_POLY`, `ALPHA_X_POLY`, `GAMMA_Y`, `ALPHA_Y`, `BETA_1`,
`T_STAR`, `BETA_2`) comes from `src/config.py` unchanged, and the field thresholds use
`RISK_THRESHOLD = 0.005` — the value `build_scenes` already uses for neighbour selection, i.e. a
train-side constant, not a test-tuned one.

---

## 3. The 30 measures

The contract is `fn(target (B,T,2), neighbors (B,T,N,2), dt, threshold, **ctx) -> (B,N)`.
`threshold` always means the collision radius in metres; each measure's own scale constants live
in the module. Every measure is total on the base contract, so the existing registry-wide
contract test covers all 30.

### A1 — deterministic surrogates (20)

The original eleven, unchanged for continuity with the 39 committed JSONs:
`min_dist`, `ttc`, `pet`, `thw`, `tet`, `tit`, `drac`, `rd`, `tmd`, `s_field`, `o_field`.

> `ttc` is **not** a true TTC — it is the first-crossing time of `distance < threshold`. It keeps
> its name because 39 committed result JSONs and the `--collision_metric` default refer to it.
> `ttc_true` is the real thing.

Added:

| key | definition | aggregation | dir |
|---|---|---|---|
| `ttc_true` | smallest `τ ≥ 0` with `‖D + τV‖ = threshold`, per step | min over time | lower |
| `ttce` | time to closest encounter (constant relative velocity) | min-CPA step | lower |
| `dce` | distance of closest encounter | min-CPA step | lower |
| `psd` | distance ÷ minimum stopping distance | min over time | lower |
| `btn` | required deceleration ÷ available; ≥1 means braking cannot avoid | max over time | higher |
| `stn` | required lateral acceleration ÷ available | max over time | higher |
| `a_lat_req` | lateral acceleration needed to clear the remaining width | max over time | higher |
| `dst` | deceleration to hold `T_SAFETY` seconds | max over time | higher |
| `pret` | predictive encroachment time at the predicted path crossing | at crossing | lower |

`ttce` and `dce` reuse `src/risk.py::closest_approach`, which existed but was unused outside
tests. `dst` reduces exactly to `a_long,req` at `T_SAFETY = 0`, which is asserted.

### A2 — collision probability (2)

| key | definition | dir |
|---|---|---|
| `p_coll_gauss` | collision **state** probability: analytic box integral of the predicted bivariate Gaussian, max over the horizon | higher |
| `p_coll_mc` | collision **event** probability: Monte Carlo over the Cholesky factor, any step | higher |

Naming one of each is deliberate — the methodology doc flags that state and event probability
are routinely conflated. `p_coll_gauss` factorises the rectangle integral across axes and so
ignores `ρ`; `p_coll_mc` is ρ-exact, and the two are asserted to agree to 0.02 at `ρ = 0`, which
is the test for that approximation.

`p_coll_mc` uses **common random numbers**: noise is drawn once per horizon and broadcast across
the batch, so a pair's score does not depend on which batch it landed in, and comparisons
between measures share a perturbation set.

### A3 — calibrated risk fields (3)

`s_field_cal`, `o_field_cal`, `so_field_cal` — the `[22]` calibration via `pair_risk`
(bumper-to-bumper gaps, speed-dependent `γx/αx`, `d* = ½(w_i + w_j)`, closing-only). Distinct
from the existing `s_field`/`o_field`, which use the library defaults and *not* the `config.py`
calibration. `so_field_cal` is the quantity the risk-scaled loss and the loader's neighbour
selection actually use.

### A4 — severity-weighted (3)

| key | definition | dir |
|---|---|---|
| `delta_v` | `m_j/(m_i+m_j)·‖v_i − v_j‖` at the step of minimum predicted distance | higher |
| `conflict_index` | `δ·ΔKE·exp(−γ·PET)` — severity × a PET-based probability proxy | higher |
| `p_pdrf` | `max_t [ P_t · (Δv_t/31.74)⁴ ]` — the only true `probability × severity` measure | higher |

Two deliberate deviations, both documented in code:

- **Mass is a proxy**: `RHO_VEHICLE × length × width` (≈1620 kg for a 4.5 × 1.8 m car). NGSIM
  carries no masses.
- **`conflict_index` uses a product, not a quotient.** The review renders CI as a division by
  `exp(−γ·PET)`, which makes risk *grow* with post-encroachment time. The product matches the
  source's own description ("PET enhanced with a collision probability estimation as well as a
  severity factor") and is what makes it a probability × severity form. `PET = ∞` then gives
  zero risk, as it should.

### A6 — formal (2)

`rss_ds` (binary: both safe distances violated at some step) and `rss_margin` (the worst
longitudinal intrusion in metres).

> `rss_ds` is **nominal-scale**. Its ROC has two points, so its AUC collapses to
> `(1 + TPR − FPR)/2` and is **not comparable with the continuous rows** — exactly Westhofen's
> "output scale" suitability property. `rss_margin` is the gradable version of the same
> quantity, which is why both are reported.

### Deliberately not implemented

| dropped | why it cannot come from NGSIM |
|---|---|
| `P_SMH` (per-mode probability) | 9 of the 10 models are unimodal, so it could never appear in a cross-model table; STRAP's per-mode parameters are also local variables in its decoder |
| WTTC, TTM/TTB/TTS/TTK, TTR, TCI | need a maneuver set, control bounds and an optimisation per pair |
| TTZ/TTO, ET, SOI, AGS | need conflict-area, intersection or crosswalk geometry |
| PTTC | car-following only; the candidate set is not restricted to leaders |
| CPI, ACI | need a distribution over `a_long,min` and a calibrated causal collision tree |
| PRI | pedestrian-specific |
| `P_SRS`, conformal prediction | need reachability tooling and a calibration split whose exchangeability assumption fails under the shift being studied |
| any oriented-bounding-box measure | NGSIM has no heading or yaw; all boxes are axis-aligned |

### Known degeneracies, stated rather than hidden

- **`pret` is sparse by construction.** `collision_candidates` reports neighbour velocity as
  `[speed, 0]` — NGSIM has no lateral speed — so in the `cv` variant the two rays are
  near-parallel and PrET is `∞` for most pairs. The `non_finite_fraction` column carries this.
- The same zero lateral velocity caps what `stn` and `a_lat_req` can mean in the `cv` variant,
  which is part of why `cv_scene` and `pred` are reported beside it.
- `ttc` and `ttc_true` are mostly `∞` on freeway traffic, which is correct: most pairs never
  approach contact. `_prepare_auc_inputs` ranks `∞` as the safest finite value, so the AUCs
  remain well-defined.

---

## 4. Evaluation axes

| axis | what is reported |
|---|---|
| B2 ranking | PR-AUC and ROC-AUC per measure. Both, because they can disagree about the same model, and PR-AUC is base-rate dependent so it is not comparable across label thresholds |
| B2 operating point | precision / recall / F1 / FPR at the shared ROC threshold, plus recall at precision ≥ 0.5 |
| B3 calibration | Brier with the Murphy split, log loss and reliability bins, for the probability-valued measures only |
| B4 earliness | lead time from horizon prefixes (1–5 s), the detected fraction, and recall per prefix |
| B5 regression | each measure on predicted futures against the same measure on true futures |
| B6 stratification | every measure re-scored per `collision in 1s/2s/3s/5s` bucket against the full non-collision background |
| C4 sanity | mean score on collision vs non-collision pairs — the first thing to read |

**On earliness, an honest scoping note.** The accident-anticipation literature's TTA scans a
score series over *observation* time. The metric contract here yields one scalar per pair, so
no such series exists. What is reported instead is anticipation over the **prediction horizon**:
the earliest horizon prefix at which a measure already fires. It is named `lead_time`, not TTA,
and is not comparable with dashcam TTA figures.

**Why stratification matters most.** STRAP's own Table II shows STRAP-B and STRAP-R with an
*identical* 1.78 m average RMSE, separating only once stratified by how imminent the collision
is. With a 5.8 % spread across these ten models, the stratified view is the one with a real
chance of ranking them.

---

## 5. Results

Full NGSIM test split, all 10 models, all four neighbour sources, all 30 measures. Every table
below is printed by `python experiments/summarise_risk.py` from the committed
`experiments/results/risk/*.json`. Flat and stratified long tables are in
`experiments/comparison_risk.csv` and `experiments/comparison_risk_stratified.csv`.
The scoring run took 64.5 min (`cv` + `oracle`) plus 97.4 min (`cv_scene` + `pred`) on DSLab.

Two pair sets, as §2.2 explains. `cv`/`oracle` use the global 15 nearest: 1,814,838 pairs,
977 positives at 2 m, base rate 0.054 %. `cv_scene`/`pred` use the scene's own risk-selected
neighbours: 440,276 pairs, 912 positives, base rate 0.21 %. **Compare models within a
variant, never ROC/PR values across the two pair sets.**

### 5.1 Headline findings

1. **Model-predicted neighbour futures are worse for risk than constant velocity.** On the
   matched pair set, `pred` beats `cv_scene` in **4 of 60** (model × headline measure) cases,
   and those 4 are within noise (+0.008 to +0.015, inatran and cognitive on the probability
   measures). Averaged over models, swapping CV neighbours for predicted ones costs up to
   0.18 ROC-AUC, on every measure except `dce` and `pret` (table 5.2, last column).
2. **The target's predicted trajectory does not separate the models.** With CV neighbours
   (`cv`, `cv_scene`), the ROC-AUC spread across all 10 models is **0.003–0.022** on every
   headline measure. All of the between-model separation appears only once the model also
   predicts the neighbours (`pred`: spread 0.11–0.28).
3. **Under `pred` the ranking is consistent, and it is not the RMSE ranking.** All six
   headline measures put inatran 1st, cognitive 2nd and physics_informed 10th; dacr_amtp is
   3rd on five of the six (inatran and dacr_amtp are **LOOSE** reimplementations). Spearman correlation between risk rank and RMSE rank is
   **−0.25 to +0.07**, i.e. none. The RMSE leader (miat) is 4th–6th.
4. **Which measure wins depends on the label.** At 2 m, probability measures lead under
   `pred` (`p_coll_gauss` 0.88). At 3 m, the time-scale measures (`ttc`, `tet`, `tit`, `pet`,
   `ttc_true`) **collapse to chance** (0.41–0.55 on the model variants; even the oracle reaches
   only 0.550 for `ttc`/`tet`/`tit` and 0.639 for `ttc_true`). 3 m positives are mostly
   adjacent-lane passing with no closing motion (§2.1b), so there is nothing for a
   time-to-collision measure to detect. Distance/field measures (`s_field_cal`, `min_dist`)
   are the only ones near the top at both thresholds.
5. **No probability measure is calibrated.** Every Brier skill score is negative: the
   forecasts are worse than always predicting the base rate. `p_coll_gauss` predictions of
   0.1–0.9 all correspond to 1–2 % observed collisions (§5.5).

### 5.2 Measures — mean ROC-AUC over the 10 models, label < 2 m

Sorted by `pred`. `oracle` is model-free (one number). `spread` is max − min over models under
`pred`. The last column is the predictor's contribution over CV extrapolation on the matched
pair set.

| measure | oracle | cv | cv_scene | **pred** | spread | pred − cv_scene |
|---|---|---|---|---|---|---|
| `p_coll_gauss` | 0.771* | 0.962 | 0.925 | **0.882** | 0.111 | −0.043 |
| `pet` | 0.997 | 0.925 | 0.924 | **0.878** | 0.061 | −0.046 |
| `p_coll_mc` | 0.771* | 0.938 | 0.912 | **0.860** | 0.124 | −0.052 |
| `p_pdrf` | 0.770* | 0.960 | 0.920 | **0.857** | 0.157 | −0.063 |
| `so_field_cal` | 0.988 | 0.977 | 0.939 | **0.851** | 0.184 | −0.089 |
| `ttc_true` | 0.998 | 0.946 | 0.947 | **0.841** | 0.281 | −0.107 |
| `s_field_cal` | 0.998 | 0.981 | 0.952 | **0.839** | 0.177 | −0.113 |
| `rss_ds` | 0.924 | 0.835 | 0.821 | **0.781** | 0.062 | −0.040 |
| `min_dist` | 1.000 | 0.969 | 0.946 | **0.779** | 0.227 | −0.167 |
| `s_field` | 1.000 | 0.968 | 0.945 | **0.770** | 0.230 | −0.175 |
| `conflict_index` | 0.944 | 0.855 | 0.834 | **0.765** | 0.200 | −0.070 |
| `thw` | 0.933 | 0.886 | 0.866 | **0.747** | 0.144 | −0.119 |
| `o_field_cal` | 0.887 | 0.860 | 0.761 | **0.732** | 0.267 | −0.029 |
| `ttc` / `tet` / `tit` | 1.000 | 0.869 | 0.882 | **0.719** | 0.206 | −0.163 |
| `rss_margin` | 0.825 | 0.789 | 0.793 | **0.702** | 0.136 | −0.091 |
| `psd` / `rd` | 0.82 | 0.773 | 0.765 | **0.690** | 0.098 | −0.075 |
| `o_field` | 0.918 | 0.882 | 0.808 | **0.685** | 0.148 | −0.123 |
| `dce` | 0.805 | 0.874 | 0.647 | **0.671** | 0.115 | +0.024 |
| `drac` | 0.885 | 0.832 | 0.793 | **0.666** | 0.256 | −0.127 |
| `stn` / `a_lat_req` | 0.790 | 0.744 | 0.676 | **0.663** | 0.060 | −0.013 |
| `btn` | 0.836 | 0.784 | 0.794 | **0.643** | 0.246 | −0.150 |
| `dst` | 0.799 | 0.764 | 0.773 | **0.641** | 0.246 | −0.131 |
| `delta_v` | 0.750 | 0.704 | 0.679 | **0.580** | 0.186 | −0.099 |
| `pret` | 0.545 | 0.509 | 0.507 | **0.515** | 0.038 | +0.008 |
| `ttce` | 0.627 | 0.480 | 0.598 | **0.501** | 0.161 | −0.097 |
| `tmd` | 0.432 | 0.424 | 0.529 | **0.466** | 0.161 | −0.063 |

\* The oracle's probability measures use `ORACLE_SIGMA`, which collapses them to a binary
box-overlap indicator (§2.2). A binary score has a degenerate ROC curve, so these oracle entries
are not a ceiling. Elsewhere `oracle` ≈ 1.0 is partly circular: the label is a threshold on the
same true distance `min_dist` reports, so `min_dist`/`s_field` = 1.000 is a pipeline check, not
a finding.

`pret` is at chance because NGSIM lateral velocity is identically zero (§3). `tmd` and `ttce` are
at or below chance on every variant, so neither is usable as a ranking score here.

**At 3 m** (base rate 0.54 % on the global set, 1.96 % on the scene set), `pred` ROC-AUC leaders are `s_field_cal`
0.861, `so_field_cal` 0.848, `min_dist` 0.843, `s_field` 0.834, `thw` 0.774. Every time-scale
measure falls to 0.41–0.55 on the model variants (oracle `ttc` 0.550, `ttc_true` 0.639).

### 5.3 Models — `pred`, label < 2 m, ROC-AUC (PR-AUC in brackets)

Ordered by RMSE. Base rate 0.21 %, so a PR-AUC of 0.2 is ~100× chance.

| model | RMSE avg | `p_coll_gauss` | `p_coll_mc` | `s_field_cal` | `min_dist` | `ttc_true` | `tet` | rank |
|---|---|---|---|---|---|---|---|---|
| miat | 2.072 | 0.903 (0.20) | 0.880 (0.24) | 0.852 (0.22) | 0.774 (0.12) | 0.889 (0.36) | 0.694 (0.30) | 4–6 |
| traj_mamba | 2.086 | 0.853 (0.18) | 0.829 (0.22) | 0.809 (0.20) | 0.734 (0.04) | 0.797 (0.29) | 0.683 (0.13) | 7–8 |
| x_track | 2.090 | 0.848 (0.19) | 0.824 (0.24) | 0.824 (0.23) | 0.756 (0.04) | 0.730 (0.32) | 0.699 (0.20) | 5–9 |
| dacr_amtp **LOOSE** | 2.105 | 0.924 (0.22) | 0.903 (0.27) | 0.883 (0.25) | 0.835 (0.39) | 0.901 (0.41) | 0.768 (0.49) | **3–4** |
| stdan | 2.114 | 0.869 (0.19) | 0.843 (0.23) | 0.812 (0.21) | 0.736 (0.12) | 0.888 (0.34) | 0.694 (0.26) | 6–7 |
| mfp | 2.114 | 0.850 (0.18) | 0.824 (0.22) | 0.791 (0.20) | 0.723 (0.04) | 0.747 (0.29) | 0.675 (0.10) | 8–9 |
| cognitive | 2.114 | 0.934 (0.22) | 0.919 (0.28) | 0.911 (0.28) | 0.890 (0.22) | 0.930 (0.58) | 0.806 (0.37) | **2** |
| inatran **LOOSE** | 2.118 | 0.938 (0.23) | 0.921 (0.28) | 0.933 (0.30) | 0.910 (0.47) | 0.943 (0.59) | 0.836 (0.50) | **1** |
| physics_informed | 2.124 | 0.827 (0.19) | 0.797 (0.21) | 0.757 (0.20) | 0.683 (0.19) | 0.661 (0.26) | 0.630 (0.39) | **10** |
| mha_lstm | 2.130 | 0.875 (0.21) | 0.858 (0.26) | 0.819 (0.24) | 0.749 (0.34) | 0.920 (0.41) | 0.706 (0.52) | 3–6 |

Spearman(risk rank, RMSE rank) per measure: `p_coll_gauss` −0.07, `p_coll_mc` −0.07,
`s_field_cal` +0.07, `min_dist` +0.07, `ttc_true` −0.25, `tet` −0.15.

At 3 m the top three are unchanged on `s_field_cal`/`min_dist`: inatran 0.903/0.933,
cognitive 0.898/0.909, dacr_amtp 0.886/0.860. mha_lstm leads on the probability measures
(0.789/0.784).

Under `cv`, the same table is flat: `p_coll_gauss` 0.960–0.965, `s_field_cal` 0.980–0.982,
`min_dist` 0.967–0.971 across all 10 models.

**How much weight the ranking bears.** The gaps are large next to the CV-variant spread, and
the ordering repeats across six measures. But those measures are computed from the same
predictions, so their agreement is not independent evidence. The positives are ~100 events,
not 912 (§2.1b), and no clustered confidence interval has been computed (§6, items 10 and 13). Treat it
as **a strong signal that neighbour-prediction quality, not target RMSE, decides risk
accuracy**, not as a final ranking. Two of the top three are LOOSE, so the result speaks to
those implementations, not the published models.

### 5.4 Horizon-stratified view — `pred`, mean ROC-AUC (spread over models)

Each bucket's positives against all negatives, bucketed by when the true closest approach
occurs. Positives per bucket: 292 / 176 / 163 / 281.

| measure | collision ≤ 1 s | 1–2 s | 2–3 s | 3–5 s |
|---|---|---|---|---|
| `p_coll_gauss` | 0.943 (0.07) | 0.918 (0.11) | 0.873 (0.18) | **0.802** (0.24) |
| `p_coll_mc` | 0.929 (0.10) | 0.900 (0.15) | 0.847 (0.23) | 0.771 (0.27) |
| `ttc_true` | 0.895 (0.19) | 0.838 (0.33) | 0.822 (0.34) | 0.796 (0.33) |
| `s_field_cal` | **0.973** (0.04) | 0.906 (0.15) | 0.807 (0.28) | 0.676 (0.30) |
| `min_dist` | 0.956 (0.06) | 0.821 (0.27) | 0.717 (0.35) | 0.604 (0.30) |
| `tet` | 0.857 (0.19) | 0.719 (0.29) | 0.661 (0.26) | 0.610 (0.14) |

- **Every measure is near-trivial for imminent collisions.** On the global set, about a fifth
  of positives are already within 2 m at the last observed frame (§2.1b).
- **The probability measures degrade least with horizon** (`p_coll_gauss` 0.94 → 0.80). The
  deterministic distance/field measures fall to 0.60–0.68 for collisions 3–5 s out, which is
  where predictive spread matters.
- **The models separate as the horizon grows**: spread 0.04–0.07 within 1 s, 0.24–0.30 at
  3–5 s. As STRAP's own Table II showed for STRAP-B/R, the stratified view is where models
  that look alike on average come apart.

### 5.5 Calibration — probability measures

Brier skill score = 1 − Brier / uncertainty; 0 means no better than always forecasting the
base rate. Mean over models (range):

| measure | BSS, `cv` | BSS, `pred` | reliability, `pred` | resolution, `pred` |
|---|---|---|---|---|
| `p_coll_gauss` | −9.8 | −3.7 (−8.2 … −1.5) | 7.7e-3 | 1.3e-4 |
| `p_coll_mc` | −45.5 | −25.4 | 5.2e-2 | 3.6e-5 |
| `p_pdrf` | −2.4e-4 | −1.9e-3 | 4.3e-6 | 0 |
| `rss_ds` | −62.4 | −44.4 | 9.2e-2 | 1.7e-5 |

Reliability bins, `p_coll_gauss`, miat, `pred`:

| predicted | 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| pairs | 420,685 | 8,913 | 3,963 | 1,915 | 1,212 | 871 | 632 | 486 | 362 | 1,237 |
| observed | 0.10 % | 1.2 % | 1.4 % | 1.6 % | 1.4 % | 1.3 % | 2.1 % | 1.2 % | 1.7 % | **21.7 %** |

The forecasts are informative only as a ranking. Between 0.1 and 0.9 the observed rate is flat
at 1–2 %, and only the top bin carries signal. `p_pdrf`'s near-zero BSS is not skill: it
forecasts ≈ 0 everywhere. Recalibration (isotonic or Platt, fitted on validation) is needed
before any of these can be read as a probability. It is not attempted here.

### 5.6 Earliness — `cv`, mean over models

Recall at the alarm threshold when only the first *k* seconds of the predicted horizon are
scored.

| measure | ≤ 1 s | ≤ 2 s | ≤ 3 s | ≤ 5 s | mean lead (s) |
|---|---|---|---|---|---|
| `so_field_cal` | 0.784 | 0.874 | 0.925 | **0.954** | 0.98 |
| `ttc_true` | **0.876** | 0.885 | 0.892 | 0.897 | **1.15** |
| `p_coll_gauss` | 0.497 | 0.704 | 0.854 | 0.909 | 0.68 |
| `min_dist` | 0.496 | 0.651 | 0.768 | 0.882 | 0.41 |
| `btn` | 0.671 | 0.704 | 0.735 | 0.780 | 0.90 |
| `drac` | 0.690 | 0.703 | 0.715 | 0.741 | 0.96 |

`ttc_true` fires earliest: recall is already 0.876 at 1 s, of a final 0.897.
`so_field_cal` reaches the highest final recall. `p_coll_gauss` and `min_dist` need most of the
horizon.

### 5.7 What to take from this

- **For ranking trajectory predictors by safety:** use the `pred` variant and the stratified
  view. RMSE, and risk computed with CV neighbours, are both blind to the differences that
  matter.
- **For a deployed risk score:** model-predicted neighbours currently *hurt*. CV neighbours
  plus the model's target prediction, scored with `s_field_cal` or `p_coll_gauss`, is the
  strongest configuration measured (ROC-AUC 0.95/0.93 on the matched set). A probability
  output needs recalibration first.
- **For the label:** 2 m is dominated by side-by-side encounters, a third of which are
  tracking artefacts. 3 m is cleaner but suppresses time-scale measures entirely. Any
  conclusion should be checked at both.

---

## 6. Caveats

1. **These are surrogate-label agreements, not crash predictions.** NGSIM has no real
   collisions; the positives are whatever the distance threshold declares. The `oracle` column
   is the ceiling this protocol can reach, and the gap between `oracle` and the real crash
   question is not measured here at all.
2. **Validity is largely unevidenced in the literature.** Westhofen et al. note that for most
   criticality metrics no sources provide evidence for their properties. A claim that measure X
   ranks models better is a claim about rank correlation with these surrogate labels.
3. **PR-AUC is base-rate dependent.** Do not compare it across label thresholds, or against
   figures from another dataset, as though the numbers were commensurable.
4. **Two of the ten models are LOOSE reimplementations.** Flagged in every row.
5. **`rss_ds`'s AUC is not comparable** with the continuous measures (nominal scale).
6. **Mass, and therefore every severity-weighted measure, rests on a footprint proxy.**
7. **`conflict_index` deviates from the reviewed formula** as described in §3.
8. **No measure here is calibrated against observed crashes.** Extreme-value modelling, the
   established route to that, is not attempted.
9. **The comparison literature's own numbers are not commensurable with these.** Every figure
   cited in the methodology doc was measured under a different protocol.
10. **~101 independent events, not 977 positives** (§2.1b). Overlapping windows repeat each
    encounter ~9 times, so per-pair AUCs overstate their own precision. Small cross-model AUC
    gaps are not evidence of a ranking.
11. **Some positives are tracking artefacts, not near-misses.** 36 of 101 events at 2 m, and
    all 13 at 1 m, have physically overlapping vehicle boxes. NGSIM's trajectory errors are
    documented in M. Montanino and V. Punzo, "Trajectory data reconstruction and
    simulation-based validation against macroscopic traffic patterns", *Transportation
    Research Part B* 80 (2015), and B. Coifman and L. Li, "A critical evaluation of the Next
    Generation Simulation (NGSIM) vehicle trajectory dataset", *Transportation Research Part B*
    105 (2017). A measure can score well partly by detecting the artefact.
12. **`pred` predicts each neighbour inside the original target's scene.** The neighbour
    becomes the target of `create_sample(scene, target_idx=j)`, but its context is the vehicle
    set selected around the *original* target, not its own risk-selected neighbourhood. That is
    a harder, off-distribution input than the models were trained on. Part of the "predicted
    neighbours are worse than CV" finding (§5.1) may reflect this context mismatch rather than
    the predictors themselves. Models that are robust to it rank higher, which is itself a
    property worth having.
13. **Per-pair scores are not persisted**, only aggregates. An event-clustered bootstrap of
    the §5.3 ranking needs a rescoring run that saves them; it is the next step to make the
    ranking claim defensible.

---

## 7. Reproducing

```bash
# on the GPU box, with the trained checkpoints present
pip3 install --user pytest
python3 -m pytest tests -q                     # hard gate

python3 experiments/score_risk.py --list       # selection, with LOOSE flagged
python3 experiments/score_risk.py --top 10 --sources cv,oracle
python3 experiments/score_risk.py --top 10 --sources cv_scene,pred
```

The committed `cv_scene`/`pred` results cover the **full** test split (`n_samples` 118,009 of
121,130; a stride of 4 would allow at most 30,283), and the stride-1 rescoring in
[RISK_TUNING_RESULTS.md](RISK_TUNING_RESULTS.md) reproduces every one of their AUCs exactly. An
earlier version of this command carried `--sample-stride 4`, which is not what produced them.

```bash

python3 -c "from experiments.benchmark import compare_risk; compare_risk()"
```

`experiments/RUN_ALL.md` has the full runbook, including the warning that `--delete` must never
be added to the upload rsync — `experiments/checkpoints` is excluded from it and holds the only
copy of all 39 trained models.
