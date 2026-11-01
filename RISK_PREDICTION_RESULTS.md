# Risk Prediction from Trajectory Predictions — Protocol and Results

Companion to [RISK_PREDICTION_METHODOLOGIES.md](RISK_PREDICTION_METHODOLOGIES.md), which
catalogues the methods. This document records what was actually implemented, the protocol it
was measured under, and the results.

Family letters (A1–A6) and axis letters (B1–B8) refer to that document.

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

*Pending the DSLab scoring run; this section will be filled from
`experiments/comparison_risk.csv` and `comparison_risk_stratified.csv`.*

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

---

## 7. Reproducing

```bash
# on the GPU box, with the trained checkpoints present
pip3 install --user pytest
python3 -m pytest tests -q                     # hard gate

python3 experiments/score_risk.py --list       # selection, with LOOSE flagged
python3 experiments/score_risk.py --top 10 --sources cv,oracle
python3 experiments/score_risk.py --top 10 --sources cv_scene,pred --sample-stride 4

python3 -c "from experiments.benchmark import compare_risk; compare_risk()"
```

`experiments/RUN_ALL.md` has the full runbook, including the warning that `--delete` must never
be added to the upload rsync — `experiments/checkpoints` is excluded from it and holds the only
copy of all 39 trained models.
