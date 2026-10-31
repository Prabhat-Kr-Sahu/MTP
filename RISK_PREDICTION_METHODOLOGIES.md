# Risk Prediction from Trajectory Predictions — Methodologies and Evaluation

**Scope.** How to turn a trajectory predictor's output into a collision/risk prediction, and
how to measure whether that risk prediction is any good so that methods can be compared.
Written as a reference for the MTP project (STRAP reproduction + pairwise collision-risk
extension, `plan.md` §48–56).

**Source confidence.** Sections marked **[read]** are drawn from the full text of the paper
(downloaded and extracted). Sections marked **[snippet]** rest on search results / abstracts
only and should be verified before you cite a number from them.

---

## 0. The core framing

Nearly every method in the literature decomposes into the same three stages:

```
history ──► (1) PREDICT future motion ──► (2) SCORE pairwise interaction ──► (3) AGGREGATE ──► risk
            point / Gaussian / multimodal   surrogate measure, collision      over time,
            / sampled set / reachable set   probability, field potential      neighbours, modes
```

Two framings of "risk" compete, and they are not interchangeable:

| Framing | Definition | Implication |
|---|---|---|
| **Criticality / conflict** | How close this situation is to a collision | Surrogate measures (TTC, DRAC, PET…). Validity is assumed, not measured. |
| **Risk proper** | `Risk = P(collision) × Severity` | Needs a probability model *and* a consequence model. Follows ISO 26262 / SOTIF framing. |

The second is what `Risk = probability × cost` means in the risk-field literature. Westhofen
et al. make the sharper point: criticality is *"the combined risk of the involved actors when
the traffic situation is continued"* — i.e. a metric measures criticality of a **continuation**,
so the metric is only as good as its prediction model. **[read]**

A third framing, increasingly common, treats a conflict as an **extreme event of normal
interaction** and asks for `P(proximity ≤ 0 | context)` under a learned proximity distribution
(Jiao et al. 2024). **[read]**

---

## PART A — Methodologies for deriving risk from predicted trajectories

### A1. Deterministic surrogate measures on the predicted trajectory

Compute a classical surrogate safety measure (SSM) / criticality metric, but evaluate it on the
*predicted* future rather than on a constant-velocity extrapolation. This is the cheapest and
most common approach, and it is what `src/evaluation/risk_metrics.py` already does.

Westhofen et al. (2022) is the authoritative catalogue — ~37 metrics with unified notation.
Grouped: **[read]**

**Time-scale (temporal proximity)**

| Metric | Definition |
|---|---|
| **TTC** | Time until predicted collision under the dynamic motion model (DMM) |
| **PTTC** | TTC specialised to car-following with constant velocity of follower, constant deceleration of leader. Cheaper, much less valid. |
| **TTCE / DCE** | Time to Closest Encounter and the Distance of Closest Encounter. Generalises TTC to the *non-collision* case — critical, because TTC is `∞` in most non-following scenes. `DCE→0 ⟺ TTCE→TTC`. |
| **WTTC** | Worst TTC: min over *all* trajectories available to both actors under an over-approximating DMM. Over-approximates ⇒ sensitivity 1. Good for data filtering. |
| **TTZ / TTO** | Time to Zebra / Time to Object — TTC against a static conflict area |
| **TET** | Time Exposed TTC: `∫ 𝟙[TTC < τ] dt`. Generalises to *any* metric + threshold. |
| **TIT** | Time Integrated TTC: `∫ (τ − TTC)⁺ dt`. Reflects criticality more finely than TET. |
| **TTM / TTB / TTS / TTK** | Time to Maneuver — latest time within `[0, TTC]` at which maneuver `m` still avoids collision. Specialisations: Brake, Steer, Kickdown. Returns `−∞` if unavoidable. |
| **TTR** | Time to React = `max over maneuver set M of TTM` |
| **THW / HW** | Time Headway / Headway (distance). Used in regulation. |
| **ET** | Encroachment Time — time an actor occupies a conflict area. Ignores other actors entirely. |
| **PET** | Post-Encroachment Time. **Retrospective** — cannot anticipate. |
| **PrET / TA / SPrET** | *Predictive* Encroachment Time: anticipated PET w.r.t. a predicted intersection point. `TA` is PrET under constant velocity. `SPrET` multiplies by `(t₁+t₂)` to down-weight far-future intersections — a cheap way to encode prediction uncertainty. |

**Deceleration / acceleration-scale (evasive effort)**

| Metric | Definition |
|---|---|
| **a_long,req** | Max longitudinal deceleration needed to avoid collision. Under constant acceleration in car-following this **is DRAC**. |
| **a_lat,req** | Min absolute lateral acceleration in either direction that evades |
| **a_req** | Norm of the two: `√(a²_long,req + a²_lat,req)` |
| **a_req,cond** | `a_req` if `SPrET < 3 s`, else 0. A worked example of **combining metrics to raise validity** — dynamical criticality only matters once temporal criticality is present. |
| **DRAC** | Deceleration Rate to Avoid Crash (closing speed² / 2·distance) |
| **DST** | Deceleration to Safety Time: deceleration needed to keep a time gap `t_s`. `t_s = 0` ⇒ DRAC. |
| **BTN / STN** | Brake / Steer Threat Number = `a_req` ÷ *available* acceleration. **Normalised to [0,∞), where ≥1 means unavoidable by that maneuver.** Far more interpretable across vehicles than raw DRAC. |
| **DSSM** | BTN variant adding worst-case lead braking + follower reaction time |
| **PSD** | Proportion of Stopping Distance = distance to conflict area ÷ minimum stopping distance |
| **LatJ / LongJ** | Jerk. Measures maneuver abruptness; used for driving-style classification. |

**Distance / occupancy-scale**

- **SOI** — Space Occupancy Index: counts violations of a per-actor "personal space" over time.
- **AGS** — Accepted Gap Size: the gap at which an actor is predicted to act. A complexity measure.
- **MDM** — Minimal Distance Margin (longitudinal + lateral decomposition), used in Ref. [2304.05610]. **[read]**

**Composite / probability-weighted**

- **CPI** — Crash Potential Index: `(1/Δt)∫ P(a_long,req > a_long,min) dt`, with `a_long,min`
  treated as a random variable (road surface, brakes). *The aggregation-of-exceedance-probability
  trick generalises to any metric with a distributed target value.* **[read]**
- **ACI** — Aggregated Crash Index: extends CPI to a probabilistic causal **collision tree**;
  `ACI = Σ_leaves P(leaf) · 𝟙[collision at leaf]`.
- **CI** — Conflict Index: `CI = (δ·ΔKE) / exp(−γ·PET)`. Denominator = collision probability
  proxy, numerator = severity from predicted kinetic-energy change. Calibration factor `γ` is
  country/geometry/visibility dependent.
- **TCI** — Trajectory Criticality Index: solves a **minimisation** over `(a_long, a_lat)` for the
  least-difficult evasive option under physical (Kamm's circle) + regulatory constraints. Risk =
  difficulty of the easiest escape.
- **PRI** — Pedestrian Risk Index: `∫ v²_impact · (t_stop − TTZ) dt` over the conflict period.
  Combines probability proxy with impact-speed severity.

**Practical caveats from the comparison literature** **[snippet]**

- DRAC repeatedly comes out as the strongest single crash-risk predictor; TTC is supplementary;
  PET is weakest, and negligible at 1 Hz sampling.
- TTC with a constant-velocity DMM, PET, and DRAC are all **insufficient when successive
  vehicles run at similar speeds** — exactly the NGSIM car-following regime.
- TTC and PET can move in *opposite directions* as CAV penetration rises.
- A single-point-kinematics TTC evaluates to `∞` in most non-following scenes. Mitigations: use
  point *sets* in the DMM, discretise space, or switch to TTCE/DCE.

### A2. Probabilistic collision probability

Treat the prediction as a distribution and integrate the collision event.

| Method | How |
|---|---|
| **P_MC** — Monte Carlo | Sample control inputs `u_i(t)`, roll out ODEs `ṡ_i = f(s_i, u_i)`, detect bounding-box intersection, weight by goal-function priors: `P(C) = ∫ P(C\|U) P(U) dU`. **This is the reference standard** — exact collision probability has no closed form even for circles with Gaussian uncertainty. Too slow for real-time; motivates everything below. **[read + snippet]** |
| **Analytic Gaussian overlap** | Closed-form or bounded integral of the product of two Gaussians over the collision set. Distinguish **collision *state* probability** (overlap at one instant) from **collision *event* probability** (integrated over the horizon) — they are different quantities and are frequently conflated. **[snippet]** |
| **Rate-based** | Derive a collision-probability *rate* from 2D overlap and integrate; reproduces MC results and yields an upper bound valid for arbitrary prediction models incl. process noise. **[snippet]** |
| **P_SRS** — Stochastic reachable sets | Over-approximate each actor's reachable set, discretise state/input space, model behaviour as Markov chains, get transition probabilities `Φ_ji(T) = V(R_i(T) ∩ X_j)/V(R_i(T))`, then `P = Σ p_pos · p_pos` over colliding path pairs. **[read]** |
| **P_SMH** — Scoring multiple hypotheses | `P = Σ_i Σ_j δ_ij · p_{A,i} · p_{¬A,j}` over `N` ego and `M` other-actor candidate trajectories, `δ_ij = 1` iff that pair collides. **The natural fit for a multimodal predictor with mode probabilities.** **[read]** |
| **Bayesian extreme-event** | Learn a *proximity distribution* conditioned on interaction context, then `P(conflict) = P(proximity ≤ 0 \| context)`; conflict **intensity** is a separate output. Unifies PSD/DRAC/TTC as special cases. **[read]** |

**Important for STRAP specifically.** STRAP's decoder emits a bivariate Gaussian
`(μx, μy, σx, σy, ρ)` per step per mode. That is *exactly* the input to A2 — you can compute a
genuine collision probability rather than thresholding a point-estimate distance. The
`P-PDRF` method (Wang, Alonso-Mora & Wang, T-ITS 2022) does precisely this with an LSTM
multimodal predictor and is the closest published analogue to what the MTP extension should do. **[read]**

### A3. Risk / potential field methods

Quantify risk as a continuous scalar field, so there is no `∞` and no threshold cliff.

| Model | Formulation |
|---|---|
| **DRF** (Kolekar et al. 2020) | Driver's *subjective* probability of occupying a position × potential consequence |
| **Driving Safety Field** (Wang et al.) | Unified field over human + vehicle + road factors |
| **PDRF** (Mullakkal-Babu et al.) | Probabilistic driving risk field: risk = consequence of ego maintaining its plan despite unknown surrounding motion; accelerations assumed Gaussian. **Shortcomings (per Wang et al.): severity estimated from *current* velocity because future velocity isn't predicted; Gaussian acceleration params must be hand-set; no ego↔surrounding interaction.** **[read]** |
| **P-PDRF** (Wang, Alonso-Mora & Wang 2022) | Fixes all three. `Risk = max over t in horizon of Σ_modes [ P(mode i) × P(collision \| i, t) × E[severity \| i, t] ]`. Collision probability from the predicted bivariate normal of position per mode. Optionally adds a second bivariate normal as a *subjective* risk factor when modelling human drivers. **[read]** |
| **Composite SPF** (Zuo et al. 2025) | Subjective field (human-like proximity perception) + objective field (imminent collision) — **this is STRAP's S-field / O-field** |
| **STRAP S-field** | `r^s_ij = exp(−\|Δx/γx\|^αx − \|Δy/γy\|^αy)` |
| **STRAP O-field** | `r^o_ij = exp[−(d̂_m/d*)^β1] · exp[−(t̂_m/t*)^β2]` from predicted closest-approach distance and its time |
| **Driver perception-response field** (2025) | Adds risk gain and attenuation through the perception-response process |
| **PF / SP / SFF** | Generic superposition of per-object potentials; Safety Force Field's Safety Potential `SP = ⟨t_stop(A₁) − t_int, t_stop(A₂) − t_int⟩^k`. **[read]** |

**Field methods trade a threshold for a parameter set.** Note how many free parameters STRAP's
fields carry (`γx, γy, αx, αy, d*, t*, β1, β2`) — your `README.md` already flags these as
unverified. Whatever you do, the *comparison* in Part B must hold these fixed or sweep them
honestly; tuning field parameters on the test set is the obvious way to fake a win.

### A4. Severity models (the second factor)

Most SSMs ignore severity entirely, which is a known defect of the family.

| Model | Formulation |
|---|---|
| **Δv** | `Δv(A₁,A₂) = m₂/(m₁+m₂) · (v₂² − v₁²)`. Standard in crash databases. Joksch's model maps Δv → fatality probability: `P ≈ (Δv / 31.74 m·s⁻¹)⁴`. **[read]** |
| **Extended Δv** | Adds masses *and* collision angles (Laureshyn et al.) |
| **CS** — Conflict Severity | Compares Δv at evasive-maneuver time against predicted-collision Δv, scaled by mass ratio `m₂/(m₁+m₂)` |
| **ΔKE** | Predicted absolute change in kinetic energy (used inside CI) |
| **Impact speed** | `v_imp = √(v₁² + 2·a_min·(d − v₁·t_r))` — used inside PRI |

### A5. Learning-based direct risk heads

Rather than deriving risk analytically, train a head to output it.

- **Auxiliary risk prediction task.** Wang et al. (T-ITS 2025) add a driver-centered risk
  indicator as (a) an **input** to the scene encoder, (b) **risk-combined intention queries** as
  decoder priors, and (c) an **auxiliary risk-prediction loss**. Risk is 3-dimensional
  (probability, cost, comprehensive risk) and per-agent risks sum for the target. **[read]**
  — Your `experiments/strap_aux_risk.ipynb` and `src/losses.py` aux-loss hook are the same idea.
- **Risk-scaled loss.** STRAP: `γ_risk = max[exp(Rˢ + Rᵒ) − β, 1]`, `L = γ_risk(L_goal + L_traj)`.
  Doesn't predict risk; re-weights learning toward high-risk samples. **[read]**
- **Accident anticipation.** Frame-level binary risk score from dashcam video; evaluated by
  AP + TTA (see B4). **[snippet]**
- **Ordinal / comparison-based learning.** Learn a risk *ranking* rather than a calibrated
  value — sidesteps the absence of absolute risk labels. **[snippet]**
- **Crash-risk classification on trajectory data.** Multi-level (low/med/high) crash risk from
  NGSIM/highD using CNN-Transformer, LightGBM+SHAP, BiLSTM. Headline accuracies of 96–98% in
  this literature are usually on **lane-change/intention labels**, not rare conflict events —
  do not treat them as comparable to collision prediction. **[snippet]**

### A6. Formal / set-based and distribution-free methods

| Method | What it gives |
|---|---|
| **RSS-DS** | Binary: dangerous iff `d_lat < d_lat,min ∧ d_long < d_long,min` for some actor. Known edge-case failures under braking, varying road surface/slope, and perception uncertainty. **[read]** |
| **SDI / PICUD** | Safe/unsafe by comparing lead vs follower stopping distances under emergency braking |
| **SFF / SP** | Unsafe set = state pairs where *all* safe control policies still collide |
| **Reachability** | Over-approximate occupancy; risk = non-empty intersection (or its measure) |
| **Conformal prediction** | Distribution-free prediction regions around predicted trajectories with a user-set coverage `1−α`, then plan/flag against those regions. Guarantees hold **under exchangeability of calibration and test data** — which breaks exactly in the distribution shift you care about. Variants: adaptive CP for multi-agent, conformalised quantile regression for forward-collision warning. **[snippet]** |

### A7. Aggregation operators — the under-reported design choice

Two methods using the *same* metric can disagree entirely because of how they aggregate. State
yours explicitly.

- **Over time:** `min` (TTC, min-dist), `max` (fields, DRAC), time-exposed (TET), time-integrated
  (TIT), mean (CPI). TIT > TET for fidelity. Divide TET/TIT by `(t_e − t_0)` to remove scenario-
  length dependence. **[read]**
- **Over neighbours:** `max_j` for "risk to this target vehicle"; `mean_{i≠j}` for a
  perspective-neutral scene criticality. Westhofen explicitly flags this as necessary but
  out of scope — so almost nobody reports it. **[read]**
- **Over modes:** weight by mode probability (P-PDRF, P_SMH), or take the worst mode (WTTC-style
  over-approximation). Probability-weighted is the honest default; worst-mode is right only if
  you want sensitivity = 1.
- **Over horizon:** report per-horizon, not just averaged (see B6).

---

## PART B — Measuring the accuracy of risk predictions

This is where most papers are weakest, and where the MTP project can contribute.

### B1. The ground-truth problem (read this before anything else)

**NGSIM and highD contain essentially no real collisions.** Any "collision label" you produce is
itself derived from a surrogate applied to ground-truth future trajectories. So what you are
measuring is:

> Does risk computed from the *predicted* future agree with risk computed from the *actual* future?

That is a legitimate and well-defined question — it isolates predictor quality from
surrogate-definition quality — but it is **not** crash prediction, and the doc/thesis must say so.
Westhofen's `validity` property is exactly this gap: *"the closeness of the metric's measurement
to representing the actual accident probability and severity"* — and they note that for **most
metrics no sources providing evidence for validity exist in the literature**. **[read]**

Options for closing it, in increasing order of cost:

1. **Surrogate-derived labels** (what `plan.md` §51 does). Honest, cheap, circular. Mitigate by
   using a *different*, geometry-aware criterion for labels than for predictions, and by reporting
   sensitivity to the label threshold.
2. **Geometry-aware collision criterion** rather than centre-to-centre distance — bounding-box or
   oriented-rectangle overlap. `plan.md` §50 already specifies this; do not skip it, because
   centre-distance thresholds silently encode vehicle length.
3. **Extreme Value Theory.** Fit block-maxima or peak-over-threshold models to the surrogate
   distribution and extrapolate to crash frequency. Bivariate EVT (pair a proximity indicator
   with an evasive-action indicator) lands closer to observed crash counts than univariate.
   Validity varies considerably across applications. **[snippet]**
4. **Real near-crash data.** 100-Car NDS near-crashes, or simulator-collected safety-critical
   events (Wang et al. built 2088 safety-critical events / 1879 usable from a 22.5 h driving-
   simulator study, because WOMD and Argoverse were *deliberately curated to exclude hazards*).
   **This curation point matters:** models trained on standard benchmarks learn to predict
   "friendlier" agents and systematically under-predict collisions. **[read]**

### B2. Classification metrics (risk as a detector)

Given continuous risk scores and binary labels:

| Metric | Use | Caveat |
|---|---|---|
| **ROC-AUC** | Threshold-free ranking quality | Optimistic under heavy imbalance. Can be 0.95 while AP is 0.145. |
| **PR-AUC / AP** | The headline number under imbalance | Sensitive to base rate — not comparable across datasets with different collision prevalence |
| **Precision / Recall / F1** | At a chosen operating point | Must state the threshold and how it was chosen |
| **FPR / false-alarm rate** | Operational feasibility | A warning system with 10% FPR is unusable |
| **TPR@fixed FPR**, **Recall@fixed precision** | Comparing at a *fixed* operating constraint | The fairest single-number comparison for safety systems |
| **Recall-weighted recall** | Safety asymmetry | Missing a true collision is worse than a false alarm (`plan.md` §54 is right about this) — use F_β with β>1, or report at fixed high recall |

**Threshold selection must be stated.** Jiao et al. pick, per metric, the ROC point closest to
`(0%, 100%)` — a defensible, metric-agnostic rule that avoids tuning each method differently.
Adopt it. **[read]**

ROC-AUC and PR-AUC **can disagree about the same model** — in one study, lengthening the
evaluation window raised PR-AUC and lowered ROC-AUC. Report both. **[snippet]**

### B3. Probabilistic calibration (risk as a probability)

If your method outputs a probability (A2, A3, A5), ranking metrics are not enough — a
miscalibrated score ranks fine and still breaks any downstream threshold or expected-cost rule.

- **Brier score** — strictly proper; uniquely minimised at the true conditional probabilities.
- **Murphy decomposition** — `Brier = Reliability − Resolution + Uncertainty`. A *useful*
  forecast has Brier below its uncertainty term. **Report the decomposition, not just the score**:
  a decent total can hide bad reliability. **[snippet]**
- **NLL / log loss** — also strictly proper. Already the natural fit for STRAP's bivariate
  Gaussian output (`src/losses.py`), and used as a prediction-quality metric by Wang et al. **[read]**
- **Reliability diagram** — observed frequency vs forecast probability; diagonal = calibrated.
  Always pair with the scalar score, because scalars obscure *local* miscalibration.
- **ECE** — convenient but **not a proper scoring rule**; never report it alone.
- **Conformal coverage** — empirical coverage vs nominal `1−α`, plus region size (coverage alone
  is gameable by inflating regions).

### B4. Earliness / lead time

A risk prediction that is correct but late is worthless. This axis is routinely omitted.

| Metric | Definition |
|---|---|
| **TTA / mTTA** | Time-to-Accident: actual event time − first confident warning time, averaged over positives. **Missed events are scored TTA = 0.** **[snippet]** |
| **TTA @ fixed recall** (e.g. 80%) | Removes threshold gaming — TTA alone is trivially inflated by lowering the threshold |
| **TTA at peak-AP threshold** | Alternative anchoring |
| **Warning period (%)** | Fraction of annotated conflict-period timesteps that are warned. Ideally →100%. **[read]** |
| **Warning timeliness (s)** | Interval from the last safe→unsafe warning transition to the critical moment **[read]** |
| **Recall at specific pre-event intervals** | Aggregate scores hide performance at the lead times that matter **[snippet]** |

**Earlier is not better.** Human braking response is 1–1.3 s (<1 s in emergencies), so good
timeliness is *bounded on both sides*: too early distracts, too late misses the avoidance
window. Academic baselines reporting 7–10 s lead times on dashcam benchmarks are flagged in the
literature as physically implausible; a median TTA of ~3.0 s against a human-expert median of
~1.70 s is the realistic regime. **[read + snippet]**

### B5. Risk-as-regression metrics

When the target is a continuous risk value or a collision attribute rather than a label.
Wang et al.'s tailored safety-critical metric suite is the best worked example — 11 metrics,
each in unimodal (`k=1`, max-probability trajectory) and multimodal (`k=6`, best-of-6) form: **[read]**

| Metric | Definition |
|---|---|
| **Traffic risk MSE** (`MSE_risk`) | Error in the predicted driving-risk indicator, for **non-collision** samples |
| **MR_risk** | Fraction of scenarios where normalised risk error > **0.15** |
| **Collision velocity MSE / MR** | Severity-estimation accuracy, for **collision** samples. `MR` threshold **2.5 m/s**. |
| **Collision time MSE / MR** | When the collision happens. `MR` threshold **0.25 s**. |
| **Collision Miss Rate (CMR)** | **A collision miss occurs when *none* of the multimodal predicted trajectories coincides with a surrounding vehicle's trajectory.** The safety-critical failure mode: predicting a collision as a non-collision. |

Their result is the argument for this whole section: **HPNet beat their model on classical
metrics but lost on most safety-critical metrics.** Classical accuracy and risk accuracy rank
models differently. They also report estimated collision probability split by case —
78.8% on collision cases vs 28.0% on non-collision cases — as evidence of discriminative power,
which is a cheap and effective sanity check to copy. **[read]**

### B6. Horizon- and risk-stratified evaluation

Report performance **conditioned on time-to-collision**, not just averaged. STRAP's Table II is
the pattern (NGSIM, 5 s horizon, average RMSE in m): **[read]**

| Model | Coll. 1 s | Coll. 2 s | Coll. 3 s | Coll. 5 s | Non-collision |
|---|---|---|---|---|---|
| STDAN | 4.11 | 3.50 | 2.81 | 2.22 | 2.01 |
| STRAP-B | 3.61 | 3.24 | 2.67 | 2.18 | 2.01 |
| STRAP-R | 3.53 | 3.14 | 2.63 | 2.16 | 1.87 |

Two things to read off this, both important for your project:

1. Accuracy degrades monotonically as collision becomes more imminent — behaviour is less
   predictable precisely where it matters.
2. **STRAP-B and STRAP-R have identical average RMSE (1.78) overall** (Table I) and separate only
   in the stratified view. The risk-scaled loss is invisible to aggregate metrics. If you only
   report average RMSE, you cannot measure the contribution you are trying to study.

`plan.md` §55–56 already specifies this stratification. It is the single highest-value evaluation
decision in the project.

### B7. Downstream / planner-aware evaluation

The strongest critique of the whole field: displacement error does not predict driving safety.

- Two predictions with **identical ADE and FDE** can affect ego planning completely differently.
  `piADE`/`piFDE` weight error by planner sensitivity, reducing to ADE/FDE when sensitivity → 0. **[snippet]**
- In safety-critical scenes with very short inter-agent distances, the *same* displacement error
  produces either a collision or no collision. **[snippet]**
- Measured in a closed-loop simulator: **no significant correlation between static ADE and
  driving performance**. Decomposing the disparity, the **dynamics gap** (static dataset vs
  interactive rollout) accounts for **77.0%** of the inconsistency under an RVO planner and
  **70.3%** under DESPOT; `Dynamic ADE` correlates far better. Contribution to the correlation
  coefficient: dynamics gap 0.77/1.16, multimodal prediction 0.70/0.45, prediction asymmetry
  0.17/0.20, occlusion −0.22/0.16 (RVO/DESPOT). **[read]**
- Also found **not** to explain the gap, contrary to common belief: temporal consistency of
  predictions and the error distribution. **[read]**

Implication for MTP: your pairwise risk layer *is* a crude downstream task, and that is a point
in its favour — risk-classification metrics on predicted trajectories are closer to
planner-relevant evaluation than RMSE is. Say so explicitly; it is a defensible framing of the
contribution.

### B8. Metric-level properties (evaluating the *metric*, not the model)

Westhofen's suitability analysis — the properties you should state for whatever risk measure you
adopt: **[read]**

| Property | Question |
|---|---|
| **Runtime capability** | Computable online within the sampling period? |
| **Target values** | Does research/standards suggest a threshold for this subject type? |
| **Subject type** | Developed for human drivers or automation? (Human reaction times invalidate AV predictions and vice versa.) |
| **Scenario type** | Validity is stated *w.r.t. scenario type*; using a metric outside its type reduces validity. |
| **Inputs** | What entities/properties are required, and are they available? |
| **Output scale** | Nominal / ordinal / interval / ratio. An optimiser needs at least a partial order; a nominal-scale metric cannot be used. |
| **Reliability** | Does a slight change in scene criticality produce a correspondingly slight change in output? |
| **Validity** | Closeness to actual accident probability and severity. **High validity ⇒ reliable; reliable ⇏ valid.** |
| **Sensitivity (TPR)** | Over-approximating metrics have sensitivity 1 — appropriate for data filtering, not for planning |
| **Specificity (TNR)** | A planner needs guarantees that a scene *is* uncritical |
| **Prediction model** | (a) horizon over which predictions are useful; (b) linear time (single worst-case evolution) vs **branching time** (multiple parallel developments) |

Two conclusions they draw that are worth lifting directly:

- **No single metric suffices.** "For adequate coverage in identifying safety-critical scenarios
  in a data set, computing multiple metrics is therefore imperative." Combining metrics —
  `a_req,cond` being the worked example — "can lead to significant improvements in validity."
- Most metrics measure **physics-based symptoms near the end of the accident causal network**
  (small distances, high relative speeds, hard decelerations). Earlier causal factors
  (environment, road-network complexity, rule violations) would detect criticality earlier and
  more reliably, but need rigorous formalisation first.

---

## PART C — A concrete comparison protocol for MTP

Synthesising the above into something runnable against what's already in `src/`.

**C1. Fix the pipeline.**
```
per test sample:
  predicted target traj (B, T_f, 2)  +  neighbour trajs (B, T_f, N_v, 2)
       │                                      │
       ├── RISK SCORE ◄── one of A1/A2/A3/A4  │   ← the method under comparison
       │                                      │
  ground-truth future ──► LABEL ◄── geometry-aware collision criterion (plan.md §50)
```
The label criterion must be **fixed once** across all methods and must differ from the scoring
criterion, or the comparison is circular.

**C2. Report this table per method.**

| Axis | Metrics | Status in repo |
|---|---|---|
| Ranking | ROC-AUC, PR-AUC | ✅ `collision.py` |
| Operating point | Precision, Recall, F1, FPR, Recall@fixed-precision | ✅ (add Recall@precision) |
| Threshold rule | ROC point closest to (0,1), stated per method | ❌ add |
| Calibration | Brier + Murphy decomposition, NLL, reliability diagram | ❌ add (only for probabilistic methods) |
| Earliness | mTTA, TTA@80%-recall, warning period, warning timeliness | ❌ add |
| Risk regression | risk MSE + MR@threshold, collision-time MSE/MR, Collision Miss Rate | ❌ add |
| Stratification | all of the above × {coll-in-1s, 2s, 3s, 5s, non-collision} | ⚠️ `plan.md` §55 specifies, not implemented |
| Trajectory baseline | per-horizon RMSE, minADE/minFDE | ✅ `evaluate.py` |

**C3. Baselines the comparison needs.** At minimum: (i) constant-velocity + TTC, as the
physics-only floor; (ii) the best point-metric from A1 — use **DRAC or BTN**, not min-dist,
given the comparison literature; (iii) the field method (S/O-field) you already have;
(iv) a genuine collision probability from the Gaussian output (A2 / `P_SMH` over the 100 intention
modes) — this is the methodologically novel one and the obvious thesis contribution;
(v) oracle: the same risk score computed on ground-truth futures, which upper-bounds what any
predictor can achieve under your surrogate definition. **(v) is essential** — it separates
"predictor is bad" from "surrogate definition is bad".

**C4. Cheap sanity check to run first.** Mean predicted risk on collision cases vs non-collision
cases, as a ratio. Wang et al. report 78.8% vs 28.0%. If your separation is weak, no amount of
AUC reporting will save it.

---

## PART D — Gaps and honest caveats

1. **Validity is largely unevidenced.** Westhofen: for most metrics, no literature sources
   provide evidence for their properties. Any claim that metric X "measures risk better" is a
   claim about rank correlation with your surrogate labels, not about crash risk.
2. **Surrogate outputs are determined by surrogate assumptions**, not by calibration against
   observed collisions. Stating this plainly is stronger than pretending otherwise.
3. **PR-AUC is base-rate dependent** — do not compare PR-AUC across NGSIM and highD, or across
   different label thresholds, as if the numbers were commensurable.
4. **Standard benchmarks exclude hazards by construction** (WOMD, Argoverse). NGSIM/highD are
   not curated that way, which is actually an argument in your favour, but they contain no real
   crashes either.
5. **Conformal guarantees assume exchangeability** between calibration and test data — the
   assumption fails precisely under the distribution shift that safety-critical evaluation targets.
6. **Field-parameter overfitting** is the easiest way to produce a spurious win. Freeze
   `γx, γy, αx, αy, d*, t*, β1, β2, β` on train/val only; report the values used.
7. **No agreed-upon benchmark protocol exists** for risk-prediction accuracy. There is a real
   opportunity here, but it also means every number you cite from the literature was measured
   under a different protocol and is not directly comparable to yours.

---

## References

**Full text read and used directly**

- Westhofen, Neurohr, Koopmann, Butz, Schütt, Utesch, Neurohr, Gutenkunst & Böde (2022).
  *Criticality Metrics for Automated Driving: A Review and Suitability Analysis of the State of the Art.*
  Archives of Computational Methods in Engineering.
  [Springer](https://link.springer.com/article/10.1007/s11831-022-09788-7) ·
  [DLR PDF](https://elib.dlr.de/187762/1/Westhofen2022_Article_CriticalityMetricsForAutomated.pdf) ·
  supplementary metric property database: `http://purl.org/criticality-metrics`
  — **The single most important reference here. Start with this.**
- Ning, Bian, Zuo & Ergan (2025). *STRAP: Spatial-Temporal Risk-Attentive Vehicle Trajectory
  Prediction for Autonomous Driving.* [arXiv:2507.08563](https://arxiv.org/pdf/2507.08563) (local: `strap.pdf`)
- Wang, Xu, Kuang, Lv, Li & Nie (2025). *Risk-Aware Vehicle Trajectory Prediction Under
  Safety-Critical Scenarios.* IEEE T-ITS. [arXiv:2407.13480](https://arxiv.org/pdf/2407.13480)
  — tailored safety-critical metric suite (B5)
- Wang, Alonso-Mora & Wang (2022). *Probabilistic Risk Metric for Highway Driving Leveraging
  Multi-Modal Trajectory Predictions.* IEEE T-ITS 23:19399–19412.
  [PDF](https://autonomousrobots.nl/assets/files/publications/22-xinwei-ITST.pdf)
  — P-PDRF; the closest analogue to the MTP extension
- Jiao, Calvert, van Cranenburgh & van Lint (2024). *A Unified Probabilistic Approach to Traffic
  Conflict Detection.* Analytic Methods in Accident Research.
  [arXiv:2407.10959](https://arxiv.org/pdf/2407.10959) — warning-effectiveness protocol (B2, B4);
  code: [2D TTC](https://github.com/Yiru-Jiao/Two-Dimensional-Time-To-Collision),
  [100-Car reconstruction](https://github.com/Yiru-Jiao/Reconstruct100CarNDSData)
- *What Truly Matters in Trajectory Prediction for Autonomous Driving?*
  [arXiv:2306.15136](https://arxiv.org/pdf/2306.15136) — the dynamics-gap correlation numbers (B7)
- *Vehicle Trajectory Prediction based Predictive Collision Risk Assessment for Autonomous
  Driving in Highway Scenarios.* [arXiv:2304.05610](https://arxiv.org/pdf/2304.05610)
  — TTC + MDM time-continuous risk function

**Surveys and comparisons (snippet-level; verify before citing numbers)**

- *Risk assessment in autonomous driving: a comprehensive survey of risk sources, methodologies,
  and system architectures* (2025). [Springer](https://link.springer.com/article/10.1007/s43684-025-00112-1)
- Lefèvre, Vasquez & Laugier (2014). *A survey on motion prediction and risk assessment for
  intelligent vehicles.* [Springer](https://link.springer.com/article/10.1186/s40648-014-0001-z)
- *Advances and Applications of Computer Vision Techniques in Vehicle Trajectory Generation and
  Surrogate Traffic Safety Indicators.* [arXiv:2303.15231](https://arxiv.org/pdf/2303.15231) — EVT (BM/POT)
- Do, Saunier & Miranda-Moreno (2025). *Evaluation of Conventional Surrogate Indicators of Safety
  for CAVs in Car Following at Signalized Intersections.* [TRR](https://journals.sagepub.com/doi/10.1177/03611981241265849)
  — TTC/PET can point in opposite directions
- *Can surrogate safety measures explain crash patterns at signalized intersections?*
  [JSR](https://www.sciencedirect.com/science/article/abs/pii/S0022437526000654) — DRAC > TTC > PET
- *A Review on Surrogate Safety Measures Using Extreme Value Theory.*
  [ResearchGate](https://www.researchgate.net/publication/376600303_A_Review_on_Surrogate_Safety_Measures_Using_Extreme_Value_Theory)
- *Bivariate extreme value modeling for road safety estimation.*
  [AAP](https://sciencedirect.com/science/article/abs/pii/S0001457518304007)
- *Crash frequency prediction based on extreme value theory using roadside lidar-based vehicle
  trajectory data.* [PubMed 37769480](https://pubmed.ncbi.nlm.nih.gov/37769480/)

**Collision probability**

- *Efficient Probabilistic Collision Detection for Non-Convex Shapes.*
  [arXiv:1610.03651](https://arxiv.org/pdf/1610.03651) — MC as reference; bounds for real-time
- *A New Approach To Estimate The Collision Probability.* [arXiv:1711.07060](https://arxiv.org/pdf/1711.07060)
  — overlap-based rate + upper bound for arbitrary prediction models
- *Analytic Collision Risk Calculation for Autonomous Vehicle Navigation.*
  [FU Berlin](https://www.mi.fu-berlin.de/inf/groups/ag-ki/publications/analytic-collision-risk-calculation/CollisionRisk.pdf)
  — collision *state* vs collision *event* probability
- *Reachability-Based Confidence-Aware Probabilistic Collision Detection in Highway Driving.*
  [arXiv:2302.07109](https://arxiv.org/pdf/2302.07109)

**Evaluation, calibration, planner-awareness, conformal**

- Ivanovic & Pavone. *Rethinking Trajectory Forecasting Evaluation.* [arXiv:2107.10297](https://arxiv.org/pdf/2107.10297) — piADE/piFDE
- Ivanovic & Pavone. *Injecting Planning-Awareness into Prediction and Detection Evaluation.* [arXiv:2110.03270](https://arxiv.org/pdf/2110.03270)
- *Measuring What Matters: Scenario-Driven Evaluation for Trajectory Predictors.* [arXiv:2512.12211](https://arxiv.org/pdf/2512.12211)
- *Classifier Calibration: A survey on how to assess and improve predicted class probabilities.* [arXiv:2112.10327](https://arxiv.org/pdf/2112.10327)
- *Reliability, Sufficiency, and the Decomposition of Proper Scores.* [arXiv:0806.0813](https://arxiv.org/pdf/0806.0813) — Murphy decomposition
- *Safe Planning in Dynamic Environments using Conformal Prediction.* [arXiv:2210.10254](https://arxiv.org/abs/2210.10254)
- *Adaptive Conformal Prediction for Motion Planning among Dynamic Agents.* [arXiv:2212.00278](https://arxiv.org/pdf/2212.00278)
- *BADAS: Context Aware Collision Prediction Using Real-World Dashcam Data.* [arXiv:2510.14876](https://arxiv.org/html/2510.14876v1) — realistic TTA windows
- *Accident Anticipation via Temporal Occurrence Prediction.* [arXiv:2510.22260](https://arxiv.org/pdf/2510.22260) — recall at fixed pre-event intervals
- *Hierarchical Spatio-Temporal Attention Network with Adaptive Risk-Aware Decision for Forward
  Collision Warning.* [arXiv:2511.19952](https://arxiv.org/pdf/2511.19952) — conformalised quantile regression; lead time

**Risk fields**

- Kolekar et al. (2020). Driver's Risk Field — via STRAP ref. [20]
- Mullakkal-Babu et al. PDRF — via Wang et al. 2022 ref. [3]
- Zuo et al. (2025). Composite safety potential field — via STRAP ref. [22]
- *Quantification Method of Driving Risks for Networked Autonomous Vehicles Based on Molecular
  Potential Fields.* [Appl. Sci. 15:1306](https://doi.org/10.3390/app15031306)
- *A real-time synthesized driving risk quantification model based on driver risk
  perception-response mechanism.* [TR-C](https://www.sciencedirect.com/science/article/abs/pii/S0968090X25000774)
- *Risk Assessment of Roundabout Scenarios in Virtual Testing Based on an Improved Driving Safety
  Field.* [PMC11397973](https://pmc.ncbi.nlm.nih.gov/articles/PMC11397973/)
