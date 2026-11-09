# MTP Experiments: From STRAP Reproduction to Hyperparameter Tuning

*Prabhat Kumar Sahu · November 2026*

This document follows every experiment in the project in order: why it was run, which papers
it draws on, what was optimised, what we expected, what happened, and what it taught us. It
ends with a methodology review: what would make the conclusions stronger.

The detailed write-ups for individual stages are linked where they apply:

| Stage | Detailed document |
|---|---|
| Original plan | [plan.md](plan.md) |
| First correction plan | [STRAP_REPRODUCTION_PLAN.md](STRAP_REPRODUCTION_PLAN.md) |
| Paper-vs-code audit, 25 assumptions | [PAPER_VS_IMPLEMENTATION.md](PAPER_VS_IMPLEMENTATION.md) |
| Fixes for the audit | [SOLUTIONS_PAPER_VS_IMPLEMENTATION.md](SOLUTIONS_PAPER_VS_IMPLEMENTATION.md) |
| Alternative models and their papers | [experiments/MODEL_ALTERNATIVES.md](experiments/MODEL_ALTERNATIVES.md) |
| Risk-measure literature | [RISK_PREDICTION_METHODOLOGIES.md](RISK_PREDICTION_METHODOLOGIES.md) |
| Risk prediction study | [RISK_PREDICTION_RESULTS.md](RISK_PREDICTION_RESULTS.md) |
| Tuning study | [RISK_TUNING_RESULTS.md](RISK_TUNING_RESULTS.md) |

---

## Contents

1. [The research question](#1-the-research-question)
2. [Common setup](#2-common-setup)
3. [Experiment 1: First STRAP implementation](#experiment-1-first-strap-implementation-september-2026)
4. [Experiment 2: STRAP v2, the first full NGSIM run](#experiment-2-strap-v2-the-first-full-ngsim-run-early-october)
5. [Experiment 3: Paper-vs-implementation audit and fixes](#experiment-3-paper-vs-implementation-audit-and-fixes-1920-october)
6. [Experiment 4: Risk-field calibration sweep](#experiment-4-risk-field-calibration-sweep-20-october)
7. [Experiment 5: STRAP v3, the corrected pipeline](#experiment-5-strap-v3-the-corrected-pipeline-2526-october)
8. [Experiment 6: The 39-model benchmark](#experiment-6-the-39-model-benchmark-2331-october)
9. [Experiment 7: Collision-label profiling](#experiment-7-collision-label-profiling-1-november)
10. [Experiment 8: Risk prediction from trajectory predictions](#experiment-8-risk-prediction-from-trajectory-predictions-1-november)
11. [Experiment 9: Hyperparameter tuning and its effect on risk](#experiment-9-hyperparameter-tuning-and-its-effect-on-risk-68-november)
12. [What the experiments say together](#what-the-experiments-say-together)
13. [Making the research methodology stronger](#making-the-research-methodology-stronger)
14. [Appendix: papers referenced](#appendix-papers-referenced)

---

## 1. The research question

**STRAP** (Ning, Bian, Zuo & Ergan, 2025, arXiv:2507.08563) predicts the 5-second future of a
vehicle from 3 seconds of history. It adds a *risk potential field* to the input and to the
loss, so that the model attends to dangerous neighbours and is penalised more for errors in
risky scenes. On NGSIM it reports 1.78 m average RMSE (3.52 m at 5 s).

The MTP project uses STRAP as a starting point for a different question:

```text
history of a traffic scene
   → predict future trajectories            (STRAP and alternatives)
   → compare predicted trajectories pairwise
   → estimate collision / interaction risk   (MTP extension)
   → judge whether a collision is likely
```

So there are two layers to get right, and the plan ([plan.md §1.3](plan.md)) deliberately
kept them separate:

1. **Trajectory prediction:** reproduce STRAP, then compare it with alternatives.
2. **Risk prediction:** turn predicted trajectories into a collision-risk judgement, and
   measure how good that judgement is.

The experiments below follow that order. The main finding, built up across Experiments 6, 8
and 9, is that **the two layers do not agree**. The models that predict trajectories best are
not the ones whose predictions give the best risk judgements, and making a model better at
trajectories made it worse at risk.

---

## 2. Common setup

These choices hold from Experiment 5 onward unless an experiment says otherwise. Experiments
1–4 are where most of them were established.

| Item | Setting |
|---|---|
| Data | NGSIM US-101 and I-80 freeway recordings (Socrata API), 3 recording periods per site, converted to metres |
| Window | 8 s at 10 Hz: 30 frames of history, 50 frames of future; sliding stride 10 frames (1 s) |
| Targets | every vehicle with a complete 80-frame window |
| Neighbours | up to 15, selected when S-field or O-field risk > 0.005 (the paper's rule) |
| Features | 11 per vehicle per step: relative x, y, speed, lateral speed (always 0 in NGSIM), acceleration, type, relative lane, length, width, S-risk, O-risk |
| Split | each of the 6 recordings cut 70 / 10 / 20 in time with an 80-frame guard gap, then pooled. 540,149 train / 86,077 val / 121,130 test samples |
| Normalisation | z-score with training-split statistics only |
| Output | one bivariate Gaussian per future step: μx, μy, σx, σy, ρ |
| Metric | RMSE of the predicted mean at 1, 2, 3, 4, 5 s, and their average (the paper's convention) |
| Default training budget | Adam, lr 5e-4 decayed ×0.6 per epoch, batch 128, 12 epochs, weight decay 1e-5, gradient clip 1.0 (the paper's settings) |
| Checkpoint | lowest validation loss (rule "S4") |
| Hardware | DSLab: AMD EPYC 7313 (32 threads), 125 GB RAM, one RTX A5000 24 GB |

---

## Experiment 1: First STRAP implementation (September 2026)

**Why.** Before any comparison, the project needed a working STRAP: the encoders, the risk
field, the 100 intention modes and the risk-scaled loss, end to end.

**Papers involved.**
- STRAP (Ning et al., 2025): the architecture, Eq. 1–14, Tables I–III.
- The papers STRAP builds on: STDAN (Chen et al., IEEE T-ITS 2022) for the bivariate-Gaussian
  output, MTR (Shi et al., NeurIPS 2022) for k-means intention modes, and TNT for goal-based
  prediction.

**Setup.** A first implementation under `ling/` on synthetic data, then the NGSIM loader
(Socrata API) on US-101, with 21 unit tests and train/eval/predict modes.

**Objective function (STRAP-R, paper Eq. 11–14):**

$$
\mathcal{L}_{goal} = \lVert \hat G - G \rVert^2, \qquad
\mathcal{L}_{traj} = \frac{1}{T_f}\sum_t \left[\lVert \hat\mu_t - \mu_t \rVert^2 - \log \mathcal N(\mu_t \mid \hat\mu_t, \hat\sigma_t, \hat\rho_t)\right]
$$

$$
\gamma_{risk} = \max\left[\exp(R^s + R^o) - \beta,\ 1\right], \qquad \mathcal L = \gamma_{risk}\,(\mathcal L_{goal} + \mathcal L_{traj})
$$

STRAP-B drops γ_risk, so it trains on L_goal + L_traj alone.

**Expected.** A pipeline that runs and trains; numbers in the paper's range once on real data.

**Actual.** The pipeline ran and the tests passed, but a first audit of the code against the
paper ([STRAP_REPRODUCTION_PLAN.md](STRAP_REPRODUCTION_PLAN.md)) found critical
mismatches:
- target future labels left in absolute coordinates while inputs were relative;
- swapped longitudinal and lateral axes in the risk code;
- the acceleration feature dropped;
- an O-field sign error that treated separating vehicles as closing;
- an "RMSE" that was actually mean displacement error.

**Observations.**
- A model can train and produce plausible losses while computing the wrong quantity. Several
  of these bugs would never have shown up as a crash.
- This set the working rule for the rest of the project: **no retraining until the data and
  model contracts are pinned by tests.**

---

## Experiment 2: STRAP v2, the first full NGSIM run (early October)

**Why.** To get a first real-data number against the paper's 1.78 m, and a first look at the
MTP collision layer.

**Papers involved.**
- STRAP, for the protocol and the reference numbers.
- The 11 classical surrogate safety measures used for collision scoring (TTC, PET, THW, TET,
  TIT, DRAC and others), catalogued in Westhofen et al. (2022).

**Setup.**
- Data: US-101 and I-80, a single global temporal split, 567,653 samples.
- Training: the paper's budget. The collision layer took the predicted target trajectory and
  rolled neighbours forward at constant velocity.
- Collision label: true centre distance < 2 m within 5 s.

**Objective function.** STRAP-R as above. The collision layer has no training objective; it is
scored by precision, recall, F1, PR-AUC and ROC-AUC against the label.

**Expected.** Trajectory RMSE near the paper's 1.78 m. Collision measures well above chance
(ROC-AUC > 0.5).

**Actual.**

| | 1 s | 2 s | 3 s | 4 s | 5 s | Average |
|---|---|---|---|---|---|---|
| STRAP v2 | 0.62 | 1.48 | 2.56 | 3.91 | **5.49** | **2.81** |
| Paper STRAP-R | 0.37 | 0.94 | 1.61 | 2.47 | 3.52 | 1.78 |

**The collision layer failed outright.** 157,654 of 1,641,679 pairs (**9.6 %**) were
labelled collisions, and all 11 measures scored ROC-AUC between 0.488 and 0.506, which is
chance.

**Observations.**
- **The 9.6 % collision rate was the clue.** Real freeway traffic has nowhere near that many
  near-collisions. Tracing it led to the most consequential bug in the project (Experiment 3,
  Problem 1): NGSIM's three recording periods reuse the same frame and vehicle ids, so scenes
  overlaid vehicles from different times. Phantom vehicles sat on top of real ones and created
  phantom collisions.
- The trajectory error was 58 % above the paper at the average and 56 % at 5 s, too large to
  be explained by protocol differences alone.

**Other factors.** v2 used one global temporal cut, so validation and test were I-80 only;
the test number was effectively an I-80 PM-peak number.

---

## Experiment 3: Paper-vs-implementation audit and fixes (19–20 October)

**Why.** v2's gap to the paper and its broken collision labels showed that the
implementation could not be trusted line by line. A systematic audit was needed: every input,
feature, output and assumption, compared with the paper.

**Papers involved.**
- STRAP, every section and equation.
- Zuo, Bian, Zuo & Ozbay (2025), *Composite Safety Potential Field* (arXiv 2504.21158), STRAP's
  reference [22]. It is the actual source of the S-field and O-field. It shares two authors
  with STRAP, and STRAP cites a different paper ([21]) for it, which looks like a citation slip.
- CS-LSTM (Deo & Trivedi, 2018), for the RMSE convention.

**Setup.** A table-by-table comparison
([PAPER_VS_IMPLEMENTATION.md](PAPER_VS_IMPLEMENTATION.md)) that labels every item as
*Matches*, *Paper silent*, *Deviates* or *Extension*. It led to an assumption register of 25
choices, 5 rated high-impact, and a solutions document with options, rejected alternatives
and the reasons
([SOLUTIONS_PAPER_VS_IMPLEMENTATION.md](SOLUTIONS_PAPER_VS_IMPLEMENTATION.md)).

**The problems found, and what was done about them:**

| # | Problem | Fix |
|---|---|---|
| 1 | Recording periods overlaid (phantom vehicles) | Key by (location, period, id) |
| 1b | Absolute lane id means different things per site | Relative lane (target lane − vehicle lane) |
| 2 | γ_risk computed from the model's own predicted risk, so the model could lower its loss by predicting "safe" | γ_risk from ground-truth future risk, detached |
| 3 | Intention codebook trainable, and silently reset at evaluation | Frozen buffer |
| 4 | Risk-field parameters were placeholders (S-field effectively dead) | Full calibration from [22] (Experiment 4) |
| 5 | Validation and test were I-80 only | Each recording split 70/10/20 in time, with guard gaps |
| 6 | Encoder stacking order differed from Fig. 2 | Interleaved S, T, S, T, S, T |
| 9 | Nearest-15 fallback let in vehicles the paper's rule rejects | Fallback removed; zero-neighbour scenes allowed |
| 7, 8 | Target history positions always 0; lateral velocity always 0 | Deprioritised, with measured impact (≈ 0.02–0.03 m lateral RMSE) |
| 10 | Table II risk-level breakdown | Deprioritised |

**Objective function.** None; this experiment is an audit. Each fix carries a test in
`tests/test_solutions.py`.

**Expected.** That fixing these would bring the reproduction close to the paper.

**Actual.** Fixes 1–6, 9 and D were implemented and tested (Experiment 5 measures their
effect). The audit also showed that the paper leaves many decisions open:
- the split method and window stride;
- every risk-field constant;
- how the 100 intention modes collapse into one output Gaussian;
- the RMSE formula;
- the checkpoint rule.

**Observations.**
- **"Reproduction" of this paper can only ever be approximate.** The paper does not specify
  enough to reproduce it exactly, so every result has to be read as "STRAP under our stated
  assumptions".
- Problem 2 is a general trap worth naming: **a loss weight computed from the model's own
  outputs gives the model a way to cheat.**
- The paper has typos in Eq. 4, Eq. 7 and Eq. 10 and the probable citation slip above; these
  are recorded for the report.

---

## Experiment 4: Risk-field calibration sweep (20 October)

**Why.** The paper gives no values for any risk-field parameter. These parameters drive four
things at once:
- the input risk features;
- which vehicles count as neighbours;
- the decoder's risk field;
- the loss weight γ_risk, which is *exponential* in the risk.

So the placeholders mattered, and a principled choice was needed.

**Papers involved.**
- Zuo et al. (2025) [22], for calibrated values: speed-dependent γx(v) and αx(v), γy = 1.431,
  αy = 4.996, d* = ½(wᵢ + wⱼ), β1 = 10, t* = 7.5 s, β2 = 2, with bumper-to-bumper gaps.
- STRAP §IV-A, for the indirect targets: threshold 0.005, about 6 neighbours on average.

**Setup.**
- Data: training data only (US-101, period 1), 1.67 M target–vehicle pairs, of which 770 are
  real conflicts (lateral overlap and bumper gap < 2 m within 5 s).
- Method: one parameter varied at a time around the [22] values.

**Objective function.** For each setting:
- the ROC-AUC with which the risk ranks the real conflicts;
- the average neighbour count;
- the mean loss weight γ_risk.

**Expected.** That [22]'s values, being the field's own calibration, would rank conflicts
well, and that the placeholders would be weak.

**Actual.**

| Setting | Avg neighbours | Conflict AUC | Mean γ_risk |
|---|---|---|---|
| Placeholder values (γ = 1.5 m, centre distance) | 8.5 | 0.910 | 2.3 |
| Plain 1 / (1 + distance) | 15 | 0.968 | — |
| **[22] calibration** | **3.2** | **0.985** | **3.0** |
| [22] with γx × 10 | 14.0 | 0.978 | 5,838 |
| [22] with γx × 0.1 | 1.5 | 0.840 | 1.7 |

**Observations.**
- **The placeholders ranked real conflicts worse than plain distance** (0.910 vs 0.968). The
  S-field was "dead": above 0.005 in only 0.14 % of pairs.
- **The [22] values sit at or near the best AUC for every parameter.** Neither extreme is
  safe: a small scale makes the risk blind, and a large one makes everything risky and blows
  up the exponential loss weight (5,838× at γx × 10).
- The sweep used training data only, never the test split, so choosing [22] did not tune
  anything on test.

**Other factors.** The calibrated field gives 3.2 neighbours on average, not the paper's 6.
With lateral velocity fixed at 0 (Problem 8), the O-field only sees same-lane closing.

---

## Experiment 5: STRAP v3, the corrected pipeline (25–26 October)

**Why.** To measure what the Experiment 3 and 4 fixes did, using the paper's training budget.

**Papers involved.** STRAP, for the protocol and the reference numbers.

**Setup.**
- Model: STRAP-R with the [22] risk field and the corrected pipeline.
- Split: the new per-recording split, 121,130 test samples covering both sites.
- Training: 12 epochs.

**Objective function.** STRAP-R, γ_risk from ground-truth future risk (Problem 2 fix).

**Expected.** A large improvement over v2, moving towards the paper's 1.78 m.

**Actual.**

| | 1 s | 3 s | 5 s | Average | US-101 | I-80 | Zero-neighbour scenes |
|---|---|---|---|---|---|---|---|
| STRAP v2 | 0.62 | 2.56 | 5.49 | 2.81 | — | — | — |
| **STRAP v3** | 0.70 | 2.25 | 4.53 | **2.44** | 2.49 | 2.39 | 2.89 |
| Paper STRAP-R | 0.37 | 1.61 | 3.52 | 1.78 | — | — | — |

v3 averaged 3.77 neighbours per sample (paper: 6), and 2.5 % of samples had no neighbours.

**Observations.**
- **The fixes cut average RMSE by 13 %** (2.81 → 2.44) and the 5 s error by almost a metre.
- **The selected checkpoint was from epoch 2 of 12.** The best checkpoint is chosen by
  lowest validation *loss*, and the validation loss bottomed out at epoch 2 (val RMSE 2.37),
  while validation RMSE kept improving to **2.12 at epoch 12**. The SOLUTIONS doc had already
  flagged why: in eval mode the predicted σ collapses towards 1e-5, so the validation NLL, and
  with it the validation loss, becomes huge and erratic. **v3's test number is therefore
  measured on an under-trained checkpoint.** At its final epoch it would likely score close to
  the notebook STRAP-R in Experiment 6 (2.16).
- The v2 → v3 comparison mixes the fixes with a different split and test set, so it is not a
  clean before/after for any single fix.

---

## Experiment 6: The 39-model benchmark (23–31 October)

**Why.**
- STRAP was still about 0.4 m from its published number.
- The literature contained several models reporting a lower 5 s NGSIM error than STRAP.
- Published numbers come from different papers' tables, splits and preprocessing, so the only
  fair comparison is to **run every model on one pipeline**.

The benchmark also served as the place to test STRAP's own claims (its ablations) and
alternative ways of using risk.

**Papers involved.** Each notebook implements one paper and opens with a table of where it
departs from that paper. The full list with links is in
[MODEL_ALTERNATIVES.md](experiments/MODEL_ALTERNATIVES.md).

| Family | Models (papers) |
|---|---|
| Baselines | Constant velocity; target-only LSTM |
| STRAP and its ablations | STRAP-R, STRAP-B, −SE, −TE, −RFF (Ning et al., 2025) |
| Tests of the risk idea on STRAP | risk as input features (TTC/TET/TIT, after Liao et al.'s cognitive-driven model, 2024); kinematic bicycle-model output (after X-TRACK, 2025); auxiliary risk head (Wang et al., T-ITS 2025) |
| RNN and social pooling | S-LSTM (Alahi et al., CVPR 2016), CS-LSTM (Deo & Trivedi, CVPRW 2018), NLS-LSTM, MHA-LSTM, CF-LSTM, MFP, WSiP |
| Graph and attention | STDAN (Chen et al., T-ITS 2022), iNATran, DACR-AMTP, FHIF, GRIP++ (Li et al., 2019), BAT (Liao et al., 2024), HLTP (Liao et al., 2024), cognitive-driven (Liao et al., 2024), GaVa, MIAT (Raskoti et al., IROS 2025), PiP-noPlan (Song et al., ECCV 2020) |
| Generative | Social GAN, MATF-GAN, TS-GAN; diffusion: CDSTraj (Liao et al., 2024), C2F-TP (Wang et al., AAAI 2025), Westny et al. (2024) |
| Physics and goal-based | physics-informed (Liao et al., IJCAI 2024), GNP (Gan et al., 2024), X-TRACK (2025) |
| Large transformer / state-space | MTR agent-only (Shi et al., NeurIPS 2022), Trajectory Mamba (Huang et al., CVPR 2025) |

**Fidelity labels.** Eight notebooks are **LOOSE**: they were built from the model's name and a
one-line description because no paper or code was available (iNATran, DACR-AMTP, FHIF, GaVa,
TS-GAN, MATF-GAN, CF-LSTM, Westny diffusion). Their results do not represent the published
models, and every results row carries the flag.

**Setup.**
- **Model interface:** every model plugs into one interface, `forward(states, mask, dims) →
  (trajectory Gaussian, neighbour goals, extra)`.
- **Training:** every model trains with **the same budget** (the paper's 12 epochs) on the same
  data, split and seed. The fixed budget was deliberate: tuning per model would favour
  whichever model was tuned most.
- **Execution:** 39 notebooks ran on DSLab through `run_all.py` in a resumable batch of about
  42.8 h, with collision scoring off. They were checked first with a debug smoke pass.

**Objective function.** All non-STRAP models train on the *basic* loss: trajectory MSE +
bivariate-Gaussian NLL + neighbour-goal MSE. STRAP-R variants use the risk-scaled loss. Two
kinds of model add an auxiliary term through an `aux_loss` hook:
- diffusion models add their denoising loss;
- manoeuvre-based models add a manoeuvre cross-entropy, on labels derived from the true future.

**Expected.**
1. STRAP-R would beat STRAP-B, as the paper claims in risky scenes, and the classical
   baselines.
2. The paper's ablation ordering would hold: −SE hurts most, then −TE and −RFF.
3. Models that reported lower NGSIM errors than STRAP would also beat it here.
4. The diffusion models, with the best published 5 s numbers, would rank near the top.

**Actual: the top 10 and the reference rows** (test RMSE, metres):

| Rank | Model | 1 s | 5 s | Average | Parameters |
|---|---|---|---|---|---|
| 1 | MIAT | 0.449 | 4.055 | **2.0716** | 194,986 |
| 2 | Trajectory Mamba | 0.445 | 4.094 | 2.0860 | 121,207 |
| 3 | X-TRACK | 0.459 | 4.091 | 2.0903 | 166,405 |
| 4 | DACR-AMTP (LOOSE) | 0.460 | 4.114 | 2.1055 | 254,150 |
| 5 | STDAN | 0.455 | 4.136 | 2.1136 | 217,483 |
| 6 | MFP | 0.454 | 4.139 | 2.1139 | 77,786 |
| 7 | Cognitive-driven | 0.462 | 4.137 | 2.1139 | 227,979 |
| 8 | iNATran (LOOSE) | 0.474 | 4.129 | 2.1176 | 173,381 |
| 9 | Physics-informed | 0.478 | 4.137 | 2.1240 | 212,684 |
| 10 | MHA-LSTM | 0.464 | 4.160 | 2.1300 | 146,825 |
| 11 | STRAP + risk input features | 0.502 | 4.157 | 2.1399 | 527,881 |
| 12 | STRAP + auxiliary risk head | 0.492 | 4.169 | 2.1419 | 531,979 |
| 17 | STRAP-B | 0.533 | 4.167 | 2.1548 | 527,689 |
| 20 | **STRAP-R** | 0.499 | 4.181 | **2.1589** | 527,689 |
| 21 | STRAP −RFF | 0.513 | 4.203 | 2.1688 | 527,689 |
| 23 | STRAP + kinematic output | 0.505 | 4.250 | 2.1831 | 527,689 |
| 26 | CS-LSTM | 0.474 | 4.502 | 2.2593 | 192,117 |
| 28 | S-LSTM | 0.474 | 4.512 | 2.2635 | 351,861 |
| 34 | STRAP −TE | 0.774 | 4.879 | 2.6111 | 527,689 |
| 35 | STRAP −SE | 0.497 | 5.735 | 2.8188 | 527,689 |
| 36 | LSTM (target only) | 0.520 | 5.742 | 2.8503 | 35,962 |
| 37 | CDSTraj (diffusion) | 1.140 | 5.286 | 2.9256 | 210,492 |
| 38 | Constant velocity | 0.602 | 6.326 | 3.1728 | 0 |
| 39 | C2F-TP (diffusion) | 5.148 | 6.700 | 6.3684 | 475,334 |

The full table is in `experiments/comparison_rmse.csv`.

**How the reproduction compares with the paper's Table I:**

| Model | Paper average | Ours | Paper 5 s | Ours 5 s |
|---|---|---|---|---|
| Constant velocity | 3.15 | 3.17 | 6.25 | 6.33 |
| S-LSTM | 2.38 | 2.26 | 4.55 | 4.51 |
| CS-LSTM | 2.32 | 2.26 | 4.50 | 4.50 |
| STDAN | 1.87 | 2.11 | 3.65 | 4.14 |
| STRAP-B | 1.78 | 2.15 | 3.51 | 4.17 |
| STRAP-R | 1.78 | 2.16 | 3.52 | 4.18 |

**Observations.**

1. **Our data matches the paper's data; our strong models do not.** Constant velocity is
   within about 2 % of the paper at every horizon, and S-LSTM and CS-LSTM match or beat the
   paper's numbers. So the windows, horizons, units and difficulty of our test set are
   comparable with the paper's. Yet STDAN and STRAP fall 0.5–0.7 m short at 5 s. The
   reproduction gap is specific to the modern attention models. That points to their
   implementation or training (unspecified details, the per-timestep relative coordinates,
   1 s stride and so fewer samples, the 12-epoch budget), not to the dataset.
2. **STRAP-R = STRAP-B** (2.159 vs 2.155). This matches the paper's own Table I, where both
   score 1.78. The risk-scaled loss does not change average accuracy. The paper's claimed
   benefit is in risky scenes (Table II), which this benchmark does not measure.
3. **The ablations only partly reproduce:**

   | Ablation | Paper change | Our change |
   |---|---|---|
   | −SE (no spatial encoder) | +14.6 % | **+30.6 %** |
   | −TE (no temporal encoder) | +3.9 % | +20.9 % |
   | −RFF (no risk feature fusion) | +3.4 % | **+0.5 %** |

   The −TE figure is **not trustworthy**: its checkpoint was selected at epoch 1 (val RMSE
   2.47) by the validation-loss rule, while its validation RMSE reached 2.12 at epoch 12,
   close to full STRAP (2.09). By validation RMSE the temporal encoder adds about 1–2 %. So in
   our hands the spatial encoder matters a lot, and the temporal encoder and the risk feature
   fusion matter little.
4. **None of the three "risk ideas" moves the needle.** Risk as input features (−0.9 %), an
   auxiliary risk head (−0.8 %) and a kinematic output layer (+1.1 %) are all within about
   1 % of STRAP-R. These are single runs, so they are within plausible seed noise.
5. **The diffusion models failed.** C2F-TP (6.37 m) is worse than constant velocity, and
   CDSTraj has a 1.14 m error at 1 s. Their published numbers (2.85 m at 5 s) depend on
   sampling and training details our implementations did not recover in 12 epochs. Treat
   these as failed reimplementations, not evidence against diffusion.
6. **The top 10 are within 2.8 %** (2.072–2.130 m), and the LOOSE models rank as high as
   faithful ones. At this spread, single-seed RMSE cannot reliably rank models. That
   motivated Experiment 8: if RMSE cannot separate them, does risk accuracy?
7. **Smaller models do as well.** The top 10 have 78 k–255 k parameters; STRAP has 528 k.

**Other factors.**
- One seed per model.
- The checkpoint rule (S4) is affected by the σ-collapse issue.
- The 12-epoch budget favours fast-converging models. Experiment 9 later shows that most
  models were still improving at epoch 12.
- An earlier internal summary ([MODEL_COMPARISON.md](experiments/MODEL_COMPARISON.md))
  draws two conclusions this evidence does not support:
  - that risk-as-feature beats risk-as-loss. On the same STRAP backbone the gap is 0.9 %,
    within noise.
  - that lower displacement error will mean fewer collision false alarms. Experiment 9
    contradicts this.

---

## Experiment 7: Collision-label profiling (1 November)

**Why.** NGSIM contains essentially no real crashes, so every "collision" label is a
distance rule applied to the true future. Before scoring any risk measure against those
labels, the project needed to know what the labels actually contain.

**Papers involved.**
- Montanino & Punzo (2015), *Trajectory data reconstruction and simulation-based validation
  against macroscopic traffic patterns*, Transportation Research Part B 80. They document
  NGSIM trajectory errors.
- Coifman & Li (2017), *A critical evaluation of the Next Generation Simulation (NGSIM)
  vehicle trajectory dataset*, Transportation Research Part B 105.

**Setup.** A model-free pass over the test split (`profile_collision_labels.py`): 121,130
samples and 1,814,838 target–neighbour pairs. Labels at 1, 2, 3 and 5 m. Independent
encounters counted as unordered vehicle pairs.

**Objective function.** None; descriptive statistics.

**Expected.** A small but usable set of genuine near-misses.

**Actual.**

| Threshold | Positive pairs | Base rate | Independent events | Events with vehicle boxes overlapping |
|---|---|---|---|---|
| 1 m | 261 | 0.014 % | 13 | 13 (100 %) |
| **2 m** | **977** | **0.054 %** | **101** | **36 (36 %)** |
| 3 m | 9,829 | 0.54 % | 914 | 54 (6 %) |

At 2 m:
- 83 % of positives are side by side, not rear-end;
- 21.5 % are already within 2 m when prediction starts;
- there are no heavy vehicles.

**Observations.**
- **The effective sample size is about 100, not 977.** Overlapping windows repeat each
  encounter about 9 times. Any confidence interval must resample events, not pairs. This
  became the basis of the bootstrap in Experiment 9.
- **A third of the 2 m positives are tracking artefacts.** NGSIM has no crashes, yet the
  boxes physically overlap. At 1 m every event is one vehicle tracked twice, so the 1 m label
  is unusable.
- **The 2 m label is a side-by-side label.** That explains why the rear-end measures (TTC,
  DRAC, PSD) rank poorly later.
- 3 m is a cleaner but milder label: close proximity rather than conflict.

---

## Experiment 8: Risk prediction from trajectory predictions (1 November)

**Why.** Experiment 6 left the top models tied on RMSE. The MTP question is not "which model
predicts paths best" but **"which model's predictions support the best collision-risk
judgement"**. This was the first time risk was measured in the project; the 39-model batch
ran with collision scoring off.

**Papers involved.**
- **Measures:** Westhofen et al. (2022), *Criticality Metrics for Automated Driving*, the main
  catalogue.
- **Collision probability and severity:**
  - Wang, Alonso-Mora & Wang (2022), *Probabilistic Risk Metric for Highway Driving* (P-PDRF);
  - Shalev-Shwartz et al. (2017), RSS, for `rss_ds` / `rss_margin`;
  - Zuo et al. (2025) [22], for the calibrated risk fields.
- **Evaluation protocol:**
  - Jiao et al. (2024), *A Unified Probabilistic Approach to Traffic Conflict Detection*, for
    the metric-agnostic threshold rule;
  - Wang et al. (T-ITS 2025), for the safety-critical metric suite;
  - *What Truly Matters in Trajectory Prediction* (arXiv:2306.15136), for why static ADE need
    not predict downstream safety.

**Setup.**
- **Models:** the 10 best by RMSE.
- **Measures:** 30 risk measures:
  - 20 deterministic surrogates (TTC, true TTC, PET, THW, TET, TIT, DRAC, PSD, BTN, STN and
    others);
  - 2 collision probabilities: an analytic Gaussian box integral and Monte Carlo;
  - 3 calibrated risk fields;
  - 3 severity-weighted measures, including P-PDRF;
  - 2 RSS measures.
- **Neighbour-future sources:**

  | Source | Target | Neighbours | What it isolates |
  |---|---|---|---|
  | `cv` | model | constant velocity | continuity with the 11-measure definition |
  | `cv_scene` | model | constant velocity, scene's own neighbours | matched control for `pred` |
  | `pred` | model | the model also predicts each neighbour | the predictor's contribution |
  | `oracle` | ground truth | ground truth | the ceiling |

- **Evaluation axes:**
  - ROC-AUC and PR-AUC;
  - an operating point chosen by one shared rule;
  - calibration (Brier score with the Murphy decomposition);
  - earliness;
  - horizon stratification;
  - regression against the oracle.

**Objective function.** No training. Each measure is scored by ROC-AUC against the label
"true minimum centre distance < 2 m within 5 s". The threshold is swept at 1, 2, 3 and 5 m.
The decision threshold is the ROC point nearest (0, 1), the same rule for every measure, so
no measure gets a hand-tuned operating point.

**Expected.**
1. Better trajectory predictors (lower RMSE) would give better risk judgements.
2. Model-predicted neighbours would beat constant velocity, since the model "knows more".
3. The probability measures would be roughly calibrated.

**Actual** (details in [RISK_PREDICTION_RESULTS.md](RISK_PREDICTION_RESULTS.md)):

| Finding | Evidence |
|---|---|
| With constant-velocity neighbours, all 10 models are indistinguishable | ROC-AUC spread ≤ 0.022 on every headline measure under `cv` / `cv_scene` |
| With predicted neighbours, models separate sharply | spread 0.11–0.28 under `pred`; inatran 1st, cognitive 2nd, dacr_amtp 3rd, physics_informed last |
| The risk ranking is unrelated to the RMSE ranking | Spearman −0.25 to +0.07; the RMSE leader (miat) ranks 4th–6th |
| **Predicted neighbours are worse than constant velocity** | `pred` < `cv_scene` in 56 of 60 model × measure cases, by up to 0.18 ROC-AUC |
| Which measure wins depends on the label | at 2 m the probability measures lead (`p_coll_gauss` 0.88); at 3 m the time-scale measures fall to chance |
| No probability measure is calibrated | every Brier skill score is negative; `p_coll_gauss` 0.1–0.9 all correspond to 1–2 % observed collisions |

**Observations.**
- **Expectations 1 and 2 were both wrong**, and that is the main result of the study. For a
  deployable risk score, the model's target prediction with constant-velocity neighbours is
  the strongest configuration measured (ROC-AUC about 0.95 for `s_field_cal` on the matched
  set).
- **A likely reason for expectation 2 failing** is the study's own caveat 12. In `pred`, each
  neighbour is predicted as if it were the target, but inside the scene built around the
  *original* target. That is an off-distribution input.
- At the time, the ranking had no confidence intervals. With about 100 independent events,
  its significance was an open question (caveat 13). Experiment 9 resolves it.

---

## Experiment 9: Hyperparameter tuning and its effect on risk (6–8 November)

**Why.**
- Every model in Experiment 6 trained on the same 12-epoch budget, and three of the risk
  leaders hit their best validation loss at **epoch 12 of 12**, so they were still improving.
  The risk ranking could partly reflect how well each architecture suits the default schedule.
- Two open questions remained from Experiment 8:
  - Is its ranking statistically real?
  - Do better-trained trajectory models give better risk judgements?

**Papers involved.**
- Optuna (Akiba et al., KDD 2019) and the Tree-structured Parzen Estimator (Bergstra et al.,
  NeurIPS 2011) for the search.
- Each model's own paper for its architecture (Experiment 6).
- The papers from Experiment 8 for the risk measures.

**Setup** (details in [RISK_TUNING_RESULTS.md](RISK_TUNING_RESULTS.md)):
- **Models:** the union of the top 5 by risk and the top 5 by RMSE, 8 models. traj_mamba was
  stopped at 20/30 trials by request (about 2.5 h per trial), so **7 were tuned to completion**.
- **Search:** Optuna TPE, 30 trials per model, median pruning. Search space: learning rate,
  lr decay, batch size, weight decay, epochs (12–30), and dropout where the model has it.
- **Trial training:** 25 % of the training split, scored on the full validation split.
- **Final model:** one retrain on the full training data with the best setting.
- **Infrastructure:**
  - resumable at every level (study, trial checkpoint, retrain, scoring source);
  - progress through tqdm and a `--status` table;
  - three GPU lanes and two CPU scoring lanes running in parallel, about 47.5 h in total.
- **Risk:** re-scored for baseline and tuned versions on identical pair sets, all 4 sources,
  the full split.
- **Statistics:** ROC curves saved, plus **event-clustered paired bootstrap CIs**. Each
  positive's AUC placement is stored, so baseline and tuned, or any two models, can be
  compared on the same event resamples.

**Objective functions.**
- **Search objective:** minimum validation RMSE over a trial's epochs. RMSE was chosen
  deliberately, not a risk objective: it allows pruning, and it keeps the risk evaluation out
  of model selection.
- **Training objective:** each model's own loss (all eight use the basic loss).

**Expected.** Better RMSE for every model. If risk tracks prediction quality, risk accuracy
should also improve, or at least hold.

**Actual: trajectory accuracy**

| Model | Before | After | Change | Best setting (lr, decay, epochs) |
|---|---|---|---|---|
| X-TRACK | 2.090 | **1.9998** | −4.3 % | 1.8e-3, 0.84, 21 |
| STDAN | 2.114 | 2.004 | **−5.2 %** | 1.7e-3, 0.76, 18 |
| Cognitive | 2.114 | 2.012 | −4.8 % | 1.7e-3, 0.82, 21 |
| DACR-AMTP | 2.105 | 2.024 | −3.8 % | 1.9e-3, 0.79, 16 |
| MIAT | 2.072 | 2.028 | −2.1 % | 1.4e-3, 0.86, 22 |
| MHA-LSTM | 2.130 | 2.047 | −3.9 % | 1.2e-3, 0.86, 24 |
| iNATran | 2.118 | 2.051 | −3.1 % | 5.4e-4, 0.88, 30 |

**Actual: risk accuracy** (`pred` source, mean headline ROC-AUC, label < 2 m, paired 95 % CI
on the change)

| Model | `pred` change | `cv_scene` control |
|---|---|---|
| DACR-AMTP | **−0.082** [−0.111, −0.058] | −0.003 |
| X-TRACK | **−0.057** [−0.075, −0.040] | −0.003 |
| MHA-LSTM | **−0.047** [−0.063, −0.034] | −0.003 |
| STDAN | **−0.032** [−0.056, −0.009] | −0.009 |
| MIAT | **−0.020** [−0.035, −0.005] | +0.001 |
| iNATran | **−0.018** [−0.030, −0.008] | −0.004 |
| Cognitive | **−0.018** [−0.033, −0.006] | +0.000 |

**Observations.**
1. **Tuning removed a shared handicap, not an architectural one.** Every search found the
   same fix: decay the learning rate far more slowly (0.76–0.88 instead of 0.6) over 16–30
   epochs. The default schedule shrinks the learning rate about 450× in 12 epochs, freezing
   models before they converge. That is why the tuned models end up so close together
   (2.000–2.051 m): with the handicap gone, they all approach the same data-limited floor.
2. **All seven tuned models beat every untuned model in the benchmark.** The study's
   fairness caveat applies: they had about 30× the training budget.
3. **Better trajectories, worse risk.** In all seven models, risk accuracy with predicted
   neighbours fell, with every CI excluding zero. With constant-velocity neighbours it did not
   move. Ranked by `pred` risk accuracy, the best trajectory model in the study
   (x_track_tuned, 1.9998 m) comes last of 15. Spearman between risk accuracy and −RMSE is
   −0.45 (baselines) and −0.54 (tuned) at 2 m.
4. **The mechanism is mostly false alarms.** In six of seven models, the probability assigned
   to *safe* pairs rose by 6–143 % while the probability on real collisions held. Cognitive is
   the exception: it became better calibrated but placed neighbours less accurately. Either
   way, the neighbour predictions became less discriminative.
5. **The baseline risk ranking is now statistically supported.** Paired event-bootstrap CIs
   separate inatran > cognitive > dacr_amtp > {miat ≈ mha_lstm} > stdan > {traj_mamba ≈
   x_track}. This closes caveat 13 of Experiment 8.
6. **Experiment 8 reproduces exactly.** Re-scoring the baselines through the extended pipeline
   gave AUCs identical to the committed ones (maximum difference 0.0).

**Other factors.**
- One retrain per tuned model, so seed variance is not measured.
- All studies shared a sampler seed, which is why traj_mamba's best trial equals
  dacr_amtp's.
- The `pred` source carries the off-distribution caveat from Experiment 8.
- Wall-clock times reflect a shared GPU.

---

## What the experiments say together

1. **Specification gaps limit any reproduction.** STRAP leaves the split, stride, risk
   constants, mode reduction, RMSE formula and checkpoint rule unspecified (Experiment 3).
   Results are "STRAP under stated assumptions", and the assumption register is part of the
   result.
2. **Bugs that change the question are worse than bugs that crash.** Overlaid recording
   periods (Experiment 2) silently turned 9.6 % of pairs into collisions. A loss weight
   computed from the model's own outputs (Problem 2) let the model lower its loss by
   predicting "safe". Tests on data contracts, not just shapes, caught these.
3. **Our dataset matches the paper; the strong models do not** (Experiment 6). CV, S-LSTM and
   CS-LSTM reproduce the paper's numbers, while STDAN and STRAP fall 0.5–0.7 m short at 5 s.
   The remaining gap lies in the models and how they are trained, not in the data.
4. **Average RMSE barely separates good models.** The top 10 are within 2.8 % (Experiment 6),
   and tuning closes most of the remaining gaps (Experiment 9).
5. **Risk accuracy separates them, and it is not RMSE.** Separation appears only when the
   model predicts the neighbours too (Experiment 8). It is statistically supported once the
   CIs are paired and event-clustered, and its relation to RMSE is negative (Experiment 9).
6. **Predicted neighbours hurt risk, and RMSE tuning makes it worse** (Experiments 8 and 9).
   The best deployable configuration measured is the model's target prediction with
   constant-velocity neighbours.
7. **The labels are thin.** About 100 independent events, a third of them tracking artefacts,
   mostly side-by-side encounters (Experiment 7). Every risk number is a surrogate-label
   agreement, not crash prediction.
8. **The checkpoint rule is a hidden confound** (Experiments 5 and 6). Selecting by validation
   loss under σ collapse picked epoch 2 for STRAP v3 and epoch 1 for the −TE ablation.

---

## Making the research methodology stronger

Ordered by how much each would change the confidence of the conclusions. The first group
fixes known problems in the current evidence; the second makes the claims more general; the
third extends the research.

### A. Fix known problems in the current evidence

1. **Fix checkpoint selection, then rerun the affected results.** Select by validation RMSE,
   or fix the σ collapse that inflates the validation NLL (for example, a σ floor, or
   evaluating NLL with the training-mode head). Then re-evaluate STRAP v3 and rerun the −TE
   ablation, whose penalty is currently an artefact of an epoch-1 checkpoint. Check every
   model's best-by-loss vs best-by-RMSE epoch, as done for the STRAP variants here.
2. **Run several seeds.** Every number in this project is one training run. Three to five
   seeds per model would give a mean and spread. With a 2.8 % spread in the top 10, seed
   noise could reorder much of the ranking. Re-check the main claims under seeds:
   - STRAP-R vs STRAP-B;
   - the three "risk idea" variants;
   - the tuned risk drop.
3. **Paired significance tests for RMSE too.** Risk now has paired event-bootstrap CIs; RMSE
   has none. Bootstrap the per-sample squared errors, clustered by vehicle or recording, so
   that "model A beats model B" carries a CI.
4. **Remove the off-distribution confound from the `pred` source.** Predict each neighbour
   inside its *own* risk-selected scene rather than the original target's. This needs about
   1.8 M scene rebuilds, but it is the only way to know whether "predicted neighbours are
   worse than constant velocity" is a property of the predictors or of the evaluation.
   Measuring neighbour prediction error directly (neighbour RMSE) would also help.
5. **Clean the labels.**
   - Remove or flag the box-overlap tracking artefacts (36 of 101 events at 2 m).
   - Use bounding-box gaps instead of centre distance.
   - Report results with and without the near-trivial positives (21.5 % already within 2 m
     at the last observed frame).
   - Consider a reconstructed NGSIM (Montanino & Punzo) that corrects known trajectory errors.
6. **Equalise budgets when comparing tuned and untuned models.** Either tune every compared
   model with the same budget, or report budget curves (performance vs trials or epochs).
   Comparing seven tuned models with 32 untuned ones mixes architecture with effort.
7. **Use independent sampler seeds per study.** All studies shared seed 42, so they explored
   the same first ten configurations. Independent seeds give each model a fair, independent
   search.

### B. Make the claims more general

8. **Close the remaining reproduction gap before attributing it.** The data matches the paper
   (CV, CS-LSTM), so the STRAP and STDAN gap is in the models. Test the open choices:
   - a fixed-origin history so the target's own path is visible (Problem 7);
   - derived lateral velocity (Problem 8);
   - a denser stride, closer to the paper's 7.7 M trajectories;
   - the per-mode output reduction.
   Reproducing STDAN near its published 1.87 m would give the comparison a trusted anchor.
9. **Get authors' code, or drop the LOOSE models from claims.** Two of the top-3 risk models
   (iNATran, DACR-AMTP) are LOOSE. The ranking's headline result rests partly on models that
   are not the published ones.
10. **Test on more data.** HighD (the paper's second dataset) or exiD would test whether the
    findings survive another site, sensor and traffic mix. A dataset with real near-misses or
    crashes would test risk prediction against events rather than distance surrogates.
11. **Use a conflict label, not only proximity.** The 2 m label is mostly side-by-side
    proximity, and at 3 m the time-scale measures sit at chance. A label requiring closing
    speed (for example PET- or 2-D-TTC-based conflicts, as in Jiao et al.) would test the
    rear-end and crossing scenarios that TTC, DRAC and PSD are designed for. It would also
    reproduce STRAP's Table II risk breakdown, which this project never built.

### C. Extend the research

12. **Tune for what matters.** Tuning on target RMSE hurt risk. Try a multi-objective search
    (validation RMSE plus validation risk ROC-AUC or neighbour-prediction error), or add
    neighbour-prediction or risk terms to the training objective. The validation split has
    only about 20 positive events at 2 m, so this needs a larger validation set or a
    proximity-based surrogate.
13. **Joint prediction instead of target-centric models.** The `pred` source asks
    target-centric models to predict neighbours they were never trained to predict. A joint
    model that predicts all agents together is the natural fix, and the fairer test of
    "model-predicted neighbours".
14. **Calibrate before reading probabilities.** Fit isotonic or Platt recalibration on the
    validation split. Report expected calibration error and reliability diagrams. Consider
    conformal prediction for distribution-free coverage.
15. **Evaluate the uncertainty, not only the mean.** RMSE scores only μ, yet the risk
    measures that moved most depend on σ. Report NLL with a fixed σ floor, interval coverage,
    and for multimodal models minADE/minFDE. Find out why σ collapses in eval mode.
16. **Downstream, planner-aware evaluation.** The methodologies review (§B7) points to evidence
    that static displacement error does not predict closed-loop driving. A simple
    planner-in-the-loop check would test which risk score leads to safer decisions.
17. **Severity, not only probability.** Most measures here score closeness. A
    probability × severity score (P-PDRF is the only true one, and it collapses near zero)
    deserves a better mass and severity model.
18. **The planned novel objective.** The risk-appetite-constrained predictor (SOLUTIONS
    problem N), with a CVaR constraint over intention modes, is still open. Experiment 9 gives
    it a motivation: unconstrained accuracy training does not protect risk quality.

### D. Reproducibility habits worth keeping and extending

19. **Keep what worked:**
    - the assumption register;
    - fixes with rejected alternatives and reasons;
    - fidelity labels (LOOSE) on every result;
    - frozen constants recorded in every result file;
    - the one-way results flow from the GPU box;
    - a test gate before every GPU stage;
    - resumable runs with per-stage commits.
20. **Add:**
    - the git commit hash and a data checksum in every result JSON;
    - a pinned environment (lockfile);
    - a short **pre-registered analysis plan** before each study: hypotheses, headline
      measures and tests fixed in advance. The headline six measures and the 2 m label were
      chosen in Experiment 8 and reused, which is good; writing them down before running
      guards against picking results after seeing them.

---

## Appendix: papers referenced

**Core.**
- Ning, Bian, Zuo & Ergan (2025). *STRAP: Spatial-Temporal Risk-Attentive Vehicle Trajectory
  Prediction for Autonomous Driving.* arXiv:2507.08563 (local: `strap.pdf`).
- Zuo, Bian, Zuo & Ozbay (2025). *Composite Safety Potential Field for Highway Driving Risk
  Assessment.* arXiv:2504.21158 (STRAP reference [22]; source of the risk-field calibration).

**Trajectory models.** The full list with links is in
[experiments/MODEL_ALTERNATIVES.md](experiments/MODEL_ALTERNATIVES.md).
- Alahi et al. (CVPR 2016), S-LSTM.
- Deo & Trivedi (CVPRW 2018), CS-LSTM.
- Chen et al. (IEEE T-ITS 2022), STDAN.
- Li et al. (2019), GRIP++.
- Song et al. (ECCV 2020), PiP.
- Liao et al. (2024): BAT; HLTP / HLTP++; the cognitive-driven model; physics-informed under
  missing observations (IJCAI 2024); CDSTraj.
- Wang et al. (AAAI 2025), C2F-TP.
- Raskoti et al. (IROS 2025), MIAT.
- Shi et al. (NeurIPS 2022), MTR.
- Huang et al. (CVPR 2025), Trajectory Mamba.
- X-TRACK (2025).
- Gan et al. (2024), GNP.
- Westny et al. (2024), environment-aware diffusion.

**Risk measurement and evaluation.** The full list is in
[RISK_PREDICTION_METHODOLOGIES.md](RISK_PREDICTION_METHODOLOGIES.md).
- Westhofen et al. (2022). *Criticality Metrics for Automated Driving: A Review and
  Suitability Analysis of the State of the Art.* Archives of Computational Methods in
  Engineering.
- Wang, Alonso-Mora & Wang (2022). *Probabilistic Risk Metric for Highway Driving Leveraging
  Multi-Modal Trajectory Predictions.* IEEE T-ITS.
- Wang, Xu, Kuang, Lv, Li & Nie (2025). *Risk-Aware Vehicle Trajectory Prediction Under
  Safety-Critical Scenarios.* IEEE T-ITS (arXiv:2407.13480).
- Jiao, Calvert, van Cranenburgh & van Lint (2024). *A Unified Probabilistic Approach to
  Traffic Conflict Detection.* Analytic Methods in Accident Research (arXiv:2407.10959).
- Shalev-Shwartz, Shammah & Shashua (2017). *On a Formal Model of Safe and Scalable
  Self-driving Cars* (RSS). arXiv:1708.06374.
- *What Truly Matters in Trajectory Prediction for Autonomous Driving?* arXiv:2306.15136.

**Data quality.**
- Montanino & Punzo (2015). *Trajectory data reconstruction and simulation-based validation
  against macroscopic traffic patterns.* Transportation Research Part B 80.
- Coifman & Li (2017). *A critical evaluation of the Next Generation Simulation (NGSIM)
  vehicle trajectory dataset.* Transportation Research Part B 105.

**Tuning.**
- Akiba, Sano, Yanase, Ohta & Koyama (2019). *Optuna: A Next-generation Hyperparameter
  Optimization Framework.* KDD.
- Bergstra, Bardenet, Bengio & Kégl (2011). *Algorithms for Hyper-Parameter Optimization.*
  NeurIPS (the Tree-structured Parzen Estimator).
