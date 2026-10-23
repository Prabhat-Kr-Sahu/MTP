# Alternatives to STRAP for NGSIM Trajectory Prediction

As of 2026-10-23. Live, editable version (with the interactive chart): https://claude.ai/artifact/Avv1vw8EnsggJWa4BsBmJL

Seven published models report lower 5 s NGSIM error than STRAP-R (3.52 m). The best reach 2.85 m: a cognitive-driven model that feeds TTC-style risk features, and CDSTraj, a diffusion model. STDAN, STRAP's strongest baseline, reports 3.65 m.

The shortlist to run through `experiments/`, in order:

1. **Baselines:** CV, a target-only LSTM, STRAP-B and the STRAP ablations. Low effort, and they anchor the results table.
2. **The paper's rivals:** CS-LSTM and STDAN.
3. **Graph models:** BAT and GRIP++.
4. **Tests of the risk idea:** STRAP with TTC/TET/TIT input features, and STRAP with a kinematic output layer.
5. **High-effort:** diffusion (CDSTraj or C2F-TP).

Published numbers come from different papers' tables, not one shared split, so only our own runs settle the ranking.

## What a replacement must plug into

Any candidate drops into `experiments/_template.ipynb` if its `forward(states, mask, dims)` returns the three outputs below; training, evaluation, collision scoring and the comparison need no changes.

| Item | Shape | Notes |
| --- | --- | --- |
| Input `states` | (B, 30, N, 11) | 3 s history at 10 Hz; vehicle 0 = target, up to 15 neighbours, zero-padded, normalized |
| Input `mask` | (B, N) | True = real vehicle |
| Input `dims` | (B, N, 2) | raw length/width in metres |
| Output `traj_dist` | (B, 50, 5) | 5 s future of the target: mu_x, mu_y, sigma_x, sigma_y, rho (metres, relative to last observed position) |
| Output `goals` | (B, N-1, 4) | neighbour goals; models without a goal head return zeros |
| Output 3 | any | ignored (STRAP returns its risk field) |

Two adaptation rules follow from this:

- **Multimodal models** (several trajectories + mode probabilities) must collapse to one Gaussian per step, e.g. take the most likely mode, or report minADE/minFDE separately. Our RMSE compares a single mean trajectory, which is how STRAP and its NGSIM baselines are scored.
- **Map-based models** (VectorNet, MTR, QCNet, Wayformer) expect lane polylines. NGSIM has none in our loader, so they run agent-only or with lane IDs encoded as features; their reported Argoverse/Waymo gains will not transfer one to one.

## Models the STRAP paper compares against

STRAP reports four baselines on NGSIM (Table I of `strap.pdf`). STDAN is the strongest, 1.87 m average RMSE against STRAP-R's 1.78 m. These four are the minimum set our comparison should reproduce, because they give the paper's numbers a reference point on our own data split.

| Model | Family | What it does | Effort in our repo |
| --- | --- | --- | --- |
| CV (constant velocity) | Physics | Rolls the last velocity forward | Trivial: `constant_velocity` already exists in `src/pipeline.py` |
| S-LSTM (Alahi et al., CVPR 2016) | RNN + social pooling | LSTM per vehicle, hidden states pooled on a grid | Low |
| CS-LSTM (Deo & Trivedi, CVPRW 2018) | RNN + conv social pooling | LSTM encoder, convolutional pooling over a 13x3 lane grid, manoeuvre-conditioned decoder | Low to medium |
| STDAN (Chen et al., IEEE T-ITS 2022) | Spatial-temporal dynamic attention + intention | Attention over neighbours and time, lateral/longitudinal intention heads | Medium |
| STRAP ablations (-SE, -TE, -RFF) | STRAP variants | Drop spatial encoder, temporal encoder or risk feature fusion | Low: subclass that skips the module in `STRAP` |

The paper also cites further candidates in its related work, none of which it benchmarks: GRIP++ [7] (graph), AI-TP [11] (attention + GRU), MTR [13] (transformer, source of STRAP's k-means intentions), a diffusion model by Westny et al. [14], Social GAN [15], physics-informed models [16], [17], TNT [18] and a goal-based neural-physics model [19]. Sections below cover them.

## Recurrent and social-pooling models

These are the cheapest strong baselines: an LSTM encoder per vehicle plus some way to pool neighbours. All of them sit between 3.7 m and 4.6 m RMSE at 5 s on NGSIM, so they bound the low end of our comparison. The LSTM baseline in our template is the no-pooling version of this family.

| Model | Interaction mechanism | Output | Why try it |
| --- | --- | --- | --- |
| S-LSTM | Social pooling of hidden states on a spatial grid | Single Gaussian | Paper baseline; simplest interaction model |
| CS-LSTM | Convolution + max-pool over a 13x3 lane grid | 6 manoeuvre-conditioned Gaussians | Paper baseline; the most reproduced NGSIM model ([code](https://github.com/nachiket92/conv-social-pooling)) |
| NLS-LSTM | Non-local multi-head attention on the social grid | Manoeuvre-conditioned | Attention variant of CS-LSTM |
| MHA-LSTM | Multi-head attention over neighbour encodings | Multimodal | 0.41 m at 1 s, close to STDAN for little code |
| CF-LSTM | Cascaded feature-based LSTM | Single | 3.82 m at 5 s with a plain LSTM stack |
| MFP (multiple futures prediction) | Attention + latent modes | Multimodal | Strong mid-range (3.78 m at 5 s) |
| S-GAN, MATF-GAN, TS-GAN | GAN training on social features | Sampled | Generative baselines; noisy to train |
| WSiP (wave superposition) | Wave-superposition-inspired social pooling | Manoeuvre-conditioned | Physics-flavoured pooling, AAAI 2023 |

Numbers are from the NGSIM tables in [BAT](https://arxiv.org/html/2312.06371) and [HLTP++](https://arxiv.org/html/2407.07020v1); the full table is in the comparison section below.

## Graph and attention interaction models

This family is STRAP's direct competition: the best NGSIM numbers at 5 s (2.85 to 3.65 m) all come from attention or graph encoders. Most add a manoeuvre or behaviour signal on top of the interaction encoder.

| Model | Key idea | Code | Fit for our repo |
| --- | --- | --- | --- |
| GRIP++ (Li et al., 2019) | Graph convolution over a dynamic vehicle graph, seq2seq GRU decoder; fixed + dynamic graphs; claims 21.7x faster inference than CS-LSTM | Not checked | Good: our `states` + `mask` already form the graph |
| PiP (Song et al., ECCV 2020) | Conditions the target's prediction on the ego vehicle's planned trajectory | [Third-party](https://github.com/ShuxuanSheng/PiP-Planning-informed-Prediction) | Needs an ego plan; NGSIM has none, so run as PiP-noPlan |
| STDAN (Chen et al., T-ITS 2022) | Spatial-temporal dynamic attention + lateral/longitudinal intention | Not found | STRAP's best baseline; must have |
| iNATran (Chen et al., 2022) | Interaction-aware transformer, same group as STDAN | Not found | 3.43 m at 5 s |
| DACR-AMTP (Cong et al., 2023) | Dynamic attention + context reasoning, multimodal | Not found | Best 5 s among older models (3.40 m) |
| FHIF (Zuo et al., 2023) | Feature-level hierarchical interaction fusion | Not found | Similar to STDAN accuracy |
| BAT (Liao et al., 2023/24) | Behaviour from graph centrality (degree, closeness, eigenvector) + polar pooling + priority attention, GMM decoder | [Public](https://github.com/Petrichor625/BATraj-Behavior-aware-Model) | Best 1 s (0.23 m); code available, so a realistic port |
| HLTP / HLTP++ (Liao et al., 2024) | Teacher with visual-sector pooling, GAT and iTransformer fusion, distilled into a spiking-network student | Not found | 3.02 m at 5 s; lightweight student is attractive |
| Cognitive-driven model (Liao et al., 2024) | TTC/TET/TIT safety features + graph centrality + "Leanformer" fusion | Not found | Best 5 s found (2.85 m); also risk-aware, see below |
| GaVa (Liao et al., 2024) | Same group; reported in the IJCAI 2024 table | Not found | 3.13 m at 5 s |
| MIAT (Raskoti et al., IROS 2025) | Manoeuvre-intention-aware transformer with a control knob for intention strength | [Public](https://github.com/cpraskoti/MIAT) | Recent, NGSIM-only, code released |

Sources: [BAT](https://arxiv.org/html/2312.06371), [HLTP++](https://arxiv.org/html/2407.07020v1), [cognitive-driven model](https://arxiv.org/html/2404.17520), [MIAT](https://arxiv.org/abs/2504.05059), [GNP](https://arxiv.org/pdf/2409.15182) (PiP numbers).

## Transformer, generative and state-space models

Only the diffusion models in this group report NGSIM RMSE. The large map-based transformers are benchmarked on Waymo and Argoverse, so on NGSIM they are untested architectures rather than known winners.

| Model | Type | Benchmarks | NGSIM result | Code |
| --- | --- | --- | --- | --- |
| CDSTraj (Liao et al., 2024) | Characterized diffusion + spatial-temporal interaction network | NGSIM, highD, MoCAD | 0.36 / 0.86 / 1.36 / 2.02 / 2.85 m (1 to 5 s) | Not found |
| C2F-TP (Wang et al., AAAI 2025) | Coarse multimodal interaction stage, then conditional denoising refinement | NGSIM, highD | Not in abstract; check the paper's table | Not found |
| Diffusion, environment-aware (Westny et al., 2024; STRAP ref [14]) | Diffusion over trajectories with environment conditioning | highD and others | Not checked | Not checked |
| MTR (Shi et al., NeurIPS 2022) | Transformer with learnable motion query pairs; STRAP borrows its k-means intention idea | Waymo (ranked 1st) | None | [Public](https://github.com/sshaoshuai/MTR) |
| Trajectory Mamba (CVPR 2025) | Attention + selective state-space (Mamba) encoder, built for speed and small size | Argoverse 1 and 2 | None | Not checked |
| X-TRACK / X-TRAJ (2025) | xLSTM encoder + GAT, decodes acceleration and yaw rate through a kinematic bicycle model | highD, NGSIM | 6.67 m at 5 s under its own protocol (its CS-LSTM is 7.66 m, not 4.37 m), so not comparable | [Public](https://github.com/archugh02/X-TRACK) |

No LLM-based predictor with NGSIM RMSE turned up in the search. Sources: [CDSTraj](https://arxiv.org/pdf/2405.02145), [C2F-TP](https://arxiv.org/abs/2412.13231), [MTR](https://arxiv.org/abs/2209.13508), [X-TRACK](https://arxiv.org/html/2511.00266v2), [Trajectory Mamba](https://arxiv.org/abs/2503.10898).

## Risk-aware and physics-informed models

These are the closest to STRAP's thesis, and the best comparison for its high-risk results (Table II of the paper). Two of them beat STRAP's 5 s NGSIM error while also modelling safety explicitly.

| Model | How risk or physics enters | NGSIM 5 s RMSE | Relation to STRAP |
| --- | --- | --- | --- |
| Cognitive-driven model (Liao et al., 2024) | TTC, TET, TIT and a risk-tendency index as input features, plus graph-centrality driver profiling | 2.85 m | Risk as features, not as a loss weight; a direct rival |
| Physics-informed, missing observations (Liao et al., IJCAI 2024; STRAP ref [16]) | Wavelet reconstruction of missing frames, wave fusion encoder, risk-aware module, kinematic bicycle model on the output | 2.89 m (25% missing) | Combines a risk module with physics constraints |
| Risk-aware prediction under safety-critical scenarios (Wang et al., 2024; STRAP ref [5]) | Risk-incorporated scene encoder, intention queries that mix endpoint risk with spatial intent, auxiliary risk-prediction task | Not on NGSIM; own safety-critical dataset | Same goal as STRAP-R; its auxiliary risk head is a cheap add-on to STRAP |
| GNP (Gan et al., 2024; STRAP ref [19]) | Attention predicts goals, a neural social-force model rolls the trajectory towards them | 1.59 m reported; treat as unverified, its 0.27 m at 1 s is far below every other model | Interpretable forces; goal-based like STRAP |
| X-TRACK (2025) | Kinematic bicycle-model output layer bounds acceleration and yaw rate | Own protocol only | Physics layer can be bolted on to any decoder |

A practical takeaway for our code: the kinematic bicycle-model output layer and the auxiliary risk head are both small modules that can be added to STRAP itself, so they are ablations as much as new models.

Sources: [cognitive-driven model](https://arxiv.org/html/2404.17520), [physics-informed](https://www.ijcai.org/proceedings/2024/0756.pdf), [risk-aware](https://arxiv.org/abs/2407.13480), [GNP](https://arxiv.org/pdf/2409.15182), [X-TRACK](https://arxiv.org/html/2511.00266v2).

## Reported NGSIM accuracy

Published RMSE (m) at 1 to 5 s, sorted by 5 s error (lower is better). The live doc shows this as a bar chart.

| Model | 1 s | 2 s | 3 s | 4 s | 5 s | Source table |
| --- | --- | --- | --- | --- | --- | --- |
| CDSTraj (diffusion) | 0.36 | 0.86 | 1.36 | 2.02 | 2.85 | CDSTraj Table 1 |
| Cognitive-driven model | 0.31 | 0.81 | 1.44 | 2.08 | 2.85 | arXiv 2404.17520 Table 1 |
| Physics-informed (IJCAI 24, 25% missing) | 0.42 | 0.92 | 1.45 | 2.07 | 2.89 | IJCAI 2024 Table 1 |
| HLTP++ | 0.46 | 0.98 | 1.52 | 2.17 | 3.02 | HLTP++ Table 1 |
| GaVa | 0.40 | 0.94 | 1.52 | 2.24 | 3.13 | IJCAI 2024 Table 1 |
| DACR-AMTP | 0.57 | 1.07 | 1.68 | 2.53 | 3.40 | HLTP++ Table 1 |
| iNATran | 0.39 | 0.96 | 1.61 | 2.42 | 3.43 | HLTP++ Table 1 |
| **STRAP-R** | 0.37 | 0.94 | 1.61 | 2.47 | 3.52 | STRAP Table I |
| BAT | 0.23 | 0.81 | 1.54 | 2.52 | 3.62 | BAT Table 2 |
| FHIF | 0.40 | 0.98 | 1.66 | 2.52 | 3.63 | HLTP++ Table 1 |
| STDAN | 0.43 | 1.01 | 1.69 | 2.56 | 3.65 | STRAP Table I |
| TS-GAN | 0.60 | 1.24 | 1.95 | 2.78 | 3.72 | BAT Table 2 |
| MFP | 0.54 | 1.16 | 1.89 | 2.75 | 3.78 | BAT Table 2 |
| CF-LSTM | 0.55 | 1.10 | 1.78 | 2.73 | 3.82 | BAT Table 2 |
| MHA-LSTM | 0.41 | 1.01 | 1.74 | 2.67 | 3.83 | BAT Table 2 |
| PiP | 0.55 | 1.18 | 1.94 | 2.88 | 4.04 | GNP Table 1 |
| MATF-GAN | 0.66 | 1.34 | 2.08 | 2.97 | 4.13 | BAT Table 2 |
| NLS-LSTM | 0.56 | 1.22 | 2.02 | 3.03 | 4.30 | BAT Table 2 |
| WSiP | 0.56 | 1.23 | 2.05 | 3.08 | 4.34 | BAT Table 2 |
| S-GAN | 0.57 | 1.32 | 2.22 | 3.26 | 4.40 | BAT Table 2 |
| CS-LSTM | 0.57 | 1.26 | 2.11 | 3.18 | 4.50 | STRAP Table I |
| S-LSTM | 0.65 | 1.31 | 2.16 | 3.25 | 4.55 | STRAP Table I |
| CV (constant velocity) | 0.59 | 1.59 | 2.87 | 4.44 | 6.25 | STRAP Table I |

These numbers are copied between papers, not rerun on one split, so treat gaps under about 0.1 m as noise. STDAN alone appears as 0.39, 0.42 and 0.43 m at 1 s in different tables. STRAP-B (3.51 m) is left out because it overlaps STRAP-R. GNP (1.59 m) and X-TRACK are excluded: GNP's numbers are far outside every other paper's, and X-TRACK uses a different protocol. Our own `experiments/` results are the only like-for-like comparison.

## Which notebooks to build, in order

Start with baselines that bound the problem, then the paper's strongest rival, then models that test STRAP's risk idea directly. Each row is one notebook copied from `experiments/_template.ipynb`.

| # | Notebook | Model | Effort | Adaptation needed |
| --- | --- | --- | --- | --- |
| 1 | `cv.ipynb`, `lstm.ipynb` | Constant velocity, target-only LSTM | Hours | CV: wrap `constant_velocity` as a module with tiny fixed sigma, skip training. LSTM: uncomment the template example |
| 2 | `strap_b.ipynb`, `strap_no_se.ipynb`, `strap_no_te.ipynb`, `strap_no_rff.ipynb` | STRAP-B, STRAP minus spatial / temporal encoder / risk fusion | Hours | `use_risk_loss=False`; ablations subclass `STRAP` and skip one module |
| 3 | `cs_lstm.ipynb` | CS-LSTM | 1 to 2 days | Build the 13x3 grid from relative lane and longitudinal offset in `states`; port the public PyTorch model |
| 4 | `stdan.ipynb` | STDAN | 2 to 4 days | No public code; reimplement from the paper. Our encoder blocks cover most of it |
| 5 | `grip_pp.ipynb` | GRIP++ | 2 days | Code not confirmed; adjacency from `mask` and neighbour distances |
| 6 | `bat.ipynb` | BAT | 3 days | Public code; add polar coordinates and centrality features from neighbour positions |
| 7 | `strap_risk_feats.ipynb` | STRAP + TTC/TET/TIT input features | 1 day | Tests risk-as-feature (cognitive-driven model) against risk-as-loss (STRAP-R) on the same backbone |
| 8 | `strap_kinematic.ipynb` | STRAP with a kinematic bicycle-model output layer | 1 day | Decoder predicts acceleration and yaw rate; integrate to positions (X-TRACK idea) |
| 9 | `c2f_tp.ipynb` | Diffusion refinement (C2F-TP-style) | 1 to 2 weeks | No public code; highest reported accuracy, highest cost |

Status: all nine rows are implemented as notebooks in this folder, plus `strap_r.ipynb` as the reference. Each notebook opens with a "Decisions & assumptions" table listing where it departs from its paper. Row 9 uses C2F-TP rather than CDSTraj because its coarse-plus-refine design fits our training loop; the denoising loss enters through the `aux_loss` hook in `src/train.py`.

Map-based transformers (MTR, Trajectory Mamba) are left out: without lane polylines in our loader they lose the input their reported gains depend on.

Two checks before trusting any ranking:

- [ ] Run every notebook with `DEBUG = False` and the same `SEED`; debug results are not comparable.
- [ ] Reproduce STRAP-R near the paper's 1.78 m average first, so the comparison has a trusted anchor.

## Sources

Pages opened for this review:

- [STRAP](https://arxiv.org/html/2507.08563v2): Ning et al., 2025 (local copy `strap.pdf`), Tables I to III and reference list
- [BAT: Behavior-Aware Human-Like Trajectory Prediction](https://arxiv.org/html/2312.06371): Liao et al., arXiv 2312.06371, NGSIM Table 2
- [HLTP++ / Less is More: Efficient Brain-Inspired Learning](https://arxiv.org/html/2407.07020v1): Liao et al., arXiv 2407.07020, NGSIM Table 1
- [A Cognitive-Driven Trajectory Prediction Model for Mixed Autonomy](https://arxiv.org/html/2404.17520): Liao et al., arXiv 2404.17520, NGSIM Table 1
- [Physics-Informed Trajectory Prediction under Missing Observation](https://www.ijcai.org/proceedings/2024/0756.pdf): Liao et al., IJCAI 2024, Table 1
- [Characterized Diffusion and Spatial-Temporal Interaction Network (CDSTraj)](https://arxiv.org/pdf/2405.02145): Liao et al., arXiv 2405.02145, NGSIM Table 1
- [Goal-based Neural Physics Vehicle Trajectory Prediction (GNP)](https://arxiv.org/pdf/2409.15182): Gan et al., arXiv 2409.15182, Table 1
- [Risk-Aware Vehicle Trajectory Prediction Under Safety-Critical Scenarios](https://arxiv.org/abs/2407.13480): Wang et al., arXiv 2407.13480
- [C2F-TP: Coarse-to-Fine Denoising Framework](https://arxiv.org/abs/2412.13231): Wang et al., AAAI 2025
- [MIAT: Maneuver-Intention-Aware Transformer](https://arxiv.org/abs/2504.05059): Raskoti et al., IROS 2025
- [Motion Transformer (MTR)](https://arxiv.org/abs/2209.13508): Shi et al., NeurIPS 2022
- [Trajectory Mamba](https://arxiv.org/abs/2503.10898): Huang et al., CVPR 2025
- [X-TRACK: Physics-Aware xLSTM](https://arxiv.org/html/2511.00266v2): arXiv 2511.00266
- [PiP: Planning-informed Trajectory Prediction](https://arxiv.org/abs/2003.11476) and a third-party [code](https://github.com/ShuxuanSheng/PiP-Planning-informed-Prediction)
- [GRIP++](https://arxiv.org/abs/1907.07792): Li et al., arXiv 1907.07792

Cited from STRAP's reference list but not opened: S-LSTM (Alahi et al., CVPR 2016), CS-LSTM (Deo & Trivedi, CVPRW 2018), STDAN (Chen et al., IEEE T-ITS 2022), AI-TP, Social GAN, TNT, diffusion by Westny et al. 2024.
