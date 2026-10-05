# STRAP Reproduction Correction Plan

## Goal

Correct the identified data, model, loss, and metric mismatches before claiming reproduction of the STRAP paper's NGSIM results. Preserve the currently trained checkpoint and its reported evaluation as a baseline. **Do not retrain until implementation and data-contract tests pass.**

## Current Findings

- `src/data/generate_synthetic.py` builds target-relative historical inputs but leaves target future positions and neighbor goal positions in raw coordinates.
- `src/data/loader.py` chooses neighbors using S-field only, although the paper uses S-field OR O-field above the threshold. Its current O-field TTC sign treats increasing separation as closing.
- The loader drops NGSIM acceleration; its scene motion representation also approximates lateral velocity as zero.
- `src/models/risk_decoder.py` computes a distance-only O-field approximation, ignores its neighbor mask during risk aggregation, and collapses all intention queries through an inverse-risk weighted mean.
- `src/losses/loss.py` reduces risk globally, normalizes/caps it, and computes goal MSE over padded neighbors. This differs from the paper's per-sample `max(exp(Rs + Ro) - beta, 1)` scaling.
- `src/evaluation/evaluate.py` reports mean Euclidean displacement error under RMSE labels and averages batch metrics equally, including a smaller final batch.
- The paper specifies 70/10/20 but does not specify temporal-vs-random splitting in the paper text. This implementation currently uses a temporal split, 47,116 test samples, one NGSIM location (`us-101`), and 235,574 generated scenes. The paper reports 7.7M NGSIM trajectories; the sample definitions and covered locations must be reconciled before comparing scores.

## Deep Paper-to-Code Audit

The six-page `strap.pdf` was checked directly, focusing on Sections III-A through IV, equations (1)-(14), implementation details, and Tables I-III. Findings below distinguish direct paper conflicts from reproduction details the paper does not define.

### Confirmed deviations to fix

| Priority | Paper requirement or statement | Current implementation | Required correction / plan location |
|---|---|---|---|
| Critical | States use longitudinal/lateral coordinates relative to the target; S-field gamma-x/gamma-y are longitudinal/lateral (paper III-A/B). | Loader comments map `local_x` to lateral and `local_y` to longitudinal, then risk code treats array coordinate 0 as x/gamma-x and coordinate 1 as y/gamma-y. | Establish one canonical `(longitudinal, lateral)` coordinate order; map NGSIM columns explicitly and use it consistently for feature construction, risk, endpoints, intention modes, and metrics. Add asymmetric-axis tests. Plan 1 and 2. |
| Critical | Predict future target positions conditioned on observed histories (III-A, III-C). | `create_sample()` makes each historical scene position relative to the target at each historical instant, but target future labels remain raw absolute local coordinates. The target's absolute origin is not provided to the decoder. Neighbor goal labels and k-means endpoints are also raw absolute coordinates. | Define and store an origin at the final observed target position. Make all future target/neighbor endpoints and intention endpoints relative to that fixed origin; reconstruct world coordinates only for world-space metrics. Test translation invariance and inverse reconstruction. Plan 1. |
| Critical | 8-second windows: 3 seconds history and 5 seconds future; each vehicle is a possible target (IV-A). | Scene eligibility requires target rows >=80 and contiguous target windows; all selected neighbors must be present in all 80 frames. Scenes with no neighbor are discarded. Requiring future neighbor presence filters samples using future availability and changes which targets/windows exist. | Enumerate valid target windows from target history/future availability only. Select neighbors using history at observation time; do not require future neighbor observations for input membership. Allow zero-neighbor scenes and mask them correctly. Quantify resulting counts. Plan 1/2/4. |
| High | State includes 2D position, velocity, acceleration, dimensions, vehicle type, lane ID, and risk features (III-A). | `v_acc` is fetched and unit-converted but dropped in scene construction and model input. Lane is captured once per vehicle and repeated throughout the window, so lane changes are erased. | Preserve per-timestep acceleration and lane ID; implement the documented feature schema and normalization. Dimensions/type may remain static if source data confirms they are static. Update input dimensions and fixtures. Plan 1. |
| High | Neighbor selection includes vehicles whose S-field OR O-field risk exceeds 0.005; max 15; paper reports average six neighbors (III-A, IV-A). | `build_scenes()` selects only by S-field. When no candidate exceeds threshold it ranks by descending S-risk; verify this risk-based fallback against geometric nearest-neighbor selection. | Compute both observed-history S/O risks, select OR-threshold candidates, rank deterministically by documented combined risk, enforce max 15, and define closest fallback. Log risk-selection statistics including mean neighbor count. Plan 2. |
| High | S-field uses longitudinal and lateral distances with the paper's anisotropic potential equation (1), and parameters satisfy gamma-x/y > 1, alpha-x/y >=2 (III-B). | Code applies the same functional form but currently feeds swapped axes; config gamma values are both 1.0, contrary to the paper's stated strict greater-than-one constraint. | Correct axes and establish paper parameter values from the paper's cited source/author implementation. If values cannot be recovered, mark them as reproduction assumptions and run a sensitivity report; do not silently claim exact settings. Plan 2. |
| High | O-field uses predicted future minimum distance and the time associated with closest approach / when the gap stops narrowing (equation (2)); decoder uses predicted neighbor goals and target intentions (III-B/E). | Live sample generation uses a constant-relative-velocity TTC approximation with the sign reversed: with `relative = neighbor - target`, positive radial dot product means separating, but code treats it as approaching. `risk_features.py` divides radial projection by relative speed instead of distance. Decoder uses endpoint distance alone, does not model time-to-closest-approach, and ignores `neighbor_mask` in its risk sums. The trajectory-based O-field helper is not called by the live path and scales time index by `T_f * 0.1` rather than `DT`. | Correct radial closing-speed sign and formula, implement closest-approach distance/time in the actual input and decoder paths, honor masks, use `DT`, and remove or test dead alternatives so there is one authoritative implementation. Test approaching/receding/crossing/padded-neighbor cases. Plan 2. |
| High | Goal predictor flattens each neighbor's `T_h x D` encoded history and applies an MLP to predict endpoint position/velocity (III-E, equation (8)). | `GoalPredictor` mean-pools the history over time, then applies an MLP. Mean pooling discards temporal order and is a different model. | Use the paper's flattened temporal feature contract (or label any alternative as non-faithful and compare it separately). Add shape/behavior tests ensuring temporal order can affect goals. Plan 3. |
| High | Decoder retains K=100 spatial intention modes, computes per-mode risk, embeds `[risk, intention]`, cross-attends risk queries to target history, then sends final risk-attended features to trajectory generation (III-E, equations (9)-(10)). | Decoder computes endpoint risk but reduces all K queries to a single inverse-risk-weighted mean, then repeats that single vector at every future step as the LSTM input. This is a handwritten simplification; it does not preserve mode-conditioned decoder features. | Implement the paper-described per-mode risk-attention path. The PDF does not fully specify how K mode features become the final distribution; document that unresolved aggregation choice and avoid calling it exact reproduction without source-code confirmation. Ensure the LSTM receives temporally meaningful inputs. Plan 3. |
| High | Loss uses goal MSE plus trajectory MSE and Gaussian NLL, scaled by `gamma=max(exp(Rs+Ro)-beta,1)` per risk context/sample (III-F, equations (11)-(14)). | Current risk loss averages risk globally across batch/modes, divides/clamps risk, caps gamma at 10, and applies one scalar to the whole batch. Goal MSE counts padded zero neighbors. `trajectory_loss()` uses coordinate-wise mean MSE plus Gaussian NLL; exact reduction/weighting needs comparison with equations/source. | Compute valid-neighbor goal loss and per-sample risk scaling; match equations and document exact reduction. Any overflow-safe implementation must preserve intended per-sample weighting and disclose a mathematically justified bound rather than silently changing it. Add hand-calculated loss tests. Plan 3. |
| High | Evaluation reports RMSE by 1s-5s prediction horizons (IV-A and Tables I-III). Table III's average is the arithmetic mean of the five displayed horizon RMSE values. | `compute_rmse()` takes mean Euclidean distance at one timestep (not root-mean-square error), and `rmse_avg` averages errors across all 50 timesteps. `evaluate_trajectory()` then averages batch means equally, overweighting a short last batch. | Implement per-horizon RMSE with sample-weighted aggregation and set paper-style average to the arithmetic mean of the five horizon values. Optionally report ADE separately. Test on hand-computed tensors and uneven batches. Plan 4. |
| High | Data split ratio is 70/10/20 (IV-A). | The paper does not say that its split is temporal. The implementation sorts individual windows by start frame and slices sample indices. Logged train/val/test frame bounds share boundary frames (7729 and 8279), and 8-second windows can overlap across splits, so the claim that this split prevents leakage is not established. | Implement an explicitly named paper-style split only if its exact protocol can be established. Separately implement a leakage-resistant temporal split with boundaries assigned before window construction and an 8-second guard gap. Report both protocols and verify no frame/window overlap. Plan 4. |
| High | Paper reports 7.7M NGSIM trajectories and 17.7M HighD trajectories (IV-A). | This run uses one Socrata `us-101` extract: 4,802,933 rows, producing 235,574 windows under this repository's filters. Dataset rows, generated scenes, and paper's trajectories are different units. | Determine source locations, dataset version, cleaning, sampling rate, and precise definition of one paper trajectory. Do not equate rows or windows to paper trajectory count. Label one-location results honestly; add other locations only after source/schema equivalence is verified. Plan 4. |
| Medium | Input histories use vehicle states over time; temporal encoder models timestamp dependencies (III-A/D). | The target-relative history subtracts the target position independently at each timestamp; this removes target absolute displacement from the input sequence. Whether this matches the authors' coordinate convention is not fully specified by the paper. | Compare against cited source/author code if available. Keep the target-relative convention only with a clear definition; ensure future labels use a fixed observation-time origin so output coordinates are learnable. Track as an explicit reproduction assumption. Plan 1. |
| Medium | Training details list 12 epochs, Adam, batch 128, LR 0.0005, decay 0.6, D=64, three encoder layers, two decoder layers, K=100, max 15 neighbors (IV-A). | These headline settings match. However, `--lr` is parsed by `pipeline.py` but `train()` uses the global `LEARNING_RATE`, so overriding the CLI value has no effect. | Preserve paper defaults; make CLI override effective or remove the misleading flag. Record exact settings in each run manifest. This is a code defect, not a difference when the default is used. |
| Low / verify | Paper's listed feature vector and attention/GLU/LN blocks are broadly represented. | Current `MotionEncoder` adds a LayerNorm not explicitly described in the paper; its spatial and temporal encoders otherwise resemble the described blocks. | Keep as an unverified implementation detail unless the reference implementation clarifies it; do not prioritize over data, labels, decoder, loss, and metric. |

### Paper details that remain unspecified

- Whether the 70/10/20 split is random, by recording/site, or temporal; the PDF states only the ratio.
- Exact NGSIM locations/files, cleaning rules, 10-Hz resampling details, and the unit meant by its 7.7M trajectory count.
- The full numeric risk-field constants (the paper defines symbols/ranges and points to prior work, but this PDF does not provide all calibrated values).
- The target output coordinate origin and the decoder's precise reduction from K risk-attended intention queries to a trajectory distribution.
- Exact RMSE reduction equation and checkpoint/model selection protocol beyond the table labels and reported settings.

Treat these as open reproduction assumptions. Seek the cited source code or referenced risk-field paper where feasible; otherwise publish the assumption and avoid asserting exact paper reproduction.

### Required updates to the implementation phases below

- **Plan 1:** Include axis ordering, fixed output origin, acceleration, time-varying lane, and target-window eligibility that does not use future neighbor presence.
- **Plan 2:** Fix both live O-field computations and the unused helper or remove the unused path; verify radial sign, closest-approach time units, OR-based threshold selection, axis semantics, and padded-neighbor invariance.
- **Plan 3:** Replace goal mean-pooling, fix the mode-query collapse with an explicitly documented paper-uncertain aggregation, mask goal loss, and implement per-sample paper risk weighting without the current global average/cap.
- **Plan 4:** Report both metric definitions and split protocols; prove zero overlap for the leakage-resistant split; report dataset location/version/rows/scenes/samples separately and explain why its count is or is not comparable to 7.7M.
- **Plan 5:** Keep old checkpoints as immutable baseline; train a new version only after all four earlier gates pass, then report corrected full-test metrics and the remaining unresolved paper assumptions.

## Plan 1: Fix Coordinate and Feature Contracts

**Depends on:** Nothing. Complete before changing training or interpreting metrics.

1. In `src/data/generate_synthetic.py`, define one documented coordinate contract. Keep historical interaction features target-relative as intended; express target future labels and every neighbor endpoint in the same fixed frame anchored at the target's last observed position. Keep velocities in physical units in targets. Use this same frame for intention-codebook endpoints. Retain the origin so predictions can be translated back to NGSIM coordinates for world-space metrics.
   - Verify with a unit test that adding a constant translation to every scene position leaves model inputs and relative targets unchanged, and reconstructs the translated world trajectory correctly.
2. Carry NGSIM acceleration from `src/data/loader.py` into scene samples and add it to `create_sample()` input features. Represent the NGSIM scalar longitudinal acceleration as one normalized feature; update `STATE_DIM`, pipeline/model input dimensions, synthetic fixtures, normalization expectations, and relevant tests to the resulting 11 input channels (9 base + 2 risk).
   - Verify NGSIM and synthetic samples have the documented feature count and finite values.
3. In `src/intentions/kmeans.py`, fit the codebook only from corrected training-set-relative target endpoints and a documented final-velocity estimate. Do not reuse the old absolute-coordinate codebook for a corrected run.
   - Verify translated copies of the same training trajectories yield identical codebook inputs/results for a fixed seed.

**Gate:** Tests prove labels, neighbor goals, intention endpoints, and prediction-to-world reconstruction use the same origin; no training starts before this gate passes.

## Plan 2: Match Risk and Neighborhood Behavior

**Depends on:** Plan 1 coordinate contract.

1. In `src/data/generate_synthetic.py` and `src/risk/risk_features.py`, compute input O-field from observed relative position and velocity using closest-approach distance/time. Correct the closing/approaching sign. Use observations only for input risk; never use future test labels to build a feature or choose a neighbor.
   - Verify approaching, receding, and constant-separation fixtures produce expected closest-approach time/distance and risk ordering.
2. In `src/data/loader.py`, select neighbors for which either S-field or O-field exceeds `RISK_THRESHOLD`, rank interacting candidates by combined risk, retain at most 15, and preserve a deterministic closest-neighbor fallback when none exceed threshold.
   - Verify S-only and O-only candidates are selected, non-interacting candidates are excluded when interacting candidates exist, and selection never exceeds 15.
3. In `src/models/risk_decoder.py`, mask padded neighbors in both subjective and objective risk sums. Compute future O-field from mode/neighbor predicted goals and velocities using closest approach across the prediction horizon, rather than endpoint distance alone.
   - Verify adding padded neighbors does not change risk or predictions and that future risk responds to both minimum distance and time to closest approach.

**Gate:** Risk features use no future ground truth, OR-threshold neighbor selection matches the paper description, and padding has no effect.

## Plan 3: Align Decoder and Risk-Scaled Loss

**Depends on:** Plans 1 and 2.

1. In `src/models/risk_decoder.py`, replace the explicitly simplified inverse-risk weighted average of all intention queries with a mode-conditioned risk-attentive fusion consistent with paper equations (9)-(10) and Fig. 3. Keep all 100 intention queries through cross-attention. The paper does not fully specify the final mode-reduction/output-probability rule; document the chosen rule and do not present it as specified by the paper.
   - Verify changing a mode's endpoint or risk can affect its conditioned output; all outputs remain finite and shaped for the trajectory loss/evaluator.
2. In `src/losses/loss.py` and `src/training/train.py`, implement the paper's risk scale per sample: `gamma = max(exp(Rs + Ro) - beta, 1)`, then apply it to that sample's combined goal and trajectory loss. Preserve numerical safety without silently imposing the current global-risk reduction or cap. Apply neighbor masks to goal loss so padded zero goals contribute nothing.
   - Verify hand-computed risk scales, per-sample weighting, and mask invariance on small deterministic tensors.
3. Keep the current trained weights untouched; update model/loss tests for the corrected input and output contracts.
   - Verify the existing suite and all new focused tests pass before a new training run.

**Gate:** Forward pass, losses, and padding behavior are tested on corrected coordinates/features; previous checkpoint remains available as the pre-correction baseline.

## Plan 4: Reconcile Dataset, Split, and Metric Protocol

**Depends on:** Plans 1-3.

1. Document the exact NGSIM source locations, time segments, row counts, scene/window definition, and filtering used to approach the paper's reported 7.7M trajectories. Do not assume all API locations are equivalent to the paper's data. Keep `us-101`-only results labeled as such if full coverage cannot be established.
2. Implement reproducible named split protocols in the data/evaluation path. Match a paper split only to the extent supported by the paper/source data; paper text states 70/10/20 but does not state temporal splitting. Report the existing temporal split as a separate leakage-resistant result. For temporal splitting, assign time partitions before window construction or enforce a full-window guard gap so 8-second windows cannot overlap across partitions.
   - Verify stable counts and no raw frame overlap across leakage-resistant splits.
3. In `src/evaluation/evaluate.py`, implement actual RMSE with a written aggregation definition, computed over all samples rather than equally averaging batch means. Report the five horizon metrics and average with units. Keep mean Euclidean displacement error separately labeled (ADE/mean displacement), so the paper's RMSE is not conflated with it.
   - Verify the metric on hand-computable tensors, including uneven batch sizes and exact 10-Hz horizon indices.

**Gate:** Evaluation output names the dataset locations, split protocol, sample counts, metric formula, and units; a score is not called paper-comparable if dataset/split parity is unknown.

## Plan 5: Retrain and Evaluate (Only After Prior Gates)

**Depends on:** Plans 1-4. This is the first plan that trains a model.

1. Save corrected-run outputs in a new checkpoint/results directory; never overwrite the existing `best_model.pt`, paper-mismatched result, or artifact commit. Train the corrected model with the paper's stated settings where applicable: 12 epochs, Adam, batch size 128, learning rate 0.0005, per-epoch factor 0.6, hidden size 64, three encoder layers, two decoder layers, and 100 intention modes.
2. Run full test evaluation using the matching checkpoint, train-only normalization/codebook, declared split, and paper RMSE implementation. Record per-horizon metrics, average, data/split provenance, seed, and checkpoint path.
3. Compare corrected results with both the paper table and the old run, explicitly listing unresolved paper details (for example exact dataset coverage, split assignment, and decoder mode reduction). Do not claim exact reproduction when those details remain unmatched.

**Success Criteria**

- Coordinate/feature and risk contract tests pass; no data leakage from future labels.
- Full test RMSE is computed with a defined formula over the intended sample set.
- Dataset and split provenance make the comparison reproducible.
- Old artifacts remain preserved; corrected weights are produced only after all prior gates pass.

**Validation Commands**

Run from `ling/`:

```powershell
python -m pytest tests/test_loader.py tests/test_model.py tests/test_risk.py tests/test_mtp_extension.py -q
python -m pytest tests/ -q
python src/pipeline.py --mode prepare
python src/pipeline.py --mode eval
```

The last two commands are full-data operations. `--mode train` must not be run until Plans 1-4 are implemented and their gates pass.