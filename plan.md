# STRAP Implementation Plan for the MTP Project

## 0. Purpose of this document

This document is the implementation blueprint for reproducing the
**STRAP: Spatial-Temporal Risk-Attentive Vehicle Trajectory Prediction
for Autonomous Driving** framework on the NGSIM dataset and then
extending the prediction pipeline toward the MTP project's final
objective: **pairwise vehicle collision-risk prediction**.

The implementation is intentionally organized like a standard ML
pipeline:

``` text
Raw NGSIM
   ↓
Data validation / cleaning
   ↓
Coordinate and feature preparation
   ↓
Temporal window generation
   ↓
Target-vehicle scene construction
   ↓
Risk-field calculation
   ↓
Risk-aware neighborhood selection
   ↓
Feature normalization / tensor construction
   ↓
Train / validation / test split
   ↓
STRAP model
   ├── Motion encoder
   ├── Spatial encoder
   ├── Temporal encoder
   ├── Neighbor goal predictor
   ├── Intention-mode generation
   ├── Future risk-field construction
   ├── Risk-attentive feature fusion
   └── Probabilistic trajectory generator
   ↓
Loss calculation
   ↓
Backpropagation
   ↓
Validation / model selection
   ↓
Test-time trajectory prediction
   ↓
Pairwise trajectory interaction analysis
   ↓
Collision / risk prediction
```

The STRAP paper is the primary source for the model design. Where the
paper does not specify an implementation detail sufficiently for exact
reproduction, this plan explicitly marks the item as **TO VERIFY /
IMPLEMENTATION DECISION** rather than silently inventing a value.

------------------------------------------------------------------------

# 1. Research objective and scope

## 1.1 Original STRAP objective

STRAP learns a mapping from:

-   historical trajectory of a target vehicle,
-   historical trajectories/states of its surrounding vehicles,
-   subjective risk,
-   objective risk,

to a **probabilistic future trajectory of the target vehicle**.

The paper uses:

-   NGSIM and HighD,
-   3-second historical observation,
-   5-second prediction horizon,
-   each vehicle in a scene can be treated as the target,
-   risk-aware neighborhood selection,
-   maximum 15 neighboring vehicles,
-   100 spatial intention modes,
-   a spatial-temporal encoder,
-   a risk-attentive feature-fusion decoder,
-   a bivariate Gaussian output distribution,
-   risk-scaled training loss.

## 1.2 MTP objective

The MTP project ultimately needs to move beyond "predict the future path
of one vehicle" toward:

``` text
Historical traffic scene
        ↓
Predict future trajectories
        ↓
Compare future trajectories pairwise
        ↓
Estimate collision / interaction risk
        ↓
Identify whether a collision is likely
```

Therefore, STRAP should initially be implemented faithfully as a
**trajectory-prediction module**, and the collision-risk layer should be
added after the trajectory prediction pipeline is working.

## 1.3 Important scope decision

Do not change the STRAP architecture while debugging the first
implementation.

Use the following progression:

1.  Reproduce STRAP trajectory prediction.
2.  Validate trajectory RMSE.
3.  Reproduce risk-aware behavior.
4.  Add pairwise collision-risk computation.
5.  Evaluate collision-risk prediction separately.

This separates **trajectory-prediction errors** from
**collision-risk-layer errors**.

------------------------------------------------------------------------

# 2. Source basis and what is directly supported

The primary STRAP paper states that the model uses NGSIM and HighD,
8-second segments, 3 seconds of history, and 5 seconds of future
prediction. It reports a 70/10/20 train/validation/test split, a risk
threshold of 0.005, average six neighbors, maximum 15 neighbors, hidden
dimension 64, 100 intention modes, three spatial-temporal encoder
layers, two risk-feature-fusion decoder layers, Adam, batch size 128,
initial learning rate 0.0005, 12 epochs, and a per-epoch learning-rate
decay factor of 0.6.

The paper's vehicle state contains:

-   2D coordinates relative to the target vehicle,
-   velocity,
-   acceleration,
-   vehicle dimensions,
-   vehicle type,
-   lane ID.

The paper adds subjective and objective risk features to those states.

The S-field is a spatial proximity risk field. The O-field estimates
future collision-related risk using predicted minimum distance and the
time associated with the narrowing gap.

The STRAP decoder predicts future goals for surrounding vehicles,
evaluates risk for 100 possible target intentions, performs
risk-attentive cross-attention with the target representation, and
generates a bivariate Gaussian trajectory distribution.

The paper's training objective combines:

-   surrounding-vehicle goal loss,
-   target-vehicle trajectory loss,
-   risk scaling.

------------------------------------------------------------------------

# 3. Dataset

## 3.1 Dataset

Primary dataset:

``` text
NGSIM
```

The STRAP paper also evaluates HighD, but the first implementation for
this project should use NGSIM because it is the dataset already selected
for the MTP project.

## 3.2 NGSIM information to retain

The raw NGSIM data contains trajectory information such as:

-   Vehicle ID
-   Frame ID
-   Total Frames
-   Global Time
-   Local X
-   Local Y
-   Global X
-   Global Y
-   vehicle length
-   vehicle width
-   vehicle type/class
-   velocity
-   acceleration
-   lane ID
-   preceding vehicle ID
-   following vehicle ID
-   headway-related fields

Not every raw column needs to enter STRAP.

## 3.3 Dataset schema to establish

Create a dataset-schema document/table before modeling.

Example:

  Raw field            Use            Purpose
  -------------------- -------------- -----------------------------------
  Vehicle ID           Yes            Vehicle identity
  Frame ID             Yes            Temporal ordering
  Global Time          Yes/optional   Timestamp
  Local X              Yes            Position
  Local Y              Yes            Position
  Global X/Y           Optional       Visualization / coordinate checks
  Vehicle Length       Yes            Geometry/risk
  Vehicle Width        Yes            Geometry/risk
  Vehicle Type/Class   Yes            Vehicle state
  Velocity             Yes            Motion
  Acceleration         Yes            Motion
  Lane ID              Yes            Lane context
  Preceding ID         Useful         Scene/context
  Following ID         Useful         Scene/context
  Space Headway        Optional       Analysis / validation
  Time Headway         Optional       Analysis / validation

The exact feature representation entering the neural network must follow
the STRAP paper's state definition rather than automatically feeding
every NGSIM column.

------------------------------------------------------------------------

# 4. Data validation and cleaning

## 4.1 Initial checks

Before creating training samples:

-   inspect column names,
-   inspect data types,
-   count missing values,
-   count unique vehicles,
-   check frame ranges,
-   check timestamp/frame consistency,
-   check duplicate vehicle-frame records,
-   check impossible/obviously corrupted values,
-   verify lane IDs,
-   verify vehicle dimensions,
-   verify units.

Create a data-quality report.

## 4.2 Sort the trajectories

For each vehicle:

``` text
Vehicle ID
   ↓
Frame ID / timestamp
   ↓
chronological order
```

All temporal operations must be performed after sorting.

## 4.3 Duplicate records

There should be at most one record for a given:

``` text
(vehicle_id, frame_id)
```

Duplicates must be investigated rather than blindly averaged.

## 4.4 Missing observations

A sample is usable only when enough historical and future observations
exist to construct the required 8-second segment.

Do not silently fill long gaps with interpolation.

If interpolation is used for small gaps, document:

-   which columns are interpolated,
-   maximum allowed gap,
-   interpolation method,
-   whether interpolated values are used in training.

The STRAP paper does not provide a detailed interpolation recipe, so
this is an implementation decision.

## 4.5 Boundary vehicles

Vehicles entering/leaving the observation area may not have enough
history or future trajectory.

For each candidate target at time T:

``` text
Need:
3 s history
+
5 s future
```

If either part is incomplete:

``` text
do not create a training sample
```

The vehicle itself does not need to be deleted from the raw dataset.

------------------------------------------------------------------------

# 5. Coordinate system and feature engineering

## 5.1 Coordinate representation

STRAP defines the input coordinates **relative to the target vehicle**.

For target vehicle 0 and neighbor j:

``` text
Δx_j = x_j - x_target
Δy_j = y_j - y_target
```

The exact longitudinal/lateral mapping must be verified against the
selected NGSIM coordinate convention before coding.

### Required validation

Select a few frames and visualize:

-   target,
-   front vehicle,
-   rear vehicle,
-   left vehicle,
-   right vehicle.

Confirm that the signs of relative coordinates agree with the intended
longitudinal/lateral interpretation.

## 5.2 State vector

For each vehicle i at timestep t, construct a state containing:

``` text
relative 2D position
velocity
acceleration
vehicle dimensions
vehicle type
lane ID
```

Conceptually:

``` text
s_i(t) =
[
  Δx,
  Δy,
  velocity,
  acceleration,
  length,
  width,
  vehicle_type,
  lane_id
]
```

Velocity and acceleration may need to remain 2D if the source data and
implementation provide directional components; otherwise use the
representation supported by the actual NGSIM columns and STRAP
implementation.

### IMPORTANT

Do not assume a final numeric feature dimension until the NGSIM columns
and the intended encoding of:

-   velocity,
-   acceleration,
-   vehicle type,
-   lane ID

are fixed.

The STRAP paper specifies the semantic features, but it does not provide
enough detail in the paper to justify inventing a particular one-hot
encoding dimension.

------------------------------------------------------------------------

# 6. Feature encoding

## 6.1 Continuous features

Candidate continuous features:

-   relative longitudinal position,
-   relative lateral position,
-   velocity,
-   acceleration,
-   length,
-   width.

Normalize using statistics calculated **only from the training split**.

Recommended process:

``` text
training data
     ↓
calculate mean/std
     ↓
save normalization parameters
     ↓
apply same parameters to validation
     ↓
apply same parameters to test
```

Never calculate normalization statistics using the complete dataset
because that leaks test information.

## 6.2 Categorical features

Vehicle type and lane ID are categorical/discrete.

Possible representations:

-   integer + embedding,
-   one-hot,
-   normalized scalar.

The exact STRAP paper does not specify the encoding sufficiently to
prescribe one implementation.

For a faithful first implementation, choose one encoding and document it
in the experiment configuration.

## 6.3 Do not normalize target outputs incorrectly

Trajectory targets are positions.

Keep a consistent physical coordinate system for:

-   ground truth trajectory,
-   predicted mean trajectory,
-   risk calculation,
-   RMSE.

If coordinates are normalized before training, convert predicted means
back to physical meters before computing physical RMSE and collision
distances.

------------------------------------------------------------------------

# 7. Temporal window creation

## 7.1 STRAP temporal setup

The STRAP experiment uses:

``` text
8-second window
├── first 3 seconds → history
└── next 5 seconds → future ground truth
```

So:

\[ T_h = 3s \]

\[ T_f = 5s \]

## 7.2 Number of frames

NGSIM is commonly represented at 10 Hz.

If the implementation uses the original 0.1-second sampling without
resampling:

``` text
3 s history = 30 timesteps
5 s future  = 50 timesteps
```

This is a **derived implementation choice from the sampling frequency**,
not a separately stated STRAP tensor dimension in the paper. Verify this
against the exact NGSIM files being used.

If a different sampling interval is chosen, define:

``` text
Th = round(3 / Δt)
Tf = round(5 / Δt)
```

and record Δt in the experiment configuration.

## 7.3 Sliding windows

For every eligible target vehicle:

``` text
T = current reference time

history:
[T - 3 s, T]

future:
[T + Δt, T + 5 s]
```

The exact overlap/stride must be configured.

A simple initial implementation can use a fixed stride, but the stride
must be recorded because it affects:

-   sample count,
-   training time,
-   correlation between samples.

## 7.4 Sample definition

Each sample must contain:

``` text
sample_id
scene/session ID
target_vehicle_id
reference_frame/time

historical target state
historical neighbor candidates
historical risk features

future target trajectory
future neighbor trajectories
```

------------------------------------------------------------------------

# 8. Scene construction

For each target vehicle at reference time T:

``` text
All vehicles present in the same scene/frame
             ↓
Candidate target-neighbor relationships
             ↓
Risk computation
             ↓
Risk-aware neighbor selection
```

Each vehicle can become a target.

This is important because the STRAP paper treats each vehicle in a scene
as a target vehicle.

------------------------------------------------------------------------

# 9. Risk potential fields

STRAP has two risk components:

``` text
Risk
├── S-field: subjective spatial proximity risk
└── O-field: objective future collision-related risk
```

These are central to both:

-   neighborhood selection,
-   risk-aware features,
-   risk-aware decoding,
-   risk-scaled loss.

------------------------------------------------------------------------

# 10. S-field implementation

For target vehicle i and surrounding vehicle j:

\[ r\^s\_{ij} = `\exp`{=tex} `\left`{=tex}( - `\left`{=tex}\|
`\frac{\Delta x_{ij}}{\gamma_x}`{=tex}
`\right`{=tex}\|\^{`\alpha`{=tex}\_x} - `\left`{=tex}\|
`\frac{\Delta y_{ij}}{\gamma_y}`{=tex}
`\right`{=tex}\|\^{`\alpha`{=tex}\_y} `\right`{=tex}) \]

where:

-   Δx = relative longitudinal distance,
-   Δy = relative lateral distance,
-   γx = longitudinal scaling factor,
-   γy = lateral scaling factor,
-   αx = longitudinal shape factor,
-   αy = lateral shape factor.

The paper states:

\[ `\gamma`{=tex}\_x\>1,`\quad `{=tex}`\gamma`{=tex}\_y\>1 \]

and:

\[
`\alpha`{=tex}\_x`\ge2`{=tex},`\quad `{=tex}`\alpha`{=tex}\_y`\ge2`{=tex}
\]

## 10.1 What S-field means

High S-field:

``` text
vehicles are spatially close
        ↓
target perceives greater proximity risk
```

Low S-field:

``` text
vehicles are far apart
        ↓
low proximity risk
```

## 10.2 Risk-field parameter issue

The paper gives the formula and parameter roles but does not provide
enough information in the supplied paper text to justify inventing exact
values for:

``` text
γx
γy
αx
αy
```

Therefore these values must be:

1.  located in the authors' implementation/supplementary material if
    available, or
2.  treated as configurable hyperparameters and reported transparently.

Do not silently choose values and claim exact reproduction.

------------------------------------------------------------------------

# 11. O-field implementation

The O-field estimates future collision-related risk:

\[ r\^o\_{ij} = `\exp`{=tex} `\left[
-
\left(
\frac{\hat d_{m,ij}}{d^*}
\right)^{\beta_1}
\right]`{=tex}`\exp`{=tex} `\left[
-
\left(
\frac{\hat t_{m,ij}}{t^*}
\right)^{\beta_2}
\right]`{=tex}\]

where:

-   (`\hat `{=tex}d\_{m,ij}) = predicted future minimum distance between
    vehicles i and j,
-   (`\hat `{=tex}t\_{m,ij}) = time associated with the narrowing gap,
-   d\* = distance scaling factor,
-   t\* = time scaling factor,
-   β1, β2 = shape factors.

## 11.1 Important implementation issue

Unlike S-field, O-field is not purely a current-frame spatial
calculation.

It depends on future interaction.

The STRAP architecture handles this by:

``` text
encoded surrounding-vehicle history
        ↓
neighbor goal predictor
        ↓
predicted neighbor future goal
        +
candidate target intention
        ↓
future risk calculation
```

Therefore O-field must be implemented in two contexts:

### A. Initial risk-aware neighborhood selection

Use the risk information available from the current scene / prescribed
risk-field procedure.

### B. Decoder risk-field generation

Use:

-   predicted surrounding-vehicle goals,
-   candidate target intention,
-   future interaction geometry,

to calculate the risk associated with each candidate intention.

The exact numerical O-field trajectory extrapolation procedure needs to
be verified carefully against the paper/author implementation before
claiming exact reproduction.

------------------------------------------------------------------------

# 12. Risk-aware neighborhood selection

This replaces a simple "nearest N vehicles" strategy.

## 12.1 Candidate generation

For the target vehicle:

``` text
all other vehicles in the scene
        ↓
calculate S-field and O-field risk
```

A vehicle is considered interacting if:

\[ r\^s\_{0j}\>0.005 \]

OR:

\[ r\^o\_{0j}\>0.005 \]

The STRAP experiment uses:

``` text
risk threshold = 0.005
```

## 12.2 Maximum number of neighbors

Set:

\[ N_v=15 \]

The paper reports approximately six neighboring vehicles on average.

If more than 15 candidates satisfy the criterion:

``` text
rank candidates by risk
        ↓
select highest-risk 15
```

The exact tie-breaking rule should be deterministic.

## 12.3 Target + neighbors

The final scene tensor contains:

``` text
1 target vehicle
+
up to 15 neighbors
```

so:

\[ N_v+1=16 \]

vehicle slots.

------------------------------------------------------------------------

# 13. Padding and masking

Because the number of neighbors varies:

``` text
sample A → 4 neighbors
sample B → 7 neighbors
sample C → 15 neighbors
```

the neural network needs fixed-size batches.

Use:

``` text
maximum slots = 15 neighbors
```

and pad missing slots.

Recommended tensor:

``` text
[target, neighbor1, ..., neighbor15]
```

with a corresponding:

``` text
vehicle_mask
```

where:

``` text
1 = real vehicle
0 = padding
```

The mask should be propagated into attention so padding does not receive
meaningful attention.

This is an implementation safeguard; the paper states the maximum
neighborhood size but does not fully specify a coding-level
padding/masking implementation.

------------------------------------------------------------------------

# 14. Final model input

For each historical timestep:

``` text
vehicle state
+
subjective risk
+
objective risk
```

Conceptually:

\[ x_i(t)=\[s_i(t),r_i(t)\] \]

where:

\[ r_i(t)=\[r^s_i(t),r^o_i(t)\] \]

The complete input can be represented as:

\[ X`\in`{=tex}
`\mathbb{R}`{=tex}\^{B`\times `{=tex}T_h`\times`{=tex}(N_v+1)`\times `{=tex}F}
\]

where:

-   B = batch size,
-   Th = historical timesteps,
-   Nv+1 = target + maximum neighbors,
-   F = encoded feature dimension.

The exact F depends on the categorical feature encoding.

------------------------------------------------------------------------

# 15. Train / validation / test split

STRAP reports:

``` text
70% training
10% validation
20% testing
```

Use the same split ratio.

## 15.1 Avoid temporal leakage

This is particularly important for NGSIM.

Do not create overlapping windows and then randomly distribute highly
overlapping windows across train and test without checking scene/session
leakage.

Preferred hierarchy:

``` text
dataset/session/trajectory segments
          ↓
split into train/validation/test
          ↓
create windows within each split
```

If reproducing the paper requires a different exact splitting protocol,
document that separately.

## 15.2 Save split IDs

Create:

``` text
train_ids.csv
val_ids.csv
test_ids.csv
```

so the experiment is reproducible.

------------------------------------------------------------------------

# 16. Ground-truth targets

The main target is the future trajectory of the target vehicle:

\[ Y\^0\_{T+1:T+T_f} \]

with 2D position at each future timestep.

For each surrounding vehicle, the decoder also needs the ground-truth
final goal for goal-prediction supervision:

``` text
final future position
+
final future velocity
```

represented conceptually as:

\[ G^j\_{gt}`\in`{=tex}`\mathbb{R}`{=tex}^4 \]

The STRAP paper defines the predicted surrounding-vehicle goals as final
position and velocity.

------------------------------------------------------------------------

# 17. Intention-mode dataset

STRAP uses a goal-based representation.

The paper applies k-means clustering to the final positions of
ground-truth trajectories to obtain:

\[ K=100 \]

spatial intention modes.

Each intention contains:

``` text
longitudinal position
lateral position
velocity information
```

and is represented as:

\[ I`\in`{=tex}`\mathbb{R}`{=tex}\^{K`\times4`{=tex}} \]

## 17.1 Intention generation procedure

Build the intention codebook only from the **training data**.

Procedure:

``` text
training GT future endpoints
        ↓
extract final position / goal representation
        ↓
k-means
        ↓
K = 100 clusters
        ↓
save cluster centers
```

Do not fit k-means on the test set.

## 17.2 Why this is needed

The 100 modes represent possible future target destinations.

For each sample:

``` text
100 possible future intentions
          ↓
calculate risk associated with each
          ↓
risk distribution over possible futures
```

------------------------------------------------------------------------

# 18. STRAP model architecture

The complete model has:

``` text
Input
  ↓
Motion Encoder
  ↓
Spatial Encoder
  ↓
Temporal Encoder
  ↓
Encoded scene
  ├── surrounding vehicles → Goal Predictor
  │                              ↓
  │                        predicted goals
  │
  └── target vehicle → target representation

predicted neighbor goals
+
100 target intentions
        ↓
future S/O risk field
        ↓
risk-field embedding
        ↓
cross-attention with target features
        ↓
LSTM trajectory generator
        ↓
MLP
        ↓
5 output parameters per future timestep
```

------------------------------------------------------------------------

# 19. Motion encoder

The motion encoder:

1.  takes vehicle state + risk features,
2.  maps them into an embedding,
3.  applies ELU,
4.  processes the embeddings using LSTM to capture temporal motion
    patterns.

Conceptually:

``` text
Raw features
     ↓
Fully Connected
     ↓
ELU
     ↓
LSTM
     ↓
Motion embedding Hm
```

The paper denotes:

\[ H_m`\in`{=tex}
`\mathbb{R}`{=tex}\^{T_h`\times`{=tex}(N_v+1)`\times `{=tex}D} \]

with:

\[ D=64 \]

in the reported experiment.

------------------------------------------------------------------------

# 20. Spatial encoder

The spatial encoder models interaction between vehicles.

For each timestep:

``` text
target
neighbor 1
neighbor 2
...
neighbor 15
        ↓
multi-head self-attention
        ↓
GLU
        ↓
residual + LayerNorm
```

The paper defines the spatial attention as:

\[ `\tilde `{=tex}H_s = MultiHeadAttn(q=H_m,k=H_m,v=H_m) \]

followed by GLU and LayerNorm.

Purpose:

``` text
learn which surrounding vehicles
matter to the target and to each other
```

------------------------------------------------------------------------

# 21. Temporal encoder

The temporal encoder models dependency across historical timestamps.

Process:

``` text
spatial features
      ↓
sinusoidal positional encoding
      ↓
multi-head self-attention
      ↓
GLU
      ↓
residual + LayerNorm
      ↓
final spatial-temporal representation
```

The paper denotes the final encoded representation as:

\[ C`\in`{=tex}
`\mathbb{R}`{=tex}\^{(N_v+1)`\times `{=tex}T_h`\times `{=tex}D} \]

with:

\[ D=64 \]

The spatial and temporal encoder blocks are stacked.

Reported configuration:

``` text
3 spatial-temporal encoder layers
```

------------------------------------------------------------------------

# 22. Positional encoding

Use standard sinusoidal positional encoding as specified by the paper.

Its purpose is to tell attention the ordering of historical timesteps.

Without positional information:

``` text
t1, t2, t3
```

could be treated too similarly because attention itself is
permutation-aware.

With positional encoding:

``` text
t1 ≠ t2 ≠ t3
```

and temporal order is represented.

------------------------------------------------------------------------

# 23. Neighbor goal predictor

After the spatial-temporal encoder:

``` text
encoded neighbor history
        ↓
flatten Th × D
        ↓
MLP
        ↓
predicted final goal
```

For each surrounding vehicle j:

\[ `\hat `{=tex}G_j`\in`{=tex}`\mathbb{R}`{=tex}\^{4} \]

The 4 components correspond to final position and velocity.

For all neighbors:

\[ `\hat `{=tex}G\_{1:N_v} = MLP(C\^{1:N_v}) \]

This goal predictor is supervised using ground-truth future neighbor
goals.

------------------------------------------------------------------------

# 24. Candidate future intentions

The target vehicle is not forced into one predetermined future.

Instead:

``` text
K = 100 intention modes
```

are considered.

For each mode k:

``` text
candidate target endpoint/goal
        +
predicted neighbor goals
        ↓
future interaction geometry
        ↓
S-field
        +
O-field
        ↓
risk for intention k
```

This produces a discrete future risk field.

------------------------------------------------------------------------

# 25. Future risk field

For each target intention k:

\[ R_k=\[R^s_k,R^o_k\] \]

where:

\[ R\^s_k=`\sum`{=tex}*{j=1}^{N_v}r^s*{0j} \]

and:

\[ R\^o_k=`\sum`{=tex}*{j=1}^{N_v}r^o*{0j} \]

The resulting risk matrix is:

\[ R`\in`{=tex}`\mathbb{R}`{=tex}\^{K`\times2`{=tex}} \]

For K=100:

``` text
100 intentions × 2 risk values
```

------------------------------------------------------------------------

# 26. Risk-attentive feature fusion

Concatenate:

``` text
risk vector
+
intention representation
```

Conceptually:

\[ \[R,I\] \]

where:

\[ R`\in`{=tex}`\mathbb{R}`{=tex}\^{K`\times2`{=tex}} \]

and:

\[ I`\in`{=tex}`\mathbb{R}`{=tex}\^{K`\times4`{=tex}} \]

The concatenated representation is passed through an MLP to obtain:

\[ H_q^0`\in`{=tex}`\mathbb{R}`{=tex}^{K`\times `{=tex}D} \]

This becomes the risk-field query.

------------------------------------------------------------------------

# 27. Cross-attention

The risk-field query attends to the encoded target vehicle
representation.

The paper defines:

\[ H_q\^j= MultiHeadCrAttn (q=H_q\^{j-1}, k=C\^0, v=C\^0) \]

where:

-   C\^0 = target vehicle encoded history,
-   query = intention/risk representation.

Interpretation:

``` text
"What does the target's historical behavior
look like under each possible risk-conditioned future?"
```

This is the key mechanism by which the risk field affects trajectory
prediction.

------------------------------------------------------------------------

# 28. Risk-attentive decoder depth

Reported STRAP configuration:

``` text
2 risk feature fusion decoder layers
```

Use:

``` text
D = 64
```

for hidden representations.

------------------------------------------------------------------------

# 29. Trajectory generator

The final risk-conditioned representation goes through:

``` text
LSTM
  ↓
MLP
  ↓
future trajectory distribution
```

The output shape is:

\[ `\hat `{=tex}Y\^0\_{T+1:T+T_f} `\in`{=tex}
`\mathbb{R}`{=tex}\^{T_f`\times5`{=tex}} \]

for one selected/intended trajectory representation.

Each future timestep produces:

``` text
μx
μy
σx
σy
ρ
```

------------------------------------------------------------------------

# 30. Bivariate Gaussian output

At each future timestep:

\[ Y_t`\sim`{=tex} `\mathcal{N}`{=tex}\_{2D}
(`\mu`{=tex}\_x,`\mu`{=tex}\_y,`\sigma`{=tex}\_x,`\sigma`{=tex}\_y,`\rho`{=tex})
\]

The five output parameters are:

1.  μx --- predicted mean x
2.  μy --- predicted mean y
3.  σx --- x uncertainty
4.  σy --- y uncertainty
5.  ρ --- x/y correlation

Constraints:

``` text
σx > 0
σy > 0
-1 < ρ < 1
```

Recommended numerical implementation:

``` text
sigma = softplus(raw_sigma) + epsilon
rho   = tanh(raw_rho)
```

The paper specifies the probabilistic bivariate Gaussian formulation;
the exact numerical stabilization function should be documented as an
implementation detail.

------------------------------------------------------------------------

# 31. Loss functions

STRAP uses two main losses.

## 31.1 Goal loss

For surrounding vehicle goals:

\[ L\_{goal} = \|`\hat `{=tex}G_i-G\_{i,gt}\|\^2 \]

Use MSE.

## 31.2 Trajectory loss

The trajectory loss combines:

-   position MSE,
-   negative log-likelihood of the ground-truth position under the
    predicted bivariate Gaussian.

Conceptually:

\[ L\_{traj} = `\frac{1}{T_f}`{=tex} `\sum`{=tex}\_t \[
\|`\hat`{=tex}`\mu`{=tex}*t-`\mu`{=tex}*{gt,t}\|\^2 -
`\log `{=tex}p(y\_{gt,t})\] \]

where:

\[ p(y\_{gt,t}) \]

is the bivariate Gaussian likelihood.

------------------------------------------------------------------------

# 32. Risk-scaled loss

STRAP introduces:

\[ `\gamma`{=tex}\_{risk} = `\max
[
\exp(R^s+R^o)-\beta,
1
]`{=tex}\]

and:

\[ L\_{total} = `\gamma`{=tex}*{risk} (L*{goal}+L\_{traj}) \]

The purpose is:

``` text
low-risk example
    ↓
normal loss weight

high-risk example
    ↓
larger loss weight
    ↓
model receives stronger learning signal
```

The paper identifies β as a predefined bias term but the supplied paper
text does not provide enough information to safely invent its numerical
value.

Therefore:

``` text
β = configurable hyperparameter
```

until the authors' implementation or supplementary information is
verified.

------------------------------------------------------------------------

# 33. STRAP-B baseline

Implement a basic-loss version:

\[ L\_{basic}=L\_{goal}+L\_{traj} \]

This is the paper's STRAP-B configuration.

Use it as an internal baseline.

Then train:

``` text
STRAP-B
vs.
STRAP-R
```

where STRAP-R uses risk scaling.

This is important because it tests whether the risk-scaled loss actually
helps.

------------------------------------------------------------------------

# 34. Training configuration

Use the paper's reported configuration as the reproduction target:

  Parameter                      STRAP setting
  ---------------------------- ---------------
  Train split                              70%
  Validation split                         10%
  Test split                               20%
  History                                  3 s
  Future horizon                           5 s
  Risk threshold                         0.005
  Maximum neighbors                         15
  Average neighbors reported               \~6
  Hidden dimension D                        64
  Intention modes K                        100
  Encoder layers                             3
  Risk-fusion decoder layers                 2
  Optimizer                               Adam
  Batch size                               128
  Initial learning rate                 0.0005
  Epochs                                    12
  LR decay factor                0.6 per epoch

The hardware used by the paper is not a requirement for your
implementation; record your own hardware separately.

------------------------------------------------------------------------

# 35. Development training configuration

Do not immediately start with the complete NGSIM dataset and full
configuration.

First use a small debug configuration:

``` text
small subset
↓
small number of target vehicles
↓
few windows
↓
small batch
↓
1–2 epochs
```

Goal:

``` text
make sure the entire forward pass works
```

Then:

``` text
small dataset
↓
overfit test
```

A very small dataset should be overfit sufficiently to prove:

-   tensor dimensions are correct,
-   target alignment is correct,
-   loss decreases,
-   gradients are non-zero,
-   output parameters are numerically stable.

Only then move to the full dataset.

------------------------------------------------------------------------

# 36. Training loop

Pseudo-pipeline:

``` text
for epoch:

    model.train()

    for batch in train_loader:

        historical_features
        risk_features
        neighbor_mask
        ground_truth_target
        ground_truth_neighbor_goals

        ↓

        STRAP forward()

        ↓

        predicted neighbor goals
        predicted risk-conditioned trajectory
        Gaussian parameters

        ↓

        calculate Lgoal
        calculate Ltraj
        calculate γrisk

        ↓

        Ltotal

        ↓

        optimizer.zero_grad()
        Ltotal.backward()
        optimizer.step()

    ↓

    model.eval()

    run validation
    calculate validation loss
    calculate validation RMSE

    save best checkpoint
    update learning rate
```

------------------------------------------------------------------------

# 37. Validation

Track at least:

``` text
train loss
validation loss
validation RMSE @ 1 s
validation RMSE @ 2 s
validation RMSE @ 3 s
validation RMSE @ 4 s
validation RMSE @ 5 s
```

Select the best checkpoint using a clearly defined validation criterion.

Recommended:

``` text
lowest validation average RMSE
```

or another criterion chosen before experiments.

Do not choose the checkpoint using test performance.

------------------------------------------------------------------------

# 38. Testing

After training is finished:

``` text
load best validation checkpoint
        ↓
test set
        ↓
predict trajectories
        ↓
convert means to physical coordinates
        ↓
calculate metrics
```

Do not calculate test-derived normalization statistics or refit k-means.

------------------------------------------------------------------------

# 39. Trajectory evaluation

The STRAP paper uses RMSE.

Report:

``` text
1-second RMSE
2-second RMSE
3-second RMSE
4-second RMSE
5-second RMSE
average RMSE
```

Use the predicted Gaussian mean:

\[ (`\hat`{=tex}`\mu`{=tex}\_x,`\hat`{=tex}`\mu`{=tex}\_y) \]

as the predicted trajectory for deterministic RMSE.

------------------------------------------------------------------------

# 40. Expected reference values

The STRAP paper reports the following NGSIM RMSE values:

  Method       1 s    2 s    3 s    4 s    5 s   Average
  --------- ------ ------ ------ ------ ------ ---------
  CV          0.59   1.59   2.87   4.44   6.25      3.15
  S-LSTM      0.65   1.31   2.16   3.25   4.55      2.38
  CS-LSTM     0.57   1.26   2.11   3.18   4.50      2.32
  STDAN       0.43   1.01   1.69   2.56   3.65      1.87
  STRAP-B     0.38   0.94   1.60   2.47   3.51      1.78
  STRAP-R     0.37   0.94   1.61   2.47   3.52      1.78

These are **reference results from the paper, not expected guaranteed
results for our implementation**.

A reproduction should compare its numbers against these values while
documenting any differences in preprocessing, sampling, split
construction, or risk-field parameters.

------------------------------------------------------------------------

# 41. Risk-level evaluation

The paper specifically evaluates performance under different risk
levels.

For NGSIM, it reports average RMSE for scenarios classified by collision
horizon and non-collision scenarios.

The reported values include:

  Scenario             STDAN   STRAP-B   STRAP-R
  ------------------ ------- --------- ---------
  Collision in 1 s      4.11      3.61      3.53
  Collision in 2 s      3.50      3.24      3.14
  Collision in 3 s      2.81      2.67      2.63
  Collision in 5 s      2.22      2.18      2.16
  Non-collision         2.01      1.87      1.87

This evaluation is important for the MTP because the final objective is
safety-related.

------------------------------------------------------------------------

# 42. Ablation experiments

Implement the paper's ablations after the main model works.

## 42.1 STRAP(-SE)

Remove spatial encoder.

Purpose:

``` text
test importance of spatial interactions
```

## 42.2 STRAP(-TE)

Remove temporal encoder.

Purpose:

``` text
test importance of temporal modeling
```

## 42.3 STRAP(-RFF)

Remove risk feature fusion.

Purpose:

``` text
test importance of risk-aware decoding
```

## 42.4 STRAP-B

Use basic loss instead of risk-scaled loss.

Purpose:

``` text
test importance of risk-scaled training
```

------------------------------------------------------------------------

# 43. Paper-reported ablation reference

The paper reports the following NGSIM results:

  Model            1 s    2 s    3 s    4 s    5 s   Average
  ------------- ------ ------ ------ ------ ------ ---------
  STRAP(-SE)      0.47   1.12   1.86   2.79   3.94      2.04
  STRAP(-TE)      0.45   1.04   1.68   2.50   3.56      1.85
  STRAP(-RFF)     0.44   1.01   1.68   2.51   3.56      1.84
  STRAP-R         0.37   0.94   1.61   2.47   3.52      1.78

Use these as a reproduction reference.

------------------------------------------------------------------------

# 44. Visualization requirements

The implementation should produce qualitative plots.

## 44.1 Single-vehicle trajectory

Plot:

``` text
past trajectory
        ↓
predicted future trajectory
        vs
ground-truth future trajectory
```

## 44.2 Scene plot

Plot:

``` text
target vehicle
neighbors
past trajectories
predicted target future
ground-truth target future
```

## 44.3 Risk field

For a target vehicle:

``` text
spatial positions
      ↓
S-field heatmap
```

and separately:

``` text
future candidate intentions
      ↓
O-field / combined risk
```

## 44.4 Intention visualization

Show the 100 candidate endpoints and highlight:

-   ground-truth endpoint,
-   selected/lowest-risk intentions,
-   high-risk intentions.

------------------------------------------------------------------------

# 45. Reproducibility artifacts

Create and save:

``` text
config.yaml
normalization.json
intention_clusters.pkl
train_ids.csv
val_ids.csv
test_ids.csv
model_best.pt
model_last.pt
training_log.csv
validation_metrics.csv
test_metrics.csv
```

Every experiment should have a unique experiment ID.

------------------------------------------------------------------------

# 46. Recommended project directory

``` text
project/
│
├── data/
│   ├── raw/
│   │   └── ngsim/
│   ├── interim/
│   ├── processed/
│   └── splits/
│
├── configs/
│   ├── strap.yaml
│   ├── debug.yaml
│   └── ablation_*.yaml
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   ├── coordinate.py
│   │   ├── windowing.py
│   │   ├── scene_builder.py
│   │   └── normalization.py
│   │
│   ├── risk/
│   │   ├── s_field.py
│   │   ├── o_field.py
│   │   ├── risk_features.py
│   │   └── neighbor_selection.py
│   │
│   ├── intentions/
│   │   └── kmeans_intentions.py
│   │
│   ├── models/
│   │   ├── motion_encoder.py
│   │   ├── spatial_encoder.py
│   │   ├── temporal_encoder.py
│   │   ├── goal_predictor.py
│   │   ├── risk_decoder.py
│   │   ├── trajectory_generator.py
│   │   └── strap.py
│   │
│   ├── losses/
│   │   ├── gaussian_nll.py
│   │   ├── trajectory_loss.py
│   │   └── risk_scaled_loss.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── checkpoint.py
│   │
│   ├── evaluation/
│   │   ├── trajectory_metrics.py
│   │   ├── risk_metrics.py
│   │   └── collision_metrics.py
│   │
│   └── visualization/
│       ├── trajectories.py
│       └── risk_fields.py
│
├── checkpoints/
├── logs/
├── results/
├── notebooks/
└── README.md
```

------------------------------------------------------------------------

# 47. Implementation order

Do not implement everything simultaneously.

## Phase 1 --- Dataset pipeline

Deliverables:

-   raw NGSIM loader,
-   schema validation,
-   cleaning,
-   chronological sorting,
-   coordinate conversion,
-   sample window generation.

Success criterion:

``` text
Can inspect one sample and verify its complete 8-second scene.
```

------------------------------------------------------------------------

## Phase 2 --- Feature engineering

Implement:

-   relative coordinates,
-   velocity,
-   acceleration,
-   dimensions,
-   type,
-   lane,
-   normalization.

Success criterion:

``` text
X tensor and Y tensor have correct values and dimensions.
```

------------------------------------------------------------------------

## Phase 3 --- S-field

Implement:

\[ r\^s\_{ij} \]

Create visual tests.

Success criterion:

``` text
risk increases as vehicles become closer.
```

------------------------------------------------------------------------

## Phase 4 --- O-field

Implement and validate the future-risk calculation.

This is the highest-risk preprocessing component because it depends on
future interaction geometry.

Success criterion:

``` text
simple synthetic trajectories produce sensible
high-risk and low-risk cases.
```

------------------------------------------------------------------------

## Phase 5 --- Risk-aware neighborhood selection

Implement:

``` text
risk > 0.005
+
max 15 neighbors
+
padding
+
mask
```

Success criterion:

``` text
for a selected sample, manually verify the chosen neighbors.
```

------------------------------------------------------------------------

## Phase 6 --- Basic trajectory baseline

Before complete STRAP:

``` text
historical target + neighbors
        ↓
simple LSTM/attention baseline
        ↓
future target trajectory
```

This establishes whether the data pipeline itself is sensible.

------------------------------------------------------------------------

## Phase 7 --- STRAP encoder

Implement:

``` text
motion encoder
→ spatial attention
→ temporal attention
```

Test tensor shapes after every block.

------------------------------------------------------------------------

## Phase 8 --- Goal predictor

Implement:

``` text
neighbor encoded features
→ flatten
→ MLP
→ final position + velocity
```

Train/validate goal prediction separately before connecting the risk
decoder.

------------------------------------------------------------------------

## Phase 9 --- Intention module

Implement:

``` text
training GT endpoints
→ k-means
→ 100 intention modes
```

Save the fitted codebook.

------------------------------------------------------------------------

## Phase 10 --- Risk-attentive decoder

Implement:

``` text
intentions
+
predicted neighbor goals
        ↓
risk fields
        ↓
risk + intention embedding
        ↓
cross-attention with target representation
        ↓
trajectory generator
```

------------------------------------------------------------------------

## Phase 11 --- Loss

Implement and test separately:

``` text
Lgoal
Ltraj
γrisk
Ltotal
```

Then verify that:

``` text
STRAP-B = no risk scaling
STRAP-R = risk scaling
```

------------------------------------------------------------------------

## Phase 12 --- Full training

Use the paper's reported configuration:

``` text
D=64
K=100
Nv=15
3 encoder layers
2 decoder layers
Adam
batch=128
lr=0.0005
12 epochs
decay=0.6
```

------------------------------------------------------------------------

# 48. MTP collision-risk extension

After STRAP trajectory prediction is validated, add the MTP-specific
layer.

## 48.1 Why this is separate

STRAP predicts:

``` text
future trajectory of target vehicle
```

The MTP requires:

``` text
future interaction / collision risk between vehicles
```

Therefore:

``` text
STRAP
=
trajectory prediction engine
```

and:

``` text
collision-risk module
=
MTP application layer
```

------------------------------------------------------------------------

# 49. Pairwise trajectory construction

For vehicles i and j:

``` text
Predicted trajectory i
+
Predicted trajectory j
        ↓
align future timestamps
        ↓
calculate pairwise separation
```

For predicted center positions:

\[ d\_{ij}(t) = `\sqrt{
(\hat x_i(t)-\hat x_j(t))^2+
(\hat y_i(t)-\hat y_j(t))^2
}`{=tex} \]

Then calculate:

\[ d\^{min}\_{ij} = `\min`{=tex}*t d*{ij}(t) \]

over the prediction horizon.

------------------------------------------------------------------------

# 50. Geometry-aware collision criterion

Do not initially classify a collision using center-to-center distance
alone.

Vehicles have:

-   length,
-   width.

Therefore the final collision detector should consider vehicle geometry.

Possible progression:

### Version 1

Center-distance threshold.

### Version 2

Axis-aligned bounding-box overlap.

### Version 3

Oriented vehicle rectangle overlap.

Use the simplest version first for debugging, then move toward
geometry-aware collision detection.

The exact final collision threshold must be defined experimentally and
documented.

------------------------------------------------------------------------

# 51. Ground-truth collision labels

For test/evaluation only:

``` text
actual future trajectory i
+
actual future trajectory j
        ↓
ground-truth collision detector
        ↓
collision / non-collision
```

This produces:

\[ y\_{ij}`\in`{=tex}{0,1} \]

or a continuous ground-truth risk measure.

IMPORTANT:

The future ground truth must **never** be fed to STRAP as an input
during prediction.

It is used only after prediction to evaluate performance.

------------------------------------------------------------------------

# 52. Predicted collision risk

At test time:

``` text
past only
 ↓
STRAP
 ↓
predicted trajectories
 ↓
pairwise interaction analysis
 ↓
predicted collision risk
```

No future ground-truth trajectory is used.

This avoids target leakage.

------------------------------------------------------------------------

# 53. Possible MTP risk outputs

The collision layer can eventually output:

``` text
pair
predicted minimum distance
time of minimum distance
TTC-like quantity
risk score
collision probability
binary collision prediction
```

Keep the first version simple.

Recommended initial output:

``` text
pairwise predicted minimum distance
+
binary collision prediction
```

Then expand to a continuous risk score.

------------------------------------------------------------------------

# 54. Evaluation of collision prediction

For collision classification:

``` text
TP
TN
FP
FN
```

Report:

-   Precision
-   Recall
-   F1
-   Accuracy
-   ROC-AUC if probability output is available
-   PR-AUC if collisions are highly imbalanced

For safety applications, do not rely on accuracy alone.

Also report:

``` text
false negatives
```

because missing a true collision is more safety-critical than producing
a false alarm.

------------------------------------------------------------------------

# 55. Pairwise evaluation protocol

For each test scene:

``` text
1. Select all target vehicles.
2. Generate predicted future trajectories.
3. Form relevant vehicle pairs.
4. Calculate predicted pairwise interaction.
5. Calculate actual pairwise interaction from future ground truth.
6. Compare predicted vs actual.
```

Avoid evaluating every theoretically possible pair if the experimental
definition is supposed to focus on interacting vehicles.

Use a clearly documented pair-selection rule.

------------------------------------------------------------------------

# 56. Risk horizon evaluation

Because STRAP predicts 5 seconds:

calculate risk/collision performance at:

``` text
1 s
2 s
3 s
4 s
5 s
```

This gives:

``` text
early warning performance
```

which is particularly relevant to collision avoidance.

Example:

``` text
collision in 5 s → can model detect it?
collision in 3 s → can model detect it?
collision in 1 s → can model detect it?
```

------------------------------------------------------------------------

# 57. Critical data-leakage checks

Before reporting results, verify all of the following.

## Check 1

Normalization statistics:

``` text
TRAIN ONLY
```

## Check 2

K-means intention centers:

``` text
TRAIN ONLY
```

## Check 3

Future target trajectory:

``` text
NOT an input
```

## Check 4

Future neighbor trajectory:

``` text
NOT an input
```

## Check 5

Ground-truth collision label:

``` text
TEST/EVALUATION ONLY
```

## Check 6

Test performance:

``` text
not used to tune hyperparameters
```

------------------------------------------------------------------------

# 58. Unit tests that must exist

## Data tests

-   vehicle IDs unique per frame,
-   timestamps sorted,
-   no unexpected duplicate samples,
-   coordinate differences correct,
-   window lengths correct.

## Risk tests

-   S-field decreases with distance,
-   S-field symmetry is understood/documented,
-   O-field behaves sensibly on synthetic approaching/separating
    vehicles,
-   thresholding works at 0.005.

## Tensor tests

Verify:

``` text
B × Th × (Nv+1) × F
```

and all subsequent dimensions.

## Gaussian tests

Verify:

``` text
sigma > 0
-1 < rho < 1
NLL is finite
```

## Mask tests

Verify padded vehicles:

``` text
do not affect attention
do not contribute incorrectly to losses
```

------------------------------------------------------------------------

# 59. Experiment tracking

For every experiment save:

``` text
experiment_id
dataset version
split version
feature list
normalization version
risk parameters
risk threshold
Nv
D
K
encoder depth
decoder depth
learning rate
batch size
epochs
random seed
checkpoint
validation metrics
test metrics
```

This is essential because risk-field parameters and preprocessing
choices may materially affect results.

------------------------------------------------------------------------

# 60. Main experiment matrix

At minimum, run:

  Experiment      Purpose
  --------------- --------------------------------
  CV              simple physics baseline
  LSTM baseline   sequence baseline
  STRAP-B         STRAP without risk-scaled loss
  STRAP-R         complete STRAP
  STRAP(-SE)      spatial ablation
  STRAP(-TE)      temporal ablation
  STRAP(-RFF)     risk-fusion ablation

The STRAP paper itself compares against CV, Social-LSTM, CS-LSTM, STDAN,
STRAP-B and STRAP-R.

------------------------------------------------------------------------

# 61. Recommended result tables

## Table A --- Data statistics

``` text
Dataset
Scenes
Vehicles
Windows
Average neighbors
Maximum neighbors
Collision samples
Non-collision samples
```

## Table B --- Trajectory prediction

``` text
Model | 1s | 2s | 3s | 4s | 5s | Avg
```

## Table C --- Risk-level performance

``` text
Model | collision 1s | collision 2s | ...
```

## Table D --- Collision classification

``` text
Model | Precision | Recall | F1 | ROC-AUC | PR-AUC
```

## Table E --- Ablation

``` text
Model | RMSE | collision F1 | false negatives
```

------------------------------------------------------------------------

# 62. Expected research story

The eventual research story should be:

``` text
Problem:
Standard trajectory predictors treat future motion primarily as a
motion-pattern prediction problem.

Observation:
Safety-critical behavior is strongly affected by interactions and risk.

STRAP:
Adds subjective and objective risk fields and risk-aware attention.

Result:
Better trajectory prediction, particularly in high-risk scenarios.

MTP extension:
Use predicted trajectories to evaluate pairwise future vehicle
interaction and collision risk.

Final system:
Risk-aware trajectory prediction → pairwise collision-risk prediction.
```

------------------------------------------------------------------------

# 63. Important limitations to document

## 63.1 STRAP is fundamentally a trajectory predictor

It does not by itself provide the final MTP collision classifier.

## 63.2 O-field implementation needs careful reproduction

The paper gives the formula but some low-level implementation details
are not sufficiently specified for an exact code reproduction from the
paper alone.

## 63.3 Risk parameters need verification

Do not invent:

``` text
γx, γy, αx, αy
d*, t*, β1, β2
β
```

without documenting that they are implementation choices.

## 63.4 NGSIM is highway trajectory data

The paper itself identifies future extension to more complex urban
settings as future work.

## 63.5 NGSIM data leakage is a major concern

Overlapping temporal windows can be highly correlated.

The split protocol must therefore be explicitly documented.

------------------------------------------------------------------------

# 64. Definition of "done"

The STRAP implementation is considered complete only when all of the
following are true:

### Data

-   [ ] NGSIM loader works.
-   [ ] Data schema validated.
-   [ ] Missing/invalid data handled.
-   [ ] Coordinates verified.
-   [ ] 3 s → 5 s windows generated.
-   [ ] Train/validation/test split saved.
-   [ ] Normalization parameters saved.

### Risk

-   [ ] S-field implemented.
-   [ ] O-field implemented.
-   [ ] Risk parameters documented.
-   [ ] 0.005 threshold implemented.
-   [ ] Maximum 15 neighbors implemented.
-   [ ] Neighbor mask implemented.

### STRAP

-   [ ] Motion encoder.
-   [ ] Spatial encoder.
-   [ ] Temporal encoder.
-   [ ] Neighbor goal predictor.
-   [ ] 100 intention modes.
-   [ ] Future risk-field generation.
-   [ ] Risk-field MLP.
-   [ ] Cross-attention.
-   [ ] LSTM trajectory generator.
-   [ ] Gaussian output.
-   [ ] Goal loss.
-   [ ] Trajectory MSE + NLL.
-   [ ] Risk-scaled loss.

### Training

-   [ ] STRAP-B trained.
-   [ ] STRAP-R trained.
-   [ ] Best checkpoint selected from validation.
-   [ ] Training curves saved.

### Evaluation

-   [ ] RMSE at 1--5 s.
-   [ ] Risk-level evaluation.
-   [ ] Qualitative trajectory plots.
-   [ ] Ablation experiments.

### MTP extension

-   [ ] Predicted trajectories converted to pairwise interactions.
-   [ ] Ground-truth pairwise collisions generated independently.
-   [ ] Predicted collision/risk score generated without future ground
    truth.
-   [ ] Precision/recall/F1/PR-AUC reported.
-   [ ] False-negative analysis performed.

------------------------------------------------------------------------

# 65. Final end-to-end algorithm

The implementation should ultimately behave as follows.

``` text
INPUT:
NGSIM trajectory data

FOR every eligible reference time T:

    1. Select target vehicle i.

    2. Extract 3-second history.

    3. Verify 5-second future exists
       for training/evaluation samples.

    4. Collect all vehicles in the scene.

    5. Convert vehicle states to the target-relative coordinate system.

    6. Calculate S-field risk.

    7. Calculate O-field risk according to the verified
       STRAP risk-field procedure.

    8. Select vehicles with:
           S-risk > 0.005 OR O-risk > 0.005

    9. Rank/select at most 15 neighbors.

   10. Pad missing neighbor slots and create mask.

   11. Add S/O risk features to vehicle states.

   12. Normalize using training-set statistics.

   13. Feed historical tensor into STRAP.

   14. Motion encoder extracts temporal motion features.

   15. Spatial encoder models vehicle-to-vehicle interactions.

   16. Temporal encoder models historical temporal dependencies.

   17. Neighbor goal predictor estimates final position/velocity
       goals of surrounding vehicles.

   18. Generate 100 candidate target intentions.

   19. For every intention:
           calculate future S-field risk
           calculate future O-field risk

   20. Concatenate:
           intention representation + risk representation

   21. Apply risk-field embedding.

   22. Cross-attend risk queries to target representation.

   23. Generate future target trajectory distribution.

   24. Calculate:
           Lgoal
           Ltraj
           γrisk
           Ltotal

   25. Backpropagate during training.

END

TEST:

    historical scene
         ↓
       STRAP
         ↓
    predicted trajectories
         ↓
    pairwise trajectory analysis
         ↓
    predicted collision/risk

EVALUATION:

    actual future trajectories
         ↓
    actual pairwise interaction
         ↓
    ground-truth collision/risk
         ↓
    compare predicted vs actual
```

------------------------------------------------------------------------

# 66. Immediate next implementation task

The first coding task should **not** be the Transformer.

Start with:

``` text
NGSIM raw CSV
      ↓
clean + sort
      ↓
choose one target vehicle
      ↓
construct one 8-second sample
      ↓
verify 3-second history
      ↓
verify 5-second future
      ↓
construct target-relative features
      ↓
print/plot the sample
```

Then implement risk fields.

Only after the sample-generation pipeline is demonstrably correct should
the STRAP neural network be added.

This ordering is important because in this project the
preprocessing/risk representation is almost as important as the neural
architecture itself.

------------------------------------------------------------------------

# 67. References to the uploaded project sources

1.  **STRAP --- Spatial-Temporal Risk-Attentive Vehicle Trajectory
    Prediction for Autonomous Driving**
    -   Primary source for this implementation plan.
    -   Provides the risk-field formulation, STRAP architecture,
        intention modes, loss functions, experimental setup and
        ablations.
2.  **MTP Problem Statement**
    -   Defines the project direction around NGSIM, V2V collision-risk
        prediction and a learning-based sequence model.
3.  **TrajectoFormer**
    -   Useful secondary reference for NGSIM preprocessing,
        temporal-window construction, target/neighborhood representation
        and trajectory-prediction evaluation.
4.  **Double-Layer LSTM / Intention-Trajectory Prediction**
    -   Useful secondary reference for NGSIM trajectory segmentation,
        Gaussian trajectory output and PyTorch training/evaluation
        practices.

------------------------------------------------------------------------

# 68. Key implementation principle

The project should be developed in this order:

``` text
DATA CORRECTNESS
      ↓
FEATURE CORRECTNESS
      ↓
RISK-FIELD CORRECTNESS
      ↓
TENSOR CORRECTNESS
      ↓
MODEL CORRECTNESS
      ↓
TRAINING CORRECTNESS
      ↓
TRAJECTORY PERFORMANCE
      ↓
COLLISION-RISK EXTENSION
```

Do not use a strong RMSE result to compensate for incorrect data
construction, and do not build the collision-risk layer until the
underlying predicted trajectories have been independently validated.
