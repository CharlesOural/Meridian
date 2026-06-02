# Grounding Dossier 09 — Back-end Optimization: iSAM2 / GTSAM Factor Graphs

> **Scope.** This is the grounding dossier for Meridian's back-end: the incremental
> nonlinear factor-graph optimizer. It covers the iSAM2 algorithm (Bayes tree,
> incremental reordering, fluid relinearization, partial state recovery), the
> GTSAM API that implements it, the exact variable/factor set for a
> LiDAR-inertial-visual-GNSS system with online extrinsics, the **L2→L3 hand-off
> contract** that prevents double-counting, marginalization for bounded
> multi-hour graphs, robust loop/GNSS factors (switchable constraints + GNC +
> Huber), and how to query marginal covariance for the place-recognition
> pre-filter. It ends with concrete GTSAM recommendations for Meridian.
>
> **Meridian layer vocabulary used below.** L1 = sensor drivers / sync. L2 =
> front-end odometry (continuous-time B-spline **or** FAST-LIO2-style iEKF behind
> one interface). L3 = this back-end (keyframe pose graph + iSAM2). L4 = mapping
> (voxel-hash → TSDF+RGB → Marching Cubes). The L2→L3 hand-off contract is the
> single most load-bearing part of this document.

---

## Sourcing note (read this first)

Primary sources consulted for this dossier (all confirmed live during authoring;
GTSAM is BSD-licensed, **current stable 4.2, developing toward 4.3**). A few
items not pinnable to an exact line in this session remain flagged **[VERIFY]**
(notably the GNC `muStep` direction per loss type and the exact `NavState`
tangent ordering); confirm those, and the `ISAM2Params` defaults against your
*pinned* GTSAM minor version, before writing the formal spec.

- iSAM2 paper: Kaess, Johannsson, Roberts, Ila, Leonard, Dellaert,
  *"iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree,"*
  **IJRR 31(2):216–235, 2012** (preprint: `cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf`;
  the Bayes tree was introduced in Kaess et al., WAFR 2010).
- GTSAM Doxygen: `https://gtsam.org/doxygen/` (classes `ISAM2`, `ISAM2Params`,
  `ISAM2Result`, `NonlinearFactorGraph`, `Values`, `Marginals`,
  `CombinedImuFactor`, `PreintegratedCombinedMeasurements`, etc.) and the GTSAM
  source tree under `gtsam/nonlinear/`, `gtsam/navigation/`, `gtsam/slam/`,
  `gtsam_unstable/nonlinear/`.
- IMU preintegration: Forster, Carlone, Dellaert, Scaramuzza,
  *"On-Manifold Preintegration for Real-Time Visual-Inertial Odometry,"*
  **IEEE T-RO 33(1):1–21, 2017** (this is what GTSAM's
  `Preintegrated(Combined)Measurements` implements).
- GNC: Yang, Antonante, Tzoumas, Carlone,
  *"Graduated Non-Convexity for Robust Spatial Perception…,"*
  **IEEE RA-L / T-RO 2020** (arXiv:1909.08605). GTSAM implements this as
  `gtsam::GncOptimizer<T>`.
- Switchable constraints: Sünderhauf & Protzel,
  *"Switchable Constraints for Robust Pose Graph SLAM,"* **IROS 2012**
  (and Sünderhauf's PhD thesis, 2012).

---

## 1. What iSAM2 is and why Meridian uses it

### 1.1 The problem

SLAM back-ends solve a **MAP (maximum a posteriori)** estimate over robot states
(and, in Meridian, extrinsics, IMU biases, velocities, and a gravity-aligned world
frame). The full SLAM posterior factors as a product of measurement likelihoods
and priors. Taking the negative log and assuming Gaussian noise turns MAP into a
**nonlinear least-squares** problem:

```
X* = argmin_X  Σ_i  || h_i(X_i) ⊖ z_i ||²_{Σ_i}
```

where `h_i` is the measurement function of factor `i`, `z_i` the measurement,
`Σ_i` the measurement covariance, `⊖` the manifold local-coordinates (tangent)
difference, and `||e||²_Σ = eᵀ Σ⁻¹ e` the squared Mahalanobis norm. This is the
objective the iSAM2 paper, Eq. (1)–(2) region, defines (it phrases it as the
nonlinear factor graph whose product is the posterior).

Solving from scratch every keyframe is `O(n)`–`O(n^{1.5})` and does not scale to
multi-hour runs. **iSAM2** solves it *incrementally*: when you add a few new
factors/variables, it updates only the part of the solution that actually
changed.

### 1.2 The four ideas of iSAM2 (and their GTSAM knobs)

| Idea (iSAM2 §) | What it does | GTSAM control |
|---|---|---|
| **Bayes tree** (§3) | A directed clique tree = the symbolic Cholesky factor `R` of the information matrix, organized so that an incremental update touches only a sub-tree near the root. | internal; you don't touch it directly |
| **Incremental variable reordering** (§4) | Re-eliminates only affected cliques and chooses a *good* local elimination order (CCOLAMD) so fill-in stays low — instead of a global reorder every step. | internal; influenced by factor topology |
| **Fluid relinearization** (§5) | Relinearizes only variables whose linearization point has drifted past a threshold, lazily, rather than relinearizing everything. | `relinearizeThreshold`, `relinearizeSkip`, `enableRelinearization` |
| **Partial state recovery** (§6) | Back-substitution recovers only the variables you ask for (or only those that changed), not the whole state vector. | `calculateEstimate(key)` vs `calculateEstimate()` |

---

## 2. The Bayes tree (iSAM2 §3)

### 2.1 From factor graph to Bayes tree

The transformation chain (iSAM2 §2–§3):

```
Nonlinear factor graph
   │  linearize at current estimate X̄ (Jacobians)
   ▼
Gaussian factor graph  (A x = b, info form Λ = AᵀA)
   │  variable elimination in some order  (= sparse QR / Cholesky)
   ▼
Bayes net (chordal)   (a sequence of conditionals  p(xj | parents))
   │  group into cliques (the maximal cliques of the chordal graph)
   ▼
Bayes tree            (a directed tree of cliques)
```

- **Elimination** of variable `xj` from the Gaussian factor graph produces one
  **conditional** `p(xj | Sj)` (where `Sj` is its *separator* — the other
  variables it now depends on) and a new **joint factor** over `Sj` (the
  marginalization residual). This is exactly one step of sparse Cholesky/QR:
  the conditional row is a row of the upper-triangular factor `R`. (iSAM2 §2.2,
  Eq. for the elimination of a single variable.)
- The **Bayes net** is the collection of all conditionals; because elimination
  on a chordal graph yields a chordal result, the Bayes net is chordal.
- A **clique** groups a "frontal" set of variables `F` with their shared
  separator `S`. The clique stores the conditional `p(F | S)`.
- The **Bayes tree** arranges cliques so each clique's separator `S` is a subset
  of its parent clique's frontal+separator variables. The **root clique** has an
  empty separator.

> **Key structural fact (iSAM2 Theorem in §3).** The Bayes tree of a factor
> graph corresponds one-to-one with the **square-root information matrix `R`**
> (the Cholesky factor `R` with `RᵀR = Λ = AᵀA`). Each clique is a dense block
> on the diagonal of `R`; the tree edges encode the off-diagonal coupling.

### 2.2 Why a *tree* makes incremental updates cheap

The crucial property (iSAM2 §3.3, §4.1): **adding new factors only changes the
cliques between the affected variables and the root.** Concretely:

1. The new factors touch a set of variables `V`.
2. Find the cliques containing `V`. Detach the sub-tree from those cliques up to
   the root (the path to the root is the only part whose factorization can
   change, because everything below an untouched clique is conditionally
   independent of the new information given that clique's separator).
3. Re-form a small factor graph from: (a) the detached cliques converted back to
   factors, plus (b) the new factors.
4. Re-eliminate just that small graph and reattach.

Because the affected set is usually `O(1)`–`O(small)` per keyframe in
exploration (you only touch recent poses), the per-step cost is roughly constant
in exploration and grows only when a **loop closure** couples a far-away clique
to the recent ones (then the affected path to the root is long — see failure
modes §11).

### 2.3 GTSAM data structures implementing this

- `gtsam::ISAM2` — the optimizer object; owns the Bayes tree.
- `gtsam::ISAM2Clique` — a clique node (frontal conditional + cached factors).
- `gtsam::GaussianConditional`, `gtsam::HessianFactor`/`JacobianFactor` — the
  linearized building blocks stored in cliques.
- `gtsam::ISAM2Result` — returned by `ISAM2::update(...)`; contains
  `variablesRelinearized`, `variablesReeliminated`, `cliques`, `factorsRecalculated`,
  and (if `evaluateNonlinearError=true`) `errorBefore` / `errorAfter`. Use these
  for health monitoring (§12).

---

## 3. Incremental variable reordering (iSAM2 §4)

- Elimination order determines fill-in (the number of nonzeros in `R`). A bad
  order makes `R` dense and the Bayes tree shallow-and-fat → slow.
- iSAM2 reorders **only the variables involved in the re-eliminated sub-tree**
  each update, using **CCOLAMD** (Constrained COLAMD). The "constrained" part
  forces the most recently added / most-likely-to-be-relinearized variables to
  be eliminated **last** (placed nearest the root), so future updates re-touch a
  small sub-tree. (iSAM2 §4.2.)
- This is why **adding the newest keyframe pose as the variable that is
  eliminated last** keeps incremental cost low: recent poses live near the root.

> **Practical implication for Meridian:** do *not* manually constrain ordering
> unless profiling shows pathology. GTSAM's `ISAM2` applies CCOLAMD internally.
> If you ever do need it, `ISAM2UpdateParams::constrainedKeys` /
> `ISAM2Params` ordering hooks exist, but the default is correct for a
> time-ordered keyframe graph.

---

## 4. Fluid relinearization (iSAM2 §5) — the parameters that matter most

A linearized factor graph is only valid near the linearization point `X̄`. As the
estimate moves, factors that were linearized far away become wrong. Batch
methods relinearize *everything* every iteration. iSAM2 relinearizes **fluidly**:
only variables whose estimate has moved "enough".

### 4.1 The mechanism

After each update, iSAM2 computes the **delta** `Δ` (the back-substituted
correction in tangent space for each variable). A variable `xj` is marked for
relinearization when **any component of its delta exceeds a per-type threshold**:

```
relinearize xj   ⇔   max_k | Δj[k] |  >  relinearizeThreshold(type of xj)
```

(iSAM2 §5.1.) Marked variables — *and the cliques on their path to the root* —
are relinearized and re-eliminated on the next update. Untouched variables keep
their cached linear factors (this is the `cacheLinearizedFactors` optimization).

### 4.2 The GTSAM knobs (`gtsam::ISAM2Params`)

All defaults below were read from the `borglab/gtsam` `ISAM2Params.h` constructor
(member initializer list).

| Param | Type | Default (confirmed) | Meaning / how to set for Meridian |
|---|---|---|---|
| `relinearizeThreshold` | `double` **or** `FastMap<char,Vector>` (per-symbol-type thresholds) | `0.1` (scalar) | Relinearize any variable whose **delta vector magnitude ≥ threshold**. **Use the per-type map**, because a Pose3's 6 DoF (3 rad rotation + 3 m translation) need different thresholds than a velocity (m/s) or a bias. See §4.3. |
| `relinearizeSkip` | `int` | `10` | Relinearization is only *attempted* every Nth `update()` call. With `=1` you check every step (more accurate, slower); LIO-SAM and many LIO back-ends set this small. |
| `enableRelinearization` | `bool` | `true` | If false, iSAM2 never relinearizes → pure incremental linear updates (don't disable for Meridian). |
| `enablePartialRelinearizationCheck` | `bool` | `false` | When true, only checks deltas for variables likely to have changed (faster; slightly less thorough). Reasonable to enable for large graphs after profiling. |
| `cacheLinearizedFactors` | `bool` | `true` | Keep cached linear factors for unrelinearized variables. Keep true. |
| `evaluateNonlinearError` | `bool` | `false` | If true, `ISAM2Result` reports `errorBefore`/`errorAfter`. Enable in Meridian's diagnostic build / health monitor; small cost. |
| `factorization` | enum `CHOLESKY` / `QR` | `CHOLESKY` | `QR` is more numerically robust to near-singular (degenerate) factors but ~2× slower. **Recommend `CHOLESKY` normally; consider `QR` only if you see indefinite-matrix exceptions from degeneracy** (LiDAR corridor cases). |
| `optimizationParams` | `ISAM2GaussNewtonParams` or `ISAM2DoglegParams` | GaussNewton | Dogleg is more robust to bad linearization points / overshoot (trust region) at modest extra cost. **Recommend `ISAM2DoglegParams` for Meridian** given visual + GNSS factors can be far from linearization at init. |
| `findUnusedFactorSlots` | `bool` | `false` | When you `removeFactors`, reuse slots to keep the factor index compact. Enable if Meridian churns factors (switchable constraints that get pruned). |
| `enableDetailedResults` | `bool` | `false` | Populate `ISAM2Result.detail` with per-variable info. Diagnostic only; small cost. |

> Note on iSAM2 vs the original iSAM: in iSAM2 the relinearization *threshold*
> replaces the old fixed batch-relinearization period — iSAM2 decides *when* and
> *for which variables* to relinearize automatically (fluid relinearization),
> rather than relinearizing everything every `k` steps.

### 4.3 Recommended per-type relinearize thresholds (Meridian)

Use the per-symbol-type form. Example (tune during bring-up):

```cpp
gtsam::ISAM2Params p;
gtsam::FastMap<char, gtsam::Vector> th;
th['x'] = (gtsam::Vector(6) << 0.05, 0.05, 0.05,   // rot rad  (~3 deg)
                               0.10, 0.10, 0.10).finished(); // trans m
th['v'] = gtsam::Vector3::Constant(0.10);          // velocity m/s
th['b'] = (gtsam::Vector(6) << 1e-3,1e-3,1e-3,     // accel bias
                               1e-4,1e-4,1e-4).finished(); // gyro bias
th['e'] = (gtsam::Vector(6) << 1e-3,1e-3,1e-3,     // extrinsic rot
                               1e-3,1e-3,1e-3).finished(); // extrinsic trans
p.relinearizeThreshold = th;
p.relinearizeSkip = 1;            // LIO back-ends want every-step checks
p.optimizationParams = gtsam::ISAM2DoglegParams();
p.evaluateNonlinearError = true;  // diagnostics
```

(Symbol convention: `x`=pose, `v`=velocity, `b`=imu bias, `e`=extrinsic; choose
your own with `gtsam::Symbol`.)

---

## 5. Partial state recovery (iSAM2 §6)

- After the Bayes tree update, the corrected estimate is obtained by
  **back-substitution from the root down**. iSAM2 only descends into the parts of
  the tree whose deltas exceed a (small) threshold or which the user requested.
- GTSAM API:
  - `Values ISAM2::calculateEstimate()` — full back-substitution, all variables.
  - `T ISAM2::calculateEstimate<T>(Key)` — recovers **one** variable (cheap).
  - `Values ISAM2::calculateBestEstimate()` — full back-substitution forcing all
    deltas (more accurate, more expensive).
- **Meridian guidance:** in the hot path call `calculateEstimate<Pose3>(latestKey)`
  to publish the current pose; only call full `calculateEstimate()` when L4
  needs to re-integrate the map (e.g., after a loop closure adjusts many poses).
  This is exactly the "recover only what you need" property iSAM2 §6 provides.

---

## 6. The GTSAM nonlinear factor-graph API (the pieces Meridian assembles)

### 6.1 Containers

- `gtsam::NonlinearFactorGraph` — append-only container of factors; you build a
  *delta* graph each keyframe (only the new factors) and hand it to
  `ISAM2::update(newFactors, newValues)`. **Do not** rebuild the whole graph.
- `gtsam::Values` — typed heterogeneous map `Key → manifold value`
  (`Pose3`, `Vector3`, `imuBias::ConstantBias`, …). You provide a `Values` of
  **only the new variables' initial guesses** each update.
- `gtsam::Key` / `gtsam::Symbol` — `Symbol('x', i)` etc. Use
  `gtsam::symbol_shorthand::X(i)`, `V(i)`, `B(i)` from
  `<gtsam/inference/Symbol.h>`.

### 6.2 The state types

- `gtsam::Pose3` — SE(3); tangent order in GTSAM is **(rotation, translation)**,
  i.e. `[rx ry rz | tx ty tz]`. This matters for noise-model ordering and for
  reading marginal covariances.
- `gtsam::NavState` — (Pose3 R,t; Velocity v) packaged together; used by the IMU
  factors. Tangent order `[R(3) | t(3) | v(3)]` **[VERIFY ordering]**.
- `gtsam::Vector3` — velocity.
- `gtsam::imuBias::ConstantBias` — `[accel bias(3), gyro bias(3)]`.

### 6.3 The basic factors

| Factor | Header | Connects | Use in Meridian |
|---|---|---|---|
| `PriorFactor<T>` | `slam/PriorFactor.h` | one var | anchor first pose; prior on first velocity/bias; **GNSS-derived prior on world origin** |
| `BetweenFactor<Pose3>` | `slam/BetweenFactor.h` | two poses | **the L2→L3 relative-odometry factor** (see §8); also loop-closure factor |
| `GPSFactor` / `GPSFactor2` | `navigation/GPSFactor.h` | one pose (or NavState) | unary position factor from GNSS (ENU/UTM); `GPSFactor2` works on `NavState` |
| `CombinedImuFactor` | `navigation/CombinedImuFactor.h` | (pose_i,v_i,bias_i,pose_j,v_j,bias_j) | window-restart fallback only (see §8.4) |
| `ImuFactor` | `navigation/ImuFactor.h` | (pose_i,v_i,pose_j,v_j,bias_i) | older 5-way variant; prefer `CombinedImuFactor` |
| `GenericProjectionFactor` / `SmartProjectionPoseFactor` | `slam/ProjectionFactor.h`, `slam/SmartProjectionPoseFactor.h` | poses (+landmark) | optional visual factors if Meridian's front-end exposes features rather than only fused odometry |

### 6.4 Noise models (`gtsam::noiseModel::`)

- `Gaussian::Covariance(Σ)` / `Gaussian::Information(Λ)` / `Gaussian::SqrtInformation(R)`
  — full dense model. **Use `Covariance(Σ)` for the L2→L3 factor** to feed the
  front-end's marginal covariance directly (§8).
- `Diagonal::Sigmas(σ)` / `Diagonal::Variances` / `Diagonal::Precisions` —
  per-axis. Use for priors and GNSS (GNSS gives per-axis position σ).
- `Isotropic::Sigma(dim, σ)` — single σ.
- `Constrained::All(dim)` / `MixedSigmas` — hard constraints (σ→0). Avoid except
  for true equality constraints.
- `Robust::Create(mEstimator, baseNoise)` — wraps a base model with an
  M-estimator:
  - `mEstimator::Huber::Create(k)` — quadratic inside `k`, linear outside.
    `k≈1.345` for 95% Gaussian efficiency (Huber's classic tuning).
  - `mEstimator::GemanMcClure::Create(c)` — redescending; strongly rejects
    gross outliers (weight → 0). This is the kernel GNC uses (§9).
  - `mEstimator::Cauchy`, `Tukey`, `DCS`, `Welsch` also available.

> The M-estimator reweights the factor's residual via Iteratively Reweighted
> Least Squares (IRLS) inside the linearization. **Use Robust+Huber on GNSS and
> visual factors; use Robust+GemanMcClure (or GNC) on loop closures** (§9–§10).

### 6.5 Marginals

- `gtsam::Marginals(graph, values [, factorization])` then
  `marginalCovariance(Key)` → `Matrix` (the per-variable covariance), or
  `jointMarginalCovariance({keys})`.
- **For an iSAM2 graph, prefer `ISAM2::marginalCovariance(Key)`** — it reuses the
  Bayes tree and is far cheaper than building a `Marginals` object from scratch.
  This is what the place-recognition pre-filter queries (§13).

---

## 7. IMU preintegration in GTSAM (Forster et al. T-RO 2017)

### 7.1 Why preintegrate

IMU runs at 100–400 Hz; you cannot add a state per IMU sample. **Preintegration**
summarizes all IMU samples between two keyframes `i`,`j` into a single relative
motion constraint `(ΔR_ij, Δv_ij, Δp_ij)` *in the body frame at `i`*, with a
propagated covariance and a first-order bias-update Jacobian so the summary can be
**corrected for a new bias estimate without re-integrating** (Forster §III,
Eqs. for ΔR̃, Δṽ, Δp̃ and the bias-correction Jacobians, Eqs. (44)).

### 7.2 GTSAM objects

- `PreintegrationCombinedParams` (or `PreintegrationParams`) — set:
  - `n_gravity` (gravity vector in world; e.g. `(0,0,-9.81)` for ENU **[sign/axis VERIFY for your world frame]**),
  - `accelerometerCovariance`, `gyroscopeCovariance` (continuous-time noise
    density² · I),
  - `integrationCovariance` (discretization noise),
  - `biasAccCovariance`, `biasOmegaCovariance` (bias random-walk; **Combined**
    variant only),
  - `biasAccOmegaInt` (initial bias covariance coupling; **Combined** only).
- `PreintegratedCombinedMeasurements pim(params, biasHat)` — accumulate with
  `pim.integrateMeasurement(acc, gyro, dt)` per IMU sample.
- `CombinedImuFactor(X(i),V(i),B(i), X(j),V(j),B(j), pim)` — 6-way factor that
  jointly constrains both poses, both velocities, **and both biases** (the bias
  random-walk is baked in; this is why `CombinedImuFactor` is preferred over the
  older `ImuFactor` + separate `BetweenFactor<imuBias>`).
- `pim.predict(NavState_i, biasHat)` → `NavState_j` — used to seed the new
  keyframe's initial `Values`.
- After committing keyframe `j`, call `pim.resetIntegrationAndSetBias(newBias)`.

> **Critical for Meridian's hand-off:** the IMU is *already consumed by the L2
> front-end*. If you also add a `CombinedImuFactor` in L3 for the same interval,
> you double-count the IMU. The contract in §8 resolves this: the IMU factor is a
> **fallback only**, used on a window restart when the front-end could not supply
> a trustworthy relative pose + covariance.

---

## 8. ★ The L2 → L3 hand-off contract (most important section) ★

This is the part most likely to be implemented wrong, and the bugs (double
counting, over-confident covariance, gauge drift) are silent. State it precisely.

### 8.1 The principle: one factor, no double counting

The L2 front-end (B-spline CT odometry **or** FAST-LIO2 iEKF) **already fuses
LiDAR + IMU (+ camera)** over a window. Its output for the interval between two
keyframes `i` and `j` is a **relative pose** `T_ij` with a **marginal
covariance** `Σ_ij` (6×6, in GTSAM tangent order). The back-end must treat that
as a *single summarized measurement* and **must not** re-add the raw IMU or raw
LiDAR factors for the same interval. Doing so would count the same information
twice → artificially small covariance → overconfident, brittle estimate.

> **Contract rule #1 — Normal operation:** between consecutive keyframes, L3 adds
> **exactly one** `BetweenFactor<Pose3>(X(i), X(j), T_ij, noiseModel)` where
> `noiseModel = noiseModel::Gaussian::Covariance(Σ_ij)` and `Σ_ij` is the
> front-end's marginal covariance for that relative motion. **No IMU factor, no
> LiDAR scan-matching factor, in the normal path.**

### 8.2 Where `T_ij` and `Σ_ij` come from per front-end

- **FAST-LIO2-style iEKF:** the filter maintains a covariance `P` over its state.
  The relative pose between two keyframe times and its covariance are obtained by
  composing the filter's pose deltas and propagating `P` through the relative
  transform (the relative-pose covariance is `J P Jᵀ` with `J` the Jacobian of
  the relative pose w.r.t. the two absolute poses; or accumulate via the
  filter's process covariance over the interval). **The interface must expose a
  `Σ_ij` that already reflects observability** — e.g., in a feature-poor corridor
  the along-corridor uncertainty must be large.
- **Continuous-time B-spline:** after the sliding-window NLS solve, take the
  **marginal covariance of the relative pose between the two knot/keyframe times**
  from the Hessian (`Σ = H⁻¹` block, or Schur-complement of the relevant knots).
  This is the natural CT analogue.

### 8.3 Why marginal (not conditional) covariance, and the frame convention

- Use the **marginal** covariance of the relative motion, expressed as the
  uncertainty of `Log(T_i⁻¹ T_j)` in the body frame of `i` (matching
  `BetweenFactor<Pose3>`'s residual `Log(T_ij_measured⁻¹ · (T_i⁻¹ T_j))`).
- **[VERIFY] the exact tangent ordering and frame** between your front-end's
  covariance and GTSAM's `BetweenFactor` convention (GTSAM Pose3 tangent is
  rot-then-trans, body-frame on the left operand). A mismatch here silently
  rotates/swaps the covariance axes — a classic, hard-to-find bug. Add a unit
  test that perturbs `T_j` along one axis and checks the factor error grows along
  the expected covariance axis.

### 8.4 The window-restart fallback (the ONLY place L3 adds an IMU factor)

A front-end window can **restart / lose lock** (e.g., the iEKF resets after a
LiDAR degeneracy, or the B-spline window reinitializes). In that case the
front-end **cannot** supply a trustworthy relative `T_ij`/`Σ_ij` across the gap.

> **Contract rule #2 — Fallback:** when the front-end signals
> `relative_pose_valid == false` for the interval `[i,j]`, L3 instead adds a
> `CombinedImuFactor(X(i),V(i),B(i),X(j),V(j),B(j), pim_ij)` built from the raw
> IMU over that gap (preintegrated in L3 from buffered IMU). This is the **only**
> time `V(*)` and `B(*)` variables are introduced/constrained in L3. The IMU
> bridges the gap so the graph stays connected and metric.

> **Contract rule #3 — No overlap, ever:** for any given time interval, L3 uses
> **either** the `BetweenFactor` (normal) **or** the `CombinedImuFactor`
> (fallback), never both. The front-end's valid/invalid flag partitions the
> timeline. This is what guarantees no double counting.

### 8.5 Variable lifecycle (normal vs fallback)

| Mode | New variables added at keyframe `j` | New factors |
|---|---|---|
| Normal | `X(j)` (Pose3) only | `BetweenFactor<Pose3>(X(i),X(j),T_ij, Cov(Σ_ij))` |
| Fallback | `X(j)`, `V(j)`, `B(j)` | `CombinedImuFactor(...i...,...j...)`; if `V(i)`/`B(i)` don't exist, add a loose prior on them (`PriorFactor`) so they're observable |
| GNSS available at `j` | (no new var) | `GPSFactor(X(j), enu_j, Diagonal::Sigmas(gnss_σ))` wrapped in switchable/GNC if quality uncertain (§9–§10) |
| Loop closure `j↔k` | (no new var) | switchable `BetweenFactor<Pose3>(X(k),X(j),T_kj, robust)` (§9) |

### 8.6 Keyframe selection (what triggers a new `X(j)`)

Standard LIO-SAM-style policy: add a keyframe when **translation > d_thresh**
(e.g. 1.0 m) **or rotation > θ_thresh** (e.g. 0.2 rad) **or time > t_thresh**
since the last keyframe. The accumulated front-end delta since the last keyframe
becomes `T_ij`; its accumulated covariance becomes `Σ_ij`. This keeps the graph
sparse (bounded keyframes per metre) while preserving the relative constraint.

---

## 9. Robust GNSS & loop-closure factors: Switchable Constraints (Sünderhauf 2012)

### 9.1 The idea

A wrong loop closure (perceptual aliasing) is catastrophic for a standard
least-squares back-end: one bad `BetweenFactor` can warp the whole map.
**Switchable constraints** add, *per suspect factor*, a real-valued **switch
variable** `s_ij ∈ [0,1]` (or `ℝ`, squashed) that the optimizer can turn off if
the constraint disagrees with everything else.

### 9.2 The math (Sünderhauf §III)

For a suspect loop/GNSS constraint, replace the plain error term with:

```
e_robust = Ψ(s_ij) · e_constraint(X)              (scaled measurement residual)
         + (s_ij − γ_ij) / σ_switch               (switch prior, pulls toward "on")
```

where `Ψ(·)` is a sigmoid-like squash (Sünderhauf uses `Ψ(s)=s` clamped to
`[0,1]`, or a sigmoid), `γ_ij` is the **switch prior** (set to 1 = "trust the
loop"), and `σ_switch` controls how easily the optimizer is allowed to disable
it. When the loop is consistent, `s→1` and the factor acts normally; when it
conflicts, `s→0` and the factor's influence vanishes smoothly (differentiable,
so it stays inside the same nonlinear least-squares solve).

### 9.3 In GTSAM

GTSAM does **not** ship a first-class `SwitchableConstraint` factor, but the
pattern is implemented either as:

- a small **custom factor** that takes `(Pose_i, Pose_j, Switch_j)` and scales
  the between-error by the switch, plus a `PriorFactor<double>` on the switch
  with mean `1` and `σ_switch`; **or**
- the modern equivalent: **GNC** (§10), which GTSAM *does* ship as
  `gtsam::GncOptimizer` and which subsumes switchable constraints / dynamic
  covariance scaling for the batch case.

> **Recommendation for Meridian:** prefer **GNC for loop closures** (batch
> re-solve over the affected sub-window) and **Huber-robust unary factors for
> GNSS** in the incremental path. Use explicit switch variables only if you need
> the switch *state* to be persistent and queryable in the incremental iSAM2
> graph (it adds a variable per loop). See §11 for the incremental caveat.

---

## 10. Graduated Non-Convexity — GNC (Yang et al., T-RO/RA-L 2020)

### 10.1 The idea

Robust kernels (Geman-McClure, truncated least squares) are **non-convex** →
sensitive to initialization. GNC solves a *sequence* of problems, starting from a
nearly-convex surrogate (so the global optimum is easy to find) and gradually
"sharpening" it toward the true robust cost by annealing a control parameter `μ`.
At each `μ`, it does an **IRLS** solve: compute per-factor weights `w_i ∈ [0,1]`,
solve the weighted least squares, update `μ`, repeat.

### 10.2 The math (GNC, Geman-McClure surrogate, Yang §IV)

For the Geman-McClure (GM) robust cost with truncation parameter `c̄`
(`c̄²` = max acceptable squared residual ≈ chi-square threshold), GNC introduces
`μ` and the **closed-form weight update**:

```
w_i  =  ( μ c̄² / ( r_i² + μ c̄² ) )²
```

where `r_i` is the current residual norm of factor `i`. The schedule:

- Initialize `μ` large enough that the surrogate is ~convex
  (`μ_0` chosen from the max residual at the initial guess).
- After each weighted solve, **update** `μ ← μ / 1.4` (divide by a factor >1;
  GNC-GM *decreases* `μ`; **[VERIFY exact direction/factor]** — for the
  **truncated least squares (TLS)** surrogate `μ` instead *increases*; GTSAM's
  `GncParams` exposes `muStep` and chooses the right direction per loss).
- Stop when `μ` reaches the value that makes the surrogate equal the true robust
  cost (or weights converge / max iterations).

Final `w_i ≈ 0` ⇒ factor declared an **outlier**; `w_i ≈ 1` ⇒ inlier. GNC has
strong empirical and (for some problems) certifiable global-optimality guarantees
(Yang §V).

### 10.3 In GTSAM

```cpp
using GncLM = gtsam::GncOptimizer<gtsam::GncParams<gtsam::LevenbergMarquardtParams>>;
gtsam::GncParams<gtsam::LevenbergMarquardtParams> gncParams;
gncParams.setLossType(gtsam::GncLossType::TLS);     // or GM
gncParams.setMaxIterations(...);
gncParams.setMuStep(1.4);
gncParams.setRelativeCostTol(1e-5);
gncParams.setInlierCostThresholds(barc2);           // per-factor c̄² (chi2)
GncLM gnc(graph, initial, gncParams);
gtsam::Values result = gnc.optimize();
const gtsam::Vector& weights = gnc.getWeights();    // final inlier weights
```

`GncParams` notable knobs: `lossType` (`GM`/`TLS`), `muStep`, `maxIterations`,
`relativeCostTol`, `absoluteCostTol`, `setInlierCostThresholds` (sets `c̄²` from a
chi-square quantile for the factor dimension; e.g. for 6-DoF at 99%,
`barc2 = chi2inv(0.99, 6)`), and `setKnownInliers` / `setKnownOutliers` to fix
factors you already trust (e.g. odometry between-factors).

> **GNC is a *batch* optimizer** (it wraps LM/GN over a full graph). It is **not**
> incremental. See §11 for how Meridian uses it without stalling the live graph.

---

## 11. Robustness ↔ incrementality: how the pieces fit in Meridian

This is the subtle integration point.

- **iSAM2 is incremental** (Bayes tree). **GNC is batch.** You cannot just run GNC
  inside `ISAM2::update`.
- **Recommended architecture:**
  1. **Odometry between-factors and IMU fallback factors** go straight into iSAM2
     with their (front-end) covariances — these are trusted, never switched.
  2. **GNSS** factors go into iSAM2 with a **Huber-robust** Gaussian model
     (`Robust::Create(Huber::Create(1.345·σ-equiv), Diagonal::Sigmas(gnss_σ))`).
     Huber is convex → safe inside the incremental IRLS. Optionally gate by GNSS
     quality (fix/float, HDOP, number of sats) *before* adding the factor.
  3. **Loop closures** are validated in a **separate batch GNC verification** over
     the *affected sub-graph* (the candidate loop's two cliques + path), *before*
     committing them to iSAM2. Loops that GNC marks inliers are added to iSAM2 as
     plain (or Huber) `BetweenFactor`s; outliers are dropped. This keeps the live
     iSAM2 graph clean and convex while still getting GNC's outlier rejection on
     the most dangerous factors.
  4. **PCM (Pairwise Consistency Maximization)** is the front gate for loop
     *batches*: build the consistency graph of candidate loops, find the maximum
     consistent set, and only those go to GNC/iSAM2. (PCM is covered in the loop
     dossier; here, note PCM and GNC are complementary — PCM picks a mutually
     consistent set, GNC/Huber adds graceful residual robustness.)

- **Why not switchable-constraint variables in iSAM2?** You *can* add a `Switch`
  variable + switch-prior factor per loop into the incremental graph. Costs: (a)
  one extra variable per loop grows the Bayes tree; (b) switches make that part
  of the graph non-convex → more relinearization churn. The batch-GNC-then-commit
  approach above avoids both. Use in-graph switches only if you must keep the
  switch *posterior* online (rare for Meridian).

### 11.1 Per-axis observability → inflate the BetweenFactor noise (degeneracy)

When the front-end is degenerate along an axis (LiDAR corridor: along-corridor
translation unobservable; open field: yaw weak), the **right** response is **not**
to drop the factor but to **inflate the corresponding axis of `Σ_ij`** so the
back-end trusts the IMU/GNSS along that axis instead.

- The front-end should already encode this in `Σ_ij` (its marginal covariance is
  large along the unobservable direction — this is the whole point of passing the
  *marginal* covariance, §8.3).
- As a defensive belt-and-braces measure in L3, you can additionally **inflate**
  `Σ_ij` along directions where the front-end reports a low **degeneracy /
  observability score** (e.g. smallest eigenvalue of the scan-matching Hessian
  below a threshold, à la Zhang's "On Degeneracy of Optimization-based State
  Estimation"). Concretely: rotate `Σ_ij` into the degenerate eigenbasis, set the
  bad axis variance to a large value, rotate back. This makes the back-end ignore
  that axis of the relative measurement without disconnecting the graph.
- Failure mode if you *don't*: the back-end believes a confident-but-wrong
  along-corridor displacement → drift that no loop closure can fully undo.

### 11.2 Other iSAM2 failure modes to guard against

- **Indefinite/ill-conditioned linear system** (`IndeterminantLinearSystemException`
  in GTSAM): a variable is unconstrained (no factor pins a DoF) — e.g. forgot a
  prior on the first pose/velocity/bias, or an extrinsic with no excitation. Fix:
  always anchor the gauge (prior on `X(0)`, and on `V(0)`/`B(0)` when introduced)
  and add weak priors on online extrinsics until they're observable.
- **Relinearization thrash on loop closure:** a big loop suddenly moves many
  poses past `relinearizeThreshold` → a large sub-tree re-eliminates → latency
  spike. Mitigate with Dogleg (bounded step), and by committing big loops on a
  separate thread / accepting a one-frame latency.
- **Unbounded growth:** without marginalization the graph grows for the whole
  multi-hour mission → memory and per-step cost creep up. See §12.

---

## 12. Bounded multi-hour graphs: marginalization & fixed-lag smoothing

A pure `ISAM2` keeps *all* keyframes forever. For Meridian's multi-hour runs you
need to **bound** the graph. Two GTSAM tools:

### 12.1 `IncrementalFixedLagSmoother` (in `gtsam_unstable`)

- Header: `gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h`. It wraps
  iSAM2 and **marginalizes out variables older than a lag** `L` (seconds). You
  pass, per `update`, a `KeyTimestampMap` giving each new key a timestamp; the
  smoother drops keys older than `now − L`.
- **Marginalization** = Schur-complement the old variables out, leaving a dense
  **linear prior** (a `LinearContainerFactor`) on the remaining boundary
  variables that summarizes everything removed. This is exact for the linearized
  problem at the time of marginalization (but it *freezes* that linearization —
  you can't relinearize marginalized-away factors later).
- ⚠️ **It lives in `gtsam_unstable`** → API may change between GTSAM releases,
  and it has **confirmed sharp edges** (from GTSAM users' list / docs):
  - **Do not call `update()` more than once per timestamp** (e.g. once for
    odometry then once for vision at the same time) — this is a documented cause
    of failure. Batch all factors for a keyframe into a single `update()`.
  - **Segfaults have been reported when marginalizing a variable that has no
    factors other than a prior** — always ensure a variable still entering the
    lag boundary is constrained by at least one non-prior factor.
  - Errors like *"Requested variable 'xNN' is not in this VectorValues"* appear
    when keys/timestamps get out of sync. Keep the `KeyTimestampMap` exactly in
    step with the keys you insert.
  Treat the exact API as **[VERIFY against your pinned GTSAM version]**, but the
  above failure modes are real and have bitten many users.

### 12.2 The tension for Meridian: fixed-lag vs. loop closures

- A **pure fixed-lag smoother is wrong for a mesh-mapping SLAM** that needs loop
  closures: once you marginalize old keyframes you can no longer add a loop
  closure to them, and you lose the global consistency that the colourised mesh
  needs.
- **Recommended Meridian design (hybrid):**
  - Keep a **full iSAM2 keyframe pose graph** for the *poses* (these are cheap:
    one Pose3 per ~1 m). Multi-hour at 1 keyframe/m and a few km is only
    thousands–tens-of-thousands of poses — well within iSAM2's range, especially
    with sparse loop topology.
  - **Marginalize only the high-rate / transient variables** — velocities `V`,
    biases `B`, and any visual landmarks — using a fixed-lag window, because
    those are only needed locally (for the IMU fallback factor and bias
    estimation). The persistent pose graph is what loop closures attach to.
  - If even the pose graph gets too big (very long missions), apply **graph
    sparsification / keyframe culling** (drop redundant keyframes in
    overlapping-view regions, merging their constraints) rather than naive
    fixed-lag marginalization of poses.

> **Practical recommendation:** start with **plain `ISAM2`** for poses + a small
> **fixed-lag window for V/B** (either a second `IncrementalFixedLagSmoother`
> instance, or just don't keep V/B variables beyond the window — only introduce
> them on fallback intervals, §8.4, so they're naturally transient). Only adopt
> full `IncrementalFixedLagSmoother` for the pose graph if profiling proves
> iSAM2's pose graph is the bottleneck — and accept that it caps loop-closure
> range to the lag.

---

## 13. Querying marginal covariance for the place-recognition pre-filter

The loop-closure module (Scan Context++ / STD) should only *attempt* expensive
GICP verification when the current pose is uncertain enough that a loop is
plausible *and* the candidate is within the **search radius implied by the
pose covariance** (a 3σ gate). The back-end provides that covariance.

### 13.1 API

- **Cheap, from the live graph:** `Matrix Σ = isam2.marginalCovariance(X(j));`
  returns the 6×6 marginal covariance of the latest pose (GTSAM tangent order
  rot-then-trans). This reuses the Bayes tree (a partial back-substitution +
  block inverse along the path to the root) — much cheaper than a batch
  `Marginals`.
- For a **relative** uncertainty between current pose and a candidate keyframe
  `k`, use `gtsam::Marginals::jointMarginalCovariance({X(j),X(k)})` and compose,
  or query both marginals and add (approximate). For the pre-filter a per-pose
  marginal is usually sufficient.

### 13.2 How the pre-filter uses it

```
Σ_j  = isam2.marginalCovariance(X(j))      // 6x6
σ_pos = sqrt(largest eigenvalue of Σ_j[trans-block])   // positional 1σ
search_radius = k_gate * σ_pos             // e.g. k_gate = 3
// only run Scan Context++ NN search / STD matching against keyframes whose
// estimated position is within search_radius of X(j)'s estimate.
```

This (a) bounds the place-recognition search, (b) avoids proposing loops the
geometry says are impossible, and (c) gives PCM/GNC a smaller, higher-quality
candidate set. **Cost note:** `marginalCovariance` is cheap but not free — call
it at keyframe rate (already throttled), not per LiDAR scan.

### 13.3 Failure mode

Over-confident `Σ_j` (e.g. from a double-counted hand-off, §8) → too-small search
radius → **missed loop closures** → unbounded drift. This is another reason the
L2→L3 no-double-counting contract is load-bearing: it keeps the covariance
honest, which keeps loop closure working.

---

## 14. Concrete recommendations for Meridian (the spec authors' checklist)

**Optimizer object**

- Use `gtsam::ISAM2` with `ISAM2Params` as in §4.2/§4.3:
  `optimizationParams = ISAM2DoglegParams()`, `factorization = CHOLESKY`
  (switch to `QR` only on degeneracy exceptions), `relinearizeSkip = 1`,
  per-type `relinearizeThreshold` map, `evaluateNonlinearError = true` in
  diagnostic builds, `cacheLinearizedFactors = true`,
  `findUnusedFactorSlots = true` (you will churn switchable/loop factors).
- Call `update(newFactors, newValues)` once per keyframe; for big loop closures
  call `update()` a second time the same step (empty graph) to let it iterate, or
  a few extra LM iterations — iSAM2 only does **one** Gauss-Newton/Dogleg step per
  `update`, so 1–2 extra empty `update()` calls help convergence after a loop.

**State & symbols**

- `X(i)`=`Pose3`, `V(i)`=`Vector3`, `B(i)`=`imuBias::ConstantBias`,
  `E(s)`=`Pose3` per online extrinsic (LiDAR↔IMU, cam↔IMU, etc.).
- Gauge: `PriorFactor<Pose3>` on `X(0)` (tight), priors on `V(0)`/`B(0)` when
  first introduced, **weak** priors on each `E(s)` until observable.

**Factors (the canonical set)**

| Purpose | Factor | Noise model |
|---|---|---|
| First-pose anchor | `PriorFactor<Pose3>` | tight `Diagonal::Sigmas` |
| World origin from GNSS | `PriorFactor<Pose3>` or `GPSFactor` | `Diagonal` from GNSS σ |
| **Normal odometry (L2→L3)** | **`BetweenFactor<Pose3>`** | **`Gaussian::Covariance(Σ_ij)`** from front-end marginal |
| Window-restart fallback | `CombinedImuFactor` | from `PreintegrationCombinedParams` |
| GNSS position | `GPSFactor` | `Robust(Huber, Diagonal::Sigmas(gnss_σ))` |
| Loop closure | `BetweenFactor<Pose3>` | post-GNC: plain or `Robust(Huber,·)`; pre-commit validate via batch GNC |
| Online extrinsics | embedded in the relevant measurement factor (compose `E(s)` into `h`) | weak prior + data |

**Robustness**

- GNSS & visual reprojection: **Huber** (`k≈1.345`) inside iSAM2 (convex, safe).
- Loop closures: **batch GNC** (`GncOptimizer`, TLS loss, `setInlierCostThresholds`
  from chi-square) on the affected sub-graph *before* committing; PCM upstream to
  pick a mutually consistent loop set.
- Degeneracy: front-end inflates `Σ_ij` along unobservable axes; L3 optionally
  re-inflates from a degeneracy score (§11.1). **Never** silently drop a factor —
  inflate its covariance.

**Bounding the graph**

- Full `ISAM2` for the **pose graph**; keep `V`/`B`/landmarks **transient**
  (fixed-lag window or only-introduced-on-fallback). Reserve
  `IncrementalFixedLagSmoother` for V/B; adopt it for poses only if profiling
  forces it and you can accept a bounded loop-closure range.

**Covariance for loop pre-filter**

- `isam2.marginalCovariance(X(latest))` at keyframe rate → 3σ search radius for
  Scan Context++/STD candidate gating (§13).

**Health monitoring**

- Log `ISAM2Result.errorBefore/errorAfter`, `variablesRelinearized`,
  `variablesReeliminated`, `cliques` each update; alarm on chi-square spikes
  (bad factor), relinearization-count spikes (loop thrash), and
  `IndeterminantLinearSystemException` (gauge/observability bug).

---

## 15. Open questions / things to verify before writing the formal spec

1. **Exact `ISAM2Params` defaults** (`relinearizeThreshold=0.1`?,
   `relinearizeSkip=10`?, `enablePartialRelinearizationCheck` default,
   `findUnusedFactorSlots` default) against the **pinned GTSAM version** —
   defaults have shifted across 4.0/4.1/4.2/develop.
2. **GNC `muStep` direction** for GM vs TLS and the exact stopping rule — read
   `gtsam/nonlinear/GncOptimizer.h` + `GncParams.h` and Yang §IV; the surrogate
   sharpening direction differs between GM and TLS.
3. **`NavState` / `Pose3` tangent ordering and the `BetweenFactor` frame
   convention** vs. the front-end's covariance convention (§8.3) — write a unit
   test; this is the highest-risk silent bug.
4. **`IncrementalFixedLagSmoother` API stability** in the pinned GTSAM
   (`gtsam_unstable`) and whether it can coexist with a separate full-`ISAM2`
   pose graph in the same process, or whether you marginalize V/B by simply not
   retaining them.
5. **How each front-end (B-spline CT vs FAST-LIO2 iEKF) actually exposes
   `Σ_ij`** — confirm both produce a *marginal relative-pose* covariance in a
   documented frame/order, and define the `relative_pose_valid` flag semantics
   that drive the fallback (§8.4). This is part of the L2 interface dossier and
   must match this contract exactly.
6. **Gravity sign/axis (`n_gravity`)** for Meridian's chosen world frame (ENU vs
   NED) used by `CombinedImuFactor` (§7.2).
7. Whether Meridian's camera enters L3 at all as separate visual factors
   (`SmartProjectionPoseFactor`) or only via the fused L2 odometry — affects the
   variable set and the double-counting contract for the visual channel too
   (same rule: fuse-once in L2, summarize as the between-factor).

---

*End of dossier 09 — back-end optimization (iSAM2 / GTSAM).*
