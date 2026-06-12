# 04 — L2 Front-End: Discrete LiDAR-Inertial Odometry

> **Spec status:** normative for the L2 layer, and written **from the implemented
> system**: `meridian::lio::LioFrontEnd` (`src/meridian_frontend/src/lio/`). This
> document is the contract for what the front-end does; where prose and code ever
> disagree, fix one of them — do not let them drift.
>
> **It implements, it does not redefine.** The boundary value types
> (`KeyframePacket`, `NavState`, `ObservabilityReport`, `LidarScan`, `ImuSample`,
> `CalibrationSet`, `FrontEndDiagnostics`, `PreprocessedGroup`) and the
> `IFrontEnd` interface are canonically defined in **spec 01** and must not be
> redefined here; structs shown here are quotations. If the two disagree, spec 01
> wins and this document is the bug.
>
> **Scope of the estimator.** L2 is a **discrete LiDAR-inertial odometry**: one
> LiDAR + one IMU, sweep-at-a-time. Camera frames pass through to the
> `KeyframePacket` untouched (no visual residual); GNSS is not consumed by L2.
> The L0/L1 plumbing for both stays in place (spec 02/03) so re-adding them is a
> spec-04 extension, not a re-architecture.
>
> **Provenance.** The registration/IMU-prior algorithm is derived from **rko_lio**
> (M.V.R. Malladi et al., MIT-licensed); the implementation is an internal
> rewrite onto Meridian types, conventions, and determinism rules. This note is
> the one place that lineage is recorded — never in code comments.

---

## Table of contents

0. [Scope and dataflow](#0-scope-and-dataflow)
1. [State and frames](#1-state-and-frames)
2. [L1→L2 handoff](#2-l1l2-handoff)
3. [Internal constant-screw deskew](#3-internal-constant-screw-deskew)
4. [The IMU tracker](#4-the-imu-tracker)
5. [The local voxel map](#5-the-local-voxel-map)
6. [Registration: Gauss-Newton + adaptive gravity regularizer](#6-registration-gauss-newton--adaptive-gravity-regularizer)
7. [The covariance chain (rung 0)](#7-the-covariance-chain-rung-0)
8. [Per-axis observability](#8-per-axis-observability)
9. [Keyframe cadence and KeyframePacket emission](#9-keyframe-cadence-and-keyframepacket-emission)
10. [Failure and reseed semantics](#10-failure-and-reseed-semantics)
11. [Determinism guarantees](#11-determinism-guarantees)
12. [Threading](#12-threading)
13. [Telemetry surface](#13-telemetry-surface)
14. [Parameters](#14-parameters)
15. [Failure modes](#15-failure-modes)
16. [Known limitations](#16-known-limitations)

---

## 0. Scope and dataflow

Per sweep, in order, all synchronous inside `ingest()`:

```
PreprocessedGroup ──▶ gap check ──▶ feed group IMU to tracker ──▶ interval prior
   ──▶ constant-screw deskew to sweep end ──▶ keypoint downsample
   ──▶ GN registration against the voxel map (+ gravity regularizer)
   ──▶ rebase tracker & live state on the solved pose
   ──▶ map insert + range clip ──▶ relative-covariance accumulation
   ──▶ observability ──▶ keyframe emission (cadence / chain rules)
   ──▶ telemetry (body/scan, path sample, scalars)
```

The only L2→L3 product is the `KeyframePacket` stream (spec 01 §6); the sideways
products are `live_state()` (IMU-rate `NavState`) and `diagnostics()`. There is
no sliding window, no carried prior, and no asynchronous worker: every sweep
is solved to completion before `ingest()` returns.

`makeFrontEnd` (`frontend_factory.cpp`) builds the implementation selected by
`frontend.kind`; `lio` is the only kind. Packets carry `frontend_kind = 2`
(`1` is the retired continuous-time front-end and must never reappear in new
data; `0` was the even earlier iEKF oracle).

## 1. State and frames

**Frames.** The estimation frame $F_e$ is `imu_link` (`Frame::ImuLink`);
emitted body-frame quantities are tagged `Frame::Body`. The world frame is
`Frame::Odom`, **gravity-aligned at static initialization**: world $+z$ is the
measured up direction, the origin is the standstill pose, and yaw is zero by
construction (§4.1). L2 owns `odom`; L3 owns `map` and the `odom→map` relation.

**State.** Three coupled estimates, all `NavState`-shaped
(`[p, R, v, b_g, b_a, g]`, spec 01 §3.2):

| estimate | rate | owner | role |
|---|---|---|---|
| registered state | sweep end (10 Hz) | `LioFrontEnd` | the solved pose chain; everything downstream derives from it |
| propagated state | IMU rate | `ImuTracker` | dead-reckoning between sweeps; the registration initial guess; rebased onto each solved pose |
| live state | IMU rate | `LioFrontEnd` | `live_state()` output, advanced by `ingest_imu_live` and rebased after every solve; never read by estimation |

Biases are fixed at their static-init values for the whole run (§16.1); gravity
is fixed at magnitude 9.81 m/s² along world $-z$ after init.

**Tangent conventions.** All internal 6-DoF tangents are **translation-first**
`[ρ; φ]` (spec 01 §3.1). Three charts appear in the math and each is named where
used: the *left/world* chart of the GN solve (§6), the *right/body* chart of
the covariance and observability (§7, §8), and the **rotation-first** layout of
`KeyframePacket.constraint_cov`, produced by exactly one reorder at pack time
(§9). Do not touch any 6×6 block here without identifying its chart.

## 2. L1→L2 handoff

The front-end ingests one `PreprocessedGroup` per sweep — a thin wrapper over
the assembled `MeasureGroup` (spec 01 §7.3, Appendix A):

* **`scan`** — the L1-filtered `LidarScan`: range/validity gated and decimated,
  **not deskewed** (spec 03: L1 output is raw geometry; motion compensation is
  L2-internal, §3). Per-point `t_offset_ns` is **mandatory**; a scan without
  per-point time is rejected upstream and never reaches L2.
* **`imu`** — the `ImuSample`s spanning `(lower, t_end]` with the one sample
  straddling `t_begin` re-delivered at the head of the next group. Estimation
  consumes **only** this bundled stream (deterministic, eviction-protected);
  the straddle re-delivery is deduplicated by stamp inside the tracker (§4).
* **`image` / `gnss`** — passed through (§9) / ignored, respectively.

`ingest_imu_live` is the low-latency side tap: it advances the live state only
and never influences a solve — the same samples arrive again inside the next
group, which is what registration actually uses.

Raw IMU samples are rotated from `imu_link` into $F_e$ by the extrinsic
rotation only; the lever arm is deliberately ignored (the transport terms need
differentiated rates, and with the IMU centimeters from $F_e$ the centripetal
contribution sits below the accelerometer noise floor). LiDAR points fold in
the full extrinsic $T_{F_e L}$ during deskew.

## 3. Internal constant-screw deskew

Deskew happens inside the front-end, per sweep, before association
(`ScanRegistration::deskew`). The sweep's intra-scan motion is modelled as a
**constant body screw** $\xi = [v_b;\ \omega_b]$:

* $\omega_b$ — the interval-mean unbiased gyro rate over $[t_\text{begin}, t_\text{end}]$ (§4.2);
* $v_b$ — the tracker's propagated world velocity at $t_\text{end}$, rotated
  into the body frame.

Each return with absolute time $t_i = \texttt{stamp\_start} + \texttt{t\_offset\_ns}$
is re-expressed in the body frame at the sweep end:

$$
p_{F_e}(t_\text{end}) \;=\; \mathrm{Exp}\big((t_i - t_\text{end})\,\xi\big)\;\big(T_{F_e L}\, p_L\big),
$$

since under a constant screw the body frame at $t_i$ relates to the frame at
$t_\text{end}$ by $\mathrm{Exp}((t_i - t_\text{end})\xi)$. Returns with
non-positive range or range > `lio.max_range_m` are dropped here. The deskewed
sweep is the **one** cloud used for everything downstream: keypoints, map
insertion, `body/scan` telemetry, and `KeyframePacket.cloud_body`.

The keypoint set fed to registration is a voxel downsample of the deskewed
sweep at edge `keypoint_voxel_factor × voxel_size_m`, keeping the **first**
point per occupied cell in input order (deterministic, §11).

## 4. The IMU tracker

`ImuTracker` owns static initialization, IMU-rate dead reckoning, and the
interval statistics the registration prior consumes. Samples at or before its
watermark stamp are dropped (the group-boundary straddle re-delivery, §2).

### 4.1 Static initialization

The front-end holds incoming groups (bounded backlog, oldest dropped with a
warning) until the tracker initializes. With `init_stationary_s > 0` the
tracker averages gyro and accel over that standstill window, then fixes:

* **attitude**: the minimal rotation taking the mean accelerometer reading onto
  world $+z$ — roll/pitch from gravity, yaw exactly zero;
* **gyro bias** $b_g$ = mean gyro (standstill asserted);
* **accel bias** $b_a$ = mean accel minus the expected rest reaction — only the
  along-gravity component is observable at rest; the orthogonal part is
  absorbed into the recovered tilt;
* **gravity**: $(0, 0, -9.81)$ in world (direction measured, magnitude pinned).

Held groups then drain in arrival order through the normal path (their IMU is
already inside the tracker, so the re-feed dedups to a no-op). With
`init_stationary_s = 0` the tracker starts immediately from identity attitude
and zero biases — only valid for bench setups that are level and bias-corrected
upstream.

### 4.2 Propagation and the interval prior

Each new sample propagates the IMU-rate state by gravity-compensated strapdown
integration (`exp`-on-SO(3) attitude, constant accel over the sample interval).
Per scan interval the tracker accumulates the mean unbiased rate
$\bar\omega_b$, the mean gravity-compensated body acceleration $\bar a_b$, and
the **Welford variance of the specific-force magnitude** — the excitation
signal that drives the gravity-regularizer weight (§6). `interval_prior()`
consumes and resets these accumulators; an empty interval returns the last
measured rate, zero acceleration, and an uninformative variance ($10^9$) so the
regularizer is suppressed rather than wrongly confident.

A scalar predict/update filter on the body-acceleration magnitude persists
across intervals, with process noise from a uniform-jerk model (jerk uniform on
$[-j, j]$ has variance $j^2/3$; over $dt$ it perturbs the acceleration by
$j\,dt$) using `lio.max_expected_jerk`, and the interval scatter as measurement
noise. It is exported for diagnostics; the regularizer reads the raw interval
variance.

After every successful solve (and on every failure path) the tracker is
**rebased**: its pose/velocity reset to the registered state so dead reckoning
always resumes from the best estimate. Biases are never updated after init.

## 5. The local voxel map

`VoxelGridMap`: an `std::unordered_map` from integer voxel cell
(`floor(coord / voxel_size_m)` per axis — floor, not truncation, so negative
coordinates do not fold onto the origin cell) to a small vector of world-frame
points.

* **Insertion** respects two bounds per cell: at most `max_points_per_voxel`
  points, and a minimum spacing of `voxel_size_m / sqrt(max_points_per_voxel)`
  to any stored point, so one surface patch cannot saturate a cell's budget.
* **Nearest-neighbour** probes the query's cell and its 26 neighbours in a
  fixed order (x, then y, then z, each −1..1), accepting the closest point
  within `max_corr_dist_m`; ties resolve to the first candidate encountered,
  independent of hash-map iteration order.
* **Maintenance**: after each solve, voxels whose cell centre is farther than
  `max_range_m` from the solved position are erased (`clipFarFrom`) — the map
  is local by construction and its memory is bounded by travel, not mission
  length. `clear()` exists only for the reseed path (§10).

The map is estimation state, not a product: nothing outside the front-end reads
it, and `apply_correction` deliberately does not move it (§16.5).

## 6. Registration: Gauss-Newton + adaptive gravity regularizer

`ScanRegistration::registerScan` aligns the keypoint set against the map from
the tracker's propagated guess. Cost: point-to-point over the nearest-map-point
associations, re-associated **every iteration**.

**Data term.** For keypoint $p$ (body frame at sweep end) and its matched map
point $q$, with the pose $T$ acting on points, the residual is
$r = T\,p - q$. Under the **left/world perturbation** $T \leftarrow
\mathrm{Exp}(\delta x)\,T$ the Jacobian at $\delta x = 0$ is

$$
J = \big[\,I \;\big|\; -[T p]_\times\,\big] \quad\text{(translation-first columns)},
$$

accumulated strictly in keypoint order into normal equations that are
**averaged** over the correspondence count $n$: $H = \frac{1}{n}\sum J^\top J$,
$b = \frac{1}{n}\sum J^\top r$ (averaging makes the regularizer weight
scale-free in $n$).

**Adaptive gravity regularizer.** A roll/pitch anchor toward the prior
attitude, weighted to fade with inertial excitation. With $g_\uparrow$ the
world up vector ($|g| = 9.81$) and the anchor $a = R_0^\top g_\uparrow$ frozen
at the guess attitude:

$$
r_\text{ori} = R^\top g_\uparrow - a,\qquad
J_\text{ori} = \big[\,0 \;\big|\; R^\top [g_\uparrow]_\times\,\big],
$$

added to $H, b$ with weight $1/\beta$ where

$$
\beta = \texttt{min\_beta}\cdot\big(1 + \mathrm{Var}(\|f\|)\big),
$$

$\mathrm{Var}(\|f\|)$ the interval specific-force-magnitude variance (§4.2).
Still platform ⇒ small variance ⇒ strong anchor; aggressive motion ⇒ weak
anchor. The block is disabled when the interval held fewer than two samples
(no variance information). The regularizer constrains only the rotation
columns; translation degeneracy is the observability report's job (§8).

**Solve.** $H\,\delta x = -b$ by LDLT; update $T \leftarrow
\mathrm{Exp}(\delta x)\,T$; converged when $\|\delta x\| <$ `convergence_eps`,
capped at `icp_max_iterations`. The result fails (`converged = false`) when
association is lost entirely, the iteration cap is hit without convergence, or
the final correspondence count is below `min_keypoints`.

## 7. The covariance chain (rung 0)

The packet covariance is a **Laplace approximation from the data term only** —
deliberately rung 0 of the honesty ladder. The chart conversions are the
classic silent-bug surface; each step below is the contract and each is
unit-tested (including a Monte-Carlo test of the full chain against empirical
scatter).

1. **Per-scan information, left/world, translation-first.** Re-associate at the
   solved pose; $H_\text{data} = \sum J_i^\top J_i$ — **unaveraged**, and
   **without** the $\beta$-regularizer (a heuristic prior carries no
   measurement information; including it would hide degeneracy the
   observability report must expose). The isotropic noise scale is
   $\hat\sigma^2 = \chi / (3N - 6)$ ($\chi$ = final sum of squared residuals,
   $N$ = correspondences, 3 rows each, 6 pose DoF absorbed), guarded by
   $N \ge$ `min_keypoints`. Then
   $\Sigma_\text{left} = \hat\sigma^2 (H_\text{data} + 10^{-6} I)^{-1}$.
2. **→ right/body** (the spec-01 canonical chart): from
   $\mathrm{Exp}(\delta x_\text{left})\,T = T\,\mathrm{Exp}(\delta x_\text{right})$,
   $\delta x_\text{right} = \mathrm{Ad}_{T^{-1}}\,\delta x_\text{left}$ to first
   order, so
   $\Sigma_\text{right} = \mathrm{Ad}_{T^{-1}}\, \Sigma_\text{left}\, \mathrm{Ad}_{T^{-1}}^\top$.
3. **Relative accumulation between keyframes** (right-perturbation
   composition): per registered sweep with step $\delta T$ from the previous
   registered pose,
   $\Sigma_\text{rel} \leftarrow \mathrm{Ad}_{\delta T^{-1}}\, \Sigma_\text{rel}\, \mathrm{Ad}_{\delta T^{-1}}^\top + \Sigma_\text{step}$,
   reset to zero at every keyframe.
4. **→ rotation-first, exactly once, at pack time**: the single rotation-first
   block in the system is `KeyframePacket.constraint_cov`, produced by
   `reorderTransRotToRotTrans()` when the packet is assembled (§9). No other
   code path reorders.

**Honesty caveats (the rung-0 contract):** successive steps are treated as
independent — the correlation through the shared map is ignored; the noise
model is isotropic per correspondence; the regularizer is excluded. The
covariance is therefore *shape-honest but scale-uncalibrated*. The follow-up is
**NEES calibration against ground truth** (rung 1): measure the normalized
estimation error squared of the emitted relative constraints over GT runs and
introduce a calibrated scale before the back-end tuning leans on absolute
covariance magnitudes.

## 8. Per-axis observability

Point-to-point $J^\top J$ carries an isotropic translation block (each
correspondence pins all three axes equally), so it cannot flag surface-sliding
degeneracy. The `ObservabilityReport` therefore projects each correspondence
onto the **local map surface normal** — the component re-association cannot
cancel. For each matched keypoint, the map neighbourhood within
`voxel_size_m` of the match (minimum 5 neighbours, non-collinear) yields a
planar-patch normal $n$ by smallest-scatter eigenvector; the right/body-chart
information row is

$$
\big[\, n_b^\top \;\big|\; (p \times n_b)^\top \,\big],\qquad n_b = R^\top n,
$$

averaged over correspondences into a 6×6 $h$. The per-axis score is the
saturating

$$
s_i = \frac{h_{ii}}{h_{ii} + \kappa},\qquad \kappa = 0.01,
$$

so an axis carrying ≥ ~1/10 of the normal energy scores > 0.9 and an axis with
no normal support (sliding along a lone plane; yaw about its normal) scores
near 0. Frame `Body`, translation-first, per spec 01 §3.4; the report rides
every packet and feeds back-end noise inflation (spec 05).

## 9. Keyframe cadence and KeyframePacket emission

**Cadence.** A registered sweep is a keyframe when, relative to the previous
keyframe, any of:

* distance ≥ `keyframe.dist_m`,
* rotation ≥ `keyframe.rot_deg`,
* elapsed time ≥ `keyframe.time_s`,

or unconditionally when the relative chain is broken (§10) and the sweep
converged — the chain must be re-anchored at the first opportunity.

**Packet contents.** `T_ref_body` in `Frame::Odom` at the sweep-end stamp;
`kinematics_included = false` always; the observability report of the solve;
`cloud_body` = the deskewed sweep (body frame at the stamp); `image` = the most
recent `group.image` seen (passthrough; may be null) with the `T_body_cam`
extrinsic snapshot; `calib_version`; `frontend_kind = 2`.

**Constraint kinds.** Exactly two are emitted:

| situation | kind | covariance (before the pack-time reorder) |
|---|---|---|
| run's first keyframe (bootstrap sweep) | `AbsolutePrior` | $10^{-8} I$ — the pose *defines* the odom origin, tight by construction |
| first keyframe after a chain break | `AbsolutePrior` | first converged per-scan $\Sigma_\text{right}$ × `reseed_cov_inflation` (fallback: $10^{-8} \cdot \texttt{reseed\_cov\_inflation} \cdot I$ if no converged solve preceded the emission) |
| steady state | `RelativeBetween` | the accumulated $\Sigma_\text{rel}$ (§7, step 3), with `rel_to_id` / `T_relto_this` against the previous keyframe |

> **Contract change.** The retired front-end's restart fallback emitted
> `ConstraintKind::ImuPreintegration` with an IMU summary. **This front-end
> never emits `ImuPreintegration`** — a broken chain re-anchors with a
> mid-stream `AbsolutePrior`, which spec 05 explicitly supports. The enum value
> remains in spec 01 for the type's stability; L3 must not expect it from
> `frontend_kind = 2`.

## 10. Failure and reseed semantics

No exception crosses `IFrontEnd` — `ingest()` wraps its body and converts any
throw into a `frontend/lio/error` event while holding the last good state.
The structured paths, in escalation order:

1. **Sparse sweep** (keypoints < `min_keypoints`): the sweep is skipped
   (`frontend/lio/reject`); the dead-reckoned guess becomes the registered
   state so time advances; the map is kept. During bootstrap a sparse sweep
   merely defers bootstrap.
2. **Registration failure** in steady state (non-convergence or final
   correspondences below the floor): same as (1) — drop the sweep, keep the
   map, try the next sweep. `diagnostics().restarted` flags it.
3. **Sensor gap** (`t_begin − prev_t_end > max_gap_s`): the state bridges the
   hole on **constant velocity** (dead reckoning must not integrate one giant
   dt), the tracker is rebased onto the bridged state, and a **reseed is
   armed** (`frontend/lio/gap`). The map is kept; the next registration
   outcome decides:
   * **success** → the bridge held; disarm, continue normally.
   * **failure** → the bridged prediction and the map disagree beyond
     recovery: **clear the map**, zero $\Sigma_\text{rel}$, mark the keyframe
     chain broken, and re-anchor the map on the failing sweep at the
     dead-reckoned pose (`frontend/lio/reseed`). If that sweep is itself
     unusable, fall back to the bootstrap path on the next usable sweep.
4. **Chain repair**: after a reseed, the first converged covariance — inflated
   by `reseed_cov_inflation` — prices the `AbsolutePrior` that re-roots the
   keyframe chain (§9).

`apply_correction` (loop-closure feedback) applies the rigid world-frame shift
implied by the most recent moved keyframe to the registered, propagated, and
live states. The map stays in the pre-correction frame: a small correction
keeps registering against it; a large one fails the next solve, which path (3)
absorbs.

## 11. Determinism guarantees

Two runs over identical input are **bit-identical**, live or replay — there is
no asynchronous work, no wall-clock-dependent branch, and no thread count in
the estimation path. The `deterministic` factory flag is accepted for interface
compatibility and is a documented no-op. The guarantee rests on rules encoded
in the unit tests:

1. The voxel map is **never iterated to generate estimation data** —
   association walks the time-ordered keypoint vector and probes the 27-cell
   neighbourhood in fixed order; in-cell scans are insertion-ordered.
2. Map iteration occurs only for **order-independent erasure** (the range
   clip, a pure per-cell predicate).
3. GN accumulation is **sequential** in keypoint order; no OpenMP/TBB anywhere
   under `src/lio/` (CI grep gate, spec 11).
4. Voxel hashing casts each index to `uint64` **before** arithmetic (defined
   wraparound); cell indices use `std::floor`. Cell identity is decided by key
   equality, never by the hash.
5. Downsample survivors are appended in input scan order, never by walking a
   set.

Consequence: a deterministic-replay A/B that differs at all differs because the
input differed. There is no rerun-once caveat.

## 12. Threading

The front-end is **synchronous and thread-confined**: `ingest`,
`ingest_imu_live`, and `apply_correction` all execute on the pipeline's
front-end stage thread (spec 00 §11), which feeds them serially from `Q_meas`.
The keyframe sink fires inline on that thread and must only enqueue.
`live_state()`/`diagnostics()` follow the `IFrontEnd` confinement contract
(spec 01 §7.3). The full deskew + solve + map update runs to completion inside
`ingest()`; the real-time budget is therefore visible as one stage timer
(`frontend.lio.ingest`) with the deskew/solve split in `FrontEndDiagnostics`.

Bounded internal buffers: the pre-init group hold (64 groups ≈ 6 s) and the
live-IMU deque (4096 samples ≈ 20 s); both drop oldest-first when exceeded.

## 13. Telemetry surface

Per-sweep scalars (always-on unless grouped; groups are key-prefix wildcards,
spec 09):

| key | group | meaning |
|---|---|---|
| `frontend/assoc/n_attempted`, `n_matched` | `assoc` | keypoints offered / correspondences found at the solved pose |
| `frontend/solver/gn_iters`, `dx_norm`, `chi` | `solver` | iterations, last update norm, final squared-residual sum |
| `frontend/lio/beta` | `lio` | gravity-regularizer weight actually used (−1 = block off) |
| `frontend/lio/accel_var` | `lio` | interval specific-force-magnitude variance (the β driver) |
| `frontend/lio/n_corr`, `deskew_span_t_ms` | `lio` | correspondence count; sweep time span |
| `frontend/map/voxels`, `points` | `map_health` | map occupancy counters |
| `frontend/state/vel_norm`, `bias_gyr_norm`, `bias_acc_norm` | — | live kinematic sanity |
| `frontend/obs` (vec6), `frontend/obs_min` | — | per-axis observability + its minimum |

Events (group `lio`): `init_backlog`, `init_done`, `gap`, `reject`, `reseed`;
plus the ungrouped `frontend/lio/error`. Heavy payloads: `body/scan` (the
deskewed sweep, `/meridian/cloud_body`) and `frontend/path_sample` (the
odometry pose stream sampled at `debug_path_sample_hz`, aggregated by the
wrapper into `/meridian/path`). All stamps are measurement time.

## 14. Parameters

`frontend.lio.*` (defaults from `config.hpp`; `validate()` enforces the stated
ranges; every change gets a `docs/OPTIMIZE.md` row):

| key | default | role |
|---|---|---|
| `voxel_size_m` | 1.0 | map voxel edge [m]; also the observability neighbourhood radius (> 0) |
| `max_points_per_voxel` | 20 | per-cell point cap; sets the min-spacing rule `v/sqrt(cap)` (≥ 1) |
| `max_range_m` | 100.0 | deskew range crop + map clip radius [m] (> 0; per-dataset, match L1 `det_range`) |
| `keypoint_voxel_factor` | 1.5 | keypoint downsample voxel = factor × `voxel_size_m` (> 0) |
| `min_keypoints` | 30 | sweep-usability floor and final-correspondence gate (≥ 1) |
| `icp_max_iterations` | 100 | GN iteration cap (≥ 1) |
| `convergence_eps` | 1e-5 | `‖δx‖` convergence threshold (> 0) |
| `max_corr_dist_m` | 0.5 | nearest-neighbour correspondence gate [m] (> 0) |
| `min_beta` | 200.0 | floor of the gravity-regularizer weight (> 0) |
| `max_expected_jerk` | 3.0 | uniform-jerk bound of the accel-magnitude filter [m/s³] (> 0) |
| `init_stationary_s` | 1.0 | standstill window for static init [s]; 0 = start immediately (≥ 0) |
| `max_gap_s` | 0.5 | sensor gap that arms a reseed [s] (> 0) |
| `reseed_cov_inflation` | 100.0 | covariance inflation on the post-reseed `AbsolutePrior` (≥ 1) |

`frontend.keyframe.*`: `dist_m` 1.0, `rot_deg` 10.0, `time_s` 1.0 — the §9
cadence. `frontend.debug_path_sample_hz` (copied from `debug.path_sample_hz` at
pipeline construction, default 30): sampling cadence of `frontend/path_sample`;
debug-only, gates no estimator computation.

## 15. Failure modes

| symptom | mechanism | response / instrument |
|---|---|---|
| estimate frozen while platform moves | sweeps skipped (sparse/failed registration) | `frontend/lio/reject` events; `frontend/assoc/n_matched` collapse |
| jump after a data hole | constant-velocity bridge landed off | `frontend/lio/gap` then either recovery or `reseed`; expect an `AbsolutePrior` packet |
| slow drift over long missions | frozen biases (no online estimation) | `frontend/state/bias_*_norm` constant by construction; see §16.1 |
| confident-but-wrong covariance in degenerate scenes | rung-0 data-term-only Σ | `frontend/obs_min` low while `chi` small — observability, not Σ, is the degeneracy signal |
| init never completes | platform not still for `init_stationary_s` | `frontend/lio/init_backlog` warning; held groups dropping |
| tilt error from a moving start | `init_stationary_s = 0` identity-attitude fallback | gravity regularizer fights the wrong anchor; do not use 0 in the field |

## 16. Known limitations

1. **No online bias estimation.** Biases are fixed at static-init values;
   the at-rest accel bias is only observable along gravity. Long-mission gyro
   drift accumulates as heading error. Benchmark first; an online bias state is
   the designed extension point (it would ride the tracker, not the packet).
2. **Rung-0 covariance** (§7 caveats): map correlation ignored, data term only,
   isotropic $\hat\sigma^2$. NEES calibration against GT is the follow-up.
3. **Interval-mean screw deskew** assumes the rate is constant across the
   sweep; intra-sweep oscillation faster than the sweep period aliases into
   the cloud. This is measured per dataset, not assumed away.
4. **IMU lever arm ignored** in the sample rotation to $F_e$ (§2) — valid for
   centimeter-scale offsets only.
5. **`apply_correction` shifts the state, not the map.** Large corrections
   force a reseed by design; revisit when loop closure ships.
6. **Camera/GNSS are dormant**: images pass through for downstream
   colourisation; GNSS gate verdicts are produced by L1 and consumed by no one.
7. **Observability neighbourhood is one voxel deep** — `neighborsWithin` radii
   beyond `voxel_size_m` silently truncate to the 27-cell envelope; the normal
   estimate is local by construction.
