# 05 — L3 Keyframe Factor-Graph Back-End (iSAM2)

> **Spec status:** normative implementation spec for layer **L3**. It implements
> `IBackEnd` (`01 §7.4`) and consumes the boundary types `KeyframePacket`
> (`01 §6`), `LoopConstraint` (`01 §7.6`), and `GnssFix` (`01 §4.4`). It emits the
> `GraphUpdate` (`01 §7.4`), the `corrected_trajectory()` (`01 §7.4`,
> `StampedPose`), and the refined `CalibrationSet` snapshot (`01 §5.3`). **This
> document does not redefine any boundary type** — a new field on a boundary type
> amends spec 01, per spec 01's own rule. Everything below is *internal* to
> `meridian_backend` except where it names a spec-01 type.
>
> **Where L3 sits.** The **L2 CT LIVO+GNSS front-end** fuses a single LiDAR, IMU,
> camera, and GNSS inside a sliding B-spline window and summarises each keyframe
> interval as **one** `KeyframePacket`. **L3 is the global, drift-free layer:** it
> stitches those packets, loop closures, and GNSS into a single iSAM2 factor graph
> in the `map` frame and broadcasts corrections to the live front-end (L2) and the
> nvblox map (L4). L3 does not estimate the trajectory inside a window — that is
> L2's job; L3 estimates the *globally consistent* keyframe poses, the GNSS-datum
> alignment, and the online extrinsics.
>
> **Engine:** GTSAM **4.2** `ISAM2` (Bayes tree, fluid relinearization, partial
> variable recovery); familiarity with the GTSAM API and the iSAM2 algorithm is
> assumed. This document specifies only how Meridian *uses* the engine; every
> GTSAM knob below is verified against the pinned GTSAM 4.2 source and the
> reference systems digested in **Appendix R** (which also holds the bibliography).
>
> **The single most important rule.** Between any two consecutive keyframes the
> back-end adds the L2 information **exactly once**: one relative `BetweenFactor`
> built from the packet's `RelativeBetween` constraint, with its marginal
> covariance. There is **no** companion absolute prior and **no** IMU factor on
> that edge. The IMU factor (`CombinedImuFactor`) appears **only** on the
> window-restart fallback (`constraint_kind == ImuPreintegration`), where it
> *replaces* the between-factor. The two are mutually exclusive *by construction*
> because `KeyframePacket::constraint_kind` is a single enum (`01 §6.4`). This
> L2→L3 hand-off rule (Appendix R.2) is the most load-bearing part of the
> back-end; §4 and §5 make it airtight.

---

## Table of contents

1. [Scope & responsibilities](#1-scope--responsibilities)
2. [Variables: the state the graph estimates](#2-variables-the-state-the-graph-estimates)
3. [Factors: the constraints](#3-factors-the-constraints)
4. [The L2→L3 normal edge (the common case)](#4-the-l2l3-normal-edge-the-common-case)
5. [The restart edge (the only IMU factor)](#5-the-restart-edge-the-only-imu-factor)
6. [GNSS factors & the GNSS-origin variable](#6-gnss-factors--the-gnss-origin-variable)
7. [PCM pre-filter (loop outlier rejection)](#7-pcm-pre-filter-loop-outlier-rejection)
8. [Robust kernels (loops + GNSS)](#8-robust-kernels-loops--gnss)
9. [iSAM2 update, relinearization & the correction outputs](#9-isam2-update-relinearization--the-correction-outputs)
10. [Online extrinsic refinement](#10-online-extrinsic-refinement)
11. [Marginalization for long sessions](#11-marginalization-for-long-sessions)
12. [Tangent-ordering adapter (Meridian ↔ GTSAM)](#12-tangent-ordering-adapter-meridian--gtsam)
13. [Marginal covariance for the loop pre-filter](#13-marginal-covariance-for-the-loop-pre-filter)
14. [Debug & introspection](#14-debug--introspection)
15. [Failure modes & handling](#15-failure-modes--handling)
16. [Configuration (BackendConfig)](#16-configuration-backendconfig)
17. [The back-end thread loop](#17-the-back-end-thread-loop)
18. [Module integration order](#18-module-integration-order)
19. [Cross-reference index](#19-cross-reference-index)

---

## 1. Scope & responsibilities

### 1.1 In scope

1. Own a single GTSAM `ISAM2` instance and the canonical **`map`-frame** estimate
   of every keyframe pose (and, where they cross the boundary, velocity/bias).
2. Translate each `KeyframePacket` into graph variables + exactly **one**
   odometry factor (§4/§5), respecting the no-double-counting contract (§3.1).
3. Fold in **switchable** constraints: loop closures (`LoopConstraint`, L5) and
   GNSS (`GnssFix`, L0), each guarded by a robust kernel (§8) and an outlier
   pre-filter (PCM for loops, §7).
4. Estimate the **GNSS-origin** alignment (`map ← ENU`) and, when enabled
   per-platform, the **online extrinsics** (off by default, §10) as graph variables,
   publishing refined calibration back to L2 as a versioned `CalibrationSet`
   snapshot (`01 §5.3`).
5. Run incremental `ISAM2::update` on a batched cadence decoupled from keyframe
   insertion (§9.2/§17); relinearize fluidly; marginalize transient inertial
   variables to keep the steady-state graph pose-only (§11).
6. Produce the `IBackEnd` outputs: `optimize()` → `GraphUpdate`,
   `corrected_trajectory()` → the `map`-frame `StampedPose` list,
   `refined_calibration()` → the snapshot.
7. Serve the latest pose marginal covariance to L5's place-recognition gate (§13).
8. Emit rich debug (`BackEndDiagnostics` + telemetry keys, §14): graph summary,
   per-axis observability inflation, loop/GNSS accept/reject, chi-square,
   relinearization counts, timing.

### 1.2 Out of scope (handled elsewhere)

- Front-end estimation, deskew, keyframe *emission policy*, and the construction
  of `RelativeBetween` / `ImuPreintegration` constraints — **L2**
  (`04_frontend_estimation.md`; boundary types fixed in `01 §6`).
- Loop-candidate *generation* and geometric verification — **L5** (`ILoopDetector`,
  `01 §7.6`; `07_loop_closure.md`). L3 only *consumes* a verified `LoopConstraint`
  and decides accept/reject via PCM (§7) + the robust kernel (§8).
- TSDF / colour / mesh and the clear-and-rebuild **re-integration** itself — **L4**
  (`IMapLayer::apply_graph_update`, `01 §7.5`; `06_mapping.md`). L4 is nvblox on
  the GPU, the only map backend. L3 *triggers* a region rebuild by emitting a
  `GraphUpdate`; it never touches voxels.
- Cloud storage — the `KeyframeStore` (`01 §7.5`). L3 forwards the packet's
  Shared-immutable `cloud_body` / `image` handles to the store and to L5; it does
  not copy or own clouds.

### 1.3 Position in the data flow

```
        KeyframePacket (L2)      LoopConstraint (L5)      GnssFix (L0)
               |                        |                      |
               v                        v                      v
        +--------------------------------------------------------------+
        |                  L3 back-end  (this spec)                     |
        |  vars:   X_i (Pose) ; [V_i,B_i on restart] ; E_s ; G         |
        |  factors: between | combinedImu(restart) | loop | gnss |     |
        |           damp(gauge anchor) | extrinsic-prior                |
        |  engine: GTSAM ISAM2  (Bayes tree, fluid relinearization)     |
        +--------------------------------------------------------------+
               |                        |                      |
               v                        v                      v
        GraphUpdate -> L4         corrected_trajectory     refined CalibrationSet
        (clear-and-rebuild)       (StampedPose, map)       snapshot -> L2 (01 §5.3)
```

Every keyframe carries exactly **one** `obs[6]` observability report (`01 §3.4`),
because there is exactly one LiDAR in the system; there is no per-sensor merge of
observability streams. (Multi-LiDAR would be a future extension behind the same
`KeyframePacket` interface; it is not designed here.)

> **Note on the `PoseCorrection` / `map→odom` concept.** Spec 00 §3 sketches a
> "`PoseCorrection` broadcast to L4/L2." In the normative type system (`01 §7.4`)
> that broadcast is realized as: (a) the `GraphUpdate` to L4, and (b) the refined
> `CalibrationSet` + the corrected trajectory the live front-end rebases against.
> The front-end keeps publishing smooth `odom`-frame `NavState` (`01 §7.3`); L3
> owns `map`, and §9.3 specifies how a consumer computes the `map→odom` transform
> from L3's outputs without a new boundary type.

### 1.4 Current L2 emission state (implementation status — read before building L3)

> **Implementation status, not design.** This records what the L2 CT front-end
> **actually emits today**, so an L3 implementer knows which factor paths can be
> exercised against the live system and which are spec-complete but dormant. The
> normative contract above is unchanged. Verify against
> `meridian_frontend/src/ct/ct_frontend.cpp::emitKeyframe` (and
> `src/ct/keyframe_finalizer.cpp`) before relying on it.
>
> The boundary type is real: `KeyframePacket` is defined in
> `meridian_common/include/meridian/common/keyframe_packet.hpp` and is emitted on a
> live `keyframe_sink` callback. `meridian_backend` does **not** exist yet — L3 is
> greenfield. The pipeline already forwards every packet through
> `MeridianPipeline::on_keyframe` (counts it, raises `frontend/keyframe`
> telemetry) and re-emits it on the pipeline's own outgoing `keyframe_sink`, which
> **nothing currently consumes** — that callback is the L3 insertion point.

What L2 populates per packet today:

| Field | Emitted today | Notes for L3 |
|-------|---------------|--------------|
| `constraint_kind` | `AbsolutePrior` (first keyframe **and** the first keyframe after a hard reseed — the chain break of `01 §6.4`); `RelativeBetween` (all others) | **`ImuPreintegration` is never emitted** — see GAP-A. A mid-stream `AbsolutePrior` in `Frame::Odom` opens a new relative segment: anchor it value-free like the first keyframe (§3), do not read it as an absolute measurement |
| `kinematics_included` | always `false` | `V`/`B` never cross the boundary today (§2.2 steady state) |
| `constraint_cov` | populated, `Covariance` form, **rotation-first** (`01 §6.1`) | normal edge: the **conservative sum** of the two endpoints' absolute pose marginals (upper-bounds the true relative marginal, which would subtract their cross-covariance); anchor keyframes ship the absolute pose marginal. Source is the **exact window-posterior marginal** (LiDAR+IMU+prior), LiDAR-only marginal as rank-deficiency fallback |
| `observability` | populated every sweep (`computeObservability` over the LiDAR hit set) | §4.3 inflation has real per-axis scores to act on |
| `cloud_body` | populated (downsampled scan deskewed to the body frame at `stamp`) | ready for L5 / the `KeyframeStore` |
| `calib_version` | populated | |
| `image`, `T_body_cam` | **not yet populated** (`image` stays null even when the visual path is active) | no live input for colourisation / visual loop verification yet |

**Async covariance, complete stream.** On the live path `constraint_cov` is filled
by an **async finalizer worker** (`KeyframeFinalizer`): the hot thread captures the
final solve round's build inputs, the worker rebuilds a bit-identical window
problem off-thread, recovers the exact window-posterior pose marginal, chains the
relative covariance, fills the packet, and forwards it to the sink. The worker is
FIFO over a bounded **blocking** queue: packets are never dropped or reordered, so
the L3 input stream is **complete and in order** — only delayed by up to one
keyframe interval. The deterministic/replay path computes the same marginal inline
so replay stays bit-exact against live.

**GAP-A — the restart edge (§5) is spec-complete but FE-dormant.** A data hole up
to five spline-knot intervals is *bridged*: the FE integrates IMU across it and
solves normally, so the chain stays contiguous and the packet is an ordinary
`RelativeBetween`. The FE therefore **never** emits
`constraint_kind == ImuPreintegration` and never populates `imu_summary`. Build §5
(`CombinedImuFactor`) to spec, but cover it with synthetic `ImuPreintegration`
packets in unit tests — it has no live trigger today.

**Hard reseed (the chain break; closes the old GAP-B).** Beyond the bridge
horizon the FE *reseeds*: it carries the anchor across the hole on a
constant-velocity prediction, restarts the window, and emits **no** keyframe for
the reseed sweep. There is no data inside the hole, so an `ImuPreintegration`
summary is not constructible; instead the **next** keyframe is emitted as
`AbsolutePrior` (window-posterior absolute marginal — the chain-break form of
`01 §6.4`) and the relative chain resumes off it. No relative edge ever spans the
predicted discontinuity. L3 anchors the new segment value-free (§3); the
trajectory becomes two relatively-constrained segments that only loops/GNSS can
stitch in `map` — the honest representation of an unobserved hole.

**Covariance-honesty note (bridged sweeps).** A sweep whose hole was IMU-bridged
carries more real uncertainty than its window-posterior marginal reflects (the
marginal sees only the factors that were present, not the missing LiDAR). The
effect is second-order and largely absorbed by the conservative-sum relative
covariance; a future FE improvement is to inflate `constraint_cov` on sweeps
flagged `frontend/sweep_gap_bridged`. Flagged here so the contract stays honest.

---

## 2. Variables: the state the graph estimates

All variables live in one GTSAM `Values`. Keys use `gtsam::Symbol(char, index)`
with a fixed character namespace so the elimination ordering and debug dumps are
legible (the `x/v/b/e` convention; cf. Appendix R.1).

| Symbol | Meaning | GTSAM type | When present |
|--------|---------|------------|--------------|
| `X(i)` | keyframe pose `T_map_body(i)` | `gtsam::Pose3` | every keyframe |
| `V(i)` | body velocity in `map` | `gtsam::Vector3` | **only** on restart-edge endpoints (§5) |
| `B(i)` | IMU bias `[b_a; b_g]` | `gtsam::imuBias::ConstantBias` | **only** on restart-edge endpoints (§5) |
| `E(s)` | extrinsic `T_body_sensor(s)` | `gtsam::Pose3` | for each sensor with online refinement (§10) |
| `G`    | GNSS-origin `T_map_enu` | `gtsam::Pose3` | only if GNSS used (§6) |

`i` is the `KeyframePacket::id` (`std::uint64_t`); `s` is the `Frame` enum value
(`01 Appendix A`).

### 2.1 Pose convention and frame

`X(i) = T_map_body(i)` maps a point in the estimation/body frame `F_e` at
keyframe `i` into `map`, consistent with `01 §2.2` ($p_A = T_{A\_B}\,p_B$). The
packet's `T_ref_body` is expressed in `ref_frame` (normally `Frame::Odom`,
`01 §6.2`). L3 owns the `map↔odom` relationship; it lifts the packet pose into
`map` for the *initial value* of `X(i)` (§4.2) but **never** turns the odom hint
into a factor.

### 2.2 Velocity & bias: do they cross the boundary?

**Steady state: NO** — fixed by the contract, not a local choice. Spec 01 §6.4:
in the normal path `kinematics_included = false`, velocity and biases stay
*inside* the front-end and their effect is folded into `constraint_cov`. L3
therefore creates **no** `V`/`B` variables for a normal keyframe; the keyframe is
a pure pose node. The packet's `v_ref`/`b_a`/`b_g` fields, when present, are
recorded for debug/seeding only.

**Exception — restart edges.** When `constraint_kind == ImuPreintegration`
(`kinematics_included == true`, `01 §6.4/§6.5`), L3 creates `V`/`B` on **both
endpoints of that one edge** and links them with a `CombinedImuFactor` built from
the packet's `ImuPreintegrationSummary` (§5). These inertial variables are *local
to the restart bridge*; they are marginalized at the next normal keyframe unless
`backend.keep_inertial` is set (default `false`). The steady-state graph stays
pose-only — the cheap, well-conditioned regime (Appendix R.5).

> **Grounding.** A FAST-LIO2-style filter keeps the full inertial state (pose,
> vel, biases, gravity, extrinsic) live in one filter state; Meridian's L2 carries
> the equivalent inside its window. The relative `BetweenFactor` already carries
> the vel/bias information through its marginal covariance (`01 §6.4`). Carrying
> `V`/`B` on every edge **and** a relative factor would re-inject the same IMU
> evidence twice — the double-count the hand-off contract forbids (Appendix R.2).

### 2.3 ID → key mapping

```cpp
namespace meridian::backend {
inline gtsam::Key keyX(std::uint64_t id){ return gtsam::Symbol('x', id); }
inline gtsam::Key keyV(std::uint64_t id){ return gtsam::Symbol('v', id); }
inline gtsam::Key keyB(std::uint64_t id){ return gtsam::Symbol('b', id); }
inline gtsam::Key keyE(Frame s)         { return gtsam::Symbol('e', std::uint64_t(s)); }
inline constexpr gtsam::Key keyG = gtsam::Symbol('g', 0);
}
```

The GTSAM symbol index *is* the `KeyframePacket::id`; no side table is required
for poses. A small `std::unordered_map<std::uint64_t, KfRecord>` holds
per-keyframe bookkeeping (stamp, cloud/image handles, observability, the stored
odom hint — §4).

---

## 3. Factors: the constraints

Every factor is a negative-log-likelihood term: the squared Mahalanobis residual
$\lVert h_i(X)\ominus z_i\rVert^2_{\Sigma_i}$ whose sum is the graph's MAP cost.
Meridian uses these and **only** these factor classes:

| Factor | GTSAM class | Connects | Source | Switchable | Robust |
|--------|-------------|----------|--------|-----------|--------|
| Gauge anchor | `LinearDampingFactor` | `X(first)` | bootstrap | no | no |
| Odometry (normal) | `BetweenFactor<Pose3>` | `X(rel_to_id), X(id)` | `RelativeBetween` | no | no |
| Odometry (restart) | `CombinedImuFactor` | `X,V,B(from), X,V,B(to)` | `ImuPreintegration` | no | no |
| Bias-walk (restart) | folded into `CombinedImuFactor` | `B(from), B(to)` | summary | no | no |
| Loop closure | `BetweenFactor<Pose3>` | `X(from_id), X(to_id)` | `LoopConstraint` | **yes** | **GNC** |
| GNSS position | `GnssFactor` (custom `NoiseModelFactor3`, fix-time interpolated) | `X(i), X(j), G` | `GnssFix` | **yes** | **Huber** |
| GNSS-origin prior | `PriorFactor<Pose3>` | `G` | first fix | no | weak |
| Extrinsic prior | `PriorFactor<Pose3>` | `E(s)` | calibration | no | tight (§10) |

The gauge anchor fixes the gauge: a relative-only graph has a 6-DoF null space, so
without it iSAM2 throws `IndeterminantLinearSystemException`. The anchor is a
`gtsam::LinearDampingFactor` on `X(first)`, **not** a `PriorFactor<Pose3>` toward
a fixed pose value: a hard prior with a tight σ pulls `X(first)` toward its
arbitrary bootstrap value with real information, biasing the global estimate and
fighting loop/GNSS corrections that should be free to translate/rotate the whole
trajectory rigidly. `LinearDampingFactor` adds an isotropic damping term `λ I₆` to
the first pose's block of the linearised Hessian — it removes the null space
without contributing a residual, so the MAP estimate is unbiased and the gauge
floats with global corrections. `λ = 1 / backend.anchor_sigma²` (default
`anchor_sigma = 1e-4`: strong but value-free). Added once, at the first keyframe,
**never** re-added. The first keyframe normally arrives with
`constraint_kind == AbsolutePrior` (`01 §6.4`); L3 maps that to this damping
anchor.

### 3.1 Why exactly one odometry factor per edge

The failure the hand-off contract prevents: an absolute marginal prior per
keyframe, a relative between-factor, **and** an IMU-preintegration factor — all
derived from the **same** measurements the L2 window already fused — would
triple-count the evidence. Summing their information shrinks covariances
unrealistically; the over-confident graph rejects loop/GNSS corrections and
collapses the marginal-covariance search radius, silently *missing loop closures*
(§13).

Meridian's contract, keyed off the single `KeyframePacket::constraint_kind` enum:

```cpp
switch (kf.constraint_kind) {
  case ConstraintKind::RelativeBetween: add_between_edge(kf);    break;  // §4
  case ConstraintKind::ImuPreintegration: add_restart_imu_edge(kf); break; // §5
  case ConstraintKind::AbsolutePrior:   add_anchor_or_absolute(kf); break; // §3 / §6
}
```

Because the source is a single enum, you *cannot* ship two odometry factors for
one edge. The information crosses L2→L3 exactly once, audited at one site.

### 3.2 The canonical squared-Mahalanobis / chi-square convention (binding)

**This clause is the single, system-wide definition of the squared-Mahalanobis
distance and its chi-square gate. Every consistency test, robust inlier threshold,
and gating quantile across the system — here and in the loop-closure spec — uses
*this* convention. Spec 07 references this clause rather than restating it; where any
doc defines a Mahalanobis gate it must reduce to the form below, or it is wrong.**

For a residual $r \in \mathbb{R}^n$ with **covariance** $\Sigma$ (not information),
the squared Mahalanobis distance is

$$
d^2 \;=\; r^\top\,\Sigma^{-1}\,r \;\in\;\mathbb{R}_{\ge 0},
$$

a single scalar. Under the Gaussian-residual assumption $d^2$ is distributed
$\chi^2_n$, so a two-sided acceptance gate at confidence $\alpha$ is

$$
d^2 \;\le\; \chi^2_{n,\alpha},
$$

where $\chi^2_{n,\alpha}$ is the **inverse CDF (quantile)** of the chi-square
distribution with $n$ degrees of freedom evaluated at $\alpha$ — i.e.
`chi2inv(α, n)`, *not* the density and *not* a per-axis sigma count. Conventions
fixed here, binding everywhere:

- **DoF $n$ is the residual dimension**: $n=6$ for a `Pose3` relative/loop residual
  (`SE(3)` tangent), $n=3$ for a `Point3` / GNSS-position residual. The quantile is
  taken at the matching $n$; a 6-DoF gate and a 3-DoF gate are different numbers.
- **The reference confidence is $\alpha = 0.99$**, giving $\chi^2_{6,0.99}\approx16.81$
  and $\chi^2_{3,0.99}\approx11.34$. The loop PCM gate (`backend.pcm_chi2_alpha`, §7.1)
  and the GNC inlier threshold both use this same $\alpha$ at their own $n$.
- **GNC inlier cost threshold is the *squared* gate**: `setInlierCostThresholds`
  takes $\bar c^2 = \chi^2_{n,\alpha}$ (the same quantile, already squared — it is a
  squared-distance threshold, never the un-squared $\sqrt{\chi^2}$). The PCM gate and
  the GNC threshold are therefore the *identical* number at a given $n$, so a loop the
  PCM front gate admits is judged on the same statistical scale by the robust kernel
  (§8) and the off-thread consolidation (§8.3). Keeping these two in lock-step is
  mandatory; a mismatch silently makes one stage stricter than the other.
- **$\Sigma$ must be SPD before inversion.** Compose covariances (e.g. the two loop
  covariances plus the odometry-chain covariance in a PCM cycle test, §7.1) by adding
  covariances in a common tangent, then guard the result PSD (clamp eigenvalues to
  $[\sigma_{\min}^2,\infty)$ as in FM-2) before forming $\Sigma^{-1}$. A near-singular
  composed $\Sigma$ otherwise yields a spuriously huge $d^2$ and rejects a valid loop.
- **Tangent ordering is irrelevant to $d^2$ but not to building $\Sigma$**: $d^2$ is
  a scalar invariant under a consistent reordering of $r$ and $\Sigma$, but $r$ and
  $\Sigma$ must share one ordering. Meridian-order covariances are reordered to match
  their residual exactly once at the boundary (§12); never mix orderings inside one
  $d^2$.

This is the convention assumed wherever this spec writes $d^2 = \epsilon^\top\Omega\epsilon$
(with $\Omega=\Sigma^{-1}$, §7.1), $\bar c^2 = \chi^2_{6,0.99}$ (§8), or a
chi-square health/innovation gate (§14, FM-6).

---

## 4. The L2→L3 normal edge (the common case)

### 4.1 Inputs (from `KeyframePacket`, `01 §6.1`)

- `id`, `stamp`
- `ref_frame`, `T_ref_body` — pose hint, init only
- `constraint_kind == RelativeBetween`, `rel_to_id`, `T_relto_this`,
  `constraint_cov` (a `GaussianBlock<6>`, **rotation-first** `[rx,ry,rz,tx,ty,tz]` —
  the GTSAM-boundary block, NOT a `PoseCov6`, `01 §6.1`)
- `observability : ObservabilityReport` — `frame`, `score[6]` in
  `[tx,ty,tz,rx,ry,rz]`, optional `eigvecs` (`01 §3.4`)
- `cloud_body`, `image`, `T_body_cam`, `calib_version`

### 4.2 Algorithm

```text
add_between_edge(kf):
  assert kf.rel_to_id == last_kf_id           # contiguity (FM-1)
  Xk = keyX(kf.id)

  # 1. Initial value: lift the odom hint into map.
  T_map_ref = current_map_to_ref(kf.ref_frame)   # last known map<-odom (§9.3)
  X_init    = T_map_ref * kf.T_ref_body           # Pose (Meridian), then ->Pose3
  new_values.insert(Xk, to_gtsam(X_init))

  # 2. The SINGLE odometry factor (the hand-off contract, §3.1).
  Sigma6_gt = inflate_by_observability(kf.constraint_cov, kf.observability)  # §4.3, GTSAM order
  noise     = noiseModel::Gaussian::Covariance(Sigma6_gt)
  new_graph.add( BetweenFactor<Pose3>( keyX(kf.rel_to_id), Xk,
                                       to_gtsam(kf.T_relto_this), noise ) )

  # 3. Bookkeeping: stamp, handles, observ, the odom hint (for §9.3).
  kf_record[kf.id] = { kf.stamp, kf.cloud_body, kf.image, kf.observability,
                       kf.ref_frame, kf.T_ref_body, kf.calib_version }
  store.put(kf.id, kf.cloud_body, kf.image, X_init)   # KeyframeStore (01 §7.5)
  loop_detector.add_keyframe(kf.id, kf.cloud_body, X_init)   # L5 (01 §7.6)

  last_kf_id = kf.id
```

`optimize()` (the `IBackEnd` method) is what actually runs `ISAM2::update`; the
back-end thread calls it on its own cadence (§9.2, §17). Contiguity
(`rel_to_id == last_kf_id`) is asserted: a gap means L2 dropped a keyframe, a
fatal contract violation (FM-1) — L3 does **not** fabricate a bridging factor.

The noise model is `Gaussian::Covariance(Σ_ij)` fed the front-end's marginal
covariance directly: we ship `Σ` (not an information matrix), tagged `Covariance`
by the `GaussianBlock` form (`01 §3.3`), so the conversion is explicit and
auditable at this one site.

### 4.3 Observability → noise inflation (the degeneracy contract)

The packet's `ObservabilityReport` (`01 §3.4`) carries per-axis scores
`score[k] ∈ [0,1]` (1 = fully observable) in a **named frame**
`observability.frame`, ordering `[tx,ty,tz,rx,ry,rz]`. L3 maps a low score to an
**inflated covariance** along that axis of the between-factor, so the optimiser
does not trust a direction the front-end could not constrain (e.g. translation
along a featureless corridor). This is the degeneracy signal spec 01 §3.4
promises would be defined for L3 — defined here. Never silently drop a factor —
inflate its covariance: rotate `Σ` into the degenerate eigenbasis, raise the
bad-axis variance, rotate back (Appendix R.7, Zhang et al.).

Per-axis multiplier:

$$
\lambda_k \;=\; 1 + (\rho_{\max}-1)\,(1 - s_k)^{\gamma}, \qquad \lambda_k \in [1,\rho_{\max}],
$$

with `ρ_max = backend.obs_inflation_max` (default `1e4`) and shape exponent
`γ = backend.obs_inflation_gamma` (default `2.0`). A fully observable axis
(`s_k=1`) gives `λ_k=1`; a fully degenerate axis (`s_k=0`) inflates that variance
by `ρ_max`.

**Frame handling.** The scores are expressed in `observability.frame`; the
covariance lives in the factor's body tangent. Build a 6×6 block-diagonal
rotation `R6` that maps the observability frame's axes to the factor frame, on
`constraint_cov`'s **rotation-first** axes `[rx,ry,rz,tx,ty,tz]` (`eigvecs`,
stored translation-first per `01 §3.4`, are permuted onto them via `to_rotfirst`).
If `eigvecs` is present (non-axis-aligned degeneracy, e.g. a tunnel at 30°), use
it directly as the rotation of the degenerate sub-space instead of an
axis-aligned `R6`:

$$
\Sigma'_{\text{meridian}} \;=\; R_6\,\Lambda^{1/2}\,(R_6^\top\, \Sigma_{\text{meridian}}\, R_6)\,\Lambda^{1/2}\,R_6^\top,
\quad \Lambda = \mathrm{diag}(\lambda_{tx},\lambda_{ty},\lambda_{tz},\lambda_{rx},\lambda_{ry},\lambda_{rz}).
$$

`constraint_cov` is already rotation-first (GTSAM order, `01 §6.1`), so the matrix
is **not** reordered here — only the translation-first `observability` score (and
any `eigvecs`) are permuted onto its axes. (`LoopConstraint.cov`, a
translation-first `PoseCov6`, still goes through `reorder_meridian_to_gtsam`, §12.)

```cpp
// Returns a 6x6 covariance in GTSAM order [rx,ry,rz, tx,ty,tz], ready for noiseModel.
// constraint_cov is ALREADY rotation-first (the GTSAM-boundary block, 01 §6.1), so the
// matrix is used as-is; only the translation-first observability inputs are permuted onto
// its axes. P maps a rotation-first axis index -> the matching o.score index.
Eigen::Matrix<double,6,6> inflate_by_observability(const GaussianBlock<6>& cov, const ObservabilityReport& o){
  static constexpr int P[6] = {3,4,5, 0,1,2};         // [rx,ry,rz,tx,ty,tz] <- score [tx,ty,tz,rx,ry,rz]
  Eigen::Matrix<double,6,6> Sig = cov.M;              // rotation-first; used as-is (no reorder)
  Eigen::Matrix<double,6,6> Lam_sqrt = Eigen::Matrix<double,6,6>::Identity();
  for (int k=0;k<6;++k){
    double s   = std::clamp(o.score[P[k]], 0.0, 1.0);
    double lam = 1.0 + (cfg.obs_inflation_max - 1.0)
                       * std::pow(1.0 - s, cfg.obs_inflation_gamma);
    Lam_sqrt(k,k) = std::sqrt(lam);
  }
  // Non-axis-aligned degeneracy: bring the eigenvector basis onto the rotation-first axes too.
  Eigen::Matrix<double,6,6> R6 = o.eigvecs ? to_rotfirst(*o.eigvecs)
                                           : rotate_tangent(o.frame, kFactorFrame); // I if same
  Sig = R6 * (Lam_sqrt * (R6.transpose() * Sig * R6) * Lam_sqrt) * R6.transpose();
  return Sig;                                         // already GTSAM order — no reorder
}
```

If a degenerate axis is detected (`s_k` below `backend.degenerate_thresh`, default
`0.05`) and `backend.degenerate_lock` is set, L3 hard-sets that axis's
`λ_k = ρ_max` regardless of the smooth law — a guard against near-zero eigenvalues
the smooth curve might under-inflate — and emits a degeneracy marker (§14).

---

## 5. The restart edge (the only IMU factor)

When the front-end restarts its sliding window and can summarise IMU across the
gap (`00 §7.4`, `01 §6.4`), it emits a packet with
`constraint_kind == ImuPreintegration` and a populated
`imu_summary : ImuPreintegrationSummary` (`01 §6.5`). L3 turns this into a single
`gtsam::CombinedImuFactor` — the **only** place in the entire system an IMU factor
enters the graph. (Currently FE-dormant — see §1.4 GAP-A; today's hard reseeds
break the chain with `AbsolutePrior` instead.)

### 5.1 The summary (already a boundary type — `01 §6.5`)

`ImuPreintegrationSummary` carries `delta_R / delta_v / delta_p`, the bias
linearization point (`bias_g_lin`, `bias_a_lin`), the first-order bias Jacobians
(`dR_dbg`, `dv_dbg`, `dv_dba`, `dp_dbg`, `dp_dba`), a 9-DoF `preint_cov`, and
`gravity_mag` — exactly what GTSAM's preintegration needs (Appendix R.3). L3
rebuilds a `PreintegratedCombinedMeasurements` (PIM) from these fields rather than
re-integrating raw IMU (which never crosses the boundary — `01 §6.5`).

```cpp
namespace meridian::backend {
gtsam::PreintegratedCombinedMeasurements
pim_from_summary(const ImuPreintegrationSummary& s, const BackendConfig::Imu& cfg);
}
```

`CombinedImuFactor` is preferred over the older `ImuFactor` + separate bias
between-factor because the bias random-walk is baked in (Appendix R.3).

### 5.2 Adding the edge

```text
add_restart_imu_edge(kf):
  i = kf.rel_to_id ; j = kf.id
  assert kf.kinematics_included and kf.imu_summary.has_value()   # contract (01 §6.4)

  # V,B appear ONLY here. Seed from the packet's inertial fields.
  ensure_value(keyV(i), to_gtsam(kf_record[i].v_ref_or_zero()))
  ensure_value(keyB(i), gtsam::imuBias::ConstantBias(kf.b_a, kf.b_g))   # if i had none
  ensure_value(keyV(j), to_gtsam(kf.v_ref))
  ensure_value(keyB(j), gtsam::imuBias::ConstantBias(kf.b_a, kf.b_g))
  ensure_value(keyX(j), lifted_init(kf))                                # as in §4.2 step 1

  # If V(i)/B(i) were freshly created (i had no inertial vars), pin them with a
  # loose prior so they are observable — else the new V/B DoF are unconstrained
  # and ISAM2 throws IndeterminantLinearSystem (Appendix R.5).
  if created_Vi: new_graph.add(PriorFactor<Vector3>(keyV(i), v_seed, loose_vel_noise))
  if created_Bi: new_graph.add(PriorFactor<ConstantBias>(keyB(i), b_seed, loose_bias_noise))

  pim = pim_from_summary(*kf.imu_summary, cfg.imu)
  new_graph.add( gtsam::CombinedImuFactor(keyX(i),keyV(i),
                                          keyX(j),keyV(j),
                                          keyB(i),keyB(j), pim) )
  # NO BetweenFactor on this edge — the CombinedImuFactor IS the odometry.
  mark_inertial_edge(i, j)     # so §11 can marginalize V,B when it returns to normal
  last_kf_id = kf.id
```

### 5.3 Bias / noise defaults (restart edges)

The `CombinedImuFactor` includes the between-bias random-walk term. Defaults
follow the FAST-LIO reference configuration (`IMU_Processing.hpp`,
`config/ouster64.yaml`); per-platform overrides flow in via `CalibrationSet`
(`01 §5.3`: `imu_acc_noise`, `imu_gyr_noise`, `imu_acc_bias_rw`,
`imu_gyr_bias_rw` → `PreintegrationCombinedParams`):

| Quantity | Default |
|----------|---------|
| accel noise `σ_a` | 0.1 m/s²/√Hz |
| gyro noise `σ_g` | 0.1 rad/s/√Hz |
| accel bias RW `σ_ba` | 1e-4 m/s³/√Hz |
| gyro bias RW `σ_bg` | 1e-4 rad/s²/√Hz |

The gravity vector for the PIM is ENU `(0,0,-9.81)`, consistent with the `map`
frame (gravity-aligned ENU, `00 §2.2`).

### 5.4 Returning to normal

The restart edge is a **bridge, not a mode**. The next keyframe carries a clean
`RelativeBetween` and is added as a normal `BetweenFactor` (§4). The bridge's
local `V`/`B` become candidates for marginalization at the next update (§11)
unless `backend.keep_inertial` is true — keeping the steady-state graph pose-only.

---

## 6. GNSS factors & the GNSS-origin variable

`GnssFix` (`01 §4.4`) carries WGS84 `lat/lon/alt`, an ENU `cov_enu` (3×3 m²), a
`fix` quality enum, and `num_sats`. It reaches L3 via
`IBackEnd::add_absolute(fix, nearest_kf_id)` (`01 §7.4`); the supplied keyframe id
is only a **hint** to locate the bracketing keyframe pair for the fix-time
interpolation (§6.3) — the fix is never bound directly to that keyframe's pose.
(Spec 01 also allows GNSS folded into an `AbsolutePrior` packet; L3 routes that
through the same machinery using `constraint_cov` as the position covariance.)

### 6.1 The origin variable `G`

GNSS arrives in a local-ENU tangent plane whose origin is the first valid fix.
The transform `G = T_map_enu` is a graph variable so the residual misalignment
between the SLAM `map` frame and the GNSS datum is *estimated*, not assumed. `G`
is seeded by a weak `PriorFactor<Pose3>` near identity (translation prior tight,
yaw prior loose — yaw between `map` and ENU is least observable until motion
accrues). Modelling the datum as a variable (rather than baking GNSS into a unary
world prior) makes the residual misalignment observable and correctable.

### 6.2 Datum initialization (the `G` seed)

`G` cannot be seeded from a single fix: a lone fix fixes translation but leaves
yaw entirely unobserved, and a wrong yaw bakes a fixed heading error into every
subsequent GNSS residual — the dominant long-run GNSS failure. L3 therefore
**defers** activating `G` until a spread of buffered fixes can determine the
alignment, then seeds `G` by a 4-DOF Umeyama fit gated on observability.

Buffer each accepted fix as a correspondence `(p_enu, X(i)·l)` — the projected ENU
position paired with the current `map`-frame antenna position at the fix-time
interpolated pose (§6.3). Datum init runs only when **all** pre-gates pass:

- **Min-baseline pre-gate.** Cumulative travelled baseline since the first buffered
  fix exceeds `backend.gnss_min_baseline` (default `5 m`). A short baseline cannot
  separate yaw from translation.
- **Velocity-excitation pre-gate.** The buffered fixes contain genuine planar
  motion, not jitter-in-place: the span of the de-meaned ENU positions along the
  dominant horizontal axis exceeds `backend.gnss_min_excitation` (default `3 m`),
  and the platform speed exceeded `backend.gnss_min_speed` (default `0.5 m/s`) for
  at least `backend.gnss_min_moving_fixes` (default `5`) fixes. This rejects a
  datum fit from a stationary rig whose apparent "baseline" is multipath wander.

When the pre-gates pass, fit the 4-DOF similarity (yaw + 3-translation; roll/pitch
fixed to gravity, scale fixed to 1) between the `map` antenna track and the ENU
track by **Umeyama** alignment restricted to the horizontal yaw rotation. The fit
also yields its information: form the Gauss-Newton Hessian of the yaw parameter
from the same correspondences, and accept the datum only if its **yaw
uncertainty** $\sigma_{\text{yaw}} = 1/\sqrt{H_{\text{yaw}}}$ is below
`backend.gnss_datum_yaw_sigma_max` (default `5°`). This Hessian gate makes
"enough baseline" precise: a near-collinear track (driving in a straight line)
leaves yaw weakly observable and inflates $\sigma_{\text{yaw}}$ even when the
baseline is long, so the fit is rejected and buffering continues. On acceptance,
seed `G` from the Umeyama transform and add the weak `PriorFactor<Pose3>` (§6.1)
with the translation block tight and the yaw block set from the fitted
$\sigma_{\text{yaw}}$; the datum is then **locked** (telemetry `datum_locked`) and
subsequent fixes flow as factors (§6.3). Until acceptance, buffered fixes do
**not** enter the graph.

If the pre-gates never pass (e.g. a purely stationary GNSS session), `G` stays
unlocked and GNSS contributes nothing — loop closure and odometry carry the global
estimate, the correct degenerate behaviour.

### 6.3 The GNSS factor (bound to the interpolated pose at fix time)

A `GnssFix` is stamped at its own time, which almost never coincides with a
keyframe. Binding the fix to the *nearest* keyframe pose injects that keyframe's
residual motion (up to a keyframe interval of travel) into the GNSS residual as a
silent error. L3 instead binds the fix to the pose **interpolated to the fix
timestamp** between the two bracketing keyframes.

`add_absolute(fix, hint_kf_id)` locates the keyframe pair `(i, j)` with
`stamp(i) ≤ fix.stamp ≤ stamp(j)` (`hint_kf_id` seeds the search). It first
geodetic-projects `fix` into the ENU datum (the L3 adapter owns the datum,
`01 §4.4`), yielding `p_enu ∈ ℝ³` with covariance `cov_enu`. With interpolation
weight
$\beta = (\texttt{fix.stamp} - \texttt{stamp}(i)) / (\texttt{stamp}(j) - \texttt{stamp}(i)) \in [0,1]$,
the antenna position at fix time uses the constant-velocity body interpolant
$\hat X(\beta) = X(i)\cdot \mathrm{Exp}\!\bigl(\beta\,\mathrm{Log}(X(i)^{-1}X(j))\bigr)$
and antenna lever arm `l = T_body_gnss.t` (from `CalibrationSet`):

$$
r_{\text{gnss}} \;=\; G^{-1}\bigl(\hat X(\beta)\cdot l\bigr) \;-\; p_{\text{enu}}.
$$

This is a custom `NoiseModelFactor3<Pose3, Pose3, Pose3>` over `X(i)`, `X(j)`, and
`G` (a lever-arm + datum + on-edge-interpolation variant of GTSAM's `GPSFactor`);
`β` is a fixed factor parameter, not a variable. Its analytic Jacobians chain the
standard `Pose3`-action Jacobians with the interpolation Jacobians
(`gtsam::interpolate(Pose3,Pose3,t,H1,H2)` returns the two pose blocks; chain
`transformFrom`/`transformTo` and `G^{-1}` onto them). When a fix coincides with a
keyframe ($\beta \in \{0,1\}$, or no successor keyframe yet) the factor
degenerates to a `NoiseModelFactor2<Pose3,Pose3>` over the single endpoint and `G`.

```cpp
// β-interpolated antenna position; H1,H2 are the two bracketing-pose blocks, H3 the G block.
class GnssFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {
  gtsam::Point3 lever_, meas_enu_;
  double beta_;                                   // fix-time interpolation weight in [0,1]
 public:
  gtsam::Vector evaluateError(const gtsam::Pose3& Xi, const gtsam::Pose3& Xj, const gtsam::Pose3& G,
                              gtsam::OptionalMatrixType H1,
                              gtsam::OptionalMatrixType H2,
                              gtsam::OptionalMatrixType H3) const override {
    gtsam::Matrix6 Hxi, Hxj;  gtsam::Matrix36 Ha, Hg_to;  gtsam::Matrix33 Ham;
    gtsam::Pose3  Xb      = gtsam::interpolate(Xi, Xj, beta_, (H1||H2)?&Hxi:nullptr,
                                                              (H1||H2)?&Hxj:nullptr);
    gtsam::Point3 ant_map = Xb.transformFrom(lever_, (H1||H2)?&Ha:nullptr);
    gtsam::Point3 ant_enu = G.transformTo(ant_map, H3?&Hg_to:nullptr, (H1||H2)?&Ham:nullptr);
    if (H1) *H1 = Ham * Ha * Hxi;   // chain through the i-pose interpolation block
    if (H2) *H2 = Ham * Ha * Hxj;   // and the j-pose block
    if (H3) *H3 = Hg_to;            // d(ant_enu)/dG
    return ant_enu - meas_enu_;
  }
};
```

### 6.4 Switchable, gated, decimated & robust

GNSS is **switchable** (`backend.gnss_enabled`, `[hot]`) and **robustified**:
multipath / NLOS produce gross outliers. GNSS uses a **Huber** kernel inside iSAM2
by default — convex, therefore safe inside the incremental IRLS solve; the heavier
redescending machinery is reserved for the loop sub-graph's off-thread
consolidation (§8.3). Gating and decimation before a fix becomes a factor:

- **Quality gate.** Drop a fix with `trace(cov_enu) > backend.gnss_max_cov`, or
  `fix == None`. The `fix` enum and `cov_enu` together let the kernel weight an
  `SPP` fix far below an `RTK_Fixed` one (`01 §4.4`).
- **Confidence gate (skip-if-confident).** When `backend.gnss_skip_if_confident` is
  set (default `true`), skip a fix whose ENU position covariance is no better than
  the back-end's own marginal at that pose — drop when
  $\mathrm{trace}(\texttt{cov\_enu}) \ge \kappa^2\cdot\mathrm{trace}(\Sigma^{\text{pos}}_{X(i)})$
  with $\kappa = $ `backend.gnss_skip_confidence_k` (default `1.0`). A fix less
  certain than the existing estimate cannot improve it and only adds churn; the
  marginal is the cheap §13 query. GNSS contributes where it helps (re-anchoring
  after drift) and stays silent where the SLAM estimate is tighter.
- **Min-spacing decimation.** Admit at most one fix per `backend.gnss_min_spacing`
  (default `1.0 m`) of travelled baseline. Raw 5–10 Hz GNSS over slow motion
  produces near-duplicate, strongly correlated fixes whose independent-noise
  assumption is false; distance (not time) decimation keeps factors statistically
  independent, bounds factor growth, and means a stationary platform contributes a
  single fix.
- **Deferred datum.** Until `G` is locked (§6.2) fixes are buffered, not added.

### 6.5 Yaw observability & origin locking

Until the platform moves enough, `G`'s yaw is weakly observable; the datum-init
Hessian gate (§6.2) is the primary defence. Additionally:

- **Heading lock.** If `backend.gnss_lock_yaw` is set (e.g. dual-antenna heading
  available), pin `G`'s roll/pitch to gravity and fix yaw by a tight prior,
  bypassing the Umeyama yaw fit.
- Otherwise the deferred, Hessian-gated datum activation (§6.2) is the safeguard.

### 6.6 Drift redistribution on GNSS re-acquisition (designed; deferred)

> **Status: designed-but-deferred (post-MVP).** Until implemented, a re-acquired
> GNSS stream re-enters as ordinary §6.4 factors and iSAM2 absorbs the correction
> through relinearisation — correct, but concentrated near the re-acquisition
> keyframe rather than spread along the drifted span.

When GNSS is lost (tunnel, urban canyon) and re-acquired, the first strong
post-outage fix implies a large `map`↔ENU mismatch that, applied as a single
factor, snaps the latest poses while leaving the intervening trajectory bent.
**Drift redistribution** instead spreads the accumulated error along the no-GNSS
span. Design: on re-acquisition (`fix ≥ backend.gnss_reacq_fix` for
`backend.gnss_reacq_persist` consecutive accepted fixes), L3 computes the closing
error between the re-acquired ENU position and the predicted antenna position and
distributes it as a set of **soft `BetweenFactor<Pose3>` adjustments** across the
keyframes spanning the outage, weighted by inter-keyframe arc length (not a hard
re-warp), so iSAM2 still balances them against odometry and any loops crossing the
span. Parameters (`gnss_reacq_fix`, `gnss_reacq_persist`,
`gnss_redistribute_span_max`) are in §16; `backend.gnss_redistribute = false` (the
MVP default) reproduces today's single-re-entry-factor behaviour.

---

## 7. PCM pre-filter (loop outlier rejection)

Loop closures are the highest-stakes factors: one wrong loop welds two unrelated
places and ruins the map. L5 hands L3 a verified `LoopConstraint` (`01 §7.6`:
`from_id`, `to_id`, `T_from_to`, `cov`, `fitness`). L3 gates every loop through
**Pairwise Consistent Measurement (PCM)** maximisation *before* it enters the
graph, then a committed convex robust kernel inside the graph as a second line of
defence, with optional off-thread batch-GNC consolidation as the heavy safety net
(§8).

### 7.1 Pairwise consistency

Two loops $l_{pq}=(X_p\!\to\!X_q,\,Z_{pq})$ and $l_{rs}=(X_r\!\to\!X_s,\,Z_{rs})$
are *pairwise consistent* if the cycle they form with the (trusted) odometry chain
closes within noise. Using current estimates $\hat X_\bullet$:

$$
\epsilon_{pq,rs} = \mathrm{Log}\!\Bigl(Z_{pq}^{-1}\,\hat X_p^{-1}\hat X_r\,Z_{rs}\,\hat X_s^{-1}\hat X_q\Bigr),
\qquad
d^2_{pq,rs} = \epsilon^\top\,\Omega\,\epsilon,
$$

with $\Omega = \Sigma^{-1}$ the combined information of the two loop covariances
plus the odometry-chain covariance, composed and PSD-guarded per the §3.2
convention (`cov.M` are `PoseCov6` in Meridian order; combine as covariances, then
invert). The test $d^2 \le \chi^2_{6,\alpha}$ uses the canonical form of **§3.2**
with $\alpha = $ `backend.pcm_chi2_alpha` (default 0.99 → $\chi^2_{6,0.99}\approx16.81$).

### 7.2 Maximum-clique consistent set

Build a consistency graph: nodes = pending loops, edge = pairwise consistent. Only
the members of the **maximum clique** (largest mutually consistent subset) are
admitted to iSAM2. Maintain the graph incrementally; run a bounded max-clique
(exact Bron–Kerbosch up to `backend.pcm_max_nodes`, default 64; greedy fallback
above the cap, FM-4).

```text
on LoopConstraint lc (from ILoopDetector::detect, 01 §7.6):
  if lc.fitness < cfg.loop_min_fitness: reject("low fitness"); emit_debug; return
  pending.push(lc)
  for q in pending: update_consistency_edge(lc, q)            # §7.1
  clique = max_clique(consistency_graph, cap=cfg.pcm_max_nodes)
  for lc* in clique \ in_graph:
     Sig_gt = reorder_meridian_to_gtsam(lc*.cov.M)               # §12
     noise  = noiseModel::Robust( committed_kernel(), Gaussian::Covariance(Sig_gt) )  # §8.1 (Huber)
     fid    = new_graph.add( BetweenFactor<Pose3>(keyX(lc*.from_id),
                                                  keyX(lc*.to_id),
                                                  to_gtsam(lc*.T_from_to), noise) )
     loop_factor_index[lc*] = fid; mark_in_graph(lc*)
  for lc# in in_graph \ clique:                               # newer clique evicted it (rare)
     schedule_removal(loop_factor_index[lc#])                 # §9.5
```

This maps to `IBackEnd::add_loop_constraint` (`01 §7.4`): the method enqueues the
constraint; the back-end thread runs the PCM step on the next iteration.

> **Why PCM *and* a robust kernel.** PCM is a *combinatorial set-level* pre-filter
> (which loops are mutually consistent); the committed convex kernel (§8.1) is a
> *continuous factor-level* robustifier. PCM stops a wrong loop from ever
> distorting the linearisation point; the kernel absorbs residual mis-weighting;
> the off-thread batch-GNC consolidation (§8.3) re-judges the admitted set with
> the full redescending machinery without touching the live solve.

---

## 8. Robust kernels (loops + GNSS)

The robustness of the back-end is layered in **series**: PCM pre-admission (§7)
picks a mutually consistent loop set; a **committed convex M-estimator** in the
incremental path is the per-factor robustifier; an **off-thread batch-GNC
consolidation** (§8.3, optional) is the heavy safety net that re-judges the loop
sub-graph without destabilising the live `ISAM2`. GNSS uses the same convex Huber
kernel in the incremental path (§6.4).

### 8.1 The incremental loop kernel (default: committed Huber)

A loop factor admitted by PCM enters iSAM2 wrapped in a **committed convex
M-estimator** — **Huber** by default (`backend.robust_kernel = "huber"`, tuning
`backend.loop_huber_k`, default `1.345`). Huber is convex, so its IRLS reweighting
is well-behaved inside a single-shot `ISAM2::update`; the loop is *committed* —
its kernel is fixed at admission, not annealed step-to-step. PCM has already
removed the mutually inconsistent loops upstream, so the incremental kernel only
has to absorb a borderline residual, not perform global outlier rejection. The
kernel is selectable to Cauchy / Geman–McClure / Truncated-Least-Squares via
`backend.robust_kernel` for experiments, but only Huber is a *known-safe*
incremental choice; the redescending options are non-convex and should be paired
with the batch consolidation of §8.3.

The inlier cost threshold $\bar c^2$ for any kernel is set from the chi-square
quantile for the factor dimension (`setInlierCostThresholds` — for 6-DoF at 99%,
$\bar c^2 = \chi^2_{6,0.99}$; the canonical convention of §3.2). Odometry
between-factors, restart IMU factors, and the gauge anchor are declared *known
inliers* and are never reweighted.

### 8.2 Amortised in-graph GNC (experimental)

Full Graduated Non-Convexity anneals a control parameter `μ` from a near-convex
surrogate toward a redescending robust cost, recovering per-factor weights
`w ∈ [0,1]` via the closed-form GM/TLS weight update (Appendix R.4). GTSAM's
`GncOptimizer` re-runs the whole schedule per solve — **batch, not incremental**;
it cannot run inside a single-shot `ISAM2::update`. Meridian *can* amortise the
schedule across iSAM2 steps, but this path is **experimental and off by default**
(`backend.gnc_enabled = false`): annealing `μ` re-linearises a growing,
deliberately non-convex region of the Bayes tree every step, and a mid-anneal
weight can transiently destabilise the live estimate. When explicitly enabled:

- A new loop factor enters with a **conservative** kernel (high `μ`, near-convex).
- Over the next `backend.gnc_anneal_steps` (default 5) iSAM2 updates, step `μ` on
  the schedule (`muStep`, direction per loss type — Appendix R.4), recompute each
  factor's weight from its current whitened residual, and re-insert the factor with
  the updated robust noise (remove-by-index + re-add, §9.5).
- A factor whose converged weight `w < backend.gnc_reject_w` (default 0.1) is
  **removed**; if it is a loop, it returns to the PCM pending buffer marked
  "GNC-rejected" so it is not retried immediately (FM-6).

The committed convex kernel (§8.1) plus PCM (§7) already deliver robust loop
handling on the live thread; the off-thread batch consolidation (§8.3) captures
the redescending benefit more safely, which is why this stays off.

### 8.3 Batch-GNC consolidation (off-thread)

The heavy GNC machinery runs where it is statistically correct — as a **batch**
optimisation — but off the live solve. Periodically (every
`backend.gnc_consolidate_interval` admitted loops, or on request) a worker
extracts the **loop sub-graph** — the loop `BetweenFactor`s plus the odometry
chain spanning their endpoints, with the rest of the trajectory held as fixed
priors at the current estimate — and runs a full GTSAM `GncOptimizer` (TLS/GM,
`setInlierCostThresholds(barc2 = χ²₆,₀.₉₉)` per §3.2, `setKnownInliers(odometry)`)
to convergence on a **copy**. The hardened verdict per loop: factors GNC drives to
`w < backend.gnc_reject_w` are scheduled for removal from the live graph (§9.5)
and returned to the PCM pending buffer marked "GNC-rejected" (FM-6); survivors
keep their committed kernel. The consolidation mutates only the live graph's
*factor membership* (a small remove-set applied on the back-end thread under
§9.5), never the live linearisation point, so it cannot inject the churn §8.2
warns about. It runs on the back-end worker, not the odometry path (FM-7).

Sünderhauf-style in-graph **switch variables are not used**: one real switch
variable per loop grows the Bayes tree and makes that region non-convex (more
relinearization churn). The PCM + committed-kernel + off-thread-batch-GNC layering
gets the outlier rejection without the extra variable.

---

## 9. iSAM2 update, relinearization & the correction outputs

### 9.1 ISAM2 configuration

```cpp
gtsam::ISAM2Params p;

// Per-symbol-type relinearize thresholds (Appendix R.1): a Pose3's rad/m
// DoF need different thresholds than a velocity or a bias.
gtsam::FastMap<char, gtsam::Vector> th;
th['x'] = (gtsam::Vector(6) << 0.05,0.05,0.05, 0.10,0.10,0.10).finished(); // rot rad, trans m
th['v'] = gtsam::Vector3::Constant(0.10);                                  // m/s
th['b'] = (gtsam::Vector(6) << 1e-3,1e-3,1e-3, 1e-4,1e-4,1e-4).finished(); // ba, bg
th['e'] = gtsam::Vector6::Constant(1e-3);                                  // extrinsic
th['g'] = (gtsam::Vector(6) << 1e-3,1e-3,1e-3, 5e-3,5e-3,5e-3).finished(); // GNSS origin (eager)
p.relinearizeThreshold   = th;
p.relinearizeSkip        = cfg.isam2_relinearize_skip;     // 1 (LIO back-ends want every-step)
p.optimizationParams     = gtsam::ISAM2DoglegParams();     // trust-region; robust to bad init
p.cacheLinearizedFactors = true;
p.evaluateNonlinearError = true;                           // errorBefore/After for health (§14)
p.findUnusedFactorSlots  = true;                           // we churn loop/GNSS factors (§9.5)
p.enableDetailedResults  = true;                           // per-variable detail for debug (§14)
p.factorization          = cfg.isam2_use_qr ? gtsam::ISAM2Params::QR
                                            : gtsam::ISAM2Params::CHOLESKY;
isam2_ = std::make_unique<gtsam::ISAM2>(p);
```

Rationale (deltas vs GTSAM defaults tabulated in Appendix R.1): **Dogleg** rather
than Gauss-Newton because visual + GNSS factors can sit far from their
linearisation point at init and a trust region bounds the step (also bounds
loop-closure relinearization spikes, FM-7). **`relinearizeSkip = 1`** so every
update checks deltas. **`findUnusedFactorSlots = true`** keeps the factor index
compact under loop/GNSS churn. **`evaluateNonlinearError`** feeds the chi-square
health alarm. `factorization` defaults to **CHOLESKY**; observability inflation
(§4.3) keeps degenerate axes from going rank-deficient, and if a corridor still
trips `IndeterminantLinearSystemException`, FM-3 flips to QR and retries. The `g`
threshold is tighter so the GNSS origin relinearizes eagerly while settling.

### 9.2 The update step (`IBackEnd::optimize`)

```text
optimize() -> GraphUpdate:
  # new_graph/new_values/remove_indices already hold the BATCH accumulated since the
  # last optimize() — every KEYFRAME/LOOP/GNSS item drained from the queue this cadence
  # tick (§17). The single ISAM2::update folds the whole batch at once.
  result        = isam2_.update(new_graph, new_values, remove_indices)
  loop_admitted = batch_has_loop()                        # a loop factor was staged this batch (§17)
  extra_iters   = loop_admitted ? cfg.extra_iters_loop : cfg.extra_iters_normal
  for k in 1..extra_iters: isam2_.update()                # extra Dogleg passes, only when warranted
  estimate = isam2_.calculateEstimate()                    # full only when L4 needs it (§9.3)
  gu       = build_graph_update(estimate, result)          # §9.3
  log_graph_summary(result); publish_debug()               # §14
  new_graph.clear(); new_values.clear(); remove_indices.clear()
  return gu
```

iSAM2 re-eliminates only the part of the Bayes tree touched by the new factors and
any variable whose linearisation point moved past its `relinearizeThreshold`.
Steady-state per-batch cost is near-constant; a loop touches the cycle's variables
(a larger sub-tree) but not the whole graph.

**Cadence is decoupled from keyframe insertion.** `add_keyframe`,
`add_loop_constraint`, and `add_absolute` only *stage* graph mutations into
`new_graph`/`new_values`; they do not call `ISAM2::update`. The back-end thread
(§17) calls `optimize()` on its own cadence — at most once per
`backend.optimize_interval_ms` (default `100 ms`) — so a burst of keyframes (or a
keyframe arriving with a loop and a GNSS fix) collapses into a **single** batched
`ISAM2::update`. This bounds back-end load independent of L2's keyframe rate and
amortises relinearisation over a batch. A loop closure or a GNSS datum lock forces
an immediate `optimize()` regardless of the timer. Queue depth and per-batch size
are published as telemetry (`backend/queue_depth`, `backend/optimize_lag`, §14).

**Event-conditional extra iterations.** iSAM2 does only **one** Dogleg step per
`update`; a large rigid correction needs extra passes, a steady-state batch does
not. `backend.extra_iters_normal` (default `0`) versus `backend.extra_iters_loop`
(default `4`; range 4–5) when the batch admitted a loop factor.

In the hot path, publishing the latest pose only needs
`calculateEstimate<Pose3>(latestKey)` (cheap partial back-substitution); the full
`calculateEstimate()` is reserved for when L4 must re-integrate after a loop moved
many poses.

### 9.3 Building `GraphUpdate` and the implicit `map→odom` correction

`GraphUpdate` (`01 §7.4`) is the L3→L4 currency:

```cpp
struct GraphUpdate {            // 01 §7.4 — do not redefine; shown for reference
  struct Moved { std::uint64_t id; Pose new_T_map_body; std::optional<PoseCov6> cov; };
  std::vector<Moved> moved;     // keyframes whose pose changed materially
  bool loop_closed = false;     // a loop just snapped (large rigid correction)
};
```

`build_graph_update` compares each affected keyframe's new estimate to its
last-emitted pose and includes it in `moved` if it shifted by more than
`backend.reintegrate_thresh` (translation) or the rotational analogue. `cov` is
filled from the iSAM2 marginal when `backend.emit_moved_cov` is set (moderately
expensive). `loop_closed` is set when this update admitted a loop factor. L4
consumes `moved` to clear-and-rebuild the affected nvblox TSDF region from the
`KeyframeStore` clouds at the corrected poses (`01 §7.5`).

**The `map→odom` correction** (spec 00 §3's "`PoseCorrection`") is recovered by
any consumer from L3's outputs without a new boundary type. For the latest
keyframe `n`, with the stored odom hint `T_odom_body(n) = kf_record[n].T_ref_body`
and the optimised `X(n) = T_map_body(n)`:

$$
T_{\text{map}\,\text{odom}} \;=\; X(n)\cdot \bigl(T_{\text{odom}\,\text{body}}(n)\bigr)^{-1}.
$$

The live front-end stays smooth in `odom` (`01 §7.3`); the global correction is
this transform, refreshed each keyframe (`current_map_to_ref` in §4.2 returns
exactly this, cached). The refined extrinsics travel separately as the versioned
`CalibrationSet` snapshot via `refined_calibration()` (§10, `01 §5.3`).

### 9.4 `corrected_trajectory()`

Returns the `map`-frame `StampedPose` list (`01 §7.4`, `01 Appendix A`:
`{stamp, kf_id, T_map_body}`) for every keyframe still represented in the
estimate. Used by L6 / evaluation / TF publication in the wrapper.

### 9.5 Factor removal

iSAM2 removes factors by internal index (`ISAM2::update(..., removeFactorIndices)`).
With `findUnusedFactorSlots = true` (§9.1) the freed slots are reused so the index
stays compact under churn. L3 tracks `factor → internal index` for every removable
factor (loops, GNSS) so PCM eviction (§7.2) and GNC rejection (§8) can pull a
factor cleanly. **Odometry between-factors, restart IMU factors, and the gauge
anchor are never removed.**

---

## 10. Online extrinsic refinement

Online extrinsic refinement is **off by default** (`backend.extrinsic_refine = false`)
and enabled per-platform. L3 is the **authority** on refined extrinsics
(`01 §5.3`): when enabled it holds them as graph variables and publishes refined
values to L2.

The default is off because an extrinsic is **weakly observable without
excitation**: the platform must both rotate and translate to separate `E(s)` from
the pose, and an extrinsic left free under stationary or pure-translation motion
is a canonical indeterminate-system cause. Defaulting it on would add a near-null
DoF to every graph and slowly bias a well-calibrated rig away from a trustworthy
offline calibration. Turn it on per-platform only when **both** hold: (a) the
offline calibration is *trusted but suspected to drift* (e.g. mounts that flex
with temperature), and (b) the mission is expected to deliver the rotational
**and** translational excitation the §10.2 gate requires. The per-sensor
`Extrinsic::refine_online` flag (`01 §5.2`) selects *which* sensors are refined
once the master switch is on.

### 10.1 Model

Each refinable sensor `s` gets an `E(s) = T_body_sensor(s)` `Pose3` variable,
seeded by the offline calibration with a tight `PriorFactor<Pose3>` (sigma
`backend.extrinsic_prior_sigma`, default 1e-3 m / 1e-3 rad — derived from
`Extrinsic::prior_cov` when available). When refinement is active the prior is
*loosened* to `extrinsic_refine_sigma` (default 1e-2) and the relevant sensor
factors are made to depend on `E(s)` (a GNSS lever arm can likewise be promoted
from a constant to `E(gnss).translation()`).

L3 publishes updated values as a versioned `CalibrationSet` snapshot that the
front-end picks up at keyframe boundaries via `IFrontEnd::set_calibration`. The
version counter lets L2 detect "calibration changed, reset linearization"
(`01 §5.3`). The snapshot crosses L3→L2 as Shared-immutable — no data race.

### 10.2 Observability & safeguards

- **Excitation gate.** Update/relinearize `E(s)` only once cumulative rotation and
  translation since enabling exceed `extrinsic_excite_rot` / `extrinsic_excite_trans`;
  before that the tight prior dominates and `E` stays pinned. (FAST-LIO2 keeps the
  LiDAR-IMU extrinsic in-state but pinned by default for the same reason —
  `extrinsic_est_en`.)
- **Convergence freeze.** Once `E(s)`'s marginal covariance (via
  `ISAM2::marginalCovariance`, §13) falls below `extrinsic_freeze_cov`, re-tighten
  its prior to the converged value, stop treating it as free (removing it from
  future relinearization), and publish the frozen value in the next calibration
  snapshot.
- **Sanity clamp.** If `E(s)` strays from the offline prior by more than
  `extrinsic_max_dev`, reject the update, re-pin to the prior, and warn (FM-5).

---

## 11. Marginalization for long sessions

Meridian's bounded-graph strategy (Appendix R.5): **keep the full `ISAM2` pose
graph** — poses are cheap (one `Pose3` per ~1 m; multi-km missions are tens of
thousands of poses, well within iSAM2) and loop closures must be able to attach to
old keyframes, which a fixed-lag smoother would have marginalized away — and
**marginalize only the transient inertial variables** (`V`/`B`) introduced by the
restart bridge (§5).

### 11.1 What stays active

- The full keyframe **pose graph** stays in iSAM2 (any past keyframe can still
  receive a loop closure). The anchor keyframe, the GNSS-origin `G`, and any
  non-frozen extrinsic `E(s)` are likewise never marginalized.
- Transient `V`/`B` from a restart bridge (§5) are marginalized once the next
  normal keyframe is added, unless `backend.keep_inertial`.

### 11.2 Mechanism

Marginalizing variable set $M$ (the restart bridge's local `V`/`B`) yields, via
the Schur complement on the linearised system, a dense Gaussian factor on the
**separator** $S$ (retained variables that shared a factor with $M$):

$$
\Lambda_S^{+} \;=\; \Lambda_{SS} - \Lambda_{SM}\,\Lambda_{MM}^{-1}\,\Lambda_{MS},
$$

re-added as a `gtsam::LinearContainerFactor` at the current linearisation point —
exact for the linearisation at marginalization time, but frozen there. Because
`V`/`B` are introduced only on fallback intervals (§5), they are naturally
transient: no fixed-lag smoother is needed for the pose graph, which sidesteps the
documented sharp edges of `gtsam_unstable`'s `IncrementalFixedLagSmoother`
(Appendix R.5).

If a *very* long mission makes even the pose graph too large, the escalation is
**keyframe culling / graph sparsification** (drop redundant keyframes in
overlapping-view regions, merging their constraints) rather than fixed-lag
marginalization of poses — a deferred utility, not a normal path.

### 11.3 Caveats

- The marginal prior cannot be relinearized later. Because only transient `V`/`B`
  are marginalized — never poses a future loop might move — this freezing is
  harmless: the persistent pose graph stays fully relinearizable.
- The `KeyframeStore` (`01 §7.5`) is **independent** of graph variable lifetime:
  clouds are kept (for final meshing and possible future loop verification)
  regardless of marginalization. Their poses track the live pose-graph estimate.

---

## 12. Tangent-ordering adapter (Meridian ↔ GTSAM)

This is the **highest-risk silent bug** in the whole back-end, to be pinned by a
unit test. **Meridian orders the 6-DoF tangent translation-first:
`[ρ(trans); φ(rot)]`** (`01 §3.1`), and `PoseCov6` /
`ObservabilityReport.score` follow that. **GTSAM `Pose3` orders rotation-first:
`[rx,ry,rz, tx,ty,tz]`**. Every covariance and Jacobian crossing into GTSAM must
be reordered, or the noise model is silently wrong (a classic block-swap bug;
`01 §3.1`'s "pitfall fixed here").

```cpp
// Permutation P swapping [rho; phi] (Meridian) <-> [phi; rho] (GTSAM Pose3 order).
inline const Eigen::Matrix<double,6,6>& P_meridian_gtsam(){
  static const Eigen::Matrix<double,6,6> P = [](){
    Eigen::Matrix<double,6,6> m = Eigen::Matrix<double,6,6>::Zero();
    m.block<3,3>(0,3) = Eigen::Matrix3d::Identity();   // GTSAM rot rows  <- Meridian phi
    m.block<3,3>(3,0) = Eigen::Matrix3d::Identity();   // GTSAM trans rows<- Meridian rho
    return m;
  }();
  return P;
}
inline Eigen::Matrix<double,6,6> reorder_meridian_to_gtsam(const Eigen::Matrix<double,6,6>& S){ const Eigen::Matrix<double,6,6>& P=P_meridian_gtsam(); return P*S*P.transpose(); }
inline Eigen::Matrix<double,6,6> reorder_gtsam_to_meridian(const Eigen::Matrix<double,6,6>& S){ const Eigen::Matrix<double,6,6>& P=P_meridian_gtsam(); return P.transpose()*S*P; }
```

`to_gtsam(Pose) → gtsam::Pose3` and `from_gtsam(gtsam::Pose3) → Pose` convert the
group elements (`Pose` stores quaternion + translation, `01 §3.1`); the
permutation above handles the *tangent* (covariance/score) ordering. **All
translation-first spec-01 covariances (`PoseCov6` — loop, GNSS, prior) entering a
GTSAM noise model pass through `reorder_meridian_to_gtsam` exactly once** (e.g.
§7.2). The `KeyframePacket.constraint_cov` block is the exception: it is already
rotation-first (`01 §6.1`) and enters the noise model directly (§4.3). The adapter
at the boundary converts conventions; the core math never does (`01 §3.1`).

> **Mandatory regression test:** perturb `T_j` along one tangent axis and assert
> the between-factor error grows along the *expected* covariance axis. This
> catches a swapped/rotated covariance the moment it appears.

---

## 13. Marginal covariance for the loop pre-filter

L5's place-recognition gate should only attempt expensive GICP verification
against keyframes within the search radius implied by the current pose
uncertainty. L3 serves that covariance cheaply:

- `ISAM2::marginalCovariance(keyX(latest))` returns the 6×6 marginal of the latest
  pose (GTSAM tangent order; reorder to Meridian via §12 before handing out). It
  reuses the Bayes tree — far cheaper than building a `Marginals` object. Call it
  at keyframe rate, not per scan.
- The positional 1σ is $\sqrt{\lambda_{\max}}$ of the translation block; L5's gate
  radius is `k_gate · σ_pos`. The same call serves the extrinsic
  convergence-freeze test (§10.2).

This is the third reason the no-double-counting contract (§3.1) is load-bearing:
an over-confident marginal shrinks the search radius and **misses loop closures**,
causing unbounded drift.

---

## 14. Debug & introspection

The structured channel is `BackEndDiagnostics` (`01 Appendix A`: `num_keyframes`,
`num_loops`, `isam_update_ms`, `last_optimize_diverged`); L3 additionally writes
the telemetry below through the `TelemetrySink` abstraction (`00 §10.1`) so the
wrapper maps it to ROS topics / rviz markers. L3 core never calls `rclcpp`
(`00 §1.1`). Health signals follow the iSAM2 monitoring set: `errorBefore/After`,
`variablesRelinearized`, `variablesReeliminated`, chi-square spikes,
indeterminate-system alarms.

### 14.1 Telemetry keys (wrapper-bound, `00 §10.2`)

| Telemetry key | Channel | Cadence | Contents |
|---------------|---------|---------|----------|
| `pose("map/keyframe/<id>")` | pose | kf rate | optimised `X(id)` for path/TF |
| `scalar("backend/chi2")` | scalar | kf rate | total graph χ² from `errorAfter` (graph health, `00 §10.2`) |
| `scalar("backend/n_factors")` | scalar | kf rate | factor count by type (between/imu/loop/gnss) |
| `scalar("backend/update_ms")` | timing | per update | `ISAM2::update` time → `BackEndDiagnostics.isam_update_ms` |
| `scalar("backend/relin_count")` | scalar | per update | `variablesRelinearized` / `variablesReeliminated` (loop-thrash detector) |
| `vec("backend/observability/<id>")` | vec | kf rate | per-axis `s_k`, multiplier `λ_k` (§4.3) |
| `marker("backend/loop_edge")` | marker | on event | green = accepted, red = PCM/GNC-rejected, with `from→to` and weight `w` |
| `event("backend/relinearize")` | event | on big relin | #relinearized vars, Bayes-tree size/depth |
| `event("backend/loop_accepted\|rejected_pcm\|rejected_gnc")` | event | on event | provenance + reason |
| `vec("backend/gnss/residual")` | vec | on fix | residual, χ², robust weight, accepted? + `G` estimate |
| `event("backend/gnss/datum_locked")` | event | on lock | the §6.2 datum fit accepted: fitted yaw, `σ_yaw`, baseline, # fixes used |
| `pose("calib/T_body_sensor/<s>")` | pose | on change | `E(s)` estimate, deviation from prior, marginal σ, frozen? (§10) |
| `event("backend/window_restart_bridge")` | event | on restart edge | the §5 bridge made visible |
| `scalar("backend/queue_depth")` | scalar | per optimize | depth of the back-end input queue (KEYFRAME/LOOP/GNSS), the back-pressure signal — the L3-input twin of the pipeline `q_kf_depth` gauge (spec 09 §5.6) |
| `scalar("backend/optimize_lag")` | scalar | per optimize | number of queued items folded into the last `optimize()` (§9.2/§17), spec 09 §5.3's canonical key for the cadence/batch-size signal |
| `scalar("backend/fallback_count")` | scalar | on event | cumulative count of FM-3b iSAM2 rebuilds (last-resort recovery); a nonzero value is a field-survival alarm |

These align with the debug map in `00 §10.2 / App. C`: observability and residuals
must be *plottable*, not just printed.

### 14.2 Snapshot dump

A `backend.snapshot_on_request` hook writes a `.g2o` (+ values) snapshot of the
whole graph + estimate to `backend.snapshot_dir`, for offline replay and
regression testing. The pipeline can trigger it through the debug control path
(`00 §10.5`).

---

## 15. Failure modes & handling

| ID | Failure | Detection | Handling |
|----|---------|-----------|----------|
| FM-1 | Keyframe gap (`rel_to_id != last_kf_id`) | contiguity assert (§4.2) | log fatal; request L2 resync; do **not** fabricate a bridging factor |
| FM-2 | Indefinite / non-PSD `constraint_cov` | eigen-check on `cov.M` | clamp eigenvalues to `[σ_min², ∞)`; warn; re-check PSD before building the noise model |
| FM-3 | iSAM2 `IndeterminantLinearSystemException` | GTSAM throw caught in `optimize` | identify the near-null variable from the exception key; regularise it with a `LinearDampingFactor` (the same value-free gauge-damping the anchor uses, §3) and/or inflate its connected edges (degeneracy lock §4.3); if it persists, switch `factorization` to QR and retry once; if it still recurs, escalate to FM-3b |
| FM-3b | FM-3 unrecoverable (QR + damping retry still throws) | second `IndeterminantLinearSystemException` after FM-3 escalation | rebuild a fresh `ISAM2` from `isam2_.getFactorsUnsafe()` re-linearised at the last good `calculateEstimate()`, dropping the offending pending batch; if the rebuild also throws, freeze the near-null variable, alert (`last_optimize_diverged = true`), and `++fallback_count` (§14) |
| FM-4 | Loop storm (PCM thrash) | `pending.size() > pcm_max_nodes` | switch to greedy max-clique; back-pressure L5; newest-fitness-first |
| FM-5 | Extrinsic divergence | `‖E(s) ⊟ prior‖ > extrinsic_max_dev` (§10.2) | reject update; re-pin `E(s)`; disable refine for `s`; warn |
| FM-6 | GNSS jump / multipath | per-fix χ² > gate, or robust `w < reject_w` | drop fix; if sustained, auto-disable GNSS and continue loop-only |
| FM-7 | Loop relinearization spike | `relin_count` / `update_ms` spike after a loop | Dogleg already bounds the step; accept one-frame latency; commit big loops on the back-end thread off the odometry path |
| FM-8 | Real-time overrun (`update_ms` over budget) | timing telemetry (§14) | raise `optimize_interval_ms` so more keyframes batch per `ISAM2::update` (§9.2); raise `relinearizeSkip`; request larger keyframe spacing from L2; shed debug topics |
| FM-9 | Back-end input queue saturating (producers outrun the optimise cadence) | `queue_depth` over `backend.queue_warn_depth` (§14) | the queue is bounded and drops oldest non-essential items with a logged count; loops/GNSS/restart edges are never dropped; if depth stays high, fall back to FM-8 (raise cadence interval) |

FM-2 clamps to a valid PSD covariance and proceeds with the (now-valid)
front-end-supplied uncertainty; it does not substitute an arbitrary isotropic
guess, because the per-axis structure (corridor degeneracy, §4.3) is exactly the
information the back-end must preserve.

---

## 16. Configuration (BackendConfig)

The typed sub-tree of `meridian::Config` (`00 §8`) for `meridian_backend`,
expanding spec 00 §8.2's `backend:` block. The `robust` key there names the robust
*strategy family*, realised in §8 as PCM + committed-Huber-incremental +
off-thread-GNC; amortised in-graph GNC is experimental and off by default (§8.2).
`[hot]` = runtime-reconfigurable via the debug-control path (`00 §10.5`);
`[cold]` = restart required. `Config::validate()` (`00 §8.3`) range-checks these
on load.

```cpp
namespace meridian {
struct BackendConfig {
  std::string kind = "isam2";              // [cold] the IBackEnd impl (00 §8.2)

  // --- gauge / anchor ---
  double anchor_sigma                = 1e-4;  // [cold] LinearDampingFactor sigma on X(first); lambda = 1/sigma^2

  // --- iSAM2 (§9.1; Appendix R.1) ---
  int    isam2_relinearize_skip      = 1;     // [hot]
  double isam2_relinearize_thresh    = 0.1;   // [hot] scalar fallback; per-type map is built in code
  int    extra_iters_normal          = 0;     // [hot] extra Dogleg passes on a normal batch (§9.2)
  int    extra_iters_loop            = 4;     // [hot] extra Dogleg passes when a loop was admitted (§9.2)
  bool   isam2_use_qr                = false; // [cold] QR vs Cholesky (Cholesky default; QR on degeneracy)

  // --- optimise cadence (§9.2 / §17), decoupled from keyframe insert ---
  double optimize_interval_ms        = 100.0; // [hot] min wall-clock between batched ISAM2::update calls
  int    queue_warn_depth            = 32;    // [hot] back-end input queue depth that raises FM-9

  // --- observability -> noise (§4.3) ---
  double obs_inflation_max           = 1e4;   // [hot] rho_max
  double obs_inflation_gamma         = 2.0;   // [hot]
  double degenerate_thresh           = 0.05;  // [hot] score below = degenerate
  bool   degenerate_lock             = true;  // [hot] hard-lock worst axis if degenerate

  // --- loops + PCM (§7) ---
  double loop_min_fitness            = 0.5;   // [hot] reject low GICP fitness
  double pcm_chi2_alpha              = 0.99;  // [hot]
  int    pcm_max_nodes               = 64;    // [hot] exact max-clique cap

  // --- robust kernels (§8) ---
  std::string robust_kernel          = "huber";    // [hot] committed incremental loop kernel; huber|cauchy|gm|tls (only huber is known-safe incremental, §8.1)
  double loop_huber_k                = 1.345;       // [hot] Huber tuning for the committed loop kernel
  double gnss_huber_k                = 1.345;       // [hot] Huber tuning for GNSS (§6.4)
  double gnc_reject_w                = 0.1;         // [hot] weight below which a loop is removed + PCM-marked
  // amortised in-graph GNC — EXPERIMENTAL, off by default (§8.2)
  bool   gnc_enabled                 = false;       // [hot] experimental amortised GNC inside iSAM2
  int    gnc_anneal_steps            = 5;           // [hot]
  // off-thread batch-GNC consolidation (§8.3)
  int    gnc_consolidate_interval    = 10;          // [hot] admitted loops between batch consolidations (0 = off)

  // --- GNSS (§6) ---
  bool   gnss_enabled                = true;  // [hot]
  double gnss_max_cov                = 25.0;  // [hot] drop fix if trace(cov_enu) above (m^2)
  bool   gnss_lock_yaw               = false; // [hot] external heading available (§6.5)
  // datum init (§6.2)
  double gnss_min_baseline           = 5.0;   // [hot] m, min travelled baseline before datum fit
  double gnss_min_excitation         = 3.0;   // [hot] m, min dominant-axis span of buffered ENU track
  double gnss_min_speed              = 0.5;   // [hot] m/s, speed a fix must exceed to count as moving
  int    gnss_min_moving_fixes       = 5;     // [hot] # moving fixes required before datum fit
  double gnss_datum_yaw_sigma_max    = 5.0;   // [hot] deg, reject datum fit above this yaw uncertainty
  // gating + decimation (§6.4)
  bool   gnss_skip_if_confident      = true;  // [hot] skip fix no tighter than the back-end marginal
  double gnss_skip_confidence_k      = 1.0;   // [hot] kappa in the skip-if-confident test
  double gnss_min_spacing            = 1.0;   // [hot] m, min travelled baseline between admitted fixes
  // drift redistribution (§6.6, designed-but-deferred)
  bool   gnss_redistribute           = false; // [hot] spread re-acquisition error across the outage span
  int    gnss_reacq_fix              = 1;     // [hot] min fix-quality enum to count as re-acquired
  int    gnss_reacq_persist          = 5;     // [hot] consecutive accepted fixes to declare re-acquisition
  int    gnss_redistribute_span_max  = 200;   // [hot] max keyframes back redistribution reaches

  // --- online extrinsics (§10), off by default; enable per-platform ---
  bool   extrinsic_refine            = false; // [cold] master switch; enable only when calib is trusted-but-drifting AND excitation is expected (§10)
  double extrinsic_prior_sigma       = 1e-3;  // [cold]
  double extrinsic_refine_sigma      = 1e-2;  // [hot]
  double extrinsic_excite_rot        = 0.5;   // [hot] rad cumulative before refine
  double extrinsic_excite_trans      = 2.0;   // [hot] m   cumulative before refine
  double extrinsic_freeze_cov        = 1e-6;  // [hot] freeze when marginal below
  double extrinsic_max_dev           = 0.1;   // [hot] sanity clamp (FM-5)

  // --- marginalization (§11) ---
  bool   keep_inertial               = false; // [hot] keep restart V/B variables live

  // --- L4 re-integration trigger (§9.3) ---
  double reintegrate_thresh          = 0.1;   // [hot] m, pose-move threshold for GraphUpdate.moved
  bool   emit_moved_cov              = false; // [hot] fill GraphUpdate.Moved.cov (costly)

  // --- loop pre-filter marginal (§13) ---
  double loop_gate_k                 = 3.0;   // [hot] k_gate * sigma_pos search radius for L5

  // --- IMU noise for restart edges (§5.3); overridden by CalibrationSet (01 §5.3) ---
  struct Imu {
    double acc_noise   = 0.1;    // m/s^2/sqrt(Hz)
    double gyr_noise   = 0.1;    // rad/s/sqrt(Hz)
    double acc_bias_rw = 1e-4;   // m/s^3/sqrt(Hz)
    double gyr_bias_rw = 1e-4;   // rad/s^2/sqrt(Hz)
  } imu;

  // --- debug (§14) ---
  bool        debug_dump_residuals   = false; // [hot]
  bool        snapshot_on_request    = false; // [hot]
  std::string snapshot_dir           = "/tmp/meridian";
};
}
```

---

## 17. The back-end thread loop

The back-end runs on its own thread (`01 §2.4`: "a back-end thread (L3 + L5)") and
owns the `ISAM2` instance exclusively; every graph mutation flows through this
single loop — the invariant that makes the no-double-counting contract (§3.1)
enforceable in one place. The thread *stages* mutations as items arrive but
*optimises* on the §9.2 cadence: it drains every queued item into the pending
batch, then calls `optimize()` at most once per `optimize_interval_ms` — or
immediately when the batch admitted a loop or a GNSS datum lock.

```text
backend_thread():
  isam2 = ISAM2(params from cfg)                       # §9.1
  last_opt = now()
  loop forever:
    # Drain the whole queue into the pending batch; block only when it is empty.
    item = backend_queue.pop(timeout = optimize_interval_ms)   # bounded [TS] from L2/L5/L0
    while item:
      switch item.kind:
        case KEYFRAME:                                  # IBackEnd::add_keyframe (01 §7.4)
          switch item.kf.constraint_kind:               # §3.1 — single odometry factor
            RelativeBetween:   add_between_edge(item.kf)        # §4   (stages new_graph/new_values)
            ImuPreintegration: add_restart_imu_edge(item.kf)    # §5
            AbsolutePrior:     add_anchor_or_absolute(item.kf)  # §3 (first KF) / §6 (GNSS)
        case LOOP:    process_loop(item.lc)             # PCM admit, §7  (add_loop_constraint)
        case GNSS:    buffer_or_add_gnss(item.fix, item.kf_id)  # §6  (add_absolute)
      item = backend_queue.try_pop()                    # keep draining without blocking
    publish_queue_depth()                                # §14  backend/queue_depth
    force = batch_has_loop() or gnss_datum_just_locked()
    if not force and elapsed(last_opt) < optimize_interval_ms: continue   # wait for more, batch it
    if nothing_staged(): continue
    if gnc_active:        step_gnc_anneal()             # §8
    if should_marginalize(): marginalize_transient()    # §11
    gu = optimize()                                      # §9.2  -> GraphUpdate (folds the whole batch)
    last_opt = now()
    map_thread_queue.push(gu)                            # §9.3  -> L4 re-integration [TS]
    serve_loop_gate_cov()                                # §13   -> L5 marginal cov
    if extrinsics_changed(): publish_calibration_snapshot()  # §10 -> L2 [TS]
    publish_debug()                                      # §14
```

The input `backend_queue` is the bounded primitive of `00 §11`: it blocks
producers only on shutdown and otherwise drops the **oldest non-essential** item
(a stale `corrected_trajectory()`-style request) with a logged drop count when
depth exceeds `queue_warn_depth` (FM-9). KEYFRAME (any kind), LOOP, and GNSS items
are essential and never dropped — losing a keyframe would break the §4.2
contiguity contract (FM-1).

`optimize()`, `corrected_trajectory()`, `refined_calibration()`, and
`diagnostics()` are the `IBackEnd` methods (`01 §7.4`) the pipeline calls; the
loop above is the internal driver that backs them.

---

## 18. Module integration order

This is the **compile/integration order** for bringing the back-end up against
its collaborators — not a feature rollout; each item is the same final design.

1. **Variables + the normal between-edge (§2, §4) + gauge anchor + iSAM2 wiring
   (§9).** Drive it with `RelativeBetween` packets only; assert the
   single-factor-per-edge invariant and the tangent-ordering regression test
   (§12). This is the spine.
2. **Restart bridge (§5)** and the transient-`V`/`B` marginalization (§11) that
   keeps the graph pose-only afterward.
3. **GNSS (§6)** — the origin variable `G`, the lever-arm factor, Huber gating.
4. **Loop closures (§7 PCM + §8)** and the marginal-covariance gate served to
   L5 (§13).
5. **Online extrinsics (§10)**, off by default, with the excitation/freeze
   safeguards.

Which front-end produced a packet is irrelevant from L3's side (the retired iEKF
test oracle proved the point, `00 §5.4`): L3 consumes the same `KeyframePacket`
regardless of producer and never branches on `frontend_kind` (`01 §6.2`).

---

## 19. Cross-reference index

- Boundary types — `KeyframePacket`/`ConstraintKind`/`ImuPreintegrationSummary`
  (`01 §6`), `ObservabilityReport`/`PoseCov6`/`GaussianBlock` (`01 §3`),
  `LoopConstraint` (`01 §7.6`), `GnssFix` (`01 §4.4`), `GraphUpdate`/`StampedPose`
  (`01 §7.4`, App. A), `CalibrationSet`/`Extrinsic` (`01 §5`), `Frame` (`01 App. A`),
  `IBackEnd` (`01 §7.4`), `KeyframeStore` (`01 §7.5`).
- Architecture — frame tree & time (`00 §2`; `01 §2.2`), threading (`01 §2.4`,
  `00 §11`), config (`00 §8`), debug/telemetry (`00 §10`), `PoseCorrection`
  concept (`00 §3`, reconciled §9.3 here), nvblox single-map mandate (`00 §2.1`,
  `00 §9.5`), iEKF-as-oracle (`00 §5.4`).
- Sibling specs — L2 front-end `04_frontend_estimation.md` (produces
  `RelativeBetween` and the restart `ImuPreintegration` summary), L4 map
  `06_mapping.md` (consumes `GraphUpdate`), L5 place recognition
  `07_loop_closure.md` (produces `LoopConstraint`, consumes the
  marginal-covariance gate, references the §3.2 chi-square convention).
- Reference grounding — **Appendix R** (non-normative): ISAM2Params defaults +
  Meridian deltas R.1; L2→L3 hand-off contract R.2; IMU preintegration objects
  R.3; robust kernels / GNC / switchable / PCM contrast R.4; marginalization +
  gtsam_unstable sharp edges R.5; marginal covariance for the loop pre-filter R.6;
  bibliography R.7.
- Reference code (under `/home/user/slam-reference`) — `FAST_LIO`
  (`use-ikfom.hpp`, `IMU_Processing.hpp`, `config/ouster64.yaml`), GTSAM-4.2 iSAM2
  usage in `LIO-SAM/src/mapOptmization.cpp`, PCM/GNC in `Kimera-RPGO`, fixed-lag
  in `glim` / `gtsam_points` (Appendix R).

---

*End of spec 05. The L3 back-end consumes only the spec-01 boundary types and
emits only the spec-01 outputs; its sole non-negotiable internal invariant is the
single-odometry-factor-per-edge rule of §3.1.*

---

## Appendix R — SOTA reference grounding (non-normative)

Evidence, not contract: digests of the reference systems this design was validated
against; nothing here binds Meridian's behavior. Clones live in
`/home/user/slam-reference`. GTSAM itself is not cloned; GTSAM-knob facts were
verified against the **pinned GTSAM 4.2** source plus the GTSAM-4.2-using
references (LIO-SAM, Kimera-RPGO, glim, gtsam_points). The iSAM2/Bayes-tree
algorithm theory is owned by the normative body and the papers (R.7).

### R.1 ISAM2Params defaults (GTSAM 4.2) & Meridian deltas

Verified against GTSAM 4.2.0 (the spec-11 pin); cross-checked against
LIO-SAM@0be1fbe `src/mapOptmization.cpp:159-161`, `src/imuPreintegration.cpp:233-235`.

| Param | Type | GTSAM 4.2 default | Meridian (§9.1) |
|---|---|---|---|
| `relinearizeThreshold` | `double` **or** `FastMap<char,Vector>` | `0.1` (scalar) | per-symbol-type map (`x/v/b/e/g`) |
| `relinearizeSkip` | `int` | `10` | `1` (every-step check; matches LIO-SAM) |
| `enableRelinearization` | `bool` | `true` | `true` |
| `enablePartialRelinearizationCheck` | `bool` | `false` | `false` |
| `cacheLinearizedFactors` | `bool` | `true` | `true` |
| `evaluateNonlinearError` | `bool` | `false` | `true` (health χ²) |
| `factorization` | `CHOLESKY`/`QR` | `CHOLESKY` | `CHOLESKY`; QR only on degeneracy (FM-3) |
| `optimizationParams` | GaussNewton/Dogleg | GaussNewton | **Dogleg** (trust region) |
| `findUnusedFactorSlots` | `bool` | `false` | `true` (loop/GNSS churn) |
| `enableDetailedResults` | `bool` | `false` | `true` (debug) |

`ISAM2Result` fields used for health monitoring (§14): `variablesRelinearized`,
`variablesReeliminated`, `cliques`, `factorsRecalculated`, and (with
`evaluateNonlinearError`) `errorBefore`/`errorAfter`.

### R.2 The L2→L3 hand-off contract (the load-bearing rule)

Design contract owned by §3.1/§4/§5; kept here as the one-line invariant table.

| Mode | New vars at kf `j` | Factor(s) | Rule |
|---|---|---|---|
| Normal | `X(j)` | one `BetweenFactor<Pose3>` with `Gaussian::Covariance(Σ_ij)` | no IMU, no LiDAR factor on the edge |
| Restart | `X(j),V(j),B(j)` (transient) | one `CombinedImuFactor` | **replaces** the between-factor; never both |
| GNSS @ fix time | none | `GnssFactor` over the two bracketing poses + `G` (robust/switchable) | bound to the fix-time interpolated pose, gated/decimated before insert |
| Loop `j↔k` | none | switchable `BetweenFactor<Pose3>` (GNC) | PCM-gated before insert |

The hazard prevented: absolute-prior + between + IMU factor from the *same*
L2-fused measurements → triple-counted information → over-confident covariance →
rejected loop/GNSS corrections and a collapsed loop-search radius. Passing the
*marginal* relative covariance (not conditional) means it already reflects
observability, so the back-end trusts it correctly per axis.

### R.3 IMU preintegration (Forster et al. T-RO 2017) — GTSAM objects

Verified against GTSAM 4.2 `gtsam/navigation/`; usage cross-checked against
LIO-SAM@0be1fbe `src/imuPreintegration.cpp` and glim@25ad190 `src/glim/odometry/`.

- `PreintegrationCombinedParams`: `accelerometerCovariance`,
  `gyroscopeCovariance`, `integrationCovariance`, `biasAccCovariance`,
  `biasOmegaCovariance`, `biasAccOmegaInt`, `n_gravity`.
- `PreintegratedCombinedMeasurements pim(params, biasHat)`; accumulate via
  `pim.integrateMeasurement(acc, gyro, dt)`.
- `CombinedImuFactor(X(i),V(i),B(i), X(j),V(j),B(j), pim)` — 6-way; bakes in the
  bias random walk, preferred over `ImuFactor` + separate `BetweenFactor<imuBias>`.
- `pim.predict(NavState_i, biasHat) → NavState_j` seeds the new keyframe value;
  `pim.resetIntegrationAndSetBias(newBias)` after commit.
- `n_gravity` sign/axis is world-frame dependent — Meridian's `map` is
  gravity-aligned ENU, so `(0,0,-9.81)` (§5.3).

Meridian rebuilds the PIM from the `ImuPreintegrationSummary` boundary type (§5.1);
raw IMU never crosses L2→L3.

### R.4 Robust kernels, GNC & switchable constraints — cross-system contrast

Verified against gtsam_points@85d0f4c
(`registration/impl/graduated_non_convexity_impl.hpp`), Kimera-RPGO@d28b4df
(`include/KimeraRPGO/SolverParams.h`, `outlier/Pcm.h`, `src/RobustSolver.cpp`),
and GTSAM 4.2 `gtsam/nonlinear/GncOptimizer.h`.

| Concept | Reference impl | Verified detail |
|---|---|---|
| GNC GM weight | gtsam_points `graduated_non_convexity_impl.hpp:165` | `w = (μ/(μ+e))²`, schedule `μ /= div_factor` (GM **decreases** μ) |
| GNC `muStep` | Kimera-RPGO `SolverParams.h:65` | `mu_step_ = 1.4` default; direction is per-loss (GM divides, TLS multiplies) |
| GNC inlier gate | Kimera-RPGO `SolverParams.h:61-62` | `PROBABILITY` mode, `gnc_inlier_threshold_ = 0.9`; GTSAM's `setInlierCostThresholds` derives `c̄²` from a χ² quantile for the factor dim |
| GNC is batch | GTSAM `GncOptimizer` wraps LM/GN | not incremental — cannot run inside `ISAM2::update` |
| PCM | Kimera-RPGO `outlier/Pcm.h`, `max_clique_finder/` | pairwise-consistency + max-clique; defaults `odom_threshold=10.0`, `lc_threshold=5.0` |
| Switchable constraints | not first-class in GTSAM 4.2 | custom factor + `PriorFactor<double>` switch, **or** GNC (the modern subsumption) — Meridian uses no switch variable |
| Robust M-estimators | GTSAM `noiseModel::Robust` | `Huber` (k≈1.345, convex → safe in incremental IRLS); `GemanMcClure`/`Cauchy`/`Tukey`/`DCS`/`Welsch` are redescending, non-convex |

### R.5 Bounded graphs: marginalization & the gtsam_unstable sharp edges

Verified against gtsam_points@85d0f4c
(`optimizers/incremental_fixed_lag_smoother_ext.cpp`, the `…WithFallback` wrapper)
and GTSAM 4.2 `gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h`.

`IncrementalFixedLagSmoother` wraps iSAM2 and marginalizes keys older than a lag
via Schur complement, leaving a `LinearContainerFactor` — exact at the
marginalization-time linearization, but frozen there. Confirmed sharp edges
(gtsam_points ships a fallback wrapper because the base class throws in practice):

- Do **not** call `update()` more than once per timestamp; batch a keyframe's
  factors into one `update()`.
- Segfaults when marginalizing a variable whose only factor is a prior — keep
  every boundary-entering variable constrained by a non-prior factor.
- "Requested variable … not in this VectorValues" when keys/timestamps drift —
  keep the `KeyTimestampMap` exactly in step.

Meridian sidesteps all three by marginalizing only transient `V`/`B` (§5/§11) —
never poses — so no fixed-lag smoother is needed.

### R.6 Marginal covariance for the loop pre-filter

Verified against GTSAM 4.2 `gtsam/nonlinear/ISAM2.h` / `Marginals.h`.

- `ISAM2::marginalCovariance(Key)` returns the 6×6 marginal (GTSAM tangent order)
  reusing the Bayes tree — far cheaper than constructing a `Marginals` object;
  `jointMarginalCovariance({keys})` for relative uncertainty.
- Pre-filter use (§13): `σ_pos = √λ_max` of the translation block; gate radius
  `k_gate · σ_pos`. Call at keyframe rate, not per scan.

### R.7 Bibliography (compact)

- **iSAM2** — Kaess, Johannsson, Roberts, Ila, Leonard, Dellaert, *"iSAM2:
  Incremental Smoothing and Mapping Using the Bayes Tree,"* IJRR 31(2):216–235,
  2012. (Bayes tree introduced in Kaess et al., WAFR 2010.)
- **IMU preintegration** — Forster, Carlone, Dellaert, Scaramuzza, *"On-Manifold
  Preintegration for Real-Time Visual-Inertial Odometry,"* IEEE T-RO 33(1):1–21,
  2017 (what GTSAM's `Preintegrated(Combined)Measurements` implements).
- **GNC** — Yang, Antonante, Tzoumas, Carlone, *"Graduated Non-Convexity for
  Robust Spatial Perception…,"* IEEE RA-L/T-RO 2020 (arXiv:1909.08605); GTSAM
  `GncOptimizer<T>`.
- **Switchable constraints** — Sünderhauf & Protzel, *"Switchable Constraints for
  Robust Pose Graph SLAM,"* IROS 2012.
- **PCM** — Mangelson, Dominic, Eustice, Vasudevan, *"Pairwise Consistent
  Measurement Set Maximization for Robust Multi-robot Map Merging,"* ICRA 2018
  (Kimera-RPGO `outlier/Pcm.h`).
- **Degeneracy** — Zhang, Kaess, Singh, *"On Degeneracy of Optimization-based
  State Estimation Problems,"* ICRA 2016 (basis for the §4.3 observability
  inflation).
