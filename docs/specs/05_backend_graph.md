# 05 — L3 Keyframe Factor-Graph Back-End (iSAM2)

> **Spec status:** normative implementation spec for layer **L3**. It implements
> the `IBackEnd` interface defined in `01_interfaces_and_data_types.md §7.4` and
> consumes the `KeyframePacket` (`01 §6`), `LoopConstraint` (`01 §7.6`), and
> `GnssFix` (`01 §4.4`) value types defined there. It emits the `GraphUpdate`
> (`01 §7.4`), the `corrected_trajectory()` (`01 §7.4`, `StampedPose`), and the
> refined `CalibrationSet` snapshot (`01 §5.3`). **This document does not
> redefine any boundary type** — if it needs a new field on a boundary type it
> amends spec 01, per spec 01's own rule. Everything below is *internal* to
> `meridian_backend` except where it names a spec-01 type.
>
> **Where L3 sits.** Meridian is one complete system: a continuous-time (CT),
> tightly-coupled LiDAR-Inertial-Visual-GNSS estimator (`00 §0`). The **L2 CT
> LIVO+GNSS front-end** fuses a single LiDAR, a single IMU, a single camera, and
> GNSS inside a sliding B-spline window and summarises each keyframe interval as
> **one** `KeyframePacket`. **L3 is the global, drift-free layer:** it stitches
> those packets, loop closures, and GNSS into a single iSAM2 factor graph in the
> `map` frame and broadcasts corrections to the live front-end (L2) and the
> nvblox map (L4). L3 does not estimate the trajectory inside a window — that is
> L2's job; L3 estimates the *globally consistent* keyframe poses, the GNSS-datum
> alignment, and the online extrinsics.
>
> **Engine:** GTSAM's `ISAM2` (incremental smoothing and mapping). Familiarity
> with the GTSAM factor-graph API and the iSAM2 / Bayes-tree algorithm is
> assumed; the course covers the theory in **§03 (factor graphs, MAP as nonlinear
> least squares, the information form)** and **§09 (incremental smoothing, the
> Bayes tree, fluid relinearization, marginalization)**, and the engineering is
> grounded line-by-line in **`docs/grounding/09_backend_isam2.md`** (the iSAM2 /
> GTSAM back-end dossier: Bayes tree, fluid relinearization, the L2→L3 hand-off
> contract, GNC, marginalization, marginal-covariance queries). We cite those for
> derivations and specify here only how Meridian *uses* them; every GTSAM knob below
> traces to a dossier §, cross-checked against the pinned GTSAM **4.2** source.
>
> **The single most important rule.** Between any two consecutive keyframes the
> back-end adds the L2 information **exactly once**: one relative `BetweenFactor`
> built from the packet's `RelativeBetween` constraint, with its marginal
> covariance. There is **no** companion absolute prior and **no** IMU factor on
> that edge. The IMU factor (`CombinedImuFactor`) appears **only** on the
> window-restart fallback (`constraint_kind == ImuPreintegration`), where it
> *replaces* the between-factor. The two are mutually exclusive *by construction*
> because `KeyframePacket::constraint_kind` is a single enum (`01 §6.4`), and the
> dossier (`grounding 09 §8`, "★ the L2→L3 hand-off contract ★") makes the same
> rule the most load-bearing part of the back-end. §4 and §5 make this airtight.

---

## Table of contents

1. [Scope & responsibilities](#1-scope--responsibilities)
2. [Variables: the state the graph estimates](#2-variables-the-state-the-graph-estimates)
3. [Factors: the constraints](#3-factors-the-constraints)
4. [The L2→L3 normal edge (the common case)](#4-the-l2l3-normal-edge-the-common-case)
5. [The restart edge (the only IMU factor)](#5-the-restart-edge-the-only-imu-factor)
6. [GNSS factors & the GNSS-origin variable](#6-gnss-factors--the-gnss-origin-variable)
7. [PCM pre-filter (loop outlier rejection)](#7-pcm-pre-filter-loop-outlier-rejection)
8. [Robust kernels & GNC (loops + GNSS)](#8-robust-kernels--gnc-loops--gnss)
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
   odometry factor (§4/§5), respecting the no-double-counting contract
   (`grounding 09 §8.1`).
3. Fold in **switchable** constraints: loop closures (`LoopConstraint`, L5) and
   GNSS (`GnssFix`, L0), each guarded by a robust kernel (GNC, §8) and an outlier
   pre-filter (PCM for loops, §7).
4. Estimate the **GNSS-origin** alignment (`map ← ENU`) and the **online
   extrinsics** (on by default, §10) as graph variables, publishing refined
   calibration back to L2 as a versioned `CalibrationSet` snapshot (`01 §5.3`).
5. Run incremental `ISAM2::update` per keyframe; relinearize fluidly; marginalize
   transient inertial variables to keep the steady-state graph pose-only (§11).
6. Produce the outputs `IBackEnd` promises: `optimize()` returns a `GraphUpdate`
   (which keyframes moved and to what), `corrected_trajectory()` returns the
   `map`-frame `StampedPose` list, `refined_calibration()` returns the snapshot.
7. Serve the latest pose marginal covariance to L5's place-recognition gate (§13).
8. Emit rich debug (`BackEndDiagnostics` plus dedicated telemetry keys): graph
   summary, per-axis observability inflation, loop/GNSS accept/reject, chi-square,
   relinearization counts, timing.

### 1.2 Out of scope (handled elsewhere)

- Front-end estimation, deskew, keyframe *emission policy*, and the construction
  of `RelativeBetween` / `ImuPreintegration` constraints — **L2**
  (`04_frontend_estimation.md`; the boundary types are fixed in `01 §6`).
- Loop-candidate *generation* and geometric verification (Scan Context++ →
  STD/BTC → GICP) — **L5** (`ILoopDetector`, `01 §7.6`; `07_loop_closure.md`).
  L3 only *consumes* a verified `LoopConstraint` and decides accept/reject via PCM
  (§7) + GNC (§8).
- TSDF / colour / mesh and the clear-and-rebuild **re-integration** itself — **L4**
  (`IMapLayer::apply_graph_update`, `01 §7.5`; `06_mapping.md`). L4 is **nvblox on
  the GPU** — the one and only map backend, no CPU path. L3 *triggers* a region
  rebuild by emitting a `GraphUpdate`; it never touches voxels.
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
        |           prior(anchor) | extrinsic-prior                     |
        |  engine: GTSAM ISAM2  (Bayes tree, fluid relinearization)     |
        +--------------------------------------------------------------+
               |                        |                      |
               v                        v                      v
        GraphUpdate -> L4         corrected_trajectory     refined CalibrationSet
        (clear-and-rebuild)       (StampedPose, map)       snapshot -> L2 (01 §5.3)
```

Every keyframe carries exactly **one** `obs[6]` observability report (`01 §3.4`),
because there is exactly **one** LiDAR in the system; there is no per-sensor merge
of observability streams to perform. (Multi-LiDAR would be a future extension
behind the same `KeyframePacket` interface; it is not designed here.)

> **Note on the `PoseCorrection` / `map→odom` concept.** Spec 00 §3 sketches a
> "`PoseCorrection` broadcast to L4/L2." In the *normative* type system (spec 01
> §7.4) that broadcast is realized concretely as: (a) the `GraphUpdate` to L4, and
> (b) the refined `CalibrationSet` + the corrected trajectory the live front-end
> rebases against. The front-end keeps publishing smooth `odom`-frame `NavState`
> (`01 §7.3`); L3 owns `map` and the `map`-relative correction is *implicit* in
> the difference between `corrected_trajectory()` and the front-end's odom poses
> at the same keyframe stamps. §9.3 specifies how a consumer computes the
> `map→odom` transform from these outputs without L3 minting a new boundary type.

---

## 2. Variables: the state the graph estimates

All variables live in one GTSAM `Values`. Keys use `gtsam::Symbol(char, index)`
with a fixed character namespace so the elimination ordering and debug dumps are
legible (the `x/v/b/e` convention matches `grounding 09 §4.3, §14`).

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

**Steady state: NO** — and this is fixed by the contract, not a local choice.
Spec 01 §6.4 states the decision explicitly: in the normal path
`kinematics_included = false`, velocity and biases stay *inside* the front-end and
their effect is folded into the relative covariance `constraint_cov`. L3 therefore
creates **no** `V`/`B` variables for a normal keyframe; the keyframe is a pure pose
node. The packet's `v_ref`/`b_a`/`b_g` fields, when present, are recorded for
debug/seeding only. This is exactly the dossier's recommended design: keep `V`/`B`
**transient** — introduced only on fallback intervals, so they are naturally
short-lived and never accumulate in a multi-hour graph (`grounding 09 §8.5, §12.2`).

**Exception — restart edges.** When `constraint_kind == ImuPreintegration`
(`kinematics_included == true`, `01 §6.4/§6.5`), L3 creates `V`/`B` on **both
endpoints of that one edge** and links them with a `CombinedImuFactor` built from
the packet's `ImuPreintegrationSummary` (§5). These inertial variables are *local
to the restart bridge*; they are marginalized at the next normal keyframe unless
`backend.keep_inertial` is set (default `false`). This keeps the steady-state
graph pose-only — the cheap, well-conditioned regime the dossier recommends
(`grounding 09 §12.2`: full iSAM2 for the pose graph, transient `V`/`B`).

> **Grounding.** The full inertial state (pos, rot, vel, `b_g`, `b_a`, gravity,
> plus the LiDAR-IMU extrinsic) is what a FAST-LIO2-style filter keeps live in one
> state — `state_ikfom` in `FAST_LIO/include/use-ikfom.hpp:12–21`
> (`pos, rot, offset_R_L_I, offset_T_L_I, vel, bg, ba, grav`). Meridian's L2
> CT front-end carries the equivalent inside its window; L3 deliberately keeps
> only what is needed *at the keyframe-graph level*. The front-end already
> estimated vel/bias, and the relative `BetweenFactor` already carries that
> information through its marginal covariance (`01 §6.4`; `grounding 09 §8.2`).
> Carrying `V`/`B` on every edge **and** a relative factor would re-inject the same
> IMU evidence twice — the double-count the hand-off contract forbids
> (`grounding 09 §8.1`).

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

The mapping `id → X(id)` is identity on the integer payload, so the GTSAM symbol
index *is* the `KeyframePacket::id`. No side table is required for poses; a small
`std::unordered_map<std::uint64_t, KfRecord>` holds per-keyframe bookkeeping
(stamp, cloud/image handles, observability, the stored odom hint — §4).

---

## 3. Factors: the constraints

Every factor is a negative-log-likelihood term (course §03). Meridian uses these and
**only** these factor classes (the canonical set of `grounding 09 §14`,
"Factors"):

| Factor | GTSAM class | Connects | Source | Switchable | Robust |
|--------|-------------|----------|--------|-----------|--------|
| Anchor prior | `PriorFactor<Pose3>` | `X(first)` | bootstrap | no | no |
| Odometry (normal) | `BetweenFactor<Pose3>` | `X(rel_to_id), X(id)` | `RelativeBetween` | no | no |
| Odometry (restart) | `CombinedImuFactor` | `X,V,B(from), X,V,B(to)` | `ImuPreintegration` | no | no |
| Bias-walk (restart) | folded into `CombinedImuFactor` | `B(from), B(to)` | summary | no | no |
| Loop closure | `BetweenFactor<Pose3>` | `X(from_id), X(to_id)` | `LoopConstraint` | **yes** | **GNC** |
| GNSS position | `GnssFactor` (custom `NoiseModelFactor2`) | `X(i), G` | `GnssFix` | **yes** | **GNC** |
| GNSS-origin prior | `PriorFactor<Pose3>` | `G` | first fix | no | weak |
| Extrinsic prior | `PriorFactor<Pose3>` | `E(s)` | calibration | no | tight (§10) |

The anchor prior fixes the gauge (a relative-only graph has a 6-DoF null space,
course §03; `grounding 09 §11.2` lists "forgot a prior on the first pose" as the
canonical `IndeterminantLinearSystemException` cause). It is added once, at the
first keyframe, with a tight covariance (`backend.anchor_sigma`), and **never**
re-added. The first keyframe normally arrives with
`constraint_kind == AbsolutePrior` (`01 §6.4`: "Used for the first keyframe … to
fix the gauge"); L3 maps that to this anchor prior.

### 3.1 Why exactly one odometry factor per edge

The hand-off contract (`grounding 09 §8`) is precise about the failure it
prevents: an absolute marginal prior per keyframe, a relative between-factor,
**and** an IMU-preintegration factor — all derived from the **same** LiDAR+IMU(+
camera) measurements the L2 window already fused — would triple-count the
evidence. Summing their information shrinks covariances unrealistically, the graph
becomes over-confident, and loop/GNSS corrections are rejected because the
optimiser trusts odometry too much. The same over-confidence also collapses the
marginal-covariance search radius and silently *misses loop closures*
(`grounding 09 §13.3`).

Meridian's contract, keyed off the single `KeyframePacket::constraint_kind` enum:

```cpp
switch (kf.constraint_kind) {
  case ConstraintKind::RelativeBetween: add_between_edge(kf);    break;  // §4
  case ConstraintKind::ImuPreintegration: add_restart_imu_edge(kf); break; // §5
  case ConstraintKind::AbsolutePrior:   add_anchor_or_absolute(kf); break; // §3 / §6
}
```

Because the source is a single enum, you *cannot* ship two odometry factors for
one edge (`grounding 09 §8.4` "Contract rule #3 — No overlap, ever"). The
information crosses L2→L3 exactly once, audited at one site.

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

  # 2. The SINGLE odometry factor (the hand-off contract, grounding 09 §8.1).
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
pipeline calls it after `add_keyframe` (§9, §17). Contiguity (`rel_to_id ==
last_kf_id`) is asserted: a gap means L2 dropped a keyframe, a fatal contract
violation (FM-1) — L3 does **not** fabricate a bridging factor.

The noise model is `Gaussian::Covariance(Σ_ij)` fed the front-end's marginal
covariance directly, exactly as the dossier prescribes (`grounding 09 §6.4, §8.1,
§14`): we ship `Σ` (not an information matrix), tagged `Covariance` by the
`GaussianBlock` form (`01 §3.3`), so the conversion is explicit and auditable at
this one site.

### 4.3 Observability → noise inflation (the degeneracy contract)

The packet's single `ObservabilityReport` (`01 §3.4`) carries per-axis scores
`score[k] ∈ [0,1]` (1 = fully observable) in a **named frame** `observability.frame`,
ordering `[tx,ty,tz,rx,ry,rz]`. L3 maps a low score to an **inflated covariance**
along that axis of the between-factor, so the optimiser does not trust a direction
the front-end could not constrain (e.g. translation along a featureless corridor).
This is the X-ICP / D²-LIO-style signal flowing into back-end noise that spec 01
§3.4 promises would be defined for L3 — defined here. The dossier prescribes
exactly this response to degeneracy (`grounding 09 §11.1`: "Never silently drop a
factor — inflate its covariance"; rotate `Σ` into the degenerate eigenbasis, raise
the bad-axis variance, rotate back).

Per-axis multiplier:

$$
\lambda_k \;=\; 1 + (\rho_{\max}-1)\,(1 - s_k)^{\gamma}, \qquad \lambda_k \in [1,\rho_{\max}],
$$

with `ρ_max = backend.obs_inflation_max` (default `1e4`) and shape exponent
`γ = backend.obs_inflation_gamma` (default `2.0`). A fully observable axis
(`s_k=1`) gives `λ_k=1` (no change); a fully degenerate axis (`s_k=0`) inflates
that variance by `ρ_max`.

**Frame handling.** The scores are expressed in `observability.frame`. The
covariance lives in the factor's body tangent. Build a 6×6 block-diagonal
rotation `R6` that maps the observability frame's axes to the factor frame, on
`constraint_cov`'s **rotation-first** axes `[rx,ry,rz,tx,ty,tz]` (`eigvecs`, stored
translation-first per `01 §3.4`, are permuted onto them via `to_rotfirst`). If
`eigvecs` is present (non-axis-aligned degeneracy,
e.g. a tunnel at 30°), use it directly as the rotation of the degenerate
sub-space instead of an axis-aligned `R6`:

$$
\Sigma'_{\text{meridian}} \;=\; R_6\,\Lambda^{1/2}\,(R_6^\top\, \Sigma_{\text{meridian}}\, R_6)\,\Lambda^{1/2}\,R_6^\top,
\quad \Lambda = \mathrm{diag}(\lambda_{tx},\lambda_{ty},\lambda_{tz},\lambda_{rx},\lambda_{ry},\lambda_{rz}).
$$

`constraint_cov` is already rotation-first (GTSAM order, `01 §6.1`), so the matrix is
**not** reordered here — only the translation-first `observability` score (and any
`eigvecs`) are permuted onto its axes. (`LoopConstraint.cov`, a translation-first
`PoseCov6`, still goes through `reorder_meridian_to_gtsam`, §12.)

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
`0.05`) and `backend.degenerate_lock` is set, L3 hard-sets that axis's `λ_k =
ρ_max` regardless of the smooth law — a belt-and-braces guard against near-zero
eigenvalues the smooth curve might under-inflate (`grounding 09 §11.1`'s defensive
re-inflation) — and emits a degeneracy marker (§14).

---

## 5. The restart edge (the only IMU factor)

When the front-end restarts its sliding window (divergence, observability
collapse, IMU saturation; the window-restart fallback of `00 §7.4`, `01 §6.4`), it
cannot summarise a trustworthy relative-motion marginal across the gap. It instead
emits a packet with `constraint_kind == ImuPreintegration` and a populated
`imu_summary : ImuPreintegrationSummary` (`01 §6.5`). L3 turns this into a single
`gtsam::CombinedImuFactor` — the **only** place in the entire system an IMU factor
enters the graph (`grounding 09 §8.4`, "Contract rule #2 — Fallback").

### 5.1 The summary (already a boundary type — `01 §6.5`)

`ImuPreintegrationSummary` carries `delta_R / delta_v / delta_p`, the bias
linearization point (`bias_g_lin`, `bias_a_lin`), the first-order bias Jacobians
(`dR_dbg`, `dv_dbg`, `dv_dba`, `dp_dbg`, `dp_dba`), a 9-DoF `preint_cov`, and
`gravity_mag`. This is exactly what GTSAM's preintegration needs (Forster et al.
T-RO 2017, the math GTSAM's `Preintegrated(Combined)Measurements` implements —
`grounding 09 §7`). L3 rebuilds a `PreintegratedCombinedMeasurements` (PIM) from
these fields rather than re-integrating raw IMU (which never crosses the boundary
— `01 §6.5`).

```cpp
namespace meridian::backend {
gtsam::PreintegratedCombinedMeasurements
pim_from_summary(const ImuPreintegrationSummary& s, const BackendConfig::Imu& cfg);
}
```

`CombinedImuFactor` is preferred over the older `ImuFactor` + separate bias
between-factor because the bias random-walk is baked in (`grounding 09 §6.3, §7.2`).

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
  # and ISAM2 throws IndeterminantLinearSystem (grounding 09 §8.5, §11.2).
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

The `CombinedImuFactor` includes the between-bias random-walk term. Defaults are
grounded in the references and overridable per-platform via `CalibrationSet`
(`01 §5.3` carries `imu_acc_noise`, `imu_gyr_noise`, `imu_acc_bias_rw`,
`imu_gyr_bias_rw`; these feed `PreintegrationCombinedParams`, `grounding 09 §7.2`):

| Quantity | Default | Grounding |
|----------|---------|-----------|
| accel noise `σ_a` | 0.1 m/s²/√Hz | `FAST_LIO/include/IMU_Processing.hpp` (`cov_acc` init), `config/ouster64.yaml` (`acc_cov: 0.1`) |
| gyro noise `σ_g` | 0.1 rad/s/√Hz | `IMU_Processing.hpp` (`cov_gyr` init), `ouster64.yaml` (`gyr_cov: 0.1`) |
| accel bias RW `σ_ba` | 1e-4 m/s³/√Hz | `IMU_Processing.hpp` (`cov_bias_acc`), `ouster64.yaml` (`b_acc_cov: 0.0001`) |
| gyro bias RW `σ_bg` | 1e-4 rad/s²/√Hz | `IMU_Processing.hpp` (`cov_bias_gyr`), `ouster64.yaml` (`b_gyr_cov: 0.0001`) |

The gravity vector for the PIM is ENU `(0,0,-9.81)` consistent with the `map`
frame (`grounding 09 §7.2` flags `n_gravity` sign/axis as a thing to pin to the
chosen world frame — Meridian's `map` is gravity-aligned ENU, `00 §2.2`).

> Verify the exact line numbers when implementing; `IMU_Processing.hpp` sets these
> in the `ImuProcess` constructor and `ouster64.yaml` lists them under the IMU
> covariance keys. FusionPortable / M2DGR IMU datasheets (`Meridian/docs/DATASET.md`)
> provide the per-platform overrides that flow in via `CalibrationSet`.

### 5.4 Returning to normal

The restart edge is a **bridge, not a mode**. The next keyframe (assuming the
window recovered) carries a clean `RelativeBetween` and is added as a normal
`BetweenFactor` (§4). The local `V`/`B` introduced for the bridge become
candidates for marginalization at the next update (§11) unless
`backend.keep_inertial` is true — keeping the steady-state graph pose-only, the
transient-`V`/`B` design of `grounding 09 §12.2`.

---

## 6. GNSS factors & the GNSS-origin variable

`GnssFix` (`01 §4.4`) carries WGS84 `lat/lon/alt`, an ENU `cov_enu` (3×3 m²), a
`fix` quality enum, and `num_sats`. It reaches L3 via `IBackEnd::add_absolute(fix,
nearest_kf_id)` (`01 §7.4`) — the front-end/pipeline supplies the nearest
keyframe id. (Spec 01 also allows GNSS folded into an `AbsolutePrior` packet;
when that happens L3 routes it through the same machinery below using
`constraint_cov` as the position covariance.)

### 6.1 The origin variable `G`

GNSS arrives in a local-ENU tangent plane whose origin is the first valid fix
(`00 §2.2`: `map` is global, gravity-aligned, ENU-ish). The transform
`G = T_map_enu` is a graph variable so the residual misalignment between the SLAM
`map` frame and the GNSS datum is *estimated*, not assumed. `G` is seeded by a
weak `PriorFactor<Pose3>` near identity (translation prior tight, yaw prior loose
— yaw between `map` and ENU is least observable until motion accrues). Modelling
the datum alignment as a variable rather than baking GNSS into a unary world prior
is the cleaner of the two patterns the dossier lists (`grounding 09 §6.3` GNSS via
`PriorFactor`/`GPSFactor`; here promoted to an estimated origin so the residual
misalignment is observable and correctable).

### 6.2 The GNSS factor

`add_absolute` first geodetic-projects `fix` into the ENU datum (the L3 adapter
owns the datum, `01 §4.4`), yielding `p_enu ∈ ℝ³` with covariance `cov_enu`. With
antenna lever arm `l = T_body_gnss.t` (from `CalibrationSet`):

$$
r_{\text{gnss}} \;=\; G^{-1}\bigl(X(i)\cdot l\bigr) \;-\; p_{\text{enu}},
$$

where `X(i)·l` is the antenna position in `map`, `G^{-1}(\cdot)` maps it into the
ENU datum, and the result is a 3-vector residual with noise `cov_enu`. This is a
custom `NoiseModelFactor2<Pose3, Pose3>` over `X(i)` and `G` (the lever-arm + datum
variant of `grounding 09 §6.3`'s `GPSFactor`). Its analytic Jacobians are the
standard `Pose3`-action Jacobians (GTSAM's `Pose3::transformFrom` /
`transformTo` return the `H` blocks; chain through `G^{-1}`).

```cpp
class GnssFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
  gtsam::Point3 lever_, meas_enu_;
 public:
  gtsam::Vector evaluateError(const gtsam::Pose3& Xi, const gtsam::Pose3& G,
                              gtsam::OptionalMatrixType H1,
                              gtsam::OptionalMatrixType H2) const override {
    gtsam::Matrix36 Ha, Hg_to;  gtsam::Matrix33 Ham;
    gtsam::Point3 ant_map = Xi.transformFrom(lever_, H1 ? &Ha : nullptr);
    gtsam::Point3 ant_enu = G.transformTo(ant_map, H2 ? &Hg_to : nullptr,
                                          H1 ? &Ham : nullptr);
    if (H1) *H1 = Ham * Ha;     // chain: d(ant_enu)/dXi
    if (H2) *H2 = Hg_to;        // d(ant_enu)/dG
    return ant_enu - meas_enu_;
  }
};
```

### 6.3 Switchable + gated + robust

GNSS is **switchable** (`backend.gnss_enabled`, `[hot]`) and **robustified**:
multipath / NLOS produce gross outliers. Per the dossier's incremental-path
recommendation, GNSS uses a **Huber** robust kernel inside iSAM2 by default —
Huber is convex and therefore safe inside the incremental IRLS solve
(`grounding 09 §11`, point 2; §14 "Robustness"), with the heavier GNC reserved for
loops (§8). Gating before the graph:

- **Quality gate.** Drop a fix with `trace(cov_enu) > backend.gnss_max_cov`, or
  `fix == None`. The `fix` enum and `cov_enu` together let the kernel weight an
  `SPP` fix far below an `RTK_Fixed` one (`01 §4.4`; `grounding 09 §11` "gate by
  GNSS quality before adding the factor").
- **Deferred origin activation.** Buffer GNSS factors until cumulative baseline
  exceeds `backend.gnss_min_baseline` (default 5 m), so `G` is estimated from a
  well-spread set of fixes, not a single point (yaw observability, §6.4).

### 6.4 Yaw observability & origin locking

Until the platform moves enough, `G`'s yaw is weakly observable and can wander:

- **Heading lock.** If `backend.gnss_lock_yaw` is set (e.g. dual-antenna heading
  available), pin `G`'s roll/pitch to gravity and fix yaw by a tight prior,
  effectively reducing `G` to translation + yaw.
- The deferred activation above is the default safeguard when no external heading
  exists.

---

## 7. PCM pre-filter (loop outlier rejection)

Loop closures are the highest-stakes factors: one wrong loop welds two unrelated
places and ruins the map (`grounding 09 §9.1`). L5 already runs Scan Context++ →
STD/BTC → GICP and hands L3 a verified `LoopConstraint` (`01 §7.6`: `from_id`,
`to_id`, `T_from_to`, `cov`, `fitness`). L3 gates every loop through **Pairwise
Consistent Measurement (PCM)** maximisation *before* it enters the graph, then GNC
inside the graph as a second line of defence (§8). PCM and GNC are complementary —
PCM picks a mutually consistent set, GNC adds graceful residual robustness
(`grounding 09 §11`, point 4).

### 7.1 Pairwise consistency

Two loops $l_{pq}=(X_p\!\to\!X_q,\,Z_{pq})$ and $l_{rs}=(X_r\!\to\!X_s,\,Z_{rs})$
are *pairwise consistent* if the cycle they form with the (trusted) odometry chain
closes within noise. Using current estimates $\hat X_\bullet$:

$$
\epsilon_{pq,rs} = \mathrm{Log}\!\Bigl(Z_{pq}^{-1}\,\hat X_p^{-1}\hat X_r\,Z_{rs}\,\hat X_s^{-1}\hat X_q\Bigr),
\qquad
d^2_{pq,rs} = \epsilon^\top\,\Omega\,\epsilon,
$$

with $\Omega$ the combined information of the two loop covariances (`cov.M`
inverted; both are `PoseCov6` in Meridian order). Consistent iff
$d^2 \le \chi^2_{6,\alpha}$ with $\alpha = $ `backend.pcm_chi2_alpha` (default
0.99 → threshold ≈ 16.81 for 6 DoF).

### 7.2 Maximum-clique consistent set

Build a consistency graph: nodes = pending loops, edge = pairwise consistent. The
largest mutually consistent subset is the **maximum clique**; only its members are
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
     noise  = noiseModel::Robust( gnc_kernel(), Gaussian::Covariance(Sig_gt) )  # §8
     fid    = new_graph.add( BetweenFactor<Pose3>(keyX(lc*.from_id),
                                                  keyX(lc*.to_id),
                                                  to_gtsam(lc*.T_from_to), noise) )
     loop_factor_index[lc*] = fid; mark_in_graph(lc*)
  for lc# in in_graph \ clique:                               # newer clique evicted it (rare)
     schedule_removal(loop_factor_index[lc#])                 # §9.5
```

This maps to `IBackEnd::add_loop_constraint` (`01 §7.4`): the method enqueues the
constraint; the back-end thread runs the PCM step above on the next iteration.

> **Why PCM *and* GNC.** PCM is a *combinatorial set-level* pre-filter (which
> loops are mutually consistent, using current estimates); GNC is a *continuous
> factor-level* robustifier (per-factor weights during optimisation). PCM stops a
> wrong loop from ever distorting the linearisation point; GNC handles residual
> mis-weighting and borderline fixes (`grounding 09 §9–§11`). Course §06 (loop
> closure) + §11 (robust).

---

## 8. Robust kernels & GNC (loops + GNSS)

Loop factors are wrapped in a GTSAM `noiseModel::Robust` kernel. GNSS uses Huber
in the incremental path (§6.3); loops use the heavier **GNC** machinery because a
wrong loop is the single most damaging factor (`grounding 09 §10–§11`). The
default loop kernel is **Cauchy** (matching spec 00 §8.2's `robust: gnc_cauchy`),
selectable to Geman–McClure or Truncated-Least-Squares via `backend.robust_kernel`.

Full Graduated Non-Convexity (course §11; `grounding 09 §10`) anneals a control
parameter `μ` from a near-convex surrogate toward the target robust cost,
recovering per-factor weights `w ∈ [0,1]` via the closed-form GM/TLS weight
update. GTSAM's batch `GncOptimizer` re-runs the whole schedule per solve, which
is **batch, not incremental** — incompatible with single-shot iSAM2 updates
(`grounding 09 §10.3, §11`: "GNC is a batch optimizer … you cannot just run GNC
inside `ISAM2::update`"). Meridian amortises it across iSAM2 steps:

- A new loop factor enters with a **conservative** kernel (high `μ`, near-convex)
  so it cannot immediately dominate.
- Over the next `backend.gnc_anneal_steps` (default 5) iSAM2 updates, step `μ` on
  the schedule (`grounding 09 §10.2`: `muStep`, direction per loss type),
  recompute each factor's weight from its current whitened residual, and re-insert
  the factor with the updated robust noise (remove-by-index + re-add, §9.5).
- A factor whose converged weight `w < backend.gnc_reject_w` (default 0.1) is
  **removed**; if it is a loop, it returns to the PCM pending buffer marked
  "GNC-rejected" so it is not retried immediately (FM-6).

This keeps each `ISAM2::update` cheap while still recovering robust weights. The
inlier cost threshold $\bar c^2$ is set from the chi-square quantile for the factor
dimension (`setInlierCostThresholds`, `grounding 09 §10.3`: for 6-DoF at 99%,
$\bar c^2 = \chi^2_{6,0.99}$); odometry between-factors and the anchor are declared
*known inliers* and never reweighted.

> **Why not in-graph switch variables.** Sünderhauf-style switchable constraints
> (`grounding 09 §9`) add one real switch variable per loop. In an incremental
> graph that grows the Bayes tree and makes that region non-convex → more
> relinearization churn (`grounding 09 §11`, "Why not switchable-constraint
> variables in iSAM2?"). The amortised-GNC-then-commit approach above gets the
> outlier rejection without the extra variable, so Meridian uses it.

---

## 9. iSAM2 update, relinearization & the correction outputs

### 9.1 ISAM2 configuration

```cpp
gtsam::ISAM2Params p;

// Per-symbol-type relinearize thresholds (grounding 09 §4.3): a Pose3's rad/m
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

These follow the dossier's spec-author checklist (`grounding 09 §4.2/§4.3, §14`):
**Dogleg** rather than Gauss-Newton, because the visual + GNSS factors can sit far
from their linearisation point at init and a trust region bounds the step
(`grounding 09 §4.2` `optimizationParams`, §11.2 "Dogleg (bounded step)" against
loop-closure relinearization spikes). **`relinearizeSkip = 1`** so every update
checks deltas (LIO back-ends want this). **`findUnusedFactorSlots = true`** keeps
the factor index compact under loop/GNSS churn. **`evaluateNonlinearError`** feeds
the chi-square health alarm.

`factorization` defaults to **CHOLESKY** (`grounding 09 §4.2`: CHOLESKY normally,
QR only on indeterminate-matrix exceptions from degeneracy). Observability
inflation (§4.3) already keeps degenerate axes from going rank-deficient; if a
corridor still trips `IndeterminantLinearSystemException`, FM-3 flips to QR and
retries. The `g` (GNSS-origin) threshold is tighter so it relinearizes eagerly
while still settling.

### 9.2 The update step (`IBackEnd::optimize`)

```text
optimize() -> GraphUpdate:
  result   = isam2_.update(new_graph, new_values, remove_indices)
  for k in 1..cfg.isam2_extra_iters: isam2_.update()      # extra Dogleg passes after a big change
  estimate = isam2_.calculateEstimate()                    # full only when L4 needs it (§9.3)
  gu       = build_graph_update(estimate, result)          # §9.3
  log_graph_summary(result); publish_debug()               # §14
  new_graph.clear(); new_values.clear(); remove_indices.clear()
  return gu
```

iSAM2 re-eliminates only the part of the Bayes tree touched by the new factors and
any variable whose linearisation point moved past its `relinearizeThreshold` (fluid
relinearization, course §09; `grounding 09 §4`). Steady-state per-keyframe cost is
near-constant; a loop touches the cycle's variables (a larger sub-tree) but not the
whole graph — the iSAM2 win over batch (`grounding 09 §2.2`). Because iSAM2 does
only **one** Gauss-Newton/Dogleg step per `update`, a big loop benefits from 1–2
extra empty `update()` passes to converge (`isam2_extra_iters`,
`grounding 09 §14` "Optimizer object").

In the hot path, publishing the latest pose only needs
`calculateEstimate<Pose3>(latestKey)` (cheap partial back-substitution); the full
`calculateEstimate()` is reserved for when L4 must re-integrate after a loop moved
many poses (`grounding 09 §5`, partial state recovery).

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
filled from the iSAM2 marginal when `backend.emit_moved_cov` is set (it is
moderately expensive). `loop_closed` is set when this update admitted a loop
factor. L4 consumes `moved` to clear-and-rebuild the affected nvblox TSDF region
from the `KeyframeStore` clouds at the corrected poses (`01 §7.5`) — L3 names the
moved keyframes; it never touches voxels.

**The `map→odom` correction** spec 00 §3 calls a "`PoseCorrection`" is recovered
by any consumer from L3's outputs without a new boundary type. For the latest
keyframe `n`, with the stored odom hint `T_odom_body(n) = kf_record[n].T_ref_body`
and the optimised `X(n) = T_map_body(n)`:

$$
T_{\text{map}\,\text{odom}} \;=\; X(n)\cdot \bigl(T_{\text{odom}\,\text{body}}(n)\bigr)^{-1}.
$$

The live front-end stays smooth in `odom` (`01 §7.3`); the global correction is
this transform, refreshed each keyframe. (`current_map_to_ref` in §4.2 returns
exactly this, cached.) The refined extrinsics travel separately as the versioned
`CalibrationSet` snapshot via `refined_calibration()` (§10, `01 §5.3`).

### 9.4 `corrected_trajectory()`

Returns the `map`-frame `StampedPose` list (`01 §7.4`, `01 Appendix A`:
`{stamp, kf_id, T_map_body}`) for every keyframe still represented in the estimate.
Used by L6 / evaluation / TF publication in the wrapper.

### 9.5 Factor removal

iSAM2 removes factors by internal index (`ISAM2::update(..., removeFactorIndices)`).
With `findUnusedFactorSlots = true` (§9.1) the freed slots are reused so the index
stays compact under churn (`grounding 09 §4.2, §14`). L3 tracks `factor → internal
index` for every removable factor (loops, GNSS) so PCM eviction (§7.2) and GNC
rejection (§8) can pull a factor cleanly. **Odometry between-factors, restart IMU
factors, and the anchor prior are never removed.**

---

## 10. Online extrinsic refinement

Online extrinsic refinement is **on by default** (`00 §3`, L2/L3 own the online
calibration variables). L3 is the **authority** on refined extrinsics (`01 §5.3`):
it holds them as graph variables and publishes refined values to L2.

### 10.1 Model

Each refinable sensor `s` (those whose `Extrinsic::refine_online == true`,
`01 §5.2`) gets an `E(s) = T_body_sensor(s)` `Pose3` variable, seeded by the
offline calibration prior with a tight `PriorFactor<Pose3>` (sigma
`backend.extrinsic_prior_sigma`, default 1e-3 m / 1e-3 rad — itself derived from
`Extrinsic::prior_cov` when available; `grounding 09 §14`: "weak priors on each
`E(s)` until observable"). When refinement is active the prior is *loosened* to
`extrinsic_refine_sigma` (default 1e-2) and the relevant sensor factors are made
to depend on `E(s)` (a GNSS lever arm can likewise be promoted from a constant to
`E(gnss).translation()`).

L3 holds the refined extrinsics as graph variables and publishes updated values as
a versioned `CalibrationSet` snapshot that the front-end picks up at keyframe
boundaries via `IFrontEnd::set_calibration`. The version counter lets L2 detect
"calibration changed, reset linearization" (`01 §5.3`). The snapshot crosses L3→L2
as Shared-immutable, never as live shared mutable — no data race.

### 10.2 Observability & safeguards

Extrinsics are weakly observable without excitation (the platform must rotate
*and* translate to separate `E` from pose; `grounding 09 §11.2`: an extrinsic with
no excitation is a canonical indeterminate-system cause):

- **Excitation gate.** Update/relinearize `E(s)` only once cumulative rotation and
  translation since enabling exceed `extrinsic_excite_rot` / `extrinsic_excite_trans`;
  before that the tight prior dominates and `E` stays pinned. (A FAST-LIO2-style
  filter keeps the LiDAR-IMU extrinsic in-state but pinned by default for the same
  reason — `extrinsic_est_en`, `config/ouster64.yaml`; carried in `state_ikfom`,
  `use-ikfom.hpp:15–16`.)
- **Convergence freeze.** Once `E(s)`'s marginal covariance (queried via
  `ISAM2::marginalCovariance`, §13) falls below `extrinsic_freeze_cov`, re-tighten
  its prior to the converged value and stop treating it as free, removing it from
  future relinearization to save cost — and publish the frozen value in the next
  calibration snapshot.
- **Sanity clamp.** If `E(s)` strays from the offline prior by more than
  `extrinsic_max_dev`, reject the update, re-pin to the prior, and warn (FM-5).

---

## 11. Marginalization for long sessions

Unbounded graphs grow without limit; tactical sessions run for hours
(`grounding 09 §11.2` "Unbounded growth", §12). Meridian's design follows the
dossier's explicit recommendation (`grounding 09 §12.2`): **keep a full `ISAM2`
pose graph** — poses are cheap (one `Pose3` per ~1 m; even multi-km missions are
only tens of thousands of poses, well within iSAM2, and loop closures must be able
to attach to old keyframes, which a fixed-lag smoother would have marginalized
away) — and **marginalize only the transient inertial variables** (`V`/`B`), which
are needed only locally for the restart bridge (§5).

### 11.1 What stays active

- The full keyframe **pose graph** stays in iSAM2 (so any past keyframe can still
  receive a loop closure). The anchor keyframe, the GNSS-origin `G`, and any
  non-frozen extrinsic `E(s)` are likewise never marginalized.
- Transient `V`/`B` from a restart bridge (§5) are marginalized once the next
  normal keyframe is added, unless `backend.keep_inertial`.

### 11.2 Mechanism

Marginalizing variable set $M$ (the restart bridge's local `V`/`B`) yields, via the
Schur complement on the linearised system, a dense Gaussian factor on the
**separator** $S$ (retained variables that shared a factor with $M$):

$$
\Lambda_S^{+} \;=\; \Lambda_{SS} - \Lambda_{SM}\,\Lambda_{MM}^{-1}\,\Lambda_{MS},
$$

re-added as a `gtsam::LinearContainerFactor` at the current linearisation point
(`grounding 09 §12.1`: marginalization = Schur-complement leaving a dense linear
prior; exact for the linearisation at marginalization time, but frozen there).
Meridian marginalizes `V`/`B` by **simply not retaining them past the restart
window** — they are introduced only on fallback intervals (§5), so they are
naturally transient and need no fixed-lag smoother for the pose graph
(`grounding 09 §12.2`, "Practical recommendation"). This avoids the documented
sharp edges of `gtsam_unstable`'s `IncrementalFixedLagSmoother` (one `update` per
timestamp, segfaults marginalizing a prior-only variable, key/timestamp drift —
`grounding 09 §12.1`).

If a *very* long mission makes even the pose graph too large, the escalation is
**keyframe culling / graph sparsification** (drop redundant keyframes in
overlapping-view regions, merging their constraints) rather than fixed-lag
marginalization of poses (`grounding 09 §12.2`) — a deferred utility, not a normal
path.

### 11.3 Caveats

- The marginal prior is computed at a fixed linearisation point and cannot be
  relinearized later (`grounding 09 §12.1`). Because only transient `V`/`B` are
  marginalized — never poses a future loop might move — this freezing is harmless:
  the persistent pose graph stays fully relinearizable.
- The `KeyframeStore` (`01 §7.5`) is **independent** of graph variable lifetime:
  clouds are kept (for final meshing and possible future loop verification)
  regardless of whether any transient inertial variable was marginalized. Their
  poses track the live pose-graph estimate.

---

## 12. Tangent-ordering adapter (Meridian ↔ GTSAM)

This is a correctness-critical detail — the dossier flags it as the **highest-risk
silent bug** in the whole back-end (`grounding 09 §8.3, §15` open question #3:
"write a unit test; this is the highest-risk silent bug"). **Meridian orders the
6-DoF tangent translation-first: `[ρ(trans); φ(rot)]`** (`01 §3.1`: "orders
translation first, rotation second; $\xi=[\rho;\phi]$"), and `PoseCov6` /
`ObservabilityReport.score` follow that. **GTSAM `Pose3` orders rotation-first:
`[rx,ry,rz, tx,ty,tz]`** (`grounding 09 §6.2`). Every covariance and Jacobian
crossing into GTSAM must be reordered, or the noise model is silently wrong (a
classic block-swap bug; spec 01 §3.1's "pitfall fixed here" warns about exactly
this).

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
at the boundary converts conventions; the core math never does (`01 §3.1` mandate).

> **Mandatory regression test** (`grounding 09 §8.3`): perturb `T_j` along one
> tangent axis and assert the between-factor error grows along the *expected*
> covariance axis. This catches a swapped/rotated covariance the moment it
> appears.

---

## 13. Marginal covariance for the loop pre-filter

L5's place-recognition gate should only attempt expensive GICP verification
against keyframes within the search radius implied by the current pose
uncertainty (`grounding 09 §13`). L3 serves that covariance cheaply.

- **Cheap, from the live graph:** `ISAM2::marginalCovariance(keyX(latest))` returns
  the 6×6 marginal of the latest pose (GTSAM tangent order rot-then-trans; reorder
  to Meridian via §12 before handing out). It reuses the Bayes tree — far cheaper than
  building a `Marginals` object from scratch (`grounding 09 §13.1`). Call it at
  keyframe rate, not per scan.
- The positional 1σ is $\sqrt{\lambda_{\max}}$ of the translation block; L5's gate
  radius is `k_gate · σ_pos` (`grounding 09 §13.2`). The same call serves the
  extrinsic convergence-freeze test (§10.2).

This is the third reason the no-double-counting contract is load-bearing: an
over-confident marginal (from a double-counted hand-off) shrinks the search radius
and **misses loop closures**, causing unbounded drift (`grounding 09 §13.3`).
Keeping the hand-off honest (§3.1) keeps loop closure working.

---

## 14. Debug & introspection

Per the introspection pillar (`00 §10`, "debug in the right places"), L3 makes its
reasoning visible. The structured channel is `BackEndDiagnostics` (`01 Appendix A`:
`num_keyframes`, `num_loops`, `isam_update_ms`, `last_optimize_diverged`); L3
additionally writes the richer telemetry below through the `TelemetrySink`
abstraction (`00 §10.1`) so the wrapper maps it to ROS topics / rviz markers. L3
core emits typed telemetry; it never calls `rclcpp` (`00 §1.1`). The health
signals follow the dossier's monitoring list (`grounding 09 §14` "Health
monitoring": `errorBefore/After`, `variablesRelinearized`, `variablesReeliminated`,
chi-square spikes, indeterminate-system alarms).

### 14.1 Telemetry keys (wrapper-bound, `00 §10.2`)

| Telemetry key | Channel | Cadence | Contents |
|---------------|---------|---------|----------|
| `pose("map/keyframe/<id>")` | pose | kf rate | optimised `X(id)` for path/TF |
| `scalar("backend/chi2")` | scalar | kf rate | total graph χ² from `errorAfter` (graph health, `00 §10.2`; `grounding 09 §14`) |
| `scalar("backend/n_factors")` | scalar | kf rate | factor count by type (between/imu/loop/gnss) |
| `scalar("backend/update_ms")` | timing | per update | `ISAM2::update` time → `BackEndDiagnostics.isam_update_ms` |
| `scalar("backend/relin_count")` | scalar | per update | `variablesRelinearized` / `variablesReeliminated` (loop-thrash detector) |
| `vec("backend/observability/<id>")` | vec | kf rate | per-axis `s_k`, multiplier `λ_k` (§4.3) |
| `marker("backend/loop_edge")` | marker | on event | green = accepted, red = PCM/GNC-rejected, with `from→to` and weight `w` |
| `event("backend/relinearize")` | event | on big relin | #relinearized vars, Bayes-tree size/depth |
| `event("backend/loop_accepted\|rejected_pcm\|rejected_gnc")` | event | on event | provenance + reason |
| `vec("backend/gnss/residual")` | vec | on fix | residual, χ², robust weight, accepted? + `G` estimate |
| `pose("calib/T_body_sensor/<s>")` | pose | on change | `E(s)` estimate, deviation from prior, marginal σ, frozen? (§10) |
| `event("backend/window_restart_bridge")` | event | on restart edge | the §5 bridge made visible |

These align one-to-one with the FAST-LIO-grounded debug map in `00 §10.2 / App. C`
(back-end health, loop markers, per-stage timing) and the principle that
observability and residuals must be *plottable*, not just printed.

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
| FM-3 | iSAM2 `IndeterminantLinearSystemException` | GTSAM throw caught in `optimize` | identify the near-null variable from the exception key (`grounding 09 §11.2`); inflate its connected edges (degeneracy lock §4.3), or switch `factorization` to QR and retry once; if it recurs, freeze the variable and alert (`last_optimize_diverged = true`) |
| FM-4 | Loop storm (PCM thrash) | `pending.size() > pcm_max_nodes` | switch to greedy max-clique; back-pressure L5; newest-fitness-first |
| FM-5 | Extrinsic divergence | `‖E(s) ⊟ prior‖ > extrinsic_max_dev` (§10.2) | reject update; re-pin `E(s)`; disable refine for `s`; warn |
| FM-6 | GNSS jump / multipath | per-fix χ² > gate, or robust `w < reject_w` | drop fix; if sustained, auto-disable GNSS and continue loop-only |
| FM-7 | Loop relinearization spike | `relin_count` / `update_ms` spike after a loop | Dogleg already bounds the step; accept one-frame latency; commit big loops on the back-end thread off the odometry path (`grounding 09 §11.2`) |
| FM-8 | Real-time overrun (`update_ms` over budget) | timing telemetry (§14) | raise `relinearizeSkip`; request larger keyframe spacing from L2; shed debug topics |

FM-2 clamps to a valid PSD covariance and proceeds with the (now-valid)
front-end-supplied uncertainty; it does not substitute an arbitrary isotropic
guess, because the per-axis structure (corridor degeneracy, §4.3) is exactly the
information the back-end must preserve.

---

## 16. Configuration (BackendConfig)

This is the typed sub-tree of `meridian::Config` (`00 §8`) for `meridian_backend`. It
expands spec 00 §8.2's `backend: { kind: isam2, relinearize_thresh: 0.1, robust:
gnc_cauchy }`. `[hot]` = runtime-reconfigurable via the debug-control path
(`00 §10.5`); `[cold]` = restart required. `Config::validate()` (`00 §8.3`) range-
checks these on load.

```cpp
namespace meridian {
struct BackendConfig {
  std::string kind = "isam2";              // [cold] the IBackEnd impl (00 §8.2)

  // --- gauge / anchor ---
  double anchor_sigma                = 1e-4;  // [cold] prior on the first keyframe

  // --- iSAM2 (§9.1; grounding 09 §4.2/§4.3, §14) ---
  int    isam2_relinearize_skip      = 1;     // [hot]
  double isam2_relinearize_thresh    = 0.1;   // [hot] scalar fallback; per-type map is built in code
  int    isam2_extra_iters           = 1;     // [hot] extra Dogleg update() passes after a big change
  bool   isam2_use_qr                = false; // [cold] QR vs Cholesky (Cholesky default; QR on degeneracy)

  // --- observability -> noise (§4.3) ---
  double obs_inflation_max           = 1e4;   // [hot] rho_max
  double obs_inflation_gamma         = 2.0;   // [hot]
  double degenerate_thresh           = 0.05;  // [hot] score below = degenerate
  bool   degenerate_lock             = true;  // [hot] hard-lock worst axis if degenerate

  // --- loops + PCM (§7) ---
  double loop_min_fitness            = 0.5;   // [hot] reject low GICP fitness
  double pcm_chi2_alpha              = 0.99;  // [hot]
  int    pcm_max_nodes               = 64;    // [hot] exact max-clique cap

  // --- robust / GNC (§8) ---
  std::string robust_kernel          = "cauchy";  // [hot] cauchy|gm|tls (00 §8.2: gnc_cauchy)
  bool   gnc_enabled                 = true;       // [hot]
  int    gnc_anneal_steps            = 5;          // [hot]
  double gnc_reject_w                = 0.1;        // [hot]
  double gnss_huber_k                = 1.345;      // [hot] Huber tuning for GNSS (grounding 09 §6.4)

  // --- GNSS (§6) ---
  bool   gnss_enabled                = true;  // [hot]
  double gnss_max_cov                = 25.0;  // [hot] drop fix if trace(cov_enu) above (m^2)
  double gnss_min_baseline           = 5.0;   // [hot] defer G activation
  bool   gnss_lock_yaw               = false; // [hot]

  // --- online extrinsics (§10), on by default ---
  bool   extrinsic_refine            = true;  // [cold] master switch (00 §3 default-on)
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

The back-end runs on its own thread (`01 §2.4`: "a back-end thread (L3 + L5)").
It owns the `ISAM2` instance exclusively; every graph mutation flows through this
single loop, the invariant that makes the no-double-counting contract (§3.1)
enforceable in one place.

```text
backend_thread():
  isam2 = ISAM2(params from cfg)                       # §9.1
  loop forever:
    item = backend_queue.pop()                          # blocking [TS] from L2/L5/L0
    switch item.kind:
      case KEYFRAME:                                    # IBackEnd::add_keyframe (01 §7.4)
        switch item.kf.constraint_kind:                 # §3.1 — single odometry factor
          RelativeBetween:   add_between_edge(item.kf)        # §4
          ImuPreintegration: add_restart_imu_edge(item.kf)    # §5
          AbsolutePrior:     add_anchor_or_absolute(item.kf)  # §3 (first KF) / §6 (GNSS)
      case LOOP:    process_loop(item.lc)               # PCM admit, §7  (add_loop_constraint)
      case GNSS:    buffer_or_add_gnss(item.fix, item.kf_id)  # §6  (add_absolute)
    if gnc_active:        step_gnc_anneal()             # §8
    if should_marginalize(): marginalize_transient()    # §11
    gu = optimize()                                      # §9.2  -> GraphUpdate
    map_thread_queue.push(gu)                            # §9.3  -> L4 re-integration [TS]
    serve_loop_gate_cov()                                # §13   -> L5 marginal cov
    if extrinsics_changed(): publish_calibration_snapshot()  # §10 -> L2 [TS]
    publish_debug()                                      # §14
```

`optimize()`, `corrected_trajectory()`, `refined_calibration()`, and
`diagnostics()` are the `IBackEnd` methods (`01 §7.4`) the pipeline calls; the
loop above is the internal driver that backs them.

---

## 18. Module integration order

This is a single complete system; the list below is only the **compile/integration
order** for bringing the back-end up against its collaborators — not a feature
rollout, and nothing is organised around it. Each item is the same final design.

1. **Variables + the normal between-edge (§2, §4) + anchor prior + iSAM2 wiring
   (§9).** Drive it with `RelativeBetween` packets only; assert the
   single-factor-per-edge invariant and the tangent-ordering regression test
   (§12). This is the spine; everything else attaches to it.
2. **Restart bridge (§5)** so the graph survives a window restart, and the
   transient-`V`/`B` marginalization (§11) that keeps it pose-only afterward.
3. **GNSS (§6)** — the origin variable `G`, the lever-arm factor, Huber gating.
4. **Loop closures (§7 PCM + §8 GNC)** and the marginal-covariance gate served to
   L5 (§13).
5. **Online extrinsics (§10)**, on by default, with the excitation/freeze
   safeguards.

A FAST-LIO2-style iEKF front-end exists only as an optional reference/test oracle
behind `IFrontEnd` (`00 §5.4`); from L3's side it is irrelevant — L3 consumes the
same `KeyframePacket` regardless of which front-end produced it
(`grounding 09 §8`), so the back-end never branches on `frontend_kind` (`01 §6.2`).

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
  `06_mapping.md` (consumes `GraphUpdate` for nvblox clear-and-rebuild), L5 place
  recognition `07_loop_closure.md` (produces `LoopConstraint`, consumes the
  marginal-covariance gate).
- Grounding dossier — **`docs/grounding/09_backend_isam2.md`** (iSAM2/GTSAM
  back-end): Bayes tree §2; reordering §3; fluid relinearization + per-type
  thresholds §4; partial recovery §5; factor/noise API §6; IMU preintegration §7;
  **★ L2→L3 hand-off contract §8 ★**; switchable constraints §9; GNC §10;
  robustness↔incrementality §11; marginalization / fixed-lag §12; marginal
  covariance for the loop pre-filter §13; spec-author checklist §14.
- Course — factor graphs / smoothing §03; iSAM2, Bayes tree, marginalization §09;
  robust / GNC §11; loop closure §06.
- Grounding code — `FAST_LIO/include/use-ikfom.hpp:12–21` (the full inertial state
  Meridian keeps inside L2's window, not at the L3 graph level);
  `FAST_LIO/include/IMU_Processing.hpp` + `config/ouster64.yaml` (IMU / bias noise
  defaults, online-extrinsic-pinned default).

---

*End of spec 05. The L3 back-end consumes only the spec-01 boundary types and
emits only the spec-01 outputs; its sole non-negotiable internal invariant is the
single-odometry-factor-per-edge rule of §3.1, the load-bearing L2→L3 hand-off
contract of `grounding 09 §8`.*
