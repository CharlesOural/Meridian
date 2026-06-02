# 04 — L2 Front-End: Continuous-Time Tightly-Coupled LiDAR-Inertial-Visual-GNSS Estimator

> **Spec status:** normative for the L2 layer. This document specifies **the
> Meridian front-end**: a real-time, fixed-lag, **continuous-time (CT)** estimator
> that represents the sensor trajectory as a **split SO(3)×ℝ³ cubic B-spline** and
> fuses, in **one** sliding-window non-linear least-squares solve, a **single
> LiDAR** (direct point-to-plane at each point's true time), an **IMU**
> (derivative residual on the spline), a **camera** (FAST-LIVO2-style sparse-direct
> photometric, LiDAR-depth, exposure-compensated), and **GNSS** (conservative
> absolute position). It distills that trajectory into a stream of
> **`KeyframePacket`** for the back-end (L3). It implements the `IFrontEnd`
> interface declared in spec 01 §7.3 and obeys the boundary, ownership, and
> threading contracts fixed there.
>
> **This IS the design — one complete system.** There is no "v1 filter we ship
> first" and no phased feature rollout. The CT LIVO+GNSS estimator described here
> is the front-end. A FAST-LIO2-style iterated error-state Kalman filter (iEKF)
> exists **only** as an optional offline reference/oracle for differential testing
> (§5.6, one paragraph); it is never a milestone and nothing here is organised
> around it.
>
> **It implements, it does not redefine.** The boundary value types
> (`KeyframePacket`, `NavState`, `ObservabilityReport`, `LidarScan`,
> `CameraFrame`, `ImuSample`, `GnssFix`, `CalibrationSet`, `FrontEndDiagnostics`)
> and the `IFrontEnd` interface are **canonically defined in spec 01** and **must
> not** be redefined here. Where this spec shows a struct/signature it is a
> *quotation* of spec 01 for the reader's convenience; if the two ever disagree,
> **spec 01 wins** and this document is the bug. New fields require amending spec
> 01 first (spec 01 §1, R-rule).
>
> **Resolved defaults (system direction).** Deployment target **NVIDIA Jetson
> Orin (CUDA always present)**; ROS 2 **Humble**; C++20. **Estimation frame is
> `imu_link`** ($F_e$); operator-facing poses are re-expressed to `base_link` at
> the ROS edge, not here. **One LiDAR + one IMU + one camera + GNSS.** L2→L3 in the
> steady state is a **single relative `BetweenFactor`** carrying the front-end
> marginal covariance; velocity/biases are seeds/telemetry only
> (`kinematics_included = false`); a GTSAM `CombinedImuFactor` is emitted **only**
> on the window-restart fallback (mutually exclusive with the between). **Bias
> estimation lives in L2.** Online extrinsic refinement is **on by default**
> (gated by observability). Scope **stops at the colourised mesh** (ESDF/semantics
> are deferred hooks elsewhere).
>
> **Cross-refs.** Spec 00 (architecture: `IFrontEnd` seam, deskew feedback edge §7,
> threading §11, debug telemetry §10), spec 01 (all boundary types; tangent
> ordering `[ρ(trans); φ(rot)]`, spec 01 §3.1; `KeyframePacket` §6; `IFrontEnd`
> §7.3), spec 02 (L0 time sync / PTP), spec 03 (L1 preprocessing;
> `ILidarPreprocessor`; deskew is *not* done in L1), spec 05 (L3 back-end: how
> `constraint_kind` becomes a factor), spec 06 (L4 nvblox GPU map /
> `IKeyframeStore` / clear-and-rebuild de-integration), spec 07 (L5 place
> recognition consumes the retained clouds).
>
> **Grounding.** Every algorithmic claim is anchored to the grounding dossiers
> and the reference code on disk: CT spline math —
> `docs/grounding/10_continuous_time.md` (CLINS, Coco-LIC, Sommer et al.,
> basalt-headers); LiDAR residual / association / map —
> `docs/grounding/03_lidar_residual_ikdtree.md` (FAST-LIO2
> `FAST_LIO/src/laserMapping.cpp::h_share_model`, `common_lib.h::esti_plane`);
> photometric residual — `docs/grounding/04_livo2_visual.md` (FAST-LIVO2
> `FAST-LIVO2/src/vio.cpp`); IMU model / deskew —
> `docs/grounding/02_imu_propagation_deskew.md`; manifold/ESIKF (for the oracle and
> for sign conventions) — `docs/grounding/01_state_manifold_esikf.md`. Citations
> are `file:line` and `dossier §`. Read the real files; do not trust memory.

---

## Table of contents

0. [Scope and module bring-up order](#0-scope-and-module-bring-up-order)
1. [State: the split SO(3)×ℝ³ spline, biases, nuisances; frames; notation](#1-state-the-split-so3r3-spline-biases-nuisances-frames-notation)
2. [L1↔L2 handoff and the deskew-by-spline feedback edge](#2-l1l2-handoff-and-the-deskew-by-spline-feedback-edge)
3. [The CT residual set on control points](#3-the-ct-residual-set-on-control-points)
4. [Per-axis observability / degeneracy extraction](#4-per-axis-observability--degeneracy-extraction)
5. [The sliding-window solve, marginalization, and adaptive knots](#5-the-sliding-window-solve-marginalization-and-adaptive-knots)
6. [The KeyframePacket L2 emits](#6-the-keyframepacket-l2-emits)
7. [`IFrontEnd`, threading, and back-end feedback](#7-ifrontend-threading-and-back-end-feedback)
8. [Debug, introspection, timing](#8-debug-introspection-timing)
9. [Parameters](#9-parameters)
10. [Failure modes & mitigations](#10-failure-modes--mitigations)
11. [Library choices and what was rejected](#11-library-choices-and-what-was-rejected)
12. [Consistency checklist](#12-consistency-checklist)

---

## 0. Scope and module bring-up order

L2 is the real-time trajectory estimator. Its inputs are the four ingest streams
of `IFrontEnd` (spec 01 §7.3: `ImuSample`, `LidarScan`, `CameraFrame`,
`GnssFix`); its outputs are the live `NavState` (spec 01 §3.2, sideways, for
telemetry/control/L4 live integration) and the `KeyframePacket` stream (spec 01
§6, the **only** thing handed to L3). One concrete class — `CtFrontEnd` —
implements `IFrontEnd`. There is no second production implementation behind the
seam; the seam exists for clean modularity and for swapping in the test oracle
(§5.6), not for a phased rollout.

**This is not a phased plan.** What follows is purely a sensible *module
compile/integration order* for a single full system, so the build comes up
incrementally without ever being a "reduced product":

1. **Spline kernel + IMU residual** — stand up the split SO(3)×ℝ³ basalt spline
   (§1.2), the analytic derivatives (§1.3), the IMU derivative residual and
   bias random-walk (§3.3), and the Ceres problem skeleton (§5). At this point
   the window is an inertial dead-reckoner with bias estimation — useful only as
   a bring-up checkpoint, not a shippable mode.
2. **LiDAR point-to-plane** — add the direct, deskew-free LiDAR residual (§3.1)
   against the nvblox-adjacent NN structure (§3.1, spec 06). This is LIO.
3. **Sparse-direct visual** — add the FAST-LIVO2 photometric residual with
   LiDAR-depth and exposure compensation (§3.2). This is LIVO.
4. **GNSS + marginalization + observability** — add the conservative GNSS
   residual (§3.4), Schur-complement marginalization (§5.4), and the per-axis
   observability extraction (§4) that modulates the emitted covariance.

Each step is `git`-bisectable and unit-testable in isolation, but the **delivered
front-end is the whole of §1–§6**. The differential-test oracle (§5.6) is wired
in alongside, never in front.

---

## 1. State: the split SO(3)×ℝ³ spline, biases, nuisances; frames; notation

### 1.1 Frames

L2 uses the canonical frames of spec 01 §2.2 (REP-105 aligned). Repeated for
self-containment; spec 01 is authoritative.

| Symbol | `Frame` (spec 01) | Meaning |
|---|---|---|
| $W$ | `Odom` | **L2 odometry world frame** — smooth, drift-prone, continuous; gravity-aligned; origin = first keyframe. **L2 owns `odom`** (spec 01 §2.2). |
| $M$ | `Map` | Global back-end frame. **L3 owns `map`** and the `odom→map` relation; L2 never writes `map`. |
| $G$ | (ENU datum) | GNSS local ENU; the LLA→ENU datum is an L3/system decision (spec 01 §4.4). L2 holds the latest $T_{MW}$/datum snapshot read-only. |
| $F_e$ | `ImuLink` | **Estimation frame** — the body frame whose trajectory the spline represents (spec 01 §2.3; resolved default). The spline *is* $T_{W\,F_e}(t)$. |
| $L$ | `OsSensor0` | The (single) LiDAR sensor frame. |
| $C$ | `CamLink` | Camera optical frame. |
| $B$ | `BaseLink` | Operator-facing body frame; poses are re-expressed to $B$ at the ROS edge, not inside L2. |

A transform $T_{A\,B}\in SE(3)$ maps a point in $B$ to $A$: $p_A=T_{A\,B}\,p_B$
(spec 01 §2.2). The **`KeyframePacket` pose is always $T_{\text{ref}\,F_e}$**
(spec 01 §2.3, §6.2): the estimation-frame pose in a named `ref` frame. Extrinsics
$T_{F_e\,L},\,T_{F_e\,C}$, the GNSS antenna lever-arm $t_{F_e,\text{ant}}$, and IMU
noise densities live in the shared `CalibrationSet` (spec 01 §5.3) — **never** in
the spline state; that keeps the trajectory dimension fixed regardless of online
extrinsic refinement.

> **Single LiDAR.** Meridian estimates one LiDAR + one IMU + one camera + GNSS. There
> is no multi-LiDAR merge, no per-sensor residual stream, no dome-Ouster special
> case. (A second LiDAR would be a future extension behind the same
> `CalibrationSet`/residual interfaces — an extra extrinsic and an extra point
> stream into the same §3.1 residual — and is not designed here.)

### 1.2 The trajectory: a split SO(3)×ℝ³ cubic B-spline

The trajectory is **one continuous function** $T_{W\,F_e}(t)\in SE(3)$, stored as
a **split** cumulative cubic ($k=4$, degree 3, $C^2$) B-spline — a separate
`So3Spline<4>` for rotation $R_{W\,F_e}(t)$ and a separate `RdSpline<3,4>` for
translation $p_{W\,F_e}(t)$ — exactly basalt-headers' `Se3Spline` internals (dossier
10 §2.4, §3). The split representation is the consensus best choice (Sommer et al.:
"using the split representation is better both in terms of trajectory
representation and computation time"; used by CLINS and Coco-LIC; dossier 10
§2.4): it is faster (no SE(3) adjoints in cross terms), avoids the screw-coupling
of a unified SE(3) spline, and cleanly decouples the gyro residual (↔ SO(3) spline)
from the accel residual (↔ ℝ³ spline 2nd derivative).

For $t$ in knot segment $i$ (normalised time $u=u(t)\in[0,1)$), the value depends on
exactly **four** local control points (local support of a cubic; dossier 10 §2.1):

$$
R_{W\,F_e}(t)=R_i\prod_{j=1}^{3}\mathrm{Exp}\!\big(\lambda_j(u)\,d^{R}_j\big),\quad
d^{R}_j=\mathrm{Log}\!\big(R_{i+j-1}^{\top}R_{i+j}\big)\in\mathfrak{so}(3),
$$
$$
p_{W\,F_e}(t)=p_i+\sum_{j=1}^{3}\lambda_j(u)\,d^{p}_j,\quad
d^{p}_j=p_{i+j}-p_{i+j-1},
$$

with cumulative blending weights $\lambda(u)=\tilde M^{(4)}\,[1,u,u^2,u^3]^\top$ and
the cubic cumulative matrix (dossier 10 §2.1, Kim/basalt convention — cross-check
against basalt `RdSpline::computeBlendingMatrix<4,...,true>`):

$$
\tilde M^{(4)}=\tfrac16\begin{bmatrix} 6&5&1&0\\ 0&3&3&0\\ 0&-3&3&0\\ 0&1&-2&1 \end{bmatrix},
\qquad \lambda_0\equiv 1.
$$

Control points are non-uniformly spaced (adaptive knots, §5.5); the non-uniform
behaviour is realised by running the uniform basalt kernels over a **virtual time**
remapping (Coco-LIC's engineering pattern; dossier 10 §5.2), so the analytic
Jacobians stay the vendored, battle-tested ones.

### 1.3 Analytic derivatives (the reason for CT)

Because the basis is polynomial in $u$, **angular velocity, velocity, and
acceleration are closed-form** in the control points and basis derivatives
($\dot\lambda=\tilde M\,\dot u/\Delta t$, $\ddot\lambda=\tilde M\,\ddot u/\Delta
t^2$; dossier 10 §2.3). From the SO(3) spline, the **body angular velocity** the
gyro sees is $\omega_{W\,F_e}(t)=(R_{W\,F_e}^\top\dot R_{W\,F_e})^\vee$, computed by
the $O(k)$ Sommer recurrence (dossier 10 §2.3 Eq. 38/46; basalt
`So3Spline::velocityBody`). From the ℝ³ spline, $\ddot p_{W\,F_e}(t)$ is the world
linear acceleration (basalt `RdSpline::evaluate<2>`). These feed the IMU residual
(§3.3) with **no numerical differentiation** and give the **exact per-point deskew
pose** $T_{W\,F_e}(t_{\text{offset}})$ for every LiDAR point (§2.3) — deskew *is*
the trajectory, the central advantage over the piecewise-constant-$\omega$
backward-propagation of FAST-LIO (`IMU_Processing.hpp:331`; dossier 02 §4).

### 1.4 The non-trajectory state: biases, gravity, exposure

Beyond the control points, the optimised state carries:

* **IMU biases** $b_g,\,b_a\in\mathbb R^3$ — **bias estimation lives in L2**
  (resolved default). Carried as **per-window bias knots** at a coarse cadence,
  interpolated piecewise-linearly, tied by random-walk residuals between
  consecutive knots (§3.3; Coco-LIC Eq. 16, dossier 10 §4.1). There is **no
  separate velocity state** — velocity is $\dot p_{W\,F_e}(t)$, read from the
  spline.
* **Gravity** $g_W\in S^2$ — a 2-DoF direction of fixed magnitude $|g|=9.81$,
  optionally refined once excitation is sufficient (§3.6). Same $S^2$ rationale as
  FAST-LIO (fix magnitude, avoid $|g|$ drift; dossier 01 §3.3, §7.1). Initialised
  $g_W=-\,\overline{a_m}/\|\overline{a_m}\|\cdot 9.81$ from a short static start
  (dossier 02 §2.2).
* **Inverse exposure** $\tau$ — the FAST-LIVO2 affine photometric nuisance
  (`inv_expo_time`; dossier 04 §3), one per camera frame in the window, optimised
  when `exposure_estimate_en` (§3.2).

None of these cross the `IFrontEnd` boundary as live variables. At a keyframe
instant the spline is **evaluated** to produce the `NavState`/`KeyframePacket` pose
(spec 01 §6.2); biases/velocity are exported only as **seeds/telemetry**
(`kinematics_included = false`, §6.4) on the normal path.

### 1.5 Manifold operators & tangent ordering

Right-invariant box-plus/box-minus (spec 01 §3.1; dossier 01 §0): on $SO(3)$,
$R\boxplus\theta=R\,\mathrm{Exp}(\theta)$, $R_1\boxminus R_2=\mathrm{Log}(R_2^{\top}R_1)$
(Rodrigues `so3_math.h:Exp/Log`). Control-point increments use this on each
$\{R_i\}$ via a Ceres SO(3) manifold/`LocalParameterization`; $\{p_i\}$, biases are
Euclidean. **All 6-DoF tangents and 6×6 blocks in anything that crosses the
boundary order translation-first then rotation** (`[ρ;φ]`, spec 01 §3.1). The
spline's native rotation Jacobians are rotation-first inside basalt; the packet
adapter (§6) performs the block-swap when assembling `constraint_cov` (exactly the
reorder FAST-LIO does when packing `pose.covariance`, `laserMapping.cpp:597-606`).

### 1.6 The IMU kinematic model (the residual's reference equations)

Continuous-time strapdown, in $W$, with measured specific force $a_m$ and rate
$\omega_m$ (dossier 02 §1.2; `use-ikfom.hpp:get_f`):

$$
\dot R_{W\,F_e}=R_{W\,F_e}\,[\omega_m-b_g-n_g]_\times,\quad
\dot p_{W\,F_e}=v_W,\quad
\dot v_W=R_{W\,F_e}(a_m-b_a-n_a)+g_W,\quad
\dot b_g=n_{bg},\ \dot b_a=n_{ba},\ \dot g_W=0.
$$

White noises $(n_g,n_a)$ and bias random walks $(n_{bg},n_{ba})$ have densities
$(\sigma_g^2,\sigma_a^2,\sigma_{bg}^2,\sigma_{ba}^2)$ from the `CalibrationSet`
(Allan-derived, spec 01 §5.3; FAST-LIO defaults 0.1, 0.1, 1e-4, 1e-4,
`avia.yaml:14-17`; dossier 02 §1.4). In the CT estimator this ODE is **not
integrated** — it is the equation the spline's analytic derivatives must satisfy,
turned into the residual of §3.3.

---

## 2. L1↔L2 handoff and the deskew-by-spline feedback edge

### 2.1 What L1 hands up (spec 03)

Per spec 01 §4 and spec 03, on the PTP grandmaster timeline (spec 02;
`Timestamp = int64 ns`, spec 01 §2.1):

* **LiDAR:** `LidarScan` (spec 01 §4.2) — `stamp_start`, `sweep_duration`,
  `sensor_frame`, and a Shared-immutable `points` vector of
  `LidarPoint{xyz (sensor frame, NOT deskewed), intensity, t_offset_ns (int32,
  per-point time vs stamp_start), ring, range, …}`. L1 has already done
  range/intensity gating, ring/voxel decimation (`ILidarPreprocessor::process`,
  spec 03; FAST-LIO `point_filter_num`, `preprocess.cpp`) and NaN removal. **Deskew
  is deliberately NOT done in L1** (spec 03 §7.2) — in a CT system deskew is not a
  separate pass at all, see §2.3.
* **Camera:** `CameraFrame` (spec 01 §4.3) — `stamp` (mid-exposure), Shared-immutable
  `data`, `exposure_s`, `gain`. L1 builds the 3-level pyramid (FAST-LIVO2
  `patch_pyrimid_level=3`; dossier 04 §1.3).
* **IMU:** raw `ImuSample{stamp, acc, gyro}` at full rate (spec 01 §4.1; no
  orientation — Meridian never trusts a vendor AHRS).
* **GNSS:** `GnssFix{stamp, lat/lon/alt, cov_enu, fix, num_sats}` (spec 01 §4.4),
  gated by L1.

### 2.2 Measurement intake (no merge-of-N)

There is no `sync_packages`-style merge of multiple LiDARs. Each stream is ingested
asynchronously through its `IFrontEnd::ingest` overload and time-stamped onto the
spline timeline. The window solve (§5) consumes every measurement **at its own
`stamp`/`t_offset_ns`**; asynchrony is native (dossier 10 §1) — IMU at 200–400 Hz,
LiDAR points streaming across the sweep, camera at 10–30 Hz, GNSS at 1–10 Hz all
evaluate the same $T_{W\,F_e}(\cdot)$ at different $t$. The only batching is the
**outer optimisation cadence** (§5.1, ~0.1 s): when the spline has been extended to
cover new measurements and enough new data has arrived, one window solve runs.

> **Known-timing caveat (dossier 10 §1, §9).** A CT spline absorbs *known* timing;
> it does **not** self-correct an unknown inter-sensor clock offset $t_d$ — it would
> faithfully fit the wrong time and bake the offset into the trajectory. Meridian's
> sensors are PTP-disciplined (spec 02), so per-sensor $t_d$ is nominally zero. The
> spline supports adding $t_d$ as an optional optimised parameter (evaluate that
> sensor's residual at $T(t+t_d)$, differentiable since $T(\cdot)$ is; dossier 10
> §9), defaulting to fixed-zero post-calibration. This is an L2 hook, off by
> default, owned with the rest of online calibration (spec 08).

### 2.3 Deskew is the trajectory; the cold-start feedback edge

**The classic circularity** — deskew needs the trajectory over the sweep, but the
trajectory is estimated from the deskewed sweep — **does not arise in steady
state**, because there is no deskew pass: each LiDAR point's residual (§3.1) is
evaluated at $T_{W\,F_e}(t_{\text{offset}})$, its own time, queried from the
current spline. As the window solve iterates, every point is implicitly
re-deskewed because the control points it hangs from move. Deskew is exact and
free (dossier 10 §1; CLINS abstract "simultaneously removes the motion
distortion").

Architecturally L1 still transforms points only through an injected
`IDeskewProvider::poseAt(Time)→SE3` (spec 00 §5.2, §7.3) whose concrete
implementation is **backed by the L2 spline**, so the build graph stays acyclic
(L1 depends only on the interface) while data flows L2→L1→L2. **L2 owns the pose
source.** Two regimes:

* **STEADY (spline-backed).** `poseAt(t)` returns $T_{W\,F_e}(t)$ from spline
  evaluation; any frame L1 needs to deskew for visualisation uses it. The estimator
  itself never calls it — it consumes raw `t_offset_ns` directly in §3.1.

* **COLD-START / RESTART (IMU-only).** Before the spline spans valid data (first
  knots), or after a window restart (§10), there is no trajectory. `poseAt(t)` falls
  back to **IMU-only forward integration** from the last known state (the FAST-LIO
  `UndistortPcl` forward pass, `IMU_Processing.hpp:216-305`; dossier 02 §3–§4):
  integrate the bias-corrected IMU over the sweep with constant-$\omega$/constant-
  accel intra-interval, seed the first control points from it. Require a short
  static start ≥ `init_time_s` (FAST-LIO `INIT_TIME=0.1 s`; dossier 02 §2) to
  initialise gravity and gyro bias. **No keyframe is emitted while the window is
  not yet initialised** (FAST-LIO `flg_EKF_inited` analogue; spec 00 §7.2). Once the
  first full window has converged, the spline takes over the provider.

**Deskew target frame.** Points are *not* re-baked into a single epoch (that is the
discrete-time approximation we are replacing). Each point keeps its raw
`t_offset_ns`; the **retained per-keyframe cloud** (§6.5) is rendered once at
keyframe emission by evaluating the spline at each point's time into the keyframe
estimation frame $F_{e,\text{stamp}}$ — a *true* per-point deskew, stored compactly
as body-frame points + the keyframe pose so a corrected pose re-places them without
re-querying the spline.

### 2.4 Contract summary

```
L1 ──(undeskewed LidarScan + ImuSample + CameraFrame + GnssFix)──▶ L2
L1 ◀──(IDeskewProvider::poseAt, backed by L2 spline; IMU-only at cold start)── L2
L2 ──(KeyframePacket)──▶ L3            (the only L2→L3 value, spec 01 §6)
L3 ──(PoseCorrection / refined CalibrationSet snapshot)──▶ L2   (spec 01 §7.3, §5.3)
```

Deskew is **owned by L2** and is intrinsic to the spline; L1 publishes raw per-point
times only. This is the documented break of the circular dependency (spec 00 §7).

---

## 3. The CT residual set on control points

Every residual is a function of the spline $T_{W\,F_e}(t)$ evaluated at the
measurement's own time $t$, plus biases/gravity/exposure. Its Jacobian w.r.t. the
active control points is the **measurement Jacobian** (below) chained through the
**spline Jacobian** $\partial T_{W\,F_e}(t)/\partial\delta c_{i+m}$ (basalt's
analytic Jacobians; dossier 10 §3). Each measurement touches only **4 control
points per spline** ⇒ the window Hessian is **banded** ⇒ the solve is real-time
(§5.2). For each residual: measurement model, residual, weight, code/dossier anchor.

### 3.1 LiDAR point-to-plane (direct, deskew-free)

FAST-LIO2's *direct* registration adapted to CT (dossier 03 §1–§2; Coco-LIC Eq.
12–13, dossier 10 §4.2). No edge/planar feature extraction.

**Per-point world transform at true time.** For LiDAR point $\mathbf p^{L}_j$ with
offset $t_j$, evaluate the spline at $t_j$ and fold in the extrinsic
$T_{F_e\,L}$ (from `CalibrationSet`):
$$
\mathbf p^{W}_j = R_{W\,F_e}(t_j)\big(R_{F_e\,L}\,\mathbf p^{L}_j + t_{F_e\,L}\big) + p_{W\,F_e}(t_j).
$$
Because every point uses its own $t_j$, this **is** the deskew (§2.3).

**Association.** Query the map (spec 06; nvblox is the map backend, but the
front-end's nearest-neighbour query for plane fitting uses a vendored **ikd-Tree**
or adaptive voxel-hash over the recent point set — the role
`ikdtree.Nearest_Search` plays in FAST-LIO, `laserMapping.cpp:670`) for
`num_match_points = 5` neighbours. Reject if the farthest squared distance
> `max_match_dist_sq = 5` m² (`:671`). Fit a plane $(\mathbf n, d)$, $\|\mathbf
n\|=1$, by 5-point QR least-squares of $A\mathbf u=-\mathbf 1$ (`esti_plane`,
`common_lib.h:241-247`; dossier 03 §2.1); reject unless **all 5** neighbours lie
within `plane_thresh = 0.1` m of the plane.

**Residual** (scalar signed point-to-plane distance):
$$
r^{\text{lid}}_j = \mathbf n^\top \mathbf p^{W}_j + d
$$
(`pd2 = pabcd·point_world`, `laserMapping.cpp:680`; dossier 03 §2.2).

**Measurement Jacobian w.r.t. the pose at $t_j$** (then chained through the
spline). With $\mathbf q_j = R_{F_e\,L}\mathbf p^{L}_j + t_{F_e\,L}$:
$$
\frac{\partial r_j}{\partial \delta p_{W\,F_e}(t_j)} = \mathbf n^\top,\qquad
\frac{\partial r_j}{\partial \delta R_{W\,F_e}(t_j)} = -\,\mathbf n^\top R_{W\,F_e}(t_j)\,[\mathbf q_j]_\times
= \big([\mathbf q_j]_\times R_{W\,F_e}(t_j)^\top\mathbf n\big)^\top,
$$
the exact FAST-LIO row (`C=R^\top n`, `A=[q]_× C`; `laserMapping.cpp:738-739`;
dossier 03 §2.4). The extrinsic columns ($B=[\mathbf p^L_j]_\times R_{F_e\,L}^\top
C$, $C$) feed the **online extrinsic variable** $T_{F_e\,L}$ when refinement is
active (§3.6).

**Robust weight.** FAST-LIO's range-aware acceptance $s=1-0.9\,|r_j|/\sqrt{\|\mathbf
p^{L}_j\|}$, keep iff $s>0.9$ (`laserMapping.cpp:681-683`; dossier 03 §2.2) as the
inlier gate, **plus** a robust kernel (Huber, GNC-able; spec 11 / GTSAM
`noiseModel::Robust`) on the survivors so a few bad correspondences cannot dominate
the NLLS.

**Measurement covariance.** Range- and incidence-aware (improving on FAST-LIO's
isotropic `LASER_POINT_COV=0.001`, `laserMapping.cpp:64`):
$\sigma^2_{\text{lid}}(j)=\sigma_0^2+\sigma_r^2\|\mathbf p^{L}_j\|^2+\sigma_\theta^2(1-|\mathbf n^\top\hat{\mathbf r}_j|)$,
$\hat{\mathbf r}_j$ the ray direction (grazing-incidence inflation).

### 3.2 Sparse-direct visual photometric (FAST-LIVO2)

FAST-LIVO2's *sparse-direct* visual residual (dossier 04 §2–§5; `vio.cpp`). No
descriptors, no feature matching, **no triangulation** — the **LiDAR map points
double as visual map points** (unified voxel hash; dossier 04 §1.4), each carrying
three-level $8\times8$ reference patches (`patch_size=8`, 3 pyramid levels) and the
reference pose/exposure. Depth always comes from the LiDAR map, so there is no VIO
initialisation problem (dossier 04 §7).

**Measurement model.** Project map point $\mathbf P^W$ into the current camera using
the spline pose at the **image** stamp $t$ and the camera extrinsic
$T_{C\,F_e}$ (from `CalibrationSet`):
$\mathbf P^C = R_{C\,F_e}\big(R_{W\,F_e}(t)^\top(\mathbf P^W-p_{W\,F_e}(t))\big)+t_{C\,F_e}$,
$\mathbf u=\pi(\mathbf P^C)$ (`world2cam`). The **patch photometric residual** for
pixel offset $\Delta$, with the inverse-exposure affine model (dossier 04 §3;
`vio.cpp:1619-1621`):
$$
r^{\text{vis}}_{\Delta} = \tau_{\text{cur}}\,I_{\text{cur}}(\mathbf u+\Delta) - \tau_{\text{ref}}\,P^{\text{warp}}[\Delta],
$$
$\tau$ the per-frame inverse exposure (state nuisance, §1.4), $P^{\text{warp}}$ the
reference patch affine-warped into the current view.

**Jacobian (photometric chain rule).** With image gradient $\nabla I=[I_u,I_v]$
(central differences), $J_\pi=\partial\mathbf u/\partial\mathbf P^C$ (2×3,
`computeProjectionJacobian`), $J_{\text{img}}=\tau_{\text{cur}}\,2^{-\text{level}}\nabla
I$ (dossier 04 §4; `vio.cpp:1611-1617`):
$$
\frac{\partial r}{\partial\delta R_{W\,F_e}(t)} = J_{\text{img}}J_\pi\,R_{C\,F_e}\,[R_{W\,F_e}(t)^\top(\mathbf P^W-p_{W\,F_e}(t))]_\times,\qquad
\frac{\partial r}{\partial\delta p_{W\,F_e}(t)} = -J_{\text{img}}J_\pi\,R_{C\,F_e}R_{W\,F_e}(t)^\top,
$$
chained through the spline as in §3 head. Exposure gets its own column
($\partial r/\partial\tau_{\text{cur}}=I_{\text{cur}}$, the 7th column;
`vio.cpp:1628`); the camera extrinsic $T_{C\,F_e}$ follows the same chain
(online-refined, gated, §3.6).

**Affine warp + LiDAR-plane prior.** Reference→current patches are affine-warped
using the **LiDAR plane normal** as the patch depth model (homography form
`getWarpMatrixAffineHomography`, `vio.cpp:252-273`; dossier 04 §2 step 5) —
FAST-LIVO2's key improvement over constant-depth. `getBestSearchLevel` picks the
pyramid level with a well-conditioned warp determinant; coarse-to-fine over the
iterated solve widens the basin.

**Outlier handling.** Depth-continuity gate (reject patches straddling occlusion
edges, Δdepth > 0.5 m; dossier 04 §2 step 4); per-patch NCC gate (`calculateNCC`,
`ncc_thre`); `outlier_threshold = 1000` on photometric SSD; robust kernel on
survivors; oblique-view patches (>60° view-angle change) dropped.

**Measurement covariance.** `img_point_cov` (FAST-LIVO2 `IMG_POINT_COV=100`;
dossier 04 §5), inflated for high-gradient/low-texture patches.

### 3.3 IMU derivative residual + bias random-walk (the CT IMU factor)

Unlike the iEKF, where the IMU is the *process model*, here the IMU is a **direct
residual** on the spline's analytic derivatives (dossier 10 §4.1; CLINS Eq. 8–9,
Coco-LIC Eq. 15–16). For each raw sample $(a_m,\omega_m)$ at $t_i$, with
$\omega_{W\,F_e}(t_i)$ from the SO(3) spline and $\ddot p_{W\,F_e}(t_i)$ from the ℝ³
spline (§1.3):
$$
r^{\omega}_i = \big(\omega_{W\,F_e}(t_i) + b_g\big) - \omega_m,\qquad
r^{a}_i = \Big(R_{W\,F_e}(t_i)^\top\big(\ddot p_{W\,F_e}(t_i)-g_W\big) + b_a\Big) - a_m,
$$
weighted by $(\sigma_g^2,\sigma_a^2)$. Bias **random-walk** residuals tie
consecutive bias knots (dossier 10 §4.1 Eq. 16):
$$
r^{b_g}_k = b_g^{k}-b_g^{k-1},\qquad r^{b_a}_k = b_a^{k}-b_a^{k-1},
$$
weighted by $(\sigma_{bg}^2,\sigma_{ba}^2)\Delta t$. $r^{\omega}$ touches the 4
active SO(3) control points + the local $b_g$ knot; $r^{a}$ touches 4 SO(3) + 4 ℝ³
control points + the local $b_a$ knot + gravity. The IMU therefore **densely
constrains** the spline between sparse LiDAR/visual measurements — this is what
keeps under-determined control points (§10, failure 2) regularised.

### 3.4 GNSS (conservative absolute position)

Absolute antenna position, fused conservatively, gated by `fix`/`cov_enu` (spec 01
§4.4). With antenna lever-arm $t_{F_e,\text{ant}}$ (`CalibrationSet`), the antenna
world position at fix time $t$ is $p_{W,\text{ant}}=p_{W\,F_e}(t)+R_{W\,F_e}(t)\,t_{F_e,\text{ant}}$;
convert the fix LLA→ENU→$W$ via the latest datum/$T_{MW}$ snapshot (held fixed in
L2 — L2 reads, L3 owns it):
$$
r^{\text{gnss}} = z_{W}(\text{fix}) - p_{W,\text{ant}} \in\mathbb R^3,
$$
$\partial r/\partial\delta p_{W\,F_e}(t)=-I$,
$\partial r/\partial\delta R_{W\,F_e}(t)=R_{W\,F_e}(t)[t_{F_e,\text{ant}}]_\times$,
chained through the spline; weighted by `cov_enu` inflated per `fix` (SPP/FLOAT
large, FIXED trusted), under a robust kernel so one bad fix cannot snap the window.

**Policy.** GNSS in **L2 is optional and conservative** — it bounds drift over long
open-sky stretches and aids yaw. The **authoritative** GNSS fusion and datum
estimation live in **L3** (spec 05). L2 runs GNSS-disabled (tactical / GNSS-denied)
with no code-path change — it is just an absent residual.

### 3.5 The joint window cost

The fixed-lag MAP cost minimised each outer step (dossier 10 §6, Coco-LIC Eq.
17–18) over the active state $X$ = {control points in the window} ∪ {bias knots} ∪
{gravity} ∪ {window exposures} ∪ {active online extrinsics}:
$$
\mathcal C(X) = \sum_j \rho\!\big(\|r^{\text{lid}}_j\|^2_{\Sigma_{\text{lid}}}\big)
+ \sum_{p,\Delta}\rho\!\big(\|r^{\text{vis}}_{p,\Delta}\|^2_{\Sigma_{\text{photo}}}\big)
+ \sum_i\Big(\|r^{\omega}_i\|^2_{\Sigma_{g}}+\|r^{a}_i\|^2_{\Sigma_{a}}\Big)
+ \sum_k\|r^{b}_k\|^2_{\Sigma_{b}}
+ \sum\|r^{\text{gnss}}\|^2_{\Sigma_{\text{gnss}}}
+ \underbrace{\|X\boxminus X^{\text{prior}}\|^2_{\Omega_{\text{prior}}}}_{\text{marginalization prior, §5.4}}.
$$
$\rho$ is a Huber/Cauchy robust kernel. The **marginalization prior** is the
*single* clean representation of all past data inside L2 (§5.4); the IMU appears
**exactly once**, as residuals (never also as a prior on the same interval). That
single-representation discipline is what the L2→L3 handoff must also respect (§6.4).

### 3.6 Extrinsic / gravity / exposure refinement gating

Online refinement of $T_{F_e\,L}$, $T_{F_e\,C}$, gravity $g_W$, and inverse-exposure
$\tau$ is **on by default** (resolved default) but **enabled only when motion
excites the relevant DoF** (sufficient rotational + translational excitation over
the window, read from the §4 observability). Under degenerate motion the relevant
Jacobian columns are **frozen** (zeroed — FAST-LIO `extrinsic_est_en`,
`laserMapping.cpp:740-748`; FAST-LIVO2 `exposure_estimate_en`), preventing drift.
The freeze decision is logged (§8). Per spec 01 §5.3 the **authoritative** refined
extrinsics are an **L3** product fed back as a versioned `CalibrationSet` snapshot;
L2's online estimate is a fast local refinement that L3 ratifies.

---

## 4. Per-axis observability / degeneracy extraction

A first-class Meridian requirement (spec 00 §10.2; spec 01 §3.4). L2 measures, per
6-DoF axis, how well the geometry constrains the **current keyframe pose**, and
ships it in `ObservabilityReport` (spec 01 §3.4). This is the X-ICP / D²-LIO idea
on the information matrix — the principled replacement for a binary degeneracy flag.

### 4.1 The information matrix

The window solve already forms the Gauss-Newton information (Hessian) $J^\top\Sigma^{-1}J$
over all active variables. Restrict it, via the spline Jacobian at the keyframe
time, to the **6-DoF pose block** of $T_{W\,F_e}(\text{stamp})$:
$$
\Lambda_{\text{pose}} = \sum_j w_j\,J_j^\top\Sigma_j^{-1}J_j \in\mathbb R^{6\times6},\qquad J_j=\partial r_j/\partial\delta\boldsymbol\xi(\text{stamp}),
$$
summed over the LiDAR + visual residuals touching the keyframe's control points.
**Extracted for free** from the solver — no extra pass. (The iEKF oracle forms the
identical block as $H^\top R^{-1}H$, `vio.cpp:1660`; dossier 01 §4.5 — useful for
cross-checking §5.6.)

### 4.2 Per-axis scores

`ObservabilityReport` (spec 01 §3.4) carries `frame` (default `Body`/`ImuLink`),
`score[6]` in `[0,1]` ordered `[tx,ty,tz,rx,ry,rz]` (**translation-first**), and an
optional `eigvecs` 6×6 for non-axis-aligned degeneracy (e.g. a tunnel at 30°).
Compute all of the following; map the canonical `score[6]` from #1, expose the rest
in debug:

1. **Eigen-decomposition (Zhang/X-ICP):** eigenpairs $(\lambda_m,\mathbf v_m)$ of
   $\Lambda_{\text{pose}}$. Directions with $\lambda_m<\kappa_{\text{deg}}$ are
   degenerate; if not axis-aligned, fill `eigvecs`. Normalise to $[0,1]$ by
   $\lambda_m/(\lambda_m+\kappa_{\text{deg}})$.
2. **Marginal conditional precision per axis:** $1/[\Lambda_{\text{pose}}^{-1}]_{aa}$.
3. **Effective points per axis:** count of effective LiDAR points whose plane
   normal aligns with each axis (operator-friendly: "this axis is held by $N$
   planes").

### 4.3 Flow into the back-end

The scores **modulate the covariance** L2 attaches to the `KeyframePacket`
(`constraint_cov`, spec 01 §6.1) — degenerate axes get inflated variance — so L3
naturally down-weights ill-constrained directions (spec 05 maps `obs` to the noise
model). Under full degeneracy L2 may additionally **clamp** the update along weak
directions (let IMU/GNSS carry them — the X-ICP "solution remapping"), gated by
`degeneracy_handling` mode (§10). Because LiDAR and visual residuals sum into one
Hessian *before* inversion, a geometrically degenerate scene that the photometric
block constrains (or vice-versa) is automatically rescued (dossier 04 §0).

---

## 5. The sliding-window solve, marginalization, and adaptive knots

### 5.1 The outer loop

```
on each measurement ingest:
  append to per-stream time-ordered buffers; advance spline end if needed
on outer cadence (~0.1 s of new trajectory, knot_dt):           # Coco-LIC outer step, dossier 10 §5.2
  (1) extend the spline: place adaptive knots over the new segment (§5.5),
      seeded by IMU-only integration (continuity of pose/vel/orientation)   # dossier 10 §5.1
  (2) associate: for each new LiDAR point, kNN + plane fit at its spline pose (§3.1);
      for the new image, retrieve visual submap with LiDAR depth (§3.2)
  (3) build the Ceres problem: §3 residuals over active control points + biases
      + gravity + exposure + active extrinsics, + the marginalization prior (§5.4)
  (4) solve: Levenberg-Marquardt, analytic Jacobians, exploit band-sparsity      # §5.2
      - re-associate LiDAR planes when the pose has moved enough (rematch policy)
      - iterate to convergence (per-DoF |Δ| < epsi) or max_iterations
  (5) extract Λ_pose, observability (§4)
  (6) marginalize control points/biases that left the window → update prior (§5.4)
  (7) update the visual map (promote new LiDAR pts to visual pts; add obs; refine
      ref patches) — dossier 04 §6
  (8) maybe_emit_keyframe (§6.2); push retained per-keyframe cloud to IKeyframeStore (§6.5)
```

### 5.2 Solver

* **Solver:** **Ceres** (Levenberg-Marquardt; analytic cost functions; SO(3)
  manifold via `LocalParameterization`; built-in Schur for marginalization) — the
  choice of all three CT references (dossier 10 §11). GTSAM is reserved for the L3
  global iSAM2 graph; the CT window is its own Ceres problem feeding L3.
* **Band-sparsity:** each measurement touches 4 control points per spline ⇒ the
  Hessian is banded ⇒ sparse Cholesky / Schur on biases keeps cost bounded
  (dossier 10 §3, §6). Converges in 4–5 iterations, comparable to CLINS' ~200 ms
  budget but well within the Orin's GPU-offloaded headroom (LiDAR/visual map ops
  run on GPU, §11).
* **Iteration cap / tolerance:** `max_iterations`, per-DoF `epsi` (mirroring the
  iEKF `epsi=1e-3`; dossier 01 §4.5).

### 5.3 Sliding window (true fixed-lag)

The active set is the control points spanning the current window
(`window_seconds`, ~0.3–1.0 s) plus a small overlap of recent ones; older control
points are **marginalized** (§5.4), not held fixed (marginalizing keeps their
information consistently, unlike CLINS' "hold-fixed" which biases as the window
slides; dossier 10 §6). A fixed number of active control points ⇒ bounded
per-step cost.

### 5.4 Marginalization (the single clean prior)

When the window slides, the control points/bias knots leaving the window, and the
residuals connecting *only* to them, are **marginalized via Schur complement** into
a dense linear prior on the remaining **boundary** variables (VINS-Mono /
Coco-LIC-style; dossier 10 §6, Eq. 17–18). The retained prior set =
{control points shared between the dropped and kept windows} ∪ {current bias
knots} ∪ {gravity}. This prior is the **only** memory of past data inside L2 and is
**mutually exclusive** with re-introducing those measurements — guaranteeing no
double counting. (Linearization staleness is a known risk; §10 failure 6.) The same
single-representation discipline is carried to the L2→L3 boundary in §6.4.

### 5.5 Adaptive knots (Coco-LIC)

Knot density follows motion (dossier 10 §5.2–§5.3). Fixed **outer cadence**
`knot_dt ≈ 0.1 s`; within each outer segment, the number of control points
`n_cp ∈ {1..n_cp_max}` is chosen from IMU-measured dynamics:
$$
N_g = \tfrac1n\big\|\textstyle\sum_i R_i\,\omega_i\big\|,\qquad
N_a = \tfrac1n\big\|\textstyle\sum_i (R_i\,a_i - g)\big\|,
$$
mapped to a discrete `n_cp` (the larger of the two governs); knots are placed
uniformly within the segment in **virtual time** so the uniform basalt kernels
apply (dossier 10 §5.2). Net effect: more control points under aggressive motion,
fewer when smooth — accuracy when needed, cheap when not. (Exact $(N_g,N_a)\to
n_{cp}$ thresholds are Coco-LIC Fig. 2, graphical — tune against the released code,
dossier 10 §12 item 1.)

### 5.6 Optional reference/oracle: the iEKF baseline (not a milestone)

A FAST-LIO2-style sequential ESIKF over a single scan-end `NavState` (dossier 01;
LiDAR point-to-plane then FAST-LIVO2 photometric, sequential update, `vio.cpp`
order) is provided **only** as an offline differential-test oracle behind the same
`IFrontEnd`: it must produce a `KeyframePacket` indistinguishable in *type* from the
CT front-end, and its trajectory/observability are cross-checked against the CT
output on recorded bags (the iEKF forms the identical $H^\top R^{-1}H$ pose block,
§4.1, making the comparison direct). It is never on the live path, never a "v1",
and the system is not organised around it.

---

## 6. The `KeyframePacket` L2 emits

`KeyframePacket` is **canonically defined in spec 01 §6.1** and is the **one and
only** value L2 hands L3 (spec 00 §6, spec 01 §6). This section specifies how **L2
fills** it from the spline window — it does **not** redefine it.

### 6.1 The fields L2 must fill (quoting spec 01 §6.1)

```cpp
struct KeyframePacket {                 // SPEC 01 §6.1 is authoritative
  std::uint64_t  id;                     // monotonic, L2-assigned
  Timestamp      stamp;                  // PTP ns; a real measurement instant
  Frame          ref_frame;              // = Frame::Odom for a live front-end
  Pose           T_ref_body;             // T_ref_Fe (Fe = ImuLink) = spline eval at stamp
  bool           kinematics_included;    // false on the normal path (§6.4)
  Eigen::Vector3d v_ref, b_g, b_a;       // seeds/telemetry; authoritative only iff included
  enum class ConstraintKind { RelativeBetween, AbsolutePrior, ImuPreintegration }
                 constraint_kind;        // picks the L3 factor (§6.4)
  std::uint64_t  rel_to_id;              // previous KF id (RelativeBetween/ImuPreint)
  Pose           T_relto_this;           // relative transform (RelativeBetween)
  PoseCov6       constraint_cov;         // 6x6 [ρ;φ] block; meaning set by kind
  ObservabilityReport observability;     // 6 per-axis scores + frame (§4)
  std::shared_ptr<const std::vector<LidarPoint>> cloud_body;  // handle into store (§6.5)
  std::shared_ptr<const CameraFrame>            image;         // nullable
  Pose           T_body_cam;             // extrinsic snapshot used at this KF
  std::optional<ImuPreintegrationSummary> imu_summary;        // iff ImuPreintegration
  std::uint32_t  calib_version;          // CalibrationSet snapshot provenance
  std::uint32_t  frontend_kind;          // 1 = CT; diagnostics only; L3 must NOT branch
};
```

How each field is produced from the CT window:

| Field | Source |
|---|---|
| `id`/`stamp` | monotonic id; `stamp` = the chosen keyframe instant (a real measurement time). |
| `ref_frame`/`T_ref_body` | `Odom` + **spline evaluation** $T_{W\,F_e}(\text{stamp})$. |
| `kinematics_included`,`v_ref`,`b_g`,`b_a` | **false** on the normal path; `v_ref=\dot p_{W\,F_e}(\text{stamp})$ and the local bias knots are filled as **seeds/telemetry** (L3 ignores them in `RelativeBetween`). `true` only on the `ImuPreintegration` restart (§6.4). |
| `constraint_kind` | `RelativeBetween` default; `AbsolutePrior` on first KF / fresh GNSS anchor; `ImuPreintegration` on window-restart fallback (§6.4). |
| `T_relto_this`,`constraint_cov` | $\hat T_{\text{prev}}^{-1}\hat T_{\text{cur}}$ (both spline-evaluated); marginal **relative** covariance read directly from the window posterior / marginalization, **inflated on weak axes by `obs`** (§4.3), reordered to `[ρ;φ]` (§1.5). |
| `observability` | $\Lambda_{\text{pose}}$ pose block (§4). |
| `cloud_body` | per-point spline-deskewed scan in $F_{e,\text{stamp}}$, retained in `IKeyframeStore` (§6.5). |
| `image`/`T_body_cam` | the camera frame at this KF + the `CalibrationSet` $T_{F_e\,C}$ snapshot used. |
| `imu_summary` | filled only on `ImuPreintegration` restart (§6.4). |
| `calib_version`/`frontend_kind` | snapshot version; `1` (CT). |

Every field the boundary demands is pinned: timestamp; **SE(3) pose in a named
frame**; **optional velocity+bias with an "included" flag**; **covariance block**
(`[ρ;φ]`, spec 01 §3.3/§6.4); **6 per-axis observability with reference frame**;
**point-cloud handle**; **camera/RGB handle**.

### 6.2 Keyframe-selection policy

Emit on **any** trigger, measured from the last emitted keyframe (spec 00 config
`frontend.keyframe`):
* translation $> d_{\text{kf}}$ (0.5–1.0 m), **or**
* rotation $> \theta_{\text{kf}}$ (10–15°), **or**
* elapsed time $> \Delta t_{\text{kf}}$ (heartbeat 1–2 s, so a stationary platform
  still produces graph nodes), **or**
* **information-driven:** accumulated new visual/LiDAR information / observability
  change (densifies keyframes in feature-rich / manoeuvring sections).

Never while the window is uninitialised (§2.3). The first post-init keyframe anchors
the `odom` origin and uses `AbsolutePrior` (§6.4).

### 6.3 (reserved — kinematics across the boundary handled in §6.4)

### 6.4 The single clean constraint contract

Encoded by `constraint_kind` (spec 01 §6.3, §6.4; resolved default):

* **Default (steady state) — `RelativeBetween`.** L2 has fused IMU+LiDAR+visual
  (+GNSS) internally and *summarised* it as a single relative transform
  `T_relto_this` ($\hat T_{\text{prev,cur}}$) with marginal `constraint_cov`. L3
  adds **one** `BetweenFactor(rel_to_id → id)` and nothing else. The absolute
  `T_ref_body` is an **initialisation/visualisation guess only** — L3 does **not**
  treat it as an independent absolute prior. **No IMU factor in this mode** — the
  IMU information is already inside the relative covariance. `kinematics_included =
  false`; velocity/biases ride as seeds/telemetry, and L3 does not create
  velocity/bias nodes.

* **Anchor — `AbsolutePrior`.** First keyframe (fixes the gauge) and fresh
  GNSS-anchored poses. `constraint_cov` is the marginal cov of `T_ref_body` itself.
  No companion relative factor for an anchor, so no double-count; the next keyframe
  returns to `RelativeBetween`. (For GNSS-bearing keyframes L3 adds a *separate*
  GNSS factor, not derived from this pose.)

* **Restart fallback only (§10) — `ImuPreintegration`.** Across a window restart the
  relative pose is unreliable; L2 ships the raw `imu_summary` ($\Delta R,\Delta
  v,\Delta p$ + 9-DoF cov + bias Jacobians, spec 01 §6.5) and sets
  `kinematics_included = true`. L3 builds a GTSAM **`CombinedImuFactor`** (+ the
  velocity/bias nodes it needs) **instead of** a between. Because `constraint_kind`
  is a single enum, `RelativeBetween` and `ImuPreintegration` are **mutually
  exclusive by construction**. If even IMU integration is untrustworthy
  (saturation), L2 emits an `AbsolutePrior` with wide cov to start a fresh segment
  (loop closure / GNSS stitches segments later).

This guarantees the L2 information is represented **exactly once** at the boundary.
The contract matches the resolved default verbatim: single relative `BetweenFactor`
with front-end marginal covariance on the normal path; `CombinedImuFactor` only on
restart, mutually exclusive; bias estimation stays in L2 and crosses the boundary
only on restart.

### 6.5 Retained per-keyframe point-cloud store

L2 is the **producer** side of the retained per-keyframe cloud store; the store is
`IKeyframeStore` (spec 01 §7.5) owned by L4 (spec 06).

* On each keyframe L2 stores the **per-point spline-deskewed cloud in the keyframe
  estimation frame** $F_{e,\text{stamp}}$ (compact: body-frame points + the keyframe
  pose, so a corrected pose re-places them without re-querying the spline), keyed by
  `id`. `KeyframePacket.cloud_body` is a **Shared-immutable handle** into this store,
  never a copy (spec 01 §6.3, §6.5).
* This is the data structure loop-closure **map de-integration** needs. Because
  clouds are retained body-frame + pose, L4 does **clear-and-rebuild** of an
  affected region's nvblox voxels/TSDF from retained clouds at *corrected* poses
  after a loop closure (spec 06; `IMapLayer::apply_graph_update`, matching nvblox
  running-average semantics) rather than fragile per-voxel subtraction. L5 GICP and
  the final mesh read the same store (spec 01 §6.5).
* Retention is bounded by the store's ring-buffer / on-disk spill policy (spec 06);
  consumers re-fetch by `id` and never cache raw pointers across an eviction.

---

## 7. `IFrontEnd`, threading, and back-end feedback

### 7.1 The interface (spec 01 §7.3 is authoritative; quoted)

```cpp
class IFrontEnd {                                         // SPEC 01 §7.3
public:
  virtual ~IFrontEnd() = default;
  virtual void set_calibration(std::shared_ptr<const CalibrationSet>) = 0; // + on L3 refine
  virtual void ingest(const ImuSample&)   = 0;
  virtual void ingest(const LidarScan&)   = 0;            // filtered, undeskewed
  virtual void ingest(const CameraFrame&) = 0;            // optional modality
  virtual void ingest(const GnssFix&)     = 0;            // optional modality
  virtual NavState live_state() const = 0;                // smooth odom-frame estimate (spline eval at now)
  using KeyframeSink = std::function<void(KeyframePacket&&)>;
  virtual void set_keyframe_sink(KeyframeSink) = 0;       // the ONE thing L2 gives L3 (§6)
  virtual FrontEndDiagnostics diagnostics() const = 0;    // §8 (spec 01 §… canonical struct)
};
IFrontEnd::Ptr makeFrontEnd(const FrontEndConfig&, const CalibrationSet&);  // factory, spec 00 §5.1
```

The production class is **`CtFrontEnd`** (`frontend_kind = 1`). The factory may also
construct the iEKF oracle (§5.6) for offline differential tests; both fill the same
`KeyframePacket`, so the ROS 2 wrapper and L3 are written against `IFrontEnd` only.
`live_state()` returns the spline evaluated at the latest valid time (a `NavState`,
spec 01 §3.2) — for telemetry/control/L4 live integration, **not** the handoff.

### 7.2 Threading & ownership

* All `ingest`/`live_state`/`diagnostics` run on the **front-end thread**
  (Meridian's priority stage, spec 00 §11). Ingested samples are Borrowed; L2 copies
  what it must.
* `set_keyframe_sink`'s callback fires on the front-end thread but **enqueues** the
  `KeyframePacket` (Moved) onto a lossless thread-safe queue to the back-end thread
  (spec 00 §11; spec 01 §6.3). After the move L2 no longer owns it.
* `cloud_body`/`image` are Shared-immutable; their lifetime is reference-counted
  across {graph node, nvblox map store, loop detector} (spec 01 §6.3).

### 7.3 Back-end feedback

L3 corrections arrive via the wrapper calling `set_calibration` (refined extrinsics
as a versioned `CalibrationSet` snapshot, spec 01 §5.3) and a `PoseCorrection` path
(spec 00 §11.2, applied at safe points). On a `PoseCorrection`, L2 re-anchors the
odom frame and **shifts the spline control poses** by the correction (and updates
the cached `odom→map`/datum); it does **not** restart. On a new `CalibrationSet`
snapshot, L2 re-seeds its online extrinsic/exposure estimates and bumps
`calib_version` for provenance on subsequent packets.

---

## 8. Debug, introspection, timing

Spec 00 §10 mandates a first-class telemetry subsystem: the **core writes to a
ROS-agnostic `TelemetrySink`** and the wrapper maps it to topics. The canonical
synchronous return is `FrontEndDiagnostics` (spec 01 §… — `scan_time_ms`,
`solve_time_ms`, `deskew_time_ms`, `effective_points`, `mean_residual`,
`final_residual`, `observability`, `iterations`, `restarted`). L2 emits the
following, all gated by `debug.*` (counters/WARN default on; heavy clouds opt-in).

### 8.1 Telemetry keys → ROS 2 topics (wrapper-bound, spec 00 §10.2)

* `cloud("body/scan")`, `cloud("map/registered")` — spline-deskewed / world cloud
  (FAST-LIO `/cloud_registered{,_body}`, `laserMapping.cpp:849,851`).
* `cloud("frontend/effective")` + `scalar("frontend/n_effective")` +
  `scalar("frontend/res_mean")` — the inlier set that constrained the solve, plus
  count and mean residual (degeneracy made plottable).
* `cloud("frontend/visual_points")` — projected visual map points.
* `pose("odom/body")` + `vec("frontend/cov_diag")` — live odometry + typed 6×6 cov
  (not smuggled into a pose message as FAST-LIO does, `:597-606`).
* `vec("frontend/observability", obs[6])` — the 6 scores driving L3 noise; rendered
  as the **observability hexagon** marker (spec 00 §10.4), degenerate axes red.
* `pose("calib/T_imu_lidar")`, `pose("calib/T_imu_cam")` — online extrinsic
  estimates over time; `scalar("frontend/exposure")` — inverse exposure $\tau$.
* `vec("frontend/bias_acc"|"bias_gyr")`, `scalar("frontend/grav_norm")` — biases /
  $|g|$ from the active knots.
* `marker("spline/control_poses")` — the active control points + knot times, so the
  adaptive-knot density (§5.5) is visible.
* `timing(stage,ms)` per stage (extend/associate/solve/marginalize/map-update) →
  `StageTiming` (FAST-LIVO2 prints a per-stage table, `vio.cpp:1851-1868`; dossier
  04 §7).
* `event(WARN,"frontend/window_restart",reason)` — restart recovery made visible
  (spec 00 §10.2).
* `marker("keyframe_links")` — emitted keyframes + links coloured
  green=RelativeBetween, yellow=ImuPreintegration, red=AbsolutePrior/new-segment.

### 8.2 Structured logging

Per-outer-step one-line structured `INFO` (id, eff points, visual patches, iters,
mean residual, solve ms, min eigenvalue, n_cp, KF?); `WARN` on no-effective-points /
restart / extrinsic-or-exposure freeze; `ERROR` on IMU saturation / total tracking
loss. Heavy per-point dumps at `TRACE`. Via `meridian::log` (spec 00 §10.3); the core
never calls `RCLCPP_*`.

---

## 9. Parameters

| Param | Default | Source / dossier |
|---|---|---|
| `frontend.kind` | `ct_livo` | the front-end; `iekf_oracle` is offline-only (§5.6) |
| `init_time_s` | 0.1 | `INIT_TIME`, `laserMapping.cpp:63`; dossier 02 §2 |
| `imu.gyr_cov` $\sigma_g^2$ | 0.1 | `avia.yaml:15`; dossier 02 §1.4 |
| `imu.acc_cov` $\sigma_a^2$ | 0.1 | `avia.yaml:14` |
| `imu.b_gyr_cov` $\sigma_{bg}^2$ | 1e-4 | `avia.yaml:17` |
| `imu.b_acc_cov` $\sigma_{ba}^2$ | 1e-4 | `avia.yaml:16` |
| `ct.spline_order` | 4 (cubic, $C^2$) | dossier 10 §2.1, §11 |
| `ct.representation` | split SO(3)×ℝ³ | dossier 10 §2.4 (consensus best) |
| `ct.knot_dt_s` | 0.1 (outer cadence) | Coco-LIC; dossier 10 §5.2 |
| `ct.n_cp_max` | tune | adaptive knot cap; dossier 10 §5.2 |
| `ct.window_seconds` | 0.3–1.0 | fixed-lag window; dossier 10 §6 |
| `ct.time_offset_estimate` | false | per-sensor $t_d$ hook; dossier 10 §9 |
| `lidar.num_match_points` | 5 | `NUM_MATCH_POINTS`; dossier 03 §1.2 |
| `lidar.max_match_dist_sq` | 5.0 m² | `laserMapping.cpp:671`; dossier 03 §1.2 |
| `lidar.plane_thresh` | 0.1 m | `esti_plane`; dossier 03 §2.1 |
| `lidar.point_cov` $\sigma_0^2$ | 1e-3 | `LASER_POINT_COV`; dossier 03 §2.5 |
| `map.voxel_m` | 0.5 | `filter_size_map`; dossier 03 §3.2 |
| `solver.max_iterations` | 5 | CLINS 4–5; dossier 10 §6 |
| `solver.epsi` | 1e-3 | `epsi[23]`; dossier 01 §4.5 |
| `visual.patch_size` | 8 | `patch_size`; dossier 04 §3 |
| `visual.pyramid_levels` | 3 | `patch_pyrimid_level`; dossier 04 §3 |
| `visual.img_point_cov` | 100 | `IMG_POINT_COV`; dossier 04 §5 |
| `visual.outlier_threshold` | 1000 | dossier 04 §2 |
| `visual.ncc_thre` | tune | NCC gate; dossier 04 §2 |
| `visual.exposure_estimate_en` | true (gated) | dossier 04 §3 |
| `kf.trans_m` | 0.5–1.0 | keyframe trigger |
| `kf.rot_deg` | 10–15 | keyframe trigger |
| `kf.time_s` | 1–2 | heartbeat trigger |
| `obs.degeneracy_thresh` $\kappa_{\text{deg}}$ | tune | eigenvalue floor (§4.2) |
| `extrinsic.refine` | true (gated) | `extrinsic_est_en`; dossier 03 §2.4 |
| `gnss.enable` | env-dependent | off in GNSS-denied (§3.4) |

(Covariance/weight values $\Sigma_L,\Sigma_I,\Sigma_C$ are not given numerically in
the CT papers — tune empirically; dossier 10 §12 item 3.)

---

## 10. Failure modes & mitigations

| Failure | Detection | Response |
|---|---|---|
| **Cold start, no trajectory** | spline spans < first valid knots | IMU-only deskew/seed (§2.3); build window from first data; no keyframe until window converged (spec 00 §7.2). |
| **No effective LiDAR points** | effective count < 1 (`laserMapping.cpp:708`) | mark step degraded; lean on IMU + visual residuals (the spline stays constrained by §3.3); bump counter; if persistent → window restart. |
| **Knot spacing too coarse** for the motion | systematic IMU/LiDAR residual the solver cannot null (dossier 10 §10.1) | adaptive `n_cp` raises control-point density (§5.5). |
| **Control points under-constrained** between sparse measurements | ill-conditioning / jitter (dossier 10 §10.2) | dense IMU residuals regularise; cap `n_cp`; marginalization prior anchors the boundary. |
| **Geometric degeneracy** (corridor, tunnel, open field) | small eigenvalue(s) of $\Lambda_{\text{pose}}<\kappa_{\text{deg}}$ (§4) | inflate `constraint_cov` on weak axes; optionally clamp the update along weak dirs (solution remapping); lean on IMU/GNSS/visual; **freeze extrinsic/gravity refinement** (§3.6). |
| **Photometric failure** (low light, blur, over/under-exposure) | high photo residual / NCC fail / depth-continuity fail / `outlier_threshold` hit | drop visual residuals this step (LIO only); freeze exposure $\tau$; WARN. Degrade gracefully. |
| **IMU saturation / dropout** | accel/gyro at FSR or gap > `imu_gap_max` | mark; if integration untrustworthy emit `AbsolutePrior` wide-cov new segment; ERROR. |
| **Unknown clock offset $t_d$** | systematic, motion-correlated residual (dossier 10 §10.4) | nominally prevented by PTP (spec 02); enable `ct.time_offset_estimate` hook (§2.2). |
| **Marginalization linearization staleness** | prior inconsistent after a large correction (dossier 10 §10.6) | re-linearize prior on `PoseCorrection`; bound window so the prior is never far from the current estimate. |
| **GNSS spoof/jump/multipath** | fix-residual gate / `fix` type / jump vs odom | reject / inflate / disable; robust kernel; one bad fix never snaps the window. Authoritative robustness (PCM) in L3. |
| **Divergence** (residual blow-up, all-axis obs collapse) | residual norm > thresh; min eigenvalue ≈ 0 across axes | **window restart**: discard the window, re-bootstrap from IMU-only deskew, emit `ImuPreintegration` (or wide `AbsolutePrior`) so L3 stitches the gap; prior keyframes preserved (§6.4). |
| **Stale cloud handle after loop closure** | store eviction | consumers re-fetch from `IKeyframeStore` by `id`, never cache raw pointers (§6.5). |

---

## 11. Library choices and what was rejected

One choice per job; alternatives are named here only to record that they were
considered and rejected (library canon, system direction).

| Job | Chosen | One-line justification | Rejected (why) |
|---|---|---|---|
| CT spline kernel | **basalt-headers (vendored)** | canonical, battle-tested $O(k)$ analytic derivatives **and** Jacobians w.r.t. control points — removes the single biggest implementation risk (dossier 10 §11). | hand-rolled spline (months of subtle Jacobian bugs); full-SE(3) (non-split) spline (screw coupling, slower; dossier 10 §2.4). |
| Window solver / marginalization | **Ceres** (LM + Schur) | what all three CT references use; manifold SO(3), built-in Schur (dossier 10 §11). | GTSAM here (reserved for the L3 iSAM2 global graph). |
| NN for plane fitting | **ikd-Tree (vendored) or adaptive voxel-hash** | incremental, box-delete, lock-safe rebuild for the recent-point query (dossier 03 §3). | brute-force kNN. |
| Map (TSDF + colour + mesh) | **nvblox (GPU)** | GPU TSDF + colour + Marching Cubes on the Orin, the single map backend (spec 06). **No CPU fallback, no VDBFusion, no OpenVDB/NanoVDB.** | CPU TSDF / VDBFusion (Orin always has CUDA — a second path is dead weight). |
| Robust kernels / IMU factor (L3 side) | **GTSAM** `noiseModel::Robust`+Huber, `CombinedImuFactor`, `GncOptimizer` | the restart factor and global graph (spec 05). | per-interval preintegration on the normal path (the relative cov already carries it; §6.4). |
| Linear algebra / Lie | **Eigen 3.4 + Sophus** | basalt/Ceres substrate. | — |

**One paragraph on the iEKF.** A FAST-LIO2-style sequential ESIKF is the well-known
fast alternative to a CT window. Meridian keeps it **only** as an offline
reference/oracle behind `IFrontEnd` (§5.6): it is faster and battle-tested, but it
handles intra-scan motion by approximate backward-propagation (not joint
optimisation), fuses an asynchronous camera awkwardly, and cannot natively
represent the sub-scan trajectory the colourised-mesh goal benefits from (dossier
10 §8). The CT estimator is the design; the iEKF is the test partner.

---

## 12. Consistency checklist

* **Spec 00 (architecture):** L2 behind `IFrontEnd`; **one** production front-end
  (`CtFrontEnd`), iEKF is an offline oracle only; deskew via injected
  `IDeskewProvider` backed by the spline (§2, spec 00 §7); telemetry through
  `TelemetrySink` keys (§8, spec 00 §10); L2 the priority stage, `KeyframePacket`
  moved on a lossless queue (§7, spec 00 §11). Honoured.
* **Spec 01 (interfaces/types):** `KeyframePacket`, `NavState`,
  `ObservabilityReport`, `LidarScan/LidarPoint`, `CameraFrame`, `ImuSample`,
  `GnssFix`, `CalibrationSet`, `FrontEndDiagnostics`, `IFrontEnd` used **verbatim**;
  tangent order `[ρ;φ]` and obs order `[tx,ty,tz,rx,ry,rz]`; extrinsics **not** in
  the trajectory state (in `CalibrationSet`); `cloud_body` is a handle. **Spec 01 is
  canonical; §6/§7 here restate, never redefine.** If they diverge, fix this file.
* **Spec 02 (L0 time):** all stamps PTP ns; per-point `t_offset_ns` consumed at true
  time (§3.1); the known-timing caveat (§2.2) relies on PTP.
* **Spec 03 (L1):** consumes undeskewed `LidarScan`, pyramided `CameraFrame`, raw
  IMU, gated `GnssFix`; deskew is L2-owned (intrinsic to the spline; `IDeskewProvider`
  feedback at cold start), not L1 (§2).
* **Spec 05 (L3):** receives exactly one `KeyframePacket` stream; **one factor per
  keyframe** via `constraint_kind` — single relative `BetweenFactor` (front-end
  marginal cov) on the normal path, `CombinedImuFactor` only on restart
  (mutually exclusive), `AbsolutePrior` for anchors/GNSS; `kinematics_included =
  false` except on restart; bias estimation in L2 (§6.4). Feeds back refined
  `CalibrationSet` / `PoseCorrection` (§7.3).
* **Spec 06 (L4 / nvblox / store):** L2 fills the retained per-keyframe body-frame
  cloud enabling nvblox clear-and-rebuild de-integration; handles only; nvblox is
  the sole GPU map backend (§6.5, §11).
* **Spec 07 (L5):** GICP/place-recognition reads the same retained store.

### Direction conformance (this rewrite)

1. **One complete CT LIVO+GNSS system** — no v1/v2, no Phase 0-6; §0 is module
   bring-up order only. iEKF demoted to one paragraph (§5.6, §11) as an offline
   oracle.
2. **Single LiDAR + IMU + camera + GNSS** — multi-LiDAR/dome merge logic removed
   (§1.1, §2.2); mentioned once as a future extension.
3. **CT design grounded in the apex refs** — split SO(3)×ℝ³ cubic spline
   (basalt-headers), adaptive knots (Coco-LIC), Schur marginalization
   (Coco-LIC/VINS), direct LiDAR (FAST-LIO2), sparse-direct photometric +
   exposure + unified voxel map (FAST-LIVO2), iSAM2/`CombinedImuFactor` (GTSAM).
4. **Simplicity mandate honoured** — nvblox GPU-only, no fallbacks; one solver
   (Ceres) for the window; one map backend; single best option per job (§11), no
   "or alternatively" hedges.
5. **Resolved defaults honoured** — estimation frame `imu_link`; single relative
   `BetweenFactor` w/ marginal cov; `kinematics_included = false` normal path;
   `CombinedImuFactor` only on restart (mutually exclusive); bias estimation in L2;
   online extrinsic refinement on by default; scope stops at the colourised mesh.
