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
> **Grounding.** The normative design above stands on its own. Non-normative
> reference grounding — curated equation↔`file:line` maps and parameter tables for
> the SOTA systems this spec was validated against (CT spline math from
> CLINS/Coco-LIC/basalt; the LiDAR map's ikd-Tree internals; FAST-LIVO2's
> photometric residual; FAST-LIO's IMU model/deskew and manifold/ESIKF) — is
> collected in **Appendix R**. The reference clones live in
> `/home/user/slam-reference`. Read the real files before quoting a line; do not
> trust memory.

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
13. [Appendix R — SOTA reference grounding (non-normative)](#appendix-r--sota-reference-grounding-non-normative)

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
translation $p_{W\,F_e}(t)$ — exactly basalt-headers' `Se3Spline` internals, which
are themselves a split `So3Spline` + `RdSpline<3,N>`. The split representation is the
consensus best choice (the literature finds split better than a unified SE(3) spline
both in trajectory representation and in computation time; it is what CLINS and
Coco-LIC ship): it is faster (no SE(3) adjoints in cross terms), avoids the
screw-coupling of a unified SE(3) spline, and cleanly decouples the gyro residual (↔
SO(3) spline) from the accel residual (↔ ℝ³ spline 2nd derivative). See Appendix R.1.

For $t$ in knot segment $i$ (normalised time $u=u(t)\in[0,1)$), the value depends on
exactly **four** local control points (the local support of a cubic):

$$
R_{W\,F_e}(t)=R_i\prod_{j=1}^{3}\mathrm{Exp}\!\big(\lambda_j(u)\,d^{R}_j\big),\quad
d^{R}_j=\mathrm{Log}\!\big(R_{i+j-1}^{\top}R_{i+j}\big)\in\mathfrak{so}(3),
$$
$$
p_{W\,F_e}(t)=p_i+\sum_{j=1}^{3}\lambda_j(u)\,d^{p}_j,\quad
d^{p}_j=p_{i+j}-p_{i+j-1},
$$

with cumulative blending weights $\lambda(u)=\tilde M^{(4)}\,[1,u,u^2,u^3]^\top$ and
the cubic cumulative matrix (the Kim/basalt convention; cross-check against basalt's
`computeBlendingMatrix<4,...,true>`, Appendix R.1):

$$
\tilde M^{(4)}=\tfrac16\begin{bmatrix} 6&5&1&0\\ 0&3&3&0\\ 0&-3&3&0\\ 0&1&-2&1 \end{bmatrix},
\qquad \lambda_0\equiv 1.
$$

Control points are non-uniformly spaced (adaptive knots, §5.5); the non-uniform
behaviour is realised by running the uniform basalt kernels over a **virtual time**
remapping (Coco-LIC's engineering pattern; Appendix R.1), so the analytic
Jacobians stay the vendored, battle-tested ones.

> **Considered and rejected: the Point-LIO per-point-update model.** Point-LIO
> handles intra-scan motion without any deskew by promoting angular velocity and
> acceleration to filter states and running a sparse Kalman update at *every*
> LiDAR point's true time — a discrete-sample filter view of a continuous-time
> SDE. It is the strongest filter-side alternative to a CT window for
> aggressive-motion robustness. Meridian does **not** adopt it: the trajectory here
> is an explicit, jointly-optimised CT spline, which gives the same per-point true-time
> registration *plus* analytic sub-scan poses for colourisation, native multi-sensor
> asynchrony, and a window NLLS that fuses LiDAR and photometric residuals in one
> Hessian — exactly the CT side of the contrast in Appendix R.1. A per-point EKF
> would re-introduce a separate filter recursion and discard the smooth queryable
> trajectory the colourised-mesh goal depends on.

> **A control point is not an on-trajectory pose, and warm-start seeding must
> account for it.** Because the blending is cumulative, the control point with grid
> time $t_j$ dominates the curve roughly **one knot interval earlier**: at the start
> of interval $i$ the spline evaluates to $\approx(C_i+4C_{i+1}+C_{i+2})/6$, so $C_j$
> peaks in influence near $t_j-\Delta t$, not $t_j$. Consequently the IMU warm start
> (step (1) of §5.1) does **not** set $C_j = T_{\text{seed}}(t_j)$ — that would
> reproduce the seed trajectory shifted *late* by one interval. It seeds each new
> control point from the seed trajectory sampled **one local knot interval back**,
> $C_j\leftarrow T_{\text{seed}}(t_j-\Delta t_{\text{local}})$ (the ℝ³ seed may carry
> a small $-\tfrac{\Delta t^2}{6}\ddot p$ second-order correction), so the warm-started
> curve lands on the seed trajectory rather than a delayed copy of it. Knot seeding by
> IMU integration is a property of the *curve*, never $C_j=T(t_j)$.

### 1.3 Analytic derivatives (the reason for CT)

Because the basis is polynomial in $u$, **angular velocity, velocity, and
acceleration are closed-form** in the control points and basis derivatives
($\dot\lambda=\tilde M\,\dot u/\Delta t$, $\ddot\lambda=\tilde M\,\ddot u/\Delta
t^2$). From the SO(3) spline, the **body angular velocity** the
gyro sees is $\omega_{W\,F_e}(t)=(R_{W\,F_e}^\top\dot R_{W\,F_e})^\vee$, computed by
the $O(k)$ Sommer recurrence (`So3Spline::velocityBody`). From the ℝ³ spline,
$\ddot p_{W\,F_e}(t)$ is the world linear acceleration (`RdSpline::evaluate<2>`).
These feed the IMU residual (§3.3) with **no numerical differentiation** and give the
**exact per-point deskew pose** $T_{W\,F_e}(t_{\text{offset}})$ for every LiDAR point
(§2.3) — deskew *is* the trajectory, the central advantage over the
piecewise-constant-$\omega$ backward-propagation of FAST-LIO (Appendix R.1, R.4).

### 1.3.1 The data horizon: tail-knot pinning, re-seeding, and tail anchors

Each cubic interval's value depends on exactly **four** local control points (the
local support of a cubic, §1.2), so a spline extended to cover a new sweep ends with
a run of control points that **no measurement reaches** — they lie past the newest
measured time $t_{\text{end}}$. Their basis support over $[\,\cdot\,,t_{\text{end}}]$
is zero (or, for the first such knot, near-zero), so the window cost is *flat* in
them: left free they are an exact null space the solver fills with noise, and a
poisoned seed at the tail can drag the supported knots through their coupling. The
front-end handles this unmeasured span with three coordinated mechanisms; the
overarching discipline is that the tail is governed by **knot index**, never by
sampling the curve at times (§5.4).

* **Tail-knot pinning.** Knots whose basis weight over all measured times is exactly
  zero — deque index $\ge \text{interval}(t_{\text{end}})+4$ — are held constant
  (`SetParameterBlockConstant`) at their IMU warm-start values for this solve. The
  one near-zero-support knot at index $\text{interval}(t_{\text{end}})+3$ is also
  pinned **when** its peak basis weight $u_{\text{end}}^3/6$ over the segment falls
  below a small floor (`< 0.02`, i.e. the sweep ends just past a knot boundary so the
  knot is effectively unconstrained); above that floor it carries enough support to
  be left free. Pinned knots **unfreeze naturally** on a later sweep once real data
  covers them.
* **Tail re-seeding.** A pinned tail knot's stored value is pure extrapolation, so it
  is **re-seeded each sweep** from the new IMU-integrated seed
  (`SplineWindow::reseedFrom`, overwriting from the first stale tail index onward with
  the same one-interval-back warm-start placement of §1.2) — the freshest IMU
  prediction always supersedes the previous sweep's guess rather than letting a stale
  tail persist.
* **Tail anchors.** To tie the *velocity* and *body rate* of the unmeasured span to
  the seed's constant extrapolation (the pinning removes the strict null directions;
  the anchors hold the read-out where the seed carries model error such as an
  unconverged bias), a **`VelocityAnchorResidual`** on the ℝ³-spline derivative
  toward the seed's end velocity and a **`RateAnchorResidual`** on the SO(3)-spline
  body rate toward the seed's end rate are added at a `window_dt/4` stride past
  $t_{\text{end}}$, with $\sigma_{\text{vel}}=0.2$ m/s and $\sigma_{\text{rate}}=0.1$
  rad/s — weak enough that any real measurement overrides them. The anchors **never
  touch the marginalized knot**, so they never enter the marginalization prior (§5.4)
  and cannot double-count past data.

### 1.4 The non-trajectory state: biases, gravity, exposure

Beyond the control points, the optimised state carries:

* **IMU biases** $b_g,\,b_a\in\mathbb R^3$ — **bias estimation lives in L2**
  (resolved default). Carried as a **sliding multi-knot bias timeline** (`BiasKnots`):
  a per-knot $(b_g,b_a)$ pair at a coarse cadence `bias.knot_dt_ms` (default 500 ms),
  the bias at an arbitrary time being the piecewise-linear interpolation between the
  bracketing knots, tied by random-walk residuals between consecutive knots (§3.3;
  Appendix R.1). The timeline is held in **deque storage** so the per-knot pointers
  handed to Ceres and the marginalization prior stay valid as it grows and slides; it
  **grows** with the trajectory (`extendTo` appends knots copying the last value) and
  **slides** in lock-step with the spline (`dropOldest`, front-trim, in `slideWindow`,
  §5.4). Both $b_g$ and $b_a$ are always free, held inside box bounds (§5.2). There
  is **no separate velocity state** — velocity is $\dot p_{W\,F_e}(t)$, read from the
  spline.
* **Gravity** $g_W\in S^2$ — a 2-DoF direction of fixed magnitude $|g|=9.81$,
  optionally refined once excitation is sufficient (§3.6). Fixing the magnitude on
  $S^2$ avoids $|g|$ drift (the FAST-LIO rationale; Appendix R.5). Stored as a
  **unit 3-vector held on the sphere by a Ceres `SphereManifold`** with the magnitude
  kept out of the optimised state (so $|g_W|$ is exactly constant across solves, and
  $g_W=|g|\,\hat g$); this is the same 2-DoF $S^2$ state as the CLINS/VINS
  tangent-basis $g$-refinement, in the cleaner manifold form. Initialised
  $g_W=-\,\overline{a_m}/\|\overline{a_m}\|\cdot 9.81$ from a short static start
  (Appendix R.4).
* **Inverse exposure** $\tau$ — the FAST-LIVO2 affine photometric nuisance
  (`inv_expo_time`; Appendix R.3), **one scalar per camera frame in the window**,
  optimised when `exposure_estimate_en` (§3.2). It is **not** a free per-frame
  unknown: consecutive in-window exposures are tied by a **random-walk residual**
  $r^{\tau}_k=\tau_k-\tau_{k-1}$ weighted by $\sigma_\tau^2\Delta t$
  (`visual.inv_expo_cov`, default 1e-2 /s), so a frame with little photometric
  evidence inherits its neighbour's exposure rather than floating, and a sudden
  auto-exposure step is tracked rather than absorbed into the pose. The first
  in-window exposure is anchored by a prior toward `inv_expo_prior` with std
  `inv_expo_std` from the camera intrinsics (spec 01 §5.1; `inv_expo_std = 0` ⇒ that
  frame's $\tau$ is held fixed at the prior). Each $\tau$ is **clamped strictly
  positive** ($\tau\ge$ `visual.inv_expo_min`, default 1e-3) — a non-positive
  inverse exposure is physically meaningless and would flip the sign of the
  photometric residual — realised as a Ceres lower bound on the parameter.

None of these cross the `IFrontEnd` boundary as live variables. At a keyframe
instant the spline is **evaluated** to produce the `NavState`/`KeyframePacket` pose
(spec 01 §6.2); biases/velocity are exported only as **seeds/telemetry**
(`kinematics_included = false`, §6.4) on the normal path.

### 1.5 Manifold operators & tangent ordering

Right-invariant box-plus/box-minus (spec 01 §3.1; Appendix R.5): on $SO(3)$,
$R\boxplus\theta=R\,\mathrm{Exp}(\theta)$, $R_1\boxminus R_2=\mathrm{Log}(R_2^{\top}R_1)$
(Rodrigues `so3_math.h:Exp/Log`). Control-point increments use this on each
$\{R_i\}$ via a Ceres SO(3) manifold/`LocalParameterization`; $\{p_i\}$, biases are
Euclidean. **All 6-DoF tangents and 6×6 blocks in the core order translation-first
then rotation** (`[ρ;φ]`, spec 01 §3.1) — with **one exception**: the
`KeyframePacket.constraint_cov` block crossing L2→L3 is ordered **rotation-first**
`[rx,ry,rz,tx,ty,tz]` to match the GTSAM `Pose3` boundary (spec 01 §6.1). The
spline's native rotation Jacobians are rotation-first inside basalt; the packet
adapter (§6) assembles `constraint_cov` by reordering the window's translation-first
marginal into that rotation-first layout exactly once (the same reorder FAST-LIO does
when packing its `pose.covariance`).

### 1.6 The IMU kinematic model (the residual's reference equations)

Continuous-time strapdown, in $W$, with measured specific force $a_m$ and rate
$\omega_m$ (the strapdown INS model; Appendix R.4, `use-ikfom.hpp::get_f`):

$$
\dot R_{W\,F_e}=R_{W\,F_e}\,[\omega_m-b_g-n_g]_\times,\quad
\dot p_{W\,F_e}=v_W,\quad
\dot v_W=R_{W\,F_e}(a_m-b_a-n_a)+g_W,\quad
\dot b_g=n_{bg},\ \dot b_a=n_{ba},\ \dot g_W=0.
$$

White noises $(n_g,n_a)$ and bias random walks $(n_{bg},n_{ba})$ have densities
$(\sigma_g^2,\sigma_a^2,\sigma_{bg}^2,\sigma_{ba}^2)$ from the `CalibrationSet`
(Allan-derived, spec 01 §5.3; for reference, FAST-LIO's `avia.yaml` defaults are
0.1, 0.1, 1e-4, 1e-4 — Appendix R.4). In the CT estimator this ODE is **not
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
  `patch_pyrimid_level=3`; Appendix R.3).
* **IMU:** raw `ImuSample{stamp, acc, gyro}` at full rate (spec 01 §4.1; no
  orientation — Meridian never trusts a vendor AHRS).
* **GNSS:** `GnssFix{stamp, lat/lon/alt, cov_enu, fix, num_sats}` (spec 01 §4.4),
  gated by L1.

### 2.2 Measurement intake (no merge-of-N)

There is no `sync_packages`-style merge of multiple LiDARs. Intake is the per-sweep
`PreprocessedGroup` (plus the `ingest_imu_live` low-latency tap), but inside the
window every measurement keeps its own clock: the solve (§5) consumes each one
**at its own `stamp`/`t_offset_ns`**; asynchrony is native to a CT spline (Appendix R.1) — IMU at 200–400 Hz,
LiDAR points streaming across the sweep, camera at 10–30 Hz, GNSS at 1–10 Hz all
evaluate the same $T_{W\,F_e}(\cdot)$ at different $t$. The only batching is the
**outer optimisation cadence** (§5.1, ~0.1 s): when the spline has been extended to
cover new measurements and enough new data has arrived, one window solve runs.

> **Sweep gaps: bridged vs unbridgeable.** The steady-state seed integrates a group's
> IMU forward from `last_solved_t_`, which is only valid when the IMU actually reaches
> back that far. A gap is detected when a sweep opens more than half a sweep span past
> `last_solved_t_`, and the response splits on whether the aggregator-supplied group
> IMU **bridges** it: if the group's earliest IMU sample reaches `last_solved_t_`
> within **2 IMU periods**, the seed integrates across the hole and the sweep is
> **solved normally** (logged `frontend/sweep_gap_bridged`). If the IMU itself starts
> after `last_solved_t_` — a genuine data hole the seed cannot span — the front-end
> calls **`reseedAfterGap`**: it advances the live anchor across the hole on a
> **constant-velocity** prediction, **rebuilds the window** (and the bias timeline)
> from that anchor, and inserts the reseed sweep into the map **without solving**
> (logged as a window reseed). The next sweep with continuous IMU solves normally
> against the rebuilt window.

> **Known-timing caveat.** A CT spline absorbs *known* timing;
> it does **not** self-correct an unknown inter-sensor clock offset $t_d$ — it would
> faithfully fit the wrong time and bake the offset into the trajectory. Meridian's
> sensors are PTP-disciplined (spec 02), so per-sensor $t_d$ is nominally zero. The
> spline supports adding $t_d$ as an optional optimised parameter (evaluate that
> sensor's residual at $T(t+t_d)$, differentiable since $T(\cdot)$ is),
> defaulting to fixed-zero post-calibration. This is an L2 hook, off by
> default, owned with the rest of online calibration (spec 08). See Appendix R.1.

### 2.3 Deskew is the trajectory; the cold-start feedback edge

**The classic circularity** — deskew needs the trajectory over the sweep, but the
trajectory is estimated from the deskewed sweep — **does not arise in steady
state**, because there is no deskew pass: each LiDAR point's residual (§3.1) is
evaluated at $T_{W\,F_e}(t_{\text{offset}})$, its own time, queried from the
current spline. As the window solve iterates, every point is implicitly
re-deskewed because the control points it hangs from move. Deskew is exact and
free — it *is* the trajectory, with no separate motion-compensation pass (Appendix R.1).

Architecturally L1 still transforms points only through an injected
`IDeskewProvider::poseAt(Timestamp)→Pose` (spec 00 §5.2, §7.3) whose concrete
implementation is **backed by the L2 spline**, so the build graph stays acyclic
(L1 depends only on the interface) while data flows L2→L1→L2. **L2 owns the pose
source.** Two regimes:

* **STEADY (spline-backed).** `poseAt(t)` returns $T_{W\,F_e}(t)$ from spline
  evaluation; any frame L1 needs to deskew for visualisation uses it. The estimator
  itself never calls it — it consumes raw `t_offset_ns` directly in §3.1.

* **COLD-START / RESTART (IMU-only).** Before the spline spans valid data (first
  knots), or after a window restart (§10), there is no trajectory. `poseAt(t)` falls
  back to **IMU-only forward integration** from the last known state (the FAST-LIO
  `UndistortPcl` forward pass; Appendix R.4):
  integrate the bias-corrected IMU over the sweep with constant-$\omega$/constant-
  accel intra-interval, seed the first control points from it. Require a short
  static start ≥ `init_time_s` to initialise gravity and gyro bias; a static start of
  **~1.0 s** materially tightens that initialisation (FAST-LIO's `INIT_TIME=0.1 s`
  default, Appendix R.4, is too short for the gravity/bias estimate Meridian seeds the
  window from). **No keyframe is emitted while the window is
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
L3 ──(GraphUpdate / refined CalibrationSet snapshot)──▶ L2   (spec 01 §7.3, §5.3)
```

Deskew is **owned by L2** and is intrinsic to the spline; L1 publishes raw per-point
times only. This is the documented break of the circular dependency (spec 00 §7).

---

## 3. The CT residual set on control points

Every residual is a function of the spline $T_{W\,F_e}(t)$ evaluated at the
measurement's own time $t$, plus biases/gravity/exposure. Its Jacobian w.r.t. the
active control points is the **measurement Jacobian** (below) chained through the
**spline Jacobian** $\partial T_{W\,F_e}(t)/\partial\delta c_{i+m}$ (basalt's
analytic Jacobians; Appendix R.1). Each measurement touches only **4 control
points per spline** ⇒ the window Hessian is **banded** ⇒ the solve is real-time
(§5.2). For each residual: measurement model, residual, weight, code anchor.

### 3.1 LiDAR point-to-plane (direct, deskew-free)

FAST-LIO2's *direct* registration adapted to CT (Coco-LIC's spline form of the same
point-to-plane factor; Appendix R.1). No edge/planar feature extraction.

**Per-point world transform at true time.** For LiDAR point $\mathbf p^{L}_j$ with
offset $t_j$, evaluate the spline at $t_j$ and fold in the extrinsic
$T_{F_e\,L}$ (from `CalibrationSet`):
$$
\mathbf p^{W}_j = R_{W\,F_e}(t_j)\big(R_{F_e\,L}\,\mathbf p^{L}_j + t_{F_e\,L}\big) + p_{W\,F_e}(t_j).
$$
Because every point uses its own $t_j$, this **is** the deskew (§2.3).

**Association.** Query the map (spec 06; nvblox is the map backend, but the
front-end's nearest-neighbour query for plane fitting uses a vendored **ikd-Tree**
or adaptive voxel-hash over the recent point set — the role `ikdtree.Nearest_Search`
plays in FAST-LIO; the ikd-Tree's internal contract is Appendix R.2) for
`num_match_points = 5` neighbours. Reject if the farthest squared distance
> `max_match_dist_sq = 5` m². Fit a plane $(\mathbf n, d)$, $\|\mathbf
n\|=1$, by 5-point QR least-squares of $A\mathbf u=-\mathbf 1$ (FAST-LIO's
`esti_plane`); reject unless **all 5** neighbours lie within `plane_thresh = 0.1` m
of the plane.

**Residual** (scalar signed point-to-plane distance):
$$
r^{\text{lid}}_j = \mathbf n^\top \mathbf p^{W}_j + d.
$$

**Measurement Jacobian w.r.t. the pose at $t_j$** (then chained through the
spline). With $\mathbf q_j = R_{F_e\,L}\mathbf p^{L}_j + t_{F_e\,L}$:
$$
\frac{\partial r_j}{\partial \delta p_{W\,F_e}(t_j)} = \mathbf n^\top,\qquad
\frac{\partial r_j}{\partial \delta R_{W\,F_e}(t_j)} = -\,\mathbf n^\top R_{W\,F_e}(t_j)\,[\mathbf q_j]_\times
= \big([\mathbf q_j]_\times R_{W\,F_e}(t_j)^\top\mathbf n\big)^\top,
$$
the exact FAST-LIO measurement row ($C=R^\top n$, $A=[q]_\times C$). The extrinsic
columns ($B=[\mathbf p^L_j]_\times R_{F_e\,L}^\top C$, $C$) feed the **online
extrinsic variable** $T_{F_e\,L}$ when refinement is active (§3.6).

**Robust weight.** FAST-LIO's range-aware acceptance $s=1-0.9\,|r_j|/\sqrt{\|\mathbf
p^{L}_j\|}$, keep iff $s>0.9$, as the
inlier gate, **plus** a robust kernel (Huber, GNC-able; spec 11 / GTSAM
`noiseModel::Robust`) on the survivors so a few bad correspondences cannot dominate
the NLLS.

**Measurement covariance.** Range- and incidence-aware (improving on FAST-LIO's
isotropic `LASER_POINT_COV=0.001`):
$\sigma^2_{\text{lid}}(j)=\sigma_0^2+\sigma_r^2\|\mathbf p^{L}_j\|^2+\sigma_\theta^2(1-|\mathbf n^\top\hat{\mathbf r}_j|)$,
$\hat{\mathbf r}_j$ the ray direction (grazing-incidence inflation).

**Bounded factor count (deterministic cost, observability-preserving).** A dense
sweep can pass thousands of inlier point-to-plane correspondences; admitting all of
them makes the per-step Hessian-assembly cost and thus the solve latency
data-dependent and unbounded — incompatible with the deadline-bounded solve (§5.2).
The number of LiDAR residuals built per outer step is capped at
`max_lidar_factors` (default 1500). When the inlier set exceeds the cap it is
**subsampled by plane-normal stratification, never by uniform or spatial
decimation**, because uniform decimation preferentially thins the rarest normals and
silently collapses the very axis they constrain — exactly the weak axis the §4
observability exists to protect:

1. Bin survivors by their fitted plane normal $\mathbf n$ into a fixed set of normal
   strata (the six axis-aligned hemispheres plus an "oblique" stratum;
   `lidar_normal_strata`).
2. Allocate the budget across strata by **equal-then-proportional** filling: first
   guarantee each non-empty stratum a floor of `min_factors_per_normal` (default 50,
   clamped to that stratum's population) so a sparsely-sampled axis is never starved;
   distribute the remainder of the budget proportionally to stratum population.
3. Within a stratum, pick points by a **deterministic stride over a stable
   per-point key** (`t_offset_ns`, tie-broken by ring then range) and **rotate the
   stride phase by outer-step index** so that over consecutive steps the full inlier
   set is visited — a moving subset, never a frozen one, and identical on replay.

A point whose normal lies along a §4 weak axis (eigenvalue $<\kappa_{\text{deg}}$)
is **exempt from the cap entirely**: all such points are kept regardless of budget,
so degeneracy mitigation never competes with the factor cap. The retained
per-keyframe cloud (§6.5) is unaffected — the cap governs only the residuals built
for the solve, not the stored cloud.

### 3.2 Sparse-direct visual photometric (FAST-LIVO2)

FAST-LIVO2's *sparse-direct* visual residual (Appendix R.3). No
descriptors, no feature matching, **no triangulation** — the **LiDAR map points
double as visual map points** (unified voxel hash), each carrying
three-level $8\times8$ reference patches (`patch_size=8`, 3 pyramid levels) and the
reference pose/exposure. Depth always comes from the LiDAR map, so there is no VIO
initialisation problem.

**Measurement model.** Project map point $\mathbf P^W$ into the current camera using
the spline pose at the **image** stamp $t$ and the camera extrinsic
$T_{C\,F_e}$ (from `CalibrationSet`):
$\mathbf P^C = R_{C\,F_e}\big(R_{W\,F_e}(t)^\top(\mathbf P^W-p_{W\,F_e}(t))\big)+t_{C\,F_e}$,
$\mathbf u=\pi(\mathbf P^C)$ (`world2cam`). The **patch photometric residual** for
pixel offset $\Delta$, with the inverse-exposure affine model (Appendix R.3):
$$
r^{\text{vis}}_{\Delta} = \tau_{\text{cur}}\,I_{\text{cur}}(\mathbf u+\Delta) - \tau_{\text{ref}}\,P^{\text{warp}}[\Delta],
$$
$\tau$ the per-frame inverse exposure (state nuisance, §1.4), $P^{\text{warp}}$ the
reference patch affine-warped into the current view.

**Jacobian (photometric chain rule).** With image gradient $\nabla I=[I_u,I_v]$
(central differences), $J_\pi=\partial\mathbf u/\partial\mathbf P^C$ (2×3,
`computeProjectionJacobian`), $J_{\text{img}}=\tau_{\text{cur}}\,2^{-\text{level}}\nabla
I$ (Appendix R.3):
$$
\frac{\partial r}{\partial\delta R_{W\,F_e}(t)} = J_{\text{img}}J_\pi\,R_{C\,F_e}\,[R_{W\,F_e}(t)^\top(\mathbf P^W-p_{W\,F_e}(t))]_\times,\qquad
\frac{\partial r}{\partial\delta p_{W\,F_e}(t)} = -J_{\text{img}}J_\pi\,R_{C\,F_e}R_{W\,F_e}(t)^\top,
$$
chained through the spline as in §3 head. Exposure gets its own column
($\partial r/\partial\tau_{\text{cur}}=I_{\text{cur}}$, the 7th column);
the camera extrinsic $T_{C\,F_e}$ follows the same chain
(online-refined, gated, §3.6).

**Affine warp + LiDAR-plane prior.** Reference→current patches are affine-warped
using the **LiDAR plane normal** as the patch depth model (homography form
`getWarpMatrixAffineHomography`; Appendix R.3) — FAST-LIVO2's key improvement over
constant-depth. The homography warp is **only valid once the point carries an
initialised plane normal**: a freshly promoted map point whose normal has not yet
been fitted (or whose plane fit was rejected, §3.1) is **excluded from the
photometric residual** for that frame rather than warped against a guessed
fronto-parallel plane — it is reconsidered next frame once a normal exists, so a
bad-warp residual never enters the cost. Two further guards make the warp
numerically safe:

* **Degenerate-warp guard.** The $2\times2$ affine part $A$ of the warp is rejected
  if it is near-singular or near-degenerate — $|\det A|$ outside
  $[$`warp_det_min`$,$ `warp_det_max`$]$ (defaults 0.1 and 10.0) or condition number
  above `warp_cond_max` (default 50) — which is exactly the **pure-rotation / grazing
  / extreme-scale** regime where the LiDAR-plane homography blows up and would
  produce a NaN-valued or wildly stretched warped patch. A point whose warp is
  rejected is dropped for this frame (not fed an unwarped patch), so a degenerate
  geometry silently *drops* a point instead of silently *corrupting* the solve.
* **Normal-sign guard.** The plane normal is oriented toward the camera before the
  warp is formed (flip $\mathbf n\to-\mathbf n$ if $\mathbf n^\top(\,\mathbf P^C\,)>0$
  in the camera frame); a back-facing normal otherwise mirrors the patch and yields a
  spurious low residual.

`getBestSearchLevel` picks the pyramid level with a well-conditioned warp
determinant; coarse-to-fine over the iterated solve widens the basin.

**Candidate selection: one best per grid cell, fixed gate order.** Admitting every
in-FOV map point would make the visual residual count and the selection outcome
data- and order-dependent. The image is divided into a fixed cell grid
(`visual.grid_cell_px`, default 32 px) and **at most one** map point is kept per
cell — the single best-scoring candidate — capping the visual residual count at the
number of cells and giving an even spatial spread (no clustering on one
high-gradient corner). Each candidate is run through a **fixed, deterministic gate
sequence**, and the *first* gate it fails ends its evaluation, so the accept/reject
decision and the per-cell winner are a pure function of the input independent of
iteration order:

1. **grid occupancy** — the cell still has a free or worse-scoring slot;
2. **depth-continuity** — patch does not straddle an occlusion edge (Δdepth > 0.5 m
   rejected);
3. **warp / level** — homography warp passes the degenerate-warp and normal-sign
   guards above, and `getBestSearchLevel` yields a valid pyramid level;
4. **NCC** — normalised cross-correlation against the reference patch ≥ `ncc_thre`
   (`calculateNCC`);
5. **SSD** — photometric sum-of-squared-difference below `outlier_threshold = 1000`;
6. **robust kernel** — survivors enter the cost under the Huber/Cauchy kernel.

The per-cell "best" score is the NCC (gate 4); ties broken by the larger image
gradient then the smaller depth, both deterministic. Oblique-view patches (>60°
view-angle change between reference and current) are dropped at gate 3.

**Measurement covariance.** `img_point_cov` (FAST-LIVO2 `IMG_POINT_COV=100`;
Appendix R.3) for the gate-6 survivors, inflated for high-gradient/low-texture
patches.

### 3.2.1 The active visual-point map: reference-patch lifecycle and bounded growth

The visual map points are the LiDAR map points (shared voxel hash, §3.2 head), but
each visual point additionally owns the **reference-patch state** the photometric
residual reads: a small set of observations, each an $8\times8$ three-level patch
plus the camera pose and exposure $\tau$ it was seen under, and the plane normal
used for the warp. This state is L2-internal (never crosses the `IFrontEnd`
boundary), RAII-owned (smart-pointer, no raw `new`/`delete` of `VisualPoint`/
`Feature`, the sharp edge of Appendix R.3), and must be **bounded** — a multi-hour
mission would otherwise leak both observation lists and the active point set
without end.

**Reference-patch lifecycle.** For each visual point:

* **Add gate.** A new observation is appended only when it adds genuine parallax /
  appearance information: the view direction differs from every existing observation
  by at least `visual.ref_add_angle_deg` (default 10°) **or** the score improves on
  the current best by `visual.ref_add_score` — otherwise the frame is used for the
  residual but not stored, so near-duplicate views do not fill the list.
* **Observation cap.** The list is capped at `visual.ref_obs_cap` (default 30). When
  full, the **lowest-scoring** observation is evicted (min-score eviction), so the
  cap retains the most informative spread rather than the most recent frames.
* **Reference selection (medoid).** The patch actually warped into the current view
  is the **medoid** observation — the one whose viewing direction is most central to
  the retained set (minimises summed angular distance to the others), not simply the
  first or newest. The medoid is the most representative reference and resists the
  drift a fixed first-seen reference suffers as the platform moves.
* **Re-score.** On each use an observation's stored score is refreshed as a
  combination of NCC against the current patch and the cosine of the view-angle
  change (`visual.ref_score = w·NCC + (1−w)·cosθ`, `visual.ref_score_w` default
  0.7), and the medoid is recomputed; a reference whose appearance has decorrelated
  loses score and is eventually evicted by the cap.
* **Converged latch.** Once a point has accumulated `visual.ref_converged_obs`
  (default 8) observations spanning at least `visual.ref_converged_angle_deg`
  (default 30°) of view change with a stable medoid, it is **latched converged**: its
  observation list stops growing (re-scoring still runs) so a well-triangulated,
  well-observed point costs no further memory or per-frame add-gate work.

**Spatial eviction (bounded active set).** The active visual-point map is an
independent deterministic voxel-keyed store (an ordered map of voxel keys to
id-sorted point lists — not the LiDAR ikd-Tree; the two structures are separate by
implementation). As the platform moves, points that fall outside the **active map
box** — a cube of half-extent `visual.active_box_m` (default 60 m) centred on the
current live pose — are removed by a box eviction that mirrors the LiDAR map's
spatial eviction policy, so the two maps cover the same neighbourhood. A
box-deleted visual point releases its RAII-owned reference-patch state immediately.
Points are *evicted from the active solving/association set only*; the retained
per-keyframe clouds (§6.5) and the loop-closure store (spec 06/07) are the
durable record and are untouched, so a later loop closure re-introducing an old
region rebuilds its visual points from the store rather than relying on a leaked
active set. Eviction is deterministic (a pure function of pose and box extent),
preserving replay equivalence.

### 3.2.2 Rolling-shutter per-row evaluation (designed, deferred)

The camera model above evaluates every pixel of a frame at the single image stamp
$t$ (§2.1, mid-exposure), which is exact for a **global-shutter** sensor. A
rolling-shutter sensor exposes image rows sequentially over the readout interval, so
a pixel in row $y$ is actually seen at $t(y)=t_0 + y\cdot t_{\text{row}}$ — under
fast motion the constant-$t$ assumption skews the patch geometry just as an
undeskewed LiDAR sweep skews the cloud. Because the trajectory here is a continuous
function, the correct treatment is **structurally free**: evaluate each patch's
photometric residual at the spline pose $T_{W\,F_e}(t(y))$ of **its own row's time**
rather than the frame stamp — the exact analogue of the per-point LiDAR deskew
(§3.1), differing only in that the time offset is a function of image row instead of
`t_offset_ns`. The projection Jacobian chains through the spline at $t(y)$ unchanged.

This mode is **specified now but deferred** (`visual.rolling_shutter_en`, default
**false**; `visual.row_time_s` the per-row readout time from the camera datasheet).
It is opt-in because (i) the deployment cameras are global-shutter, where it is a
no-op, and (ii) it multiplies the spline evaluations per frame by the number of
distinct row-times sampled. When enabled it changes no interface and no boundary
type — it is purely a different time argument to the same evaluation — so it can be
turned on per platform without touching the rest of L2.

### 3.3 IMU derivative residual + bias random-walk (the CT IMU factor)

Unlike the iEKF, where the IMU is the *process model*, here the IMU is a **direct
residual** on the spline's analytic derivatives (the CLINS/Coco-LIC IMU factor;
Appendix R.1). For each raw sample $(a_m,\omega_m)$ at $t_i$, with
$\omega_{W\,F_e}(t_i)$ from the SO(3) spline and $\ddot p_{W\,F_e}(t_i)$ from the ℝ³
spline (§1.3):
$$
r^{\omega}_i = \big(\omega_{W\,F_e}(t_i) + b_g\big) - \omega_m,\qquad
r^{a}_i = \Big(R_{W\,F_e}(t_i)^\top\big(\ddot p_{W\,F_e}(t_i)-g_W\big) + b_a\Big) - a_m,
$$
weighted by $(\sigma_g^2,\sigma_a^2)$. Bias **random-walk** residuals tie
consecutive bias knots of the sliding timeline (§1.4; the Coco-LIC `BiasFactor`;
Appendix R.1):
$$
r^{b_g}_k = b_g^{k}-b_g^{k-1},\qquad r^{b_a}_k = b_a^{k}-b_a^{k-1},
$$
weighted by $(\sigma_{bg}^2,\sigma_{ba}^2)\Delta t$. Because the bias cadence
(`bias.knot_dt_ms`) is coarser than the outer solve cadence, a single knot bracket is
crossed by several successive sweeps. To avoid double-counting that one tie, a tie is
added **per solve only over the knot brackets that overlap the sweep's
$[t_{\text{begin}},t_{\text{end}}]$**, each weighted by $\sqrt{\text{overlap
fraction}}$ of its bracket; since quadratic information sums linearly, the
accumulated information across the overlapping sweeps equals **exactly one full tie**
once the bracket is fully covered (and a data gap leaves proportionally less
constraint, as the random-walk model demands). $r^{\omega}$ touches the 4
active SO(3) control points + the local $b_g$ knot; $r^{a}$ touches 4 SO(3) + 4 ℝ³
control points + the local $b_a$ knot + gravity. The IMU therefore **densely
constrains** the spline between sparse LiDAR/visual measurements — this is what
keeps under-determined control points (§10, failure 2) regularised.

> **Optional under-excitation regularizer (low-weight, gated).** When the IMU is
> nearly static — `n_cp` has dropped to its floor and the §4 observability shows a
> weak axis — there can be control points in the segment that no LiDAR, visual, or
> GNSS residual touches and that the near-constant IMU residual leaves nearly
> unobservable, producing solver jitter (§10, failure "control points
> under-constrained"). A **low-weight smoothness residual** penalises the spline's
> highest unconstrained derivative on exactly those spans: a **jerk** residual
> $r^{\text{jerk}} = \dddot p_{W\,F_e}(t)$ on the ℝ³ spline (the third virtual-time
> derivative, available in closed form from the basis) and an **angular-acceleration**
> residual $r^{\dot\omega} = \dot{\boldsymbol\omega}_{W\,F_e}(t)$ on the SO(3) spline,
> each evaluated at the segment's knot midpoints. It is **off unless** the segment is
> flagged under-excited (`motion_regularizer_en`, default on; engaged only when the
> peak excitation statistics $N_\omega,N_a$ of §5.5 are below
> `motion_reg_excitation_floor` *and* a §4 eigenvalue is below $\kappa_{\text{deg}}$),
> and its weight `motion_reg_weight` (default 1e-3, relative to the IMU accel weight)
> is small enough to bias nothing once any real measurement constrains the span.
> Because it is a pull toward *constant velocity / constant rate*, not toward a
> particular pose, it cannot fight a true motion the data supports; it only removes
> the null-space float. The regularizer is excluded from the §4 information matrix
> and the marginalization prior so it never inflates apparent observability or
> double-counts across windows.

> **The IMU residual set is gauge-deficient: a pure-IMU span has a 9-DoF null
> space.** The gyro residual depends only on the *body* angular velocity and the
> accel residual only on $R^\top(\ddot p-g)$, so the IMU cost is invariant under (i) a
> **global left rotation** of the whole trajectory together with the matching rotation
> of $g_W$ (3 DoF), and (ii) **any affine function added to the position spline**,
> $p(t)\mapsto p(t)+a+bt$ (6 DoF: constant offset + constant velocity), because an
> affine term has zero acceleration. That is **9 unobservable DoF** from inertial
> residuals alone. In production the gauge is anchored by the **other** residuals and
> the marginalization prior, not by the IMU: the global rotation and absolute
> position/velocity are pinned by the **LiDAR point-to-plane** factors (§3.1) tying
> the trajectory to the map, by **GNSS** when present (§3.4), and by the
> **marginalization prior on the boundary control points** carried from the previous
> window (§5.4); the very first window fixes it with the first-keyframe
> `AbsolutePrior` (§6.4). A span covered **only** by IMU residuals (cold start before
> the first LiDAR association, or a LiDAR dropout interval) is therefore
> gauge-deficient and its absolute pose, velocity and yaw-vs-gravity are determined
> only up to this null space until a map-anchoring residual re-enters — such a span
> must not be emitted as an absolute anchor (§6.2 forbids keyframes while the window
> is uninitialised; §10 covers the dropout case).

### 3.4 GNSS (conservative absolute position)

Absolute antenna position, fused conservatively, gated by `fix`/`cov_enu` (spec 01
§4.4). With antenna lever-arm $t_{F_e,\text{ant}}$ (`CalibrationSet`), the antenna
world position **at the fix's own time** $t$ is
$p_{W,\text{ant}}=p_{W\,F_e}(t)+R_{W\,F_e}(t)\,t_{F_e,\text{ant}}$;
convert the fix LLA→ENU→$W$ via the latest datum/$T_{MW}$ snapshot (held fixed in
L2 — L2 reads, L3 owns it):
$$
r^{\text{gnss}} = z_{W}(\text{fix}) - p_{W,\text{ant}} \in\mathbb R^3,
$$
$\partial r/\partial\delta p_{W\,F_e}(t)=-I$,
$\partial r/\partial\delta R_{W\,F_e}(t)=R_{W\,F_e}(t)[t_{F_e,\text{ant}}]_\times$,
chained through the spline.

**Bind to the spline pose at the fix time — never the nearest keyframe.** Because
$T_{W\,F_e}(\cdot)$ is a continuous function, the residual is evaluated at exactly
the fix's PTP timestamp $t$ — $p_{W\,F_e}(t)$ is the spline *interpolated* to $t$,
not the pose of the nearest control point or keyframe. Snapping a GNSS fix to a
neighbouring discrete pose would bake the antenna's displacement over the
intervening interval (up to a full keyframe spacing at vehicle speed) into the
residual as a systematic position error, the classic latent GNSS bug. Continuous-
time evaluation removes it for free; this is one of the structural advantages of the
CT trajectory and L2 relies on it for all four sensors (§2.2). *(Cross-spec
interaction: the L3 absolute-fix interface `IBackEnd::add_absolute(GnssFix,
nearest_kf_id)` in spec 01 §7 carries a `nearest_kf_id`. Inside L2 that id is only a
**graph-attachment hint** for which keyframe node the L3 factor hangs near; the
**residual geometry is bound to the interpolated spline pose at fix time**, and L3
must interpolate its own pose to the fix time rather than treat `nearest_kf_id` as
the measurement location. L3 honours exactly this in spec 05 §6.3.)*

**Fix-quality covariance floors and innovation gating.** The fix is weighted by
`cov_enu`, but a receiver's reported covariance is optimistic, so each weight is
**floored by fix type** before use — the actual covariance is
$\max(\text{cov\_enu},\,\Sigma_{\text{floor}}(\text{fix}))$ element-wise on the ENU
diagonal:

| `GnssFix::FixType` | horizontal floor $\sigma_h$ | vertical floor $\sigma_v$ | config key |
|---|---|---|---|
| `RTK_Fixed` | 0.05 m | 0.10 m | `gnss.floor_fixed` |
| `RTK_Float` | 0.50 m | 1.00 m | `gnss.floor_float` |
| `DGPS` | 1.50 m | 3.00 m | `gnss.floor_dgps` |
| `SPP` | 3.00 m | 6.00 m | `gnss.floor_spp` |
| `None` | — fix rejected, never used — | | |

Each accepted fix passes an **innovation (Mahalanobis) gate**: the fix is admitted
only if the normalised innovation
$\|r^{\text{gnss}}\|_{\Sigma}=\sqrt{r^\top(\Sigma_{\text{floored}}+\Lambda_{\text{pose}}^{-1})^{-1}r}\le k_{\text{gnss}}$
(`gnss.innovation_k`, default 3.0), where $\Sigma_{\text{floored}}$ is the
floored fix covariance and $\Lambda_{\text{pose}}^{-1}$ the §4 pose marginal at $t$,
so the gate accounts for the current odometry uncertainty and does not reject good
fixes after a long GNSS gap when odometry has itself drifted. A fix that fails the
gate is **not** silently dropped: it increments a reject counter and the **re-
acquisition persistence** rule applies — after any gap or a run of gated-out fixes,
GNSS is re-admitted only once `gnss.reacquire_count` (default 5) consecutive fixes
pass the gate, preventing a single post-gap multipath fix from snapping the window.
Survivors enter the cost under a robust kernel so one in-gate-but-still-bad fix
cannot dominate. All of this is conservative-by-design (§3.4 policy); the
authoritative GNSS robustness and datum estimation remain L3's (spec 05 §6).

**Policy.** GNSS in **L2 is optional and conservative** — it bounds drift over long
open-sky stretches and aids yaw. The **authoritative** GNSS fusion and datum
estimation live in **L3** (spec 05). L2 runs GNSS-disabled (tactical / GNSS-denied)
with no code-path change — it is just an absent residual.

### 3.5 The joint window cost

The fixed-lag MAP cost minimised each outer step (the Coco-LIC window cost;
Appendix R.1) over the active state $X$ = {control points in the window} ∪ {bias knots} ∪
{gravity} ∪ {window exposures} ∪ {active online extrinsics}:
$$
\mathcal C(X) = \sum_j \rho\!\big(\|r^{\text{lid}}_j\|^2_{\Sigma_{\text{lid}}}\big)
+ \sum_{p,\Delta}\rho\!\big(\|r^{\text{vis}}_{p,\Delta}\|^2_{\Sigma_{\text{photo}}}\big)
+ \sum_i\Big(\|r^{\omega}_i\|^2_{\Sigma_{g}}+\|r^{a}_i\|^2_{\Sigma_{a}}\Big)
+ \sum_k\|r^{b}_k\|^2_{\Sigma_{b}}
+ \sum_k\|r^{\tau}_k\|^2_{\Sigma_{\tau}}
+ \sum\rho\!\big(\|r^{\text{gnss}}\|^2_{\Sigma_{\text{gnss}}}\big)
+ \underbrace{\sum_m\Big(\|r^{v}_m\|^2_{\sigma_{\text{vel}}^2}+\|r^{\dot\theta}_m\|^2_{\sigma_{\text{rate}}^2}\Big)}_{\text{tail anchors past }t_{\text{end}},\ §1.3.1}
+ \underbrace{\|X\boxminus X^{\text{prior}}\|^2_{\Omega_{\text{prior}}}}_{\text{marginalization prior, §5.4}}.
$$
$\rho$ is a Huber/Cauchy robust kernel. The **tail-anchor** terms
($r^{v}$ a world-velocity anchor with $\sigma_{\text{vel}}=0.2$ m/s, $r^{\dot\theta}$
a body-rate anchor with $\sigma_{\text{rate}}=0.1$ rad/s, evaluated at `window_dt/4`
stride past the newest measurement, §1.3.1) brace the unmeasured span toward the
seed's constant extrapolation; they are deliberately **excluded from the
marginalization prior** — they never touch the marginalized knot, so they never
enter $\Omega_{\text{prior}}$ and cannot double-count. The **marginalization prior**
is the *single* clean representation of all past data inside L2 (§5.4); the IMU
appears **exactly once**, as residuals (never also as a prior on the same interval).
That single-representation discipline is what the L2→L3 handoff must also respect
(§6.4).

### 3.6 Extrinsic / gravity / exposure refinement gating

Online refinement of $T_{F_e\,L}$, $T_{F_e\,C}$, gravity $g_W$, and inverse-exposure
$\tau$ is **on by default** (resolved default) but **enabled only when the relevant
DoF is excited** (extrinsics/gravity: sufficient rotational + translational
excitation over the window, read from the §4 observability; exposure: sufficient
photometric evidence this frame). Under degenerate motion the relevant Jacobian
columns are **frozen** (zeroed — the FAST-LIO `extrinsic_est_en` and FAST-LIVO2
`exposure_estimate_en` gates; Appendix R.5, R.3), preventing drift. A frozen $\tau$
is not removed from the state — it is **held at its random-walk prediction** (the
neighbour-tied / prior value of §1.4), so the photometric residual stays
well-defined and the exposure chain resumes free refinement once evidence returns.
The freeze decision is logged (§8). Per spec 01 §5.3 the **authoritative** refined
extrinsics are an **L3** product fed back as a versioned `CalibrationSet` snapshot;
L2's online estimate is a fast local refinement that L3 ratifies.

> **The IMU biases are not part of this gate.** Both the gyro bias $b_g$ and the
> accel bias $b_a$ are **always free**, regularised by the random-walk ties (§3.3)
> and held inside the §5.2 box bounds — there is no excitation gate that freezes the
> gyro bias under poor observability. (An earlier design froze $b_g$ off an
> observability flag; that gate is removed — a clamped bias adds no robustness the
> box bound and random-walk tie do not already give, and freezing it merely stalls
> convergence when excitation returns.)

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
identical block as $H^\top R^{-1}H$; Appendix R.5 — useful for
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
naturally down-weights ill-constrained directions (spec 05 maps `observability` to the noise
model). Under full degeneracy L2 may additionally **clamp** the update along weak
directions (let IMU/GNSS carry them — the X-ICP "solution remapping"), gated by
`degeneracy_handling` mode (§10). Because LiDAR and visual residuals sum into one
Hessian *before* inversion, a geometrically degenerate scene that the photometric
block constrains (or vice-versa) is automatically rescued.

---

## 5. The sliding-window solve, marginalization, and adaptive knots

### 5.1 The outer loop

```
on each measurement ingest:
  append to per-stream time-ordered buffers; advance spline end if needed
on outer cadence (~0.1 s of new trajectory, knot_dt):           # Coco-LIC outer step (App. R.1)
  (1) extend the spline: place adaptive knots over the new segment (§5.5),
      seeded by IMU-only integration (continuity of pose/vel/orientation)
      — seeding samples the seed trajectory one local knot interval BACK (§1.2);
      re-seed the stale tail (reseedFrom, index >= interval(t_end)+4, and +3 when
      its peak support u_end^3/6 < 0.02) from the fresh IMU seed each sweep (§1.3.1)
  (2) associate: for each new LiDAR point, kNN + plane fit at its spline pose (§3.1);
      for the new image, retrieve visual submap with LiDAR depth (§3.2)
  (3) build the Ceres problem: §3 residuals over active control points + biases
      + gravity + exposure + active extrinsics, + the marginalization prior (§5.4);
      PIN the unsupported tail knots constant (SetParameterBlockConstant at their
      IMU warm-start values; same index rule as step (1)), and add the tail
      VelocityAnchor/RateAnchor residuals at window_dt/4 stride past t_end (§1.3.1)
  (4) solve as nested loops (fixed-cadence re-association):                       # §5.1.1, §5.2
      repeat up to max_outer_iters:
        - (re)associate LiDAR planes (§3.1) + visual patches (§3.2) at current pose
        - inner Ceres LM solve: reassoc_steps inner iterations, analytic Jacobians,
          band-sparsity, deadline-bounded (§5.2)
        - stop early once the keyframe pose shift since the last association
          is < assoc_shift_thresh (associations have stopped moving)
  (5) extract Λ_pose, observability (§4)
  (6) marginalize control points/biases that left the window → update prior (§5.4)
  (7) update the visual map (promote new LiDAR pts to visual pts; add obs under the
      add-gate; refine ref patches; recompute medoid; box-delete out-of-box pts) # §3.2.1
  (8) maybe_emit_keyframe (§6.2); push retained per-keyframe cloud to IKeyframeStore (§6.5)
```

### 5.1.1 Fixed-cadence re-association

LiDAR plane associations (§3.1, the kNN + plane fit) and visual patch selections
(§3.2) are **linearisation-point dependent**: they are computed against the current
spline pose, and as the solve moves the control points the correct correspondence
for a point can change (a different plane, a different patch, or a now-rejected
outlier). Re-associating on **every** inner Ceres iteration is both expensive and a
source of non-determinism (the correspondence set churns with each LM trial step);
never re-associating bakes in stale, possibly wrong matches from the warm-start
pose. The solve is therefore a **fixed-cadence** alternation of two nested loops:

* The **outer (association) loop** runs at most `max_outer_iters` times (default 4,
  range 3–5). At the top of each pass it rebuilds the LiDAR and visual residual sets
  from scratch against the *current* converged-so-far pose, then hands a fresh Ceres
  problem to the inner loop.
* The **inner (solve) loop** is `reassoc_steps` LM iterations (default 2) on the
  fixed correspondence set, under the deadline of §5.2.

The outer loop **terminates early** when associations have stabilised: if the
keyframe-time pose has moved by less than `assoc_shift_thresh` (default 0.02 m
translation **and** 0.2° rotation, both must hold) since the previous association
pass, a re-association would not change the correspondence set, so the loop stops.
This makes association cost proportional to how far the pose actually travelled this
step — typically one or two passes in steady tracking, the full `max_outer_iters`
only on a hard manoeuvre or after a window restart. The bias bounds
of §5.2 apply within every inner solve, so a single bad pass cannot run the pose
away before the next re-association sees it.

The determinism path (§5.2, fixed-iteration replay) runs exactly `max_outer_iters`
outer passes with the early-stop disabled, so the correspondence sequence is a pure
function of the input and is bit-reproducible.

### 5.2 Solver

* **Solver:** **Ceres** (Levenberg-Marquardt; analytic cost functions; SO(3)
  manifold via `LocalParameterization`; built-in Schur for marginalization) — the
  choice of all three CT references (Appendix R.1). GTSAM is reserved for the L3
  global iSAM2 graph; the CT window is its own Ceres problem feeding L3.
* **Band-sparsity:** each measurement touches 4 control points per spline ⇒ the
  Hessian is banded ⇒ sparse Cholesky / Schur on biases keeps cost bounded
  (Appendix R.1). Converges in 4–5 iterations, comparable to CLINS' ~200 ms
  budget but well within the Orin's GPU-offloaded headroom (LiDAR/visual map ops
  run on GPU, §11).
* **Iteration cap / tolerance:** `max_iterations`, per-DoF `epsi` (mirroring the
  iEKF `epsi=1e-3`; Appendix R.5).
```
██████████████████████████████████████████████████████████████████████████████
██                                                                          ██
██   REAL-TIME HEADROOM DEBT: THE SPLINE RESIDUALS RUN ON CERES AUTODIFF    ██
██                                                                          ██
██   Every LiDAR / IMU / visual residual differentiates through the 4-knot  ██
██   cubic-spline evaluation with DynamicAutoDiffCostFunction. Measured     ██
██   solves reach 70-170 ms against the 90 ms / 10 Hz budget ON THE DEV     ██
██   BOX; the Jetson Orin target is slower. The deadline guard makes this   ██
██   SAFE (best-iterate-so-far is published, deadline_hit telemetry fires)  ██
██   but NOT FREE: a starved solve is a shallower solve.                    ██
██                                                                          ██
██   THE KNOWN, MECHANICAL FIX IS ANALYTIC JACOBIANS for the spline         ██
██   residuals (every reference CT system ships them: CLINS, Coco-LIC,      ██
██   SLICT). This is the single highest-leverage performance item in the    ██
██   front-end and MUST land before any real-time claim on the Orin.        ██
██   Watch: frontend/deadline_hit, frontend/solve_ms (spec 09).             ██
██                                                                          ██
██████████████████████████████████████████████████████████████████████████████
```

* **Deadline-bounded solve (real-time control on the wall-clock path).** The window
  solve runs under a hard wall-clock budget so that front-end latency is *enforced
  at runtime*, not merely measured after the fact. Ceres is given
  `solver.time_limit_ms` (default 90, validated at 10 Hz on the dev box) as its
  `max_solver_time_in_seconds`, with
  `solver.min_iterations` (default 2) and `solver.max_iterations` (default 5)
  bracketing the count: the solver always completes at least the minimum iterations
  (so a single starved step never publishes a barely-improved pose) and is cut off
  at the deadline or the maximum, whichever comes first. The budget is the sum over
  the §5.1.1 outer/inner passes, not per inner solve, so re-association cannot blow
  the step. Hitting the deadline before convergence is **not** a failure — it yields
  the best iterate so far and is surfaced as telemetry
  (`scalar("frontend/deadline_hit")`, spec 09 §5.1, §8); a sustained deadline-hit rate is
  the operator signal that the window/knot budget exceeds the platform, handled by
  the §10 degraded path, never by silently running long. (A `deadline_hit` boolean
  on the canonical `FrontEndDiagnostics` would carry the same signal on the
  synchronous return; that struct is owned by spec 01, so the field is *flagged* for
  amendment there rather than assumed here.)
  **The determinism / replay path ignores the time limit** and runs a fixed
  iteration schedule (exactly `solver.max_iterations` inner steps per the fixed
  `max_outer_iters` passes of §5.1.1) so a recorded run is bit-reproducible
  independent of wall-clock load. These two modes are selected by the same
  `deterministic` flag the rest of L2 honours. The 90 ms budget holds the 10 Hz
  cadence on the dev box; the **analytic Jacobians** (§1.3) — not a tighter deadline —
  remain the lever for reclaiming headroom on the Jetson, where the dev-box wall-clock
  margin does not directly carry over.
* **Bias box bounds (pre-restart stability guard).** Before a window restart (§10) is
  ever triggered, the solve defends itself from a divergent bias estimate. Every knot
  of the sliding bias timeline (§1.4) — **both** $b_g$ and $b_a$, with no excitation
  gate on either — is held inside physically-motivated **box bounds**:
  $|b_g|\le$ `bias.gyr_max` (default 0.5 rad/s) and $|b_a|\le$ `bias.acc_max`
  (default 5.0 m/s²), installed as Ceres lower/upper parameter bounds on each live
  knot's gyro and accel block, so a transient outlier burst cannot drive any bias
  knot to an absurd value that then corrupts the IMU residual for the whole window.
  The bounds mean §10's window restart is a genuine last resort reached only on true
  divergence (residual blow-up with a bias knot pinned at a bound), not on a
  recoverable excursion. There is **no per-iteration step clamp**: the trajectory
  null space the clamp once guarded is removed structurally by the tail-knot pinning
  of §1.3.1, and a step clamp over a structurally-sound solve only blocks recovery
  from a poisoned seed.

### 5.3 Sliding window (true fixed-lag)

The active set is the control points spanning the current window
(`window_seconds`, ~0.3–1.0 s) plus a small overlap of recent ones; older control
points are **marginalized** (§5.4), not held fixed (marginalizing keeps their
information consistently, unlike CLINS' "hold-fixed" which biases as the window
slides; Appendix R.1). A fixed number of active control points ⇒ bounded
per-step cost.

### 5.4 Marginalization (the single clean prior)

When the window slides, the control points/bias knots leaving the window, and the
residuals connecting *only* to them, are **marginalized via Schur complement** into
a dense linear prior on the remaining **boundary** variables (VINS-Mono /
Coco-LIC-style; Appendix R.1). The retained prior set =
{control points shared between the dropped and kept windows} ∪ {current bias
knots} ∪ {gravity}. This prior is the **only** memory of past data inside L2 and is
**mutually exclusive** with re-introducing those measurements — guaranteeing no
double counting. (Linearization staleness is a known risk; §10 failure 6.) The same
single-representation discipline is carried to the L2→L3 boundary in §6.4.

> **Use-after-free guard on the front-trim.** The marginalization prior holds raw
> pointers into the **kept-block** knot storage (the deque nodes shared with the
> previous window). The `SplineWindow` front-trim (`dropOldest`) must therefore never
> free a deque node those pointers reference: the number of knots dropped is bounded
> by `lowestKnotIndexOf(prior kept-block pointers)`, so the prior never dangles into
> freed storage. The sliding **bias timeline** carries the identical guard — its
> `dropOldest` is bounded by `BiasKnots::lowestKnotIndexOf` over the same prior-held
> pointers — so neither the spline nor the bias front-trim can outrun the prior.
> (Deque storage is what makes this sound: `pop_front` invalidates only the erased
> node, leaving every surviving knot's address stable as a Ceres parameter block.)

> **Invariant — enumerate knot indices, never sample by time.** Any code that must
> *cover a discrete set of knots* — which knots to marginalize (step (6)), which tail
> knots to pin (§1.3.1), which knots a prior references — must **enumerate the deque
> indices** of that set, never sample the curve at a set of times and infer the knots
> from `segmentFor(t)`. Time-sampling cannot guarantee it visits every knot exactly
> once: near a knot boundary or under a knot-density change two sampled times map to
> the same support set while a knot between them is missed, or one knot is counted
> twice. (This was the concrete cause of the clamp-snapshot explosion bug, §10: a
> per-knot snapshot built by time-sampling skipped knots and the resulting partial
> coverage poisoned the solve. The clamp is gone, but the discipline binds the
> marginalization and pinning code that remains.)

### 5.5 Adaptive knots (Coco-LIC)

Knot density follows motion (Coco-LIC's adaptive scheme; Appendix R.1). Fixed **outer cadence**
`knot_dt ≈ 0.1 s`; within each outer segment, the number of control points
`n_cp ∈ {1..n_cp_max}` is chosen from IMU-measured dynamics over the **raw samples
of that segment** — never from a summed-then-normed aggregate. The gating statistic
is the **peak per-sample excitation in the segment**, taken on quantities that are
frame-invariant so attitude change never hides motion:
$$
N_\omega = \max_{i\in\text{seg}}\,\big\|\boldsymbol\omega_{m,i}-b_g\big\|,\qquad
N_a = \max_{i\in\text{seg}}\,\big|\,\|\mathbf a_{m,i}-b_a\| - |g|\,\big|,
$$
the bias-corrected **body angular rate magnitude** and the **gravity-removed
specific-force magnitude** ($|g|=9.81$). `n_cp` is the larger of the two mapped
thresholds. Both statistics use the *magnitude of each sample taken individually*,
so a fast pure rotation — where the world-frame vectors $R_i\boldsymbol\omega_i$
sweep through every direction and a vector sum cancels toward zero — produces a
**large** $N_\omega$ and correctly raises the knot density. (Removing gravity by
subtracting its *magnitude* from $\|\mathbf a_m-b_a\|$, rather than subtracting the
world vector $g_W$ after rotating each sample, keeps $N_a$ exact under any attitude
without needing the current orientation estimate; it slightly under-reads when
specific force is near-orthogonal to gravity, which the angular term $N_\omega$ and
the IMU residual itself then cover.)

The mapping is monotone and piecewise-constant with hysteresis: $N_\omega$ is
compared against the rising thresholds `knot_omega_thresh[]` (rad/s) and $N_a$
against `knot_accel_thresh[]` (m/s²), each band adding one control point up to
`n_cp_max`; a transition down a band requires the statistic to fall **below** the
band edge by `knot_density_hysteresis` (default 0.15, fractional) so density does
not chatter at a threshold. Knots are placed uniformly within the segment in
**virtual time** so the uniform basalt kernels apply (Appendix R.1). Net effect:
more control points under aggressive motion, fewer when smooth — accuracy when
needed, cheap when not. The default thresholds are tuned against the released
Coco-LIC code (Appendix R.1) but the gating *quantity* is fixed normatively as the
peak per-sample form above.

> **Shipped gated at `n_cp = 1` pending a non-uniform-grid validation campaign.** The
> adaptive-density machinery above (`n_cp > 1`) is **design-present but currently
> shipped disabled** — `ct.n_cp_max` defaults to **1**, the uniform-spline special
> case — because it is **not yet validated end-to-end and is measurably wrong** in
> interaction with the current tail machinery (§1.3.1): on the validation bag a run
> with adaptive density enabled regressed ATE to **2.5 m** versus **0.029 m** for the
> uniform spline. The likely culprit is that a knot-density change shifts the
> virtual→real slope (and so the tail-knot index arithmetic and the
> $u_{\text{end}}^3/6$ support test) in a way the pinning/anchor code has not yet been
> validated against. Until a dedicated non-uniform-grid campaign clears it, `n_cp_max`
> stays at 1; the design and code remain in place behind that gate.

> **Continuity across a knot-density transition.** The virtual→real time map is a
> monotone piecewise-linear stretch whose slope $\mathrm dv/\mathrm dt$ is constant
> *within* one outer segment but **jumps** at a boundary where `n_cp` changes (the
> local knot spacing changes by the `n_cp` ratio). The spline is $C^2$ in *virtual*
> time everywhere, but real-time derivatives are the virtual ones scaled by powers of
> that slope ($\dot p_{\text{real}}=\dot p_v\,s$, $\ddot p_{\text{real}}=\ddot p_v\,s^2$),
> so at a density change the slope discontinuity makes the trajectory only $C^1$ in
> **real** time across that join: real velocity stays continuous, but real
> acceleration and body angular acceleration step. Two consequences the rest of this
> spec relies on: (i) a **constant-rate motion is exactly representable only within a
> run of uniform-density segments** — across a slope break a constant real velocity/
> rate incurs a small representation residual until the four supporting knots are all
> on one side of the transition; (ii) place density changes where motion is benign,
> and let the IMU residual (§3.3), which is evaluated in real time, absorb the
> $C^2$-in-real-time deficit at the seam rather than fighting it. Within any
> uniform-density run (including the `n_cp ≡ 1` uniform-spline case) the map is linear
> with constant slope and full $C^2$-in-real-time continuity is recovered.

### 5.6 Optional reference/oracle: the iEKF baseline (not a milestone)

A FAST-LIO2-style sequential ESIKF over a single scan-end `NavState` (Appendix R.5;
LiDAR point-to-plane then FAST-LIVO2 photometric, sequential update in the FAST-LIVO2
order) is provided **only** as an offline differential-test oracle behind the same
`IFrontEnd`: it must produce a `KeyframePacket` indistinguishable in *type* from the
CT front-end, and its trajectory/observability are cross-checked against the CT
output on recorded bags (the iEKF forms the identical $H^\top R^{-1}H$ pose block,
§4.1, making the comparison direct). It is never on the live path, never a "v1",
and the system is not organised around it.

#### 5.6.1 The iEKF's internal error chart (decoupled, not the boundary box-plus)

`NavState::boxplus` (spec 01 §3.2) is the **boundary** convention: its pose part is
the *coupled* $SE(3)$ exponential (`Pose::boxplus`, $T\boxplus\xi = T\,\mathrm{Exp}(\xi)$,
$\xi=[\rho;\phi]$). The iEKF realisation does **not** retract along that chart
internally. Its predict Jacobians and its analytic measurement rows are derived in
a **decoupled** chart, matching FAST-LIO's own state manifold (where `pos` is a
`vect3` and `rot` an `SO3` retracted independently; Appendix R.5):

* **position** is world-additive — $p \leftarrow p + \delta\rho$ (Euclidean);
* **rotation** is a right $SO(3)$ perturbation — $R \leftarrow R\,\mathrm{Exp}(\delta\phi)$;
* **velocity, biases** are Euclidean;
* **gravity** is handled on $S^2$ (§5.6.2).

The filter implements this chart as private `boxplusNav`/`boxminusNav` helpers
(error order `[p|R|v|b_g|b_a|g]`, translation-first, the same `NavState::kDof=18`).
This is what makes the analytic rows exact: with a decoupled chart the
point-to-plane measurement row is $\partial r/\partial\rho = n^\top$ (a pure
world-frame plane normal — FAST-LIO's `h_x` position block is literally the plane
normal; Appendix R.5) and $\partial r/\partial\phi = -\,n^\top R\,[p_b]_\times$, and
the predict transition has $\partial\dot p/\partial v = I$, $\partial\dot R/\partial b_g
= -I$ with no $SE(3)$ cross-coupling. Under the coupled $SE(3)$ box-plus the
translation update would mix in the rotation increment whenever $R\neq I$, so the
position columns of $H$ would no longer be $n^\top$ and the authored Jacobians would
be wrong. The boundary `NavState::boxplus` (coupled) is used **only** to produce the
typed `NavState`/`KeyframePacket` pose at the interface; it is never the chart the
covariance $P$ or the gain are expressed in.

#### 5.6.2 Gravity on $S^2$: tangent-plane projection, not the full $S^2$ chart

Gravity is kept at fixed magnitude $|g|=9.81$ (the $S^2$ rationale of §1.4). FAST-LIO
realises $S^2$ with an explicit orthonormal tangent basis $B_x(g)$ and the exact
exponential retraction $g\leftarrow R(B_x\,\delta)\,g$, carrying $M_x/N_x$ transport
blocks through a strictly 2-DoF covariance (the `S2` boxplus and the `df_dx` gravity
block; Appendix R.5).
The oracle uses a lighter **first-order tangent-plane** scheme equivalent to first
order: the 3-DoF gravity increment is projected onto the plane orthogonal to the
current $g$ (removing the radial component), the state is retracted additively, then
$g$ is **renormalised** back to $|g|$. Gravity is thus carried as a full 3-vector with
a 3-DoF covariance block, but the radial direction is never excited (zero process
noise; the increment's radial part is projected out at retraction), so the magnitude
does not drift. This is the oracle's deliberate simplification of the full $S^2$
chart machinery.

#### 5.6.3 Gain form for the high-dimensional LiDAR update

A LiDAR sweep contributes $m \gg n$ point-to-plane rows ($n=18$). The oracle solves
the iterated update in **information form**, accumulating $H^\top R^{-1}H$ (18×18) and
$H^\top R^{-1}r$ one effective point at a time (never materialising the $m\times 18$
$H$), then taking the step from
$$(H^\top R^{-1}H + P^{-1})\,\delta = -\big(H^\top R^{-1}r + P^{-1}(x\boxminus x_{\text{prop}})\big),$$
and setting the posterior $P \leftarrow (H^\top R^{-1}H + P^{-1})^{-1}$. This is the
maximum-a-posteriori normal-equation form, algebraically identical to FAST-LIO's
Woodbury Kalman gain $K = P H^\top (HPH^\top/R + I)^{-1}/R$ with update
$\delta = -Kr + (KH-I)(x\boxminus x_{\text{prop}})$ used in its information-form
branch (Appendix R.5), but it inverts the fixed $18\times18$ instead of the
sweep-sized $m\times m$ — the natural choice when $m$ is in the thousands. The
$P^{-1}(x\boxminus x_{\text{prop}})$ term is the prior pull-back toward the frozen
propagated prior, the same role as FAST-LIO's $(KH-I)\,dx_{\text{new}}$.

> **First-order chart-transport note.** FAST-LIO transports the prior delta
> $dx_{\text{new}} = x\boxminus x_{\text{prop}}$ through the inverse right-Jacobian
> ($A(\delta\phi)^\top$ for $SO(3)$, $N_xM_x$ for $S^2$; Appendix R.5)
> before forming the step, accounting for the curvature between the relinearisation
> point and the propagated mean. The oracle takes this transport as identity to
> first order (the decoupled position/velocity/bias blocks transport trivially, and
> the rotation increment is small across the few iterations of one sweep). This is a
> first-order approximation acceptable for an offline cross-check oracle; it is
> **not** exact when the rotation moves substantially within a single update.

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
  GaussianBlock<6> constraint_cov;       // 6x6 ROTATION-FIRST [rx,ry,rz,tx,ty,tz] block
                                         // (GTSAM Pose3 boundary); meaning set by kind
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
| `T_relto_this`,`constraint_cov` | $\hat T_{\text{prev}}^{-1}\hat T_{\text{cur}}$ (both spline-evaluated); marginal **relative** covariance read directly from the window posterior / marginalization, **inflated on weak axes by `observability`** (§4.3), reordered once into rotation-first `[rx,ry,rz,tx,ty,tz]` (the GTSAM-boundary exception, §1.5). |
| `observability` | $\Lambda_{\text{pose}}$ pose block (§4). |
| `cloud_body` | per-point spline-deskewed scan in $F_{e,\text{stamp}}$, retained in `IKeyframeStore` (§6.5). |
| `image`/`T_body_cam` | the camera frame at this KF + the `CalibrationSet` $T_{F_e\,C}$ snapshot used. |
| `imu_summary` | filled only on `ImuPreintegration` restart (§6.4). |
| `calib_version`/`frontend_kind` | snapshot version; `1` (CT). |

Every field the boundary demands is pinned: timestamp; **SE(3) pose in a named
frame**; **optional velocity+bias with an "included" flag**; **covariance block**
(rotation-first `[rx,ry,rz,tx,ty,tz]`, the §1.5 exception; spec 01 §3.3/§6.4);
**6 per-axis observability with reference frame**;
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
  virtual void ingest(const PreprocessedGroup&) = 0;      // PRIMARY: one sweep bundle, triggers the window solve
  virtual void ingest_imu_live(const ImuSample&) = 0;     // live-state propagation only, never a solve
  virtual void apply_correction(const GraphUpdate&) = 0;  // L3 feedback at safe points (§7.3)
  virtual NavState live_state() const = 0;                // smooth odom-frame estimate (spline eval at now)
  using KeyframeSink = std::function<void(KeyframePacket&&)>;
  virtual void set_keyframe_sink(KeyframeSink) = 0;       // the ONE thing L2 gives L3 (§6)
  virtual FrontEndDiagnostics diagnostics() const = 0;    // §8 (spec 01 canonical struct)
};
std::unique_ptr<IFrontEnd> makeFrontEnd(const FrontendConfig&,
                                        std::shared_ptr<const CalibrationSet>,
                                        TelemetrySink*);  // factory, spec 00 §5.1
```

The production class is **`CtFrontEnd`** (`frontend_kind = 1`). The factory may also
construct the iEKF oracle (§5.6) for offline differential tests; both fill the same
`KeyframePacket`, so the ROS 2 wrapper and L3 are written against `IFrontEnd` only.
`live_state()` returns the spline evaluated at the latest valid time (a `NavState`,
spec 01 §3.2) — for telemetry/control/L4 live integration, **not** the handoff.

### 7.2 Threading & ownership

L2 separates a **fast propagate/publish path** from the **window-solve path** so the
operator-facing pose never inherits solver latency or jitter:

* **Window-solve path.** `ingest(PreprocessedGroup)` runs the §5.1 outer loop on the
  front-end thread (Meridian's priority stage, spec 00 §11): extend the spline,
  associate, solve under the §5.2 deadline, marginalize, maybe emit a keyframe. On
  completing a solve it publishes the new window result — the latest knots and the
  bias/gravity estimate — into a **double-buffered last-solved snapshot** by an
  atomic pointer swap (writer never blocks; readers always see one consistent,
  complete window).
* **Propagate/publish path.** `ingest_imu_live(ImuSample)` and `live_state()` do
  **not** wait on or trigger a solve. `live_state()` produces the control-facing
  `NavState` by evaluating the trajectory **off the last-solved snapshot**: it reads
  the snapshot's spline up to its newest knot and, for time beyond that knot,
  **re-propagates** by extending the spline's analytic motion under the snapshot's
  most recent IMU samples (constant body rate / specific force over the short
  extrapolation, the same closed-form evaluation used at cold start, §2.3). The
  published live pose therefore advances smoothly at IMU rate and is decoupled from
  the irregular cadence and step jitter of the window solve. The next solve replaces
  the snapshot wholesale; there is no incremental write-back from the fast path into
  the solver state.
* **Why a snapshot, not a lock.** A reader taking `live_state()` mid-solve must never
  see a half-updated window. The double buffer makes the read a single atomic load of
  a consistent snapshot; the solver writes the *other* buffer and swaps. This is the
  same lossless-to-the-solver, lossy-to-control discipline the rest of the system
  uses for live state.
* `set_keyframe_sink`'s callback fires on the front-end thread but **enqueues** the
  `KeyframePacket` (Moved) onto a lossless thread-safe queue to the back-end thread
  (spec 00 §11; spec 01 §6.3). After the move L2 no longer owns it.
* `cloud_body`/`image` are Shared-immutable; their lifetime is reference-counted
  across {graph node, nvblox map store, loop detector} (spec 01 §6.3).

The split is invisible at the `IFrontEnd` boundary: `live_state()` and the
`KeyframePacket` stream are unchanged in type and meaning; only the internal timing
of the live pose is decoupled from the solve.

### 7.3 Back-end feedback

L3 corrections arrive via the wrapper calling `set_calibration` (refined extrinsics
as a versioned `CalibrationSet` snapshot, spec 01 §5.3) and a `GraphUpdate` path
(spec 00 §11.2, applied at safe points). On a `GraphUpdate`, L2 re-anchors the
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
  (the FAST-LIO `/cloud_registered{,_body}` analogue).
* `cloud("frontend/effective")` + `scalar("frontend/n_effective")` +
  `scalar("frontend/res_mean")` — the inlier set that constrained the solve, plus
  count and mean residual (degeneracy made plottable).
* `cloud("frontend/visual_points")` — projected visual map points;
  `scalar("frontend/visual/map_points")` — active visual-map point count, and
  `scalar("frontend/visual/n_candidates")` — per-frame visual candidates evaluated
  (the visual funnel: candidates in vs map points retained).
* `pose("odom/body")` + `vec("frontend/cov_diag")` — live odometry + typed 6×6 cov
  (not smuggled into a pose message's covariance field as FAST-LIO does).
* `vec("frontend/observability", obs[6])` — the 6 scores driving L3 noise; rendered
  as the **observability hexagon** marker (spec 00 §10.4), degenerate axes red.
* `pose("calib/T_imu_lidar")`, `pose("calib/T_imu_cam")` — online extrinsic
  estimates over time; `scalar("frontend/exposure")` — inverse exposure $\tau$.
* `vec("frontend/bias_acc"|"bias_gyr")`, `scalar("frontend/grav_norm")` — biases /
  $|g|$ from the active knots.
* `marker("spline/control_poses")` — the active control points + knot times, so the
  adaptive-knot density (§5.5) is visible.
* `timing(stage,ms)` per stage (extend/associate/solve/marginalize/map-update) →
  `StageTiming` (FAST-LIVO2 prints an analogous per-stage timing table; Appendix R.3).
* `scalar("frontend/deadline_hit")` + `scalar("frontend/solve_ms")` +
  `scalar("frontend/outer_iters")` — the deadline-bounded-solve signal (§5.2; the
  ratio of solves cut off at `solver.time_limit_ms`), the wall-clock solve time, and
  the §5.1.1 re-association pass count, so a sustained deadline-hit rate (the
  over-budget warning) and association churn are plottable. (`frontend/deadline_hit`
  and `frontend/solve_ms` are spec 09 §5.1's canonical keys; `frontend/outer_iters`
  is a new L2 signal spec 09 should register alongside `frontend/iter_count`.)
* `scalar("frontend/gnss/innovation_m")` + `scalar("frontend/gnss/accept_rate")` —
  per-fix innovation (§3.4) and the rolling accept rate (spec 09 §5.1's canonical
  `frontend/gnss/*` keys); reject reasons are logged.
* `event(WARN,"frontend/window_restart",reason)` — restart recovery made visible
  (spec 00 §10.2).
* `marker("keyframe_links")` — emitted keyframes + links coloured
  green=RelativeBetween, yellow=ImuPreintegration, red=AbsolutePrior/new-segment.

### 8.2 Structured logging

Per-outer-step one-line structured `INFO` (id, eff points, visual patches,
outer/inner iters, mean residual, solve ms, deadline-hit, min eigenvalue, n_cp,
KF?); `WARN` on no-effective-points / restart / extrinsic-or-exposure freeze /
sustained deadline-hit / GNSS re-acquisition; `ERROR` on IMU saturation / total
tracking loss. Heavy per-point dumps at `TRACE`. Via `meridian::log` (spec 00 §10.3); the core
never calls `RCLCPP_*`.

---

## 9. Parameters

| Param | Default | Source / appendix |
|---|---|---|
| `frontend.kind` | `ct_livo` | the front-end; `iekf_oracle` is offline-only (§5.6) |
| `init_time_s` | 0.1 | static-init window; **~1.0 s recommended** (the FAST-LIO `INIT_TIME=0.1 s` default is too short — a longer static start materially tightens gravity/bias init, §2.3); App. R.4 |
| `imu.cov_gyr` $\sigma_g^2$ | 0.1 | **squared** continuous-time noise density (variance, $(\text{rad/s})^2$); `calibration_from_config` takes $\sqrt{\cdot}$ on load. FAST-LIO `avia.yaml`; App. R.4 |
| `imu.cov_acc` $\sigma_a^2$ | 0.1 | **squared** continuous-time noise density (variance, $(\text{m/s}^2)^2$); sqrt-on-load. FAST-LIO `avia.yaml`; App. R.4 |
| `imu.b_gyr_cov` $\sigma_{bg}^2$ | 1e-4 | **squared** gyro-bias random-walk density (variance); sqrt-on-load. FAST-LIO `avia.yaml`; App. R.4 |
| `imu.b_acc_cov` $\sigma_{ba}^2$ | 1e-4 | **squared** accel-bias random-walk density (variance); sqrt-on-load. FAST-LIO `avia.yaml`; App. R.4 |
| `ct.spline_order` | 4 (cubic, $C^2$) | App. R.1 |
| `ct.representation` | split SO(3)×ℝ³ | App. R.1 (consensus best) |
| `ct.knot_dt_s` | 0.1 (outer cadence) | Coco-LIC; App. R.1 |
| `ct.n_cp_max` | 1 | adaptive knot cap; **shipped gated at 1** (uniform spline) pending non-uniform-grid validation (§5.5); App. R.1 |
| `ct.window_seconds` | 0.3–1.0 | fixed-lag window; App. R.1 |
| `ct.time_offset_estimate` | false | per-sensor $t_d$ hook; App. R.1 |
| `knot_omega_thresh[]` | tune (rad/s) | adaptive-knot $N_\omega$ band edges (§5.5) |
| `knot_accel_thresh[]` | tune (m/s²) | adaptive-knot $N_a$ band edges (§5.5) |
| `knot_density_hysteresis` | 0.15 (frac) | band-down hysteresis (§5.5) |
| `motion_regularizer_en` | true | gated jerk / ang-accel regularizer (§3.3) |
| `motion_reg_weight` | 1e-3 (rel.) | regularizer weight vs accel weight (§3.3) |
| `motion_reg_excitation_floor` | tune | engage-below excitation level (§3.3) |
| `lidar.num_match_points` | 5 | FAST-LIO `NUM_MATCH_POINTS` |
| `lidar.max_match_dist_sq` | 5.0 m² | FAST-LIO `h_share_model` |
| `lidar.plane_thresh` | 0.1 m | FAST-LIO `esti_plane` |
| `lidar.point_cov` $\sigma_0^2$ | 1e-3 | FAST-LIO `LASER_POINT_COV` |
| `lidar.max_lidar_factors` | 1500 | bounded factor count (§3.1) |
| `lidar.min_factors_per_normal` | 50 | per-stratum floor (§3.1) |
| `lidar.normal_strata` | 7 (6 axes + oblique) | stratified subsample bins (§3.1) |
| `map.voxel_m` | 0.5 | FAST-LIO `filter_size_map`; App. R.2 |
| `solver.max_iterations` | 5 | CLINS 4–5; App. R.1 |
| `solver.min_iterations` | 2 | deadline lower bracket (§5.2) |
| `solver.epsi` | 1e-3 | FAST-LIO `epsi[23]`; App. R.5 |
| `solver.time_limit_ms` | 90 ms | deadline-bounded solve, wall-clock path; 90 ms validated at 10 Hz on dev box (§5.2) |
| `bias.gyr_max` | 0.5 rad/s | gyro bias box bound on every bias-timeline knot (§1.4, §5.2) |
| `bias.acc_max` | 5.0 m/s² | accel bias box bound on every bias-timeline knot (§1.4, §5.2) |
| `bias.knot_dt_ms` | 500 | sliding bias-timeline knot cadence (§1.4) |
| `frontend.tail_anchor.stride` | `window_dt/4` | tail VelocityAnchor/RateAnchor placement past $t_{\text{end}}$ (§1.3.1) |
| `frontend.tail_anchor.sigma_vel` | 0.2 m/s | tail velocity-anchor std (§1.3.1) |
| `frontend.tail_anchor.sigma_rate` | 0.1 rad/s | tail rate-anchor std (§1.3.1) |
| `max_outer_iters` | 4 (3–5) | re-association passes (§5.1.1) |
| `reassoc_steps` | 2 | inner LM steps per pass (§5.1.1) |
| `assoc_shift_thresh` | 0.02 m / 0.2° | re-association early-stop (§5.1.1) |
| `visual.patch` | 8 | FAST-LIVO2 `patch_size`; App. R.3 |
| `visual.levels` | 3 | FAST-LIVO2 `patch_pyrimid_level`; App. R.3 |
| `visual.img_point_cov` | 100 | FAST-LIVO2 `IMG_POINT_COV`; App. R.3 |
| `visual.outlier_threshold` | 1000 | FAST-LIVO2; App. R.3 |
| `visual.ncc_thre` | tune | NCC gate (§3.2 gate 4); App. R.3 |
| `visual.exposure_estimate_en` | true (gated) | FAST-LIVO2; App. R.3 |
| `visual.inv_expo_cov` $\sigma_\tau^2$ | 1e-2 /s | exposure random-walk (§1.4) |
| `visual.inv_expo_min` | 1e-3 | $\tau>0$ clamp (§1.4) |
| `visual.grid_cell_px` | 32 px | one-best-per-cell grid (§3.2) |
| `visual.warp_det_min`/`_max` | 0.1 / 10.0 | degenerate-warp guard (§3.2) |
| `visual.warp_cond_max` | 50 | degenerate-warp guard (§3.2) |
| `visual.ref_obs_cap` | 30 | ref-patch observation cap (§3.2.1) |
| `visual.ref_add_angle_deg` | 10° | ref-patch add gate (§3.2.1) |
| `visual.ref_score_w` | 0.7 | NCC vs view-cos re-score weight (§3.2.1) |
| `visual.ref_converged_obs` | 8 | converged-latch obs count (§3.2.1) |
| `visual.ref_converged_angle_deg` | 30° | converged-latch view span (§3.2.1) |
| `visual.active_box_m` | 60 m | visual-point spatial eviction box half-extent (§3.2.1) |
| `visual.rolling_shutter_en` | false | per-row spline eval, deferred (§3.2.2) |
| `visual.row_time_s` | datasheet | rolling-shutter per-row readout time (§3.2.2) |
| `kf.trans_m` | 0.5–1.0 | keyframe trigger |
| `kf.rot_deg` | 10–15 | keyframe trigger |
| `kf.time_s` | 1–2 | heartbeat trigger |
| `obs.degeneracy_thresh` $\kappa_{\text{deg}}$ | tune | eigenvalue floor (§4.2) |
| `extrinsic.refine` | true (gated) | FAST-LIO `extrinsic_est_en`; App. R.5 |
| `gnss.enable` | env-dependent | off in GNSS-denied (§3.4) |
| `gnss.floor_fixed`/`_float`/`_dgps`/`_spp` | see §3.4 table | per-fix-type covariance floors (§3.4) |
| `gnss.innovation_k` | 3.0 | Mahalanobis innovation gate (§3.4) |
| `gnss.reacquire_count` | 5 | consecutive in-gate fixes to re-admit (§3.4) |

(Covariance/weight values $\Sigma_L,\Sigma_I,\Sigma_C$ are not given numerically in
the CT papers — tune empirically; Appendix R.1.)

---

## 10. Failure modes & mitigations

| Failure | Detection | Response |
|---|---|---|
| **Cold start, no trajectory** | spline spans < first valid knots | IMU-only deskew/seed (§2.3); build window from first data; no keyframe until window converged (spec 00 §7.2). |
| **No effective LiDAR points** | effective correspondence count < 1 | mark step degraded; lean on IMU + visual residuals (the spline stays constrained by §3.3); bump counter; if persistent → window restart. |
| **Knot spacing too coarse** for the motion | systematic IMU/LiDAR residual the solver cannot null | adaptive `n_cp` raises control-point density (§5.5). |
| **Control points under-constrained** between sparse measurements | ill-conditioning / jitter; weak §4 axis at low `n_cp` | dense IMU residuals regularise; cap `n_cp`; marginalization prior anchors the boundary; engage the optional low-weight jerk / angular-acceleration regularizer (§3.3) on the under-excited span. |
| **Geometric degeneracy** (corridor, tunnel, open field) | small eigenvalue(s) of $\Lambda_{\text{pose}}<\kappa_{\text{deg}}$ (§4) | inflate `constraint_cov` on weak axes; optionally clamp the update along weak dirs (solution remapping); lean on IMU/GNSS/visual; **freeze extrinsic/gravity refinement** (§3.6). |
| **Photometric failure** (low light, blur, over/under-exposure) | high photo residual / NCC fail / depth-continuity fail / `outlier_threshold` hit | drop visual residuals this step (LIO only); freeze exposure $\tau$; WARN. Degrade gracefully. |
| **IMU saturation / dropout** | accel/gyro at FSR or gap > `imu_gap_max` | mark; if integration untrustworthy emit `AbsolutePrior` wide-cov new segment; ERROR. |
| **Sweep gap (bridged)** | sweep opens > ½ span past `last_solved_t_`, but group IMU reaches `last_solved_t_` within 2 IMU periods | the seed integrates across the hole; **solve normally**; log `frontend/sweep_gap_bridged` (§2.2). |
| **Sweep gap (unbridgeable)** | the group IMU itself starts after `last_solved_t_` (a true data hole) | **`reseedAfterGap`**: predict the anchor across the hole on **constant velocity**, **rebuild** the window + bias timeline, insert the reseed sweep into the map **without solving**; the next continuous-IMU sweep solves against the rebuilt window (§2.2). |
| **Unknown clock offset $t_d$** | systematic, motion-correlated residual | nominally prevented by PTP (spec 02); enable `ct.time_offset_estimate` hook (§2.2). |
| **Marginalization linearization staleness** | prior inconsistent after a large correction | re-linearize prior on `GraphUpdate`; bound window so the prior is never far from the current estimate. |
| **Discrete-knot coverage by time-sampling** (the clamp-snapshot bug class) | a knot set covered by sampling the curve at times instead of enumerating deque indices skips or double-counts knots near a boundary / density change → partial coverage poisons the solve | **enumerate knot indices, never sample by time** (§5.4 invariant); the original snapshot path is gone with the clamp, but marginalization (§5.4) and tail-knot pinning (§1.3.1) must obey it. |
| **GNSS spoof/jump/multipath** | innovation $k\cdot\sigma$ gate fail (§3.4); `fix` type; jump vs odom | reject the fix (counted); fix-type covariance floor caps trust; re-admit only after `gnss.reacquire_count` consecutive in-gate fixes; robust kernel; one bad fix never snaps the window. Authoritative robustness (PCM) in L3. |
| **Bias runaway** (outlier burst drives a bias knot to an absurd value) | a bias-timeline knot pins at its box bound | **bias box bounds** (§5.2), applied to every $b_g$/$b_a$ knot of the sliding timeline (§1.4), absorb it in-window — no restart; if a bias stays pinned across the step it escalates to the divergence row below. (There is no step clamp; the tail null space it once guarded is removed structurally by tail-knot pinning, §1.3.1.) |
| **Divergence** (residual blow-up, all-axis obs collapse) | residual norm > thresh **with a bias knot pinned at a bound** (§5.2); min eigenvalue ≈ 0 across axes | **window restart** (last resort, only after the §5.2 bias-bounds guard fails): discard the window, re-bootstrap from IMU-only deskew, emit `ImuPreintegration` (or wide `AbsolutePrior`) so L3 stitches the gap; prior keyframes preserved (§6.4). |
| **Visual-map / ref-patch unbounded growth** (long mission) | active visual-point count or observation-list size rising without bound | observation cap + min-score eviction + converged latch on ref patches; spatial box-delete of out-of-box visual points mirroring the LiDAR map (§3.2.1); active set is RAII-owned, store is the durable record. |
| **Degenerate / NaN photometric warp** (pure rotation, grazing, uninitialised normal) | $|\det A|$ / condition out of bounds, or no fitted normal (§3.2) | drop the point for that frame (never warp an unwarped or guessed patch); reconsider next frame once a normal exists. |
| **Stale cloud handle after loop closure** | store eviction | consumers re-fetch from `IKeyframeStore` by `id`, never cache raw pointers (§6.5). |

---

## 11. Library choices and what was rejected

One choice per job; alternatives are named here only to record that they were
considered and rejected (library canon, system direction).

| Job | Chosen | One-line justification | Rejected (why) |
|---|---|---|---|
| CT spline kernel | **basalt-headers (vendored)** | canonical, battle-tested $O(k)$ analytic derivatives **and** Jacobians w.r.t. control points — removes the single biggest implementation risk (App. R.1). | hand-rolled spline (months of subtle Jacobian bugs); full-SE(3) (non-split) spline (screw coupling, slower; App. R.1). |
| Window solver / marginalization | **Ceres** (LM + Schur) | what all three CT references use; manifold SO(3), built-in Schur (App. R.1). | GTSAM here (reserved for the L3 iSAM2 global graph). |
| NN for plane fitting | **ikd-Tree (vendored) or adaptive voxel-hash** | incremental, box-delete, lock-safe rebuild for the recent-point query (App. R.2). | brute-force kNN. |
| Map (TSDF + colour + mesh) | **nvblox (GPU)** | GPU TSDF + colour + Marching Cubes on the Orin, the single map backend (spec 06). **No CPU fallback, no VDBFusion, no OpenVDB/NanoVDB.** | CPU TSDF / VDBFusion (Orin always has CUDA — a second path is dead weight). |
| Robust kernels / IMU factor (L3 side) | **GTSAM** `noiseModel::Robust`+Huber, `CombinedImuFactor`, `GncOptimizer` | the restart factor and global graph (spec 05). | per-interval preintegration on the normal path (the relative cov already carries it; §6.4). |
| Linear algebra / Lie | **Eigen 3.4 + Sophus** | basalt/Ceres substrate. | — |

**One paragraph on the iEKF.** A FAST-LIO2-style sequential ESIKF is the well-known
fast alternative to a CT window. Meridian keeps it **only** as an offline
reference/oracle behind `IFrontEnd` (§5.6): it is faster and battle-tested, but it
handles intra-scan motion by approximate backward-propagation (not joint
optimisation), fuses an asynchronous camera awkwardly, and cannot natively
represent the sub-scan trajectory the colourised-mesh goal benefits from (Appendix
R.1). The CT estimator is the design; the iEKF is the test partner.

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
  `CalibrationSet` / `GraphUpdate` (§7.3). **GNSS-binding interaction:** L2 binds the
  GNSS residual to the spline pose *interpolated to the fix time* (§3.4); the
  `nearest_kf_id` of L3's `add_absolute` is a graph-attachment hint only, and L3 must
  likewise interpolate to fix time rather than treat the nearest keyframe pose as the
  measurement location.
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

---

## Appendix R — SOTA reference grounding (non-normative)

This appendix is evidence, not contract: curated digests of the reference systems
this spec's design was validated against. Nothing here binds Meridian's behavior —
the normative sections above own the design. Each block names the reference checkout
it was verified against; the clones live in `/home/user/slam-reference`.

### R.1 Continuous-time B-spline trajectory (CLINS / Coco-LIC / basalt)

*Verified against `Coco-LIC@4ead7e4`, `clins@6ceb011` (both vendor basalt-style
spline headers). basalt-headers itself is **not present** in
`/home/user/slam-reference`; its `RdSpline`/`So3Spline`/`Se3Spline` API is
described from the live Coco-LIC/clins copies.*

Split cubic ($k=4$, $C^2$) cumulative B-spline: a `So3Spline<4>` for rotation + an
`RdSpline<3,4>` for translation. The cumulative blending matrix
$\tilde M^{(4)}=\tfrac16[[6,5,1,0],[0,3,3,0],[0,-3,3,0],[0,1,-2,1]]$ ($\lambda_0\equiv1$).
Value/derivatives are the $O(k)$ Sommer recurrence; non-uniform (adaptive `n_cp`) is
realised via virtual-time remapping over the uniform kernels (NURBS variants).

| Concept | Coco-LIC `file:symbol` |
|---|---|
| cumulative blending matrix `M̃` | `src/spline/spline_common.h::computeBlendingMatrix<N,_,Cumulative=true>` (line 73) |
| SO(3) value / gyro `ω` / accel | `src/spline/so3_spline.h::evaluate` (237), `velocityBody` (311), `accelerationBody` (370) |
| non-uniform (virtual-time) variants | `so3_spline.h::evaluateNURBS` (283), `velocityBodyNURBS` (340) |
| IMU residual (gyro+accel on spline) | `src/odom/factor/analytic_diff/trajectory_value_factor.h::IMUFactor` (128) |
| bias random-walk residual | same file, `BiasFactor` (63); gravity `GravityFactor` (31) |
| LiDAR point-to-plane on spline | `src/odom/factor/analytic_diff/lidar_feature_factor.h::LoamFeatureFactor` (30); NURBS form `LoamFeatureFactorNURBS` (175) — residual `pᵀ·n + d` at line 93 |
| visual frame-to-map reprojection (LiDAR-depth) | `lidar_feature_factor.h` / `image_feature_factor.h` |
| Schur marginalization prior | `src/odom/factor/analytic_diff/marginalization_factor.h::MarginalizationInfo` (106), `MarginalizationFactor` (161) |
| adaptive knot insertion | `src/odom/trajectory_manager.cpp` (`non_uniform` extend path) |

**CT vs iEKF contrast (the conclusion driving the §1 choice and §5.6/§11 oracle).**

| Dimension | CT B-spline (Coco-LIC/CLINS) | iEKF (FAST-LIO2 / Point-LIO) |
|---|---|---|
| intra-scan motion | absorbed natively — deskew *is* the trajectory | backward-prop deskew (FAST-LIO2) / per-point update (Point-LIO) |
| sensor asynchrony | native — each sensor at its own `t` | per-sensor handling; async camera awkward |
| IMU handling | direct residual on spline 2nd-deriv + `ω` | propagation (FAST-LIO2) / augmented-state measurement (Point-LIO) |
| sub-scan trajectory for colour | queryable at any rate | not represented |
| compute | heavier NLLS window (Ceres) | lighter filter |
| maturity | newer | very mature |

**Sharp edges.** (i) A CT spline absorbs *known* timing only — an unknown
inter-sensor offset $t_d$ is silently baked in unless added as an optimised
parameter (`T(t+t_d)`, differentiable). (ii) Coco-LIC's published $(N_g,N_a)\to n_{cp}$ adaptive
thresholds are graphical in the paper — tune against the released code. (Meridian's
*normative* gating statistic is the peak per-sample $\,N_\omega,N_a$ form of §5.5,
not Coco-LIC's summed-then-normed aggregate, which cancels under fast rotation.)
(iii) Covariance weights $\Sigma_L,\Sigma_I,\Sigma_C$ are not numerically given in
the CT papers — tune empirically. (iv) Across a knot-density change the real-time
trajectory is only $C^1$ (the virtual→real slope jumps); a constant-rate motion is
exactly representable only within a uniform-density run.

### R.2 ikd-Tree incremental map internals (the registration-oracle contract)

*Verified against `FAST_LIO@7cc4175`. The `ikd-Tree` itself is a git submodule
(`hku-mars/ikd-Tree`, pinned `e2e3f4e`) that is **not checked out** in this clone —
`include/ikd-Tree/` is empty. The contract below is the digest the spec's NN query
relies on; re-confirm body line numbers against an initialized submodule before
quoting them.*

The front-end's plane-fit NN query (`ikdtree.Nearest_Search`, 5-NN) sits on an
incremental k-d tree with these load-bearing properties:

| Property | Contract |
|---|---|
| node bookkeeping | per-node subtree AABB (`node_range_{x,y,z}`), `TreeSize`, `invalid_point_num`, `down_del_num`, recorded `alpha_bal`/`alpha_del` |
| kNN search | bounded max-heap, box-distance pruning on the per-node AABB (not single-axis), skips lazy-deleted nodes/subtrees; result sorted nearest-first |
| lazy delete | a deleted point/subtree is flagged (`point_deleted`/`tree_deleted`), not removed; re-insertion flips the flag back |
| box-wise delete | `Delete_Point_Boxes(BoxPointType[])` — subtree fully inside box → O(1) `tree_deleted`; disjoint → prune; partial → recurse. Used by the sliding-map eviction |
| scapegoat re-balance | rebuild subtree at $T$ when $\alpha_{bal}=\max(\#L,\#R)/(\text{TreeSize}-1)>0.7$ **or** $\alpha_{del}=\text{invalid}/\text{TreeSize}>0.5$; never if `TreeSize < 10` |
| parallel rebuild | subtree > 1500 pts rebuilds on a worker thread; concurrent ops are logged to a ring buffer and **replayed** onto the new balanced subtree before swap; `size()`/`validnum()` use `trylock` + cached fallback so search/odometry never stall |
| on-insert downsample | `Add_Points(.., downsample=true)` keeps one point per voxel (nearest the voxel centre), lazy-deletes the rest |

Constants (from the dossier's header read; **unverified against the live empty
submodule**): `Minimal_Unbalanced_Tree_Size=10`, `Multi_Thread_Rebuild_Point_Num=1500`,
`delete_criterion_param=0.5`, `balance_criterion_param=0.7`, `downsample_size` set to
`filter_size_map_min` (0.5 m). Meridian may instead use an adaptive voxel-hash with
the same box-delete / lock-safe-rebuild contract.

### R.3 Sparse-direct photometric residual (FAST-LIVO2)

*Verified against `FAST-LIVO2@0d2c034`. Exact line numbers within `updateState`
drifted from the dossier's earlier read; the anchor functions below were
re-confirmed in this checkout.* Note: FAST-LIVO2 keeps a **custom 19-DoF
`StatesGroup`** (`include/common_lib.h:30`, manual `operator+/-`), **not** the
FAST-LIO IKFoM `state_ikfom`; the LiDAR and visual updates share that one covariance
sequentially (LiDAR, then photometric, no re-propagation).

| Element | `src/vio.cpp` symbol |
|---|---|
| per-frame pipeline | `processFrame` (1786): retrieve → `computeJacobianAndUpdateEKF` (784) → generate/update visual points → refine ref patch |
| association (depth from LiDAR scan, in-FOV gather, raycast, depth-continuity gate, warp+NCC gate) | `retrieveFromVisualSparseMap` (352) |
| photometric residual $r=\tau_{cur}I_{cur}(\mathbf u+\Delta)-\tau_{ref}P^{warp}[\Delta]$ | inside `updateState` (1520) |
| projection Jacobian $J_\pi$ (2×3) | `computeProjectionJacobian` (189) |
| affine warp via LiDAR-plane homography | `getWarpMatrixAffineHomography` (252); pyramid pick `getBestSearchLevel` (320) |
| NCC gate | `calculateNCC` (333) |
| sequential ESIKF update $K=(H^\top R^{-1}H+P^{-1})^{-1}\dots$ | `updateState` (1520) |

Map points double as LiDAR points in a shared **0.5 m voxel hash** (`feat_map`,
`vio.h:126`); depth always from LiDAR → no VIO init/triangulation. Parameters:
`patch_size=8`, `patch_pyrimid_level=3`, `IMG_POINT_COV=100`, `outlier_threshold=1000`,
`ncc_thre`, `exposure_estimate_en` (`vio.h:100,109`). Gates: depth-continuity Δ>0.5 m,
NCC, oblique-view (>60°). Sharp edge to improve on: heavy raw-pointer ownership
(`new`/`delete` of `VisualPoint`/`Feature`) — Meridian uses RAII/smart pointers.

### R.4 IMU model, init, propagation, backward-deskew (FAST-LIO)

*Verified against `FAST_LIO@7cc4175`.*

| Quantity | Equation | `file:symbol` |
|---|---|---|
| de-biased gyro | $\omega = \omega_m - b_g$ | `use-ikfom.hpp::get_f` (47) |
| world accel | $a = R(a_m-b_a)+g$ | `use-ikfom.hpp::get_f` (47) |
| state ODE | $\dot p=v,\ \dot R=R[\omega]_\times,\ \dot v=R(a_m-b_a)+g$ | `get_f` |
| mean propagation | $x_{k+1}=x_k\boxplus(f\,\Delta t)$ | `esekfom.hpp::predict` |
| cov propagation | $P=F_xPF_x^\top+(\Delta t F_w)Q(\Delta t F_w)^\top$ | `esekfom.hpp::predict` |
| init gravity (on $S^2$) | $g_0=-(\overline{a_m}/\|\overline{a_m}\|)\cdot 9.81$ | `IMU_Processing.hpp::IMU_init` |
| init gyro bias | $b_{g0}=\overline{\omega_m}$ | `IMU_init` |
| per-point rotation | $R_i=R_h\,\mathrm{Exp}(\omega\,\Delta t)$ | `IMU_Processing.hpp::UndistortPcl` |
| per-point world disp. | $T_{ei}=p_h+v_h\Delta t+\tfrac12 a_h\Delta t^2-p_e$ | `UndistortPcl` |
| deskew transform | $\hat P=R_{LI}^\top[R_e^\top(R_i(R_{LI}P_i+t_{LI})+T_{ei})-t_{LI}]$ | `UndistortPcl` |

Defaults (`config/avia.yaml`): $\sigma_g^2=\sigma_a^2=0.1$, $\sigma_{bg}^2=\sigma_{ba}^2=10^{-4}$,
`INIT_TIME=0.1 s`. Deskew is a **backward pass over the prior (propagated) trajectory**
with constant-$\omega$/constant-accel intra-interval (the code's `// not accurate!`
ceiling) — the discrete approximation Meridian's CT spline replaces.

**Correction to a prior false dossier claim.** An earlier dossier asserted (i) that
this `esekfom.hpp` copy *omits* the `f_w_final` manifold projection, and (ii) that
FAST-LIVO2 reuses the *identical* deskew on the *same* IKFoM 23-tangent state. Both
are **false** against the live clones: `f_w_final` is fully assigned in FAST-LIO's
`esekfom.hpp::predict` (the SO(3)/$S^2$ blocks at lines 300/328/369, used in the
covariance update at 381); and FAST-LIVO2 carries a custom 19-DoF `StatesGroup`
(R.3), not IKFoM, and its deskew uses precomputed `extR_Ri`/`exrR_extT` extrinsic
composites (`IMU_Processing.cpp:526`), not the IKFoM `offset_R_L_I`. The two systems
share the *idea* of backward-propagation deskew, not the *state* or the exact code.

### R.5 State manifold + iterated ESIKF (FAST-LIO IKFoM) — for the §5.6 oracle

*Verified against `FAST_LIO@7cc4175`.* Grounds the optional offline iEKF oracle's
sign conventions and the $\Lambda_{pose}=H^\top R^{-1}H$ block the CT solver
cross-checks against.

State `state_ikfom` (DOF=23 tangent, DIM=24 ambient): $(p,R,R_{LI},t_{LI},v,b_g,b_a,g)$,
$g\in S^2$ (fixed $|g|$, 2-DoF). `use-ikfom.hpp:12` (build), `:8` ($S^2$ type), `get_f` (47),
`df_dx` 24×23 (61), `df_dw` 24×12 (80).

| Element | Equation / `file:symbol` |
|---|---|
| right-perturbation chart | $R\boxplus\delta=R\,\mathrm{Exp}(\delta)$, Euclidean for vects, $S^2$ retraction for $g$ |
| point-to-plane row $H_j$ (1×23, 12 nonzero) | $[\,n^\top,\ -n^\top R[q]_\times,\ -n^\top R R_{LI}[p^L]_\times,\ n^\top R R_{LI},\ 0\dots]$ — `laserMapping.cpp::h_share_model` (638) |
| residual | $z_j=n^\top p^W_j+d$ stored as $-z_j$ |
| information-form gain | $K=(H^\top R^{-1}H+P^{-1})^{-1}H^\top R^{-1}$ — inverts 23×23 not $m\times m$ |
| iterated update + prior pull-back | $\delta=Kz+(KH-I)\,J\,(x\boxminus x_{prop})$; `update_iterated_dyn_share_modified` |
| chart transport $J$ | SO(3): $A(\delta\phi)^\top$; $S^2$: $N_xM_x$ |
| LiDAR meas. noise | scalar `LASER_POINT_COV=0.001`; convergence per-DoF `epsi=0.001`, `maximum_iter=4` |

Plane fit: 5-point QR of $A\mathbf u=-\mathbf 1$ then unit-normalise (`common_lib.h::esti_plane`,
226), 0.1 m planarity gate, range-scaled acceptance $s=1-0.9|r|/\sqrt{\|p^L\|}$ (keep
$s>0.9$). Robustness is **gating, not per-point weighting**. `extrinsic_est_en=true`
default fills cols 6–11 of $H$. The Meridian oracle uses a decoupled
`[p|R|v|b_g|b_a|g]` chart (so position columns stay $n^\top$), a first-order
tangent-plane $S^2$ scheme, and identity chart-transport — deliberate
simplifications acceptable for an offline cross-check.
