# Meridian — System Overview: How the Whole Thing Works, and Why

> Prepared 2026-06-01. This is the **narrative** document for project Meridian: the
> end-to-end story of how points and photons become a colourised mesh, and the
> reasoning behind every major structural choice. It is written to be read
> *once, top to bottom*, by someone new to the project who already knows SLAM at
> a graduate level.
>
> **Where this sits in the doc set.**
> - Design rationale and the decision register live in `reference/NEXT_GEN_DESIGN_archived.md` (the abstractions-and-tradeoffs companion).
> - The 2026 survey that justifies the SOTA framing lives in `reference/SOTA.md`.
> - The validation datasets are fixed in `./DATASET.md` (FusionPortable primary, M2DGR co-primary) — referenced, not repeated here.
> - The per-module **specs** (interface contracts, message definitions) live under `./specs/`, and the textbook-depth derivations live under `./course/`. This document points to them; it does not duplicate them.
> - Per-module grounding notes (code:line and paper-eq citations from the reference systems) live under `./grounding/` (dossiers 01–10).
>
> Every technical claim about the reference systems below is grounded in the code
> on disk and cited as `file:line`, verified against the live checkout in
> `C:/Users/charl/Sources/slam-reference/`. Meridian is a **clean rebuild** — it
> reuses no prior code and supersedes the arc-slam (LIO-SAM derivative) line
> entirely — but it learns from those reference implementations.

---

## 0. Reading map

This document has nine parts. If you read nothing else, read §1 (the problem),
§3 (the one-paragraph mental model), and §5 (the data-flow walk).

1. The problem Meridian solves, and the scope boundary.
2. The sensor suite and what each modality buys you.
3. The central idea — one continuous-time, tightly-coupled estimator feeding a global graph — in one mental model.
4. The layered architecture L0–L6, layer by layer.
5. The data flow, traced end to end from a single LiDAR firing to a mesh triangle.
6. Robustness and degeneracy, treated as one cross-cutting system, not a checklist.
7. Time synchronisation and calibration — the two threads that touch every layer.
8. Build order — module integration sequence for one full system (not a feature rollout).
9. Glossary and cross-references.

---

## 1. The problem

Meridian is a **from-scratch, continuous-time, tightly-coupled
LiDAR-Inertial-Visual-GNSS 3D SLAM system for tactical operational use**, built
to run on an **NVIDIA Jetson Orin** (a CUDA GPU is always present). Strip away
the adjectives and the job is: a vehicle or a person carries a sensor rig through
an unknown, possibly hostile environment, and Meridian must answer two questions
continuously and in real time:

1. **Where am I?** — a trajectory that is locally accurate to sub-decimetre over
   room/vehicle scale (< 0.5 % drift over 100 m) and globally consistent to the
   metre over kilometres (< 1 m ATE over 1 km after loop closure).
2. **What does the world look like?** — a **colourised triangle mesh** of the
   environment, streamed to an operator with sub-200-ms latency from measurement
   to mapped surface.

The operating envelope is the hard part. "Tactical" means the system must keep
working when things go wrong: the LiDAR fogs over or is occluded, the camera is
blinded by darkness or glare, GNSS is denied or actively **spoofed**, the scene
is geometrically **degenerate** (a tunnel, a snowfield, a flat corridor where
the geometry cannot tell you how far you have moved along the axis), the platform
moves fast (0–30 m/s with sharp yaw), and the sensor clocks drift relative to one
another. A system that is accurate only when all sensors are healthy is a demo,
not a tool. Meridian's entire architecture is an argument for *graceful degradation*:
every degree of freedom of the estimate should be observable by **more than one**
modality, so that losing any single sensor narrows the system's confidence
without collapsing it.

### 1.1 The scope boundary (read this before you over-build)

Meridian **stops at the colourised mesh.** Two large capabilities are
**explicitly deferred**:

- **Path planning** (ESDF / traversability). Deferred.
- **Semantics** (segmentation, object detection, semantic loop closure). Deferred.

This is not an oversight; it is a deliberate boundary. The map substrate is
chosen (a GPU TSDF — see §4.6) *precisely* so that both deferred capabilities can
be bolted onto the same data structure later without a redesign: an ESDF derives
from a TSDF by wavefront propagation, and a semantic channel is one more
per-voxel field. Wherever leaving a hook is cheap, the hook is left — nvblox even
ships a GPU ESDF integrator we simply do not call. But nothing in the current
scope *depends* on either capability, and you should not build toward them now.
When in doubt, build the mesh path and leave the door open.

---

## 2. The sensor suite — what each modality contributes

Meridian's rig is **one LiDAR + one camera + one IMU + GNSS.** The design philosophy
is best understood by asking of each sensor: *which axes of the state does it make
observable, and when does it fail?* The point of the suite is that the failure
modes are uncorrelated, so the *combination* stays well-conditioned even where any
single modality goes blind.

| Modality | What it observes well | How it fails | Rate |
|---|---|---|---|
| **LiDAR** | Yaw and horizontal translation from surrounding structure; roll/pitch from any non-vertical structure; dense geometry for the map | Geometric degeneracy (tunnels, open fields); fog/smoke/dust; occlusion | 10–20 Hz |
| **Camera** | Motion *across* geometric ambiguity via photometric texture — independent of geometry, so it carries axes the LiDAR cannot; per-surface colour for the mesh | Darkness, glare, motion blur, textureless walls | 10–30 Hz |
| **IMU** | High-rate motion between LiDAR firings; gravity direction (pitch/roll anchor); the smooth backbone that ties asynchronous measurements together | Drifts without correction; bias wanders | 200–500 Hz |
| **GNSS** | Global position anchor — kills long-range drift when available | Denial (indoors, tunnels, urban canyon), spoofing | 1–10 Hz |

The argument the rig makes is **redundant observability**. Consider the worst
tactical scene, a long straight tunnel:

- The LiDAR loses **along-axis translation** — every cross-section looks the
  same, so the geometry cannot say how far you have advanced.
- The **camera** sees wall texture sliding past → direct along-axis translation,
  completely independent of LiDAR geometry.
- The **IMU** integrates acceleration → along-axis translation over short
  windows, bounded by the others.
- **GNSS** is denied, contributes nothing — and the system must not collapse
  because of it.

No single sensor saves the tunnel; the *combination* does, and only because the
fusion is tight (§3.1) so the camera's along-axis information enters the *same*
normal equations as the LiDAR's lateral information. The whole design argument, in
one sentence: **there is no single point of failure in the state estimate**
(`reference/NEXT_GEN_DESIGN_archived.md` §4.2). The corollary, which §6 develops,
is that the estimator must *know* which sensor is currently carrying which axis,
so that when one drops out the uncertainty grows only on the axes it was
responsible for.

> **Future extension, not designed now.** A second LiDAR (e.g. an upward-facing
> "dome" for stronger roll/pitch from overhead structure) would slot in behind the
> *same* sensor interface and add residual streams to the *same* trajectory. The
> current system is single-LiDAR; multi-LiDAR is a one-line forward-compatibility
> note, not part of this design.

---

## 3. The central idea, in one mental model

Before the layer-by-layer tour, hold this picture in your head. It is the whole
system compressed to a paragraph.

> **Meridian estimates a single continuous-time trajectory in a short sliding window,
> fusing direct LiDAR, sparse-direct photometry, IMU, and GNSS tightly at each
> measurement's true timestamp (the front-end), and it stitches the frozen tail of
> that window into a long-lived, globally consistent factor graph that absorbs loop
> closures and GNSS (the back-end). The trajectory drives a GPU map: an nvblox
> TSDF for the surface, coloured from the camera, and a marching-cubes mesh for the
> operator. Two cross-cutting concerns — a single hardware-synchronised clock and
> continuously-refined calibration — make the tight fusion possible.**

Three words in that paragraph carry most of the weight: **tightly coupled**,
**continuous-time**, and **two-tier**. The rest of this section unpacks each.

### 3.1 Tightly coupled — and why it is non-negotiable here

A *loosely* coupled system runs each modality as its own estimator and fuses the
**outputs** (e.g. a visual odometry pose enters a LiDAR graph as a single
factor). A *tightly* coupled system fuses the **raw measurements** in one
estimator: a LiDAR point-to-plane residual, an IMU residual, and a camera
photometric residual all constrain the *same* trajectory in the *same*
optimisation.

For Meridian, tight coupling is the only way to get redundant observability to pay
off. In the tunnel example, a loosely coupled system would get a degenerate LiDAR
pose (with the along-axis direction unconstrained or, worse, filled with
garbage), and only *then* try to fix it with vision — but the damage is already
baked into the LiDAR sub-estimate. A tightly coupled system never forms a bad
intermediate pose: the camera's along-axis information enters the *same* normal
equations as the LiDAR's lateral information, and the combined Hessian is
well-conditioned even though neither modality alone is. FAST-LIVO2 states this
structural argument exactly — in a degenerate scene the LiDAR block adds almost no
information to the unobservable axis, but the photometric block does, and **they
sum in one Hessian before inversion** (`grounding/04_livo2_visual.md` §0). Tight
fusion lets the strong axes of one sensor fill the weak axes of another *inside a
single linear system*, which is exactly where per-axis observability (§6) becomes
a first-class concept rather than an afterthought.

### 3.2 Continuous-time — the trajectory is a *function of time*

This is the deepest conceptual shift from the arc-slam (LIO-SAM) lineage, and it
is the design from the start — not a later upgrade. The classical view treats the
trajectory as a *sequence of discrete poses*, one per keyframe, and "deskews" each
LiDAR scan to a single canonical keyframe time. That approximation is fine when
the platform moves slowly and all sensors share a clock; it breaks when they do
not.

The continuous-time view: the trajectory is a smooth function $T(t) \in SE(3)$,
and **every measurement constrains $T$ at its own true timestamp.** A LiDAR beam
fired at time $t_i$ constrains $T(t_i)$; an IMU sample at $t_j$ constrains
$T(t_j)$ and its derivatives; a camera frame at $t_k$ constrains $T(t_k)$. There
is no deskewing step because *the trajectory is the deskew* — you simply query
$T$ at each point's timestamp (`grounding/10_continuous_time.md` §1). Meridian
represents the trajectory as a **split SO(3)×ℝ³ cubic B-spline** (§4.3): a cubic
(order-4) spline is $C^2$ continuous — continuous position, velocity, *and*
acceleration — which is exactly enough to write the IMU acceleration residual
directly against the spline's second derivative, with no pre-integration. The
split (separate rotation and translation splines) is the consensus choice in
Sommer et al., CLINS, and Coco-LIC — faster and as-or-more-accurate than a coupled
full-SE(3) spline (`grounding/10_continuous_time.md` §2.4).

Why this matters concretely: a spinning LiDAR firing one beam every ~2 μs, an IMU
at 200–500 Hz, and a camera shuttering at 30 Hz are *three different clocks with
three different jitters*. A vehicle at 20 m/s moves 2 m within a single 100-ms
LiDAR scan. The discrete-pose approximation assigns all those beams to one pose
and falls off a cliff; the CT trajectory absorbs the within-scan motion exactly
(`reference/SOTA.md` §2.4). And because every sensor evaluates the *same*
$T(\cdot)$ at its own $t$, asynchrony is native: there is no timestamp snapping, no
per-pair synchronisation hack.

> **One caveat, stated loudly** (`grounding/10_continuous_time.md` §1, §9): a CT
> spline absorbs *known* timing — it does **not** correct an *unknown* clock
> offset $t_d$ between sensors. If a sensor's stamp is biased, the spline fits the
> wrong time and bakes the offset in. The fix is to add $t_d$ as an estimated
> parameter and evaluate that sensor's residual at $T(t+t_d)$ — clean in CT
> precisely because $T(\cdot)$ is differentiable in $t$. Meridian exposes a per-sensor
> $t_d$ as an optional optimised parameter, fixed-zero once calibrated (§7).

### 3.3 Two-tier — local continuous-time front-end + global discrete back-end

No single estimator is good at both *fast, high-rate, asynchronous local fusion*
and *globally consistent, loop-closing, multi-hour optimisation*. The 2026 field
converged — independently, in Wildcat (CSIRO) and Coco-LIC — on splitting the job
(`reference/SOTA.md` §2.1):

- **L2, the front-end**, is the **continuous-time sliding-window** estimator
  described above. It holds ~1–3 s of trajectory as B-spline control points,
  fuses LiDAR + visual + IMU + GNSS tightly at scan rate, and emits **keyframes**.
  It is where fast motion and asynchronous timing are handled. It has no global
  memory and does not close loops.
- **L3, the back-end**, is a **discrete keyframe factor graph** solved
  incrementally with **iSAM2** (GTSAM). It is the system's global memory: it
  holds keyframe poses over the whole mission, absorbs loop closures and GNSS,
  refines extrinsics, and keeps the map globally consistent. It runs at keyframe
  rate (≈ 1 Hz).

The handoff between them is a **single relative-pose factor** carrying the
front-end's marginal covariance (§4.5). The back-end never sees raw points; the
front-end never thinks about loops. Each tier does the one thing it is good at.

The front-end sits behind a clean **`IFrontEnd` interface** for modularity — the
back-end, map, and operator layers consume keyframes-with-covariance and never
care about the front-end's internals. (Behind that interface, a FAST-LIO2-style
iterated-ESKF can serve as an optional reference baseline and test oracle — it is
the same point-to-plane and IMU physics in a filter form, useful for cross-checking
the CT solver. It is *not* a milestone, not a "v1", and the system is not organised
around it; the full CT LIVO+GNSS front-end is the design.)

---

## 4. The layered architecture, L0–L6

The system is six layers, each consuming the previous and exposing a stable
interface, plus two cross-cutting concerns (§7). The full diagram is in
`reference/NEXT_GEN_DESIGN_archived.md` §3; here is the compressed form with the
*why* of each layer.

```
 L6  Operator interface ......... colour mesh + per-voxel/keyframe confidence overlay
 L5  Place recognition .......... Scan Context++ -> STD/BTC -> GICP (small_gicp) -> PCM (loop closure only)
 L4  Map ........................ nvblox GPU: TSDF + colour -> Marching Cubes mesh (GPU, only backend)
 L3  Back-end ................... GTSAM iSAM2 keyframe factor graph (global consistency, loops, GNSS, extrinsics)
 L2  Front-end .................. CT B-spline sliding window, tightly coupled LiDAR+visual+IMU+GNSS  [IFrontEnd]
 L1  Per-sensor preprocessing ... filter, photometric calib, GNSS gating, health
 L0  Sensor abstraction ......... PTP time sync, calibrated streams, (ts, measurement, covariance)
        ^ cross-cutting: TIME SYNC (PTP from GNSS PPS)  +  CALIBRATION (offline prior + online refinement)
```

A governing principle runs through all of it: a **ROS-agnostic core C++ library**
(C++20) holds the estimation, mapping, and recognition logic, and **thin ROS 2
(Humble) wrapper nodes** adapt it to topics and parameters. The core has no ROS
dependency so it can be unit-tested, replayed offline against the datasets in
`./DATASET.md`, and reused outside ROS. Layers are interfaces with a **single**
committed implementation each (`IFrontEnd` → the CT estimator; `IMapLayer` →
nvblox) — the seam exists for testability and clean ownership, not to host a
parade of alternatives.

### 4.1 L0 — Sensor abstraction and time

Every sensor becomes a uniform stream of
`(timestamp_monotonic, calibrated_measurement, covariance)`. The single most
important job of L0 is the **clock**: every timestamp must live on one monotonic
clock to sub-microsecond precision, because the CT front-end can only resolve
sub-scan motion if each point's timestamp is trustworthy. The mechanism is **PTP
distributed from a GNSS PPS pulse** (§7.1). The Ouster LiDAR supports PTP
natively; the camera is triggered by GPIO; the host clock is disciplined with
`ptp4l`/`phc2sys`. L0 also runs a **sensor health channel** — a graded,
out-of-band confidence stream per sensor (data rate nominal? points-per-scan
nominal? intensity histogram sane? no saturated camera frame? IMU bias residual
bounded?) — that downstream layers consult to decide how much to trust each
measurement. This is the modern, *graded* replacement for "is the cable plugged
in."

### 4.2 L1 — Per-sensor preprocessing

Each modality is conditioned in parallel before fusion:

- **LiDAR**: range/intensity clipping and self-hit removal (the rig structure).
  Note there is **no separate deskew step** — in the CT front-end, deskew *is* the
  trajectory: each point is transformed by $T$ evaluated at its own timestamp
  (§4.3, `grounding/10_continuous_time.md` §4.2). (For contrast, the reference
  FAST-LIO does motion compensation by forward-propagating the IMU and then
  back-propagating each point to the scan-end frame using its own timestamp,
  `P_{compensate} = R_{e}^\top (R_i P_i + T_{ei})`, at
  `slam-reference/FAST_LIO/src/IMU_Processing.hpp:311-340` — the discrete analogue
  of what the spline does for free.)
- **Camera**: **photometric calibration** (gamma, vignetting, an auto-exposure
  model). Without it, direct photometric residuals are unstable — FAST-LIVO2's
  whole approach depends on an exposure-compensation term that is optionally a
  state variable (`grounding/04_livo2_visual.md` §3). Plus geometric undistortion
  / rectification and a reused image pyramid (`createImgPyramid`,
  `slam-reference/FAST-LIVO2/src/frame.cpp:54-63`).
- **IMU**: stationary **bias initialisation** at startup and gravity alignment
  from the stationary accel mean — in the reference,
  `init_state.grav = S2(-mean_acc/‖mean_acc‖ · g)`
  (`slam-reference/FAST_LIO/src/IMU_Processing.hpp:196`, mean from `:185`) — plus
  an Allan-variance noise model feeding the residual weights, and saturation
  flagging.
- **GNSS**: quality gating (satellites, HDOP, fix type), **spoofing detection**
  (compare GNSS-derived velocity to IMU-derived velocity over a 1 s window; large
  disagreement → drop and flag), and LLA → ECEF → ENU conversion. ENU centred at
  the first valid fix is the working map frame.

### 4.3 L2 — The front-end (the heart)

**State.** The trajectory is a **split SO(3)×ℝ³ cubic (order-4, $C^2$) cumulative
B-spline**: a set of rotation control points $\{R_k\}\subset SO(3)$ and
translation control points $\{p_k\}\subset\mathbb{R}^3$ at the window's knots,
plus slowly-varying IMU biases $b_g, b_a$ (and gravity, where estimated). The
trajectory is
$$T(t) = \big(R(t),\,p(t)\big), \qquad R(t)=R_i\textstyle\prod_{j=1}^{3}\mathrm{Exp}\!\big(\lambda_j(u)\,d_j\big),\ \ d_j=\mathrm{Log}(R_{i+j-1}^{-1}R_{i+j}),$$
with the translation spline in the analogous Euclidean cumulative form
(`grounding/10_continuous_time.md` §2.2). Because the spline is cubic, **velocity
and acceleration are closed-form** functions of the control points — there is no
separate velocity state, and the IMU sees the spline's derivatives directly. The
window uses **adaptive (Coco-LIC-style) knot placement**: a fixed ~0.1 s outer
cadence, with the number of control points per segment chosen from the
IMU-measured motion dynamics $(N_g, N_a)$ — dense during aggressive motion, sparse
when smooth (`grounding/10_continuous_time.md` §5).

**Measurement factors.** Inside the window, four residual families constrain the
trajectory. The two that define Meridian's character are direct LiDAR registration
and sparse-direct photometry.

*Direct LiDAR point-to-plane.* For each (voxel-subsampled) point measured at its
own timestamp $t$, evaluate $T(t)$, transform the point to world (folding in the
LiDAR→IMU extrinsic), find the local plane in the map, and penalise the signed
distance:
$$r_L = \mathbf{n}^\top\big(R_{LG}(t)\,\mathbf{p}^L + p_{LG}(t)\big) + d.$$
This is the CT form (`grounding/10_continuous_time.md` §4.2); its Jacobian flows
through $R(t), p(t)$ to the four overlapping control points active at $t$, and the
*overlap* of these control-point sets across a scan is what stitches the
trajectory together. The plane $(\mathbf{n}, d)$ comes from a PCA fit to the $k$
nearest map points — no edge/plane feature extraction, no LOAM smoothness sort.
The reference FAST-LIO computes exactly this residual and plane fit (in its filter
form): `esti_plane(pabcd, points_near, 0.1f)`
(`slam-reference/FAST_LIO/src/laserMapping.cpp:678`) and
`pd2 = pabcd(0)*x + pabcd(1)*y + pabcd(2)*z + pabcd(3)`
(`:680`), with a robustifying weight `s = 1 - 0.9*|pd2|/sqrt(‖p‖)` (`:681`) that
down-weights points far from their plane. The world transform with the extrinsic
is `s.rot*(s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos` (`:657`), and the
measurement-Jacobian rows (`h_x`, `:720`) carry the normal in the position block,
$A$ in the rotation block, and — when online extrinsic estimation is on
(`extrinsic_est_en`, default true, `:73`) — an extrinsic block $B$ (`:740-743`).
This is the *direct* paradigm: every point that lands on a plane is a constraint.
A modern LiDAR produces millions of points per second; throwing away 99 % of them
to pick a few hundred edge features (the LOAM way) is indefensible. Direct
registration uses them all, subsampled only by the voxel grid, and it stays robust
in feature-poor scenes — point-to-plane keeps registering as long as there are
*any* surfaces (`reference/NEXT_GEN_DESIGN_archived.md` §6.6).

*Sparse-direct photometric (FAST-LIVO2 style).* Pick image patches around points
whose depth the LiDAR already knows — vision **never triangulates depth**, it
borrows it from the LiDAR map, which removes the entire VIO
initialisation/triangulation problem (`grounding/04_livo2_visual.md` §0, §7).
Project a 3D map point into the current image at $T(t_{\text{img}})$ and penalise
the **exposure-compensated photometric difference** between the warped reference
patch and the current image sample:
$$r_C = \tau_{\text{cur}}\,I_{\text{cur}}\big(\pi(R_{cw}\mathbf{p}^w + p_{cw})\big) - \tau_{\text{ref}}\,P^{\text{warp}}[k],$$
where $\tau$ is the inverse exposure time (the affine exposure compensation that
makes direct methods survive auto-exposure, optionally a state variable). In the
reference this is `res = ref_patch[i] - cur_value`
(`slam-reference/FAST-LIVO2/src/vio.cpp:1619-1621`), with the projection Jacobian
`computeProjectionJacobian` (`:189-201`) and the measurement row assembled as
*image-gradient × projection-Jacobian × pose-Jacobian* (`:1611-1617`). The
photometric residual is independent of geometry, so it carries exactly the axes
LiDAR loses in degeneracy. No descriptors, no feature matching — which is why it
stays robust where ORB-style feature VO falls apart in low texture
(`reference/SOTA.md` §2.6). FAST-LIVO2's **unified voxel hash** indexes
both LiDAR planes and visual points (`grounding/04_livo2_visual.md` §1.4); Meridian
adopts the same one-index design.

The other two families are **IMU** (per sample, 200–500 Hz: a *direct* residual on
the spline's angular velocity and acceleration plus a bias random-walk —
$r_\omega = \omega(t)+b_g-\omega_m$, $r_a = R(t)^\top(a(t)-g)+b_a-a_m$,
`grounding/10_continuous_time.md` §4.1) and **GNSS position** (per surviving fix,
a switchable constraint). A weak **extrinsic prior** holds each sensor→IMU
transform near its calibrated value while letting residuals refine it.

**Solver and window.** A fixed-lag sliding-window non-linear least squares over
the control points + biases runs at scan rate (the consensus tooling is **Ceres**
with analytic spline Jacobians from vendored **basalt-headers**,
`grounding/10_continuous_time.md` §11). As the window slides, states leaving it are
**marginalised by Schur complement** into a Gaussian prior on the boundary control
points + current biases (Coco-LIC-style, `grounding/10_continuous_time.md` §6) —
this both keeps the window's cost bounded and produces the keyframe handoff to L3
(§4.5). At each solve, the marginal Hessian on the emitted keyframe is
eigendecomposed to extract **six per-axis observability scores** (X-ICP / D²-LIO
style), which travel with the keyframe so the back-end knows which DOFs are weak
(§6). This is the principled replacement for arc-slam's binary `isDegenerate` flag.

**Bias and kinematics live here.** IMU bias estimation is the front-end's job; the
back-end consumes a clean relative-pose factor and treats velocity/bias as seeds
and telemetry only (§4.5). This keeps the back-end's variable set small and its
relinearisation cheap.

### 4.4 The L2 → L3 handoff (one relative-pose factor)

This deserves its own subsection because it is where the two tiers meet. The
front-end window covers ~1–3 s. When the oldest slice of the window falls off the
back edge, its information is **marginalised** (Schur complement against the
window's information matrix) into the prior described above. From that, the
front-end emits, at roughly one keyframe per second or per metre travelled, a
**single relative `BetweenFactor<Pose3>`** between consecutive keyframes, carrying
the **front-end marginal covariance** as its noise model. On the normal path,
velocity and bias are *not* re-estimated by the back-end — they ride along as
seeds/telemetry (`kinematics_included = false`). The covariance is *anisotropic
and honest*: a corridor keyframe arrives with a huge variance on the along-axis
direction, and iSAM2 will weight it down on that axis automatically — no
information lost, no binary switch.

> **One mutually-exclusive fallback.** If the front-end has to restart its window
> (e.g. it lost track and re-initialised), the relative factor is not trustworthy
> across the seam; for that one step the back-end instead consumes a GTSAM
> `CombinedImuFactor` over the buffered IMU to bridge the gap. The two paths are
> mutually exclusive — `BetweenFactor` on the normal path, `CombinedImuFactor` only
> on a window restart — and that is the only place back-end kinematics enter.

### 4.5 L3 — The back-end (global memory)

The back-end is a GTSAM **iSAM2** factor graph over keyframe variables: poses
$X_k$, per-sensor extrinsics $E_s$, and the GNSS ENU origin $G$ (with a hook for
future multi-robot peer poses). Velocities and biases are **not** standing
variables on the normal path — they are seeds carried from L2 — which keeps the
graph small. iSAM2 remains the right backbone in 2026 because its Bayes-tree,
variable-local relinearisation lets a 10 000-keyframe graph absorb a fresh loop
closure in tens of milliseconds — no batch optimiser keeps up incrementally, and
no pure sliding window gives kilometre-scale global consistency
(`grounding/09_backend_isam2.md` §1, `reference/SOTA.md` §3.1).

The factors are: the **relative keyframe factor** from L2 (the dominant
constraint, with its anisotropic covariance); **GNSS factors** (switchable);
**loop-closure factors** from L5 (robust kernel + switchable); **extrinsic
priors** (weak, so the graph can refine $E_s$ online); and **zero-velocity
factors** during detected stationary periods. The back-end is where the
robustness stack (§6) lives, and where **online extrinsic refinement** happens —
extrinsics drift in the field (thermal, vibration, a bumped sensor), so carrying
each $E_s$ as a variable with a weak prior lets SLAM residuals continuously
re-calibrate the rig. (Online extrinsic refinement is on by default; the reference
already supports it — the LiDAR→IMU extrinsic appears in the state and in the
measurement Jacobian when `extrinsic_est_en` is set,
`slam-reference/FAST_LIO/src/laserMapping.cpp:73,740-743`.) For multi-hour
missions, old keyframes (> 30 min *and* > 1 km behind, outside any active loop
window) are marginalised to fixed priors so the active graph stays bounded while
remaining visible for map assembly.

The estimation frame is `imu_link`; operator-facing poses are re-expressed to
`base_link` at the edge (the ROS wrapper), not inside the core.

### 4.6 L4 — The map (nvblox, GPU, the only backend)

The map is **nvblox** (`isaac_ros_nvblox`), running entirely on the GPU. There is
no CPU fallback and no second map implementation — the Jetson Orin always has a
CUDA GPU, so the map commits to it. nvblox does, in one library and all on-device:

1. **TSDF + colour.** A truncated signed-distance field integrated from each
   keyframe's LiDAR cloud (transformed by its current best pose), via nvblox's
   **projective** integrator (iterate frustum voxels, look each up in the depth
   image — embarrassingly parallel, GPU-ideal,
   `grounding/07_mapping_tsdf_mesh.md` §2.1). Each voxel carries a running
   weighted average distance and weight,
   $D_{\text{new}} = (W D + w\,d)/(W+w)$,
   $W_{\text{new}} = \min(W+w, W_{\max})$ (Curless–Levoy,
   `grounding/07_mapping_tsdf_mesh.md` §1.3). A separate **colour** layer fuses
   per-voxel RGB projected from the camera at integration time, blended only in
   the thin band near the zero-crossing so grazing-angle colour never corrupts
   geometry (`§1.5`). 8³ voxel blocks in CUDA unified memory.
2. **Mesh — Marching Cubes (GPU).** nvblox extracts a per-vertex-coloured triangle
   mesh at the TSDF zero-crossing, incrementally re-meshing only blocks whose TSDF
   changed (`grounding/07_mapping_tsdf_mesh.md` §5.1). This mesh **is** the map
   deliverable — there is no separate surface tier.

**The load-bearing constraint: clear-and-re-integrate on loop closure.** A
running-average TSDF is **not per-keyframe reversible** — the individual
contributions are summed away, so you cannot "subtract" a stale observation when a
loop closure shifts historical poses (`grounding/07_mapping_tsdf_mesh.md` §8.1).
Meridian therefore keeps a **first-class, persistent, body-frame per-keyframe cloud
store** keyed to the iSAM2 pose nodes. When L3 moves keyframe poses after a loop
closure, Meridian identifies the affected keyframes, **clears the touched region,
re-integrates their clouds at the corrected poses, and re-meshes** the affected
blocks — all on GPU (`grounding/07_mapping_tsdf_mesh.md` §8.2, §8.4). This is fast
for a voxel TSDF (the structure was designed for clear-and-re-integrate) and
intractable for 3D Gaussian Splatting — one concrete reason Meridian uses nvblox and
not a neural/Gaussian map. The keyframe cloud store is the authoritative geometry
record; the TSDF and mesh are a *pure function* of `{(cloud_k, pose_k)}`.

> **Optional archival export, not a core path.** For a final watertight archival
> mesh, a one-shot offline **Screened-Poisson** pass over the retained keyframe
> clouds (with TSDF-gradient normals) may be run after the trajectory is final
> (`grounding/07_mapping_tsdf_mesh.md` §5.3). The live, streamed mesh is always
> nvblox Marching Cubes; Poisson is an export utility, not part of the running
> system.

### 4.7 L5 — Place recognition (loop closure only)

Geometric-proximity loop detection (the arc-slam `radiusSearch(15 m)` pattern)
fails the moment drift exceeds the search radius — and after kilometres of
open-loop odometry, drift routinely does. Meridian replaces it with **hierarchical
place recognition** (`grounding/08_loop_closure_placerec.md` §0):

1. **Scan Context++** — a polar descriptor per keyframe (20 rings × 60 sectors);
   a Kd-tree over the rotation-invariant ring key returns top-k candidates in
   sub-millisecond CPU time, with a column-shift giving an initial yaw.
2. **STD / BTC** (optional) — triangle descriptors over stable keypoints re-rank
   and add a 6-DoF pose guess for reverse/viewpoint-changed loops.
3. **GICP** — fine alignment of the candidate's local submap via **small_gicp**
   (header-only, multithreaded); accept on fitness/inlier ratio.
4. **PCM** — Pairwise Consistency Maximisation (max-clique on the consistency
   graph) rejects mutually-inconsistent loops before any factor reaches L3.

The resulting loop factor is a `BetweenFactor<Pose3>` whose noise scales with GICP
fitness, wrapped in Huber + GNC + switchable constraints (§6) before it reaches the
graph (`grounding/08_loop_closure_placerec.md` §0). Object detection and semantic
loop closure are deferred (§1.1); a semantic-histogram proposer would compose
cleanly here later.

### 4.8 L6 — Operator interface

The consumer contract: an **incremental colour-mesh stream** (sub-second
latency, straight from nvblox), a **queryable trajectory** ($T(t)$ at any
timestamp, useful for downstream sensor fusion — the spline answers any-rate
queries natively), and — the tactical differentiator — a **confidence overlay**.
Per-voxel TSDF weight and per-keyframe iSAM2 marginal covariance are surfaced so
the operator can *see where the map is uncertain*. This is the visible face of the
rich-introspection principle that runs through the whole system: every estimator
stage publishes debug topics, rviz markers, and per-stage timing (FAST-LIVO2's
per-stage timing table, `slam-reference/FAST-LIVO2/src/vio.cpp:1851-1868`, is the
pattern — surfaced here as ROS 2 diagnostics) so a developer or operator can watch
residuals, observability, effective point counts, and stage latencies in real
time.

---

## 5. Data flow, end to end — from one LiDAR firing to one mesh triangle

Here is the life of a measurement, traced through every layer. Follow it once and
the architecture clicks.

```
 PHOTON / RANGE                                                        MESH
      |                                                                  ^
      v                                                                  |
 [L0] LiDAR fires a beam at t_i (PTP clock).             [L4] nvblox Marching Cubes
      Host stamps it on the shared monotonic clock.           extracts a colour
      |                                                        triangle at the
      v                                                        TSDF zero-crossing (GPU).
 [L1] Range/self-hit filter (NO deskew step).                ^
      |                                                       |
      v                                                  [L4] nvblox TSDF integrates
 [L2] Subsample by voxel grid; evaluate T(t_i) on the        the cloud (+ camera RGB)
      spline; transform to world; find the local plane;       at the keyframe's
      form r_L = n^T(...)+d ALONGSIDE the IMU residuals       current best pose (GPU).
      on the spline's derivatives, the camera photometric     ^
      residuals, and any GNSS fix in this window.             |
      Solve the tightly-coupled CT window.               [L3] iSAM2 refines all
      |                                                        keyframe poses;
      v                                                        loop closure / GNSS
 [L2] Window slides; marginalise the tail (Schur);            correct global drift.
      emit a relative BetweenFactor with marginal      -->  [L3] Add the keyframe +
      covariance + per-axis observability.                     its relative factor.
                                                               (L5 may add a loop
                                                                factor here.)
```

In words:

1. **A beam is fired (L0).** The LiDAR timestamps it on the PTP clock; the host
   keeps it on the single monotonic clock. Without this, everything downstream
   misassigns the point to the wrong instant of the trajectory.
2. **The point is conditioned (L1).** Filtered for range and self-hits. There is
   no deskew pass — deskew happens implicitly in L2.
3. **The point becomes a constraint (L2).** It is subsampled by the voxel grid,
   the trajectory is evaluated at *its own* timestamp $T(t_i)$, it is matched to a
   local plane, and turned into a point-to-plane residual that enters the *same*
   window optimisation as the IMU residuals (on the spline's derivatives), the
   camera photometric residuals, and any GNSS fix in this interval. The window is
   solved tightly — the trajectory that comes out explains all modalities jointly.
   This is the moment "tightly coupled" and "continuous-time" stop being slogans
   and become a single linear system over control points.
4. **The window discharges a keyframe (L2 → L3).** The trailing slice is
   marginalised and a relative `BetweenFactor` (with honest anisotropic covariance
   and per-axis observability) is handed to the back-end. The keyframe's
   body-frame cloud is appended to the persistent cloud store.
5. **The back-end keeps the world consistent (L3).** iSAM2 incorporates the new
   keyframe; if L5 recognises a previously visited place, a verified loop factor
   (PCM + GNC + switchable) snaps the global trajectory back into alignment; GNSS
   anchors absolute position when trustworthy.
6. **The map is (re)built (L4).** Each keyframe's cloud is integrated into the
   nvblox TSDF at the keyframe's current best pose, coloured from the camera;
   Marching Cubes extracts the surface — all on GPU. When L3 moves poses, the
   affected region is cleared, re-integrated from the cloud store, and re-meshed.
7. **The operator sees it (L6).** The colour mesh streams out with a confidence
   overlay; the trajectory is queryable at any timestamp.

The latency budget — measurement to mapped voxel in < 200 ms — is met because L2
runs at scan rate, the L2→L3 handoff is one small marginalisation and one factor,
and L4's TSDF integration and meshing are GPU-accelerated (nvblox reports up to
177× surface-reconstruction speed-up over CPU SOTA,
`grounding/07_mapping_tsdf_mesh.md` §11).

---

## 6. Robustness and degeneracy as one system

Robustness in Meridian is not a list of `if` statements; it is a single coherent
strategy that threads from L1 to L3. The unifying idea: **uncertainty is tracked
per-axis and per-sensor, and it flows downstream so that every consumer knows
exactly how much to trust each constraint.** This replaces the 2016-era binary
"degenerate / not-degenerate" decision throughout.

### 6.1 Per-axis observability (the spine)

At the front-end, the marginal Hessian on each emitted keyframe is
eigendecomposed into six per-axis observability scores (X-ICP / D²-LIO style,
`reference/SOTA.md` §3.3). A corridor keyframe arrives with a weak
along-axis score; a featureless-ceiling keyframe arrives with a weak yaw score.
These scores set the *anisotropic* covariance of the relative keyframe factor, so
the back-end naturally down-weights the weak axes — **no information is projected
away** (unlike Zhang 2016), and the constraint still contributes on its strong
axes. Because the fusion is tight (§3.1), the scores reflect the *joint*
observability of all sensors: an axis that LiDAR cannot see but the camera can
will read as observable, and the system will not flag a false degeneracy.

### 6.2 Multi-modal redundancy (graceful degradation)

This is where the sensor suite (§2) and per-axis observability meet. Each sensor
contributes to the joint Hessian on the axes it observes. When a sensor degrades —
LiDAR in fog, camera in darkness, GNSS denied — its contribution to those axes
simply drops out, the joint Hessian's conditioning on those axes worsens, the
observability scores fall, and the covariance grows *only on the affected axes*.
The system does not crash; it widens its confidence exactly where it has lost
information and keeps running on the modalities that remain. The sensor health
channel (L0/§4.1) feeds this by flagging degraded sensors before their residuals
poison the estimate.

### 6.3 GNSS adversaries

GNSS is the one modality with an active adversary. Two defences: a **spoofing
check** at L1 (GNSS velocity vs IMU velocity over 1 s; drop on disagreement,
§4.2) and a **switchable constraint** at L3 so that a fix surviving the gate can
still be turned off by the optimiser if it disagrees with everything else. An
always-on Gaussian GNSS factor would let a single spoofed fix or an RTK dropout
collapse the graph; the switch prevents that.

### 6.4 Back-end robustness stack (three simultaneous layers)

The back-end is the last line of defence against bad factors, and it runs three
mechanisms at once (`grounding/09_backend_isam2.md` §9–§11,
`reference/NEXT_GEN_DESIGN_archived.md` §7.4):

1. **PCM** (Pairwise Consistency Maximisation) — a *pre-optimisation* filter that
   keeps only the maximally mutually-consistent subset of a loop-closure batch
   (max-clique on the consistency graph).
2. **GNC** (Graduated Non-Convexity, GTSAM's `GncOptimizer`) — a *per-factor*
   robust kernel that tightens from quadratic toward truncated-least-squares over
   a few iterations; provably finds the global optimum under bounded outlier
   ratios.
3. **Switchable constraints** — a *latent inlier/outlier* per suspect factor, so
   persistently bad loops or GNSS fixes are driven off.

The combined cost is low single-digit milliseconds per iSAM2 update — cheap
insurance against the adversarial scenes the system is built for.

---

## 7. The two cross-cutting threads

Two concerns are not layers but threads that touch every layer; the whole tight
fusion stands or falls on them.

### 7.1 Time synchronisation

The CT front-end *cannot converge* without tight clock alignment, because
sub-scan motion can only be resolved if each point's timestamp is trustworthy to
the microsecond, and because every sensor evaluates the *same* $T(\cdot)$ at its
own $t$. The mechanism is a **PTP grandmaster disciplined by the GNSS PPS pulse**:
the GNSS receiver's PPS + UTC message drives a PTP grandmaster on the local
network (`linuxptp`: `ptp4l`/`phc2sys`); the Ouster LiDAR slaves to PTP natively,
the camera is GPIO-triggered, the IMU is PPS-stamped (or host-stamped with bounded
jitter), and the host clock is disciplined accordingly. End result:
sub-microsecond alignment across the rig. The investment (a few hundred euros of
PTP hardware) pays back many times over in front-end robustness
(`reference/NEXT_GEN_DESIGN_archived.md` §11). Any *residual* sync error after
PTP is absorbed by the per-sensor time-offset $t_d$ the front-end can estimate
(§3.2) — that is the one supported mechanism, not a parallel "software-sync"
fallback path.

### 7.2 Calibration

Three stages, each feeding the next:

1. **Factory / one-time intrinsics** — camera intrinsics + distortion, LiDAR
   per-beam offsets, IMU Allan variance.
2. **Per-deployment extrinsics** — rig poses between all sensors (Kalibr for
   camera–IMU, target-based for LiDAR–camera, hand-eye motion calibration for
   LiDAR–IMU). These become **priors**, not hard truths.
3. **Online refinement** — every extrinsic $E_s$ is a *variable* in the L3 graph
   with the prior above; SLAM residuals refine it continuously, on by default
   (`reference/NEXT_GEN_DESIGN_archived.md` §12).

The reference code already supports online extrinsic estimation: the LiDAR→IMU
extrinsic appears in the state (`offset_R_L_I`/`offset_T_L_I`) and in the
measurement Jacobian when `extrinsic_est_en` is on
(`slam-reference/FAST_LIO/src/laserMapping.cpp:73,657,740-743`). This three-stage
pattern handles the real tactical failure mode — the rig gets bumped, optics get
dirty, a sensor is swapped — without a recalibration session.

---

## 8. Build order — one system, in integration order

Meridian is **one complete system**, not a sequence of shippable versions. There is
no "v1 we ship first" and no feature rollout: the full CT LIVO+GNSS front-end,
the iSAM2 back-end, and the nvblox map are the design. What follows is purely a
**module compile/integration order** — the order in which pieces come online so
that each can be tested against working scaffolding — not a product roadmap.

| Bring-up step | What is integrated | Why this order |
|---|---|---|
| **Contracts & time base** | L0 sensor abstraction, the core/ROS-2 split, the `IFrontEnd` and `IMapLayer` interfaces, message/debug-topic contracts, the PTP clock | Nothing fuses correctly until every measurement is on one clock and the interfaces compile. |
| **CT front-end** | The split SO(3)×ℝ³ B-spline window with direct LiDAR + IMU residuals (basalt-headers + Ceres), then sparse-direct photometry, then GNSS — all into the one trajectory | The heart of the system; the optional iEKF reference oracle behind `IFrontEnd` is wired here only to cross-check the CT solver. |
| **Back-end & loop closure** | L3 iSAM2 keyframe graph consuming the relative factor; L5 place recognition (Scan Context++ → STD → small_gicp → PCM); GNSS and loop factors with the robustness stack | Global consistency consumes the front-end's keyframes; build it once those keyframes exist. |
| **Map & operator** | L4 nvblox TSDF + colour → Marching Cubes on GPU, the persistent keyframe cloud store + clear-and-re-integrate, L6 colour-mesh stream + confidence overlay | The deliverable; it consumes the corrected poses from L3 and the clouds from L2. |

The through-line: this is a single tightly-coupled estimator brought up
module-by-module, with the `IFrontEnd`/`IMapLayer` seams existing for testability
and clean ownership — not to stage a simpler system first.

---

## 9. Glossary and cross-references

**Tightly coupled** — fusing raw measurements from all modalities in one
optimisation, not fusing per-sensor pose outputs. §3.1.

**Continuous-time (CT) trajectory** — $T(t)\in SE(3)$ as a split SO(3)×ℝ³ cubic
B-spline; measurements constrain $T$ at their true timestamps; deskew is the
trajectory. §3.2, §4.3. Ground: `grounding/10_continuous_time.md`.

**Two-tier estimation** — CT sliding-window front-end (L2) + discrete iSAM2
back-end (L3), bridged by one marginalised relative factor. §3.3, §4.4.

**`IFrontEnd`** — the interface the back-end/map consume; its committed
implementation is the CT estimator, with an optional iEKF reference oracle behind
it for cross-checking. §3.3, §4.3.

**Per-axis observability** — six localizability scores per keyframe (X-ICP /
D²-LIO) that set anisotropic covariance and flow into back-end noise; the
principled replacement for a binary degeneracy flag. §6.1.

**Direct registration** — point-to-plane residuals against the map with no
feature extraction; uses all points. §4.3, ground:
`slam-reference/FAST_LIO/src/laserMapping.cpp:657,678-681,720,740-743`.

**Sparse-direct photometric** — exposure-compensated photometric residuals on
patches around LiDAR-depth points, no descriptors, no triangulation. §4.3, ground:
`slam-reference/FAST-LIVO2/src/vio.cpp:189-201,1611-1621`.

**nvblox map** — GPU TSDF + colour → GPU Marching Cubes mesh; the only map
backend; re-integrable from the keyframe cloud store on loop closure. §4.6, ground:
`grounding/07_mapping_tsdf_mesh.md`.

**Robustness stack** — PCM (pre-filter) + GNC (per-factor kernel) + switchable
constraints (latent on/off), all at once in L3. §6.4.

**Deferred** — path planning (ESDF) and semantics; designed to slot onto the same
nvblox TSDF substrate later, not built now. §1.1.

**Resolved defaults.** ROS 2 Humble; C++20; Jetson Orin (CUDA always present);
estimation frame `imu_link`, operator poses re-expressed to `base_link` at the
edge; single relative `BetweenFactor` with front-end marginal covariance on the
normal path; `CombinedImuFactor` only on a window-restart fallback (mutually
exclusive); bias estimation in L2; online extrinsic refinement on by default;
scope stops at the colourised mesh.

**Companion documents.** Design rationale: `reference/NEXT_GEN_DESIGN_archived.md`.
SOTA survey: `reference/SOTA.md`. Datasets: `./DATASET.md`. Per-module
specs and textbook-depth derivations: `./specs/` and `./course/`. Grounding
notes: `./grounding/` (dossiers 01–10).

*End of system overview.*
