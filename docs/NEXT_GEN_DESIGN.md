# Next‑Generation arc‑slam — System Design (abstractions and tech specs)

> Prepared 2026‑05‑28. Companion to `SOTA.md` (the survey) and `SLAM_REFRESHER.md` (the code today).
> Scope: this is design discussion. No implementation order, no file paths, no roadmap. Decisions and tradeoffs only.

---

## 1. Vision

A **multi‑sensor 3D SLAM system optimized for tactical operational use**: real‑time mapping with sub‑decimetre local accuracy and metre‑level global consistency over kilometres; outputs a colour 3D **mesh reconstruction** of the environment for operator situational awareness. Robust to sensor degradation, geometric degeneracy, GPS denial or spoofing, fast motion, and asynchronous sensor timing.

The system is single‑platform but should not assume single‑platform (multi‑robot collaborative mapping must be an extension, not a rewrite).

**Scope boundary.** This design stops at the geometric, colourised mesh. Path planning (ESDF, traversability) and semantic reconstruction (segmentation, object detection, semantic loop closure) are **deferred** — they are natural extensions on top of this map, but they are explicitly out of scope here and are not designed for. Wherever a hook for them is cheap to leave in place, it is noted; nothing in the current spec depends on them.

## 2. Requirements analysis

### 2.1 Functional

| Requirement | Quantitative target | Why it matters |
|---|---|---|
| Local odometry accuracy | < 0.5 % drift over 100 m | Operator believes the map matches reality at room/vehicle scale |
| Global trajectory accuracy after loop closure | < 1 m ATE over 1 km | Cross‑mission map alignment, repeat visits |
| Map update latency | < 200 ms from sensor measurement to mapped voxel | Real‑time operator map |
| Sensor rate | LiDAR 10–20 Hz, IMU 200–500 Hz, camera 10–30 Hz, GNSS 1–10 Hz | Standard sensors |
| Map output | Colourised TSDF → triangle mesh, streamed | Operator situational awareness |
| Operating range | Indoor‑outdoor, kilometre scale, multi‑hour sessions | Tactical mission profile |

### 2.2 Non‑functional / robustness

| Failure mode | Behaviour required |
|---|---|
| Loss of one LiDAR (mechanical, smoke, occlusion) | Continue with remaining LiDARs + IMU + camera |
| Loss of camera (darkness, glare, lens cover) | Continue with LiDAR + IMU + GNSS |
| GNSS denial | Continue with LIVO; no global drift correction |
| GNSS spoofing | Detect (velocity disagreement) and ignore |
| Geometric degeneracy (tunnel, snow, water) | Continue with constrained DOFs from IMU + camera; widened uncertainty on degenerate axes |
| Fast motion (vehicle 0–30 m/s, rapid yaw rates) | No accuracy collapse; trajectory smoothness preserved |
| Sensor desynchronization (clock drift, network jitter) | Measurements fused at their true timestamps, not at the closest keyframe |
| Online recalibration of extrinsics | Yes; extrinsics are variables in the back‑end with priors |

### 2.3 Operating constraints

- Compute: 1 modest GPU (Jetson AGX Orin / consumer RTX class) + multi‑core CPU. Not a server farm.
- Power: vehicle or backpack budget (50–200 W).
- ROS 2 (the 2026 default for tactical robotics deployments).
- C++17 / C++20.

### 2.4 What we are explicitly not chasing

- Photorealistic rendering. Tactical use wants flat‑shaded coloured mesh, not 3DGS.
- Sub‑millimetre geometric accuracy. Survey‑grade is a different system.
- Pure visual operation. We always have LiDAR + IMU.

## 3. Top‑level architecture

Six layers, each consuming the previous and exposing a stable interface:

```
+---------------------------------------------------------------+
| L6  Tactical interface                                         |
|     - Colour mesh streaming, confidence overlay               |
+---------------------------------------------------------------+
| L5  Place recognition                                          |
|     - Loop closure detection + verification only              |
+---------------------------------------------------------------+
| L4  Mapping (layered)                                          |
|     - Voxel hash (registration)                                |
|     - TSDF + RGB -> Marching Cubes mesh (render)              |
+---------------------------------------------------------------+
| L3  Back-end: discrete keyframe factor graph (iSAM2)           |
|     - Pose, velocity, bias, extrinsic variables                |
|     - GPS, loop, prior factors                                 |
|     - Robust kernels + switchable constraints + PCM            |
+---------------------------------------------------------------+
| L2  Front-end: continuous-time local fusion                    |
|     - B-spline trajectory over sliding window                  |
|     - Direct LiDAR registration + photometric + IMU + GNSS     |
|     - Per-axis observability tracking                          |
+---------------------------------------------------------------+
| L1  Per-sensor preprocessing                                   |
|     - Calibration, deskew, health monitoring                   |
+---------------------------------------------------------------+
| L0  Sensor abstraction                                         |
|     - Hardware sync (PPS), timestamping, calibrated streams    |
+---------------------------------------------------------------+
```

Deferred layers — path planning (ESDF, traversability) and semantic perception (segmentation, detection) — would sit beside L4/L5 in a later phase. They are not in this diagram by design.

The two cross‑cutting concerns — **time synchronization** and **calibration** — touch every layer and are discussed separately in §11 and §12.

## 4. L0 — Sensor abstraction

Design abstraction: every sensor is a stream of `(timestamp_monotonic, intrinsic_calibrated_measurement, covariance)`.

### 4.1 Time

The hard rule: every sensor timestamp lives on a single monotonic clock. The mechanism is **GNSS PPS distribution**:

- GNSS receiver emits a PPS pulse and a UBX/NMEA UTC time message.
- A small FPGA or microcontroller (or the GNSS receiver itself if it supports it) generates PTP grandmaster on the local network.
- Each LiDAR (Ouster supports PTP natively), the camera (via GPIO trigger), and the IMU (if it supports it; otherwise timestamped by the host with bounded jitter) align to PTP.
- The host clock is disciplined to PTP via `ptp4l`/`phc2sys`.

End result: sub‑microsecond timestamp alignment across sensors. This is non‑negotiable for the CT front‑end to converge.

If PPS/PTP is unavailable, fall back to host‑clock timestamping with **kalibr_allan**‑style online time‑offset estimation per sensor. Accept ~1‑5 ms residual sync error and let the CT trajectory absorb it.

### 4.2 Coverage geometry

For your planned rig (horizontal Ousters + dome Ouster + camera + IMU + GNSS):

- The dome Ouster (upward / hemispherical) is critical for **degeneracy immunity in vertical structures**: tunnels, overpasses, buildings with overhangs. It contributes massively to roll/pitch observability.
- Horizontal Ousters give yaw and translation. Two opposed horizontal Ousters give 360° azimuth coverage even with one occluded.
- The camera (or stereo pair) bridges across geometric ambiguity.
- IMU provides smoothness between LiDAR frames.
- GNSS provides global anchor when available.

This rig has no single failure point. **That is the whole design argument.**

### 4.3 Sensor health channel

Every sensor produces an out‑of‑band confidence stream: data‑rate ok, points/scan within nominal, intensity histogram nominal, no saturated camera frame, IMU bias residual bounded. The next layer consumes these to decide whether to trust a measurement.

This is the modern equivalent of "is the cable plugged in" — but graded, not binary.

## 5. L1 — Per‑sensor preprocessing

### 5.1 LiDAR (per sensor, parallel)

- **Range / intensity filtering**: clip min/max range, drop saturated intensity, drop self‑hits (rig structure).
- **Sensor‑frame deskew**: each LiDAR is deskewed using the *current best estimate of the CT trajectory* (from L2). The trajectory is queried at each point's true timestamp. This is fundamentally different from arc‑slam's "linearly interpolated IMU rotation, no translation" approach.
- **Point classification (optional)**: ground vs non‑ground split if the downstream wants it (LIO‑GC pattern, 2025). Mostly useful for planning, not for registration.

Result: a deskewed, calibrated point stream per LiDAR, each in its own sensor frame, with per‑point timestamps preserved.

### 5.2 Camera

- **Photometric calibration**: gamma, vignetting, auto‑exposure model. FAST‑LIVO2 does this; without it, direct visual residuals are unstable.
- **Rectification** (if stereo) or geometric undistortion.
- **Optional feature pyramid**: pre‑compute multi‑scale image pyramid once; reused by the front‑end (sparse direct) and the loop closure layer (feature‑based verification).

### 5.3 IMU

- **Bias initialization**: from stationary detection at startup (5 s of vehicle still).
- **Allan variance noise model**: white noise + bias random walk from offline calibration; feeds straight into the GTSAM `PreintegrationParams`.
- **Saturation flagging**: if accel or gyro hit the configured FSR, mark and propagate.

### 5.4 GNSS

- **Quality gating**: drop fixes below configured satellites‑in‑view, HDOP, fix‑type (no float, no autonomous if RTK is expected).
- **Spoofing detection**: compare GNSS‑derived velocity to IMU‑derived velocity over a 1 s window; large disagreement → drop the fix and raise a flag.
- **Coordinate conversion**: LLA → ECEF → ENU centred at the first valid fix. ENU is the working map frame.

## 6. L2 — Continuous‑time front‑end

This is the biggest architectural shift from your current system. The argument:

**The trajectory is a continuous function of time, not a sequence of poses.** Every measurement (LiDAR point, IMU sample, camera frame, GNSS fix) has a true timestamp; the trajectory at that timestamp is what the measurement constrains. Discrete pose graphs approximate this by deskewing measurements to keyframe times — fine when motion is slow and sensors are synced, broken when not.

### 6.1 Representation

A **non‑uniform cubic B‑spline** on $SE(3)$:

$$T(t) \;=\; \mathrm{SplineSE3}(t; \{c_k\})$$

with control points $c_k \in SE(3)$ at non‑uniformly spaced knots. Non‑uniformity is important: place knots dense when motion is fast or measurements are rich; sparse during stationary periods. Coco‑LIC (RA‑L 2023) is the closest published reference; Wildcat uses a similar idea.

Why cubic (degree 3): gives $C^2$ continuity — position, velocity, acceleration all continuous. Enough for IMU constraints. Quintic (degree 5) gives $C^4$ — overkill for our motion regime, doubles compute.

State: control points $\{c_k\}$ + per‑knot IMU bias $b_k$ (treated as piecewise‑linear interpolation; bias varies slowly enough). No separate velocity state — velocity is the derivative of $T(t)$.

### 6.2 Sliding window

The CT window holds 1–3 s of trajectory at any time. Older portions **marginalize** into a discrete keyframe pose at the window's tail edge; the marginalized prior is what hands off to L3.

Marginalization mechanism: Schur complement against the window's information matrix, producing a Gaussian prior on the trailing keyframe pose (and possibly velocity + bias). This is the same trick VINS‑Mono uses.

The window is the unit of high‑rate optimization. Within the window, the trajectory is continuously refined as new measurements arrive (incremental fixed‑lag smoother). At the window's tail, the trajectory is frozen and integrated into L3.

### 6.3 Measurement factors in the window

| Factor | Variables it constrains | Frequency |
|---|---|---|
| LiDAR point‑to‑plane | Control points near point timestamp | Tens of thousands per scan |
| IMU integration | Control points + bias near IMU sample | Per IMU sample (200–500 Hz) |
| Camera photometric patch | Control points near frame timestamp | Per visual frame (10–30 Hz) |
| GNSS position | Control point at GNSS timestamp | Per fix (1–10 Hz) |
| Extrinsic prior | Sensor → IMU extrinsic vars | Per sensor, weak prior |

LiDAR points are subsampled by voxel hash before contributing factors — we don't run 1 M factors per scan. Direct registration against the local voxel hash (L4.1) produces the residuals.

Camera factors are **sparse direct photometric** in the FAST‑LIVO2 style: pick image patches around projected LiDAR points (so depth is known), residual is photometric error, no descriptor matching. This is the most robust visual modality for a system with co‑located LiDAR.

### 6.4 Solver

Local fixed‑lag iSAM2 (GTSAM's `IncrementalFixedLagSmoother`) over the window's control points + biases. Relinearize whenever the threshold is crossed. The solver runs at LiDAR scan rate.

### 6.5 Per‑axis observability tracking

At each window optimization, eigendecompose the marginal Hessian on the trailing keyframe pose. Extract six per‑axis observability scores (X‑ICP / D²‑LIO style). Publish as part of the marginalized prior to L3, so the back‑end knows which DOFs are weak.

This is the **principled replacement** for arc‑slam's binary `isDegenerate` flag.

### 6.6 Why direct LiDAR registration here

Two reasons:

1. **Multi‑LiDAR makes direct overwhelmingly the right choice.** With 3 Ousters at 128 channels × 1024 columns × 10 Hz, you have ~4 M points/sec. LOAM smoothness extraction throws away 99 % of that. Direct registration uses all of it (subsampled by the voxel hash).
2. **Robustness in feature‑poor scenes.** Tunnels, snow, water — exactly the tactical scenes where LOAM fails. Direct point‑to‑plane keeps registering as long as there are *any* surfaces.

The cost is no edge/line classification for the back‑end — but loop closure descriptors at L5 don't need it (they work on raw clouds or organized scans).

## 7. L3 — Discrete keyframe back‑end

The CT front‑end discharges one keyframe per second (or one per metre travelled, whichever comes first). The back‑end is over keyframe poses and is the system's global memory.

### 7.1 Why keep iSAM2

Your original instinct was right. iSAM2 remains the right backbone in 2026 for incrementally‑updated factor graphs with loop closure. The Bayes tree relinearization with `relinearizeThreshold` lets a 10000‑keyframe graph absorb a fresh loop closure in tens of milliseconds. No competitor (g2o, Ceres in incremental mode, SymForce alone) does this better.

What changes vs your current code:
- **GTSAM ≥ 4.2 hybrid factor graphs**: model loop and GPS validity as discrete switches, not soft Sünderhauf approximations.
- **SymForce or symbolic factor generation**: for the custom factors you'll write (per‑axis observability prior, etc.), code‑gen the Jacobians. 5–10× faster than runtime expressions.
- **Schur complement on extrinsic / bias / velocity blocks** for marginal queries used by loop closure pre‑filtering.

### 7.2 Variables in the graph

| Variable | Symbol | When introduced |
|---|---|---|
| Keyframe pose | $X_k$ | Each keyframe (1 Hz nominal) |
| Keyframe velocity | $V_k$ | Each keyframe |
| IMU bias | $B_k$ | Each keyframe |
| Per‑sensor extrinsic | $E_s$ | Once per sensor; refined over time |
| GNSS → ENU origin | $G$ | Once at first fix |
| Multi‑robot peer pose (optional, future) | $R_p$ | Per peer |

### 7.3 Factor classes

| Factor | Source | Notes |
|---|---|---|
| Marginalized keyframe prior | CT window marginalization | Per‑axis covariance from observability |
| Sequential between‑factor | Marginalized from L2 | Tight |
| IMU preintegration (Forster) | Between keyframes (redundant with CT marg, used as a fallback) | Useful if CT window restarts |
| GNSS factor | Per fix that survives gating | **Switchable constraint** |
| Loop closure factor | L5 | **Robust kernel (GNC) + switchable** |
| Extrinsic prior | Calibration | Weak Gaussian, lets graph refine |
| Zero‑velocity factor | Detected stationary periods | Strong constraint when applicable |

### 7.4 Robustness mechanisms

The back‑end is the system's last line of defence against bad measurements. Three layers:

1. **Pre‑optimization filter — PCM (Mangelson 2018)**: before adding a batch of loop factors, find the maximum mutually consistent subset. Reject the rest.
2. **Per‑factor robustness — GNC (Yang 2020)**: each loop and GPS factor optimized with a graduated kernel that tightens from quadratic toward truncated‑least‑squares over a few iSAM iterations. Provably finds global optimum under bounded outlier ratios.
3. **Switchable constraints (Sünderhauf 2012, or GTSAM hybrid discrete)**: each suspect factor has an inlier/outlier latent. Optimization can deactivate persistently bad factors.

All three operate simultaneously. The cost is modest (low single‑digit ms per iSAM update).

### 7.5 Online extrinsic refinement

Extrinsics between sensors drift in tactical use (thermal, vibration, micro‑damage). The back‑end carries each $E_s$ as a variable with a weak prior centred on the offline calibration. As the rig moves through feature‑rich scenes, residuals refine the extrinsics. This is significantly more robust than locking in factory or one‑shot calibration values.

For multi‑LiDAR specifically, this is the only way to avoid month‑long miscalibration drift in the field.

### 7.6 Marginalization for long sessions

Multi‑hour missions produce 10000+ keyframes. iSAM2 stays performant by marginalizing old keyframes (outside any active loop closure window, > 1 km from current). Marginalized portions become **fixed priors** and remain visible for map assembly — but no longer optimization variables.

Conservative policy: marginalize keyframes only after `T_marg = 30 min` AND `d_marg = 1 km` from current pose. This preserves loop closure capability over the largest plausible operational range while keeping the active graph bounded.

## 8. L4 — Mapping (layered)

As decided earlier: no single representation covers registration and rendering well. The layers below are all keyed off the same keyframe poses from L3 and updated incrementally as poses are corrected.

The map stack here stops at the **colourised surface mesh**. The same TSDF substrate is what a future planning layer (ESDF) and semantic layer would build on — so this is forward‑compatible — but neither is specified here.

### 8.1 Registration layer — voxel hash

A flat **hash‑indexed voxel grid** over the active region (e.g., 200 m × 200 m × 50 m around current pose). Each voxel stores: point count, mean position, covariance, mean normal, mean intensity. Used by L2 for direct point‑to‑plane registration.

Voxel size: **adaptive (AVK‑SLAM 2026 pattern)**. Default 20 cm; subdivide to 10 cm in high‑density regions; merge to 40 cm in sparse regions. Maintains roughly constant points‑per‑voxel.

Active region slides with the pose. Voxels outside the active region are evicted to a slower global tier (next).

### 8.2 Global tier — NanoVDB sparse hierarchical grid

Outside the active region, voxel data lives in **NanoVDB** (NVIDIA's GPU‑friendly OpenVDB). Hierarchical, sparse, streams to disk for inactive cells. Used by L5 (place recognition) and L6 (rendering) for global queries. NOT used by L2 — front‑end only sees the active flat hash.

### 8.3 Surface layer — TSDF + RGB

**TSDF**: signed distance to nearest surface per voxel, with weight, integrated from each keyframe's LiDAR cloud (transformed by current best pose). **NVBlox** using GPU. Voxel size matches registration layer.

**Colour channel**: per‑voxel running average of camera RGB, projected at integration time. This is the "real mapper" colour you wanted.

> Deferred hook: NVBlox can derive an ESDF from this same TSDF (wavefront propagation) for free when the planning phase begins. It is simply not consumed in the current scope.

### 8.4 Mesh extraction

On demand (every N seconds), run **Marching Cubes** over the active TSDF region, producing a triangle mesh with per‑vertex colour. Streamed to L6 for visualization. For the final deliverable map at end of mission, optionally run **Screened Poisson reconstruction** (Kazhdan & Hoppe 2013) over the full integrated cloud for a watertight, smooth archival mesh.

### 8.5 Loop closure propagation

When L3 reports updated keyframe poses (after loop closure):

1. Identify affected keyframes ($\Delta \mathrm{pose} > \epsilon$).
2. Clear voxels last touched by any of those keyframes.
3. Re‑integrate each affected keyframe's points into the cleared voxels with the new pose.
4. Re‑extract the mesh over the affected region.

This is fast for voxel hash + TSDF (the structure was designed for it). It is intractable for 3DGS — which is one reason we are not putting a Gaussian layer in the system.

### 8.6 What is deliberately NOT in this stack

- **3D Gaussian Splatting layer**: not needed for tactical render; expensive; doesn't survive loop closure cleanly.
- **Pure neural implicit map**: tempting (INF‑SLiM, PIN‑SLAM) but adds GPU training time inside the live SLAM loop. Possible as an offline post‑processing layer for final maps.
- **ESDF / planning layer**: deferred. The TSDF substrate makes it cheap to add later.
- **Semantic layer (per‑voxel labels)**: deferred. Would attach a class channel to the TSDF when a segmenter is introduced.

## 9. L5 — Place recognition

Loop closure only. Object detection and semantic segmentation are deferred (out of scope, see §1).

**Hierarchical detection** as the SOTA pattern (§5.3 of `SOTA.md`):

1. **Scan Context++** for fast candidate retrieval. Compute polar descriptor per keyframe. KdTree over ring keys. CPU, sub‑millisecond per query.
2. **STD or BTC** for geometric re‑ranking. Triangle descriptors over stable keypoints. Filter top‑5 candidates.
3. **GICP** (not vanilla ICP) for fine pose alignment. Use the candidate's local submap from the TSDF. Verify by fitness score.
4. **PCM batch check** before factor submission to L3.

Loop factor noise: scaled by GICP fitness; wrapped in GNC + switchable. Submitted to L3 with all the robustness guarantees.

> Deferred hook: once a semantic layer exists, a **semantic histogram match** can be added as a parallel high‑recall proposer that is robust to season / lighting changes. It would compose cleanly with the geometric pipeline above, but it is not part of the current scope.

## 10. L6 — Tactical interface

Out of scope for the SLAM core but the consumer contract matters:

- **Map stream**: incremental colour mesh over WebSocket or DDS. Sub‑second latency.
- **Trajectory query**: the CT trajectory is queryable at any timestamp; useful for sensor fusion downstream.
- **Confidence visualization**: per‑voxel weight (TSDF) and per‑keyframe marginal covariance (iSAM2) exposed as a confidence channel. Operators see where the map is uncertain.

## 11. Cross‑cutting: time synchronization

Already in L0 but worth re‑emphasizing: the entire design depends on tight clock alignment. Without it:

- The CT trajectory cannot resolve sub‑scan motion correctly. LiDAR points get assigned to wrong poses.
- Camera photometric residuals miss their target.
- IMU preintegration intervals are wrong → bias drift baked in.

Investment: a few hundred euros of FPGA/PTP grandmaster + Ouster's native PTP support + camera GPIO trigger + IMU with PPS. Pays back ten times over in front‑end robustness.

Fallback when hardware sync fails: per‑sensor **online time offset estimation** (each $E_s$ extrinsic carries a $\Delta t_s$ scalar). Sloppy but graceful.

## 12. Cross‑cutting: calibration

Three calibration stages:

1. **Factory / one‑time intrinsic**: camera intrinsics + distortion, LiDAR per‑beam offsets (from manufacturer), IMU Allan variance.
2. **Per‑deployment extrinsic**: rig poses between all sensors. **Kalibr** for camera‑IMU; **target‑based** (board with reflective tape) for LiDAR‑camera; **hand‑eye motion calibration** for LiDAR‑IMU as a fallback. All produce priors.
3. **Online refinement**: extrinsics are variables in L3 with the priors above. Refined continuously by SLAM residuals.

This three‑stage pattern handles the real failure mode of tactical deployment: rig gets bumped, optics get dirty, sensors get replaced. Online refinement absorbs the change without a recalibration session.

## 13. Decision register — alternatives considered and why rejected

| Decision | Chosen | Considered | Why rejected |
|---|---|---|---|
| Backend optimizer | iSAM2 | Sliding‑window BA (VINS) | No global consistency over km |
| Backend optimizer | iSAM2 | Batch g2o / Ceres | Not incremental; can't keep up with mission length |
| Backend optimizer | iSAM2 | SymForce alone | Not a solver; a code‑gen layer — use with iSAM2 |
| Trajectory representation | CT B‑spline (window) + discrete keyframes | Pure discrete keyframes | Loses sub‑scan timing; fails fast motion |
| Trajectory representation | CT B‑spline (window) + discrete keyframes | Pure CT global | Doesn't scale to multi‑hour sessions; loop closure clumsy |
| Trajectory representation | CT B‑spline (window) + discrete keyframes | GP trajectory | Heavier; mostly research |
| Front‑end LiDAR | Direct point‑to‑plane | LOAM smoothness | Throws away points; brittle in feature‑poor scenes |
| Front‑end LiDAR | Direct point‑to‑plane | NDT | Less accurate than ikd‑Tree + point‑to‑plane on dense lidars |
| Front‑end visual | Sparse direct photometric (FAST‑LIVO2) | ORB/feature‑based | Less robust in low‑texture; descriptor cost |
| Front‑end visual | Sparse direct photometric | Dense photometric | Compute prohibitive at km scale |
| Map registration layer | Voxel hash (adaptive, AVK pattern) | ikd‑Tree | Equivalent accuracy, simpler; multi‑LiDAR fits voxel hash better |
| Map surface layer | TSDF + RGB → Marching Cubes | 3DGS | Tactical use doesn't need photorealism; loop closure breaks 3DGS |
| Map surface layer | TSDF + RGB → Marching Cubes | Implicit neural (INF‑SLiM) | Real‑time GPU budget; possible as offline post‑process |
| Map substrate | TSDF (NVBlox) | OctoMap | Strictly worse geometry; TSDF also forward‑compatible with deferred ESDF |
| Loop detection | Scan Context++ + STD + GICP + PCM | Geometric proximity (your code) | Fails after > 15 m drift |
| Loop detection | Scan Context++ + STD + GICP + PCM | Pure learned descriptors (MinkLoc3D) | GPU‑heavy; less explainable failure modes |
| Degeneracy handling | Per‑axis observability into noise | Binary projection (Zhang 2016) | Loses information; doesn't compose with multi‑sensor |
| Robust factors | GNC + switchable + PCM | Huber alone | Not aggressive enough for adversarial GPS / persistent bad loops |
| GNSS fusion | Switchable constraint + spoofing check | Always‑on Gaussian | Spoofing or RTK loss collapses the graph |
| Sensor sync | PTP grandmaster + per‑sensor sync | Software timestamping only | CT front‑end won't converge with 5 ms jitter |
| Camera fusion | Sparse direct, LiDAR‑provided depth | Stereo / mono with descriptors | Lower robustness in feature‑poor; more compute |

## 14. Open questions worth chewing on

1. **Single CT window vs per‑sensor CT windows.** A single global window simplifies; per‑sensor windows might handle very different sensor rates better. Coco‑LIC uses single; CLINS uses per‑sensor. No clear winner in the literature.
2. **Should the dome Ouster get its own front‑end?** The viewpoint is so different (looking up vs out) that joint registration might not converge as fast as parallel registration with a shared trajectory. Worth empirical test.
3. **How aggressively to lean on the camera in tactical operating modes.** Photometric residuals fail in darkness; do we drop them entirely or just inflate sigma? Adaptive answer requires a lighting model.
4. **Multi‑robot extensibility.** The single‑platform design above does not preclude multi‑robot, but the back‑end would need DC²SLAM‑style consensus. Decide now whether to architect for it.
5. **GPU dependency policy.** Embedded GPU (Jetson Orin) handles NVBlox TSDF + mesh, and optionally sparse‑direct visual. The deferred perception layers (segmentation, detection) would add the heaviest GPU load when introduced. Decide which subsystems are GPU‑required and which are optional.
6. **Map persistence and reuse across missions.** If today's map is tomorrow's prior, what's the storage format and the fusion rule for "this part of the map is from October, this part from May"? Not solved in the open literature.
7. **Adversarial scenarios.** Spoofing is in scope; jamming, deception via projected light or reflective targets is not. Should it be?

## 15. Where this lands vs the SOTA in `SOTA.md`

This design is unambiguously in the 2026 SOTA bracket because of:

- CT local + discrete global (Wildcat / Coco‑LIC class).
- Direct multi‑LiDAR registration with adaptive voxel hash (FAST‑LIO2 / AVK‑SLAM lineage).
- Tight LIVO with FAST‑LIVO2‑style sparse direct visual.
- Per‑axis observability into the back‑end (X‑ICP / D²‑LIO).
- GNC + switchable + PCM robustness stack.
- Hierarchical place recognition.
- Layered map (voxel hash → TSDF → colour mesh) — not the fashionable neural/Gaussian path, but the *right* path for the tactical use case.
- Online extrinsic refinement, sensor health channels, switchable GNSS.

What it consciously does not chase: photorealism, end‑to‑end learning, pure neural implicit maps, single‑representation unification. These are research bets; we want a system that ships.

What it consciously defers (out of scope, built on top later): path planning (ESDF, traversability) and semantic reconstruction (segmentation, object detection, semantic loop closure). The TSDF substrate is chosen partly so these slot in without redesign.

*End of design.*
