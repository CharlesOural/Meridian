# Meridian — Documentation

**Meridian** is a from-scratch, SOTA, **continuous-time tightly-coupled LiDAR-Inertial-Visual-GNSS SLAM** system for tactical operational mapping. It produces a **colourised 3D mesh** in real time on an **NVIDIA Jetson Orin**.

It is a clean-slate rebuild — it reuses no prior code. Every technical claim in these docs is grounded in the apex open-source references (FAST-LIO2, FAST-LIVO2, Point-LIO, ikd-Tree) and the canonical papers (Coco-LIC/CLINS for continuous-time, iSAM2/GTSAM, nvblox, Scan Context++), captured in `grounding/`.

## The system in one paragraph

One LiDAR + one IMU + one camera + GNSS feed a **continuous-time B-spline trajectory** (the front-end, L2): every measurement attaches a residual to the spline at its true timestamp — direct LiDAR point-to-plane, FAST-LIVO2-style sparse-direct photometric (LiDAR gives depth, so no triangulation), IMU as a derivative residual, and GNSS — all fused into one trajectory by a sliding-window solver. Keyframes flow to a **GTSAM iSAM2 back-end** (L3) for global consistency and loop closure (Scan Context++ → STD/BTC → GICP → PCM, L5). Keyframe poses drive an **nvblox GPU map** (L4) — TSDF + colour → Marching Cubes mesh — corrected by clear-and-rebuild on loop closure. The result streams to the operator as a colour mesh with a confidence overlay (L6).

**Design commitments (deliberately simple — best option per job, no fallbacks):**
- Full CT LIVO+GNSS from the start. No phased rollout, no "iEKF v1 then CT v2". (An iEKF is kept only as an offline test oracle behind `IFrontEnd`.)
- Single LiDAR. (Multi-LiDAR is a future extension behind the same interfaces, not designed now.)
- nvblox, GPU-only. No CPU fallback, no second map backend.
- ROS 2 **Humble**, C++20, colcon/ament_cmake. ROS-agnostic core C++ library + thin ROS 2 wrappers.

## How to read these docs

**If you want to understand the system →** start with [`SYSTEM_OVERVIEW.md`](SYSTEM_OVERVIEW.md) (the end-to-end narrative, photons-to-mesh), then the [`course/`](course/) chapter for the math.

**If you want to build it →** read [`specs/00_architecture.md`](specs/00_architecture.md) and [`specs/01_interfaces_and_data_types.md`](specs/01_interfaces_and_data_types.md) first (they fix the module layout and the contracts every other spec depends on), then [`specs/11_build_system_libraries.md`](specs/11_build_system_libraries.md) to stand up the workspace, then the per-component specs.

**If you want to verify a claim →** every spec cites a `grounding/` dossier; the dossiers cite `file:line` in the reference code and paper sections.

---

## `specs/` — implementation specs (the build target)

| # | Spec | Scope |
|---|---|---|
| 00 | [`00_architecture.md`](specs/00_architecture.md) | ROS-agnostic core vs thin ROS 2 wrappers; package/dependency graph; threading; config; debug bus |
| 01 | [`01_interfaces_and_data_types.md`](specs/01_interfaces_and_data_types.md) | **The contracts.** Core math/sensor/calibration types, the `KeyframePacket` (L2→L3), the six layer interfaces |
| 02 | [`02_sensors_timesync.md`](specs/02_sensors_timesync.md) | L0 — sensor abstraction; PTP time sync (linuxptp) from GNSS PPS |
| 03 | [`03_preprocessing.md`](specs/03_preprocessing.md) | L1 — filtering, organized cloud, deskew (spline-query + IMU-only cold-start) |
| 04 | [`04_frontend_estimation.md`](specs/04_frontend_estimation.md) | **L2 — the CT B-spline tightly-coupled LIVO+GNSS estimator** (basalt-headers + Ceres) |
| 05 | [`05_backend_graph.md`](specs/05_backend_graph.md) | L3 — GTSAM iSAM2 keyframe graph; no-double-counting hand-off; robust factors |
| 06 | [`06_mapping.md`](specs/06_mapping.md) | L4 — nvblox GPU TSDF+RGB → Marching Cubes mesh; clear-and-rebuild de-integration |
| 07 | [`07_loop_closure.md`](specs/07_loop_closure.md) | L5 — Scan Context++ → STD/BTC → small_gicp → PCM |
| 08 | [`08_calibration.md`](specs/08_calibration.md) | Calibration: offline prior + online extrinsic refinement |
| 09 | [`09_debug_introspection.md`](specs/09_debug_introspection.md) | Debug topics, rviz markers, per-module timing — "see what the estimator is doing" |
| 10 | [`10_evaluation_harness.md`](specs/10_evaluation_harness.md) | Replay==live harness; evo ATE/RPE; per-dataset acceptance |
| 11 | [`11_build_system_libraries.md`](specs/11_build_system_libraries.md) | Library choice per job (justified); colcon workspace; CUDA/nvblox on Orin; version pins |

## `grounding/` — evidence base

Code- and paper-grounded dossiers (`file:line` + paper section citations). See [`grounding/README.md`](grounding/README.md) for the full index. 01 manifold/ESIKF · 02 IMU/deskew · 03 LiDAR/ikd-Tree · 04 FAST-LIVO2 visual · 05 Point-LIO · 06 engineering/debug · 07 TSDF/mesh · 08 loop closure · 09 iSAM2 · 10 continuous-time.

## `course/` — the textbook chapter

[`course/TIGHTLY_COUPLED_ESTIMATION.md`](course/TIGHTLY_COUPLED_ESTIMATION.md) — a ~430 KB graduate-level chapter on tightly-coupled multi-sensor estimation and residuals (12 sections, also under `course/sections/`): manifolds → probability → IMU → LiDAR → visual → GNSS → batch & filter solving → continuous-time → robustness → synthesis. Written to the depth of a robotics textbook, grounded in the reference code.

## Other

- [`DATASET.md`](DATASET.md) — dev/eval datasets (FusionPortable primary, M2DGR co-primary — all modalities at once).
- [`reference/`](reference/) — the 2026 SOTA survey (`SOTA.md`) and the archived arc-slam design exploration (`NEXT_GEN_DESIGN_archived.md`, superseded — kept for rationale only).

## Status

Specs complete and internally consistent (reframed to full-CT / single-LiDAR / nvblox-only; cross-references self-contained within this repo). **Next step: Phase 0 scaffolding** — stand up the colcon workspace (`meridian_common`, `meridian_core` layers, `meridian_ros`, `meridian_msgs`, `meridian_tools`) per spec 00/11 and the bag-replay harness on FusionPortable.
