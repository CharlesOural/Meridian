# Meridian — Documentation

**Meridian** is a from-scratch, **tightly-coupled LiDAR-Inertial SLAM** system for tactical operational mapping. It produces a **colourised 3D mesh** in real time on an **NVIDIA Jetson Orin**.

It is a clean-slate rebuild — it reuses no prior code. Every technical claim in these docs is grounded in the apex open-source references (FAST-LIO2, FAST-LIVO2, Point-LIO) and the canonical papers (iSAM2/GTSAM, nvblox, Scan Context++); that reference grounding lives in each spec's non-normative **Appendix R**, verified against the clones in `/home/user/slam-reference`.

## The system in one paragraph

One LiDAR + one IMU (plus a camera and GNSS carried through unfused) feed a **discrete LIO front-end** (L2): each sweep is deskewed internally with a constant-screw model from its own IMU samples, then registered against a voxel-hash local map by a Gauss-Newton ICP seeded and regularised by an interval-averaged IMU prior. Keyframes flow to a **GTSAM iSAM2 back-end** (L3) for global consistency and loop closure (Scan Context++ → STD/BTC → GICP → PCM, L5). Keyframe poses drive an **nvblox GPU map** (L4) — TSDF + colour → Marching Cubes mesh — corrected by clear-and-rebuild on loop closure. The result streams to the operator as a colour mesh with a confidence overlay (L6).

**Design commitments (deliberately simple — best option per job, no fallbacks):**

- One front-end: the discrete LIO estimator, validated directly against ground truth. (Two predecessors — an iEKF test oracle and a continuous-time estimator — are retired; camera/GNSS fusion are future stages on the same seams.)
- Single LiDAR. (Multi-LiDAR is a future extension behind the same interfaces, not designed now.)
- Surface map behind the pluggable `ISurfaceMap` backend: nvblox (GPU) in production, a portable `cpu` backend for dev/non-CUDA boxes, deferred `vulkan`. One selected per run, explicitly and fail-fast — never a silent downgrade.
- ROS 2 **Humble**, C++20, colcon/ament_cmake. ROS-agnostic core C++ library + thin ROS 2 wrappers.

## How to read these docs

**If you want to build it →** read [`specs/00_architecture.md`](specs/00_architecture.md) and [`specs/01_interfaces_and_data_types.md`](specs/01_interfaces_and_data_types.md) first (they fix the module layout and the contracts every other spec depends on), then [`specs/11_build_system_libraries.md`](specs/11_build_system_libraries.md) to stand up the workspace, then the per-component specs.

**If you want to verify a claim →** each spec's non-normative **Appendix R** cites the SOTA references by `repo@sha` and `file:symbol`, verified against the clones in `/home/user/slam-reference`, with paper sections for the math.

---

## `specs/` — implementation specs (the build target)

| #   | Spec                                                                       | Scope                                                                                                         |
| --- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 00  | [`00_architecture.md`](specs/00_architecture.md)                           | ROS-agnostic core vs thin ROS 2 wrappers; package/dependency graph; threading; config; debug bus              |
| 01  | [`01_interfaces_and_data_types.md`](specs/01_interfaces_and_data_types.md) | **The contracts.** Core math/sensor/calibration types, the `KeyframePacket` (L2→L3), the six layer interfaces |
| 02  | [`02_sensors_timesync.md`](specs/02_sensors_timesync.md)                   | L0 — sensor abstraction; PTP time sync (linuxptp) from GNSS PPS                                               |
| 03  | [`03_preprocessing.md`](specs/03_preprocessing.md)                         | L1 — filtering, organized cloud, GNSS gate (deskew lives inside L2)                                           |
| 04  | [`04_frontend_estimation.md`](specs/04_frontend_estimation.md)             | **L2 — the discrete tightly-coupled LIO estimator** (internal screw deskew + voxel map + GN ICP)              |
| 05  | [`05_backend_graph.md`](specs/05_backend_graph.md)                         | L3 — GTSAM iSAM2 keyframe graph; no-double-counting hand-off; robust factors                                  |
| 06  | [`06_mapping.md`](specs/06_mapping.md)                                     | L4 — nvblox GPU TSDF+RGB → Marching Cubes mesh; clear-and-rebuild de-integration                              |
| 07  | [`07_loop_closure.md`](specs/07_loop_closure.md)                           | L5 — Scan Context++ → STD/BTC → small_gicp → PCM                                                              |
| 08  | [`08_calibration.md`](specs/08_calibration.md)                             | Calibration: offline prior + online extrinsic refinement                                                      |
| 09  | [`09_debug_introspection.md`](specs/09_debug_introspection.md)             | Debug topics, rviz markers, per-module timing — "see what the estimator is doing"                             |
| 10  | [`10_evaluation_harness.md`](specs/10_evaluation_harness.md)               | Replay==live harness; evo ATE/RPE; per-dataset acceptance                                                     |
| 11  | [`11_build_system_libraries.md`](specs/11_build_system_libraries.md)       | Library choice per job (justified); colcon workspace; CUDA/nvblox on Orin; version pins                       |

## Reference grounding — each spec's Appendix R

Each spec carries a non-normative **Appendix R — SOTA reference grounding** that cites the apex open-source references by `repo@sha` and `file:symbol` (plus paper eq/section for the math), verified against the clones in `/home/user/slam-reference`. By topic: manifold/ESIKF and IMU/deskew → `04_frontend_estimation.md`; iSAM2/GTSAM → `05_backend_graph.md`; TSDF/mesh → `06_mapping.md`; loop closure → `07_loop_closure.md`; engineering/debug → `03_preprocessing.md` and `09_debug_introspection.md`.

## Other

- [`DATASET.md`](DATASET.md) — the benchmark dataset (Newer College 2021, Ouster OS0-128 + Alphasense): bag setup, ground truth, calibration, and per-sequence roles including the `quad-hard` holdout.

