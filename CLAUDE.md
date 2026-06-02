# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**Meridian** is a from-scratch, continuous-time, tightly-coupled
LiDAR-Inertial-Visual-GNSS SLAM system targeting the **NVIDIA Jetson Orin**,
producing a colourised TSDF mesh in real time. It is proprietary (see `LICENSE`).

**Current state: docs + dev-environment only — there is no source code yet.**
`src/` is the (empty) colcon source space. The system is *fully specified* in
`docs/specs/` (normative) before being built. When implementing, the specs are
the contract — read the relevant spec first; do not invent behaviour that
contradicts it. Build/integration order is spec 00 §13 (cross-cutting types →
L0/L1 → L2 CT front-end → L4 map → L3 back-end → L5 loop closure → L6/wrapper).

## Documentation map (read these, in this order)

- `docs/SYSTEM_OVERVIEW.md` — end-to-end narrative (photons → mesh) and the *why*.
- `docs/specs/00_architecture.md` — **authoritative**: package layout, dependency
  rules, threading, the no-ROS rule, the `KeyframePacket` contract.
- `docs/specs/01_interfaces_and_data_types.md` — the value types & interfaces every
  other spec depends on.
- `docs/specs/11_build_system_libraries.md` — library canon, version pins, colcon
  workspace, CUDA/nvblox build.
- `docs/specs/02..10` — per-layer specs (sensors/timesync, preprocessing, front-end,
  back-end, mapping, loop closure, calibration, debug, evaluation).
- `docs/grounding/` — `file:line` + paper-eq citations into the reference systems
  (FAST-LIO2, FAST-LIVO2, nvblox, iSAM2, Coco-LIC). `docs/course/` — textbook-depth math.
- `DEVELOPMENT.md` — the dev-environment runbook (summarised below).

## Non-negotiable invariants (the things that must not break)

These thread through every spec; violating one is an architecture bug, several
are CI-enforced (spec 00 §9.4, spec 11 §9.3):

1. **ROS-agnostic core + thin ROS 2 wrapper.** No package except `meridian_ros`
   (and `meridian_msgs`) may depend on `rclcpp`/`*_msgs`/DDS. Core uses plain C++
   (and CUDA). Time is `int64` nanoseconds (`meridian::Time`), never `ros::Time`,
   below the wrapper. Logging/telemetry are sinks the wrapper binds.
2. **`KeyframePacket` is the only value crossing L2→L3** (def: spec 01 §6 /
   spec 00 §6). Exactly **one** geometric constraint per keyframe interval:
   `RELATIVE_BETWEEN` (default) or `IMU_PREINT` (only on window-restart, mutually
   exclusive) — never both. Velocity/bias ride as seeds, not graph variables, by
   default. This is the no-double-counting rule.
3. **The front-end IS the CT B-spline LIVO+GNSS estimator** (`meridian_frontend/src/ct/`).
   The FAST-LIO2-style iEKF (`src/iekf/`) is a **test oracle only**, never a product
   path or a "v1".
4. **The map is nvblox, GPU-only — no CPU fallback, no second backend.**
   `meridian_map` has exactly one impl (`src/nvblox/`). Loop correction =
   clear-and-re-integrate the affected region from the persistent keyframe cloud
   store (a running-average TSDF is not per-keyframe reversible).
5. **One LiDAR + one IMU + one camera + GNSS.** No multi-LiDAR logic.
6. **Layers depend downward + cross-cutting only.** Siblings talk through `I*`
   interfaces + `meridian_common` value types; `meridian_pipeline` is the only wirer.
   The L2→L1 deskew edge is a runtime `IDeskewProvider` injection, not a build dep.
7. **One library per job, one production impl per interface** (spec 11 §1). No
   dual code paths, no defensive fallbacks. Uncertainty is tracked **per-axis**
   (6 observability scores), never a binary degeneracy flag.

## Architecture in one screen

Six layers behind one-implementation interfaces, ROS-agnostic core + ROS 2 wrapper:

```
L6 operator surface ....... colour mesh + confidence overlay        (meridian_ros)
L5 place recognition ...... ScanContext++ → STD/BTC → small_gicp → PCM   (meridian_place)
L4 map .................... nvblox GPU TSDF+colour → Marching Cubes   (meridian_map)   [GPU only]
L3 back-end ............... GTSAM iSAM2 keyframe graph + GNC/PCM       (meridian_backend)
L2 front-end .............. CT split-SO(3)×ℝ³ B-spline window, tight   (meridian_frontend)  [IFrontEnd]
                            LiDAR p2p + sparse-direct photometric + IMU + GNSS
L1 preprocess ............. filter, photometric calib, pyramid, deskew (meridian_preprocess)
L0 sensors+time ........... PTP/PPS sync → one monotonic clock         (meridian_sensors, meridian_time)
   cross-cutting: meridian_common (math/types, KeyframePacket), meridian_config, meridian_calib, meridian_debug
```

Data flow: L0 stamps every measurement on one clock → L1 conditions it → **L2
fuses LiDAR/visual/IMU/GNSS into one continuous trajectory at each measurement's
true timestamp** (deskew is implicit — the trajectory *is* the deskew) → emits
`KeyframePacket` → L3 iSAM2 keeps the global graph consistent (loops via L5,
GNSS, online extrinsics) → broadcasts `PoseCorrection` → L4 (re)builds the
nvblox map from the keyframe cloud store at corrected poses → L6 streams the mesh.

Threading (spec 00 §11): fixed stage threads + bounded queues, not a pool. Front-end
(T2) is the priority thread; back-end/map run on separate threads. `Q_meas` is
lossy under overload; `Q_kf`/`Q_map` are lossless. A `--single-thread` mode gives
bit-reproducible replay.

### Package / dependency layout

`meridian_cmake` (shared toolchain/deps cmake) → X-cut (`meridian_common`,
`meridian_config`, `meridian_time`, `meridian_debug`, `meridian_calib`) → layer
packages (`meridian_sensors`, `meridian_preprocess`, `meridian_frontend`,
`meridian_backend`, `meridian_map`, `meridian_place`) → `meridian_pipeline` (wires
all, NO ROS) → `meridian_msgs` + `meridian_ros` (the only ROS-dependent packages)
→ `meridian_tools` (offline bag replay / eval). Vendored under `vendor/` as git
submodules: `basalt-headers` (CT spline kernel), `ikd-Tree` (registration oracle),
`scancontext`. Library canon

## Dev environment & commands

The toolchain lives in a container — never install on the host. `docker/install-deps.sh`
is the single source of truth for the dependency stack (ROS 2 Humble, C++20,
Eigen/Sophus/Ceres 2.1/GTSAM 4.2/PCL/OpenCV/small_gicp, etc.); both Dockerfiles
run it. **CUDA/nvblox is Linux-GPU only** — Apple Silicon builds everything except
`meridian_map` (and the packages that link it).

**`DEVELOPMENT.md` is the runbook** — container setup (distrobox / Docker), the
one-time workspace bring-up, build invocations, and viz. Follow it rather than
duplicating commands here. Tests run via ament/GoogleTest (`colcon test`;
`--packages-select <pkg>` for one package, `--ctest-args -R <name>` for one test).

CI gates to keep green (spec 11 §9.3): no-ROS grep over `src/meridian_*` (excl.
`meridian_ros`/`meridian_msgs`); dependency lint (the §4 edge set); no CUDA outside
`meridian_map`; clang-tidy/clang-format with `-Werror` in core.

## Conventions

- C++20, `CMAKE_CXX_EXTENSIONS OFF`. One public class ≈ one header + one `.cpp`;
- Public headers under `include/meridian/<module>/`; interfaces are `I*.hpp`
  pure-virtual + a factory free function; impls live in `src/` and are not exported.
- Tangent ordering is explicit and differs by type — `KeyframePacket.information`
  is `[rx,ry,rz,tx,ty,tz]`; `NavState`/observability are translation-first. Always
  check the spec; mis-ordering covariance is a classic silent bug.
- Validation datasets (`docs/DATASET.md`): FusionPortable (primary), M2DGR (GNSS).
  Replay == live: the offline harness drives the same `MeridianPipeline` via the
  same `ISensorSource` as the live node.

## Comment discipline

Comments describe **what the code does and why the logic is correct** — control
flow that isn't obvious, invariants, edge cases, units, and technical/mathematical
reasoning (derivations, why a formula holds, numerical caveats). That is the only
thing they are for.

Do **not** put in comments:
- **External references** — no `per spec 11 §3`, `see grounding/...`, paper/equation
  citations, or doc section pointers. The code must read self-contained.
- **Project rationale / decisions** — no "chosen for reproducibility", "matches the
  reference", "simplicity mandate". Design rationale belongs in the specs/docs and
  in commit messages, never in code.
- **Environment or personal context** — no "your daily driver", "RTX 3070",
  "the Mac friend", build-host asides.

Bad:  `// Ceres 2.1 needed for the Manifold API (spec 11 §3)`
Bad:  `// clear-and-reintegrate because the TSDF isn't reversible (see grounding 07)`
Good: `// running-average TSDF can't subtract a stale pose, so clear the region first`
Good: `// r = n·(R·p_L + t) + d  — signed point-to-plane distance in world frame`

Keep them sparse: prefer self-explanatory names and structure; comment the
non-obvious, not the line-by-line.
