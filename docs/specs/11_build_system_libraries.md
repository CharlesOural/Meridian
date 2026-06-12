# 11 — Build System & Library Choices

> **Spec status:** normative *engineering* spec. This document fixes **what
> Meridian is built from and how the workspace is stood up**: the single chosen
> third-party library for every job, the colcon/ament_cmake workspace layout,
> the inter-package dependency graph, how vendored headers are pulled in, the
> CUDA/nvblox build setup for the Jetson Orin target, and exact version pins. A
> real engineer must be able to clone, `colcon build`, and run from this spec
> plus the package skeletons it describes.
>
> **Scope.** This spec is *infrastructure*, not algorithms. The math and runtime
> behaviour of each component live in their own specs; here we only fix the
> *library that implements it* and *how it links*. First-pass scope ends at a
> colourised triangle mesh (spec 00 §0); ESDF and semantics are deferred hooks,
> so their libraries (none new) are not in the build.
>
> **System framing (supersedes any earlier phased plan).** Meridian is **one
> complete system**: a discrete, tightly-coupled LiDAR-Inertial estimator. The
> L2 front-end is a **per-sweep Gauss-Newton point-to-point ICP** against a
> voxel-hash local map, with internal constant-screw deskew and an
> interval-averaged IMU prior, all behind
> the `IFrontEnd` interface (spec 01 §7.3); camera and GNSS are carried through
> the pipeline unfused. The estimator links **no solver, NN, or image library**
> — Eigen + Sophus only. The deployment
> target is an **NVIDIA Jetson Orin with CUDA always present**; mapping is
> **nvblox, GPU-only, no CPU fallback**.
>
> **Companion specs.** `00_architecture.md` (package layout §2, dependency rules
> §4, build system §9 — this spec is the full expansion of §9), `01_interfaces_
> and_data_types.md` (the core value types the math packages compile against),
> `06_mapping.md` (L4 / nvblox runtime), `04_frontend_estimation.md` (the LIO
> front-end), `05_backend_graph.md` (GTSAM/iSAM2 runtime). Reference grounding for
> the libraries chosen here lives in those specs' non-normative Appendix R: TSDF/
> mesh in `06_mapping.md` Appendix R, iSAM2/GTSAM in `05_backend_graph.md`
> Appendix R.

---

## Table of contents

1. [Library canon — one chosen library per job](#1-library-canon--one-chosen-library-per-job)
2. [Toolchain & language baseline](#2-toolchain--language-baseline)
3. [Version pin table](#3-version-pin-table)
4. [colcon workspace layout](#4-colcon-workspace-layout)
5. [ament_cmake target & dependency graph](#5-ament_cmake-target--dependency-graph)
6. [Vendored headers (Scan Context)](#6-vendored-headers-scan-context)
7. [CUDA / nvblox build setup for Jetson Orin](#7-cuda--nvblox-build-setup-for-jetson-orin)
8. [Finding & exporting each dependency in CMake](#8-finding--exporting-each-dependency-in-cmake)
9. [colcon invocation, mixins, and CI](#9-colcon-invocation-mixins-and-ci)
10. [Standing up the workspace from scratch (the runbook)](#10-standing-up-the-workspace-from-scratch-the-runbook)
11. [Considered & rejected](#11-considered--rejected)

---

## 1. Library canon — one chosen library per job

The governing principle (spec 00, and the project owner's simplicity mandate):
**pick the single best library per job and commit.** No dual code paths, no "or
alternatively" hedges, no defensive CPU fallback. The table below is the whole
build's bill of materials; every later section just wires these in. The
"considered & rejected" column is a one-line pointer to §11 — the design itself
takes exactly one option.

| Job | **Chosen library** | One-line justification | Rejected (→ §11) |
|---|---|---|---|
| Linear algebra (vectors, matrices, dense solves) | **Eigen 3.4** | The de-facto C++ numerical core; every other library here (Sophus, GTSAM, PCL, OpenCV) already speaks `Eigen::`, so it is the lingua franca with zero glue. | Blaze, Armadillo |
| Lie groups (SO(3)/SE(3) exp/log, manifolds) | **Sophus** | Header-only `SO3`/`SE3` on Eigen, matching the box-plus/box-minus and right-perturbation convention spec 01 §3.1 mandates. | hand-rolled so3_math, manif |
| Back-end optimiser (incremental factor graph) | **GTSAM 4.2** | Ships `ISAM2` (the incremental Bayes-tree smoother spec 05 needs), `CombinedImuFactor`, `noiseModel::Robust`+`Huber`, and `GncOptimizer` — the exact factor/robustness set spec 05 specifies, in one BSD library. | g2o, Ceres-only back-end, SE-Sync |
| Per-sweep registration solver | **Meridian GN (in `meridian_frontend/src/lio/`)** | The front-end's solve is a 6-DoF Gauss-Newton point-to-point ICP whose normal equations are a closed-form 6×6 LDLT — small enough that a library solver buys nothing and costs determinism. (Ceres remains installed in the container but is **unlinked**: its consumer, the removed CT window solver, no longer exists.) | Ceres, GTSAM for the per-sweep solve |
| Dense mapping: TSDF + colour + mesh | **nvblox (isaac_ros_nvblox)** | GPU TSDF + GPU colour fusion + GPU Marching Cubes in one CUDA library with a maintained ROS 2 wrapper; on the guaranteed-CUDA Jetson Orin it is the only map backend (spec 06 Appendix R). **No CPU fallback.** | VDBFusion, Voxblox, OpenVDB/NanoVDB |
| Nearest-neighbour map for registration | **voxel-hash maps (Meridian)** | The front-end's local map (`meridian_frontend/src/lio/`) and the L4 registration map (spec 06 §3) are both `std::unordered_map`-based voxel hashes with fixed-order neighbour probes — deterministic by construction and validated against brute-force references in unit tests, with no external NN library. | nanoflann, PCL KdTree, reference k-d trees |
| Fine registration (loop-closure GICP verify) | **small_gicp** | Header-light, multi-threaded GICP/VGICP that takes Eigen point buffers directly — the L5 verify step (spec 07) wants a fast, dependency-thin GICP, not full PCL registration. | PCL GICP, libpointmatcher |
| Point-cloud I/O & filters | **PCL** (io/filters only, sparingly) | Used *only* for bag/file I/O and a couple of filters at the edges; the hot path computes on `meridian::LidarPoint` buffers, never `pcl::PointCloud` (spec 01 §1 R1). PCL is a heavy dependency, so it is fenced to `meridian_preprocess`/`meridian_tools`. | PCL everywhere (rejected by R1) |
| Images (pyramids, undistort, Bayer) | **OpenCV 4** | L1 image conditioning (debayer, rectify, pyramid; spec 03 §5) and the ROS edge need exactly OpenCV's `imgproc`/`calib3d`. Fenced to `meridian_preprocess`/`meridian_ros`/`meridian_tools`; the estimator never links it. | bespoke image code |
| Loop / place descriptors | **vendored Scan Context++** (+ optional STD/BTC hooks) | A small vendored header implementing the rotation-invariant ring-key descriptor spec 07's pre-filter needs; no external package, no version drift. | DBoW2/3, learned descriptors |
| ROS 2 middleware & message types | **ROS 2 Humble** (`rclcpp`, `sensor_msgs`, `message_filters`, `tf2`, `visualization_msgs`, `diagnostic_updater`, `rosbag2`) | Humble is the current Jetson/JetPack-aligned LTS; confined to `meridian_ros` only (spec 00 §1.1) so the core stays middleware-free. | ROS 2 Iron/Jazzy (not Jetson-LTS), ROS 1 |
| LiDAR driver | **ouster-ros** | The single-LiDAR target is an Ouster; the official driver already emits per-point time (`ouster_ros::Point::t`) which spec 01 §4.2 makes mandatory for deskew. | bespoke UDP parser |
| Time sync (PTP / PPS discipline) | **linuxptp** (`ptp4l` + `phc2sys`) | System-level PTP daemons discipline the NIC/PHC clock spec 02 relies on; this is host configuration, not a linked library. | chrony-only, software-only sync |
| Build / test / packaging | **colcon + ament_cmake + GoogleTest** | The ROS 2 standard build+test trio; one `colcon build` / `colcon test` covers core libraries, the wrapper, and unit tests uniformly (spec 00 §9). | plain CMake, catkin, Bazel |
| Offline trajectory eval | **evo** | The community-standard APE/RPE tool the evaluation harness (spec 10) shells out to; offline only, not a build dependency of the core. | rpg_trajectory_evaluation |

Everything below is the *plumbing* for this one table.

---

## 2. Toolchain & language baseline

| Setting | Value | Rationale / where enforced |
|---|---|---|
| OS / platform | Ubuntu 22.04 (`jammy`) on **NVIDIA Jetson Orin**, JetPack 6.x | Humble's Tier-1 platform and the JetPack 6 base image; the deployment target (CUDA always present). |
| ROS 2 distro | **Humble Hawksbill** (LTS) | §3 pin; confined to `meridian_ros`. |
| C++ standard | **C++20**, required, no GNU extensions | `set(CMAKE_CXX_STANDARD 20)`, `CMAKE_CXX_STANDARD_REQUIRED ON`, `CMAKE_CXX_EXTENSIONS OFF` in the top `meridian_cmake` toolchain include; per-target `target_compile_features(... cxx_std_20)`. C++20 buys concepts (constrain the `I*` interfaces), `std::span` (zero-copy `LidarPoint` views), and designated initialisers for `Config` (spec 00 §9.1). There is **no C++17 fallback path** — the Jetson GCC 11 / JetPack toolchain supports C++20. |
| Host compiler | GCC 11 (Ubuntu 22.04 default) | Matches the ROS 2 Humble binary ABI; `clang` 14+ allowed for local dev but CI gates on GCC 11. |
| CUDA toolkit | **CUDA 12.x** (the JetPack 6 system CUDA) | nvblox's only requirement; `CMAKE_CUDA_STANDARD 17` (nvblox kernels are C++17; the *host* side is C++20). See §7. |
| CUDA architecture | `CMAKE_CUDA_ARCHITECTURES = 87` (Orin = SM 8.7) | One arch, the deployment SoC. Add `89`/`86` only if an x86+RTX dev box is used for offline runs; the Orin build pins `87`. |
| Warnings | `-Wall -Wextra -Wpedantic`, warnings-as-errors in **core** packages | spec 00 §9.4; the wrapper and third-party-heavy edges (PCL/OpenCV includes) relax `-Werror` selectively. Vendored upstreams are brought in through **`SYSTEM` include directories** so their headers are exempt from the house warning set (a third-party header must never be able to fail a `-Werror` core build). |
| Default build type | **`Release` when `CMAKE_BUILD_TYPE` is unset** | An unset build type compiles `-O0`, which silently disables Eigen's vectorisation and inlining and makes the per-scan hot loops unusably slow (the "unoptimized-Eigen trap"). `MeridianToolchain.cmake` forces `CMAKE_BUILD_TYPE=Release` into the cache when neither it nor `CMAKE_CONFIGURATION_TYPES` is set; an explicit `-DCMAKE_BUILD_TYPE=...` (or a colcon mixin, §9.1) always overrides it. |
| Position-independent code | **`CMAKE_POSITION_INDEPENDENT_CODE ON` workspace-wide** | Set in `MeridianToolchain.cmake` so every static archive (any per-package `STATIC` helper) is `-fPIC` and folds cleanly into the `SHARED` layer libraries and into Python bindings. Without it a static-into-shared link fails on x86-64. |
| Sanitizers | ASan/UBSan via a colcon mixin for the test build (host only, not on-device) | §9. |
| Formatting / lint | `clang-format` + `clang-tidy` on every core TU | spec 00 §9.4. |

> **Why one C++ standard, one CUDA arch, one ROS distro.** The simplicity
> mandate applies to the build too: every "fall back to X if Y" in a build
> system is a second configuration nobody tests. We pin one of each and commit.

---

## 3. Version pin table

These are the exact versions the workspace is validated against. Pins live in
`meridian_ws/dependencies.repos` (vcs), the per-package `package.xml`
`<depend>`/`<version_*>` constraints, and `meridian_cmake/cmake/MeridianDeps.cmake`
(the `find_package(... <ver> EXACT?)` calls in §8).

| Dependency | Pinned version | How obtained | Notes |
|---|---|---|---|
| ROS 2 | **Humble** (latest patch on the Humble apt line) | apt (`ros-humble-desktop` on x86 dev; `ros-humble-ros-base` + needed pkgs on Orin) | Driver of `rclcpp`, `sensor_msgs`, `tf2`, `message_filters`, `visualization_msgs`, `diagnostic_updater`, `rosbag2`. |
| Eigen | **3.4.0** | apt `libeigen3-dev` (3.4.0 on jammy) | Header-only; ABI-stable. The single most pinned-down dep because everything aligns its `Eigen::` types to it. |
| Sophus | **1.22.10** (the 2022 tagged release) | vcs source build *or* apt where available | Header-only; must be the Eigen-3.4-compatible tag. |
| GTSAM | **4.2.0** | source build (CMake), `GTSAM_USE_SYSTEM_EIGEN=ON` | Build with `GTSAM_BUILD_WITH_MARCH_NATIVE=OFF` (reproducible Orin binary), `GTSAM_USE_SYSTEM_EIGEN=ON` (one Eigen, §8), `GTSAM_WITH_TBB=OFF` unless TBB is also pinned. Provides ISAM2 / CombinedImuFactor / Robust+Huber / GncOptimizer (spec 05 Appendix R). |
| Ceres | **2.1.0** | apt `libceres-dev` (2.1 on jammy) | **Installed in the container but unlinked**: its consumer, the CT window solver, was removed with the CT front-end. The pin documents the container image (avoids an image rebuild), not a link dependency — no `meridian_*` package may link it. |
| nvblox | **isaac_ros_nvblox, Humble release line** (pin the commit in `dependencies.repos`) | vcs source build (CUDA) | The only map backend; §7. Built against system CUDA 12.x. |
| CUDA | **12.x** (JetPack 6 system CUDA) | JetPack / apt | `nvcc`, cuBLAS, Thrust — all nvblox needs. |
| PCL | **1.12.x** (jammy) | apt `libpcl-dev` | io/filters only; fenced to `meridian_preprocess`/`meridian_tools`. |
| OpenCV | **4.5.x** (jammy) | apt `libopencv-dev` | imgproc/video/calib3d for the visual track. On Orin, the JetPack OpenCV (CUDA-enabled) is acceptable but Meridian uses only CPU OpenCV calls — do not depend on CUDA-OpenCV. |
| small_gicp | pinned commit | vcs source (header + small lib) | GICP verify (spec 07). |
| ouster-ros | **Humble branch**, pinned commit | vcs source | LiDAR driver; only `meridian_ros`/`meridian_tools` depend on it. |
| linuxptp | distro `linuxptp` (apt) | apt | Host PTP daemons; runtime, not linked. |
| GoogleTest | the `ament_cmake_gtest` / `gtest_vendor` that ships with Humble | ament | Test framework. |
| evo | latest `pip` `evo` | pip (offline tooling) | Not a build dependency; used by spec 10's harness scripts. |
| **Scan Context++** | pinned commit/file, **vendored** | git submodule / copied header (§6) | Loop descriptor; the only vendored upstream. |

> **Pin discipline.** Source-built deps (Sophus, GTSAM, nvblox,
> small_gicp, ouster-ros) are listed in `meridian_ws/dependencies.repos` with an
> exact `version:` (tag or commit). apt deps are pinned by the Humble/jammy apt
> snapshot used in the CI container image (a tagged Docker base, §9). Vendored
> headers are pinned by submodule SHA. **Nothing floats on `main`.**

---

## 4. colcon workspace layout

The workspace is the one spec 00 §2 defines; this section adds the *build-system
artefacts* (vcs repos file, vendored submodules, cmake helper package, colcon
mixins) and shows where they sit.

```
meridian_ws/
├─ src/
│  ├─ meridian_cmake/              # NEW: shared CMake — toolchain, deps finder, helpers
│  │   cmake/MeridianToolchain.cmake     #   C++20 / CUDA arch / warning flags
│  │   cmake/MeridianDeps.cmake          #   find_package() wrappers for every §1 lib
│  │   cmake/MeridianVendored.cmake      #   INTERFACE target for vendor/scancontext
│  │   cmake/MeridianTesting.cmake       #   gtest + sanitizer mixins
│  │   package.xml                    #   ament_cmake, exports the cmake/ dir
│  │
│  ├─ meridian_common/            # X-cut: math/types, KeyframePacket   (Eigen, Sophus)
│  ├─ meridian_config/            # X-cut: typed Config + YAML loader    (yaml-cpp)
│  ├─ meridian_time/              # X-cut: time model, PTP/PPS interp     (—)
│  ├─ meridian_debug/             # X-cut: telemetry bus                  (—)
│  ├─ meridian_calib/             # X-cut: extrinsic/intrinsic models     (Eigen, Sophus)
│  │
│  ├─ meridian_sensors/           # L0: ISensorSource + sync             (Eigen)
│  ├─ meridian_preprocess/        # L1: downsample/validity/gnss-gate     (Eigen, OpenCV[image])
│  ├─ meridian_frontend/          # L2: IFrontEnd = discrete LIO          (Eigen, Sophus — nothing else)
│  │   include/meridian/frontend/
│  │   src/lio/                #       THE front-end: voxel map, IMU tracker,
│  │   │                       #       screw deskew + GN ICP, orchestration
│  │   src/frontend_factory.cpp
│  │
│  ├─ meridian_backend/           # L3: IBackEnd = GTSAM iSAM2            (GTSAM)
│  ├─ meridian_map/               # L4: IMapLayer = nvblox + voxel-hash   (CUDA, nvblox, Eigen)
│  │   include/meridian/map/
│  ├─ meridian_place/             # L5: IPlaceRecognizer                  (Eigen, small_gicp,
│  │                           #       vendored Scan Context++)
│  │
│  ├─ meridian_pipeline/          # orchestration: wires L0..L6 + threads (all layers; NO ROS)
│  │
│  ├─ meridian_msgs/              # ROS 2 .msg/.srv (telemetry, timing)   (rosidl)
│  ├─ meridian_ros/               # THIN WRAPPER: *_node, converters       (rclcpp, *_msgs,
│  │   src/odometry_node.cpp   #       message_filters, tf2, ouster-ros,
│  │   src/mapping_node.cpp    #       visualization_msgs, diagnostic_updater, rosbag2)
│  │   src/conversions/
│  │   launch/  config/  rviz/
│  │
│  └─ meridian_tools/             # offline: bag replay, evo eval, calib  (rosbag2, PCL, ouster-ros)
│
├─ vendor/                     # vendored upstreams as git SUBMODULES (§6)
│  └─ scancontext/             #   pinned SHA — loop descriptor
│
├─ dependencies.repos          # vcs: source-built deps (Sophus, GTSAM, nvblox, small_gicp, ouster-ros)
├─ colcon.meta                 # per-package CMake args (e.g. GTSAM flags propagation)
├─ colcon-mixins/              # release / debug / asan / cuda mixins (§9)
├─ .gitmodules                 # the vendor/ submodule
├─ docs/specs/                 # this file
└─ .clang-format  .clang-tidy
```

Two additions over spec 00 §2 that the build needs:

1. **`meridian_cmake`** — an `ament_cmake` package that exports shared `.cmake`
   includes. Every other package does `find_package(meridian_cmake REQUIRED)` and
   then `include(MeridianToolchain)` / `include(MeridianDeps)`, so the C++20 flags,
   CUDA arch, warning policy, and `find_package` invocations are written **once**.
   This is what stops 18 packages from each re-discovering Eigen/GTSAM/nvblox
   slightly differently.

2. **`vendor/`** with git submodules — the physical home of the vendored
   upstream (§6). It is **not** a colcon package (no `package.xml`); it is
   pulled in as a CMake `INTERFACE` target by `MeridianVendored.cmake`.

---

## 5. ament_cmake target & dependency graph

Each core package exports exactly one library target with a namespaced alias and
a clean public include dir (spec 00 §9.2). The dependency edges below are the
spec 00 §4 DAG, annotated with the *third-party* deps each node pulls. **The
load-bearing rule: core layer/X-cut packages depend on Eigen / Sophus / GTSAM /
nvblox as needed, but NOT on `rclcpp` — only `meridian_ros` does** (spec 00
§1.1, §4 R4; CI-enforced §9).

```
                third-party (system / vendored)
   Eigen3  Sophus  GTSAM  PCL  OpenCV  CUDA+nvblox  small_gicp  yaml-cpp
       │      │      │      │     │         │            │          │
 ┌─────┴──────┴──────┼──────┼─────┼─────────┼────────────┼──────────┴──────────┐
 │  meridian_cmake  (exports MeridianToolchain / MeridianDeps / MeridianVendored)            │
 └───────────────────────────────────────────────────────────────────────────────┘
       ▲ (every package find_package(meridian_cmake))
 ┌─────┴───────────────────────────────────────────────────────────────────────┐
 │ X-cut:  meridian_common(Eigen,Sophus)  meridian_config(yaml-cpp)  meridian_time        │
 │         meridian_debug   meridian_calib(Eigen,Sophus)                               │
 └─────┬───────────────────────────────────────────────────────────────────────┘
       │   (all layer packages depend on the X-cut set + meridian_cmake)
 ┌─────┴──────────┬──────────────┬───────────────┬───────────────┬──────────────┐
 │ meridian_sensors  │ meridian_preproc│ meridian_frontend│ meridian_backend │ meridian_map     │
 │ (Eigen)        │ (Eigen,      │ (Eigen,Sophus │ (GTSAM)       │ (CUDA,nvblox, │
 │                │  OpenCV)     │  — only)      │               │  Eigen)       │
 └────────────────┴──────────────┴───────┬───────┴───────┬───────┴──────┬────────┘
                                          │   meridian_place (Eigen, small_gicp, scancontext)
                                          ▼               ▼              ▼
                          ┌───────────────────────────────────────────────────┐
                          │ meridian_pipeline  (links ALL layers; NO rclcpp)      │
                          └───────────────────────────┬───────────────────────┘
                                                       ▼
                          ┌───────────────────────────────────────────────────┐
                          │ meridian_msgs (rosidl)  ◀──  meridian_ros               │
                          │ meridian_ros (rclcpp, *_msgs, message_filters, tf2,  │
                          │   visualization_msgs, diagnostic_updater,         │
                          │   rosbag2, ouster-ros)  → links meridian_pipeline    │
                          └───────────────────────────────────────────────────┘
                                                       ▲
                          ┌───────────────────────────┴───────────────────────┐
                          │ meridian_tools (rosbag2, PCL, ouster-ros, evo-scripts)│
                          └───────────────────────────────────────────────────┘
```

### 5.1 Canonical core-library `CMakeLists.txt` (non-CUDA, no ROS)

`meridian_frontend` is the leanest layer package; it is the template.

```cmake
cmake_minimum_required(VERSION 3.22)            # Humble floor
project(meridian_frontend LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)              # shared toolchain/deps
include(MeridianToolchain)                         # C++20, Release-default, PIC, warnings-as-errors
include(MeridianDeps)                              # Eigen + Sophus (no MERIDIAN_NEED_* set)

find_package(meridian_common REQUIRED)
find_package(meridian_config REQUIRED)
find_package(meridian_debug REQUIRED)
find_package(meridian_calib REQUIRED)

add_library(meridian_frontend SHARED
  src/frontend_factory.cpp
  src/lio/voxel_grid_map.cpp
  src/lio/imu_tracker.cpp
  src/lio/scan_registration.cpp
  src/lio/lio_frontend.cpp)

target_include_directories(meridian_frontend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
# Internal estimator headers live under src/ and are not exported.
target_include_directories(meridian_frontend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_features(meridian_frontend PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_frontend)         # house -Wall/-Wextra/-Wpedantic [-Werror]

target_link_libraries(meridian_frontend PUBLIC
  meridian_common::meridian_common
  meridian_config::meridian_config
  meridian_debug::meridian_debug
  meridian_calib::meridian_calib
  Eigen3::Eigen
  Sophus::Sophus)
# NO rclcpp, NO *_msgs — CI no-ROS gate (spec 00 §9.4); no Ceres/OpenCV/PCL/TBB/
# OpenMP/Threads — the estimator is Eigen+Sophus only and strictly sequential.

ament_export_targets(meridian_frontendTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  meridian_common meridian_config meridian_debug meridian_calib
  Eigen3 Sophus)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS meridian_frontend EXPORT meridian_frontendTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  include(MeridianTesting)                         # meridian_add_gtest helper
  # Tests exercise internal estimator classes, so they see src/ directly.
  meridian_add_gtest(test_voxel_grid_map test/test_voxel_grid_map.cpp)
  meridian_add_gtest(test_imu_tracker test/test_imu_tracker.cpp)
  meridian_add_gtest(test_scan_registration test/test_scan_registration.cpp)
  meridian_add_gtest(test_lio_frontend test/test_lio_frontend.cpp TIMEOUT 180)
  foreach(t test_voxel_grid_map test_imu_tracker test_scan_registration test_lio_frontend)
    target_link_libraries(${t} meridian_frontend)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  endforeach()
endif()

ament_package()
```

Key points: the package **only** names third-party math libs and lower
Meridian packages; the `cxx_std_20` is `PUBLIC` so consumers inherit it; the
heavier `MERIDIAN_NEED_*` switches (OpenCV, GTSAM, PCL, yaml-cpp) stay unset
because the estimator needs none of them; and there is no
`ament_target_dependencies(... rclcpp)` anywhere — that line existing in a core
package is exactly what the CI no-ROS gate greps for.

### 5.2 The wrapper `meridian_ros` — the only ROS-dependent package

```cmake
cmake_minimum_required(VERSION 3.22)
project(meridian_ros LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)

find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(meridian_msgs REQUIRED)
find_package(meridian_pipeline REQUIRED)          # the whole core, via one package

add_library(meridian_ros_conversions src/conversions/ros2core.cpp src/conversions/core2ros.cpp)
ament_target_dependencies(meridian_ros_conversions
  rclcpp sensor_msgs nav_msgs geometry_msgs visualization_msgs tf2 tf2_ros meridian_msgs)
target_link_libraries(meridian_ros_conversions meridian_pipeline::meridian_pipeline)

add_executable(odometry_node src/odometry_node.cpp)
ament_target_dependencies(odometry_node
  rclcpp rclcpp_components sensor_msgs nav_msgs message_filters tf2_ros diagnostic_updater meridian_msgs)
target_link_libraries(odometry_node meridian_ros_conversions meridian_pipeline::meridian_pipeline)

add_executable(mapping_node src/mapping_node.cpp)
ament_target_dependencies(mapping_node
  rclcpp visualization_msgs sensor_msgs rosbag2_cpp meridian_msgs)
target_link_libraries(mapping_node meridian_ros_conversions meridian_pipeline::meridian_pipeline)

install(TARGETS odometry_node mapping_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config rviz DESTINATION share/${PROJECT_NAME})
ament_package()
```

`meridian_ros` depends on **`meridian_pipeline`** (which transitively links every core
layer) and on `rclcpp` + the message/transform/bag packages — and that is the
*only* place those ROS packages appear in the whole graph.

### 5.3 `package.xml` discipline

- Core packages list their math deps by **rosdep key** (the lowercase name rosdep
  resolves to an apt package, *not* the CMake config name): `<depend>eigen</depend>`
  (→ `libeigen3-dev`), `<depend>sophus</depend>`, `<depend>gtsam</depend>`,
  `<depend>libopencv-dev</depend>` — plus the lower
  `meridian_*` packages and **nothing ROS-runtime**. A core `package.xml` containing
  `<depend>rclcpp</depend>` fails the CI dependency lint (spec 00 §9.4).
- `meridian_map` additionally `<depend>`s the nvblox package key and declares the
  CUDA buildtool (§7).
- `meridian_ros` is the only `package.xml` with `rclcpp`, `sensor_msgs`,
  `message_filters`, `tf2*`, `visualization_msgs`, `diagnostic_updater`,
  `rosbag2_cpp`, `ouster_ros`.
- `meridian_msgs` is `<buildtool_depend>rosidl_default_generators` +
  `<exec_depend>rosidl_default_runtime` and `<member_of_group>rosidl_interface_packages`.

---

## 6. Vendored headers (Scan Context)

One upstream is **vendored as a git submodule** under `vendor/` rather than
fetched at build time. Rationale: it is pinned by SHA (reproducible,
offline-buildable on the Orin) and small/header-dominant. `FetchContent`
is **not** used — it adds a network dependency to every clean build and is
harder to pin auditable on an air-gapped tactical box.

`.gitmodules`:

```
[submodule "vendor/scancontext"]
    path = vendor/scancontext
    url  = https://github.com/gisbi-kim/scancontext_tro.git
```

(The exact pinned SHA lives in the superproject's submodule entry / §3 row.)

`MeridianVendored.cmake` (in `meridian_cmake`, included by the packages that need it)
wraps it as an `INTERFACE` library so consumers just `target_link_libraries`.
A consuming package declares the vendored targets it links by setting the matching
`MERIDIAN_NEED_VENDOR_*` flag before `include(MeridianVendored)`; a needed-but-absent
submodule then fails the configure immediately with the fetch command, while
packages that link no vendored target still configure cleanly. `MERIDIAN_VENDOR_DIR`
defaults to `<workspace>/src/../../vendor` (i.e. `meridian_ws/vendor`) but the
caller may override it.

```cmake
# MeridianVendored.cmake (excerpt)
if(NOT DEFINED MERIDIAN_VENDOR_DIR)
  get_filename_component(MERIDIAN_VENDOR_DIR "${CMAKE_SOURCE_DIR}/../../vendor" ABSOLUTE)
endif()

# Scan Context++: the C++ module inside the upstream evaluation repo; header(s) only.
if(NOT TARGET meridian::vendor_scancontext)
  if(EXISTS "${MERIDIAN_VENDOR_DIR}/scancontext/cpp/module/Scancontext")
    add_library(meridian_vendor_scancontext INTERFACE)
    target_include_directories(meridian_vendor_scancontext INTERFACE
      "${MERIDIAN_VENDOR_DIR}/scancontext/cpp/module/Scancontext")
    target_link_libraries(meridian_vendor_scancontext INTERFACE Eigen3::Eigen)
    add_library(meridian::vendor_scancontext ALIAS meridian_vendor_scancontext)
  elseif(MERIDIAN_NEED_VENDOR_SCANCONTEXT)
    _meridian_vendor_missing("scancontext")
  endif()
endif()
```

Usage:

| Vendored upstream | Consumed by | As | Why vendored not packaged |
|---|---|---|---|
| **Scan Context++** | `meridian_place` (loop pre-filter) | `meridian::vendor_scancontext` INTERFACE | spec 07: a small self-contained descriptor; no upstream package, no version drift. |

(The vendored estimator kernels that served the retired CT front-end have
been removed; the front-end vendors nothing.)

> **Patching policy.** Vendored code is consumed *as-is* through the include
> path; any Meridian modification is a thin wrapper in `meridian_*`. If an upstream
> bug must be patched in place, the patch is a tracked commit on the submodule
> fork, and §3's pinned SHA points at the fork. We never carry an unpinned local
> edit.

---

## 7. CUDA / nvblox build setup for Jetson Orin

nvblox is the **only** map backend and the **only** CUDA in the build. There is
**no CPU fallback** — `meridian_map` requires CUDA to compile and the deployment
target always has it (spec 06 Appendix R; the simplicity mandate). A no-GPU
developer box does not get a CPU map; it simply omits the map from the build via
the compile-time `MERIDIAN_WITH_MAP` guard (§7.6), which is a build exclusion, not
a runtime alternative.

### 7.1 Platform assumptions

- **Jetson Orin, JetPack 6.x**, Ubuntu 22.04, **CUDA 12.x system toolkit**,
  driver/L4T from JetPack. `nvcc` on `PATH`, `CUDA_HOME=/usr/local/cuda`.
- Orin GPU is **compute capability 8.7** → `CMAKE_CUDA_ARCHITECTURES=87`.
- nvblox is built **from source** against the system CUDA (pinned commit in
  `dependencies.repos`). On Orin we build the nvblox core CMake library; the
  `isaac_ros_nvblox` ROS wrapper is **not** linked into the core — `meridian_map`
  talks to nvblox's C++ library directly (spec 00 R1: the core is ROS-agnostic,
  so the ROS nvblox node is not in our data path).

### 7.2 `meridian_map/CMakeLists.txt` (the CUDA core package)

```cmake
cmake_minimum_required(VERSION 3.22)
project(meridian_map NONE)                          # decide language after the guard

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)                          # also sets CUDA std/arch (below)

# Build minus map on a no-GPU dev box (spec 00 §9.5, §7.6). When OFF, this package
# contributes nothing and never touches CUDA/nvblox — it is NOT a CPU map.
if(NOT MERIDIAN_WITH_MAP)
  ament_package()
  return()
endif()

enable_language(CXX CUDA)                            # CUDA is first-class only past the guard
include(MeridianDeps)
include(MeridianVendored)

find_package(CUDAToolkit REQUIRED)               # modern FindCUDAToolkit (CMake ≥3.17)
find_package(nvblox REQUIRED)                     # nvblox's exported CMake config
find_package(meridian_common REQUIRED)
find_package(meridian_debug  REQUIRED)
find_package(meridian_calib  REQUIRED)

add_library(meridian_map
  src/voxel_hash_map.cpp           # Tier R registration (CPU)
  src/keyframe_store.cpp           # retained cloud store (CPU)
  src/nvblox_surface_map.cpp       # Tier S: wraps nvblox TSDF+colour+mesh (host side)
  src/nvblox_integrate.cu          # CUDA: cloud->depth projection feeding nvblox
  src/map_factory.cpp)

set_target_properties(meridian_map PROPERTIES
  CUDA_STANDARD 17                 # nvblox kernels are C++17
  CUDA_STANDARD_REQUIRED ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 87)           # Orin SM 8.7 (single arch)

target_compile_features(meridian_map PUBLIC cxx_std_20)   # host side is C++20

target_include_directories(meridian_map PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(meridian_map PUBLIC
  meridian_common::meridian_common meridian_debug::meridian_debug meridian_calib::meridian_calib
  Eigen3::Eigen
  nvblox::nvblox                   # TSDF + colour + Marching Cubes (GPU)
  CUDA::cudart)
# NO rclcpp — meridian_map is core.

ament_export_targets(meridian_mapTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_common meridian_debug meridian_calib Eigen3 nvblox CUDAToolkit)
install(TARGETS meridian_map EXPORT meridian_mapTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
ament_package()
```

### 7.3 What `MeridianToolchain.cmake` sets for CUDA

```cmake
# inside MeridianToolchain.cmake, guarded so non-CUDA packages ignore it
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "Jetson Orin SM 8.7")
endif()
set(CMAKE_CUDA_STANDARD 17 CACHE STRING "")
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Let host C++20 and device C++17 coexist; nvcc forwards host flags via -Xcompiler.
```

### 7.4 nvblox source build (`dependencies.repos` entry)

```yaml
repositories:
  nvblox:
    type: git
    url: https://github.com/nvidia-isaac/nvblox.git
    version: <pinned-Humble-line-commit>          # §3
```

Built either (a) as part of the colcon build if nvblox ships an `ament`/CMake
config, or (b) once into an install prefix that `CMAKE_PREFIX_PATH` then exposes
to `find_package(nvblox)`. The core CMake build (cuBLAS/Thrust only, no ROS) is
preferred so `meridian_map` links `nvblox::nvblox` without dragging the ROS wrapper.

### 7.5 The one boundary `.cu` file

The only CUDA Meridian *writes* is `nvblox_integrate.cu`: it projects a retained
`meridian::LidarPoint` cloud (body frame) at a corrected pose into the depth/colour
representation nvblox's `ProjectiveTsdfIntegrator` / `ProjectiveColorIntegrator`
consume, and drives `updateMesh()`. Everything else (TSDF fusion, ESDF if ever
enabled, Marching Cubes) is nvblox's own kernels. We do not reimplement fusion;
spec 06 fixes the runtime, this spec just compiles it.

> **No fallback, stated once.** There is no `#ifdef MERIDIAN_NO_CUDA`, no
> VDBFusion/OpenVDB path, no `IMapLayer` second implementation. `meridian_map`
> fails to configure without CUDA + nvblox, and that is correct for the Jetson
> Orin target.

### 7.6 The `MERIDIAN_WITH_MAP` compile-time guard (build minus map, never a CPU map)

A developer box without CUDA (a plain x86 laptop, an Apple-Silicon dev machine) must still be able to build and unit-test the non-map layers — the front-end, back-end, place recognition, sensors, and all cross-cutting libraries. This is a **build-configuration exclusion**, not a runtime fallback: the switch removes `meridian_map` from the build graph entirely; it never substitutes a CPU map.

```cmake
# MeridianToolchain.cmake — workspace-wide switch, default ON (full GPU deployment build).
option(MERIDIAN_WITH_MAP "Build the GPU map layer (meridian_map, nvblox/CUDA)" ON)
```

Rules:

- **`MERIDIAN_WITH_MAP=ON` (default, the only deployment configuration).** `meridian_map` is built exactly as §7.2 specifies; CUDA + nvblox are required and the build hard-fails without them. Every deployed binary is built this way.
- **`MERIDIAN_WITH_MAP=OFF` (development / CI of non-map layers only).** `meridian_map` is skipped (its `CMakeLists.txt` early-returns before `enable_language(... CUDA)`, so no CUDA toolkit is required and `find_package(nvblox)` is never reached — see the guard at the top of the §7.2 sketch). `meridian_pipeline` and `meridian_ros`, which link the map, compile the map-facing wiring out behind the same guard: the pipeline constructs the map stage (T4/T5 in spec 00 §11) and the `Q_map` queue **only** when `MERIDIAN_WITH_MAP` is defined, and exposes no `IMapLayer` otherwise. The resulting binary has **no map at all** — it is not a degraded map, not a CPU map, not a runtime-selectable mode.
- **No runtime branch and no second `IMapLayer`.** The guard is preprocessor/CMake only. There is never an `#ifdef`-selected CPU integrator, never a `map.backend` value other than `nvblox`, and never a path where a `MERIDIAN_WITH_MAP=ON` binary degrades to CPU on a missing GPU — it fail-fasts (spec 00 §9.5). The guard's sole purpose is letting a no-GPU box build the rest of the system; it preserves the "nvblox GPU-only, no CPU fallback" invariant rather than weakening it.
- **CI coverage.** CI builds the workspace **both** ways: the on-device/x86-GPU image builds with `MERIDIAN_WITH_MAP=ON` (the full system, §9.3); a no-CUDA x86 image builds with `MERIDIAN_WITH_MAP=OFF` and runs the non-map unit/replay tests, so the dev-box build path stays green.

---

## 8. Finding & exporting each dependency in CMake

`MeridianDeps.cmake` centralises every `find_package` so versions/targets are
discovered identically everywhere. Sketch:

Eigen and Sophus are *universal* (every core package compiles against them) and so
are found unconditionally; every heavier dep is gated by a `MERIDIAN_NEED_*` switch
the consuming package sets **before** `include(MeridianDeps)`, so a package only
discovers (and only pays the configure cost of) what it actually links.

```cmake
# MeridianDeps.cmake  (included after MeridianToolchain)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)        # → Eigen3::Eigen  (universal)
find_package(Sophus REQUIRED)                       # → Sophus::Sophus (universal, header-only)

if(MERIDIAN_NEED_OPENCV)
  find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc video calib3d)  # → ${OpenCV_LIBS}
endif()
if(MERIDIAN_NEED_GTSAM)
  find_package(GTSAM 4.2 REQUIRED)                  # → gtsam (exports its own target)
endif()
if(MERIDIAN_NEED_PCL)
  find_package(PCL 1.12 REQUIRED COMPONENTS common io filters)
endif()
if(MERIDIAN_NEED_YAMLCPP)
  find_package(yaml-cpp REQUIRED)                   # → yaml-cpp (config loader)
endif()
```

There is **no `MERIDIAN_NEED_CERES`**: nothing links Ceres any more (§3) and a
switch with no consumer is a trap, so it was removed with the CT window solver.

| Dep | `find_package` | Imported target | Notes |
|---|---|---|---|
| Eigen | `find_package(Eigen3 3.4 REQUIRED NO_MODULE)` | `Eigen3::Eigen` | `NO_MODULE` to take Eigen's own config; one Eigen for the whole tree (GTSAM built `GTSAM_USE_SYSTEM_EIGEN=ON`). |
| Sophus | `find_package(Sophus REQUIRED)` | `Sophus::Sophus` | Header-only; depends on Eigen. |
| GTSAM | `find_package(GTSAM 4.2 REQUIRED)` | `gtsam` | Only `meridian_backend` sets `MERIDIAN_NEED_GTSAM`. |
| nvblox | `find_package(nvblox REQUIRED)` | `nvblox::nvblox` | Only `meridian_map`; requires `CUDAToolkit`. |
| CUDA | `find_package(CUDAToolkit REQUIRED)` | `CUDA::cudart`, `CUDA::cublas` | Only `meridian_map`. |
| PCL | `find_package(PCL 1.12 ... )` | `${PCL_LIBRARIES}` (or `pcl_*` targets) | `MERIDIAN_NEED_PCL`: `meridian_tools` (io/filters at the offline edges). |
| OpenCV | `find_package(OpenCV 4 ...)` | `${OpenCV_LIBS}` | `MERIDIAN_NEED_OPENCV`: `meridian_preprocess` (image conditioning), `meridian_tools`. |
| small_gicp | `find_package(small_gicp REQUIRED)` or vendored | `small_gicp::small_gicp` | Only `meridian_place`. |
| vendored | (no find_package) | `meridian::vendor_*` | via `MeridianVendored.cmake` (§6). |

Every package that produces a library calls `ament_export_dependencies(...)`
listing the deps a downstream consumer needs to re-find, so
`find_package(meridian_frontend)` transitively pulls Eigen/Sophus
target definitions. This is what lets `meridian_ros` depend on the whole core via a
single `find_package(meridian_pipeline)`.

---

## 9. colcon invocation, mixins, and CI

### 9.1 Mixins (`colcon-mixins/meridian.mixin`)

```yaml
release:   { cmake-args: ["-DCMAKE_BUILD_TYPE=Release"] }
debug:     { cmake-args: ["-DCMAKE_BUILD_TYPE=RelWithDebInfo"] }
asan:      { cmake-args: ["-DCMAKE_BUILD_TYPE=Debug",
                          "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined"] }  # host only
orin:      { cmake-args: ["-DCMAKE_BUILD_TYPE=Release",
                          "-DCMAKE_CUDA_ARCHITECTURES=87"] }
```

### 9.2 The build commands

```bash
# one-time: pull source deps and vendored submodules
cd meridian_ws
vcs import src < dependencies.repos          # Sophus, GTSAM, nvblox, small_gicp, ouster-ros
git submodule update --init --recursive      # vendor/scancontext
rosdep install --from-paths src --ignore-src -y   # apt deps (Eigen, PCL, OpenCV, ROS pkgs)

# build (Orin)
source /opt/ros/humble/setup.bash
colcon build --mixin orin --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# test
colcon test --mixin debug && colcon test-result --verbose
```

`--symlink-install` for fast iterate; `compile_commands.json` for clang-tidy and
the LSP. On the x86 dev box, swap `--mixin orin` for `--mixin release` (and set
`CMAKE_CUDA_ARCHITECTURES=89;87` if an RTX card is used for offline runs).

### 9.3 CI gates (the architecture-enforcing ones, spec 00 §9.4)

1. **No-ROS gate.** Grep `src/meridian_*` *excluding* `meridian_ros` and `meridian_msgs`
   for `rclcpp|ros/ros.h|sensor_msgs|rclcpp/clock` → must be empty.
2. **Dependency lint.** Parse every `package.xml`/`CMakeLists.txt`; assert the §5
   edge set; fail if a core `package.xml` lists `rclcpp`/`*_msgs`, or a layer
   `#include`s a sibling's `src/`.
3. **No-CUDA-outside-map gate.** Only `meridian_map` may declare the `CUDA`
   language; any other package enabling CUDA fails (keeps GPU fenced).
4. **Sequential-front-end gate.** Grep `meridian_frontend/src/lio/` for
   `omp|tbb|std::thread|std::async` → must be empty; the estimator is
   single-threaded by contract (spec 00 §11.2) so two identical replays stay
   bit-identical.
5. **No-grounding-in-code gate.** Grep `src/meridian_*` source (`*.hpp`/`*.cpp`/`*.cu`,
   comments included) for the regex `grounding[ /][0-9]` → must be empty. Code comments
   must read self-contained, with no pointer into the reference dossiers; the character
   class `[ /]` catches both the slash form (`grounding/07`) and the space-separated
   form (`grounding 07`) — the earlier slash-only pattern missed the latter.
6. **clang-tidy / clang-format** on every core TU; `-Werror` in core.
7. **Build + unit + replay tests** under `colcon test` (GoogleTest); the replay
   test drives `meridian_pipeline` from a bag via `meridian_tools` with no ROS spinning
   (proves the off-ROS path). The non-map build (`MERIDIAN_WITH_MAP=OFF`, §7.6) runs
   the non-map unit/replay tests so the no-GPU dev-box path stays green.
8. **Reproducible, dual base images.** CI runs in **two** tagged Docker images that
   snapshot the §3 apt versions so apt deps never float: an **Orin/JetPack** image
   (`nvcr.io/nvidia/l4t-jetpack`-based, CUDA 12.x, `CMAKE_CUDA_ARCHITECTURES=87`,
   builds `MERIDIAN_WITH_MAP=ON`) and an **x86 dev** image (`ros:humble`-based, builds
   `MERIDIAN_WITH_MAP=OFF`, or `ON` with `CMAKE_CUDA_ARCHITECTURES=89;87` when a CUDA
   runner is available). Both images derive their dependency stack from a single
   `docker/install-deps.sh`, so the Orin and x86 toolchains can never drift; the image
   tags are the build's apt/CUDA pin (§3 pin discipline).

---

## 10. Standing up the workspace from scratch (the runbook)

A real engineer, fresh Jetson Orin with JetPack 6, follows this exactly:

1. **Host prep.** Install ROS 2 Humble (`ros-humble-ros-base` + dev tools),
   `python3-colcon-common-extensions`, `python3-vcstool`, `python3-rosdep`.
   Confirm JetPack CUDA 12.x: `nvcc --version`, `echo $CUDA_HOME`.
2. **Time sync (linuxptp).** `apt install linuxptp`; configure `ptp4l` on the
   sensor NIC and `phc2sys` to discipline the system clock (spec 02 owns the
   config). This is host setup, not a build step, but the sensors won't be
   correctly stamped without it.
3. **Clone + submodules.**
   `git clone <meridian> meridian_ws && cd meridian_ws && git submodule update --init --recursive`
   (pulls `vendor/scancontext` at its pinned SHA).
4. **Source deps.** `vcs import src < dependencies.repos` (Sophus, GTSAM 4.2,
   nvblox, small_gicp, ouster-ros at pinned tags/commits).
5. **System deps.** `rosdep install --from-paths src --ignore-src -y` (Eigen 3.4,
   PCL 1.12, OpenCV 4, yaml-cpp, the ROS message/tf/bag packages).
6. **Build GTSAM** (if not already an install prefix): CMake with
   `GTSAM_USE_SYSTEM_EIGEN=ON`, `GTSAM_BUILD_WITH_MARCH_NATIVE=OFF`,
   `GTSAM_WITH_TBB=OFF`; install to a prefix on `CMAKE_PREFIX_PATH`. (Or let
   colcon build it from the `src/` import.)
7. **Build nvblox** against system CUDA (pinned commit), install to a prefix
   `find_package(nvblox)` can see. Verify `CMAKE_CUDA_ARCHITECTURES=87`.
8. **Build Meridian.** `colcon build --mixin orin --symlink-install`. The build
   order colcon resolves is: `meridian_cmake` → X-cut (`meridian_common`,
   `meridian_config`, `meridian_time`, `meridian_debug`, `meridian_calib`) → layers
   (`meridian_sensors`, `meridian_preprocess`, `meridian_frontend`, `meridian_backend`,
   `meridian_map`, `meridian_place`) → `meridian_pipeline` → `meridian_msgs` → `meridian_ros` →
   `meridian_tools`.
9. **Test.** `colcon test --mixin debug && colcon test-result --verbose`.
10. **Run.** `source install/setup.bash`; launch `meridian_ros odometry_node` +
    `mapping_node` with `config/*.yaml`, or replay a bag through
    `meridian_tools replay`.

If steps 1–10 succeed, the LIO estimator builds, the
nvblox GPU map links, and the off-ROS replay path runs — which is the bar this
spec must clear.

> **No-GPU dev box.** On an x86 laptop or Apple-Silicon machine without CUDA,
> skip steps 6's nvblox build and 7, and build with `MERIDIAN_WITH_MAP=OFF`
> (§7.6): `colcon build --mixin release --cmake-args -DMERIDIAN_WITH_MAP=OFF`.
> This compiles and tests every layer except `meridian_map` (the front-end,
> back-end, place recognition, sensors, all cross-cutting). It is a development
> convenience only — the deployed Orin build is always `MERIDIAN_WITH_MAP=ON`
> with the full GPU map; there is never a runtime CPU map path (spec 00 §9.5).
> Both configurations are produced from the same pinned Docker bases (§9.3 gate 8),
> whose dependency stack comes from a single `docker/install-deps.sh` so the Orin
> and x86 toolchains cannot drift.

---

## 11. Considered & rejected

The design commits to exactly one library per job (§1). For the load-bearing
choices, here is the one-line reason the alternative lost. This section is the
*only* place alternatives appear; nothing in the running design hedges.

| Job | Rejected option | Why rejected |
|---|---|---|
| **Dense mapping** | **VDBFusion / OpenVDB / NanoVDB (CPU, km-scale)** | The deployment target is a CUDA Jetson Orin where nvblox does GPU TSDF + colour + Marching Cubes in one library; a CPU OpenVDB path is a second, untested map backend that the simplicity mandate forbids. We do **not** ship it even as a fallback. (Earlier drafts floated VDBFusion as a "CPU/km-scale fallback" and a NanoVDB "Tier G archive" — both are dropped here; if extreme-extent outdoor mapping is ever needed it is a *future* `IMapLayer` implementation, not a current dual path.) |
| **Dense mapping** | **Voxblox (CPU)** | Ageing CPU codebase; nvblox supersedes it on GPU with the same block-hash semantics. Kept only as the ESDF *algorithm* reference (spec 06 Appendix R), never linked. |
| Back-end optimiser | **g2o** | GTSAM ships `ISAM2` (true incremental Bayes tree), `GncOptimizer`, and `CombinedImuFactor` out of the box (spec 05 Appendix R); g2o would mean re-implementing incremental smoothing and GNC. |
| Back-end optimiser | **Ceres as the global back-end** | Ceres is batch; the global graph needs *incremental* iSAM2. |
| Per-sweep solver | **Ceres / GTSAM for the registration solve** | The GN normal equations are a closed-form 6×6 LDLT; a general NLLS framework adds link weight, allocation churn, and (with internal threading) nondeterminism for zero benefit at this problem size. |
| NN / registration | **nanoflann / PCL KdTree as primary** | The maps need incremental insert + range-clip near the robot; a static k-d tree forces rebuilds. The voxel-hash gives incremental behaviour, O(1) insert, and a fixed-order neighbour probe that keeps the solve deterministic. |
| GICP | **PCL GICP** | Drags the full PCL registration stack; small_gicp is faster, multi-threaded, and takes Eigen buffers directly — matching the L5 verify need without the PCL weight in the hot path. |
| Lie groups | **manif / hand-rolled** | Sophus is the convention the codebase and the references share; using anything else adds a conversion seam at every boundary. |
| ROS distro | **Iron / Jazzy** | Not the Jetson/JetPack-aligned LTS; Humble is the supported on-device distro. |
| Front-end parallelism | **OpenMP/TBB in the estimator** | Parallel reductions sum in nondeterministic order, breaking the bit-identical replay guarantee; the per-sweep GN fits the budget single-threaded, so the estimator stays strictly sequential (CI-enforced, §9.3). |
| Single vs multi-LiDAR | **multi-LiDAR / dome-Ouster merge** | The system is single LiDAR + single IMU + single camera + GNSS; multi-LiDAR is at most a future extension behind the same `ISensorSource`/`IFrontEnd` interfaces, not designed or built now. |

---

*End of spec 11 — build system & library choices.*
