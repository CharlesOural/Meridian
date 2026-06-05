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
> complete system**: a continuous-time (CT), tightly-coupled
> LiDAR-Inertial-Visual-GNSS estimator. The L2 front-end is a **CT cubic
> B-spline sliding-window** solve (direct LiDAR point-to-plane at each point's
> true time + FAST-LIVO2-style sparse-direct photometric visual with LiDAR-depth
> and exposure compensation + direct-derivative IMU residuals + GNSS), all behind
> the `IFrontEnd` interface (spec 01 §7.3). A FAST-LIO2-style iEKF MAY exist
> behind that same interface purely as a reference/baseline and test oracle; it is
> not a milestone and the build does not organise around it. The deployment
> target is an **NVIDIA Jetson Orin with CUDA always present**; mapping is
> **nvblox, GPU-only, no CPU fallback**.
>
> **Companion specs.** `00_architecture.md` (package layout §2, dependency rules
> §4, build system §9 — this spec is the full expansion of §9), `01_interfaces_
> and_data_types.md` (the core value types the math packages compile against),
> `06_mapping.md` (L4 / nvblox runtime), `04_frontend_estimation.md` (CT spline
> runtime), `05_backend_graph.md` (GTSAM/iSAM2 runtime). Grounding:
> `docs/grounding/07_mapping_tsdf_mesh.md`, `docs/grounding/09_backend_isam2.md`,
> `docs/grounding/10_continuous_time.md`.

---

## Table of contents

1. [Library canon — one chosen library per job](#1-library-canon--one-chosen-library-per-job)
2. [Toolchain & language baseline](#2-toolchain--language-baseline)
3. [Version pin table](#3-version-pin-table)
4. [colcon workspace layout](#4-colcon-workspace-layout)
5. [ament_cmake target & dependency graph](#5-ament_cmake-target--dependency-graph)
6. [Vendored headers (basalt-headers, ikd-Tree, Scan Context)](#6-vendored-headers-basalt-headers-ikd-tree-scan-context)
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
| Linear algebra (vectors, matrices, dense solves) | **Eigen 3.4** | The de-facto C++ numerical core; every other library here (Sophus, GTSAM, basalt-headers, Ceres, PCL, OpenCV) already speaks `Eigen::`, so it is the lingua franca with zero glue. | Blaze, Armadillo |
| Lie groups (SO(3)/SE(3) exp/log, manifolds) | **Sophus** | Header-only `SO3`/`SE3` on Eigen, matching the box-plus/box-minus and right-perturbation convention spec 01 §3.1 mandates; it is exactly what basalt-headers' splines store as control points. | hand-rolled so3_math, manif |
| Back-end optimiser (incremental factor graph) | **GTSAM 4.2** | Ships `ISAM2` (the incremental Bayes-tree smoother spec 05 needs), `CombinedImuFactor`, `noiseModel::Robust`+`Huber`, and `GncOptimizer` — the exact factor/robustness set grounding 09 specifies, in one BSD library. | g2o, Ceres-only back-end, SE-Sync |
| CT spline kernel (cubic split SO(3)×ℝ³, analytic Jacobians) | **basalt-headers (vendored)** | The canonical, battle-tested implementation of the Sommer-et-al. O(k) cumulative-spline derivatives *and analytic Jacobians w.r.t. control points* — the hard, error-prone part — header-only on Eigen+Sophus; it is what CLINS/Coco-LIC reuse (grounding 10 §3, §11). | hand-derived spline, Kontiki |
| Sliding-window NLLS solver + Schur marginalisation (CT window) | **Ceres 2.x** | The solver all three CT references (CLINS, Coco-LIC, basalt) use: LM, analytic cost functions, `Manifold`/`LocalParameterization` for SO(3), and built-in Schur complement for the boundary-control-point marginalisation prior (grounding 10 §6, §11). GTSAM stays the *global* back-end; Ceres is the *front-end window* solver. | GTSAM for the window, custom LM |
| Dense mapping: TSDF + colour + mesh | **nvblox (isaac_ros_nvblox)** | GPU TSDF + GPU colour fusion + GPU Marching Cubes in one CUDA library with a maintained ROS 2 wrapper; on the guaranteed-CUDA Jetson Orin it is the only map backend (grounding 07 §0, §10). **No CPU fallback.** | VDBFusion, Voxblox, OpenVDB/NanoVDB |
| Nearest-neighbour map for registration | **adaptive voxel-hash (Meridian, in `meridian_map`)** backed by **ikd-Tree (vendored) as the reference/oracle** | Spec 06 §3 replaces ikd-Tree's *layout* with an adaptive voxel hash but keeps its incremental-insert / box-delete / k-NN *behaviour*; the vendored ikd-Tree header is retained as the correctness oracle the voxel-hash is tested against. | nanoflann, PCL KdTree, raw ikd-Tree as primary |
| Fine registration (loop-closure GICP verify) | **small_gicp** | Header-light, multi-threaded GICP/VGICP that takes Eigen point buffers directly — the L5 verify step (spec 07) wants a fast, dependency-thin GICP, not full PCL registration. | PCL GICP, libpointmatcher |
| Point-cloud I/O & filters | **PCL** (io/filters only, sparingly) | Used *only* for bag/file I/O and a couple of filters at the edges; the hot path computes on `meridian::LidarPoint` buffers, never `pcl::PointCloud` (spec 01 §1 R1). PCL is a heavy dependency, so it is fenced to `meridian_preprocess`/`meridian_tools`. | PCL everywhere (rejected by R1) |
| Images (pyramids, KLT, undistort, Bayer) | **OpenCV 4** | The CT front-end's sparse-direct visual track (LiDAR-depth reprojection, KLT optical flow, image pyramids, exposure handling) needs exactly OpenCV's `imgproc`/`video`/`calib3d`; it is also what FAST-LIVO2 and Coco-LIC use. | bespoke image code |
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
| Warnings | `-Wall -Wextra -Wpedantic`, warnings-as-errors in **core** packages | spec 00 §9.4; the wrapper and third-party-heavy edges (PCL/OpenCV includes) relax `-Werror` selectively. |
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
| GTSAM | **4.2.0** | source build (CMake), `GTSAM_USE_SYSTEM_EIGEN=ON` | Build with `GTSAM_BUILD_WITH_MARCH_NATIVE=OFF` (reproducible Orin binary), `GTSAM_USE_SYSTEM_EIGEN=ON` (one Eigen, §8), `GTSAM_WITH_TBB=OFF` unless TBB is also pinned. Provides ISAM2 / CombinedImuFactor / Robust+Huber / GncOptimizer (grounding 09). |
| Ceres | **2.1.0** (≥2.0; 2.1 for the `Manifold` API) | apt `libceres-dev` (2.1 on jammy) or source | CT-window solver (grounding 10 §11). Pulls SuiteSparse/glog/gflags via apt. |
| nvblox | **isaac_ros_nvblox, Humble release line** (pin the commit in `dependencies.repos`) | vcs source build (CUDA) | The only map backend; §7. Built against system CUDA 12.x. |
| CUDA | **12.x** (JetPack 6 system CUDA) | JetPack / apt | `nvcc`, cuBLAS, Thrust — all nvblox needs. |
| PCL | **1.12.x** (jammy) | apt `libpcl-dev` | io/filters only; fenced to `meridian_preprocess`/`meridian_tools`. |
| OpenCV | **4.5.x** (jammy) | apt `libopencv-dev` | imgproc/video/calib3d for the visual track. On Orin, the JetPack OpenCV (CUDA-enabled) is acceptable but Meridian uses only CPU OpenCV calls — do not depend on CUDA-OpenCV. |
| small_gicp | pinned commit | vcs source (header + small lib) | GICP verify (spec 07). |
| ouster-ros | **Humble branch**, pinned commit | vcs source | LiDAR driver; only `meridian_ros`/`meridian_tools` depend on it. |
| linuxptp | distro `linuxptp` (apt) | apt | Host PTP daemons; runtime, not linked. |
| GoogleTest | the `ament_cmake_gtest` / `gtest_vendor` that ships with Humble | ament | Test framework. |
| evo | latest `pip` `evo` | pip (offline tooling) | Not a build dependency; used by spec 10's harness scripts. |
| **basalt-headers** | pinned commit, **vendored** | git submodule (§6) | CT spline kernel. |
| **ikd-Tree** | pinned commit, **vendored** | git submodule (§6) | Registration oracle. |
| **Scan Context++** | pinned commit/file, **vendored** | git submodule / copied header (§6) | Loop descriptor. |

> **Pin discipline.** Source-built deps (Sophus, GTSAM, nvblox, Ceres-if-source,
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
│  │   cmake/MeridianVendored.cmake      #   INTERFACE targets for basalt/ikd/scancontext
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
│  ├─ meridian_preprocess/        # L1: deskew/downsample/validity        (Eigen, PCL[io/filters])
│  ├─ meridian_frontend/          # L2: IFrontEnd = CT B-spline LIVO+GNSS (Eigen, Sophus, Ceres,
│  │   include/meridian/frontend/ #       basalt-headers, OpenCV, ikd-Tree-oracle)
│  │   src/ct/                 #       THE front-end: CT spline window solve
│  │   src/iekf/               #       reference/oracle iEKF (baseline only)
│  │   src/frontend_factory.cpp
│  │
│  ├─ meridian_backend/           # L3: IBackEnd = GTSAM iSAM2            (GTSAM)
│  ├─ meridian_map/               # L4: IMapLayer = nvblox + voxel-hash   (CUDA, nvblox, Eigen,
│  │   include/meridian/map/      #       ikd-Tree-oracle)
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
│  ├─ basalt-headers/          #   pinned SHA — CT spline kernel
│  ├─ ikd-Tree/                #   pinned SHA — registration oracle
│  └─ scancontext/             #   pinned SHA — loop descriptor
│
├─ dependencies.repos          # vcs: source-built deps (Sophus, GTSAM, nvblox, small_gicp, ouster-ros)
├─ colcon.meta                 # per-package CMake args (e.g. GTSAM flags propagation)
├─ colcon-mixins/              # release / debug / asan / cuda mixins (§9)
├─ .gitmodules                 # the three vendor/ submodules
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

2. **`vendor/`** with git submodules — the physical home of the three vendored
   upstreams (§6). They are **not** colcon packages (no `package.xml`); they are
   pulled in as CMake `INTERFACE` targets by `MeridianVendored.cmake`.

---

## 5. ament_cmake target & dependency graph

Each core package exports exactly one library target with a namespaced alias and
a clean public include dir (spec 00 §9.2). The dependency edges below are the
spec 00 §4 DAG, annotated with the *third-party* deps each node pulls. **The
load-bearing rule: core layer/X-cut packages depend on Eigen / Sophus / GTSAM /
Ceres / nvblox as needed, but NOT on `rclcpp` — only `meridian_ros` does** (spec 00
§1.1, §4 R4; CI-enforced §9).

```
                third-party (system / vendored)
   Eigen3  Sophus  GTSAM  Ceres  PCL  OpenCV  CUDA+nvblox  small_gicp  yaml-cpp
       │      │      │      │      │     │         │            │          │
 ┌─────┴──────┴──────┼──────┼──────┼─────┼─────────┼────────────┼──────────┴─────┐
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
 │ (Eigen)        │ (Eigen,PCL)  │ (Eigen,Sophus,│ (GTSAM)       │ (CUDA,nvblox, │
 │                │              │  Ceres,basalt,│               │  Eigen,       │
 │                │              │  OpenCV,ikd)  │               │  ikd-oracle)  │
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

`meridian_frontend` is the richest non-CUDA core package; it is the template.

```cmake
cmake_minimum_required(VERSION 3.22)            # Humble floor; CUDA-arch list needs ≥3.18
project(meridian_frontend LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)              # shared toolchain/deps
include(MeridianToolchain)                         # C++20, warnings-as-errors
include(MeridianDeps)                              # find Eigen/Sophus/Ceres/OpenCV/GTSAM...
include(MeridianVendored)                          # basalt-headers, ikd-Tree INTERFACE targets

find_package(meridian_common REQUIRED)
find_package(meridian_config  REQUIRED)
find_package(meridian_debug   REQUIRED)
find_package(meridian_calib   REQUIRED)

add_library(meridian_frontend
  src/ct/ct_frontend.cpp
  src/ct/ct_spline_window.cpp
  src/ct/ct_lidar_factor.cpp
  src/ct/ct_imu_factor.cpp
  src/ct/ct_visual_factor.cpp
  src/ct/ct_gnss_factor.cpp
  src/iekf/iekf_frontend.cpp        # reference/oracle baseline only
  src/frontend_factory.cpp)

target_include_directories(meridian_frontend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(meridian_frontend PUBLIC cxx_std_20)

target_link_libraries(meridian_frontend PUBLIC
  meridian_common::meridian_common meridian_config::meridian_config meridian_debug::meridian_debug meridian_calib::meridian_calib
  Eigen3::Eigen Sophus::Sophus
  Ceres::ceres
  ${OpenCV_LIBS}
  meridian::vendor_basalt          # INTERFACE: basalt-headers include dir (§6)
  meridian::vendor_ikdtree)        # INTERFACE: ikd-Tree oracle (test/registration)
# NO rclcpp, NO *_msgs here — CI no-ROS gate (spec 00 §9.4) enforces this.

ament_export_targets(meridian_frontendTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_common meridian_config meridian_debug meridian_calib
                          Eigen3 Sophus Ceres OpenCV)
install(TARGETS meridian_frontend EXPORT meridian_frontendTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_ct_spline test/test_ct_spline.cpp)
  target_link_libraries(test_ct_spline meridian_frontend)
endif()

ament_package()
```

Key points: the package **only** names third-party math libs and lower
Meridian packages; the `cxx_std_20` is `PUBLIC` so consumers inherit it; the vendored
headers come in as `meridian::vendor_*` INTERFACE targets (§6); and there is no
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

- Core packages list their math deps as `<depend>Eigen3</depend>`,
  `<depend>sophus</depend>`, `<depend>gtsam</depend>` (the keys that resolve via
  rosdep / `find_package`), the lower `meridian_*` packages, and **nothing
  ROS-runtime**. A core `package.xml` containing `<depend>rclcpp</depend>` fails
  the CI dependency lint (spec 00 §9.4).
- `meridian_map` additionally `<depend>`s the nvblox package key and declares the
  CUDA buildtool (§7).
- `meridian_ros` is the only `package.xml` with `rclcpp`, `sensor_msgs`,
  `message_filters`, `tf2*`, `visualization_msgs`, `diagnostic_updater`,
  `rosbag2_cpp`, `ouster_ros`.
- `meridian_msgs` is `<buildtool_depend>rosidl_default_generators` +
  `<exec_depend>rosidl_default_runtime` and `<member_of_group>rosidl_interface_packages`.

---

## 6. Vendored headers (basalt-headers, ikd-Tree, Scan Context)

Three upstreams are **vendored as git submodules** under `vendor/` rather than
fetched at build time. Rationale: they are pinned by SHA (reproducible,
offline-buildable on the Orin), they are small/header-dominant, and (basalt,
ikd-Tree) they are reference code we may need to read and patch. `FetchContent`
is **not** used — it adds a network dependency to every clean build and is
harder to pin auditable on an air-gapped tactical box.

`.gitmodules`:

```
[submodule "vendor/basalt-headers"]
    path = vendor/basalt-headers
    url  = https://gitlab.com/VladyslavUsenko/basalt-headers.git
[submodule "vendor/ikd-Tree"]
    path = vendor/ikd-Tree
    url  = https://github.com/hku-mars/ikd-Tree.git
[submodule "vendor/scancontext"]
    path = vendor/scancontext
    url  = https://github.com/gisbi-kim/scancontext_tro.git
```

(Exact pinned SHAs live in the superproject's submodule entries / §3 row.)

`MeridianVendored.cmake` (in `meridian_cmake`, included by the packages that need them)
wraps each as an `INTERFACE` library so consumers just `target_link_libraries`:

```cmake
# MeridianVendored.cmake
set(MERIDIAN_VENDOR_DIR "${CMAKE_CURRENT_LIST_DIR}/../../vendor")  # resolves to meridian_ws/vendor

# basalt-headers: header-only, needs Eigen + Sophus on the include path.
if(NOT TARGET meridian::vendor_basalt)
  add_library(meridian_vendor_basalt INTERFACE)
  target_include_directories(meridian_vendor_basalt INTERFACE
    "${MERIDIAN_VENDOR_DIR}/basalt-headers/include")
  target_link_libraries(meridian_vendor_basalt INTERFACE Eigen3::Eigen Sophus::Sophus)
  add_library(meridian::vendor_basalt ALIAS meridian_vendor_basalt)
endif()

# ikd-Tree: small header + one .cpp; build as a tiny static lib (oracle/registration).
if(NOT TARGET meridian::vendor_ikdtree)
  add_library(meridian_vendor_ikdtree STATIC
    "${MERIDIAN_VENDOR_DIR}/ikd-Tree/ikd_Tree.cpp")
  target_include_directories(meridian_vendor_ikdtree PUBLIC
    "${MERIDIAN_VENDOR_DIR}/ikd-Tree")
  target_link_libraries(meridian_vendor_ikdtree PUBLIC Eigen3::Eigen)
  set_target_properties(meridian_vendor_ikdtree PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_library(meridian::vendor_ikdtree ALIAS meridian_vendor_ikdtree)
endif()

# Scan Context++: header(s) only.
if(NOT TARGET meridian::vendor_scancontext)
  add_library(meridian_vendor_scancontext INTERFACE)
  target_include_directories(meridian_vendor_scancontext INTERFACE
    "${MERIDIAN_VENDOR_DIR}/scancontext")
  target_link_libraries(meridian_vendor_scancontext INTERFACE Eigen3::Eigen)
  add_library(meridian::vendor_scancontext ALIAS meridian_vendor_scancontext)
endif()
```

Usage:

| Vendored upstream | Consumed by | As | Why vendored not packaged |
|---|---|---|---|
| **basalt-headers** | `meridian_frontend` (CT spline kernel) | `meridian::vendor_basalt` INTERFACE (header-only, +Eigen/Sophus) | grounding 10 §11: it is the canonical analytic-Jacobian spline; we may wrap it in virtual-time for adaptive knots, so we want the source pinned and patchable. |
| **ikd-Tree** | `meridian_frontend`/`meridian_map` (registration oracle) | `meridian::vendor_ikdtree` tiny STATIC lib | spec 06 §3: we keep its *behaviour* as the correctness oracle for the adaptive voxel-hash; reference code, read-and-test. |
| **Scan Context++** | `meridian_place` (loop pre-filter) | `meridian::vendor_scancontext` INTERFACE | spec 07: a small self-contained descriptor; no upstream package, no version drift. |

> **Patching policy.** Vendored code is consumed *as-is* through the include
> path; any Meridian modification is a thin wrapper in `meridian_*` (e.g. the
> virtual-time adaptive-knot shim around basalt lives in
> `meridian_frontend/src/ct/`, not inside `vendor/basalt-headers/`). If an upstream
> bug must be patched in place, the patch is a tracked commit on the submodule
> fork, and §3's pinned SHA points at the fork. We never carry an unpinned local
> edit.

---

## 7. CUDA / nvblox build setup for Jetson Orin

nvblox is the **only** map backend and the **only** CUDA in the build. There is
**no CPU fallback** — `meridian_map` requires CUDA to compile and the deployment
target always has it (grounding 07 §0, §10; the simplicity mandate).

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
project(meridian_map LANGUAGES CXX CUDA)           # CUDA is a first-class language here

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)                          # also sets CUDA std/arch (below)
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
  CUDA::cudart
  meridian::vendor_ikdtree)           # registration oracle
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

---

## 8. Finding & exporting each dependency in CMake

`MeridianDeps.cmake` centralises every `find_package` so versions/targets are
discovered identically everywhere. Sketch:

```cmake
# MeridianDeps.cmake  (included after MeridianToolchain)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)        # → Eigen3::Eigen
find_package(Sophus REQUIRED)                       # → Sophus::Sophus (header-only)
find_package(Ceres 2.1 REQUIRED)                    # → Ceres::ceres
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc video calib3d)  # → ${OpenCV_LIBS}
# GTSAM only where the back-end needs it; guarded so non-backend pkgs skip it:
if(MERIDIAN_NEED_GTSAM)
  find_package(GTSAM 4.2 REQUIRED)                  # → gtsam (exports its own target)
endif()
# PCL only where preprocess/tools need it (io + filters components only):
if(MERIDIAN_NEED_PCL)
  find_package(PCL 1.12 REQUIRED COMPONENTS common io filters)
endif()
```

| Dep | `find_package` | Imported target | Notes |
|---|---|---|---|
| Eigen | `find_package(Eigen3 3.4 REQUIRED NO_MODULE)` | `Eigen3::Eigen` | `NO_MODULE` to take Eigen's own config; one Eigen for the whole tree (GTSAM built `GTSAM_USE_SYSTEM_EIGEN=ON`). |
| Sophus | `find_package(Sophus REQUIRED)` | `Sophus::Sophus` | Header-only; depends on Eigen. |
| GTSAM | `find_package(GTSAM 4.2 REQUIRED)` | `gtsam` | Only `meridian_backend` sets `MERIDIAN_NEED_GTSAM`. |
| Ceres | `find_package(Ceres 2.1 REQUIRED)` | `Ceres::ceres` | Only `meridian_frontend`. Pulls glog/SuiteSparse transitively. |
| nvblox | `find_package(nvblox REQUIRED)` | `nvblox::nvblox` | Only `meridian_map`; requires `CUDAToolkit`. |
| CUDA | `find_package(CUDAToolkit REQUIRED)` | `CUDA::cudart`, `CUDA::cublas` | Only `meridian_map`. |
| PCL | `find_package(PCL 1.12 ... )` | `${PCL_LIBRARIES}` (or `pcl_*` targets) | Only `meridian_preprocess`/`meridian_tools`. |
| OpenCV | `find_package(OpenCV 4 ...)` | `${OpenCV_LIBS}` | `meridian_frontend` (visual), `meridian_tools`. |
| small_gicp | `find_package(small_gicp REQUIRED)` or vendored | `small_gicp::small_gicp` | Only `meridian_place`. |
| vendored | (no find_package) | `meridian::vendor_*` | via `MeridianVendored.cmake` (§6). |

Every package that produces a library calls `ament_export_dependencies(...)`
listing the deps a downstream consumer needs to re-find, so
`find_package(meridian_frontend)` transitively pulls Eigen/Sophus/Ceres/OpenCV
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
git submodule update --init --recursive      # vendor/basalt-headers, ikd-Tree, scancontext
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
4. **clang-tidy / clang-format** on every core TU; `-Werror` in core.
5. **Build + unit + replay tests** under `colcon test` (GoogleTest); the replay
   test drives `meridian_pipeline` from a bag via `meridian_tools` with no ROS spinning
   (proves the off-ROS path).
6. **Reproducible base image.** CI runs in a tagged Docker image
   (`ros:humble` + JetPack CUDA layer) that snapshots the apt versions in §3, so
   apt deps do not float.

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
   (pulls `vendor/basalt-headers`, `vendor/ikd-Tree`, `vendor/scancontext` at
   their pinned SHAs).
4. **Source deps.** `vcs import src < dependencies.repos` (Sophus, GTSAM 4.2,
   nvblox, small_gicp, ouster-ros at pinned tags/commits).
5. **System deps.** `rosdep install --from-paths src --ignore-src -y` (Eigen 3.4,
   Ceres 2.1, PCL 1.12, OpenCV 4, yaml-cpp, the ROS message/tf/bag packages).
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

If steps 1–10 succeed, the single-system CT LIVO+GNSS estimator builds, the
nvblox GPU map links, and the off-ROS replay path runs — which is the bar this
spec must clear.

---

## 11. Considered & rejected

The design commits to exactly one library per job (§1). For the load-bearing
choices, here is the one-line reason the alternative lost. This section is the
*only* place alternatives appear; nothing in the running design hedges.

| Job | Rejected option | Why rejected |
|---|---|---|
| **Dense mapping** | **VDBFusion / OpenVDB / NanoVDB (CPU, km-scale)** | The deployment target is a CUDA Jetson Orin where nvblox does GPU TSDF + colour + Marching Cubes in one library; a CPU OpenVDB path is a second, untested map backend that the simplicity mandate forbids. We do **not** ship it even as a fallback. (Earlier drafts and grounding 07 floated VDBFusion as a "CPU/km-scale fallback" and a NanoVDB "Tier G archive" — both are dropped here; if extreme-extent outdoor mapping is ever needed it is a *future* `IMapLayer` implementation, not a current dual path.) |
| **Dense mapping** | **Voxblox (CPU)** | Ageing CPU codebase; nvblox supersedes it on GPU with the same block-hash semantics. Kept only as the ESDF *algorithm* reference (grounding 07 §4), never linked. |
| Back-end optimiser | **g2o** | GTSAM ships `ISAM2` (true incremental Bayes tree), `GncOptimizer`, and `CombinedImuFactor` out of the box (grounding 09); g2o would mean re-implementing incremental smoothing and GNC. |
| Back-end optimiser | **Ceres as the global back-end** | Ceres is batch; the global graph needs *incremental* iSAM2. Ceres is used for the *front-end window* instead (where batch-per-window is correct). |
| CT spline kernel | **hand-derived spline Jacobians** | The O(k) analytic derivatives + control-point Jacobians are the highest-risk code to get right; basalt-headers is the proven implementation CLINS/Coco-LIC reuse (grounding 10 §11). Writing them from scratch is months of subtle bugs. |
| CT spline representation | **full SE(3) (non-split) spline** | Couples rotation/translation into an unnatural screw interpolation, slower and no more accurate than split SO(3)×ℝ³ (grounding 10 §2.4); basalt's split is the consensus. |
| NN / registration | **nanoflann / PCL KdTree as primary** | Spec 06 §3 needs incremental insert + box-delete + on-tree downsample near the robot; a static k-d tree forces rebuilds. The adaptive voxel-hash (with ikd-Tree as oracle) gives incremental behaviour without ikd-Tree's layout cost. |
| GICP | **PCL GICP** | Drags the full PCL registration stack; small_gicp is faster, multi-threaded, and takes Eigen buffers directly — matching the L5 verify need without the PCL weight in the hot path. |
| Lie groups | **manif / hand-rolled** | Sophus is what basalt-headers stores as control points and what the references use; using anything else adds a conversion seam at the spline boundary. |
| ROS distro | **Iron / Jazzy** | Not the Jetson/JetPack-aligned LTS; Humble is the supported on-device distro. |
| Front-end framing | **"v1 iEKF, ship CT later"** | Superseded by the DIRECTION: Meridian is one complete CT system from the start. The iEKF remains only as a reference/oracle behind `IFrontEnd`, never a milestone the build is organised around. |
| Single vs multi-LiDAR | **multi-LiDAR / dome-Ouster merge** | The system is single LiDAR + single IMU + single camera + GNSS; multi-LiDAR is at most a future extension behind the same `ISensorSource`/`IFrontEnd` interfaces, not designed or built now. |

---

*End of spec 11 — build system & library choices.*
