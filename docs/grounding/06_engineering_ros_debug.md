# Meridian Grounding Dossier — Engineering, Code Organization, ROS Integration, Parameters, Logging & Visualization across HKU-MARS repos

Scope: what to **emulate** and what to **improve** for a clean, modular, extensible, well-instrumented from-scratch rebuild ("Meridian"). Every claim below was checked against the cloned reference code (FAST_LIO, FAST-LIVO2, Point-LIO) with `file:line` citations. Reference code is read-only; nothing was modified. Where line numbers were verified against the real files they are exact.

---

## 0. TL;DR for downstream authors

- **FAST_LIO** is a *single 1055-line `laserMapping.cpp`* (`FAST_LIO/src/laserMapping.cpp`) that is `main()` + ROS glue + EKF measurement model + map management + all publishers + timing + logging, coordinated by ~80 file-scope globals (lines 69–141). Fast and battle-tested but **monolithic, global-state-coupled, ROS1-only, under-instrumented for live debugging**, and shipping with the two most useful debug publishers commented out (lines 983–984).
- **FAST-LIVO2** is the *most modular and best-instrumented* HKU-MARS design: an 11-line `main.cpp` constructs a `LIVMapper` class, the build is split into separate libraries `vio / lio / pre / imu_proc / laser_mapping`, and it publishes per-residual normal markers, voxel/plane covariance cube markers, an RGB overlay image, and a high-rate IMU-propagated odom. Emulate this.
- **Point-LIO** factored parameter loading into `parameters.cpp` and init into `li_initialization.cpp` (good instinct) but still stitches them via `extern` globals in `parameters.h`, and carries dual-mode branching (`use_imu_as_input`, `prop_at_freq_of_imu`) that hurts readability.
- All three are **ROS1/catkin**. This clone of FAST-LIVO2 has **no ROS2 package/launch** (only catkin `package.xml`, `vikit`, OpenCV, Sophus). **Meridian should be ROS2-native and keep a ROS-free estimator core.**

---

## 1. Module / file structure and build

### 1.1 FAST_LIO — monolithic (one executable, three .cpp)

`FAST_LIO/CMakeLists.txt:87`:
```cmake
add_executable(fastlio_mapping src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
```
plus header-only `IMU_Processing.hpp`, `use-ikfom.hpp`, the IKFoM toolkit, `common_lib.h`, `so3_math.h`. CMake also auto-selects OpenMP core count via `ProcessorCount` → `-DMP_EN -DMP_PROC_NUM=3` for >4 cores (`CMakeLists.txt:18-36`), depends on `PythonLibs`/matplotlibcpp for plotting (`CMakeLists.txt:42-43`), and pins `CMAKE_BUILD_TYPE "Debug"` with `-O3` (`CMakeLists.txt:4,8`). Function inventory of `laserMapping.cpp` (verified line numbers):

| line | function | responsibility |
|---|---|---|
| 143 | `SigHandle` | SIGINT → set exit flag |
| 150 | `dump_lio_state_to_log` | write full state row to `pos_log.txt` |
| 166 | `pointBodyToWorld_ikfom` | frame transform via ikfom state |
| 200 | `RGBpointBodyToWorld` | transform for publishing |
| 211 | `RGBpointBodyLidarToIMU` | LiDAR→IMU body transform |
| 222 | `points_cache_collect` | collect ikd-Tree removed points |
| 231 | `lasermap_fov_segment` | moving local-map cube management |
| 279 | `standard_pcl_cbk` | PointCloud2 callback (Velodyne/Ouster) |
| 302 | `livox_pcl_cbk` | Livox CustomMsg callback + soft time-sync |
| 336 | `imu_cbk` | IMU callback + time-offset compensation |
| 368 | `sync_packages` | LiDAR+IMU temporal bundling into `MeasureGroup` |
| 427 | `map_incremental` | incremental ikd-Tree insertion |
| 478 | `publish_frame_world` | `/cloud_registered` + PCD save |
| 532 | `publish_frame_body` | `/cloud_registered_body` |
| 551 | `publish_effect_world` | `/cloud_effected` (effective points) |
| 567 | `publish_map` | `/Laser_map` |
| 589 | `publish_odometry` | `/Odometry` + covariance + TF |
| 622 | `publish_path` | `/path` (decimated 1/10) |
| 638 | `h_share_model` | **measurement model**: NN search, plane fit, residual, Jacobian H |
| 756 | `main` | param load, subscribers, publishers, main loop, timing, logging |

**Strengths:** trivial to read top-to-bottom; zero indirection; one TU. **Weaknesses:** no separation of *algorithm* vs *ROS transport*; ~80 file-scope mutable globals (`laserMapping.cpp:69-141`) preclude unit testing and re-entrancy; the measurement model is a free function `h_share_model` with global side-effects (`laserCloudOri`, `corr_normvect`, `effct_feat_num`, `res_last[]`, `normvec`) registered into the filter by raw function pointer: `kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi)` (`laserMapping.cpp:828`).

### 1.2 FAST-LIVO2 — class + libraries (EMULATE)

`FAST-LIVO2/src/main.cpp` in full:
```cpp
#include "LIVMapper.h"
int main(int argc, char **argv) {
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  LIVMapper mapper(nh);
  mapper.initializeSubscribersAndPublishers(nh, it);
  mapper.run();
  return 0;
}
```
Build splits concerns into linkable libraries (`FAST-LIVO2/CMakeLists.txt`):
```cmake
add_library(vio          src/vio.cpp src/frame.cpp src/visual_point.cpp)
add_library(lio          src/voxel_map.cpp)
add_library(pre          src/preprocess.cpp)
add_library(imu_proc     src/IMU_Processing.cpp)
add_library(laser_mapping src/LIVMapper.cpp)
add_executable(fastlivo_mapping src/main.cpp)
target_link_libraries(fastlivo_mapping laser_mapping vio lio pre imu_proc ...)
```
It also has ARM/x86-specific `-O3 -march/-mcpu=native` tuning, optional `mimalloc`, and C++17 (`CMakeLists.txt`). The orchestration class `LIVMapper` exposes a clean lifecycle (`FAST-LIVO2/include/LIVMapper.h:27-63`): `readParameters()`, `initializeComponents()`, `initializeFiles()`, `initializeSubscribersAndPublishers()`, `run()`, then phase methods `gravityAlignment()`, `handleFirstFrame()`, `processImu()`, `handleVIO()`, `handleLIO()`, `stateEstimationAndMapping()`, `savePCD()`. **This is the cleanest separation in the family** and is the template Meridian should follow.
Remaining weakness: `LIVMapper` is a *god-object* — it owns ~60 config scalars, all buffers, all 18 publishers/subscribers, and every callback as members (`LIVMapper.h:65-186`). State is class-scoped (better than file-scoped) but not decomposed into estimator vs I/O.

### 1.3 Point-LIO — split TUs, shared externs

`Point-LIO/CMakeLists.txt` builds one executable from `laserMapping.cpp + parameters.cpp + preprocess.cpp + Estimator.cpp + IMU_Processing.cpp + li_initialization.cpp + ikd_Tree.cpp`. Parameter loading lives in `parameters.cpp` (`readParameters` at line 47), init in `li_initialization.cpp`, estimator math in `Estimator.cpp`. **Good instinct (one concern per file)** but they are stitched by `extern` globals declared in `parameters.h` — still one shared mutable namespace, just spread across files. The dual-mode design (`use_imu_as_input`, `prop_at_freq_of_imu`, `Point-LIO/config/avia.yaml`) forces global-flag branching throughout `laserMapping.cpp` (e.g. lines 59/70/100/408/588/683/873/1005/1041).

---

## 2. Parameter loading

### 2.1 Mechanism (all three)

ROS1 `NodeHandle::param<T>(name, var, default)` reads the parameter server, populated by `<rosparam command="load" file=".../config/*.yaml"/>` plus inline `<param>` tags. FAST_LIO loads every parameter in `main` (`FAST_LIO/src/laserMapping.cpp:761-793`), e.g.:
```cpp
nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");           // :767
nh.param<bool>("common/time_sync_en", time_sync_en, false);                // :769
nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);                          // :777
nh.param<int>("preprocess/lidar_type", lidar_type, AVIA);                   // :782
nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>()); // :792
```
FAST-LIVO2 does the same in `LIVMapper::readParameters` with richer namespaces — `common/ vio/ time_offset/ uav/ evo/ imu/ preprocess/ pcd_save/ image_save/ extrin_calib/ debug/ publish/` (`src/LIVMapper.cpp:50-114`).

### 2.2 Where parameters live — split across YAML *and* launch (anti-pattern)

Sensor/algorithm params are split between YAML and launch XML. `FAST_LIO/launch/mapping_avia.launch`:
```xml
<rosparam command="load" file="$(find fast_lio)/config/avia.yaml" />
<param name="feature_extract_enable" type="bool"   value="0"/>
<param name="point_filter_num"       type="int"    value="3"/>
<param name="max_iteration"          type="int"    value="3"/>
<param name="filter_size_surf"       type="double" value="0.5"/>
<param name="filter_size_map"        type="double" value="0.5"/>
<param name="cube_side_length"       type="double" value="1000"/>
<param name="runtime_pos_log_enable" type="bool"   value="0"/>
```
So `point_filter_num`, `max_iteration`, `filter_size_*`, `cube_side_length`, the logging toggle, and feature toggle are **in the launch file**, while topics, covariances, extrinsics, FOV/det_range, publish toggles, pcd_save are **in the YAML** (`config/avia.yaml`). The split is undocumented and drift-prone: the launch sets `cube_side_length=1000` while the code default is **200** (`laserMapping.cpp:774`). YAML namespaces are `common/ preprocess/ mapping/ publish/ pcd_save/`. FAST-LIVO2 repeats the same habit (launch sets `point_filter_num`, `max_iteration`, etc. on top of `config/avia.yaml`).

### 2.3 No schema, no validation, silent defaults

`nh.param` silently falls back to the hard-coded default if a key is missing/misspelled — **no validation, no range checking, no unknown-key warning, no dump of the resolved config**. A typo in YAML yields a silently wrong run. Point-LIO centralizes loading in `parameters.cpp` but uses the same silent-default mechanism.

### 2.4 Per-sensor config files

Each repo ships one YAML per LiDAR. FAST_LIO `config/`: `avia.yaml`, `horizon.yaml`, `ouster64.yaml`, `velodyne.yaml`, `mid360.yaml`, `marsim.yaml`. They differ mainly in `lid_topic`/`imu_topic`, `lidar_type`, `scan_line`, `timestamp_unit`, `scan_rate`, `blind`, `fov_degree`, `det_range`, `extrinsic_T/R`, and occasionally `extrinsic_est_en` (true in `horizon.yaml`, false elsewhere). Verified differences: avia `det_range 450 / fov 90`, velodyne `det_range 100 / fov 180 / scan_line 32 / timestamp_unit 2`, ouster64 `det_range 150 / fov 180 / scan_line 64 / timestamp_unit 3`. The duplicated covariance/publish/pcd_save blocks across files invite drift.

---

## 3. Threading & control-flow model

### 3.1 FAST_LIO — single-thread spin loop + lock-guarded buffers

- ROS callbacks (`standard_pcl_cbk` :279, `livox_pcl_cbk` :302, `imu_cbk` :336) run on `ros::spinOnce()` in the **main thread**, each guarded by one global `mutex mtx_buffer` and signalling `condition_variable sig_buffer` (`laserMapping.cpp:81-82`). They only *deserialize → preprocess (`p_pre->process`) → push to buffer* (`lidar_buffer`, `imu_buffer`, `time_buffer`, declared :104-106).
- The estimator runs in `main`'s busy loop at `ros::Rate rate(5000)` (`laserMapping.cpp:863`): `ros::spinOnce()` → `sync_packages(Measures)` → `p_imu->Process(...)` → `lasermap_fov_segment()` → `downSizeFilterSurf.filter()` → `kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time)` (:960) → `publish_odometry` (:972) → `map_incremental()` (:976) → publishers → logging. Loop ends with `rate.sleep()` (:1018).
- So the estimator is **single-threaded**, with callbacks interleaved on the same thread. The only parallelism is **OpenMP inside `h_share_model`**: `#pragma omp parallel for` over `feats_down_size`, gated by `MP_EN`/`MP_PROC_NUM` (`laserMapping.cpp:646-650`).

**Weaknesses:** the 5000 Hz busy-wait is a CPU-burning poll — the cv `sig_buffer` is declared and `notify_all`'d in every callback but the main loop **never `wait`s on it**, so the cv is effectively dead code. Subscribe queue depth is **200000** (`laserMapping.cpp:845-848`) and buffers are unbounded `deque`s with no backpressure.

### 3.2 FAST-LIVO2 — adds a real-time IMU-propagation path + second mutex

Same buffer pattern but with a second mutex `mtx_buffer_imu_prop` (`LIVMapper.h:65`) and a high-rate IMU-propagation odom path (`imu_prop_callback` at `LIVMapper.cpp:576`, `prop_imu_once` at :556, publisher `/LIVO2/imu_propagate` advertised at :214) gated by `uav/imu_rate_odom` (`imu_prop_enable`). This decouples low-latency pose output from the LiDAR-rate EKF — the pattern to emulate for low-latency odom. It also has image, IMU, LiDAR, and (optional) gravity-alignment sub-paths sequenced through `run()` (`LIVMapper.cpp:534`).

### 3.3 Point-LIO — dual estimator mode

IMU as **input** vs as **measurement**, and propagation at IMU vs LiDAR frequency (`use_imu_as_input`, `prop_at_freq_of_imu`, `config/avia.yaml`). Same single-loop transport; the branching is what makes its `laserMapping.cpp` hard to follow.

---

## 4. Published topics (debug / visualization surface)

### 4.1 FAST_LIO advertised topics (`laserMapping.cpp:849-860`)

| topic | type | source fn | purpose |
|---|---|---|---|
| `/cloud_registered` | `sensor_msgs/PointCloud2` | `publish_frame_world` (478) | undistorted scan in world frame `"camera_init"`; dense or downsampled per `dense_pub_en` (:482) |
| `/cloud_registered_body` | `PointCloud2` | `publish_frame_body` (532) | scan in IMU-body frame `"body"` |
| `/cloud_effected` | `PointCloud2` | `publish_effect_world` (551) | **effective points** (`laserCloudOri`, `effct_feat_num`) that produced residuals |
| `/Laser_map` | `PointCloud2` | `publish_map` (567) | flattened ikd-Tree map (`featsFromMap`) |
| `/Odometry` | `nav_msgs/Odometry` | `publish_odometry` (589) | pose + 6×6 covariance from `kf.get_P()`; TF `camera_init→body` (:619) |
| `/path` | `nav_msgs/Path` | `publish_path` (622) | trajectory, decimated 1-in-10 to avoid rviz crash (:631) |

**Critical detail:** in the main loop `publish_effect_world` and `publish_map` are **commented out** (`laserMapping.cpp:983-984`):
```cpp
// publish_effect_world(pubLaserCloudEffect);
// publish_map(pubLaserCloudMap);
```
So out of the box you **cannot see which points the filter used** or the live map — the two most useful debug views are disabled. Likewise the map-flatten block is gated `if(0)` (`laserMapping.cpp:942`). Odometry covariance is repacked to `[pos | rot]` order (`laserMapping.cpp:597-606`). Frame convention: world `"camera_init"`, body `"body"`; TF tree is just `camera_init → body` (:619).

### 4.2 FAST-LIVO2 — much richer debug surface (EMULATE) (`src/LIVMapper.cpp:200-216`)

| topic | type | purpose |
|---|---|---|
| `/cloud_registered` | `PointCloud2` | registered scan |
| `/cloud_visual_sub_map_before` | `PointCloud2` | visual sub-map used by VIO |
| `/cloud_effected` | `PointCloud2` | effective LIO points (from `std::vector<PointToPlane> ptpl_list`) |
| `/Laser_map` | `PointCloud2` | map |
| **`visualization_marker`** | **`visualization_msgs/MarkerArray`** (`pubNormal`) | **per-residual surface normals as markers** — direct view of point-to-plane constraints |
| **`/planner_normal`** | `Marker` (`plane_pub`) | plane normal marker |
| **`/voxels`**, **`/planes`** | `MarkerArray` (`voxel_pub`, `voxelmap_manager->voxel_map_pub_`) | **voxel/plane cubes colored by plane covariance** |
| `/dyn_obj`, `/dyn_obj_removed`, `/dyn_obj_dbg_hist` | `PointCloud2` | dynamic-object visualization |
| **`/rgb_img`** | `image_transport` image (`pubImage`) | **RGB photometric overlay** for the visual model |
| `/aft_mapped_to_init` | `Odometry` | pose |
| `/path` | `Path` | trajectory |
| `/LIVO2/imu_propagate` | `Odometry` | high-rate IMU-propagated odom |
| `/mavros/vision_pose/pose` | `geometry_msgs/PoseStamped` | UAV integration |

`publish_effect_world(const ros::Publisher&, const std::vector<PointToPlane>& ptpl_list)` (`LIVMapper.cpp:1308`) takes the residual list directly, and is only called when `pub_effect_point_en` (:446). `VoxelMapManager::pubVoxelMap` (`src/voxel_map.cpp:788`) builds a `MarkerArray` of plane cylinders via `pubSinglePlane` (:837) and `CalcVectQuation` (:864), colored by plane covariance — the closest any repo gets to **map-uncertainty / observability visualization**, gated by `publish/pub_plane_en` (`voxel_map.cpp:38`).

### 4.3 Point-LIO (`src/laserMapping.cpp:379-391`)

Advertises `/cloud_registered`, `/cloud_registered_body`, `/Laser_map`, `/aft_mapped_to_init` (Odometry), `/path`. The `/cloud_effected` publisher and a `visualization_msgs/Marker plane_pub` are present but **commented out** (`laserMapping.cpp:383, 391`). No marker-based residual/plane visualization is active.

---

## 5. Timing / CPU logging

### 5.1 FAST_LIO instrumentation

Timing uses `omp_get_wtime()` snapshots `t0..t5` around pipeline stages in the main loop (`laserMapping.cpp:886-977`), plus accumulators `match_time`, `solve_time`, `solve_const_H_time`, `kdtree_search_time`, `kdtree_incremental_time`, `kdtree_delete_time` (declared :69-71; reset each frame :881-885). When `runtime_pos_log` is true it computes running averages and prints (`laserMapping.cpp:1009`):
```
[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f
ave ICP: %0.6f map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f
```
It also fills 12 parallel fixed C arrays `T1, s_plot..s_plot11[MAXN]` (`MAXN=720000`, `laserMapping.cpp:65,70`) with per-frame total time, point count, kdtree incremental/search/delete times, delete counter, tree size start/end, average total, added points (`laserMapping.cpp:997-1008`); per-scan preprocess time is stashed in `s_plot11[scan_count]` inside the callbacks (:295, :331). These dump to `Log/fast_lio_time_log.csv` at shutdown (`laserMapping.cpp:1040-1051`).

**Logging files** (`laserMapping.cpp:832-838`): `Log/pos_log.txt` (full state via `dump_lio_state_to_log` :150), `Log/mat_pre.txt`, `Log/mat_out.txt`, `Log/dbg.txt`, written through the `DEBUG_FILE_DIR(name)` macro (`common_lib.h:34`). Offline analysis is two scripts: `Log/plot.py` (attitude/translation/extrinsics/velocity/bias/gravity from `mat_pre.txt`/`mat_out.txt`) and `Log/fast_lio_time_log_analysis.m` (MATLAB). `Log/guide.md` documents the workflow.

**Weaknesses:** all logging is `printf`/`fprintf` to fixed-size global arrays and flat text — no levels, no per-line timestamps, no rotation, no structured/queryable format, no live timing topic. `MAXN`×12 doubles (~69 MB) is a fixed cost that overflows on long runs. FAST-LIVO2 prints a nicer per-stage table with ANSI colors (e.g. `printf("\033[1;36m| %-29s | %-27f |\033[0m
", "updateVoxelMap", t4 - t3)`, `LIVMapper.cpp:472`) but it is still stdout-only.

---

## 6. RViz configuration

### 6.1 FAST_LIO `rviz_cfg/loam_livox.rviz`

Display inventory (verified by class counts): **5× PointCloud2, 1× Odometry, 1× Path, 1× Grid, 2× Axes, 1× MarkerArray**. Topics wired: `/cloud_registered` (displays "surround" and "currPoints"), `/Laser_map` (display "mapped" + a second PointCloud2), `/cloud_effected`, `/Odometry`, `/path`, and `/MarkerArray`. Fixed frame `camera_init`. Note the **MarkerArray display points at `/MarkerArray`** but no FAST_LIO code publishes that topic — and the effective-points/map publishers are commented out — so the rviz config references views the binary never produces. (FAST_LIO publishes `/Odometry`, and this rviz wires `/Odometry` correctly; an earlier mismatch claim is corrected here.)

### 6.2 FAST-LIVO2 `rviz_cfg/fast_livo2.rviz` (+ `M300.rviz`, `hilti.rviz`, `ntu_viral.rviz`)

Ships per-platform rviz configs that actually exercise the richer topic set (Image `/rgb_img`, the normal/voxel/plane `MarkerArray`s, clouds, odom, path). Point-LIO ships `rviz_cfg/loam_livox.rviz`.

---

## 7. Key shared data structures (interfaces for the rebuild)

From `FAST_LIO/include/common_lib.h`:
- `MeasureGroup` (lines 55-66): the synchronized bundle — `double lidar_beg_time, lidar_end_time`, `PointCloudXYZI::Ptr lidar`, `deque<sensor_msgs::Imu::ConstPtr> imu`. Natural **input message** for a sensor-agnostic estimator step, but it embeds the ROS `sensor_msgs::Imu` type, coupling the core to ROS.
- `PointType = pcl::PointXYZINormal`; `PointCloudXYZI = pcl::PointCloud<PointType>` (`common_lib.h:37-38`). FAST_LIO **abuses the `curvature` field** to carry per-point relative timestamp in ms: `sync_packages` uses `meas.lidar->points.back().curvature / 1000` to derive `lidar_end_time` (`laserMapping.cpp:386,393`). This is an implicit, undocumented contract between `preprocess` and the estimator.
- Constants/macros (`common_lib.h:19-48`): `DIM_STATE=18`, `DIM_PROC_N=12`, `NUM_MATCH_POINTS=5`, `MAX_MEAS_DIM=10000`, plus `VEC_FROM_ARRAY`, `MAT_FROM_ARRAY`, `DEBUG_FILE_DIR`, `SKEW_SYM_MATRX`, and a legacy `StatesGroup` with manifold `operator+` (lines 68-120) used by the non-ikfom path.
- `Preprocess` class (`FAST_LIO/src/preprocess.h:86-129`): two overloaded `process()` (Livox `CustomMsg` and `PointCloud2`), per-type handlers `avia_handler`/`oust64_handler`/`velodyne_handler`/`sim_handler`, `enum LID_TYPE{AVIA=1,VELO16,OUST64,MARSIM}`, `enum TIME_UNIT{SEC,MS,US,NS}`, and registered PCL point structs for `velodyne_ros::Point`/`ouster_ros::Point` (lines 41-84). Already a clean reusable module and a good template — but it depends on ROS message types directly.

---

## 8. Concrete recommendations for Meridian

### 8.1 Layering — separate the estimator from ROS entirely
Create a **ROS-free core library** (`meridian_core`) that never includes `ros::`/`rclcpp`:
- `meridian_core/preprocess` — input clouds → an internal `PointCloud` with its own POD point type carrying an **explicit `float rel_time` field** instead of abusing `curvature` (cf. the implicit contract at `laserMapping.cpp:386,393`). Decouple from ROS message types (FAST_LIO's `Preprocess`, `preprocess.h:94-95`, takes ROS msgs directly).
- `meridian_core/imu` — integration/undistortion (today `IMU_Processing.hpp`).
- `meridian_core/map` — ikd-Tree / voxel map behind an abstract `IMap` interface (FAST_LIO uses ikd-Tree; FAST-LIVO2 uses a voxel octo-map `voxel_map.cpp`; Point-LIO uses iVox `include/ivox/`). Make the choice swappable.
- `meridian_core/estimator` — the IESKF plus a **`MeasurementModel` interface** replacing the raw function pointer `h_share_model` (registered with hidden global outputs at `laserMapping.cpp:828`, body at :638). Define `class PlaneToPointResidual : public IResidual` returning `{H_block, z, weight, valid}` plus an optional debug payload (normal, footpoint, residual). This makes per-residual data first-class and testable, and kills the global `laserCloudOri/corr_normvect/normvec/res_last` arrays.
- A thin **`meridian_ros2`** node (the only place `rclcpp` appears) adapting topics ↔ core, following FAST-LIVO2's `LIVMapper` lifecycle (`readParameters / initializeComponents / initializeSubscribersAndPublishers / run`, `LIVMapper.h:29-32,59`) but with the god-object decomposed: separate `SensorIO`, `PublisherBundle`, and `Estimator` collaborators rather than one class holding everything (contrast `LIVMapper.h:65-186`).

Build with separate libraries like FAST-LIVO2 (`add_library(vio/lio/pre/imu_proc/laser_mapping ...)`) and a trivial node. **Eliminate the ~80 file-scope globals** of `laserMapping.cpp:69-141`; own all state in objects.

### 8.2 Parameters — single source of truth + validation
- One YAML, **no params in launch files** — eliminate the FAST_LIO split where `point_filter_num`/`max_iteration`/`filter_size_*`/`cube_side_length` live in `mapping_avia.launch` while the rest live in `avia.yaml`, and where launch `cube_side_length=1000` contradicts code default `200` (`laserMapping.cpp:774`).
- In ROS2, use `declare_parameter` with `ParameterDescriptor` ranges, and **fail loudly on unknown/out-of-range keys** — the opposite of `nh.param`'s silent default (`laserMapping.cpp:761-793`).
- **Echo the fully-resolved config at startup** and publish it on a latched `/meridian/config` topic so each run is self-documenting.
- Keep per-sensor profiles but factor shared blocks (covariances, publish toggles, pcd_save) into a base/defaults file to stop the cross-file drift seen across `config/*.yaml`.

### 8.3 Debug topics & markers — make the estimator *visible* (default ON, throttled)
- Publish **`/cloud_effected`** and the **live map** by default (FAST_LIO leaves both commented out, `laserMapping.cpp:983-984`, and gates the map flatten `if(0)` at :942).
- Adopt FAST-LIVO2's **per-residual normal `MarkerArray`** (`pubNormal`, `LIVMapper.cpp:201`) — one arrow per active point-to-plane residual, colored by residual magnitude or robust weight. Highest-value debug view.
- Adopt FAST-LIVO2's **voxel/plane cube markers colored by plane covariance** (`voxel_map.cpp:788 pubVoxelMap` / `:837 pubSinglePlane`) for map-quality inspection.
- **Add explicit observability/degeneracy visualization** that none ship live: from `Hᵀ H` (or `kf.get_P()`, currently used only for odom covariance at `laserMapping.cpp:596`), compute eigenvalues/eigenvectors and publish (a) per-axis degeneracy scores on `/meridian/diagnostics` and (b) `MarkerArray` arrows for weakly-observed world-frame directions. This directly tells the operator "the corridor/tunnel is degenerate along X".
- Publish **per-frame residual stats** (`res_mean_last`, `effct_feat_num`, computed at `laserMapping.cpp:704-715` but only logged to text) as a small plottable diagnostic message.
- Add a high-rate **IMU-propagated odom** like FAST-LIVO2's `/LIVO2/imu_propagate` path (`LIVMapper.cpp:556-636, :214`) for low-latency pose.

### 8.4 Structured logging & timing
- Replace `printf`/`fprintf` (`laserMapping.cpp:1009, 150`) and the fixed `s_plot[MAXN]` arrays (`:65,70`, ~69 MB) with a **leveled, timestamped structured logger** (spdlog-style) writing to a **ring buffer**, not fixed arrays that overflow.
- Emit **per-module timing as a ROS2 diagnostics message** (`diagnostic_msgs` or a custom `StageTimings`): keep the existing breakdown (preprocess, IMU undistort, downsample, kdtree search/incremental/delete, H construction, solve, total — accumulators at `laserMapping.cpp:69-71`, printed at :1009) but make it live and queryable with a stable schema (FAST-LIVO2's colored table at `LIVMapper.cpp:472` is the nicer UX to copy, made into a topic).
- Wrap each stage in an RAII scoped timer (`omp_get_wtime` deltas today, :886-977) so adding a stage auto-registers its timing.
- Record diagnostics + odom + effective points to rosbag2 so failed runs can be replayed.

### 8.5 ROS2-idiomatic patterns
- One executable, **component/lifecycle node** (`rclcpp_lifecycle`) so configure/activate is explicit, replacing the FAST_LIO busy-loop init (`laserMapping.cpp:863-877`).
- Use **callback groups + a real executor**, with bounded sensor queues and explicit QoS (sensor-data QoS for clouds/IMU), instead of FAST_LIO's `200000`-deep unbounded subscriptions (`laserMapping.cpp:845-848`) and the 5000 Hz `rate.sleep()` poll loop (`:863,1018`). Drive the estimator off a condition variable / message-triggered callback (FAST_LIO declares `sig_buffer` but never waits on it — `:82`).
- Publish TF via `tf2_ros::TransformBroadcaster` with documented REP-105 frames (`map`/`odom`/`base_link`) instead of the bare `camera_init→body` (`laserMapping.cpp:619`).
- Ship a **ROS2-native Python launch + parameter file**. This FAST-LIVO2 clone has **no ROS2 launch/package** at all (only catkin `package.xml` with `vikit`, OpenCV, Sophus); do not inherit the ROS1 launch XML.
- Ship rviz configs that wire exactly the published topics (avoid FAST_LIO's `/MarkerArray` display that nothing publishes, and the commented-out effect/map publishers).

### 8.6 Net "emulate vs improve" table
| aspect | emulate from | improve |
|---|---|---|
| modular libs + thin main | FAST-LIVO2 (`main.cpp`, CMake libs) | decompose the `LIVMapper` god-object |
| lifecycle methods | FAST-LIVO2 `LIVMapper.h:29-32,59` | make them collaborators, not one class |
| per-residual normal markers | FAST-LIVO2 `pubNormal` (`LIVMapper.cpp:201`) | add degeneracy/observability markers (none exist) |
| voxel/plane covariance markers | FAST-LIVO2 `voxel_map.cpp:788` | publish by default, throttled |
| effective-points topic | FAST_LIO `/cloud_effected` (`laserMapping.cpp:551`) | enable by default (commented out at :983) |
| high-rate IMU odom | FAST-LIVO2 `/LIVO2/imu_propagate` (`LIVMapper.cpp:556`) | lifecycle + QoS |
| split param/init files | Point-LIO `parameters.cpp`, `li_initialization.cpp` | single YAML, validation, no launch params, config echo |
| stage timing breakdown | FAST_LIO accumulators (`laserMapping.cpp:69-71,1009`) | structured logger + diagnostics msg + ring buffer |
| Preprocess class | FAST_LIO `preprocess.h:86` | decouple from ROS message types; explicit per-point time field |
| globals + busy-loop | — (avoid) | object state + executor/cv-driven loop |

---

## 9. Citations index
See the `citations` field for the concrete `file:line` references underpinning each section.
