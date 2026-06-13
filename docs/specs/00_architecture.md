# Meridian — Software Architecture Specification (Spec 00)

> **Status:** authoritative. This is the top-of-stack design document for Meridian, a from-scratch, SOTA, tightly-coupled multi-sensor 3D SLAM system for tactical operational use. It defines *how the code is organized* — not *how the estimator works* (the math lives in the per-module specs `01_*` onward). Everything here is normative: where **MUST**, **SHOULD**, **MAY** appears it carries its RFC-2119 meaning.
>
> **What Meridian is, in one line.** A **tightly-coupled LiDAR-Inertial** SLAM system: a **discrete-time LIO** front-end — voxel-hash local map, Gauss-Newton point-to-point ICP seeded by an interval-averaged IMU screw prior with adaptive gravity regularization, deskew internal to the estimator — feeding an iSAM2 factor-graph back-end and a GPU **nvblox** TSDF+colour+mesh map. Camera and GNSS streams are ingested and carried through the pipeline (image passthrough on keyframes, GNSS quality gate) but are **not yet fused** by the estimator.
>
> **Deployment target.** **NVIDIA Jetson Orin.** A CUDA GPU is **always present**, so the deployed surface map runs the **nvblox GPU backend**. The surface tier sits behind the backend-pluggable `IMapLayer`/`ISurfaceMap` seam (`nvblox` GPU / `cpu` host / deferred `vulkan`); the backend is chosen explicitly and fail-fast — the robot never silently downgrades off nvblox (spec 06 §0).
>
> **Scope reminder.** First-pass scope ends at a **colourised triangle mesh** (nvblox Marching Cubes). ESDF/path-planning and semantics/object-detection are *designed-for* but *not built now*; the architecture must leave clean seams for them (§12).
>
> **Companion docs.** `DATASET.md` (the Newer College 2021 benchmark set — evaluation data). Sibling specs: `01_interfaces_and_data_types`, `02_sensors_timesync`, `03_preprocessing`, `04_frontend_estimation`, `05_backend_graph`, `06_mapping`, `07_loop_closure`, `08_calibration`, `09_debug_introspection`, `10_evaluation_harness`. Reference grounding (SOTA `repo@sha`/`file:symbol` citations, verified against the clones in `/home/user/slam-reference`) lives in each spec's non-normative Appendix R.
>
> **Grounding.** Meridian combines proven components — a **discrete LIO** odometry front-end (voxel-hash map, GN point-to-point ICP, IMU screw prior; the algorithmic design is specified in `04_frontend_estimation.md`), **nvblox** (GPU TSDF+colour+Marching-Cubes mesh), and **iSAM2/GTSAM** (incremental factor graph). Engineering and debug recommendations are additionally grounded in the reference implementation `slam-reference/FAST_LIO/src/laserMapping.cpp`, cited as `laserMapping.cpp:NNN`; we keep what is good there and fix what is not.

---

## 0. The one-paragraph mental model

Meridian is a **ROS-agnostic C++ core** (plain libraries that know nothing about ROS, DDS, or any middleware) wrapped by a **thin ROS 2 layer** (nodes that do nothing but marshal messages in, call the core, and publish results out). The core is a stack of **layers L0–L6**, each behind a small **interface** (`ISensorSource`, `IFrontEnd`, `IBackEnd`, `IMapLayer`, `IPlaceRecognizer`, …) so any one layer can be reasoned about, tested, and (where it makes sense) swapped without touching its neighbours. The contract that ties the latency-sensitive front-end (L2) to the globally-consistent back-end (L3) is **one immutable value type — the `KeyframePacket`** — and nothing else crosses that boundary. Threading is a small fixed set of stages connected by bounded queues. Configuration is a single typed parameter tree loaded identically from a YAML file (offline/bag replay) or the ROS 2 parameter server (live). Introspection is a first-class subsystem: every module emits structured timing, residual, and observability telemetry on dedicated debug topics and rviz markers, so an operator can *see* what the estimator is doing — explicitly improving on the ad-hoc debug publishing entangled inside FAST-LIO's `laserMapping.cpp`.

One diagram to remember:

```
                          ┌────────────────────────────────────────────────────────┐
   sensors  ─────────────▶│  ROS 2 WRAPPER (thin):  *_node executables               │
   (live or bag)          │  subscribe → convert msg→core type → call core → publish │
                          └───────────────┬────────────────────────────────────────┘
                                          │  (plain C++ types only; no ros msgs below)
   ┌──────────────────────────────────────▼─────────────────────────────────────────┐
   │  MERIDIAN CORE  (ROS-agnostic libraries)                                            │
   │                                                                                  │
   │  L0 sensor abstraction + time sync ──▶ L1 preprocessing ──▶ L2 LIO               │
   │   (1 LiDAR, 1 IMU, 1 cam, GNSS)                            front-end (odom)       │
   │                                                                  │ KeyframePacket │
   │                                                                  ▼                │
   │  L4 nvblox map  ◀── L3 back-end (factor graph + loop) ◀── L5 place recognition    │
   │       │  (GPU TSDF+colour+mesh)                                                   │
   │       ▼                                                                          │
   │  L6 operator surface (colour mesh + confidence)                                  │
   │                                                                                  │
   │  cross-cutting: meridian_common (math/types), meridian_time (PTP), meridian_calib,        │
   │                 meridian_debug (telemetry), meridian_config                            │
   └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Why ROS-agnostic core + thin ROS 2 wrapper

This split is the single most important architectural decision; the project owner made it non-negotiable. The rationale, stated so future-you cannot argue with it:

1. **Testability without a middleware.** A SLAM estimator is a deterministic function of a stream of timestamped measurements. If the estimator only ever sees plain C++ structs, you can unit-test it, replay a recorded measurement stream, fuzz it, and run it under `perf`/`valgrind` — all without spinning up DDS, a ROS graph, or a clock. The wrapper is then trivial to test because it contains no algorithm, only translation.

2. **Replaceability of the transport.** Tactical deployment may not want full ROS 2 at runtime (DDS discovery storms, security posture, footprint). With a clean core, the wrapper can be ROS 2 today, a Zenoh shim tomorrow, or a bare `main()` reading a bag — the algorithm is untouched. This is also how the **offline test harness** (`DATASET.md`) and the **live robot** become *the same code path*: both feed the same `ISensorSource`.

3. **Reasoning at the right altitude.** ROS message types (`sensor_msgs::PointCloud2`, `nav_msgs::Odometry`) are wire formats, not domain types. They carry middleware concerns (QoS, `frame_id` strings), are awkward to compute on (XYZIRT packed in a byte blob), and version with ROS. The core computes on domain types (`meridian::PointCloud`, `meridian::Pose`, `KeyframePacket`) designed for the math, not the wire.

4. **Contrast with the reference code.** FAST-LIO fuses everything into one translation unit: `laserMapping.cpp` *is* the ROS node, the parameter loader, the buffer manager, the estimator driver, and the publisher. It directly `news` publishers (`ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000)`, `laserMapping.cpp:849`), reads ~40 parameters inline (`nh.param<...>`, `laserMapping.cpp:761-793`), and runs the estimator loop in `main()` (`laserMapping.cpp:865-1019`) — 1055 lines, all concerns interleaved. That is fine for a research prototype and wrong for a system we intend to extend for years and run operationally. Meridian deliberately inverts it.

### 1.1 The hard rule

> **No core library may depend on `rclcpp`, `rcl`, any `*_msgs` package, DDS, or the ROS parameter/clock/logging APIs.** Time is passed as `Timestamp` (int64 nanoseconds, §6.2). Logging goes through `meridian::log` (a sink the wrapper binds to `rclcpp` logging at startup, §10.3). Parameters arrive as a populated `meridian::Config` struct (§8). **CUDA is not ROS:** GPU code (nvblox) is core, not wrapper — the no-ROS rule is about middleware, not about the GPU, which is always present (§9.5).

This is checkable mechanically (§9.4) and is the line that keeps the architecture honest.

---

## 2. Package / module layout

Meridian is a **colcon workspace** of ROS 2 *ament_cmake* packages (ROS 2 **Humble**). Core libraries are ament packages too (so colcon builds and tests them uniformly) but their `package.xml` lists **no `rclcpp` dependency** — only Eigen, PCL, GTSAM, Ceres, nvblox, etc. Wrappers are separate packages depending on both the core and `rclcpp`.

```
meridian_ws/
├─ src/
│  ├─ meridian_common/            # X-cut: math, geometry, core value types, KeyframePacket
│  │   include/meridian/common/   #   pose.hpp, twist.hpp, nav_state.hpp, time.hpp, frame.hpp,
│  │   test/                   #   point.hpp, cloud.hpp, sample.hpp, gaussian.hpp,
│  │                           #   observability.hpp, keyframe_packet.hpp, graph_update.hpp,
│  │                           #   loop_constraint.hpp, stamped_pose.hpp,
│  │                           #   imu_preintegration.hpp, status.hpp, ring_buffer.hpp,
│  │                           #   bounded_queue.hpp
│  │
│  ├─ meridian_config/            # X-cut: typed Config tree + YAML loader (no ROS)
│  ├─ meridian_time/              # X-cut: time model, PTP/PPS clock model, interpolation
│  ├─ meridian_debug/             # X-cut: telemetry bus (timing/residual/observability), no ROS
│  ├─ meridian_calib/             # X-cut: extrinsic/intrinsic models, offline prior load,
│  │                           #        camera photometric (exposure/gain) params
│  │
│  ├─ meridian_sensors/           # L0: ISensorSource, sensor models, time-sync + aggregation
│  │                           #     (single LiDAR + single IMU + single camera + GNSS)
│  ├─ meridian_preprocess/        # L1: downsample, validity, image pyramid, GNSS gate
│  │                           #     (NB: deskew happens inside L2; see §7)
│  ├─ meridian_frontend/          # L2: IFrontEnd + the discrete LIO estimator
│  │   include/meridian/frontend/ #   ifrontend.hpp     <-- THE interface
│  │   src/lio/                #   PRIMARY: discrete LIO estimator
│  │   │                       #     voxel_grid_map.{hpp,cpp}  — voxel-hash local map
│  │   │                       #     imu_tracker.{hpp,cpp}     — static init, propagation, priors
│  │   │                       #     scan_registration.{hpp,cpp} — screw deskew + GN ICP
│  │   │                       #     relative_cov.hpp          — relative-covariance transport
│  │   │                       #     lio_frontend.{hpp,cpp}    — IFrontEnd orchestration
│  │   src/frontend_factory.cpp
│  │
│  ├─ meridian_backend/           # L3: IBackEnd + GTSAM iSAM2 factor-graph impl
│  ├─ meridian_map/               # L4: IMapLayer + ISurfaceMap + IKeyframeStore
│  │   include/meridian/map/      #   imaplayer.hpp, isurface_map.hpp, ikeyframe_store.hpp
│  │   src/nvblox/             #   surface backend: nvblox GPU TSDF+colour+MC mesh (production)
│  │   src/cpu/               #   surface backend: host TSDF+colour+MC (dev/CI/non-CUDA)
│  │                           #     (src/vulkan/ deferred — agnostic SPIR-V backend, §12)
│  ├─ meridian_place/             # L5: IPlaceRecognizer + ScanContext++/STD/BTC/GICP/PCM
│  │
│  ├─ meridian_pipeline/          # orchestration: wires L0..L6 + threads + queues (NO ROS)
│  │                           #   the "core application object" the node drives
│  │
│  ├─ meridian_msgs/              # ROS 2 .msg/.srv: debug telemetry, optional KeyframePacket mirror
│  ├─ meridian_ros/               # THIN WRAPPER: *_node executables, msg<->core converters
│  │   src/odometry_node.cpp
│  │   src/mapping_node.cpp
│  │   src/conversions/        #   ros2core.hpp / core2ros.hpp (the ONLY translation code)
│  │   launch/  config/  rviz/
│  │
│  └─ meridian_tools/             # offline: bag→core replay harness, eval (evo), calib apps
│
├─ docs/specs/                 # this file lives here
└─ colcon.meta, .clang-format, .clang-tidy
```

### 2.1 Naming and translation-unit discipline

- One public class ≈ one header + one `.cpp`. **No 1500-line translation units** (FAST-LIO's `laserMapping.cpp` is 1055 lines doing everything; Point-LIO's is comparable — we will not repeat that). A module growing past ~600 lines is a refactor signal.
- Public headers under `include/meridian/<module>/`; private headers under `src/`. The `meridian/` prefix makes every include site self-documenting about which library it pulls.
- Interfaces are `I*.hpp`, pure-virtual, with `using Ptr = std::unique_ptr<I*>;` and a factory free function. Implementations live in `src/` subdirectories and are **not** exported in public headers — only the interface is.
- **`meridian_map`'s surface tier `ISurfaceMap` is the one backend-pluggable seam**: `src/nvblox/` (GPU, production), `src/cpu/` (host, dev/CI/non-CUDA), and a deferred `src/vulkan/` (§12). Exactly one backend is selected per build-and-run, by `map.backend`, explicitly and fail-fast (spec 06 §0, §8.1). There is **no VDBFusion/OpenVDB/NanoVDB path** and no second *tier* — the backends are interchangeable implementations of one interface, not a defensive dual map. The registration tier and `IKeyframeStore` have a single implementation each.

---

## 3. Layer responsibilities (the contracts, not the math)

Each layer is summarized by **what it owns / consumes / produces**. Math is deferred to the per-layer spec.

| Layer | Package | Consumes | Produces | Owns |
|---|---|---|---|---|
| **L0** sensor abstraction + time sync | `meridian_sensors`, `meridian_time` | raw driver buffers | time-stamped, frame-tagged measurements (`ImuSample`, `LidarScan`, `CameraFrame`, `GnssFix`) on one monotonic timeline | PTP/PPS clock model, per-sensor `ISensorSource` (one LiDAR, one IMU, one camera, GNSS), measurement aggregation |
| **L1** preprocessing | `meridian_preprocess` | L0 measurements | downsampled, validity-flagged clouds (**undeskewed**, time-sorted); rectified images + pyramid; GNSS quality verdicts | downsample (voxel grid), validity (blind radius, NaN/intensity), image pyramid, GNSS gate — deskew is **not** done here (it lives inside L2, §7) |
| **L2** front-end (odometry) | `meridian_frontend` | L1 output + IMU | high-rate `NavState` (live pose) **and** `KeyframePacket` on keyframe events | the **discrete tightly-coupled LIO** estimator (internal constant-screw deskew; voxel-hash map; GN point-to-point ICP with IMU screw prior + adaptive gravity regularization); per-axis observability scoring |
| **L3** back-end | `meridian_backend` | `KeyframePacket` (L2) + loop constraints (L5) | optimized global trajectory; `GraphUpdate` broadcast to L4/L2 | GTSAM iSAM2 graph, online extrinsic variables, GNC robust kernels, PCM gate |
| **L4** layered map | `meridian_map` | corrected poses + retained clouds + RGB | TSDF+colour + Marching-Cubes mesh behind the pluggable `ISurfaceMap` backend (**nvblox** GPU production / `cpu` host / deferred `vulkan`); **per-keyframe cloud store** | de-integration / region rebuild on loop correction |
| **L5** place recognition | `meridian_place` | keyframe descriptors + retained clouds | verified loop constraints (with covariance) → L3 | ScanContext++ → STD/BTC candidate → GICP verify → PCM consistency |
| **L6** operator surface | (`meridian_ros` + `meridian_map`) | mesh + per-vertex confidence | colour mesh, confidence overlay markers | the human-facing view |

**Cross-cutting** (`meridian_common`, `meridian_time`, `meridian_calib`, `meridian_debug`, `meridian_config`) may be used by any layer; they depend on nothing in L0–L6.

---

## 4. Dependency rules (who may depend on whom)

Dependencies form a DAG. The rule is **downward and cross-cutting only**; never upward, never sideways between sibling layer implementations.

```
        ┌─────────────────────────────────────────────┐
        │ meridian_common  meridian_time  meridian_config       │  (leaf cross-cutting:
        │ meridian_debug   meridian_calib                    │   depend on 3rd-party only)
        └───────▲───────────▲────────────▲─────────────┘
                │           │            │
   L0 meridian_sensors    L1 meridian_preprocess    L2 meridian_frontend
                │           │            │
                └─────┬─────┴──────┬─────┘
                      │            │
              L3 meridian_backend   L4 meridian_map   L5 meridian_place
                      │            │              │
                      └──────┬─────┴──────────────┘
                             │
                     meridian_pipeline   (knows all layers; wires them)
                             │
                     meridian_ros        (knows pipeline + rclcpp + meridian_msgs)
```

Normative rules:

- **R1.** Cross-cutting libs depend **only** on third-party (Eigen, Sophus, PCL, yaml-cpp, GTSAM where needed) — never on a layer.
- **R2.** A layer Lk depends on cross-cutting libs and **may** name lower layers' *value types* (which live in `meridian_common`), but **MUST NOT** depend on a sibling layer's *implementation*. L3 talks to L2 only through `KeyframePacket`, never by including `meridian_frontend/src/lio/...`.
- **R3.** Only `meridian_pipeline` may `#include` concrete factories from multiple layers (it constructs them). Layers receive collaborators by **interface pointer** via constructor injection, so they never name a concrete sibling.
- **R4.** `meridian_ros` is the **only** package allowed to depend on `rclcpp`/`*_msgs`. It depends on `meridian_pipeline` and `meridian_msgs`. Nothing depends on `meridian_ros`.
- **R5.** No cyclic build dependencies. Deskew is internal to L2 (§7), so there is no L2→L1 feedback of any kind — the build graph *and* the data flow are strictly bottom-up.
- **R6.** `meridian_map` (and only `meridian_map`) links CUDA/nvblox. CUDA is a build dependency of that layer, not a transport dependency — it does not violate R4 (§9.5).

Enforced in CI by parsing each `package.xml`/`CMakeLists.txt` and asserting the edge set (§9.4).

---

## 5. The swappable interfaces (how plug-in works)

Every layer boundary is a pure-virtual interface plus a factory. The pattern is identical everywhere; learn it once. Note the word "swap" here means **clean modularity for testing and future extension**, not a product feature-rollout: each boundary has exactly one production implementation (the LIO front-end, the iSAM2 back-end, the nvblox map). Interfaces buy us isolation, mockability, and a place to hang a test oracle — not a menu of shipping variants.

### 5.1 The pattern

```cpp
// include/meridian/frontend/ifrontend.hpp   (in meridian_frontend, NO ros)
namespace meridian {

class IFrontEnd {                                          // canonical def: spec 01 §7.3
public:
  virtual ~IFrontEnd() = default;

  // Calibration snapshot: at start + on every back-end refine (spec 01 §5.3).
  virtual void set_calibration(std::shared_ptr<const CalibrationSet> calib) = 0;

  // PRIMARY input, thread-confined to the front-end stage thread: one
  // PreprocessedGroup per sweep (the assembled MeasureGroup).
  // Triggers deskew + registration.
  virtual void ingest(const PreprocessedGroup& group) = 0;

  // High-rate IMU between sweeps: live-state propagation only, never a solve.
  virtual void ingest_imu_live(const ImuSample& imu) = 0;

  // Back-end correction feedback (loop closure / global optimization result),
  // applied by the pipeline at a safe point between ingests.
  virtual void apply_correction(const GraphUpdate& update) = 0;

  // Pull the latest high-rate estimate (live pose, for TF / control).
  virtual NavState live_state() const = 0;

  // Keyframe handoff: invoked whenever a keyframe is born. The ONE thing L2 hands L3 (§6).
  using KeyframeSink = std::function<void(KeyframePacket&&)>;
  virtual void set_keyframe_sink(KeyframeSink sink) = 0;

  // Introspection pull; live telemetry flows through the sink given at construction (§10).
  virtual FrontEndDiagnostics diagnostics() const = 0;
};

// Factory selects the implementation by name from config (§8). Default: "lio".
// Telemetry may be null (NullSink semantics), matching every other module factory.
// `deterministic` is a no-op for the synchronous LIO front-end (§11.2); the
// parameter stays so future implementations with internal parallelism honour it.
std::unique_ptr<IFrontEnd> makeFrontEnd(const FrontendConfig& cfg,
                                        std::shared_ptr<const CalibrationSet> calib,
                                        TelemetrySink* telemetry, bool deterministic);

} // namespace meridian
```

The **factory free function** is the only place that names concrete implementations. `meridian_pipeline` calls `makeFrontEnd(cfg.frontend, calib, sink, deterministic)` and gets back a `std::unique_ptr<IFrontEnd>`. The production path is `FrontEndKind::Lio` (`kind: lio`), today the only accepted value (the former `iekf_oracle` differential-test entry and the `ct_livo` continuous-time estimator are both retired, §5.4).

### 5.2 The interface roster

| Interface | Header | Key methods | Role / notes |
|---|---|---|---|
| `ISensorSource` | `meridian/sensors/isensor_source.hpp` | `onSample(cb)` | swap: live driver, ROS-bag replay, simulator (same code path for live & offline) |
| `IFrontEnd` | `meridian/frontend/ifrontend.hpp` | `ingest(PreprocessedGroup)`, `ingest_imu_live`, `set_keyframe_sink`, `apply_correction` | **production: discrete LIO** (`src/lio/`), the only impl (the iEKF test oracle and the CT estimator are retired, §5.4) |
| `IBackEnd` | `meridian/backend/ibackend.hpp` | `add_keyframe(KeyframePacket&&)`, `add_loop_constraint`, `add_absolute`, `optimize()→GraphUpdate`, `corrected_trajectory`, `refined_calibration`, `diagnostics`, `write_g2o` (canonical: 01 §7.4) | iSAM2 (production); batch LM available for offline debugging only |
| `IMapLayer` | `meridian/map/imaplayer.hpp` | `integrate(kf)`, `deintegrateRegion(aabb)`, `query`, `extractMesh` | façade over Tier R + `ISurfaceMap` + store; surface backend pluggable (**nvblox** GPU / `cpu` host / deferred `vulkan`), one selected per run. Seam left for a deferred ESDF layer (§12) |
| `IKeyframeStore` | `meridian/map/ikeyframe_store.hpp` | `put(id,cloud,rgb,pose)`, `get(id)`, `clouds(region)` | RAM store (production); mmap'd on-disk store is a future option behind the same seam |
| `IPlaceRecognizer` | `meridian/place/iplace_recognizer.hpp` | `add(kf)`, `query()→candidates`, `verify()→LoopConstraint` | ScanContext++ → STD/BTC → GICP → PCM |
| `TelemetrySink` | `meridian/debug/telemetry.hpp` | `timing/scalar/vec/cloud/pose/marker/event` | ros, recording (tests), null (off) |

Each interface lives in its **owning layer's package** so the contract and its implementation ship together; the *value types* they exchange (`KeyframePacket`, `NavState`, `GraphUpdate`, `LoopConstraint`) live in `meridian_common`, so no layer needs to depend on another's package to name them.

### 5.3 Why interfaces, not templates

Layer boundaries use **runtime polymorphism (virtual)**, not templates/CRTP, deliberately: boundaries are crossed a few hundred times per second at most (keyframes, scans), so virtual-call cost is irrelevant, while the benefits — separate compilation, mockable in tests, no template error walls, a place to hang the test oracle — are large. Inside a layer's hot loop (e.g. the per-point residual, the GPU kernels) we use concrete types, templates, and CUDA freely. The abstraction tax is paid only where it buys modularity.

### 5.4 Retired front-ends (the seam outlived its first two occupants)

Two earlier `IFrontEnd` implementations have been removed from the codebase: a FAST-LIO2-style iterated-EKF LIO (`src/iekf/`, a differential-test oracle only, never a product path) and the continuous-time LIVO+GNSS estimator (`src/ct/`, `frontend_kind = 1`), retired after it hit a measured trajectory-bandwidth wall under high-frequency motion. The default and only front-end is the **discrete LIO estimator** (`src/lio/`, `frontend_kind = 2`); the `IFrontEnd` seam the predecessors validated remains the extension point. (This is the *one* paragraph the history gets; everything else in this document assumes the LIO front-end.)

---

## 6. The L2→L3 contract: `KeyframePacket`

The single most important boundary in the system is what the front-end hands the back-end. **The `KeyframePacket` is the only value that crosses L2→L3.** It is defined concretely so the contract is auditable and the back-end is fully decoupled from how the front-end produced the estimate. (Full field-level treatment is in `01_interfaces_and_data_types.md` §6; this section states the architectural invariants.)

### 6.1 Definition

```cpp
// include/meridian/common/keyframe_packet.hpp   — canonical def: spec 01 §6.1
namespace meridian {

struct KeyframePacket {
  // identity / time
  std::uint64_t   id    = 0;                 // monotonic keyframe id (graph node key)
  Timestamp       stamp = 0;                 // int64 ns, a real measurement instant

  // pose (REQUIRED)
  Frame           ref_frame = Frame::Odom;   // frame T_ref_body is expressed in
  Pose            T_ref_body;                // pose of the estimation frame F_e in ref_frame

  // kinematic state (OPTIONAL, gated; true only on the restart fallback)
  bool            kinematics_included = false;
  Eigen::Vector3d v_ref = Eigen::Vector3d::Zero();   // [m/s]    valid iff flag
  Eigen::Vector3d b_g   = Eigen::Vector3d::Zero();   // [rad/s]  valid iff flag
  Eigen::Vector3d b_a   = Eigen::Vector3d::Zero();   // [m/s^2]  valid iff flag

  // uncertainty: ONE block, ONE constraint per interval
  enum class ConstraintKind { RelativeBetween, AbsolutePrior, ImuPreintegration }
                  constraint_kind = ConstraintKind::RelativeBetween;
  std::uint64_t   rel_to_id = 0;             // RelativeBetween/ImuPreintegration: previous keyframe id
  Pose            T_relto_this;              // RelativeBetween: the relative transform
  GaussianBlock<6> constraint_cov;           // 6-DoF block, ROTATION-FIRST [rx,ry,rz,tx,ty,tz]
                                             // (GTSAM Pose3 boundary; the one rotation-first block)

  // observability (REQUIRED) — translation-first scores in a named frame
  ObservabilityReport observability;

  // data (REQUIRED), shared-immutable, no copy
  std::shared_ptr<const std::vector<LidarPoint>> cloud_body;  // deskewed, body frame at stamp
  std::shared_ptr<const CameraFrame>             image;       // null if no cam at this KF
  Pose            T_body_cam;                // extrinsic snapshot for colourisation

  // restart-fallback IMU summary (only when constraint_kind == ImuPreintegration)
  std::optional<ImuPreintegrationSummary> imu_summary;

  // provenance
  std::uint32_t   calib_version = 0;         // CalibrationSet snapshot that produced this
  std::uint32_t   frontend_kind = 1;         // 1 = retired CT front-end, 2 = LIO (diagnostics only; do not branch)
};

} // namespace meridian
```

Canonical field-level definition: spec 01 §6.1.

### 6.2 Time model

`Timestamp` is **`int64` nanoseconds** on one monotonic timeline established by L0/PTP. There is no `ros::Time` below the wrapper. The per-point LiDAR sample times, the camera mid-exposure time, and the front-end's deskew interval (§7) all use this scalar. The wrapper converts `rclcpp::Time` ↔ `Timestamp` in exactly one place (`conversions/`). This mirrors how FAST-LIO threads a single `double lidar_end_time` through everything (`laserMapping.cpp:593`) — we keep the single-scalar-time discipline but make it integer-ns to avoid double-precision drift at long uptime.

### 6.3 One-constraint rule (kill double-counting)

A naïve handoff double-counts information by simultaneously passing (a) an absolute marginal prior, (b) a relative between-factor, **and** (c) an IMU-preintegration factor derived from *the same* measurements. Meridian picks **one** clean contract per interval:

> **Default and recommended:** `constraint_kind = RelativeBetween`. The packet carries a single relative pose `T_relto_this = T(ref)^{-1} · T(this)` with its marginal covariance, and L3 inserts exactly **one** `BetweenFactor` between consecutive keyframes. Velocity and biases **do not** cross as live optimization variables in the default path (`kinematics_included = false`); they ride along only as *seeds / telemetry*, not as graph variables. The front-end has already fused IMU+LiDAR over the interval, so that information is *in* the relative covariance.
>
> **Mutually-exclusive fallback:** `constraint_kind = AbsolutePrior` is used **only to root the chain**: on the run's first keyframe and on the re-anchor after a reseed (§7.4), when no valid relative pose spans the gap. Then the packet carries the absolute pose with a priced covariance, L3 inserts a prior factor, and **no `BetweenFactor`** is added for that interval. The two are never both present for the same interval. (`ImuPreintegration` remains in the enum for a future producer; the LIO front-end never emits it.) Bias estimation otherwise lives in L2.

So for any keyframe interval the back-end receives **exactly one** geometric constraint. That is the line that keeps the information budget correct. (See `05_backend_graph.md` for GTSAM factor construction and how `constraint_cov` and `observability` assemble the noise model — `observability` inflates the diagonal of weakly-observed axes, X-ICP / D²-LIO style, instead of a binary degeneracy switch.)

### 6.4 Tangent ordering and frames

The `GaussianBlock<6> constraint_cov` uses the ordering **`[rotation(x,y,z), translation(x,y,z)]`** (ROTATION-FIRST) in the *body* tangent space at `T_ref_body`, expressed in `ref_frame`. This is the **one** rotation-first block in the system — it matches the GTSAM `Pose3` convention at the L2→L3 boundary, and the front-end's translation-first marginal is reordered exactly once when packing the packet. Every other Meridian core type (`Pose`, `NavState`, `observability`) is **translation-first** `[tx,ty,tz,rx,ry,rz]` per spec 01 §3. `ref_frame` is a `Frame` enum (`Frame::Odom`, `Frame::Map`); the back-end owns the `odom→map` relationship. This explicitness kills the classic "is this covariance in world or body, rotation-first or translation-first?" bug. (FAST-LIO hand-packs the EKF `P` into `pose.covariance` with a rotation/translation block-swap, `laserMapping.cpp:597-606` — easy to get wrong precisely because the convention is implicit. We make it explicit and typed.)

### 6.5 Shared-immutable cloud, not cloud copy

The packet carries `std::shared_ptr<const std::vector<LidarPoint>> cloud_body` and `std::shared_ptr<const CameraFrame> image` **directly** — not an opaque id. The packet never copies the point cloud across the thread boundary: the cloud lives once, shared-immutable, and the same `shared_ptr` bytes are handed to the **`IKeyframeStore`**, so **L4** (nvblox re-integration on loop correction), **L5** (GICP verification), and an optional offline export pass all read the very same buffer with no copy. (Full treatment in `06_mapping.md`.)

---

## 7. Bootstrap, deskew, and recovery

### 7.1 Where deskew lives (L2-internal)

L1 hands up **undeskewed, time-sorted** scans with per-point sample times; the front-end warps each point to the sweep-end instant with a **constant-screw model** (interval-mean IMU angular rate plus the propagated body-frame velocity) before registration. Deskew is therefore an L2-internal step, not a layer boundary: L1 never needs a trajectory, there is no deskew-provider interface, and no L2→L1 feedback edge exists — the build graph *and* the data flow are strictly bottom-up. (Math in `04_frontend_estimation.md`.)

### 7.2 Cold-start (static initialization inside the front-end)

On startup the front-end holds incoming groups internally and runs a **static initialization** over the first `frontend.lio.init_stationary_s` of standstill IMU: gravity from the mean specific force, gyro and accelerometer biases at rest. When initialization completes, the held groups are drained in arrival order and normal processing begins. The pipeline does no initialization of its own — L1 forwards every group immediately — and the completion is exposed as a telemetry `event` (`frontend/lio/init_done`) so the transition is visible.

### 7.3 No feedback edge

There is nothing to wire back: the only L3→L2 data path is the `GraphUpdate` correction snapshot (§11.2), and the only L1→L2 currency is the `PreprocessedGroup`. Any module that wants a pose at "now" reads `IFrontEnd::live_state()`.

### 7.4 Gap handling and the reseed fallback

If the inter-group gap exceeds `frontend.lio.max_gap_s`, the state bridges the hole on **constant velocity**, the map is kept, and a reseed is armed. If the first post-gap registration then fails its gates (keypoint floor, zero correspondences, non-convergence), the front-end **clears the map and re-anchors** on the propagated pose; the next keyframe carries `constraint_kind = AbsolutePrior` (§6.3) priced by the first converged covariance inflated by `frontend.lio.reseed_cov_inflation`, because no valid relative pose spans the gap. A steady-state registration failure (no armed reseed) simply drops the sweep and keeps the map — FAST-LIO's analogue is bailing with `ekfom_data.valid = false` when `effct_feat_num < 1` (`laserMapping.cpp:708-712`). Every path is published on the debug bus (§10) as a `frontend/lio/{gap,reject,reseed}` event so an operator sees exactly when and why recovery happened.

---

## 8. Configuration / parameter strategy

### 8.1 One typed tree, two loaders

There is **one** configuration type, `meridian::Config`, a nested struct tree of plain fields (no `std::any`, no string-keyed maps in the hot path), populated by either:

- **YAML loader** (`meridian_config`, yaml-cpp) — used for bag replay, tests, and as the *source of truth* file checked into `meridian_ros/config/`.
- **ROS 2 parameter loader** (`meridian_ros`) — declares the same keys as ROS parameters and fills the same `Config` struct, reusing the YAML key paths so the two never drift.

The core only ever sees a fully-populated `Config`. This avoids FAST-LIO's pattern of scattering ~40 `nh.param<...>("preprocess/blind", p_pre->blind, 0.01)` reads through the node body (`laserMapping.cpp:761-793`); instead the schema is one declarative thing.

### 8.2 Schema shape (single LiDAR + IMU + camera + GNSS, LIO front-end, nvblox map)

```yaml
meridian:
  pipeline:   { mode: live|replay, threads: { frontend: 1, backend: 1, map: 1 },
                queue: { meas_capacity: 8, kf_capacity: 64, map_capacity: 64 } }  # BoundedQueue sizes (§11.1.1)
  time:       { source: ptp|pps|host, max_skew_ms: 5 }
  sensors:
    lidar:    { topic: /os/points, model: ouster_os0_128,
                extrinsic_T: [..], extrinsic_R: [..] }     # single LiDAR
    imu:      { topic: /os/imu, rate_hz: 200, cov_acc: .., cov_gyr: ..,
                b_acc_cov: .., b_gyr_cov: .. }              # single IMU (estimation frame)
              # cov_acc/cov_gyr and b_acc_cov/b_gyr_cov are SQUARED continuous-time
              # noise densities — i.e. variances ((m/s^2)^2, (rad/s)^2, and the bias
              # random-walk variances), NOT standard deviations. calibration_from_config
              # takes the sqrt of each to fill the CalibrationSet noise-density fields.
    camera:   { topic: /cam0, model: pinhole, intrinsics: [..], extrinsic: [..],
                photometric: { exposure_comp: true } }      # single camera (passthrough)
              # extrinsic is [tx,ty,tz,qx,qy,qz,qw] = T_imu_cam (camera optical frame
              # -> body/IMU). The estimator does not consume the camera; the extrinsic
              # rides on each KeyframePacket as the T_body_cam snapshot (colourisation).
    gnss:     { topic: /gnss/fix, enable: true }
  preprocess: { blind: 0.5, point_filter_num: 3, voxel_surf_m: 0.5 }
  frontend:   { kind: lio,                                  # the discrete LIO estimator
                lio: { voxel_size_m: 1.0, max_points_per_voxel: 20, max_range_m: 100.0,
                       keypoint_voxel_factor: 1.5, min_keypoints: 30,
                       icp_max_iterations: 100, convergence_eps: 1.0e-5,
                       max_corr_dist_m: 0.5, min_beta: 200.0, max_expected_jerk: 3.0,
                       init_stationary_s: 1.0, max_gap_s: 0.5, reseed_cov_inflation: 100.0 },
                keyframe: { dist_m: 1.0, rot_deg: 10, time_s: 1.0 } }
  backend:    { kind: isam2, relinearize_thresh: 0.1, robust: huber }   # committed incremental loop kernel; GNC runs off-thread (spec 05 §8)
  map:        { backend: nvblox, tsdf_voxel_m: 0.05, mesh: marching_cubes,
                colour: true }                              # nvblox|cpu|vulkan; explicit, fail-fast
  place:      { kind: scan_context_pp, pcm: true, gicp_fitness_max: 0.3 }
  debug:      { level: info, publish_clouds: true, publish_markers: true,
                timing: true, telemetry_rate_hz: 10 }
```

### 8.3 Rules

- **Validated on load.** `Config::validate(std::string* error)` checks units, ranges, and cross-field consistency (e.g. `tsdf_voxel_m ≤ preprocess.voxel_surf_m`, `frontend.lio.voxel_size_m > 0`, every `pipeline.queue.*_capacity > 0` per §11.1.1), returning `false` with a precise message on the first violation; `load_config_yaml()` calls it and throws, so a bad config fails fast — no silent defaults masking a typo (a real FAST-LIO foot-gun: a mistyped `nh.param` key silently uses the default).
- **`kind` strings select implementations** through the factories (§5.1). `frontend.kind` accepts only `lio` (the former `ct_livo` and `iekf_oracle` values are retired and rejected, §5.4); `backend.kind` is `isam2`. `map.backend` selects the surface backend — `nvblox` (GPU, production), `cpu` (host, dev/CI/non-CUDA), or deferred `vulkan` — and must name a backend compiled into this build (spec 11 §7.6); an uncompiled or misspelled backend fails validation, and there is never a silent substitution between backends (spec 06 §0, §8.1).
- **Immutable after start.** `Config` is `const` once the pipeline is built. Runtime-tunable knobs (only debug verbosity and publish toggles) go through a separate `DebugControl` service (§10.5), never by mutating `Config`.

---

## 9. Build system

### 9.1 colcon + ament_cmake + C++ standard

- Built with **colcon**; every package is `ament_cmake` on **ROS 2 Humble**. One command (`colcon build`) covers core libs, wrappers, and tests.
- **C++20** is the standard. Set `CMAKE_CXX_STANDARD 20`, `CMAKE_CXX_STANDARD_REQUIRED ON`, `CMAKE_CXX_EXTENSIONS OFF` (concepts to constrain interfaces, `std::span` for zero-copy point views, designated initializers for `Config`). FAST-LIO ships C++14 catkin CMake; we modernize.

### 9.2 Targets per package

Each core package exports a single library target with a namespaced alias and a clean public include dir:

```cmake
# meridian_frontend/CMakeLists.txt (sketch)
add_library(meridian_frontend SHARED
  src/frontend_factory.cpp
  src/lio/voxel_grid_map.cpp
  src/lio/imu_tracker.cpp
  src/lio/scan_registration.cpp
  src/lio/lio_frontend.cpp)
target_include_directories(meridian_frontend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(meridian_frontend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(meridian_frontend PUBLIC cxx_std_20)
target_link_libraries(meridian_frontend PUBLIC
  meridian_common::meridian_common meridian_config::meridian_config
  meridian_debug::meridian_debug meridian_calib::meridian_calib
  Eigen3::Eigen Sophus::Sophus)  # NO rclcpp — enforced (§9.4)
ament_export_targets(meridian_frontendTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_common meridian_config meridian_debug meridian_calib
                          Eigen3 Sophus)
```

The C++20 flags, the `Release`-when-unset build type, workspace-wide
position-independent code, and the warning policy come from `MeridianToolchain.cmake`
(via `find_package(meridian_cmake)` + `include(MeridianToolchain)`); the imported
targets above are discovered once in `MeridianDeps.cmake`. See spec 11 §§5–8 for the
full templates.

CUDA is opt-in **per hot-loop library**: only `meridian_map` links CUDA/nvblox. The front-end is deliberately **sequential** — no OpenMP, no TBB anywhere under `src/lio/` — so two identical runs of the estimator are bit-identical (§11.2); we keep accelerator flags library-local so non-parallel modules stay deterministic.

### 9.3 Two binaries, both thin

`meridian_ros` builds the wrapper executables (`odometry_node`, `mapping_node`) and the `conversions` translation library. Each `*_node.cpp` is ~100 lines: declare params → build `Config` → construct `MeridianPipeline` → subscribe (convert msg→core, push into pipeline) → register publish callbacks (core→msg). All algorithm lives below. `meridian_tools` builds the offline `replay` binary that drives the *same* `MeridianPipeline` from a bag through `ISensorSource`, proving the core runs identically off-ROS.

### 9.4 CI gates (enforce the architecture)

- **No-ROS gate:** a grep over `src/meridian_*` excluding `meridian_ros` for `rclcpp | ros/ros.h | sensor_msgs | rclcpp/clock` must return empty.
- **Dependency lint:** a script asserts the §4 edge set and fails if any `meridian_<layer>/package.xml` lists `rclcpp`/`*_msgs`, or if a layer `#include`s a sibling's `src/`.
- **Surface-backend gate:** assert every `ISurfaceMap` implementation under `meridian_map` lives in its own backend directory (`src/nvblox/`, `src/cpu/`, and a future `src/vulkan/`) and that no source under `meridian_map` references VDBFusion/OpenVDB/NanoVDB — the backends are interchangeable implementations of one interface, not a smuggled-in second mapping library.
- **No-grounding-in-code gate:** assert no source under `src/meridian_*` contains a reference dossier pointer (regex `grounding[ /][0-9]`, catching both `grounding/NN` and `grounding NN`), enforcing the self-contained-comment rule. Regex owned by spec 11 §9.3.
- **Sequential-front-end gate:** a grep over `meridian_frontend/src/lio/` for `omp|tbb|std::thread|std::async` must return empty — the estimator is single-threaded by contract (§11.2).
- **clang-tidy / clang-format** on every TU; warnings-as-errors in core.
- **Unit + replay tests** under `colcon test`, including the LIO synthetic-world suite (tracking bound, covariance-chain Monte-Carlo, packet-stream contract) and the bit-identical two-run replay check (spec 04, spec 10). Per-module testing detailed in `02..10`. CI builds the full GPU configuration (nvblox backend) and the no-CUDA configuration (`MERIDIAN_MAP_NVBLOX=OFF`, which keeps the `cpu` backend so the dev-box build has a working map, §9.5), plus the map-excluded build (`MERIDIAN_WITH_MAP=OFF`); the surface-backend conformance suite (spec 06 §8.4) runs per compiled backend so the off-device path stays green.

### 9.5 CUDA / nvblox is core, not wrapper; the surface backend is selectable

Meridian targets **Jetson Orin** and assumes a CUDA GPU is **always present** on the deployed device, where the surface map runs the **nvblox GPU backend**. nvblox (TSDF + colour + Marching-Cubes mesh) runs entirely on the GPU and lives in `meridian_map`, a *core* package — it is built and tested without any ROS. CUDA being a dependency of the nvblox backend (and of any GPU residual work in `meridian_frontend`) does **not** violate the no-ROS rule (§1.1): that rule is about *middleware*, not about the accelerator.

**The surface tier is one interface (`ISurfaceMap`) with selectable backends** (spec 06 §0, §8.1): `nvblox` (GPU, the production/on-device backend), `cpu` (a host TSDF+colour+mesh of the same contract, for the dev Mac / a non-NVIDIA box / CI / the cross-backend correctness oracle), and a deferred `vulkan` (agnostic SPIR-V). Backend choice is **explicit and fail-fast, never silent**: `map.backend` names it, it must be compiled into the build, and a `nvblox` deployment **still hard-fails on a missing GPU** — the system never downgrades itself from `nvblox` to `cpu`. The `cpu` backend is a *deliberately chosen* peer backend, not a runtime fallback, so a deployed robot can never quietly run a degraded map while believing it is on nvblox.

**Build-time backend availability is governed by compile flags** (spec 11 §7.6): `MERIDIAN_WITH_MAP` (default `ON`) builds the map layer at all; within it `MERIDIAN_MAP_NVBLOX` (default `ON`, requires CUDA+nvblox) compiles the GPU backend, while the `cpu` backend always compiles when the map is built (no special toolchain). A no-CUDA developer box therefore builds with `MERIDIAN_MAP_NVBLOX=OFF` and gets a **working `cpu` map**, not an absent one — and can compile and unit-test L0–L6 end to end on a laptop. `MERIDIAN_WITH_MAP=OFF` remains available to drop the map entirely (a binary with no `IMapLayer` at all) for minimal non-map builds. Spec 11 §7 owns the CMake mechanics.

---

## 10. Debug / introspection strategy (a first-class subsystem)

This is where Meridian most deliberately *improves on* FAST-LIO. FAST-LIO's introspection is real but ad-hoc and entangled with the node: it advertises a fixed set of topics inline (`/cloud_registered`, `/cloud_registered_body`, `/cloud_effected`, `/Laser_map`, `/Odometry`, `/path` — `laserMapping.cpp:849-860`); it stuffs the EKF covariance into the odometry message by hand (`odomAftMapped.pose.covariance[i*6+...] = P(k,...)`, `laserMapping.cpp:597-606`); it computes the inlier ("effective") set and residuals in `h_share_model` but only optionally publishes the cloud, with the count left in a global (`effct_feat_num`, `res_mean_last`, `laserMapping.cpp:695-715`); and it logs timing/state to `fout_*` files and `printf` (`laserMapping.cpp:835-838, 991-1013`). Good signals, wrong location. Meridian keeps **every one of those signals** but routes them through a dedicated telemetry subsystem so the *core* emits structured data and the *wrapper* decides how to surface it.

### 10.1 Architecture: a ROS-agnostic telemetry bus

`meridian_debug` defines `TelemetrySink`, a pure interface the core writes to and the wrapper implements:

```cpp
// include/meridian/debug/telemetry.hpp  (NO ros)
namespace meridian {
class TelemetrySink {
public:
  virtual ~TelemetrySink() = default;
  virtual void timing(const char* stage, double ms)              = 0;  // per-stage timer
  virtual void scalar(const char* key, double v, Timestamp t)    = 0;  // residual, #eff pts…
  virtual void vec(const char* key, const VecX&, Timestamp t)    = 0;  // bias, observability[6]…
  virtual void cloud(const char* key, const PointCloud&, Frame, Timestamp) = 0;
  virtual void pose (const char* key, const Pose&,        Frame, Timestamp) = 0;
  virtual void marker(const Marker&, Timestamp)                  = 0;  // geometric overlays
  virtual void event(Level, const char* tag, const std::string&, Timestamp) = 0;
};
// RAII scoped timer: calls sink->timing(stage, elapsed) on destruction.
struct ScopedTimer { ScopedTimer(TelemetrySink*, const char* stage); ~ScopedTimer(); };
} // namespace meridian
```

The core never knows whether telemetry becomes a ROS topic, a CSV, or `/dev/null`. The wrapper binds a `RosTelemetrySink` mapping `cloud→PointCloud2`, `pose→Odometry/TF`, `marker→visualization_msgs/MarkerArray`, `scalar/vec→meridian_msgs/Telemetry`, `timing→meridian_msgs/StageTiming`. Tests bind a `RecordingSink` that captures everything for assertions. The default `NullSink` makes debug truly zero-cost when off.

### 10.2 What to publish — grounded in FAST-LIO and improved

| Signal | FAST-LIO analogue | Meridian: channel & key (improvement) |
|---|---|---|
| Registered cloud (world) | `/cloud_registered` (`:849`, built in `publish_frame_world` `:478`) | `cloud("map/registered")` → PC2, best-effort QoS, rate-limited |
| Body cloud | `/cloud_registered_body` (`:851`, `publish_frame_body` `:532`) | `cloud("body/scan")` |
| **Correspondences used in update** | `/cloud_effected` (`:853`; count in `effct_feat_num` `:695`) | `scalar("frontend/assoc/n_attempted")` **+** `scalar("frontend/assoc/n_matched")` — FAST-LIO published only the cloud (commented out at the call site, `:983`); we publish the *counts* so degeneracy is **plottable**, not just visible |
| Live odometry + covariance | `/Odometry` with hand-packed `pose.covariance` (`:597-606`) | `pose("odom/body")`; the 6×6 covariance crosses typed on the `KeyframePacket` (§6.4), not smuggled into the pose message |
| Trajectory | `/path` (`:859`, throttled `jjj % 10`) | `pose("frontend/path_sample")` history aggregated by the wrapper into `nav_msgs/Path` |
| **Solver health** | none (printf timing only) | `scalar("frontend/solver/gn_iters")` + `scalar("frontend/solver/dx_norm")` + `scalar("frontend/solver/chi")` — the GN ICP's convergence trace, plottable per sweep |
| **LIO internals** | none | `scalar("frontend/lio/{beta, accel_var, n_corr, deskew_span_t_ms}")` + `event("frontend/lio/{init_done, gap, reject, reseed}")` — gravity-regularizer weight, interval accel variance, deskew span, and every recovery path made visible. Strict new signal. |
| **Map health** | none | `scalar("frontend/map/voxels")` + `scalar("frontend/map/points")` — local-map growth and clipping, cheap and always on |
| **Per-axis observability** | none (only the binary `flg_EKF_inited`) | `vec("frontend/obs")` + `scalar("frontend/obs_min")` — the 6 scores driving back-end noise; rendered as a 6-bar rviz marker (§10.4). Strict upgrade. |
| Per-stage timing | `aver_time_match/solve/icp/...` to `printf`+CSV (`:991-1009`) | `ScopedTimer` on every stage → `StageTiming` topic: `preprocess`, `frontend.lio.ingest`, `backend.optimize`, `map.integrate`, `mesh.extract`. Live breakdown, not a logfile. |
| State/bias trace | `dump_lio_state_to_log` (`:150`) | `scalar("frontend/state/vel_norm")`, `scalar("frontend/state/bias_gyr_norm")`, `scalar("frontend/state/bias_acc_norm")` |
| Back-end health | none | `scalar("backend/chi2")`, `scalar("backend/n_factors")`, `event("backend/relinearize")` |
| Loop closure | none | `marker("place/loop_edge")` (line between matched keyframes, coloured by GICP fitness) + `event("place/loop_accepted|rejected_pcm")` |
| Map de-integration | none | `event("map/region_rebuild")` + `marker("map/dirty_region")` AABB |
| Recovery | none (silent `continue` / `ROS_WARN`) | `event(WARN, "frontend/lio/reseed", reason)` — the §7.4 fallback is **visible** |

### 10.3 Structured logging

`meridian::log` is a thin macro layer (`MERIDIAN_INFO(...)`, `MERIDIAN_WARN(...)`) forwarding to a `LogSink`. The wrapper binds it to `rclcpp` logging; tests to a buffer; offline to stdout+file. Logs are **structured** (key=value), greppable and parseable, with module tag and `Timestamp` always present. The core never calls `RCLCPP_INFO` or `ROS_WARN`.

### 10.4 rviz markers (geometric introspection)

The wrapper turns `marker(...)` calls into a `MarkerArray`. Standard markers Meridian ships:

- **Observability hexagon:** 6 bars (tx,ty,tz,rx,ry,rz) scaled by `observability.score`, coloured green→red, anchored at the body — the operator instantly sees which axes are weakly observed (a long corridor → weak forward translation).
- **Loop edges:** lines between keyframe centroids for accepted/rejected loops (PCM result coloured).
- **Dirty-region AABB:** the voxel region L4 (nvblox) is rebuilding after a loop correction.
- **Confidence overlay (L6):** the mesh tinted by per-vertex confidence (TSDF weight + pose covariance).

### 10.5 Control: toggling debug at runtime

Because `Config` is immutable (§8.3), debug is toggled through a `DebugControl` interface the wrapper exposes as a ROS 2 service / parameter callback: set telemetry rate, enable/disable per-key publishing, raise/lower log level. This keeps heavy clouds off the wire in production while letting an operator turn them on for a forensic session without restarting.

### 10.6 Cost discipline

Telemetry is **pull-rate-limited and sink-gated**: a `scalar` call when the sink is `NullSink` is one virtual call to an empty body (inlinable to nothing in release). Clouds and markers are gated by `debug.publish_*` plus a token-bucket rate limiter in the wrapper, so introspection never starves the estimator. `ScopedTimer` reads a steady clock twice — negligible. Principle: **introspection is always wired, cheap when off, rich when on.**

---

## 11. Threading model

### 11.1 Stages and queues

Meridian uses a **small fixed set of stage threads connected by bounded multi-producer/single-consumer queues** — not a thread pool, not a task graph — because the data flow is a near-linear pipeline and predictability beats cleverness for a real-time estimator. (FAST-LIO runs single-threaded in `main()` with a `mutex` + `condition_variable` guarding the sensor buffers, `laserMapping.cpp:81-82, 281`; we make the stage boundaries explicit instead of implicit in one loop.)

Every inter-stage edge is one concrete primitive — `BoundedQueue<T>` — and stages exchange nothing else. There is no second queue type and no ad-hoc `mutex`+`condition_variable` per edge; the policies in §11.2 are properties of this one primitive, configured per edge, so teardown and overflow behave identically everywhere.

```
[ROS sub callbacks]            (rclcpp executor threads; wrapper only)
   │  convert msg→core, push (non-blocking)
   ▼
(Q_sensors) ─▶ [L0/L1 stage: time-sync + preprocess]              thread T1
                   │ synced measurements (1 LiDAR, IMU, cam, GNSS)
                   ▼
              (Q_meas) ─▶ [L2 LIO front-end]                      thread T2 (hot)
                              │ NavState  ──────────────▶ TF/odom publish (low latency)
                              │ KeyframePacket (finished synchronously, §11.5)
                              ▼
                         (Q_kf) ─▶ [L3 back-end + L5 place-rec]     thread T3
                                       │ GraphUpdate ──▶ feedback to L2 & L4
                                       ▼
                                  (Q_map) ─▶ [L4 nvblox integrate (GPU)] thread T4
                                                │ mesh extract (on demand, GPU) thread T5
                                                ▼
                                           [L6 surface publish]
```

### 11.1.1 The queue primitive: `BoundedQueue<T>`

All inter-stage edges are instances of a single capacity-bounded, blocking-capable, move-only queue. It is the only concurrency primitive in the pipeline; a stage's interaction with the rest of the system is fully described by which `BoundedQueue<T>` it pops from and which it pushes to.

```cpp
// include/meridian/common/bounded_queue.hpp   (NO ros)
namespace meridian {

template <typename T>
class BoundedQueue {
public:
  explicit BoundedQueue(std::size_t capacity);   // fixed for the queue's life (0 clamps to 1)

  // Non-blocking push; false if the queue is full or closed.
  bool try_push(T&& v);

  // Lossy producer (Q_sensors): never blocks. If full, evicts the OLDEST element to
  // make room and enqueues `v` (no-op when closed). Returns the number of elements
  // dropped (0 or 1); the caller emits it as the q_*_dropped telemetry scalar (§10).
  std::size_t push_or_drop_oldest(T&& v);

  // Lossy producer with an eviction preference (Q_meas): when full, evicts the oldest
  // element for which `protect` returns false; only if EVERY queued element is
  // protected does it fall back to evicting the oldest outright (so it can never
  // refuse to make room). The new element is always enqueued (unless closed); the
  // returned `evicted` element lets the caller report which kind was lost.
  struct ProtectedPush { bool enqueued; std::optional<T> evicted; };
  ProtectedPush push_protecting(T&& v, const std::function<bool(const T&)>& protect);

  // Lossless producer (Q_kf, Q_map): blocks while the queue is full and
  // not closed. Returns true once enqueued; false iff the queue was closed while
  // waiting (the item is NOT enqueued in that case).
  bool push_blocking(T&& v);

  // Consumer: blocks while empty and not closed. Returns false iff the queue is
  // closed AND drained, so a worker loop is simply `T v; while (q.pop(v)) { ... }`
  // and exits cleanly on teardown.
  bool pop(T& out);
  bool try_pop(T& out);              // non-blocking; false if empty

  // Idempotent teardown signal. Wakes ALL blocked producers and consumers
  // (broadcast). After close(): pushes fail (false / no enqueue), pop drains the
  // remaining elements then returns false. The single signal the pipeline uses to
  // stop a stage; no stage spins or polls a flag.
  void close();

  std::size_t size() const;          // snapshot, for the queue-depth telemetry (§10)
  bool closed() const;
};

} // namespace meridian
```

Mechanics that the rest of §11 relies on:

- **One mutex + two condition variables per queue (data-ready, space-free), never sleep-polling.** `pop` and a full `push_blocking` wait on them; producers/consumers/`close` notify. Safe for multiple producers and one consumer. A stage thread that has no work consumes no CPU. This replaces FAST-LIO's single hand-rolled `mutex`+`condition_variable` over shared buffers with one audited, reused type.
- **Capacity is fixed at construction** from `pipeline.queue` config (§8) and never resized at runtime; an overrun is a policy decision (drop vs. block), not an allocation.
- **`close()` broadcast is the teardown contract.** `MeridianPipeline` destruction (§11.3) calls `close()` on every queue in reverse dependency order; each stage's `while (q.pop(v))` loop then returns `false` and the thread joins. No queue is destroyed while a thread still blocks on it.
- **Overflow policy is chosen at the call site, not baked into the type.** A lossy edge calls `push_or_drop_oldest` (or `push_protecting` when some element kinds must outlive others); a lossless edge calls `push_blocking`. The same `BoundedQueue<T>` serves all, which is why §11.2's per-edge policy table is just a statement of *which method each edge's producer calls*.

### 11.2 Rules

- **Each stage is single-threaded internally** (its module is *not* required to be thread-safe; it is *thread-confined*). Concurrency lives only in the queues. The simplest model that keeps the front-end deterministic and easy to reason about. A stage MAY own internal parallelism *behind* its interface — nvblox launches GPU kernels from its single map thread — provided its external surface stays thread-confined and determinism mode (below) collapses it all onto one thread. The LIO front-end owns **no** internal parallelism: deskew, association, and the GN solve run sequentially on T2.
- **Queues are bounded** with an explicit per-edge overflow policy realised by the `BoundedQueue<T>` method the producer calls (§11.1.1). The sensor-side edges are **lossy** under overload (real-time: never block the sensor thread): `Q_sensors` calls `push_or_drop_oldest`; `Q_meas` — which interleaves whole sweeps with live IMU samples — calls `push_protecting`, evicting the oldest live-IMU sample first and a whole sweep only when nothing else is queued (a dropped sweep takes its IMU span with it, leaving the next solve's seed unrecoverable, so it is the last resort and is escalated as an error `event`). Every drop is emitted as a `q_*_dropped` telemetry scalar. `Q_kf` and `Q_map` are **lossless** (back-pressure) — their producers call `push_blocking` — because losing a keyframe corrupts the map; if the back-end falls behind, the front-end keeps producing odometry but keyframe creation slows. Each queue's instantaneous `size()` is published as a queue-depth telemetry scalar (§10) so a building backlog is visible before it becomes a drop.
- **Front-end is the priority thread** (T2). Expensive, lower-rate work (back-end optimize, GPU map integration, meshing) runs on separate threads so a 200 ms iSAM2 relinearization never stalls the 10–20 Hz odometry. This is the structural reason to split threads at all.
- **Feedback (`GraphUpdate`) is applied at safe points and is never lossless.** The L3→L2 correction edge is **not** a `BoundedQueue`: it is a **size-1 latest-wins snapshot** — a single mailbox slot the back-end overwrites with its most recent `GraphUpdate` and the front-end reads (and clears) between ingests via `apply_correction`. A slow front-end therefore *coalesces* corrections (it only ever applies the newest) rather than building a backlog, and the back-end never blocks waiting to publish one. This deliberately breaks the L3→L2 cycle the references avoid: a stalled or slow L3 can never add latency to L2 odometry, because L2 only ever does a bounded, non-blocking read of the latest snapshot. L4 receives the same `GraphUpdate` and schedules a region rebuild. No mid-iteration mutation; corrections are consumed at a scan boundary, rebasing the odometry origin. A fault-injection test stalls L3 and asserts L2 odometry latency is unchanged. (Whether `Q_kf` itself may become lossy is a separate, open question — gated on confirming a dropped keyframe degrades map *density*, not *correctness* — and is **not** decided here: `Q_kf` stays lossless per the rule above until that is shown.)
- **Determinism mode:** for tests and bag replay, **Replay mode** (`pipeline.mode: replay`, §8.2) runs the whole pipeline synchronously on the caller's thread — no stage threads, no queues — so results are bit-reproducible. Stage interfaces are identical; only the executor differs. The front-end itself is synchronous and sequential **in both modes** (the factory's `deterministic` flag is a no-op for it, §5.1), so two replays of the same stream are bit-identical; GPU kernels that reduce non-deterministically run in their deterministic variant where available. Pipeline-level bit-reproducibility is a **test-only** guarantee, not a production one.

### 11.3 Ownership and lifetime

- `MeridianPipeline` (in `meridian_pipeline`) owns every module (by `Ptr`), owns the queues, and owns the stage threads. Construction order = dependency order; destruction = reverse, calling `close()` on each queue (§11.1.1) so blocked stage loops return and join. The wrapper owns exactly one `MeridianPipeline`.
- Point clouds cross thread boundaries by **handle** (`IKeyframeStore`) or by `shared_ptr` move into a queue — never copied. The store is the single owner of keyframe clouds (§6.5).

### 11.4 Front-end propagate/solve split (control path vs. per-sweep solve)

The L2 front-end (T2) runs **two decoupled paths** so the operational, control-facing pose never inherits solver jitter:

- a **per-sweep solve path** — deskew + GN registration, triggered by each `ingest(PreprocessedGroup)`, which produces a freshly solved pose at the sweep end; and
- an **IMU-rate propagate/publish path** — the high-rate `NavState` consumed by TF/odom/control, produced by `ingest_imu_live` between solves.

The propagate path holds its own IMU-propagated live state; after every solve it is **rebased** by re-propagating the buffered post-sweep IMU samples from the solved state, so the live pose never drifts away from the registration result and never waits on it. The **estimator math** (the propagation model, the rebase) is owned by `04_frontend_estimation.md`; this section fixes only the *thread shape* — one solve path, one propagate path, both inside the single thread-confined T2 stage (§11.2) — and the invariant that `live_state()` latency is bounded by the propagate step alone and is independent of per-sweep solve cost. "Two paths" means two code paths over shared state, not two threads, so determinism mode (§11.2) runs them in deterministic order on the single thread.

### 11.5 Keyframe finalisation is synchronous

`constraint_cov` (§6.4) for the LIO front-end is a closed-form 6×6: the per-scan Laplace covariance of the GN solve, transported and composed along the interval since the previous keyframe. That costs microseconds, not milliseconds, so packets are finished **synchronously on T2** at emit time and pushed complete, in id order, straight into `Q_kf`. There is no finalizer worker, no `Q_finalize`, and no partially-filled packet anywhere in the system; replay and live take the identical code path.

---

## 12. Deferred-but-designed seams (first-pass scope ends at mesh)

The architecture is built so the deferred work *slots onto the same substrate* without rework. The guiding rule: **deferred features attach as new interface implementations or new consumers of existing value types — never as edits to existing layer contracts.**

- **Agnostic GPU surface backend (`vulkan`):** a SPIR-V compute implementation of `ISurfaceMap` (TSDF + colour fusion + Marching Cubes in Vulkan shaders) that runs the dense surface at GPU speed on Metal/AMD/Intel/NVIDIA alike — so the map is no longer CUDA-locked. It attaches as one more backend behind the existing `map.backend` selection and the §8.4 conformance suite; Tier R, the `KeyframeStore`, and the loop-rebuild path are untouched. First pass ships without it (spec 06 §10).
- **ESDF / path planning:** an additional `IMapLayer` implementation consuming the **same nvblox TSDF** the surface layer maintains (nvblox already computes ESDF on the GPU). Because map layers sit behind `IMapLayer` and are fed by the same corrected keyframes + retained clouds, adding ESDF is "register one more layer," not "rebuild the map." No L0–L3 change.
- **Semantics / object detection:** `KeyframePacket` already carries a shared-immutable `image`, and the keyframe store retains cloud + image. A semantics module becomes a consumer of the store plus an annotation channel in L4; the TSDF/colour voxel can carry an extra label channel. Boundary types do not change.
- **Offline high-fidelity export:** the nvblox Marching-Cubes mesh **is** the deliverable mesh. An *optional* one-shot Screened-Poisson pass over the retained keyframe clouds may be offered as an export utility in `meridian_tools`, never as a core runtime path.
- **Multi-LiDAR:** out of scope and not designed now. Meridian is single-LiDAR. If ever needed, additional LiDARs would attach behind the same `ISensorSource`/extrinsic machinery as added measurement streams into the same CT window — a future extension, not a current contract.
- **Multi-robot:** out of scope, but because the back-end consumes `KeyframePacket`s and `LoopConstraint`s by value, a peer's keyframes are just more inputs to L3/L5 — no contract change to the single-robot path.

---

## 13. Module bring-up order (single system, not feature phasing)

There is **one** system — the LIO estimator with nvblox mapping. The list below is *compile/integration order* for getting that one system standing up, not a roadmap of shippable versions. Nothing here is a "v1" that gets deployed before the rest exists.

1. **Cross-cutting + value types** (`meridian_common`, `meridian_time`, `meridian_config`, `meridian_debug`, `meridian_calib`) — the types every other module consumes; stand these up first so the contracts compile.
2. **L0/L1** (`meridian_sensors`, `meridian_preprocess`) — get real measurements onto the monotonic timeline and through preprocessing; verify with the bag replay harness.
3. **L2 LIO front-end** (`meridian_frontend/src/lio/`) — the voxel-hash map, the IMU tracker (static init + propagation + interval priors), then scan registration (screw deskew + GN ICP) and the orchestrating `LioFrontEnd`. (The earlier `src/iekf/` oracle and `src/ct/` estimator are retired, §5.4.)
4. **L4 surface map** (`meridian_map`) — Tier R voxel-hash, the `KeyframeStore`, and the `ISurfaceMap` surface tier consuming keyframe clouds. Bring up the `cpu` backend (`src/cpu/`) first — it builds everywhere and anchors the conformance suite (spec 06 §8.4) — then the `nvblox` backend (`src/nvblox/`) on CUDA hardware behind the same interface.
5. **L3 back-end** (`meridian_backend`) — iSAM2 graph consuming `KeyframePacket`s, broadcasting `GraphUpdate`.
6. **L5 place recognition** (`meridian_place`) — ScanContext++ → STD/BTC → GICP → PCM feeding loop constraints to L3, closing the map-correction loop into L4.
7. **L6 + wrapper polish** — operator surface, full debug-bus binding, launch/config/rviz.

This ordering is a convenience for integration; the **design** is the complete system from the start, and every interface is in place from step 1.

---

## 14. Summary of the contracts (the things you must not break)

1. **No ROS below `meridian_ros`.** Core is plain C++ (and CUDA); time is int64 ns; logging/telemetry are sinks bound by the wrapper. CUDA/nvblox is core, not middleware. (§1, §6.2, §9.5, §10)
2. **`KeyframePacket` is the only L2→L3 value.** Concretely defined; one geometric constraint per interval (`RelativeBetween` default; `AbsolutePrior` only on the first keyframe and the post-reseed re-anchor, mutually exclusive); velocity/bias ride as seeds, not graph variables, by default; bias estimation lives in L2. (§6)
3. **The front-end is the discrete LIO estimator** (same packet contract as its predecessors). A voxel-hash map + GN point-to-point ICP with an interval-averaged IMU screw prior, deskewing internally; camera and GNSS ride through unfused. The iEKF oracle and the continuous-time estimator are both retired. (§5.4, §7)
4. **The surface map is TSDF + colour + Marching-Cubes mesh behind the pluggable `ISurfaceMap` backend** — `nvblox` (GPU, production), `cpu` (host, dev/CI/non-CUDA), or deferred `vulkan` — one selected per run, explicitly and fail-fast, never a silent downgrade off nvblox. No VDBFusion/OpenVDB archive. Loop correction = clear-and-rebuild of the affected region from retained clouds at corrected poses, identical across backends. (§2, §9.5, and `06_mapping.md`)
5. **One LiDAR + one IMU + one camera + GNSS.** No multi-LiDAR merge logic in the design. (§2, §3, §8.2)
6. **Deskew is internal to L2** — a constant-screw warp at each point's sample time; L1 hands up undeskewed, time-sorted scans, and no feedback edge exists. (§7)
7. **Layers depend downward + cross-cutting only;** siblings talk through interfaces + `meridian_common` value types; `meridian_pipeline` is the only wirer. (§4, §5)
8. **Threading is fixed stages + bounded queues,** front-end prioritized, lossless keyframe/map queues, deterministic single-thread mode for tests. (§11)
9. **Introspection is first-class:** structured telemetry bus, dedicated debug topics / markers / timing, grounded in and improving on FAST-LIO's publishing. (§10)

---

## Appendix A — `KeyframePacket` schema (canonical, returnable)

```cpp
// include/meridian/common/keyframe_packet.hpp   — canonical def: spec 01 §6.1
namespace meridian {

struct KeyframePacket {
  // identity / time
  std::uint64_t   id    = 0;                 // monotonic keyframe id (graph node key)
  Timestamp       stamp = 0;                 // int64 ns, a real measurement instant

  // pose (REQUIRED)
  Frame           ref_frame = Frame::Odom;   // frame T_ref_body is expressed in
  Pose            T_ref_body;                // pose of the estimation frame F_e in ref_frame

  // kinematic state (OPTIONAL, gated; true only on the restart fallback)
  bool            kinematics_included = false;
  Eigen::Vector3d v_ref = Eigen::Vector3d::Zero();   // [m/s]    valid iff flag
  Eigen::Vector3d b_g   = Eigen::Vector3d::Zero();   // [rad/s]  valid iff flag
  Eigen::Vector3d b_a   = Eigen::Vector3d::Zero();   // [m/s^2]  valid iff flag

  // uncertainty: ONE block, ONE constraint per interval
  enum class ConstraintKind { RelativeBetween, AbsolutePrior, ImuPreintegration }
                  constraint_kind = ConstraintKind::RelativeBetween;
  std::uint64_t   rel_to_id = 0;             // RelativeBetween/ImuPreintegration: previous keyframe id
  Pose            T_relto_this;              // RelativeBetween: the relative transform
  GaussianBlock<6> constraint_cov;           // 6-DoF block, ROTATION-FIRST [rx,ry,rz,tx,ty,tz]
                                             // (GTSAM Pose3 boundary; the one rotation-first block)

  // observability (REQUIRED) — translation-first scores in a named frame
  ObservabilityReport observability;

  // data (REQUIRED), shared-immutable, no copy
  std::shared_ptr<const std::vector<LidarPoint>> cloud_body;  // deskewed, body frame at stamp
  std::shared_ptr<const CameraFrame>             image;       // null if no cam at this KF
  Pose            T_body_cam;                // extrinsic snapshot for colourisation

  // restart-fallback IMU summary (only when constraint_kind == ImuPreintegration)
  std::optional<ImuPreintegrationSummary> imu_summary;

  // provenance
  std::uint32_t   calib_version = 0;         // CalibrationSet snapshot that produced this
  std::uint32_t   frontend_kind = 1;         // 1 = retired CT front-end, 2 = LIO (diagnostics only; do not branch)
};

} // namespace meridian
```

Canonical field-level definition: spec 01 §6.1.

## Appendix B — Interface roster (one line each)

```
ISensorSource    : onSample(cb)                                              — L0,  swap: live/bag/sim
IFrontEnd        : ingest(PreprocessedGroup); ingest_imu_live; set_keyframe_sink;
                   live_state; apply_correction; diagnostics                — L2,  prod: lio, only impl (iekf + ct retired)
IBackEnd         : addKeyframe(KeyframePacket); addLoop; optimize; onResult  — L3,  isam2 (batch LM = offline debug)
IMapLayer        : integrate(kf); deintegrateRegion(aabb); query; extractMesh— L4,  ISurfaceMap backend: nvblox(GPU)|cpu(host)|vulkan(deferred); seam for ESDF
IKeyframeStore   : put(id,cloud,rgb,pose); get(id); clouds(region)          — L4,  ram (mmap = future)
IPlaceRecognizer : add(kf); query()->candidates; verify()->LoopConstraint   — L5,  sc++ -> std/btc -> gicp -> pcm
TelemetrySink    : timing/scalar/vec/cloud/pose/marker/event                — X-cut, swap: ros/recording/null
LogSink / Config : structured log + typed param tree                        — X-cut
```

## Appendix C — Debug topic map (wrapper-bound, grounded in FAST-LIO)

```
core telemetry key             → ROS 2 topic / type           (FAST-LIO origin, laserMapping.cpp)
cloud("map/registered")        → /meridian/cloud_registered  PC2  (/cloud_registered           :849)
cloud("body/scan")             → /meridian/cloud_body        PC2  (/cloud_registered_body       :851)
scalar("frontend/assoc/n_matched") → /meridian/telemetry     Tel  (NEW: count from effct_feat_num :695)
scalar("frontend/solver/chi")  → /meridian/telemetry         Tel  (NEW: final GN cost per sweep)
scalar("frontend/lio/*")       → /meridian/telemetry         Tel  (NEW: beta, accel_var, n_corr, deskew span)
pose("odom/body")              → /meridian/odom              Odom (/Odometry                     :857)
pose("frontend/path_sample")   → /meridian/path              Path (/path                         :859)
vec("frontend/obs")            → /meridian/markers           Mark (NEW: 6-axis scores, hexagon)
timing(stage,ms)               → /meridian/stage_timing      ST   (was printf/CSV aver_time_*  :1009)
event("place/loop_*")          → /meridian/events            Evt  (NEW)
event("map/region_rebuild")    → /meridian/events            Evt  (NEW)
event("frontend/lio/reseed")   → /meridian/events            Evt  (NEW: recovery made visible)
```
