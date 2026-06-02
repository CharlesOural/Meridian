# Meridian — Software Architecture Specification (Spec 00)

> **Status:** authoritative. This is the top-of-stack design document for Meridian, a from-scratch, SOTA, tightly-coupled multi-sensor 3D SLAM system for tactical operational use. It defines *how the code is organized* — not *how the estimator works* (the math lives in the per-module specs `01_*` onward). Everything here is normative: where **MUST**, **SHOULD**, **MAY** appears it carries its RFC-2119 meaning.
>
> **What Meridian is, in one line.** A **continuous-time (CT), tightly-coupled LiDAR-Inertial-Visual-GNSS** estimator: a sliding-window B-spline trajectory fusing direct point-to-plane LiDAR (each point at its true sample time), FAST-LIVO2-style sparse-direct photometric vision, IMU, and GNSS into one trajectory — feeding an iSAM2 factor-graph back-end and a GPU **nvblox** TSDF+colour+mesh map. The full CT LIVO+GNSS front-end **is** the design; there is no "ship a simpler filter first" milestone.
>
> **Deployment target.** **NVIDIA Jetson Orin.** A CUDA GPU is **always present** — the mapping stack is GPU-only (nvblox), with no CPU fallback path anywhere in the design.
>
> **Scope reminder.** First-pass scope ends at a **colourised triangle mesh** (nvblox Marching Cubes). ESDF/path-planning and semantics/object-detection are *designed-for* but *not built now*; the architecture must leave clean seams for them (§12).
>
> **Companion docs.** `DATASET.md` (FusionPortable primary, M2DGR co-primary — evaluation data). Sibling specs: `01_interfaces_and_data_types`, `02_sensors_timesync`, `03_preprocessing`, `04_frontend_estimation`, `05_backend_graph`, `06_mapping`, `07_loop_closure`, `08_calibration`, `09_debug_introspection`, `10_evaluation_harness`. For broader background see `docs/grounding/` and `docs/course/`.
>
> **Grounding.** Meridian combines the best parts of the apex references — **FAST-LIVO2** (sequential ESIKF, sparse-direct photometric vision, unified voxel map, exposure comp), **nvblox** (GPU TSDF+colour+Marching-Cubes mesh), **iSAM2/GTSAM** (incremental factor graph), and **Coco-LIC/CLINS + basalt-headers** (continuous-time SE(3) B-spline) — to beat any single published system. Engineering and debug recommendations are additionally grounded in the reference implementation `slam-reference/FAST_LIO/src/laserMapping.cpp`, cited as `laserMapping.cpp:NNN`; we keep what is good there and fix what is not.

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
   │  L0 sensor abstraction + time sync ──▶ L1 preprocessing ──▶ L2 CT LIVO+GNSS      │
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

3. **Reasoning at the right altitude.** ROS message types (`sensor_msgs::PointCloud2`, `nav_msgs::Odometry`) are wire formats, not domain types. They carry middleware concerns (QoS, `frame_id` strings), are awkward to compute on (XYZIRT packed in a byte blob), and version with ROS. The core computes on domain types (`meridian::PointCloud`, `meridian::SE3`, `KeyframePacket`) designed for the math, not the wire.

4. **Contrast with the reference code.** FAST-LIO fuses everything into one translation unit: `laserMapping.cpp` *is* the ROS node, the parameter loader, the buffer manager, the estimator driver, and the publisher. It directly `news` publishers (`ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000)`, `laserMapping.cpp:849`), reads ~40 parameters inline (`nh.param<...>`, `laserMapping.cpp:761-793`), and runs the estimator loop in `main()` (`laserMapping.cpp:865-1019`) — 1055 lines, all concerns interleaved. That is fine for a research prototype and wrong for a system we intend to extend for years and run operationally. Meridian deliberately inverts it.

### 1.1 The hard rule

> **No core library may depend on `rclcpp`, `rcl`, any `*_msgs` package, DDS, or the ROS parameter/clock/logging APIs.** Time is passed as `meridian::Time` (int64 nanoseconds, §6.2). Logging goes through `meridian::log` (a sink the wrapper binds to `rclcpp` logging at startup, §10.3). Parameters arrive as a populated `meridian::Config` struct (§8). **CUDA is not ROS:** GPU code (nvblox) is core, not wrapper — the no-ROS rule is about middleware, not about the GPU, which is always present (§9.5).

This is checkable mechanically (§9.4) and is the line that keeps the architecture honest.

---

## 2. Package / module layout

Meridian is a **colcon workspace** of ROS 2 *ament_cmake* packages (ROS 2 **Humble**). Core libraries are ament packages too (so colcon builds and tests them uniformly) but their `package.xml` lists **no `rclcpp` dependency** — only Eigen, PCL, GTSAM, Ceres, nvblox, etc. Wrappers are separate packages depending on both the core and `rclcpp`.

```
meridian_ws/
├─ src/
│  ├─ meridian_common/            # X-cut: math, geometry, core value types, KeyframePacket
│  │   include/meridian/common/   #   se3.hpp, so3.hpp, time.hpp, point.hpp, cloud.hpp,
│  │   src/                    #   keyframe_packet.hpp, odometry.hpp, pose_correction.hpp,
│  │   test/                   #   loop_constraint.hpp, status.hpp, ring_buffer.hpp
│  │
│  ├─ meridian_config/            # X-cut: typed Config tree + YAML loader (no ROS)
│  ├─ meridian_time/              # X-cut: time model, PTP/PPS clock model, interpolation
│  ├─ meridian_debug/             # X-cut: telemetry bus (timing/residual/observability), no ROS
│  ├─ meridian_calib/             # X-cut: extrinsic/intrinsic models, offline prior load,
│  │                           #        camera photometric (exposure/gain) params
│  │
│  ├─ meridian_sensors/           # L0: ISensorSource, sensor models, time-sync + aggregation
│  │                           #     (single LiDAR + single IMU + single camera + GNSS)
│  ├─ meridian_preprocess/        # L1: IDeskewProvider, downsample, validity, image pyramid
│  │                           #     (NB: CT deskew is implicit in L2; see §7)
│  ├─ meridian_frontend/          # L2: IFrontEnd + the CT LIVO+GNSS estimator
│  │   include/meridian/frontend/ #   ifrontend.hpp     <-- THE interface
│  │   src/ct/                 #   PRIMARY: continuous-time B-spline LIVO+GNSS estimator
│  │   │                       #     (basalt-headers CT spline + Ceres window solve;
│  │   │                       #      LiDAR point-to-plane @ true point time,
│  │   │                       #      FAST-LIVO2 sparse-direct photometric, IMU, GNSS)
│  │   src/iekf/               #   OPTIONAL baseline — FAST-LIO2-style iEKF.
│  │   │                       #     *** TEST ORACLE / REFERENCE ONLY ***, not a product
│  │   │                       #     path; compiled behind the same IFrontEnd. (§5.4)
│  │   src/frontend_factory.cpp
│  │
│  ├─ meridian_backend/           # L3: IBackEnd + GTSAM iSAM2 factor-graph impl
│  ├─ meridian_map/               # L4: IMapLayer + IKeyframeStore
│  │   include/meridian/map/      #   imaplayer.hpp, ikeyframe_store.hpp
│  │   src/nvblox/             #   THE map backend: nvblox GPU TSDF + colour +
│  │                           #     Marching-Cubes mesh. No CPU fallback, no second impl.
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
- **`meridian_map` has exactly one map implementation: nvblox** (`src/nvblox/`). There is no second backend directory, no `src/cpu/`, no VDBFusion/OpenVDB/NanoVDB path. `IMapLayer` stays as a clean seam (so a deferred ESDF layer can register behind it, §12), but its one and only surface implementation is nvblox.

---

## 3. Layer responsibilities (the contracts, not the math)

Each layer is summarized by **what it owns / consumes / produces**. Math is deferred to the per-layer spec.

| Layer | Package | Consumes | Produces | Owns |
|---|---|---|---|---|
| **L0** sensor abstraction + time sync | `meridian_sensors`, `meridian_time` | raw driver buffers | time-stamped, frame-tagged measurements (`ImuSample`, `LidarScan`, `ImageFrame`, `GnssFix`) on one monotonic timeline | PTP/PPS clock model, per-sensor `ISensorSource` (one LiDAR, one IMU, one camera, GNSS), measurement aggregation |
| **L1** preprocessing | `meridian_preprocess` | L0 measurements | downsampled, validity-flagged clouds; rectified images + pyramid | downsample (voxel grid), validity (blind radius, NaN/intensity), image pyramid; **IMU-only cold-start deskew provider** (steady-state deskew is implicit in L2's CT trajectory, §7) |
| **L2** front-end (odometry) | `meridian_frontend` | L1 output + IMU + GNSS | high-rate `Odometry` (live pose) **and** `KeyframePacket` on keyframe events | the **CT sliding-window tightly-coupled LIVO+GNSS** estimator (B-spline trajectory; per-point LiDAR point-to-plane, sparse-direct photometric, IMU-derivative, GNSS residuals); per-axis observability scoring; online extrinsic refinement (default on) |
| **L3** back-end | `meridian_backend` | `KeyframePacket` (L2) + loop constraints (L5) | optimized global trajectory; `PoseCorrection` broadcast to L4/L2 | GTSAM iSAM2 graph, online extrinsic variables, GNC robust kernels, PCM gate |
| **L4** layered map | `meridian_map` | corrected poses + retained clouds + RGB | **nvblox** GPU TSDF+colour; Marching-Cubes mesh; **per-keyframe cloud store** | de-integration / region rebuild on loop correction (GPU) |
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
- **R2.** A layer Lk depends on cross-cutting libs and **may** name lower layers' *value types* (which live in `meridian_common`), but **MUST NOT** depend on a sibling layer's *implementation*. L3 talks to L2 only through `KeyframePacket`, never by including `meridian_frontend/src/ct/...`.
- **R3.** Only `meridian_pipeline` may `#include` concrete factories from multiple layers (it constructs them). Layers receive collaborators by **interface pointer** via constructor injection, so they never name a concrete sibling.
- **R4.** `meridian_ros` is the **only** package allowed to depend on `rclcpp`/`*_msgs`. It depends on `meridian_pipeline` and `meridian_msgs`. Nothing depends on `meridian_ros`.
- **R5.** No cyclic build dependencies. The **deskew feedback edge** L2→L1 (§7.3) is *not* a build dependency: it is a runtime callback injected as an `IDeskewProvider`, so the build graph stays acyclic even though data flows back.
- **R6.** `meridian_map` (and only `meridian_map`, plus `meridian_frontend` for any GPU-resident residual work) links CUDA/nvblox. CUDA is a build dependency of these layers, not a transport dependency — it does not violate R4 (§9.5).

Enforced in CI by parsing each `package.xml`/`CMakeLists.txt` and asserting the edge set (§9.4).

---

## 5. The swappable interfaces (how plug-in works)

Every layer boundary is a pure-virtual interface plus a factory. The pattern is identical everywhere; learn it once. Note the word "swap" here means **clean modularity for testing and future extension**, not a product feature-rollout: each boundary has exactly one production implementation (the CT front-end, the iSAM2 back-end, the nvblox map). Interfaces buy us isolation, mockability, and a place to hang a test oracle — not a menu of shipping variants.

### 5.1 The pattern

```cpp
// include/meridian/frontend/ifrontend.hpp   (in meridian_frontend, NO ros)
namespace meridian {

class IFrontEnd {
public:
  using Ptr = std::unique_ptr<IFrontEnd>;
  virtual ~IFrontEnd() = default;

  // Feed measurements (thread-confined: called from the front-end stage thread).
  virtual void addImu(const ImuSample&)   = 0;
  virtual void addScan(LidarScan&&)       = 0;   // L1-preprocessed (single LiDAR)
  virtual void addImage(ImageFrame&&)     = 0;   // sparse-direct photometric modality
  virtual void addGnss(const GnssFix&)    = 0;   // GNSS modality

  // Pull the latest high-rate estimate (live pose, for TF / control).
  virtual Odometry latestOdometry() const = 0;

  // Keyframe handoff: invoked whenever a keyframe is born. The ONE thing L2 hands L3 (§6).
  using KeyframeSink = std::function<void(KeyframePacket&&)>;
  virtual void setKeyframeSink(KeyframeSink) = 0;

  // Back-end correction feedback (loop closure / global optimization result).
  virtual void onGlobalCorrection(const PoseCorrection&) = 0;

  // Introspection: hand the module a telemetry channel (§10).
  virtual void setTelemetry(TelemetrySink*) = 0;
};

// Factory selects the implementation by name from config (§8). Default: "ct_livo".
IFrontEnd::Ptr makeFrontEnd(const FrontEndConfig&, const Calibration&);

} // namespace meridian
```

The **factory free function** is the only place that names concrete implementations. `meridian_pipeline` calls `makeFrontEnd(cfg.frontend, calib)` and gets back an `IFrontEnd::Ptr`. The production path is `cfg.frontend.kind == "ct_livo"`; the `iekf_lio` entry exists strictly as a reference oracle for differential testing (§5.4), not as a deployable alternative.

### 5.2 The interface roster

| Interface | Header | Key methods | Role / notes |
|---|---|---|---|
| `ISensorSource` | `meridian/sensors/isensor_source.hpp` | `onSample(cb)` | swap: live driver, ROS-bag replay, simulator (same code path for live & offline) |
| `IDeskewProvider` | `meridian/preprocess/ideskew_provider.hpp` | `SE3 poseAt(Time)` | cold-start IMU-only; steady-state CT-trajectory-backed (§7) |
| `IFrontEnd` | `meridian/frontend/ifrontend.hpp` | `addScan/addImu/addImage/addGnss`, `setKeyframeSink`, `onGlobalCorrection` | **production: CT B-spline LIVO+GNSS** (`src/ct/`). Reference oracle: iEKF (`src/iekf/`), test-only (§5.4) |
| `IBackEnd` | `meridian/backend/ibackend.hpp` | `addKeyframe(KeyframePacket)`, `addLoop`, `optimize`, `onResult(cb)` | iSAM2 (production); batch LM available for offline debugging only |
| `IMapLayer` | `meridian/map/imaplayer.hpp` | `integrate(kf)`, `deintegrateRegion(aabb)`, `query`, `extractMesh` | **one impl: nvblox** (GPU TSDF+colour+mesh). Seam left for a deferred ESDF layer (§12) |
| `IKeyframeStore` | `meridian/map/ikeyframe_store.hpp` | `put(id,cloud,rgb,pose)`, `get(id)`, `clouds(region)` | RAM store (production); mmap'd on-disk store is a future option behind the same seam |
| `IPlaceRecognizer` | `meridian/place/iplace_recognizer.hpp` | `add(kf)`, `query()→candidates`, `verify()→LoopConstraint` | ScanContext++ → STD/BTC → GICP → PCM |
| `TelemetrySink` | `meridian/debug/telemetry.hpp` | `timing/scalar/vec/cloud/pose/marker/event` | ros, recording (tests), null (off) |

Each interface lives in its **owning layer's package** so the contract and its implementation ship together; the *value types* they exchange (`KeyframePacket`, `Odometry`, `PoseCorrection`, `LoopConstraint`) live in `meridian_common`, so no layer needs to depend on another's package to name them.

### 5.3 Why interfaces, not templates

Layer boundaries use **runtime polymorphism (virtual)**, not templates/CRTP, deliberately: boundaries are crossed a few hundred times per second at most (keyframes, scans), so virtual-call cost is irrelevant, while the benefits — separate compilation, mockable in tests, no template error walls, a place to hang the test oracle — are large. Inside a layer's hot loop (e.g. the per-point residual, the GPU kernels) we use concrete types, templates, and CUDA freely. The abstraction tax is paid only where it buys modularity.

### 5.4 The iEKF baseline is a test oracle, not a milestone

`meridian_frontend/src/iekf/` contains a FAST-LIO2-style iterated-EKF LIO. It exists for exactly one reason: as a **reference oracle for differential testing and bring-up sanity** behind the same `IFrontEnd` interface. On a given bag, the CT front-end's trajectory and the iEKF's should agree to within expected bounds; divergence flags a bug in one of them. It is **never** a shipped product path, **never** a "v1 we deploy first," and nothing in the architecture is organised around it. The default and only production front-end is the CT LIVO+GNSS estimator. (This is the *one* paragraph the baseline gets; everything else in this document assumes the CT front-end.)

---

## 6. The L2→L3 contract: `KeyframePacket`

The single most important boundary in the system is what the front-end hands the back-end. **The `KeyframePacket` is the only value that crosses L2→L3.** It is defined concretely so the contract is auditable and the back-end is fully decoupled from how the front-end produced the estimate. (Full field-level treatment is in `01_interfaces_and_data_types.md` §6; this section states the architectural invariants.)

### 6.1 Definition

```cpp
// include/meridian/common/keyframe_packet.hpp
namespace meridian {

enum class EdgeKind { RELATIVE_BETWEEN, ABSOLUTE_PRIOR, IMU_PREINT };

struct ObservabilityScore {
  FrameId              frame;     // reference frame the scores are expressed in
  std::array<double,6> axis;      // [rx, ry, rz, tx, ty, tz], higher = better observed
};

struct KeyframePacket {
  // --- identity / time ---
  KeyframeId   id;            // monotonic, assigned by the front-end
  Time         stamp;         // int64 ns, the keyframe's reference time (§6.2)

  // --- pose (REQUIRED) ---
  FrameId      frame;         // the named frame this pose is expressed IN (e.g. "odom")
  SE3          T_frame_body;  // pose of body in `frame` at `stamp`

  // --- kinematic state (OPTIONAL, gated) ---
  bool         has_velocity_bias = false;   // the "included" flag
  Vec3         v_frame;       // body velocity in `frame`     (valid iff flag)
  Vec3         bias_acc;      // IMU accel bias               (valid iff flag)
  Vec3         bias_gyr;      // IMU gyro bias                (valid iff flag)

  // --- uncertainty (REQUIRED) ---
  EdgeKind     edge_kind;     // RELATIVE_BETWEEN (default) | ABSOLUTE_PRIOR | IMU_PREINT
  KeyframeId   ref_id;        // for RELATIVE_BETWEEN/IMU_PREINT: the previous keyframe id
  Mat6         information;   // 6x6 info matrix, ordering [rx,ry,rz,tx,ty,tz] (§6.4)

  // --- observability (REQUIRED) ---
  ObservabilityScore obs;     // 6 per-axis scores + their reference frame; shapes L3 noise

  // --- data handles (REQUIRED for map / loop) ---
  CloudHandle  cloud;         // handle into IKeyframeStore (NOT a copy) — §6.5
  RgbHandle    rgb;           // handle to the keyframe image (for colourisation + visual cues)

  // --- provenance (debug) ---
  std::string  producer;      // "ct_livo" (or "iekf_oracle" in test runs) — telemetry/forensics
};

} // namespace meridian
```

### 6.2 Time model

`Time` is **`int64` nanoseconds** on one monotonic timeline established by L0/PTP. There is no `ros::Time` below the wrapper. The CT B-spline knots, the per-point LiDAR sample times, the camera mid-exposure time, and the deskew/interpolation query (§7.3) all use this scalar. The wrapper converts `rclcpp::Time` ↔ `meridian::Time` in exactly one place (`conversions/`). This mirrors how FAST-LIO threads a single `double lidar_end_time` through everything (`laserMapping.cpp:593`) — we keep the single-scalar-time discipline but make it integer-ns to avoid double-precision drift at long uptime (critical for a CT spline whose knot times span the whole mission).

### 6.3 One-constraint rule (kill double-counting)

A naïve handoff double-counts information by simultaneously passing (a) an absolute marginal prior, (b) a relative between-factor, **and** (c) an IMU-preintegration factor derived from *the same* measurements. Meridian picks **one** clean contract per interval:

> **Default and recommended:** `edge_kind = RELATIVE_BETWEEN`. The packet carries a single relative pose `T_ref_body = T(ref)^{-1} · T(this)` with its marginal covariance, and L3 inserts exactly **one** `BetweenFactor` between consecutive keyframes. Velocity and biases **do not** cross as live optimization variables in the default path (`kinematics_included = false`); they ride along only as *seeds / telemetry*, not as graph variables. The CT front-end has already fused IMU+LiDAR+visual+GNSS inside the window, so that information is *in* the relative covariance.
>
> **Mutually-exclusive fallback:** `edge_kind = IMU_PREINT` is used **only on the window-restart fallback** (§7.4) when no valid relative pose spans the gap. Then the packet carries preintegrated IMU between `ref` and `this`, L3 inserts a GTSAM `CombinedImuFactor` (+ the velocity/bias variables that factor needs), and **no `BetweenFactor`** is added for that interval. The two are never both present for the same interval. Bias estimation otherwise lives in L2.

So for any keyframe interval the back-end receives **exactly one** geometric constraint. That is the line that keeps the information budget correct. (See `05_backend_graph.md` for GTSAM factor construction and how `information` and `obs` assemble the noise model — `obs` inflates the diagonal of weakly-observed axes, X-ICP / D²-LIO style, instead of a binary degeneracy switch.)

### 6.4 Tangent ordering and frames

The `Mat6 information` and all 6-vectors use the ordering **`[rotation(x,y,z), translation(x,y,z)]`** in the *body* tangent space at `T_frame_body`, expressed in `frame`. `frame` is a named string id (`"odom"`, `"map"`); the back-end owns the `odom→map` relationship. This explicitness kills the classic "is this covariance in world or body, rotation-first or translation-first?" bug. (FAST-LIO hand-packs the EKF `P` into `pose.covariance` with a rotation/translation block-swap, `laserMapping.cpp:597-606` — easy to get wrong precisely because the convention is implicit. We make it explicit and typed.)

### 6.5 Cloud handle, not cloud copy

`CloudHandle` is an opaque id into the **`IKeyframeStore`** — the data structure that holds each keyframe's deskewed cloud once. The packet never copies the point cloud across the thread boundary — it carries a handle, the cloud lives once in the store, and **L4** (nvblox re-integration on loop correction), **L5** (GICP verification), and an optional offline export pass all read it from there. (Full treatment in `06_mapping.md`.)

---

## 7. Bootstrap, deskew, and the feedback edge

### 7.1 The problem

A motion-compensated LiDAR scan needs a trajectory; the trajectory comes from L2; L2 needs motion-compensated scans. Circular, with no defined cold-start. Meridian specifies the bootstrap explicitly and documents deskew as a **feedback edge**, not a strict bottom-up dependency. Note that in steady state the CT front-end makes deskew *implicit*: each LiDAR point is registered using the continuous trajectory `T(t_i)` evaluated at that point's true sample time, so there is no separate "deskew then register" step (§7.5). The explicit `IDeskewProvider` matters only at cold-start, before a CT window exists.

### 7.2 Cold-start (IMU-only)

On startup the front-end has no trajectory. We bootstrap as FAST-LIO does: an **IMU initialization** phase over the first frames, estimating gravity from `mean_acc`, gyro bias from `mean_gyr`, and the acc/gyro covariances via a running mean (FAST-LIO's `ImuProcess::IMU_init` accumulates `mean_acc += (cur_acc - mean_acc)/N`, records `first_lidar_time`, and only flips `imu_need_init_` false after `MAX_INI_COUNT` frames — `FAST_LIO/include/IMU_Processing.hpp`). During this phase:

1. The first scan(s) are deskewed by **IMU-only forward integration** (constant-bias, gravity-aligned): `IDeskewProvider` = `ImuOnlyDeskew`.
2. Once enough span exists, the CT estimator initializes its first control points and takes over: thereafter point registration uses the continuous trajectory directly (`TrajectoryDeskew`, backed by the CT spline), and the explicit deskew provider is no longer on the critical path.

FAST-LIO gates the analogous transition with `flg_EKF_inited = (lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true` (`laserMapping.cpp:898`); Meridian keeps the same idea but exposes the transition as a telemetry `event` so it is visible.

### 7.3 Deskew as a feedback edge

`IDeskewProvider::poseAt(Time)` is injected into L1 by the pipeline. Its steady-state implementation is *backed by L2's current CT trajectory*, but L1 does **not** depend on `meridian_frontend` at build time — it depends on the `IDeskewProvider` interface (which lives in `meridian_preprocess`), and the pipeline wires the concrete provider that wraps the front-end. So:

- **Build graph:** acyclic (L1 → interface only).
- **Data flow:** L2 → (poses) → L1 → L2 — a genuine feedback loop, documented and intentional.

This is the explicit answer to "deskew is not strict bottom-up": it is a feedback edge realized through dependency injection, so the cycle exists in data but not in the build. (In steady-state CT operation most of this folds into L2's per-point registration; the provider remains the clean seam for cold-start and for any module that wants a pose at an arbitrary time.)

### 7.4 Window-restart fallback

If the estimator diverges or loses observability (degeneracy via per-axis `obs`, or residual blow-up — FAST-LIO's analogue is bailing with `ekfom_data.valid = false` when `effct_feat_num < 1`, `laserMapping.cpp:708-712`), the front-end performs a **window restart**: freeze the last good pose, re-run IMU-only deskew for the next scan, re-initialize the sliding CT window from there. On a restart the *first* keyframe emitted carries `edge_kind = IMU_PREINT` (§6.3) because no continuous relative pose spans the gap — the *only* time the IMU factor crosses the boundary, mutually exclusive with the between-factor. The restart is published on the debug bus (§10) so an operator sees exactly when and why recovery happened.

### 7.5 Continuous-time registration (why deskew is mostly implicit)

The CT front-end represents the trajectory as a B-spline `T(t) ∈ SE(3)` (basalt-headers cubic split SO(3)×R³). A LiDAR point sampled at time `t_i` is registered against the map using `T(t_i)` *directly* — there is no intermediate "warp the whole scan to one reference time" operation, because every point already knows its own pose. The same `T(t)` evaluated at a camera's mid-exposure time gives the pose for the sparse-direct photometric residual, and its analytic derivative gives the IMU-rate residual. This is the structural reason Meridian is a continuous-time system and not a discrete-time filter: motion distortion, rolling exposure, and multi-rate fusion are handled by one trajectory representation rather than by per-sensor compensation hacks. (Math in `04_frontend_estimation.md`.)

---

## 8. Configuration / parameter strategy

### 8.1 One typed tree, two loaders

There is **one** configuration type, `meridian::Config`, a nested struct tree of plain fields (no `std::any`, no string-keyed maps in the hot path), populated by either:

- **YAML loader** (`meridian_config`, yaml-cpp) — used for bag replay, tests, and as the *source of truth* file checked into `meridian_ros/config/`.
- **ROS 2 parameter loader** (`meridian_ros`) — declares the same keys as ROS parameters and fills the same `Config` struct, reusing the YAML key paths so the two never drift.

The core only ever sees a fully-populated `Config`. This avoids FAST-LIO's pattern of scattering ~40 `nh.param<...>("preprocess/blind", p_pre->blind, 0.01)` reads through the node body (`laserMapping.cpp:761-793`); instead the schema is one declarative thing.

### 8.2 Schema shape (single LiDAR + IMU + camera + GNSS, CT front-end, nvblox map)

```yaml
meridian:
  pipeline:   { mode: live|replay, threads: { frontend: 1, backend: 1, map: 1 } }
  time:       { source: ptp|pps|host, max_skew_ms: 5 }
  sensors:
    lidar:    { topic: /os/points, model: ouster_os1_128,
                extrinsic_T: [..], extrinsic_R: [..] }     # single LiDAR
    imu:      { topic: /os/imu, rate_hz: 200, cov_acc: .., cov_gyr: ..,
                b_acc_cov: .., b_gyr_cov: .. }              # single IMU (estimation frame)
    camera:   { topic: /cam0, model: pinhole, intrinsics: [..], extrinsic: [..],
                photometric: { exposure_comp: true } }      # single camera, sparse-direct
    gnss:     { topic: /gnss/fix, enable: true }
  preprocess: { blind: 0.5, point_filter_num: 3, voxel_surf_m: 0.5 }
  frontend:   { kind: ct_livo,                              # production CT estimator
                spline: { order: cubic, knot_dt_ms: 25, window_knots: 8 },
                lidar: { voxel_map_m: 0.5 }, visual: { patch: 8, levels: 3 },
                gnss: { use: true }, extrinsic_refine: true,
                keyframe: { dist_m: 1.0, rot_deg: 10, time_s: 1.0 } }
  backend:    { kind: isam2, relinearize_thresh: 0.1, robust: gnc_cauchy }
  map:        { backend: nvblox, tsdf_voxel_m: 0.05, mesh: marching_cubes,
                colour: true }                              # GPU; no fallback key
  place:      { kind: scan_context_pp, pcm: true, gicp_fitness_max: 0.3 }
  debug:      { level: info, publish_clouds: true, publish_markers: true,
                timing: true, telemetry_rate_hz: 10 }
```

### 8.3 Rules

- **Validated on load.** `Config::validate()` checks units, ranges, and cross-field consistency (e.g. `tsdf_voxel_m ≤ preprocess.voxel_surf_m`, CT `knot_dt_ms > 0`) and fails fast with a precise message — no silent defaults masking a typo (a real FAST-LIO foot-gun: a mistyped `nh.param` key silently uses the default).
- **`kind` strings select implementations** through the factories (§5.1). The production values are fixed (`ct_livo`, `isam2`, `nvblox`); `frontend.kind: iekf_lio` is reserved for the test oracle (§5.4) and is not a deployment configuration. `map.backend` has exactly one valid value, `nvblox` — there is no CPU/VDB alternative to select.
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
add_library(meridian_frontend
  src/ct/ct_frontend.cpp          # PRIMARY: continuous-time LIVO+GNSS estimator
  src/ct/spline_window.cpp
  src/ct/residuals_lidar.cpp src/ct/residuals_visual.cpp src/ct/residuals_imu.cpp
  src/iekf/iekf_frontend.cpp      # baseline oracle (test-only, §5.4)
  src/frontend_factory.cpp)
target_include_directories(meridian_frontend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(meridian_frontend PUBLIC cxx_std_20)
target_link_libraries(meridian_frontend PUBLIC
  meridian_common meridian_config meridian_debug meridian_calib
  Eigen3::Eigen Sophus::Sophus Ceres::ceres ${PCL_LIBRARIES})  # NO rclcpp — enforced (§9.4)
ament_target_dependencies(meridian_frontend Eigen3 PCL)
add_library(meridian::frontend ALIAS meridian_frontend)
ament_export_targets(meridian_frontendTargets HAS_LIBRARY_TARGET)
```

OpenMP / CUDA are opt-in **per hot-loop library**: the CT residual assembly may use OpenMP, and `meridian_map` links CUDA/nvblox. We keep these flags **library-local** rather than a global flag so non-parallel modules stay deterministic, and we make any parallel section honour the determinism mode (§11.2).

### 9.3 Two binaries, both thin

`meridian_ros` builds the wrapper executables (`odometry_node`, `mapping_node`) and the `conversions` translation library. Each `*_node.cpp` is ~100 lines: declare params → build `Config` → construct `MeridianPipeline` → subscribe (convert msg→core, push into pipeline) → register publish callbacks (core→msg). All algorithm lives below. `meridian_tools` builds the offline `replay` binary that drives the *same* `MeridianPipeline` from a bag through `ISensorSource`, proving the core runs identically off-ROS.

### 9.4 CI gates (enforce the architecture)

- **No-ROS gate:** a grep over `src/meridian_*` excluding `meridian_ros` for `rclcpp | ros/ros.h | sensor_msgs | rclcpp/clock` must return empty.
- **Dependency lint:** a script asserts the §4 edge set and fails if any `meridian_<layer>/package.xml` lists `rclcpp`/`*_msgs`, or if a layer `#include`s a sibling's `src/`.
- **Single-map-backend gate:** assert `meridian_map` contains exactly one map implementation directory (`src/nvblox/`) and that no source under `meridian_map` references VDBFusion/OpenVDB/NanoVDB — there is no CPU map path.
- **clang-tidy / clang-format** on every TU; warnings-as-errors in core.
- **Unit + replay tests** under `colcon test`, including the CT-vs-iEKF-oracle differential check (§5.4). Per-module testing detailed in `02..10`.

### 9.5 CUDA / nvblox is core, not wrapper

Meridian targets **Jetson Orin** and assumes a CUDA GPU is **always present**. nvblox (TSDF + colour + Marching-Cubes mesh) runs entirely on the GPU and lives in `meridian_map`, a *core* package — it is built and tested without any ROS. CUDA being a dependency of `meridian_map` (and of any GPU residual work in `meridian_frontend`) does **not** violate the no-ROS rule (§1.1): that rule is about *middleware*, not about the accelerator. There is **no CPU fallback** for mapping and the build does not provide one; a missing GPU at runtime is a hard, fail-fast configuration error, not a degraded mode.

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
  virtual void scalar(const char* key, double v, Time t)         = 0;  // residual, #eff pts…
  virtual void vec(const char* key, const VecX&, Time t)         = 0;  // bias, obs[6]…
  virtual void cloud(const char* key, const PointCloud&, FrameId, Time) = 0;
  virtual void pose (const char* key, const SE3&,        FrameId, Time) = 0;
  virtual void marker(const Marker&, Time)                       = 0;  // geometric overlays
  virtual void event(Level, const char* tag, const std::string&, Time) = 0;
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
| **Effective/inlier points used in update** | `/cloud_effected` (`:853`; count in `effct_feat_num` `:695`) | `cloud("frontend/effective")` **+** `scalar("frontend/n_effective")` **+** `scalar("frontend/res_mean")` — FAST-LIO published only the cloud (commented out at the call site, `:983`); we publish the *count* and *mean residual* so degeneracy is **plottable**, not just visible |
| Live odometry + covariance | `/Odometry` with hand-packed `pose.covariance` (`:597-606`) | `pose("odom/body")` + `vec("frontend/cov_diag")` — covariance is a typed 6×6 in known ordering (§6.4), not smuggled into the pose message |
| Trajectory | `/path` (`:859`, throttled `jjj % 10`) | `pose` history aggregated by the wrapper into `nav_msgs/Path` |
| **CT spline / window state** | none (FAST-LIO is discrete-time) | `pose("frontend/spline_knots")` + `marker("frontend/window")` — the B-spline control points and active window, so the CT trajectory is **visible**. Strict new signal. |
| **Visual (sparse-direct) residual** | none (LiDAR-only reference) | `scalar("frontend/photometric_res")` + `cloud("frontend/visual_patches")` — FAST-LIVO2-style photometric inliers exposed |
| Online extrinsic | (FAST-LIO logs `offset_R_L_I/offset_T_L_I` to `fout_*` only) | `pose("calib/T_imu_lidar")`, `pose("calib/T_imu_cam")` — exposes the online extrinsic refinement |
| **Per-axis observability** | none (only the binary `flg_EKF_inited`) | `vec("frontend/observability", obs[6])` — the 6 scores driving back-end noise; rendered as a 6-bar rviz marker (§10.4). Strict upgrade. |
| Per-stage timing | `aver_time_match/solve/icp/...` to `printf`+CSV (`:991-1009`) | `ScopedTimer` on every stage → `StageTiming` topic: `preprocess`, `frontend.ct_solve`, `frontend.lidar_assoc`, `frontend.visual`, `backend.optimize`, `map.integrate`, `mesh.extract`. Live breakdown, not a logfile. |
| State/bias trace | `dump_lio_state_to_log` (`:150`) | `vec("frontend/bias_acc")`, `vec("frontend/bias_gyr")`, `scalar("frontend/grav_norm")` |
| GNSS | none (LiDAR-only reference) | `pose("frontend/gnss_anchor")` + `event("frontend/gnss_fix", fix_type)` — GNSS fusion made visible |
| Back-end health | none | `scalar("backend/chi2")`, `scalar("backend/n_factors")`, `event("backend/relinearize")` |
| Loop closure | none | `marker("place/loop_edge")` (line between matched keyframes, coloured by GICP fitness) + `event("place/loop_accepted|rejected_pcm")` |
| Map de-integration | none | `event("map/region_rebuild")` + `marker("map/dirty_region")` AABB |
| Recovery | none (silent `continue` / `ROS_WARN`) | `event(WARN, "frontend/window_restart", reason)` — the §7.4 fallback is **visible** |

### 10.3 Structured logging

`meridian::log` is a thin macro layer (`MERIDIAN_INFO(...)`, `MERIDIAN_WARN(...)`) forwarding to a `LogSink`. The wrapper binds it to `rclcpp` logging; tests to a buffer; offline to stdout+file. Logs are **structured** (key=value), greppable and parseable, with module tag and `Time` always present. The core never calls `RCLCPP_INFO` or `ROS_WARN`.

### 10.4 rviz markers (geometric introspection)

The wrapper turns `marker(...)` calls into a `MarkerArray`. Standard markers Meridian ships:

- **Observability hexagon:** 6 bars (rx,ry,rz,tx,ty,tz) scaled by `obs`, coloured green→red, anchored at the body — the operator instantly sees which axes are weakly observed (a long corridor → weak forward translation).
- **CT sliding-window view:** the B-spline control points and the active window span, so the operator sees the trajectory the front-end is currently optimising.
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

Meridian uses a **small fixed set of stage threads connected by bounded single-producer queues** — not a thread pool, not a task graph — because the data flow is a near-linear pipeline and predictability beats cleverness for a real-time estimator. (FAST-LIO runs single-threaded in `main()` with a `mutex` + `condition_variable` guarding the sensor buffers, `laserMapping.cpp:81-82, 281`; we make the stage boundaries explicit instead of implicit in one loop.)

```
[ROS sub callbacks]            (rclcpp executor threads; wrapper only)
   │  convert msg→core, push (non-blocking)
   ▼
(Q_sensors) ─▶ [L0/L1 stage: time-sync + preprocess]              thread T1
                   │ synced measurements (1 LiDAR, IMU, cam, GNSS)
                   ▼
              (Q_meas) ─▶ [L2 CT LIVO+GNSS front-end]             thread T2 (hot)
                              │ Odometry  ──────────────▶ TF/odom publish (low latency)
                              │ KeyframePacket
                              ▼
                         (Q_kf) ─▶ [L3 back-end + L5 place-rec]     thread T3
                                       │ PoseCorrection ──▶ feedback to L2 & L4
                                       ▼
                                  (Q_map) ─▶ [L4 nvblox integrate (GPU)] thread T4
                                                │ mesh extract (on demand, GPU) thread T5
                                                ▼
                                           [L6 surface publish]
```

### 11.2 Rules

- **Each stage is single-threaded internally** (its module is *not* required to be thread-safe; it is *thread-confined*). Concurrency lives only in the queues. The simplest model that keeps the front-end deterministic and easy to reason about. (nvblox launches GPU kernels from its single map thread; the host-side stage is still thread-confined.)
- **Queues are bounded** with an explicit per-edge overflow policy. `Q_meas` is lossy under overload (real-time: never block the sensor thread; drop oldest and emit a telemetry `event`). `Q_kf` and `Q_map` are **lossless** (back-pressure) — losing a keyframe corrupts the map, so if the back-end falls behind, the front-end keeps producing odometry but keyframe creation slows.
- **Front-end is the priority thread** (T2). Expensive, lower-rate work (back-end optimize, GPU map integration, meshing) runs on separate threads so a 200 ms iSAM2 relinearization never stalls the 10–20 Hz odometry. This is the structural reason to split threads at all.
- **Feedback (`PoseCorrection`) is applied at safe points:** L2 consumes corrections between scans (rebasing its odometry origin via `onGlobalCorrection`); L4 receives them and schedules a region rebuild. No mid-iteration mutation.
- **Determinism mode:** for tests and bag replay, a `--single-thread` pipeline mode runs all stages sequentially on one thread so results are bit-reproducible. Stage interfaces are identical; only the executor differs. In this mode any OpenMP in the CT residual assembly is disabled (or fixed-scheduled) so residual sums are order-deterministic; GPU kernels that reduce non-deterministically are run in their deterministic variant where available. Bit-reproducibility is therefore a **test-only** guarantee, not a production one.

### 11.3 Ownership and lifetime

- `MeridianPipeline` (in `meridian_pipeline`) owns every module (by `Ptr`), owns the queues, and owns the stage threads. Construction order = dependency order; destruction = reverse, draining queues. The wrapper owns exactly one `MeridianPipeline`.
- Point clouds cross thread boundaries by **handle** (`IKeyframeStore`) or by `shared_ptr` move into a queue — never copied. The store is the single owner of keyframe clouds (§6.5).

---

## 12. Deferred-but-designed seams (first-pass scope ends at mesh)

The architecture is built so the deferred work *slots onto the same substrate* without rework. The guiding rule: **deferred features attach as new interface implementations or new consumers of existing value types — never as edits to existing layer contracts.**

- **ESDF / path planning:** an additional `IMapLayer` implementation consuming the **same nvblox TSDF** the surface layer maintains (nvblox already computes ESDF on the GPU). Because map layers sit behind `IMapLayer` and are fed by the same corrected keyframes + retained clouds, adding ESDF is "register one more layer," not "rebuild the map." No L0–L3 change.
- **Semantics / object detection:** `KeyframePacket` already carries an `RgbHandle`, and the keyframe store retains cloud + image. A semantics module becomes a consumer of the store plus an annotation channel in L4; the TSDF/colour voxel can carry an extra label channel. Boundary types do not change.
- **Offline high-fidelity export:** the nvblox Marching-Cubes mesh **is** the deliverable mesh. An *optional* one-shot Screened-Poisson pass over the retained keyframe clouds may be offered as an export utility in `meridian_tools`, never as a core runtime path.
- **Multi-LiDAR:** out of scope and not designed now. Meridian is single-LiDAR. If ever needed, additional LiDARs would attach behind the same `ISensorSource`/extrinsic machinery as added measurement streams into the same CT window — a future extension, not a current contract.
- **Multi-robot:** out of scope, but because the back-end consumes `KeyframePacket`s and `LoopConstraint`s by value, a peer's keyframes are just more inputs to L3/L5 — no contract change to the single-robot path.

---

## 13. Module bring-up order (single system, not feature phasing)

There is **one** system — the full CT LIVO+GNSS estimator with nvblox mapping. The list below is *compile/integration order* for getting that one system standing up, not a roadmap of shippable versions. Nothing here is a "v1" that gets deployed before the rest exists.

1. **Cross-cutting + value types** (`meridian_common`, `meridian_time`, `meridian_config`, `meridian_debug`, `meridian_calib`) — the types every other module consumes; stand these up first so the contracts compile.
2. **L0/L1** (`meridian_sensors`, `meridian_preprocess`) — get real measurements onto the monotonic timeline and through preprocessing; verify with the bag replay harness.
3. **L2 CT front-end** (`meridian_frontend/src/ct/`) — the CT spline window, then the residual blocks (LiDAR point-to-plane, IMU-derivative, sparse-direct photometric, GNSS) integrated into one solve. The `src/iekf/` oracle is brought up alongside purely to differential-test the CT estimator (§5.4).
4. **L4 nvblox map** (`meridian_map/src/nvblox/`) — GPU TSDF+colour+mesh consuming keyframe clouds; the keyframe store.
5. **L3 back-end** (`meridian_backend`) — iSAM2 graph consuming `KeyframePacket`s, broadcasting `PoseCorrection`.
6. **L5 place recognition** (`meridian_place`) — ScanContext++ → STD/BTC → GICP → PCM feeding loop constraints to L3, closing the map-correction loop into L4.
7. **L6 + wrapper polish** — operator surface, full debug-bus binding, launch/config/rviz.

This ordering is a convenience for integration; the **design** is the complete system from the start, and every interface is in place from step 1.

---

## 14. Summary of the contracts (the things you must not break)

1. **No ROS below `meridian_ros`.** Core is plain C++ (and CUDA); time is int64 ns; logging/telemetry are sinks bound by the wrapper. CUDA/nvblox is core, not middleware. (§1, §6.2, §9.5, §10)
2. **`KeyframePacket` is the only L2→L3 value.** Concretely defined; one geometric constraint per interval (`RELATIVE_BETWEEN` default; `IMU_PREINT` only on restart, mutually exclusive, via GTSAM `CombinedImuFactor`); velocity/bias ride as seeds, not graph variables, by default; bias estimation lives in L2. (§6)
3. **The front-end is the CT LIVO+GNSS estimator.** A B-spline trajectory fusing per-point LiDAR, sparse-direct vision, IMU, and GNSS. The iEKF is a test oracle only, never a product path. (§5.4, §7.5)
4. **The map is nvblox, GPU-only.** One backend, no CPU fallback, no VDB path. nvblox does TSDF + colour + Marching-Cubes mesh; loop correction = clear-and-rebuild of the affected region from retained clouds at corrected poses. (§2, §9.5, §6.5, and `06_mapping.md`)
5. **One LiDAR + one IMU + one camera + GNSS.** No multi-LiDAR merge logic in the design. (§2, §3, §8.2)
6. **Deskew is a feedback edge via `IDeskewProvider`** for cold-start; in steady state CT registration makes it implicit (each point at its true time). Build graph stays acyclic. (§7)
7. **Layers depend downward + cross-cutting only;** siblings talk through interfaces + `meridian_common` value types; `meridian_pipeline` is the only wirer. (§4, §5)
8. **Threading is fixed stages + bounded queues,** front-end prioritized, lossless keyframe/map queues, deterministic single-thread mode for tests. (§11)
9. **Introspection is first-class:** structured telemetry bus, dedicated debug topics / markers / timing, grounded in and improving on FAST-LIO's publishing. (§10)

---

## Appendix A — `KeyframePacket` schema (canonical, returnable)

```cpp
enum class EdgeKind { RELATIVE_BETWEEN, ABSOLUTE_PRIOR, IMU_PREINT };
struct ObservabilityScore { FrameId frame; std::array<double,6> axis; }; // [rx,ry,rz,tx,ty,tz]

struct KeyframePacket {
  KeyframeId   id;                    // monotonic, front-end assigned
  Time         stamp;                 // int64 ns, single monotonic timeline
  FrameId      frame;                 // name of the frame T_frame_body is in
  SE3          T_frame_body;          // REQUIRED pose
  bool         has_velocity_bias;     // "included" flag (true only on restart fallback)
  Vec3         v_frame, bias_acc, bias_gyr;   // valid iff flag (seed/telemetry, not graph vars by default)
  EdgeKind     edge_kind;             // RELATIVE_BETWEEN | ABSOLUTE_PRIOR | IMU_PREINT
  KeyframeId   ref_id;                // previous keyframe (relative / preint edge)
  Mat6         information;           // 6x6, ordering [rx,ry,rz,tx,ty,tz] in frame's body tangent
  ObservabilityScore obs;            // 6 per-axis scores + reference frame
  CloudHandle  cloud;                 // handle into IKeyframeStore (no copy)
  RgbHandle    rgb;                   // image handle (colourisation + visual cues)
  std::string  producer;             // "ct_livo" (or "iekf_oracle" in test runs)
};
```

## Appendix B — Interface roster (one line each)

```
ISensorSource    : onSample(cb)                                              — L0,  swap: live/bag/sim
IDeskewProvider  : SE3 poseAt(Time)                                         — L1,  cold-start imu-only / steady-state CT-backed
IFrontEnd        : addScan/addImu/addImage/addGnss; setKeyframeSink;
                   latestOdometry; onGlobalCorrection; setTelemetry         — L2,  prod: ct_livo (iekf oracle = test-only)
IBackEnd         : addKeyframe(KeyframePacket); addLoop; optimize; onResult  — L3,  isam2 (batch LM = offline debug)
IMapLayer        : integrate(kf); deintegrateRegion(aabb); query; extractMesh— L4,  one impl: nvblox (GPU); seam for deferred ESDF
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
cloud("frontend/effective")    → /meridian/cloud_effective   PC2  (/cloud_effected              :853)
scalar("frontend/n_effective") → /meridian/telemetry         Tel  (NEW: count from effct_feat_num :695)
scalar("frontend/res_mean")    → /meridian/telemetry         Tel  (NEW: from res_mean_last       :715)
scalar("frontend/photometric_res")→ /meridian/telemetry      Tel  (NEW: sparse-direct visual residual)
pose("odom/body")              → /meridian/odom              Odom (/Odometry                     :857)
pose("frontend/spline_knots")  → /meridian/markers           Mark (NEW: CT B-spline control points)
vec("frontend/cov_diag")       → /meridian/telemetry         Tel  (was packed into pose.cov   :597-606)
vec("frontend/observability")  → /meridian/markers           Mark (NEW: 6-axis scores, hexagon)
pose("calib/T_imu_lidar")      → /meridian/extrinsic         Odom (was fout_* log of offset_R/T_L_I)
pose("calib/T_imu_cam")        → /meridian/extrinsic         Odom (NEW: online camera extrinsic)
pose("frontend/gnss_anchor")   → /meridian/gnss              Odom (NEW: GNSS fusion)
timing(stage,ms)               → /meridian/stage_timing      ST   (was printf/CSV aver_time_*  :1009)
event("place/loop_*")          → /meridian/events            Evt  (NEW)
event("map/region_rebuild")    → /meridian/events            Evt  (NEW)
event("frontend/window_restart")→ /meridian/events           Evt  (NEW: recovery made visible)
```
