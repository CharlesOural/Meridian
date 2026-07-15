# Meridian v2 — System and Implementation Specification

Status: implementation contract, pre-implementation revision 0  
Scope: local LiDAR–visual–inertial SLAM, global loop closure, GNSS, ROS 2 adapters, observability, and the hand-off to future dense mapping  
Target: NVIDIA Jetson Orin, ROS 2 Humble, Ouster LiDAR, camera, IMU, and solution-level GNSS/RTK  
Legacy extraction baseline: Git commit `f5ca513158c95aaf88223486ec481c1d42730a21`

This is the sole design specification for Meridian v2. The four operational documents under `docs/` explain how to develop, tune, debug, and test the implementation; they may not redefine the architecture. Numeric defaults in this document are benchmark seeds until a decision gate marks them accepted.

## 1. Mission, constraints, and definition of success

Meridian shall provide a smooth, bounded-latency local pose during normal operation and sensor degradation, plus a revisioned globally consistent pose after GNSS or loop-closure commits. It must use all three local modalities as measurement sources without making either LiDAR or vision the pose authority.

The design deliberately chooses:

- discrete-time smoothing, not continuous-time trajectory optimization;
- one IMU state backbone and one local graph, not independent VIO and LIO pose filters fused afterward;
- modality-specific frontends that fail independently and submit residual-bearing observations to the same estimator;
- a bounded fixed-lag local estimator and a separate sparse global submap graph;
- global corrections through `map -> odom`, never by moving the live local state;
- immutable, revisioned hand-off records and single-writer state owners;
- a ROS-free algorithmic core, with ROS types, QoS, TF, lifecycle, and serialization confined to adapters;
- measured degradation and explicit state machines instead of silent fallback.

The system is successful only if all of these hold on the target Jetson:

1. Local deadlines remain bounded during loop retrieval, global optimization, logging, and dense fusion.
2. Losing any one of camera, LiDAR, or GNSS degrades declared capabilities without corrupting the local trajectory.
3. LiDAR degeneracy removes unsupported LiDAR directions; it does not masquerade as a precise six-degree-of-freedom constraint.
4. A false loop or GNSS multipath burst cannot mutate the committed graph without passing consensus, robust shadow optimization, and commit validation.
5. Live and replay use the same core code and produce the same ordered commits under deterministic settings.
6. Every accepted factor is traceable to immutable measurements, calibration/config revisions, a frontend report, and a graph revision.
7. Memory, queue depth, factor count, and retained payload volume have hard caps.

### 1.1 Current scope

This implementation cycle includes:

- time-normalized IMU, camera, LiDAR, and GNSS ingestion;
- initialization, IMU propagation, LiDAR deskew, visual tracking, LiDAR registration;
- the local fixed-lag graph and high-rate odometry output;
- sparse submap creation and condensation;
- visual and LiDAR place recognition and geometric verification;
- the global submap graph, PCM, GNC, GNSS alignment/reacquisition, and `map -> odom`;
- ROS 2 lifecycle adapters, TF, diagnostics, forensics, replay, and benchmarks.

Dense TSDF/ESDF implementation and the full mission/lifelong map-store service are not part of this cycle. Minimal correctness persistence **is** in scope: the local process owns a durable sparse-seal spool/outbox and the global process owns a durable seal cache plus graph journal/checkpoint. These facilities provide crash consistency and bounded replay only; they do not provide atlas queries, lifelong-map policy, or dense-map persistence. The existing nvblox work is retained as extraction material and connected through a stable contract described in section 13. Path planning is out of scope.

### 1.2 Explicit non-goals

- No continuous-time B-spline or Gaussian-process estimator.
- No learned end-to-end odometry.
- No fusion of two independently optimized VIO/LIO poses or covariances.
- No GNSS factor in the local real-time graph.
- No loop closure against mutable live scans.
- No globally corrected pose fed back into local matching or deskew.
- No online extrinsic/time-offset calibration in the first production implementation. Calibration challengers are benchmarked offline before promotion.
- No assumption that robust kernels solve geometric degeneracy, measurement correlation, or false place recognition.

## 2. Research basis and how references may be used

Meridian is an engineered synthesis, not a line-by-line fork. A reference provides an algorithm, failure lesson, or tested implementation pattern; Meridian contracts remain its own.

| Area | Primary basis | Code used as implementation context | Meridian use |
|---|---|---|---|
| Discrete multi-sensor factor graph | [VILENS](https://arxiv.org/abs/2107.07243), [Unified LiDAR–visual–inertial estimator](https://arxiv.org/abs/2011.06838), [OKVIS2-X](https://arxiv.org/abs/2510.04612) | `../slam-reference/OKVIS2-X`, `../slam-reference/okvis2` | One state backbone; direct modality constraints; keyframe/submap lifecycle |
| IMU preintegration | [Forster et al.](https://arxiv.org/abs/1512.02363), [VINS-Mono](https://arxiv.org/abs/1708.03852) | GTSAM navigation factors; `../slam-reference/OKVIS2-X` | Manifold preintegration, bias correction, initialization and failure checks |
| Fixed-lag LiDAR–IMU graph | [GLIM](https://arxiv.org/abs/2407.10344) | `../slam-reference/glim`, especially `odometry_estimation_imu.cpp` and `odometry_estimation_gpu.cpp` | Five-second seed lag, one update owner, binary direct registration factors and recovery pattern; its unary pattern is reference-only |
| Fast local visual–LiDAR integration | [FAST-LIVO2](https://arxiv.org/abs/2408.14035), [Edge FAST-LIVO2](https://arxiv.org/abs/2501.13876) | `../slam-reference/FAST-LIVO2` | Scheduling and resource benchmark; not its filter state architecture |
| Visual tracking/matching | [SuperPoint](https://arxiv.org/abs/1712.07629), [LightGlue](https://arxiv.org/abs/2306.13643) | OKVIS2-X/OKVIS2 feature paths | Classical default; TensorRT learned challenger |
| LiDAR registration | [VGICP](https://doi.org/10.1109/ICRA48506.2021.9560835), GLIM | `../slam-reference/glim`, `../slam-reference/gtsam_points`, small_gicp | Multiresolution Gaussian voxels; direct factors; CPU degraded lane |
| LiDAR observability | [X-ICP](https://arxiv.org/abs/2211.16335), [Perfectly Constrained](https://arxiv.org/abs/2408.11809) | `../slam-reference/perfectlyconstrained` | Directional localizability, normalized spectra, truncated weak directions |
| LiDAR place recognition | [MapClosures](https://arxiv.org/abs/2501.07399), Scan Context | `../slam-reference/MapClosures`; legacy Scan Context implementation at the baseline commit | MapClosures primary, Scan Context auxiliary and benchmark baseline |
| Visual place recognition | [EigenPlaces](https://arxiv.org/abs/2308.10832), [MixVPR](https://arxiv.org/abs/2303.02190), OKVIS2-X | OKVIS2-X DBoW/PnP path | Compact learned target, conservative DBoW first slice |
| Robust global graph | [PCM](https://robots.engin.umich.edu/publications/jmangelson-2018a.pdf), [GNC](https://arxiv.org/abs/1909.08605), [ROVER](https://arxiv.org/abs/2508.13488) | `../slam-reference/Kimera-RPGO`; legacy Meridian PCM/GNC | Consensus then complete bounded-active-graph shadow solve; trajectory-prior verification is a challenger |
| GNSS | [GVINS](https://arxiv.org/abs/2103.07899), [dropout-tolerant GPS fusion](https://arxiv.org/abs/2208.00709), OKVIS2-X | `../slam-reference/GVINS`, OKVIS2-X `ViGraph.cpp`; legacy Meridian geodesy/Jacobians | Robust 4-DoF alignment, exact-time lever arm, explicit reacquisition |
| Dense hand-off | [nvblox](https://arxiv.org/abs/2311.00626), [Voxgraph](https://arxiv.org/abs/2004.13154), OKVIS2-X | legacy `meridian_map` at the named baseline; imported `src/nvblox` | Immutable local submaps with revisioned global anchors; never loop-driven TSDF reintegration |

Rules for reference reuse:

- Record upstream URL, license, adapted files, and behavioral differences in the implementing package README.
- A copied threshold is only a benchmark seed. It becomes a Meridian default only through section 17's decision process.
- Re-derive frame direction, perturbation side, tangent ordering, residual units, and covariance meaning; never infer them from variable names.
- Port the smallest testable algorithm seam, not an upstream runtime or ROS node.
- Reference snapshots and moving branches are valid implementation context. Any copied code still requires a license check, an explicit adapted-file record, review, and Meridian-owned tests.

## 3. Global architecture

```text
 ROS sensor topics                     ROS-free process boundary records
       │
       ▼
┌──────── meridian_apps local node + meridian_ros adapters ─────────────┐
│ validate schema/QoS/time/calibration; convert; enqueue; publish/TF     │
└──────────────┬───────────────┬────────────────┬────────────────────────┘
               │ IMU           │ camera         │ LiDAR
               ▼               ▼                ▼
        IMU buffer/propagator  visual worker   LiDAR worker
               │               │ tracks/factor  │ deskew/match/factor
               └───────────────┴────────┬───────┘
                                        ▼
                          ┌──── meridian_local_rt ────┐
                          │ coalescer + fixed-lag graph│
                          │ one transactional writer   │
                          └──┬───────────┬─────────────┘
                odom→body ───┘           │ final sparse submap seal
                                         ▼
                   ┌──────────── meridian_global ─────────────┐
                   │ visual/LiDAR retrieval + verification    │
 GNSS ────────────►│ GNSS FSM/alignment                       │
                   │ PCM → GNC bounded full shadow → commit   │
                   └──────────────┬────────────────────────────┘
                                  │ map→odom + anchor revisions
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
             ROS global adapter         future meridian_dense
                    │                    local fusion + anchors
                    ▼
             TF / paths / health
```

There are two factor graphs, with an intentional ownership boundary:

| Graph | Variables | Factors | Rate/lifetime | May affect local odometry? |
|---|---|---|---|---|
| Local fixed-lag | IMU pose, velocity, bias; visual log-inverse-range landmarks | marginal prior, IMU preintegration, bias random walk, visual reprojection, LiDAR direct registration | scan/keyframe rate; nominal five-second lag | It is local odometry |
| Global sparse graph | one SE(3) anchor per sealed sparse submap; one 4-DoF ENU alignment after GNSS initialization | adjacent-submap constraints, rank-aware visual loops, rank-aware LiDAR loops, GNSS antenna-position factors, mission gauge | asynchronous, hard-bounded active mission graph, durably revisioned | No; only publishes `map -> odom` |

The frontends are separate for fault containment and independent benchmarking. They do not own pose states. Each uses IMU prediction, and the LiDAR worker uses IMU for deskew, but both submit state-dependent observations to one local graph.

## 4. Coordinate, time, and uncertainty conventions

### 4.1 Frames and transform direction

`T_A_B` maps coordinates from frame `B` into frame `A`. Composition is left to right: `p_A = T_A_B * p_B`. Public aliases shall not hide frame direction.

Required frames:

- `map`: fixed, gravity-aligned global coordinate frame owned by `meridian_global`; anchor and `map -> odom` **estimates** are revisioned, but the frame definition is never silently changed.
- `odom`: continuous, gravity-aligned local frame owned by `meridian_local_rt` for one `OdomEpoch`.
- `base_link`: robot body frame exposed to ROS.
- `imu_link`: estimator body state; `T_base_imu` is a fixed calibrated transform. The equation shorthand `B` (body) means exactly `imu_link`, never `base_link`; `T_O_B`, `T_B_C`, and `p_B_antenna` therefore use the IMU-state frame.
- `lidar_link`, `camera_<id>`, `gnss_link`: calibrated sensor frames.
- `submap_<id>`: immutable, gravity-aligned local frame of a sealed sparse/dense submap. Its origin is the finalized boundary-body origin, its `+z` follows gravity-up, and its yaw is the finalized boundary-body yaw. Boundary roll and pitch do not tilt the submap axes.

Exactly one authority publishes each TF edge:

- robot description/static calibration: base/sensor static transforms;
- local node: `odom -> base_link`;
- global node: `map -> odom`;
- no other component publishes these edges.

### 4.2 Perturbations and tangent order

Core SE(3) uses a documented right perturbation by default:

`T(δξ) = T Exp(δξ)`, with the Sophus-compatible translation-first tangent `δξ = [δp_x, δp_y, δp_z, δθ_x, δθ_y, δθ_z]`.

Every covariance/information type carries a compile-time semantic tag or named accessor. Raw `Matrix6d` is forbidden at public boundaries. GTSAM adapters reorder explicitly and have round-trip unit tests. Each custom factor must pass central finite-difference Jacobian tests over nominal and adversarial poses.

### 4.3 Time

All fusion time is signed integer nanoseconds in a selected monotonic fusion clock. Domain records never use ROS time types. Each sample retains:

- raw device stamp;
- mapped fusion stamp;
- host arrival stamp;
- immutable `ClockRevision`;
- source epoch, adapter ingress sequence, and the device sequence only when the source actually supplies one;
- uncertainty and status of the time mapping.

Intervals are half-open `[start, end)`. A boundary measurement belongs to the later interval. Camera time is the calibrated exposure midpoint. LiDAR sweep metadata states whether the ROS header is first return, last return, or another driver convention; per-point time is mandatory. Timestamp interpolation between discrete states is allowed and named. It is not continuous-time optimization.

`ClockRevision` is an immutable, content-addressed raw-device-to-fusion-clock mapping for one source epoch. It is separate from `CalibrationEpoch`. A newly estimated clock revision applies only to records that have not yet been admitted; the mapped fusion stamp of an admitted record is immutable. Successive revisions may coexist in one local window only when their mappings are continuous at the revision boundary, preserve ordering, and carry bounded mapping uncertainty. That uncertainty is propagated into the affected measurement model. A revision that would move or reorder an admitted sample, a clock discontinuity, non-monotonic stamps, missing per-point time, or a change of driver stamp convention causes a typed rejection and new source epoch, never silent remapping or clamping.

### 4.4 Calibration

A `CalibrationEpoch` is immutable and content-addressed. It contains intrinsics, distortion model, sensor extrinsics, IMU noise/random-walk densities, gravity convention, LiDAR timing and fixed hardware-latency conventions, camera exposure convention, and GNSS lever arm. The evolving raw-to-fusion clock mapping is owned by `ClockRevision`, not by calibration. A local raw-measurement factor may use only its explicitly named calibration epoch. A global factor may connect submaps produced under different epochs only when both endpoint refs and the full calibration set/manifest are explicit; it never reinterprets either endpoint under the other epoch. A calibration change drains the local transaction, seals or aborts the current sparse submap, increments the epoch, and reinitializes affected frontend state.

Online calibration is initially disabled. Offline observability and calibration benchmarks must demonstrate improvement without harming weak-motion sequences before any variable is admitted to a production graph.

### 4.5 Uncertainty

- Residual covariance describes the measurement/model error represented by that factor, not optimizer confidence in general.
- A registration Hessian inverse is not automatically a calibrated covariance.
- Rank-deficient measurements use `RankAwareInformation {basis, eigenvalues, rank, nullspace_policy}`; unsupported eigenvalues are zero, not a tiny fabricated precision.
- Shared source measurements are tracked through `MeasurementManifest`; the graph prevents duplicate evidence or applies explicit correlation inflation/information caps.
- Full-rank covariance/information matrices are checked for finiteness, symmetry, PSD, condition, units, frame, and tangent convention at ingress. A rank-aware matrix is instead checked for its declared numerical rank, orthonormal basis, zero unsupported eigenvalues, and the condition number of its supported nonzero subspace; its ordinary full-space condition number is intentionally infinite.

## 5. Software architecture and repository shape

The target source tree is:

```text
src/
  meridian_cmake/       shared toolchain and test helpers
  meridian_core/        IDs, time, geometry, calibration, contracts, queues, telemetry API
  meridian_local_rt/    IMU, visual, LiDAR, coalescer, local fixed-lag graph
  meridian_global/      sparse submaps, retrieval, verification, GNSS, PCM/GNC graph
  meridian_dense/       later: nvblox/CPU dense integration behind the section 13 seam
  meridian_map_store/   later: atlas/lifelong catalog, tile/blob lifecycle, migration from minimal spools
  meridian_msgs/        ROS interface definitions only
  meridian_ros/         lifecycle nodes and ROS/domain conversion only
  meridian_apps/        ROS composition roots; constructs local/global runtimes and adapters
  meridian_tools/       replay/evaluation/forensic executables using the same core
```

`meridian_place`, `meridian_backend`, `meridian_pipeline`, and the many v1 cross-cutting packages do not return as separate packages. Their useful algorithms are extracted into the owner above.

Dependency rules:

```text
meridian_core      -> Eigen, Sophus, fmt (no ROS, GTSAM, OpenCV, CUDA)
meridian_local_rt  -> core, GTSAM adapter-private, OpenCV, optional CUDA backends
meridian_global    -> core, GTSAM adapter-private, OpenCV, small_gicp, optional TensorRT
meridian_dense     -> core, CUDA/nvblox (or CPU oracle); never local/global internals
meridian_map_store -> core, persistence libraries; never optimizer internals
meridian_msgs      -> ROS IDL only
meridian_ros       -> core public contracts, meridian_msgs, ROS; no algorithm implementations
meridian_apps      -> core, local_rt/global, meridian_ros, ROS; composition only
meridian_tools     -> public contracts; may link algorithm packages for replay
```

Enforced build checks:

- `rg`-based CI guard: no ROS include or package dependency in core/local/global/dense/store.
- no GTSAM type in a public header; `LocalGraphAdapter` and `GlobalGraphAdapter` are private pImpls.
- no CUDA header outside backend-private translation units.
- dependency direction check and package-level cycle test.
- format, warnings-as-errors, clang-tidy, sanitizers on CPU, CUDA memcheck subset, and license scan.

### 5.1 Process and thread ownership

Initial deployment uses two required processes and one optional process:

1. `meridian_local_node`: an executable composition root in `meridian_apps` containing the ROS adapter, `meridian_local_rt`, and the local `SealSpool`/outbox.
2. `meridian_global_node`: an executable composition root in `meridian_apps` containing the global ROS adapter, `meridian_global`, its durable seal cache, and its `GraphJournal`.
3. Later `meridian_dense_node`: dense/store; failure must not stop either estimator.

The local process has bounded owners:

- ROS ingress/conversion workers;
- one IMU buffer/propagation owner;
- one visual worker;
- one LiDAR preprocessing/matching worker, with a bounded GPU stream;
- one local graph writer;
- one publisher/snapshot reader.

The global process has:

- seal/index coordinator;
- visual retrieval worker;
- LiDAR retrieval worker;
- bounded verification pool;
- GNSS FSM owner;
- one global graph transaction writer;
- asynchronous journal/checkpoint writer;
- snapshot/TF publisher.

No callback performs image decode, point-cloud conversion beyond bounded parsing/copy, feature extraction, registration, CUDA synchronization, graph optimization, or disk I/O. No module starts an unregistered background thread.

### 5.2 State, records, and queues

Mutable state has one writer. Readers receive immutable `shared_ptr<const Snapshot>` objects atomically. Cross-thread messages are move-only records or immutable blob handles. A queue declares:

- maximum count and byte budget;
- maximum acceptable age;
- ordering key;
- overflow policy: reject newest, drop oldest, coalesce by key, or force recovery;
- whether loss is allowed;
- telemetry for wait, service, drops, and oldest age.

Sensor queues prefer newest valid data and reject stale work. State/commit/seal queues are loss-intolerant and use an application journal/replay protocol; DDS reliability alone is not persistence.

### 5.3 Error model

Expected runtime outcomes use typed results such as `Accepted`, `Rejected<Reason>`, `Deferred<Reason>`, and `Degraded<Capability>`. Exceptions are reserved for construction/programming failures and are contained at process boundaries. No boolean-only factor admission API is allowed.

Every public record starts with identity/configuration metadata. Raw observations additionally carry source-clock metadata; odometry association is explicit only on records that actually depend on an odom epoch.

```cpp
struct RecordHeader {
  SchemaVersion schema;
  TraceId trace;
  ProducerId producer;
  SessionId session;
  FusionTime created_at;
  ConfigRevision config;
  optional<CalibrationEpoch> direct_calibration;
};

struct SourceStamp {
  RawDeviceTime raw_time;
  FusionTime fusion_time;
  ArrivalTime host_arrival_time;
  ClockRevision clock_revision;
  SourceEpoch source_epoch;
  optional<SourceSequence> device_sequence;  // only when supplied by the device/driver
  IngressSequence ingress_sequence;          // always generated monotonically by the adapter
  TimeMappingUncertainty uncertainty;
  TimeMappingStatus status;
};
```

IDs are strong types. `(producer, source_epoch, ingress_sequence)` uniquely identifies delivered sensor input. `device_sequence`, when present, audits upstream loss; it is never confused with the adapter-generated ingress sequence. When no device sequence exists, expected stamp gaps are explicitly labeled an inferred upstream-loss signal. Graph/submap/commit IDs are monotonic within their owner and persisted.

Every raw sensor record and every single-calibration frontend record requires `direct_calibration`. A multi-input record spanning different epochs leaves that header field empty and carries the exact `CalibrationEpochSet` through its endpoint references and `MeasurementManifest`; absence never means “use the latest calibration.” A sparse submap contains one calibration epoch because a calibration change closes or aborts it. A loop may connect two different epochs, while a `GlobalCommit` explicitly lists the union of epochs touched by its changed anchors/factors.

`MeasurementManifest` is the binding duplicate/correlation ledger, not an opaque list of IDs:

```cpp
enum class MeasurementRole {
  PrimaryResidual,       // contributes rows to an objective
  ConditioningOnly,      // affects deskew, association, covariance, or an initial value
  RetrievalSeedOnly,     // schedules verification; contributes no graph information
  DerivedSummary,        // cache/diagnostic product; retains transitive ancestry
};

using RawMeasurementKey = variant<MeasurementId, GnssObservationId>;

struct MeasurementSlice {
  RawMeasurementKey root;         // tagged MeasurementId | GnssObservationId
  SliceSelector selector;          // Whole, half-open index/time range, or immutable bitmap
  ContentHash source_checksum;
  CalibrationEpoch calibration;
};

struct MeasurementUse {
  MeasurementSlice slice;
  MeasurementRole role;
  DerivedRecordId consumer;
  optional<FactorGroupId> factor_group;
  optional<CorrelationGroupId> correlation_group;
};

enum class CorrelationTreatment {
  JointCompositeWhitening,
  CovarianceInflationAndInformationCap,
  NotIndependent,
};

struct CorrelationDeclaration {
  CorrelationGroupId group;
  CorrelationPolicyRevision policy;
  CorrelationTreatment treatment;
  double covariance_inflation;             // finite and >= 1 when applicable
  optional<InformationCapRef> information_cap;
};

struct MeasurementManifest {
  ManifestId id;
  vector<MeasurementUse> uses;              // canonical sorted order
  vector<CorrelationDeclaration> correlations;
  ContentHash checksum;
};
```

Selectors have decidable overlap; a malformed selector or unknown overlap is conservatively overlapping. A derived record keeps raw-root slices transitively—derivation never creates independent information. Two overlapping `PrimaryResidual` slices may enter one objective only as one declared factor group with joint whitening or a calibrated inflation/information-cap treatment; otherwise admission rejects `DUPLICATE_MEASUREMENT`. Reusing a slice as `ConditioningOnly` across factors requires a declared correlation group and its treatment. Retrieval seeds never contribute precision. “Independent” loop/GNSS support means disjoint transitive primary **and conditioning** root slices; unknown or overlapping ancestry is `NotIndependent`. Manifest composition is canonical union plus correlation declarations, so submap condensation and marginalization preserve the same ledger rather than replacing it with a new untraceable ID.

`LidarCorrelationPolicy` is a typed specialization naming its `CorrelationPolicyRevision`, factor group, conditioning groups (IMU deskew, target reuse, clock, calibration), calibrated covariance inflation, per-mode and total per-sweep information caps, and the validation dataset/profile that selected them. Inflation is applied before whitening; caps are applied after the section 11.4 supported-subspace projection. A missing group, factor-of-one placeholder, or cap with incompatible frame/tangent semantics is a typed admission failure.

## 6. Canonical cross-module contracts

The exact C++ layout is finalized in implementation slice 0, but the semantics below are binding.

### 6.1 Sensor observations

```cpp
struct ImuSample {
  RecordHeader header;
  MeasurementId id;
  SourceStamp stamp;
  Vec3 specific_force_mps2;
  Vec3 angular_velocity_radps;
  ImuStatus status;
};

struct CameraObservation {
  RecordHeader header;
  MeasurementId id;
  CameraId camera;
  SourceStamp stamp;
  FusionTime exposure_midpoint;
  Duration exposure;
  ImageLayout layout;
  BlobRef pixels;
};

struct LidarSweep {
  RecordHeader header;
  MeasurementId id;
  LidarId lidar;
  SourceStamp stamp;
  TimeRange acquisition;
  LidarPointLayout layout;
  BlobRef points;
  LidarTimingConvention timing;
};

struct GnssObservation {
  RecordHeader header;
  GnssObservationId id;
  SourceStamp stamp;
  GeodeticPosition wgs84;
  PositionCovarianceEnu covariance;
  GnssSolutionType solution;
  optional<Duration> correction_age;
  optional<double> hdop, vdop;
  optional<uint16_t> satellites;
  GnssStatus status;
};
```

Unknown GNSS fields remain unknown. `NavSatFix.status` alone cannot invent float/fixed RTK state; a receiver-specific adapter may enrich the domain record.

### 6.2 Frontend outputs

The suffix `Evidence` is not used. A pixel/point measurement is an observation; a residual is produced only when evaluated against a state.

```cpp
struct KnotRequest {
  RecordHeader header;
  KnotRequestId id;
  KnotRequestKind kind;                 // visual keyframe, LiDAR reference, or IMU guard
  FusionTime exact_time;
  MeasurementManifest measurements;
};

struct KnotResolution {
  RecordHeader header;
  KnotResolutionId id;
  KnotRequestId request;
  OdomEpoch odom_epoch;
  StateId state;
  FusionTime exact_time;
  LocalGraphRevision created_at_revision;
  vector<KnotRequestId> exactly_shared_requests;
};

struct VisualObservationBatch {
  RecordHeader header;
  CameraFrameId frame;
  FusionTime exposure_midpoint;
  optional<KnotResolutionId> factor_knot;
  vector<TrackObservation> tracks;
  VisualQuality quality;
  MeasurementManifest measurements;
};

struct LidarResidualFactorSpec {
  RecordHeader header;
  SweepId sweep;
  KnotResolutionId source_knot;
  vector<ResolvedRegistrationTargetRef> targets;
  DeskewRevision source_deskew;
  BlobRef source_gaussians;
  CorrespondencePolicy policy;
  CorrespondenceBudget budget;
  LidarCorrelationPolicy correlation;
  optional<FrontendLocalizabilityHint> frontend_hint;
  LidarQuality quality;
  MeasurementManifest measurements;
};
```

`KnotRequest` and `KnotResolution` form a two-phase handshake. A modality requests a state at an exact fusion timestamp; only the graph owner allocates the `StateId` and returns a resolution. A factor spec may be emitted only after that resolution exists and must cite it. Two sensor requests share a state only when `exact_time` is bit-identical. An IMU guard is not a measurement and may instead be cancelled when another exact state already satisfies the interval cap; cancellation never shifts a sensor timestamp.

The visual worker emits tracking observations for every image, but only a keyframe batch with `factor_knot` may construct reprojection factors. It does not emit a VIO pose. The LiDAR worker emits immutable source/target references and matching diagnostics; it does not emit an LIO pose/covariance factor derived from an independent filter. `frontend_hint` is diagnostic only. The graph recomputes authoritative LiDAR localizability from current correspondences and robust weights at every factor linearization.

### 6.3 Local commit

```cpp
struct LocalCommit {
  RecordHeader header;
  OdomEpoch odom_epoch;
  LocalGraphRevision revision;
  LocalGraphRevision parent;
  StateId state;
  FusionTime state_time;
  NavStateEstimate estimate;             // T_odom_imu, v, biases
  PoseCovariance local_pose_covariance;
  CapabilitySet capabilities;
  LocalSolveReport solve;
  vector<FactorDisposition> dispositions;
  MeasurementManifest measurements;
};
```

A commit is atomic. Readers either observe the old or new snapshot. A failed transaction publishes a report/event but no state mutation. High-rate propagated odometry between commits names its anchoring `LocalGraphRevision` and is replaceable by the next optimized state.

### 6.4 Sparse global hand-off

Large payloads cross thread/process boundaries through an explicit immutable reference:

```cpp
enum class BlobStorage {
  InProcessPool,
  SharedMemoryLease,
  DurableSpool,
};

struct BlobRef {
  BlobStoreId store;
  BlobId id;
  ContentHash checksum;
  LayoutId layout;
  uint64_t bytes;
  BlobStorage storage;
  optional<LeaseToken> lease;  // required only for expiring shared memory
};

struct BlobStoreDescriptor {
  BlobStoreId store;
  ProducerId owner;
  StoreInstanceEpoch instance;
  BlobTransportCapabilities capabilities;
  TransportEndpointId endpoint;  // resolved by deployment, not an algorithmic filesystem path
};

struct BlobReadRequest {
  BlobStoreId store;
  BlobId id;
  ContentHash expected_checksum;
  ByteRange range;
};

struct TrajectoryKnot {
  StateId state;
  FusionTime time;
  Pose T_submap_imu;
  Vec3 velocity_submap;
  PoseCovariance conditional_pose_covariance;
};

struct SealedTrajectorySegment {
  TrajectorySegmentId id;
  StateId before, after;
  TimeRange support;
  InterpolationPolicyRevision interpolation_policy;
  JointEndpointPoseVelocityCovariance joint_covariance;
  InterpolationModelNoise model_noise;
  MeasurementManifest support_measurements;
};

struct SealedTrajectoryLayout {
  vector<TrajectoryKnot> knots;
  vector<SealedTrajectorySegment> segments;
  ContentHash checksum;
};

struct MaturePlaceLandmark {
  LandmarkId id;
  Vec3 position_submap;
  PointCovariance covariance;
  BlobSlice descriptor;
  vector<FeatureObservationRef> observations;
  LandmarkQuality quality;
};

struct VisualPlaceKeyframe {
  KeyframeId id;
  CameraFrameId frame;
  FusionTime exposure_midpoint;
  Pose T_submap_camera;
  CameraModelRevision camera_model;
  BlobRef keypoints_and_descriptors;
  BlobRef mature_landmarks;
  optional<BlobRef> immutable_image;
  VisualQuality quality;
};

struct VisualPlaceManifest {
  BlobRef keyframes;                // array of VisualPlaceKeyframe
  DescriptorModelRevision model;   // may be Uncomputed
  optional<BlobRef> global_descriptors;
};

enum class SummaryStability { Provisional, Final };

struct GlobalKeyframeSummary {
  RecordHeader header;
  OdomEpoch odom_epoch;
  KeyframeId id;
  SubmapId provisional_submap;
  LocalGraphRevision local_revision;
  SummaryStability stability;
  FusionTime time;
  Pose T_provisional_submap_imu;
  optional<BlobRef> visual_place_keyframe;
  optional<BlobRef> lidar_keyframe;
  FusionTime valid_until;
};

struct SubmapFrameDefinition {
  StateId boundary_state;
  FusionTime boundary_time;
  Vec3 gravity_up_odom;
  double boundary_yaw_odom;
};

struct SubmapRef {
  SessionId session;
  OdomEpoch odom_epoch;
  SubmapId id;
  CalibrationEpoch calibration;
  SubmapContentRevision content_revision;
  ContentHash local_content_checksum;
};

struct RelativeSubmapConstraint {
  SubmapRef from, to;
  Pose T_from_to;                       // maps `to` coordinates into `from`
  RankAwareInformation information;    // right, translation-first tangent of T_from_to
  MeasurementManifest measurements;
  ContentHash checksum;
};

struct SparseSubmapSeal {
  RecordHeader header;
  OdomEpoch odom_epoch;
  SubmapId id;
  optional<SubmapRef> previous;
  SubmapContentRevision content_revision;
  ContentHash local_content_checksum;
  LocalGraphRevision final_local_revision;
  TimeRange support_time;
  SubmapFrameDefinition frame;
  Pose T_odom_submap;
  BlobRef internal_trajectory;       // SealedTrajectoryLayout, final and time ordered
  BlobRef keyframe_manifest;         // exact visual/LiDAR/source record membership
  BlobRef registration_proxy;       // bounded surfel/Gaussian cloud
  BlobRef dense_input_manifest;      // exact canonical frames available to future dense
  VisualPlaceManifest visual_place;
  optional<RelativeSubmapConstraint> from_previous;
  MeasurementManifest measurements;
  SubmapQuality quality;
  ContentHash seal_checksum;
};
```

`BlobStoreId` is stable for a producer/session even if its process restarts; `StoreInstanceEpoch` detects a stale endpoint. The composition root publishes/resolves a `BlobStoreDescriptor`, then the common blob protocol provides idempotent `stat(ref)`, bounded/ranged `read(BlobReadRequest)`, `acquire(ref, ConsumerId)`, and `release(AcquisitionToken)` operations. `acquire` creates a durable consumer pin for final spool content and is idempotent by `(ConsumerId, BlobStoreId, BlobId, checksum)`; `release` is also idempotent. Reads return the named immutable bytes or a typed `NOT_FOUND`, `STALE_STORE_INSTANCE`, `LEASE_EXPIRED`, or integrity error and never substitute another object with the same layout. Every complete read is byte-count and checksum verified before deserialization.

`DurableSpool` objects remain readable across owner restart until the seal-ACK/consumer-pin GC rules release them. The store reconstructs its object index and durable acquisitions before advertising a new instance. `SharedMemoryLease` is valid only through its named store and lease token and cannot appear in a final seal. A client may retry `stat/read/acquire/release` after transport failure without changing ownership; copying an object into another durable store creates a new local `BlobRef` with the same content checksum and explicit copy provenance. Absolute paths, raw pointers, CUDA handles, and process-local file descriptors never cross this contract.

`GlobalKeyframeSummary` is replaceable cache input only. Provisional poses, submap IDs, descriptors, and blobs may be superseded or expire and can never support a graph factor. A `Final` summary is still not graph evidence until the matching `SparseSubmapSeal` pins it.

Every blob in the transitive `BlobRef` closure named by a seal is `DurableSpool` content or is copied into durable spool content before the seal is published. Shared-memory leases are allowed for provisional summaries and debug only. A `BlobRef` checksum covers exactly `bytes` under the named immutable layout; each manifest checksum covers its ordered entries and child references, and the parent seal checksum covers its scalar fields and ordered direct-child `(store, id, checksum, layout, bytes)` tuples.

`local_content_checksum` is computed before the adjacent constraint and covers the immutable frame definition, support, trajectory, place/registration/dense manifests, measurements, and quality. This makes a non-circular `SubmapRef`; `seal_checksum` covers that local content plus `previous`, `from_previous`, and the seal envelope. `(SessionId, OdomEpoch, SubmapId, SubmapContentRevision)` is unique. Re-delivery with the same `seal_checksum` is idempotent. The same identity with a different seal checksum is an integrity fault that quarantines the producer epoch; it is never treated as another valid revision.

When present, `from_previous.from` equals `previous` and `from_previous.to` equals the current seal's derived `SubmapRef`, including both calibration epochs, revisions, and local-content checksums. Its transform maps current-submap coordinates into previous-submap coordinates. Its rank-aware basis/information use the section 4.2 right, translation-first relative tangent, and its manifest is the exact disjoint local factor partition. Endpoint mismatch, a checksum conflict, or a basis in another frame is rejected before graph insertion; the constraint is not a heuristic odometry covariance sum.

### 6.5 Global proposals and commit

```cpp
struct LoopMeasurement {
  RecordHeader header;
  ProposalId proposal;
  LoopModality modality;
  SubmapRef from, to;
  CalibrationEpochSet calibration_epochs;
  Pose T_from_to;
  RankAwareInformation information;
  VerificationReport verification;
  optional<PlaceSeed> retrieval_seed;
  MeasurementManifest measurements;
};

struct GnssDatumRevision {
  DatumRevision revision;
  GeodeticPosition enu_origin_wgs84;
  GnssObservationIdSet source_window;
  ContentHash source_checksum;
};

// Maps map coordinates into local ENU: p_enu = Rz(yaw_enu_from_map) p_map + translation_enu.
struct YawTranslation4 {
  Vec3 translation_enu;
  double yaw_enu_from_map;
};

struct GnssAlignmentRevision {
  RecordHeader header;
  AlignmentRevision revision;
  CalibrationEpochSet calibration_epochs;
  optional<GnssDatumRevision> datum;
  optional<YawTranslation4> T_enu_map;
  optional<Covariance4TranslationYaw> covariance;
  GnssObservationIdSet fit_source_window;
  ContentHash fit_source_checksum;
  GnssFusionState state;
  FusionTime state_since;
  TransitionReason reason;
  optional<TimeRange> quarantine_range;
  optional<GnssObservationId> last_admitted_observation;
};

struct SealedTrajectoryBracket {
  TrajectorySegmentId segment;
  StateId before, after;
  TimeRange support;
  double interpolation_fraction;
  InterpolationPolicyRevision interpolation_policy;
  ContentHash joint_covariance_checksum;
  LocalGraphRevision final_local_revision;
  ContentHash trajectory_checksum;
};

struct GnssFactorProposal {
  RecordHeader header;
  ProposalId proposal;
  CalibrationEpochSet calibration_epochs;
  SubmapRef submap;
  GnssObservationId observation;
  FusionTime time;
  SealedTrajectoryBracket bracket;
  Vec3 antenna_position_submap;
  PointCovariance local_interpolation_covariance;
  Vec3 measured_position_enu;
  PositionCovarianceEnu effective_covariance;
  MeasurementManifest measurements;
};

enum class MapOdomCovarianceSemantics {
  ConditionalOnSealedLocalFrame,
};

enum class MapOdomState { Aligned, Unaligned, StaleOrInvalid };

struct MapOdomStatus {
  OdomEpoch odom_epoch;
  MapOdomState state;
  optional<SubmapRef> reference_submap;
  TransitionReason reason;
};

struct SubmapAnchorRevision {
  RecordHeader header;
  GlobalGraphRevision graph_revision;
  SubmapRef submap;
  Pose T_map_submap;
  PoseCovariance covariance;
  FusionTime valid_from;
};

struct MapOdomTransform {
  GlobalGraphRevision graph_revision;
  OdomEpoch odom_epoch;
  SubmapRef reference_submap;
  Pose T_map_odom;
  optional<PoseCovariance> covariance;
  MapOdomCovarianceSemantics covariance_semantics;
  FusionTime valid_from;
  GlobalHealth health;
};

struct GlobalCommit {
  RecordHeader header;
  GlobalGraphRevision revision, parent;
  CalibrationEpochSet calibration_epochs;
  vector<SubmapAnchorRevision> changed_anchors;
  MapOdomStatus map_odom_status;
  optional<MapOdomTransform> map_odom;
  optional<GnssAlignmentRevision> gnss_alignment;
  vector<FactorDisposition> dispositions;
  GlobalSolveReport solve;
  JournalOffset durable_at;
};
```

Every binary adjacent-submap or loop factor pins both endpoint `SubmapRef`s, including calibration epochs, content revisions, and local-content checksums. A GNSS factor pins its one `SubmapRef`, observation, trajectory segment, and `G` alignment revision; the mission gauge and an independent surveyed-`G` prior have their own explicit manifests. Visual and LiDAR loop measurements remain separate. If both use correlated local geometry, enforce a pair-level information cap or calibrated correlation inflation; never average them into one opaque pose.

`Covariance4TranslationYaw` uses `[t_E, t_N, t_U, yaw]` order. The alignment revision is present whenever the datum, `T_enu_map`, covariance, source window, quarantine bounds, last admitted observation, or GNSS FSM state changes. Its covariance is an observability/reporting result unless a separately manifested independent survey prior is configured; it is not silently inserted as a graph prior. A commit may legitimately carry `map_odom_status = Unaligned` and no `map_odom`; no placeholder identity pose is serialized.

### 6.6 Minimal correctness persistence

The current cycle implements three deliberately small persistence components:

- `SealSpool` in the local process: a section 6.4 blob store containing content-addressed immutable sparse blobs plus seal records;
- `SealOutbox` in the local process: monotonically sequenced, replayable seal announcements and global acknowledgements;
- `GlobalSealCache` and `GraphJournal` in the global process: a durable copy of all sparse payload needed by the active retrieval/verification database, graph checkpoints, transactions, and commit markers.

They are bounded correctness infrastructure, not the future `meridian_map_store`. They expose no atlas/tile/lifelong-map API. Their byte, seal, graph-node, and journal limits are deployment configuration and drive explicit capability degradation rather than deletion of unacknowledged evidence.

Seal publication order is:

1. write every blob in the seal's transitive child closure to a temporary spool object, verify checksum, atomically rename, and durably flush it;
2. write and durably flush the seal record that names those blobs;
3. append and durably flush the outbox sequence entry;
4. publish the seal metadata over ROS;
5. global resolves the seal's `BlobStoreId`, idempotently acquires/reads every required blob, verifies byte counts/layouts/checksums, copies and pins the closure into `GlobalSealCache`, then durably records ingestion;
6. global acknowledges `(outbox_sequence, submap identity, seal_checksum)` only after its durable copy is committed, then idempotently releases the temporary source-store acquisitions;
7. local may garbage-collect only acknowledged objects not pinned by another consumer and older than the configured recovery retention.

An expired shared-memory lease can never satisfy step 5; global requests replay from the durable local spool. If either spool reaches its hard limit before safe garbage collection, new sparse-map growth stops with `DEGRADED_STORAGE_CAPACITY`; local odometry continues.

Graph transaction crash ordering is:

1. solve and validate a candidate transaction against durable parent `R`—incremental for a trusted adjacent-only insertion or a complete bounded shadow for robust/global changes;
2. append a `PREPARED` record containing input IDs, parent, dispositions, solution/checkpoint hash, and write it durably;
3. recheck the parent under the single graph writer;
4. append the complete `COMMITTED(R+1)` delta/checkpoint reference and durable commit marker;
5. only after the commit-marker flush succeeds, atomically publish the in-memory snapshot and ROS outputs.

A crash before step 4 discards the prepared transaction. A crash after step 4 recovers and publishes `R+1`, even if ROS publication never occurred. The asynchronous journal worker may perform I/O, but the graph writer waits for its durability acknowledgement before publication; it never blocks the local process. ACK/replay and `get revisions since N` operations are idempotent and checksum-verified.

A journal segment is garbage-collected only after a complete graph/GNSS checkpoint and referenced-seal manifest have been checksum-verified, durably flushed, and selected by an atomically replaced durable checkpoint pointer. Retain the preceding complete checkpoint until the next checkpoint is proven recoverable. If safe compaction cannot complete before the journal budget, global becomes read-only/capacity-degraded; it never deletes the only committed recovery path.

## 7. Lifecycle and recovery state machines

### 7.1 Local estimator

```text
UNCONFIGURED → WAITING_FOR_TIME_AND_CALIBRATION → IMU_INITIALIZING
IMU_INITIALIZING → TRACKING
TRACKING → DEGRADED_VISUAL | DEGRADED_LIDAR | IMU_ONLY_HOLD
any active state → RECOVERING → TRACKING
unrecoverable time/calibration/IMU fault → RESETTING(new OdomEpoch)
```

Initialization requires a contiguous IMU interval, finite noise configuration, gravity/bias plausibility, and consistent clocks. Static initialization estimates gravity direction and biases; motion initialization is a later challenger. Camera and LiDAR may join after the IMU backbone is running. An initialization timeout is reported, not bypassed with zero biases.

`IMU_ONLY_HOLD` is prediction with rapidly growing uncertainty and a short configured horizon; it is not healthy SLAM. If the horizon or uncertainty cap is exceeded, the node stops claiming a valid localization capability and starts a new odom epoch only under explicit reset policy.

### 7.2 GNSS

```text
UNINITIALIZED ──quality window──► ALIGNING ──observable fit──► TRUSTED
UNINITIALIZED ──stream timeout/disabled──► UNAVAILABLE
ALIGNING ──timeout/no stream──► UNAVAILABLE
TRUSTED ──innovation/quality failure streak──► SUSPECT
TRUSTED ──stream timeout──► UNAVAILABLE
SUSPECT ──qualified moving window──► REACQUIRING
SUSPECT ──stream timeout──► UNAVAILABLE
REACQUIRING ──shadow commit passes──► TRUSTED
REACQUIRING ──validation fails──► SUSPECT
REACQUIRING ──stream timeout──► UNAVAILABLE
UNAVAILABLE ──stream returns, no committed alignment──► ALIGNING
UNAVAILABLE ──stream returns, committed alignment──► REACQUIRING
```

All transitions have hysteresis, a typed reason, timestamps, and forensic snapshot. `SUSPECT` and `REACQUIRING` observations are quarantined; they do not enter the committed graph unless a complete reacquisition shadow transaction commits them atomically. GNSS is never permanently auto-disabled after a failure streak. Datum, alignment, covariance, FSM state, quarantine bounds, and last admitted observation are checkpointed through `GnssAlignmentRevision`.

### 7.3 Global graph

The global graph restores the last durable `GraphJournal` revision and `GnssAlignmentRevision`, verifies referenced `GlobalSealCache` hashes, catches up missing local outbox seals, and only then becomes writable. Journal/cache failure makes it read-only/degraded; it does not block local odometry. A crash during a shadow solve cannot expose partial anchors. Full atlas loading, cross-mission session alignment, and lifelong map restoration remain future `meridian_map_store` capabilities and are not claimed by this revision.

The first seal of the first `OdomEpoch` may enter an empty graph with the single mission gauge. A reset starts a new epoch whose first seal has no `previous`/`from_previous`; all its seals are durably cached and indexed as `PendingUnconnected`, but their anchors and within-epoch adjacent chain do not enter the committed graph and produce no `map -> odom`. Global ACK of a seal means durable cache ownership, not graph admission.

A pending epoch becomes connected only in one complete bounded shadow transaction that adds its pending anchors and adjacent factors together with either (a) a verified loop to an already connected `SubmapRef`, or (b) a qualified GNSS batch that connects it through the committed `G` alignment. The connector and pending chain must jointly pass rank/gauge, consensus/robustness, manifest, capacity, and full connectedness checks. No second mission gauge or artificial bridge prior is permitted. Failure leaves the component `PendingUnconnected`; success atomically marks it `CommittedConnected` and may publish the first transform for that epoch. If the graph is intentionally started as a new standalone mission rather than a continuation, that is a new `SessionId` and graph/store instance, not a hidden second gauge in the existing mission.

## 8. Sensor admission, synchronization, and IMU backbone

### 8.1 Adapter admission

Each ROS adapter first validates the complete wire schema and byte bounds, converts into a domain record, and only then offers it to ingress. Validation includes:

- required field names, datatypes, offsets, endianness, row/point strides, and buffer bounds;
- finite physical values and sensor saturation flags;
- image encoding, step, payload size, and calibrated dimensions;
- per-point LiDAR time and ring/range semantics;
- monotonic stamp/sequence behavior and host-arrival delay;
- calibration, clock, and source epoch existence.

Malformed input is rejected with a reason and a bounded sample of metadata; the core never sees a partially initialized record. The Ouster parser and image conversions at the legacy baseline are extraction candidates, but must gain malformed-message, padded-row, alternate-time-field, endian, and golden-packet tests before reuse.

### 8.2 Independent event streams

The local runtime admits independent `ImuSample`, `CameraObservation`, and `LidarSweep` streams. There is no LiDAR-owned `MeasureGroup`.

Each source has a bounded reorder window and publishes a monotonic `ProcessedWatermark`. A watermark at time `t` means every admitted record at or before `t` in that source epoch has reached a terminal frontend decision, or the source has been explicitly declared unavailable through `t` after the configured maximum-lateness bound. Live and replay advance watermarks from the same recorded arrival stamps, source epochs, and lateness configuration. A LiDAR sweep schedule is not finalized until the camera processed watermark has passed the sweep end; it therefore cannot depend on worker completion order.

Once the local scheduling watermark passes a timestamp, a late record may add a factor only to already resolved live knots. A record that would require inserting or moving a knot behind that watermark is rejected as `SCHEDULE_FINALIZED`, even if surrounding states remain in the lag. This deliberately avoids retrospective IMU-factor splitting. A factor whose referenced state or immutable revision has left the lag is rejected as `TOO_OLD`. The coordinator obtains exact IMU support that brackets every accepted interval; each raw sample-to-sample segment is clipped at knot boundaries and owned by exactly one adjacent preintegration factor.

### 8.3 Stationary initialization default

The production default is a stationary bootstrap because it is observable, testable, and inexpensive. Initial benchmark seeds are:

- at least 2 s of contiguous timestamp support with raw samples bracketing both interval boundaries;
- no unaccounted coverage hole, no sample interval greater than `1.5 / f_nominal`, no non-monotonic stamp, and no saturation; sample count alone is never a coverage test;
- mean angular rate below 0.05 rad/s and gyro standard deviation below 0.01 rad/s;
- acceleration norm within 0.5 m/s² of configured gravity and acceleration standard deviation below 0.2 m/s²;
- finite clock uncertainty below the initialization profile's declared bound.

Set gyro bias from mean angular rate. With gravity vector `g_odom`, choose roll/pitch so `R_odom_imu mean(f_measured) = -g_odom`; yaw defines the odom gauge. Initialize position and velocity to zero. A stationary pose does not fully observe accelerometer bias, so it receives a broad Allan-derived prior rather than a falsely precise estimate.

The initialization prior is a joint prior over the first pose, velocity, and biases. Position and yaw are the only pose gauge directions and receive the tight frame-defining prior. Roll/pitch covariance comes from the stationary sample statistics, velocity covariance from the zero-motion test, and accelerometer-bias covariance remains broad; a tight full-SE(3) prior is forbidden. Preserve estimated cross-correlation where the initializer produces it.

If stationary tests fail, remain `IMU_INITIALIZING` or require an explicit externally validated seed. A loose LiDAR–IMU moving initialization patterned after GLIM is a challenger with its own benchmark gate, not an automatic fallback.

### 8.4 State and preintegration

The IMU adapter exposes specific force, not gravity-removed translational acceleration. With `R_O_I` from `T_odom_imu`, the binding continuous measurement convention is

```text
f_m = R_I_O (a_O - g_O) + b_a + n_a
w_m = w_I + b_g + n_g,
```

where `R_I_O = R_O_I^T`, `w_I` is body angular velocity expressed in the IMU frame, and configured white-noise densities and bias random walks use SI units per square-root hertz. The ROS adapter must prove its message convention against this equation.

At knot `k`, the 15-DoF navigation state is

```text
x_k = { T_odom_imu(k), v_odom(k), b_g(k), b_a(k) }.
```

Gravity and physical extrinsic/timing calibration are fixed for a calibration/odom epoch. `ClockRevision` may advance only under section 4.3; every admitted sample and derived factor pins the revisions that produced its immutable fusion timestamps.

For interval `[t_k, t_j]`, linearly interpolate boundary samples at the exact knot times, then midpoint-integrate every clipped raw segment. Store the ordered sample IDs, clipped durations, linearization biases, `ΔR`, `Δv`, `Δp`, bias Jacobians, and full covariance. Propagate bounded timestamp-mapping uncertainty through duration/boundary Jacobians into that covariance; reject `TIME_UNCERTAIN` if the uncertainty could change sample ordering or interval support. No sample segment may appear in two adjacent factors. The canonical Meridian residual row order is fixed as `[R, v, p, b_g, b_a]`:

```text
r_R = Log( ΔR(b_gk)^T R_k^T R_j )
r_v = R_k^T (v_j - v_k - g Δt) - Δv(b_k)
r_p = R_k^T (p_j - p_k - v_k Δt - 0.5 g Δt²) - Δp(b_k)
r_bg = b_gj - b_gk
r_ba = b_aj - b_ak
r_imu = [ r_R, r_v, r_p, r_bg, r_ba ].
```

The covariance and square-root information use exactly that row order, including all retained cross terms. The private GTSAM adapter applies an explicit compile-time-tested permutation between Meridian order and the linked GTSAM build's navigation/bias order; passing a raw 15-by-15 matrix through is forbidden. The preferred implementation is one combined preintegration factor with the full propagated covariance. A split `ImuFactor` plus bias random-walk factor is acceptable only if Monte Carlo NEES, graph equivalence, and Jetson timing show no material loss. First-order bias correction is allowed; seed reintegration triggers are `||δb_g|| > 0.01 rad/s` or `||δb_a|| > 0.1 m/s²` and are tuned through `OPTIMIZE.md`.

An interval no longer than two nominal periods may use the last valid sample as a flagged zero-order hold. For that segment, integrate its actual duration and use the benchmark-seed process covariance `Q_gap = (Δt / Δt_nominal)^2 Q_nominal(Δt)`; record the inferred missing-tick count and inflated covariance. This is the only production gap bridge and its inflation law remains a benchmark decision.

An interval longer than two nominal periods is an epoch-breaking IMU fault. Close the current `OdomEpoch` at the last supported time, publish no valid propagation across the gap, create no bridge or constant-velocity pseudo-factor, and mark localization `UNOBSERVABLE`. After contiguous IMU support and initialization checks return, start a new `OdomEpoch` from stationary initialization or an explicitly validated external seed. Recovery must not reconnect the two local graphs implicitly.

### 8.5 High-rate propagation

The odometry publisher propagates from the last committed state using the same midpoint model and raw IMU ring. When a commit changes its anchor state or bias, propagation is replayed from that anchor. A propagated sample names the anchor graph revision, raw IMU interval, clock revisions, and propagation status. Propagation stops at the first epoch-breaking gap. Vendor AHRS orientation is not fused as truth in the initial design.

## 9. Knot scheduling and local transaction flow

### 9.1 Knot schedule

The schedule is the deterministic union of modality requests:

1. The camera tracker processes every admitted image. An accepted visual keyframe emits a `KnotRequest` at its bit-exact exposure midpoint. The frame remains frontend-only until the graph returns a `KnotResolution`.
2. Every complete LiDAR sweep gets a reference request. After the camera `ProcessedWatermark` has passed the original unfiltered sweep end, choose the accepted camera request inside the sweep support closest to the sweep midpoint, breaking equal-distance ties by lowest `KnotRequestId`; otherwise choose the original raw sweep end. If the chosen camera request is reused, the LiDAR requester cites the same exact timestamp and resolution.
3. Coalesce only requests whose integer-nanosecond `exact_time` values are identical; no measurement timestamp is snapped by an IMU-period tolerance. The scheduler also enforces a configured `minimum_state_interval`. Because visual minimum spacing, LiDAR's deliberate camera-time reuse, and guard cancellation normally prevent collisions, a nonidentical request inside that interval is rejected as `KNOT_TOO_CLOSE` rather than creating an ill-conditioned near-zero IMU interval or shifting the observation.
4. Insert an IMU guard request when an exteroceptive outage would make a preintegration interval exceed the configured cap. Because it carries no sensor measurement, cancel it instead of creating a state when another resolved knot already keeps both neighboring intervals within the cap.

Benchmark seeds are a five-second lag, maximum 64 navigation knots, 0.1 s minimum interval for optional visual knots, 10 Hz optional visual-knot cap, 0.5 s heartbeat, and 0.1 s maximum IMU guard interval. At the hard cap, suppress optional visual keyframes first, then marginalize the oldest complete state; publish `WINDOW_CAP_ACTIVE`. Never silently drop already accepted factors.

`KnotResolution` is published only after the schedule transaction that created the state and its adjacent exact-support IMU factor(s) commits. Frontends then submit factor specs against the resolution in a later transaction. Requests behind the finalized scheduling watermark are rejected as specified in section 8.2; the production scheduler never retrospectively inserts a state or splits a committed IMU factor.

### 9.2 One transaction writer

Only the local graph owner may create knots, initial values, factor IDs, and graph revisions. A transaction:

1. validates request/factor-spec IDs, odom/calibration/config/clock revisions, exact time support, correlation policy, budgets, and duplicate manifests;
2. exactly shares or creates requested knots from IMU prediction, and records a deterministic `KnotResolution`;
3. materializes only factors whose resolutions already exist, including any atomic deskew replacements;
4. updates and performs the configured relinearization checks on the active graph;
5. applies typed active-factor dispositions without mutating immutable frontend history;
6. runs the pre-marginalization finality barrier and marginalizes if required;
7. validates numerical state, cost, rank, delta, memory, revisions, and temporal usability;
8. atomically publishes the `LocalCommit` and any newly committed `KnotResolution`, or discards the candidate and reports failure.

On numerical failure or detected journal corruption, restore the last accepted checkpoint and deterministically replay the active-window factor-spec journal. A deadline miss alone does not trigger identical replay; section 12.4 defines its separate policy. Artificial strong priors that make a singular solve appear healthy are forbidden. A configured degraded factor backend is allowed only as a visible capability transition.

## 10. Visual frontend and visual factors

### 10.1 Classical production default

The first production lane is a grid-balanced KLT tracker with keyframe descriptors. It is computationally predictable, debuggable, and leaves GPU capacity for LiDAR/dense work.

Pipeline:

1. validate/rectify photometry and build a four-level grayscale pyramid;
2. retain long-lived tracks and seed Shi–Tomasi/FAST corners into sparse grid cells;
3. seed optical flow with IMU-predicted rotation;
4. run pyramidal KLT and forward/backward checking;
5. gate border, gradient, saturation, epipolar, and rotation-only consistency;
6. compute BRISK descriptors only on visual keyframes and recovery frames;
7. emit every frame for tracking diagnostics, but request a graph knot and factor construction only for an accepted visual keyframe.

Initial benchmark seeds: 8×6 grid, at most 400 tracks and 8 per cell; Shi–Tomasi quality 0.01, 15 px spacing, 16 px border; KLT window 21×21 over levels 0–3, 30 iterations, 0.01 termination, 0.75 px forward/backward error. These are not interface constants.

Visual-keyframe triggers after the minimum interval are any of:

- rotation-compensated median parallax at least 15 px;
- mature-landmark overlap below 0.55;
- spatial coverage below 0.5 or approximately fewer than 120 tracks;
- the 0.5 s heartbeat.

Pure rotation must not be classified as general tracking failure. Recovery uses mutual-best descriptor matching, ratio and spatial gates, geometric RANSAC, then new track IDs; descriptor similarity alone never reconnects an old `TrackId`. A non-keyframe observation has no local pose variable and is tracking-only in the production design. Using it in a graph would require a separately benchmarked exact-time discrete interpolation factor; silently attaching it to a neighboring keyframe is forbidden.

TensorRT SuperPoint plus LightGlue and compact ORB variants are benchmark challengers. They cannot become default until accuracy/failure recovery and GPU contention are measured concurrently with LiDAR and dense workloads on the target Jetson.

### 10.2 Landmark and reprojection factor

A landmark stores immutable anchor metadata `(anchor knot, camera, unit bearing)` and one unconstrained log-inverse-range variable `η`. For every supported central camera, `u_a = unproject(z_a) / ||unproject(z_a)||`, `ρ = exp(η) > 0`, and depth is Euclidean range along that unit ray. The log parameterization prevents an optimizer update from crossing zero; an out-of-range `η`, projection failure, or cheirality violation rejects that observation for the transaction rather than clamping it.

```text
p_Ca = u_a exp(-η)
p_O  = T_O_Ba T_B_Ca p_Ca
p_Cj = T_Cj_B T_Bj_O p_O
r_lj = L_j [ z_lj - π_j(p_Cj) ].
```

`L_j` whitens using pixel covariance derived from pyramid level and tracking quality plus the camera clock uncertainty projected by IMU-predicted image velocity. Reject a frame when that time uncertainty exceeds the profile bound instead of pretending the exposure midpoint is exact. The seed robust loss is Huber at approximately 2.5 px, applied once to each observation residual. Analytic Jacobians for both poses and `η`, including `∂p_Ca/∂η = -p_Ca`, are mandatory and must match finite differences for every supported camera model.

Default triangulation requires at least two non-anchor observations, positive depth within the configured operating range, at least 1.5° parallax, bounded depth condition/variance, and at most 2 px reprojection RMSE. LiDAR-derived depth, if later evaluated, is a separate provenance-bearing factor; it is never an invisible initializer that creates cross-modal correlation.

Landmarks are Schur-eliminated before navigation states. The safe initial marginalization policy terminates a landmark segment when its anchor is about to leave the lag. A later segment receives a new `LandmarkSegmentId` and starts only from observations not already admitted as residuals or anchor metadata in the old segment; no `TrackObservationId` may influence both segments. It remains pending until enough new observations satisfy triangulation. Exact landmark re-anchoring and reuse of an old observer as a new anchor are challengers only after duplicate-evidence and prior-equivalence tests pass.

An observation over the configured 2-DoF chi-square gate for two accepted commits is retired; one residual does not delete the whole track or camera modality. Global-shutter exposure is the initial supported model. Rolling-shutter frames are rejected or explicitly degraded until a row-time factor is implemented.

### 10.3 Visual failure isolation

Visual health is driven by its own track count, spatial coverage, geometric inliers, residual/NIS distribution, blur/saturation, and time validity. It transitions through `BOOTSTRAP -> PROBATION -> HEALTHY -> DEGRADED -> QUARANTINED -> RECOVERING`. While quarantined, tracking/recovery continues but submits no graph factors. A recovery seed is three consecutive well-distributed frames with sane normalized innovations; probation inflates measurement covariance and is observable.

## 11. LiDAR frontend, direct factors, and degeneracy

### 11.1 Discrete IMU deskew

Never recompute sweep support from filtered points. Preserve original first/last acquisition times and signed per-point offsets.

Build a discrete pose table at Ouster column timestamps from committed-anchor IMU propagation. The table carries a pose covariance and the IMU sample/clock revisions used at every interval. For a point acquired at `t_p` and expressed at reference time `t_r`:

```text
p_Br = T_O_B(t_r)^-1 T_O_B(t_p) T_B_L p_L.
```

The reference may lie inside the sweep, so propagation/interpolation must support points on both sides. Output is immutable and cites sweep ID, raw support, bias/prediction/calibration/clock revisions, and pose-table revision. Each source-point covariance includes local surface/sensor uncertainty plus the first-order propagated deskew-pose, timestamp-mapping, and extrinsic uncertainty. The relevant Jacobians and covariance frames are part of the deskew unit-test contract.

The same IMU samples may condition deskew and an adjacent preintegration factor, so they are correlated. A `MeasurementManifest` distinguishes each LiDAR source row as a primary measurement from IMU samples, target geometry, clock mappings, and calibration as conditioning measurements. Conditioning reuse is allowed only under the factor's explicit `LidarCorrelationPolicy`, which supplies a calibrated covariance inflation and per-source-sweep information cap; a missing policy or an implicit unity inflation is an admission error.

Re-deskew when the maximum predicted point displacement between old and new pose tables exceeds `min(0.05 m, 0.25 × base_voxel)`. While the state is live, crossing that threshold creates a `DeskewReplacement` transaction: build a new immutable deskew/source/target revision, atomically remove every active factor or pyramid derived from the old revision, insert the replacements, and record both dispositions in the journals. Permit at most one replacement for a sweep per accepted graph revision and a benchmark-seed maximum of two solve/re-deskew cycles before suppressing that sweep's LiDAR factors and reporting `DESKEW_NOT_CONVERGED`. The pre-marginalization barrier rejects a state with a pending replacement. A final deskew for a sealed mapping artifact creates new geometry provenance; it never retroactively mutates a local graph factor already condensed into a prior.

Validity, range, ring, self-hit, vehicle-mask, and deterministic voxel filters are separate stages with before/after counts. Initial covariance/normals use organized neighbors when valid or bounded local PCA. GLIM's 0.15 m downsample, eight-neighbor covariance, and plane regularization are benchmark seeds alongside 0.25 m and 0.5 m profiles.

### 11.2 Multiresolution VGICP target

The accuracy target is direct multiresolution voxelized GICP. An immutable source contains means/covariances/normals in its body-reference frame. An immutable target pyramid stores Gaussian voxels plus its pose/content revision. For source state `j` and target state `i`:

```text
T_ij = T_O_Bi^-1 T_O_Bj
e_s  = T_ij μ_s - μ_v
S_bar_s = C_v + R_bar_ij C_s R_bar_ij^T + C_model
cost_local(T_ij ; T_bar_ij) = Σ_s rho( e_s(T_ij)^T S_bar_s^-1 e_s(T_ij) ).
```

A custom nonlinear factor owns association/reassociation and evaluates state-dependent residuals; the frontend does not compress matching into a pose plus inverse Hessian. At each factor linearization point `T_bar_ij`, select correspondences, compute `S_bar_s`, and freeze both while forming that Gauss–Newton linear factor. Differentiate the transformed mean residual, not `S_bar_s`; this is the frozen-covariance approximation used by the referenced integrated VGICP implementation and is intentionally distinct from the full derivative of a pose-dependent covariance objective. Refresh correspondences and covariance whenever the factor is relinearized; threshold-cached association is allowed only as a separately tested performance profile. Surface/visibility validation precedes whitening.

The binding order is: deterministic association and source-row ownership, geometric/chi-square pre-gate, whitening by frozen `S_bar_s`, one per-correspondence robust IRLS weight, then section 11.4's LiDAR-only rank projection and information cap. Do not wrap the resulting factor in a second aggregate robust noise model. Huber is the initial benchmark seed; Cauchy/Tukey are challengers.

GLIM-derived benchmark seeds are adaptive 0.25–0.5 m base voxels from median range, two levels at ×2 scale, connections to the previous two frames, and at most 15 overlap-selected live targets. Benchmark three previous frames for rough terrain. Add/remove overlap targets with hysteresis. A target may not contain geometry derived from the current source sweep.

Multiple targets and pyramid levels must not multiply one sweep's evidence. At a committed linearization, each primary source Gaussian is assigned deterministically to at most one target and one resolution; coarse levels may seed capture/recovery but do not add the same row again to the final graph. Reuse of a target scan by later source sweeps is conditioning reuse and invokes `LidarCorrelationPolicy`. After rank projection, cap both each supported eigenvalue and the total supported information contributed by one source sweep according to `CorrespondenceBudget`; increasing target count may improve geometry/capture range but may not increase the configured information budget. The budget also hard-caps source rows, accepted correspondences, target count, blob bytes, CPU/GPU workspace, and linearization time. Optional targets are removed deterministically before an accepted source row or factor is silently dropped.

The production local graph uses binary factors only while both source and target states are live. A target outside the lag may provide a recovery proposal or place seed, but it does not create a unary local factor: immutability does not remove its pose uncertainty or correlation with the marginal prior. A Schmidt/consider-state or calibrated condensed-target factor is a future challenger. The graph input is therefore a factor spec with source/target blob handles, exact live knot resolutions, correlation policy, and budget, never merely a relative-pose observation.

### 11.3 Always-available point-to-plane profile

The degraded/reference lane is a CPU bounded voxel-surfel point-to-plane implementation:

```text
r = n_v^T (T_ij p_s - μ_v)
σ_bar² = n_v^T (R_bar_ij Σ_s R_bar_ij^T + Σ_v) n_v + σ_floor².
```

It uses immutable multiresolution snapshots, deterministic reservoir/provenance rules, and robust whitening. Correspondences, normals, and `σ_bar²` are frozen at the current linearization point exactly as in the VGICP lane; the linear factor differentiates only the point-to-plane mean residual. It obeys the same primary-row ownership, conditioning-correlation, information-cap, binary-live-target, and redeskew-replacement contracts. Concepts in the legacy voxel-hash implementation are extraction material; its mutable live/global map is not. Point-to-point registration remains a benchmark baseline only.

VGICP is the intended production default only if its concurrent Jetson deadline, memory, thermal, and accuracy gates pass. Otherwise point-to-plane is an explicit deployment profile, recorded in every commit—not an invisible algorithm swap.

### 11.4 Directional degeneracy

Compute localizability separately for each binary source-target registration factor after association, pre-gating, whitening, and its single robust-weight update. Let its relative-pose linearization be `r + J_L δξ_ij`; `J_L` contains only LiDAR measurement rows and excludes IMU, vision, marginal priors, and the other targets' state blocks. A sweep-level report may aggregate pair reports for health, but it is not a six-column factor Jacobian.

Normalize translation and rotation with a strictly positive characteristic length `ℓ` derived from source radius/median range and clamped to the algorithm profile's declared `[ℓ_min, ℓ_max]`. With `D = diag(ℓ I_translation, I_rotation)`:

```text
J_n = J_L D
g_n = J_n^T r
H_n = J_n^T J_n = V Λ V^T.
```

Run an X-ICP-style correspondence-contribution classifier alongside this spectrum. Initial hysteresis seeds mark a normalized eigen-subspace unsupported below `λ/λ_max = 10^-3` for two sweeps and restore it only above `3×10^-3` for three sweeps; `λ_max = 0` means rank zero. These values require benchmark calibration. Before temporal matching, transport normalized eigenvectors into a common odom tangent frame and re-orthonormalize. Cluster near-repeated eigenvalues with a configured tolerance and associate clusters by principal angles/projector overlap; never carry hysteresis by sorted eigenvalue index or eigenvector sign. An X-ICP nonlocalizable classification may remove a mode but may not fabricate one.

For supported eigenvectors `V_s` and positive eigenvalues `Λ_s`, construct the relative low-rank linear residual exactly as

```text
A_rel = Λ_s^(1/2) V_s^T D^-1
c_rel = Λ_s^(-1/2) V_s^T g_n
r_rank(δξ_ij) = c_rel + A_rel δξ_ij.
```

If an information cap scales a mode, scale the corresponding rows of both `A_rel` and `c_rel` by the same square root. Under the section 4.2 right perturbation and `T_ij = T_i^-1 T_j`, map the relative increment to the live pose blocks with

```text
δξ_ij = -Ad_(T_ij^-1) δξ_i + δξ_j.
```

The custom factor returns this rank-row `JacobianFactor` (or algebraically identical Hessian blocks); standard full-rank noise models are not used. Unsupported modes therefore contribute exactly zero LiDAR Hessian and gradient, while IMU and vision remain free to constrain them. The graph-computed report publishes pair/factor IDs, source-body and common-frame bases, `ℓ`, eigenvalues, supported rank, cap scale, classifier, and hysteresis state. The frontend hint is never authoritative.

Robust kernels reject outliers; they do not repair an unobservable corridor/floor/open-field geometry. Tests must verify the low-rank factor's Hessian and gradient against an SVD projection, both-pose numeric Jacobians, zero unsupported rows, eigen-cluster swaps, coordinate-frame changes, and meter/centimeter scaling.

### 11.5 LiDAR recovery

Rank loss alone is normal and does not quarantine LiDAR. Quarantine requires modality-specific evidence such as invalid deskew/map revisions, time failure, association collapse, implausible normalized cost, or sustained overlap failure. Coarse-to-fine recovery continues against immutable recent targets. Recovery and probation thresholds are benchmarked and published with every transition.

## 12. Local fixed-lag graph in detail

The local objective is

```text
min  ||A_prior boxminus(x0, x) - b_prior||²
   + Σ_imu ||r_imu,bias||²
   + Σ_visual_observation rho_v(r_reprojection^T r_reprojection)
   + Σ_owned_lidar_row rho_l(r_registration^T r_registration),
```

over live navigation states and visual log-inverse-range variables. Visual robustification is once per observation. LiDAR robustification is once per source row before the low-rank projection; there is no aggregate outer robust kernel. There are no local GNSS, loop, dense-map, global-anchor, or corrected-pose factors.

### 12.1 Factor inventory

| Factor | Variables | Construction | Failure policy |
|---|---|---|---|
| Gauge/initial prior | first `T,V,B` | joint initialization covariance; only yaw/position are tight gauge definitions | new odom epoch on reset; never tightly freeze roll/pitch or silently retighten |
| Frozen marginal prior | remaining boundary variables | square-root QR marginalization at stored linearization point | validate ordering/rank; rebuild from journal on corruption |
| IMU preintegration + bias | `T,V,B` at adjacent knots | exact clipped IMU support, propagated covariance | no factor across unsupported gap |
| Reprojection | anchor pose, observer pose, log inverse range | immutable pixels/camera/calibration; one robust pixel loss | retire active individual observations; quarantine lane on sustained failure |
| Direct VGICP | live source and live target poses | immutable Gaussian pyramids, frozen-covariance state-dependent associations | rank-project LiDAR-only information; replace redeskew revisions atomically |
| Point-to-plane | live source and live target poses | immutable surfel snapshots, same ownership/correlation rules | explicit degraded profile |

### 12.2 Solver and marginalization

The private graph adapter initially uses an iSAM2-based fixed-lag wrapper with `relinearizeSkip = 1`: every accepted commit checks configured variable-specific relinearization thresholds, but does not force every variable and direct factor to relinearize. Forced full-window relinearization is a benchmark/rebuild operation, not the real-time default. A five-second lag is the seed, not a guarantee that time alone controls size; knot, factor, landmark, byte, and solve-time caps also apply.

Marginalize landmarks first, then the oldest navigation state blocks, retaining newest states last in ordering. Form the prior in square-root form using rank-revealing QR; do not explicitly invert normal equations. In the stored block order define

```text
boxminus(x0, x) = [ Log(T0^-1 T), v-v0, b_g-b_g0, b_a-b_a0, eta-eta0, ... ],
r_m(x) = A boxminus(x0, x) - b,
```

where every pose `Log` uses the section 4.2 right, translation-first tangent and only variables retained by that prior appear. Store `A`, `b`, `x0`, ordering, numerical rank/tolerance, and factor/measurement provenance. `A`, `b`, and `x0` are frozen. Evaluating the chart derivative needed for `boxminus(x0, x)` at a later estimate is allowed; recentering the prior or treating it as the original nonlinear measurements is not.

Use Cholesky only when rank/conditioning tests permit; QR and deterministic active-window rebuild are recovery paths. Benchmark Gauss–Newton versus Dogleg and periodic bounded batch LM. GTSAM remains behind the private adapter so solver replacement does not change contracts.

### 12.3 Factor finality and journals

Before any state or landmark enters a marginal prior, the graph owner runs a finality barrier in the same transaction:

1. complete or reject every pending `DeskewReplacement` and verify all factor/blob/clock/calibration revisions;
2. evaluate current visual outlier streaks and LiDAR association/robust weights, remove typed active outliers, and resolve duplicate/correlation dispositions;
3. close landmark segments whose anchors leave the lag and verify that no observation ID crosses segments;
4. freeze the accepted factors' current linearization, robust weights, associations, rank projection, and information-cap scale for QR marginalization;
5. reserve and persist the journal records needed to replay and later condense that exact raw-factor partition.

Once a factor has entered a marginal prior, later quarantine or residual changes cannot selectively remove its information. Quarantine stops or removes active factors only. Proven corruption of already marginalized evidence requires deterministic reconstruction from retained raw factor specs; if those records are unavailable or reconstruction fails validation, start a new `OdomEpoch` rather than subtracting a guessed information matrix.

Two distinct bounded journals are mandatory:

- `ActiveWindowJournal` retains every live knot request/resolution, raw factor spec, blob/revision handle, replacement lineage, disposition, and checkpoint needed to reproduce the current lag. It is released as variables leave the lag except for records handed to staging.
- `SubmapStagingJournal` retains the disjoint raw IMU, visual, and LiDAR factor partition plus immutable blobs/configuration needed by the later submap seal. It survives fixed-lag marginalization and is released only after the corresponding seal is durable.

Both journals have independent hard count, byte, and age budgets. A transaction reserves space in every required journal before accepting a factor. Optional factors are rejected before acceptance when reservation fails; accepted records are never dropped. Approaching a staging cap requests an early submap split and suppresses optional frontend work. Failure to reserve mandatory IMU/state history is a declared storage fault that stops new commits rather than creating unjournaled state.

These two local-estimator journals are process-local bounded recovery structures in the first implementation; “persist” here means retain across graph transactions, not promise recovery across a process crash. A local-process crash discards the unsealed active tail, starts a new `OdomEpoch`, and replays only already durable sparse seals from `SealSpool`. Recovering an in-progress local graph across process restarts is a later checkpoint capability and must not be inferred from the global durability protocol in section 6.6.

### 12.4 Commit gates

Numerical commit validity requires:

- all states, residuals, and covariances finite;
- no unintended rank loss or missing gauge;
- bounded pose, velocity, and bias delta relative to prediction;
- innovation/residual distributions compatible with declared modality state;
- correct factor counts, measurement uniqueness, and live revisions;
- queue, memory, knot/factor, blob, and journal reservations respected;
- propagated output replayable from the new anchor.

Deadline service health is evaluated separately from numerical validity. If a numerically valid candidate misses its deadline but its state age is below the configured `max_publish_age`, commit it atomically with `DEADLINE_MISSED`, degrade the declared real-time capability, and advance the governor before the next transaction. If it is older than `max_publish_age`, do not publish it and do not replay the identical transaction/profile: first select a lower bounded workload/profile or shed work that was never accepted, record all dispositions, then retry under a new `ConfigRevision`. Repeated misses beyond the configured horizon transition to `UNOBSERVABLE`. Checkpoint replay without a profile/input change is reserved for numerical failure or corruption, not timing.

Cross-modal disagreement alone does not choose a winner. It marks combined health degraded and evaluates each modality's own evidence. With both exteroceptive lanes absent, publish bounded-duration IMU dead reckoning with growing covariance, then `UNOBSERVABLE`.

## 13. Sparse submaps and the dense-mapping seam

### 13.1 Why submaps are the boundary

The local graph owns live states; the global graph owns only finalized submap anchors; dense fusion owns voxels. This avoids three failure modes in v1-style architectures:

- a global correction changing the coordinate frame of the live registration map;
- loop closure requiring destructive TSDF clear/reintegration;
- global optimization depending on mission-length raw keyframe state.

Sparse submaps exist even when dense mapping is disabled. They are bounded registration/place-recognition artifacts, not TSDFs.

### 13.2 Sparse submap lifecycle

Each submap has a non-overlapping core time interval `[start, end)` and optional read-only context from neighbors for retrieval. A factor/measurement belongs to exactly one core transition by its terminal timestamp. This prevents adjacent condensed edges from counting the same local information twice.

Request a split on a benchmarked combination of travel, rotation, duration, visual/LiDAR overlap, keyframe count, payload bytes, calibration epoch, and local reset. Seeds from OKVIS2-X are inputs to the submap-policy benchmark, not copied constants. Online lifecycle is:

```text
ACTIVE ──split request at provisional boundary b──► FINALIZING
FINALIZING ──b exits lag successfully──► SEALED + next ACTIVE
FINALIZING ──b/reset transaction aborts──► discard provisional successor and reassign buffered records
```

A split request does **not** claim that `b` is final. The coordinator closes the old core interval provisionally, opens a bounded successor builder, and buffers immutable source references while `b` remains in the lag. It may retain read-only context on both sides, but each measurement/factor ID has one provisional core owner. When `b` leaves the lag, the coordinator freezes its final pose, deterministically replays the bounded successor buffer into the final new frame, and atomically finalizes the two half-open intervals. Thus no sensor gap is introduced while waiting one lag duration, and no provisional pose becomes global evidence.

Sealing occurs only after every contributing local state has left the active lag and its final accepted pose is recorded. At seal time:

1. final-deskew retained LiDAR keyframes with their final local state/bias revision;
2. reconstruct the stored final linearized submap batch from `SubmapStagingJournal`, express trajectory, mature visual landmarks, and registration geometry in immutable `submap_<id>` coordinates, and compute each consecutive endpoint pose/velocity joint covariance before releasing raw factors; this covariance/condensation pass does not re-optimize or move states already declared final;
3. build a bounded voxel/surfel/Gaussian registration proxy once;
4. reconstruct the disjoint raw-factor journal partition between consecutive submap boundaries, exclude the incoming marginal prior, condition the earlier boundary as the relative gauge, and Schur/QR-condense internal states and landmarks into a relative constraint;
5. compute manifests, quality, checksums, and durable blob records;
6. publish the idempotent seal through the loss-intolerant outbox.

An IMU interval, visual observation factor, or LiDAR factor is assigned by its terminal state/time. Cross-boundary landmarks may be retained as read-only place geometry on both sides, but each pixel factor and raw measurement ID occurs in exactly one condensed partition. The conditioner records the eliminated factor IDs and both boundary state IDs, and a test reconstructs adjacent partitions jointly against the corresponding raw local factor journal. The incoming marginal prior is never re-exported.

The submap origin is the finalized boundary-body origin. Its `+z` is gravity-up and its yaw is the finalized boundary-body yaw; roll and pitch are zero by definition. `T_odom_submap` and this frame definition are immutable after sealing. For the first submap of a new standalone mission, the map gauge is initialized at `T_map_submap0 = T_odom_submap0`, so `T_map_odom` begins as identity. An identity prior on `T_map_submap0` is valid only if `T_odom_submap0` is itself identity.

### 13.3 Dense input contract

The future dense package consumes only finalized, revisioned local records:

```cpp
struct DenseColorFrame {
  CameraFrameId frame;
  CameraId camera;
  FusionTime exposure_midpoint;
  Pose T_submap_camera;
  CameraModelRevision camera_model;
  BlobRef image;
  optional<BlobRef> visibility_or_dynamic_mask;
};

enum class DenseLayerKind { Tsdf, Color, Occupancy, Esdf, Mesh };

struct DenseLayerRef {
  DenseLayerKind kind;
  DenseLayerEncodingRevision encoding;
  double resolution_m;
  Aabb bounds_submap;
  BlobRef payload;
};

struct DenseLayerManifest {
  vector<DenseLayerRef> layers;
};

struct DenseFrameCommit {
  RecordHeader header;
  DenseFrameId id;
  SubmapRef sparse_submap;
  DenseInputPolicyRevision input_policy_revision;
  LocalGraphRevision final_local_revision;
  FusionTime reference_time;
  Pose T_submap_lidar;
  BlobRef final_deskewed_cloud;
  optional<DenseColorFrame> color;
  MeasurementManifest measurements;
  ContentHash checksum;
};

struct DenseInputManifestEntry {
  DenseFrameId id;
  ContentHash record_checksum;
  BlobRef serialized_record;
};

enum class DenseSealState { Complete, Incomplete };

struct DenseSubmapSeal {
  RecordHeader header;
  SubmapRef sparse_submap;
  ContentHash sparse_seal_checksum;
  DenseContentRevision dense_content_revision;
  DenseConfigRevision dense_config_revision;
  DenseSealState state;
  BlobRef integrated_frame_manifest;
  optional<BlobRef> missing_frame_manifest;
  DenseLayerManifest layers;
  ContentHash checksum;
};
```

`DenseFrameCommit` is emitted only after every state used by the frame pose, color association, and deskew has left the fixed lag; its pose is final in the local submap and its `SubmapRef` pins the exact session, odom epoch, calibration, sparse content revision, and local-content checksum. Its optional color record names the selected final camera pose, calibrated camera-model revision, image checksum, and optional immutable mask; the association policy and maximum time offset are identified by `DenseInputPolicyRevision`. Re-delivery of `(SubmapRef, DenseFrameId)` with the same record checksum is idempotent; a different checksum is an integrity fault. Canonical dense fusion therefore runs with the lag delay. The sparse seal's `dense_input_manifest` has layout `DenseInputManifestEntry[]`: the complete ordered ID set, each record checksum, and a durable reference to the serialized record and its child blobs. Each layer manifest entry names its encoding revision, metric resolution, submap-frame bounds, payload layout, and payload checksum. Transport order is irrelevant: dense waits until it has the matching `SparseSubmapSeal` and every listed frame, verifies checksums/revisions, integrates each frame exactly once, and emits `DenseSubmapSeal` that pins both the exact `SubmapRef` and sparse seal checksum. A timeout or permanently unavailable frame produces an immutable `Incomplete` dense revision with the exact missing manifest, never a silently complete map; a later complete result is a new dense content revision. `Incomplete` layers are diagnostic only and may not be advertised as planner-ready.

A separate resettable `DensePreviewFrame` may later provide low-latency visualization, but it is never canonical or consumed by planning. A loop/GNSS commit changes only a `SubmapAnchorRevision` whose `SubmapRef` exactly matches the dense seal's `sparse_submap`; the dense seal also verifies the enclosing sparse seal checksum. It never asks nvblox to subtract keyframes or rebuild globally corrected voxels. Dense content is immutable by `(SubmapRef, dense_content_revision)`; re-delivery with the same checksum is idempotent and a conflicting checksum is an integrity fault. Consumers render/query a collection of dense seals through their current anchors. The future dense process consumes `SparseSubmapSeal`, `DenseFrameCommit`, and `SubmapAnchorRevision`, and acknowledges the corresponding spool objects before they are eligible for GC. Layer blobs and frame/image blobs remain pinned for the lifetime of a published dense seal or a downstream durable copy; anchor revisions never extend an expiring lease.

For navigation mapping later, nvblox must add the layers actually required by planning—typically occupancy/ESDF and explicit free/unknown semantics. The retained v1 work currently demonstrates TSDF/color/mesh integration only; it is not yet a planner-ready global map.

### 13.4 Disposition of the current nvblox code

The old `meridian_map` package is not kept active or quarantined because its public API combines registration, dense fusion, mission-length keyframe storage, mutable global poses, and graph-driven clear/rebuild. Keeping it would preserve the ownership model being removed and all of its v1 dependencies.

The code remains exactly recoverable from the legacy baseline. When `meridian_dense` begins, extract and adapt only:

- `src/meridian_map/src/nvblox/nvblox_integrate.{hpp,cu}`;
- integration, color, mesh-cache, meshing, and confidence sampling from `nvblox_surface_map.{hpp,cpp}`;
- analytic sphere/color/occlusion/long-range/determinism tests from `test_nvblox_surface_map.cpp`.

Do not port `IMapLayer`, `ISurfaceMap`, `IRegistrationMap`, `KeyframeStore`, `LayeredMap`, `GraphUpdate`, per-keyframe block ownership, `clear_keyframes`, or the CPU TSDF as a claimed numerical oracle. The new private nvblox target takes `DenseFrameCommit`-equivalent views and exports no nvblox/CUDA type.

Before extraction is accepted, import a clean nvblox checkout and pass a clean Jetson GPU build. The ignored working checkout is not build evidence.

## 14. Global backend

### 14.1 Variables and factors

For sealed submap `s`, the global variable is

`A_s = T_map_submap_s ∈ SE(3)`.

After GNSS alignment, add a true four-dimensional variable

`G = T_enu_map = {yaw, translation_enu}`.

`YawTranslation4` uses the translation-first tangent `[δt_E, δt_N, δt_U, δyaw]`, retraction

`G ⊕ δ = {wrap(yaw + δyaw), translation_enu + δt}`,

and the action `p_enu = Rz(yaw) p_map + translation_enu`. Do not represent fixed roll/pitch with artificially tiny six-dimensional priors. Analytic action/factor Jacobians must match central differences across yaw wrapping. Multi-session and atlas-alignment variables are not present in this revision.

The graph contains:

| Factor | Variables | Residual/source |
|---|---|---|
| Mission gauge | first anchor | fixes the coordinate gauge of one standalone mission; not sensor evidence |
| Adjacent sparse submap | `A_i, A_j` | condensed local relative constraint, exact manifest |
| Visual loop | `A_i, A_j` | robust/rank-aware verified relative pose |
| LiDAR loop | `A_i, A_j` | robust/rank-aware verified relative pose |
| GNSS antenna position | `A_s, G` | exact-time antenna ENU residual |
| Optional surveyed alignment prior | `G` | independent configured survey manifest; never derived from admitted fixes |

For a relative measurement `Z_ij = T_Si_Sj`, the full-rank residual is

`r_ij = Log( Z_ij^-1 A_i^-1 A_j )`,

then projected/whitened by its rank-aware square root. A solver robust kernel is defense in depth; it does not bypass proposal admission.

For the first seal, fix `A_0 = T_odom_submap0` (or apply the numerically equivalent documented gauge constraint). This makes the initial `T_map_odom` identity without asserting that `T_map_submap0` itself is identity. The current graph contains one standalone mission/session only.

The committed active graph is hard-bounded by configured maximum anchors, active factors by class, retained proposal states, serialized graph bytes, journal/checkpoint bytes, shadow bytes, and solve wall time. These are admission limits checked before allocation and again before commit. A trusted adjacent-submap insertion may use an incremental transaction over the bounded active graph. Any loop batch, GNSS initialization/reacquisition batch, robust-factor state change, or large correction uses a complete shadow copy of that bounded graph and validates all active factors.

This revision has no lossy global marginalization, silent node eviction, or archive/reactivation mechanism. If a new seal or factor would exceed a hard graph/cache/journal limit, global enters `GLOBAL_GRAPH_CAPACITY`, leaves the last durable graph and `map -> odom` readable, stops accepting and acknowledging new map-growth seals, and continues no transaction that would require truncated evidence. Local odometry continues and its spool applies the explicit storage-capacity policy in section 6.6. Resuming map growth requires operator-provided capacity or the future map-store/atlas design; dropping old anchors to make room is forbidden.

### 14.2 Visual place recognition and verification

The index interface is independent of network choice. The target lane uses a compact EigenPlaces/MixVPR-class global descriptor at selected keyframes or 1–2 Hz, grouped and voted at submap level. A conservative first slice may use the OKVIS2-X DBoW/BRISK path. In either case:

- same, overlapping, and immediately adjacent submaps are excluded by explicit policy;
- frame candidates require temporal/covisibility voting;
- retrieval top-K, TTL, model revision, and scores are recorded;
- retrieval score schedules work but never becomes graph information.

The visual verifier independently matches query features to mature candidate 3D landmarks, then:

1. mutual/ratio/dynamic-mask filters;
2. generalized 3D–2D PnP RANSAC without using odometry as the only seed;
3. gates inlier count/ratio, grid coverage, bearing diversity, and depth/parallax;
4. robust relative-pose refinement against fixed candidate landmarks;
5. outlier reclassification and final Hessian spectrum;
6. calibrated rank-aware covariance/information.

If only monocular 2D–2D geometry is available, reject it in the first implementation or emit a correctly partial factor. Never fabricate metric translation scale or full-rank covariance.

### 14.3 LiDAR place recognition and verification

MapClosures is the target primary retriever: ground normalization, density-preserving BEV, ORB/self-similarity pruning, HBST matching, 2D RANSAC, and full seed recovery through ground transforms. Its database mutation, sequential integer IDs, and temporal exclusion policy must be removed from the algorithm seam and owned by Meridian. Scan Context is an auxiliary vote and regression baseline, not the sole production detector.

The verifier uses the independent place seed to run submap-to-submap VGICP/GICP, optionally with a small bounded seed set. It requires bidirectional overlap, distributed spatial support, residual median/upper quantiles, covariance/normal consistency, bounded dynamic fraction, and a sane information spectrum. Recompute final correspondences before producing a `LoopMeasurement`.

No hard gate on current corrected XY distance is allowed: that rejects the large drift corrections loops are intended to find. Odometry consistency is a scored input to consensus, not an absolute single-loop veto.

### 14.4 Proposal admission: PCM then GNC

Pairwise consistency is rank-aware; the legacy six-dimensional `Σ_cycle^-1` test is not valid for partial constraints. For each two-loop cycle:

1. express the cycle error `ε = Log(T_cycle)` in one declared right-perturbation tangent;
2. transport every constituent loop and adjacent-chain supported basis/covariance into that tangent with the appropriate adjoint/Jacobian;
3. compute by SVD the common supported subspace of those transported bases using a calibrated absolute/relative tolerance;
4. project and sum the constituent covariances in that subspace, eigendecompose the result, and retain only finite, positive, calibrated supported modes;
5. evaluate `d² = εᵀ U_r Σ_r^-1 U_rᵀ ε` against `χ²(r, α)`, where `r` is the retained rank.

If `r` is below the configured nonzero minimum, the pairwise relation is `UndefinedInsufficientCommonRank`, not a consistency edge. PSD repair may clamp only bounded numerical roundoff and is reported; it may not invent support. Every basis transport, support intersection, rank, eigenvalue, threshold, and `d²` is retained in `LoopReport`.

PCM admission requires a clique containing at least two distinct proposal IDs and a defined pairwise edge; a singleton is never admitted, regardless of retrieval score or odometry agreement. Proposals claimed as independent must have disjoint raw measurement/factor manifests. Visual and LiDAR proposals at the same endpoints may both contribute only after that proof and the pair-level information cap/correlation inflation in section 6.5. A correction above the configured large-correction threshold additionally requires support from at least two distinct endpoint pairs, normally repeated query submaps.

The pending consistency graph is partitioned deterministically into connected components ordered by `(oldest_proposal_time, smallest_proposal_id)`. A component within the configured PCM vertex/edge/solve limits is solved completely with deterministic maximum clique. An oversized component remains explicitly `DeferredResourceLimit`; it is never prefix-truncated and its unexamined vertices are never reported as rejected or admitted. TTL expiry is a separate recorded rejection.

One serialized global owner performs each robust atomic transaction:

1. validate endpoint/content/calibration/session revisions, frames, rank, PSD, TTL, duplicates, and manifests;
2. construct the rank-aware consistency graph and resolve complete bounded components;
3. enforce the non-singleton, independence, correlation, and large-correction support rules;
4. select one deterministic admitted batch within the configured robust-factor and shadow budgets;
5. copy the complete bounded active graph at durable parent revision `R` into a shadow transaction;
6. keep adjacent factors and the mission gauge as known inliers, and add selected loops or qualified GNSS proposals as robust candidates;
7. run GNC-TLS with factor-dimension-correct cutoffs, normalizing each whitened squared cost by its `χ²(dof, α)` gate or an explicitly calibrated per-factor threshold;
8. validate the complete candidate solution and all resource/deadline gates;
9. append and durably flush the `PREPARED` inputs, dispositions, solve report, and solution hash;
10. under the single writer, recheck parent `R`, append and durably flush `COMMITTED(R+1)`, then atomically install the snapshot;
11. publish anchors and `map -> odom` only after the durable marker succeeds.

A changed parent discards or deterministically rebuilds the shadow; no partial result is rebased in place. Shadow budget/deadline exhaustion defers the batch without graph mutation.

Shadow validation requires finite convergence, connectedness, sane gauge/rank/marginals, no unexplained collapse of robust weights, bounded degradation of trusted-factor residuals, improved or justified cost per effective DoF, a finite current anchor, and resource/deadline compliance. A large correction alone is not a rejection; it invokes the stronger multi-endpoint consensus rule and a forensic event.

Rollback or factor disabling is a new revision. Committed history is never edited in place. The legacy PCM mathematics, deterministic maximum clique, chain covariance, tangent/PSD guards, and last-good recovery are extraction candidates. The old keyframe graph and removal-only GNC subproblem are not.

### 14.5 GNSS datum, alignment, and factor

Select a configured/persisted ENU origin when available. Otherwise use a sustained high-quality moving window, preferably fixed RTK; never let the first plausible fix alone define the datum. The origin is immutable once committed.

Estimate `T_enu_map` with a covariance-weighted robust yaw-plus-translation fit over observations associated with final sealed trajectory brackets. Require a minimum horizontal baseline, multiple moving samples, sufficient spatial excitation, bounded yaw uncertainty, and qualified solution states. Poor excitation remains `ALIGNING`.

The robust fit is an initializer and observability/covariance report only. It emits no graph factor or prior. In the same shadow transaction that first creates `G`, replay each qualified source-window observation exactly once as a `GnssFactorProposal`, keyed by `GnssObservationId`; the graph objective therefore contains each fix once. `GnssAlignmentRevision.covariance` is not a solver prior. A configured surveyed `G` prior is allowed only when it is genuinely independent and carries a source manifest disjoint from every GNSS fix factor. Reacquisition follows the same rule: a quarantine-window fit initializes the shadow, while only the individually manifested fixes contribute information.

A raw `GnssObservation` has no odom association. Proposal construction waits until its timestamp lies in exactly one sealed submap support interval, retrieves two final trajectory knots from that seal, and records the bracket, interpolation fraction, final local revision, trajectory checksum, `SubmapRef`, and source observation ID. Half-open interval ownership chooses the later submap at a boundary. Unbracketed, cross-epoch, ambiguous, or excessive-gap observations are rejected with a typed reason; they are never attached to the nearest current pose.

The exact interpolation is binding and revisioned. For consecutive segment endpoints `(R_0, p_0, v_0)` and `(R_1, p_1, v_1)` with `α = (t-t_0)/(t_1-t_0)`, use shortest-branch geodesic rotation

`R(t) = R_0 Exp(α Log(R_0^T R_1))`

and cubic Hermite translation from `(p_0, v_0, p_1, v_1)` over the actual segment duration. This is deterministic interpolation between discrete finalized states, not a continuous-time optimization variable. `JointEndpointPoseVelocityCovariance` has fixed block order `[δp_0^I0, δθ_0^I0, δv_0^S, δp_1^I1, δθ_1^I1, δv_1^S]`: each pose pair is the translation/rotation part of the right-local SE(3) tangent in its own endpoint IMU frame, while each additive velocity perturbation is in submap frame `S`; cross-covariances between those declared coordinates are retained. The interpolation Jacobian explicitly rotates the right-local pose-translation blocks into `S` before differentiating the Hermite position. Analytic interpolation/lever-arm Jacobians propagate that joint covariance and the segment's calibrated `InterpolationModelNoise` into `GnssFactorProposal.local_interpolation_covariance`. The proposal records the segment/policy/covariance/trajectory checksums. A segment beyond configured duration, rotation/log-branch, model-noise, rank, or conditioning limits yields `INTERPOLATION_UNSUPPORTED`; two pose marginals are never combined as if independent.

For GNSS observation time `t` owned by sealed submap `s`:

```text
p_Sa(t)       = T_S_B(t) p_B_antenna
p_enu_hat(t)  = T_enu_map T_map_S p_Sa(t)
r_gnss        = p_enu_hat - p_enu_measured.
```

The factor touches only `A_s` and `G`. The interpolated local pose and lever arm are fixed inputs with uncertainty. Effective covariance includes receiver ENU covariance, local trajectory interpolation covariance, lever-arm calibration covariance, and explicit model inflation for sparse brackets/dynamics.

Multiple fixes in one submap share local uncertainty. Apply configured time and antenna-distance spacing first. For each remaining proposal, linearize its anchor block at the shadow seed and form `H = J_Aᵀ Σ_eff^-1 J_A` in the declared anchor tangent. Maintain the accepted information sum `I_s` and a configured positive-definite directional cap `C_s`. Choose the largest deterministic `α ∈ [0,1]` satisfying

`λ_max(C_s^-1/2 (I_s + α H) C_s^-1/2) ≤ 1`

by bounded bisection; insert information `α H` (equivalently inflate covariance by `1/α`) or reject when `α` is below the configured minimum. Record spacing decisions, `α`, spectra, cap revision, and cumulative `I_s`. This is a correlation guard, not a claim that the retained fixes are independent.

After an outage, preserve the origin and normally the established alignment. Accumulate a qualified moving window, build individually identified candidate factors across the affected anchors, run the complete bounded active graph shadow with GNC, and require sustained post-solve NIS consistency before one atomic reacquisition commit. Failure returns to `SUSPECT`; it never corrupts local `odom`. Datum, fit window, covariance, FSM/quarantine state, last admitted observation, and source checksums are committed through `GnssAlignmentRevision` and recovered before new GNSS admission.

Legacy WGS84 conversion, exact timestamp interpolation, lever-arm Jacobians, endpoint tests, and datum-excitation concepts are extraction material. Replace the unweighted datum fit and permanent auto-disable behavior.

### 14.6 `map -> odom`

At each durable global commit, choose the reference deterministically as the map-connected sealed submap in the current `OdomEpoch` with greatest `(support_time.end, SubmapId, SubmapContentRevision)`. A newly sealed submap does not replace the previous reference until its anchor commit is durable. For that named reference `s`:

`T_map_odom = T_map_submap_s (T_odom_submap_s)^-1`.

`MapOdomTransform` records graph revision, odom epoch, exact `SubmapRef`, covariance semantics, fusion-clock valid-from time, and health. It is not derived from whichever body pose is latest when optimization finishes. With right perturbations and immutable `B = T_odom_submap_s`, its optional covariance is the global marginal transported as

`Σ_map_odom = Ad_B Σ_A_s Ad_Bᵀ`.

This covariance is conditional on the sealed local frame: it includes uncertainty of the global anchor under the committed global graph, but excludes current local-body/odometry uncertainty and any unavailable global/local cross-correlation. Consumers must not use it as global `base_link` covariance; a globally referenced body covariance requires explicit composition with the named local estimate and a stated correlation approximation.

`map_odom_status = Aligned` if and only if the commit carries a transform whose epoch/reference match that status. `Unaligned` or `StaleOrInvalid` commits carry no transform; the last durable value remains forensic history but is not current output.

`odom -> base_link` remains smooth and is never corrected. `map -> odom` may step only at a successful durable global commit, including a deterministic reference switch in that same commit. Do not estimator-smooth it; an optional visualization-only `map_smooth` frame may interpolate for display but is forbidden for planning/localization truth.

A local reset increments `OdomEpoch` and invalidates the old transform for the new epoch. Until the current epoch has a durable map-connected anchor, `GlobalCommit` carries an explicit `Unaligned` status and no transform/TF; it never substitutes identity. Atlas localization and cross-session alignment are future map-store capabilities and are not available in the current revision.

## 15. ROS 2 boundary

### 15.1 Node responsibilities

`meridian_ros` supplies reusable lifecycle adapter components, wire validation/conversion, QoS, TF, parameter/config loading, diagnostics bridging, and bounded enqueue/dequeue. It contains no estimator, descriptor, registration, graph, map, persistence algorithm, or executable composition root. `meridian_apps` owns the two required executables, instantiates the local/global algorithm runtimes and persistence owners, and wires them to these adapters.

Both required nodes use `rclcpp_lifecycle::LifecycleNode`:

- `configure`: load/validate schema, rig, clocks, models, queue budgets, and persistent state through the `meridian_apps` composition root while ingress is closed;
- `activate`: verify publisher QoS matches, open ingress, activate outputs, and begin TF only when the owning estimate is valid;
- `deactivate`: close ingress, drain/abort bounded work, journal/checkpoint, stop TF;
- `cleanup`: release runtimes, blobs, GPU contexts, and storage;
- `error`: freeze the affected writer, publish a fault when possible; global failure never stops local odometry.

Lifecycle state alone does not disable subscriptions, so callbacks have an explicit admission gate.

### 15.2 Configurable inputs

Topic names and QoS are deployment configuration, not compiled constants.

| Input role | ROS type | Initial QoS policy | Adapter requirements |
|---|---|---|---|
| IMU | `sensor_msgs/msg/Imu` | sensor-data, best effort, bounded depth sized from measured jitter | stamp, adapter-ingress sequence, inferred-gap, and saturation audit; no invented device sequence and no AHRS fusion |
| LiDAR | `sensor_msgs/msg/PointCloud2` | match Ouster publisher; small volatile history | mandatory `x/y/z` and per-point `t` ns or validated `time` s; row/field bounds |
| Camera | `sensor_msgs/msg/Image` or `CompressedImage`; `CameraInfo` | sensor-data, keep newest | decode in conversion worker; exposure convention; exact calibration match |
| GNSS PVT | `sensor_msgs/msg/NavSatFix` | reliable when publisher supports it, bounded volatile history | covariance type/status validation; WGS84 only |
| GNSS receiver status | receiver-specific adapter | reliable, bounded | RTK state, correction age, DOP, satellites when actually available |

Deployment preflight records actual publisher reliability/durability/history and fails or explicitly selects a compatible subscriber. Endpoint compatibility and publisher changes remain monitored after activation; an incompatible replacement closes that ingress and reports a typed transport degradation. Deep histories are not used to hide an underscheduled consumer; stale age is always measured.

The local node subscribes to raw IMU, camera, and LiDAR only. The global node receives sparse-seal/provisional-summary metadata plus raw GNSS/status, and runs the seal-outbox replay protocol; it never subscribes to raw IMU, camera, or LiDAR.

### 15.3 Publications and TF

| Output | ROS type/contract | QoS | Authority/meaning |
|---|---|---|---|
| `/meridian/local/odometry` | `nav_msgs/Odometry` | best effort, volatile, keep last 5 | high-rate `odom -> base_link`, anchored to named local revision |
| `/tf` `odom→base_link` | TF dynamic | standard TF QoS | local node only |
| `/meridian/local/commit` | versioned custom summary | reliable, bounded | optimized local commit metadata |
| `/meridian/local/global_keyframe_summary` | versioned `GlobalKeyframeSummary` | reliable, volatile, bounded | replaceable retrieval scheduling cache; never graph evidence |
| `/meridian/local/sparse_submap_seal` | versioned seal DTO + blob refs | reliable, transient-local, bounded + outbox replay | loss-intolerant final sparse artifact |
| `/meridian/global/commit` | versioned custom DTO | reliable, transient-local, keep last 16 + journal replay | durable revision delta/dispositions |
| `/meridian/global/anchor_snapshot` | custom snapshot/blob ref | reliable, transient-local, keep last 1 | complete latest bounded-active anchor set |
| `/meridian/global/map_to_odom` | versioned status + optional transform record | reliable, transient-local, keep last 4 | revision/reference/covariance metadata companion to TF; explicitly absent when unaligned |
| `/tf` `map→odom` | TF dynamic | standard TF QoS | global node only; rebroadcast latest valid durable value |
| local/global paths | `nav_msgs/Path` | reliable, transient-local, keep last 1 | throttled debug visualization only |
| health/status | typed v2 messages | reliable, transient-local, keep last 1 | current capabilities, FSMs, causes, revisions |
| debug images/clouds/markers | standard/custom debug | best effort, volatile, keep last 1 | gated and rate-limited, never a control input |

DDS durability is not a journal. The local composition root exposes idempotent `get_seals_since(outbox_sequence)` and `ack_seal(...)` operations backed by `SealOutbox`; global exposes `get_commits_since(GlobalGraphRevision)` backed by `GraphJournal`. Every blob owner also exposes the section 6.4 `describe/stat/read/acquire/release` protocol keyed by `BlobStoreId`; the seal-replay response includes the current `BlobStoreDescriptor`. These operations may use ROS services/actions, a bounded deployment-local RPC, or a shared durable volume adapter, but their domain IDs, range limits, retry, acquisition, and checksum semantics are identical. Bulk bytes do not ride in unbounded DDS service responses. Heavy immutable final data moves by `BlobRef` through this protocol; shared-memory leases are limited to replaceable provisional/debug data. DDS carries metadata and small diagnostics.

The revisioned `/meridian/global/map_to_odom` record carries `MapOdomStatus` plus optional `MapOdomTransform`, retaining fusion-clock `valid_from`, reference submap, graph revision, and covariance semantics when aligned. While active and `Aligned`, the global adapter rebroadcasts the latest durable transform at a configured fixed rate using the current ROS clock so late TF consumers can resolve it; rebroadcast does not create a graph revision or change the value. It publishes no `map -> odom` TF while inactive, unaligned, current-epoch-mismatched, or explicitly stale/invalid. A versioned fusion-to-ROS clock bridge detects `/clock` activation and backward/forward jumps, withholds future-dated TF until the new clock epoch is valid, and records the bridge revision in ROS-side diagnostics. No identity fallback is allowed.

### 15.4 ROS-free enforcement and conversion tests

Domain types contain no `rclcpp`, ROS message, allocator, clock, TF, or parameter type. Conversions are total functions returning typed validation errors. Required tests cover malformed fields, wrong datatypes, endian variants, truncated/padded rows, unaligned fields, missing point time, alternate Ouster layouts, image strides/encodings, NaN/Inf, covariance status, timestamp overflow, and golden captured packets.

Transport delivery uses two distinct counters. `ingress_sequence` is generated after successful adapter conversion and exposes internal queue/replay gaps; `device_sequence`, when actually present in the driver payload, exposes upstream loss. When no device counter exists, timestamp gaps are reported as an explicitly inferred upstream-loss signal and never mislabeled as a source-sequence gap.

## 16. Observability and real-time resource control

Observability is part of each API, not an afterthought. Every processing report includes trace and input/output IDs, calibration/config/clock/content/local/global revisions, queue wait, compute time, disposition, capability state, and resource profile.

### 16.1 Typed reports

The core defines typed snapshots/reports; ROS may additionally expose a low-rate scalar bridge for ad hoc dashboards.

- `IngressReport`: schema, raw/fusion/arrival timestamps, source epoch, optional device-sequence gap, adapter ingress sequence/queue gap, conversion latency, and rejection.
- `ImuReport`: support interval, gaps, saturation, initialization, bias/reintegration, covariance.
- `VisualReport`: tracks, cells, parallax, keyframe cause, inliers, NIS, recovery state.
- `LidarReport`: raw/valid/deskewed counts, support, overlap, residual quantiles, rank spectrum, deskew revisions.
- `LocalSolveReport`: variables/factors by type, cost/NIS, relinearization, marginal rank, solve/rebuild/commit timing.
- `SubmapReport`: split/seal cause, finalization lag, trajectory/proxy size, manifest/checksum.
- `LoopReport`: retrieval candidates/votes, verification gates, covariance/rank, PCM/GNC disposition.
- `GnssReport`: solution quality, FSM transition, baseline/excitation, alignment uncertainty, innovation/NIS, quarantine.
- `GlobalSolveReport`: parent/candidate revision, factor weights, cost, validation gates, correction, durability.
- `PersistenceReport`: spool/cache bytes and pins, outbox head/ACK lag, replay requests, prepared/committed journal offsets, checkpoint age, fsync latency, and capacity state.
- `RuntimeReport`: queues by count/bytes/age, drops/coalesces, CPU/GPU memory, utilization, temperature, power, deadline misses.

String-keyed telemetry may mirror these reports, but cannot be the canonical record because it loses schema, units, revisions, and relationships.

### 16.2 Flight recorder

Maintain a bounded in-memory ring of reports and selected low-rate/raw references. A trigger writes a self-contained forensic bundle asynchronously. Triggers include:

- time/sequence discontinuity or IMU gap;
- initialization timeout or odom-epoch reset;
- modality quarantine/recovery;
- LiDAR rank collapse beyond expected geometry;
- local rebuild, solve failure, or deadline miss;
- loop verification acceptance with weak rank, PCM clique replacement, or GNC weight collapse;
- GNSS state transition/sustained NIS failure;
- large global correction or shadow-transaction abort;
- missing blob/checksum, outbox ACK stall, journal recovery/capacity event, queue budget breach, thermal/power throttling.

A bundle includes the configuration/calibration manifests, build/repository IDs, relevant measurement/blob IDs, graph summaries before/after, reports, and a deterministic replay entry point. Debug publication and bundle writing are gated, rate/byte-limited, and may never block the local graph writer.

### 16.3 Degradation governor

The runtime governor reacts to measured deadline pressure in a fixed declared order. An initial policy is:

1. reduce debug image/cloud/path rate;
2. defer global descriptor work and reduce retrieval frequency;
3. cap loop-verification concurrency;
4. suppress optional visual keyframes/features within accepted bounds;
5. select an explicitly configured LiDAR resolution/backend profile;
6. pause dense fusion/meshing later;
7. if the estimator still misses hard deadlines, declare degraded/unobservable rather than silently running stale.

Each transition has hysteresis and is logged as a configuration/profile revision. The governor never changes calibration, robust thresholds, or graph semantics opportunistically.

## 17. Benchmark-first implementation decisions

Every row is an open research decision with a provisional default. Implement the common contract/harness first, then candidates in isolated benchmark targets. Promotion requires accuracy, robustness, resource, and failure-behavior evidence recorded in `docs/OPTIMIZE.md`.

| ID | Decision | Provisional default | Required challengers | Promotion evidence |
|---|---|---|---|---|
| D001 | visual tracking | grid Shi–Tomasi + IMU-seeded KLT; keyframe BRISK recovery | ORB flow/matching; TensorRT SuperPoint+LightGlue | track survival, geometric inliers, recovery, blur/dark/repeat failures, Orin CPU/GPU p99 |
| D002 | visual landmark factor | anchored log inverse range + robust reprojection + Schur | structureless/smart factor; exact re-anchor | ATE/RPE, NEES/NIS, marginal equivalence, memory/solve p99 |
| D003 | visual knot policy | adaptive parallax/coverage/overlap + heartbeat | fixed 5/10/15 Hz | accuracy versus state/factor count and deadline |
| D004 | LiDAR deskew | discrete per-column IMU propagation/interpolation | per-point table; simplified constant-twist baseline | synthetic deskew error, high-dynamics ATE, compute/memory; no CT candidate |
| D005 | local LiDAR residual | multiresolution direct VGICP | voxel-surfel point-to-plane; point-to-point baseline | degeneracy suite, capture range, ATE, calibrated NIS, GPU contention/p99 |
| D006 | registration targets | immutable multires Gaussian frame/keyframe pyramid; recent + overlap selected | local surfel submap variants | overlap/accuracy, source-exclusion proof, bounded bytes/factors |
| D007 | graph representation of LiDAR | nonlinear direct source/target factor | rank-aware prelinearized square root | relinearization accuracy, consistency, solve cost; pose+inverse-Hessian is not a candidate default |
| D008 | degeneracy | normalized spectrum + X-ICP classifier + hysteretic low-rank projection | PerfectlyConstrained variants | correct nullspaces, scale/frame invariance, multimodal recovery |
| D009 | local optimizer | private iSAM2 fixed-lag, square-root marginal prior, rebuild journal | bounded batch LM; Dogleg/GN variants | batch equivalence, NEES, failures/recovery, p99/memory |
| D010 | IMU factor realization | full combined preintegration semantics | split IMU + bias factors | Monte Carlo NEES, graph equivalence, timing |
| D011 | submap split | bounded overlap/quality/travel/time policy | OKVIS2-X seeds and fixed-size policies | loop recall, condensed-edge consistency, memory/finalization latency |
| D012 | visual global retrieval | compact learned descriptor target; DBoW conservative slice | EigenPlaces, MixVPR, DBoW variants | recall@K at fixed false-candidate/compute budget and GPU contention |
| D013 | LiDAR global retrieval | MapClosures primary, Scan Context auxiliary | STD/iBTC where datasets justify | recall/precision under viewpoint, seasons, repeats, compute/index bytes |
| D014 | loop geometric verification | visual generalized PnP+refinement; LiDAR seeded VGICP | GICP/p2plane and ROVER-style trajectory prior | zero catastrophic accepts in adversarial suite, calibrated rank/covariance |
| D015 | robust global admission | rank-aware PCM then complete bounded-active-graph shadow GNC-TLS and atomic durable commit | group-k/riSAM-style incremental robustness | adversarial false clusters, rollback/crash consistency, runtime |
| D016 | GNSS | solution-level PVT/RTK, robust 4-DoF alignment, stateful full-graph reacquisition | raw pseudorange/Doppler only if receiver data/effort justify | NIS, multipath/outage recovery, global accuracy, no local discontinuity |
| D017 | process layout | local and global separate; dense later separate | composition/intra-process deployments | deadline isolation, serialization/blob overhead, restart tests |

Point-to-point, point-to-plane, and VGICP benchmarks must share preprocessing, deskew, target content, initial guesses, robust gate policy, and evaluation timestamps; otherwise the comparison is not causal. Visual benchmarks similarly share images, calibration, keyframe budget, and graph state schedule.

## 18. Testing and acceptance strategy

`docs/TESTING.md` is the operational runbook. The binding requirements are:

### 18.1 Unit and property tests

- transform direction, group operations, right perturbation, and tangent reordering;
- analytic versus numerical Jacobians for IMU, camera, LiDAR, loop, and GNSS factors;
- covariance PSD/order/frame/scale properties and Monte Carlo NEES/NIS;
- exact IMU segment ownership at randomized knot boundaries;
- deterministic strong IDs, content hashes, manifests, revision conflicts, and idempotent seals;
- rank-aware PCM support-intersection/DoF tests, undefined low-rank pairs, no-singleton admission, and oversized-component deferral;
- seal-spool/outbox and graph-journal crash points before/after every durable marker, ACK/GC safety, and checksum-conflict recovery;
- deterministic `map -> odom` reference switching, covariance transport, unaligned absence, ROS-clock jumps, and TF rebroadcast;
- bounded queue count/bytes/age and every overflow policy.

### 18.2 Component and oracle tests

- stationary and motion-init challenge cases, IMU gaps/saturation;
- KLT pure rotation, low parallax, blur, dark, repeated texture, and recovery;
- exact deskew under known angular velocity/acceleration with reference before/inside/after support;
- VGICP/point-to-plane association, source exclusion, robust weighting, and numeric Jacobians;
- tunnel/floor/open field nullspaces where unsupported LiDAR information is exactly zero while visual/IMU factors remain active;
- fixed-lag result versus bounded batch oracle, square-root prior equivalence, delayed factors, duplicate manifests, crash/replay;
- visual PnP and LiDAR loop verification with controlled outliers/partial overlap;
- PCM clique determinism, mutually consistent false clusters, partial-rank mixed-modal cycles, GNC collapse, stale-parent CAS;
- WGS84/ENU reference cases, GNSS alignment observability, interpolation/lever Jacobians, multipath/outage/reacquisition;
- ROS malformed-wire and QoS/late-joiner/restart recovery tests;
- retained nvblox candidate analytic GPU tests when dense work begins.

### 18.3 Replay/fault/endurance gates

Use the exact live core with deterministic scheduling and recorded domain inputs. Required campaigns include sensor dropout/delay/reorder/duplication, time jumps, corrupted payloads, dynamic scenes, LiDAR corridors/open fields, visual darkness/repetition, GNSS multipath/outages, false loop storms, local/global process kills across spool/journal operations, CPU/GPU contention, thermal throttling, and 24-hour bounded-memory replay.

Release acceptance emphasizes zero catastrophic false global commits in the adversarial suite, explicit capability declaration, bounded local p99/deadline under maximum global load, calibrated uncertainty, deterministic revision history, and successful restart/replay. A single average ATE number cannot pass the system.

## 19. Implementation sequence

Each slice ends with build/tests, replay artifacts, metrics, and updated operational docs. Do not scaffold empty packages merely to match the target tree.

1. **Slice 0 — contracts and harness.** Create `meridian_core`; strong IDs/time/geometry/calibration/revisions/manifests; bounded queues; typed reports; deterministic domain-record replay; ROS-free/dependency CI guards. Replace `meridian_msgs` only when real v2 DTOs exist.
2. **Slice 1 — sensor adapters and IMU spine.** Harden ROS conversions, clock/source epochs, stationary initialization, exact segmentation/preintegration, propagation, lifecycle, local odometry and observability.
3. **Slice 2 — visual lane benchmark.** Shared visual contracts/harness, classical and selected challenger, log-inverse-range factors/Jacobians, failure FSM. Record D001–D003 decisions.
4. **Slice 3 — LiDAR lane benchmark.** Exact discrete deskew, immutable target snapshots, point-to-plane oracle/degraded path, VGICP target, degeneracy projection. Record D004–D008.
5. **Slice 4 — local fixed-lag integration.** Coalesced schedule, transactional graph, square-root marginalization/rebuild, commits, combined failure injection and Jetson gates. Record D009–D010.
6. **Slice 5 — sparse submaps and global spine.** Finalization/seals/spool/outbox, condensed adjacent factors, hard-bounded global anchor graph, journal/crash recovery/snapshots, `map -> odom`.
7. **Slice 6 — GNSS.** Domain adapters/status, datum/alignment, exact factor, FSM/quarantine/reacquisition and fault campaigns. Record D016.
8. **Slice 7 — loop closure.** Visual/LiDAR retrieval benchmarks, independent verification, PCM/GNC shadow transactions, adversarial campaigns. Record D012–D015.
9. **Slice 8 — operational hardening.** Lifecycle restart, QoS audits, forensic bundles, degradation governor, endurance/thermal/power gates and deployment manifests.
10. **Later — dense/store.** Extract the nvblox candidate after a clean import, implement local-submap fusion and anchor revisions, then add occupancy/ESDF/map maintenance requirements for planning.

## 20. Legacy extraction and deletion ledger

Git commit `f5ca513158c95aaf88223486ec481c1d42730a21` is the immutable v1 extraction baseline. Use `git show <commit>:<path>` when a slice reaches the item; do not keep broken compatibility packages in the active tree.

Extract and rewrite with new tests:

- pose/frame conventions and selected geometry/covariance test cases;
- affine clock/reorder/monotonicity mathematics, not the LiDAR-owned aggregator;
- camera projection/Jacobians and safe photometric conversion;
- Ouster PointCloud2, image, IMU, GNSS conversion concepts and upstream delivery-gap audit;
- telemetry gating/stage timing/forensic-ring mechanics, upgraded to typed revisioned reports;
- PCM, deterministic maximum clique, chain covariance, GTSAM tangent/PSD guards, and last-good transaction patterns;
- geodesy, lever-arm and exact-time GNSS Jacobian tests;
- deterministic GICP preprocessing/verification and covariance-shaping tests;
- nvblox integration/color/mesh kernels and selected analytic GPU tests under section 13.4;
- generic `eval_ate.py`, `record_tum.py`, and compile-command merge tooling.

Do not port:

- the keyframe-oriented global `IBackEnd`, IMU summaries duplicated in the global graph, or correction feedback;
- monolithic pipeline/node/config packages and compatibility interfaces;
- LiDAR-grouped measurement scheduling, heuristic Euler IMU tracker, constant-screw deskew, mutable rolling global registration map, or point-to-point relative-pose frontend;
- corrected-XY loop eligibility, single-loop odometry veto, raw Hessian-as-covariance, removal-only GNC subproblem, or permanent GNSS auto-disable;
- `IMapLayer`, global mutable `KeyframeStore`, loop-driven dense clear/rebuild, or mission-length raw keyframe retention;
- v1 debug message schemas, launch/RViz files, dataset-hardcoded scripts, generated overlays, or the vendored Scan Context submodule.

The cleanup removes the old implementation packages rather than leaving forwarding targets. New code arrives only through the slices above.

## 21. Configuration, provenance, and tuning

Configuration is schema-versioned and split by ownership:

- rig/calibration artifact: physical sensors, frames, intrinsics, noise, timing conventions, provenance/trust;
- deployment artifact: topics, QoS, executors, CPU affinity, queue/resource budgets, enabled profiles;
- algorithm profile: reviewed defaults and benchmark-selected thresholds;
- benchmark manifest: dataset/bag checksum, topic mapping, ground truth, faults, hardware/build IDs;

Unknown or placeholder calibration cannot be silently promoted to trusted production calibration. The current Barracuda and dataset YAMLs are historical inputs whose transforms/noise/provenance must be audited during Slice 0; v1 pipeline/frontend/backend/map sections are deleted.

Every run writes resolved configuration, calibration hashes, repository/build/container IDs, model hashes, random seeds, hardware/JetPack/CUDA state, dataset checksum, and graph/profile revisions. Every changed tunable follows the ledger in `docs/OPTIMIZE.md`.

## 22. Required pre-implementation reviews

Before Slice 1 algorithm work starts, review and freeze:

- core frame/tangent/time/calibration contracts and serialization versioning;
- exact Ouster stamp/per-point-time convention and IMU/camera/GNSS clock source on Barracuda;
- target Jetson model, sensor rates/resolutions, CPU affinity and GPU memory/power budget;
- trustworthy rig calibration values versus placeholders;
- benchmark corpus with independent holdout and adversarial sequences;
- license and source status of every external code seam;
- factor covariance semantics and measurement-correlation policy;
- what downstream planning requires from future occupancy/ESDF/free/unknown maps.

An unresolved item is recorded as an explicit open decision with an owner and benchmark, not filled with a guessed constant.

## 23. Source-reading index for implementation

These are the exact retained source seams already inspected for this design. Read the complete local context and its tests again when the owning slice begins; paths are not permission to copy incompatible runtime architecture.

| Meridian seam | Reference files to read | Concrete lesson/use |
|---|---|---|
| IMU graph/update ownership | `../slam-reference/glim/src/glim/odometry/odometry_estimation_imu.cpp`; `../slam-reference/glim/src/glim/common/imu_integration.cpp` | one graph-update owner, IMU prediction/preintegration, five-second lag seed, marginalized-frame handling and validation |
| Discrete deskew | `../slam-reference/glim/src/glim/common/cloud_deskewing.cpp`; `../slam-reference/glim/src/glim/preprocess/cloud_preprocessor.cpp` | pose table/interpolation and retained per-point time; Meridian adds exact raw support, revisions and redeskew policy |
| IMU midpoint/bias math | `../slam-reference/GVINS/estimator/src/factor/integration_base.h` | preintegration covariance and bias Jacobian implementation pattern; verify convention/order independently |
| Classical visual tracking | `../slam-reference/GVINS/feature_tracker/src/feature_tracker.cpp`; `../slam-reference/OKVIS2-X/okvis_frontend/src/Frontend.cpp` | grid KLT flow, descriptor recovery, overlap/keyframe and place-verification patterns |
| Reprojection/marginalization | `../slam-reference/GVINS/estimator/src/factor/projection_factor.cpp`; `../slam-reference/GVINS/estimator/src/factor/marginalization_factor.cpp`; `../slam-reference/OKVIS2-X/okvis_ceres/include/okvis/ceres/implementation/ReprojectionError.hpp` | analytic projection Jacobians, inverse-depth/landmark handling, square-root/order lessons; do not copy Ceres ownership |
| Direct multiresolution VGICP | `../slam-reference/glim/src/glim/odometry/odometry_estimation_gpu.cpp`; `../slam-reference/gtsam_points/include/gtsam_points/factors/impl/integrated_vgicp_factor_impl.hpp`; `../slam-reference/gtsam_points/include/gtsam_points/factors/integrated_vgicp_factor_gpu.hpp`; `../slam-reference/gtsam_points/src/gtsam_points/factors/integrated_vgicp_derivatives_*.cu` | immutable Gaussian voxel levels, recent/overlap target selection, binary direct factors, GPU association/derivatives; unary code is not a production contract |
| Fixed-lag implementation | `../slam-reference/gtsam_points/src/gtsam_points/optimizers/incremental_fixed_lag_smoother_ext.cpp`; `../slam-reference/glim/include/glim/odometry/odometry_estimation_imu.hpp` | iSAM2 marginalization/order plumbing; Meridian replaces artificial-prior fallback with checkpoint/journal rebuild |
| LiDAR rank handling | `../slam-reference/perfectlyconstrained/libpointmatcher/pointmatcher/ICP.cpp`; `../slam-reference/perfectlyconstrained/libpointmatcher/pointmatcher/TSVD.hpp`; `../slam-reference/perfectlyconstrained/libpointmatcher/pointmatcher/ErrorMinimizers/PointToPlane.cpp` | TSVD/null-direction mechanics and field failure cases; Meridian projects only LiDAR information and adds length normalization/X-ICP classification |
| Sparse submap lifecycle | `../slam-reference/OKVIS2-X/okvis_multisensor_processing/src/SubmappingInterface.cpp` | overlapping bounded submap policy and finalization cues; Meridian separates non-overlapping factor ownership from retrieval context |
| Visual global verification | `../slam-reference/OKVIS2-X/okvis_frontend/src/Frontend.cpp` | DBoW candidate path, generalized PnP, inlier/distinctiveness gates, refinement/Hessian pattern |
| LiDAR global retrieval | `../slam-reference/MapClosures/cpp/map_closures/MapClosures.cpp` | ground-normalized BEV/feature/RANSAC pipeline; Meridian removes auto-insert, sequential-ID and policy assumptions |
| PCM/GNC | `../slam-reference/Kimera-RPGO/include/KimeraRPGO/outlier/Pcm.h` | pairwise consistency and robust graph patterns; Meridian runs complete revisioned shadow transactions |
| GNSS global behavior | `../slam-reference/OKVIS2-X/okvis_ceres/src/ViGraph.cpp`; `../slam-reference/GVINS` | dropout/reacquisition and GNSS factor lessons; Meridian initially uses solution-level PVT rather than GVINS raw observations |
| Dense integration | legacy baseline paths in section 13.4; upstream imported `src/nvblox` | reuse only integration/color/mesh kernels/tests behind finalized local-submap contracts |

Legacy Meridian extraction targets are named in section 20. For example, inspect old PCM without restoring its package using:

```bash
git show f5ca513158c95aaf88223486ec481c1d42730a21:src/meridian_backend/src/pcm.cpp
git show f5ca513158c95aaf88223486ec481c1d42730a21:src/meridian_ros/src/conversions/ros2core.cpp
git show f5ca513158c95aaf88223486ec481c1d42730a21:src/meridian_map/src/nvblox/nvblox_integrate.cu
```

Several reference folders are source snapshots without `.git` metadata, including key architecture references. They remain valid implementation context. Before copying code from any reference, verify its upstream source and license and update the implementing package's provenance record; an exact-commit checkout is not required. The current production-default design has no missing conceptual reference; learned or alternative benchmark lanes add their official code only when their decision benchmark begins.
