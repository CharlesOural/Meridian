# 09 — Cross-cutting: Debug, Introspection & ROS 2 Tooling

> **Spec status:** normative. This is the *user-priority* spec. The project owner
> made it non-negotiable: **debugging the estimator is a first-class feature**, not
> an afterthought bolted onto the node. An operator or developer must be able to
> *see* what the **continuous-time LiDAR-Inertial-Visual-GNSS (CT LIVO+GNSS)**
> front-end is doing — which LiDAR points it used (at their true sample times),
> which photometric patches it tracked, how big each residual stream is, which axes
> are observable, where the B-spline window is, how long each stage took, where the
> trajectory is and how uncertain — plus what the **nvblox** GPU map is doing
> (TSDF growth, colourisation, mesh, loop-driven region rebuilds) — all without
> recompiling, at a cost that is *zero when off* and *bounded when on*.
>
> **Position in the stack.** This spec defines the `meridian_debug` cross-cutting
> library (the `TelemetrySink` bus, declared in `00_architecture.md` §10.1 and
> injected into every layer at construction via its factory), the `meridian_msgs` debug
> message set, the `RosTelemetrySink` adapter inside `meridian_ros`, the rviz layout,
> the replay/bag path-equivalence guarantee, and the runtime toggle service. It is
> the implementation spec behind `00_architecture.md` §10 and Appendix C; it amends
> neither the architecture (spec 00) nor the boundary types
> (`01_interfaces_and_data_types.md`) — it *consumes* them.
>
> **What the system is (one line, no phasing).** Meridian is *one complete system*: a
> CT B-spline sliding-window tightly-coupled LIVO+GNSS front-end (L2) feeding an
> iSAM2 factor-graph back-end (L3) and a GPU nvblox TSDF+colour+mesh map (L4). The
> CT front-end **is** the design — there is no "filter we ship first" and no
> feature rollout. A FAST-LIO2-style iEKF exists *only* as an optional offline test
> oracle behind the same `IFrontEnd` (spec 00 §5.4); it is never a product path and
> nothing in this debug surface is organised around it. Every channel below is a
> channel of the full CT LIVO+GNSS + nvblox system.
>
> **Grounding.** Meridian combines the apex references — **FAST-LIVO2** (sequential
> ESIKF, sparse-direct photometric vision, unified voxel map, exposure comp),
> **nvblox** (GPU TSDF+colour+Marching-Cubes mesh), **iSAM2/GTSAM** (incremental
> factor graph), **Coco-LIC/CLINS + basalt-headers** (CT SE(3) B-spline). The
> *engineering pattern* for "what signal to publish and how to keep it cheap" is
> additionally anchored to the FAST_LIO reference implementation, cited as
> `laserMapping.cpp:NNN`; the verified inventory of that node's introspection (the
> publisher/topic surface, timing instrumentation, and rviz configs) is collected in
> **Appendix R** (non-normative). FAST-LIO's introspection has the *right signals* but
> is **entangled with the node, hard-coded, and partly disabled**; Meridian keeps the
> good signals, fixes the location/structure/cost discipline, and adds the signals
> a CT LIVO+GNSS system needs that a LiDAR-only filter never had (spline window,
> photometric residuals, per-stream inlier counts, GNSS anchoring). Where this spec
> says "NEW" the signal has no FAST-LIO analogue and is a strict upgrade.
>
> Notation follows the shared block (spec 01 §0–§2): poses $T_{A\_B}\in SE(3)$;
> trajectory $T(t)\in SE(3)$ (the B-spline, spec 00 §7.5); residual $r$; Jacobian
> $H$; information $\Omega=\Sigma^{-1}$; per-axis observability score
> $s\in[0,1]^6$ (spec 01 §3.4). Tangent order is **`[tx,ty,tz,rx,ry,rz]`** (spec 01
> §3.1, §3.4) — *translation-first*, the `ObservabilityReport`/`NavState`
> convention; the `KeyframePacket.constraint_cov` block (spec 00 §6.4, spec 01 §6.1)
> uses rotation-first `[rx,ry,rz,tx,ty,tz]`. **The debug layer always re-states the
> order in the message** (§4.4) so a plot is never mis-axed.

---

## Table of contents

1. [Design goals & the FAST-LIO baseline we improve on](#1-design-goals--the-fast-lio-baseline-we-improve-on)
2. [Architecture: the `TelemetrySink` bus and its sinks](#2-architecture-the-telemetrysink-bus-and-its-sinks)
3. [The telemetry value types (`meridian_debug`, ROS-agnostic)](#3-the-telemetry-value-types-meridian_debug-ros-agnostic)
4. [The `meridian_msgs` debug message set](#4-the-meridian_msgs-debug-message-set)
5. [The complete debug-channel catalogue (what every key publishes)](#5-the-complete-debug-channel-catalogue-what-every-key-publishes)
6. [Per-stage timing](#6-per-stage-timing)
7. [rviz markers: geometric introspection](#7-rviz-markers-geometric-introspection)
8. [rviz layout & display config](#8-rviz-layout--display-config)
9. [Structured logging](#9-structured-logging)
10. [The `RosTelemetrySink` adapter (wrapper-side)](#10-the-rostelemetrysink-adapter-wrapper-side)
11. [Runtime control: toggling debug cheaply](#11-runtime-control-toggling-debug-cheaply)
12. [Cost discipline & the zero-cost-when-off guarantee](#12-cost-discipline--the-zero-cost-when-off-guarantee)
13. [Replay / bag path equivalence](#13-replay--bag-path-equivalence)
14. [Per-module debug-hook checklist](#14-per-module-debug-hook-checklist)
15. [Failure modes & how the debug layer surfaces them](#15-failure-modes--how-the-debug-layer-surfaces-them)
16. [Returnable schema](#16-returnable-schema)

---

## 1. Design goals & the FAST-LIO baseline we improve on

### 1.1 What FAST-LIO does (and where it lives)

FAST-LIO's `laserMapping.cpp` is a 1055-line translation unit that *is* the ROS
node, the parameter loader, the buffer manager, the estimator driver, **and** the
publisher (spec 00 §1.4). Its introspection is concretely:

| FAST-LIO signal | Location |
|---|---|
| `/cloud_registered` (world cloud) | advertised `laserMapping.cpp:849`; built in `publish_frame_world`, `laserMapping.cpp:478` |
| `/cloud_registered_body` (body cloud) | advertised `:851`; built in `publish_frame_body`, `:532` |
| `/cloud_effected` (inlier/effective points) | advertised `:853`; built in `publish_effect_world`, `:551` |
| `/Laser_map` (local map) | advertised `:855` |
| `/Odometry` with hand-packed covariance | advertised `:857`; covariance packed by hand into `pose.covariance`, `:597-606` |
| `/path` (trajectory) | advertised `:859`; `publish_path`, `:622`, throttled `if (jjj % 10 == 0)` |
| effective-point **count** | global `effct_feat_num`, computed in `h_share_model`, `:695` |
| mean residual | global `res_mean_last`, `:715` |
| per-stage timing | `aver_time_match/solve/icp/...` to `printf` + CSV, `:991-1009`, finalised `:1042-1044` |
| state/bias trace | `dump_lio_state_to_log`, `:150` |
| init gate | `flg_EKF_inited`, `:898` |
| parallel residual loop | `#ifdef MP_EN #pragma omp parallel for`, `:646-649` |

These are exactly the right *quantities* for the LiDAR + IMU part of the problem.
The problems are structural, and there is a whole class of signal FAST-LIO has *no*
analogue for because it is a LiDAR-only discrete-time filter (the CT trajectory,
the photometric/visual stream, GNSS):

1. **Wrong location.** Publishers, covariance packing, and timing are interleaved
   in the node and the EKF callback. You cannot reuse them off-ROS, mock them in a
   test, or hang a debug surface off them cleanly.
2. **Hard-coded and partly disabled.** Topic names, queue depths, and throttles
   are literals; the effective-cloud publish call is **commented out at the call
   site** (`laserMapping.cpp:983`) so the most diagnostically valuable cloud is
   off by default and only re-enabled by editing source.
3. **Signals smuggled into the wrong type.** The 6×6 EKF covariance is hand-copied
   into `nav_msgs/Odometry.pose.covariance` with a rotation/translation block-swap
   (`:597-606`) — an implicit convention that is easy to get wrong (spec 00 §6.4).
   The effective count and mean residual live in globals, never published as
   first-class plottable scalars.
4. **No observability / degeneracy signal.** The only health bit is the binary
   `flg_EKF_inited`. There is nothing an operator can watch to see a corridor
   becoming degenerate.
5. **Timing is a logfile, not a live stream.** `aver_time_*` is printed and
   CSV-dumped; you read it *after* the run, not *during*.
6. **No CT, no vision, no GNSS surface.** A CT LIVO+GNSS estimator has a *trajectory*
   (B-spline control points + active window), a *photometric* residual stream
   (sparse-direct patches with LiDAR-depth), and a *GNSS* residual stream — none of
   which exist in FAST-LIO's introspection because none exist in FAST-LIO. These are
   first-class in Meridian (§5.1).

### 1.2 Meridian's goals (normative)

- **G1 — Every useful FAST-LIO signal is retained**, routed through one
  ROS-agnostic bus (`TelemetrySink`, spec 00 §10.1), and *improved*: counts and
  residuals become first-class plottable scalars; covariance is a typed block with
  a stated order; the effective/inlier cloud is on by default (rate-limited), not
  commented out.
- **G2 — The core emits structured telemetry; the wrapper decides surfacing.** No
  `RCLCPP_INFO`, no `ros::Publisher`, no `MarkerArray` below `meridian_ros`. The core
  calls `sink->scalar(...)`; the wrapper maps it to a topic, a CSV, or `/dev/null`.
- **G3 — The full CT LIVO+GNSS state is visible.** The B-spline window and knots,
  the three measurement-residual streams (LiDAR point-to-plane at true point time,
  sparse-direct photometric, GNSS) and the IMU-derivative residual, are each
  separately plottable. The system fuses many modalities into one trajectory; the
  debug surface lets you see *each modality's contribution and health* separately.
- **G4 — Degeneracy is plottable.** Per-axis observability $s\in[0,1]^6$ (spec 01
  §3.4) is published as a vector *and* drawn as a 6-bar marker. This is the single
  most important upgrade for tactical operation: the operator *sees* a tunnel.
- **G5 — The nvblox map is observable.** TSDF block growth, colourisation, mesh
  extraction timing, and loop-driven clear-and-rebuild regions are published so the
  operator watches the GPU map heal after a loop closes.
- **G6 — Zero cost when off, bounded cost when on.** A `scalar()` to a `NullSink`
  is one inlinable virtual call to an empty body. Clouds/markers/patch-overlays are
  gated by a config flag *and* a token-bucket rate limiter. Introspection never
  starves the estimator (spec 00 §10.6).
- **G7 — Toggle without restart.** A `DebugControl` ROS service flips per-key
  publishing, telemetry rate, and log level live (spec 00 §8.3, §10.5).
- **G8 — Replay == live.** The exact same telemetry is produced whether driven by
  a live `ISensorSource` or a bag-replay one, so a forensic session on a recorded
  bag is bit-for-bit the same debug view as the field run (§13).

---

## 2. Architecture: the `TelemetrySink` bus and its sinks

### 2.1 One interface, many sinks

`meridian_debug` defines a single pure-virtual `TelemetrySink` (declared in spec 00
§10.1, finalised in §3 below). Every layer holds a `TelemetrySink*` injected by
the pipeline at construction (a factory parameter — `makeFrontEnd(cfg, calib, sink)`,
spec 00 §5.1; the same shape on every module factory) and writes to it. The
pointer is **never owned** by the layer;
`MeridianPipeline` owns the sink and outlives every module (spec 00 §11.3).

```
  L0  L1  L2  L3  L4  L5   (each holds a TelemetrySink*, writes structured calls)
   \   \   |   |   /   /
    \   \  |   |  /   /
     ▼  ▼  ▼   ▼ ▼   ▼
        ┌──────────────────────────────┐
        │   TelemetrySink (interface)   │   meridian_debug, NO ros
        └──────────────┬───────────────┘
        ┌──────────────┼───────────────┬──────────────────┐
        ▼              ▼                ▼                  ▼
  RosTelemetrySink  RecordingSink   CsvTelemetrySink    NullSink
  (meridian_ros)       (tests)         (meridian_tools)       (meridian_debug)
   → topics/TF/        → captures      → per-key CSV       → empty bodies
     markers             for asserts     for offline         (default off)
```

Sinks are **selected at pipeline construction** by `Config.debug` (spec 00 §8.2).
The live node binds `RosTelemetrySink`; the offline `replay` tool binds either
`RosTelemetrySink` (to view in rviz) or `CsvTelemetrySink` (to plot in pandas);
unit tests bind `RecordingSink`; a production run with debug off binds the
*default* `NullSink`. This is the architectural inversion of FAST-LIO: the
publishing decision is at the *edge*, the signal generation is in the *core*.

### 2.2 Why a sink interface and not just ROS publishers

The four reasons are exactly the architecture's (spec 00 §1): testability without
middleware, transport replaceability, reasoning at the right altitude, and not
repeating FAST-LIO's entanglement. Concretely for debug: a regression test can
assert "`frontend/lidar/n_inlier` never dropped below 50 over this bag" or
"`frontend/visual/n_tracked` stayed above 30" by binding a `RecordingSink` and
reading its buffer — **impossible** in FAST-LIO where those counts are globals
printed to a CSV.

### 2.3 The `MultiSink` fan-out

`meridian_debug` provides a `MultiSink` that forwards each call to an ordered list of
child sinks. This is how the offline tool publishes to rviz **and** records to CSV
in one run, and how the live node can simultaneously publish ROS topics and write
a black-box CSV for post-incident forensics. `MultiSink` short-circuits when all
children are gated off for a key (§12).

### 2.4 The `IntrospectionHooks` slot — white-box access without serialisation

`TelemetrySink` is a *fan-out of copies*: every channel is a value type that the core
hands off and forgets, by design, so a slow or misbehaving sink can never reach back
into estimator state. That property is exactly what makes it unsuitable for one job —
**white-box testing of rich, non-serialisable internal structures**: the live iSAM2
Bayes tree and factor graph, the `ISAM2Result` detail, the active spline knot vector,
the windowed Hessian before it is reduced to six observability scores. Serialising
these into a `meridian_msgs` type to assert on them would be a large, brittle parallel
representation that drifts from the real object — the very entanglement this spec
exists to avoid.

`meridian_debug` therefore defines a *second, parallel* slot, `IntrospectionHooks`,
held by a layer alongside its `TelemetrySink*` and injected the same way (a factory
parameter, never owned by the layer). Where the sink *pushes copies out*, the hook
slot lets a consumer *look in* at a live object **by `const` reference**, under three
non-negotiable constraints that keep it from becoming a backdoor:

- **`const`-ref only, read-only.** A hook receives `const T&` to the live structure
  and MUST NOT mutate it or retain the reference beyond the call. The signature is
  `const`-qualified on the producer side; the slot exposes no non-`const` access.
- **Subscriber-gated, zero-cost when unsubscribed.** The slot has a cheap
  `bool subscribed(Hook) const` predicate — the white-box analogue of
  `TelemetrySink::enabled()`. The default (production, replay-without-test) slot is a
  `NullHooks` whose `subscribed()` is always `false`, so the producer skips the hook
  call entirely. No serialisation, no copy, no allocation occurs unless a consumer is
  registered. A live build binds `NullHooks`; a unit/integration test binds a
  recording consumer.
- **Thread-confined on the producer.** A hook fires **synchronously, on the producer
  layer's own thread, inside the call that holds the structure** — never from another
  thread and never deferred. The consumer runs to completion before the producer
  proceeds, so the `const&` is valid for exactly the hook's duration and no locking of
  estimator state is introduced. A consumer that blocks blocks that one stage; in the
  single-threaded determinism/replay mode (spec 00 §11.2) used by tests this is simply
  in-line execution. The hook is therefore **forbidden in production postures by
  policy** — it exists for the deterministic test/replay path, and `NullHooks` is the
  only binding a live node ever uses.

This gives the back-end test surface (spec 10 §8.1–§8.2) a way to assert on the
*actual* iSAM2 graph — factor count by type, the relinearised variables in a
`GraphUpdate`, the gauge-anchor presence, that a window-restart produced exactly one
`CombinedImuFactor` and no `BetweenFactor` — by reading the live object, with no
message schema in between and no path that runs only under test in a way the live
code does not (the call site is always present; only the binding differs, exactly the
`TelemetrySink`/`NullSink` discipline of §2.1). It is ROS-agnostic: `IntrospectionHooks`
lives in `meridian_debug` and names only core/GTSAM types, never a ROS message.

### 2.5 The `ParquetTelemetrySink` — the harness recording sink and its schema

The evaluation harness (spec 10) records every telemetry call to disk as columnar
Parquet, one file per domain, and then computes metrics and the determinism hash over
those files. That sink is `ParquetTelemetrySink`, defined here because it is a
`meridian_debug`/`meridian_tools` `TelemetrySink` implementation (it owns no ROS) and
because the harness depends on a *fixed* schema it can read without coordinating with
the producer. It is the recording peer of `CsvTelemetrySink` — same input (the bus),
durable columnar output instead of per-key CSV.

**Routing.** `ParquetTelemetrySink` writes one file per domain prefix, splitting on
the first path segment of the key so a query touches only the relevant file:
`frontend.parquet` (`frontend/*`, `odom/*`, `calib/*`), `backend.parquet` (`backend/*`,
`place/*`), `map.parquet` (`map/*`, `pipeline/*`), and `timing.parquet` (every
`timing()` call). This is the `run_dir/debug/*.parquet` layout the harness consumes
(spec 10 §1.2).

**Fixed columnar schema.** Every scalar/vector/event/timing call lands as one row in a
single, fixed wide schema (heavy payloads — clouds, images, markers, poses — are
*not* written to Parquet; they go to their own artefact writers, spec 10 §1.1, and
Parquet records only that they fired, via a `kind` tag and a null value vector). The
columns are:

| column | type | meaning |
|---|---|---|
| `stamp_ns` | `int64` | the `meridian::Timestamp` of the call (measurement time, never wall time, §13.2) |
| `key` | `string` (dictionary) | the `const char*` channel key, or the stage name for `timing()` |
| `kind` | `uint8` | `0 scalar, 1 vec, 2 timing, 3 event, 4 payload-marker` |
| `values` | `list<float64>` | scalar → 1 elem; vec → N; timing → `[ms]`; event/payload-marker → empty |
| `axis_order` | `string` (dictionary) | the `vec()` axis order (§4.4), else `""` |
| `unit` | `string` (dictionary) | the `unit_of(key)` string (§10) |
| `level` | `uint8` | for `kind=event`: the `Level`; else `0` |
| `tag` | `string` (dictionary) | for `kind=event`: the event tag; else `""` |
| `message` | `string` | for `kind=event`: the `key=value` detail line; else `""` |
| `seq` | `uint64` | monotone per-(file) write counter, the deterministic tie-break for equal `(stamp_ns,key)` (§13.4) |

The schema is **append-only and versioned** by a `schema_version` Parquet
file-metadata key; a reader keys off it so adding a column never breaks an archived
run (spec 10 §7.3 re-runs old baselines). Dictionary-encoding the low-cardinality
string columns keeps the files small and fast to scan in pandas/duckdb (spec 10 §10).

**Determinism contract.** The hash the harness pins (§13.4, spec 10 §7.4) is computed
over these records under a canonical ordering, and `ParquetTelemetrySink` is the
single producer of that record set. The sink is only used on the single-threaded
replay/determinism path; it is not a live-node sink (the live node uses
`RosTelemetrySink` plus an optional black-box `CsvTelemetrySink`, §11.1).

---

## 3. The telemetry value types (`meridian_debug`, ROS-agnostic)

This finalises the interface sketched in spec 00 §10.1. It depends only on
`meridian_common` (for `Pose`, `PointCloud`/`LidarPoint`, `Frame`, `Timestamp`,
`VecX`) and on nothing ROS. All identifiers are `const char*` keys (compile-time
literals at the call sites) so a `NullSink` does zero allocation.

```cpp
// include/meridian/debug/telemetry.hpp        (meridian_debug — NO ros, NO pcl-msgs)
namespace meridian {

enum class Level { Trace, Debug, Info, Warn, Error };   // §9

// A geometric overlay primitive — the ROS-agnostic precursor of
// visualization_msgs/Marker. The wrapper maps it 1:1 (§10). Kept minimal.
struct Marker {
  enum class Type { Points, LineList, LineStrip, Arrow, Cube, Sphere, Text, Hexagon };
  Type                       type   = Type::LineList;
  Frame                      frame  = Frame::Map;        // frame_id of the marker
  std::string                ns;                         // marker namespace (e.g. "place/loop_edge")
  std::int32_t               id     = 0;                 // stable id within ns (so it updates, not piles up)
  std::vector<Eigen::Vector3f> points;                   // geometry (meaning per Type)
  std::array<float,4>        color  = {1,1,1,1};         // rgba
  std::vector<std::array<float,4>> colors;               // optional per-point color (e.g. confidence)
  float                      scale  = 0.1f;              // line width / point size / arrow shaft [m]
  std::string                text;                       // for Type::Text
  Duration                   lifetime_ns = 0;            // 0 = forever (until overwritten by same ns/id)
};

// An image-overlay primitive — the ROS-agnostic precursor of a debug image
// (sensor_msgs/Image). Used for the sparse-direct photometric patch overlay (§7.5):
// the raw camera image with the tracked patches drawn on it, coloured by residual.
struct ImageOverlay {
  Frame                 frame = Frame::CamLink;          // camera frame
  int                   width = 0, height = 0;
  std::span<const std::uint8_t> base;                    // borrowed mono/RGB image bytes (no copy)
  enum class Encoding { Mono8, RGB8 } encoding = Encoding::Mono8;
  struct Patch { Eigen::Vector2f uv; float residual; float depth; int level; };
  std::vector<Patch>    patches;                         // tracked patches: pixel, photometric r, LiDAR depth
};

// The bus. The core writes; a sink implements. Spec 00 §10.1 canonical.
class TelemetrySink {
public:
  virtual ~TelemetrySink() = default;

  // Cheap-gate query: the core MAY call this to skip building an expensive payload
  // (e.g. a 1M-point cloud, or a patch overlay) when the key is currently disabled.
  // Returns false for NullSink and for any key gated off by DebugControl (§11).
  // This is the hook that makes "don't even build the payload" possible — see §12.
  virtual bool enabled(const char* key) const = 0;

  // --- scalar / vector telemetry (plottable) ---
  virtual void scalar(const char* key, double v, Timestamp t)            = 0;
  virtual void vec   (const char* key, const Eigen::Ref<const Eigen::VectorXd>&,
                      Timestamp t, const char* axis_order = nullptr)      = 0;

  // --- heavy payloads (gated + rate-limited downstream) ---
  virtual void cloud (const char* key, const PointCloudView&, Frame, Timestamp) = 0;
  virtual void pose  (const char* key, const Pose&,           Frame, Timestamp) = 0;
  virtual void marker(const Marker&,                                  Timestamp) = 0;
  virtual void image (const char* key, const ImageOverlay&,           Timestamp) = 0;

  // --- timing ---
  virtual void timing(const char* stage, double ms, Timestamp t)         = 0;

  // --- structured events (also feed the log, §9) ---
  virtual void event (Level, const char* tag, std::string_view msg, Timestamp t) = 0;
};

// RAII scoped timer: on destruction calls sink->timing(stage, elapsed_ms, t0).
// Reads a steady clock twice. The ONLY way the core measures stage time (§6).
class ScopedTimer {
public:
  ScopedTimer(TelemetrySink* s, const char* stage, Timestamp t)
      : s_(s), stage_(stage), t_(t), start_(Clock::now()) {}
  ~ScopedTimer() {
    if (s_) s_->timing(stage_, ms_since(start_), t_);
  }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;
private:
  TelemetrySink* s_; const char* stage_; Timestamp t_; Clock::time_point start_;
};

// Convenience macro: compiles to nothing extra in release; the steady-clock reads
// are the only cost, and they vanish if s==nullptr is hoisted by the optimizer.
#define MERIDIAN_SCOPED_TIME(sink, stage, t) ::meridian::ScopedTimer _ht_##__LINE__((sink),(stage),(t))

// The white-box introspection slot (§2.4). Parallel to TelemetrySink, but instead
// of fanning value copies OUT it lets a registered consumer look IN at a live,
// non-serialisable internal structure by const reference, synchronously on the
// producer's own thread. The producer guards every hook on subscribed() so an
// unsubscribed slot (NullHooks, the production/replay-without-test default) costs one
// bool. Enumerate the inspectable structures; each new hook is a new enumerator and a
// new const-ref-taking virtual. ROS-agnostic — names only core/GTSAM types.
enum class Hook {
  BackendGraph,      // const gtsam::NonlinearFactorGraph& : the live factor graph
  BackendISAM2,      // const gtsam::ISAM2&                 : Bayes tree / linearisation point
  BackendUpdate,     // const ISAM2ResultExt&              : the last update's relinearised set
  FrontendWindow,    // const SplineWindow&                : active knot vector + state
  FrontendHessian    // const Eigen::MatrixXd&             : windowed Hessian pre-observability
};

class IntrospectionHooks {
public:
  virtual ~IntrospectionHooks() = default;
  // Cheap gate: false for NullHooks and for any hook with no registered consumer.
  // The producer MUST call this before assembling/handing a structure to a hook.
  virtual bool subscribed(Hook) const = 0;

  // Each visit hands the consumer a CONST reference, valid only for the call's
  // duration, executed synchronously on the producer thread. The consumer MUST NOT
  // mutate or retain. Defined per inspectable structure so the type is explicit and
  // there is no type-erased downcast at the boundary.
  virtual void visit_graph (const gtsam::NonlinearFactorGraph&) = 0;
  virtual void visit_isam2 (const gtsam::ISAM2&)                = 0;
  virtual void visit_update(const ISAM2ResultExt&)              = 0;
  virtual void visit_window(const SplineWindow&)                = 0;
  virtual void visit_hessian(const Eigen::MatrixXd&)            = 0;
};

// The production / live / replay-without-test binding: every subscribed() is false,
// every visit is an empty body. Bound by default exactly as NullSink is.
class NullHooks final : public IntrospectionHooks {
public:
  bool subscribed(Hook) const override { return false; }
  void visit_graph (const gtsam::NonlinearFactorGraph&) override {}
  void visit_isam2 (const gtsam::ISAM2&)                override {}
  void visit_update(const ISAM2ResultExt&)              override {}
  void visit_window(const SplineWindow&)                override {}
  void visit_hessian(const Eigen::MatrixXd&)            override {}
};

} // namespace meridian
```

**Usage at the producer (back-end), guarded exactly like `enabled()`:**

```cpp
if (hooks_->subscribed(Hook::BackendUpdate))
  hooks_->visit_update(result_ext);   // synchronous, const&, on the back-end thread
```

**`PointCloudView`** is a non-owning `std::span`-like view over `LidarPoint`s (or
a typed XYZI subset) so the core hands the sink a *view*, not a copy; the
`RosTelemetrySink` copies into a `PointCloud2` only if the key is enabled (§12).
This is the structural fix for FAST-LIO building the full `PointCloudXYZI` *before*
deciding whether to publish (`publish_frame_world`, `laserMapping.cpp:478`).

**`ImageOverlay`** is likewise non-owning: `base` is a borrowed `std::span` over
the camera bytes the front-end already holds (spec 01 §4.3 `CameraFrame::data`),
and `patches` are the sparse-direct tracked points. The wrapper rasterises this
into a `sensor_msgs/Image` only when the key is enabled. This is the visual-stream
analogue of the cloud view: build nothing until someone wants it.

**Why `enabled(key)` is on the interface.** The single most expensive debug actions
are materialising a registered world cloud (transform every point by the
trajectory at its true time) and rasterising the patch overlay. FAST-LIO always
builds the cloud in `publish_frame_world` even when the subscriber count is zero.
Meridian's `enabled()` lets the hot loop guard the *construction*:

```cpp
if (tele_->enabled("map/registered")) {
  PointCloudView v = build_registered_view(scan, traj);   // each point via T(t_i), only if wanted
  tele_->cloud("map/registered", v, Frame::Map, scan.stamp);
}
```

---

## 4. The `meridian_msgs` debug message set

`meridian_msgs` is the *only* place ROS message definitions live for telemetry. The
core never sees them; the `RosTelemetrySink` (§10) converts `meridian_debug` value
types into these. Definitions (`.msg`):

### 4.1 `Telemetry.msg` — the scalar/vector channel

```
# meridian_msgs/msg/Telemetry.msg
builtin_interfaces/Time stamp        # converted from meridian::Timestamp (int64 ns) in ONE place
string   key                         # e.g. "frontend/lidar/n_inlier"
float64[] values                     # 1 element for scalar(), N for vec()
string   axis_order                  # "" for scalar; e.g. "tx,ty,tz,rx,ry,rz" for vec
string   unit                        # documented unit, e.g. "count","m","rad","ratio"
```

One topic `/meridian/telemetry` carries *all* scalars and vectors, keyed by `key`.
This is deliberately a **single multiplexed topic** (not one topic per scalar): it
keeps the ROS graph small, lets a plotter subscribe once and filter by `key`, and
mirrors how FAST-LIO's globals were "all in one place" — but now typed, stamped,
and on the wire. (A power user who wants `rqt_plot` on one scalar can run the
provided `telemetry_split` relay node that fans `/meridian/telemetry` into
per-key `std_msgs/Float64` topics.)

### 4.2 `StageTiming.msg` — per-stage timing

```
# meridian_msgs/msg/StageTiming.msg
builtin_interfaces/Time stamp
string   stage                       # "preprocess","frontend.window_solve","frontend.visual",...
float64  ms                          # this invocation's wall time
float64  ms_avg                      # running average (wrapper-maintained EWMA)
float64  ms_max                      # running max since reset
uint64   count                       # invocations since reset
```

`/meridian/stage_timing`. The wrapper maintains the EWMA and max so the *core* only
emits a single `ms` per call (it stays stateless about timing history — FAST-LIO
kept `aver_time_*` accumulators inline, `laserMapping.cpp:991-1009`; we move that
bookkeeping to the sink).

### 4.3 `Event.msg` — structured events

```
# meridian_msgs/msg/Event.msg
builtin_interfaces/Time stamp
uint8    level                       # 0 Trace .. 4 Error (matches meridian::Level)
string   tag                         # "frontend/window_restart","place/loop_accepted",...
string   message                     # human-readable detail (structured key=value)
```

`/meridian/events`. Events ALSO go to the log (§9) so they appear in both a queryable
topic and the text log.

### 4.4 Reused standard messages

| Telemetry call | ROS 2 message | Topic |
|---|---|---|
| `cloud(...)` | `sensor_msgs/PointCloud2` | per key (§5) |
| `pose(...)` | `nav_msgs/Odometry` (+ TF) | per key (§5) |
| `marker(...)` | `visualization_msgs/MarkerArray` | `/meridian/markers` |
| `image(...)` | `sensor_msgs/Image` | per key (§5) |
| path aggregation | `nav_msgs/Path` | `/meridian/path` |

> **Axis-order discipline.** `vec()` *always* carries `axis_order`. The
> observability vector is published as `axis_order="tx,ty,tz,rx,ry,rz"`
> (translation-first, the `ObservabilityReport` order, spec 01 §3.4); the
> covariance diagonal extracted from a `KeyframePacket.constraint_cov` block is
> published as `axis_order="rx,ry,rz,tx,ty,tz"` (rotation-first, spec 00 §6.4).
> A plotter therefore never has to *guess* the order — FAST-LIO's block-swap bug
> (`laserMapping.cpp:597-606`) is impossible because the order is in the message.

---

## 5. The complete debug-channel catalogue (what every key publishes)

This is the authoritative, exhaustive list. It supersedes and details spec 00
Appendix C. Columns: **key** (the `const char*` the core passes), **call**, **ROS
topic/type**, **default on?**, **FAST-LIO origin / improvement**. "Default on" =
published when `debug.level >= info` and the relevant `publish_*` flag is true;
all are individually toggleable at runtime (§11).

The L2 catalogue is organised by *measurement stream* — `frontend/lidar/*`,
`frontend/visual/*`, `frontend/imu/*`, `frontend/gnss/*` — plus the shared
trajectory/solver/state channels. This mirrors the system: the CT front-end fuses
four streams into one B-spline trajectory, and the debug surface exposes each
stream's contribution and health independently so you can tell *which modality* is
failing, not just *that* the estimate is.

### 5.1 Front-end (L2) — the heart of estimator debugging

**Trajectory & solver (the CT window):**

| key | call | topic / type | default | FAST-LIO origin → improvement |
|---|---|---|---|---|
| `frontend/spline_knots` | `pose`×K → markers | `/meridian/markers` (Points+LineStrip) | **on** | **NEW. None in FAST-LIO** (discrete-time). The active B-spline control points $c_k$ (spec 00 §7.5) drawn as a strip, so the operator *sees the trajectory the front-end is currently optimising*. |
| `frontend/window_box` | `marker` | `/meridian/markers` (Cube) | on | The sliding-window working-region AABB — analogue of FAST-LIO's `LocalMap_Points` box (`laserMapping.cpp:229`), here spanning the active knot set. |
| `frontend/window_span_s` | `scalar` | `/meridian/telemetry` (`s`) | on | NEW. Time span of the active knot window (the optimisation horizon). |
| `frontend/ct/n_cp` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Control points placed on the most recent outer segment by the adaptive-knot-density rule (spec 04 §5.5): `1` in low-excitation motion, climbing toward `n_cp_max` under aggressive rotation/acceleration. A jump tracks where the spline added degrees of freedom to follow the motion. |
| `frontend/iter_count` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Window NLLS (Ceres) iterations to convergence. Spikes = hard scene. |
| `frontend/dx_norm` | `scalar` | `/meridian/telemetry` (`-`) | on | NEW. $\lVert\delta x\rVert$ of the last iterate over the windowed state (convergence proof). |
| `frontend/cost_total` | `scalar` | `/meridian/telemetry` (`-`) | on | NEW. Total windowed cost after the solve (sum over all residual streams). |
| `frontend/solve_ms` | `scalar` | `/meridian/telemetry` (`ms`) | on | NEW. Wall time of the windowed solve on the live (wall-clock) path, the quantity the deadline controller bounds. Plotted against the LiDAR period it shows headroom. |
| `frontend/deadline_hit` | `scalar` | `/meridian/telemetry` (`ratio`) | on | NEW. On the live wall-clock path only, the fraction (sliding-count EWMA) of window solves that returned early because they hit `solver.time_limit_ms` rather than converging — i.e. ran the deadline-bounded fallback after `min_iters`. The determinism path is fixed-iteration and never sets this. A sustained non-zero value is the runtime real-time-pressure signal that the post-hoc p99 gate (spec 10 §5.5) only sees after the run. |
| `frontend/observability` | `vec` | `/meridian/telemetry` (6, `ratio`) + `/meridian/markers` | **on** | **NEW. None in FAST-LIO** (only binary `flg_EKF_inited` `:898`). The 6 per-axis scores $s\in[0,1]^6$ (spec 01 §3.4) from the windowed Hessian; drives back-end noise inflation. Rendered as the observability hexagon (§7.1). |
| `frontend/cov_diag` | `vec` | `/meridian/telemetry` (6) | on | The 6 diagonal entries of the marginal keyframe-pose covariance, `axis_order` stated. Replaces FAST-LIO's hand-packed `pose.covariance` `:597-606`. |
| `frontend/spline/path_sample` | `pose`×N → Path | `/meridian/path` `nav_msgs/Path` | on (`debug.publish_path`) | NEW. The **solved B-spline itself** sampled at `debug.path_sample_hz` (default 30 Hz) and aggregated by the wrapper's `PathAggregator` (§10) into one `nav_msgs/Path`, ring-capped at `debug.path_max_poses` and republished at `debug.path_publish_hz`. Unlike FAST-LIO's `/path` `:622` (keyframe-rate, discrete poses) this shows the *continuous trajectory the spline actually solved*, exposing inter-sweep jitter invisible at keyframe rate. Samples near the window's leading edge are still refined by later sweeps, so small kinks vs. the final spline are expected (same property as `odom/body`). |
| `frontend/resid/<fam>/{rms,mean_abs,n_rows}` | `scalar` | `/meridian/telemetry` | on, sampled | NEW. Per-residual-family statistics of the converged window problem (raw whitened residuals, loss off, so families are comparable): families `lidar, imu, bias, anchor, motionreg, visual, gnss, prior`. Run ~1-in-5 sweeps + keyframe sweeps; every sweep when the `solver` debug group is on. |
| `frontend/resid/<fam>/cost_frac` | `scalar` | `/meridian/telemetry` (`ratio`) | group `solver` | NEW. The family's share of the total (squared, unweighted-by-loss) window cost — which stream dominates the solve. |

**LiDAR stream (direct point-to-plane at true point time):**

| key | call | topic / type | default | FAST-LIO origin → improvement |
|---|---|---|---|---|
| `map/registered` | `cloud` | `/meridian/cloud_registered` `PointCloud2` | on, rate-limited | `/cloud_registered` `:849`, `publish_frame_world` `:478`. Improvement: each point is placed via $T(t_i)$ at its **true sample time** (CT registration, spec 00 §7.5), and the transform is gated by `enabled()` so it is **not computed** when nobody subscribes (FAST-LIO always builds it). |
| `frontend/lidar/inliers` | `cloud` | `/meridian/cloud_effective` `PointCloud2` | **on** | `/cloud_effected` `:853`, `publish_effect_world` `:551`. **Strict upgrade: FAST-LIO comments the call out at `:983`.** The points that actually contributed a point-to-plane residual this window; Meridian ships it on. |
| `frontend/lidar/n_inlier` | `scalar` | `/meridian/telemetry` (`count`) | on | From `effct_feat_num` `:695` (was a global). Now plottable; a drop precedes divergence. |
| `frontend/lidar/n_input` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Downsampled input point count, so `n_inlier / n_input` = LiDAR inlier ratio. |
| `frontend/lidar/inlier_ratio` | `scalar` | `/meridian/telemetry` (`ratio`) | on | NEW. The single best one-number health gauge of LiDAR registration. |
| `frontend/lidar/res_mean` | `scalar` | `/meridian/telemetry` (`m`) | on | From `res_mean_last` `:715` (was a global). Mean point-to-plane residual. |
| `frontend/lidar/res_max` | `scalar` | `/meridian/telemetry` (`m`) | on | NEW. Tail residual — catches a few wild correspondences a mean hides. |
| `frontend/lidar/n_factors_kept` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Point-to-plane residuals actually built after the bounded-factor cap (normal-stratified subsample, spec 04 §3.1). With `n_factors_dropped` it shows how hard the cap is biting; a high drop fraction with a falling `inlier_ratio` is the bounded-solve warning. |
| `frontend/lidar/n_factors_dropped` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Inlier hits discarded by the factor cap this step (`kept + dropped` = inlier count; zero when the inlier set fit the budget). |

**Deep-debug groups (config-seeded key-prefix wildcards, default off — §11).** Each
group below is one wildcard entry (`frontend/assoc/*`, `frontend/solver/*`,
`frontend/deskew/*`, `frontend/spline/*`, `frontend/map/*`) seeded into the sink's
gate table from `debug.<group>.{enable,max_hz}` and flippable live through
`SetDebugKey` with the same wildcard. In the core, each group's entire computation is
hoisted behind **one** `enabled()` probe per sweep, so an off group costs one hash
lookup and builds nothing (§12). `map_health` defaults **on** (cheap counters); the
other four default **off**.

*`assoc` — LiDAR association quality (the registration-accuracy instrument):*

| key | call | unit | what it shows |
|---|---|---|---|
| `frontend/assoc/n_attempted`, `n_matched` | `scalar` | count | the funnel between `n_input` and `n_inlier`: points whose time the spline covers, and those that passed the plane gates. |
| `frontend/assoc/reject_nn`, `reject_dist`, `reject_plane`, `reject_score` | `scalar` | count | per-reason rejections (too few neighbours / farthest-NN gate / non-planar fit / range-aware score gate). `attempted = matched + reject_nn + reject_dist + reject_plane`; `matched = accepted + reject_score`. |
| `frontend/assoc/res_p95` | `scalar` | m | tail of the post-solve point-to-plane misfit \|n·p_w + d\| over the final hit set. |
| `frontend/assoc/res_hist` | `vec`(16) | count | fixed-bin \|r\| histogram over [0, 2·`plane_thresh`] (overflow in the last bin); the residual-vs-time heatmap source — distribution fattening precedes an RMS move. |
| `frontend/assoc/nn_dist_mean`, `nn_dist_p95` | `scalar` | m | farthest-fit-neighbour distance of accepted plane fits (match locality). |
| `frontend/assoc/plane_rms_mean`, `plane_rms_p95` | `scalar` | m | RMS point-to-fitted-plane distance of the 5-NN fits (map surface quality). |
| `frontend/assoc/strata_kept` | `vec`(`normal_strata`) | count | kept factors per plane-normal stratum (`+x,-x,+y,-y,+z,-z,obl` at the default 7) — the directional balance of the constraint set after the cap. |
| `frontend/assoc/outliers` | `cloud` | — | `/meridian/cloud_outliers`: scan points that attempted association but produced no hit, at their solved poses (heavy, rate-limited). |

*`solver` — window-solve internals:*

| key | call | unit | what it shows |
|---|---|---|---|
| `frontend/solver/iter_cost`, `iter_step`, `iter_grad`, `iter_tr_radius` | `vec`(N) | — | per-LM-iteration cost / step norm / gradient norm / trust-region radius, concatenated across the outer re-association schedule. |
| `frontend/solver/cost_initial` | `scalar` | — | round-0 entry cost (the IMU-seeded problem); with `frontend/cost_total` it gives the per-sweep cost reduction. |
| `frontend/solver/termination` | `scalar` | enum | Ceres `TerminationType` of the final round (0 convergence, 1 no-convergence, 2 failure). |
| (side effect) | | | the `frontend/resid/*` probe runs **every** sweep and emits `cost_frac` (≈1–3 ms post-solve `Problem::Evaluate`). |

*`deskew` — continuous-time placement quality:*

| key | call | unit | what it shows |
|---|---|---|---|
| `frontend/deskew/corr_mean_m`, `corr_p95_m`, `corr_max_m` | `scalar` | m | per-point CT correction ‖T(t_i)·p − T(t_end)·p‖ over the downsampled sweep — how much motion compensation is actually doing. |
| `frontend/deskew/sweep_trans_m`, `sweep_rot_deg` | `scalar` | m, deg | body motion span across the sweep, T(t_begin)→T(t_end). |
| `frontend/deskew/pre` | `cloud` | — | `/meridian/cloud_deskew_pre`: the sweep frozen at the end pose (heavy); pairs with `map/registered` as the deskew before/after view. |

*`spline` — trajectory kinematics profiles:*

| key | call | unit | what it shows |
|---|---|---|---|
| `frontend/spline/vel`, `acc`, `omega` | `vec`(3) | m/s, m/s², rad/s | analytic spline derivatives at t_end (`x,y,z` world for vel/acc; `wx,wy,wz` body for omega). |
| `frontend/spline/acc_rms_span`, `omega_rms_span` | `scalar` | m/s², rad/s | RMS magnitude over ~100 Hz samples across the solved span. |
| `frontend/spline/jerk_rms` | `scalar` | m/s³ | forward-difference jerk RMS over the span samples — the vibration / knot-cadence-aliasing instrument (gait frequency beating the knot grid shows here first). |

*`map_health` — local-map growth (default on):*

| key | call | unit | what it shows |
|---|---|---|---|
| `frontend/map/size` | `scalar` | count | resident ikd-Tree point count. |
| `frontend/map/n_inserted` | `scalar` | count | points offered to the tree this sweep (growth rate). |
| `frontend/map/insert_rejected` | `scalar` | count | world points discarded by the insert range guard (a non-zero value flags a pathological trajectory warp). |

**Visual stream (FAST-LIVO2-style sparse-direct photometric, LiDAR-depth):**

| key | call | topic / type | default | FAST-LIO origin → improvement |
|---|---|---|---|---|
| `frontend/visual/patches` | `image` | `/meridian/visual_patches` `Image` | on, rate-limited | **NEW. None in FAST-LIO** (LiDAR-only). The camera frame with the tracked sparse-direct patches overlaid, **coloured by photometric residual** and labelled by pyramid level (§7.5). The single most useful visual-stream debug view. |
| `frontend/visual/n_tracked` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Patches successfully tracked into this frame (warped through the warp/level gate). A drop = visual degradation (low texture, motion blur, exposure). |
| `frontend/visual/map_points` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Live visual-map population (active patch count). The funnel gauge: read with `n_candidates`/`n_tracked`/`n_converged` it localises a tracking dropout to promotion (map starved), candidacy gating, or the warp/NCC stages. Emitted only when a visual map is present. |
| `frontend/visual/n_candidates` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Candidate map points considered for this frame (before the warp/NCC gates). The funnel entry count that pairs with `map_points` and `n_tracked` to separate "nothing to track" from "tracking failed." |
| `frontend/visual/n_converged` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Patches that contributed a converged photometric residual to the solve. |
| `frontend/visual/res_mean` | `scalar` | `/meridian/telemetry` (`intensity`) | on | NEW. Mean photometric (intensity) residual after exposure compensation. |
| `frontend/visual/exposure_gain` | `vec` | `/meridian/telemetry` (2) | on | NEW. The estimated affine brightness `[a, b]` (exposure/gain compensation, spec 01 §4.3). Drift here explains a photometric residual climb. |
| `frontend/visual/depth_source` | `cloud` | `/meridian/visual_depth` `PointCloud2` | off | NEW. The LiDAR points projected into the image that supplied per-patch depth — shows where the visual stream has geometric support. |

**IMU stream (derivative residual against the spline) & estimated state:**

| key | call | topic / type | default | FAST-LIO origin → improvement |
|---|---|---|---|---|
| `frontend/imu/res_acc` | `scalar` | `/meridian/telemetry` (`m/s^2`) | on | NEW. Mean residual between measured specific force and the spline's analytic acceleration at the IMU stamps (the CT IMU-derivative residual, spec 00 §7.5). |
| `frontend/imu/res_gyr` | `scalar` | `/meridian/telemetry` (`rad/s`) | on | NEW. Mean residual between measured angular rate and the spline's analytic angular velocity. |
| `frontend/bias_acc` | `vec` | `/meridian/telemetry` (3, `m/s^2`) | on | From `dump_lio_state_to_log` `:150`. Estimated accel bias (bias estimation lives in L2, spec 00 §6.3). |
| `frontend/bias_gyr` | `vec` | `/meridian/telemetry` (3, `rad/s`) | on | From `dump_lio_state_to_log` `:150`. Estimated gyro bias. |
| `frontend/grav_norm` | `scalar` | `/meridian/telemetry` (`m/s^2`) | on | NEW as topic. Should hold ≈9.81; drift signals a gravity/extrinsic problem. |
| `frontend/vel` | `vec` | `/meridian/telemetry` (3, `m/s`) | off | NEW. Body velocity from the spline's analytic derivative at "now". |

**GNSS stream:**

| key | call | topic / type | default | FAST-LIO origin → improvement |
|---|---|---|---|---|
| `frontend/gnss/anchor` | `pose` | `/meridian/gnss` `Odometry` | on | **NEW. None in FAST-LIO** (LiDAR-only). The GNSS-derived position anchor projected into the estimation datum, as fused into the window. |
| `frontend/gnss/innovation_m` | `scalar` | `/meridian/telemetry` (`m`) | on | NEW. The GNSS position innovation — the norm of the fix minus the trajectory position interpolated to the fix time — i.e. the quantity the $k\sigma$ acceptance gate is applied to. Large + persistent ⇒ datum/extrinsic or GNSS-quality problem. (Supersedes the older `frontend/gnss/res` name; the residual *fed to the solve* and the *gating innovation* are the same quantity computed at the fix-interpolated pose.) |
| `frontend/gnss/accept_rate` | `scalar` | `/meridian/telemetry` (`ratio`) | on | NEW. Fraction of GNSS fixes accepted into the window over a sliding count (accepted / (accepted+rejected)). A collapse to zero with a healthy fix type is the signature of a datum-init heading error or a stale extrinsic. |
| `frontend/gnss/yaw_uncertainty_deg` | `scalar` | `/meridian/telemetry` (`deg`) | on | NEW. The $1\sigma$ heading uncertainty of the `T_map_enu` datum, read off the yaw block of the datum-alignment Hessian. Watched at datum lock: a high value is the gate that keeps a poorly-excited baseline from baking in a wrong heading. |
| `frontend/gnss/datum` | `pose` | `/meridian/gnss_datum` `Odometry` | on | NEW. The estimated `T_map_enu` datum transform (map frame → local-ENU). Published once at lock and on every drift-redistribution update so a global-frame jump is correlated with a datum change. |
| `frontend/gnss/datum_lock` | `event` | `/meridian/events` (Info) | on | NEW. The datum-initialisation lock event, carrying the locked `T_map_enu`, the `yaw_uncertainty_deg` at lock, and the baseline length that satisfied the `min_baseline` and velocity-excitation pre-gates. A second `datum_lock` after a re-acquisition gap carries the drift-redistribution decision. |
| `frontend/gnss/fix` | `event` | `/meridian/events` (Info) | on | NEW. Fix-type transitions (`SPP`/`DGPS`/`RTK_Float`/`RTK_Fixed`, spec 01 §4.4) so a residual change is correlated with fix quality. |
| `frontend/gnss/reject` | `event` | `/meridian/events` (Warn) | on | NEW. A fix rejected by the innovation gate, the fix-quality floor, or the re-acquisition persistence check, carrying the `reason` (`gate`/`quality`/`reacq_persist`/`spacing`) and the innovation in metres. The false-anchor guard made visible, the GNSS analogue of `place/loop_rejected_pcm`. |

**Live output, lifecycle & recovery:**

| key | call | topic / type | default | FAST-LIO origin → improvement |
|---|---|---|---|---|
| `odom/body` | `pose` | `/meridian/odom` `Odometry` + TF `odom→base_link` | on | `/Odometry` `:857`. Live high-rate pose from the spline at "now". Covariance NOT smuggled in (it is `cov_diag`); the `Odometry.pose.covariance` carries the full 6×6 in the **stated** order, redundantly, for tools that expect it there. |
| `frontend/init_done` | `event` | `/meridian/events` (Info) | on | NEW. The IMU-init → CT-tracking transition (cold-start, spec 00 §7.2; FAST-LIO's `flg_EKF_inited` flip, `:898`) as a visible event carrying estimated gravity/biases and the first knot time. |
| `frontend/window_restart` | `event` | `/meridian/events` (Warn) | on | **NEW. Recovery made visible.** FAST-LIO bails silently (`ekfom_data.valid=false` when `effct_feat_num<1`, `:708-712`). Meridian's window-restart fallback (spec 00 §7.4) is an explicit event with a `reason`; the *next* keyframe carries `ImuPreintegration` (GTSAM `CombinedImuFactor`, mutually exclusive with the relative factor). |

### 5.2 Calibration (online extrinsic refinement, default on)

The rig is **one LiDAR + one IMU + one camera + GNSS** (spec 00 §2). There is a
single LiDAR→IMU extrinsic and a single camera→IMU extrinsic; the keys are not
per-sensor-name'd because there is exactly one of each.

| key | call | topic / type | default | origin → improvement |
|---|---|---|---|---|
| `calib/T_imu_lidar` | `pose` | `/meridian/extrinsic` `Odometry` | on | FAST-LIO logs only `offset_R_L_I/offset_T_L_I` to `fout_*`. **Meridian publishes the live online-refined LiDAR→IMU extrinsic** so a pose jump can be correlated with extrinsic motion. |
| `calib/T_imu_cam` | `pose` | `/meridian/extrinsic` `Odometry` | on | NEW. The live online-refined camera→IMU extrinsic (the one frozen into each `KeyframePacket.T_body_cam`, spec 01 §6.1). |
| `calib/version` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. The `CalibrationSet` snapshot version (spec 01 §5.3) so you can correlate a pose jump or relinearisation with a calibration update. |

### 5.3 Back-end (L3)

| key | call | topic / type | default | origin → improvement |
|---|---|---|---|---|
| `backend/chi2` | `scalar` | `/meridian/telemetry` | on | **NEW. None in FAST-LIO** (no back-end). Total graph $\chi^2$ after `optimize()`. |
| `backend/n_factors` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Factor count — graph growth. |
| `backend/n_keyframes` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. |
| `backend/relinearize` | `event` | `/meridian/events` | on | NEW. iSAM2 relinearization event (and its wall time via `timing("backend.optimize")`). |
| `backend/n_moved` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Keyframes whose pose changed in the last `GraphUpdate` (spec 01 §7.4) — i.e. how much the world just shifted. |
| `backend/trajectory` | `pose`×N → Path | `/meridian/path_optimized` `Path` | on | The corrected `map`-frame trajectory. FAST-LIO `/path` `:859` is **odom-only** (no back-end); this is the globally consistent one. |
| `backend/loop_correction_norm` | `scalar` | `/meridian/telemetry` (`m`) | on | NEW. Magnitude of the rigid jump when a loop snaps — the "how big was the correction" number. |
| `backend/optimize_lag` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Queued items batched into the last `optimize()` because the solve cadence is decoupled from keyframe insertion — i.e. how many keyframes/constraints one optimise consumed. A persistently rising value means the back-end is not keeping its `optimize_interval_ms` budget. |
| `backend/fallback_count` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Cumulative count of iSAM2 last-resort recoveries — a rebuild of the estimator from the retained factors and last linearisation point after an indefinite/ill-conditioned update. It MUST stay flat in nominal operation; any increment is paired with a `backend/fallback` event and is a field-survival signal, not a routine one. |
| `backend/fallback` | `event` | `/meridian/events` (Warn) | on | NEW. The iSAM2 rebuild-from-factors recovery made visible, carrying the trigger (`indefinite`/`relinearize_fail`) and the factor/keyframe count rebuilt. |

### 5.4 Place recognition (L5)

| key | call | topic / type | default | origin → improvement |
|---|---|---|---|---|
| `place/loop_edge` | `marker` | `/meridian/markers` (LineList) | on | **NEW.** Line between matched keyframe centroids, **coloured by GICP fitness** (green=tight, red=loose). |
| `place/loop_accepted` | `event` | `/meridian/events` (Info) | on | NEW. Loop passed GICP + PCM (spec 00 §5, L5). Carries fitness + PCM score. |
| `place/loop_rejected_pcm` | `event` | `/meridian/events` (Warn) | on | NEW. Candidate rejected by Pairwise-Consistency-Maximisation — the false-loop guard made visible. |
| `place/candidates` | `scalar` | `/meridian/telemetry` (`count`) | off | NEW. ScanContext++ candidate count per query. |
| `place/gicp_fitness` | `scalar` | `/meridian/telemetry` (`m`) | off | NEW. Verification residual of the accepted loop. |

### 5.5 Map (L4 — nvblox, GPU)

The map is **nvblox**, GPU-only: TSDF + colour + Marching-Cubes mesh, the single
backend (spec 00 §9.5). These channels observe that one GPU map.

| key | call | topic / type | default | origin → improvement |
|---|---|---|---|---|
| `map/registered` | (see §5.1) | | | the CT-registered current scan integrated into the TSDF. |
| `map/mesh` | (custom) | `/meridian/mesh` `visualization_msgs/Marker` (TriangleList) | on demand | NEW. The colourised nvblox Marching-Cubes triangle mesh (L6 surface, scope endpoint). |
| `map/region_rebuild` | `event` | `/meridian/events` | on | **NEW.** A loop correction triggered a GPU clear-and-rebuild of a TSDF region from retained clouds at corrected poses (spec 01 §7.5). |
| `map/dirty_region` | `marker` | `/meridian/markers` (Cube) | on | NEW. The AABB being rebuilt, drawn so the operator sees the map "healing." |
| `map/tsdf_blocks` | `scalar` | `/meridian/telemetry` (`count`) | off | NEW. Allocated nvblox TSDF block count (GPU memory growth). |
| `map/integrate_lag` | `scalar` | `/meridian/telemetry` (`ms`) | off | NEW. Queue lag of `Q_map` (spec 00 §11.1) — back-pressure visibility. |

### 5.6 Pipeline / queues (cross-cutting health)

| key | call | topic / type | default | origin → improvement |
|---|---|---|---|---|
| `pipeline/q_meas_depth` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. `Q_meas` occupancy; a rising value = front-end falling behind. |
| `pipeline/q_kf_depth` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. `Q_kf` occupancy (lossless; back-pressure, spec 00 §11.2). |
| `pipeline/q_map_depth` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. `Q_map` occupancy — the GPU-map ingest queue (spec 00 §11.1). Distinct from `map/integrate_lag` (which is the *time* lag): this is the *depth* gauge that pairs with the other `q_*_depth` keys so all three pipeline queues are watched the same way. |
| `pipeline/q_meas_dropped` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Cumulative lossy drops on `Q_meas` overload (spec 00 §11.2) — the dropped count reported by the bounded-queue primitive — must stay flat. `Q_meas` interleaves sweeps and live IMU; the drop policy is type-aware and only ever evicts an `ImuSample`, never a `PreprocessedGroup`, so a sweep is never lost behind a benign IMU drop. |
| `pipeline/q_meas_dropped_imu` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Subset of `q_meas_dropped` that evicted a live `ImuSample` (degrades only between-sweep live-state propagation; the next sweep's solve is unaffected). |
| `pipeline/q_meas_dropped_sweep` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Subset of `q_meas_dropped` that evicted a whole `PreprocessedGroup` — only reachable when the queue holds nothing but sweeps. Paired with an `Error` event; must stay zero. A dropped sweep leaves the front-end's last-solved time behind the next group's `t_begin`; the front-end's gap guard then classifies the hole. If the next group's IMU still reaches back to within two IMU periods of the last solve the gap is **bridged** — the seed integrates straight across it and the sweep solves normally (an Info `frontend/sweep_gap_bridged`). Only an **unbridgeable** gap (the IMU itself starts after the last solve) triggers `reseedAfterGap`: the anchor is predicted across the hole on constant velocity, the window is rebuilt, and the post-gap sweep is inserted into the map *without solving* — surfaced as a `frontend/window_restart` (Warn). |
| `pipeline/scan_to_odom_ms` | `scalar` | `/meridian/telemetry` (`ms`) | on | NEW. End-to-end latency scan-in → odom-out, the real-time SLA gauge. |

---

## 6. Per-stage timing

### 6.1 The stage set

Timing is produced *only* through `ScopedTimer`/`MERIDIAN_SCOPED_TIME` (§3). The
canonical stage keys (each emitted on `/meridian/stage_timing`) follow the CT
LIVO+GNSS pipeline:

```
preprocess              L1 filter + validity + image pyramid build
frontend.lidar_assoc    point-to-plane association @ true point time   (cf FAST-LIO aver_time_match, :991)
frontend.visual         sparse-direct photometric residual assembly     (NEW; no FAST-LIO analogue)
frontend.window_solve   the sliding-window Ceres NLLS over control pts  (cf FAST-LIO aver_time_solve, :993)
frontend.total          whole scan callback                             (cf FAST-LIO aver_time_icp,   :1009)
backend.optimize        iSAM2 update
place.query             SC++/STD/BTC candidate search
place.verify            GICP + PCM
map.integrate           nvblox TSDF fusion of one keyframe (GPU)
map.deintegrate         nvblox region clear-and-rebuild    (GPU)
mesh.extract            nvblox Marching Cubes (on demand, GPU)
```

This is a superset of FAST-LIO's `aver_time_match / aver_time_solve /
aver_time_icp / aver_time_const_H_time` (`laserMapping.cpp:991-1009`), reorganised
per *pipeline stage* (LiDAR association, visual assembly, window solve) rather than
per *EKF internal*, with the CT-and-vision stages added, and **live on a topic**
rather than printed at shutdown (`:1042-1044`). nvblox stages report host-side wall
time around the GPU launch + sync.

### 6.2 Worked example — reading a timing stream

A healthy 10 Hz LiDAR + 20 Hz camera run should show roughly (Jetson Orin):

```
preprocess              ~3 ms      frontend.lidar_assoc   ~6 ms
frontend.visual         ~5 ms      frontend.window_solve  ~9 ms
frontend.total          ~22 ms     (< 100 ms budget  ⇒  real-time OK)
backend.optimize        ~30 ms     (off the hot thread — T3, spec 00 §11)
map.integrate           ~8 ms      (GPU)
```

If `frontend.total` creeps toward 100 ms while `frontend.lidar_assoc`
dominates, the registration voxel map is too dense → raise
`preprocess.voxel_surf_m`. If `frontend.window_solve` dominates, the window is too
long → lower `spline.window_knots`. If `frontend.visual` spikes while
`frontend/visual/n_tracked` is *low*, the cost is in re-acquisition, not tracking.
If `backend.optimize` spikes to 200 ms but `pipeline/scan_to_odom_ms` stays ~22 ms,
the thread split is doing its job (the back-end stall did *not* reach odometry) —
exactly the structural property spec 00 §11.2 promises and FAST-LIO's single
`main()` loop cannot offer.

### 6.3 The EWMA/max bookkeeping lives in the sink

The core emits one `ms` per call; the `RosTelemetrySink` keeps the per-stage
`{count, ms_avg (EWMA, α=0.1), ms_max}` and fills the `StageTiming` message. Reset
via `DebugControl::reset_timing()` (§11). This keeps the core stateless and means a
`NullSink` does no timing bookkeeping at all.

---

## 7. rviz markers: geometric introspection

The wrapper turns `marker()` calls into one `visualization_msgs/MarkerArray` on
`/meridian/markers`, and `image()` calls into per-key `sensor_msgs/Image`. Each marker
carries a stable `(ns, id)` so it *updates in place* rather than piling up (a common
rviz foot-gun). The standard overlay set:

### 7.1 Observability hexagon (the flagship)

Six bars/arrows, one per DoF in order `[tx,ty,tz,rx,ry,rz]`, anchored at the body
origin, **length ∝ score $s_i$** and **colour green→red as $s_i: 1\to 0$**. This
is the single most important operator view: in a long corridor the forward-axis
bar collapses and reddens *before* the estimate drifts, so the operator sees
degeneracy coming. There is **no FAST-LIO analogue** (it has only binary
`flg_EKF_inited`). Drawn from `frontend/observability` (the windowed Hessian's
per-axis conditioning).

```
        rz                     legend:  full bar (green) = well observed
         |                              short red bar    = degenerate axis
   ry ___|___ rx
        /|\
   tz  / | \  ty            corridor example: tx (forward) bar is short+red,
      /  |  \                                 ty/tz/rx/ry/rz green → "weak forward"
    tx (forward)
```

If `ObservabilityReport.eigvecs` is present (non-axis-aligned degeneracy, spec 01
§3.4), the bars are drawn along the eigenvector directions instead of the body
axes, with a thin text label of the tunnel angle.

### 7.2 CT sliding-window view

`frontend/spline_knots` + `frontend/window_box`: the active B-spline control points
$c_k$ (spec 00 §7.5) drawn as `Points` + a `LineStrip`, plus the AABB of the active
window. This lets the operator *see the trajectory the front-end is currently
optimising* and watch the window slide — the CT analogue of FAST-LIO's
`LocalMap_Points` box (`laserMapping.cpp:229`), now showing the optimisation horizon
rather than a discrete local map.

### 7.3 Loop edges

`place/loop_edge`: a `LineList` connecting matched keyframe centroids, coloured by
GICP fitness. Accepted loops solid; PCM-rejected candidates dashed/translucent so
the operator can see *what was considered and thrown out*.

### 7.4 Dirty-region AABB

`map/dirty_region`: the nvblox TSDF region being cleared-and-rebuilt after a loop
correction (spec 01 §7.5). It blinks for `lifetime_ns ≈ 1 s` so the operator sees
the map healing without the marker lingering.

### 7.5 Sparse-direct photometric patch overlay

`frontend/visual/patches`: the raw camera image (the front-end's `CameraFrame`,
spec 01 §4.3) with the tracked sparse-direct patches drawn on it, each coloured by
its photometric residual (green=converged-tight, red=large-residual) and labelled
by pyramid level. This is the visual-stream flagship — the FAST-LIVO2-style
introspection that lets an operator *see* the camera losing texture, blurring, or
mis-exposing before `frontend/visual/n_tracked` collapses. Built only when enabled
(§3, §12); the wrapper rasterises the `ImageOverlay` into a `sensor_msgs/Image`.

### 7.6 Confidence overlay (L6)

The colourised nvblox mesh tinted per-vertex by confidence (TSDF weight × inverse
pose covariance), via `Marker.colors`. This is the operator-facing surface (spec 00
§3, L6) and the scope endpoint. A toggle switches between true colour and
confidence colour.

---

## 8. rviz layout & display config

`meridian_ros/rviz/meridian_debug.rviz` ships a curated layout (the structured
replacement for FAST-LIO's ad-hoc `rviz_cfg/*.rviz`, which wire
`/cloud_registered`, `/Laser_map`, `/Odometry`, `/path` by hand). Display groups,
all under fixed frame `map`:

```
[Global]   Fixed Frame: map        Background: dark
[TF]       map → odom → base_link → {imu_link, os_lidar, cam_link, gnss_link}
[Estimate]
  ├─ Odometry        /meridian/odom               (axes, keep 1)
  ├─ Path (odom)     /meridian/path               (white)
  ├─ Path (optim)    /meridian/path_optimized     (cyan)   ← global, post loop
  └─ Spline window   /meridian/markers ns=frontend/spline_knots   ← CT control points
[Clouds]
  ├─ Registered      /meridian/cloud_registered   (intensity colormap)
  ├─ LiDAR inliers ★ /meridian/cloud_effective    (flat green, size 3)   ← ON by default
  └─ Body            /meridian/cloud_body         (off)
[Visual]
  └─ Patch overlay ★ /meridian/visual_patches     (Image; residual-coloured patches)  ← ON
[Map]
  ├─ Mesh            /meridian/mesh               (triangle list, nvblox)
  └─ Dirty region    /meridian/markers ns=map/dirty_region
[Health]
  ├─ Observability   /meridian/markers ns=frontend/observability  (hexagon)
  ├─ Window box      /meridian/markers ns=frontend/window_box
  └─ Loop edges      /meridian/markers ns=place/loop_edge
[Calibration]
  └─ Extrinsics      /meridian/extrinsic          (T_imu_lidar, T_imu_cam axes)
```

A companion **`rqt` perspective** (`meridian_ros/rqt/meridian.perspective`) preloads
`rqt_plot` on the high-value scalars filtered from `/meridian/telemetry`:
`frontend/lidar/inlier_ratio`, `frontend/lidar/res_mean`, `frontend/visual/n_tracked`,
`frontend/visual/res_mean`, `frontend/gnss/innovation_m`, `pipeline/scan_to_odom_ms`,
`backend/chi2`; an `Image` view on `/meridian/visual_patches`; and a `StageTiming` bar
view. This is the "single glance" dashboard FAST-LIO never had (its equivalents
were a CSV read post-run).

---

## 9. Structured logging

### 9.1 The `LogSink` and the macros

The core logs through `meridian::log` macros that forward to a `LogSink` (spec 00
§10.3). **The core never calls `RCLCPP_INFO`/`ROS_WARN`.** The wrapper binds the
sink to `rclcpp` logging; tests bind a buffer; the offline tool binds stdout+file.

```cpp
// include/meridian/debug/log.hpp   (meridian_debug — NO ros)
namespace meridian {
class LogSink {
public:
  virtual ~LogSink() = default;
  virtual void log(Level, const char* module, std::string_view kvline, Timestamp) = 0;
  virtual bool enabled(Level, const char* module) const = 0;   // cheap gate
};
LogSink* log_sink();              // process-global, set once at startup
void     set_log_sink(LogSink*);

#define MERIDIAN_LOG(lvl, mod, ...)                                            \
  do { auto* _s = ::meridian::log_sink();                                      \
       if (_s && _s->enabled(lvl, mod))                                     \
         _s->log(lvl, mod, ::meridian::fmt_kv(__VA_ARGS__), ::meridian::now()); } while (0)

#define MERIDIAN_TRACE(mod, ...) MERIDIAN_LOG(::meridian::Level::Trace, mod, __VA_ARGS__)
#define MERIDIAN_DEBUG(mod, ...) MERIDIAN_LOG(::meridian::Level::Debug, mod, __VA_ARGS__)
#define MERIDIAN_INFO(mod, ...)  MERIDIAN_LOG(::meridian::Level::Info,  mod, __VA_ARGS__)
#define MERIDIAN_WARN(mod, ...)  MERIDIAN_LOG(::meridian::Level::Warn,  mod, __VA_ARGS__)
#define MERIDIAN_ERROR(mod, ...) MERIDIAN_LOG(::meridian::Level::Error, mod, __VA_ARGS__)
```

### 9.2 Levels and what belongs at each

| Level | Use | Example |
|---|---|---|
| `Trace` | per-point / per-iteration; off unless hunting a specific bug | each window-solve iterate's $\delta x$ |
| `Debug` | per-scan internals | `mod=frontend stamp=… lidar_inl=812 lidar_res=0.021 vis_trk=46 iters=4` |
| `Info` | lifecycle / milestones | init done, first keyframe, GNSS fix-type change, loop accepted |
| `Warn` | recoverable degradation | window restart, PCM rejection, queue drop |
| `Error` | the estimator cannot proceed correctly | calibration load failure, no IMU, no GPU |

### 9.3 Structured = key=value

Every log line is `key=value` tokens with a module tag and `Timestamp` always
present, so logs are greppable and parseable (`mod=place stamp=172… event=loop_accepted from=412 to=87 fitness=0.08 pcm=0.97`). This replaces FAST-LIO's `printf`/`fout_*` free-text (`laserMapping.cpp:150, 835-838, 991-1013`). The default log level is `Config.debug.level` (spec 00 §8.2), runtime-overridable per module (§11).

### 9.4 Events are logged *and* published

`TelemetrySink::event()` and `MERIDIAN_*` are unified at the wrapper: an `event()`
call produces both an `/meridian/events` message (§4.3) *and* a log line at the same
level. One call, two surfaces — the operator's rviz/rqt and the developer's
`ros2 bag`/text log stay consistent.

### 9.5 The always-on log ring buffer and flush-on-Error

The text/structured log a developer reads after the fact is not a forensic store on
its own: the line that explains a failure is usually written *seconds before* the
failure is detected, and on a long field run those seconds have already scrolled out
of any console and may never have reached a file the operator can retrieve. The
estimator therefore keeps an **always-on, bounded, in-process log ring buffer** —
running in every posture including production, at zero allocation in steady state and
with no dependence on whether any sink is bound to disk.

- **Structure.** A fixed-capacity ring of the most recent log records (level, module,
  the `key=value` line, `Timestamp`) sized by `debug.log_ring_capacity` (default
  `4096` records) and capped by `debug.log_ring_bytes` (default `1 MiB`); whichever
  bound is hit first evicts oldest-first. The records are stored as already-formatted
  lines so a flush is a copy, never a re-render. The ring lives in `meridian_debug`
  behind the `LogSink` boundary (a `RingLogSink` decorator that the bound terminal
  sink wraps), so it is ROS-agnostic and present in replay exactly as in the live
  node.
- **Admission floor.** The ring admits every record at or above `debug.log_ring_level`
  (default `Debug`) regardless of the active surfacing level, so the buffer retains
  the per-scan internals (§9.2 `Debug`) that the console may be suppressing. The
  surfacing level (`Config.debug.level`, §11) still governs what is *emitted live*;
  the ring is a separate, lower floor whose only cost is the bounded copy. `Trace`
  is admitted only when `debug.log_ring_level=trace` is set, because per-point lines
  would churn the ring.
- **Flush-on-Error.** Any record at `Error` level — and any `event()` at
  `Level::Error` (§9.4), since events are logged — triggers an **automatic forensic
  flush**: the ring's current contents *and* the last `debug.forensic_window_s`
  (default `10 s`) of the telemetry ring (§11.2) are written to a self-contained
  forensic file, with the same payload and on-disk shape the manual
  `SnapshotForensics` service produces (§11.2). The flush is **rate-limited** (one
  per `debug.forensic_min_interval_s`, default `30 s`) so an `Error` storm cannot
  itself starve the estimator with disk I/O, and it runs **off the hot path** — the
  triggering call only sets a flag and notifies the forensics thread; the write
  happens on that thread, never in the producing layer. A flush emits one `Info`
  event (`forensic/flushed` carrying the path and the trigger tag) so the operator
  knows a case file exists.
- **Cost.** When no `Error` ever fires, the ring is a bounded circular write and
  nothing is serialised — the always-on guarantee costs the steady-state copy of the
  admitted lines only, inside the §12 budget. With a `NullSink`/`error`-only posture
  the ring still runs (it is what makes a production incident reproducible), but its
  admission floor may be raised to `Warn` via `debug.log_ring_level` to shrink it
  further.

The result: when the estimator hits an `Error`, the developer gets the run-up to it
for free, paired with the telemetry of the same window, without having had foresight
to turn anything on.

### 9.6 Config provenance: fragments in, resolved snapshot out

A telemetry number is meaningless without the configuration that produced it, and the
single most common reproducibility failure is a run whose effective config nobody can
reconstruct — a default that changed, a fragment that was overridden, a key the
operator forgot they set. The debug/provenance contract closes this from both ends.

- **Per-package YAML fragments under a root index (authoring side).** The
  configuration source is split into one fragment per package
  (`config/<package>.yaml` — e.g. `frontend.yaml`, `backend.yaml`, `map.yaml`,
  `gnss.yaml`, `debug.yaml`) referenced from a single root index
  (`config/meridian.yaml`) that lists the fragments and any profile overlays. The
  fragments are merged into the one typed, validated `Config` tree (spec 00 §8.2) at
  load — the typed model is unchanged and remains the source of truth; the split is
  an authoring and review convenience (a package owner edits one file; a diff is
  scoped) and does **not** introduce an untyped or per-package runtime config object.
  The same merged tree is produced whether loaded from the index, from a single
  flattened YAML, or from ROS parameters, so there is no drift between load paths
  (§13.2).
- **`Config::dump(run_dir)` of the fully-resolved tree (provenance side).** At startup
  the pipeline writes the **fully-resolved, post-validation, post-overlay** config —
  every key with its effective value, defaults included — to `run_dir/config.resolved.yaml`
  and records its content hash in the `RunManifest` (spec 00 §6). This is the
  *resolved* tree, not the input fragments: it captures the value the estimator
  actually ran with, after every default and overlay is applied, so a run is
  reproducible from the manifest alone. The dump is the canonical config artefact the
  evaluation harness reads (spec 10 §1.2 `manifest.json` / `run_dir/`), and the same
  resolved snapshot is bundled into every forensic case file (§9.5, §11.2) so an
  incident carries its own configuration.

The `RunManifest` serializer (the canonical owner of `git SHA`, config hash, calib
hash, dataset id, input content hash; consumed by spec 10 §7.1) records the
`config.resolved.yaml` hash as the config-hash field, making "what config produced
this number" a one-line lookup and a regression-baseline invariant (spec 10 §7.2's
`manifest.config_hash` check).

---

## 10. The `RosTelemetrySink` adapter (wrapper-side)

`meridian_ros/src/debug/ros_telemetry_sink.{hpp,cpp}` implements `TelemetrySink` by
owning the publishers and the conversion code. It is the *only* place a telemetry
key becomes a ROS topic. Sketch:

```cpp
class RosTelemetrySink : public meridian::TelemetrySink {
public:
  RosTelemetrySink(rclcpp::Node* node, const DebugConfig& cfg);

  bool enabled(const char* key) const override {       // §12 gate
    return gate_.enabled(key);                          // flag ∧ subscriber-aware ∧ rate-limit-aware
  }
  void scalar(const char* key, double v, Timestamp t) override {
    if (!gate_.pass(key, t)) return;                    // token-bucket rate limit
    meridian_msgs::msg::Telemetry m; m.stamp = to_ros(t);
    m.key = key; m.values = {v}; m.unit = unit_of(key);
    pub_telemetry_->publish(m);
  }
  void cloud(const char* key, const PointCloudView& v, Frame f, Timestamp t) override {
    if (!gate_.pass(key, t)) return;
    auto& pub = pub_cloud_for(key);                     // lazily-created per-key PC2 publisher
    pub->publish(to_pointcloud2(v, frame_name(f), t));  // the ONLY copy, only when enabled
  }
  void image(const char* key, const ImageOverlay& ov, Timestamp t) override {
    if (!gate_.pass(key, t)) return;
    auto& pub = pub_image_for(key);                     // lazily-created per-key Image publisher
    pub->publish(rasterise(ov, t));                     // draw patches over base, only when enabled
  }
  // pose → Odometry + TF; marker → MarkerArray; timing → StageTiming(EWMA/max); event → Event + log
  ...
private:
  Gate gate_;   // §12: per-key enable flag + token bucket + (optional) subscriber count check
};
```

Key properties:

- **`to_ros(Timestamp)` is the single int64-ns → `rclcpp::Time` conversion** in the
  whole system (spec 00 §6.2). The core never touches `rclcpp::Time`.
- **Per-key publishers created lazily** on first enabled use, with QoS from config:
  clouds/images use *best-effort, depth 5* (drop under load, never block the
  estimator — FAST-LIO uses a depth-100000 reliable queue, `laserMapping.cpp:849`,
  which can back-pressure; we do not); the multiplexed telemetry topic uses
  *reliable, depth 512* (the per-sweep burst is ~60 messages with every debug group
  on, so the history must hold several full bursts or the earliest-emitted keys of
  each burst are silently dropped toward a slow subscriber — measured as chaotic
  1–98 % per-key capture at depth 50), stage timing *reliable, depth 256*, events
  *reliable, depth 50*; TF uses the standard TF QoS. A burst-rate consumer
  (`ros2 topic echo` capture) must likewise deepen its subscription queue
  (`--qos-depth`, as the run scripts do) — the default depth 10 overflows on every
  sweep.
- **`unit_of(key)`** is a static table so units are consistent and self-documenting.

The wrapper also runs a `PathAggregator` that turns the stream of `odom/body` and
`backend/trajectory` poses into `nav_msgs/Path` (FAST-LIO's `publish_path`
throttled `jjj % 10`, `laserMapping.cpp:622`; we make the decimation a config knob,
not a literal).

---

## 11. Runtime control: toggling debug cheaply

Because `Config` is immutable after start (spec 00 §8.3), debug is toggled through
a `DebugControl` surface the wrapper exposes as ROS 2 services + a parameter
callback. The core is unaffected — it always calls the sink; the *sink's gate*
decides.

```cpp
// meridian_msgs/srv  (wrapper-side services)
SetDebugKey.srv    : string key, bool enable, float64 max_hz   → bool ok
SetLogLevel.srv    : string module, uint8 level                → bool ok
SetTelemetryRate.srv: float64 default_hz                       → bool ok
ResetTiming.srv    : (empty)                                   → bool ok
SnapshotForensics.srv: float64 seconds                         → string path   # §11.2
```

Examples:

```bash
# Turn on the body cloud and the visual depth-source cloud for a forensic look, no restart:
ros2 service call /meridian/set_debug_key meridian_msgs/SetDebugKey "{key: 'body/scan', enable: true, max_hz: 5.0}"
ros2 service call /meridian/set_debug_key meridian_msgs/SetDebugKey "{key: 'frontend/visual/depth_source', enable: true, max_hz: 5.0}"
# Drop the front-end to debug-level logging while leaving the back-end at info:
ros2 service call /meridian/set_log_level meridian_msgs/SetLogLevel "{module: 'frontend', level: 1}"
```

A `*` wildcard key (`"frontend/visual/*"`) toggles a whole stream. The gate state is
queryable (`/meridian/debug_state` latched topic) so a UI can show what is on.

**Config-seeded wildcard groups are the flag taxonomy.** The deep-debug groups of
§5.1 (`assoc`, `solver`, `deskew`, `spline`, `map_health`) introduce **no new
mechanism**: each is exactly one wildcard entry in this same gate table, seeded at
construction from `Config.debug.<group>.{enable, max_hz}` (immutable defaults, spec
00 §8.3) and flipped live with the same `SetDebugKey` call
(`{key: 'frontend/assoc/*', enable: true, max_hz: 10}`). A group `max_hz` of `0`
keeps each key's class default (scalars at `debug.telemetry_rate_hz`, heavy payloads
at 2 Hz). The replay `FileSink` seeds the identical prefix gate from the same
`DebugConfig` (no rate limit — a replay wants every sample), preserving §13
replay==live for the recorded key set. `debug.publish_path` plus
`path_sample_hz`/`path_publish_hz`/`path_max_poses` configure the
`frontend/spline/path_sample` → `/meridian/path` aggregation the same way.

### 11.1 Production posture vs. forensic posture

- **Production:** `debug.level=info`; heavy payloads (`map/registered`, `body/scan`,
  `frontend/lidar/inliers`, `frontend/visual/patches`) rate-limited to 1–2 Hz or
  off; scalars/events on (cheap); a black-box `CsvTelemetrySink` recording all
  scalars/events to disk via `MultiSink` for post-incident analysis.
- **Forensic:** flip clouds/patch-overlays on, raise rate, drop a module to
  `trace` — live, no restart. This is the operational reason the toggle exists: you
  cannot restart a SLAM run in the field to debug it.

### 11.2 Forensic snapshot

`SnapshotForensics(seconds)` dumps the last *N* seconds of an **always-on, bounded
telemetry ring** the `RecordingSink` keeps (poses, scalars, inlier/patch overlays,
events) to a self-contained file the offline `replay` tool can load — so a field
anomaly becomes a desk-reproducible case (§13). The telemetry ring is sized by
`debug.forensic_window_s` (default `10 s`) and runs in every posture, the heavy-payload
peer of the log ring buffer (§9.5); the two share one forensic-file format so a
manual `SnapshotForensics` call and an automatic flush-on-Error produce
interchangeable case files (the log lines from the §9.5 ring, the telemetry/overlays
from this ring, and the resolved-config snapshot of §9.6, bundled with the
`RunManifest` provenance).

The same payload is emitted on two triggers — the explicit service call and the
automatic flush-on-Error (§9.5) — so the forensic store is not contingent on an
operator having reacted in time. Both paths share the `debug.forensic_min_interval_s`
rate limit and run on the dedicated forensics thread, never on a producing layer's
hot path. The dump is gated only by writable storage; on a full disk it logs one
`Error` and disables further flushes (the debug layer's own failure mode, §15.1).

---

## 12. Cost discipline & the zero-cost-when-off guarantee

The principle (spec 00 §10.6): **introspection is always wired, cheap when off,
rich when on.** Mechanisms:

1. **`NullSink` default.** With no debug sink bound, every
   `scalar/vec/pose/image/event` is one virtual call to an empty body; in release
   the optimizer inlines the `s_==nullptr`-guarded macros to nothing. `enabled()`
   returns `false`, so the core skips payload construction entirely.

2. **`enabled(key)` before expensive construction.** The hot loop guards cloud
   building (the registered-cloud transform via $T(t_i)$, the inlier-cloud copy)
   and patch-overlay rasterisation behind `tele_->enabled(key)`. This is the fix
   for FAST-LIO building `publish_frame_world` unconditionally
   (`laserMapping.cpp:478`). Cost of a disabled key: one bool return.

3. **Token-bucket rate limiting in the gate.** Even when enabled, each key has a
   `max_hz` (default from `debug.telemetry_rate_hz`, spec 00 §8.2). `gate_.pass()`
   drops the call if the bucket is empty. Heavy clouds/patch-overlays default to
   1–2 Hz; scalars to 10 Hz. The limiter lives in the *sink* (wrapper), so the
   core's call is always made and always cheap.

4. **Subscriber-aware gating (optional).** The gate may consult
   `get_subscription_count()` so a cloud/image with zero subscribers is neither
   built nor serialised — strictly better than FAST-LIO, which builds regardless.

5. **View, not copy.** `cloud()` takes a `PointCloudView` and `image()` takes an
   `ImageOverlay` with a borrowed `base` span (§3); the only copy is the
   `PointCloud2`/`Image` serialisation, made *after* the gate passes.

6. **Timing is two clock reads.** `ScopedTimer` cost is negligible and disappears
   with a null sink (the `if (s_)` guard).

7. **No telemetry on the back-pressure path.** Telemetry that itself touches a
   lossless queue (`Q_kf`, `Q_map`) is forbidden from blocking; `pipeline/q_*`
   gauges are sampled, never synchronised.

> **Guarantee.** A release build with `debug.level=error` and all `publish_*=false`
> runs the estimator with no measurable telemetry overhead (target < 0.1 % wall
> time, verified by the timing harness comparing `frontend.total` with sink =
> `NullSink` vs. `RecordingSink` discarded). This is the property that lets
> debugging be first-class *without* a production cost.

---

## 13. Replay / bag path equivalence

This is the property that makes the whole debug subsystem trustworthy: **a bag
replay produces byte-identical telemetry to the live run that recorded it**, so a
field issue is reproduced and debugged at a desk.

### 13.1 Why it holds (structural)

The core is a deterministic function of a timestamped measurement stream (spec 00
§1.1). The *only* inputs are `ISensorSource` callbacks; the *only* outputs are
`TelemetrySink` calls and `KeyframeSink` packets. Live and replay differ **only**
in the `ISensorSource` implementation (`OusterSource`/`CameraSource`/`GnssSource`
vs `BagReplaySource`, spec 01 §7.1) and the bound sink. Therefore:

```
   live:    {Ouster,Cam,Gnss}Source ─┐                          ┌─→ RosTelemetrySink → rviz
   replay:  BagReplaySource          ┼─→ identical MeridianPipeline ┼─→ CsvTelemetrySink  → pandas
                                      ┘  (same Config, same code) └─→ RecordingSink     → asserts
```

### 13.2 What the replay tool guarantees

`meridian_tools/replay`:

- Loads the **same `Config` YAML** the live node used (spec 00 §8.1) — the YAML and
  ROS-param loaders fill the *same* struct, so there is no drift.
- Drives the pipeline in **`--single-thread` deterministic mode** (spec 00 §11.2)
  with any OpenMP in the residual assembly disabled/fixed-scheduled and nvblox GPU
  reductions in their deterministic variant, so the windowed cost is
  order-deterministic and the run is **bit-reproducible** (a test-only guarantee,
  spec 00 §11.2).
- Feeds measurements **in recorded timestamp order** at either wall-clock-scaled
  or as-fast-as-possible rate; telemetry stamps are the *measurement* times, not
  wall times, so plots align across runs.
- Binds a `MultiSink{RosTelemetrySink, CsvTelemetrySink}` so you can watch in rviz
  *and* diff the CSV against a golden run.

### 13.3 The regression use

A CI replay test binds a `RecordingSink`, runs a short Newer College segment
(`DATASET.md`), and asserts on telemetry: e.g. `frontend/lidar/inlier_ratio`
median > 0.6, `frontend/visual/n_tracked` median > 30 (once the visual stage
runs on these bags), no `frontend/window_restart`
event, final `backend/chi2` within tolerance of golden, ATE from
`corrected_trajectory()` under threshold. **This is debugging-as-testing** — the
same signals an operator watches are the signals CI gates on. FAST-LIO cannot do
this (its signals are globals + CSV).

### 13.4 The pinned determinism hash

The determinism guarantee — two replays of the same `(bag, config, calib, git SHA)`
produce byte-identical artefacts (spec 10 §1.1, §7.4) — is enforced by a single
content hash over the recorded telemetry. To make that hash well-defined and stable
it is pinned, not left to the order rows happened to be written:

- **Canonical record set and ordering.** The hash is computed over the
  `ParquetTelemetrySink` record set (§2.5) **sorted by `(stamp_ns, key, seq)`** — the
  measurement timestamp first, the channel key second, and the monotone per-write
  `seq` counter as the final, total tie-break for records that share a stamp and key.
  This canonical sort removes any dependence on the order calls reached the sink, so
  the hash is invariant to write interleaving and to harmless reorderings; only a real
  change in *what was computed* moves it.
- **Hashed fields.** Each record contributes its `(stamp_ns, key, kind, values,
  axis_order, unit, level, tag, message)` to a streaming SHA-256 in the canonical
  order; `seq` orders but is **not** itself hashed (it is a within-run artefact, not a
  property of the computation). Floating-point `values` are hashed by their exact IEEE
  bit pattern — the determinism mode is bit-reproducible (spec 00 §11.2), so this is
  an equality, not a tolerance.
- **Single-thread replay only.** The hash is a property of the
  **single-threaded determinism/replay path** alone (spec 00 §11.2). The production
  pipeline is multi-threaded and makes no bit-reproducibility claim; the hash is never
  computed on, nor asserted against, a live multi-threaded run. The harness writes the
  digest as `determinism.trajectory_hash` (spec 10 §4.5) and CI runs the replay twice
  on the smoke clip and asserts equality (spec 10 §7.4, the M0 gate).

This is what makes the regression baselines (spec 10 §7) meaningful: a hash change at
a fixed SHA/config/calib is, by construction, a nondeterminism bug — a stray thread, a
wall-clock read, an unordered-container iteration, or a non-deterministic GPU reduction
not run in its deterministic variant.

---

## 14. Per-module debug-hook checklist

Each layer's own spec (02–05) defines its math; this checklist is the *normative
minimum* debug surface every implementation MUST emit. (Pulled together here so a
reviewer can verify a module is "debuggable" before merge.)

- **L1 preprocess:** `timing("preprocess")`, `scalar("frontend/lidar/n_input")`,
  reject-reason `event` on a scan dropped for missing per-point time (spec 01 §4.2).
- **L2 front-end (the priority surface — CT LIVO+GNSS):**
  - *trajectory/solver:* `pose("frontend/spline_knots")`,
    `marker("frontend/window_box")`,
    `scalar("frontend/iter_count"|"dx_norm"|"cost_total"|"window_span_s"|"solve_ms"|"deadline_hit"|"frontend/ct/n_cp")`,
    `vec("frontend/observability"|"cov_diag")`, `pose("odom/body")`,
    `timing("frontend.window_solve"|"frontend.total")`.
  - *LiDAR stream:* `cloud("frontend/lidar/inliers")`,
    `scalar("frontend/lidar/n_inlier"|"n_input"|"inlier_ratio"|"res_mean"|"res_max"|"n_factors_kept"|"n_factors_dropped")`,
    `timing("frontend.lidar_assoc")`.
  - *visual stream:* `image("frontend/visual/patches")`,
    `scalar("frontend/visual/n_tracked"|"n_converged"|"res_mean")`,
    `vec("frontend/visual/exposure_gain")`, `timing("frontend.visual")`.
  - *IMU/state:* `scalar("frontend/imu/res_acc"|"res_gyr"|"grav_norm")`,
    `vec("frontend/bias_acc"|"bias_gyr")`.
  - *GNSS stream:* `pose("frontend/gnss/anchor"|"frontend/gnss/datum")`,
    `scalar("frontend/gnss/innovation_m"|"accept_rate"|"yaw_uncertainty_deg")`,
    `event("frontend/gnss/fix"|"datum_lock"|"reject")`.
  - *lifecycle/calib:* `event("frontend/init_done"|"frontend/window_restart")`,
    `pose("calib/T_imu_lidar"|"calib/T_imu_cam")`, `scalar("calib/version")`.

  **This is the user-priority module; its debug surface is the richest by design.**
- **L3 back-end:** `scalar("backend/chi2"|"n_factors"|"n_keyframes"|"n_moved"
  |"loop_correction_norm"|"optimize_lag"|"fallback_count")`,
  `event("backend/relinearize"|"backend/fallback")`,
  `pose`→`backend/trajectory`, `timing("backend.optimize")`.
- **L4 map (nvblox):** `event("map/region_rebuild")`, `marker("map/dirty_region")`,
  `timing("map.integrate"|"map.deintegrate"|"mesh.extract")`,
  `scalar("map/tsdf_blocks"|"map/integrate_lag")`.
- **L5 place:** `marker("place/loop_edge")`, `event("place/loop_accepted"
  |"place/loop_rejected_pcm")`, `timing("place.query"|"place.verify")`.
- **Pipeline:** `scalar("pipeline/q_meas_depth"|"q_kf_depth"|"q_map_depth"
  |"q_meas_dropped"|"scan_to_odom_ms")`.

A module that does not emit its checklist set fails the `debug-surface` CI lint
(a grep over the module's sources for the required keys), making debuggability a
*merge gate*, not a hope.

---

## 15. Failure modes & how the debug layer surfaces them

| Failure | What the operator sees | Underlying signal |
|---|---|---|
| Corridor / degeneracy | observability hexagon's forward bar shortens + reddens *before* drift | `frontend/observability` low on `tx` |
| LiDAR losing lock | `frontend/lidar/inlier_ratio` falls, `res_mean`/`res_max` climb | §5.1 LiDAR scalars |
| Visual losing lock (texture/blur/exposure) | patch overlay reddens + thins, `frontend/visual/n_tracked` drops, `res_mean` climbs | §5.1 visual scalars + `/meridian/visual_patches` |
| Exposure/gain drift | `frontend/visual/exposure_gain` wanders while scene is static | `frontend/visual/exposure_gain` |
| Window not converging | `iter_count` hits the cap, `dx_norm`/`cost_total` plateau high | §5.1 solver scalars |
| Window restart / divergence | yellow `window_restart` event in rviz/log with reason; odom origin rebases; next KF is `ImuPreintegration` | `frontend/window_restart` (spec 00 §7.4) |
| IMU init failure | no `init_done` event; estimator stays in cold-start | `frontend/init_done` absent |
| Bad IMU / extrinsic | `frontend/imu/res_acc`/`res_gyr` climb; `grav_norm` ≠ 9.81 | §5.1 IMU scalars |
| GNSS quality problem | `frontend/gnss/innovation_m` large + persistent; correlates with `gnss/fix` type | `frontend/gnss/innovation_m`, `frontend/gnss/fix` |
| GNSS datum / heading error | `frontend/gnss/accept_rate` collapses with a healthy fix type; `yaw_uncertainty_deg` was high at `datum_lock` | `frontend/gnss/accept_rate`, `frontend/gnss/yaw_uncertainty_deg`, `frontend/gnss/datum_lock` |
| False loop closure | `loop_rejected_pcm` event; no map jump | `place/loop_rejected_pcm` |
| Loop snap (good) | path_optimized jumps, dirty-region AABB blinks, `loop_correction_norm` reported | `backend/loop_correction_norm`, `map/region_rebuild` |
| Front-end falling behind | `q_meas_depth` rising, `q_meas_dropped` incrementing, `scan_to_odom_ms` growing | §5.6 |
| Back-end stalling odometry | would show as `scan_to_odom_ms` spike *correlated* with `backend.optimize` — if it does NOT, the thread split is healthy | §6.2 |
| Calibration drift | `calib/T_imu_lidar` / `calib/T_imu_cam` axes wander; `grav_norm` ≠ 9.81 | §5.2 |
| nvblox memory growth | `map/tsdf_blocks` climbs without bound | `map/tsdf_blocks` |
| Telemetry starving the estimator | `frontend.total` rises when a cloud/patch key is enabled | the gate/rate-limit (§12) is the fix; a misconfigured `max_hz` is the cause |

### 15.1 The debug layer's own failure modes

- **Sink slower than producer:** never blocks the core — best-effort QoS drops, and
  `MultiSink` short-circuits gated-off keys. A slow CSV sink on a full disk logs one
  `Error` and disables itself.
- **Forensic flush cannot keep up / disk full:** the flush-on-Error path (§9.5) is
  rate-limited (`debug.forensic_min_interval_s`) and runs on its own thread, so an
  `Error` storm degrades to dropped flushes, never to a stalled producer; the always-on
  ring keeps overwriting regardless. A full disk disables further flushes after one
  `Error` (as for any disk sink) — the ring itself stays live for the next reachable
  flush.
- **Marker id collisions:** avoided by the stable `(ns, id)` contract (§7); a
  module reusing an id across semantically different markers is a lint failure.
- **Clock skew in replay:** telemetry uses *measurement* stamps, so a replay at a
  different wall rate still produces time-aligned plots (§13.2).

---

## 16. Returnable schema

The canonical, returnable artefacts of this spec.

### 16.1 `TelemetrySink` (core-side, ROS-agnostic)

```cpp
enum class Level { Trace, Debug, Info, Warn, Error };

struct Marker {
  enum class Type { Points, LineList, LineStrip, Arrow, Cube, Sphere, Text, Hexagon };
  Type type; Frame frame; std::string ns; std::int32_t id;
  std::vector<Eigen::Vector3f> points;
  std::array<float,4> color; std::vector<std::array<float,4>> colors;
  float scale; std::string text; Duration lifetime_ns;
};

struct ImageOverlay {                                    // sparse-direct patch overlay
  Frame frame; int width, height;
  std::span<const std::uint8_t> base; enum class Encoding { Mono8, RGB8 } encoding;
  struct Patch { Eigen::Vector2f uv; float residual; float depth; int level; };
  std::vector<Patch> patches;
};

class TelemetrySink {
public:
  virtual ~TelemetrySink() = default;
  virtual bool enabled(const char* key) const = 0;                       // cheap gate
  virtual void scalar(const char* key, double v, Timestamp t) = 0;
  virtual void vec   (const char* key, const Eigen::Ref<const Eigen::VectorXd>&,
                      Timestamp t, const char* axis_order = nullptr) = 0;
  virtual void cloud (const char* key, const PointCloudView&, Frame, Timestamp) = 0;
  virtual void pose  (const char* key, const Pose&,           Frame, Timestamp) = 0;
  virtual void marker(const Marker&,                                  Timestamp) = 0;
  virtual void image (const char* key, const ImageOverlay&,           Timestamp) = 0;
  virtual void timing(const char* stage, double ms, Timestamp t) = 0;
  virtual void event (Level, const char* tag, std::string_view, Timestamp t) = 0;
};

class ScopedTimer {                                       // RAII per-stage timer
public: ScopedTimer(TelemetrySink*, const char* stage, Timestamp);  ~ScopedTimer();
};

class LogSink {
public:
  virtual ~LogSink() = default;
  virtual void log(Level, const char* module, std::string_view kvline, Timestamp) = 0;
  virtual bool enabled(Level, const char* module) const = 0;
};

class IntrospectionHooks {                                // §2.4 white-box slot (parallel to TelemetrySink)
public:
  virtual ~IntrospectionHooks() = default;
  virtual bool subscribed(Hook) const = 0;                // cheap gate; NullHooks→false
  virtual void visit_graph (const gtsam::NonlinearFactorGraph&) = 0;   // const-ref, producer-thread, synchronous
  virtual void visit_isam2 (const gtsam::ISAM2&)                = 0;
  virtual void visit_update(const ISAM2ResultExt&)              = 0;
  virtual void visit_window(const SplineWindow&)                = 0;
  virtual void visit_hessian(const Eigen::MatrixXd&)            = 0;
};
```

### 16.2 Sink and hook implementations (selected by `Config.debug`)

```
NullSink             default; empty bodies; enabled()→false                 (meridian_debug)
RecordingSink        captures all calls into the always-on bounded ring     (meridian_debug; tests + forensics, §9.5/§11.2)
MultiSink            fan-out to ordered children, short-circuits             (meridian_debug)
CsvTelemetrySink     per-key CSV append                                      (meridian_tools)
ParquetTelemetrySink columnar, fixed schema, per-domain file; determinism    (meridian_tools; harness, §2.5)
                       record set, single-thread replay only
RosTelemetrySink     →topics/TF/markers/images; owns the one to_ros          (meridian_ros)
NullHooks            default IntrospectionHooks; subscribed()→false          (meridian_debug; live + replay-without-test)
```

### 16.3 `meridian_msgs` debug messages

```
Telemetry.msg   : Time stamp; string key; float64[] values; string axis_order; string unit
StageTiming.msg : Time stamp; string stage; float64 ms; float64 ms_avg; float64 ms_max; uint64 count
Event.msg       : Time stamp; uint8 level; string tag; string message
# reused: sensor_msgs/PointCloud2, sensor_msgs/Image, nav_msgs/Odometry, nav_msgs/Path, visualization_msgs/MarkerArray
SetDebugKey.srv     : string key, bool enable, float64 max_hz   → bool ok
SetLogLevel.srv     : string module, uint8 level                → bool ok
SetTelemetryRate.srv: float64 default_hz                        → bool ok
ResetTiming.srv     : ()                                        → bool ok
SnapshotForensics.srv: float64 seconds                          → string path
```

### 16.4 Debug-channel catalogue (key → topic), CT LIVO+GNSS + nvblox

```
core key                       call    → ROS topic / type            origin (laserMapping.cpp where applicable)
# --- L2 trajectory / solver (CT window) ---
frontend/spline_knots          pose×K  → /meridian/markers          Mark NEW (CT B-spline control points)
frontend/window_box            marker  → /meridian/markers          Mark cf LocalMap_Points box        :229
frontend/observability         vec(6)  → /meridian/telemetry+markers     NEW (none; only flg_EKF_inited :898)
frontend/cov_diag              vec(6)  → /meridian/telemetry        Tel  was packed pose.covariance    :597-606
frontend/iter_count|dx_norm|cost_total|window_span_s|solve_ms|deadline_hit  scalar → /meridian/telemetry  Tel  NEW (window NLLS + deadline ctrl)
frontend/ct/n_cp               scalar  → /meridian/telemetry        Tel  NEW (adaptive knot density)
odom/body                      pose    → /meridian/odom + TF        Odom /Odometry                     :857
# --- L2 LiDAR stream (point-to-plane @ true point time) ---
map/registered                 cloud   → /meridian/cloud_registered PC2  /cloud_registered  :849 / build :478 (via T(t_i))
frontend/lidar/inliers         cloud   → /meridian/cloud_effective  PC2  /cloud_effected :853 / :551 (UNCOMMENTED vs :983)
frontend/lidar/n_inlier        scalar  → /meridian/telemetry        Tel  effct_feat_num                :695  (NEW topic)
frontend/lidar/n_input|inlier_ratio|res_mean|res_max  scalar → /meridian/telemetry  Tel  res_mean_last :715 + NEW
frontend/lidar/n_factors_kept|n_factors_dropped       scalar → /meridian/telemetry  Tel  NEW (bounded-factor cap)
frontend/resid/<fam>/rms|mean_abs|n_rows|cost_frac    scalar → /meridian/telemetry  Tel  NEW (per-family probe; cost_frac under solver group)
# --- L2 deep-debug groups (config-seeded wildcards, default off; §5.1/§11) ---
frontend/assoc/n_attempted|n_matched|reject_nn|reject_dist|reject_plane|reject_score  scalar → /meridian/telemetry  Tel  NEW (assoc group)
frontend/assoc/res_p95|nn_dist_mean|nn_dist_p95|plane_rms_mean|plane_rms_p95          scalar → /meridian/telemetry  Tel  NEW (assoc group)
frontend/assoc/res_hist        vec(16) → /meridian/telemetry        Tel  NEW (|r| histogram, heatmap source)
frontend/assoc/strata_kept     vec(7)  → /meridian/telemetry        Tel  NEW (kept factors per normal stratum)
frontend/assoc/outliers        cloud   → /meridian/cloud_outliers   PC2  NEW (rejected points, world frame)
frontend/solver/iter_cost|iter_step|iter_grad|iter_tr_radius  vec(N) → /meridian/telemetry  Tel  NEW (per-LM-iteration trace)
frontend/solver/cost_initial|termination               scalar → /meridian/telemetry  Tel  NEW
frontend/deskew/corr_mean_m|corr_p95_m|corr_max_m|sweep_trans_m|sweep_rot_deg  scalar → /meridian/telemetry  Tel  NEW (CT placement)
frontend/deskew/pre            cloud   → /meridian/cloud_deskew_pre PC2  NEW (frozen sweep-end pose)
frontend/spline/vel|acc|omega  vec(3)  → /meridian/telemetry        Tel  NEW (analytic derivatives at t_end)
frontend/spline/acc_rms_span|omega_rms_span|jerk_rms   scalar → /meridian/telemetry  Tel  NEW (span-RMS / aliasing)
frontend/spline/path_sample    pose×N  → /meridian/path             Path NEW (solved spline @ path_sample_hz; cf /path :622/:859)
frontend/map/size|n_inserted|insert_rejected           scalar → /meridian/telemetry  Tel  NEW (map_health group, default on)
# --- L2 visual stream (FAST-LIVO2 sparse-direct photometric) ---
frontend/visual/patches        image   → /meridian/visual_patches   Img  NEW (patch overlay, residual-coloured)
frontend/visual/n_tracked|n_converged|res_mean        scalar → /meridian/telemetry  Tel  NEW
frontend/visual/exposure_gain  vec(2)  → /meridian/telemetry        Tel  NEW (affine brightness comp)
frontend/visual/depth_source   cloud   → /meridian/visual_depth     PC2  NEW (LiDAR depth support)
# --- L2 IMU / state ---
frontend/imu/res_acc|res_gyr   scalar  → /meridian/telemetry        Tel  NEW (CT IMU-derivative residual)
frontend/bias_acc|bias_gyr     vec(3)  → /meridian/telemetry        Tel  dump_lio_state_to_log         :150
frontend/grav_norm             scalar  → /meridian/telemetry        Tel  NEW topic
# --- L2 GNSS ---
frontend/gnss/anchor           pose    → /meridian/gnss             Odom NEW (GNSS fusion)
frontend/gnss/datum            pose    → /meridian/gnss_datum       Odom NEW (T_map_enu, lock + redistribute)
frontend/gnss/innovation_m|accept_rate|yaw_uncertainty_deg  scalar → /meridian/telemetry  Tel  NEW
frontend/gnss/fix|datum_lock   event   → /meridian/events           Evt  NEW (fix-type / datum lock)
frontend/gnss/reject           event   → /meridian/events           Evt  NEW (gate/quality/reacq reason)
# --- L2 lifecycle / calibration ---
frontend/init_done             event   → /meridian/events           Evt  NEW (cf flg_EKF_inited        :898)
frontend/window_restart        event   → /meridian/events           Evt  NEW (recovery visible; cf :708-712)
calib/T_imu_lidar              pose    → /meridian/extrinsic        Odom was fout_* offset_R/T_L_I
calib/T_imu_cam                pose    → /meridian/extrinsic        Odom NEW (camera extrinsic)
calib/version                  scalar  → /meridian/telemetry        Tel  NEW
# --- L3 back-end ---
backend/chi2|n_factors|n_keyframes|n_moved|loop_correction_norm|optimize_lag|fallback_count  scalar → /meridian/telemetry  Tel  NEW
backend/relinearize|fallback   event   → /meridian/events           Evt  NEW (relin / rebuild-from-factors)
backend/trajectory             path    → /meridian/path_optimized   Path global; cf odom-only /path    :859/:622
# --- L5 place ---
place/loop_edge                marker  → /meridian/markers          Mark NEW (coloured by GICP fitness)
place/loop_accepted|rejected   event   → /meridian/events           Evt  NEW
# --- L4 map (nvblox, GPU) ---
map/mesh                       custom  → /meridian/mesh             Mark NEW (nvblox Marching-Cubes mesh)
map/region_rebuild             event   → /meridian/events           Evt  NEW (loop-driven clear-and-rebuild)
map/dirty_region               marker  → /meridian/markers          Mark NEW
map/tsdf_blocks|integrate_lag  scalar  → /meridian/telemetry        Tel  NEW
# --- pipeline ---
pipeline/q_meas_depth|q_kf_depth|q_map_depth|q_meas_dropped  scalar → /meridian/telemetry  Tel  NEW (queue health)
pipeline/scan_to_odom_ms       scalar  → /meridian/telemetry        Tel  NEW (RT SLA)
timing(stage,ms)               timing  → /meridian/stage_timing     ST   aver_time_* (live, not CSV)   :991-1009/:1042-1044
```

### 16.5 Stage-timing keys

```
preprocess  frontend.lidar_assoc  frontend.visual  frontend.window_solve  frontend.total
backend.optimize  place.query  place.verify  map.integrate  map.deintegrate  mesh.extract
```

---

## Appendix R — SOTA reference grounding (non-normative)

This appendix is evidence, not contract: curated digests of the reference systems
this spec's design was validated against. Nothing here binds Meridian's behavior —
the normative sections above own the design. Each block names the reference checkout
it was verified against; the clones live in /home/user/slam-reference.

### R.1 FAST_LIO `laserMapping.cpp` — the introspection-bearing functions
*verified against FAST_LIO@7cc4175 (the file is 1055 lines: `main()` + ROS glue + EKF
measurement model + map mgmt + all publishers + timing/logging, ~80 file-scope globals)*

The `laserMapping.cpp:NNN` citations throughout this spec resolve here:

| line | symbol | role |
|---|---|---|
| 143 | `SigHandle` | SIGINT → exit flag |
| 150 | `dump_lio_state_to_log` | full state row → `pos_log.txt` (bias/state trace source, §5.1) |
| 229 | `LocalMap_Points` | moving local-map cube (the `window_box` analogue, §7.2) |
| 279 / 302 / 336 | `standard_pcl_cbk` / `livox_pcl_cbk` / `imu_cbk` | sensor callbacks |
| 368 | `sync_packages` | LiDAR+IMU bundling into `MeasureGroup` |
| 427 | `map_incremental` | ikd-Tree insertion |
| 478 / 532 | `publish_frame_world` / `_body` | `/cloud_registered[_body]` |
| 551 | `publish_effect_world` | `/cloud_effected` (effective/inlier points) |
| 567 | `publish_map` | `/Laser_map` |
| 589 | `publish_odometry` | `/Odometry` + 6×6 cov (block-swapped `:597-606`) + TF |
| 622 | `publish_path` | `/path`, throttled `jjj % 10` |
| 638 | `h_share_model` | measurement model; sets globals `effct_feat_num` `:695`, `res_mean_last` `:715` |
| 756 | `main` | param load, sub/pub setup, loop, timing, logging |

**Disabled-by-default debug views (sharp edge):** in `main` the two most diagnostic
publishers are commented out at the call site — `publish_effect_world` and
`publish_map` (`:983-984`) — and the map-flatten is gated `if(0)` (`:942`). Out of
the box you cannot see the effective points or the live map. Meridian ships the
inlier cloud (`frontend/lidar/inliers`) on by default (§5.1).

### R.2 Topic / rviz surface contrast: FAST_LIO vs FAST-LIVO2
*verified against FAST_LIO@7cc4175, FAST-LIVO2@0d2c034*

| aspect | FAST_LIO | FAST-LIVO2 (the richer surface to emulate) |
|---|---|---|
| node shape | monolithic `laserMapping.cpp` | 11-line `main.cpp` → `LIVMapper` class; libs `vio/lio/pre/imu_proc/laser_mapping` |
| residual normals | none | `pubNormal` MarkerArray, per-residual surface normals (`LIVMapper.cpp:201`) |
| map-uncertainty viz | none | voxel/plane cubes coloured by plane covariance (`voxel_map.cpp:788 pubVoxelMap`, `:837 pubSinglePlane`) |
| photometric overlay | n/a (LiDAR-only) | `/rgb_img` RGB overlay (`pubImage`, `LIVMapper.cpp:213`) |
| low-latency odom | none (LiDAR-rate only) | `/LIVO2/imu_propagate` high-rate IMU-propagated odom (`LIVMapper.cpp:214`) |
| rviz configs | one `loam_livox.rviz`; wires a `/MarkerArray` display nothing publishes | per-platform configs exercising image + normal/voxel/plane markers |

FAST-LIVO2 still prints its per-stage timing as an ANSI-coloured stdout table
(`LIVMapper.cpp:472`), not a topic — the UX Meridian copies into `StageTiming` (§6).

### R.3 Timing / logging instrumentation
*verified against FAST_LIO@7cc4175*

`omp_get_wtime()` snapshots + accumulators (`match_time`, `solve_time`,
`kdtree_*_time`, declared `:69-71`); 12 fixed global arrays `s_plot…s_plot11[MAXN]`
with `MAXN=720000` (`:65,70`); running averages `printf`'d when `runtime_pos_log`
(`:991-1009`) and CSV-dumped only at shutdown (`:1042-1051`). The block-swap that
Meridian's `axis_order` field eliminates: the 6×6 covariance is hand-packed into
`Odometry.pose.covariance` in `[pos|rot]` order at `:597-606`. Sharp edges: fixed
arrays (≈69 MB) overflow on long runs; flat text with no levels, timestamps, or
rotation; timing is read post-run, never live.

### R.4 Point-LIO engineering observations (source NOT in /home/user/slam-reference)
*digest only — recorded from a prior read of the Point-LIO tree, not re-verifiable
against the current clone set; treat line numbers as indicative*

- Split TUs (`parameters.cpp` for loading, `li_initialization.cpp` for init,
  `Estimator.cpp` for math) — the right "one concern per file" instinct — but glued
  by `extern` globals in `parameters.h` (one shared mutable namespace across files).
- Dual estimator mode (`use_imu_as_input` vs as measurement; propagation at IMU vs
  LiDAR frequency) forces global-flag branching throughout `laserMapping.cpp`,
  hurting readability.
- Debug surface narrower than FAST-LIVO2: `/cloud_effected` and a `Marker plane_pub`
  exist but are commented out; no live marker-based residual/plane visualisation.
