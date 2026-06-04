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
> library (the `TelemetrySink` bus, declared in `00_architecture.md` §10.1 and used
> by every layer via `setTelemetry`/`set_telemetry`), the `meridian_msgs` debug
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
> additionally anchored to the reference implementation
> `slam-reference/FAST_LIO/src/laserMapping.cpp` and its `rviz_cfg/`, cited as
> `laserMapping.cpp:NNN`. FAST-LIO's introspection has the *right signals* but is
> **entangled with the node, hard-coded, and partly disabled**; Meridian keeps the
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
the pipeline (`IFrontEnd::setTelemetry`, spec 00 §5.1; analogous `set_telemetry`
on every interface) and writes to it. The pointer is **never owned** by the layer;
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

} // namespace meridian
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
| `frontend/iter_count` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Window NLLS (Ceres) iterations to convergence. Spikes = hard scene. |
| `frontend/dx_norm` | `scalar` | `/meridian/telemetry` (`-`) | on | NEW. $\lVert\delta x\rVert$ of the last iterate over the windowed state (convergence proof). |
| `frontend/cost_total` | `scalar` | `/meridian/telemetry` (`-`) | on | NEW. Total windowed cost after the solve (sum over all residual streams). |
| `frontend/observability` | `vec` | `/meridian/telemetry` (6, `ratio`) + `/meridian/markers` | **on** | **NEW. None in FAST-LIO** (only binary `flg_EKF_inited` `:898`). The 6 per-axis scores $s\in[0,1]^6$ (spec 01 §3.4) from the windowed Hessian; drives back-end noise inflation. Rendered as the observability hexagon (§7.1). |
| `frontend/cov_diag` | `vec` | `/meridian/telemetry` (6) | on | The 6 diagonal entries of the marginal keyframe-pose covariance, `axis_order` stated. Replaces FAST-LIO's hand-packed `pose.covariance` `:597-606`. |

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

**Visual stream (FAST-LIVO2-style sparse-direct photometric, LiDAR-depth):**

| key | call | topic / type | default | FAST-LIO origin → improvement |
|---|---|---|---|---|
| `frontend/visual/patches` | `image` | `/meridian/visual_patches` `Image` | on, rate-limited | **NEW. None in FAST-LIO** (LiDAR-only). The camera frame with the tracked sparse-direct patches overlaid, **coloured by photometric residual** and labelled by pyramid level (§7.5). The single most useful visual-stream debug view. |
| `frontend/visual/n_tracked` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Patches successfully tracked into this frame. A drop = visual degradation (low texture, motion blur, exposure). |
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
| `frontend/gnss/res` | `scalar` | `/meridian/telemetry` (`m`) | on | NEW. GNSS position residual against the trajectory — large + persistent ⇒ datum/extrinsic or GNSS-quality problem. |
| `frontend/gnss/fix` | `event` | `/meridian/events` (Info) | on | NEW. Fix-type transitions (`SPP`/`DGPS`/`RTK_Float`/`RTK_Fixed`, spec 01 §4.4) so a residual change is correlated with fix quality. |

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
| `pipeline/q_meas_dropped` | `scalar` | `/meridian/telemetry` (`count`) | on | NEW. Cumulative lossy drops on `Q_meas` overload (spec 00 §11.2) — must stay flat. |
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

A healthy 10 Hz LiDAR + 20 Hz camera run on the FusionPortable rig should show
roughly (Jetson Orin):

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
`frontend/visual/res_mean`, `frontend/gnss/res`, `pipeline/scan_to_odom_ms`,
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
  which can back-pressure; we do not); telemetry/events use *reliable, depth 50*; TF
  uses the standard TF QoS.
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

### 11.1 Production posture vs. forensic posture

- **Production:** `debug.level=info`; heavy payloads (`map/registered`, `body/scan`,
  `frontend/lidar/inliers`, `frontend/visual/patches`) rate-limited to 1–2 Hz or
  off; scalars/events on (cheap); a black-box `CsvTelemetrySink` recording all
  scalars/events to disk via `MultiSink` for post-incident analysis.
- **Forensic:** flip clouds/patch-overlays on, raise rate, drop a module to
  `trace` — live, no restart. This is the operational reason the toggle exists: you
  cannot restart a SLAM run in the field to debug it.

### 11.2 Forensic snapshot

`SnapshotForensics(seconds)` dumps the last *N* seconds of a ring buffer the
`RecordingSink` keeps (poses, scalars, inlier/patch overlays, events) to a self-
contained file the offline `replay` tool can load — so a field anomaly becomes a
desk-reproducible case (§13).

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

A CI replay test binds a `RecordingSink`, runs a short FusionPortable/M2DGR
segment (`DATASET.md`), and asserts on telemetry: e.g. `frontend/lidar/inlier_ratio`
median > 0.6, `frontend/visual/n_tracked` median > 30, no `frontend/window_restart`
event, final `backend/chi2` within tolerance of golden, ATE from
`corrected_trajectory()` under threshold. **This is debugging-as-testing** — the
same signals an operator watches are the signals CI gates on. FAST-LIO cannot do
this (its signals are globals + CSV).

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
    `scalar("frontend/iter_count"|"dx_norm"|"cost_total"|"window_span_s")`,
    `vec("frontend/observability"|"cov_diag")`, `pose("odom/body")`,
    `timing("frontend.window_solve"|"frontend.total")`.
  - *LiDAR stream:* `cloud("frontend/lidar/inliers")`,
    `scalar("frontend/lidar/n_inlier"|"n_input"|"inlier_ratio"|"res_mean"|"res_max")`,
    `timing("frontend.lidar_assoc")`.
  - *visual stream:* `image("frontend/visual/patches")`,
    `scalar("frontend/visual/n_tracked"|"n_converged"|"res_mean")`,
    `vec("frontend/visual/exposure_gain")`, `timing("frontend.visual")`.
  - *IMU/state:* `scalar("frontend/imu/res_acc"|"res_gyr"|"grav_norm")`,
    `vec("frontend/bias_acc"|"bias_gyr")`.
  - *GNSS stream:* `pose("frontend/gnss/anchor")`, `scalar("frontend/gnss/res")`,
    `event("frontend/gnss/fix")`.
  - *lifecycle/calib:* `event("frontend/init_done"|"frontend/window_restart")`,
    `pose("calib/T_imu_lidar"|"calib/T_imu_cam")`, `scalar("calib/version")`.

  **This is the user-priority module; its debug surface is the richest by design.**
- **L3 back-end:** `scalar("backend/chi2"|"n_factors"|"n_keyframes"|"n_moved"
  |"loop_correction_norm")`, `event("backend/relinearize")`,
  `pose`→`backend/trajectory`, `timing("backend.optimize")`.
- **L4 map (nvblox):** `event("map/region_rebuild")`, `marker("map/dirty_region")`,
  `timing("map.integrate"|"map.deintegrate"|"mesh.extract")`,
  `scalar("map/tsdf_blocks"|"map/integrate_lag")`.
- **L5 place:** `marker("place/loop_edge")`, `event("place/loop_accepted"
  |"place/loop_rejected_pcm")`, `timing("place.query"|"place.verify")`.
- **Pipeline:** `scalar("pipeline/q_*_depth"|"q_meas_dropped"|"scan_to_odom_ms")`.

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
| GNSS datum / quality problem | `frontend/gnss/res` large + persistent; correlates with `gnss/fix` type | `frontend/gnss/res`, `frontend/gnss/fix` |
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
```

### 16.2 Sink implementations (selected by `Config.debug`)

```
NullSink           default; empty bodies; enabled()→false        (meridian_debug)
RecordingSink      captures all calls into a ring buffer          (meridian_debug; tests + forensics)
MultiSink          fan-out to ordered children, short-circuits    (meridian_debug)
CsvTelemetrySink   per-key CSV append                             (meridian_tools)
RosTelemetrySink   →topics/TF/markers/images; owns the one to_ros (meridian_ros)
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
frontend/iter_count|dx_norm|cost_total|window_span_s  scalar → /meridian/telemetry  Tel  NEW (window NLLS)
odom/body                      pose    → /meridian/odom + TF        Odom /Odometry                     :857
# --- L2 LiDAR stream (point-to-plane @ true point time) ---
map/registered                 cloud   → /meridian/cloud_registered PC2  /cloud_registered  :849 / build :478 (via T(t_i))
frontend/lidar/inliers         cloud   → /meridian/cloud_effective  PC2  /cloud_effected :853 / :551 (UNCOMMENTED vs :983)
frontend/lidar/n_inlier        scalar  → /meridian/telemetry        Tel  effct_feat_num                :695  (NEW topic)
frontend/lidar/n_input|inlier_ratio|res_mean|res_max  scalar → /meridian/telemetry  Tel  res_mean_last :715 + NEW
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
frontend/gnss/res              scalar  → /meridian/telemetry        Tel  NEW
frontend/gnss/fix              event   → /meridian/events           Evt  NEW (fix-type transitions)
# --- L2 lifecycle / calibration ---
frontend/init_done             event   → /meridian/events           Evt  NEW (cf flg_EKF_inited        :898)
frontend/window_restart        event   → /meridian/events           Evt  NEW (recovery visible; cf :708-712)
calib/T_imu_lidar              pose    → /meridian/extrinsic        Odom was fout_* offset_R/T_L_I
calib/T_imu_cam                pose    → /meridian/extrinsic        Odom NEW (camera extrinsic)
calib/version                  scalar  → /meridian/telemetry        Tel  NEW
# --- L3 back-end ---
backend/chi2|n_factors|n_keyframes|n_moved|loop_correction_norm  scalar → /meridian/telemetry  Tel  NEW
backend/relinearize            event   → /meridian/events           Evt  NEW
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
pipeline/q_*_depth|dropped     scalar  → /meridian/telemetry        Tel  NEW
pipeline/scan_to_odom_ms       scalar  → /meridian/telemetry        Tel  NEW (RT SLA)
timing(stage,ms)               timing  → /meridian/stage_timing     ST   aver_time_* (live, not CSV)   :991-1009/:1042-1044
```

### 16.5 Stage-timing keys

```
preprocess  frontend.lidar_assoc  frontend.visual  frontend.window_solve  frontend.total
backend.optimize  place.query  place.verify  map.integrate  map.deintegrate  mesh.extract
```
