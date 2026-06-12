# 02 — L0: Sensor Abstraction & Time Synchronization

> **Spec status:** normative (implementation spec). This document specifies the
> bottom layer of the Meridian stack, **L0**: the code that turns a heterogeneous,
> asynchronous, multi-vendor sensor rig — **one LiDAR, one IMU, one camera, one
> GNSS receiver** — into a single stream of **time-synced, frame-tagged,
> plain-C++ samples** on one monotonic timeline, plus the machinery that
> *establishes* that timeline (PTP grandmaster from GNSS PPS) and *measures its
> own residual error* (software time-offset estimation, sensor-health channel).
>
> **Reading order.** This spec implements the boundary types declared in
> [`01_interfaces_and_data_types.md`](01_interfaces_and_data_types.md) (`ImuSample`
> §4.1, `LidarScan`/`LidarPoint` §4.2, `CameraFrame` §4.3, `GnssFix` §4.4,
> `Extrinsic`/`CalibrationSet` §5, `ISensorSource` §7.1) and obeys the
> architecture in [`00_architecture.md`](00_architecture.md) (ROS-agnostic core,
> §1; package `meridian_sensors` + `meridian_time`, §2; threading stage T1, §11;
> telemetry bus, §10). Where this spec needs a type not yet in spec 01, it says
> so explicitly and the change is amended into spec 01, never redefined here
> (spec 01 R-rules).
>
> **Grounding.** Sensor *variety* and the per-sensor stamping pitfalls are
> grounded in the reference implementation `FAST_LIO/src/preprocess.{h,cpp}`,
> cited as `preprocess.cpp:NNN` / `preprocess.h:NNN`. FAST-LIO's `Preprocess`
> class is where the reference code copes with several LiDAR families and four
> time units (`SEC/MS/US/NS`, `preprocess.h:18`); we keep what it got right
> (per-point time is first-class, a per-sensor handler, a time-unit scale) and
> fix what it got wrong (no health signal, no offset estimation, time smuggled
> through `curvature`, sync done by a single `time_sync_en` boolean). The
> system Meridian builds — a **discrete, tightly-coupled
> LiDAR-Inertial** estimator (arch §0) — depends on the timeline everywhere: it
> warps *each point at its true sample time*, integrates *every IMU instant*
> into propagation and the motion prior, and stamps the camera at
> *mid-exposure* for the passthrough image. All three
> demands are demands on the **single timeline** L0 produces.
>
> **Scope of L0.** L0 owns: per-sensor acquisition (`ISensorSource`), conversion
> from wire format to the L1 currency types, **timestamping policy** (which clock,
> what instant a stamp denotes), the **clock model** (`meridian_time`: PTP/PPS
> disciplined estimate of true time), **software time-offset estimation** (the
> verifier for hardware sync, and the estimator when a sensor is not
> hardware-disciplined), and the **sensor-health channel**. L0 does **not**
> deskew, downsample, extract features, or transform into the estimation frame —
> those are L1/L2 (spec 03). The one exception worth stating: L0 *aggregates*
> samples into a `MeasureGroup` (the LiDAR-sweep-plus-spanning-IMU bundle the
> front-end consumes), because aggregation is fundamentally a *timing* operation
> and belongs with the clock.

---

## Table of contents

1. [Why L0 exists: the problem statement](#1-why-l0-exists-the-problem-statement)
2. [The unified sample contract (recap + L0 obligations)](#2-the-unified-sample-contract-recap--l0-obligations)
3. [`ISensorSource` and the source taxonomy](#3-isensorsource-and-the-source-taxonomy)
4. [Per-modality sources](#4-per-modality-sources)
   - 4.1 [`OusterLidarSource`](#41-ousterlidarsource)
   - 4.2 [`ImuSource`](#42-imusource)
   - 4.3 [`CameraSource`](#43-camerasource)
   - 4.4 [`GnssSource`](#44-gnsssource)
   - 4.5 [`BagReplaySource`](#45-bagreplaysource)
5. [The time model (`meridian_time`)](#5-the-time-model-meridian_time)
   - 5.1 [Three timelines and the one we estimate](#51-three-timelines-and-the-one-we-estimate)
   - 5.2 [PTP grandmaster from GNSS PPS](#52-ptp-grandmaster-from-gnss-pps)
   - 5.3 [`ClockModel`: the per-sensor offset/skew estimator](#53-clockmodel-the-per-sensor-offsetskew-estimator)
6. [Per-sensor stamping policy](#6-per-sensor-stamping-policy)
   - 6.1 [Ouster PTP path](#61-ouster-ptp-path)
   - 6.2 [Camera GPIO-trigger path](#62-camera-gpio-trigger-path)
   - 6.3 [IMU stamping path](#63-imu-stamping-path)
   - 6.4 [GNSS PPS path](#64-gnss-pps-path)
7. [Software time-offset estimation (verify + estimate)](#7-software-time-offset-estimation-verify--estimate)
8. [`MeasureGroup` aggregation (L0→L1/L2 handoff)](#8-measuregroup-aggregation-l0l1l2-handoff)
9. [The sensor-health channel](#9-the-sensor-health-channel)
10. [Configuration schema](#10-configuration-schema)
11. [Failure modes and recovery](#11-failure-modes-and-recovery)
12. [Debug / introspection hooks](#12-debug--introspection-hooks)
13. [Threading, ownership, and lifetime summary](#13-threading-ownership-and-lifetime-summary)
14. [Worked example: a 100 ms tick through L0](#14-worked-example-a-100-ms-tick-through-l0)
15. [Test plan](#15-test-plan)

---

## 1. Why L0 exists: the problem statement

A tightly-coupled multi-sensor estimator is, mathematically, a function of a set
of measurements $\{z_i\}$ each indexed by the **true instant** $t_i$ at which the
physical quantity was sampled. Every downstream guarantee — the per-point deskew
warp at each point's true sample time (arch §7.1), IMU propagation and the
interval-averaged motion prior, the camera's mid-exposure stamp — depends on
$t_i$ being **correct on a single timeline to a known tolerance**. If two sensors
disagree about "now" by $\delta t$, a tightly-coupled fuse of them injects an
error proportional to the platform velocity times $\delta t$: at 5 m/s, a 10 ms
offset is a 5 cm phantom translation per fused constraint — far above the
few-centimetre map resolution Meridian targets, and worse, *systematic* (it biases,
it does not average out).

The hardware does not hand us this for free:

* **Each sensor runs its own oscillator.** An Ouster LiDAR, a MEMS IMU at 200 Hz,
  a global-shutter camera, and a GNSS receiver each free-run on a local crystal
  with its own offset and drift (ppm-level skew → tens of µs/s).
* **Each sensor stamps a different *instant*.** Ouster stamps each *column* at
  acquisition; a camera's meaningful instant is **mid-exposure**, not arrival; an
  IMU sample is "the interval just integrated"; a GNSS fix is "the PPS edge."
* **The transport adds jitter.** A USB/Ethernet/driver path adds a variable
  arrival delay that has nothing to do with the sampling instant. Stamping on
  *arrival* (`msg->header.stamp.toSec()` as FAST-LIO does for the scan time,
  `preprocess.cpp:255`) folds transport jitter into the measurement time.

FAST-LIO's answer is pragmatic but thin: a single boolean `time_sync_en` in the
config decides whether to trust sensor stamps or to slam the IMU and LiDAR onto a
common base, and per-point time is recovered however the sensor allows (Ouster's
`t` field, `preprocess.cpp:226`; Velodyne's `time` field, `:344`; or
*reconstructed from yaw* when absent, `:346–372`). There is no estimate of the
residual offset, no notion of which sensor is the time master, and no health
signal when sync degrades. For a research prototype on a calibrated bag that is
fine. For a tactical system that must run for hours and *report when it can no
longer be trusted*, it is not.

**L0's job, precisely:**

1. **Acquire** each sensor stream behind one interface (`ISensorSource`).
2. **Stamp** every sample with its best estimate of the true sampling instant on
   one monotonic timeline (the *Meridian timeline*, §5.1), using the best available
   mechanism per sensor (PTP, PPS, GPIO trigger, or software offset).
3. **Convert** wire format → the typed L1 currency (spec 01 §4), losslessly,
   with no algorithmic processing.
4. **Measure its own error**: continuously estimate residual per-sensor offset/
   skew and emit a **health** verdict per sensor.
5. **Aggregate** into the `MeasureGroup` the front-end consumes.

Everything else is forbidden to L0.

---

## 2. The unified sample contract (recap + L0 obligations)

The four sample types are defined in spec 01 §4 and are **not** redefined here.
L0's contract is a set of *obligations* on the fields it fills:

| Type | Stamp field semantics (what the instant means) | L0 must guarantee |
|---|---|---|
| `ImuSample` (01 §4.1) | `stamp` = instant the IMU integrated the reported $\Delta v,\Delta\theta$ over its sample interval, reported as the **interval end** (the convention preintegration assumes). | Monotonic non-decreasing per `sensor_id`; on Meridian timeline; **raw** acc/gyro (no de-bias, no orientation). |
| `LidarScan` + `LidarPoint` (01 §4.2) | `stamp_start` = true instant of the **first** point; `LidarPoint::t_offset_ns` = signed ns of *this* point relative to `stamp_start`. | Per-point time present and correct (reject scan otherwise, §11); `sweep_duration` = **max** `t_offset_ns` over the cloud (the sweep-end offset, since offsets are not guaranteed column-ordered) — so `t_end = stamp_start + sweep_duration` is the instant of the last point, not the surviving span after filtering; `points` Shared-immutable. |
| `CameraFrame` (01 §4.3) | `stamp` = **mid-exposure** instant. | Mid-exposure, not arrival; carry `exposure_s`,`gain`; `data` Shared-immutable. |
| `GnssFix` (01 §4.4) | `stamp` = the **PPS edge** the fix is referenced to. | PPS-disciplined; raw geodetic + ENU covariance + `fix` type. |

Two L0-specific obligations cut across all four:

* **O1 — single timeline.** Every `stamp` is in **integer nanoseconds on the
  Meridian timeline** (§5.1). No sample leaves L0 carrying a vendor clock value.
  This is the int64-ns discipline spec 01 §2.1 mandates and the reason we never
  smuggle time through a float field the way FAST-LIO reuses
  `PointType::curvature` to hold the per-point offset (`preprocess.cpp:226,344`).

* **O2 — provenance of the stamp.** Each source records, *per sample*, **how**
  the stamp was derived (hardware PTP / PPS / GPIO trigger / software offset /
  arrival-fallback). This is not stored in the sample (the front-end must not
  branch on it) but is emitted to the health channel (§9) and telemetry (§12),
  so an operator can see, e.g., "camera dropped from GPIO-trigger to
  arrival-fallback at t=…". This is the missing forensic signal in the reference.

### 2.1 The `StampSource` enum (new, owned by `meridian_time`)

```cpp
// meridian_time/include/meridian/time/stamp_source.hpp  (NO ros)
namespace meridian {
enum class StampSource : std::uint8_t {
  HwPtp        = 0,  // device stamped from a PTP-disciplined hardware clock (Ouster)
  HwPps        = 1,  // device stamped against a PPS edge (GNSS, PTP grandmaster)
  HwTrigger    = 2,  // device exposed/sampled on an external GPIO trigger (camera)
  SwOffset     = 3,  // host stamp corrected by an estimated software offset (§7)
  ArrivalOnly  = 4,  // last resort: host arrival time, no correction (degraded)
  Replay       = 5,  // stamp came from a recorded bag (offline; trusted as-is)
};
} // namespace meridian
```

This is the **only new public type** this spec introduces; it is added to spec
01 §2 as the companion of `Timestamp`.

> **On the per-sensor stamping ladder vs. "no fallbacks."** The architecture
> forbids *defensive dual code paths and "or alternatively" hedges* (arch §0:
> pick the single best option per job and commit) — that mandate governs
> structural choices like "one map backend, not a CPU fallback" and "one LiDAR,
> not a merge-of-N." The `HwPtp → SwOffset → ArrivalOnly` ladder below is **not**
> such a hedge: it is a single, fixed **health-degradation policy** for *one*
> sensor whose hardware sync transiently fails. There is exactly one stamping
> path per sensor; when its hardware mechanism drops, the sensor *degrades along
> a defined ladder and says so on the health channel* (§9) rather than silently
> emitting a wrong stamp. That is a measured-error report, not a second
> implementation.

---

## 3. `ISensorSource` and the source taxonomy

Spec 01 §7.1 declares `ISensorSource<SampleT>` as a **push-model template**: the
source owns an acquisition thread and invokes a `Callback = void(SampleT&&)` with
a finished, time-synced sample. L0 implements that interface; this section pins
down the obligations the template signature leaves implicit and adds the small
amount of *shared* machinery every source needs.

The rig is **one source per modality**: a single `OusterLidarSource`, a single
`ImuSource`, a single `CameraSource`, a single `GnssSource` (plus
`BagReplaySource` for offline runs). `sensor_id` is still carried on every sample
(spec 01 §4) so the *contract* is unchanged — but in this rig each modality has
exactly one id. (A multi-sensor rig is a future extension behind the *same*
`ISensorSource` interface and the same `sensor_id` field; it is not designed
here, and nothing in L0 is built around merging streams.)

### 3.1 The interface, restated with L0 obligations

```cpp
// spec 01 §7.1 — reproduced for obligations only; the canonical decl is in spec 01.
template <class SampleT>
class ISensorSource {
public:
  using Callback = std::function<void(SampleT&&)>;
  virtual void       set_callback(Callback cb) = 0;
  virtual void       start() = 0;
  virtual void       stop()  = 0;
  virtual SensorInfo info() const = 0;
};
```

L0 obligations on **every** implementation:

* **B1 — stamp before callback.** The sample handed to `cb` carries a *final*
  Meridian-timeline stamp (O1). No downstream code re-stamps.
* **B2 — monotonic per sensor.** Stamps are non-decreasing within one
  `sensor_id`. A regression (clock step backwards, §11) is *clamped and
  reported*, never silently emitted, because a non-monotonic stamp corrupts
  IMU integration and the time-sorted deskew order.
* **B3 — no blocking in the callback's caller.** The source thread must not block
  on the consumer. The consumer (L1 stage T1) is fed through the bounded
  `Q_sensors` queue (arch §11.1); the source pushes non-blocking and the queue's
  overflow policy (lossy for `Q_sensors`, arch §11.2) decides drops.
* **B4 — health emission.** On every sample *and* on every state change (sync
  acquired/lost, dropout, clock step) the source updates the `SensorHealth`
  record for its id (§9).
* **B5 — clean stop.** After `stop()` returns, no further callbacks fire; the
  acquisition thread is joined.
* **B6 — validate before callback.** Every sample passes the standing
  `InputValidator` (§9.4) before it reaches `cb`: the validator owns B2's
  monotonicity clamp, the dropout/skew checks, and the per-point-time / NaN-Inf /
  emptiness rejects. A `Reject` verdict means the sample never reaches `cb` (nor
  `Q_sensors`); a `Clamped` verdict means the stamp was repaired to hold B2 and the
  repair was reported on the health channel. The validator is always on (not a
  debug option), because a bad stamp corrupts the deskew warp and the IMU
  integration interval rather than adding one noisy measurement.

### 3.2 `SensorInfo` (static descriptor)

```cpp
// meridian_sensors/include/meridian/sensors/sensor_info.hpp
struct SensorInfo {
  std::uint8_t id          = 0;          // unique within the rig
  Modality     modality    = Modality::Imu;     // Imu|Lidar|Camera|Gnss
  Frame        sensor_frame = Frame::Unknown;   // os_sensor0 / cam_link / imu_link / gnss_link
  std::string  model        = "";        // "ouster_os0_128", "bmi085", ...
  double       nominal_rate_hz = 0.0;    // expected sample/scan rate (for dropout detection)
  StampSource  configured_stamp_source = StampSource::ArrivalOnly; // best mechanism this sensor is wired for
};
```

`nominal_rate_hz` is the analogue of FAST-LIO's `SCAN_RATE` (`preprocess.h:103`,
used to *reconstruct* point time from yaw when the LiDAR gives none,
`preprocess.cpp:297`); we use it instead for **dropout detection** (§9) and as the
band centre for rate health. The two checks use different signals on purpose:

- **Dropout** keys on the **raw** inter-sample gap: a single gap exceeding
  $k/\text{nominal\_rate}$ ($k = 2.5$ nominal periods) raises `Dropout` immediately,
  because a real hole must not be smoothed away. When `nominal_rate_hz` is unset the
  threshold falls back to `failed_timeout_ms`.
- **Rate band** (`RateLow`/`RateHigh`) keys on a **smoothed** rate estimate — an
  EWMA of the instantaneous $1/\text{gap}$ (factor $0.1$) — compared against
  `nominal_rate_hz · (1 ± rate_tolerance_frac)`. Smoothing keeps transport jitter on
  one gap from flipping the rate codes; the instantaneous rate is never the band
  signal.

All health transitions are **edge-throttled**: a code is raised once on entry and
cleared once on exit (the active-code bitset is checked first), so a sustained
condition produces a single event, not one per sample.

### 3.3 Shared base: `SourceBase`

To avoid re-implementing B1–B6 in each source, L0 provides a non-virtual
mix-in (not a base class in the inheritance chain crossing the interface — the
interface stays clean per arch R1):

```cpp
// meridian_sensors/src/source_base.hpp  (private header; impl detail)
class SourceBase {
protected:
  SourceBase(SensorInfo info, ClockModel* clock, HealthSink* health);

  // Convert a vendor instant on a NAMED clock into a Meridian-timeline stamp,
  // applying the ClockModel correction (§5.3) and recording provenance.
  Timestamp stamp_from(Timestamp vendor_ns, ClockId clock, StampSource src);

  // Monotonicity clamp + dropout/skew/step bookkeeping → health. Both delegate to
  // the standing InputValidator (§9.4), which owns last_stamp_ and the active-code
  // bitset; these are the thin per-source entry points (B2, B4, B6).
  Timestamp enforce_monotonic(Timestamp t);          // = validator_.on_stamp(...).stamp
  void      note_sample(Timestamp t, StampSource src);

  const SensorInfo info_;
  ClockModel*      clock_;     // borrowed; lives in meridian_time, owned by pipeline
  HealthSink*      health_;    // borrowed; §9
  InputValidator   validator_; // §9.4 — owns last_stamp_, edge-throttled health codes
};
```

Each concrete source *composes* `SourceBase` and only writes the wire→type
conversion plus the choice of which `stamp_from` to call. This is the structural
fix for FAST-LIO's `Preprocess` god-class, where AVIA/Velodyne/Ouster handlers
each re-derive timing inline (`avia_handler` `preprocess.cpp:92`,
`velodyne_handler` `:284`, `oust64_handler` `:189`) with subtly different unit
handling.

---

## 4. Per-modality sources

Every source: (a) subscribes to / opens its raw stream in the **wrapper**
(`meridian_ros`), which converts the ROS/driver message to a thin POD and calls the
core source's `ingest_raw(...)`; OR (b) in offline mode, `BagReplaySource`
feeds the same `ingest_raw`. The **core source never touches ROS** (arch §1.1):
the wrapper does the `sensor_msgs`→POD step, the core does POD→typed-sample +
stamping. This keeps L0 testable with synthetic POD frames.

> **Wire→POD→type, not wire→type.** The intermediate POD (e.g.
> `RawLidarFrame { Timestamp host_arrival; Timestamp device_ns; bool has_device_ns;
> std::span<const RawPoint> pts; ... }`) is the boundary that lets the core be
> ROS-free yet still see the *device* clock value it needs for stamping. The
> wrapper fills `device_ns` from the Ouster column timestamps / the
> `sensor_msgs` header; the core decides what to do with it.

### 4.1 `OusterLidarSource`

**Modality:** LiDAR. The rig has **one** LiDAR; `OusterLidarSource` is the single
LiDAR source (the generalised `OUST64` path of FAST-LIO's `Preprocess`,
`preprocess.cpp:189`, reduced to the one device Meridian runs).

**Wire → POD.** The wrapper hands `ingest_raw` a `RawLidarFrame` containing, per
return: `x,y,z` (sensor frame, metres), `intensity`, **`t`** (the Ouster
per-column nanosecond offset within the sweep — `ouster_ros::Point::t`, a
`uint32_t`, `preprocess.h:63,79`), `reflectivity`, `ring`, `ambient`, `range`;
plus the frame-level **`device_ns`** (the sweep's reference time on the LiDAR's
PTP-disciplined clock) and the host arrival time.

**Stamping.**

* `stamp_start` is derived from the **device** clock, not arrival:
  `stamp_start = stamp_from(device_ns_of_first_column, ClockId::Lidar, src)`
  where `src = HwPtp` when PTP lock is asserted (§6.1), else the software-offset
  path (§7).
* `LidarPoint::t_offset_ns = int32_t(point.t)` — the Ouster `t` field is already
  a per-column offset from the sweep reference in nanoseconds, so it maps
  **directly** to our `t_offset_ns` with **no unit conversion**. This is strictly
  better than FAST-LIO, which multiplies `t` by a `time_unit_scale` to land in
  *milliseconds* inside `curvature` (`preprocess.cpp:226`, with
  `time_unit_scale` chosen from the `TIME_UNIT` enum `preprocess.h:18`,
  `preprocess.cpp:52–68`); we keep native ns and never round-trip through a
  float. Per-point time is exactly what the front-end needs: it warps each
  point by its own offset within the constant-screw model (arch §7.1), so the
  fidelity of this field is the fidelity of deskew.
* `sweep_duration = max t_offset_ns` over the cloud (≈100 ms at 10 Hz). Offsets
  are relative to `stamp_start` and are **not** guaranteed ordered by column time,
  so the sweep end is the largest offset anywhere in the cloud, not the offset of
  the last-iterated point.

**Per-point validity is NOT applied here** (no blind-radius cull, no
`point_filter_num` decimation). FAST-LIO does those in `oust64_handler`
(`range < blind*blind` cull `:210,264`; `i % point_filter_num` decimation
`:260`). In Meridian those are **L1** (spec 03); L0 ships the full raw cloud so L1
and the retained keyframe store (MUST-FIX #4) can decide. L0's *only* point-level
action is to populate `LidarPoint` losslessly.

```cpp
class OusterLidarSource final : public ISensorSource<LidarScan>, SourceBase {
public:
  OusterLidarSource(SensorInfo, ClockModel*, HealthSink*, OusterCfg);
  void set_callback(Callback cb) override { cb_ = std::move(cb); }
  void start() override; void stop() override;
  SensorInfo info() const override { return info_; }

  // Called by the wrapper (or BagReplaySource) with a wire-free POD frame.
  void ingest_raw(const RawLidarFrame& f);

private:
  Callback   cb_;
  OusterCfg  cfg_;     // ptp_domain, expected sweep ns, intensity scale, ...
};

void OusterLidarSource::ingest_raw(const RawLidarFrame& f) {
  const bool ptp_ok = f.has_device_ns && clock_->ptp_locked(ClockId::Lidar);
  const StampSource src = ptp_ok ? StampSource::HwPtp : StampSource::SwOffset;

  LidarScan scan;
  scan.sensor_id     = info_.id;
  scan.sensor_frame  = info_.sensor_frame;
  scan.stamp_start   = enforce_monotonic(
                         stamp_from(f.device_ns_first_column, ClockId::Lidar, src));
  auto pts = std::make_shared<std::vector<LidarPoint>>();
  pts->reserve(f.pts.size());
  std::int32_t max_t_offset = 0;
  for (const RawPoint& p : f.pts) {
    LidarPoint lp;
    lp.xyz         = {p.x, p.y, p.z};
    lp.intensity   = p.intensity;
    lp.t_offset_ns = static_cast<std::int32_t>(p.t);   // native ns, NO scale (cf. preprocess.cpp:226)
    lp.ring        = p.ring;
    lp.ambient     = p.ambient;
    lp.range       = p.range_m;
    max_t_offset   = std::max(max_t_offset, lp.t_offset_ns);
    pts->push_back(lp);
  }
  // Offsets are not guaranteed column-ordered, so the sweep ends at the largest
  // offset anywhere in the cloud, not at the last-iterated point.
  scan.sweep_duration = static_cast<Duration>(max_t_offset);
  scan.points = std::move(pts);                         // Shared-immutable hereafter

  note_sample(scan.stamp_start, src);                   // health/dropout (§9)
  if (validator_.on_lidar(scan) == Reject) return;      // §9.4: per-point-time, NaN/Inf, empty
  cb_(std::move(scan));                                 // push to Q_sensors (non-blocking)
}
```

### 4.2 `ImuSource`

**Modality:** IMU. The rig has **one** IMU — the estimation-frame IMU
(`imu_link`, arch §8.2, spec 01 §2.3). Whether it is a discrete tactical IMU or
the LiDAR's built-in IMU is a wiring/config choice; either way it is a single
`ImuSource`.

**Wire → POD.** `RawImuFrame { Timestamp device_ns; bool has_device_ns;
Eigen::Vector3d acc; Eigen::Vector3d gyro; Timestamp host_arrival; }`. **No orientation** is carried
across, even if the device provides an AHRS quaternion — spec 01 §4.1 forbids
trusting vendor fusion; the estimator owns orientation.

**Stamping.** Three cases, in priority order:

1. **Device timestamp on a PTP/PPS-disciplined clock** → `HwPtp`/`HwPps`.
2. **Device timestamp on a free-running device clock** → `SwOffset`: apply the
   `ClockModel` offset+skew for `ClockId::Imu` (§5.3), which has been
   continuously estimated against the disciplined host (§7).
3. **No device timestamp** (driver only gives arrival) → `ArrivalOnly`,
   health = DEGRADED. The IMU is the *worst* sensor to leave on arrival time
   because its high rate makes transport jitter a large fraction of the sample
   interval; we flag this loudly.

**Interval-end convention.** The `stamp` denotes the **end** of the integration
interval the sample summarises. If a device documents mid-interval or
start-of-interval semantics, the source shifts by ±half the nominal period so the
contract (spec 01 §4.1, "interval end") holds. This matters because the
front-end integrates each sample over $[\text{stamp}_{k-1},
\text{stamp}_k]$; getting the convention wrong biases every IMU
contribution by one sample.

```cpp
void ImuSource::ingest_raw(const RawImuFrame& f) {
  StampSource src; Timestamp base;
  if (f.has_device_ns && clock_->disciplined(ClockId::Imu)) {
    src  = clock_->stamp_source(ClockId::Imu);             // HwPtp/HwPps or SwOffset
    base = stamp_from(f.device_ns, ClockId::Imu, src);
  } else {
    src  = StampSource::ArrivalOnly;
    base = f.host_arrival;                                  // degraded
    health_->degrade(info_.id, HealthCode::ImuNoDeviceClock);
  }
  ImuSample s;
  s.sensor_id = info_.id;
  s.acc       = f.acc;     // raw specific force [m/s^2], gravity+bias included (spec 01 §4.1)
  s.gyro      = f.gyro;    // raw angular rate [rad/s]
  s.stamp     = enforce_monotonic(base + cfg_.interval_end_shift_ns);
  note_sample(s.stamp, src);
  cb_(std::move(s));
}
```

### 4.3 `CameraSource`

**Modality:** camera. The rig has **one** global-shutter camera feeding the
FAST-LIVO2-style sparse-direct photometric residual (arch §7.5).

> **Rolling shutter (designed, deferred).** A global-shutter sensor has one
> exposure interval for the whole frame, so a single mid-exposure stamp is
> well-defined. A rolling-shutter sensor exposes each row at a different time, so
> the correct treatment mirrors the LiDAR per-point-time model: L0 would carry a
> per-row time offset (a `row_t_offset_ns` analogous to `LidarPoint::t_offset_ns`)
> and the front-end would query `T(t)` per row, exactly as it does per LiDAR point.
> That capability is **not built now** — the rig is global-shutter and
> `sensors.camera.shutter: global` (§10) is the committed configuration. The
> rolling-shutter path is a config flag (`shutter: rolling`) that, when added,
> switches L0 to stamp per row and L2 to reproject per row; until then a frame
> declared `rolling` is rejected at start (`Config::validate`) rather than
> silently stamped as if global.

**The canonical camera instant is mid-exposure.** For a global-shutter sensor the
whole frame integrates light over a single interval `[trigger_edge, trigger_edge +
exposure_s]`; the instant whose pose best represents the photons that formed the
image is the **centre** of that interval. Stamping on arrival is wrong by the
transport delay; stamping on the trigger edge (or the frame-header time) is wrong
by half the exposure — a fixed bias that scales with shutter speed and platform
velocity (a 16 ms exposure at 5 m/s is a 4 cm phantom translation on every
photometric constraint). L0 therefore pins the stamp to **trigger_edge +
exposure_s/2** and ships exactly that one instant; the photometric residual queries
`T(t)` at it (arch §7.5):

```
stamp_mid = stamp(trigger_edge) + round(exposure_s / 2 * 1e9)   // ns
```

This is a **single, fixed convention**, not a configurable choice. It differs from
the FAST-LIVO2 lineage, which carries the frame's raw header/sync stamp into the
VIO residual with no half-exposure correction; that omission is acceptable for
short exposures on a slow platform and a silent bias otherwise. Meridian pins
mid-exposure once, at L0, so every downstream consumer (the front-end residual, the
`MeasureGroup` interval test in §8, keyframe selection) sees the same instant and no
module re-derives it. The `CameraFrame::stamp` contract (spec 01 §4.3 and the §2
obligation table) means mid-exposure everywhere the field is read.

> **Residual camera↔body-IMU time offset.** The mid-exposure convention corrects
> the *intra-frame* exposure bias; it does not correct a residual *inter-sensor*
> offset between the camera shutter timeline and the body-IMU timeline. An
> uncompensated offset is a systematic reprojection bias on every photometric
> constraint (10 ms at 1 m/s ≈ 1 cm of phantom translation), so it must be
> folded into the stamp, not absorbed by robust weighting. The committed
> correction today is the constant `sensors.camera.time_offset_ms` shift applied
> once at ingest (§5 note), loaded from calibration and trusted only after an
> empirical A/B on the actual recording — a calibration-session timeshift is not
> automatically valid for a recording (§5, spec 08 §7.2). The §7 estimator path
> (estimate `ClockId::Cam` against the body-IMU reference and apply it through
> `to_meridian`, the same way the IMU↔LiDAR offset is handled) is the intended
> replacement for the constant; it is specified but not yet wired (§7.1).

**Stamping paths (priority):**

1. **GPIO hardware trigger** (`HwTrigger`): the camera is triggered by a pulse
   the PTP grandmaster (or the Ouster sync output) generates on a known schedule.
   The source matches each arrived frame to the trigger edge whose stamp is
   closest within a gate (§6.2), then adds `exposure_s/2`. This is the only path
   that achieves sub-ms camera/LiDAR alignment, which sparse-direct photometric
   fusion needs (a 5-pixel reprojection error at 5 m/s and 30 Hz is a ~1.5 ms
   timing error — within reach of GPIO, hopeless on arrival).
2. **Software offset** (`SwOffset`): no GPIO; use the device timestamp corrected
   by the `ClockModel`, plus `exposure_s/2`.
3. **Arrival only** (`ArrivalOnly`): degraded; still add `exposure_s/2` so at
   least the mid-exposure shift is honoured.

`exposure_s` and `gain` are carried into `CameraFrame` for L2 photometric
normalisation (spec 01 §4.3) — the data the FAST-LIVO2-style exposure
compensation needs. If the driver does not report exposure, `exposure_s = 0` and
L0 flags `HealthCode::CamNoExposure` (the mid-exposure shift then can't be
applied, and L2 falls back to robust photometric weighting).

**Encoding.** L0 passes the raw encoding through (`Mono8`/`Bayer_RGGB8`/`RGB8`);
debayering and pyramid construction are **L1** (spec 01 §7.2 note). L0 must not
allocate a pyramid.

**Compressed streams.** When `sensors.camera.compressed: true`, the wire payload
is a JPEG/PNG `CompressedImage`; the transport wrapper decodes it straight to a
tightly-packed `Mono8` frame at ingest (the photometric stage consumes intensity
only, and the JPEG luma plane already exists at the decode). A payload that does
not decode to an 8-bit image is dropped with a throttled warning, never forwarded
malformed. This is how the Alphasense cameras (compressed-only mono8 JPEG in the
Newer College bags) enter the same intake as a raw stream.

### 4.4 `GnssSource`

**Modality:** GNSS receiver. Produces `GnssFix` (spec 01 §4.4) **and** drives the
PPS path that disciplines the whole rig (§5.2) — so `GnssSource` is special: it
is both a measurement source and (with the PTP daemon) the *root of the time
tree*.

**Two outputs from one device:**

* **The PPS edge** (one pulse per second, rising edge aligned to GPS-second
  boundary to <100 ns). This is consumed by the **PTP grandmaster** (§5.2), not
  shipped as a sample. `GnssSource` exposes it to `meridian_time` via a callback.
* **The fix** (`GnssFix`), stamped to the **PPS edge** it is referenced to
  (`HwPps`), carrying lat/lon/alt, ENU covariance, `fix` type, `num_sats`. L0
  does **not** convert to a metric/ENU position — the datum is the back-end's
  decision (spec 01 §4.4); L0 reports the raw geodetic fix.

**Fix gating is NOT done here.** Weighting an `SPP` fix vs. an `RTK_Fixed` fix is
the back-end's job (spec 01 §4.4, GNC kernels spec 05 §8). L0 only reports `fix` and
`cov_enu` honestly so the back-end can gate.

### 4.5 `BagReplaySource`

**Modality:** any (one `BagReplaySource<SampleT>` per recorded stream). This is
the offline path that makes "the same core runs on a bag and on the robot" true
(arch §1, §9.3; `DATASET.md`: Newer College primary).

* It reads recorded **typed samples** (or POD frames) and replays them through
  the *same* `Callback`, preserving their recorded stamps with
  `StampSource::Replay` (trusted as-is — a bag's timeline is already fixed).
* It supports **deterministic ordering** by stamp (merge across streams) for the
  single-thread determinism mode (arch §11.2), and a `--rate` knob (1×, faster,
  step) for debugging.
* It **bypasses the `ClockModel`** (no live discipline offline) but *can*
  optionally re-run software offset estimation (§7) against the recorded streams
  to **validate** a dataset's sync — a useful forensic tool the reference lacks.

---

## 5. The time model (`meridian_time`)

> **Constant per-sensor stamp correction (`time_offset_ms`).**
> `sensors.lidar.time_offset_ms` and `sensors.camera.time_offset_ms` apply a
> constant shift onto the body-IMU timeline (t_corrected = t_sensor + offset) once,
> in the pipeline `ingest()` overloads — BEFORE the validator and the aggregator,
> so every stamp-driven decision (monotonicity clamps, rate estimation, group
> assembly, image–sweep matching) operates on corrected time, identically live and
> in replay. LiDAR per-point offsets are relative to the sweep reference and ride
> along unchanged. Default 0 = bit-identical to the uncorrected path. A
> calibration-session timeshift is NOT evidence for a recording (hardware sync
> paths differ per sensor and per session — spec 08 §7.2): set the key only after
> an empirical A/B on the actual data.

### 5.1 Three timelines and the one we estimate

There are three clocks in play:

* **TAI/GPS time** $t_{\text{gps}}$ — the absolute, monotonic, drift-free
  reference the GNSS receiver recovers (modulo a fixed GPS-UTC leap offset).
* **The Meridian timeline** $t_H$ — the int64-ns monotonic timeline *every sample
  stamp is expressed in* (spec 01 §2.1). When GNSS is available, $t_H$ is
  **disciplined to GPS time** so it is also absolute (good for georeferencing and
  multi-robot). When GNSS is absent, $t_H$ free-runs on the host monotonic clock;
  it stays *internally* consistent (all sensors on one timeline) but loses
  absolute meaning — which is fine for local SLAM and flagged in health.
* **Each device clock** $t_{\text{dev},s}$ — a free-running per-sensor oscillator
  with offset $o_s$ and skew $\alpha_s$ relative to $t_H$:
  $$ t_{\text{dev},s} = (1+\alpha_s)\,t_H + o_s + \varepsilon_s . $$

L0 estimates the per-device $(o_s, \alpha_s)$ so it can invert the relation and
place every device's events on $t_H$:
$$ \hat t_H = \frac{t_{\text{dev},s} - \hat o_s}{1 + \hat\alpha_s}. $$
For PTP/PPS-disciplined devices the hardware drives $(o_s,\alpha_s)\to(0,0)$ and
the correction is the identity; for free-running devices the **software offset
estimator** (§7) supplies $(\hat o_s,\hat\alpha_s)$.

### 5.2 PTP grandmaster from GNSS PPS

The hardware time tree: a single **PTP (IEEE-1588) grandmaster** disciplined by
the **GNSS PPS** distributes time over Ethernet to the LiDAR (an Ouster PTP
slave) and the camera (GPIO-triggered off the same master). A PTP-aware IMU
slaves directly; a free-running IMU is brought on via the software offset
estimator (§7).

```
                 GNSS receiver
                   │  ┌──────────── NMEA/fix (→ GnssSource → GnssFix)
                   │  │
            PPS ───┘  └── time-of-day
             │ (1 Hz, <100 ns edge)
             ▼
        ┌─────────────────────┐    PTP (IEEE-1588) over Ethernet
        │ PTP GRANDMASTER     │───────────────┬───────────────────┐
        │ (host: ptp4l+phc2sys│               │                   │
        │  disciplined by PPS)│        ┌──────▼─────┐      ┌───────▼────┐
        └─────────────────────┘        │  Ouster    │      │ PTP IMU    │
             │  (host clock = t_H)      │ (PTP slave)│      │ (PTP slave)│
             │                          └────────────┘      └────────────┘
             │  GPIO trigger (N Hz, derived from PPS)
             └─────────────────────────────────────────▶ Camera (HwTrigger)
```

* **`ptp4l`** runs the IEEE-1588 protocol on the host NIC; **`phc2sys`** slews the
  system clock to the NIC PHC; the PPS in turn disciplines the PHC (via the kernel
  PPS API / `ts2phc`). Net effect: the **host system clock is GPS-disciplined**
  and *is* $t_H$, and every PTP slave's hardware clock tracks it to sub-µs.
* **L0 does not implement the PTP protocol** — that is OS/daemon territory.
  `meridian_time` *observes* its quality through the OS: PTP offset/path-delay from
  `ptp4l` statistics, PPS edge presence from the kernel PPS device, and the PHC↔
  system offset from `phc2sys`. It exposes a `ClockModel::ptp_locked(ClockId)` and
  a numeric `ptp_offset_ns(ClockId)` for stamping decisions and health.
* **Grandmaster selection** is config (`time.source: ptp`); if PTP is configured
  but `ptp4l` reports unlocked, the affected source degrades to the software
  offset (§7) and health degrades — the transition is a telemetry event (§12).

> **Why GNSS-PPS-disciplined PTP and not "trust each sensor's NTP/host stamp."**
> NTP gives millisecond sync; PTP gives sub-microsecond; PPS gives the absolute
> second boundary. Tightly-coupled LiDAR-visual fusion needs sub-millisecond
> (§4.3). Only the PPS→PTP chain reaches it. FAST-LIO sidesteps this entirely by
> running on pre-synced bags or by forcing a common base with `time_sync_en`;
> Meridian must produce the sync, so the chain is part of the system.

### 5.3 `ClockModel`: the per-sensor offset/skew estimator

```cpp
// meridian_time/include/meridian/time/clock_model.hpp  (NO ros)
namespace meridian {

enum class ClockId : std::uint16_t { Host = 0, Lidar, Imu, Cam, Gnss };

struct ClockState {
  double      offset_ns = 0.0;   // o_s : device - host, at t_ref
  double      skew_ppm  = 0.0;   // alpha_s in parts-per-million
  Timestamp   t_ref     = 0;     // reference instant the (offset,skew) linearization holds at
  double      offset_std_ns = 0; // 1-sigma of the offset estimate (→ health, →L2 time-jitter noise)
  bool        disciplined = false;     // hardware (PTP/PPS) drives it
  StampSource source      = StampSource::ArrivalOnly;
  Timestamp   last_update = 0;
};

class ClockModel {
public:
  // Convert a device-clock value to the Meridian timeline (§5.1 inverse relation).
  Timestamp to_meridian(Timestamp device_ns, ClockId) const;

  // Hardware observers (from OS PTP/PPS), called by the platform-time daemon shim.
  void on_ptp_stats(ClockId, double offset_ns, double path_delay_ns, bool locked);
  void on_pps_edge (Timestamp host_ns_of_edge);

  // Software estimator update (§7): one (device, matched-host) correspondence.
  void on_correspondence(ClockId, Timestamp device_ns, Timestamp host_ns, double meas_std_ns);

  bool        ptp_locked(ClockId) const;
  bool        disciplined(ClockId) const;
  StampSource stamp_source(ClockId) const;
  ClockState  state(ClockId) const;     // for health/telemetry
};
} // namespace meridian
```

`to_meridian` applies $\hat t_H = (t_{\text{dev}} - \hat o_s)/(1+\hat\alpha_s\cdot
10^{-6})$ around `t_ref`. The estimator that fills `ClockState` for *non*-
disciplined clocks is §7; for disciplined clocks `on_ptp_stats`/`on_pps_edge`
set `offset≈0, skew≈0, disciplined=true` and feed the *residual* offset into
health.

`ClockId` is a flat enum of the rig's five clocks (`Host`, `Lidar`, `Imu`, `Cam`,
`Gnss`) — one per device, no per-instance packing, because there is one device
per modality.

---

## 6. Per-sensor stamping policy

This section is the decision table each source uses. The principle: **prefer the
mechanism that times the physical event closest to the silicon, fall back
explicitly down the defined degradation ladder, and record which one was used
(O2).** Each table below is a *single* policy for *one* sensor, not a menu of
alternative implementations.

### 6.1 Ouster PTP path

* **Best:** Ouster configured `timestamp_mode = TIME_FROM_PTP_1588`, PTP slave
  locked to the grandmaster. Each column's `t` is on the disciplined clock;
  `stamp_start = to_meridian(device_ns_first_column)` is the identity-plus-residual
  correction; `src = HwPtp`. Per-point `t_offset_ns` is native (§4.1).
* **Degrade A (transient):** PTP configured but transiently unlocked → `src =
  SwOffset` using the LiDAR's `ClockModel` state (which the software estimator
  keeps warm even while PTP is locked, §7, precisely so the fallback is instant
  and bounded). Health = DEGRADED `SyncLost`; resumes `HwPtp` on relock.
* **Degrade B (mis-wired):** Ouster in `TIME_FROM_INTERNAL_OSC` (no PTP at all) →
  permanent `SwOffset`; the estimator is the only thing keeping it aligned.
  Health = DEGRADED with a distinct code so the operator knows it's a *wiring*
  issue, not a transient.
* **Reject:** a scan with no per-point `t`. We do **not** reconstruct point time
  from yaw geometry — that path (`preprocess.cpp:346–372`) existed in FAST-LIO
  for Velodyne clouds lacking a `time` field; an Ouster always provides `t`, so
  its absence means a malformed stream, which we flag (§11 F6) rather than paper
  over. (Single LiDAR, known to provide `t`: we commit to requiring it.)

### 6.2 Camera GPIO-trigger path

* **Best (`HwTrigger`):** the grandmaster/Ouster sync-out drives the camera's
  external-trigger input at a fixed phase. `meridian_time` knows the trigger
  schedule (a periodic sequence of edge stamps on $t_H$). On frame arrival the
  source matches to the nearest scheduled edge within a **match gate**
  $\pm g$ (default $g = $ half the trigger period): if exactly one edge falls in
  the gate, `trigger_stamp = that edge`; `stamp = trigger_stamp + exposure_s/2`.
* **Ambiguity handling:** if zero or >1 edges fall in the gate (a dropped frame,
  or a frame-rate/trigger-rate mismatch), the match is *rejected* for that frame:
  degrade to `SwOffset`, increment `HealthCode::CamTriggerMismatch`, and *resync*
  the edge counter on the next clean match. This is the classic trigger-frame
  alignment bug; making it explicit (count + resync) is the fix.
* **`SwOffset` degrade:** device timestamp + `ClockModel` + `exposure_s/2`.
* **`ArrivalOnly` degrade:** host arrival + `exposure_s/2`, DEGRADED.

### 6.3 IMU stamping path

* **Best:** PTP-aware IMU (`HwPtp`) or IMU sharing the Ouster's sync (`HwPps`),
  device stamp on a disciplined clock.
* **Common case:** free-running IMU with a device timestamp → `SwOffset` via the
  estimator (§7). The IMU's high rate makes it the **best probe** for software
  offset estimation against the LiDAR (its dense samples interpolate cleanly to
  LiDAR sweep boundaries), so even when we *consume* it on `SwOffset` we *use* it
  to anchor the cross-correlation in §7.
* **Worst:** arrival-only IMU → `ArrivalOnly`, DEGRADED (loudly — §4.2 rationale).
* **Interval-end normalisation** applied per §4.2.

### 6.4 GNSS PPS path

* The fix is stamped `HwPps` to the PPS edge it references (the receiver reports
  which GPS-second the fix belongs to; we map that second's PPS edge — already on
  $t_H$ via §5.2 — to the fix). `src = HwPps`.
* If PPS is not wired (GNSS over serial only, no pulse), the fix degrades to
  `SwOffset` against the host clock and **PTP loses its absolute anchor** → the
  whole timeline degrades from "absolute/GPS" to "relative/host" (health event),
  but internal consistency is preserved.

---

## 7. Software time-offset estimation (verify + estimate)

L0 always runs a software offset estimator, for two jobs from one mechanism:

* **Verify** the hardware-disciplined clocks — compute the *residual* offset PTP/
  PPS failed to remove, and raise an alarm if it exceeds tolerance (the
  silent-bad-sync failure no boolean flag catches).
* **Estimate** the offset (and skew) for any clock that is *not*
  hardware-disciplined (a free-running IMU, an Ouster on internal oscillator),
  and keep the model **warm** so a transient PTP unlock degrades smoothly (§6.1).

This is the capability FAST-LIO lacks entirely: its `time_sync_en` either trusts
stamps or zeroes the LiDAR/IMU base difference once, with no ongoing estimate and
no error bar.

### 7.1 What is estimated

For each `ClockId` we estimate, against the **reference clock** (the disciplined
host $t_H$; if nothing is disciplined, the host monotonic clock serves as an
arbitrary-but-consistent reference):

* **Offset** $o_s$ [ns] — the dominant term, re-estimated every update.
* **Skew** $\alpha_s$ [ppm] — slow; estimated by linear regression of matched
  offsets over a sliding window (default 30 s). Skew matters over a mission: a
  20 ppm crystal drifts 20 µs/s → 72 ms/hour, enough to wreck deskew if ignored.
* **Offset std** $\sigma_{o_s}$ — fed to health (§9) *and* to L2 as a
  time-jitter measurement-noise inflation (a sensor we can only time to ±2 ms
  should contribute weaker constraints than one timed to ±50 µs).

> **Camera offset is the unwired case.** The estimator is specified for every
> `ClockId`, but the **camera** offset is not yet estimated live: the stamp the
> front-end consumes carries only the constant `sensors.camera.time_offset_ms`
> shift loaded from calibration (§5 note; §4.3, *Residual camera↔body-IMU time
> offset*). The intended end state lives here — estimate `ClockId::Cam`'s offset
> against the body-IMU reference and apply it through `to_meridian`, exactly as
> the IMU↔LiDAR offset is — rather than as a constant from a calibration
> session that must be re-verified per recording.

### 7.2 The estimator: cross-correlation of motion signals + recursive filter

Two complementary mechanisms:

**(a) Motion cross-correlation (coarse, robust, for the IMU↔LiDAR pair).**
The dominant unknown is the IMU-to-LiDAR offset. We exploit that *both* sensors
see the same platform motion. Form two scalar motion signals over a short window:

* IMU angular-rate magnitude $\omega_{\text{imu}}(t) = \lVert \text{gyro}(t)
  \rVert$, densely sampled.
* LiDAR-derived angular rate $\omega_{\text{lid}}(t)$ from scan-to-scan rotation
  (a cheap incremental ICP yaw, or the front-end's own per-scan $\Delta R$ once
  it is running), sampled per sweep.

The offset is the lag $\tau$ maximising the normalised cross-correlation:
$$ \hat\tau = \arg\max_{\tau}\; \frac{\sum_t \tilde\omega_{\text{imu}}(t-\tau)\,
   \tilde\omega_{\text{lid}}(t)}{\sigma_{\text{imu}}\,\sigma_{\text{lid}}}, $$
with $\tilde\omega$ the zero-mean signals. Evaluated on a grid then parabolic-
interpolated for sub-sample $\tau$. This is robust (no correspondence needed) and
gives the **coarse** offset to seed (b). It runs only during motion (a stationary
window has no signal — we gate on $\text{var}(\omega) > \theta$).

**(b) Recursive offset/skew filter (fine, continuous).**
Once seeded, each matched correspondence $(t_{\text{dev}}, t_{\text{host}})$ —
e.g. an IMU sample whose motion peak aligns with a LiDAR motion peak, or any
event observable on both clocks — updates a tiny 2-state Kalman filter on
$[o_s, \alpha_s]$:
$$
\begin{aligned}
\text{predict:}\quad & o \mathrel{+}= \alpha\cdot 10^{-6}\cdot \Delta t, \quad
   P \mathrel{+}= Q\,\Delta t,\\
\text{measure:}\quad & z = t_{\text{dev}} - t_{\text{host}}, \quad
   H = [1,\; (t-t_{\text{ref}})\cdot10^{-6}],\\
& K = P H^\top (H P H^\top + R)^{-1}, \quad
   x \mathrel{+}= K(z - Hx), \quad P = (I-KH)P,
\end{aligned}
$$
with $R$ from the correspondence's measurement std (cross-correlation peak
sharpness, or trigger-gate width). $\sigma_{o_s} = \sqrt{P_{00}}$ is published.

```
                      ┌──────────────── on_correspondence(id, dev, host, std) ───────┐
 motion windows ──▶ cross-correlate(ω_imu, ω_lid) ──▶ coarse τ̂ (seed) ──▶ 2-state KF │
                                                                          (o, α) + σ_o │
                                                          ClockModel.state(id) ◀───────┘
```

### 7.3 Verification mode (running even when hardware sync is up)

When PTP/PPS *is* locked, the estimator keeps running in **verify** mode against
the disciplined clock. Its output is the *residual* offset the hardware failed to
remove. If that residual exceeds a threshold (default 200 µs) the source raises
`HealthCode::SyncResidualHigh` — i.e. "PTP says locked but the data disagrees."
This catches the silent-bad-sync failure that no boolean flag can. It also keeps
the software model **warm** so the §6.1 transient-unlock transition is bounded.

### 7.4 Parameters

| Param | Default | Meaning |
|---|---|---|
| `xcorr.window_s` | 4.0 | motion window for coarse cross-correlation |
| `xcorr.grid_ms` | 0.5 | lag search resolution before parabolic interp |
| `xcorr.motion_gate_radps2` | 0.05 | min angular-rate variance to attempt a match |
| `kf.skew_window_s` | 30 | window over which skew is regressed |
| `kf.Q_offset_ns2_per_s` | 1e4 | offset process noise (random walk) |
| `kf.R_default_ns` | 500 | default correspondence std if none supplied |
| `verify.residual_warn_us` | 200 | residual offset → DEGRADED when hardware claims lock |
| `offset.reject_jump_ms` | 5 | a single correspondence implying >5 ms jump is rejected as outlier |

---

## 8. `MeasureGroup` aggregation (L0→L1/L2 handoff)

The front-end (spec 01 §7.3, FAST-LIO's update loop) does not consume a raw
sample stream; it consumes a **bundle**: one LiDAR sweep plus *all IMU samples
spanning that sweep* (and any image/GNSS that fall in the interval). FAST-LIO
builds exactly this in its sync step (its `MeasureGroup` with `lidar`,
`lidar_beg_time`, `lidar_end_time`, and the `imu` deque trimmed to the sweep —
`common_lib.h` and the sync logic in `laserMapping.cpp`). Aggregation is a
*timing* operation (it is defined entirely by interval containment on $t_H$), so
it lives in L0.

### 8.1 Type (canonical declaration in spec 01 Appendix A)

`MeasureGroup` is a **boundary type**, so its authoritative declaration lives in
spec 01 (Appendix A) per the §1 rule; it is repeated here for reference only:

```cpp
// meridian_common/include/meridian/common/measure_group.hpp — declared in spec 01 App. A
struct MeasureGroup {
  Timestamp                  t_begin = 0;   // sweep start on t_H (= scan.stamp_start)
  Timestamp                  t_end   = 0;   // sweep end   on t_H (= stamp_start + sweep_duration)
  LidarScan                  scan;          // the LiDAR sweep (Shared-immutable points)
  std::vector<ImuSample>     imu;           // samples in (lower, t_end], lower = min(prev_t_end, t_begin),
                                            // plus the straddler at/before t_begin (see §8.2 policy)
  std::optional<CameraFrame> image;         // image whose mid-exposure falls in [t_begin, t_end], if any
  std::vector<GnssFix>       gnss;          // fixes in the interval (usually 0 or 1)
};
```

The group carries the single LiDAR sweep directly. (A multi-LiDAR rig — a future
extension only — would attach additional sweeps behind the same interface; that
is explicitly *not* designed here, so `MeasureGroup` has no secondary-sweep field
and the Aggregator has no merge logic.)

### 8.2 Aggregation policy (the `Aggregator`)

```cpp
class Aggregator {                 // runs on stage T1 (arch §11.1)
public:
  // Fed every sample from every source's callback (after Q_sensors).
  void on(ImuSample&&);   void on(LidarScan&&);
  void on(CameraFrame&&); void on(GnssFix&&);
  using GroupSink = std::function<void(MeasureGroup&&)>;
  void set_sink(GroupSink);
};
```

Rules:

* **Trigger on sweep end.** A group closes when the LiDAR's sweep `[t_begin,
  t_end]` is complete **and** IMU has been received up to `t_end` (so the spanning
  IMU set is whole). The IMU deque is the gate: we hold the group until
  `imu.back().stamp >= t_end`, with a **timeout** (`agg.max_wait_ms`, default =
  1.5 sweep periods) after which we emit with whatever IMU arrived and flag
  `HealthCode::ImuLate`. This is FAST-LIO's "wait for IMU to cover the LiDAR end"
  condition made explicit and bounded.
* **IMU interval convention.** Include IMU with `stamp ∈ (lower, t_end]` where
  `lower = min(prev_t_end, t_begin)`, plus the **one** sample straddling `t_begin`
  (the newest IMU at or before `t_begin`, needed for the CT window / preintegration
  to start exactly at the sweep boundary). The straddler is emitted once, ahead of
  the in-interval set, even when it sits at or before `lower`. Clamping the lower
  edge to `t_begin` (rather than taking `prev_t_end` unconditionally) matters when
  header-stamp jitter against a fixed sweep period makes a sweep *begin before the
  previous one ended* (`t_begin < prev_t_end`): both overlapping groups then carry
  the IMU in their **common** window, so each sweep's IMU set spans its own
  `[t_begin, t_end]` with no hole at the seam. The front-end interpolates the
  boundary; L0 just guarantees coverage.
* **Reordering window.** Samples can arrive slightly out of order across sources
  (different transport latencies). The Aggregator keeps a small bounded reorder
  buffer per modality (`agg.reorder_ms`, default 20) and emits in stamp order;
  anything later than the window — `stamp < watermark − reorder_ms`, the watermark
  being the highest stamp accepted for that modality — is dropped with
  `HealthCode::LateDrop`.
* **IMU retention across the seam.** Emitting a group does **not** drain the IMU
  deque up to `t_end`. The Aggregator keeps every sample back to the newest one at
  or before `t_begin − reorder_ms`, because the next sweep can begin anywhere past
  that bound (stamp jitter, §8.2) and its straddling sample — and the IMU shared in
  an overlapping window — must survive this group's emission. Only IMU strictly
  older than that retained sample is dropped.
* **Lossy under overload.** If T2 (front-end) falls behind and `Q_meas`
  back-pressures, the Aggregator drops the **oldest** complete group (arch §11.2:
  `Q_meas` is lossy) and emits a telemetry `event` — never blocks the sensor
  thread.

### 8.3 Bootstrap interaction (MUST-FIX #2)

The Aggregator does not deskew and carries no bootstrap state (deskew and
initialization are L2-internal, arch §7.1–§7.2). Groups are forwarded
immediately and identically from the first one; the front-end holds them itself
until its static initialization completes. There is no cold-start label, flag,
or staging anywhere in L0/L1.

---

## 9. The sensor-health channel

A first-class output of L0, missing entirely from the reference. The estimator
and the operator must both be able to answer "is sensor X trustworthy *right
now*, and why?" The health channel is the structured answer.

### 9.1 Types (new; in `meridian_sensors`)

```cpp
enum class HealthLevel : std::uint8_t { Ok=0, Degraded=1, Failed=2 };

enum class HealthCode : std::uint16_t {
  None=0,
  // timing
  NoSync, SyncLost, SyncResidualHigh, ClockStepDetected, SkewOutOfRange,
  // dataflow
  Dropout, RateLow, RateHigh, LateDrop, ImuLate,
  // per-sensor specifics
  ImuNoDeviceClock, CamNoExposure, CamTriggerMismatch, GnssPpsLost, GnssFixDropped,
  // content
  LidarNoPointTime, LidarHighNanRatio, EmptyScan,
};

struct SensorHealth {
  std::uint8_t  sensor_id  = 0;
  Modality      modality   = Modality::Imu;
  HealthLevel   level      = HealthLevel::Ok;
  HealthCode    code       = HealthCode::None;   // dominant active code
  std::uint32_t code_bits  = 0;                  // bitset of all active codes
  StampSource   stamp_src  = StampSource::ArrivalOnly;  // current stamping mechanism
  double        rate_hz    = 0.0;                // measured (EWMA of 1/gap; feeds the rate band)
  double        offset_ns  = 0.0;                // ClockModel offset (0 if disciplined)
  double        offset_std_ns = 0.0;            // timing uncertainty (→ L2 noise)
  Timestamp     last_sample = 0;
  Timestamp     since      = 0;                  // when the current level began
};
```

### 9.2 `HealthSink` and aggregation

```cpp
class HealthSink {                 // implemented by meridian_debug bridge → telemetry
public:
  virtual void update(const SensorHealth&) = 0;     // per-sensor, called from sources/aggregator
  virtual void degrade(std::uint8_t id, HealthCode) = 0;   // convenience: raise a code
};

// Rig-level rollup the pipeline/back-end can query.
struct RigHealth {
  HealthLevel worst = HealthLevel::Ok;
  std::vector<SensorHealth> sensors;
  bool   timeline_absolute = false;   // true iff GPS/PPS anchor present (§5.1)
  double max_offset_std_ns = 0.0;     // worst timing uncertainty across sensors
};
```

### 9.3 Level policy (how codes map to Ok/Degraded/Failed)

* **Failed:** no samples for `> failed_timeout` (default 1 s) → the sensor is
  *dead*; the front-end must drop it from the fuse (spec 05 §6 switchable
  constraints). `EmptyScan`/`LidarNoPointTime` persistently → Failed.
* **Degraded:** any of `SyncLost`, `SyncResidualHigh`, `SwOffset` when
  hardware-sync was configured, `RateLow/High` beyond ±20 % nominal,
  `ImuNoDeviceClock`, `CamNoExposure`, `GnssPpsLost`, `LidarHighNanRatio`. Data
  still usable but down-weighted; `offset_std_ns` rises and flows to L2.
* **Ok:** stamping at the configured best mechanism, rate within band, sync
  residual within bound.

The level **flows into the estimator**, not just the screen: `offset_std_ns`
inflates the time-jitter component of a sensor's measurement noise in L2, and a
`Failed` sensor is excluded from the constraint set. This closes the loop between
L0's self-assessment and the fuse — the architectural reason health lives in the
core and not only in the wrapper.

### 9.4 The standing `InputValidator` (live stamp-integrity gate)

The health codes above are *raised by* a single standing component every sample of
every stream passes through before it is handed to the callback: the
`InputValidator`. It is the engine behind `SourceBase::enforce_monotonic` and
`note_sample` and the per-modality content check (`on_lidar`/`on_imu`, §3.3, §4.1),
pulled out as one named gate so the checks are identical across all four modalities
and so the *one* place that decides "this stamp is trustworthy" is testable in
isolation. It runs **always-on, live** — not a
debug-build option — because a single out-of-order or skewed stamp does not merely
add one noisy measurement: it corrupts the deskew warp for every point in the
sweep and the integration interval of every IMU sample that brackets it.
Validating stamp
integrity at the source is therefore a correctness obligation, not instrumentation.

```cpp
// meridian_sensors/src/input_validator.hpp  (private; one instance per sensor_id)
class InputValidator {
public:
  InputValidator(SensorInfo, const ValidatorCfg&, HealthSink*);
  // Called by SourceBase for every sample, before the callback. Returns the
  // (possibly clamped) stamp and an accept/clamp/reject verdict; updates health.
  struct Verdict { Timestamp stamp; enum { Accept, Clamped, Reject } kind; };
  Verdict on_stamp(Timestamp proposed, StampSource src);
  // Content checks for the typed sample (per-point time, NaN/Inf ratio, emptiness).
  void    on_lidar(const LidarScan&);   // sets reject on F6/F11 conditions
  void    on_imu(const ImuSample&);     // NaN/Inf on acc/gyro
};
```

The validator runs **six standing checks**; each maps to a `HealthCode` (§9.1) and
is **edge-throttled** ("dup-filtered"): the *first* sample entering a fault state
emits one `event` and sets the code bit; subsequent samples in the same state are
counted into a `<code>_count` scalar but emit **no** further events, and exit emits
one clearing event (§3.2 edge-throttle policy). This is what stops a sustained
fault — a mis-wired clock, a sensor stuck off-rate — from drowning the event stream
in one message per sample while still surfacing the condition immediately and
keeping an exact occurrence count for forensics.

| Check | What it catches | Signal | Action | Code |
|---|---|---|---|---|
| **Rewind** | stamp not strictly non-decreasing per `sensor_id` (clock step back, replay seam, driver re-order) | `proposed <= last_stamp` | clamp to `last_stamp` (`Clamped`), re-anchor `ClockModel.t_ref` on a large jump; never emit a regressing stamp (B2) | `ClockStepDetected` |
| **Gap** | a hole in the stream (dropout) | raw inter-sample gap `> k/nominal_rate` (`k = gap_periods`, default 2.5), or `> failed_timeout_ms` when `nominal_rate_hz` unset | accept the post-gap sample; raise `Dropout`, escalate to `Failed` past `failed_timeout` | `Dropout` |
| **Skew** | crystal drift beyond the model's validity (a fault, not normal ppm) | `|ClockModel.skew_ppm|` over `skew_warn_ppm` (default 200) or a per-update offset implying `> offset.reject_jump_ms` | flag; the implied-jump correspondence is rejected as an outlier (§7.4) rather than absorbed | `SkewOutOfRange` |
| **Per-point time** | a LiDAR scan that cannot be deskewed | all `t_offset_ns == 0` / field absent | **reject** the scan (`Reject`); do not reconstruct (§6.1, F6) | `LidarNoPointTime` |
| **NaN/Inf** | corrupt sample content | `!isfinite` on any LiDAR `xyz` (ratio over `nan_ratio_warn`, default 0.05) or any IMU `acc`/`gyro` component | drop the offending points / reject an all-NaN scan / reject the IMU sample | `LidarHighNanRatio` / `EmptyScan` |
| **Empty** | a scan with no points | `points` empty (raw, or after the NaN drop) | reject (`Reject`) | `EmptyScan` |

The validator **does not** apply geometric or quality filtering — blind radius,
decimation, intensity gating are all L1 (spec 03). Its remit is exactly *stamp
integrity and sample well-formedness*: the minimum a sample must satisfy to be a
legal input to the single timeline. A `Reject` verdict means the sample never
reaches the callback (and never reaches `Q_sensors`); a `Clamped` verdict means the
stamp was repaired to preserve monotonicity and the repair was reported. The
content checks (`on_lidar`/`on_imu`) are the live, always-on counterparts of the
camera under/over-exposure and IMU saturation checks L1 adds downstream — L0 owns
only the timing- and finiteness-level guarantees here.

`ValidatorCfg` lives under `meridian.time.validator` (§10). The validator holds no
state beyond `last_stamp` and the active-code bitset per sensor, so it is cheap and
thread-confined to its source's acquisition thread (the bitset is read by the
health rollup under the same lightweight lock the `ClockModel` table uses, §13).

---

## 10. Configuration schema

Extends `meridian.sensors` / `meridian.time` (arch §8.2). Typed, validated on load
(arch §8.3). Mirrors and supersedes FAST-LIO's `common`/`preprocess` keys
(`lid_topic`, `imu_topic`, `time_sync_en`, `lidar_type`, `blind` — note `blind`
and filtering move to L1/spec 03). One LiDAR, one IMU, one camera, one GNSS:

```yaml
meridian:
  time:
    source: ptp            # ptp | pps | host   (host = no hardware sync, sw-offset only)
    ptp:
      domain: 0
      iface: eth0          # NIC running ptp4l
      require_lock: false  # if true, refuse to start until PTP locked
    pps:
      device: /dev/pps0
      expect_hz: 1
    swoffset:              # §7 — always on (verify when hw-synced, estimate when not)
      enable: true
      xcorr: { window_s: 4.0, grid_ms: 0.5, motion_gate_radps2: 0.05 }
      kf:    { skew_window_s: 30, Q_offset_ns2_per_s: 1e4, R_default_ns: 500 }
      verify_residual_warn_us: 200
      reject_jump_ms: 5
    health:
      failed_timeout_ms: 1000
      rate_tolerance_frac: 0.20
    validator:               # §9.4 — standing live stamp-integrity gate, always on
      gap_periods: 2.5       # dropout when raw gap > gap_periods / nominal_rate_hz
      skew_warn_ppm: 200     # |ClockModel.skew_ppm| above this raises SkewOutOfRange
      nan_ratio_warn: 0.05   # LiDAR scan NaN/Inf fraction above this raises LidarHighNanRatio
  sensors:
    lidar:   { id: 0, name: main, frame: os_sensor0, model: ouster_os0_128,
               topic: /os/points, nominal_rate_hz: 10,
               ptp: true, timestamp_mode: TIME_FROM_PTP_1588 }
    imu:     { id: 0, name: main, frame: imu_link, model: bmi085,
               topic: /imu/data_raw, nominal_rate_hz: 200,
               has_device_clock: true, interval_end_shift_ns: 0 }
    camera:  { id: 0, name: front, frame: cam_link, model: alphasense,
               topic: /cam/image_raw, nominal_rate_hz: 20,
               trigger: gpio, exposure_from_meta: true, shutter: global }
    gnss:    { id: 0, name: rover, frame: gnss_link, topic: /gnss/fix,
               pps_disciplines_clock: true, enable: true }
  aggregation:
    max_wait_ms: 150         # = ~1.5 sweep periods at 10 Hz
    reorder_ms: 20
```

**Validation rules** (`Config::validate`, arch §8.3): every `lidar/imu/camera/
gnss` `id` is set; if `time.source == ptp` then at least one sensor has
`ptp: true`; `nominal_rate_hz > 0` for each sensor; `max_wait_ms >=
1000/lidar.nominal_rate_hz`; `camera.shutter == global` (the rolling-shutter path
is designed but not built — §4.3 — so `shutter: rolling` is rejected at load rather
than stamped as if global).

---

## 11. Failure modes and recovery

| # | Failure | Detection | L0 response | Health |
|---|---|---|---|---|
| F1 | **PTP unlock** (transient) | `ptp4l` reports unlocked / `ptp_offset_ns` spikes | switch affected source to `SwOffset` (warm model, §7.3); resume `HwPtp` on relock | Degraded `SyncLost` → Ok |
| F2 | **PTP never locks** (mis-wired) | unlocked for `> require_lock` window | run on `SwOffset` permanently; if `require_lock:true`, refuse start | Degraded `NoSync` |
| F3 | **PPS lost** (GNSS dropout) | no PPS edge for >2 s | timeline drops absolute→relative; PTP keeps running off host (free-wheeling) | Degraded `GnssPpsLost`, `timeline_absolute=false` |
| F4 | **Clock step** (NTP slew, host time jump) | host `t_H` non-monotonic / jump > threshold | clamp (B2), re-anchor `ClockModel.t_ref`, drop the straddling group | Degraded `ClockStepDetected` |
| F5 | **Sensor dropout** | inter-sample gap > $k$/nominal_rate | emit nothing for that stream; aggregator emits group without it after `max_wait_ms` | Degraded `Dropout` → Failed after `failed_timeout` |
| F6 | **LiDAR scan without per-point time** | `t_offset_ns` all zero / missing | **reject the scan** (no deskew possible; §6.1) — do *not* reconstruct (Ouster always has `t`) | Degraded `LidarNoPointTime` |
| F7 | **Camera trigger mismatch** | 0 or >1 edges in match gate (§6.2) | fall to `SwOffset` for that frame; resync edge counter | Degraded `CamTriggerMismatch` |
| F8 | **High software-sync residual** | verify-mode residual > 200 µs while PTP "locked" | keep data, raise warning; L2 inflates that sensor's noise | Degraded `SyncResidualHigh` |
| F9 | **IMU arrival-only** | no device clock | stamp on arrival; loud flag (high-rate jitter) | Degraded `ImuNoDeviceClock` |
| F10 | **Out-of-order arrival** beyond reorder window | stamp < emitted watermark − `reorder_ms` | drop the late sample | Degraded `LateDrop` |
| F11 | **Empty / all-NaN scan** | `points` empty or NaN ratio > thresh | drop scan | Degraded `EmptyScan`/`LidarHighNanRatio` |

Recovery principle: **degrade, never crash; always tell L2 and the operator.**
Every transition is a telemetry `event` (§12). The stamp-integrity rows — F4
(clock step / rewind), F5 (dropout / gap), F6 (no per-point time), F10 (late drop),
F11 (empty / all-NaN) — are detected by the standing `InputValidator` (§9.4) on
every sample of every stream; their events are edge-throttled there so a sustained
fault emits one event plus a running count, never one per sample. Compare the
reference, where the analogous conditions are a silent `continue` or a one-off
`ROS_WARN` (arch §10.2 notes this for the front-end; L0 had no such signals at all).

---

## 12. Debug / introspection hooks

Routed through the `TelemetrySink` bus (arch §10.1) — the core emits structured
data, the wrapper decides how to surface it. L0's contributions:

| Telemetry key | Type | Content | Why (FAST-LIO had none of these at L0) |
|---|---|---|---|
| `time/ptp_offset/<clock>` | `scalar` | PTP offset-from-master [ns] | see PTP quality live |
| `time/pps_present` | `scalar` (0/1) | PPS edge seen this second | absolute-anchor presence |
| `time/offset/<sensor>` | `scalar` | `ClockModel` offset [ns] | per-sensor sw-offset estimate |
| `time/offset_std/<sensor>` | `scalar` | offset 1-σ [ns] | the timing uncertainty fed to L2 |
| `time/skew_ppm/<sensor>` | `scalar` | estimated skew | crystal drift, slow trend |
| `time/sync_residual/<sensor>` | `scalar` | verify-mode residual [ns] | catches "locked but wrong" (§7.3) |
| `time/stamp_source/<sensor>` | `scalar` (enum) | current `StampSource` | provenance O2; step-downs visible |
| `sensors/rate/<sensor>` | `scalar` | measured Hz | dropout/rate band |
| `sensors/health/<sensor>` | `event`/`vec` | `SensorHealth` snapshot | the health channel itself |
| `sensors/group_latency` | `scalar` | `t_now − t_end` at emit [ms] | aggregation backlog |
| `sensors/imu_in_group` | `scalar` | #IMU per `MeasureGroup` | sweep/IMU coverage sanity |
| `sensors/validator/<sensor>/<code>_count` | `scalar` | occurrences of each validator fault while active | the dup-filtered count behind the edge-throttled event (§9.4) |
| `time/xcorr_peak/<pair>` | `scalar` | cross-corr peak value | sw-offset estimator confidence |
| `event time/sync_changed` | `event` | "lidar HwPtp→SwOffset (unlock)" | the §6.1/F1 transition, visible |
| `event sensors/dropout` | `event` | "imu0 dropout 0.43 s" | F5 |
| `marker time/clock_tree` | `marker` | grandmaster→slaves tree, coloured by lock | one-glance sync topology |

`DebugControl` (arch §10.5) can toggle the heavier per-sample scalars; the health
events and `stamp_source` changes are always on (cheap, forensically vital).

---

## 13. Threading, ownership, and lifetime summary

* **Acquisition threads.** Each `ISensorSource` owns one acquisition thread
  (spec 01 §7.1). In the live system these are driven by the wrapper's rclcpp
  callbacks (one per subscription) calling `ingest_raw`; in replay, by
  `BagReplaySource`'s reader thread. Source callbacks push **non-blocking** into
  `Q_sensors` (arch §11.1, B3).
* **Stage T1.** The `Aggregator` runs on the single L0/L1 stage thread T1 (arch
  §11.1), consuming `Q_sensors`, producing `MeasureGroup` into `Q_meas`.
* **`ClockModel` / `HealthSink`** are **shared, borrowed** by every source
  (`ClockModel*`, `HealthSink*`), owned by `MeridianPipeline`. `ClockModel` updates
  come from two threads (OS-time shim + T1's estimator); it guards its
  `ClockState` table with a lightweight lock (updates are infrequent — Hz-rate
  PTP stats, per-correspondence offset updates — so contention is nil).
* **`InputValidator`** (§9.4) is **owned per source** inside `SourceBase` and is
  **thread-confined** to that source's acquisition thread: its `last_stamp_` and
  active-code bitset are touched only on the stamping path. The health rollup reads
  the bitset through `HealthSink` under the same lightweight lock; nothing on the
  hot path blocks.
* **Ownership of buffers.** `LidarScan::points`, `CameraFrame::data` are
  **Shared-immutable** (spec 01 §2.4) the instant the source finishes filling
  them; from then on they cross threads by `shared_ptr<const>` and are never
  mutated — this is what lets the retained keyframe store (MUST-FIX #4) and L1
  reference the same bytes.
* **Lifetime of stamps.** Once a sample leaves a source its stamp is **final and
  immutable**. Corrections to the `ClockModel` change *future* samples only; we
  never retroactively re-stamp emitted samples (that would violate monotonicity
  and the front-end's causal assumptions). A large model correction instead
  surfaces as a `ClockStepDetected` event so L2 can choose to restart its CT
  window (arch §7.4) rather than absorb a discontinuity silently.

---

## 14. Worked example: a 100 ms tick through L0

One Ouster (id 0, PTP-locked) + IMU (id 0, free-running, device clock) + camera
(id 0, GPIO trigger) + GNSS (id 0, PPS), 10 Hz LiDAR, 200 Hz IMU, 20 Hz camera.
The system is initialised; PTP locked; GNSS has fix.

1. **t = 0 ms.** The Ouster begins a sweep; columns stamp on its PTP clock.
2. **0–100 ms.** ~20 IMU samples arrive; `ImuSource` device clock is on
   `SwOffset` (no PTP IMU), so each is stamped via `ClockModel.to_meridian` (offset
   ≈ +37 µs, skew ≈ −12 ppm, σ ≈ 60 µs from §7's verify-warm filter). Two
   camera frames arrive (20 Hz → frames at ≈25 ms and ≈75 ms); `CameraSource`
   matches each to its GPIO edge (gate ±25 ms), adds `exposure_s/2 = 4 ms`,
   stamps `HwTrigger`. One PPS edge at the GPS-second; `GnssSource` emits a
   `GnssFix` stamped `HwPps`, and forwards the edge to `meridian_time`.
3. **t ≈ 100 ms.** The sweep completes: `OusterLidarSource` builds a `LidarScan`,
   `stamp_start` on `t_H` (`HwPtp`), `t_offset_ns` native per column,
   `sweep_duration ≈ 99.7 ms`, `points` Shared-immutable.
4. **Aggregation.** The `Aggregator` (T1) has the sweep `[t_begin,t_end]`; it
   waits until `imu.back().stamp ≥ t_end` (the IMU straddling 100 ms arrives at
   ~102 ms → 2 ms group latency, well under `max_wait_ms=150`). It assembles
   `MeasureGroup`: `scan`=the sweep, `imu`=the ~20 spanning samples + the
   straddler at `t_begin`, `image`=the 75 ms frame (its mid-exposure ∈ interval),
   `gnss`=the one fix. Emits into `Q_meas`.
5. **Health.** All sensors `Ok` except IMU `Degraded(SwOffset)` with
   `offset_std_ns=60000`; `RigHealth.timeline_absolute=true`. Telemetry publishes
   `time/offset/imu0=37000`, `sensors/imu_in_group=21`,
   `sensors/group_latency=2.0`.
6. **Downstream.** L2's LIO front-end warps each LiDAR point by its
   `t_offset_ns` within the sweep's screw model and solves at `t_end`;
   the IMU's 60 µs timing σ inflates its noise slightly. None of L0's stamps are
   ever revisited.

Total L0 work this tick: O(points) lossless copy into Shared-immutable buffers +
O(IMU) interval selection + a handful of clock-model lookups. No filtering, no
deskew, no transforms — those are above.

---

## 15. Test plan

L0 is unusually testable because, per arch §1, it is pure C++ fed by POD frames.

* **Unit: stamping arithmetic.** Feed `RawLidarFrame`/`RawImuFrame` with known
  device clocks and a synthetic `ClockModel` (offset 1 ms, skew 50 ppm);
  assert emitted stamps equal the inverse-mapped values to the ns. Assert native
  `t_offset_ns` pass-through (no scale), the bug-class FAST-LIO's `time_unit_scale`
  invites.
* **Unit: `InputValidator` (§9.4, B6).** Drive each of the six standing checks in
  isolation: inject a backward device step (rewind → clamp + `ClockStepDetected`,
  never a regressing stamp); a stream hole (gap → `Dropout`); skew past
  `skew_warn_ppm` (`SkewOutOfRange`); an all-zero `t_offset_ns` scan
  (`LidarNoPointTime` reject); a NaN-laced cloud and an all-NaN/empty scan
  (`LidarHighNanRatio` / `EmptyScan`). Assert the **dup-filter**: a sustained fault
  emits exactly one entry event and one exit event while `<code>_count` increments
  every sample; assert a `Reject` verdict never reaches the callback.
* **Unit: aggregation policy.** Synthetic interleaved streams; assert the IMU set
  spans `[t_begin,t_end]` incl. straddler, the timeout path
  (`ImuLate`) fires, and reorder/late-drop behave at the window edges.
* **Unit: camera trigger match (§6.2).** Drop a frame; assert gate rejects,
  `CamTriggerMismatch` increments, counter resyncs on the next clean frame.
* **Property: sw-offset estimator (§7).** Generate gyro and LiDAR-yaw signals
  with a known injected lag; assert recovered $\hat\tau$ within `grid_ms` and the
  KF converges; assert verify-mode residual flags when the injected offset
  exceeds threshold.
* **Integration: bag replay.** Run `BagReplaySource` on a Newer College
  sequence (`DATASET.md`); assert deterministic single-thread ordering, and run
  the estimator in verify mode to *report* the dataset's residual sync as a
  sanity check on the data itself.
* **Health state machine.** Drive each F1–F11 condition; assert the documented
  level/code transitions and telemetry events.

---

## Appendix A — types this spec introduces

**Boundary type (canonical declaration is in spec 01 Appendix A; this spec only uses it):**
```
MeasureGroup (struct)              — L1→L2 bundle (sweep + spanning IMU/image/GNSS)  (§8.1)
```

**L0-internal types (defined here; never cross a layer boundary, so they are NOT in spec 01):**
```
StampSource (enum)                 — provenance of a stamp                 (§2.1)
SensorInfo (struct)                — static per-source descriptor          (§3.2)
ClockId (enum), ClockState,        — per-device clock model                (§5.3)
  ClockModel (class)
HealthLevel, HealthCode (enums),   — sensor-health channel                 (§9.1)
  SensorHealth, RigHealth (struct),
  HealthSink (interface)
InputValidator (class, private),   — standing live stamp-integrity gate    (§9.4)
  ValidatorCfg (struct)
SourceBase (impl mix-in, private)  — shared B1–B6 machinery                (§3.3)
RawLidarFrame/RawImuFrame/…(POD)   — wire-free conversion boundary         (§4)
```
(If any L0-internal type ever needs to cross a boundary, it must first be amended
into spec 01 per the §1 rule.)

## Appendix B — L0 ↔ FAST-LIO reference map

```
Meridian L0 concern                     FAST-LIO analogue (file:line)            Meridian change
per-sensor handler                   Preprocess::{avia,oust64,velodyne}_handler  one ISensorSource per modality, SourceBase
                                       (preprocess.cpp:92, 189, 284)             (no god-class)
per-point time                       ouster t / velodyne time → curvature*scale  native int32 t_offset_ns, no scale
                                       (preprocess.cpp:226, 344; preprocess.h:49)
time-unit handling                   TIME_UNIT enum + time_unit_scale            single int64-ns timeline (O1)
                                       (preprocess.h:18; preprocess.cpp:52-68)
recover time when absent             yaw-based reconstruction                     Ouster always has t → reject if absent (F6)
                                       (preprocess.cpp:297, 346-372)
time sync                            time_sync_en boolean                         PTP/PPS chain + sw-offset estimator + verify
                                       (config / common params)                   (§5,§6,§7)
blind / decimation                   range<blind cull, point_filter_num          MOVED to L1 (spec 03); L0 ships raw
                                       (preprocess.cpp:210,260,264)
scan/IMU bundling                    MeasureGroup sync in laserMapping main       Aggregator on T1, bounded+timed (§8)
sensor health                        (none)                                       first-class health channel (§9)
offset estimate / error bar          (none)                                       ClockModel offset+skew+σ (§5.3,§7)
```
