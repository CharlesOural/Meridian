#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <variant>

#include "meridian/common/bounded_queue.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/measure_group.hpp"
#include "meridian/common/nav_state.hpp"
#include "meridian/common/preprocessed_group.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"
#include "meridian/sensors/raw_frames.hpp"

namespace meridian {

class Aggregator;
class CameraPreprocessor;
class CameraSource;
class ClockModel;
class GnssGate;
class GnssSource;
class HealthSink;
class IFrontEnd;
class ILidarPreprocessor;
class ImuInitializer;
class ImuSource;
class OusterLidarSource;
class TelemetrySink;

// Owns and wires the processing stages: the sensor sources (stamping + health), the
// lossy sensor queue, the L0/L1 stage (validity filter, IMU init, GNSS gate,
// aggregation, cold-start deskew), the L2 front-end (CT LIVO+GNSS estimator), and the
// telemetry sink every module borrows.
//
// Threading: in Live mode two stage threads run. T1 drains the sensor queue and runs
// the L0/L1 stage; each preprocessed sweep plus its live IMU samples cross a second
// bounded queue (Q_meas) into the front-end thread T2, which owns the estimator.
// ingest() is callable from any thread and never blocks (each queue drops oldest under
// overload and the drop is counted). In Replay mode there is no thread and no queue:
// ingest() processes synchronously on the caller's thread and the front-end runs inline,
// so a replay is deterministic.
//
// Lifetime: construction wires modules in dependency order; destruction stops the stage
// threads (front-end first, then sensor stage) and tears down in reverse. The wrapper
// owns exactly one pipeline.
class MeridianPipeline {
public:
  // `sink` is the telemetry sink for the whole pipeline; the pipeline takes ownership
  // and outlives every module that borrows it. nullptr installs a NullSink. `cfg` must
  // already be validated. In Live mode the sink is written from both the ingest threads
  // and the stage thread, so it must be thread-safe; the non-locking RecordingSink and
  // NullSink are only valid for Replay mode and tests.
  MeridianPipeline(const Config& cfg, std::unique_ptr<TelemetrySink> sink);
  ~MeridianPipeline();

  MeridianPipeline(const MeridianPipeline&) = delete;
  MeridianPipeline& operator=(const MeridianPipeline&) = delete;

  // Spawns the sensor stage and front-end threads (Live mode). No-op in Replay mode.
  void start();
  // Closes the queues, drains them, and joins both stage threads. Idempotent.
  void stop();

  // Wrapper ingest: one wire-free frame per call. Non-blocking; safe from any thread.
  void ingest(const RawLidarFrame& f);
  void ingest(const RawImuFrame& f);
  void ingest(const RawCameraFrame& f);
  void ingest(const RawGnssFrame& f);

  // Receives every preprocessed sweep, on the front-end thread (Live) or the caller's
  // thread (Replay), before the front-end ingests it. Set before start().
  using GroupSink = std::function<void(PreprocessedGroup&&)>;
  void set_group_sink(GroupSink sink);

  // Receives every keyframe the front-end emits, moved, on the front-end's keyframe
  // finalizer thread (Live; packets arrive in id order with constraint_cov filled) or
  // the caller's thread (Replay, inline). Set before start(); must only enqueue. When
  // unset the pipeline still counts keyframes and raises the "frontend/keyframe"
  // telemetry event. The first keyframe flips the front-end into steady-state deskew,
  // so emitted groups stop reporting cold_start after it.
  using KeyframeSink = std::function<void(KeyframePacket&&)>;
  void set_keyframe_sink(KeyframeSink sink);

  // The front-end's smooth odom-frame estimate at the latest valid time. Thread-confined
  // to the front-end thread; the group sink runs there, so reading this from within a
  // group-sink callback is safe.
  NavState live_state() const;

  // The sink every module writes to (borrowed; owned by the pipeline).
  TelemetrySink* telemetry();

private:
  using SensorSample = std::variant<ImuSample, LidarScan, CameraFrame, GnssFix>;
  // What crosses Q_meas into the front-end thread: a finished sweep, or one live IMU
  // sample to advance live_state() between sweeps.
  using MeasSample = std::variant<PreprocessedGroup, ImuSample>;

  // Routes a stamped sample from a source callback into the stage: synchronously in
  // Replay mode, through the lossy queue in Live mode.
  void enqueue(SensorSample&& s);
  // Drains the sensor queue until close (the sensor stage thread body).
  void stage_loop();
  // One sample through the L0/L1 stage (thread-confined to the sensor stage thread).
  void process(SensorSample&& s);
  // Aggregated-group tail: bootstrap buffering, cold-start deskew, telemetry, emit.
  void on_group(MeasureGroup&& g);
  // Deskew + telemetry + hand-off of one group past the bootstrap gate.
  void emit_group(MeasureGroup&& g);
  // Deskews one group's scan against an IMU-only trajectory built from its IMU set.
  std::optional<LidarScan> deskew_group(const MeasureGroup& g);

  // Hands a measurement to the front-end stage: through Q_meas in Live mode, inline in
  // Replay mode (deterministic, on the caller/sensor thread).
  void dispatch_to_frontend(MeasSample&& m);
  // Drains Q_meas until close (the front-end thread body).
  void frontend_loop();
  // Runs the group sink then feeds one measurement to the front-end (thread-confined to
  // the front-end thread, or the caller's thread in Replay).
  void process_meas(MeasSample&& m);
  // The front-end's keyframe callback: counts, raises telemetry, flips spline_active_ on
  // the first keyframe, and forwards to the wrapper sink.
  void on_keyframe(KeyframePacket&& kf);

  Config cfg_;
  bool sync_mode_ = false;  // Replay: process inline, no thread/queue
  // Flipped true once the front-end takes over deskew with its own trajectory; until
  // then every emitted group reports cold_start. Written on the front-end thread (first
  // keyframe), read on the sensor stage thread, so it is atomic.
  std::atomic<bool> spline_active_{false};

  std::unique_ptr<TelemetrySink> sink_;
  std::unique_ptr<ClockModel> clock_;
  std::unique_ptr<HealthSink> health_;

  std::unique_ptr<OusterLidarSource> lidar_source_;
  std::unique_ptr<ImuSource> imu_source_;
  std::unique_ptr<CameraSource> camera_source_;
  std::unique_ptr<GnssSource> gnss_source_;

  std::unique_ptr<ILidarPreprocessor> lidar_preprocessor_;
  std::unique_ptr<ImuInitializer> imu_init_;
  std::unique_ptr<GnssGate> gnss_gate_;
  std::unique_ptr<CameraPreprocessor> camera_preprocessor_;
  std::unique_ptr<Aggregator> aggregator_;
  std::unique_ptr<IFrontEnd> frontend_;

  BoundedQueue<SensorSample> q_sensors_;
  // Front-end ingest edge: sweeps + live IMU. Lossy under overload, oldest dropped, so a
  // slow front-end never back-pressures the sensor stage.
  BoundedQueue<MeasSample> q_meas_;

  // Constant per-sensor stamp corrections onto the body-IMU timeline [ns], applied
  // once in ingest() before validation/aggregation. Zero means no correction.
  Timestamp lidar_offset_ns_ = 0;
  Timestamp camera_offset_ns_ = 0;
  // Sweeps held back until the IMU bootstrap converges; bounded, oldest dropped.
  std::deque<MeasureGroup> bootstrap_groups_;

  GroupSink group_sink_;
  KeyframeSink keyframe_sink_;
  std::atomic<std::uint64_t> keyframe_count_{0};
  std::thread stage_thread_;
  std::thread frontend_thread_;
  std::atomic<bool> running_{false};
};

}  // namespace meridian
