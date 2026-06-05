#pragma once

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <variant>

#include "meridian/common/measure_group.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"
#include "meridian/pipeline/bounded_queue.hpp"
#include "meridian/sensors/raw_frames.hpp"

namespace meridian {

class Aggregator;
class CameraPreprocessor;
class CameraSource;
class ClockModel;
class GnssGate;
class GnssSource;
class HealthSink;
class ILidarPreprocessor;
class ImuInitializer;
class ImuSource;
class OusterLidarSource;
class TelemetrySink;

// One preprocessed sweep leaving the L0/L1 stage: the assembled MeasureGroup plus, once
// the IMU bootstrap has converged, the cold-start deskewed scan (points expressed in
// the body frame at the sweep end). `cold_start` stays true until a front-end takes
// over deskew with its own trajectory.
struct PreprocessedGroup {
  MeasureGroup group;
  std::optional<LidarScan> deskewed;
  bool cold_start = true;
};

// Owns and wires the processing stages: the sensor sources (stamping + health), the
// lossy sensor queue, the L0/L1 stage (validity filter, IMU init, GNSS gate,
// aggregation, cold-start deskew), and the telemetry sink every module borrows.
//
// Threading: in Live mode a single stage thread drains the sensor queue; ingest() is
// callable from any thread and never blocks (the queue drops oldest under overload and
// the drop is counted). In Replay mode there is no thread and no queue: ingest()
// processes synchronously on the caller's thread, so a replay is deterministic.
//
// Lifetime: construction wires modules in dependency order; destruction stops the
// stage thread and tears down in reverse. The wrapper owns exactly one pipeline.
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

  // Spawns the stage thread (Live mode). No-op in Replay mode.
  void start();
  // Closes the queue, drains it, and joins the stage thread. Idempotent.
  void stop();

  // Wrapper ingest: one wire-free frame per call. Non-blocking; safe from any thread.
  void ingest(const RawLidarFrame& f);
  void ingest(const RawImuFrame& f);
  void ingest(const RawCameraFrame& f);
  void ingest(const RawGnssFrame& f);

  // Receives every preprocessed sweep, on the stage thread (Live) or the caller's
  // thread (Replay). Set before start().
  using GroupSink = std::function<void(PreprocessedGroup&&)>;
  void set_group_sink(GroupSink sink);

  // The sink every module writes to (borrowed; owned by the pipeline).
  TelemetrySink* telemetry();

 private:
  using SensorSample = std::variant<ImuSample, LidarScan, CameraFrame, GnssFix>;

  // Routes a stamped sample from a source callback into the stage: synchronously in
  // Replay mode, through the lossy queue in Live mode.
  void enqueue(SensorSample&& s);
  // Drains the queue until close (the stage thread body).
  void stage_loop();
  // One sample through the L0/L1 stage (thread-confined to the stage thread).
  void process(SensorSample&& s);
  // Aggregated-group tail: bootstrap buffering, cold-start deskew, telemetry, emit.
  void on_group(MeasureGroup&& g);
  // Deskew + telemetry + hand-off of one group past the bootstrap gate.
  void emit_group(MeasureGroup&& g);
  // Deskews one group's scan against an IMU-only trajectory built from its IMU set.
  std::optional<LidarScan> deskew_group(const MeasureGroup& g);

  Config cfg_;
  bool sync_mode_ = false;  // Replay: process inline, no thread/queue
  // Flipped true once the front-end takes over deskew with its own trajectory; until
  // then every emitted group reports cold_start.
  bool spline_active_ = false;

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

  BoundedQueue<SensorSample> q_sensors_;
  // Sweeps held back until the IMU bootstrap converges; bounded, oldest dropped.
  std::deque<MeasureGroup> bootstrap_groups_;

  GroupSink group_sink_;
  std::thread stage_thread_;
  std::atomic<bool> running_{false};
};

}  // namespace meridian
