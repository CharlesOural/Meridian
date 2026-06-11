#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "meridian/backend/ibackend.hpp"
#include "meridian/common/bounded_queue.hpp"
#include "meridian/common/cloud.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/loop_constraint.hpp"
#include "meridian/common/measure_group.hpp"
#include "meridian/common/nav_state.hpp"
#include "meridian/common/preprocessed_group.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"
#include "meridian/place/iloop_detector.hpp"
#include "meridian/place/keyframe_store.hpp"
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
// aggregation, cold-start deskew), the L2 front-end (CT LIVO+GNSS estimator), the L3
// back-end (global graph optimizer), and the telemetry sink every module borrows.
//
// Threading: in Live mode three stage threads run. T1 drains the sensor queue and runs
// the L0/L1 stage; each preprocessed sweep plus its live IMU samples cross a second
// bounded queue (Q_meas) into the front-end thread T2, which owns the estimator. T2
// feeds keyframes, anchored GNSS fixes, and loop closures through a lossless queue
// (Q_backend) into the back-end thread T3, which owns the global optimizer and folds
// staged items on the configured cadence. Each material correction is parked in a
// single slot that T2 drains strictly between sweep ingests, so apply_correction()
// never interleaves with an ingest. ingest() is callable from any thread and never
// blocks (the sensor and measurement queues drop oldest under overload and the drop is
// counted; Q_backend never drops — its depth is the overload signal). In Replay mode
// there is no thread and no queue: ingest() processes synchronously on the caller's
// thread, the front-end and back-end run inline (the back-end folds after every
// keyframe), so a replay is deterministic.
//
// Lifetime: construction wires modules in dependency order; destruction stops the stage
// threads (sensor stage, then front-end, then back-end — flow order, so each closed
// edge has no producer left) and tears down in reverse. The wrapper owns exactly one
// pipeline.
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

  // Spawns the back-end, front-end, and sensor stage threads (Live mode). No-op in
  // Replay mode.
  void start();
  // Closes the queues, drains them, and joins the stage threads. Idempotent.
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

  // A GNSS fix on its way to the back-end, tagged with the id of the freshest keyframe
  // at the time the fix's sweep was ingested (the fix's anchor in the pose graph).
  struct GnssForBackend {
    GnssFix fix;
    std::uint64_t nearest_kf_id = 0;
  };
  // One item of the back-end input stream: a keyframe, a verified loop closure, or a
  // tagged GNSS fix.
  using BackendItem = std::variant<KeyframePacket, LoopConstraint, GnssForBackend>;
  // Observes every back-end input item in feed order, on the front-end thread (Live) or
  // the caller's thread (Replay). Set before start(); must not block.
  using BackendTap = std::function<void(const BackendItem&)>;
  void set_backend_tap(BackendTap tap);

  // Receives every graph update the back-end publishes (one per fold), on the back-end
  // thread (Live) or the caller's thread (Replay). Set before start(); must not block.
  using GraphUpdateSink = std::function<void(const GraphUpdate&)>;
  void set_graph_update_sink(GraphUpdateSink sink);

  // The back-end's corrected map-frame trajectory; empty when the back-end is disabled.
  // Replay: callable between ingests or after stop(). Live: only after stop().
  std::vector<StampedPose> corrected_trajectory() const;

  // T_map_odom: the correction between the back-end map frame and the front-end odom frame
  // (identity when there is no back-end). For broadcasting the map->odom TF in viz tooling.
  Pose map_odom() const;

  // False when the config disabled the back-end and the pipeline runs without it.
  bool backend_enabled() const;

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
  // The single feed point of the back-end input stream; every item passes through here
  // (and the tap, when set) in feed order.
  void feed_backend(BackendItem&& item);
  // Stages one item into the back-end (thread-confined to the back-end driver: T3 in
  // Live, the caller's thread in Replay). Returns true when the staged batch should be
  // folded without waiting for the cadence timer.
  bool stage_backend_item(BackendItem&& item);
  // Drains Q_backend and folds staged batches on the optimize cadence (the back-end
  // thread body, Live only).
  void backend_loop();
  // Surfaces post-fold diagnostics, parks a material correction for the front-end, and
  // forwards the update to the wrapper sink.
  void publish_graph_update(const GraphUpdate& gu);
  // Applies a parked correction to the front-end. Runs strictly between ingests on the
  // front-end thread (the caller's in Replay), never from inside the keyframe sink.
  void drain_pending_correction();
  // Offers the keyframes staged since the last fold to the loop detector (with their now
  // corrected poses) and returns any verified loops. Runs on the back-end driver after a
  // fold, so the poses it reads are deterministic. Empty when L5 is disabled.
  std::vector<LoopConstraint> detect_loops();
  // Assembles the back-end map: every retained keyframe cloud placed at its corrected map pose,
  // voxel-downsampled, published in the map frame. Throttled, and forced on a loop closure (when
  // the map de-warps). No-op without the keyframe store / back-end / an enabled sink key.
  void publish_map_cloud(Timestamp ts, bool force);
  // Measurement-time anchor for back-end-path telemetry sampled outside a measurement
  // callback (queue depth, post-optimize diagnostics): the last keyframe's stamp, or 0
  // before the first keyframe. Data-driven, not wall-clock, so replay stays deterministic.
  Timestamp tele_stamp() const;

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
  std::unique_ptr<IBackEnd> backend_;  // null when the config disabled the back-end
  // L5 loop detection (null when disabled). The store retains keyframe clouds for the
  // detector; pose_source_ gives it read-only corrected poses/chain-cov so it never links
  // the back-end. Declared after backend_ so it is destroyed first (it borrows backend_).
  std::shared_ptr<KeyframeStore> store_;
  std::unique_ptr<ILoopDetector> loop_detector_;
  KeyframePoseSource pose_source_;

  BoundedQueue<SensorSample> q_sensors_;
  // Front-end ingest edge: sweeps + live IMU. Lossy under overload, oldest dropped, so a
  // slow front-end never back-pressures the sensor stage.
  BoundedQueue<MeasSample> q_meas_;
  // Back-end input edge: keyframes, anchored GNSS, loops. Lossless — every item must
  // reach the graph — so the queue grows past capacity under overload and the depth is
  // the signal, not a drop count.
  BoundedQueue<BackendItem> q_backend_;

  // Constant per-sensor stamp corrections onto the body-IMU timeline [ns], applied
  // once in ingest() before validation/aggregation. Zero means no correction.
  Timestamp lidar_offset_ns_ = 0;
  Timestamp camera_offset_ns_ = 0;
  // Sweeps held back until the IMU bootstrap converges; bounded, oldest dropped.
  std::deque<MeasureGroup> bootstrap_groups_;

  GroupSink group_sink_;
  KeyframeSink keyframe_sink_;
  BackendTap backend_tap_;
  GraphUpdateSink graph_update_sink_;
  // Id of the most recent keyframe, the anchor GNSS fixes are tagged with; empty until
  // the first keyframe. Thread-confined to the front-end thread (caller's in Replay).
  std::optional<std::uint64_t> last_kf_id_;
  std::atomic<std::uint64_t> keyframe_count_{0};
  // Keyframe stamps, for the loop detector's time-gap gate; confined to the back-end driver.
  std::unordered_map<std::uint64_t, Timestamp> kf_stamps_;
  // Keyframes staged into the back-end since the last fold, with their clouds, awaiting
  // offer to the detector at corrected poses. Confined to the back-end driver.
  std::vector<std::pair<std::uint64_t, PointCloudPtr>> pending_kf_for_detector_;
  // Folds since the assembled map cloud was last published (throttle). Back-end driver only.
  std::uint64_t folds_since_map_cloud_ = 0;
  // Items staged into the back-end since its last fold (Replay's inline driver only;
  // the Live count lives in backend_loop()).
  std::uint64_t staged_since_opt_ = 0;
  // Latch for the Q_backend overload warning: raised when the depth first crosses the
  // threshold, re-armed once it falls back under. Confined to the front-end thread.
  bool backend_queue_over_ = false;
  // Single-slot hand-off of the freshest material correction from the back-end driver
  // to the front-end thread; a newer update overwrites an undrained one.
  std::mutex correction_mu_;
  std::optional<GraphUpdate> pending_correction_;
  std::thread stage_thread_;
  std::thread frontend_thread_;
  std::thread backend_thread_;
  std::atomic<bool> running_{false};
};

}  // namespace meridian
