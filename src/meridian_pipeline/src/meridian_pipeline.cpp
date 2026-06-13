#include "meridian/pipeline/meridian_pipeline.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include "health_bridge.hpp"
#include "meridian/calib/calibration_from_config.hpp"
#include "meridian/calib/extrinsic.hpp"
#include "meridian/calib/intrinsics.hpp"
#include "meridian/debug/telemetry.hpp"
#include "meridian/debug/telemetry_keys.hpp"
#include "meridian/frontend/ifrontend.hpp"
#include "meridian/preprocess/camera_preprocess.hpp"
#include "meridian/preprocess/gnss_gate.hpp"
#include "meridian/preprocess/ilidar_preprocessor.hpp"
#include "meridian/sensors/aggregator.hpp"
#include "meridian/sensors/sources.hpp"
#include "meridian/time/clock_model.hpp"

namespace meridian {

namespace {

constexpr std::size_t kSensorQueueCapacity = 512;
// Q_meas holds at most a few sweeps plus their interleaved live IMU; one 10 Hz sweep
// carries ~20 IMU samples, so a small multiple of that absorbs short stalls without
// letting the front-end fall far behind real time.
constexpr std::size_t kMeasQueueCapacity = 8 * 24;

// Assembled map-cloud cadence and resolution. The cloud is rebuilt from all stored
// keyframes once every kMapCloudFoldPeriod folds (or immediately on a loop fold), then
// collapsed to a single voxel grid at kMapCloudVoxel metres so its size stays bounded.
constexpr std::uint64_t kMapCloudFoldPeriod = 5;
constexpr float kMapCloudVoxel = 0.3f;
// The L4 surface mesh is large (re-extracted whole each time), so it publishes less often
// than the map cloud to keep the per-publish transport cost off the operator link.
constexpr std::uint64_t kMeshFoldPeriod = 10;

SensorInfo make_info(std::uint8_t id, Modality modality, Frame frame, const std::string& model,
                     double rate_hz, StampSource src) {
  SensorInfo info;
  info.id = id;
  info.modality = modality;
  info.sensor_frame = frame;
  info.model = model;
  info.nominal_rate_hz = rate_hz;
  info.configured_stamp_source = src;
  return info;
}

// The LiDAR->body extrinsic from config, packaged for the validity filter.
Extrinsic lidar_extrinsic(const LidarSensorConfig& lidar) {
  Extrinsic ext;
  ext.child = Frame::OsSensor0;
  ext.parent = Frame::ImuLink;
  ext.T_parent_child = Pose{Eigen::Quaterniond(lidar.extrinsic_R), lidar.extrinsic_T};
  return ext;
}

}  // namespace

MeridianPipeline::MeridianPipeline(const Config& cfg, std::unique_ptr<TelemetrySink> sink)
    : cfg_(cfg),
      sync_mode_(cfg.pipeline.mode == PipelineMode::Replay),
      sink_(sink ? std::move(sink) : std::make_unique<NullSink>()),
      clock_(std::make_unique<ClockModel>()),
      health_(std::make_unique<TelemetryHealthBridge>(sink_.get())),
      q_sensors_(kSensorQueueCapacity),
      q_meas_(kMeasQueueCapacity),
      q_backend_(static_cast<std::size_t>(cfg.pipeline.queue.kf_capacity)),
      lidar_offset_ns_(
          static_cast<Timestamp>(std::llround(cfg.sensors.lidar.time_offset_ms * 1e6))),
      camera_offset_ns_(
          static_cast<Timestamp>(std::llround(cfg.sensors.camera.time_offset_ms * 1e6))) {
  const auto& s = cfg_.sensors;
  const auto& th = cfg_.time.health;
  const auto& vc = cfg_.time.validator;

  lidar_source_ = std::make_unique<OusterLidarSource>(
      make_info(static_cast<std::uint8_t>(s.lidar.id), Modality::Lidar, Frame::OsSensor0,
                s.lidar.model, s.lidar.nominal_rate_hz,
                s.lidar.ptp ? StampSource::HwPtp : StampSource::SwOffset),
      s.lidar, clock_.get(), health_.get(), sink_.get(), th, vc);
  imu_source_ = std::make_unique<ImuSource>(
      make_info(static_cast<std::uint8_t>(s.imu.id), Modality::Imu, Frame::ImuLink, s.imu.model,
                s.imu.rate_hz,
                s.imu.has_device_clock ? StampSource::HwPtp : StampSource::ArrivalOnly),
      s.imu, clock_.get(), health_.get(), sink_.get(), th, vc);
  camera_source_ = std::make_unique<CameraSource>(
      make_info(static_cast<std::uint8_t>(s.camera.id), Modality::Camera, Frame::CamLink,
                s.camera.model, s.camera.nominal_rate_hz, StampSource::SwOffset),
      s.camera, clock_.get(), health_.get(), sink_.get(), th, vc);
  gnss_source_ = std::make_unique<GnssSource>(
      make_info(static_cast<std::uint8_t>(s.gnss.id), Modality::Gnss, Frame::GnssLink, "gnss", 1.0,
                s.gnss.pps_disciplines_clock ? StampSource::HwPps : StampSource::SwOffset),
      s.gnss, clock_.get(), health_.get(), sink_.get(), th, vc);

  // A defaulted (identity) LiDAR extrinsic is a valid-looking value the estimator
  // will silently consume as a real calibration; surface it loudly once at startup.
  if (!s.lidar.extrinsic_set) {
    sink_->event(Level::Warn, keys::sensors::LidarExtrinsicDefault,
                 "lidar extrinsic not configured; using identity T_imu_lidar", 0);
  }

  lidar_preprocessor_ = makeLidarPreprocessor(cfg_.preprocess, lidar_extrinsic(s.lidar),
                                              s.lidar.nominal_rate_hz, sink_.get());
  gnss_gate_ = std::make_unique<GnssGate>(cfg_.preprocess.gnss,
                                          /*velocity_source=*/nullptr, sink_.get());
  // The full camera intrinsics (distortion model + coeffs + size) drive the undistort
  // map; the same calibration snapshot is shared with L2/L3 below.
  const auto calib = calibrationFromConfig(cfg_.sensors);
  const auto cam_it = calib->cam_intrinsics.find(static_cast<std::uint8_t>(s.camera.id));
  const IntrinsicsCamera cam_intr =
      cam_it != calib->cam_intrinsics.end() ? cam_it->second : IntrinsicsCamera{};
  camera_preprocessor_ = std::make_unique<CameraPreprocessor>(
      cfg_.preprocess.camera, cam_intr, sink_.get());
  aggregator_ =
      std::make_unique<Aggregator>(cfg_.aggregation, cfg_.sensors, health_.get(), sink_.get());
  aggregator_->set_sink([this](MeasureGroup&& g) { on_group(std::move(g)); });

  // Every stamped sample funnels into the one stage entry, whatever its modality.
  lidar_source_->set_callback([this](LidarScan&& scan) { enqueue(std::move(scan)); });
  imu_source_->set_callback([this](ImuSample&& imu) { enqueue(std::move(imu)); });
  camera_source_->set_callback([this](CameraFrame&& frame) { enqueue(std::move(frame)); });
  gnss_source_->set_callback([this](GnssFix&& fix) { enqueue(std::move(fix)); });

  // The L2 estimator. A construction failure (bad config) is fatal — the pipeline has
  // no useful output without it — so surface it as a clear exception rather than
  // running a half-wired pipeline.
  try {
    // The path-sample cadence is authored under debug: (a telemetry knob, not an
    // estimator parameter); hand it to the front-end through its config copy. The
    // calibration snapshot is shared with the back-end so both layers agree.
    FrontendConfig fe_cfg = cfg_.frontend;
    fe_cfg.debug_path_sample_hz = cfg_.debug.path_sample_hz;
    frontend_ = makeFrontEnd(fe_cfg, calib, sink_.get(), /*deterministic=*/sync_mode_);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("front-end construction failed: ") + e.what());
  }
  frontend_->set_keyframe_sink([this](KeyframePacket&& kf) { on_keyframe(std::move(kf)); });

  // The L3 global optimizer, sharing the front-end's calibration snapshot so both
  // layers agree on the extrinsics until a refinement publishes a new set.
  if (cfg_.backend.enable) {
    backend_ = makeBackEnd(cfg_.backend, calib, sink_.get(), /*deterministic=*/sync_mode_);
    // The keyframe store feeds two consumers: the loop detector and the assembled map
    // cloud. Create it when either is live, so the map cloud does not silently require
    // loop closure to be on.
    if (cfg_.place.enable || sink_->wants_clouds()) {
      store_ = std::make_shared<KeyframeStore>();
    }
    // L5 loop detector, driven on the back-end path. It reads corrected poses and the
    // odometry-chain covariance through read-only callbacks so it never links the back-end.
    if (cfg_.place.enable) {
      IBackEnd* be = backend_.get();
      pose_source_.pose = [be](std::uint64_t id) { return be->pose_of(id); };
      pose_source_.chain_cov = [be](std::uint64_t a, std::uint64_t b) {
        return be->chain_cov_between(a, b);
      };
      pose_source_.stamp = [this](std::uint64_t id) -> std::optional<Timestamp> {
        const auto it = kf_stamps_.find(id);
        if (it == kf_stamps_.end()) return std::nullopt;
        return it->second;
      };
      loop_detector_ = makeLoopDetector(cfg_.place, store_, pose_source_,
                                        /*deterministic=*/sync_mode_, sink_.get());
    }
    // L4 layered map (Tier R + the selected ISurfaceMap backend), sharing the canonical
    // store. It integrates keyframes at corrected back-end poses and re-integrates on a
    // loop correction; the surface mesh is published to the operator.
    if (cfg_.map.enable) {
      if (!store_) store_ = std::make_shared<KeyframeStore>();
      map_ = makeMapLayer(cfg_.map, calib, store_, sink_.get());
    }
  }
}

MeridianPipeline::~MeridianPipeline() {
  stop();
}

void MeridianPipeline::start() {
  if (sync_mode_ || running_.load()) return;
  running_.store(true);
  // Consumers spawn upstream-last so every edge has its consumer before any producer.
  if (backend_) backend_thread_ = std::thread([this] { backend_loop(); });
  frontend_thread_ = std::thread([this] { frontend_loop(); });
  stage_thread_ = std::thread([this] { stage_loop(); });
}

void MeridianPipeline::stop() {
  // Drain in flow order: close the sensor edge and join its stage first so no new
  // measurement is produced, then close Q_meas and join the front-end thread. Only then
  // is Q_backend producer-free, so close it and join the back-end thread, which folds
  // whatever is still staged on its way out.
  q_sensors_.close();
  if (stage_thread_.joinable()) stage_thread_.join();
  q_meas_.close();
  if (frontend_thread_.joinable()) frontend_thread_.join();
  q_backend_.close();
  if (backend_thread_.joinable()) backend_thread_.join();
  if (sync_mode_ && backend_) {
    // Inline driver: fold the tail items fed after the last keyframe, then apply any
    // still-parked correction so the front-end's final state matches the graph.
    if (staged_since_opt_ > 0) {
      publish_graph_update(backend_->optimize());
      staged_since_opt_ = 0;
    }
    drain_pending_correction();
  }
  running_.store(false);
}

void MeridianPipeline::ingest(const RawLidarFrame& f) {
  // Constant per-sensor stamp correction onto the body-IMU timeline, applied here --
  // before validation and aggregation -- so every stamp-driven decision (monotonicity,
  // grouping, image-sweep matching) sees corrected time. Per-point offsets are
  // relative to the sweep reference and ride along unchanged.
  if (lidar_offset_ns_ != 0 && f.has_device_ns) {
    RawLidarFrame shifted = f;
    shifted.device_ns_first_column += lidar_offset_ns_;
    lidar_source_->ingest_raw(shifted);
    return;
  }
  lidar_source_->ingest_raw(f);
}
void MeridianPipeline::ingest(const RawImuFrame& f) {
  imu_source_->ingest_raw(f);
}
void MeridianPipeline::ingest(const RawCameraFrame& f) {
  if (camera_offset_ns_ != 0 && f.has_device_ns) {
    RawCameraFrame shifted = f;
    shifted.device_ns += camera_offset_ns_;
    camera_source_->ingest_raw(shifted);
    return;
  }
  camera_source_->ingest_raw(f);
}
void MeridianPipeline::ingest(const RawGnssFrame& f) {
  gnss_source_->ingest_raw(f);
}

void MeridianPipeline::set_group_sink(GroupSink sink) {
  group_sink_ = std::move(sink);
}

void MeridianPipeline::set_keyframe_sink(KeyframeSink sink) {
  keyframe_sink_ = std::move(sink);
}

void MeridianPipeline::set_backend_tap(BackendTap tap) { backend_tap_ = std::move(tap); }

void MeridianPipeline::set_graph_update_sink(GraphUpdateSink sink) {
  graph_update_sink_ = std::move(sink);
}

std::vector<StampedPose> MeridianPipeline::corrected_trajectory() const {
  return backend_ ? backend_->corrected_trajectory() : std::vector<StampedPose>{};
}

Pose MeridianPipeline::map_odom() const { return backend_ ? backend_->map_odom() : Pose{}; }

bool MeridianPipeline::backend_enabled() const { return backend_ != nullptr; }

NavState MeridianPipeline::live_state() const { return frontend_->live_state(); }

TelemetrySink* MeridianPipeline::telemetry() {
  return sink_.get();
}

void MeridianPipeline::enqueue(SensorSample&& s) {
  if (sync_mode_) {
    process(std::move(s));
    return;
  }
  const Timestamp t = std::visit(
      [](const auto& sample) {
        using T = std::decay_t<decltype(sample)>;
        if constexpr (std::is_same_v<T, LidarScan>)
          return sample.stamp_start;
        else
          return sample.stamp;
      },
      s);
  // After close() the queue accepts nothing: samples still arriving on an ingest thread
  // during shutdown are dropped silently and are not counted as an overload drop.
  const std::size_t dropped = q_sensors_.push_or_drop_oldest(std::move(s));
  if (dropped > 0) sink_->scalar(keys::pipeline::QSensorsDropped, 1.0, t);
}

void MeridianPipeline::stage_loop() {
  SensorSample s;
  while (q_sensors_.pop(s)) process(std::move(s));
}

void MeridianPipeline::dispatch_to_frontend(MeasSample&& m) {
  if (sync_mode_) {
    process_meas(std::move(m));
    return;
  }
  // Q_meas interleaves whole sweeps with the live IMU that advances the front-end
  // between sweeps. Under overload only live IMU may be evicted: losing a sweep leaves
  // the front-end's last-solved time behind the next sweep's t_begin, and its steady-
  // state seed would then extrapolate across a span whose IMU were in the dropped
  // sweep and are gone. So evict the oldest IMU sample first, and only fall back to a
  // sweep when nothing else is in flight (handled inside the queue).
  auto outcome = q_meas_.push_protecting(std::move(m), [](const MeasSample& s) {
    return std::holds_alternative<PreprocessedGroup>(s);
  });
  if (outcome.evicted) {
    if (std::holds_alternative<PreprocessedGroup>(*outcome.evicted)) {
      const Timestamp te = std::get<PreprocessedGroup>(*outcome.evicted).group.t_end;
      sink_->scalar(keys::pipeline::QMeasDropped, 1.0, te);
      sink_->scalar(keys::pipeline::QMeasDroppedSweep, 1.0, te);
      sink_->event(Level::Error, keys::pipeline::QMeasDroppedSweep,
                   "front-end overloaded: dropped a whole sweep from Q_meas", te);
    } else {
      const Timestamp ts = std::get<ImuSample>(*outcome.evicted).stamp;
      sink_->scalar(keys::pipeline::QMeasDropped, 1.0, ts);
      sink_->scalar(keys::pipeline::QMeasDroppedImu, 1.0, ts);
    }
  }
}

void MeridianPipeline::frontend_loop() {
  MeasSample m;
  while (q_meas_.pop(m)) process_meas(std::move(m));
}

void MeridianPipeline::process_meas(MeasSample&& m) {
  std::visit(
      [this](auto&& sample) {
        using T = std::decay_t<decltype(sample)>;
        if constexpr (std::is_same_v<T, PreprocessedGroup>) {
          // The wrapper sees the group before the solve so it can publish the live pose
          // it carries; then the front-end optimises the window against it.
          if (group_sink_) group_sink_(PreprocessedGroup{sample});
          frontend_->ingest(sample);
          // The group's GNSS fixes follow the ingest, so a keyframe this very sweep
          // produced is already the freshest anchor. Before the first keyframe there is
          // no node to anchor a fix to, so pre-init fixes are not forwarded.
          if (last_kf_id_) {
            for (const GnssFix& fix : sample.group.gnss) {
              feed_backend(BackendItem{GnssForBackend{fix, *last_kf_id_}});
            }
          }
          // A parked back-end correction folds in here, strictly between ingests, so
          // the front-end never rebases mid-solve or from inside its keyframe sink.
          drain_pending_correction();
        } else {  // ImuSample
          frontend_->ingest_imu_live(sample);
        }
      },
      std::move(m));
}

void MeridianPipeline::on_keyframe(KeyframePacket&& kf) {
  const auto n = ++keyframe_count_;
  sink_->event(Level::Info, keys::frontend::Keyframe, "front-end emitted a keyframe", kf.stamp);
  sink_->scalar(keys::frontend::KeyframeCount, static_cast<double>(n), kf.stamp);
  last_kf_id_ = kf.id;
  // The back-end stream sees the keyframe before the wrapper sink consumes the move;
  // the copy is cheap because the heavy payloads are shared-immutable handles.
  feed_backend(BackendItem{kf});
  if (keyframe_sink_) keyframe_sink_(std::move(kf));
}

void MeridianPipeline::feed_backend(BackendItem&& item) {
  // The single hand-off into the back-end input stream. The tap observes every item in
  // feed order even when the back-end itself is disabled.
  if (backend_tap_) backend_tap_(item);
  if (!backend_) return;

  if (sync_mode_) {
    // Inline driver: stage, and fold on every keyframe so corrections land at
    // deterministic points in the stream. GNSS/loop items wait for the next keyframe
    // (or stop()) — same arrival order as the queue would give the live driver.
    const bool is_kf = std::holds_alternative<KeyframePacket>(item);
    stage_backend_item(std::move(item));
    ++staged_since_opt_;
    if (is_kf) {
      const GraphUpdate gu = backend_->optimize();
      staged_since_opt_ = 0;
      publish_graph_update(gu);
      // Offer the just-folded keyframe to the detector at its corrected pose; fold any loops
      // immediately so even a revisit found on the final keyframe lands a correction.
      const std::vector<LoopConstraint> loops = detect_loops();
      for (const LoopConstraint& lc : loops) feed_backend(BackendItem{lc});
      if (!loops.empty()) {
        const GraphUpdate lu = backend_->optimize();
        staged_since_opt_ = 0;
        publish_graph_update(lu);
      }
    }
    return;
  }

  const Timestamp t = std::visit(
      [](const auto& v) -> Timestamp {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, KeyframePacket>) {
          return v.stamp;
        } else if constexpr (std::is_same_v<T, GnssForBackend>) {
          return v.fix.stamp;
        } else {  // LoopConstraint carries no stamp of its own
          return 0;
        }
      },
      item);
  // Lossless edge: the graph must see every item, so the queue grows instead of
  // dropping and the resulting depth is the overload signal.
  const std::size_t depth = q_backend_.push_always(std::move(item));
  sink_->scalar(keys::backend::QueueDepth, static_cast<double>(depth), t);
  const bool over = static_cast<int>(depth) > cfg_.backend.queue_warn_depth;
  if (over && !backend_queue_over_) {
    sink_->event(Level::Warn, keys::backend::QueueOverload,
                 "back-end input queue depth crossed the warn threshold", t);
  }
  backend_queue_over_ = over;
}

bool MeridianPipeline::stage_backend_item(BackendItem&& item) {
  std::visit(
      [this](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, KeyframePacket>) {
          if (store_) {
            // Retain the cloud and stamp before the packet is moved into the back-end;
            // the map cloud composes from here, and the detector reads the same store.
            store_->put(v.id, v.cloud_body, v.image, v.T_ref_body);
            kf_stamps_[v.id] = v.stamp;
          }
          if (loop_detector_) pending_kf_for_detector_.emplace_back(v.id, v.cloud_body);
          backend_->add_keyframe(std::move(v));
        } else if constexpr (std::is_same_v<T, LoopConstraint>) {
          backend_->add_loop_constraint(v);
        } else {  // GnssForBackend
          backend_->add_absolute(v.fix, v.nearest_kf_id);
        }
      },
      std::move(item));
  return backend_->wants_immediate_optimize();
}

void MeridianPipeline::backend_loop() {
  using Clock = std::chrono::steady_clock;
  const auto interval =
      std::chrono::milliseconds(std::llround(cfg_.backend.optimize_interval_ms));
  auto last_opt = Clock::now();
  std::uint64_t staged = 0;
  bool force = false;
  BackendItem item;
  for (;;) {
    const PopResult r = q_backend_.pop_for(item, interval);
    if (r == PopResult::Closed) break;
    if (r == PopResult::Item) {
      // Stage the whole burst in one wake so it folds in a single update.
      force |= stage_backend_item(std::move(item));
      ++staged;
      while (q_backend_.try_pop(item)) {
        force |= stage_backend_item(std::move(item));
        ++staged;
      }
    }
    sink_->scalar(keys::backend::QueueDepth, static_cast<double>(q_backend_.size()), tele_stamp());
    if (staged == 0) continue;
    // The cadence timer batches inserts; an immediate request (an admitted loop, a
    // just-locked datum) bypasses it.
    if (!force && Clock::now() - last_opt < interval) continue;
    publish_graph_update(backend_->optimize());
    staged = 0;
    force = false;
    last_opt = Clock::now();
    // Detected loops re-enter through the lossless queue and fold on the next wake (an
    // admitted loop forces it via wants_immediate_optimize).
    for (const LoopConstraint& lc : detect_loops()) feed_backend(BackendItem{lc});
  }
  // Closed: stop() joined every producer first, so what remains is the final batch.
  while (q_backend_.try_pop(item)) {
    stage_backend_item(std::move(item));
    ++staged;
  }
  if (staged > 0) publish_graph_update(backend_->optimize());
  // Final loop detection: the queue is closed, so stage detected loops directly and fold once.
  const std::vector<LoopConstraint> final_loops = detect_loops();
  if (!final_loops.empty()) {
    for (const LoopConstraint& lc : final_loops) {
      if (backend_tap_) backend_tap_(BackendItem{lc});
      stage_backend_item(BackendItem{lc});
    }
    publish_graph_update(backend_->optimize());
  }
}

void MeridianPipeline::publish_graph_update(const GraphUpdate& gu) {
  const BackEndDiagnostics diag = backend_->diagnostics();
  const Timestamp ts = tele_stamp();
  sink_->scalar(keys::backend::OptimizeLag, static_cast<double>(diag.optimize_lag), ts);
  sink_->scalar(keys::backend::NKeyframes, static_cast<double>(diag.num_keyframes), ts);
  // The front-end is re-anchored only when the back-end performs a real rigid correction
  // (a loop closure or GNSS realignment). Routine per-keyframe moves and the floating-gauge
  // drift are not corrections to fold back: feeding them would repeatedly perturb the spline
  // against a marginalization prior that stays at its old linearization point, which the
  // front-end then has to fight on every sweep. L4 still sees every moved keyframe through
  // the unconditional graph-update sink below.
  if (gu.loop_closed && cfg_.backend.correct_frontend) {
    std::lock_guard<std::mutex> lock(correction_mu_);
    pending_correction_ = gu;
  }
  // Moved keyframes stale the detector's cached submaps; drop those so the next compose
  // reflects the corrected geometry.
  if (loop_detector_ && !gu.moved.empty()) loop_detector_->on_graph_update(gu);
  // Assembled map cloud, stamped at the latest keyframe. Forced on a loop fold so the
  // de-warp is visible the instant the graph snaps; throttled otherwise.
  if (store_ && last_kf_id_) {
    publish_map_cloud(ts, /*force=*/gu.loop_closed);
  }
  if (map_) update_l4_map(gu, ts, /*force=*/gu.loop_closed);
  if (graph_update_sink_) graph_update_sink_(gu);
}

void MeridianPipeline::update_l4_map(const GraphUpdate& gu, Timestamp ts, bool force) {
  // Correct already-integrated keyframes first (clear-and-rebuild of the moved region).
  if (!gu.moved.empty()) map_->apply_graph_update(gu);
  // Integrate any newly-placed keyframes at their corrected back-end poses. A keyframe is
  // mapped exactly once; subsequent moves arrive through apply_graph_update above.
  for (const std::uint64_t id : store_->ids()) {
    if (mapped_ids_.count(id) != 0) continue;
    const std::optional<Pose> p = backend_->pose_of(id);
    if (!p) continue;  // not yet placed in the graph; integrate on a later fold
    const PointCloudPtr cloud = store_->cloud(id);
    if (!cloud) continue;
    KeyframePacket kf;
    kf.id = id;
    kf.cloud_body = cloud;
    kf.image = store_->image(id);
    kf.T_body_cam = store_->body_cam(id).value_or(Pose{});
    map_->integrate(kf, *p);
    mapped_ids_.insert(id);
  }
  // The surface mesh is O(voxels) to extract and large to transport, so throttle it
  // (forced on a loop fold, where the whole point is to show the corrected surface).
  if (!force && ++folds_since_mesh_ < kMeshFoldPeriod) return;
  folds_since_mesh_ = 0;
  publish_map_mesh(ts);
}

void MeridianPipeline::publish_map_mesh(Timestamp ts) {
  if (!map_ || !sink_->wants_clouds()) return;
  const ColorMesh& mesh = map_->extract_mesh();
  if (mesh.vertices.empty()) return;
  // Triangle-list marker: vertices are already in 3-per-triangle order with per-vertex
  // turbo colours, so a viewer renders the coloured surface with no setup.
  Marker mk;
  mk.type = Marker::Type::Triangles;
  mk.frame = Frame::Map;
  mk.ns = "map/mesh";
  mk.id = 0;
  mk.scale = 1.0f;
  mk.points = mesh.vertices;
  mk.colors.resize(mesh.colors.size());
  for (std::size_t i = 0; i < mesh.colors.size(); ++i) {
    mk.colors[i] = {mesh.colors[i][0] / 255.f, mesh.colors[i][1] / 255.f,
                    mesh.colors[i][2] / 255.f, 1.f};
  }
  sink_->marker(mk, ts);
}

std::vector<LoopConstraint> MeridianPipeline::detect_loops() {
  std::vector<LoopConstraint> loops;
  if (!loop_detector_) {
    pending_kf_for_detector_.clear();
    return loops;
  }
  for (auto& [id, cloud] : pending_kf_for_detector_) {
    const std::optional<Pose> p = backend_->pose_of(id);
    if (!p) continue;
    loop_detector_->add_keyframe(id, cloud, *p);
    std::vector<LoopConstraint> found = loop_detector_->detect();
    for (LoopConstraint& lc : found) loops.push_back(std::move(lc));
  }
  pending_kf_for_detector_.clear();
  return loops;
}

Timestamp MeridianPipeline::tele_stamp() const {
  if (!last_kf_id_) return 0;
  const auto it = kf_stamps_.find(*last_kf_id_);
  return it != kf_stamps_.end() ? it->second : 0;
}

void MeridianPipeline::publish_map_cloud(Timestamp ts, bool force) {
  // Skip the O(all-points) rebuild entirely when no sink renders clouds (file/null): the
  // assembled map cloud is viz-only, so building it just to drop it wastes the whole cost.
  if (!store_ || !backend_ || !sink_->wants_clouds() || !sink_->enabled(keys::map::Cloud)) return;
  // Throttle: rebuilding the whole map cloud from every stored keyframe is O(points), so
  // only do it every few folds — unless a loop just folded, where the whole point is to
  // show the de-warp the instant it happens.
  if (!force && ++folds_since_map_cloud_ < kMapCloudFoldPeriod) return;
  folds_since_map_cloud_ = 0;

  // Compose each keyframe's body-frame cloud at its corrected map pose, then collapse to a
  // single voxel grid (first return per cell wins) so the assembled cloud stays bounded as
  // the trajectory grows. The grid key quantises the map-frame position at the voxel pitch.
  constexpr float kInvVoxel = 1.0f / kMapCloudVoxel;
  std::map<std::array<std::int64_t, 3>, LidarPoint> grid;
  for (const std::uint64_t id : store_->ids()) {
    const std::optional<Pose> p = backend_->pose_of(id);
    if (!p) continue;
    const PointCloudPtr cloud = store_->cloud(id);
    if (!cloud) continue;
    for (const LidarPoint& src : *cloud) {
      const Eigen::Vector3d pm = (*p) * src.xyz.cast<double>();
      const std::array<std::int64_t, 3> key{
          static_cast<std::int64_t>(std::floor(pm.x() * kInvVoxel)),
          static_cast<std::int64_t>(std::floor(pm.y() * kInvVoxel)),
          static_cast<std::int64_t>(std::floor(pm.z() * kInvVoxel))};
      if (grid.find(key) != grid.end()) continue;
      LidarPoint out = src;
      out.xyz = pm.cast<float>();
      grid.emplace(key, out);
    }
  }

  std::vector<LidarPoint> out;
  out.reserve(grid.size());
  for (auto& [key, pt] : grid) out.push_back(pt);
  sink_->cloud(keys::map::Cloud, PointCloudView{out}, Frame::Map, ts);
}

void MeridianPipeline::drain_pending_correction() {
  std::optional<GraphUpdate> gu;
  {
    std::lock_guard<std::mutex> lock(correction_mu_);
    gu.swap(pending_correction_);
  }
  if (gu) frontend_->apply_correction(*gu);
}

void MeridianPipeline::process(SensorSample&& s) {
  std::visit(
      [this](auto&& sample) {
        using T = std::decay_t<decltype(sample)>;
        if constexpr (std::is_same_v<T, ImuSample>) {
          // The same sample feeds the aggregator (which the next sweep optimises
          // against) and the front-end's live-state advance between sweeps.
          dispatch_to_frontend(MeasSample{std::in_place_type<ImuSample>, sample});
          aggregator_->on(std::move(sample));
        } else if constexpr (std::is_same_v<T, LidarScan>) {
          LidarScan filtered;
          {
            MERIDIAN_SCOPED_TIME(sink_.get(), keys::stage::Preprocess, sample.stamp_start);
            filtered = lidar_preprocessor_->process(sample);
          }
          // Measured on the filtered scan: start-vs-end spacing is the tripwire for a
          // driver stamping the end of the sweep instead of the first column.
          sink_->scalar(keys::sensors::LidarSweepDurationMs,
                        static_cast<double>(filtered.sweep_duration) / 1e6, filtered.stamp_start);
          aggregator_->on(std::move(filtered));
        } else if constexpr (std::is_same_v<T, CameraFrame>) {
          aggregator_->on(std::move(sample));
        } else {  // GnssFix
          const GnssVerdict verdict = gnss_gate_->evaluate(sample);
          if (verdict.accepted) aggregator_->on(std::move(sample));
        }
      },
      std::move(s));
}

void MeridianPipeline::on_group(MeasureGroup&& g) {
  sink_->scalar(keys::pipeline::QSensorsDepth, static_cast<double>(q_sensors_.size()), g.t_end);
  sink_->scalar(keys::pipeline::GroupImuCount, static_cast<double>(g.imu.size()), g.t_end);
  sink_->scalar(keys::pipeline::GroupPoints,
                static_cast<double>(g.scan.points ? g.scan.points->size() : 0), g.t_end);

  if (g.image && sink_->enabled(keys::preprocess::CameraPyramid)) {
    const ProcessedCamera cam = camera_preprocessor_->process(*g.image);
    sink_->scalar(keys::preprocess::CameraPyramidLevels, static_cast<double>(cam.pyramid.size()),
                  g.image->stamp);
    // Surface both the decoded original and its undistorted form so the rectification can
    // be watched side by side. This is an inspection path; the front-end is fed the raw
    // group image, not this processed buffer.
    const auto publish_mono = [&](const char* key, const cv::Mat& img) {
      if (img.empty() || !img.isContinuous() || img.type() != CV_8UC1 || !sink_->enabled(key)) {
        return;
      }
      ImageOverlay ov;
      ov.frame = Frame::CamLink;
      ov.width = img.cols;
      ov.height = img.rows;
      ov.encoding = ImageOverlay::Encoding::Mono8;
      ov.base = std::span<const std::uint8_t>(img.data, img.total());
      sink_->image(key, ov, g.image->stamp);
    };
    publish_mono(keys::preprocess::CameraRaw, cam.intensity_raw);
    publish_mono(keys::preprocess::CameraIntensity, cam.intensity);
  }

  dispatch_to_frontend(MeasSample{PreprocessedGroup{std::move(g)}});
}

}  // namespace meridian
