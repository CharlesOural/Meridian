#include "meridian/pipeline/meridian_pipeline.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>

#include "calibration_from_config.hpp"
#include "health_bridge.hpp"
#include "meridian/calib/extrinsic.hpp"
#include "meridian/calib/intrinsics.hpp"
#include "meridian/debug/telemetry.hpp"
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

IntrinsicsCamera camera_intrinsics(const CameraSensorConfig& cam) {
  IntrinsicsCamera k;
  k.fx = cam.intrinsics[0];
  k.fy = cam.intrinsics[1];
  k.cx = cam.intrinsics[2];
  k.cy = cam.intrinsics[3];
  return k;
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
    sink_->event(Level::Warn, "sensors/lidar/extrinsic_default",
                 "lidar extrinsic not configured; using identity T_imu_lidar", 0);
  }

  lidar_preprocessor_ = makeLidarPreprocessor(cfg_.preprocess, lidar_extrinsic(s.lidar),
                                              s.lidar.nominal_rate_hz, sink_.get());
  gnss_gate_ = std::make_unique<GnssGate>(cfg_.preprocess.gnss,
                                          /*velocity_source=*/nullptr, sink_.get());
  camera_preprocessor_ = std::make_unique<CameraPreprocessor>(
      cfg_.preprocess.camera, camera_intrinsics(s.camera), sink_.get());
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
    // estimator parameter); hand it to the front-end through its config copy.
    FrontendConfig fe_cfg = cfg_.frontend;
    fe_cfg.debug_path_sample_hz = cfg_.debug.path_sample_hz;
    frontend_ = makeFrontEnd(fe_cfg, calibrationFromConfig(cfg_.sensors), sink_.get(),
                             /*deterministic=*/sync_mode_);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("front-end construction failed: ") + e.what());
  }
  frontend_->set_keyframe_sink([this](KeyframePacket&& kf) { on_keyframe(std::move(kf)); });
}

MeridianPipeline::~MeridianPipeline() {
  stop();
}

void MeridianPipeline::start() {
  if (sync_mode_ || running_.load()) return;
  running_.store(true);
  frontend_thread_ = std::thread([this] { frontend_loop(); });
  stage_thread_ = std::thread([this] { stage_loop(); });
}

void MeridianPipeline::stop() {
  // Drain in flow order: close the sensor edge and join its stage first so no new
  // measurement is produced, then close Q_meas and join the front-end thread.
  q_sensors_.close();
  if (stage_thread_.joinable()) stage_thread_.join();
  q_meas_.close();
  if (frontend_thread_.joinable()) frontend_thread_.join();
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

NavState MeridianPipeline::live_state() const {
  return frontend_->live_state();
}

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
  if (dropped > 0) sink_->scalar("pipeline/q_sensors_dropped", 1.0, t);
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
      sink_->scalar("pipeline/q_meas_dropped", 1.0, te);
      sink_->scalar("pipeline/q_meas_dropped_sweep", 1.0, te);
      sink_->event(Level::Error, "pipeline/q_meas_dropped_sweep",
                   "front-end overloaded: dropped a whole sweep from Q_meas", te);
    } else {
      const Timestamp ts = std::get<ImuSample>(*outcome.evicted).stamp;
      sink_->scalar("pipeline/q_meas_dropped", 1.0, ts);
      sink_->scalar("pipeline/q_meas_dropped_imu", 1.0, ts);
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
        } else {  // ImuSample
          frontend_->ingest_imu_live(sample);
        }
      },
      std::move(m));
}

void MeridianPipeline::on_keyframe(KeyframePacket&& kf) {
  const auto n = ++keyframe_count_;
  sink_->event(Level::Info, "frontend/keyframe", "front-end emitted a keyframe", kf.stamp);
  sink_->scalar("frontend/keyframe_count", static_cast<double>(n), kf.stamp);
  if (keyframe_sink_) keyframe_sink_(std::move(kf));
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
            MERIDIAN_SCOPED_TIME(sink_.get(), "preprocess", sample.stamp_start);
            filtered = lidar_preprocessor_->process(sample);
          }
          // Measured on the filtered scan: start-vs-end spacing is the tripwire for a
          // driver stamping the end of the sweep instead of the first column.
          sink_->scalar("sensors/lidar/sweep_duration_ms",
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
  sink_->scalar("pipeline/q_sensors_depth", static_cast<double>(q_sensors_.size()), g.t_end);
  sink_->scalar("pipeline/group_imu_count", static_cast<double>(g.imu.size()), g.t_end);
  sink_->scalar("pipeline/group_points",
                static_cast<double>(g.scan.points ? g.scan.points->size() : 0), g.t_end);

  if (g.image && sink_->enabled("preprocess/camera_pyramid")) {
    const ProcessedCamera cam = camera_preprocessor_->process(*g.image);
    sink_->scalar("preprocess/camera_pyramid_levels", static_cast<double>(cam.pyramid.size()),
                  g.image->stamp);
    // The pyramid's base level is what the visual front-end will consume; surface it
    // so the operator sees the camera path's output, not just a level count.
    if (!cam.intensity.empty() && cam.intensity.isContinuous() && cam.intensity.type() == CV_8UC1 &&
        sink_->enabled("preprocess/camera_intensity")) {
      ImageOverlay ov;
      ov.frame = Frame::CamLink;
      ov.width = cam.intensity.cols;
      ov.height = cam.intensity.rows;
      ov.encoding = ImageOverlay::Encoding::Mono8;
      ov.base = std::span<const std::uint8_t>(cam.intensity.data, cam.intensity.total());
      sink_->image("preprocess/camera_intensity", ov, g.image->stamp);
    }
  }

  dispatch_to_frontend(MeasSample{PreprocessedGroup{std::move(g)}});
}

}  // namespace meridian
