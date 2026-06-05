#include "meridian/pipeline/meridian_pipeline.hpp"

#include <cstdio>
#include <span>
#include <utility>

#include <Eigen/Geometry>

#include "health_bridge.hpp"
#include "meridian/calib/extrinsic.hpp"
#include "meridian/calib/intrinsics.hpp"
#include "meridian/debug/telemetry.hpp"
#include "meridian/preprocess/camera_preprocess.hpp"
#include "meridian/preprocess/gnss_gate.hpp"
#include "meridian/preprocess/ilidar_preprocessor.hpp"
#include "meridian/preprocess/imu_init.hpp"
#include "meridian/preprocess/imu_only_deskew.hpp"
#include "meridian/sensors/aggregator.hpp"
#include "meridian/sensors/sources.hpp"
#include "meridian/time/clock_model.hpp"

namespace meridian {

namespace {

constexpr std::size_t kSensorQueueCapacity = 512;

SensorInfo make_info(std::uint8_t id, Modality modality, Frame frame,
                     const std::string& model, double rate_hz, StampSource src) {
  SensorInfo info;
  info.id = id;
  info.modality = modality;
  info.sensor_frame = frame;
  info.model = model;
  info.nominal_rate_hz = rate_hz;
  info.configured_stamp_source = src;
  return info;
}

// The LiDAR->body extrinsic from config, packaged for the validity filter / deskew.
Extrinsic lidar_extrinsic(const LidarSensorConfig& lidar) {
  Extrinsic ext;
  ext.child = Frame::OsSensor0;
  ext.parent = Frame::ImuLink;
  ext.T_parent_child =
      Pose{Eigen::Quaterniond(lidar.extrinsic_R), lidar.extrinsic_T};
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
      q_sensors_(kSensorQueueCapacity) {
  const auto& s = cfg_.sensors;
  const auto& th = cfg_.time.health;

  lidar_source_ = std::make_unique<OusterLidarSource>(
      make_info(static_cast<std::uint8_t>(s.lidar.id), Modality::Lidar, Frame::OsSensor0,
                s.lidar.model, s.lidar.nominal_rate_hz,
                s.lidar.ptp ? StampSource::HwPtp : StampSource::SwOffset),
      s.lidar, clock_.get(), health_.get(), sink_.get(), th);
  imu_source_ = std::make_unique<ImuSource>(
      make_info(static_cast<std::uint8_t>(s.imu.id), Modality::Imu, Frame::ImuLink,
                s.imu.model, s.imu.rate_hz,
                s.imu.has_device_clock ? StampSource::HwPtp : StampSource::ArrivalOnly),
      s.imu, clock_.get(), health_.get(), sink_.get(), th);
  camera_source_ = std::make_unique<CameraSource>(
      make_info(static_cast<std::uint8_t>(s.camera.id), Modality::Camera, Frame::CamLink,
                s.camera.model, s.camera.nominal_rate_hz, StampSource::SwOffset),
      s.camera, clock_.get(), health_.get(), sink_.get(), th);
  gnss_source_ = std::make_unique<GnssSource>(
      make_info(static_cast<std::uint8_t>(s.gnss.id), Modality::Gnss, Frame::GnssLink,
                "gnss", 1.0,
                s.gnss.pps_disciplines_clock ? StampSource::HwPps
                                             : StampSource::SwOffset),
      s.gnss, clock_.get(), health_.get(), sink_.get(), th);

  lidar_preprocessor_ =
      makeLidarPreprocessor(cfg_.preprocess, lidar_extrinsic(s.lidar), sink_.get());
  imu_init_ = std::make_unique<ImuInitializer>(cfg_.preprocess.imu,
                                               cfg_.preprocess.deskew.imu_init_count);
  gnss_gate_ = std::make_unique<GnssGate>(cfg_.preprocess.gnss,
                                          /*velocity_source=*/nullptr, sink_.get());
  camera_preprocessor_ = std::make_unique<CameraPreprocessor>(
      cfg_.preprocess.camera, camera_intrinsics(s.camera), sink_.get());
  aggregator_ = std::make_unique<Aggregator>(cfg_.aggregation, cfg_.sensors,
                                             health_.get(), sink_.get());
  aggregator_->set_sink([this](MeasureGroup&& g) { on_group(std::move(g)); });

  // Every stamped sample funnels into the one stage entry, whatever its modality.
  lidar_source_->set_callback([this](LidarScan&& scan) { enqueue(std::move(scan)); });
  imu_source_->set_callback([this](ImuSample&& imu) { enqueue(std::move(imu)); });
  camera_source_->set_callback(
      [this](CameraFrame&& frame) { enqueue(std::move(frame)); });
  gnss_source_->set_callback([this](GnssFix&& fix) { enqueue(std::move(fix)); });
}

MeridianPipeline::~MeridianPipeline() { stop(); }

void MeridianPipeline::start() {
  if (sync_mode_ || running_.load()) return;
  running_.store(true);
  stage_thread_ = std::thread([this] { stage_loop(); });
}

void MeridianPipeline::stop() {
  q_sensors_.close();
  if (stage_thread_.joinable()) stage_thread_.join();
  running_.store(false);
}

void MeridianPipeline::ingest(const RawLidarFrame& f) { lidar_source_->ingest_raw(f); }
void MeridianPipeline::ingest(const RawImuFrame& f) { imu_source_->ingest_raw(f); }
void MeridianPipeline::ingest(const RawCameraFrame& f) { camera_source_->ingest_raw(f); }
void MeridianPipeline::ingest(const RawGnssFrame& f) { gnss_source_->ingest_raw(f); }

void MeridianPipeline::set_group_sink(GroupSink sink) { group_sink_ = std::move(sink); }

TelemetrySink* MeridianPipeline::telemetry() { return sink_.get(); }

void MeridianPipeline::enqueue(SensorSample&& s) {
  if (sync_mode_) {
    process(std::move(s));
    return;
  }
  const Timestamp t = std::visit(
      [](const auto& sample) {
        using T = std::decay_t<decltype(sample)>;
        if constexpr (std::is_same_v<T, LidarScan>) return sample.stamp_start;
        else return sample.stamp;
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

void MeridianPipeline::process(SensorSample&& s) {
  std::visit(
      [this](auto&& sample) {
        using T = std::decay_t<decltype(sample)>;
        if constexpr (std::is_same_v<T, ImuSample>) {
          if (!imu_init_->done()) {
            const bool became_done = imu_init_->add(sample);
            if (became_done) {
              sink_->event(Level::Info, "preprocess/imu_init_done", "static init converged",
                           sample.stamp);
            } else if (imu_init_->failed()) {
              sink_->event(Level::Warn, "preprocess/imu_init_retry",
                           "motion gate rejected the static window", sample.stamp);
              imu_init_->clear();
            }
          }
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
                        static_cast<double>(filtered.sweep_duration) / 1e6,
                        filtered.stamp_start);
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
  sink_->scalar("pipeline/q_sensors_depth", static_cast<double>(q_sensors_.size()),
                g.t_end);

  if (!imu_init_->done()) {
    // Surface bootstrap progress so the operator can see why nothing is published yet.
    const double target = static_cast<double>(cfg_.preprocess.deskew.imu_init_count);
    sink_->scalar("preprocess/imu_init_progress",
                  static_cast<double>(imu_init_->state().count) / target, g.t_end);
    // Hold sweeps until the static init converges; the freshest scans win.
    const auto cap = static_cast<std::size_t>(cfg_.preprocess.deskew.bootstrap_max_scans);
    if (bootstrap_groups_.size() >= cap) {
      bootstrap_groups_.pop_front();
      sink_->event(Level::Warn, "preprocess/bootstrap_drop",
                   "bootstrap buffer full, dropped oldest sweep", g.t_end);
    }
    bootstrap_groups_.push_back(std::move(g));
    return;
  }

  // Init just converged: flush held sweeps in order before the current one.
  while (!bootstrap_groups_.empty()) {
    MeasureGroup held = std::move(bootstrap_groups_.front());
    bootstrap_groups_.pop_front();
    emit_group(std::move(held));
  }
  emit_group(std::move(g));
}

void MeridianPipeline::emit_group(MeasureGroup&& g) {
  std::optional<LidarScan> deskewed;
  {
    MERIDIAN_SCOPED_TIME(sink_.get(), "preprocess.deskew", g.t_end);
    deskewed = deskew_group(g);
  }

  if (deskewed && sink_->enabled("body/scan")) {
    sink_->cloud("body/scan", PointCloudView{*deskewed->points}, Frame::Body, g.t_end);
  }
  sink_->scalar("pipeline/group_imu_count", static_cast<double>(g.imu.size()), g.t_end);
  sink_->scalar("pipeline/group_points",
                static_cast<double>(g.scan.points ? g.scan.points->size() : 0), g.t_end);

  if (g.image && sink_->enabled("preprocess/camera_pyramid")) {
    const ProcessedCamera cam = camera_preprocessor_->process(*g.image);
    sink_->scalar("preprocess/camera_pyramid_levels",
                  static_cast<double>(cam.pyramid.size()), g.image->stamp);
    // The pyramid's base level is what the visual front-end will consume; surface it
    // so the operator sees the camera path's output, not just a level count.
    if (!cam.intensity.empty() && cam.intensity.isContinuous() &&
        cam.intensity.type() == CV_8UC1 &&
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

  if (group_sink_) {
    group_sink_(
        PreprocessedGroup{std::move(g), std::move(deskewed), /*cold_start=*/!spline_active_});
  }
}

std::optional<LidarScan> MeridianPipeline::deskew_group(const MeasureGroup& g) {
  if (g.imu.empty() || !g.scan.points) return std::nullopt;

  // Cold-start trajectory: identity start pose and zero initial velocity. The warp uses
  // only relative motion within the sweep, so the arbitrary start pose cancels; the
  // zero-velocity seed means within-sweep translation is what the IMU alone integrates.
  ImuOnlyDeskew provider(imu_init_->state(),
                         lidar_extrinsic(cfg_.sensors.lidar), Pose{},
                         Eigen::Vector3d::Zero());
  // When the bracketing sample lands just after t_begin, hold its measurement back to
  // the sweep start — the same zero-order hold as the tail below. The hold is bounded
  // by the group's own mean sample period: a gap beyond one period is a real IMU hole
  // and must keep failing the horizon check.
  const Duration head_gap = g.imu.front().stamp - g.t_begin;
  if (head_gap > 0 && g.imu.size() >= 2) {
    const Duration period = (g.imu.back().stamp - g.imu.front().stamp) /
                            static_cast<Duration>(g.imu.size() - 1);
    if (head_gap <= period) {
      ImuSample head = g.imu.front();
      head.stamp = g.t_begin;
      provider.pushImu(head);
    }
  }
  for (const ImuSample& s : g.imu) provider.pushImu(s);
  // The group's IMU set ends at the last sample at-or-before t_end; hold the last
  // measurement constant up to the sweep end (at most one IMU period) so the horizon
  // covers the anchor.
  if (g.imu.back().stamp < g.t_end) {
    ImuSample tail = g.imu.back();
    tail.stamp = g.t_end;
    provider.pushImu(tail);
  }
  provider.setAnchor(g.t_end);

  LidarScan out;
  if (!provider.deskew(g.scan, &out)) {
    // Positive head: first IMU after t_begin; positive tail: last IMU before t_end.
    char msg[96];
    std::snprintf(msg, sizeof(msg),
                  "imu horizon misses the sweep: head %+.0f us, tail %+.0f us",
                  static_cast<double>(g.imu.front().stamp - g.t_begin) / 1e3,
                  static_cast<double>(g.t_end - g.imu.back().stamp) / 1e3);
    sink_->event(Level::Warn, "preprocess/deskew_horizon", msg, g.t_end);
    return std::nullopt;
  }
  return out;
}

}  // namespace meridian
