#include "meridian/sensors/sources.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "meridian/debug/log.hpp"
#include "meridian/debug/telemetry.hpp"
#include "meridian/time/clock_model.hpp"
#include "source_base.hpp"

namespace meridian {

namespace {

// Mid-exposure shift in nanoseconds for a given exposure time.
Timestamp mid_exposure_shift_ns(float exposure_s) {
  return static_cast<Timestamp>(std::llround(static_cast<double>(exposure_s) / 2.0 * 1e9));
}

}  // namespace

// ---------------------------------------------------------------------------
// OusterLidarSource
// ---------------------------------------------------------------------------

struct OusterLidarSource::Impl {
  Impl(SensorInfo info, const LidarSensorConfig& cfg, ClockModel* clock,
       HealthSink* health, TelemetrySink* telemetry, const TimeHealth& health_cfg,
       const ValidatorConfig& validator_cfg)
      : base(std::move(info), clock, health, telemetry, health_cfg.failed_timeout_ms,
             health_cfg.rate_tolerance_frac, validator_cfg),
        clock_(clock),
        cfg(cfg) {}

  SourceBase base;
  ClockModel* clock_;
  LidarSensorConfig cfg;
  Callback cb;
};

OusterLidarSource::OusterLidarSource(SensorInfo info, const LidarSensorConfig& cfg,
                                     ClockModel* clock, HealthSink* health,
                                     TelemetrySink* telemetry,
                                     const TimeHealth& health_cfg,
                                     const ValidatorConfig& validator_cfg)
    : impl_(std::make_unique<Impl>(std::move(info), cfg, clock, health, telemetry,
                                   health_cfg, validator_cfg)) {}

OusterLidarSource::~OusterLidarSource() = default;

void OusterLidarSource::set_callback(Callback cb) { impl_->cb = std::move(cb); }
void OusterLidarSource::start() {}
void OusterLidarSource::stop() {}
SensorInfo OusterLidarSource::info() const { return impl_->base.info(); }

void OusterLidarSource::ingest_raw(const RawLidarFrame& f) {
  // PTP lock decides between the hardware path and the warm software-offset fallback;
  // there is one stamping path that degrades along this ladder.
  const bool ptp_ok =
      f.has_device_ns && impl_->clock_ && impl_->clock_->ptp_locked(ClockId::Lidar);
  const StampSource src = ptp_ok ? StampSource::HwPtp : StampSource::SwOffset;

  LidarScan scan;
  scan.sensor_id = impl_->base.info().id;
  scan.sensor_frame = impl_->base.info().sensor_frame;
  scan.stamp_start = impl_->base.note_stamp(
      impl_->base.stamp_from(f.device_ns_first_column, ClockId::Lidar, src), src);

  auto pts = std::make_shared<PointCloud>();
  pts->reserve(f.pts.size());
  std::int32_t max_t_offset = 0;
  for (const RawPoint& p : f.pts) {
    LidarPoint lp;
    lp.xyz = {p.x, p.y, p.z};
    lp.intensity = p.intensity;
    // Ouster t is already a per-column ns offset from the sweep reference, so it maps
    // directly with no unit conversion.
    lp.t_offset_ns = static_cast<std::int32_t>(p.t);
    lp.ring = p.ring;
    lp.ambient = p.ambient;
    lp.range = p.range_m;
    max_t_offset = std::max(max_t_offset, lp.t_offset_ns);
    pts->push_back(lp);
  }

  // Offsets are relative to stamp_start and not guaranteed ordered by column time, so
  // the sweep ends at the largest offset seen anywhere in the cloud.
  scan.sweep_duration = static_cast<Duration>(max_t_offset);
  scan.points = std::move(pts);

  // The standing validator owns the per-point-time, NaN/Inf, and emptiness checks; it
  // drops non-finite returns in place and rejects only an empty/time-less/all-NaN scan.
  if (!impl_->base.accept_lidar(scan)) return;

  if (impl_->cb) impl_->cb(std::move(scan));
}

// ---------------------------------------------------------------------------
// ImuSource
// ---------------------------------------------------------------------------

struct ImuSource::Impl {
  Impl(SensorInfo info, const ImuSensorConfig& cfg, ClockModel* clock,
       HealthSink* health, TelemetrySink* telemetry, const TimeHealth& health_cfg,
       const ValidatorConfig& validator_cfg)
      : base(std::move(info), clock, health, telemetry, health_cfg.failed_timeout_ms,
             health_cfg.rate_tolerance_frac, validator_cfg),
        clock_(clock),
        cfg(cfg) {}

  SourceBase base;
  ClockModel* clock_;
  ImuSensorConfig cfg;
  Callback cb;
};

ImuSource::ImuSource(SensorInfo info, const ImuSensorConfig& cfg, ClockModel* clock,
                     HealthSink* health, TelemetrySink* telemetry,
                     const TimeHealth& health_cfg,
                     const ValidatorConfig& validator_cfg)
    : impl_(std::make_unique<Impl>(std::move(info), cfg, clock, health, telemetry,
                                   health_cfg, validator_cfg)) {}

ImuSource::~ImuSource() = default;

void ImuSource::set_callback(Callback cb) { impl_->cb = std::move(cb); }
void ImuSource::start() {}
void ImuSource::stop() {}
SensorInfo ImuSource::info() const { return impl_->base.info(); }

void ImuSource::ingest_raw(const RawImuFrame& f) {
  StampSource src;
  Timestamp base;
  if (f.has_device_ns && impl_->clock_) {
    // A disciplined clock reports its hardware mechanism; an undisciplined one stays on
    // SwOffset, which is identity on an unseeded clock so the recorded stamp passes
    // through. Either way the device stamp keeps every modality on one timeline.
    src = impl_->clock_->disciplined(ClockId::Imu)
              ? impl_->clock_->stamp_source(ClockId::Imu)
              : StampSource::SwOffset;
    base = impl_->base.stamp_from(f.device_ns, ClockId::Imu, src);
    impl_->base.clear(HealthCode::ImuNoDeviceClock, f.host_arrival);
  } else {
    // The IMU is the worst sensor to leave on arrival time: its high rate makes
    // transport jitter a large fraction of the sample interval, so flag it loudly.
    src = StampSource::ArrivalOnly;
    base = f.host_arrival;
    impl_->base.raise(HealthCode::ImuNoDeviceClock, f.host_arrival);
  }

  ImuSample s;
  s.sensor_id = impl_->base.info().id;
  s.acc = f.acc;    // raw specific force [m/s^2], gravity + bias included
  s.gyro = f.gyro;  // raw angular rate [rad/s]
  // The stamp denotes the end of the integration interval; the configured shift maps a
  // device's mid/start-of-interval convention onto interval-end.
  s.stamp = impl_->base.note_stamp(base + impl_->cfg.interval_end_shift_ns, src);

  // A non-finite acc/gyro component would become a corrupt spline residual; reject it.
  if (!impl_->base.accept_imu(s)) return;
  if (impl_->cb) impl_->cb(std::move(s));
}

// ---------------------------------------------------------------------------
// CameraSource
// ---------------------------------------------------------------------------

struct CameraSource::Impl {
  Impl(SensorInfo info, const CameraSensorConfig& cfg, ClockModel* clock,
       HealthSink* health, TelemetrySink* telemetry, const TimeHealth& health_cfg,
       const ValidatorConfig& validator_cfg)
      : base(std::move(info), clock, health, telemetry, health_cfg.failed_timeout_ms,
             health_cfg.rate_tolerance_frac, validator_cfg),
        clock_(clock),
        cfg(cfg) {}

  SourceBase base;
  ClockModel* clock_;
  CameraSensorConfig cfg;
  Callback cb;
};

CameraSource::CameraSource(SensorInfo info, const CameraSensorConfig& cfg,
                           ClockModel* clock, HealthSink* health,
                           TelemetrySink* telemetry, const TimeHealth& health_cfg,
                           const ValidatorConfig& validator_cfg)
    : impl_(std::make_unique<Impl>(std::move(info), cfg, clock, health, telemetry,
                                   health_cfg, validator_cfg)) {}

CameraSource::~CameraSource() = default;

void CameraSource::set_callback(Callback cb) { impl_->cb = std::move(cb); }
void CameraSource::start() {}
void CameraSource::stop() {}
SensorInfo CameraSource::info() const { return impl_->base.info(); }

void CameraSource::ingest_raw(const RawCameraFrame& f) {
  // Mid-exposure is the correct instant for a global shutter; the GPIO-trigger match
  // is the seam that is not yet wired, so the base instant comes from the device clock
  // (SwOffset) or host arrival, and the mid-exposure shift is always applied.
  StampSource src;
  Timestamp base;
  if (f.has_device_ns && impl_->clock_) {
    // A disciplined clock reports its hardware mechanism; an undisciplined one stays on
    // SwOffset, which is identity on an unseeded clock so the recorded stamp passes
    // through. Arrival time is reserved for frames carrying no device stamp.
    src = impl_->clock_->disciplined(ClockId::Cam)
              ? impl_->clock_->stamp_source(ClockId::Cam)
              : StampSource::SwOffset;
    base = impl_->base.stamp_from(f.device_ns, ClockId::Cam, src);
  } else {
    src = StampSource::ArrivalOnly;
    base = f.host_arrival;
  }

  CameraFrame frame;
  frame.sensor_id = impl_->base.info().id;
  frame.sensor_frame = impl_->base.info().sensor_frame;
  frame.width = f.width;
  frame.height = f.height;
  frame.encoding = f.encoding;
  frame.data = f.data;  // shared-immutable, passed through; no debayer, no pyramid
  frame.exposure_s = f.exposure_s;
  frame.gain = f.gain;

  if (f.exposure_s <= 0.f) {
    // Without exposure the mid-exposure shift cannot be applied; downstream falls back
    // to robust photometric weighting.
    impl_->base.raise(HealthCode::CamNoExposure, base);
    frame.stamp = impl_->base.note_stamp(base, src);
  } else {
    impl_->base.clear(HealthCode::CamNoExposure, base);
    frame.stamp = impl_->base.note_stamp(base + mid_exposure_shift_ns(f.exposure_s), src);
  }

  if (impl_->cb) impl_->cb(std::move(frame));
}

// ---------------------------------------------------------------------------
// GnssSource
// ---------------------------------------------------------------------------

struct GnssSource::Impl {
  Impl(SensorInfo info, const GnssSensorConfig& cfg, ClockModel* clock,
       HealthSink* health, TelemetrySink* telemetry, const TimeHealth& health_cfg,
       const ValidatorConfig& validator_cfg)
      : base(std::move(info), clock, health, telemetry, health_cfg.failed_timeout_ms,
             health_cfg.rate_tolerance_frac, validator_cfg),
        clock_(clock),
        cfg(cfg) {}

  SourceBase base;
  ClockModel* clock_;
  GnssSensorConfig cfg;
  Callback cb;
  bool pps_lost = false;  // true once the rig has dropped to the no-PPS state
};

GnssSource::GnssSource(SensorInfo info, const GnssSensorConfig& cfg, ClockModel* clock,
                       HealthSink* health, TelemetrySink* telemetry,
                       const TimeHealth& health_cfg,
                       const ValidatorConfig& validator_cfg)
    : impl_(std::make_unique<Impl>(std::move(info), cfg, clock, health, telemetry,
                                   health_cfg, validator_cfg)) {}

GnssSource::~GnssSource() = default;

void GnssSource::set_callback(Callback cb) { impl_->cb = std::move(cb); }
void GnssSource::start() {}
void GnssSource::stop() {}
SensorInfo GnssSource::info() const { return impl_->base.info(); }

void GnssSource::ingest_raw(const RawGnssFrame& f) {
  // The fix is stamped to the PPS edge it references. Without a PPS reference the rig
  // loses its absolute anchor; the fix degrades to the software offset against the
  // host clock and GnssPpsLost is raised.
  StampSource src;
  Timestamp base;
  if (f.has_pps_reference) {
    src = StampSource::HwPps;
    base = f.gps_second_ns;
    impl_->base.clear(HealthCode::GnssPpsLost, base);
    impl_->pps_lost = false;
  } else if (f.has_device_ns && impl_->clock_) {
    src = StampSource::SwOffset;
    base = impl_->base.stamp_from(f.device_ns, ClockId::Gnss, src);
    // Raise only on the edge into the no-PPS state, not on every subsequent fix.
    if (!impl_->pps_lost) {
      impl_->base.raise(HealthCode::GnssPpsLost, base);
      impl_->pps_lost = true;
    }
  } else {
    src = StampSource::ArrivalOnly;
    base = f.host_arrival;
    if (!impl_->pps_lost) {
      impl_->base.raise(HealthCode::GnssPpsLost, base);
      impl_->pps_lost = true;
    }
  }

  GnssFix fix;
  fix.sensor_id = impl_->base.info().id;
  fix.sensor_frame = impl_->base.info().sensor_frame;
  fix.stamp = impl_->base.note_stamp(base, src);
  fix.lat_deg = f.lat_deg;   // raw geodetic; no metric/ENU conversion here
  fix.lon_deg = f.lon_deg;
  fix.alt_m = f.alt_m;
  fix.cov_enu = f.cov_enu;
  fix.fix = f.fix;           // fix-quality gating is the back-end's job
  fix.num_sats = f.num_sats;

  if (impl_->cb) impl_->cb(std::move(fix));
}

}  // namespace meridian
