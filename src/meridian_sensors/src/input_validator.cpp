#include "input_validator.hpp"

#include <cmath>
#include <string>
#include <utility>

#include "meridian/debug/telemetry.hpp"

namespace meridian {

namespace {

// Codes that, while active, hold the sensor at Failed: a persistently empty or time-less
// scan means there is nothing usable to fuse.
bool is_failed_code(HealthCode code) {
  switch (code) {
    case HealthCode::EmptyScan:
    case HealthCode::LidarNoPointTime:
      return true;
    default:
      return false;
  }
}

// Anything raised and not failing degrades the sensor: data is still usable but
// down-weighted.
bool is_degraded_code(HealthCode code) {
  return code != HealthCode::None && !is_failed_code(code);
}

// Smoothing factor for the rate estimate: one inter-sample gap moves the estimate a tenth
// of the way, so transport jitter on a single gap cannot flip the rate codes.
constexpr double kRateAlpha = 0.1;

}  // namespace

InputValidator::InputValidator(SensorInfo info, const ValidatorConfig& cfg,
                               ClockModel* clock, HealthSink* health,
                               TelemetrySink* telemetry, double failed_timeout_ms,
                               double rate_tolerance_frac)
    : info_(std::move(info)),
      cfg_(cfg),
      clock_(clock),
      health_(health),
      telemetry_(telemetry),
      rate_tolerance_frac_(rate_tolerance_frac) {
  // A gap beyond gap_periods nominal periods is a dropout. When the nominal rate is
  // unset the failed timeout is the only available ceiling, so fall back to it.
  failed_gap_ns_ = static_cast<Duration>(failed_timeout_ms * 1e6);
  if (info_.nominal_rate_hz > 0.0) {
    const double period_ns = 1e9 / info_.nominal_rate_hz;
    dropout_gap_ns_ = static_cast<Duration>(cfg_.gap_periods * period_ns);
  } else {
    dropout_gap_ns_ = failed_gap_ns_;
  }

  health_state_.sensor_id = info_.id;
  health_state_.modality = info_.modality;
  health_state_.level = HealthLevel::Nominal;
  health_state_.stamp_src = info_.configured_stamp_source;
}

InputValidator::Verdict InputValidator::on_stamp(Timestamp proposed, StampSource src) {
  health_state_.stamp_src = src;

  Verdict v{proposed, Verdict::Accept};

  // Rewind: a stamp that does not strictly advance would corrupt preintegration and knot
  // ordering. Clamp it up to the last emitted stamp and report the regression; never emit
  // a regressing stamp. A sustained regression is edge-throttled and counted.
  if (proposed < last_stamp_) {
    enter(HealthCode::ClockStepDetected, last_stamp_);
    v.stamp = last_stamp_;
    v.kind = Verdict::Clamped;
  } else {
    exit(HealthCode::ClockStepDetected, proposed);
    last_stamp_ = proposed;
  }

  const Timestamp t = v.stamp;
  health_state_.last_sample = t;

  // Gap / rate band: both key on the inter-sample interval but on different signals. The
  // dropout uses the raw gap so a real hole is not smoothed away; the rate band uses a
  // smoothed estimate so transport jitter does not flip the band codes.
  if (have_prev_sample_) {
    const Duration gap = t - prev_sample_;
    if (gap > 0) {
      const double inst_hz = 1e9 / static_cast<double>(gap);
      rate_hz_ = rate_hz_ > 0.0 ? rate_hz_ + kRateAlpha * (inst_hz - rate_hz_) : inst_hz;
    }
    health_state_.rate_hz = rate_hz_;

    if (gap > dropout_gap_ns_) {
      // A gap past failed_gap_ns_ proves no sample arrived for longer than the failed
      // timeout, so this post-gap sample escalates the dropout from Degraded to Failed.
      failed_gap_active_ = gap > failed_gap_ns_;
      enter(HealthCode::Dropout, t);
    } else {
      failed_gap_active_ = false;
      exit(HealthCode::Dropout, t);
    }

    if (info_.nominal_rate_hz > 0.0 && rate_hz_ > 0.0) {
      const double lo = info_.nominal_rate_hz * (1.0 - rate_tolerance_frac_);
      const double hi = info_.nominal_rate_hz * (1.0 + rate_tolerance_frac_);
      if (rate_hz_ < lo) {
        exit(HealthCode::RateHigh, t);
        enter(HealthCode::RateLow, t);
      } else if (rate_hz_ > hi) {
        exit(HealthCode::RateLow, t);
        enter(HealthCode::RateHigh, t);
      } else {
        exit(HealthCode::RateLow, t);
        exit(HealthCode::RateHigh, t);
      }
    }
  }

  // A configured hardware mechanism that has dropped to software offset is a degraded
  // timing state; raise it so the level reflects the loss of hardware sync.
  if (src == StampSource::SwOffset &&
      info_.configured_stamp_source != StampSource::SwOffset &&
      info_.configured_stamp_source != StampSource::ArrivalOnly) {
    enter(HealthCode::SyncLost, t);
  } else if (src == info_.configured_stamp_source) {
    exit(HealthCode::SyncLost, t);
  }

  // Skew: crystal drift beyond the model's validity is a fault, not normal ppm wander.
  // Pull the clock model's offset uncertainty into the snapshot so it flows to L2, and
  // flag skew past the warn band.
  if (clock_) {
    const ClockState cs = clock_->state(modality_clock());
    health_state_.offset_ns = cs.disciplined ? 0.0 : cs.offset_ns;
    health_state_.offset_std_ns = cs.offset_std_ns;
    if (std::abs(cs.skew_ppm) > cfg_.skew_warn_ppm) {
      enter(HealthCode::SkewOutOfRange, t);
    } else {
      exit(HealthCode::SkewOutOfRange, t);
    }
  }

  prev_sample_ = t;
  have_prev_sample_ = true;

  publish_health(t);

  emit_rate(rate_hz_, t);
  return v;
}

InputValidator::Verdict InputValidator::on_lidar(LidarScan& scan) {
  const Timestamp t = scan.stamp_start;

  // Empty before any content inspection: a sweep with no returns cannot contribute.
  if (!scan.points || scan.points->empty()) {
    enter(HealthCode::EmptyScan, t);
    return {t, Verdict::Reject};
  }

  const PointCloud& src = *scan.points;
  const std::size_t n = src.size();

  // Per-point time: an Ouster always provides per-column time, so all-zero offsets mean a
  // malformed stream. The scan cannot be deskewed and is rejected rather than
  // reconstructed from geometry.
  bool any_point_time = false;
  std::size_t nan_count = 0;
  for (const LidarPoint& p : src) {
    if (p.t_offset_ns != 0) any_point_time = true;
    if (!p.xyz.allFinite()) ++nan_count;
  }
  if (!any_point_time) {
    enter(HealthCode::LidarNoPointTime, t);
    return {t, Verdict::Reject};
  }
  exit(HealthCode::LidarNoPointTime, t);

  // NaN/Inf policy: every non-finite return is dropped regardless of the warn band so no
  // downstream consumer ever sees a NaN coordinate; an entirely non-finite scan is
  // rejected as empty. The warn band only governs whether LidarHighNanRatio is raised.
  if (nan_count == n) {
    enter(HealthCode::EmptyScan, t);
    return {t, Verdict::Reject};
  }
  if (static_cast<double>(nan_count) / static_cast<double>(n) > cfg_.nan_ratio_warn) {
    enter(HealthCode::LidarHighNanRatio, t);
  } else {
    exit(HealthCode::LidarHighNanRatio, t);
  }
  if (nan_count > 0) {
    // Rebuild the cloud without the non-finite returns; the survivors replace the original
    // shared-immutable buffer so the keyframe store and L1 both see a finite cloud.
    auto kept = std::make_shared<PointCloud>();
    kept->reserve(n - nan_count);
    for (const LidarPoint& p : src) {
      if (p.xyz.allFinite()) kept->push_back(p);
    }
    scan.points = std::move(kept);
  }

  if (!scan.points || scan.points->empty()) {
    enter(HealthCode::EmptyScan, t);
    return {t, Verdict::Reject};
  }
  exit(HealthCode::EmptyScan, t);

  publish_health(t);
  return {t, Verdict::Accept};
}

InputValidator::Verdict InputValidator::on_imu(const ImuSample& s) {
  // A non-finite acc/gyro component is corrupt content; reject the sample so it never
  // becomes a spline residual.
  if (!s.acc.allFinite() || !s.gyro.allFinite()) {
    enter(HealthCode::ImuNonFinite, s.stamp);
    return {s.stamp, Verdict::Reject};
  }
  exit(HealthCode::ImuNonFinite, s.stamp);
  return {s.stamp, Verdict::Accept};
}

void InputValidator::raise(HealthCode code, Timestamp t) { enter(code, t); }
void InputValidator::clear(HealthCode code, Timestamp t) { exit(code, t); }

void InputValidator::enter(HealthCode code, Timestamp t) {
  if (code == HealthCode::None) return;
  const auto idx = static_cast<std::size_t>(code);
  const bool was_active = code_active(code);
  if (!was_active) {
    // First sample into the fault state: set the bit, reset the counter, emit one event.
    health_state_.code_bits |= health_code_bit(code);
    code_counts_[idx] = 0;
    publish_health(t);
    if (health_) health_->degrade(info_.id, code);
  }
  // Every sample in the state — including the entry sample — increments the count and
  // surfaces it, but emits no further event while the state persists.
  ++code_counts_[idx];
  emit_count(code, t);
}

void InputValidator::exit(HealthCode code, Timestamp t) {
  if (code == HealthCode::None) return;
  if (!code_active(code)) return;
  health_state_.code_bits &= ~health_code_bit(code);
  publish_health(t);
}

void InputValidator::publish_health(Timestamp t) {
  // The dominant code is the lowest-numbered active code; the level is the worst the
  // active set implies.
  HealthLevel level = HealthLevel::Nominal;
  HealthCode dominant = HealthCode::None;
  for (std::uint16_t i = 1; i <= static_cast<std::uint16_t>(HealthCode::ImuNonFinite); ++i) {
    const auto code = static_cast<HealthCode>(i);
    if (!code_active(code)) continue;
    if (dominant == HealthCode::None) dominant = code;
    // A dropout whose gap exceeded the failed timeout is itself a Failed-level condition;
    // a dropout within the band stays Degraded.
    const bool failed = is_failed_code(code) ||
                        (code == HealthCode::Dropout && failed_gap_active_);
    if (failed) {
      level = HealthLevel::Failed;
    } else if (is_degraded_code(code) && level == HealthLevel::Nominal) {
      level = HealthLevel::Degraded;
    }
  }

  if (level != health_state_.level) {
    health_state_.since = t;
  }
  health_state_.level = level;
  health_state_.code = dominant;

  if (health_) health_->update(health_state_);
}

ClockId InputValidator::modality_clock() const {
  switch (info_.modality) {
    case Modality::Lidar:
      return ClockId::Lidar;
    case Modality::Camera:
      return ClockId::Cam;
    case Modality::Gnss:
      return ClockId::Gnss;
    case Modality::Imu:
    default:
      return ClockId::Imu;
  }
}

namespace {

// Stable short names for the modality prefix and the fault code, used to build the
// per-sensor telemetry keys.
const char* modality_name(Modality m) {
  switch (m) {
    case Modality::Lidar:
      return "lidar";
    case Modality::Camera:
      return "cam";
    case Modality::Gnss:
      return "gnss";
    case Modality::Imu:
    default:
      return "imu";
  }
}

// The <sensor> token shared by every per-sensor key: <modality><id>, e.g. "lidar0".
std::string sensor_tag(const SensorInfo& info) {
  std::string tag = modality_name(info.modality);
  tag += std::to_string(static_cast<unsigned>(info.id));
  return tag;
}

const char* code_name(HealthCode code) {
  switch (code) {
    case HealthCode::NoSync:
      return "NoSync";
    case HealthCode::SyncLost:
      return "SyncLost";
    case HealthCode::SyncResidualHigh:
      return "SyncResidualHigh";
    case HealthCode::ClockStepDetected:
      return "ClockStepDetected";
    case HealthCode::SkewOutOfRange:
      return "SkewOutOfRange";
    case HealthCode::Dropout:
      return "Dropout";
    case HealthCode::RateLow:
      return "RateLow";
    case HealthCode::RateHigh:
      return "RateHigh";
    case HealthCode::LateDrop:
      return "LateDrop";
    case HealthCode::ImuLate:
      return "ImuLate";
    case HealthCode::ImuNoDeviceClock:
      return "ImuNoDeviceClock";
    case HealthCode::CamNoExposure:
      return "CamNoExposure";
    case HealthCode::CamTriggerMismatch:
      return "CamTriggerMismatch";
    case HealthCode::GnssPpsLost:
      return "GnssPpsLost";
    case HealthCode::GnssFixDropped:
      return "GnssFixDropped";
    case HealthCode::LidarNoPointTime:
      return "LidarNoPointTime";
    case HealthCode::LidarHighNanRatio:
      return "LidarHighNanRatio";
    case HealthCode::EmptyScan:
      return "EmptyScan";
    case HealthCode::ImuNonFinite:
      return "ImuNonFinite";
    case HealthCode::None:
    default:
      return "None";
  }
}

}  // namespace

void InputValidator::emit_count(HealthCode code, Timestamp t) {
  if (!telemetry_) return;
  // Build the per-code key once and gate on it so an idle key skips the format step.
  std::string key = "sensors/validator/";
  key += sensor_tag(info_);
  key += '/';
  key += code_name(code);
  key += "_count";
  if (telemetry_->enabled(key.c_str())) {
    telemetry_->scalar(key.c_str(),
                       static_cast<double>(code_counts_[static_cast<std::size_t>(code)]),
                       t);
  }
}

void InputValidator::emit_rate(double rate_hz, Timestamp t) {
  if (!telemetry_) return;
  std::string key = "sensors/rate/";
  key += sensor_tag(info_);
  if (telemetry_->enabled(key.c_str())) {
    telemetry_->scalar(key.c_str(), rate_hz, t);
  }
}

}  // namespace meridian
