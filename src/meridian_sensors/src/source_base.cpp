#include "source_base.hpp"

#include <utility>

#include "meridian/debug/telemetry.hpp"

namespace meridian {

namespace {

// Codes that, while active, hold the sensor at Failed: a persistently empty or
// time-less scan means there is nothing usable to fuse.
bool is_failed_code(HealthCode code) {
  switch (code) {
    case HealthCode::EmptyScan:
    case HealthCode::LidarNoPointTime:
      return true;
    default:
      return false;
  }
}

// Codes that, while active, hold the sensor at Degraded: data is still usable but
// down-weighted. Anything raised and not failing degrades the sensor.
bool is_degraded_code(HealthCode code) {
  return code != HealthCode::None && !is_failed_code(code);
}

// Smoothing factor for the rate estimate: one inter-sample gap moves the estimate a
// tenth of the way, so transport jitter cannot flip the rate codes on a single gap.
constexpr double kRateAlpha = 0.1;

}  // namespace

SourceBase::SourceBase(SensorInfo info, ClockModel* clock, HealthSink* health,
                       TelemetrySink* telemetry, double failed_timeout_ms,
                       double rate_tolerance_frac)
    : info_(std::move(info)),
      clock_(clock),
      health_(health),
      telemetry_(telemetry),
      rate_tolerance_frac_(rate_tolerance_frac) {
  // A gap beyond a few nominal periods is a dropout. The failed timeout is a hard
  // ceiling; below it, k = 2.5 nominal periods triggers the dropout flag.
  if (info_.nominal_rate_hz > 0.0) {
    const double period_ns = 1e9 / info_.nominal_rate_hz;
    dropout_gap_ns_ = static_cast<Duration>(2.5 * period_ns);
  } else {
    dropout_gap_ns_ = static_cast<Duration>(failed_timeout_ms * 1e6);
  }

  health_state_.sensor_id = info_.id;
  health_state_.modality = info_.modality;
  health_state_.level = HealthLevel::Nominal;
  health_state_.stamp_src = info_.configured_stamp_source;
}

Timestamp SourceBase::stamp_from(Timestamp vendor_ns, ClockId clock, StampSource src) {
  // Arrival and replay stamps are already on the host/recorded timeline; everything
  // else is a device-clock value the model corrects onto the Meridian timeline.
  if (src == StampSource::ArrivalOnly || src == StampSource::Replay) {
    return vendor_ns;
  }
  return clock_ ? clock_->to_meridian(vendor_ns, clock) : vendor_ns;
}

Timestamp SourceBase::enforce_monotonic(Timestamp t) {
  if (t < last_stamp_) {
    // A backwards stamp would corrupt preintegration and knot ordering. Clamp to the
    // last emitted stamp and report the regression rather than emit it.
    raise(HealthCode::ClockStepDetected, last_stamp_);
    return last_stamp_;
  }
  last_stamp_ = t;
  return t;
}

void SourceBase::note_sample(Timestamp t, StampSource src) {
  health_state_.stamp_src = src;
  health_state_.last_sample = t;

  if (have_prev_sample_) {
    const Duration gap = t - prev_sample_;
    if (gap > 0) {
      const double inst_hz = 1e9 / static_cast<double>(gap);
      rate_hz_ =
          rate_hz_ > 0.0 ? rate_hz_ + kRateAlpha * (inst_hz - rate_hz_) : inst_hz;
    }
    health_state_.rate_hz = rate_hz_;

    if (gap > dropout_gap_ns_) {
      raise(HealthCode::Dropout, t);
    } else {
      clear(HealthCode::Dropout, t);
    }

    if (info_.nominal_rate_hz > 0.0 && rate_hz_ > 0.0) {
      const double lo = info_.nominal_rate_hz * (1.0 - rate_tolerance_frac_);
      const double hi = info_.nominal_rate_hz * (1.0 + rate_tolerance_frac_);
      if (rate_hz_ < lo) {
        clear(HealthCode::RateHigh, t);
        raise(HealthCode::RateLow, t);
      } else if (rate_hz_ > hi) {
        clear(HealthCode::RateLow, t);
        raise(HealthCode::RateHigh, t);
      } else {
        clear(HealthCode::RateLow, t);
        clear(HealthCode::RateHigh, t);
      }
    }
  }

  // A configured hardware mechanism that has dropped to software offset is a degraded
  // timing state; raise it so the level reflects the loss of hardware sync.
  if (src == StampSource::SwOffset &&
      info_.configured_stamp_source != StampSource::SwOffset &&
      info_.configured_stamp_source != StampSource::ArrivalOnly) {
    raise(HealthCode::SyncLost, t);
  } else if (src == info_.configured_stamp_source) {
    clear(HealthCode::SyncLost, t);
  }

  // Pull the clock model's offset uncertainty into the snapshot so it flows to L2.
  if (clock_) {
    const ClockState cs = clock_->state(modality_clock());
    health_state_.offset_ns = cs.disciplined ? 0.0 : cs.offset_ns;
    health_state_.offset_std_ns = cs.offset_std_ns;
  }

  prev_sample_ = t;
  have_prev_sample_ = true;

  publish_health(t);

  if (telemetry_ && telemetry_->enabled("sensors/rate")) {
    telemetry_->scalar("sensors/rate", rate_hz_, t);
  }
}

void SourceBase::raise(HealthCode code, Timestamp t) {
  const std::uint32_t bit = health_code_bit(code);
  if ((health_state_.code_bits & bit) != 0) return;  // already active
  health_state_.code_bits |= bit;
  publish_health(t);
  if (health_) health_->degrade(info_.id, code);
}

void SourceBase::clear(HealthCode code, Timestamp t) {
  const std::uint32_t bit = health_code_bit(code);
  if ((health_state_.code_bits & bit) == 0) return;
  health_state_.code_bits &= ~bit;
  publish_health(t);
}

void SourceBase::publish_health(Timestamp t) {
  // The dominant code is the lowest-numbered active code; the level is the worst the
  // active set implies.
  HealthLevel level = HealthLevel::Nominal;
  HealthCode dominant = HealthCode::None;
  for (std::uint16_t i = 1;
       i <= static_cast<std::uint16_t>(HealthCode::EmptyScan); ++i) {
    const auto code = static_cast<HealthCode>(i);
    if ((health_state_.code_bits & health_code_bit(code)) == 0) continue;
    if (dominant == HealthCode::None) dominant = code;
    if (is_failed_code(code)) {
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

ClockId SourceBase::modality_clock() const {
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

}  // namespace meridian
