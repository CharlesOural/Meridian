#include "source_base.hpp"

#include <utility>

namespace meridian {

SourceBase::SourceBase(SensorInfo info, ClockModel* clock, HealthSink* health,
                       TelemetrySink* telemetry, double failed_timeout_ms,
                       double rate_tolerance_frac, const ValidatorConfig& validator_cfg)
    : info_(std::move(info)),
      clock_(clock),
      validator_(info_, validator_cfg, clock, health, telemetry, failed_timeout_ms,
                 rate_tolerance_frac) {}

Timestamp SourceBase::stamp_from(Timestamp vendor_ns, ClockId clock, StampSource src) {
  // Arrival and replay stamps are already on the host/recorded timeline; everything else
  // is a device-clock value the model corrects onto the Meridian timeline. This is a pure
  // conversion; the validator runs separately, once, in note_stamp.
  if (src == StampSource::ArrivalOnly || src == StampSource::Replay) {
    return vendor_ns;
  }
  return clock_ ? clock_->to_meridian(vendor_ns, clock) : vendor_ns;
}

Timestamp SourceBase::note_stamp(Timestamp proposed, StampSource src) {
  return validator_.on_stamp(proposed, src).stamp;
}

bool SourceBase::accept_lidar(LidarScan& scan) {
  return validator_.on_lidar(scan).kind != InputValidator::Verdict::Reject;
}

bool SourceBase::accept_imu(const ImuSample& s) {
  return validator_.on_imu(s).kind != InputValidator::Verdict::Reject;
}

void SourceBase::raise(HealthCode code, Timestamp t) { validator_.raise(code, t); }
void SourceBase::clear(HealthCode code, Timestamp t) { validator_.clear(code, t); }

}  // namespace meridian
