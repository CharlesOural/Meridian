#pragma once

#include "input_validator.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"
#include "meridian/sensors/health.hpp"
#include "meridian/sensors/sensor_info.hpp"
#include "meridian/time/clock_model.hpp"
#include "meridian/time/stamp_source.hpp"

namespace meridian {

class TelemetrySink;

// Shared per-source machinery composed (not inherited through the interface) by every
// concrete source. It converts a vendor instant on a named clock to the Meridian
// timeline and routes every sample through the standing InputValidator before the
// callback. The interface stays clean because sources hold a SourceBase member rather
// than deriving the interface from it.
//
// Thread-confined to the owning source's acquisition thread. ClockModel/HealthSink/
// TelemetrySink pointers are borrowed (non-owning), owned by the pipeline.
class SourceBase {
 public:
  SourceBase(SensorInfo info, ClockModel* clock, HealthSink* health,
             TelemetrySink* telemetry, double failed_timeout_ms,
             double rate_tolerance_frac, const ValidatorConfig& validator_cfg);

  // Convert a vendor instant on a named clock to a Meridian-timeline stamp, applying the
  // ClockModel correction and recording the stamping mechanism for health. For Replay and
  // ArrivalOnly the value is taken as-is (no clock correction).
  Timestamp stamp_from(Timestamp vendor_ns, ClockId clock, StampSource src);

  // Run the standing stamp checks (monotonicity clamp, dropout, skew, rate band, sync
  // state) on a proposed stamp and return the validated value the source must emit. A
  // regression is clamped up to the last stamp and reported; the clamped value is
  // returned. This is the single per-sample stamp gate.
  Timestamp note_stamp(Timestamp proposed, StampSource src);

  // LiDAR content gate: per-point time, NaN/Inf ratio (drops offending points in place),
  // and emptiness. Returns false (and does not let the caller emit) when the scan is
  // rejected; mutates scan.points when it drops non-finite returns.
  bool accept_lidar(LidarScan& scan);

  // IMU content gate: rejects a sample carrying any non-finite acc/gyro component.
  bool accept_imu(const ImuSample& s);

  const SensorInfo& info() const { return info_; }

  // Raise a code into the active set (e.g. sync/exposure conditions the source owns); the
  // raise is edge-throttled and the per-code count is tracked by the validator.
  void raise(HealthCode code, Timestamp t);

  // Clear a previously-raised code (e.g. sync reacquired).
  void clear(HealthCode code, Timestamp t);

 private:
  SensorInfo info_;
  ClockModel* clock_;         // borrowed; for the device->Meridian conversion in stamp_from
  InputValidator validator_;  // owns last_stamp_, the active-code bitset, per-code counts
};

}  // namespace meridian
