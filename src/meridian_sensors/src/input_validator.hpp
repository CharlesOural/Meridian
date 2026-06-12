#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"
#include "meridian/sensors/health.hpp"
#include "meridian/sensors/sensor_info.hpp"
#include "meridian/time/clock_model.hpp"
#include "meridian/time/stamp_source.hpp"

namespace meridian {

class TelemetrySink;

// The single standing stamp-integrity gate every sample of every stream passes through
// before it reaches the source callback. One instance per sensor_id. It is the engine
// behind SourceBase's stamp gate and the per-modality content checks (on_lidar / on_imu),
// pulled out as one named, uniformly-testable component so the one place that decides
// "this stamp is trustworthy" is identical across all four modalities.
//
// It runs always-on (not a debug option): a single out-of-order or skewed stamp does not
// merely add one noisy measurement, it corrupts the IMU integration interval and the
// per-point deskew times, bending the pose for every sample that spans it.
//
// Six standing checks, each mapped to a HealthCode and edge-throttled: the first sample
// entering a fault state raises the code once (one event, one bit set); subsequent samples
// in the same state increment a per-code occurrence counter and emit no further event; the
// transition out of the state clears the code once. The running counts surface as the
// sensors/validator/<sensor>/<code>_count telemetry.
//
// Thread-confined to the owning source's acquisition thread: last_stamp_, the active-code
// bitset, and the counters are touched only on the stamping path. Borrowed pointers
// (ClockModel*, HealthSink*, TelemetrySink*) are non-owning, owned by the pipeline.
class InputValidator {
 public:
  InputValidator(SensorInfo info, const ValidatorConfig& cfg, ClockModel* clock,
                 HealthSink* health, TelemetrySink* telemetry, double failed_timeout_ms,
                 double rate_tolerance_frac);

  // The verdict a check returns. A Reject means the sample never reaches the callback
  // (nor Q_sensors); a Clamped means the stamp was repaired to preserve monotonicity and
  // the repair was reported. The carried stamp is the (possibly clamped) value to emit.
  struct Verdict {
    Timestamp stamp = 0;
    enum Kind { Accept, Clamped, Reject } kind = Accept;
  };

  // Standing stamp checks for every modality, run before the typed content check.
  // Rewind (clamp + ClockStepDetected), Gap (Dropout), Skew (SkewOutOfRange), plus the
  // rate-band and sync-state bookkeeping that feed the health snapshot. Returns the
  // (possibly clamped) stamp; the caller emits that value, never the proposed one.
  Verdict on_stamp(Timestamp proposed, StampSource src);

  // LiDAR content check (per-point time, NaN/Inf ratio, emptiness). Non-finite points are
  // dropped in place; the scan is rejected only if it had no per-point time, was empty to
  // begin with, or became empty after the drop. Mutates `scan.points` when it drops.
  Verdict on_lidar(LidarScan& scan);

  // IMU content check: any non-finite acc/gyro component rejects the sample.
  Verdict on_imu(const ImuSample& s);

  // Raise/clear a code directly (sync transitions, per-sensor specifics the source owns).
  // Both are edge-throttled and update the published snapshot.
  void raise(HealthCode code, Timestamp t);
  void clear(HealthCode code, Timestamp t);

  const SensorHealth& health_state() const { return health_state_; }

 private:
  // Number of distinct HealthCode enumerators, sized for the per-code counter array.
  static constexpr std::size_t kCodeCount =
      static_cast<std::size_t>(HealthCode::ImuNonFinite) + 1;

  // Edge-throttled fault transition. enter() raises the code once and resets its counter;
  // every subsequent call while the code is active increments the counter and emits the
  // running count to telemetry; exit() clears the code once.
  void enter(HealthCode code, Timestamp t);
  void exit(HealthCode code, Timestamp t);

  // Recompute the dominant code and level from the active bitset and publish the snapshot.
  void publish_health(Timestamp t);

  // The ClockId for this sensor's modality (for offset/skew lookups).
  ClockId modality_clock() const;

  // Emit the sensors/validator/<sensor>/<code>_count scalar for `code`.
  void emit_count(HealthCode code, Timestamp t);

  // Emit the per-sensor sensors/rate/<sensor> measured-rate scalar.
  void emit_rate(double rate_hz, Timestamp t);

  bool code_active(HealthCode code) const {
    return (health_state_.code_bits & health_code_bit(code)) != 0;
  }

  SensorInfo info_;
  ValidatorConfig cfg_;
  ClockModel* clock_;         // borrowed
  HealthSink* health_;        // borrowed
  TelemetrySink* telemetry_;  // borrowed

  // Stamp-integrity state (moved here from SourceBase): the last emitted stamp gates
  // monotonicity and the inter-sample gap gates dropout.
  Timestamp last_stamp_ = std::numeric_limits<Timestamp>::min();
  Timestamp prev_sample_ = 0;
  bool have_prev_sample_ = false;

  // Rate/dropout state.
  double rate_hz_ = 0.0;
  Duration dropout_gap_ns_ = 0;  // inter-sample gap above which a dropout is flagged
  Duration failed_gap_ns_ = 0;   // gap above which a sustained dropout escalates to Failed
  bool failed_gap_active_ = false;  // the active Dropout was a failed-timeout-class gap
  double rate_tolerance_frac_ = 0.20;

  // Per-code occurrence counters behind the edge-throttled events; the count of samples
  // seen while each code was active, including the entry sample.
  std::array<std::uint64_t, kCodeCount> code_counts_{};

  SensorHealth health_state_{};
};

}  // namespace meridian
