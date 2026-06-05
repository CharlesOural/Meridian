#pragma once

#include <cstdint>
#include <limits>

#include "meridian/common/time.hpp"
#include "meridian/sensors/health.hpp"
#include "meridian/sensors/sensor_info.hpp"
#include "meridian/time/clock_model.hpp"
#include "meridian/time/stamp_source.hpp"

namespace meridian {

class TelemetrySink;

// Shared per-source machinery composed (not inherited through the interface) by every
// concrete source. It converts a vendor instant on a named clock to the Meridian
// timeline, enforces per-sensor monotonicity, and tracks rate/dropout/sync state into
// health. The interface stays clean because sources hold a SourceBase member rather
// than deriving the interface from it.
//
// Thread-confined to the owning source's acquisition thread. ClockModel/HealthSink/
// TelemetrySink pointers are borrowed (non-owning), owned by the pipeline.
class SourceBase {
 public:
  SourceBase(SensorInfo info, ClockModel* clock, HealthSink* health,
             TelemetrySink* telemetry, double failed_timeout_ms,
             double rate_tolerance_frac);

  // Convert a vendor instant on a named clock to a Meridian-timeline stamp, applying
  // the ClockModel correction and recording the stamping mechanism for health. For
  // Replay and ArrivalOnly the value is taken as-is (no clock correction).
  Timestamp stamp_from(Timestamp vendor_ns, ClockId clock, StampSource src);

  // Enforce non-decreasing stamps per sensor. A regression is clamped up to the last
  // stamp and reported as a clock step; the clamped value is returned.
  Timestamp enforce_monotonic(Timestamp t);

  // Per-sample bookkeeping: measured rate, dropout against nominal_rate_hz, and the
  // current stamping mechanism, pushed to the health snapshot.
  void note_sample(Timestamp t, StampSource src);

  const SensorInfo& info() const { return info_; }

  // Raise a code into the active set and recompute the dominant level, then push the
  // updated snapshot. Used by sources for content/sync conditions.
  void raise(HealthCode code, Timestamp t);

  // Clear a previously-raised code (e.g. sync reacquired) and recompute the level.
  void clear(HealthCode code, Timestamp t);

 private:
  // Recompute level from the active code bitset under the level policy and publish.
  void publish_health(Timestamp t);

  // The ClockId for this sensor's modality.
  ClockId modality_clock() const;

  SensorInfo info_;
  ClockModel* clock_;         // borrowed
  HealthSink* health_;        // borrowed
  TelemetrySink* telemetry_;  // borrowed

  Timestamp last_stamp_ = std::numeric_limits<Timestamp>::min();

  // Rate/dropout state.
  Timestamp prev_sample_ = 0;
  bool have_prev_sample_ = false;
  double rate_hz_ = 0.0;
  Duration dropout_gap_ns_ = 0;  // inter-sample gap above which a dropout is flagged
  double rate_tolerance_frac_ = 0.20;

  // Current health snapshot, mutated incrementally.
  SensorHealth health_state_{};
};

}  // namespace meridian
