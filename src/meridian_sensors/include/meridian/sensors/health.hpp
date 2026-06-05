#pragma once

#include <cstdint>
#include <vector>

#include "meridian/common/status.hpp"
#include "meridian/common/time.hpp"
#include "meridian/sensors/sensor_info.hpp"
#include "meridian/time/stamp_source.hpp"

namespace meridian {

// A specific health condition raised on a sensor. Grouped by cause: timing, dataflow,
// per-sensor specifics, and scan content. Enumerators are sequential so they can index
// the SensorHealth::code_bits bitset.
enum class HealthCode : std::uint16_t {
  None = 0,
  // timing
  NoSync,
  SyncLost,
  SyncResidualHigh,
  ClockStepDetected,
  SkewOutOfRange,
  // dataflow
  Dropout,
  RateLow,
  RateHigh,
  LateDrop,
  ImuLate,
  // per-sensor specifics
  ImuNoDeviceClock,
  CamNoExposure,
  CamTriggerMismatch,
  GnssPpsLost,
  GnssFixDropped,
  // content
  LidarNoPointTime,
  LidarHighNanRatio,
  EmptyScan,
};

// Per-sensor health snapshot. `code` is the dominant active condition; `code_bits` is
// the OR of (1u << code) over every condition currently active, so consumers can see
// the full set. Timing fields feed both the operator surface and the L2 noise model.
struct SensorHealth {
  std::uint8_t sensor_id = 0;
  Modality modality = Modality::Imu;
  HealthLevel level = HealthLevel::Nominal;
  HealthCode code = HealthCode::None;                  // dominant active code
  std::uint32_t code_bits = 0;                         // bitset of all active codes
  StampSource stamp_src = StampSource::ArrivalOnly;    // current stamping mechanism
  double rate_hz = 0.0;                                // measured rate [Hz]
  double offset_ns = 0.0;                              // ClockModel offset [ns], 0 if disciplined
  double offset_std_ns = 0.0;                          // timing uncertainty [ns]
  Timestamp last_sample = 0;
  Timestamp since = 0;                                 // when the current level began
};

// Set the bit for `code` in a SensorHealth bitset. None has no bit.
inline std::uint32_t health_code_bit(HealthCode code) {
  return code == HealthCode::None
             ? 0u
             : (1u << static_cast<std::uint32_t>(code));
}

// Sink the sources and aggregator push health to. The concrete implementation (a
// meridian_debug bridge to telemetry) is bound by the pipeline; sources borrow a
// non-owning pointer to it.
class HealthSink {
 public:
  virtual ~HealthSink() = default;

  // Replace the snapshot for one sensor.
  virtual void update(const SensorHealth&) = 0;

  // Raise a single code on a sensor without rebuilding the whole snapshot.
  virtual void degrade(std::uint8_t id, HealthCode code) = 0;
};

// Rig-level rollup the pipeline and back-end can query. `timeline_absolute` is true
// only while a GPS/PPS anchor is present; without it the timeline is internally
// consistent but loses absolute meaning.
struct RigHealth {
  HealthLevel worst = HealthLevel::Nominal;
  std::vector<SensorHealth> sensors;
  bool timeline_absolute = false;
  double max_offset_std_ns = 0.0;  // worst timing uncertainty across sensors [ns]
};

}  // namespace meridian
