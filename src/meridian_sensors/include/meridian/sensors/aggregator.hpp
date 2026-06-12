#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <utility>

#include "meridian/common/measure_group.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"
#include "meridian/sensors/health.hpp"

namespace meridian {

class TelemetrySink;

// Bundles per-sweep measurements into a MeasureGroup. It is fed every sample from
// every source (after the lossy sensor queue) and emits one group per LiDAR sweep
// once the spanning IMU is present, or after a timeout. Aggregation is a pure timing
// operation: which samples land in a group is decided entirely by interval
// containment on the Meridian timeline.
//
// Grouping rules:
//   - a group closes when the LiDAR sweep [t_begin, t_end] is complete AND IMU has
//     been received up to t_end (imu.back().stamp >= t_end); a timeout of
//     agg.max_wait_ms after the sweep arrives forces emission with whatever IMU
//     arrived and flags ImuLate,
//   - the IMU set is every sample with stamp in (prev_t_end, t_end] plus the one
//     sample straddling t_begin (the last sample at or before t_begin),
//   - the image is the one whose mid-exposure stamp falls in [t_begin, t_end],
//   - GNSS fixes whose stamp falls in [t_begin, t_end] are included,
//   - each modality keeps a bounded reorder buffer (agg.reorder_ms); a sample older
//     than the emitted watermark by more than that window is dropped (LateDrop),
//   - groups are emitted in stamp order.
//
// Thread-confined: every method runs on the single stage thread that drains the
// sensor queue. Borrowed pointers (HealthSink*, TelemetrySink*) are non-owning.
class Aggregator {
 public:
  Aggregator(const AggregationConfig& agg, const SensorsConfig& sensors,
             HealthSink* health, TelemetrySink* telemetry);

  using GroupSink = std::function<void(MeasureGroup&&)>;
  void set_sink(GroupSink sink);

  // Fed every sample from every source's callback. Each call may close and emit a
  // group through the sink.
  void on(ImuSample&& s);
  void on(LidarScan&& s);
  void on(CameraFrame&& s);
  void on(GnssFix&& s);

 private:
  // A LiDAR sweep awaiting its spanning IMU, with the arrival watermark it was staged at
  // so the timeout is measured from sweep arrival rather than from t_begin.
  struct PendingSweep {
    LidarScan scan;
    Timestamp t_begin = 0;
    Timestamp t_end = 0;
    Timestamp received_meridian_ns = 0;  // arrival watermark at staging, timeout reference
  };

  // True once `t` is past the reorder window behind the modality watermark.
  bool past_reorder_window(Timestamp watermark, Timestamp t) const;

  // Build and emit the group for `sweep`, draining the spanning IMU/image/GNSS, then
  // advance prev_t_end_. `imu_late` flags a timeout emission.
  void emit_group(PendingSweep&& sweep, bool imu_late);

  // Try to close the head pending sweep: emit if IMU covers t_end or the timeout has
  // elapsed against `now_ns` (the stamp of the sample currently being processed).
  void try_close(Timestamp now_ns);

  // Raise a code on a sensor, increment its per-code occurrence counter, and surface the
  // running count as sensors/validator/<sensor>/<code>_count.
  void flag(std::uint8_t sensor_id, Modality modality, HealthCode code, Timestamp t);

  // Drop the oldest entries of a staging deque until it is within `cap`. Each call that
  // had to drop raises Dropout once for the modality so a saturated stage is visible.
  template <typename T>
  void bound_deque(std::deque<T>& dq, std::size_t cap, std::uint8_t sensor_id,
                   Modality modality, Timestamp t);

  AggregationConfig agg_;
  SensorsConfig sensors_;
  HealthSink* health_;        // borrowed
  TelemetrySink* telemetry_;  // borrowed
  GroupSink sink_;

  // Per-modality time-ordered staging buffers. IMU is kept across sweeps so the
  // straddling sample survives; the rest are drained per group.
  std::deque<ImuSample> imu_;
  std::deque<CameraFrame> images_;
  std::deque<GnssFix> gnss_;
  std::deque<PendingSweep> sweeps_;

  // Highest stamp emitted (or accepted into staging) per modality; the reorder window
  // is measured behind this.
  Timestamp imu_watermark_ = 0;
  Timestamp image_watermark_ = 0;
  Timestamp gnss_watermark_ = 0;
  Timestamp lidar_watermark_ = 0;

  // End of the previous emitted sweep; the lower (exclusive) edge of the IMU set.
  Timestamp prev_t_end_ = 0;
  bool have_prev_t_end_ = false;

  Duration max_wait_ns_ = 0;
  Duration reorder_ns_ = 0;

  // Per-(sensor, code) occurrence count of the flags the aggregator raises (LateDrop,
  // ImuLate, staging-overflow Dropout), surfaced as the validator count telemetry.
  std::map<std::pair<std::uint8_t, std::uint16_t>, std::uint64_t> flag_counts_;
};

}  // namespace meridian
