#include "meridian/sensors/aggregator.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

#include "meridian/debug/telemetry.hpp"

namespace meridian {

namespace {

// Hard caps on the staging deques. A stage above its cap means a downstream stall or a
// missing modality; the oldest entries are dropped to keep memory and latency bounded.
constexpr std::size_t kImuCap = 8192;
constexpr std::size_t kImageCap = 16;
constexpr std::size_t kGnssCap = 64;
constexpr std::size_t kSweepCap = 8;

// Stable short names for the modality prefix and the fault code, used to build the
// sensors/validator/<sensor>/<code>_count telemetry key.
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

const char* code_name(HealthCode code) {
  switch (code) {
    case HealthCode::Dropout:
      return "Dropout";
    case HealthCode::LateDrop:
      return "LateDrop";
    case HealthCode::ImuLate:
      return "ImuLate";
    default:
      return "None";
  }
}

}  // namespace

Aggregator::Aggregator(const AggregationConfig& agg, const SensorsConfig& sensors,
                       HealthSink* health, TelemetrySink* telemetry)
    : agg_(agg),
      sensors_(sensors),
      health_(health),
      telemetry_(telemetry),
      max_wait_ns_(static_cast<Duration>(agg.max_wait_ms * 1e6)),
      reorder_ns_(static_cast<Duration>(agg.reorder_ms * 1e6)) {}

void Aggregator::set_sink(GroupSink sink) { sink_ = std::move(sink); }

bool Aggregator::past_reorder_window(Timestamp watermark, Timestamp t) const {
  // A sample is too late once it lies more than the reorder window behind the highest
  // stamp already accepted for its modality.
  return watermark > 0 && t < watermark - reorder_ns_;
}

void Aggregator::flag(std::uint8_t sensor_id, Modality modality, HealthCode code,
                      Timestamp t) {
  if (health_) health_->degrade(sensor_id, code);

  // Keep an exact occurrence count per (sensor, code) and surface it as the validator
  // count so a sustained aggregator fault is visible without one event per occurrence.
  const std::uint64_t count =
      ++flag_counts_[{sensor_id, static_cast<std::uint16_t>(code)}];
  if (telemetry_) {
    std::string key = "sensors/validator/";
    key += modality_name(modality);
    key += std::to_string(static_cast<unsigned>(sensor_id));
    key += '/';
    key += code_name(code);
    key += "_count";
    if (telemetry_->enabled(key.c_str())) {
      telemetry_->scalar(key.c_str(), static_cast<double>(count), t);
    }
  }
}

template <typename T>
void Aggregator::bound_deque(std::deque<T>& dq, std::size_t cap, std::uint8_t sensor_id,
                             Modality modality, Timestamp t) {
  if (dq.size() <= cap) {
    return;
  }
  // The head is the oldest after the sorted insert, so dropping from the front sheds the
  // stalest staged data. Flag once per overflow event, not once per dropped element.
  while (dq.size() > cap) {
    dq.pop_front();
  }
  flag(sensor_id, modality, HealthCode::Dropout, t);
}

void Aggregator::on(ImuSample&& s) {
  if (past_reorder_window(imu_watermark_, s.stamp)) {
    flag(s.sensor_id, Modality::Imu, HealthCode::LateDrop, s.stamp);
    return;
  }
  imu_watermark_ = std::max(imu_watermark_, s.stamp);

  // Insert in stamp order so the staging deque stays sorted under bounded reordering.
  const std::uint8_t sensor_id = s.sensor_id;
  auto pos = std::upper_bound(
      imu_.begin(), imu_.end(), s,
      [](const ImuSample& a, const ImuSample& b) { return a.stamp < b.stamp; });
  imu_.insert(pos, std::move(s));
  bound_deque(imu_, kImuCap, sensor_id, Modality::Imu, imu_watermark_);

  try_close(imu_watermark_);
}

void Aggregator::on(LidarScan&& s) {
  const Timestamp t_begin = s.stamp_start;
  const Timestamp t_end = s.stamp_start + s.sweep_duration;
  if (past_reorder_window(lidar_watermark_, t_begin)) {
    flag(s.sensor_id, Modality::Lidar, HealthCode::LateDrop, t_begin);
    return;
  }
  lidar_watermark_ = std::max(lidar_watermark_, t_begin);

  const std::uint8_t sensor_id = s.sensor_id;

  PendingSweep sweep;
  sweep.t_begin = t_begin;
  sweep.t_end = t_end;
  // Measure the timeout from when the sweep was staged, not from its content time, so a
  // sweep whose IMU never arrives ages against the arrival watermark.
  sweep.received_meridian_ns = std::max(imu_watermark_, lidar_watermark_);
  sweep.scan = std::move(s);

  // Sweeps arrive monotonically, but reorder defensively to keep the head the oldest.
  auto pos = std::upper_bound(
      sweeps_.begin(), sweeps_.end(), sweep,
      [](const PendingSweep& a, const PendingSweep& b) { return a.t_end < b.t_end; });
  sweeps_.insert(pos, std::move(sweep));
  bound_deque(sweeps_, kSweepCap, sensor_id, Modality::Lidar, lidar_watermark_);

  // The IMU watermark is the gating modality; closing against it alone keeps an early
  // image or GNSS stamp from forcing a premature ImuLate emission.
  try_close(imu_watermark_);
}

void Aggregator::on(CameraFrame&& s) {
  if (past_reorder_window(image_watermark_, s.stamp)) {
    flag(s.sensor_id, Modality::Camera, HealthCode::LateDrop, s.stamp);
    return;
  }
  image_watermark_ = std::max(image_watermark_, s.stamp);

  const std::uint8_t sensor_id = s.sensor_id;
  auto pos = std::upper_bound(
      images_.begin(), images_.end(), s,
      [](const CameraFrame& a, const CameraFrame& b) { return a.stamp < b.stamp; });
  images_.insert(pos, std::move(s));
  bound_deque(images_, kImageCap, sensor_id, Modality::Camera, image_watermark_);
}

void Aggregator::on(GnssFix&& s) {
  if (past_reorder_window(gnss_watermark_, s.stamp)) {
    flag(s.sensor_id, Modality::Gnss, HealthCode::LateDrop, s.stamp);
    return;
  }
  gnss_watermark_ = std::max(gnss_watermark_, s.stamp);

  const std::uint8_t sensor_id = s.sensor_id;
  auto pos = std::upper_bound(
      gnss_.begin(), gnss_.end(), s,
      [](const GnssFix& a, const GnssFix& b) { return a.stamp < b.stamp; });
  gnss_.insert(pos, std::move(s));
  bound_deque(gnss_, kGnssCap, sensor_id, Modality::Gnss, gnss_watermark_);
}

void Aggregator::try_close(Timestamp now_ns) {
  // Close as many head sweeps as their gates allow. The head is always the oldest, so
  // groups emit in stamp order.
  while (!sweeps_.empty()) {
    PendingSweep& head = sweeps_.front();

    const bool imu_covers = !imu_.empty() && imu_.back().stamp >= head.t_end;
    const bool timed_out = now_ns - head.received_meridian_ns >= max_wait_ns_;

    if (imu_covers) {
      PendingSweep sweep = std::move(sweeps_.front());
      sweeps_.pop_front();
      emit_group(std::move(sweep), /*imu_late=*/false);
    } else if (timed_out) {
      PendingSweep sweep = std::move(sweeps_.front());
      sweeps_.pop_front();
      emit_group(std::move(sweep), /*imu_late=*/true);
    } else {
      // The head still waits for its spanning IMU and has not timed out; younger
      // sweeps must wait behind it to preserve ordering.
      break;
    }
  }
}

void Aggregator::emit_group(PendingSweep&& sweep, bool imu_late) {
  MeasureGroup group;
  group.t_begin = sweep.t_begin;
  group.t_end = sweep.t_end;
  group.scan = std::move(sweep.scan);

  // IMU set: every sample with stamp in (lower, t_end], plus the single sample
  // straddling t_begin (the last sample at or before t_begin) so preintegration can
  // start exactly at the sweep boundary. Stamp jitter can begin this sweep before the
  // previous one ended; clamping the lower edge to t_begin keeps the shared window in
  // both groups so each sweep's IMU spans its own interval.
  const Timestamp lower =
      have_prev_t_end_ ? std::min(prev_t_end_, sweep.t_begin) : sweep.t_begin;

  // Find the straddling sample: the newest IMU at or before t_begin.
  std::optional<ImuSample> straddler;
  for (const ImuSample& s : imu_) {
    if (s.stamp <= sweep.t_begin) {
      straddler = s;
    } else {
      break;
    }
  }
  if (straddler && straddler->stamp <= lower) {
    // The straddler sits at or before the exclusive lower edge; include it once,
    // ahead of the in-interval set, rather than letting the interval test drop it.
    group.imu.push_back(*straddler);
  }

  for (const ImuSample& s : imu_) {
    if (s.stamp > lower && s.stamp <= sweep.t_end) {
      group.imu.push_back(s);
    }
  }

  // Drain consumed IMU, but retain everything from the newest sample at or before
  // t_begin minus the reorder window: the next sweep may begin anywhere past that
  // bound, and its straddling sample must survive this group's emission.
  const Timestamp retain_bound = sweep.t_begin - reorder_ns_;
  Timestamp keep_from = retain_bound;  // nothing at-or-before the bound: keep all
  for (const ImuSample& s : imu_) {
    if (s.stamp <= retain_bound) {
      keep_from = s.stamp;  // newest at-or-before the bound, found by the sorted scan
    } else {
      break;
    }
  }
  while (!imu_.empty() && imu_.front().stamp < keep_from) {
    imu_.pop_front();
  }

  // Image: the one whose mid-exposure falls in [t_begin, t_end]. Drop images strictly
  // older than the interval; carry the matching one out.
  while (!images_.empty() && images_.front().stamp < sweep.t_begin) {
    images_.pop_front();
  }
  if (!images_.empty() && images_.front().stamp >= sweep.t_begin &&
      images_.front().stamp <= sweep.t_end) {
    group.image = std::move(images_.front());
    images_.pop_front();
  }

  // GNSS: every fix in [t_begin, t_end]; discard older fixes.
  while (!gnss_.empty() && gnss_.front().stamp < sweep.t_begin) {
    gnss_.pop_front();
  }
  while (!gnss_.empty() && gnss_.front().stamp <= sweep.t_end) {
    group.gnss.push_back(std::move(gnss_.front()));
    gnss_.pop_front();
  }

  prev_t_end_ = sweep.t_end;
  have_prev_t_end_ = true;

  if (imu_late) {
    flag(group.scan.sensor_id, Modality::Imu, HealthCode::ImuLate, sweep.t_end);
  }

  if (telemetry_ && telemetry_->enabled("sensors/imu_in_group")) {
    telemetry_->scalar("sensors/imu_in_group",
                       static_cast<double>(group.imu.size()), sweep.t_end);
  }

  if (sink_) sink_(std::move(group));
}

}  // namespace meridian
