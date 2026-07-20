#include "meridian/apps/rerun_debug_sink.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <rerun.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace meridian::apps {
namespace {

template <typename... Callables>
struct Overloaded : Callables... {
  using Callables::operator()...;
};
template <typename... Callables>
Overloaded(Callables...) -> Overloaded<Callables...>;

void requireSuccess(const rerun::Error& error, const std::string& operation) {
  if (error.is_err()) {
    throw std::runtime_error(operation + ": " + error.description);
  }
}

std::int64_t sequence(core::MeasurementId id) noexcept {
  constexpr auto kMaximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  return static_cast<std::int64_t>(std::min(id.value(), kMaximum));
}

}  // namespace

RerunDebugSink::RerunDebugSink(Options options) : options_(std::move(options)) {
  if (options_.output_path.empty()) {
    throw std::invalid_argument("Rerun output path must not be empty");
  }
  if (options_.queue_capacity == 0U) {
    throw std::invalid_argument("Rerun queue capacity must be greater than zero");
  }

  auto recording = std::make_unique<rerun::RecordingStream>("meridian");
  requireSuccess(recording->save(options_.output_path.string()), "open Rerun recording");
  worker_ = std::thread(
      [this, recording = std::move(recording)]() mutable { workerLoop(std::move(recording)); });
}

RerunDebugSink::~RerunDebugSink() {
  shutdown();
}

void RerunDebugSink::record(const core::ImuAcceptedEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::LidarAcceptedEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::LidarPreviewEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::IngressFailureEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

std::uint64_t RerunDebugSink::droppedEvents() const noexcept {
  return dropped_events_.load(std::memory_order_relaxed);
}

std::uint64_t RerunDebugSink::logErrors() const noexcept {
  return log_errors_.load(std::memory_order_relaxed);
}

void RerunDebugSink::enqueue(Event event) noexcept {
  try {
    std::lock_guard lock(mutex_);
    if (closing_ || queue_.size() >= options_.queue_capacity) {
      dropped_events_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }
    queue_.push_back(std::move(event));
    queue_condition_.notify_one();
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::shutdown() noexcept {
  {
    std::lock_guard lock(mutex_);
    closing_ = true;
  }
  queue_condition_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void RerunDebugSink::workerLoop(std::unique_ptr<rerun::RecordingStream> recording) noexcept {
  auto noteError = [this]() noexcept { log_errors_.fetch_add(1U, std::memory_order_relaxed); };
  auto note = [this](const rerun::Error& error) noexcept {
    if (error.is_err()) {
      log_errors_.fetch_add(1U, std::memory_order_relaxed);
    }
  };

  try {
    auto setMeasurementContext = [&recording](core::TimeNs time, core::MeasurementId id) {
      recording->set_time_timestamp_nanos_since_epoch("sensor_time", time.count());
      recording->set_time_sequence("measurement_id", sequence(id));
    };

    while (true) {
      std::optional<Event> event;
      {
        std::unique_lock lock(mutex_);
        queue_condition_.wait(lock, [this] { return closing_ || !queue_.empty(); });
        if (queue_.empty()) {
          if (closing_) {
            break;
          }
          continue;
        }
        event.emplace(std::move(queue_.front()));
        queue_.pop_front();
      }

      try {
        std::visit(
            Overloaded{
                [&](const core::ImuAcceptedEvent& item) {
                  setMeasurementContext(item.measurement_time, item.measurement_id);
                  note(recording->try_log(
                      "/sensors/imu/sample",
                      rerun::Scalars({item.specific_force_m_s2.x, item.specific_force_m_s2.y,
                                      item.specific_force_m_s2.z, item.angular_velocity_rad_s.x,
                                      item.angular_velocity_rad_s.y, item.angular_velocity_rad_s.z,
                                      static_cast<double>(item.arrival_steady_ns),
                                      static_cast<double>(item.conversion_duration_ns)})));
                },
                [&](const core::LidarAcceptedEvent& item) {
                  setMeasurementContext(item.measurement_time, item.measurement_id);
                  note(recording->try_log(
                      "/sensors/lidar/metadata",
                      rerun::Scalars({static_cast<double>(item.source_points),
                                      static_cast<double>(item.accepted_points),
                                      static_cast<double>(item.nonfinite_xyz_points),
                                      static_cast<double>(item.zero_xyz_points),
                                      static_cast<double>(item.flattened_time_regressions),
                                      static_cast<double>(item.arrival_steady_ns),
                                      static_cast<double>(item.conversion_duration_ns),
                                      static_cast<double>(item.decode_queue_depth),
                                      static_cast<double>(item.acquisition_duration_ns)})));
                },
                [&](const core::LidarPreviewEvent& item) {
                  setMeasurementContext(item.measurement_time, item.measurement_id);
                  std::vector<std::array<float, 3>> positions;
                  if (item.points) {
                    positions.reserve(item.points->size());
                    for (const auto& point : *item.points) {
                      positions.push_back({point.x, point.y, point.z});
                    }
                  }
                  note(recording->try_log("/sensors/lidar/preview",
                                          rerun::Points3D(std::move(positions)).with_radii(0.02F)));
                },
                [&](const core::IngressFailureEvent& item) {
                  recording->set_time_sequence("measurement_id", sequence(item.measurement_id));
                  if (item.measurement_time.has_value()) {
                    recording->set_time_timestamp_nanos_since_epoch("sensor_time",
                                                                    item.measurement_time->count());
                  } else {
                    recording->disable_timeline("sensor_time");
                  }
                  note(recording->try_log("/runtime/ingress_failure",
                                          rerun::TextLog(item.sensor_id + ":" + item.error_code +
                                                         ":" + item.field + ":" + item.detail)));
                },
            },
            *event);
      } catch (...) {
        noteError();
      }
    }
  } catch (...) {
    noteError();
  }

  try {
    recording->disable_timeline("sensor_time");
    recording->disable_timeline("measurement_id");
    note(recording->try_log(
        "/run/debug_events_dropped",
        rerun::Scalars(static_cast<double>(dropped_events_.load(std::memory_order_relaxed)))));
    // Log this last so it includes any error produced while writing the drop
    // counter. The process exit status remains authoritative if this write or
    // the following flush itself fails.
    note(recording->try_log(
        "/run/debug_log_errors",
        rerun::Scalars(static_cast<double>(log_errors_.load(std::memory_order_relaxed)))));
  } catch (...) {
    noteError();
  }

  try {
    note(recording->flush_blocking(30.0F));
  } catch (...) {
    noteError();
  }
}

}  // namespace meridian::apps
