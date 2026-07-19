#include "meridian/ros/pipeline_diagnostics.hpp"

#include <rmw/qos_profiles.h>

#include <cstdint>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace meridian::ros {
namespace {

using diagnostic_msgs::msg::DiagnosticStatus;
using diagnostic_msgs::msg::KeyValue;

void addValue(DiagnosticStatus* status, std::string key, std::string value) {
  KeyValue field;
  field.key = std::move(key);
  field.value = std::move(value);
  status->values.push_back(std::move(field));
}

void addUnsigned(DiagnosticStatus* status, std::string key, std::uint64_t value) {
  addValue(status, std::move(key), std::to_string(value));
}

void addSize(DiagnosticStatus* status, std::string key, std::size_t value) {
  addValue(status, std::move(key), std::to_string(value));
}

void addDuration(DiagnosticStatus* status, std::string key, core::Duration value) {
  addValue(status, std::move(key), std::to_string(value.nanoseconds));
}

void addQueue(DiagnosticStatus* status, std::string_view prefix,
              const core::PipelineQueueSnapshot& queue) {
  const std::string base(prefix);
  addValue(status, base + "name", std::string(queue.name.view()));
  addSize(status, base + "count", queue.count);
  addSize(status, base + "bytes", queue.bytes);
  addValue(status, base + "count_capacity_available", queue.count_capacity ? "true" : "false");
  if (queue.count_capacity) {
    addSize(status, base + "count_capacity", *queue.count_capacity);
  }
  addValue(status, base + "byte_capacity_available", queue.byte_capacity ? "true" : "false");
  if (queue.byte_capacity) {
    addSize(status, base + "byte_capacity", *queue.byte_capacity);
  }
  addValue(status, base + "oldest_age_available", queue.oldest_age ? "true" : "false");
  if (queue.oldest_age) {
    addDuration(status, base + "oldest_age_ns", *queue.oldest_age);
  }
  addUnsigned(status, base + "accepted", queue.accepted);
  addUnsigned(status, base + "rejected", queue.rejected);
  addUnsigned(status, base + "dropped_oldest", queue.dropped_oldest);
  addUnsigned(status, base + "dropped_newest", queue.dropped_newest);
  addUnsigned(status, base + "skipped_stale", queue.skipped_stale);
  addUnsigned(status, base + "skipped_policy", queue.skipped_policy);
}

void addDurationStatistics(DiagnosticStatus* status, std::string_view prefix,
                           const core::DurationStatistics& statistics) {
  const std::string base(prefix);
  addUnsigned(status, base + "total_samples", statistics.total_samples);
  addSize(status, base + "window_samples", statistics.window_samples);
  addSize(status, base + "window_capacity", statistics.window_capacity);
  if (statistics.minimum) {
    addDuration(status, base + "min_ns", *statistics.minimum);
  }
  if (statistics.p50) {
    addDuration(status, base + "p50_ns", *statistics.p50);
  }
  if (statistics.p95) {
    addDuration(status, base + "p95_ns", *statistics.p95);
  }
  if (statistics.p99) {
    addDuration(status, base + "p99_ns", *statistics.p99);
  }
  if (statistics.maximum) {
    addDuration(status, base + "max_ns", *statistics.maximum);
  }
  if (statistics.mean_nanoseconds) {
    addValue(status, base + "mean_ns", std::to_string(*statistics.mean_nanoseconds));
  }
}

[[nodiscard]] std::string diagnosticName(const core::PipelineStage& stage) {
  return "meridian/local/pipeline/" + std::string(stage.name.view());
}

[[nodiscard]] std::uint8_t sampleLevel(core::PipelineDisposition disposition) noexcept {
  switch (disposition) {
    case core::PipelineDisposition::Completed:
    case core::PipelineDisposition::Accepted:
      return DiagnosticStatus::OK;
    case core::PipelineDisposition::Rejected:
    case core::PipelineDisposition::Deferred:
    case core::PipelineDisposition::Dropped:
    case core::PipelineDisposition::Skipped:
      return DiagnosticStatus::WARN;
    case core::PipelineDisposition::Failed:
      return DiagnosticStatus::ERROR;
  }
  return DiagnosticStatus::ERROR;
}

[[nodiscard]] std::uint8_t statisticsLevel(const core::PipelineDispositionCounts& counts) noexcept {
  if (counts.failed != 0U) {
    return DiagnosticStatus::ERROR;
  }
  if (counts.rejected != 0U || counts.deferred != 0U || counts.dropped != 0U ||
      counts.skipped != 0U) {
    return DiagnosticStatus::WARN;
  }
  return DiagnosticStatus::OK;
}

[[nodiscard]] DiagnosticStatus baseStatus(const core::PipelineStage& stage) {
  DiagnosticStatus status;
  status.name = diagnosticName(stage);
  status.hardware_id = "cpu";
  addUnsigned(&status, "stage_id", stage.id.value());
  addValue(&status, "stage_name", std::string(stage.name.view()));
  return status;
}

}  // namespace

rclcpp::QoS pipelineTimingQos(std::size_t history_depth) {
  if (history_depth == 0U) {
    throw std::invalid_argument("pipeline timing QoS history depth must be nonzero");
  }
  return rclcpp::QoS(rclcpp::KeepLast(history_depth)).reliable().durability_volatile();
}

diagnostic_msgs::msg::DiagnosticArray toDiagnosticArray(
    const core::PipelineTimingSample& sample,
    const builtin_interfaces::msg::Time& publication_stamp) {
  diagnostic_msgs::msg::DiagnosticArray output;
  output.header.stamp = publication_stamp;
  DiagnosticStatus status = baseStatus(sample.stage);
  status.level = sample.valid() ? sampleLevel(sample.disposition) : DiagnosticStatus::ERROR;
  status.message = sample.valid() ? std::string(core::pipelineDispositionName(sample.disposition))
                                  : "invalid_pipeline_timing_sample";
  addUnsigned(&status, "schema_version", sample.schema_version);
  addValue(&status, "record_type", "sample");
  addValue(&status, "valid", sample.valid() ? "true" : "false");
  addDuration(&status, "wall_duration_ns", sample.wall_duration);
  addValue(&status, "thread_cpu_available", sample.thread_cpu_duration ? "true" : "false");
  if (sample.thread_cpu_duration) {
    addDuration(&status, "thread_cpu_duration_ns", *sample.thread_cpu_duration);
  }
  addValue(&status, "disposition", std::string(core::pipelineDispositionName(sample.disposition)));
  if (sample.detail) {
    addValue(&status, "disposition_detail", std::string(sample.detail->view()));
  }
  if (sample.work.measurement) {
    addUnsigned(&status, "measurement_id", sample.work.measurement->value());
  }
  if (sample.work.state) {
    addUnsigned(&status, "state_id", sample.work.state->value());
  }
  if (sample.work.local_revision) {
    addUnsigned(&status, "local_graph_revision", sample.work.local_revision->value());
  }
  if (sample.work.global_revision) {
    addUnsigned(&status, "global_graph_revision", sample.work.global_revision->value);
  }
  addValue(&status, "queue_present", sample.queue ? "true" : "false");
  if (sample.queue) {
    addQueue(&status, "queue.", *sample.queue);
  }
  output.status.push_back(std::move(status));
  return output;
}

diagnostic_msgs::msg::DiagnosticArray toDiagnosticArray(
    const core::PipelineTimingStatisticsSnapshot& snapshot,
    const builtin_interfaces::msg::Time& publication_stamp) {
  diagnostic_msgs::msg::DiagnosticArray output;
  output.header.stamp = publication_stamp;
  DiagnosticStatus status = baseStatus(snapshot.stage);
  status.level =
      snapshot.valid() ? statisticsLevel(snapshot.dispositions) : DiagnosticStatus::ERROR;
  status.message = snapshot.valid() ? "rolling_statistics" : "invalid_pipeline_timing_statistics";
  addUnsigned(&status, "schema_version", snapshot.schema_version);
  addValue(&status, "record_type", "statistics");
  addValue(&status, "valid", snapshot.valid() ? "true" : "false");
  addDurationStatistics(&status, "wall.", snapshot.wall);
  addDurationStatistics(&status, "thread_cpu.", snapshot.thread_cpu);
  addUnsigned(&status, "disposition.completed", snapshot.dispositions.completed);
  addUnsigned(&status, "disposition.accepted", snapshot.dispositions.accepted);
  addUnsigned(&status, "disposition.rejected", snapshot.dispositions.rejected);
  addUnsigned(&status, "disposition.deferred", snapshot.dispositions.deferred);
  addUnsigned(&status, "disposition.dropped", snapshot.dispositions.dropped);
  addUnsigned(&status, "disposition.skipped", snapshot.dispositions.skipped);
  addUnsigned(&status, "disposition.failed", snapshot.dispositions.failed);
  addValue(&status, "latest_queue_present", snapshot.latest_queue ? "true" : "false");
  if (snapshot.latest_queue) {
    addQueue(&status, "latest_queue.", *snapshot.latest_queue);
  }
  output.status.push_back(std::move(status));
  return output;
}

PipelineTimingPublisher::PipelineTimingPublisher(rclcpp::Node& node, std::size_t history_depth,
                                                 std::string topic)
    : topic_name_(std::move(topic)), clock_(node.get_clock()) {
  if (topic_name_.empty()) {
    throw std::invalid_argument("pipeline timing publisher topic must be nonempty");
  }
  publisher_ = node.create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      topic_name_, pipelineTimingQos(history_depth));
}

void PipelineTimingPublisher::publish(const core::PipelineTimingSample& sample) {
  const auto stamp = static_cast<builtin_interfaces::msg::Time>(clock_->now());
  publisher_->publish(toDiagnosticArray(sample, stamp));
}

void PipelineTimingPublisher::publish(const core::PipelineTimingStatisticsSnapshot& snapshot) {
  const auto stamp = static_cast<builtin_interfaces::msg::Time>(clock_->now());
  publisher_->publish(toDiagnosticArray(snapshot, stamp));
}

}  // namespace meridian::ros
