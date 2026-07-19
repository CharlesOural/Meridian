#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <cstddef>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/qos.hpp>
#include <string>
#include <string_view>

#include "meridian/core/pipeline_observability.hpp"

namespace meridian::ros {

inline constexpr std::string_view kLocalPipelineTimingTopic = "/meridian/local/pipeline_timing";
inline constexpr std::size_t kDefaultPipelineTimingHistoryDepth = 32U;

[[nodiscard]] rclcpp::QoS pipelineTimingQos(
    std::size_t history_depth = kDefaultPipelineTimingHistoryDepth);

// Deterministic, side-effect-free conversions. DiagnosticArray is the
// machine-readable ROS bridge only; the typed core record remains canonical.
[[nodiscard]] diagnostic_msgs::msg::DiagnosticArray toDiagnosticArray(
    const core::PipelineTimingSample& sample,
    const builtin_interfaces::msg::Time& publication_stamp);
[[nodiscard]] diagnostic_msgs::msg::DiagnosticArray toDiagnosticArray(
    const core::PipelineTimingStatisticsSnapshot& snapshot,
    const builtin_interfaces::msg::Time& publication_stamp);

// Thin reusable adapter: it owns ROS QoS/time/publication and no estimator or
// pipeline policy. A composition root decides which stages to instrument.
class PipelineTimingPublisher {
public:
  explicit PipelineTimingPublisher(rclcpp::Node& node,
                                   std::size_t history_depth = kDefaultPipelineTimingHistoryDepth,
                                   std::string topic = std::string(kLocalPipelineTimingTopic));

  void publish(const core::PipelineTimingSample& sample);
  void publish(const core::PipelineTimingStatisticsSnapshot& snapshot);

  [[nodiscard]] const std::string& topicName() const noexcept { return topic_name_; }

private:
  std::string topic_name_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr publisher_;
};

}  // namespace meridian::ros
