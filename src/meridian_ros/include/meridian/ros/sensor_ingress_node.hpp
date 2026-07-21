#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <string_view>
#include <thread>

#include "meridian/core/debug_sink.hpp"
#include "meridian/core/time.hpp"
#include "meridian/ros/conversion.hpp"

namespace meridian::ros {

struct ObservationCallbacks final {
  std::function<void(core::ImuSample)> imu;
  std::function<void(core::LidarSweep)> lidar;
};

class SensorIngressNode final : public rclcpp::Node {
public:
  explicit SensorIngressNode(const rclcpp::NodeOptions& options, core::DebugSink& debug_sink,
                             ObservationCallbacks observation_callbacks = {});
  ~SensorIngressNode() override;

  SensorIngressNode(const SensorIngressNode&) = delete;
  SensorIngressNode& operator=(const SensorIngressNode&) = delete;

  // Installs the observation handoff exactly once. This must be called before
  // the node is added to an executor so callbacks cannot race the assignment.
  void setObservationCallbacks(ObservationCallbacks observation_callbacks);

  // Stops admission and drains the LiDAR decoder. Safe to call more than once.
  void stop() noexcept;

private:
  struct QueuedCloud final {
    sensor_msgs::msg::PointCloud2::ConstSharedPtr message;
    core::MeasurementId measurement_id;
    std::int64_t arrival_steady_ns{};
  };

  void imuCallback(sensor_msgs::msg::Imu::ConstSharedPtr message) noexcept;
  void lidarCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr message) noexcept;
  void lidarDecodeLoop(std::stop_token stop_token) noexcept;
  void processCloud(QueuedCloud queued) noexcept;
  void recordPreview(const core::LidarSweep& sweep);
  void recordFailure(std::string_view sensor_id, core::MeasurementId measurement_id,
                     std::optional<core::TimeNs> measurement_time, std::int64_t arrival_steady_ns,
                     std::int64_t conversion_duration_ns, std::string_view error_code,
                     std::string_view field, std::string_view detail) noexcept;

  [[nodiscard]] static std::int64_t steadyNowNs() noexcept;

  core::DebugSink& debug_sink_;
  ObservationCallbacks observation_callbacks_;
  bool observation_callbacks_configured_{};
  ImuConversionConfig imu_config_;
  LidarConversionConfig lidar_config_;

  std::string imu_topic_;
  std::string lidar_topic_;
  std::size_t lidar_decode_queue_capacity_{};
  std::size_t lidar_max_points_{};
  std::int64_t lidar_max_scan_duration_ns_{};
  std::int64_t preview_period_ns_{};
  std::size_t preview_max_points_{};

  rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
  rclcpp::CallbackGroup::SharedPtr lidar_callback_group_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscription_;

  mutable std::mutex lidar_queue_mutex_;
  std::condition_variable lidar_queue_condition_;
  std::deque<QueuedCloud> lidar_queue_;
  std::jthread lidar_decode_thread_;
  std::atomic<bool> accepting_{true};
  std::atomic<bool> stopped_{false};

  std::atomic<std::uint64_t> next_measurement_id_{1U};
  std::optional<core::TimeNs> next_preview_time_;
};

}  // namespace meridian::ros
