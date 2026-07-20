#include "meridian/ros/sensor_ingress_node.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <sensor_msgs/msg/point_field.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace meridian::ros {
namespace {

using sensor_msgs::msg::PointField;

std::string requiredStringParameter(rclcpp::Node& node, const std::string& name) {
  std::string value = node.declare_parameter<std::string>(name, "");
  if (value.empty()) {
    throw std::invalid_argument(name + " must be configured and non-empty");
  }
  return value;
}

std::size_t positiveSizeParameter(rclcpp::Node& node, const std::string& name,
                                  std::int64_t default_value) {
  const std::int64_t value = node.declare_parameter<std::int64_t>(name, default_value);
  if (value <= 0 || static_cast<std::uint64_t>(value) >
                        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument(name + " must be a positive size");
  }
  return static_cast<std::size_t>(value);
}

std::int64_t positiveInt64Parameter(rclcpp::Node& node, const std::string& name,
                                    std::int64_t default_value) {
  const std::int64_t value = node.declare_parameter<std::int64_t>(name, default_value);
  if (value <= 0) {
    throw std::invalid_argument(name + " must be positive");
  }
  return value;
}

std::string normalized(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return value;
}

std::uint8_t parsePointFieldDatatype(const std::string& parameter_name, std::string value) {
  value = normalized(std::move(value));
  if (value == "int8") {
    return PointField::INT8;
  }
  if (value == "uint8") {
    return PointField::UINT8;
  }
  if (value == "int16") {
    return PointField::INT16;
  }
  if (value == "uint16") {
    return PointField::UINT16;
  }
  if (value == "int32") {
    return PointField::INT32;
  }
  if (value == "uint32") {
    return PointField::UINT32;
  }
  if (value == "float32") {
    return PointField::FLOAT32;
  }
  if (value == "float64") {
    return PointField::FLOAT64;
  }
  throw std::invalid_argument(parameter_name +
                              " must be one of int8, uint8, int16, uint16, int32, uint32, "
                              "float32, or float64");
}

std::uint8_t requiredDatatypeParameter(rclcpp::Node& node, const std::string& name) {
  return parsePointFieldDatatype(name, requiredStringParameter(node, name));
}

std::uint8_t optionalFieldDatatypeParameter(rclcpp::Node& node, const std::string& name,
                                            bool field_is_enabled) {
  const std::string value = node.declare_parameter<std::string>(name, "");
  if (!field_is_enabled) {
    if (!value.empty()) {
      return parsePointFieldDatatype(name, value);
    }
    return PointField::FLOAT32;
  }
  if (value.empty()) {
    throw std::invalid_argument(name + " must be configured when its field name is non-empty");
  }
  return parsePointFieldDatatype(name, value);
}

PointTimeReference pointTimeReferenceParameter(rclcpp::Node& node) {
  const std::string value =
      normalized(node.declare_parameter<std::string>("lidar_time_reference", "offset_from_header"));
  if (value == "offset_from_header") {
    return PointTimeReference::kOffsetFromHeader;
  }
  if (value == "absolute") {
    return PointTimeReference::kAbsolute;
  }
  throw std::invalid_argument("lidar_time_reference must be 'offset_from_header' or 'absolute'");
}

ImuConversionConfig makeImuConfig(rclcpp::Node& node) {
  return ImuConversionConfig{
      .sensor_id = core::SensorId(requiredStringParameter(node, "imu_sensor_id")),
      .calibration_id = core::CalibrationId(requiredStringParameter(node, "imu_calibration_id")),
      .expected_frame_id = requiredStringParameter(node, "imu_frame"),
  };
}

LidarConversionConfig makeLidarConfig(rclcpp::Node& node) {
  const std::string x_field = requiredStringParameter(node, "lidar_x_field");
  const std::string y_field = requiredStringParameter(node, "lidar_y_field");
  const std::string z_field = requiredStringParameter(node, "lidar_z_field");
  const std::string time_field = node.declare_parameter<std::string>("lidar_time_field", "");
  const std::string intensity_field =
      node.declare_parameter<std::string>("lidar_intensity_field", "");
  const std::string ring_field = node.declare_parameter<std::string>("lidar_ring_field", "");
  const double time_scale = node.declare_parameter<double>("lidar_time_scale_to_nanoseconds", 1.0);
  if (!time_field.empty() && (!std::isfinite(time_scale) || time_scale <= 0.0)) {
    throw std::invalid_argument(
        "lidar_time_scale_to_nanoseconds must be finite and positive when point time is enabled");
  }

  return LidarConversionConfig{
      .sensor_id = core::SensorId(requiredStringParameter(node, "lidar_sensor_id")),
      .calibration_id = core::CalibrationId(requiredStringParameter(node, "lidar_calibration_id")),
      .expected_frame_id = requiredStringParameter(node, "lidar_frame"),
      .x_field = x_field,
      .y_field = y_field,
      .z_field = z_field,
      .time_field = time_field,
      .intensity_field = intensity_field,
      .ring_field = ring_field,
      .x_datatype = requiredDatatypeParameter(node, "lidar_x_datatype"),
      .y_datatype = requiredDatatypeParameter(node, "lidar_y_datatype"),
      .z_datatype = requiredDatatypeParameter(node, "lidar_z_datatype"),
      .time_datatype =
          optionalFieldDatatypeParameter(node, "lidar_time_datatype", !time_field.empty()),
      .intensity_datatype = optionalFieldDatatypeParameter(node, "lidar_intensity_datatype",
                                                           !intensity_field.empty()),
      .ring_datatype =
          optionalFieldDatatypeParameter(node, "lidar_ring_datatype", !ring_field.empty()),
      .time_scale_to_nanoseconds = time_scale,
      .time_reference = pointTimeReferenceParameter(node),
  };
}

std::optional<core::TimeNs> rosMessageTime(const builtin_interfaces::msg::Time& stamp) noexcept {
  return core::TimeNs::fromSecNanosec(stamp.sec, stamp.nanosec);
}

}  // namespace

SensorIngressNode::SensorIngressNode(const rclcpp::NodeOptions& options,
                                     core::DebugSink& debug_sink,
                                     ObservationCallbacks observation_callbacks)
    : rclcpp::Node("meridian_ingress", options),
      debug_sink_(debug_sink),
      observation_callbacks_(std::move(observation_callbacks)),
      imu_config_(makeImuConfig(*this)),
      lidar_config_(makeLidarConfig(*this)),
      imu_topic_(requiredStringParameter(*this, "imu_topic")),
      lidar_topic_(requiredStringParameter(*this, "lidar_topic")),
      lidar_decode_queue_capacity_(positiveSizeParameter(*this, "lidar_decode_queue_capacity", 4)),
      lidar_max_points_(positiveSizeParameter(*this, "lidar_max_points", 1'000'000)),
      lidar_max_scan_duration_ns_(
          positiveInt64Parameter(*this, "lidar_max_scan_duration_ns", 1'000'000'000)),
      preview_max_points_(positiveSizeParameter(*this, "preview_max_points", 4096)) {
  const double preview_rate_hz = declare_parameter<double>("preview_rate_hz", 1.0);
  if (!std::isfinite(preview_rate_hz) || preview_rate_hz <= 0.0) {
    throw std::invalid_argument("preview_rate_hz must be finite and positive");
  }
  const double preview_period_ns = 1.0e9 / preview_rate_hz;
  if (!std::isfinite(preview_period_ns) || preview_period_ns < 1.0 ||
      preview_period_ns > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("preview_rate_hz is outside nanosecond scheduling range");
  }
  preview_period_ns_ = static_cast<std::int64_t>(std::llround(preview_period_ns));

  imu_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  lidar_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions imu_options;
  imu_options.callback_group = imu_callback_group_;
  rclcpp::SubscriptionOptions lidar_options;
  lidar_options.callback_group = lidar_callback_group_;

  imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Imu::ConstSharedPtr message) { imuCallback(std::move(message)); },
      imu_options);
  lidar_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
        lidarCallback(std::move(message));
      },
      lidar_options);

  lidar_decode_thread_ = std::jthread([this](std::stop_token token) { lidarDecodeLoop(token); });

  RCLCPP_INFO(get_logger(),
              "Sensor ingress ready: IMU=%s, LiDAR=%s, debug preview=%.3f Hz/%zu points",
              imu_topic_.c_str(), lidar_topic_.c_str(), preview_rate_hz, preview_max_points_);
}

SensorIngressNode::~SensorIngressNode() {
  stop();
}

std::int64_t SensorIngressNode::steadyNowNs() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void SensorIngressNode::imuCallback(sensor_msgs::msg::Imu::ConstSharedPtr message) noexcept {
  if (!message || !accepting_.load(std::memory_order_relaxed)) {
    return;
  }
  const std::int64_t arrival_ns = steadyNowNs();
  const core::MeasurementId measurement_id(
      next_measurement_id_.fetch_add(1U, std::memory_order_relaxed));
  const auto begin = std::chrono::steady_clock::now();
  bool accepted_event_recorded = false;
  try {
    auto converted = convertImu(*message, measurement_id, imu_config_);
    const std::int64_t conversion_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - begin)
                                           .count();
    if (!converted) {
      const auto& error = converted.error();
      recordFailure(imu_config_.sensor_id.value(), measurement_id,
                    rosMessageTime(message->header.stamp), arrival_ns, conversion_ns,
                    toString(error.code), error.field, error.detail);
      return;
    }

    core::ImuSample sample = std::move(converted).value();
    debug_sink_.record(core::ImuAcceptedEvent{
        .measurement_id = measurement_id,
        .measurement_time = sample.header().measurementTime(),
        .angular_velocity_rad_s = sample.angularVelocityRadS(),
        .specific_force_m_s2 = sample.specificForceMS2(),
        .arrival_steady_ns = arrival_ns,
        .conversion_duration_ns = conversion_ns,
    });
    accepted_event_recorded = true;
    if (observation_callbacks_.imu) {
      observation_callbacks_.imu(std::move(sample));
    }
  } catch (const std::exception& error) {
    const std::int64_t duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - begin)
                                         .count();
    recordFailure(imu_config_.sensor_id.value(), measurement_id,
                  rosMessageTime(message->header.stamp), arrival_ns, duration_ns,
                  accepted_event_recorded ? "downstream_callback_exception" : "callback_exception",
                  accepted_event_recorded ? "imu_observation_callback" : "imu_callback",
                  error.what());
    RCLCPP_ERROR(get_logger(), "IMU processing failed after an exception: %s", error.what());
  } catch (...) {
    const std::int64_t duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - begin)
                                         .count();
    recordFailure(imu_config_.sensor_id.value(), measurement_id,
                  rosMessageTime(message->header.stamp), arrival_ns, duration_ns,
                  accepted_event_recorded ? "downstream_callback_exception" : "callback_exception",
                  accepted_event_recorded ? "imu_observation_callback" : "imu_callback",
                  "unknown exception");
    RCLCPP_ERROR(get_logger(), "IMU processing failed after an unknown exception");
  }
}

void SensorIngressNode::lidarCallback(
    sensor_msgs::msg::PointCloud2::ConstSharedPtr message) noexcept {
  if (!message || !accepting_.load(std::memory_order_relaxed)) {
    return;
  }
  const std::int64_t arrival_ns = steadyNowNs();
  const core::MeasurementId measurement_id(
      next_measurement_id_.fetch_add(1U, std::memory_order_relaxed));
  try {
    const std::uint64_t source_points =
        static_cast<std::uint64_t>(message->width) * message->height;
    if (source_points > static_cast<std::uint64_t>(lidar_max_points_)) {
      recordFailure(lidar_config_.sensor_id.value(), measurement_id,
                    rosMessageTime(message->header.stamp), arrival_ns, 0, "point_count_limit",
                    "width/height", "width * height exceeds lidar_max_points");
      return;
    }

    bool queue_rejected = false;
    {
      std::lock_guard lock(lidar_queue_mutex_);
      if (!accepting_.load(std::memory_order_relaxed)) {
        return;
      }
      if (lidar_queue_.size() >= lidar_decode_queue_capacity_) {
        queue_rejected = true;
      } else {
        lidar_queue_.push_back(QueuedCloud{message, measurement_id, arrival_ns});
      }
    }
    if (queue_rejected) {
      recordFailure(lidar_config_.sensor_id.value(), measurement_id,
                    rosMessageTime(message->header.stamp), arrival_ns, 0, "decode_queue_full",
                    "lidar_decode_queue", "bounded LiDAR decode queue rejected the newest cloud");
      return;
    }
    lidar_queue_condition_.notify_one();
  } catch (const std::exception& error) {
    recordFailure(lidar_config_.sensor_id.value(), measurement_id,
                  rosMessageTime(message->header.stamp), arrival_ns, 0, "callback_exception",
                  "lidar_callback", error.what());
    RCLCPP_ERROR(get_logger(), "LiDAR callback failed after an exception: %s", error.what());
  } catch (...) {
    recordFailure(lidar_config_.sensor_id.value(), measurement_id,
                  rosMessageTime(message->header.stamp), arrival_ns, 0, "callback_exception",
                  "lidar_callback", "unknown exception");
    RCLCPP_ERROR(get_logger(), "LiDAR callback failed after an unknown exception");
  }
}

void SensorIngressNode::lidarDecodeLoop(std::stop_token stop_token) noexcept {
  while (true) {
    std::optional<QueuedCloud> queued;
    try {
      {
        std::unique_lock lock(lidar_queue_mutex_);
        lidar_queue_condition_.wait(lock, [this, &stop_token] {
          return !lidar_queue_.empty() || stop_token.stop_requested() ||
                 !accepting_.load(std::memory_order_relaxed);
        });
        if (lidar_queue_.empty()) {
          if (stop_token.stop_requested() || !accepting_.load(std::memory_order_relaxed)) {
            break;
          }
          continue;
        }
        queued.emplace(std::move(lidar_queue_.front()));
        lidar_queue_.pop_front();
      }
    } catch (const std::exception& error) {
      RCLCPP_ERROR(get_logger(), "LiDAR decoder stopped after a queue exception: %s", error.what());
      return;
    } catch (...) {
      RCLCPP_ERROR(get_logger(), "LiDAR decoder stopped after an unknown queue exception");
      return;
    }
    processCloud(std::move(*queued));
  }
}

void SensorIngressNode::processCloud(QueuedCloud queued) noexcept {
  const auto begin = std::chrono::steady_clock::now();
  bool accepted_event_recorded = false;
  try {
    auto converted = convertPointCloud2(*queued.message, queued.measurement_id, lidar_config_);
    const std::int64_t conversion_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - begin)
                                           .count();
    if (!converted) {
      const auto& error = converted.error();
      recordFailure(lidar_config_.sensor_id.value(), queued.measurement_id,
                    rosMessageTime(queued.message->header.stamp), queued.arrival_steady_ns,
                    conversion_ns, toString(error.code), error.field, error.detail);
      return;
    }

    core::LidarSweep sweep = std::move(converted).value();
    const auto scan_duration =
        core::TimeNs::checkedDifference(sweep.acquisitionEnd(), sweep.acquisitionBegin());
    if (!scan_duration.has_value() || *scan_duration > lidar_max_scan_duration_ns_) {
      recordFailure(lidar_config_.sensor_id.value(), queued.measurement_id,
                    sweep.header().measurementTime(), queued.arrival_steady_ns, conversion_ns,
                    "invalid_scan_duration", lidar_config_.time_field,
                    "point-time span exceeds lidar_max_scan_duration_ns");
      return;
    }

    std::size_t queue_depth = 0U;
    {
      std::lock_guard lock(lidar_queue_mutex_);
      queue_depth = lidar_queue_.size();
    }
    const auto& stats = converted.stats();
    debug_sink_.record(core::LidarAcceptedEvent{
        .measurement_id = queued.measurement_id,
        .measurement_time = sweep.header().measurementTime(),
        .acquisition_duration_ns = *scan_duration,
        .source_points = stats.source_points,
        .accepted_points = stats.accepted_points,
        .nonfinite_xyz_points = stats.nonfinite_xyz_points,
        .zero_xyz_points = stats.zero_xyz_points,
        .flattened_time_regressions = stats.flattened_time_regressions,
        .arrival_steady_ns = queued.arrival_steady_ns,
        .conversion_duration_ns = conversion_ns,
        .decode_queue_depth = queue_depth,
    });
    accepted_event_recorded = true;

    if (debug_sink_.wantsLidarPreview()) {
      const bool preview_due = !next_preview_time_.has_value() ||
                               sweep.header().measurementTime() >= *next_preview_time_;
      if (preview_due) {
        try {
          recordPreview(sweep);
          core::TimeNs deadline = next_preview_time_.value_or(sweep.header().measurementTime());
          do {
            const auto advanced = core::TimeNs::checkedAdd(deadline, preview_period_ns_);
            if (!advanced.has_value()) {
              next_preview_time_.reset();
              break;
            }
            deadline = *advanced;
            next_preview_time_ = deadline;
          } while (deadline <= sweep.header().measurementTime());
        } catch (const std::exception& error) {
          recordFailure(lidar_config_.sensor_id.value(), queued.measurement_id,
                        sweep.header().measurementTime(), queued.arrival_steady_ns, conversion_ns,
                        "debug_preview_exception", "lidar_preview", error.what());
        } catch (...) {
          recordFailure(lidar_config_.sensor_id.value(), queued.measurement_id,
                        sweep.header().measurementTime(), queued.arrival_steady_ns, conversion_ns,
                        "debug_preview_exception", "lidar_preview", "unknown exception");
        }
      }
    }

    if (observation_callbacks_.lidar) {
      observation_callbacks_.lidar(std::move(sweep));
    }
  } catch (const std::exception& error) {
    const std::int64_t duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - begin)
                                         .count();
    recordFailure(
        lidar_config_.sensor_id.value(), queued.measurement_id,
        rosMessageTime(queued.message->header.stamp), queued.arrival_steady_ns, duration_ns,
        accepted_event_recorded ? "downstream_callback_exception" : "decoder_exception",
        accepted_event_recorded ? "lidar_observation_callback" : "lidar_decoder", error.what());
    RCLCPP_ERROR(get_logger(), "LiDAR processing failed after an exception: %s", error.what());
  } catch (...) {
    const std::int64_t duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - begin)
                                         .count();
    recordFailure(lidar_config_.sensor_id.value(), queued.measurement_id,
                  rosMessageTime(queued.message->header.stamp), queued.arrival_steady_ns,
                  duration_ns,
                  accepted_event_recorded ? "downstream_callback_exception" : "decoder_exception",
                  accepted_event_recorded ? "lidar_observation_callback" : "lidar_decoder",
                  "unknown exception");
    RCLCPP_ERROR(get_logger(), "LiDAR processing failed after an unknown exception");
  }
}

void SensorIngressNode::recordPreview(const core::LidarSweep& sweep) {
  std::size_t nonzero_points = 0U;
  for (const auto& point : sweep.points()) {
    if (point.x != 0.0F || point.y != 0.0F || point.z != 0.0F) {
      ++nonzero_points;
    }
  }
  const std::size_t stride =
      std::max<std::size_t>(1U, (nonzero_points + preview_max_points_ - 1U) / preview_max_points_);
  auto preview_points = std::make_shared<std::vector<core::LidarPreviewPoint>>();
  preview_points->reserve(std::min(nonzero_points, preview_max_points_));
  std::size_t nonzero_index = 0U;
  for (const auto& point : sweep.points()) {
    if (point.x == 0.0F && point.y == 0.0F && point.z == 0.0F) {
      continue;
    }
    if (nonzero_index % stride == 0U && preview_points->size() < preview_max_points_) {
      preview_points->push_back(core::LidarPreviewPoint{
          .x = point.x,
          .y = point.y,
          .z = point.z,
      });
    }
    ++nonzero_index;
  }

  debug_sink_.record(core::LidarPreviewEvent{
      .measurement_id = sweep.header().measurementId(),
      .measurement_time = sweep.header().measurementTime(),
      .points = std::move(preview_points),
  });
}

void SensorIngressNode::recordFailure(std::string_view sensor_id,
                                      core::MeasurementId measurement_id,
                                      std::optional<core::TimeNs> measurement_time,
                                      std::int64_t arrival_steady_ns,
                                      std::int64_t conversion_duration_ns,
                                      std::string_view error_code, std::string_view field,
                                      std::string_view detail) noexcept {
  try {
    debug_sink_.record(core::IngressFailureEvent{
        .sensor_id = std::string(sensor_id),
        .measurement_id = measurement_id,
        .measurement_time = measurement_time,
        .arrival_steady_ns = arrival_steady_ns,
        .conversion_duration_ns = conversion_duration_ns,
        .error_code = std::string(error_code),
        .field = std::string(field),
        .detail = std::string(detail),
    });
  } catch (...) {
  }
}

void SensorIngressNode::stop() noexcept {
  bool expected = false;
  if (!stopped_.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
    return;
  }
  {
    std::lock_guard lock(lidar_queue_mutex_);
    accepting_.store(false, std::memory_order_relaxed);
  }
  lidar_queue_condition_.notify_all();
  if (lidar_decode_thread_.joinable()) {
    lidar_decode_thread_.request_stop();
    lidar_decode_thread_.join();
  }
  RCLCPP_INFO(get_logger(), "Sensor ingress stopped after draining the LiDAR decode queue");
}

}  // namespace meridian::ros
