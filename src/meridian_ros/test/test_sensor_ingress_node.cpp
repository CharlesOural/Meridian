#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "meridian/core/debug_sink.hpp"
#include "meridian/ros/sensor_ingress_node.hpp"

namespace meridian::ros {
namespace {

using namespace std::chrono_literals;
using sensor_msgs::msg::PointField;

class CapturingDebugSink final : public core::DebugSink {
public:
  explicit CapturingDebugSink(bool wants_preview = false) noexcept
      : wants_preview_(wants_preview) {}

  [[nodiscard]] bool wantsLidarPreview() const noexcept override { return wants_preview_; }
  void record(const core::ImuAcceptedEvent&) noexcept override {
    imu_events_.fetch_add(1U, std::memory_order_relaxed);
  }
  void record(const core::LidarAcceptedEvent&) noexcept override {
    lidar_events_.fetch_add(1U, std::memory_order_relaxed);
  }
  void record(const core::LidarPreviewEvent& event) noexcept override {
    if (event.points && !event.points->empty()) {
      preview_events_.fetch_add(1U, std::memory_order_relaxed);
    }
  }
  void record(const core::IngressFailureEvent& event) noexcept override {
    {
      std::lock_guard lock(error_mutex_);
      last_error_code_ = event.error_code;
    }
    failure_events_.fetch_add(1U, std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t droppedEvents() const noexcept override { return 0U; }
  [[nodiscard]] std::uint64_t logErrors() const noexcept override { return 0U; }

  [[nodiscard]] std::uint64_t imuEvents() const noexcept {
    return imu_events_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t lidarEvents() const noexcept {
    return lidar_events_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t previewEvents() const noexcept {
    return preview_events_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t failureEvents() const noexcept {
    return failure_events_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::string lastErrorCode() const {
    std::lock_guard lock(error_mutex_);
    return last_error_code_;
  }

private:
  bool wants_preview_{};
  std::atomic<std::uint64_t> imu_events_{0U};
  std::atomic<std::uint64_t> lidar_events_{0U};
  std::atomic<std::uint64_t> preview_events_{0U};
  std::atomic<std::uint64_t> failure_events_{0U};
  mutable std::mutex error_mutex_;
  std::string last_error_code_;
};

template <typename Predicate>
bool spinUntil(rclcpp::executors::SingleThreadedExecutor& executor, Predicate predicate,
               std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some();
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(5ms);
  }
  executor.spin_some();
  return predicate();
}

rclcpp::NodeOptions optionsFor(const std::string& suffix) {
  rclcpp::NodeOptions options;
  options.parameter_overrides({
      rclcpp::Parameter("imu_topic", "/meridian_test/imu_" + suffix),
      rclcpp::Parameter("imu_frame", "configured_imu_frame"),
      rclcpp::Parameter("imu_sensor_id", "configured_imu"),
      rclcpp::Parameter("imu_calibration_id", "configured_imu_calibration"),
      rclcpp::Parameter("lidar_topic", "/meridian_test/lidar_" + suffix),
      rclcpp::Parameter("lidar_frame", "configured_lidar_frame"),
      rclcpp::Parameter("lidar_sensor_id", "configured_lidar"),
      rclcpp::Parameter("lidar_calibration_id", "configured_lidar_calibration"),
      rclcpp::Parameter("lidar_x_field", "px"),
      rclcpp::Parameter("lidar_y_field", "py"),
      rclcpp::Parameter("lidar_z_field", "pz"),
      rclcpp::Parameter("lidar_time_field", "dt"),
      rclcpp::Parameter("lidar_intensity_field", "signal"),
      rclcpp::Parameter("lidar_ring_field", "beam"),
      rclcpp::Parameter("lidar_x_datatype", "float32"),
      rclcpp::Parameter("lidar_y_datatype", "float32"),
      rclcpp::Parameter("lidar_z_datatype", "float32"),
      rclcpp::Parameter("lidar_time_datatype", "float32"),
      rclcpp::Parameter("lidar_intensity_datatype", "uint16"),
      rclcpp::Parameter("lidar_ring_datatype", "uint16"),
      rclcpp::Parameter("lidar_time_scale_to_nanoseconds", 1.0e9),
      rclcpp::Parameter("lidar_time_reference", "offset_from_header"),
      rclcpp::Parameter("lidar_decode_queue_capacity", std::int64_t{4}),
      rclcpp::Parameter("lidar_max_points", std::int64_t{100}),
      rclcpp::Parameter("lidar_max_scan_duration_ns", std::int64_t{1'000'000'000}),
      rclcpp::Parameter("preview_rate_hz", 1.0),
      rclcpp::Parameter("preview_max_points", std::int64_t{16}),
  });
  return options;
}

template <typename T>
void writeScalar(std::vector<std::uint8_t>& data, std::size_t offset, T value) {
  std::array<std::uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  if constexpr (std::endian::native == std::endian::big) {
    std::reverse(bytes.begin(), bytes.end());
  }
  std::memcpy(data.data() + offset, bytes.data(), sizeof(T));
}

sensor_msgs::msg::Imu validImu() {
  sensor_msgs::msg::Imu message;
  message.header.stamp.sec = 1;
  message.header.frame_id = "configured_imu_frame";
  message.angular_velocity.x = 1.0;
  message.linear_acceleration.z = 9.8;
  return message;
}

sensor_msgs::msg::PointCloud2 validCloud() {
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp.sec = 2;
  cloud.header.frame_id = "configured_lidar_frame";
  cloud.width = 1U;
  cloud.height = 1U;
  cloud.fields = {
      PointField{}
          .set__name("px")
          .set__offset(0U)
          .set__datatype(PointField::FLOAT32)
          .set__count(1U),
      PointField{}
          .set__name("py")
          .set__offset(4U)
          .set__datatype(PointField::FLOAT32)
          .set__count(1U),
      PointField{}
          .set__name("pz")
          .set__offset(8U)
          .set__datatype(PointField::FLOAT32)
          .set__count(1U),
      PointField{}
          .set__name("dt")
          .set__offset(12U)
          .set__datatype(PointField::FLOAT32)
          .set__count(1U),
      PointField{}
          .set__name("signal")
          .set__offset(16U)
          .set__datatype(PointField::UINT16)
          .set__count(1U),
      PointField{}
          .set__name("beam")
          .set__offset(18U)
          .set__datatype(PointField::UINT16)
          .set__count(1U),
  };
  cloud.point_step = 20U;
  cloud.row_step = 20U;
  cloud.data.resize(20U);
  writeScalar(cloud.data, 0U, 1.0F);
  writeScalar(cloud.data, 4U, 2.0F);
  writeScalar(cloud.data, 8U, 3.0F);
  writeScalar(cloud.data, 12U, 0.01F);
  writeScalar(cloud.data, 16U, std::uint16_t{42});
  writeScalar(cloud.data, 18U, std::uint16_t{7});
  return cloud;
}

class SensorIngressNodeTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    int argc = 0;
    char** argv = nullptr;
    rclcpp::init(argc, argv);
  }

  static void TearDownTestSuite() { rclcpp::shutdown(); }
};

TEST_F(SensorIngressNodeTest, ForwardsOwnedConfiguredObservationsAndRecordsDebugEvents) {
  CapturingDebugSink sink(true);
  std::atomic<std::uint64_t> forwarded_imu{0U};
  std::atomic<std::uint64_t> forwarded_lidar{0U};
  std::atomic<bool> configured_values_preserved{false};
  ObservationCallbacks callbacks{
      .imu =
          [&](core::ImuSample sample) {
            if (sample.header().sensorId().value() == "configured_imu" &&
                sample.angularVelocityRadS().x == 1.0) {
              forwarded_imu.fetch_add(1U, std::memory_order_relaxed);
            }
          },
      .lidar =
          [&](core::LidarSweep sweep) {
            if (sweep.header().sensorId().value() == "configured_lidar" && sweep.size() == 1U &&
                sweep.points()[0].time_offset_ns == 10'000'000 &&
                sweep.points()[0].ring == std::optional<std::uint16_t>(7U)) {
              configured_values_preserved.store(true, std::memory_order_relaxed);
            }
            forwarded_lidar.fetch_add(1U, std::memory_order_relaxed);
          },
  };
  auto ingress =
      std::make_shared<SensorIngressNode>(optionsFor("forward"), sink, std::move(callbacks));
  auto peer = std::make_shared<rclcpp::Node>("meridian_ingress_forward_test_peer");
  auto imu_publisher = peer->create_publisher<sensor_msgs::msg::Imu>("/meridian_test/imu_forward",
                                                                     rclcpp::SensorDataQoS());
  auto lidar_publisher = peer->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/meridian_test/lidar_forward", rclcpp::SensorDataQoS());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(ingress);
  executor.add_node(peer);
  ASSERT_TRUE(spinUntil(executor, [&] {
    return imu_publisher->get_subscription_count() == 1U &&
           lidar_publisher->get_subscription_count() == 1U;
  }));

  imu_publisher->publish(validImu());
  lidar_publisher->publish(validCloud());
  ASSERT_TRUE(spinUntil(executor, [&] {
    return forwarded_imu.load(std::memory_order_relaxed) == 1U &&
           forwarded_lidar.load(std::memory_order_relaxed) == 1U;
  }));
  EXPECT_EQ(sink.imuEvents(), 1U);
  EXPECT_EQ(sink.lidarEvents(), 1U);
  EXPECT_EQ(sink.previewEvents(), 1U);
  EXPECT_EQ(sink.failureEvents(), 0U);
  EXPECT_TRUE(configured_values_preserved.load(std::memory_order_relaxed));

  executor.remove_node(peer);
  executor.remove_node(ingress);
  ingress->stop();
}

TEST_F(SensorIngressNodeTest, OversizedCloudBecomesFailureEventWithoutForwarding) {
  CapturingDebugSink sink;
  std::atomic<std::uint64_t> forwarded_lidar{0U};
  auto options = optionsFor("oversized");
  options.append_parameter_override("lidar_max_points", std::int64_t{1});
  auto ingress = std::make_shared<SensorIngressNode>(
      options, sink,
      ObservationCallbacks{
          .imu = {},
          .lidar =
              [&](core::LidarSweep) { forwarded_lidar.fetch_add(1U, std::memory_order_relaxed); },
      });
  auto peer = std::make_shared<rclcpp::Node>("meridian_ingress_oversized_test_peer");
  auto publisher = peer->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/meridian_test/lidar_oversized", rclcpp::SensorDataQoS());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(ingress);
  executor.add_node(peer);
  ASSERT_TRUE(spinUntil(executor, [&] { return publisher->get_subscription_count() == 1U; }));

  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp.sec = 1;
  cloud.header.frame_id = "configured_lidar_frame";
  cloud.width = 2U;
  cloud.height = 1U;
  publisher->publish(cloud);
  ASSERT_TRUE(spinUntil(executor, [&] { return sink.failureEvents() == 1U; }));
  EXPECT_EQ(sink.lastErrorCode(), "point_count_limit");
  EXPECT_EQ(forwarded_lidar.load(std::memory_order_relaxed), 0U);

  executor.remove_node(peer);
  executor.remove_node(ingress);
  ingress->stop();
}

TEST_F(SensorIngressNodeTest, DownstreamExceptionIsContainedAndRecorded) {
  CapturingDebugSink sink;
  auto ingress = std::make_shared<SensorIngressNode>(
      optionsFor("exception"), sink,
      ObservationCallbacks{
          .imu = [](core::ImuSample) { throw std::runtime_error("test downstream failure"); },
          .lidar = {},
      });
  auto peer = std::make_shared<rclcpp::Node>("meridian_ingress_exception_test_peer");
  auto publisher = peer->create_publisher<sensor_msgs::msg::Imu>("/meridian_test/imu_exception",
                                                                 rclcpp::SensorDataQoS());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(ingress);
  executor.add_node(peer);
  ASSERT_TRUE(spinUntil(executor, [&] { return publisher->get_subscription_count() == 1U; }));

  publisher->publish(validImu());
  ASSERT_TRUE(spinUntil(executor, [&] { return sink.failureEvents() == 1U; }));
  EXPECT_EQ(sink.imuEvents(), 1U);
  EXPECT_EQ(sink.lastErrorCode(), "downstream_callback_exception");

  executor.remove_node(peer);
  executor.remove_node(ingress);
  ingress->stop();
}

}  // namespace
}  // namespace meridian::ros
