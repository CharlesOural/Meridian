#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <string>

#include "meridian/apps/rerun_debug_sink.hpp"
#include "meridian/ros/sensor_ingress_node.hpp"

namespace {

std::filesystem::path parseRrdPath(int argc, char** argv) {
  for (int index = 1; index < argc; ++index) {
    if (std::string(argv[index]) == "--rrd") {
      if (index + 1 >= argc || std::string(argv[index + 1]).empty()) {
        throw std::invalid_argument("--rrd requires a non-empty output path");
      }
      return argv[index + 1];
    }
  }
  throw std::invalid_argument("missing required --rrd OUTPUT.rrd argument");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::filesystem::path rrd_path = parseRrdPath(argc, argv);
    if (rrd_path.has_parent_path()) {
      std::filesystem::create_directories(rrd_path.parent_path());
    }

    rclcpp::init(argc, argv);
    meridian::apps::RerunDebugSink debug_sink({.output_path = rrd_path});
    auto node =
        std::make_shared<meridian::ros::SensorIngressNode>(rclcpp::NodeOptions{}, debug_sink);

    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions{}, 3U);
    executor.add_node(node);
    executor.spin();

    // MultiThreadedExecutor::spin has joined its callback workers here. Stop
    // admission, drain the LiDAR worker, then drain and flush Rerun.
    executor.remove_node(node);
    node->stop();
    node.reset();
    debug_sink.shutdown();

    const std::uint64_t debug_drops = debug_sink.droppedEvents();
    const std::uint64_t debug_errors = debug_sink.logErrors();
    const std::uintmax_t rrd_bytes = std::filesystem::file_size(rrd_path);
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    std::cerr << "meridian_ingress: recording complete; rrd_bytes=" << rrd_bytes
              << ", debug_events_dropped=" << debug_drops << ", debug_log_errors=" << debug_errors
              << '\n';
    return debug_errors == 0U ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "meridian_ingress: " << error.what() << '\n';
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return 1;
  }
}
