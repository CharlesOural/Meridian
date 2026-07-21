#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <string>
#include <utility>

#include "meridian/apps/local_rt_config_loader.hpp"
#include "meridian/apps/rerun_debug_sink.hpp"
#include "meridian/local_rt/pipeline.hpp"
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
    meridian::local_rt::LocalRtPipelineConfig local_rt_config =
        meridian::apps::loadLocalRtPipelineConfig(*node);
    auto local_rt_pipeline = std::make_shared<meridian::local_rt::LocalRtPipeline>(
        std::move(local_rt_config), debug_sink);
    node->setObservationCallbacks({
        .imu =
            [local_rt_pipeline](meridian::core::ImuSample sample) {
              local_rt_pipeline->submit(std::move(sample));
            },
        .lidar =
            [local_rt_pipeline](meridian::core::LidarSweep sweep) {
              local_rt_pipeline->submit(std::move(sweep));
            },
    });

    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions{}, 3U);
    executor.add_node(node);
    executor.spin();

    // MultiThreadedExecutor::spin has joined its callback workers here. Stop
    // ingress admission and drain its decoder before stopping the downstream
    // pipeline, so no producer can submit after the local worker is drained.
    executor.remove_node(node);
    node->stop();
    local_rt_pipeline->stopAndDrain();
    node.reset();
    local_rt_pipeline.reset();
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
    return debug_errors == 0U && debug_drops == 0U ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "meridian_ingress: " << error.what() << '\n';
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return 1;
  }
}
