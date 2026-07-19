#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "meridian/core/result.hpp"
#include "meridian/ros/newer_college_calibration.hpp"

namespace meridian::apps {

enum class BagLocalizationSensorMode {
  Imu,
  LidarImu,
  Full,
};

enum class BagLocalizationInitializationMode {
  Dynamic,
  Static,
  SupervisedAuto,
};

enum class BagLocalizationReplayMode {
  Scheduled,
  Unpaced,
};

struct BagLocalizationReplayOptions {
  static constexpr std::size_t kDefaultQueueCountCapacity = 128U;
  static constexpr std::size_t kDefaultQueueByteCapacity = 256U * 1024U * 1024U;
  static constexpr std::size_t kDefaultTimingWindowCapacity = 1'024U;

  BagLocalizationReplayMode mode{BagLocalizationReplayMode::Scheduled};
  // Scheduled replay requires a finite positive value. Unpaced replay requires
  // zero so a stale pacing configuration cannot be mistaken for an active one.
  double playback_rate{};
  std::size_t queue_count_capacity{kDefaultQueueCountCapacity};
  std::size_t queue_byte_capacity{kDefaultQueueByteCapacity};
  std::size_t timing_window_capacity{kDefaultTimingWindowCapacity};
};

struct BagLocalizationOptions {
  std::filesystem::path bag_uri;
  std::filesystem::path calibration_root;
  ros::NewerCollegeCollection collection{ros::NewerCollegeCollection::Collection1};
  std::filesystem::path output_directory;
  std::size_t maximum_events{};
  std::size_t maximum_bag_messages{};
  BagLocalizationSensorMode sensor_mode{BagLocalizationSensorMode::Full};
  // Dynamic is the unsupervised default. Selecting Static or SupervisedAuto is
  // an explicit replay-operator zero-motion assertion and is translated into
  // the ROS-free LocalEstimator ZeroMotionPrior.
  BagLocalizationInitializationMode initialization_mode{BagLocalizationInitializationMode::Dynamic};
  // nullopt selects the full-mode default (enabled). Presence records an
  // explicit caller choice and is rejected for isolated IMU or LiDAR--IMU
  // modes, where no camera frontend exists.
  std::optional<bool> visual_graph_enabled;
  // Benchmark-only profile override. The estimator still has one production
  // registration path; this changes only the bounded number of pose-aware
  // scan-local targets considered by that path and is serialized through the
  // effective estimator configuration.
  std::optional<std::size_t> maximum_lidar_targets;
  BagLocalizationReplayOptions replay;
};

struct BagLocalizationRuntimeSummary {
  BagLocalizationReplayMode replay_mode{BagLocalizationReplayMode::Scheduled};
  double playback_rate{};
  std::size_t queue_count_capacity{};
  std::size_t queue_byte_capacity{};
  std::size_t timing_window_capacity{};
  std::size_t queue_terminal_count{};
  std::size_t queue_terminal_bytes{};
  std::size_t queue_maximum_count{};
  std::size_t queue_maximum_bytes{};
  std::optional<std::int64_t> queue_maximum_oldest_age_ns;
  std::uint64_t queue_rejected{};
  std::uint64_t queue_dropped{};
  std::uint64_t queue_skipped{};
  std::optional<std::int64_t> producer_total_wall_ns;
  std::optional<std::int64_t> producer_p95_wall_ns;
  std::optional<std::int64_t> consumer_total_wall_ns;
  std::optional<std::int64_t> consumer_p95_wall_ns;
  std::optional<std::int64_t> process_ready_p95_wall_ns;
  std::optional<std::int64_t> report_output_p95_wall_ns;
};

struct BagLocalizationOutcome {
  bool execution_succeeded{};
  bool localization_available{};
  bool full_bag_consumed{};
  int suggested_exit_code{1};
  std::size_t states_written{};
  std::size_t fixed_lag_states_written{};
  std::filesystem::path imu_trajectory;
  std::filesystem::path base_trajectory;
  std::filesystem::path fixed_lag_imu_trajectory;
  std::filesystem::path fixed_lag_base_trajectory;
  std::filesystem::path navigation_diagnostics;
  std::filesystem::path run_report;
  std::optional<BagLocalizationRuntimeSummary> runtime;
  std::string error;
};

struct BagLocalizationCommandLine {
  BagLocalizationOptions options;
  bool help{};
};

struct BagLocalizationCommandLineError {
  std::string detail;
};

using BagLocalizationCommandLineResult =
    core::Result<BagLocalizationCommandLine, BagLocalizationCommandLineError>;

// Parses arguments after argv[0]. The benchmark CLI requires an explicit
// replay mode. Scheduled replay also requires --playback-rate; unpaced replay
// forbids it.
[[nodiscard]] BagLocalizationCommandLineResult parseBagLocalizationArguments(
    std::span<const std::string_view> arguments);

// Offline composition root used by both the CLI and its integration test.
// It owns files and ROS 2 replay adapters, but delegates all estimation to the
// unchanged ROS-free LocalEstimator API.
[[nodiscard]] BagLocalizationOutcome runBagLocalization(const BagLocalizationOptions& options);

}  // namespace meridian::apps
