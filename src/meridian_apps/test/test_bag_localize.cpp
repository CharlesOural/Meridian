#include <gtest/gtest.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "meridian/apps/bag_localize.hpp"
#include "process_report_normalizer.hpp"

namespace meridian::apps {
namespace {

[[nodiscard]] std::filesystem::path repositoryRoot() {
  return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
}

[[nodiscard]] std::string readFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

[[nodiscard]] std::optional<std::size_t> unsignedAfter(std::string_view contents,
                                                       std::string_view prefix) {
  const std::size_t prefix_position = contents.find(prefix);
  if (prefix_position == std::string_view::npos) {
    return std::nullopt;
  }
  const char* const begin = contents.data() + prefix_position + prefix.size();
  const char* const end = contents.data() + contents.size();
  std::size_t value{};
  const auto parsed = std::from_chars(begin, end, value);
  if (parsed.ec != std::errc{}) {
    return std::nullopt;
  }
  return value;
}

[[nodiscard]] local::LocalGraphCommit makeCommit(std::uint64_t state, std::int64_t time_ns,
                                                 std::uint64_t revision) {
  local::LocalGraphCommit commit;
  commit.odom_epoch = core::OdomEpoch{1U};
  commit.revision = core::LocalGraphRevision{revision};
  commit.parent = core::LocalGraphRevision{revision - 1U};
  commit.state = core::StateId{state};
  commit.state_time = core::FusionTime{time_ns};
  return commit;
}

[[nodiscard]] local::CameraKnotCommitReport makeCameraCommit(const local::LocalGraphCommit& commit,
                                                             std::uint64_t request) {
  local::CameraKnotCommitReport report;
  report.exact_time = commit.state_time;
  report.commit = commit;
  local::VisualResolvedKeyframeReport resolved;
  resolved.resolution.request = core::KnotRequestId{request};
  resolved.resolution.timeline =
      local::StateResolution{commit.state_time, {core::KnotRequestId{request}}, commit.state};
  resolved.resolution.odom_epoch = commit.odom_epoch;
  resolved.resolution.state = commit.state;
  resolved.resolution.created_at_revision = commit.revision;
  report.resolved_keyframes.push_back(std::move(resolved));
  return report;
}

[[nodiscard]] local::LidarCommitReport makeLidarCommit(const local::LocalGraphCommit& commit,
                                                       std::uint64_t measurement) {
  local::LidarCommitReport report;
  report.measurement = core::MeasurementId{measurement};
  report.disposition = local::LidarCommitDisposition::Registered;
  report.commit = commit;
  return report;
}

[[nodiscard]] std::vector<long double> tumTimestamps(const std::string& contents) {
  std::istringstream input(contents);
  std::vector<long double> timestamps;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream row(line);
    long double timestamp{};
    if (row >> timestamp) {
      timestamps.push_back(timestamp);
    }
  }
  return timestamps;
}

[[nodiscard]] std::size_t nonemptyLineCount(const std::string& contents) {
  std::istringstream input(contents);
  std::size_t count{};
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty()) {
      ++count;
    }
  }
  return count;
}

[[nodiscard]] std::vector<std::size_t> csvFieldCounts(const std::string& contents) {
  std::istringstream input(contents);
  std::vector<std::size_t> field_counts;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::size_t fields = 1U;
    bool quoted = false;
    for (std::size_t index = 0U; index < line.size(); ++index) {
      if (line[index] == '"') {
        if (quoted && index + 1U < line.size() && line[index + 1U] == '"') {
          ++index;
        } else {
          quoted = !quoted;
        }
      } else if (line[index] == ',' && !quoted) {
        ++fields;
      }
    }
    field_counts.push_back(fields);
  }
  return field_counts;
}

[[nodiscard]] std::optional<std::size_t> jsonSizeField(const std::string& contents,
                                                       std::string_view name) {
  const std::string prefix = "\"" + std::string(name) + "\": ";
  const std::size_t value_begin = contents.find(prefix);
  if (value_begin == std::string::npos) {
    return std::nullopt;
  }
  const char* begin = contents.data() + value_begin + prefix.size();
  std::size_t value{};
  const auto parsed = std::from_chars(begin, contents.data() + contents.size(), value);
  if (parsed.ec != std::errc{}) {
    return std::nullopt;
  }
  return value;
}

[[nodiscard]] std::optional<double> jsonDoubleField(const std::string& contents,
                                                    std::string_view name) {
  const std::string prefix = "\"" + std::string(name) + "\": ";
  const std::size_t value_begin = contents.find(prefix);
  if (value_begin == std::string::npos) {
    return std::nullopt;
  }
  const char* begin = contents.data() + value_begin + prefix.size();
  double value{};
  const auto parsed = std::from_chars(begin, contents.data() + contents.size(), value);
  if (parsed.ec != std::errc{}) {
    return std::nullopt;
  }
  return value;
}

[[nodiscard]] std::optional<std::size_t> pipelineStageTotalSamples(const std::string& contents,
                                                                   std::string_view stage_name) {
  const std::string stage_token = "\"name\": \"" + std::string(stage_name) + "\"";
  const std::size_t stage_begin = contents.find(stage_token);
  if (stage_begin == std::string::npos) {
    return std::nullopt;
  }
  constexpr std::string_view kTotalToken = "\"wall\": {\"total_samples\": ";
  const std::size_t total_begin = contents.find(kTotalToken, stage_begin);
  if (total_begin == std::string::npos) {
    return std::nullopt;
  }
  const char* begin = contents.data() + total_begin + kTotalToken.size();
  std::size_t value{};
  const auto parsed = std::from_chars(begin, contents.data() + contents.size(), value);
  if (parsed.ec != std::errc{}) {
    return std::nullopt;
  }
  return value;
}

void useUnpacedReplay(BagLocalizationOptions* options) {
  options->replay.mode = BagLocalizationReplayMode::Unpaced;
}

TEST(ProcessReportNormalizer, SharedCameraAndLidarStateEmitsOnceWithLidarMetadata) {
  const local::LocalGraphCommit commit = makeCommit(11U, 1'100LL, 21U);
  local::LocalEstimatorProcessReport report;
  report.commits.push_back(makeLidarCommit(commit, 31U));
  report.camera_commits.push_back(makeCameraCommit(commit, 41U));

  auto normalized = detail::normalizeProcessReport(report);

  ASSERT_TRUE(normalized) << normalized.error().detail;
  ASSERT_EQ(normalized.value().commits.size(), 1U);
  const detail::NormalizedProcessCommit& state = normalized.value().commits.front();
  EXPECT_EQ(state.commit, &report.commits.front().commit);
  EXPECT_EQ(state.lidar, &report.commits.front());
  ASSERT_EQ(state.cameras.size(), 1U);
  EXPECT_EQ(state.cameras.front(), &report.camera_commits.front());
  EXPECT_EQ(normalized.value().input_commit_references, 2U);
  EXPECT_EQ(normalized.value().duplicate_references_removed, 1U);
  EXPECT_EQ(normalized.value().shared_camera_lidar_states, 1U);
  EXPECT_EQ(normalized.value().camera_only_states, 0U);
  EXPECT_TRUE(state.emits_navigation_state);
}

TEST(ProcessReportNormalizer, FactorOnlyLidarReferenceDoesNotEmitDuplicateNavigationState) {
  const local::LocalGraphCommit commit = makeCommit(11U, 1'100LL, 22U);
  local::LocalEstimatorProcessReport report;
  report.commits.push_back(makeLidarCommit(commit, 31U));
  report.commits.front().navigation_state_created = false;

  auto normalized = detail::normalizeProcessReport(report);

  ASSERT_TRUE(normalized) << normalized.error().detail;
  ASSERT_EQ(normalized.value().commits.size(), 1U);
  EXPECT_FALSE(normalized.value().commits.front().emits_navigation_state);
  EXPECT_EQ(normalized.value().commits.front().commit, &report.commits.front().commit);
}

TEST(ProcessReportNormalizer, CameraStateAndLaterLidarFactorRevisionEmitOneUpdatedState) {
  const local::LocalGraphCommit camera_commit = makeCommit(11U, 1'100LL, 21U);
  local::LocalGraphCommit lidar_commit = makeCommit(11U, 1'100LL, 22U);
  lidar_commit.parent = camera_commit.revision;
  local::LocalEstimatorProcessReport report;
  report.camera_commits.push_back(makeCameraCommit(camera_commit, 41U));
  report.commits.push_back(makeLidarCommit(lidar_commit, 31U));
  report.commits.front().navigation_state_created = false;

  auto normalized = detail::normalizeProcessReport(report);

  ASSERT_TRUE(normalized) << normalized.error().detail;
  ASSERT_EQ(normalized.value().commits.size(), 1U);
  const detail::NormalizedProcessCommit& state = normalized.value().commits.front();
  EXPECT_TRUE(state.emits_navigation_state);
  EXPECT_EQ(state.commit, &report.commits.front().commit);
  EXPECT_EQ(state.commit->revision, core::LocalGraphRevision{22U});
  ASSERT_EQ(state.cameras.size(), 1U);
}

TEST(ProcessReportNormalizer, AcceptsRootInitializationWithoutAParentRevision) {
  local::LocalEstimatorProcessReport report;
  report.initialization = makeCommit(1U, 1'000LL, 1U);
  report.initialization->parent = core::LocalGraphRevision{};

  auto normalized = detail::normalizeProcessReport(report);

  ASSERT_TRUE(normalized) << normalized.error().detail;
  ASSERT_EQ(normalized.value().commits.size(), 1U);
  EXPECT_TRUE(normalized.value().commits.front().initialization);
  EXPECT_EQ(normalized.value().commits.front().commit, &*report.initialization);
}

TEST(ProcessReportNormalizer, OrdersInterleavedModalityReferencesByStateId) {
  const local::LocalGraphCommit camera_commit = makeCommit(11U, 1'100LL, 21U);
  const local::LocalGraphCommit lidar_commit = makeCommit(12U, 1'200LL, 22U);
  local::LocalEstimatorProcessReport report;
  report.commits.push_back(makeLidarCommit(lidar_commit, 31U));
  report.camera_commits.push_back(makeCameraCommit(camera_commit, 41U));

  auto normalized = detail::normalizeProcessReport(report);

  ASSERT_TRUE(normalized) << normalized.error().detail;
  ASSERT_EQ(normalized.value().commits.size(), 2U);
  EXPECT_EQ(normalized.value().commits[0].commit->state, core::StateId{11U});
  EXPECT_EQ(normalized.value().commits[1].commit->state, core::StateId{12U});
  EXPECT_EQ(normalized.value().camera_only_states, 1U);
  EXPECT_EQ(normalized.value().duplicate_references_removed, 0U);
}

TEST(ProcessReportNormalizer, RejectsConflictingMetadataForSharedState) {
  const local::LocalGraphCommit lidar_commit = makeCommit(11U, 1'100LL, 21U);
  local::LocalGraphCommit camera_commit = lidar_commit;
  camera_commit.parent = core::LocalGraphRevision{19U};
  local::LocalEstimatorProcessReport report;
  report.commits.push_back(makeLidarCommit(lidar_commit, 31U));
  report.camera_commits.push_back(makeCameraCommit(camera_commit, 41U));

  auto normalized = detail::normalizeProcessReport(report);

  ASSERT_FALSE(normalized);
  EXPECT_EQ(normalized.error().code,
            detail::ProcessReportNormalizationErrorCode::ConflictingCommitReferences);
}

TEST(ProcessReportNormalizer, RejectsCameraResolutionThatIsNotExact) {
  const local::LocalGraphCommit commit = makeCommit(11U, 1'100LL, 21U);
  local::LocalEstimatorProcessReport report;
  report.camera_commits.push_back(makeCameraCommit(commit, 41U));
  report.camera_commits.front().resolved_keyframes.front().resolution.timeline.exact_time =
      core::FusionTime{1'101LL};

  auto normalized = detail::normalizeProcessReport(report);

  ASSERT_FALSE(normalized);
  EXPECT_EQ(normalized.error().code,
            detail::ProcessReportNormalizationErrorCode::InvalidCommitMetadata);
}

TEST(ProcessReportNormalizer, RejectsStateOrderThatDisagreesWithExactTime) {
  local::LocalEstimatorProcessReport report;
  report.commits.push_back(makeLidarCommit(makeCommit(12U, 1'100LL, 22U), 32U));
  report.camera_commits.push_back(makeCameraCommit(makeCommit(11U, 1'200LL, 21U), 41U));

  auto normalized = detail::normalizeProcessReport(report);

  ASSERT_FALSE(normalized);
  EXPECT_EQ(normalized.error().code,
            detail::ProcessReportNormalizationErrorCode::NonMonotonicCommitOrder);
}

TEST(BagLocalizeCommandLine, ParsesScheduledAndUnpacedReplayWithExclusivePlaybackRate) {
  const std::vector<std::string_view> valid{"--bag",
                                            "/bags/quad-easy",
                                            "--calib",
                                            "/bags/calib",
                                            "--collection",
                                            "1",
                                            "--sensor-mode",
                                            "lidar-imu",
                                            "--output",
                                            "/tmp/output",
                                            "--max-events",
                                            "100",
                                            "--max-bag-messages",
                                            "200",
                                            "--replay-mode",
                                            "scheduled",
                                            "--playback-rate",
                                            "1.5"};

  auto parsed = parseBagLocalizationArguments(valid);

  ASSERT_TRUE(parsed) << parsed.error().detail;
  EXPECT_FALSE(parsed.value().help);
  EXPECT_EQ(parsed.value().options.bag_uri, "/bags/quad-easy");
  EXPECT_EQ(parsed.value().options.replay.mode, BagLocalizationReplayMode::Scheduled);
  EXPECT_DOUBLE_EQ(parsed.value().options.replay.playback_rate, 1.5);
  EXPECT_EQ(parsed.value().options.initialization_mode, BagLocalizationInitializationMode::Dynamic);
  EXPECT_EQ(parsed.value().options.replay.queue_count_capacity,
            BagLocalizationReplayOptions::kDefaultQueueCountCapacity);
  EXPECT_EQ(parsed.value().options.replay.queue_byte_capacity,
            BagLocalizationReplayOptions::kDefaultQueueByteCapacity);
  EXPECT_EQ(parsed.value().options.replay.timing_window_capacity,
            BagLocalizationReplayOptions::kDefaultTimingWindowCapacity);

  std::vector<std::string_view> missing_rate = valid;
  missing_rate.resize(missing_rate.size() - 2U);
  auto missing = parseBagLocalizationArguments(missing_rate);
  ASSERT_FALSE(missing);
  EXPECT_NE(missing.error().detail.find("--playback-rate"), std::string::npos);

  std::vector<std::string_view> zero_rate = valid;
  zero_rate.back() = "0";
  auto zero = parseBagLocalizationArguments(zero_rate);
  ASSERT_FALSE(zero);
  EXPECT_NE(zero.error().detail.find("finite positive"), std::string::npos);

  std::vector<std::string_view> unpaced = valid;
  const auto replay_value = std::find(unpaced.begin(), unpaced.end(), "scheduled");
  ASSERT_NE(replay_value, unpaced.end());
  *replay_value = "unpaced";
  unpaced.resize(unpaced.size() - 2U);
  auto unpaced_result = parseBagLocalizationArguments(unpaced);
  ASSERT_TRUE(unpaced_result) << unpaced_result.error().detail;
  EXPECT_EQ(unpaced_result.value().options.replay.mode, BagLocalizationReplayMode::Unpaced);
  EXPECT_DOUBLE_EQ(unpaced_result.value().options.replay.playback_rate, 0.0);

  std::vector<std::string_view> forbidden_rate = valid;
  const auto forbidden_replay_value =
      std::find(forbidden_rate.begin(), forbidden_rate.end(), "scheduled");
  ASSERT_NE(forbidden_replay_value, forbidden_rate.end());
  *forbidden_replay_value = "unpaced";
  auto unpaced_with_rate = parseBagLocalizationArguments(forbidden_rate);
  ASSERT_FALSE(unpaced_with_rate);
  EXPECT_NE(unpaced_with_rate.error().detail.find("forbids --playback-rate"), std::string::npos);

  std::vector<std::string_view> missing_mode = valid;
  const auto mode_option = std::find(missing_mode.begin(), missing_mode.end(), "--replay-mode");
  ASSERT_NE(mode_option, missing_mode.end());
  missing_mode.erase(mode_option, mode_option + 2);
  auto no_mode = parseBagLocalizationArguments(missing_mode);
  ASSERT_FALSE(no_mode);
  EXPECT_NE(no_mode.error().detail.find("--replay-mode"), std::string::npos);
}

TEST(BagLocalizeCommandLine, ParsesExplicitInitializationPolicy) {
  const std::vector<std::string_view> base{"--bag",
                                           "/bags/quad-easy",
                                           "--calib",
                                           "/bags/calib",
                                           "--collection",
                                           "1",
                                           "--sensor-mode",
                                           "lidar-imu",
                                           "--output",
                                           "/tmp/output",
                                           "--max-events",
                                           "100",
                                           "--max-bag-messages",
                                           "200",
                                           "--replay-mode",
                                           "scheduled",
                                           "--playback-rate",
                                           "1",
                                           "--initialization",
                                           "static"};
  auto static_mode = parseBagLocalizationArguments(base);
  ASSERT_TRUE(static_mode) << static_mode.error().detail;
  EXPECT_EQ(static_mode.value().options.initialization_mode,
            BagLocalizationInitializationMode::Static);

  std::vector<std::string_view> supervised = base;
  supervised.back() = "supervised-auto";
  auto supervised_mode = parseBagLocalizationArguments(supervised);
  ASSERT_TRUE(supervised_mode) << supervised_mode.error().detail;
  EXPECT_EQ(supervised_mode.value().options.initialization_mode,
            BagLocalizationInitializationMode::SupervisedAuto);

  std::vector<std::string_view> invalid = base;
  invalid.back() = "guess";
  auto invalid_mode = parseBagLocalizationArguments(invalid);
  ASSERT_FALSE(invalid_mode);
  EXPECT_NE(invalid_mode.error().detail.find("dynamic, static, or supervised-auto"),
            std::string::npos);
}

TEST(BagLocalizeCommandLine, ParsesExplicitQueueAndTimingCapacities) {
  const std::vector<std::string_view> arguments{"--bag",
                                                "/bags/quad-easy",
                                                "--calib",
                                                "/bags/calib",
                                                "--collection",
                                                "1",
                                                "--sensor-mode",
                                                "full",
                                                "--output",
                                                "/tmp/output",
                                                "--max-events",
                                                "100",
                                                "--max-bag-messages",
                                                "200",
                                                "--replay-mode",
                                                "scheduled",
                                                "--playback-rate",
                                                "2",
                                                "--queue-count-capacity",
                                                "17",
                                                "--queue-byte-capacity",
                                                "4096",
                                                "--timing-window-capacity",
                                                "31",
                                                "--maximum-lidar-targets",
                                                "2",
                                                "--visual-graph",
                                                "disabled"};

  auto parsed = parseBagLocalizationArguments(arguments);

  ASSERT_TRUE(parsed) << parsed.error().detail;
  EXPECT_EQ(parsed.value().options.replay.queue_count_capacity, 17U);
  EXPECT_EQ(parsed.value().options.replay.queue_byte_capacity, 4'096U);
  EXPECT_EQ(parsed.value().options.replay.timing_window_capacity, 31U);
  ASSERT_TRUE(parsed.value().options.maximum_lidar_targets);
  EXPECT_EQ(*parsed.value().options.maximum_lidar_targets, 2U);
  ASSERT_TRUE(parsed.value().options.visual_graph_enabled);
  EXPECT_FALSE(*parsed.value().options.visual_graph_enabled);

  std::vector<std::string_view> zero_queue = arguments;
  const auto queue_value =
      std::find(zero_queue.begin(), zero_queue.end(), "--queue-count-capacity");
  ASSERT_NE(queue_value, zero_queue.end());
  *(queue_value + 1) = "0";
  auto invalid = parseBagLocalizationArguments(zero_queue);
  ASSERT_FALSE(invalid);
  EXPECT_NE(invalid.error().detail.find("positive integer"), std::string::npos);

  std::vector<std::string_view> zero_targets = arguments;
  const auto target_value =
      std::find(zero_targets.begin(), zero_targets.end(), "--maximum-lidar-targets");
  ASSERT_NE(target_value, zero_targets.end());
  *(target_value + 1) = "0";
  auto invalid_targets = parseBagLocalizationArguments(zero_targets);
  ASSERT_FALSE(invalid_targets);
  EXPECT_NE(invalid_targets.error().detail.find("1 or 2"), std::string::npos);

  *(target_value + 1) = "3";
  invalid_targets = parseBagLocalizationArguments(zero_targets);
  ASSERT_FALSE(invalid_targets);
  EXPECT_NE(invalid_targets.error().detail.find("1 or 2"), std::string::npos);
}

TEST(BagLocalize, QuadEasyCappedMultisensorComposition) {
  const std::filesystem::path root = repositoryRoot();
  const std::filesystem::path output =
      std::filesystem::temp_directory_path() / "meridian_apps_quad_easy_smoke";
  std::error_code ignored;
  std::filesystem::remove_all(output, ignored);

  BagLocalizationOptions options;
  options.bag_uri = root / "bags/newer-college/quad-easy";
  options.calibration_root = root / "bags/newer-college/calib";
  options.collection = ros::NewerCollegeCollection::Collection1;
  options.output_directory = output;
  options.maximum_events = 1'700U;
  options.maximum_bag_messages = 5'500U;
  options.sensor_mode = BagLocalizationSensorMode::Full;
  useUnpacedReplay(&options);

  const BagLocalizationOutcome outcome = runBagLocalization(options);

  ASSERT_TRUE(outcome.execution_succeeded) << outcome.error;
  // Quad Easy begins in motion. The cap includes enough explicitly correlated
  // LiDAR--IMU support for the two-pass moving initialization and tracking.
  EXPECT_TRUE(outcome.localization_available);
  EXPECT_FALSE(outcome.full_bag_consumed);
  EXPECT_EQ(outcome.suggested_exit_code, 0);
  EXPECT_GE(outcome.states_written, 2U);
  ASSERT_TRUE(outcome.runtime);
  EXPECT_EQ(outcome.runtime->replay_mode, BagLocalizationReplayMode::Unpaced);
  EXPECT_DOUBLE_EQ(outcome.runtime->playback_rate, 0.0);
  EXPECT_FALSE(outcome.runtime->producer_total_wall_ns);
  EXPECT_TRUE(std::filesystem::exists(outcome.imu_trajectory));
  EXPECT_TRUE(std::filesystem::exists(outcome.base_trajectory));
  EXPECT_TRUE(std::filesystem::exists(outcome.fixed_lag_imu_trajectory));
  EXPECT_TRUE(std::filesystem::exists(outcome.fixed_lag_base_trajectory));
  EXPECT_TRUE(std::filesystem::exists(outcome.navigation_diagnostics));
  EXPECT_TRUE(std::filesystem::exists(outcome.run_report));

  const std::string imu = readFile(outcome.imu_trajectory);
  const std::string base = readFile(outcome.base_trajectory);
  const std::string fixed_lag_imu = readFile(outcome.fixed_lag_imu_trajectory);
  const std::string fixed_lag_base = readFile(outcome.fixed_lag_base_trajectory);
  const std::string diagnostics = readFile(outcome.navigation_diagnostics);
  const std::string report = readFile(outcome.run_report);
  EXPECT_FALSE(imu.empty());
  EXPECT_FALSE(base.empty());
  EXPECT_FALSE(fixed_lag_imu.empty());
  EXPECT_FALSE(fixed_lag_base.empty());
  const std::vector<long double> timestamps = tumTimestamps(imu);
  ASSERT_EQ(timestamps.size(), outcome.states_written);
  EXPECT_TRUE(std::is_sorted(timestamps.begin(), timestamps.end()));
  EXPECT_EQ(std::adjacent_find(timestamps.begin(), timestamps.end()), timestamps.end());
  const std::vector<long double> fixed_lag_timestamps = tumTimestamps(fixed_lag_imu);
  EXPECT_EQ(fixed_lag_timestamps.size(), outcome.states_written);
  EXPECT_EQ(outcome.fixed_lag_states_written, outcome.states_written);
  EXPECT_TRUE(std::is_sorted(fixed_lag_timestamps.begin(), fixed_lag_timestamps.end()));
  EXPECT_EQ(std::adjacent_find(fixed_lag_timestamps.begin(), fixed_lag_timestamps.end()),
            fixed_lag_timestamps.end());
  EXPECT_EQ(nonemptyLineCount(diagnostics), outcome.states_written + 1U);
  const std::vector<std::size_t> diagnostic_field_counts = csvFieldCounts(diagnostics);
  ASSERT_EQ(diagnostic_field_counts.size(), outcome.states_written + 1U);
  ASSERT_FALSE(diagnostic_field_counts.empty());
  for (const std::size_t field_count : diagnostic_field_counts) {
    EXPECT_EQ(field_count, diagnostic_field_counts.front());
  }
  EXPECT_NE(diagnostics.find("gyro_bias_x_radps"), std::string::npos);
  EXPECT_NE(diagnostics.find("motion_initialization"), std::string::npos);
  EXPECT_NE(report.find("\"execution_succeeded\": true"), std::string::npos);
  EXPECT_NE(report.find("\"partial_coverage\": true"), std::string::npos);
  EXPECT_NE(report.find("\"localization_available\": true"), std::string::npos);
  EXPECT_NE(report.find("\"motion_initializations\": 1"), std::string::npos);
  EXPECT_NE(report.find("\"body_frame\": \"base_link\""), std::string::npos);
  // NCD's published gyro-bias random walk is known to let bias absorb LiDAR
  // residuals on the full sequence. Guard both the calibration object and the
  // effective graph profile against accidentally restoring that bad import.
  EXPECT_NE(report.find("\"gyroscope_bias_random_walk\": 3.9999999999999998e-06"),
            std::string::npos);
  EXPECT_NE(report.find("\"gyroscope_bias_random_walk_radps2_sqrt_hz\": 3.9999999999999998e-06"),
            std::string::npos);
  EXPECT_NE(report.find("\"schema\": \"meridian_navigation_diagnostics_csv_v19\""),
            std::string::npos);
  EXPECT_NE(report.find("\"nonlinear_convergence_sigma_fraction\": 0.25"), std::string::npos);
  EXPECT_NE(report.find("\"complete_objective_nonsmooth_relative_allowance\": 0.001"),
            std::string::npos);
  EXPECT_NE(report.find("\"parallel_worker_count\": 4"), std::string::npos);
  EXPECT_NE(report.find("\"lidar_factor_correlation\": {\"live_only_policy_revision\": 1"),
            std::string::npos);
  EXPECT_NE(report.find("\"mixed_or_finalized_map_policy_revision\": 3"), std::string::npos);
  EXPECT_NE(report.find("\"base_covariance_inflation\": 6"), std::string::npos);
  EXPECT_NE(report.find("\"finalized_map_correlation_inflation_floor\": 1"),
            std::string::npos);
  EXPECT_NE(report.find("\"live_information_scale_formula\":"), std::string::npos);
  EXPECT_NE(report.find("\"live_translation_information_cap\":"), std::string::npos);
  EXPECT_NE(report.find("\"maximum_pending_finalized_lidar_sweeps\":"), std::string::npos);
  EXPECT_NE(report.find("\"finalized_lidar_prune_interval_sweeps\":"), std::string::npos);
  EXPECT_NE(report.find("\"finalized_lidar_target\": {\"odom_epoch\":"), std::string::npos);
  EXPECT_NE(report.find("\"maximum_finalized_lidar_owners_per_factor\":"), std::string::npos);
  EXPECT_NE(report.find("\"finalized_map_covariance_inflation_formula\":"), std::string::npos);
  EXPECT_NE(report.find("\"effective_finalized_map_covariance_inflation_formula\":"),
            std::string::npos);
  EXPECT_NE(report.find("\"finalized_map_information_scale_formula\":"), std::string::npos);
  EXPECT_NE(diagnostics.find("effective_translation_convergence_m"), std::string::npos);
  EXPECT_NE(diagnostics.find("rejected_effective_velocity_convergence_mps"), std::string::npos);
  EXPECT_NE(diagnostics.find("rejected_last_velocity_correction_mps"), std::string::npos);
  EXPECT_NE(diagnostics.find("nonlinear_cauchy_directions_attempted"), std::string::npos);
  EXPECT_NE(diagnostics.find("nonlinear_cauchy_steps_accepted"), std::string::npos);
  EXPECT_NE(diagnostics.find("rejected_nonlinear_cauchy_backtracking_trials"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_registration_termination"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_registration_overlap_fraction"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_registration_effective_correspondences"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_registration_normalized_observable_eigenvalue_threshold"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_registration_physical_information_eigenvalue_5"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_registration_live_target_count"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_registration_finalized_map_correspondences"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_registration_finalized_map_stale_fallbacks"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_registration_T_odom_source_qw"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_registration_source_right_correction_qw"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_registration_error_code"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_health_before_recovery_epoch"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_health_after_state"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_removed_factor_batches"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_batch_correlation_policy_revision"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_batch_covariance_inflation"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_batch_declared_information_cap"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_batch_conditioning_measurement_roots"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_batch_finalized_map_factors"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_batch_live_information_scale"), std::string::npos);
  EXPECT_NE(diagnostics.find("lidar_batch_finalized_map_information_scale"), std::string::npos);
  EXPECT_EQ(diagnostics.find("lidar_batch_information_scale"), std::string::npos);
  EXPECT_NE(diagnostics.find("accepted_finalized_map_unique_owners"), std::string::npos);
  EXPECT_NE(diagnostics.find("accepted_finalized_map_configured_correlation_inflation_floor"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("accepted_finalized_map_effective_covariance_inflation"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("accepted_finalized_map_information_eigenvalue_5"), std::string::npos);
  EXPECT_NE(diagnostics.find("rejected_finalized_map_owner_pose_covariance_inflation"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("rejected_finalized_map_configured_correlation_inflation_floor"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("rejected_finalized_map_effective_covariance_inflation"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("motion_calibrated_data_rank"), std::string::npos);
  EXPECT_NE(diagnostics.find("motion_prior_resolved_accel_tilt_modes"), std::string::npos);
  EXPECT_NE(diagnostics.find("motion_observability_class"), std::string::npos);
  EXPECT_NE(diagnostics.find("motion_calibrated_data_supported_condition"), std::string::npos);
  EXPECT_NE(report.find("\"initialization_diagnostics\": {"), std::string::npos);
  EXPECT_NE(report.find("\"calibrated_data_rank\":"), std::string::npos);
  EXPECT_NE(report.find("\"prior_resolved_accel_tilt_modes\":"), std::string::npos);
  EXPECT_NE(report.find("\"observability_class\":"), std::string::npos);
  EXPECT_NE(report.find("\"calibrated_data_supported_condition\":"), std::string::npos);
  EXPECT_NE(report.find("\"maximum_prior_resolved_accel_tilt_modes\": 2"), std::string::npos);
  EXPECT_NE(report.find("\"last_initialization_rejection_detail\""), std::string::npos);
  EXPECT_NE(report.find("\"registered_commits\""), std::string::npos);
  EXPECT_NE(report.find("\"registered_tracking_only_commits\""), std::string::npos);
  EXPECT_NE(report.find("\"lidar_tracking_only_registrations\""), std::string::npos);
  EXPECT_NE(report.find("\"last_lidar_keyframe_time_ns\""), std::string::npos);
  EXPECT_NE(report.find("\"minimum_lidar_factor_interval_ns\": 150000000"), std::string::npos);
  EXPECT_NE(report.find("\"terminations\""), std::string::npos);
  EXPECT_NE(report.find("\"converged\""), std::string::npos);
  EXPECT_NE(report.find("\"insufficient_correspondences\""), std::string::npos);
  EXPECT_NE(report.find("\"numerical_failure\""), std::string::npos);
  EXPECT_NE(report.find("\"effective_profile\""), std::string::npos);
  EXPECT_NE(report.find("\"local_solver_globalization\": {\"committed\":"), std::string::npos);
  EXPECT_NE(report.find("\"finalized_lidar_target\": {\"process_reports\":"), std::string::npos);
  EXPECT_NE(report.find("\"rollback_removals\":"), std::string::npos);
  EXPECT_NE(report.find("\"capacity\": {\"recovery_attempts\":"), std::string::npos);
  EXPECT_NE(report.find("\"recovery_successes\":"), std::string::npos);
  EXPECT_NE(report.find("\"skipped_sweeps\":"), std::string::npos);
  EXPECT_NE(report.find("\"retry_suppressed_sweeps\":"), std::string::npos);
  EXPECT_NE(report.find("\"last_skip\": null"), std::string::npos);
  EXPECT_NE(report.find("\"frozen_process_reports\":"), std::string::npos);
  EXPECT_NE(report.find("\"capacity_saturated\": false"), std::string::npos);
  EXPECT_NE(report.find("\"capacity_skips_since_retry\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"map_checksum\":"), std::string::npos);
  EXPECT_NE(report.find("\"finalized_lidar_capacity_recovery_attempts\":"), std::string::npos);
  EXPECT_NE(report.find("\"finalized_lidar_capacity_recovery_successes\":"), std::string::npos);
  EXPECT_NE(report.find("\"finalized_lidar_capacity_skipped_sweeps\":"), std::string::npos);
  EXPECT_NE(report.find("\"finalized_lidar_capacity_retry_suppressions\":"), std::string::npos);
  EXPECT_NE(report.find("\"accepted_finalized_map_reports\":"), std::string::npos);
  EXPECT_NE(report.find("\"last_accepted_finalized_map\":"), std::string::npos);
  EXPECT_NE(report.find("\"last_rejected_finalized_map\":"), std::string::npos);
  const auto process_calls = unsignedAfter(report, "\"processing_counts\": {\"process_calls\": ");
  const auto finalized_target_process_reports =
      unsignedAfter(report, "\"finalized_lidar_target\": {\"process_reports\": ");
  ASSERT_TRUE(process_calls);
  ASSERT_TRUE(finalized_target_process_reports);
  EXPECT_EQ(*finalized_target_process_reports, *process_calls);
  const auto graph_commits = unsignedAfter(report, "\"graph_commits\": ");
  const auto observed_solver_transactions =
      unsignedAfter(report, "\"local_solver_globalization\": {\"committed\": {\"transactions\": ");
  ASSERT_TRUE(graph_commits);
  ASSERT_TRUE(observed_solver_transactions);
  EXPECT_EQ(*observed_solver_transactions, *graph_commits);
  EXPECT_NE(report.find("\"cauchy_directions_attempted\":"), std::string::npos);
  EXPECT_NE(report.find("\"schema_version\": 27"), std::string::npos);
  EXPECT_NE(report.find("\"initialization\": \"dynamic\""), std::string::npos);
  EXPECT_NE(report.find("\"initialization\": {\"mode\": \"dynamic_only\", "
                        "\"zero_motion_prior\": null}"),
            std::string::npos);
  EXPECT_NE(report.find("\"mode\": \"unpaced\""), std::string::npos);
  EXPECT_NE(report.find("\"timing_window_capacity\": 1024"), std::string::npos);
  EXPECT_NE(report.find("\"app.driver.imu_ingress\""), std::string::npos);
  EXPECT_NE(report.find("\"app.driver.lidar_enqueue\""), std::string::npos);
  EXPECT_NE(report.find("\"app.driver.camera_ingress\""), std::string::npos);
  EXPECT_NE(report.find("\"app.driver.process_ready\""), std::string::npos);
  EXPECT_NE(report.find("\"app.driver.report_output\""), std::string::npos);
  EXPECT_NE(report.find("\"local_stages\": {\"schema_version\": 1"), std::string::npos);
  EXPECT_NE(report.find("\"span_semantics\": \"inclusive\""), std::string::npos);
  EXPECT_NE(report.find("\"pipeline_timing\": {\"window_capacity\": 1024, "
                        "\"maximum_window_capacity\": 65536}"),
            std::string::npos);
  EXPECT_NE(report.find("\"local.initialization.stationary_probe\""), std::string::npos);
  EXPECT_NE(report.find("\"local.initialization.bootstrap_acquisition_deskew\""),
            std::string::npos);
  EXPECT_NE(report.find("\"local.lidar.registration_view_build\""), std::string::npos);
  EXPECT_NE(report.find("\"local.lidar.target_build_update\""), std::string::npos);
  EXPECT_NE(report.find("\"local.lidar.correspondence_registration_solve\""), std::string::npos);
  EXPECT_NE(report.find("\"local.lidar.finalized_target_update\""), std::string::npos);
  EXPECT_NE(report.find("\"local.initialization.motion_batch_solve_refinement\""),
            std::string::npos);
  EXPECT_NE(report.find("\"local.tracking.deskew\""), std::string::npos);
  EXPECT_NE(report.find("\"local.graph.transaction_update\""), std::string::npos);
  EXPECT_NE(report.find("\"retained_samples_high_watermark\":"), std::string::npos);
  EXPECT_NE(report.find("\"newest_gap_maximum_observed_ns\":"), std::string::npos);
  EXPECT_NE(report.find("\"newest_gap\": {\"total_samples\":"), std::string::npos);
  EXPECT_NE(report.find("\"sensor_mode\": \"full\""), std::string::npos);
  EXPECT_NE(report.find("\"sensor_mode_proof\": {"), std::string::npos);
  EXPECT_NE(report.find("\"configured\": {\"imu_topics\": 1, \"lidar_topics\": 1, "
                        "\"camera_topics\": 4, \"gnss_topics\": 0, \"visual_lanes\": 4}"),
            std::string::npos);
  EXPECT_NE(report.find("\"configuration_matches\": true, "
                        "\"runtime_isolation_matches\": true, \"passed\": true"),
            std::string::npos);
  EXPECT_NE(report.find("\"visual_graph\": \"enabled\""), std::string::npos);
  EXPECT_NE(report.find("\"graph_submission_enabled\": true"), std::string::npos);
  EXPECT_NE(report.find("\"visual_lane_finality_updates\":"), std::string::npos);
  EXPECT_NE(report.find("\"visual_finalized_landmarks\":"), std::string::npos);
  EXPECT_NE(report.find("\"visual_finalized_factors\":"), std::string::npos);
  EXPECT_NE(report.find("\"visual_finalized_tracks_pruned\":"), std::string::npos);
  EXPECT_NE(report.find("\"visual_finality_pending_factors_pruned\":"), std::string::npos);
  EXPECT_NE(report.find("\"lidar_registration_diagnostics\":"), std::string::npos);
  EXPECT_NE(report.find("\"overlap_fraction\":"), std::string::npos);
  EXPECT_NE(report.find("\"normalized_observable_eigenvalue_threshold\":"), std::string::npos);
  EXPECT_NE(report.find("\"physical_information\":"), std::string::npos);
  EXPECT_NE(report.find("\"last_error\":"), std::string::npos);
  EXPECT_NE(report.find("\"work\": {\"source_points_considered\":"), std::string::npos);
  EXPECT_NE(report.find("\"lidar_sensor_health\":"), std::string::npos);
  EXPECT_NE(report.find("\"health_quarantined_commits\":"), std::string::npos);
  EXPECT_NE(report.find("\"factor_batches_removed\":"), std::string::npos);
  EXPECT_NE(report.find("\"sensor_health_policy\":"), std::string::npos);
  EXPECT_NE(report.find("\"maximum_recent_faulty_batches_to_remove\":"), std::string::npos);
  EXPECT_NE(report.find("\"maximum_active_factor_batches\":"), std::string::npos);
  EXPECT_NE(report.find("\"target_state_unavailable_frozen_commits\":"), std::string::npos);
  EXPECT_NE(report.find("\"registration_target_frozen\":"), std::string::npos);
  EXPECT_NE(report.find("\"lidar_target_state_unavailable_freezes\":"), std::string::npos);
  EXPECT_NE(report.find("\"deskew_solve_passes\": 2"), std::string::npos);
  EXPECT_NE(report.find("\"holdout_msw\""), std::string::npos);
  EXPECT_NE(report.find("\"conditioned_lidar_imu_approximation\": true"), std::string::npos);
  EXPECT_NE(report.find("\"name\": \"newer-college-ros2\""), std::string::npos);
  EXPECT_NE(report.find("\"max_events\": 1700"), std::string::npos);
  EXPECT_NE(report.find("\"camera_ingest_rejected\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"storage_filter_matches\": true"), std::string::npos);
  EXPECT_NE(report.find("\"camera_counts\""), std::string::npos);
  EXPECT_NE(report.find("\"keyframes_suppressed_by_timeline\""), std::string::npos);
  EXPECT_NE(report.find("\"keyframes_rejected_by_timeline\""), std::string::npos);
  EXPECT_NE(report.find("\"visual_cameras\""), std::string::npos);
  EXPECT_NE(report.find("\"wire_format\": \"compressed_image\""), std::string::npos);
  EXPECT_NE(report.find("\"stamp_to_exposure_midpoint_ns\""), std::string::npos);
  EXPECT_NE(report.find("\"cameras\": ["), std::string::npos);
  EXPECT_NE(report.find("/alphasense_driver_ros/cam0/compressed"), std::string::npos);
  EXPECT_NE(report.find("/alphasense_driver_ros/cam4/compressed"), std::string::npos);
}

TEST(BagLocalize, VisualGraphDisabledRetainsCameraTrackingWithoutGraphKnots) {
  const std::filesystem::path root = repositoryRoot();
  const std::filesystem::path output =
      std::filesystem::temp_directory_path() / "meridian_apps_quad_easy_visual_graph_disabled";
  std::error_code ignored;
  std::filesystem::remove_all(output, ignored);

  BagLocalizationOptions options;
  options.bag_uri = root / "bags/newer-college/quad-easy";
  options.calibration_root = root / "bags/newer-college/calib";
  options.collection = ros::NewerCollegeCollection::Collection1;
  options.output_directory = output;
  options.maximum_events = 1'700U;
  options.maximum_bag_messages = 5'500U;
  options.sensor_mode = BagLocalizationSensorMode::Full;
  options.visual_graph_enabled = false;
  useUnpacedReplay(&options);

  const BagLocalizationOutcome outcome = runBagLocalization(options);

  ASSERT_TRUE(outcome.execution_succeeded) << outcome.error;
  EXPECT_TRUE(outcome.localization_available);
  ASSERT_TRUE(std::filesystem::exists(outcome.run_report));
  const std::string report = readFile(outcome.run_report);
  EXPECT_NE(report.find("\"schema_version\": 27"), std::string::npos);
  EXPECT_NE(report.find("\"schema\": \"meridian_navigation_diagnostics_csv_v19\""),
            std::string::npos);
  EXPECT_NE(report.find("\"visual_graph\": \"disabled\""), std::string::npos);
  EXPECT_NE(report.find("\"graph_submission_enabled\": false"), std::string::npos);
  EXPECT_NE(report.find("\"tracking_only_graph_submission_disabled\":"), std::string::npos);
  EXPECT_NE(report.find("\"camera_ingest_rejected\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"keyframe_requests\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"pending_camera_knots_at_end\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"visual_graph_attachments\": 0"), std::string::npos);
}

TEST(BagLocalize, LidarImuModeConfiguresAndProcessesNoCameraFrontend) {
  const std::filesystem::path root = repositoryRoot();
  const std::filesystem::path output =
      std::filesystem::temp_directory_path() / "meridian_apps_quad_easy_lidar_imu";
  std::error_code ignored;
  std::filesystem::remove_all(output, ignored);

  BagLocalizationOptions options;
  options.bag_uri = root / "bags/newer-college/quad-easy";
  options.calibration_root = root / "bags/newer-college/calib";
  options.collection = ros::NewerCollegeCollection::Collection1;
  options.output_directory = output;
  options.maximum_events = 550U;
  options.maximum_bag_messages = 5'500U;
  options.sensor_mode = BagLocalizationSensorMode::LidarImu;
  useUnpacedReplay(&options);

  const BagLocalizationOutcome outcome = runBagLocalization(options);

  ASSERT_TRUE(outcome.execution_succeeded) << outcome.error;
  EXPECT_TRUE(outcome.localization_available);
  EXPECT_EQ(outcome.suggested_exit_code, 0);
  ASSERT_TRUE(std::filesystem::exists(outcome.run_report));
  const std::string report = readFile(outcome.run_report);
  EXPECT_NE(report.find("\"schema_version\": 27"), std::string::npos);
  const std::optional<double> absolute_rank_tolerance =
      jsonDoubleField(report, "hessian_absolute_rank_tolerance");
  const std::optional<double> relative_rank_tolerance =
      jsonDoubleField(report, "hessian_relative_rank_tolerance");
  ASSERT_TRUE(absolute_rank_tolerance);
  ASSERT_TRUE(relative_rank_tolerance);
  EXPECT_DOUBLE_EQ(*absolute_rank_tolerance, 1.0e-11);
  EXPECT_DOUBLE_EQ(*relative_rank_tolerance, 1.0e-11);
  // Geometry-preserving LiDAR information retains the rotational lever arm;
  // this bounded moving batch no longer needs an accelerometer-tilt prior to
  // manufacture an otherwise clipped direction.
  EXPECT_NE(report.find("\"prior_resolved_accel_tilt_modes\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"observability_class\": \"sensor_observable\""), std::string::npos);
  EXPECT_NE(report.find("\"sensor_mode\": \"lidar-imu\""), std::string::npos);
  EXPECT_NE(report.find("\"visual_graph\": \"not_applicable\""), std::string::npos);
  EXPECT_NE(report.find("\"configured\": {\"imu_topics\": 1, \"lidar_topics\": 1, "
                        "\"camera_topics\": 0, \"gnss_topics\": 0, \"visual_lanes\": 0}"),
            std::string::npos);
  EXPECT_NE(report.find("\"camera_topics_with_stats\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"camera_messages_seen\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"camera_image_decode_errors\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"camera_events_emitted\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"camera_domain_events\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"camera_ingest_accepted\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"camera_ingest_rejected\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"configuration_matches\": true, "
                        "\"runtime_isolation_matches\": true, \"passed\": true"),
            std::string::npos);
  EXPECT_NE(report.find("\"visual_cameras\": []"), std::string::npos);
  const std::optional<std::size_t> stationary_samples =
      pipelineStageTotalSamples(report, "local.initialization.stationary_probe");
  ASSERT_TRUE(stationary_samples);
  EXPECT_EQ(*stationary_samples, 0U);
  for (const std::string_view stage :
       {"local.initialization.bootstrap_acquisition_deskew", "local.lidar.registration_view_build",
        "local.lidar.target_build_update", "local.lidar.correspondence_registration_solve",
        "local.lidar.finalized_target_update", "local.lidar.composite_target_index_build",
        "local.lidar.factor_batch_prepare", "local.initialization.motion_batch_solve_refinement",
        "local.tracking.deskew", "local.graph.transaction_update"}) {
    const std::optional<std::size_t> samples = pipelineStageTotalSamples(report, stage);
    ASSERT_TRUE(samples) << stage;
    EXPECT_GT(*samples, 0U) << stage;
  }
  const std::optional<std::size_t> bag_messages_read = jsonSizeField(report, "bag_messages_read");
  const std::optional<std::size_t> configured_messages_seen =
      jsonSizeField(report, "configured_messages_seen");
  ASSERT_TRUE(bag_messages_read);
  ASSERT_TRUE(configured_messages_seen);
  EXPECT_EQ(*bag_messages_read, *configured_messages_seen);
  EXPECT_NE(report.find("\"unknown_topics\": {}"), std::string::npos);
}

TEST(BagLocalize, ImuModeIsIsolatedAndHonestlyReportsUnavailableMovingStart) {
  const std::filesystem::path root = repositoryRoot();
  const std::filesystem::path output =
      std::filesystem::temp_directory_path() / "meridian_apps_quad_easy_imu";
  std::error_code ignored;
  std::filesystem::remove_all(output, ignored);

  BagLocalizationOptions options;
  options.bag_uri = root / "bags/newer-college/quad-easy";
  options.calibration_root = root / "bags/newer-college/calib";
  options.collection = ros::NewerCollegeCollection::Collection1;
  options.output_directory = output;
  options.maximum_events = 100U;
  options.maximum_bag_messages = 1'000U;
  options.sensor_mode = BagLocalizationSensorMode::Imu;
  useUnpacedReplay(&options);

  const BagLocalizationOutcome outcome = runBagLocalization(options);

  ASSERT_TRUE(outcome.execution_succeeded) << outcome.error;
  EXPECT_FALSE(outcome.localization_available);
  EXPECT_EQ(outcome.suggested_exit_code, 5);
  EXPECT_EQ(outcome.states_written, 0U);
  ASSERT_TRUE(std::filesystem::exists(outcome.run_report));
  const std::string report = readFile(outcome.run_report);
  EXPECT_NE(report.find("\"sensor_mode\": \"imu\""), std::string::npos);
  EXPECT_NE(report.find("\"visual_graph\": \"not_applicable\""), std::string::npos);
  EXPECT_NE(report.find("\"configured\": {\"imu_topics\": 1, \"lidar_topics\": 0, "
                        "\"camera_topics\": 0, \"gnss_topics\": 0, \"visual_lanes\": 0}"),
            std::string::npos);
  EXPECT_NE(report.find("\"camera_domain_events\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"lidar_topics_with_stats\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"lidar_messages_seen\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"lidar_events_emitted\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"lidar_domain_events\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"lidar_enqueue_accepted\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"lidar_enqueue_rejected\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"storage_filter_matches\": true"), std::string::npos);
  EXPECT_NE(report.find("\"configuration_matches\": true, "
                        "\"runtime_isolation_matches\": true, \"passed\": true"),
            std::string::npos);
  EXPECT_NE(report.find("\"stage\": \"coverage\""), std::string::npos);
  const std::optional<std::size_t> bag_messages_read = jsonSizeField(report, "bag_messages_read");
  const std::optional<std::size_t> configured_messages_seen =
      jsonSizeField(report, "configured_messages_seen");
  ASSERT_TRUE(bag_messages_read);
  ASSERT_TRUE(configured_messages_seen);
  EXPECT_EQ(*bag_messages_read, *configured_messages_seen);
  EXPECT_NE(report.find("\"unknown_topics\": {}"), std::string::npos);
}

TEST(BagLocalize, ScheduledImuAuditIsPacedLossFreeAndReportsContinuity) {
  const std::filesystem::path root = repositoryRoot();
  const std::filesystem::path output =
      std::filesystem::temp_directory_path() / "meridian_apps_scheduled_imu_audit";
  std::error_code ignored;
  std::filesystem::remove_all(output, ignored);

  BagLocalizationOptions options;
  options.bag_uri = root / "bags/newer-college/quad-easy";
  options.calibration_root = root / "bags/newer-college/calib";
  options.collection = ros::NewerCollegeCollection::Collection1;
  options.output_directory = output;
  options.maximum_events = 3U;
  options.maximum_bag_messages = 10U;
  options.sensor_mode = BagLocalizationSensorMode::Imu;
  options.replay.playback_rate = 0.1;
  options.replay.queue_count_capacity = 8U;
  options.replay.queue_byte_capacity = 1U * 1024U * 1024U;
  options.replay.timing_window_capacity = 8U;

  const BagLocalizationOutcome outcome = runBagLocalization(options);

  ASSERT_TRUE(outcome.execution_succeeded) << outcome.error;
  EXPECT_FALSE(outcome.localization_available);
  EXPECT_EQ(outcome.suggested_exit_code, 5);
  ASSERT_TRUE(outcome.runtime);
  EXPECT_EQ(outcome.runtime->replay_mode, BagLocalizationReplayMode::Scheduled);
  EXPECT_EQ(outcome.runtime->queue_terminal_count, 0U);
  EXPECT_EQ(outcome.runtime->queue_terminal_bytes, 0U);
  EXPECT_EQ(outcome.runtime->queue_rejected, 0U);
  EXPECT_EQ(outcome.runtime->queue_dropped, 0U);
  EXPECT_EQ(outcome.runtime->queue_skipped, 0U);
  ASSERT_TRUE(outcome.runtime->producer_total_wall_ns);
  EXPECT_GT(*outcome.runtime->producer_total_wall_ns, 20'000'000LL);
  EXPECT_TRUE(outcome.runtime->producer_p95_wall_ns);
  EXPECT_TRUE(outcome.runtime->consumer_p95_wall_ns);
  EXPECT_TRUE(outcome.runtime->process_ready_p95_wall_ns);
  EXPECT_TRUE(outcome.runtime->report_output_p95_wall_ns);

  const std::string report = readFile(outcome.run_report);
  EXPECT_NE(report.find("\"schema_version\": 27"), std::string::npos);
  EXPECT_NE(report.find("\"mode\": \"scheduled\""), std::string::npos);
  EXPECT_NE(report.find("\"queue_count_capacity\": 8"), std::string::npos);
  EXPECT_NE(report.find("\"queue_byte_capacity\": 1048576"), std::string::npos);
  EXPECT_NE(report.find("\"timing_window_capacity\": 8"), std::string::npos);
  EXPECT_NE(report.find("\"local_stages\": {\"schema_version\": 1, \"window_capacity\": 8, "
                        "\"span_semantics\": \"inclusive\""),
            std::string::npos);
  EXPECT_NE(report.find("\"name\": \"scheduled_replay.events\""), std::string::npos);
  EXPECT_NE(report.find("\"accepted\": 3, \"rejected\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"dropped_oldest\": 0, \"dropped_newest\": 0, "
                        "\"skipped_stale\": 0, \"skipped_policy\": 0"),
            std::string::npos);
  EXPECT_NE(report.find("\"replay.producer.total\""), std::string::npos);
  EXPECT_NE(report.find("\"replay.consumer.total\""), std::string::npos);
  EXPECT_NE(report.find("\"replay.producer.schedule_enqueue\""), std::string::npos);
  EXPECT_NE(report.find("\"replay.consumer.visit\""), std::string::npos);
  EXPECT_NE(report.find("\"retained_samples_high_watermark\": 3"), std::string::npos);
  EXPECT_NE(report.find("\"newest_gap_maximum_observed_ns\":"), std::string::npos);
  EXPECT_NE(report.find("\"newest_gap\": {\"total_samples\": 2"), std::string::npos);
  EXPECT_NE(report.find("\"thread_cpu\": {\"total_samples\": 3"), std::string::npos);
}

TEST(BagLocalize, ScheduledQueueOverloadFailsAndRetainsTerminalReport) {
  const std::filesystem::path root = repositoryRoot();
  const std::filesystem::path output =
      std::filesystem::temp_directory_path() / "meridian_apps_scheduled_queue_overload";
  std::error_code ignored;
  std::filesystem::remove_all(output, ignored);

  BagLocalizationOptions options;
  options.bag_uri = root / "bags/newer-college/quad-easy";
  options.calibration_root = root / "bags/newer-college/calib";
  options.collection = ros::NewerCollegeCollection::Collection1;
  options.output_directory = output;
  options.maximum_events = 2U;
  options.maximum_bag_messages = 10U;
  options.sensor_mode = BagLocalizationSensorMode::Imu;
  options.replay.playback_rate = 1.0e9;
  options.replay.queue_count_capacity = 1U;
  options.replay.queue_byte_capacity = 1U;
  options.replay.timing_window_capacity = 4U;

  const BagLocalizationOutcome outcome = runBagLocalization(options);

  EXPECT_FALSE(outcome.execution_succeeded);
  EXPECT_FALSE(outcome.localization_available);
  EXPECT_EQ(outcome.suggested_exit_code, 4);
  ASSERT_TRUE(outcome.runtime);
  EXPECT_EQ(outcome.runtime->replay_mode, BagLocalizationReplayMode::Scheduled);
  EXPECT_EQ(outcome.runtime->queue_terminal_count, 0U);
  EXPECT_EQ(outcome.runtime->queue_terminal_bytes, 0U);
  EXPECT_EQ(outcome.runtime->queue_rejected, 1U);
  EXPECT_EQ(outcome.runtime->queue_dropped, 0U);
  EXPECT_EQ(outcome.runtime->queue_skipped, 0U);
  EXPECT_NE(outcome.error.find("byte capacity exceeded"), std::string::npos);

  const std::string report = readFile(outcome.run_report);
  EXPECT_NE(report.find("\"execution_succeeded\": false"), std::string::npos);
  EXPECT_NE(report.find("\"stage\": \"replay\""), std::string::npos);
  EXPECT_NE(report.find("\"name\": \"scheduled_replay.events\""), std::string::npos);
  EXPECT_NE(report.find("\"accepted\": 0, \"rejected\": 1"), std::string::npos);
  EXPECT_NE(report.find("\"maximum_count\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"maximum_bytes\": 0"), std::string::npos);
  EXPECT_NE(report.find("\"driver_stages\": {"), std::string::npos);
}

TEST(BagLocalize, RejectsExplicitVisualGraphSelectionOutsideFullMode) {
  const std::filesystem::path output =
      std::filesystem::temp_directory_path() / "meridian_apps_invalid_visual_mode";
  std::error_code ignored;
  std::filesystem::remove_all(output, ignored);

  BagLocalizationOptions options;
  options.output_directory = output;
  options.maximum_events = 1U;
  options.maximum_bag_messages = 1U;
  options.sensor_mode = BagLocalizationSensorMode::LidarImu;
  options.visual_graph_enabled = false;
  useUnpacedReplay(&options);

  const BagLocalizationOutcome outcome = runBagLocalization(options);

  EXPECT_FALSE(outcome.execution_succeeded);
  EXPECT_FALSE(outcome.localization_available);
  EXPECT_EQ(outcome.suggested_exit_code, 2);
  ASSERT_TRUE(std::filesystem::exists(outcome.run_report));
  const std::string report = readFile(outcome.run_report);
  EXPECT_NE(report.find("\"schema_version\": 27"), std::string::npos);
  EXPECT_NE(report.find("\"sensor_mode\": \"lidar-imu\""), std::string::npos);
  EXPECT_NE(report.find("\"stage\": \"options\""), std::string::npos);
  EXPECT_NE(report.find("visual graph selection is valid only in full sensor mode"),
            std::string::npos);
}

}  // namespace
}  // namespace meridian::apps
