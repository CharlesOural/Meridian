#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

#include "identity_transaction.hpp"
#include "meridian/local/local_estimator.hpp"

namespace meridian::local {
namespace {

constexpr std::int64_t kMillisecond = 1'000'000LL;
constexpr std::int64_t kImuPeriod = 5 * kMillisecond;

[[nodiscard]] core::CalibrationBundle calibration() {
  core::ImuNoiseModel noise(1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6);
  core::ImuCalibration imu("imu", "/imu", core::ImuSensorModel::Generic, 200.0, 9.80665, noise);
  core::LidarCalibration lidar(
      core::LidarId{1U}, "lidar", "/points", core::ImuFromLidarTransform{core::Pose3d{}},
      core::LidarTimingCalibration{core::LidarSweepTimestampReference::SweepStart,
                                   core::LidarPointTimeConvention::OffsetFromSweepTimestamp});
  auto result = core::CalibrationBundle::create(core::CalibrationEpoch{1U}, std::move(imu),
                                                core::BaseFromImuTransform{core::Pose3d{}},
                                                std::move(lidar), {});
  EXPECT_TRUE(result);
  return std::move(result).value();
}

[[nodiscard]] core::RecordHeader header() {
  core::RecordHeader result;
  result.trace = core::TraceId{1U};
  result.producer = core::ProducerId{1U};
  result.session = core::SessionId{1U};
  result.config = core::ConfigRevision{1U};
  result.direct_calibration = core::CalibrationEpoch{1U};
  return result;
}

[[nodiscard]] core::SourceStamp stamp(std::int64_t time_ns, std::uint64_t sequence) {
  core::SourceStamp result;
  result.raw_time = core::RawDeviceTime{time_ns};
  result.fusion_time = core::FusionTime{time_ns};
  result.host_arrival_time = core::ArrivalTime{time_ns};
  result.clock_revision = core::ClockRevision{1U};
  result.source_epoch = core::SourceEpoch{1U};
  result.ingress_sequence = core::IngressSequence{sequence};
  return result;
}

[[nodiscard]] core::ImuSample stationaryImu(std::int64_t time_ns, std::uint64_t id) {
  core::ImuSample sample;
  sample.header = header();
  sample.id = core::MeasurementId{id};
  sample.stamp = stamp(time_ns, id);
  sample.specific_force_mps2 = Eigen::Vector3d{0.0, 0.0, 9.80665};
  return sample;
}

[[nodiscard]] core::LidarSweep sweep(std::uint64_t id, std::int64_t start_ns) {
  constexpr std::uint32_t kWidth = 64U;
  constexpr std::uint32_t kRowsPerPlane = 8U;
  constexpr std::uint32_t kHeight = 3U * kRowsPerPlane;
  constexpr std::int64_t kDuration = 90 * kMillisecond;
  auto points = std::make_shared<core::LidarPoints>();
  points->reserve(static_cast<std::size_t>(kWidth) * kHeight);
  for (std::uint32_t row = 0U; row < kHeight; ++row) {
    const std::uint32_t plane = row / kRowsPerPlane;
    const std::uint32_t plane_row = row % kRowsPerPlane;
    const double narrow_axis = -1.38 + 0.09 * static_cast<double>(plane_row);
    for (std::uint32_t column = 0U; column < kWidth; ++column) {
      const double long_axis = 4.13 + 0.09 * static_cast<double>(column);
      Eigen::Vector3d position;
      if (plane == 0U) {
        position = Eigen::Vector3d{long_axis, narrow_axis, -1.73};
      } else if (plane == 1U) {
        position = Eigen::Vector3d{long_axis, 3.37, narrow_axis + 1.02};
      } else {
        position = Eigen::Vector3d{9.41, long_axis - 6.92, narrow_axis + 0.51};
      }
      core::LidarPoint point;
      point.x = static_cast<float>(position.x());
      point.y = static_cast<float>(position.y());
      point.z = static_cast<float>(position.z());
      point.intensity = static_cast<float>(column + row);
      point.ring = static_cast<std::uint16_t>(row);
      point.source_index = row * kWidth + column;
      point.time_offset_ns =
          static_cast<std::int32_t>((static_cast<std::int64_t>(column) * (kDuration - 1)) /
                                    static_cast<std::int64_t>(kWidth - 1U));
      points->push_back(point);
    }
  }

  core::LidarSweep result;
  result.header = header();
  result.id = core::MeasurementId{id};
  result.lidar = core::LidarId{1U};
  result.stamp = stamp(start_ns, id);
  result.acquisition =
      core::TimeRange{core::FusionTime{start_ns}, core::FusionTime{start_ns + kDuration}};
  result.layout = core::LidarLayout{kWidth, kHeight, true};
  result.points = std::move(points);
  return result;
}

[[nodiscard]] core::LidarSweep verticallyShiftedSweep(std::uint64_t id, std::int64_t start_ns,
                                                      double offset_m) {
  core::LidarSweep result = sweep(id, start_ns);
  auto shifted = std::make_shared<core::LidarPoints>(*result.points);
  for (core::LidarPoint& point : *shifted) {
    point.z += static_cast<float>(offset_m);
  }
  result.points = std::move(shifted);
  return result;
}

[[nodiscard]] core::LidarSweep halfVerticallyShiftedSweep(std::uint64_t id, std::int64_t start_ns,
                                                          double offset_m) {
  core::LidarSweep result = sweep(id, start_ns);
  auto shifted = std::make_shared<core::LidarPoints>(*result.points);
  for (core::LidarPoint& point : *shifted) {
    if ((point.source_index % result.layout.width) >= result.layout.width / 2U) {
      point.z += static_cast<float>(offset_m);
    }
  }
  result.points = std::move(shifted);
  return result;
}

[[nodiscard]] core::LidarSweep deskewInvalidSweep(std::uint64_t id, std::int64_t start_ns) {
  core::LidarSweep result = sweep(id, start_ns);
  auto points = std::make_shared<core::LidarPoints>(*result.points);
  points->front().time_offset_ns = static_cast<std::int32_t>(
      (result.acquisition.end - result.acquisition.start).nanoseconds + 1LL);
  result.points = std::move(points);
  return result;
}

struct GraphLineageIdentities {
  std::uint64_t lineage{};
  std::uint64_t consumer{};
  std::uint64_t factor_group{};
};

struct CompleteLineageIdentities {
  std::uint64_t lineage{};
  std::uint64_t consumer{};
  std::uint64_t factor_group{};
  std::uint64_t correlation_group{};

  friend bool operator==(const CompleteLineageIdentities&,
                         const CompleteLineageIdentities&) = default;
};

struct BootstrapRecoverySnapshot {
  core::StateId initialization_state;
  CompleteLineageIdentities lidar;
  CompleteLineageIdentities imu;
  CompleteLineageIdentities initialization;

  friend bool operator==(const BootstrapRecoverySnapshot&,
                         const BootstrapRecoverySnapshot&) = default;
};

[[nodiscard]] CompleteLineageIdentities completeLineageIdentities(
    const core::ObservationLineage& lineage) {
  CompleteLineageIdentities result;
  result.lineage = lineage.id.value();
  if (!lineage.usage.empty()) {
    result.consumer = lineage.usage.front().consumer.value();
  }
  for (const core::ObservationUsage& usage : lineage.usage) {
    if (usage.factor_group && result.factor_group == 0U) {
      result.factor_group = usage.factor_group->value();
    }
    if (usage.correlation_group && result.correlation_group == 0U) {
      result.correlation_group = usage.correlation_group->value();
    }
  }
  return result;
}

[[nodiscard]] GraphLineageIdentities graphLineageIdentities(
    const core::ObservationLineage& lineage) {
  EXPECT_FALSE(lineage.usage.empty());
  if (lineage.usage.empty()) {
    return {};
  }
  EXPECT_TRUE(lineage.usage.front().factor_group);
  return GraphLineageIdentities{
      lineage.id.value(), lineage.usage.front().consumer.value(),
      lineage.usage.front().factor_group ? lineage.usage.front().factor_group->value() : 0U};
}

[[nodiscard]] std::set<std::uint64_t> measurementRootsWithRole(
    const core::ObservationLineage& lineage, core::ObservationRole role) {
  std::set<std::uint64_t> roots;
  for (const core::ObservationUsage& usage : lineage.usage) {
    if (usage.role != role) {
      continue;
    }
    if (const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root)) {
      roots.insert(measurement->value());
    }
  }
  return roots;
}

TEST(LocalEstimatorIdentityTransaction,
     DiscardLeavesLiveCountersAndMapsUnchangedWhileCommitPublishesOnce) {
  struct IdentityMap {
    std::map<std::uint64_t, std::uint64_t> landmarks;
    std::map<std::uint64_t, std::uint64_t> factors;

    bool operator==(const IdentityMap&) const = default;
  };

  detail::IdentityCounters live_counters{11U, 21U, 31U, 41U, 51U, 61U};
  IdentityMap live_maps{{{1U, 101U}}, {{2U, 202U}}};
  const detail::IdentityCounters original_counters = live_counters;
  const IdentityMap original_maps = live_maps;

  {
    detail::IdentityTransaction<IdentityMap> discarded(live_counters, live_maps);
    ++discarded.counters().next_lineage;
    ++discarded.counters().next_visual_factor;
    discarded.state().landmarks.emplace(3U, 303U);
    discarded.state().factors.emplace(4U, 404U);
  }
  EXPECT_EQ(live_counters, original_counters);
  EXPECT_EQ(live_maps, original_maps);

  detail::IdentityTransaction<IdentityMap> accepted(live_counters, live_maps);
  ++accepted.counters().next_lineage;
  ++accepted.counters().next_visual_factor;
  accepted.state().landmarks.emplace(3U, 303U);
  accepted.state().factors.emplace(4U, 404U);
  auto committed = accepted.commit();
  ASSERT_TRUE(committed);
  live_counters = committed.value().counters;
  live_maps = std::move(committed).value().state;
  const auto second_commit = accepted.commit();
  ASSERT_FALSE(second_commit);
  EXPECT_EQ(second_commit.error().code, detail::IdentityTransactionErrorCode::TransactionConsumed);

  EXPECT_EQ(live_counters.next_lineage, original_counters.next_lineage + 1U);
  EXPECT_EQ(live_counters.next_visual_factor, original_counters.next_visual_factor + 1U);
  EXPECT_EQ(live_maps.landmarks.size(), original_maps.landmarks.size() + 1U);
  EXPECT_EQ(live_maps.factors.size(), original_maps.factors.size() + 1U);
}

TEST(LocalEstimatorIdentityTransaction, IsMoveOnlyAndRejectsMovedFromOrDoubleConsumption) {
  using Transaction = detail::IdentityTransaction<std::vector<std::uint64_t>>;
  static_assert(!std::is_copy_constructible_v<Transaction>);
  static_assert(!std::is_copy_assignable_v<Transaction>);
  static_assert(std::is_nothrow_move_constructible_v<Transaction>);
  static_assert(std::is_nothrow_move_assignable_v<Transaction>);

  const detail::IdentityCounters checkpoint{11U, 21U, 31U, 41U, 51U, 61U};
  Transaction original(checkpoint, std::vector<std::uint64_t>{1U});
  ++original.counters().next_lineage;
  original.state().push_back(2U);
  Transaction moved(std::move(original));
  EXPECT_TRUE(original.consumed());

  const auto moved_from_commit = original.commit();
  ASSERT_FALSE(moved_from_commit);
  EXPECT_EQ(moved_from_commit.error().code,
            detail::IdentityTransactionErrorCode::TransactionConsumed);

  const auto aborted = moved.abort();
  ASSERT_TRUE(aborted);
  EXPECT_EQ(aborted.value(), checkpoint);
  const auto second_abort = moved.abort();
  ASSERT_FALSE(second_abort);
  EXPECT_EQ(second_abort.error().code, detail::IdentityTransactionErrorCode::TransactionConsumed);
  const auto abort_then_commit = moved.commit();
  ASSERT_FALSE(abort_then_commit);
  EXPECT_EQ(abort_then_commit.error().code,
            detail::IdentityTransactionErrorCode::TransactionConsumed);

  Transaction assignment_source(checkpoint, std::vector<std::uint64_t>{3U});
  ++assignment_source.counters().next_derived;
  Transaction assignment_target(checkpoint, std::vector<std::uint64_t>{99U});
  assignment_target = std::move(assignment_source);
  EXPECT_TRUE(assignment_source.consumed());
  const auto move_assigned_from_abort = assignment_source.abort();
  ASSERT_FALSE(move_assigned_from_abort);
  EXPECT_EQ(move_assigned_from_abort.error().code,
            detail::IdentityTransactionErrorCode::TransactionConsumed);
  auto move_assignment_commit = assignment_target.commit();
  ASSERT_TRUE(move_assignment_commit);
  EXPECT_EQ(move_assignment_commit.value().counters.next_derived, checkpoint.next_derived + 1U);
  EXPECT_EQ(move_assignment_commit.value().state, (std::vector<std::uint64_t>{3U}));
}

[[nodiscard]] LocalEstimatorConfig estimatorConfig() {
  LocalEstimatorConfig config;
  config.odom_epoch = core::OdomEpoch{3U};
  config.first_state = core::StateId{10U};
  config.initialization.mode = InitializationMode::SupervisedAuto;
  config.initialization.zero_motion_prior =
      ZeroMotionPrior{config.odom_epoch, ZeroMotionPriorSource::MissionScenario};
  config.stationary_initializer.minimum_support = core::Duration{2'000 * kMillisecond};
  config.stationary_retry_period = core::Duration{50 * kMillisecond};
  config.lidar_preprocessing.minimum_range_m = 0.5;
  config.lidar_preprocessing.maximum_range_m = 30.0;
  config.lidar_preprocessing.voxel_size_m = 0.1;
  config.lidar_preprocessing.maximum_output_points = 2'000U;
  config.rolling_target.maximum_retained_sweeps = 4U;
  config.rolling_target.maximum_retained_points = 10'000U;
  config.lidar_registration.target_voxel_resolution_m = 0.5;
  config.lidar_registration.source_voxel_size_m = 0.5;
  config.lidar_registration.maximum_target_points_per_target = 10'000U;
  config.lidar_registration.minimum_correspondences = 20U;
  config.lidar_registration.residual_standard_deviation_m = 0.05;
  config.lidar_registration.absolute_normalized_observable_eigenvalue = 1.0e-8;
  config.finalized_lidar_target.query_voxel_size_m =
      config.lidar_registration.target_voxel_resolution_m;
  config.finalized_lidar_target.maximum_supported_query_distance_m =
      config.lidar_registration.maximum_correspondence_distance_m;
  return config;
}

void appendImuThrough(LocalEstimator& estimator, std::int64_t begin_ns, std::int64_t end_ns,
                      std::uint64_t& next_id) {
  for (std::int64_t time = begin_ns; time <= end_ns; time += kImuPeriod) {
    auto ingested = estimator.ingestImu(stationaryImu(time, next_id++));
    ASSERT_TRUE(ingested) << ingested.error().detail;
  }
}

TEST(LocalEstimatorConfig, TrackingAcceptsLowRankWhileBootstrapRetainsStricterProfile) {
  const LocalEstimatorConfig config;
  EXPECT_EQ(config.initialization.mode, InitializationMode::DynamicOnly);
  EXPECT_FALSE(config.initialization.zero_motion_prior);
  EXPECT_EQ(config.graph.maximum_navigation_states, 64U);
  EXPECT_EQ(config.state_timeline.maximum_navigation_states, 65U);
  EXPECT_EQ(config.lidar_registration.minimum_observable_rank, 1U);
  EXPECT_EQ(config.lidar_bootstrap.registration.minimum_observable_rank, 1U);
  EXPECT_EQ(config.minimum_lidar_factor_interval, core::Duration{});
  EXPECT_DOUBLE_EQ(config.lidar_target_reuse_covariance_inflation, 1.5);
  EXPECT_DOUBLE_EQ(config.lidar_imu_conditioning_covariance_inflation, 4.0);
  EXPECT_DOUBLE_EQ(config.finalized_map_correlation_inflation_floor, 1.0);
}

TEST(LocalEstimatorConfig, RejectsUnityOrNonFiniteTrackingLidarCorrelationPolicy) {
  LocalEstimatorConfig unity = estimatorConfig();
  unity.lidar_target_reuse_covariance_inflation = 1.0;
  unity.lidar_imu_conditioning_covariance_inflation = 1.0;
  auto unity_result = LocalEstimator::create(calibration(), unity);
  ASSERT_FALSE(unity_result);
  EXPECT_EQ(unity_result.error().code, LocalEstimatorErrorCode::InvalidConfiguration);

  LocalEstimatorConfig non_finite = estimatorConfig();
  non_finite.lidar_imu_conditioning_covariance_inflation = std::numeric_limits<double>::infinity();
  auto non_finite_result = LocalEstimator::create(calibration(), non_finite);
  ASSERT_FALSE(non_finite_result);
  EXPECT_EQ(non_finite_result.error().code, LocalEstimatorErrorCode::InvalidConfiguration);

  LocalEstimatorConfig zero_map_floor = estimatorConfig();
  zero_map_floor.finalized_map_correlation_inflation_floor = 0.0;
  auto zero_map_floor_result = LocalEstimator::create(calibration(), zero_map_floor);
  ASSERT_FALSE(zero_map_floor_result);
  EXPECT_EQ(zero_map_floor_result.error().code, LocalEstimatorErrorCode::InvalidConfiguration);

  LocalEstimatorConfig negative_map_floor = estimatorConfig();
  negative_map_floor.finalized_map_correlation_inflation_floor = -1.0;
  auto negative_map_floor_result = LocalEstimator::create(calibration(), negative_map_floor);
  ASSERT_FALSE(negative_map_floor_result);
  EXPECT_EQ(negative_map_floor_result.error().code,
            LocalEstimatorErrorCode::InvalidConfiguration);

  LocalEstimatorConfig non_finite_map_floor = estimatorConfig();
  non_finite_map_floor.finalized_map_correlation_inflation_floor =
      std::numeric_limits<double>::infinity();
  auto non_finite_map_floor_result =
      LocalEstimator::create(calibration(), non_finite_map_floor);
  ASSERT_FALSE(non_finite_map_floor_result);
  EXPECT_EQ(non_finite_map_floor_result.error().code,
            LocalEstimatorErrorCode::InvalidConfiguration);

  LocalEstimatorConfig nan_map_floor = estimatorConfig();
  nan_map_floor.finalized_map_correlation_inflation_floor =
      std::numeric_limits<double>::quiet_NaN();
  auto nan_map_floor_result = LocalEstimator::create(calibration(), nan_map_floor);
  ASSERT_FALSE(nan_map_floor_result);
  EXPECT_EQ(nan_map_floor_result.error().code, LocalEstimatorErrorCode::InvalidConfiguration);

  LocalEstimatorConfig positive_subunity_map_floor = estimatorConfig();
  positive_subunity_map_floor.finalized_map_correlation_inflation_floor = 0.5;
  EXPECT_TRUE(LocalEstimator::create(calibration(), positive_subunity_map_floor));
}

TEST(LocalEstimatorConfig, RejectsNegativeLidarFactorInterval) {
  LocalEstimatorConfig config = estimatorConfig();
  config.minimum_lidar_factor_interval = core::Duration{-1LL};
  const auto created = LocalEstimator::create(calibration(), config);
  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().code, LocalEstimatorErrorCode::InvalidConfiguration);
}

TEST(LocalEstimatorConfig, RejectsInvalidFinalizedTargetBoundsAndRegistrationMismatch) {
  LocalEstimatorConfig zero_capacity = estimatorConfig();
  zero_capacity.maximum_pending_finalized_lidar_sweeps = 0U;
  EXPECT_FALSE(LocalEstimator::create(calibration(), zero_capacity));

  LocalEstimatorConfig zero_prune_interval = estimatorConfig();
  zero_prune_interval.finalized_lidar_prune_interval_sweeps = 0U;
  EXPECT_FALSE(LocalEstimator::create(calibration(), zero_prune_interval));

  LocalEstimatorConfig mismatched_query = estimatorConfig();
  mismatched_query.finalized_lidar_target.query_voxel_size_m *= 2.0;
  EXPECT_FALSE(LocalEstimator::create(calibration(), mismatched_query));

  LocalEstimatorConfig mismatched_gate = estimatorConfig();
  mismatched_gate.finalized_lidar_target.maximum_supported_query_distance_m *= 0.5;
  EXPECT_FALSE(LocalEstimator::create(calibration(), mismatched_gate));
}

TEST(LocalEstimatorConfig, RejectsTimelineWithoutOnePrecommitSlotBeyondGraphCapacity) {
  LocalEstimatorConfig equal_capacity = estimatorConfig();
  equal_capacity.graph.maximum_navigation_states = 8U;
  equal_capacity.state_timeline.maximum_navigation_states = 8U;
  const auto equal_result = LocalEstimator::create(calibration(), equal_capacity);
  ASSERT_FALSE(equal_result);
  EXPECT_EQ(equal_result.error().code, LocalEstimatorErrorCode::InvalidConfiguration);

  LocalEstimatorConfig overflow_capacity = estimatorConfig();
  overflow_capacity.graph.maximum_navigation_states = std::numeric_limits<std::size_t>::max();
  overflow_capacity.state_timeline.maximum_navigation_states =
      std::numeric_limits<std::size_t>::max();
  const auto overflow_result = LocalEstimator::create(calibration(), overflow_capacity);
  ASSERT_FALSE(overflow_result);
  EXPECT_EQ(overflow_result.error().code, LocalEstimatorErrorCode::InvalidConfiguration);
}

TEST(LocalEstimatorConfig, StaticInitializationRequiresMatchingExplicitZeroMotionPrior) {
  LocalEstimatorConfig missing = estimatorConfig();
  missing.initialization.mode = InitializationMode::StaticOnly;
  missing.initialization.zero_motion_prior.reset();
  auto missing_result = LocalEstimator::create(calibration(), missing);
  ASSERT_FALSE(missing_result);
  EXPECT_EQ(missing_result.error().code, LocalEstimatorErrorCode::InvalidConfiguration);

  LocalEstimatorConfig mismatched = estimatorConfig();
  mismatched.initialization.mode = InitializationMode::StaticOnly;
  mismatched.initialization.zero_motion_prior =
      ZeroMotionPrior{core::OdomEpoch{99U}, ZeroMotionPriorSource::VehicleSupervisor};
  auto mismatched_result = LocalEstimator::create(calibration(), mismatched);
  ASSERT_FALSE(mismatched_result);
  EXPECT_EQ(mismatched_result.error().code, LocalEstimatorErrorCode::InvalidConfiguration);

  LocalEstimatorConfig forbidden = estimatorConfig();
  forbidden.initialization.mode = InitializationMode::DynamicOnly;
  auto forbidden_result = LocalEstimator::create(calibration(), forbidden);
  ASSERT_FALSE(forbidden_result);
  EXPECT_EQ(forbidden_result.error().code, LocalEstimatorErrorCode::InvalidConfiguration);
}

TEST(LocalEstimatorInitializationPolicy, DynamicOnlyNeverInfersStaticStartup) {
  LocalEstimatorConfig config = estimatorConfig();
  config.initialization.mode = InitializationMode::DynamicOnly;
  config.initialization.zero_motion_prior.reset();
  auto created = LocalEstimator::create(calibration(), config);
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);

  const auto processed = estimator.processReady();

  ASSERT_TRUE(processed) << processed.error().detail;
  EXPECT_FALSE(processed.value().initialization);
  EXPECT_EQ(processed.value().lifecycle, LocalEstimatorLifecycle::AwaitingInitialization);
  EXPECT_EQ(estimator.statistics().stationary_initialization_attempts, 0U);
  EXPECT_EQ(estimator.statistics().motion_initialization_attempts, 0U);
}

TEST(LocalEstimator, InitializesBootstrapsAndRegistersDeterministically) {
  auto created = LocalEstimator::create(calibration(), estimatorConfig());
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);

  auto initialized = estimator.processReady();
  ASSERT_TRUE(initialized) << initialized.error().detail;
  ASSERT_TRUE(initialized.value().initialization);
  ASSERT_EQ(initialized.value().graph_transactions.size(), 1U);
  EXPECT_TRUE(initialized.value().graph_transactions.front().navigation_state_created);
  EXPECT_EQ(initialized.value().graph_transactions.front().revision,
            initialized.value().initialization->revision);
  EXPECT_EQ(initialized.value().initialization_method, LocalInitializationMethod::StationaryImu);
  EXPECT_EQ(initialized.value().initialization->state, core::StateId{10U});
  EXPECT_EQ(estimator.statistics().stationary_initialization_attempts, 1U);
  EXPECT_EQ(estimator.lifecycle(), LocalEstimatorLifecycle::Tracking);

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  auto bootstrap = estimator.processReady();
  ASSERT_TRUE(bootstrap) << bootstrap.error().detail;
  ASSERT_EQ(bootstrap.value().commits.size(), 1U);
  ASSERT_EQ(bootstrap.value().graph_transactions.size(), 1U);
  EXPECT_TRUE(bootstrap.value().graph_transactions.front().navigation_state_created);
  EXPECT_EQ(bootstrap.value().commits.front().disposition, LidarCommitDisposition::BootstrapTarget);
  EXPECT_EQ(bootstrap.value().commits.front().commit.state, core::StateId{11U});
  EXPECT_GT(bootstrap.value().commits.front().preprocessing->deterministic_voxel_points, 20U);
  ASSERT_TRUE(bootstrap.value().commits.front().map_input);
  EXPECT_EQ(bootstrap.value().commits.front().map_input->sweep.id, core::MeasurementId{10'000U});
  EXPECT_EQ(bootstrap.value().commits.front().map_input->accepted_revision,
            bootstrap.value().commits.front().commit.revision);

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'001U, 2'500 * kMillisecond)));
  appendImuThrough(estimator, 2'405 * kMillisecond, 2'600 * kMillisecond, imu_id);
  auto registered = estimator.processReady();
  ASSERT_TRUE(registered) << registered.error().detail;
  ASSERT_EQ(registered.value().commits.size(), 1U);
  ASSERT_EQ(registered.value().graph_transactions.size(), 2U);
  const LocalGraphTransactionSolveReport& navigation_transaction =
      registered.value().graph_transactions.front();
  const LocalGraphTransactionSolveReport& lidar_transaction =
      registered.value().graph_transactions.back();
  EXPECT_TRUE(navigation_transaction.navigation_state_created);
  EXPECT_FALSE(lidar_transaction.navigation_state_created);
  EXPECT_EQ(lidar_transaction.parent, navigation_transaction.revision);
  EXPECT_EQ(navigation_transaction.solve.factor_batches_added, 0U);
  EXPECT_EQ(lidar_transaction.solve.factor_batches_added, 1U);
  EXPECT_EQ(lidar_transaction.revision, registered.value().commits.front().commit.revision);
  EXPECT_EQ(registered.value().commits.front().disposition, LidarCommitDisposition::Registered);
  ASSERT_TRUE(registered.value().commits.front().registration);
  ASSERT_TRUE(registered.value().commits.front().map_input);
  EXPECT_EQ(registered.value().commits.front().map_input->sweep.id, core::MeasurementId{10'001U});
  EXPECT_GE(registered.value().commits.front().registration->diagnostics.observable_rank, 1U);
  EXPECT_GE(registered.value().commits.front().registration->diagnostics.correspondences, 20U);
  // Initialization and bootstrap each publish one navigation revision. The
  // registered sweep publishes its navigation state first, then one
  // sensor-pure LiDAR FactorBatch revision against already-live poses.
  EXPECT_EQ(estimator.statistics().graph_commits, 4U);
  EXPECT_EQ(estimator.statistics().rolling_target_pose_synchronizations, 2U);
  EXPECT_EQ(estimator.statistics().rolling_target_sweeps_synchronized, 2U);
  EXPECT_EQ(estimator.statistics().lidar_registrations, 1U);

  const auto estimate = estimator.estimate();
  ASSERT_TRUE(estimate);
  EXPECT_EQ(estimate.value().state, core::StateId{12U});
  EXPECT_NEAR(estimate.value().estimate.T_odom_imu.translation().norm(), 0.0, 1.0e-3);
}

TEST(LocalEstimator, SuccessfulNonKeyframeRegistrationMutatesNeitherGraphFactorsNorMap) {
  LocalEstimatorConfig config = estimatorConfig();
  config.minimum_lidar_factor_interval = core::Duration{300 * kMillisecond};
  auto created = LocalEstimator::create(calibration(), config);
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);

  const auto initialized = estimator.processReady();
  ASSERT_TRUE(initialized) << initialized.error().detail;
  ASSERT_TRUE(initialized.value().initialization);

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  const auto bootstrap = estimator.processReady();
  ASSERT_TRUE(bootstrap) << bootstrap.error().detail;
  ASSERT_EQ(bootstrap.value().commits.size(), 1U);
  ASSERT_EQ(bootstrap.value().commits.front().disposition, LidarCommitDisposition::BootstrapTarget);
  ASSERT_TRUE(bootstrap.value().commits.front().target_add);
  ASSERT_TRUE(bootstrap.value().commits.front().map_input);
  EXPECT_EQ(bootstrap.value().commits.front().target_add->retained_sweeps, 1U);

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'001U, 2'500 * kMillisecond)));
  appendImuThrough(estimator, 2'405 * kMillisecond, 2'600 * kMillisecond, imu_id);
  const auto tracking_only = estimator.processReady();
  ASSERT_TRUE(tracking_only) << tracking_only.error().detail;
  ASSERT_EQ(tracking_only.value().commits.size(), 1U);
  ASSERT_EQ(tracking_only.value().graph_transactions.size(), 1U);
  EXPECT_TRUE(tracking_only.value().graph_transactions.front().navigation_state_created);
  const LidarCommitReport& withheld = tracking_only.value().commits.front();
  EXPECT_EQ(withheld.disposition, LidarCommitDisposition::RegisteredTrackingOnly);
  ASSERT_TRUE(withheld.registration);
  EXPECT_FALSE(withheld.target_add);
  EXPECT_FALSE(withheld.map_input);
  EXPECT_EQ(withheld.commit.solve.factor_batches_added, 0U);
  EXPECT_EQ(withheld.commit.solve.active_lidar_direct_batch_factors, 0U);
  EXPECT_TRUE(withheld.degradation_detail.empty());
  EXPECT_EQ(estimator.statistics().lidar_tracking_only_registrations, 1U);
  EXPECT_EQ(estimator.statistics().lidar_registrations, 0U);
  EXPECT_EQ(estimator.statistics().lidar_degraded_commits, 0U);
  EXPECT_EQ(estimator.statistics().last_lidar_keyframe_time,
            std::optional<core::FusionTime>{core::FusionTime{2'390 * kMillisecond}});

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'002U, 2'700 * kMillisecond)));
  appendImuThrough(estimator, 2'605 * kMillisecond, 2'800 * kMillisecond, imu_id);
  const auto keyframe = estimator.processReady();
  ASSERT_TRUE(keyframe) << keyframe.error().detail;
  ASSERT_EQ(keyframe.value().commits.size(), 1U);
  ASSERT_EQ(keyframe.value().graph_transactions.size(), 2U);
  EXPECT_TRUE(keyframe.value().graph_transactions.front().navigation_state_created);
  EXPECT_FALSE(keyframe.value().graph_transactions.back().navigation_state_created);
  EXPECT_EQ(keyframe.value().graph_transactions.back().parent,
            keyframe.value().graph_transactions.front().revision);
  const LidarCommitReport& admitted = keyframe.value().commits.front();
  EXPECT_EQ(admitted.disposition, LidarCommitDisposition::Registered)
      << admitted.degradation_detail;
  ASSERT_TRUE(admitted.registration);
  ASSERT_TRUE(admitted.target_add);
  ASSERT_TRUE(admitted.map_input);
  EXPECT_EQ(admitted.map_input->sweep.id, admitted.measurement);
  EXPECT_EQ(admitted.map_input->accepted_revision, admitted.commit.revision);
  EXPECT_EQ(admitted.commit.solve.factor_batches_added, 1U);
  EXPECT_GT(admitted.commit.solve.active_lidar_direct_batch_factors, 0U);
  EXPECT_EQ(admitted.target_add->retained_sweeps, 2U);
  EXPECT_EQ(estimator.statistics().lidar_tracking_only_registrations, 1U);
  EXPECT_EQ(estimator.statistics().lidar_registrations, 1U);
  EXPECT_EQ(estimator.statistics().last_lidar_keyframe_time,
            std::optional<core::FusionTime>{core::FusionTime{2'790 * kMillisecond}});
}

TEST(LocalEstimator, FinalizedTargetCapacityFailsClosedBeforeFactorOrMapAdmission) {
  LocalEstimatorConfig config = estimatorConfig();
  config.maximum_pending_finalized_lidar_sweeps = 1U;
  auto created = LocalEstimator::create(calibration(), config);
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  ASSERT_TRUE(estimator.enqueueLidar(sweep(19'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  const auto seeded = estimator.processReady();
  ASSERT_TRUE(seeded) << seeded.error().detail;
  ASSERT_EQ(seeded.value().commits.size(), 1U);
  ASSERT_EQ(seeded.value().commits.front().disposition, LidarCommitDisposition::BootstrapTarget);
  EXPECT_EQ(seeded.value().finalized_lidar_target.pending_sweeps, 1U);

  ASSERT_TRUE(estimator.enqueueLidar(sweep(19'001U, 2'500 * kMillisecond)));
  appendImuThrough(estimator, 2'405 * kMillisecond, 2'600 * kMillisecond, imu_id);
  const auto rejected = estimator.processReady();
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalEstimatorErrorCode::PendingFinalizedTargetCapacity);
  EXPECT_EQ(rejected.error().stage, LocalEstimatorStage::FinalizedTarget);
  EXPECT_EQ(estimator.lifecycle(), LocalEstimatorLifecycle::Faulted);
  EXPECT_EQ(estimator.statistics().finalized_lidar_pending_high_watermark, 1U);
  EXPECT_EQ(estimator.statistics().lidar_registrations, 0U);
  EXPECT_EQ(estimator.finalizedLidarTargetStatistics().admitted_sweeps, 0U);
  const auto graph = estimator.estimate();
  ASSERT_TRUE(graph);
  EXPECT_EQ(graph.value().solve.active_lidar_direct_batch_factors, 0U);
}

TEST(LocalEstimator, LidarFailureRemovesRecentBatchFreezesMapAndRequiresShadowRecoveryEpoch) {
  LocalEstimatorConfig config = estimatorConfig();
  config.sensor_health_policy.consecutive_failures_to_suspect = 1U;
  config.sensor_health_policy.consecutive_failures_to_failed = 3U;
  config.sensor_health_policy.recovery_good_shadow_results = 3U;
  config.maximum_recent_faulty_batches_to_remove = 1U;
  auto created = LocalEstimator::create(calibration(), config);
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();

  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  auto initialized = estimator.processReady();
  ASSERT_TRUE(initialized) << initialized.error().detail;
  ASSERT_TRUE(initialized.value().initialization);

  std::int64_t next_imu_time = 2'205 * kMillisecond;
  FinalizedLidarTargetProcessReport finalized_process;
  const auto process_sweep = [&](core::LidarSweep candidate,
                                 std::int64_t imu_end) -> std::optional<LidarCommitReport> {
    auto enqueued = estimator.enqueueLidar(std::move(candidate));
    if (!enqueued) {
      ADD_FAILURE() << "LiDAR enqueue failed: " << enqueued.error().detail;
      return std::nullopt;
    }
    appendImuThrough(estimator, next_imu_time, imu_end, imu_id);
    next_imu_time = imu_end + kImuPeriod;
    auto processed = estimator.processReady();
    if (!processed) {
      ADD_FAILURE() << "LiDAR processing failed: " << processed.error().detail;
      return std::nullopt;
    }
    if (processed.value().commits.size() != 1U) {
      ADD_FAILURE() << "expected exactly one LiDAR commit, received "
                    << processed.value().commits.size();
      return std::nullopt;
    }
    finalized_process = processed.value().finalized_lidar_target;
    return std::move(processed).value().commits.front();
  };

  auto seed = process_sweep(sweep(20'000U, 2'300 * kMillisecond), 2'400 * kMillisecond);
  ASSERT_TRUE(seed);
  ASSERT_EQ(seed->disposition, LidarCommitDisposition::BootstrapTarget) << seed->degradation_detail;
  ASSERT_TRUE(seed->target_add);
  EXPECT_EQ(seed->target_add->retained_sweeps, 1U);

  auto accepted = process_sweep(sweep(20'001U, 2'500 * kMillisecond), 2'600 * kMillisecond);
  ASSERT_TRUE(accepted);
  ASSERT_EQ(accepted->disposition, LidarCommitDisposition::Registered)
      << accepted->degradation_detail;
  ASSERT_TRUE(accepted->health_update);
  ASSERT_TRUE(accepted->target_add);
  EXPECT_EQ(accepted->health_update->after.state, core::SensorHealthState::Active);
  EXPECT_EQ(accepted->commit.solve.active_lidar_direct_batch_factors, 1U);
  EXPECT_EQ(accepted->target_add->retained_sweeps, 2U);
  const core::FactorBatchId accepted_batch = accepted->health_update->batch_id;

  std::vector<LidarCommitReport> failures;
  failures.reserve(3U);
  for (std::size_t index = 0U; index < 3U; ++index) {
    const std::int64_t start = (2'700 + static_cast<std::int64_t>(index) * 200) * kMillisecond;
    auto failed = process_sweep(verticallyShiftedSweep(20'002U + index, start, 15.0),
                                start + 100 * kMillisecond);
    ASSERT_TRUE(failed);
    EXPECT_EQ(failed->disposition,
              LidarCommitDisposition::ImuOnlyRegistrationRejectedTargetRetained)
        << failed->degradation_detail;
    ASSERT_TRUE(failed->health_update);
    EXPECT_FALSE(failed->target_add);
    failures.push_back(std::move(*failed));
  }

  EXPECT_EQ(failures[0].health_update->before.state, core::SensorHealthState::Active);
  EXPECT_EQ(failures[0].health_update->after.state, core::SensorHealthState::Suspect);
  EXPECT_EQ(failures[0].commit.solve.active_lidar_direct_batch_factors, 1U);
  EXPECT_TRUE(failures[0].removed_factor_batches.empty());
  EXPECT_EQ(failures[1].health_update->before.state, core::SensorHealthState::Suspect);
  EXPECT_EQ(failures[1].health_update->after.state, core::SensorHealthState::Suspect);
  EXPECT_EQ(failures[1].commit.solve.active_lidar_direct_batch_factors, 1U);
  EXPECT_TRUE(failures[1].removed_factor_batches.empty());
  EXPECT_EQ(failures[2].health_update->before.state, core::SensorHealthState::Suspect);
  EXPECT_EQ(failures[2].health_update->after.state, core::SensorHealthState::Failed);
  ASSERT_EQ(failures[2].removed_factor_batches.size(), 1U);
  EXPECT_EQ(failures[2].removed_factor_batches.front().sensor,
            core::SensorInstanceId::lidar(core::LidarId{1U}));
  EXPECT_EQ(failures[2].removed_factor_batches.front().batch_id, accepted_batch);
  EXPECT_EQ(failures[2].commit.solve.factor_batches_removed, 1U);
  EXPECT_EQ(failures[2].commit.solve.lidar_direct_batch_factors_removed, 1U);
  EXPECT_EQ(failures[2].commit.solve.active_lidar_direct_batch_factors, 0U);
  ASSERT_TRUE(failures[2].target_removal);
  EXPECT_EQ(failures[2].target_removal->requested_batches, 1U);
  EXPECT_EQ(failures[2].target_removal->matched_batches, 1U);
  EXPECT_EQ(failures[2].target_removal->absent_batches, 0U);
  EXPECT_EQ(failures[2].target_removal->removed_sweeps, 1U);
  EXPECT_GT(failures[2].target_removal->removed_points, 0U);
  EXPECT_EQ(failures[2].target_removal->retained_sweeps, 1U);
  EXPECT_EQ(finalized_process.rollback_removals, 1U);
  EXPECT_TRUE(finalized_process.insertion_frozen);
  EXPECT_EQ(finalized_process.lidar_health, core::SensorHealthState::Failed);
  EXPECT_EQ(estimator.statistics().finalized_lidar_rollback_removals, 1U);

  // The first three successful registrations are shadow-only: Failed opens
  // recovery epoch 1, then two Recovering results build support. None may
  // publish a factor or mutate the accepted rolling map.
  for (std::size_t index = 0U; index < 3U; ++index) {
    const std::int64_t start = (3'300 + static_cast<std::int64_t>(index) * 200) * kMillisecond;
    auto shadow = process_sweep(sweep(20'005U + index, start), start + 100 * kMillisecond);
    ASSERT_TRUE(shadow);
    EXPECT_EQ(shadow->disposition, LidarCommitDisposition::ImuOnlyHealthQuarantinedTargetRetained)
        << shadow->degradation_detail;
    ASSERT_TRUE(shadow->registration);
    ASSERT_TRUE(shadow->health_update);
    EXPECT_EQ(shadow->health_update->after.state, core::SensorHealthState::Recovering);
    EXPECT_EQ(shadow->health_update->after.recovery_epoch, core::SensorRecoveryEpoch{1U});
    EXPECT_EQ(shadow->commit.solve.active_lidar_direct_batch_factors, 0U);
    EXPECT_FALSE(shadow->target_add);
    EXPECT_TRUE(shadow->removed_factor_batches.empty());
  }

  // The threshold-crossing shadow result atomically promotes epoch 1 and is
  // the first recovery batch allowed to enter localization and the map.
  auto recovered = process_sweep(sweep(20'008U, 3'900 * kMillisecond), 4'000 * kMillisecond);
  ASSERT_TRUE(recovered);
  EXPECT_EQ(recovered->disposition, LidarCommitDisposition::Registered)
      << recovered->degradation_detail;
  ASSERT_TRUE(recovered->health_update);
  EXPECT_EQ(recovered->health_update->before.state, core::SensorHealthState::Recovering);
  EXPECT_EQ(recovered->health_update->after.state, core::SensorHealthState::Active);
  EXPECT_EQ(recovered->health_update->after.recovery_epoch, core::SensorRecoveryEpoch{1U});
  EXPECT_FALSE(finalized_process.insertion_frozen);
  EXPECT_EQ(finalized_process.lidar_health, core::SensorHealthState::Active);
  EXPECT_GT(recovered->commit.solve.active_lidar_direct_batch_factors, 0U);
  EXPECT_EQ(recovered->commit.solve.factor_batches_added, 1U);
  ASSERT_TRUE(recovered->target_add);
  // The failed batch's cloud was removed with its localization factors. The
  // initialization seed stayed frozen through quarantine, then the newly
  // accepted recovery-epoch payload became the second retained target.
  EXPECT_EQ(recovered->target_add->retained_sweeps, 2U);
  EXPECT_EQ(estimator.statistics().lidar_failure_removal_transactions, 1U);
  EXPECT_EQ(estimator.statistics().lidar_faulty_batches_removed, 1U);
  EXPECT_EQ(estimator.statistics().lidar_faulty_target_sweeps_removed, 1U);
  EXPECT_GT(estimator.statistics().lidar_faulty_target_points_removed, 0U);
  EXPECT_EQ(estimator.statistics().lidar_health_transitions, 4U);
}

TEST(LocalEstimator, InactiveLidarFactorRollsBackCommitsImuAndRetainsVerifiedTarget) {
  auto config = estimatorConfig();
  config.lidar_registration.minimum_correspondences = 10'000U;
  auto created = LocalEstimator::create(calibration(), config);
  ASSERT_TRUE(created);
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  auto bootstrap = estimator.processReady();
  ASSERT_TRUE(bootstrap);
  ASSERT_EQ(bootstrap.value().commits.size(), 1U);

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'001U, 2'500 * kMillisecond)));
  appendImuThrough(estimator, 2'405 * kMillisecond, 2'600 * kMillisecond, imu_id);
  auto degraded = estimator.processReady();
  ASSERT_TRUE(degraded) << degraded.error().detail;
  ASSERT_EQ(degraded.value().commits.size(), 1U);
  EXPECT_EQ(degraded.value().commits.front().disposition,
            LidarCommitDisposition::ImuOnlyRegistrationRejectedTargetRetained);
  EXPECT_FALSE(degraded.value().commits.front().commit.lidar_registration);
  EXPECT_EQ(degraded.value().commits.front().commit.solve.active_lidar_direct_batch_factors, 0U);
  EXPECT_FALSE(degraded.value().commits.front().target_add.has_value());
  EXPECT_EQ(estimator.statistics().rolling_target_pose_synchronizations, 1U);
  EXPECT_EQ(estimator.statistics().rolling_target_sweeps_synchronized, 1U);
  EXPECT_EQ(estimator.statistics().lidar_rejections_target_retained, 1U);
  EXPECT_EQ(estimator.statistics().lidar_target_state_unavailable_freezes, 0U);
  EXPECT_EQ(estimator.statistics().lidar_degraded_commits, 1U);
  EXPECT_EQ(estimator.lifecycle(), LocalEstimatorLifecycle::Tracking);
}

TEST(LocalEstimator, FinalizedPayloadFreezesDuringFailureAndDrainsOnActiveRecovery) {
  LocalEstimatorConfig config = estimatorConfig();
  config.graph.target_fixed_lag = core::Duration{500 * kMillisecond};
  config.rolling_target.maximum_retained_sweeps = 1U;
  config.sensor_health_policy.consecutive_failures_to_suspect = 1U;
  config.sensor_health_policy.consecutive_failures_to_failed = 3U;
  config.sensor_health_policy.recovery_good_shadow_results = 3U;
  config.maximum_recent_faulty_batches_to_remove = 1U;
  auto created = LocalEstimator::create(calibration(), config);
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();

  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  ASSERT_TRUE(estimator.enqueueLidar(sweep(21'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  const auto seeded = estimator.processReady();
  ASSERT_TRUE(seeded) << seeded.error().detail;
  ASSERT_EQ(seeded.value().finalized_lidar_target.pending_sweeps, 1U);

  // Finalize the seed while health is Active. This old, immutable map remains
  // available to evaluate recovery scans when a newer payload is frozen.
  appendImuThrough(estimator, 2'405 * kMillisecond, 3'000 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.enqueueImuGuard(core::FusionTime{2'990 * kMillisecond}));
  const auto seed_finality = estimator.processReady();
  ASSERT_TRUE(seed_finality) << seed_finality.error().detail;
  ASSERT_EQ(seed_finality.value().finalized_lidar_target.finality_matches, 1U);
  ASSERT_EQ(seed_finality.value().finalized_lidar_target.insertions.size(), 1U);
  EXPECT_EQ(estimator.finalizedLidarTargetStatistics().admitted_sweeps, 1U);

  std::int64_t next_imu_time = 3'005 * kMillisecond;
  const auto process_sweep =
      [&](core::LidarSweep candidate,
          std::int64_t imu_end) -> std::optional<LocalEstimatorProcessReport> {
    if (!estimator.enqueueLidar(std::move(candidate))) {
      ADD_FAILURE() << "LiDAR enqueue failed";
      return std::nullopt;
    }
    appendImuThrough(estimator, next_imu_time, imu_end, imu_id);
    next_imu_time = imu_end + kImuPeriod;
    auto processed = estimator.processReady();
    if (!processed) {
      const auto& error = processed.error();
      ADD_FAILURE() << error.detail;
      if (error.rejected_solve) {
        const LocalSolveReport& solve = *error.rejected_solve;
        ADD_FAILURE() << "rejected solve iterations=" << solve.nonlinear_iterations
                      << " last_translation_m=" << solve.last_iteration_translation_correction_m
                      << " last_rotation_rad=" << solve.last_iteration_rotation_correction_rad
                      << " last_velocity_mps=" << solve.last_iteration_velocity_correction_mps
                      << " last_objective_change=" << solve.last_iteration_objective_change
                      << " full_steps_rejected=" << solve.nonlinear_full_steps_rejected
                      << " backtracking_trials=" << solve.nonlinear_backtracking_trials
                      << " cauchy_trials=" << solve.nonlinear_cauchy_backtracking_trials
                      << " zero_steps=" << solve.nonlinear_zero_step_terminations;
      }
      return std::nullopt;
    }
    return std::move(processed).value();
  };

  // Attach the next accepted keyframe to the already-live guard state. This
  // isolates the unary finalized-map transaction from another IMU append.
  auto accepted = process_sweep(sweep(21'001U, 2'900 * kMillisecond), 3'000 * kMillisecond);
  ASSERT_TRUE(accepted);
  ASSERT_EQ(accepted->commits.size(), 1U);
  ASSERT_EQ(accepted->commits.front().disposition, LidarCommitDisposition::Registered)
      << accepted->commits.front().degradation_detail;
  const LocalSolveReport& accepted_solve = accepted->commits.front().commit.solve;
  EXPECT_LT(accepted_solve.nonlinear_iterations, 8U);
  ASSERT_TRUE(accepted_solve.error_before);
  ASSERT_TRUE(accepted_solve.error_after);
  EXPECT_LE(*accepted_solve.error_after, *accepted_solve.error_before);
  EXPECT_EQ(accepted->finalized_lidar_target.pending_unfinalized_sweeps, 1U);

  std::optional<LocalEstimatorProcessReport> failed;
  for (std::size_t index = 0U; index < 3U; ++index) {
    const std::int64_t start = (3'100 + static_cast<std::int64_t>(index) * 200) * kMillisecond;
    failed = process_sweep(verticallyShiftedSweep(21'002U + index, start, 15.0),
                           start + 100 * kMillisecond);
    ASSERT_TRUE(failed);
    ASSERT_EQ(failed->commits.size(), 1U);
    EXPECT_EQ(failed->commits.front().disposition,
              LidarCommitDisposition::ImuOnlyRegistrationRejectedTargetRetained);
  }
  ASSERT_TRUE(failed);
  ASSERT_TRUE(failed->commits.front().health_update);
  EXPECT_EQ(failed->commits.front().health_update->after.state, core::SensorHealthState::Failed);
  EXPECT_EQ(failed->finalized_lidar_target.finality_matches, 1U);
  EXPECT_TRUE(failed->finalized_lidar_target.insertions.empty());
  EXPECT_EQ(failed->finalized_lidar_target.finalized_ready_sweeps, 1U);
  EXPECT_TRUE(failed->finalized_lidar_target.insertion_frozen);
  EXPECT_EQ(failed->finalized_lidar_target.lidar_health, core::SensorHealthState::Failed);
  EXPECT_EQ(estimator.finalizedLidarTargetStatistics().admitted_sweeps, 1U);
  EXPECT_EQ(estimator.statistics().finalized_lidar_freeze_events, 1U);
  EXPECT_EQ(estimator.statistics().finalized_lidar_frozen_high_watermark, 1U);

  // Three good shadow results remain quarantined. The threshold-crossing
  // fourth result promotes recovery epoch 1; processLidarHealth then drains
  // the already-final payload before admitting the new recovery keyframe.
  for (std::size_t index = 0U; index < 3U; ++index) {
    const std::int64_t start = (3'700 + static_cast<std::int64_t>(index) * 200) * kMillisecond;
    auto shadow = process_sweep(sweep(21'005U + index, start), start + 100 * kMillisecond);
    ASSERT_TRUE(shadow);
    ASSERT_EQ(shadow->commits.size(), 1U);
    EXPECT_EQ(shadow->commits.front().disposition,
              LidarCommitDisposition::ImuOnlyHealthQuarantinedTargetRetained);
    EXPECT_TRUE(shadow->finalized_lidar_target.insertion_frozen);
    EXPECT_EQ(shadow->finalized_lidar_target.finalized_ready_sweeps, 1U);
  }
  auto recovered = process_sweep(sweep(21'008U, 4'300 * kMillisecond), 4'400 * kMillisecond);
  ASSERT_TRUE(recovered);
  ASSERT_EQ(recovered->commits.size(), 1U);
  EXPECT_EQ(recovered->commits.front().disposition, LidarCommitDisposition::Registered)
      << recovered->commits.front().degradation_detail;
  ASSERT_TRUE(recovered->commits.front().health_update);
  EXPECT_EQ(recovered->commits.front().health_update->after.state, core::SensorHealthState::Active);
  EXPECT_FALSE(recovered->finalized_lidar_target.insertion_frozen);
  EXPECT_EQ(recovered->finalized_lidar_target.finalized_ready_sweeps, 0U);
  ASSERT_EQ(recovered->finalized_lidar_target.insertions.size(), 1U);
  EXPECT_EQ(estimator.finalizedLidarTargetStatistics().admitted_sweeps, 2U);
  EXPECT_EQ(estimator.lifecycle(), LocalEstimatorLifecycle::Tracking);
}

TEST(LocalEstimator, DeskewFailureStillUpdatesLidarHealthAndKeepsMapFrozen) {
  auto created = LocalEstimator::create(calibration(), estimatorConfig());
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  const auto bootstrap = estimator.processReady();
  ASSERT_TRUE(bootstrap) << bootstrap.error().detail;
  ASSERT_EQ(bootstrap.value().commits.size(), 1U);
  ASSERT_TRUE(bootstrap.value().commits.front().target_add);

  ASSERT_TRUE(estimator.enqueueLidar(deskewInvalidSweep(10'001U, 2'500 * kMillisecond)));
  appendImuThrough(estimator, 2'405 * kMillisecond, 2'600 * kMillisecond, imu_id);
  const auto processed = estimator.processReady();
  ASSERT_TRUE(processed) << processed.error().detail;
  ASSERT_EQ(processed.value().commits.size(), 1U);
  const LidarCommitReport& lidar = processed.value().commits.front();
  EXPECT_EQ(lidar.disposition, LidarCommitDisposition::ImuOnlyDeskewRejected);
  ASSERT_TRUE(lidar.health_update);
  EXPECT_EQ(lidar.health_update->before.state, core::SensorHealthState::Active);
  EXPECT_EQ(lidar.health_update->after.state, core::SensorHealthState::Suspect);
  EXPECT_TRUE(lidar.health_update->transitioned);
  EXPECT_FALSE(lidar.preprocessing);
  EXPECT_FALSE(lidar.registration);
  EXPECT_FALSE(lidar.target_add);
  EXPECT_EQ(estimator.statistics().lidar_health_transitions, 1U);
  EXPECT_EQ(estimator.statistics().lidar_degraded_commits, 1U);
}

TEST(LocalEstimator, ExactSharedLidarAddsOnlyOneFactorBatchAndNoNavigationOrImu) {
  auto created = LocalEstimator::create(calibration(), estimatorConfig());
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  const auto bootstrap = estimator.processReady();
  ASSERT_TRUE(bootstrap) << bootstrap.error().detail;
  ASSERT_EQ(bootstrap.value().commits.size(), 1U);
  ASSERT_EQ(bootstrap.value().commits.front().disposition, LidarCommitDisposition::BootstrapTarget);

  appendImuThrough(estimator, 2'405 * kMillisecond, 2'600 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.enqueueImuGuard(core::FusionTime{2'590 * kMillisecond}));
  const auto guard = estimator.processReady();
  ASSERT_TRUE(guard) << guard.error().detail;
  ASSERT_EQ(guard.value().imu_guard_commits.size(), 1U);
  const LocalGraphCommit before = estimator.estimate().value();

  const auto enqueued = estimator.enqueueLidar(sweep(10'001U, 2'500 * kMillisecond));
  ASSERT_TRUE(enqueued) << enqueued.error().detail;
  EXPECT_EQ(enqueued.value().state_admission, StateAdmissionDisposition::ExactShare);
  const auto processed = estimator.processReady();
  ASSERT_TRUE(processed) << processed.error().detail;
  EXPECT_TRUE(processed.value().dropped_sweeps.empty());
  ASSERT_EQ(processed.value().commits.size(), 1U);
  const LidarCommitReport& lidar = processed.value().commits.front();
  EXPECT_EQ(lidar.disposition, LidarCommitDisposition::Registered);
  EXPECT_FALSE(lidar.navigation_state_created);
  EXPECT_TRUE(lidar.graph_revision_created);
  EXPECT_EQ(lidar.commit.state, before.state);
  EXPECT_EQ(lidar.commit.state_time, before.state_time);
  EXPECT_EQ(lidar.commit.solve.navigation_states, before.solve.navigation_states);
  EXPECT_EQ(lidar.commit.solve.combined_imu_factors, before.solve.combined_imu_factors);
  EXPECT_EQ(lidar.commit.revision.value(), before.revision.value() + 1U);
  EXPECT_EQ(lidar.commit.solve.factor_batches_added, 1U);
  EXPECT_GT(lidar.commit.solve.lidar_direct_batch_factors_added, 0U);
  EXPECT_LT(lidar.commit.solve.nonlinear_iterations, 8U);
  ASSERT_TRUE(lidar.commit.solve.error_before);
  ASSERT_TRUE(lidar.commit.solve.error_after);
  EXPECT_LE(*lidar.commit.solve.error_after, *lidar.commit.solve.error_before);
  EXPECT_TRUE(lidar.target_add.has_value());
}

TEST(LocalEstimator, OlderLiveExactSharedLidarUsesThatStatesFullXvbAndAddsNoImuEdge) {
  auto created = LocalEstimator::create(calibration(), estimatorConfig());
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  appendImuThrough(estimator, 2'405 * kMillisecond, 2'800 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.enqueueImuGuard(core::FusionTime{2'590 * kMillisecond}));
  const auto first_guard = estimator.processReady();
  ASSERT_TRUE(first_guard) << first_guard.error().detail;
  ASSERT_EQ(first_guard.value().imu_guard_commits.size(), 1U);
  const core::StateId shared_source = first_guard.value().imu_guard_commits.front().commit.state;

  ASSERT_TRUE(estimator.enqueueImuGuard(core::FusionTime{2'790 * kMillisecond}));
  const auto second_guard = estimator.processReady();
  ASSERT_TRUE(second_guard) << second_guard.error().detail;
  ASSERT_EQ(second_guard.value().imu_guard_commits.size(), 1U);
  const LocalGraphCommit before = estimator.estimate().value();
  ASSERT_NE(shared_source, before.state);

  const auto enqueued = estimator.enqueueLidar(sweep(10'001U, 2'500 * kMillisecond));
  ASSERT_TRUE(enqueued) << enqueued.error().detail;
  ASSERT_EQ(enqueued.value().state_admission, StateAdmissionDisposition::ExactShare);
  const auto processed = estimator.processReady();
  ASSERT_TRUE(processed) << processed.error().detail;
  EXPECT_TRUE(processed.value().dropped_sweeps.empty());
  ASSERT_EQ(processed.value().commits.size(), 1U);
  const LidarCommitReport& lidar = processed.value().commits.front();
  EXPECT_EQ(lidar.disposition, LidarCommitDisposition::Registered) << lidar.degradation_detail;
  EXPECT_FALSE(lidar.navigation_state_created);
  EXPECT_TRUE(lidar.graph_revision_created);
  EXPECT_EQ(lidar.commit.state, before.state);
  EXPECT_EQ(lidar.commit.state_time, before.state_time);
  EXPECT_EQ(lidar.commit.solve.navigation_states, before.solve.navigation_states);
  EXPECT_EQ(lidar.commit.solve.combined_imu_factors, before.solve.combined_imu_factors);
  EXPECT_EQ(lidar.commit.revision.value(), before.revision.value() + 1U);
  ASSERT_EQ(lidar.commit.factor_batch_transitions.size(), 1U);
  EXPECT_EQ(lidar.commit.factor_batch_transitions.front().batch.source_state, shared_source);
  EXPECT_EQ(lidar.commit.factor_batch_transitions.front().batch.source_time,
            core::FusionTime{2'590 * kMillisecond});
  EXPECT_TRUE(lidar.target_add.has_value());
}

TEST(LocalEstimator, ExactSharedLidarRejectionLeavesGraphAndRollingTargetFrozen) {
  LocalEstimatorConfig config = estimatorConfig();
  config.lidar_registration.minimum_correspondences = 10'000U;
  auto created = LocalEstimator::create(calibration(), config);
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  appendImuThrough(estimator, 2'405 * kMillisecond, 2'600 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.enqueueImuGuard(core::FusionTime{2'590 * kMillisecond}));
  ASSERT_TRUE(estimator.processReady());
  const LocalGraphCommit before = estimator.estimate().value();

  const auto exact = estimator.enqueueLidar(sweep(10'001U, 2'500 * kMillisecond));
  ASSERT_TRUE(exact) << exact.error().detail;
  ASSERT_EQ(exact.value().state_admission, StateAdmissionDisposition::ExactShare);
  const auto rejected = estimator.processReady();
  ASSERT_TRUE(rejected) << rejected.error().detail;
  ASSERT_EQ(rejected.value().commits.size(), 1U);
  const LidarCommitReport& exact_report = rejected.value().commits.front();
  EXPECT_EQ(exact_report.disposition,
            LidarCommitDisposition::ImuOnlyRegistrationRejectedTargetRetained);
  EXPECT_FALSE(exact_report.navigation_state_created);
  EXPECT_FALSE(exact_report.graph_revision_created);
  EXPECT_FALSE(exact_report.target_add);
  EXPECT_FALSE(exact_report.map_input);
  const LocalGraphCommit after = estimator.estimate().value();
  EXPECT_EQ(after.revision, before.revision);
  EXPECT_EQ(after.state, before.state);
  EXPECT_EQ(after.solve.navigation_states, before.solve.navigation_states);
  EXPECT_EQ(after.solve.combined_imu_factors, before.solve.combined_imu_factors);

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'002U, 2'700 * kMillisecond)));
  appendImuThrough(estimator, 2'605 * kMillisecond, 2'800 * kMillisecond, imu_id);
  const auto next = estimator.processReady();
  ASSERT_TRUE(next) << next.error().detail;
  ASSERT_EQ(next.value().commits.size(), 1U);
  ASSERT_TRUE(next.value().commits.front().target);
  EXPECT_EQ(next.value().commits.front().target->selected_targets, 1U);
  EXPECT_FALSE(next.value().commits.front().target_add);
}

TEST(LocalEstimator, GraphFinalitySurvivesRollingEvictionAndEnablesMapOnlyContinuation) {
  auto config = estimatorConfig();
  config.graph.target_fixed_lag = core::Duration{250 * kMillisecond};
  config.rolling_target.maximum_retained_sweeps = 1U;
  config.maximum_recent_faulty_batches_to_remove = 1U;
  config.finalized_lidar_prune_interval_sweeps = 1U;
  auto created = LocalEstimator::create(calibration(), config);
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  auto bootstrap = estimator.processReady();
  ASSERT_TRUE(bootstrap) << bootstrap.error().detail;
  ASSERT_EQ(bootstrap.value().commits.size(), 1U);
  ASSERT_EQ(bootstrap.value().commits.front().disposition, LidarCommitDisposition::BootstrapTarget);
  EXPECT_EQ(estimator.finalizedLidarTargetStatistics().admitted_sweeps, 0U);
  EXPECT_EQ(estimator.finalizedLidarTargetStatistics().retained_points, 0U);
  EXPECT_EQ(bootstrap.value().finalized_lidar_target.pending_sweeps, 1U);
  EXPECT_EQ(bootstrap.value().finalized_lidar_target.pending_unfinalized_sweeps, 1U);

  // The 400 ms gap places state 11 behind the 250 ms fixed-lag boundary. The
  // rolling target evicts it, but the independent pending-finality queue
  // publishes the exact finalized pose into the persistent map before this
  // scan registers. With no live owner left, the same single registration
  // call therefore produces a unary finalized-map factor.
  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'001U, 2'700 * kMillisecond)));
  appendImuThrough(estimator, 2'405 * kMillisecond, 2'800 * kMillisecond, imu_id);
  auto continued = estimator.processReady();
  ASSERT_TRUE(continued) << continued.error().detail;
  ASSERT_EQ(continued.value().commits.size(), 1U);
  const LidarCommitReport& commit = continued.value().commits.front();
  EXPECT_EQ(commit.disposition, LidarCommitDisposition::Registered) << commit.degradation_detail;
  EXPECT_EQ(commit.commit.state, core::StateId{12U});
  EXPECT_FALSE(commit.target);
  ASSERT_TRUE(commit.registration);
  EXPECT_EQ(commit.registration->diagnostics.live_target_count, 0U);
  EXPECT_EQ(commit.registration->diagnostics.finalized_map_target_count, 1U);
  EXPECT_GT(commit.registration->diagnostics.finalized_map_correspondences, 0U);
  EXPECT_TRUE(commit.commit.lidar_pairs.empty());
  ASSERT_TRUE(commit.commit.lidar_finalized_map);
  EXPECT_EQ(commit.commit.lidar_finalized_map->unique_finalized_owners, 1U);
  EXPECT_EQ(commit.commit.lidar_finalized_map->map_odom_epoch, config.odom_epoch);
  EXPECT_EQ(commit.commit.lidar_finalized_map->map_sensor,
            core::SensorInstanceId::lidar(core::LidarId{1U}));
  EXPECT_GE(commit.commit.lidar_finalized_map->owner_pose_covariance_inflation, 1.0);
  EXPECT_GT(commit.commit.lidar_finalized_map->information_scale, 0.0);
  EXPECT_LE(commit.commit.lidar_finalized_map->information_scale, 1.0 / 6.0);
  ASSERT_EQ(commit.commit.lineage.correlations.size(), 1U);
  EXPECT_EQ(commit.commit.lineage.correlations.front().policy, core::CorrelationPolicyRevision{3U});
  EXPECT_DOUBLE_EQ(commit.commit.lineage.correlations.front().covariance_inflation,
                   config.lidar_target_reuse_covariance_inflation *
                       config.lidar_imu_conditioning_covariance_inflation);
  EXPECT_DOUBLE_EQ(*commit.commit.lineage.correlations.front().total_information_cap,
                   config.lidar_registration.maximum_translation_information /
                       commit.commit.lineage.correlations.front().covariance_inflation);
  EXPECT_EQ(measurementRootsWithRole(commit.commit.lineage, core::ObservationRole::PrimaryResidual),
            (std::set<std::uint64_t>{10'001U}));
  const std::set<std::uint64_t> conditioning_roots =
      measurementRootsWithRole(commit.commit.lineage, core::ObservationRole::ConditioningOnly);
  EXPECT_TRUE(conditioning_roots.contains(10'000U));
  const std::size_t conditioning_usage_count =
      std::count_if(commit.commit.lineage.usage.begin(), commit.commit.lineage.usage.end(),
                    [](const core::ObservationUsage& usage) {
                      return usage.role == core::ObservationRole::ConditioningOnly;
                    });
  EXPECT_EQ(conditioning_usage_count, conditioning_roots.size());
  ASSERT_EQ(continued.value().finalized_lidar_target.finality_matches, 1U);
  ASSERT_EQ(continued.value().finalized_lidar_target.insertions.size(), 1U);
  ASSERT_EQ(continued.value().finalized_lidar_target.prunes.size(), 1U);
  EXPECT_EQ(continued.value().finalized_lidar_target.prunes.front().removed_points, 0U);
  EXPECT_GT(continued.value().finalized_lidar_target.insertions.front().admitted_points, 0U);
  EXPECT_EQ(continued.value().finalized_lidar_target.pending_sweeps, 1U);
  EXPECT_EQ(continued.value().finalized_lidar_target.pending_unfinalized_sweeps, 1U);
  EXPECT_EQ(continued.value().finalized_lidar_target.finalized_ready_sweeps, 0U);
  EXPECT_GT(continued.value().finalized_lidar_target.retained_points, 0U);
  EXPECT_EQ(estimator.finalizedLidarTargetStatistics().admitted_sweeps, 1U);
  EXPECT_EQ(estimator.statistics().rolling_target_finalized_sweeps_evicted, 1U);
  EXPECT_EQ(estimator.statistics().finalized_lidar_finality_matches, 1U);
  EXPECT_EQ(estimator.statistics().finalized_lidar_insertions, 1U);
  EXPECT_EQ(estimator.statistics().finalized_lidar_prune_attempts, 1U);
  EXPECT_EQ(estimator.statistics().finalized_lidar_prune_transactions, 0U);
  EXPECT_EQ(estimator.statistics().lidar_target_state_unavailable_freezes, 0U);
  EXPECT_EQ(estimator.statistics().lidar_registrations, 1U);
  EXPECT_EQ(estimator.statistics().lidar_degraded_commits, 0U);
  const auto* finalized_timing =
      estimator.pipelineTimingReport().find(LocalPipelineTimingStage::FinalizedTargetUpdate);
  ASSERT_NE(finalized_timing, nullptr);
  EXPECT_GT(finalized_timing->wall.total_samples, 0U);
  EXPECT_GT(finalized_timing->thread_cpu.total_samples, 0U);
  EXPECT_EQ(estimator.lifecycle(), LocalEstimatorLifecycle::Tracking);
}

TEST(LocalEstimator, MixedLiveAndFinalizedMapKeepIndependentInformationScaling) {
  auto config = estimatorConfig();
  config.graph.target_fixed_lag = core::Duration{250 * kMillisecond};
  config.finalized_map_correlation_inflation_floor = 34.629718841665124;
  auto created = LocalEstimator::create(calibration(), config);
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  // Seed the eventual finalized map with the complete synthetic scene.
  ASSERT_TRUE(estimator.enqueueLidar(sweep(30'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  const auto seeded = estimator.processReady();
  ASSERT_TRUE(seeded) << seeded.error().detail;
  ASSERT_EQ(seeded.value().commits.size(), 1U);
  ASSERT_EQ(seeded.value().commits.front().disposition, LidarCommitDisposition::BootstrapTarget);

  // Retain a newer live target, but move half of its scene outside the ICP
  // gate. The final source can then select the unchanged half from this live
  // target and the complementary half from the older finalized-map target.
  core::LidarSweep partial_live = halfVerticallyShiftedSweep(30'001U, 2'500 * kMillisecond, 3.0);
  ASSERT_TRUE(estimator.enqueueLidar(std::move(partial_live)));
  appendImuThrough(estimator, 2'405 * kMillisecond, 2'600 * kMillisecond, imu_id);
  const auto live = estimator.processReady();
  ASSERT_TRUE(live) << live.error().detail;
  ASSERT_EQ(live.value().commits.size(), 1U);
  ASSERT_EQ(live.value().commits.front().disposition, LidarCommitDisposition::Registered)
      << live.value().commits.front().degradation_detail;

  // Advancing only the IMU spine makes the first LiDAR state graph-final and
  // publishes it into the persistent map while the second LiDAR state remains
  // live inside the fixed-lag window.
  appendImuThrough(estimator, 2'605 * kMillisecond, 2'700 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.enqueueImuGuard(core::FusionTime{2'690 * kMillisecond}));
  const auto finality = estimator.processReady();
  ASSERT_TRUE(finality) << finality.error().detail;
  ASSERT_EQ(finality.value().finalized_lidar_target.finality_matches, 1U);
  ASSERT_EQ(finality.value().finalized_lidar_target.insertions.size(), 1U);

  ASSERT_TRUE(estimator.enqueueLidar(sweep(30'002U, 2'700 * kMillisecond)));
  appendImuThrough(estimator, 2'705 * kMillisecond, 2'800 * kMillisecond, imu_id);
  const auto mixed = estimator.processReady();
  ASSERT_TRUE(mixed) << mixed.error().detail;
  ASSERT_EQ(mixed.value().commits.size(), 1U);
  const LidarCommitReport& commit = mixed.value().commits.front();
  ASSERT_EQ(commit.disposition, LidarCommitDisposition::Registered) << commit.degradation_detail;
  ASSERT_TRUE(commit.registration);
  EXPECT_EQ(commit.registration->diagnostics.live_target_count, 1U);
  EXPECT_EQ(commit.registration->diagnostics.finalized_map_target_count, 1U);
  EXPECT_GT(commit.registration->diagnostics.live_correspondences, 0U);
  EXPECT_GT(commit.registration->diagnostics.finalized_map_correspondences, 0U);

  const double base_covariance_inflation = config.lidar_target_reuse_covariance_inflation *
                                           config.lidar_imu_conditioning_covariance_inflation;
  ASSERT_FALSE(commit.commit.lidar_pairs.empty());
  for (const DirectLidarPairReport& pair : commit.commit.lidar_pairs) {
    EXPECT_DOUBLE_EQ(pair.information_scale, 1.0 / base_covariance_inflation);
  }
  ASSERT_TRUE(commit.commit.lidar_finalized_map);
  ASSERT_GT(commit.commit.lidar_finalized_map->owner_pose_covariance_inflation, 1.0);
  EXPECT_GT(config.finalized_map_correlation_inflation_floor,
            commit.commit.lidar_finalized_map->owner_pose_covariance_inflation);
  EXPECT_DOUBLE_EQ(commit.commit.lidar_finalized_map->configured_correlation_inflation_floor,
                   config.finalized_map_correlation_inflation_floor);
  const double expected_effective_covariance_inflation =
      std::max(commit.commit.lidar_finalized_map->owner_pose_covariance_inflation,
               config.finalized_map_correlation_inflation_floor);
  EXPECT_DOUBLE_EQ(commit.commit.lidar_finalized_map->effective_covariance_inflation,
                   expected_effective_covariance_inflation);
  EXPECT_DOUBLE_EQ(commit.commit.lidar_finalized_map->effective_covariance_inflation,
                   config.finalized_map_correlation_inflation_floor);
  const double expected_map_information_scale =
      1.0 / (base_covariance_inflation * expected_effective_covariance_inflation);
  EXPECT_DOUBLE_EQ(commit.commit.lidar_finalized_map->information_scale,
                   expected_map_information_scale);
  EXPECT_LT(commit.commit.lidar_finalized_map->information_scale, 1.0 / base_covariance_inflation);

  ASSERT_EQ(commit.commit.lineage.correlations.size(), 1U);
  const core::CorrelationDeclaration& declaration = commit.commit.lineage.correlations.front();
  EXPECT_EQ(declaration.policy, core::CorrelationPolicyRevision{3U});
  EXPECT_DOUBLE_EQ(declaration.covariance_inflation, base_covariance_inflation);
  ASSERT_TRUE(declaration.total_information_cap);
  EXPECT_DOUBLE_EQ(
      *declaration.total_information_cap,
      config.lidar_registration.maximum_translation_information / base_covariance_inflation);
}

TEST(LocalEstimator, FinalizedMapCapacitySkipsOnlyTheOptionalMapCopyAndKeepsTracking) {
  auto config = estimatorConfig();
  config.graph.target_fixed_lag = core::Duration{250 * kMillisecond};
  config.finalized_lidar_target.hard_point_capacity = 100U;
  config.finalized_lidar_prune_interval_sweeps = 2U;
  auto created = LocalEstimator::create(calibration(), config);
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  ASSERT_TRUE(estimator.enqueueLidar(sweep(31'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  const auto seeded = estimator.processReady();
  ASSERT_TRUE(seeded) << seeded.error().detail;
  ASSERT_EQ(seeded.value().commits.size(), 1U);
  ASSERT_EQ(seeded.value().commits.front().disposition, LidarCommitDisposition::BootstrapTarget);

  ASSERT_TRUE(
      estimator.enqueueLidar(halfVerticallyShiftedSweep(31'001U, 2'500 * kMillisecond, 3.0)));
  appendImuThrough(estimator, 2'405 * kMillisecond, 2'600 * kMillisecond, imu_id);
  const auto localized = estimator.processReady();
  ASSERT_TRUE(localized) << localized.error().detail;
  ASSERT_EQ(localized.value().commits.size(), 1U);
  ASSERT_EQ(localized.value().commits.front().disposition, LidarCommitDisposition::Registered)
      << localized.value().commits.front().degradation_detail;

  // Finalize the seed first. It fits below the hard cap and leaves one valid,
  // readable persistent target for subsequent map-only localization.
  appendImuThrough(estimator, 2'605 * kMillisecond, 2'700 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.enqueueImuGuard(core::FusionTime{2'690 * kMillisecond}));
  const auto seed_finality = estimator.processReady();
  ASSERT_TRUE(seed_finality) << seed_finality.error().detail;
  ASSERT_EQ(seed_finality.value().finalized_lidar_target.insertions.size(), 1U);
  const auto map_before_pressure = estimator.finalizedLidarTargetStatistics();
  ASSERT_GT(map_before_pressure.retained_points, 0U);
  ASSERT_LT(map_before_pressure.retained_points, config.finalized_lidar_target.hard_point_capacity);

  // The shifted half remains localization-valid through its overlapping half,
  // but its new finalized geometry cannot fit. Capacity recovery prunes once,
  // retries the identical whole sweep, and terminally skips only that optional
  // map copy without changing LiDAR health or the existing map.
  appendImuThrough(estimator, 2'705 * kMillisecond, 3'000 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.enqueueImuGuard(core::FusionTime{2'990 * kMillisecond}));
  const auto pressure = estimator.processReady();
  ASSERT_TRUE(pressure) << pressure.error().detail;
  ASSERT_EQ(pressure.value().finalized_lidar_target.capacity_recovery_attempts, 1U);
  ASSERT_EQ(pressure.value().finalized_lidar_target.capacity_recovery_successes, 0U);
  ASSERT_EQ(pressure.value().finalized_lidar_target.capacity_skips.size(), 1U);
  const FinalizedLidarTargetCapacitySkip& skipped =
      pressure.value().finalized_lidar_target.capacity_skips.front();
  EXPECT_EQ(skipped.reason, FinalizedLidarTargetCapacitySkipReason::RetryAfterPruneStillFull);
  EXPECT_EQ(skipped.sweep, core::MeasurementId{31'001U});
  EXPECT_EQ(skipped.hard_point_capacity, config.finalized_lidar_target.hard_point_capacity);
  EXPECT_EQ(skipped.retained_points, map_before_pressure.retained_points);
  EXPECT_EQ(skipped.map_version, map_before_pressure.version);
  EXPECT_EQ(skipped.map_checksum, map_before_pressure.checksum);
  EXPECT_TRUE(pressure.value().finalized_lidar_target.capacity_saturated);
  EXPECT_EQ(pressure.value().finalized_lidar_target.pending_sweeps, 0U);
  EXPECT_EQ(pressure.value().finalized_lidar_target.lidar_health, core::SensorHealthState::Active);
  EXPECT_EQ(estimator.lifecycle(), LocalEstimatorLifecycle::Tracking);
  EXPECT_EQ(estimator.finalizedLidarTargetStatistics().retained_points,
            map_before_pressure.retained_points);
  EXPECT_EQ(estimator.finalizedLidarTargetStatistics().version, map_before_pressure.version);
  EXPECT_EQ(estimator.finalizedLidarTargetStatistics().checksum, map_before_pressure.checksum);

  // The rolling owner is now final, so this continuation exercises the old
  // persistent map directly. Resource pressure must not make localization or
  // its FactorBatch/map gate unavailable.
  ASSERT_TRUE(estimator.enqueueLidar(sweep(31'002U, 3'100 * kMillisecond)));
  appendImuThrough(estimator, 3'005 * kMillisecond, 3'200 * kMillisecond, imu_id);
  const auto continued = estimator.processReady();
  ASSERT_TRUE(continued) << continued.error().detail;
  ASSERT_EQ(continued.value().commits.size(), 1U);
  const LidarCommitReport& commit = continued.value().commits.front();
  EXPECT_EQ(commit.disposition, LidarCommitDisposition::Registered) << commit.degradation_detail;
  ASSERT_TRUE(commit.registration);
  EXPECT_EQ(commit.registration->diagnostics.finalized_map_target_count, 1U);
  EXPECT_GT(commit.registration->diagnostics.finalized_map_correspondences, 0U);
  EXPECT_EQ(estimator.statistics().finalized_lidar_capacity_skipped_sweeps, 1U);
  EXPECT_EQ(estimator.statistics().finalized_lidar_capacity_recovery_attempts, 1U);
  EXPECT_EQ(estimator.statistics().finalized_lidar_capacity_recovery_successes, 0U);

  // When that continuation becomes graph-final, saturation suppresses a
  // repeated whole-map scan and records the skipped payload. The configured
  // cadence is now reached, so the next finalized payload may probe once.
  appendImuThrough(estimator, 3'205 * kMillisecond, 3'600 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.enqueueImuGuard(core::FusionTime{3'590 * kMillisecond}));
  const auto suppressed = estimator.processReady();
  ASSERT_TRUE(suppressed) << suppressed.error().detail;
  ASSERT_EQ(suppressed.value().finalized_lidar_target.capacity_skips.size(), 1U);
  EXPECT_EQ(suppressed.value().finalized_lidar_target.capacity_skips.front().reason,
            FinalizedLidarTargetCapacitySkipReason::RetrySuppressedWhileSaturated);
  EXPECT_TRUE(suppressed.value().finalized_lidar_target.capacity_saturated);
  EXPECT_EQ(suppressed.value().finalized_lidar_target.capacity_skips_since_retry, 2U);
  EXPECT_EQ(estimator.statistics().finalized_lidar_capacity_retry_suppressions, 1U);

  ASSERT_TRUE(estimator.enqueueLidar(sweep(31'003U, 3'700 * kMillisecond)));
  appendImuThrough(estimator, 3'605 * kMillisecond, 3'800 * kMillisecond, imu_id);
  const auto probe_source = estimator.processReady();
  ASSERT_TRUE(probe_source) << probe_source.error().detail;
  ASSERT_EQ(probe_source.value().commits.size(), 1U);
  EXPECT_EQ(probe_source.value().commits.front().disposition, LidarCommitDisposition::Registered)
      << probe_source.value().commits.front().degradation_detail;

  // This cloud overlaps the trusted seed geometry closely enough that the
  // cadence probe fits atomically, clearing saturation.
  appendImuThrough(estimator, 3'805 * kMillisecond, 4'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.enqueueImuGuard(core::FusionTime{4'190 * kMillisecond}));
  const auto recovered = estimator.processReady();
  ASSERT_TRUE(recovered) << recovered.error().detail;
  ASSERT_EQ(recovered.value().finalized_lidar_target.insertions.size(), 1U);
  EXPECT_GT(recovered.value().finalized_lidar_target.insertions.front().admitted_points, 0U);
  EXPECT_LE(estimator.finalizedLidarTargetStatistics().retained_points,
            config.finalized_lidar_target.hard_point_capacity);
  EXPECT_FALSE(recovered.value().finalized_lidar_target.capacity_saturated);
  EXPECT_EQ(recovered.value().finalized_lidar_target.capacity_skips_since_retry, 0U);
}

TEST(LocalEstimator, PreprocessingRejectionLeavesTheNextCommittedIdentitiesContiguous) {
  auto created = LocalEstimator::create(calibration(), estimatorConfig());
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  auto initialized = estimator.processReady();
  ASSERT_TRUE(initialized) << initialized.error().detail;
  ASSERT_TRUE(initialized.value().initialization);

  ASSERT_TRUE(estimator.enqueueLidar(verticallyShiftedSweep(10'000U, 2'300 * kMillisecond, 100.0)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  auto rejected = estimator.processReady();
  ASSERT_TRUE(rejected) << rejected.error().detail;
  ASSERT_EQ(rejected.value().commits.size(), 1U);
  const LidarCommitReport& fallback = rejected.value().commits.front();
  EXPECT_EQ(fallback.disposition, LidarCommitDisposition::ImuOnlyPreprocessingRejected);
  EXPECT_FALSE(fallback.map_input);
  EXPECT_EQ(fallback.commit.state, core::StateId{11U});
  const GraphLineageIdentities fallback_ids = graphLineageIdentities(fallback.commit.lineage);
  EXPECT_EQ(fallback_ids.lineage, 3U);
  EXPECT_EQ(fallback_ids.consumer, 3U);
  EXPECT_EQ(fallback_ids.factor_group, 2U);

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'001U, 2'500 * kMillisecond)));
  appendImuThrough(estimator, 2'405 * kMillisecond, 2'600 * kMillisecond, imu_id);
  auto accepted = estimator.processReady();
  ASSERT_TRUE(accepted) << accepted.error().detail;
  ASSERT_EQ(accepted.value().commits.size(), 1U);
  const LidarCommitReport& bootstrap = accepted.value().commits.front();
  EXPECT_EQ(bootstrap.disposition, LidarCommitDisposition::BootstrapTarget);
  EXPECT_TRUE(bootstrap.map_input);
  EXPECT_EQ(bootstrap.commit.state, core::StateId{12U});
  const GraphLineageIdentities accepted_ids = graphLineageIdentities(bootstrap.commit.lineage);
  EXPECT_EQ(accepted_ids.lineage, 5U);
  EXPECT_EQ(accepted_ids.consumer, 5U);
  EXPECT_EQ(accepted_ids.factor_group, 3U);
}

TEST(LocalEstimator, RejectedLidarCandidatesDoNotConsumeGraphFacingIdentityRanges) {
  auto config = estimatorConfig();
  config.lidar_registration.minimum_correspondences = 10'000U;
  auto created = LocalEstimator::create(calibration(), config);
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  auto bootstrap = estimator.processReady();
  ASSERT_TRUE(bootstrap) << bootstrap.error().detail;
  ASSERT_EQ(bootstrap.value().commits.size(), 1U);

  std::array<GraphLineageIdentities, 2> fallback_identities;
  for (std::size_t index = 0U; index < fallback_identities.size(); ++index) {
    const std::int64_t start = (2'500 + static_cast<std::int64_t>(index) * 200) * kMillisecond;
    ASSERT_TRUE(estimator.enqueueLidar(sweep(10'001U + index, start)));
    appendImuThrough(estimator, (2'405 + static_cast<std::int64_t>(index) * 200) * kMillisecond,
                     (2'600 + static_cast<std::int64_t>(index) * 200) * kMillisecond, imu_id);
    auto degraded = estimator.processReady();
    ASSERT_TRUE(degraded) << degraded.error().detail;
    ASSERT_EQ(degraded.value().commits.size(), 1U);
    const LidarCommitReport& commit = degraded.value().commits.front();
    EXPECT_EQ(commit.disposition,
              LidarCommitDisposition::ImuOnlyRegistrationRejectedTargetRetained);
    fallback_identities[index] = graphLineageIdentities(commit.commit.lineage);
  }

  // The second sweep legitimately allocates one knot-request lineage and
  // consumer. Only its committed IMU factor may consume another graph
  // lineage/consumer and one factor group; rejected source/target/LiDAR
  // candidate identities are restored.
  EXPECT_EQ(fallback_identities[1].lineage, fallback_identities[0].lineage + 2U);
  EXPECT_EQ(fallback_identities[1].consumer, fallback_identities[0].consumer + 2U);
  EXPECT_EQ(fallback_identities[1].factor_group, fallback_identities[0].factor_group + 1U);
}

TEST(LocalEstimator, AcceptedLidarBatchContainsOnlyItsSourceSweepAsPrimaryEvidence) {
  auto created = LocalEstimator::create(calibration(), estimatorConfig());
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  appendImuThrough(estimator, 0, 2'200 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'000U, 2'300 * kMillisecond)));
  appendImuThrough(estimator, 2'205 * kMillisecond, 2'400 * kMillisecond, imu_id);
  ASSERT_TRUE(estimator.processReady());

  ASSERT_TRUE(estimator.enqueueLidar(verticallyShiftedSweep(10'001U, 2'500 * kMillisecond, 15.0)));
  appendImuThrough(estimator, 2'405 * kMillisecond, 2'600 * kMillisecond, imu_id);
  auto rejected = estimator.processReady();
  ASSERT_TRUE(rejected) << rejected.error().detail;
  ASSERT_EQ(rejected.value().commits.size(), 1U);
  EXPECT_EQ(rejected.value().commits.front().disposition,
            LidarCommitDisposition::ImuOnlyRegistrationRejectedTargetRetained);
  EXPECT_FALSE(rejected.value().commits.front().map_input);

  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'002U, 2'700 * kMillisecond)));
  appendImuThrough(estimator, 2'605 * kMillisecond, 2'800 * kMillisecond, imu_id);
  auto registered = estimator.processReady();
  ASSERT_TRUE(registered) << registered.error().detail;
  ASSERT_EQ(registered.value().commits.size(), 1U);
  const LidarCommitReport& commit = registered.value().commits.front();
  EXPECT_EQ(commit.disposition, LidarCommitDisposition::Registered);
  EXPECT_TRUE(commit.map_input);
  ASSERT_TRUE(commit.target);
  EXPECT_EQ(commit.target->selected_targets, 1U);
  ASSERT_EQ(commit.commit.lineage.correlations.size(), 1U);
  EXPECT_EQ(commit.commit.lineage.correlations.front().group, core::CorrelationGroupId{1U});
  EXPECT_EQ(commit.commit.lineage.correlations.front().policy, core::CorrelationPolicyRevision{1U});
  EXPECT_EQ(commit.commit.lineage.correlations.front().treatment,
            core::CorrelationTreatment::CovarianceInflationAndInformationCap);
  EXPECT_DOUBLE_EQ(commit.commit.lineage.correlations.front().covariance_inflation, 6.0);
  ASSERT_TRUE(commit.commit.lineage.correlations.front().total_information_cap);
  EXPECT_DOUBLE_EQ(*commit.commit.lineage.correlations.front().total_information_cap,
                   estimatorConfig().lidar_registration.maximum_translation_information / 6.0);
  EXPECT_TRUE(core::contentHashPresent(commit.commit.lineage.checksum));
  EXPECT_EQ(measurementRootsWithRole(commit.commit.lineage, core::ObservationRole::PrimaryResidual),
            (std::set<std::uint64_t>{10'002U}));
  const std::set<std::uint64_t> conditioning =
      measurementRootsWithRole(commit.commit.lineage, core::ObservationRole::ConditioningOnly);
  EXPECT_TRUE(conditioning.contains(10'000U));
  EXPECT_FALSE(conditioning.contains(10'001U));
  EXPECT_FALSE(conditioning.contains(10'002U));
  std::size_t conditioning_usage_count = 0U;
  bool has_imu_conditioning = false;
  for (const core::ObservationUsage& usage : commit.commit.lineage.usage) {
    if (usage.role != core::ObservationRole::ConditioningOnly) {
      continue;
    }
    ++conditioning_usage_count;
    EXPECT_FALSE(usage.factor_group);
    EXPECT_EQ(usage.correlation_group, commit.commit.lineage.correlations.front().group);
    const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
    ASSERT_NE(measurement, nullptr);
    has_imu_conditioning = has_imu_conditioning || measurement->value() < 10'000U;
  }
  EXPECT_TRUE(has_imu_conditioning);
  EXPECT_EQ(conditioning_usage_count, conditioning.size());
  for (const DirectLidarPairReport& pair : commit.commit.lidar_pairs) {
    EXPECT_NEAR(pair.information_scale, 1.0 / 6.0, 1.0e-12);
  }
  EXPECT_EQ(estimator.statistics().lidar_rejections_target_retained, 1U);
  EXPECT_EQ(estimator.statistics().lidar_target_state_unavailable_freezes, 0U);
}

TEST(LocalEstimator, RejectedBootstrapConsumesOnlyItsTerminalStateRequestProvenance) {
  const auto run =
      [](bool inject_preprocessing_rejection) -> std::optional<BootstrapRecoverySnapshot> {
    LocalEstimatorConfig config = estimatorConfig();
    config.lidar_bootstrap.preprocessing = config.lidar_preprocessing;
    config.lidar_bootstrap.registration = config.lidar_registration;
    config.lidar_bootstrap.registration.target_voxel_resolution_m = 0.5;
    config.lidar_bootstrap.registration.maximum_target_points_per_target = 10'000U;
    config.lidar_bootstrap.registration.minimum_correspondences = 20U;
    config.lidar_bootstrap.registration.residual_standard_deviation_m = 0.05;
    config.lidar_bootstrap.registration.absolute_normalized_observable_eigenvalue = 1.0e-8;
    config.lidar_bootstrap.registration.minimum_observable_rank = 1U;

    auto created = LocalEstimator::create(calibration(), config);
    if (!created) {
      ADD_FAILURE() << created.error().detail;
      return std::nullopt;
    }
    auto estimator = std::move(created).value();
    std::uint64_t imu_id = 1U;
    appendImuThrough(estimator, 0, 100 * kMillisecond, imu_id);
    if (!estimator.enqueueLidar(sweep(10'000U, 0))) {
      ADD_FAILURE() << "bootstrap anchor enqueue failed";
      return std::nullopt;
    }
    auto anchor = estimator.processReady();
    if (!anchor || anchor.value().bootstrap.size() != 1U ||
        !anchor.value().bootstrap.front().commit ||
        anchor.value().bootstrap.front().commit->disposition !=
            LidarBootstrapDisposition::AnchorCreated) {
      ADD_FAILURE() << "bootstrap anchor did not commit";
      return std::nullopt;
    }

    appendImuThrough(estimator, 105 * kMillisecond, 220 * kMillisecond, imu_id);
    if (inject_preprocessing_rejection) {
      if (!estimator.enqueueLidar(verticallyShiftedSweep(10'001U, 125 * kMillisecond, 100.0))) {
        ADD_FAILURE() << "rejected bootstrap candidate enqueue failed";
        return std::nullopt;
      }
      auto rejected = estimator.processReady();
      if (!rejected || rejected.value().bootstrap.size() != 1U ||
          !rejected.value().bootstrap.front().rejection ||
          rejected.value().bootstrap.front().rejection->code !=
              LidarBootstrapErrorCode::RegistrationCloudBuildFailed) {
        ADD_FAILURE() << "bootstrap preprocessing candidate was not rejected as expected";
        return std::nullopt;
      }
    }

    appendImuThrough(estimator, 225 * kMillisecond, 340 * kMillisecond, imu_id);
    if (!estimator.enqueueLidar(sweep(10'002U, 250 * kMillisecond))) {
      ADD_FAILURE() << "bootstrap recovery enqueue failed";
      return std::nullopt;
    }
    auto recovered = estimator.processReady();
    if (!recovered || recovered.value().bootstrap.size() != 1U ||
        !recovered.value().bootstrap.front().commit ||
        !recovered.value().bootstrap.front().commit->segment) {
      ADD_FAILURE() << "bootstrap recovery increment did not commit";
      return std::nullopt;
    }
    const MotionInitializationSegment segment =
        *recovered.value().bootstrap.front().commit->segment;

    appendImuThrough(estimator, 345 * kMillisecond, 2'200 * kMillisecond, imu_id);
    auto initialized = estimator.processReady();
    if (!initialized || !initialized.value().initialization) {
      ADD_FAILURE() << "stationary graph initialization did not commit after bootstrap recovery";
      return std::nullopt;
    }
    return BootstrapRecoverySnapshot{
        initialized.value().initialization->state, completeLineageIdentities(segment.lidar.lineage),
        completeLineageIdentities(segment.imu_lineage),
        completeLineageIdentities(initialized.value().initialization->lineage)};
  };

  const auto uninterrupted = run(false);
  const auto recovered = run(true);
  ASSERT_TRUE(uninterrupted);
  ASSERT_TRUE(recovered);
  const auto shifted_only_by_request = [](const CompleteLineageIdentities& rejected_run,
                                          const CompleteLineageIdentities& baseline) {
    EXPECT_EQ(rejected_run.lineage, baseline.lineage + 1U);
    EXPECT_EQ(rejected_run.consumer, baseline.consumer + 1U);
    EXPECT_EQ(rejected_run.factor_group, baseline.factor_group);
    EXPECT_EQ(rejected_run.correlation_group, baseline.correlation_group);
  };
  EXPECT_EQ(recovered->initialization_state, uninterrupted->initialization_state);
  shifted_only_by_request(recovered->lidar, uninterrupted->lidar);
  shifted_only_by_request(recovered->imu, uninterrupted->imu);
  shifted_only_by_request(recovered->initialization, uninterrupted->initialization);
  EXPECT_EQ(recovered->initialization_state, core::StateId{10U});
  EXPECT_EQ(uninterrupted->lidar, (CompleteLineageIdentities{5U, 5U, 1U, 1U}));
  EXPECT_EQ(uninterrupted->imu, (CompleteLineageIdentities{6U, 6U, 2U, 0U}));
  EXPECT_EQ(uninterrupted->initialization, (CompleteLineageIdentities{7U, 7U, 3U, 0U}));
}

TEST(LocalEstimator, RejectsWrongCalibrationWithoutMutatingPendingQueue) {
  auto created = LocalEstimator::create(calibration(), estimatorConfig());
  ASSERT_TRUE(created);
  auto estimator = std::move(created).value();
  auto bad = sweep(10'000U, 1'000 * kMillisecond);
  bad.header.direct_calibration = core::CalibrationEpoch{9U};
  auto rejected = estimator.enqueueLidar(std::move(bad));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalEstimatorErrorCode::LidarRejected);
  EXPECT_EQ(estimator.statistics().lidar_sweeps_enqueued, 0U);
}

}  // namespace
}  // namespace meridian::local
