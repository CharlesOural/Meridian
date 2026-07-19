#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <utility>

#include "meridian/local/local_estimator.hpp"

namespace meridian::local {
namespace {

constexpr std::int64_t kMillisecond = 1'000'000LL;
constexpr std::int64_t kImuPeriod = 5 * kMillisecond;

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

[[nodiscard]] core::ImuSample stationaryImu(std::int64_t time_ns, std::uint64_t id) {
  core::ImuSample sample;
  sample.header = header();
  sample.id = core::MeasurementId{id};
  sample.stamp = stamp(time_ns, id);
  sample.specific_force_mps2 = Eigen::Vector3d{0.0, 0.0, 9.80665};
  return sample;
}

void appendImuThrough(LocalEstimator& estimator, std::int64_t begin_ns, std::int64_t end_ns,
                      std::uint64_t& next_id) {
  for (std::int64_t time = begin_ns; time <= end_ns; time += kImuPeriod) {
    const auto ingested = estimator.ingestImu(stationaryImu(time, next_id++));
    ASSERT_TRUE(ingested) << ingested.error().detail;
  }
}

[[nodiscard]] core::LidarSweep sweep(std::uint64_t id, std::int64_t start_ns) {
  constexpr std::uint32_t kWidth = 32U;
  constexpr std::uint32_t kHeight = 8U;
  constexpr std::int64_t kDuration = 90 * kMillisecond;
  auto points = std::make_shared<core::LidarPoints>();
  points->reserve(static_cast<std::size_t>(kWidth) * kHeight);
  for (std::uint32_t row = 0U; row < kHeight; ++row) {
    const double elevation =
        -0.22 + 0.44 * static_cast<double>(row) / static_cast<double>(kHeight - 1U);
    for (std::uint32_t column = 0U; column < kWidth; ++column) {
      const double azimuth =
          2.0 * std::numbers::pi * static_cast<double>(column) / static_cast<double>(kWidth);
      const double range = 7.0 + 0.25 * std::sin(3.0 * azimuth) + 0.08 * row;
      core::LidarPoint point;
      point.x = static_cast<float>(range * std::cos(elevation) * std::cos(azimuth));
      point.y = static_cast<float>(range * std::cos(elevation) * std::sin(azimuth));
      point.z = static_cast<float>(range * std::sin(elevation));
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

[[nodiscard]] LocalEstimatorConfig estimatorConfig() {
  LocalEstimatorConfig config;
  config.odom_epoch = core::OdomEpoch{3U};
  config.first_state = core::StateId{10U};
  config.initialization.mode = InitializationMode::StaticOnly;
  config.initialization.zero_motion_prior =
      ZeroMotionPrior{config.odom_epoch, ZeroMotionPriorSource::MissionScenario};
  config.maximum_pending_imu_guards = 16U;
  config.stationary_initializer.minimum_support = core::Duration{500 * kMillisecond};
  config.stationary_retry_period = core::Duration{50 * kMillisecond};
  config.graph.maximum_navigation_states = 8U;
  config.graph.target_fixed_lag = core::Duration{500 * kMillisecond};
  config.state_timeline.maximum_navigation_states = 16U;
  config.state_timeline.maximum_retained_requests = 32U;
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

void initializeStationaryEstimator(LocalEstimator& estimator, std::uint64_t& imu_id) {
  appendImuThrough(estimator, 0, 500 * kMillisecond, imu_id);
  const auto initialized = estimator.processReady();
  ASSERT_TRUE(initialized) << initialized.error().detail;
  ASSERT_TRUE(initialized.value().initialization);
  EXPECT_EQ(initialized.value().initialization->state, core::StateId{10U});
  EXPECT_EQ(estimator.lifecycle(), LocalEstimatorLifecycle::Tracking);
}

TEST(LocalEstimatorImuSpine, PendingCapacityIsExplicitAndRejectedRequestsDoNotConsumeIdentity) {
  LocalEstimatorConfig config = estimatorConfig();
  config.maximum_pending_imu_guards = 2U;
  auto created = LocalEstimator::create(calibration(), config);
  ASSERT_TRUE(created) << created.error().detail;
  LocalEstimator estimator = std::move(created).value();

  const auto before_initialization =
      estimator.enqueueImuGuard(core::FusionTime{600 * kMillisecond});
  ASSERT_FALSE(before_initialization);
  EXPECT_EQ(before_initialization.error().code, LocalEstimatorErrorCode::ImuGuardRejected);

  std::uint64_t imu_id = 1U;
  initializeStationaryEstimator(estimator, imu_id);
  appendImuThrough(estimator, 505 * kMillisecond, 800 * kMillisecond, imu_id);

  const auto first = estimator.enqueueImuGuard(core::FusionTime{600 * kMillisecond});
  const auto second = estimator.enqueueImuGuard(core::FusionTime{700 * kMillisecond});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first.value().request, core::KnotRequestId{1U});
  EXPECT_EQ(second.value().request, core::KnotRequestId{2U});
  EXPECT_EQ(second.value().pending_guards, 2U);

  const auto full = estimator.enqueueImuGuard(core::FusionTime{800 * kMillisecond});
  ASSERT_FALSE(full);
  EXPECT_EQ(full.error().code, LocalEstimatorErrorCode::PendingImuGuardCapacity);

  const auto processed = estimator.processReady();
  ASSERT_TRUE(processed) << processed.error().detail;
  ASSERT_EQ(processed.value().imu_guard_commits.size(), 2U);
  EXPECT_EQ(processed.value().pending_imu_guards, 0U);

  const auto retried = estimator.enqueueImuGuard(core::FusionTime{800 * kMillisecond});
  ASSERT_TRUE(retried);
  EXPECT_EQ(retried.value().request, core::KnotRequestId{3U});
  EXPECT_EQ(estimator.statistics().imu_guard_enqueue_rejections, 2U);
  EXPECT_EQ(estimator.statistics().imu_guard_capacity_rejections, 1U);
  EXPECT_EQ(estimator.statistics().imu_guard_pending_high_watermark, 2U);
}

TEST(LocalEstimatorImuSpine, LongStationaryGuardChainUsesNormalGraphAndStaysFixedLagBounded) {
  auto created = LocalEstimator::create(calibration(), estimatorConfig());
  ASSERT_TRUE(created) << created.error().detail;
  LocalEstimator estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  initializeStationaryEstimator(estimator, imu_id);
  appendImuThrough(estimator, 505 * kMillisecond, 4'000 * kMillisecond, imu_id);

  constexpr std::size_t kGuards = 32U;
  std::size_t finalized_states = 0U;
  for (std::size_t index = 0U; index < kGuards; ++index) {
    const core::FusionTime exact_time{(603 + static_cast<std::int64_t>(100U * index)) *
                                      kMillisecond};
    const auto enqueued = estimator.enqueueImuGuard(exact_time);
    ASSERT_TRUE(enqueued) << enqueued.error().detail;
    EXPECT_EQ(enqueued.value().request, core::KnotRequestId{index + 1U});

    const auto processed = estimator.processReady();
    ASSERT_TRUE(processed) << processed.error().detail;
    ASSERT_EQ(processed.value().imu_guard_commits.size(), 1U);
    const ImuGuardCommitReport& guard = processed.value().imu_guard_commits.front();
    ASSERT_EQ(guard.resolutions.size(), 1U);
    EXPECT_EQ(guard.resolutions.front().request, enqueued.value().request);
    EXPECT_EQ(guard.commit.state_time, exact_time);
    EXPECT_EQ(guard.resolutions.front().state, core::StateId{11U + index});
    EXPECT_EQ(guard.resolutions.front().created_at_revision, core::LocalGraphRevision{2U + index});
    EXPECT_EQ(guard.commit.solve.navigation_states <=
                  estimator.effectiveConfig().graph.maximum_navigation_states,
              true);
    // This count is cumulative across marginalization. One new normal
    // CombinedImuFactor must be constructed for every explicit guard.
    EXPECT_EQ(guard.commit.solve.combined_imu_factors, index + 1U);
    EXPECT_NEAR(guard.commit.estimate.T_odom_imu.translation().norm(), 0.0, 1.0e-6);
    EXPECT_NEAR(guard.commit.estimate.velocity_odom.norm(), 0.0, 1.0e-6);
    finalized_states += guard.commit.finalized_states.size();
  }

  EXPECT_GT(finalized_states, 0U);
  EXPECT_EQ(estimator.statistics().imu_guard_knots_committed, kGuards);
  EXPECT_EQ(estimator.statistics().imu_guard_requests_resolved, kGuards);
  EXPECT_EQ(estimator.statistics().graph_commits, kGuards + 1U);
  const auto estimate = estimator.estimate();
  ASSERT_TRUE(estimate);
  EXPECT_EQ(estimate.value().state, core::StateId{10U + kGuards});
  EXPECT_EQ(estimate.value().revision, core::LocalGraphRevision{1U + kGuards});
  EXPECT_LE(estimate.value().solve.navigation_states,
            estimator.effectiveConfig().graph.maximum_navigation_states);
}

TEST(LocalEstimatorImuSpine, TimelineHeadroomAdmitsStateThatTriggersGraphCapFinality) {
  LocalEstimatorConfig config = estimatorConfig();
  config.graph.maximum_navigation_states = 3U;
  config.graph.target_fixed_lag = core::Duration{10'000 * kMillisecond};
  config.state_timeline.maximum_navigation_states = 4U;
  config.state_timeline.maximum_retained_requests = 4U;
  auto created = LocalEstimator::create(calibration(), std::move(config));
  ASSERT_TRUE(created) << created.error().detail;
  LocalEstimator estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  initializeStationaryEstimator(estimator, imu_id);
  appendImuThrough(estimator, 505 * kMillisecond, 1'000 * kMillisecond, imu_id);

  const std::array<std::int64_t, 4U> guard_times_ms{600, 700, 800, 900};
  for (std::size_t index = 0U; index < guard_times_ms.size(); ++index) {
    const core::FusionTime exact_time{guard_times_ms[index] * kMillisecond};
    const auto enqueued = estimator.enqueueImuGuard(exact_time);
    ASSERT_TRUE(enqueued) << enqueued.error().detail;
    const auto processed = estimator.processReady();
    ASSERT_TRUE(processed) << processed.error().detail;
    ASSERT_EQ(processed.value().imu_guard_commits.size(), 1U);
    const LocalGraphCommit& commit = processed.value().imu_guard_commits.front().commit;
    EXPECT_EQ(commit.state_time, exact_time);
    EXPECT_LE(commit.solve.navigation_states, 3U);
    if (index < 2U) {
      EXPECT_TRUE(commit.finalized_states.empty());
    } else {
      ASSERT_EQ(commit.finalized_states.size(), 1U);
      EXPECT_EQ(commit.finalized_states.front().state, core::StateId{10U + index - 2U});
    }
  }

  EXPECT_EQ(estimator.statistics().imu_guard_knots_committed, guard_times_ms.size());
  EXPECT_EQ(estimator.statistics().graph_commits, guard_times_ms.size() + 1U);
}

TEST(LocalEstimatorImuSpine, GuardSharesExactLidarAndEarlierGuardSuppressesOnlyCloseLidar) {
  LocalEstimatorConfig config = estimatorConfig();
  // This test exercises an explicit 100 ms coalescing policy. The production
  // default is only a 1 ms numerical guard and intentionally admits the
  // 10 ms-separated requests below as distinct states.
  config.state_timeline.minimum_state_interval = core::Duration{100 * kMillisecond};
  auto created = LocalEstimator::create(calibration(), std::move(config));
  ASSERT_TRUE(created) << created.error().detail;
  LocalEstimator estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  initializeStationaryEstimator(estimator, imu_id);

  appendImuThrough(estimator, 505 * kMillisecond, 700 * kMillisecond, imu_id);
  const auto shared_guard = estimator.enqueueImuGuard(core::FusionTime{690 * kMillisecond});
  ASSERT_TRUE(shared_guard);
  ASSERT_TRUE(estimator.enqueueLidar(sweep(10'000U, 600 * kMillisecond)));
  const auto shared = estimator.processReady();
  ASSERT_TRUE(shared) << shared.error().detail;
  ASSERT_EQ(shared.value().commits.size(), 1U);
  ASSERT_EQ(shared.value().imu_guard_commits.size(), 1U);
  ASSERT_EQ(shared.value().imu_guard_commits.front().resolutions.size(), 1U);
  const ImuGuardResolution& resolution =
      shared.value().imu_guard_commits.front().resolutions.front();
  EXPECT_EQ(resolution.state, shared.value().commits.front().commit.state);
  EXPECT_EQ(resolution.created_at_revision, shared.value().commits.front().commit.revision);
  ASSERT_EQ(resolution.exactly_shared_requests.size(), 2U);
  EXPECT_EQ(resolution.exactly_shared_requests.front(), core::KnotRequestId{1U});
  EXPECT_EQ(resolution.exactly_shared_requests.back(), core::KnotRequestId{2U});

  appendImuThrough(estimator, 705 * kMillisecond, 900 * kMillisecond, imu_id);
  const auto guard = estimator.enqueueImuGuard(core::FusionTime{900 * kMillisecond});
  ASSERT_TRUE(guard);
  EXPECT_EQ(guard.value().request, core::KnotRequestId{3U});
  const auto lidar = estimator.enqueueLidar(sweep(10'001U, 800 * kMillisecond));
  ASSERT_TRUE(lidar);
  EXPECT_EQ(lidar.value().state_admission, StateAdmissionDisposition::SuppressedTooClose);
  const auto processed = estimator.processReady();
  ASSERT_TRUE(processed) << processed.error().detail;
  EXPECT_TRUE(processed.value().commits.empty());
  ASSERT_EQ(processed.value().dropped_sweeps.size(), 1U);
  EXPECT_EQ(processed.value().dropped_sweeps.front().reason,
            LidarDropReason::StateRequestSuppressedTooClose);
  ASSERT_EQ(processed.value().imu_guard_commits.size(), 1U);
  EXPECT_EQ(processed.value().imu_guard_commits.front().exact_time,
            core::FusionTime{900 * kMillisecond});
  EXPECT_EQ(estimator.statistics().lidar_state_requests_suppressed_by_timeline, 1U);

  const auto next = estimator.enqueueImuGuard(core::FusionTime{1'000 * kMillisecond});
  ASSERT_TRUE(next);
  EXPECT_EQ(next.value().request, core::KnotRequestId{5U});
}

TEST(LocalEstimatorImuSpine, ReadOnlyPropagationReportsReplayMissingTicksAndHardGaps) {
  auto created = LocalEstimator::create(calibration(), estimatorConfig());
  ASSERT_TRUE(created) << created.error().detail;
  LocalEstimator estimator = std::move(created).value();
  std::uint64_t imu_id = 1U;
  initializeStationaryEstimator(estimator, imu_id);
  const auto anchor = estimator.estimate();
  ASSERT_TRUE(anchor);

  const auto replay = estimator.propagateTo(anchor.value().state_time);
  ASSERT_TRUE(replay) << replay.error().detail;
  EXPECT_EQ(replay.value().gap_status, ImuPropagationGapStatus::AnchorOnly);
  EXPECT_TRUE(replay.value().raw_imu_support.empty());
  EXPECT_EQ(replay.value().anchor_state, anchor.value().state);
  EXPECT_EQ(replay.value().anchor_revision, anchor.value().revision);
  EXPECT_TRUE(replay.value().propagated_state.T_odom_imu.matrix().isApprox(
      anchor.value().estimate.T_odom_imu.matrix(), 0.0));

  ASSERT_TRUE(estimator.ingestImu(stationaryImu(505 * kMillisecond, imu_id++)));
  ASSERT_TRUE(estimator.ingestImu(stationaryImu(515 * kMillisecond, imu_id++)));
  const auto missing_tick = estimator.propagateTo(core::FusionTime{515 * kMillisecond});
  ASSERT_TRUE(missing_tick) << missing_tick.error().detail;
  EXPECT_EQ(missing_tick.value().gap_status, ImuPropagationGapStatus::InferredMissingTicks);
  EXPECT_EQ(missing_tick.value().inferred_missing_ticks, 1U);
  EXPECT_EQ(missing_tick.value().maximum_raw_gap, core::Duration{10 * kMillisecond});
  EXPECT_FALSE(missing_tick.value().contains_saturation);
  EXPECT_NEAR(missing_tick.value().propagated_state.T_odom_imu.translation().norm(), 0.0, 1.0e-12);
  EXPECT_NEAR(missing_tick.value().propagated_state.velocity_odom.norm(), 0.0, 1.0e-12);

  const auto repeated = estimator.propagateTo(core::FusionTime{515 * kMillisecond});
  ASSERT_TRUE(repeated);
  EXPECT_EQ(repeated.value().raw_imu_support, missing_tick.value().raw_imu_support);
  EXPECT_TRUE(repeated.value().propagated_state.T_odom_imu.matrix().isApprox(
      missing_tick.value().propagated_state.T_odom_imu.matrix(), 0.0));

  ASSERT_TRUE(estimator.ingestImu(stationaryImu(530 * kMillisecond, imu_id++)));
  const auto hard_gap = estimator.propagateTo(core::FusionTime{530 * kMillisecond});
  ASSERT_FALSE(hard_gap);
  EXPECT_EQ(hard_gap.error().code, LocalEstimatorErrorCode::ImuSupportFailed);
  ASSERT_TRUE(hard_gap.error().imu_buffer_error_code);
  EXPECT_EQ(*hard_gap.error().imu_buffer_error_code, ImuBufferErrorCode::EpochBreakingGap);

  const auto unchanged = estimator.estimate();
  ASSERT_TRUE(unchanged);
  EXPECT_EQ(unchanged.value().state, anchor.value().state);
  EXPECT_EQ(unchanged.value().revision, anchor.value().revision);
  EXPECT_TRUE(unchanged.value().estimate.T_odom_imu.matrix().isApprox(
      anchor.value().estimate.T_odom_imu.matrix(), 0.0));
}

}  // namespace
}  // namespace meridian::local
