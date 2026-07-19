#include <gtest/gtest.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <boost/make_shared.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "meridian/local/local_estimator.hpp"

namespace meridian::local {
namespace {

constexpr double kGravity = 9.80665;
constexpr std::int64_t kMillisecond = 1'000'000LL;
constexpr std::int64_t kImuPeriod = 5 * kMillisecond;
constexpr std::int64_t kScanDuration = 90 * kMillisecond;
constexpr std::int64_t kSegmentDuration = 125 * kMillisecond;
constexpr std::size_t kStepsPerSegment = static_cast<std::size_t>(kSegmentDuration / kImuPeriod);
constexpr std::size_t kSegments = 10U;
constexpr std::size_t kContinuationScans = 1U;
constexpr std::uint64_t kFirstLidarId = 10'000U;
constexpr std::uint64_t kFirstStateId = 40U;
// Geometry-preserving LiDAR information retains rotational lever arm instead
// of forcing an isotropic physical spectrum. The synthetic batch remains
// tightly bounded, but its coupled velocity solution differs from the former
// per-axis-clipped objective.
constexpr double kMaximumInitializationVelocityErrorMps = 0.20;

struct SyntheticScenario {
  std::vector<core::ImuSample> imu;
  std::vector<core::LidarSweep> lidar;
  core::Pose3d T_imu_lidar;
  core::NavStateEstimate expected_final;
  core::NavStateEstimate expected_tracking_final;
  Eigen::Vector3d expected_accel_bias;
  Eigen::Vector3d expected_gyro_bias;
};

struct RunResult {
  LocalEstimatorProcessReport final_report;
  LocalGraphCommit estimate;
  LocalEstimatorStatistics statistics;
  std::vector<LidarBootstrapCommit> bootstrap;
  std::vector<LidarCommitReport> tracking_commits;
  std::optional<int> stationary_rejection_code;
};

struct RunOutcome {
  std::optional<RunResult> value;
  std::string error;
};

[[nodiscard]] core::RecordHeader header() {
  core::RecordHeader result;
  result.trace = core::TraceId{7U};
  result.producer = core::ProducerId{3U};
  result.session = core::SessionId{11U};
  result.config = core::ConfigRevision{5U};
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

[[nodiscard]] core::CalibrationBundle calibration(const core::Pose3d& T_imu_lidar) {
  core::ImuNoiseModel noise(0.02, 0.0015, 0.0002, 0.00002);
  core::ImuCalibration imu("synthetic_imu", "/imu", core::ImuSensorModel::Generic, 200.0, kGravity,
                           noise);
  core::LidarCalibration lidar(
      core::LidarId{1U}, "synthetic_lidar", "/points", core::ImuFromLidarTransform{T_imu_lidar},
      core::LidarTimingCalibration{core::LidarSweepTimestampReference::SweepStart,
                                   core::LidarPointTimeConvention::OffsetFromSweepTimestamp});
  auto created = core::CalibrationBundle::create(core::CalibrationEpoch{1U}, std::move(imu),
                                                 core::BaseFromImuTransform{core::Pose3d{}},
                                                 std::move(lidar), {});
  if (!created) {
    throw std::runtime_error("synthetic calibration is invalid");
  }
  return std::move(created).value();
}

[[nodiscard]] Eigen::Vector3d unbiasedSpecificForce(double time_seconds) {
  // These are valid body-frame measurements of a dynamically accelerating
  // rigid body. The corresponding world acceleration is R*f + g and is
  // integrated below with the same exact sample boundaries admitted by the
  // estimator.
  return Eigen::Vector3d{0.9 * std::sin(2.3 * time_seconds) + 0.35 * std::cos(5.1 * time_seconds),
                         0.7 * std::cos(1.7 * time_seconds) - 0.25 * std::sin(4.2 * time_seconds),
                         kGravity + 0.65 * std::sin(2.9 * time_seconds)};
}

[[nodiscard]] Eigen::Vector3d unbiasedAngularRate(double time_seconds) {
  return Eigen::Vector3d{0.20 + 0.09 * std::sin(2.1 * time_seconds),
                         -0.14 + 0.08 * std::cos(1.6 * time_seconds),
                         0.24 + 0.07 * std::sin(2.8 * time_seconds)};
}

[[nodiscard]] core::Pose3d corePose(const gtsam::Pose3& pose) {
  return core::Pose3d{Sophus::SO3d{pose.rotation().matrix()}, pose.translation()};
}

[[nodiscard]] core::Pose3d interpolatedPose(const std::vector<gtsam::NavState>& states,
                                            std::int64_t time_ns) {
  const std::size_t left = static_cast<std::size_t>(time_ns / kImuPeriod);
  if (left + 1U >= states.size() || time_ns % kImuPeriod == 0LL) {
    return corePose(states.at(left).pose());
  }
  const double alpha = static_cast<double>(time_ns - static_cast<std::int64_t>(left) * kImuPeriod) /
                       static_cast<double>(kImuPeriod);
  const core::Pose3d first = corePose(states[left].pose());
  const core::Pose3d second = corePose(states[left + 1U].pose());
  return first * core::Pose3d::exp(alpha * (first.inverse() * second).log());
}

[[nodiscard]] core::LidarSweep organizedSweep(std::uint64_t id, std::int64_t start_ns,
                                              const std::vector<gtsam::NavState>& states,
                                              const core::Pose3d& T_imu_lidar) {
  constexpr std::uint32_t kWidth = 72U;
  constexpr std::uint32_t kRowsPerPlane = 10U;
  constexpr std::uint32_t kHeight = 3U * kRowsPerPlane;
  auto points = std::make_shared<core::LidarPoints>();
  points->reserve(static_cast<std::size_t>(kWidth) * kHeight);
  // Three fixed, asymmetric plane patches are observed by every scan. Their
  // sampling provides multiple preprocessed points per 1 m target voxel, so
  // the production point support and two-dimensional observability gates are tested
  // instead of bypassed. Each point is transformed by the true body pose at
  // its own acquisition timestamp; constructing every point at scan end would
  // not test deskew.
  for (std::uint32_t row = 0U; row < kHeight; ++row) {
    const std::uint32_t plane = row / kRowsPerPlane;
    const std::uint32_t plane_row = row % kRowsPerPlane;
    const double narrow_axis = -0.63 + 0.14 * static_cast<double>(plane_row);
    for (std::uint32_t column = 0U; column < kWidth; ++column) {
      const double long_axis = 3.27 + 0.14 * static_cast<double>(column);
      Eigen::Vector3d point_bootstrap;
      if (plane == 0U) {
        point_bootstrap = Eigen::Vector3d{long_axis, narrow_axis - 0.41, -1.73};
      } else if (plane == 1U) {
        point_bootstrap = Eigen::Vector3d{long_axis - 0.23, 3.37, narrow_axis - 0.18};
      } else {
        point_bootstrap = Eigen::Vector3d{9.41, long_axis - 8.24, narrow_axis + 0.29};
      }
      const std::int64_t offset_ns = (static_cast<std::int64_t>(column) * (kScanDuration - 1LL)) /
                                     static_cast<std::int64_t>(kWidth - 1U);
      const core::Pose3d T_odom_imu = interpolatedPose(states, start_ns + offset_ns);
      const Eigen::Vector3d point_imu = T_odom_imu.inverse() * point_bootstrap;
      const Eigen::Vector3d point_lidar = T_imu_lidar.inverse() * point_imu;

      core::LidarPoint point;
      point.x = static_cast<float>(point_lidar.x());
      point.y = static_cast<float>(point_lidar.y());
      point.z = static_cast<float>(point_lidar.z());
      point.intensity = static_cast<float>(3U * row + column);
      point.ring = static_cast<std::uint16_t>(row);
      point.source_index = row * kWidth + column;
      point.time_offset_ns = static_cast<std::int32_t>(offset_ns);
      points->push_back(point);
    }
  }

  core::LidarSweep result;
  result.header = header();
  result.id = core::MeasurementId{id};
  result.lidar = core::LidarId{1U};
  result.stamp = stamp(start_ns, id);
  result.acquisition =
      core::TimeRange{core::FusionTime{start_ns}, core::FusionTime{start_ns + kScanDuration}};
  result.layout = core::LidarLayout{kWidth, kHeight, true};
  result.points = std::move(points);
  return result;
}

[[nodiscard]] SyntheticScenario scenario(core::Pose3d T_imu_lidar = {}) {
  SyntheticScenario result;
  result.T_imu_lidar = std::move(T_imu_lidar);
  result.expected_accel_bias = Eigen::Vector3d{0.08, -0.05, 0.04};
  result.expected_gyro_bias = Eigen::Vector3d{0.012, -0.008, 0.006};
  const gtsam::imuBias::ConstantBias true_bias(result.expected_accel_bias,
                                               result.expected_gyro_bias);

  constexpr std::int64_t kFinalTime =
      static_cast<std::int64_t>(kSegments + kContinuationScans) * kSegmentDuration + kScanDuration;
  constexpr std::size_t kImuSamples = static_cast<std::size_t>(kFinalTime / kImuPeriod) + 1U;
  result.imu.reserve(kImuSamples);
  for (std::size_t index = 0U; index < kImuSamples; ++index) {
    const std::int64_t time_ns = static_cast<std::int64_t>(index) * kImuPeriod;
    const double time_seconds = static_cast<double>(time_ns) * 1.0e-9;
    const std::uint64_t id = 1U + index;
    core::ImuSample sample;
    sample.header = header();
    sample.id = core::MeasurementId{id};
    sample.stamp = stamp(time_ns, id);
    sample.specific_force_mps2 = unbiasedSpecificForce(time_seconds) + result.expected_accel_bias;
    sample.angular_velocity_radps = unbiasedAngularRate(time_seconds) + result.expected_gyro_bias;
    result.imu.push_back(std::move(sample));
  }

  auto parameters =
      boost::make_shared<gtsam::PreintegrationParams>(Eigen::Vector3d{0.0, 0.0, -kGravity});
  parameters->accelerometerCovariance = Eigen::Matrix3d::Identity() * std::pow(0.02, 2);
  parameters->gyroscopeCovariance = Eigen::Matrix3d::Identity() * std::pow(0.0015, 2);
  parameters->integrationCovariance = Eigen::Matrix3d::Identity() * std::pow(1.0e-8, 2);

  std::vector<gtsam::NavState> states;
  states.reserve(kImuSamples);
  states.emplace_back(gtsam::Pose3{}, gtsam::Vector3{1.1, -0.45, 0.25});
  for (std::size_t index = 1U; index < result.imu.size(); ++index) {
    gtsam::PreintegratedImuMeasurements preintegrated(parameters, true_bias);
    const auto& previous = result.imu[index - 1U];
    const auto& current = result.imu[index];
    const double dt =
        static_cast<double>((current.stamp.fusion_time - previous.stamp.fusion_time).nanoseconds) *
        1.0e-9;
    preintegrated.integrateMeasurement(
        0.5 * (previous.specific_force_mps2 + current.specific_force_mps2),
        0.5 * (previous.angular_velocity_radps + current.angular_velocity_radps), dt);
    states.push_back(preintegrated.predict(states.back(), true_bias));
  }

  result.lidar.reserve(kSegments + kContinuationScans + 1U);
  for (std::size_t scan = 0U; scan <= kSegments + kContinuationScans; ++scan) {
    const std::int64_t scan_start = static_cast<std::int64_t>(scan) * kSegmentDuration;
    result.lidar.push_back(
        organizedSweep(kFirstLidarId + scan, scan_start, states, result.T_imu_lidar));
  }

  const std::size_t first_reference_index = static_cast<std::size_t>(kScanDuration / kImuPeriod);
  const core::Pose3d first_reference = corePose(states[first_reference_index].pose());
  const double first_yaw =
      std::atan2(first_reference.so3().matrix()(1, 0), first_reference.so3().matrix()(0, 0));
  const Sophus::SO3d R_gauge_world = Sophus::SO3d::exp(Eigen::Vector3d{0.0, 0.0, -first_yaw});
  const std::int64_t initialization_time =
      static_cast<std::int64_t>(kSegments) * kSegmentDuration + kScanDuration;
  const std::size_t initialization_index =
      static_cast<std::size_t>(initialization_time / kImuPeriod);
  const core::Pose3d final_world = corePose(states[initialization_index].pose());
  result.expected_final.T_odom_imu =
      core::Pose3d(R_gauge_world * final_world.so3(),
                   R_gauge_world * (final_world.translation() - first_reference.translation()));
  result.expected_final.velocity_odom = R_gauge_world * states[initialization_index].velocity();
  result.expected_final.accel_bias = result.expected_accel_bias;
  result.expected_final.gyro_bias = result.expected_gyro_bias;
  const core::Pose3d tracking_world = corePose(states.back().pose());
  result.expected_tracking_final.T_odom_imu =
      core::Pose3d(R_gauge_world * tracking_world.so3(),
                   R_gauge_world * (tracking_world.translation() - first_reference.translation()));
  result.expected_tracking_final.velocity_odom = R_gauge_world * states.back().velocity();
  result.expected_tracking_final.accel_bias = result.expected_accel_bias;
  result.expected_tracking_final.gyro_bias = result.expected_gyro_bias;
  return result;
}

[[nodiscard]] LocalEstimatorConfig estimatorConfig() {
  LocalEstimatorConfig config;
  config.odom_epoch = core::OdomEpoch{9U};
  config.first_state = core::StateId{kFirstStateId};
  // Exercise supervised fallback explicitly: the scenario assertion is
  // intentionally contradicted by the dynamic synthetic IMU sequence.
  config.initialization.mode = InitializationMode::SupervisedAuto;
  config.initialization.zero_motion_prior =
      ZeroMotionPrior{config.odom_epoch, ZeroMotionPriorSource::MissionScenario};
  config.maximum_pending_lidar_sweeps = kSegments + 2U;
  // The moving-initialization window exactly fills the timeline. The first
  // post-root sensor/guard request can succeed only if root publication
  // retires every stale bootstrap reservation strictly before the root.
  config.graph.maximum_navigation_states = kSegments;
  config.state_timeline.maximum_navigation_states = kSegments + 1U;
  config.state_timeline.maximum_retained_requests = kSegments + 1U;
  config.stationary_initializer.minimum_support = core::Duration{1'000 * kMillisecond};
  // One rejection is sufficient for this bounded bootstrap sequence; avoid
  // conflating retry-policy testing with motion initialization.
  config.stationary_retry_period = core::Duration{10'000 * kMillisecond};

  // This synthetic calibration has a known startup-bias mean. Verify that
  // the coordinator forwards the calibrated mean into the one shared moving-
  // batch prior instead of silently replacing it with zero.
  config.graph.imu.initial_accelerometer_bias_mean_mps2 = Eigen::Vector3d{0.08, -0.05, 0.04};
  config.graph.imu.initial_gyroscope_bias_mean_radps = Eigen::Vector3d{0.012, -0.008, 0.006};

  config.lidar_bootstrap.maximum_sweeps = kSegments + 2U;
  config.lidar_bootstrap.preprocessing.minimum_range_m = 0.5;
  config.lidar_bootstrap.preprocessing.maximum_range_m = 30.0;
  config.lidar_bootstrap.preprocessing.voxel_size_m = 0.15;
  config.lidar_bootstrap.preprocessing.maximum_output_points = 4'000U;
  config.lidar_bootstrap.registration.target_voxel_resolution_m = 1.0;
  config.lidar_bootstrap.registration.maximum_target_points_per_target = 4'000U;
  config.lidar_bootstrap.registration.maximum_composite_points_per_voxel = 64U;
  config.lidar_bootstrap.registration.minimum_correspondences = 30U;
  config.lidar_bootstrap.registration.residual_standard_deviation_m = 0.05;
  config.lidar_bootstrap.registration.absolute_normalized_observable_eigenvalue = 1.0e-8;
  config.lidar_bootstrap.registration.relative_normalized_observable_eigenvalue = 1.0e-6;
  config.lidar_bootstrap.registration.minimum_observable_rank = 4U;
  config.lidar_bootstrap.registration.translation_convergence_m = 1.0e-5;
  config.lidar_bootstrap.registration.rotation_convergence_rad = 1.0e-5;
  config.lidar_bootstrap.maximum_increment_translation_m = 3.0;
  config.lidar_bootstrap.maximum_increment_rotation_rad = 0.8;
  config.lidar_bootstrap.maximum_observable_condition = 1.0e10;

  config.motion_initializer.minimum_segments = kSegments;
  config.motion_initializer.maximum_segments = kSegments;
  config.motion_initializer.minimum_support = core::Duration{1'200 * kMillisecond};
  config.motion_initializer.maximum_support = core::Duration{1'500 * kMillisecond};
  config.motion_initializer.maximum_raw_imu_gap = core::Duration{2 * kImuPeriod};
  return config;
}

[[nodiscard]] RunOutcome runScenario(const SyntheticScenario& synthetic,
                                     std::size_t scans_to_process = kSegments + 1U) {
  auto created = LocalEstimator::create(calibration(synthetic.T_imu_lidar), estimatorConfig());
  if (!created) {
    return RunOutcome{std::nullopt, "create failed: " + created.error().detail};
  }
  auto estimator = std::move(created).value();
  RunResult result;
  std::string last_rejection;

  std::size_t next_imu = 0U;
  for (std::size_t scan = 0U; scan < scans_to_process; ++scan) {
    while (next_imu < synthetic.imu.size() &&
           synthetic.imu[next_imu].stamp.fusion_time <= synthetic.lidar[scan].acquisition.end) {
      auto appended = estimator.ingestImu(synthetic.imu[next_imu]);
      if (!appended) {
        return RunOutcome{std::nullopt, "IMU ingest failed: " + appended.error().detail};
      }
      ++next_imu;
    }
    auto queued = estimator.enqueueLidar(synthetic.lidar[scan]);
    if (!queued) {
      return RunOutcome{std::nullopt, "LiDAR enqueue failed: " + queued.error().detail};
    }
    auto ready = estimator.processReady();
    if (!ready) {
      return RunOutcome{std::nullopt, "process failed: " + ready.error().detail};
    }
    for (const auto& entry : ready.value().bootstrap) {
      if (entry.commit) {
        result.bootstrap.push_back(*entry.commit);
      }
      if (entry.rejection) {
        last_rejection = entry.rejection->detail;
      }
    }
    if (ready.value().initialization_rejection &&
        ready.value().initialization_rejection->stage ==
            LocalInitializationRejectionStage::StationaryTest) {
      result.stationary_rejection_code = ready.value().initialization_rejection->code;
    }
    if (ready.value().initialization_rejection) {
      last_rejection = ready.value().initialization_rejection->detail;
    }
    for (const LidarCommitReport& commit : ready.value().commits) {
      result.tracking_commits.push_back(commit);
    }
    if (ready.value().initialization) {
      result.final_report = std::move(ready).value();
    }
  }

  const auto estimate = estimator.estimate();
  if (!estimate) {
    return RunOutcome{
        std::nullopt,
        "final estimate failed: " + estimate.error().detail + "; last rejection: " +
            last_rejection + "; bootstrap commits=" + std::to_string(result.bootstrap.size()) +
            ", increments=" + std::to_string(estimator.statistics().lidar_bootstrap_increments) +
            ", rejected=" + std::to_string(estimator.statistics().lidar_bootstrap_rejections)};
  }
  result.estimate = estimate.value();
  result.statistics = estimator.statistics();
  return RunOutcome{std::move(result), {}};
}

[[nodiscard]] std::set<std::uint64_t> measurementRoots(const core::ObservationLineage& lineage) {
  std::set<std::uint64_t> roots;
  for (const auto& usage : lineage.usage) {
    if (const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root)) {
      roots.insert(measurement->value());
    }
  }
  return roots;
}

void expectExactStatistics(const LocalEstimatorStatistics& statistics) {
  constexpr std::size_t kExpectedImuSamples = static_cast<std::size_t>(
      (static_cast<std::int64_t>(kSegments) * kSegmentDuration + kScanDuration) / kImuPeriod + 1LL);
  EXPECT_EQ(statistics.imu_samples_accepted, kExpectedImuSamples);
  EXPECT_EQ(statistics.lidar_sweeps_enqueued, kSegments + 1U);
  EXPECT_EQ(statistics.lidar_sweeps_dropped, 0U);
  EXPECT_EQ(statistics.initialization_attempts, 2U);
  EXPECT_EQ(statistics.initialization_rejections, 1U);
  EXPECT_EQ(statistics.stationary_initialization_attempts, 1U);
  EXPECT_EQ(statistics.stationary_initialization_rejections, 1U);
  EXPECT_EQ(statistics.lidar_bootstrap_anchors, 1U);
  EXPECT_EQ(statistics.lidar_bootstrap_increments, kSegments);
  EXPECT_EQ(statistics.lidar_bootstrap_rejections, 0U);
  EXPECT_EQ(statistics.motion_initialization_attempts, 1U);
  EXPECT_EQ(statistics.motion_initialization_rejections, 0U);
  EXPECT_EQ(statistics.motion_initialization_commits, 1U);
  EXPECT_EQ(statistics.graph_commits, 1U);
  EXPECT_EQ(statistics.lidar_registrations, 0U);
  EXPECT_EQ(statistics.lidar_bootstraps, 0U);
  EXPECT_EQ(statistics.lidar_degraded_commits, 0U);
  EXPECT_EQ(statistics.lidar_rejections_target_retained, 0U);
  EXPECT_EQ(statistics.lidar_target_state_unavailable_freezes, 0U);
}

TEST(LocalEstimatorMovingInitialization,
     RejectsStationaryLaneAndCommitsDeterministicLidarImuBatch) {
  const SyntheticScenario synthetic = scenario();
  const RunOutcome first_outcome = runScenario(synthetic);
  ASSERT_TRUE(first_outcome.value) << first_outcome.error;
  const RunResult& first = *first_outcome.value;

  ASSERT_TRUE(first.stationary_rejection_code);
  EXPECT_EQ(*first.stationary_rejection_code, static_cast<int>(InitializationErrorCode::Moving));
  ASSERT_TRUE(first.final_report.initialization);
  ASSERT_TRUE(first.final_report.initialization_method);
  EXPECT_EQ(*first.final_report.initialization_method, LocalInitializationMethod::MotionLidarImu);
  EXPECT_EQ(first.final_report.lifecycle, LocalEstimatorLifecycle::Tracking);
  EXPECT_EQ(first.final_report.pending_sweeps, 0U);
  EXPECT_FALSE(first.final_report.waiting_for_future_imu);
  EXPECT_TRUE(first.final_report.commits.empty());

  ASSERT_EQ(first.bootstrap.size(), kSegments + 1U);
  EXPECT_EQ(first.bootstrap.front().disposition, LidarBootstrapDisposition::AnchorCreated);
  EXPECT_FALSE(first.bootstrap.front().segment);
  for (std::size_t index = 1U; index < first.bootstrap.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(first.bootstrap[index].disposition, LidarBootstrapDisposition::IncrementCommitted);
    ASSERT_TRUE(first.bootstrap[index].segment);
    ASSERT_TRUE(first.bootstrap[index].registration);
    EXPECT_GE(first.bootstrap[index].registration->diagnostics.observable_rank, 4U);
    EXPECT_GE(first.bootstrap[index].registration->diagnostics.correspondences, 30U);
    EXPECT_GE(first.bootstrap[index].segment->lidar.information.rank, 4U);
    EXPECT_DOUBLE_EQ(first.bootstrap[index].segment->lidar.imu_conditioning_covariance_inflation,
                     4.0);
    EXPECT_DOUBLE_EQ(first.bootstrap[index].segment->lidar.applied_covariance_inflation, 6.0);
    EXPECT_EQ(
        first.bootstrap[index].segment->lidar.start_time,
        core::FusionTime{kScanDuration + static_cast<std::int64_t>(index - 1U) * kSegmentDuration});
    EXPECT_EQ(
        first.bootstrap[index].segment->lidar.end_time,
        core::FusionTime{kScanDuration + static_cast<std::int64_t>(index) * kSegmentDuration});
  }

  const LocalGraphCommit& commit = *first.final_report.initialization;
  EXPECT_EQ(commit.odom_epoch, core::OdomEpoch{9U});
  EXPECT_EQ(commit.state, core::StateId{kFirstStateId});
  EXPECT_EQ(commit.revision, core::LocalGraphRevision{1U});
  EXPECT_EQ(
      commit.state_time,
      core::FusionTime{kScanDuration + static_cast<std::int64_t>(kSegments) * kSegmentDuration});
  EXPECT_EQ(commit.state_time, first.estimate.state_time);
  EXPECT_EQ(commit.solve.navigation_states, 1U);
  EXPECT_EQ(commit.solve.joint_initial_priors, 1U);
  EXPECT_EQ(commit.solve.combined_imu_factors, 0U);
  EXPECT_EQ(commit.solve.active_lidar_direct_batch_factors, 0U);

  ASSERT_TRUE(first.final_report.motion_initialization_diagnostics);
  const MotionInitializationDiagnostics& diagnostics =
      *first.final_report.motion_initialization_diagnostics;
  EXPECT_EQ(diagnostics.pass, MotionInitializationRequest::Pass::FullDeskewCommitCandidate);
  EXPECT_EQ(diagnostics.deskew_solve_passes, 2U);
  EXPECT_EQ(diagnostics.refined_sweeps, kSegments + 1U);
  EXPECT_EQ(diagnostics.refined_registrations, kSegments);
  EXPECT_GT(diagnostics.refined_deskew_pose_interpolations, 0U);
  EXPECT_EQ(diagnostics.holdout_lidar_segments, 1U);
  EXPECT_GT(diagnostics.holdout_residual_dimension, 0U);
  EXPECT_TRUE(diagnostics.statistically_compatible);
  EXPECT_DOUBLE_EQ(diagnostics.minimum_imu_conditioning_covariance_inflation, 4.0);
  EXPECT_DOUBLE_EQ(diagnostics.minimum_applied_lidar_covariance_inflation, 6.0);

  EXPECT_LT(
      (synthetic.expected_final.T_odom_imu.inverse() * commit.estimate.T_odom_imu).log().norm(),
      1.6e-1);
  EXPECT_LT((commit.estimate.velocity_odom - synthetic.expected_final.velocity_odom).norm(),
            kMaximumInitializationVelocityErrorMps);
  EXPECT_LT((commit.estimate.gyro_bias - synthetic.expected_gyro_bias).norm(), 8.0e-3);
  EXPECT_LT((commit.estimate.accel_bias - synthetic.expected_accel_bias).norm(), 5.0e-2);
  EXPECT_GT(commit.estimate.velocity_odom.norm(), 0.5);
  EXPECT_GT(commit.estimate.gyro_bias.norm(), 0.005);
  EXPECT_GT(commit.estimate.accel_bias.norm(), 0.03);

  EXPECT_EQ(commit.covariance.order,
            NavigationCovarianceOrder::RotationVelocityPositionGyroBiasAccelBias);
  EXPECT_TRUE(commit.covariance.matrix.allFinite());
  EXPECT_TRUE(commit.covariance.matrix.isApprox(commit.covariance.matrix.transpose(), 1.0e-10));
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> eigensolver(commit.covariance.matrix,
                                                                           Eigen::EigenvaluesOnly);
  ASSERT_EQ(eigensolver.info(), Eigen::Success);
  EXPECT_GT(eigensolver.eigenvalues().minCoeff(), 0.0);

  EXPECT_EQ(core::validateLineage(commit.lineage), core::LineageValidationError::None);
  EXPECT_TRUE(commit.lineage.correlations.empty());
  const auto roots = measurementRoots(commit.lineage);
  constexpr std::size_t kInitializationImuSamples = static_cast<std::size_t>(
      (static_cast<std::int64_t>(kSegments) * kSegmentDuration + kScanDuration) / kImuPeriod + 1LL);
  EXPECT_EQ(roots.size(), kInitializationImuSamples + kSegments + 1U);
  EXPECT_EQ(commit.lineage.usage.size(), roots.size());
  for (std::size_t index = 0U; index < kInitializationImuSamples; ++index) {
    EXPECT_TRUE(roots.contains(1U + index));
  }
  for (std::size_t index = 0U; index <= kSegments; ++index) {
    EXPECT_TRUE(roots.contains(kFirstLidarId + index));
  }
  ASSERT_FALSE(commit.lineage.usage.empty());
  const core::DerivedRecordId consumer = commit.lineage.usage.front().consumer;
  const auto find_usage = [&](core::MeasurementId measurement) -> const core::ObservationUsage* {
    const auto found = std::find_if(commit.lineage.usage.begin(), commit.lineage.usage.end(),
                                    [&](const core::ObservationUsage& usage) {
                                      const auto* root =
                                          std::get_if<core::MeasurementId>(&usage.slice.root);
                                      return root != nullptr && *root == measurement;
                                    });
    return found == commit.lineage.usage.end() ? nullptr : &*found;
  };
  std::optional<core::FactorGroupId> factor_group;
  for (const auto& usage : commit.lineage.usage) {
    EXPECT_EQ(usage.consumer, consumer);
    EXPECT_FALSE(usage.correlation_group);
    if (usage.role == core::ObservationRole::PrimaryResidual) {
      ASSERT_TRUE(usage.factor_group);
      if (!factor_group) {
        factor_group = usage.factor_group;
      }
      EXPECT_EQ(usage.factor_group, factor_group);
    } else {
      EXPECT_EQ(usage.role, core::ObservationRole::ConditioningOnly);
      EXPECT_FALSE(usage.factor_group);
    }
  }
  ASSERT_TRUE(factor_group);
  for (std::size_t index = 0U; index < kInitializationImuSamples; ++index) {
    const core::ImuSample& sample = synthetic.imu[index];
    const core::ObservationUsage* usage = find_usage(sample.id);
    ASSERT_NE(usage, nullptr);
    EXPECT_EQ(usage->role, sample.stamp.fusion_time < core::FusionTime{kScanDuration}
                               ? core::ObservationRole::ConditioningOnly
                               : core::ObservationRole::PrimaryResidual);
  }
  for (std::size_t index = 0U; index <= kSegments; ++index) {
    const core::ObservationUsage* usage = find_usage(core::MeasurementId{kFirstLidarId + index});
    ASSERT_NE(usage, nullptr);
    const bool fused_scan = index > 0U && index < kSegments;
    EXPECT_EQ(usage->role, fused_scan ? core::ObservationRole::PrimaryResidual
                                      : core::ObservationRole::ConditioningOnly);
  }
  expectExactStatistics(first.statistics);

  const RunOutcome second_outcome = runScenario(synthetic);
  ASSERT_TRUE(second_outcome.value) << second_outcome.error;
  const RunResult& second = *second_outcome.value;
  expectExactStatistics(second.statistics);
  ASSERT_EQ(second.bootstrap.size(), first.bootstrap.size());
  EXPECT_EQ(second.estimate.state_time, first.estimate.state_time);
  EXPECT_TRUE(second.estimate.estimate.T_odom_imu.matrix().isApprox(
      first.estimate.estimate.T_odom_imu.matrix(), 1.0e-12));
  EXPECT_TRUE(second.estimate.estimate.velocity_odom.isApprox(first.estimate.estimate.velocity_odom,
                                                              1.0e-12));
  EXPECT_TRUE(
      second.estimate.estimate.accel_bias.isApprox(first.estimate.estimate.accel_bias, 1.0e-12));
  EXPECT_TRUE(
      second.estimate.estimate.gyro_bias.isApprox(first.estimate.estimate.gyro_bias, 1.0e-12));
  EXPECT_TRUE(
      second.estimate.covariance.matrix.isApprox(first.estimate.covariance.matrix, 1.0e-12));
  EXPECT_EQ(measurementRoots(second.estimate.lineage), measurementRoots(first.estimate.lineage));
  for (std::size_t index = 0U; index < first.bootstrap.size(); ++index) {
    EXPECT_TRUE(second.bootstrap[index].T_bootstrap_imu.matrix().isApprox(
        first.bootstrap[index].T_bootstrap_imu.matrix(), 1.0e-12));
  }
}

TEST(LocalEstimatorMovingInitialization,
     FirstPostInitializationSweepUsesSeededRefinedTargetContinuously) {
  const SyntheticScenario synthetic = scenario();
  const RunOutcome outcome = runScenario(synthetic, synthetic.lidar.size());
  ASSERT_TRUE(outcome.value) << outcome.error;
  const RunResult& result = *outcome.value;
  ASSERT_TRUE(result.final_report.initialization_map_input);
  EXPECT_EQ(result.final_report.initialization_map_input->sweep.id,
            core::MeasurementId{kFirstLidarId + kSegments});
  ASSERT_EQ(result.tracking_commits.size(), kContinuationScans);
  const LidarCommitReport& tracking = result.tracking_commits.front();
  EXPECT_EQ(tracking.disposition, LidarCommitDisposition::Registered)
      << tracking.degradation_detail;
  ASSERT_TRUE(tracking.registration);
  ASSERT_TRUE(tracking.map_input);
  EXPECT_GE(tracking.registration->diagnostics.observable_rank, 4U);
  EXPECT_GE(tracking.registration->diagnostics.correspondences, 30U);
  EXPECT_EQ(tracking.commit.solve.combined_imu_factors, 1U);
  EXPECT_EQ(tracking.commit.solve.active_lidar_direct_batch_factors, 1U);
  EXPECT_EQ(tracking.commit.solve.lidar_direct_batch_factors_added, 1U);
  EXPECT_EQ(result.statistics.graph_commits, 3U);
  EXPECT_EQ(result.statistics.lidar_registrations, 1U);
  EXPECT_EQ(result.statistics.lidar_bootstraps, 0U);
  EXPECT_EQ(result.statistics.lidar_degraded_commits, 0U);
  EXPECT_LT(
      (synthetic.expected_tracking_final.T_odom_imu.inverse() * result.estimate.estimate.T_odom_imu)
          .log()
          .norm(),
      2.0e-1);
  EXPECT_LT(
      (result.estimate.estimate.velocity_odom - synthetic.expected_tracking_final.velocity_odom)
          .norm(),
      2.0e-1);
}

TEST(LocalEstimatorMovingInitialization, TransitionsIntoExplicitImuGuardHold) {
  const SyntheticScenario synthetic = scenario();
  auto created = LocalEstimator::create(calibration(synthetic.T_imu_lidar), estimatorConfig());
  ASSERT_TRUE(created) << created.error().detail;
  LocalEstimator estimator = std::move(created).value();

  std::size_t next_imu = 0U;
  std::optional<LocalGraphCommit> initialization;
  for (std::size_t scan = 0U; scan <= kSegments; ++scan) {
    while (next_imu < synthetic.imu.size() &&
           synthetic.imu[next_imu].stamp.fusion_time <= synthetic.lidar[scan].acquisition.end) {
      const auto ingested = estimator.ingestImu(synthetic.imu[next_imu]);
      ASSERT_TRUE(ingested) << ingested.error().detail;
      ++next_imu;
    }
    ASSERT_TRUE(estimator.enqueueLidar(synthetic.lidar[scan]));
    const auto processed = estimator.processReady();
    ASSERT_TRUE(processed) << processed.error().detail;
    if (processed.value().initialization) {
      initialization = processed.value().initialization;
    }
  }
  ASSERT_TRUE(initialization);
  ASSERT_EQ(estimator.lifecycle(), LocalEstimatorLifecycle::Tracking);
  ASSERT_EQ(initialization->state_time + core::Duration{100 * kMillisecond},
            core::FusionTime{1'440 * kMillisecond});
  const core::FusionTime guard_time =
      initialization->state_time + core::Duration{100 * kMillisecond};

  while (next_imu < synthetic.imu.size() &&
         synthetic.imu[next_imu].stamp.fusion_time <= guard_time) {
    const auto ingested = estimator.ingestImu(synthetic.imu[next_imu]);
    ASSERT_TRUE(ingested) << ingested.error().detail;
    ++next_imu;
  }
  const auto prediction = estimator.propagateTo(guard_time);
  ASSERT_TRUE(prediction) << prediction.error().detail;
  EXPECT_FALSE(prediction.value().raw_imu_support.empty());

  const auto enqueued = estimator.enqueueImuGuard(guard_time);
  ASSERT_TRUE(enqueued) << enqueued.error().detail;
  const auto held = estimator.processReady();
  ASSERT_TRUE(held) << held.error().detail;
  ASSERT_EQ(held.value().imu_guard_commits.size(), 1U);
  ASSERT_EQ(held.value().imu_guard_commits.front().resolutions.size(), 1U);
  const ImuGuardCommitReport& guard = held.value().imu_guard_commits.front();
  EXPECT_EQ(guard.commit.state, core::StateId{initialization->state.value() + 1U});
  EXPECT_EQ(guard.commit.revision, core::LocalGraphRevision{initialization->revision.value() + 1U});
  EXPECT_EQ(guard.commit.state_time, guard_time);
  EXPECT_EQ(guard.commit.solve.combined_imu_factors, 1U);
  EXPECT_EQ(guard.commit.solve.active_lidar_direct_batch_factors, 0U);
  EXPECT_LT(
      (prediction.value().propagated_state.T_odom_imu.inverse() * guard.commit.estimate.T_odom_imu)
          .log()
          .norm(),
      1.0e-3);
  EXPECT_LT(
      (prediction.value().propagated_state.velocity_odom - guard.commit.estimate.velocity_odom)
          .norm(),
      1.0e-3);
  EXPECT_EQ(estimator.statistics().imu_guard_knots_committed, 1U);
  EXPECT_EQ(estimator.statistics().imu_guard_requests_resolved, 1U);

  const auto replay = estimator.propagateTo(guard_time);
  ASSERT_TRUE(replay);
  EXPECT_EQ(replay.value().gap_status, ImuPropagationGapStatus::AnchorOnly);
  EXPECT_EQ(replay.value().anchor_state, guard.commit.state);
  EXPECT_EQ(replay.value().anchor_revision, guard.commit.revision);
}

TEST(LocalEstimatorMovingInitialization,
     NonIdentityLidarExtrinsicPreservesPoseAndVelocityDirection) {
  // A non-commuting rotation plus asymmetric lever arm prevents a mistaken
  // T_lidar_imu/T_imu_lidar inversion from being hidden by the identity
  // extrinsic used by the main deterministic fixture.
  const core::Pose3d T_imu_lidar(Sophus::SO3d::exp(Eigen::Vector3d{0.31, -0.27, 0.19}),
                                 Eigen::Vector3d{-0.037, 0.081, -0.026});
  const SyntheticScenario synthetic = scenario(T_imu_lidar);
  const RunOutcome outcome = runScenario(synthetic);
  ASSERT_TRUE(outcome.value) << outcome.error;
  const core::NavStateEstimate& actual = outcome.value->estimate.estimate;

  EXPECT_LT((synthetic.expected_final.T_odom_imu.inverse() * actual.T_odom_imu).log().norm(),
            1.6e-1);
  EXPECT_LT((actual.velocity_odom - synthetic.expected_final.velocity_odom).norm(),
            kMaximumInitializationVelocityErrorMps);
  EXPECT_GT(actual.T_odom_imu.translation().dot(synthetic.expected_final.T_odom_imu.translation()),
            0.0);
  EXPECT_GT(actual.velocity_odom.dot(synthetic.expected_final.velocity_odom), 0.0);
}

}  // namespace
}  // namespace meridian::local
