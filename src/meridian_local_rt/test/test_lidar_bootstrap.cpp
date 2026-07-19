#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <variant>
#include <vector>

#include "meridian/local/lidar_bootstrap.hpp"

namespace meridian::local {
namespace {

constexpr std::int64_t kScanDuration = 90'000'000LL;

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

[[nodiscard]] core::LidarSweep makeSweep(std::uint64_t id, std::int64_t start_ns,
                                         const core::Pose3d& T_first_current) {
  constexpr std::uint32_t kWidth = 64U;
  constexpr std::uint32_t kHeight = 16U;
  auto points = std::make_shared<core::LidarPoints>();
  points->reserve(static_cast<std::size_t>(kWidth) * kHeight);
  const core::Pose3d T_current_first = T_first_current.inverse();
  for (std::uint32_t row = 0U; row < kHeight; ++row) {
    for (std::uint32_t column = 0U; column < kWidth; ++column) {
      Eigen::Vector3d point_first;
      if (row < 6U) {
        point_first = Eigen::Vector3d{1.0 + 0.08 * static_cast<double>(column),
                                      -1.0 + 0.20 * static_cast<double>(row), -1.5};
      } else if (row < 11U) {
        point_first = Eigen::Vector3d{6.0, -2.0 + 0.08 * static_cast<double>(column),
                                      -1.0 + 0.25 * static_cast<double>(row - 6U)};
      } else {
        point_first = Eigen::Vector3d{1.0 + 0.08 * static_cast<double>(column), 4.0,
                                      -1.0 + 0.25 * static_cast<double>(row - 11U)};
      }
      const Eigen::Vector3d point_current = T_current_first * point_first;
      core::LidarPoint point;
      point.x = static_cast<float>(point_current.x());
      point.y = static_cast<float>(point_current.y());
      point.z = static_cast<float>(point_current.z());
      point.intensity = static_cast<float>(row + column);
      point.time_offset_ns =
          static_cast<std::int32_t>((static_cast<std::int64_t>(column) * (kScanDuration - 1LL)) /
                                    static_cast<std::int64_t>(kWidth - 1U));
      point.ring = static_cast<std::uint16_t>(row);
      point.source_index = row * kWidth + column;
      points->push_back(point);
    }
  }

  core::LidarSweep sweep;
  sweep.header = header();
  sweep.id = core::MeasurementId{id};
  sweep.lidar = core::LidarId{1U};
  sweep.stamp = stamp(start_ns, id);
  sweep.acquisition =
      core::TimeRange{core::FusionTime{start_ns}, core::FusionTime{start_ns + kScanDuration}};
  sweep.layout = core::LidarLayout{kWidth, kHeight, true};
  sweep.points = std::move(points);
  return sweep;
}

[[nodiscard]] ImuInterval makeImuInterval(
    core::FusionTime start, core::FusionTime end, std::uint64_t first_id = 100U,
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero()) {
  ImuInterval interval;
  interval.support = core::TimeRange{start, end};
  const core::FusionTime middle{start.nanoseconds + (end.nanoseconds - start.nanoseconds) / 2LL};
  const std::vector<core::FusionTime> times{start, middle, end};
  for (std::size_t index = 0U; index < times.size(); ++index) {
    const core::MeasurementId id{first_id + index};
    interval.knots.push_back(InterpolatedImuSample{times[index], Eigen::Vector3d{0.0, 0.0, 9.80665},
                                                   angular_velocity, id, id});
    interval.raw_measurements.push_back(id);
  }
  interval.maximum_raw_gap = middle - start;
  return interval;
}

[[nodiscard]] core::ObservationLineage imuLineage(const ImuInterval& interval,
                                                  std::uint64_t lineage_id) {
  core::ObservationLineage lineage;
  lineage.id = core::ObservationLineageId{lineage_id};
  for (const core::MeasurementId measurement : interval.raw_measurements) {
    core::ObservationSlice slice;
    slice.root = measurement;
    slice.calibration = core::CalibrationEpoch{1U};
    lineage.usage.push_back(core::ObservationUsage{slice, core::ObservationRole::PrimaryResidual,
                                                   core::DerivedRecordId{lineage_id},
                                                   core::FactorGroupId{lineage_id}, std::nullopt});
  }
  return lineage;
}

[[nodiscard]] core::ObservationLineage lidarLineage(core::MeasurementId previous,
                                                    core::MeasurementId current,
                                                    std::uint64_t lineage_id,
                                                    const ImuInterval& previous_acquisition,
                                                    const ImuInterval& current_acquisition,
                                                    const ImuInterval& segment) {
  core::ObservationLineage lineage;
  lineage.id = core::ObservationLineageId{lineage_id};
  core::ObservationSlice source;
  source.root = current;
  source.calibration = core::CalibrationEpoch{1U};
  core::ObservationSlice target;
  target.root = previous;
  target.calibration = core::CalibrationEpoch{1U};
  const core::CorrelationGroupId correlation{lineage_id};
  lineage.usage.push_back(core::ObservationUsage{source, core::ObservationRole::PrimaryResidual,
                                                 core::DerivedRecordId{lineage_id},
                                                 core::FactorGroupId{lineage_id}, std::nullopt});
  lineage.usage.push_back(core::ObservationUsage{target, core::ObservationRole::ConditioningOnly,
                                                 core::DerivedRecordId{lineage_id}, std::nullopt,
                                                 correlation});
  std::set<std::uint64_t> conditioned;
  const auto add_imu = [&](const ImuInterval& interval) {
    for (core::MeasurementId measurement : interval.raw_measurements) {
      if (!conditioned.insert(measurement.value()).second) {
        continue;
      }
      core::ObservationSlice slice;
      slice.root = measurement;
      slice.calibration = core::CalibrationEpoch{1U};
      lineage.usage.push_back(core::ObservationUsage{slice, core::ObservationRole::ConditioningOnly,
                                                     core::DerivedRecordId{lineage_id},
                                                     std::nullopt, correlation});
    }
  };
  add_imu(previous_acquisition);
  add_imu(current_acquisition);
  add_imu(segment);
  lineage.correlations.push_back(core::CorrelationDeclaration{
      correlation, core::CorrelationPolicyRevision{1U},
      core::CorrelationTreatment::CovarianceInflationAndInformationCap, 6.0, 1.0e6 / 6.0});
  return lineage;
}

[[nodiscard]] LidarBootstrapOdometryConfig bootstrapConfig() {
  LidarBootstrapOdometryConfig config;
  config.maximum_sweeps = 4U;
  config.preprocessing.minimum_range_m = 0.5;
  config.preprocessing.maximum_range_m = 30.0;
  config.preprocessing.voxel_size_m = 0.15;
  config.preprocessing.maximum_output_points = 4'000U;
  config.registration.target_voxel_resolution_m = 0.5;
  config.registration.source_voxel_size_m = 0.5;
  config.registration.maximum_correspondence_distance_m = 1.0;
  config.registration.maximum_source_points = 4'000U;
  config.registration.maximum_target_points_per_target = 4'000U;
  config.registration.maximum_targets = 1U;
  config.registration.minimum_correspondences = 30U;
  config.registration.residual_standard_deviation_m = 0.05;
  config.registration.huber_delta_multiplier = 3.0;
  config.registration.absolute_normalized_observable_eigenvalue = 1.0e-8;
  config.registration.relative_normalized_observable_eigenvalue = 1.0e-6;
  config.registration.maximum_translation_information = 1.0e6;
  config.registration.minimum_observable_rank = 4U;
  config.registration.translation_convergence_m = 1.0e-5;
  config.registration.rotation_convergence_rad = 1.0e-5;
  return config;
}

TEST(LidarBootstrapOdometry, CommitsImuConditionedRelativePoseAndExactSegment) {
  const core::Pose3d T_imu_lidar(Sophus::SO3d::exp(Eigen::Vector3d{0.025, -0.018, 0.012}),
                                 Eigen::Vector3d{0.31, -0.08, 0.14});
  auto created = LidarBootstrapOdometry::create(core::CalibrationEpoch{1U}, T_imu_lidar,
                                                Eigen::Vector3d{0.0, 0.0, -9.80665},
                                                Eigen::Vector3d::Zero(), bootstrapConfig());
  ASSERT_TRUE(created) << created.error().detail;
  auto odometry = std::move(created).value();

  LidarBootstrapInput first;
  first.sweep = makeSweep(10U, 0LL, core::Pose3d{});
  first.cloud_record = core::DerivedRecordId{1U};
  first.cloud_lineage = core::ObservationLineageId{1U};
  const ImuInterval first_acquisition =
      makeImuInterval(core::FusionTime{0LL}, core::FusionTime{kScanDuration}, 200U);
  first.acquisition_imu = first_acquisition;
  auto anchored = odometry.add(std::move(first));
  ASSERT_TRUE(anchored) << anchored.error().detail;
  EXPECT_EQ(anchored.value().disposition, LidarBootstrapDisposition::AnchorCreated);
  EXPECT_FALSE(anchored.value().segment);
  EXPECT_EQ(anchored.value().preprocessing.input_points, 1'024U);
  EXPECT_GT(anchored.value().preprocessing.deterministic_voxel_points, 100U);

  const core::Pose3d T_first_second(Sophus::SO3d::exp(Eigen::Vector3d{0.01, -0.015, 0.035}),
                                    Eigen::Vector3d{0.12, -0.07, 0.04});
  const core::Pose3d T_first_lidar_second_lidar =
      T_imu_lidar.inverse() * T_first_second * T_imu_lidar;
  const core::FusionTime first_time{kScanDuration};
  const core::FusionTime second_time{100'000'000LL + kScanDuration};
  ImuInterval interval =
      makeImuInterval(first_time, second_time, 300U, Eigen::Vector3d{0.0, 0.0, 0.3});
  const ImuInterval second_acquisition =
      makeImuInterval(core::FusionTime{100'000'000LL}, second_time, 400U);
  LidarBootstrapInput second;
  second.sweep = makeSweep(11U, 100'000'000LL, T_first_lidar_second_lidar);
  second.cloud_record = core::DerivedRecordId{2U};
  second.cloud_lineage = core::ObservationLineageId{2U};
  second.acquisition_imu = second_acquisition;
  second.lidar_factor_lineage = lidarLineage(core::MeasurementId{10U}, core::MeasurementId{11U}, 3U,
                                             first_acquisition, second_acquisition, interval);
  second.imu_factor_lineage = imuLineage(interval, 4U);
  second.between_reference_imu = interval;
  auto increment = odometry.add(std::move(second));

  ASSERT_TRUE(increment) << increment.error().detail;
  ASSERT_TRUE(increment.value().segment);
  ASSERT_TRUE(increment.value().registration);
  EXPECT_GE(increment.value().registration->diagnostics.observable_rank, 4U);
  EXPECT_EQ(increment.value().registration->target_snapshots.size(), 1U);
  ASSERT_NE(increment.value().registration->target_snapshots.front(), nullptr);
  EXPECT_EQ(increment.value().registration->target_snapshots.front()->targetState(),
            core::StateId{1U});
  EXPECT_EQ(increment.value().registration->target_snapshots.front()->sourceState(),
            core::StateId{2U});
  EXPECT_EQ(increment.value().segment->lidar.start_time, first_time);
  EXPECT_EQ(increment.value().segment->lidar.end_time, second_time);
  EXPECT_GE(increment.value().segment->lidar.information.rank, 4U);
  EXPECT_EQ(increment.value().segment->lidar.information.rank,
            increment.value().registration->diagnostics.physical_information.rank);
  for (Eigen::Index index = 0; index < 6; ++index) {
    const double expected =
        increment.value().registration->diagnostics.physical_information.eigenvalues(index);
    EXPECT_NEAR(6.0 * increment.value().segment->lidar.information.eigenvalues(index), expected,
                1.0e-12 * std::max(1.0, expected));
  }
  EXPECT_LT((T_first_second.inverse() * increment.value().segment->lidar.T_imu_start_imu_end)
                .log()
                .norm(),
            2.0e-2)
      << "estimated tangent="
      << increment.value().segment->lidar.T_imu_start_imu_end.log().transpose()
      << " correspondences=" << increment.value().registration->diagnostics.correspondences
      << " cost=" << increment.value().registration->final_robust_cost
      << " rank=" << increment.value().registration->diagnostics.observable_rank;
  EXPECT_TRUE(increment.value().T_bootstrap_imu.matrix().isApprox(
      increment.value().segment->lidar.T_imu_start_imu_end.matrix(), 1.0e-12));
  EXPECT_EQ(odometry.retainedSweeps(), 2U);

  core::NavStateEstimate first_state;
  core::NavStateEstimate second_state;
  second_state.T_odom_imu = T_first_second;
  const std::vector<TimedNavState> states{TimedNavState{first_time, first_state},
                                          TimedNavState{second_time, second_state}};
  const auto invalid_refinement = odometry.refine(
      {TimedNavState{core::FusionTime{first_time.nanoseconds + 1LL}, first_state}, states.back()});
  ASSERT_FALSE(invalid_refinement);
  EXPECT_EQ(invalid_refinement.error().code, LidarBootstrapErrorCode::RefinementStateMismatch);

  const auto refinement = odometry.refine(states);
  ASSERT_TRUE(refinement) << refinement.error().detail;
  ASSERT_EQ(refinement.value().segments.size(), 1U);
  ASSERT_NE(refinement.value().final_cloud, nullptr);
  EXPECT_EQ(refinement.value().final_cloud->source_sweep, core::MeasurementId{11U});
  EXPECT_TRUE(refinement.value().final_cloud->T_odom_imu_seed.matrix().isApprox(
      T_first_second.matrix(), 1.0e-12));
  EXPECT_EQ(refinement.value().final_cloud->stats.input_points, 1'024U);
  EXPECT_EQ(refinement.value().final_lineage.id, core::ObservationLineageId{2U});
  EXPECT_EQ(core::validateLineage(refinement.value().final_lineage),
            core::LineageValidationError::None);
  ASSERT_EQ(refinement.value().final_lineage.usage.size(),
            1U + second_acquisition.raw_measurements.size());
  const auto find_lineage_usage = [&](core::MeasurementId measurement) {
    return std::find_if(refinement.value().final_lineage.usage.begin(),
                        refinement.value().final_lineage.usage.end(),
                        [&](const core::ObservationUsage& usage) {
                          const auto* root = std::get_if<core::MeasurementId>(&usage.slice.root);
                          return root != nullptr && *root == measurement;
                        });
  };
  const auto source_usage = find_lineage_usage(core::MeasurementId{11U});
  ASSERT_NE(source_usage, refinement.value().final_lineage.usage.end());
  EXPECT_EQ(source_usage->role, core::ObservationRole::DerivedSummary);
  EXPECT_EQ(source_usage->consumer, core::DerivedRecordId{2U});
  for (const core::MeasurementId measurement : second_acquisition.raw_measurements) {
    const auto usage = find_lineage_usage(measurement);
    ASSERT_NE(usage, refinement.value().final_lineage.usage.end());
    EXPECT_EQ(usage->role, core::ObservationRole::ConditioningOnly);
    EXPECT_FALSE(usage->factor_group);
  }
  ASSERT_EQ(refinement.value().segments.front().lidar.lineage.correlations.size(), 1U);
  EXPECT_DOUBLE_EQ(
      *refinement.value().segments.front().lidar.lineage.correlations.front().total_information_cap,
      bootstrapConfig().registration.maximum_translation_information / 6.0);
  EXPECT_GE(refinement.value().diagnostics.minimum_observable_rank, 4U);
  EXPECT_EQ(refinement.value().diagnostics.registrations, 1U);
  EXPECT_TRUE(std::isfinite(refinement.value().diagnostics.total_registration_cost));
  EXPECT_DOUBLE_EQ(refinement.value().diagnostics.applied_covariance_inflation, 6.0);
  EXPECT_LT(
      (T_first_second.inverse() * refinement.value().segments.front().lidar.T_imu_start_imu_end)
          .log()
          .norm(),
      2.0e-2)
      << "estimated tangent="
      << refinement.value().segments.front().lidar.T_imu_start_imu_end.log().transpose();
  EXPECT_EQ(odometry.retainedSweeps(), 2U);
}

TEST(LidarBootstrapOdometry, InvalidCandidateDoesNotMutateCommittedAnchor) {
  auto created = LidarBootstrapOdometry::create(core::CalibrationEpoch{1U}, core::Pose3d{},
                                                Eigen::Vector3d{0.0, 0.0, -9.80665},
                                                Eigen::Vector3d::Zero(), bootstrapConfig());
  ASSERT_TRUE(created);
  auto odometry = std::move(created).value();
  LidarBootstrapInput first;
  first.sweep = makeSweep(20U, 0LL, core::Pose3d{});
  first.cloud_record = core::DerivedRecordId{10U};
  first.cloud_lineage = core::ObservationLineageId{10U};
  const ImuInterval first_acquisition =
      makeImuInterval(core::FusionTime{0LL}, core::FusionTime{kScanDuration}, 500U);
  first.acquisition_imu = first_acquisition;
  ASSERT_TRUE(odometry.add(std::move(first)));

  const core::FusionTime first_time{kScanDuration};
  const core::FusionTime second_time{100'000'000LL + kScanDuration};
  ImuInterval interval = makeImuInterval(first_time, second_time, 600U);
  const ImuInterval second_acquisition =
      makeImuInterval(core::FusionTime{100'000'000LL}, second_time, 700U);
  LidarBootstrapInput bad;
  bad.sweep = makeSweep(21U, 100'000'000LL, core::Pose3d{});
  bad.cloud_record = core::DerivedRecordId{11U};
  bad.cloud_lineage = core::ObservationLineageId{11U};
  bad.acquisition_imu = second_acquisition;
  bad.between_reference_imu = interval;
  bad.lidar_factor_lineage = lidarLineage(core::MeasurementId{999U}, core::MeasurementId{21U}, 12U,
                                          first_acquisition, second_acquisition, interval);
  bad.imu_factor_lineage = imuLineage(interval, 13U);
  const auto rejected = odometry.add(std::move(bad));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LidarBootstrapErrorCode::InvalidLineage);
  EXPECT_EQ(odometry.retainedSweeps(), 1U);

  LidarBootstrapInput recovered;
  recovered.sweep = makeSweep(21U, 100'000'000LL, core::Pose3d{});
  recovered.cloud_record = core::DerivedRecordId{11U};
  recovered.cloud_lineage = core::ObservationLineageId{11U};
  recovered.acquisition_imu = second_acquisition;
  recovered.between_reference_imu = interval;
  recovered.lidar_factor_lineage =
      lidarLineage(core::MeasurementId{20U}, core::MeasurementId{21U}, 12U, first_acquisition,
                   second_acquisition, interval);
  recovered.imu_factor_lineage = imuLineage(interval, 13U);
  const auto accepted = odometry.add(std::move(recovered));
  ASSERT_TRUE(accepted) << accepted.error().detail;
  EXPECT_EQ(accepted.value().disposition, LidarBootstrapDisposition::IncrementCommitted);
  EXPECT_EQ(odometry.retainedSweeps(), 2U);
}

TEST(LidarBootstrapOdometry, RejectsMissingExactAcquisitionImuBeforeMutation) {
  auto created = LidarBootstrapOdometry::create(core::CalibrationEpoch{1U}, core::Pose3d{},
                                                Eigen::Vector3d{0.0, 0.0, -9.80665},
                                                Eigen::Vector3d::Zero(), bootstrapConfig());
  ASSERT_TRUE(created);
  auto odometry = std::move(created).value();

  LidarBootstrapInput input;
  input.sweep = makeSweep(30U, 0LL, core::Pose3d{});
  input.cloud_record = core::DerivedRecordId{20U};
  input.cloud_lineage = core::ObservationLineageId{20U};
  const auto rejected = odometry.add(std::move(input));

  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LidarBootstrapErrorCode::InvalidAcquisitionImu);
  EXPECT_TRUE(odometry.empty());
  EXPECT_EQ(odometry.retainedSweeps(), 0U);
}

TEST(LidarBootstrapOdometry, RejectsCloudBuildAndRegistrationFailuresWithoutMutation) {
  auto cloud_config = bootstrapConfig();
  cloud_config.preprocessing.minimum_range_m = 40.0;
  cloud_config.preprocessing.maximum_range_m = 60.0;
  auto cloud_created = LidarBootstrapOdometry::create(core::CalibrationEpoch{1U}, core::Pose3d{},
                                                      Eigen::Vector3d{0.0, 0.0, -9.80665},
                                                      Eigen::Vector3d::Zero(), cloud_config);
  ASSERT_TRUE(cloud_created);
  auto cloud_odometry = std::move(cloud_created).value();
  LidarBootstrapInput cloud_input;
  cloud_input.sweep = makeSweep(35U, 0LL, core::Pose3d{});
  cloud_input.cloud_record = core::DerivedRecordId{25U};
  cloud_input.cloud_lineage = core::ObservationLineageId{25U};
  cloud_input.acquisition_imu =
      makeImuInterval(core::FusionTime{0LL}, core::FusionTime{kScanDuration}, 1'100U);
  const auto cloud_rejected = cloud_odometry.add(std::move(cloud_input));
  ASSERT_FALSE(cloud_rejected);
  EXPECT_EQ(cloud_rejected.error().code, LidarBootstrapErrorCode::RegistrationCloudBuildFailed);
  EXPECT_TRUE(cloud_rejected.error().preprocessing.has_value());
  EXPECT_TRUE(cloud_odometry.empty());

  auto registration_config = bootstrapConfig();
  registration_config.registration.minimum_correspondences = 4'000U;
  auto registration_created = LidarBootstrapOdometry::create(
      core::CalibrationEpoch{1U}, core::Pose3d{}, Eigen::Vector3d{0.0, 0.0, -9.80665},
      Eigen::Vector3d::Zero(), registration_config);
  ASSERT_TRUE(registration_created);
  auto registration_odometry = std::move(registration_created).value();
  LidarBootstrapInput anchor;
  anchor.sweep = makeSweep(36U, 0LL, core::Pose3d{});
  anchor.cloud_record = core::DerivedRecordId{26U};
  anchor.cloud_lineage = core::ObservationLineageId{26U};
  const ImuInterval anchor_acquisition =
      makeImuInterval(core::FusionTime{0LL}, core::FusionTime{kScanDuration}, 1'200U);
  anchor.acquisition_imu = anchor_acquisition;
  ASSERT_TRUE(registration_odometry.add(std::move(anchor)));

  const core::FusionTime anchor_time{kScanDuration};
  const core::FusionTime source_time{100'000'000LL + kScanDuration};
  const ImuInterval source_acquisition =
      makeImuInterval(core::FusionTime{100'000'000LL}, source_time, 1'300U);
  const ImuInterval segment = makeImuInterval(anchor_time, source_time, 1'400U);
  LidarBootstrapInput source;
  source.sweep = makeSweep(37U, 100'000'000LL, core::Pose3d{});
  source.cloud_record = core::DerivedRecordId{27U};
  source.cloud_lineage = core::ObservationLineageId{27U};
  source.acquisition_imu = source_acquisition;
  source.between_reference_imu = segment;
  source.lidar_factor_lineage = lidarLineage(core::MeasurementId{36U}, core::MeasurementId{37U},
                                             28U, anchor_acquisition, source_acquisition, segment);
  source.imu_factor_lineage = imuLineage(segment, 29U);
  const auto registration_rejected = registration_odometry.add(std::move(source));
  ASSERT_FALSE(registration_rejected);
  EXPECT_EQ(registration_rejected.error().code, LidarBootstrapErrorCode::RegistrationFailed);
  EXPECT_TRUE(registration_rejected.error().registration.has_value());
  EXPECT_EQ(registration_odometry.retainedSweeps(), 1U);
}

TEST(LidarBootstrapOdometry, EnforcesBoundedCommittedSweepCapacity) {
  auto config = bootstrapConfig();
  config.maximum_sweeps = 2U;
  auto created = LidarBootstrapOdometry::create(core::CalibrationEpoch{1U}, core::Pose3d{},
                                                Eigen::Vector3d{0.0, 0.0, -9.80665},
                                                Eigen::Vector3d::Zero(), config);
  ASSERT_TRUE(created);
  auto odometry = std::move(created).value();

  LidarBootstrapInput first;
  first.sweep = makeSweep(37U, 0LL, core::Pose3d{});
  first.cloud_record = core::DerivedRecordId{27U};
  first.cloud_lineage = core::ObservationLineageId{27U};
  const ImuInterval first_acquisition =
      makeImuInterval(core::FusionTime{0LL}, core::FusionTime{kScanDuration}, 1'300U);
  first.acquisition_imu = first_acquisition;
  ASSERT_TRUE(odometry.add(std::move(first)));

  const core::FusionTime first_time{kScanDuration};
  const core::FusionTime second_time{100'000'000LL + kScanDuration};
  const ImuInterval segment = makeImuInterval(first_time, second_time, 1'400U);
  const ImuInterval second_acquisition =
      makeImuInterval(core::FusionTime{100'000'000LL}, second_time, 1'500U);
  LidarBootstrapInput second;
  second.sweep = makeSweep(38U, 100'000'000LL, core::Pose3d{});
  second.cloud_record = core::DerivedRecordId{28U};
  second.cloud_lineage = core::ObservationLineageId{28U};
  second.acquisition_imu = second_acquisition;
  second.between_reference_imu = segment;
  second.lidar_factor_lineage = lidarLineage(core::MeasurementId{37U}, core::MeasurementId{38U},
                                             29U, first_acquisition, second_acquisition, segment);
  second.imu_factor_lineage = imuLineage(segment, 30U);
  ASSERT_TRUE(odometry.add(std::move(second)));
  EXPECT_EQ(odometry.retainedSweeps(), 2U);

  const auto rejected = odometry.add(LidarBootstrapInput{});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LidarBootstrapErrorCode::Capacity);
  EXPECT_EQ(odometry.retainedSweeps(), 2U);
}

TEST(LidarBootstrapOdometry, RejectsLineageWhoseDeclaredInflationDoesNotMatchAppliedPolicy) {
  auto created = LidarBootstrapOdometry::create(core::CalibrationEpoch{1U}, core::Pose3d{},
                                                Eigen::Vector3d{0.0, 0.0, -9.80665},
                                                Eigen::Vector3d::Zero(), bootstrapConfig());
  ASSERT_TRUE(created);
  auto odometry = std::move(created).value();

  LidarBootstrapInput first;
  first.sweep = makeSweep(40U, 0LL, core::Pose3d{});
  first.cloud_record = core::DerivedRecordId{30U};
  first.cloud_lineage = core::ObservationLineageId{30U};
  const ImuInterval first_acquisition =
      makeImuInterval(core::FusionTime{0LL}, core::FusionTime{kScanDuration}, 800U);
  first.acquisition_imu = first_acquisition;
  ASSERT_TRUE(odometry.add(std::move(first)));

  const core::FusionTime first_time{kScanDuration};
  const core::FusionTime second_time{100'000'000LL + kScanDuration};
  const ImuInterval segment = makeImuInterval(first_time, second_time, 900U);
  const ImuInterval second_acquisition =
      makeImuInterval(core::FusionTime{100'000'000LL}, second_time, 1'000U);
  LidarBootstrapInput second;
  second.sweep = makeSweep(41U, 100'000'000LL, core::Pose3d{});
  second.cloud_record = core::DerivedRecordId{31U};
  second.cloud_lineage = core::ObservationLineageId{31U};
  second.acquisition_imu = second_acquisition;
  second.between_reference_imu = segment;
  second.lidar_factor_lineage = lidarLineage(core::MeasurementId{40U}, core::MeasurementId{41U},
                                             32U, first_acquisition, second_acquisition, segment);
  ASSERT_FALSE(second.lidar_factor_lineage->correlations.empty());
  second.lidar_factor_lineage->correlations.front().covariance_inflation = 4.0;
  second.imu_factor_lineage = imuLineage(segment, 33U);

  const auto rejected = odometry.add(std::move(second));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LidarBootstrapErrorCode::InvalidLineage);
  EXPECT_EQ(odometry.retainedSweeps(), 1U);
}

}  // namespace
}  // namespace meridian::local
