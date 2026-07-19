#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "finalized_lidar_factor_test_utils.hpp"
#include "lidar_registration_cloud_test_utils.hpp"
#include "meridian/local/graph.hpp"

namespace meridian::local {
namespace {

constexpr std::int64_t kSecond = 1'000'000'000LL;
constexpr double kGravity = 9.80665;
constexpr double kVelocityX = 0.05;

[[nodiscard]] core::ContentHash presentHash(std::uint8_t value) {
  core::ContentHash hash{};
  hash.front() = value;
  return hash;
}

[[nodiscard]] core::ObservationLineage initializationLineage(std::uint64_t id) {
  core::ObservationLineage lineage;
  lineage.id = core::ObservationLineageId{id};
  return lineage;
}

[[nodiscard]] core::ObservationLineage batchLineage(
    std::uint64_t batch_id, core::MeasurementId source_sweep,
    std::span<const core::MeasurementId> target_sweeps) {
  core::ObservationLineage lineage;
  lineage.id = core::ObservationLineageId{100U + batch_id};
  lineage.checksum = presentHash(static_cast<std::uint8_t>(1U + batch_id % 250U));
  core::ObservationSlice slice;
  slice.root = source_sweep;
  slice.calibration = core::CalibrationEpoch{5U};
  lineage.usage.push_back(core::ObservationUsage{
      slice, core::ObservationRole::PrimaryResidual, core::DerivedRecordId{2'000U + batch_id},
      core::FactorGroupId{3'000U + batch_id}, std::nullopt});
  const core::CorrelationGroupId correlation{4'000U + batch_id};
  for (const core::MeasurementId target_sweep : target_sweeps) {
    core::ObservationSlice target;
    target.root = target_sweep;
    target.calibration = core::CalibrationEpoch{5U};
    lineage.usage.push_back(core::ObservationUsage{target, core::ObservationRole::ConditioningOnly,
                                                   core::DerivedRecordId{2'000U + batch_id},
                                                   std::nullopt, correlation});
  }
  core::ObservationSlice imu;
  imu.root = core::MeasurementId{90'000U + batch_id};
  imu.calibration = core::CalibrationEpoch{5U};
  lineage.usage.push_back(core::ObservationUsage{imu, core::ObservationRole::ConditioningOnly,
                                                 core::DerivedRecordId{2'000U + batch_id},
                                                 std::nullopt, correlation});
  lineage.correlations.push_back(core::CorrelationDeclaration{
      correlation, core::CorrelationPolicyRevision{1U},
      core::CorrelationTreatment::CovarianceInflationAndInformationCap, 6.0, 1'000.0 / 6.0});
  return lineage;
}

[[nodiscard]] core::ObservationLineage finalizedMapBatchLineage(
    std::uint64_t batch_id, const LidarRegistrationCloud& source,
    const FinalizedMapLidarFactorSnapshot& snapshot, double covariance_inflation,
    double information_cap) {
  core::ObservationLineage lineage;
  lineage.id = core::ObservationLineageId{100U + batch_id};
  lineage.checksum = presentHash(static_cast<std::uint8_t>(1U + batch_id % 250U));
  const core::DerivedRecordId consumer{2'000U + batch_id};
  const core::FactorGroupId factor_group{3'000U + batch_id};
  const core::CorrelationGroupId correlation{4'000U + batch_id};
  const auto observation_slice = [](core::MeasurementId measurement) {
    core::ObservationSlice slice;
    slice.root = measurement;
    slice.calibration = core::CalibrationEpoch{1U};
    slice.source_checksum = presentHash(static_cast<std::uint8_t>(1U + measurement.value() % 250U));
    return slice;
  };
  lineage.usage.push_back(core::ObservationUsage{observation_slice(source.source_sweep),
                                                 core::ObservationRole::PrimaryResidual, consumer,
                                                 factor_group, std::nullopt});

  std::set<core::MeasurementId> conditioning(source.imu_support.begin(), source.imu_support.end());
  for (const auto& owner : snapshot.owners()) {
    if (owner) {
      conditioning.insert(owner->sweep);
    }
  }
  for (const core::MeasurementId measurement : conditioning) {
    lineage.usage.push_back(core::ObservationUsage{observation_slice(measurement),
                                                   core::ObservationRole::ConditioningOnly,
                                                   consumer, std::nullopt, correlation});
  }
  lineage.correlations.push_back(
      core::CorrelationDeclaration{correlation, core::CorrelationPolicyRevision{3U},
                                   core::CorrelationTreatment::CovarianceInflationAndInformationCap,
                                   covariance_inflation, information_cap});
  return lineage;
}

[[nodiscard]] NavigationCovariance initialCovariance() {
  NavigationCovariance covariance;
  covariance.matrix.setIdentity();
  covariance.matrix *= 1.0e-3;
  return covariance;
}

[[nodiscard]] LocalGraphInitialization initialization() {
  core::NavStateEstimate estimate;
  estimate.velocity_odom = Eigen::Vector3d{kVelocityX, 0.0, 0.0};
  return LocalGraphInitialization{core::OdomEpoch{1U},   core::StateId{10U},
                                  core::FusionTime{0LL}, std::move(estimate),
                                  initialCovariance(),   initializationLineage(1U)};
}

[[nodiscard]] ImuInterval stationaryInterval(std::int64_t start_ns, std::int64_t end_ns,
                                             std::uint64_t measurement_offset) {
  ImuInterval interval;
  interval.support = core::TimeRange{core::FusionTime{start_ns}, core::FusionTime{end_ns}};
  constexpr std::size_t kSegments = 20U;
  for (std::size_t index = 0U; index <= kSegments; ++index) {
    const std::int64_t time = start_ns + (end_ns - start_ns) * static_cast<std::int64_t>(index) /
                                             static_cast<std::int64_t>(kSegments);
    const core::MeasurementId measurement{measurement_offset + index};
    interval.knots.push_back(
        InterpolatedImuSample{core::FusionTime{time}, Eigen::Vector3d{0.0, 0.0, kGravity},
                              Eigen::Vector3d::Zero(), measurement, measurement});
  }
  interval.maximum_raw_gap =
      core::Duration{(end_ns - start_ns) / static_cast<std::int64_t>(kSegments)};
  return interval;
}

[[nodiscard]] ImuKnotAppend stationaryKnot(std::uint64_t state, std::int64_t start_ns,
                                           std::int64_t end_ns, std::uint64_t lineage_id) {
  return ImuKnotAppend{core::StateId{state}, core::FusionTime{end_ns},
                       stationaryInterval(start_ns, end_ns, 10'000U + 100U * lineage_id),
                       initializationLineage(lineage_id)};
}

void initializeThreeStateGraph(LocalGraph* graph) {
  ASSERT_TRUE(graph->initialize(initialization()));
  ASSERT_TRUE(graph->appendImuKnot(stationaryKnot(11U, 0LL, kSecond, 2U)));
  ASSERT_TRUE(graph->appendImuKnot(stationaryKnot(12U, kSecond, 2LL * kSecond, 3U)));
}

struct DirectPairFixture {
  std::shared_ptr<const LidarFactorSnapshot> snapshot;
  core::RankAwareInformation physical_information;
};

struct DirectBatchFixture {
  core::MeasurementId source_sweep;
  std::shared_ptr<const LidarRegistrationCloud> source;
  LidarRegistrationConfig registration;
  LidarRegistrationResult registration_result;
  std::array<DirectPairFixture, 2U> pairs;
};

[[nodiscard]] std::vector<Eigen::Vector3d> volumeCluster(double x_offset) {
  std::vector<Eigen::Vector3d> points;
  for (int x = -2; x <= 2; ++x) {
    for (int y = -2; y <= 2; ++y) {
      for (int z = -1; z <= 1; ++z) {
        points.emplace_back(x_offset + 0.45 * static_cast<double>(x), 0.40 * static_cast<double>(y),
                            0.50 * static_cast<double>(z) + 0.03 * static_cast<double>(x * y));
      }
    }
  }
  return points;
}

[[nodiscard]] std::vector<Eigen::Vector3d> transformed(const std::vector<Eigen::Vector3d>& points,
                                                       const core::Pose3d& T_target_source) {
  std::vector<Eigen::Vector3d> output;
  output.reserve(points.size());
  for (const Eigen::Vector3d& point : points) {
    output.push_back(T_target_source * point);
  }
  return output;
}

[[nodiscard]] std::shared_ptr<const LidarRegistrationCloud> gaussianCloud(
    const std::vector<Eigen::Vector3d>& points, core::MeasurementId sweep, core::FusionTime time,
    const core::Pose3d& T_odom_imu_seed) {
  return test::sealedLidarRegistrationCloud(points, sweep, time, T_odom_imu_seed);
}

[[nodiscard]] core::RankAwareInformation pairInformation(
    const std::shared_ptr<const LidarFactorSnapshot>& snapshot,
    const LidarRegistrationConfig& config, double information_scale) {
  if (!snapshot) {
    return {};
  }
  const auto information = lidarFactorInformation(snapshot, config, information_scale);
  EXPECT_TRUE(information) << (information ? std::string{} : information.error().detail);
  return information ? information.value() : core::RankAwareInformation{};
}

[[nodiscard]] core::DirectionalObservability directionalObservability(
    const core::RankAwareInformation& information, core::StateId target_state,
    core::FusionTime target_time) {
  core::DirectionalObservability result;
  for (Eigen::Index output_index = 0; output_index < 6; ++output_index) {
    const Eigen::Index input_index = 5 - output_index;
    result.basis.col(output_index) = information.basis.col(input_index);
    result.eigenvalues(output_index) = std::max(0.0, information.eigenvalues(input_index));
  }
  result.rank = static_cast<std::uint32_t>(information.rank);
  const std::size_t unsupported = 6U - information.rank;
  if (unsupported == 0U) {
    result.absolute_eigenvalue_threshold =
        std::max(std::numeric_limits<double>::min(), 0.5 * result.eigenvalues(0));
  } else {
    const double lower = result.eigenvalues(static_cast<Eigen::Index>(unsupported - 1U));
    const double upper = result.eigenvalues(static_cast<Eigen::Index>(unsupported));
    result.absolute_eigenvalue_threshold = lower + 0.5 * (upper - lower);
  }
  result.relative_eigenvalue_threshold = 0.0;
  result.supported_variables = {core::DirectionalVariable::PoseTranslation,
                                core::DirectionalVariable::PoseRotation};
  result.endpoints = {
      {core::DirectionalEndpointRole::Target, target_state, target_time},
      {core::DirectionalEndpointRole::Source, core::StateId{12U}, core::FusionTime{2LL * kSecond}}};
  return result;
}

[[nodiscard]] core::DirectionalObservability unaryDirectionalObservability(
    const core::RankAwareInformation& information, core::StateId source_state,
    core::FusionTime source_time) {
  core::DirectionalObservability result;
  for (Eigen::Index output_index = 0; output_index < 6; ++output_index) {
    const Eigen::Index input_index = 5 - output_index;
    result.basis.col(output_index) = information.basis.col(input_index);
    result.eigenvalues(output_index) = std::max(0.0, information.eigenvalues(input_index));
  }
  result.rank = static_cast<std::uint32_t>(information.rank);
  const std::size_t unsupported = 6U - information.rank;
  result.absolute_eigenvalue_threshold =
      unsupported == 0U
          ? std::max(std::numeric_limits<double>::min(), 0.5 * result.eigenvalues(0))
          : result.eigenvalues(static_cast<Eigen::Index>(unsupported - 1U)) +
                0.5 * (result.eigenvalues(static_cast<Eigen::Index>(unsupported)) -
                       result.eigenvalues(static_cast<Eigen::Index>(unsupported - 1U)));
  result.relative_eigenvalue_threshold = 0.0;
  result.supported_variables = {core::DirectionalVariable::PoseTranslation,
                                core::DirectionalVariable::PoseRotation};
  result.endpoints = {{core::DirectionalEndpointRole::Unary, source_state, source_time}};
  return result;
}

[[nodiscard]] DirectBatchFixture directFixture(std::uint64_t sweep_id,
                                               double source_translation_x = 2.0 * kVelocityX,
                                               double target_distortion_m = 0.0) {
  DirectBatchFixture fixture;
  fixture.source_sweep = core::MeasurementId{sweep_id};
  fixture.registration.maximum_translation_information = 1'000.0;
  fixture.registration.source_voxel_size_m = 0.25;

  const std::vector<Eigen::Vector3d> first_source = volumeCluster(0.0);
  const std::vector<Eigen::Vector3d> second_source = volumeCluster(4.0);
  std::vector<Eigen::Vector3d> all_source = first_source;
  all_source.insert(all_source.end(), second_source.begin(), second_source.end());
  const core::Pose3d source_pose{Sophus::SO3d{}, Eigen::Vector3d{source_translation_x, 0.0, 0.0}};
  const core::Pose3d first_target_pose;
  const core::Pose3d second_target_pose{Sophus::SO3d{}, Eigen::Vector3d{kVelocityX, 0.0, 0.0}};
  fixture.source =
      gaussianCloud(all_source, fixture.source_sweep, core::FusionTime{2LL * kSecond}, source_pose);
  auto first_target_points = transformed(first_source, first_target_pose.inverse() * source_pose);
  auto second_target_points =
      transformed(second_source, second_target_pose.inverse() * source_pose);
  for (std::size_t index = 0U; index < first_target_points.size(); ++index) {
    const double pattern = static_cast<double>(static_cast<int>(index % 5U) - 2);
    first_target_points[index].z() += target_distortion_m * pattern;
  }
  for (std::size_t index = 0U; index < second_target_points.size(); ++index) {
    const double pattern = static_cast<double>(static_cast<int>(index % 7U) - 3);
    second_target_points[index].y() += target_distortion_m * pattern;
  }
  const auto first_target = gaussianCloud(first_target_points, core::MeasurementId{sweep_id + 1U},
                                          core::FusionTime{0LL}, first_target_pose);
  const auto second_target = gaussianCloud(second_target_points, core::MeasurementId{sweep_id + 2U},
                                           core::FusionTime{kSecond}, second_target_pose);
  const std::array<LidarRegistrationTarget, 2U> targets{
      LidarRegistrationTarget{core::StateId{10U}, core::FusionTime{0LL}, first_target,
                              first_target_pose, first_target_pose.inverse() * source_pose},
      LidarRegistrationTarget{core::StateId{11U}, core::FusionTime{kSecond}, second_target,
                              second_target_pose, second_target_pose.inverse() * source_pose}};
  const auto registered =
      registerLidarScan(core::StateId{12U}, fixture.source, targets, fixture.registration);
  EXPECT_TRUE(registered) << (registered ? std::string{} : registered.error().detail);
  if (!registered) {
    return fixture;
  }
  fixture.registration_result = registered.value();
  EXPECT_EQ(fixture.registration_result.target_snapshots.size(), fixture.pairs.size());
  if (fixture.registration_result.target_snapshots.size() != fixture.pairs.size()) {
    return fixture;
  }
  constexpr double kInformationScale = 1.0 / 6.0;
  for (std::size_t index = 0U; index < fixture.pairs.size(); ++index) {
    fixture.pairs[index].snapshot = fixture.registration_result.target_snapshots[index];
    fixture.pairs[index].physical_information =
        pairInformation(fixture.pairs[index].snapshot, fixture.registration, kInformationScale);
  }
  return fixture;
}

[[nodiscard]] DirectPairFixture overlappingSecondTarget(const DirectBatchFixture& fixture) {
  DirectPairFixture output;
  if (!fixture.source) {
    return output;
  }
  std::vector<Eigen::Vector3d> source_points;
  source_points.reserve(fixture.source->points.size());
  for (const LidarRegistrationPoint& point : fixture.source->points) {
    source_points.push_back(point.point);
  }
  const core::Pose3d target_pose{Sophus::SO3d{}, Eigen::Vector3d{kVelocityX, 0.0, 0.0}};
  const auto target = gaussianCloud(
      transformed(source_points, target_pose.inverse() * fixture.source->T_odom_imu_seed),
      core::MeasurementId{fixture.source_sweep.value() + 1'000U}, core::FusionTime{kSecond},
      target_pose);
  const LidarRegistrationTarget record{core::StateId{11U}, core::FusionTime{kSecond}, target,
                                       target_pose,
                                       target_pose.inverse() * fixture.source->T_odom_imu_seed};
  const auto registered = registerLidarScan(core::StateId{12U}, fixture.source,
                                            std::span{&record, 1U}, fixture.registration);
  EXPECT_TRUE(registered) << (registered ? std::string{} : registered.error().detail);
  if (!registered || registered.value().target_snapshots.size() != 1U) {
    return output;
  }
  output.snapshot = registered.value().target_snapshots.front();
  output.physical_information = pairInformation(output.snapshot, fixture.registration, 1.0 / 6.0);
  return output;
}

[[nodiscard]] LidarDirectFactorBatch directBatch(
    const DirectBatchFixture& fixture, std::uint64_t batch_id,
    core::SensorHealthState health_state = core::SensorHealthState::Active) {
  LidarDirectFactorBatch output;
  output.metadata.header.schema_version = 1U;
  output.metadata.header.trace = core::TraceId{10U + batch_id};
  output.metadata.header.producer = core::ProducerId{20U};
  output.metadata.header.session = core::SessionId{30U};
  output.metadata.header.config = core::ConfigRevision{40U};
  output.metadata.header.direct_calibration = core::CalibrationEpoch{5U};
  output.metadata.batch_id = core::FactorBatchId{batch_id};
  output.metadata.odom_epoch = core::OdomEpoch{1U};
  output.metadata.sensor = core::SensorInstanceId::lidar(core::LidarId{1U});
  output.metadata.timing.support =
      core::TimeRange{core::FusionTime{0LL}, core::FusionTime{2LL * kSecond + 1LL}};
  output.metadata.timing.measurement_timestamps = {core::FusionTime{0LL}, core::FusionTime{kSecond},
                                                   core::FusionTime{2LL * kSecond}};
  output.metadata.timing.reference_time = core::FusionTime{2LL * kSecond};
  output.metadata.health.sensor = output.metadata.sensor;
  output.metadata.health.state = health_state;
  output.metadata.health.recovery_epoch = core::SensorRecoveryEpoch{0U};
  output.metadata.health.transition_sequence = 0U;
  output.metadata.health.assessed_at =
      core::FusionTime{2LL * kSecond + 100LL + static_cast<std::int64_t>(batch_id)};
  output.metadata.timing.produced_at =
      core::FusionTime{2LL * kSecond + 200LL + static_cast<std::int64_t>(batch_id)};
  output.metadata.header.created_at =
      core::FusionTime{2LL * kSecond + 300LL + static_cast<std::int64_t>(batch_id)};
  output.metadata.map_eligible = health_state == core::SensorHealthState::Active;
  std::vector<core::MeasurementId> target_sweeps;
  target_sweeps.reserve(fixture.pairs.size());
  for (const DirectPairFixture& pair : fixture.pairs) {
    target_sweeps.push_back(pair.snapshot->targetSweep());
  }
  output.metadata.lineage = batchLineage(batch_id, fixture.source_sweep, target_sweeps);
  output.source_state = core::StateId{12U};
  output.source_time = core::FusionTime{2LL * kSecond};
  output.registration = fixture.registration;
  output.registration_report =
      DirectLidarRegistrationReport{fixture.registration_result.termination,
                                    fixture.registration_result.initial_robust_cost,
                                    fixture.registration_result.final_robust_cost,
                                    fixture.registration_result.diagnostics,
                                    fixture.registration_result.work,
                                    fixture.registration_result.T_odom_source,
                                    fixture.registration_result.source_right_correction};
  for (std::size_t index = 0U; index < fixture.pairs.size(); ++index) {
    const core::StateId target_state{10U + index};
    const core::FusionTime target_time{static_cast<std::int64_t>(index) * kSecond};
    output.pairs.push_back(LidarDirectFactorPairSpec{target_state, target_time,
                                                     fixture.pairs[index].snapshot, 1.0 / 6.0});
    output.metadata.directional_observability.push_back(directionalObservability(
        fixture.pairs[index].physical_information, target_state, target_time));
  }
  return output;
}

[[nodiscard]] LidarDirectFactorBatch finalizedMapBatch(
    const test::FinalizedMapRegistrationFixture& fixture, std::uint64_t batch_id,
    double configured_correlation_inflation_floor = 1.0) {
  const auto snapshot = fixture.registration.finalized_map_snapshot;
  if (!snapshot || !fixture.source) {
    throw std::runtime_error("finalized-map graph test fixture is incomplete");
  }
  const auto owner_inflation = lidarFinalizedMapOwnerPoseCovarianceInflation(snapshot);
  if (!owner_inflation) {
    throw std::runtime_error("finalized-map graph test owner inflation failed");
  }
  constexpr double kBaseCovarianceInflation = 6.0;
  const double effective_inflation =
      std::max(owner_inflation.value(), configured_correlation_inflation_floor);
  const double total_inflation = kBaseCovarianceInflation * effective_inflation;
  const double information_scale = 1.0 / total_inflation;
  const auto information =
      lidarFinalizedMapFactorInformation(snapshot, fixture.config, information_scale);
  if (!information) {
    throw std::runtime_error("finalized-map graph test information failed: " +
                             information.error().detail);
  }

  LidarDirectFactorBatch output;
  output.metadata.header.schema_version = 1U;
  output.metadata.header.trace = core::TraceId{10U + batch_id};
  output.metadata.header.producer = core::ProducerId{20U};
  output.metadata.header.session = core::SessionId{30U};
  output.metadata.header.config = core::ConfigRevision{40U};
  output.metadata.header.direct_calibration = core::CalibrationEpoch{1U};
  output.metadata.batch_id = core::FactorBatchId{batch_id};
  output.metadata.odom_epoch = core::OdomEpoch{1U};
  output.metadata.sensor = core::SensorInstanceId::lidar(core::LidarId{1U});
  output.metadata.timing.support =
      core::TimeRange{core::FusionTime{0LL}, fixture.source->reference_time + core::Duration{1LL}};
  output.metadata.timing.measurement_timestamps = {fixture.source->reference_time};
  output.metadata.timing.reference_time = fixture.source->reference_time;
  output.metadata.health = core::SensorHealthSnapshot{
      output.metadata.sensor, core::SensorHealthState::Active, core::SensorRecoveryEpoch{0U}, 0U,
      fixture.source->reference_time + core::Duration{100LL + static_cast<std::int64_t>(batch_id)}};
  output.metadata.timing.produced_at =
      fixture.source->reference_time + core::Duration{200LL + static_cast<std::int64_t>(batch_id)};
  output.metadata.header.created_at =
      fixture.source->reference_time + core::Duration{300LL + static_cast<std::int64_t>(batch_id)};
  output.metadata.map_eligible = true;
  output.metadata.directional_observability.push_back(unaryDirectionalObservability(
      information.value(), fixture.registration.source_state, fixture.registration.source_time));
  output.metadata.lineage =
      finalizedMapBatchLineage(
          batch_id, *fixture.source, *snapshot, kBaseCovarianceInflation,
          fixture.config.maximum_translation_information / kBaseCovarianceInflation);
  output.source_state = fixture.registration.source_state;
  output.source_time = fixture.registration.source_time;
  output.registration = fixture.config;
  output.base_covariance_inflation = kBaseCovarianceInflation;
  output.registration_report =
      DirectLidarRegistrationReport{fixture.registration.termination,
                                    fixture.registration.initial_robust_cost,
                                    fixture.registration.final_robust_cost,
                                    fixture.registration.diagnostics,
                                    fixture.registration.work,
                                    fixture.registration.T_odom_source,
                                    fixture.registration.source_right_correction};
  output.finalized_map = LidarFinalizedMapFactorSpec{
      std::move(snapshot), configured_correlation_inflation_floor, information_scale};
  return output;
}

[[nodiscard]] SensorFactorBatchRef batchRef(std::uint64_t batch_id) {
  return SensorFactorBatchRef{core::SensorInstanceId::lidar(core::LidarId{1U}),
                              core::FactorBatchId{batch_id}};
}

TEST(FactorBatchGraph, InsertsPoseAwareMultiTargetDirectFactorsWithoutNewNavigationOrImu) {
  LocalGraph graph;
  initializeThreeStateGraph(&graph);
  const LocalGraphCommit before = graph.estimate().value();
  const DirectBatchFixture fixture = directFixture(41U);
  for (const DirectPairFixture& pair : fixture.pairs) {
    ASSERT_NE(pair.snapshot, nullptr);
    EXPECT_GE(pair.snapshot->rows().size(), fixture.registration.minimum_correspondences);
    EXPECT_GT(pair.physical_information.rank, 0U);
    EXPECT_LE(pair.physical_information.rank, 6U);
  }
  ASSERT_NE(before.navigation_poses[0].T_odom_imu.translation().x(),
            before.navigation_poses[1].T_odom_imu.translation().x());

  const auto inserted = graph.insertFactorBatch(directBatch(fixture, 1U));
  ASSERT_TRUE(inserted) << (inserted ? std::string{} : inserted.error().detail);
  EXPECT_EQ(inserted.value().solve.navigation_states, before.solve.navigation_states);
  EXPECT_EQ(inserted.value().solve.combined_imu_factors, before.solve.combined_imu_factors);
  EXPECT_EQ(inserted.value().navigation_poses.size(), before.navigation_poses.size());
  EXPECT_EQ(inserted.value().solve.active_factor_batches, 1U);
  EXPECT_EQ(inserted.value().solve.active_lidar_direct_batch_factors, 2U);
  EXPECT_EQ(inserted.value().solve.lidar_direct_batch_factors_added, 2U);
  ASSERT_TRUE(inserted.value().lidar_registration);
  EXPECT_EQ(inserted.value().lidar_registration->termination,
            fixture.registration_result.termination);
  EXPECT_EQ(inserted.value().lidar_registration->diagnostics.correspondences,
            fixture.registration_result.diagnostics.correspondences);
  EXPECT_EQ(inserted.value().lidar_registration->work.source_points_selected,
            fixture.registration_result.work.source_points_selected);
  ASSERT_EQ(inserted.value().lidar_pairs.size(), 2U);
  EXPECT_EQ(inserted.value().lidar_pairs[0].snapshot_checksum,
            fixture.pairs[0].snapshot->checksum());
  EXPECT_DOUBLE_EQ(inserted.value().lidar_pairs[0].information_scale, 1.0 / 6.0);

  const auto provenance = graph.factorBatchProvenance(batchRef(1U));
  ASSERT_TRUE(provenance);
  ASSERT_EQ(provenance->batch.pairs.size(), 2U);
  EXPECT_EQ(provenance->batch.source_state, core::StateId{12U});
  EXPECT_EQ(provenance->batch.pairs[0].target_state, core::StateId{10U});
  EXPECT_EQ(provenance->batch.pairs[1].target_state, core::StateId{11U});
  EXPECT_NE(provenance->batch.pairs[0].snapshot.get(), provenance->batch.pairs[1].snapshot.get());

  std::set<std::size_t> owned_rows;
  for (const LidarDirectFactorPairSpec& pair : provenance->batch.pairs) {
    for (const FrozenPointCorrespondence& row : pair.snapshot->rows()) {
      EXPECT_TRUE(owned_rows.insert(row.source_point_storage_index).second);
    }
  }
  EXPECT_EQ(owned_rows.size(), fixture.registration_result.diagnostics.correspondences);
  EXPECT_EQ(graph.factorBatchJournalStats().active_lidar_direct_factors, 2U);
}

TEST(FactorBatchGraph, InsertsAndRemovesOneFinalizedMapUnaryFactorAtomically) {
  LocalGraph graph;
  initializeThreeStateGraph(&graph);
  const LocalGraphCommit before = graph.estimate().value();
  const test::FinalizedMapRegistrationFixture fixture = test::finalizedMapRegistrationFixture();
  const LidarDirectFactorBatch batch =
      finalizedMapBatch(fixture, 5U, 34.629718841665124);
  ASSERT_TRUE(batch.pairs.empty());
  ASSERT_TRUE(batch.finalized_map);

  const auto inserted = graph.insertFactorBatch(batch);
  ASSERT_TRUE(inserted) << (inserted ? std::string{} : inserted.error().detail);
  EXPECT_EQ(inserted.value().revision.value(), before.revision.value() + 1U);
  EXPECT_EQ(inserted.value().solve.navigation_states, before.solve.navigation_states);
  EXPECT_EQ(inserted.value().solve.combined_imu_factors, before.solve.combined_imu_factors);
  EXPECT_EQ(inserted.value().solve.active_factor_batches, 1U);
  EXPECT_EQ(inserted.value().solve.active_lidar_direct_batch_factors, 1U);
  EXPECT_EQ(inserted.value().solve.lidar_direct_batch_factors_added, 1U);
  EXPECT_TRUE(inserted.value().lidar_pairs.empty());
  ASSERT_TRUE(inserted.value().lidar_finalized_map);
  EXPECT_EQ(inserted.value().lidar_finalized_map->map_version,
            batch.finalized_map->snapshot->mapVersion());
  EXPECT_EQ(inserted.value().lidar_finalized_map->map_checksum,
            batch.finalized_map->snapshot->mapChecksum());
  EXPECT_EQ(inserted.value().lidar_finalized_map->unique_finalized_owners,
            batch.finalized_map->snapshot->uniqueOwnerCount());
  EXPECT_DOUBLE_EQ(inserted.value().lidar_finalized_map->configured_correlation_inflation_floor,
                   batch.finalized_map->configured_correlation_inflation_floor);
  EXPECT_DOUBLE_EQ(
      inserted.value().lidar_finalized_map->effective_covariance_inflation,
      std::max(inserted.value().lidar_finalized_map->owner_pose_covariance_inflation,
               batch.finalized_map->configured_correlation_inflation_floor));
  EXPECT_DOUBLE_EQ(inserted.value().lidar_finalized_map->information_scale,
                   batch.finalized_map->information_scale);

  const auto provenance = graph.factorBatchProvenance(batchRef(5U));
  ASSERT_TRUE(provenance);
  EXPECT_TRUE(provenance->batch.pairs.empty());
  ASSERT_TRUE(provenance->batch.finalized_map);
  EXPECT_EQ(provenance->batch.finalized_map->snapshot.get(), batch.finalized_map->snapshot.get());
  EXPECT_TRUE(provenance->removable);

  FactorBatchRemovalRequest removal;
  removal.batches = {batchRef(5U)};
  removal.reason = FactorBatchRemovalReason::FrontendInvalidation;
  const auto removed = graph.removeFactorBatches(std::move(removal));
  ASSERT_TRUE(removed) << (removed ? std::string{} : removed.error().detail);
  EXPECT_EQ(removed.value().solve.factor_batches_removed, 1U);
  EXPECT_EQ(removed.value().solve.lidar_direct_batch_factors_removed, 1U);
  EXPECT_EQ(removed.value().solve.active_factor_batches, 0U);
  EXPECT_EQ(removed.value().solve.active_lidar_direct_batch_factors, 0U);
  const auto terminal = graph.factorBatchProvenance(batchRef(5U));
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->status, FactorBatchJournalStatus::Removed);
  EXPECT_FALSE(terminal->removable);
}

TEST(FactorBatchGraph, RejectsFinalizedMapPolicyProvenanceAndObservabilityTamperingAtomically) {
  LocalGraph graph;
  initializeThreeStateGraph(&graph);
  const LocalGraphCommit before = graph.estimate().value();
  const test::FinalizedMapRegistrationFixture fixture = test::finalizedMapRegistrationFixture();

  const auto expect_rejected = [&](LidarDirectFactorBatch batch, std::string_view detail) {
    const auto rejected = graph.insertFactorBatch(std::move(batch));
    ASSERT_FALSE(rejected);
    EXPECT_NE(rejected.error().detail.find(detail), std::string::npos) << rejected.error().detail;
    EXPECT_EQ(graph.estimate().value().revision, before.revision);
    EXPECT_EQ(graph.factorBatchJournalStats().active_batches, 0U);
  };

  LidarDirectFactorBatch stale_policy = finalizedMapBatch(fixture, 60U);
  stale_policy.metadata.lineage.correlations.front().policy = core::CorrelationPolicyRevision{1U};
  expect_rejected(std::move(stale_policy), "revision-specific");

  LidarDirectFactorBatch wrong_base = finalizedMapBatch(fixture, 61U);
  wrong_base.base_covariance_inflation = 5.0;
  expect_rejected(std::move(wrong_base), "base and owner covariance inflation");

  LidarDirectFactorBatch wrong_scale = finalizedMapBatch(fixture, 62U);
  wrong_scale.finalized_map->information_scale *= 0.5;
  expect_rejected(std::move(wrong_scale), "directional observability");

  LidarDirectFactorBatch invalid_floor = finalizedMapBatch(fixture, 66U);
  invalid_floor.finalized_map->configured_correlation_inflation_floor = 0.0;
  expect_rejected(std::move(invalid_floor), "identity");

  LidarDirectFactorBatch wrong_endpoint = finalizedMapBatch(fixture, 63U);
  wrong_endpoint.metadata.directional_observability.back().endpoints.front().role =
      core::DirectionalEndpointRole::Source;
  expect_rejected(std::move(wrong_endpoint), "metadata validation");

  LidarDirectFactorBatch omitted_owner_sweep = finalizedMapBatch(fixture, 64U);
  ASSERT_FALSE(omitted_owner_sweep.finalized_map->snapshot->owners().empty());
  const core::MeasurementId required_sweep =
      omitted_owner_sweep.finalized_map->snapshot->owners().front()->sweep;
  std::erase_if(omitted_owner_sweep.metadata.lineage.usage,
                [&](const core::ObservationUsage& usage) {
    const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
    return measurement != nullptr && *measurement == required_sweep;
  });
  expect_rejected(std::move(omitted_owner_sweep), "finalized-owner sweep");

  LidarDirectFactorBatch duplicate_conditioning = finalizedMapBatch(fixture, 65U);
  duplicate_conditioning.metadata.lineage.usage.push_back(
      duplicate_conditioning.metadata.lineage.usage.back());
  expect_rejected(std::move(duplicate_conditioning), "deduplicated");

  LidarDirectFactorBatch wrong_frontend_pose = finalizedMapBatch(fixture, 66U);
  wrong_frontend_pose.registration_report.T_odom_source = core::Pose3d{};
  expect_rejected(std::move(wrong_frontend_pose), "registration report");
}

TEST(FactorBatchGraph, MapOnlyBatchRemainsRemovableUntilItsSourceStateFinalizes) {
  LocalGraph graph;
  initializeThreeStateGraph(&graph);
  const test::FinalizedMapRegistrationFixture fixture = test::finalizedMapRegistrationFixture();
  ASSERT_TRUE(graph.insertFactorBatch(finalizedMapBatch(fixture, 70U)));

  const auto earlier_finality =
      graph.appendImuKnot(stationaryKnot(13U, 2LL * kSecond, 5'500'000'000LL, 71U));
  ASSERT_TRUE(earlier_finality) << (earlier_finality ? std::string{}
                                                     : earlier_finality.error().detail);
  ASSERT_FALSE(earlier_finality.value().finalized_states.empty());
  EXPECT_EQ(earlier_finality.value().finalized_states.back().state, core::StateId{10U});
  const auto still_live = graph.factorBatchProvenance(batchRef(70U));
  ASSERT_TRUE(still_live);
  EXPECT_EQ(still_live->status, FactorBatchJournalStatus::Active);
  EXPECT_TRUE(still_live->removable);

  const auto source_finality =
      graph.appendImuKnot(stationaryKnot(14U, 5'500'000'000LL, 8'000'000'000LL, 72U));
  ASSERT_TRUE(source_finality) << (source_finality ? std::string{}
                                                   : source_finality.error().detail);
  ASSERT_FALSE(source_finality.value().finalized_states.empty());
  EXPECT_EQ(source_finality.value().finalized_states.back().state, core::StateId{12U});
  const auto terminal = graph.factorBatchProvenance(batchRef(70U));
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->status, FactorBatchJournalStatus::FinalizedByMarginalization);
  EXPECT_FALSE(terminal->removable);
  EXPECT_EQ(graph.factorBatchJournalStats().active_batches, 0U);
}

TEST(FactorBatchGraph, MixedBatchSealsWhenItsLiveTargetFinalizesAndTerminatesAtSourceFinality) {
  LocalGraph graph;
  initializeThreeStateGraph(&graph);
  test::FinalizedMapRegistrationFixture fixture = test::finalizedMapRegistrationFixture();

  const core::Pose3d target_pose{Sophus::SO3d{}, Eigen::Vector3d{kVelocityX, 0.0, 0.0}};
  std::vector<Eigen::Vector3d> live_target_points;
  for (const LidarRegistrationPoint& point : fixture.source->points) {
    if (point.point.x() < 0.0) {
      live_target_points.push_back(target_pose.inverse() * fixture.source->T_odom_imu_seed *
                                   point.point);
    }
  }
  ASSERT_GE(live_target_points.size(), fixture.config.minimum_correspondences);
  const auto live_target_cloud = test::sealedLidarRegistrationCloud(
      live_target_points, core::MeasurementId{901U}, core::FusionTime{kSecond}, target_pose);
  const LidarRegistrationTarget live_target{
      core::StateId{11U}, core::FusionTime{kSecond}, live_target_cloud, target_pose,
      target_pose.inverse() * fixture.source->T_odom_imu_seed};
  const auto mixed_registration =
      registerLidarScan(core::StateId{12U}, fixture.source, std::span{&live_target, 1U},
                        fixture.map.readView(), core::LocalGraphRevision{10U}, fixture.config);
  ASSERT_TRUE(mixed_registration) << (mixed_registration ? std::string{}
                                                         : mixed_registration.error().detail);
  ASSERT_EQ(mixed_registration.value().target_snapshots.size(), 1U);
  ASSERT_TRUE(mixed_registration.value().finalized_map_snapshot);
  fixture.registration = mixed_registration.value();

  LidarDirectFactorBatch batch = finalizedMapBatch(fixture, 75U);
  const auto live_snapshot = fixture.registration.target_snapshots.front();
  const double live_information_scale = 1.0 / batch.base_covariance_inflation;
  const double map_information_scale = batch.finalized_map->information_scale;
  ASSERT_GT(batch.finalized_map->snapshot->ownerPoseCovarianceInflation(), 1.0);
  EXPECT_LT(map_information_scale, live_information_scale);
  EXPECT_DOUBLE_EQ(batch.metadata.lineage.correlations.front().covariance_inflation,
                   batch.base_covariance_inflation);
  const auto live_information =
      lidarFactorInformation(live_snapshot, fixture.config, live_information_scale);
  ASSERT_TRUE(live_information) << (live_information ? std::string{}
                                                     : live_information.error().detail);
  batch.pairs.push_back(LidarDirectFactorPairSpec{
      live_snapshot->targetState(), live_snapshot->targetTime(), live_snapshot,
      live_information_scale});
  batch.metadata.directional_observability.insert(
      batch.metadata.directional_observability.begin(),
      directionalObservability(live_information.value(), live_snapshot->targetState(),
                               live_snapshot->targetTime()));
  core::ObservationSlice target_slice;
  target_slice.root = live_snapshot->targetSweep();
  target_slice.calibration = core::CalibrationEpoch{1U};
  target_slice.source_checksum = presentHash(91U);
  const core::ObservationUsage& primary = batch.metadata.lineage.usage.front();
  batch.metadata.lineage.usage.push_back(core::ObservationUsage{
      std::move(target_slice), core::ObservationRole::ConditioningOnly, primary.consumer,
      std::nullopt, batch.metadata.lineage.correlations.front().group});

  // A coherent-looking revision-2 regression must still fail even if its
  // directional metadata is recomputed to match the incorrectly weakened
  // live factor. Finalized-owner uncertainty is local to the unary map
  // channel; it can never become the live binary scale.
  LidarDirectFactorBatch owner_inflated_live = batch;
  owner_inflated_live.pairs.front().information_scale = map_information_scale;
  const auto weakened_live_information =
      lidarFactorInformation(live_snapshot, fixture.config, map_information_scale);
  ASSERT_TRUE(weakened_live_information)
      << (weakened_live_information ? std::string{}
                                    : weakened_live_information.error().detail);
  owner_inflated_live.metadata.directional_observability.front() = directionalObservability(
      weakened_live_information.value(), live_snapshot->targetState(), live_snapshot->targetTime());
  const auto rejected_owner_leak = graph.insertFactorBatch(std::move(owner_inflated_live));
  ASSERT_FALSE(rejected_owner_leak);
  EXPECT_NE(rejected_owner_leak.error().detail.find("per-channel scaling"), std::string::npos)
      << rejected_owner_leak.error().detail;
  EXPECT_EQ(graph.factorBatchJournalStats().active_batches, 0U);

  const auto inserted = graph.insertFactorBatch(batch);
  ASSERT_TRUE(inserted) << (inserted ? std::string{} : inserted.error().detail);
  EXPECT_EQ(inserted.value().solve.lidar_direct_batch_factors_added, 2U);
  EXPECT_EQ(inserted.value().lidar_pairs.size(), 1U);
  EXPECT_DOUBLE_EQ(inserted.value().lidar_pairs.front().information_scale,
                   live_information_scale);
  EXPECT_TRUE(inserted.value().lidar_finalized_map);
  EXPECT_DOUBLE_EQ(inserted.value().lidar_finalized_map->information_scale,
                   map_information_scale);

  const auto target_finality =
      graph.appendImuKnot(stationaryKnot(13U, 2LL * kSecond, 6'500'000'000LL, 76U));
  ASSERT_TRUE(target_finality) << (target_finality ? std::string{}
                                                   : target_finality.error().detail);
  ASSERT_FALSE(target_finality.value().finalized_states.empty());
  EXPECT_EQ(target_finality.value().finalized_states.back().state, core::StateId{11U});
  const auto sealed = graph.factorBatchProvenance(batchRef(75U));
  ASSERT_TRUE(sealed);
  EXPECT_EQ(sealed->status, FactorBatchJournalStatus::SealedByMarginalization);
  EXPECT_FALSE(sealed->removable);

  const auto source_finality =
      graph.appendImuKnot(stationaryKnot(14U, 6'500'000'000LL, 8'000'000'000LL, 77U));
  ASSERT_TRUE(source_finality) << (source_finality ? std::string{}
                                                   : source_finality.error().detail);
  ASSERT_FALSE(source_finality.value().finalized_states.empty());
  EXPECT_EQ(source_finality.value().finalized_states.back().state, core::StateId{12U});
  const auto terminal = graph.factorBatchProvenance(batchRef(75U));
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->status, FactorBatchJournalStatus::FinalizedByMarginalization);
}

TEST(FactorBatchGraph, RejectsMissingUnityOrMismatchedLidarConditioningPolicyAtomically) {
  LocalGraph graph;
  initializeThreeStateGraph(&graph);
  const LocalGraphCommit before = graph.estimate().value();
  const DirectBatchFixture fixture = directFixture(44U);

  const auto expect_rejected = [&](LidarDirectFactorBatch batch, std::string_view detail) {
    const auto rejected = graph.insertFactorBatch(std::move(batch));
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, LocalGraphErrorCode::InvalidFactorBatch);
    EXPECT_NE(rejected.error().detail.find(detail), std::string::npos);
    EXPECT_EQ(graph.estimate().value().revision, before.revision);
    EXPECT_EQ(graph.factorBatchJournalStats().active_batches, 0U);
  };

  LidarDirectFactorBatch unity = directBatch(fixture, 50U);
  unity.metadata.lineage.correlations.front().covariance_inflation = 1.0;
  unity.metadata.lineage.correlations.front().total_information_cap = 1'000.0;
  expect_rejected(std::move(unity), "unity");

  LidarDirectFactorBatch missing_group = directBatch(fixture, 51U);
  ASSERT_GT(missing_group.metadata.lineage.usage.size(), 1U);
  missing_group.metadata.lineage.usage[1].correlation_group.reset();
  expect_rejected(std::move(missing_group), "conditioning-only");

  LidarDirectFactorBatch missing_target = directBatch(fixture, 52U);
  const core::MeasurementId omitted_target = missing_target.pairs.front().snapshot->targetSweep();
  std::erase_if(missing_target.metadata.lineage.usage, [&](const core::ObservationUsage& usage) {
    const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
    return measurement != nullptr && *measurement == omitted_target;
  });
  expect_rejected(std::move(missing_target), "omits a live target");

  LidarDirectFactorBatch mismatched_cap = directBatch(fixture, 53U);
  mismatched_cap.metadata.lineage.correlations.front().total_information_cap = 1'000.0 / 5.0;
  expect_rejected(std::move(mismatched_cap), "declared information cap disagree");

  LidarDirectFactorBatch imu_primary = directBatch(fixture, 54U);
  core::ObservationUsage& imu_usage = imu_primary.metadata.lineage.usage.back();
  ASSERT_EQ(imu_usage.role, core::ObservationRole::ConditioningOnly);
  imu_usage.role = core::ObservationRole::PrimaryResidual;
  imu_usage.factor_group = imu_primary.metadata.lineage.usage.front().factor_group;
  expect_rejected(std::move(imu_primary), "primary lineage");
}

TEST(FactorBatchGraph, AcceptsSnapshotVerifiedFinalCostWhenInitialAssociationCostIsLower) {
  LocalGraph graph;
  initializeThreeStateGraph(&graph);
  const DirectBatchFixture fixture = directFixture(48U, 2.0 * kVelocityX, 0.01);
  LidarDirectFactorBatch batch = directBatch(fixture, 41U);
  ASSERT_GT(batch.registration_report.final_robust_cost, 0.0);
  batch.registration_report.initial_robust_cost = 0.5 * batch.registration_report.final_robust_cost;

  const auto inserted = graph.insertFactorBatch(std::move(batch));

  ASSERT_TRUE(inserted) << (inserted ? std::string{} : inserted.error().detail);
  ASSERT_TRUE(inserted.value().lidar_registration);
  EXPECT_GT(inserted.value().lidar_registration->final_robust_cost,
            inserted.value().lidar_registration->initial_robust_cost);
  EXPECT_EQ(graph.factorBatchJournalStats().active_batches, 1U);
}

TEST(FactorBatchGraph, RejectsTamperedFinalSnapshotCostAtomically) {
  LocalGraph graph;
  initializeThreeStateGraph(&graph);
  const LocalGraphCommit before = graph.estimate().value();
  const DirectBatchFixture fixture = directFixture(49U);
  LidarDirectFactorBatch batch = directBatch(fixture, 42U);
  batch.registration_report.final_robust_cost +=
      1.0e-3 * std::max(1.0, std::abs(batch.registration_report.final_robust_cost));

  const auto rejected = graph.insertFactorBatch(std::move(batch));

  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalGraphErrorCode::InvalidFactorBatch);
  EXPECT_NE(rejected.error().detail.find("final robust cost report/rows"), std::string::npos);
  EXPECT_EQ(graph.estimate().value().revision, before.revision);
  EXPECT_EQ(graph.factorBatchJournalStats().active_batches, 0U);
}

TEST(FactorBatchGraph, RejectsOverlappingSourceRowsAtomically) {
  LocalGraph graph;
  initializeThreeStateGraph(&graph);
  const LocalGraphCommit before = graph.estimate().value();
  const DirectBatchFixture fixture = directFixture(42U);
  LidarDirectFactorBatch overlapping = directBatch(fixture, 2U);
  ASSERT_EQ(overlapping.pairs.size(), 2U);
  const DirectPairFixture second_target = overlappingSecondTarget(fixture);
  ASSERT_TRUE(second_target.snapshot);
  ASSERT_EQ(second_target.snapshot->targetState(), core::StateId{11U});
  ASSERT_EQ(second_target.snapshot->sourceState(), core::StateId{12U});
  overlapping.pairs[1].snapshot = second_target.snapshot;
  overlapping.metadata.directional_observability[1] = directionalObservability(
      second_target.physical_information, core::StateId{11U}, core::FusionTime{kSecond});

  const auto rejected = graph.insertFactorBatch(std::move(overlapping));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalGraphErrorCode::FactorBatchLineageConflict);
  const LocalGraphCommit after = graph.estimate().value();
  EXPECT_EQ(after.revision, before.revision);
  EXPECT_EQ(after.solve.navigation_states, before.solve.navigation_states);
  EXPECT_EQ(after.solve.combined_imu_factors, before.solve.combined_imu_factors);
  EXPECT_EQ(graph.factorBatchJournalStats().active_batches, 0U);
}

TEST(FactorBatchGraph, RejectedCandidateRetainsRegistrationAndBoundedSolveDiagnostics) {
  LocalGraphConfig config;
  config.maximum_transaction_translation_correction_m = 1.0e-6;
  LocalGraph graph(config);
  initializeThreeStateGraph(&graph);
  const LocalGraphCommit before = graph.estimate().value();
  const DirectBatchFixture fixture = directFixture(46U, 0.15);

  const auto rejected = graph.insertFactorBatch(directBatch(fixture, 3U));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalGraphErrorCode::PoseCorrectionLimit);
  ASSERT_TRUE(rejected.error().lidar_registration);
  EXPECT_EQ(rejected.error().lidar_registration->termination,
            fixture.registration_result.termination);
  EXPECT_EQ(rejected.error().lidar_registration->work.source_points_selected,
            fixture.registration_result.work.source_points_selected);
  EXPECT_EQ(rejected.error().lidar_pairs.size(), 2U);
  ASSERT_TRUE(rejected.error().rejected_solve);
  EXPECT_GT(rejected.error().rejected_solve->nonlinear_iterations, 0U);
  EXPECT_GT(rejected.error().rejected_solve->maximum_transaction_translation_correction_m,
            config.maximum_transaction_translation_correction_m);
  ASSERT_TRUE(rejected.error().factor_batch);
  EXPECT_EQ(*rejected.error().factor_batch, batchRef(3U));
  EXPECT_EQ(graph.estimate().value().revision, before.revision);
  EXPECT_EQ(graph.factorBatchJournalStats().active_batches, 0U);
}

TEST(FactorBatchGraph, KeepsEveryNonActiveHealthStateShadowOnly) {
  for (const core::SensorHealthState state :
       {core::SensorHealthState::Suspect, core::SensorHealthState::Failed,
        core::SensorHealthState::Recovering}) {
    LocalGraph graph;
    initializeThreeStateGraph(&graph);
    const LocalGraphCommit before = graph.estimate().value();
    const DirectBatchFixture fixture = directFixture(50U + static_cast<std::uint64_t>(state));

    const auto rejected = graph.insertFactorBatch(directBatch(fixture, 10U, state));
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, LocalGraphErrorCode::FactorBatchHealthUnavailable);
    EXPECT_EQ(graph.estimate().value().revision, before.revision);
    EXPECT_EQ(graph.factorBatchJournalStats().active_batches, 0U);
  }
}

TEST(FactorBatchGraph, RemovalIsAtomicAndRetainsExactTypedProvenance) {
  LocalGraph graph;
  initializeThreeStateGraph(&graph);
  const DirectBatchFixture fixture = directFixture(43U);
  const auto inserted = graph.insertFactorBatch(directBatch(fixture, 7U));
  ASSERT_TRUE(inserted) << (inserted ? std::string{} : inserted.error().detail);
  const SensorFactorBatchRef known = batchRef(7U);
  const SensorFactorBatchRef unknown = batchRef(99U);

  const auto rejected = graph.removeFactorBatches(
      FactorBatchRemovalRequest{{known, unknown}, FactorBatchRemovalReason::SensorFailure});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalGraphErrorCode::FactorBatchRemovalUnavailable);
  EXPECT_EQ(graph.estimate().value().revision, inserted.value().revision);
  EXPECT_EQ(graph.factorBatchJournalStats().active_batches, 1U);

  const auto removed = graph.removeFactorBatches(
      FactorBatchRemovalRequest{{known}, FactorBatchRemovalReason::SensorFailure});
  ASSERT_TRUE(removed) << (removed ? std::string{} : removed.error().detail);
  EXPECT_EQ(removed.value().solve.factor_batches_removed, 1U);
  EXPECT_EQ(removed.value().solve.lidar_direct_batch_factors_removed, 2U);
  EXPECT_EQ(removed.value().solve.navigation_states, inserted.value().solve.navigation_states);
  EXPECT_EQ(removed.value().solve.combined_imu_factors,
            inserted.value().solve.combined_imu_factors);

  const auto provenance = graph.factorBatchProvenance(known);
  ASSERT_TRUE(provenance);
  EXPECT_EQ(provenance->status, FactorBatchJournalStatus::Removed);
  EXPECT_FALSE(provenance->removable);
  ASSERT_EQ(provenance->batch.pairs.size(), 2U);
  EXPECT_EQ(provenance->batch.pairs[0].snapshot.get(), fixture.pairs[0].snapshot.get());
  EXPECT_EQ(provenance->batch.pairs[1].snapshot->checksum(), fixture.pairs[1].snapshot->checksum());
  EXPECT_EQ(graph.factorBatchJournalStats().active_lidar_direct_factors, 0U);
  EXPECT_EQ(graph.factorBatchJournalStats().terminal_records, 1U);
}

TEST(FactorBatchGraph, BoundsRemovalToTheConfiguredRecentBatchWindow) {
  LocalGraphConfig config;
  config.maximum_removable_factor_batches = 1U;
  config.maximum_factor_batches_per_removal_transaction = 1U;
  LocalGraph graph(config);
  initializeThreeStateGraph(&graph);
  ASSERT_TRUE(graph.insertFactorBatch(directBatch(directFixture(44U), 30U)));
  ASSERT_TRUE(graph.insertFactorBatch(directBatch(directFixture(45U), 31U)));

  const SensorFactorBatchRef older = batchRef(30U);
  const SensorFactorBatchRef newest = batchRef(31U);
  ASSERT_TRUE(graph.factorBatchProvenance(older));
  EXPECT_FALSE(graph.factorBatchProvenance(older)->removable);
  ASSERT_TRUE(graph.factorBatchProvenance(newest));
  EXPECT_TRUE(graph.factorBatchProvenance(newest)->removable);
  EXPECT_EQ(graph.factorBatchJournalStats().removable_batches, 1U);

  const auto sealed_removal = graph.removeFactorBatches(
      FactorBatchRemovalRequest{{older}, FactorBatchRemovalReason::SensorFailure});
  ASSERT_FALSE(sealed_removal);
  EXPECT_EQ(sealed_removal.error().code, LocalGraphErrorCode::FactorBatchRemovalUnavailable);
  const auto newest_removal = graph.removeFactorBatches(
      FactorBatchRemovalRequest{{newest}, FactorBatchRemovalReason::SensorFailure});
  ASSERT_TRUE(newest_removal) << (newest_removal ? std::string{} : newest_removal.error().detail);
  EXPECT_EQ(newest_removal.value().solve.active_factor_batches, 1U);
}

TEST(FactorBatchGraph, SealsOnTargetFinalityAndArchivesOnSourceFinality) {
  LocalGraphConfig config;
  config.maximum_navigation_states = 3U;
  config.target_fixed_lag = core::Duration{60LL * kSecond};
  LocalGraph graph(config);
  initializeThreeStateGraph(&graph);
  const DirectBatchFixture fixture = directFixture(47U);
  ASSERT_TRUE(graph.insertFactorBatch(directBatch(fixture, 40U)));

  const auto target_finality =
      graph.appendImuKnot(stationaryKnot(13U, 2LL * kSecond, 3LL * kSecond, 4U));
  ASSERT_TRUE(target_finality) << target_finality.error().detail;
  ASSERT_EQ(target_finality.value().finalized_states.size(), 1U);
  EXPECT_EQ(target_finality.value().finalized_states.front().state, core::StateId{10U});
  EXPECT_EQ(target_finality.value().solve.factor_batches_sealed, 1U);
  EXPECT_EQ(target_finality.value().solve.lidar_direct_batch_factors_sealed, 2U);
  const auto sealed = graph.factorBatchProvenance(batchRef(40U));
  ASSERT_TRUE(sealed);
  EXPECT_EQ(sealed->status, FactorBatchJournalStatus::SealedByMarginalization);
  EXPECT_FALSE(sealed->removable);

  ASSERT_TRUE(graph.appendImuKnot(stationaryKnot(14U, 3LL * kSecond, 4LL * kSecond, 5U)));
  const auto source_finality =
      graph.appendImuKnot(stationaryKnot(15U, 4LL * kSecond, 5LL * kSecond, 6U));
  ASSERT_TRUE(source_finality) << source_finality.error().detail;
  ASSERT_EQ(source_finality.value().finalized_states.size(), 1U);
  EXPECT_EQ(source_finality.value().finalized_states.front().state, core::StateId{12U});
  EXPECT_EQ(source_finality.value().solve.factor_batches_finalized, 1U);
  EXPECT_EQ(source_finality.value().solve.lidar_direct_batch_factors_finalized, 2U);
  EXPECT_EQ(source_finality.value().solve.active_factor_batches, 0U);
  const auto archived = graph.factorBatchProvenance(batchRef(40U));
  ASSERT_TRUE(archived);
  EXPECT_EQ(archived->status, FactorBatchJournalStatus::FinalizedByMarginalization);
  EXPECT_EQ(graph.factorBatchJournalStats().terminal_records, 1U);
}

}  // namespace
}  // namespace meridian::local
