#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <tuple>
#include <variant>
#include <vector>

#include "lidar_registration_cloud_test_utils.hpp"
#include "meridian/core/blob.hpp"
#include "meridian/local/finalized_lidar_target_map.hpp"
#include "meridian/local/lidar_registration.hpp"

namespace meridian::local {
namespace {

[[nodiscard]] core::Pose3d pose(const Eigen::Vector3d& translation,
                                const Eigen::Vector3d& rotation) {
  return core::Pose3d{Sophus::SO3d::exp(rotation), translation};
}

[[nodiscard]] std::vector<Eigen::Vector3d> volumePoints() {
  std::vector<Eigen::Vector3d> points;
  points.reserve(140U);
  for (int x = -3; x <= 3; ++x) {
    for (int y = -2; y <= 2; ++y) {
      for (int z = -1; z <= 2; ++z) {
        points.emplace_back(0.71 * static_cast<double>(x) + 0.021 * static_cast<double>(y * z),
                            0.73 * static_cast<double>(y) + 0.017 * static_cast<double>(x * z),
                            0.79 * static_cast<double>(z) + 0.013 * static_cast<double>(x * y));
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

[[nodiscard]] std::shared_ptr<const LidarRegistrationCloud> cloud(
    const std::vector<Eigen::Vector3d>& points, core::MeasurementId sweep, core::FusionTime time,
    const core::Pose3d& T_odom_cloud, double exact_index_resolution_m = 1.0) {
  return test::sealedLidarRegistrationCloud(points, sweep, time, T_odom_cloud,
                                            exact_index_resolution_m);
}

[[nodiscard]] LidarRegistrationTarget targetRecord(
    core::StateId state, const std::shared_ptr<const LidarRegistrationCloud>& target,
    const std::shared_ptr<const LidarRegistrationCloud>& source) {
  return LidarRegistrationTarget{state, target->reference_time, target, target->T_odom_imu_seed,
                                 target->T_odom_imu_seed.inverse() * source->T_odom_imu_seed};
}

[[nodiscard]] LidarRegistrationConfig preciseConfig() {
  LidarRegistrationConfig config;
  config.target_voxel_resolution_m = 1.0;
  config.source_voxel_size_m = 0.10;
  config.maximum_correspondence_distance_m = 0.50;
  config.maximum_source_points = 2'000U;
  config.maximum_target_points_per_target = 2'000U;
  config.parallel_worker_count = 1U;
  config.minimum_correspondences = 20U;
  config.residual_standard_deviation_m = 0.10;
  config.minimum_huber_delta_m = 0.04;
  config.maximum_huber_delta_m = 0.40;
  config.maximum_outer_iterations = 50U;
  config.translation_convergence_m = 1.0e-6;
  config.rotation_convergence_rad = 1.0e-6;
  config.minimum_observable_rank = 6U;
  config.absolute_normalized_observable_eigenvalue = 1.0e-8;
  config.relative_normalized_observable_eigenvalue = 1.0e-6;
  config.maximum_translation_information = 100.0;
  return config;
}

[[nodiscard]] core::FactorBatchMetadata finalizedMapMetadata(
    const LidarRegistrationCloud& cloud, core::StateId state, core::FactorBatchId batch,
    core::SensorRecoveryEpoch recovery = core::SensorRecoveryEpoch{1U}) {
  core::ObservationLineage lineage = cloud.lineage;
  lineage.id = core::ObservationLineageId{20'000U + batch.value()};
  for (core::ObservationUsage& usage : lineage.usage) {
    usage.consumer = core::DerivedRecordId{batch.value()};
    const auto* source = std::get_if<core::MeasurementId>(&usage.slice.root);
    if (source != nullptr && *source == cloud.source_sweep) {
      usage.role = core::ObservationRole::PrimaryResidual;
      usage.factor_group = core::FactorGroupId{batch.value()};
    }
  }
  lineage.checksum = {};
  auto checksum = recomputeAcceptedLidarLineageChecksum(lineage);
  if (!checksum) {
    throw std::runtime_error("failed to checksum finalized-map test lineage");
  }
  lineage.checksum = checksum.value();

  core::DirectionalObservability observability;
  observability.eigenvalues.setOnes();
  observability.rank = 6U;
  observability.absolute_eigenvalue_threshold = 0.1;
  observability.relative_eigenvalue_threshold = 0.0;
  observability.supported_variables = {core::DirectionalVariable::PoseTranslation,
                                       core::DirectionalVariable::PoseRotation};
  observability.endpoints = {{core::DirectionalEndpointRole::Unary, state, cloud.reference_time}};

  const core::SensorInstanceId sensor = core::SensorInstanceId::lidar(core::LidarId{1U});
  core::FactorBatchMetadata metadata;
  metadata.header.schema_version = 1U;
  metadata.header.trace = core::TraceId{1U};
  metadata.header.producer = core::ProducerId{1U};
  metadata.header.session = core::SessionId{1U};
  metadata.header.created_at = cloud.reference_time + core::Duration{4LL};
  metadata.header.config = core::ConfigRevision{1U};
  metadata.header.direct_calibration = core::CalibrationEpoch{1U};
  metadata.batch_id = batch;
  metadata.odom_epoch = core::OdomEpoch{1U};
  metadata.sensor = sensor;
  metadata.timing.support = core::TimeRange{cloud.reference_time - core::Duration{1LL},
                                            cloud.reference_time + core::Duration{1LL}};
  metadata.timing.measurement_timestamps = {cloud.reference_time};
  metadata.timing.reference_time = cloud.reference_time;
  metadata.timing.produced_at = cloud.reference_time + core::Duration{3LL};
  metadata.health = core::SensorHealthSnapshot{sensor, core::SensorHealthState::Active, recovery,
                                               1U, cloud.reference_time + core::Duration{2LL}};
  metadata.map_eligible = true;
  metadata.directional_observability.push_back(std::move(observability));
  metadata.lineage = std::move(lineage);
  if (core::validateFactorBatchMetadata(metadata) !=
      core::FactorBatchMetadataValidationError::None) {
    throw std::runtime_error("failed to construct finalized-map test metadata");
  }
  return metadata;
}

[[nodiscard]] FinalizedLidarSweep finalizedMapSweep(
    std::shared_ptr<const LidarRegistrationCloud> target, core::StateId state,
    core::FactorBatchId batch, core::LocalGraphRevision revision, core::Pose3d final_pose,
    double pose_variance = 1.0e-4) {
  FinalizedLidarSweep sweep;
  sweep.batch = SensorFactorBatchRef{core::SensorInstanceId::lidar(core::LidarId{1U}), batch};
  sweep.accepted_batch_metadata = finalizedMapMetadata(*target, state, batch);
  sweep.admission_revision = revision;
  sweep.admission_kind = MapAdmissionBatchKind::Regular;
  sweep.finalized_state.state = state;
  sweep.finalized_state.exact_time = target->reference_time;
  sweep.finalized_state.odom_epoch = core::OdomEpoch{1U};
  sweep.finalized_state.final_revision = revision;
  sweep.finalized_state.final_estimate.T_odom_imu = std::move(final_pose);
  sweep.finalized_state.pose_covariance.matrix = core::Matrix6d::Identity() * pose_variance;
  sweep.calibration = core::CalibrationEpoch{1U};
  sweep.cloud = std::move(target);
  return sweep;
}

[[nodiscard]] FinalizedLidarTargetMap finalizedMap() {
  FinalizedLidarTargetMapConfig config;
  config.odom_epoch = core::OdomEpoch{1U};
  config.sensor = core::SensorInstanceId::lidar(core::LidarId{1U});
  config.query_voxel_size_m = 1.0;
  config.insertion_voxel_size_m = 0.05;
  config.maximum_points_per_query_voxel = 1'000U;
  config.minimum_point_separation_m = 0.01;
  config.maximum_supported_query_distance_m = 0.5;
  config.maximum_radius_m = 1'000.0;
  config.hard_point_capacity = 100'000U;
  auto created = FinalizedLidarTargetMap::create(config);
  if (!created) {
    throw std::runtime_error("failed to create finalized-map registration fixture: " +
                             created.error().detail);
  }
  return std::move(created).value();
}

[[nodiscard]] std::shared_ptr<const FinalizedLidarTargetOwner> insertFinalizedMapCloud(
    FinalizedLidarTargetMap& map,
    const std::shared_ptr<const LidarRegistrationCloud>& target, core::StateId state,
    core::FactorBatchId batch, core::LocalGraphRevision revision,
    const core::Pose3d& final_pose, double pose_variance = 1.0e-4) {
  auto inserted = map.insertFinalizedSweep(
      finalizedMapSweep(target, state, batch, revision, final_pose, pose_variance));
  if (!inserted || !inserted.value().owner) {
    throw std::runtime_error("failed to insert finalized-map registration fixture: " +
                             (inserted ? std::string{"missing owner"}
                                       : inserted.error().detail));
  }
  return inserted.value().owner;
}

void expectPoseNear(const core::Pose3d& actual, const core::Pose3d& expected,
                    double translation_tolerance, double rotation_tolerance) {
  const core::Pose3d error = expected.inverse() * actual;
  EXPECT_LT(error.translation().norm(), translation_tolerance);
  EXPECT_LT(error.so3().log().norm(), rotation_tolerance);
}

void expectBitIdenticalDouble(double expected, double actual) {
  EXPECT_EQ(std::bit_cast<std::uint64_t>(actual), std::bit_cast<std::uint64_t>(expected));
}

template <typename Expected, typename Actual>
void expectBitIdenticalMatrix(const Eigen::MatrixBase<Expected>& expected,
                              const Eigen::MatrixBase<Actual>& actual) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  for (Eigen::Index row = 0; row < expected.rows(); ++row) {
    for (Eigen::Index column = 0; column < expected.cols(); ++column) {
      expectBitIdenticalDouble(expected(row, column), actual(row, column));
    }
  }
}

void expectBitIdenticalWork(const LidarRegistrationWorkCounters& expected,
                            const LidarRegistrationWorkCounters& actual) {
  EXPECT_EQ(actual.source_points_considered, expected.source_points_considered);
  EXPECT_EQ(actual.source_points_selected, expected.source_points_selected);
  EXPECT_EQ(actual.source_points_omitted_by_capacity, expected.source_points_omitted_by_capacity);
  EXPECT_EQ(actual.invalid_source_points, expected.invalid_source_points);
  EXPECT_EQ(actual.invalid_target_points, expected.invalid_target_points);
  EXPECT_EQ(actual.target_index_voxels, expected.target_index_voxels);
  EXPECT_EQ(actual.composite_index_builds, expected.composite_index_builds);
  EXPECT_EQ(actual.composite_index_input_owners, expected.composite_index_input_owners);
  EXPECT_EQ(actual.composite_index_input_points, expected.composite_index_input_points);
  EXPECT_EQ(actual.composite_index_retained_owners, expected.composite_index_retained_owners);
  EXPECT_EQ(actual.composite_index_retained_points, expected.composite_index_retained_points);
  EXPECT_EQ(actual.composite_index_occupied_voxels, expected.composite_index_occupied_voxels);
  EXPECT_EQ(actual.composite_index_per_voxel_discarded_points,
            expected.composite_index_per_voxel_discarded_points);
  EXPECT_EQ(actual.composite_index_total_discarded_points,
            expected.composite_index_total_discarded_points);
  EXPECT_EQ(actual.association_builds, expected.association_builds);
  EXPECT_EQ(actual.candidate_voxel_lookups, expected.candidate_voxel_lookups);
  EXPECT_EQ(actual.candidate_occupied_voxels, expected.candidate_occupied_voxels);
  EXPECT_EQ(actual.candidate_points_examined, expected.candidate_points_examined);
  EXPECT_EQ(actual.finalized_map_candidate_voxel_lookups,
            expected.finalized_map_candidate_voxel_lookups);
  EXPECT_EQ(actual.finalized_map_candidate_occupied_voxels,
            expected.finalized_map_candidate_occupied_voxels);
  EXPECT_EQ(actual.finalized_map_candidate_points_examined,
            expected.finalized_map_candidate_points_examined);
  EXPECT_EQ(actual.finalized_map_stale_fallbacks, expected.finalized_map_stale_fallbacks);
  EXPECT_EQ(actual.frozen_objective_evaluations, expected.frozen_objective_evaluations);
  EXPECT_EQ(actual.normal_equation_evaluations, expected.normal_equation_evaluations);
  EXPECT_EQ(actual.outer_iterations, expected.outer_iterations);
  EXPECT_EQ(actual.gauss_newton_trials, expected.gauss_newton_trials);
  EXPECT_EQ(actual.lm_damping_trials, expected.lm_damping_trials);
  EXPECT_EQ(actual.rejected_frozen_cost_trials, expected.rejected_frozen_cost_trials);
  EXPECT_EQ(actual.accepted_steps, expected.accepted_steps);
}

void expectBitIdenticalRow(const FrozenPointCorrespondence& expected,
                           const FrozenPointCorrespondence& actual) {
  EXPECT_EQ(actual.source_point_storage_index, expected.source_point_storage_index);
  EXPECT_EQ(actual.source_index, expected.source_index);
  expectBitIdenticalMatrix(expected.source_point, actual.source_point);
  EXPECT_EQ(actual.target_point_storage_index, expected.target_point_storage_index);
  EXPECT_EQ(actual.target_source_index, expected.target_source_index);
  expectBitIdenticalMatrix(expected.target_point, actual.target_point);
  expectBitIdenticalDouble(expected.association_distance_squared_m2,
                           actual.association_distance_squared_m2);
  expectBitIdenticalDouble(expected.association_huber_weight, actual.association_huber_weight);
}

void expectBitIdenticalSnapshot(const LidarFactorSnapshot& expected,
                                const LidarFactorSnapshot& actual) {
  EXPECT_EQ(actual.targetState(), expected.targetState());
  EXPECT_EQ(actual.targetTime(), expected.targetTime());
  EXPECT_EQ(actual.targetSweep(), expected.targetSweep());
  EXPECT_EQ(actual.sourceState(), expected.sourceState());
  EXPECT_EQ(actual.sourceTime(), expected.sourceTime());
  EXPECT_EQ(actual.sourceSweep(), expected.sourceSweep());
  EXPECT_EQ(actual.targetCloudChecksum(), expected.targetCloudChecksum());
  EXPECT_EQ(actual.sourceCloudChecksum(), expected.sourceCloudChecksum());
  EXPECT_EQ(actual.sourcePointCount(), expected.sourcePointCount());
  expectBitIdenticalMatrix(expected.associationPose().matrix(), actual.associationPose().matrix());
  EXPECT_EQ(actual.candidateVoxelLookups(), expected.candidateVoxelLookups());
  EXPECT_EQ(actual.candidatePointsExamined(), expected.candidatePointsExamined());
  EXPECT_EQ(actual.sourceRowsExcludedByOwnership(), expected.sourceRowsExcludedByOwnership());
  expectBitIdenticalDouble(expected.residualStandardDeviationM(),
                           actual.residualStandardDeviationM());
  expectBitIdenticalDouble(expected.huberDeltaM(), actual.huberDeltaM());
  expectBitIdenticalDouble(expected.characteristicLengthM(), actual.characteristicLengthM());
  expectBitIdenticalDouble(expected.geometricInformationScale(),
                           actual.geometricInformationScale());
  EXPECT_EQ(actual.checksum(), expected.checksum());
  ASSERT_EQ(actual.rows().size(), expected.rows().size());
  for (std::size_t index = 0U; index < expected.rows().size(); ++index) {
    expectBitIdenticalRow(expected.rows()[index], actual.rows()[index]);
  }
}

void expectBitIdenticalMapSnapshot(const FinalizedMapLidarFactorSnapshot& expected,
                                   const FinalizedMapLidarFactorSnapshot& actual) {
  EXPECT_EQ(actual.sourceState(), expected.sourceState());
  EXPECT_EQ(actual.sourceTime(), expected.sourceTime());
  EXPECT_EQ(actual.sourceSweep(), expected.sourceSweep());
  EXPECT_EQ(actual.sourceCloudChecksum(), expected.sourceCloudChecksum());
  EXPECT_EQ(actual.sourcePointCount(), expected.sourcePointCount());
  EXPECT_EQ(actual.mapOdomEpoch(), expected.mapOdomEpoch());
  EXPECT_EQ(actual.mapSensor(), expected.mapSensor());
  EXPECT_EQ(actual.mapVersion(), expected.mapVersion());
  EXPECT_EQ(actual.mapChecksum(), expected.mapChecksum());
  expectBitIdenticalMatrix(expected.associationPose().matrix(), actual.associationPose().matrix());
  EXPECT_EQ(actual.uniqueOwnerCount(), expected.uniqueOwnerCount());
  EXPECT_EQ(actual.candidateVoxelLookups(), expected.candidateVoxelLookups());
  EXPECT_EQ(actual.candidateOccupiedVoxels(), expected.candidateOccupiedVoxels());
  EXPECT_EQ(actual.candidatePointsExamined(), expected.candidatePointsExamined());
  EXPECT_EQ(actual.sourceRowsExcludedByOwnership(), expected.sourceRowsExcludedByOwnership());
  expectBitIdenticalDouble(expected.residualStandardDeviationM(),
                           actual.residualStandardDeviationM());
  expectBitIdenticalDouble(expected.huberDeltaM(), actual.huberDeltaM());
  expectBitIdenticalDouble(expected.characteristicLengthM(), actual.characteristicLengthM());
  expectBitIdenticalDouble(expected.geometricInformationScale(),
                           actual.geometricInformationScale());
  expectBitIdenticalDouble(expected.ownerPoseCovarianceInflation(),
                           actual.ownerPoseCovarianceInflation());
  EXPECT_EQ(actual.checksum(), expected.checksum());
  ASSERT_EQ(actual.owners().size(), expected.owners().size());
  for (std::size_t index = 0U; index < expected.owners().size(); ++index) {
    ASSERT_TRUE(expected.owners()[index]);
    ASSERT_TRUE(actual.owners()[index]);
    EXPECT_EQ(actual.owners()[index]->batch, expected.owners()[index]->batch);
    EXPECT_EQ(actual.owners()[index]->finalized_state.state,
              expected.owners()[index]->finalized_state.state);
    EXPECT_EQ(actual.owners()[index]->sweep, expected.owners()[index]->sweep);
  }
  ASSERT_EQ(actual.rows().size(), expected.rows().size());
  for (std::size_t index = 0U; index < expected.rows().size(); ++index) {
    const auto& expected_row = expected.rows()[index];
    const auto& actual_row = actual.rows()[index];
    EXPECT_EQ(actual_row.source_point_storage_index, expected_row.source_point_storage_index);
    EXPECT_EQ(actual_row.source_index, expected_row.source_index);
    expectBitIdenticalMatrix(expected_row.source_point, actual_row.source_point);
    EXPECT_EQ(actual_row.target_source_index, expected_row.target_source_index);
    expectBitIdenticalMatrix(expected_row.target_point_odom, actual_row.target_point_odom);
    EXPECT_EQ(actual_row.owner_index, expected_row.owner_index);
    expectBitIdenticalDouble(expected_row.association_distance_squared_m2,
                             actual_row.association_distance_squared_m2);
    expectBitIdenticalDouble(expected_row.association_huber_weight,
                             actual_row.association_huber_weight);
  }
}

void expectBitIdenticalResult(const LidarRegistrationResult& expected,
                              const LidarRegistrationResult& actual) {
  EXPECT_EQ(actual.source_state, expected.source_state);
  EXPECT_EQ(actual.source_time, expected.source_time);
  expectBitIdenticalMatrix(expected.T_odom_source.matrix(), actual.T_odom_source.matrix());
  expectBitIdenticalMatrix(expected.source_right_correction.matrix(),
                           actual.source_right_correction.matrix());
  expectBitIdenticalMatrix(expected.raw_physical_hessian, actual.raw_physical_hessian);
  expectBitIdenticalMatrix(expected.projected_physical_information,
                           actual.projected_physical_information);
  expectBitIdenticalMatrix(expected.supported_physical_covariance,
                           actual.supported_physical_covariance);
  expectBitIdenticalMatrix(expected.normalized_observability_projector,
                           actual.normalized_observability_projector);
  expectBitIdenticalDouble(expected.initial_robust_cost, actual.initial_robust_cost);
  expectBitIdenticalDouble(expected.final_robust_cost, actual.final_robust_cost);
  EXPECT_EQ(actual.termination, expected.termination);
  EXPECT_EQ(actual.diagnostics.target_count, expected.diagnostics.target_count);
  EXPECT_EQ(actual.diagnostics.live_target_count, expected.diagnostics.live_target_count);
  EXPECT_EQ(actual.diagnostics.finalized_map_target_count,
            expected.diagnostics.finalized_map_target_count);
  EXPECT_EQ(actual.diagnostics.correspondences, expected.diagnostics.correspondences);
  EXPECT_EQ(actual.diagnostics.live_correspondences, expected.diagnostics.live_correspondences);
  EXPECT_EQ(actual.diagnostics.finalized_map_correspondences,
            expected.diagnostics.finalized_map_correspondences);
  expectBitIdenticalDouble(expected.diagnostics.overlap_fraction,
                           actual.diagnostics.overlap_fraction);
  expectBitIdenticalDouble(expected.diagnostics.effective_correspondences,
                           actual.diagnostics.effective_correspondences);
  expectBitIdenticalDouble(expected.diagnostics.maximum_squared_residual_m2,
                           actual.diagnostics.maximum_squared_residual_m2);
  expectBitIdenticalDouble(expected.diagnostics.huber_delta_m, actual.diagnostics.huber_delta_m);
  expectBitIdenticalDouble(expected.diagnostics.characteristic_length_m,
                           actual.diagnostics.characteristic_length_m);
  expectBitIdenticalDouble(expected.diagnostics.normalized_observable_eigenvalue_threshold,
                           actual.diagnostics.normalized_observable_eigenvalue_threshold);
  EXPECT_EQ(actual.diagnostics.observable_rank, expected.diagnostics.observable_rank);
  expectBitIdenticalDouble(expected.diagnostics.maximum_supported_normalized_information,
                           actual.diagnostics.maximum_supported_normalized_information);
  expectBitIdenticalDouble(expected.diagnostics.normalized_information_cap,
                           actual.diagnostics.normalized_information_cap);
  expectBitIdenticalDouble(expected.diagnostics.geometric_information_scale,
                           actual.diagnostics.geometric_information_scale);
  expectBitIdenticalMatrix(expected.diagnostics.raw_normalized_hessian_eigenvalues,
                           actual.diagnostics.raw_normalized_hessian_eigenvalues);
  EXPECT_EQ(actual.diagnostics.normalized_directional_information.rank,
            expected.diagnostics.normalized_directional_information.rank);
  expectBitIdenticalMatrix(expected.diagnostics.normalized_directional_information.basis,
                           actual.diagnostics.normalized_directional_information.basis);
  expectBitIdenticalMatrix(expected.diagnostics.normalized_directional_information.eigenvalues,
                           actual.diagnostics.normalized_directional_information.eigenvalues);
  EXPECT_EQ(actual.diagnostics.physical_information.rank,
            expected.diagnostics.physical_information.rank);
  expectBitIdenticalMatrix(expected.diagnostics.physical_information.basis,
                           actual.diagnostics.physical_information.basis);
  expectBitIdenticalMatrix(expected.diagnostics.physical_information.eigenvalues,
                           actual.diagnostics.physical_information.eigenvalues);
  expectBitIdenticalWork(expected.work, actual.work);
  ASSERT_EQ(actual.target_snapshots.size(), expected.target_snapshots.size());
  for (std::size_t index = 0U; index < expected.target_snapshots.size(); ++index) {
    ASSERT_TRUE(expected.target_snapshots[index]);
    ASSERT_TRUE(actual.target_snapshots[index]);
    expectBitIdenticalSnapshot(*expected.target_snapshots[index], *actual.target_snapshots[index]);
  }
  EXPECT_EQ(static_cast<bool>(actual.finalized_map_snapshot),
            static_cast<bool>(expected.finalized_map_snapshot));
  if (expected.finalized_map_snapshot && actual.finalized_map_snapshot) {
    expectBitIdenticalMapSnapshot(*expected.finalized_map_snapshot, *actual.finalized_map_snapshot);
  }
}

TEST(DirectPointIcp, RecoversRigidTransformAndPublishesFiniteDirectionalInformation) {
  const std::vector<Eigen::Vector3d> source_points = volumePoints();
  const core::Pose3d truth =
      pose(Eigen::Vector3d{0.08, -0.05, 0.04}, Eigen::Vector3d{0.018, -0.012, 0.015});
  const core::Pose3d seed =
      pose(Eigen::Vector3d{0.03, -0.01, 0.01}, Eigen::Vector3d{0.006, -0.004, 0.005});
  const auto source =
      cloud(source_points, core::MeasurementId{10U}, core::FusionTime{2'000U}, seed);
  const auto target = cloud(transformed(source_points, truth), core::MeasurementId{11U},
                            core::FusionTime{1'000U}, core::Pose3d{});
  const LidarRegistrationTarget record = targetRecord(core::StateId{1U}, target, source);
  const LidarRegistrationConfig config = preciseConfig();

  const auto result = registerLidarScan(core::StateId{2U}, source, std::span{&record, 1U}, config);
  ASSERT_TRUE(result) << result.error().detail;
  expectPoseNear(result.value().T_odom_source, truth, 5.0e-4, 2.0e-4);
  EXPECT_LT(result.value().final_robust_cost, result.value().initial_robust_cost);
  EXPECT_EQ(result.value().diagnostics.observable_rank, 6U);
  EXPECT_EQ(result.value().diagnostics.physical_information.rank, 6U);
  EXPECT_EQ(result.value().diagnostics.normalized_directional_information.rank, 6U);
  EXPECT_EQ(result.value().diagnostics.correspondences, source_points.size());
  EXPECT_DOUBLE_EQ(result.value().diagnostics.overlap_fraction, 1.0);
  EXPECT_TRUE(result.value().raw_physical_hessian.allFinite());
  EXPECT_TRUE(result.value().projected_physical_information.allFinite());
  EXPECT_TRUE(result.value().supported_physical_covariance.allFinite());
  EXPECT_TRUE(result.value().normalized_observability_projector.allFinite());
  EXPECT_TRUE(result.value().raw_physical_hessian.isApprox(
      result.value().raw_physical_hessian.transpose(), 1.0e-12));
  EXPECT_LE(result.value().diagnostics.geometric_information_scale *
                result.value().diagnostics.maximum_supported_normalized_information,
            result.value().diagnostics.normalized_information_cap * (1.0 + 1.0e-12));
  ASSERT_EQ(result.value().target_snapshots.size(), 1U);
  EXPECT_TRUE(core::contentHashPresent(result.value().target_snapshots.front()->checksum()));
  EXPECT_DOUBLE_EQ(result.value().target_snapshots.front()->residualStandardDeviationM(),
                   config.residual_standard_deviation_m);
  EXPECT_GE(result.value().target_snapshots.front()->huberDeltaM(), config.minimum_huber_delta_m);
  EXPECT_LE(result.value().target_snapshots.front()->huberDeltaM(), config.maximum_huber_delta_m);
  EXPECT_DOUBLE_EQ(result.value().target_snapshots.front()->characteristicLengthM(),
                   result.value().diagnostics.characteristic_length_m);
  EXPECT_DOUBLE_EQ(result.value().target_snapshots.front()->geometricInformationScale(),
                   result.value().diagnostics.geometric_information_scale);

  const auto aggregate = summarizeLidarFactorSnapshots(result.value().target_snapshots);
  ASSERT_TRUE(aggregate) << aggregate.error().detail;
  EXPECT_EQ(aggregate.value().correspondences, result.value().diagnostics.correspondences);
  EXPECT_NEAR(aggregate.value().final_robust_cost, result.value().final_robust_cost,
              1.0e-10 * std::max(1.0, result.value().final_robust_cost));

  const auto information =
      lidarFactorInformation(result.value().target_snapshots.front(), config, 0.4);
  ASSERT_TRUE(information) << information.error().detail;
  EXPECT_EQ(information.value().rank, 6U);
  EXPECT_TRUE(information.value().basis.allFinite());
  EXPECT_TRUE(information.value().eigenvalues.allFinite());
  EXPECT_GT(information.value().eigenvalues.head<6>().minCoeff(), 0.0);
  core::Matrix6d factor_information = core::Matrix6d::Zero();
  for (std::size_t mode = 0U; mode < information.value().rank; ++mode) {
    const Eigen::Index index = static_cast<Eigen::Index>(mode);
    factor_information.noalias() += information.value().eigenvalues(index) *
                                    information.value().basis.col(index) *
                                    information.value().basis.col(index).transpose();
  }
  EXPECT_TRUE(factor_information.isApprox(
      0.4 * result.value().projected_physical_information,
      1.0e-8 * std::max(1.0, factor_information.cwiseAbs().maxCoeff())));
}

TEST(DirectPointIcp, SourceSelectionUsesOneCanonicalRawPointPerCoarseVoxelAndStableCap) {
  const std::vector<Eigen::Vector3d> points{
      {0.10, 0.10, 0.10}, {0.80, 0.20, 0.20}, {1.10, 0.10, 0.10}, {1.80, 0.20, 0.20},
      {2.10, 0.10, 0.10}, {2.80, 0.20, 0.20}, {3.10, 0.10, 0.10}, {3.80, 0.20, 0.20}};
  const auto source =
      cloud(points, core::MeasurementId{20U}, core::FusionTime{2'000U}, core::Pose3d{});
  LidarRegistrationConfig config;
  config.source_voxel_size_m = 1.0;
  config.maximum_source_points = 3U;
  config.minimum_correspondences = 1U;

  const auto selection = selectLidarRegistrationSourcePoints(source, config);
  ASSERT_TRUE(selection) << selection.error().detail;
  EXPECT_EQ(selection.value().points_considered, points.size());
  EXPECT_EQ(selection.value().points_omitted, 5U);
  ASSERT_EQ(selection.value().point_storage_indices.size(), 3U);
  const std::array<std::uint32_t, 3U> expected_source_indices{0U, 2U, 6U};
  for (std::size_t index = 0U; index < expected_source_indices.size(); ++index) {
    const std::size_t storage = selection.value().point_storage_indices[index];
    ASSERT_LT(storage, source->points.size());
    EXPECT_EQ(source->points[storage].source_index, expected_source_indices[index]);
  }
}

TEST(DirectPointIcp, AdaptiveHuberKeepsVegetationLikeOutliersButDownweightsThem) {
  const std::vector<Eigen::Vector3d> source_points = volumePoints();
  std::vector<Eigen::Vector3d> target_points = source_points;
  for (std::size_t index = 0U; index < 10U; ++index) {
    target_points[index].y() += 0.30;
  }
  for (std::size_t index = 10U; index < 20U; ++index) {
    target_points[index].y() -= 0.30;
  }
  const auto source =
      cloud(source_points, core::MeasurementId{30U}, core::FusionTime{2'000U}, core::Pose3d{});
  const auto target =
      cloud(target_points, core::MeasurementId{31U}, core::FusionTime{1'000U}, core::Pose3d{});
  const LidarRegistrationTarget record = targetRecord(core::StateId{3U}, target, source);
  LidarRegistrationConfig config = preciseConfig();
  config.minimum_huber_delta_m = 0.04;
  config.maximum_huber_delta_m = 0.40;

  const auto result = registerLidarScan(core::StateId{4U}, source, std::span{&record, 1U}, config);
  ASSERT_TRUE(result) << result.error().detail;
  expectPoseNear(result.value().T_odom_source, core::Pose3d{}, 4.0e-2, 2.0e-2);
  ASSERT_EQ(result.value().target_snapshots.size(), 1U);
  const auto rows = result.value().target_snapshots.front()->rows();
  std::size_t downweighted = 0U;
  std::size_t full_weight = 0U;
  for (const FrozenPointCorrespondence& row : rows) {
    EXPECT_GT(row.association_huber_weight, 0.0);
    EXPECT_LE(row.association_huber_weight, 1.0);
    downweighted += row.association_huber_weight < 0.50 ? 1U : 0U;
    full_weight += row.association_huber_weight == 1.0 ? 1U : 0U;
  }
  EXPECT_GE(downweighted, 15U);
  EXPECT_GE(full_weight, 100U);
  EXPECT_LT(result.value().diagnostics.effective_correspondences,
            static_cast<double>(result.value().diagnostics.correspondences));
  EXPECT_GT(result.value().diagnostics.maximum_squared_residual_m2, 0.04);
  EXPECT_GE(result.value().diagnostics.huber_delta_m, config.minimum_huber_delta_m);
  EXPECT_LE(result.value().diagnostics.huber_delta_m, config.maximum_huber_delta_m);
}

TEST(DirectPointIcp, MultiTargetOwnershipPreservesPoseSeparationAndIsOrderWorkerDeterministic) {
  std::vector<Eigen::Vector3d> target_frame_points;
  target_frame_points.reserve(60U);
  for (int x = -2; x <= 2; ++x) {
    for (int y = -2; y <= 1; ++y) {
      for (int z = -1; z <= 1; ++z) {
        target_frame_points.emplace_back(
            0.51 * static_cast<double>(x) + 0.013 * static_cast<double>(y * z),
            0.53 * static_cast<double>(y) + 0.011 * static_cast<double>(x * z),
            0.57 * static_cast<double>(z) + 0.017 * static_cast<double>(x * y));
      }
    }
  }
  std::vector<Eigen::Vector3d> source_points = target_frame_points;
  for (const Eigen::Vector3d& point : target_frame_points) {
    source_points.push_back(point + Eigen::Vector3d{10.0, 0.0, 0.0});
  }
  const auto source =
      cloud(source_points, core::MeasurementId{40U}, core::FusionTime{3'000U}, core::Pose3d{});
  const auto target_a = cloud(target_frame_points, core::MeasurementId{41U},
                              core::FusionTime{1'000U}, core::Pose3d{});
  const core::Pose3d T_odom_target_b{Sophus::SO3d{}, Eigen::Vector3d{10.0, 0.0, 0.0}};
  const auto target_b = cloud(target_frame_points, core::MeasurementId{42U},
                              core::FusionTime{2'000U}, T_odom_target_b);
  const LidarRegistrationTarget record_a = targetRecord(core::StateId{5U}, target_a, source);
  const LidarRegistrationTarget record_b = targetRecord(core::StateId{6U}, target_b, source);

  LidarRegistrationConfig single_config = preciseConfig();
  single_config.source_voxel_size_m = 0.08;
  single_config.maximum_correspondence_distance_m = 0.30;
  single_config.maximum_huber_delta_m = 0.25;
  single_config.parallel_worker_count = 1U;
  const std::array canonical_targets{record_a, record_b};
  const auto single =
      registerLidarScan(core::StateId{7U}, source, canonical_targets, single_config);
  ASSERT_TRUE(single) << single.error().detail;

  LidarRegistrationConfig parallel_config = single_config;
  parallel_config.parallel_worker_count = 4U;
  const std::array reversed_targets{record_b, record_a};
  const auto reordered_parallel =
      registerLidarScan(core::StateId{7U}, source, reversed_targets, parallel_config);
  ASSERT_TRUE(reordered_parallel) << reordered_parallel.error().detail;
  expectBitIdenticalResult(single.value(), reordered_parallel.value());

  ASSERT_EQ(single.value().target_snapshots.size(), 2U);
  EXPECT_EQ(single.value().target_snapshots[0]->targetState(), core::StateId{5U});
  EXPECT_EQ(single.value().target_snapshots[1]->targetState(), core::StateId{6U});
  expectPoseNear(single.value().target_snapshots[0]->associationPose(), core::Pose3d{}, 1.0e-12,
                 1.0e-12);
  expectPoseNear(single.value().target_snapshots[1]->associationPose(), T_odom_target_b.inverse(),
                 1.0e-12, 1.0e-12);

  std::set<std::uint32_t> owned_source_indices;
  std::size_t row_count = 0U;
  for (const auto& snapshot : single.value().target_snapshots) {
    ASSERT_TRUE(snapshot);
    EXPECT_GE(snapshot->rows().size(), single_config.minimum_correspondences);
    for (const FrozenPointCorrespondence& row : snapshot->rows()) {
      EXPECT_TRUE(owned_source_indices.insert(row.source_index).second);
      ++row_count;
    }
  }
  EXPECT_EQ(row_count, source_points.size());
  EXPECT_EQ(single.value().diagnostics.correspondences, source_points.size());
  EXPECT_DOUBLE_EQ(single.value().diagnostics.overlap_fraction, 1.0);

  const auto aggregate = summarizeLidarFactorSnapshots(single.value().target_snapshots);
  ASSERT_TRUE(aggregate) << aggregate.error().detail;
  EXPECT_EQ(aggregate.value().target_count, 2U);
  EXPECT_EQ(aggregate.value().correspondences, source_points.size());
}

TEST(DirectPointIcp, FinalizedMapOnlyRecoversKnownTransformAndSealsOwnerUncertainty) {
  const std::vector<Eigen::Vector3d> source_points = volumePoints();
  const core::Pose3d truth =
      pose(Eigen::Vector3d{0.07, -0.04, 0.03}, Eigen::Vector3d{0.014, -0.010, 0.012});
  const core::Pose3d seed =
      pose(Eigen::Vector3d{0.02, -0.01, 0.01}, Eigen::Vector3d{0.004, -0.003, 0.004});
  const auto target =
      cloud(source_points, core::MeasurementId{60U}, core::FusionTime{1'000U}, core::Pose3d{});
  auto map = finalizedMap();
  static_cast<void>(insertFinalizedMapCloud(
      map, target, core::StateId{1U}, core::FactorBatchId{60U},
      core::LocalGraphRevision{1U}, truth));
  const auto map_view = map.readView();
  const auto source =
      cloud(source_points, core::MeasurementId{61U}, core::FusionTime{2'000U}, seed);
  LidarRegistrationConfig config = preciseConfig();
  config.maximum_targets = 1U;

  const auto result = registerLidarScan(core::StateId{2U}, source,
                                        std::span<const LidarRegistrationTarget>{}, map_view,
                                        core::LocalGraphRevision{2U}, config);
  ASSERT_TRUE(result) << result.error().detail;
  expectPoseNear(result.value().T_odom_source, truth, 6.0e-4, 3.0e-4);
  EXPECT_TRUE(result.value().target_snapshots.empty());
  ASSERT_TRUE(result.value().finalized_map_snapshot);
  EXPECT_EQ(result.value().diagnostics.target_count, 1U);
  EXPECT_EQ(result.value().diagnostics.live_target_count, 0U);
  EXPECT_EQ(result.value().diagnostics.finalized_map_target_count, 1U);
  EXPECT_EQ(result.value().diagnostics.finalized_map_correspondences, source_points.size());
  EXPECT_EQ(result.value().finalized_map_snapshot->uniqueOwnerCount(), 1U);
  EXPECT_GT(result.value().finalized_map_snapshot->ownerPoseCovarianceInflation(), 1.0);
  EXPECT_GT(result.value().work.finalized_map_candidate_voxel_lookups, 0U);
  EXPECT_GT(result.value().work.finalized_map_candidate_points_examined, 0U);

  const auto aggregate = summarizeLidarFactorSnapshots(result.value().target_snapshots,
                                                       result.value().finalized_map_snapshot);
  ASSERT_TRUE(aggregate) << aggregate.error().detail;
  EXPECT_EQ(aggregate.value().target_count, 1U);
  EXPECT_EQ(aggregate.value().correspondences, result.value().diagnostics.correspondences);
  EXPECT_NEAR(aggregate.value().final_robust_cost, result.value().final_robust_cost,
              1.0e-10 * std::max(1.0, result.value().final_robust_cost));
  const auto inflation =
      lidarFinalizedMapOwnerPoseCovarianceInflation(result.value().finalized_map_snapshot);
  ASSERT_TRUE(inflation) << inflation.error().detail;
  EXPECT_DOUBLE_EQ(inflation.value(),
                   result.value().finalized_map_snapshot->ownerPoseCovarianceInflation());
  const auto information =
      lidarFinalizedMapFactorInformation(result.value().finalized_map_snapshot, config, 0.5);
  ASSERT_TRUE(information) << information.error().detail;
  EXPECT_EQ(information.value().rank, 6U);
}

TEST(DirectPointIcp, PersistentMapOnlySnapshotPreservesManyPoseOwners) {
  const std::vector<Eigen::Vector3d> local_points = volumePoints();
  std::vector<Eigen::Vector3d> source_points = local_points;
  for (const Eigen::Vector3d& point : local_points) {
    source_points.push_back(point + Eigen::Vector3d{10.0, 0.0, 0.0});
  }
  const auto first =
      cloud(local_points, core::MeasurementId{62U}, core::FusionTime{1'000U}, core::Pose3d{});
  const auto second =
      cloud(local_points, core::MeasurementId{63U}, core::FusionTime{1'500U}, core::Pose3d{});
  const auto source =
      cloud(source_points, core::MeasurementId{64U}, core::FusionTime{2'000U}, core::Pose3d{});
  auto map = finalizedMap();
  static_cast<void>(insertFinalizedMapCloud(
      map, first, core::StateId{1U}, core::FactorBatchId{62U}, core::LocalGraphRevision{1U},
      core::Pose3d{}));
  static_cast<void>(insertFinalizedMapCloud(
      map, second, core::StateId{2U}, core::FactorBatchId{63U}, core::LocalGraphRevision{2U},
      pose(Eigen::Vector3d{10.0, 0.0, 0.0}, Eigen::Vector3d::Zero())));
  const auto map_view = map.readView();
  LidarRegistrationConfig config = preciseConfig();
  config.maximum_targets = 1U;

  const auto result = registerLidarScan(core::StateId{3U}, source,
                                        std::span<const LidarRegistrationTarget>{}, map_view,
                                        core::LocalGraphRevision{3U}, config);
  ASSERT_TRUE(result) << result.error().detail;
  EXPECT_TRUE(result.value().target_snapshots.empty());
  ASSERT_TRUE(result.value().finalized_map_snapshot);
  EXPECT_EQ(result.value().finalized_map_snapshot->uniqueOwnerCount(), 2U);
  EXPECT_EQ(result.value().finalized_map_snapshot->owners()[0]->finalized_state.state,
            core::StateId{1U});
  EXPECT_EQ(result.value().finalized_map_snapshot->owners()[1]->finalized_state.state,
            core::StateId{2U});
  EXPECT_EQ(result.value().diagnostics.finalized_map_correspondences, source_points.size());
  EXPECT_EQ(result.value().work.composite_index_builds, 0U);
  EXPECT_GT(result.value().work.finalized_map_candidate_points_examined, 0U);
}

TEST(DirectPointIcp, EmptyPersistentMapViewIsRejectedBeforeAssociation) {
  const std::vector<Eigen::Vector3d> points = volumePoints();
  const auto source =
      cloud(points, core::MeasurementId{65U}, core::FusionTime{2'000U}, core::Pose3d{});
  auto map = finalizedMap();
  const auto empty_view = map.readView();

  const auto rejected = registerLidarScan(
      core::StateId{2U}, source, std::span<const LidarRegistrationTarget>{}, empty_view,
      core::LocalGraphRevision{2U}, preciseConfig());
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LidarRegistrationErrorCode::InvalidTarget);
  EXPECT_EQ(rejected.error().work.source_points_considered, 0U);
}

TEST(DirectPointIcp, ExactCrossIndexDistanceTieBelongsToLivePoseOwner) {
  const std::vector<Eigen::Vector3d> points = volumePoints();
  const core::Pose3d global_pose =
      pose(Eigen::Vector3d{37.0, -9.0, 1.0}, Eigen::Vector3d{0.03, -0.02, 0.12});
  const auto live_target =
      cloud(points, core::MeasurementId{66U}, core::FusionTime{1'000U}, global_pose);
  const auto map_target =
      cloud(points, core::MeasurementId{67U}, core::FusionTime{1'500U}, global_pose);
  const auto source =
      cloud(points, core::MeasurementId{68U}, core::FusionTime{2'000U}, global_pose);
  auto map = finalizedMap();
  static_cast<void>(insertFinalizedMapCloud(
      map, map_target, core::StateId{2U}, core::FactorBatchId{67U},
      core::LocalGraphRevision{2U}, global_pose));
  const auto map_view = map.readView();
  const LidarRegistrationTarget live =
      targetRecord(core::StateId{1U}, live_target, source);
  LidarRegistrationConfig config = preciseConfig();
  config.maximum_targets = 2U;

  const auto result = registerLidarScan(core::StateId{3U}, source, std::span{&live, 1U},
                                        map_view, core::LocalGraphRevision{3U}, config);
  ASSERT_TRUE(result) << result.error().detail;
  ASSERT_EQ(result.value().target_snapshots.size(), 1U);
  EXPECT_FALSE(result.value().finalized_map_snapshot);
  EXPECT_EQ(result.value().diagnostics.live_correspondences, points.size());
  EXPECT_EQ(result.value().diagnostics.finalized_map_correspondences, 0U);
  EXPECT_EQ(result.value().target_snapshots.front()->rows().size(), points.size());
}

TEST(DirectPointIcp, MixedLiveAndMapUseExclusiveOwnershipOneHuberAndDeterministicChecksums) {
  const std::vector<Eigen::Vector3d> cluster_a = volumePoints();
  std::vector<Eigen::Vector3d> live_points = cluster_a;
  for (std::size_t index = 0U; index < 10U; ++index) {
    live_points[index].y() += 0.30;
  }
  for (std::size_t index = 10U; index < 20U; ++index) {
    live_points[index].y() -= 0.30;
  }
  std::vector<Eigen::Vector3d> cluster_b;
  cluster_b.reserve(cluster_a.size());
  for (std::size_t index = 0U; index < cluster_a.size(); ++index) {
    Eigen::Vector3d point = cluster_a[index] + Eigen::Vector3d{10.0, 0.0, 0.0};
    if (index < 10U) {
      point.y() += 0.30;
    } else if (index < 20U) {
      point.y() -= 0.30;
    }
    cluster_b.push_back(point);
  }
  std::vector<Eigen::Vector3d> source_points = cluster_a;
  for (const Eigen::Vector3d& point : cluster_a) {
    source_points.push_back(point + Eigen::Vector3d{10.0, 0.0, 0.0});
  }
  const core::Pose3d global_pose =
      pose(Eigen::Vector3d{49.0, -13.0, 2.0}, Eigen::Vector3d{0.08, -0.03, 0.21});

  const auto live_target =
      cloud(live_points, core::MeasurementId{70U}, core::FusionTime{1'000U}, global_pose);
  const auto map_target =
      cloud(cluster_b, core::MeasurementId{71U}, core::FusionTime{2'000U}, global_pose);
  const auto source =
      cloud(source_points, core::MeasurementId{72U}, core::FusionTime{3'000U}, global_pose);
  auto map = finalizedMap();
  static_cast<void>(insertFinalizedMapCloud(
      map, map_target, core::StateId{2U}, core::FactorBatchId{71U},
      core::LocalGraphRevision{2U}, global_pose));
  const auto map_view = map.readView();
  LidarRegistrationTarget live = targetRecord(core::StateId{1U}, live_target, source);
  // Exercise the API's admitted seed-consistency tolerance. The internal
  // objective must canonicalize this redundant relative seed from the two
  // odometry poses and freeze residual metrics in the exact factor frame.
  live.T_target_source_seed.translation().y() += 5.0e-7;

  LidarRegistrationConfig single_config = preciseConfig();
  single_config.maximum_targets = 2U;
  single_config.parallel_worker_count = 1U;
  single_config.maximum_correspondence_distance_m = 0.4;
  single_config.maximum_huber_delta_m = 0.35;
  const auto single =
      registerLidarScan(core::StateId{3U}, source, std::span{&live, 1U}, map_view,
                        core::LocalGraphRevision{3U}, single_config);
  ASSERT_TRUE(single) << single.error().detail;
  ASSERT_EQ(single.value().target_snapshots.size(), 1U);
  ASSERT_TRUE(single.value().finalized_map_snapshot);
  EXPECT_EQ(single.value().diagnostics.target_count, 2U);
  EXPECT_EQ(single.value().diagnostics.live_correspondences, cluster_a.size());
  // The two separated spatial clusters split ownership exactly between the
  // pose-aware live factor and the persistent finalized-map factor.
  EXPECT_EQ(single.value().diagnostics.finalized_map_correspondences, cluster_b.size());
  EXPECT_EQ(single.value().diagnostics.correspondences, source_points.size());
  EXPECT_EQ(single.value().target_snapshots.front()->huberDeltaM(),
            single.value().finalized_map_snapshot->huberDeltaM());
  EXPECT_EQ(single.value().target_snapshots.front()->huberDeltaM(),
            single.value().diagnostics.huber_delta_m);
  const auto live_information =
      lidarFactorInformation(single.value().target_snapshots.front(), single_config, 0.5);
  ASSERT_TRUE(live_information) << live_information.error().detail;
  const auto aggregate = summarizeLidarFactorSnapshots(
      single.value().target_snapshots, single.value().finalized_map_snapshot);
  ASSERT_TRUE(aggregate) << aggregate.error().detail;
  EXPECT_EQ(aggregate.value().correspondences, single.value().diagnostics.correspondences);
  std::size_t downweighted_map_rows = 0U;
  for (const auto& row : single.value().finalized_map_snapshot->rows()) {
    downweighted_map_rows += row.association_huber_weight < 0.5 ? 1U : 0U;
  }
  EXPECT_GE(downweighted_map_rows, 15U);

  std::set<std::uint32_t> stable_source_owners;
  for (const auto& row : single.value().target_snapshots.front()->rows()) {
    EXPECT_TRUE(stable_source_owners.insert(row.source_index).second);
  }
  for (const auto& row : single.value().finalized_map_snapshot->rows()) {
    EXPECT_TRUE(stable_source_owners.insert(row.source_index).second);
    EXPECT_GE(row.source_index, cluster_a.size());
  }
  EXPECT_EQ(stable_source_owners.size(), single.value().diagnostics.correspondences);

  LidarRegistrationConfig parallel_config = single_config;
  parallel_config.parallel_worker_count = 4U;
  const auto parallel =
      registerLidarScan(core::StateId{3U}, source, std::span{&live, 1U}, map_view,
                        core::LocalGraphRevision{3U}, parallel_config);
  ASSERT_TRUE(parallel) << parallel.error().detail;
  expectBitIdenticalResult(single.value(), parallel.value());

  auto& tampered = const_cast<FrozenFinalizedMapPointCorrespondence&>(
      parallel.value().finalized_map_snapshot->rows().front());
  tampered.target_point_odom.x() += 0.01;
  const auto rejected =
      lidarFinalizedMapFactorInformation(parallel.value().finalized_map_snapshot, parallel_config);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LidarRegistrationErrorCode::InvalidTarget);
}

TEST(DirectPointIcp, UnderSupportedMapChannelIsPrunedAndLiveOwnershipIsRebuilt) {
  const std::vector<Eigen::Vector3d> live_points = volumePoints();
  std::vector<Eigen::Vector3d> sparse_map_points;
  for (std::size_t index = 0U; index < 10U; ++index) {
    sparse_map_points.push_back(live_points[index] + Eigen::Vector3d{10.0, 0.0, 0.0});
  }
  std::vector<Eigen::Vector3d> source_points = live_points;
  source_points.insert(source_points.end(), sparse_map_points.begin(), sparse_map_points.end());
  const auto live_target =
      cloud(live_points, core::MeasurementId{80U}, core::FusionTime{1'000U}, core::Pose3d{});
  const auto map_target =
      cloud(sparse_map_points, core::MeasurementId{81U}, core::FusionTime{2'000U}, core::Pose3d{});
  const auto source =
      cloud(source_points, core::MeasurementId{82U}, core::FusionTime{3'000U}, core::Pose3d{});
  auto map = finalizedMap();
  static_cast<void>(insertFinalizedMapCloud(
      map, map_target, core::StateId{2U}, core::FactorBatchId{81U},
      core::LocalGraphRevision{2U}, core::Pose3d{}));
  const auto map_view = map.readView();
  const LidarRegistrationTarget live = targetRecord(core::StateId{1U}, live_target, source);
  LidarRegistrationConfig config = preciseConfig();
  config.maximum_targets = 2U;

  const auto result =
      registerLidarScan(core::StateId{3U}, source, std::span{&live, 1U}, map_view,
                        core::LocalGraphRevision{3U}, config);
  ASSERT_TRUE(result) << result.error().detail;
  ASSERT_EQ(result.value().target_snapshots.size(), 1U);
  EXPECT_FALSE(result.value().finalized_map_snapshot);
  EXPECT_EQ(result.value().diagnostics.target_count, 1U);
  EXPECT_EQ(result.value().diagnostics.correspondences, live_points.size());
  EXPECT_GE(result.value().work.association_builds, 2U);
  EXPECT_GE(result.value().work.composite_index_builds, 2U);
  EXPECT_GT(result.value().work.finalized_map_candidate_voxel_lookups, 0U);
  const auto information =
      lidarFactorInformation(result.value().target_snapshots.front(), config, 0.5);
  ASSERT_TRUE(information) << information.error().detail;
  const auto aggregate = summarizeLidarFactorSnapshots(result.value().target_snapshots);
  ASSERT_TRUE(aggregate) << aggregate.error().detail;
  EXPECT_EQ(aggregate.value().correspondences, result.value().diagnostics.correspondences);
}

TEST(DirectPointIcp, StaleFinalizedMapReadViewFailsTheAssociationAtomically) {
  const std::vector<Eigen::Vector3d> points = volumePoints();
  const auto live_target =
      cloud(points, core::MeasurementId{90U}, core::FusionTime{1'000U}, core::Pose3d{});
  const auto first_map_target =
      cloud(points, core::MeasurementId{91U}, core::FusionTime{1'100U}, core::Pose3d{});
  const auto source =
      cloud(points, core::MeasurementId{93U}, core::FusionTime{3'000U}, core::Pose3d{});
  auto map = finalizedMap();
  static_cast<void>(insertFinalizedMapCloud(
      map, first_map_target, core::StateId{1U}, core::FactorBatchId{91U},
      core::LocalGraphRevision{1U}, core::Pose3d{}));
  const auto stale_view = map.readView();
  const auto second_map_target = cloud({Eigen::Vector3d{30.0, 0.0, 0.0}}, core::MeasurementId{92U},
                                       core::FusionTime{1'200U}, core::Pose3d{});
  static_cast<void>(insertFinalizedMapCloud(
      map, second_map_target, core::StateId{2U}, core::FactorBatchId{92U},
      core::LocalGraphRevision{2U}, core::Pose3d{}));
  const LidarRegistrationTarget live = targetRecord(core::StateId{3U}, live_target, source);
  LidarRegistrationConfig config = preciseConfig();
  config.maximum_targets = 2U;

  const auto rejected =
      registerLidarScan(core::StateId{4U}, source, std::span{&live, 1U}, stale_view,
                        core::LocalGraphRevision{3U}, config);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LidarRegistrationErrorCode::NumericalFailure);
  EXPECT_EQ(rejected.error().work.finalized_map_stale_fallbacks,
            rejected.error().work.source_points_selected);
  EXPECT_EQ(rejected.error().work.invalid_target_points,
            rejected.error().work.source_points_selected);
}

TEST(DirectPointIcp, InvalidConfigSeedAndCorrectionGateFailClosed) {
  const std::vector<Eigen::Vector3d> source_points = volumePoints();
  const auto source =
      cloud(source_points, core::MeasurementId{50U}, core::FusionTime{2'000U}, core::Pose3d{});
  const auto coincident_target =
      cloud(source_points, core::MeasurementId{51U}, core::FusionTime{1'000U}, core::Pose3d{});
  LidarRegistrationTarget record = targetRecord(core::StateId{8U}, coincident_target, source);

  LidarRegistrationConfig invalid_config = preciseConfig();
  invalid_config.source_voxel_size_m = 0.0;
  EXPECT_FALSE(isValidLidarRegistrationConfig(invalid_config));
  const auto invalid =
      registerLidarScan(core::StateId{9U}, source, std::span{&record, 1U}, invalid_config);
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().code, LidarRegistrationErrorCode::InvalidConfig);

  record.T_target_source_seed = core::Pose3d{Sophus::SO3d{}, Eigen::Vector3d{0.1, 0.0, 0.0}};
  const auto inconsistent =
      registerLidarScan(core::StateId{9U}, source, std::span{&record, 1U}, preciseConfig());
  ASSERT_FALSE(inconsistent);
  EXPECT_EQ(inconsistent.error().code, LidarRegistrationErrorCode::InconsistentSeed);

  const core::Pose3d correction{Sophus::SO3d{}, Eigen::Vector3d{0.25, 0.0, 0.0}};
  const auto displaced_target =
      cloud(transformed(source_points, correction), core::MeasurementId{52U},
            core::FusionTime{1'100U}, core::Pose3d{});
  const LidarRegistrationTarget displaced_record =
      targetRecord(core::StateId{8U}, displaced_target, source);
  LidarRegistrationConfig gated_config = preciseConfig();
  gated_config.maximum_correction_translation_m = 0.05;
  gated_config.maximum_lm_damping_retries = 2U;
  const auto gated =
      registerLidarScan(core::StateId{9U}, source, std::span{&displaced_record, 1U}, gated_config);
  ASSERT_FALSE(gated);
  EXPECT_EQ(gated.error().code, LidarRegistrationErrorCode::NoDecreasingStep);
  EXPECT_EQ(gated.error().work.accepted_steps, 0U);
}

}  // namespace
}  // namespace meridian::local
