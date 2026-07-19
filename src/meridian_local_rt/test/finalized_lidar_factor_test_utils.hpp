#pragma once

#include <cstdint>
#include <memory>
#include <sophus/so3.hpp>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "lidar_registration_cloud_test_utils.hpp"
#include "meridian/local/finalized_lidar_target_map.hpp"
#include "meridian/local/lidar_registration.hpp"

namespace meridian::local::test {

[[nodiscard]] inline std::vector<Eigen::Vector3d> finalizedMapVolumePoints() {
  std::vector<Eigen::Vector3d> points;
  for (int x = -4; x <= 4; ++x) {
    for (int y = -3; y <= 3; ++y) {
      for (int z = -2; z <= 2; ++z) {
        points.emplace_back(0.55 * static_cast<double>(x), 0.57 * static_cast<double>(y),
                            0.61 * static_cast<double>(z) + 0.017 * static_cast<double>(x * y));
      }
    }
  }
  return points;
}

[[nodiscard]] inline core::FactorBatchMetadata finalizedOwnerMetadata(
    const LidarRegistrationCloud& cloud, core::StateId state, core::FactorBatchId batch_id,
    core::SensorInstanceId sensor, core::OdomEpoch odom_epoch,
    core::CalibrationEpoch calibration = core::CalibrationEpoch{1U}) {
  core::ObservationLineage lineage = cloud.lineage;
  lineage.id = core::ObservationLineageId{10'000U + batch_id.value()};
  for (core::ObservationUsage& usage : lineage.usage) {
    usage.consumer = core::DerivedRecordId{20'000U + batch_id.value()};
    const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
    if (measurement != nullptr && *measurement == cloud.source_sweep) {
      usage.role = core::ObservationRole::PrimaryResidual;
      usage.factor_group = core::FactorGroupId{30'000U + batch_id.value()};
    }
  }
  lineage.checksum = {};
  const auto lineage_checksum = recomputeAcceptedLidarLineageChecksum(lineage);
  if (!lineage_checksum) {
    throw std::runtime_error("failed to checksum finalized-owner test lineage");
  }
  lineage.checksum = lineage_checksum.value();

  core::DirectionalObservability observability;
  observability.eigenvalues.setOnes();
  observability.rank = 6U;
  observability.absolute_eigenvalue_threshold = 0.1;
  observability.relative_eigenvalue_threshold = 0.0;
  observability.supported_variables = {core::DirectionalVariable::PoseTranslation,
                                       core::DirectionalVariable::PoseRotation};
  observability.endpoints = {{core::DirectionalEndpointRole::Unary, state, cloud.reference_time}};

  core::FactorBatchMetadata metadata;
  metadata.header.schema_version = 1U;
  metadata.header.trace = core::TraceId{1U};
  metadata.header.producer = core::ProducerId{2U};
  metadata.header.session = core::SessionId{3U};
  metadata.header.created_at = cloud.reference_time + core::Duration{4LL};
  metadata.header.config = core::ConfigRevision{4U};
  metadata.header.direct_calibration = calibration;
  metadata.batch_id = batch_id;
  metadata.odom_epoch = odom_epoch;
  metadata.sensor = sensor;
  metadata.timing.support = core::TimeRange{cloud.reference_time - core::Duration{1LL},
                                            cloud.reference_time + core::Duration{1LL}};
  metadata.timing.measurement_timestamps = {cloud.reference_time};
  metadata.timing.reference_time = cloud.reference_time;
  metadata.timing.produced_at = cloud.reference_time + core::Duration{3LL};
  metadata.health = core::SensorHealthSnapshot{sensor, core::SensorHealthState::Active,
                                               core::SensorRecoveryEpoch{1U}, 1U,
                                               cloud.reference_time + core::Duration{2LL}};
  metadata.map_eligible = true;
  metadata.directional_observability.push_back(std::move(observability));
  metadata.lineage = std::move(lineage);
  if (core::validateFactorBatchMetadata(metadata) !=
      core::FactorBatchMetadataValidationError::None) {
    throw std::runtime_error("failed to construct finalized-owner test metadata");
  }
  return metadata;
}

struct FinalizedMapRegistrationFixture {
  FinalizedLidarTargetMap map;
  std::shared_ptr<const LidarRegistrationCloud> source;
  LidarRegistrationConfig config;
  LidarRegistrationResult registration;
};

[[nodiscard]] inline FinalizedMapRegistrationFixture finalizedMapRegistrationFixture(
    core::StateId source_state = core::StateId{12U},
    core::FusionTime source_time = core::FusionTime{2'000'000'000LL}) {
  constexpr std::uint64_t kOwnerSweep = 501U;
  constexpr std::uint64_t kSourceSweep = 601U;
  const core::OdomEpoch odom_epoch{1U};
  const core::SensorInstanceId sensor = core::SensorInstanceId::lidar(core::LidarId{1U});

  FinalizedLidarTargetMapConfig map_config;
  map_config.odom_epoch = odom_epoch;
  map_config.sensor = sensor;
  auto created_map = FinalizedLidarTargetMap::create(map_config);
  if (!created_map) {
    throw std::runtime_error("failed to create finalized-map factor test map: " +
                             created_map.error().detail);
  }
  FinalizedLidarTargetMap map = std::move(created_map).value();

  const std::vector<Eigen::Vector3d> points_odom = finalizedMapVolumePoints();
  const std::vector<core::MeasurementId> owner_imu_support{core::MeasurementId{701U},
                                                           core::MeasurementId{702U}};
  const auto owner_cloud = sealedLidarRegistrationCloud(
      points_odom, core::MeasurementId{kOwnerSweep}, core::FusionTime{0LL}, core::Pose3d{}, 1.0, {},
      owner_imu_support);
  FinalizedLidarSweep finalized;
  finalized.batch = SensorFactorBatchRef{sensor, core::FactorBatchId{77U}};
  finalized.accepted_batch_metadata = finalizedOwnerMetadata(
      *owner_cloud, core::StateId{10U}, finalized.batch.batch_id, sensor, odom_epoch);
  finalized.admission_revision = core::LocalGraphRevision{3U};
  finalized.admission_kind = MapAdmissionBatchKind::Regular;
  finalized.finalized_state.state = core::StateId{10U};
  finalized.finalized_state.exact_time = owner_cloud->reference_time;
  finalized.finalized_state.odom_epoch = odom_epoch;
  finalized.finalized_state.final_revision = core::LocalGraphRevision{9U};
  finalized.finalized_state.final_estimate.T_odom_imu = core::Pose3d{};
  finalized.finalized_state.pose_covariance.matrix = core::Matrix6d::Identity() * 1.0e-4;
  finalized.calibration = core::CalibrationEpoch{1U};
  finalized.cloud = owner_cloud;
  const auto inserted = map.insertFinalizedSweep(std::move(finalized));
  if (!inserted || !inserted.value().owner) {
    throw std::runtime_error("failed to insert finalized-map factor test sweep: " +
                             (inserted ? std::string{"missing sealed owner"}
                                       : inserted.error().detail));
  }

  const core::Pose3d T_odom_source{Sophus::SO3d::exp(Eigen::Vector3d{0.012, -0.008, 0.009}),
                                   Eigen::Vector3d{0.08, -0.045, 0.03}};
  std::vector<Eigen::Vector3d> source_points;
  source_points.reserve(points_odom.size());
  for (const Eigen::Vector3d& point_odom : points_odom) {
    source_points.push_back(T_odom_source.inverse() * point_odom);
  }
  const std::vector<core::MeasurementId> source_imu_support{core::MeasurementId{801U},
                                                            core::MeasurementId{802U}};
  auto source =
      sealedLidarRegistrationCloud(source_points, core::MeasurementId{kSourceSweep}, source_time,
                                   T_odom_source, 1.0, {}, source_imu_support);

  LidarRegistrationConfig registration_config;
  registration_config.source_voxel_size_m = 0.20;
  registration_config.relative_normalized_observable_eigenvalue = 1.0e-5;
  registration_config.maximum_translation_information = 120.0;
  const auto finalized_view = map.readView();
  const auto registered = registerLidarScan(source_state, source, {}, finalized_view,
                                             core::LocalGraphRevision{10U}, registration_config);
  if (!registered) {
    throw std::runtime_error("finalized-map factor test registration failed: " +
                             registered.error().detail);
  }
  if (!registered.value().finalized_map_snapshot || !registered.value().target_snapshots.empty()) {
    throw std::runtime_error("finalized-map factor test registration has invalid target topology");
  }
  return FinalizedMapRegistrationFixture{std::move(map), std::move(source), registration_config,
                                         registered.value()};
}

}  // namespace meridian::local::test
