#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sophus/so3.hpp>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "lidar_registration_cloud_test_utils.hpp"
#include "meridian/local/finalized_lidar_target_map.hpp"

namespace meridian::local {
namespace {

struct IndexedPoint {
  Eigen::Vector3d point{Eigen::Vector3d::Zero()};
  std::uint32_t source_index{};
};

[[nodiscard]] std::shared_ptr<const LidarRegistrationCloud> indexedCloud(
    std::vector<IndexedPoint> rows, std::uint64_t sweep_id, std::int64_t time_ns = -1,
    std::vector<core::MeasurementId> imu_support = {}) {
  if (rows.empty()) {
    throw std::invalid_argument("test cloud requires rows");
  }
  std::sort(rows.begin(), rows.end(), [](const IndexedPoint& lhs, const IndexedPoint& rhs) {
    return lhs.source_index < rhs.source_index;
  });
  if (std::adjacent_find(rows.begin(), rows.end(),
                         [](const IndexedPoint& lhs, const IndexedPoint& rhs) {
                           return lhs.source_index == rhs.source_index;
                         }) != rows.end()) {
    throw std::invalid_argument("test cloud source indices must be unique");
  }
  const std::uint32_t source_domain = rows.back().source_index + 1U;
  LidarRegistrationCloudData data;
  data.source_sweep = core::MeasurementId{sweep_id};
  data.reference_time =
      core::FusionTime{time_ns >= 0 ? time_ns : static_cast<std::int64_t>(sweep_id * 1'000U)};
  data.T_odom_imu_seed = core::Pose3d{};
  data.layout.width = source_domain;
  data.layout.height = 1U;
  data.layout.organized = false;
  data.points_in_reference_imu = std::make_unique<core::LidarPoints>();
  data.points_in_reference_imu->reserve(source_domain);
  data.points.reserve(rows.size());
  auto selected = rows.begin();
  for (std::uint32_t source_index = 0U; source_index < source_domain; ++source_index) {
    core::LidarPoint raw;
    raw.x = -1'000.0F - static_cast<float>(source_index);
    raw.y = 1'000.0F + static_cast<float>(source_index);
    raw.z = -500.0F;
    if (selected != rows.end() && selected->source_index == source_index) {
      raw.x = static_cast<float>(selected->point.x());
      raw.y = static_cast<float>(selected->point.y());
      raw.z = static_cast<float>(selected->point.z());
    }
    raw.intensity = static_cast<float>(source_index) + 0.5F;
    raw.ring = static_cast<std::uint16_t>(source_index % 64U);
    raw.source_index = source_index;
    data.points_in_reference_imu->push_back(raw);
    if (selected != rows.end() && selected->source_index == source_index) {
      data.points.push_back(LidarRegistrationPoint{
          Eigen::Vector3d{static_cast<double>(raw.x), static_cast<double>(raw.y),
                          static_cast<double>(raw.z)},
          raw.source_index, raw.intensity, raw.ring});
      ++selected;
    }
  }
  data.stats.input_points = data.points_in_reference_imu->size();
  data.stats.valid_range_points = data.points_in_reference_imu->size();
  data.stats.deterministic_voxel_points = rows.size();
  data.imu_support = std::move(imu_support);
  data.lineage = test::lidarLineage(data.source_sweep, 0U, data.imu_support);
  auto cloud = LidarRegistrationCloud::create(std::move(data));
  if (!cloud) {
    throw std::runtime_error("failed to create indexed test cloud: " + cloud.error().detail);
  }
  return std::move(cloud).value();
}

[[nodiscard]] core::FactorBatchMetadata acceptedMetadata(
    const LidarRegistrationCloud& cloud, std::uint64_t state, std::uint64_t batch,
    std::uint64_t recovery, MapAdmissionBatchKind kind, core::OdomEpoch epoch = core::OdomEpoch{1U},
    core::CalibrationEpoch calibration = core::CalibrationEpoch{1U}) {
  core::ObservationLineage lineage = cloud.lineage;
  lineage.id = core::ObservationLineageId{10'000U + batch};
  for (core::ObservationUsage& usage : lineage.usage) {
    usage.consumer = core::DerivedRecordId{batch};
    const auto* source = std::get_if<core::MeasurementId>(&usage.slice.root);
    if (source != nullptr && *source == cloud.source_sweep &&
        kind == MapAdmissionBatchKind::Regular) {
      usage.role = core::ObservationRole::PrimaryResidual;
      usage.factor_group = core::FactorGroupId{batch};
    }
  }
  lineage.checksum = {};
  auto lineage_checksum = recomputeAcceptedLidarLineageChecksum(lineage);
  if (!lineage_checksum) {
    throw std::runtime_error("failed to checksum accepted test lineage");
  }
  lineage.checksum = lineage_checksum.value();

  core::DirectionalObservability observability;
  observability.eigenvalues.setOnes();
  observability.rank = 6U;
  observability.absolute_eigenvalue_threshold = 0.1;
  observability.relative_eigenvalue_threshold = 0.0;
  observability.supported_variables = {core::DirectionalVariable::PoseTranslation,
                                       core::DirectionalVariable::PoseRotation};
  observability.endpoints = {
      {core::DirectionalEndpointRole::Unary, core::StateId{state}, cloud.reference_time}};

  const core::SensorInstanceId sensor = core::SensorInstanceId::lidar(core::LidarId{1U});
  core::FactorBatchMetadata metadata;
  metadata.header.schema_version = 1U;
  metadata.header.trace = core::TraceId{1U};
  metadata.header.producer = core::ProducerId{1U};
  metadata.header.session = core::SessionId{1U};
  metadata.header.created_at = cloud.reference_time + core::Duration{4LL};
  metadata.header.config = core::ConfigRevision{1U};
  metadata.header.direct_calibration = calibration;
  metadata.batch_id = core::FactorBatchId{batch};
  metadata.odom_epoch = epoch;
  metadata.sensor = sensor;
  metadata.timing.support = core::TimeRange{cloud.reference_time - core::Duration{1LL},
                                            cloud.reference_time + core::Duration{1LL}};
  metadata.timing.measurement_timestamps = {cloud.reference_time};
  metadata.timing.reference_time = cloud.reference_time;
  metadata.timing.produced_at = cloud.reference_time + core::Duration{3LL};
  metadata.health = core::SensorHealthSnapshot{sensor, core::SensorHealthState::Active,
                                               core::SensorRecoveryEpoch{recovery}, 1U,
                                               cloud.reference_time + core::Duration{2LL}};
  metadata.map_eligible = true;
  metadata.directional_observability.push_back(std::move(observability));
  metadata.lineage = std::move(lineage);
  if (core::validateFactorBatchMetadata(metadata) !=
      core::FactorBatchMetadataValidationError::None) {
    throw std::runtime_error("failed to construct accepted test FactorBatch metadata");
  }
  return metadata;
}

[[nodiscard]] FinalizedLidarSweep finalizedSweep(
    std::shared_ptr<const LidarRegistrationCloud> cloud, std::uint64_t state, std::uint64_t batch,
    std::uint64_t recovery = 1U, std::uint64_t revision = 1U,
    core::Pose3d final_pose = core::Pose3d{}, core::OdomEpoch epoch = core::OdomEpoch{1U},
    core::CalibrationEpoch calibration = core::CalibrationEpoch{1U},
    MapAdmissionBatchKind kind = MapAdmissionBatchKind::Regular,
    std::uint64_t admission_revision = 1U) {
  FinalizedLidarSweep result;
  result.batch = SensorFactorBatchRef{core::SensorInstanceId::lidar(core::LidarId{1U}),
                                      core::FactorBatchId{batch}};
  result.accepted_batch_metadata =
      acceptedMetadata(*cloud, state, batch, recovery, kind, epoch, calibration);
  result.admission_revision = core::LocalGraphRevision{admission_revision};
  result.admission_kind = kind;
  result.finalized_state.state = core::StateId{state};
  result.finalized_state.exact_time = cloud->reference_time;
  result.finalized_state.odom_epoch = epoch;
  result.finalized_state.final_revision = core::LocalGraphRevision{revision};
  result.finalized_state.final_estimate.T_odom_imu = std::move(final_pose);
  result.finalized_state.pose_covariance.matrix = core::Matrix6d::Identity() * 0.01;
  result.calibration = calibration;
  result.cloud = std::move(cloud);
  return result;
}

[[nodiscard]] FinalizedLidarTargetMap mapWith(FinalizedLidarTargetMapConfig config = {}) {
  if (!config.odom_epoch.valid()) {
    config.odom_epoch = core::OdomEpoch{1U};
  }
  if (!config.sensor.valid()) {
    config.sensor = core::SensorInstanceId::lidar(core::LidarId{1U});
  }
  auto created = FinalizedLidarTargetMap::create(config);
  if (!created) {
    throw std::runtime_error("failed to create finalized test map: " + created.error().detail);
  }
  return std::move(created).value();
}

struct MapSemanticSnapshot {
  std::size_t retained_points{};
  std::size_t retained_voxels{};
  std::size_t admitted_sweeps{};
  std::uint64_t version{};
  core::ContentHash checksum{};
};

TEST(AcceptedLidarLineageChecksum, AuthenticatesAbsentRawChecksumsAndMetadataTampering) {
  const std::array imu_support{core::MeasurementId{11U}, core::MeasurementId{12U}};
  core::ObservationLineage lineage = test::lidarLineage(
      core::MeasurementId{700U}, 700U, imu_support);
  const auto with_raw_checksums = recomputeAcceptedLidarLineageChecksum(lineage);
  ASSERT_TRUE(with_raw_checksums);
  for (core::ObservationUsage& usage : lineage.usage) {
    usage.slice.source_checksum = {};
  }
  const auto absent_raw_checksums = recomputeAcceptedLidarLineageChecksum(lineage);
  ASSERT_TRUE(absent_raw_checksums);
  EXPECT_NE(absent_raw_checksums.value(), with_raw_checksums.value());
  lineage.checksum = absent_raw_checksums.value();
  const auto stable = recomputeAcceptedLidarLineageChecksum(lineage);
  ASSERT_TRUE(stable);
  EXPECT_EQ(stable.value(), lineage.checksum);

  lineage.usage.front().consumer =
      core::DerivedRecordId{lineage.usage.front().consumer.value() + 1U};
  const auto tampered = recomputeAcceptedLidarLineageChecksum(lineage);
  ASSERT_TRUE(tampered);
  EXPECT_NE(tampered.value(), lineage.checksum);
}

[[nodiscard]] MapSemanticSnapshot semanticSnapshot(const FinalizedLidarTargetMap& map) {
  const auto& stats = map.statistics();
  return MapSemanticSnapshot{stats.retained_points, stats.retained_query_voxels,
                             stats.admitted_sweeps, stats.version, stats.checksum};
}

void expectSemanticSnapshot(const FinalizedLidarTargetMap& map,
                            const MapSemanticSnapshot& expected) {
  const auto actual = semanticSnapshot(map);
  EXPECT_EQ(actual.retained_points, expected.retained_points);
  EXPECT_EQ(actual.retained_voxels, expected.retained_voxels);
  EXPECT_EQ(actual.admitted_sweeps, expected.admitted_sweeps);
  EXPECT_EQ(actual.version, expected.version);
  EXPECT_EQ(actual.checksum, expected.checksum);
}

TEST(FinalizedLidarTargetMap, RejectsInvalidConfiguration) {
  FinalizedLidarTargetMapConfig config;
  EXPECT_FALSE(FinalizedLidarTargetMap::create(config));

  config.odom_epoch = core::OdomEpoch{1U};
  config.sensor = core::SensorInstanceId::camera(core::CameraId{1U});
  EXPECT_FALSE(FinalizedLidarTargetMap::create(config));
  config.sensor = core::SensorInstanceId::lidar(core::LidarId{1U});
  config.query_voxel_size_m = 0.0;
  EXPECT_FALSE(FinalizedLidarTargetMap::create(config));
  config.query_voxel_size_m = 1.0;
  config.insertion_voxel_size_m = 1.1;
  EXPECT_FALSE(FinalizedLidarTargetMap::create(config));
  config.insertion_voxel_size_m = 0.5;
  config.maximum_points_per_query_voxel = 0U;
  EXPECT_FALSE(FinalizedLidarTargetMap::create(config));
  config.maximum_points_per_query_voxel = 20U;
  config.minimum_point_separation_m = 0.0;
  EXPECT_FALSE(FinalizedLidarTargetMap::create(config));
  config.minimum_point_separation_m = kDefaultFinalizedLidarMinimumSeparationM;
  config.maximum_supported_query_distance_m = 1.01;
  EXPECT_FALSE(FinalizedLidarTargetMap::create(config));
  config.maximum_supported_query_distance_m = 0.5;
  config.maximum_radius_m = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(FinalizedLidarTargetMap::create(config));
  config.maximum_radius_m = 100.0;
  config.hard_point_capacity = 0U;
  EXPECT_FALSE(FinalizedLidarTargetMap::create(config));
}

TEST(FinalizedLidarTargetMap, IsDeterministicAcrossConstructibleInputOrdering) {
  const std::vector<IndexedPoint> canonical{{{0.10, 0.10, 0.10}, 7U},
                                            {{0.24, 0.25, 0.25}, 2U},
                                            {{0.75, 0.10, 0.10}, 11U},
                                            {{1.25, 0.10, 0.10}, 4U}};
  std::vector<IndexedPoint> reversed = canonical;
  std::reverse(reversed.begin(), reversed.end());
  const auto cloud_a = indexedCloud(canonical, 10U);
  const auto cloud_b = indexedCloud(reversed, 10U);
  ASSERT_EQ(cloud_a->checksum, cloud_b->checksum);

  auto map_a = mapWith();
  auto map_b = mapWith();
  const auto inserted_a = map_a.insertFinalizedSweep(finalizedSweep(cloud_a, 5U, 8U));
  const auto inserted_b = map_b.insertFinalizedSweep(finalizedSweep(cloud_b, 5U, 8U));
  ASSERT_TRUE(inserted_a) << inserted_a.error().detail;
  ASSERT_TRUE(inserted_b) << inserted_b.error().detail;
  EXPECT_EQ(inserted_a.value().admitted_points, inserted_b.value().admitted_points);
  EXPECT_EQ(inserted_a.value().checksum, inserted_b.value().checksum);
  EXPECT_EQ(semanticSnapshot(map_a).checksum, semanticSnapshot(map_b).checksum);
}

TEST(FinalizedLidarTargetMap, SelectsOneCenterNearestPointPerHalfMeterInsertionVoxel) {
  auto map = mapWith();
  const auto cloud = indexedCloud(
      {{{0.10, 0.10, 0.10}, 9U}, {{0.25, 0.25, 0.25}, 4U}, {{0.40, 0.40, 0.40}, 2U}}, 20U);
  const auto inserted = map.insertFinalizedSweep(finalizedSweep(cloud, 2U, 20U));
  ASSERT_TRUE(inserted) << inserted.error().detail;
  EXPECT_EQ(inserted.value().input_points, 3U);
  EXPECT_EQ(inserted.value().insertion_voxels, 1U);
  EXPECT_EQ(inserted.value().insertion_selection_discarded_points, 2U);
  EXPECT_EQ(inserted.value().admitted_points, 1U);
  EXPECT_EQ(inserted.value().admitted_points, inserted.value().retained_points);

  const auto neighbor = map.nearestExact(Eigen::Vector3d{0.25, 0.25, 0.25}, 0.5);
  ASSERT_TRUE(neighbor.query_valid);
  ASSERT_TRUE(neighbor.found);
  EXPECT_EQ(neighbor.point.source_index, 4U);
  EXPECT_DOUBLE_EQ(neighbor.distance_squared_m2, 0.0);
}

TEST(FinalizedLidarTargetMap, EnforcesDefaultSeparationAndTwentyPointQueryBlockCap) {
  EXPECT_DOUBLE_EQ(
      kDefaultFinalizedLidarMinimumSeparationM * kDefaultFinalizedLidarMinimumSeparationM,
      1.0 / 20.0);
  auto separation_map = mapWith();
  ASSERT_TRUE(separation_map.insertFinalizedSweep(
      finalizedSweep(indexedCloud({{{0.49, 0.10, 0.10}, 0U}}, 30U), 1U, 30U)));
  const auto too_close = separation_map.insertFinalizedSweep(
      finalizedSweep(indexedCloud({{{0.51, 0.10, 0.10}, 0U}}, 31U), 2U, 31U));
  ASSERT_TRUE(too_close) << too_close.error().detail;
  EXPECT_EQ(too_close.value().admitted_points, 0U);
  EXPECT_EQ(too_close.value().minimum_separation_discarded_points, 1U);
  EXPECT_EQ(separation_map.statistics().retained_points, 1U);

  auto capped_map = mapWith();
  std::vector<Eigen::Vector3d> grid;
  for (double x : {0.05, 0.30, 0.55, 0.80}) {
    for (double y : {0.05, 0.30, 0.55, 0.80}) {
      for (double z : {0.05, 0.30, 0.55, 0.80}) {
        grid.emplace_back(x, y, z);
      }
    }
  }
  for (std::size_t index = 0U; index < 21U; ++index) {
    const auto inserted = capped_map.insertFinalizedSweep(
        finalizedSweep(indexedCloud({{grid[index], 0U}}, 100U + index), index + 1U, 100U + index));
    ASSERT_TRUE(inserted) << inserted.error().detail;
    if (index < 20U) {
      EXPECT_EQ(inserted.value().admitted_points, 1U);
    } else {
      EXPECT_EQ(inserted.value().admitted_points, 0U);
      EXPECT_EQ(inserted.value().query_voxel_capacity_discarded_points, 1U);
    }
  }
  EXPECT_EQ(capped_map.statistics().retained_query_voxels, 1U);
  EXPECT_EQ(capped_map.statistics().retained_points, 20U);
}

TEST(FinalizedLidarTargetMap, QueriesAdjacentCellsExactlyAndUsesStableTieBreak) {
  auto adjacent_map = mapWith();
  ASSERT_TRUE(adjacent_map.insertFinalizedSweep(
      finalizedSweep(indexedCloud({{{0.75, 0.0, 0.0}, 3U}}, 200U), 8U, 200U)));
  const auto adjacent = adjacent_map.nearestExact(Eigen::Vector3d{1.01, 0.0, 0.0}, 0.5);
  ASSERT_TRUE(adjacent.query_valid);
  ASSERT_TRUE(adjacent.found);
  ASSERT_TRUE(adjacent.point.owner);
  EXPECT_EQ(adjacent.point.owner->sweep, core::MeasurementId{200U});
  EXPECT_EQ(adjacent.voxel_lookups, 27U);
  EXPECT_EQ(adjacent.occupied_voxels, 1U);
  EXPECT_EQ(adjacent.points_examined, 1U);

  auto tie_map = mapWith();
  ASSERT_TRUE(tie_map.insertFinalizedSweep(
      finalizedSweep(indexedCloud({{{1.25, 0.0, 0.0}, 1U}}, 201U), 2U, 201U)));
  ASSERT_TRUE(tie_map.insertFinalizedSweep(
      finalizedSweep(indexedCloud({{{0.75, 0.0, 0.0}, 9U}}, 202U), 9U, 202U)));
  const auto tie = tie_map.nearestExact(Eigen::Vector3d{1.0, 0.0, 0.0}, 0.5);
  ASSERT_TRUE(tie.found);
  EXPECT_DOUBLE_EQ(tie.distance_squared_m2, 0.25 * 0.25);
  ASSERT_TRUE(tie.point.owner);
  EXPECT_EQ(tie.point.owner->finalized_state.state, core::StateId{2U});
  EXPECT_EQ(tie.point.owner->sweep, core::MeasurementId{201U});
  EXPECT_EQ(tie.voxel_lookups, 27U);
  EXPECT_EQ(tie.occupied_voxels, 2U);
  EXPECT_EQ(tie.points_examined, 2U);

  const auto invalid = tie_map.nearestExact(
      Eigen::Vector3d{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}, 0.5);
  EXPECT_FALSE(invalid.query_valid);
  EXPECT_FALSE(invalid.found);
  EXPECT_EQ(invalid.voxel_lookups, 0U);
}

TEST(FinalizedLidarTargetMap, TransformsOnceAndReturnsCompleteStableProvenance) {
  auto map = mapWith();
  const core::Pose3d final_pose{Sophus::SO3d::exp(Eigen::Vector3d{0.0, 0.0, std::acos(-1.0) / 2.0}),
                                Eigen::Vector3d{10.0, 2.0, -1.0}};
  const auto cloud = indexedCloud({{{1.0, 0.0, 0.0}, 42U}, {{2.0, 0.0, 0.0}, 43U}}, 300U, 9'000,
                                  {core::MeasurementId{9'001U}, core::MeasurementId{9'002U}});
  const core::ContentHash raw_lineage_checksum = cloud->lineage.checksum;
  const auto inserted =
      map.insertFinalizedSweep(finalizedSweep(cloud, 12U, 77U, 4U, 99U, final_pose));
  ASSERT_TRUE(inserted) << inserted.error().detail;

  const Eigen::Vector3d expected{10.0, 3.0, -1.0};
  const auto neighbor = map.nearestExact(expected, 0.5);
  ASSERT_TRUE(neighbor.found);
  EXPECT_TRUE(neighbor.point.point_odom.isApprox(expected, 1.0e-12));
  ASSERT_TRUE(neighbor.point.owner);
  EXPECT_EQ(neighbor.point.owner->finalized_state.odom_epoch, core::OdomEpoch{1U});
  EXPECT_EQ(neighbor.point.owner->finalized_state.state, core::StateId{12U});
  EXPECT_EQ(neighbor.point.owner->finalized_state.exact_time, core::FusionTime{9'000});
  EXPECT_EQ(neighbor.point.owner->sweep, core::MeasurementId{300U});
  EXPECT_EQ(neighbor.point.owner->batch.batch_id, core::FactorBatchId{77U});
  EXPECT_EQ(neighbor.point.owner->batch.sensor, core::SensorInstanceId::lidar(core::LidarId{1U}));
  EXPECT_EQ(neighbor.point.owner->admission_revision, core::LocalGraphRevision{1U});
  EXPECT_EQ(neighbor.point.owner->admission_kind, MapAdmissionBatchKind::Regular);
  EXPECT_EQ(neighbor.point.owner->admission.health.recovery_epoch,
            core::SensorRecoveryEpoch{4U});
  EXPECT_TRUE(neighbor.point.owner->admission.map_eligible);
  EXPECT_EQ(neighbor.point.owner->admission.health.state,
            core::SensorHealthState::Active);
  EXPECT_EQ(neighbor.point.source_index, 42U);
  EXPECT_EQ(neighbor.point.owner->cloud_checksum, cloud->checksum);
  EXPECT_EQ(neighbor.point.owner->calibration, core::CalibrationEpoch{1U});
  EXPECT_EQ(neighbor.point.owner->cloud_lineage.id, cloud->lineage.id);
  EXPECT_EQ(cloud->lineage.checksum, raw_lineage_checksum);
  core::ObservationLineage accepted_owner_lineage = neighbor.point.owner->cloud_lineage;
  const core::ContentHash accepted_owner_checksum = accepted_owner_lineage.checksum;
  accepted_owner_lineage.checksum = {};
  const auto recomputed_owner_checksum =
      recomputeAcceptedLidarLineageChecksum(accepted_owner_lineage);
  ASSERT_TRUE(recomputed_owner_checksum) << recomputed_owner_checksum.error().detail;
  EXPECT_EQ(accepted_owner_checksum, recomputed_owner_checksum.value());
  EXPECT_NE(accepted_owner_checksum, raw_lineage_checksum);
  EXPECT_EQ(neighbor.point.owner->cloud_lineage.usage.size(), cloud->lineage.usage.size());
  EXPECT_EQ(neighbor.point.owner->imu_support, cloud->imu_support);
  EXPECT_EQ(neighbor.point.owner->admission.accepted_lineage,
            inserted.value().owner->admission.accepted_lineage);
  EXPECT_TRUE(core::contentHashPresent(
      neighbor.point.owner->admission.accepted_lineage_checksum));
  EXPECT_TRUE(core::contentHashPresent(
      neighbor.point.owner->admission.accepted_batch_metadata_checksum));
  EXPECT_EQ(neighbor.point.owner->finalized_state.final_revision, core::LocalGraphRevision{99U});
  EXPECT_TRUE(neighbor.point.owner->finalized_state.final_estimate.T_odom_imu.matrix().isApprox(
      final_pose.matrix(), 1.0e-12));
  EXPECT_TRUE(neighbor.point.owner->finalized_state.pose_covariance.matrix.isApprox(
      core::Matrix6d::Identity() * 0.01, 1.0e-15));
  EXPECT_TRUE(core::contentHashPresent(neighbor.point.owner->final_pose_covariance_checksum));

  const auto same_owner = map.nearestExact(Eigen::Vector3d{10.0, 4.0, -1.0}, 0.5);
  ASSERT_TRUE(same_owner.found);
  EXPECT_EQ(same_owner.point.source_index, 43U);
  EXPECT_EQ(same_owner.point.owner, neighbor.point.owner);
}

TEST(FinalizedLidarTargetMap, DuplicateAndInvalidAdmissionsAreSemanticallyAtomic) {
  auto map = mapWith();
  const auto first_cloud = indexedCloud({{{0.0, 0.0, 0.0}, 0U}}, 400U);
  ASSERT_TRUE(map.insertFinalizedSweep(finalizedSweep(first_cloud, 1U, 400U)));
  const MapSemanticSnapshot before = semanticSnapshot(map);

  const auto duplicate_sweep = map.insertFinalizedSweep(finalizedSweep(first_cloud, 2U, 401U));
  ASSERT_FALSE(duplicate_sweep);
  EXPECT_EQ(duplicate_sweep.error().code, FinalizedLidarTargetMapErrorCode::DuplicateSweep);
  expectSemanticSnapshot(map, before);

  const auto second_cloud = indexedCloud({{{2.0, 0.0, 0.0}, 0U}}, 401U);
  const auto duplicate_batch = map.insertFinalizedSweep(finalizedSweep(second_cloud, 2U, 400U));
  ASSERT_FALSE(duplicate_batch);
  EXPECT_EQ(duplicate_batch.error().code, FinalizedLidarTargetMapErrorCode::DuplicateFactorBatch);
  expectSemanticSnapshot(map, before);

  const auto wrong_epoch = map.insertFinalizedSweep(
      finalizedSweep(second_cloud, 2U, 401U, 1U, 1U, core::Pose3d{}, core::OdomEpoch{2U}));
  ASSERT_FALSE(wrong_epoch);
  EXPECT_EQ(wrong_epoch.error().code, FinalizedLidarTargetMapErrorCode::EpochMismatch);
  expectSemanticSnapshot(map, before);

  core::Pose3d invalid_pose{};
  invalid_pose.translation().x() = std::numeric_limits<double>::quiet_NaN();
  const auto non_finite_pose =
      map.insertFinalizedSweep(finalizedSweep(second_cloud, 2U, 401U, 1U, 1U, invalid_pose));
  ASSERT_FALSE(non_finite_pose);
  EXPECT_EQ(non_finite_pose.error().code, FinalizedLidarTargetMapErrorCode::InvalidPose);
  expectSemanticSnapshot(map, before);

  const auto wrong_calibration =
      map.insertFinalizedSweep(finalizedSweep(second_cloud, 2U, 401U, 1U, 1U, core::Pose3d{},
                                              core::OdomEpoch{1U}, core::CalibrationEpoch{2U}));
  ASSERT_FALSE(wrong_calibration);
  EXPECT_EQ(wrong_calibration.error().code, FinalizedLidarTargetMapErrorCode::InvalidCloud);
  expectSemanticSnapshot(map, before);

  auto invalid_covariance_sweep = finalizedSweep(second_cloud, 2U, 401U);
  invalid_covariance_sweep.finalized_state.pose_covariance.matrix(0, 0) = -1.0;
  const auto invalid_covariance = map.insertFinalizedSweep(std::move(invalid_covariance_sweep));
  ASSERT_FALSE(invalid_covariance);
  EXPECT_EQ(invalid_covariance.error().code, FinalizedLidarTargetMapErrorCode::InvalidCovariance);
  expectSemanticSnapshot(map, before);
}

TEST(FinalizedLidarTargetMap, ReadViewEnforcesGateTracksVersionAndSurvivesMapMove) {
  auto map = mapWith();
  ASSERT_TRUE(map.insertFinalizedSweep(
      finalizedSweep(indexedCloud({{{0.0, 0.0, 0.0}, 0U}}, 700U), 1U, 700U)));
  const auto view = map.readView();
  EXPECT_EQ(view.odomEpoch(), core::OdomEpoch{1U});
  EXPECT_EQ(view.sensor(), core::SensorInstanceId::lidar(core::LidarId{1U}));
  EXPECT_EQ(view.version(), 1U);
  EXPECT_EQ(view.checksum(), map.statistics().checksum);
  const auto found = view.nearestExact(Eigen::Vector3d::Zero(), 0.25);
  EXPECT_TRUE(found.view_current);
  EXPECT_TRUE(found.query_valid);
  EXPECT_TRUE(found.found);

  const auto unsupported = view.nearestExact(Eigen::Vector3d::Zero(), 0.500001);
  EXPECT_TRUE(unsupported.view_current);
  EXPECT_FALSE(unsupported.query_valid);
  EXPECT_FALSE(unsupported.found);

  auto moved = std::move(map);
  const auto after_move = view.nearestExact(Eigen::Vector3d::Zero(), 0.5);
  EXPECT_TRUE(after_move.view_current);
  EXPECT_TRUE(after_move.found);

  ASSERT_TRUE(moved.insertFinalizedSweep(
      finalizedSweep(indexedCloud({{{2.0, 0.0, 0.0}, 0U}}, 701U), 2U, 701U)));
  const auto stale = view.nearestExact(Eigen::Vector3d::Zero(), 0.5);
  EXPECT_FALSE(stale.view_current);
  EXPECT_FALSE(stale.query_valid);
  EXPECT_FALSE(stale.found);
  const auto current = moved.readView();
  EXPECT_EQ(current.version(), 2U);
  EXPECT_TRUE(current.nearestExact(Eigen::Vector3d{2.0, 0.0, 0.0}, 0.5).found);
}

TEST(FinalizedLidarTargetMap,
     ZeroPointAdmissionAdvancesMonotonicFrontierAndSameFinalRevisionIsAllowed) {
  FinalizedLidarTargetMapConfig config;
  config.odom_epoch = core::OdomEpoch{1U};
  config.maximum_points_per_query_voxel = 1U;
  auto map = mapWith(config);
  const auto first_cloud = indexedCloud({{{0.1, 0.1, 0.1}, 0U}}, 710U);
  ASSERT_TRUE(map.insertFinalizedSweep(finalizedSweep(first_cloud, 1U, 710U, 2U, 5U)));

  const auto second_cloud = indexedCloud({{{0.8, 0.8, 0.8}, 0U}}, 711U);
  const auto zero = map.insertFinalizedSweep(finalizedSweep(second_cloud, 2U, 711U, 2U, 5U));
  ASSERT_TRUE(zero) << zero.error().detail;
  EXPECT_EQ(zero.value().admitted_points, 0U);
  EXPECT_EQ(zero.value().retained_points, 1U);
  EXPECT_EQ(zero.value().version, 2U);
  const MapSemanticSnapshot after_zero = semanticSnapshot(map);

  const auto replay = map.insertFinalizedSweep(finalizedSweep(second_cloud, 2U, 711U, 2U, 5U));
  ASSERT_FALSE(replay);
  EXPECT_EQ(replay.error().code, FinalizedLidarTargetMapErrorCode::DuplicateSweep);
  expectSemanticSnapshot(map, after_zero);

  const auto third_cloud = indexedCloud({{{0.7, 0.7, 0.7}, 0U}}, 712U);
  const auto stale_state = map.insertFinalizedSweep(finalizedSweep(third_cloud, 2U, 712U, 2U, 5U));
  ASSERT_FALSE(stale_state);
  EXPECT_EQ(stale_state.error().code, FinalizedLidarTargetMapErrorCode::StaleFrontier);
  expectSemanticSnapshot(map, after_zero);

  const auto regressed_recovery =
      map.insertFinalizedSweep(finalizedSweep(third_cloud, 3U, 712U, 1U, 5U));
  ASSERT_FALSE(regressed_recovery);
  EXPECT_EQ(regressed_recovery.error().code, FinalizedLidarTargetMapErrorCode::StaleFrontier);
  expectSemanticSnapshot(map, after_zero);

  const auto regressed_revision =
      map.insertFinalizedSweep(finalizedSweep(third_cloud, 3U, 712U, 2U, 4U));
  ASSERT_FALSE(regressed_revision);
  EXPECT_EQ(regressed_revision.error().code, FinalizedLidarTargetMapErrorCode::StaleFrontier);
  expectSemanticSnapshot(map, after_zero);

  const auto same_revision =
      map.insertFinalizedSweep(finalizedSweep(third_cloud, 3U, 712U, 2U, 5U));
  ASSERT_TRUE(same_revision) << same_revision.error().detail;
  EXPECT_EQ(same_revision.value().admitted_points, 0U);
  EXPECT_EQ(same_revision.value().version, 3U);
}

TEST(FinalizedLidarTargetMap, RejectsIneligibleTamperedOrIncompleteAcceptedMetadataAtomically) {
  auto map = mapWith();
  const MapSemanticSnapshot empty = semanticSnapshot(map);

  const auto cloud = indexedCloud({{{0.0, 0.0, 0.0}, 0U}}, 720U);
  auto ineligible = finalizedSweep(cloud, 1U, 720U);
  ineligible.accepted_batch_metadata.map_eligible = false;
  const auto denied = map.insertFinalizedSweep(std::move(ineligible));
  ASSERT_FALSE(denied);
  EXPECT_EQ(denied.error().code, FinalizedLidarTargetMapErrorCode::MapIneligible);
  expectSemanticSnapshot(map, empty);

  auto tampered = finalizedSweep(cloud, 1U, 720U);
  tampered.accepted_batch_metadata.lineage.checksum.front() ^= 0x1U;
  const auto bad_checksum = map.insertFinalizedSweep(std::move(tampered));
  ASSERT_FALSE(bad_checksum);
  EXPECT_EQ(bad_checksum.error().code, FinalizedLidarTargetMapErrorCode::InvalidMetadata);
  expectSemanticSnapshot(map, empty);

  const auto imu_cloud =
      indexedCloud({{{0.0, 0.0, 0.0}, 0U}}, 721U, -1, {core::MeasurementId{99U}});
  auto missing_imu = finalizedSweep(imu_cloud, 1U, 721U);
  auto& lineage = missing_imu.accepted_batch_metadata.lineage;
  lineage.usage.erase(std::remove_if(lineage.usage.begin(), lineage.usage.end(),
                                     [](const auto& usage) {
                                       const auto* id =
                                           std::get_if<core::MeasurementId>(&usage.slice.root);
                                       return id != nullptr && *id == core::MeasurementId{99U};
                                     }),
                      lineage.usage.end());
  lineage.checksum = {};
  const auto recomputed = recomputeAcceptedLidarLineageChecksum(lineage);
  ASSERT_TRUE(recomputed);
  lineage.checksum = recomputed.value();
  ASSERT_EQ(core::validateFactorBatchMetadata(missing_imu.accepted_batch_metadata),
            core::FactorBatchMetadataValidationError::None);
  const auto incomplete = map.insertFinalizedSweep(std::move(missing_imu));
  ASSERT_FALSE(incomplete);
  EXPECT_EQ(incomplete.error().code, FinalizedLidarTargetMapErrorCode::InvalidCloud);
  expectSemanticSnapshot(map, empty);
}

TEST(FinalizedLidarTargetMap, RadiusPruningIsExplicitDeterministicAndRetainsIdentityHistory) {
  FinalizedLidarTargetMapConfig config;
  config.odom_epoch = core::OdomEpoch{1U};
  config.maximum_radius_m = 2.0;
  const std::vector<IndexedPoint> points{
      {{-3.0, 0.0, 0.0}, 3U}, {{0.0, 0.0, 0.0}, 0U}, {{3.0, 0.0, 0.0}, 7U}};
  std::vector<IndexedPoint> reversed = points;
  std::reverse(reversed.begin(), reversed.end());
  auto map_a = mapWith(config);
  auto map_b = mapWith(config);
  const auto cloud_a = indexedCloud(points, 500U);
  const auto cloud_b = indexedCloud(reversed, 500U);
  ASSERT_TRUE(map_a.insertFinalizedSweep(finalizedSweep(cloud_a, 5U, 500U)));
  ASSERT_TRUE(map_b.insertFinalizedSweep(finalizedSweep(cloud_b, 5U, 500U)));

  const auto pruned_a = map_a.pruneAround(Eigen::Vector3d::Zero());
  const auto pruned_b = map_b.pruneAround(Eigen::Vector3d::Zero());
  ASSERT_TRUE(pruned_a) << pruned_a.error().detail;
  ASSERT_TRUE(pruned_b) << pruned_b.error().detail;
  EXPECT_EQ(pruned_a.value().examined_points, 3U);
  EXPECT_EQ(pruned_a.value().removed_points, 2U);
  EXPECT_EQ(pruned_a.value().retained_points, 1U);
  EXPECT_EQ(pruned_a.value().checksum, pruned_b.value().checksum);
  EXPECT_TRUE(map_a.nearestExact(Eigen::Vector3d::Zero(), 0.5).found);
  EXPECT_FALSE(map_a.nearestExact(Eigen::Vector3d{3.0, 0.0, 0.0}, 0.5).found);

  const MapSemanticSnapshot before_noop = semanticSnapshot(map_a);
  const auto no_op = map_a.pruneAround(Eigen::Vector3d::Zero());
  ASSERT_TRUE(no_op);
  EXPECT_EQ(no_op.value().removed_points, 0U);
  expectSemanticSnapshot(map_a, before_noop);

  const auto replay = map_a.insertFinalizedSweep(finalizedSweep(cloud_a, 6U, 501U));
  ASSERT_FALSE(replay);
  EXPECT_EQ(replay.error().code, FinalizedLidarTargetMapErrorCode::DuplicateSweep);
  expectSemanticSnapshot(map_a, before_noop);
}

TEST(FinalizedLidarTargetMap, HardPointCapacityFailureRejectsTheWholeSweep) {
  FinalizedLidarTargetMapConfig config;
  config.odom_epoch = core::OdomEpoch{1U};
  config.hard_point_capacity = 2U;
  auto map = mapWith(config);
  ASSERT_TRUE(map.insertFinalizedSweep(
      finalizedSweep(indexedCloud({{{0.0, 0.0, 0.0}, 0U}}, 600U), 1U, 600U)));
  const MapSemanticSnapshot before = semanticSnapshot(map);

  const auto rejected = map.insertFinalizedSweep(
      finalizedSweep(indexedCloud({{{2.0, 0.0, 0.0}, 0U}, {{4.0, 0.0, 0.0}, 1U}}, 601U), 2U, 601U));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, FinalizedLidarTargetMapErrorCode::PointCapacity);
  expectSemanticSnapshot(map, before);
  EXPECT_FALSE(map.nearestExact(Eigen::Vector3d{2.0, 0.0, 0.0}, 0.5).found);

  const auto later_valid = map.insertFinalizedSweep(
      finalizedSweep(indexedCloud({{{2.0, 0.0, 0.0}, 0U}}, 601U), 2U, 601U));
  ASSERT_TRUE(later_valid) << later_valid.error().detail;
  EXPECT_EQ(later_valid.value().admitted_points, 1U);
  EXPECT_EQ(map.statistics().retained_points, 2U);
}

}  // namespace
}  // namespace meridian::local
