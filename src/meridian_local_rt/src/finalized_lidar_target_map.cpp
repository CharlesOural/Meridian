#include "meridian/local/finalized_lidar_target_map.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "meridian/core/canonical_bytes.hpp"

namespace meridian::local {
namespace {

inline constexpr std::string_view kMapCreationChecksumDomain{
    "meridian.local.finalized_lidar_target_map.creation"};
inline constexpr std::uint32_t kMapCreationChecksumSchemaVersion{1U};
inline constexpr std::string_view kMapInsertChecksumDomain{
    "meridian.local.finalized_lidar_target_map.insert"};
inline constexpr std::uint32_t kMapInsertChecksumSchemaVersion{2U};
inline constexpr std::string_view kMapPruneChecksumDomain{
    "meridian.local.finalized_lidar_target_map.prune"};
inline constexpr std::uint32_t kMapPruneChecksumSchemaVersion{2U};
inline constexpr std::string_view kFinalPoseCovarianceChecksumDomain{
    "meridian.local.finalized_lidar_target_map.final_pose_covariance"};
inline constexpr std::uint32_t kFinalPoseCovarianceChecksumSchemaVersion{1U};
inline constexpr std::string_view kAcceptedBatchMetadataChecksumDomain{
    "meridian.local.finalized_lidar_target_map.accepted_batch_metadata"};
inline constexpr std::uint32_t kAcceptedBatchMetadataChecksumSchemaVersion{1U};

struct VoxelKey {
  std::int64_t x{};
  std::int64_t y{};
  std::int64_t z{};

  auto operator<=>(const VoxelKey&) const = default;
};

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
  value ^= value >> 30U;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31U;
  return value;
}

struct VoxelKeyHash {
  [[nodiscard]] std::size_t operator()(const VoxelKey& key) const noexcept {
    const auto x = mix64(static_cast<std::uint64_t>(key.x));
    const auto y = mix64(static_cast<std::uint64_t>(key.y) + 0x9e3779b97f4a7c15ULL);
    const auto z = mix64(static_cast<std::uint64_t>(key.z) + 0x3c6ef372fe94f82aULL);
    return static_cast<std::size_t>(mix64(x ^ (y << 1U) ^ (z << 7U)));
  }
};

[[nodiscard]] FinalizedLidarTargetMapError mapError(FinalizedLidarTargetMapErrorCode code,
                                                    std::string detail) {
  return FinalizedLidarTargetMapError{code, std::move(detail)};
}

[[nodiscard]] bool finitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool finiteSquare(double value) noexcept {
  return finitePositive(value) && value <= std::sqrt(std::numeric_limits<double>::max());
}

[[nodiscard]] std::optional<VoxelKey> checkedVoxelKey(const Eigen::Vector3d& point,
                                                      double voxel_size_m) noexcept {
  if (!point.allFinite() || !finitePositive(voxel_size_m)) {
    return std::nullopt;
  }
  std::array<std::int64_t, 3U> coordinates{};
  for (Eigen::Index axis = 0; axis < 3; ++axis) {
    const long double scaled =
        static_cast<long double>(point(axis)) / static_cast<long double>(voxel_size_m);
    const long double floored = std::floor(scaled);
    if (!std::isfinite(floored) ||
        floored < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
        floored > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
      return std::nullopt;
    }
    coordinates[static_cast<std::size_t>(axis)] = static_cast<std::int64_t>(floored);
  }
  return VoxelKey{coordinates[0U], coordinates[1U], coordinates[2U]};
}

[[nodiscard]] std::optional<Eigen::Vector3d> voxelCenter(const VoxelKey& key,
                                                         double voxel_size_m) noexcept {
  Eigen::Vector3d center;
  const std::array<std::int64_t, 3U> coordinates{key.x, key.y, key.z};
  for (std::size_t axis = 0U; axis < coordinates.size(); ++axis) {
    const long double value = (static_cast<long double>(coordinates[axis]) + 0.5L) *
                              static_cast<long double>(voxel_size_m);
    const double converted = static_cast<double>(value);
    if (!std::isfinite(converted)) {
      return std::nullopt;
    }
    center(static_cast<Eigen::Index>(axis)) = converted;
  }
  return center;
}

[[nodiscard]] bool canOffset(std::int64_t coordinate, std::int64_t offset) noexcept {
  if (offset < 0) {
    return coordinate >= std::numeric_limits<std::int64_t>::min() - offset;
  }
  if (offset > 0) {
    return coordinate <= std::numeric_limits<std::int64_t>::max() - offset;
  }
  return true;
}

[[nodiscard]] bool sourceCalibrationMatches(const FinalizedLidarSweep& sweep) noexcept {
  bool found = false;
  for (const core::ObservationUsage& usage : sweep.cloud->lineage.usage) {
    const auto* source = std::get_if<core::MeasurementId>(&usage.slice.root);
    if (source == nullptr || *source != sweep.cloud->source_sweep ||
        (usage.role != core::ObservationRole::PrimaryResidual &&
         usage.role != core::ObservationRole::DerivedSummary)) {
      continue;
    }
    found = true;
    if (usage.slice.calibration != sweep.calibration) {
      return false;
    }
  }
  return found;
}

[[nodiscard]] bool validAdmissionKind(MapAdmissionBatchKind kind) noexcept {
  switch (kind) {
    case MapAdmissionBatchKind::Regular:
    case MapAdmissionBatchKind::InitializationSeed:
      return true;
  }
  return false;
}

[[nodiscard]] bool metadataReferencesSourceCloud(const core::FactorBatchMetadata& metadata,
                                                 const LidarRegistrationCloud& cloud,
                                                 MapAdmissionBatchKind kind) noexcept {
  return std::any_of(metadata.lineage.usage.begin(), metadata.lineage.usage.end(),
                     [&](const core::ObservationUsage& usage) {
                       const auto* source = std::get_if<core::MeasurementId>(&usage.slice.root);
                       if (source == nullptr || *source != cloud.source_sweep) {
                         return false;
                       }
                       return kind == MapAdmissionBatchKind::Regular
                                  ? usage.role == core::ObservationRole::PrimaryResidual
                                  : (usage.role == core::ObservationRole::PrimaryResidual ||
                                     usage.role == core::ObservationRole::DerivedSummary);
                     });
}

[[nodiscard]] bool metadataConditionsOnImuSupport(
    const core::FactorBatchMetadata& metadata,
    std::span<const core::MeasurementId> imu_support) noexcept {
  return std::all_of(imu_support.begin(), imu_support.end(), [&](core::MeasurementId imu) {
    std::size_t count = 0U;
    for (const core::ObservationUsage& usage : metadata.lineage.usage) {
      const auto* source = std::get_if<core::MeasurementId>(&usage.slice.root);
      if (source == nullptr || *source != imu) {
        continue;
      }
      if (usage.role != core::ObservationRole::ConditioningOnly) {
        return false;
      }
      ++count;
    }
    return count == 1U;
  });
}

[[nodiscard]] bool validPoseCovariance(const core::PoseCovariance& covariance) {
  if (covariance.tangent != core::PoseTangentConvention::RightTranslationFirst ||
      !covariance.matrix.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, covariance.matrix.cwiseAbs().maxCoeff());
  const double symmetry_error =
      (covariance.matrix - covariance.matrix.transpose()).cwiseAbs().maxCoeff();
  if (symmetry_error > 1.0e-10 * scale) {
    return false;
  }
  const Eigen::SelfAdjointEigenSolver<core::Matrix6d> eigen_solver(covariance.matrix);
  return eigen_solver.info() == Eigen::Success &&
         eigen_solver.eigenvalues().minCoeff() >= -1.0e-10 * scale;
}

[[nodiscard]] bool provenanceLess(const FinalizedLidarTargetPoint& lhs,
                                  const FinalizedLidarTargetPoint& rhs) noexcept {
  return std::tie(lhs.owner->finalized_state.state, lhs.owner->sweep, lhs.source_index) <
         std::tie(rhs.owner->finalized_state.state, rhs.owner->sweep, rhs.source_index);
}

[[nodiscard]] bool canonicalPointLess(const FinalizedLidarTargetPoint& lhs,
                                      const FinalizedLidarTargetPoint& rhs) noexcept {
  for (Eigen::Index axis = 0; axis < 3; ++axis) {
    if (lhs.point_odom(axis) < rhs.point_odom(axis)) {
      return true;
    }
    if (rhs.point_odom(axis) < lhs.point_odom(axis)) {
      return false;
    }
  }
  return std::tie(
             lhs.owner->finalized_state.odom_epoch, lhs.owner->finalized_state.state,
             lhs.owner->finalized_state.exact_time, lhs.owner->sweep, lhs.owner->batch.batch_id,
             lhs.owner->admission.health.recovery_epoch, lhs.source_index,
             lhs.owner->cloud_checksum, lhs.owner->calibration,
             lhs.owner->finalized_state.final_revision, lhs.owner->final_pose_covariance_checksum) <
         std::tie(
             rhs.owner->finalized_state.odom_epoch, rhs.owner->finalized_state.state,
             rhs.owner->finalized_state.exact_time, rhs.owner->sweep, rhs.owner->batch.batch_id,
             rhs.owner->admission.health.recovery_epoch, rhs.source_index,
             rhs.owner->cloud_checksum, rhs.owner->calibration,
             rhs.owner->finalized_state.final_revision, rhs.owner->final_pose_covariance_checksum);
}

[[nodiscard]] bool writeOwner(core::CanonicalEncoder& encoder,
                              const FinalizedLidarTargetOwner& owner) {
  const auto ok = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  return ok(encoder.writeU8(static_cast<std::uint8_t>(owner.batch.sensor.modality))) &&
         ok(encoder.writeU64(owner.batch.sensor.instance)) &&
         ok(encoder.writeU64(owner.batch.batch_id.value())) &&
         ok(encoder.writeU32(owner.admission.header.schema_version)) &&
         ok(encoder.writeU64(owner.admission.header.trace.value())) &&
         ok(encoder.writeU64(owner.admission.header.producer.value())) &&
         ok(encoder.writeU64(owner.admission.header.session.value())) &&
         ok(encoder.writeI64(owner.admission.header.created_at.nanoseconds)) &&
         ok(encoder.writeU64(owner.admission.header.config.value())) &&
         ok(encoder.writeU64(owner.admission.odom_epoch.value())) &&
         ok(encoder.writeI64(owner.admission.reference_time.nanoseconds)) &&
         ok(encoder.writeU8(static_cast<std::uint8_t>(owner.admission.health.state))) &&
         ok(encoder.writeU64(owner.admission.health.recovery_epoch.value())) &&
         ok(encoder.writeU64(owner.admission.health.transition_sequence)) &&
         ok(encoder.writeI64(owner.admission.health.assessed_at.nanoseconds)) &&
         ok(encoder.writeBool(owner.admission.map_eligible)) &&
         ok(encoder.writeU64(owner.admission.accepted_lineage.value())) &&
         ok(encoder.writeHash(owner.admission.accepted_lineage_checksum)) &&
         ok(encoder.writeHash(owner.admission.accepted_batch_metadata_checksum)) &&
         ok(encoder.writeU64(owner.admission_revision.value())) &&
         ok(encoder.writeU8(static_cast<std::uint8_t>(owner.admission_kind))) &&
         ok(encoder.writeU64(owner.finalized_state.state.value())) &&
         ok(encoder.writeI64(owner.finalized_state.exact_time.nanoseconds)) &&
         ok(encoder.writeU64(owner.finalized_state.odom_epoch.value())) &&
         ok(encoder.writeU64(owner.finalized_state.final_revision.value())) &&
         ok(encoder.writePose3(owner.finalized_state.final_estimate.T_odom_imu)) &&
         ok(encoder.writeEigenVector(owner.finalized_state.final_estimate.velocity_odom)) &&
         ok(encoder.writeEigenVector(owner.finalized_state.final_estimate.gyro_bias)) &&
         ok(encoder.writeEigenVector(owner.finalized_state.final_estimate.accel_bias)) &&
         ok(encoder.writeU64(owner.sweep.value())) && ok(encoder.writeHash(owner.cloud_checksum)) &&
         ok(encoder.writeU64(owner.calibration.value())) &&
         ok(encoder.writeHash(owner.cloud_lineage.checksum)) &&
         ok(encoder.writeHash(owner.final_pose_covariance_checksum)) &&
         ok(encoder.writeU64(static_cast<std::uint64_t>(owner.imu_support.size()))) &&
         std::all_of(owner.imu_support.begin(), owner.imu_support.end(),
                     [&](const auto& id) { return ok(encoder.writeU64(id.value())); });
}

[[nodiscard]] bool writePoint(core::CanonicalEncoder& encoder,
                              const FinalizedLidarTargetPoint& point) {
  const auto ok = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  return point.owner && ok(encoder.writeEigenVector(point.point_odom)) &&
         ok(encoder.writeU32(point.source_index)) && writeOwner(encoder, *point.owner);
}

[[nodiscard]] core::Result<core::ContentHash, FinalizedLidarTargetMapError> poseCovarianceChecksum(
    const core::PoseCovariance& covariance) {
  using Result = core::Result<core::ContentHash, FinalizedLidarTargetMapError>;
  auto encoder = core::CanonicalEncoder::create(kFinalPoseCovarianceChecksumDomain,
                                                kFinalPoseCovarianceChecksumSchemaVersion, 2048U);
  if (!encoder) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to initialize final-pose covariance checksum"));
  }
  const auto ok = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!ok(encoder.value().writeU8(static_cast<std::uint8_t>(covariance.tangent))) ||
      !ok(encoder.value().writeEigenMatrix(covariance.matrix))) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to encode final-pose covariance"));
  }
  auto encoded = encoder.value().finish();
  if (!encoded || !core::contentHashPresent(encoded.value().digest())) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to finalize final-pose covariance checksum"));
  }
  return Result::success(encoded.value().digest());
}

[[nodiscard]] bool metadataLineageChecksumMatches(const core::FactorBatchMetadata& metadata) {
  const auto recomputed = recomputeAcceptedLidarLineageChecksum(metadata.lineage);
  return recomputed && recomputed.value() == metadata.lineage.checksum;
}

[[nodiscard]] core::Result<core::ContentHash, FinalizedLidarTargetMapError>
acceptedBatchMetadataChecksum(const core::FactorBatchMetadata& metadata) {
  using Result = core::Result<core::ContentHash, FinalizedLidarTargetMapError>;
  constexpr std::uint64_t kFixedBytes = 8192U;
  constexpr std::uint64_t kBytesPerTimestamp = 16U;
  constexpr std::uint64_t kBytesPerObservability = 2048U;
  std::uint64_t maximum_bytes = kFixedBytes;
  const auto append = [&maximum_bytes](std::size_t count, std::uint64_t bytes_per_row) {
    if (count > static_cast<std::size_t>(
                    (std::numeric_limits<std::uint64_t>::max() - maximum_bytes) / bytes_per_row)) {
      return false;
    }
    maximum_bytes += static_cast<std::uint64_t>(count) * bytes_per_row;
    return true;
  };
  if (!append(metadata.timing.measurement_timestamps.size(), kBytesPerTimestamp) ||
      !append(metadata.directional_observability.size(), kBytesPerObservability)) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "accepted FactorBatch metadata is too large to checksum"));
  }
  auto encoder =
      core::CanonicalEncoder::create(kAcceptedBatchMetadataChecksumDomain,
                                     kAcceptedBatchMetadataChecksumSchemaVersion, maximum_bytes);
  if (!encoder) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to initialize accepted FactorBatch checksum"));
  }
  const auto ok = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  const core::RecordHeader& header = metadata.header;
  const core::FactorBatchTiming& timing = metadata.timing;
  const core::SensorHealthSnapshot& health = metadata.health;
  if (!ok(encoder.value().writeU32(header.schema_version)) ||
      !ok(encoder.value().writeU64(header.trace.value())) ||
      !ok(encoder.value().writeU64(header.producer.value())) ||
      !ok(encoder.value().writeU64(header.session.value())) ||
      !ok(encoder.value().writeI64(header.created_at.nanoseconds)) ||
      !ok(encoder.value().writeU64(header.config.value())) ||
      !ok(encoder.value().writeOptionalMarker(header.direct_calibration.has_value())) ||
      (header.direct_calibration &&
       !ok(encoder.value().writeU64(header.direct_calibration->value()))) ||
      !ok(encoder.value().writeU64(metadata.batch_id.value())) ||
      !ok(encoder.value().writeU64(metadata.odom_epoch.value())) ||
      !ok(encoder.value().writeU8(static_cast<std::uint8_t>(metadata.sensor.modality))) ||
      !ok(encoder.value().writeU64(metadata.sensor.instance)) ||
      !ok(encoder.value().writeI64(timing.support.start.nanoseconds)) ||
      !ok(encoder.value().writeI64(timing.support.end.nanoseconds)) ||
      !ok(encoder.value().writeU64(
          static_cast<std::uint64_t>(timing.measurement_timestamps.size())))) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to encode accepted FactorBatch header"));
  }
  for (core::FusionTime timestamp : timing.measurement_timestamps) {
    if (!ok(encoder.value().writeI64(timestamp.nanoseconds))) {
      return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                      "failed to encode FactorBatch measurement timestamps"));
    }
  }
  if (!ok(encoder.value().writeI64(timing.reference_time.nanoseconds)) ||
      !ok(encoder.value().writeI64(timing.produced_at.nanoseconds)) ||
      !ok(encoder.value().writeU8(static_cast<std::uint8_t>(health.sensor.modality))) ||
      !ok(encoder.value().writeU64(health.sensor.instance)) ||
      !ok(encoder.value().writeU8(static_cast<std::uint8_t>(health.state))) ||
      !ok(encoder.value().writeU64(health.recovery_epoch.value())) ||
      !ok(encoder.value().writeU64(health.transition_sequence)) ||
      !ok(encoder.value().writeI64(health.assessed_at.nanoseconds)) ||
      !ok(encoder.value().writeBool(metadata.map_eligible)) ||
      !ok(encoder.value().writeU64(
          static_cast<std::uint64_t>(metadata.directional_observability.size())))) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to encode FactorBatch timing/health"));
  }
  for (const core::DirectionalObservability& observability : metadata.directional_observability) {
    if (!ok(encoder.value().writeU8(static_cast<std::uint8_t>(observability.tangent))) ||
        !ok(encoder.value().writeEigenMatrix(observability.basis)) ||
        !ok(encoder.value().writeEigenVector(observability.eigenvalues)) ||
        !ok(encoder.value().writeU32(observability.rank)) ||
        !ok(encoder.value().writeDouble(observability.absolute_eigenvalue_threshold)) ||
        !ok(encoder.value().writeDouble(observability.relative_eigenvalue_threshold)) ||
        !ok(encoder.value().writeU64(
            static_cast<std::uint64_t>(observability.supported_variables.size())))) {
      return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                      "failed to encode FactorBatch observability"));
    }
    for (core::DirectionalVariable variable : observability.supported_variables) {
      if (!ok(encoder.value().writeU8(static_cast<std::uint8_t>(variable)))) {
        return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                        "failed to encode supported FactorBatch variable"));
      }
    }
    if (!ok(encoder.value().writeU64(static_cast<std::uint64_t>(observability.endpoints.size())))) {
      return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                      "failed to encode FactorBatch endpoint count"));
    }
    for (const core::DirectionalObservabilityEndpoint& endpoint : observability.endpoints) {
      if (!ok(encoder.value().writeU8(static_cast<std::uint8_t>(endpoint.role))) ||
          !ok(encoder.value().writeU64(endpoint.state.value())) ||
          !ok(encoder.value().writeI64(endpoint.exact_time.nanoseconds))) {
        return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                        "failed to encode FactorBatch endpoint"));
      }
    }
  }
  if (!ok(encoder.value().writeU64(metadata.lineage.id.value())) ||
      !ok(encoder.value().writeHash(metadata.lineage.checksum))) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to encode FactorBatch lineage dependency"));
  }
  auto encoded = encoder.value().finish();
  if (!encoded || !core::contentHashPresent(encoded.value().digest())) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to finalize accepted FactorBatch checksum"));
  }
  return Result::success(encoded.value().digest());
}

[[nodiscard]] core::Result<core::ContentHash, FinalizedLidarTargetMapError> creationChecksum(
    const FinalizedLidarTargetMapConfig& config) {
  using Result = core::Result<core::ContentHash, FinalizedLidarTargetMapError>;
  auto encoder = core::CanonicalEncoder::create(kMapCreationChecksumDomain,
                                                kMapCreationChecksumSchemaVersion, 2048U);
  if (!encoder) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to initialize finalized-map creation checksum"));
  }
  const auto ok = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!ok(encoder.value().writeU64(config.odom_epoch.value())) ||
      !ok(encoder.value().writeU8(static_cast<std::uint8_t>(config.sensor.modality))) ||
      !ok(encoder.value().writeU64(config.sensor.instance)) ||
      !ok(encoder.value().writeDouble(config.query_voxel_size_m)) ||
      !ok(encoder.value().writeDouble(config.insertion_voxel_size_m)) ||
      !ok(encoder.value().writeU64(
          static_cast<std::uint64_t>(config.maximum_points_per_query_voxel))) ||
      !ok(encoder.value().writeDouble(config.minimum_point_separation_m)) ||
      !ok(encoder.value().writeDouble(config.maximum_supported_query_distance_m)) ||
      !ok(encoder.value().writeDouble(config.maximum_radius_m)) ||
      !ok(encoder.value().writeU64(static_cast<std::uint64_t>(config.hard_point_capacity)))) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to encode finalized-map configuration"));
  }
  auto encoded = encoder.value().finish();
  if (!encoded || !core::contentHashPresent(encoded.value().digest())) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to finalize finalized-map creation checksum"));
  }
  return Result::success(encoded.value().digest());
}

[[nodiscard]] bool capacityForRows(std::size_t rows, std::uint64_t bytes_per_row,
                                   std::uint64_t* maximum_bytes) noexcept {
  if (rows > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
    return false;
  }
  const auto count = static_cast<std::uint64_t>(rows);
  if (count > (std::numeric_limits<std::uint64_t>::max() - *maximum_bytes) / bytes_per_row) {
    return false;
  }
  *maximum_bytes += count * bytes_per_row;
  return true;
}

[[nodiscard]] core::Result<core::ContentHash, FinalizedLidarTargetMapError> insertChecksum(
    const core::ContentHash& previous, std::uint64_t next_version,
    const FinalizedLidarTargetOwner& owner, std::span<const FinalizedLidarTargetPoint> admitted) {
  using Result = core::Result<core::ContentHash, FinalizedLidarTargetMapError>;
  std::uint64_t maximum_bytes = 4096U;
  if (!capacityForRows(admitted.size(), 256U, &maximum_bytes)) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "finalized-map insert transaction is too large to checksum"));
  }
  auto encoder = core::CanonicalEncoder::create(kMapInsertChecksumDomain,
                                                kMapInsertChecksumSchemaVersion, maximum_bytes);
  if (!encoder) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to initialize finalized-map insert checksum"));
  }
  const auto ok = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!ok(encoder.value().writeHash(previous)) || !ok(encoder.value().writeU64(next_version)) ||
      !writeOwner(encoder.value(), owner) ||
      !ok(encoder.value().writeU64(static_cast<std::uint64_t>(admitted.size())))) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to encode finalized-map insert header"));
  }
  for (const FinalizedLidarTargetPoint& point : admitted) {
    if (!ok(encoder.value().writeEigenVector(point.point_odom)) ||
        !ok(encoder.value().writeU32(point.source_index))) {
      return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                      "failed to encode finalized-map admitted point"));
    }
  }
  auto encoded = encoder.value().finish();
  if (!encoded || !core::contentHashPresent(encoded.value().digest())) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to finalize finalized-map insert checksum"));
  }
  return Result::success(encoded.value().digest());
}

[[nodiscard]] core::Result<core::ContentHash, FinalizedLidarTargetMapError> pruneChecksum(
    const core::ContentHash& previous, std::uint64_t next_version, const Eigen::Vector3d& origin,
    double radius, std::span<const FinalizedLidarTargetPoint> removed) {
  using Result = core::Result<core::ContentHash, FinalizedLidarTargetMapError>;
  std::uint64_t maximum_bytes = 2048U;
  if (!capacityForRows(removed.size(), 1024U, &maximum_bytes)) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "finalized-map prune transaction is too large to checksum"));
  }
  auto encoder = core::CanonicalEncoder::create(kMapPruneChecksumDomain,
                                                kMapPruneChecksumSchemaVersion, maximum_bytes);
  if (!encoder) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to initialize finalized-map prune checksum"));
  }
  const auto ok = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!ok(encoder.value().writeHash(previous)) || !ok(encoder.value().writeU64(next_version)) ||
      !ok(encoder.value().writeEigenVector(origin)) || !ok(encoder.value().writeDouble(radius)) ||
      !ok(encoder.value().writeU64(static_cast<std::uint64_t>(removed.size())))) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to encode finalized-map prune header"));
  }
  for (const FinalizedLidarTargetPoint& point : removed) {
    if (!writePoint(encoder.value(), point)) {
      return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                      "failed to encode finalized-map removed point"));
    }
  }
  auto encoded = encoder.value().finish();
  if (!encoded || !core::contentHashPresent(encoded.value().digest())) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                                    "failed to finalize finalized-map prune checksum"));
  }
  return Result::success(encoded.value().digest());
}

}  // namespace

core::Result<core::ContentHash, AcceptedLidarLineageChecksumError>
recomputeAcceptedLidarLineageChecksum(const core::ObservationLineage& lineage) {
  using Result =
      core::Result<core::ContentHash, AcceptedLidarLineageChecksumError>;
  const auto fail = [](AcceptedLidarLineageChecksumErrorCode code, std::string detail) {
    return Result::failure(AcceptedLidarLineageChecksumError{code, std::move(detail)});
  };
  if (core::validateLineage(lineage) != core::LineageValidationError::None) {
    return fail(AcceptedLidarLineageChecksumErrorCode::InvalidLineage,
                "accepted LiDAR lineage is not semantically valid");
  }
  constexpr std::uint64_t kFixedBytes = 1'024U;
  constexpr std::uint64_t kBytesPerUsage = 256U;
  constexpr std::uint64_t kBytesPerCorrelation = 128U;
  std::uint64_t maximum_bytes = kFixedBytes;
  const auto append_capacity = [&maximum_bytes](std::size_t count,
                                                 std::uint64_t bytes_per_item) {
    if (count > static_cast<std::size_t>(
                    (std::numeric_limits<std::uint64_t>::max() - maximum_bytes) /
                    bytes_per_item)) {
      return false;
    }
    maximum_bytes += static_cast<std::uint64_t>(count) * bytes_per_item;
    return true;
  };
  if (!append_capacity(lineage.usage.size(), kBytesPerUsage) ||
      !append_capacity(lineage.correlations.size(), kBytesPerCorrelation)) {
    return fail(AcceptedLidarLineageChecksumErrorCode::Capacity,
                "accepted LiDAR lineage checksum capacity overflowed");
  }
  auto encoder = core::CanonicalEncoder::create(
      kAcceptedLidarLineageChecksumDomain, kAcceptedLidarLineageChecksumSchemaVersion,
      maximum_bytes);
  if (!encoder) {
    return fail(AcceptedLidarLineageChecksumErrorCode::Capacity,
                "accepted LiDAR lineage checksum encoder could not be created");
  }
  const auto write = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  bool encoded = write(encoder.value().writeU64(lineage.id.value())) &&
                 write(encoder.value().writeU64(lineage.usage.size()));
  for (const core::ObservationUsage& usage : lineage.usage) {
    const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
    const auto* gnss = std::get_if<core::GnssObservationId>(&usage.slice.root);
    if (measurement == nullptr && gnss == nullptr) {
      return fail(AcceptedLidarLineageChecksumErrorCode::InvalidLineage,
                  "accepted LiDAR lineage contains an unsupported root identity");
    }
    const bool source_checksum_present =
        core::contentHashPresent(usage.slice.source_checksum);
    encoded = encoded && write(encoder.value().writeU8(measurement != nullptr ? 0U : 1U)) &&
              write(encoder.value().writeU64(measurement != nullptr ? measurement->value()
                                                                    : gnss->value())) &&
              write(encoder.value().writeU8(static_cast<std::uint8_t>(usage.slice.kind))) &&
              write(encoder.value().writeU64(usage.slice.begin)) &&
              write(encoder.value().writeU64(usage.slice.end)) &&
              write(encoder.value().writeOptionalMarker(source_checksum_present)) &&
              (!source_checksum_present ||
               write(encoder.value().writeHash(usage.slice.source_checksum))) &&
              write(encoder.value().writeU64(usage.slice.calibration.value())) &&
              write(encoder.value().writeU8(static_cast<std::uint8_t>(usage.role))) &&
              write(encoder.value().writeU64(usage.consumer.value())) &&
              write(encoder.value().writeOptionalMarker(usage.factor_group.has_value()));
    if (usage.factor_group) {
      encoded = encoded && write(encoder.value().writeU64(usage.factor_group->value()));
    }
    encoded = encoded &&
              write(encoder.value().writeOptionalMarker(usage.correlation_group.has_value()));
    if (usage.correlation_group) {
      encoded = encoded && write(encoder.value().writeU64(usage.correlation_group->value()));
    }
  }
  encoded = encoded && write(encoder.value().writeU64(lineage.correlations.size()));
  for (const core::CorrelationDeclaration& declaration : lineage.correlations) {
    encoded =
        encoded && write(encoder.value().writeU64(declaration.group.value())) &&
        write(encoder.value().writeU64(declaration.policy.value())) &&
        write(encoder.value().writeU8(static_cast<std::uint8_t>(declaration.treatment))) &&
        write(encoder.value().writeDouble(declaration.covariance_inflation)) &&
        write(encoder.value().writeOptionalMarker(declaration.total_information_cap.has_value()));
    if (declaration.total_information_cap) {
      encoded = encoded && write(encoder.value().writeDouble(*declaration.total_information_cap));
    }
  }
  if (!encoded) {
    return fail(AcceptedLidarLineageChecksumErrorCode::Encoding,
                "accepted LiDAR lineage checksum encoding failed");
  }
  auto bytes = encoder.value().finish();
  if (!bytes || !core::contentHashPresent(bytes.value().digest())) {
    return fail(AcceptedLidarLineageChecksumErrorCode::Encoding,
                "accepted LiDAR lineage checksum finalization failed");
  }
  return Result::success(bytes.value().digest());
}

struct detail::FinalizedLidarTargetMapImpl {
  struct Block {
    std::vector<FinalizedLidarTargetPoint> points;
  };

  explicit FinalizedLidarTargetMapImpl(FinalizedLidarTargetMapConfig config_in,
                                       core::ContentHash checksum)
      : config(std::move(config_in)) {
    statistics.checksum = checksum;
  }

  FinalizedLidarTargetMapConfig config;
  std::unordered_map<VoxelKey, Block, VoxelKeyHash> blocks;
  // One strictly monotonic frontier is sufficient to reject every replay and
  // stale transaction without retaining mission-long owner metadata.
  std::shared_ptr<const FinalizedLidarTargetOwner> last_owner_frontier;
  FinalizedLidarTargetMapStatistics statistics;
};

FinalizedLidarTargetMap::FinalizedLidarTargetMap(
    std::unique_ptr<detail::FinalizedLidarTargetMapImpl> implementation)
    : implementation_(std::move(implementation)) {}

FinalizedLidarTargetMap::~FinalizedLidarTargetMap() = default;
FinalizedLidarTargetMap::FinalizedLidarTargetMap(FinalizedLidarTargetMap&&) noexcept = default;
FinalizedLidarTargetMap& FinalizedLidarTargetMap::operator=(FinalizedLidarTargetMap&&) noexcept =
    default;

core::Result<FinalizedLidarTargetMap, FinalizedLidarTargetMapError> FinalizedLidarTargetMap::create(
    FinalizedLidarTargetMapConfig config) {
  using Result = core::Result<FinalizedLidarTargetMap, FinalizedLidarTargetMapError>;
  if (!config.odom_epoch.valid() || !config.sensor.valid() ||
      config.sensor.modality != core::SensorModality::Lidar ||
      !finitePositive(config.query_voxel_size_m) ||
      !finitePositive(config.insertion_voxel_size_m) ||
      config.insertion_voxel_size_m > config.query_voxel_size_m ||
      config.maximum_points_per_query_voxel == 0U ||
      !finiteSquare(config.minimum_point_separation_m) ||
      !finiteSquare(config.maximum_supported_query_distance_m) ||
      config.maximum_supported_query_distance_m > config.query_voxel_size_m ||
      !finiteSquare(config.maximum_radius_m) || config.hard_point_capacity == 0U) {
    return Result::failure(
        mapError(FinalizedLidarTargetMapErrorCode::InvalidConfig,
                 "finalized LiDAR map requires a valid epoch/LiDAR sensor, positive bounded "
                 "voxel/query/radius parameters, positive per-voxel/capacity bounds, and "
                 "insertion/query distances no larger than the query voxel"));
  }
  auto checksum = creationChecksum(config);
  if (!checksum) {
    return Result::failure(checksum.error());
  }
  auto implementation =
      std::make_unique<detail::FinalizedLidarTargetMapImpl>(std::move(config), checksum.value());
  return Result::success(FinalizedLidarTargetMap(std::move(implementation)));
}

core::Result<FinalizedLidarTargetInsertStats, FinalizedLidarTargetMapError>
FinalizedLidarTargetMap::insertFinalizedSweep(FinalizedLidarSweep sweep) {
  using Result = core::Result<FinalizedLidarTargetInsertStats, FinalizedLidarTargetMapError>;
  detail::FinalizedLidarTargetMapImpl& map = *implementation_;
  ++map.statistics.insert_attempts;
  const auto reject = [&map](FinalizedLidarTargetMapErrorCode code, std::string detail) {
    ++map.statistics.rejected_insertions;
    return Result::failure(mapError(code, std::move(detail)));
  };

  const core::FactorBatchMetadata& metadata = sweep.accepted_batch_metadata;
  const LocalGraphFinalizedState& finalized = sweep.finalized_state;
  if (metadata.odom_epoch != map.config.odom_epoch ||
      finalized.odom_epoch != map.config.odom_epoch) {
    return reject(FinalizedLidarTargetMapErrorCode::EpochMismatch,
                  "finalized LiDAR sweep belongs to a different odometry epoch");
  }
  if (!sweep.batch.sensor.valid() || !sweep.batch.batch_id.valid() || !finalized.state.valid()) {
    return reject(FinalizedLidarTargetMapErrorCode::InvalidIdentity,
                  "finalized LiDAR sweep has an invalid sensor, batch, or state identity");
  }
  if (sweep.batch.sensor != map.config.sensor || metadata.sensor != map.config.sensor ||
      metadata.health.sensor != map.config.sensor || metadata.batch_id != sweep.batch.batch_id) {
    return reject(FinalizedLidarTargetMapErrorCode::InvalidIdentity,
                  "finalized LiDAR sensor/batch identity does not match map or metadata");
  }
  if (core::validateFactorBatchMetadata(metadata) !=
          core::FactorBatchMetadataValidationError::None ||
      !metadataLineageChecksumMatches(metadata)) {
    return reject(FinalizedLidarTargetMapErrorCode::InvalidMetadata,
                  "finalized LiDAR sweep requires complete canonical accepted FactorBatch "
                  "metadata and lineage");
  }
  if (!metadata.map_eligible || metadata.health.state != core::SensorHealthState::Active) {
    return reject(FinalizedLidarTargetMapErrorCode::MapIneligible,
                  "finalized LiDAR batch was not accepted as active map-eligible evidence");
  }
  if (!validAdmissionKind(sweep.admission_kind) || !sweep.admission_revision.valid() ||
      sweep.admission_revision > finalized.final_revision) {
    return reject(FinalizedLidarTargetMapErrorCode::InvalidAdmission,
                  "finalized LiDAR admission kind/revision is invalid or newer than finality");
  }
  if (!finalized.final_revision.valid()) {
    return reject(FinalizedLidarTargetMapErrorCode::InvalidRevision,
                  "finalized LiDAR sweep requires a valid final graph revision");
  }
  if (!finalized.final_estimate.T_odom_imu.matrix().allFinite() ||
      !finalized.final_estimate.velocity_odom.allFinite() ||
      !finalized.final_estimate.gyro_bias.allFinite() ||
      !finalized.final_estimate.accel_bias.allFinite()) {
    return reject(FinalizedLidarTargetMapErrorCode::InvalidPose,
                  "finalized LiDAR navigation estimate contains a non-finite value");
  }
  if (!validPoseCovariance(finalized.pose_covariance)) {
    return reject(FinalizedLidarTargetMapErrorCode::InvalidCovariance,
                  "finalized LiDAR pose covariance must be finite, symmetric, PSD, and use "
                  "the right-translation-first tangent convention");
  }
  if (!sweep.calibration.valid()) {
    return reject(FinalizedLidarTargetMapErrorCode::InvalidCalibration,
                  "finalized LiDAR sweep requires a valid calibration epoch");
  }
  if (!sweep.cloud || !sweep.cloud->source_sweep.valid() || sweep.cloud->points.empty() ||
      !core::contentHashPresent(sweep.cloud->checksum) ||
      finalized.exact_time != sweep.cloud->reference_time ||
      metadata.timing.reference_time != sweep.cloud->reference_time ||
      metadata.header.direct_calibration != sweep.calibration || !sourceCalibrationMatches(sweep) ||
      !metadataReferencesSourceCloud(metadata, *sweep.cloud, sweep.admission_kind) ||
      !metadataConditionsOnImuSupport(metadata, sweep.cloud->imu_support)) {
    return reject(
        FinalizedLidarTargetMapErrorCode::InvalidCloud,
        "finalized LiDAR cloud time/calibration/source/IMU lineage does not match accepted "
        "metadata and finality");
  }
  if (map.last_owner_frontier) {
    const FinalizedLidarTargetOwner& frontier = *map.last_owner_frontier;
    if (sweep.cloud->source_sweep == frontier.sweep) {
      return reject(FinalizedLidarTargetMapErrorCode::DuplicateSweep,
                    "finalized LiDAR sweep identity was already admitted");
    }
    if (sweep.batch.batch_id == frontier.batch.batch_id) {
      return reject(FinalizedLidarTargetMapErrorCode::DuplicateFactorBatch,
                    "finalized LiDAR FactorBatch identity was already admitted");
    }
    if (!(finalized.state > frontier.finalized_state.state) ||
        !(finalized.exact_time > frontier.finalized_state.exact_time) ||
        !(sweep.cloud->source_sweep > frontier.sweep) ||
        !(sweep.batch.batch_id > frontier.batch.batch_id) ||
        finalized.final_revision < frontier.finalized_state.final_revision ||
        metadata.health.recovery_epoch < frontier.admission.health.recovery_epoch) {
      return reject(FinalizedLidarTargetMapErrorCode::StaleFrontier,
                    "finalized LiDAR owner did not strictly advance state/time/sweep/batch or "
                    "regressed final revision/recovery epoch");
    }
  }
  auto covariance_checksum = poseCovarianceChecksum(finalized.pose_covariance);
  if (!covariance_checksum) {
    ++map.statistics.rejected_insertions;
    return Result::failure(covariance_checksum.error());
  }
  auto metadata_checksum = acceptedBatchMetadataChecksum(metadata);
  if (!metadata_checksum) {
    ++map.statistics.rejected_insertions;
    return Result::failure(metadata_checksum.error());
  }
  core::ObservationLineage owner_cloud_lineage = sweep.cloud->lineage;
  owner_cloud_lineage.checksum = {};
  auto owner_lineage_checksum = recomputeAcceptedLidarLineageChecksum(owner_cloud_lineage);
  if (!owner_lineage_checksum) {
    return reject(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                  "finalized LiDAR owner lineage checksum could not be sealed: " +
                      owner_lineage_checksum.error().detail);
  }
  owner_cloud_lineage.checksum = owner_lineage_checksum.value();
  FinalizedLidarAdmissionReceipt receipt;
  receipt.header = metadata.header;
  receipt.odom_epoch = metadata.odom_epoch;
  receipt.reference_time = metadata.timing.reference_time;
  receipt.health = metadata.health;
  receipt.map_eligible = metadata.map_eligible;
  receipt.accepted_lineage = metadata.lineage.id;
  receipt.accepted_lineage_checksum = metadata.lineage.checksum;
  receipt.accepted_batch_metadata_checksum = metadata_checksum.value();
  const auto owner = std::make_shared<const FinalizedLidarTargetOwner>(FinalizedLidarTargetOwner{
      sweep.batch, std::move(receipt), sweep.admission_revision, sweep.admission_kind,
      std::move(sweep.finalized_state), sweep.cloud->source_sweep, sweep.cloud->checksum,
      sweep.calibration, std::move(owner_cloud_lineage),
      sweep.cloud->imu_support, covariance_checksum.value()});

  struct Candidate {
    FinalizedLidarTargetPoint point;
    VoxelKey insertion_key;
    VoxelKey query_key;
    double insertion_center_distance_squared{};
  };
  std::vector<Candidate> candidates;
  candidates.reserve(sweep.cloud->points.size());
  for (const LidarRegistrationPoint& source : sweep.cloud->points) {
    const Eigen::Vector3d point_odom =
        owner->finalized_state.final_estimate.T_odom_imu * source.point;
    const auto insertion_key = checkedVoxelKey(point_odom, map.config.insertion_voxel_size_m);
    const auto query_key = checkedVoxelKey(point_odom, map.config.query_voxel_size_m);
    if (!insertion_key || !query_key) {
      return reject(FinalizedLidarTargetMapErrorCode::SpatialIndexFailure,
                    "transformed finalized LiDAR point cannot be indexed safely");
    }
    const auto center = voxelCenter(*insertion_key, map.config.insertion_voxel_size_m);
    if (!center) {
      return reject(FinalizedLidarTargetMapErrorCode::SpatialIndexFailure,
                    "finalized LiDAR insertion voxel center is not representable");
    }
    const double center_distance_squared = (point_odom - *center).squaredNorm();
    if (!std::isfinite(center_distance_squared)) {
      return reject(FinalizedLidarTargetMapErrorCode::SpatialIndexFailure,
                    "finalized LiDAR insertion distance is not finite");
    }
    candidates.push_back(
        Candidate{FinalizedLidarTargetPoint{point_odom, source.source_index, owner}, *insertion_key,
                  *query_key, center_distance_squared});
  }

  std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
    return std::tie(lhs.insertion_key, lhs.insertion_center_distance_squared,
                    lhs.point.owner->finalized_state.state, lhs.point.owner->sweep,
                    lhs.point.source_index) <
           std::tie(rhs.insertion_key, rhs.insertion_center_distance_squared,
                    rhs.point.owner->finalized_state.state, rhs.point.owner->sweep,
                    rhs.point.source_index);
  });

  std::vector<Candidate> selected;
  selected.reserve(candidates.size());
  for (std::size_t begin = 0U; begin < candidates.size();) {
    selected.push_back(candidates[begin]);
    std::size_t end = begin + 1U;
    while (end < candidates.size() &&
           candidates[end].insertion_key == candidates[begin].insertion_key) {
      ++end;
    }
    begin = end;
  }
  std::sort(selected.begin(), selected.end(), [](const Candidate& lhs, const Candidate& rhs) {
    return std::tie(lhs.query_key, lhs.insertion_key, lhs.insertion_center_distance_squared,
                    lhs.point.owner->finalized_state.state, lhs.point.owner->sweep,
                    lhs.point.source_index) <
           std::tie(rhs.query_key, rhs.insertion_key, rhs.insertion_center_distance_squared,
                    rhs.point.owner->finalized_state.state, rhs.point.owner->sweep,
                    rhs.point.source_index);
  });

  struct StagedBlock {
    VoxelKey key;
    bool existed{};
    std::size_t original_size{};
    std::vector<FinalizedLidarTargetPoint> points;
  };
  std::vector<StagedBlock> staged;
  staged.reserve(selected.size());
  std::vector<FinalizedLidarTargetPoint> admitted;
  admitted.reserve(selected.size());
  std::size_t separation_discarded = 0U;
  std::size_t capacity_discarded = 0U;
  const double minimum_separation_squared =
      map.config.minimum_point_separation_m * map.config.minimum_point_separation_m;
  for (std::size_t begin = 0U; begin < selected.size();) {
    std::size_t end = begin + 1U;
    while (end < selected.size() && selected[end].query_key == selected[begin].query_key) {
      ++end;
    }
    const auto existing = map.blocks.find(selected[begin].query_key);
    StagedBlock block;
    block.key = selected[begin].query_key;
    block.existed = existing != map.blocks.end();
    if (block.existed) {
      block.points = existing->second.points;
    }
    const std::size_t original_size = block.points.size();
    block.original_size = original_size;
    block.points.reserve(
        std::min(map.config.maximum_points_per_query_voxel, original_size + (end - begin)));
    for (std::size_t index = begin; index < end; ++index) {
      if (block.points.size() >= map.config.maximum_points_per_query_voxel) {
        ++capacity_discarded;
        continue;
      }
      const bool too_close = std::any_of(
          block.points.begin(), block.points.end(), [&](const FinalizedLidarTargetPoint& present) {
            return (present.point_odom - selected[index].point.point_odom).squaredNorm() <
                   minimum_separation_squared;
          });
      if (too_close) {
        ++separation_discarded;
        continue;
      }
      block.points.push_back(selected[index].point);
      admitted.push_back(selected[index].point);
    }
    if (block.points.size() != original_size) {
      staged.push_back(std::move(block));
    }
    begin = end;
  }

  std::size_t staged_point_delta = 0U;
  for (const StagedBlock& block : staged) {
    if (block.points.size() < block.original_size ||
        block.points.size() - block.original_size >
            std::numeric_limits<std::size_t>::max() - staged_point_delta) {
      return reject(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                    "finalized LiDAR staged-point accounting overflowed");
    }
    staged_point_delta += block.points.size() - block.original_size;
  }
  if (staged_point_delta != admitted.size()) {
    return reject(FinalizedLidarTargetMapErrorCode::ChecksumFailure,
                  "finalized LiDAR staged-point delta does not match admitted rows");
  }

  if (admitted.size() > map.config.hard_point_capacity - map.statistics.retained_points) {
    return reject(FinalizedLidarTargetMapErrorCode::PointCapacity,
                  "finalized LiDAR sweep would exceed the hard persistent-point capacity");
  }
  if (map.statistics.version == std::numeric_limits<std::uint64_t>::max()) {
    return reject(FinalizedLidarTargetMapErrorCode::InvalidRevision,
                  "finalized LiDAR map version is exhausted");
  }
  const std::uint64_t next_version = map.statistics.version + 1U;
  auto next_checksum = insertChecksum(map.statistics.checksum, next_version, *owner, admitted);
  if (!next_checksum) {
    ++map.statistics.rejected_insertions;
    return Result::failure(next_checksum.error());
  }

  // Allocate new unordered_map nodes before semantic mutation. Once the main
  // table is reserved, node-handle merge plus vector move-assignment are
  // allocation-free because both the hasher and equality relation are
  // noexcept. Only touched blocks were copied above; no whole-map COW occurs.
  const std::size_t new_block_count = static_cast<std::size_t>(std::count_if(
      staged.begin(), staged.end(), [](const StagedBlock& block) { return !block.existed; }));
  std::unordered_map<VoxelKey, detail::FinalizedLidarTargetMapImpl::Block, VoxelKeyHash> new_blocks;
  new_blocks.reserve(new_block_count);
  for (StagedBlock& block : staged) {
    if (!block.existed) {
      new_blocks.emplace(block.key,
                         detail::FinalizedLidarTargetMapImpl::Block{std::move(block.points)});
    }
  }
  map.blocks.reserve(map.blocks.size() + new_block_count);
  map.blocks.merge(new_blocks);
  for (StagedBlock& block : staged) {
    if (block.existed) {
      map.blocks.find(block.key)->second.points = std::move(block.points);
    }
  }
  map.last_owner_frontier = owner;

  const std::size_t insertion_discarded = candidates.size() - selected.size();
  map.statistics.admitted_sweeps += 1U;
  map.statistics.input_points += candidates.size();
  map.statistics.insertion_voxels += selected.size();
  map.statistics.insertion_selection_discarded_points += insertion_discarded;
  map.statistics.minimum_separation_discarded_points += separation_discarded;
  map.statistics.query_voxel_capacity_discarded_points += capacity_discarded;
  map.statistics.admitted_points += admitted.size();
  map.statistics.retained_points += admitted.size();
  map.statistics.retained_query_voxels = map.blocks.size();
  map.statistics.version = next_version;
  map.statistics.checksum = next_checksum.value();

  FinalizedLidarTargetInsertStats result;
  result.input_points = candidates.size();
  result.insertion_voxels = selected.size();
  result.insertion_selection_discarded_points = insertion_discarded;
  result.minimum_separation_discarded_points = separation_discarded;
  result.query_voxel_capacity_discarded_points = capacity_discarded;
  result.admitted_points = admitted.size();
  result.touched_query_voxels = staged.size();
  result.retained_query_voxels = map.blocks.size();
  result.retained_points = map.statistics.retained_points;
  result.version = next_version;
  result.checksum = next_checksum.value();
  result.owner = owner;
  return Result::success(std::move(result));
}

core::Result<FinalizedLidarTargetPruneStats, FinalizedLidarTargetMapError>
FinalizedLidarTargetMap::pruneAround(const Eigen::Vector3d& origin_odom) {
  using Result = core::Result<FinalizedLidarTargetPruneStats, FinalizedLidarTargetMapError>;
  detail::FinalizedLidarTargetMapImpl& map = *implementation_;
  ++map.statistics.prune_attempts;
  if (!origin_odom.allFinite()) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::InvalidQuery,
                                    "finalized LiDAR prune origin must be finite"));
  }

  FinalizedLidarTargetPruneStats stats;
  stats.examined_query_voxels = map.blocks.size();
  const double radius_squared = map.config.maximum_radius_m * map.config.maximum_radius_m;
  std::vector<FinalizedLidarTargetPoint> removed;
  for (const auto& [key, block] : map.blocks) {
    static_cast<void>(key);
    stats.examined_points += block.points.size();
    for (const FinalizedLidarTargetPoint& point : block.points) {
      if ((point.point_odom - origin_odom).squaredNorm() > radius_squared) {
        removed.push_back(point);
      }
    }
  }
  if (removed.empty()) {
    stats.retained_query_voxels = map.blocks.size();
    stats.retained_points = map.statistics.retained_points;
    stats.version = map.statistics.version;
    stats.checksum = map.statistics.checksum;
    return Result::success(std::move(stats));
  }
  std::sort(removed.begin(), removed.end(), canonicalPointLess);
  if (map.statistics.version == std::numeric_limits<std::uint64_t>::max()) {
    return Result::failure(mapError(FinalizedLidarTargetMapErrorCode::InvalidRevision,
                                    "finalized LiDAR map version is exhausted"));
  }
  const std::uint64_t next_version = map.statistics.version + 1U;
  auto next_checksum = pruneChecksum(map.statistics.checksum, next_version, origin_odom,
                                     map.config.maximum_radius_m, removed);
  if (!next_checksum) {
    return Result::failure(next_checksum.error());
  }

  for (auto block = map.blocks.begin(); block != map.blocks.end();) {
    auto& points = block->second.points;
    points.erase(std::remove_if(points.begin(), points.end(),
                                [&](const auto& point) {
                                  return (point.point_odom - origin_odom).squaredNorm() >
                                         radius_squared;
                                }),
                 points.end());
    if (points.empty()) {
      block = map.blocks.erase(block);
      ++stats.removed_query_voxels;
    } else {
      ++block;
    }
  }
  stats.removed_points = removed.size();
  map.statistics.retained_points -= removed.size();
  map.statistics.retained_query_voxels = map.blocks.size();
  ++map.statistics.prune_transactions;
  map.statistics.pruned_points += removed.size();
  map.statistics.version = next_version;
  map.statistics.checksum = next_checksum.value();
  stats.retained_query_voxels = map.blocks.size();
  stats.retained_points = map.statistics.retained_points;
  stats.version = next_version;
  stats.checksum = next_checksum.value();
  return Result::success(std::move(stats));
}

namespace {

[[nodiscard]] FinalizedLidarTargetNeighbor nearestExactInMap(
    const detail::FinalizedLidarTargetMapImpl& map, const Eigen::Vector3d& query_odom,
    double requested_maximum_distance_m) noexcept {
  FinalizedLidarTargetNeighbor result;
  result.view_current = true;
  result.distance_squared_m2 = std::numeric_limits<double>::infinity();
  if (!finitePositive(requested_maximum_distance_m) ||
      requested_maximum_distance_m > map.config.maximum_supported_query_distance_m) {
    return result;
  }
  const auto center = checkedVoxelKey(query_odom, map.config.query_voxel_size_m);
  if (!center) {
    return result;
  }
  result.query_valid = true;
  const double maximum_distance_squared =
      requested_maximum_distance_m * requested_maximum_distance_m;
  for (std::int64_t dx = -1; dx <= 1; ++dx) {
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dz = -1; dz <= 1; ++dz) {
        ++result.voxel_lookups;
        if (!canOffset(center->x, dx) || !canOffset(center->y, dy) || !canOffset(center->z, dz)) {
          continue;
        }
        const VoxelKey key{center->x + dx, center->y + dy, center->z + dz};
        const auto found = map.blocks.find(key);
        if (found == map.blocks.end()) {
          continue;
        }
        ++result.occupied_voxels;
        for (const FinalizedLidarTargetPoint& point : found->second.points) {
          ++result.points_examined;
          const double distance_squared = (point.point_odom - query_odom).squaredNorm();
          if (!std::isfinite(distance_squared) || distance_squared > maximum_distance_squared) {
            continue;
          }
          if (!result.found || distance_squared < result.distance_squared_m2 ||
              (distance_squared == result.distance_squared_m2 &&
               provenanceLess(point, result.point))) {
            result.found = true;
            result.point = point;
            result.distance_squared_m2 = distance_squared;
          }
        }
      }
    }
  }
  return result;
}

}  // namespace

FinalizedLidarTargetNeighbor FinalizedLidarTargetMap::nearestExact(
    const Eigen::Vector3d& query_odom, double requested_maximum_distance_m) const noexcept {
  return nearestExactInMap(*implementation_, query_odom, requested_maximum_distance_m);
}

FinalizedLidarTargetReadView FinalizedLidarTargetMap::readView() const noexcept {
  return FinalizedLidarTargetReadView(
      implementation_.get(), implementation_->config.odom_epoch, implementation_->config.sensor,
      implementation_->statistics.version, implementation_->statistics.checksum);
}

FinalizedLidarTargetNeighbor FinalizedLidarTargetReadView::nearestExact(
    const Eigen::Vector3d& query_odom, double requested_maximum_distance_m) const noexcept {
  if (map_ == nullptr || map_->statistics.version != version_ ||
      map_->statistics.checksum != checksum_) {
    return FinalizedLidarTargetNeighbor{};
  }
  return nearestExactInMap(*map_, query_odom, requested_maximum_distance_m);
}

const FinalizedLidarTargetMapConfig& FinalizedLidarTargetMap::config() const noexcept {
  return implementation_->config;
}

const FinalizedLidarTargetMapStatistics& FinalizedLidarTargetMap::statistics() const noexcept {
  return implementation_->statistics;
}

bool FinalizedLidarTargetMap::empty() const noexcept {
  return implementation_->statistics.retained_points == 0U;
}

}  // namespace meridian::local
