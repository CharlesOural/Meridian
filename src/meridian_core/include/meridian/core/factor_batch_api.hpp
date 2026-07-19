#pragma once

#include <Eigen/Core>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "meridian/core/geometry.hpp"
#include "meridian/core/observation_lineage.hpp"
#include "meridian/core/strong_id.hpp"
#include "meridian/core/time.hpp"

namespace meridian::core {

enum class SensorModality {
  Lidar,
  Visual,
  Gnss,
};

// Modality and deployment-local instance form one identity. Camera and LiDAR
// callers use typed factories so their independent strong-ID domains cannot be
// mixed accidentally. GNSS currently has no separate public receiver ID.
struct SensorInstanceId {
  static constexpr std::uint64_t kInvalidInstance = std::numeric_limits<std::uint64_t>::max();

  SensorModality modality{SensorModality::Lidar};
  std::uint64_t instance{kInvalidInstance};

  [[nodiscard]] static constexpr SensorInstanceId camera(CameraId camera_id) noexcept {
    return SensorInstanceId{SensorModality::Visual, camera_id.value()};
  }
  [[nodiscard]] static constexpr SensorInstanceId lidar(LidarId lidar_id) noexcept {
    return SensorInstanceId{SensorModality::Lidar, lidar_id.value()};
  }
  [[nodiscard]] static constexpr SensorInstanceId gnss(std::uint64_t receiver_instance) noexcept {
    return SensorInstanceId{SensorModality::Gnss, receiver_instance};
  }

  [[nodiscard]] bool valid() const noexcept;
  auto operator<=>(const SensorInstanceId&) const = default;
};

enum class SensorHealthState {
  Active,
  Suspect,
  Failed,
  Recovering,
};

inline constexpr std::uint64_t kInvalidSensorHealthTransitionSequence =
    std::numeric_limits<std::uint64_t>::max();

struct SensorHealthSnapshot {
  SensorInstanceId sensor;
  SensorHealthState state{SensorHealthState::Failed};
  SensorRecoveryEpoch recovery_epoch;
  std::uint64_t transition_sequence{kInvalidSensorHealthTransitionSequence};
  FusionTime assessed_at;
};

enum class SensorHealthValidationError {
  None,
  InvalidSensor,
  InvalidState,
  InvalidRecoveryEpoch,
  InvalidTransitionSequence,
};

[[nodiscard]] SensorHealthValidationError validateSensorHealthSnapshot(
    const SensorHealthSnapshot& snapshot) noexcept;

inline constexpr std::size_t kMaximumFactorBatchMeasurementTimestamps = 4'096U;

struct FactorBatchTiming {
  // Half-open support [start, end), matching core::TimeRange semantics.
  TimeRange support;
  // Strictly increasing fusion timestamps of the raw observations represented
  // by the batch. The support range may include acquisition between samples.
  std::vector<FusionTime> measurement_timestamps;
  FusionTime reference_time;
  FusionTime produced_at;
};

enum class FactorBatchTimingValidationError {
  None,
  InvalidSupport,
  InvalidMeasurementCount,
  NonCanonicalMeasurementTimestamps,
  MeasurementOutsideSupport,
  ReferenceOutsideSupport,
  ProducedBeforeSupportEnd,
};

[[nodiscard]] FactorBatchTimingValidationError validateFactorBatchTiming(
    const FactorBatchTiming& timing) noexcept;

enum class DirectionalVariable {
  PoseTranslation,
  PoseRotation,
  Velocity,
  AccelerometerBias,
  GyroscopeBias,
};

enum class DirectionalEndpointRole {
  Target,
  Source,
  Unary,
};

struct DirectionalObservabilityEndpoint {
  DirectionalEndpointRole role{DirectionalEndpointRole::Unary};
  StateId state;
  FusionTime exact_time;

  auto operator<=>(const DirectionalObservabilityEndpoint&) const = default;
};

inline constexpr std::size_t kMaximumDirectionalObservabilityRecords = 64U;
inline constexpr std::size_t kMaximumDirectionalObservabilityEndpoints = 2U;
inline constexpr std::size_t kMaximumDirectionalObservabilityVariables = 5U;

// Solver-neutral pose-direction evidence. Columns of basis correspond to the
// ascending eigenvalues. The final `rank` columns are therefore the supported
// directions when rank is nonzero. This is not a covariance or a replacement
// for the exact covariance/information retained by each typed factor payload.
struct DirectionalObservability {
  PoseTangentConvention tangent{PoseTangentConvention::RightTranslationFirst};
  Eigen::Matrix<double, 6, 6> basis{Eigen::Matrix<double, 6, 6>::Identity()};
  Eigen::Matrix<double, 6, 1> eigenvalues{Eigen::Matrix<double, 6, 1>::Zero()};
  std::uint32_t rank{};
  double absolute_eigenvalue_threshold{};
  double relative_eigenvalue_threshold{};
  // Canonical ascending unique variable blocks touched by the typed sensor
  // factors themselves. Conditioning or seed information must not be reported
  // as a factor-supported variable.
  std::vector<DirectionalVariable> supported_variables;
  // One endpoint describes unary evidence; two describe relative evidence.
  // Unary records contain one Unary endpoint. Binary records are canonical as
  // Target then Source and name distinct, time-ordered navigation states.
  std::vector<DirectionalObservabilityEndpoint> endpoints;
};

enum class DirectionalObservabilityValidationError {
  None,
  UnsupportedTangent,
  NonFiniteBasis,
  NonOrthonormalBasis,
  InvalidEigenvalues,
  InvalidThresholds,
  RankMismatch,
  InvalidSupportedVariables,
  InvalidEndpoints,
};

[[nodiscard]] DirectionalObservabilityValidationError validateDirectionalObservability(
    const DirectionalObservability& observability) noexcept;

struct FactorBatchMetadata {
  RecordHeader header;
  FactorBatchId batch_id;
  OdomEpoch odom_epoch;
  SensorInstanceId sensor;
  FactorBatchTiming timing;
  SensorHealthSnapshot health;
  bool map_eligible{};
  std::vector<DirectionalObservability> directional_observability;
  ObservationLineage lineage;
};

enum class FactorBatchMetadataValidationError {
  None,
  InvalidHeader,
  InvalidBatchId,
  InvalidOdomEpoch,
  InvalidSensor,
  InvalidTiming,
  InvalidHealth,
  SensorHealthMismatch,
  InvalidHealthAssessmentTime,
  InvalidObservabilityCount,
  InvalidDirectionalObservability,
  MapEligibilityRequiresActiveHealth,
  InvalidLineage,
};

[[nodiscard]] FactorBatchMetadataValidationError validateFactorBatchMetadata(
    const FactorBatchMetadata& metadata) noexcept;

}  // namespace meridian::core
