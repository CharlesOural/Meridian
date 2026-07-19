#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "meridian/core/blob.hpp"
#include "meridian/core/geometry.hpp"
#include "meridian/core/observation_lineage.hpp"
#include "meridian/core/strong_id.hpp"
#include "meridian/core/time.hpp"

namespace meridian::core {

// A factor identity is scoped by one odometry epoch. A bare FactorId is not a
// stable cross-module identity because the allocator restarts for a new epoch.
struct LocalFactorRef {
  OdomEpoch odom_epoch;
  FactorId factor;

  auto operator<=>(const LocalFactorRef&) const = default;
};

enum class LocalVariableKind {
  Pose,
  NavigationVelocity,
  GyroBias,
  AccelBias,
  LandmarkLogInverseRange,
};

// Exactly one of state or landmark is populated, as selected by kind.
struct LocalVariableRef {
  LocalVariableKind kind{LocalVariableKind::Pose};
  std::optional<StateId> state;
  std::optional<LandmarkSegmentId> landmark;

  auto operator<=>(const LocalVariableRef&) const = default;
};

struct SquareRootColumnBlock {
  LocalVariableRef variable;
  std::uint32_t column_offset{};
  std::uint32_t dimension{};
};

// Calibration metadata for interpreting the complete frozen objective
//   ||A dx - rhs||^2 + constant_squared_error.
// If a calibrated cutoff is present it applies to that total cost, not merely
// to the retained-row term. m and q describe the source residual system before
// and during elimination; nu=m-q is retained explicitly for auditability.
struct FrozenFactorCostStatistics {
  std::uint64_t source_residual_dof{};        // m
  std::uint64_t eliminated_numerical_rank{};  // q
  std::uint64_t effective_dof{};              // nu = m - q
  ResidualCalibrationRevision calibration_revision;
  std::optional<double> calibrated_total_cost_cutoff;
};

// Canonical solver-neutral rows for
//   ||A * boxminus(x0, x) - rhs||^2 + constant_squared_error.
// Pose columns use a right, translation-first tangent. The record owning these
// rows owns the corresponding chart centers; no solver key or factor pointer is
// part of this API.
struct FrozenSquareRootFactor {
  PoseTangentConvention pose_tangent{PoseTangentConvention::RightTranslationFirst};
  std::uint32_t rows{};
  std::uint32_t columns{};
  std::vector<SquareRootColumnBlock> layout;
  std::vector<double> row_major_A;
  std::vector<double> rhs;
  double constant_squared_error{};
  std::uint32_t numerical_rank{};
  double absolute_rank_tolerance{};
  double relative_rank_tolerance{};
  FrozenFactorCostStatistics cost_statistics;
  ContentHash checksum{};
};

enum class FrozenFactorValidationError {
  None,
  InvalidDimensions,
  InvalidVariable,
  InvalidColumnLayout,
  DuplicateVariable,
  NonFiniteValue,
  InvalidConstant,
  InvalidRankTolerance,
  InvalidCostStatistics,
  RankMismatch,
  MissingChecksum,
  UnsupportedTangent,
};

[[nodiscard]] FrozenFactorValidationError validateFrozenSquareRootFactor(
    const FrozenSquareRootFactor& factor);

struct FinalizedNavigationState {
  StateId state;
  FusionTime exact_time;
  LocalGraphRevision final_revision;
  NavStateEstimate final_estimate;
  PoseCovariance pose_covariance;
};

enum class LandmarkFinalityReason {
  Marginalized,
  SplitBarrier,
  CalibrationBoundary,
  ResetBoundary,
};

struct FinalizedLandmarkSegment {
  LandmarkSegmentId segment;
  StateId anchor_state;
  LocalGraphRevision final_revision;
  double final_log_inverse_range{};
  LandmarkFinalityReason reason{LandmarkFinalityReason::Marginalized};
};

struct BoundaryNavigationLinearization {
  StateId state;
  FusionTime exact_time;
  LocalGraphRevision final_revision;
  Pose3d T_odom_imu;
  Vector3d velocity_odom{Vector3d::Zero()};
  Vector3d gyro_bias{Vector3d::Zero()};
  Vector3d accel_bias{Vector3d::Zero()};
};

struct CondensedBoundaryTransition {
  RecordHeader header;
  OdomEpoch odom_epoch;
  BoundaryNavigationLinearization from;
  BoundaryNavigationLinearization to;
  FrozenSquareRootFactor boundary_factor;
  std::vector<LocalFactorRef> source_factors;
  ObservationLineage lineage;
  LocalGraphRevision final_revision;
  ContentHash input_partition_checksum{};
  ContentHash checksum{};
};

enum class FinalityValidationError {
  None,
  InvalidIdentity,
  NonFiniteEstimate,
  InvalidCovariance,
  InvalidInverseRange,
};

[[nodiscard]] FinalityValidationError validateFinalizedNavigationState(
    const FinalizedNavigationState& state);
[[nodiscard]] FinalityValidationError validateFinalizedLandmarkSegment(
    const FinalizedLandmarkSegment& landmark);

enum class CondensedTransitionValidationError {
  None,
  InvalidHeader,
  InvalidIdentity,
  InvalidEndpointOrder,
  NonFiniteEndpoint,
  InvalidBoundaryFactor,
  RankZeroBoundaryFactor,
  InvalidBoundaryLayout,
  InvalidRevisionOrder,
  EmptySourcePartition,
  InvalidSourceFactor,
  NonCanonicalSourceFactors,
  InvalidLineage,
  MissingChecksum,
};

[[nodiscard]] CondensedTransitionValidationError validateCondensedBoundaryTransition(
    const CondensedBoundaryTransition& transition);

}  // namespace meridian::core
