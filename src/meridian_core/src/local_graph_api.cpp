#include "meridian/core/local_graph_api.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <utility>

namespace meridian::core {
namespace {

bool hashPresent(const ContentHash& hash) noexcept {
  return std::any_of(hash.begin(), hash.end(), [](std::uint8_t byte) { return byte != 0U; });
}

bool finitePose(const Pose3d& pose) noexcept {
  return pose.matrix().allFinite();
}

bool validVariable(const LocalVariableRef& variable) noexcept {
  switch (variable.kind) {
    case LocalVariableKind::Pose:
    case LocalVariableKind::NavigationVelocity:
    case LocalVariableKind::GyroBias:
    case LocalVariableKind::AccelBias:
      return variable.state.has_value() && variable.state->valid() &&
             !variable.landmark.has_value();
    case LocalVariableKind::LandmarkLogInverseRange:
      return !variable.state.has_value() && variable.landmark.has_value() &&
             variable.landmark->valid();
  }
  return false;
}

std::uint32_t expectedDimension(LocalVariableKind kind) noexcept {
  switch (kind) {
    case LocalVariableKind::Pose:
      return 6U;
    case LocalVariableKind::NavigationVelocity:
    case LocalVariableKind::GyroBias:
    case LocalVariableKind::AccelBias:
      return 3U;
    case LocalVariableKind::LandmarkLogInverseRange:
      return 1U;
  }
  return 0U;
}

bool finiteBoundary(const BoundaryNavigationLinearization& boundary) noexcept {
  return finitePose(boundary.T_odom_imu) && boundary.velocity_odom.allFinite() &&
         boundary.gyro_bias.allFinite() && boundary.accel_bias.allFinite();
}

bool validPoseCovariance(const PoseCovariance& covariance) {
  if (covariance.tangent != PoseTangentConvention::RightTranslationFirst || !covariance.finite() ||
      !covariance.matrix.isApprox(covariance.matrix.transpose(), 1e-10)) {
    return false;
  }
  const Eigen::SelfAdjointEigenSolver<Matrix6d> eigen(covariance.matrix);
  if (eigen.info() != Eigen::Success) {
    return false;
  }
  const double scale = std::max(1.0, covariance.matrix.cwiseAbs().maxCoeff());
  return eigen.eigenvalues().minCoeff() >= -64.0 * std::numeric_limits<double>::epsilon() * scale;
}

LocalVariableRef stateVariable(LocalVariableKind kind, StateId state) {
  LocalVariableRef variable;
  variable.kind = kind;
  variable.state = state;
  return variable;
}

bool isExpectedBoundaryLayout(const FrozenSquareRootFactor& factor, StateId from, StateId to) {
  const std::vector<std::pair<LocalVariableRef, std::uint32_t>> expected{
      {stateVariable(LocalVariableKind::Pose, from), 6U},
      {stateVariable(LocalVariableKind::NavigationVelocity, from), 3U},
      {stateVariable(LocalVariableKind::GyroBias, from), 3U},
      {stateVariable(LocalVariableKind::AccelBias, from), 3U},
      {stateVariable(LocalVariableKind::Pose, to), 6U},
      {stateVariable(LocalVariableKind::NavigationVelocity, to), 3U},
      {stateVariable(LocalVariableKind::GyroBias, to), 3U},
      {stateVariable(LocalVariableKind::AccelBias, to), 3U},
  };
  if (factor.columns != 30U || factor.layout.size() != expected.size()) {
    return false;
  }
  std::uint32_t offset = 0U;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const SquareRootColumnBlock& block = factor.layout[index];
    if (block.variable != expected[index].first || block.dimension != expected[index].second ||
        block.column_offset != offset) {
      return false;
    }
    offset += block.dimension;
  }
  return offset == factor.columns;
}

}  // namespace

FrozenFactorValidationError validateFrozenSquareRootFactor(const FrozenSquareRootFactor& factor) {
  if (factor.pose_tangent != PoseTangentConvention::RightTranslationFirst) {
    return FrozenFactorValidationError::UnsupportedTangent;
  }
  if (factor.columns == 0U || factor.layout.empty() ||
      factor.row_major_A.size() != static_cast<std::size_t>(factor.rows) * factor.columns ||
      factor.rhs.size() != factor.rows) {
    return FrozenFactorValidationError::InvalidDimensions;
  }

  std::uint32_t next_column = 0U;
  std::set<LocalVariableRef> variables;
  for (const SquareRootColumnBlock& block : factor.layout) {
    if (!validVariable(block.variable) ||
        block.dimension != expectedDimension(block.variable.kind)) {
      return FrozenFactorValidationError::InvalidVariable;
    }
    if (block.column_offset != next_column || block.dimension > factor.columns - next_column) {
      return FrozenFactorValidationError::InvalidColumnLayout;
    }
    if (!variables.insert(block.variable).second) {
      return FrozenFactorValidationError::DuplicateVariable;
    }
    next_column += block.dimension;
  }
  if (next_column != factor.columns) {
    return FrozenFactorValidationError::InvalidColumnLayout;
  }

  if (!std::all_of(factor.row_major_A.begin(), factor.row_major_A.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::all_of(factor.rhs.begin(), factor.rhs.end(),
                   [](double value) { return std::isfinite(value); })) {
    return FrozenFactorValidationError::NonFiniteValue;
  }
  if (!std::isfinite(factor.constant_squared_error) || factor.constant_squared_error < 0.0) {
    return FrozenFactorValidationError::InvalidConstant;
  }
  if (!std::isfinite(factor.absolute_rank_tolerance) ||
      !std::isfinite(factor.relative_rank_tolerance) || factor.absolute_rank_tolerance <= 0.0 ||
      factor.relative_rank_tolerance < 0.0) {
    return FrozenFactorValidationError::InvalidRankTolerance;
  }
  const FrozenFactorCostStatistics& statistics = factor.cost_statistics;
  if (statistics.source_residual_dof < statistics.eliminated_numerical_rank ||
      statistics.effective_dof == 0U ||
      statistics.effective_dof !=
          statistics.source_residual_dof - statistics.eliminated_numerical_rank ||
      statistics.effective_dof < factor.numerical_rank ||
      !statistics.calibration_revision.valid() ||
      (statistics.calibrated_total_cost_cutoff.has_value() &&
       (!std::isfinite(*statistics.calibrated_total_cost_cutoff) ||
        *statistics.calibrated_total_cost_cutoff <= 0.0))) {
    return FrozenFactorValidationError::InvalidCostStatistics;
  }
  if (factor.numerical_rank > std::min(factor.rows, factor.columns)) {
    return FrozenFactorValidationError::RankMismatch;
  }

  using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Map<const RowMajorMatrix> matrix(factor.row_major_A.data(),
                                                static_cast<Eigen::Index>(factor.rows),
                                                static_cast<Eigen::Index>(factor.columns));
  std::uint32_t computed_rank = 0U;
  if (factor.rows > 0U) {
    const Eigen::JacobiSVD<RowMajorMatrix> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.info() != Eigen::Success || !svd.singularValues().allFinite()) {
      return FrozenFactorValidationError::NonFiniteValue;
    }
    const double largest = svd.singularValues().size() == 0 ? 0.0 : svd.singularValues()(0);
    const double threshold =
        std::max(factor.absolute_rank_tolerance, factor.relative_rank_tolerance * largest);
    computed_rank = static_cast<std::uint32_t>((svd.singularValues().array() > threshold).count());
  }
  if (computed_rank != factor.numerical_rank) {
    return FrozenFactorValidationError::RankMismatch;
  }
  if (!hashPresent(factor.checksum)) {
    return FrozenFactorValidationError::MissingChecksum;
  }
  return FrozenFactorValidationError::None;
}

FinalityValidationError validateFinalizedNavigationState(const FinalizedNavigationState& state) {
  if (!state.state.valid() || !state.final_revision.valid()) {
    return FinalityValidationError::InvalidIdentity;
  }
  if (!finitePose(state.final_estimate.T_odom_imu) ||
      !state.final_estimate.velocity_odom.allFinite() ||
      !state.final_estimate.gyro_bias.allFinite() || !state.final_estimate.accel_bias.allFinite()) {
    return FinalityValidationError::NonFiniteEstimate;
  }
  if (!validPoseCovariance(state.pose_covariance)) {
    return FinalityValidationError::InvalidCovariance;
  }
  return FinalityValidationError::None;
}

FinalityValidationError validateFinalizedLandmarkSegment(const FinalizedLandmarkSegment& landmark) {
  if (!landmark.segment.valid() || !landmark.anchor_state.valid() ||
      !landmark.final_revision.valid()) {
    return FinalityValidationError::InvalidIdentity;
  }
  if (!std::isfinite(landmark.final_log_inverse_range)) {
    return FinalityValidationError::InvalidInverseRange;
  }
  return FinalityValidationError::None;
}

CondensedTransitionValidationError validateCondensedBoundaryTransition(
    const CondensedBoundaryTransition& transition) {
  if (transition.header.schema_version == 0U || !transition.header.trace.valid() ||
      !transition.header.producer.valid() || !transition.header.session.valid() ||
      !transition.header.config.valid() || !transition.header.direct_calibration.has_value() ||
      !transition.header.direct_calibration->valid()) {
    return CondensedTransitionValidationError::InvalidHeader;
  }
  if (!transition.odom_epoch.valid() || !transition.from.state.valid() ||
      !transition.to.state.valid() || !transition.from.final_revision.valid() ||
      !transition.to.final_revision.valid() || !transition.final_revision.valid() ||
      transition.from.state == transition.to.state) {
    return CondensedTransitionValidationError::InvalidIdentity;
  }
  if (!(transition.from.exact_time < transition.to.exact_time)) {
    return CondensedTransitionValidationError::InvalidEndpointOrder;
  }
  if (!finiteBoundary(transition.from) || !finiteBoundary(transition.to)) {
    return CondensedTransitionValidationError::NonFiniteEndpoint;
  }
  if (validateFrozenSquareRootFactor(transition.boundary_factor) !=
      FrozenFactorValidationError::None) {
    return CondensedTransitionValidationError::InvalidBoundaryFactor;
  }
  if (transition.boundary_factor.numerical_rank == 0U) {
    return CondensedTransitionValidationError::RankZeroBoundaryFactor;
  }
  if (!isExpectedBoundaryLayout(transition.boundary_factor, transition.from.state,
                                transition.to.state)) {
    return CondensedTransitionValidationError::InvalidBoundaryLayout;
  }
  // Endpoint finality is monotonic. One successful local transaction may
  // finalize both endpoints and commit their condensation, so equality is
  // valid; revision reversal is not.
  if (transition.to.final_revision < transition.from.final_revision ||
      transition.final_revision < transition.to.final_revision) {
    return CondensedTransitionValidationError::InvalidRevisionOrder;
  }
  if (transition.source_factors.empty()) {
    return CondensedTransitionValidationError::EmptySourcePartition;
  }
  for (const LocalFactorRef& factor : transition.source_factors) {
    if (!factor.odom_epoch.valid() || !factor.factor.valid() ||
        factor.odom_epoch != transition.odom_epoch) {
      return CondensedTransitionValidationError::InvalidSourceFactor;
    }
  }
  if (!std::is_sorted(transition.source_factors.begin(), transition.source_factors.end()) ||
      std::adjacent_find(transition.source_factors.begin(), transition.source_factors.end()) !=
          transition.source_factors.end()) {
    return CondensedTransitionValidationError::NonCanonicalSourceFactors;
  }
  if (!transition.lineage.id.valid() ||
      validateLineage(transition.lineage) != LineageValidationError::None ||
      !hashPresent(transition.lineage.checksum)) {
    return CondensedTransitionValidationError::InvalidLineage;
  }
  if (!hashPresent(transition.input_partition_checksum) || !hashPresent(transition.checksum)) {
    return CondensedTransitionValidationError::MissingChecksum;
  }
  return CondensedTransitionValidationError::None;
}

}  // namespace meridian::core
