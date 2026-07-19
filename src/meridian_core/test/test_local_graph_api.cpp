#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "meridian/core/local_graph_api.hpp"

namespace meridian::core {
namespace {

ContentHash presentHash(std::uint8_t value) {
  ContentHash hash{};
  hash.front() = value;
  return hash;
}

LocalVariableRef stateVariable(LocalVariableKind kind, StateId state) {
  LocalVariableRef variable;
  variable.kind = kind;
  variable.state = state;
  return variable;
}

FrozenSquareRootFactor makeBoundaryFactor(StateId from, StateId to) {
  FrozenSquareRootFactor factor;
  factor.rows = 2U;
  factor.columns = 30U;
  factor.layout = {
      {stateVariable(LocalVariableKind::Pose, from), 0U, 6U},
      {stateVariable(LocalVariableKind::NavigationVelocity, from), 6U, 3U},
      {stateVariable(LocalVariableKind::GyroBias, from), 9U, 3U},
      {stateVariable(LocalVariableKind::AccelBias, from), 12U, 3U},
      {stateVariable(LocalVariableKind::Pose, to), 15U, 6U},
      {stateVariable(LocalVariableKind::NavigationVelocity, to), 21U, 3U},
      {stateVariable(LocalVariableKind::GyroBias, to), 24U, 3U},
      {stateVariable(LocalVariableKind::AccelBias, to), 27U, 3U},
  };
  factor.row_major_A.assign(60U, 0.0);
  factor.row_major_A[0U] = 2.0;
  factor.row_major_A[31U] = 3.0;
  factor.rhs = {1.0, -1.0};
  factor.constant_squared_error = 0.25;
  factor.numerical_rank = 2U;
  factor.absolute_rank_tolerance = 1e-12;
  factor.relative_rank_tolerance = 1e-9;
  factor.cost_statistics.source_residual_dof = 5U;
  factor.cost_statistics.eliminated_numerical_rank = 1U;
  factor.cost_statistics.effective_dof = 4U;
  factor.cost_statistics.calibration_revision = ResidualCalibrationRevision{1U};
  factor.cost_statistics.calibrated_total_cost_cutoff = 12.0;
  factor.checksum = presentHash(1U);
  return factor;
}

ObservationLineage makeLineage() {
  ObservationLineage lineage;
  lineage.id = ObservationLineageId{1U};
  lineage.checksum = presentHash(2U);
  ObservationSlice slice;
  slice.root = MeasurementId{3U};
  slice.calibration = CalibrationEpoch{4U};
  lineage.usage.push_back(ObservationUsage{slice, ObservationRole::PrimaryResidual,
                                           DerivedRecordId{5U}, FactorGroupId{6U}, std::nullopt});
  return lineage;
}

CondensedBoundaryTransition makeTransition() {
  CondensedBoundaryTransition transition;
  transition.header.schema_version = 1U;
  transition.header.trace = TraceId{1U};
  transition.header.producer = ProducerId{2U};
  transition.header.session = SessionId{3U};
  transition.header.config = ConfigRevision{4U};
  transition.header.direct_calibration = CalibrationEpoch{5U};
  transition.odom_epoch = OdomEpoch{6U};
  transition.from.state = StateId{10U};
  transition.from.exact_time = FusionTime{100};
  transition.from.final_revision = LocalGraphRevision{5U};
  transition.to.state = StateId{20U};
  transition.to.exact_time = FusionTime{200};
  transition.to.final_revision = LocalGraphRevision{6U};
  transition.boundary_factor = makeBoundaryFactor(transition.from.state, transition.to.state);
  transition.source_factors = {
      {transition.odom_epoch, FactorId{1U}},
      {transition.odom_epoch, FactorId{2U}},
  };
  transition.lineage = makeLineage();
  transition.final_revision = LocalGraphRevision{7U};
  transition.input_partition_checksum = presentHash(8U);
  transition.checksum = presentHash(9U);
  return transition;
}

TEST(FrozenSquareRootFactor, AcceptsCanonicalRankDeficientRows) {
  const auto factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  EXPECT_EQ(validateFrozenSquareRootFactor(factor), FrozenFactorValidationError::None);
}

TEST(FrozenSquareRootFactor, RejectsGapsDuplicateVariablesAndRankClaims) {
  auto factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.layout[1].column_offset = 7U;
  EXPECT_EQ(validateFrozenSquareRootFactor(factor),
            FrozenFactorValidationError::InvalidColumnLayout);

  factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.layout[1].variable = factor.layout[2].variable;
  EXPECT_EQ(validateFrozenSquareRootFactor(factor), FrozenFactorValidationError::DuplicateVariable);

  factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.numerical_rank = 1U;
  EXPECT_EQ(validateFrozenSquareRootFactor(factor), FrozenFactorValidationError::RankMismatch);
}

TEST(FrozenSquareRootFactor, RejectsNonFiniteRowsAndNegativeConstant) {
  auto factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.row_major_A.front() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(validateFrozenSquareRootFactor(factor), FrozenFactorValidationError::NonFiniteValue);

  factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.constant_squared_error = -1.0;
  EXPECT_EQ(validateFrozenSquareRootFactor(factor), FrozenFactorValidationError::InvalidConstant);
}

TEST(FrozenSquareRootFactor, ValidatesCalibratedCompleteCostStatistics) {
  auto factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.cost_statistics.source_residual_dof = 1U;
  factor.cost_statistics.eliminated_numerical_rank = 2U;
  EXPECT_EQ(validateFrozenSquareRootFactor(factor),
            FrozenFactorValidationError::InvalidCostStatistics);

  factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.cost_statistics.effective_dof = 3U;
  EXPECT_EQ(validateFrozenSquareRootFactor(factor),
            FrozenFactorValidationError::InvalidCostStatistics);

  factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.cost_statistics.source_residual_dof = 1U;
  factor.cost_statistics.eliminated_numerical_rank = 1U;
  factor.cost_statistics.effective_dof = 0U;
  EXPECT_EQ(validateFrozenSquareRootFactor(factor),
            FrozenFactorValidationError::InvalidCostStatistics);

  factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.cost_statistics.source_residual_dof = 2U;
  factor.cost_statistics.eliminated_numerical_rank = 1U;
  factor.cost_statistics.effective_dof = 1U;
  EXPECT_EQ(validateFrozenSquareRootFactor(factor),
            FrozenFactorValidationError::InvalidCostStatistics);

  factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.cost_statistics.calibration_revision = ResidualCalibrationRevision{};
  EXPECT_EQ(validateFrozenSquareRootFactor(factor),
            FrozenFactorValidationError::InvalidCostStatistics);

  factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.cost_statistics.calibrated_total_cost_cutoff = std::numeric_limits<double>::infinity();
  EXPECT_EQ(validateFrozenSquareRootFactor(factor),
            FrozenFactorValidationError::InvalidCostStatistics);

  factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.cost_statistics.calibrated_total_cost_cutoff = 0.0;
  EXPECT_EQ(validateFrozenSquareRootFactor(factor),
            FrozenFactorValidationError::InvalidCostStatistics);

  factor = makeBoundaryFactor(StateId{1U}, StateId{2U});
  factor.cost_statistics.calibrated_total_cost_cutoff.reset();
  EXPECT_EQ(validateFrozenSquareRootFactor(factor), FrozenFactorValidationError::None);
}

TEST(Finality, RejectsNonPsdPoseCovarianceAndNonFiniteLandmark) {
  FinalizedNavigationState state;
  state.state = StateId{1U};
  state.final_revision = LocalGraphRevision{2U};
  state.pose_covariance.matrix.setIdentity();
  state.pose_covariance.matrix(0, 0) = -1.0;
  EXPECT_EQ(validateFinalizedNavigationState(state), FinalityValidationError::InvalidCovariance);

  FinalizedLandmarkSegment landmark;
  landmark.segment = LandmarkSegmentId{1U};
  landmark.anchor_state = StateId{2U};
  landmark.final_revision = LocalGraphRevision{3U};
  landmark.final_log_inverse_range = std::numeric_limits<double>::infinity();
  EXPECT_EQ(validateFinalizedLandmarkSegment(landmark),
            FinalityValidationError::InvalidInverseRange);
}

TEST(CondensedBoundaryTransition, AcceptsExactCanonicalBoundaryApi) {
  const auto transition = makeTransition();
  EXPECT_EQ(validateCondensedBoundaryTransition(transition),
            CondensedTransitionValidationError::None);
}

TEST(CondensedBoundaryTransition, RejectsWrongLayoutEpochAndOrdering) {
  auto transition = makeTransition();
  std::swap(transition.boundary_factor.layout[0], transition.boundary_factor.layout[4]);
  EXPECT_EQ(validateCondensedBoundaryTransition(transition),
            CondensedTransitionValidationError::InvalidBoundaryFactor);

  transition = makeTransition();
  transition.source_factors[0].odom_epoch = OdomEpoch{99U};
  EXPECT_EQ(validateCondensedBoundaryTransition(transition),
            CondensedTransitionValidationError::InvalidSourceFactor);

  transition = makeTransition();
  std::reverse(transition.source_factors.begin(), transition.source_factors.end());
  EXPECT_EQ(validateCondensedBoundaryTransition(transition),
            CondensedTransitionValidationError::NonCanonicalSourceFactors);
}

TEST(CondensedBoundaryTransition, RequiresOrderedEndpointFinality) {
  auto transition = makeTransition();
  transition.from.final_revision = transition.to.final_revision;
  EXPECT_EQ(validateCondensedBoundaryTransition(transition),
            CondensedTransitionValidationError::None);

  transition = makeTransition();
  transition.from.final_revision = LocalGraphRevision{7U};
  EXPECT_EQ(validateCondensedBoundaryTransition(transition),
            CondensedTransitionValidationError::InvalidRevisionOrder);

  transition = makeTransition();
  transition.final_revision = LocalGraphRevision{5U};
  EXPECT_EQ(validateCondensedBoundaryTransition(transition),
            CondensedTransitionValidationError::InvalidRevisionOrder);
}

TEST(CondensedBoundaryTransition, RejectsRankZeroBoundaryInformation) {
  auto transition = makeTransition();
  std::fill(transition.boundary_factor.row_major_A.begin(),
            transition.boundary_factor.row_major_A.end(), 0.0);
  transition.boundary_factor.numerical_rank = 0U;
  ASSERT_EQ(validateFrozenSquareRootFactor(transition.boundary_factor),
            FrozenFactorValidationError::None);
  EXPECT_EQ(validateCondensedBoundaryTransition(transition),
            CondensedTransitionValidationError::RankZeroBoundaryFactor);
}

TEST(CondensedBoundaryTransition, RejectsMissingLineageAndPartitionChecksums) {
  auto transition = makeTransition();
  transition.lineage.checksum = {};
  EXPECT_EQ(validateCondensedBoundaryTransition(transition),
            CondensedTransitionValidationError::InvalidLineage);

  transition = makeTransition();
  transition.input_partition_checksum = {};
  EXPECT_EQ(validateCondensedBoundaryTransition(transition),
            CondensedTransitionValidationError::MissingChecksum);
}

}  // namespace
}  // namespace meridian::core
