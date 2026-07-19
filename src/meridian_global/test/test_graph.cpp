#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

#include "meridian/global/loop_consensus.hpp"
#include "sparse_seal_test_utils.hpp"

namespace meridian::global {
namespace {

[[nodiscard]] double poseError(const core::Pose3d& expected, const core::Pose3d& actual) {
  return (expected.inverse() * actual).log().norm();
}

[[nodiscard]] const SubmapAnchorEstimate& anchor(const GlobalGraphCommit& commit,
                                                 core::SubmapId id) {
  const auto found =
      std::find_if(commit.anchors.begin(), commit.anchors.end(),
                   [&](const SubmapAnchorEstimate& estimate) { return estimate.submap.id == id; });
  EXPECT_NE(found, commit.anchors.end());
  return *found;
}

[[nodiscard]] core::RankAwareInformation fullInformation(double value = 100.0) {
  core::RankAwareInformation information;
  information.basis.setIdentity();
  information.eigenvalues.setConstant(value);
  information.rank = 6U;
  return information;
}

[[nodiscard]] RobustLoopCandidate loopCandidate(std::uint64_t proposal,
                                                const core::SparseSubmapSeal& from,
                                                const core::SparseSubmapSeal& to) {
  core::RecordHeader header;
  header.schema_version = 1U;
  header.trace = core::TraceId(800U + proposal);
  header.producer = core::ProducerId(7U);
  header.session = from.ref.session;
  header.created_at = to.support_time.end;
  header.config = core::ConfigRevision(2U);
  header.direct_calibration = from.ref.calibration;
  return RobustLoopCandidate{LoopMeasurement{header,
                                             ProposalId(proposal),
                                             LoopModality::Lidar,
                                             from.ref,
                                             to.ref,
                                             {from.ref.calibration},
                                             from.T_odom_submap.inverse() * to.T_odom_submap,
                                             fullInformation(500.0),
                                             test_support::lineage(800U + proposal)},
                             GncTlsFactorScale{6U, 16.81189383}};
}

[[nodiscard]] LoopConsensusReport admittedConsensus(ProposalId proposal,
                                                    core::FusionTime evaluated_at) {
  LoopConsensusReport report;
  report.evaluated_at = evaluated_at;
  report.decisions.push_back(LoopProposalDecision{
      proposal, LoopProposalDisposition::AdmittedToGncCandidateBatch, 0U, std::nullopt});
  return report;
}

struct JointlyObservableChain {
  core::SparseSubmapSeal first;
  core::SparseSubmapSeal second;
  core::SparseSubmapSeal third;
};

[[nodiscard]] JointlyObservableChain jointlyObservableChain() {
  JointlyObservableChain chain;
  chain.first = test_support::firstSeal();
  chain.second = test_support::successor(chain.first);
  std::vector<std::uint32_t> first_columns;
  for (std::uint32_t column = 6U; column <= 28U; ++column) {
    first_columns.push_back(column);
  }
  test_support::replaceBoundaryFactor(
      &chain.second,
      test_support::boundaryFactorOnColumns(chain.first.boundary_navigation.state,
                                            chain.second.boundary_navigation.state, first_columns));

  chain.third = test_support::successor(chain.second);
  std::vector<std::uint32_t> second_columns{14U};
  for (std::uint32_t column = 15U; column <= 29U; ++column) {
    second_columns.push_back(column);
  }
  test_support::replaceBoundaryFactor(
      &chain.third,
      test_support::boundaryFactorOnColumns(chain.second.boundary_navigation.state,
                                            chain.third.boundary_navigation.state, second_columns));
  return chain;
}

TEST(GlobalGraph, FirstSealMaterializesOnlySixScalarAnchorAndMissionGauge) {
  GlobalGraph graph;
  const core::Pose3d T_odom_submap = test_support::pose({4.0, -2.0, 0.8}, {0.08, -0.04, 0.45});
  const core::SparseSubmapSeal first = test_support::firstSeal(
      11U, 0, T_odom_submap, T_odom_submap * test_support::pose({0.2, 0.0, 0.1}));

  const auto initialized = graph.initializeMission(first);
  ASSERT_TRUE(initialized) << initialized.error().detail;
  const GlobalGraphCommit& commit = initialized.value();
  EXPECT_EQ(commit.revision, GlobalGraphRevision(0U));
  EXPECT_FALSE(commit.parent.has_value());
  ASSERT_EQ(commit.anchors.size(), 1U);
  EXPECT_LT(poseError(T_odom_submap, commit.anchors.front().T_map_submap), 1.0e-10);
  EXPECT_LT(poseError(core::Pose3d{}, commit.map_odom.T_map_odom), 1.0e-10);
  EXPECT_EQ(commit.map_odom.reference_submap, first.ref);
  EXPECT_EQ(commit.solve.scalar_dimension, 6U);
  EXPECT_EQ(commit.solve.materialized_navigation_boundaries, 0U);
  EXPECT_EQ(commit.solve.numerical_rank, 6U);
  EXPECT_TRUE(commit.solve.connected);

  const auto checkpoint = graph.checkpoint();
  ASSERT_TRUE(checkpoint);
  ASSERT_EQ(checkpoint.value().boundaries.size(), 1U);
  EXPECT_FALSE(checkpoint.value().boundaries.front().velocity_map.has_value());
  EXPECT_FALSE(checkpoint.value().boundaries.front().gyro_bias.has_value());
  EXPECT_FALSE(checkpoint.value().boundaries.front().accel_bias.has_value());
  EXPECT_TRUE(checkpoint.value().adjacent_factors.empty());
  EXPECT_EQ(checkpoint.value().next_boundary_slot, 1U);
  EXPECT_EQ(checkpoint.value().next_factor_id, 1U);
}

TEST(GlobalGraph, FirstAdjacencyMaterializesAllLatentsAndOneExactEightVariableFactor) {
  GlobalGraph graph;
  const core::SparseSubmapSeal first = test_support::firstSeal();
  const core::SparseSubmapSeal second = test_support::successor(first);
  ASSERT_TRUE(graph.initializeMission(first));

  const auto appended = graph.appendAdjacent(second);
  ASSERT_TRUE(appended) << appended.error().detail;
  EXPECT_EQ(appended.value().solve.scalar_dimension, 30U);
  EXPECT_EQ(appended.value().solve.materialized_navigation_boundaries, 2U);
  EXPECT_EQ(appended.value().solve.adjacent_factors, 1U);
  EXPECT_EQ(appended.value().solve.numerical_rank, 30U);

  const auto checkpoint = graph.checkpoint();
  ASSERT_TRUE(checkpoint);
  ASSERT_EQ(checkpoint.value().boundaries.size(), 2U);
  for (const BoundaryNavigationCheckpoint& boundary : checkpoint.value().boundaries) {
    EXPECT_TRUE(boundary.velocity_map.has_value());
    EXPECT_TRUE(boundary.gyro_bias.has_value());
    EXPECT_TRUE(boundary.accel_bias.has_value());
  }
  ASSERT_EQ(checkpoint.value().adjacent_factors.size(), 1U);
  const auto& exact = checkpoint.value().adjacent_factors.front();
  EXPECT_EQ(exact.from_slot, 0U);
  EXPECT_EQ(exact.to_slot, 1U);
  EXPECT_EQ(exact.transition.from, second.from_previous->from);
  EXPECT_EQ(exact.transition.to, second.from_previous->to);
  EXPECT_EQ(exact.transition.local_transition.checksum,
            second.from_previous->local_transition.checksum);
  EXPECT_EQ(exact.transition.checksum, second.from_previous->checksum);
  // Gauge id 0 and exact adjacent id 1: no synthetic pose prior was inserted.
  EXPECT_EQ(checkpoint.value().next_factor_id, 2U);
}

TEST(GlobalGraph, ExactPredecessorReferenceRejectionIsAtomic) {
  GlobalGraph graph;
  const core::SparseSubmapSeal first = test_support::firstSeal();
  core::SparseSubmapSeal second = test_support::successor(first);
  ASSERT_TRUE(graph.initializeMission(first));
  const GlobalGraphCommit before = graph.snapshot().value();
  const GlobalGraphCheckpoint checkpoint_before = graph.checkpoint().value();

  core::SubmapRef altered = first.ref;
  altered.local_content_checksum.front() ^= 0x5aU;
  test_support::replacePredecessorRef(&second, altered);
  ASSERT_TRUE(core::verifyCanonicalSparseSubmapSeal(second));
  const auto rejected = graph.appendAdjacent(second);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().code == GlobalGraphErrorCode::DisconnectedProposal ||
              rejected.error().code == GlobalGraphErrorCode::StaleSubmapReference);

  const GlobalGraphCommit after = graph.snapshot().value();
  const GlobalGraphCheckpoint checkpoint_after = graph.checkpoint().value();
  EXPECT_EQ(after.revision, before.revision);
  EXPECT_EQ(checkpoint_after.next_boundary_slot, checkpoint_before.next_boundary_slot);
  EXPECT_EQ(checkpoint_after.next_factor_id, checkpoint_before.next_factor_id);
  EXPECT_EQ(checkpoint_after.next_candidate_id, checkpoint_before.next_candidate_id);
  EXPECT_EQ(checkpoint_after.boundaries.size(), checkpoint_before.boundaries.size());
}

TEST(GlobalGraph, CanonicalDigestFailureRetainsTypedVerificationEvidence) {
  GlobalGraph graph;
  core::SparseSubmapSeal corrupted = test_support::firstSeal();
  corrupted.T_odom_submap.translation().x() += 1.0;

  const auto rejected = graph.initializeMission(corrupted);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GlobalGraphErrorCode::InvalidSparseSeal);
  ASSERT_TRUE(rejected.error().canonical_verification_error.has_value());
  EXPECT_EQ(rejected.error().canonical_verification_error->failure,
            core::CanonicalVerificationFailure::DigestMismatch);
  EXPECT_FALSE(graph.initialized());
}

TEST(GlobalGraph, SingleLowRankTransitionCannotCommitWithoutAtomicChainStaging) {
  GlobalGraph graph;
  const core::SparseSubmapSeal first = test_support::firstSeal();
  const core::SparseSubmapSeal low_rank = test_support::successor(first, 1U);
  ASSERT_TRUE(graph.initializeMission(first));
  const GlobalGraphCheckpoint before = graph.checkpoint().value();

  const auto rejected = graph.appendAdjacent(low_rank);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GlobalGraphErrorCode::RankDeficientCandidate);
  const GlobalGraphCheckpoint after = graph.checkpoint().value();
  EXPECT_EQ(after.revision, before.revision);
  EXPECT_EQ(after.next_boundary_slot, before.next_boundary_slot);
  EXPECT_EQ(after.next_factor_id, before.next_factor_id);
}

TEST(GlobalGraph, AtomicAdjacentBatchCommitsJointlyObservableLowRankChainOnce) {
  const JointlyObservableChain chain = jointlyObservableChain();

  GlobalGraph single_graph;
  ASSERT_TRUE(single_graph.initializeMission(chain.first));
  const auto individually_rejected = single_graph.appendAdjacent(chain.second);
  ASSERT_FALSE(individually_rejected);
  EXPECT_EQ(individually_rejected.error().code, GlobalGraphErrorCode::RankDeficientCandidate);
  EXPECT_EQ(single_graph.snapshot().value().revision, GlobalGraphRevision(0U));

  GlobalGraph batch_graph;
  ASSERT_TRUE(batch_graph.initializeMission(chain.first));
  const auto committed = batch_graph.appendAdjacentBatch({chain.second, chain.third});
  ASSERT_TRUE(committed) << committed.error().detail;
  EXPECT_EQ(committed.value().revision, GlobalGraphRevision(1U));
  EXPECT_EQ(committed.value().solve.anchors, 3U);
  EXPECT_EQ(committed.value().solve.materialized_navigation_boundaries, 3U);
  EXPECT_EQ(committed.value().solve.adjacent_seals_in_transaction, 2U);
  EXPECT_EQ(committed.value().solve.adjacent_factors, 2U);
  EXPECT_EQ(committed.value().solve.scalar_dimension, 45U);
  EXPECT_EQ(committed.value().solve.numerical_rank, 45U);
  const GlobalGraphCheckpoint checkpoint = batch_graph.checkpoint().value();
  EXPECT_EQ(checkpoint.next_boundary_slot, 3U);
  EXPECT_EQ(checkpoint.next_factor_id, 3U);
  ASSERT_EQ(checkpoint.adjacent_factors.size(), 2U);
  EXPECT_EQ(checkpoint.adjacent_factors[0].factor, GlobalFactorId(1U));
  EXPECT_EQ(checkpoint.adjacent_factors[1].factor, GlobalFactorId(2U));
}

TEST(GlobalGraph, AdjacentBatchOrderingFailureIsAtomic) {
  const JointlyObservableChain chain = jointlyObservableChain();
  GlobalGraph graph;
  ASSERT_TRUE(graph.initializeMission(chain.first));
  const GlobalGraphCheckpoint before = graph.checkpoint().value();

  const auto rejected = graph.appendAdjacentBatch({chain.third, chain.second});
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().code == GlobalGraphErrorCode::DisconnectedProposal ||
              rejected.error().code == GlobalGraphErrorCode::NonConsecutiveAdjacentBatch);
  const GlobalGraphCheckpoint after = graph.checkpoint().value();
  EXPECT_EQ(after.revision, before.revision);
  EXPECT_EQ(after.next_boundary_slot, before.next_boundary_slot);
  EXPECT_EQ(after.next_factor_id, before.next_factor_id);
  EXPECT_EQ(after.boundaries.size(), before.boundaries.size());
}

TEST(GlobalGraph, AdjacentBatchLateExactLinkFailureRollsBackEarlierStaging) {
  JointlyObservableChain chain = jointlyObservableChain();
  core::SubmapRef altered_second = chain.second.ref;
  altered_second.local_content_checksum.front() ^= 0x6dU;
  test_support::replacePredecessorRef(&chain.third, altered_second);
  ASSERT_TRUE(core::verifyCanonicalSparseSubmapSeal(chain.third));

  GlobalGraph graph;
  ASSERT_TRUE(graph.initializeMission(chain.first));
  const GlobalGraphCheckpoint before = graph.checkpoint().value();
  const auto rejected = graph.appendAdjacentBatch({chain.second, chain.third});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GlobalGraphErrorCode::StaleSubmapReference);
  const GlobalGraphCheckpoint after = graph.checkpoint().value();
  EXPECT_EQ(after.revision, before.revision);
  EXPECT_EQ(after.next_boundary_slot, before.next_boundary_slot);
  EXPECT_EQ(after.next_factor_id, before.next_factor_id);
  EXPECT_EQ(after.boundaries.size(), before.boundaries.size());
  EXPECT_TRUE(after.adjacent_factors.empty());
}

TEST(GlobalGraph, AdjacentBatchAndAggregateWorkCapsFailBeforeCommit) {
  const JointlyObservableChain chain = jointlyObservableChain();

  GlobalGraphConfig batch_config;
  batch_config.maximum_adjacent_seals_per_transaction = 1U;
  GlobalGraph batch_limited(batch_config);
  ASSERT_TRUE(batch_limited.initializeMission(chain.first));
  const auto batch_rejected = batch_limited.appendAdjacentBatch({chain.second, chain.third});
  ASSERT_FALSE(batch_rejected);
  EXPECT_EQ(batch_rejected.error().code, GlobalGraphErrorCode::AdjacentBatchCapacity);
  EXPECT_EQ(batch_limited.snapshot().value().revision, GlobalGraphRevision(0U));
  EXPECT_EQ(batch_limited.checkpoint().value().next_factor_id, 1U);

  GlobalGraphConfig work_config;
  work_config.maximum_adjacent_factor_rows = 23U;
  work_config.maximum_adjacent_factor_coefficients = 23U * 30U;
  work_config.maximum_total_adjacent_factor_rows = 30U;
  work_config.maximum_total_adjacent_factor_coefficients = 30U * 30U;
  GlobalGraph work_limited(work_config);
  ASSERT_TRUE(work_limited.initializeMission(chain.first));
  const auto work_rejected = work_limited.appendAdjacentBatch({chain.second, chain.third});
  ASSERT_FALSE(work_rejected);
  EXPECT_EQ(work_rejected.error().code, GlobalGraphErrorCode::AdjacentFactorCapacity);
  EXPECT_EQ(work_limited.snapshot().value().revision, GlobalGraphRevision(0U));
  EXPECT_EQ(work_limited.checkpoint().value().next_factor_id, 1U);
}

TEST(GlobalGraph, EmptyAdjacentBatchIsRejectedWithoutMutation) {
  GlobalGraph graph;
  const core::SparseSubmapSeal first = test_support::firstSeal();
  ASSERT_TRUE(graph.initializeMission(first));
  const auto rejected = graph.appendAdjacentBatch({});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GlobalGraphErrorCode::EmptyAdjacentBatch);
  EXPECT_EQ(graph.snapshot().value().revision, GlobalGraphRevision(0U));
  EXPECT_EQ(graph.checkpoint().value().next_factor_id, 1U);
}

TEST(GlobalGraph, MapOdomUsesLatestAnchorCompositionAndRightTangentAdjoint) {
  GlobalGraph graph;
  const core::Pose3d T_odom_first = test_support::pose({12.0, -7.0, 2.0}, {0.25, -0.1, 0.7});
  const core::SparseSubmapSeal first = test_support::firstSeal(
      1U, 0, T_odom_first, T_odom_first * test_support::pose({0.1, 0.2, -0.1}));
  const core::SparseSubmapSeal second = test_support::successor(first, 24U, 0.75);
  ASSERT_TRUE(graph.initializeMission(first));
  const auto appended = graph.appendAdjacent(second);
  ASSERT_TRUE(appended) << appended.error().detail;
  const GlobalGraphCommit& commit = appended.value();
  const SubmapAnchorEstimate& reference = anchor(commit, second.ref.id);

  const core::Pose3d expected = reference.T_map_submap * reference.T_odom_submap.inverse();
  EXPECT_LT(poseError(expected, commit.map_odom.T_map_odom), 1.0e-9);
  EXPECT_GT(commit.map_odom.T_map_odom.log().norm(), 0.1);
  const core::Matrix6d expected_covariance = reference.T_odom_submap.Adj() *
                                             reference.covariance.matrix *
                                             reference.T_odom_submap.Adj().transpose();
  EXPECT_LT((expected_covariance - commit.map_odom.covariance.matrix).cwiseAbs().maxCoeff(),
            1.0e-10);
  EXPECT_EQ(commit.map_odom.covariance_semantics,
            MapOdomCovarianceSemantics::ConditionalOnSealedLocalFrame);
}

TEST(GlobalGraph, GnssFactorsRecoverFourDofAlignmentWithoutInitializerPrior) {
  GlobalGraph graph;
  const core::SparseSubmapSeal first = test_support::firstSeal();
  ASSERT_TRUE(graph.initializeMission(first));
  const YawTranslation4 truth{{100.0, -30.0, 7.0}, 0.72};
  const std::array<Eigen::Vector3d, 6> antenna_points{
      Eigen::Vector3d{0.0, 0.0, 0.0},  Eigen::Vector3d{12.0, 0.0, 0.5},
      Eigen::Vector3d{0.0, 9.0, -0.2}, Eigen::Vector3d{6.0, -5.0, 1.0},
      Eigen::Vector3d{-4.0, 7.0, 0.3}, Eigen::Vector3d{15.0, 11.0, -0.6}};
  GnssBatchAppend batch;
  batch.initial_alignment = YawTranslation4{{98.0, -27.0, 5.0}, 0.5};
  for (std::size_t index = 0U; index < antenna_points.size(); ++index) {
    batch.constraints.push_back(GnssAntennaConstraint{
        first.ref, core::GnssObservationId(index + 1U), antenna_points[index],
        truth.apply(antenna_points[index]), Eigen::Matrix3d::Identity() * 0.01});
  }

  const auto admitted = graph.appendGnssBatch(std::move(batch));
  ASSERT_TRUE(admitted) << admitted.error().detail;
  ASSERT_TRUE(admitted.value().alignment.has_value());
  EXPECT_LT((admitted.value().alignment->translation_enu - truth.translation_enu).norm(), 1.0e-6);
  EXPECT_NEAR(std::remainder(admitted.value().alignment->yaw_enu_map_rad - truth.yaw_enu_map_rad,
                             2.0 * std::numbers::pi),
              0.0, 1.0e-8);
  EXPECT_EQ(admitted.value().solve.gnss_factors, antenna_points.size());
  EXPECT_EQ(admitted.value().solve.scalar_dimension, 10U);
}

TEST(GlobalGraph, CommittedLoopHasIndependentFactorCandidateAndProposalIdentities) {
  GlobalGraphConfig config;
  config.loop_gnc.mu_step = 2.0;
  GlobalGraph graph(config);
  const core::SparseSubmapSeal first = test_support::firstSeal();
  const core::SparseSubmapSeal second = test_support::successor(first);
  ASSERT_TRUE(graph.initializeMission(first));
  ASSERT_TRUE(graph.appendAdjacent(second));

  constexpr std::uint64_t kProposal = 77U;
  RobustLoopBatchAppend batch;
  batch.expected_parent = graph.snapshot().value().revision;
  batch.candidates.push_back(loopCandidate(kProposal, first, second));
  const LoopConsensusReport consensus =
      admittedConsensus(ProposalId(kProposal), core::FusionTime{10'000});
  const auto result = graph.appendRobustLoopBatch(std::move(batch), consensus);
  ASSERT_TRUE(result) << result.error().detail;
  ASSERT_TRUE(result.value().commit.has_value());
  const auto adjacent_monitor = std::find_if(
      result.value().report.gnc.known_inliers.begin(),
      result.value().report.gnc.known_inliers.end(),
      [](const GncTlsKnownInlierReport& factor) { return factor.factor_id == GlobalFactorId(1U); });
  ASSERT_NE(adjacent_monitor, result.value().report.gnc.known_inliers.end());
  ASSERT_TRUE(adjacent_monitor->monitoring_scale.has_value());
  EXPECT_EQ(adjacent_monitor->monitoring_scale->degrees_of_freedom, 24U);
  EXPECT_DOUBLE_EQ(adjacent_monitor->monitoring_scale->calibrated_chi_squared_cutoff, 36.0);

  const GlobalGraphCheckpoint checkpoint = graph.checkpoint().value();
  ASSERT_EQ(checkpoint.loop_factors.size(), 1U);
  const LoopFactorCheckpoint& loop = checkpoint.loop_factors.front();
  EXPECT_EQ(loop.factor, GlobalFactorId(2U));
  EXPECT_EQ(loop.candidate, CandidateId(0U));
  EXPECT_EQ(loop.measurement.proposal, ProposalId(kProposal));
  EXPECT_EQ(checkpoint.next_factor_id, 3U);
  EXPECT_EQ(checkpoint.next_candidate_id, 1U);
}

}  // namespace
}  // namespace meridian::global
