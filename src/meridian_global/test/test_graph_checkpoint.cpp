#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numbers>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "../src/graph_checkpoint_internal.hpp"
#include "meridian/global/loop_consensus.hpp"
#include "sparse_seal_test_utils.hpp"

namespace meridian::global {
namespace {

[[nodiscard]] core::RankAwareInformation fullInformation(double value = 500.0) {
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
  header.trace = core::TraceId(1800U + proposal);
  header.producer = core::ProducerId(17U);
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
                                             fullInformation(),
                                             test_support::lineage(1800U + proposal)},
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

struct RecoveryFixture {
  GlobalGraphConfig config;
  core::SparseSubmapSeal first;
  core::SparseSubmapSeal second;
  GlobalGraphCheckpoint checkpoint;
  GlobalGraphCommit commit;
};

[[nodiscard]] RecoveryFixture makeRecoveryFixture() {
  RecoveryFixture fixture;
  fixture.config.loop_gnc.mu_step = 2.0;
  fixture.first = test_support::firstSeal();
  fixture.second = test_support::successor(fixture.first);
  GlobalGraph graph(fixture.config);
  (void)test_support::required(graph.initializeMission(fixture.first));
  (void)test_support::required(graph.appendAdjacent(fixture.second));

  const YawTranslation4 truth{{100.0, -30.0, 7.0}, 0.72};
  const std::array<Eigen::Vector3d, 6> antenna_points{
      Eigen::Vector3d{0.0, 0.0, 0.0},  Eigen::Vector3d{12.0, 0.0, 0.5},
      Eigen::Vector3d{0.0, 9.0, -0.2}, Eigen::Vector3d{6.0, -5.0, 1.0},
      Eigen::Vector3d{-4.0, 7.0, 0.3}, Eigen::Vector3d{15.0, 11.0, -0.6}};
  GnssBatchAppend gnss;
  gnss.initial_alignment = YawTranslation4{{98.0, -27.0, 5.0}, 0.5};
  for (std::size_t index = 0U; index < antenna_points.size(); ++index) {
    gnss.constraints.push_back(GnssAntennaConstraint{
        fixture.first.ref, core::GnssObservationId(index + 1U), antenna_points[index],
        truth.apply(antenna_points[index]), Eigen::Matrix3d::Identity() * 0.01});
  }
  (void)test_support::required(graph.appendGnssBatch(std::move(gnss)));

  constexpr std::uint64_t kProposal = 77U;
  RobustLoopBatchAppend loops;
  loops.expected_parent = test_support::required(graph.snapshot()).revision;
  loops.candidates.push_back(loopCandidate(kProposal, fixture.first, fixture.second));
  const auto loop_result = test_support::required(graph.appendRobustLoopBatch(
      std::move(loops), admittedConsensus(ProposalId(kProposal), core::FusionTime{20'000})));
  if (!loop_result.commit) {
    std::abort();
  }
  fixture.commit = *loop_result.commit;
  fixture.checkpoint = test_support::required(graph.checkpoint());
  return fixture;
}

void expectEquivalent(const GlobalGraphCommit& actual, const GlobalGraphCommit& expected) {
  ASSERT_EQ(actual.revision, expected.revision);
  ASSERT_EQ(actual.parent, expected.parent);
  ASSERT_EQ(actual.anchors.size(), expected.anchors.size());
  for (std::size_t index = 0U; index < actual.anchors.size(); ++index) {
    EXPECT_EQ(actual.anchors[index].submap, expected.anchors[index].submap);
    EXPECT_LT((expected.anchors[index].T_map_submap.inverse() * actual.anchors[index].T_map_submap)
                  .log()
                  .norm(),
              1.0e-12);
    EXPECT_LT((actual.anchors[index].covariance.matrix - expected.anchors[index].covariance.matrix)
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-12);
  }
  ASSERT_EQ(actual.alignment.has_value(), expected.alignment.has_value());
  if (actual.alignment) {
    EXPECT_LT((actual.alignment->translation_enu - expected.alignment->translation_enu).norm(),
              1.0e-12);
    EXPECT_NEAR(
        std::remainder(actual.alignment->yaw_enu_map_rad - expected.alignment->yaw_enu_map_rad,
                       2.0 * std::numbers::pi),
        0.0, 1.0e-12);
  }
  EXPECT_EQ(actual.map_odom.reference_submap, expected.map_odom.reference_submap);
  EXPECT_LT((expected.map_odom.T_map_odom.inverse() * actual.map_odom.T_map_odom).log().norm(),
            1.0e-12);
}

TEST(GlobalGraphCheckpoint, CanonicalBytesRoundTripAndRestoreCompleteGraph) {
  const RecoveryFixture fixture = makeRecoveryFixture();
  const GlobalGraphCheckpointLimits limits = checkpoint_internal::limitsForConfig(fixture.config);
  const auto encoded = encodeGlobalGraphCheckpoint(fixture.checkpoint, limits);
  ASSERT_TRUE(encoded) << encoded.error().detail;
  const auto decoded = decodeGlobalGraphCheckpoint(encoded.value().bytes(), limits);
  ASSERT_TRUE(decoded)
      << decoded.error().detail << " domain="
      << (decoded.error().canonical_verification_error
              ? static_cast<int>(decoded.error().canonical_verification_error->domain)
              : -1)
      << " failure="
      << (decoded.error().canonical_verification_error
              ? static_cast<int>(decoded.error().canonical_verification_error->failure)
              : -1);
  const auto reencoded = encodeGlobalGraphCheckpoint(decoded.value(), limits);
  ASSERT_TRUE(reencoded) << reencoded.error().detail;
  EXPECT_TRUE(std::ranges::equal(encoded.value().bytes(), reencoded.value().bytes()));
  EXPECT_EQ(decoded.value().checksum, fixture.checkpoint.checksum);

  GlobalGraph restored(fixture.config);
  const auto commit = restored.restoreCheckpoint(decoded.value());
  ASSERT_TRUE(commit) << commit.error().detail;
  expectEquivalent(commit.value(), fixture.commit);
  EXPECT_EQ(restored.checkpoint().value().checksum, fixture.checkpoint.checksum);
}

TEST(GlobalGraphCheckpoint, DecodeIsBoundedAndFailsClosedOnWireCorruption) {
  const RecoveryFixture fixture = makeRecoveryFixture();
  GlobalGraphCheckpointLimits limits = checkpoint_internal::limitsForConfig(fixture.config);
  const auto encoded = encodeGlobalGraphCheckpoint(fixture.checkpoint, limits);
  ASSERT_TRUE(encoded);
  std::vector<std::byte> bytes(encoded.value().bytes().begin(), encoded.value().bytes().end());

  ASSERT_GT(bytes.size(), 32U);
  bytes[bytes.size() / 2U] ^= std::byte{0x01};
  EXPECT_FALSE(decodeGlobalGraphCheckpoint(bytes, limits));

  std::vector<std::byte> truncated(encoded.value().bytes().begin(),
                                   encoded.value().bytes().end() - 1);
  const auto rejected_truncated = decodeGlobalGraphCheckpoint(truncated, limits);
  ASSERT_FALSE(rejected_truncated);
  EXPECT_TRUE(rejected_truncated.error().code == GlobalGraphCheckpointErrorCode::TruncatedRecord ||
              rejected_truncated.error().code == GlobalGraphCheckpointErrorCode::ChecksumMismatch);

  std::vector<std::byte> trailing(encoded.value().bytes().begin(), encoded.value().bytes().end());
  trailing.push_back(std::byte{0});
  const auto rejected_trailing = decodeGlobalGraphCheckpoint(trailing, limits);
  ASSERT_FALSE(rejected_trailing);
  EXPECT_EQ(rejected_trailing.error().code, GlobalGraphCheckpointErrorCode::TrailingBytes);

  limits.maximum_boundaries = 1U;
  const auto rejected_capacity = decodeGlobalGraphCheckpoint(encoded.value().bytes(), limits);
  ASSERT_FALSE(rejected_capacity);
  EXPECT_EQ(rejected_capacity.error().code, GlobalGraphCheckpointErrorCode::CapacityExceeded);
}

TEST(GlobalGraphCheckpoint, TypedDomainTamperingAndNoncanonicalOrderAreAtomic) {
  const RecoveryFixture fixture = makeRecoveryFixture();
  const std::array<GlobalGraphCheckpoint, 7> tampered = [&]() {
    std::array<GlobalGraphCheckpoint, 7> cases;
    cases.fill(fixture.checkpoint);
    cases[0].configuration.checksum.front() ^= 0x01U;
    cases[1].boundaries[1].seal.seal_checksum.front() ^= 0x01U;
    cases[2].adjacent_factors[0].transition.checksum.front() ^= 0x01U;
    cases[3].gnss_factors[0].constraint.measured_position_enu.x() += 1.0;
    cases[4].loop_factors[0].measurement.T_from_to.translation().x() += 1.0;
    cases[5].recovery.boundary_marginals[0].covariance.matrix(0, 0) += 1.0;
    std::swap(cases[6].factor_order[0], cases[6].factor_order[1]);
    return cases;
  }();

  for (const GlobalGraphCheckpoint& candidate : tampered) {
    GlobalGraph graph(fixture.config);
    const auto rejected = graph.restoreCheckpoint(candidate);
    EXPECT_FALSE(rejected);
    EXPECT_FALSE(graph.initialized());
    const auto fresh = graph.initializeMission(fixture.first);
    ASSERT_TRUE(fresh) << fresh.error().detail;
    EXPECT_EQ(graph.checkpoint().value().mission_gauge.factor, GlobalFactorId(0U));
  }
}

TEST(GlobalGraphCheckpoint, UnsupportedSchemaAndConfigurationMismatchFailBeforePublication) {
  const RecoveryFixture fixture = makeRecoveryFixture();
  GlobalGraphCheckpoint unsupported = fixture.checkpoint;
  ++unsupported.schema_version;
  GlobalGraph unsupported_graph(fixture.config);
  const auto rejected_schema = unsupported_graph.restoreCheckpoint(unsupported);
  ASSERT_FALSE(rejected_schema);
  EXPECT_EQ(rejected_schema.error().code, GlobalGraphCheckpointErrorCode::UnsupportedSchema);
  EXPECT_FALSE(unsupported_graph.initialized());

  GlobalGraphConfig changed = fixture.config;
  ++changed.maximum_anchors;
  GlobalGraph changed_graph(changed);
  const auto rejected_config = changed_graph.restoreCheckpoint(fixture.checkpoint);
  ASSERT_FALSE(rejected_config);
  EXPECT_EQ(rejected_config.error().code, GlobalGraphCheckpointErrorCode::ConfigurationMismatch);
  EXPECT_FALSE(changed_graph.initialized());
}

TEST(GlobalGraphCheckpoint, ObjectiveMismatchWithValidChecksumFailsAtomically) {
  const RecoveryFixture fixture = makeRecoveryFixture();
  GlobalGraphCheckpoint altered = fixture.checkpoint;
  altered.recovery.factor_objectives.back().whitened_squared_cost += 2.0;
  altered.recovery.whitened_squared_objective += 2.0;
  const auto finalized = checkpoint_internal::finalize(
      std::move(altered), checkpoint_internal::limitsForConfig(fixture.config));
  ASSERT_TRUE(finalized) << finalized.error().detail;

  GlobalGraph graph(fixture.config);
  const auto rejected = graph.restoreCheckpoint(finalized.value());
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GlobalGraphCheckpointErrorCode::ObjectiveMismatch);
  EXPECT_FALSE(graph.initialized());
}

TEST(GlobalGraphCheckpoint, LowRankReconstructionWithValidChecksumsFailsAtomically) {
  const RecoveryFixture fixture = makeRecoveryFixture();
  GlobalGraphCheckpoint altered = fixture.checkpoint;
  core::SparseSubmapSeal successor = altered.boundaries[1].seal;
  std::vector<std::uint32_t> columns;
  for (std::uint32_t column = 6U; column <= 28U; ++column) {
    columns.push_back(column);
  }
  test_support::replaceBoundaryFactor(
      &successor,
      test_support::boundaryFactorOnColumns(altered.boundaries[0].seal.boundary_navigation.state,
                                            successor.boundary_navigation.state, columns));
  altered.boundaries[1].seal = successor;
  altered.adjacent_factors[0].transition = *successor.from_previous;
  const auto finalized = checkpoint_internal::finalize(
      std::move(altered), checkpoint_internal::limitsForConfig(fixture.config));
  ASSERT_TRUE(finalized) << finalized.error().detail;

  GlobalGraph graph(fixture.config);
  const auto rejected = graph.restoreCheckpoint(finalized.value());
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GlobalGraphCheckpointErrorCode::RankMismatch)
      << rejected.error().detail;
  EXPECT_FALSE(graph.initialized());
}

TEST(GlobalGraphCheckpoint, RestoredRevisionAndIdentityAllocatorsContinueWithoutConsumption) {
  const RecoveryFixture fixture = makeRecoveryFixture();
  GlobalGraph graph(fixture.config);
  ASSERT_TRUE(graph.restoreCheckpoint(fixture.checkpoint));
  const GlobalGraphCheckpoint before = graph.checkpoint().value();

  constexpr std::uint64_t kNextProposal = 78U;
  RobustLoopBatchAppend loops;
  loops.expected_parent = graph.snapshot().value().revision;
  loops.candidates.push_back(loopCandidate(kNextProposal, fixture.first, fixture.second));
  const auto result = graph.appendRobustLoopBatch(
      std::move(loops), admittedConsensus(ProposalId(kNextProposal), core::FusionTime{30'000}));
  ASSERT_TRUE(result) << result.error().detail;
  ASSERT_TRUE(result.value().commit.has_value());
  EXPECT_EQ(result.value().commit->revision, GlobalGraphRevision(before.revision.value() + 1U));
  const GlobalGraphCheckpoint after = graph.checkpoint().value();
  const auto added = std::find_if(after.loop_factors.begin(), after.loop_factors.end(),
                                  [](const LoopFactorCheckpoint& loop) {
                                    return loop.measurement.proposal == ProposalId(kNextProposal);
                                  });
  ASSERT_NE(added, after.loop_factors.end());
  EXPECT_EQ(added->factor, GlobalFactorId(before.next_factor_id));
  EXPECT_EQ(added->candidate, CandidateId(before.next_candidate_id));
  EXPECT_EQ(after.next_factor_id, before.next_factor_id + 1U);
  EXPECT_EQ(after.next_candidate_id, before.next_candidate_id + 1U);
}

}  // namespace
}  // namespace meridian::global
