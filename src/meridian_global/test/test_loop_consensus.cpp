#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "meridian/global/loop_consensus.hpp"

namespace meridian::global {
namespace {

constexpr std::int64_t kSecondNs = 1'000'000'000LL;
constexpr double kChiSquaredGate = 12.592;

[[nodiscard]] core::SubmapRef submap(std::uint64_t id) {
  core::ContentHash hash{};
  hash[0] = static_cast<std::uint8_t>(id + 1U);
  return core::SubmapRef{core::SessionId(9U), core::OdomEpoch(1U), core::SubmapId(id),
                         core::CalibrationEpoch(1U), core::SubmapContentRevision(0U), hash};
}

[[nodiscard]] core::RankAwareInformation fullInformation() {
  core::RankAwareInformation information;
  information.basis.setIdentity();
  information.eigenvalues.setOnes();
  information.rank = 6U;
  return information;
}

[[nodiscard]] core::ObservationLineage lineage(
    std::uint64_t lineage_id, std::uint64_t measurement_id,
    std::optional<core::CorrelationTreatment> treatment = std::nullopt) {
  core::ObservationSlice slice;
  slice.root = core::MeasurementId(measurement_id);
  slice.kind = core::SliceKind::Whole;
  slice.calibration = core::CalibrationEpoch(1U);
  slice.source_checksum[0] = static_cast<std::uint8_t>(measurement_id & 0xffU);

  core::ObservationUsage usage;
  usage.slice = slice;
  usage.role = core::ObservationRole::PrimaryResidual;
  usage.consumer = core::DerivedRecordId(lineage_id);
  usage.factor_group = core::FactorGroupId(lineage_id);

  core::ObservationLineage result;
  result.id = core::ObservationLineageId(lineage_id);
  result.usage.push_back(usage);
  result.checksum[0] = static_cast<std::uint8_t>(lineage_id & 0xffU);
  if (treatment) {
    result.usage.front().correlation_group = core::CorrelationGroupId(77U);
    core::CorrelationDeclaration declaration;
    declaration.group = core::CorrelationGroupId(77U);
    declaration.policy = core::CorrelationPolicyRevision(2U);
    declaration.treatment = *treatment;
    declaration.covariance_inflation = 1.4;
    if (*treatment == core::CorrelationTreatment::CovarianceInflationAndInformationCap) {
      declaration.total_information_cap = 25.0;
    }
    result.correlations.push_back(declaration);
  }
  return result;
}

[[nodiscard]] LoopMeasurement proposal(
    std::uint64_t id, double created_s, std::uint64_t measurement_id, std::uint64_t from_id = 100U,
    std::uint64_t to_id = 200U, LoopModality modality = LoopModality::Visual,
    std::optional<core::CorrelationTreatment> treatment = std::nullopt) {
  core::RecordHeader header;
  header.schema_version = 1U;
  header.trace = core::TraceId(id);
  header.producer = core::ProducerId(3U);
  header.session = core::SessionId(9U);
  header.created_at = core::FusionTime{static_cast<std::int64_t>(created_s * 1.0e9)};
  header.config = core::ConfigRevision(4U);

  return LoopMeasurement{header,
                         ProposalId(id),
                         modality,
                         submap(from_id),
                         submap(to_id),
                         {core::CalibrationEpoch(1U)},
                         core::Pose3d(Sophus::SO3d{}, Eigen::Vector3d{1.0, 0.0, 0.0}),
                         fullInformation(),
                         lineage(id, measurement_id, treatment)};
}

[[nodiscard]] PairwiseConsistencyInput pair(std::uint64_t first, std::uint64_t second,
                                            double residual = 0.1, std::size_t rank = 6U,
                                            double threshold = kChiSquaredGate) {
  PairwiseConsistencyInput input;
  input.first = ProposalId(first);
  input.second = ProposalId(second);
  input.cycle_residual.setZero();
  input.cycle_residual(0) = residual;
  input.common_basis.setIdentity();
  input.common_covariance_eigenvalues.setZero();
  input.common_covariance_eigenvalues.head(static_cast<Eigen::Index>(rank)).setOnes();
  input.common_rank = rank;
  input.chi_squared_threshold = threshold;
  return input;
}

[[nodiscard]] const LoopProposalDecision& decision(const LoopConsensusReport& report,
                                                   std::uint64_t id) {
  const auto found = std::find_if(
      report.decisions.begin(), report.decisions.end(),
      [id](const LoopProposalDecision& item) { return item.proposal == ProposalId(id); });
  EXPECT_NE(found, report.decisions.end());
  return *found;
}

[[nodiscard]] std::vector<std::uint64_t> values(const std::vector<ProposalId>& ids) {
  std::vector<std::uint64_t> output;
  output.reserve(ids.size());
  for (const ProposalId id : ids) {
    output.push_back(id.value());
  }
  return output;
}

class FixedAncestryApi final : public LoopAncestryApi {
public:
  explicit FixedAncestryApi(LoopAncestryAssessment assessment) : assessment_(assessment) {}

  [[nodiscard]] LoopAncestryAssessment assess(const LoopMeasurement&,
                                              const LoopMeasurement&) const noexcept override {
    return assessment_;
  }

private:
  LoopAncestryAssessment assessment_;
};

TEST(LoopConsensusRankAware, ProjectsOnlyCommonSupportAndMarksInsufficientRankUndefined) {
  const std::vector<LoopMeasurement> proposals{proposal(1U, 1.0, 11U), proposal(2U, 1.1, 12U)};
  PairwiseConsistencyInput supported = pair(1U, 2U, 0.0, 2U, 5.991);
  supported.cycle_residual << 0.2, -0.1, 1.0e6, -1.0e6, 4.0e5, -3.0e5;
  const CoreLoopAncestryApi ancestry;
  const auto resolved =
      LoopConsensus{}.resolve(proposals, std::span<const PairwiseConsistencyInput>(&supported, 1U),
                              core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_TRUE(resolved) << resolved.error().detail;
  ASSERT_EQ(resolved.value().pairs.size(), 1U);
  EXPECT_EQ(resolved.value().pairs.front().relation, PairwiseRelation::Consistent);
  ASSERT_TRUE(resolved.value().pairs.front().squared_mahalanobis);
  EXPECT_NEAR(*resolved.value().pairs.front().squared_mahalanobis, 0.05, 1.0e-12);
  EXPECT_EQ(decision(resolved.value(), 1U).disposition,
            LoopProposalDisposition::AdmittedToGncCandidateBatch);
  EXPECT_EQ(decision(resolved.value(), 2U).disposition,
            LoopProposalDisposition::AdmittedToGncCandidateBatch);

  PairwiseConsistencyInput insufficient = pair(1U, 2U, 0.0, 1U, 0.0);
  const auto undefined = LoopConsensus{}.resolve(
      proposals, std::span<const PairwiseConsistencyInput>(&insufficient, 1U),
      core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_TRUE(undefined);
  EXPECT_EQ(undefined.value().pairs.front().relation,
            PairwiseRelation::UndefinedInsufficientCommonRank);
  EXPECT_EQ(undefined.value().pairs.front().disposition,
            PairwiseEdgeDisposition::NoEdgeInsufficientCommonRank);
  EXPECT_EQ(decision(undefined.value(), 1U).disposition,
            LoopProposalDisposition::RejectedSingleton);
  EXPECT_EQ(decision(undefined.value(), 2U).disposition,
            LoopProposalDisposition::RejectedSingleton);

  PairwiseConsistencyInput inconsistent = pair(1U, 2U, 4.0, 2U, 5.991);
  const auto gated = LoopConsensus{}.resolve(
      proposals, std::span<const PairwiseConsistencyInput>(&inconsistent, 1U),
      core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_TRUE(gated);
  EXPECT_EQ(gated.value().pairs.front().relation, PairwiseRelation::Inconsistent);
  EXPECT_EQ(gated.value().pairs.front().disposition, PairwiseEdgeDisposition::NoEdgeInconsistent);
  ASSERT_TRUE(gated.value().pairs.front().squared_mahalanobis);
  EXPECT_NEAR(*gated.value().pairs.front().squared_mahalanobis, 16.0, 1.0e-12);
}

TEST(LoopConsensusClique, FindsExactLexicographicallySmallestMaximumIndependentOfInputOrder) {
  std::vector<LoopMeasurement> proposals{proposal(40U, 1.4, 140U), proposal(10U, 1.1, 110U),
                                         proposal(30U, 1.3, 130U), proposal(20U, 1.2, 120U)};
  std::vector<PairwiseConsistencyInput> pairs{pair(40U, 20U), pair(30U, 10U), pair(20U, 10U),
                                              pair(40U, 10U), pair(30U, 20U)};
  const CoreLoopAncestryApi ancestry;
  const auto first =
      LoopConsensus{}.resolve(proposals, pairs, core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_TRUE(first) << first.error().detail;
  ASSERT_EQ(first.value().components.size(), 1U);
  EXPECT_EQ(values(first.value().components.front().maximum_clique),
            (std::vector<std::uint64_t>{10U, 20U, 30U}));
  EXPECT_EQ(decision(first.value(), 40U).disposition,
            LoopProposalDisposition::RejectedOutsideMaximumClique);
  EXPECT_GT(first.value().components.front().exact_search_expansions, 0U);

  std::reverse(proposals.begin(), proposals.end());
  std::reverse(pairs.begin(), pairs.end());
  const auto second =
      LoopConsensus{}.resolve(proposals, pairs, core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_TRUE(second);
  EXPECT_EQ(values(second.value().components.front().maximum_clique),
            values(first.value().components.front().maximum_clique));
  ASSERT_EQ(second.value().pairs.size(), first.value().pairs.size());
  for (std::size_t index = 0U; index < first.value().pairs.size(); ++index) {
    EXPECT_EQ(second.value().pairs[index].first, first.value().pairs[index].first);
    EXPECT_EQ(second.value().pairs[index].second, first.value().pairs[index].second);
  }
}

TEST(LoopConsensusComponents, OrdersComponentsByOldestTimeThenSmallestProposalId) {
  const std::vector<LoopMeasurement> proposals{proposal(50U, 5.0, 150U), proposal(60U, 6.0, 160U),
                                               proposal(10U, 3.0, 110U), proposal(20U, 4.0, 120U),
                                               proposal(30U, 3.0, 130U), proposal(40U, 4.5, 140U)};
  const std::vector<PairwiseConsistencyInput> pairs{pair(50U, 60U), pair(10U, 20U), pair(30U, 40U)};
  const CoreLoopAncestryApi ancestry;
  const auto resolved =
      LoopConsensus{}.resolve(proposals, pairs, core::FusionTime{7 * kSecondNs}, ancestry);
  ASSERT_TRUE(resolved);
  ASSERT_EQ(resolved.value().components.size(), 3U);
  EXPECT_EQ(resolved.value().components[0].smallest_proposal_id, ProposalId(10U));
  EXPECT_EQ(resolved.value().components[1].smallest_proposal_id, ProposalId(30U));
  EXPECT_EQ(resolved.value().components[2].smallest_proposal_id, ProposalId(50U));
}

TEST(LoopConsensusDisposition, RejectsSingletonAndRecordsTtlExpirySeparately) {
  const std::vector<LoopMeasurement> proposals{proposal(1U, 1.0, 11U), proposal(2U, 35.0, 12U)};
  const PairwiseConsistencyInput pair_with_expired = pair(1U, 2U);
  const CoreLoopAncestryApi ancestry;
  const auto resolved = LoopConsensus{}.resolve(
      proposals, std::span<const PairwiseConsistencyInput>(&pair_with_expired, 1U),
      core::FusionTime{40 * kSecondNs}, ancestry);
  ASSERT_TRUE(resolved);
  ASSERT_EQ(resolved.value().pairs.size(), 1U);
  EXPECT_EQ(resolved.value().pairs.front().disposition,
            PairwiseEdgeDisposition::IgnoredExpiredEndpoint);
  ASSERT_EQ(resolved.value().components.size(), 1U);
  EXPECT_EQ(resolved.value().components.front().smallest_proposal_id, ProposalId(2U));
  EXPECT_EQ(resolved.value().components.front().disposition,
            PcmComponentDisposition::RejectedSingleton);
  EXPECT_EQ(decision(resolved.value(), 1U).disposition,
            LoopProposalDisposition::RejectedExpiredTtl);
  EXPECT_FALSE(decision(resolved.value(), 1U).component_index);
  EXPECT_EQ(decision(resolved.value(), 2U).disposition, LoopProposalDisposition::RejectedSingleton);
}

TEST(LoopConsensusResources, DefersEntireOversizedComponentWithoutPrefixSelection) {
  LoopConsensusConfig config;
  config.maximum_component_vertices = 2U;
  const std::vector<LoopMeasurement> proposals{proposal(1U, 1.0, 11U), proposal(2U, 1.1, 12U),
                                               proposal(3U, 1.2, 13U)};
  const std::vector<PairwiseConsistencyInput> pairs{pair(1U, 2U), pair(2U, 3U)};
  const CoreLoopAncestryApi ancestry;
  const auto resolved =
      LoopConsensus(config).resolve(proposals, pairs, core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_TRUE(resolved);
  ASSERT_EQ(resolved.value().components.size(), 1U);
  const PcmComponentReport& component = resolved.value().components.front();
  EXPECT_EQ(component.disposition, PcmComponentDisposition::DeferredResourceLimit);
  EXPECT_EQ(component.resource_limit, PcmResourceLimit::ComponentVertices);
  EXPECT_TRUE(component.maximum_clique.empty());
  EXPECT_EQ(component.exact_search_expansions, 0U);
  for (const LoopProposalDecision& item : resolved.value().decisions) {
    EXPECT_EQ(item.disposition, LoopProposalDisposition::DeferredResourceLimit);
    EXPECT_EQ(item.resource_limit, PcmResourceLimit::ComponentVertices);
  }
}

TEST(LoopConsensusResources, DefersOnEdgeAndExactSearchLimits) {
  const std::vector<LoopMeasurement> proposals{proposal(1U, 1.0, 11U), proposal(2U, 1.1, 12U),
                                               proposal(3U, 1.2, 13U)};
  const std::vector<PairwiseConsistencyInput> triangle{pair(1U, 2U), pair(1U, 3U), pair(2U, 3U)};
  const CoreLoopAncestryApi ancestry;

  LoopConsensusConfig edge_config;
  edge_config.maximum_component_edges = 2U;
  const auto edge_limited =
      LoopConsensus(edge_config)
          .resolve(proposals, triangle, core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_TRUE(edge_limited);
  EXPECT_EQ(edge_limited.value().components.front().resource_limit,
            PcmResourceLimit::ComponentEdges);

  LoopConsensusConfig search_config;
  search_config.maximum_exact_clique_expansions = 1U;
  const std::array<LoopMeasurement, 2> two{proposal(1U, 1.0, 11U), proposal(2U, 1.1, 12U)};
  const PairwiseConsistencyInput connected = pair(1U, 2U);
  const auto search_limited =
      LoopConsensus(search_config)
          .resolve(two, std::span<const PairwiseConsistencyInput>(&connected, 1U),
                   core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_TRUE(search_limited);
  EXPECT_EQ(search_limited.value().components.front().resource_limit,
            PcmResourceLimit::ExactCliqueExpansions);
  EXPECT_TRUE(search_limited.value().components.front().maximum_clique.empty());
  for (const LoopProposalDecision& item : search_limited.value().decisions) {
    EXPECT_EQ(item.disposition, LoopProposalDisposition::DeferredResourceLimit);
  }
}

TEST(LoopConsensusAncestry, OverlappingAndUnknownLineageNeverCreateConsistencyEdges) {
  const std::vector<LoopMeasurement> overlapping{proposal(1U, 1.0, 99U), proposal(2U, 1.1, 99U)};
  const PairwiseConsistencyInput consistent = pair(1U, 2U);
  const CoreLoopAncestryApi core_ancestry;
  const auto overlap = LoopConsensus{}.resolve(
      overlapping, std::span<const PairwiseConsistencyInput>(&consistent, 1U),
      core::FusionTime{2 * kSecondNs}, core_ancestry);
  ASSERT_TRUE(overlap);
  EXPECT_EQ(overlap.value().pairs.front().disposition,
            PairwiseEdgeDisposition::NoEdgeNotIndependent);
  EXPECT_EQ(decision(overlap.value(), 1U).disposition, LoopProposalDisposition::RejectedSingleton);

  const std::vector<LoopMeasurement> independent{proposal(1U, 1.0, 11U), proposal(2U, 1.1, 12U)};
  const FixedAncestryApi unknown(
      {AncestryIndependence::Unknown, PairCorrelationTreatment::NotRequired});
  const auto unresolved = LoopConsensus{}.resolve(
      independent, std::span<const PairwiseConsistencyInput>(&consistent, 1U),
      core::FusionTime{2 * kSecondNs}, unknown);
  ASSERT_TRUE(unresolved);
  EXPECT_EQ(unresolved.value().pairs.front().disposition,
            PairwiseEdgeDisposition::NoEdgeUnknownAncestry);

  const FixedAncestryApi dishonest(
      {AncestryIndependence::Independent, PairCorrelationTreatment::NotRequired});
  const auto contradicted = LoopConsensus{}.resolve(
      overlapping, std::span<const PairwiseConsistencyInput>(&consistent, 1U),
      core::FusionTime{2 * kSecondNs}, dishonest);
  ASSERT_FALSE(contradicted);
  EXPECT_EQ(contradicted.error().code, LoopConsensusErrorCode::InconsistentAncestryAssessment);
}

TEST(LoopConsensusAncestry, CrossModalSameEndpointsRequireDeclaredPairTreatment) {
  const PairwiseConsistencyInput consistent = pair(1U, 2U);
  const CoreLoopAncestryApi ancestry;
  const std::vector<LoopMeasurement> untreated{
      proposal(1U, 1.0, 11U, 100U, 200U, LoopModality::Visual),
      proposal(2U, 1.1, 12U, 200U, 100U, LoopModality::Lidar)};
  const auto blocked =
      LoopConsensus{}.resolve(untreated, std::span<const PairwiseConsistencyInput>(&consistent, 1U),
                              core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_TRUE(blocked);
  EXPECT_EQ(blocked.value().pairs.front().disposition,
            PairwiseEdgeDisposition::NoEdgeMissingPairTreatment);

  constexpr auto treatment = core::CorrelationTreatment::CovarianceInflationAndInformationCap;
  const std::vector<LoopMeasurement> treated{
      proposal(1U, 1.0, 11U, 100U, 200U, LoopModality::Visual, treatment),
      proposal(2U, 1.1, 12U, 200U, 100U, LoopModality::Lidar, treatment)};
  const auto admitted =
      LoopConsensus{}.resolve(treated, std::span<const PairwiseConsistencyInput>(&consistent, 1U),
                              core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_TRUE(admitted) << admitted.error().detail;
  EXPECT_EQ(admitted.value().pairs.front().disposition, PairwiseEdgeDisposition::ConsistencyEdge);
  EXPECT_EQ(admitted.value().pairs.front().ancestry.pair_treatment,
            PairCorrelationTreatment::CovarianceInflationAndInformationCap);
  EXPECT_EQ(decision(admitted.value(), 1U).disposition,
            LoopProposalDisposition::AdmittedToGncCandidateBatch);
}

TEST(LoopConsensusValidation, RejectsDuplicateReversedPairsUnknownIdsAndUnboundedPsdRepair) {
  const std::vector<LoopMeasurement> proposals{proposal(1U, 1.0, 11U), proposal(2U, 1.1, 12U)};
  const CoreLoopAncestryApi ancestry;
  const std::vector<PairwiseConsistencyInput> duplicates{pair(1U, 2U), pair(2U, 1U)};
  const auto duplicate =
      LoopConsensus{}.resolve(proposals, duplicates, core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, LoopConsensusErrorCode::DuplicatePairwiseInput);

  const PairwiseConsistencyInput unknown = pair(1U, 3U);
  const auto missing =
      LoopConsensus{}.resolve(proposals, std::span<const PairwiseConsistencyInput>(&unknown, 1U),
                              core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code, LoopConsensusErrorCode::UnknownPairwiseProposal);

  PairwiseConsistencyInput repaired = pair(1U, 2U);
  repaired.psd_roundoff_repaired = true;
  repaired.maximum_psd_clamp = 1.0e-5;
  const auto excessive =
      LoopConsensus{}.resolve(proposals, std::span<const PairwiseConsistencyInput>(&repaired, 1U),
                              core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_FALSE(excessive);
  EXPECT_EQ(excessive.error().code, LoopConsensusErrorCode::InvalidPairwiseInput);
}

TEST(LoopConsensusValidation, EnforcesBatchAndConfigurationCapacities) {
  const std::vector<LoopMeasurement> proposals{proposal(1U, 1.0, 11U), proposal(2U, 1.1, 12U)};
  const PairwiseConsistencyInput consistent = pair(1U, 2U);
  const CoreLoopAncestryApi ancestry;

  LoopConsensusConfig proposal_cap;
  proposal_cap.maximum_proposals = 1U;
  proposal_cap.maximum_component_vertices = 1U;
  const auto too_many_proposals =
      LoopConsensus(proposal_cap).resolve(proposals, {}, core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_FALSE(too_many_proposals);
  EXPECT_EQ(too_many_proposals.error().code, LoopConsensusErrorCode::ProposalCapacity);

  LoopConsensusConfig pair_cap;
  pair_cap.maximum_pairwise_inputs = 0U;
  const auto invalid_config = LoopConsensus(pair_cap).resolve(
      proposals, std::span<const PairwiseConsistencyInput>(&consistent, 1U),
      core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_FALSE(invalid_config);
  EXPECT_EQ(invalid_config.error().code, LoopConsensusErrorCode::InvalidConfig);

  pair_cap.maximum_pairwise_inputs = 1U;
  const std::array<PairwiseConsistencyInput, 2> over_pair_cap{consistent, pair(2U, 1U)};
  const auto too_many_pairs = LoopConsensus(pair_cap).resolve(
      proposals, over_pair_cap, core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_FALSE(too_many_pairs);
  EXPECT_EQ(too_many_pairs.error().code, LoopConsensusErrorCode::PairwiseInputCapacity);
}

TEST(LoopConsensusValidation, RejectsDuplicateIdsMixedSessionsAndMalformedLineageBeforePcm) {
  const CoreLoopAncestryApi ancestry;
  const std::vector<LoopMeasurement> duplicates{proposal(1U, 1.0, 11U), proposal(1U, 1.1, 12U)};
  const auto duplicate =
      LoopConsensus{}.resolve(duplicates, {}, core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, LoopConsensusErrorCode::DuplicateProposalId);

  std::vector<LoopMeasurement> mixed{proposal(1U, 1.0, 11U), proposal(2U, 1.1, 12U)};
  mixed.back().header.session = core::SessionId(10U);
  mixed.back().from.session = core::SessionId(10U);
  mixed.back().to.session = core::SessionId(10U);
  const auto mixed_session =
      LoopConsensus{}.resolve(mixed, {}, core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_FALSE(mixed_session);
  EXPECT_EQ(mixed_session.error().code, LoopConsensusErrorCode::MixedSessionBatch);

  std::vector<LoopMeasurement> malformed{proposal(1U, 1.0, 11U)};
  core::ObservationUsage duplicate_usage = malformed.front().lineage.usage.front();
  duplicate_usage.consumer = core::DerivedRecordId(55U);
  duplicate_usage.factor_group = core::FactorGroupId(55U);
  malformed.front().lineage.usage.push_back(duplicate_usage);
  const auto invalid_lineage =
      LoopConsensus{}.resolve(malformed, {}, core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_FALSE(invalid_lineage);
  EXPECT_EQ(invalid_lineage.error().code, LoopConsensusErrorCode::InvalidLineage);
}

TEST(LoopConsensusValidation, RejectsEndpointSessionCalibrationAndChecksumDrift) {
  const CoreLoopAncestryApi ancestry;

  auto wrong_session = proposal(1U, 1.0, 11U);
  wrong_session.to.session = core::SessionId(10U);
  auto rejected = LoopConsensus{}.resolve(std::span<const LoopMeasurement>(&wrong_session, 1U), {},
                                          core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LoopConsensusErrorCode::InvalidProposal);

  auto missing_calibration = proposal(2U, 1.0, 12U);
  missing_calibration.calibration_epochs = {core::CalibrationEpoch(2U)};
  rejected = LoopConsensus{}.resolve(std::span<const LoopMeasurement>(&missing_calibration, 1U),
                                     {}, core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LoopConsensusErrorCode::InvalidProposal);

  auto missing_checksum = proposal(3U, 1.0, 13U);
  missing_checksum.from.local_content_checksum = {};
  rejected = LoopConsensus{}.resolve(std::span<const LoopMeasurement>(&missing_checksum, 1U), {},
                                     core::FusionTime{2 * kSecondNs}, ancestry);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LoopConsensusErrorCode::InvalidProposal);
}

}  // namespace
}  // namespace meridian::global
