#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/global/graph.hpp"

namespace meridian::global {

// Rank-aware result of the upstream two-loop cycle construction. The upstream
// verifier/chain API owns frame transport and covariance propagation; PCM does
// not reconstruct unavailable chain covariance from current graph estimates.
// Columns [0, common_rank) span the common supported right,
// translation-first cycle tangent. Their corresponding covariance eigenvalues
// are positive; unsupported eigenvalues are zero.
struct PairwiseConsistencyInput {
  ProposalId first;
  ProposalId second;
  Eigen::Matrix<double, 6, 1> cycle_residual{Eigen::Matrix<double, 6, 1>::Zero()};
  Eigen::Matrix<double, 6, 6> common_basis{Eigen::Matrix<double, 6, 6>::Identity()};
  Eigen::Matrix<double, 6, 1> common_covariance_eigenvalues{Eigen::Matrix<double, 6, 1>::Zero()};
  std::size_t common_rank{};
  double chi_squared_threshold{};
  bool psd_roundoff_repaired{false};
  double maximum_psd_clamp{};
  core::PoseTangentConvention tangent{core::PoseTangentConvention::RightTranslationFirst};
};

enum class AncestryIndependence {
  Independent,
  NotIndependent,
  Unknown,
};

enum class PairCorrelationTreatment {
  NotRequired,
  CovarianceInflationAndInformationCap,
  JointCompositeWhitening,
  Missing,
};

struct LoopAncestryAssessment {
  AncestryIndependence independence{AncestryIndependence::Unknown};
  PairCorrelationTreatment pair_treatment{PairCorrelationTreatment::Missing};
};

// Replaceable seam for transitive ancestry resolution. An implementation may
// conservatively return Unknown, but it may never claim Independent when the
// canonical raw-root slices overlap.
class LoopAncestryApi {
public:
  virtual ~LoopAncestryApi() = default;

  [[nodiscard]] virtual LoopAncestryAssessment assess(
      const LoopMeasurement& first, const LoopMeasurement& second) const noexcept = 0;
};

// Default implementation for the currently decidable core lineage selectors.
// Cross-modal proposals at identical endpoint pairs additionally require a
// shared declared pair treatment.
class CoreLoopAncestryApi final : public LoopAncestryApi {
public:
  [[nodiscard]] LoopAncestryAssessment assess(
      const LoopMeasurement& first, const LoopMeasurement& second) const noexcept override;
};

enum class PairwiseRelation {
  Consistent,
  Inconsistent,
  UndefinedInsufficientCommonRank,
};

enum class PairwiseEdgeDisposition {
  ConsistencyEdge,
  NoEdgeInconsistent,
  NoEdgeInsufficientCommonRank,
  NoEdgeNotIndependent,
  NoEdgeUnknownAncestry,
  NoEdgeMissingPairTreatment,
  IgnoredExpiredEndpoint,
};

struct PairwiseConsensusReport {
  ProposalId first;
  ProposalId second;
  PairwiseConsistencyInput input;
  PairwiseRelation relation{PairwiseRelation::UndefinedInsufficientCommonRank};
  std::optional<double> squared_mahalanobis;
  LoopAncestryAssessment ancestry;
  PairwiseEdgeDisposition disposition{PairwiseEdgeDisposition::NoEdgeInsufficientCommonRank};
};

enum class PcmResourceLimit {
  ComponentVertices,
  ComponentEdges,
  ExactCliqueExpansions,
};

enum class PcmComponentDisposition {
  ResolvedMaximumClique,
  RejectedSingleton,
  DeferredResourceLimit,
};

struct PcmComponentReport {
  core::FusionTime oldest_proposal_time;
  ProposalId smallest_proposal_id;
  std::vector<ProposalId> proposals;
  std::size_t consistency_edges{};
  std::vector<ProposalId> maximum_clique;
  std::uint64_t exact_search_expansions{};
  PcmComponentDisposition disposition{PcmComponentDisposition::RejectedSingleton};
  std::optional<PcmResourceLimit> resource_limit;
};

enum class LoopProposalDisposition {
  AdmittedToGncCandidateBatch,
  RejectedOutsideMaximumClique,
  RejectedSingleton,
  DeferredResourceLimit,
  RejectedExpiredTtl,
};

struct LoopProposalDecision {
  ProposalId proposal;
  LoopProposalDisposition disposition{LoopProposalDisposition::RejectedSingleton};
  std::optional<std::size_t> component_index;
  std::optional<PcmResourceLimit> resource_limit;
};

struct LoopConsensusReport {
  core::FusionTime evaluated_at;
  std::vector<PairwiseConsensusReport> pairs;
  // Ordered by (oldest_proposal_time, smallest_proposal_id).
  std::vector<PcmComponentReport> components;
  // Ordered by proposal ID for stable downstream joins.
  std::vector<LoopProposalDecision> decisions;
};

struct LoopConsensusConfig {
  std::size_t maximum_proposals{256U};
  std::size_t maximum_pairwise_inputs{32'640U};
  std::size_t maximum_component_vertices{64U};
  std::size_t maximum_component_edges{2'016U};
  std::uint64_t maximum_exact_clique_expansions{1'000'000U};
  core::Duration proposal_ttl{30'000'000'000LL};
  std::size_t minimum_common_rank{2U};

  std::size_t maximum_calibration_epochs_per_proposal{8U};
  std::size_t maximum_lineage_usages_per_proposal{4'096U};
  std::size_t maximum_lineage_correlations_per_proposal{256U};

  double information_basis_orthonormal_tolerance{1.0e-8};
  double information_zero_tolerance{1.0e-12};
  double minimum_supported_value{1.0e-12};
  double maximum_psd_roundoff_clamp{1.0e-10};
};

enum class LoopConsensusErrorCode {
  InvalidConfig,
  ProposalCapacity,
  PairwiseInputCapacity,
  InvalidProposal,
  DuplicateProposalId,
  MixedSessionBatch,
  InvalidLineage,
  InvalidPairwiseInput,
  DuplicatePairwiseInput,
  UnknownPairwiseProposal,
  InconsistentAncestryAssessment,
};

struct LoopConsensusError {
  LoopConsensusErrorCode code{LoopConsensusErrorCode::InvalidConfig};
  std::optional<ProposalId> first;
  std::optional<ProposalId> second;
  std::optional<std::size_t> input_index;
  std::string detail;
};

// Stateless, ROS-free PCM batch resolver. It selects GNC candidates only; it
// neither mutates the global graph nor implements robust optimization.
class LoopConsensus {
public:
  explicit LoopConsensus(LoopConsensusConfig config = {});

  [[nodiscard]] core::Result<LoopConsensusReport, LoopConsensusError> resolve(
      std::span<const LoopMeasurement> proposals,
      std::span<const PairwiseConsistencyInput> pairwise_inputs, core::FusionTime now,
      const LoopAncestryApi& ancestry_api) const;

  [[nodiscard]] const LoopConsensusConfig& config() const noexcept { return config_; }

private:
  LoopConsensusConfig config_;
};

}  // namespace meridian::global
