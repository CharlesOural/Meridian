#include "meridian/global/loop_consensus.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

namespace meridian::global {
namespace {

using Vector6d = Eigen::Matrix<double, 6, 1>;

[[nodiscard]] bool sameSubmap(const core::SubmapRef& lhs,
                              const core::SubmapRef& rhs) noexcept {
  return lhs == rhs;
}

[[nodiscard]] bool sameSubmapObject(const core::SubmapRef& lhs,
                                    const core::SubmapRef& rhs) noexcept {
  return lhs.session == rhs.session && lhs.odom_epoch == rhs.odom_epoch && lhs.id == rhs.id;
}

[[nodiscard]] bool sameEndpointPair(const LoopMeasurement& lhs,
                                    const LoopMeasurement& rhs) noexcept {
  return (sameSubmap(lhs.from, rhs.from) && sameSubmap(lhs.to, rhs.to)) ||
         (sameSubmap(lhs.from, rhs.to) && sameSubmap(lhs.to, rhs.from));
}

[[nodiscard]] bool validSubmap(const core::SubmapRef& submap) noexcept {
  return core::validateSubmapRef(submap) == core::SubmapRefValidationError::None;
}

[[nodiscard]] bool finitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool finiteNonnegative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] bool validRankAwareValues(const Eigen::Matrix<double, 6, 6>& basis,
                                        const Vector6d& values, std::size_t rank,
                                        const LoopConsensusConfig& config,
                                        bool require_nonzero_rank) noexcept {
  if (!basis.allFinite() || !values.allFinite() || rank > 6U ||
      (require_nonzero_rank && rank == 0U)) {
    return false;
  }

  if (rank > 0U) {
    const auto supported = basis.leftCols(static_cast<Eigen::Index>(rank));
    const Eigen::MatrixXd gram = supported.transpose() * supported;
    if (!gram.isApprox(Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(rank),
                                                 static_cast<Eigen::Index>(rank)),
                       config.information_basis_orthonormal_tolerance)) {
      return false;
    }
  }

  for (std::size_t index = 0U; index < rank; ++index) {
    if (values(static_cast<Eigen::Index>(index)) < config.minimum_supported_value) {
      return false;
    }
  }
  for (std::size_t index = rank; index < 6U; ++index) {
    if (std::abs(values(static_cast<Eigen::Index>(index))) > config.information_zero_tolerance) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool validHeader(const core::RecordHeader& header) noexcept {
  return header.schema_version > 0U && header.trace.valid() && header.producer.valid() &&
         header.session.valid() && header.config.valid() &&
         (!header.direct_calibration || header.direct_calibration->valid());
}

[[nodiscard]] bool containsCalibration(std::span<const core::CalibrationEpoch> epochs,
                                       core::CalibrationEpoch calibration) noexcept {
  return std::binary_search(epochs.begin(), epochs.end(), calibration);
}

[[nodiscard]] bool validProposal(const LoopMeasurement& proposal, const LoopConsensusConfig& config,
                                 core::FusionTime now, std::string* detail) {
  if (!validHeader(proposal.header) || !proposal.proposal.valid()) {
    *detail = "invalid record header or proposal ID";
    return false;
  }
  switch (proposal.modality) {
    case LoopModality::Visual:
    case LoopModality::Lidar:
      break;
    default:
      *detail = "unknown loop modality";
      return false;
  }
  if (proposal.header.created_at > now) {
    *detail = "proposal creation time is in the future";
    return false;
  }
  if (!validSubmap(proposal.from) || !validSubmap(proposal.to) ||
      sameSubmapObject(proposal.from, proposal.to) ||
      proposal.header.session != proposal.from.session ||
      proposal.header.session != proposal.to.session) {
    *detail = "loop endpoints must be valid and distinct";
    return false;
  }
  if (!proposal.T_from_to.matrix().allFinite()) {
    *detail = "relative pose is non-finite";
    return false;
  }
  if (proposal.information.tangent != core::PoseTangentConvention::RightTranslationFirst ||
      !validRankAwareValues(proposal.information.basis, proposal.information.eigenvalues,
                            proposal.information.rank, config, true)) {
    *detail = "invalid rank-aware loop information";
    return false;
  }

  if (proposal.calibration_epochs.empty() ||
      proposal.calibration_epochs.size() > config.maximum_calibration_epochs_per_proposal ||
      !std::is_sorted(proposal.calibration_epochs.begin(), proposal.calibration_epochs.end()) ||
      std::adjacent_find(proposal.calibration_epochs.begin(), proposal.calibration_epochs.end()) !=
          proposal.calibration_epochs.end() ||
      std::any_of(proposal.calibration_epochs.begin(), proposal.calibration_epochs.end(),
                  [](core::CalibrationEpoch epoch) { return !epoch.valid(); })) {
    *detail = "calibration epochs are empty, invalid, non-canonical, or over capacity";
    return false;
  }
  if (proposal.header.direct_calibration &&
      !containsCalibration(proposal.calibration_epochs, *proposal.header.direct_calibration)) {
    *detail = "direct calibration is absent from the proposal epoch set";
    return false;
  }
  if (!containsCalibration(proposal.calibration_epochs, proposal.from.calibration) ||
      !containsCalibration(proposal.calibration_epochs, proposal.to.calibration)) {
    *detail = "endpoint calibration is absent from the proposal epoch set";
    return false;
  }

  if (!proposal.lineage.id.valid() || proposal.lineage.usage.empty() ||
      proposal.lineage.usage.size() > config.maximum_lineage_usages_per_proposal ||
      proposal.lineage.correlations.size() > config.maximum_lineage_correlations_per_proposal) {
    *detail = "lineage identity, roots, or capacity is invalid";
    return false;
  }
  bool has_primary_root = false;
  for (const core::ObservationUsage& usage : proposal.lineage.usage) {
    if (!containsCalibration(proposal.calibration_epochs, usage.slice.calibration)) {
      *detail = "lineage calibration is absent from the proposal epoch set";
      return false;
    }
    has_primary_root = has_primary_root || usage.role == core::ObservationRole::PrimaryResidual;
  }
  if (!has_primary_root) {
    *detail = "loop lineage has no primary residual root";
    return false;
  }
  if (core::validateLineage(proposal.lineage) != core::LineageValidationError::None) {
    *detail = "loop lineage failed canonical validation";
    return false;
  }
  return true;
}

[[nodiscard]] bool validConfig(const LoopConsensusConfig& config) noexcept {
  return config.maximum_proposals > 0U && config.maximum_pairwise_inputs > 0U &&
         config.maximum_component_vertices > 0U &&
         config.maximum_component_vertices <= config.maximum_proposals &&
         config.maximum_component_edges > 0U && config.maximum_exact_clique_expansions > 0U &&
         config.proposal_ttl.nanoseconds > 0 && config.minimum_common_rank > 0U &&
         config.minimum_common_rank <= 6U && config.maximum_calibration_epochs_per_proposal > 0U &&
         config.maximum_lineage_usages_per_proposal > 0U &&
         config.maximum_lineage_correlations_per_proposal > 0U &&
         finitePositive(config.information_basis_orthonormal_tolerance) &&
         finitePositive(config.information_zero_tolerance) &&
         finitePositive(config.minimum_supported_value) &&
         finiteNonnegative(config.maximum_psd_roundoff_clamp);
}

[[nodiscard]] bool validPairwiseInput(const PairwiseConsistencyInput& input,
                                      const LoopConsensusConfig& config, std::string* detail) {
  if (!input.first.valid() || !input.second.valid() || input.first == input.second) {
    *detail = "pairwise input IDs must be valid and distinct";
    return false;
  }
  if (input.tangent != core::PoseTangentConvention::RightTranslationFirst ||
      !input.cycle_residual.allFinite() ||
      !validRankAwareValues(input.common_basis, input.common_covariance_eigenvalues,
                            input.common_rank, config, false)) {
    *detail = "invalid common supported cycle subspace";
    return false;
  }
  if (!finiteNonnegative(input.maximum_psd_clamp) ||
      input.maximum_psd_clamp > config.maximum_psd_roundoff_clamp ||
      (!input.psd_roundoff_repaired && input.maximum_psd_clamp != 0.0)) {
    *detail = "PSD repair exceeds the numerical-roundoff allowance";
    return false;
  }
  if (input.common_rank >= config.minimum_common_rank) {
    if (!finitePositive(input.chi_squared_threshold)) {
      *detail = "defined pairwise input requires a positive chi-squared threshold";
      return false;
    }
  } else if (!finiteNonnegative(input.chi_squared_threshold)) {
    *detail = "undefined pairwise threshold must remain finite and nonnegative";
    return false;
  }
  return true;
}

[[nodiscard]] std::pair<ProposalId, ProposalId> canonicalPair(ProposalId first,
                                                              ProposalId second) noexcept {
  return second < first ? std::pair{second, first} : std::pair{first, second};
}

[[nodiscard]] bool isExpired(core::FusionTime now, core::FusionTime created_at,
                             core::Duration ttl) noexcept {
  return now - created_at >= ttl;
}

[[nodiscard]] double squaredMahalanobis(const PairwiseConsistencyInput& input) noexcept {
  const Eigen::Index rank = static_cast<Eigen::Index>(input.common_rank);
  const Eigen::VectorXd projected =
      input.common_basis.leftCols(rank).transpose() * input.cycle_residual;
  double squared_distance = 0.0;
  for (Eigen::Index index = 0; index < rank; ++index) {
    squared_distance +=
        projected(index) * projected(index) / input.common_covariance_eigenvalues(index);
  }
  return squared_distance;
}

[[nodiscard]] bool crossModalSameEndpointPair(const LoopMeasurement& first,
                                              const LoopMeasurement& second) noexcept {
  return first.modality != second.modality && sameEndpointPair(first, second);
}

[[nodiscard]] bool hasRequiredPairTreatment(PairCorrelationTreatment treatment) noexcept {
  return treatment == PairCorrelationTreatment::CovarianceInflationAndInformationCap ||
         treatment == PairCorrelationTreatment::JointCompositeWhitening;
}

[[nodiscard]] bool validAssessment(const LoopAncestryAssessment& assessment) noexcept {
  switch (assessment.independence) {
    case AncestryIndependence::Independent:
    case AncestryIndependence::NotIndependent:
    case AncestryIndependence::Unknown:
      break;
    default:
      return false;
  }
  switch (assessment.pair_treatment) {
    case PairCorrelationTreatment::NotRequired:
    case PairCorrelationTreatment::CovarianceInflationAndInformationCap:
    case PairCorrelationTreatment::JointCompositeWhitening:
    case PairCorrelationTreatment::Missing:
      return true;
    default:
      return false;
  }
}

struct CliqueSearchResult {
  bool complete{true};
  std::vector<std::size_t> clique;
  std::uint64_t expansions{};
};

// Meridian-owned bounded exact search, informed by Kimera-RPGO's PCM/max-clique
// separation and the legacy Meridian deterministic tie-break. Unlike both
// legacy heuristic fallbacks, budget exhaustion discards the partial result so
// the complete component can be explicitly deferred.

[[nodiscard]] bool betterClique(const std::vector<std::size_t>& candidate,
                                const std::vector<std::size_t>& incumbent) {
  if (candidate.size() != incumbent.size()) {
    return candidate.size() > incumbent.size();
  }
  return std::lexicographical_compare(candidate.begin(), candidate.end(), incumbent.begin(),
                                      incumbent.end());
}

bool searchCliques(const std::vector<std::vector<bool>>& adjacency,
                   const std::uint64_t maximum_expansions, std::vector<std::size_t>* current,
                   std::vector<std::size_t> candidates, CliqueSearchResult* result) {
  if (result->expansions >= maximum_expansions) {
    result->complete = false;
    return false;
  }
  ++result->expansions;

  if (betterClique(*current, result->clique)) {
    result->clique = *current;
  }
  while (!candidates.empty()) {
    if (current->size() + candidates.size() < result->clique.size()) {
      break;
    }
    const std::size_t vertex = candidates.front();
    std::vector<std::size_t> next;
    next.reserve(candidates.size() - 1U);
    for (std::size_t index = 1U; index < candidates.size(); ++index) {
      const std::size_t candidate = candidates[index];
      if (adjacency[vertex][candidate]) {
        next.push_back(candidate);
      }
    }

    current->push_back(vertex);
    if (!searchCliques(adjacency, maximum_expansions, current, std::move(next), result)) {
      current->pop_back();
      return false;
    }
    current->pop_back();
    candidates.erase(candidates.begin());
  }
  return true;
}

[[nodiscard]] CliqueSearchResult exactMaximumClique(const std::vector<std::vector<bool>>& adjacency,
                                                    std::uint64_t maximum_expansions) {
  CliqueSearchResult result;
  std::vector<std::size_t> current;
  std::vector<std::size_t> candidates(adjacency.size());
  std::iota(candidates.begin(), candidates.end(), std::size_t{0U});
  (void)searchCliques(adjacency, maximum_expansions, &current, std::move(candidates), &result);
  if (!result.complete) {
    result.clique.clear();
  }
  return result;
}

struct WorkingComponent {
  std::vector<std::size_t> members;
  core::FusionTime oldest_time;
  ProposalId smallest_id;
  std::size_t edge_count{};
};

[[nodiscard]] LoopConsensusError error(LoopConsensusErrorCode code, std::string detail,
                                       std::optional<ProposalId> first = std::nullopt,
                                       std::optional<ProposalId> second = std::nullopt,
                                       std::optional<std::size_t> input_index = std::nullopt) {
  return LoopConsensusError{code, first, second, input_index, std::move(detail)};
}

}  // namespace

LoopAncestryAssessment CoreLoopAncestryApi::assess(const LoopMeasurement& first,
                                                   const LoopMeasurement& second) const noexcept {
  const bool independent = core::lineagesAreIndependent(first.lineage, second.lineage);
  if (!independent) {
    return LoopAncestryAssessment{AncestryIndependence::NotIndependent,
                                  PairCorrelationTreatment::Missing};
  }
  if (!crossModalSameEndpointPair(first, second)) {
    return LoopAncestryAssessment{AncestryIndependence::Independent,
                                  PairCorrelationTreatment::NotRequired};
  }

  for (const core::CorrelationDeclaration& left : first.lineage.correlations) {
    for (const core::CorrelationDeclaration& right : second.lineage.correlations) {
      if (left.group != right.group || left.treatment != right.treatment) {
        continue;
      }
      if (left.treatment == core::CorrelationTreatment::JointCompositeWhitening) {
        return LoopAncestryAssessment{AncestryIndependence::Independent,
                                      PairCorrelationTreatment::JointCompositeWhitening};
      }
      if (left.treatment == core::CorrelationTreatment::CovarianceInflationAndInformationCap &&
          left.total_information_cap && right.total_information_cap &&
          finitePositive(*left.total_information_cap) &&
          finitePositive(*right.total_information_cap)) {
        return LoopAncestryAssessment{
            AncestryIndependence::Independent,
            PairCorrelationTreatment::CovarianceInflationAndInformationCap};
      }
    }
  }
  return LoopAncestryAssessment{AncestryIndependence::Independent,
                                PairCorrelationTreatment::Missing};
}

LoopConsensus::LoopConsensus(LoopConsensusConfig config) : config_(std::move(config)) {}

core::Result<LoopConsensusReport, LoopConsensusError> LoopConsensus::resolve(
    std::span<const LoopMeasurement> proposals,
    std::span<const PairwiseConsistencyInput> pairwise_inputs, core::FusionTime now,
    const LoopAncestryApi& ancestry_api) const {
  using Result = core::Result<LoopConsensusReport, LoopConsensusError>;
  if (!validConfig(config_)) {
    return Result::failure(
        error(LoopConsensusErrorCode::InvalidConfig, "loop-consensus configuration is invalid"));
  }
  if (proposals.size() > config_.maximum_proposals) {
    return Result::failure(
        error(LoopConsensusErrorCode::ProposalCapacity, "proposal capacity exceeded"));
  }
  if (pairwise_inputs.size() > config_.maximum_pairwise_inputs) {
    return Result::failure(
        error(LoopConsensusErrorCode::PairwiseInputCapacity, "pairwise-input capacity exceeded"));
  }

  std::vector<const LoopMeasurement*> ordered_proposals;
  std::vector<std::size_t> original_proposal_indices;
  ordered_proposals.reserve(proposals.size());
  original_proposal_indices.reserve(proposals.size());
  for (std::size_t index = 0U; index < proposals.size(); ++index) {
    std::string detail;
    if (!validProposal(proposals[index], config_, now, &detail)) {
      const auto code =
          core::validateLineage(proposals[index].lineage) == core::LineageValidationError::None
              ? LoopConsensusErrorCode::InvalidProposal
              : LoopConsensusErrorCode::InvalidLineage;
      return Result::failure(
          error(code, std::move(detail), proposals[index].proposal, std::nullopt, index));
    }
    ordered_proposals.push_back(&proposals[index]);
    original_proposal_indices.push_back(index);
  }

  std::vector<std::size_t> proposal_order(proposals.size());
  std::iota(proposal_order.begin(), proposal_order.end(), std::size_t{0U});
  std::sort(proposal_order.begin(), proposal_order.end(),
            [&ordered_proposals](std::size_t lhs, std::size_t rhs) {
              return ordered_proposals[lhs]->proposal < ordered_proposals[rhs]->proposal;
            });
  {
    std::vector<const LoopMeasurement*> sorted;
    std::vector<std::size_t> sorted_original_indices;
    sorted.reserve(proposals.size());
    sorted_original_indices.reserve(proposals.size());
    for (const std::size_t index : proposal_order) {
      sorted.push_back(ordered_proposals[index]);
      sorted_original_indices.push_back(original_proposal_indices[index]);
    }
    ordered_proposals = std::move(sorted);
    original_proposal_indices = std::move(sorted_original_indices);
  }

  for (std::size_t index = 1U; index < ordered_proposals.size(); ++index) {
    if (ordered_proposals[index - 1U]->proposal == ordered_proposals[index]->proposal) {
      return Result::failure(error(
          LoopConsensusErrorCode::DuplicateProposalId, "proposal ID appears more than once",
          ordered_proposals[index]->proposal, std::nullopt, original_proposal_indices[index]));
    }
  }
  if (!ordered_proposals.empty()) {
    const core::SessionId session = ordered_proposals.front()->header.session;
    for (std::size_t index = 1U; index < ordered_proposals.size(); ++index) {
      if (ordered_proposals[index]->header.session != session) {
        return Result::failure(error(LoopConsensusErrorCode::MixedSessionBatch,
                                     "one PCM batch may contain only one mission session",
                                     ordered_proposals[index]->proposal, std::nullopt,
                                     original_proposal_indices[index]));
      }
    }
  }

  const auto find_proposal = [&ordered_proposals](ProposalId id) -> std::optional<std::size_t> {
    const auto found = std::lower_bound(ordered_proposals.begin(), ordered_proposals.end(), id,
                                        [](const LoopMeasurement* proposal, ProposalId sought) {
                                          return proposal->proposal < sought;
                                        });
    if (found == ordered_proposals.end() || (*found)->proposal != id) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(std::distance(ordered_proposals.begin(), found));
  };

  std::vector<bool> expired(ordered_proposals.size(), false);
  for (std::size_t index = 0U; index < ordered_proposals.size(); ++index) {
    expired[index] =
        isExpired(now, ordered_proposals[index]->header.created_at, config_.proposal_ttl);
  }

  std::vector<std::size_t> pair_order(pairwise_inputs.size());
  std::iota(pair_order.begin(), pair_order.end(), std::size_t{0U});
  std::vector<std::pair<ProposalId, ProposalId>> pair_keys;
  pair_keys.reserve(pairwise_inputs.size());
  for (std::size_t index = 0U; index < pairwise_inputs.size(); ++index) {
    std::string detail;
    if (!validPairwiseInput(pairwise_inputs[index], config_, &detail)) {
      return Result::failure(error(LoopConsensusErrorCode::InvalidPairwiseInput, std::move(detail),
                                   pairwise_inputs[index].first, pairwise_inputs[index].second,
                                   index));
    }
    pair_keys.push_back(canonicalPair(pairwise_inputs[index].first, pairwise_inputs[index].second));
  }
  std::sort(pair_order.begin(), pair_order.end(), [&pair_keys](std::size_t lhs, std::size_t rhs) {
    return pair_keys[lhs] < pair_keys[rhs];
  });
  for (std::size_t sorted_index = 1U; sorted_index < pair_order.size(); ++sorted_index) {
    if (pair_keys[pair_order[sorted_index - 1U]] == pair_keys[pair_order[sorted_index]]) {
      const std::size_t input_index = pair_order[sorted_index];
      return Result::failure(
          error(LoopConsensusErrorCode::DuplicatePairwiseInput,
                "pairwise input appears more than once, including reversed order",
                pair_keys[input_index].first, pair_keys[input_index].second, input_index));
    }
  }

  LoopConsensusReport report;
  report.evaluated_at = now;
  report.pairs.reserve(pairwise_inputs.size());
  report.decisions.resize(ordered_proposals.size());
  std::vector<std::vector<std::size_t>> adjacency(ordered_proposals.size());

  for (const std::size_t input_index : pair_order) {
    const PairwiseConsistencyInput& input = pairwise_inputs[input_index];
    const auto [first_id, second_id] = pair_keys[input_index];
    const std::optional<std::size_t> first_index = find_proposal(first_id);
    const std::optional<std::size_t> second_index = find_proposal(second_id);
    if (!first_index || !second_index) {
      return Result::failure(error(LoopConsensusErrorCode::UnknownPairwiseProposal,
                                   "pairwise input references a proposal absent from this batch",
                                   first_id, second_id, input_index));
    }

    PairwiseConsensusReport pair_report;
    pair_report.first = first_id;
    pair_report.second = second_id;
    pair_report.input = input;
    if (input.common_rank < config_.minimum_common_rank) {
      pair_report.relation = PairwiseRelation::UndefinedInsufficientCommonRank;
      pair_report.disposition = PairwiseEdgeDisposition::NoEdgeInsufficientCommonRank;
    } else {
      const double squared_distance = squaredMahalanobis(input);
      if (!finiteNonnegative(squared_distance)) {
        return Result::failure(
            error(LoopConsensusErrorCode::InvalidPairwiseInput,
                  "pairwise Mahalanobis distance overflowed or became non-finite", first_id,
                  second_id, input_index));
      }
      pair_report.squared_mahalanobis = squared_distance;
      pair_report.relation = squared_distance <= input.chi_squared_threshold
                                 ? PairwiseRelation::Consistent
                                 : PairwiseRelation::Inconsistent;
      pair_report.disposition = pair_report.relation == PairwiseRelation::Consistent
                                    ? PairwiseEdgeDisposition::ConsistencyEdge
                                    : PairwiseEdgeDisposition::NoEdgeInconsistent;
    }

    if (expired[*first_index] || expired[*second_index]) {
      pair_report.disposition = PairwiseEdgeDisposition::IgnoredExpiredEndpoint;
      report.pairs.push_back(std::move(pair_report));
      continue;
    }
    if (pair_report.relation != PairwiseRelation::Consistent) {
      report.pairs.push_back(std::move(pair_report));
      continue;
    }

    const LoopMeasurement& first = *ordered_proposals[*first_index];
    const LoopMeasurement& second = *ordered_proposals[*second_index];
    pair_report.ancestry = ancestry_api.assess(first, second);
    if (!validAssessment(pair_report.ancestry) ||
        (pair_report.ancestry.independence == AncestryIndependence::Independent &&
         !core::lineagesAreIndependent(first.lineage, second.lineage))) {
      return Result::failure(error(LoopConsensusErrorCode::InconsistentAncestryAssessment,
                                   "ancestry API claimed an invalid or contradicted assessment",
                                   first_id, second_id, input_index));
    }
    if (pair_report.ancestry.independence == AncestryIndependence::NotIndependent) {
      pair_report.disposition = PairwiseEdgeDisposition::NoEdgeNotIndependent;
    } else if (pair_report.ancestry.independence == AncestryIndependence::Unknown) {
      pair_report.disposition = PairwiseEdgeDisposition::NoEdgeUnknownAncestry;
    } else if (crossModalSameEndpointPair(first, second) &&
               !hasRequiredPairTreatment(pair_report.ancestry.pair_treatment)) {
      pair_report.disposition = PairwiseEdgeDisposition::NoEdgeMissingPairTreatment;
    } else {
      adjacency[*first_index].push_back(*second_index);
      adjacency[*second_index].push_back(*first_index);
    }
    report.pairs.push_back(std::move(pair_report));
  }

  for (std::vector<std::size_t>& neighbours : adjacency) {
    std::sort(neighbours.begin(), neighbours.end());
  }

  std::vector<bool> visited(ordered_proposals.size(), false);
  std::vector<WorkingComponent> components;
  components.reserve(ordered_proposals.size());
  for (std::size_t start = 0U; start < ordered_proposals.size(); ++start) {
    if (expired[start] || visited[start]) {
      continue;
    }
    WorkingComponent component;
    component.oldest_time = ordered_proposals[start]->header.created_at;
    component.smallest_id = ordered_proposals[start]->proposal;
    std::vector<std::size_t> stack{start};
    visited[start] = true;
    while (!stack.empty()) {
      const std::size_t current = stack.back();
      stack.pop_back();
      component.members.push_back(current);
      component.oldest_time =
          std::min(component.oldest_time, ordered_proposals[current]->header.created_at);
      component.smallest_id = std::min(component.smallest_id, ordered_proposals[current]->proposal);
      component.edge_count += adjacency[current].size();
      for (const std::size_t neighbour : adjacency[current]) {
        if (!visited[neighbour]) {
          visited[neighbour] = true;
          stack.push_back(neighbour);
        }
      }
    }
    component.edge_count /= 2U;
    std::sort(component.members.begin(), component.members.end());
    components.push_back(std::move(component));
  }
  std::sort(components.begin(), components.end(),
            [](const WorkingComponent& lhs, const WorkingComponent& rhs) {
              return std::tie(lhs.oldest_time, lhs.smallest_id) <
                     std::tie(rhs.oldest_time, rhs.smallest_id);
            });

  for (std::size_t index = 0U; index < ordered_proposals.size(); ++index) {
    report.decisions[index].proposal = ordered_proposals[index]->proposal;
    if (expired[index]) {
      report.decisions[index].disposition = LoopProposalDisposition::RejectedExpiredTtl;
    }
  }

  report.components.reserve(components.size());
  for (std::size_t component_index = 0U; component_index < components.size(); ++component_index) {
    const WorkingComponent& working = components[component_index];
    PcmComponentReport component_report;
    component_report.oldest_proposal_time = working.oldest_time;
    component_report.smallest_proposal_id = working.smallest_id;
    component_report.consistency_edges = working.edge_count;
    component_report.proposals.reserve(working.members.size());
    for (const std::size_t member : working.members) {
      component_report.proposals.push_back(ordered_proposals[member]->proposal);
      report.decisions[member].component_index = component_index;
    }

    if (working.members.size() == 1U) {
      const std::size_t member = working.members.front();
      component_report.maximum_clique.push_back(ordered_proposals[member]->proposal);
      component_report.disposition = PcmComponentDisposition::RejectedSingleton;
      report.decisions[member].disposition = LoopProposalDisposition::RejectedSingleton;
      report.components.push_back(std::move(component_report));
      continue;
    }

    std::optional<PcmResourceLimit> resource_limit;
    if (working.members.size() > config_.maximum_component_vertices) {
      resource_limit = PcmResourceLimit::ComponentVertices;
    } else if (working.edge_count > config_.maximum_component_edges) {
      resource_limit = PcmResourceLimit::ComponentEdges;
    }
    if (resource_limit) {
      component_report.disposition = PcmComponentDisposition::DeferredResourceLimit;
      component_report.resource_limit = resource_limit;
      for (const std::size_t member : working.members) {
        report.decisions[member].disposition = LoopProposalDisposition::DeferredResourceLimit;
        report.decisions[member].resource_limit = resource_limit;
      }
      report.components.push_back(std::move(component_report));
      continue;
    }

    const std::size_t count = working.members.size();
    std::vector<std::vector<bool>> local_adjacency(count, std::vector<bool>(count, false));
    for (std::size_t lhs = 0U; lhs < count; ++lhs) {
      for (std::size_t rhs = lhs + 1U; rhs < count; ++rhs) {
        const bool adjacent =
            std::binary_search(adjacency[working.members[lhs]].begin(),
                               adjacency[working.members[lhs]].end(), working.members[rhs]);
        local_adjacency[lhs][rhs] = adjacent;
        local_adjacency[rhs][lhs] = adjacent;
      }
    }
    CliqueSearchResult clique =
        exactMaximumClique(local_adjacency, config_.maximum_exact_clique_expansions);
    component_report.exact_search_expansions = clique.expansions;
    if (!clique.complete) {
      component_report.disposition = PcmComponentDisposition::DeferredResourceLimit;
      component_report.resource_limit = PcmResourceLimit::ExactCliqueExpansions;
      for (const std::size_t member : working.members) {
        report.decisions[member].disposition = LoopProposalDisposition::DeferredResourceLimit;
        report.decisions[member].resource_limit = PcmResourceLimit::ExactCliqueExpansions;
      }
      report.components.push_back(std::move(component_report));
      continue;
    }

    std::vector<bool> selected(count, false);
    for (const std::size_t local_index : clique.clique) {
      selected[local_index] = true;
      component_report.maximum_clique.push_back(
          ordered_proposals[working.members[local_index]]->proposal);
    }
    if (component_report.maximum_clique.size() < 2U) {
      component_report.disposition = PcmComponentDisposition::RejectedSingleton;
      for (const std::size_t member : working.members) {
        report.decisions[member].disposition = LoopProposalDisposition::RejectedSingleton;
      }
    } else {
      component_report.disposition = PcmComponentDisposition::ResolvedMaximumClique;
      for (std::size_t local_index = 0U; local_index < count; ++local_index) {
        report.decisions[working.members[local_index]].disposition =
            selected[local_index] ? LoopProposalDisposition::AdmittedToGncCandidateBatch
                                  : LoopProposalDisposition::RejectedOutsideMaximumClique;
      }
    }
    report.components.push_back(std::move(component_report));
  }

  return Result::success(std::move(report));
}

}  // namespace meridian::global
