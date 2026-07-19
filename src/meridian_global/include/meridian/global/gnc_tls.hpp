#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "meridian/core/result.hpp"
#include "meridian/global/ids.hpp"

namespace meridian::global {

// The cutoff has the same units as the whitened squared cost: r^T r, without
// the one-half convention used by some nonlinear solvers. It is supplied by
// the factor producer from a calibrated chi-squared gate for this exact factor
// model and number of degrees of freedom.
struct GncTlsFactorScale {
  std::uint32_t degrees_of_freedom{};
  double calibrated_chi_squared_cutoff{};
};

struct KnownInlierFactor {
  GlobalFactorId factor_id;
  // Structural definitions and condensed likelihoods without a calibrated
  // total-cost DoF remain fixed unit-weight known factors. Their raw cost is
  // reported without inventing a TLS scale.
  std::optional<GncTlsFactorScale> monitoring_scale;
};

struct RobustCandidateFactor {
  CandidateId candidate_id;
  GncTlsFactorScale scale;
};

struct ShadowKnownInlierWeight {
  GlobalFactorId factor_id;
  double weight{1.0};
};

struct ShadowCandidateWeight {
  CandidateId candidate_id;
  double weight{1.0};
};

enum class ShadowSolvePhase {
  InitialFullGraph,
  GraduatedFullGraph,
};

// A request always identifies every known inlier and every robust candidate in
// the complete bounded active shadow graph. Known-inlier weights are exactly
// one. A zero candidate weight is still an identified factor; the shadow
// solver must not turn a weighted solve into a factor-removal subproblem.
struct GncTlsShadowSolveRequest {
  std::size_t solver_call{};
  ShadowSolvePhase phase{ShadowSolvePhase::InitialFullGraph};
  std::optional<double> mu;
  std::vector<ShadowKnownInlierWeight> known_inliers;
  std::vector<ShadowCandidateWeight> robust_candidates;
};

struct ShadowKnownInlierCost {
  GlobalFactorId factor_id;
  double whitened_squared_cost{};
};

struct ShadowCandidateCost {
  CandidateId candidate_id;
  double whitened_squared_cost{};
};

struct GncTlsShadowSolveResult {
  // Number of iterations consumed by the bounded inner nonlinear solver.
  std::size_t solver_iterations{};
  bool finite_solution{false};
  std::vector<ShadowKnownInlierCost> known_inliers;
  std::vector<ShadowCandidateCost> robust_candidates;
};

enum class ShadowSolveFailureCode {
  DidNotConverge,
  IterationLimit,
  NumericalFailure,
  ResourceLimit,
  Deadline,
};

struct ShadowSolveFailure {
  ShadowSolveFailureCode code{ShadowSolveFailureCode::NumericalFailure};
  std::string detail;
};

// Adapter implemented by the global graph's disposable shadow transaction.
// Each call optimizes the same complete active graph with the supplied
// information weights and returns post-solve unweighted r^T r values. The
// adapter may update its shadow estimate between calls, but it must not mutate
// graph topology or any committed graph state.
class GncTlsShadowSolverApi {
public:
  virtual ~GncTlsShadowSolverApi() = default;

  [[nodiscard]] virtual core::Result<GncTlsShadowSolveResult, ShadowSolveFailure> solve(
      const GncTlsShadowSolveRequest& request) = 0;
};

struct GncTlsConfig {
  std::size_t maximum_known_inliers{16384};
  std::size_t maximum_robust_candidates{512};
  std::size_t maximum_iterations{100};
  std::size_t maximum_inner_solver_iterations{100};
  std::uint32_t maximum_degrees_of_freedom{64};

  double mu_step{1.4};
  double minimum_mu{1.0e-6};
  double maximum_mu{1.0e12};
  double binary_weight_tolerance{1.0e-4};
  double stable_weight_tolerance{1.0e-6};

  double minimum_chi_squared_cutoff{1.0e-12};
  double maximum_chi_squared_cutoff{1.0e12};
  double maximum_whitened_squared_cost{1.0e18};
  double maximum_normalized_squared_cost{1.0e12};
};

enum class GncTlsCandidateDisposition {
  TlsInlier,
  TlsOutlier,
};

enum class GncTlsConvergenceReason {
  InitialResidualsInsideConvexRegion,
  StableBinaryWeights,
};

struct GncTlsFactorCost {
  GncTlsFactorScale scale;
  double whitened_squared_cost{};
  double normalized_squared_cost{};
};

struct GncTlsKnownInlierReport {
  GlobalFactorId factor_id;
  std::optional<GncTlsFactorScale> monitoring_scale;
  double whitened_squared_cost{};
  std::optional<double> normalized_squared_cost;
  double weight{1.0};
};

struct GncTlsCandidateReport {
  CandidateId candidate_id;
  GncTlsFactorCost cost;
  double final_weight{};
  GncTlsCandidateDisposition disposition{GncTlsCandidateDisposition::TlsOutlier};
};

struct GncTlsIterationReport {
  std::size_t solver_call{};
  ShadowSolvePhase phase{ShadowSolvePhase::InitialFullGraph};
  std::optional<double> mu;
  std::size_t inner_solver_iterations{};
  double normalized_weighted_cost{};
  double relative_normalized_weighted_cost_change{};
  double minimum_candidate_weight{1.0};
  double maximum_candidate_weight{1.0};
  double maximum_weight_change_after_solve{};
  std::size_t binary_inlier_weights{};
  std::size_t binary_outlier_weights{};
  std::size_t nonbinary_weights{};
};

struct GncTlsReport {
  GncTlsConvergenceReason convergence{GncTlsConvergenceReason::InitialResidualsInsideConvexRegion};
  std::size_t solver_calls{};
  std::size_t gnc_iterations{};
  std::vector<GncTlsIterationReport> iterations;
  std::vector<GncTlsKnownInlierReport> known_inliers;
  std::vector<GncTlsCandidateReport> robust_candidates;
};

enum class GncTlsErrorCode {
  InvalidConfig,
  EmptyRobustCandidateSet,
  KnownInlierCapacity,
  RobustCandidateCapacity,
  InvalidKnownInlier,
  InvalidRobustCandidate,
  DuplicateKnownInlier,
  DuplicateRobustCandidate,
  ShadowSolverFailure,
  InnerSolverIterationBound,
  NonFiniteShadowSolution,
  IncompleteShadowCosts,
  InvalidWhitenedSquaredCost,
  NumericalBound,
  IterationLimit,
};

struct GncTlsError {
  GncTlsErrorCode code{GncTlsErrorCode::InvalidConfig};
  std::optional<GlobalFactorId> factor_id;
  std::optional<CandidateId> candidate_id;
  std::optional<ShadowSolveFailureCode> shadow_failure;
  std::string detail;
};

// Deterministic GNC-TLS outer controller following Yang et al.'s TLS surrogate
// schedule. It owns no poses, factors, or graph; all nonlinear optimization is
// delegated through GncTlsShadowSolverApi.
class GncTlsController {
public:
  static constexpr std::size_t kHardMaximumKnownInliers = 65536;
  static constexpr std::size_t kHardMaximumRobustCandidates = 4096;
  static constexpr std::size_t kHardMaximumIterations = 1024;
  static constexpr std::size_t kHardMaximumInnerSolverIterations = 100000;
  static constexpr std::uint32_t kHardMaximumDegreesOfFreedom = 256;
  static constexpr double kHardMaximumMuStep = 1.0e6;
  static constexpr double kHardMaximumMu = 1.0e15;
  static constexpr double kHardMaximumChiSquaredCutoff = 1.0e18;
  static constexpr double kHardMaximumWhitenedSquaredCost = 1.0e30;
  static constexpr double kHardMaximumNormalizedSquaredCost = 1.0e18;

  explicit GncTlsController(GncTlsConfig config = {}) : config_(config) {}

  [[nodiscard]] core::Result<GncTlsReport, GncTlsError> run(
      std::span<const KnownInlierFactor> known_inliers,
      std::span<const RobustCandidateFactor> robust_candidates,
      GncTlsShadowSolverApi& shadow_solver) const;

  [[nodiscard]] const GncTlsConfig& config() const noexcept { return config_; }

private:
  GncTlsConfig config_;
};

}  // namespace meridian::global
