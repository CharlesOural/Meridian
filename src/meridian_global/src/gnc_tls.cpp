#include "meridian/global/gnc_tls.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace meridian::global {
namespace {

struct Evaluation {
  std::size_t inner_solver_iterations{};
  std::vector<GncTlsKnownInlierReport> known_inliers;
  std::vector<GncTlsCandidateReport> robust_candidates;
};

[[nodiscard]] GncTlsError error(GncTlsErrorCode code, std::string detail) {
  GncTlsError result;
  result.code = code;
  result.detail = std::move(detail);
  return result;
}

[[nodiscard]] bool finitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] std::optional<GncTlsError> validateConfig(const GncTlsConfig& config) {
  const bool valid_counts =
      config.maximum_known_inliers <= GncTlsController::kHardMaximumKnownInliers &&
      config.maximum_robust_candidates > 0U &&
      config.maximum_robust_candidates <= GncTlsController::kHardMaximumRobustCandidates &&
      config.maximum_iterations > 0U &&
      config.maximum_iterations <= GncTlsController::kHardMaximumIterations &&
      config.maximum_inner_solver_iterations > 0U &&
      config.maximum_inner_solver_iterations <=
          GncTlsController::kHardMaximumInnerSolverIterations &&
      config.maximum_degrees_of_freedom > 0U &&
      config.maximum_degrees_of_freedom <= GncTlsController::kHardMaximumDegreesOfFreedom;
  const bool valid_schedule =
      std::isfinite(config.mu_step) && config.mu_step > 1.0 &&
      config.mu_step <= GncTlsController::kHardMaximumMuStep && finitePositive(config.minimum_mu) &&
      finitePositive(config.maximum_mu) && config.minimum_mu <= config.maximum_mu &&
      config.maximum_mu <= GncTlsController::kHardMaximumMu &&
      finitePositive(config.binary_weight_tolerance) && config.binary_weight_tolerance < 0.5 &&
      finitePositive(config.stable_weight_tolerance) &&
      config.stable_weight_tolerance <= config.binary_weight_tolerance;
  const bool valid_numerics =
      finitePositive(config.minimum_chi_squared_cutoff) &&
      finitePositive(config.maximum_chi_squared_cutoff) &&
      config.minimum_chi_squared_cutoff <= config.maximum_chi_squared_cutoff &&
      config.maximum_chi_squared_cutoff <= GncTlsController::kHardMaximumChiSquaredCutoff &&
      finitePositive(config.maximum_whitened_squared_cost) &&
      config.maximum_whitened_squared_cost <= GncTlsController::kHardMaximumWhitenedSquaredCost &&
      finitePositive(config.maximum_normalized_squared_cost) &&
      config.maximum_normalized_squared_cost <= GncTlsController::kHardMaximumNormalizedSquaredCost;
  if (!valid_counts || !valid_schedule || !valid_numerics) {
    return error(GncTlsErrorCode::InvalidConfig,
                 "GNC-TLS configuration violates a finite positive or hard resource bound");
  }
  return std::nullopt;
}

[[nodiscard]] bool validScale(const GncTlsFactorScale& scale, const GncTlsConfig& config) {
  return scale.degrees_of_freedom > 0U &&
         scale.degrees_of_freedom <= config.maximum_degrees_of_freedom &&
         std::isfinite(scale.calibrated_chi_squared_cutoff) &&
         scale.calibrated_chi_squared_cutoff >= config.minimum_chi_squared_cutoff &&
         scale.calibrated_chi_squared_cutoff <= config.maximum_chi_squared_cutoff;
}

[[nodiscard]] double tlsWeight(double normalized_squared_cost, double mu) noexcept {
  // Yang et al., Eq. (14), after dividing r^T r by this factor's calibrated
  // chi-squared cutoff. Kimera-RPGO uses the same three-region update through
  // GTSAM; this ROS-free controller keeps that solver dependency outside.
  const double upper_bound = (mu + 1.0) / mu;
  const double lower_bound = mu / (mu + 1.0);
  if (normalized_squared_cost >= upper_bound) {
    return 0.0;
  }
  if (normalized_squared_cost <= lower_bound) {
    return 1.0;
  }
  const double weight = std::sqrt((mu * (mu + 1.0)) / normalized_squared_cost) - mu;
  return std::clamp(weight, 0.0, 1.0);
}

[[nodiscard]] bool binaryWeight(double weight, double tolerance) noexcept {
  return std::min(weight, 1.0 - weight) <= tolerance;
}

[[nodiscard]] GncTlsIterationReport makeIterationReport(
    std::size_t solver_call, ShadowSolvePhase phase, std::optional<double> mu,
    const Evaluation& evaluation, std::span<const ShadowCandidateWeight> applied_weights,
    std::span<const double> post_solve_weights, double previous_cost, const GncTlsConfig& config) {
  GncTlsIterationReport report;
  report.solver_call = solver_call;
  report.phase = phase;
  report.mu = mu;
  report.inner_solver_iterations = evaluation.inner_solver_iterations;
  report.minimum_candidate_weight = 1.0;
  report.maximum_candidate_weight = 0.0;

  for (const auto& factor : evaluation.known_inliers) {
    if (factor.normalized_squared_cost) {
      report.normalized_weighted_cost += *factor.normalized_squared_cost;
    }
  }
  for (std::size_t index = 0; index < evaluation.robust_candidates.size(); ++index) {
    const double weight = applied_weights[index].weight;
    report.normalized_weighted_cost +=
        weight * evaluation.robust_candidates[index].cost.normalized_squared_cost;
    report.minimum_candidate_weight = std::min(report.minimum_candidate_weight, weight);
    report.maximum_candidate_weight = std::max(report.maximum_candidate_weight, weight);
    if (weight <= config.binary_weight_tolerance) {
      ++report.binary_outlier_weights;
    } else if (1.0 - weight <= config.binary_weight_tolerance) {
      ++report.binary_inlier_weights;
    } else {
      ++report.nonbinary_weights;
    }
    if (!post_solve_weights.empty()) {
      report.maximum_weight_change_after_solve = std::max(
          report.maximum_weight_change_after_solve, std::abs(post_solve_weights[index] - weight));
    }
  }
  if (std::isfinite(previous_cost)) {
    report.relative_normalized_weighted_cost_change =
        std::abs(report.normalized_weighted_cost - previous_cost) /
        std::max(std::abs(previous_cost), 1.0e-12);
  }
  return report;
}

[[nodiscard]] core::Result<Evaluation, GncTlsError> solveAndEvaluate(
    const GncTlsShadowSolveRequest& request, std::span<const KnownInlierFactor> known_factors,
    std::span<const RobustCandidateFactor> candidate_factors, const GncTlsConfig& config,
    GncTlsShadowSolverApi& shadow_solver) {
  auto solved = shadow_solver.solve(request);
  if (!solved) {
    GncTlsError failure = error(GncTlsErrorCode::ShadowSolverFailure,
                                "bounded full-graph shadow solve failed: " + solved.error().detail);
    failure.shadow_failure = solved.error().code;
    return core::Result<Evaluation, GncTlsError>::failure(std::move(failure));
  }
  GncTlsShadowSolveResult response = std::move(solved).value();
  if (response.solver_iterations > config.maximum_inner_solver_iterations) {
    return core::Result<Evaluation, GncTlsError>::failure(
        error(GncTlsErrorCode::InnerSolverIterationBound,
              "shadow solver reported more iterations than its configured hard bound"));
  }
  if (!response.finite_solution) {
    return core::Result<Evaluation, GncTlsError>::failure(
        error(GncTlsErrorCode::NonFiniteShadowSolution,
              "shadow solver reported a non-finite candidate solution"));
  }

  std::sort(response.known_inliers.begin(), response.known_inliers.end(),
            [](const auto& left, const auto& right) {
              return left.factor_id.value() < right.factor_id.value();
            });
  std::sort(response.robust_candidates.begin(), response.robust_candidates.end(),
            [](const auto& left, const auto& right) {
              return left.candidate_id.value() < right.candidate_id.value();
            });
  if (response.known_inliers.size() != known_factors.size() ||
      response.robust_candidates.size() != candidate_factors.size()) {
    return core::Result<Evaluation, GncTlsError>::failure(
        error(GncTlsErrorCode::IncompleteShadowCosts,
              "shadow result does not contain exactly one cost for every requested factor"));
  }

  Evaluation evaluation;
  evaluation.inner_solver_iterations = response.solver_iterations;
  evaluation.known_inliers.reserve(known_factors.size());
  evaluation.robust_candidates.reserve(candidate_factors.size());

  const auto make_cost =
      [&](GncTlsFactorScale scale,
          double whitened_squared_cost) -> core::Result<GncTlsFactorCost, GncTlsError> {
    if (!std::isfinite(whitened_squared_cost) || whitened_squared_cost < 0.0 ||
        whitened_squared_cost > config.maximum_whitened_squared_cost) {
      return core::Result<GncTlsFactorCost, GncTlsError>::failure(
          error(GncTlsErrorCode::InvalidWhitenedSquaredCost,
                "factor cost is negative, non-finite, or beyond the configured numerical bound"));
    }
    const double normalized = whitened_squared_cost / scale.calibrated_chi_squared_cutoff;
    if (!std::isfinite(normalized) || normalized > config.maximum_normalized_squared_cost) {
      return core::Result<GncTlsFactorCost, GncTlsError>::failure(
          error(GncTlsErrorCode::NumericalBound,
                "factor cost normalized by its calibrated chi-squared cutoff exceeds the bound"));
    }
    return core::Result<GncTlsFactorCost, GncTlsError>::success(
        GncTlsFactorCost{scale, whitened_squared_cost, normalized});
  };

  for (std::size_t index = 0; index < known_factors.size(); ++index) {
    if (response.known_inliers[index].factor_id != known_factors[index].factor_id) {
      return core::Result<Evaluation, GncTlsError>::failure(
          error(GncTlsErrorCode::IncompleteShadowCosts,
                "shadow result contains a missing, duplicate, or unexpected known-inlier ID"));
    }
    const double raw_cost = response.known_inliers[index].whitened_squared_cost;
    if (!std::isfinite(raw_cost) || raw_cost < 0.0 ||
        raw_cost > config.maximum_whitened_squared_cost) {
      GncTlsError failure =
          error(GncTlsErrorCode::InvalidWhitenedSquaredCost,
                "known-inlier cost is negative, non-finite, or beyond the numerical bound");
      failure.factor_id = known_factors[index].factor_id;
      return core::Result<Evaluation, GncTlsError>::failure(std::move(failure));
    }
    std::optional<double> normalized;
    if (known_factors[index].monitoring_scale) {
      auto cost = make_cost(*known_factors[index].monitoring_scale, raw_cost);
      if (!cost) {
        GncTlsError failure = std::move(cost).error();
        failure.factor_id = known_factors[index].factor_id;
        return core::Result<Evaluation, GncTlsError>::failure(std::move(failure));
      }
      normalized = cost.value().normalized_squared_cost;
    }
    evaluation.known_inliers.push_back(
        GncTlsKnownInlierReport{known_factors[index].factor_id,
                                known_factors[index].monitoring_scale, raw_cost, normalized, 1.0});
  }
  for (std::size_t index = 0; index < candidate_factors.size(); ++index) {
    if (response.robust_candidates[index].candidate_id != candidate_factors[index].candidate_id) {
      return core::Result<Evaluation, GncTlsError>::failure(
          error(GncTlsErrorCode::IncompleteShadowCosts,
                "shadow result contains a missing, duplicate, or unexpected candidate ID"));
    }
    auto cost = make_cost(candidate_factors[index].scale,
                          response.robust_candidates[index].whitened_squared_cost);
    if (!cost) {
      GncTlsError failure = std::move(cost).error();
      failure.candidate_id = candidate_factors[index].candidate_id;
      return core::Result<Evaluation, GncTlsError>::failure(std::move(failure));
    }
    evaluation.robust_candidates.push_back(
        GncTlsCandidateReport{candidate_factors[index].candidate_id, std::move(cost).value(), 1.0,
                              GncTlsCandidateDisposition::TlsOutlier});
  }
  return core::Result<Evaluation, GncTlsError>::success(std::move(evaluation));
}

[[nodiscard]] GncTlsReport finalReport(GncTlsConvergenceReason reason, Evaluation evaluation,
                                       std::span<const ShadowCandidateWeight> final_weights,
                                       std::vector<GncTlsIterationReport> iterations,
                                       std::size_t gnc_iterations, double binary_weight_tolerance) {
  for (std::size_t index = 0; index < evaluation.robust_candidates.size(); ++index) {
    const double weight = final_weights[index].weight;
    evaluation.robust_candidates[index].final_weight = weight;
    evaluation.robust_candidates[index].disposition = weight >= 1.0 - binary_weight_tolerance
                                                          ? GncTlsCandidateDisposition::TlsInlier
                                                          : GncTlsCandidateDisposition::TlsOutlier;
  }
  GncTlsReport report;
  report.convergence = reason;
  report.solver_calls = iterations.size();
  report.gnc_iterations = gnc_iterations;
  report.iterations = std::move(iterations);
  report.known_inliers = std::move(evaluation.known_inliers);
  report.robust_candidates = std::move(evaluation.robust_candidates);
  return report;
}

}  // namespace

core::Result<GncTlsReport, GncTlsError> GncTlsController::run(
    std::span<const KnownInlierFactor> known_inliers,
    std::span<const RobustCandidateFactor> robust_candidates,
    GncTlsShadowSolverApi& shadow_solver) const {
  if (const auto config_error = validateConfig(config_); config_error.has_value()) {
    return core::Result<GncTlsReport, GncTlsError>::failure(*config_error);
  }
  if (robust_candidates.empty()) {
    return core::Result<GncTlsReport, GncTlsError>::failure(
        error(GncTlsErrorCode::EmptyRobustCandidateSet,
              "GNC-TLS requires at least one robust candidate"));
  }
  if (known_inliers.size() > config_.maximum_known_inliers) {
    return core::Result<GncTlsReport, GncTlsError>::failure(
        error(GncTlsErrorCode::KnownInlierCapacity,
              "known-inlier count exceeds the configured complete-graph bound"));
  }
  if (robust_candidates.size() > config_.maximum_robust_candidates) {
    return core::Result<GncTlsReport, GncTlsError>::failure(
        error(GncTlsErrorCode::RobustCandidateCapacity,
              "robust-candidate count exceeds the configured batch bound"));
  }

  std::vector<KnownInlierFactor> sorted_known(known_inliers.begin(), known_inliers.end());
  std::vector<RobustCandidateFactor> sorted_candidates(robust_candidates.begin(),
                                                       robust_candidates.end());
  std::sort(sorted_known.begin(), sorted_known.end(), [](const auto& left, const auto& right) {
    return left.factor_id.value() < right.factor_id.value();
  });
  std::sort(sorted_candidates.begin(), sorted_candidates.end(),
            [](const auto& left, const auto& right) {
              return left.candidate_id.value() < right.candidate_id.value();
            });

  for (std::size_t index = 0; index < sorted_known.size(); ++index) {
    if (!sorted_known[index].factor_id.valid() ||
        (sorted_known[index].monitoring_scale &&
         !validScale(*sorted_known[index].monitoring_scale, config_))) {
      GncTlsError failure =
          error(GncTlsErrorCode::InvalidKnownInlier,
                "known inlier has an invalid ID, DoF, or calibrated chi-squared cutoff");
      failure.factor_id = sorted_known[index].factor_id;
      return core::Result<GncTlsReport, GncTlsError>::failure(std::move(failure));
    }
    if (index > 0U && sorted_known[index - 1U].factor_id == sorted_known[index].factor_id) {
      GncTlsError failure =
          error(GncTlsErrorCode::DuplicateKnownInlier, "known-inlier ID appears more than once");
      failure.factor_id = sorted_known[index].factor_id;
      return core::Result<GncTlsReport, GncTlsError>::failure(std::move(failure));
    }
  }
  for (std::size_t index = 0; index < sorted_candidates.size(); ++index) {
    if (!sorted_candidates[index].candidate_id.valid() ||
        !validScale(sorted_candidates[index].scale, config_)) {
      GncTlsError failure =
          error(GncTlsErrorCode::InvalidRobustCandidate,
                "robust candidate has an invalid ID, DoF, or calibrated chi-squared cutoff");
      failure.candidate_id = sorted_candidates[index].candidate_id;
      return core::Result<GncTlsReport, GncTlsError>::failure(std::move(failure));
    }
    if (index > 0U &&
        sorted_candidates[index - 1U].candidate_id == sorted_candidates[index].candidate_id) {
      GncTlsError failure = error(GncTlsErrorCode::DuplicateRobustCandidate,
                                  "robust-candidate ID appears more than once");
      failure.candidate_id = sorted_candidates[index].candidate_id;
      return core::Result<GncTlsReport, GncTlsError>::failure(std::move(failure));
    }
  }

  std::vector<ShadowKnownInlierWeight> known_weights;
  known_weights.reserve(sorted_known.size());
  for (const auto& factor : sorted_known) {
    known_weights.push_back(ShadowKnownInlierWeight{factor.factor_id, 1.0});
  }
  std::vector<ShadowCandidateWeight> candidate_weights;
  candidate_weights.reserve(sorted_candidates.size());
  for (const auto& factor : sorted_candidates) {
    candidate_weights.push_back(ShadowCandidateWeight{factor.candidate_id, 1.0});
  }

  GncTlsShadowSolveRequest initial_request;
  initial_request.solver_call = 0U;
  initial_request.phase = ShadowSolvePhase::InitialFullGraph;
  initial_request.known_inliers = known_weights;
  initial_request.robust_candidates = candidate_weights;
  auto initial =
      solveAndEvaluate(initial_request, sorted_known, sorted_candidates, config_, shadow_solver);
  if (!initial) {
    return core::Result<GncTlsReport, GncTlsError>::failure(std::move(initial).error());
  }
  Evaluation current = std::move(initial).value();
  std::vector<GncTlsIterationReport> iteration_reports;
  iteration_reports.reserve(config_.maximum_iterations + 1U);
  iteration_reports.push_back(makeIterationReport(
      0U, ShadowSolvePhase::InitialFullGraph, std::nullopt, current, candidate_weights, {},
      std::numeric_limits<double>::quiet_NaN(), config_));

  double maximum_normalized_cost = 0.0;
  for (const auto& candidate : current.robust_candidates) {
    maximum_normalized_cost =
        std::max(maximum_normalized_cost, candidate.cost.normalized_squared_cost);
  }

  // Remark 5 of Yang et al.: when every normalized residual is at most 1/2,
  // the TLS initialization is already in the quadratic region.
  if (maximum_normalized_cost <= 0.5) {
    return core::Result<GncTlsReport, GncTlsError>::success(finalReport(
        GncTlsConvergenceReason::InitialResidualsInsideConvexRegion, std::move(current),
        candidate_weights, std::move(iteration_reports), 0U, config_.binary_weight_tolerance));
  }

  const double denominator = 2.0 * maximum_normalized_cost - 1.0;
  double mu = std::max(config_.minimum_mu, 1.0 / denominator);
  if (!std::isfinite(mu) || mu > config_.maximum_mu) {
    return core::Result<GncTlsReport, GncTlsError>::failure(
        error(GncTlsErrorCode::NumericalBound,
              "initial GNC-TLS mu is non-finite or exceeds the configured bound"));
  }

  double previous_cost = iteration_reports.back().normalized_weighted_cost;
  for (std::size_t iteration = 0; iteration < config_.maximum_iterations; ++iteration) {
    for (std::size_t index = 0; index < candidate_weights.size(); ++index) {
      candidate_weights[index].weight =
          tlsWeight(current.robust_candidates[index].cost.normalized_squared_cost, mu);
    }

    GncTlsShadowSolveRequest request;
    request.solver_call = iteration + 1U;
    request.phase = ShadowSolvePhase::GraduatedFullGraph;
    request.mu = mu;
    request.known_inliers = known_weights;
    request.robust_candidates = candidate_weights;
    auto solved =
        solveAndEvaluate(request, sorted_known, sorted_candidates, config_, shadow_solver);
    if (!solved) {
      return core::Result<GncTlsReport, GncTlsError>::failure(std::move(solved).error());
    }
    Evaluation next = std::move(solved).value();

    std::vector<double> post_solve_weights;
    post_solve_weights.reserve(next.robust_candidates.size());
    bool applied_binary = true;
    bool post_solve_binary = true;
    double maximum_weight_change = 0.0;
    for (std::size_t index = 0; index < next.robust_candidates.size(); ++index) {
      const double post_solve =
          tlsWeight(next.robust_candidates[index].cost.normalized_squared_cost, mu);
      post_solve_weights.push_back(post_solve);
      applied_binary = applied_binary && binaryWeight(candidate_weights[index].weight,
                                                      config_.binary_weight_tolerance);
      post_solve_binary =
          post_solve_binary && binaryWeight(post_solve, config_.binary_weight_tolerance);
      maximum_weight_change =
          std::max(maximum_weight_change, std::abs(post_solve - candidate_weights[index].weight));
    }
    iteration_reports.push_back(
        makeIterationReport(iteration + 1U, ShadowSolvePhase::GraduatedFullGraph, mu, next,
                            candidate_weights, post_solve_weights, previous_cost, config_));
    previous_cost = iteration_reports.back().normalized_weighted_cost;

    if (applied_binary && post_solve_binary &&
        maximum_weight_change <= config_.stable_weight_tolerance) {
      return core::Result<GncTlsReport, GncTlsError>::success(finalReport(
          GncTlsConvergenceReason::StableBinaryWeights, std::move(next), candidate_weights,
          std::move(iteration_reports), iteration + 1U, config_.binary_weight_tolerance));
    }
    current = std::move(next);

    if (iteration + 1U < config_.maximum_iterations) {
      if (mu > config_.maximum_mu / config_.mu_step) {
        return core::Result<GncTlsReport, GncTlsError>::failure(
            error(GncTlsErrorCode::NumericalBound,
                  "GNC-TLS mu schedule would exceed the configured numerical bound"));
      }
      mu *= config_.mu_step;
    }
  }

  return core::Result<GncTlsReport, GncTlsError>::failure(
      error(GncTlsErrorCode::IterationLimit,
            "GNC-TLS did not reach stable binary weights within the iteration bound"));
}

}  // namespace meridian::global
