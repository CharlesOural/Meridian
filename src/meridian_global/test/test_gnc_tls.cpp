#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "meridian/global/gnc_tls.hpp"

namespace meridian::global {
namespace {

class StaticCostShadowSolver final : public GncTlsShadowSolverApi {
public:
  std::vector<ShadowKnownInlierCost> known_costs;
  std::vector<ShadowCandidateCost> candidate_costs;
  std::vector<GncTlsShadowSolveRequest> requests;
  std::size_t inner_iterations{3U};
  bool finite_solution{true};

  core::Result<GncTlsShadowSolveResult, ShadowSolveFailure> solve(
      const GncTlsShadowSolveRequest& request) override {
    requests.push_back(request);
    return core::Result<GncTlsShadowSolveResult, ShadowSolveFailure>::success(
        GncTlsShadowSolveResult{inner_iterations, finite_solution, known_costs, candidate_costs});
  }
};

[[nodiscard]] KnownInlierFactor known(std::uint64_t id, std::uint32_t dof, double cutoff) {
  return KnownInlierFactor{GlobalFactorId(id), GncTlsFactorScale{dof, cutoff}};
}

[[nodiscard]] RobustCandidateFactor candidate(std::uint64_t id, std::uint32_t dof, double cutoff) {
  return RobustCandidateFactor{CandidateId(id), GncTlsFactorScale{dof, cutoff}};
}

[[nodiscard]] const GncTlsCandidateReport& reportFor(const GncTlsReport& report, CandidateId id) {
  const auto found = std::find_if(report.robust_candidates.begin(), report.robust_candidates.end(),
                                  [&](const auto& item) { return item.candidate_id == id; });
  EXPECT_NE(found, report.robust_candidates.end());
  return *found;
}

TEST(GncTlsController, KeepsKnownInliersAtOneAndSeparatesInlierFromOutlier) {
  StaticCostShadowSolver solver;
  solver.known_costs = {{GlobalFactorId(9U), 250.0}};
  solver.candidate_costs = {{CandidateId(2U), 1.0}, {CandidateId(1U), 100.0}};
  const std::vector<KnownInlierFactor> known_factors{known(9U, 6U, 12.592)};
  const std::vector<RobustCandidateFactor> candidates{candidate(2U, 1U, 3.841),
                                                      candidate(1U, 1U, 3.841)};

  GncTlsConfig config;
  config.mu_step = 2.0;
  GncTlsController controller(config);
  const auto result = controller.run(known_factors, candidates, solver);
  ASSERT_TRUE(result);
  const auto& report = result.value();
  EXPECT_EQ(report.convergence, GncTlsConvergenceReason::StableBinaryWeights);
  EXPECT_EQ(reportFor(report, CandidateId(1U)).disposition, GncTlsCandidateDisposition::TlsOutlier);
  EXPECT_LE(reportFor(report, CandidateId(1U)).final_weight, config.binary_weight_tolerance);
  EXPECT_EQ(reportFor(report, CandidateId(2U)).disposition, GncTlsCandidateDisposition::TlsInlier);
  EXPECT_GE(reportFor(report, CandidateId(2U)).final_weight, 1.0 - config.binary_weight_tolerance);
  ASSERT_FALSE(solver.requests.empty());
  for (const auto& request : solver.requests) {
    ASSERT_EQ(request.known_inliers.size(), 1U);
    EXPECT_EQ(request.known_inliers.front().factor_id, GlobalFactorId(9U));
    EXPECT_DOUBLE_EQ(request.known_inliers.front().weight, 1.0);
    EXPECT_EQ(request.robust_candidates.size(), 2U);
  }
  EXPECT_EQ(solver.requests.front().phase, ShadowSolvePhase::InitialFullGraph);
  EXPECT_FALSE(solver.requests.front().mu.has_value());
  EXPECT_EQ(solver.requests[1].phase, ShadowSolvePhase::GraduatedFullGraph);
  EXPECT_TRUE(solver.requests[1].mu.has_value());
}

TEST(GncTlsController, NormalizesMixedDegreesOfFreedomByEachCalibratedCutoff) {
  StaticCostShadowSolver solver;
  solver.candidate_costs = {{CandidateId(10U), 6.0}, {CandidateId(11U), 6.0}};
  // Approximate 95% chi-squared gates for one and four degrees of freedom.
  const std::vector<RobustCandidateFactor> candidates{candidate(10U, 1U, 3.841),
                                                      candidate(11U, 4U, 9.488)};
  GncTlsConfig config;
  config.mu_step = 2.0;
  GncTlsController controller(config);

  const auto result = controller.run({}, candidates, solver);
  ASSERT_TRUE(result);
  const auto& one_dof = reportFor(result.value(), CandidateId(10U));
  const auto& four_dof = reportFor(result.value(), CandidateId(11U));
  EXPECT_NEAR(one_dof.cost.normalized_squared_cost, 6.0 / 3.841, 1.0e-12);
  EXPECT_NEAR(four_dof.cost.normalized_squared_cost, 6.0 / 9.488, 1.0e-12);
  EXPECT_EQ(one_dof.disposition, GncTlsCandidateDisposition::TlsOutlier);
  EXPECT_EQ(four_dof.disposition, GncTlsCandidateDisposition::TlsInlier);
}

TEST(GncTlsController, ScheduleAndReportsAreDeterministicForUnsortedInputs) {
  const std::vector<KnownInlierFactor> known_factors{known(7U, 3U, 7.815), known(3U, 6U, 12.592)};
  const std::vector<RobustCandidateFactor> candidates{
      candidate(8U, 2U, 5.991), candidate(1U, 6U, 12.592), candidate(4U, 3U, 7.815)};
  auto make_solver = [] {
    StaticCostShadowSolver solver;
    solver.known_costs = {{GlobalFactorId(3U), 2.0}, {GlobalFactorId(7U), 1.0}};
    solver.candidate_costs = {
        {CandidateId(4U), 8.0}, {CandidateId(8U), 0.2}, {CandidateId(1U), 100.0}};
    return solver;
  };
  GncTlsConfig config;
  config.mu_step = 1.8;
  GncTlsController controller(config);
  auto first_solver = make_solver();
  auto second_solver = make_solver();
  const auto first = controller.run(known_factors, candidates, first_solver);
  const auto second = controller.run(known_factors, candidates, second_solver);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  ASSERT_EQ(first_solver.requests.size(), second_solver.requests.size());
  ASSERT_EQ(first.value().iterations.size(), second.value().iterations.size());
  for (std::size_t call = 0; call < first_solver.requests.size(); ++call) {
    const auto& left = first_solver.requests[call];
    const auto& right = second_solver.requests[call];
    EXPECT_EQ(left.phase, right.phase);
    EXPECT_EQ(left.mu, right.mu);
    ASSERT_EQ(left.known_inliers.size(), 2U);
    EXPECT_EQ(left.known_inliers[0].factor_id, GlobalFactorId(3U));
    EXPECT_EQ(left.known_inliers[1].factor_id, GlobalFactorId(7U));
    ASSERT_EQ(left.robust_candidates.size(), right.robust_candidates.size());
    for (std::size_t index = 0; index < left.robust_candidates.size(); ++index) {
      EXPECT_EQ(left.robust_candidates[index].candidate_id,
                right.robust_candidates[index].candidate_id);
      EXPECT_DOUBLE_EQ(left.robust_candidates[index].weight, right.robust_candidates[index].weight);
    }
  }
  for (std::size_t index = 0; index < first.value().iterations.size(); ++index) {
    EXPECT_EQ(first.value().iterations[index].mu, second.value().iterations[index].mu);
    EXPECT_DOUBLE_EQ(first.value().iterations[index].normalized_weighted_cost,
                     second.value().iterations[index].normalized_weighted_cost);
  }
}

TEST(GncTlsController, CandidateCapacityFailsBeforeCallingShadowSolver) {
  GncTlsConfig config;
  config.maximum_robust_candidates = 2U;
  GncTlsController controller(config);
  StaticCostShadowSolver solver;
  const std::vector<RobustCandidateFactor> candidates{
      candidate(1U, 1U, 3.841), candidate(2U, 1U, 3.841), candidate(3U, 1U, 3.841)};
  const auto result = controller.run({}, candidates, solver);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, GncTlsErrorCode::RobustCandidateCapacity);
  EXPECT_TRUE(solver.requests.empty());
}

TEST(GncTlsController, NonFiniteCostFailsWithoutASecondShadowSolve) {
  StaticCostShadowSolver solver;
  solver.candidate_costs = {{CandidateId(1U), std::numeric_limits<double>::quiet_NaN()}};
  const std::vector<RobustCandidateFactor> candidates{candidate(1U, 3U, 7.815)};
  GncTlsController controller;
  const auto result = controller.run({}, candidates, solver);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, GncTlsErrorCode::InvalidWhitenedSquaredCost);
  ASSERT_TRUE(result.error().candidate_id.has_value());
  EXPECT_EQ(*result.error().candidate_id, CandidateId(1U));
  EXPECT_EQ(solver.requests.size(), 1U);
}

TEST(GncTlsController, ExactCutoffCannotBeGivenAHeuristicDisposition) {
  StaticCostShadowSolver solver;
  solver.candidate_costs = {{CandidateId(1U), 3.841}};
  const std::vector<RobustCandidateFactor> candidates{candidate(1U, 1U, 3.841)};
  GncTlsConfig config;
  config.maximum_iterations = 6U;
  config.mu_step = 2.0;
  GncTlsController controller(config);
  const auto result = controller.run({}, candidates, solver);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, GncTlsErrorCode::IterationLimit);
  EXPECT_EQ(solver.requests.size(), config.maximum_iterations + 1U);
}

}  // namespace
}  // namespace meridian::global
