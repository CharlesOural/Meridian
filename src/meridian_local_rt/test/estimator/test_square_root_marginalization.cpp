#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "meridian/local_rt/estimator/square_root_marginalization.hpp"

namespace meridian::local_rt::estimator {
namespace {

Eigen::MatrixXd deterministicMatrix(Eigen::Index rows, Eigen::Index columns, double phase) {
  Eigen::MatrixXd matrix(rows, columns);
  for (Eigen::Index row = 0; row < rows; ++row) {
    for (Eigen::Index column = 0; column < columns; ++column) {
      const double index = static_cast<double>((row + 1) * (column + 1));
      matrix(row, column) = std::sin(0.071 * index + phase) +
                            0.3 * std::cos(0.113 * static_cast<double>(row + column) - phase);
    }
  }
  return matrix;
}

void expectReducedCostParity(const Eigen::VectorXd& residual, const Eigen::MatrixXd& jacobian,
                             const SquareRootMarginalizationResult& result,
                             double elimination_tolerance) {
  const Eigen::MatrixXd marginalized = jacobian.leftCols(15);
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solve(marginalized);
  solve.setThreshold(elimination_tolerance);

  const Eigen::Index retained_size = jacobian.cols() - 15;
  for (int sample = 0; sample < 5; ++sample) {
    Eigen::VectorXd retained(retained_size);
    for (Eigen::Index index = 0; index < retained_size; ++index) {
      retained[index] = 0.17 * static_cast<double>(sample - 2) +
                        0.03 * static_cast<double>(index * index - 2 * index);
    }
    const Eigen::VectorXd conditioned = residual + jacobian.rightCols(retained_size) * retained;
    const Eigen::VectorXd eliminated = solve.solve(-conditioned);
    const double full_minimum = (conditioned + marginalized * eliminated).squaredNorm();
    const double reduced =
        (result.square_root_matrix * retained - result.right_hand_side).squaredNorm() +
        result.omitted_constant_squared_residual;
    EXPECT_NEAR(full_minimum, reduced, 2.0e-8 * std::max(1.0, full_minimum));
  }
}

TEST(SquareRootMarginalization, PreservesTheConditionedLeastSquaresCostWithoutNormalEquations) {
  constexpr Eigen::Index kRows = 64;
  constexpr Eigen::Index kRetained = 21;
  Eigen::MatrixXd jacobian(kRows, 15 + kRetained);
  jacobian.leftCols(15) = deterministicMatrix(kRows, 15, 0.2);
  jacobian.rightCols(kRetained) = deterministicMatrix(kRows, kRetained, -0.7);
  // Break the shared deterministic basis and make both partitions well ranked.
  jacobian.leftCols(15).diagonal().array() += 2.0;
  jacobian.rightCols(kRetained).topRows(kRetained).diagonal().array() += 1.5;
  const Eigen::VectorXd residual = deterministicMatrix(kRows, 1, 1.1);
  const SquareRootMarginalizationOptions options{
      .elimination_rank_tolerance = 1.0e-11,
      .compression_rank_tolerance = 1.0e-11,
  };

  const SquareRootMarginalizationResult result =
      squareRootMarginalizeFirstState(residual, jacobian, options);
  EXPECT_EQ(result.eliminated_rank, 15U);
  EXPECT_EQ(result.retained_rank, static_cast<std::size_t>(kRetained));
  EXPECT_EQ(result.square_root_matrix.rows(), kRetained);
  EXPECT_EQ(result.square_root_matrix.cols(), kRetained);
  EXPECT_EQ(result.right_hand_side.size(), kRetained);
  EXPECT_GE(result.omitted_constant_squared_residual, 0.0);
  expectReducedCostParity(residual, jacobian, result, options.elimination_rank_tolerance);
}

TEST(SquareRootMarginalization, PreservesRankDeficientEliminationAndRetainedNullspaces) {
  constexpr Eigen::Index kRows = 46;
  constexpr Eigen::Index kRetained = 12;
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(kRows, 15 + kRetained);
  const Eigen::MatrixXd eliminated_basis = deterministicMatrix(kRows, 9, 0.4);
  jacobian.leftCols(9) = eliminated_basis;
  jacobian.col(9) = eliminated_basis.col(0) + 0.5 * eliminated_basis.col(2);
  jacobian.col(10) = -2.0 * eliminated_basis.col(3);
  jacobian.col(11) = eliminated_basis.col(4) - eliminated_basis.col(5);
  jacobian.col(12) = eliminated_basis.col(6);
  jacobian.col(13) = eliminated_basis.col(7) + eliminated_basis.col(8);
  jacobian.col(14) = eliminated_basis.col(1);

  const Eigen::MatrixXd retained_basis = deterministicMatrix(kRows, 7, -0.3);
  jacobian.middleCols(15, 7) = retained_basis;
  jacobian.col(22) = retained_basis.col(0) - retained_basis.col(1);
  jacobian.col(23) = 0.25 * retained_basis.col(2);
  jacobian.col(24) = retained_basis.col(4) + retained_basis.col(5);
  jacobian.col(25) = retained_basis.col(6);
  jacobian.col(26) = -retained_basis.col(3);
  const Eigen::VectorXd residual = deterministicMatrix(kRows, 1, 0.9);
  const SquareRootMarginalizationOptions options{
      .elimination_rank_tolerance = 1.0e-9,
      .compression_rank_tolerance = 1.0e-9,
  };

  const SquareRootMarginalizationResult result =
      squareRootMarginalizeFirstState(residual, jacobian, options);
  EXPECT_EQ(result.eliminated_rank, 9U);
  EXPECT_LE(result.retained_rank, 7U);
  EXPECT_EQ(result.square_root_matrix.cols(), kRetained);
  expectReducedCostParity(residual, jacobian, result, options.elimination_rank_tolerance);
}

TEST(SquareRootMarginalization, RejectsInvalidShapesCoefficientsAndThresholds) {
  const Eigen::VectorXd residual = Eigen::VectorXd::Zero(4);
  EXPECT_THROW(
      static_cast<void>(squareRootMarginalizeFirstState(residual, Eigen::MatrixXd::Zero(3, 20))),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(squareRootMarginalizeFirstState(residual, Eigen::MatrixXd::Zero(4, 15))),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(squareRootMarginalizeFirstState(
                   residual, Eigen::MatrixXd::Zero(4, 20),
                   {.elimination_rank_tolerance = 0.0, .compression_rank_tolerance = 1.0e-10})),
               std::invalid_argument);
}

}  // namespace
}  // namespace meridian::local_rt::estimator
