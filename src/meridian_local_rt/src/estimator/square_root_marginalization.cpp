#include "meridian/local_rt/estimator/square_root_marginalization.hpp"

#include <Eigen/QR>
#include <cmath>
#include <stdexcept>
#include <string>

namespace meridian::local_rt::estimator {
namespace {

constexpr Eigen::Index kNavigationStateLocalSize = 15;

void validateTolerance(double tolerance, const char* name) {
  if (!std::isfinite(tolerance) || tolerance <= 0.0 || tolerance >= 1.0) {
    throw std::invalid_argument(std::string(name) + " must be finite and lie in (0, 1)");
  }
}

}  // namespace

SquareRootMarginalizationResult squareRootMarginalizeFirstState(
    const Eigen::VectorXd& residual, const Eigen::MatrixXd& local_jacobian,
    SquareRootMarginalizationOptions options) {
  validateTolerance(options.elimination_rank_tolerance, "elimination_rank_tolerance");
  validateTolerance(options.compression_rank_tolerance, "compression_rank_tolerance");
  if (residual.size() == 0 || local_jacobian.rows() != residual.size() ||
      local_jacobian.cols() <= kNavigationStateLocalSize || !residual.array().isFinite().all() ||
      !local_jacobian.array().isFinite().all()) {
    throw std::invalid_argument(
        "square-root marginalization requires finite r,J with matching positive rows and "
        "more than 15 local columns");
  }

  const Eigen::Index retained_size = local_jacobian.cols() - kNavigationStateLocalSize;
  const Eigen::MatrixXd marginalized_jacobian = local_jacobian.leftCols(kNavigationStateLocalSize);
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> elimination(marginalized_jacobian);
  elimination.setThreshold(options.elimination_rank_tolerance);
  const Eigen::Index eliminated_rank = elimination.rank();

  Eigen::MatrixXd retained_and_residual(local_jacobian.rows(), retained_size + 1);
  retained_and_residual.leftCols(retained_size) = local_jacobian.rightCols(retained_size);
  retained_and_residual.rightCols<1>() = residual;
  const Eigen::MatrixXd projected = elimination.householderQ().adjoint() * retained_and_residual;
  const Eigen::Index projected_rows = projected.rows() - eliminated_rank;

  SquareRootMarginalizationResult result;
  result.eliminated_rank = static_cast<std::size_t>(eliminated_rank);
  if (projected_rows == 0) {
    result.square_root_matrix.resize(0, retained_size);
    result.right_hand_side.resize(0);
    return result;
  }

  const Eigen::MatrixXd reduced_jacobian =
      projected.bottomRows(projected_rows).leftCols(retained_size);
  const Eigen::VectorXd reduced_residual = projected.bottomRows(projected_rows).rightCols<1>();

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> compression(reduced_jacobian);
  compression.setThreshold(options.compression_rank_tolerance);
  const Eigen::Index retained_rank = compression.rank();
  result.retained_rank = static_cast<std::size_t>(retained_rank);

  const Eigen::VectorXd transformed_residual =
      compression.householderQ().adjoint() * reduced_residual;
  result.omitted_constant_squared_residual =
      transformed_residual.tail(projected_rows - retained_rank).squaredNorm();

  if (retained_rank == 0) {
    result.square_root_matrix.resize(0, retained_size);
    result.right_hand_side.resize(0);
    return result;
  }

  Eigen::MatrixXd upper = Eigen::MatrixXd::Zero(retained_rank, retained_size);
  const Eigen::MatrixXd packed_qr = compression.matrixR();
  for (Eigen::Index row = 0; row < retained_rank; ++row) {
    for (Eigen::Index column = row; column < retained_size; ++column) {
      upper(row, column) = packed_qr(row, column);
    }
  }
  result.square_root_matrix = upper * compression.colsPermutation().transpose();
  // Q^T(c + A*x) = c' + R*P^T*x. The public prior convention is A*x - b.
  result.right_hand_side = -transformed_residual.head(retained_rank);
  if (!result.square_root_matrix.array().isFinite().all() ||
      !result.right_hand_side.array().isFinite().all() ||
      !std::isfinite(result.omitted_constant_squared_residual)) {
    throw std::runtime_error("square-root marginalization produced non-finite output");
  }
  return result;
}

}  // namespace meridian::local_rt::estimator
