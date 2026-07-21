#pragma once

#include <Eigen/Core>
#include <cstddef>

namespace meridian::local_rt::estimator {

struct SquareRootMarginalizationOptions final {
  // Relative pivot thresholds used independently for eliminating the first
  // navigation state and compressing the retained square-root residual.
  double elimination_rank_tolerance{1.0e-10};
  double compression_rank_tolerance{1.0e-10};
};

struct SquareRootMarginalizationResult final {
  // Reduced residual: square_root_matrix * delta_retained - right_hand_side.
  Eigen::MatrixXd square_root_matrix;
  Eigen::VectorXd right_hand_side;
  std::size_t eliminated_rank{};
  std::size_t retained_rank{};
  // Residual energy independent of retained variables and therefore omitted
  // from the returned Ceres prior.
  double omitted_constant_squared_residual{};
};

// Eliminates the first navigation state's 15 local columns from the linearized
// residual r + J * delta. The implementation projects with column-pivoted QR,
// then rank-compresses the retained residual with a second pivoted QR. It never
// forms normal equations and never adds damping. The output sign convention is
// A * delta_retained - b.
[[nodiscard]] SquareRootMarginalizationResult squareRootMarginalizeFirstState(
    const Eigen::VectorXd& residual, const Eigen::MatrixXd& local_jacobian,
    SquareRootMarginalizationOptions options = {});

}  // namespace meridian::local_rt::estimator
