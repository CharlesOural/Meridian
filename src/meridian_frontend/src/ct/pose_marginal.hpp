#pragma once

#include <Eigen/Core>

#include "ct/spline_window.hpp"
#include "meridian/common/time.hpp"

namespace ceres {
class Problem;
}  // namespace ceres

namespace meridian::ct {

// Symmetrised pseudo-inverse: invert eigenvalues above an absolute floor relative to
// the largest, zero the rest. Returns false if no eigenvalue clears the floor (the
// matrix is effectively null), so the caller can reject a rank-deficient block.
bool robustInverse(const Eigen::MatrixXd& A, Eigen::MatrixXd* out);

// Pose-block marginal covariance (translation-first [rho; phi]) of T_W_Fe(stamp) from
// the solved window posterior: inverts the joint information of the pose segment's knots
// over every residual touching them, then chains through the spline pose Jacobian at
// stamp. Returns false when the joint information is rank-deficient. Identical math
// whether the hot thread runs it on the live problem or the keyframe worker runs it on a
// rebuilt clone, so a bit-exact rebuilt problem yields a bit-exact marginal.
bool poseMarginalFromProblem(ceres::Problem& problem, SplineWindow& spline, Timestamp stamp,
                             Eigen::Matrix<double, 6, 6>* cov);

}  // namespace meridian::ct
