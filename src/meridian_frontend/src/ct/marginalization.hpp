#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <ceres/problem.h>

namespace ceres {
class CostFunction;
}  // namespace ceres

namespace meridian::ct {

// Gaussian prior over a set of parameter blocks, produced by Schur-complementing
// dropped blocks out of a linearized subproblem. Quaternion blocks are handled in
// their 3-DoF tangent through ceres::EigenQuaternionManifold -- the same manifold
// the window solver installs on the SO(3) knots -- so the prior boxminus, its
// Jacobian lift, and every other factor on those blocks share one tangent
// convention. Each block keeps its own local size; no cross-block reordering.
class MarginalizationPrior {
 public:
  // Describes one block kept in the prior: pointer, global size, local size
  // (4->3 for quaternions), and its linearization value (copied).
  struct Block {
    double* ptr = nullptr;
    int global_size = 0;
    int local_size = 0;
  };

  // Builds the prior from a linear system H dx = -b expressed over
  // kept+dropped tangent blocks: drops the tail [kept | dropped] layout via
  // Schur complement and stores sqrt-information J0 and residual r0 with the
  // kept blocks' linearization points.
  static std::unique_ptr<MarginalizationPrior> fromSchur(
      const Eigen::MatrixXd& H, const Eigen::VectorXd& b,
      const std::vector<Block>& kept, int dropped_dim);

  // Ceres cost: residual = J0 * (x boxminus x0) + r0 over the kept blocks. The
  // returned object is owned by the caller (hand it to ceres::Problem, which
  // takes ownership, or delete it).
  ceres::CostFunction* makeCost() const;

  const std::vector<Block>& blocks() const { return blocks_; }
  int residualDim() const { return static_cast<int>(r0_.size()); }

  // The kept tangent dimension (sum of local sizes) and the stored quantities,
  // exposed for tests and for the window manager's diagnostics.
  int keptTangentDim() const { return static_cast<int>(j0_.cols()); }
  const Eigen::MatrixXd& sqrtInfoJacobian() const { return j0_; }
  const Eigen::VectorXd& residual0() const { return r0_; }

  // Tangent-space information matrix J0^T J0 of the kept blocks; the analytic
  // marginal a window manager compares against.
  Eigen::MatrixXd information() const { return j0_.transpose() * j0_; }

  // The stored linearization value of kept block i (global_size doubles).
  const std::vector<double>& linearizationPoint(int i) const {
    return lin_points_[static_cast<std::size_t>(i)];
  }

 private:
  std::vector<Block> blocks_;
  std::vector<std::vector<double>> lin_points_;  // per kept block, global size
  Eigen::MatrixXd j0_;                           // sqrt-information (kept tangent)
  Eigen::VectorXd r0_;                           // sqrt-info-weighted residual
};

// Helper that linearizes a ceres::Problem restricted to the given residual block
// ids, assembles the tangent-space H/b in a caller-given block order, and returns
// them (used by the window manager to build the Schur input).
//
// Each block's contribution occupies a local_size-wide column range; quaternion
// blocks (global 4, local 3) are reduced to their right tangent via the manifold
// installed on the Problem. Residual blocks not touching some listed block leave
// that block's rows/columns at zero. H = J^T J, b = J^T r over the restricted set.
void linearizeBlocks(ceres::Problem& problem,
                     const std::vector<ceres::ResidualBlockId>& residuals,
                     const std::vector<MarginalizationPrior::Block>& order,
                     Eigen::MatrixXd* H, Eigen::VectorXd* b);

}  // namespace meridian::ct
