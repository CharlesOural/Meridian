#include "marginalization.hpp"

#include <cmath>
#include <cstring>
#include <utility>

#include <Eigen/Eigenvalues>
#include <ceres/cost_function.h>
#include <ceres/crs_matrix.h>
#include <ceres/manifold.h>
#include <ceres/problem.h>

namespace meridian::ct {

namespace {

// Eigenvalue floor for the dropped-block inverse and for the marginal
// sqrt-information. Eigenvalues at or below this are treated as null directions:
// their reciprocal is taken as zero (pseudo-inverse) so a rank-deficient dropped
// block contributes no spurious information instead of producing NaNs/infinities.
constexpr double kEigenFloor = 1e-8;

// Symmetrised pseudo-inverse of a symmetric matrix: invert eigenvalues above the
// floor, zero the rest. Robust when the block is singular or near-singular.
Eigen::MatrixXd robustInverse(const Eigen::MatrixXd& A) {
  const Eigen::MatrixXd sym = 0.5 * (A + A.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sym);
  const Eigen::VectorXd& lambda = es.eigenvalues();
  Eigen::VectorXd inv(lambda.size());
  for (int i = 0; i < lambda.size(); ++i) {
    inv(i) = (lambda(i) > kEigenFloor) ? 1.0 / lambda(i) : 0.0;
  }
  return es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
}

// The Ceres cost backing a MarginalizationPrior. residual = J0 * (x boxminus x0)
// + r0; the boxminus is the per-block tangent (quaternion: the EigenQuaternion
// manifold's Minus, Euclidean: x - x0). Jacobians are returned at GLOBAL parameter
// size: the kept local columns of J0 times the manifold's minus-Jacobian. Ceres
// then reduces that back to local size via the manifold's plus-Jacobian installed
// on the block; minus-Jac times plus-Jac at the same point is identity, so the
// reduced Jacobian stays fixed at J0 (the first-estimate linearization) while the
// residual tracks the live boxminus.
class PriorCost final : public ceres::CostFunction {
 public:
  PriorCost(std::vector<MarginalizationPrior::Block> blocks,
            std::vector<std::vector<double>> lin_points, Eigen::MatrixXd j0,
            Eigen::VectorXd r0)
      : blocks_(std::move(blocks)),
        lin_points_(std::move(lin_points)),
        j0_(std::move(j0)),
        r0_(std::move(r0)) {
    set_num_residuals(static_cast<int>(r0_.size()));
    for (const auto& blk : blocks_) {
      mutable_parameter_block_sizes()->push_back(blk.global_size);
    }
    // Column offset of each block inside the tangent stack J0 acts on.
    col_offset_.reserve(blocks_.size());
    int off = 0;
    for (const auto& blk : blocks_) {
      col_offset_.push_back(off);
      off += blk.local_size;
    }
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const int n = static_cast<int>(r0_.size());
    const int tangent = static_cast<int>(j0_.cols());

    // Stack the per-block tangent perturbation x boxminus x0.
    Eigen::VectorXd dx(tangent);
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
      const MarginalizationPrior::Block& blk = blocks_[i];
      const int off = col_offset_[i];
      const double* x = parameters[i];
      const double* x0 = lin_points_[i].data();
      if (blk.local_size == 3 && blk.global_size == 4) {
        quat_.Minus(x, x0, dx.segment(off, 3).data());
      } else {
        for (int j = 0; j < blk.local_size; ++j) {
          dx(off + j) = x[j] - x0[j];
        }
      }
    }

    Eigen::Map<Eigen::VectorXd>(residuals, n) = j0_ * dx + r0_;

    if (jacobians == nullptr) {
      return true;
    }
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
      if (jacobians[i] == nullptr) {
        continue;
      }
      const MarginalizationPrior::Block& blk = blocks_[i];
      const int off = col_offset_[i];
      // Row-major (num_residuals x global_size).
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          jac(jacobians[i], n, blk.global_size);
      jac.setZero();
      const Eigen::MatrixXd local = j0_.middleCols(off, blk.local_size);
      if (blk.local_size == 3 && blk.global_size == 4) {
        // d residual / d x_global = local * (d boxminus / d x_global): the
        // manifold's minus-Jacobian at the current point, a 3x4 lift.
        Eigen::Matrix<double, 3, 4, Eigen::RowMajor> minus_jac;
        quat_.MinusJacobian(parameters[i], minus_jac.data());
        jac = local * minus_jac;
      } else {
        jac.leftCols(blk.local_size) = local;
      }
    }
    return true;
  }

 private:
  std::vector<MarginalizationPrior::Block> blocks_;
  std::vector<std::vector<double>> lin_points_;
  std::vector<int> col_offset_;
  Eigen::MatrixXd j0_;
  Eigen::VectorXd r0_;
  ceres::EigenQuaternionManifold quat_;
};

}  // namespace

std::unique_ptr<MarginalizationPrior> MarginalizationPrior::fromSchur(
    const Eigen::MatrixXd& H, const Eigen::VectorXd& b,
    const std::vector<Block>& kept, int dropped_dim) {
  int k = 0;
  for (const Block& blk : kept) {
    k += blk.local_size;
  }
  const int d = dropped_dim;

  auto prior = std::unique_ptr<MarginalizationPrior>(new MarginalizationPrior());
  prior->blocks_ = kept;

  // Copy each kept block's linearization value (global size) from its storage.
  prior->lin_points_.reserve(kept.size());
  for (const Block& blk : kept) {
    std::vector<double> v(static_cast<std::size_t>(blk.global_size));
    if (blk.ptr != nullptr) {
      std::memcpy(v.data(), blk.ptr,
                  sizeof(double) * static_cast<std::size_t>(blk.global_size));
    }
    prior->lin_points_.push_back(std::move(v));
  }

  Eigen::MatrixXd h_marg;
  Eigen::VectorXd b_marg;
  if (d > 0) {
    const Eigen::MatrixXd H_kk = H.topLeftCorner(k, k);
    const Eigen::MatrixXd H_kd = H.topRightCorner(k, d);
    const Eigen::MatrixXd H_dk = H.bottomLeftCorner(d, k);
    const Eigen::MatrixXd H_dd = H.bottomRightCorner(d, d);
    const Eigen::VectorXd b_k = b.head(k);
    const Eigen::VectorXd b_d = b.tail(d);
    const Eigen::MatrixXd H_dd_inv = robustInverse(H_dd);
    const Eigen::MatrixXd kd_ddinv = H_kd * H_dd_inv;
    h_marg = H_kk - kd_ddinv * H_dk;
    b_marg = b_k - kd_ddinv * b_d;
  } else {
    h_marg = H.topLeftCorner(k, k);
    b_marg = b.head(k);
  }

  // Symmetrise and factor the marginal information into sqrt-information J0 with
  // J0^T J0 = H_marg (negative eigenvalues, from round-off, clamped to zero). The
  // residual r0 is chosen so J0^T r0 = b_marg on the supported subspace, making
  // the prior's Gauss-Newton step solve H_marg dx = -b_marg.
  const Eigen::MatrixXd sym = 0.5 * (h_marg + h_marg.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sym);
  const Eigen::VectorXd& lambda = es.eigenvalues();
  Eigen::VectorXd s_sqrt(k);
  Eigen::VectorXd s_inv_sqrt(k);
  for (int i = 0; i < k; ++i) {
    const double l = (lambda(i) > kEigenFloor) ? lambda(i) : 0.0;
    s_sqrt(i) = std::sqrt(l);
    s_inv_sqrt(i) = (l > 0.0) ? 1.0 / std::sqrt(l) : 0.0;
  }
  const Eigen::MatrixXd Vt = es.eigenvectors().transpose();
  prior->j0_ = s_sqrt.asDiagonal() * Vt;
  prior->r0_ = s_inv_sqrt.asDiagonal() * (Vt * b_marg);
  return prior;
}

ceres::CostFunction* MarginalizationPrior::makeCost() const {
  return new PriorCost(blocks_, lin_points_, j0_, r0_);
}

void linearizeBlocks(ceres::Problem& problem,
                     const std::vector<ceres::ResidualBlockId>& residuals,
                     const std::vector<MarginalizationPrior::Block>& order,
                     Eigen::MatrixXd* H, Eigen::VectorXd* b) {
  // Column offset and width (local size) of each ordered block in the tangent
  // layout the caller asked for.
  std::vector<int> col_offset(order.size());
  int total = 0;
  for (std::size_t i = 0; i < order.size(); ++i) {
    col_offset[i] = total;
    total += order[i].local_size;
  }

  H->setZero(total, total);
  b->setZero(total);
  if (residuals.empty() || total == 0) {
    return;
  }

  // Pin quaternion blocks (local 3, global 4) to the Eigen quaternion manifold so
  // that Problem::Evaluate returns their jacobian columns already reduced to the
  // 3-DoF tangent (J_local = J_global * PlusJacobian, applied internally). Only
  // blocks that are actually in the problem are passed to Evaluate; a listed block
  // the residuals never touch simply leaves its rows/columns at zero.
  std::vector<double*> eval_blocks;
  std::vector<int> eval_to_order;
  eval_blocks.reserve(order.size());
  eval_to_order.reserve(order.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    double* ptr = order[i].ptr;
    if (ptr == nullptr || !problem.HasParameterBlock(ptr)) {
      continue;
    }
    if (order[i].local_size == 3 && order[i].global_size == 4 &&
        problem.GetManifold(ptr) == nullptr) {
      problem.SetManifold(ptr, new ceres::EigenQuaternionManifold());
    }
    eval_blocks.push_back(ptr);
    eval_to_order.push_back(static_cast<int>(i));
  }
  if (eval_blocks.empty()) {
    return;
  }

  ceres::Problem::EvaluateOptions opts;
  opts.parameter_blocks = eval_blocks;
  opts.residual_blocks = residuals;
  opts.apply_loss_function = true;
  opts.num_threads = 1;

  double cost = 0.0;
  std::vector<double> res;
  ceres::CRSMatrix jac;
  problem.Evaluate(opts, &cost, &res, nullptr, &jac);

  const int rows = jac.num_rows;
  // Dense jacobian over the evaluated blocks, in the column order of eval_blocks
  // (each block contributes local_size columns, matching the installed manifolds).
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(rows, total);
  // Starting column of each evaluated block inside the CRS jacobian; Ceres lays
  // them out in parameter_blocks order using tangent sizes.
  std::vector<int> eval_col(eval_blocks.size());
  int c = 0;
  for (std::size_t e = 0; e < eval_blocks.size(); ++e) {
    eval_col[e] = c;
    c += order[static_cast<std::size_t>(eval_to_order[e])].local_size;
  }
  // Map a CRS source column to its destination column in the caller's layout.
  std::vector<int> src_to_dst(static_cast<std::size_t>(jac.num_cols), -1);
  for (std::size_t e = 0; e < eval_blocks.size(); ++e) {
    const int ls = order[static_cast<std::size_t>(eval_to_order[e])].local_size;
    const int dst0 = col_offset[static_cast<std::size_t>(eval_to_order[e])];
    for (int j = 0; j < ls; ++j) {
      src_to_dst[static_cast<std::size_t>(eval_col[e] + j)] = dst0 + j;
    }
  }
  for (int r = 0; r < rows; ++r) {
    for (int idx = jac.rows[r]; idx < jac.rows[r + 1]; ++idx) {
      const int dst = src_to_dst[static_cast<std::size_t>(jac.cols[idx])];
      if (dst >= 0) {
        J(r, dst) = jac.values[idx];
      }
    }
  }

  Eigen::Map<const Eigen::VectorXd> r(res.data(), rows);
  *H = J.transpose() * J;
  *b = J.transpose() * r;
}

}  // namespace meridian::ct
