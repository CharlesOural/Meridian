#include "ct/marginalization.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <gtest/gtest.h>
#include <sophus/so3.hpp>

using meridian::ct::linearizeBlocks;
using meridian::ct::MarginalizationPrior;
using Block = MarginalizationPrior::Block;

namespace {

// A dense linear residual r = A x - c over a list of blocks (each a contiguous
// slice of the stacked state). Stored as a Ceres cost function so the same factor
// drives both linearizeBlocks and a real solve.
class LinearResidual final : public ceres::CostFunction {
 public:
  LinearResidual(std::vector<Eigen::MatrixXd> a_blocks, Eigen::VectorXd c)
      : a_blocks_(std::move(a_blocks)), c_(std::move(c)) {
    set_num_residuals(static_cast<int>(c_.size()));
    for (const auto& a : a_blocks_) {
      mutable_parameter_block_sizes()->push_back(static_cast<int>(a.cols()));
    }
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const int n = static_cast<int>(c_.size());
    Eigen::Map<Eigen::VectorXd> r(residuals, n);
    r = -c_;
    for (std::size_t i = 0; i < a_blocks_.size(); ++i) {
      Eigen::Map<const Eigen::VectorXd> x(parameters[i], a_blocks_[i].cols());
      r += a_blocks_[i] * x;
    }
    if (jacobians != nullptr) {
      for (std::size_t i = 0; i < a_blocks_.size(); ++i) {
        if (jacobians[i] == nullptr) {
          continue;
        }
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
            jac(jacobians[i], n, a_blocks_[i].cols());
        jac = a_blocks_[i];
      }
    }
    return true;
  }

 private:
  std::vector<Eigen::MatrixXd> a_blocks_;
  Eigen::VectorXd c_;
};

}  // namespace

// Euclidean-only: three R^2 blocks tied by Gaussian factors. The Schur complement
// of the dropped block must equal the analytic marginal information on (x1, x2).
TEST(Marginalization, EuclideanSchurMatchesAnalytic) {
  std::array<Eigen::Vector2d, 3> x = {Eigen::Vector2d(0.1, -0.2),
                                      Eigen::Vector2d(-0.3, 0.4),
                                      Eigen::Vector2d(0.5, 0.05)};

  ceres::Problem problem;
  for (auto& xi : x) {
    problem.AddParameterBlock(xi.data(), 2);
  }

  std::vector<ceres::ResidualBlockId> ids;

  // Block-anchor factors give each variable its own information.
  auto anchor = [&](int i, double w) {
    Eigen::MatrixXd a = w * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd c = a * Eigen::Vector2d(0.0, 0.0);
    auto* cost = new LinearResidual({a}, c);
    ids.push_back(problem.AddResidualBlock(cost, nullptr, x[i].data()));
  };
  anchor(0, 1.3);
  anchor(1, 0.9);
  anchor(2, 1.7);

  // Pairwise coupling factors so the joint information is dense across blocks.
  auto couple = [&](int i, int j, double w) {
    Eigen::MatrixXd ai = w * Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd aj = -w * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd c = Eigen::Vector2d(0.05 * w, -0.02 * w);
    auto* cost = new LinearResidual({ai, aj}, c);
    ids.push_back(problem.AddResidualBlock(cost, nullptr, x[i].data(),
                                           x[j].data()));
  };
  couple(0, 1, 0.7);
  couple(1, 2, 1.1);
  couple(0, 2, 0.5);

  // Kept = [x1, x2], dropped = x3 (each local == global == 2).
  std::vector<Block> kept = {{x[0].data(), 2, 2}, {x[1].data(), 2, 2}};
  std::vector<Block> order = kept;
  order.push_back({x[2].data(), 2, 2});

  Eigen::MatrixXd H;
  Eigen::VectorXd b;
  linearizeBlocks(problem, ids, order, &H, &b);
  ASSERT_EQ(H.rows(), 6);
  ASSERT_EQ(H.cols(), 6);

  const int k = 4, d = 2;
  const Eigen::MatrixXd H_kk = H.topLeftCorner(k, k);
  const Eigen::MatrixXd H_kd = H.topRightCorner(k, d);
  const Eigen::MatrixXd H_dk = H.bottomLeftCorner(d, k);
  const Eigen::MatrixXd H_dd = H.bottomRightCorner(d, d);
  const Eigen::MatrixXd analytic = H_kk - H_kd * H_dd.inverse() * H_dk;

  auto prior = MarginalizationPrior::fromSchur(H, b, kept, d);
  const Eigen::MatrixXd info = prior->information();

  EXPECT_TRUE(info.isApprox(analytic, 1e-9))
      << "info=\n" << info << "\nanalytic=\n" << analytic;
}

// Solving a fresh problem holding only the prior reproduces the joint solution's
// (x1, x2) components -- i.e. the original marginal mean of the kept blocks.
TEST(Marginalization, PriorReproducesMarginalMean) {
  std::array<Eigen::Vector2d, 3> x = {Eigen::Vector2d::Zero(),
                                      Eigen::Vector2d::Zero(),
                                      Eigen::Vector2d::Zero()};

  // Fixed factor data, chosen so the joint optimum is non-trivial.
  const std::array<double, 3> anchor_w = {1.3, 0.9, 1.7};
  const std::array<Eigen::Vector2d, 3> anchor_target = {
      Eigen::Vector2d(0.2, -0.1), Eigen::Vector2d(-0.4, 0.3),
      Eigen::Vector2d(0.6, 0.1)};
  struct Couple {
    int i, j;
    double w;
    Eigen::Vector2d off;
  };
  const std::array<Couple, 3> couples = {
      Couple{0, 1, 0.7, Eigen::Vector2d(0.05, -0.02)},
      Couple{1, 2, 1.1, Eigen::Vector2d(-0.03, 0.04)},
      Couple{0, 2, 0.5, Eigen::Vector2d(0.02, 0.01)}};

  auto buildProblem = [&](ceres::Problem& problem,
                          std::array<Eigen::Vector2d, 3>& s,
                          std::vector<ceres::ResidualBlockId>* ids) {
    for (auto& si : s) {
      problem.AddParameterBlock(si.data(), 2);
    }
    for (int i = 0; i < 3; ++i) {
      Eigen::MatrixXd a = anchor_w[i] * Eigen::MatrixXd::Identity(2, 2);
      Eigen::VectorXd c = anchor_w[i] * anchor_target[i];
      auto* cost = new LinearResidual({a}, c);
      auto id = problem.AddResidualBlock(cost, nullptr, s[i].data());
      if (ids != nullptr) ids->push_back(id);
    }
    for (const auto& cp : couples) {
      Eigen::MatrixXd ai = cp.w * Eigen::MatrixXd::Identity(2, 2);
      Eigen::MatrixXd aj = -cp.w * Eigen::MatrixXd::Identity(2, 2);
      Eigen::VectorXd c = cp.w * cp.off;
      auto* cost = new LinearResidual({ai, aj}, c);
      auto id = problem.AddResidualBlock(cost, nullptr, s[cp.i].data(),
                                         s[cp.j].data());
      if (ids != nullptr) ids->push_back(id);
    }
  };

  auto tightOptions = []() {
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = 100;
    opts.function_tolerance = 1e-16;
    opts.gradient_tolerance = 1e-16;
    opts.parameter_tolerance = 1e-16;
    return opts;
  };

  // Joint solve to get the reference marginal means of x1, x2.
  std::array<Eigen::Vector2d, 3> joint = x;
  {
    ceres::Problem problem;
    buildProblem(problem, joint, nullptr);
    ceres::Solver::Summary summary;
    ceres::Solve(tightOptions(), &problem, &summary);
  }

  // Linearize at zero, marginalize x3, build the prior.
  std::array<Eigen::Vector2d, 3> lin = {Eigen::Vector2d::Zero(),
                                        Eigen::Vector2d::Zero(),
                                        Eigen::Vector2d::Zero()};
  std::unique_ptr<MarginalizationPrior> prior;
  {
    ceres::Problem problem;
    std::vector<ceres::ResidualBlockId> ids;
    buildProblem(problem, lin, &ids);
    std::vector<Block> kept = {{lin[0].data(), 2, 2}, {lin[1].data(), 2, 2}};
    std::vector<Block> order = kept;
    order.push_back({lin[2].data(), 2, 2});
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    linearizeBlocks(problem, ids, order, &H, &b);
    prior = MarginalizationPrior::fromSchur(H, b, kept, 2);
  }

  // Solve a problem with only the prior over fresh x1, x2.
  std::array<Eigen::Vector2d, 2> kept_x = {Eigen::Vector2d::Zero(),
                                           Eigen::Vector2d::Zero()};
  {
    ceres::Problem problem;
    problem.AddParameterBlock(kept_x[0].data(), 2);
    problem.AddParameterBlock(kept_x[1].data(), 2);
    problem.AddResidualBlock(prior->makeCost(), nullptr, kept_x[0].data(),
                             kept_x[1].data());
    ceres::Solver::Summary summary;
    ceres::Solve(tightOptions(), &problem, &summary);
  }

  EXPECT_TRUE(kept_x[0].isApprox(joint[0], 1e-7))
      << "prior x1=" << kept_x[0].transpose()
      << " joint x1=" << joint[0].transpose();
  EXPECT_TRUE(kept_x[1].isApprox(joint[1], 1e-7))
      << "prior x2=" << kept_x[1].transpose()
      << " joint x2=" << joint[1].transpose();
}

// A residual linear in the quaternion's manifold tangent and the vector:
//   r = Wq * Minus(q, q0) + Wv * (v - v0),
// where Minus is the EigenQuaternionManifold tangent (the same convention the
// solver and the prior use for quaternion blocks). The global 4-column Jacobian is
// Wq * MinusJacobian(q); at q == q0 this is the exact derivative, so a problem
// built at (q0, v0) has zero residual and zero gradient there.
class QvResidual final : public ceres::CostFunction {
 public:
  QvResidual(Eigen::Matrix3d wq, Eigen::Matrix3d wv, Eigen::Quaterniond q0,
             Eigen::Vector3d v0, bool use_q, bool use_v)
      : wq_(std::move(wq)),
        wv_(std::move(wv)),
        v0_(std::move(v0)),
        use_q_(use_q),
        use_v_(use_v) {
    q0_[0] = q0.x();
    q0_[1] = q0.y();
    q0_[2] = q0.z();
    q0_[3] = q0.w();
    set_num_residuals(3);
    if (use_q_) mutable_parameter_block_sizes()->push_back(4);
    if (use_v_) mutable_parameter_block_sizes()->push_back(3);
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    ceres::EigenQuaternionManifold qm;
    int pi = 0;
    Eigen::Vector3d phi = Eigen::Vector3d::Zero();
    int q_param = -1;
    if (use_q_) {
      q_param = pi;
      qm.Minus(parameters[pi], q0_, phi.data());
      ++pi;
    }
    Eigen::Vector3d dv = Eigen::Vector3d::Zero();
    int v_param = -1;
    if (use_v_) {
      v_param = pi;
      dv = Eigen::Map<const Eigen::Vector3d>(parameters[pi]) - v0_;
      ++pi;
    }
    Eigen::Map<Eigen::Vector3d> r(residuals);
    r = wq_ * phi + wv_ * dv;

    if (jacobians != nullptr) {
      if (use_q_ && jacobians[q_param] != nullptr) {
        Eigen::Matrix<double, 3, 4, Eigen::RowMajor> mj;
        qm.MinusJacobian(parameters[q_param], mj.data());
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jq(
            jacobians[q_param]);
        jq = wq_ * mj;
      }
      if (use_v_ && jacobians[v_param] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jv(
            jacobians[v_param]);
        jv = wv_;
      }
    }
    return true;
  }

 private:
  Eigen::Matrix3d wq_, wv_;
  double q0_[4];
  Eigen::Vector3d v0_;
  bool use_q_, use_v_;
};

// Quaternion kept block + vector dropped block. At the linearization point the
// prior residual is zero; a small quaternion perturbation produces a prior
// Gauss-Newton step matching the analytically marginalized system.
TEST(Marginalization, QuaternionPriorZeroAndGaussNewton) {
  // Linearization point: a non-identity quaternion and a vector.
  Eigen::Quaterniond q0 =
      Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d(0.2, -0.5, 0.8)
                                                    .normalized()));
  q0.normalize();
  Eigen::Vector3d v0(0.4, -0.2, 0.1);

  // q storage in Eigen (x, y, z, w) order; v storage is the 3-vector.
  std::array<double, 4> q = {q0.x(), q0.y(), q0.z(), q0.w()};
  Eigen::Vector3d v = v0;

  // Factors built so every residual is exactly zero at (q0, v0): the linearized
  // gradient b vanishes, hence the prior residual r0 must be zero at the point.
  const Eigen::Matrix3d W1 =
      (Eigen::Matrix3d() << 1.2, 0.1, 0.0, 0.1, 0.9, 0.05, 0.0, 0.05, 1.4)
          .finished();
  const Eigen::Matrix3d W2a =
      (Eigen::Matrix3d() << 0.5, 0.0, 0.1, 0.0, 0.6, 0.0, 0.1, 0.0, 0.4)
          .finished();
  const Eigen::Matrix3d W2b =
      (Eigen::Matrix3d() << 0.3, 0.05, 0.0, 0.05, 0.35, 0.0, 0.0, 0.0, 0.45)
          .finished();
  const Eigen::Matrix3d W3 =
      (Eigen::Matrix3d() << 1.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.8)
          .finished();

  ceres::Problem problem;
  problem.AddParameterBlock(q.data(), 4, new ceres::EigenQuaternionManifold());
  problem.AddParameterBlock(v.data(), 3);
  std::vector<ceres::ResidualBlockId> ids;
  ids.push_back(problem.AddResidualBlock(
      new QvResidual(W1, Eigen::Matrix3d::Zero(), q0, v0, true, false), nullptr,
      q.data()));
  ids.push_back(problem.AddResidualBlock(
      new QvResidual(W2a, W2b, q0, v0, true, true), nullptr, q.data(),
      v.data()));
  ids.push_back(problem.AddResidualBlock(
      new QvResidual(Eigen::Matrix3d::Zero(), W3, q0, v0, false, true), nullptr,
      v.data()));

  // Kept = quaternion (local 3, global 4); dropped = vector (3).
  std::vector<Block> kept = {{q.data(), 4, 3}};
  std::vector<Block> order = kept;
  order.push_back({v.data(), 3, 3});

  Eigen::MatrixXd H;
  Eigen::VectorXd b;
  linearizeBlocks(problem, ids, order, &H, &b);
  ASSERT_EQ(H.rows(), 6);

  // b must be ~0 at the linearization point (all residuals vanish there).
  EXPECT_LT(b.norm(), 1e-9) << "b=" << b.transpose();

  auto prior = MarginalizationPrior::fromSchur(H, b, kept, 3);

  // Residual is zero at the linearization point.
  {
    ceres::CostFunction* cost = prior->makeCost();
    std::vector<double> r(prior->residualDim());
    const double* params[] = {q.data()};
    cost->Evaluate(params, r.data(), nullptr);
    double rn = 0.0;
    for (double e : r) rn += e * e;
    EXPECT_LT(std::sqrt(rn), 1e-9);
    delete cost;
  }

  // Analytic marginal information on the quaternion tangent.
  const Eigen::Matrix3d H_kk = H.topLeftCorner(3, 3);
  const Eigen::MatrixXd H_kd = H.topRightCorner(3, 3);
  const Eigen::MatrixXd H_dd = H.bottomRightCorner(3, 3);
  const Eigen::Matrix3d info_marg =
      H_kk - H_kd * H_dd.inverse() * H_kd.transpose();

  // The prior's information matches the analytic marginal.
  EXPECT_TRUE(prior->information().isApprox(info_marg, 1e-9));

  // Gauss-Newton step from the prior alone for a small quaternion perturbation.
  const Eigen::Vector3d delta(1e-3, -2e-3, 1.5e-3);
  Eigen::Quaterniond q_pert = q0 * Sophus::SO3d::exp(delta).unit_quaternion();
  q_pert.normalize();
  std::array<double, 4> qp = {q_pert.x(), q_pert.y(), q_pert.z(), q_pert.w()};

  ceres::CostFunction* cost = prior->makeCost();
  const int n = prior->residualDim();
  std::vector<double> r(n);
  Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> Jg(n, 4);
  const double* params[] = {qp.data()};
  double* jacs[] = {Jg.data()};
  cost->Evaluate(params, r.data(), jacs);

  // Reduce the global 3x4 Jacobian to the 3-DoF tangent via the manifold's
  // plus-Jacobian, then take the Gauss-Newton step in the tangent.
  ceres::EigenQuaternionManifold qm;
  Eigen::Matrix<double, 4, 3, Eigen::RowMajor> plus_jac;
  qm.PlusJacobian(qp.data(), plus_jac.data());
  const Eigen::MatrixXd J_local = Jg * plus_jac;
  Eigen::Map<Eigen::VectorXd> rv(r.data(), n);
  const Eigen::Vector3d gn_step =
      -(J_local.transpose() * J_local)
           .ldlt()
           .solve(J_local.transpose() * rv);
  delete cost;

  // The prior residual at the perturbed point is J0 * dx, with dx the manifold
  // boxminus of the perturbation; reducing the Jacobian recovers J0, so the
  // Gauss-Newton step drives the tangent perturbation back to zero, i.e.
  // gn_step ~ -Minus(q_pert, q0).
  Eigen::Vector3d dx_pert;
  qm.Minus(qp.data(), q.data(), dx_pert.data());
  EXPECT_TRUE(gn_step.isApprox(-dx_pert, 1e-4))
      << "gn_step=" << gn_step.transpose()
      << " -dx=" << (-dx_pert).transpose();
}

// The information-deflation scale: scale s multiplies the marginal information by
// exactly s (sqrt(s) on J0 and r0), and the default scale 1.0 must take the
// untouched code path so its J0/r0 are bit-identical to a build that never passes
// the parameter.
TEST(Marginalization, ScaleDeflatesInformationDefaultBitExact) {
  // A well-conditioned 3-row system over one kept R^2 block and one dropped R^2
  // block, with a non-zero gradient so r0 is exercised too.
  const int total = 4;
  Eigen::MatrixXd J(3, total);
  J.row(0) << 1.0, 0.2, 0.5, -0.1;
  J.row(1) << 0.0, 1.1, 0.3, 0.4;
  J.row(2) << 0.2, 0.4, 1.0, 0.6;
  Eigen::VectorXd r(3);
  r << 0.1, -0.2, 0.05;
  const Eigen::MatrixXd H = J.transpose() * J;
  const Eigen::VectorXd b = J.transpose() * r;

  double kept_storage[2] = {0.3, -0.4};
  std::vector<Block> kept = {{kept_storage, 2, 2}};

  auto unscaled = MarginalizationPrior::fromSchur(H, b, kept, 2);
  auto scaled = MarginalizationPrior::fromSchur(H, b, kept, 2, 0.25);
  ASSERT_NE(unscaled, nullptr);
  ASSERT_NE(scaled, nullptr);

  // information() of the scaled prior is exactly 0.25x the unscaled one.
  const Eigen::MatrixXd info_u = unscaled->information();
  const Eigen::MatrixXd info_s = scaled->information();
  EXPECT_TRUE((info_s - 0.25 * info_u).cwiseAbs().maxCoeff() < 1e-12)
      << "scaled info=\n" << info_s << "\n0.25 * unscaled=\n" << 0.25 * info_u;

  // The Gauss-Newton step the prior alone induces is unchanged by the scale:
  // (s H)^-1 (s b) == H^-1 b.
  const Eigen::VectorXd step_u =
      unscaled->information().ldlt().solve(unscaled->sqrtInfoJacobian().transpose() *
                                           unscaled->residual0());
  const Eigen::VectorXd step_s =
      scaled->information().ldlt().solve(scaled->sqrtInfoJacobian().transpose() *
                                         scaled->residual0());
  EXPECT_TRUE((step_s - step_u).cwiseAbs().maxCoeff() < 1e-12);

  // scale == 1.0 must be BIT-identical to the default (no-parameter) build: the
  // 1.0 path skips the multiply entirely.
  auto one = MarginalizationPrior::fromSchur(H, b, kept, 2, 1.0);
  ASSERT_NE(one, nullptr);
  ASSERT_EQ(one->sqrtInfoJacobian().rows(), unscaled->sqrtInfoJacobian().rows());
  ASSERT_EQ(one->residual0().size(), unscaled->residual0().size());
  for (int i = 0; i < one->sqrtInfoJacobian().rows(); ++i) {
    for (int j = 0; j < one->sqrtInfoJacobian().cols(); ++j) {
      EXPECT_EQ(one->sqrtInfoJacobian()(i, j), unscaled->sqrtInfoJacobian()(i, j))
          << "J0 differs at (" << i << "," << j << ")";
    }
  }
  for (int i = 0; i < one->residual0().size(); ++i) {
    EXPECT_EQ(one->residual0()(i), unscaled->residual0()(i)) << "r0 differs at " << i;
  }
}

// The per-family residual telemetry in the window solver slices the vector
// returned by Problem::GetResidualBlocks by [begin, end) index ranges recorded as
// NumResidualBlocks() snapshots around each family's adds. That is only valid if
// Ceres returns residual blocks in insertion order; this pins the contract: the
// ids returned by AddResidualBlock, family by family, are exactly the slices of
// GetResidualBlocks at the recorded ranges.
TEST(Marginalization, ResidualBlockSlicingByInsertionOrder) {
  std::array<Eigen::Vector2d, 3> x = {Eigen::Vector2d(0.1, -0.2), Eigen::Vector2d(-0.3, 0.4),
                                      Eigen::Vector2d(0.5, 0.05)};
  ceres::Problem problem;
  for (auto& xi : x) {
    problem.AddParameterBlock(xi.data(), 2);
  }

  auto addOne = [&](int i) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Identity(2, 2);
    return problem.AddResidualBlock(new LinearResidual({a}, Eigen::Vector2d::Zero()), nullptr,
                                    x[static_cast<std::size_t>(i)].data());
  };

  // Three "families" of different sizes, with the per-family marks recorded the
  // same way the solver records them.
  std::vector<std::vector<ceres::ResidualBlockId>> family_ids(3);
  std::vector<std::pair<int, int>> fam_range(3);
  const std::array<int, 3> fam_sizes = {3, 1, 2};
  for (int f = 0; f < 3; ++f) {
    const int mark = problem.NumResidualBlocks();
    for (int k = 0; k < fam_sizes[static_cast<std::size_t>(f)]; ++k) {
      family_ids[static_cast<std::size_t>(f)].push_back(addOne((f + k) % 3));
    }
    fam_range[static_cast<std::size_t>(f)] = {mark, problem.NumResidualBlocks()};
  }

  std::vector<ceres::ResidualBlockId> all;
  problem.GetResidualBlocks(&all);
  ASSERT_EQ(static_cast<int>(all.size()), problem.NumResidualBlocks());

  for (int f = 0; f < 3; ++f) {
    const auto [begin, end] = fam_range[static_cast<std::size_t>(f)];
    ASSERT_EQ(end - begin, fam_sizes[static_cast<std::size_t>(f)]);
    for (int k = begin; k < end; ++k) {
      EXPECT_EQ(all[static_cast<std::size_t>(k)],
                family_ids[static_cast<std::size_t>(f)][static_cast<std::size_t>(k - begin)])
          << "insertion-order slice broken at family " << f << " element " << (k - begin);
    }
  }
}

namespace {

// Deterministic full-rank system over the kept stack [SO3 knot (4/3), r3 knot
// (3/3), bias gyro (3/3), bias accel (3/3)] plus one dropped r3 block: identity
// rows give every coordinate information, dense extra rows couple all blocks.
// `zero_col` (>= 0) removes every measurement of that kept tangent coordinate,
// making it a null direction of the marginal.
void buildKeptBiasSystem(Eigen::MatrixXd* H, Eigen::VectorXd* b, int zero_col = -1) {
  const int total = 15;  // 12 kept tangent coords + 3 dropped
  Eigen::MatrixXd J(total + 6, total);
  J.setZero();
  J.topRows(total).setIdentity();
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < total; ++c) {
      J(total + r, c) = 0.2 * std::sin(1.0 + 0.7 * r + 0.31 * c);
    }
  }
  if (zero_col >= 0) {
    J.col(zero_col).setZero();
  }
  Eigen::VectorXd r(total + 6);
  for (int i = 0; i < r.size(); ++i) {
    r(i) = 0.05 * std::cos(0.4 * i + 0.2);
  }
  *H = J.transpose() * J;
  *b = J.transpose() * r;
}

std::vector<Block> keptBiasBlocks(std::array<double, 4>& so3, Eigen::Vector3d& r3,
                                  Eigen::Vector3d& bg, Eigen::Vector3d& ba) {
  return {{so3.data(), 4, 3}, {r3.data(), 3, 3}, {bg.data(), 3, 3}, {ba.data(), 3, 3}};
}

// Gauss-Newton step the prior alone induces on the kept tangent: the minimizer of
// ||J0 dx + r0||^2 (full-rank information assumed by the caller).
Eigen::VectorXd priorGnStep(const MarginalizationPrior& prior) {
  return -prior.information().ldlt().solve(prior.sqrtInfoJacobian().transpose() *
                                           prior.residual0());
}

}  // namespace

// Structured forgetting with q on the bias coordinates only: the prior COVARIANCE
// obeys P' = P + diag(q) exactly -- bias diagonal grown by exactly q, knot marginal
// covariance unchanged -- and the prior mean (its Gauss-Newton minimizer) is
// preserved. The information matrix legitimately changes in every block through
// the off-diagonal coupling, so the assertions are on P, not on H.
TEST(Marginalization, StructuredForgettingInflatesBiasCovarianceOnly) {
  Eigen::MatrixXd H;
  Eigen::VectorXd b;
  buildKeptBiasSystem(&H, &b);

  std::array<double, 4> so3 = {0.0, 0.0, 0.0, 1.0};
  Eigen::Vector3d r3(0.1, 0.2, 0.3), bg(0.01, -0.02, 0.005), ba(0.1, -0.05, 0.2);
  std::vector<Block> kept = keptBiasBlocks(so3, r3, bg, ba);

  const double qg = 1e-3, qa = 1e-2;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(12);
  q.segment(6, 3).setConstant(qg);
  q.segment(9, 3).setConstant(qa);

  auto legacy = MarginalizationPrior::fromSchur(H, b, kept, 3);
  auto forget = MarginalizationPrior::fromSchur(H, b, kept, 3, 1.0, &q);
  ASSERT_NE(legacy, nullptr);
  ASSERT_NE(forget, nullptr);

  const Eigen::MatrixXd P0 = legacy->information().inverse();
  const Eigen::MatrixXd P1 = forget->information().inverse();
  const Eigen::MatrixXd dP = P1 - P0;

  // Knot subspace (coords 0..5) of the covariance is untouched.
  EXPECT_LT(dP.topLeftCorner(6, 6).cwiseAbs().maxCoeff(), 1e-12);
  // The whole covariance change is exactly the q diagonal.
  const Eigen::MatrixXd q_mat = Eigen::MatrixXd(q.asDiagonal());
  EXPECT_LT((dP - q_mat).cwiseAbs().maxCoeff(), 1e-10);
  // Bias variances strictly larger, by exactly q.
  for (int i = 6; i < 12; ++i) {
    EXPECT_GT(P1(i, i), P0(i, i)) << "bias coord " << i << " did not inflate";
    EXPECT_NEAR(dP(i, i), q(i), 1e-10);
  }

  // Mean preservation: the prior's Gauss-Newton minimizer is unchanged.
  EXPECT_LT((priorGnStep(*forget) - priorGnStep(*legacy)).cwiseAbs().maxCoeff(), 1e-10);
}

// The disabled paths execute no new arithmetic: a null q_diag and an all-zero
// q_diag must both produce J0/r0 bit-identical to a legacy build (here with a
// non-trivial scalar scale, so the legacy scalar path is also exercised).
TEST(Marginalization, StructuredForgettingDisabledBitExact) {
  Eigen::MatrixXd H;
  Eigen::VectorXd b;
  buildKeptBiasSystem(&H, &b);

  std::array<double, 4> so3 = {0.0, 0.0, 0.0, 1.0};
  Eigen::Vector3d r3(0.1, 0.2, 0.3), bg(0.01, -0.02, 0.005), ba(0.1, -0.05, 0.2);
  std::vector<Block> kept = keptBiasBlocks(so3, r3, bg, ba);

  auto legacy = MarginalizationPrior::fromSchur(H, b, kept, 3, 0.7);
  auto null_q = MarginalizationPrior::fromSchur(H, b, kept, 3, 0.7, nullptr);
  const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(12);
  auto zero_q = MarginalizationPrior::fromSchur(H, b, kept, 3, 0.7, &zeros);

  for (const auto* other : {null_q.get(), zero_q.get()}) {
    ASSERT_EQ(other->sqrtInfoJacobian().rows(), legacy->sqrtInfoJacobian().rows());
    for (int i = 0; i < legacy->sqrtInfoJacobian().rows(); ++i) {
      for (int j = 0; j < legacy->sqrtInfoJacobian().cols(); ++j) {
        EXPECT_EQ(other->sqrtInfoJacobian()(i, j), legacy->sqrtInfoJacobian()(i, j))
            << "J0 differs at (" << i << "," << j << ")";
      }
    }
    for (int i = 0; i < legacy->residual0().size(); ++i) {
      EXPECT_EQ(other->residual0()(i), legacy->residual0()(i)) << "r0 differs at " << i;
    }
  }
}

// Inflating a coordinate the marginal carries NO information about must not
// manufacture information: a covariance round trip would resurrect the null
// direction at strength 1/q; the information-space update must leave it null.
TEST(Marginalization, StructuredForgettingPreservesNullSpace) {
  Eigen::MatrixXd H;
  Eigen::VectorXd b;
  buildKeptBiasSystem(&H, &b, /*zero_col=*/6);  // first gyro-bias coord unmeasured

  std::array<double, 4> so3 = {0.0, 0.0, 0.0, 1.0};
  Eigen::Vector3d r3(0.1, 0.2, 0.3), bg(0.01, -0.02, 0.005), ba(0.1, -0.05, 0.2);
  std::vector<Block> kept = keptBiasBlocks(so3, r3, bg, ba);

  const double qg = 1e-3;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(12);
  q.segment(6, 3).setConstant(qg);

  auto forget = MarginalizationPrior::fromSchur(H, b, kept, 3, 1.0, &q);
  ASSERT_NE(forget, nullptr);

  const Eigen::MatrixXd info = forget->information();
  EXPECT_TRUE(info.allFinite());
  // The unmeasured coordinate stays unconstrained: nowhere near the 1/q = 1000
  // a covariance round trip would assign it.
  EXPECT_LT(info.row(6).cwiseAbs().maxCoeff(), 1e-9);
  EXPECT_LT(info.col(6).cwiseAbs().maxCoeff(), 1e-9);
  // The measured inflated coordinates still relax (information non-increasing on
  // their diagonal versus the un-inflated build).
  auto legacy = MarginalizationPrior::fromSchur(H, b, kept, 3);
  for (int i = 7; i < 9; ++i) {
    EXPECT_LE(info(i, i), legacy->information()(i, i) + 1e-12);
  }
}

// A singular dropped block (H_dd rank-deficient) must not produce NaNs: the
// eigenvalue floor zeroes the null directions of the inverse.
TEST(Marginalization, RankDeficientDroppedBlockNoNaN) {
  // Two kept scalars (as a single R^2 block) and a dropped R^2 block whose
  // information is rank 1 (singular). Built directly as H/b.
  const int total = 4;
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, total);
  // Residual rows touching kept dims 0,1 and dropped dim 2 only (dim 3 untouched
  // -> H_dd is singular along that direction).
  J.row(0) << 1.0, 0.0, 0.5, 0.0;
  J.row(1) << 0.0, 1.0, 0.3, 0.0;
  J.row(2) << 0.2, 0.4, 1.0, 0.0;
  Eigen::VectorXd r(3);
  r << 0.1, -0.2, 0.05;
  Eigen::MatrixXd H = J.transpose() * J;
  Eigen::VectorXd b = J.transpose() * r;

  double kept_storage[2] = {0.0, 0.0};
  std::vector<Block> kept = {{kept_storage, 2, 2}};

  auto prior = MarginalizationPrior::fromSchur(H, b, kept, 2);
  ASSERT_NE(prior, nullptr);

  const Eigen::MatrixXd info = prior->information();
  EXPECT_TRUE(info.allFinite()) << "info=\n" << info;
  EXPECT_TRUE(prior->sqrtInfoJacobian().allFinite());
  EXPECT_TRUE(prior->residual0().allFinite());

  // The kept information stays symmetric positive-semidefinite and finite.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(info);
  EXPECT_GE(es.eigenvalues().minCoeff(), -1e-9);

  // The resulting cost also evaluates without NaNs.
  ceres::CostFunction* cost = prior->makeCost();
  std::vector<double> rr(prior->residualDim());
  const double* params[] = {kept_storage};
  ASSERT_TRUE(cost->Evaluate(params, rr.data(), nullptr));
  for (double e : rr) EXPECT_TRUE(std::isfinite(e));
  delete cost;
}
