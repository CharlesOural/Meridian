#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>

#include <Eigen/Core>
#include <boost/optional.hpp>
#include <cstdint>
#include <functional>

#include "gnss_factor.hpp"
#include "gnss_factor_refined.hpp"

namespace {

using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using meridian::backend::GnssFactor;
using meridian::backend::GnssFactorRefined;
using meridian::backend::GnssFactorRefinedEndpoint;

using Mat = Eigen::MatrixXd;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;

gtsam::Key keyX(int i) {
  return gtsam::Symbol('x', static_cast<std::uint64_t>(i));
}
gtsam::Key keyG() {
  return gtsam::Symbol('g', 0);
}
gtsam::Key keyE() {
  return gtsam::Symbol('e', 7);
}

Pose3 makePose(double rx, double ry, double rz, double x, double y, double z) {
  return Pose3(Rot3::RzRyRx(rx, ry, rz), Point3(x, y, z));
}

Mat numericJacobian(const std::function<Vec3(const Pose3&)>& residual, const Pose3& at) {
  constexpr double kEps = 1e-6;
  Mat jac(3, 6);
  for (int i = 0; i < 6; ++i) {
    Vec6 d = Vec6::Zero();
    d(i) = kEps;
    const Vec3 r_plus = residual(at.retract(d));
    const Vec3 r_minus = residual(at.retract(-d));
    jac.col(i) = (r_plus - r_minus) / (2.0 * kEps);
  }
  return jac;
}

Pose3 randXi() {
  return makePose(0.10, -0.20, 0.30, 1.0, -2.0, 0.5);
}
Pose3 randXj() {
  return makePose(0.22, 0.05, -0.18, 1.7, -1.4, 0.9);
}
Pose3 randG() {
  return makePose(-0.07, 0.31, 0.12, -3.0, 4.0, -1.0);
}
// A non-identity rotation makes E.translation()'s Jacobian non-trivial; the lever is E.t.
Pose3 randE() {
  return makePose(0.20, -0.15, 0.40, 0.13, -0.27, 0.41);
}

Vec3 fullResidual(const Pose3& Xi, const Pose3& Xj, const Pose3& G, const Pose3& E, double beta,
                  const Point3& meas) {
  const Pose3 Xb = gtsam::interpolate(Xi, Xj, beta);
  return G.transformTo(Xb.transformFrom(E.translation())) - meas;
}

Vec3 endpointResidual(const Pose3& X, const Pose3& G, const Pose3& E, const Point3& meas) {
  return G.transformTo(X.transformFrom(E.translation())) - meas;
}

}  // namespace

TEST(GnssFactorRefined, AnalyticJacobiansMatchNumericForInteriorBeta) {
  const Pose3 Xi = randXi();
  const Pose3 Xj = randXj();
  const Pose3 G = randG();
  const Pose3 E = randE();
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(3, 1.0);

  for (double beta : {0.25, 0.5, 0.75}) {
    const Point3 meas(0.3, -0.1, 0.2);
    const GnssFactorRefined factor(keyX(0), keyX(1), keyG(), keyE(), beta, meas, noise);

    Mat H1, H2, H3, H4;
    const Vec3 r = factor.evaluateError(Xi, Xj, G, E, H1, H2, H3, H4);
    EXPECT_TRUE(r.allFinite());

    const Mat H1n =
        numericJacobian([&](const Pose3& p) { return fullResidual(p, Xj, G, E, beta, meas); }, Xi);
    const Mat H2n =
        numericJacobian([&](const Pose3& p) { return fullResidual(Xi, p, G, E, beta, meas); }, Xj);
    const Mat H3n =
        numericJacobian([&](const Pose3& p) { return fullResidual(Xi, Xj, p, E, beta, meas); }, G);
    const Mat H4n =
        numericJacobian([&](const Pose3& p) { return fullResidual(Xi, Xj, G, p, beta, meas); }, E);

    EXPECT_LT((H1 - H1n).cwiseAbs().maxCoeff(), 1e-5) << "H1 at beta=" << beta;
    EXPECT_LT((H2 - H2n).cwiseAbs().maxCoeff(), 1e-5) << "H2 at beta=" << beta;
    EXPECT_LT((H3 - H3n).cwiseAbs().maxCoeff(), 1e-5) << "H3 at beta=" << beta;
    EXPECT_LT((H4 - H4n).cwiseAbs().maxCoeff(), 1e-5) << "H4 (extrinsic) at beta=" << beta;
  }
}

TEST(GnssFactorRefined, EndpointAnalyticJacobiansMatchNumeric) {
  const Pose3 X = randXi();
  const Pose3 G = randG();
  const Pose3 E = randE();
  const Point3 meas(-0.4, 0.2, 0.05);
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(3, 1.0);
  const GnssFactorRefinedEndpoint factor(keyX(0), keyG(), keyE(), meas, noise);

  Mat H1, H2, H3;
  const Vec3 r = factor.evaluateError(X, G, E, H1, H2, H3);
  EXPECT_TRUE(r.allFinite());

  const Mat H1n =
      numericJacobian([&](const Pose3& p) { return endpointResidual(p, G, E, meas); }, X);
  const Mat H2n =
      numericJacobian([&](const Pose3& p) { return endpointResidual(X, p, E, meas); }, G);
  const Mat H3n =
      numericJacobian([&](const Pose3& p) { return endpointResidual(X, G, p, meas); }, E);

  EXPECT_LT((H1 - H1n).cwiseAbs().maxCoeff(), 1e-5);
  EXPECT_LT((H2 - H2n).cwiseAbs().maxCoeff(), 1e-5);
  EXPECT_LT((H3 - H3n).cwiseAbs().maxCoeff(), 1e-5) << "endpoint extrinsic Jacobian";
}

// With E.translation() equal to the constant lever, the refined factor must reproduce the
// validated constant-lever GnssFactor residual exactly.
TEST(GnssFactorRefined, MatchesConstantLeverFactorWhenLeverEqualsTranslation) {
  const Pose3 Xi = randXi();
  const Pose3 Xj = randXj();
  const Pose3 G = randG();
  const Pose3 E = randE();
  const Point3 meas(0.11, -0.22, 0.33);
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(3, 1.0);

  for (double beta : {0.0, 0.4, 1.0}) {
    const GnssFactor base(keyX(0), keyX(1), keyG(), beta, E.translation(), meas, noise);
    const GnssFactorRefined refined(keyX(0), keyX(1), keyG(), keyE(), beta, meas, noise);
    const Vec3 r_base = base.evaluateError(Xi, Xj, G);
    const Vec3 r_ref = refined.evaluateError(Xi, Xj, G, E);
    EXPECT_LT((r_base - r_ref).norm(), 1e-12) << "beta=" << beta;
  }
}
