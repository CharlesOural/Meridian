#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>

#include <Eigen/Core>
#include <boost/optional.hpp>
#include <cmath>
#include <functional>

#include "gnss_factor.hpp"

namespace {

using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using meridian::backend::GnssFactor;
using meridian::backend::GnssFactorEndpoint;

using Mat = Eigen::MatrixXd;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;

gtsam::Key keyX(int i) {
  return gtsam::Symbol('x', static_cast<std::uint64_t>(i));
}
gtsam::Key keyG() {
  return gtsam::Symbol('g', 0);
}

Pose3 makePose(double rx, double ry, double rz, double x, double y, double z) {
  return Pose3(Rot3::RzRyRx(rx, ry, rz), Point3(x, y, z));
}

// Central-difference of a 3-vector residual w.r.t. a single Pose3 argument, perturbed on its
// own 6-tangent via retract. Returns the 3x6 numerical Jacobian.
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

const Point3 kLever(0.13, -0.27, 0.41);

Pose3 randXi() {
  return makePose(0.10, -0.20, 0.30, 1.0, -2.0, 0.5);
}
Pose3 randXj() {
  return makePose(0.22, 0.05, -0.18, 1.7, -1.4, 0.9);
}
Pose3 randG() {
  return makePose(-0.07, 0.31, 0.12, -3.0, 4.0, -1.0);
}

// The residual the interpolating factor evaluates, for a chosen measurement.
Vec3 fullResidual(const Pose3& Xi, const Pose3& Xj, const Pose3& G, double beta,
                  const Point3& meas) {
  const Pose3 Xb = gtsam::interpolate(Xi, Xj, beta);
  return G.transformTo(Xb.transformFrom(kLever)) - meas;
}

Vec3 endpointResidual(const Pose3& X, const Pose3& G, const Point3& meas) {
  return G.transformTo(X.transformFrom(kLever)) - meas;
}

}  // namespace

TEST(GnssFactor, AnalyticJacobiansMatchNumericForInteriorBeta) {
  const Pose3 Xi = randXi();
  const Pose3 Xj = randXj();
  const Pose3 G = randG();
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(3, 1.0);

  for (double beta : {0.25, 0.5, 0.75}) {
    // Offset the measurement so the residual is non-zero (Jacobians are residual-independent,
    // but a non-trivial linearization point is the realistic case).
    const Point3 meas(0.3, -0.1, 0.2);
    const GnssFactor factor(keyX(0), keyX(1), keyG(), beta, kLever, meas, noise);

    Mat H1, H2, H3;
    const Vec3 r = factor.evaluateError(Xi, Xj, G, H1, H2, H3);
    EXPECT_TRUE(r.allFinite());

    const Mat H1_num =
        numericJacobian([&](const Pose3& p) { return fullResidual(p, Xj, G, beta, meas); }, Xi);
    const Mat H2_num =
        numericJacobian([&](const Pose3& p) { return fullResidual(Xi, p, G, beta, meas); }, Xj);
    const Mat H3_num =
        numericJacobian([&](const Pose3& p) { return fullResidual(Xi, Xj, p, beta, meas); }, G);

    EXPECT_LT((H1 - H1_num).cwiseAbs().maxCoeff(), 1e-5) << "H1 mismatch at beta=" << beta;
    EXPECT_LT((H2 - H2_num).cwiseAbs().maxCoeff(), 1e-5) << "H2 mismatch at beta=" << beta;
    EXPECT_LT((H3 - H3_num).cwiseAbs().maxCoeff(), 1e-5) << "H3 mismatch at beta=" << beta;
  }
}

TEST(GnssFactor, EndpointAnalyticJacobiansMatchNumeric) {
  const Pose3 X = randXi();
  const Pose3 G = randG();
  const Point3 meas(-0.4, 0.2, 0.05);
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(3, 1.0);
  const GnssFactorEndpoint factor(keyX(0), keyG(), kLever, meas, noise);

  Mat H1, H2;
  const Vec3 r = factor.evaluateError(X, G, H1, H2);
  EXPECT_TRUE(r.allFinite());

  const Mat H1_num =
      numericJacobian([&](const Pose3& p) { return endpointResidual(p, G, meas); }, X);
  const Mat H2_num =
      numericJacobian([&](const Pose3& p) { return endpointResidual(X, p, meas); }, G);

  EXPECT_LT((H1 - H1_num).cwiseAbs().maxCoeff(), 1e-5);
  EXPECT_LT((H2 - H2_num).cwiseAbs().maxCoeff(), 1e-5);
}

TEST(GnssFactor, ZeroResidualWhenMeasurementMatchesPrediction) {
  const Pose3 Xi = randXi();
  const Pose3 Xj = randXj();
  const Pose3 G = randG();
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(3, 1.0);

  for (double beta : {0.25, 0.5, 0.75}) {
    // Construct meas as the exact predicted ENU antenna position, so the residual must vanish.
    const Point3 meas = fullResidual(Xi, Xj, G, beta, Point3(0, 0, 0));
    const GnssFactor factor(keyX(0), keyX(1), keyG(), beta, kLever, meas, noise);
    const Vec3 r = factor.evaluateError(Xi, Xj, G);
    EXPECT_LT(r.norm(), 1e-12) << "non-zero residual at beta=" << beta;
  }

  const Point3 meas_ep = endpointResidual(Xi, G, Point3(0, 0, 0));
  const GnssFactorEndpoint endpoint(keyX(0), keyG(), kLever, meas_ep, noise);
  const Vec3 r_ep = endpoint.evaluateError(Xi, G);
  EXPECT_LT(r_ep.norm(), 1e-12);
}

TEST(GnssFactor, EndpointMatchesInterpolatingFactorAtBetaZero) {
  // interpolate(Xi, Xj, 0) == Xi, so the interpolating factor at beta=0 must reproduce the
  // endpoint factor's residual built on Xi alone.
  const Pose3 Xi = randXi();
  const Pose3 Xj = randXj();
  const Pose3 G = randG();
  const Point3 meas(0.11, -0.22, 0.33);
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(3, 1.0);

  const GnssFactor full(keyX(0), keyX(1), keyG(), 0.0, kLever, meas, noise);
  const GnssFactorEndpoint endpoint(keyX(0), keyG(), kLever, meas, noise);

  const Vec3 r_full = full.evaluateError(Xi, Xj, G);
  const Vec3 r_ep = endpoint.evaluateError(Xi, G);
  EXPECT_LT((r_full - r_ep).norm(), 1e-9);
}
