#include <cmath>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "meridian/common/pose.hpp"

using meridian::Pose;

namespace {
Pose sample_pose() {
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.3, Eigen::Vector3d(0.2, 0.5, 0.84).normalized()));
  return Pose{q, Eigen::Vector3d(1.0, -2.0, 0.5)};
}
}  // namespace

TEST(Pose, IdentityDefault) {
  const Pose p;
  EXPECT_TRUE(p.q.isApprox(Eigen::Quaterniond::Identity()));
  EXPECT_TRUE(p.t.isZero());
}

TEST(Pose, ComposeWithInverseIsIdentity) {
  const Pose a = sample_pose();
  const Pose i = a * a.inverse();
  EXPECT_TRUE(i.t.isZero(1e-9));
  EXPECT_NEAR(std::abs(i.q.w()), 1.0, 1e-9);
}

TEST(Pose, ApplyToPointMatchesRotationPlusTranslation) {
  const Pose a = sample_pose();
  const Eigen::Vector3d p(0.7, 0.1, -0.3);
  EXPECT_TRUE((a * p).isApprox(a.R() * p + a.t, 1e-12));
}

// Right-perturbation, translation-first tangent xi = [rho; phi]:
// (a boxplus xi) boxminus a == xi.
TEST(Pose, BoxplusBoxminusRoundTrip) {
  const Pose a = sample_pose();
  Eigen::Matrix<double, 6, 1> xi;
  xi << 0.05, -0.02, 0.10, 0.01, -0.03, 0.02;  // [tx,ty,tz, rx,ry,rz]
  const Pose b = a.boxplus(xi);
  EXPECT_TRUE(b.boxminus(a).isApprox(xi, 1e-9));
}

TEST(Pose, BoxplusZeroIsNoop) {
  const Pose a = sample_pose();
  const Pose b = a.boxplus(Eigen::Matrix<double, 6, 1>::Zero());
  EXPECT_TRUE(b.t.isApprox(a.t, 1e-12));
  EXPECT_TRUE(b.q.isApprox(a.q, 1e-12) ||
              b.q.coeffs().isApprox(-a.q.coeffs(), 1e-12));
}

// The first three tangent components are translation: perturbing only them moves t and
// leaves the rotation untouched.
TEST(Pose, TranslationFirstTangentBlock) {
  const Pose a = sample_pose();
  Eigen::Matrix<double, 6, 1> xi = Eigen::Matrix<double, 6, 1>::Zero();
  xi.head<3>() = Eigen::Vector3d(0.1, 0.2, 0.3);
  const Pose b = a.boxplus(xi);
  EXPECT_TRUE(b.q.isApprox(a.q, 1e-12));
  EXPECT_TRUE((b.t - a.t).isApprox(a.R() * xi.head<3>(), 1e-9));
}
