#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <algorithm>
#include <random>

#include "gtsam_adapter.hpp"
#include "meridian/common/gaussian.hpp"
#include "meridian/common/pose.hpp"

using meridian::Pose;
using meridian::PoseCov6;
using meridian::backend::ensure_psd;
using meridian::backend::from_gtsam;
using meridian::backend::P_meridian_gtsam;
using meridian::backend::reorder_gtsam_to_meridian;
using meridian::backend::reorder_meridian_to_gtsam;
using meridian::backend::to_gtsam;

namespace {

using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

Pose randomPose(std::mt19937& rng) {
  std::normal_distribution<double> n(0.0, 1.0);
  const Eigen::Quaterniond q = Eigen::Quaterniond(n(rng), n(rng), n(rng), n(rng)).normalized();
  return Pose(q, Eigen::Vector3d(n(rng), n(rng), n(rng)));
}

Mat6 randomMatrix(std::mt19937& rng) {
  std::normal_distribution<double> n(0.0, 1.0);
  Mat6 m;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      m(i, j) = n(rng);
    }
  }
  return m;
}

// Coefficient distance up to the q ~ -q double cover.
double quatDistance(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b) {
  return std::min((a.coeffs() - b.coeffs()).norm(), (a.coeffs() + b.coeffs()).norm());
}

}  // namespace

TEST(GtsamAdapter, PoseRoundTripIsExact) {
  std::mt19937 rng(42);
  for (int i = 0; i < 100; ++i) {
    const Pose original = randomPose(rng);
    const Pose back = from_gtsam(to_gtsam(original));
    EXPECT_LT(quatDistance(back.q, original.q), 1e-15);
    EXPECT_LT((back.t - original.t).norm(), 1e-15);
  }
}

TEST(GtsamAdapter, PermutationIsItsOwnInverse) {
  const Mat6& P = P_meridian_gtsam();
  EXPECT_EQ((P * P.transpose() - Mat6::Identity()).norm(), 0.0);
  EXPECT_EQ((P - P.transpose()).norm(), 0.0);
}

TEST(GtsamAdapter, ReorderRoundTripRestoresInput) {
  std::mt19937 rng(7);
  const Mat6 m = randomMatrix(rng);
  const Mat6 s = 0.5 * (m + m.transpose());
  // A permutation conjugation only moves entries, so the round trip is bit-exact.
  EXPECT_EQ((reorder_gtsam_to_meridian(reorder_meridian_to_gtsam(s)) - s).norm(), 0.0);
  EXPECT_EQ((reorder_meridian_to_gtsam(reorder_gtsam_to_meridian(s)) - s).norm(), 0.0);
}

TEST(GtsamAdapter, MeridianDiagonalLandsOnGtsamAxes) {
  PoseCov6 cov;
  cov.M = Mat6::Zero();
  cov.M.diagonal() << 1, 2, 3, 4, 5, 6;  // Meridian order [tx,ty,tz,rx,ry,rz]

  const Mat6 g = reorder_meridian_to_gtsam(cov.M);
  Vec6 expected;
  expected << 4, 5, 6, 1, 2, 3;  // GTSAM order [rx,ry,rz,tx,ty,tz]
  EXPECT_EQ((g - Mat6(expected.asDiagonal())).norm(), 0.0);

  const Mat6 back = reorder_gtsam_to_meridian(Mat6(expected.asDiagonal()));
  EXPECT_EQ((back - cov.M).norm(), 0.0);
}

TEST(GtsamAdapter, EnsurePsdClampsIndefiniteMatrix) {
  std::mt19937 rng(3);
  const Mat6 q = Eigen::HouseholderQR<Mat6>(randomMatrix(rng)).householderQ();
  Vec6 evals;
  evals << -1, 1, 2, 3, 4, 5;
  Mat6 s = q * evals.asDiagonal() * q.transpose();

  EXPECT_TRUE(ensure_psd(s));

  const Eigen::SelfAdjointEigenSolver<Mat6> eig(s);
  EXPECT_GE(eig.eigenvalues().minCoeff(), 0.0);
  // The positive part of the spectrum is preserved.
  for (int k = 1; k < 6; ++k) {
    EXPECT_NEAR(eig.eigenvalues()(k), static_cast<double>(k), 1e-9);
  }
  EXPECT_EQ((s - s.transpose()).norm(), 0.0);
}

TEST(GtsamAdapter, EnsurePsdLeavesPsdMatrixUntouched) {
  std::mt19937 rng(11);
  const Mat6 m = randomMatrix(rng);
  // m*m^T is exactly symmetric; +I keeps every eigenvalue well above the clamp floor.
  Mat6 s = m * m.transpose() + Mat6::Identity();
  const Mat6 original = s;

  EXPECT_FALSE(ensure_psd(s));
  EXPECT_EQ((s - original).norm(), 0.0);
}
