#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "chain_covariance.hpp"

using meridian::Pose;
using meridian::backend::ChainCovariance;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

namespace {

Eigen::Matrix3d skewOf(const Eigen::Vector3d& v) {
  Eigen::Matrix3d s;
  s << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return s;
}

// Independent hand implementation of the SE(3) adjoint in the translation-first
// [rho; phi] tangent: Ad_T = [[R, skew(t)*R], [0, R]].
Mat6 adjointOf(const Pose& T) {
  const Eigen::Matrix3d r = T.R();
  Mat6 ad = Mat6::Zero();
  ad.topLeftCorner<3, 3>() = r;
  ad.topRightCorner<3, 3>() = skewOf(T.t) * r;
  ad.bottomRightCorner<3, 3>() = r;
  return ad;
}

const Pose kT1{Eigen::Quaterniond::Identity(), Eigen::Vector3d(1.0, 0.0, 0.0)};
const Pose kT2{Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ())),
               Eigen::Vector3d(0.0, 1.0, 0.0)};

Mat6 cov1() {
  return (Vec6() << 0.01, 0.02, 0.03, 0.001, 0.002, 0.003).finished().asDiagonal();
}
Mat6 cov2() {
  return (Vec6() << 0.04, 0.05, 0.06, 0.004, 0.005, 0.006).finished().asDiagonal();
}

// Anchor at 0, then 0 -> 1 -> 2 with the two edges above.
ChainCovariance makeChain() {
  ChainCovariance chain;
  chain.start(0);
  chain.extend(0, 1, kT1, cov1());
  chain.extend(1, 2, kT2, cov2());
  return chain;
}

}  // namespace

TEST(ChainCovariance, KnownTracksInsertedIds) {
  const ChainCovariance chain = makeChain();
  EXPECT_TRUE(chain.known(0));
  EXPECT_TRUE(chain.known(1));
  EXPECT_TRUE(chain.known(2));
  EXPECT_FALSE(chain.known(3));
}

// The anchor carries zero covariance, so between(0, 2) is the full hand-composed chain:
// cov_chain(2) = Ad(T2^{-1}) * cov1 * Ad(T2^{-1})^T + cov2.
TEST(ChainCovariance, TwoEdgeComposeMatchesHandComputation) {
  const ChainCovariance chain = makeChain();
  const Mat6 ad = adjointOf(kT2.inverse());
  const Mat6 expected = ad * cov1() * ad.transpose() + cov2();

  const auto s02 = chain.between(0, 2);
  ASSERT_TRUE(s02.has_value());
  EXPECT_LT((*s02 - expected).norm(), 1e-12);

  const auto s01 = chain.between(0, 1);
  ASSERT_TRUE(s01.has_value());
  EXPECT_LT((*s01 - cov1()).norm(), 1e-12);
}

// between(1, 2) subtracts the transported chain covariance of 1, leaving exactly the noise
// of the single 1 -> 2 edge.
TEST(ChainCovariance, BetweenRecoversSingleEdgeCovariance) {
  const ChainCovariance chain = makeChain();
  const auto s12 = chain.between(1, 2);
  ASSERT_TRUE(s12.has_value());
  EXPECT_LT((*s12 - cov2()).norm(), 1e-12);
}

TEST(ChainCovariance, UnknownIdOrWrongOrderIsNullopt) {
  const ChainCovariance chain = makeChain();
  EXPECT_FALSE(chain.between(0, 99).has_value());    // unknown b
  EXPECT_FALSE(chain.between(99, 100).has_value());  // both unknown
  EXPECT_FALSE(chain.between(2, 0).has_value());     // a not older than b
  EXPECT_FALSE(chain.between(1, 1).has_value());     // a == b
}

TEST(ChainCovariance, ExtendFromUnknownIdIsNoOp) {
  ChainCovariance chain = makeChain();
  chain.extend(10, 11, kT1, cov1());
  EXPECT_FALSE(chain.known(11));
  EXPECT_FALSE(chain.between(0, 11).has_value());
}

TEST(ChainCovariance, StartResetsTheChain) {
  ChainCovariance chain = makeChain();
  chain.start(5);
  EXPECT_TRUE(chain.known(5));
  EXPECT_FALSE(chain.known(0));
  EXPECT_FALSE(chain.known(2));
}

TEST(ChainCovariance, BetweenIsSymmetricPsd) {
  ChainCovariance chain = makeChain();
  // A zero-covariance edge makes the subtraction in between() cancel to ~0, exercising the
  // PSD clamp's boundary.
  chain.extend(2, 3, kT1, Mat6::Zero());

  const std::vector<std::pair<std::uint64_t, std::uint64_t>> pairs = {{0, 2}, {1, 2}, {2, 3}};
  for (const auto& [a, b] : pairs) {
    const auto s = chain.between(a, b);
    ASSERT_TRUE(s.has_value()) << a << "->" << b;
    EXPECT_LT((*s - s->transpose()).norm(), 1e-15) << a << "->" << b;
    const Eigen::SelfAdjointEigenSolver<Mat6> eig(*s);
    EXPECT_GE(eig.eigenvalues().minCoeff(), -1e-12) << a << "->" << b;
  }
}
