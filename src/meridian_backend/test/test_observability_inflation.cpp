#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <cstddef>

#include "observability_inflation.hpp"

using meridian::GaussianBlock;
using meridian::ObservabilityReport;
using meridian::backend::inflate_by_observability;
using meridian::backend::InflationResult;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

namespace {

GaussianBlock<6> covOf(const Mat6& m) {
  GaussianBlock<6> g;
  g.M = m;
  return g;
}

// A dense, exactly symmetric SPD matrix with distinct entries.
Mat6 densePsd() {
  Mat6 a;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      a(i, j) = 0.1 * static_cast<double>(i + 2 * j) + (i == j ? 1.0 : 0.0);
    }
  }
  Mat6 s = a * a.transpose();
  s = 0.5 * (s + s.transpose()).eval();
  return s;
}

}  // namespace

TEST(ObservabilityInflation, FullyObservableIsIdentityMap) {
  const Mat6 sig = densePsd();
  ObservabilityReport obs;  // all scores 1
  const InflationResult res =
      inflate_by_observability(covOf(sig), obs, /*rho_max=*/100.0, /*gamma=*/2.0,
                               /*degenerate_thresh=*/0.05, /*degenerate_lock=*/true);
  EXPECT_FALSE(res.any_locked);
  for (int k = 0; k < 6; ++k) {
    EXPECT_DOUBLE_EQ(res.lambda[static_cast<std::size_t>(k)], 1.0) << "axis " << k;
  }
  EXPECT_LT((res.cov - sig).norm(), 1e-12);
}

// Score index 1 is ty (translation-first); on the rotation-first output that is axis 4.
TEST(ObservabilityInflation, DegenerateTyInflatesRotationFirstAxis4) {
  const double rho_max = 50.0;
  ObservabilityReport obs;
  obs.score[1] = 0.0;
  const InflationResult res = inflate_by_observability(covOf(Mat6::Identity()), obs, rho_max,
                                                       /*gamma=*/2.0, /*degenerate_thresh=*/0.0,
                                                       /*degenerate_lock=*/false);
  EXPECT_FALSE(res.any_locked);
  Vec6 expected_diag = Vec6::Ones();
  expected_diag(4) = rho_max;
  for (int k = 0; k < 6; ++k) {
    EXPECT_DOUBLE_EQ(res.lambda[static_cast<std::size_t>(k)], expected_diag(k)) << "axis " << k;
  }
  const Mat6 expected = expected_diag.asDiagonal();
  EXPECT_LT((res.cov - expected).norm(), 1e-12);
}

TEST(ObservabilityInflation, GammaLawMidpoint) {
  // s = 0.5, gamma = 2, rho_max = 100  ->  lambda = 1 + 99 * 0.25 = 25.75.
  ObservabilityReport obs;
  obs.score[0] = 0.5;  // tx -> rotation-first axis 3
  const InflationResult res =
      inflate_by_observability(covOf(Mat6::Identity()), obs, /*rho_max=*/100.0, /*gamma=*/2.0,
                               /*degenerate_thresh=*/0.0, /*degenerate_lock=*/false);
  EXPECT_DOUBLE_EQ(res.lambda[3], 25.75);
  EXPECT_NEAR(res.cov(3, 3), 25.75, 1e-12);
  EXPECT_DOUBLE_EQ(res.lambda[0], 1.0);
}

TEST(ObservabilityInflation, ScoresAreClamped) {
  ObservabilityReport obs;
  obs.score[0] = -0.5;  // clamps to 0 -> full inflation
  obs.score[1] = 1.5;   // clamps to 1 -> no inflation
  const InflationResult res =
      inflate_by_observability(covOf(Mat6::Identity()), obs, /*rho_max=*/10.0, /*gamma=*/2.0,
                               /*degenerate_thresh=*/0.0, /*degenerate_lock=*/false);
  EXPECT_DOUBLE_EQ(res.lambda[3], 10.0);
  EXPECT_DOUBLE_EQ(res.lambda[4], 1.0);
}

// Eigvec basis: the degenerate direction is the translation xy-plane rotated by 30 degrees.
// The inflation must land along that rotated direction, not along a frame axis.
TEST(ObservabilityInflation, EigvecsRotateTheInflatedDirection) {
  const double th = M_PI / 6.0;
  Mat6 eigvecs = Mat6::Identity();  // translation-first columns
  eigvecs(0, 0) = std::cos(th);
  eigvecs(1, 0) = std::sin(th);
  eigvecs(0, 1) = -std::sin(th);
  eigvecs(1, 1) = std::cos(th);

  ObservabilityReport obs;
  obs.eigvecs = eigvecs;
  obs.score = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0};  // degenerate along column 0 only

  const double rho_max = 100.0;
  const InflationResult res =
      inflate_by_observability(covOf(Mat6::Identity()), obs, rho_max, /*gamma=*/2.0,
                               /*degenerate_thresh=*/0.0, /*degenerate_lock=*/false);

  // The rotated directions expressed on the rotation-first axes (tx,ty at indices 3,4).
  Vec6 d_degen = Vec6::Zero();
  d_degen(3) = std::cos(th);
  d_degen(4) = std::sin(th);
  Vec6 d_perp = Vec6::Zero();
  d_perp(3) = -std::sin(th);
  d_perp(4) = std::cos(th);

  EXPECT_NEAR(d_degen.dot(res.cov * d_degen), rho_max, 1e-9);
  EXPECT_NEAR(d_perp.dot(res.cov * d_perp), 1.0, 1e-9);
  // Axes outside the rotated plane stay at unit variance.
  EXPECT_NEAR(res.cov(5, 5), 1.0, 1e-12);  // tz
  EXPECT_NEAR(res.cov(0, 0), 1.0, 1e-12);  // rx
}

TEST(ObservabilityInflation, DegenerateLockOverridesSmoothLaw) {
  const double rho_max = 100.0;
  const double gamma = 2.0;
  ObservabilityReport obs;
  obs.score[2] = 0.04;  // tz -> rotation-first axis 5, below the 0.05 threshold

  const InflationResult locked =
      inflate_by_observability(covOf(Mat6::Identity()), obs, rho_max, gamma,
                               /*degenerate_thresh=*/0.05, /*degenerate_lock=*/true);
  EXPECT_TRUE(locked.any_locked);
  EXPECT_DOUBLE_EQ(locked.lambda[5], rho_max);

  const InflationResult smooth =
      inflate_by_observability(covOf(Mat6::Identity()), obs, rho_max, gamma,
                               /*degenerate_thresh=*/0.05, /*degenerate_lock=*/false);
  EXPECT_FALSE(smooth.any_locked);
  EXPECT_NEAR(smooth.lambda[5], 1.0 + 99.0 * std::pow(0.96, 2.0), 1e-9);
  EXPECT_LT(smooth.lambda[5], rho_max);
}
