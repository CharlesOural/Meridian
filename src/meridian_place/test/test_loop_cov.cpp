#include <gtest/gtest.h>

#include <optional>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "meridian/common/observability.hpp"
#include "meridian/config/config.hpp"
#include "loop_cov.hpp"

using meridian::ObservabilityReport;
using meridian::PlaceConfig;
using meridian::PoseCov6;
using meridian::shapeLoopCov;
using M6 = Eigen::Matrix<double, 6, 6>;

namespace {
PlaceConfig cfg() {
  PlaceConfig c;  // defaults: cov_lambda 1e-3, degenerate_eig 1.0, mult 100, fitness_min 0.6
  return c;
}
const std::optional<ObservabilityReport> kNoObs;
}  // namespace

TEST(LoopCov, LowerFitnessGivesLooserCovariance) {
  const M6 H = 10.0 * M6::Identity();
  const PoseCov6 tight = shapeLoopCov(H, 0.9, cfg(), kNoObs, kNoObs);
  const PoseCov6 loose = shapeLoopCov(H, 0.6, cfg(), kNoObs, kNoObs);
  EXPECT_GT(loose.M.trace(), tight.M.trace());
}

TEST(LoopCov, PermutesRotationFirstToTranslationFirst) {
  // H rotation-first: rotation info 10 (block 0:2), translation info 5 (block 3:5).
  M6 H = M6::Zero();
  H.diagonal() << 10, 10, 10, 5, 5, 5;
  const PoseCov6 c = shapeLoopCov(H, 0.6, cfg(), kNoObs, kNoObs);  // fitness=fitness_min => s=1
  // Translation-first output: first block is translation (drawn from info 5), last is rotation
  // (info 10). So the translation variance is the LARGER one.
  EXPECT_NEAR(c.M(0, 0), 1.0 / (5.0 + 1e-3), 1e-4);   // tx variance from translation info
  EXPECT_NEAR(c.M(3, 3), 1.0 / (10.0 + 1e-3), 1e-4);  // rx variance from rotation info
  EXPECT_GT(c.M(0, 0), c.M(3, 3));
}

TEST(LoopCov, DegenerateEigenDirectionInflated) {
  // Rotation-first index 5 == translation z; give it tiny info (< degenerate_eig).
  M6 H = M6::Zero();
  H.diagonal() << 10, 10, 10, 10, 10, 0.5;
  const PoseCov6 c = shapeLoopCov(H, 0.6, cfg(), kNoObs, kNoObs);
  // tz maps to translation-first index 2; it must be inflated ~100x relative to a normal axis.
  EXPECT_GT(c.M(2, 2), 50.0);
  EXPECT_LT(c.M(0, 0), 1.0);
}

TEST(LoopCov, ObservabilityLoosensNamedAxis) {
  const M6 H = 10.0 * M6::Identity();
  const PoseCov6 base = shapeLoopCov(H, 0.6, cfg(), kNoObs, kNoObs);

  ObservabilityReport obs;             // score order [tx,ty,tz,rx,ry,rz]
  obs.score = {1, 1, 0.1, 1, 1, 1};    // tz under-observed
  const PoseCov6 loosened = shapeLoopCov(H, 0.6, cfg(), kNoObs, obs);

  EXPECT_NEAR(loosened.M(2, 2), base.M(2, 2) * 100.0, base.M(2, 2) * 1.0);  // ~1/0.1^2
  EXPECT_NEAR(loosened.M(0, 0), base.M(0, 0), 1e-9);  // tx untouched
}

TEST(LoopCov, ResultIsPositiveDefinite) {
  M6 H = M6::Zero();
  H.diagonal() << 8, 6, 4, 10, 2, 0.3;
  const PoseCov6 c = shapeLoopCov(H, 0.7, cfg(), kNoObs, kNoObs);
  Eigen::SelfAdjointEigenSolver<M6> es(c.M);
  EXPECT_GT(es.eigenvalues()(0), 0.0);
  EXPECT_TRUE(c.M.isApprox(c.M.transpose()));
}
