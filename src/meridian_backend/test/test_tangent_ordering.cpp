#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

#include <Eigen/Core>
#include <array>

#include "gtsam_adapter.hpp"
#include "meridian/common/gaussian.hpp"

// Regression for the tangent-ordering polarity at the GTSAM boundary: a covariance that is
// tight (1e-6) on exactly one physical axis and loose (1.0) elsewhere must produce a between
// factor that is stiff along exactly that physical axis, whether the block arrives in the
// Meridian translation-first order (and is reordered) or already rotation-first (used as-is).

namespace {

using gtsam::Pose3;
using meridian::PoseCov6;
using meridian::backend::reorder_meridian_to_gtsam;

using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

constexpr double kTightVar = 1e-6;
constexpr double kLooseVar = 1.0;
constexpr double kDelta = 1e-3;  // perturbation step [m] or [rad]

const gtsam::Key kKeyI = gtsam::Symbol('x', 0);
const gtsam::Key kKeyJ = gtsam::Symbol('x', 1);

Pose3 poseI() {
  return Pose3(gtsam::Rot3::RzRyRx(0.2, -0.1, 0.3), gtsam::Point3(1.0, -2.0, 0.5));
}

Pose3 measured() {
  return Pose3(gtsam::Rot3::RzRyRx(-0.05, 0.15, 0.1), gtsam::Point3(0.4, 0.1, -0.3));
}

// Whitened error norm with T_j pushed off the zero-error solution along one GTSAM tangent
// axis [rx,ry,rz,tx,ty,tz]. At zero error the unwhitened residual equals the perturbation
// exactly, so the norm isolates the noise model's stiffness on that axis.
double whitenedNormAlongAxis(const gtsam::BetweenFactor<Pose3>& factor, int axis) {
  Vec6 xi = Vec6::Zero();
  xi(axis) = kDelta;
  gtsam::Values values;
  values.insert(kKeyI, poseI());
  values.insert(kKeyJ, poseI() * measured() * Pose3::Expmap(xi));
  return factor.whitenedError(values).norm();
}

void expectTightAxis(const Mat6& cov_rotation_first, int tight_axis) {
  const auto noise = gtsam::noiseModel::Gaussian::Covariance(cov_rotation_first);
  const gtsam::BetweenFactor<Pose3> factor(kKeyI, kKeyJ, measured(), noise);

  std::array<double, 6> norms{};
  for (int k = 0; k < 6; ++k) {
    norms[k] = whitenedNormAlongAxis(factor, k);
  }
  for (int k = 0; k < 6; ++k) {
    if (k == tight_axis) {
      continue;
    }
    EXPECT_GT(norms[tight_axis], 100.0 * norms[k])
        << "tight axis " << tight_axis << " vs loose axis " << k;
  }
}

}  // namespace

TEST(TangentOrdering, TranslationFirstCovReorderedLandsOnPhysicalAxis) {
  for (int m = 0; m < 6; ++m) {
    PoseCov6 cov;
    cov.M = kLooseVar * Mat6::Identity();
    cov.M(m, m) = kTightVar;
    // Meridian axis m ([tx,ty,tz,rx,ry,rz]) is GTSAM axis (m+3)%6 ([rx,ry,rz,tx,ty,tz]).
    expectTightAxis(reorder_meridian_to_gtsam(cov.M), (m + 3) % 6);
  }
}

TEST(TangentOrdering, RotationFirstBoundaryCovUsedAsIs) {
  for (int g = 0; g < 6; ++g) {
    Mat6 cov = kLooseVar * Mat6::Identity();
    cov(g, g) = kTightVar;
    expectTightAxis(cov, g);
  }
}

TEST(TangentOrdering, BothPolaritiesAgreeOnTheSamePhysicalAxis) {
  // Tight rotation-z: Meridian index 5, rotation-first index 2.
  PoseCov6 meridian_cov;
  meridian_cov.M = kLooseVar * Mat6::Identity();
  meridian_cov.M(5, 5) = kTightVar;

  Mat6 boundary_cov = kLooseVar * Mat6::Identity();
  boundary_cov(2, 2) = kTightVar;

  EXPECT_LT((reorder_meridian_to_gtsam(meridian_cov.M) - boundary_cov).norm(), 1e-18);
  expectTightAxis(reorder_meridian_to_gtsam(meridian_cov.M), 2);
  expectTightAxis(boundary_cov, 2);
}
