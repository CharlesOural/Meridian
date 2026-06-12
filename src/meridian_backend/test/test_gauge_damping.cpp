#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

#include <Eigen/Core>
#include <algorithm>
#include <boost/make_shared.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "gauge_damping_factor.hpp"

namespace {

using gtsam::Pose3;
using meridian::backend::GaugeDampingFactor;

using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

constexpr int kChainLength = 10;  // between factors; poses x0..x10
constexpr double kLambda = 100.0;

gtsam::Key key(int i) {
  return gtsam::Symbol('x', static_cast<std::uint64_t>(i));
}

double poseDiff(const Pose3& a, const Pose3& b) {
  return Pose3::Logmap(a.between(b)).norm();
}

Pose3 startPose() {
  return Pose3(gtsam::Rot3::RzRyRx(0.1, 0.2, -0.1), gtsam::Point3(2.0, 1.0, 0.0));
}

std::vector<Pose3> measurements() {
  std::vector<Pose3> m;
  m.reserve(kChainLength);
  for (int i = 0; i < kChainLength; ++i) {
    m.emplace_back(gtsam::Rot3::RzRyRx(0.03 * i - 0.1, 0.02, -0.01 * i),
                   gtsam::Point3(1.0, 0.1 * i - 0.3, 0.05));
  }
  return m;
}

std::vector<Pose3> composedChain(const Pose3& first) {
  std::vector<Pose3> poses{first};
  for (const Pose3& m : measurements()) {
    poses.push_back(poses.back() * m);
  }
  return poses;
}

// Deterministic small twists so the solve has real residuals to remove.
Vec6 perturbation(int i) {
  Vec6 xi;
  xi << 0.01 * std::sin(1.0 + i), -0.008 * std::cos(0.5 * i), 0.009 * std::sin(2.0 + i),
      0.01 * std::cos(1.0 + i), -0.01 * std::sin(0.7 * i + 0.3), 0.008 * std::cos(2.0 * i);
  return xi;
}

gtsam::Values initialValues(const Pose3& first) {
  const std::vector<Pose3> truth = composedChain(first);
  gtsam::Values values;
  for (int i = 0; i <= kChainLength; ++i) {
    values.insert(key(i), truth[i] * Pose3::Expmap(perturbation(i)));
  }
  return values;
}

// Chain of between factors with NO prior; the gauge damping factor on x0 is the only
// thing keeping the linear system full rank.
gtsam::NonlinearFactorGraph buildChainGraph() {
  gtsam::NonlinearFactorGraph graph;
  graph.add(boost::make_shared<GaugeDampingFactor>(key(0), kLambda));
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(6, 0.1);
  const std::vector<Pose3> meas = measurements();
  for (int i = 0; i < kChainLength; ++i) {
    graph.emplace_shared<gtsam::BetweenFactor<Pose3>>(key(i), key(i + 1), meas[i], noise);
  }
  return graph;
}

// Full relinearization every update and no wildfire cutoff, so repeated update() calls
// behave as exact Gauss-Newton iterations.
gtsam::ISAM2Params isamParams() {
  gtsam::ISAM2Params params;
  params.relinearizeSkip = 1;
  params.relinearizeThreshold = 0.0;
  params.optimizationParams = gtsam::ISAM2GaussNewtonParams(0.0);
  return params;
}

gtsam::Values solve(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& initial) {
  gtsam::ISAM2 isam(isamParams());
  isam.update(graph, initial);
  for (int k = 0; k < 8; ++k) {
    isam.update();
  }
  return isam.calculateEstimate();
}

}  // namespace

TEST(GaugeDamping, ErrorIsAlwaysZeroAndDimIsSix) {
  const GaugeDampingFactor factor(key(0), kLambda);
  EXPECT_EQ(factor.dim(), 6u);

  gtsam::Values values;
  values.insert(key(0), startPose());
  EXPECT_EQ(factor.error(values), 0.0);

  values.update(key(0), Pose3(gtsam::Rot3::RzRyRx(-1.2, 0.7, 2.9), gtsam::Point3(1e3, -1e4, 5e2)));
  EXPECT_EQ(factor.error(values), 0.0);
}

TEST(GaugeDamping, LinearizesToScaledIdentityInformation) {
  const GaugeDampingFactor factor(key(0), kLambda);
  gtsam::Values values;
  values.insert(key(0), startPose());

  const auto linear = factor.linearize(values);
  const auto jacobian = boost::dynamic_pointer_cast<gtsam::JacobianFactor>(linear);
  ASSERT_TRUE(jacobian != nullptr);
  ASSERT_EQ(jacobian->keys().size(), 1u);
  EXPECT_EQ(jacobian->keys()[0], key(0));
  EXPECT_LT((jacobian->information() - kLambda * Mat6::Identity()).norm(), 1e-12);
  EXPECT_EQ(jacobian->getb().norm(), 0.0);
}

TEST(GaugeDamping, CloneAndEquals) {
  const GaugeDampingFactor factor(key(0), kLambda);
  const auto copy = factor.clone();
  EXPECT_TRUE(factor.equals(*copy));

  const GaugeDampingFactor other_lambda(key(0), kLambda + 1.0);
  const GaugeDampingFactor other_key(key(1), kLambda);
  EXPECT_FALSE(factor.equals(other_lambda));
  EXPECT_FALSE(factor.equals(other_key));
}

TEST(GaugeDamping, ChainSolvesInIsam2WithoutPrior) {
  const gtsam::NonlinearFactorGraph graph = buildChainGraph();
  const gtsam::Values initial = initialValues(startPose());

  gtsam::ISAM2 isam(isamParams());
  EXPECT_NO_THROW({
    isam.update(graph, initial);
    for (int k = 0; k < 8; ++k) {
      isam.update();
    }
  });

  const gtsam::Values est = isam.calculateEstimate();
  const std::vector<Pose3> meas = measurements();
  for (int i = 0; i < kChainLength; ++i) {
    const Pose3 rel = est.at<Pose3>(key(i)).between(est.at<Pose3>(key(i + 1)));
    EXPECT_LT(poseDiff(rel, meas[i]), 1e-6);
  }

  // The whole solution is the measurement chain composed off the estimated first pose.
  const std::vector<Pose3> composed = composedChain(est.at<Pose3>(key(0)));
  for (int i = 0; i <= kChainLength; ++i) {
    EXPECT_LT(poseDiff(est.at<Pose3>(key(i)), composed[i]), 1e-5);
  }

  // Zero residual is reachable while holding x0, so the damping leaves x0 at its initial value.
  EXPECT_LT(poseDiff(est.at<Pose3>(key(0)), initial.at<Pose3>(key(0))), 1e-9);
}

TEST(GaugeDamping, RigidTranslationOfInitialValuesTranslatesSolution) {
  const gtsam::NonlinearFactorGraph graph = buildChainGraph();
  const gtsam::Values initial_a = initialValues(startPose());

  const Eigen::Vector3d d(10.0, 20.0, 30.0);
  gtsam::Values initial_b;
  for (int i = 0; i <= kChainLength; ++i) {
    const Pose3 p = initial_a.at<Pose3>(key(i));
    initial_b.insert(key(i), Pose3(p.rotation(), p.translation() + d));
  }

  const gtsam::Values est_a = solve(graph, initial_a);
  const gtsam::Values est_b = solve(graph, initial_b);

  for (int i = 0; i < kChainLength; ++i) {
    const Pose3 rel_a = est_a.at<Pose3>(key(i)).between(est_a.at<Pose3>(key(i + 1)));
    const Pose3 rel_b = est_b.at<Pose3>(key(i)).between(est_b.at<Pose3>(key(i + 1)));
    EXPECT_LT(poseDiff(rel_a, rel_b), 1e-9);
  }
  for (int i = 0; i <= kChainLength; ++i) {
    const Pose3 pa = est_a.at<Pose3>(key(i));
    const Pose3 pb = est_b.at<Pose3>(key(i));
    EXPECT_LT((pb.translation() - pa.translation() - d).norm(), 1e-6);
    EXPECT_LT(gtsam::Rot3::Logmap(pa.rotation().between(pb.rotation())).norm(), 1e-9);
  }
}

TEST(GaugeDamping, GaugeFloatsUnderLaterAbsoluteInformation) {
  gtsam::ISAM2 isam(isamParams());
  isam.update(buildChainGraph(), initialValues(startPose()));
  for (int k = 0; k < 5; ++k) {
    isam.update();
  }
  const gtsam::Values before = isam.calculateEstimate();
  const Pose3 first_before = before.at<Pose3>(key(0));
  const Pose3 last_before = before.at<Pose3>(key(kChainLength));

  // Tight absolute pose on the LAST keyframe at a shifted location.
  const Eigen::Vector3d shift(1.0, -2.0, 1.5);
  gtsam::NonlinearFactorGraph absolute;
  absolute.emplace_shared<gtsam::PriorFactor<Pose3>>(
      key(kChainLength), Pose3(last_before.rotation(), last_before.translation() + shift),
      gtsam::noiseModel::Isotropic::Sigma(6, 1e-3));
  isam.update(absolute);

  // Two claims, asserted separately because the damping throttles the rigid-translation
  // mode at roughly chain_info/(chain_info+lambda) per Gauss-Newton step:
  //
  // (1) Unbiasedness of the OBJECTIVE: the damping contributes zero gradient at delta = 0,
  //     so the optimum is the rigid translation of the whole chain onto the prior. A batch
  //     solve must reach it (a hard PriorFactor anchor would not).
  gtsam::NonlinearFactorGraph full = buildChainGraph();
  full.push_back(absolute.begin(), absolute.end());
  gtsam::GaussNewtonParams gn_params;
  gn_params.maxIterations = 5000;
  gn_params.relativeErrorTol = 1e-14;
  gn_params.absoluteErrorTol = 1e-14;
  const gtsam::Values batch =
      gtsam::GaussNewtonOptimizer(full, initialValues(startPose()), gn_params).optimize();
  const double batch_frac =
      (batch.at<Pose3>(key(0)).translation() - first_before.translation()).norm() / shift.norm();
  EXPECT_GT(batch_frac, 0.999);
  const std::vector<Pose3> meas = measurements();
  for (int i = 0; i < kChainLength; ++i) {
    const Pose3 rel = batch.at<Pose3>(key(i)).between(batch.at<Pose3>(key(i + 1)));
    EXPECT_LT(poseDiff(rel, meas[i]), 1e-5);
  }

  // (2) The incremental solver tracks the same optimum: the first pose walks toward the
  //     demanded shift, monotonically across decades of iterations.
  double frac_10 = 0.0, frac_100 = 0.0, frac_1000 = 0.0;
  for (int it = 1; it <= 1000; ++it) {
    isam.update();
    if (it == 10 || it == 100 || it == 1000) {
      const double f =
          (isam.calculateEstimate<Pose3>(key(0)).translation() - first_before.translation())
              .norm() /
          shift.norm();
      if (it == 10) frac_10 = f;
      if (it == 100) frac_100 = f;
      if (it == 1000) frac_1000 = f;
    }
  }
  EXPECT_GT(frac_10, 0.05);  // moving from the very first updates
  EXPECT_GT(frac_100, frac_10);
  EXPECT_GT(frac_1000, frac_100);
  EXPECT_GT(frac_1000, 0.9);  // most of the shift absorbed
  // The prior end is satisfied long before the gauge end finishes drifting.
  const Eigen::Vector3d last_after = isam.calculateEstimate<Pose3>(key(kChainLength)).translation();
  EXPECT_LT((last_after - (last_before.translation() + shift)).norm(), 1e-2);
}
