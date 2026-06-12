#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "gnc_consolidation.hpp"
#include "meridian/common/pose.hpp"

using meridian::Pose;
using meridian::backend::gnc_consolidate;
using meridian::backend::GncConsolidationInput;
using meridian::backend::GncLoop;
using meridian::backend::GncOdom;

namespace {

constexpr double kBarc2 = 16.8119;  // chi^2_{6,0.99}
constexpr double kRejectW = 0.1;

Pose make_pose(double x, double y, double yaw) {
  const Eigen::Quaterniond q(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
  return Pose(q.normalized(), Eigen::Vector3d(x, y, 0.0));
}

Eigen::Matrix<double, 6, 6> tight_cov() {
  // Small isotropic covariance so loop residuals scale to large Mahalanobis distances; this
  // makes the inlier/outlier separation crisp under the chi^2 threshold.
  return Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
}

// Relative transform from a to b consistent with the supplied estimates: T_a_b = a^{-1} * b.
Pose relative(const Pose& a, const Pose& b) {
  return a.inverse() * b;
}

// A unit square traversed counter-clockwise, headings tangent to the path. Four keyframes.
std::unordered_map<std::uint64_t, Pose> square_estimate() {
  std::unordered_map<std::uint64_t, Pose> est;
  est[0] = make_pose(0.0, 0.0, 0.0);
  est[1] = make_pose(1.0, 0.0, M_PI_2);
  est[2] = make_pose(1.0, 1.0, M_PI);
  est[3] = make_pose(0.0, 1.0, -M_PI_2);
  return est;
}

GncOdom odom_edge(std::uint64_t a, std::uint64_t b,
                  const std::unordered_map<std::uint64_t, Pose>& est) {
  return GncOdom{a, b, relative(est.at(a), est.at(b)), tight_cov()};
}

GncLoop loop_edge(std::size_t handle, std::uint64_t a, std::uint64_t b, const Pose& meas,
                  const std::unordered_map<std::uint64_t, Pose>& /*est*/) {
  return GncLoop{handle, a, b, meas, tight_cov()};
}

}  // namespace

TEST(GncConsolidation, RejectsSingleBadLoopKeepsGood) {
  const auto est = square_estimate();
  GncConsolidationInput in;
  in.keyframes = {0, 1, 2, 3};
  in.estimate = est;
  in.barc2 = kBarc2;
  in.reject_w = kRejectW;
  in.prior_sigma = 1.0;

  // Trusted odometry chain along the square edges.
  in.odom = {odom_edge(0, 1, est), odom_edge(1, 2, est), odom_edge(2, 3, est)};

  // Three good loops exactly consistent with the estimates.
  in.loops.push_back(loop_edge(100, 0, 2, relative(est.at(0), est.at(2)), est));
  in.loops.push_back(loop_edge(101, 1, 3, relative(est.at(1), est.at(3)), est));
  in.loops.push_back(loop_edge(102, 3, 0, relative(est.at(3), est.at(0)), est));

  // One bad loop closing 0->3 but claiming a 5 m translation error: grossly inconsistent.
  Pose bad = relative(est.at(0), est.at(3));
  bad.t += Eigen::Vector3d(5.0, -3.0, 0.0);
  in.loops.push_back(loop_edge(999, 0, 3, bad, est));

  const std::vector<std::size_t> rejected = gnc_consolidate(in);
  ASSERT_EQ(rejected.size(), 1u);
  EXPECT_EQ(rejected.front(), 999u);
}

TEST(GncConsolidation, AllGoodLoopsRejectsNothing) {
  const auto est = square_estimate();
  GncConsolidationInput in;
  in.keyframes = {0, 1, 2, 3};
  in.estimate = est;
  in.barc2 = kBarc2;
  in.reject_w = kRejectW;
  in.prior_sigma = 1.0;
  in.odom = {odom_edge(0, 1, est), odom_edge(1, 2, est), odom_edge(2, 3, est)};

  in.loops.push_back(loop_edge(10, 0, 2, relative(est.at(0), est.at(2)), est));
  in.loops.push_back(loop_edge(11, 1, 3, relative(est.at(1), est.at(3)), est));
  in.loops.push_back(loop_edge(12, 3, 0, relative(est.at(3), est.at(0)), est));
  in.loops.push_back(loop_edge(13, 0, 3, relative(est.at(0), est.at(3)), est));

  const std::vector<std::size_t> rejected = gnc_consolidate(in);
  EXPECT_TRUE(rejected.empty());
}

TEST(GncConsolidation, EmptyInputReturnsEmptyWithoutThrowing) {
  GncConsolidationInput in;
  in.barc2 = kBarc2;
  in.reject_w = kRejectW;
  std::vector<std::size_t> rejected;
  ASSERT_NO_THROW(rejected = gnc_consolidate(in));
  EXPECT_TRUE(rejected.empty());
}

TEST(GncConsolidation, MissingEstimateRejectsNothing) {
  // A keyframe referenced with no estimate is unanchorable: the call must bail safely.
  const auto est = square_estimate();
  GncConsolidationInput in;
  in.keyframes = {0, 1, 2, 3, 7};  // id 7 has no estimate entry
  in.estimate = est;
  in.barc2 = kBarc2;
  in.reject_w = kRejectW;
  in.odom = {odom_edge(0, 1, est)};
  in.loops.push_back(loop_edge(1, 0, 2, relative(est.at(0), est.at(2)), est));

  std::vector<std::size_t> rejected;
  ASSERT_NO_THROW(rejected = gnc_consolidate(in));
  EXPECT_TRUE(rejected.empty());
}
