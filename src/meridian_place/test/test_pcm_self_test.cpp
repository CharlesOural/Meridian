#include <gtest/gtest.h>

#include <Eigen/Core>

#include "meridian/common/pose.hpp"
#include "pcm_self_test.hpp"

using meridian::chi2InvDof6;
using meridian::loopAgreesWithOdometry;
using meridian::Pose;
using M6 = Eigen::Matrix<double, 6, 6>;

namespace {
Pose pose_t(double x, double y, double z) {
  return Pose{Eigen::Quaterniond::Identity(), Eigen::Vector3d(x, y, z)};
}
}  // namespace

TEST(PcmSelfTest, Chi2InverseMatchesKnownQuantiles) {
  EXPECT_NEAR(chi2InvDof6(0.99), 16.8119, 0.02);
  EXPECT_NEAR(chi2InvDof6(0.95), 12.5916, 0.02);
  EXPECT_NEAR(chi2InvDof6(0.50), 5.3481, 0.02);
}

TEST(PcmSelfTest, LoopAgreeingWithOdometryAccepted) {
  const Pose odom = pose_t(2, 0, 0);
  const Pose loop = pose_t(2, 0, 0);  // identical to the odometry estimate
  const M6 cov = 0.01 * M6::Identity();
  double chi2 = -1;
  EXPECT_TRUE(loopAgreesWithOdometry(odom, loop, cov, chi2InvDof6(0.99), &chi2));
  EXPECT_NEAR(chi2, 0.0, 1e-9);
}

TEST(PcmSelfTest, ContradictingLoopWithTightCovRejected) {
  const Pose odom = pose_t(0, 0, 0);
  const Pose loop = pose_t(5, 0, 0);  // 5 m off what odometry says
  const M6 cov = 0.01 * M6::Identity();  // confident -> the disagreement is significant
  EXPECT_FALSE(loopAgreesWithOdometry(odom, loop, cov, chi2InvDof6(0.99)));
}

TEST(PcmSelfTest, LargeCorrectionUnderWideChainCovAccepted) {
  // The same 5 m disagreement, but the chain has drifted a lot so the combined covariance is
  // wide: a genuine large-correction loop must NOT be rejected.
  const Pose odom = pose_t(0, 0, 0);
  const Pose loop = pose_t(5, 0, 0);
  const M6 cov = 10.0 * M6::Identity();
  EXPECT_TRUE(loopAgreesWithOdometry(odom, loop, cov, chi2InvDof6(0.99)));
}

TEST(PcmSelfTest, RotationDisagreementIsWhitenedToo) {
  const Pose odom = pose_t(0, 0, 0);
  const Pose loop{Eigen::Quaterniond(Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitZ())),
                  Eigen::Vector3d::Zero()};  // ~34 deg yaw off
  const M6 tight = 1e-3 * M6::Identity();
  EXPECT_FALSE(loopAgreesWithOdometry(odom, loop, tight, chi2InvDof6(0.99)));
  const M6 wide = 1.0 * M6::Identity();
  EXPECT_TRUE(loopAgreesWithOdometry(odom, loop, wide, chi2InvDof6(0.99)));
}
