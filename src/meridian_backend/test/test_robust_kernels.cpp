#include <gtest/gtest.h>
#include <gtsam/linear/NoiseModel.h>

#include <Eigen/Core>

#include "robust_kernels.hpp"

using meridian::backend::chi2inv;
using meridian::backend::make_gnss_noise;
using meridian::backend::make_huber_noise;

TEST(RobustKernels, Chi2InvMatchesKnownQuantiles) {
  EXPECT_NEAR(chi2inv(0.99, 6), 16.81, 1e-2);
  EXPECT_NEAR(chi2inv(0.99, 3), 11.34, 1e-2);
  EXPECT_NEAR(chi2inv(0.95, 3), 7.815, 1e-2);
  EXPECT_NEAR(chi2inv(0.95, 1), 3.841, 1e-2);
}

TEST(RobustKernels, HuberNoiseIsNonNullRobustModel) {
  Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
  const gtsam::SharedNoiseModel model = make_huber_noise(cov, 1.345);
  ASSERT_TRUE(model != nullptr);

  const auto robust = boost::dynamic_pointer_cast<const gtsam::noiseModel::Robust>(model);
  ASSERT_TRUE(robust != nullptr);
}

TEST(RobustKernels, HuberLeavesSmallResidualUndamped) {
  Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(3, 3);
  const gtsam::SharedNoiseModel model = make_huber_noise(cov, 1.345);
  const auto robust = boost::dynamic_pointer_cast<const gtsam::noiseModel::Robust>(model);
  ASSERT_TRUE(robust != nullptr);

  // Inside the Huber band the weight stays unity (quadratic regime).
  EXPECT_NEAR(robust->robust()->weight(0.5), 1.0, 1e-9);
}

TEST(RobustKernels, HuberDownweightsLargeResidual) {
  Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(3, 3);
  const double huber_k = 1.345;
  const gtsam::SharedNoiseModel model = make_huber_noise(cov, huber_k);
  const auto robust = boost::dynamic_pointer_cast<const gtsam::noiseModel::Robust>(model);
  ASSERT_TRUE(robust != nullptr);

  // A 10-sigma whitened residual sits in the linear regime: weight = k / |r| < 1.
  const double w = robust->robust()->weight(10.0);
  EXPECT_LT(w, 1.0);
  EXPECT_NEAR(w, huber_k / 10.0, 1e-9);
}

TEST(RobustKernels, GnssNoiseIsRobustAndUsesEnuCovariance) {
  Eigen::Matrix3d cov_enu = Eigen::Matrix3d::Identity();
  cov_enu.diagonal() << 4.0, 4.0, 9.0;  // [m^2]
  const gtsam::SharedNoiseModel model = make_gnss_noise(cov_enu, 1.345);
  ASSERT_TRUE(model != nullptr);

  const auto robust = boost::dynamic_pointer_cast<const gtsam::noiseModel::Robust>(model);
  ASSERT_TRUE(robust != nullptr);
  EXPECT_EQ(robust->noise()->dim(), 3u);
}
