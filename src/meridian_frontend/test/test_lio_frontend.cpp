#include <gtest/gtest.h>

#include <memory>

#include "meridian/calib/calibration_set.hpp"
#include "meridian/config/config.hpp"
#include "meridian/frontend/ifrontend.hpp"

namespace {

std::shared_ptr<const meridian::CalibrationSet> identityCalib() {
  auto c = std::make_shared<meridian::CalibrationSet>();
  c->estimation_frame = meridian::Frame::ImuLink;
  c->imu_acc_noise = 5e-2;
  c->imu_gyr_noise = 5e-3;
  c->imu_acc_bias_rw = 1e-4;
  c->imu_gyr_bias_rw = 1e-5;
  c->version = 1;
  meridian::Extrinsic e;
  e.child = meridian::Frame::OsSensor0;
  e.parent = meridian::Frame::ImuLink;
  e.T_parent_child = meridian::Pose{};  // LiDAR coincident with IMU for the synthetic rig
  c->extrinsics.push_back(e);
  return c;
}

}  // namespace

TEST(LioFrontEnd, FactoryBuildsLioFrontEnd) {
  auto fe = meridian::makeFrontEnd(meridian::FrontendConfig{}, identityCalib(), nullptr, true);
  ASSERT_NE(fe, nullptr);
  EXPECT_EQ(fe->live_state().stamp, 0);
}
