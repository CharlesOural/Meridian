#include "lio/imu_tracker.hpp"

#include <gtest/gtest.h>

#include "meridian/config/config.hpp"

TEST(ImuTracker, ConstructsUninitialized) {
  meridian::lio::ImuTracker tracker{meridian::FrontendLio{}};
  EXPECT_FALSE(tracker.initialized());
  EXPECT_TRUE(tracker.bias_gyro().isZero());
  EXPECT_TRUE(tracker.bias_accel().isZero());
}
