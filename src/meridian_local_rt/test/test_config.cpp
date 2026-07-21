#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>

#include "meridian/local_rt/config.hpp"

namespace meridian::local_rt {
namespace {

TEST(Config, DefaultsAreValidAndTimestampOffsetIsExplicit) {
  Config config;
  config.extrinsics.lidar_time_offset_to_imu = std::chrono::microseconds(275);

  EXPECT_TRUE(config.validate().empty());
  EXPECT_EQ(config.extrinsics.lidar_time_offset_to_imu, std::chrono::microseconds(275));
  EXPECT_EQ(config.extrinsics.T_base_imu, core::Pose3d());
  EXPECT_EQ(config.extrinsics.T_imu_lidar, core::Pose3d());
}

TEST(Config, ReportsCovarianceAndOrderingErrorsWithoutFailingFast) {
  Config config;
  config.imu_model.gyroscope_covariance_density(0, 1) = 1.0;
  config.initialization.static_mode.block_duration = std::chrono::seconds(3);
  config.initialization.static_mode.support = std::chrono::seconds(2);
  config.imu_buffer.capacity = 0U;

  const std::vector<ConfigIssue> issues = config.validate();
  EXPECT_GE(issues.size(), 3U);
  EXPECT_NE(std::find_if(issues.begin(), issues.end(),
                         [](const ConfigIssue& issue) {
                           return issue.code == ConfigIssueCode::kNotSymmetric &&
                                  issue.field == "imu_model.gyroscope_covariance_density";
                         }),
            issues.end());
  EXPECT_NE(std::find_if(issues.begin(), issues.end(),
                         [](const ConfigIssue& issue) {
                           return issue.code == ConfigIssueCode::kInvalidOrdering;
                         }),
            issues.end());
  EXPECT_NE(
      std::find_if(issues.begin(), issues.end(),
                   [](const ConfigIssue& issue) { return issue.field == "imu_buffer.capacity"; }),
      issues.end());
}

}  // namespace
}  // namespace meridian::local_rt
