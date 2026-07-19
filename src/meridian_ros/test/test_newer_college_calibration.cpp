#include <cmath>
#include <filesystem>

#include <gtest/gtest.h>

#include "meridian/core/calibration.hpp"
#include "meridian/ros/newer_college_calibration.hpp"

namespace meridian::ros {
namespace {

[[nodiscard]] std::filesystem::path calibrationRoot() {
  const std::filesystem::path source_file(__FILE__);
  return source_file.parent_path()
             .parent_path()
             .parent_path()
             .parent_path() /
         "bags/newer-college/calib";
}

[[nodiscard]] core::ImuCalibration testImuCalibration() {
  // The NCD archive has no Allan-variance file. These distinctive values
  // verify that the caller-owned noise profile survives loading unchanged.
  return core::ImuCalibration(
      "alphasense_imu", "/alphasense_driver_ros/imu",
      core::ImuSensorModel::BoschBmi085, 200.0, 9.80665,
      core::ImuNoiseModel(0.0011002607647952406,
                          0.00022632861789099884,
                          3.390710627779767e-05,
                          8.252445860125436e-06));
}

TEST(NewerCollegeCalibration, LoadsQuadCollectionOne) {
  auto calibration = loadNewerCollegeCalibration(
      calibrationRoot(), NewerCollegeCollection::Collection1,
      core::CalibrationEpoch(11U), testImuCalibration());

  ASSERT_TRUE(calibration) << calibration.error().detail;
  const core::CalibrationBundle& bundle = calibration.value();
  EXPECT_EQ(bundle.epoch(), core::CalibrationEpoch(11U));
  ASSERT_EQ(bundle.cameras().size(), 4U);

  const core::CameraCalibration* cam0 = bundle.camera(core::CameraId(0U));
  ASSERT_NE(cam0, nullptr);
  EXPECT_EQ(cam0->model().imageSize().width, 720U);
  EXPECT_EQ(cam0->model().imageSize().height, 540U);
  EXPECT_NEAR(cam0->model().fx(), 352.779, 1.0e-12);
  EXPECT_NEAR(cam0->model().distortion()[0], -0.04217, 1.0e-12);
  EXPECT_EQ(cam0->timing().imuTimeMinusCameraTime().nanoseconds, 1'800'800);
  EXPECT_EQ(cam0->timing().convention(),
            core::CameraImuTimeConvention::ImuTimeEqualsCameraTimePlusOffset);

  // Source YAML stores T_cam_imu; the framework API deliberately stores the
  // inverse T_imu_camera.
  const core::Pose3d T_cam_imu =
      cam0->extrinsics().T_imu_camera().inverse();
  EXPECT_NEAR(T_cam_imu.translation().x(), -0.04746141784769429, 1.0e-12);
  EXPECT_NEAR(T_cam_imu.translation().y(), 0.009164529205567274, 1.0e-12);
  EXPECT_NEAR(T_cam_imu.translation().z(), -0.04904955326238581, 1.0e-12);

  const core::Pose3d& T_imu_lidar =
      bundle.lidar().extrinsics().T_imu_lidar();
  EXPECT_NEAR(T_imu_lidar.translation().x(), -0.037, 1.0e-12);
  EXPECT_NEAR(T_imu_lidar.translation().y(), -0.008, 1.0e-12);
  EXPECT_NEAR(T_imu_lidar.translation().z(), -0.026, 1.0e-12);
  EXPECT_NEAR(T_imu_lidar.so3().matrix()(0, 0), 1.0, 1.0e-12);
  EXPECT_NEAR(T_imu_lidar.so3().matrix()(1, 1), -1.0, 1.0e-12);
  EXPECT_NEAR(T_imu_lidar.so3().matrix()(2, 2), -1.0, 1.0e-12);

  const core::ImuNoiseModel& noise = bundle.imu().noise();
  EXPECT_DOUBLE_EQ(noise.accelerometerNoiseDensity(),
                   0.0011002607647952406);
  EXPECT_DOUBLE_EQ(noise.gyroscopeNoiseDensity(), 0.00022632861789099884);
  EXPECT_DOUBLE_EQ(noise.accelerometerBiasRandomWalk(),
                   3.390710627779767e-05);
  EXPECT_DOUBLE_EQ(noise.gyroscopeBiasRandomWalk(),
                   8.252445860125436e-06);
}

TEST(NewerCollegeCalibration, LoadsParkCollectionTwo) {
  auto calibration = loadNewerCollegeCalibration(
      calibrationRoot(), NewerCollegeCollection::Collection2,
      core::CalibrationEpoch(12U), testImuCalibration());

  ASSERT_TRUE(calibration) << calibration.error().detail;
  const core::CameraCalibration* cam0 =
      calibration.value().camera(core::CameraId(0U));
  const core::CameraCalibration* cam4 =
      calibration.value().camera(core::CameraId(4U));
  ASSERT_NE(cam0, nullptr);
  ASSERT_NE(cam4, nullptr);
  EXPECT_NEAR(cam0->model().fx(), 352.61035231977274, 1.0e-12);
  EXPECT_EQ(cam0->timing().imuTimeMinusCameraTime().nanoseconds, 2'010'722);
  EXPECT_EQ(cam4->sourceTopic(), "/alphasense_driver_ros/cam4");

  const core::Pose3d composed =
      calibration.value().baseFromImu().T_base_imu() *
      calibration.value().lidar().extrinsics().T_imu_lidar();
  EXPECT_NEAR(composed.translation().x(), 0.001, 1.0e-12);
  EXPECT_NEAR(composed.translation().y(), 0.0, 1.0e-12);
  EXPECT_NEAR(composed.translation().z(), 0.091, 1.0e-12);
  EXPECT_TRUE(composed.so3().matrix().isApprox(Eigen::Matrix3d::Identity(),
                                               1.0e-12));
}

TEST(NewerCollegeCalibration, MissingTreeReturnsTypedError) {
  auto calibration = loadNewerCollegeCalibration(
      calibrationRoot() / "does-not-exist",
      NewerCollegeCollection::Collection1, core::CalibrationEpoch(1U),
      testImuCalibration());

  ASSERT_FALSE(calibration);
  EXPECT_EQ(calibration.error().code, CalibrationLoadErrorCode::FileNotFound);
  EXPECT_FALSE(calibration.error().file.empty());
}

}  // namespace
}  // namespace meridian::ros
