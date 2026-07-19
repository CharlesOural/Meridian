#include <array>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "meridian/core/calibration.hpp"

namespace meridian::core {
namespace {

[[nodiscard]] ImuCalibration validImu() {
  return ImuCalibration(
      "imu", "/imu", ImuSensorModel::BoschBmi085, 200.0, 9.80665,
      ImuNoiseModel(1.0e-3, 2.0e-4, 3.0e-5, 4.0e-6));
}

[[nodiscard]] CameraCalibration validCamera(CameraId id) {
  return CameraCalibration(
      id, "cam" + std::to_string(id.value()), "/cam",
      PinholeEquidistantCameraModel(350.0, 351.0, 360.0, 270.0,
                                    std::array<double, 4>{0.1, 0.0, 0.0, 0.0},
                                    ImageDimensions{720U, 540U}),
      ImuFromCameraTransform(Pose3d()),
      CameraTimingCalibration(CameraTimestampReference::SensorReportedImageTime,
                              Duration{2'000'000}));
}

[[nodiscard]] LidarCalibration validLidar() {
  return LidarCalibration(
      LidarId(0U), "lidar", "/points", ImuFromLidarTransform(Pose3d()),
      LidarTimingCalibration(LidarSweepTimestampReference::SweepStart,
                             LidarPointTimeConvention::OffsetFromSweepTimestamp));
}

TEST(CalibrationApi, TransformTypesExposeOnlyTheirNamedDirection) {
  const Pose3d T_imu_lidar(Sophus::SO3d(), Eigen::Vector3d(1.0, 2.0, 3.0));
  const ImuFromLidarTransform transform(T_imu_lidar);

  EXPECT_TRUE(transform.T_imu_lidar().matrix().isApprox(T_imu_lidar.matrix()));
}

TEST(CalibrationApi, CameraOffsetUsesDocumentedKalibrSign) {
  const CameraTimingCalibration timing(
      CameraTimestampReference::SensorReportedImageTime,
      Duration{2'000'000});

  EXPECT_EQ(timing.convention(),
            CameraImuTimeConvention::ImuTimeEqualsCameraTimePlusOffset);
  EXPECT_EQ(timing.cameraToImuTime(FusionTime{10'000'000}).nanoseconds,
            12'000'000);
}

TEST(CalibrationApi, RejectsNonFiniteImuNoise) {
  const ImuCalibration imu(
      "imu", "/imu", ImuSensorModel::Generic, 200.0, 9.80665,
      ImuNoiseModel(std::numeric_limits<double>::quiet_NaN(), 2.0e-4,
                    3.0e-5, 4.0e-6));

  auto bundle = CalibrationBundle::create(
      CalibrationEpoch(1U), imu, BaseFromImuTransform(Pose3d()), validLidar(),
      {validCamera(CameraId(0U))});

  ASSERT_FALSE(bundle);
  EXPECT_EQ(bundle.error(), CalibrationValidationError::InvalidImu);
}

TEST(CalibrationApi, RejectsDuplicateCameraIds) {
  auto bundle = CalibrationBundle::create(
      CalibrationEpoch(1U), validImu(), BaseFromImuTransform(Pose3d()),
      validLidar(),
      {validCamera(CameraId(0U)), validCamera(CameraId(0U))});

  ASSERT_FALSE(bundle);
  EXPECT_EQ(bundle.error(), CalibrationValidationError::DuplicateCameraId);
}

TEST(CalibrationApi, FindsCameraByTypedId) {
  auto bundle = CalibrationBundle::create(
      CalibrationEpoch(7U), validImu(), BaseFromImuTransform(Pose3d()),
      validLidar(),
      {validCamera(CameraId(0U)), validCamera(CameraId(3U))});

  ASSERT_TRUE(bundle);
  EXPECT_EQ(bundle.value().epoch(), CalibrationEpoch(7U));
  ASSERT_NE(bundle.value().camera(CameraId(3U)), nullptr);
  EXPECT_EQ(bundle.value().camera(CameraId(3U))->name(), "cam3");
  EXPECT_EQ(bundle.value().camera(CameraId(4U)), nullptr);
}

}  // namespace
}  // namespace meridian::core
