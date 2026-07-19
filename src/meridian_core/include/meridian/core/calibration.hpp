#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "meridian/core/geometry.hpp"
#include "meridian/core/result.hpp"
#include "meridian/core/strong_id.hpp"
#include "meridian/core/time.hpp"

namespace meridian::core {

// Meridian follows the conventional T_destination_source notation: applying
// T_imu_lidar maps a point expressed in the lidar frame into the IMU frame.
// Distinct value types make transform direction mistakes visible at compile
// time instead of passing unlabelled SE(3) values between modules.
class ImuFromLidarTransform {
 public:
  explicit ImuFromLidarTransform(Pose3d T_imu_lidar)
      : T_imu_lidar_(std::move(T_imu_lidar)) {}

  [[nodiscard]] const Pose3d& T_imu_lidar() const noexcept {
    return T_imu_lidar_;
  }

 private:
  Pose3d T_imu_lidar_;
};

class ImuFromCameraTransform {
 public:
  explicit ImuFromCameraTransform(Pose3d T_imu_camera)
      : T_imu_camera_(std::move(T_imu_camera)) {}

  [[nodiscard]] const Pose3d& T_imu_camera() const noexcept {
    return T_imu_camera_;
  }

 private:
  Pose3d T_imu_camera_;
};

class BaseFromImuTransform {
 public:
  explicit BaseFromImuTransform(Pose3d T_base_imu)
      : T_base_imu_(std::move(T_base_imu)) {}

  [[nodiscard]] const Pose3d& T_base_imu() const noexcept {
    return T_base_imu_;
  }

 private:
  Pose3d T_base_imu_;
};

enum class ImuSensorModel {
  Generic,
  BoschBmi085,
};

// Continuous-time Kalibr/GTSAM-style quantities.  They must not be confused
// with per-sample standard deviations: discretization belongs to the IMU
// integration API and depends on the actual sample interval.
class ImuNoiseModel {
 public:
  ImuNoiseModel(double accelerometer_noise_density_mps2_per_sqrt_hz,
                double gyroscope_noise_density_radps_per_sqrt_hz,
                double accelerometer_bias_random_walk_mps3_per_sqrt_hz,
                double gyroscope_bias_random_walk_radps2_per_sqrt_hz)
      : accelerometer_noise_density_mps2_per_sqrt_hz_(
            accelerometer_noise_density_mps2_per_sqrt_hz),
        gyroscope_noise_density_radps_per_sqrt_hz_(
            gyroscope_noise_density_radps_per_sqrt_hz),
        accelerometer_bias_random_walk_mps3_per_sqrt_hz_(
            accelerometer_bias_random_walk_mps3_per_sqrt_hz),
        gyroscope_bias_random_walk_radps2_per_sqrt_hz_(
            gyroscope_bias_random_walk_radps2_per_sqrt_hz) {}

  [[nodiscard]] double accelerometerNoiseDensity() const noexcept {
    return accelerometer_noise_density_mps2_per_sqrt_hz_;
  }
  [[nodiscard]] double gyroscopeNoiseDensity() const noexcept {
    return gyroscope_noise_density_radps_per_sqrt_hz_;
  }
  [[nodiscard]] double accelerometerBiasRandomWalk() const noexcept {
    return accelerometer_bias_random_walk_mps3_per_sqrt_hz_;
  }
  [[nodiscard]] double gyroscopeBiasRandomWalk() const noexcept {
    return gyroscope_bias_random_walk_radps2_per_sqrt_hz_;
  }

  [[nodiscard]] bool valid() const noexcept {
    return positiveFinite(accelerometer_noise_density_mps2_per_sqrt_hz_) &&
           positiveFinite(gyroscope_noise_density_radps_per_sqrt_hz_) &&
           positiveFinite(
               accelerometer_bias_random_walk_mps3_per_sqrt_hz_) &&
           positiveFinite(gyroscope_bias_random_walk_radps2_per_sqrt_hz_);
  }

 private:
  [[nodiscard]] static bool positiveFinite(double value) noexcept {
    return std::isfinite(value) && value > 0.0;
  }

  double accelerometer_noise_density_mps2_per_sqrt_hz_;
  double gyroscope_noise_density_radps_per_sqrt_hz_;
  double accelerometer_bias_random_walk_mps3_per_sqrt_hz_;
  double gyroscope_bias_random_walk_radps2_per_sqrt_hz_;
};

class ImuCalibration {
 public:
  ImuCalibration(std::string name, std::string source_topic,
                 ImuSensorModel sensor_model, double nominal_rate_hz,
                 double gravity_magnitude_mps2, ImuNoiseModel noise)
      : name_(std::move(name)),
        source_topic_(std::move(source_topic)),
        sensor_model_(sensor_model),
        nominal_rate_hz_(nominal_rate_hz),
        gravity_magnitude_mps2_(gravity_magnitude_mps2),
        noise_(std::move(noise)) {}

  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] const std::string& sourceTopic() const noexcept {
    return source_topic_;
  }
  [[nodiscard]] ImuSensorModel sensorModel() const noexcept {
    return sensor_model_;
  }
  [[nodiscard]] double nominalRateHz() const noexcept {
    return nominal_rate_hz_;
  }
  [[nodiscard]] double gravityMagnitude() const noexcept {
    return gravity_magnitude_mps2_;
  }
  [[nodiscard]] const ImuNoiseModel& noise() const noexcept { return noise_; }

  [[nodiscard]] bool valid() const noexcept {
    return !name_.empty() && !source_topic_.empty() &&
           std::isfinite(nominal_rate_hz_) && nominal_rate_hz_ > 0.0 &&
           std::isfinite(gravity_magnitude_mps2_) &&
           gravity_magnitude_mps2_ > 0.0 && noise_.valid();
  }

 private:
  std::string name_;
  std::string source_topic_;
  ImuSensorModel sensor_model_;
  double nominal_rate_hz_;
  double gravity_magnitude_mps2_;
  ImuNoiseModel noise_;
};

struct ImageDimensions {
  std::uint32_t width{};
  std::uint32_t height{};

  [[nodiscard]] bool valid() const noexcept { return width > 0U && height > 0U; }
};

// The Newer College cameras use Kalibr's pinhole projection with the
// equidistant four-coefficient model. The k1,k2,k3,k4 order matches the
// reference OKVIS2-X PinholeCamera<EquidistantDistortion> construction.
class PinholeEquidistantCameraModel {
 public:
  PinholeEquidistantCameraModel(double fx, double fy, double cx, double cy,
                                std::array<double, 4> distortion,
                                ImageDimensions image_size)
      : fx_(fx),
        fy_(fy),
        cx_(cx),
        cy_(cy),
        distortion_(distortion),
        image_size_(image_size) {}

  [[nodiscard]] double fx() const noexcept { return fx_; }
  [[nodiscard]] double fy() const noexcept { return fy_; }
  [[nodiscard]] double cx() const noexcept { return cx_; }
  [[nodiscard]] double cy() const noexcept { return cy_; }
  [[nodiscard]] const std::array<double, 4>& distortion() const noexcept {
    return distortion_;
  }
  [[nodiscard]] ImageDimensions imageSize() const noexcept {
    return image_size_;
  }

  [[nodiscard]] Eigen::Matrix3d cameraMatrix() const noexcept {
    Eigen::Matrix3d matrix = Eigen::Matrix3d::Identity();
    matrix(0, 0) = fx_;
    matrix(1, 1) = fy_;
    matrix(0, 2) = cx_;
    matrix(1, 2) = cy_;
    return matrix;
  }

  [[nodiscard]] bool valid() const noexcept {
    if (!image_size_.valid() || !positiveFinite(fx_) || !positiveFinite(fy_) ||
        !std::isfinite(cx_) || !std::isfinite(cy_)) {
      return false;
    }
    for (const double coefficient : distortion_) {
      if (!std::isfinite(coefficient)) {
        return false;
      }
    }
    return true;
  }

 private:
  [[nodiscard]] static bool positiveFinite(double value) noexcept {
    return std::isfinite(value) && value > 0.0;
  }

  double fx_;
  double fy_;
  double cx_;
  double cy_;
  std::array<double, 4> distortion_;
  ImageDimensions image_size_;
};

enum class CameraTimestampReference {
  SensorReportedImageTime,
  ExposureMidpoint,
};

enum class CameraImuTimeConvention {
  ImuTimeEqualsCameraTimePlusOffset,
};

class CameraTimingCalibration {
 public:
  CameraTimingCalibration(CameraTimestampReference timestamp_reference,
                          Duration imu_time_minus_camera_time)
      : timestamp_reference_(timestamp_reference),
        imu_time_minus_camera_time_(imu_time_minus_camera_time) {}

  [[nodiscard]] CameraTimestampReference timestampReference() const noexcept {
    return timestamp_reference_;
  }
  [[nodiscard]] CameraImuTimeConvention convention() const noexcept {
    return CameraImuTimeConvention::ImuTimeEqualsCameraTimePlusOffset;
  }
  [[nodiscard]] Duration imuTimeMinusCameraTime() const noexcept {
    return imu_time_minus_camera_time_;
  }
  [[nodiscard]] FusionTime cameraToImuTime(FusionTime camera_time) const noexcept {
    return camera_time + imu_time_minus_camera_time_;
  }

 private:
  CameraTimestampReference timestamp_reference_;
  Duration imu_time_minus_camera_time_;
};

class CameraCalibration {
 public:
  CameraCalibration(CameraId id, std::string name, std::string source_topic,
                    PinholeEquidistantCameraModel model,
                    ImuFromCameraTransform extrinsics,
                    CameraTimingCalibration timing)
      : id_(id),
        name_(std::move(name)),
        source_topic_(std::move(source_topic)),
        model_(std::move(model)),
        extrinsics_(std::move(extrinsics)),
        timing_(timing) {}

  [[nodiscard]] CameraId id() const noexcept { return id_; }
  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] const std::string& sourceTopic() const noexcept {
    return source_topic_;
  }
  [[nodiscard]] const PinholeEquidistantCameraModel& model() const noexcept {
    return model_;
  }
  [[nodiscard]] const ImuFromCameraTransform& extrinsics() const noexcept {
    return extrinsics_;
  }
  [[nodiscard]] const CameraTimingCalibration& timing() const noexcept {
    return timing_;
  }

 private:
  CameraId id_;
  std::string name_;
  std::string source_topic_;
  PinholeEquidistantCameraModel model_;
  ImuFromCameraTransform extrinsics_;
  CameraTimingCalibration timing_;
};

enum class LidarSweepTimestampReference {
  SweepStart,
};

enum class LidarPointTimeConvention {
  OffsetFromSweepTimestamp,
};

class LidarTimingCalibration {
 public:
  LidarTimingCalibration(LidarSweepTimestampReference sweep_reference,
                         LidarPointTimeConvention point_time_convention)
      : sweep_reference_(sweep_reference),
        point_time_convention_(point_time_convention) {}

  [[nodiscard]] LidarSweepTimestampReference sweepReference() const noexcept {
    return sweep_reference_;
  }
  [[nodiscard]] LidarPointTimeConvention pointTimeConvention() const noexcept {
    return point_time_convention_;
  }

 private:
  LidarSweepTimestampReference sweep_reference_;
  LidarPointTimeConvention point_time_convention_;
};

class LidarCalibration {
 public:
  LidarCalibration(LidarId id, std::string name, std::string source_topic,
                   ImuFromLidarTransform extrinsics,
                   LidarTimingCalibration timing)
      : id_(id),
        name_(std::move(name)),
        source_topic_(std::move(source_topic)),
        extrinsics_(std::move(extrinsics)),
        timing_(timing) {}

  [[nodiscard]] LidarId id() const noexcept { return id_; }
  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] const std::string& sourceTopic() const noexcept {
    return source_topic_;
  }
  [[nodiscard]] const ImuFromLidarTransform& extrinsics() const noexcept {
    return extrinsics_;
  }
  [[nodiscard]] const LidarTimingCalibration& timing() const noexcept {
    return timing_;
  }

 private:
  LidarId id_;
  std::string name_;
  std::string source_topic_;
  ImuFromLidarTransform extrinsics_;
  LidarTimingCalibration timing_;
};

enum class CalibrationValidationError {
  InvalidEpoch,
  InvalidImu,
  InvalidBaseFromImuTransform,
  InvalidLidar,
  InvalidLidarTransform,
  InvalidCamera,
  InvalidCameraModel,
  InvalidCameraTransform,
  DuplicateCameraId,
};

[[nodiscard]] inline bool isRigidTransform(const Pose3d& pose) noexcept {
  const Eigen::Matrix4d matrix = pose.matrix();
  if (!matrix.allFinite()) {
    return false;
  }
  const Eigen::Matrix3d rotation = matrix.template block<3, 3>(0, 0);
  return (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() <
             1.0e-8 &&
         std::abs(rotation.determinant() - 1.0) < 1.0e-8;
}

class CalibrationBundle {
 public:
  [[nodiscard]] static Result<CalibrationBundle, CalibrationValidationError>
  create(CalibrationEpoch epoch, ImuCalibration imu,
         BaseFromImuTransform base_from_imu, LidarCalibration lidar,
         std::vector<CameraCalibration> cameras) {
    if (!epoch.valid()) {
      return Result<CalibrationBundle, CalibrationValidationError>::failure(
          CalibrationValidationError::InvalidEpoch);
    }
    if (!imu.valid()) {
      return Result<CalibrationBundle, CalibrationValidationError>::failure(
          CalibrationValidationError::InvalidImu);
    }
    if (!isRigidTransform(base_from_imu.T_base_imu())) {
      return Result<CalibrationBundle, CalibrationValidationError>::failure(
          CalibrationValidationError::InvalidBaseFromImuTransform);
    }
    if (!lidar.id().valid() || lidar.name().empty() ||
        lidar.sourceTopic().empty()) {
      return Result<CalibrationBundle, CalibrationValidationError>::failure(
          CalibrationValidationError::InvalidLidar);
    }
    if (!isRigidTransform(lidar.extrinsics().T_imu_lidar())) {
      return Result<CalibrationBundle, CalibrationValidationError>::failure(
          CalibrationValidationError::InvalidLidarTransform);
    }
    for (std::size_t index = 0; index < cameras.size(); ++index) {
      const CameraCalibration& camera = cameras[index];
      if (!camera.id().valid() || camera.name().empty() ||
          camera.sourceTopic().empty()) {
        return Result<CalibrationBundle, CalibrationValidationError>::failure(
            CalibrationValidationError::InvalidCamera);
      }
      if (!camera.model().valid()) {
        return Result<CalibrationBundle, CalibrationValidationError>::failure(
            CalibrationValidationError::InvalidCameraModel);
      }
      if (!isRigidTransform(camera.extrinsics().T_imu_camera())) {
        return Result<CalibrationBundle, CalibrationValidationError>::failure(
            CalibrationValidationError::InvalidCameraTransform);
      }
      for (std::size_t previous = 0; previous < index; ++previous) {
        if (cameras[previous].id() == camera.id()) {
          return Result<CalibrationBundle,
                        CalibrationValidationError>::failure(
              CalibrationValidationError::DuplicateCameraId);
        }
      }
    }

    return Result<CalibrationBundle, CalibrationValidationError>::success(
        CalibrationBundle(epoch, std::move(imu), std::move(base_from_imu),
                          std::move(lidar), std::move(cameras)));
  }

  [[nodiscard]] CalibrationEpoch epoch() const noexcept { return epoch_; }
  [[nodiscard]] const ImuCalibration& imu() const noexcept { return imu_; }
  [[nodiscard]] const BaseFromImuTransform& baseFromImu() const noexcept {
    return base_from_imu_;
  }
  [[nodiscard]] const LidarCalibration& lidar() const noexcept {
    return lidar_;
  }
  [[nodiscard]] const std::vector<CameraCalibration>& cameras() const noexcept {
    return cameras_;
  }
  [[nodiscard]] const CameraCalibration* camera(CameraId id) const noexcept {
    for (const CameraCalibration& camera : cameras_) {
      if (camera.id() == id) {
        return &camera;
      }
    }
    return nullptr;
  }

 private:
  CalibrationBundle(CalibrationEpoch epoch, ImuCalibration imu,
                    BaseFromImuTransform base_from_imu,
                    LidarCalibration lidar,
                    std::vector<CameraCalibration> cameras)
      : epoch_(epoch),
        imu_(std::move(imu)),
        base_from_imu_(std::move(base_from_imu)),
        lidar_(std::move(lidar)),
        cameras_(std::move(cameras)) {}

  CalibrationEpoch epoch_;
  ImuCalibration imu_;
  BaseFromImuTransform base_from_imu_;
  LidarCalibration lidar_;
  std::vector<CameraCalibration> cameras_;
};

}  // namespace meridian::core
