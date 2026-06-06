#include "calibration_from_config.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <string>

namespace meridian {

namespace {

// The front-end squares each noise field to recover a variance, so a config variance
// maps to its standard deviation here.
double std_from_variance(double variance) {
  return variance > 0.0 ? std::sqrt(variance) : 0.0;
}

// Map the config distortion-model string to the IntrinsicsCamera tag. plumb_bob is a
// radtan alias; an empty/none string is no distortion; an unknown string falls back
// to no distortion (the config validator is the authority that rejects bad strings).
IntrinsicsCamera::Distortion distortionFromString(const std::string& model) {
  if (model == "radtan" || model == "plumb_bob") {
    return IntrinsicsCamera::Distortion::RadTan;
  }
  if (model == "equidistant" || model == "equi" || model == "fisheye") {
    return IntrinsicsCamera::Distortion::Equidistant;
  }
  return IntrinsicsCamera::Distortion::None;
}

}  // namespace

std::shared_ptr<const CalibrationSet> calibrationFromConfig(const SensorsConfig& sensors) {
  auto set = std::make_shared<CalibrationSet>();
  set->estimation_frame = Frame::ImuLink;

  Extrinsic lidar;
  lidar.child = Frame::OsSensor0;
  lidar.parent = Frame::ImuLink;
  lidar.T_parent_child =
      Pose{Eigen::Quaterniond(sensors.lidar.extrinsic_R), sensors.lidar.extrinsic_T};
  lidar.source = CalibSource::Manual;
  set->extrinsics.push_back(lidar);

  // T_imu_cam from the configured T_body_cam; the camera path needs this geometry to
  // place visual map points into the camera frame.
  Extrinsic cam;
  cam.child = Frame::CamLink;
  cam.parent = Frame::ImuLink;
  cam.T_parent_child = sensors.camera.extrinsic;
  cam.source = CalibSource::Manual;
  set->extrinsics.push_back(cam);

  IntrinsicsCamera intr;
  intr.fx = sensors.camera.intrinsics[0];
  intr.fy = sensors.camera.intrinsics[1];
  intr.cx = sensors.camera.intrinsics[2];
  intr.cy = sensors.camera.intrinsics[3];
  intr.model = distortionFromString(sensors.camera.distortion_model);
  intr.coeffs = sensors.camera.distortion_coeffs;
  intr.width = sensors.camera.width;
  intr.height = sensors.camera.height;
  set->cam_intrinsics[static_cast<std::uint8_t>(sensors.camera.id)] = intr;

  set->imu_acc_noise = std_from_variance(sensors.imu.cov_acc);
  set->imu_gyr_noise = std_from_variance(sensors.imu.cov_gyr);
  set->imu_acc_bias_rw = std_from_variance(sensors.imu.b_acc_cov);
  set->imu_gyr_bias_rw = std_from_variance(sensors.imu.b_gyr_cov);

  return set;
}

}  // namespace meridian
