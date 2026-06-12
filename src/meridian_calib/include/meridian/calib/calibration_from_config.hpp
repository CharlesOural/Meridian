#pragma once

#include <memory>

#include "meridian/calib/calibration_set.hpp"
#include "meridian/config/config.hpp"

namespace meridian {

// Builds the front-end's calibration snapshot from the static config: the LiDAR->F_e
// and camera->F_e extrinsics from the sensor block, the camera intrinsics (pinhole +
// distortion model/coeffs + image size), and the IMU noise densities from the sensor
// covariance fields. The estimation frame is the IMU link. The snapshot is a value; a
// later refinement publishes a new one rather than mutating this.
std::shared_ptr<const CalibrationSet> calibrationFromConfig(const SensorsConfig& sensors);

}  // namespace meridian
