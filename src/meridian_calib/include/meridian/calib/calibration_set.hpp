#pragma once

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "meridian/calib/extrinsic.hpp"
#include "meridian/calib/intrinsics.hpp"
#include "meridian/common/frame.hpp"

namespace meridian {

// The complete, queryable calibration of the rig. Served as an immutable value
// snapshot: a refinement publishes a new set, never mutates an existing one.
struct CalibrationSet {
  Frame estimation_frame = Frame::ImuLink;  // F_e

  std::vector<Extrinsic> extrinsics;  // all sensor->F_e transforms
  std::unordered_map<std::uint8_t, IntrinsicsCamera> cam_intrinsics;  // by camera sensor_id

  // IMU noise parameters (Allan-variance derived), feeding the preintegration noise.
  double imu_acc_noise = 0.0, imu_gyr_noise = 0.0;        // continuous-time noise density
  double imu_acc_bias_rw = 0.0, imu_gyr_bias_rw = 0.0;    // bias random-walk

  // Sum of member Extrinsic::versions plus an epoch; any refinement bumps it.
  std::uint32_t version = 0;

  // Lookup by sensor frame; throws if no extrinsic for that frame is present.
  const Extrinsic& extrinsic(Frame sensor) const {
    for (const Extrinsic& e : extrinsics) {
      if (e.child == sensor) {
        return e;
      }
    }
    throw std::out_of_range("CalibrationSet::extrinsic: no extrinsic for frame");
  }
};

}  // namespace meridian
