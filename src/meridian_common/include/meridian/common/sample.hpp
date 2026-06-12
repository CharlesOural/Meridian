#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "meridian/common/cloud.hpp"
#include "meridian/common/frame.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

// A single IMU measurement, raw (bias + gravity included). No orientation field: a
// vendor AHRS is never trusted; orientation is estimated downstream.
struct ImuSample {
  Timestamp stamp = 0;                             // PTP-disciplined valid time [ns]
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();   // specific force [m/s^2], imu_link, raw
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();  // angular rate  [rad/s], imu_link, raw
  std::uint8_t sensor_id = 0;
};

// One LiDAR sweep. The struct is a value; `points` is shared-immutable.
struct LidarScan {
  Timestamp stamp_start = 0;             // valid time of the first point [ns]
  Duration sweep_duration = 0;           // span first..last point [ns]
  std::uint8_t sensor_id = 0;            // single LiDAR -> always 0
  Frame sensor_frame = Frame::Unknown;   // os_sensor0, for extrinsic lookup
  PointCloudPtr points;                  // shared-immutable
};

// One camera frame. `stamp` is the mid-exposure instant (the time T(t) is queried at).
// exposure_s and gain feed photometric normalization for the direct visual track.
struct CameraFrame {
  Timestamp stamp = 0;
  std::uint8_t sensor_id = 0;
  Frame sensor_frame = Frame::CamLink;
  int width = 0;
  int height = 0;
  enum class Encoding { Mono8, Bayer_RGGB8, RGB8 } encoding = Encoding::Mono8;
  std::shared_ptr<const std::vector<std::uint8_t>> data;  // shared-immutable, row-major
  float exposure_s = 0.f;  // [s], 0 = unknown
  float gain = 1.f;        // analog/digital gain
};

// One GNSS fix. `fix` quality gates how strongly it is trusted downstream.
struct GnssFix {
  Timestamp stamp = 0;  // PPS-disciplined time [ns]
  std::uint8_t sensor_id = 0;
  Frame sensor_frame = Frame::GnssLink;
  double lat_deg = 0;  // WGS84
  double lon_deg = 0;
  double alt_m = 0;  // ellipsoidal height [m]
  Eigen::Matrix3d cov_enu = Eigen::Matrix3d::Identity();  // position cov in local ENU [m^2]
  enum class FixType { None, SPP, DGPS, RTK_Float, RTK_Fixed } fix = FixType::None;
  std::uint8_t num_sats = 0;
};

}  // namespace meridian
