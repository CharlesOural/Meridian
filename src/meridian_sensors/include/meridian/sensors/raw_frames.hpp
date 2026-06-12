#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

// Wire-free intermediate frames produced by the transport wrapper and consumed by the
// core sources. The wrapper decodes the wire message into a POD; the core turns the
// POD into a typed sample and stamps it. The POD carries the raw device clock value so
// the core can place the sample on the Meridian timeline without any transport types.

// One LiDAR return as it leaves the device, before typed conversion. `t` is the
// Ouster per-column nanosecond offset within the sweep and maps directly onto
// LidarPoint::t_offset_ns with no scaling.
struct RawPoint {
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;        // [m], sensor frame
  float intensity = 0.f;
  std::uint32_t t = 0;  // per-column offset from the sweep reference [ns]
  std::uint16_t ring = 0;
  std::uint16_t ambient = 0;
  float range_m = 0.f;
};

// One LiDAR sweep as a wire-free frame. `device_ns_first_column` is the sweep
// reference instant on the LiDAR's PTP-disciplined clock; `pts` is a borrowed view of
// the wrapper's decoded points, valid only for the duration of the ingest call.
struct RawLidarFrame {
  Timestamp host_arrival = 0;
  Timestamp device_ns_first_column = 0;
  bool has_device_ns = false;
  std::span<const RawPoint> pts;
};

// One IMU sample as a wire-free frame. No orientation is carried even if the device
// reports an AHRS quaternion; orientation is estimated downstream.
struct RawImuFrame {
  Timestamp device_ns = 0;
  bool has_device_ns = false;
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();   // specific force [m/s^2], raw
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();  // angular rate [rad/s], raw
  Timestamp host_arrival = 0;
};

// One camera frame as a wire-free frame. `data` is moved into the typed CameraFrame as
// shared-immutable. `exposure_s` is 0 when the driver does not report it.
struct RawCameraFrame {
  Timestamp device_ns = 0;
  bool has_device_ns = false;
  Timestamp host_arrival = 0;
  int width = 0;
  int height = 0;
  CameraFrame::Encoding encoding = CameraFrame::Encoding::Mono8;
  std::shared_ptr<const std::vector<std::uint8_t>> data;
  float exposure_s = 0.f;  // [s], 0 = unknown
  float gain = 1.f;
};

// One GNSS fix as a wire-free frame. `gps_second_ns` identifies the PPS edge the fix
// is referenced to (the start of the GPS second the fix belongs to, on the Meridian
// timeline); when no PPS is wired the fix degrades to the device/host time.
struct RawGnssFrame {
  Timestamp gps_second_ns = 0;
  bool has_pps_reference = false;
  Timestamp device_ns = 0;
  bool has_device_ns = false;
  Timestamp host_arrival = 0;
  double lat_deg = 0.0;
  double lon_deg = 0.0;
  double alt_m = 0.0;
  Eigen::Matrix3d cov_enu = Eigen::Matrix3d::Identity();  // [m^2], local ENU
  GnssFix::FixType fix = GnssFix::FixType::None;
  std::uint8_t num_sats = 0;
};

}  // namespace meridian
