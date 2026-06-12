#pragma once

#include <cstdint>

#include <Eigen/Core>

namespace meridian {

// One LiDAR return in the sensor frame, raw (NOT deskewed). t_offset_ns is the signed
// offset of this return from its scan's stamp_start; it is mandatory (a scan without
// per-point time is rejected upstream) and is stored explicitly, never hidden in a
// float curvature field.
struct LidarPoint {
  Eigen::Vector3f xyz = Eigen::Vector3f::Zero();  // [m] in the LiDAR sensor frame
  float intensity = 0.f;                          // calibrated intensity
  std::int32_t t_offset_ns = 0;                   // this return's time - scan.stamp_start [ns]
  std::uint16_t ring = 0;                         // laser/row id
  std::uint16_t ambient = 0;                      // ambient/NIR, 0 if N/A
  float range = 0.f;                              // [m] precomputed range, 0 if invalid
};

}  // namespace meridian
