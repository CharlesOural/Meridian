#pragma once

#include <cstdint>

namespace meridian {

// Named coordinate frames. Every pose/covariance/observability report carries the
// frame it is expressed in, so frame mix-ups are a type-level error, not a silent bug.
enum class Frame : std::uint16_t {
  Unknown = 0,
  Map,
  Odom,
  BaseLink,
  ImuLink,
  CamLink,
  GnssLink,
  OsSensor0,  // the single LiDAR sensor frame
  Body,       // the estimation frame F_e (alias of ImuLink in the current rig)
};

}  // namespace meridian
