#pragma once

#include <gtsam/inference/Symbol.h>

#include <cstdint>

#include "meridian/common/frame.hpp"

namespace meridian::backend {

// Key namespaces of the global graph: x = keyframe pose, v = velocity, b = IMU bias,
// e = extrinsic of a sensor frame, g = the single global datum node.
inline gtsam::Key keyX(std::uint64_t id) {
  return gtsam::Symbol('x', id);
}
inline gtsam::Key keyV(std::uint64_t id) {
  return gtsam::Symbol('v', id);
}
inline gtsam::Key keyB(std::uint64_t id) {
  return gtsam::Symbol('b', id);
}
inline gtsam::Key keyE(Frame s) {
  return gtsam::Symbol('e', static_cast<std::uint64_t>(s));
}
inline const gtsam::Key kKeyG = gtsam::Symbol('g', 0);

}  // namespace meridian::backend
