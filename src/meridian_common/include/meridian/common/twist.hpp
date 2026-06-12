#pragma once

#include <Eigen/Core>

namespace meridian {

// Body-frame spatial velocity (optional companion to a Pose for continuous-time queries).
struct Twist {
  Eigen::Vector3d v_lin = Eigen::Vector3d::Zero();  // linear  velocity [m/s]
  Eigen::Vector3d v_ang = Eigen::Vector3d::Zero();  // angular velocity [rad/s]
};

}  // namespace meridian
