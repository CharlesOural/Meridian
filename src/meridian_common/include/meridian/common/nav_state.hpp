#pragma once

#include <Eigen/Core>

#include "meridian/common/frame.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

// The full navigation state the front-end estimates. The 18-DoF error state is ordered
// [ p(3) | R(3) | v(3) | b_g(3) | b_a(3) | g(3) ] (translation/position-first); its first
// six dims are the Pose tangent [rho; phi]. Gravity is exposed as a 3-DoF tangent for a
// uniform interface; the front-end impl restricts updates to the S2 tangent plane.
struct NavState {
  Timestamp stamp = 0;
  Pose T_world_body;                                       // pose of F_e in ref_frame
  Eigen::Vector3d v_world = Eigen::Vector3d::Zero();       // [m/s] in ref_frame
  Eigen::Vector3d b_g = Eigen::Vector3d::Zero();           // gyro bias  [rad/s]
  Eigen::Vector3d b_a = Eigen::Vector3d::Zero();           // accel bias [m/s^2]
  Eigen::Vector3d g_world = Eigen::Vector3d(0, 0, -9.81);  // gravity [m/s^2] in ref_frame
  Frame ref_frame = Frame::Odom;

  static constexpr int kDof = 18;

  NavState boxplus(const Eigen::Matrix<double, kDof, 1>& dx) const {
    NavState out = *this;
    out.T_world_body = T_world_body.boxplus(dx.head<6>());  // [p(3)|R(3)] == [rho;phi]
    out.v_world = v_world + dx.segment<3>(6);
    out.b_g = b_g + dx.segment<3>(9);
    out.b_a = b_a + dx.segment<3>(12);
    out.g_world = g_world + dx.segment<3>(15);
    return out;
  }

  Eigen::Matrix<double, kDof, 1> boxminus(const NavState& rhs) const {
    Eigen::Matrix<double, kDof, 1> dx;
    dx.head<6>() = T_world_body.boxminus(rhs.T_world_body);
    dx.segment<3>(6) = v_world - rhs.v_world;
    dx.segment<3>(9) = b_g - rhs.b_g;
    dx.segment<3>(12) = b_a - rhs.b_a;
    dx.segment<3>(15) = g_world - rhs.g_world;
    return dx;
  }
};

}  // namespace meridian
