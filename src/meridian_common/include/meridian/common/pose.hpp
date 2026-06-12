#pragma once

#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace meridian {

// A rigid-body transform T_target_source in SE(3): p_target = R*p_source + t.
// Stored as a unit quaternion + translation. The 6-DoF tangent is translation-first,
// xi = [rho (trans); phi (rot)], with the right (body-frame) perturbation convention:
//   this boxplus xi == this * Exp(xi)
//   a    boxminus b == Log(b^{-1} * a)
// This matches Sophus::SE3 exp/log, whose tangent is also [upsilon; omega].
struct Pose {
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();  // rotation, unit
  Eigen::Vector3d t = Eigen::Vector3d::Zero();            // translation [m]

  Pose() = default;
  Pose(const Eigen::Quaterniond& q_in, const Eigen::Vector3d& t_in)
      : q(q_in.normalized()), t(t_in) {}

  Pose operator*(const Pose& rhs) const {
    return Pose{(q * rhs.q).normalized(), q * rhs.t + t};
  }
  Eigen::Vector3d operator*(const Eigen::Vector3d& p) const { return q * p + t; }

  Pose inverse() const {
    const Eigen::Quaterniond qi = q.conjugate();
    return Pose{qi, qi * (-t)};
  }

  Pose boxplus(const Eigen::Matrix<double, 6, 1>& xi) const {
    const Sophus::SE3d r = Sophus::SE3d(q, t) * Sophus::SE3d::exp(xi);
    return Pose{r.unit_quaternion(), r.translation()};
  }
  Eigen::Matrix<double, 6, 1> boxminus(const Pose& rhs) const {
    return (Sophus::SE3d(rhs.q, rhs.t).inverse() * Sophus::SE3d(q, t)).log();
  }

  Eigen::Matrix3d R() const { return q.toRotationMatrix(); }
  Eigen::Matrix4d matrix() const {
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    m.topLeftCorner<3, 3>() = q.toRotationMatrix();
    m.topRightCorner<3, 1>() = t;
    return m;
  }
};

}  // namespace meridian
