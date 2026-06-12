#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "meridian/common/gaussian.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

// Preintegrated IMU between two keyframe times, with first-order bias Jacobians so the
// back-end can re-evaluate at an updated bias. Crosses L2->L3 only on the window-restart
// fallback (KeyframePacket::ConstraintKind::ImuPreintegration).
struct ImuPreintegrationSummary {
  Timestamp t_i = 0;  // integration interval [ns]
  Timestamp t_j = 0;
  Eigen::Quaterniond delta_R = Eigen::Quaterniond::Identity();  // rotation increment
  Eigen::Vector3d delta_v = Eigen::Vector3d::Zero();            // [m/s]
  Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();            // [m]
  Eigen::Vector3d bias_g_lin = Eigen::Vector3d::Zero();         // bias linearization point
  Eigen::Vector3d bias_a_lin = Eigen::Vector3d::Zero();
  Eigen::Matrix3d dR_dbg = Eigen::Matrix3d::Zero();             // first-order bias Jacobians
  Eigen::Matrix3d dv_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dba = Eigen::Matrix3d::Zero();
  GaussianBlock<9> preint_cov;  // 9-DoF cov of [dR | dv | dp]
  double gravity_mag = 9.81;
};

}  // namespace meridian
