#include "meridian/preprocess/imu_only_deskew.hpp"

#include <algorithm>
#include <memory>

#include <sophus/so3.hpp>

#include "meridian/common/cloud.hpp"
#include "meridian/common/point.hpp"

namespace meridian {

ImuOnlyDeskew::ImuOnlyDeskew(const ImuInitState& init, const Extrinsic& T_imu_lidar,
                             const Pose& start_pose, const Eigen::Vector3d& start_velocity)
    : gravity_(init.gravity),
      gyro_bias_(init.gyro_bias),
      accel_bias_(init.accel_bias),
      T_imu_lidar_(T_imu_lidar.T_parent_child) {
  // Seed the trajectory with a single open interval at the start pose; pushImu() fills
  // in the per-interval constants and advances the head as samples arrive.
  IntervalState s0;
  s0.pose = start_pose;
  s0.vel = start_velocity;
  intervals_.push_back(s0);
}

void ImuOnlyDeskew::pushImu(const ImuSample& s) {
  const Eigen::Vector3d omega = s.gyro - gyro_bias_;
  // World-frame specific force minus gravity gives the kinematic world acceleration.
  // The bias is constant and the accel bias is the zero prior from static init.
  const Eigen::Vector3d acc_body = s.acc - accel_bias_;

  if (!seeded_) {
    // First sample seeds the anchor stamp and the interval-head constants.
    IntervalState& head = intervals_.back();
    head.stamp = s.stamp;
    head.omega = omega;
    head.acc_world = head.pose.q * acc_body + gravity_;
    anchor_ = s.stamp;
    seeded_ = true;
    return;
  }

  // Advance the previous head to this sample's stamp, then open a new interval here.
  IntervalState& prev = intervals_.back();
  const double dt = to_seconds(s.stamp - prev.stamp);
  if (dt <= 0.0) {
    return;  // ignore out-of-order / duplicate stamps
  }

  IntervalState next;
  next.stamp = s.stamp;
  next.pose.q = (prev.pose.q * Sophus::SO3d::exp(prev.omega * dt).unit_quaternion())
                    .normalized();
  next.pose.t = prev.pose.t + prev.vel * dt + 0.5 * prev.acc_world * dt * dt;
  next.vel = prev.vel + prev.acc_world * dt;
  next.omega = omega;
  next.acc_world = next.pose.q * acc_body + gravity_;

  intervals_.push_back(next);
  anchor_ = s.stamp;
}

bool ImuOnlyDeskew::imuPoseAt(Timestamp t, Pose* T_ref_imu) const {
  if (!seeded_) {
    return false;
  }
  const Timestamp t0 = intervals_.front().stamp;
  const Timestamp tn = intervals_.back().stamp;
  if (t < t0 || t > tn) {
    return false;
  }

  // Find the interval whose head precedes t (the last head with stamp <= t).
  std::size_t idx = 0;
  for (std::size_t i = 0; i < intervals_.size(); ++i) {
    if (intervals_[i].stamp <= t) {
      idx = i;
    } else {
      break;
    }
  }
  const IntervalState& h = intervals_[idx];
  const double dt = to_seconds(t - h.stamp);
  // Advance within the interval with constant body rate and constant world accel.
  T_ref_imu->q =
      (h.pose.q * Sophus::SO3d::exp(h.omega * dt).unit_quaternion()).normalized();
  T_ref_imu->t = h.pose.t + h.vel * dt + 0.5 * h.acc_world * dt * dt;
  return true;
}

bool ImuOnlyDeskew::poseAt(Timestamp t, Pose* T_ref_Fe) const {
  // F_e is the IMU/body frame in the current rig, so the IMU pose is T_ref_Fe directly.
  return imuPoseAt(t, T_ref_Fe);
}

bool ImuOnlyDeskew::validHorizonCovers(Timestamp t_begin, Timestamp t_end) const {
  if (!seeded_) {
    return false;
  }
  return t_begin >= intervals_.front().stamp && t_end <= intervals_.back().stamp;
}

bool ImuOnlyDeskew::deskew(const LidarScan& in, LidarScan* out) const {
  const Timestamp t_begin = in.stamp_start;
  const Timestamp t_end = in.stamp_start + in.sweep_duration;
  if (!validHorizonCovers(t_begin, t_end)) {
    return false;
  }

  Pose T_ref_end;
  if (!poseAt(anchor_, &T_ref_end)) {
    return false;
  }
  const Pose T_ref_end_inv = T_ref_end.inverse();
  const Pose T_imu_lidar = T_imu_lidar_;  // LiDAR -> IMU

  const PointCloud* src = in.points.get();
  if (src == nullptr) {
    *out = in;
    return true;
  }

  // Copy-on-write: build a fresh buffer; never touch the input bytes.
  PointCloud warped = *src;
  for (LidarPoint& p : warped) {
    const Timestamp t_i = in.stamp_start + p.t_offset_ns;
    Pose T_ref_i;
    if (!poseAt(t_i, &T_ref_i)) {
      return false;
    }
    // LiDAR@t_i -> IMU@t_i -> ref -> IMU/body@anchor; the output stays in the body frame.
    const Eigen::Vector3d p_imu = T_imu_lidar * p.xyz.cast<double>();
    const Eigen::Vector3d p_ref = T_ref_i * p_imu;
    const Eigen::Vector3d p_end_imu = T_ref_end_inv * p_ref;
    p.xyz = p_end_imu.cast<float>();
  }

  *out = in;
  out->points = std::make_shared<const PointCloud>(std::move(warped));
  return true;
}

}  // namespace meridian
