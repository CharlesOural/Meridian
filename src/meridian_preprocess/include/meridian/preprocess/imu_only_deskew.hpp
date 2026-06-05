#pragma once

#include <vector>

#include <Eigen/Core>

#include "meridian/calib/extrinsic.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/preprocess/ideskew_provider.hpp"
#include "meridian/preprocess/imu_init.hpp"

namespace meridian {

// Cold-start / restart IMU-only forward integration. From the static init state
// (gravity direction, gyro bias) and a buffer of IMU samples, integrates a
// piecewise-constant-omega / piecewise-constant-acceleration trajectory and exposes it
// as an IDeskewProvider. The trajectory is gravity-aligned and uses a constant bias.
//
// Thread-confined: built and queried from a single stage thread.
class ImuOnlyDeskew : public IDeskewProvider {
 public:
  // T_imu_lidar is the LiDAR->IMU extrinsic used by the per-point warp. The provider
  // is seeded from the init state (gravity + gyro bias) at the start pose given by
  // start_pose, with start_velocity in the reference (body-at-anchor) frame's world.
  ImuOnlyDeskew(const ImuInitState& init, const Extrinsic& T_imu_lidar,
                const Pose& start_pose, const Eigen::Vector3d& start_velocity);

  // Feeds one IMU interval. Samples must arrive in non-decreasing stamp order; each
  // sample defines the head of the interval that runs until the next sample. The first
  // sample seeds the integration anchor at its own stamp.
  void pushImu(const ImuSample& s);

  // Sets which time the warp compensates TO; defaults to the last integrated stamp.
  void setAnchor(Timestamp anchor) { anchor_ = anchor; }

  // IDeskewProvider:
  bool poseAt(Timestamp t, Pose* T_ref_Fe) const override;
  Timestamp anchor() const override { return anchor_; }
  bool validHorizonCovers(Timestamp t_begin, Timestamp t_end) const override;
  Kind kind() const override { return Kind::ImuOnly; }

  // Deskews a whole scan against this provider's trajectory, producing a NEW scan whose
  // points are expressed in the body frame at the anchor (scan-end by default).
  // Copy-on-write: the input buffer is never mutated. Returns false (out untouched) if
  // the provider's horizon does not cover the sweep, signalling a window restart.
  bool deskew(const LidarScan& in, LidarScan* out) const;

 private:
  // One integrated IMU interval head: world pose, world velocity, and the
  // interval-constant body rate / world acceleration used to advance within it.
  struct IntervalState {
    Timestamp stamp = 0;
    Pose pose;                                       // T_ref_imu at interval head
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();   // world velocity [m/s]
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();  // body rate [rad/s], bias-corrected
    Eigen::Vector3d acc_world = Eigen::Vector3d::Zero();  // world accel [m/s^2], gravity-removed
  };

  // Evaluates the IMU pose T_ref_imu at time t by advancing within the interval whose
  // head precedes t (constant omega / constant world accel). Returns false if t is
  // outside the integrated horizon.
  bool imuPoseAt(Timestamp t, Pose* T_ref_imu) const;

  Eigen::Vector3d gravity_;     // world gravity [m/s^2]
  Eigen::Vector3d gyro_bias_;   // [rad/s]
  Eigen::Vector3d accel_bias_;  // [m/s^2]
  Pose T_imu_lidar_;            // LiDAR -> IMU extrinsic
  std::vector<IntervalState> intervals_;
  bool seeded_ = false;  // true once the first IMU sample latched the anchor stamp
  Timestamp anchor_ = 0;
};

}  // namespace meridian
