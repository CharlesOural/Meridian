#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <Eigen/Core>

#include "meridian/common/cloud.hpp"
#include "meridian/common/frame.hpp"
#include "meridian/common/gaussian.hpp"
#include "meridian/common/imu_preintegration.hpp"
#include "meridian/common/observability.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

// The single value type crossing L2 -> L3. Exactly one geometric constraint per keyframe
// interval, selected by constraint_kind (mutually exclusive by construction, so the
// information budget is never double-counted).
struct KeyframePacket {
  enum class ConstraintKind {
    RelativeBetween,    // default: constraint_cov is the marginal cov of T_relto_this
    AbsolutePrior,      // first KF / GNSS-anchored: marginal cov of T_ref_body in ref_frame
    ImuPreintegration,  // restart fallback only: pairs with imu_summary, no between-factor
  };

  // identity / time
  std::uint64_t id = 0;  // monotonic, front-end assigned
  Timestamp stamp = 0;   // a real measurement instant

  // pose (required)
  Frame ref_frame = Frame::Odom;
  Pose T_ref_body;

  // kinematic state (optional, gated; true only on the restart fallback)
  bool kinematics_included = false;
  Eigen::Vector3d v_ref = Eigen::Vector3d::Zero();  // [m/s] in ref_frame, valid iff flag
  Eigen::Vector3d b_g = Eigen::Vector3d::Zero();    // gyro bias  [rad/s], valid iff flag
  Eigen::Vector3d b_a = Eigen::Vector3d::Zero();    // accel bias [m/s^2], valid iff flag

  // uncertainty: ONE block, meaning set by constraint_kind
  ConstraintKind constraint_kind = ConstraintKind::RelativeBetween;
  std::uint64_t rel_to_id = 0;  // previous keyframe id (RelativeBetween/ImuPreintegration)
  Pose T_relto_this;            // the relative transform (RelativeBetween)
  // ROTATION-FIRST [rx,ry,rz,tx,ty,tz] — the one rotation-first block in the system,
  // matching the GTSAM Pose3 boundary; the front-end's translation-first marginal is
  // reordered exactly once when packing this.
  GaussianBlock<6> constraint_cov;

  // observability: 6 per-axis scores in a named frame, translation-first
  ObservabilityReport observability;

  // data handles, shared-immutable, no copy
  PointCloudPtr cloud_body;                  // deskewed, body frame at stamp
  std::shared_ptr<const CameraFrame> image;  // may be null if no camera at this KF
  Pose T_body_cam;                           // extrinsic snapshot for colourisation

  // restart-fallback IMU summary (present iff constraint_kind == ImuPreintegration)
  std::optional<ImuPreintegrationSummary> imu_summary;

  // provenance
  std::uint32_t calib_version = 0;  // CalibrationSet snapshot that produced this
  std::uint32_t frontend_kind = 1;  // 1 = CT, the only producer; diagnostics only
};

}  // namespace meridian
