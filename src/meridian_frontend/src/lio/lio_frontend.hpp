#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "lio/imu_tracker.hpp"
#include "lio/scan_registration.hpp"
#include "lio/voxel_grid_map.hpp"
#include "meridian/calib/calibration_set.hpp"
#include "meridian/common/cloud.hpp"
#include "meridian/common/frontend_diagnostics.hpp"
#include "meridian/common/measure_group.hpp"
#include "meridian/common/nav_state.hpp"
#include "meridian/common/observability.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/preprocessed_group.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"
#include "meridian/frontend/ifrontend.hpp"

namespace meridian {
class TelemetrySink;
}  // namespace meridian

namespace meridian::lio {

// Discrete LiDAR-inertial front-end. Each sweep is deskewed under a constant body
// screw, voxel-downsampled, and registered against the local voxel map by Gauss-Newton
// point-to-point alignment with an interval-averaged IMU motion prior; the solved pose
// updates the map and drives keyframe emission. Emitted keyframes carry frontend_kind 2.
//
// The estimation frame F_e is imu_link, tagged Frame::Body on emitted quantities; the
// world/odom frame is Frame::Odom. Internal 6-DoF tangents are translation-first
// [rho; phi]; the one rotation-first block is KeyframePacket::constraint_cov, reordered
// exactly once at pack time. Every path is synchronous on the caller's thread, so the
// `deterministic` flag changes nothing: two identical runs are bit-identical either way.
class LioFrontEnd final : public IFrontEnd {
public:
  LioFrontEnd(const FrontendConfig& cfg, std::shared_ptr<const CalibrationSet> calib,
              TelemetrySink* telemetry, bool deterministic = false);

  void set_calibration(std::shared_ptr<const CalibrationSet> calib) override;
  void ingest(const PreprocessedGroup& group) override;
  void ingest_imu_live(const ImuSample& imu) override;
  void apply_correction(const GraphUpdate& update) override;
  NavState live_state() const override;
  void set_keyframe_sink(KeyframeSink sink) override;
  FrontEndDiagnostics diagnostics() const override;

private:
  using Mat6 = Eigen::Matrix<double, 6, 6>;

  void adoptCalibration(const CalibrationSet& calib);
  // Rotates a raw imu_link sample into F_e. The lever arm is deliberately ignored: the
  // transport terms need differentiated rates, and with the IMU centimeters from F_e
  // the centripetal contribution sits below the accelerometer noise floor.
  ImuSample toBody(const ImuSample& s) const;

  void processGroup(const MeasureGroup& g);
  void handleFailure(const MeasureGroup& g, const NavState& guess,
                     const std::vector<Eigen::Vector3d>& deskewed, bool scan_usable);
  // Transform body-frame points to world by T_world_body and insert into the map.
  void insertWorld(const std::vector<Eigen::Vector3d>& points_body, const Pose& T_world_body);
  // Per-axis observability of the registered geometry at T_world_body (frame Body).
  ObservabilityReport observabilityAt(const std::vector<Eigen::Vector3d>& keypoints_body,
                                      const Pose& T_world_body) const;
  bool keyframeDue(const Pose& T_world_body, Timestamp stamp) const;
  void emitKeyframe(const NavState& state, PointCloudPtr cloud_body);
  // Re-anchor the live stream: drop consumed samples and re-propagate the rest from
  // the solved state, so live_state() runs ahead without influencing estimation.
  void rebaseLive(const NavState& solved);
  void propagateLive(const ImuSample& s);
  void publishSweepTelemetry(const MeasureGroup& g, const MotionPrior& prior,
                             const RegistrationResult& res, std::size_t n_keypoints);
  void publishBodyScan(const std::vector<Eigen::Vector3d>& deskewed, Timestamp t,
                       PointCloudPtr* cloud);
  void publishPathSample(const Pose& T_world_body, Timestamp t);

  FrontendConfig cfg_;
  std::shared_ptr<const CalibrationSet> calib_;
  TelemetrySink* telemetry_ = nullptr;
  // No-op by design: ingest() runs the whole deskew + solve synchronously on the
  // caller's thread with a fixed iteration schedule, so the live and replay paths are
  // already the same code path and bit-identical for identical input.
  bool deterministic_ = false;
  KeyframeSink keyframe_sink_;

  // calibration snapshot
  Eigen::Matrix3d R_body_imu_ = Eigen::Matrix3d::Identity();
  Pose T_body_lidar_;
  Pose T_body_cam_;
  std::uint32_t calib_version_ = 0;

  // estimation units
  ImuTracker tracker_;
  ScanRegistration registration_;
  VoxelGridMap map_;

  // groups held until static initialization fixes gravity and the at-rest biases
  std::deque<MeasureGroup> held_;
  bool held_drop_warned_ = false;

  // registered-state chain
  bool bootstrapped_ = false;
  bool have_prev_group_ = false;
  Timestamp prev_t_end_ = 0;
  NavState last_state_;
  // Armed by a data gap; the first registration outcome after it consumes the flag:
  // success keeps the map, failure clears it and re-anchors on the failing sweep.
  bool reseed_armed_ = false;
  // A reseed broke the relative chain; the next keyframe must be an absolute anchor.
  bool chain_broken_ = false;
  bool have_reseed_cov_ = false;
  Mat6 pending_prior_cov_ = Mat6::Identity();
  // Relative covariance accumulated since the last keyframe, right/body [rho; phi].
  Mat6 cov_rel_ = Mat6::Zero();
  ObservabilityReport last_obs_;

  // keyframe chain
  std::uint64_t next_kf_id_ = 1;
  bool have_prev_kf_ = false;
  std::uint64_t prev_kf_id_ = 0;
  Pose prev_kf_pose_;
  Timestamp prev_kf_stamp_ = 0;
  std::shared_ptr<const CameraFrame> last_image_;

  // live-state stream (estimation never reads these)
  std::deque<ImuSample> live_imu_;
  NavState live_state_;

  // telemetry / diagnostics
  Timestamp last_path_t_ = 0;
  bool restart_pending_ = false;
  FrontEndDiagnostics diag_;
};

}  // namespace meridian::lio
