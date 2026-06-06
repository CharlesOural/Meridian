#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

#include "ct/image_pyramid_view.hpp"
#include "ct/visual_map.hpp"
#include "meridian/calib/calibration_set.hpp"
#include "meridian/calib/camera_model.hpp"
#include "meridian/common/frontend_diagnostics.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/nav_state.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/preprocessed_group.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"
#include "meridian/frontend/ifrontend.hpp"
#include "meridian/preprocess/imu_init.hpp"

template <typename PointType>
class KD_TREE;
struct ikdTree_PointType;

namespace meridian {

class TelemetrySink;

// Iterated error-state Kalman filter front-end running a LiDAR-inertial pipeline:
// predict -> deskew -> iterated point-to-plane update against an incremental
// KD-tree map. Emitted keyframes carry frontend_kind 0.
//
// State is the Meridian 18-DoF NavState error order [p|R|v|bg|ba|g],
// translation-first. The LiDAR<->IMU extrinsic is NOT a filter variable; it is
// read from the calibration snapshot and held fixed. Gravity is stored as a full
// 3-vector but its updates are projected onto the tangent plane of the sphere of
// fixed magnitude |g|, so the magnitude does not drift (a simple two-axis
// projection in place of the full S2 chart machinery).
class IekfFrontEnd : public IFrontEnd {
public:
  static constexpr int kDof = NavState::kDof;  // 18

  explicit IekfFrontEnd(const FrontendConfig& cfg, TelemetrySink* telemetry);
  ~IekfFrontEnd() override;

  void set_calibration(std::shared_ptr<const CalibrationSet> calib) override;
  void ingest(const PreprocessedGroup& group) override;
  void ingest_imu_live(const ImuSample& imu) override;
  void apply_correction(const GraphUpdate& update) override;
  NavState live_state() const override;
  void set_keyframe_sink(KeyframeSink sink) override;
  FrontEndDiagnostics diagnostics() const override;

  // Reorders a 6x6 marginal between the internal translation-first [rho;phi]
  // convention and the rotation-first [rx,ry,rz,tx,ty,tz] boundary convention.
  // Self-inverse: applying it twice returns the original block.
  static Eigen::Matrix<double, 6, 6> reorderTransRotToRotTrans(
      const Eigen::Matrix<double, 6, 6>& trans_first);

  // Decoupled NavState retraction: position is world-additive (p += rho), the
  // orientation is right-multiplied (q <- q * Exp(phi)), and the remaining blocks
  // are additive. This keeps position and orientation independent so the analytic
  // Jacobians (d p_dot / d v = I, measurement position block n^T) stay exact; it
  // deliberately does not use the coupled SE(3) exponential of Pose::boxplus.
  static NavState boxplusNav(const NavState& s, const Eigen::Matrix<double, kDof, 1>& dx);

  // Local difference matching boxplusNav: the position block is the plain world
  // difference and the orientation block is Log(rhs.q^-1 * q).
  static Eigen::Matrix<double, kDof, 1> boxminusNav(const NavState& a, const NavState& b);

private:
  using Cov = Eigen::Matrix<double, kDof, kDof>;
  using Tangent = Eigen::Matrix<double, kDof, 1>;

  // One propagated waypoint: the filter state frozen at an IMU sample time, plus
  // the constants used to interpolate within the following interval.
  struct Waypoint {
    Timestamp stamp = 0;
    Pose T_world_imu;
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();      // body rate, bias-removed
    Eigen::Vector3d acc_world = Eigen::Vector3d::Zero();  // world accel incl. gravity
  };

  // Advance the filter mean + covariance across one IMU interval of length dt[s]
  // using the midpoint acc/gyro input. Mutates state_ and P_.
  void predict(const Eigen::Vector3d& acc_mid, const Eigen::Vector3d& gyro_mid, double dt);

  // Propagate a private copy of the state across one interval; used by the live
  // path so live_state() can advance without disturbing the filter.
  static void predictMean(NavState& s, const Eigen::Vector3d& acc_mid,
                          const Eigen::Vector3d& gyro_mid, double dt);

  // Build the per-sample waypoint trail across a group's IMU and run predict on
  // each interval. Returns the waypoints (scan-end is the last entry).
  std::vector<Waypoint> propagateGroup(const std::vector<ImuSample>& imu, Timestamp t_scan_end);

  // Deskew a raw sweep to the scan-end body frame using the waypoint trail and
  // the fixed LiDAR extrinsic, then voxel-downsample. Output points are in the
  // IMU/body frame at scan-end.
  std::vector<Eigen::Vector3d> deskewAndDownsample(const LidarScan& scan,
                                                   const std::vector<Waypoint>& trail,
                                                   const Pose& T_world_end) const;

  // The iterated point-to-plane update against the local map. Returns the number
  // of effective (plane-associated) points used by the final iteration.
  int iteratedUpdate(const std::vector<Eigen::Vector3d>& body_pts);

  // Sequential FAST-LIVO2-style photometric update, run after the point-to-plane
  // update (the FAST-LIVO2 sequential order). A simplified single-level patch update:
  // each visible visual point's reference patch is compared against the current image
  // (inverse-exposure affine model, image-gradient linearization), accumulated in the
  // 18-DoF information form against the frozen post-LiDAR prior, and one MAP step is
  // taken. Returns the number of effective (gated) photometric points. No-op when the
  // camera model is invalid, the visual map is empty, or no image is present.
  int photometricUpdate(const ct::ImagePyramidView& img);

  // Rebuild the camera model + camera extrinsic + visual map from the calibration
  // snapshot. Leaves the model invalid (visual stage off) when no usable intrinsics
  // are present.
  void refreshCamera();

  // Decode a raw CameraFrame to a single-channel intensity image and build the
  // level pyramid the photometric update samples. Empty when undecodable.
  std::vector<cv::Mat> buildImagePyramid(const CameraFrame& frame) const;

  // Insert the registered (world-frame) points into the incremental map and run
  // its downsample maintenance.
  void updateMap(const std::vector<Eigen::Vector3d>& body_pts);

  // Right-perturbation projection of a 3-vector gravity increment onto the
  // tangent plane of |g| = const, expressed back as a 3-vector.
  Eigen::Vector3d projectGravityIncrement(const Eigen::Vector3d& dg_raw) const;

  // boxplusNav with the gravity block constrained to the |g| sphere; returns the
  // retracted state.
  static NavState boxplusConstrained(const NavState& s, const Tangent& dx);

  // Per-axis observability of the last solve from the pose information block
  // last_info_pose_: each eigenpair's conditioning score is assigned to the axis
  // its eigenvector most aligns with, kept translation-first [tx,ty,tz,rx,ry,rz];
  // degenerate solves additionally export the eigenvector basis.
  ObservabilityReport computeObservability() const;

  // Decide whether the current scan-end pose/time warrants a new keyframe.
  bool keyframeDue(const Pose& T_world_end, Timestamp stamp) const;

  // Pack and emit a keyframe for the just-solved scan.
  void emitKeyframe(const Pose& T_world_end, Timestamp stamp,
                    const std::vector<Eigen::Vector3d>& body_pts);

  // Lazily fetch the LiDAR->IMU extrinsic (T_imu_lidar) from the calibration set.
  void refreshExtrinsic();

  FrontendConfig cfg_;
  TelemetrySink* telemetry_ = nullptr;

  std::shared_ptr<const CalibrationSet> calib_;
  std::uint32_t calib_version_ = 0;
  Pose T_imu_lidar_;  // LiDAR sensor -> IMU/body
  bool have_extrinsic_ = false;

  // ---- camera / visual stage (offline oracle photometric cross-check) ----
  CameraModel cam_model_;  // invalid until valid intrinsics are configured
  Pose T_imu_cam_;         // camera optical frame -> IMU/body
  bool have_cam_extrinsic_ = false;
  double inv_expo_ = 1.0;  // carried inverse exposure (held fixed in the oracle update)
  std::unique_ptr<ct::VisualMap> vmap_;

  NavState state_;  // odom-frame filter state at last valid time
  Cov P_ = Cov::Identity();
  bool filter_initialized_ = false;
  Timestamp last_stamp_ = 0;

  // Shared static-start initializer: Welford mean/variance + motion gate over the
  // first IMU samples, recovering gravity direction and gyro bias. Lazily built once
  // the first group's IMU rate is known so init_time_s maps to a sample count.
  std::unique_ptr<ImuInitializer> imu_init_;

  // Live-output state advanced by ingest_imu_live without touching the filter.
  NavState live_state_;
  Timestamp live_stamp_ = 0;

  std::shared_ptr<KD_TREE<ikdTree_PointType>> map_;
  // World-frame cubic local-map bounds [min_xyz, max_xyz] and whether they are seeded.
  // The ikd-Tree is segmented to this cube around the body each sweep so map RAM and
  // nearest-neighbour search depth stay bounded over a long mission.
  std::array<double, 6> map_cube_{};
  bool map_cube_init_ = false;

  KeyframeSink keyframe_sink_;
  std::uint64_t next_kf_id_ = 0;
  bool have_prev_kf_ = false;
  std::uint64_t prev_kf_id_ = 0;
  Pose prev_kf_pose_;                  // T_world_body of the previous keyframe
  Timestamp prev_kf_stamp_ = 0;        // stamp of the previous keyframe
  Cov prev_kf_cov_ = Cov::Identity();  // filter cov captured at the previous keyframe

  // Last-solve Jacobian pose information block H^T R^-1 H (translation-first),
  // retained for observability scoring.
  Eigen::Matrix<double, 6, 6> last_info_pose_ = Eigen::Matrix<double, 6, 6>::Zero();

  FrontEndDiagnostics diag_;
};

}  // namespace meridian
