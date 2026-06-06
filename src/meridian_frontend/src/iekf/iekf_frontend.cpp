#include "iekf/iekf_frontend.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <sophus/so3.hpp>
#include <unordered_map>
#include <vector>

#include "ct/residuals_visual.hpp"
#include "ikd_Tree.h"
#include "meridian/common/cloud.hpp"
#include "meridian/common/point.hpp"
#include "meridian/debug/telemetry.hpp"

namespace meridian {
namespace {

constexpr double kGravityMag = 9.81;
constexpr double kSmallAngle = 1e-7;

// Decode a raw CameraFrame to a single-channel CV_8UC1 intensity image. Empty when the
// frame carries no usable bytes. No photometric scaling is applied.
cv::Mat decodeIntensity(const CameraFrame& frame) {
  const int rows = frame.height;
  const int cols = frame.width;
  const std::uint8_t* bytes = frame.data ? frame.data->data() : nullptr;
  if (bytes == nullptr || rows <= 0 || cols <= 0) {
    return cv::Mat();
  }
  switch (frame.encoding) {
    case CameraFrame::Encoding::Mono8: {
      const cv::Mat view(rows, cols, CV_8UC1, const_cast<std::uint8_t*>(bytes));
      return view.clone();
    }
    case CameraFrame::Encoding::Bayer_RGGB8: {
      const cv::Mat view(rows, cols, CV_8UC1, const_cast<std::uint8_t*>(bytes));
      cv::Mat rgb;
      cv::cvtColor(view, rgb, cv::COLOR_BayerRG2RGB);
      cv::Mat grey;
      cv::cvtColor(rgb, grey, cv::COLOR_RGB2GRAY);
      return grey;
    }
    case CameraFrame::Encoding::RGB8: {
      const cv::Mat view(rows, cols, CV_8UC3, const_cast<std::uint8_t*>(bytes));
      cv::Mat grey;
      cv::cvtColor(view, grey, cv::COLOR_RGB2GRAY);
      return grey;
    }
  }
  return cv::Mat();
}

// Tangent block offsets in the 18-DoF NavState error order [p|R|v|bg|ba|g].
constexpr int kP = 0;
constexpr int kR = 3;
constexpr int kV = 6;
constexpr int kBg = 9;
constexpr int kBa = 12;
constexpr int kG = 15;

// Process-noise tangent offsets in the 12-DoF order [n_g | n_a | n_bg | n_ba].
constexpr int kNg = 0;
constexpr int kNa = 3;
constexpr int kNbg = 6;
constexpr int kNba = 9;

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}

// Right Jacobian of SO(3): the factor that makes the on-manifold transition
// consistent. A(v) = I + ((1-cos t)/t^2) [v]x + ((1 - sin t/t)/t^2) [v]x^2.
Eigen::Matrix3d rightJacobian(const Eigen::Vector3d& v) {
  const double t = v.norm();
  if (t < kSmallAngle) {
    return Eigen::Matrix3d::Identity();
  }
  const Eigen::Matrix3d K = skew(v);
  const double t2 = t * t;
  return Eigen::Matrix3d::Identity() + ((1.0 - std::cos(t)) / t2) * K +
         ((1.0 - std::sin(t) / t) / t2) * (K * K);
}

ikdTree_PointType toIkd(const Eigen::Vector3d& p) {
  return ikdTree_PointType(static_cast<float>(p.x()), static_cast<float>(p.y()),
                           static_cast<float>(p.z()));
}

// Pack a [min_xyz, max_xyz] cube into an ikd-Tree axis-aligned box.
BoxPointType toBox(const std::array<double, 6>& cube) {
  BoxPointType box;
  for (int i = 0; i < 3; ++i) {
    box.vertex_min[i] = static_cast<float>(cube[i]);
    box.vertex_max[i] = static_cast<float>(cube[i + 3]);
  }
  return box;
}

// Fit a plane n^T p + d = 0 with ||n|| = 1 to the neighbour set; reject if any
// neighbour lies farther than `thresh` from the fitted plane. Returns false if
// the fit is rejected.
bool fitPlane(const std::vector<Eigen::Vector3d>& pts, double thresh, Eigen::Vector3d* n,
              double* d) {
  const int m = static_cast<int>(pts.size());
  if (m < 3) {
    return false;
  }
  // Solve A x = -1 with x = n/d, then normalise to recover ||n|| = 1 and d.
  Eigen::MatrixXd A(m, 3);
  Eigen::VectorXd b = -Eigen::VectorXd::Ones(m);
  for (int i = 0; i < m; ++i) {
    A.row(i) = pts[i].transpose();
  }
  const Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);
  const double inv = 1.0 / x.norm();
  *n = x * inv;
  *d = inv;
  for (int i = 0; i < m; ++i) {
    if (std::abs(n->dot(pts[i]) + *d) > thresh) {
      return false;
    }
  }
  return true;
}

}  // namespace

IekfFrontEnd::IekfFrontEnd(const FrontendConfig& cfg, TelemetrySink* telemetry)
    : cfg_(cfg), telemetry_(telemetry) {
  state_.g_world = Eigen::Vector3d(0, 0, -kGravityMag);
  state_.ref_frame = Frame::Odom;
  live_state_ = state_;
  const double v = std::max(cfg_.lidar.voxel_map_m, 1e-3);
  map_ = std::make_shared<KD_TREE<ikdTree_PointType>>(0.3f, 0.6f, static_cast<float>(v));
}

IekfFrontEnd::~IekfFrontEnd() = default;

void IekfFrontEnd::set_calibration(std::shared_ptr<const CalibrationSet> calib) {
  calib_ = std::move(calib);
  if (calib_) {
    calib_version_ = calib_->version;
  }
  have_extrinsic_ = false;
  refreshExtrinsic();
  refreshCamera();
}

void IekfFrontEnd::refreshCamera() {
  cam_model_ = CameraModel{};
  have_cam_extrinsic_ = false;
  if (!calib_) {
    return;
  }
  for (const Extrinsic& e : calib_->extrinsics) {
    if (e.child == Frame::CamLink) {
      T_imu_cam_ = e.T_parent_child;
      have_cam_extrinsic_ = true;
      break;
    }
  }
  if (!calib_->cam_intrinsics.empty()) {
    const IntrinsicsCamera& k = calib_->cam_intrinsics.begin()->second;
    cam_model_ = CameraModel(k);
    inv_expo_ = k.inv_expo_prior > 0.0 ? k.inv_expo_prior : 1.0;
  }
  vmap_ = std::make_unique<ct::VisualMap>(ct::VisualMapConfig(cfg_.visual));
}

std::vector<cv::Mat> IekfFrontEnd::buildImagePyramid(const CameraFrame& frame) const {
  const cv::Mat intensity = decodeIntensity(frame);
  std::vector<cv::Mat> pyr;
  if (intensity.empty()) {
    return pyr;
  }
  const int levels = std::max(cfg_.visual.levels, 1);
  pyr.reserve(static_cast<std::size_t>(levels));
  pyr.push_back(intensity);
  for (int l = 1; l < levels; ++l) {
    if (pyr.back().cols < 2 || pyr.back().rows < 2) {
      break;
    }
    cv::Mat down;
    cv::pyrDown(pyr.back(), down);
    pyr.push_back(down);
  }
  return pyr;
}

void IekfFrontEnd::refreshExtrinsic() {
  if (have_extrinsic_ || !calib_) {
    return;
  }
  // The LiDAR extrinsic is stored as the sensor-to-F_e transform; the deskew and
  // measurement model want T_imu_lidar directly. Fall back to identity if the
  // rig has no LiDAR extrinsic registered (single-sensor / unit tests).
  for (const Extrinsic& e : calib_->extrinsics) {
    if (e.child == Frame::OsSensor0) {
      T_imu_lidar_ = e.T_parent_child;
      have_extrinsic_ = true;
      return;
    }
  }
  T_imu_lidar_ = Pose{};
  have_extrinsic_ = true;
}

void IekfFrontEnd::set_keyframe_sink(KeyframeSink sink) {
  keyframe_sink_ = std::move(sink);
}

NavState IekfFrontEnd::live_state() const {
  return live_state_;
}

FrontEndDiagnostics IekfFrontEnd::diagnostics() const {
  return diag_;
}

Eigen::Matrix<double, 6, 6> IekfFrontEnd::reorderTransRotToRotTrans(
    const Eigen::Matrix<double, 6, 6>& trans_first) {
  // Swap the two 3-blocks: input is [rho(0:2); phi(3:5)], output is [phi; rho].
  // Build the permutation [3,4,5,0,1,2] and apply it symmetrically.
  Eigen::Matrix<double, 6, 6> perm = Eigen::Matrix<double, 6, 6>::Zero();
  for (int i = 0; i < 3; ++i) {
    perm(i, i + 3) = 1.0;  // rotation rows take the old rotation block
    perm(i + 3, i) = 1.0;  // translation rows take the old translation block
  }
  return perm * trans_first * perm.transpose();
}

Eigen::Vector3d IekfFrontEnd::projectGravityIncrement(const Eigen::Vector3d& dg_raw) const {
  // Remove the component of the increment along the current gravity direction so
  // the retracted gravity stays (to first order) on the |g| = const sphere.
  const double n2 = state_.g_world.squaredNorm();
  if (n2 < 1e-12) {
    return dg_raw;
  }
  const Eigen::Vector3d ghat = state_.g_world / std::sqrt(n2);
  return dg_raw - ghat * (ghat.dot(dg_raw));
}

NavState IekfFrontEnd::boxplusNav(const NavState& s, const Eigen::Matrix<double, kDof, 1>& dx) {
  NavState out = s;
  // Position is world-additive, orientation is a right perturbation; the two stay
  // decoupled so the filter Jacobians authored against this chart are exact.
  out.T_world_body.t = s.T_world_body.t + dx.segment<3>(kP);
  out.T_world_body.q =
      (s.T_world_body.q * Sophus::SO3d::exp(dx.segment<3>(kR)).unit_quaternion()).normalized();
  out.v_world = s.v_world + dx.segment<3>(kV);
  out.b_g = s.b_g + dx.segment<3>(kBg);
  out.b_a = s.b_a + dx.segment<3>(kBa);
  out.g_world = s.g_world + dx.segment<3>(kG);
  return out;
}

Eigen::Matrix<double, IekfFrontEnd::kDof, 1> IekfFrontEnd::boxminusNav(const NavState& a,
                                                                       const NavState& b) {
  Eigen::Matrix<double, kDof, 1> dx = Eigen::Matrix<double, kDof, 1>::Zero();
  dx.segment<3>(kP) = a.T_world_body.t - b.T_world_body.t;
  dx.segment<3>(kR) = Sophus::SO3d(b.T_world_body.q.conjugate() * a.T_world_body.q).log();
  dx.segment<3>(kV) = a.v_world - b.v_world;
  dx.segment<3>(kBg) = a.b_g - b.b_g;
  dx.segment<3>(kBa) = a.b_a - b.b_a;
  dx.segment<3>(kG) = a.g_world - b.g_world;
  return dx;
}

NavState IekfFrontEnd::boxplusConstrained(const NavState& s, const Tangent& dx) {
  // Retract on the decoupled chart, then renormalise gravity to its original
  // magnitude so the update lives on the fixed-|g| sphere.
  const double g_mag = s.g_world.norm();
  NavState out = boxplusNav(s, dx);
  if (g_mag > 1e-9 && out.g_world.norm() > 1e-9) {
    out.g_world = out.g_world.normalized() * g_mag;
  }
  return out;
}

void IekfFrontEnd::predictMean(NavState& s, const Eigen::Vector3d& acc_mid,
                               const Eigen::Vector3d& gyro_mid, double dt) {
  const Eigen::Vector3d omega = gyro_mid - s.b_g;
  const Eigen::Matrix3d R = s.T_world_body.R();
  const Eigen::Vector3d acc_world = R * (acc_mid - s.b_a) + s.g_world;

  // Right-invariant SO(3) integration with constant-acceleration translation.
  s.T_world_body.t += s.v_world * dt + 0.5 * acc_world * dt * dt;
  s.T_world_body.q =
      (s.T_world_body.q * Sophus::SO3d::exp(omega * dt).unit_quaternion()).normalized();
  s.v_world += acc_world * dt;
  // biases, gravity unchanged by the mean propagation.
}

void IekfFrontEnd::predict(const Eigen::Vector3d& acc_mid, const Eigen::Vector3d& gyro_mid,
                           double dt) {
  MERIDIAN_SCOPED_TIME(telemetry_, "frontend.iekf.predict", last_stamp_);
  if (dt <= 0.0) {
    return;
  }
  const Eigen::Matrix3d R = state_.T_world_body.R();
  const Eigen::Vector3d omega = gyro_mid - state_.b_g;
  const Eigen::Vector3d acc_corr = acc_mid - state_.b_a;

  // Continuous error-state Jacobian df_dx in [p|R|v|bg|ba|g] order.
  Cov Fx = Cov::Zero();
  Fx.block<3, 3>(kP, kV) = Eigen::Matrix3d::Identity();    // d(p_dot)/d(v)
  Fx.block<3, 3>(kR, kBg) = -Eigen::Matrix3d::Identity();  // d(theta_dot)/d(bg)
  Fx.block<3, 3>(kV, kR) = -R * skew(acc_corr);            // d(v_dot)/d(theta)
  Fx.block<3, 3>(kV, kBa) = -R;                            // d(v_dot)/d(ba)
  Fx.block<3, 3>(kV, kG) = Eigen::Matrix3d::Identity();    // d(v_dot)/d(g)

  // Continuous noise Jacobian df_dw, noise order [n_g | n_a | n_bg | n_ba].
  Eigen::Matrix<double, kDof, 12> Fw = Eigen::Matrix<double, kDof, 12>::Zero();
  Fw.block<3, 3>(kR, kNg) = -Eigen::Matrix3d::Identity();   // theta_dot wrt n_g
  Fw.block<3, 3>(kV, kNa) = -R;                             // v_dot wrt n_a
  Fw.block<3, 3>(kBg, kNbg) = Eigen::Matrix3d::Identity();  // bg_dot wrt n_bg
  Fw.block<3, 3>(kBa, kNba) = Eigen::Matrix3d::Identity();  // ba_dot wrt n_ba

  // Discrete transition: identity base, the rotation block carries the inverse
  // incremental rotation Exp(-omega dt), and the analytic part is multiplied by
  // the SO(3) right Jacobian for the rows that integrate through a rotation.
  Cov F = Cov::Identity();
  const Eigen::Vector3d seg = omega * dt;
  const Eigen::Matrix3d A = rightJacobian(seg);
  F.block<3, 3>(kR, kR) = Sophus::SO3d::exp(-seg).matrix();
  // Apply A only to the rotation-row coupling (d theta_dot / d bg).
  Fx.block<3, 3>(kR, kBg) = A * Fx.block<3, 3>(kR, kBg);
  Fw.block<3, 3>(kR, kNg) = A * Fw.block<3, 3>(kR, kNg);
  F += Fx * dt;

  // Process-noise covariance from the calibration continuous-time densities.
  Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Zero();
  double ng = 1e-4, na = 1e-4, nbg = 1e-5, nba = 1e-5;
  if (calib_) {
    if (calib_->imu_gyr_noise > 0) ng = calib_->imu_gyr_noise * calib_->imu_gyr_noise;
    if (calib_->imu_acc_noise > 0) na = calib_->imu_acc_noise * calib_->imu_acc_noise;
    if (calib_->imu_gyr_bias_rw > 0) nbg = calib_->imu_gyr_bias_rw * calib_->imu_gyr_bias_rw;
    if (calib_->imu_acc_bias_rw > 0) nba = calib_->imu_acc_bias_rw * calib_->imu_acc_bias_rw;
  }
  Q.block<3, 3>(kNg, kNg) = ng * Eigen::Matrix3d::Identity();
  Q.block<3, 3>(kNa, kNa) = na * Eigen::Matrix3d::Identity();
  Q.block<3, 3>(kNbg, kNbg) = nbg * Eigen::Matrix3d::Identity();
  Q.block<3, 3>(kNba, kNba) = nba * Eigen::Matrix3d::Identity();

  // Advance the mean, then the covariance with the discrete transition.
  predictMean(state_, acc_mid, gyro_mid, dt);
  const Eigen::Matrix<double, kDof, 12> Fwdt = Fw * dt;
  P_ = F * P_ * F.transpose() + Fwdt * Q * Fwdt.transpose();
  P_ = 0.5 * (P_ + P_.transpose()).eval();  // keep symmetric against drift
}

std::vector<IekfFrontEnd::Waypoint> IekfFrontEnd::propagateGroup(const std::vector<ImuSample>& imu,
                                                                 Timestamp t_scan_end) {
  std::vector<Waypoint> trail;
  if (imu.empty()) {
    return trail;
  }
  auto snapshot = [&](Timestamp stamp, const Eigen::Vector3d& acc_mid,
                      const Eigen::Vector3d& gyro_mid) {
    Waypoint w;
    w.stamp = stamp;
    w.T_world_imu = state_.T_world_body;
    w.vel = state_.v_world;
    w.omega = gyro_mid - state_.b_g;
    w.acc_world = state_.T_world_body.R() * (acc_mid - state_.b_a) + state_.g_world;
    trail.push_back(w);
  };

  // The first waypoint anchors the trail at the first sample's state (scan start
  // side); its interval constants come from the first midpoint input.
  Eigen::Vector3d acc0 = imu.front().acc, gyro0 = imu.front().gyro;
  if (imu.size() >= 2) {
    acc0 = 0.5 * (imu[0].acc + imu[1].acc);
    gyro0 = 0.5 * (imu[0].gyro + imu[1].gyro);
  }
  snapshot(imu.front().stamp, acc0, gyro0);

  for (std::size_t i = 1; i < imu.size(); ++i) {
    const ImuSample& prev = imu[i - 1];
    const ImuSample& curr = imu[i];
    const double dt = to_seconds(curr.stamp - prev.stamp);
    if (dt <= 0.0) {
      continue;
    }
    const Eigen::Vector3d acc_mid = 0.5 * (prev.acc + curr.acc);
    const Eigen::Vector3d gyro_mid = 0.5 * (prev.gyro + curr.gyro);
    last_stamp_ = curr.stamp;
    predict(acc_mid, gyro_mid, dt);
    snapshot(curr.stamp, acc_mid, gyro_mid);
  }

  // Step to the exact scan-end time (signed; may extrapolate backward slightly).
  const Timestamp t_last = imu.back().stamp;
  const double dt_end = to_seconds(t_scan_end - t_last);
  if (std::abs(dt_end) > 0.0) {
    const Eigen::Vector3d acc_mid = imu.back().acc;
    const Eigen::Vector3d gyro_mid = imu.back().gyro;
    last_stamp_ = t_scan_end;
    predict(acc_mid, gyro_mid, dt_end);
    snapshot(t_scan_end, acc_mid, gyro_mid);
  }
  return trail;
}

std::vector<Eigen::Vector3d> IekfFrontEnd::deskewAndDownsample(const LidarScan& scan,
                                                               const std::vector<Waypoint>& trail,
                                                               const Pose& T_world_end) const {
  std::vector<Eigen::Vector3d> out;
  if (!scan.points || trail.empty()) {
    return out;
  }
  const Pose T_world_end_inv = T_world_end.inverse();

  auto poseAt = [&](Timestamp t) -> Pose {
    // Find the last waypoint whose stamp precedes t, then advance within the
    // interval with constant body rate and constant world acceleration.
    std::size_t idx = 0;
    for (std::size_t i = 0; i < trail.size(); ++i) {
      if (trail[i].stamp <= t) {
        idx = i;
      } else {
        break;
      }
    }
    const Waypoint& h = trail[idx];
    const double dt = to_seconds(t - h.stamp);
    Pose p;
    p.q = (h.T_world_imu.q * Sophus::SO3d::exp(h.omega * dt).unit_quaternion()).normalized();
    p.t = h.T_world_imu.t + h.vel * dt + 0.5 * h.acc_world * dt * dt;
    return p;
  };

  // Deskew every point into the body frame at scan-end, accumulating into a voxel
  // hash for the downsample.
  const double voxel = std::max(cfg_.lidar.voxel_map_m, 1e-3);
  const double inv_voxel = 1.0 / voxel;
  std::unordered_map<std::int64_t, Eigen::Vector3d> voxels;
  voxels.reserve(scan.points->size());

  const Timestamp t0 = scan.stamp_start;
  for (const LidarPoint& pt : *scan.points) {
    if (pt.range <= 0.f && pt.xyz.squaredNorm() <= 0.f) {
      continue;
    }
    const Timestamp t_i = t0 + pt.t_offset_ns;
    const Pose T_world_i = poseAt(t_i);
    // LiDAR@t_i -> IMU@t_i -> world -> IMU/body@end.
    const Eigen::Vector3d p_imu = T_imu_lidar_ * pt.xyz.cast<double>();
    const Eigen::Vector3d p_world = T_world_i * p_imu;
    const Eigen::Vector3d p_body = T_world_end_inv * p_world;

    const auto key = [&](double c) { return static_cast<std::int64_t>(std::floor(c * inv_voxel)); };
    const std::int64_t kx = key(p_body.x());
    const std::int64_t ky = key(p_body.y());
    const std::int64_t kz = key(p_body.z());
    const std::int64_t h = (kx * 73856093LL) ^ (ky * 19349663LL) ^ (kz * 83492791LL);
    auto it = voxels.find(h);
    if (it == voxels.end()) {
      voxels.emplace(h, p_body);
    }
  }
  out.reserve(voxels.size());
  for (const auto& kv : voxels) {
    out.push_back(kv.second);
  }
  return out;
}

int IekfFrontEnd::iteratedUpdate(const std::vector<Eigen::Vector3d>& body_pts) {
  MERIDIAN_SCOPED_TIME(telemetry_, "frontend.iekf.update", last_stamp_);
  last_info_pose_.setZero();
  if (body_pts.empty() || map_->size() == 0) {
    diag_.iterations = 0;
    diag_.effective_points = 0;
    diag_.mean_residual = 0.0;
    diag_.final_residual = 0.0;
    return 0;
  }

  const double r_inv = 1.0 / std::max(cfg_.lidar.point_cov, 1e-9);
  const int max_iter = std::max(cfg_.solver_max_iterations, 1);
  const double epsi = cfg_.solver_epsi > 0 ? cfg_.solver_epsi : 1e-3;
  const int k_nn = std::max(cfg_.lidar.num_match_points, 3);
  const double max_dist_sq = cfg_.lidar.max_match_dist_sq;
  const double plane_thresh = cfg_.lidar.plane_thresh;

  const NavState state_prop = state_;  // frozen prior
  const Cov P_prop = P_;
  const Cov Pinv = P_prop.ldlt().solve(Cov::Identity());
  int effective = 0;
  int iters = 0;
  double mean_abs_res = 0.0;
  double final_cost = 0.0;
  Cov Sigma = Pinv;  // last iteration's information matrix (HtRinvH + Pinv)

  for (int it = 0; it < max_iter; ++it) {
    iters = it + 1;
    const Eigen::Matrix3d R = state_.T_world_body.R();
    const Eigen::Vector3d p = state_.T_world_body.t;

    // Accumulate the normal equations in the information form: HtRinvH (18x18),
    // HtRinvz (18), built one effective point at a time so a full m x 18 H is
    // never materialised.
    Cov HtRinvH = Cov::Zero();
    Tangent HtRinvz = Tangent::Zero();
    Eigen::Matrix<double, 6, 6> info_pose = Eigen::Matrix<double, 6, 6>::Zero();
    int eff = 0;
    double sum_abs = 0.0;
    double cost = 0.0;

    KD_TREE<ikdTree_PointType>::PointVector nbrs;
    std::vector<float> dists;
    for (const Eigen::Vector3d& pb : body_pts) {
      // Body point here is already in the IMU/body frame at scan-end, so the
      // world transform uses only the IMU pose.
      const Eigen::Vector3d pw = R * pb + p;
      nbrs.clear();
      dists.clear();
      map_->Nearest_Search(toIkd(pw), k_nn, nbrs, dists);
      if (static_cast<int>(nbrs.size()) < k_nn || dists.back() > max_dist_sq) {
        continue;
      }
      std::vector<Eigen::Vector3d> near;
      near.reserve(nbrs.size());
      for (const auto& q : nbrs) {
        near.emplace_back(q.x, q.y, q.z);
      }
      Eigen::Vector3d n;
      double d;
      if (!fitPlane(near, plane_thresh, &n, &d)) {
        continue;
      }
      // Point-to-plane robustness gate: confidence falls with residual and rises
      // with range; accept only well-supported planar matches.
      const double res = n.dot(pw) + d;
      const double s = 1.0 - 0.9 * std::abs(res) / std::sqrt(pb.norm() + 1e-9);
      if (s <= 0.9) {
        continue;
      }

      // Per-point Jacobian row (right perturbation): only the pose block is
      // populated for point-to-plane in the no-extrinsic state.
      //   dz/d(rho) = n^T ; dz/d(phi) = -n^T R [pb]x
      Eigen::Matrix<double, 1, kDof> H = Eigen::Matrix<double, 1, kDof>::Zero();
      H.block<1, 3>(0, kP) = n.transpose();
      H.block<1, 3>(0, kR) = -(n.transpose() * R * skew(pb));

      HtRinvH += (H.transpose() * H) * r_inv;
      HtRinvz += H.transpose() * (r_inv * res);
      info_pose += (H.block<1, 6>(0, kP).transpose() * H.block<1, 6>(0, kP)) * r_inv;
      sum_abs += std::abs(res);
      cost += res * res * r_inv;
      ++eff;
    }

    eff > 0 ? mean_abs_res = sum_abs / eff : mean_abs_res = 0.0;
    effective = eff;
    final_cost = cost;
    last_info_pose_ = info_pose;
    if (eff == 0) {
      break;
    }

    // Prior pull-back: difference from the propagated (frozen) prior on the same
    // decoupled chart the update retracts along. For this additive-position /
    // right-rotation state the chart transport is the identity to first order, so
    // the local difference is used directly.
    const Tangent dx_prior = boxminusNav(state_, state_prop);

    // Information-form normal equations, inverting only 18x18:
    //   (HtRinvH + P^-1) delta = -(HtRinvz + P^-1 dx_prior).
    // The right-hand side is the gradient of the MAP cost at the current iterate;
    // the P^-1 dx_prior term is the prior pull-back toward the frozen prior.
    Sigma = HtRinvH + Pinv;
    const Tangent rhs = -(HtRinvz + Pinv * dx_prior);
    const Tangent delta = Sigma.ldlt().solve(rhs);

    state_ = boxplusConstrained(state_, delta);

    // Per-DoF convergence on the increment.
    bool converged = true;
    for (int j = 0; j < kDof; ++j) {
      if (std::abs(delta(j)) >= epsi) {
        converged = false;
        break;
      }
    }
    if (converged) {
      break;
    }
  }

  // Posterior covariance is the inverse of the last iteration's information
  // matrix (HtRinvH + P^-1), kept symmetric against numerical drift.
  P_ = Sigma.ldlt().solve(Cov::Identity());
  P_ = 0.5 * (P_ + P_.transpose()).eval();
  diag_.iterations = iters;
  diag_.effective_points = effective;
  diag_.mean_residual = mean_abs_res;
  diag_.final_residual = final_cost;
  return effective;
}

int IekfFrontEnd::photometricUpdate(const ct::ImagePyramidView& img) {
  if (!cam_model_.valid() || !have_cam_extrinsic_ || !vmap_ || vmap_->size() == 0 ||
      img.levels() == 0) {
    return 0;
  }

  // Frozen post-LiDAR prior: the photometric step is a sequential MAP update that pulls
  // back toward the state and covariance the point-to-plane update left, so the two
  // measurements compose without double-counting the prior.
  const NavState state_prop = state_;
  const Cov P_prop = P_;
  const Cov Pinv = P_prop.ldlt().solve(Cov::Identity());

  // Single-level (level 0) patch photometric model. img_point_cov whitens the residual;
  // a Huber-like cap keeps a few decorrelated patches from dominating the normal
  // equations the way the LiDAR update's range gate does for planes.
  const int patch = std::max(cfg_.visual.patch, 2);
  const int half = patch / 2;
  const double r_inv = 1.0 / std::max(cfg_.visual.img_point_cov, 1e-9);
  const double tau_cur = inv_expo_;
  constexpr int kLevel = 0;
  const double kHuber = 10.0;  // residual magnitude (intensity units) at robust knee

  const Pose T_world_cam = state_.T_world_body * T_imu_cam_;
  const std::vector<const ct::VisualPoint*> candidates =
      vmap_->visibleCandidates(cam_model_, T_world_cam);
  if (candidates.empty()) {
    return 0;
  }

  const Eigen::Matrix3d R_wb = state_.T_world_body.R();
  const Eigen::Vector3d p_wb = state_.T_world_body.t;
  const Pose T_cam_body = T_imu_cam_.inverse();
  const Eigen::Matrix3d R_cb = T_cam_body.R();  // R_cam_body = R_body_cam^T
  const Eigen::Vector3d t_cb = T_cam_body.t;

  Cov HtRinvH = Cov::Zero();
  Tangent HtRinvz = Tangent::Zero();
  int effective = 0;
  for (const ct::VisualPoint* pt : candidates) {
    if (pt == nullptr || !pt->normal_initialized) {
      continue;
    }
    const Eigen::Vector3d q = R_wb.transpose() * (pt->p_world - p_wb);  // body frame
    const Eigen::Vector3d p_c = R_cb * q + t_cb;
    if (p_c.z() <= 0.0) {
      continue;
    }
    Eigen::Vector2d u0;
    if (!cam_model_.project(p_c, &u0)) {
      continue;
    }
    if (!img.inBounds(kLevel, u0, half + 1)) {
      continue;
    }
    const Eigen::Matrix<double, 2, 3> Jpi = cam_model_.projectJacobian(p_c);
    const Eigen::Matrix3d dPc_drho = -R_cb * R_wb.transpose();
    const Eigen::Matrix3d dPc_dphi = R_cb * skew(q);

    const auto& P_ref = pt->ref_patches[kLevel];
    for (int yy = 0; yy < patch; ++yy) {
      for (int xx = 0; xx < patch; ++xx) {
        const Eigen::Vector2d u(u0.x() + (xx - half), u0.y() + (yy - half));
        const double i_cur = img.intensity(kLevel, u);
        const Eigen::Vector2d g = img.gradient(kLevel, u);  // level-0 units
        // The reference patch is stored patch(x_index, y_index).
        const double i_ref = static_cast<double>(P_ref(xx, yy));
        const double res = tau_cur * i_cur - pt->inv_expo_ref * i_ref;

        // Photometric chain: J_img (1x2) * J_pi (2x3) * dPc/dstate (3x6, pose block).
        const Eigen::RowVector2d J_img = tau_cur * g.transpose();
        const Eigen::RowVector3d row = J_img * Jpi;
        Eigen::Matrix<double, 1, kDof> H = Eigen::Matrix<double, 1, kDof>::Zero();
        H.block<1, 3>(0, kP) = row * dPc_drho;
        H.block<1, 3>(0, kR) = row * dPc_dphi;

        // Huber robust weight on the whitened residual.
        const double w_res = std::abs(res) * std::sqrt(r_inv);
        const double robust = w_res > kHuber * std::sqrt(r_inv)
                                  ? (kHuber * std::sqrt(r_inv)) / std::max(w_res, 1e-12)
                                  : 1.0;
        const double rw = r_inv * robust;
        HtRinvH += (H.transpose() * H) * rw;
        HtRinvz += H.transpose() * (rw * res);
      }
    }
    ++effective;
  }
  if (effective == 0) {
    return 0;
  }

  // One MAP step in information form: (HtRinvH + P^-1) delta = -(HtRinvz + P^-1 dx_prior).
  const Tangent dx_prior = boxminusNav(state_, state_prop);
  const Cov Sigma = HtRinvH + Pinv;
  const Tangent rhs = -(HtRinvz + Pinv * dx_prior);
  const Tangent delta = Sigma.ldlt().solve(rhs);
  if (!delta.allFinite()) {
    return effective;
  }
  state_ = boxplusConstrained(state_, delta);
  P_ = Sigma.ldlt().solve(Cov::Identity());
  P_ = 0.5 * (P_ + P_.transpose()).eval();
  return effective;
}

void IekfFrontEnd::updateMap(const std::vector<Eigen::Vector3d>& body_pts) {
  if (body_pts.empty()) {
    return;
  }
  const Eigen::Matrix3d R = state_.T_world_body.R();
  const Eigen::Vector3d p = state_.T_world_body.t;
  KD_TREE<ikdTree_PointType>::PointVector add;
  add.reserve(body_pts.size());
  for (const Eigen::Vector3d& pb : body_pts) {
    add.push_back(toIkd(R * pb + p));
  }
  if (map_->size() == 0) {
    map_->Build(add);
  } else {
    map_->Add_Points(add, true);
  }

  // Segment the ikd-Tree to a cube around the current body so map RAM and the per-point
  // nearest-neighbour search depth stay bounded over a long mission. The cube is
  // recentred only when the body nears a face (within move_margin, which exceeds the NN
  // search radius so points around the body are never deleted); the exited slab between
  // the old and new face is then box-deleted.
  const double cube_m = cfg_.lidar.local_map_cube_m;
  if (cube_m > 0.0 && map_->size() > 0) {
    const float half = static_cast<float>(cube_m) * 0.5f;
    const Eigen::Vector3f c = p.cast<float>();
    if (!map_cube_init_) {
      for (int i = 0; i < 3; ++i) {
        map_cube_[i] = c(i) - half;
        map_cube_[i + 3] = c(i) + half;
      }
      map_cube_init_ = true;
    } else {
      const float search_radius = std::sqrt(static_cast<float>(cfg_.lidar.max_match_dist_sq));
      const float move_margin = std::max(search_radius, half * 0.2f);
      const float mov_dist = std::max(half - move_margin, move_margin);
      std::vector<BoxPointType> need_remove;
      std::array<double, 6> updated = map_cube_;
      for (int i = 0; i < 3; ++i) {
        const float lo = static_cast<float>(map_cube_[i]);
        const float hi = static_cast<float>(map_cube_[i + 3]);
        if (std::abs(c(i) - lo) <= move_margin) {
          updated[i] -= mov_dist;
          updated[i + 3] -= mov_dist;
          BoxPointType slab = toBox(map_cube_);
          slab.vertex_min[i] = static_cast<float>(updated[i + 3]);
          need_remove.push_back(slab);
        } else if (std::abs(c(i) - hi) <= move_margin) {
          updated[i] += mov_dist;
          updated[i + 3] += mov_dist;
          BoxPointType slab = toBox(map_cube_);
          slab.vertex_max[i] = static_cast<float>(updated[i]);
          need_remove.push_back(slab);
        }
      }
      map_cube_ = updated;
      if (!need_remove.empty()) {
        map_->Delete_Point_Boxes(need_remove);
      }
    }
  }
}

void IekfFrontEnd::ingest(const PreprocessedGroup& group) {
  const auto t_start = Clock::now();
  refreshExtrinsic();
  const MeasureGroup& mg = group.group;
  const Timestamp t_end = mg.t_end != 0 ? mg.t_end : mg.scan.stamp_start + mg.scan.sweep_duration;

  // Initialise the filter on the first usable static window: gravity direction and
  // gyro bias from the shared Welford initializer (with its motion gate), pose at
  // identity in the odom frame. The accel bias stays at its zero prior (unobservable
  // while static). Folds IMU across groups until the gate accepts; emits nothing and
  // returns until then.
  if (!filter_initialized_) {
    if (mg.imu.empty()) {
      return;
    }
    if (!imu_init_) {
      // Map init_time_s to a sample count from the first group's measured IMU period
      // so the static window length is independent of the IMU rate.
      double imu_dt_s = 0.01;
      if (mg.imu.size() >= 2) {
        const double span = to_seconds(mg.imu.back().stamp - mg.imu.front().stamp);
        if (span > 0.0) {
          imu_dt_s = span / static_cast<double>(mg.imu.size() - 1);
        }
      }
      const int count =
          std::max(1, static_cast<int>(std::lround(std::max(cfg_.init_time_s, 0.0) / imu_dt_s)));
      imu_init_ = std::make_unique<ImuInitializer>(PreprocImu{}, count);
    }
    for (const ImuSample& s : mg.imu) {
      imu_init_->add(s);
      if (imu_init_->done()) {
        break;
      }
    }
    if (!imu_init_->done()) {
      return;  // still accumulating a static window (or the gate rejected this batch)
    }

    const ImuInitState& init = imu_init_->state();
    state_.g_world = init.gravity;
    state_.b_g = init.gyro_bias;
    state_.b_a = init.accel_bias;
    state_.T_world_body = Pose{};
    state_.v_world.setZero();
    state_.stamp = mg.imu.front().stamp;
    last_stamp_ = state_.stamp;

    P_ = Cov::Identity();
    P_.block<3, 3>(kP, kP) *= 1e-6;
    P_.block<3, 3>(kR, kR) *= 1e-6;
    P_.block<3, 3>(kV, kV) *= 1e-4;
    P_.block<3, 3>(kBg, kBg) *= 1e-4;
    P_.block<3, 3>(kBa, kBa) *= 1e-3;
    P_.block<3, 3>(kG, kG) *= 1e-5;
    filter_initialized_ = true;
    live_state_ = state_;
    live_stamp_ = state_.stamp;
  }

  // Propagate across the group's IMU and deskew using the resulting trail.
  std::vector<Waypoint> trail;
  double deskew_ms = 0.0;
  std::vector<Eigen::Vector3d> body_pts;
  {
    trail = propagateGroup(mg.imu, t_end);
    state_.stamp = t_end;
    const auto t_dk = Clock::now();
    const Pose T_world_end = state_.T_world_body;
    if (!trail.empty()) {
      body_pts = deskewAndDownsample(mg.scan, trail, T_world_end);
    }
    deskew_ms = ms_since(t_dk);
  }

  const auto t_solve = Clock::now();
  const int eff = iteratedUpdate(body_pts);

  // Sequential FAST-LIVO2-style photometric update after the point-to-plane update.
  // Builds the intensity pyramid once; the view borrows it, so both outlive the update.
  // No-op when the camera model is invalid (zero intrinsics), gating the visual stage
  // off so the oracle stays a clean LIO baseline on a camera-less rig.
  int eff_vis = 0;
  std::vector<cv::Mat> pyramid;
  if (cfg_.visual.enable && cam_model_.valid() && have_cam_extrinsic_ && mg.image.has_value()) {
    pyramid = buildImagePyramid(*mg.image);
    if (!pyramid.empty()) {
      ct::ImagePyramidView img_view(pyramid);
      eff_vis = photometricUpdate(img_view);
      // Visual map lifecycle against the now-updated pose: re-score / add observations,
      // promote this sweep's plane-associated points, and box-evict points that left the
      // active box. Promotion needs world points + plane normals, which the deskewed
      // body points lack, so it is fed LiDAR hits derived from the registered cloud.
      const Pose T_world_cam_now = state_.T_world_body * T_imu_cam_;
      vmap_->updateAfterSolve(img_view, cam_model_, T_world_cam_now, inv_expo_);
      std::vector<ct::LidarHit> vis_hits;
      vis_hits.reserve(body_pts.size());
      const Eigen::Matrix3d Rwb = state_.T_world_body.R();
      const Eigen::Vector3d pwb = state_.T_world_body.t;
      for (const Eigen::Vector3d& pb : body_pts) {
        const Eigen::Vector3d pw = Rwb * pb + pwb;
        Eigen::Vector3d n;
        double d;
        // A 5-NN plane fit at the registered point gives the warp normal the visual
        // point needs; points without a clean plane are skipped at promotion.
        KD_TREE<ikdTree_PointType>::PointVector nbrs;
        std::vector<float> dists;
        const int k_nn = std::max(cfg_.lidar.num_match_points, 3);
        map_->Nearest_Search(toIkd(pw), k_nn, nbrs, dists);
        if (static_cast<int>(nbrs.size()) < k_nn || dists.back() > cfg_.lidar.max_match_dist_sq) {
          continue;
        }
        std::vector<Eigen::Vector3d> near;
        near.reserve(nbrs.size());
        for (const auto& qn : nbrs) {
          near.emplace_back(qn.x, qn.y, qn.z);
        }
        if (!fitPlane(near, cfg_.lidar.plane_thresh, &n, &d)) {
          continue;
        }
        ct::LidarHit h;
        h.p_world = pw;
        h.plane.n = n;
        h.plane.d = d;
        h.plane.valid = true;
        h.t_offset_ns = static_cast<std::int32_t>(vis_hits.size());
        vis_hits.push_back(h);
      }
      vmap_->promote(img_view, cam_model_, T_world_cam_now, inv_expo_, vis_hits);
      vmap_->evict(T_world_cam_now.t);
    }
  }
  const double solve_ms = ms_since(t_solve);

  // Per-axis observability of the solve just completed, from its pose information
  // block; emitKeyframe reuses this same report.
  diag_.observability = computeObservability();

  // Seed / extend the local map with the registered cloud, then maintain it.
  updateMap(body_pts);

  // Live output tracks the filter after a solve.
  live_state_ = state_;
  live_stamp_ = state_.stamp;

  // Keyframe decision and emission.
  const Pose T_world_end = state_.T_world_body;
  if (keyframeDue(T_world_end, t_end)) {
    emitKeyframe(T_world_end, t_end, body_pts);
  }

  diag_.deskew_time_ms = deskew_ms;
  diag_.solve_time_ms = solve_ms;
  diag_.scan_time_ms = ms_since(t_start);
  diag_.restarted = false;

  if (telemetry_ && telemetry_->enabled("frontend/lidar/n_inlier")) {
    telemetry_->scalar("frontend/lidar/n_inlier", static_cast<double>(eff), last_stamp_);
  }
  if (telemetry_ && telemetry_->enabled("frontend/iter_count")) {
    telemetry_->scalar("frontend/iter_count", static_cast<double>(diag_.iterations), last_stamp_);
  }
  if (telemetry_ && telemetry_->enabled("frontend/solve_ms")) {
    telemetry_->scalar("frontend/solve_ms", diag_.solve_time_ms, last_stamp_);
  }
  if (telemetry_ && telemetry_->enabled("odom/body")) {
    telemetry_->pose("odom/body", state_.T_world_body, Frame::Body, last_stamp_);
  }
  if (telemetry_ && telemetry_->enabled("frontend/visual/n_converged")) {
    telemetry_->scalar("frontend/visual/n_converged", static_cast<double>(eff_vis), last_stamp_);
  }
}

void IekfFrontEnd::ingest_imu_live(const ImuSample& imu) {
  if (!filter_initialized_) {
    return;
  }
  if (live_stamp_ == 0) {
    live_state_ = state_;
    live_stamp_ = imu.stamp;
    return;
  }
  const double dt = to_seconds(imu.stamp - live_stamp_);
  if (dt <= 0.0) {
    return;
  }
  // Advance a private copy only; the filter state is untouched until the next
  // ingest() re-derives the live state from the solved filter.
  predictMean(live_state_, imu.acc, imu.gyro, dt);
  live_state_.stamp = imu.stamp;
  live_stamp_ = imu.stamp;
}

void IekfFrontEnd::apply_correction(const GraphUpdate& update) {
  if (update.moved.empty()) {
    return;
  }
  // Shift the filter pose by the rigid correction implied by the most recent
  // moved keyframe. The local map is left in the old odom frame: no map rebase
  // is performed.
  const GraphUpdate::Moved& m = update.moved.back();
  if (m.id == prev_kf_id_ && have_prev_kf_) {
    const Pose delta = m.new_T_map_body * prev_kf_pose_.inverse();
    state_.T_world_body = delta * state_.T_world_body;
    state_.v_world = delta.q * state_.v_world;
    live_state_ = state_;
  }
}

ObservabilityReport IekfFrontEnd::computeObservability() const {
  // Eigen-decompose the pose information block, assign each eigenvalue's
  // conditioning score = lambda/(lambda+kappa) to the unassigned axis its
  // eigenvector most aligns with, and keep the score array translation-first
  // [tx,ty,tz,rx,ry,rz]. When any direction is degenerate the eigenvectors are
  // exported so the back-end can read the true degenerate directions.
  ObservabilityReport obs;
  obs.frame = Frame::Body;
  const double kappa = cfg_.degeneracy_thresh > 0 ? cfg_.degeneracy_thresh : 10.0;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(last_info_pose_);
  const Eigen::Matrix<double, 6, 1> lambdas = es.eigenvalues();
  const Eigen::Matrix<double, 6, 6> V = es.eigenvectors();
  std::array<double, 6> axis_score = {1, 1, 1, 1, 1, 1};
  std::array<bool, 6> assigned = {false, false, false, false, false, false};
  for (int e = 0; e < 6; ++e) {
    int best_axis = -1;
    double best_abs = -1.0;
    for (int a = 0; a < 6; ++a) {
      if (assigned[a]) continue;
      const double c = std::abs(V(a, e));
      if (c > best_abs) {
        best_abs = c;
        best_axis = a;
      }
    }
    if (best_axis >= 0) {
      const double lambda = std::max(lambdas(e), 0.0);
      axis_score[best_axis] = lambda / (lambda + kappa);
      assigned[best_axis] = true;
    }
  }
  obs.score = axis_score;
  bool degenerate = false;
  for (double s : axis_score) {
    if (s < 0.1) degenerate = true;
  }
  if (degenerate) {
    obs.eigvecs = V;
  }
  return obs;
}

bool IekfFrontEnd::keyframeDue(const Pose& T_world_end, Timestamp stamp) const {
  if (!have_prev_kf_) {
    return true;  // first keyframe after init
  }
  const Pose rel = prev_kf_pose_.inverse() * T_world_end;
  const double dist = rel.t.norm();
  const double rot_deg = Sophus::SO3d(rel.q).log().norm() * 180.0 / M_PI;
  const double elapsed_s = to_seconds(stamp - prev_kf_stamp_);
  return dist >= cfg_.keyframe.dist_m || rot_deg >= cfg_.keyframe.rot_deg ||
         elapsed_s >= cfg_.keyframe.time_s;
}

void IekfFrontEnd::emitKeyframe(const Pose& T_world_end, Timestamp stamp,
                                const std::vector<Eigen::Vector3d>& body_pts) {
  KeyframePacket pkt;
  pkt.id = next_kf_id_++;
  pkt.stamp = stamp;
  pkt.ref_frame = Frame::Odom;
  pkt.T_ref_body = T_world_end;
  pkt.calib_version = calib_version_;
  pkt.frontend_kind = 0;
  pkt.kinematics_included = false;

  // Pose marginal (translation-first [rho;phi]) from the filter covariance.
  Eigen::Matrix<double, 6, 6> pose_cov = P_.block<6, 6>(kP, kP);

  // Observability from the pose information block, computed once per solve and
  // reused here so the packet and diagnostics() report the same value.
  pkt.observability = diag_.observability;

  if (!have_prev_kf_) {
    pkt.constraint_kind = KeyframePacket::ConstraintKind::AbsolutePrior;
    pkt.rel_to_id = 0;
    pkt.T_relto_this = Pose{};
    pkt.constraint_cov.form = GaussianBlock<6>::Form::Covariance;
    pkt.constraint_cov.M = reorderTransRotToRotTrans(pose_cov);
  } else {
    pkt.constraint_kind = KeyframePacket::ConstraintKind::RelativeBetween;
    pkt.rel_to_id = prev_kf_id_;
    pkt.T_relto_this = prev_kf_pose_.inverse() * T_world_end;
    // Conservative relative covariance: sum the two absolute pose marginals (the
    // previous keyframe's and the current one's). This upper-bounds the true
    // relative marginal, which would subtract their cross-covariance, so the
    // emitted edge over-states rather than under-states uncertainty.
    Eigen::Matrix<double, 6, 6> prev_pose_cov = prev_kf_cov_.block<6, 6>(kP, kP);
    Eigen::Matrix<double, 6, 6> rel_cov = pose_cov + prev_pose_cov;
    pkt.constraint_cov.form = GaussianBlock<6>::Form::Covariance;
    pkt.constraint_cov.M = reorderTransRotToRotTrans(rel_cov);
  }

  // cloud_body: the deskewed, downsampled cloud in the body frame at stamp.
  auto cloud = std::make_shared<PointCloud>();
  cloud->reserve(body_pts.size());
  for (const Eigen::Vector3d& pb : body_pts) {
    LidarPoint lp;
    lp.xyz = pb.cast<float>();
    cloud->push_back(lp);
  }
  pkt.cloud_body = std::move(cloud);

  prev_kf_pose_ = T_world_end;
  prev_kf_stamp_ = stamp;
  prev_kf_cov_ = P_;
  prev_kf_id_ = pkt.id;
  have_prev_kf_ = true;

  if (keyframe_sink_) {
    keyframe_sink_(std::move(pkt));
  }
}

}  // namespace meridian
