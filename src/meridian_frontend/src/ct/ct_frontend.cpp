#include "ct/ct_frontend.hpp"

#include <ceres/manifold.h>
#include <ceres/problem.h>
#include <ceres/solver.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <sophus/so3.hpp>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "iekf/iekf_frontend.hpp"
#include "meridian/common/cloud.hpp"
#include "meridian/common/point.hpp"
#include "meridian/debug/log.hpp"
#include "meridian/debug/telemetry.hpp"

namespace meridian {

namespace {

constexpr const char* kLogModule = "frontend.ct";
constexpr double kGravityMag = 9.81;
constexpr int kSplineOrder = 4;

// A LiDAR plane normal counts as aligned with a weak translation axis (and so its hit
// is exempt from the factor cap) when |n . axis| clears this cosine. cos(15 deg) keeps
// the exemption to normals genuinely constraining the weak direction.
constexpr double kWeakAxisCos = 0.966;

// Whitened-residual Huber threshold for the GNSS absolute-position residual. The
// residual is divided by the floored fix covariance's square root, so it is unit-
// variance; a 3-sigma switch to the linear regime keeps a survivor that still drifts
// past the innovation gate from dominating the cost.
constexpr double kHuberGnss = 3.0;

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}

// Symmetrised pseudo-inverse: invert eigenvalues above an absolute floor relative to
// the largest, zero the rest. Returns false if no eigenvalue clears the floor (the
// matrix is effectively null), so the caller can reject a rank-deficient block.
bool robustInverse(const Eigen::MatrixXd& A, Eigen::MatrixXd* out) {
  const Eigen::MatrixXd sym = 0.5 * (A + A.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sym);
  const Eigen::VectorXd& lambda = es.eigenvalues();
  const double lam_max = lambda.size() > 0 ? lambda.cwiseAbs().maxCoeff() : 0.0;
  const double floor = std::max(1e-12, 1e-9 * lam_max);
  Eigen::VectorXd inv(lambda.size());
  bool any = false;
  for (int i = 0; i < lambda.size(); ++i) {
    if (lambda(i) > floor) {
      inv(i) = 1.0 / lambda(i);
      any = true;
    } else {
      inv(i) = 0.0;
    }
  }
  if (!any) {
    return false;
  }
  *out = es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
  return true;
}

// Decode a raw CameraFrame to a single-channel CV_8UC1 intensity image. Mono passes
// through, Bayer is demosaiced to grey, RGB is converted to grey. Returns an empty Mat
// when the frame carries no usable bytes. No photometric/exposure scaling is applied:
// the inverse exposure is an optimised state, not a fixed preprocessing factor.
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

// Drop blind / invalid returns; the rest pass through unchanged so their per-point
// time offset and sensor-frame xyz reach the association step intact.
std::vector<LidarPoint> validReturns(const LidarScan& scan) {
  std::vector<LidarPoint> out;
  if (!scan.points) {
    return out;
  }
  out.reserve(scan.points->size());
  for (const LidarPoint& p : *scan.points) {
    if (p.range <= 0.f && p.xyz.squaredNorm() <= 0.f) {
      continue;
    }
    out.push_back(p);
  }
  return out;
}

}  // namespace

CtFrontEnd::CtFrontEnd(const FrontendConfig& cfg, std::shared_ptr<const CalibrationSet> calib,
                       TelemetrySink* telemetry, bool deterministic)
    : cfg_(cfg), telemetry_(telemetry), deterministic_(deterministic) {
  window_dt_ns_ = from_seconds(std::max(cfg_.spline.knot_dt_ms, 1.0) * 1e-3);
  gravity_ = std::make_unique<ct::GravityBlock>(Eigen::Vector3d(0, 0, -kGravityMag), kGravityMag);
  map_ = std::make_unique<ct::LidarLocalMap>(cfg_.lidar);
  live_state_.ref_frame = Frame::Odom;
  live_state_.g_world = gravity_->gravityWorld();
  if (calib) {
    set_calibration(std::move(calib));
  }
}

CtFrontEnd::~CtFrontEnd() = default;

void CtFrontEnd::set_calibration(std::shared_ptr<const CalibrationSet> calib) {
  calib_ = std::move(calib);
  if (calib_) {
    calib_version_ = calib_->version;
  }
  have_extrinsic_ = false;
  refreshExtrinsic();
  refreshCamera();
}

void CtFrontEnd::refreshExtrinsic() {
  if (have_extrinsic_ || !calib_) {
    return;
  }
  // The LiDAR extrinsic is stored as the sensor-to-F_e transform; the measurement
  // model wants T_fe_lidar directly. Fall back to identity for a single-sensor rig.
  for (const Extrinsic& e : calib_->extrinsics) {
    if (e.child == Frame::OsSensor0) {
      T_fe_lidar_ = e.T_parent_child;
      have_extrinsic_ = true;
      return;
    }
  }
  T_fe_lidar_ = Pose{};
  have_extrinsic_ = true;
}

void CtFrontEnd::refreshCamera() {
  // Rebuild the camera model + extrinsic from the calibration snapshot. The model
  // is invalid (and the visual stage gated off) whenever no camera intrinsics with
  // positive focal lengths are present.
  cam_model_ = CameraModel{};
  have_cam_extrinsic_ = false;
  inv_expo_prior_ = 1.0;
  inv_expo_std_ = 0.0;
  if (!calib_) {
    return;
  }
  // Camera optical frame -> F_e from the extrinsic list.
  for (const Extrinsic& e : calib_->extrinsics) {
    if (e.child == Frame::CamLink) {
      T_fe_cam_ = e.T_parent_child;
      have_cam_extrinsic_ = true;
      break;
    }
  }
  // Intrinsics: a single-camera rig uses the first entry. The CameraModel ctor leaves
  // the model invalid for zero focal lengths, which is exactly the disabled gate.
  if (!calib_->cam_intrinsics.empty()) {
    const IntrinsicsCamera& k = calib_->cam_intrinsics.begin()->second;
    cam_model_ = CameraModel(k);
    inv_expo_prior_ = k.inv_expo_prior > 0.0 ? k.inv_expo_prior : 1.0;
    inv_expo_std_ = std::max(k.inv_expo_std, 0.0);
  }
  // Build the visual map fresh against the configured patch/level/grid keys.
  vmap_ = std::make_unique<ct::VisualMap>(ct::VisualMapConfig(cfg_.visual));
  inv_expo_ = inv_expo_prior_;
}

std::vector<cv::Mat> CtFrontEnd::buildImagePyramid(const CameraFrame& frame) const {
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
      break;  // cannot halve a degenerate level further
    }
    cv::Mat down;
    cv::pyrDown(pyr.back(), down);
    pyr.push_back(down);
  }
  return pyr;
}

bool CtFrontEnd::visualEnabled(const PreprocessedGroup& group) const {
  return cfg_.visual.enable && cam_model_.valid() && have_cam_extrinsic_ &&
         group.group.image.has_value();
}

void CtFrontEnd::set_keyframe_sink(KeyframeSink sink) {
  keyframe_sink_ = std::move(sink);
}

NavState CtFrontEnd::live_state() const {
  return live_state_;
}

FrontEndDiagnostics CtFrontEnd::diagnostics() const {
  return diag_;
}

Duration CtFrontEnd::biasKnotDt() const {
  return static_cast<Duration>(std::max(cfg_.bias.knot_dt_ms, 1.0) * 1e6);
}

ct::ImuWeights CtFrontEnd::imuWeights(double imu_dt_s) const {
  ct::ImuWeights w;
  // Calibration carries continuous-time noise DENSITIES (per sqrt-Hz). A single IMU
  // sample's white-noise std is the density divided by sqrt(sample period), so the
  // sqrt-information weight 1/sigma the assembler applies stays scale-correct as the
  // rate changes and does not over-trust the IMU relative to the LiDAR. The bias
  // densities pass through unchanged; addBiasRandomWalk applies its own sqrt(dt).
  double dens_g = 1e-2, dens_a = 1e-1, dens_bg = 1e-4, dens_ba = 1e-4;
  if (calib_) {
    if (calib_->imu_gyr_noise > 0) dens_g = calib_->imu_gyr_noise;
    if (calib_->imu_acc_noise > 0) dens_a = calib_->imu_acc_noise;
    if (calib_->imu_gyr_bias_rw > 0) dens_bg = calib_->imu_gyr_bias_rw;
    if (calib_->imu_acc_bias_rw > 0) dens_ba = calib_->imu_acc_bias_rw;
  }
  const double sqrt_dt = std::sqrt(std::max(imu_dt_s, 1e-4));
  w.sigma_gyro = dens_g / sqrt_dt;
  w.sigma_accel = dens_a / sqrt_dt;
  w.sigma_bias_gyro = dens_bg;
  w.sigma_bias_accel = dens_ba;
  return w;
}

std::vector<Eigen::Vector3d> CtFrontEnd::weakTranslationAxes(
    const std::vector<ct::LidarHit>& hits) const {
  return ct::weakTranslationAxes(hits, cfg_.degeneracy_thresh);
}

int CtFrontEnd::selectKnotDensity(const std::vector<ImuSample>& imu) {
  // n_cp_max <= 1 disables adaptive density: one control point per segment (the
  // uniform-spline case), so the excitation gating below is skipped entirely.
  const int n_cp_max = cfg_.spline.n_cp_max > 0 ? cfg_.spline.n_cp_max : 1;
  if (n_cp_max <= 1) {
    last_n_cp_ = 1;
    return 1;
  }
  const ct::ImuExcitation exc =
      ct::imuExcitation(imu, anchor_bg_, anchor_ba_, gravity_->magnitude());
  const int n_cp = ct::knotDensityFromExcitation(
      exc, cfg_.spline.knot_omega_thresh, cfg_.spline.knot_accel_thresh,
      cfg_.spline.knot_density_hysteresis, n_cp_max, last_n_cp_);
  last_n_cp_ = n_cp;
  return n_cp;
}


std::vector<Timestamp> CtFrontEnd::segmentMidpointTimes(Timestamp t_begin, Timestamp t_end) const {
  // One evaluation point per outer segment the sweep spans: the midpoint of each
  // [seg_start, seg_start + window_dt_ns_) interval clamped into [t_begin, t_end].
  std::vector<Timestamp> mids;
  if (window_dt_ns_ <= 0 || t_end <= t_begin) {
    return mids;
  }
  const Timestamp half = window_dt_ns_ / 2;
  for (Timestamp s = t_begin; s < t_end; s += window_dt_ns_) {
    Timestamp m = s + half;
    if (m >= t_end) {
      m = (s + t_end) / 2;
    }
    if (spline_ && spline_->covers(m)) {
      mids.push_back(m);
    }
  }
  return mids;
}

std::function<Pose(Timestamp)> CtFrontEnd::buildSeed(const std::vector<ImuSample>& imu,
                                                     Timestamp t_begin, Timestamp t_end) {
  // Forward-integrate the body trajectory across the group's IMU from the current
  // anchor (last solved pose/velocity) with the current bias and gravity, then hand
  // back a closure that interpolates within the resulting waypoint trail. Each
  // interval advances with constant body rate and constant world acceleration.
  struct Waypoint {
    Timestamp stamp = 0;
    Pose T;
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc_world = Eigen::Vector3d::Zero();
  };

  const Eigen::Vector3d g_world = gravity_->gravityWorld();
  const Eigen::Vector3d bg = anchor_bg_;
  const Eigen::Vector3d ba = anchor_ba_;

  Pose T = anchor_pose_;
  Eigen::Vector3d vel = anchor_vel_;
  Timestamp t_prev = t_begin;  // the anchor pose is valid at the sweep's lower bound

  auto trail = std::make_shared<std::vector<Waypoint>>();
  auto pushWaypoint = [&](Timestamp stamp, const Eigen::Vector3d& acc_mid,
                          const Eigen::Vector3d& gyro_mid) {
    Waypoint w;
    w.stamp = stamp;
    w.T = T;
    w.vel = vel;
    w.omega = gyro_mid - bg;
    w.acc_world = T.q * (acc_mid - ba) + g_world;
    trail->push_back(w);
  };

  if (!imu.empty()) {
    // Anchor the trail at t_begin with the first available input so the warm start
    // starts exactly where the previous solve ended.
    Eigen::Vector3d acc0 = imu.front().acc, gyro0 = imu.front().gyro;
    if (imu.size() >= 2) {
      acc0 = 0.5 * (imu[0].acc + imu[1].acc);
      gyro0 = 0.5 * (imu[0].gyro + imu[1].gyro);
    }
    pushWaypoint(t_begin, acc0, gyro0);

    // Integrate forward across the IMU samples that advance time past the anchor;
    // samples at or before t_begin only seed the first interval's midpoint input.
    for (std::size_t i = 0; i < imu.size(); ++i) {
      const Timestamp stamp = imu[i].stamp;
      if (stamp <= t_prev) {
        continue;
      }
      const std::size_t lo = i > 0 ? i - 1 : i;
      const double dt = to_seconds(stamp - t_prev);
      const Eigen::Vector3d acc_mid = 0.5 * (imu[lo].acc + imu[i].acc);
      const Eigen::Vector3d gyro_mid = 0.5 * (imu[lo].gyro + imu[i].gyro);
      const Eigen::Vector3d omega = gyro_mid - bg;
      const Eigen::Vector3d acc_world = T.q * (acc_mid - ba) + g_world;
      T.t += vel * dt + 0.5 * acc_world * dt * dt;
      T.q = (T.q * Sophus::SO3d::exp(omega * dt).unit_quaternion()).normalized();
      vel += acc_world * dt;
      t_prev = stamp;
      pushWaypoint(stamp, acc_mid, gyro_mid);
    }
    // Step to the exact sweep end so the last waypoint covers t_end.
    const double dt_end = to_seconds(t_end - t_prev);
    if (dt_end > 0.0) {
      const Eigen::Vector3d acc_mid = imu.back().acc;
      const Eigen::Vector3d gyro_mid = imu.back().gyro;
      const Eigen::Vector3d omega = gyro_mid - bg;
      const Eigen::Vector3d acc_world = T.q * (acc_mid - ba) + g_world;
      T.t += vel * dt_end + 0.5 * acc_world * dt_end * dt_end;
      T.q = (T.q * Sophus::SO3d::exp(omega * dt_end).unit_quaternion()).normalized();
      vel += acc_world * dt_end;
      pushWaypoint(t_end, acc_mid, gyro_mid);
    }
  }

  // Terminal velocity / body rate of the integration: the constant-extrapolation
  // prediction past t_end, consumed by the tail anchors of the upcoming solve.
  seed_vel_end_ = trail->empty() ? anchor_vel_ : trail->back().vel;
  seed_omega_end_ = trail->empty() ? Eigen::Vector3d::Zero() : trail->back().omega;

  const Pose anchor = anchor_pose_;
  return [trail, anchor](Timestamp t) -> Pose {
    if (trail->empty()) {
      return anchor;
    }
    // Find the last waypoint preceding t and advance within its interval.
    std::size_t idx = 0;
    for (std::size_t i = 0; i < trail->size(); ++i) {
      if ((*trail)[i].stamp <= t) {
        idx = i;
      } else {
        break;
      }
    }
    const Waypoint& h = (*trail)[idx];
    const double dt = to_seconds(t - h.stamp);
    Pose p;
    p.q = (h.T.q * Sophus::SO3d::exp(h.omega * dt).unit_quaternion()).normalized();
    p.t = h.T.t + h.vel * dt + 0.5 * h.acc_world * dt * dt;
    return p;
  };
}

void CtFrontEnd::reseedAfterGap(const PreprocessedGroup& group, Timestamp t_begin,
                                Timestamp t_end) {
  const MeasureGroup& mg = group.group;

  // Carry the anchor across the data hole on a constant-velocity prediction; with no
  // IMU for the span there is nothing better, and the orientation is held. The map is
  // kept — the predicted anchor keeps trajectory and map in one frame to within the
  // prediction error, which the next solves absorb.
  anchor_pose_.t += anchor_vel_ * to_seconds(t_begin - last_solved_t_);
  restartWindow(t_begin);

  // Cover the reseed sweep and seed the map from it WITHOUT solving: directly after a
  // reseed the gauge pins hold most of the only evaluable segment's control points, so
  // a solve here would crush the whole residual budget into the one free knot and warp
  // the sweep. Inserting at the predicted poses instead mirrors the bootstrap, and the
  // next sweep solves on a fully extendable window.
  auto seed = buildSeed(mg.imu, t_begin, t_end);
  spline_->extendTo(t_end, seed, 1);
  bias_->extendTo(t_end);

  const std::vector<LidarPoint> ds =
      ct::voxelDownsample(validReturns(mg.scan), std::max(cfg_.lidar.voxel_map_m, 1e-3));
  std::vector<Eigen::Vector3d> world;
  world.reserve(ds.size());
  const Timestamp t0 = mg.scan.stamp_start;
  for (const LidarPoint& p : ds) {
    const Timestamp t = t0 + p.t_offset_ns;
    if (spline_->covers(t)) {
      world.push_back(spline_->pose(t) * (T_fe_lidar_ * p.xyz.cast<double>()));
    }
  }
  if (!world.empty()) {
    map_->insert(world);
  }

  const Pose T_world_end = spline_->covers(t_end) ? spline_->pose(t_end) : anchor_pose_;
  anchor_pose_ = T_world_end;
  last_solved_t_ = t_end;

  live_state_.T_world_body = T_world_end;
  live_state_.v_world = anchor_vel_;
  live_state_.b_g = anchor_bg_;
  live_state_.b_a = anchor_ba_;
  live_state_.g_world = gravity_->gravityWorld();
  live_state_.stamp = t_end;
  live_stamp_ = t_end;
}

void CtFrontEnd::restartWindow(Timestamp t_begin) {
  const int n_cp_max = cfg_.spline.n_cp_max > 0 ? cfg_.spline.n_cp_max : 1;
  spline_ = std::make_unique<SplineWindow>(window_dt_ns_, n_cp_max);
  spline_->initialize(t_begin, anchor_pose_);
  traj_start_t_ = t_begin;

  bias_ = std::make_unique<ct::BiasKnots>(t_begin, biasKnotDt(), 2);
  for (int k = 0; k < bias_->numKnots(); ++k) {
    bias_->setGyroKnot(k, anchor_bg_);
    bias_->setAccelKnot(k, anchor_ba_);
  }

  // The prior's parameter blocks point into the now-discarded knots; keeping it would
  // alias freed storage on the next solve.
  prior_.reset();
  diag_.prior_residual_dim = 0;

  last_n_cp_ = 1;
  last_solved_t_ = t_begin;
  window_floor_t_ = t_begin;
}

void CtFrontEnd::gateGnss(const PreprocessedGroup& group, Timestamp t_end, GnssContext* ctx) {
  // Inactive path: config gate off, or no fixes in the group. Nothing is allocated, no
  // anchor is created, no telemetry advances beyond the one informational inactive
  // state — the trajectory is bit-identical to a build without this stage.
  if (!cfg_.gnss.use || group.group.gnss.empty()) {
    if (telemetry_ && telemetry_->enabled("frontend/gnss/accept_rate") && !gnss_inactive_logged_) {
      gnss_inactive_logged_ = true;
      telemetry_->event(Level::Info, "frontend/gnss/inactive",
                        cfg_.gnss.use ? "gnss enabled, no fixes yet" : "gnss disabled in config",
                        t_end);
    }
    return;
  }

  // Active path. The anchor/gate are value members (no heap allocation), so the heap
  // layout the solver sees is unchanged whether or not any fix is processed; the gate is
  // emplaced from cfg on first use.
  if (!gnss_gate_) {
    gnss_gate_.emplace(cfg_.gnss);
  }

  // Antenna lever-arm from the calibration snapshot (GnssLink extrinsic translation);
  // identity-safe when absent so a rig without a configured antenna offset still fuses.
  ctx->t_fe_ant = Eigen::Vector3d::Zero();
  if (calib_) {
    for (const Extrinsic& e : calib_->extrinsics) {
      if (e.child == Frame::GnssLink) {
        ctx->t_fe_ant = e.T_parent_child.t;
        break;
      }
    }
  }

  // Position marginal of the current odometry (previous sweep's pose covariance
  // translation block), added to the floored fix covariance in the innovation gate so a
  // long GNSS gap during which odometry drifted does not reject an otherwise good fix.
  Eigen::Matrix3d pose_marg = Eigen::Matrix3d::Zero();
  if (have_pose_cov_) {
    pose_marg = last_pose_cov_.topLeftCorner<3, 3>();
  }

  for (const GnssFix& fix : group.group.gnss) {
    if (fix.fix == GnssFix::FixType::None) {
      continue;  // a None-quality fix is never used
    }
    const Timestamp t_fix = fix.stamp;
    if (!spline_->covers(t_fix)) {
      continue;  // the fix falls outside the spline's covered span this sweep
    }

    // Antenna world position at the fix's own time from the current seeded spline.
    const Pose T_w_fe = spline_->pose(t_fix);
    const Eigen::Vector3d p_w_ant = T_w_fe * ctx->t_fe_ant;

    // Anchor the local ENU frame at the first accepted fix so its antenna sits at the
    // spline antenna position; later fixes convert relative to it.
    if (!gnss_anchor_.set()) {
      gnss_anchor_.anchor(fix, p_w_ant);
    }
    const Eigen::Vector3d z_world = gnss_anchor_.toWorld(fix);

    const Eigen::Matrix3d cov_floored = ct::flooredCovEnu(fix, cfg_.gnss);
    const Eigen::Vector3d r = z_world - p_w_ant;
    const ct::GnssGateResult res = gnss_gate_->gate(r, cov_floored, pose_marg);

    if (telemetry_) {
      if (telemetry_->enabled("frontend/gnss/innovation_m")) {
        telemetry_->scalar("frontend/gnss/innovation_m", res.innovation_m, t_fix);
      }
      if (!res.accepted && telemetry_->enabled("frontend/gnss/reject")) {
        const char* reason = res.reason == ct::GnssGateResult::Reason::Gate ? "gate"
                             : res.reason == ct::GnssGateResult::Reason::ReacquirePersist
                                 ? "reacq_persist"
                                 : "quality";
        telemetry_->event(Level::Warn, "frontend/gnss/reject", reason, t_fix);
      }
    }

    if (!res.accepted) {
      continue;
    }

    // Whitening matrix: with cov = L L^T, sqrt_info = L^-1 satisfies
    // sqrt_info^T sqrt_info = cov^-1, so the whitened residual is unit-variance. A
    // non-PD covariance falls back to the per-axis floor diagonal.
    Eigen::Matrix3d sqrt_info;
    const Eigen::LLT<Eigen::Matrix3d> llt(cov_floored);
    if (llt.info() == Eigen::Success) {
      sqrt_info = llt.matrixL().solve(Eigen::Matrix3d::Identity());
    } else {
      sqrt_info = cov_floored.diagonal().cwiseMax(1e-6).cwiseInverse().cwiseSqrt().asDiagonal();
    }

    GnssMeasurement m;
    m.t_fix = t_fix;
    m.z_world = z_world;
    m.sqrt_info = sqrt_info;
    ctx->fixes.push_back(m);
  }

  if (telemetry_ && telemetry_->enabled("frontend/gnss/accept_rate")) {
    const int total = gnss_gate_->accepted() + gnss_gate_->rejected();
    const double rate =
        total > 0 ? static_cast<double>(gnss_gate_->accepted()) / static_cast<double>(total) : 0.0;
    telemetry_->scalar("frontend/gnss/accept_rate", rate, t_end);
  }
}

int CtFrontEnd::solveWindow(const PreprocessedGroup& group, Timestamp t_begin, Timestamp t_end,
                            std::vector<LidarPoint>* scan_ds, std::vector<ct::LidarHit>* hits,
                            VisualContext* vis, const GnssContext* gnss) {
  const Timestamp t0_scan = group.group.scan.stamp_start;

  // Downsample the raw scan once with the map voxel size as the scan voxel.
  const std::vector<LidarPoint> raw = validReturns(group.group.scan);
  *scan_ds = ct::voxelDownsample(raw, std::max(cfg_.lidar.voxel_map_m, 1e-3));

  // Representative IMU sample period for converting noise densities to per-sample
  // stds; the span over the count is robust to occasional jitter.
  double imu_dt_s = 0.005;
  const std::vector<ImuSample>& imu = group.group.imu;
  if (imu.size() >= 2) {
    imu_dt_s =
        to_seconds(imu.back().stamp - imu.front().stamp) / static_cast<double>(imu.size() - 1);
    imu_dt_s = std::max(imu_dt_s, 1e-4);
  }
  const ct::ImuWeights weights = imuWeights(imu_dt_s);

  // The window solver uses a sparse Cholesky on the banded normal equations: every
  // residual touches only the 4+4 local knots of its segment, so the Hessian is
  // banded and SPARSE_NORMAL_CHOLESKY is both correct and cheap here.
  ceres::Solver::Options sopts;
  sopts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  sopts.num_threads = 1;
  sopts.minimizer_progress_to_stdout = false;
  sopts.function_tolerance = 1e-8;

  // Outer/inner schedule. The outer loop re-associates against the current pose at
  // most max_outer_iters times; each pass runs reassoc_steps inner LM iterations on
  // the fixed correspondence set. The deterministic path runs the full schedule with
  // the early-stop disabled so the correspondence sequence is a pure function of the
  // input; the live path may stop early once associations stop moving.
  const int max_outer = std::max(cfg_.max_outer_iters, 1);
  const int inner_steps = std::max(cfg_.reassoc_steps, 1);
  // Inner LM iteration cap per pass, never above the solver cap.
  const int cap_inner = std::max(cfg_.solver_max_iterations, 1);
  sopts.max_num_iterations = std::min(inner_steps, cap_inner);
  // Minimum-iterations floor for the deadline bracket. Ceres has no native minimum-
  // iteration option, so the floor is enforced across the outer schedule: the wall-clock
  // cutoff is not allowed to terminate the schedule until at least this many total LM
  // iterations have run, so a single starved step never publishes a barely-improved pose.
  const int min_iters_floor = std::max(cfg_.solver.min_iterations, 1);

  // Total wall-clock budget for the whole outer/inner schedule (live path only); the
  // remaining budget is handed to each inner solve so re-association cannot blow the
  // per-step latency. The deterministic path ignores the deadline entirely.
  const double total_budget_s =
      deterministic_ ? 0.0 : std::max(cfg_.solver.time_limit_ms, 0.0) * 1e-3;
  const auto solve_start = Clock::now();

  // IMU bias box bounds: Ceres parameter bounds holding each bias axis inside its
  // physical range so an outlier burst cannot drive a bias to an absurd value that
  // corrupts the IMU residual for the whole window.
  const double bias_gyr_max = std::max(cfg_.bias.gyr_max, 0.0);
  const double bias_acc_max = std::max(cfg_.bias.acc_max, 0.0);

  // Each problem owns its cost functions and manifolds (default ceres ownership) and
  // frees them on destruction; the persistent prior_ object rebuilds its cost via
  // makeCost() per problem, so problem teardown never invalidates it.
  ceres::Problem problem;

  int accepted = 0;
  int total_iters = 0;
  bool deadline_hit = false;
  bool bound_engaged = false;
  Pose prev_assoc_pose;
  bool have_prev_assoc = false;
  std::vector<ct::LidarHit> capped_hits;  // factor-cap survivors, refilled each round
  int outer_passes = 0;
  for (int round = 0; round < max_outer; ++round) {
    ++outer_passes;
    // associate() appends, so start each round from an empty hit set evaluated at
    // the current (seeded, then solved) spline.
    hits->clear();
    {
      MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.assoc", t_end);
      ct::LidarAssocStats stats =
          ct::associate(*spline_, T_fe_lidar_, *scan_ds, t0_scan, *map_, cfg_.lidar, hits);
      accepted = stats.accepted;
    }

    problem = ceres::Problem();

    // Bound the LiDAR residual count: cap at max_lidar_factors by normal-stratified
    // subsample, exempting points whose normal aligns with a weak translation axis so
    // the cap never collapses the very axis observability protects. The outer-step
    // index rotates the per-stratum stride phase so the visited subset moves across
    // rounds while staying replay-identical.
    const std::vector<Eigen::Vector3d> weak_axes = weakTranslationAxes(*hits);
    last_cap_stats_ =
        ct::capHitsByNormalStrata(*hits, cfg_.lidar, round, weak_axes, kWeakAxisCos, &capped_hits);

    // LiDAR point-to-plane residuals at the current spline (capped set).
    ct::addLidarResiduals(problem, *spline_, T_fe_lidar_, capped_hits, cfg_.lidar);
    // IMU derivative residuals over the new sweep span + bias random walk.
    ct::addImuResiduals(problem, *spline_, *bias_, *gravity_, group.group.imu, t_begin, t_end,
                        weights);
    ct::addBiasRandomWalk(problem, *bias_, weights);

    // Tail anchors over the span past the newest measurement: tie the spline's
    // velocity and body rate there to the seed's constant extrapolation. The pinning
    // rule removes the strict null directions; the anchors additionally hold the
    // velocity read-out to the IMU prediction when the seed carries model error
    // (an unconverged bias), at sigmas a real measurement immediately overrides.
    {
      constexpr double kTailSigmaVel = 0.2;   // [m/s]
      constexpr double kTailSigmaRate = 0.1;  // [rad/s]
      const Timestamp stride = std::max<Timestamp>(window_dt_ns_ / 4, 1);
      std::vector<Timestamp> tail_times;
      for (int k = 1; k <= 8; ++k) {
        const Timestamp t = t_end + k * stride;
        if (!spline_->covers(t)) {
          break;
        }
        tail_times.push_back(t);
      }
      ct::addTailAnchors(problem, *spline_, tail_times, seed_vel_end_, seed_omega_end_,
                         kTailSigmaVel, kTailSigmaRate);
    }


    // Under-excitation regularizer (off by default, config-gated): a low-weight jerk /
    // angular-acceleration pull on the segment knot-midpoints, engaged only when the
    // peak IMU excitation is below the floor AND a translation axis is degenerate (the
    // weak-axis set is non-empty), so a span no LiDAR/IMU residual constrains cannot
    // float in the null space. Excluded from the observability and prior assembly so it
    // never inflates apparent observability.
    if (cfg_.motion_reg.enable && !weak_axes.empty()) {
      const ct::ImuExcitation exc =
          ct::imuExcitation(group.group.imu, anchor_bg_, anchor_ba_, gravity_->magnitude());
      const double floor = cfg_.motion_reg.excitation_floor;
      if (exc.n_omega <= floor && exc.n_accel <= floor) {
        const std::vector<Timestamp> mids = segmentMidpointTimes(t_begin, t_end);
        ct::addMotionRegularizer(problem, *spline_, mids, cfg_.motion_reg.weight,
                                 1.0 / std::max(weights.sigma_accel, 1e-9),
                                 1.0 / std::max(weights.sigma_gyro, 1e-9));
      }
    }

    // Sparse-direct photometric residuals into the SAME problem, against the visual
    // map carried from previous sweeps. Re-associated each outer round (the warp /
    // gate selection is linearisation-point dependent like the LiDAR planes). The
    // map is empty on the first sweep, so this is a no-op until promote() has run.
    if (vis != nullptr && vis->img != nullptr && vis->expo != nullptr && vmap_) {
      MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.visual", t_end);
      std::vector<ct::VisualUsedPoint> round_used;
      const ct::VisualAssocStats round_stats =
          ct::addVisualResiduals(problem, *spline_, cam_model_, T_fe_cam_, *vis->img, vis->t_image,
                                 *vis->expo, /*expo_index=*/0, *vmap_, cfg_.visual, &round_used);
      // Report the round with the most acceptances over the outer schedule rather than
      // the last round's count: re-association at a not-yet-converged pose can transiently
      // reject patches a converged pose accepts, so the peak is the faithful "how much
      // visual evidence did this sweep carry" signal. The overlay records ride along so
      // they describe exactly the patches that constrained this kept round.
      if (round_stats.accepted >= vis->stats.accepted) {
        vis->stats = round_stats;
        vis->used = std::move(round_used);
      }
      // Register the exposure block's positivity bound + prior pull. The prior anchors
      // the single in-window frame toward the exposure carried from the previous sweep,
      // realising the random-walk tie across the window. When exposure estimation is
      // off, or the camera intrinsics fix the prior (std 0), the block stays constant.
      if (round_stats.accepted > 0 && cfg_.visual.exposure_estimate_en && inv_expo_std_ > 0.0) {
        vis->expo->addTo(problem, inv_expo_, inv_expo_std_);
      } else if (problem.HasParameterBlock(vis->expo->block(0))) {
        problem.SetParameterBlockConstant(vis->expo->block(0));
      }
    }

    // Conservative GNSS absolute-position residuals into the SAME problem. Each gated
    // fix binds the antenna world position at the fix's own PTP time (the spline
    // interpolated to that time, never a nearest keyframe) to the ENU-converted fix,
    // under a Huber kernel so one in-gate-but-still-bad fix cannot dominate the cost.
    // Gating (quality floor, innovation, persistence) already ran once at the seeded
    // pose, so this only rebuilds the survivors' residuals each outer round.
    if (gnss != nullptr) {
      for (const GnssMeasurement& m : gnss->fixes) {
        ct::addGnssResidual(problem, *spline_, m.z_world, gnss->t_fe_ant, m.t_fix, m.sqrt_info,
                            kHuberGnss);
      }
    }

    // Gravity is pinned to the unit sphere; SO(3) knots carry the quaternion
    // manifold so their 4-vector storage updates in the 3-DoF tangent. Gravity is
    // then held constant: it is well determined from the static-start mean specific
    // force at bootstrap, while a single short, near-static window cannot separate a
    // gravity tilt from an orientation error and lets the tilt leak into a vertical
    // drift. Refining gravity is the documented excitation-gated seam.
    if (problem.HasParameterBlock(gravity_->data())) {
      if (problem.GetManifold(gravity_->data()) == nullptr) {
        problem.SetManifold(gravity_->data(), ct::makeGravityManifold());
      }
      problem.SetParameterBlockConstant(gravity_->data());
    }

    // Hold the unsupported tail knots constant. A knot at index >= interval(t_end)+4
    // has exactly zero basis weight at every t <= t_end; the one at interval(t_end)+3
    // peaks at u_end^3/6, which for a sweep ending just past a knot boundary is ~1e-5
    // -- not "weakly observed" but a pure lever arm: the solver can move it radians to
    // chase millimetres of residual elsewhere. Both classes hold their IMU warm-start
    // values (extendTo / reseedFrom) and unfreeze naturally on a later sweep once data
    // genuinely covers them -- the leading edge is dead-reckoning until corrected,
    // matching the filter's predict-then-update causality.
    {
      constexpr double kMinSupportWeight = 0.02;
      const SplineWindow::SegmentRef seg_end = spline_->segmentFor(t_end);
      const double tail_weight = seg_end.u * seg_end.u * seg_end.u / 6.0;
      const int lead_end = spline_->leadingKnotIndex(t_end);
      const int first_pinned = lead_end + (tail_weight < kMinSupportWeight ? 3 : 4);
      for (int j = first_pinned; j < spline_->numKnots(); ++j) {
        for (double* p : {spline_->so3KnotData(j), spline_->r3KnotData(j)}) {
          if (problem.HasParameterBlock(p)) {
            problem.SetParameterBlockConstant(p);
          }
        }
      }
    }
    std::vector<double*> params;
    problem.GetParameterBlocks(&params);
    for (double* p : params) {
      if (problem.ParameterBlockSize(p) == 4 && problem.GetManifold(p) == nullptr) {
        problem.SetManifold(p, new ceres::EigenQuaternionManifold());
      }
    }

    // Bias freeze seam. The accel bias is always frozen: a single bias knot cannot
    // separate a constant accel bias from gravity, so an unpinned accel bias absorbs
    // the gravity/acceleration ambiguity and drives a vertical drift; it re-enables
    // only once a multi-knot random walk ties consecutive knots (not yet built — one
    // bias knot spans the window). The gyro bias unfreezes under strong rotational
    // excitation, where the gyro residual senses it directly so its box bound becomes
    // live; otherwise it too is held to avoid solver jitter. The seam stays closed by

    // Both biases are free states: the gyro residual senses b_g directly, and with
    // gravity pinned the accel residual's DC error is exactly what b_a must absorb
    // (only the sum b_a + R^T*(gravity error) is observable, and for odometry the
    // sum is all that matters). Wander is constrained by physics, not gating: the
    // random-walk ties bound the increment per unit time, the marginalization prior
    // carries the estimate across sweeps, and the box bounds cap the absolute value.
    // A free block's stored value is first projected into its box (Ceres requires
    // the initial iterate to satisfy the bounds).
    for (int k = 0; k < bias_->numKnots(); ++k) {
      double* gp = bias_->gyroBlock(k);
      double* ap = bias_->accelBlock(k);
      if (problem.HasParameterBlock(gp) && bias_gyr_max > 0.0) {
        for (int a = 0; a < 3; ++a) {
          gp[a] = std::clamp(gp[a], -bias_gyr_max, bias_gyr_max);
          problem.SetParameterLowerBound(gp, a, -bias_gyr_max);
          problem.SetParameterUpperBound(gp, a, bias_gyr_max);
        }
      }
      if (problem.HasParameterBlock(ap) && bias_acc_max > 0.0) {
        for (int a = 0; a < 3; ++a) {
          ap[a] = std::clamp(ap[a], -bias_acc_max, bias_acc_max);
          problem.SetParameterLowerBound(ap, a, -bias_acc_max);
          problem.SetParameterUpperBound(ap, a, bias_acc_max);
        }
      }
    }

    // Marginalization prior (if any). Once a prior exists it pins the gauge.
    if (prior_) {
      problem.AddResidualBlock(prior_->makeCost(), nullptr, [&] {
        std::vector<double*> ptrs;
        ptrs.reserve(prior_->blocks().size());
        for (const auto& b : prior_->blocks()) {
          ptrs.push_back(b.ptr);
        }
        return ptrs;
      }());
    } else {
      // Gauge fix while no prior exists: pin knots of the window's leading active
      // segment. We anchor on the interior support knots (so3_knots[1], r3_knots[1..2])
      // rather than the leading [0] pair, leaving that oldest knot free so slideWindow()
      // can marginalize it and seed the prior, which then takes over the gauge.
      const SplineWindow::SegmentRef seg = spline_->segmentFor(t_begin);
      if (problem.HasParameterBlock(seg.so3_knots[1])) {
        problem.SetParameterBlockConstant(seg.so3_knots[1]);
      }
      for (int j = 1; j <= 2; ++j) {
        if (problem.HasParameterBlock(seg.r3_knots[j])) {
          problem.SetParameterBlockConstant(seg.r3_knots[j]);
        }
      }
    }

    {
      MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.solve", t_end);
      // The deadline is suspended until the minimum-iterations floor is met: while
      // total_iters < min_iters_floor the inner solve runs deadline-free so the floor
      // is always honoured even under a very tight budget.
      const bool floor_met = total_iters >= min_iters_floor;
      if (total_budget_s > 0.0 && floor_met) {
        const double spent = ms_since(solve_start) * 1e-3;
        const double remaining = total_budget_s - spent;
        // A non-positive remaining budget means the schedule is already over the
        // deadline; give the inner solve a vanishing slice so it returns the current
        // iterate immediately.
        sopts.max_solver_time_in_seconds = remaining > 1e-4 ? remaining : 1e-4;
      } else {
        sopts.max_solver_time_in_seconds = std::numeric_limits<double>::max();
      }
      ceres::Solver::Summary summary;
      ceres::Solve(sopts, &problem, &summary);
      total_iters += static_cast<int>(summary.iterations.size());
      if (total_budget_s > 0.0 && total_iters >= min_iters_floor &&
          ms_since(solve_start) * 1e-3 >= total_budget_s) {
        deadline_hit = true;
      }
    }

    // A bias the solve drove to its box edge means the bound bit this pass; record it
    // so the bias_bounded telemetry reflects a live clamp, not just a pre-solve seed
    // already at the edge.
    {
      for (int k = 0; k < bias_->numKnots(); ++k) {
        for (auto [bp, bmax] : {std::pair{bias_->gyroBlock(k), bias_gyr_max},
                                std::pair{bias_->accelBlock(k), bias_acc_max}}) {
          if (bmax <= 0.0 || !problem.HasParameterBlock(bp) ||
              problem.IsParameterBlockConstant(bp)) {
            continue;
          }
          for (int a = 0; a < 3; ++a) {
            if (std::abs(bp[a]) >= bmax - 1e-9) bound_engaged = true;
          }
        }
      }
    }

    // Early stop once associations have stabilised: if the keyframe-time pose moved
    // less than the translation AND rotation thresholds since the previous pass, a
    // re-association would not change the correspondence set. Disabled in the
    // deterministic path so the full schedule always runs.
    if (!deterministic_ && spline_->covers(t_end)) {
      const Pose cur = spline_->pose(t_end);
      if (have_prev_assoc) {
        const Pose rel = prev_assoc_pose.inverse() * cur;
        const double d_trans = rel.t.norm();
        const double d_rot = Sophus::SO3d(rel.q).log().norm();
        if (d_trans < cfg_.assoc_shift_thresh_m &&
            d_rot < cfg_.assoc_shift_thresh_deg * M_PI / 180.0) {
          break;
        }
      }
      prev_assoc_pose = cur;
      have_prev_assoc = true;
    }

    // Hitting the deadline ends the schedule with the best iterate so far.
    if (deadline_hit) {
      break;
    }
  }

  // Pose-block marginal covariance from the solved posterior (LiDAR + IMU + prior),
  // read here while the Ceres problem is alive; emitKeyframe prefers it over the
  // LiDAR-only marginal. Must precede slideWindow, which may re-linearize the problem.
  Eigen::Matrix<double, 6, 6> pose_cov;
  have_pose_cov_ = poseMarginalFromProblem(problem, t_end, &pose_cov);
  if (have_pose_cov_) {
    last_pose_cov_ = pose_cov;
  }

  // Slide the window on the final, solved problem.
  slideWindow(problem, t_begin, t_end);

  diag_.iterations = total_iters;
  diag_.outer_iters = outer_passes;
  diag_.effective_points = accepted;
  diag_.deadline_hit = deadline_hit;

  if (telemetry_) {
    if (telemetry_->enabled("frontend/deadline_hit")) {
      telemetry_->scalar("frontend/deadline_hit", deadline_hit ? 1.0 : 0.0, t_end);
    }
    if (telemetry_->enabled("frontend/bias_bounded")) {
      telemetry_->scalar("frontend/bias_bounded", bound_engaged ? 1.0 : 0.0, t_end);
    }
  }
  return accepted;
}

void CtFrontEnd::slideWindow(ceres::Problem& problem, Timestamp t_begin, Timestamp t_end) {
  // Per-sweep fixed-lag marginalization. The just-solved problem holds only this
  // sweep's residuals plus the previous prior, so a knot can transfer real LiDAR/IMU
  // information into the prior only while those residuals still touch it. A knot's
  // residual support is the cubic B-spline's 4-knot span; the leading knot of the
  // segment at t_begin is the oldest knot this sweep's factors reach, and it is the
  // one that drops out of the support as the window steps to the next sweep. So we
  // marginalize THAT knot (not the long-dead one near minTime, which no live residual
  // touches): fromSchur then captures the sweep's information on the boundary, and the
  // resulting prior keeps the next-oldest knot connected, so the chain self-sustains.
  // This is a regularizing per-sweep prior, not a fully-persistent information chain:
  // information older than the support already lives in the prior and is carried, not
  // re-marginalized from the original measurements.
  const int window_knots = std::max(cfg_.spline.window_knots, kSplineOrder);
  // Hold off until the trajectory has grown longer than the information window so the
  // dropped knot is genuinely exiting rather than still needed to keep the active set
  // full. The horizon is measured against the FIXED trajectory start (front-trimming
  // advances minTime() each sweep, so minTime() can no longer stand in for it). Also
  // require t_begin to have advanced past the last slide so a knot is dropped at most
  // once (consecutive sweeps advance t_begin by one knot interval).
  const Timestamp horizon = t_end - static_cast<Timestamp>(window_knots) * window_dt_ns_;
  if (horizon <= traj_start_t_ || t_begin <= window_floor_t_) {
    return;
  }
  const int lead = spline_->leadingKnotIndex(t_begin);

  // The oldest knot still touched by this sweep's residuals: the leading knot of the
  // segment covering t_begin. It is marginalizable precisely because the just-solved
  // problem still references it as a parameter block of those residuals.
  const SplineWindow::SegmentRef seg = spline_->segmentFor(t_begin);
  std::vector<double*> drop_ptrs = {seg.so3_knots[0], seg.r3_knots[0]};

  // Bias knots whose bracket lies wholly behind the current window front retire with
  // this slide. They join the Schur drop so the random-walk tie linking them to the
  // surviving knot folds into the prior instead of being discarded; their blocks are
  // popped from the timeline after the prior is built. The decision tests the
  // pre-trim front (one sweep of lag), so it precedes the prior build safely.
  int bias_retire = 0;
  while (bias_->numKnots() - bias_retire > 2 &&
         bias_->knotTime(bias_retire + 1) <= spline_->minTime()) {
    drop_ptrs.push_back(bias_->gyroBlock(bias_retire));
    drop_ptrs.push_back(bias_->accelBlock(bias_retire));
    ++bias_retire;
  }

  // A gauge-pinned knot has no tangent freedom to eliminate; never marginalize it.
  for (double* p : drop_ptrs) {
    if (problem.HasParameterBlock(p) && problem.IsParameterBlockConstant(p)) {
      window_floor_t_ = t_begin;
      return;
    }
  }

  std::vector<ceres::ResidualBlockId> residuals;
  std::unordered_set<const void*> res_seen;
  for (double* p : drop_ptrs) {
    if (!problem.HasParameterBlock(p)) {
      continue;
    }
    std::vector<ceres::ResidualBlockId> rs;
    problem.GetResidualBlocksForParameterBlock(p, &rs);
    for (ceres::ResidualBlockId r : rs) {
      if (res_seen.insert(r).second) {
        residuals.push_back(r);
      }
    }
  }
  if (residuals.empty()) {
    // Nothing live touches the leaving knot (e.g. it is still gauge-pinned); advance
    // the floor so the next sweep targets the following knot.
    window_floor_t_ = t_begin;
    return;
  }

  // The Markov blanket: every parameter block any of those residuals touches. The
  // dropped knots go last in the order so fromSchur eliminates the tail; everything
  // else is kept boundary. The blanket is gathered in first-seen order over the
  // deterministic (residual id, then passed-block) sequence rather than as a pointer
  // set, so the Schur block order -- and hence the prior's layout and the next solve --
  // is independent of heap addresses (which differ run to run once knots are trimmed).
  std::unordered_set<double*> drop_set(drop_ptrs.begin(), drop_ptrs.end());
  std::vector<double*> blanket;
  std::unordered_set<double*> blanket_seen;
  for (ceres::ResidualBlockId r : residuals) {
    std::vector<double*> ps;
    problem.GetParameterBlocksForResidualBlock(r, &ps);
    for (double* p : ps) {
      if (blanket_seen.insert(p).second) {
        blanket.push_back(p);
      }
    }
  }

  auto makeBlock = [&](double* p) {
    ct::MarginalizationPrior::Block b;
    b.ptr = p;
    b.global_size = problem.ParameterBlockSize(p);
    b.local_size = problem.ParameterBlockTangentSize(p);
    return b;
  };

  std::vector<ct::MarginalizationPrior::Block> kept;
  std::vector<ct::MarginalizationPrior::Block> dropped;
  int dropped_dim = 0;
  for (double* p : blanket) {
    if (drop_set.count(p)) {
      dropped.push_back(makeBlock(p));
      dropped_dim += dropped.back().local_size;
    } else if (!problem.IsParameterBlockConstant(p)) {
      // Constant gauge pins carry no information to transfer; skip them so the prior
      // is built only over free boundary blocks.
      kept.push_back(makeBlock(p));
    }
  }
  if (dropped.empty() || kept.empty()) {
    window_floor_t_ = t_begin;
    return;
  }

  // Schur input order is [kept | dropped]; fromSchur drops the dropped tail.
  std::vector<ct::MarginalizationPrior::Block> order = kept;
  order.insert(order.end(), dropped.begin(), dropped.end());

  Eigen::MatrixXd H;
  Eigen::VectorXd bvec;
  ct::linearizeBlocks(problem, residuals, order, &H, &bvec);
  std::unique_ptr<ct::MarginalizationPrior> next =
      ct::MarginalizationPrior::fromSchur(H, bvec, kept, dropped_dim);
  if (next) {
    prior_ = std::move(next);
    diag_.knots_marginalized += static_cast<int>(dropped.size());
    diag_.prior_residual_dim = prior_->residualDim();

    // Physically drop dead front knots — but never one the new prior still points at.
    // On a sweep that bridged an upstream gap, the residual span reaches back before
    // t_begin, so the Markov blanket (and hence the prior's kept blocks) can include
    // knots BEHIND the leading index; freeing those would leave the prior's parameter
    // blocks dangling and the next solve reading freed memory. The drop bound is the
    // minimum of the leading index and the lowest knot index any kept block references;
    // later sweeps trim the remainder once the prior no longer reaches it.
    std::vector<const double*> kept_ptrs;
    kept_ptrs.reserve(prior_->blocks().size());
    for (const auto& b : prior_->blocks()) {
      kept_ptrs.push_back(b.ptr);
    }
    const int min_kept = spline_->lowestKnotIndexOf(kept_ptrs);
    spline_->dropOldest(std::min(lead, min_kept));

    // Pop the retired bias knots: they were Schur-dropped above, so the new prior
    // cannot reference them; the lowestKnotIndexOf guard stays as a hard invariant.
    const int bias_kept = bias_->lowestKnotIndexOf(kept_ptrs);
    bias_->dropOldest(std::min(bias_retire, bias_kept));
  }
  window_floor_t_ = t_begin;
}

void CtFrontEnd::updateMap(const std::vector<LidarPoint>& scan_ds, Timestamp t0_scan) {
  MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.map", last_solved_t_);
  // The validity filter caps point range well below this, so a world point farther
  // than the bound from the sweep-end body position can only come from a pathological
  // trajectory warp. Keeping it out of the map stops one bad solve from poisoning
  // every later association.
  constexpr double kMaxInsertRangeM = 200.0;
  std::vector<Eigen::Vector3d> world;
  world.reserve(scan_ds.size());
  int rejected = 0;
  for (const LidarPoint& p : scan_ds) {
    const Timestamp t = t0_scan + p.t_offset_ns;
    if (!spline_->covers(t)) {
      continue;
    }
    const Eigen::Vector3d p_fe = T_fe_lidar_ * p.xyz.cast<double>();
    const Eigen::Vector3d p_w = spline_->pose(t) * p_fe;
    if ((p_w - anchor_pose_.t).norm() > kMaxInsertRangeM) {
      ++rejected;
      continue;
    }
    world.push_back(p_w);
  }
  if (rejected > 0 && telemetry_) {
    telemetry_->scalar("frontend/map/insert_rejected", static_cast<double>(rejected),
                       last_solved_t_);
  }
  if (!world.empty()) {
    map_->insert(world);
  }
  // Segment the ikd-Tree to a cube around the current body so map RAM and the per-point
  // nearest-neighbour search depth stay bounded over a long mission.
  map_->trimAround(anchor_pose_.t, cfg_.lidar.local_map_cube_m);
}

void CtFrontEnd::publishPatchOverlay(const cv::Mat& intensity, Timestamp t_image,
                                     const std::vector<ct::VisualUsedPoint>& used) {
  if (intensity.empty() || intensity.type() != CV_8UC1) {
    return;
  }

  ImageOverlay ov;
  ov.frame = Frame::CamLink;
  ov.width = intensity.cols;
  ov.height = intensity.rows;
  ov.encoding = ImageOverlay::Encoding::Mono8;
  ov.base =
      std::span<const std::uint8_t>(intensity.data, static_cast<std::size_t>(intensity.total()));
  ov.patches.reserve(used.size());
  for (const ct::VisualUsedPoint& u : used) {
    ImageOverlay::Patch patch;
    patch.uv = u.uv.cast<float>();
    patch.residual = u.residual;
    patch.depth = u.depth;
    patch.level = u.level;
    ov.patches.push_back(patch);
  }
  telemetry_->image("frontend/visual/patches", ov, t_image);
}

Eigen::Matrix<double, 6, 6> CtFrontEnd::poseMarginal(const Pose& T_world_body,
                                                     const std::vector<ct::LidarHit>& hits) const {
  // Accumulate the point-to-plane information block in the keyframe-pose tangent,
  // translation-first [rho; phi]. Each hit contributes a rank-1 outer product of
  // its 1x6 Jacobian row weighted by the hit's sqrt-information squared.
  const Eigen::Matrix3d R = T_world_body.R();
  Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();
  for (const ct::LidarHit& h : hits) {
    if (!h.plane.valid) {
      continue;
    }
    // q = R_fe_l p_l + t_fe_l, the point in the body frame; the residual is
    // n . (R q + p) + d, so dr/drho = n^T, dr/dphi = -n^T R [q]x.
    const Eigen::Vector3d q = T_fe_lidar_ * h.p_lidar;
    Eigen::Matrix<double, 1, 6> J;
    J.block<1, 3>(0, 0) = h.plane.n.transpose();
    J.block<1, 3>(0, 3) = -(h.plane.n.transpose() * R * skew(q));
    info += (h.weight * h.weight) * (J.transpose() * J);
  }
  // A small isotropic prior keeps the block invertible under weak excitation.
  info += 1e-6 * Eigen::Matrix<double, 6, 6>::Identity();
  return info.ldlt().solve(Eigen::Matrix<double, 6, 6>::Identity());
}

bool CtFrontEnd::poseMarginalFromProblem(ceres::Problem& problem, Timestamp stamp,
                                         Eigen::Matrix<double, 6, 6>* cov) {
  if (!spline_->covers(stamp)) {
    return false;
  }
  // The pose at stamp depends only on its segment's 4 SO(3) + 4 R^3 knots. Collect
  // them in a fixed tangent order (so3 then r3, 3 DoF each), and gather every residual
  // in the solved problem that touches any of them -- this is their Markov blanket and
  // carries the LiDAR, IMU, bias, and prior information that constrains the pose.
  const SplineWindow::SegmentRef seg = spline_->segmentFor(stamp);
  std::array<double*, kSplineOrder * 2> pose_knots{};
  std::array<bool, kSplineOrder * 2> knot_is_so3{};
  for (int j = 0; j < kSplineOrder; ++j) {
    pose_knots[j] = seg.so3_knots[j];
    knot_is_so3[j] = true;
    pose_knots[kSplineOrder + j] = seg.r3_knots[j];
    knot_is_so3[kSplineOrder + j] = false;
  }

  std::vector<ceres::ResidualBlockId> residuals;
  std::unordered_set<const void*> res_seen;
  for (double* p : pose_knots) {
    if (!problem.HasParameterBlock(p)) {
      return false;  // a pose knot is missing from the problem: cannot trust the block
    }
    std::vector<ceres::ResidualBlockId> rs;
    problem.GetResidualBlocksForParameterBlock(p, &rs);
    for (ceres::ResidualBlockId r : rs) {
      if (res_seen.insert(r).second) {
        residuals.push_back(r);
      }
    }
  }
  if (residuals.empty()) {
    return false;
  }

  // Order the blanket with the 8 pose knots first (free, never gauge-pinned at the
  // window's trailing edge), then the remaining free blocks the residuals reach.
  // Constant gauge pins carry no information and are dropped from the order.
  auto makeBlock = [&](double* p) {
    ct::MarginalizationPrior::Block b;
    b.ptr = p;
    b.global_size = problem.ParameterBlockSize(p);
    b.local_size = problem.ParameterBlockTangentSize(p);
    return b;
  };
  std::unordered_set<double*> pose_set(pose_knots.begin(), pose_knots.end());
  std::vector<ct::MarginalizationPrior::Block> order;
  int pose_dim = 0;
  for (double* p : pose_knots) {
    if (problem.IsParameterBlockConstant(p)) {
      return false;  // a pinned pose knot has no posterior spread; defer to the fallback
    }
    order.push_back(makeBlock(p));
    pose_dim += order.back().local_size;
  }
  std::unordered_set<double*> seen(pose_set);
  for (ceres::ResidualBlockId r : residuals) {
    std::vector<double*> ps;
    problem.GetParameterBlocksForResidualBlock(r, &ps);
    for (double* p : ps) {
      if (seen.count(p) || problem.IsParameterBlockConstant(p)) {
        continue;
      }
      seen.insert(p);
      order.push_back(makeBlock(p));
    }
  }

  Eigen::MatrixXd H;
  Eigen::VectorXd b;
  ct::linearizeBlocks(problem, residuals, order, &H, &b);

  // Joint marginal covariance of the blanket; the pose-knot marginal is its leading
  // pose_dim x pose_dim block (a sub-block of the joint covariance is the marginal).
  Eigen::MatrixXd cov_full;
  if (!robustInverse(H, &cov_full)) {
    return false;
  }
  const Eigen::MatrixXd cov_knots = cov_full.topLeftCorner(pose_dim, pose_dim);

  // Spline pose Jacobian J (6 x pose_dim) at stamp, by central finite difference of
  // T_W_Fe(stamp) under each knot's tangent perturbation, in the same knot ordering.
  // The pose tangent is the decoupled translation-first chart [delta t_world; delta
  // phi_body] that the LiDAR Jacobian and the rotation-first reorder both assume.
  const Pose T0 = spline_->pose(stamp);
  const Eigen::Quaterniond q0_inv = T0.q.conjugate();
  Eigen::MatrixXd Jpose = Eigen::MatrixXd::Zero(6, pose_dim);
  const double eps = 1e-6;
  int col = 0;
  for (int k = 0; k < kSplineOrder * 2; ++k) {
    double* ptr = pose_knots[k];
    if (knot_is_so3[k]) {
      Eigen::Map<Eigen::Quaterniond> qk(ptr);
      const Eigen::Quaterniond saved = qk;
      for (int a = 0; a < 3; ++a) {
        Eigen::Vector3d d = Eigen::Vector3d::Zero();
        d(a) = eps;
        qk = (saved * Sophus::SO3d::exp(d).unit_quaternion()).normalized();
        const Pose Tp = spline_->pose(stamp);
        qk = (saved * Sophus::SO3d::exp(-d).unit_quaternion()).normalized();
        const Pose Tm = spline_->pose(stamp);
        qk = saved;
        // Body-frame right-perturbation tangent of each sample relative to T0, then a
        // central difference: phi_+- = Log(q0^{-1} q_+-), J = (phi_+ - phi_-)/(2 eps).
        const Eigen::Vector3d phi_p = Sophus::SO3d((q0_inv * Tp.q).normalized()).log();
        const Eigen::Vector3d phi_m = Sophus::SO3d((q0_inv * Tm.q).normalized()).log();
        Jpose.block<3, 1>(0, col) = (Tp.t - Tm.t) / (2.0 * eps);
        Jpose.block<3, 1>(3, col) = (phi_p - phi_m) / (2.0 * eps);
        ++col;
      }
    } else {
      Eigen::Map<Eigen::Vector3d> pk(ptr);
      const Eigen::Vector3d saved = pk;
      for (int a = 0; a < 3; ++a) {
        pk = saved;
        pk(a) = saved(a) + eps;
        const Pose Tp = spline_->pose(stamp);
        pk(a) = saved(a) - eps;
        const Pose Tm = spline_->pose(stamp);
        pk = saved;
        const Eigen::Vector3d phi_p = Sophus::SO3d((q0_inv * Tp.q).normalized()).log();
        const Eigen::Vector3d phi_m = Sophus::SO3d((q0_inv * Tm.q).normalized()).log();
        Jpose.block<3, 1>(0, col) = (Tp.t - Tm.t) / (2.0 * eps);
        Jpose.block<3, 1>(3, col) = (phi_p - phi_m) / (2.0 * eps);
        ++col;
      }
    }
  }

  *cov = Jpose * cov_knots * Jpose.transpose();
  *cov = 0.5 * (*cov + cov->transpose()).eval();
  return true;
}

ObservabilityReport CtFrontEnd::computeObservability(const Pose& T_world_body,
                                                     const std::vector<ct::LidarHit>& hits) const {
  // Eigen-decompose the pose information block, assign each eigenvalue's
  // conditioning score = lambda/(lambda+kappa) to the unassigned axis its
  // eigenvector most aligns with, kept translation-first [tx,ty,tz,rx,ry,rz].
  const Eigen::Matrix3d R = T_world_body.R();
  Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();
  for (const ct::LidarHit& h : hits) {
    if (!h.plane.valid) {
      continue;
    }
    const Eigen::Vector3d q = T_fe_lidar_ * h.p_lidar;
    Eigen::Matrix<double, 1, 6> J;
    J.block<1, 3>(0, 0) = h.plane.n.transpose();
    J.block<1, 3>(0, 3) = -(h.plane.n.transpose() * R * skew(q));
    info += (h.weight * h.weight) * (J.transpose() * J);
  }

  ObservabilityReport obs;
  obs.frame = Frame::Body;
  const double kappa = cfg_.degeneracy_thresh > 0 ? cfg_.degeneracy_thresh : 10.0;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(info);
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

bool CtFrontEnd::keyframeDue(const Pose& T_world_body, Timestamp stamp) const {
  if (!have_prev_kf_) {
    return true;  // first keyframe after the first converged solve
  }
  const Pose rel = prev_kf_pose_.inverse() * T_world_body;
  const double dist = rel.t.norm();
  const double rot_deg = Sophus::SO3d(rel.q).log().norm() * 180.0 / M_PI;
  const double elapsed_s = to_seconds(stamp - prev_kf_stamp_);
  return dist >= cfg_.keyframe.dist_m || rot_deg >= cfg_.keyframe.rot_deg ||
         elapsed_s >= cfg_.keyframe.time_s;
}

void CtFrontEnd::emitKeyframe(const Pose& T_world_body, Timestamp stamp,
                              const std::vector<LidarPoint>& scan_ds, Timestamp t0_scan,
                              const std::vector<ct::LidarHit>& hits) {
  KeyframePacket pkt;
  pkt.id = next_kf_id_++;
  pkt.stamp = stamp;
  pkt.ref_frame = Frame::Odom;
  pkt.T_ref_body = T_world_body;
  pkt.calib_version = calib_version_;
  pkt.frontend_kind = 1;
  pkt.kinematics_included = false;
  pkt.observability = last_obs_;

  // Prefer the window-posterior pose marginal (carries the IMU and prior information
  // L3 relies on being baked into the relative edge); fall back to the LiDAR-only
  // marginal only when the posterior extraction failed for this sweep.
  const Eigen::Matrix<double, 6, 6> pose_cov =
      have_pose_cov_ ? last_pose_cov_ : poseMarginal(T_world_body, hits);

  if (!have_prev_kf_) {
    pkt.constraint_kind = KeyframePacket::ConstraintKind::AbsolutePrior;
    pkt.rel_to_id = 0;
    pkt.T_relto_this = Pose{};
    pkt.constraint_cov.form = GaussianBlock<6>::Form::Covariance;
    pkt.constraint_cov.M = IekfFrontEnd::reorderTransRotToRotTrans(pose_cov);
  } else {
    pkt.constraint_kind = KeyframePacket::ConstraintKind::RelativeBetween;
    pkt.rel_to_id = prev_kf_id_;
    pkt.T_relto_this = prev_kf_pose_.inverse() * T_world_body;
    // Conservative relative cov: the sum of the two absolute pose marginals upper-
    // bounds the true relative marginal (which subtracts their cross-covariance).
    const Eigen::Matrix<double, 6, 6> rel_cov = pose_cov + prev_kf_pose_cov_;
    pkt.constraint_cov.form = GaussianBlock<6>::Form::Covariance;
    pkt.constraint_cov.M = IekfFrontEnd::reorderTransRotToRotTrans(rel_cov);
  }

  // cloud_body: the downsampled scan warped to the body frame at stamp. Each point
  // goes to world via its own spline time T(t_i), then back into the body frame at
  // stamp via T(stamp)^{-1}.
  const Pose T_world_body_inv = T_world_body.inverse();
  auto cloud = std::make_shared<PointCloud>();
  cloud->reserve(scan_ds.size());
  for (const LidarPoint& p : scan_ds) {
    const Timestamp t = t0_scan + p.t_offset_ns;
    if (!spline_->covers(t)) {
      continue;
    }
    const Eigen::Vector3d p_world = spline_->pose(t) * (T_fe_lidar_ * p.xyz.cast<double>());
    const Eigen::Vector3d p_body = T_world_body_inv * p_world;
    LidarPoint lp;
    lp.xyz = p_body.cast<float>();
    cloud->push_back(lp);
  }
  pkt.cloud_body = std::move(cloud);

  prev_kf_pose_ = T_world_body;
  prev_kf_stamp_ = stamp;
  prev_kf_pose_cov_ = pose_cov;
  prev_kf_id_ = pkt.id;
  have_prev_kf_ = true;

  if (keyframe_sink_) {
    keyframe_sink_(std::move(pkt));
  }
}

void CtFrontEnd::ingest(const PreprocessedGroup& group) {
  const auto t_start = Clock::now();
  refreshExtrinsic();

  const MeasureGroup& mg = group.group;
  const Timestamp t_begin = mg.t_begin != 0 ? mg.t_begin : mg.scan.stamp_start;
  const Timestamp t_end = mg.t_end != 0 ? mg.t_end : mg.scan.stamp_start + mg.scan.sweep_duration;

  // ---- Bootstrap: defer until a static window passes the motion gate, then seed
  // gravity + bias + spline + map. Emits nothing while the window is not initialised.
  if (!bootstrapped_) {
    if (mg.imu.empty()) {
      return;  // need IMU to initialise gravity and the integration anchor
    }
    if (!imu_init_) {
      // Map init_time_s to a sample count from the first group's measured IMU period
      // so the static-window length is independent of the IMU rate.
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
    gravity_ = std::make_unique<ct::GravityBlock>(init.gravity, kGravityMag);
    anchor_bg_ = init.gyro_bias;
    anchor_ba_ = init.accel_bias;
    anchor_vel_ = Eigen::Vector3d::Zero();
    anchor_pose_ = Pose{};  // identity anchors the odom origin

    // Spline at identity; one coarse bias knot per window; the window opens at
    // t_begin and is extended to t_end so the first sweep has full coverage. The
    // spline carries the configured adaptive control-point cap; the first (static)
    // segment opens at a single control point and density follows motion thereafter.
    const int n_cp_max = cfg_.spline.n_cp_max > 0 ? cfg_.spline.n_cp_max : 1;
    spline_ = std::make_unique<SplineWindow>(window_dt_ns_, n_cp_max);
    spline_->initialize(t_begin, anchor_pose_);
    bias_ = std::make_unique<ct::BiasKnots>(t_begin, biasKnotDt(), 2);
    for (int k = 0; k < bias_->numKnots(); ++k) {
      bias_->setGyroKnot(k, anchor_bg_);
      bias_->setAccelKnot(k, anchor_ba_);
    }
    last_n_cp_ = 1;

    auto seed = buildSeed(mg.imu, t_begin, t_end);
    spline_->extendTo(t_end, seed, 1);
    bias_->extendTo(t_end);

    // Insert the cold-start deskew product (or the raw scan as a fallback) at the
    // identity pose so the map anchors odom; this is the only map seed before the
    // first solve.
    const bool deskewed = group.deskewed.has_value();
    const LidarScan& seed_scan = deskewed ? group.deskewed.value() : mg.scan;
    const std::vector<LidarPoint> ds =
        ct::voxelDownsample(validReturns(seed_scan), std::max(cfg_.lidar.voxel_map_m, 1e-3));
    std::vector<Eigen::Vector3d> world;
    world.reserve(ds.size());
    const Timestamp t0 = seed_scan.stamp_start;
    for (const LidarPoint& p : ds) {
      if (deskewed) {
        // The deskew already mapped LiDAR -> IMU/body@scan-end (extrinsic folded in);
        // with an identity anchor pose body == world, so the point is taken as-is.
        world.push_back(p.xyz.cast<double>());
      } else {
        // A raw fallback still carries the sensor extrinsic and per-point timing, so
        // place each point through its own spline pose with T_fe_lidar applied once.
        const Timestamp t = t0 + p.t_offset_ns;
        if (spline_->covers(t)) {
          world.push_back(spline_->pose(t) * (T_fe_lidar_ * p.xyz.cast<double>()));
        }
      }
    }
    if (!world.empty()) {
      map_->insert(world);
    }

    bootstrapped_ = true;
    last_solved_t_ = t_end;
    live_state_.T_world_body = anchor_pose_;
    live_state_.v_world = anchor_vel_;
    live_state_.b_g = anchor_bg_;
    live_state_.b_a = anchor_ba_;
    live_state_.g_world = gravity_->gravityWorld();
    live_state_.stamp = t_end;
    live_stamp_ = t_end;
    diag_.scan_time_ms = ms_since(t_start);
    return;
  }

  // Cleared per sweep; the gap guard below sets it for the sweep that restarts so the
  // diagnostic reflects this solve until the next one runs.
  diag_.restarted = false;

  // ---- Gap guard: the steady-state seed integrates this group's IMU forward from
  // last_solved_t_, which holds only when the IMU reaches back that far. A sweep
  // dropped upstream of the aggregation leaves no IMU hole: the next group's window
  // opens at the previous DELIVERED sweep's end, so its samples span the gap and the
  // seed integrates straight across it — the trajectory simply extends over the hole.
  // Only when the IMU itself starts after last_solved_t_ (a genuine data hole) is the
  // window reseeded; integrating across a span with no samples would extrapolate
  // garbage knots into the solve.
  const Duration sweep_span = t_end > t_begin ? t_end - t_begin : window_dt_ns_;
  const bool sweep_gap = t_begin > last_solved_t_ + sweep_span / 2;
  if (sweep_gap) {
    Duration imu_period = window_dt_ns_ / 20;
    if (mg.imu.size() >= 2) {
      imu_period = (mg.imu.back().stamp - mg.imu.front().stamp) /
                   static_cast<Duration>(mg.imu.size() - 1);
    }
    const bool imu_bridges =
        !mg.imu.empty() && mg.imu.front().stamp <= last_solved_t_ + 2 * imu_period;
    const double gap_ms = static_cast<double>(t_begin - last_solved_t_) / 1e6;
    if (imu_bridges) {
      if (telemetry_) {
        char msg[96];
        std::snprintf(msg, sizeof(msg), "sweep gap %.0f ms bridged by IMU", gap_ms);
        telemetry_->event(Level::Info, "frontend/sweep_gap_bridged", msg, t_begin);
      }
      // Fall through: buildSeed integrates from last_solved_t_ across the hole and
      // extendTo appends the missing segments from the integrated trail.
    } else {
      if (telemetry_) {
        char msg[96];
        std::snprintf(msg, sizeof(msg),
                      "sweep gap %.0f ms exceeds IMU horizon; window reseeded", gap_ms);
        telemetry_->event(Level::Warn, "frontend/window_restart", msg, t_begin);
      }
      reseedAfterGap(group, t_begin, t_end);
      diag_.restarted = true;
      diag_.scan_time_ms = ms_since(t_start);
      return;
    }
  }

  // ---- Steady state: extend, associate, solve, map, slide, keyframe.
  // Drive the new segment's control-point density from the group's peak per-sample IMU
  // excitation (with hysteresis) so aggressive motion gets denser knots and smooth
  // motion stays cheap; the constant-1 case is recovered when n_cp_max <= 1.
  const int n_cp = selectKnotDensity(mg.imu);
  auto seed = buildSeed(mg.imu, last_solved_t_, t_end);
  spline_->extendTo(t_end, seed, n_cp);
  bias_->extendTo(t_end);
  // Refresh the trailing knots the previous solve held constant (no real measurement
  // support; see the pinning rule in solveWindow): their stored values are pure
  // extrapolation, never entered the prior, and this sweep's IMU integration is a
  // strictly fresher prediction for that span.
  if (spline_->covers(last_solved_t_)) {
    constexpr double kMinSupportWeight = 0.02;
    const SplineWindow::SegmentRef seg_prev = spline_->segmentFor(last_solved_t_);
    const double tail_weight = seg_prev.u * seg_prev.u * seg_prev.u / 6.0;
    const int from =
        spline_->leadingKnotIndex(last_solved_t_) + (tail_weight < kMinSupportWeight ? 3 : 4);
    spline_->reseedFrom(from, seed);
  }

  const Timestamp t0_scan = mg.scan.stamp_start;
  std::vector<LidarPoint> scan_ds;
  std::vector<ct::LidarHit> hits;

  // ---- Visual stage setup (contract E). Build the intensity pyramid once for this
  // sweep, anchor the image to a spline-covered mid-exposure time, and seed a single-
  // frame exposure chain. The pyramid storage and the view that borrows it must outlive
  // the solve, so they live across the steady-state block. The disabled path (gate off,
  // invalid intrinsics, or no image) builds nothing and emits the one-time event.
  std::vector<cv::Mat> pyramid;
  std::unique_ptr<ct::ImagePyramidView> img_view;
  ct::ExposureChain expo(cfg_.visual);
  VisualContext vctx;
  const bool want_visual = cfg_.visual.enable && mg.image.has_value();
  if (want_visual && !visualEnabled(group) && !visual_disabled_logged_) {
    visual_disabled_logged_ = true;
    if (telemetry_) {
      telemetry_->event(Level::Warn, "frontend/visual/disabled",
                        "camera intrinsics invalid or extrinsic missing; running LIO only", t_end);
    }
    MERIDIAN_WARN(kLogModule, "event", "frontend/visual/disabled", "cam_valid",
                  cam_model_.valid() ? 1 : 0, "have_extrinsic", have_cam_extrinsic_ ? 1 : 0);
  }
  if (visualEnabled(group)) {
    pyramid = buildImagePyramid(*mg.image);
    if (!pyramid.empty()) {
      // The image mid-exposure time, clamped into the spline's covered span so the
      // photometric residual always evaluates against a defined pose.
      Timestamp t_img = mg.image->stamp;
      if (!spline_->covers(t_img)) {
        t_img = t_end;  // fall back to the sweep end, which the solve always covers
      }
      img_view = std::make_unique<ct::ImagePyramidView>(pyramid);
      expo.addFrame(t_img, inv_expo_);
      vctx.img = img_view.get();
      vctx.expo = &expo;
      vctx.t_image = t_img;
    }
  }

  // ---- GNSS stage setup. Gate the group's fixes against the seeded spline pose
  // (quality floor, innovation, persistence) before the solve so the survivors enter the
  // cost each outer round. The inactive path (config off or no fixes) allocates nothing.
  GnssContext gctx;
  gateGnss(group, t_end, &gctx);

  const auto t_solve = Clock::now();
  const int accepted =
      solveWindow(group, last_solved_t_, t_end, &scan_ds, &hits,
                  vctx.img != nullptr ? &vctx : nullptr, gctx.fixes.empty() ? nullptr : &gctx);
  diag_.solve_time_ms = ms_since(t_solve);
  have_solved_ = true;

  // The body pose / velocity at the sweep end drive the live state and become the
  // next sweep's integration anchor (decoupled from the solver storage).
  const Pose T_world_end = spline_->covers(t_end) ? spline_->pose(t_end) : anchor_pose_;
  const Eigen::Vector3d v_end =
      spline_->covers(t_end) ? spline_->linearVelocityWorld(t_end) : anchor_vel_;
  anchor_pose_ = T_world_end;
  anchor_vel_ = v_end;
  anchor_bg_ = bias_->gyroBiasAt(t_end);
  anchor_ba_ = bias_->accelBiasAt(t_end);

  last_obs_ = computeObservability(T_world_end, hits);
  diag_.observability = last_obs_;

  {
    const auto t_map = Clock::now();
    updateMap(scan_ds, t0_scan);
    diag_.deskew_time_ms = ms_since(t_map);
  }

  live_state_.T_world_body = T_world_end;
  live_state_.v_world = v_end;
  live_state_.b_g = anchor_bg_;
  live_state_.b_a = anchor_ba_;
  live_state_.g_world = gravity_->gravityWorld();
  live_state_.stamp = t_end;
  live_stamp_ = t_end;
  last_solved_t_ = t_end;

  // ---- Visual map lifecycle: carry the solved exposure forward, then
  // re-score / add observations against the now-solved pose, promote this sweep's
  // accepted hits to new visual points, and box-evict points that left the active box.
  // Promotion uses the accepted LiDAR hits (which carry world point + fitted plane
  // normal) so a freshly promoted point already has the normal the warp needs.
  if (vctx.img != nullptr && vmap_) {
    MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.visual_map", t_end);
    if (cfg_.visual.exposure_estimate_en && inv_expo_std_ > 0.0) {
      const double solved = expo.value(0);
      if (std::isfinite(solved) && solved > 0.0) {
        inv_expo_ = solved;
      }
    }
    const Timestamp t_img = vctx.t_image;
    const Pose T_w_fe = spline_->covers(t_img) ? spline_->pose(t_img) : T_world_end;
    const Pose T_w_c = T_w_fe * T_fe_cam_;
    // Tracked-patch overlay (gated): draw the patches captured from this sweep's kept
    // solve round before the map is mutated below, so the overlay shows exactly the
    // points that constrained the solve (not the post-promote/post-evict selection).
    if (telemetry_ && telemetry_->enabled("frontend/visual/patches") && !pyramid.empty()) {
      publishPatchOverlay(pyramid.front(), t_img, vctx.used);
    }

    vmap_->updateAfterSolve(*vctx.img, cam_model_, T_w_c, inv_expo_);
    vmap_->promote(*vctx.img, cam_model_, T_w_c, inv_expo_, hits);
    vmap_->evict(T_w_c.t);
  }

  if (telemetry_) {
    // n_tracked: patches that passed the warp/level gate into this frame; n_converged:
    // those that also passed NCC/SSD and contributed a photometric residual to the solve.
    if (telemetry_->enabled("frontend/visual/n_tracked")) {
      telemetry_->scalar("frontend/visual/n_tracked", static_cast<double>(vctx.stats.warped),
                         t_end);
    }
    // Funnel observability: map population and per-frame candidate count localize a
    // tracking dropout to promotion, candidacy gating, or the warp/NCC stages.
    if (vmap_ && telemetry_->enabled("frontend/visual/map_points")) {
      telemetry_->scalar("frontend/visual/map_points", static_cast<double>(vmap_->size()),
                         t_end);
    }
    if (telemetry_->enabled("frontend/visual/n_candidates")) {
      telemetry_->scalar("frontend/visual/n_candidates",
                         static_cast<double>(vctx.stats.candidates), t_end);
    }
    if (telemetry_->enabled("frontend/visual/n_converged")) {
      telemetry_->scalar("frontend/visual/n_converged", static_cast<double>(vctx.stats.accepted),
                         t_end);
    }
    if (telemetry_->enabled("frontend/visual/res_mean")) {
      const double res_mean = vctx.stats.accepted > 0
                                  ? vctx.stats.res_sum / static_cast<double>(vctx.stats.accepted)
                                  : 0.0;
      telemetry_->scalar("frontend/visual/res_mean", res_mean, t_end);
    }
    // The photometric model estimates a multiplicative inverse-exposure gain only; the
    // affine brightness pair is [a, b] = [inv_expo, 0] (no additive offset estimated).
    if (telemetry_->enabled("frontend/visual/exposure_gain")) {
      Eigen::Vector2d gain(inv_expo_, 0.0);
      telemetry_->vec("frontend/visual/exposure_gain", gain, t_end);
    }
  }

  if (keyframeDue(T_world_end, t_end)) {
    emitKeyframe(T_world_end, t_end, scan_ds, t0_scan, hits);
  }

  diag_.scan_time_ms = ms_since(t_start);

  if (telemetry_) {
    if (telemetry_->enabled("frontend/lidar/n_inlier")) {
      telemetry_->scalar("frontend/lidar/n_inlier", static_cast<double>(accepted), t_end);
    }
    if (telemetry_->enabled("frontend/iter_count")) {
      telemetry_->scalar("frontend/iter_count", static_cast<double>(diag_.iterations), t_end);
    }
    if (telemetry_->enabled("frontend/solve_ms")) {
      telemetry_->scalar("frontend/solve_ms", diag_.solve_time_ms, t_end);
    }
    if (telemetry_->enabled("frontend/outer_iters")) {
      telemetry_->scalar("frontend/outer_iters", static_cast<double>(diag_.outer_iters), t_end);
    }
    if (telemetry_->enabled("frontend/lidar/n_factors_kept")) {
      telemetry_->scalar("frontend/lidar/n_factors_kept", static_cast<double>(last_cap_stats_.kept),
                         t_end);
    }
    if (telemetry_->enabled("frontend/lidar/n_factors_dropped")) {
      telemetry_->scalar("frontend/lidar/n_factors_dropped",
                         static_cast<double>(last_cap_stats_.dropped), t_end);
    }
    if (telemetry_->enabled("frontend/ct/n_cp")) {
      telemetry_->scalar("frontend/ct/n_cp", static_cast<double>(last_n_cp_), t_end);
    }
    if (telemetry_->enabled("odom/body")) {
      telemetry_->pose("odom/body", T_world_end, Frame::Body, t_end);
    }
    if (telemetry_->enabled("frontend/lidar/inliers") && !hits.empty()) {
      PointCloud inliers;
      inliers.reserve(hits.size());
      for (const ct::LidarHit& h : hits) {
        if (!spline_->covers(h.t)) {
          continue;
        }
        const Eigen::Vector3d pw = spline_->pose(h.t) * (T_fe_lidar_ * h.p_lidar);
        LidarPoint lp;
        lp.xyz = pw.cast<float>();
        inliers.push_back(lp);
      }
      PointCloudView view{std::span<const LidarPoint>(inliers.data(), inliers.size())};
      telemetry_->cloud("frontend/lidar/inliers", view, Frame::Odom, t_end);
    }
    if (telemetry_->enabled("map/registered") && !scan_ds.empty()) {
      PointCloud reg;
      reg.reserve(scan_ds.size());
      for (const LidarPoint& p : scan_ds) {
        const Timestamp t = t0_scan + p.t_offset_ns;
        if (!spline_->covers(t)) {
          continue;
        }
        const Eigen::Vector3d pw = spline_->pose(t) * (T_fe_lidar_ * p.xyz.cast<double>());
        LidarPoint lp;
        lp.xyz = pw.cast<float>();
        reg.push_back(lp);
      }
      PointCloudView view{std::span<const LidarPoint>(reg.data(), reg.size())};
      telemetry_->cloud("map/registered", view, Frame::Odom, t_end);
    }
  }
}

void CtFrontEnd::ingest_imu_live(const ImuSample& imu) {
  if (!bootstrapped_) {
    return;
  }
  if (live_stamp_ == 0) {
    live_stamp_ = imu.stamp;
    return;
  }
  const double dt = to_seconds(imu.stamp - live_stamp_);
  if (dt <= 0.0) {
    return;
  }
  // Advance a private NavState copy only; solver state is untouched until the next
  // ingest() re-derives the live state from the solved spline.
  const Eigen::Vector3d omega = imu.gyro - live_state_.b_g;
  const Eigen::Vector3d acc_world =
      live_state_.T_world_body.q * (imu.acc - live_state_.b_a) + live_state_.g_world;
  live_state_.T_world_body.t += live_state_.v_world * dt + 0.5 * acc_world * dt * dt;
  live_state_.T_world_body.q =
      (live_state_.T_world_body.q * Sophus::SO3d::exp(omega * dt).unit_quaternion()).normalized();
  live_state_.v_world += acc_world * dt;
  live_state_.stamp = imu.stamp;
  live_stamp_ = imu.stamp;
}

void CtFrontEnd::apply_correction(const GraphUpdate& update) {
  if (update.moved.empty() || !have_prev_kf_) {
    return;
  }
  // Re-anchor odom: shift every spline control pose by the rigid correction implied
  // by the most recent moved keyframe. The local map is left in the old odom frame;
  // no map rebuild is performed (the map is re-anchored implicitly at the next
  // insert from the shifted trajectory).
  const GraphUpdate::Moved& m = update.moved.back();
  if (m.id != prev_kf_id_) {
    return;
  }
  const Pose delta = m.new_T_map_body * prev_kf_pose_.inverse();

  // Shift every spline control point by the world-frame rigid correction. For a
  // cumulative B-spline, left-acting all control points by delta left-acts the whole
  // curve: T_W_Fe(t) -> delta * T_W_Fe(t). SO(3) knots rotate; R^3 knots map through
  // the full rigid transform. Each resident knot is visited exactly once.
  if (spline_) {
    const Eigen::Matrix3d dR = delta.q.toRotationMatrix();
    const int n = spline_->numKnots();
    for (int i = 0; i < n; ++i) {
      Eigen::Map<Eigen::Quaterniond> q(spline_->so3KnotData(i));
      q = (delta.q * q).normalized();
      Eigen::Map<Eigen::Vector3d> p(spline_->r3KnotData(i));
      p = dR * p + delta.t;
    }
  }

  // Re-anchor odom: the integration anchor, the previous-keyframe pose, and the live
  // state all shift by the same delta. The local map is left in the old odom frame
  // (no rebuild); subsequent inserts re-anchor it implicitly from the shifted curve.
  anchor_pose_ = delta * anchor_pose_;
  anchor_vel_ = delta.q * anchor_vel_;
  prev_kf_pose_ = m.new_T_map_body;

  live_state_.T_world_body = delta * live_state_.T_world_body;
  live_state_.v_world = delta.q * live_state_.v_world;
}

}  // namespace meridian
