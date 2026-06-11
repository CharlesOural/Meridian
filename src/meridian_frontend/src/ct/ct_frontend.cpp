#include "ct/ct_frontend.hpp"

#include <ceres/crs_matrix.h>
#include <ceres/manifold.h>
#include <ceres/problem.h>
#include <ceres/solver.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <sophus/so3.hpp>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "ct/pose_marginal.hpp"
#include "meridian/common/cloud.hpp"
#include "meridian/common/cov_reorder.hpp"
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

// Tail-anchor strengths over the unmeasured span past the newest measurement: an
// honest constant-extrapolation uncertainty, weak enough for real measurements to
// override the moment they arrive. Shared by the window solve and the adaptive-
// density warm-start mini-solve so both brace the same tail identically.
constexpr double kTailSigmaVel = 0.2;   // [m/s]
constexpr double kTailSigmaRate = 0.1;  // [rad/s]

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
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
  sync_cov_ = deterministic_ || std::getenv("MERIDIAN_SYNC_KEYFRAME_COV") != nullptr;
  window_dt_ns_ = from_seconds(std::max(cfg_.spline.knot_dt_ms, 1.0) * 1e-3);
  gravity_ = std::make_unique<ct::GravityBlock>(Eigen::Vector3d(0, 0, -kGravityMag), kGravityMag);
  map_ = std::make_unique<ct::LidarLocalMap>(cfg_.lidar);
  live_state_.ref_frame = Frame::Odom;
  live_state_.g_world = gravity_->gravityWorld();
  if (calib) {
    set_calibration(std::move(calib));
  }
}

CtFrontEnd::~CtFrontEnd() {
  // Join the worker before any member it forwards into (the keyframe sink) is torn down.
  if (finalizer_) {
    finalizer_->stop();
  }
}

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
  // Live path: spin up the async covariance worker. The worker forwards finished packets
  // (constraint_cov filled) to the same sink. The deterministic path keeps the inline
  // synchronous path so replay stays bit-exact against the golden recording.
  if (!deterministic_) {
    if (finalizer_) {
      finalizer_->stop();
    }
    finalizer_ = std::make_unique<KeyframeFinalizer>(
        [this](KeyframePacket&& kf) {
          if (keyframe_sink_) {
            keyframe_sink_(std::move(kf));
          }
        },
        telemetry_);
    finalizer_->start();
  }
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

int CtFrontEnd::maxControlPoints() const {
  return cfg_.ncp.enabled ? std::max(cfg_.ncp.n_cp_max, 1) : 1;
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

int CtFrontEnd::firstKeptKnotIndex(Timestamp t_begin) const {
  // First resident knot index at or past the end of the segment containing t_begin.
  // Every knot strictly before it has its whole 4-knot residual support behind the
  // next window front, so the slide marginalizes the departing group up to this index.
  // Enumerated over indices so multi-knot segments resolve exactly.
  const Timestamp boundary_t = spline_->segmentEndContaining(t_begin);
  const int n = spline_->numKnots();
  for (int j = 0; j < n; ++j) {
    if (spline_->knotTime(j) >= boundary_t) {
      return j;
    }
  }
  return n;
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

  // The predicted hole has no covariance accounting, so no relative keyframe edge may
  // span it: the next keyframe breaks the chain with an AbsolutePrior. The stored
  // posterior marginal belongs to the pre-gap window and must not seed that anchor.
  kf_chain_broken_ = true;
  have_pose_cov_ = false;

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
    // Settle deferred push-downs before the next sweep's parallel association so its
    // concurrent searches only read resolved node flags.
    map_->flushPendingDeletes();
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
  const int n_cp_max = maxControlPoints();
  spline_ = std::make_unique<SplineWindow>(window_dt_ns_, n_cp_max, cfg_.ncp.enabled);
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
  sweep_max_gear_ = 1;
  // The pin index belonged to the discarded knot deque; invalid until the next solve.
  last_first_pinned_ = -1;
  last_solved_t_ = t_begin;
  window_floor_t_ = t_begin;
  last_prior_build_t_ = 0;
}

void CtFrontEnd::ncpExtend(const MeasureGroup& mg, Timestamp t_begin, Timestamp t_end) {
  const bool ncp_dbg = telemetry_ != nullptr && telemetry_->enabled("frontend/ncp/gear");

  // (1) IMU warm start: integrate this sweep's IMU forward from the anchor. The seed
  // is valid at arbitrary times in [t_begin, t_end] (constant-rate extrapolated past
  // both ends), so it can re-seed every re-laid and appended knot below.
  auto seed = buildSeed(mg.imu, t_begin, t_end);

  // (2) Tail re-knot. Segments laid wholly past the previous data end carried a
  // provisional gear (no samples had arrived; the gear was held). Now their data is
  // here: truncate them and let the extension below re-lay them at the gear their
  // own samples select, so every segment's FINAL gear is decided on its real data.
  // cut_t is the start of the first segment with t_real_start >= t_begin. A knot the
  // carried prior references must never be popped (its parameter block would
  // dangle); that can only happen after an unusual residual-support reach-back, so
  // skip the re-knot loudly and keep the provisional layout for this sweep.
  int relayed = 0;
  {
    const Timestamp seg_start = spline_->segmentStartContaining(t_begin);
    const Timestamp cut_t =
        seg_start >= t_begin ? seg_start : spline_->segmentEndContaining(t_begin);
    // Knots of the to-be-removed segments are exactly those with time > cut_t (each
    // segment's knots lie in (seg_start, seg_end]).
    int first_removed = spline_->numKnots();
    for (int j = 0; j < spline_->numKnots(); ++j) {
      if (spline_->knotTime(j) > cut_t) {
        first_removed = j;
        break;
      }
    }
    std::vector<const double*> held;
    if (prior_) {
      held.reserve(prior_->blocks().size());
      for (const auto& b : prior_->blocks()) {
        held.push_back(b.ptr);
      }
    }
    if (spline_->highestKnotIndexOf(held) >= first_removed) {
      if (telemetry_) {
        telemetry_->event(Level::Warn, "frontend/ncp/reknot_skipped",
                          "prior holds a tail knot; keeping the provisional layout", t_begin);
      }
    } else {
      relayed = spline_->truncateTailSegments(cut_t);
    }
  }

  // (3) Refresh the surviving trailing knots the previous solve held constant: their
  // values are pure extrapolation, never entered the prior, and this sweep's IMU
  // integration is a strictly fresher prediction. The stored pin index is invalid
  // (-1) directly after a restart, where there is nothing pinned to refresh.
  if (last_first_pinned_ >= 0 && last_first_pinned_ < spline_->numKnots()) {
    spline_->reseedFrom(last_first_pinned_, seed);
  }

  // (4) Extend with a per-segment gear decided from that segment's OWN samples
  // (t in (s0, min(s1, t_end)]). A segment past t_end has no data yet and holds the
  // carried gear; it is provisional and gets re-laid by step (2) next sweep, so the
  // hysteresis state advances only on finalized (data-carrying) segments, in lay
  // order.
  int max_finalized = 0;
  spline_->extendTo(t_end, seed, [&](Timestamp s0, Timestamp s1) -> int {
    const Timestamp hi = std::min(s1, t_end);
    const ct::ImuMeanExcitation stats =
        ct::segmentMeanExcitation(mg.imu, s0, hi, gravity_->magnitude());
    const int gear =
        ct::gearFromMeanExcitation(stats, cfg_.ncp.omega_thresh, cfg_.ncp.accel_thresh,
                                   cfg_.ncp.hysteresis, cfg_.ncp.n_cp_max, last_n_cp_);
    if (stats.n > 0) {
      if (gear != last_n_cp_ && telemetry_ != nullptr) {
        char msg[48];
        std::snprintf(msg, sizeof(msg), "gear %d -> %d", last_n_cp_, gear);
        telemetry_->event(Level::Info, "frontend/ncp/gear_change", msg, s1);
      }
      last_n_cp_ = gear;
      max_finalized = std::max(max_finalized, gear);
      if (ncp_dbg) {
        telemetry_->scalar("frontend/ncp/gear", static_cast<double>(gear), s1);
        telemetry_->scalar("frontend/ncp/mean_omega", stats.mean_omega, s1);
        telemetry_->scalar("frontend/ncp/mean_accel_dev", stats.mean_accel_dev, s1);
      }
    }
    return gear;
  });
  sweep_max_gear_ = max_finalized > 0 ? max_finalized : last_n_cp_;
  if (ncp_dbg) {
    telemetry_->scalar("frontend/ncp/relayed_knots", static_cast<double>(relayed), t_end);
  }

  // (5) Bias timeline on its own coarse cadence: the IMU residual brackets bias
  // blocks by time, independent of spline knot density.
  bias_->extendTo(t_end);

  // (6) Warm-start mini-solve over the freshly laid tail. The raw IMU-propagated
  // seed places each control point at an on-trajectory pose sample, which for a
  // dense (gear > 1) segment leaves a wiggly curve whose deskew corrupts data
  // association before the main solve can correct it. Fitting ONLY this sweep's IMU
  // residuals + the tail anchors, with everything except the freshly laid knots
  // held constant, irons the tail onto the inertial trajectory at a few LM
  // iterations' cost. Single-threaded, fixed iteration cap, no wall clock: a pure
  // function of the inputs on the deterministic path.
  //
  // The free boundary: after a successful re-knot every knot past the data front
  // (knotTime > t_begin) is freshly laid; on the skipped path nothing was re-laid
  // and only the knots the previous solve never supported (the stored pin index)
  // were reseeded, so only those may move.
  int free_from = -1;
  if (relayed > 0) {
    free_from = spline_->numKnots();
    for (int j = 0; j < spline_->numKnots(); ++j) {
      if (spline_->knotTime(j) > t_begin) {
        free_from = j;
        break;
      }
    }
  } else {
    free_from = last_first_pinned_;
  }
  int max_gear = 1;
  for (Timestamp t = t_begin; t < t_end;) {
    max_gear = std::max(max_gear, spline_->segmentGearContaining(t));
    const Timestamp e = spline_->segmentEndContaining(t);
    if (e <= t) {
      break;
    }
    t = e;
  }
  if (cfg_.ncp.warmstart_iters > 0 && max_gear > 1 && free_from >= 0 &&
      free_from < spline_->numKnots()) {
    MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.warmstart", t_end);
    double imu_dt_s = 0.005;
    if (mg.imu.size() >= 2) {
      imu_dt_s = to_seconds(mg.imu.back().stamp - mg.imu.front().stamp) /
                 static_cast<double>(mg.imu.size() - 1);
      imu_dt_s = std::max(imu_dt_s, 1e-4);
    }
    const ct::ImuWeights weights = imuWeights(imu_dt_s);

    ceres::Problem wp;
    ct::addImuResiduals(wp, *spline_, *bias_, *gravity_, mg.imu, t_begin, t_end, weights);
    const Timestamp stride = std::max<Timestamp>(window_dt_ns_ / 4, 1);
    std::vector<Timestamp> tail_times;
    for (int k = 1; k <= 8; ++k) {
      const Timestamp t = t_end + k * stride;
      if (!spline_->covers(t)) {
        break;
      }
      tail_times.push_back(t);
    }
    ct::addTailAnchors(wp, *spline_, tail_times, seed_vel_end_, seed_omega_end_, kTailSigmaVel,
                       kTailSigmaRate);

    std::vector<double*> params;
    wp.GetParameterBlocks(&params);
    for (double* p : params) {
      if (wp.ParameterBlockSize(p) == 4 && wp.GetManifold(p) == nullptr) {
        wp.SetManifold(p, new ceres::EigenQuaternionManifold());
      }
    }
    // Free exactly the freshly laid tail knots; biases, gravity, and every supported
    // knot hold their solved values (the set is queried for membership only, so it
    // feeds no ordering into the solve).
    std::unordered_set<const double*> free_blocks;
    for (int j = free_from; j < spline_->numKnots(); ++j) {
      free_blocks.insert(spline_->so3KnotData(j));
      free_blocks.insert(spline_->r3KnotData(j));
    }
    int n_free = 0;
    for (double* p : params) {
      if (free_blocks.count(p) == 0) {
        wp.SetParameterBlockConstant(p);
      } else {
        ++n_free;
      }
    }

    if (n_free > 0 && wp.NumResidualBlocks() > 0) {
      ceres::Solver::Options wopts;
      wopts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      wopts.max_num_iterations = cfg_.ncp.warmstart_iters;
      wopts.num_threads = 1;
      wopts.minimizer_progress_to_stdout = false;
      ceres::Solver::Summary ws;
      ceres::Solve(wopts, &wp, &ws);
      if (ncp_dbg) {
        telemetry_->scalar("frontend/ncp/warmstart_iters",
                           static_cast<double>(ws.iterations.size()), t_end);
        telemetry_->scalar("frontend/ncp/warmstart_cost_drop", ws.initial_cost - ws.final_cost,
                           t_end);
      }
    }
  }
}

void CtFrontEnd::steadyStateExtend(const MeasureGroup& mg, Timestamp t_begin, Timestamp t_end) {
  // The adaptive path only earns its tail re-knot when a segment can carry more than
  // one control point. At a cap of one every segment is forced to gear 1, so the
  // re-knot would re-lay gear-1 segments with raw extrapolation -- discarding the knot
  // values the previous solve refined, for a layout identical to what is already there.
  // The uniform extend is then both correct and bit-identical to the disabled build, so
  // route through it whenever the cap cannot exceed one (disabled, or n_cp_max <= 1).
  if (cfg_.ncp.enabled && maxControlPoints() > 1) {
    ncpExtend(mg, t_begin, t_end);
  } else {
    legacyExtend(mg, t_begin, t_end);
  }
}

void CtFrontEnd::legacyExtend(const MeasureGroup& mg, Timestamp t_begin, Timestamp t_end) {
  // Uniform single-knot extension: one outer segment per nominal cadence, every new
  // knot warm-started from this sweep's IMU integration.
  auto seed = buildSeed(mg.imu, t_begin, t_end);
  spline_->extendTo(t_end, seed, 1);
  bias_->extendTo(t_end);
  // Refresh the trailing knots the previous solve held constant (no real measurement
  // support; see the pinning rule in solveWindow): their stored values are pure
  // extrapolation, never entered the prior, and this sweep's IMU integration is a
  // strictly fresher prediction for that span.
  if (spline_->covers(t_begin)) {
    constexpr double kMinSupportWeight = 0.02;
    const SplineWindow::SegmentRef seg_prev = spline_->segmentFor(t_begin);
    const double tail_weight = seg_prev.u * seg_prev.u * seg_prev.u / 6.0;
    const int from =
        spline_->leadingKnotIndex(t_begin) + (tail_weight < kMinSupportWeight ? 3 : 4);
    spline_->reseedFrom(from, seed);
  }
}

CtFrontEnd::ExtendKnots CtFrontEnd::probe_extend_for_test(const MeasureGroup& mg, Timestamp t_begin,
                                                          Timestamp t_end, bool use_production) {
  // Swap independent clones into the live members so the chosen extend mutates only the
  // copy; the originals and every scalar the extend touches are restored before return,
  // letting a test drive both paths from one identical window state.
  std::unique_ptr<SplineWindow> spline_save = std::move(spline_);
  std::unique_ptr<ct::BiasKnots> bias_save = std::move(bias_);
  spline_ = spline_save->clone();
  bias_ = std::make_unique<ct::BiasKnots>(bias_save->clone());
  const int last_n_cp_save = last_n_cp_;
  const int sweep_max_gear_save = sweep_max_gear_;
  const Eigen::Vector3d seed_vel_save = seed_vel_end_;
  const Eigen::Vector3d seed_omega_save = seed_omega_end_;

  if (use_production) {
    steadyStateExtend(mg, t_begin, t_end);
  } else {
    legacyExtend(mg, t_begin, t_end);
  }

  ExtendKnots out;
  for (int i = 0; i < spline_->numKnots(); ++i) {
    out.knot_t.push_back(spline_->knotTime(i));
    const double* s = spline_->so3KnotData(i);
    out.so3.push_back({s[0], s[1], s[2], s[3]});
    const double* r = spline_->r3KnotData(i);
    out.r3.push_back({r[0], r[1], r[2]});
  }
  for (int k = 0; k < bias_->numKnots(); ++k) {
    out.bias_t.push_back(bias_->knotTime(k));
    const Eigen::Vector3d& g = bias_->gyroKnot(k);
    out.bias_gyro.push_back({g.x(), g.y(), g.z()});
    const Eigen::Vector3d& a = bias_->accelKnot(k);
    out.bias_accel.push_back({a.x(), a.y(), a.z()});
  }

  spline_ = std::move(spline_save);
  bias_ = std::move(bias_save);
  last_n_cp_ = last_n_cp_save;
  sweep_max_gear_ = sweep_max_gear_save;
  seed_vel_end_ = seed_vel_save;
  seed_omega_end_ = seed_omega_save;
  return out;
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

  // Position marginal of the current odometry (the last keyframe's pose covariance
  // translation block), added to the floored fix covariance in the innovation gate so a
  // long GNSS gap during which odometry drifted does not reject an otherwise good fix. On
  // the live path the worker produces this marginal; reading it here is stale by up to one
  // keyframe (the gate only widens with it, so the lag is harmless).
  Eigen::Matrix3d pose_marg = Eigen::Matrix3d::Zero();
  if (finalizer_) {
    if (finalizer_->have_pose_cov()) {
      pose_marg = finalizer_->last_pose_cov().topLeftCorner<3, 3>();
    }
  } else if (have_pose_cov_) {
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
  // The deterministic (replay) path stays single-threaded: Ceres parallel reductions
  // sum in a nondeterministic order, so >1 thread breaks bit-exact replay==live. The
  // live path parallelises the per-residual evaluation across the available cores.
  sopts.num_threads = (deterministic_ || force_single_thread_)
                          ? 1
                          : std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
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
  ct::LidarAssocStats assoc_stats;        // final executed round's association funnel

  // Solver-trace bookkeeping. The cheap scalars (cost_initial / final cost / last step
  // norm / termination) are kept every sweep; the per-LM-iteration vectors are
  // accumulated across the outer schedule only when the solver debug group is on, so
  // the off cost is one enabled() lookup.
  const bool solver_dbg =
      telemetry_ != nullptr && telemetry_->enabled("frontend/solver/iter_cost");
  double cost_initial = std::numeric_limits<double>::quiet_NaN();
  double cost_final = std::numeric_limits<double>::quiet_NaN();
  double last_step_norm = 0.0;
  int termination = -1;
  std::vector<double> iter_cost, iter_step, iter_grad, iter_tr;

  // Final-round build-input captures for the async covariance worker (live path). Each
  // is overwritten every round, so after the loop they hold the final round's values; the
  // capture is materialized into capture_ only on a keyframe sweep, below.
  const bool want_capture = !deterministic_;
  std::vector<ct::VisualPatchParams> cap_patches;
  std::vector<Timestamp> cap_tail_times;
  std::vector<Timestamp> cap_motion_times;
  bool cap_motion_engaged = false;
  int cap_first_pinned = std::numeric_limits<int>::max();
  int cap_gauge_knot_idx = -1;

  // Select the visible visual-map points once, at the seed pose: the O(map)
  // projection scan is the per-sweep cost driver, and the per-cell winners are stable
  // across cm-level pose refinement. Every outer round's residual build and the
  // post-solve map update then reuse this cache instead of re-scanning the map.
  if (vis != nullptr && vis->img != nullptr && vmap_ && spline_->covers(vis->t_image)) {
    MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.vmap.select", t_end);
    const Pose T_w_c_seed = spline_->pose(vis->t_image) * T_fe_cam_;
    vmap_->refreshVisibleCache(cam_model_, T_w_c_seed);
  }

  // Residual-family bookkeeping for the per-family telemetry probe below: the
  // half-open [begin, end) residual-block index range each family's adds occupy in
  // the problem, in insertion order. Re-recorded on every round (each round rebuilds
  // the problem from scratch), so after the loop the ranges describe exactly the
  // final round's problem.
  enum Family : int {
    kFamLidar = 0,
    kFamImu,
    kFamBias,
    kFamAnchor,
    kFamMotionReg,
    kFamVisual,
    kFamGnss,
    kFamPrior,
    kFamCount
  };
  static constexpr std::array<const char*, kFamCount> kFamilyName = {
      "lidar", "imu", "bias", "anchor", "motionreg", "visual", "gnss", "prior"};
  std::array<std::pair<int, int>, kFamCount> fam_range{};

  // IMU-innovation reference: the spline pose at t_end as the pure IMU-propagated
  // seed left it, captured at round-0 entry (NOT per round -- a per-round capture
  // would only measure the last re-association step). The emitted innovation is the
  // total correction the whole outer/inner schedule applied to the IMU prediction.
  const bool have_innov_seed = spline_->covers(t_end);
  const Pose T_seed_end = have_innov_seed ? spline_->pose(t_end) : Pose{};

  int outer_passes = 0;
  for (int round = 0; round < max_outer; ++round) {
    // Whole-sweep wall-clock gate: only START another association+solve round if
    // budget remains. The inner Ceres solve is already deadline-bounded, but the
    // assoc/build that precede it are not -- so without this a hard-geometry sweep
    // runs two full passes well past the budget and backs up Q_meas (which, dropped,
    // forces a window restart). Round 0 always runs so every sweep solves at least
    // once; later rounds (re-association refinement) yield to the deadline.
    if (round > 0 && total_budget_s > 0.0 && total_iters >= min_iters_floor &&
        ms_since(solve_start) * 1e-3 >= total_budget_s) {
      break;
    }
    ++outer_passes;
    // associate() appends, so start each round from an empty hit set evaluated at
    // the current (seeded, then solved) spline.
    hits->clear();
    {
      MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.assoc", t_end);
      assoc_stats =
          ct::associate(*spline_, T_fe_lidar_, *scan_ds, t0_scan, *map_, cfg_.lidar, hits);
      accepted = assoc_stats.accepted;
    }

    problem = ceres::Problem();
    fam_range.fill({0, 0});
    int fam_mark = 0;

    // Bound the LiDAR residual count: cap at max_lidar_factors by normal-stratified
    // subsample, exempting points whose normal aligns with a weak translation axis so
    // the cap never collapses the very axis observability protects. The outer-step
    // index rotates the per-stratum stride phase so the visited subset moves across
    // rounds while staying replay-identical.
    const auto t_assemble = Clock::now();
    const std::vector<Eigen::Vector3d> weak_axes = weakTranslationAxes(*hits);
    last_cap_stats_ =
        ct::capHitsByNormalStrata(*hits, cfg_.lidar, round, weak_axes, kWeakAxisCos, &capped_hits);

    // LiDAR point-to-plane residuals at the current spline (capped set).
    ct::addLidarResiduals(problem, *spline_, T_fe_lidar_, capped_hits, cfg_.lidar);
    fam_range[kFamLidar] = {fam_mark, problem.NumResidualBlocks()};
    // IMU derivative residuals over the new sweep span + bias random walk.
    fam_mark = problem.NumResidualBlocks();
    ct::addImuResiduals(problem, *spline_, *bias_, *gravity_, group.group.imu, t_begin, t_end,
                        weights);
    fam_range[kFamImu] = {fam_mark, problem.NumResidualBlocks()};
    fam_mark = problem.NumResidualBlocks();
    ct::addBiasRandomWalk(problem, *bias_, weights);
    fam_range[kFamBias] = {fam_mark, problem.NumResidualBlocks()};
    if (telemetry_) {
      telemetry_->timing("frontend.ct.assemble", ms_since(t_assemble), t_end);
    }

    // Tail anchors over the span past the newest measurement: tie the spline's
    // velocity and body rate there to the seed's constant extrapolation. The pinning
    // rule removes the strict null directions; the anchors additionally hold the
    // velocity read-out to the IMU prediction when the seed carries model error
    // (an unconverged bias), at sigmas a real measurement immediately overrides.
    {
      const Timestamp stride = std::max<Timestamp>(window_dt_ns_ / 4, 1);
      std::vector<Timestamp> tail_times;
      for (int k = 1; k <= 8; ++k) {
        const Timestamp t = t_end + k * stride;
        if (!spline_->covers(t)) {
          break;
        }
        tail_times.push_back(t);
      }
      fam_mark = problem.NumResidualBlocks();
      ct::addTailAnchors(problem, *spline_, tail_times, seed_vel_end_, seed_omega_end_,
                         kTailSigmaVel, kTailSigmaRate);
      fam_range[kFamAnchor] = {fam_mark, problem.NumResidualBlocks()};
      if (want_capture) {
        cap_tail_times = tail_times;
      }
    }

    // Under-excitation regularizer (off by default, config-gated): a low-weight jerk /
    // angular-acceleration pull on the segment knot-midpoints, engaged only when the
    // peak IMU excitation is below the floor AND a translation axis is degenerate (the
    // weak-axis set is non-empty), so a span no LiDAR/IMU residual constrains cannot
    // float in the null space. Excluded from the observability and prior assembly so it
    // never inflates apparent observability.
    bool round_motion_engaged = false;
    std::vector<Timestamp> round_motion_times;
    if (cfg_.motion_reg.enable && !weak_axes.empty()) {
      const ct::ImuExcitation exc =
          ct::imuExcitation(group.group.imu, anchor_bg_, anchor_ba_, gravity_->magnitude());
      const double floor = cfg_.motion_reg.excitation_floor;
      if (exc.n_omega <= floor && exc.n_accel <= floor) {
        const std::vector<Timestamp> mids = segmentMidpointTimes(t_begin, t_end);
        fam_mark = problem.NumResidualBlocks();
        ct::addMotionRegularizer(problem, *spline_, mids, cfg_.motion_reg.weight,
                                 1.0 / std::max(weights.sigma_accel, 1e-9),
                                 1.0 / std::max(weights.sigma_gyro, 1e-9));
        fam_range[kFamMotionReg] = {fam_mark, problem.NumResidualBlocks()};
        round_motion_engaged = true;
        round_motion_times = mids;
      }
    }
    if (want_capture) {
      cap_motion_engaged = round_motion_engaged;
      cap_motion_times = std::move(round_motion_times);
    }

    // Sparse-direct photometric residuals into the SAME problem, against the visual
    // map carried from previous sweeps. Re-associated each outer round (the warp /
    // gate selection is linearisation-point dependent like the LiDAR planes). The
    // map is empty on the first sweep, so this is a no-op until promote() has run.
    if (vis != nullptr && vis->img != nullptr && vis->expo != nullptr && vmap_) {
      MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.visual", t_end);
      fam_mark = problem.NumResidualBlocks();
      std::vector<ct::VisualUsedPoint> round_used;
      std::vector<ct::VisualPatchParams> round_patches;
      const ct::VisualAssocStats round_stats = ct::addVisualResiduals(
          problem, *spline_, cam_model_, T_fe_cam_, *vis->img, vis->t_image, *vis->expo,
          /*expo_index=*/0, *vmap_, cfg_.visual, &round_used,
          want_capture ? &round_patches : nullptr);
      if (want_capture) {
        cap_patches = std::move(round_patches);
      }
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
      // The visual family covers the photometric residuals plus the exposure prior.
      fam_range[kFamVisual] = {fam_mark, problem.NumResidualBlocks()};
    }

    // Conservative GNSS absolute-position residuals into the SAME problem. Each gated
    // fix binds the antenna world position at the fix's own PTP time (the spline
    // interpolated to that time, never a nearest keyframe) to the ENU-converted fix,
    // under a Huber kernel so one in-gate-but-still-bad fix cannot dominate the cost.
    // Gating (quality floor, innovation, persistence) already ran once at the seeded
    // pose, so this only rebuilds the survivors' residuals each outer round.
    if (gnss != nullptr) {
      fam_mark = problem.NumResidualBlocks();
      for (const GnssMeasurement& m : gnss->fixes) {
        ct::addGnssResidual(problem, *spline_, m.z_world, gnss->t_fe_ant, m.t_fix, m.sqrt_info,
                            kHuberGnss);
      }
      fam_range[kFamGnss] = {fam_mark, problem.NumResidualBlocks()};
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
      if (!cfg_.gravity_refine) {
        problem.SetParameterBlockConstant(gravity_->data());
      }
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
      // Carried for the next sweep's tail reseed / warm-start mini-solve; the slide's
      // front-trim re-bases it so it keeps addressing the same knots.
      last_first_pinned_ = first_pinned;
      if (want_capture) {
        cap_first_pinned = first_pinned;
      }
    }
    std::vector<double*> params;
    problem.GetParameterBlocks(&params);
    for (double* p : params) {
      if (problem.ParameterBlockSize(p) == 4 && problem.GetManifold(p) == nullptr) {
        problem.SetManifold(p, new ceres::EigenQuaternionManifold());
      }
    }

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
      fam_mark = problem.NumResidualBlocks();
      problem.AddResidualBlock(prior_->makeCost(), nullptr, [&] {
        std::vector<double*> ptrs;
        ptrs.reserve(prior_->blocks().size());
        for (const auto& b : prior_->blocks()) {
          ptrs.push_back(b.ptr);
        }
        return ptrs;
      }());
      fam_range[kFamPrior] = {fam_mark, problem.NumResidualBlocks()};
    } else if (cfg_.ncp.enabled) {
      // Gauge fix while no prior exists, adaptive density: pin by deque index on the
      // first KEPT knot of the t_begin boundary (so3[D], r3[D..D+1]). Every knot
      // before D belongs to the departing group, so anchoring at D leaves the whole
      // group free for the slide to marginalize and seed the prior. On a uniform grid
      // D is exactly the segment's interior support knot [1].
      const int D = firstKeptKnotIndex(t_begin);
      if (D < spline_->numKnots()) {
        double* sp = spline_->so3KnotData(D);
        if (problem.HasParameterBlock(sp)) {
          problem.SetParameterBlockConstant(sp);
        }
      }
      for (int j = D; j <= D + 1; ++j) {
        if (j < spline_->numKnots()) {
          double* rp = spline_->r3KnotData(j);
          if (problem.HasParameterBlock(rp)) {
            problem.SetParameterBlockConstant(rp);
          }
        }
      }
      if (want_capture) {
        cap_gauge_knot_idx = D;
      }
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
      // Round-0 entry cost is the cost of the IMU-seeded problem; the last round's
      // exit cost / step norm / termination describe the converged (or cut-off) state.
      if (std::isnan(cost_initial)) {
        cost_initial = summary.initial_cost;
      }
      cost_final = summary.final_cost;
      termination = static_cast<int>(summary.termination_type);
      if (!summary.iterations.empty()) {
        last_step_norm = summary.iterations.back().step_norm;
      }
      if (solver_dbg) {
        for (const ceres::IterationSummary& it : summary.iterations) {
          iter_cost.push_back(it.cost);
          iter_step.push_back(it.step_norm);
          iter_grad.push_back(it.gradient_norm);
          iter_tr.push_back(it.trust_region_radius);
        }
      }
      if (telemetry_) {
        telemetry_->timing("frontend.ct.solve.residual_eval",
                           summary.residual_evaluation_time_in_seconds * 1e3, t_end);
        telemetry_->timing("frontend.ct.solve.jacobian_eval",
                           summary.jacobian_evaluation_time_in_seconds * 1e3, t_end);
        telemetry_->timing("frontend.ct.solve.linear_solver",
                           summary.linear_solver_time_in_seconds * 1e3, t_end);
      }
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

  // IMU innovation: how far the converged solve moved the sweep-end pose away from
  // the pure IMU-propagated seed (round-0 entry; see T_seed_end above). A growing
  // translation/rotation innovation under steady motion means the measurements are
  // fighting the inertial prediction. Cheap, emitted every sweep.
  if (telemetry_ && have_innov_seed && spline_->covers(t_end)) {
    const Pose T_solved_end = spline_->pose(t_end);
    if (telemetry_->enabled("frontend/innov/trans_m")) {
      telemetry_->scalar("frontend/innov/trans_m", (T_solved_end.t - T_seed_end.t).norm(), t_end);
    }
    if (telemetry_->enabled("frontend/innov/rot_deg")) {
      // angle(R_seed^T R_solved): the geodesic rotation between seed and solution.
      const Eigen::Quaterniond dq = (T_seed_end.q.conjugate() * T_solved_end.q).normalized();
      telemetry_->scalar("frontend/innov/rot_deg", Sophus::SO3d(dq).log().norm() * 180.0 / M_PI,
                         t_end);
    }
  }

  // ---- Registration / solver / consistency telemetry on the just-solved state.
  // Runs BEFORE slideWindow so the spline still covers every hit and IMU stamp of the
  // sweep. The cheap health scalars are emitted every sweep; everything under a debug
  // group prefix (frontend/assoc|solver|deskew|spline/*) is hoisted behind one
  // enabled() probe, so an off group costs one lookup per sweep and builds nothing.
  if (telemetry_ != nullptr) {
    const double n_input = static_cast<double>(scan_ds->size());
    if (telemetry_->enabled("frontend/lidar/n_input")) {
      telemetry_->scalar("frontend/lidar/n_input", n_input, t_end);
    }
    if (telemetry_->enabled("frontend/lidar/inlier_ratio")) {
      telemetry_->scalar("frontend/lidar/inlier_ratio",
                         n_input > 0.0 ? static_cast<double>(accepted) / n_input : 0.0, t_end);
    }

    const bool assoc_dbg = telemetry_->enabled("frontend/assoc/res_p95");

    // Point-to-plane misfit of the final hit set evaluated at the SOLVED spline:
    // r = n . (T(t_i) * T_fe_lidar * p_lidar) + d, each point re-placed at its own
    // solved pose, so this is the residual the solve left, not the one association
    // started from. The assoc group additionally collects per-hit scratch vectors
    // for the p95 / histogram / fit-quality statistics.
    {
      double res_sum = 0.0;
      double res_max = 0.0;
      int n_res = 0;
      std::vector<double> abs_res, nn_dists, fit_rms;
      if (assoc_dbg) {
        abs_res.reserve(hits->size());
        nn_dists.reserve(hits->size());
        fit_rms.reserve(hits->size());
      }
      for (const ct::LidarHit& h : *hits) {
        if (!h.plane.valid || !spline_->covers(h.t)) {
          continue;
        }
        const Eigen::Vector3d p_w = spline_->pose(h.t) * (T_fe_lidar_ * h.p_lidar);
        const double r = std::abs(h.plane.n.dot(p_w) + h.plane.d);
        res_sum += r;
        res_max = std::max(res_max, r);
        ++n_res;
        if (assoc_dbg) {
          abs_res.push_back(r);
          nn_dists.push_back(h.plane.nn_max_dist);
          fit_rms.push_back(h.plane.rms);
        }
      }
      if (telemetry_->enabled("frontend/lidar/res_mean")) {
        telemetry_->scalar("frontend/lidar/res_mean",
                           n_res > 0 ? res_sum / static_cast<double>(n_res) : 0.0, t_end);
      }
      if (telemetry_->enabled("frontend/lidar/res_max")) {
        telemetry_->scalar("frontend/lidar/res_max", res_max, t_end);
      }
      if (assoc_dbg) {
        // Percentiles via nth_element on the scratch copies; the hit set itself is
        // never reordered, so downstream deterministic iteration is undisturbed.
        const auto p95 = [](std::vector<double>& v) -> double {
          if (v.empty()) {
            return 0.0;
          }
          const std::size_t idx = std::min((v.size() * 95) / 100, v.size() - 1);
          std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(idx), v.end());
          return v[idx];
        };
        const auto mean = [](const std::vector<double>& v) -> double {
          double s = 0.0;
          for (const double x : v) {
            s += x;
          }
          return v.empty() ? 0.0 : s / static_cast<double>(v.size());
        };
        telemetry_->scalar("frontend/assoc/n_attempted", assoc_stats.attempted, t_end);
        telemetry_->scalar("frontend/assoc/n_matched", assoc_stats.matched, t_end);
        telemetry_->scalar("frontend/assoc/nn_dist_mean", mean(nn_dists), t_end);
        telemetry_->scalar("frontend/assoc/plane_rms_mean", mean(fit_rms), t_end);
        telemetry_->scalar("frontend/assoc/res_p95", p95(abs_res), t_end);
        telemetry_->scalar("frontend/assoc/nn_dist_p95", p95(nn_dists), t_end);
        telemetry_->scalar("frontend/assoc/plane_rms_p95", p95(fit_rms), t_end);
        telemetry_->scalar("frontend/assoc/reject_nn", assoc_stats.reject_no_neighbors, t_end);
        telemetry_->scalar("frontend/assoc/reject_dist", assoc_stats.reject_too_far, t_end);
        telemetry_->scalar("frontend/assoc/reject_plane", assoc_stats.reject_not_planar, t_end);
        telemetry_->scalar("frontend/assoc/reject_score", assoc_stats.reject_score, t_end);

        // Fixed-bin |r| histogram (the residual-vs-time heatmap source): 16 bins
        // spanning [0, 2*plane_thresh], overflow clamped into the last bin so the
        // vector length and bin edges are constant across the run.
        constexpr int kResBins = 16;
        const double res_span = 2.0 * std::max(cfg_.lidar.plane_thresh, 1e-6);
        Eigen::VectorXd hist = Eigen::VectorXd::Zero(kResBins);
        for (const double r : abs_res) {
          const int b = static_cast<int>(r / res_span * kResBins);
          hist[std::clamp(b, 0, kResBins - 1)] += 1.0;
        }
        char axis[64];
        std::snprintf(axis, sizeof(axis), "abs_res_bins_0_to_%.3fm", res_span);
        telemetry_->vec("frontend/assoc/res_hist", hist, t_end, axis);

        // Kept-factor distribution by plane-normal stratum: how directionally
        // balanced the capped constraint set is (a starved bin = a weak direction).
        if (!last_cap_stats_.kept_per_stratum.empty()) {
          Eigen::VectorXd strata(
              static_cast<Eigen::Index>(last_cap_stats_.kept_per_stratum.size()));
          for (std::size_t i = 0; i < last_cap_stats_.kept_per_stratum.size(); ++i) {
            strata[static_cast<Eigen::Index>(i)] = last_cap_stats_.kept_per_stratum[i];
          }
          telemetry_->vec("frontend/assoc/strata_kept", strata, t_end,
                          strata.size() == 7 ? "+x,-x,+y,-y,+z,-z,obl" : nullptr);
        }
      }

      // Rejected-point cloud (heavy): downsampled scan points that did not become
      // hits, placed at their solved poses. Built only when the key is on.
      if (telemetry_->enabled("frontend/assoc/outliers")) {
        const auto key_of = [](std::int32_t off, std::uint16_t ring) {
          return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(off)) << 16) |
                 static_cast<std::uint64_t>(ring);
        };
        std::unordered_set<std::uint64_t> hit_keys;
        hit_keys.reserve(hits->size() * 2);
        for (const ct::LidarHit& h : *hits) {
          hit_keys.insert(key_of(h.t_offset_ns, h.ring));
        }
        PointCloud rejected;
        rejected.reserve(scan_ds->size());
        for (const LidarPoint& p : *scan_ds) {
          const Timestamp t = t0_scan + p.t_offset_ns;
          if (!spline_->covers(t) || hit_keys.count(key_of(p.t_offset_ns, p.ring)) > 0) {
            continue;
          }
          const Eigen::Vector3d pw = spline_->pose(t) * (T_fe_lidar_ * p.xyz.cast<double>());
          LidarPoint lp;
          lp.xyz = pw.cast<float>();
          rejected.push_back(lp);
        }
        PointCloudView view{std::span<const LidarPoint>(rejected.data(), rejected.size())};
        telemetry_->cloud("frontend/assoc/outliers", view, Frame::Odom, t_end);
      }
    }

    // Solver health (cheap, every sweep). dx_norm is the last LM step the schedule
    // took (near zero at convergence); cost_total the final windowed cost.
    if (telemetry_->enabled("frontend/cost_total") && std::isfinite(cost_final)) {
      telemetry_->scalar("frontend/cost_total", cost_final, t_end);
    }
    if (telemetry_->enabled("frontend/dx_norm")) {
      telemetry_->scalar("frontend/dx_norm", last_step_norm, t_end);
    }
    if (telemetry_->enabled("frontend/window_span_s")) {
      telemetry_->scalar("frontend/window_span_s",
                         to_seconds(spline_->maxTime() - spline_->minTime()), t_end);
    }
    if (solver_dbg) {
      if (std::isfinite(cost_initial)) {
        telemetry_->scalar("frontend/solver/cost_initial", cost_initial, t_end);
      }
      telemetry_->scalar("frontend/solver/termination", termination, t_end);
      const auto emit_trace = [&](const char* key, const std::vector<double>& v) {
        if (!v.empty()) {
          telemetry_->vec(key,
                          Eigen::Map<const Eigen::VectorXd>(
                              v.data(), static_cast<Eigen::Index>(v.size())),
                          t_end, "per_lm_iteration");
        }
      };
      emit_trace("frontend/solver/iter_cost", iter_cost);
      emit_trace("frontend/solver/iter_step", iter_step);
      emit_trace("frontend/solver/iter_grad", iter_grad);
      emit_trace("frontend/solver/iter_tr_radius", iter_tr);
    }

    // IMU-vs-spline consistency in physical units: the mean gyro / accel residual at
    // the group's IMU stamps against the solved spline's analytic derivatives under
    // the measurement model
    //   w_meas = omega_body(t) + b_g(t),  a_meas = R(t)^T (a_world(t) - g_W) + b_a(t).
    // A climb here is over/under-trust of the IMU or a gravity/extrinsic problem.
    if (telemetry_->enabled("frontend/imu/res_acc")) {
      const Eigen::Vector3d g_world = gravity_->gravityWorld();
      double sum_g = 0.0;
      double sum_a = 0.0;
      int n_imu = 0;
      for (const ImuSample& s : imu) {
        if (s.stamp < t_begin || s.stamp > t_end || !spline_->covers(s.stamp)) {
          continue;
        }
        const Pose T = spline_->pose(s.stamp);
        const Eigen::Vector3d w_pred =
            spline_->angularVelocityBody(s.stamp) + bias_->gyroBiasAt(s.stamp);
        const Eigen::Vector3d a_pred =
            T.q.conjugate() * (spline_->linearAccelWorld(s.stamp) - g_world) +
            bias_->accelBiasAt(s.stamp);
        sum_g += (s.gyro - w_pred).norm();
        sum_a += (s.acc - a_pred).norm();
        ++n_imu;
      }
      if (n_imu > 0) {
        telemetry_->scalar("frontend/imu/res_acc", sum_a / n_imu, t_end);
        telemetry_->scalar("frontend/imu/res_gyr", sum_g / n_imu, t_end);
      }
    }
    if (telemetry_->enabled("frontend/bias_acc")) {
      telemetry_->vec("frontend/bias_acc", bias_->accelBiasAt(t_end), t_end, "bx,by,bz");
      telemetry_->vec("frontend/bias_gyr", bias_->gyroBiasAt(t_end), t_end, "bx,by,bz");
    }
    if (telemetry_->enabled("frontend/grav_norm")) {
      telemetry_->scalar("frontend/grav_norm", gravity_->gravityWorld().norm(), t_end);
    }

    // Deskew / CT-placement quality: the correction the continuous-time placement
    // applies per point relative to freezing the whole sweep at the end pose,
    // ||T(t_i) p - T(t_end) p||, plus the body motion span across the sweep.
    if (telemetry_->enabled("frontend/deskew/corr_mean_m") && spline_->covers(t_end)) {
      const Pose T_end = spline_->pose(t_end);
      std::vector<double> corr;
      corr.reserve(scan_ds->size());
      double corr_sum = 0.0;
      double corr_max = 0.0;
      for (const LidarPoint& p : *scan_ds) {
        const Timestamp t = t0_scan + p.t_offset_ns;
        if (!spline_->covers(t)) {
          continue;
        }
        const Eigen::Vector3d p_fe = T_fe_lidar_ * p.xyz.cast<double>();
        const double c = (spline_->pose(t) * p_fe - T_end * p_fe).norm();
        corr_sum += c;
        corr_max = std::max(corr_max, c);
        corr.push_back(c);
      }
      if (!corr.empty()) {
        telemetry_->scalar("frontend/deskew/corr_mean_m",
                           corr_sum / static_cast<double>(corr.size()), t_end);
        telemetry_->scalar("frontend/deskew/corr_max_m", corr_max, t_end);
        const std::size_t idx = std::min((corr.size() * 95) / 100, corr.size() - 1);
        std::nth_element(corr.begin(), corr.begin() + static_cast<std::ptrdiff_t>(idx),
                         corr.end());
        telemetry_->scalar("frontend/deskew/corr_p95_m", corr[idx], t_end);
      }
      if (spline_->covers(t_begin)) {
        const Pose rel = spline_->pose(t_begin).inverse() * T_end;
        telemetry_->scalar("frontend/deskew/sweep_trans_m", rel.t.norm(), t_end);
        telemetry_->scalar("frontend/deskew/sweep_rot_deg",
                           Sophus::SO3d(rel.q).log().norm() * 180.0 / M_PI, t_end);
      }
    }
    // The 'pre' cloud: the sweep frozen at the end pose, the visual pair of the
    // CT-registered map/registered cloud (post). Heavy; built only when on.
    if (telemetry_->enabled("frontend/deskew/pre") && spline_->covers(t_end)) {
      const Pose T_end = spline_->pose(t_end);
      PointCloud pre;
      pre.reserve(scan_ds->size());
      for (const LidarPoint& p : *scan_ds) {
        const Eigen::Vector3d pw = T_end * (T_fe_lidar_ * p.xyz.cast<double>());
        LidarPoint lp;
        lp.xyz = pw.cast<float>();
        pre.push_back(lp);
      }
      PointCloudView view{std::span<const LidarPoint>(pre.data(), pre.size())};
      telemetry_->cloud("frontend/deskew/pre", view, Frame::Odom, t_end);
    }

    // Spline kinematics profiles: instantaneous vec3 read-outs at the sweep end plus
    // span-RMS statistics from ~100 Hz samples over the solved span. The jerk RMS
    // (forward difference of the analytic acceleration) is the vibration / knot-
    // cadence-aliasing instrument.
    if (telemetry_->enabled("frontend/spline/vel") && spline_->covers(t_end)) {
      telemetry_->vec("frontend/spline/vel", spline_->linearVelocityWorld(t_end), t_end,
                      "x,y,z");
      telemetry_->vec("frontend/spline/acc", spline_->linearAccelWorld(t_end), t_end, "x,y,z");
      telemetry_->vec("frontend/spline/omega", spline_->angularVelocityBody(t_end), t_end,
                      "wx,wy,wz");
      constexpr double kSpanSampleDt = 0.01;  // ~100 Hz
      const Duration step = from_seconds(kSpanSampleDt);
      double sum_a2 = 0.0;
      double sum_w2 = 0.0;
      double sum_j2 = 0.0;
      int n_samp = 0;
      int n_jerk = 0;
      Eigen::Vector3d prev_a = Eigen::Vector3d::Zero();
      bool have_prev = false;
      for (Timestamp t = t_begin; t <= t_end; t += step) {
        if (!spline_->covers(t)) {
          have_prev = false;
          continue;
        }
        const Eigen::Vector3d a = spline_->linearAccelWorld(t);
        sum_a2 += a.squaredNorm();
        sum_w2 += spline_->angularVelocityBody(t).squaredNorm();
        ++n_samp;
        if (have_prev) {
          sum_j2 += ((a - prev_a) / kSpanSampleDt).squaredNorm();
          ++n_jerk;
        }
        prev_a = a;
        have_prev = true;
      }
      if (n_samp > 0) {
        telemetry_->scalar("frontend/spline/acc_rms_span", std::sqrt(sum_a2 / n_samp), t_end);
        telemetry_->scalar("frontend/spline/omega_rms_span", std::sqrt(sum_w2 / n_samp), t_end);
      }
      if (n_jerk > 0) {
        telemetry_->scalar("frontend/spline/jerk_rms", std::sqrt(sum_j2 / n_jerk), t_end);
      }
    }

    // Solved-spline pose samples for the wrapper's /meridian/path aggregation. The
    // cadence walks forward from the last emitted sample, clamped to t_begin so a
    // bridged gap (no solved poses inside the hole) is skipped, never interpolated.
    if (cfg_.debug_path_sample_hz > 0.0 &&
        telemetry_->enabled("frontend/spline/path_sample")) {
      const Duration step = from_seconds(1.0 / cfg_.debug_path_sample_hz);
      Timestamp t = last_path_t_ == 0 ? t_begin : last_path_t_ + step;
      if (t < t_begin) {
        t = t_begin;
      }
      for (; t <= t_end; t += step) {
        if (!spline_->covers(t)) {
          continue;
        }
        telemetry_->pose("frontend/spline/path_sample", spline_->pose(t), Frame::Odom, t);
        last_path_t_ = t;
      }
    }
  }

  // Per-family residual statistics on the converged final-round problem (raw
  // whitened residuals: loss functions off, so the stats are comparable across
  // families). The build above recorded each family's residual-block index range;
  // Ceres returns residual blocks in insertion order (asserted by unit test), so
  // slicing the full block list by a range selects exactly that family. The extra
  // Problem::Evaluate sweep is gated to ~1 in 5 sweeps plus keyframe sweeps; with
  // the solver debug group on it runs every sweep (post-solve, so it can starve
  // nothing) and additionally reports each family's share of the total cost.
  ++resid_probe_counter_;
  if (telemetry_ != nullptr &&
      (solver_dbg || resid_probe_counter_ % 5 == 0 || keyframeDue(spline_->pose(t_end), t_end))) {
    std::vector<ceres::ResidualBlockId> all_blocks;
    problem.GetResidualBlocks(&all_blocks);
    ceres::Problem::EvaluateOptions eopts;
    eopts.apply_loss_function = false;
    eopts.num_threads = 1;
    std::vector<double> fam_res;
    std::array<double, kFamCount> fam_sum_sq{};
    std::array<bool, kFamCount> fam_present{};
    double total_sum_sq = 0.0;
    for (int f = 0; f < kFamCount; ++f) {
      const auto [fam_begin, fam_end] = fam_range[static_cast<std::size_t>(f)];
      if (fam_end <= fam_begin || fam_end > static_cast<int>(all_blocks.size())) {
        continue;  // family absent this sweep
      }
      eopts.residual_blocks.assign(all_blocks.begin() + fam_begin, all_blocks.begin() + fam_end);
      fam_res.clear();
      if (!problem.Evaluate(eopts, nullptr, &fam_res, nullptr, nullptr) || fam_res.empty()) {
        continue;
      }
      double sum_sq = 0.0;
      double sum_abs = 0.0;
      for (const double v : fam_res) {
        sum_sq += v * v;
        sum_abs += std::abs(v);
      }
      fam_sum_sq[static_cast<std::size_t>(f)] = sum_sq;
      fam_present[static_cast<std::size_t>(f)] = true;
      total_sum_sq += sum_sq;
      const double n_rows = static_cast<double>(fam_res.size());
      const char* fam_name = kFamilyName[static_cast<std::size_t>(f)];
      char key[64];
      std::snprintf(key, sizeof(key), "frontend/resid/%s/rms", fam_name);
      telemetry_->scalar(key, std::sqrt(sum_sq / n_rows), t_end);
      std::snprintf(key, sizeof(key), "frontend/resid/%s/mean_abs", fam_name);
      telemetry_->scalar(key, sum_abs / n_rows, t_end);
      std::snprintf(key, sizeof(key), "frontend/resid/%s/n_rows", fam_name);
      telemetry_->scalar(key, n_rows, t_end);
    }
    // Family share of the total unweighted-by-loss cost: which stream dominates the
    // solve. Squared-residual sums, so the shares add to one over present families.
    if (solver_dbg && total_sum_sq > 0.0) {
      for (int f = 0; f < kFamCount; ++f) {
        if (!fam_present[static_cast<std::size_t>(f)]) {
          continue;
        }
        char key[64];
        std::snprintf(key, sizeof(key), "frontend/resid/%s/cost_frac",
                      kFamilyName[static_cast<std::size_t>(f)]);
        telemetry_->scalar(key, fam_sum_sq[static_cast<std::size_t>(f)] / total_sum_sq, t_end);
      }
    }
  }

  // Conditioning of the converged Gauss-Newton normal equation H = J^T J, where J is
  // the analytic Jacobian Ceres returns at the solved point in the local (manifold
  // tangent) parameterization with loss functions applied -- i.e. exactly the linear
  // system the final LM step solved. The eigenvalue spread cond = lambda_max/lambda_min
  // (eigenvalues of the symmetric information matrix, in whitened/tangent units) is the
  // 2-norm condition number: a large value means one direction is near-unobservable
  // (the cost barely penalizes it), so the solve there is ill-posed. cond_pose repeats
  // the measure over just the SO(3) knot tangent block-columns (the J^T J diagonal block
  // of the rotation subspace, not its Schur-complement marginal), where a fast shake can
  // outrun the spline's representable rotation bandwidth. Gated on the solver debug group:
  // when off, no Evaluate and no eigen-decomposition run. Pure function of the converged
  // problem (single-threaded, no clocks/RNG), so it leaves replay bit-identical.
  if (telemetry_ != nullptr && telemetry_->enabled("frontend/solver/cond_number")) {
    // Eigenvalues of J^T J over a chosen set of (non-constant) parameter blocks. The
    // Jacobian rows span all residual blocks; its columns are the listed blocks' tangent
    // DoF. Eigenvalues are invariant to column order, so no block-to-column remap is
    // needed. Returns false when the block set is empty or the decomposition fails.
    const auto condEigs = [&](const std::vector<double*>& blocks, double* lambda_min,
                              double* lambda_max, Eigen::VectorXd* eig_out) -> bool {
      if (blocks.empty()) {
        return false;
      }
      ceres::Problem::EvaluateOptions eopts;
      eopts.parameter_blocks = blocks;
      eopts.apply_loss_function = true;
      eopts.num_threads = 1;
      ceres::CRSMatrix jac;
      if (!problem.Evaluate(eopts, nullptr, nullptr, nullptr, &jac) || jac.num_rows == 0 ||
          jac.num_cols == 0) {
        return false;
      }
      Eigen::MatrixXd J = Eigen::MatrixXd::Zero(jac.num_rows, jac.num_cols);
      for (int r = 0; r < jac.num_rows; ++r) {
        for (int idx = jac.rows[r]; idx < jac.rows[r + 1]; ++idx) {
          J(r, jac.cols[idx]) = jac.values[idx];
        }
      }
      const Eigen::MatrixXd H = J.transpose() * J;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H, Eigen::EigenvaluesOnly);
      if (es.info() != Eigen::Success) {
        return false;
      }
      *lambda_min = es.eigenvalues()(0);
      *lambda_max = es.eigenvalues()(es.eigenvalues().size() - 1);
      if (eig_out != nullptr) {
        *eig_out = es.eigenvalues();
      }
      return true;
    };

    std::vector<double*> all_blocks;
    problem.GetParameterBlocks(&all_blocks);
    std::vector<double*> active_blocks;
    active_blocks.reserve(all_blocks.size());
    for (double* p : all_blocks) {
      if (!problem.IsParameterBlockConstant(p)) {
        active_blocks.push_back(p);
      }
    }
    std::vector<double*> so3_blocks;
    so3_blocks.reserve(static_cast<std::size_t>(spline_->numKnots()));
    // Closest approach of any active knot to the current sweep's scan boundaries. When a
    // knot time coincides with a scan edge the returns on either side share almost no
    // segment support, leaving that knot's tangent block weakly constrained; tracking the
    // gap lets a conditioning spike be correlated with such a layout. Gap is in ns here.
    const Timestamp scan_lo = t0_scan;
    const Timestamp scan_hi = t0_scan + group.group.scan.sweep_duration;
    Duration min_knot_scan_gap = std::numeric_limits<Duration>::max();
    for (int j = 0; j < spline_->numKnots(); ++j) {
      double* p = spline_->so3KnotData(j);
      if (problem.HasParameterBlock(p) && !problem.IsParameterBlockConstant(p)) {
        so3_blocks.push_back(p);
        const Timestamp kt = spline_->knotTime(j);
        const Duration g_lo = kt > scan_lo ? kt - scan_lo : scan_lo - kt;
        const Duration g_hi = kt > scan_hi ? kt - scan_hi : scan_hi - kt;
        min_knot_scan_gap = std::min(min_knot_scan_gap, std::min(g_lo, g_hi));
      }
    }

    // lambda_min is floored only inside the ratio so a near-zero or numerically negative
    // smallest eigenvalue (a true null direction) cannot blow the ratio to inf/NaN; the
    // raw lambda_min is logged unclamped so that genuine sign/scale stays visible.
    constexpr double kCondFloor = 1e-30;
    double lmin = 0.0;
    double lmax = 0.0;
    Eigen::VectorXd eigs;
    if (condEigs(active_blocks, &lmin, &lmax, &eigs)) {
      telemetry_->scalar("frontend/solver/lambda_min", lmin, t_end);
      telemetry_->scalar("frontend/solver/lambda_max", lmax, t_end);
      telemetry_->scalar("frontend/solver/cond_number", lmax / std::max(lmin, kCondFloor), t_end);

      // The global position, yaw, and gravity directions are unobservable (gauge freedom),
      // so the smallest eigenvalues sit at numerical zero and pin cond_number near inf even
      // on a clean trajectory. Split that null cluster off with a relative-gap floor scaled
      // to the spectrum's top, then measure conditioning over the observable subspace alone
      // -- the quantity that actually rises when fast motion outruns the spline. Eigenvalues
      // ascend, so the first one above the floor is the weakest observable mode and the
      // count at or below it is the effective null dimension: nominally the fixed gauge
      // count, a rise meaning an observable mode is collapsing into the null space.
      const double tol = lmax * 1e-9;
      int null_dim = 0;
      while (null_dim < eigs.size() && eigs(null_dim) <= tol) {
        ++null_dim;
      }
      const double lmin_pos = null_dim < eigs.size() ? eigs(null_dim) : lmax;
      telemetry_->scalar("frontend/solver/lambda_min_pos", lmin_pos, t_end);
      telemetry_->scalar("frontend/solver/null_dim", static_cast<double>(null_dim), t_end);
      telemetry_->scalar("frontend/solver/cond_obs", lmax / std::max(lmin_pos, kCondFloor), t_end);
    }
    double lmin_p = 0.0;
    double lmax_p = 0.0;
    if (condEigs(so3_blocks, &lmin_p, &lmax_p, nullptr)) {
      telemetry_->scalar("frontend/solver/cond_pose", lmax_p / std::max(lmin_p, kCondFloor), t_end);
    }
    if (min_knot_scan_gap != std::numeric_limits<Duration>::max()) {
      telemetry_->scalar("frontend/solver/min_knot_scan_gap_ms",
                         to_seconds(min_knot_scan_gap) * 1e3, t_end);
    }
  }

  // Pose-block marginal covariance from the solved posterior (LiDAR + IMU + prior),
  // read here while the Ceres problem is alive; emitKeyframe prefers it over the
  // LiDAR-only marginal. Must precede slideWindow, which may re-linearize the problem.
  // Recovering this marginal is a sparse back-substitution over the whole window -- the
  // costliest post-solve step, and unlike the solve it is not deadline-bounded. It is
  // consumed only by emitKeyframe (the keyframe-edge covariance) and, stale-tolerant, by
  // the GNSS innovation gate, so compute it only on sweeps that actually emit a keyframe.
  // Between keyframes the stored marginal stands in for the gate; emitKeyframe recomputes
  // (or falls back) here whenever a keyframe is due. The predicate is pure and re-checked
  // before emitKeyframe, so a disagreement only wastes/skips this recovery, never the edge.
  // Deterministic path: compute the marginal INLINE on the live problem, exactly as
  // before, so replay stays bit-exact against the golden recording. Live path: capture
  // the final round's build inputs (cheap value copies) so the async worker can rebuild a
  // bit-identical problem off T2 and recover the same marginal a keyframe interval later.
  capture_.valid = false;
  if (keyframeDue(spline_->pose(t_end), t_end)) {
    if (sync_cov_) {
      MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.posecov", t_end);
      Eigen::Matrix<double, 6, 6> pose_cov;
      have_pose_cov_ = poseMarginalFromProblem(problem, t_end, &pose_cov);
      if (have_pose_cov_) {
        last_pose_cov_ = pose_cov;
      }
    } else {
      MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.covsnapshot", t_end);
      captureFinalRound(t_begin, t_end, imu, weights, capped_hits, cap_patches,
                        vis != nullptr && vis->img != nullptr ? vis : nullptr, gnss,
                        cap_motion_engaged, cap_motion_times, cap_tail_times, cap_first_pinned,
                        cap_gauge_knot_idx);
    }

    // Bit-exact parity probe (tests only): compute the marginal both ways on the
    // identical solved state -- (a) the synchronous routine on the live problem, (b) the
    // worker routine on the freshly rebuilt snapshot clone -- and record both. The probe
    // evaluates at an interior, well-supported stamp one knot cadence back from t_end:
    // the bit-exactness under test is a property of the rebuilt problem and is
    // stamp-independent, and the interior stamp avoids the trailing pinned-knot edge so
    // the marginal is actually recovered (a fall-back at t_end would test nothing).
    if (parity_probe_ && capture_.valid) {
      ParityProbeResult res;
      res.absolute_prior = !have_prev_kf_ || kf_chain_broken_;
      const Timestamp probe_stamp =
          spline_->covers(t_end - window_dt_ns_) ? t_end - window_dt_ns_ : t_end;
      Eigen::Matrix<double, 6, 6> cov_a;
      res.sync_ok = ct::poseMarginalFromProblem(problem, *spline_, probe_stamp, &cov_a);
      res.sync_cov = cov_a;

      // Independent clones so the probe never disturbs the snapshot the worker will use.
      auto probe_spline = capture_.spline_clone->clone();
      auto probe_bias = std::make_unique<ct::BiasKnots>(capture_.bias_clone->clone());
      ct::GravityBlock probe_gravity = capture_.gravity_clone;
      std::unique_ptr<ct::MarginalizationPrior> probe_prior;
      if (capture_.prior_clone) {
        SplineWindow* ps = probe_spline.get();
        ct::BiasKnots* pb = probe_bias.get();
        SplineWindow* cs = capture_.spline_clone.get();
        ct::BiasKnots* cb = capture_.bias_clone.get();
        const int nk = cs->numKnots();
        const int nb = cb->numKnots();
        probe_prior = capture_.prior_clone->cloneRemapped([&](double* p) -> double* {
          for (int i = 0; i < nk; ++i) {
            if (p == cs->so3KnotData(i)) return ps->so3KnotData(i);
            if (p == cs->r3KnotData(i)) return ps->r3KnotData(i);
          }
          for (int i = 0; i < nb; ++i) {
            if (p == cb->gyroBlock(i)) return pb->gyroBlock(i);
            if (p == cb->accelBlock(i)) return pb->accelBlock(i);
          }
          return p;
        });
      }
      ct::WindowProblemInputs probe_in = capture_.inputs;
      probe_in.spline = probe_spline.get();
      probe_in.bias = probe_bias.get();
      probe_in.gravity = &probe_gravity;
      probe_in.prior = probe_prior.get();
      probe_in.capped_hits = &capture_.capped_hits;
      probe_in.imu = &capture_.imu;
      ceres::Problem probe_problem;
      ct::WindowExposureScratch probe_expo(probe_in.visual_cfg);
      ct::WindowExposureScratch* probe_expo_ptr =
          probe_in.visual_patches.empty() ? nullptr : &probe_expo;
      ct::buildWindowProblem(probe_problem, probe_in, probe_expo_ptr);
      Eigen::Matrix<double, 6, 6> cov_b;
      res.worker_ok =
          ct::poseMarginalFromProblem(probe_problem, *probe_spline, probe_stamp, &cov_b);
      res.worker_cov = cov_b;
      parity_probe_results_.push_back(res);
    }
  }

  // Slide the window on the final, solved problem.
  {
    MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.marg", t_end);
    slideWindow(problem, t_begin, t_end);
  }

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
    // Solved-state evolution at the sweep end (bias spline and velocity read-outs).
    if (telemetry_->enabled("frontend/state/bias_gyr_norm")) {
      telemetry_->scalar("frontend/state/bias_gyr_norm", bias_->gyroBiasAt(t_end).norm(), t_end);
    }
    if (telemetry_->enabled("frontend/state/bias_acc_norm")) {
      telemetry_->scalar("frontend/state/bias_acc_norm", bias_->accelBiasAt(t_end).norm(), t_end);
    }
    if (telemetry_->enabled("frontend/state/vel_norm") && spline_->covers(t_end)) {
      telemetry_->scalar("frontend/state/vel_norm", spline_->linearVelocityWorld(t_end).norm(),
                         t_end);
    }
    // Prior health after the slide: the residual dimension of the prior the NEXT
    // sweep will solve under (0 = chain broken / not yet started).
    if (telemetry_->enabled("frontend/prior/dim")) {
      telemetry_->scalar("frontend/prior/dim",
                         prior_ ? static_cast<double>(prior_->residualDim()) : 0.0, t_end);
    }
  }
  return accepted;
}

void CtFrontEnd::slideWindow(ceres::Problem& problem, Timestamp t_begin, Timestamp t_end) {
  // Per-sweep fixed-lag marginalization. The just-solved problem holds only this
  // sweep's residuals plus the previous prior, so a knot can transfer real LiDAR/IMU
  // (or carried-prior) information into the new prior only while a live residual still
  // touches it. A knot's residual support is the cubic B-spline's 4-knot span; the
  // knots whose support has fully left the active window this sweep are those from the
  // oldest one the carried prior still anchors up to the first knot kept past
  // t_begin's outer segment. Marginalizing exactly that set folds the departing
  // boundary into the prior and leaves the next-oldest knot connected, so the chain
  // self-sustains. This is a regularizing per-sweep prior, not a fully-persistent
  // information chain: information older than the support already lives in the prior
  // and is carried, not re-marginalized from the original measurements.
  const int window_knots = std::max(cfg_.spline.window_knots, kSplineOrder);
  // Hold off until the trajectory has grown longer than the information window so the
  // dropped knots are genuinely exiting rather than still needed to keep the active set
  // full. The horizon is measured against the FIXED trajectory start (front-trimming
  // advances minTime() each sweep, so minTime() can no longer stand in for it). Also
  // require t_begin to have advanced past the last slide so a knot group is dropped at
  // most once.
  const Timestamp horizon = t_end - static_cast<Timestamp>(window_knots) * window_dt_ns_;
  if (horizon <= traj_start_t_ || t_begin <= window_floor_t_) {
    return;
  }
  // The knots leaving the window this sweep span [lower, D). lead is the oldest knot
  // this sweep's factors reach; D is the first knot kept past t_begin's outer segment
  // (so3[D]/r3[D] is the no-prior gauge anchor, hence never dropped). The lower bound
  // is pulled back to the oldest knot the carried prior still references: a sweep that
  // crossed several outer segments (a skipped slide, or a gear-change re-lay) leaves
  // earlier boundary knots resident only through the prior, and they must retire here
  // too or the kept-set recurrence re-keeps them forever and the window grows without
  // bound. With one outer segment departing, the prior's oldest anchor IS lead, so the
  // bound collapses to [lead, D) -- exactly that single segment's control points, each
  // still referenced by this sweep's residuals (a live Markov blanket). A knot trimmed
  // without marginalization silently destroys its information and misaligns the prior
  // against the window boundary, so every departed knot is Schur-eliminated first.
  const int lead = spline_->leadingKnotIndex(t_begin);
  const int D = firstKeptKnotIndex(t_begin);
  int lower = lead;
  if (prior_) {
    std::vector<const double*> prior_ptrs;
    prior_ptrs.reserve(prior_->blocks().size());
    for (const auto& b : prior_->blocks()) {
      prior_ptrs.push_back(b.ptr);
    }
    lower = std::min(lead, spline_->lowestKnotIndexOf(prior_ptrs));
  }
  std::vector<double*> drop_ptrs;
  for (int j = lower; j < D; ++j) {
    drop_ptrs.push_back(spline_->so3KnotData(j));
    drop_ptrs.push_back(spline_->r3KnotData(j));
  }

  if (record_slide_drops_) {
    SlideDropRecord rec;
    rec.unified = drop_ptrs;
    const Timestamp seg_end_t = spline_->knotTime(lead) + window_dt_ns_;
    for (int j = lead; j < spline_->numKnots() && spline_->knotTime(j) < seg_end_t; ++j) {
      rec.legacy_oracle.push_back(spline_->so3KnotData(j));
      rec.legacy_oracle.push_back(spline_->r3KnotData(j));
    }
    slide_drop_records_.push_back(std::move(rec));
  }

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

  // A kept block must outlive the prior, so only blocks whose storage this class owns
  // across sweeps (spline knots, bias knots, gravity) may be kept. Anything else -- a
  // per-sweep stack-owned exposure scalar pulled into the blanket -- is Schur-dropped:
  // keeping it would dangle the prior into a dead stack frame on the next solve.
  auto persistent = [&](double* p) -> bool {
    if (p == gravity_->data()) {
      return true;
    }
    for (int i = 0; i < bias_->numKnots(); ++i) {
      if (p == bias_->gyroBlock(i) || p == bias_->accelBlock(i)) {
        return true;
      }
    }
    for (int i = 0; i < spline_->numKnots(); ++i) {
      if (p == spline_->so3KnotData(i) || p == spline_->r3KnotData(i)) {
        return true;
      }
    }
    return false;
  };

  std::vector<ct::MarginalizationPrior::Block> kept;
  std::vector<ct::MarginalizationPrior::Block> dropped;
  int dropped_dim = 0;
  for (double* p : blanket) {
    if (drop_set.count(p)) {
      dropped.push_back(makeBlock(p));
      dropped_dim += dropped.back().local_size;
    } else if (problem.IsParameterBlockConstant(p)) {
      // Constant gauge pins carry no information to transfer; skip them so the prior
      // is built only over free boundary blocks.
    } else if (!persistent(p)) {
      dropped.push_back(makeBlock(p));
      dropped_dim += dropped.back().local_size;
    } else {
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

  // Structured per-state forgetting: a per-coordinate covariance inflation over the
  // kept tangent stack, stacked in the same block order and local_size widths fromSchur
  // uses. Blocks are classified by POINTER identity, never by size (r3 knots, gyro bias,
  // and accel bias are all global 3 / local 3); an unrecognized block (e.g. a visual
  // inverse-exposure scalar pulled into the blanket) fails safe to zero inflation. The
  // bias inflation is the random-walk variance density integrated over the time the
  // window front advanced since the last successful prior build -- early returns above
  // advance window_floor_t_ without rebuilding, so the next build inflates by the
  // accumulated span, clamped so a long outage cannot erase the prior in one step.
  // Gravity, when kept, relaxes by a per-build constant (near-constant state). When
  // enabled, the legacy scalar deflation is forced to 1.0 (never stacked on top).
  const bool forgetting = cfg_.prior_forgetting.enabled;
  Eigen::VectorXd q_diag;
  if (forgetting) {
    const double dt_raw = last_prior_build_t_ > 0 ? to_seconds(t_begin - last_prior_build_t_)
                                                  : to_seconds(window_dt_ns_);
    const double dt = std::clamp(dt_raw, 0.0, 1.0);
    // Bias random-walk std densities, identical source and fallbacks as the in-window
    // residual weights; squared to variance densities for the covariance inflation.
    double dens_bg = 1e-4, dens_ba = 1e-4;
    if (calib_) {
      if (calib_->imu_gyr_bias_rw > 0) dens_bg = calib_->imu_gyr_bias_rw;
      if (calib_->imu_acc_bias_rw > 0) dens_ba = calib_->imu_acc_bias_rw;
    }
    const FrontendPriorForgetting& pf = cfg_.prior_forgetting;
    const double q_bg = (pf.use_imu_walk ? dens_bg * dens_bg : pf.q_bias_gyr) * dt;
    const double q_ba = (pf.use_imu_walk ? dens_ba * dens_ba : pf.q_bias_acc) * dt;
    const double q_knot = pf.q_knot * dt;
    auto classify = [&](double* p) -> double {
      if (p == gravity_->data()) {
        return pf.q_gravity;
      }
      for (int i = 0; i < bias_->numKnots(); ++i) {
        if (p == bias_->gyroBlock(i)) return q_bg;
        if (p == bias_->accelBlock(i)) return q_ba;
      }
      for (int i = 0; i < spline_->numKnots(); ++i) {
        if (p == spline_->so3KnotData(i) || p == spline_->r3KnotData(i)) return q_knot;
      }
      return 0.0;
    };
    int kept_dim = 0;
    for (const auto& blk : kept) {
      kept_dim += blk.local_size;
    }
    q_diag.setZero(kept_dim);
    int off = 0;
    for (const auto& blk : kept) {
      q_diag.segment(off, blk.local_size).setConstant(classify(blk.ptr));
      off += blk.local_size;
    }
  }

  Eigen::MatrixXd H;
  Eigen::VectorXd bvec;
  ct::linearizeBlocks(problem, residuals, order, &H, &bvec);
  std::unique_ptr<ct::MarginalizationPrior> next = ct::MarginalizationPrior::fromSchur(
      H, bvec, kept, dropped_dim, forgetting ? 1.0 : cfg_.marg_prior_scale,
      forgetting ? &q_diag : nullptr);
  if (next) {
    prior_ = std::move(next);
    last_prior_build_t_ = t_begin;
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
    const int knots_before = spline_->numKnots();
    spline_->dropOldest(std::min(lead, min_kept));
    // The front-trim shifts deque indexing down, so the stored adaptive tail-pin index
    // is re-based by the count actually removed to keep addressing the same knots
    // (dropOldest clamps to keep the cubic support resident, so fewer may be removed).
    const int removed = knots_before - spline_->numKnots();
    if (last_first_pinned_ >= 0) {
      last_first_pinned_ = std::max(last_first_pinned_ - removed, -1);
    }

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
  const auto t_warp = Clock::now();
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
  if (telemetry_) {
    telemetry_->timing("frontend.ct.map.warp", ms_since(t_warp), last_solved_t_);
  }
  if (rejected > 0 && telemetry_ && telemetry_->enabled("frontend/map/insert_rejected")) {
    telemetry_->scalar("frontend/map/insert_rejected", static_cast<double>(rejected),
                       last_solved_t_);
  }
  // Map growth rate: points offered to the tree this sweep (the tree's own voxel
  // gate decides what survives; size deltas reveal how much actually stuck).
  if (telemetry_ && telemetry_->enabled("frontend/map/n_inserted")) {
    telemetry_->scalar("frontend/map/n_inserted", static_cast<double>(world.size()),
                       last_solved_t_);
  }
  if (!world.empty()) {
    const auto t_insert = Clock::now();
    map_->insert(world);
    if (telemetry_) {
      telemetry_->timing("frontend.ct.map.insert", ms_since(t_insert), last_solved_t_);
    }
  }
  // Segment the ikd-Tree to a cube around the current body so map RAM and the per-point
  // nearest-neighbour search depth stay bounded over a long mission.
  const auto t_trim = Clock::now();
  map_->trimAround(anchor_pose_.t, cfg_.lidar.local_map_cube_m);
  // Resolve every deferred push-down the insert/trim left behind so the next sweep's
  // parallel association only ever reads settled node flags and cannot race on them.
  map_->flushPendingDeletes();
  if (telemetry_) {
    telemetry_->timing("frontend.ct.map.trim", ms_since(t_trim), last_solved_t_);
  }
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

void CtFrontEnd::captureFinalRound(Timestamp t_begin, Timestamp t_end,
                                   const std::vector<ImuSample>& imu, const ct::ImuWeights& weights,
                                   const std::vector<ct::LidarHit>& capped_hits,
                                   const std::vector<ct::VisualPatchParams>& patches,
                                   const VisualContext* vis, const GnssContext* gnss,
                                   bool motion_reg_engaged,
                                   const std::vector<Timestamp>& motion_reg_times,
                                   const std::vector<Timestamp>& tail_times,
                                   int first_pinned_knot, int gauge_knot_idx) {
  capture_ = WindowBuildCapture{};
  ct::WindowProblemInputs& in = capture_.inputs;
  in.gravity_refine = cfg_.gravity_refine;

  // Deep value copies the worker owns; the live spline/bias/gravity/prior keep mutating
  // on T2 and are destroyed at the next slide, so the job must not alias them.
  capture_.spline_clone = spline_->clone();
  capture_.bias_clone = std::make_unique<ct::BiasKnots>(bias_->clone());
  capture_.gravity_clone = *gravity_;

  // Remap the prior's kept-block pointers from live knot storage to the clones', matched
  // by deque index. A pointer the remap does not recognise (none expected here) passes
  // through unchanged.
  if (prior_) {
    SplineWindow* sc = capture_.spline_clone.get();
    ct::BiasKnots* bc = capture_.bias_clone.get();
    const int n_knots = spline_->numKnots();
    const int n_bias = bias_->numKnots();
    capture_.prior_clone = prior_->cloneRemapped([&](double* p) -> double* {
      for (int i = 0; i < n_knots; ++i) {
        if (p == spline_->so3KnotData(i)) return sc->so3KnotData(i);
        if (p == spline_->r3KnotData(i)) return sc->r3KnotData(i);
      }
      for (int i = 0; i < n_bias; ++i) {
        if (p == bias_->gyroBlock(i)) return bc->gyroBlock(i);
        if (p == bias_->accelBlock(i)) return bc->accelBlock(i);
      }
      return p;
    });
  }

  in.t_begin = t_begin;
  in.t_end = t_end;
  capture_.capped_hits = capped_hits;
  in.T_fe_lidar = T_fe_lidar_;
  in.lidar_cfg = cfg_.lidar;

  capture_.imu = imu;  // segment IMU samples (copied; addImuResiduals filters by span)
  in.weights = weights;

  in.tail_times = tail_times;
  in.seed_vel_end = seed_vel_end_;
  in.seed_omega_end = seed_omega_end_;
  in.tail_sigma_vel = kTailSigmaVel;
  in.tail_sigma_rate = kTailSigmaRate;

  in.motion_reg_engaged = motion_reg_engaged;
  in.motion_reg_times = motion_reg_times;
  in.motion_reg_weight = cfg_.motion_reg.weight;
  in.motion_reg_accel_weight = 1.0 / std::max(weights.sigma_accel, 1e-9);
  in.motion_reg_gyro_weight = 1.0 / std::max(weights.sigma_gyro, 1e-9);

  in.visual_patches = patches;
  in.visual_cfg = cfg_.visual;
  if (vis != nullptr && !patches.empty()) {
    in.t_image = vis->t_image;
    in.inv_expo = inv_expo_;
    in.inv_expo_prior = inv_expo_prior_;
    in.inv_expo_std = inv_expo_std_;
    in.exposure_estimate_en = cfg_.visual.exposure_estimate_en;
  }

  if (gnss != nullptr) {
    in.t_fe_ant = gnss->t_fe_ant;
    in.gnss_huber = kHuberGnss;
    in.gnss_factors.reserve(gnss->fixes.size());
    for (const GnssMeasurement& m : gnss->fixes) {
      ct::GnssFactor f;
      f.z_world = m.z_world;
      f.sqrt_info = m.sqrt_info;
      f.t_fix = m.t_fix;
      in.gnss_factors.push_back(f);
    }
  }

  in.bias_gyr_max = std::max(cfg_.bias.gyr_max, 0.0);
  in.bias_acc_max = std::max(cfg_.bias.acc_max, 0.0);
  in.gauge_seg_t = t_begin;
  in.gauge_knot_idx = gauge_knot_idx;
  in.first_pinned_knot = first_pinned_knot;

  capture_.valid = true;
}

void CtFrontEnd::submitKeyframeJob(KeyframePacket&& pkt, const Pose& T_world_body, Timestamp stamp,
                                   const std::vector<ct::LidarHit>& hits, bool absolute_prior) {
  KeyframeJob job;
  job.spline = std::move(capture_.spline_clone);
  job.bias = std::move(capture_.bias_clone);
  job.gravity = capture_.gravity_clone;
  job.prior = std::move(capture_.prior_clone);
  job.inputs = std::move(capture_.inputs);
  job.capped_hits = std::move(capture_.capped_hits);
  job.imu = std::move(capture_.imu);
  job.stamp = stamp;
  job.pkt = std::move(pkt);
  job.absolute_prior = absolute_prior;
  // LiDAR-only marginal as the worker's fallback if the rebuilt posterior is rank-
  // deficient -- the same fallback the synchronous path would have taken.
  job.fallback_pose_cov = poseMarginal(T_world_body, hits);
  capture_.valid = false;

  // The push blocks only when the finalizer queue is full (worker behind by a full
  // queue depth of keyframes); the timer makes such a stall visible in the telemetry
  // instead of appearing as an anonymous sweep overrun.
  MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.covsubmit", stamp);
  finalizer_->submit(std::move(job));
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
  return ct::poseMarginalFromProblem(problem, *spline_, stamp, cov);
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

  // First keyframe, or the window was reseeded across a data hole on a constant-velocity
  // prediction: a relative edge spanning that discontinuity would carry the prediction
  // error with no covariance accounting for it, so break the relative chain and hand the
  // consumer an absolute anchor instead.
  const bool absolute_prior = !have_prev_kf_ || kf_chain_broken_;
  if (absolute_prior) {
    pkt.constraint_kind = KeyframePacket::ConstraintKind::AbsolutePrior;
    pkt.rel_to_id = 0;
    pkt.T_relto_this = Pose{};
  } else {
    pkt.constraint_kind = KeyframePacket::ConstraintKind::RelativeBetween;
    pkt.rel_to_id = prev_kf_id_;
    pkt.T_relto_this = prev_kf_pose_.inverse() * T_world_body;
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

  // Advance the chain bookkeeping on T2 regardless of path: keyframeDue and the next
  // relative transform read these. The pose-marginal CHAIN (prev_kf_pose_cov_) is owned
  // by the synchronous path here, or by the worker on the live path.
  prev_kf_pose_ = T_world_body;
  prev_kf_stamp_ = stamp;
  prev_kf_id_ = pkt.id;
  have_prev_kf_ = true;
  // An emitted anchor consumes the pending chain break.
  kf_chain_broken_ = false;

  if (finalizer_ && capture_.valid) {
    // Live path: the worker fills constraint_cov off T2 from the captured snapshot and
    // forwards the packet to the sink. The marginal chain is the worker's.
    submitKeyframeJob(std::move(pkt), T_world_body, stamp, hits, absolute_prior);
    return;
  }

  // Deterministic path (or a sweep with no valid capture): pack constraint_cov inline,
  // exactly as before. Prefer the window-posterior marginal computed on the live problem,
  // falling back to the LiDAR-only marginal when extraction failed.
  const Eigen::Matrix<double, 6, 6> pose_cov =
      have_pose_cov_ ? last_pose_cov_ : poseMarginal(T_world_body, hits);
  if (absolute_prior) {
    pkt.constraint_cov.form = GaussianBlock<6>::Form::Covariance;
    pkt.constraint_cov.M = reorderTransRotToRotTrans(pose_cov);
  } else {
    // Conservative relative cov: the sum of the two absolute pose marginals upper-bounds
    // the true relative marginal (which subtracts their cross-covariance).
    const Eigen::Matrix<double, 6, 6> rel_cov = pose_cov + prev_kf_pose_cov_;
    pkt.constraint_cov.form = GaussianBlock<6>::Form::Covariance;
    pkt.constraint_cov.M = reorderTransRotToRotTrans(rel_cov);
  }
  prev_kf_pose_cov_ = pose_cov;

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
    if (bootstrap_first_t_ == 0) {
      bootstrap_first_t_ = t_begin;
    }
    Eigen::Vector3d init_gravity = Eigen::Vector3d::Zero();
    Eigen::Vector3d init_bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d init_ba = Eigen::Vector3d::Zero();
    if (imu_init_->done()) {
      const ImuInitState& init = imu_init_->state();
      init_gravity = init.gravity;
      init_bg = init.gyro_bias;
      init_ba = init.accel_bias;
    } else if (cfg_.init_force_after_s > 0.0 &&
               to_seconds(t_end - bootstrap_first_t_) >= cfg_.init_force_after_s) {
      // Moving-start fallback: the motion gate has rejected every window for the
      // configured span, so seed from this sweep's IMU mean. The mean specific force
      // points along -gravity plus the platform acceleration; normalizing to g leaves
      // a tilt of a few degrees that the window (and a freed gravity block) absorbs.
      Eigen::Vector3d acc = Eigen::Vector3d::Zero(), gyr = Eigen::Vector3d::Zero();
      for (const ImuSample& m : mg.imu) {
        acc += m.acc;
        gyr += m.gyro;
      }
      acc /= static_cast<double>(mg.imu.size());
      gyr /= static_cast<double>(mg.imu.size());
      if (acc.norm() < 1e-3) {
        return;  // degenerate accel mean: keep waiting
      }
      init_gravity = -acc.normalized() * kGravityMag;
      init_bg = gyr;
      init_ba = Eigen::Vector3d::Zero();
      if (telemetry_) {
        telemetry_->event(Level::Warn, "frontend/init_forced",
                          "static gate never passed; moving-start init from sweep mean",
                          t_end);
      }
    } else {
      return;  // still accumulating a static window (or the gate rejected this batch)
    }

    gravity_ = std::make_unique<ct::GravityBlock>(init_gravity, kGravityMag);
    anchor_bg_ = init_bg;
    anchor_ba_ = init_ba;
    anchor_vel_ = Eigen::Vector3d::Zero();
    anchor_pose_ = Pose{};  // identity anchors the odom origin

    // Spline at identity; one coarse bias knot per window; the window opens at
    // t_begin and is extended to t_end so the first sweep has full coverage. The
    // spline carries the configured adaptive control-point cap; the first (static)
    // segment opens at a single control point and density follows motion thereafter.
    const int n_cp_max = maxControlPoints();
    spline_ = std::make_unique<SplineWindow>(window_dt_ns_, n_cp_max, cfg_.ncp.enabled);
    spline_->initialize(t_begin, anchor_pose_);
    bias_ = std::make_unique<ct::BiasKnots>(t_begin, biasKnotDt(), 2);
    for (int k = 0; k < bias_->numKnots(); ++k) {
      bias_->setGyroKnot(k, anchor_bg_);
      bias_->setAccelKnot(k, anchor_ba_);
    }
    last_n_cp_ = 1;
    sweep_max_gear_ = 1;
    last_first_pinned_ = -1;

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
      // Settle deferred push-downs before the next sweep's parallel association so its
      // concurrent searches only read resolved node flags.
      map_->flushPendingDeletes();
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
    // A short gap (a sweep or two dropped under load) is bridged, NOT restarted:
    // buildSeed integrates the group's IMU from last_solved_t_ across the hole
    // (extrapolating the leading span from the first sample), extendTo appends the
    // missing segments, and the sweep then SOLVES normally against its LiDAR -- so a
    // dropped sweep costs centimetres of extrapolation, not a coast. Restart only for
    // a genuinely large hole, where extrapolating the seed would warp the associations
    // beyond what re-association can recover. The bound is a few sweeps of motion.
    const Duration max_bridge = static_cast<Duration>(5) * window_dt_ns_;
    const bool imu_bridges = !mg.imu.empty() && (t_begin - last_solved_t_) <= max_bridge;
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
        std::snprintf(msg, sizeof(msg), "sweep gap %.0f ms exceeds IMU horizon; window reseeded",
                      gap_ms);
        telemetry_->event(Level::Warn, "frontend/window_restart", msg, t_begin);
      }
      reseedAfterGap(group, t_begin, t_end);
      diag_.restarted = true;
      diag_.scan_time_ms = ms_since(t_start);
      return;
    }
  }

  // ---- Steady state: extend, associate, solve, map, slide, keyframe.
  steadyStateExtend(mg, last_solved_t_, t_end);

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

  {
    MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.obs", t_end);
    last_obs_ = computeObservability(T_world_end, hits);
  }
  diag_.observability = last_obs_;

  // Per-axis observability score (lambda/(lambda+kappa) in [0,1], translation-first
  // [tx,ty,tz,rx,ry,rz]); a value near 0 is an unconstrained DOF. Published every
  // solve so a degenerate axis is visible over time, not only inside a keyframe.
  // One atomic vector (not six scalars): a burst of per-axis messages is dropped
  // front-first by a best-effort subscriber, which silently loses the tx/ty axes.
  if (telemetry_) {
    const auto& s = last_obs_.score;
    Eigen::Matrix<double, 6, 1> ov(s[0], s[1], s[2], s[3], s[4], s[5]);
    telemetry_->vec("frontend/obs", ov, t_end, "tx,ty,tz,rx,ry,rz");
    telemetry_->scalar("frontend/obs_min", *std::min_element(s.begin(), s.end()), t_end);
  }

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

    {
      MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.vmap.update", t_end);
      vmap_->updateAfterSolve(*vctx.img, cam_model_, T_w_c, inv_expo_);
    }
    {
      MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.vmap.promote", t_end);
      vmap_->promote(*vctx.img, cam_model_, T_w_c, inv_expo_, hits);
    }
    {
      MERIDIAN_SCOPED_TIME(telemetry_, "frontend.ct.vmap.evict", t_end);
      vmap_->evict(T_w_c.t);
    }
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
      telemetry_->scalar("frontend/visual/map_points", static_cast<double>(vmap_->size()), t_end);
    }
    if (telemetry_->enabled("frontend/visual/n_candidates")) {
      telemetry_->scalar("frontend/visual/n_candidates", static_cast<double>(vctx.stats.candidates),
                         t_end);
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
    if (telemetry_->enabled("frontend/map/size")) {
      telemetry_->scalar("frontend/map/size", static_cast<double>(map_->size()), t_end);
    }
    // The weakest pose axis of this sweep's LiDAR observability report.
    if (telemetry_->enabled("frontend/obs/min_score")) {
      telemetry_->scalar("frontend/obs/min_score",
                         *std::min_element(last_obs_.score.begin(), last_obs_.score.end()), t_end);
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
      // Max finalized gear this sweep; equals the carried gear (1 with adaptive
      // density disabled) when no segment finalized.
      telemetry_->scalar("frontend/ct/n_cp",
                         static_cast<double>(cfg_.ncp.enabled ? sweep_max_gear_ : last_n_cp_),
                         t_end);
    }
    if (cfg_.ncp.enabled && telemetry_->enabled("frontend/ncp/gear")) {
      telemetry_->scalar("frontend/ncp/window_knots", static_cast<double>(spline_->numKnots()),
                         t_end);
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
