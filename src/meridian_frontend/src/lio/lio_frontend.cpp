#include "lio/lio_frontend.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <exception>
#include <numbers>
#include <sophus/so3.hpp>
#include <span>
#include <utility>

#include "lio/relative_cov.hpp"
#include "meridian/common/cov_reorder.hpp"
#include "meridian/common/graph_update.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/debug/telemetry.hpp"
#include "meridian/debug/telemetry_keys.hpp"

namespace meridian::lio {
namespace {

// Variance per tangent axis of the run's first keyframe: that pose defines the odom
// origin, so its prior is tight by construction rather than measured.
constexpr double kAnchorPriorVar = 1e-8;

// Knee of the per-axis observability score s = h / (h + kappa), applied to the
// per-correspondence average of the normal-projected information. A translation axis
// carrying >= ~1/10 of the normal energy (any closed scene) scores > 0.9; an axis with
// no normal support (sliding along a lone plane, yaw about its normal) scores ~0.
constexpr double kObsKappa = 0.01;

// A local surface normal needs a few non-collinear neighbors to be meaningful.
constexpr std::size_t kObsMinNeighbors = 5;

// Bound on groups held while static initialization is pending (~6 s at 10 Hz).
constexpr std::size_t kMaxHeldGroups = 64;

// Bound on buffered live IMU samples (~20 s at 200 Hz).
constexpr std::size_t kMaxLiveImu = 4096;

Pose extrinsicOrIdentity(const CalibrationSet& calib, Frame sensor) {
  for (const Extrinsic& e : calib.extrinsics) {
    if (e.child == sensor) {
      return e.T_parent_child;
    }
  }
  // The frame is F_e itself (or absent from the rig): identity.
  return Pose{};
}

std::shared_ptr<PointCloud> toCloud(const std::vector<Eigen::Vector3d>& points) {
  auto cloud = std::make_shared<PointCloud>();
  cloud->reserve(points.size());
  for (const Eigen::Vector3d& p : points) {
    LidarPoint lp;
    lp.xyz = p.cast<float>();
    cloud->push_back(lp);
  }
  return cloud;
}

}  // namespace

LioFrontEnd::LioFrontEnd(const FrontendConfig& cfg, std::shared_ptr<const CalibrationSet> calib,
                         TelemetrySink* telemetry, bool deterministic)
    : cfg_(cfg),
      calib_(std::move(calib)),
      telemetry_(telemetry),
      deterministic_(deterministic),
      tracker_(cfg.lio),
      registration_(cfg.lio),
      map_(cfg.lio) {
  if (calib_) {
    adoptCalibration(*calib_);
  }
}

void LioFrontEnd::set_calibration(std::shared_ptr<const CalibrationSet> calib) {
  calib_ = std::move(calib);
  if (calib_) {
    adoptCalibration(*calib_);
  }
}

void LioFrontEnd::adoptCalibration(const CalibrationSet& calib) {
  R_body_imu_ = extrinsicOrIdentity(calib, Frame::ImuLink).q.toRotationMatrix();
  T_body_lidar_ = extrinsicOrIdentity(calib, Frame::OsSensor0);
  T_body_cam_ = extrinsicOrIdentity(calib, Frame::CamLink);
  calib_version_ = calib.version;
}

ImuSample LioFrontEnd::toBody(const ImuSample& s) const {
  ImuSample out = s;
  out.acc = R_body_imu_ * s.acc;
  out.gyro = R_body_imu_ * s.gyro;
  return out;
}

void LioFrontEnd::ingest(const PreprocessedGroup& group) {
  try {
    const MeasureGroup& g = group.group;
    if (!tracker_.initialized()) {
      // Estimation IMU comes from the group stream only; re-delivered straddle stamps
      // dedup inside the tracker.
      for (const ImuSample& s : g.imu) {
        tracker_.add_sample(toBody(s));
      }
      held_.push_back(g);
      if (held_.size() > kMaxHeldGroups) {
        held_.pop_front();
        if (!held_drop_warned_ && telemetry_ != nullptr &&
            telemetry_->enabled(keys::frontend::LioInitBacklog)) {
          telemetry_->event(Level::Warn, keys::frontend::LioInitBacklog,
                            "static init still pending; dropping oldest held group", g.t_end);
          held_drop_warned_ = true;
        }
      }
      if (!tracker_.initialized()) {
        return;
      }
      if (telemetry_ != nullptr && telemetry_->enabled(keys::frontend::LioInitDone)) {
        telemetry_->event(Level::Info, keys::frontend::LioInitDone,
                          "static init complete: gravity aligned, at-rest biases fixed", g.t_end);
      }
      // Drain in arrival order. Every held sample is already inside the tracker, so the
      // re-feed in processGroup dedups to a no-op; the propagated guess for the older
      // groups is the standstill pose the initializer just fixed.
      while (!held_.empty()) {
        processGroup(held_.front());
        held_.pop_front();
      }
      return;
    }
    processGroup(g);
  } catch (const std::exception& e) {
    // The interface is exception-free; surface the fault and hold the last good state.
    if (telemetry_ != nullptr) {
      telemetry_->event(Level::Error, keys::frontend::LioError, e.what(), group.group.t_end);
    }
  }
}

void LioFrontEnd::processGroup(const MeasureGroup& g) {
  MERIDIAN_SCOPED_TIME(telemetry_, keys::stage::LioIngest, g.t_end);
  const auto t_ingest0 = Clock::now();

  // Gap check before this group's samples reach the tracker, so dead reckoning never
  // integrates one giant dt across the hole. The state bridges the gap on constant
  // velocity; the map is kept and the next registration outcome decides the reseed.
  if (have_prev_group_ && to_seconds(g.t_begin - prev_t_end_) > cfg_.lio.max_gap_s) {
    NavState bridged = last_state_;
    bridged.T_world_body.t += last_state_.v_world * to_seconds(g.t_begin - last_state_.stamp);
    bridged.stamp = g.t_begin;
    tracker_.rebase(bridged);
    reseed_armed_ = true;
    if (telemetry_ != nullptr && telemetry_->enabled(keys::frontend::LioGap)) {
      telemetry_->event(Level::Warn, keys::frontend::LioGap,
                        "sensor gap bridged on constant velocity; reseed armed", g.t_begin);
    }
  }

  for (const ImuSample& s : g.imu) {
    tracker_.add_sample(toBody(s));
  }
  if (g.image) {
    last_image_ = std::make_shared<const CameraFrame>(*g.image);
  }
  have_prev_group_ = true;
  prev_t_end_ = g.t_end;

  const MotionPrior prior = tracker_.interval_prior(g.t_begin, g.t_end);
  NavState guess = tracker_.propagated_state();
  guess.stamp = g.t_end;

  const auto t_deskew0 = Clock::now();
  // Constant-screw deskew to the sweep end: interval-mean rate plus the propagated
  // velocity expressed in the body frame.
  const Eigen::Vector3d vel_body = guess.T_world_body.q.conjugate() * guess.v_world;
  const std::vector<Eigen::Vector3d> deskewed =
      registration_.deskew(g.scan, T_body_lidar_, prior.omega_body, vel_body, g.t_end);
  const double deskew_ms = ms_since(t_deskew0);

  const auto t_ds0 = Clock::now();
  const std::vector<Eigen::Vector3d> keypoints =
      voxelDownsample(deskewed, cfg_.lio.keypoint_voxel_factor * cfg_.lio.voxel_size_m);
  const double downsample_ms = ms_since(t_ds0);
  const bool scan_usable = static_cast<int>(keypoints.size()) >= cfg_.lio.min_keypoints;

  if (!bootstrapped_) {
    if (!scan_usable) {
      if (telemetry_ != nullptr && telemetry_->enabled(keys::frontend::LioReject)) {
        telemetry_->event(Level::Warn, keys::frontend::LioReject,
                          "sweep below the keypoint floor; bootstrap deferred", g.t_end);
      }
      return;
    }
    // Anchor the map on this sweep at the propagated pose; no registration possible.
    insertWorld(deskewed, guess.T_world_body);
    last_state_ = guess;
    tracker_.rebase(last_state_);
    rebaseLive(last_state_);
    bootstrapped_ = true;
    reseed_armed_ = false;
    last_obs_ = observabilityAt(keypoints, guess.T_world_body);

    diag_.deskew_time_ms = deskew_ms;
    diag_.solve_time_ms = 0.0;
    diag_.effective_points = 0;
    diag_.mean_residual = 0.0;
    diag_.final_residual = 0.0;
    diag_.observability = last_obs_;
    diag_.iterations = 0;
    diag_.outer_iters = 0;
    diag_.deadline_hit = false;
    diag_.scan_time_ms = ms_since(t_ingest0);

    PointCloudPtr cloud;
    publishBodyScan(deskewed, g.t_end, &cloud);
    // Only the run's first bootstrap is a keyframe; a re-anchor after a reseed defers
    // its absolute anchor to the first converged solve, which carries the covariance.
    if (next_kf_id_ == 1) {
      if (!cloud) {
        cloud = toCloud(deskewed);
      }
      emitKeyframe(last_state_, std::move(cloud));
    }
    publishSweepTelemetry(g, prior, RegistrationResult{}, keypoints.size());
    publishPathSample(last_state_.T_world_body, g.t_end);
    return;
  }

  RegistrationResult res;
  double solve_ms = 0.0;
  bool failed = !scan_usable;
  if (!failed) {
    const auto t_solve0 = Clock::now();
    res = registration_.registerScan(keypoints, map_, guess.T_world_body, prior);
    solve_ms = ms_since(t_solve0);
    failed = !res.converged;
  }

  if (failed) {
    handleFailure(g, guess, deskewed, scan_usable);
    diag_.deskew_time_ms = deskew_ms;
    diag_.solve_time_ms = solve_ms;
    diag_.effective_points = res.n_corr;
    diag_.mean_residual = 0.0;
    diag_.final_residual = res.chi;
    diag_.iterations = res.gn_iters;
    diag_.outer_iters = res.gn_iters;
    diag_.deadline_hit = false;
    diag_.scan_time_ms = ms_since(t_ingest0);
    PointCloudPtr cloud;
    publishBodyScan(deskewed, g.t_end, &cloud);
    publishSweepTelemetry(g, prior, res, keypoints.size());
    publishPathSample(last_state_.T_world_body, g.t_end);
    return;
  }

  // Success: rebase the dead-reckoning on the solved pose, with the velocity taken
  // from the finite-difference body twist of the step.
  const double dt = to_seconds(g.t_end - last_state_.stamp);
  const Pose delta = last_state_.T_world_body.inverse() * res.pose;
  NavState solved = last_state_;
  solved.stamp = g.t_end;
  solved.T_world_body = res.pose;
  if (dt > 0.0) {
    const Eigen::Matrix<double, 6, 1> xi = res.pose.boxminus(last_state_.T_world_body);
    solved.v_world = res.pose.q * (xi.head<3>() / dt);
  }
  solved.b_g = tracker_.bias_gyro();
  solved.b_a = tracker_.bias_accel();
  tracker_.rebase(solved);

  const auto t_map0 = Clock::now();
  insertWorld(deskewed, res.pose);
  map_.clipFarFrom(res.pose.t);
  const double map_ms = ms_since(t_map0);
  cov_rel_ = composeRelativeCov(cov_rel_, delta, res.cov_right);
  if (chain_broken_ && !have_reseed_cov_) {
    // The first converged covariance after a reseed, inflated, prices the absolute
    // anchor that re-roots the keyframe chain.
    pending_prior_cov_ = res.cov_right * cfg_.lio.reseed_cov_inflation;
    have_reseed_cov_ = true;
  }
  reseed_armed_ = false;
  const auto t_obs0 = Clock::now();
  last_obs_ = observabilityAt(keypoints, res.pose);
  const double obs_ms = ms_since(t_obs0);
  last_state_ = solved;
  rebaseLive(solved);

  diag_.deskew_time_ms = deskew_ms;
  diag_.solve_time_ms = solve_ms;
  diag_.effective_points = res.n_corr;
  // RMS correspondence distance: the closest single statistic to a mean residual the
  // accumulated chi (sum of squared 3-D distances) supports.
  diag_.mean_residual = res.n_corr > 0 ? std::sqrt(res.chi / res.n_corr) : 0.0;
  diag_.final_residual = res.chi;
  diag_.observability = last_obs_;
  diag_.iterations = res.gn_iters;
  // Association refreshes every Gauss-Newton iteration, so the two counters coincide.
  diag_.outer_iters = res.gn_iters;
  diag_.restarted = restart_pending_;
  restart_pending_ = false;
  diag_.deadline_hit = false;
  diag_.scan_time_ms = ms_since(t_ingest0);

  const auto t_kf0 = Clock::now();
  PointCloudPtr cloud;
  publishBodyScan(deskewed, g.t_end, &cloud);
  if (chain_broken_ || keyframeDue(res.pose, g.t_end)) {
    if (!cloud) {
      cloud = toCloud(deskewed);
    }
    emitKeyframe(solved, std::move(cloud));
  }
  const double kf_ms = ms_since(t_kf0);
  publishSweepTelemetry(g, prior, res, keypoints.size());
  publishPathSample(res.pose, g.t_end);

  // Per-phase timing of this ingest. timing() is never group-gated, so the breakdown is
  // always recorded; assoc/ldlt/cov come from the solve's own measurement.
  if (telemetry_ != nullptr) {
    const Timestamp t = g.t_end;
    telemetry_->timing(keys::stage::LioDeskew, deskew_ms, t);
    telemetry_->timing(keys::stage::LioDownsample, downsample_ms, t);
    telemetry_->timing(keys::stage::LioRegister, solve_ms, t);
    telemetry_->timing(keys::stage::LioRegAssoc, res.assoc_ms, t);
    telemetry_->timing(keys::stage::LioRegSolve, res.ldlt_ms, t);
    telemetry_->timing(keys::stage::LioRegCov, res.cov_ms, t);
    telemetry_->timing(keys::stage::LioMapUpdate, map_ms, t);
    telemetry_->timing(keys::stage::LioObservability, obs_ms, t);
    telemetry_->timing(keys::stage::LioKeyframe, kf_ms, t);
    if (telemetry_->enabled(keys::frontend::LioScanMs)) {
      telemetry_->scalar(keys::frontend::LioScanMs, ms_since(t_ingest0), t);
    }
  }
}

void LioFrontEnd::handleFailure(const MeasureGroup& g, const NavState& guess,
                                const std::vector<Eigen::Vector3d>& deskewed, bool scan_usable) {
  restart_pending_ = true;
  diag_.restarted = true;
  // Time must advance regardless; the dead-reckoned pose is the best available.
  last_state_ = guess;
  tracker_.rebase(guess);
  rebaseLive(guess);

  if (!reseed_armed_) {
    // Steady-state failure: drop the sweep, keep the map, try again next sweep.
    if (telemetry_ != nullptr && telemetry_->enabled(keys::frontend::LioReject)) {
      telemetry_->event(Level::Warn, keys::frontend::LioReject,
                        "registration failed; sweep skipped, map kept", g.t_end);
    }
    return;
  }

  // First failure after a gap: the bridged prediction and the map disagree beyond
  // recovery, so restart the local map and break the relative keyframe chain.
  map_.clear();
  cov_rel_.setZero();
  chain_broken_ = true;
  have_reseed_cov_ = false;
  reseed_armed_ = false;
  if (scan_usable) {
    insertWorld(deskewed, guess.T_world_body);
  } else {
    // Nothing to anchor on; the next usable sweep re-runs the bootstrap path.
    bootstrapped_ = false;
  }
  if (telemetry_ != nullptr && telemetry_->enabled(keys::frontend::LioReseed)) {
    telemetry_->event(Level::Warn, keys::frontend::LioReseed,
                      "map cleared and re-anchored after post-gap registration failure", g.t_end);
  }
}

void LioFrontEnd::insertWorld(const std::vector<Eigen::Vector3d>& points_body,
                              const Pose& T_world_body) {
  std::vector<Eigen::Vector3d> world;
  world.reserve(points_body.size());
  for (const Eigen::Vector3d& p : points_body) {
    world.push_back(T_world_body * p);
  }
  map_.insert(world);
}

ObservabilityReport LioFrontEnd::observabilityAt(const std::vector<Eigen::Vector3d>& keypoints_body,
                                                 const Pose& T_world_body) const {
  ObservabilityReport rep;
  rep.frame = Frame::Body;

  // Point-to-point J^T*J carries an isotropic translation block (each correspondence
  // pins all three axes equally), so it cannot flag surface-sliding degeneracy. The
  // report instead projects each residual onto the local map surface normal — the
  // component re-association cannot cancel: r = n . (T*exp(dx)*p - q) gives at dx = 0
  // the right-chart row n^T * R * [I | -hat(p)] = [n_b^T | (p x n_b)^T], n_b = R^T n.
  Mat6 h = Mat6::Zero();
  int rows = 0;
  const Eigen::Matrix3d body_from_world = T_world_body.q.conjugate().toRotationMatrix();
  Eigen::Vector3d matched = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> nbrs;
  for (const Eigen::Vector3d& p : keypoints_body) {
    const Eigen::Vector3d pw = T_world_body * p;
    if (!map_.nearest(pw, &matched)) {
      continue;
    }
    map_.neighborsWithin(matched, cfg_.lio.voxel_size_m, &nbrs);
    if (nbrs.size() < kObsMinNeighbors) {
      continue;
    }
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& q : nbrs) {
      mean += q;
    }
    mean /= static_cast<double>(nbrs.size());
    Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d& q : nbrs) {
      const Eigen::Vector3d d = q - mean;
      scatter.noalias() += d * d.transpose();
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(scatter);
    // A collinear neighborhood has no surface normal; eigenvalues are ascending, so
    // the smallest-scatter direction (column 0) is the normal of a planar patch.
    if (es.eigenvalues()(1) <= 1e-12) {
      continue;
    }
    const Eigen::Vector3d n_body = body_from_world * es.eigenvectors().col(0);
    Eigen::Matrix<double, 6, 1> row;
    row.head<3>() = n_body;
    row.tail<3>() = p.cross(n_body);
    h.noalias() += row * row.transpose();
    ++rows;
  }
  if (rows == 0) {
    rep.score = {0, 0, 0, 0, 0, 0};
    return rep;
  }
  h /= static_cast<double>(rows);
  // h(i,i) = sum_k lambda_k * v_k(i)^2 is the spectrum projected onto axis i; the
  // score saturates well-supported axes toward 1 and leaves unsupported ones near 0.
  for (int i = 0; i < 6; ++i) {
    rep.score[static_cast<std::size_t>(i)] = h(i, i) / (h(i, i) + kObsKappa);
  }
  return rep;
}

bool LioFrontEnd::keyframeDue(const Pose& T_world_body, Timestamp stamp) const {
  if (!have_prev_kf_) {
    return true;
  }
  const Pose rel = prev_kf_pose_.inverse() * T_world_body;
  const double dist = rel.t.norm();
  const double rot_deg = Sophus::SO3d(rel.q).log().norm() * 180.0 / std::numbers::pi;
  const double elapsed_s = to_seconds(stamp - prev_kf_stamp_);
  return dist >= cfg_.keyframe.dist_m || rot_deg >= cfg_.keyframe.rot_deg ||
         elapsed_s >= cfg_.keyframe.time_s;
}

void LioFrontEnd::emitKeyframe(const NavState& state, PointCloudPtr cloud_body) {
  KeyframePacket pkt;
  pkt.id = next_kf_id_++;
  pkt.stamp = state.stamp;
  pkt.ref_frame = Frame::Odom;
  pkt.T_ref_body = state.T_world_body;
  pkt.kinematics_included = false;
  pkt.observability = last_obs_;
  pkt.cloud_body = std::move(cloud_body);
  pkt.image = last_image_;
  pkt.T_body_cam = T_body_cam_;
  pkt.calib_version = calib_version_;
  pkt.frontend_kind = 2;

  Mat6 cov;  // translation-first [rho; phi] until the pack-time reorder below
  if (!have_prev_kf_) {
    pkt.constraint_kind = KeyframePacket::ConstraintKind::AbsolutePrior;
    cov = kAnchorPriorVar * Mat6::Identity();
  } else if (chain_broken_) {
    pkt.constraint_kind = KeyframePacket::ConstraintKind::AbsolutePrior;
    cov = have_reseed_cov_
              ? pending_prior_cov_
              : Mat6(kAnchorPriorVar * cfg_.lio.reseed_cov_inflation * Mat6::Identity());
  } else {
    pkt.constraint_kind = KeyframePacket::ConstraintKind::RelativeBetween;
    pkt.rel_to_id = prev_kf_id_;
    pkt.T_relto_this = prev_kf_pose_.inverse() * state.T_world_body;
    cov = cov_rel_;
  }
  pkt.constraint_cov.form = GaussianBlock<6>::Form::Covariance;
  // The single rotation-first block in the system, reordered exactly once here.
  pkt.constraint_cov.M = reorderTransRotToRotTrans(cov);

  prev_kf_pose_ = state.T_world_body;
  prev_kf_stamp_ = state.stamp;
  prev_kf_id_ = pkt.id;
  have_prev_kf_ = true;
  chain_broken_ = false;
  have_reseed_cov_ = false;
  cov_rel_.setZero();

  if (keyframe_sink_) {
    keyframe_sink_(std::move(pkt));
  }
}

void LioFrontEnd::ingest_imu_live(const ImuSample& imu) {
  const ImuSample s = toBody(imu);
  if (!live_imu_.empty() && s.stamp <= live_imu_.back().stamp) {
    return;  // duplicate or out-of-order live sample
  }
  live_imu_.push_back(s);
  if (live_imu_.size() > kMaxLiveImu) {
    live_imu_.pop_front();
  }
  if (bootstrapped_ && s.stamp > live_state_.stamp) {
    propagateLive(s);
  }
}

void LioFrontEnd::rebaseLive(const NavState& solved) {
  while (!live_imu_.empty() && live_imu_.front().stamp <= solved.stamp) {
    live_imu_.pop_front();
  }
  live_state_ = solved;
  for (const ImuSample& s : live_imu_) {
    propagateLive(s);
  }
}

void LioFrontEnd::propagateLive(const ImuSample& s) {
  const double dt = to_seconds(s.stamp - live_state_.stamp);
  if (dt <= 0.0) {
    return;
  }
  const Eigen::Quaterniond q = live_state_.T_world_body.q;
  const Eigen::Vector3d omega = s.gyro - live_state_.b_g;
  // f = R^T (a - g)  =>  a_world = R*(f + b_a-corrected) + g_world.
  const Eigen::Vector3d accel_world = q * (s.acc - live_state_.b_a) + live_state_.g_world;
  live_state_.T_world_body.t += live_state_.v_world * dt + 0.5 * dt * dt * accel_world;
  live_state_.v_world += accel_world * dt;
  live_state_.T_world_body.q = (q * Sophus::SO3d::exp(omega * dt).unit_quaternion()).normalized();
  live_state_.stamp = s.stamp;
}

void LioFrontEnd::apply_correction(const GraphUpdate& update) {
  if (update.moved.empty() || !have_prev_kf_) {
    return;
  }
  const GraphUpdate::Moved& m = update.moved.back();
  if (m.id != prev_kf_id_) {
    return;
  }
  // World-frame rigid shift implied by the most recent moved keyframe. The map stays
  // in the pre-correction frame: a small correction keeps registering against it, and
  // a large one fails the next solve, which the reseed path absorbs.
  const Pose delta = m.new_T_map_body * prev_kf_pose_.inverse();
  prev_kf_pose_ = m.new_T_map_body;
  last_state_.T_world_body = delta * last_state_.T_world_body;
  last_state_.v_world = delta.q * last_state_.v_world;
  NavState shifted = tracker_.propagated_state();
  shifted.T_world_body = delta * shifted.T_world_body;
  shifted.v_world = delta.q * shifted.v_world;
  tracker_.rebase(shifted);
  live_state_.T_world_body = delta * live_state_.T_world_body;
  live_state_.v_world = delta.q * live_state_.v_world;
}

NavState LioFrontEnd::live_state() const {
  if (bootstrapped_) {
    return live_state_;
  }
  return tracker_.initialized() ? tracker_.propagated_state() : NavState{};
}

void LioFrontEnd::set_keyframe_sink(KeyframeSink sink) {
  keyframe_sink_ = std::move(sink);
}

FrontEndDiagnostics LioFrontEnd::diagnostics() const {
  return diag_;
}

void LioFrontEnd::publishSweepTelemetry(const MeasureGroup& g, const MotionPrior& prior,
                                        const RegistrationResult& res, std::size_t n_keypoints) {
  if (telemetry_ == nullptr) {
    return;
  }
  const Timestamp t = g.t_end;
  const auto scal = [this, t](const char* key, double v) {
    if (telemetry_->enabled(key)) {
      telemetry_->scalar(key, v, t);
    }
  };
  scal(keys::frontend::AssocNAttempted, static_cast<double>(n_keypoints));
  scal(keys::frontend::AssocNMatched, static_cast<double>(res.n_corr));
  scal(keys::frontend::SolverGnIters, static_cast<double>(res.gn_iters));
  scal(keys::frontend::SolverDxNorm, res.dx_norm);
  scal(keys::frontend::SolverChi, res.chi);
  scal(keys::frontend::MapVoxels, static_cast<double>(map_.voxel_count()));
  scal(keys::frontend::MapPoints, static_cast<double>(map_.size()));
  scal(keys::frontend::VelNorm, last_state_.v_world.norm());
  scal(keys::frontend::BiasGyrNorm, tracker_.bias_gyro().norm());
  scal(keys::frontend::BiasAccNorm, tracker_.bias_accel().norm());
  // Mirror of the solve's gravity-regularizer weight; -1 means the block was off.
  const double beta = (cfg_.lio.min_beta > 0.0 && prior.imu_count > 1)
                          ? cfg_.lio.min_beta * (1.0 + prior.accel_mag_variance)
                          : -1.0;
  scal(keys::frontend::LioBeta, beta);
  scal(keys::frontend::LioAccelVar, prior.accel_mag_variance);
  scal(keys::frontend::LioNCorr, static_cast<double>(res.n_corr));
  scal(keys::frontend::LioDeskewSpanTMs, to_seconds(g.scan.sweep_duration) * 1e3);
  if (telemetry_->enabled(keys::frontend::Observability)) {
    Eigen::Matrix<double, 6, 1> ov;
    for (int i = 0; i < 6; ++i) {
      ov[i] = last_obs_.score[static_cast<std::size_t>(i)];
    }
    telemetry_->vec(keys::frontend::Observability, ov, t, "tx,ty,tz,rx,ry,rz");
  }
  if (telemetry_->enabled(keys::frontend::ObsMin)) {
    telemetry_->scalar(keys::frontend::ObsMin,
                       *std::min_element(last_obs_.score.begin(), last_obs_.score.end()), t);
  }
}

void LioFrontEnd::publishBodyScan(const std::vector<Eigen::Vector3d>& deskewed, Timestamp t,
                                  PointCloudPtr* cloud) {
  if (telemetry_ == nullptr || !telemetry_->enabled(keys::body::Scan)) {
    return;
  }
  std::shared_ptr<PointCloud> built = toCloud(deskewed);
  PointCloudView view;
  view.points = std::span<const LidarPoint>(built->data(), built->size());
  telemetry_->cloud(keys::body::Scan, view, Frame::Body, t);
  *cloud = std::move(built);  // reusable by the keyframe pack
}

void LioFrontEnd::publishPathSample(const Pose& T_world_body, Timestamp t) {
  if (telemetry_ == nullptr || cfg_.debug_path_sample_hz <= 0.0 ||
      !telemetry_->enabled(keys::frontend::PathSample)) {
    return;
  }
  const Duration step = from_seconds(1.0 / cfg_.debug_path_sample_hz);
  if (last_path_t_ != 0 && t < last_path_t_ + step) {
    return;
  }
  telemetry_->pose(keys::frontend::PathSample, T_world_body, Frame::Odom, t);
  last_path_t_ = t;
}

}  // namespace meridian::lio
