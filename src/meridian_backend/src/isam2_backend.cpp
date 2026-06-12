#include "isam2_backend.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gauge_damping_factor.hpp"
#include "gnc_consolidation.hpp"
#include "graph_markers.hpp"
#include "gnss_factor.hpp"
#include "gnss_factor_refined.hpp"
#include "gtsam_adapter.hpp"
#include "keys.hpp"
#include "meridian/calib/calibration_set.hpp"
#include "meridian/common/frame.hpp"
#include "meridian/debug/telemetry.hpp"
#include "observability_inflation.hpp"
#include "pim_from_summary.hpp"
#include "robust_kernels.hpp"
#include "meridian/debug/telemetry_keys.hpp"

namespace meridian::backend {

namespace {

// A fold relinearizing more than this many variables is flagged as a loop-thrash / bad-fold event.
constexpr int kRelinEventThresh = 20;

TelemetrySink* fallback_sink() {
  static NullSink sink;
  return &sink;
}

std::string key_name(gtsam::Key key) {
  return gtsam::DefaultKeyFormatter(key);
}

}  // namespace

Isam2BackEnd::Isam2BackEnd(const BackendConfig& cfg, std::shared_ptr<const CalibrationSet> calib,
                           TelemetrySink* telemetry, bool deterministic)
    : cfg_(cfg),
      calib_(std::move(calib)),
      sink_(telemetry ? telemetry : fallback_sink()),
      deterministic_(deterministic),
      isam2_(make_isam2(cfg_.isam2_use_qr)),
      pcm_(chi2inv(cfg.pcm_chi2_alpha, 6), cfg.pcm_max_nodes) {
  // Online lever refinement engages only when the master switch is on AND the GNSS extrinsic
  // is present and individually flagged for refinement; otherwise the constant-lever path runs.
  if (cfg_.extrinsic_refine) {
    try {
      const Extrinsic& e = calib_->extrinsic(Frame::GnssLink);
      if (e.refine_online) {
        extrinsic_refine_gnss_ = true;
        extrinsic_offline_lever_ = e.T_parent_child.t;
        extrinsic_lever_ = e.T_parent_child.t;
      }
    } catch (const std::exception&) {
      // No GNSS extrinsic to refine; stay on the constant-lever path.
    }
  }
}

std::unique_ptr<gtsam::ISAM2> Isam2BackEnd::make_isam2(bool use_qr) const {
  gtsam::ISAM2Params params;
  // Per-variable-class relinearization thresholds, each in that variable's own tangent
  // ordering ('x' poses are rotation-first: radians then metres).
  gtsam::FastMap<char, gtsam::Vector> thresh;
  thresh['x'] = (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.10, 0.10, 0.10).finished();
  thresh['v'] = gtsam::Vector::Constant(3, 0.10);
  thresh['b'] = (gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4).finished();
  thresh['e'] = gtsam::Vector::Constant(6, 1e-3);
  thresh['g'] = (gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 5e-3, 5e-3, 5e-3).finished();
  params.relinearizeThreshold = thresh;
  params.relinearizeSkip = cfg_.isam2_relinearize_skip;
  params.optimizationParams = gtsam::ISAM2DoglegParams();
  params.cacheLinearizedFactors = true;
  params.evaluateNonlinearError = true;
  params.findUnusedFactorSlots = true;
  params.enableDetailedResults = true;
  params.factorization = use_qr ? gtsam::ISAM2Params::QR : gtsam::ISAM2Params::CHOLESKY;
  return std::make_unique<gtsam::ISAM2>(params);
}

void Isam2BackEnd::add_keyframe(KeyframePacket&& kf) {
  switch (kf.constraint_kind) {
    case KeyframePacket::ConstraintKind::AbsolutePrior: {
      if (last_kf_id_) {
        sink_->event(
            Level::Warn, keys::backend::AbsoluteIgnored,
            "absolute keyframe " + std::to_string(kf.id) + " after the first; packet dropped",
            tele_stamp());
        return;
      }
      // First keyframe: T_map_odom starts identity, so the odom pose is the map pose.
      // The gauge is fixed by damping this pose rather than by a hard prior.
      const double lambda = 1.0 / (cfg_.anchor_sigma * cfg_.anchor_sigma);
      new_graph_.emplace_shared<GaugeDampingFactor>(keyX(kf.id), lambda);
      new_values_.insert(keyX(kf.id), to_gtsam(kf.T_ref_body));
      chain_cov_.start(kf.id);
      record_keyframe(std::move(kf));
      return;
    }
    case KeyframePacket::ConstraintKind::RelativeBetween: {
      // A normal keyframe following a restart closes the inertial window: any live bridge
      // V/B is now marginalizable, so schedule it before this keyframe is folded.
      schedule_bridge_marginalization();
      // A between constraint must chain onto the previously accepted keyframe; a broken
      // chain is dropped whole rather than bridged with a fabricated constraint.
      if (!last_kf_id_ || kf.rel_to_id != *last_kf_id_) {
        const std::string expected = last_kf_id_ ? std::to_string(*last_kf_id_) : "none";
        sink_->event(Level::Error, keys::backend::Contiguity,
                     "kf " + std::to_string(kf.id) + " chains to " + std::to_string(kf.rel_to_id) +
                         " but last accepted is " + expected + "; packet dropped",
                     tele_stamp());
        return;
      }
      // The covariance block crosses the boundary already rotation-first and, by contract,
      // in covariance form. Audit the form at this one site rather than trust it silently:
      // an information-form block fed to a covariance noise model inverts every axis.
      GaussianBlock<6> cov_block = kf.constraint_cov;
      if (cov_block.form == GaussianBlock<6>::Form::Information) {
        sink_->event(Level::Warn, keys::backend::InfoForm,
                     "kf " + std::to_string(kf.id) + " shipped information-form cov; inverting",
                     tele_stamp());
        Eigen::Matrix<double, 6, 6> info = cov_block.M;
        ensure_psd(info);
        cov_block.M = info.inverse();
        cov_block.form = GaussianBlock<6>::Form::Covariance;
      }
      // Observability scores are defined in their own named frame; inflation rotates the
      // covariance about the factor (body) axes, so a non-body, non-eigvec report would
      // inflate the wrong axes. Skip inflation rather than misapply it.
      InflationResult inf;
      if (kf.observability.frame != Frame::Body && !kf.observability.eigvecs) {
        sink_->event(Level::Warn, keys::backend::ObsFrame,
                     "kf " + std::to_string(kf.id) +
                         " observability not in body frame; "
                         "inflation skipped",
                     tele_stamp());
        inf.cov = cov_block.M;
        inf.lambda = {1, 1, 1, 1, 1, 1};
      } else {
        inf = inflate_by_observability(cov_block, kf.observability, cfg_.obs_inflation_max,
                                       cfg_.obs_inflation_gamma, cfg_.degenerate_thresh,
                                       cfg_.degenerate_lock);
      }
      publish_observability(kf.id, kf.observability, inf);
      Eigen::Matrix<double, 6, 6> cov = inf.cov;  // rotation-first
      if (ensure_psd(cov)) {
        sink_->event(Level::Warn, keys::backend::PsdClamp,
                     "constraint covariance of kf " + std::to_string(kf.id) + " clamped to PSD",
                     tele_stamp());
      }
      new_graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
          keyX(kf.rel_to_id), keyX(kf.id), to_gtsam(kf.T_relto_this),
          gtsam::noiseModel::Gaussian::Covariance(cov));
      new_values_.insert(keyX(kf.id), to_gtsam(T_map_odom_ * kf.T_ref_body));
      chain_cov_.extend(kf.rel_to_id, kf.id, kf.T_relto_this, reorder_gtsam_to_meridian(cov));
      // Accumulate platform excitation since the lever variable was added; refinement engages
      // only once both rotation and translation pass the gate (a free extrinsic without
      // excitation is a canonical indeterminate-system cause).
      if (extrinsic_added_ && !extrinsic_excited_) {
        extrinsic_rot_accum_ += 2.0 * std::acos(std::min(1.0, std::abs(kf.T_relto_this.q.w())));
        extrinsic_trans_accum_ += kf.T_relto_this.t.norm();
        if (extrinsic_rot_accum_ >= cfg_.extrinsic_excite_rot &&
            extrinsic_trans_accum_ >= cfg_.extrinsic_excite_trans) {
          extrinsic_excited_ = true;
          sink_->event(Level::Info, keys::backend::ExtrinsicExcited,
                       "gnss lever refinement engaged (excitation reached)", tele_stamp());
        }
      }
      record_keyframe(std::move(kf));
      return;
    }
    case KeyframePacket::ConstraintKind::ImuPreintegration: {
      add_restart_imu_edge(std::move(kf));
      return;
    }
  }
}

void Isam2BackEnd::add_restart_imu_edge(KeyframePacket&& kf) {
  // The restart bridge needs the same contiguity guarantee as a between edge: it chains
  // onto the previous keyframe and carries the IMU summary for that exact interval.
  if (!kf.kinematics_included || !kf.imu_summary.has_value() || !last_kf_id_ ||
      kf.rel_to_id != *last_kf_id_) {
    const std::string expected = last_kf_id_ ? std::to_string(*last_kf_id_) : "none";
    sink_->event(Level::Error, keys::backend::Contiguity,
                 "restart kf " + std::to_string(kf.id) + " chains to " +
                     std::to_string(kf.rel_to_id) + " but last accepted is " + expected +
                     " (or missing kinematics); packet dropped",
                 tele_stamp());
    return;
  }

  const std::uint64_t i = kf.rel_to_id;
  const std::uint64_t j = kf.id;

  // i was a pose-only steady-state node, so its velocity and bias enter the graph here.
  // Map-frame velocity seed for j: kf.v_ref is an odom-frame velocity, rotated into map by
  // the current map<-odom alignment (translation does not affect a free vector).
  const gtsam::Vector3 v_i = gtsam::Vector3::Zero();
  const gtsam::Vector3 v_j = T_map_odom_.q * kf.v_ref;
  const gtsam::imuBias::ConstantBias bias(kf.b_a, kf.b_g);

  // Insert each inertial variable only if it is not already live: a second restart that
  // chains onto a still-inertial node (back-to-back restarts, or any restart under
  // keep_inertial) would otherwise re-insert an existing key and throw from Values::insert.
  const auto ensure = [&](gtsam::Key k, const auto& val) {
    if (new_values_.exists(k) || isam2_->valueExists(k)) {
      return false;
    }
    new_values_.insert(k, val);
    return true;
  };
  const bool created_v_i = ensure(keyV(i), v_i);
  const bool created_b_i = ensure(keyB(i), bias);
  ensure(keyV(j), v_j);
  ensure(keyB(j), bias);
  ensure(keyX(j), to_gtsam(T_map_odom_ * kf.T_ref_body));

  // Pin the older-side inertial variables only when this bridge created them; without the
  // pin ISAM2 reports an indeterminate system on the fresh inertial block, but re-pinning an
  // already-constrained node would double-count.
  if (created_v_i) {
    new_graph_.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
        keyV(i), v_i, gtsam::noiseModel::Isotropic::Sigma(3, 10.0));
  }
  if (created_b_i) {
    new_graph_.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
        keyB(i), bias, gtsam::noiseModel::Isotropic::Sigma(6, 1.0));
  }

  const gtsam::PreintegratedCombinedMeasurements pim = pim_from_summary(*kf.imu_summary, cfg_.imu);
  new_graph_.emplace_shared<gtsam::CombinedImuFactor>(keyX(i), keyV(i), keyX(j), keyV(j), keyB(i),
                                                      keyB(j), pim);

  // The bridge replaces a between edge in the pose chain. Extend chain_cov_ across i->j with
  // the pose block of the pim covariance: the 15x15 is rotation-first [theta,p,v,...]; take
  // its 6x6 {theta,p} sub-block and reorder to translation-first [rho; phi]. The relative
  // pose for the transport is the odometry hint i->j (both hints share the odom frame).
  // Rows/cols 0:3 are theta, 3:6 are p; copy the four 3x3 quadrants into a [theta, p] block.
  const Eigen::Matrix<double, 15, 15> pim_cov = pim.preintMeasCov();
  Eigen::Matrix<double, 6, 6> pose_cov_rotfirst = Eigen::Matrix<double, 6, 6>::Zero();
  pose_cov_rotfirst.topLeftCorner<3, 3>() = pim_cov.topLeftCorner<3, 3>();
  pose_cov_rotfirst.topRightCorner<3, 3>() = pim_cov.block<3, 3>(0, 3);
  pose_cov_rotfirst.bottomLeftCorner<3, 3>() = pim_cov.block<3, 3>(3, 0);
  pose_cov_rotfirst.bottomRightCorner<3, 3>() = pim_cov.block<3, 3>(3, 3);
  Eigen::Matrix<double, 6, 6> pose_cov = reorder_gtsam_to_meridian(pose_cov_rotfirst);
  // The pim position block is the gravity-free preintegrated increment; the true relative
  // translation also carries v_i*dt, whose uncertainty (the loose V(i) prior) is not in the
  // pim. Inflate the translation block by Var(v_i)*dt^2 so the bridge edge stays conservative
  // for the loop/PCM chain covariance, which would otherwise under-trust nothing and admit
  // bad cross-restart loops. Translation is the leading 3x3 in Meridian (translation-first).
  const double dt = static_cast<double>(kf.imu_summary->t_j - kf.imu_summary->t_i) * 1e-9;
  constexpr double kVelPriorVar = 100.0;  // (10 m/s)^2, matching the V(i) prior sigma
  pose_cov.topLeftCorner<3, 3>() += (kVelPriorVar * dt * dt) * Eigen::Matrix3d::Identity();
  const Pose T_i_j = kf_records_.at(i).T_ref_body.inverse() * kf.T_ref_body;
  chain_cov_.extend(i, j, T_i_j, pose_cov);

  pending_bridges_.push_back(BridgeRecord{i, j});
  sink_->event(Level::Info, keys::backend::RestartBridge,
               "restart bridge " + std::to_string(i) + "->" + std::to_string(j), tele_stamp());

  record_keyframe(std::move(kf));
}

void Isam2BackEnd::add_loop_constraint(const LoopConstraint& lc) {
  // A loop below the detector's fitness floor is rejected before it ever reaches PCM: a poor
  // geometric match is unlikely to be consistent and only inflates the consistency graph.
  if (lc.fitness < cfg_.loop_min_fitness) {
    ++diag_.num_loops_rejected;
    sink_->event(Level::Info, keys::backend::LoopRejectedPcm,
                 "loop " + std::to_string(lc.from_id) + "->" + std::to_string(lc.to_id) +
                     " fitness " + std::to_string(lc.fitness) + " below floor; dropped",
                 tele_stamp());
    return;
  }
  // Buffer the loop; the next optimize() runs PCM and admits the consistent max-clique.
  pcm_.add(lc);
}

void Isam2BackEnd::process_pending_loops(Timestamp ts) {
  // Re-judge the established loop set first so its removals ride this same fold; it reads the last
  // committed estimate, which is stable until the update below.
  if (deterministic_ && cfg_.gnc_consolidate_interval > 0 &&
      admitted_since_consolidate_ >= cfg_.gnc_consolidate_interval && !loop_factor_index_.empty()) {
    run_gnc_consolidation(ts);
    admitted_since_consolidate_ = 0;
  }

  const Pcm::PoseFn pose_fn = [this](std::uint64_t id) -> std::optional<Pose> {
    if (estimate_cache_.exists(keyX(id))) {
      return from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(id)));
    }
    return std::nullopt;
  };
  const Pcm::ChainCovFn chain_fn = [this](std::uint64_t a, std::uint64_t b) {
    return chain_cov_.between(a, b);
  };

  const PcmDecision decision = pcm_.update(pose_fn, chain_fn);

  // Rejections do not depend on the fold; apply and report them immediately.
  for (const std::size_t h : decision.to_reject) {
    const LoopConstraint& lc = pcm_.at(h);
    publish_loop_marker(lc, /*accepted=*/false, ts);
    pcm_.mark_rejected(h);
    ++diag_.num_loops_rejected;
    sink_->event(Level::Info, keys::backend::LoopRejectedPcm,
                 "loop " + std::to_string(lc.from_id) + "->" + std::to_string(lc.to_id) +
                     " PCM-inconsistent; dropped",
                 ts);
  }

  // Stage the newly consistent clique as committed-Huber between factors, recording each factor's
  // slot in the staged batch so its global index can be recovered after the update commits.
  for (const std::size_t h : decision.to_admit) {
    const LoopConstraint& lc = pcm_.at(h);
    Eigen::Matrix<double, 6, 6> cov_m = lc.cov.M;  // translation-first
    if (lc.cov.form == PoseCov6::Form::Information) {
      ensure_psd(cov_m);
      cov_m = cov_m.inverse();
    }
    Eigen::Matrix<double, 6, 6> cov_gtsam = reorder_meridian_to_gtsam(cov_m);  // rotation-first
    ensure_psd(cov_gtsam);
    staged_loop_slots_.emplace_back(h, static_cast<std::size_t>(new_graph_.size()));
    new_graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        keyX(lc.from_id), keyX(lc.to_id), to_gtsam(lc.T_from_to),
        make_huber_noise(cov_gtsam, cfg_.loop_huber_k));
    batch_has_loop_ = true;
  }

  // Queue any displaced in-graph loop for removal in this same fold.
  for (const std::size_t h : decision.to_evict) {
    const auto it = loop_factor_index_.find(h);
    if (it != loop_factor_index_.end()) {
      remove_indices_.push_back(it->second);
      staged_evict_handles_.push_back(h);
    }
  }
}

void Isam2BackEnd::finalize_pending_loops(const gtsam::ISAM2Result& result, Timestamp ts) {
  for (const auto& [handle, slot] : staged_loop_slots_) {
    // newFactorsIndices is in 1-to-1 order with the factors handed to update(), so the slot
    // recorded at stage time recovers this loop's live factor index for later eviction.
    if (slot < result.newFactorsIndices.size()) {
      loop_factor_index_[handle] = result.newFactorsIndices[slot];
    }
    pcm_.mark_admitted(handle);
    ++diag_.num_loops;
    ++admitted_since_consolidate_;
    const LoopConstraint& lc = pcm_.at(handle);
    publish_loop_marker(lc, /*accepted=*/true, ts);
    sink_->event(
        Level::Info, keys::backend::LoopAccepted,
        "loop " + std::to_string(lc.from_id) + "->" + std::to_string(lc.to_id) + " admitted", ts);
  }
  for (const std::size_t handle : staged_evict_handles_) {
    // Eviction is a displacement, not a rejection: the loop returns to Pending and may be
    // re-admitted by a later clique, so it drops out of the in-graph count but is not tallied
    // as rejected.
    pcm_.mark_evicted(handle);
    loop_factor_index_.erase(handle);
    if (diag_.num_loops > 0) {
      --diag_.num_loops;
    }
  }
  // Consolidation rejections committed: the factor is gone, so retire the loop in PCM (no retry).
  for (const std::size_t handle : staged_gnc_reject_handles_) {
    publish_loop_marker(pcm_.at(handle), /*accepted=*/false, ts);
    pcm_.mark_rejected(handle);
    loop_factor_index_.erase(handle);
    if (diag_.num_loops > 0) {
      --diag_.num_loops;
    }
    ++diag_.num_loops_rejected;
  }
  staged_loop_slots_.clear();
  staged_evict_handles_.clear();
  staged_gnc_reject_handles_.clear();
}

void Isam2BackEnd::abandon_pending_loops() {
  // The fold was dropped: the staged loop factors never entered, and neither the scheduled
  // evictions nor the consolidation removals applied, so every loop keeps its prior PCM state
  // (still in the graph, still re-judgeable) and retries on the next fold.
  staged_loop_slots_.clear();
  staged_evict_handles_.clear();
  staged_gnc_reject_handles_.clear();
}

void Isam2BackEnd::run_gnc_consolidation(Timestamp ts) {
  if (loop_factor_index_.empty()) {
    return;
  }
  // Iterate the in-graph loops in a fixed (handle-sorted) order: the loop set drives factor
  // insertion order in the sub-problem, and a hash-map order would make the batch
  // non-deterministic.
  std::vector<std::size_t> handles;
  handles.reserve(loop_factor_index_.size());
  for (const auto& [handle, idx] : loop_factor_index_) {
    (void)idx;
    handles.push_back(handle);
  }
  std::sort(handles.begin(), handles.end());

  std::uint64_t lo = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t hi = 0;
  for (const std::size_t handle : handles) {
    const LoopConstraint& lc = pcm_.at(handle);
    lo = std::min({lo, lc.from_id, lc.to_id});
    hi = std::max({hi, lc.from_id, lc.to_id});
  }

  GncConsolidationInput in;
  in.barc2 = chi2inv(cfg_.pcm_chi2_alpha, 6);
  in.reject_w = cfg_.gnc_reject_w;
  std::optional<std::uint64_t> prev;
  for (const std::uint64_t id : kf_order_) {
    if (id < lo || id > hi || !estimate_cache_.exists(keyX(id))) {
      continue;
    }
    const Pose X = from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(id)));
    in.keyframes.push_back(id);
    in.estimate[id] = X;
    if (prev) {
      if (const auto c = chain_cov_.between(*prev, id)) {
        GncOdom e;
        e.from_id = *prev;
        e.to_id = id;
        e.T_from_to = in.estimate.at(*prev).inverse() * X;
        e.cov = *c;
        ensure_psd(e.cov);  // GTSAM Gaussian::Covariance requires a PD matrix
        in.odom.push_back(e);
      }
    }
    prev = id;
  }
  for (const std::size_t handle : handles) {
    const LoopConstraint& lc = pcm_.at(handle);
    GncLoop g;
    g.handle = handle;
    g.from_id = lc.from_id;
    g.to_id = lc.to_id;
    g.T_from_to = lc.T_from_to;
    g.cov = (lc.cov.form == PoseCov6::Form::Information) ? lc.cov.M.inverse().eval() : lc.cov.M;
    ensure_psd(g.cov);  // GTSAM Gaussian::Covariance requires a PD matrix
    in.loops.push_back(g);
  }

  const std::vector<std::size_t> rejected = gnc_consolidate(in);
  for (const std::size_t handle : rejected) {
    const auto it = loop_factor_index_.find(handle);
    if (it == loop_factor_index_.end()) {
      continue;
    }
    // Stage the live factor for removal in this fold; defer the PCM-reject bookkeeping until the
    // fold commits so an abandoned update does not orphan a still-live loop factor.
    remove_indices_.push_back(it->second);
    staged_gnc_reject_handles_.push_back(handle);
  }
  if (!staged_gnc_reject_handles_.empty()) {
    sink_->event(Level::Info, keys::backend::LoopRejectedGnc,
                 "batch GNC flagged " + std::to_string(staged_gnc_reject_handles_.size()) +
                     " loop(s) for removal",
                 ts);
  }
}

void Isam2BackEnd::publish_loop_marker(const LoopConstraint& lc, bool accepted, Timestamp ts) {
  if (!sink_->enabled(keys::backend::LoopEdge)) {
    return;
  }
  if (!estimate_cache_.exists(keyX(lc.from_id)) || !estimate_cache_.exists(keyX(lc.to_id))) {
    return;
  }
  const Pose a = from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(lc.from_id)));
  const Pose b = from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(lc.to_id)));
  Marker m;
  m.type = Marker::Type::LineList;
  m.frame = Frame::Map;
  m.ns = keys::backend::LoopEdge;
  m.id = static_cast<std::int32_t>((lc.to_id << 20) ^ lc.from_id);
  m.points = {a.t.cast<float>(), b.t.cast<float>()};
  m.color = accepted ? std::array<float, 4>{0.0F, 1.0F, 0.0F, 1.0F}
                     : std::array<float, 4>{1.0F, 0.0F, 0.0F, 1.0F};
  m.scale = 0.15F;
  sink_->marker(m, ts);
}

std::optional<Isam2BackEnd::Bracket> Isam2BackEnd::find_bracket(Timestamp stamp,
                                                                std::uint64_t hint) const {
  // Only keyframes already in the estimate can anchor a factor; kf_order_ is in insert =
  // time order, so the bracket is the last estimated keyframe at/before the fix and the
  // first one after it. The hint just biases logging; the scan is authoritative.
  (void)hint;
  std::optional<std::uint64_t> prev;  // last estimated kf with stamp <= fix
  for (const std::uint64_t id : kf_order_) {
    if (!estimate_cache_.exists(keyX(id))) {
      continue;
    }
    const Timestamp s = kf_records_.at(id).stamp;
    if (s <= stamp) {
      prev = id;
    } else {
      // First estimated keyframe strictly after the fix: it closes the bracket.
      if (prev) {
        const Timestamp s_prev = kf_records_.at(*prev).stamp;
        if (s == s_prev) {
          return Bracket{*prev, *prev, true};  // zero-width interval, treat as endpoint
        }
        return Bracket{*prev, id, false};
      }
      // Fix precedes every estimated keyframe; clamp to the earliest as an endpoint.
      return Bracket{id, id, true};
    }
  }
  if (prev) {
    // Fix is at or after the latest estimated keyframe; no successor yet.
    return Bracket{*prev, *prev, true};
  }
  return std::nullopt;
}

void Isam2BackEnd::add_absolute(const GnssFix& fix, std::uint64_t nearest_kf_id) {
  if (!cfg_.gnss_enabled || gnss_auto_disabled_) {
    sink_->event(Level::Debug, keys::backend::GnssSkip,
                 "gnss disabled; fix near kf " + std::to_string(nearest_kf_id) + " dropped",
                 tele_stamp());
    return;
  }

  // The antenna lever arm in body is required to relate a fix to a body pose. A rig without
  // a GNSS extrinsic cannot use GNSS; drop quietly rather than throw out of the driver.
  Eigen::Vector3d lever;
  try {
    lever = calib_->extrinsic(Frame::GnssLink).T_parent_child.t;
  } catch (const std::exception&) {
    sink_->event(Level::Warn, keys::backend::GnssSkip, "no GnssLink extrinsic; fix dropped", tele_stamp());
    return;
  }

  // Quality floor before anything persistent: a no-fix or excessively-uncertain measurement must
  // neither seed the permanent ENU origin nor poison the datum fit. This guards the pre-lock path
  // (the post-lock path gates again through gnss_gate_).
  if (fix.fix == GnssFix::FixType::None || fix.cov_enu.trace() > cfg_.gnss_max_cov) {
    sink_->event(Level::Debug, keys::backend::GnssSkip, "gnss fix below quality floor; dropped",
                 tele_stamp());
    return;
  }

  // First accepted fix fixes the ENU tangent-plane origin. Origin set != datum locked: the
  // map<-ENU transform G is fit only once the track is observable.
  if (!gnss_origin_.set) {
    gnss_origin_.lat0_deg = fix.lat_deg;
    gnss_origin_.lon0_deg = fix.lon_deg;
    gnss_origin_.alt0_m = fix.alt_m;
    gnss_origin_.set = true;
  }
  const Eigen::Vector3d p_enu = lla_to_enu(fix.lat_deg, fix.lon_deg, fix.alt_m, gnss_origin_);

  const std::optional<Bracket> br = find_bracket(fix.stamp, nearest_kf_id);
  if (!br) {
    // No estimated keyframe to anchor against yet (before the first optimize()).
    sink_->event(Level::Debug, keys::backend::GnssSkip, "no estimated keyframe to anchor gnss fix",
                 tele_stamp());
    return;
  }

  if (!datum_locked_) {
    // Antenna position in map at the fix instant, interpolated across the bracket exactly as the
    // post-lock factor evaluates it, so the datum is fit against consistent correspondences.
    const Eigen::Vector3d ant_map = body_pose_at_fix(*br, fix.stamp) * lever;

    // Speed from the anchor keyframe and its predecessor: ||dp|| / dt over the last interval.
    // A single keyframe (or a coincident-stamp neighbour) yields zero, which the moving-fix
    // gate treats as stationary.
    double speed = 0.0;
    for (std::size_t k = 0; k + 1 < kf_order_.size(); ++k) {
      if (kf_order_[k + 1] != br->i) {
        continue;
      }
      const std::uint64_t a = kf_order_[k];
      const std::uint64_t b = kf_order_[k + 1];
      if (!estimate_cache_.exists(keyX(a)) || !estimate_cache_.exists(keyX(b))) {
        break;
      }
      const double dt = to_seconds(kf_records_.at(b).stamp - kf_records_.at(a).stamp);
      if (dt > 0.0) {
        const Pose Xa = from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(a)));
        const Pose Xb = from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(b)));
        speed = (Xb.t - Xa.t).norm() / dt;
      }
      break;
    }

    datum_.add(p_enu, ant_map, speed, fix.stamp);
    const double yaw_sigma_max_rad = cfg_.gnss_datum_yaw_sigma_max * M_PI / 180.0;
    const DatumResult lock =
        datum_.try_lock(cfg_.gnss_min_baseline, cfg_.gnss_min_excitation, cfg_.gnss_min_speed,
                        cfg_.gnss_min_moving_fixes, yaw_sigma_max_rad);
    if (!lock.locked) {
      return;  // keep buffering; buffered fixes never become factors, only the datum does
    }

    // Lock the datum: G enters the graph as a weakly-anchored variable. Translation is tight
    // (the origin is well defined), yaw carries the fitted sigma, and roll/pitch stay tight
    // since the planar fit pins them to zero. The prior order is rotation-first [r;t].
    new_values_.insert(kKeyG, to_gtsam(lock.T_map_enu));
    Eigen::Matrix<double, 6, 1> sigmas;
    constexpr double kTightRotSigma = 1e-3;   // [rad] roll/pitch
    constexpr double kTightTransSigma = 0.5;  // [m]
    sigmas << kTightRotSigma, kTightRotSigma, std::max(lock.yaw_sigma_rad, kTightRotSigma),
        kTightTransSigma, kTightTransSigma, kTightTransSigma;
    new_graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        kKeyG, to_gtsam(lock.T_map_enu), gtsam::noiseModel::Diagonal::Sigmas(sigmas));

    datum_locked_ = true;
    datum_just_locked_ = true;
    diag_.datum_locked = true;
    gnss_last_antenna_ = ant_map;
    gnss_travelled_since_admit_ = 0.0;

    const double yaw = std::atan2(2.0 * (lock.T_map_enu.q.w() * lock.T_map_enu.q.z() +
                                         lock.T_map_enu.q.x() * lock.T_map_enu.q.y()),
                                  1.0 - 2.0 * (lock.T_map_enu.q.y() * lock.T_map_enu.q.y() +
                                               lock.T_map_enu.q.z() * lock.T_map_enu.q.z()));
    sink_->event(Level::Info, keys::backend::DatumLocked,
                 "datum locked: yaw=" + std::to_string(yaw) +
                     " rad, yaw_sigma=" + std::to_string(lock.yaw_sigma_rad) +
                     " rad, fixes=" + std::to_string(datum_.size()),
                 tele_stamp());
    return;
  }

  admit_gnss_fix(fix, p_enu, *br);
}

Pose Isam2BackEnd::body_pose_at_fix(const Bracket& br, Timestamp stamp) const {
  if (br.single) {
    return from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(br.i)));
  }
  const Timestamp si = kf_records_.at(br.i).stamp;
  const Timestamp sj = kf_records_.at(br.j).stamp;
  const Timestamp span = sj - si;
  const double beta = span > 0 ? static_cast<double>(stamp - si) / static_cast<double>(span) : 0.0;
  if (beta <= 0.0) {
    return from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(br.i)));
  }
  if (beta >= 1.0) {
    return from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(br.j)));
  }
  const Pose Xi = from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(br.i)));
  const Pose Xj = from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(br.j)));
  return Xi.boxplus(Xj.boxminus(Xi) * beta);
}

void Isam2BackEnd::admit_gnss_fix(const GnssFix& fix, const Eigen::Vector3d& p_enu,
                                  const Bracket& br) {
  // Map-frame antenna position of this fix, for travel accounting and the residual check.
  Eigen::Vector3d lever;
  try {
    lever = calib_->extrinsic(Frame::GnssLink).T_parent_child.t;
  } catch (const std::exception&) {
    return;
  }
  // While refining online, the antenna geometry uses the current best lever (offline until the
  // estimate freezes); the refined-lever factor itself reads the live E(GnssLink) estimate.
  if (extrinsic_refine_gnss_) {
    lever = extrinsic_lever_;
  }

  // beta / endpoint select the factor form; the antenna position itself comes from the shared
  // interpolation helper so it matches the datum-buffering geometry exactly.
  double beta = 0.0;
  if (!br.single) {
    const Timestamp si = kf_records_.at(br.i).stamp;
    const Timestamp sj = kf_records_.at(br.j).stamp;
    const Timestamp span = sj - si;
    beta = span > 0 ? static_cast<double>(fix.stamp - si) / static_cast<double>(span) : 0.0;
  }
  const bool endpoint = br.single || beta <= 0.0 || beta >= 1.0;
  // The single keyframe an endpoint factor differentiates against: i when the fix sits at or
  // before i, j when it sits at or after j.
  const std::uint64_t end_id = (!br.single && beta >= 1.0) ? br.j : br.i;

  const Eigen::Vector3d ant_map = body_pose_at_fix(br, fix.stamp) * lever;

  // Travel since the last admitted fix accumulates the antenna arc length over consecutive
  // post-lock fixes; the gate decimates on it.
  if (gnss_last_antenna_) {
    gnss_travelled_since_admit_ += (ant_map - *gnss_last_antenna_).norm();
  }
  gnss_last_antenna_ = ant_map;

  double marginal_pos_trace = std::numeric_limits<double>::max();
  if (const std::optional<PoseCov6> m = latest_pose_marginal()) {
    // PoseCov6 is translation-first [rho; phi]; the position block is the leading 3x3.
    marginal_pos_trace = m->M.topLeftCorner<3, 3>().trace();
  }

  const GnssGate::Decision decision =
      gnss_gate_.evaluate(fix, marginal_pos_trace, gnss_travelled_since_admit_, cfg_);
  if (decision != GnssGate::Decision::Accept) {
    const char* reason = decision == GnssGate::Decision::RejectQuality   ? "quality"
                         : decision == GnssGate::Decision::SkipConfident ? "confident"
                                                                         : "spacing";
    sink_->event(Level::Debug, keys::backend::GnssSkip, std::string("gnss fix dropped: ") + reason,
                 tele_stamp());
    return;
  }

  // FM-6 health: the whitened residual against cov_enu. The datum maps map->ENU, so the
  // predicted ENU antenna is G^{-1} * ant_map; chi2 = r^T cov_enu^{-1} r.
  if (estimate_cache_.exists(kKeyG)) {
    const Pose G = from_gtsam(estimate_cache_.at<gtsam::Pose3>(kKeyG));
    const Eigen::Vector3d r = G.inverse() * ant_map - p_enu;
    const double chi2 = r.dot(fix.cov_enu.ldlt().solve(r));
    if (chi2 > chi2inv(0.99, 3)) {
      // A fix that fails the health check is dropped, never graphed; only a sustained run of
      // failures escalates to auto-disable.
      ++gnss_consecutive_chi2_reject_;
      if (gnss_consecutive_chi2_reject_ >= cfg_.gnss_reacq_persist) {
        gnss_auto_disabled_ = true;
        sink_->event(Level::Error, keys::backend::GnssDisabled,
                     "gnss auto-disabled after " + std::to_string(gnss_consecutive_chi2_reject_) +
                         " consecutive chi2-failing fixes",
                     tele_stamp());
      } else {
        sink_->event(Level::Debug, keys::backend::GnssSkip, "gnss fix failed chi2 health; dropped",
                     tele_stamp());
      }
      return;
    }
    gnss_consecutive_chi2_reject_ = 0;
    sink_->vec(keys::backend::GnssResidual, r, tele_stamp(), "e,n,u");
  }

  // First post-lock fix under refinement seeds the E(GnssLink) variable at the offline lever with
  // a loose prior; it stays pinned by that prior (and the constant-lever factors) until the
  // excitation gate opens, then the refined-lever factors below make it observable.
  if (extrinsic_refine_gnss_ && !extrinsic_added_) {
    const gtsam::Pose3 e_seed(gtsam::Rot3::Identity(), gtsam::Point3(extrinsic_offline_lever_));
    new_values_.insert(keyE(Frame::GnssLink), e_seed);
    new_graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        keyE(Frame::GnssLink), e_seed,
        gtsam::noiseModel::Isotropic::Sigma(6, cfg_.extrinsic_refine_sigma));
    extrinsic_added_ = true;
  }

  const gtsam::SharedNoiseModel noise = make_gnss_noise(fix.cov_enu, cfg_.gnss_huber_k);
  const gtsam::Point3 lever_pt(lever);
  const gtsam::Point3 meas(p_enu);
  const bool use_refined = extrinsic_refine_gnss_ && extrinsic_excited_ && !extrinsic_frozen_;
  if (use_refined) {
    if (endpoint) {
      new_graph_.emplace_shared<GnssFactorRefinedEndpoint>(keyX(end_id), kKeyG,
                                                           keyE(Frame::GnssLink), meas, noise);
    } else {
      new_graph_.emplace_shared<GnssFactorRefined>(keyX(br.i), keyX(br.j), kKeyG,
                                                   keyE(Frame::GnssLink), beta, meas, noise);
    }
  } else if (endpoint) {
    new_graph_.emplace_shared<GnssFactorEndpoint>(keyX(end_id), kKeyG, lever_pt, meas, noise);
  } else {
    new_graph_.emplace_shared<GnssFactor>(keyX(br.i), keyX(br.j), kKeyG, beta, lever_pt, meas,
                                          noise);
  }
  ++diag_.num_gnss_factors;
  gnss_gate_.note_admitted();
  gnss_travelled_since_admit_ = 0.0;
}

GraphUpdate Isam2BackEnd::optimize() {
  const Timestamp ts = tele_stamp();
  // A clamp/freeze on the previous fold left a re-pin pending: pin E tight here so the stale
  // refined-lever factors evaluate at the safe lever and stop biasing the trajectory.
  if (extrinsic_repin_value_) {
    const gtsam::Pose3 pin(gtsam::Rot3::Identity(), gtsam::Point3(*extrinsic_repin_value_));
    new_graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        keyE(Frame::GnssLink), pin,
        gtsam::noiseModel::Isotropic::Sigma(6, cfg_.extrinsic_prior_sigma));
    extrinsic_repin_value_.reset();
  }
  // Run PCM before measuring the staged batch: admitted loops add factors and evictions add
  // removals, both of which the fold below must see.
  if (pcm_.pending_count() > 0 || !loop_factor_index_.empty()) {
    process_pending_loops(ts);
  }

  const bool nothing_staged = new_graph_.empty() && new_values_.empty() && remove_indices_.empty();
  if (nothing_staged && kf_order_.empty()) {
    return {};
  }

  const int folded = staged_count_;
  diag_.last_optimize_diverged = false;

  // When V/B are scheduled for removal, the fold update must make them Bayes-tree leaves so
  // marginalizeLeaves can act right after; the group map encodes that elimination order.
  const boost::optional<gtsam::FastMap<gtsam::Key, int>> constrained = bridge_constraint_groups();

  const Clock::time_point t0 = Clock::now();
  gtsam::ISAM2Result result;
  const bool committed = run_update_with_recovery(result, ts, constrained);

  const int extra = batch_has_loop_ ? cfg_.extra_iters_loop : cfg_.extra_iters_normal;
  for (int k = 0; k < extra; ++k) {
    try {
      isam2_->update();
    } catch (const gtsam::IndeterminantLinearSystemException& e) {
      // The state is left at the last completed pass; stop refining instead of crashing.
      sink_->event(Level::Error, keys::backend::Indeterminate,
                   "extra pass failed near " + key_name(e.nearbyVariable()), ts);
      break;
    }
  }

  if (committed) {
    perform_bridge_marginalization(ts);
  }

  estimate_cache_ = isam2_->calculateEstimate();
  marginal_cache_.reset();
  diag_.isam_update_ms = ms_since(t0);

  if (!committed) {
    // The batch was abandoned; purge the keyframes it staged so the chain stays consistent.
    rollback_uncommitted_keyframes();
  }

  // Loop bookkeeping reads the refreshed estimate (for markers) and the update result (for factor
  // indices); on an abandoned fold the staged loops never entered, so they stay pending.
  if (committed) {
    finalize_pending_loops(result, ts);
  } else {
    abandon_pending_loops();
  }

  if (committed) {
    update_extrinsic(ts);
  }

  GraphUpdate update = build_graph_update();

  if (last_kf_id_ && estimate_cache_.exists(keyX(*last_kf_id_))) {
    const Pose X_last = from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(*last_kf_id_)));
    T_map_odom_ = X_last * kf_records_.at(*last_kf_id_).T_ref_body.inverse();
    sink_->pose(keys::map::Keyframe, X_last, Frame::Map, ts);
  }

  publish_graph_markers(ts);

  diag_.num_keyframes = kf_order_.size();
  diag_.chi2 = result.errorAfter ? *result.errorAfter : 0.0;
  diag_.variables_relinearized = static_cast<int>(result.variablesRelinearized);
  diag_.optimize_lag = folded;

  sink_->scalar(keys::backend::Chi2, diag_.chi2, ts);
  sink_->scalar(keys::backend::NFactors, static_cast<double>(isam2_->getFactorsUnsafe().size()), ts);
  sink_->scalar(keys::backend::NLoops, static_cast<double>(diag_.num_loops), ts);
  sink_->scalar(keys::backend::NGnss, static_cast<double>(diag_.num_gnss_factors), ts);
  sink_->scalar(keys::backend::UpdateMs, diag_.isam_update_ms, ts);
  sink_->scalar(keys::backend::RelinCount, static_cast<double>(diag_.variables_relinearized), ts);
  sink_->scalar(keys::backend::OptimizeLag, static_cast<double>(folded), ts);
  sink_->timing(keys::stage::BackendOptimize, diag_.isam_update_ms, ts);
  // A fold that relinearizes a large fraction of the trajectory is the loop-thrash / bad-fold
  // signature worth flagging for the operator, with the Bayes-tree size for context.
  if (diag_.variables_relinearized > kRelinEventThresh) {
    sink_->event(Level::Debug, keys::backend::Relinearize,
                 "relinearized " + std::to_string(diag_.variables_relinearized) + " of " +
                     std::to_string(isam2_->getFactorsUnsafe().size()) + " factors",
                 ts);
  }

  new_graph_.resize(0);
  new_values_.clear();
  remove_indices_.clear();
  batch_has_loop_ = false;
  datum_just_locked_ = false;
  staged_count_ = 0;

  return update;
}

bool Isam2BackEnd::run_update_with_recovery(
    gtsam::ISAM2Result& result, Timestamp ts,
    const boost::optional<gtsam::FastMap<gtsam::Key, int>>& constrained) {
  // Snapshot the last-good graph before mutating: a throwing update can leave isam2_
  // unusable, and the estimate cache still matches this snapshot.
  const gtsam::NonlinearFactorGraph last_good = isam2_->getFactorsUnsafe();
  try {
    result = isam2_->update(new_graph_, new_values_, remove_indices_, constrained);
    return true;
  } catch (const gtsam::IndeterminantLinearSystemException& e) {
    sink_->event(Level::Error, keys::backend::Indeterminate,
                 "indeterminate system near " + key_name(e.nearbyVariable()) + "; retrying on QR",
                 ts);
  }

  // Retry once on QR: rebuild from the last-good graph/estimate, then re-apply the batch.
  try {
    isam2_ = make_isam2(true);
    if (!last_good.empty()) {
      isam2_->update(last_good, estimate_cache_);
    }
    result = isam2_->update(new_graph_, new_values_, remove_indices_, constrained);
    return true;
  } catch (const gtsam::IndeterminantLinearSystemException& e) {
    sink_->event(Level::Error, keys::backend::Indeterminate,
                 "QR retry failed near " + key_name(e.nearbyVariable()) + "; dropping the batch",
                 ts);
  }

  // Final fallback: abandon the staged batch and restore the last-good state.
  isam2_ = make_isam2(true);
  if (!last_good.empty()) {
    try {
      result = isam2_->update(last_good, estimate_cache_);
    } catch (const gtsam::IndeterminantLinearSystemException& e) {
      sink_->event(Level::Error, keys::backend::Indeterminate,
                   "last-good restore failed near " + key_name(e.nearbyVariable()), ts);
    }
  }
  diag_.last_optimize_diverged = true;
  ++diag_.fallback_count;
  sink_->scalar(keys::backend::FallbackCount, static_cast<double>(diag_.fallback_count), ts);
  return false;
}

GraphUpdate Isam2BackEnd::build_graph_update() {
  GraphUpdate update;
  update.loop_closed = batch_has_loop_;
  for (const std::uint64_t id : kf_order_) {
    if (!estimate_cache_.exists(keyX(id))) {
      continue;  // staged but not optimized (or abandoned with a dropped batch)
    }
    KfRecord& rec = kf_records_.at(id);
    const Pose pose = from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(id)));
    bool moved = !rec.ever_emitted;  // first appearance is always news downstream
    if (!moved) {
      // One threshold gates both tangent blocks: metres on ||rho||, radians on ||phi||.
      const Eigen::Matrix<double, 6, 1> xi = pose.boxminus(rec.last_emitted);
      moved = xi.head<3>().norm() > cfg_.reintegrate_thresh ||
              xi.tail<3>().norm() > cfg_.reintegrate_thresh;
    }
    if (moved) {
      update.moved.push_back(GraphUpdate::Moved{id, pose, std::nullopt});
      rec.last_emitted = pose;
      rec.ever_emitted = true;
    }
  }
  return update;
}

std::vector<StampedPose> Isam2BackEnd::corrected_trajectory() const {
  std::vector<StampedPose> out;
  out.reserve(kf_order_.size());
  for (const std::uint64_t id : kf_order_) {
    if (!estimate_cache_.exists(keyX(id))) {
      continue;
    }
    const KfRecord& rec = kf_records_.at(id);
    const Pose pose = from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(id)));
    out.push_back(StampedPose{rec.stamp, id, pose});
  }
  return out;
}

std::shared_ptr<const CalibrationSet> Isam2BackEnd::refined_calibration() const {
  // Once the online lever has frozen, publish the snapshot carrying the refined value; otherwise
  // the offline calibration is the better estimate.
  return refined_calib_ ? refined_calib_ : calib_;
}

void Isam2BackEnd::publish_refined_lever() {
  auto snap = std::make_shared<CalibrationSet>(*calib_);
  for (Extrinsic& e : snap->extrinsics) {
    if (e.child == Frame::GnssLink) {
      e.T_parent_child.t = extrinsic_lever_;
      ++e.version;
    }
  }
  ++snap->version;
  refined_calib_ = snap;
}

void Isam2BackEnd::update_extrinsic(Timestamp ts) {
  if (!extrinsic_refine_gnss_ || !extrinsic_added_ || extrinsic_frozen_) {
    return;
  }
  if (!estimate_cache_.exists(keyE(Frame::GnssLink))) {
    return;
  }
  const Eigen::Vector3d lever =
      from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyE(Frame::GnssLink))).t;

  // FM-5 sanity clamp: a refined lever leaving the offline box is untrustworthy. Revert to the
  // offline value, publish that safe value, and stop refining rather than bias the trajectory.
  if ((lever - extrinsic_offline_lever_).norm() > cfg_.extrinsic_max_dev) {
    extrinsic_lever_ = extrinsic_offline_lever_;
    extrinsic_frozen_ = true;
    extrinsic_repin_value_ = extrinsic_offline_lever_;  // re-pin E to offline next fold (FM-5)
    publish_refined_lever();
    sink_->event(Level::Warn, keys::backend::ExtrinsicClamped,
                 "gnss lever left the offline box; reverted and frozen", ts);
    return;
  }
  extrinsic_lever_ = lever;

  // The lever is only observable, and so only published, once the excitation gate has opened.
  if (!extrinsic_excited_) {
    return;
  }
  publish_refined_lever();

  // Convergence freeze: once the lever's marginal is tight, stop treating it as free.
  try {
    const gtsam::Matrix m = isam2_->marginalCovariance(keyE(Frame::GnssLink));
    // GTSAM Pose3 tangent is rotation-first; the translation block is the bottom-right 3x3.
    if (m.bottomRightCorner<3, 3>().trace() < cfg_.extrinsic_freeze_cov) {
      extrinsic_frozen_ = true;
      extrinsic_repin_value_ = extrinsic_lever_;  // re-pin E to the converged value next fold
      sink_->event(Level::Info, keys::backend::ExtrinsicFrozen,
                   "gnss lever refined and frozen: [" + std::to_string(extrinsic_lever_.x()) +
                       ", " + std::to_string(extrinsic_lever_.y()) + ", " +
                       std::to_string(extrinsic_lever_.z()) + "]",
                   ts);
    }
  } catch (const std::exception&) {
    // Marginal not yet available (E just entered the estimate); retry next fold.
  }
}

BackEndDiagnostics Isam2BackEnd::diagnostics() const {
  return diag_;
}

bool Isam2BackEnd::wants_immediate_optimize() const {
  return batch_has_loop_ || datum_just_locked_;
}

std::optional<PoseCov6> Isam2BackEnd::latest_pose_marginal() const {
  if (marginal_cache_) {
    return marginal_cache_;
  }
  if (!last_kf_id_ || estimate_cache_.empty()) {
    return std::nullopt;
  }
  try {
    PoseCov6 cov;
    cov.form = PoseCov6::Form::Covariance;
    cov.M = reorder_gtsam_to_meridian(isam2_->marginalCovariance(keyX(*last_kf_id_)));
    marginal_cache_ = cov;
  } catch (const std::exception&) {
    // The latest keyframe may be staged but not yet optimized; report no marginal.
    return std::nullopt;
  }
  return marginal_cache_;
}

std::optional<Pose> Isam2BackEnd::pose_of(std::uint64_t id) const {
  if (!estimate_cache_.exists(keyX(id))) {
    return std::nullopt;
  }
  return from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(id)));
}

std::optional<Eigen::Matrix<double, 6, 6>> Isam2BackEnd::chain_cov_between(
    std::uint64_t a, std::uint64_t b) const {
  return chain_cov_.between(a, b);
}

void Isam2BackEnd::publish_graph_markers(Timestamp ts) const {
  if (!sink_->enabled(keys::backend::GraphNodes) && !sink_->enabled(keys::backend::GraphEdges)) return;
  // Same traversal as write_g2o: keyframe pose vertices + the relative (between/loop) edges.
  std::vector<std::pair<std::uint64_t, Pose>> nodes;
  nodes.reserve(kf_order_.size());
  for (const std::uint64_t id : kf_order_) {
    if (estimate_cache_.exists(keyX(id))) {
      nodes.emplace_back(id, from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(id))));
    }
  }
  std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;
  for (const auto& f : isam2_->getFactorsUnsafe()) {
    const auto bf = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(f);
    if (!bf) continue;
    const gtsam::Symbol s1(bf->key1());
    const gtsam::Symbol s2(bf->key2());
    if (s1.chr() == 'x' && s2.chr() == 'x') {
      edges.emplace_back(s1.index(), s2.index());
    }
  }
  emitGraphMarkers(sink_, nodes, edges, ts);
}

void Isam2BackEnd::write_g2o(const std::string& path) const {
  // Export only the pose sub-graph: Pose3 vertices for every estimated keyframe plus the
  // relative (between/loop) edges. Velocity/bias/GNSS/gauge factors are not g2o-representable,
  // so filtering them keeps the snapshot a clean, loadable trajectory graph.
  gtsam::Values poses;
  for (const std::uint64_t id : kf_order_) {
    if (estimate_cache_.exists(keyX(id))) {
      poses.insert(keyX(id), estimate_cache_.at<gtsam::Pose3>(keyX(id)));
    }
  }
  gtsam::NonlinearFactorGraph edges;
  for (const auto& f : isam2_->getFactorsUnsafe()) {
    if (f && boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(f)) {
      edges.push_back(f);
    }
  }
  try {
    gtsam::writeG2o(edges, poses, path);
  } catch (const std::exception& e) {
    sink_->event(Level::Warn, keys::backend::G2oSnapshot,
                 std::string("g2o snapshot failed: ") + e.what(), tele_stamp());
  }
}

void Isam2BackEnd::stage_for_test(gtsam::NonlinearFactorGraph graph, gtsam::Values values) {
  new_graph_.push_back(graph);
  new_values_.insert(values);
}

void Isam2BackEnd::record_keyframe(KeyframePacket&& kf) {
  KfRecord rec;
  rec.stamp = kf.stamp;
  rec.cloud_body = std::move(kf.cloud_body);
  rec.image = std::move(kf.image);
  rec.observability = kf.observability;
  rec.ref_frame = kf.ref_frame;
  rec.T_ref_body = kf.T_ref_body;
  rec.calib_version = kf.calib_version;
  kf_records_.emplace(kf.id, std::move(rec));
  kf_order_.push_back(kf.id);
  last_kf_id_ = kf.id;
  ++staged_count_;
  // A new keyframe outdates the cached "latest" marginal; recompute it lazily next query.
  marginal_cache_.reset();
}

void Isam2BackEnd::schedule_bridge_marginalization() {
  if (cfg_.keep_inertial || pending_bridges_.empty()) {
    return;
  }
  for (const BridgeRecord& b : pending_bridges_) {
    pending_marginalize_.push_back(keyV(b.i));
    pending_marginalize_.push_back(keyB(b.i));
    pending_marginalize_.push_back(keyV(b.j));
    pending_marginalize_.push_back(keyB(b.j));
  }
  pending_bridges_.clear();
}

boost::optional<gtsam::FastMap<gtsam::Key, int>> Isam2BackEnd::bridge_constraint_groups() const {
  if (pending_marginalize_.empty()) {
    return boost::none;
  }
  // Group 0 is eliminated first (leaf-ward) and holds exactly the V/B to be marginalized.
  // Group 1 must list EVERY other key, because any key the map omits silently defaults to
  // group 0 and would be forced into the leaf set too. Enumerate the actual live key set
  // (the graph's linearization point plus anything staged this batch) rather than a fixed
  // list of key classes, so the GNSS origin G and any extrinsic E are never miscategorised.
  // Both key containers iterate in sorted key order, so the result is deterministic.
  gtsam::FastMap<gtsam::Key, int> groups;
  for (const gtsam::Key key : isam2_->getLinearizationPoint().keys()) {
    groups[key] = 1;
  }
  for (const gtsam::Key key : new_values_.keys()) {
    groups[key] = 1;
  }
  for (const gtsam::Key key : pending_marginalize_) {
    groups[key] = 0;
  }
  return groups;
}

void Isam2BackEnd::perform_bridge_marginalization(Timestamp ts) {
  if (pending_marginalize_.empty()) {
    return;
  }
  // Only keys already folded into the Bayes tree are eligible; a key staged but not yet
  // present (or one that is not a leaf this round) is retried on the next fold.
  gtsam::KeyList ready;
  std::vector<gtsam::Key> still_pending;
  for (const gtsam::Key key : pending_marginalize_) {
    if (isam2_->valueExists(key)) {
      ready.push_back(key);
    } else {
      still_pending.push_back(key);
    }
  }
  if (ready.empty()) {
    pending_marginalize_ = std::move(still_pending);
    return;
  }
  try {
    isam2_->marginalizeLeaves(ready);
    pending_marginalize_ = std::move(still_pending);
  } catch (const std::exception&) {
    // gtsam throws when a scheduled key is not actually a leaf; keep all of them and retry
    // next fold rather than dropping the inertial states inconsistently.
    sink_->event(Level::Warn, keys::backend::MarginalizeSkip,
                 "bridge V/B not leaf-eligible; marginalization deferred", ts);
  }
}

void Isam2BackEnd::publish_observability(std::uint64_t id, const ObservabilityReport& obs,
                                         const InflationResult& inf) {
  if (sink_->enabled(keys::backend::ObsMin)) {
    double obs_min = obs.score[0];
    for (const double s : obs.score) {
      obs_min = std::min(obs_min, s);
    }
    sink_->scalar(keys::backend::ObsMin, obs_min, tele_stamp());
    // Per-axis inflation multipliers in factor (rotation-first) order, so a degenerate axis
    // is visible over the whole run, not only inside one keyframe.
    Eigen::Matrix<double, 6, 1> lam;
    for (int k = 0; k < 6; ++k) {
      lam(k) = inf.lambda[static_cast<std::size_t>(k)];
    }
    const std::string key = std::string(keys::backend::ObservabilityPrefix) + std::to_string(id);
    sink_->vec(key.c_str(), lam, tele_stamp(), "rx,ry,rz,tx,ty,tz");
  }
  if (inf.any_locked) {
    sink_->event(Level::Warn, keys::backend::Degenerate,
                 "kf " + std::to_string(id) + " has a degenerate axis locked to max inflation",
                 tele_stamp());
  }
}

void Isam2BackEnd::rollback_uncommitted_keyframes() {
  // After a recovery path abandons the staged batch, keyframes recorded for that batch have
  // no variable in the estimate. Drop their bookkeeping so last_kf_id_ points at the newest
  // committed keyframe; the next packet then fails the contiguity gate cleanly instead of
  // staging an edge against a variable that was never linearized.
  std::vector<std::uint64_t> kept;
  kept.reserve(kf_order_.size());
  for (const std::uint64_t id : kf_order_) {
    if (estimate_cache_.exists(keyX(id))) {
      kept.push_back(id);
    } else {
      kf_records_.erase(id);
      chain_cov_.forget(id);
    }
  }
  if (kept.size() == kf_order_.size()) {
    return;
  }
  // A rolled-back bridge keyframe never reached the graph, so drop its pending record and
  // any V/B keys it scheduled; their X/V/B were never linearized.
  pending_bridges_.erase(
      std::remove_if(pending_bridges_.begin(), pending_bridges_.end(),
                     [&](const BridgeRecord& b) { return !estimate_cache_.exists(keyX(b.j)); }),
      pending_bridges_.end());
  pending_marginalize_.erase(
      std::remove_if(pending_marginalize_.begin(), pending_marginalize_.end(),
                     [&](gtsam::Key key) { return !estimate_cache_.exists(key); }),
      pending_marginalize_.end());
  kf_order_ = std::move(kept);
  last_kf_id_ = kf_order_.empty() ? std::nullopt : std::optional<std::uint64_t>(kf_order_.back());
}

Timestamp Isam2BackEnd::tele_stamp() const {
  return last_kf_id_ ? kf_records_.at(*last_kf_id_).stamp : 0;
}

}  // namespace meridian::backend
