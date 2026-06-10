#include "isam2_backend.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/slam/BetweenFactor.h>

#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gauge_damping_factor.hpp"
#include "gtsam_adapter.hpp"
#include "keys.hpp"
#include "meridian/debug/telemetry.hpp"
#include "observability_inflation.hpp"
#include "telemetry_keys.hpp"

namespace meridian::backend {

namespace {

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
      isam2_(make_isam2(cfg_.isam2_use_qr)) {}

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
            Level::Warn, "backend/absolute_ignored",
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
      // A between constraint must chain onto the previously accepted keyframe; a broken
      // chain is dropped whole rather than bridged with a fabricated constraint.
      if (!last_kf_id_ || kf.rel_to_id != *last_kf_id_) {
        const std::string expected = last_kf_id_ ? std::to_string(*last_kf_id_) : "none";
        sink_->event(Level::Error, kTeleContiguity,
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
        sink_->event(Level::Warn, kTeleInfoForm,
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
        sink_->event(Level::Warn, kTeleObsFrame,
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
        sink_->event(Level::Warn, kTelePsdClamp,
                     "constraint covariance of kf " + std::to_string(kf.id) + " clamped to PSD",
                     tele_stamp());
      }
      new_graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
          keyX(kf.rel_to_id), keyX(kf.id), to_gtsam(kf.T_relto_this),
          gtsam::noiseModel::Gaussian::Covariance(cov));
      new_values_.insert(keyX(kf.id), to_gtsam(T_map_odom_ * kf.T_ref_body));
      chain_cov_.extend(kf.rel_to_id, kf.id, kf.T_relto_this, reorder_gtsam_to_meridian(cov));
      record_keyframe(std::move(kf));
      return;
    }
    case KeyframePacket::ConstraintKind::ImuPreintegration: {
      sink_->event(Level::Warn, "backend/restart_unsupported",
                   "restart keyframe " + std::to_string(kf.id) + " dropped", tele_stamp());
      return;
    }
  }
}

void Isam2BackEnd::add_loop_constraint(const LoopConstraint& lc) {
  ++diag_.num_loops_rejected;
  sink_->event(Level::Warn, "backend/loops_unsupported",
               "loop " + std::to_string(lc.from_id) + "->" + std::to_string(lc.to_id) + " dropped",
               tele_stamp());
}

void Isam2BackEnd::add_absolute(const GnssFix& /*fix*/, std::uint64_t nearest_kf_id) {
  sink_->event(Level::Warn, "backend/gnss_unsupported",
               "gnss fix near kf " + std::to_string(nearest_kf_id) + " dropped", tele_stamp());
}

GraphUpdate Isam2BackEnd::optimize() {
  const bool nothing_staged = new_graph_.empty() && new_values_.empty() && remove_indices_.empty();
  if (nothing_staged && kf_order_.empty()) {
    return {};
  }

  const Timestamp ts = tele_stamp();
  const int folded = staged_count_;
  diag_.last_optimize_diverged = false;

  const Clock::time_point t0 = Clock::now();
  gtsam::ISAM2Result result;
  const bool committed = run_update_with_recovery(result, ts);

  const int extra = batch_has_loop_ ? cfg_.extra_iters_loop : cfg_.extra_iters_normal;
  for (int k = 0; k < extra; ++k) {
    try {
      isam2_->update();
    } catch (const gtsam::IndeterminantLinearSystemException& e) {
      // The state is left at the last completed pass; stop refining instead of crashing.
      sink_->event(Level::Error, kTeleIndeterminate,
                   "extra pass failed near " + key_name(e.nearbyVariable()), ts);
      break;
    }
  }

  estimate_cache_ = isam2_->calculateEstimate();
  marginal_cache_.reset();
  diag_.isam_update_ms = ms_since(t0);

  if (!committed) {
    // The batch was abandoned; purge the keyframes it staged so the chain stays consistent.
    rollback_uncommitted_keyframes();
  }

  GraphUpdate update = build_graph_update();

  if (last_kf_id_ && estimate_cache_.exists(keyX(*last_kf_id_))) {
    const Pose X_last = from_gtsam(estimate_cache_.at<gtsam::Pose3>(keyX(*last_kf_id_)));
    T_map_odom_ = X_last * kf_records_.at(*last_kf_id_).T_ref_body.inverse();
    sink_->pose(kTeleMapKeyframe, X_last, Frame::Map, ts);
  }

  diag_.num_keyframes = kf_order_.size();
  diag_.chi2 = result.errorAfter ? *result.errorAfter : 0.0;
  diag_.variables_relinearized = static_cast<int>(result.variablesRelinearized);
  diag_.optimize_lag = folded;

  sink_->scalar(kTeleChi2, diag_.chi2, ts);
  sink_->scalar(kTeleNFactors, static_cast<double>(isam2_->getFactorsUnsafe().size()), ts);
  sink_->scalar(kTeleUpdateMs, diag_.isam_update_ms, ts);
  sink_->scalar(kTeleRelinCount, static_cast<double>(diag_.variables_relinearized), ts);
  sink_->scalar(kTeleOptimizeLag, static_cast<double>(folded), ts);
  sink_->timing("backend.optimize", diag_.isam_update_ms, ts);

  new_graph_.resize(0);
  new_values_.clear();
  remove_indices_.clear();
  batch_has_loop_ = false;
  staged_count_ = 0;

  return update;
}

bool Isam2BackEnd::run_update_with_recovery(gtsam::ISAM2Result& result, Timestamp ts) {
  // Snapshot the last-good graph before mutating: a throwing update can leave isam2_
  // unusable, and the estimate cache still matches this snapshot.
  const gtsam::NonlinearFactorGraph last_good = isam2_->getFactorsUnsafe();
  try {
    result = isam2_->update(new_graph_, new_values_, remove_indices_);
    return true;
  } catch (const gtsam::IndeterminantLinearSystemException& e) {
    sink_->event(Level::Error, kTeleIndeterminate,
                 "indeterminate system near " + key_name(e.nearbyVariable()) + "; retrying on QR",
                 ts);
  }

  // Retry once on QR: rebuild from the last-good graph/estimate, then re-apply the batch.
  try {
    isam2_ = make_isam2(true);
    if (!last_good.empty()) {
      isam2_->update(last_good, estimate_cache_);
    }
    result = isam2_->update(new_graph_, new_values_, remove_indices_);
    return true;
  } catch (const gtsam::IndeterminantLinearSystemException& e) {
    sink_->event(Level::Error, kTeleIndeterminate,
                 "QR retry failed near " + key_name(e.nearbyVariable()) + "; dropping the batch",
                 ts);
  }

  // Final fallback: abandon the staged batch and restore the last-good state.
  isam2_ = make_isam2(true);
  if (!last_good.empty()) {
    try {
      result = isam2_->update(last_good, estimate_cache_);
    } catch (const gtsam::IndeterminantLinearSystemException& e) {
      sink_->event(Level::Error, kTeleIndeterminate,
                   "last-good restore failed near " + key_name(e.nearbyVariable()), ts);
    }
  }
  diag_.last_optimize_diverged = true;
  ++diag_.fallback_count;
  sink_->scalar(kTeleFallbackCount, static_cast<double>(diag_.fallback_count), ts);
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
  return calib_;
}

BackEndDiagnostics Isam2BackEnd::diagnostics() const {
  return diag_;
}

bool Isam2BackEnd::wants_immediate_optimize() const {
  return batch_has_loop_;
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

void Isam2BackEnd::publish_observability(std::uint64_t id, const ObservabilityReport& obs,
                                         const InflationResult& inf) {
  if (sink_->enabled(kTeleObsMin)) {
    double obs_min = obs.score[0];
    for (const double s : obs.score) {
      obs_min = std::min(obs_min, s);
    }
    sink_->scalar(kTeleObsMin, obs_min, tele_stamp());
    // Per-axis inflation multipliers in factor (rotation-first) order, so a degenerate axis
    // is visible over the whole run, not only inside one keyframe.
    Eigen::Matrix<double, 6, 1> lam;
    for (int k = 0; k < 6; ++k) {
      lam(k) = inf.lambda[static_cast<std::size_t>(k)];
    }
    const std::string key = kTeleObservabilityPrefix + std::to_string(id);
    sink_->vec(key.c_str(), lam, tele_stamp(), "rx,ry,rz,tx,ty,tz");
  }
  if (inf.any_locked) {
    sink_->event(Level::Warn, kTeleDegenerate,
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
  kf_order_ = std::move(kept);
  last_kf_id_ = kf_order_.empty() ? std::nullopt : std::optional<std::uint64_t>(kf_order_.back());
}

Timestamp Isam2BackEnd::tele_stamp() const {
  return last_kf_id_ ? kf_records_.at(*last_kf_id_).stamp : 0;
}

}  // namespace meridian::backend
