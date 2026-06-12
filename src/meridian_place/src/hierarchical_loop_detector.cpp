#include "hierarchical_loop_detector.hpp"

#include <algorithm>
#include <utility>

#include "meridian/debug/telemetry.hpp"
#include "meridian/debug/telemetry_keys.hpp"
#include "loop_cov.hpp"
#include "pcm_self_test.hpp"

namespace meridian {

HierarchicalLoopDetector::HierarchicalLoopDetector(const PlaceConfig& cfg,
                                                   std::shared_ptr<const KeyframeStore> store,
                                                   KeyframePoseSource pose_source,
                                                   bool deterministic, TelemetrySink* telemetry)
    : cfg_(cfg),
      store_(std::move(store)),
      pose_source_(std::move(pose_source)),
      sc_db_(cfg.sc_Nr, cfg.sc_Ns, cfg.sc_rmax, cfg.sc_knn, cfg.sc_dist_thresh),
      submaps_(
          store_,
          [this](std::uint64_t id) {
            return pose_source_.pose ? pose_source_.pose(id) : std::optional<Pose>{};
          },
          cfg.submap_window, cfg.submap_voxel, cfg.submap_cache),
      gicp_(cfg, deterministic),
      chi2_threshold_(chi2InvDof6(cfg.pcm_chi2_conf)),
      sink_(telemetry) {}

void HierarchicalLoopDetector::add_keyframe(std::uint64_t id, PointCloudPtr cloud,
                                            const Pose& /*T_map_body*/) {
  // The corrected pose is read back through pose_source_; the arg is the interface contract.
  ++diag_.keyframes;
  added_ids_.push_back(id);  // keyframes arrive in ascending id, so this stays sorted
  newest_id_ = id;
  has_newest_ = true;
  // Stage-A descriptor on the anchor submap (denser than one sweep); fall back to the single
  // cloud before the submap can compose.
  const std::shared_ptr<const PointCloud> sm = submaps_.submap(id);
  if (sm && !sm->empty())
    sc_db_.add(id, *sm);
  else if (cloud)
    sc_db_.add(id, *cloud);
  else
    sc_db_.add(id, PointCloud{});
}

std::vector<std::uint64_t> HierarchicalLoopDetector::eligible_for(std::uint64_t query) const {
  std::vector<std::uint64_t> out;
  const std::optional<Timestamp> q_stamp =
      pose_source_.stamp ? pose_source_.stamp(query) : std::optional<Timestamp>{};
  const std::optional<Pose> q_pose =
      pose_source_.pose ? pose_source_.pose(query) : std::optional<Pose>{};
  const std::int64_t min_time_ns = static_cast<std::int64_t>(cfg_.min_time_gap * 1e9);
  for (const std::uint64_t id : added_ids_) {
    if (id >= query) continue;
    if (query - id < static_cast<std::uint64_t>(cfg_.min_kf_gap)) continue;
    if (q_stamp && pose_source_.stamp) {
      const std::optional<Timestamp> s = pose_source_.stamp(id);
      if (s && (*q_stamp - *s) < min_time_ns) continue;
    }
    if (q_pose && pose_source_.pose) {
      const std::optional<Pose> p = pose_source_.pose(id);
      if (p && (q_pose->t.head<2>() - p->t.head<2>()).norm() > cfg_.sc_max_xy) continue;
    }
    out.push_back(id);
  }
  return out;
}

std::vector<LoopConstraint> HierarchicalLoopDetector::detect() {
  std::vector<LoopConstraint> loops;
  if (!has_newest_) return loops;
  if (++kf_since_detect_ < static_cast<std::uint64_t>(cfg_.detect_period_kf)) return loops;
  kf_since_detect_ = 0;

  const std::uint64_t query = newest_id_;
  if (has_emitted_ && query >= last_emitted_to_ &&
      query - last_emitted_to_ < static_cast<std::uint64_t>(cfg_.cooldown_kf)) {
    return loops;
  }

  const std::vector<std::uint64_t> eligible = eligible_for(query);
  const std::vector<ScCandidate> cands =
      eligible.empty() ? std::vector<ScCandidate>{}
                       : sc_db_.retrieve(query, eligible, cfg_.sc_topK);
  diag_.candidates += cands.size();

  const std::optional<Pose> q_pose =
      pose_source_.pose ? pose_source_.pose(query) : std::optional<Pose>{};
  const std::shared_ptr<const PointCloud> q_source =
      cfg_.gicp_source_submap ? submaps_.submap(query) : store_->cloud(query);
  const std::optional<ObservabilityReport> q_obs =
      pose_source_.obs ? pose_source_.obs(query) : std::optional<ObservabilityReport>{};

  double best_fitness = 0.0;
  if (q_pose && q_source && !q_source->empty()) {
    for (const ScCandidate& c : cands) {
      const std::optional<Pose> c_pose =
          pose_source_.pose ? pose_source_.pose(c.id) : std::optional<Pose>{};
      if (!c_pose) continue;
      const std::shared_ptr<const PointCloud> target = submaps_.submap(c.id);
      if (!target || target->empty()) continue;

      // GICP init: the corrected-odometry relative pose between the endpoints.
      const Pose init_from_to = c_pose->inverse() * (*q_pose);
      const VerifiedLoop v = gicp_.verify(*target, *q_source, init_from_to);
      best_fitness = std::max(best_fitness, v.fitness);
      if (!v.accepted) continue;
      ++diag_.verified;

      const std::optional<ObservabilityReport> c_obs =
          pose_source_.obs ? pose_source_.obs(c.id) : std::optional<ObservabilityReport>{};
      const PoseCov6 cov = shapeLoopCov(v.info_rot_first, v.fitness, cfg_, c_obs, q_obs);

      // Stage D: single-loop odometry self-test on the combined loop + chain covariance.
      Eigen::Matrix<double, 6, 6> combined = cov.M;
      if (pose_source_.chain_cov) {
        const std::optional<Eigen::Matrix<double, 6, 6>> chain =
            pose_source_.chain_cov(c.id, query);
        if (chain) combined += *chain;
      }
      if (!loopAgreesWithOdometry(init_from_to, v.T_from_to, combined, chi2_threshold_)) {
        ++diag_.self_test_rejected;
        continue;
      }

      LoopConstraint lc;
      lc.from_id = c.id;
      lc.to_id = query;
      lc.T_from_to = v.T_from_to;
      lc.cov = cov;
      lc.fitness = v.fitness;
      loops.push_back(lc);
      ++diag_.emitted;
      last_emitted_to_ = query;
      has_emitted_ = true;
    }
  }

  if (sink_ && sink_->enabled(keys::place::BestScDist)) {
    const Timestamp ts =
        pose_source_.stamp ? pose_source_.stamp(query).value_or(0) : static_cast<Timestamp>(0);
    sink_->scalar(keys::place::NEligible, static_cast<double>(eligible.size()), ts);
    sink_->scalar(keys::place::NCandidates, static_cast<double>(cands.size()), ts);
    sink_->scalar(keys::place::BestScDist, cands.empty() ? 1.0 : cands.front().sc_dist, ts);
    sink_->scalar(keys::place::BestFitness, best_fitness, ts);
    sink_->scalar(keys::place::Verified, static_cast<double>(diag_.verified), ts);
    sink_->scalar(keys::place::SelfTestRejected, static_cast<double>(diag_.self_test_rejected), ts);
    sink_->scalar(keys::place::Emitted, static_cast<double>(diag_.emitted), ts);
  }
  return loops;
}

void HierarchicalLoopDetector::on_graph_update(const GraphUpdate& update) {
  if (update.moved.empty()) return;
  std::vector<std::uint64_t> moved;
  moved.reserve(update.moved.size());
  for (const GraphUpdate::Moved& m : update.moved) moved.push_back(m.id);
  submaps_.invalidate(moved);
}

}  // namespace meridian
