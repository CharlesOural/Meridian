#pragma once

#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Core>
#include <boost/optional.hpp>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "chain_covariance.hpp"
#include "datum.hpp"
#include "geodetic.hpp"
#include "gnss_gate.hpp"
#include "kf_record.hpp"
#include "meridian/backend/ibackend.hpp"
#include "observability_inflation.hpp"
#include "pcm.hpp"

namespace meridian::backend {

// Incremental pose-graph optimizer on iSAM2. add_* calls only stage factors and initial
// values; optimize() folds the whole staged batch in one incremental update and reports
// which keyframes moved enough for downstream re-integration. All methods run on one
// serial driver thread; nothing here is locked.
//
// Covariance boundary: everything entering a GTSAM noise model is rotation-first
// [rx,ry,rz,tx,ty,tz]; everything leaving toward Meridian types is reordered back to
// translation-first [rho; phi].
class Isam2BackEnd final : public IBackEnd {
public:
  Isam2BackEnd(const BackendConfig& cfg, std::shared_ptr<const CalibrationSet> calib,
               TelemetrySink* telemetry, bool deterministic);

  void add_keyframe(KeyframePacket&& kf) override;
  void add_loop_constraint(const LoopConstraint& lc) override;
  void add_absolute(const GnssFix& fix, std::uint64_t nearest_kf_id) override;

  // Identifies one inertial restart bridge: the CombinedImuFactor edge i->j brought its own
  // V(i),B(i),V(j),B(j) into the graph. The V/B keys are derivable through keyV/keyB(i|j),
  // so storing the two ids is enough to schedule their later marginalization.
  struct BridgeRecord {
    std::uint64_t i = 0;
    std::uint64_t j = 0;
  };

  GraphUpdate optimize() override;

  std::vector<StampedPose> corrected_trajectory() const override;
  std::shared_ptr<const CalibrationSet> refined_calibration() const override;
  BackEndDiagnostics diagnostics() const override;
  bool wants_immediate_optimize() const override;
  std::optional<PoseCov6> latest_pose_marginal() const override;
  std::optional<Pose> pose_of(std::uint64_t id) const override;
  std::optional<Eigen::Matrix<double, 6, 6>> chain_cov_between(std::uint64_t a,
                                                               std::uint64_t b) const override;
  void write_g2o(const std::string& path) const override;

  // Test seam: appends directly to the staged batch the next optimize() folds.
  void stage_for_test(gtsam::NonlinearFactorGraph graph, gtsam::Values values);

private:
  std::unique_ptr<gtsam::ISAM2> make_isam2(bool use_qr) const;
  // Publishes the pose graph (keyframe vertices + relative edges) as live markers.
  void publish_graph_markers(Timestamp ts) const;
  // Folds the staged batch into isam2_, recovering from an indeterminate linear system
  // by rebuilding on QR; returns false if the batch had to be abandoned. constrained, when
  // present, fixes the elimination grouping for the bridge-clearing update (group 0 keys
  // become Bayes-tree leaves so they can be marginalized straight after).
  bool run_update_with_recovery(
      gtsam::ISAM2Result& result, Timestamp ts,
      const boost::optional<gtsam::FastMap<gtsam::Key, int>>& constrained);
  GraphUpdate build_graph_update();
  void record_keyframe(KeyframePacket&& kf);
  // Stages the restart bridge: new V(i),B(i),V(j),B(j), loose priors pinning the freshly
  // created i-side states, and the CombinedImuFactor for the i->j interval.
  void add_restart_imu_edge(KeyframePacket&& kf);
  // Promotes the not-yet-marginalized bridges' V/B into the next-fold marginalize set. Run
  // when a normal keyframe follows a bridge and inertial variables are not being kept.
  void schedule_bridge_marginalization();
  // Builds the constrainedKeys group map for the fold update that clears a bridge: group 0
  // is the V/B about to be marginalized (forced leaf-ward), group 1 is every other live key
  // (X poses in kf_order_ plus any V/B still pending). Empty when nothing is pending.
  boost::optional<gtsam::FastMap<gtsam::Key, int>> bridge_constraint_groups() const;
  // After the fold update, marginalizes the scheduled V/B that are now leaves; any key not
  // yet present or not leaf-eligible stays pending and is retried next fold.
  void perform_bridge_marginalization(Timestamp ts);
  // Emits per-axis observability inflation telemetry and a marker when an axis is locked.
  void publish_observability(std::uint64_t id, const ObservabilityReport& obs,
                             const InflationResult& inf);

  // Runs one PCM pass over the buffered loops against the current estimate: drops the loops PCM
  // judges inconsistent with the admitted consensus, stages the newly consistent clique as
  // committed-Huber between factors (recording each factor's batch slot for index reconciliation),
  // and queues displaced loops for removal in this same fold. No-op when no loop is buffered or
  // in the graph.
  void process_pending_loops(Timestamp ts);
  // After a committed fold: resolves each staged loop's global factor index from the update result,
  // flips PCM state to in-graph/evicted, updates counters and edge markers, and triggers the batch
  // consolidation when enough loops have accumulated.
  void finalize_pending_loops(const gtsam::ISAM2Result& result, Timestamp ts);
  // After an abandoned fold: the staged loop factors never entered and any scheduled eviction did
  // not happen, so leave every loop in its prior PCM state to retry next fold.
  void abandon_pending_loops();
  // Off-live batch-GNC re-judgement of the in-graph loop sub-graph; loops GNC drives below the
  // reject weight are scheduled for removal and returned to the PCM buffer as rejected.
  void run_gnc_consolidation(Timestamp ts);
  // Publishes a map-frame line marker between a loop's endpoints (green admitted, red rejected).
  void publish_loop_marker(const LoopConstraint& lc, bool accepted, Timestamp ts);

  // The keyframe pair straddling a fix timestamp. When single is true the fix lands on (or
  // past) one node and only `i` is used (endpoint factor); otherwise stamp(i) <= fix <=
  // stamp(j) and beta interpolates between them. ids index into kf_records_.
  struct Bracket {
    std::uint64_t i = 0;
    std::uint64_t j = 0;
    bool single = false;
  };
  // Locates the bracketing keyframe pair for a fix stamp, seeded by the nearest-keyframe
  // hint and searching kf_order_ deterministically. nullopt when no keyframe is in the
  // estimate yet (nothing to anchor against).
  std::optional<Bracket> find_bracket(Timestamp stamp, std::uint64_t hint) const;
  // Body pose at a fix instant: the SE(3) interpolation of the bracket (or the single anchor).
  // Both the datum-buffering correspondence and the post-lock factor/residual evaluate the
  // antenna here, so the datum is fit against the exact geometry the factor later uses.
  Pose body_pose_at_fix(const Bracket& br, Timestamp stamp) const;
  // Folds one accepted GNSS fix into the staged batch after the datum is locked: gates it,
  // builds the interpolated or endpoint factor, and runs the sustained-rejection auto-disable.
  void admit_gnss_fix(const GnssFix& fix, const Eigen::Vector3d& p_enu, const Bracket& br);
  // Post-optimize maintenance of the online GNSS-lever extrinsic (§10.2): tracks the current
  // estimate, freezes + publishes it once its marginal converges, and rejects a lever that
  // strays past the offline box (FM-5).
  void update_extrinsic(Timestamp ts);
  // Builds and stores the versioned CalibrationSet snapshot carrying the current refined lever.
  void publish_refined_lever();
  // Drops bookkeeping for keyframes whose variables are not in the estimate after an
  // abandoned batch, restoring chain consistency.
  void rollback_uncommitted_keyframes();
  Timestamp tele_stamp() const;

  BackendConfig cfg_;
  std::shared_ptr<const CalibrationSet> calib_;
  TelemetrySink* sink_ = nullptr;
  bool deterministic_ = false;  // reserved: no time-based branches yet

  std::unique_ptr<gtsam::ISAM2> isam2_;

  // Staged batch, consumed and cleared by every optimize().
  gtsam::NonlinearFactorGraph new_graph_;
  gtsam::Values new_values_;
  gtsam::FactorIndices remove_indices_;
  bool batch_has_loop_ = false;
  int staged_count_ = 0;

  // Estimate after the last successful update; output paths read only keys present in it.
  gtsam::Values estimate_cache_;
  mutable std::optional<PoseCov6> marginal_cache_;

  // kf_order_ drives every output path so results never depend on hash-map order.
  std::vector<std::uint64_t> kf_order_;
  std::unordered_map<std::uint64_t, KfRecord> kf_records_;
  std::optional<std::uint64_t> last_kf_id_;
  ChainCovariance chain_cov_;
  Pose T_map_odom_;

  // Restart-bridge bookkeeping. pending_bridges_ holds bridges whose V/B are still live;
  // once a normal keyframe follows, their V/B keys move into pending_marginalize_ and are
  // removed at the next fold (unless cfg_.keep_inertial pins them forever). A key stays in
  // pending_marginalize_ until it is actually a Bayes-tree leaf and marginalizes cleanly.
  std::vector<BridgeRecord> pending_bridges_;
  std::vector<gtsam::Key> pending_marginalize_;

  // GNSS / datum state. gnss_origin_ fixes the ENU tangent plane on the first accepted fix;
  // datum_ buffers correspondences until the map<-ENU transform G is fit and locked. Once
  // locked, fixes pass through gnss_gate_ and become factors tying the trajectory to G.
  GeodeticDatum gnss_origin_;
  DatumInitializer datum_;
  GnssGate gnss_gate_;
  bool datum_locked_ = false;
  // Set when the datum locks in the current batch so optimize() folds immediately; cleared
  // at the end of optimize().
  bool datum_just_locked_ = false;
  // FM-6 auto-disable: once tripped, every later fix is dropped silently.
  bool gnss_auto_disabled_ = false;
  // Arc length travelled (map-frame antenna) since the last admitted fix, for spacing decimation.
  double gnss_travelled_since_admit_ = 0.0;
  // Map-frame antenna position of the last fix seen after lock, to accumulate travel.
  std::optional<Eigen::Vector3d> gnss_last_antenna_;
  // Sustained rejection counter: consecutive post-lock fixes whose chi2 exceeds the gate.
  int gnss_consecutive_chi2_reject_ = 0;

  // Online GNSS-lever extrinsic refinement (§10, off by default). When active the antenna lever
  // becomes an E(GnssLink) Pose3 variable: pinned by a loose prior and constant-lever factors
  // until the platform delivers enough rotational + translational excitation, then made observable
  // by refined-lever factors; once its marginal converges the value is frozen and published, and a
  // lever that leaves the offline box is rejected (FM-5).
  bool extrinsic_refine_gnss_ = false;
  bool extrinsic_added_ = false;
  bool extrinsic_excited_ = false;
  bool extrinsic_frozen_ = false;
  double extrinsic_rot_accum_ = 0.0;    // cumulative |rotation| since the variable was added [rad]
  double extrinsic_trans_accum_ = 0.0;  // cumulative |translation| since added [m]
  Eigen::Vector3d extrinsic_offline_lever_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d extrinsic_lever_ =
      Eigen::Vector3d::Zero();  // current best (offline, then frozen)
  // When a clamp or freeze fires, the lever to re-pin E to with a tight prior on the next fold;
  // this neutralizes the refined-lever factors so they can no longer bias the trajectory.
  std::optional<Eigen::Vector3d> extrinsic_repin_value_;
  std::shared_ptr<const CalibrationSet> refined_calib_;

  // Loop closures. pcm_ buffers every accepted LoopConstraint and decides admit/evict/reject via
  // pairwise-consistency max-clique. loop_factor_index_ maps an in-graph loop's PCM handle to its
  // live iSAM2 factor index so eviction/consolidation can remove it. staged_loop_slots_ /
  // staged_evict_handles_ hold this fold's pending admit (handle -> new_graph_ slot) and eviction
  // handles until the fold commits. admitted_since_consolidate_ counts admissions toward the next
  // batch-GNC pass.
  Pcm pcm_;
  std::unordered_map<std::size_t, gtsam::FactorIndex> loop_factor_index_;
  std::vector<std::pair<std::size_t, std::size_t>> staged_loop_slots_;
  std::vector<std::size_t> staged_evict_handles_;
  // Loops the batch consolidation flagged this fold: their removal is staged into remove_indices_,
  // but the PCM-reject bookkeeping is deferred to a committed fold so an abandoned update leaves
  // the (still-live) loop intact and re-judgeable.
  std::vector<std::size_t> staged_gnc_reject_handles_;
  int admitted_since_consolidate_ = 0;

  BackEndDiagnostics diag_;
};

}  // namespace meridian::backend
