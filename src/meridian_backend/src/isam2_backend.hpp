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

  // Test seam: appends directly to the staged batch the next optimize() folds.
  void stage_for_test(gtsam::NonlinearFactorGraph graph, gtsam::Values values);

private:
  std::unique_ptr<gtsam::ISAM2> make_isam2(bool use_qr) const;
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
  // Folds one accepted GNSS fix into the staged batch after the datum is locked: gates it,
  // builds the interpolated or endpoint factor, and runs the sustained-rejection auto-disable.
  void admit_gnss_fix(const GnssFix& fix, const Eigen::Vector3d& p_enu, const Bracket& br);
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

  BackEndDiagnostics diag_;
};

}  // namespace meridian::backend
