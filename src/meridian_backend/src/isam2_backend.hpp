#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "chain_covariance.hpp"
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
  // by rebuilding on QR; returns false if the batch had to be abandoned.
  bool run_update_with_recovery(gtsam::ISAM2Result& result, Timestamp ts);
  GraphUpdate build_graph_update();
  void record_keyframe(KeyframePacket&& kf);
  // Emits per-axis observability inflation telemetry and a marker when an axis is locked.
  void publish_observability(std::uint64_t id, const ObservabilityReport& obs,
                             const InflationResult& inf);
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

  BackEndDiagnostics diag_;
};

}  // namespace meridian::backend
