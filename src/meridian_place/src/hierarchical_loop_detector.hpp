#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "meridian/place/iloop_detector.hpp"
#include "gicp_verify.hpp"
#include "sc_descriptor.hpp"
#include "submap_accumulator.hpp"

namespace meridian {

class TelemetrySink;

// The staged hierarchical detector: Stage A (Scan Context retrieval) -> Stage C (GICP
// verification) -> Stage D (single-loop odometry self-test). Stage B (STD/BTC re-rank) is
// skipped this pass behind the ScanContextDb retrieval seam. Runs synchronously on the
// back-end path; deterministic when constructed deterministic.
class HierarchicalLoopDetector final : public ILoopDetector {
 public:
  HierarchicalLoopDetector(const PlaceConfig& cfg, std::shared_ptr<const KeyframeStore> store,
                           KeyframePoseSource pose_source, bool deterministic,
                           TelemetrySink* telemetry);

  void add_keyframe(std::uint64_t id, PointCloudPtr cloud, const Pose& T_map_body) override;
  std::vector<LoopConstraint> detect() override;
  void on_graph_update(const GraphUpdate& update) override;
  LoopDiagnostics diagnostics() const override { return diag_; }

 private:
  // Keyframes old enough (id + time gap) and not too far (loose xy) to be loop candidates.
  std::vector<std::uint64_t> eligible_for(std::uint64_t query) const;

  PlaceConfig cfg_;
  std::shared_ptr<const KeyframeStore> store_;
  KeyframePoseSource pose_source_;

  ScanContextDb sc_db_;
  SubmapAccumulator submaps_;
  GicpVerifier gicp_;
  double chi2_threshold_;
  TelemetrySink* sink_ = nullptr;  // borrowed; place/* diagnostics, null = no telemetry

  std::vector<std::uint64_t> added_ids_;  // ascending; the retrieval candidate pool
  std::uint64_t newest_id_ = 0;
  bool has_newest_ = false;
  std::uint64_t last_emitted_to_ = 0;
  bool has_emitted_ = false;
  std::uint64_t kf_since_detect_ = 0;

  LoopDiagnostics diag_;
};

}  // namespace meridian
