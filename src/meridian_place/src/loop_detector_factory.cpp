#include <memory>
#include <utility>

#include "meridian/place/iloop_detector.hpp"
#include "hierarchical_loop_detector.hpp"

namespace meridian {
namespace {

// Placeholder detector: ingests keyframes but never proposes a loop. The hierarchical
// cascade (Scan Context retrieval, GICP verification, single-loop consistency) replaces
// it; until then the pipeline can wire L5 end to end with no behavioural change.
class NullLoopDetector final : public ILoopDetector {
 public:
  void add_keyframe(std::uint64_t /*id*/, PointCloudPtr /*cloud*/,
                    const Pose& /*T_map_body*/) override {
    ++diag_.keyframes;
  }
  std::vector<LoopConstraint> detect() override { return {}; }
  void on_graph_update(const GraphUpdate& /*update*/) override {}
  LoopDiagnostics diagnostics() const override { return diag_; }

 private:
  LoopDiagnostics diag_;
};

}  // namespace

std::unique_ptr<ILoopDetector> makeLoopDetector(const PlaceConfig& cfg,
                                                std::shared_ptr<const KeyframeStore> store,
                                                KeyframePoseSource pose_source, bool deterministic,
                                                TelemetrySink* telemetry) {
  if (!cfg.enable) return std::make_unique<NullLoopDetector>();
  switch (cfg.kind) {
    case PlaceKind::ScanContextPp:
      return std::make_unique<HierarchicalLoopDetector>(
          cfg, std::move(store), std::move(pose_source), deterministic, telemetry);
  }
  return std::make_unique<NullLoopDetector>();
}

}  // namespace meridian
