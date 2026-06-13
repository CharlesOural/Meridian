#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "meridian/common/cloud.hpp"
#include "meridian/common/graph_update.hpp"
#include "meridian/common/loop_constraint.hpp"
#include "meridian/common/observability.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"
#include "meridian/map/keyframe_store.hpp"

namespace meridian {

class TelemetrySink;

// Read-only views into the back-end's current estimate, supplied by the pipeline so L5
// never links L3. All queries are by keyframe id and return nullopt for an unknown id.
struct KeyframePoseSource {
  std::function<std::optional<Pose>(std::uint64_t)> pose;  // corrected map pose of a keyframe
  std::function<std::optional<Eigen::Matrix<double, 6, 6>>(std::uint64_t, std::uint64_t)>
      chain_cov;  // odometry-chain covariance between two keyframes (translation-first)
  std::function<std::optional<Timestamp>(std::uint64_t)> stamp;            // measurement time
  std::function<std::optional<ObservabilityReport>(std::uint64_t)> obs;   // endpoint observability
};

// Per-detector counters for telemetry; cumulative over the mission.
struct LoopDiagnostics {
  std::uint64_t keyframes = 0;           // offered via add_keyframe
  std::uint64_t candidates = 0;          // Stage A retrieval candidates considered
  std::uint64_t verified = 0;            // Stage C (GICP) passes
  std::uint64_t self_test_rejected = 0;  // Stage D single-loop self-test rejects
  std::uint64_t emitted = 0;             // LoopConstraints handed to L3
  double last_detect_ms = 0;             // wall time of the last detect() (telemetry only)
};

// Place recognition: Scan Context++ retrieval -> (STD/BTC re-rank) -> GICP verify ->
// single-loop consistency. Operates on retained keyframe clouds, never live scans, and
// runs on the back-end path (best-effort) so it never stalls odometry.
class ILoopDetector {
 public:
  virtual ~ILoopDetector() = default;

  // Offer a new keyframe (corrected map pose) for descriptor extraction + candidate search.
  virtual void add_keyframe(std::uint64_t id, PointCloudPtr cloud, const Pose& T_map_body) = 0;

  // Run the cascade for the newest keyframe; returns verified, self-consistent loops (may be empty).
  virtual std::vector<LoopConstraint> detect() = 0;

  // A back-end correction moved keyframes; invalidate any cached geometry built on stale poses.
  virtual void on_graph_update(const GraphUpdate& update) = 0;

  virtual LoopDiagnostics diagnostics() const = 0;
};

// Builds the configured detector. `deterministic` forces single-threaded, run-to-run
// reproducible behaviour (replay/harness); false permits non-determinism-tolerant speedups.
std::unique_ptr<ILoopDetector> makeLoopDetector(const PlaceConfig& cfg,
                                                std::shared_ptr<const KeyframeStore> store,
                                                KeyframePoseSource pose_source, bool deterministic,
                                                TelemetrySink* telemetry = nullptr);

}  // namespace meridian
