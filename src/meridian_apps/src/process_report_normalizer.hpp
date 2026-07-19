#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/local_estimator.hpp"

namespace meridian::apps::detail {

enum class ProcessReportNormalizationErrorCode {
  InvalidCommitMetadata,
  ConflictingCommitReferences,
  NonMonotonicCommitOrder,
};

struct ProcessReportNormalizationError {
  ProcessReportNormalizationErrorCode code{};
  std::string detail;
};

// One graph-owned state may be referenced by both modality-specific report
// vectors. Pointers remain valid only while the source process report lives.
// LiDAR metadata is authoritative when `lidar` is present; camera reports are
// retained for visual observability and are never emitted as extra states.
struct NormalizedProcessCommit {
  const local::LocalGraphCommit* commit{};
  const local::LidarCommitReport* lidar{};
  std::vector<const local::CameraKnotCommitReport*> cameras;
  bool initialization{};
  // False only for a LiDAR report that attaches to an already-emitted
  // timeline state. Its graph/frontend diagnostics remain publishable, but
  // it must not append another TUM trajectory sample.
  bool emits_navigation_state{};
};

struct NormalizedProcessReport {
  std::vector<NormalizedProcessCommit> commits;
  std::size_t input_commit_references{};
  std::size_t duplicate_references_removed{};
  std::size_t camera_only_states{};
  std::size_t shared_camera_lidar_states{};
};

// Produces a state-ID ordered, per-process-report view. It retains no history;
// the trajectory writer's last-state watermark handles cross-report order.
[[nodiscard]] core::Result<NormalizedProcessReport, ProcessReportNormalizationError>
normalizeProcessReport(const local::LocalEstimatorProcessReport& report);

}  // namespace meridian::apps::detail
