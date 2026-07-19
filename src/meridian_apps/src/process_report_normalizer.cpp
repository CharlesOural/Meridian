#include "process_report_normalizer.hpp"

#include <algorithm>
#include <utility>

namespace meridian::apps::detail {
namespace {

enum class ReferenceKind {
  Initialization,
  Lidar,
  Camera,
};

struct CommitReference {
  const local::LocalGraphCommit* commit{};
  const local::LidarCommitReport* lidar{};
  const local::CameraKnotCommitReport* camera{};
  ReferenceKind kind{ReferenceKind::Initialization};
};

[[nodiscard]] ProcessReportNormalizationError normalizationError(
    ProcessReportNormalizationErrorCode code, std::string detail) {
  return ProcessReportNormalizationError{code, std::move(detail)};
}

[[nodiscard]] bool validCommitIdentity(const local::LocalGraphCommit& commit,
                                       bool parent_required) noexcept {
  return commit.odom_epoch.valid() && commit.revision.valid() &&
         (!parent_required || commit.parent.valid()) && commit.state.valid();
}

[[nodiscard]] bool sameCommitIdentity(const local::LocalGraphCommit& lhs,
                                      const local::LocalGraphCommit& rhs) noexcept {
  return lhs.odom_epoch == rhs.odom_epoch && lhs.revision == rhs.revision &&
         lhs.parent == rhs.parent && lhs.state == rhs.state && lhs.state_time == rhs.state_time;
}

[[nodiscard]] bool sameNavigationIdentity(const local::LocalGraphCommit& lhs,
                                          const local::LocalGraphCommit& rhs) noexcept {
  return lhs.odom_epoch == rhs.odom_epoch && lhs.state == rhs.state &&
         lhs.state_time == rhs.state_time;
}

[[nodiscard]] bool validCameraMetadata(const local::CameraKnotCommitReport& camera) noexcept {
  if (camera.exact_time != camera.commit.state_time || camera.resolved_keyframes.empty()) {
    return false;
  }
  return std::all_of(
      camera.resolved_keyframes.begin(), camera.resolved_keyframes.end(),
      [&](const local::VisualResolvedKeyframeReport& resolved) {
        const local::VisualStateResolution& resolution = resolved.resolution;
        return resolution.request.valid() && resolution.odom_epoch == camera.commit.odom_epoch &&
               resolution.state == camera.commit.state &&
               resolution.timeline.exact_time == camera.commit.state_time &&
               resolution.timeline.committed_state == camera.commit.state &&
               std::find(resolution.timeline.requests.begin(), resolution.timeline.requests.end(),
                         resolution.request) != resolution.timeline.requests.end() &&
               resolution.created_at_revision == camera.commit.revision;
      });
}

}  // namespace

core::Result<NormalizedProcessReport, ProcessReportNormalizationError> normalizeProcessReport(
    const local::LocalEstimatorProcessReport& report) {
  using Result = core::Result<NormalizedProcessReport, ProcessReportNormalizationError>;
  std::vector<CommitReference> references;
  references.reserve((report.initialization ? 1U : 0U) + report.commits.size() +
                     report.camera_commits.size());
  if (report.initialization) {
    references.push_back(
        CommitReference{&*report.initialization, nullptr, nullptr, ReferenceKind::Initialization});
  }
  for (const local::LidarCommitReport& lidar : report.commits) {
    references.push_back(CommitReference{&lidar.commit, &lidar, nullptr, ReferenceKind::Lidar});
  }
  for (const local::CameraKnotCommitReport& camera : report.camera_commits) {
    references.push_back(CommitReference{&camera.commit, nullptr, &camera, ReferenceKind::Camera});
  }

  for (const CommitReference& reference : references) {
    if (reference.commit == nullptr ||
        !validCommitIdentity(*reference.commit, reference.kind != ReferenceKind::Initialization)) {
      return Result::failure(
          normalizationError(ProcessReportNormalizationErrorCode::InvalidCommitMetadata,
                             "process report contains an invalid navigation commit identity"));
    }
    if (reference.camera != nullptr && !validCameraMetadata(*reference.camera)) {
      return Result::failure(normalizationError(
          ProcessReportNormalizationErrorCode::InvalidCommitMetadata,
          "camera commit does not carry exact state-time/revision keyframe resolution metadata"));
    }
    if (reference.lidar != nullptr && !reference.lidar->measurement.valid()) {
      return Result::failure(
          normalizationError(ProcessReportNormalizationErrorCode::InvalidCommitMetadata,
                             "LiDAR commit does not carry a valid measurement identity"));
    }
  }

  std::sort(references.begin(), references.end(),
            [](const CommitReference& lhs, const CommitReference& rhs) {
              if (lhs.commit->state != rhs.commit->state) {
                return lhs.commit->state < rhs.commit->state;
              }
              if (lhs.commit->revision != rhs.commit->revision) {
                return lhs.commit->revision < rhs.commit->revision;
              }
              return lhs.kind < rhs.kind;
            });

  NormalizedProcessReport normalized;
  normalized.input_commit_references = references.size();
  for (const CommitReference& reference : references) {
    if (normalized.commits.empty() ||
        normalized.commits.back().commit->state != reference.commit->state) {
      NormalizedProcessCommit commit;
      commit.commit = reference.commit;
      commit.lidar = reference.lidar;
      commit.initialization = reference.kind == ReferenceKind::Initialization;
      commit.emits_navigation_state =
          reference.lidar == nullptr || reference.lidar->navigation_state_created;
      if (reference.camera != nullptr) {
        commit.cameras.push_back(reference.camera);
      }
      normalized.commits.push_back(std::move(commit));
      continue;
    }

    NormalizedProcessCommit& existing = normalized.commits.back();
    if (!sameNavigationIdentity(*existing.commit, *reference.commit) || existing.initialization ||
        reference.kind == ReferenceKind::Initialization ||
        (existing.lidar != nullptr && reference.lidar != nullptr)) {
      return Result::failure(
          normalizationError(ProcessReportNormalizationErrorCode::ConflictingCommitReferences,
                             "same state ID has conflicting commit metadata or duplicate "
                             "initialization/LiDAR references"));
    }
    if (existing.commit->revision == reference.commit->revision) {
      if (!sameCommitIdentity(*existing.commit, *reference.commit)) {
        return Result::failure(
            normalizationError(ProcessReportNormalizationErrorCode::ConflictingCommitReferences,
                               "same navigation state/revision has conflicting parent metadata"));
      }
    } else {
      // One processReady call can first create a camera-shared navigation
      // state and then advance that same state through a LiDAR-only graph
      // transaction. No other modality may silently revise an existing state.
      if (reference.lidar == nullptr || !reference.lidar->graph_revision_created ||
          reference.commit->revision < existing.commit->revision ||
          reference.commit->parent < existing.commit->revision) {
        return Result::failure(
            normalizationError(ProcessReportNormalizationErrorCode::ConflictingCommitReferences,
                               "same navigation state has an invalid or non-LiDAR revision chain"));
      }
      existing.commit = reference.commit;
    }
    if (reference.lidar != nullptr) {
      existing.lidar = reference.lidar;
      existing.commit = &reference.lidar->commit;
    }
    if (reference.camera != nullptr) {
      existing.cameras.push_back(reference.camera);
    }
    existing.emits_navigation_state = existing.emits_navigation_state ||
                                      reference.lidar == nullptr ||
                                      reference.lidar->navigation_state_created;
  }

  for (std::size_t index = 0U; index < normalized.commits.size(); ++index) {
    NormalizedProcessCommit& commit = normalized.commits[index];
    if (commit.lidar != nullptr) {
      // Preserve the report carrying disposition, registration, and deliberate
      // degradation metadata even when a camera reference sorted first.
      commit.commit = &commit.lidar->commit;
    }
    if (index > 0U) {
      const local::LocalGraphCommit& previous = *normalized.commits[index - 1U].commit;
      if (commit.commit->state <= previous.state ||
          commit.commit->state_time <= previous.state_time ||
          commit.commit->revision <= previous.revision) {
        return Result::failure(
            normalizationError(ProcessReportNormalizationErrorCode::NonMonotonicCommitOrder,
                               "state-ID order disagrees with commit time or graph revision"));
      }
    }
    if (commit.lidar != nullptr && !commit.cameras.empty()) {
      ++normalized.shared_camera_lidar_states;
    } else if (commit.lidar == nullptr && !commit.initialization && !commit.cameras.empty()) {
      ++normalized.camera_only_states;
    }
  }
  normalized.duplicate_references_removed =
      normalized.input_commit_references - normalized.commits.size();
  return Result::success(std::move(normalized));
}

}  // namespace meridian::apps::detail
