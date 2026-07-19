#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/core/factor_batch_api.hpp"

namespace meridian::local {

// This is provenance metadata only. Every purpose has identical admission,
// spacing, capacity, sharing, and finality semantics.
enum class StateRequestPurpose {
  VisualKeyframe,
  LidarReference,
  ImuGuard,
};

struct StateRequest {
  core::RecordHeader header;
  core::KnotRequestId id;
  std::optional<core::SensorInstanceId> sensor;
  StateRequestPurpose purpose{StateRequestPurpose::ImuGuard};
  core::FusionTime exact_time;
  core::ObservationLineage lineage;
};

// A timeline resolution is an admitted discrete state time. StateId allocation
// and X/V/B creation remain the graph writer's responsibility. Once that
// writer publishes the state, `committed_state` binds this exact timeline time
// to the graph-owned identity. An empty request list is valid only for a state
// created externally to frontend admission, such as initialization.
struct StateResolution {
  core::FusionTime exact_time;
  std::vector<core::KnotRequestId> requests;
  std::optional<core::StateId> committed_state{};
};

enum class StateAdmissionDisposition {
  NewState,
  ExactShare,
  SuppressedTooClose,
};

struct StateAdmission {
  StateRequest request;
  StateAdmissionDisposition disposition{StateAdmissionDisposition::NewState};
  // Present only for NewState and ExactShare. Suppression never attaches the
  // request to the nearby state.
  std::optional<StateResolution> resolution;
  // Present only for SuppressedTooClose, for deterministic observability.
  std::optional<core::FusionTime> suppressing_state_time;
};

enum class StateTimelineErrorCode {
  InvalidConfig,
  InvalidRequest,
  TimelineFinalized,
  FinalityRegression,
  StateCapacity,
  RequestCapacity,
  DuplicateRequestId,
  RequestIdOutOfOrder,
  InvalidCommittedState,
  CommittedStateConflict,
};

struct StateTimelineError {
  StateTimelineErrorCode code{};
  std::string detail;
};

struct StateTimelineConfig {
  // This is a numerical/coalescing guard, not a frontend rate limiter.  Keep
  // it well below every supported sensor period: exact-time requests still
  // share one state, while independently timed LiDAR scans and visual
  // keyframes retain their own X/V/B state.  In particular, a nominal 10 Hz
  // LiDAR commonly jitters just below 100 ms and must not lose every other
  // scan to this guard.
  core::Duration minimum_state_interval{1'000'000LL};
  // LocalEstimator reserves one pre-commit admission beyond the graph's
  // 64-state live-window default. Standalone timelines may still choose any
  // positive bound; the cross-component headroom invariant belongs to the
  // estimator configuration boundary.
  std::size_t maximum_navigation_states{65U};
  std::size_t maximum_retained_requests{256U};
};

struct StateTimelineRetainedState {
  std::size_t navigation_states{};
  std::size_t retained_requests{};
  std::optional<core::FusionTime> finalized_through;
  // Successful state creation, sharing, and suppression are terminal. Failed
  // requests do not advance this identity watermark and may be retried.
  std::optional<core::KnotRequestId> last_terminal_request;
};

struct StateFinalizationReport {
  core::FusionTime finalized_through;
  std::optional<core::FusionTime> previous_finalized_through;
  std::size_t navigation_states_pruned{};
  std::size_t retained_requests_pruned{};
  StateTimelineRetainedState retained;
};

// Deterministic, sensor-neutral state-time coordinator. Frontends independently
// request exact state times; this class never waits for another sensor and does
// not allocate StateId or mutate the factor graph.
class StateTimeline {
public:
  explicit StateTimeline(StateTimelineConfig config = {});

  [[nodiscard]] core::Result<StateAdmission, StateTimelineError> request(StateRequest request);

  // Records the graph writer's bit-exact StateId/time publication. Existing
  // admitted resolutions are bound in place. A state without a frontend
  // request (notably initialization) creates an empty resolution. Repeating
  // the identical StateId/time pair is idempotent; reusing either half of the
  // binding with a different counterpart is a conflict.
  [[nodiscard]] core::Result<StateResolution, StateTimelineError> recordCommittedState(
      core::StateId state, core::FusionTime time);

  // Finality is monotonic. Every retained resolution at or before `time` is
  // pruned because later requests may no longer change that portion of time.
  [[nodiscard]] core::Result<StateFinalizationReport, StateTimelineError> finalizeThrough(
      core::FusionTime time);

  [[nodiscard]] const std::vector<StateResolution>& resolutions() const noexcept {
    return resolutions_;
  }
  [[nodiscard]] StateTimelineRetainedState retainedState() const noexcept;

private:
  [[nodiscard]] bool configValid() const noexcept;

  StateTimelineConfig config_;
  std::optional<core::FusionTime> finalized_through_;
  std::optional<core::KnotRequestId> last_terminal_request_;
  std::size_t retained_request_count_{};
  std::vector<StateResolution> resolutions_;
};

}  // namespace meridian::local
