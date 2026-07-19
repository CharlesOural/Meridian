#include "meridian/local/event_scheduler.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace meridian::local {
namespace {

[[nodiscard]] StateTimelineError timelineError(StateTimelineErrorCode code, std::string detail) {
  return StateTimelineError{code, std::move(detail)};
}

[[nodiscard]] std::uint64_t timeDistance(core::FusionTime lhs, core::FusionTime rhs) noexcept {
  const std::uint64_t left = static_cast<std::uint64_t>(lhs.nanoseconds);
  const std::uint64_t right = static_cast<std::uint64_t>(rhs.nanoseconds);
  return lhs >= rhs ? left - right : right - left;
}

[[nodiscard]] bool requestPurposeValid(StateRequestPurpose purpose) noexcept {
  switch (purpose) {
    case StateRequestPurpose::VisualKeyframe:
    case StateRequestPurpose::LidarReference:
    case StateRequestPurpose::ImuGuard:
      return true;
  }
  return false;
}

[[nodiscard]] bool requestHeaderValid(const core::RecordHeader& header) noexcept {
  return header.schema_version != 0U && header.trace.valid() && header.producer.valid() &&
         header.session.valid() && header.config.valid() && header.direct_calibration.has_value() &&
         header.direct_calibration->valid();
}

[[nodiscard]] bool sensorMatchesPurpose(const StateRequest& request) noexcept {
  switch (request.purpose) {
    case StateRequestPurpose::VisualKeyframe:
      return request.sensor.has_value() && request.sensor->valid() &&
             request.sensor->modality == core::SensorModality::Visual;
    case StateRequestPurpose::LidarReference:
      return request.sensor.has_value() && request.sensor->valid() &&
             request.sensor->modality == core::SensorModality::Lidar;
    case StateRequestPurpose::ImuGuard:
      return !request.sensor.has_value();
  }
  return false;
}

}  // namespace

StateTimeline::StateTimeline(StateTimelineConfig config) : config_(config) {}

bool StateTimeline::configValid() const noexcept {
  return config_.minimum_state_interval.nanoseconds >= 0 &&
         config_.maximum_navigation_states > 0U && config_.maximum_retained_requests > 0U;
}

core::Result<StateAdmission, StateTimelineError> StateTimeline::request(StateRequest request) {
  using Result = core::Result<StateAdmission, StateTimelineError>;

  if (!configValid()) {
    return Result::failure(
        timelineError(StateTimelineErrorCode::InvalidConfig,
                      "minimum state interval and retained state/request caps must be valid"));
  }
  if (!request.id.valid() || !requestHeaderValid(request.header) ||
      !requestPurposeValid(request.purpose) || !sensorMatchesPurpose(request) ||
      core::validateLineage(request.lineage) != core::LineageValidationError::None) {
    return Result::failure(timelineError(
        StateTimelineErrorCode::InvalidRequest,
        "state request header, identity, purpose/sensor pairing, or lineage is invalid"));
  }
  if (last_terminal_request_ && request.id == *last_terminal_request_) {
    return Result::failure(timelineError(StateTimelineErrorCode::DuplicateRequestId,
                                         "state request ID is already terminal"));
  }
  if (last_terminal_request_ && request.id < *last_terminal_request_) {
    return Result::failure(timelineError(StateTimelineErrorCode::RequestIdOutOfOrder,
                                         "terminal state request IDs must increase strictly"));
  }
  if (finalized_through_ && request.exact_time <= *finalized_through_) {
    return Result::failure(
        timelineError(StateTimelineErrorCode::TimelineFinalized,
                      "request would alter the finalized portion of the state timeline"));
  }

  auto exact = std::lower_bound(resolutions_.begin(), resolutions_.end(), request.exact_time,
                                [](const StateResolution& resolution, core::FusionTime time) {
                                  return resolution.exact_time < time;
                                });
  if (exact != resolutions_.end() && exact->exact_time == request.exact_time) {
    if (retained_request_count_ >= config_.maximum_retained_requests) {
      return Result::failure(timelineError(StateTimelineErrorCode::RequestCapacity,
                                           "retained request capacity is full"));
    }
    exact->requests.push_back(request.id);
    ++retained_request_count_;
    last_terminal_request_ = request.id;
    return Result::success(StateAdmission{std::move(request), StateAdmissionDisposition::ExactShare,
                                          *exact, std::nullopt});
  }

  const std::uint64_t minimum_distance =
      static_cast<std::uint64_t>(config_.minimum_state_interval.nanoseconds);
  const StateResolution* closest = nullptr;
  std::uint64_t closest_distance = std::numeric_limits<std::uint64_t>::max();
  for (const StateResolution& resolution : resolutions_) {
    const std::uint64_t distance = timeDistance(resolution.exact_time, request.exact_time);
    if (distance >= minimum_distance) {
      continue;
    }
    if (closest == nullptr || distance < closest_distance ||
        (distance == closest_distance && resolution.exact_time < closest->exact_time)) {
      closest = &resolution;
      closest_distance = distance;
    }
  }
  if (closest != nullptr) {
    const core::FusionTime suppressing_time = closest->exact_time;
    last_terminal_request_ = request.id;
    return Result::success(StateAdmission{std::move(request),
                                          StateAdmissionDisposition::SuppressedTooClose,
                                          std::nullopt, suppressing_time});
  }

  if (retained_request_count_ >= config_.maximum_retained_requests) {
    return Result::failure(timelineError(StateTimelineErrorCode::RequestCapacity,
                                         "retained request capacity is full"));
  }
  if (resolutions_.size() >= config_.maximum_navigation_states) {
    return Result::failure(timelineError(StateTimelineErrorCode::StateCapacity,
                                         "retained navigation-state capacity is full"));
  }

  StateResolution resolution{request.exact_time, {request.id}, std::nullopt};
  resolutions_.insert(exact, resolution);
  ++retained_request_count_;
  last_terminal_request_ = request.id;
  return Result::success(StateAdmission{std::move(request), StateAdmissionDisposition::NewState,
                                        std::move(resolution), std::nullopt});
}

core::Result<StateResolution, StateTimelineError> StateTimeline::recordCommittedState(
    core::StateId state, core::FusionTime time) {
  using Result = core::Result<StateResolution, StateTimelineError>;

  if (!configValid()) {
    return Result::failure(
        timelineError(StateTimelineErrorCode::InvalidConfig,
                      "minimum state interval and retained state/request caps must be valid"));
  }
  if (!state.valid()) {
    return Result::failure(timelineError(StateTimelineErrorCode::InvalidCommittedState,
                                         "committed graph state identity is invalid"));
  }
  if (finalized_through_ && time <= *finalized_through_) {
    return Result::failure(
        timelineError(StateTimelineErrorCode::TimelineFinalized,
                      "committed graph state would alter the finalized state timeline"));
  }

  auto exact = std::lower_bound(resolutions_.begin(), resolutions_.end(), time,
                                [](const StateResolution& resolution, core::FusionTime candidate) {
                                  return resolution.exact_time < candidate;
                                });
  const auto same_state = std::find_if(
      resolutions_.begin(), resolutions_.end(), [state](const StateResolution& resolution) {
        return resolution.committed_state && *resolution.committed_state == state;
      });

  if (exact != resolutions_.end() && exact->exact_time == time) {
    if (exact->committed_state) {
      if (*exact->committed_state != state) {
        return Result::failure(timelineError(
            StateTimelineErrorCode::CommittedStateConflict,
            "exact timeline time is already bound to a different committed graph state"));
      }
      return Result::success(*exact);
    }
    if (same_state != resolutions_.end() && same_state != exact) {
      return Result::failure(timelineError(
          StateTimelineErrorCode::CommittedStateConflict,
          "committed graph state identity is already bound to a different exact time"));
    }
    exact->committed_state = state;
    return Result::success(*exact);
  }

  if (same_state != resolutions_.end()) {
    return Result::failure(
        timelineError(StateTimelineErrorCode::CommittedStateConflict,
                      "committed graph state identity is already bound to a different exact time"));
  }
  if (resolutions_.size() >= config_.maximum_navigation_states) {
    return Result::failure(timelineError(StateTimelineErrorCode::StateCapacity,
                                         "retained navigation-state capacity is full"));
  }

  StateResolution resolution{time, {}, state};
  resolutions_.insert(exact, resolution);
  return Result::success(std::move(resolution));
}

core::Result<StateFinalizationReport, StateTimelineError> StateTimeline::finalizeThrough(
    core::FusionTime time) {
  using Result = core::Result<StateFinalizationReport, StateTimelineError>;

  if (!configValid()) {
    return Result::failure(
        timelineError(StateTimelineErrorCode::InvalidConfig,
                      "minimum state interval and retained state/request caps must be valid"));
  }
  if (finalized_through_ && time < *finalized_through_) {
    return Result::failure(timelineError(StateTimelineErrorCode::FinalityRegression,
                                         "state timeline finality cannot move backward"));
  }

  StateFinalizationReport report;
  report.finalized_through = time;
  report.previous_finalized_through = finalized_through_;
  const std::size_t states_before = resolutions_.size();
  const std::size_t requests_before = retained_request_count_;
  resolutions_.erase(std::remove_if(resolutions_.begin(), resolutions_.end(),
                                    [time](const StateResolution& resolution) {
                                      return resolution.exact_time <= time;
                                    }),
                     resolutions_.end());
  report.navigation_states_pruned = states_before - resolutions_.size();

  retained_request_count_ = 0U;
  for (const StateResolution& resolution : resolutions_) {
    retained_request_count_ += resolution.requests.size();
  }
  report.retained_requests_pruned = requests_before - retained_request_count_;
  finalized_through_ = time;
  report.retained = retainedState();
  return Result::success(std::move(report));
}

StateTimelineRetainedState StateTimeline::retainedState() const noexcept {
  return StateTimelineRetainedState{resolutions_.size(), retained_request_count_,
                                    finalized_through_, last_terminal_request_};
}

}  // namespace meridian::local
