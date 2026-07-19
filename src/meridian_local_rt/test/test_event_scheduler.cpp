#include <gtest/gtest.h>

#include <cstdint>
#include <optional>

#include "meridian/local/event_scheduler.hpp"

namespace meridian::local {
namespace {

[[nodiscard]] core::ObservationLineage lineage(std::uint64_t measurement, std::uint64_t consumer) {
  core::ObservationSlice slice;
  slice.root = core::MeasurementId{measurement};
  slice.calibration = core::CalibrationEpoch{1U};
  core::ObservationLineage result;
  result.usage.push_back(core::ObservationUsage{slice, core::ObservationRole::PrimaryResidual,
                                                core::DerivedRecordId{consumer},
                                                core::FactorGroupId{consumer}, std::nullopt});
  return result;
}

[[nodiscard]] core::RecordHeader header(std::uint64_t trace) {
  core::RecordHeader result;
  result.trace = core::TraceId{trace};
  result.producer = core::ProducerId{1U};
  result.session = core::SessionId{1U};
  result.config = core::ConfigRevision{1U};
  result.direct_calibration = core::CalibrationEpoch{1U};
  return result;
}

[[nodiscard]] StateRequest visual(std::uint64_t id, std::int64_t time, std::uint64_t camera = 0U) {
  StateRequest result;
  result.header = header(id);
  result.id = core::KnotRequestId{id};
  result.sensor = core::SensorInstanceId::camera(core::CameraId{camera});
  result.purpose = StateRequestPurpose::VisualKeyframe;
  result.exact_time = core::FusionTime{time};
  result.lineage = lineage(id, id);
  return result;
}

[[nodiscard]] StateRequest lidar(std::uint64_t id, std::int64_t time, std::uint64_t sensor = 0U) {
  StateRequest result;
  result.header = header(id);
  result.id = core::KnotRequestId{id};
  result.sensor = core::SensorInstanceId::lidar(core::LidarId{sensor});
  result.purpose = StateRequestPurpose::LidarReference;
  result.exact_time = core::FusionTime{time};
  result.lineage = lineage(id, id);
  return result;
}

[[nodiscard]] StateRequest guard(std::uint64_t id, std::int64_t time) {
  StateRequest result;
  result.header = header(id);
  result.id = core::KnotRequestId{id};
  result.purpose = StateRequestPurpose::ImuGuard;
  result.exact_time = core::FusionTime{time};
  return result;
}

TEST(StateTimeline, VisualOnlyRequestsCreateACompleteStateSequence) {
  StateTimeline timeline({core::Duration{10}, 8U, 8U});

  for (std::uint64_t id = 1U; id <= 3U; ++id) {
    const auto admitted = timeline.request(visual(id, static_cast<std::int64_t>(id * 100U)));
    ASSERT_TRUE(admitted);
    EXPECT_EQ(admitted.value().disposition, StateAdmissionDisposition::NewState);
    ASSERT_TRUE(admitted.value().resolution);
    EXPECT_EQ(admitted.value().resolution->requests,
              std::vector<core::KnotRequestId>{core::KnotRequestId{id}});
  }

  ASSERT_EQ(timeline.resolutions().size(), 3U);
  EXPECT_EQ(timeline.resolutions()[0U].exact_time, core::FusionTime{100});
  EXPECT_EQ(timeline.resolutions()[1U].exact_time, core::FusionTime{200});
  EXPECT_EQ(timeline.resolutions()[2U].exact_time, core::FusionTime{300});
}

TEST(StateTimeline, LidarOnlyAdmissionHasNoCameraBarrier) {
  StateTimeline timeline({core::Duration{10}, 8U, 8U});

  const auto admitted = timeline.request(lidar(1U, 150));
  ASSERT_TRUE(admitted);
  EXPECT_EQ(admitted.value().disposition, StateAdmissionDisposition::NewState);
  EXPECT_EQ(admitted.value().request.purpose, StateRequestPurpose::LidarReference);
  ASSERT_TRUE(admitted.value().request.sensor);
  EXPECT_EQ(admitted.value().request.sensor->modality, core::SensorModality::Lidar);
  ASSERT_EQ(timeline.resolutions().size(), 1U);
  EXPECT_EQ(timeline.resolutions().front().exact_time, core::FusionTime{150});
}

TEST(StateTimeline, DefaultGuardDoesNotRateLimitJitteredTenHertzLidar) {
  StateTimeline timeline;

  const auto first = timeline.request(lidar(1U, 100'000'000LL));
  const auto jittered = timeline.request(lidar(2U, 199'900'000LL));

  ASSERT_TRUE(first);
  ASSERT_TRUE(jittered);
  EXPECT_EQ(first.value().disposition, StateAdmissionDisposition::NewState);
  EXPECT_EQ(jittered.value().disposition, StateAdmissionDisposition::NewState);
  ASSERT_EQ(timeline.resolutions().size(), 2U);
  EXPECT_EQ(timeline.resolutions()[0U].exact_time, core::FusionTime{100'000'000LL});
  EXPECT_EQ(timeline.resolutions()[1U].exact_time, core::FusionTime{199'900'000LL});
}

TEST(StateTimeline, DefaultGuardSuppressesOnlyStrictlyInsideOneMillisecond) {
  StateTimeline timeline;

  const auto root = timeline.request(lidar(1U, 100'000'000LL));
  const auto exact = timeline.request(visual(2U, 100'000'000LL));
  const auto inside = timeline.request(lidar(3U, 100'999'999LL));
  const auto boundary = timeline.request(visual(4U, 101'000'000LL));

  ASSERT_TRUE(root);
  ASSERT_TRUE(exact);
  ASSERT_TRUE(inside);
  ASSERT_TRUE(boundary);
  EXPECT_EQ(root.value().disposition, StateAdmissionDisposition::NewState);
  EXPECT_EQ(exact.value().disposition, StateAdmissionDisposition::ExactShare);
  EXPECT_EQ(inside.value().disposition, StateAdmissionDisposition::SuppressedTooClose);
  EXPECT_EQ(inside.value().suppressing_state_time, core::FusionTime{100'000'000LL});
  EXPECT_EQ(boundary.value().disposition, StateAdmissionDisposition::NewState);
  ASSERT_EQ(timeline.resolutions().size(), 2U);
  EXPECT_EQ(timeline.resolutions()[1U].exact_time, core::FusionTime{101'000'000LL});
}

TEST(StateTimeline, InitializationRootFinalityReleasesBootstrapCapacityButRetainsRootSharing) {
  StateTimeline timeline({core::Duration{1}, 4U, 4U});
  ASSERT_TRUE(timeline.request(lidar(1U, 100)));
  ASSERT_TRUE(timeline.request(lidar(2U, 200)));
  ASSERT_TRUE(timeline.request(lidar(3U, 300)));
  ASSERT_TRUE(timeline.request(lidar(4U, 400)));
  ASSERT_TRUE(timeline.recordCommittedState(core::StateId{1U}, core::FusionTime{400}));

  const auto finalized = timeline.finalizeThrough(core::FusionTime{399});
  ASSERT_TRUE(finalized);
  EXPECT_EQ(finalized.value().navigation_states_pruned, 3U);
  EXPECT_EQ(finalized.value().retained_requests_pruned, 3U);
  EXPECT_EQ(finalized.value().retained.retained_requests, 1U);
  ASSERT_EQ(timeline.resolutions().size(), 1U);
  EXPECT_EQ(timeline.resolutions().front().exact_time, core::FusionTime{400});

  const auto stale_pre_root = timeline.request(visual(5U, 399));
  ASSERT_FALSE(stale_pre_root);
  EXPECT_EQ(stale_pre_root.error().code, StateTimelineErrorCode::TimelineFinalized);

  const auto exact_root_share = timeline.request(visual(5U, 400));
  ASSERT_TRUE(exact_root_share);
  EXPECT_EQ(exact_root_share.value().disposition, StateAdmissionDisposition::ExactShare);
  const auto next = timeline.request(lidar(6U, 500));
  ASSERT_TRUE(next);
  EXPECT_EQ(next.value().disposition, StateAdmissionDisposition::NewState);
}

TEST(StateTimeline, FixedLagFinalityPreservesOptionalRequestsStrictlyInsideLiveWindow) {
  StateTimeline timeline({core::Duration{1}, 4U, 5U});
  ASSERT_TRUE(timeline.request(lidar(1U, 100)));
  ASSERT_TRUE(timeline.recordCommittedState(core::StateId{10U}, core::FusionTime{100}));
  ASSERT_TRUE(timeline.request(visual(2U, 150)));
  ASSERT_TRUE(timeline.request(lidar(3U, 200)));
  ASSERT_TRUE(timeline.recordCommittedState(core::StateId{11U}, core::FusionTime{200}));
  ASSERT_TRUE(timeline.request(visual(4U, 250)));

  const auto finalized = timeline.finalizeThrough(core::FusionTime{150});
  ASSERT_TRUE(finalized);
  EXPECT_EQ(finalized.value().navigation_states_pruned, 2U);
  EXPECT_EQ(finalized.value().retained_requests_pruned, 2U);
  ASSERT_EQ(timeline.resolutions().size(), 2U);
  EXPECT_EQ(timeline.resolutions()[0U].exact_time, core::FusionTime{200});
  EXPECT_EQ(timeline.resolutions()[1U].exact_time, core::FusionTime{250});

  const auto late_exact_share = timeline.request(lidar(5U, 250));
  ASSERT_TRUE(late_exact_share);
  EXPECT_EQ(late_exact_share.value().disposition, StateAdmissionDisposition::ExactShare);
  ASSERT_TRUE(late_exact_share.value().resolution);
  EXPECT_EQ(late_exact_share.value().resolution->requests,
            (std::vector<core::KnotRequestId>{core::KnotRequestId{4U}, core::KnotRequestId{5U}}));

  const auto stale = timeline.request(visual(6U, 150));
  ASSERT_FALSE(stale);
  EXPECT_EQ(stale.error().code, StateTimelineErrorCode::TimelineFinalized);
  const auto later = timeline.request(visual(6U, 300));
  ASSERT_TRUE(later);
  EXPECT_EQ(later.value().disposition, StateAdmissionDisposition::NewState);

  const auto advanced = timeline.finalizeThrough(core::FusionTime{200});
  ASSERT_TRUE(advanced);
  EXPECT_EQ(advanced.value().navigation_states_pruned, 1U);
  ASSERT_EQ(timeline.resolutions().size(), 2U);
  EXPECT_EQ(timeline.resolutions()[0U].exact_time, core::FusionTime{250});
  EXPECT_EQ(timeline.resolutions()[1U].exact_time, core::FusionTime{300});
}

TEST(StateTimeline, InterleavedPurposesNeverDisplaceOrCancelAcceptedRequests) {
  StateTimeline timeline({core::Duration{50}, 8U, 8U});

  ASSERT_TRUE(timeline.request(visual(1U, 100)));
  ASSERT_TRUE(timeline.request(lidar(2U, 300)));
  ASSERT_TRUE(timeline.request(guard(3U, 500)));
  ASSERT_TRUE(timeline.request(visual(4U, 700, 1U)));

  ASSERT_EQ(timeline.resolutions().size(), 4U);
  EXPECT_EQ(timeline.resolutions()[0U].requests,
            std::vector<core::KnotRequestId>{core::KnotRequestId{1U}});
  EXPECT_EQ(timeline.resolutions()[1U].requests,
            std::vector<core::KnotRequestId>{core::KnotRequestId{2U}});
  EXPECT_EQ(timeline.resolutions()[2U].requests,
            std::vector<core::KnotRequestId>{core::KnotRequestId{3U}});
  EXPECT_EQ(timeline.resolutions()[3U].requests,
            std::vector<core::KnotRequestId>{core::KnotRequestId{4U}});
}

TEST(StateTimeline, SharesOnlyBitExactStateTimesAcrossSensors) {
  StateTimeline timeline({core::Duration{100}, 8U, 8U});
  ASSERT_TRUE(timeline.request(visual(1U, 200)));

  const auto lidar_share = timeline.request(lidar(2U, 200));
  ASSERT_TRUE(lidar_share);
  EXPECT_EQ(lidar_share.value().disposition, StateAdmissionDisposition::ExactShare);
  ASSERT_TRUE(lidar_share.value().resolution);
  EXPECT_EQ(lidar_share.value().resolution->requests,
            (std::vector<core::KnotRequestId>{core::KnotRequestId{1U}, core::KnotRequestId{2U}}));

  const auto guard_share = timeline.request(guard(3U, 200));
  ASSERT_TRUE(guard_share);
  EXPECT_EQ(guard_share.value().disposition, StateAdmissionDisposition::ExactShare);
  ASSERT_EQ(timeline.resolutions().size(), 1U);
  EXPECT_EQ(timeline.resolutions().front().requests,
            (std::vector<core::KnotRequestId>{core::KnotRequestId{1U}, core::KnotRequestId{2U},
                                              core::KnotRequestId{3U}}));
}

TEST(StateTimeline, CommittedStateBindsAnAdmissionAndIsExactlyIdempotent) {
  StateTimeline timeline({core::Duration{100}, 8U, 8U});
  ASSERT_TRUE(timeline.request(visual(1U, 200)));

  const auto bound = timeline.recordCommittedState(core::StateId{41U}, core::FusionTime{200});
  ASSERT_TRUE(bound);
  ASSERT_TRUE(bound.value().committed_state);
  EXPECT_EQ(*bound.value().committed_state, core::StateId{41U});
  EXPECT_EQ(bound.value().requests, std::vector<core::KnotRequestId>{core::KnotRequestId{1U}});

  const auto repeated = timeline.recordCommittedState(core::StateId{41U}, core::FusionTime{200});
  ASSERT_TRUE(repeated);
  ASSERT_EQ(timeline.resolutions().size(), 1U);
  ASSERT_TRUE(timeline.resolutions().front().committed_state);
  EXPECT_EQ(*timeline.resolutions().front().committed_state, core::StateId{41U});

  const auto wrong_state = timeline.recordCommittedState(core::StateId{42U}, core::FusionTime{200});
  ASSERT_FALSE(wrong_state);
  EXPECT_EQ(wrong_state.error().code, StateTimelineErrorCode::CommittedStateConflict);
  const auto wrong_time = timeline.recordCommittedState(core::StateId{41U}, core::FusionTime{201});
  ASSERT_FALSE(wrong_time);
  EXPECT_EQ(wrong_time.error().code, StateTimelineErrorCode::CommittedStateConflict);
  ASSERT_EQ(timeline.resolutions().size(), 1U);
}

TEST(StateTimeline, ExternallyCreatedStateIsRecordedAndCanReceiveALateExactRequest) {
  StateTimeline timeline({core::Duration{100}, 8U, 8U});

  const auto initialization =
      timeline.recordCommittedState(core::StateId{10U}, core::FusionTime{500});
  ASSERT_TRUE(initialization);
  EXPECT_TRUE(initialization.value().requests.empty());
  ASSERT_TRUE(initialization.value().committed_state);
  EXPECT_EQ(*initialization.value().committed_state, core::StateId{10U});
  EXPECT_FALSE(timeline.retainedState().last_terminal_request);
  EXPECT_EQ(timeline.retainedState().retained_requests, 0U);

  const auto late_exact = timeline.request(visual(1U, 500));
  ASSERT_TRUE(late_exact);
  EXPECT_EQ(late_exact.value().disposition, StateAdmissionDisposition::ExactShare);
  ASSERT_TRUE(late_exact.value().resolution);
  ASSERT_TRUE(late_exact.value().resolution->committed_state);
  EXPECT_EQ(*late_exact.value().resolution->committed_state, core::StateId{10U});
  EXPECT_EQ(late_exact.value().resolution->requests,
            std::vector<core::KnotRequestId>{core::KnotRequestId{1U}});

  const auto close = timeline.request(lidar(2U, 550));
  ASSERT_TRUE(close);
  EXPECT_EQ(close.value().disposition, StateAdmissionDisposition::SuppressedTooClose);
  ASSERT_TRUE(close.value().suppressing_state_time);
  EXPECT_EQ(*close.value().suppressing_state_time, core::FusionTime{500});
}

TEST(StateTimeline, CommittedStateValidationIsAtomicAcrossCapacityAndFinality) {
  StateTimeline timeline({core::Duration{0}, 1U, 2U});
  ASSERT_TRUE(timeline.request(visual(1U, 100)));

  const auto invalid = timeline.recordCommittedState(core::StateId{}, core::FusionTime{100});
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().code, StateTimelineErrorCode::InvalidCommittedState);
  EXPECT_FALSE(timeline.resolutions().front().committed_state);

  // Binding an existing admitted time consumes no additional state capacity.
  ASSERT_TRUE(timeline.recordCommittedState(core::StateId{1U}, core::FusionTime{100}));
  const auto full = timeline.recordCommittedState(core::StateId{2U}, core::FusionTime{200});
  ASSERT_FALSE(full);
  EXPECT_EQ(full.error().code, StateTimelineErrorCode::StateCapacity);
  ASSERT_EQ(timeline.resolutions().size(), 1U);

  ASSERT_TRUE(timeline.finalizeThrough(core::FusionTime{100}));
  const auto finalized = timeline.recordCommittedState(core::StateId{1U}, core::FusionTime{100});
  ASSERT_FALSE(finalized);
  EXPECT_EQ(finalized.error().code, StateTimelineErrorCode::TimelineFinalized);
  EXPECT_TRUE(timeline.resolutions().empty());
}

TEST(StateTimeline, CloseRequestSuppressesOnlyItselfAndConsumesItsIdentity) {
  StateTimeline timeline({core::Duration{100}, 8U, 8U});
  ASSERT_TRUE(timeline.request(visual(10U, 100)));

  const auto suppressed = timeline.request(lidar(11U, 150));
  ASSERT_TRUE(suppressed);
  EXPECT_EQ(suppressed.value().disposition, StateAdmissionDisposition::SuppressedTooClose);
  EXPECT_FALSE(suppressed.value().resolution);
  ASSERT_TRUE(suppressed.value().suppressing_state_time);
  EXPECT_EQ(*suppressed.value().suppressing_state_time, core::FusionTime{100});
  ASSERT_EQ(timeline.resolutions().size(), 1U);
  EXPECT_EQ(timeline.resolutions().front().requests,
            std::vector<core::KnotRequestId>{core::KnotRequestId{10U}});
  EXPECT_EQ(timeline.retainedState().retained_requests, 1U);
  ASSERT_TRUE(timeline.retainedState().last_terminal_request);
  EXPECT_EQ(*timeline.retainedState().last_terminal_request, core::KnotRequestId{11U});

  const auto duplicate = timeline.request(lidar(11U, 300));
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, StateTimelineErrorCode::DuplicateRequestId);
  ASSERT_TRUE(timeline.request(guard(12U, 300)));
}

TEST(StateTimeline, SuppressionTieSelectsEarlierStateDeterministically) {
  StateTimeline timeline({core::Duration{150}, 8U, 8U});
  ASSERT_TRUE(timeline.request(visual(1U, 100)));
  ASSERT_TRUE(timeline.request(lidar(2U, 300)));

  const auto suppressed = timeline.request(guard(3U, 200));
  ASSERT_TRUE(suppressed);
  EXPECT_EQ(suppressed.value().disposition, StateAdmissionDisposition::SuppressedTooClose);
  ASSERT_TRUE(suppressed.value().suppressing_state_time);
  EXPECT_EQ(*suppressed.value().suppressing_state_time, core::FusionTime{100});
  EXPECT_EQ(timeline.resolutions().size(), 2U);
}

TEST(StateTimeline, SuppressionDoesNotConsumeRetainedRequestCapacity) {
  StateTimeline timeline({core::Duration{100}, 2U, 1U});
  ASSERT_TRUE(timeline.request(visual(1U, 100)));

  const auto suppressed = timeline.request(lidar(2U, 150));
  ASSERT_TRUE(suppressed);
  EXPECT_EQ(suppressed.value().disposition, StateAdmissionDisposition::SuppressedTooClose);
  EXPECT_EQ(timeline.retainedState().retained_requests, 1U);

  const auto exact_capacity = timeline.request(guard(3U, 100));
  ASSERT_FALSE(exact_capacity);
  EXPECT_EQ(exact_capacity.error().code, StateTimelineErrorCode::RequestCapacity);
  ASSERT_TRUE(timeline.retainedState().last_terminal_request);
  EXPECT_EQ(*timeline.retainedState().last_terminal_request, core::KnotRequestId{2U});
}

TEST(StateTimeline, CapacityFailuresAreAtomicAndRetryableAfterFinalization) {
  StateTimeline timeline({core::Duration{0}, 2U, 2U});
  ASSERT_TRUE(timeline.request(visual(1U, 100)));
  ASSERT_TRUE(timeline.request(lidar(2U, 200)));

  const auto full = timeline.request(guard(3U, 300));
  ASSERT_FALSE(full);
  EXPECT_EQ(full.error().code, StateTimelineErrorCode::RequestCapacity);
  EXPECT_EQ(timeline.resolutions().size(), 2U);
  ASSERT_TRUE(timeline.retainedState().last_terminal_request);
  EXPECT_EQ(*timeline.retainedState().last_terminal_request, core::KnotRequestId{2U});

  const auto finalized = timeline.finalizeThrough(core::FusionTime{100});
  ASSERT_TRUE(finalized);
  EXPECT_EQ(finalized.value().navigation_states_pruned, 1U);
  EXPECT_EQ(finalized.value().retained_requests_pruned, 1U);
  const auto retried = timeline.request(guard(3U, 300));
  ASSERT_TRUE(retried);
  EXPECT_EQ(retried.value().disposition, StateAdmissionDisposition::NewState);
}

TEST(StateTimeline, StateCapacityFailureDoesNotConsumeRequestIdentity) {
  StateTimeline timeline({core::Duration{0}, 1U, 4U});
  ASSERT_TRUE(timeline.request(visual(1U, 100)));

  const auto full = timeline.request(lidar(2U, 200));
  ASSERT_FALSE(full);
  EXPECT_EQ(full.error().code, StateTimelineErrorCode::StateCapacity);
  ASSERT_TRUE(timeline.retainedState().last_terminal_request);
  EXPECT_EQ(*timeline.retainedState().last_terminal_request, core::KnotRequestId{1U});

  ASSERT_TRUE(timeline.finalizeThrough(core::FusionTime{100}));
  ASSERT_TRUE(timeline.request(lidar(2U, 200)));
}

TEST(StateTimeline, FinalityErrorsAreAtomicAndCorrectedRequestCanRetry) {
  StateTimeline timeline({core::Duration{0}, 4U, 4U});
  ASSERT_TRUE(timeline.request(visual(1U, 100)));
  ASSERT_TRUE(timeline.finalizeThrough(core::FusionTime{100}));

  const auto finalized = timeline.request(lidar(2U, 100));
  ASSERT_FALSE(finalized);
  EXPECT_EQ(finalized.error().code, StateTimelineErrorCode::TimelineFinalized);
  ASSERT_TRUE(timeline.retainedState().last_terminal_request);
  EXPECT_EQ(*timeline.retainedState().last_terminal_request, core::KnotRequestId{1U});
  ASSERT_TRUE(timeline.request(lidar(2U, 200)));

  const auto backward = timeline.finalizeThrough(core::FusionTime{99});
  ASSERT_FALSE(backward);
  EXPECT_EQ(backward.error().code, StateTimelineErrorCode::FinalityRegression);
  ASSERT_TRUE(timeline.retainedState().finalized_through);
  EXPECT_EQ(*timeline.retainedState().finalized_through, core::FusionTime{100});
}

TEST(StateTimeline, InvalidAndOutOfOrderRequestsNeverMutateTimeline) {
  StateTimeline timeline({core::Duration{0}, 4U, 4U});
  StateRequest invalid = visual(10U, 100);
  invalid.purpose = static_cast<StateRequestPurpose>(255);
  const auto rejected = timeline.request(invalid);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, StateTimelineErrorCode::InvalidRequest);
  EXPECT_TRUE(timeline.resolutions().empty());
  EXPECT_FALSE(timeline.retainedState().last_terminal_request);

  ASSERT_TRUE(timeline.request(visual(10U, 100)));
  const auto duplicate = timeline.request(lidar(10U, 200));
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, StateTimelineErrorCode::DuplicateRequestId);
  const auto older = timeline.request(guard(9U, 300));
  ASSERT_FALSE(older);
  EXPECT_EQ(older.error().code, StateTimelineErrorCode::RequestIdOutOfOrder);
  EXPECT_EQ(timeline.resolutions().size(), 1U);
}

TEST(StateTimeline, RejectsInvalidHeaderAndPurposeSensorPairingAtomically) {
  StateTimeline timeline({core::Duration{0}, 4U, 4U});

  StateRequest invalid_header = visual(1U, 100);
  invalid_header.header.direct_calibration.reset();
  const auto header_rejected = timeline.request(invalid_header);
  ASSERT_FALSE(header_rejected);
  EXPECT_EQ(header_rejected.error().code, StateTimelineErrorCode::InvalidRequest);

  StateRequest missing_sensor = visual(1U, 100);
  missing_sensor.sensor.reset();
  const auto missing_rejected = timeline.request(missing_sensor);
  ASSERT_FALSE(missing_rejected);
  EXPECT_EQ(missing_rejected.error().code, StateTimelineErrorCode::InvalidRequest);

  StateRequest wrong_sensor = visual(1U, 100);
  wrong_sensor.sensor = core::SensorInstanceId::lidar(core::LidarId{0U});
  const auto wrong_rejected = timeline.request(wrong_sensor);
  ASSERT_FALSE(wrong_rejected);
  EXPECT_EQ(wrong_rejected.error().code, StateTimelineErrorCode::InvalidRequest);

  StateRequest guarded_by_sensor = guard(1U, 100);
  guarded_by_sensor.sensor = core::SensorInstanceId::camera(core::CameraId{0U});
  const auto guard_rejected = timeline.request(guarded_by_sensor);
  ASSERT_FALSE(guard_rejected);
  EXPECT_EQ(guard_rejected.error().code, StateTimelineErrorCode::InvalidRequest);

  EXPECT_TRUE(timeline.resolutions().empty());
  EXPECT_FALSE(timeline.retainedState().last_terminal_request);
  ASSERT_TRUE(timeline.request(visual(1U, 100)));
}

TEST(StateTimeline, RejectsInvalidConfigurationWithoutMutation) {
  StateTimeline invalid_interval({core::Duration{-1}, 4U, 4U});
  const auto interval = invalid_interval.request(visual(1U, 100));
  ASSERT_FALSE(interval);
  EXPECT_EQ(interval.error().code, StateTimelineErrorCode::InvalidConfig);
  EXPECT_TRUE(invalid_interval.resolutions().empty());

  StateTimeline zero_capacity({core::Duration{0}, 0U, 4U});
  const auto capacity = zero_capacity.request(lidar(1U, 100));
  ASSERT_FALSE(capacity);
  EXPECT_EQ(capacity.error().code, StateTimelineErrorCode::InvalidConfig);
}

}  // namespace
}  // namespace meridian::local
