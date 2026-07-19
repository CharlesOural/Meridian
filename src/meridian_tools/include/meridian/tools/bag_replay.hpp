#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "meridian/core/api.hpp"

namespace meridian::tools {

// The wire representation is part of the replay profile.  It prevents a
// configured Image topic from being accidentally deserialized as a
// CompressedImage merely because both ultimately produce CameraFrame.
enum class CameraWireFormat {
  Image,
  CompressedImage,
};

struct ReplaySourceIdentity {
  std::string name;
  core::ProducerId producer;
  core::AffineClockModel clock;
};

struct ImuReplaySource {
  std::string topic;
  ReplaySourceIdentity source;
  bool required{true};
};

struct LidarReplaySource {
  std::string topic;
  ReplaySourceIdentity source;
  core::LidarId lidar;
  core::Duration header_to_point_time_origin{};
  bool required{true};
};

struct CameraReplaySource {
  std::string topic;
  ReplaySourceIdentity source;
  core::CameraId camera;
  CameraWireFormat wire_format{CameraWireFormat::CompressedImage};
  core::Duration stamp_to_exposure_midpoint{};
  core::Duration exposure{};
  bool required{true};
};

struct GnssReplaySource {
  std::string topic;
  ReplaySourceIdentity source;
  bool required{true};
};

using ReplaySource =
    std::variant<ImuReplaySource, LidarReplaySource, CameraReplaySource, GnssReplaySource>;

struct ReplayBounds {
  // Both limits are explicit and non-zero. The reader installs the profile's
  // topic whitelist in rosbag storage before reading payloads, so
  // max_bag_messages bounds selected sensor records rather than unrelated
  // large messages from disabled modalities.
  std::size_t max_events{};
  std::size_t max_bag_messages{};
};

enum class ReplayProfileErrorCode {
  InvalidIdentity,
  InvalidClock,
  InvalidId,
  InvalidBounds,
  DuplicateTopic,
  EmptySources,
};

struct ReplayProfileError {
  ReplayProfileErrorCode code{};
  std::string detail;
};

class ReplayProfile {
public:
  [[nodiscard]] static core::Result<ReplayProfile, ReplayProfileError> create(
      std::string name, core::SessionId session, core::ConfigRevision config,
      core::CalibrationEpoch calibration, ReplayBounds bounds, std::vector<ReplaySource> sources);

  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] core::SessionId session() const noexcept { return session_; }
  [[nodiscard]] core::ConfigRevision config() const noexcept { return config_; }
  [[nodiscard]] core::CalibrationEpoch calibration() const noexcept { return calibration_; }
  [[nodiscard]] ReplayBounds bounds() const noexcept { return bounds_; }
  [[nodiscard]] const std::vector<ReplaySource>& sources() const noexcept { return sources_; }

private:
  ReplayProfile(std::string name, core::SessionId session, core::ConfigRevision config,
                core::CalibrationEpoch calibration, ReplayBounds bounds,
                std::vector<ReplaySource> sources);

  std::string name_;
  core::SessionId session_;
  core::ConfigRevision config_;
  core::CalibrationEpoch calibration_;
  ReplayBounds bounds_;
  std::vector<ReplaySource> sources_;
};

struct NewerCollegeReplayOptions {
  ReplayBounds bounds{100'000U, 1'000'000U};
  core::SessionId session{0U};
  core::ConfigRevision config{0U};
  core::ClockRevision clock_revision{0U};
  core::SourceEpoch first_source_epoch{0U};
  core::ProducerId first_producer{0U};
  core::Duration image_exposure{};
  // IMU is always present. These switches remove excluded modalities from
  // the replay profile itself, before rosbag deserialization/conversion. They
  // are therefore suitable for compute-isolated frontend benchmarks; a
  // graph-side feature flag is not.
  bool include_lidar{true};
  bool include_cameras{true};
};

// Builds a ROS 2 Newer College sensor selection. AlphaSense IMU is mandatory;
// Ouster PointCloud2 and all calibrated compressed cameras are independently
// selectable at the adapter boundary. GNSS is intentionally absent because
// these bags do not contain it; generic GNSS topics can be added through
// ReplayProfile::create.
[[nodiscard]] core::Result<ReplayProfile, ReplayProfileError> makeNewerCollegeRos2ReplayProfile(
    const core::CalibrationBundle& calibration, const NewerCollegeReplayOptions& options = {});

struct RecordedBagTimestamp {
  std::int64_t nanoseconds{};
  auto operator<=>(const RecordedBagTimestamp&) const = default;
};

enum class ReplayConversionElementKind {
  Record,
  LidarPoint,
};

// Immutable accounting for one successful wire-to-domain conversion. LiDAR
// preserves the point counts returned by ros::convertLidar; one-to-one sensor
// records use 1 input, 1 output, and 0 discarded elements.
struct ReplayConversionAccounting {
  ReplayConversionElementKind kind{ReplayConversionElementKind::Record};
  std::size_t input_elements{};
  std::size_t output_elements{};
  std::size_t discarded_non_finite{};

  [[nodiscard]] bool valid() const noexcept {
    return output_elements <= input_elements &&
           discarded_non_finite == input_elements - output_elements;
  }
};

struct ReplayEventMetadata {
  std::string topic;
  std::string source_name;
  RecordedBagTimestamp recorded_timestamp;
  core::ArrivalTime recorded_arrival;
  core::IngressSequence ingress_sequence;
  core::MeasurementId measurement_sequence;
  std::optional<ReplayConversionAccounting> conversion_accounting;
};

using DomainObservation =
    std::variant<core::ImuSample, core::LidarSweep, core::CameraFrame, core::GnssObservation>;

// DomainEvent is move-only and exposes const access only. This keeps a bag
// adapter from sharing a mutable ROS message or mutable converted observation
// with estimator code.
class DomainEvent {
public:
  DomainEvent(ReplayEventMetadata metadata, DomainObservation observation);
  DomainEvent(const DomainEvent&) = delete;
  DomainEvent& operator=(const DomainEvent&) = delete;
  DomainEvent(DomainEvent&&) noexcept = default;
  DomainEvent& operator=(DomainEvent&&) = delete;
  ~DomainEvent() = default;

  [[nodiscard]] const ReplayEventMetadata& metadata() const noexcept { return metadata_; }
  [[nodiscard]] const DomainObservation& observation() const noexcept { return observation_; }

private:
  ReplayEventMetadata metadata_;
  DomainObservation observation_;
};

// Logical immutable footprint retained by one queued DomainEvent. It includes
// the event object, metadata text, and pointed-to LiDAR/image payload bytes.
// Shared allocator/control-block overhead is intentionally excluded; the
// result is deterministic across runs and saturates at size_t max.
[[nodiscard]] std::size_t estimateDomainEventBytes(const DomainEvent& event) noexcept;

enum class ReplayVisitAction {
  Continue,
  Stop,
};

struct ReplayVisitorError {
  std::string detail;
};

using ReplayVisitResult = core::Result<ReplayVisitAction, ReplayVisitorError>;
using DomainEventVisitor = std::function<ReplayVisitResult(DomainEvent&& event)>;

struct TopicReplayStats {
  // Wire-message counts remain independent from conversion element counts.
  std::size_t messages_seen{};
  std::size_t deserialization_errors{};
  std::size_t conversion_errors{};
  std::size_t image_decode_errors{};
  std::size_t clock_guard_errors{};
  std::size_t events_emitted{};
  // Sum of accounting attached to successfully emitted DomainEvents.
  std::size_t conversion_input_elements{};
  std::size_t conversion_output_elements{};
  std::size_t conversion_discarded_elements{};
};

struct ReplayStats {
  // Counts only messages returned by the storage-level profile filter.
  std::size_t bag_messages_read{};
  std::size_t configured_messages_seen{};
  std::size_t events_emitted{};
  // Sum across every configured topic's successfully emitted events. Element
  // kinds remain available on each immutable event and topics remain split in
  // configured_topics; these totals are intentionally not wire-message counts.
  std::size_t conversion_input_elements{};
  std::size_t conversion_output_elements{};
  std::size_t conversion_discarded_elements{};
  std::map<std::string, std::size_t> unknown_topics;
  std::map<std::string, TopicReplayStats> configured_topics;
};

enum class ReplayCompletion {
  EndOfBag,
  EventLimit,
  BagMessageLimit,
  VisitorStop,
};

struct ReplayReport {
  ReplayCompletion completion{ReplayCompletion::EndOfBag};
  ReplayStats stats;
};

enum class ReplayErrorCode {
  InvalidVisitor,
  InvalidSchedule,
  ThreadStartFailed,
  BagOpenFailed,
  BagReadFailed,
  RequiredTopicMissing,
  TopicTypeMismatch,
  SequenceOverflow,
  QueueOverflow,
  VisitorFailed,
};

struct ReplayError {
  ReplayErrorCode code{};
  std::string detail;
  ReplayStats stats;
};

using ReplayResult = core::Result<ReplayReport, ReplayError>;

// Reads through rosbag2_cpp's sequential reader. Conversion and decoding
// failures are counted and skipped. Bag I/O, profile/bag incompatibility, and
// visitor errors are terminal and return the stats accumulated so far.
[[nodiscard]] ReplayResult replayRos2Bag(const std::filesystem::path& bag_uri,
                                         const ReplayProfile& profile, DomainEventVisitor visitor);

struct ScheduledReplayOptions {
  // Recorded-arrival delta / playback_rate gives the steady-wall delivery
  // delta. 1.0 is real time; values above one accelerate replay.
  double playback_rate{1.0};
  std::size_t queue_count_capacity{128U};
  std::size_t queue_byte_capacity{256U * 1024U * 1024U};
  std::size_t timing_window_capacity{1'024U};
};

struct ScheduledReplayQueueReport {
  // Terminal queue state. A clean drain has count=bytes=0 and no oldest age.
  core::PipelineQueueSnapshot terminal;
  std::size_t maximum_count{};
  std::size_t maximum_bytes{};
  std::optional<core::Duration> maximum_oldest_age;
};

struct ScheduledReplayRuntimeReport {
  ScheduledReplayQueueReport queue;
  core::PipelineTimingSample producer_total;
  core::PipelineTimingSample consumer_total;
  core::PipelineTimingStatisticsSnapshot producer_schedule_enqueue;
  core::PipelineTimingStatisticsSnapshot consumer_visit;
};

struct ScheduledReplayReport {
  ReplayCompletion completion{ReplayCompletion::EndOfBag};
  ReplayStats stats;
  ScheduledReplayRuntimeReport runtime;
};

struct ScheduledReplayError {
  ReplayErrorCode code{};
  std::string detail;
  ReplayStats stats;
  std::optional<ScheduledReplayRuntimeReport> runtime;
};

using ScheduledReplayResult = core::Result<ScheduledReplayReport, ScheduledReplayError>;

// Replays with recorded-arrival pacing on one producer and invokes the
// existing visitor from one consumer. The event queue has explicit count and
// logical-byte capacities. Any overflow is terminal: the rejected event is
// counted and no accepted event is silently dropped. All worker threads are
// stopped and joined before this function returns.
//
// The synchronous replayRos2Bag API and behavior remain unchanged.
[[nodiscard]] ScheduledReplayResult replayRos2BagScheduled(const std::filesystem::path& bag_uri,
                                                           const ReplayProfile& profile,
                                                           const ScheduledReplayOptions& options,
                                                           DomainEventVisitor visitor);

}  // namespace meridian::tools
