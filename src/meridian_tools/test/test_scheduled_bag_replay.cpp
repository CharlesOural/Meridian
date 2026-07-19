#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <rclcpp/time.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <string>
#include <thread>
#include <vector>

#include "meridian/tools/bag_replay.hpp"

namespace meridian::tools {
namespace {

class TemporaryScheduledBag {
public:
  TemporaryScheduledBag() {
    static std::atomic<std::uint64_t> sequence{0U};
    const auto clock_value = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("meridian-scheduled-replay-" + std::to_string(clock_value) + "-" +
             std::to_string(sequence.fetch_add(1U)));
  }

  TemporaryScheduledBag(const TemporaryScheduledBag&) = delete;
  TemporaryScheduledBag& operator=(const TemporaryScheduledBag&) = delete;

  ~TemporaryScheduledBag() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
  std::filesystem::path path_;
};

sensor_msgs::msg::NavSatFix validFix(std::int64_t timestamp_ns) {
  sensor_msgs::msg::NavSatFix fix;
  fix.header.stamp.sec = static_cast<std::int32_t>(timestamp_ns / 1'000'000'000LL);
  fix.header.stamp.nanosec = static_cast<std::uint32_t>(timestamp_ns % 1'000'000'000LL);
  fix.status.status = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
  fix.status.service = sensor_msgs::msg::NavSatStatus::SERVICE_GPS;
  fix.latitude = 48.8566;
  fix.longitude = 2.3522;
  fix.altitude = 37.5;
  fix.position_covariance = {0.04, 0.0, 0.0, 0.0, 0.09, 0.0, 0.0, 0.0, 0.25};
  fix.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_KNOWN;
  return fix;
}

void writeFixBag(const std::filesystem::path& path,
                 const std::vector<std::int64_t>& timestamps_ns) {
  constexpr char kTopic[] = "/gnss/fix";
  rosbag2_cpp::Writer writer;
  writer.open(path.string());
  for (const std::int64_t timestamp : timestamps_ns) {
    writer.write(validFix(timestamp), kTopic, rclcpp::Time{timestamp});
  }
  writer.close();
}

core::Result<ReplayProfile, ReplayProfileError> scheduledProfile(std::size_t maximum_events) {
  core::AffineClockModel clock;
  clock.revision = core::ClockRevision{1U};
  clock.source_epoch = core::SourceEpoch{2U};
  clock.rate = 1.0;
  std::vector<ReplaySource> sources;
  sources.emplace_back(GnssReplaySource{
      "/gnss/fix", ReplaySourceIdentity{"scheduled_gnss", core::ProducerId{3U}, clock}, true});
  return ReplayProfile::create(
      "scheduled", core::SessionId{4U}, core::ConfigRevision{5U}, core::CalibrationEpoch{6U},
      ReplayBounds{maximum_events, maximum_events + 4U}, std::move(sources));
}

TEST(DomainEventSizeEstimate, IncludesImmutableImagePayloadAndMetadataText) {
  ReplayEventMetadata small_metadata;
  small_metadata.topic = "/imu";
  small_metadata.source_name = "imu";
  small_metadata.conversion_accounting =
      ReplayConversionAccounting{ReplayConversionElementKind::Record, 1U, 1U, 0U};
  DomainEvent small(std::move(small_metadata), core::ImuSample{});
  const std::size_t small_bytes = estimateDomainEventBytes(small);
  EXPECT_EQ(estimateDomainEventBytes(small), small_bytes);
  DomainEvent moved_small(std::move(small));
  EXPECT_EQ(estimateDomainEventBytes(moved_small), small_bytes);
  ASSERT_TRUE(moved_small.metadata().conversion_accounting);
  EXPECT_TRUE(moved_small.metadata().conversion_accounting->valid());

  ReplayEventMetadata image_metadata;
  image_metadata.topic = "/camera/longer_topic";
  image_metadata.source_name = "camera";
  image_metadata.conversion_accounting =
      ReplayConversionAccounting{ReplayConversionElementKind::Record, 1U, 1U, 0U};
  core::CameraFrame frame;
  frame.pixels = std::make_shared<const std::vector<std::byte>>(4'096U);
  DomainEvent image(std::move(image_metadata), std::move(frame));

  EXPECT_GT(estimateDomainEventBytes(image), estimateDomainEventBytes(moved_small) + 4'000U);
}

TEST(ScheduledRos2BagReplay, AppliesRecordedArrivalRateAndDrainsWithoutDropOrSkip) {
  constexpr std::int64_t kStart = 42'000'000'000LL;
  TemporaryScheduledBag bag;
  writeFixBag(bag.path(), {kStart, kStart + 40'000'000LL, kStart + 80'000'000LL});
  auto profile = scheduledProfile(8U);
  ASSERT_TRUE(profile) << profile.error().detail;

  ScheduledReplayOptions options;
  options.playback_rate = 4.0;
  options.queue_count_capacity = 4U;
  options.queue_byte_capacity = 1U * 1024U * 1024U;
  options.timing_window_capacity = 8U;
  std::vector<std::chrono::steady_clock::time_point> arrivals;
  std::size_t accounted_visits = 0U;

  const auto result =
      replayRos2BagScheduled(bag.path(), profile.value(), options, [&](DomainEvent&& event) {
        if (!event.metadata().conversion_accounting) {
          return ReplayVisitResult::failure(
              ReplayVisitorError{"scheduled replay lost conversion accounting"});
        }
        const ReplayConversionAccounting& accounting = *event.metadata().conversion_accounting;
        if (accounting.kind != ReplayConversionElementKind::Record ||
            accounting.input_elements != 1U || accounting.output_elements != 1U ||
            accounting.discarded_non_finite != 0U || !accounting.valid()) {
          return ReplayVisitResult::failure(
              ReplayVisitorError{"scheduled replay changed conversion accounting"});
        }
        ++accounted_visits;
        arrivals.push_back(std::chrono::steady_clock::now());
        return ReplayVisitResult::success(ReplayVisitAction::Continue);
      });

  ASSERT_TRUE(result) << result.error().detail;
  EXPECT_EQ(result.value().completion, ReplayCompletion::EndOfBag);
  ASSERT_EQ(arrivals.size(), 3U);
  EXPECT_EQ(accounted_visits, 3U);
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(arrivals.back() - arrivals.front());
  EXPECT_GE(elapsed.count(), 15);
  EXPECT_EQ(result.value().stats.events_emitted, 3U);
  EXPECT_EQ(result.value().stats.conversion_input_elements, 3U);
  EXPECT_EQ(result.value().stats.conversion_output_elements, 3U);
  EXPECT_EQ(result.value().stats.conversion_discarded_elements, 0U);
  const auto topic_stats = result.value().stats.configured_topics.find("/gnss/fix");
  ASSERT_NE(topic_stats, result.value().stats.configured_topics.end());
  EXPECT_EQ(topic_stats->second.messages_seen, 3U);
  EXPECT_EQ(topic_stats->second.events_emitted, 3U);
  EXPECT_EQ(topic_stats->second.conversion_input_elements, 3U);
  EXPECT_EQ(topic_stats->second.conversion_output_elements, 3U);
  EXPECT_EQ(topic_stats->second.conversion_discarded_elements, 0U);
  const auto& runtime = result.value().runtime;
  EXPECT_TRUE(runtime.producer_total.valid());
  EXPECT_TRUE(runtime.consumer_total.valid());
  EXPECT_TRUE(runtime.producer_schedule_enqueue.valid());
  EXPECT_TRUE(runtime.consumer_visit.valid());
  EXPECT_EQ(runtime.queue.terminal.count, 0U);
  EXPECT_EQ(runtime.queue.terminal.bytes, 0U);
  EXPECT_EQ(runtime.queue.terminal.accepted, 3U);
  EXPECT_EQ(runtime.queue.terminal.rejected, 0U);
  EXPECT_EQ(runtime.queue.terminal.dropped_oldest, 0U);
  EXPECT_EQ(runtime.queue.terminal.dropped_newest, 0U);
  EXPECT_EQ(runtime.queue.terminal.skipped_stale, 0U);
  EXPECT_EQ(runtime.queue.terminal.skipped_policy, 0U);
  EXPECT_EQ(runtime.queue.terminal.accepted, runtime.consumer_visit.wall.total_samples);
  EXPECT_LE(runtime.queue.maximum_count, options.queue_count_capacity);
  EXPECT_LE(runtime.queue.maximum_bytes, options.queue_byte_capacity);
}

TEST(ScheduledRos2BagReplay, CountOverloadIsTerminalAndDrainsEveryAcceptedEvent) {
  constexpr std::int64_t kStart = 42'000'000'000LL;
  TemporaryScheduledBag bag;
  writeFixBag(bag.path(), {kStart, kStart + 1'000'000LL, kStart + 2'000'000LL, kStart + 3'000'000LL,
                           kStart + 4'000'000LL});
  auto profile = scheduledProfile(8U);
  ASSERT_TRUE(profile) << profile.error().detail;

  ScheduledReplayOptions options;
  options.playback_rate = 1.0e9;
  options.queue_count_capacity = 1U;
  options.queue_byte_capacity = 1U * 1024U * 1024U;
  std::atomic<std::size_t> visits{0U};

  const auto result =
      replayRos2BagScheduled(bag.path(), profile.value(), options, [&](DomainEvent&&) {
        if (visits.fetch_add(1U) == 0U) {
          std::this_thread::sleep_for(std::chrono::milliseconds(75));
        }
        return ReplayVisitResult::success(ReplayVisitAction::Continue);
      });

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ReplayErrorCode::QueueOverflow);
  ASSERT_TRUE(result.error().runtime);
  const auto& queue = result.error().runtime->queue;
  EXPECT_EQ(queue.terminal.count, 0U);
  EXPECT_EQ(queue.terminal.bytes, 0U);
  EXPECT_EQ(queue.terminal.rejected, 1U);
  EXPECT_EQ(queue.terminal.dropped_oldest, 0U);
  EXPECT_EQ(queue.terminal.dropped_newest, 0U);
  EXPECT_EQ(queue.terminal.skipped_stale, 0U);
  EXPECT_EQ(queue.terminal.skipped_policy, 0U);
  EXPECT_EQ(queue.maximum_count, 1U);
  EXPECT_EQ(visits.load(), queue.terminal.accepted);
  EXPECT_EQ(queue.terminal.accepted, result.error().runtime->consumer_visit.wall.total_samples);
}

TEST(ScheduledRos2BagReplay, ByteOverloadRejectsBeforeVisitorWithoutSilentDisposition) {
  constexpr std::int64_t kStart = 42'000'000'000LL;
  TemporaryScheduledBag bag;
  writeFixBag(bag.path(), {kStart});
  auto profile = scheduledProfile(2U);
  ASSERT_TRUE(profile) << profile.error().detail;

  ScheduledReplayOptions options;
  options.playback_rate = 1.0e9;
  options.queue_count_capacity = 2U;
  options.queue_byte_capacity = 1U;
  std::atomic<std::size_t> visits{0U};

  const auto result =
      replayRos2BagScheduled(bag.path(), profile.value(), options, [&](DomainEvent&&) {
        ++visits;
        return ReplayVisitResult::success(ReplayVisitAction::Continue);
      });

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ReplayErrorCode::QueueOverflow);
  ASSERT_TRUE(result.error().runtime);
  EXPECT_EQ(visits.load(), 0U);
  EXPECT_EQ(result.error().runtime->queue.terminal.accepted, 0U);
  EXPECT_EQ(result.error().runtime->queue.terminal.rejected, 1U);
  EXPECT_EQ(result.error().runtime->queue.maximum_count, 0U);
  EXPECT_EQ(result.error().runtime->queue.maximum_bytes, 0U);
}

TEST(ScheduledRos2BagReplay, PropagatesVisitorFailureAndAccountsPendingCancellation) {
  constexpr std::int64_t kStart = 42'000'000'000LL;
  TemporaryScheduledBag bag;
  writeFixBag(bag.path(),
              {kStart, kStart + 1'000'000LL, kStart + 2'000'000LL, kStart + 3'000'000LL});
  auto profile = scheduledProfile(8U);
  ASSERT_TRUE(profile) << profile.error().detail;

  ScheduledReplayOptions options;
  options.playback_rate = 1.0e9;
  options.queue_count_capacity = 8U;
  options.queue_byte_capacity = 1U * 1024U * 1024U;

  const auto result =
      replayRos2BagScheduled(bag.path(), profile.value(), options, [&](DomainEvent&&) {
        return ReplayVisitResult::failure(
            ReplayVisitorError{"intentional scheduled visitor failure"});
      });

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ReplayErrorCode::VisitorFailed);
  EXPECT_EQ(result.error().detail, "intentional scheduled visitor failure");
  ASSERT_TRUE(result.error().runtime);
  EXPECT_EQ(result.error().runtime->queue.terminal.count, 0U);
  EXPECT_EQ(result.error().runtime->queue.terminal.bytes, 0U);
  EXPECT_EQ(result.error().runtime->consumer_visit.dispositions.failed, 1U);
  EXPECT_EQ(result.error().runtime->queue.terminal.dropped_oldest, 0U);
  EXPECT_EQ(result.error().runtime->queue.terminal.dropped_newest, 0U);
  EXPECT_EQ(result.error().runtime->queue.terminal.accepted,
            result.error().runtime->consumer_visit.wall.total_samples +
                result.error().runtime->queue.terminal.skipped_policy);
}

TEST(ScheduledRos2BagReplay, VisitorStopCancelsAndJoinsAsSuccessfulStop) {
  constexpr std::int64_t kStart = 42'000'000'000LL;
  TemporaryScheduledBag bag;
  writeFixBag(bag.path(), {kStart, kStart + 1'000'000LL, kStart + 2'000'000LL});
  auto profile = scheduledProfile(8U);
  ASSERT_TRUE(profile) << profile.error().detail;

  ScheduledReplayOptions options;
  options.playback_rate = 1.0e9;
  options.queue_count_capacity = 8U;
  options.queue_byte_capacity = 1U * 1024U * 1024U;
  std::atomic<std::size_t> visits{0U};

  const auto result =
      replayRos2BagScheduled(bag.path(), profile.value(), options, [&](DomainEvent&&) {
        ++visits;
        return ReplayVisitResult::success(ReplayVisitAction::Stop);
      });

  ASSERT_TRUE(result) << result.error().detail;
  EXPECT_EQ(result.value().completion, ReplayCompletion::VisitorStop);
  EXPECT_EQ(visits.load(), 1U);
  EXPECT_EQ(result.value().runtime.queue.terminal.count, 0U);
  EXPECT_EQ(result.value().runtime.queue.terminal.bytes, 0U);
  EXPECT_EQ(result.value().runtime.consumer_visit.dispositions.accepted, 1U);
  EXPECT_EQ(result.value().runtime.queue.terminal.dropped_oldest, 0U);
  EXPECT_EQ(result.value().runtime.queue.terminal.dropped_newest, 0U);
  EXPECT_EQ(result.value().runtime.queue.terminal.accepted,
            result.value().runtime.consumer_visit.wall.total_samples +
                result.value().runtime.queue.terminal.skipped_policy);
}

TEST(ScheduledRos2BagReplay, PropagatesReaderOpenFailureAfterJoiningBothWorkers) {
  auto profile = scheduledProfile(2U);
  ASSERT_TRUE(profile) << profile.error().detail;
  const std::filesystem::path missing =
      std::filesystem::temp_directory_path() / "meridian-scheduled-replay-definitely-missing";

  const auto result = replayRos2BagScheduled(
      missing, profile.value(), ScheduledReplayOptions{},
      [](DomainEvent&&) { return ReplayVisitResult::success(ReplayVisitAction::Continue); });

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ReplayErrorCode::BagOpenFailed);
  ASSERT_TRUE(result.error().runtime);
  EXPECT_TRUE(result.error().runtime->producer_total.valid());
  EXPECT_TRUE(result.error().runtime->consumer_total.valid());
  EXPECT_EQ(result.error().runtime->queue.terminal.accepted, 0U);
  EXPECT_EQ(result.error().runtime->queue.terminal.rejected, 0U);
}

TEST(ScheduledRos2BagReplay, RejectsInvalidScheduleBeforeStartingWorkers) {
  auto profile = scheduledProfile(2U);
  ASSERT_TRUE(profile) << profile.error().detail;
  ScheduledReplayOptions options;
  options.playback_rate = 0.0;

  const auto result = replayRos2BagScheduled(
      "/unused", profile.value(), options,
      [](DomainEvent&&) { return ReplayVisitResult::success(ReplayVisitAction::Continue); });

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ReplayErrorCode::InvalidSchedule);
  EXPECT_FALSE(result.error().runtime);
}

}  // namespace
}  // namespace meridian::tools
