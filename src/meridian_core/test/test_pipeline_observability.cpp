#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "meridian/core/pipeline_observability.hpp"

namespace meridian::core {
namespace {

PipelineStage stage(std::uint64_t id = 1U, std::string_view name = "lidar.registration") {
  const auto result = makePipelineStage(PipelineStageId{id}, name);
  if (!result) {
    throw std::runtime_error("test stage is invalid");
  }
  return *result;
}

PipelineTimingSample sample(const PipelineStage& sample_stage, std::int64_t wall_nanoseconds,
                            std::optional<std::int64_t> cpu_nanoseconds = std::nullopt) {
  PipelineTimingSample output;
  output.stage = sample_stage;
  output.wall_duration = Duration{wall_nanoseconds};
  if (cpu_nanoseconds) {
    output.thread_cpu_duration = Duration{*cpu_nanoseconds};
  }
  return output;
}

TEST(PipelineFixedText, RejectsEmptyAndOversizedNamesWithoutTruncation) {
  EXPECT_FALSE(PipelineStageName::make(""));
  EXPECT_TRUE(PipelineStageName::make(std::string(63U, 'a')));
  EXPECT_FALSE(PipelineStageName::make(std::string(64U, 'a')));
}

TEST(PipelineTimingSample, ValidatesIdentityDurationAndQueueBounds) {
  PipelineTimingSample value = sample(stage(), 1'000, 800);
  value.work.measurement = MeasurementId{8U};
  value.work.state = StateId{9U};
  value.work.local_revision = LocalGraphRevision{10U};
  value.work.global_revision = PipelineGlobalRevision{11U};
  PipelineQueueSnapshot queue;
  queue.name = *PipelineQueueName::make("lidar.input");
  queue.count = 2U;
  queue.bytes = 400U;
  queue.count_capacity = 8U;
  queue.byte_capacity = 1'000U;
  queue.oldest_age = Duration{20'000};
  value.queue = queue;
  ASSERT_TRUE(value.valid());

  value.queue->count = 9U;
  EXPECT_FALSE(value.valid());
  value.queue->count = 2U;
  value.thread_cpu_duration = Duration{-1};
  EXPECT_FALSE(value.valid());
}

TEST(BoundedPipelineTimingAccumulator,
     KeepsLatestWindowAndComputesDeterministicNearestRankQuantiles) {
  const PipelineStage configured_stage = stage();
  BoundedPipelineTimingAccumulator accumulator(configured_stage, 4U);
  constexpr std::array<std::int64_t, 6U> wall_values{10, 20, 30, 40, 50, 60};
  constexpr std::array<std::int64_t, 6U> cpu_values{1, 2, 3, 4, 5, 6};
  for (std::size_t index = 0; index < wall_values.size(); ++index) {
    PipelineTimingSample value = sample(configured_stage, wall_values[index], cpu_values[index]);
    value.disposition = index == 4U ? PipelineDisposition::Dropped : PipelineDisposition::Completed;
    ASSERT_EQ(accumulator.observe(value), PipelineTimingObserveStatus::Accepted);
  }

  const PipelineTimingStatisticsSnapshot output = accumulator.snapshot();
  ASSERT_TRUE(output.valid());
  EXPECT_EQ(output.wall.total_samples, 6U);
  EXPECT_EQ(output.wall.window_samples, 4U);
  ASSERT_TRUE(output.wall.minimum);
  ASSERT_TRUE(output.wall.p50);
  ASSERT_TRUE(output.wall.p95);
  ASSERT_TRUE(output.wall.p99);
  ASSERT_TRUE(output.wall.maximum);
  EXPECT_EQ(output.wall.minimum->nanoseconds, 30);
  EXPECT_EQ(output.wall.p50->nanoseconds, 40);
  EXPECT_EQ(output.wall.p95->nanoseconds, 60);
  EXPECT_EQ(output.wall.p99->nanoseconds, 60);
  EXPECT_EQ(output.wall.maximum->nanoseconds, 60);
  ASSERT_TRUE(output.wall.mean_nanoseconds);
  EXPECT_DOUBLE_EQ(*output.wall.mean_nanoseconds, 45.0);
  EXPECT_EQ(output.dispositions.completed, 5U);
  EXPECT_EQ(output.dispositions.dropped, 1U);
}

TEST(BoundedPipelineTimingAccumulator, TracksOptionalCpuSamplesAndLatestQueueWithoutGrowingWindow) {
  const PipelineStage configured_stage = stage();
  BoundedPipelineTimingAccumulator accumulator(configured_stage, 2U);

  PipelineTimingSample first = sample(configured_stage, 100, 80);
  PipelineQueueSnapshot first_queue;
  first_queue.name = *PipelineQueueName::make("lidar.input");
  first_queue.count = 1U;
  first.queue = first_queue;
  ASSERT_EQ(accumulator.observe(first), PipelineTimingObserveStatus::Accepted);

  PipelineTimingSample second = sample(configured_stage, 200);
  PipelineQueueSnapshot second_queue = first_queue;
  second_queue.count = 2U;
  second_queue.skipped_stale = 3U;
  second.queue = second_queue;
  ASSERT_EQ(accumulator.observe(second), PipelineTimingObserveStatus::Accepted);

  const auto output = accumulator.snapshot();
  EXPECT_EQ(accumulator.windowCapacity(), 2U);
  EXPECT_EQ(output.thread_cpu.total_samples, 1U);
  EXPECT_EQ(output.thread_cpu.window_samples, 1U);
  ASSERT_TRUE(output.latest_queue);
  EXPECT_EQ(output.latest_queue->count, 2U);
  EXPECT_EQ(output.latest_queue->skipped_stale, 3U);
}

TEST(BoundedPipelineTimingAccumulator, RejectsInvalidAndMismatchedSamples) {
  const PipelineStage configured_stage = stage();
  BoundedPipelineTimingAccumulator accumulator(configured_stage, 4U);
  EXPECT_EQ(accumulator.observe(sample(stage(2U, "imu.preintegration"), 10)),
            PipelineTimingObserveStatus::StageMismatch);
  EXPECT_EQ(accumulator.observe(sample(configured_stage, -1)),
            PipelineTimingObserveStatus::InvalidSample);
  EXPECT_EQ(accumulator.snapshot().wall.total_samples, 0U);
}

TEST(ThreadCpuWallTimer, ReportsNonnegativeSteadyAndLinuxThreadCpuTime) {
  const auto before = currentThreadCpuTime();
  ThreadCpuWallTimer timer;
  volatile std::uint64_t value = 0U;
  for (std::uint64_t index = 0; index < 100'000U; ++index) {
    value = value + index;
  }
  const CpuWallDuration elapsed = timer.elapsed();
  const auto after = currentThreadCpuTime();

  EXPECT_NE(value, 0U);
  EXPECT_GE(elapsed.wall.nanoseconds, 0);
#if defined(__linux__)
  ASSERT_TRUE(before);
  ASSERT_TRUE(after);
  ASSERT_TRUE(elapsed.thread_cpu);
  EXPECT_GE(after->nanoseconds, before->nanoseconds);
  EXPECT_GE(elapsed.thread_cpu->nanoseconds, 0);
#endif
}

}  // namespace
}  // namespace meridian::core
