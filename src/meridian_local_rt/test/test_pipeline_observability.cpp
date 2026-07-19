#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <string_view>

#include "meridian/local/pipeline_observability.hpp"

namespace meridian::local {
namespace {

TEST(LocalPipelineTimingStage, ExposesStableUniqueCoreRecords) {
  constexpr std::array<std::string_view, kLocalPipelineTimingStageCount> kExpectedNames{{
      "local.initialization.stationary_probe",
      "local.initialization.bootstrap_acquisition_deskew",
      "local.lidar.registration_view_build",
      "local.lidar.target_build_update",
      "local.lidar.correspondence_registration_solve",
      "local.initialization.motion_batch_solve_refinement",
      "local.tracking.deskew",
      "local.graph.transaction_update",
      "local.lidar.finalized_target_update",
      "local.lidar.composite_target_index_build",
      "local.lidar.factor_batch_prepare",
  }};

  for (std::size_t index = 0U; index < kExpectedNames.size(); ++index) {
    const auto stage = static_cast<LocalPipelineTimingStage>(index);
    const core::PipelineStage& record = localPipelineTimingStageRecord(stage);
    EXPECT_TRUE(record.valid());
    EXPECT_EQ(record.id.value(), 30'001U + index);
    EXPECT_EQ(record.name.view(), kExpectedNames[index]);
  }
  EXPECT_THROW(static_cast<void>(
                   localPipelineTimingStageRecord(static_cast<LocalPipelineTimingStage>(255U))),
               std::out_of_range);
}

TEST(LocalPipelineTimingRecorder, RetainsOnlyConfiguredRollingWindowAndAllTimeCounts) {
  LocalPipelineTimingConfig config;
  config.window_capacity = 2U;
  LocalPipelineTimingRecorder recorder(config);

  const auto observe = [&](std::int64_t wall_nanoseconds, core::PipelineDisposition disposition) {
    core::CpuWallDuration duration;
    duration.wall = core::Duration{wall_nanoseconds};
    duration.thread_cpu = core::Duration{wall_nanoseconds / 2};
    return recorder.observe(LocalPipelineTimingStage::TrackingDeskew, duration, disposition);
  };
  EXPECT_EQ(observe(300, core::PipelineDisposition::Completed),
            core::PipelineTimingObserveStatus::Accepted);
  EXPECT_EQ(observe(100, core::PipelineDisposition::Failed),
            core::PipelineTimingObserveStatus::Accepted);
  EXPECT_EQ(observe(200, core::PipelineDisposition::Completed),
            core::PipelineTimingObserveStatus::Accepted);

  const LocalPipelineTimingReport report = recorder.snapshot();
  ASSERT_TRUE(report.valid());
  EXPECT_EQ(report.schema_version, LocalPipelineTimingReport::kSchemaVersion);
  EXPECT_EQ(report.window_capacity, 2U);
  EXPECT_EQ(report.span_semantics, LocalPipelineTimingSpanSemantics::Inclusive);
  EXPECT_EQ(localPipelineTimingSpanSemanticsName(report.span_semantics), "inclusive");

  const core::PipelineTimingStatisticsSnapshot* tracking =
      report.find(LocalPipelineTimingStage::TrackingDeskew);
  ASSERT_NE(tracking, nullptr);
  EXPECT_EQ(tracking->wall.total_samples, 3U);
  EXPECT_EQ(tracking->wall.window_samples, 2U);
  EXPECT_EQ(tracking->wall.window_capacity, 2U);
  ASSERT_TRUE(tracking->wall.minimum);
  ASSERT_TRUE(tracking->wall.p50);
  ASSERT_TRUE(tracking->wall.p95);
  ASSERT_TRUE(tracking->wall.maximum);
  ASSERT_TRUE(tracking->wall.mean_nanoseconds);
  EXPECT_EQ(tracking->wall.minimum->nanoseconds, 100);
  EXPECT_EQ(tracking->wall.p50->nanoseconds, 100);
  EXPECT_EQ(tracking->wall.p95->nanoseconds, 200);
  EXPECT_EQ(tracking->wall.maximum->nanoseconds, 200);
  EXPECT_DOUBLE_EQ(*tracking->wall.mean_nanoseconds, 150.0);
  EXPECT_EQ(tracking->thread_cpu.total_samples, 3U);
  EXPECT_EQ(tracking->dispositions.completed, 2U);
  EXPECT_EQ(tracking->dispositions.failed, 1U);

  const core::PipelineTimingStatisticsSnapshot* stationary =
      report.find(LocalPipelineTimingStage::StationaryProbe);
  ASSERT_NE(stationary, nullptr);
  EXPECT_EQ(stationary->wall.total_samples, 0U);
  EXPECT_EQ(stationary->wall.window_samples, 0U);
  EXPECT_EQ(stationary->wall.window_capacity, 2U);
  EXPECT_EQ(report.find(static_cast<LocalPipelineTimingStage>(255U)), nullptr);
}

TEST(LocalPipelineTimingRecorder, RejectsUnboundedOrEmptyConfiguration) {
  LocalPipelineTimingConfig empty;
  empty.window_capacity = 0U;
  EXPECT_THROW(static_cast<void>(LocalPipelineTimingRecorder(empty)), std::invalid_argument);

  LocalPipelineTimingConfig excessive;
  excessive.window_capacity = kMaximumLocalPipelineTimingWindowCapacity + 1U;
  EXPECT_THROW(static_cast<void>(LocalPipelineTimingRecorder(excessive)), std::invalid_argument);
}

}  // namespace
}  // namespace meridian::local
