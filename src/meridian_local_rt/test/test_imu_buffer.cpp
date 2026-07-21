#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "meridian/core/observations.hpp"
#include "meridian/local_rt/imu_buffer.hpp"

namespace meridian::local_rt {
namespace {

constexpr std::int64_t kMillisecondNs = 1'000'000;

core::ImuSample sample(std::uint64_t id, std::int64_t time_ns, double value) {
  return core::ImuSample(
      core::ObservationHeader(core::SensorId("imu"), core::CalibrationId("calibration"),
                              core::MeasurementId(id), core::TimeNs(time_ns), "imu"),
      {.x = value, .y = 2.0 * value, .z = 0.0}, {.x = 0.0, .y = value, .z = 9.80665});
}

TEST(ImuBuffer, InterpolatesExactBoundariesAndBuildsDisjointMidpointSegments) {
  ImuBuffer buffer({.capacity = 8U, .maximum_gap = std::chrono::milliseconds(20)});
  ASSERT_TRUE(buffer.insert(sample(1U, 0, 0.0)).ok());
  ASSERT_TRUE(buffer.insert(sample(2U, 10 * kMillisecondNs, 10.0)).ok());
  ASSERT_TRUE(buffer.insert(sample(3U, 20 * kMillisecondNs, 20.0)).ok());

  const ImuIntervalResult result =
      buffer.interval(core::TimeNs(5 * kMillisecondNs), core::TimeNs(15 * kMillisecondNs));
  ASSERT_TRUE(result.ok());
  ASSERT_NE(result.value(), nullptr);
  const ImuInterval& interval = *result.value();
  ASSERT_EQ(interval.segments().size(), 2U);
  EXPECT_EQ(interval.sourceSampleCount(), 3U);
  EXPECT_EQ(interval.segments()[0].support(),
            core::TimeRange(core::TimeNs(5 * kMillisecondNs), core::TimeNs(10 * kMillisecondNs)));
  EXPECT_EQ(interval.segments()[1].support(),
            core::TimeRange(core::TimeNs(10 * kMillisecondNs), core::TimeNs(15 * kMillisecondNs)));
  EXPECT_NEAR(interval.segments()[0].angularVelocityRadS().x(), 7.5, 1.0e-12);
  EXPECT_NEAR(interval.segments()[1].angularVelocityRadS().x(), 12.5, 1.0e-12);
  EXPECT_NEAR(interval.durationSeconds(), 0.010, 1.0e-15);

  const ImuIntervalResult left =
      buffer.interval(core::TimeNs(5 * kMillisecondNs), core::TimeNs(10 * kMillisecondNs));
  const ImuIntervalResult right =
      buffer.interval(core::TimeNs(10 * kMillisecondNs), core::TimeNs(15 * kMillisecondNs));
  ASSERT_TRUE(left.ok());
  ASSERT_TRUE(right.ok());
  EXPECT_EQ(left.value()->segments().back().support().end(),
            right.value()->segments().front().support().begin());
  EXPECT_NEAR(left.value()->durationSeconds() + right.value()->durationSeconds(), 0.010, 1.0e-15);
}

TEST(ImuBuffer, RejectsOrderingAndReportsBoundedEviction) {
  ImuBuffer buffer({.capacity = 2U, .maximum_gap = std::chrono::milliseconds(20)});
  EXPECT_TRUE(buffer.insert(sample(1U, 0, 0.0)).ok());
  EXPECT_EQ(buffer.insert(sample(2U, 0, 1.0)).status, ImuInsertStatus::kDuplicateTimestamp);
  EXPECT_TRUE(buffer.insert(sample(3U, 10 * kMillisecondNs, 1.0)).ok());
  EXPECT_EQ(buffer.insert(sample(4U, 5 * kMillisecondNs, 1.0)).status,
            ImuInsertStatus::kOutOfOrder);
  const ImuInsertResult inserted = buffer.insert(sample(5U, 20 * kMillisecondNs, 2.0));
  EXPECT_TRUE(inserted.ok());
  EXPECT_EQ(inserted.evicted_samples, 1U);
  EXPECT_EQ(buffer.size(), 2U);

  const core::ImuSample non_finite =
      sample(6U, 30 * kMillisecondNs, std::numeric_limits<double>::quiet_NaN());
  EXPECT_EQ(buffer.insert(non_finite).status, ImuInsertStatus::kNonFinite);
}

TEST(ImuBuffer, ReturnsTypedFailuresForMissingSupportAndSourceGaps) {
  ImuBuffer buffer({.capacity = 8U, .maximum_gap = std::chrono::milliseconds(15)});
  ASSERT_TRUE(buffer.insert(sample(1U, 0, 0.0)).ok());
  ASSERT_TRUE(buffer.insert(sample(2U, 10 * kMillisecondNs, 1.0)).ok());
  ASSERT_TRUE(buffer.insert(sample(3U, 30 * kMillisecondNs, 3.0)).ok());

  const ImuIntervalResult before =
      buffer.interval(core::TimeNs(-1), core::TimeNs(5 * kMillisecondNs));
  ASSERT_FALSE(before.ok());
  EXPECT_EQ(before.error()->code, ImuIntervalErrorCode::kBeginNotBracketed);

  const ImuIntervalResult after =
      buffer.interval(core::TimeNs(5 * kMillisecondNs), core::TimeNs(40 * kMillisecondNs));
  ASSERT_FALSE(after.ok());
  EXPECT_EQ(after.error()->code, ImuIntervalErrorCode::kEndNotBracketed);

  const ImuIntervalResult gap =
      buffer.interval(core::TimeNs(5 * kMillisecondNs), core::TimeNs(25 * kMillisecondNs));
  ASSERT_FALSE(gap.ok());
  EXPECT_EQ(gap.error()->code, ImuIntervalErrorCode::kSourceGapTooLarge);
  ASSERT_TRUE(gap.error()->offending_gap.has_value());
  EXPECT_EQ(*gap.error()->offending_gap,
            core::TimeRange(core::TimeNs(10 * kMillisecondNs), core::TimeNs(30 * kMillisecondNs)));
}

}  // namespace
}  // namespace meridian::local_rt
