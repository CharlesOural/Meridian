#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "meridian/core/time.hpp"

namespace meridian::core {
namespace {

TEST(TimeNs, BuildsFromNormalizedRosComponents) {
  const auto time = TimeNs::fromSecNanosec(1'625'132'068, 858'555'136);
  ASSERT_TRUE(time.has_value());
  EXPECT_EQ(time->count(), 1'625'132'068'858'555'136LL);
}

TEST(TimeNs, RejectsUnnormalizedNanosecondsAndOverflow) {
  EXPECT_FALSE(TimeNs::fromSecNanosec(0, 1'000'000'000).has_value());
  EXPECT_FALSE(TimeNs::fromSecNanosec(std::numeric_limits<std::int64_t>::max(), 0).has_value());
}

TEST(TimeNs, CoversBothSignedInt64BoundaryValues) {
  const auto minimum = TimeNs::fromSecNanosec(-9'223'372'037LL, 145'224'192U);
  ASSERT_TRUE(minimum.has_value());
  EXPECT_EQ(minimum->count(), std::numeric_limits<std::int64_t>::min());
  EXPECT_FALSE(TimeNs::fromSecNanosec(-9'223'372'037LL, 145'224'191U).has_value());

  const auto maximum = TimeNs::fromSecNanosec(9'223'372'036LL, 854'775'807U);
  ASSERT_TRUE(maximum.has_value());
  EXPECT_EQ(maximum->count(), std::numeric_limits<std::int64_t>::max());
  EXPECT_FALSE(TimeNs::fromSecNanosec(9'223'372'036LL, 854'775'808U).has_value());
}

TEST(TimeNs, CheckedArithmeticCatchesBothOverflowDirections) {
  const TimeNs maximum(std::numeric_limits<std::int64_t>::max());
  const TimeNs minimum(std::numeric_limits<std::int64_t>::min());
  EXPECT_FALSE(TimeNs::checkedAdd(maximum, 1).has_value());
  EXPECT_FALSE(TimeNs::checkedAdd(minimum, -1).has_value());

  const auto sum = TimeNs::checkedAdd(TimeNs(10), -3);
  ASSERT_TRUE(sum.has_value());
  EXPECT_EQ(sum->count(), 7);

  EXPECT_FALSE(TimeNs::checkedDifference(maximum, TimeNs(-1)).has_value());
  EXPECT_FALSE(TimeNs::checkedDifference(minimum, TimeNs(1)).has_value());
  EXPECT_EQ(TimeNs::checkedDifference(TimeNs(10), TimeNs(3)), 7);
}

TEST(TimeRange, EnforcesOrderingAndUsesHalfOpenContainment) {
  const TimeRange range(TimeNs(10), TimeNs(20));
  EXPECT_EQ(range.begin(), TimeNs(10));
  EXPECT_EQ(range.end(), TimeNs(20));
  EXPECT_FALSE(range.empty());
  EXPECT_TRUE(range.contains(TimeNs(10)));
  EXPECT_TRUE(range.contains(TimeNs(19)));
  EXPECT_FALSE(range.contains(TimeNs(20)));
  ASSERT_TRUE(range.durationNs().has_value());
  EXPECT_EQ(*range.durationNs(), 10);

  const TimeRange empty(TimeNs(4), TimeNs(4));
  EXPECT_TRUE(empty.empty());
  EXPECT_THROW(static_cast<void>(TimeRange(TimeNs(5), TimeNs(4))), std::invalid_argument);
}

TEST(TimeRange, ReportsDurationOverflow) {
  const TimeRange full(TimeNs(std::numeric_limits<std::int64_t>::min()),
                       TimeNs(std::numeric_limits<std::int64_t>::max()));
  EXPECT_FALSE(full.durationNs().has_value());
}

}  // namespace
}  // namespace meridian::core
