#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "meridian/core/observations.hpp"

namespace meridian::core {
namespace {

ObservationHeader makeHeader() {
  return ObservationHeader(SensorId("lidar"), CalibrationId("calibration"), MeasurementId(4),
                           TimeNs(100), "os_sensor");
}

TEST(Ids, RejectEmptyTextIdentity) {
  EXPECT_THROW(SensorId(""), std::invalid_argument);
  EXPECT_THROW(CalibrationId(""), std::invalid_argument);
}

TEST(ObservationHeader, RejectsEmptyFrame) {
  EXPECT_THROW(ObservationHeader(SensorId("lidar"), CalibrationId("calibration"), MeasurementId(0),
                                 TimeNs(0), ""),
               std::invalid_argument);
}

TEST(Vec3d, ReportsNonFiniteComponents) {
  EXPECT_TRUE((Vec3d{1.0, 2.0, 3.0}.isFinite()));
  EXPECT_FALSE((Vec3d{1.0, std::numeric_limits<double>::infinity(), 3.0}.isFinite()));
}

TEST(LidarSweep, SharesImmutablePointStorageAcrossCopies) {
  std::vector<LidarPoint> points;
  points.push_back(LidarPoint{.x = 1.0F,
                              .y = 2.0F,
                              .z = 3.0F,
                              .time_offset_ns = 0,
                              .source_index = 9,
                              .intensity = std::nullopt,
                              .ring = std::nullopt});
  LidarSweep sweep(makeHeader(), TimeNs(100), TimeNs(120), std::move(points));
  const LidarSweep copy = sweep;

  ASSERT_EQ(sweep.size(), 1U);
  EXPECT_EQ(sweep.points().data(), copy.points().data());
  EXPECT_EQ(sweep.points().front().source_index, 9U);
}

TEST(LidarSweep, EnforcesIntervalAndNonEmptyStorage) {
  EXPECT_THROW(LidarSweep(makeHeader(), TimeNs(2), TimeNs(1), {LidarPoint{}}),
               std::invalid_argument);
  EXPECT_THROW(LidarSweep(makeHeader(), TimeNs(1), TimeNs(2), {}), std::invalid_argument);
}

}  // namespace
}  // namespace meridian::core
