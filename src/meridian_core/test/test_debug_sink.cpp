#include <gtest/gtest.h>

#include "meridian/core/debug_sink.hpp"

namespace meridian::core {
namespace {

TEST(NullDebugSink, RequestsNoHeavyGeometryAndAcceptsEveryEvent) {
  NullDebugSink sink;
  EXPECT_FALSE(sink.wantsLidarPreview());
  sink.record(ImuAcceptedEvent{.measurement_id = MeasurementId(1),
                               .measurement_time = TimeNs(2),
                               .angular_velocity_rad_s = {},
                               .specific_force_m_s2 = {},
                               .arrival_steady_ns = 0,
                               .conversion_duration_ns = 0});
  sink.record(LidarAcceptedEvent{.measurement_id = MeasurementId(3),
                                 .measurement_time = TimeNs(4),
                                 .acquisition_duration_ns = 0,
                                 .source_points = 0,
                                 .accepted_points = 0,
                                 .nonfinite_xyz_points = 0,
                                 .zero_xyz_points = 0,
                                 .flattened_time_regressions = 0,
                                 .arrival_steady_ns = 0,
                                 .conversion_duration_ns = 0,
                                 .decode_queue_depth = 0});
  sink.record(LidarPreviewEvent{
      .measurement_id = MeasurementId(5), .measurement_time = TimeNs(6), .points = nullptr});
  sink.record(IngressFailureEvent{.sensor_id = "lidar",
                                  .measurement_id = MeasurementId(7),
                                  .measurement_time = std::nullopt,
                                  .arrival_steady_ns = 0,
                                  .conversion_duration_ns = 0,
                                  .error_code = "invalid",
                                  .field = "t",
                                  .detail = "bad"});
}

}  // namespace
}  // namespace meridian::core
