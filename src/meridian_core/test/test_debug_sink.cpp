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
  sink.record(PreintegrationEvent{.from_state = StateId(1),
                                  .to_state = StateId(2),
                                  .support = TimeRange(TimeNs(10), TimeNs(20)),
                                  .source_sample_count = 3,
                                  .integration_segment_count = 2,
                                  .maximum_source_gap_ns = 5,
                                  .delta_rotation_vector_rad = {},
                                  .delta_velocity_m_s = {},
                                  .delta_position_m = {},
                                  .backend = "test"});
  sink.record(InitializationEvent{.event_time = TimeNs(20),
                                  .mode = InitializationMode::kStatic,
                                  .status = InitializationStatus::kCollecting,
                                  .reason = "collecting",
                                  .quality = {},
                                  .accepted_seed = std::nullopt});
  sink.record(BootstrapPoseEvent{.measurement_id = MeasurementId(8),
                                 .measurement_time = TimeNs(20),
                                 .odom_from_lidar = {},
                                 .source_point_count = 10,
                                 .correspondence_count = 8,
                                 .point_rmse_m = 0.1,
                                 .hessian_condition_number = 2.0,
                                 .accepted = true});
  sink.record(LocalRegistrationMapEvent{
      .event_time = TimeNs(20),
      .state_id = StateId(2),
      .estimator_revision = 1,
      .points = std::make_shared<const std::vector<LidarPreviewPoint>>(
          std::vector<LidarPreviewPoint>{{.x = 1.0F, .y = 2.0F, .z = 3.0F}})});
  sink.record(LocalEstimatorEvent{.state_id = StateId(2),
                                  .event_time = TimeNs(20),
                                  .estimator_revision = 1,
                                  .outcome = "accepted",
                                  .active_state_count = 2,
                                  .imu_factor_count = 1,
                                  .lidar_batch_count = 1,
                                  .active_lidar_rows = 100,
                                  .finalized_lidar_rows = 10,
                                  .finalized_map_points = 1'000,
                                  .selected_active_owners = 2,
                                  .registration_correspondences = 110,
                                  .marginal_prior_rank = 15,
                                  .registration_rmse_m = 0.1,
                                  .initial_cost = 2.0,
                                  .final_cost = 1.0,
                                  .pose_correction_translation_m = 0.01,
                                  .pose_correction_rotation_rad = 0.001,
                                  .accepted = true,
                                  .prepared_target_points = 2'000,
                                  .prepared_source_points = 1'900,
                                  .association_pass_count = 2,
                                  .association_input_points = 3'800,
                                  .association_rows_before_cap = 3'000,
                                  .registration_iterations = 0,
                                  .live_query_voxel_probes = 12'000,
                                  .finalized_query_voxel_probes = 6'000,
                                  .reassociated_rows = 1'400,
                                  .rejected_stale_rows = 100});
}

}  // namespace
}  // namespace meridian::core
