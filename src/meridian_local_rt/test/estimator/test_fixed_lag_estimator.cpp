#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "meridian/local_rt/combined_preintegration.hpp"
#include "meridian/local_rt/estimator/fixed_lag_estimator.hpp"

namespace meridian::local_rt::estimator {
namespace {

constexpr std::int64_t kSecondNs = 1'000'000'000;

lidar::PointCloud fullRankCloud() {
  return {
      lidar::Point3d{-4.0, 2.0, 1.0},  lidar::Point3d{-3.0, -2.0, -1.0},
      lidar::Point3d{-2.0, 1.0, 3.0},  lidar::Point3d{-1.0, 3.0, -2.0},
      lidar::Point3d{0.0, -3.0, 2.0},  lidar::Point3d{1.0, 2.0, 4.0},
      lidar::Point3d{2.0, -1.0, -3.0}, lidar::Point3d{2.0, 3.0, -4.0},
      lidar::Point3d{3.0, 4.0, 1.0},   lidar::Point3d{4.0, -2.0, 3.0},
  };
}

lidar::ScanToMapOptions scanOptions() {
  lidar::ScanToMapOptions options;
  options.maximum_active_owners = 3U;
  options.maximum_factor_rows = 32U;
  options.minimum_correspondences = 6U;
  options.maximum_icp_iterations = 4U;
  options.maximum_backtracking_steps = 4U;
  options.maximum_correspondence_distance_m = 0.75;
  options.translation_convergence_m = 0.2;
  options.rotation_convergence_rad = 0.2;
  options.maximum_translation_step_m = 0.5;
  options.maximum_rotation_step_rad = 0.2;
  options.maximum_prediction_correction_m = 1.0;
  options.maximum_prediction_correction_rad = 0.5;
  options.active_owner_index = lidar::VoxelTargetOptions{
      .voxel_size_m = 0.2,
      .retention_radius_m = 100.0,
      .max_voxels = 100U,
      .max_points_per_voxel = 2U,
      .minimum_point_spacing_m = 0.0,
      .max_neighbor_voxel_radius = 4U,
  };
  options.finalized_base = options.active_owner_index;
  return options;
}

ImuInterval stationaryInterval(std::int64_t begin_ns) {
  const core::TimeNs begin(begin_ns);
  const core::TimeNs end(begin_ns + kSecondNs);
  ImuIntegrationSegments segments;
  segments.emplace_back(core::TimeRange(begin, end), Eigen::Vector3d::Zero(),
                        Eigen::Vector3d(0.0, 0.0, 9.80665));
  return ImuInterval(core::TimeRange(begin, end), std::move(segments), 2U);
}

CombinedPreintegration preintegration(std::int64_t begin_ns) {
  const PreintegrationResult result =
      GtsamCombinedPreintegrator(ImuModel{}).integrate(stationaryInterval(begin_ns), {});
  if (!result.ok()) {
    throw std::runtime_error(result.error()->message);
  }
  return *result.value();
}

core::NavigationState predictedState(std::uint64_t id, std::int64_t time_ns) {
  return core::NavigationState(core::StateId(id), core::TimeNs(time_ns),
                               core::Pose3d({.x = 0.1, .y = -0.05, .z = 0.04}, {}), {}, {});
}

lidar::ScanFrame frame(std::uint64_t id, std::int64_t time_ns) {
  const lidar::PointCloud cloud = fullRankCloud();
  return lidar::ScanFrame{.state_id = core::StateId(id),
                          .time = core::TimeNs(time_ns),
                          .target_points_lidar = cloud,
                          .source_points_lidar = cloud};
}

TEST(FixedLagEstimatorTest, AcceptsAdjacentCombinedImuUpdatesAndMarginalizesOldestState) {
  FixedLagEstimatorOptions options;
  options.maximum_states = 2U;
  options.maximum_lag_ns = 10 * kSecondNs;
  options.maximum_solver_iterations = 10U;
  options.maximum_solver_time_s = 1.0;
  options.maximum_translation_correction_m = 1.0;
  options.maximum_rotation_correction_rad = 0.5;
  FixedLagEstimator estimator(options, scanOptions(), {});

  const core::NavigationState seed(core::StateId(0U), core::TimeNs(0), {}, {}, {});
  estimator.initialize(seed, frame(0U, 0));
  ASSERT_TRUE(estimator.initialized());

  const FixedLagUpdate first =
      estimator.addSweep(frame(1U, kSecondNs), preintegration(0), predictedState(1U, kSecondNs));
  ASSERT_TRUE(first.accepted()) << first.reason;
  ASSERT_TRUE(first.optimized.has_value());
  EXPECT_EQ(first.active_states, 2U);
  EXPECT_EQ(first.imu_factors, 1U);
  EXPECT_EQ(first.marginalizations, 0U);
  EXPECT_TRUE(first.newly_finalized.empty());
  EXPECT_EQ(first.association_passes, 2U);
  EXPECT_FALSE(first.first_association.rows.empty());
  EXPECT_FALSE(first.registration.rows.empty());
  EXPECT_EQ(first.reassociated_rows, first.registration.rows.size());
  EXPECT_NEAR(first.optimized->odomFromImu().translation().x, 0.0, 1.0e-4);
  EXPECT_NEAR(first.optimized->odomFromImu().translation().y, 0.0, 1.0e-4);
  EXPECT_NEAR(first.optimized->odomFromImu().translation().z, 0.0, 1.0e-4);

  const FixedLagUpdate second = estimator.addSweep(
      frame(2U, 2 * kSecondNs), preintegration(kSecondNs), predictedState(2U, 2 * kSecondNs));
  ASSERT_TRUE(second.accepted()) << second.reason;
  ASSERT_TRUE(second.optimized.has_value());
  ASSERT_EQ(second.newly_finalized.size(), 1U);
  EXPECT_EQ(second.newly_finalized.front().id(), core::StateId(0U));
  EXPECT_EQ(second.marginalizations, 1U);
  EXPECT_GT(second.prior_rank, 0U);
  EXPECT_EQ(second.active_states, 2U);
  EXPECT_EQ(second.imu_factors, 1U);
  EXPECT_GT(second.finalized_map_points, 0U);
  EXPECT_EQ(second.association_passes, 2U);
  EXPECT_EQ(second.rejected_stale_rows, 0U);

  const lidar::PointCloud registration_map = estimator.registrationMapPointCloud();
  EXPECT_GT(registration_map.size(), second.finalized_map_points);

  const std::vector<core::NavigationState> active = estimator.activeStates();
  ASSERT_EQ(active.size(), 2U);
  EXPECT_EQ(active[0].id(), core::StateId(1U));
  EXPECT_EQ(active[1].id(), core::StateId(2U));
  EXPECT_NEAR(active[1].odomFromImu().translation().x, 0.0, 1.0e-4);
  EXPECT_NEAR(active[1].odomFromImu().translation().y, 0.0, 1.0e-4);
  EXPECT_NEAR(active[1].odomFromImu().translation().z, 0.0, 1.0e-4);
}

}  // namespace
}  // namespace meridian::local_rt::estimator
