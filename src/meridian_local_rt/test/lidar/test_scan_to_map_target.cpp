#include <gtest/gtest.h>

#include <sophus/se3.hpp>

#include "meridian/local_rt/lidar/scan_to_map.hpp"

namespace meridian::local_rt::lidar {
namespace {

PointCloud fullRankCloud() {
  return {
      Point3d{-4.0, 2.0, 1.0},  Point3d{-3.0, -2.0, -1.0}, Point3d{-2.0, 1.0, 3.0},
      Point3d{-1.0, 3.0, -2.0}, Point3d{0.0, -3.0, 2.0},   Point3d{1.0, 2.0, 4.0},
      Point3d{2.0, -1.0, -3.0}, Point3d{2.0, 3.0, -4.0},   Point3d{3.0, 4.0, 1.0},
      Point3d{4.0, -2.0, 3.0},
  };
}

ScanToMapOptions testOptions() {
  ScanToMapOptions options;
  options.maximum_active_owners = 2U;
  options.maximum_factor_rows = 32U;
  options.minimum_correspondences = 6U;
  options.maximum_icp_iterations = 4U;
  options.maximum_backtracking_steps = 4U;
  options.maximum_correspondence_distance_m = 0.75;
  // One exact Gauss-Newton translation update is enough for this synthetic
  // correspondence set; stop before asking for a strict decrease at zero cost.
  options.translation_convergence_m = 0.2;
  options.rotation_convergence_rad = 0.2;
  options.maximum_translation_step_m = 0.5;
  options.maximum_rotation_step_rad = 0.2;
  options.maximum_prediction_correction_m = 1.0;
  options.maximum_prediction_correction_rad = 0.5;
  options.active_owner_index = VoxelTargetOptions{
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

Sophus::SE3d translatedPrediction() {
  return Sophus::SE3d(Eigen::Quaterniond::Identity(), Point3d{0.1, -0.05, 0.04});
}

TEST(ScanToMapTargetTest, KeepsOwnerLocalRowsActiveThenMovesGeometryToFinalizedBase) {
  const PointCloud cloud = fullRankCloud();
  ScanToMapTarget target(testOptions(), Sophus::SE3d());
  target.admit(ScanFrame{.state_id = core::StateId(1U),
                         .time = core::TimeNs(0),
                         .target_points_lidar = cloud,
                         .source_points_lidar = cloud},
               Sophus::SE3d());

  EXPECT_TRUE(target.hasActiveOwner(core::StateId(1U)));
  EXPECT_EQ(target.activeOwnerCount(), 1U);
  EXPECT_EQ(target.finalizedPointCount(), 0U);

  const ScanToMapResult active = target.registerScan(cloud, translatedPrediction());
  ASSERT_TRUE(active.accepted()) << toString(active.status);
  ASSERT_EQ(active.rows.size(), cloud.size());
  EXPECT_EQ(active.active_rows, cloud.size());
  EXPECT_EQ(active.finalized_rows, 0U);
  for (const ScanToMapRow& row : active.rows) {
    ASSERT_TRUE(row.active_target_state.has_value());
    EXPECT_EQ(*row.active_target_state, core::StateId(1U));
  }
  EXPECT_LT(active.odom_from_imu.translation().norm(), 2.0e-5);

  const FinalizedTargetStats finalized =
      target.finalize(core::StateId(1U), Sophus::SE3d(), Point3d::Zero());
  EXPECT_FALSE(target.hasActiveOwner(core::StateId(1U)));
  EXPECT_EQ(target.activeOwnerCount(), 0U);
  EXPECT_EQ(finalized.points, cloud.size());
  EXPECT_EQ(target.finalizedPointCount(), cloud.size());

  const ScanToMapResult fixed = target.registerScan(cloud, translatedPrediction());
  ASSERT_TRUE(fixed.accepted()) << toString(fixed.status);
  ASSERT_EQ(fixed.rows.size(), cloud.size());
  EXPECT_EQ(fixed.active_rows, 0U);
  EXPECT_EQ(fixed.finalized_rows, cloud.size());
  for (const ScanToMapRow& row : fixed.rows) {
    EXPECT_FALSE(row.active_target_state.has_value());
  }
  EXPECT_LT(fixed.odom_from_imu.translation().norm(), 2.0e-5);
}

TEST(ScanToMapTargetTest, ExactInitialPoseIsAConvergedRegistration) {
  const PointCloud cloud = fullRankCloud();
  ScanToMapTarget target(testOptions(), Sophus::SE3d());
  target.admit(ScanFrame{.state_id = core::StateId(1U),
                         .time = core::TimeNs(0),
                         .target_points_lidar = cloud,
                         .source_points_lidar = cloud},
               Sophus::SE3d());

  const ScanToMapResult result = target.registerScan(cloud, Sophus::SE3d());
  ASSERT_TRUE(result.accepted()) << toString(result.status);
  EXPECT_EQ(result.iterations, 0U);
  EXPECT_LT(result.rmse_m, 1.0e-12);
  EXPECT_LT(result.odom_from_imu.translation().norm(), 1.0e-12);
}

TEST(ScanToMapTargetTest, AssociatesOnceWithoutChangingPoseAndPreservesOwnerLink) {
  ScanToMapOptions options = testOptions();
  options.maximum_factor_rows = 6U;
  const PointCloud cloud = fullRankCloud();
  ScanToMapTarget target(options, Sophus::SE3d());
  target.admit(ScanFrame{.state_id = core::StateId(1U),
                         .time = core::TimeNs(0),
                         .target_points_lidar = cloud,
                         .source_points_lidar = cloud},
               Sophus::SE3d());

  const Sophus::SE3d supplied = translatedPrediction();
  const ScanToMapResult result = target.associateScan(cloud, supplied);
  ASSERT_TRUE(result.accepted()) << toString(result.status);
  EXPECT_EQ(result.iterations, 0U);
  EXPECT_EQ(result.correspondences_before_cap, cloud.size());
  EXPECT_EQ(result.rows.size(), options.maximum_factor_rows);
  EXPECT_TRUE(result.odom_from_imu.matrix().isApprox(supplied.matrix(), 1.0e-15));
  EXPECT_GT(result.timing.active_queries, 0U);
  EXPECT_GT(result.timing.active_voxel_probes, 0U);
  for (const ScanToMapRow& row : result.rows) {
    ASSERT_TRUE(row.active_target_state.has_value());
    EXPECT_EQ(*row.active_target_state, core::StateId(1U));
  }
}

TEST(ScanToMapTargetTest, CompositeQueryFiltersToNearestConfiguredOwner) {
  ScanToMapOptions options = testOptions();
  options.maximum_active_owners = 1U;
  const PointCloud cloud = fullRankCloud();
  ScanToMapTarget target(options, Sophus::SE3d());
  target.admit(ScanFrame{.state_id = core::StateId(1U),
                         .time = core::TimeNs(0),
                         .target_points_lidar = cloud,
                         .source_points_lidar = cloud},
               Sophus::SE3d(Eigen::Quaterniond::Identity(), Point3d{20.0, 0.0, 0.0}));
  const Sophus::SE3d owner_two_pose(Eigen::Quaterniond::Identity(), Point3d{2.0, 0.0, 0.0});
  target.admit(ScanFrame{.state_id = core::StateId(2U),
                         .time = core::TimeNs(1),
                         .target_points_lidar = cloud,
                         .source_points_lidar = cloud},
               owner_two_pose);

  const ScanToMapResult result = target.associateScan(cloud, owner_two_pose);
  ASSERT_TRUE(result.accepted()) << toString(result.status);
  EXPECT_EQ(result.selected_active_owners, 1U);
  for (const ScanToMapRow& row : result.rows) {
    ASSERT_TRUE(row.active_target_state.has_value());
    EXPECT_EQ(*row.active_target_state, core::StateId(2U));
  }
}

TEST(ScanToMapTargetTest, RegistrationMapSnapshotProjectsLiveAndFinalizedGeometryInOdom) {
  const PointCloud cloud = fullRankCloud();
  ScanToMapTarget target(testOptions(), Sophus::SE3d());
  target.admit(ScanFrame{.state_id = core::StateId(1U),
                         .time = core::TimeNs(0),
                         .target_points_lidar = cloud,
                         .source_points_lidar = cloud},
               Sophus::SE3d());
  const Sophus::SE3d shifted(Eigen::Quaterniond::Identity(), Point3d{1.0, 2.0, 3.0});
  target.updateOwnerPose(core::StateId(1U), shifted);

  const PointCloud live_snapshot = target.registrationMapPointCloud();
  ASSERT_EQ(live_snapshot.size(), cloud.size());
  const PointCloud repeated = target.registrationMapPointCloud();
  ASSERT_EQ(repeated.size(), live_snapshot.size());
  for (std::size_t index = 0U; index < repeated.size(); ++index) {
    EXPECT_TRUE(repeated[index].isApprox(live_snapshot[index], 1.0e-15));
  }

  static_cast<void>(target.finalize(core::StateId(1U), shifted, shifted.translation()));
  const PointCloud finalized_snapshot = target.registrationMapPointCloud();
  ASSERT_EQ(finalized_snapshot.size(), live_snapshot.size());
  for (std::size_t index = 0U; index < finalized_snapshot.size(); ++index) {
    EXPECT_TRUE(finalized_snapshot[index].isApprox(live_snapshot[index], 1.0e-15));
  }
}

TEST(ScanToMapTargetTest, CompositePrunesLiveGeometryAroundNewestLocalCentre) {
  ScanToMapOptions options = testOptions();
  options.active_owner_index.retention_radius_m = 6.0;
  options.finalized_base.retention_radius_m = 6.0;
  const PointCloud cloud = fullRankCloud();
  ScanToMapTarget target(options, Sophus::SE3d());
  target.admit(ScanFrame{.state_id = core::StateId(1U),
                         .time = core::TimeNs(0),
                         .target_points_lidar = cloud,
                         .source_points_lidar = cloud},
               Sophus::SE3d());
  const Sophus::SE3d newest(Eigen::Quaterniond::Identity(), Point3d{5.0, 0.0, 0.0});
  target.admit(ScanFrame{.state_id = core::StateId(2U),
                         .time = core::TimeNs(1),
                         .target_points_lidar = cloud,
                         .source_points_lidar = cloud},
               newest);

  const PointCloud snapshot = target.registrationMapPointCloud();
  ASSERT_FALSE(snapshot.empty());
  EXPECT_LT(snapshot.size(), 2U * cloud.size());
  for (const Point3d& point : snapshot) {
    EXPECT_LE((point - newest.translation()).norm(),
              options.active_owner_index.retention_radius_m + 1.0e-12);
  }
}

TEST(ScanToMapTargetTest, ParallelAssociationIsExactAndSourceOrderDeterministic) {
  ScanToMapOptions options = testOptions();
  options.maximum_factor_rows = 1'024U;
  options.minimum_correspondences = 100U;
  options.active_owner_index.max_voxels = 2'000U;
  options.finalized_base.max_voxels = 2'000U;
  PointCloud cloud;
  cloud.reserve(1'024U);
  for (std::size_t x = 0U; x < 16U; ++x) {
    for (std::size_t y = 0U; y < 8U; ++y) {
      for (std::size_t z = 0U; z < 8U; ++z) {
        cloud.push_back(
            0.4 * Point3d{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)});
      }
    }
  }
  ScanToMapTarget target(options, Sophus::SE3d());
  target.admit(ScanFrame{.state_id = core::StateId(7U),
                         .time = core::TimeNs(0),
                         .target_points_lidar = cloud,
                         .source_points_lidar = cloud},
               Sophus::SE3d());

  const ScanToMapResult first = target.associateScan(cloud, Sophus::SE3d());
  const ScanToMapResult second = target.associateScan(cloud, Sophus::SE3d());
  ASSERT_TRUE(first.accepted()) << toString(first.status);
  ASSERT_TRUE(second.accepted()) << toString(second.status);
  ASSERT_EQ(first.rows.size(), cloud.size());
  ASSERT_EQ(second.rows.size(), first.rows.size());
  EXPECT_EQ(first.timing.active_voxel_probes, second.timing.active_voxel_probes);
  EXPECT_EQ(first.timing.active_candidate_points, second.timing.active_candidate_points);
  for (std::size_t index = 0U; index < first.rows.size(); ++index) {
    EXPECT_TRUE(first.rows[index].source_lidar.isApprox(second.rows[index].source_lidar, 0.0));
    EXPECT_TRUE(first.rows[index].target.isApprox(second.rows[index].target, 0.0));
    EXPECT_EQ(first.rows[index].active_target_state, second.rows[index].active_target_state);
    EXPECT_DOUBLE_EQ(first.rows[index].association_distance_m,
                     second.rows[index].association_distance_m);
  }
}

}  // namespace
}  // namespace meridian::local_rt::lidar
