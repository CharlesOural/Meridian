#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "meridian/local_rt/initialization/bootstrap_odometry.hpp"

namespace meridian::local_rt::initialization {
namespace {

lidar::PointCloud makeAsymmetricCloud(double scale = 1.0) {
  lidar::PointCloud points;
  for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 4; ++y) {
      for (int z = 0; z < 3; ++z) {
        points.emplace_back(scale * (0.70 * x + 0.03 * y * y), scale * (0.60 * y + 0.05 * z * x),
                            scale * (0.50 * z + 0.02 * x * y));
      }
    }
  }
  return points;
}

lidar::PointCloud cloudInLidarFrame(const lidar::PointCloud& odom_points,
                                    const Sophus::SE3d& odom_from_lidar) {
  lidar::PointCloud lidar_points;
  lidar_points.reserve(odom_points.size());
  for (const lidar::Point3d& point : odom_points) {
    lidar_points.push_back(odom_from_lidar.inverse() * point);
  }
  return lidar_points;
}

lidar::PointCloud translatedCloud(const lidar::PointCloud& points,
                                  const lidar::Point3d& translation) {
  lidar::PointCloud translated;
  translated.reserve(points.size());
  for (const lidar::Point3d& point : points) {
    translated.push_back(point + translation);
  }
  return translated;
}

core::LidarSweep makeSweep(std::uint64_t id, std::int64_t time_ns) {
  const core::TimeNs time{time_ns};
  return core::LidarSweep(
      core::ObservationHeader(core::SensorId{"lidar"}, core::CalibrationId{"lidar-calibration"},
                              core::MeasurementId{id}, time, "lidar"),
      time, time, std::vector<core::LidarPoint>{core::LidarPoint{}});
}

BootstrapOdometryOptions options() {
  return BootstrapOdometryOptions{
      .downsample_voxel_size_m = 0.01,
      .minimum_points = 20U,
      .maximum_accepted_rmse_m = 0.10,
      .maximum_accepted_condition_number = 1.0e12,
      .target =
          lidar::VoxelTargetOptions{
              .voxel_size_m = 0.25,
              .retention_radius_m = 100.0,
              .max_voxels = 512U,
              .max_points_per_voxel = 8U,
              .minimum_point_spacing_m = 0.0,
              .max_neighbor_voxel_radius = 4U,
          },
      .registration =
          lidar::PointToPointRegistrationOptions{
              .max_iterations = 30U,
              .max_source_points = 256U,
              .minimum_correspondences = 20U,
              .max_correspondence_distance_m = 0.75,
              .geman_mcclure_scale_m = 0.25,
              .translation_convergence_m = 1.0e-9,
              .rotation_convergence_rad = 1.0e-9,
              .max_translation_step_m = 0.20,
              .max_rotation_step_rad = 0.10,
              .relative_rank_tolerance = 1.0e-10,
          },
  };
}

void expectPoseNear(const Sophus::SE3d& actual, const Sophus::SE3d& expected,
                    double tolerance = 1.0e-7) {
  EXPECT_LT((expected.inverse() * actual).log().norm(), tolerance);
}

TEST(BootstrapOdometryTest, RejectsIncompatibleLayerBounds) {
  auto invalid = options();
  invalid.minimum_points = invalid.registration.max_source_points + 1U;
  EXPECT_THROW(static_cast<void>(BootstrapOdometry{invalid}), std::invalid_argument);

  invalid = options();
  invalid.minimum_points = invalid.registration.minimum_correspondences - 1U;
  EXPECT_THROW(static_cast<void>(BootstrapOdometry{invalid}), std::invalid_argument);

  invalid = options();
  invalid.target.max_voxels = 4U;
  invalid.target.max_points_per_voxel = 4U;
  EXPECT_THROW(static_cast<void>(BootstrapOdometry{invalid}), std::invalid_argument);
}

TEST(BootstrapOdometryTest, CommitsOnlyAnAdmissibleAnchorWithoutInventingRegistrationEvidence) {
  auto configured = options();
  configured.downsample_voxel_size_m = 0.001;
  configured.target.voxel_size_m = 0.02;
  configured.target.retention_radius_m = 0.18;
  configured.target.max_neighbor_voxel_radius = 38U;
  BootstrapOdometry odometry(configured);

  const lidar::PointCloud near_points = makeAsymmetricCloud(0.04);
  const lidar::PointCloud outside_points =
      translatedCloud(near_points, lidar::Point3d{2.0, 0.0, 0.0});
  const BootstrapFrame rejected = odometry.add(makeSweep(1U, 0), outside_points);

  EXPECT_FALSE(rejected.accepted);
  EXPECT_EQ(rejected.outcome, BootstrapFrameOutcome::kRejectedAnchorAdmission);
  EXPECT_FALSE(rejected.registration_status.has_value());
  ASSERT_TRUE(rejected.target_update.has_value());
  EXPECT_EQ(rejected.target_update->inserted_points, 0U);
  EXPECT_EQ(rejected.target_update->rejected_retention_points, outside_points.size());

  const BootstrapFrame anchor = odometry.add(makeSweep(2U, 1'000'000'000), near_points);
  EXPECT_TRUE(anchor.accepted);
  EXPECT_EQ(anchor.outcome, BootstrapFrameOutcome::kAcceptedAnchor);
  EXPECT_FALSE(anchor.registration_status.has_value());
  ASSERT_TRUE(anchor.target_update.has_value());
  EXPECT_GE(anchor.target_update->inserted_points, configured.minimum_points);
  EXPECT_EQ(anchor.quality.correspondences, 0U);
  EXPECT_EQ(anchor.quality.observable_rank, 0U);
  EXPECT_DOUBLE_EQ(anchor.quality.hessian_condition_number, 0.0);
  expectPoseNear(anchor.initial_odom_from_lidar, Sophus::SE3d{});
  expectPoseNear(anchor.odom_from_lidar, Sophus::SE3d{});
}

TEST(BootstrapOdometryTest, TimeScalesTheMotionGuessAndIgnoresRejectedFrames) {
  BootstrapOdometry odometry(options());
  const lidar::PointCloud odom_points = makeAsymmetricCloud();
  const BootstrapFrame anchor = odometry.add(makeSweep(1U, 0), odom_points);
  ASSERT_TRUE(anchor.accepted);

  lidar::Vector6d motion;
  motion << 0.10, -0.05, 0.03, 0.015, -0.01, 0.02;
  const Sophus::SE3d second_truth = Sophus::SE3d::exp(motion);
  const BootstrapFrame second =
      odometry.add(makeSweep(2U, 1'000'000'000), cloudInLidarFrame(odom_points, second_truth));
  ASSERT_TRUE(second.accepted);
  ASSERT_EQ(second.outcome, BootstrapFrameOutcome::kAcceptedRegistration);
  expectPoseNear(second.odom_from_lidar, second_truth, 1.0e-6);

  const Sophus::SE3d one_step_prediction =
      second.odom_from_lidar * Sophus::SE3d::exp(second.odom_from_lidar.log());
  const lidar::PointCloud too_few_points(odom_points.begin(), odom_points.begin() + 3);
  const BootstrapFrame rejected = odometry.add(makeSweep(3U, 2'000'000'000), too_few_points);
  EXPECT_FALSE(rejected.accepted);
  EXPECT_EQ(rejected.outcome, BootstrapFrameOutcome::kRejectedPointCount);
  expectPoseNear(rejected.initial_odom_from_lidar, one_step_prediction);

  const Sophus::SE3d two_step_prediction =
      second.odom_from_lidar * Sophus::SE3d::exp(2.0 * second.odom_from_lidar.log());
  const Sophus::SE3d third_truth = second_truth * Sophus::SE3d::exp(2.0 * motion);
  const BootstrapFrame third =
      odometry.add(makeSweep(4U, 3'000'000'000), cloudInLidarFrame(odom_points, third_truth));
  EXPECT_TRUE(third.accepted);
  EXPECT_EQ(third.outcome, BootstrapFrameOutcome::kAcceptedRegistration);
  expectPoseNear(third.initial_odom_from_lidar, two_step_prediction);
  expectPoseNear(third.odom_from_lidar, third_truth, 1.0e-6);
}

TEST(BootstrapOdometryTest, RegistrationRejectionDoesNotAdvanceState) {
  BootstrapOdometry odometry(options());
  const lidar::PointCloud points = makeAsymmetricCloud();
  ASSERT_TRUE(odometry.add(makeSweep(1U, 0), points).accepted);

  const lidar::PointCloud distant = translatedCloud(points, lidar::Point3d{20.0, 0.0, 0.0});
  const BootstrapFrame rejected = odometry.add(makeSweep(2U, 2'000'000'000), distant);
  EXPECT_FALSE(rejected.accepted);
  EXPECT_EQ(rejected.outcome, BootstrapFrameOutcome::kRejectedRegistration);
  ASSERT_TRUE(rejected.registration_status.has_value());
  EXPECT_EQ(*rejected.registration_status, lidar::RegistrationStatus::kInsufficientCorrespondences);
  EXPECT_FALSE(rejected.target_update.has_value());

  const BootstrapFrame valid = odometry.add(makeSweep(3U, 1'000'000'000), points);
  EXPECT_TRUE(valid.accepted);
  EXPECT_EQ(valid.outcome, BootstrapFrameOutcome::kAcceptedRegistration);
  expectPoseNear(valid.initial_odom_from_lidar, Sophus::SE3d{});
  expectPoseNear(valid.odom_from_lidar, Sophus::SE3d{});
}

TEST(BootstrapOdometryTest, FailedTargetMaintenanceIsTransactional) {
  auto configured = options();
  configured.downsample_voxel_size_m = 0.001;
  configured.target.voxel_size_m = 0.02;
  configured.target.retention_radius_m = 0.18;
  configured.target.max_neighbor_voxel_radius = 4U;
  configured.registration.max_correspondence_distance_m = 0.08;
  configured.registration.geman_mcclure_scale_m = 0.05;
  BootstrapOdometry odometry(configured);
  const lidar::PointCloud odom_points = makeAsymmetricCloud(0.04);
  ASSERT_TRUE(odometry.add(makeSweep(1U, 0), odom_points).accepted);

  lidar::Vector6d small_motion = lidar::Vector6d::Zero();
  small_motion.x() = 0.015;
  const Sophus::SE3d second_truth = Sophus::SE3d::exp(small_motion);
  const BootstrapFrame second =
      odometry.add(makeSweep(2U, 1'000'000'000), cloudInLidarFrame(odom_points, second_truth));
  ASSERT_TRUE(second.accepted);

  const Sophus::SE3d far_prediction =
      second.odom_from_lidar * Sophus::SE3d::exp(25.0 * second.odom_from_lidar.log());
  const BootstrapFrame target_rejected =
      odometry.add(makeSweep(3U, 26'000'000'000), cloudInLidarFrame(odom_points, far_prediction));
  EXPECT_FALSE(target_rejected.accepted);
  EXPECT_EQ(target_rejected.outcome, BootstrapFrameOutcome::kRejectedTargetAdmission);
  ASSERT_TRUE(target_rejected.registration_status.has_value());
  ASSERT_TRUE(target_rejected.target_update.has_value());
  EXPECT_GE(target_rejected.target_update->pruned_points, configured.minimum_points);
  EXPECT_GE(target_rejected.target_update->rejected_retention_points, configured.minimum_points);

  const Sophus::SE3d next_prediction =
      second.odom_from_lidar * Sophus::SE3d::exp(second.odom_from_lidar.log());
  const BootstrapFrame next =
      odometry.add(makeSweep(4U, 2'000'000'000), cloudInLidarFrame(odom_points, next_prediction));
  EXPECT_TRUE(next.accepted);
  EXPECT_EQ(next.outcome, BootstrapFrameOutcome::kAcceptedRegistration);
  expectPoseNear(next.initial_odom_from_lidar, next_prediction);
}

TEST(BootstrapOdometryTest, RejectsTimestampAndSourceBoundsBeforeMutatingState) {
  auto configured = options();
  configured.registration.max_source_points = 32U;
  BootstrapOdometry odometry(configured);
  const lidar::PointCloud points = makeAsymmetricCloud();

  const BootstrapFrame oversized = odometry.add(makeSweep(1U, 0), points);
  EXPECT_FALSE(oversized.accepted);
  EXPECT_EQ(oversized.outcome, BootstrapFrameOutcome::kRejectedSourcePointLimit);
  EXPECT_FALSE(oversized.target_update.has_value());

  const lidar::PointCloud bounded(points.begin(), points.begin() + 30);
  ASSERT_TRUE(odometry.add(makeSweep(2U, 1'000'000'000), bounded).accepted);
  const BootstrapFrame duplicate_time = odometry.add(makeSweep(3U, 1'000'000'000), bounded);
  EXPECT_FALSE(duplicate_time.accepted);
  EXPECT_EQ(duplicate_time.outcome, BootstrapFrameOutcome::kRejectedTimestamp);
  EXPECT_FALSE(duplicate_time.registration_status.has_value());
  EXPECT_FALSE(duplicate_time.target_update.has_value());

  odometry.reset();
  const BootstrapFrame after_reset = odometry.add(makeSweep(4U, 1'000'000'000), bounded);
  EXPECT_TRUE(after_reset.accepted);
  EXPECT_EQ(after_reset.outcome, BootstrapFrameOutcome::kAcceptedAnchor);
}

}  // namespace
}  // namespace meridian::local_rt::initialization
