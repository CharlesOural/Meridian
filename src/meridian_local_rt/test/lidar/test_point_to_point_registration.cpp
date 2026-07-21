#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <sophus/se3.hpp>
#include <stdexcept>

#include "meridian/local_rt/lidar/point_to_point_registration.hpp"

namespace meridian::local_rt::lidar {
namespace {

PointCloud makeAsymmetricCloud() {
  PointCloud points;
  for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 4; ++y) {
      for (int z = 0; z < 3; ++z) {
        points.emplace_back(0.70 * x + 0.03 * y * y, 0.60 * y + 0.05 * z * x,
                            0.50 * z + 0.02 * x * y);
      }
    }
  }
  return points;
}

VoxelTargetOptions targetOptions() {
  return VoxelTargetOptions{
      .voxel_size_m = 0.25,
      .retention_radius_m = 100.0,
      .max_voxels = 512U,
      .max_points_per_voxel = 8U,
      .minimum_point_spacing_m = 0.0,
      .max_neighbor_voxel_radius = 8U,
  };
}

PointToPointRegistrationOptions registrationOptions() {
  return PointToPointRegistrationOptions{
      .max_iterations = 30U,
      .max_source_points = 256U,
      .minimum_correspondences = 20U,
      .max_correspondence_distance_m = 0.75,
      .geman_mcclure_scale_m = 0.25,
      .translation_convergence_m = 1.0e-10,
      .rotation_convergence_rad = 1.0e-10,
      .max_translation_step_m = 0.20,
      .max_rotation_step_rad = 0.10,
      .relative_rank_tolerance = 1.0e-10,
  };
}

PointCloud transformCloud(const PointCloud& source, const Sophus::SE3d& transform) {
  PointCloud target;
  target.reserve(source.size());
  for (const Point3d& point : source) {
    target.push_back(transform * point);
  }
  return target;
}

TEST(PointToPointRegistrationTest, RejectsUnboundedOrNonphysicalOptions) {
  auto options = registrationOptions();
  options.max_iterations = 0U;
  EXPECT_THROW(static_cast<void>(DirectPointToPointRegistration{options}), std::invalid_argument);
  options = registrationOptions();
  options.max_source_points = 0U;
  EXPECT_THROW(static_cast<void>(DirectPointToPointRegistration{options}), std::invalid_argument);
  options = registrationOptions();
  options.geman_mcclure_scale_m = 0.0;
  EXPECT_THROW(static_cast<void>(DirectPointToPointRegistration{options}), std::invalid_argument);
  options = registrationOptions();
  options.relative_rank_tolerance = 1.0;
  EXPECT_THROW(static_cast<void>(DirectPointToPointRegistration{options}), std::invalid_argument);
}

TEST(PointToPointRegistrationTest, RecoversKnownRigidTransformAndReportsInformation) {
  const PointCloud source = makeAsymmetricCloud();
  Vector6d truth_tangent;
  truth_tangent << 0.30, -0.20, 0.15, 0.04, -0.03, 0.05;
  const Sophus::SE3d truth = Sophus::SE3d::exp(truth_tangent);
  const PointCloud target_points = transformCloud(source, truth);
  BoundedVoxelTarget target(targetOptions());
  const auto update = target.updateTargetFrame(target_points, truth.translation());
  ASSERT_EQ(update.inserted_points, target_points.size());

  Vector6d initial_error;
  initial_error << 0.08, -0.05, 0.04, 0.02, -0.015, 0.01;
  const Sophus::SE3d initial = Sophus::SE3d::exp(initial_error) * truth;
  const DirectPointToPointRegistration registration(registrationOptions());
  const auto result = registration.align(source, target, initial);

  ASSERT_EQ(result.status, RegistrationStatus::kConverged);
  EXPECT_TRUE(result.hasUsableEstimate());
  EXPECT_LT((truth.inverse() * result.T_target_source).log().norm(), 1.0e-7);
  EXPECT_EQ(result.quality.correspondences, source.size());
  EXPECT_EQ(result.quality.observable_rank, 6U);
  EXPECT_TRUE(result.hessian.allFinite());
  EXPECT_GT(result.quality.hessian_eigenvalues.minCoeff(), 0.0);
  EXPECT_TRUE(std::isfinite(result.quality.hessian_condition_number));
  ASSERT_TRUE(result.covariance.has_value());
  EXPECT_TRUE(result.covariance->allFinite());
  EXPECT_LT(result.quality.point_rmse_m, 1.0e-8);
}

TEST(PointToPointRegistrationTest, DetectsDegenerateCollinearGeometry) {
  PointCloud line;
  for (int index = 0; index < 20; ++index) {
    line.emplace_back(0.2 * index, 0.0, 0.0);
  }
  BoundedVoxelTarget target(targetOptions());
  static_cast<void>(target.updateTargetFrame(line, Point3d::Zero()));
  auto options = registrationOptions();
  options.minimum_correspondences = 10U;
  const DirectPointToPointRegistration registration(options);
  const auto result = registration.align(line, target, Sophus::SE3d{});

  EXPECT_EQ(result.status, RegistrationStatus::kDegenerateGeometry);
  EXPECT_FALSE(result.hasUsableEstimate());
  EXPECT_LT(result.quality.observable_rank, 6U);
  EXPECT_FALSE(result.covariance.has_value());
}

TEST(PointToPointRegistrationTest, BoundsEveryStepAndIteration) {
  const PointCloud source = makeAsymmetricCloud();
  Vector6d truth_tangent;
  truth_tangent << 0.30, -0.20, 0.15, 0.04, -0.03, 0.05;
  const Sophus::SE3d truth = Sophus::SE3d::exp(truth_tangent);
  BoundedVoxelTarget target(targetOptions());
  static_cast<void>(target.updateTargetFrame(transformCloud(source, truth), truth.translation()));

  auto options = registrationOptions();
  options.max_iterations = 1U;
  options.max_translation_step_m = 0.01;
  options.max_rotation_step_rad = 0.005;
  const DirectPointToPointRegistration registration(options);
  Vector6d initial_error;
  initial_error << 0.25, -0.12, 0.10, 0.08, -0.04, 0.03;
  const auto result = registration.align(source, target, Sophus::SE3d::exp(initial_error) * truth);

  EXPECT_EQ(result.status, RegistrationStatus::kIterationLimit);
  EXPECT_EQ(result.quality.iterations, 1U);
  EXPECT_TRUE(result.quality.step_was_limited);
  EXPECT_LE(result.quality.final_translation_step_m, options.max_translation_step_m + 1.0e-14);
  EXPECT_LE(result.quality.final_rotation_step_rad, options.max_rotation_step_rad + 1.0e-14);
}

TEST(PointToPointRegistrationTest, ReportsEmptyCapacityAndSearchFailuresWithoutGuessing) {
  const PointCloud source = makeAsymmetricCloud();
  const DirectPointToPointRegistration registration(registrationOptions());
  BoundedVoxelTarget empty_target(targetOptions());
  EXPECT_EQ(registration.align(source, empty_target, Sophus::SE3d{}).status,
            RegistrationStatus::kEmptyTarget);

  auto small_source_options = registrationOptions();
  small_source_options.max_source_points = 20U;
  small_source_options.minimum_correspondences = 20U;
  const DirectPointToPointRegistration source_bounded(small_source_options);
  BoundedVoxelTarget populated(targetOptions());
  static_cast<void>(populated.updateTargetFrame(source, Point3d::Zero()));
  EXPECT_EQ(source_bounded.align(source, populated, Sophus::SE3d{}).status,
            RegistrationStatus::kSourcePointLimitExceeded);

  auto short_search_options = targetOptions();
  short_search_options.max_neighbor_voxel_radius = 1U;
  BoundedVoxelTarget short_search(short_search_options);
  static_cast<void>(short_search.updateTargetFrame(source, Point3d::Zero()));
  EXPECT_EQ(registration.align(source, short_search, Sophus::SE3d{}).status,
            RegistrationStatus::kIncompatibleTarget);
}

}  // namespace
}  // namespace meridian::local_rt::lidar
