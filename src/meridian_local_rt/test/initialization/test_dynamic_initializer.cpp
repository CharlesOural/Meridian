#include <gtest/gtest.h>

#include <Eigen/Core>
#include <chrono>
#include <cstdint>
#include <vector>

#include "meridian/local_rt/combined_preintegration.hpp"
#include "meridian/local_rt/imu_buffer.hpp"
#include "meridian/local_rt/initialization/dynamic_initializer.hpp"

namespace meridian::local_rt::initialization {
namespace {

constexpr double kGravity = 9.80665;
constexpr std::int64_t kImuPeriodNs = 5'000'000;
constexpr std::int64_t kSweepPeriodNs = 100'000'000;
constexpr std::int64_t kScanDurationNs = 10'000'000;
constexpr std::int64_t kFirstSweepNs = 1'000'000'000;

lidar::PointCloud asymmetricCloud() {
  lidar::PointCloud points;
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

core::LidarSweep sweep(std::uint64_t id, std::int64_t begin_ns,
                       const Eigen::Vector3d& source_shift = Eigen::Vector3d::Zero()) {
  const lidar::PointCloud cloud = asymmetricCloud();
  std::vector<core::LidarPoint> points;
  points.reserve(cloud.size());
  for (std::size_t index = 0U; index < cloud.size(); ++index) {
    const Eigen::Vector3d point = cloud[index] + source_shift;
    points.push_back(core::LidarPoint{
        .x = static_cast<float>(point.x()),
        .y = static_cast<float>(point.y()),
        .z = static_cast<float>(point.z()),
        .time_offset_ns = index % 2U == 0U ? 0 : kScanDurationNs,
        .source_index = static_cast<std::uint32_t>(index),
        .intensity = std::nullopt,
        .ring = std::nullopt,
    });
  }
  return core::LidarSweep(
      core::ObservationHeader(core::SensorId("lidar"), core::CalibrationId("calibration"),
                              core::MeasurementId(id), core::TimeNs(begin_ns), "lidar"),
      core::TimeNs(begin_ns), core::TimeNs(begin_ns + kScanDurationNs), std::move(points));
}

core::ImuSample imu(std::uint64_t id, std::int64_t time_ns, const core::Vec3d& gyroscope_bias) {
  return core::ImuSample(
      core::ObservationHeader(core::SensorId("imu"), core::CalibrationId("calibration"),
                              core::MeasurementId(id), core::TimeNs(time_ns), "imu"),
      gyroscope_bias, {.x = 0.0, .y = 0.0, .z = kGravity});
}

ImuModel imuModel() {
  ImuModel model;
  model.gravity_odom_m_s2 = {.x = 0.0, .y = 0.0, .z = -kGravity};
  model.accelerometer_covariance_density = Eigen::Matrix3d::Identity() * 1.0e-4;
  model.gyroscope_covariance_density = Eigen::Matrix3d::Identity() * 1.0e-6;
  model.integration_covariance_density = Eigen::Matrix3d::Identity() * 1.0e-8;
  model.accelerometer_bias_random_walk_covariance = Eigen::Matrix3d::Identity() * 1.0e-8;
  model.gyroscope_bias_random_walk_covariance = Eigen::Matrix3d::Identity() * 1.0e-10;
  return model;
}

DynamicInitializerOptions options() {
  return DynamicInitializerOptions{
      .target_sweeps = 20U,
      .maximum_support_ns = 5'000'000'000,
      .lidar_time_offset_to_imu_ns = 0,
      .minimum_range_m = 0.0,
      .maximum_range_m = 100.0,
      .gravity_m_s2 = kGravity,
      .gyroscope_finite_difference_step_rad_s = 1.0e-4,
      .minimum_singular_value_ratio = 1.0e-8,
      .maximum_condition_number = 1.0e8,
      .maximum_gyro_bias_correction_rad_s = 0.1,
      .maximum_gravity_magnitude_error_m_s2 = 0.5,
      .maximum_alignment_residual_rms = 10.0,
      .maximum_held_out_rotation_error_rad = 0.02,
      .maximum_held_out_translation_error_m = 0.03,
      .maximum_refinement_rotation_change_rad = 0.02,
      .maximum_refinement_translation_change_m = 0.03,
      .calibrated_bias_prior = {},
      .gyroscope_bias_prior_covariance = Eigen::Matrix3d::Identity() * 4.0e-4,
      .base_from_imu = {},
      .imu_from_lidar = core::Pose3d({.x = 0.12, .y = -0.04, .z = 0.08}, {}),
      .bootstrap =
          BootstrapOdometryOptions{
              .downsample_voxel_size_m = 0.01,
              .minimum_points = 20U,
              .maximum_accepted_rmse_m = 0.05,
              .maximum_accepted_condition_number = 1.0e8,
              .target =
                  lidar::VoxelTargetOptions{
                      .voxel_size_m = 0.25,
                      .retention_radius_m = 100.0,
                      .max_voxels = 512U,
                      .max_points_per_voxel = 8U,
                      .minimum_point_spacing_m = 0.01,
                      .max_neighbor_voxel_radius = 4U,
                  },
              .registration =
                  lidar::PointToPointRegistrationOptions{
                      .max_iterations = 30U,
                      .max_source_points = 256U,
                      .minimum_correspondences = 20U,
                      .max_correspondence_distance_m = 0.75,
                      .geman_mcclure_scale_m = 0.25,
                      .translation_convergence_m = 1.0e-8,
                      .rotation_convergence_rad = 1.0e-8,
                      .max_translation_step_m = 0.20,
                      .max_rotation_step_rad = 0.10,
                      .relative_rank_tolerance = 1.0e-10,
                  },
          },
  };
}

ImuBuffer populatedImuBuffer(const core::Vec3d& gyroscope_bias) {
  ImuBuffer buffer(
      ImuBufferConfig{.capacity = 1024U, .maximum_gap = std::chrono::milliseconds(20)});
  std::uint64_t id = 1U;
  for (std::int64_t time_ns = kFirstSweepNs - 2U * kImuPeriodNs;
       time_ns <= kFirstSweepNs + 20U * kSweepPeriodNs; time_ns += kImuPeriodNs) {
    EXPECT_TRUE(buffer.insert(imu(id++, time_ns, gyroscope_bias)).ok());
  }
  return buffer;
}

TEST(DynamicInitializer, AcceptsExactlyTwentySweepsAndReservesTheNewestTransition) {
  const core::Vec3d actual_gyroscope_bias{.x = 0.010, .y = -0.006, .z = 0.003};
  const ImuBuffer buffer = populatedImuBuffer(actual_gyroscope_bias);
  const GtsamCombinedPreintegrator preintegrator(imuModel());
  DynamicInitializer initializer(options());

  DynamicInitializationUpdate update;
  for (std::uint64_t index = 0U; index < 20U; ++index) {
    update = initializer.add(
        sweep(index + 1U, kFirstSweepNs + static_cast<std::int64_t>(index) * kSweepPeriodNs),
        buffer, preintegrator);
    if (index < 19U) {
      EXPECT_NE(update.status, core::InitializationStatus::kAccepted);
    }
  }

  ASSERT_EQ(update.status, core::InitializationStatus::kAccepted) << update.reason;
  ASSERT_TRUE(update.result.has_value());
  EXPECT_TRUE(initializer.accepted());
  EXPECT_EQ(update.quality.lidar_sweep_count, 20U);
  EXPECT_EQ(update.quality.fitted_transition_count, 18U);
  EXPECT_TRUE(update.quality.all_required_gates_passed);
  EXPECT_LT(*update.quality.held_out_rotation_error_rad, 1.0e-5);
  EXPECT_LT(*update.quality.held_out_translation_error_m, 1.0e-4);
  const core::NavigationState& seed = update.result->seedState();
  EXPECT_EQ(seed.time(), core::TimeNs(kFirstSweepNs + 19U * kSweepPeriodNs + kScanDurationNs));
  EXPECT_NEAR(seed.imuBias().gyroscopeRadS().x, actual_gyroscope_bias.x, 2.0e-5);
  EXPECT_NEAR(seed.imuBias().gyroscopeRadS().y, actual_gyroscope_bias.y, 2.0e-5);
  EXPECT_NEAR(seed.imuBias().gyroscopeRadS().z, actual_gyroscope_bias.z, 2.0e-5);
  EXPECT_LT(
      Eigen::Vector3d(seed.velocityOdomMS().x, seed.velocityOdomMS().y, seed.velocityOdomMS().z)
          .norm(),
      1.0e-3);
  EXPECT_LT(Eigen::Vector3d(seed.odomFromImu().translation().x, seed.odomFromImu().translation().y,
                            seed.odomFromImu().translation().z)
                .norm(),
            1.0e-8);
}

TEST(DynamicInitializer, RejectsAReservedTransitionThatDisagreesWithImuPropagation) {
  const core::Vec3d actual_gyroscope_bias{.x = 0.010, .y = -0.006, .z = 0.003};
  const ImuBuffer buffer = populatedImuBuffer(actual_gyroscope_bias);
  const GtsamCombinedPreintegrator preintegrator(imuModel());
  DynamicInitializer initializer(options());

  DynamicInitializationUpdate update;
  for (std::uint64_t index = 0U; index < 20U; ++index) {
    const Eigen::Vector3d shift =
        index == 19U ? Eigen::Vector3d(0.20, 0.0, 0.0) : Eigen::Vector3d::Zero();
    update = initializer.add(
        sweep(index + 1U, kFirstSweepNs + static_cast<std::int64_t>(index) * kSweepPeriodNs, shift),
        buffer, preintegrator);
  }

  EXPECT_EQ(update.status, core::InitializationStatus::kCollecting);
  EXPECT_FALSE(update.result.has_value());
  EXPECT_FALSE(initializer.accepted());
  EXPECT_NE(update.reason.find("reserved"), std::string::npos);
}

TEST(DynamicInitializer, RejectsGyroscopeBiasOutsideTheCalibratedPrior) {
  const core::Vec3d actual_gyroscope_bias{.x = 0.010, .y = -0.006, .z = 0.003};
  const ImuBuffer buffer = populatedImuBuffer(actual_gyroscope_bias);
  const GtsamCombinedPreintegrator preintegrator(imuModel());
  DynamicInitializerOptions configured = options();
  configured.gyroscope_bias_prior_covariance = Eigen::Matrix3d::Identity() * 1.0e-8;
  DynamicInitializer initializer(configured);

  DynamicInitializationUpdate update;
  for (std::uint64_t index = 0U; index < 20U; ++index) {
    update = initializer.add(
        sweep(index + 1U, kFirstSweepNs + static_cast<std::int64_t>(index) * kSweepPeriodNs),
        buffer, preintegrator);
  }

  EXPECT_EQ(update.status, core::InitializationStatus::kCollecting);
  EXPECT_FALSE(update.result.has_value());
  EXPECT_NE(update.reason.find("calibrated prior"), std::string::npos);
}

}  // namespace
}  // namespace meridian::local_rt::initialization
