#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "meridian/local_rt/initialization/static_initializer.hpp"

namespace meridian::local_rt::initialization {
namespace {

core::ImuSample sample(std::uint64_t id, std::int64_t time_ns, core::Vec3d gyro,
                       core::Vec3d accel) {
  return core::ImuSample(
      core::ObservationHeader(core::SensorId("imu"), core::CalibrationId("calibration"),
                              core::MeasurementId(id), core::TimeNs(time_ns), "imu"),
      gyro, accel);
}

StaticInitializerOptions options() {
  return StaticInitializerOptions{
      .window_duration_ns = 2'000'000'000,
      .block_duration_ns = 50'000'000,
      .minimum_samples = 300U,
      .minimum_blocks = 30U,
      .maximum_sample_gap_ns = 10'000'000,
      .gravity_m_s2 = 9.80665,
      .gyroscope_saturation_rad_s = 8.0,
      .accelerometer_saturation_m_s2 = 80.0,
      .maximum_mean_angular_rate_rad_s = 0.1,
      .maximum_block_angular_dispersion_rad_s = 0.01,
      .maximum_specific_force_norm_error_m_s2 = 0.3,
      .maximum_block_direction_dispersion_rad = 0.02,
      .calibrated_bias_prior = core::ImuBias({}, core::Vec3d{.x = 0.1, .y = -0.05, .z = 0.02}),
      .base_from_imu = {},
  };
}

Eigen::Quaterniond eigen(const core::Quaterniond& quaternion) {
  return {quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()};
}

TEST(StaticInitializer, AcceptsConfiguredZeroMotionAndKeepsAccelBiasAtPrior) {
  StaticInitializer initializer(options());
  StaticInitializationUpdate update;
  constexpr std::int64_t kPeriodNs = 5'000'000;
  for (std::uint64_t index = 0; index <= 420U; ++index) {
    const double vibration = index % 2U == 0U ? 0.0005 : -0.0005;
    update = initializer.add(sample(index + 1U, static_cast<std::int64_t>(index) * kPeriodNs,
                                    {.x = 0.01 + vibration, .y = -0.02, .z = 0.005},
                                    {.x = 0.1, .y = -0.05, .z = 9.82665 + vibration}));
    if (update.status == core::InitializationStatus::kAccepted) {
      break;
    }
  }

  ASSERT_EQ(update.status, core::InitializationStatus::kAccepted);
  ASSERT_TRUE(update.result.has_value());
  EXPECT_TRUE(initializer.accepted());
  const core::NavigationState& seed = update.result->seedState();
  EXPECT_NEAR(seed.imuBias().gyroscopeRadS().x, 0.01, 1.0e-6);
  EXPECT_EQ(seed.imuBias().accelerometerMS2(), (core::Vec3d{.x = 0.1, .y = -0.05, .z = 0.02}));
  EXPECT_NEAR(seed.odomFromImu().rotation().w(), 1.0, 1.0e-6);
  EXPECT_NEAR(seed.odomFromImu().translation().x, 0.0, 1.0e-12);
  EXPECT_NEAR(*update.quality.raw_gravity_magnitude_m_s2, 9.80665, 1.0e-4);

  const StaticInitializationUpdate repeated = initializer.add(
      sample(1'000U, seed.time().count() + kPeriodNs, {}, {.x = 0.1, .y = -0.05, .z = 9.82665}));
  ASSERT_EQ(repeated.status, core::InitializationStatus::kAccepted);
  ASSERT_TRUE(repeated.result.has_value());
  EXPECT_EQ(repeated.result->anchorTime(), update.result->anchorTime());
}

TEST(StaticInitializer, KeepsCollectingWhenDirectionIsNotStable) {
  StaticInitializerOptions config = options();
  config.maximum_block_direction_dispersion_rad = 0.01;
  StaticInitializer initializer(config);
  StaticInitializationUpdate update;
  constexpr std::int64_t kPeriodNs = 5'000'000;
  for (std::uint64_t index = 0; index <= 420U; ++index) {
    const double lateral = ((index / 10U) % 2U == 0U) ? 1.0 : -1.0;
    update = initializer.add(sample(index + 1U, static_cast<std::int64_t>(index) * kPeriodNs,
                                    {.x = 0.01, .y = -0.02, .z = 0.005},
                                    {.x = 0.1 + lateral, .y = -0.05, .z = 9.82665}));
  }

  EXPECT_EQ(update.status, core::InitializationStatus::kCollecting);
  EXPECT_FALSE(update.result.has_value());
  EXPECT_NE(update.reason.find("direction"), std::string::npos);
}

TEST(StaticInitializer, ResetsContiguousSupportOnGapAndNonFiniteData) {
  StaticInitializer initializer(options());
  constexpr std::int64_t kPeriodNs = 5'000'000;
  StaticInitializationUpdate update;
  for (std::uint64_t index = 0; index < 200U; ++index) {
    update = initializer.add(sample(index + 1U, static_cast<std::int64_t>(index) * kPeriodNs,
                                    {.x = 0.01, .y = -0.02, .z = 0.005},
                                    {.x = 0.1, .y = -0.05, .z = 9.82665}));
  }

  const std::int64_t after_gap = 250U * kPeriodNs;
  update = initializer.add(sample(201U, after_gap, {.x = 0.01, .y = -0.02, .z = 0.005},
                                  {.x = 0.1, .y = -0.05, .z = 9.82665}));
  EXPECT_EQ(update.status, core::InitializationStatus::kCollecting);
  EXPECT_EQ(update.quality.imu_sample_count, 1U);
  EXPECT_NE(update.reason.find("gap"), std::string::npos);

  update =
      initializer.add(sample(202U, after_gap + kPeriodNs,
                             {.x = std::numeric_limits<double>::quiet_NaN(), .y = 0.0, .z = 0.0},
                             {.x = 0.1, .y = -0.05, .z = 9.82665}));
  EXPECT_EQ(update.status, core::InitializationStatus::kCollecting);
  EXPECT_EQ(update.quality.imu_sample_count, 0U);
  EXPECT_NE(update.reason.find("non-finite"), std::string::npos);
  EXPECT_FALSE(initializer.accepted());
}

TEST(StaticInitializer, PlacesBaseAtTheZeroYawGaugeWithNontrivialExtrinsics) {
  StaticInitializerOptions config = options();
  const Eigen::Quaterniond base_from_imu_rotation(
      Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()));
  config.base_from_imu =
      core::Pose3d({.x = 0.12, .y = -0.04, .z = 0.08},
                   core::Quaterniond(base_from_imu_rotation.w(), base_from_imu_rotation.x(),
                                     base_from_imu_rotation.y(), base_from_imu_rotation.z()));
  StaticInitializer initializer(config);

  const Eigen::Vector3d specific_force_imu =
      base_from_imu_rotation.inverse() * (Eigen::Vector3d::UnitZ() * config.gravity_m_s2);
  StaticInitializationUpdate update;
  constexpr std::int64_t kPeriodNs = 5'000'000;
  for (std::uint64_t index = 0; index <= 420U; ++index) {
    update = initializer.add(
        sample(index + 1U, static_cast<std::int64_t>(index) * kPeriodNs, {},
               {.x = specific_force_imu.x() + config.calibrated_bias_prior.accelerometerMS2().x,
                .y = specific_force_imu.y() + config.calibrated_bias_prior.accelerometerMS2().y,
                .z = specific_force_imu.z() + config.calibrated_bias_prior.accelerometerMS2().z}));
    if (update.status == core::InitializationStatus::kAccepted) {
      break;
    }
  }

  ASSERT_EQ(update.status, core::InitializationStatus::kAccepted);
  ASSERT_TRUE(update.result.has_value());
  const core::Pose3d& odom_from_imu = update.result->seedState().odomFromImu();
  const Eigen::Matrix3d odom_from_base_rotation =
      eigen(odom_from_imu.rotation()).matrix() * base_from_imu_rotation.inverse().matrix();
  EXPECT_TRUE(odom_from_base_rotation.isApprox(Eigen::Matrix3d::Identity(), 1.0e-10));

  const Eigen::Vector3d odom_from_imu_translation(
      odom_from_imu.translation().x, odom_from_imu.translation().y, odom_from_imu.translation().z);
  const Eigen::Vector3d base_from_imu_translation(config.base_from_imu.translation().x,
                                                  config.base_from_imu.translation().y,
                                                  config.base_from_imu.translation().z);
  const Eigen::Vector3d odom_from_base_translation =
      odom_from_imu_translation - odom_from_base_rotation * base_from_imu_translation;
  EXPECT_TRUE(odom_from_base_translation.isZero(1.0e-10));
}

}  // namespace
}  // namespace meridian::local_rt::initialization
