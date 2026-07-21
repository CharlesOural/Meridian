#pragma once

#include <Eigen/Core>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "meridian/core/geometry.hpp"
#include "meridian/core/initialization.hpp"
#include "meridian/core/navigation.hpp"

namespace meridian::local_rt {

// All matrices are continuous-time covariance densities in the IMU frame.
struct ImuModel final {
  core::Vec3d gravity_odom_m_s2{0.0, 0.0, -9.80665};
  Eigen::Matrix3d accelerometer_covariance_density = Eigen::Matrix3d::Identity() * 1.0e-4;
  Eigen::Matrix3d gyroscope_covariance_density = Eigen::Matrix3d::Identity() * 1.0e-6;
  Eigen::Matrix3d integration_covariance_density = Eigen::Matrix3d::Identity() * 1.0e-8;
  Eigen::Matrix3d accelerometer_bias_random_walk_covariance = Eigen::Matrix3d::Identity() * 1.0e-6;
  Eigen::Matrix3d gyroscope_bias_random_walk_covariance = Eigen::Matrix3d::Identity() * 1.0e-8;
};

// T_A_B maps coordinates from B into A.
struct ExtrinsicsConfig final {
  core::Pose3d T_base_imu{};
  core::Pose3d T_imu_lidar{};
  // Added to LiDAR timestamps before querying IMU support.
  std::chrono::nanoseconds lidar_time_offset_to_imu{};
};

struct StaticInitializationConfig final {
  std::chrono::nanoseconds support{std::chrono::seconds(2)};
  std::chrono::nanoseconds block_duration{std::chrono::milliseconds(100)};
  std::size_t minimum_imu_samples{200};
  double maximum_mean_gyro_rad_s{0.05};
  double maximum_gyro_stddev_rad_s{0.02};
  double maximum_specific_force_norm_error_m_s2{0.5};
};

struct DynamicInitializationConfig final {
  std::size_t target_lidar_sweeps{20};
  std::size_t minimum_lidar_transitions{8};
  std::chrono::nanoseconds preferred_support{std::chrono::seconds(2)};
  std::chrono::nanoseconds maximum_support{std::chrono::seconds(5)};
  std::uint32_t maximum_refinement_passes{1};
  double maximum_condition_number{1.0e8};
};

struct InitializationConfig final {
  core::InitializationMode mode{core::InitializationMode::kStatic};
  core::ImuBias calibrated_bias_prior{};
  // Tangent order: gyro bias, then accelerometer bias.
  Eigen::Matrix<double, 6, 6> calibrated_bias_prior_covariance =
      Eigen::Matrix<double, 6, 6>::Identity() * 1.0e-2;
  StaticInitializationConfig static_mode{};
  DynamicInitializationConfig dynamic_mode{};
};

struct ImuBufferConfig final {
  std::size_t capacity{4096};
  std::chrono::nanoseconds maximum_gap{std::chrono::milliseconds(50)};
};

enum class ConfigIssueCode : std::uint8_t {
  kNonFinite,
  kNonPositive,
  kNotSymmetric,
  kNotPositiveSemidefinite,
  kInvalidOrdering,
};

struct ConfigIssue final {
  ConfigIssueCode code;
  std::string field;
};

struct Config final {
  ImuModel imu_model{};
  ExtrinsicsConfig extrinsics{};
  InitializationConfig initialization{};
  ImuBufferConfig imu_buffer{};

  [[nodiscard]] std::vector<ConfigIssue> validate() const;
};

}  // namespace meridian::local_rt
