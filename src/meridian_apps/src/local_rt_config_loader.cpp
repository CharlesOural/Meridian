#include "meridian/apps/local_rt_config_loader.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "meridian/core/geometry.hpp"
#include "meridian/core/initialization.hpp"
#include "meridian/core/navigation.hpp"
#include "meridian/local_rt/config.hpp"
#include "meridian/local_rt/initialization/bootstrap_odometry.hpp"
#include "meridian/local_rt/initialization/dynamic_initializer.hpp"
#include "meridian/local_rt/initialization/static_initializer.hpp"
#include "meridian/local_rt/lidar/point_to_point_registration.hpp"
#include "meridian/local_rt/lidar/voxel_target.hpp"

namespace meridian::apps {
namespace {

std::string parameterName(std::string_view suffix) {
  return "local_rt." + std::string(suffix);
}

double positiveDoubleParameter(rclcpp::Node& node, std::string_view suffix) {
  const std::string name = parameterName(suffix);
  const double value = node.declare_parameter<double>(name, 0.0);
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(name + " must be finite and positive");
  }
  return value;
}

double nonNegativeDoubleParameter(rclcpp::Node& node, std::string_view suffix) {
  const std::string name = parameterName(suffix);
  const double value = node.declare_parameter<double>(name, -1.0);
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(name + " must be finite and non-negative");
  }
  return value;
}

std::int64_t positiveInt64Parameter(rclcpp::Node& node, std::string_view suffix) {
  const std::string name = parameterName(suffix);
  const std::int64_t value = node.declare_parameter<std::int64_t>(name, 0);
  if (value <= 0) {
    throw std::invalid_argument(name + " must be positive");
  }
  return value;
}

std::size_t positiveSizeParameter(rclcpp::Node& node, std::string_view suffix) {
  const std::string name = parameterName(suffix);
  const std::int64_t value = node.declare_parameter<std::int64_t>(name, 0);
  if (value <= 0 || static_cast<std::uint64_t>(value) >
                        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument(name + " must be a positive size");
  }
  return static_cast<std::size_t>(value);
}

std::vector<double> finiteVectorParameter(rclcpp::Node& node, std::string_view suffix,
                                          std::size_t expected_size) {
  const std::string name = parameterName(suffix);
  const std::vector<double> value =
      node.declare_parameter<std::vector<double>>(name, std::vector<double>{});
  if (value.size() != expected_size) {
    throw std::invalid_argument(name + " must contain exactly " + std::to_string(expected_size) +
                                " values");
  }
  if (!std::all_of(value.begin(), value.end(),
                   [](double coefficient) { return std::isfinite(coefficient); })) {
    throw std::invalid_argument(name + " must contain only finite values");
  }
  return value;
}

core::Vec3d vector3Parameter(rclcpp::Node& node, std::string_view suffix) {
  const std::vector<double> value = finiteVectorParameter(node, suffix, 3U);
  return {.x = value[0], .y = value[1], .z = value[2]};
}

core::Pose3d poseParameter(rclcpp::Node& node, std::string_view suffix) {
  const std::string translation_suffix = std::string(suffix) + ".translation_m";
  const std::string quaternion_suffix = std::string(suffix) + ".quaternion_xyzw";
  const core::Vec3d translation = vector3Parameter(node, translation_suffix);
  const std::vector<double> quaternion = finiteVectorParameter(node, quaternion_suffix, 4U);
  const double squared_norm = quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                              quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3];
  if (std::abs(squared_norm - 1.0) > 1.0e-9) {
    throw std::invalid_argument(parameterName(quaternion_suffix) +
                                " must be a unit quaternion in XYZW order");
  }
  return core::Pose3d(
      translation, core::Quaterniond(quaternion[3], quaternion[0], quaternion[1], quaternion[2]));
}

core::InitializationMode initializationModeParameter(rclcpp::Node& node) {
  const std::string name = parameterName("initialization.mode");
  std::string value = node.declare_parameter<std::string>(name, "");
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
    return static_cast<char>(std::toupper(character));
  });
  if (value == "STATIC") {
    return core::InitializationMode::kStatic;
  }
  if (value == "DYNAMIC") {
    return core::InitializationMode::kDynamic;
  }
  throw std::invalid_argument(name + " must be exactly STATIC or DYNAMIC");
}

double squareDensity(double density, std::string_view suffix) {
  const double covariance_density = density * density;
  if (!std::isfinite(covariance_density) || covariance_density <= 0.0) {
    throw std::invalid_argument(parameterName(suffix) +
                                " is outside the representable covariance range");
  }
  return covariance_density;
}

Eigen::Matrix3d diagonal3(double diagonal) {
  return Eigen::Matrix3d::Identity() * diagonal;
}

void validateStaticRelationships(
    const local_rt::initialization::StaticInitializerOptions& options) {
  if (options.block_duration_ns > options.window_duration_ns) {
    throw std::invalid_argument(
        parameterName("initialization.static.block_duration_ns") +
        " must not exceed local_rt.initialization.static.window_duration_ns");
  }
  if (options.minimum_samples < 2U) {
    throw std::invalid_argument(parameterName("initialization.static.minimum_samples") +
                                " must be at least 2");
  }
  if (options.minimum_blocks < 2U) {
    throw std::invalid_argument(parameterName("initialization.static.minimum_blocks") +
                                " must be at least 2");
  }
  const std::uint64_t possible_blocks =
      static_cast<std::uint64_t>(options.window_duration_ns / options.block_duration_ns) +
      static_cast<std::uint64_t>(options.window_duration_ns % options.block_duration_ns != 0);
  if (static_cast<std::uint64_t>(options.minimum_blocks) > possible_blocks) {
    throw std::invalid_argument(parameterName("initialization.static.minimum_blocks") +
                                " exceeds the number of blocks in the configured window");
  }
}

void validateDynamicRelationships(
    const local_rt::initialization::DynamicInitializerOptions& options) {
  if (options.target_sweeps < 4U) {
    throw std::invalid_argument(parameterName("initialization.dynamic.target_sweeps") +
                                " must be at least 4 to retain a held-out transition");
  }
  if (options.minimum_range_m >= options.maximum_range_m) {
    throw std::invalid_argument(
        parameterName("initialization.dynamic.minimum_range_m") +
        " must be less than local_rt.initialization.dynamic.maximum_range_m");
  }
  if (options.minimum_singular_value_ratio >= 1.0) {
    throw std::invalid_argument(
        parameterName("initialization.dynamic.minimum_singular_value_ratio") +
        " must be less than 1");
  }
  if (options.maximum_condition_number < 1.0) {
    throw std::invalid_argument(parameterName("initialization.dynamic.maximum_condition_number") +
                                " must be at least 1");
  }
}

void validateBootstrapRelationships(
    const local_rt::initialization::BootstrapOdometryOptions& options) {
  if (options.minimum_points < 3U) {
    throw std::invalid_argument(parameterName("initialization.dynamic.bootstrap.minimum_points") +
                                " must be at least 3");
  }
  if (options.registration.minimum_correspondences < 3U ||
      options.registration.minimum_correspondences > options.registration.max_source_points) {
    throw std::invalid_argument(
        parameterName("initialization.dynamic.bootstrap.registration.minimum_correspondences") +
        " must be between 3 and max_source_points");
  }
  if (options.minimum_points > options.registration.max_source_points ||
      options.minimum_points < options.registration.minimum_correspondences) {
    throw std::invalid_argument(
        parameterName("initialization.dynamic.bootstrap.minimum_points") +
        " must be between registration.minimum_correspondences and registration.max_source_points");
  }
  const std::size_t required_voxels =
      options.minimum_points / options.target.max_points_per_voxel +
      static_cast<std::size_t>(options.minimum_points % options.target.max_points_per_voxel != 0U);
  if (options.target.max_voxels < required_voxels) {
    throw std::invalid_argument(
        parameterName("initialization.dynamic.bootstrap.target.max_voxels") +
        " cannot hold the configured minimum_points");
  }
  const double required_neighbor_radius =
      std::ceil(options.registration.max_correspondence_distance_m / options.target.voxel_size_m);
  if (!std::isfinite(required_neighbor_radius) ||
      required_neighbor_radius > static_cast<double>(options.target.max_neighbor_voxel_radius)) {
    throw std::invalid_argument(
        parameterName("initialization.dynamic.bootstrap.target.max_neighbor_voxel_radius") +
        " is too small for registration.max_correspondence_distance_m");
  }
  if (options.registration.relative_rank_tolerance >= 1.0) {
    throw std::invalid_argument(
        parameterName("initialization.dynamic.bootstrap.registration.relative_rank_tolerance") +
        " must be less than 1");
  }
  if (options.maximum_accepted_condition_number < 1.0) {
    throw std::invalid_argument(
        parameterName("initialization.dynamic.bootstrap.maximum_accepted_condition_number") +
        " must be at least 1");
  }
}

}  // namespace

local_rt::LocalRtPipelineConfig loadLocalRtPipelineConfig(rclcpp::Node& node) {
  const core::InitializationMode mode = initializationModeParameter(node);
  const double gravity_m_s2 = positiveDoubleParameter(node, "imu.gravity_m_s2");

  const double accelerometer_noise_density =
      positiveDoubleParameter(node, "imu.accelerometer_noise_density");
  const double gyroscope_noise_density =
      positiveDoubleParameter(node, "imu.gyroscope_noise_density");
  const double integration_covariance_density =
      positiveDoubleParameter(node, "imu.integration_covariance_density");
  const double accelerometer_bias_random_walk_density =
      positiveDoubleParameter(node, "imu.accelerometer_bias_random_walk_density");
  const double gyroscope_bias_random_walk_density =
      positiveDoubleParameter(node, "imu.gyroscope_bias_random_walk_density");

  const core::Pose3d base_from_imu = poseParameter(node, "extrinsics.base_from_imu");
  const core::Pose3d imu_from_lidar = poseParameter(node, "extrinsics.imu_from_lidar");
  const std::int64_t lidar_time_offset_to_imu_ns = node.declare_parameter<std::int64_t>(
      parameterName("extrinsics.lidar_time_offset_to_imu_ns"), 0);

  const core::ImuBias calibrated_bias_prior(
      vector3Parameter(node, "initialization.bias_prior.gyroscope_rad_s"),
      vector3Parameter(node, "initialization.bias_prior.accelerometer_m_s2"));
  const double gyroscope_bias_prior_sigma =
      positiveDoubleParameter(node, "initialization.bias_prior.gyroscope_sigma_rad_s");
  const double accelerometer_bias_prior_sigma =
      positiveDoubleParameter(node, "initialization.bias_prior.accelerometer_sigma_m_s2");
  const double gyroscope_bias_prior_variance =
      squareDensity(gyroscope_bias_prior_sigma, "initialization.bias_prior.gyroscope_sigma_rad_s");
  const double accelerometer_bias_prior_variance = squareDensity(
      accelerometer_bias_prior_sigma, "initialization.bias_prior.accelerometer_sigma_m_s2");
  Eigen::Matrix<double, 6, 6> calibrated_bias_prior_covariance =
      Eigen::Matrix<double, 6, 6>::Zero();
  calibrated_bias_prior_covariance.diagonal().template head<3>().setConstant(
      gyroscope_bias_prior_variance);
  calibrated_bias_prior_covariance.diagonal().template tail<3>().setConstant(
      accelerometer_bias_prior_variance);

  const std::size_t imu_buffer_capacity = positiveSizeParameter(node, "imu_buffer.capacity");
  const std::int64_t imu_maximum_gap_ns = positiveInt64Parameter(node, "imu_buffer.maximum_gap_ns");

  local_rt::estimator::FixedLagEstimatorOptions fixed_lag_options{
      .maximum_states = positiveSizeParameter(node, "fixed_lag.maximum_states"),
      .maximum_lag_ns = positiveInt64Parameter(node, "fixed_lag.maximum_duration_ns"),
      .maximum_solver_iterations =
          positiveSizeParameter(node, "fixed_lag.maximum_solver_iterations"),
      .solver_threads = positiveSizeParameter(node, "fixed_lag.solver_threads"),
      .maximum_solver_time_s = positiveDoubleParameter(node, "fixed_lag.maximum_solver_time_s"),
      .function_tolerance = positiveDoubleParameter(node, "fixed_lag.function_tolerance"),
      .gradient_tolerance = positiveDoubleParameter(node, "fixed_lag.gradient_tolerance"),
      .parameter_tolerance = positiveDoubleParameter(node, "fixed_lag.parameter_tolerance"),
      .maximum_translation_correction_m =
          positiveDoubleParameter(node, "fixed_lag.maximum_translation_correction_m"),
      .maximum_rotation_correction_rad =
          positiveDoubleParameter(node, "fixed_lag.maximum_rotation_correction_rad"),
      .maximum_prior_translation_m =
          positiveDoubleParameter(node, "fixed_lag.maximum_prior_translation_m"),
      .maximum_prior_rotation_rad =
          positiveDoubleParameter(node, "fixed_lag.maximum_prior_rotation_rad"),
      .maximum_prior_motion_norm =
          positiveDoubleParameter(node, "fixed_lag.maximum_prior_motion_norm"),
      .initial_translation_sigma_m =
          positiveDoubleParameter(node, "fixed_lag.initial_translation_sigma_m"),
      .initial_rotation_sigma_rad =
          positiveDoubleParameter(node, "fixed_lag.initial_rotation_sigma_rad"),
      .initial_velocity_sigma_m_s =
          positiveDoubleParameter(node, "fixed_lag.initial_velocity_sigma_m_s"),
      .initial_gyro_bias_sigma_rad_s =
          positiveDoubleParameter(node, "fixed_lag.initial_gyro_bias_sigma_rad_s"),
      .initial_accel_bias_sigma_m_s2 =
          positiveDoubleParameter(node, "fixed_lag.initial_accel_bias_sigma_m_s2"),
      .marginalization =
          {
              .elimination_rank_tolerance = positiveDoubleParameter(
                  node, "fixed_lag.marginalization_elimination_rank_tolerance"),
              .compression_rank_tolerance = positiveDoubleParameter(
                  node, "fixed_lag.marginalization_compression_rank_tolerance"),
          },
  };
  if (fixed_lag_options.maximum_states < 2U) {
    throw std::invalid_argument(parameterName("fixed_lag.maximum_states") + " must be at least 2");
  }

  const double lidar_minimum_range_m = positiveDoubleParameter(node, "lidar.minimum_range_m");
  const double lidar_maximum_range_m = positiveDoubleParameter(node, "lidar.maximum_range_m");
  if (lidar_minimum_range_m >= lidar_maximum_range_m) {
    throw std::invalid_argument(parameterName("lidar.minimum_range_m") +
                                " must be less than local_rt.lidar.maximum_range_m");
  }
  const double target_downsample_voxel_m =
      positiveDoubleParameter(node, "lidar.target_downsample_voxel_m");
  const double source_downsample_voxel_m =
      positiveDoubleParameter(node, "lidar.source_downsample_voxel_m");
  local_rt::lidar::ScanToMapOptions scan_to_map_options{
      .target_downsample_voxel_m = target_downsample_voxel_m,
      .source_downsample_voxel_m = source_downsample_voxel_m,
      .maximum_active_owners = positiveSizeParameter(node, "lidar.maximum_active_owners"),
      .maximum_factor_rows = positiveSizeParameter(node, "lidar.maximum_factor_rows"),
      .minimum_correspondences = positiveSizeParameter(node, "lidar.minimum_correspondences"),
      .maximum_icp_iterations = positiveSizeParameter(node, "lidar.maximum_icp_iterations"),
      .maximum_backtracking_steps = positiveSizeParameter(node, "lidar.maximum_backtracking_steps"),
      .maximum_correspondence_distance_m =
          positiveDoubleParameter(node, "lidar.maximum_correspondence_distance_m"),
      .point_sigma_m = positiveDoubleParameter(node, "lidar.point_sigma_m"),
      .huber_mad_multiplier = positiveDoubleParameter(node, "lidar.huber_mad_multiplier"),
      .minimum_huber_scale_m = positiveDoubleParameter(node, "lidar.minimum_huber_scale_m"),
      .maximum_huber_scale_m = positiveDoubleParameter(node, "lidar.maximum_huber_scale_m"),
      .translation_convergence_m = positiveDoubleParameter(node, "lidar.translation_convergence_m"),
      .rotation_convergence_rad = positiveDoubleParameter(node, "lidar.rotation_convergence_rad"),
      .maximum_translation_step_m =
          positiveDoubleParameter(node, "lidar.maximum_translation_step_m"),
      .maximum_rotation_step_rad = positiveDoubleParameter(node, "lidar.maximum_rotation_step_rad"),
      .maximum_prediction_correction_m =
          positiveDoubleParameter(node, "lidar.maximum_prediction_correction_m"),
      .maximum_prediction_correction_rad =
          positiveDoubleParameter(node, "lidar.maximum_prediction_correction_rad"),
      .relative_rank_tolerance = positiveDoubleParameter(node, "lidar.relative_rank_tolerance"),
      .lm_damping = positiveDoubleParameter(node, "lidar.lm_damping"),
      .active_owner_index =
          {
              .voxel_size_m = positiveDoubleParameter(node, "lidar.active_target.voxel_size_m"),
              .retention_radius_m =
                  positiveDoubleParameter(node, "lidar.active_target.retention_radius_m"),
              .max_voxels = positiveSizeParameter(node, "lidar.active_target.max_voxels"),
              .max_points_per_voxel =
                  positiveSizeParameter(node, "lidar.active_target.max_points_per_voxel"),
              .minimum_point_spacing_m =
                  nonNegativeDoubleParameter(node, "lidar.active_target.minimum_point_spacing_m"),
              .max_neighbor_voxel_radius =
                  positiveSizeParameter(node, "lidar.active_target.max_neighbor_voxel_radius"),
          },
      .finalized_base =
          {
              .voxel_size_m = positiveDoubleParameter(node, "lidar.finalized_target.voxel_size_m"),
              .retention_radius_m =
                  positiveDoubleParameter(node, "lidar.finalized_target.retention_radius_m"),
              .max_voxels = positiveSizeParameter(node, "lidar.finalized_target.max_voxels"),
              .max_points_per_voxel =
                  positiveSizeParameter(node, "lidar.finalized_target.max_points_per_voxel"),
              .minimum_point_spacing_m = nonNegativeDoubleParameter(
                  node, "lidar.finalized_target.minimum_point_spacing_m"),
              .max_neighbor_voxel_radius =
                  positiveSizeParameter(node, "lidar.finalized_target.max_neighbor_voxel_radius"),
          },
  };
  if (scan_to_map_options.minimum_correspondences > scan_to_map_options.maximum_factor_rows) {
    throw std::invalid_argument(parameterName("lidar.minimum_correspondences") +
                                " must not exceed local_rt.lidar.maximum_factor_rows");
  }

  local_rt::initialization::StaticInitializerOptions static_options{
      .window_duration_ns =
          positiveInt64Parameter(node, "initialization.static.window_duration_ns"),
      .block_duration_ns = positiveInt64Parameter(node, "initialization.static.block_duration_ns"),
      .minimum_samples = positiveSizeParameter(node, "initialization.static.minimum_samples"),
      .minimum_blocks = positiveSizeParameter(node, "initialization.static.minimum_blocks"),
      .maximum_sample_gap_ns = imu_maximum_gap_ns,
      .gravity_m_s2 = gravity_m_s2,
      .gyroscope_saturation_rad_s =
          positiveDoubleParameter(node, "initialization.static.gyroscope_saturation_rad_s"),
      .accelerometer_saturation_m_s2 =
          positiveDoubleParameter(node, "initialization.static.accelerometer_saturation_m_s2"),
      .maximum_mean_angular_rate_rad_s =
          positiveDoubleParameter(node, "initialization.static.maximum_mean_angular_rate_rad_s"),
      .maximum_block_angular_dispersion_rad_s = positiveDoubleParameter(
          node, "initialization.static.maximum_block_angular_dispersion_rad_s"),
      .maximum_specific_force_norm_error_m_s2 = positiveDoubleParameter(
          node, "initialization.static.maximum_specific_force_norm_error_m_s2"),
      .maximum_block_direction_dispersion_rad = positiveDoubleParameter(
          node, "initialization.static.maximum_block_direction_dispersion_rad"),
      .calibrated_bias_prior = calibrated_bias_prior,
      .base_from_imu = base_from_imu,
  };
  validateStaticRelationships(static_options);

  local_rt::initialization::BootstrapOdometryOptions bootstrap_options{
      .downsample_voxel_size_m =
          positiveDoubleParameter(node, "initialization.dynamic.bootstrap.downsample_voxel_size_m"),
      .minimum_points =
          positiveSizeParameter(node, "initialization.dynamic.bootstrap.minimum_points"),
      .maximum_accepted_rmse_m =
          positiveDoubleParameter(node, "initialization.dynamic.bootstrap.maximum_accepted_rmse_m"),
      .maximum_accepted_condition_number = positiveDoubleParameter(
          node, "initialization.dynamic.bootstrap.maximum_accepted_condition_number"),
      .target =
          {
              .voxel_size_m = positiveDoubleParameter(
                  node, "initialization.dynamic.bootstrap.target.voxel_size_m"),
              .retention_radius_m = positiveDoubleParameter(
                  node, "initialization.dynamic.bootstrap.target.retention_radius_m"),
              .max_voxels =
                  positiveSizeParameter(node, "initialization.dynamic.bootstrap.target.max_voxels"),
              .max_points_per_voxel = positiveSizeParameter(
                  node, "initialization.dynamic.bootstrap.target.max_points_per_voxel"),
              .minimum_point_spacing_m = nonNegativeDoubleParameter(
                  node, "initialization.dynamic.bootstrap.target.minimum_point_spacing_m"),
              .max_neighbor_voxel_radius = positiveSizeParameter(
                  node, "initialization.dynamic.bootstrap.target.max_neighbor_voxel_radius"),
          },
      .registration =
          {
              .max_iterations = positiveSizeParameter(
                  node, "initialization.dynamic.bootstrap.registration.max_iterations"),
              .max_source_points = positiveSizeParameter(
                  node, "initialization.dynamic.bootstrap.registration.max_source_points"),
              .minimum_correspondences = positiveSizeParameter(
                  node, "initialization.dynamic.bootstrap.registration.minimum_correspondences"),
              .max_correspondence_distance_m = positiveDoubleParameter(
                  node,
                  "initialization.dynamic.bootstrap.registration.max_correspondence_distance_m"),
              .geman_mcclure_scale_m = positiveDoubleParameter(
                  node, "initialization.dynamic.bootstrap.registration.geman_mcclure_scale_m"),
              .translation_convergence_m = positiveDoubleParameter(
                  node, "initialization.dynamic.bootstrap.registration.translation_convergence_m"),
              .rotation_convergence_rad = positiveDoubleParameter(
                  node, "initialization.dynamic.bootstrap.registration.rotation_convergence_rad"),
              .max_translation_step_m = positiveDoubleParameter(
                  node, "initialization.dynamic.bootstrap.registration.max_translation_step_m"),
              .max_rotation_step_rad = positiveDoubleParameter(
                  node, "initialization.dynamic.bootstrap.registration.max_rotation_step_rad"),
              .relative_rank_tolerance = positiveDoubleParameter(
                  node, "initialization.dynamic.bootstrap.registration.relative_rank_tolerance"),
          },
  };
  validateBootstrapRelationships(bootstrap_options);

  local_rt::initialization::DynamicInitializerOptions dynamic_options{
      .target_sweeps = positiveSizeParameter(node, "initialization.dynamic.target_sweeps"),
      .maximum_support_ns =
          positiveInt64Parameter(node, "initialization.dynamic.maximum_support_ns"),
      .lidar_time_offset_to_imu_ns = lidar_time_offset_to_imu_ns,
      .minimum_range_m = positiveDoubleParameter(node, "initialization.dynamic.minimum_range_m"),
      .maximum_range_m = positiveDoubleParameter(node, "initialization.dynamic.maximum_range_m"),
      .gravity_m_s2 = gravity_m_s2,
      .gyroscope_finite_difference_step_rad_s = positiveDoubleParameter(
          node, "initialization.dynamic.gyroscope_finite_difference_step_rad_s"),
      .minimum_singular_value_ratio =
          positiveDoubleParameter(node, "initialization.dynamic.minimum_singular_value_ratio"),
      .maximum_condition_number =
          positiveDoubleParameter(node, "initialization.dynamic.maximum_condition_number"),
      .maximum_gyro_bias_correction_rad_s = positiveDoubleParameter(
          node, "initialization.dynamic.maximum_gyro_bias_correction_rad_s"),
      .maximum_gravity_magnitude_error_m_s2 = positiveDoubleParameter(
          node, "initialization.dynamic.maximum_gravity_magnitude_error_m_s2"),
      .maximum_alignment_residual_rms =
          positiveDoubleParameter(node, "initialization.dynamic.maximum_alignment_residual_rms"),
      .maximum_held_out_rotation_error_rad = positiveDoubleParameter(
          node, "initialization.dynamic.maximum_held_out_rotation_error_rad"),
      .maximum_held_out_translation_error_m = positiveDoubleParameter(
          node, "initialization.dynamic.maximum_held_out_translation_error_m"),
      .maximum_refinement_rotation_change_rad = positiveDoubleParameter(
          node, "initialization.dynamic.maximum_refinement_rotation_change_rad"),
      .maximum_refinement_translation_change_m = positiveDoubleParameter(
          node, "initialization.dynamic.maximum_refinement_translation_change_m"),
      .calibrated_bias_prior = calibrated_bias_prior,
      .gyroscope_bias_prior_covariance = diagonal3(gyroscope_bias_prior_variance),
      .base_from_imu = base_from_imu,
      .imu_from_lidar = imu_from_lidar,
      .bootstrap = std::move(bootstrap_options),
  };
  validateDynamicRelationships(dynamic_options);

  local_rt::Config estimator{
      .imu_model =
          {
              .gravity_odom_m_s2 = {.x = 0.0, .y = 0.0, .z = -gravity_m_s2},
              .accelerometer_covariance_density = diagonal3(
                  squareDensity(accelerometer_noise_density, "imu.accelerometer_noise_density")),
              .gyroscope_covariance_density =
                  diagonal3(squareDensity(gyroscope_noise_density, "imu.gyroscope_noise_density")),
              .integration_covariance_density = diagonal3(integration_covariance_density),
              .accelerometer_bias_random_walk_covariance =
                  diagonal3(squareDensity(accelerometer_bias_random_walk_density,
                                          "imu.accelerometer_bias_random_walk_density")),
              .gyroscope_bias_random_walk_covariance = diagonal3(squareDensity(
                  gyroscope_bias_random_walk_density, "imu.gyroscope_bias_random_walk_density")),
          },
      .extrinsics =
          {
              .T_base_imu = base_from_imu,
              .T_imu_lidar = imu_from_lidar,
              .lidar_time_offset_to_imu = std::chrono::nanoseconds(lidar_time_offset_to_imu_ns),
          },
      .initialization =
          {
              .mode = mode,
              .calibrated_bias_prior = calibrated_bias_prior,
              .calibrated_bias_prior_covariance = calibrated_bias_prior_covariance,
              .static_mode =
                  {
                      .support = std::chrono::nanoseconds(static_options.window_duration_ns),
                      .block_duration = std::chrono::nanoseconds(static_options.block_duration_ns),
                      .minimum_imu_samples = static_options.minimum_samples,
                      .maximum_mean_gyro_rad_s = static_options.maximum_mean_angular_rate_rad_s,
                      .maximum_gyro_stddev_rad_s =
                          static_options.maximum_block_angular_dispersion_rad_s,
                      .maximum_specific_force_norm_error_m_s2 =
                          static_options.maximum_specific_force_norm_error_m_s2,
                  },
              .dynamic_mode =
                  {
                      .target_lidar_sweeps = dynamic_options.target_sweeps,
                      .minimum_lidar_transitions = dynamic_options.target_sweeps - 1U,
                      .preferred_support =
                          std::chrono::nanoseconds(dynamic_options.maximum_support_ns),
                      .maximum_support =
                          std::chrono::nanoseconds(dynamic_options.maximum_support_ns),
                      .maximum_refinement_passes = 1U,
                      .maximum_condition_number = dynamic_options.maximum_condition_number,
                  },
          },
      .imu_buffer =
          {
              .capacity = imu_buffer_capacity,
              .maximum_gap = std::chrono::nanoseconds(imu_maximum_gap_ns),
          },
  };
  const std::vector<local_rt::ConfigIssue> issues = estimator.validate();
  if (!issues.empty()) {
    throw std::invalid_argument("local_rt estimator configuration is invalid at " +
                                issues.front().field);
  }

  return local_rt::LocalRtPipelineConfig{
      .estimator = std::move(estimator),
      .static_initializer = std::move(static_options),
      .dynamic_initializer = std::move(dynamic_options),
      .fixed_lag = std::move(fixed_lag_options),
      .scan_to_map = scan_to_map_options,
      .scan_preprocessor =
          {
              .minimum_range_m = lidar_minimum_range_m,
              .maximum_range_m = lidar_maximum_range_m,
              .target_downsample_voxel_m = target_downsample_voxel_m,
              .source_downsample_voxel_m = source_downsample_voxel_m,
              .lidar_time_offset_to_imu_ns = lidar_time_offset_to_imu_ns,
              .T_imu_lidar = imu_from_lidar,
          },
      .imu_queue_capacity = positiveSizeParameter(node, "queues.imu_capacity"),
      .lidar_queue_capacity = positiveSizeParameter(node, "queues.lidar_capacity"),
  };
}

}  // namespace meridian::apps
