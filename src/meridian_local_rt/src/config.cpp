#include "meridian/local_rt/config.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <string_view>

namespace meridian::local_rt {
namespace {

template <typename Matrix>
void validateCovariance(const Matrix& covariance, std::string_view field,
                        std::vector<ConfigIssue>& issues) {
  if (!covariance.array().isFinite().all()) {
    issues.push_back({ConfigIssueCode::kNonFinite, std::string(field)});
    return;
  }
  if (!covariance.isApprox(covariance.transpose(), 1.0e-12)) {
    issues.push_back({ConfigIssueCode::kNotSymmetric, std::string(field)});
    return;
  }
  Eigen::SelfAdjointEigenSolver<Matrix> solver(covariance);
  if (solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() < -1.0e-12) {
    issues.push_back({ConfigIssueCode::kNotPositiveSemidefinite, std::string(field)});
  }
}

void requirePositiveFinite(double value, std::string_view field, std::vector<ConfigIssue>& issues) {
  if (!std::isfinite(value)) {
    issues.push_back({ConfigIssueCode::kNonFinite, std::string(field)});
  } else if (value <= 0.0) {
    issues.push_back({ConfigIssueCode::kNonPositive, std::string(field)});
  }
}

template <typename Rep, typename Period>
void requirePositiveDuration(std::chrono::duration<Rep, Period> value, std::string_view field,
                             std::vector<ConfigIssue>& issues) {
  if (value.count() <= 0) {
    issues.push_back({ConfigIssueCode::kNonPositive, std::string(field)});
  }
}

}  // namespace

std::vector<ConfigIssue> Config::validate() const {
  std::vector<ConfigIssue> issues;

  const core::Vec3d& gravity = imu_model.gravity_odom_m_s2;
  if (!gravity.isFinite()) {
    issues.push_back({ConfigIssueCode::kNonFinite, "imu_model.gravity_odom_m_s2"});
  } else if (gravity.x * gravity.x + gravity.y * gravity.y + gravity.z * gravity.z <= 0.0) {
    issues.push_back({ConfigIssueCode::kNonPositive, "imu_model.gravity_odom_m_s2"});
  }

  validateCovariance(imu_model.accelerometer_covariance_density,
                     "imu_model.accelerometer_covariance_density", issues);
  validateCovariance(imu_model.gyroscope_covariance_density,
                     "imu_model.gyroscope_covariance_density", issues);
  validateCovariance(imu_model.integration_covariance_density,
                     "imu_model.integration_covariance_density", issues);
  validateCovariance(imu_model.accelerometer_bias_random_walk_covariance,
                     "imu_model.accelerometer_bias_random_walk_covariance", issues);
  validateCovariance(imu_model.gyroscope_bias_random_walk_covariance,
                     "imu_model.gyroscope_bias_random_walk_covariance", issues);
  validateCovariance(initialization.calibrated_bias_prior_covariance,
                     "initialization.calibrated_bias_prior_covariance", issues);

  requirePositiveDuration(initialization.static_mode.support, "initialization.static_mode.support",
                          issues);
  requirePositiveDuration(initialization.static_mode.block_duration,
                          "initialization.static_mode.block_duration", issues);
  if (initialization.static_mode.block_duration > initialization.static_mode.support) {
    issues.push_back(
        {ConfigIssueCode::kInvalidOrdering, "initialization.static_mode.block_duration"});
  }
  if (initialization.static_mode.minimum_imu_samples == 0U) {
    issues.push_back(
        {ConfigIssueCode::kNonPositive, "initialization.static_mode.minimum_imu_samples"});
  }
  requirePositiveFinite(initialization.static_mode.maximum_mean_gyro_rad_s,
                        "initialization.static_mode.maximum_mean_gyro_rad_s", issues);
  requirePositiveFinite(initialization.static_mode.maximum_gyro_stddev_rad_s,
                        "initialization.static_mode.maximum_gyro_stddev_rad_s", issues);
  requirePositiveFinite(initialization.static_mode.maximum_specific_force_norm_error_m_s2,
                        "initialization.static_mode.maximum_specific_force_norm_error_m_s2",
                        issues);

  if (initialization.dynamic_mode.target_lidar_sweeps == 0U) {
    issues.push_back(
        {ConfigIssueCode::kNonPositive, "initialization.dynamic_mode.target_lidar_sweeps"});
  }
  if (initialization.dynamic_mode.minimum_lidar_transitions == 0U) {
    issues.push_back(
        {ConfigIssueCode::kNonPositive, "initialization.dynamic_mode.minimum_lidar_transitions"});
  }
  requirePositiveDuration(initialization.dynamic_mode.preferred_support,
                          "initialization.dynamic_mode.preferred_support", issues);
  requirePositiveDuration(initialization.dynamic_mode.maximum_support,
                          "initialization.dynamic_mode.maximum_support", issues);
  if (initialization.dynamic_mode.preferred_support > initialization.dynamic_mode.maximum_support) {
    issues.push_back(
        {ConfigIssueCode::kInvalidOrdering, "initialization.dynamic_mode.preferred_support"});
  }
  if (initialization.dynamic_mode.maximum_refinement_passes == 0U) {
    issues.push_back(
        {ConfigIssueCode::kNonPositive, "initialization.dynamic_mode.maximum_refinement_passes"});
  }
  requirePositiveFinite(initialization.dynamic_mode.maximum_condition_number,
                        "initialization.dynamic_mode.maximum_condition_number", issues);

  if (imu_buffer.capacity == 0U) {
    issues.push_back({ConfigIssueCode::kNonPositive, "imu_buffer.capacity"});
  }
  requirePositiveDuration(imu_buffer.maximum_gap, "imu_buffer.maximum_gap", issues);

  return issues;
}

}  // namespace meridian::local_rt
