#include "meridian/local/motion_initializer.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PoseTranslationPrior.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <boost/make_shared.hpp>
#include <cmath>
#include <exception>
#include <limits>
#include <set>
#include <sophus/so3.hpp>
#include <sstream>
#include <utility>

#include "pipeline_timing_internal.hpp"

namespace meridian::local {
namespace {

using PoseKey = gtsam::Key;
using VelocityKey = gtsam::Key;
using BiasKey = gtsam::Key;

struct NavigationKeys {
  PoseKey pose;
  VelocityKey velocity;
};

[[nodiscard]] NavigationKeys navigationKeys(std::size_t index) {
  return NavigationKeys{gtsam::Symbol('x', index), gtsam::Symbol('v', index)};
}

[[nodiscard]] BiasKey biasKey() {
  return gtsam::Symbol('b', 0U);
}

[[nodiscard]] MotionInitializationError error(MotionInitializationErrorCode code,
                                              std::string detail,
                                              std::optional<std::size_t> segment = std::nullopt) {
  return MotionInitializationError{code, segment, std::move(detail)};
}

[[nodiscard]] bool finitePositive(double value) {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool validConfig(const MotionInitializerConfig& config) {
  return config.minimum_segments >= 2U && config.minimum_segments <= config.maximum_segments &&
         config.maximum_imu_knots >= config.maximum_segments + 1U &&
         config.minimum_support.nanoseconds > 0 &&
         config.minimum_support <= config.maximum_support &&
         config.maximum_time_uncertainty.nanoseconds >= 0 &&
         config.maximum_raw_imu_gap.nanoseconds > 0 &&
         finitePositive(config.minimum_rotation_excitation_rad) &&
         finitePositive(config.minimum_acceleration_excitation_mps2) &&
         finitePositive(config.information_basis_orthonormal_tolerance) &&
         config.information_zero_tolerance >= 0.0 &&
         std::isfinite(config.information_zero_tolerance) &&
         finitePositive(config.hessian_absolute_rank_tolerance) &&
         finitePositive(config.hessian_relative_rank_tolerance) &&
         finitePositive(config.maximum_supported_hessian_condition) &&
         finitePositive(config.covariance_symmetry_relative_tolerance) &&
         std::isfinite(config.minimum_covariance_eigenvalue) &&
         finitePositive(config.position_gauge_sigma_m) &&
         finitePositive(config.yaw_gauge_sigma_rad) &&
         finitePositive(config.maximum_bias_prior_mahalanobis) &&
         config.minimum_imu_conditioning_covariance_inflation >= 1.0 &&
         std::isfinite(config.minimum_imu_conditioning_covariance_inflation) &&
         config.maximum_prior_resolved_accel_tilt_modes <= 2U &&
         config.holdout_lidar_segments < config.minimum_segments &&
         finitePositive(config.maximum_lidar_mean_squared_whitened_residual) &&
         finitePositive(config.maximum_imu_mean_squared_whitened_residual) &&
         finitePositive(config.maximum_reduced_chi_square) &&
         finitePositive(config.maximum_holdout_mean_squared_whitened_residual) &&
         std::isfinite(config.maximum_covariance_residual_inflation) &&
         config.maximum_covariance_residual_inflation >= 1.0 &&
         finitePositive(config.minimum_orientation_variance_rad2) &&
         finitePositive(config.minimum_velocity_variance_m2ps2) &&
         finitePositive(config.minimum_position_variance_m2) &&
         finitePositive(config.minimum_gyro_bias_variance_rad2ps2) &&
         finitePositive(config.minimum_accel_bias_variance_m2ps4) &&
         config.maximum_solver_iterations > 0U &&
         finitePositive(config.solver_relative_error_tolerance) &&
         finitePositive(config.solver_absolute_error_tolerance);
}

[[nodiscard]] bool validNoise(const MotionImuNoise& noise) {
  return noise.gravity_odom.allFinite() && finitePositive(noise.gravity_odom.norm()) &&
         finitePositive(noise.accelerometer_noise_density_mps2_sqrt_hz) &&
         finitePositive(noise.gyroscope_noise_density_radps_sqrt_hz) &&
         finitePositive(noise.integration_noise_density) &&
         noise.accelerometer_bias_prior_mean_mps2.allFinite() &&
         noise.gyroscope_bias_prior_mean_radps.allFinite() &&
         finitePositive(noise.accelerometer_bias_prior_sigma_mps2) &&
         finitePositive(noise.gyroscope_bias_prior_sigma_radps);
}

[[nodiscard]] bool validInformation(const core::RankAwareInformation& information,
                                    const MotionInitializerConfig& config) {
  if (!information.finite() || information.rank == 0U || information.rank > 6U ||
      information.tangent != core::PoseTangentConvention::RightTranslationFirst) {
    return false;
  }
  const Eigen::Matrix<double, 6, 6> orthogonality =
      information.basis.transpose() * information.basis - Eigen::Matrix<double, 6, 6>::Identity();
  if (orthogonality.cwiseAbs().maxCoeff() > config.information_basis_orthonormal_tolerance) {
    return false;
  }
  for (std::size_t index = 0U; index < 6U; ++index) {
    const double eigenvalue = information.eigenvalues(static_cast<Eigen::Index>(index));
    if (index < information.rank) {
      if (!finitePositive(eigenvalue)) {
        return false;
      }
    } else if (std::abs(eigenvalue) > config.information_zero_tolerance) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool validLineage(const core::ObservationLineage& lineage) {
  return lineage.id.valid() && core::validateLineage(lineage) == core::LineageValidationError::None;
}

[[nodiscard]] std::set<std::uint64_t> primaryMeasurementIds(const core::ObservationLineage& lineage,
                                                            bool* only_whole_measurements) {
  std::set<std::uint64_t> ids;
  *only_whole_measurements = true;
  for (const core::ObservationUsage& usage : lineage.usage) {
    if (usage.role != core::ObservationRole::PrimaryResidual) {
      continue;
    }
    if (!std::holds_alternative<core::MeasurementId>(usage.slice.root) ||
        usage.slice.kind != core::SliceKind::Whole) {
      *only_whole_measurements = false;
      continue;
    }
    ids.insert(std::get<core::MeasurementId>(usage.slice.root).value());
  }
  return ids;
}

[[nodiscard]] bool exactImuLineageCoverage(const ImuInterval& interval,
                                           const core::ObservationLineage& lineage) {
  bool only_whole_measurements = false;
  const std::set<std::uint64_t> primary = primaryMeasurementIds(lineage, &only_whole_measurements);
  if (!only_whole_measurements || primary.empty() ||
      primary.size() != interval.raw_measurements.size()) {
    return false;
  }
  std::set<std::uint64_t> raw;
  for (core::MeasurementId id : interval.raw_measurements) {
    if (!id.valid() || !raw.insert(id.value()).second) {
      return false;
    }
  }
  if (raw != primary) {
    return false;
  }
  for (const InterpolatedImuSample& knot : interval.knots) {
    if (!knot.left_source.valid() || !knot.right_source.valid() ||
        !raw.contains(knot.left_source.value()) || !raw.contains(knot.right_source.value())) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool lidarLineageHasPrimaryObservation(const core::ObservationLineage& lineage) {
  return std::any_of(lineage.usage.begin(), lineage.usage.end(),
                     [](const core::ObservationUsage& usage) {
                       return usage.role == core::ObservationRole::PrimaryResidual;
                     });
}

[[nodiscard]] bool lidarLineageConditionsOnExactImu(const ImuInterval& interval,
                                                    const core::ObservationLineage& lineage) {
  return std::all_of(interval.raw_measurements.begin(), interval.raw_measurements.end(),
                     [&](core::MeasurementId measurement) {
                       return std::any_of(
                           lineage.usage.begin(), lineage.usage.end(),
                           [&](const core::ObservationUsage& usage) {
                             return usage.role == core::ObservationRole::ConditioningOnly &&
                                    usage.slice.kind == core::SliceKind::Whole &&
                                    std::holds_alternative<core::MeasurementId>(usage.slice.root) &&
                                    std::get<core::MeasurementId>(usage.slice.root) == measurement;
                           });
                     });
}

[[nodiscard]] gtsam::Pose3 toGtsamPose(const core::Pose3d& pose) {
  return gtsam::Pose3(gtsam::Rot3(pose.so3().matrix()), pose.translation());
}

[[nodiscard]] core::Pose3d fromGtsamPose(const gtsam::Pose3& pose) {
  return core::Pose3d(Sophus::SO3d(pose.rotation().matrix()), pose.translation());
}

[[nodiscard]] Eigen::Matrix<double, 6, 1> meridianPoseError(const gtsam::Pose3& measured_start_end,
                                                            const gtsam::Pose3& T_odom_start,
                                                            const gtsam::Pose3& T_odom_end) {
  const core::Pose3d measured = fromGtsamPose(measured_start_end);
  const core::Pose3d start = fromGtsamPose(T_odom_start);
  const core::Pose3d end = fromGtsamPose(T_odom_end);
  return (measured.inverse() * start.inverse() * end).log();
}

class RankAwareRelativePoseFactor final
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
public:
  using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>;

  RankAwareRelativePoseFactor(PoseKey start_key, PoseKey end_key, core::Pose3d measurement,
                              const core::RankAwareInformation& information)
      : Base(gtsam::noiseModel::Unit::Create(information.rank), start_key, end_key),
        measured_start_end_(toGtsamPose(measurement)),
        square_root_projection_(static_cast<Eigen::Index>(information.rank), 6) {
    for (std::size_t row = 0U; row < information.rank; ++row) {
      square_root_projection_.row(static_cast<Eigen::Index>(row)) =
          std::sqrt(information.eigenvalues(static_cast<Eigen::Index>(row))) *
          information.basis.col(static_cast<Eigen::Index>(row)).transpose();
    }
  }

  gtsam::Vector evaluateError(
      const gtsam::Pose3& start, const gtsam::Pose3& end,
      boost::optional<gtsam::Matrix&> start_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> end_jacobian = boost::none) const override {
    const auto residual = [&](const gtsam::Pose3& from, const gtsam::Pose3& to) -> gtsam::Vector {
      return square_root_projection_ * meridianPoseError(measured_start_end_, from, to);
    };
    const gtsam::Vector result = residual(start, end);
    constexpr double kDerivativeStep = 1.0e-6;
    if (start_jacobian) {
      *start_jacobian = gtsam::Matrix::Zero(result.size(), 6);
      for (Eigen::Index column = 0; column < 6; ++column) {
        gtsam::Vector6 delta = gtsam::Vector6::Zero();
        delta(column) = kDerivativeStep;
        start_jacobian->col(column) = (residual(start.compose(gtsam::Pose3::Expmap(delta)), end) -
                                       residual(start.compose(gtsam::Pose3::Expmap(-delta)), end)) /
                                      (2.0 * kDerivativeStep);
      }
    }
    if (end_jacobian) {
      *end_jacobian = gtsam::Matrix::Zero(result.size(), 6);
      for (Eigen::Index column = 0; column < 6; ++column) {
        gtsam::Vector6 delta = gtsam::Vector6::Zero();
        delta(column) = kDerivativeStep;
        end_jacobian->col(column) = (residual(start, end.compose(gtsam::Pose3::Expmap(delta))) -
                                     residual(start, end.compose(gtsam::Pose3::Expmap(-delta)))) /
                                    (2.0 * kDerivativeStep);
      }
    }
    return result;
  }

private:
  gtsam::Pose3 measured_start_end_;
  Eigen::Matrix<double, Eigen::Dynamic, 6> square_root_projection_;
};

class YawGaugeFactor final : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
public:
  using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;

  YawGaugeFactor(PoseKey key, gtsam::Rot3 reference_rotation, const Eigen::Vector3d& gravity_odom,
                 double sigma)
      : Base(gtsam::noiseModel::Isotropic::Sigma(1, sigma), key),
        reference_rotation_(std::move(reference_rotation)),
        local_gravity_axis_(reference_rotation_.unrotate(gravity_odom.normalized())) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& pose, boost::optional<gtsam::Matrix&> jacobian =
                                                            boost::none) const override {
    const auto residual = [&](const gtsam::Pose3& evaluated) {
      gtsam::Vector1 value;
      value << local_gravity_axis_.dot(
          gtsam::Rot3::Logmap(reference_rotation_.between(evaluated.rotation())));
      return value;
    };
    const gtsam::Vector1 result = residual(pose);
    if (jacobian) {
      constexpr double kDerivativeStep = 1.0e-6;
      *jacobian = gtsam::Matrix::Zero(1, 6);
      for (Eigen::Index column = 0; column < 6; ++column) {
        gtsam::Vector6 delta = gtsam::Vector6::Zero();
        delta(column) = kDerivativeStep;
        jacobian->col(column) = (residual(pose.compose(gtsam::Pose3::Expmap(delta))) -
                                 residual(pose.compose(gtsam::Pose3::Expmap(-delta)))) /
                                (2.0 * kDerivativeStep);
      }
    }
    return result;
  }

private:
  gtsam::Rot3 reference_rotation_;
  Eigen::Vector3d local_gravity_axis_{Eigen::Vector3d::UnitZ()};
};

class AccelerometerBiasPriorFactor final
    : public gtsam::NoiseModelFactor1<gtsam::imuBias::ConstantBias> {
public:
  using Base = gtsam::NoiseModelFactor1<gtsam::imuBias::ConstantBias>;

  AccelerometerBiasPriorFactor(BiasKey key, Eigen::Vector3d mean, double sigma)
      : Base(gtsam::noiseModel::Isotropic::Sigma(3, sigma), key), mean_(std::move(mean)) {}

  gtsam::Vector evaluateError(
      const gtsam::imuBias::ConstantBias& bias,
      boost::optional<gtsam::Matrix&> jacobian = boost::none) const override {
    if (jacobian) {
      *jacobian = gtsam::Matrix::Zero(3, 6);
      jacobian->block<3, 3>(0, 0).setIdentity();
    }
    return bias.accelerometer() - mean_;
  }

private:
  Eigen::Vector3d mean_{Eigen::Vector3d::Zero()};
};

class GyroscopeBiasPriorFactor final
    : public gtsam::NoiseModelFactor1<gtsam::imuBias::ConstantBias> {
public:
  using Base = gtsam::NoiseModelFactor1<gtsam::imuBias::ConstantBias>;

  GyroscopeBiasPriorFactor(BiasKey key, Eigen::Vector3d mean, double sigma)
      : Base(gtsam::noiseModel::Isotropic::Sigma(3, sigma), key), mean_(std::move(mean)) {}

  gtsam::Vector evaluateError(
      const gtsam::imuBias::ConstantBias& bias,
      boost::optional<gtsam::Matrix&> jacobian = boost::none) const override {
    if (jacobian) {
      *jacobian = gtsam::Matrix::Zero(3, 6);
      jacobian->block<3, 3>(0, 3).setIdentity();
    }
    return bias.gyroscope() - mean_;
  }

private:
  Eigen::Vector3d mean_{Eigen::Vector3d::Zero()};
};

struct InitialGuess {
  std::vector<core::Pose3d> bootstrap_poses;
  std::vector<core::Pose3d> odom_poses;
  std::vector<Eigen::Vector3d> velocities_odom;
  Eigen::Vector3d gyro_bias{Eigen::Vector3d::Zero()};
  Eigen::Vector3d accel_bias{Eigen::Vector3d::Zero()};
  core::Pose3d T_odom_bootstrap;
  double rotation_excitation_rad{};
  double acceleration_excitation_mps2{};
};

[[nodiscard]] Eigen::Vector3d intervalMeanSpecificForce(const ImuInterval& interval) {
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  double duration_sum = 0.0;
  for (std::size_t index = 1U; index < interval.knots.size(); ++index) {
    const auto& previous = interval.knots[index - 1U];
    const auto& current = interval.knots[index];
    const double dt = static_cast<double>((current.time - previous.time).nanoseconds) * 1.0e-9;
    sum += 0.5 * (previous.specific_force_mps2 + current.specific_force_mps2) * dt;
    duration_sum += dt;
  }
  return sum / duration_sum;
}

[[nodiscard]] Eigen::Vector3d intervalMeanAngularRate(const ImuInterval& interval) {
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  double duration_sum = 0.0;
  for (std::size_t index = 1U; index < interval.knots.size(); ++index) {
    const auto& previous = interval.knots[index - 1U];
    const auto& current = interval.knots[index];
    const double dt = static_cast<double>((current.time - previous.time).nanoseconds) * 1.0e-9;
    sum += 0.5 * (previous.angular_velocity_radps + current.angular_velocity_radps) * dt;
    duration_sum += dt;
  }
  return sum / duration_sum;
}

[[nodiscard]] InitialGuess buildInitialGuess(const MotionInitializationRequest& request) {
  const std::size_t state_count = request.segments.size() + 1U;
  InitialGuess guess;
  guess.bootstrap_poses.reserve(state_count);
  guess.bootstrap_poses.emplace_back();
  for (const MotionInitializationSegment& segment : request.segments) {
    guess.bootstrap_poses.push_back(guess.bootstrap_poses.back() *
                                    segment.lidar.T_imu_start_imu_end);
  }

  std::vector<double> interval_seconds(request.segments.size());
  std::vector<Eigen::Vector3d> interval_velocities(request.segments.size());
  for (std::size_t index = 0U; index < request.segments.size(); ++index) {
    interval_seconds[index] = static_cast<double>((request.segments[index].lidar.end_time -
                                                   request.segments[index].lidar.start_time)
                                                      .nanoseconds) *
                              1.0e-9;
    interval_velocities[index] = (guess.bootstrap_poses[index + 1U].translation() -
                                  guess.bootstrap_poses[index].translation()) /
                                 interval_seconds[index];
    guess.rotation_excitation_rad = std::max(guess.rotation_excitation_rad,
                                             guess.bootstrap_poses[index + 1U].so3().log().norm());
  }

  std::vector<Eigen::Vector3d> accelerations(state_count, Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector3d> interior_accelerations;
  interior_accelerations.reserve(state_count - 2U);
  for (std::size_t index = 1U; index + 1U < state_count; ++index) {
    accelerations[index] = 2.0 * (interval_velocities[index] - interval_velocities[index - 1U]) /
                           (interval_seconds[index] + interval_seconds[index - 1U]);
    interior_accelerations.push_back(accelerations[index]);
  }
  accelerations.front() = accelerations[1U];
  accelerations.back() = accelerations[state_count - 2U];

  Eigen::Vector3d mean_acceleration = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& acceleration : interior_accelerations) {
    mean_acceleration += acceleration;
  }
  mean_acceleration /= static_cast<double>(interior_accelerations.size());
  double acceleration_variation = 0.0;
  for (const Eigen::Vector3d& acceleration : interior_accelerations) {
    acceleration_variation += (acceleration - mean_acceleration).squaredNorm();
  }
  guess.acceleration_excitation_mps2 =
      std::sqrt(acceleration_variation / static_cast<double>(interior_accelerations.size()));

  Eigen::Vector3d gravity_bootstrap = Eigen::Vector3d::Zero();
  for (std::size_t index = 1U; index + 1U < state_count; ++index) {
    const Eigen::Vector3d force =
        0.5 * (intervalMeanSpecificForce(request.segments[index - 1U].imu) +
               intervalMeanSpecificForce(request.segments[index].imu));
    gravity_bootstrap += accelerations[index] - guess.bootstrap_poses[index].so3() * force;
  }
  gravity_bootstrap /= static_cast<double>(state_count - 2U);

  Eigen::Quaterniond gravity_alignment = Eigen::Quaterniond::FromTwoVectors(
      gravity_bootstrap.normalized(), request.imu_noise.gravity_odom.normalized());
  Eigen::Matrix3d R_odom_bootstrap = gravity_alignment.normalized().toRotationMatrix();
  const Eigen::Vector3d horizontal_x = R_odom_bootstrap.col(0);
  const double gauge_yaw = std::atan2(horizontal_x.y(), horizontal_x.x());
  R_odom_bootstrap =
      Eigen::AngleAxisd(-gauge_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R_odom_bootstrap;
  guess.T_odom_bootstrap = core::Pose3d(Sophus::SO3d(R_odom_bootstrap), Eigen::Vector3d::Zero());

  guess.odom_poses.reserve(state_count);
  for (const core::Pose3d& pose : guess.bootstrap_poses) {
    guess.odom_poses.push_back(guess.T_odom_bootstrap * pose);
  }
  guess.velocities_odom.resize(state_count);
  guess.velocities_odom.front() = R_odom_bootstrap * interval_velocities.front();
  guess.velocities_odom.back() = R_odom_bootstrap * interval_velocities.back();
  for (std::size_t index = 1U; index + 1U < state_count; ++index) {
    guess.velocities_odom[index] = R_odom_bootstrap *
                                   (interval_seconds[index] * interval_velocities[index - 1U] +
                                    interval_seconds[index - 1U] * interval_velocities[index]) /
                                   (interval_seconds[index - 1U] + interval_seconds[index]);
  }

  double gyro_weight = 0.0;
  for (std::size_t index = 0U; index < request.segments.size(); ++index) {
    const double dt = interval_seconds[index];
    guess.gyro_bias += (intervalMeanAngularRate(request.segments[index].imu) -
                        request.segments[index].lidar.T_imu_start_imu_end.so3().log() / dt) *
                       dt;
    gyro_weight += dt;
  }
  guess.gyro_bias /= gyro_weight;

  for (std::size_t index = 1U; index + 1U < state_count; ++index) {
    const Eigen::Vector3d force =
        0.5 * (intervalMeanSpecificForce(request.segments[index - 1U].imu) +
               intervalMeanSpecificForce(request.segments[index].imu));
    const Eigen::Vector3d acceleration_odom = R_odom_bootstrap * accelerations[index];
    guess.accel_bias += force - guess.odom_poses[index].so3().inverse() *
                                    (acceleration_odom - request.imu_noise.gravity_odom);
  }
  guess.accel_bias /= static_cast<double>(state_count - 2U);
  return guess;
}

[[nodiscard]] boost::shared_ptr<gtsam::PreintegrationParams> preintegrationParams(
    const MotionImuNoise& noise) {
  auto params = boost::make_shared<gtsam::PreintegrationParams>(noise.gravity_odom);
  params->accelerometerCovariance =
      Eigen::Matrix3d::Identity() * std::pow(noise.accelerometer_noise_density_mps2_sqrt_hz, 2);
  params->gyroscopeCovariance =
      Eigen::Matrix3d::Identity() * std::pow(noise.gyroscope_noise_density_radps_sqrt_hz, 2);
  params->integrationCovariance =
      Eigen::Matrix3d::Identity() * std::pow(noise.integration_noise_density, 2);
  return params;
}

void integrateInterval(const ImuInterval& interval,
                       gtsam::PreintegratedImuMeasurements* preintegrated) {
  for (std::size_t index = 1U; index < interval.knots.size(); ++index) {
    const InterpolatedImuSample& previous = interval.knots[index - 1U];
    const InterpolatedImuSample& current = interval.knots[index];
    const double dt = static_cast<double>((current.time - previous.time).nanoseconds) * 1.0e-9;
    preintegrated->integrateMeasurement(
        0.5 * (previous.specific_force_mps2 + current.specific_force_mps2),
        0.5 * (previous.angular_velocity_radps + current.angular_velocity_radps), dt);
  }
}

struct HessianDiagnostics {
  std::size_t dimension{};
  std::size_t rank{};
  double supported_condition{std::numeric_limits<double>::infinity()};
  std::array<double, 6> smallest_scaled_eigenvalues{
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  double rank_threshold{};
};

[[nodiscard]] HessianDiagnostics inspectHessian(const gtsam::NonlinearFactorGraph& graph,
                                                const gtsam::Values& values,
                                                const gtsam::Ordering& ordering,
                                                const MotionInitializerConfig& config) {
  HessianDiagnostics result;
  const auto gaussian = graph.linearize(values);
  const auto hessian_and_gradient = gaussian->hessian(ordering);
  const gtsam::Matrix hessian =
      0.5 * (hessian_and_gradient.first + hessian_and_gradient.first.transpose());
  result.dimension = static_cast<std::size_t>(hessian.rows());
  if (hessian.rows() == 0 || hessian.rows() != hessian.cols() || !hessian.allFinite()) {
    return result;
  }
  // Pose, velocity, and bias coordinates have different units, while the
  // deliberately tight gauge is much stronger than any sensor factor. Rank
  // testing the raw Hessian would therefore let the gauge's scale hide valid
  // sensor directions. Symmetric Jacobi scaling is a change of coordinates,
  // so it preserves rank while making the threshold and condition diagnostic
  // dimensionless.
  gtsam::Vector inverse_sqrt_diagonal(hessian.rows());
  for (Eigen::Index index = 0; index < hessian.rows(); ++index) {
    const double diagonal = std::abs(hessian(index, index));
    inverse_sqrt_diagonal(index) =
        1.0 / std::sqrt(std::max(diagonal, config.hessian_absolute_rank_tolerance));
  }
  const gtsam::Matrix scaled =
      inverse_sqrt_diagonal.asDiagonal() * hessian * inverse_sqrt_diagonal.asDiagonal();
  Eigen::SelfAdjointEigenSolver<gtsam::Matrix> eigensolver(scaled, Eigen::EigenvaluesOnly);
  if (eigensolver.info() != Eigen::Success || !eigensolver.eigenvalues().allFinite()) {
    return result;
  }
  const double largest = std::max(0.0, eigensolver.eigenvalues().maxCoeff());
  const double threshold = std::max(config.hessian_absolute_rank_tolerance,
                                    config.hessian_relative_rank_tolerance * largest);
  result.rank_threshold = threshold;
  for (Eigen::Index index = 0;
       index <
       std::min<Eigen::Index>(eigensolver.eigenvalues().size(),
                              static_cast<Eigen::Index>(result.smallest_scaled_eigenvalues.size()));
       ++index) {
    result.smallest_scaled_eigenvalues.at(static_cast<std::size_t>(index)) =
        eigensolver.eigenvalues()(index);
  }
  double smallest_supported = std::numeric_limits<double>::infinity();
  for (Eigen::Index index = 0; index < eigensolver.eigenvalues().size(); ++index) {
    const double eigenvalue = eigensolver.eigenvalues()(index);
    if (eigenvalue > threshold) {
      ++result.rank;
      smallest_supported = std::min(smallest_supported, eigenvalue);
    }
  }
  if (result.rank > 0U) {
    result.supported_condition = largest / smallest_supported;
  }
  return result;
}

[[nodiscard]] NavigationCovariance finalNavigationCovariance(
    const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values,
    const NavigationKeys& final_keys) {
  gtsam::Marginals marginals(graph, values, gtsam::Marginals::QR);
  const gtsam::KeyVector requested{final_keys.pose, final_keys.velocity, biasKey()};
  const gtsam::JointMarginal joint = marginals.jointMarginalCovariance(requested);

  Eigen::Matrix<double, 15, 15> gtsam_covariance = Eigen::Matrix<double, 15, 15>::Zero();
  gtsam_covariance.block<6, 6>(0, 0) = joint(final_keys.pose, final_keys.pose);
  gtsam_covariance.block<6, 3>(0, 6) = joint(final_keys.pose, final_keys.velocity);
  gtsam_covariance.block<6, 6>(0, 9) = joint(final_keys.pose, biasKey());
  gtsam_covariance.block<3, 6>(6, 0) = joint(final_keys.velocity, final_keys.pose);
  gtsam_covariance.block<3, 3>(6, 6) = joint(final_keys.velocity, final_keys.velocity);
  gtsam_covariance.block<3, 6>(6, 9) = joint(final_keys.velocity, biasKey());
  gtsam_covariance.block<6, 6>(9, 0) = joint(biasKey(), final_keys.pose);
  gtsam_covariance.block<6, 3>(9, 6) = joint(biasKey(), final_keys.velocity);
  gtsam_covariance.block<6, 6>(9, 9) = joint(biasKey(), biasKey());

  // GTSAM joint order is [R,P,V,Ba,Bg]. Meridian's binding is
  // [R,V,P,Bg,Ba].
  constexpr std::array<Eigen::Index, 15> kGtsamToMeridian{0, 1,  2,  6,  7, 8,  3, 4,
                                                          5, 12, 13, 14, 9, 10, 11};
  NavigationCovariance converted;
  for (Eigen::Index row = 0; row < 15; ++row) {
    for (Eigen::Index column = 0; column < 15; ++column) {
      converted.matrix(kGtsamToMeridian.at(row), kGtsamToMeridian.at(column)) =
          gtsam_covariance(row, column);
    }
  }
  converted.matrix = 0.5 * (converted.matrix + converted.matrix.transpose());
  return converted;
}

[[nodiscard]] bool validCovariance(const NavigationCovariance& covariance,
                                   const MotionInitializerConfig& config) {
  if (covariance.order != NavigationCovarianceOrder::RotationVelocityPositionGyroBiasAccelBias ||
      !covariance.matrix.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, covariance.matrix.cwiseAbs().maxCoeff());
  if ((covariance.matrix - covariance.matrix.transpose()).cwiseAbs().maxCoeff() >
      config.covariance_symmetry_relative_tolerance * scale) {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> eigensolver(covariance.matrix,
                                                                           Eigen::EigenvaluesOnly);
  return eigensolver.info() == Eigen::Success && eigensolver.eigenvalues().allFinite() &&
         eigensolver.eigenvalues().minCoeff() >= config.minimum_covariance_eigenvalue * scale;
}

void calibrateCovariance(NavigationCovariance* covariance, double residual_inflation,
                         const MotionInitializerConfig& config) {
  covariance->matrix *= residual_inflation;
  const std::array<std::pair<Eigen::Index, double>, 5> blocks{{
      {0, config.minimum_orientation_variance_rad2},
      {3, config.minimum_velocity_variance_m2ps2},
      {6, config.minimum_position_variance_m2},
      {9, config.minimum_gyro_bias_variance_rad2ps2},
      {12, config.minimum_accel_bias_variance_m2ps4},
  }};
  for (const auto& [offset, floor] : blocks) {
    Eigen::Matrix3d marginal = covariance->matrix.block<3, 3>(offset, offset);
    marginal = 0.5 * (marginal + marginal.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(marginal, Eigen::EigenvaluesOnly);
    const double minimum = eigensolver.info() == Eigen::Success
                               ? eigensolver.eigenvalues().minCoeff()
                               : -std::numeric_limits<double>::infinity();
    covariance->matrix.block<3, 3>(offset, offset).diagonal().array() +=
        std::max(0.0, floor - minimum);
  }
  covariance->matrix = 0.5 * (covariance->matrix + covariance->matrix.transpose());
}

[[nodiscard]] double relativePoseCost(const MotionInitializationSegment& segment,
                                      const gtsam::Pose3& start, const gtsam::Pose3& end) {
  const Eigen::Matrix<double, 6, 1> residual =
      meridianPoseError(toGtsamPose(segment.lidar.T_imu_start_imu_end), start, end);
  double squared_norm = 0.0;
  for (std::size_t index = 0U; index < segment.lidar.information.rank; ++index) {
    const Eigen::Index column = static_cast<Eigen::Index>(index);
    const double projected = segment.lidar.information.basis.col(column).dot(residual);
    squared_norm += segment.lidar.information.eigenvalues(column) * projected * projected;
  }
  return 0.5 * squared_norm;
}

}  // namespace

MotionInitializer::MotionInitializer(MotionInitializerConfig config)
    : MotionInitializer(std::move(config), {}) {}

MotionInitializer::MotionInitializer(MotionInitializerConfig config,
                                     std::shared_ptr<LocalPipelineTimingRecorder> timing)
    : config_(std::move(config)), timing_(std::move(timing)) {}

core::Result<MotionInitialization, MotionInitializationError> MotionInitializer::initialize(
    const MotionInitializationRequest& request) const {
  using Result = core::Result<MotionInitialization, MotionInitializationError>;
  detail::LocalPipelineTimingScope solve_timing(
      timing_, LocalPipelineTimingStage::MotionBatchSolveRefinement);
  if (!validConfig(config_) || !validNoise(request.imu_noise)) {
    return Result::failure(
        error(MotionInitializationErrorCode::InvalidConfig,
              "initializer bounds, gates, gauge noise, or IMU noise are invalid"));
  }
  if (request.segments.size() < config_.minimum_segments ||
      request.segments.size() > config_.maximum_segments) {
    return Result::failure(
        error(MotionInitializationErrorCode::SegmentCountOutOfBounds,
              "motion initialization segment count is outside the configured bounded batch"));
  }

  const core::FusionTime first_time = request.segments.front().lidar.start_time;
  const core::FusionTime final_time = request.segments.back().lidar.end_time;
  const core::Duration total_support = final_time - first_time;
  if (total_support < config_.minimum_support || total_support > config_.maximum_support) {
    return Result::failure(
        error(MotionInitializationErrorCode::SupportDurationOutOfBounds,
              "motion initialization duration is outside the configured bounded window"));
  }

  std::size_t imu_knots = 0U;
  std::set<std::uint64_t> lineage_ids;
  for (std::size_t index = 0U; index < request.segments.size(); ++index) {
    const MotionInitializationSegment& segment = request.segments[index];
    const LidarBootstrapIncrement& lidar = segment.lidar;
    const ImuInterval& imu = segment.imu;
    if (lidar.start_time >= lidar.end_time ||
        (index > 0U && lidar.start_time != request.segments[index - 1U].lidar.end_time)) {
      return Result::failure(error(
          MotionInitializationErrorCode::NonContiguousSupport,
          "LiDAR bootstrap increments must form one exact, contiguous state sequence", index));
    }
    if (!imu.support.valid() || imu.knots.size() < 2U || imu.support.start != lidar.start_time ||
        imu.support.end != lidar.end_time || imu.knots.front().time != lidar.start_time ||
        imu.knots.back().time != lidar.end_time) {
      return Result::failure(error(
          MotionInitializationErrorCode::InexactImuSupport,
          "each IMU interval and both boundary knots must exactly equal its LiDAR state times",
          index));
    }
    imu_knots += imu.knots.size();
    if (imu_knots > config_.maximum_imu_knots || imu.contains_saturation ||
        imu.inferred_missing_ticks != 0U ||
        imu.maximum_time_uncertainty > config_.maximum_time_uncertainty ||
        imu.maximum_raw_gap > config_.maximum_raw_imu_gap) {
      return Result::failure(error(MotionInitializationErrorCode::InvalidImuSupport,
                                   "IMU support exceeds a bound or contains saturation, inferred "
                                   "ticks, clock uncertainty, or a raw gap",
                                   index));
    }
    for (std::size_t knot_index = 0U; knot_index < imu.knots.size(); ++knot_index) {
      const InterpolatedImuSample& knot = imu.knots[knot_index];
      if (!knot.specific_force_mps2.allFinite() || !knot.angular_velocity_radps.allFinite() ||
          (knot_index > 0U && knot.time <= imu.knots[knot_index - 1U].time)) {
        return Result::failure(error(MotionInitializationErrorCode::InvalidImuSupport,
                                     "IMU knots must be strictly ordered and finite", index));
      }
    }
    if (!lidar.T_imu_start_imu_end.matrix().allFinite()) {
      return Result::failure(error(MotionInitializationErrorCode::InvalidRelativePose,
                                   "LiDAR bootstrap relative IMU pose is non-finite", index));
    }
    if (!validInformation(lidar.information, config_)) {
      return Result::failure(error(MotionInitializationErrorCode::InvalidRankAwareInformation,
                                   "LiDAR bootstrap information must have an orthonormal basis, "
                                   "positive active eigenvalues, and zero inactive eigenvalues",
                                   index));
    }
    if (!std::isfinite(lidar.imu_conditioning_covariance_inflation) ||
        lidar.imu_conditioning_covariance_inflation <
            config_.minimum_imu_conditioning_covariance_inflation ||
        !std::isfinite(lidar.applied_covariance_inflation) ||
        lidar.applied_covariance_inflation < lidar.imu_conditioning_covariance_inflation) {
      return Result::failure(error(MotionInitializationErrorCode::InvalidRankAwareInformation,
                                   "IMU-conditioned LiDAR information lacks the configured "
                                   "conservative covariance inflation",
                                   index));
    }
    if (!validLineage(lidar.lineage) || !validLineage(segment.imu_lineage) ||
        !lidarLineageHasPrimaryObservation(lidar.lineage)) {
      return Result::failure(error(MotionInitializationErrorCode::InvalidLineage,
                                   "LiDAR and IMU lineages must be valid and the LiDAR factor must "
                                   "declare a primary observation",
                                   index));
    }
    if (!exactImuLineageCoverage(imu, segment.imu_lineage) ||
        !lidarLineageConditionsOnExactImu(imu, lidar.lineage) ||
        core::lineagesAreIndependent(lidar.lineage, segment.imu_lineage)) {
      return Result::failure(
          error(MotionInitializationErrorCode::LineageSupportMismatch,
                "IMU lineage must own every raw primary residual while the "
                "deskewed LiDAR lineage records the same samples as conditioning data",
                index));
    }
    if (!lineage_ids.insert(lidar.lineage.id.value()).second ||
        !lineage_ids.insert(segment.imu_lineage.id.value()).second) {
      return Result::failure(error(MotionInitializationErrorCode::InvalidLineage,
                                   "each factor input must carry a distinct lineage identity",
                                   index));
    }
  }

  const InitialGuess initial = buildInitialGuess(request);
  if (!initial.T_odom_bootstrap.matrix().allFinite() || !initial.gyro_bias.allFinite() ||
      !initial.accel_bias.allFinite()) {
    return Result::failure(error(MotionInitializationErrorCode::NonFiniteSolution,
                                 "motion initialization seed is non-finite"));
  }
  const bool rotation_excited =
      initial.rotation_excitation_rad >= config_.minimum_rotation_excitation_rad;
  const bool acceleration_excited =
      initial.acceleration_excitation_mps2 >= config_.minimum_acceleration_excitation_mps2;
  if (!rotation_excited && !acceleration_excited) {
    std::ostringstream detail;
    detail << "motion does not pass either physical excitation pre-gate (rotation="
           << initial.rotation_excitation_rad
           << " rad, acceleration=" << initial.acceleration_excitation_mps2 << " m/s^2)";
    return Result::failure(
        error(MotionInitializationErrorCode::InsufficientExcitation, detail.str()));
  }

  try {
    const bool commit_candidate =
        request.pass == MotionInitializationRequest::Pass::FullDeskewCommitCandidate;
    const std::size_t holdout_segments = commit_candidate ? config_.holdout_lidar_segments : 0U;
    const std::size_t lidar_segments_in_objective = request.segments.size() - holdout_segments;
    const std::size_t state_count = request.segments.size() + 1U;
    const auto preintegration_params = preintegrationParams(request.imu_noise);
    const gtsam::imuBias::ConstantBias initial_bias(initial.accel_bias, initial.gyro_bias);
    gtsam::NonlinearFactorGraph lidar_graph;
    gtsam::NonlinearFactorGraph imu_graph;
    gtsam::NonlinearFactorGraph data_graph;
    gtsam::NonlinearFactorGraph calibrated_data_graph;
    gtsam::NonlinearFactorGraph gauge_graph;
    gtsam::NonlinearFactorGraph bias_prior_graph;
    gtsam::NonlinearFactorGraph full_graph;
    gtsam::Values values;
    gtsam::Ordering ordering;

    for (std::size_t index = 0U; index < state_count; ++index) {
      const NavigationKeys keys = navigationKeys(index);
      values.insert(keys.pose, toGtsamPose(initial.odom_poses[index]));
      values.insert(keys.velocity, initial.velocities_odom[index]);
      ordering.push_back(keys.pose);
      ordering.push_back(keys.velocity);
    }
    values.insert(biasKey(), initial_bias);
    ordering.push_back(biasKey());

    std::size_t lidar_residual_dimension = 0U;
    for (std::size_t index = 0U; index < request.segments.size(); ++index) {
      const NavigationKeys previous = navigationKeys(index);
      const NavigationKeys current = navigationKeys(index + 1U);
      const MotionInitializationSegment& segment = request.segments[index];
      if (index < lidar_segments_in_objective) {
        auto factor = boost::make_shared<RankAwareRelativePoseFactor>(
            previous.pose, current.pose, segment.lidar.T_imu_start_imu_end,
            segment.lidar.information);
        lidar_graph.push_back(factor);
        data_graph.push_back(factor);
        calibrated_data_graph.push_back(factor);
        full_graph.push_back(factor);
        lidar_residual_dimension += segment.lidar.information.rank;
      }

      gtsam::PreintegratedImuMeasurements preintegrated(preintegration_params, initial_bias);
      integrateInterval(segment.imu, &preintegrated);
      auto factor =
          boost::make_shared<gtsam::ImuFactor>(previous.pose, previous.velocity, current.pose,
                                               current.velocity, biasKey(), preintegrated);
      imu_graph.push_back(factor);
      data_graph.push_back(factor);
      calibrated_data_graph.push_back(factor);
      full_graph.push_back(factor);
    }

    const NavigationKeys first_keys = navigationKeys(0U);
    auto position_gauge = boost::make_shared<gtsam::PoseTranslationPrior<gtsam::Pose3>>(
        first_keys.pose, gtsam::Point3::Zero(),
        gtsam::noiseModel::Isotropic::Sigma(3, config_.position_gauge_sigma_m));
    const gtsam::Pose3 first_pose = values.at<gtsam::Pose3>(first_keys.pose);
    auto yaw_gauge = boost::make_shared<YawGaugeFactor>(first_keys.pose, first_pose.rotation(),
                                                        request.imu_noise.gravity_odom,
                                                        config_.yaw_gauge_sigma_rad);
    gauge_graph.push_back(position_gauge);
    gauge_graph.push_back(yaw_gauge);
    full_graph.push_back(position_gauge);
    full_graph.push_back(yaw_gauge);

    auto accelerometer_bias_prior = boost::make_shared<AccelerometerBiasPriorFactor>(
        biasKey(), request.imu_noise.accelerometer_bias_prior_mean_mps2,
        request.imu_noise.accelerometer_bias_prior_sigma_mps2);
    auto gyroscope_bias_prior = boost::make_shared<GyroscopeBiasPriorFactor>(
        biasKey(), request.imu_noise.gyroscope_bias_prior_mean_radps,
        request.imu_noise.gyroscope_bias_prior_sigma_radps);
    bias_prior_graph.push_back(accelerometer_bias_prior);
    bias_prior_graph.push_back(gyroscope_bias_prior);
    calibrated_data_graph.push_back(accelerometer_bias_prior);
    full_graph.push_back(accelerometer_bias_prior);
    full_graph.push_back(gyroscope_bias_prior);

    const double initial_error = full_graph.error(values);
    gtsam::LevenbergMarquardtParams parameters;
    parameters.setMaxIterations(static_cast<int>(config_.maximum_solver_iterations));
    parameters.setRelativeErrorTol(config_.solver_relative_error_tolerance);
    parameters.setAbsoluteErrorTol(config_.solver_absolute_error_tolerance);
    parameters.setLinearSolverType("MULTIFRONTAL_QR");
    gtsam::LevenbergMarquardtOptimizer optimizer(full_graph, values, parameters);
    gtsam::Values optimized = optimizer.optimize();
    const double final_error = full_graph.error(optimized);
    if (!std::isfinite(initial_error) || !std::isfinite(final_error)) {
      return Result::failure(error(MotionInitializationErrorCode::NonFiniteSolution,
                                   "batch objective became non-finite"));
    }
    const double allowed_error_increase =
        std::max(config_.solver_absolute_error_tolerance,
                 config_.solver_relative_error_tolerance * std::max(1.0, std::abs(initial_error)));
    const bool converged =
        (static_cast<std::size_t>(optimizer.iterations()) < config_.maximum_solver_iterations ||
         final_error <= config_.solver_absolute_error_tolerance) &&
        final_error <= initial_error + allowed_error_increase;
    if (!converged) {
      std::ostringstream detail;
      detail << "bounded batch reached its iteration/error gate (iterations="
             << optimizer.iterations() << '/' << config_.maximum_solver_iterations
             << ", initial_error=" << initial_error << ", final_error=" << final_error
             << ", allowed_increase=" << allowed_error_increase << ')';
      return Result::failure(
          error(MotionInitializationErrorCode::SolverDidNotConverge, detail.str()));
    }

    const HessianDiagnostics data_hessian =
        inspectHessian(data_graph, optimized, ordering, config_);
    const HessianDiagnostics calibrated_data_hessian =
        inspectHessian(calibrated_data_graph, optimized, ordering, config_);
    const HessianDiagnostics full_hessian =
        inspectHessian(full_graph, optimized, ordering, config_);
    const std::size_t scalar_dimension = 9U * state_count + 6U;
    const std::size_t expected_data_rank = scalar_dimension - 4U;
    const bool valid_rank_dimensions = data_hessian.dimension == scalar_dimension &&
                                       calibrated_data_hessian.dimension == scalar_dimension &&
                                       full_hessian.dimension == scalar_dimension;
    const bool data_rank_not_above_generic = data_hessian.rank <= expected_data_rank;
    const std::size_t prior_resolved_accel_tilt_modes =
        data_rank_not_above_generic ? expected_data_rank - data_hessian.rank : scalar_dimension;
    const bool accepted_prior_resolution =
        prior_resolved_accel_tilt_modes <= config_.maximum_prior_resolved_accel_tilt_modes &&
        calibrated_data_hessian.rank == expected_data_rank;
    if (!valid_rank_dimensions || !data_rank_not_above_generic || !accepted_prior_resolution ||
        full_hessian.rank != scalar_dimension) {
      std::ostringstream detail;
      detail << "batch rank is data " << data_hessian.rank << '/' << expected_data_rank
             << ", calibrated_data " << calibrated_data_hessian.rank << '/' << expected_data_rank
             << ", full " << full_hessian.rank << '/' << scalar_dimension
             << ", prior_resolved_accel_tilt_modes=" << prior_resolved_accel_tilt_modes << '/'
             << config_.maximum_prior_resolved_accel_tilt_modes
             << " (full threshold=" << full_hessian.rank_threshold << ", smallest=[";
      for (std::size_t index = 0U; index < full_hessian.smallest_scaled_eigenvalues.size();
           ++index) {
        detail << (index == 0U ? "" : ",") << full_hessian.smallest_scaled_eigenvalues.at(index);
      }
      detail << "])";
      return Result::failure(
          error(MotionInitializationErrorCode::RankDeficientBatch, detail.str()));
    }
    if (!std::isfinite(data_hessian.supported_condition) ||
        !std::isfinite(calibrated_data_hessian.supported_condition) ||
        !std::isfinite(full_hessian.supported_condition) ||
        data_hessian.supported_condition > config_.maximum_supported_hessian_condition ||
        calibrated_data_hessian.supported_condition > config_.maximum_supported_hessian_condition ||
        full_hessian.supported_condition > config_.maximum_supported_hessian_condition) {
      return Result::failure(
          error(MotionInitializationErrorCode::IllConditionedBatch,
                "batch supported-space Hessian exceeds the configured condition limit"));
    }

    const NavigationKeys final_keys = navigationKeys(state_count - 1U);
    const gtsam::Pose3 final_pose = optimized.at<gtsam::Pose3>(final_keys.pose);
    const gtsam::Vector3 final_velocity = optimized.at<gtsam::Vector3>(final_keys.velocity);
    const gtsam::imuBias::ConstantBias final_bias =
        optimized.at<gtsam::imuBias::ConstantBias>(biasKey());
    const double bias_prior_mahalanobis = std::sqrt(
        (final_bias.accelerometer() - request.imu_noise.accelerometer_bias_prior_mean_mps2)
                .squaredNorm() /
            std::pow(request.imu_noise.accelerometer_bias_prior_sigma_mps2, 2) +
        (final_bias.gyroscope() - request.imu_noise.gyroscope_bias_prior_mean_radps).squaredNorm() /
            std::pow(request.imu_noise.gyroscope_bias_prior_sigma_radps, 2));
    if (!std::isfinite(bias_prior_mahalanobis) ||
        bias_prior_mahalanobis > config_.maximum_bias_prior_mahalanobis) {
      std::ostringstream detail;
      detail << "optimized IMU bias is inconsistent with its calibration prior "
             << "(normalized norm=" << bias_prior_mahalanobis
             << ", maximum=" << config_.maximum_bias_prior_mahalanobis << ')';
      return Result::failure(
          error(MotionInitializationErrorCode::BiasPlausibilityFailed, detail.str()));
    }

    MotionInitialization result;
    result.exact_time = final_time;
    result.state.T_odom_imu = fromGtsamPose(final_pose);
    result.state.velocity_odom = final_velocity;
    result.state.gyro_bias = final_bias.gyroscope();
    result.state.accel_bias = final_bias.accelerometer();
    result.T_odom_bootstrap = fromGtsamPose(optimized.at<gtsam::Pose3>(first_keys.pose));
    if (!result.state.T_odom_imu.matrix().allFinite() || !result.state.velocity_odom.allFinite() ||
        !result.state.gyro_bias.allFinite() || !result.state.accel_bias.allFinite()) {
      return Result::failure(error(MotionInitializationErrorCode::NonFiniteSolution,
                                   "optimized final navigation state is non-finite"));
    }

    MotionInitializationDiagnostics& diagnostics = result.diagnostics;
    diagnostics.pass = request.pass;
    diagnostics.segments = request.segments.size();
    diagnostics.imu_knots = imu_knots;
    diagnostics.support = total_support;
    diagnostics.rotation_excitation_rad = initial.rotation_excitation_rad;
    diagnostics.acceleration_excitation_mps2 = initial.acceleration_excitation_mps2;
    diagnostics.scalar_dimension = scalar_dimension;
    diagnostics.expected_data_rank = expected_data_rank;
    diagnostics.data_rank = data_hessian.rank;
    diagnostics.calibrated_data_rank = calibrated_data_hessian.rank;
    diagnostics.full_rank = full_hessian.rank;
    diagnostics.prior_resolved_accel_tilt_modes = prior_resolved_accel_tilt_modes;
    diagnostics.observability_class =
        prior_resolved_accel_tilt_modes == 0U
            ? MotionInitializationObservabilityClass::SensorObservable
            : MotionInitializationObservabilityClass::PriorResolvedAccelerometerTilt;
    diagnostics.data_supported_condition = data_hessian.supported_condition;
    diagnostics.calibrated_data_supported_condition = calibrated_data_hessian.supported_condition;
    diagnostics.full_hessian_condition = full_hessian.supported_condition;
    diagnostics.initial_error = initial_error;
    diagnostics.final_error = final_error;
    diagnostics.lidar_error = lidar_graph.error(optimized);
    diagnostics.imu_error = imu_graph.error(optimized);
    diagnostics.bias_prior_error = bias_prior_graph.error(optimized);
    diagnostics.gauge_error = gauge_graph.error(optimized);
    diagnostics.lidar_residual_dimension = lidar_residual_dimension;
    diagnostics.imu_residual_dimension = 9U * request.segments.size();
    diagnostics.bias_prior_residual_dimension = 6U;
    diagnostics.gauge_residual_dimension = 4U;
    diagnostics.total_residual_dimension =
        diagnostics.lidar_residual_dimension + diagnostics.imu_residual_dimension +
        diagnostics.bias_prior_residual_dimension + diagnostics.gauge_residual_dimension;
    diagnostics.effective_degrees_of_freedom =
        diagnostics.total_residual_dimension > scalar_dimension
            ? diagnostics.total_residual_dimension - scalar_dimension
            : 0U;
    diagnostics.lidar_mean_squared_whitened_residual =
        diagnostics.lidar_residual_dimension > 0U
            ? 2.0 * diagnostics.lidar_error /
                  static_cast<double>(diagnostics.lidar_residual_dimension)
            : std::numeric_limits<double>::infinity();
    diagnostics.imu_mean_squared_whitened_residual =
        2.0 * diagnostics.imu_error / static_cast<double>(diagnostics.imu_residual_dimension);
    diagnostics.reduced_chi_square =
        diagnostics.effective_degrees_of_freedom > 0U
            ? 2.0 * final_error / static_cast<double>(diagnostics.effective_degrees_of_freedom)
            : std::numeric_limits<double>::infinity();
    diagnostics.holdout_lidar_segments = holdout_segments;
    for (std::size_t index = lidar_segments_in_objective; index < request.segments.size();
         ++index) {
      diagnostics.holdout_error += relativePoseCost(
          request.segments[index], optimized.at<gtsam::Pose3>(navigationKeys(index).pose),
          optimized.at<gtsam::Pose3>(navigationKeys(index + 1U).pose));
      diagnostics.holdout_residual_dimension += request.segments[index].lidar.information.rank;
    }
    diagnostics.holdout_mean_squared_whitened_residual =
        diagnostics.holdout_residual_dimension > 0U
            ? 2.0 * diagnostics.holdout_error /
                  static_cast<double>(diagnostics.holdout_residual_dimension)
            : 0.0;
    diagnostics.statistically_compatible =
        std::isfinite(diagnostics.lidar_mean_squared_whitened_residual) &&
        std::isfinite(diagnostics.imu_mean_squared_whitened_residual) &&
        std::isfinite(diagnostics.reduced_chi_square) &&
        diagnostics.lidar_mean_squared_whitened_residual <=
            config_.maximum_lidar_mean_squared_whitened_residual &&
        diagnostics.imu_mean_squared_whitened_residual <=
            config_.maximum_imu_mean_squared_whitened_residual &&
        diagnostics.reduced_chi_square <= config_.maximum_reduced_chi_square;
    diagnostics.conditioned_lidar_imu_approximation = true;
    diagnostics.minimum_imu_conditioning_covariance_inflation =
        std::numeric_limits<double>::infinity();
    diagnostics.minimum_applied_lidar_covariance_inflation =
        std::numeric_limits<double>::infinity();
    for (const MotionInitializationSegment& segment : request.segments) {
      diagnostics.minimum_imu_conditioning_covariance_inflation =
          std::min(diagnostics.minimum_imu_conditioning_covariance_inflation,
                   segment.lidar.imu_conditioning_covariance_inflation);
      diagnostics.minimum_applied_lidar_covariance_inflation =
          std::min(diagnostics.minimum_applied_lidar_covariance_inflation,
                   segment.lidar.applied_covariance_inflation);
    }
    diagnostics.bias_prior_mahalanobis = bias_prior_mahalanobis;
    diagnostics.solver_iterations = static_cast<std::size_t>(optimizer.iterations());

    if (commit_candidate && !diagnostics.statistically_compatible) {
      std::ostringstream detail;
      detail << "full-deskew batch is statistically incompatible "
             << "(LiDAR mean squared whitened=" << diagnostics.lidar_mean_squared_whitened_residual
             << ", IMU=" << diagnostics.imu_mean_squared_whitened_residual
             << ", reduced chi-square=" << diagnostics.reduced_chi_square
             << ", DoF=" << diagnostics.effective_degrees_of_freedom << ')';
      return Result::failure(
          error(MotionInitializationErrorCode::StatisticalCompatibilityFailed, detail.str()));
    }
    if (commit_candidate && holdout_segments > 0U &&
        (!std::isfinite(diagnostics.holdout_mean_squared_whitened_residual) ||
         diagnostics.holdout_mean_squared_whitened_residual >
             config_.maximum_holdout_mean_squared_whitened_residual)) {
      std::ostringstream detail;
      detail << "withheld future LiDAR increment is incompatible with the "
                "optimized IMU trajectory (mean squared whitened="
             << diagnostics.holdout_mean_squared_whitened_residual
             << ", maximum=" << config_.maximum_holdout_mean_squared_whitened_residual << ')';
      return Result::failure(
          error(MotionInitializationErrorCode::HoldoutCompatibilityFailed, detail.str()));
    }

    diagnostics.covariance_residual_inflation =
        std::clamp(std::max(1.0, diagnostics.reduced_chi_square), 1.0,
                   config_.maximum_covariance_residual_inflation);
    result.covariance = finalNavigationCovariance(full_graph, optimized, final_keys);
    calibrateCovariance(&result.covariance, diagnostics.covariance_residual_inflation, config_);
    if (!validCovariance(result.covariance, config_)) {
      return Result::failure(error(
          MotionInitializationErrorCode::MarginalCovarianceFailure,
          "calibrated joint [R,V,P,Bg,Ba] covariance is non-finite, asymmetric, or indefinite"));
    }

    result.reference_states.reserve(state_count);
    for (std::size_t index = 0U; index < state_count; ++index) {
      const NavigationKeys keys = navigationKeys(index);
      core::NavStateEstimate state;
      state.T_odom_imu = fromGtsamPose(optimized.at<gtsam::Pose3>(keys.pose));
      state.velocity_odom = optimized.at<gtsam::Vector3>(keys.velocity);
      state.gyro_bias = final_bias.gyroscope();
      state.accel_bias = final_bias.accelerometer();
      const core::FusionTime time =
          index == 0U ? first_time : request.segments[index - 1U].lidar.end_time;
      result.reference_states.push_back(TimedNavState{time, std::move(state)});
    }
    result.lineage_uses.reserve(request.segments.size() * 2U);
    for (std::size_t index = 0U; index < request.segments.size(); ++index) {
      result.lineage_uses.push_back(MotionInitializationLineageUse{
          request.segments[index].lidar.lineage.id,
          index < lidar_segments_in_objective ? MotionInitializationLineageRole::FusedResidual
                                              : MotionInitializationLineageRole::ValidationGate});
      result.lineage_uses.push_back(MotionInitializationLineageUse{
          request.segments[index].imu_lineage.id, MotionInitializationLineageRole::FusedResidual});
    }
    solve_timing.finish();
    return Result::success(std::move(result));
  } catch (const std::exception& exception) {
    return Result::failure(
        error(MotionInitializationErrorCode::SolverFailure,
              std::string("motion initialization batch failed: ") + exception.what()));
  }
}

}  // namespace meridian::local
