#include "meridian/local_rt/initialization/dynamic_initializer.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sophus/so3.hpp>
#include <stdexcept>
#include <utility>

#include "meridian/local_rt/combined_preintegration.hpp"
#include "meridian/local_rt/imu_buffer.hpp"

namespace meridian::local_rt::initialization {
namespace {

constexpr double kNanosecondsPerSecond = 1.0e9;
constexpr double kGyroscopeBiasChiSquareThreeDof999 = 16.266236196238129;

Eigen::Vector3d eigen(const core::Vec3d& vector) {
  return {vector.x, vector.y, vector.z};
}

Eigen::Quaterniond eigen(const core::Quaterniond& quaternion) {
  return {quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()};
}

core::Vec3d coreVector(const Eigen::Vector3d& vector) {
  return {.x = vector.x(), .y = vector.y(), .z = vector.z()};
}

Sophus::SE3d sophus(const core::Pose3d& pose) {
  return {eigen(pose.rotation()), eigen(pose.translation())};
}

core::Pose3d corePose(const Sophus::SE3d& pose) {
  const Eigen::Quaterniond quaternion = pose.unit_quaternion().normalized();
  return core::Pose3d(
      coreVector(pose.translation()),
      core::Quaterniond(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()));
}

Eigen::Quaterniond rotationIncrement(const Eigen::Vector3d& rotation_vector) {
  const double angle = rotation_vector.norm();
  if (angle < 1.0e-12) {
    return Eigen::Quaterniond(1.0, 0.5 * rotation_vector.x(), 0.5 * rotation_vector.y(),
                              0.5 * rotation_vector.z())
        .normalized();
  }
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotation_vector / angle));
}

bool positiveFinite(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

void validate(const DynamicInitializerOptions& options) {
  if (options.target_sweeps < 4U || options.maximum_support_ns <= 0 ||
      !std::isfinite(options.minimum_range_m) || options.minimum_range_m < 0.0 ||
      !positiveFinite(options.maximum_range_m) ||
      options.maximum_range_m <= options.minimum_range_m || !positiveFinite(options.gravity_m_s2) ||
      !positiveFinite(options.gyroscope_finite_difference_step_rad_s) ||
      !positiveFinite(options.minimum_singular_value_ratio) ||
      options.minimum_singular_value_ratio > 1.0 ||
      !std::isfinite(options.maximum_condition_number) || options.maximum_condition_number < 1.0 ||
      !positiveFinite(options.maximum_gyro_bias_correction_rad_s) ||
      !positiveFinite(options.maximum_gravity_magnitude_error_m_s2) ||
      !positiveFinite(options.maximum_alignment_residual_rms) ||
      !positiveFinite(options.maximum_held_out_rotation_error_rad) ||
      !positiveFinite(options.maximum_held_out_translation_error_m) ||
      !positiveFinite(options.maximum_refinement_rotation_change_rad) ||
      !positiveFinite(options.maximum_refinement_translation_change_m) ||
      !options.gyroscope_bias_prior_covariance.allFinite() ||
      !options.gyroscope_bias_prior_covariance.isApprox(
          options.gyroscope_bias_prior_covariance.transpose(), 1.0e-12)) {
    throw std::invalid_argument("dynamic initialization options are invalid");
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> prior_covariance(
      options.gyroscope_bias_prior_covariance);
  if (prior_covariance.info() != Eigen::Success ||
      prior_covariance.eigenvalues().minCoeff() <= 0.0) {
    throw std::invalid_argument(
        "dynamic gyroscope bias prior covariance must be positive definite");
  }
}

std::optional<core::TimeNs> shifted(core::TimeNs time, std::int64_t offset_ns) {
  return core::TimeNs::checkedAdd(time, offset_ns);
}

std::optional<core::TimeNs> pointTime(const core::LidarSweep& sweep, const core::LidarPoint& point,
                                      std::int64_t lidar_to_imu_ns) {
  const auto lidar_time =
      core::TimeNs::checkedAdd(sweep.header().measurementTime(), point.time_offset_ns);
  return lidar_time.has_value() ? shifted(*lidar_time, lidar_to_imu_ns) : std::nullopt;
}

double secondsBetween(core::TimeNs end, core::TimeNs begin) {
  const auto duration = core::TimeNs::checkedDifference(end, begin);
  return duration.has_value() ? static_cast<double>(*duration) / kNanosecondsPerSecond
                              : std::numeric_limits<double>::quiet_NaN();
}

Eigen::Matrix3d removeYaw(const Eigen::Matrix3d& rotation) {
  const double projected_x_norm = std::hypot(rotation(0, 0), rotation(1, 0));
  const double yaw = projected_x_norm > 1.0e-12 ? std::atan2(rotation(1, 0), rotation(0, 0))
                                                : std::atan2(-rotation(0, 1), rotation(1, 1));
  return Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * rotation;
}

BootstrapPoseSummary summarize(const core::LidarSweep& sweep, const BootstrapFrame& frame) {
  return BootstrapPoseSummary{
      .measurement_id = frame.measurement_id,
      .measurement_time = sweep.acquisitionEnd(),
      .odom_from_lidar = corePose(frame.odom_from_lidar),
      .source_point_count = frame.downsample.input_points,
      .correspondence_count = frame.quality.correspondences,
      .point_rmse_m = frame.quality.point_rmse_m,
      .hessian_condition_number = frame.quality.hessian_condition_number,
      .accepted = frame.accepted,
  };
}

struct RotationSegment final {
  core::TimeNs begin;
  core::TimeNs end;
  Eigen::Quaterniond begin_from_segment{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d angular_velocity_rad_s{Eigen::Vector3d::Zero()};
};

std::vector<RotationSegment> rotationSegments(const ImuInterval& interval,
                                              const core::ImuBias& bias,
                                              Eigen::Quaterniond& begin_from_end) {
  std::vector<RotationSegment> segments;
  segments.reserve(interval.segments().size());
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d gyroscope_bias = eigen(bias.gyroscopeRadS());
  for (const ImuIntegrationSegment& segment : interval.segments()) {
    const Eigen::Vector3d angular_velocity = segment.angularVelocityRadS() - gyroscope_bias;
    segments.push_back(RotationSegment{.begin = segment.support().begin(),
                                       .end = segment.support().end(),
                                       .begin_from_segment = orientation,
                                       .angular_velocity_rad_s = angular_velocity});
    orientation = (orientation * rotationIncrement(segment.durationSeconds() * angular_velocity))
                      .normalized();
  }
  begin_from_end = orientation;
  return segments;
}

std::optional<Eigen::Quaterniond> rotationAt(core::TimeNs time,
                                             const std::vector<RotationSegment>& segments) {
  const auto item = std::lower_bound(
      segments.begin(), segments.end(), time,
      [](const RotationSegment& segment, core::TimeNs value) { return segment.end < value; });
  if (item == segments.end() || time < item->begin) {
    return std::nullopt;
  }
  const double dt = secondsBetween(time, item->begin);
  if (!std::isfinite(dt) || dt < 0.0) {
    return std::nullopt;
  }
  return (item->begin_from_segment * rotationIncrement(dt * item->angular_velocity_rad_s))
      .normalized();
}

struct LeastSquaresResult final {
  Eigen::VectorXd solution;
  double minimum_singular_value{};
  double condition_number{};
  double residual_rms{};
};

std::optional<Eigen::MatrixXd> inverseSquareRoot(const Eigen::MatrixXd& covariance) {
  if (covariance.rows() != covariance.cols() || covariance.rows() == 0 || !covariance.allFinite()) {
    return std::nullopt;
  }
  const Eigen::MatrixXd symmetric = 0.5 * (covariance + covariance.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(symmetric);
  if (eigensolver.info() != Eigen::Success || !eigensolver.eigenvalues().allFinite()) {
    return std::nullopt;
  }
  const double maximum = eigensolver.eigenvalues().maxCoeff();
  if (!(maximum > 0.0) || eigensolver.eigenvalues().minCoeff() < -1.0e-8 * maximum) {
    return std::nullopt;
  }
  const double floor = std::max(1.0e-18, 1.0e-12 * maximum);
  const Eigen::VectorXd inverse_sqrt =
      eigensolver.eigenvalues().cwiseMax(floor).cwiseSqrt().cwiseInverse();
  return eigensolver.eigenvectors() * inverse_sqrt.asDiagonal() *
         eigensolver.eigenvectors().transpose();
}

std::optional<LeastSquaresResult> solveColumnScaled(const Eigen::MatrixXd& matrix,
                                                    const Eigen::VectorXd& rhs,
                                                    const DynamicInitializerOptions& options) {
  if (matrix.rows() < matrix.cols() || matrix.rows() != rhs.rows() || matrix.cols() == 0 ||
      !matrix.allFinite() || !rhs.allFinite()) {
    return std::nullopt;
  }
  const Eigen::VectorXd column_norms =
      matrix.array().square().colwise().sum().sqrt().matrix().transpose();
  if (!column_norms.allFinite() || (column_norms.array() <= 1.0e-15).any()) {
    return std::nullopt;
  }
  const Eigen::VectorXd inverse_scales = column_norms.cwiseInverse();
  const Eigen::MatrixXd scaled = matrix * inverse_scales.asDiagonal();
  const Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(scaled,
                                                        Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (decomposition.info() != Eigen::Success ||
      decomposition.singularValues().size() != matrix.cols() ||
      !decomposition.singularValues().allFinite()) {
    return std::nullopt;
  }
  const double maximum = decomposition.singularValues()[0];
  const double minimum = decomposition.singularValues()[matrix.cols() - 1];
  if (!(maximum > 0.0) || !(minimum > 0.0)) {
    return std::nullopt;
  }
  const double ratio = minimum / maximum;
  const double condition_number = maximum / minimum;
  if (ratio < options.minimum_singular_value_ratio ||
      condition_number > options.maximum_condition_number) {
    return std::nullopt;
  }
  const Eigen::VectorXd scaled_solution = decomposition.solve(rhs);
  const Eigen::VectorXd solution = inverse_scales.asDiagonal() * scaled_solution;
  if (!solution.allFinite()) {
    return std::nullopt;
  }
  return LeastSquaresResult{
      .solution = solution,
      .minimum_singular_value = minimum,
      .condition_number = condition_number,
      .residual_rms =
          (matrix * solution - rhs).norm() / std::sqrt(static_cast<double>(matrix.rows())),
  };
}

Eigen::Vector3d rotationResidual(const CombinedPreintegration& preintegration,
                                 const Eigen::Matrix3d& relative_rotation) {
  const Sophus::SO3d delta(eigen(preintegration.deltaRotation()));
  return (delta.inverse() * Sophus::SO3d(relative_rotation)).log();
}

Eigen::Matrix<double, 3, 2> gravityTangentBasis(const Eigen::Vector3d& gravity) {
  const Eigen::Vector3d direction = gravity.normalized();
  Eigen::Vector3d reference = Eigen::Vector3d::UnitZ();
  if (std::abs(direction.dot(reference)) > 0.9) {
    reference = Eigen::Vector3d::UnitX();
  }
  const Eigen::Vector3d first = (reference - direction * direction.dot(reference)).normalized();
  Eigen::Matrix<double, 3, 2> basis;
  basis.col(0) = first;
  basis.col(1) = direction.cross(first).normalized();
  return basis;
}

std::optional<double> mahalanobisSquared(const Eigen::VectorXd& residual,
                                         const Eigen::MatrixXd& covariance) {
  const auto whitening = inverseSquareRoot(covariance);
  if (!whitening.has_value() || whitening->rows() != residual.rows()) {
    return std::nullopt;
  }
  const double value = (static_cast<const Eigen::MatrixXd&>(*whitening) * residual).squaredNorm();
  return std::isfinite(value) ? std::optional<double>(value) : std::nullopt;
}

}  // namespace

DynamicInitializer::DynamicInitializer(DynamicInitializerOptions options)
    : options_(std::move(options)), bootstrap_(options_.bootstrap) {
  validate(options_);
}

void DynamicInitializer::reset() {
  bootstrap_.reset();
  accepted_sweeps_.clear();
  rejected_transitions_ = 0U;
  accepted_result_.reset();
  accepted_anchor_points_.clear();
}

std::optional<lidar::PointCloud> DynamicInitializer::deskewRotationOnly(
    const core::LidarSweep& sweep, const ImuBuffer& imu_buffer, const core::ImuBias& bias) const {
  const auto begin = shifted(sweep.acquisitionBegin(), options_.lidar_time_offset_to_imu_ns);
  const auto end = shifted(sweep.acquisitionEnd(), options_.lidar_time_offset_to_imu_ns);
  if (!begin.has_value() || !end.has_value()) {
    return std::nullopt;
  }

  std::vector<RotationSegment> segments;
  Eigen::Quaterniond begin_from_end = Eigen::Quaterniond::Identity();
  if (*end > *begin) {
    const ImuIntervalResult support = imu_buffer.interval(*begin, *end);
    if (!support.ok()) {
      return std::nullopt;
    }
    segments = rotationSegments(*support.value(), bias, begin_from_end);
  }

  const Sophus::SE3d imu_from_lidar = sophus(options_.imu_from_lidar);
  const Sophus::SE3d lidar_from_imu = imu_from_lidar.inverse();
  const double minimum_squared = options_.minimum_range_m * options_.minimum_range_m;
  const double maximum_squared = options_.maximum_range_m * options_.maximum_range_m;
  lidar::PointCloud deskewed;
  deskewed.reserve(sweep.size());
  for (const core::LidarPoint& point : sweep.points()) {
    const Eigen::Vector3d raw(point.x, point.y, point.z);
    const double squared_range = raw.squaredNorm();
    if (!raw.allFinite() || squared_range < minimum_squared || squared_range > maximum_squared) {
      continue;
    }
    if (segments.empty()) {
      deskewed.push_back(raw);
      continue;
    }
    const auto time = pointTime(sweep, point, options_.lidar_time_offset_to_imu_ns);
    if (!time.has_value()) {
      return std::nullopt;
    }
    const auto begin_from_point = rotationAt(*time, segments);
    if (!begin_from_point.has_value()) {
      return std::nullopt;
    }
    const Eigen::Matrix3d end_from_point_rotation =
        begin_from_end.conjugate().toRotationMatrix() * begin_from_point->toRotationMatrix();
    const Sophus::SE3d end_from_point(end_from_point_rotation, Eigen::Vector3d::Zero());
    deskewed.push_back(lidar_from_imu * end_from_point * imu_from_lidar * raw);
  }
  return deskewed;
}

std::optional<lidar::PointCloud> DynamicInitializer::deskewWithEstimate(
    const AcceptedSweep& accepted, const AlignmentEstimate& estimate, std::size_t accepted_index,
    const ImuBuffer& imu_buffer, const GtsamCombinedPreintegrator& preintegrator) const {
  const auto begin =
      shifted(accepted.sweep.acquisitionBegin(), options_.lidar_time_offset_to_imu_ns);
  const auto end = shifted(accepted.sweep.acquisitionEnd(), options_.lidar_time_offset_to_imu_ns);
  if (!begin.has_value() || !end.has_value()) {
    return std::nullopt;
  }

  Eigen::Vector3d end_velocity = Eigen::Vector3d::Zero();
  if (accepted_index < estimate.velocities_bootstrap.size()) {
    end_velocity = estimate.velocities_bootstrap[accepted_index];
  } else {
    if (accepted_index != estimate.velocities_bootstrap.size() || accepted_index == 0U ||
        accepted_index >= accepted_sweeps_.size()) {
      return std::nullopt;
    }
    const AcceptedSweep& previous = accepted_sweeps_[accepted_index - 1U];
    const auto previous_time =
        shifted(previous.sweep.acquisitionEnd(), options_.lidar_time_offset_to_imu_ns);
    if (!previous_time.has_value()) {
      return std::nullopt;
    }
    const ImuIntervalResult transition = imu_buffer.interval(*previous_time, *end);
    if (!transition.ok()) {
      return std::nullopt;
    }
    const PreintegrationResult integrated =
        preintegrator.integrate(*transition.value(), estimate.bias);
    if (!integrated.ok()) {
      return std::nullopt;
    }
    const Sophus::SE3d bootstrap_from_imu =
        previous.bootstrap.odom_from_lidar * sophus(options_.imu_from_lidar).inverse();
    end_velocity = estimate.velocities_bootstrap.back() +
                   estimate.gravity_bootstrap * integrated.value()->durationSeconds() +
                   bootstrap_from_imu.rotationMatrix() * eigen(integrated.value()->deltaVelocity());
  }

  const Sophus::SE3d imu_from_lidar = sophus(options_.imu_from_lidar);
  const Sophus::SE3d bootstrap_from_lidar_end = accepted.bootstrap.odom_from_lidar;
  const Sophus::SE3d bootstrap_from_imu_end = bootstrap_from_lidar_end * imu_from_lidar.inverse();
  const double minimum_squared = options_.minimum_range_m * options_.minimum_range_m;
  const double maximum_squared = options_.maximum_range_m * options_.maximum_range_m;

  if (*begin == *end) {
    lidar::PointCloud points;
    points.reserve(accepted.sweep.size());
    for (const core::LidarPoint& point : accepted.sweep.points()) {
      const Eigen::Vector3d raw(point.x, point.y, point.z);
      const double squared_range = raw.squaredNorm();
      if (raw.allFinite() && squared_range >= minimum_squared && squared_range <= maximum_squared) {
        points.push_back(raw);
      }
    }
    return points;
  }

  const ImuIntervalResult support = imu_buffer.interval(*begin, *end);
  if (!support.ok()) {
    return std::nullopt;
  }
  struct KinematicState final {
    Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  };
  std::vector<KinematicState> segment_end_states(support.value()->segments().size());
  KinematicState current{.rotation = bootstrap_from_imu_end.rotationMatrix(),
                         .position = bootstrap_from_imu_end.translation(),
                         .velocity = end_velocity};
  const Eigen::Vector3d gyroscope_bias = eigen(estimate.bias.gyroscopeRadS());
  const Eigen::Vector3d accelerometer_bias = eigen(estimate.bias.accelerometerMS2());
  for (std::size_t reverse = support.value()->segments().size(); reverse > 0U; --reverse) {
    const std::size_t index = reverse - 1U;
    const ImuIntegrationSegment& segment = support.value()->segments()[index];
    segment_end_states[index] = current;
    const double dt = segment.durationSeconds();
    const Eigen::Vector3d angular_velocity = segment.angularVelocityRadS() - gyroscope_bias;
    const Eigen::Vector3d specific_force = segment.specificForceMS2() - accelerometer_bias;
    const Eigen::Matrix3d begin_rotation =
        current.rotation * rotationIncrement(-dt * angular_velocity).toRotationMatrix();
    const Eigen::Matrix3d midpoint_rotation =
        begin_rotation * rotationIncrement(0.5 * dt * angular_velocity).toRotationMatrix();
    const Eigen::Vector3d acceleration =
        midpoint_rotation * specific_force + estimate.gravity_bootstrap;
    const Eigen::Vector3d begin_velocity = current.velocity - dt * acceleration;
    const Eigen::Vector3d begin_position =
        current.position - dt * begin_velocity - 0.5 * dt * dt * acceleration;
    current = {.rotation = begin_rotation, .position = begin_position, .velocity = begin_velocity};
  }

  lidar::PointCloud deskewed;
  deskewed.reserve(accepted.sweep.size());
  for (const core::LidarPoint& point : accepted.sweep.points()) {
    const Eigen::Vector3d raw(point.x, point.y, point.z);
    const double squared_range = raw.squaredNorm();
    if (!raw.allFinite() || squared_range < minimum_squared || squared_range > maximum_squared) {
      continue;
    }
    const auto time = pointTime(accepted.sweep, point, options_.lidar_time_offset_to_imu_ns);
    if (!time.has_value()) {
      return std::nullopt;
    }
    const auto segment =
        std::lower_bound(support.value()->segments().begin(), support.value()->segments().end(),
                         *time, [](const ImuIntegrationSegment& item, core::TimeNs value) {
                           return item.support().end() < value;
                         });
    if (segment == support.value()->segments().end() || *time < segment->support().begin()) {
      return std::nullopt;
    }
    const std::size_t index =
        static_cast<std::size_t>(std::distance(support.value()->segments().begin(), segment));
    const KinematicState& at_segment_end = segment_end_states[index];
    const double dt = secondsBetween(segment->support().end(), *time);
    if (!std::isfinite(dt) || dt < 0.0) {
      return std::nullopt;
    }
    const Eigen::Vector3d angular_velocity = segment->angularVelocityRadS() - gyroscope_bias;
    const Eigen::Vector3d specific_force = segment->specificForceMS2() - accelerometer_bias;
    const Eigen::Matrix3d point_rotation =
        at_segment_end.rotation * rotationIncrement(-dt * angular_velocity).toRotationMatrix();
    const Eigen::Matrix3d midpoint_rotation =
        point_rotation * rotationIncrement(0.5 * dt * angular_velocity).toRotationMatrix();
    const Eigen::Vector3d acceleration =
        midpoint_rotation * specific_force + estimate.gravity_bootstrap;
    const Eigen::Vector3d point_velocity = at_segment_end.velocity - dt * acceleration;
    const Eigen::Vector3d point_position =
        at_segment_end.position - dt * point_velocity - 0.5 * dt * dt * acceleration;
    const Sophus::SE3d bootstrap_from_lidar_point(
        point_rotation * imu_from_lidar.rotationMatrix(),
        point_position + point_rotation * imu_from_lidar.translation());
    deskewed.push_back(bootstrap_from_lidar_end.inverse() * bootstrap_from_lidar_point * raw);
  }
  return deskewed;
}

std::optional<DynamicInitializer::AlignmentEstimate> DynamicInitializer::align(
    const std::vector<AcceptedSweep>& accepted, const ImuBuffer& imu_buffer,
    const GtsamCombinedPreintegrator& preintegrator, core::InitializationQuality& failure_quality,
    std::string& failure_reason) const {
  core::InitializationQuality quality;
  quality.lidar_sweep_count = static_cast<std::uint32_t>(accepted.size());
  quality.fitted_transition_count =
      accepted.size() >= 2U ? static_cast<std::uint32_t>(accepted.size() - 2U) : 0U;
  quality.rejected_transition_count = rejected_transitions_;
  const auto fail = [&](std::string reason) -> std::optional<AlignmentEstimate> {
    failure_quality = quality;
    failure_reason = std::move(reason);
    return std::nullopt;
  };
  if (accepted.size() != options_.target_sweeps || accepted.size() < 4U) {
    return fail("dynamic alignment does not have the configured sweep window");
  }

  double registration_minimum = std::numeric_limits<double>::infinity();
  double registration_condition = 1.0;
  for (std::size_t index = 1U; index < accepted.size(); ++index) {
    const lidar::RegistrationQuality& registration = accepted[index].bootstrap.quality;
    const double minimum = registration.hessian_eigenvalues.minCoeff();
    const double maximum = registration.hessian_eigenvalues.maxCoeff();
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || !(minimum > 0.0) ||
        maximum / minimum > options_.maximum_condition_number ||
        minimum / maximum < options_.minimum_singular_value_ratio) {
      return fail("temporary LiDAR odometry is not sufficiently observable");
    }
    registration_minimum = std::min(registration_minimum, minimum);
    registration_condition =
        std::max(registration_condition, registration.hessian_condition_number);
  }
  quality.registration_min_singular_value = registration_minimum;
  quality.registration_condition_number = registration_condition;

  std::vector<Sophus::SE3d> bootstrap_from_imu;
  std::vector<core::TimeNs> times;
  bootstrap_from_imu.reserve(accepted.size());
  times.reserve(accepted.size());
  const Sophus::SE3d lidar_from_imu = sophus(options_.imu_from_lidar).inverse();
  for (const AcceptedSweep& item : accepted) {
    const auto time = shifted(item.sweep.acquisitionEnd(), options_.lidar_time_offset_to_imu_ns);
    if (!time.has_value()) {
      return fail("LiDAR-to-IMU time offset overflows the measurement timeline");
    }
    times.push_back(*time);
    bootstrap_from_imu.push_back(item.bootstrap.odom_from_lidar * lidar_from_imu);
  }

  std::vector<ImuInterval> intervals;
  std::vector<CombinedPreintegration> prior_preintegrations;
  intervals.reserve(accepted.size() - 1U);
  prior_preintegrations.reserve(accepted.size() - 1U);
  std::uint64_t imu_sample_count = 0U;
  for (std::size_t index = 0U; index + 1U < accepted.size(); ++index) {
    const ImuIntervalResult interval = imu_buffer.interval(times[index], times[index + 1U]);
    if (!interval.ok()) {
      return fail("an accepted LiDAR transition lacks exact contiguous IMU support");
    }
    imu_sample_count += static_cast<std::uint64_t>(interval.value()->sourceSampleCount());
    const PreintegrationResult integrated =
        preintegrator.integrate(*interval.value(), options_.calibrated_bias_prior);
    if (!integrated.ok()) {
      return fail("combined IMU preintegration failed during dynamic alignment");
    }
    intervals.push_back(*interval.value());
    prior_preintegrations.push_back(*integrated.value());
  }
  quality.imu_sample_count = imu_sample_count;

  const std::size_t fitted_transition_count = accepted.size() - 2U;
  Eigen::MatrixXd gyro_matrix =
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(3U * fitted_transition_count), 3);
  Eigen::VectorXd gyro_rhs =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(3U * fitted_transition_count));
  const double difference_step = options_.gyroscope_finite_difference_step_rad_s;
  const Eigen::Vector3d prior_gyro = eigen(options_.calibrated_bias_prior.gyroscopeRadS());
  for (std::size_t index = 0U; index < fitted_transition_count; ++index) {
    const Eigen::Matrix3d relative_rotation =
        bootstrap_from_imu[index].rotationMatrix().transpose() *
        bootstrap_from_imu[index + 1U].rotationMatrix();
    const Eigen::Vector3d residual =
        rotationResidual(prior_preintegrations[index], relative_rotation);
    Eigen::Matrix3d jacobian;
    for (Eigen::Index axis = 0; axis < 3; ++axis) {
      Eigen::Vector3d plus_gyro = prior_gyro;
      Eigen::Vector3d minus_gyro = prior_gyro;
      plus_gyro[axis] += difference_step;
      minus_gyro[axis] -= difference_step;
      const core::ImuBias plus_bias(coreVector(plus_gyro),
                                    options_.calibrated_bias_prior.accelerometerMS2());
      const core::ImuBias minus_bias(coreVector(minus_gyro),
                                     options_.calibrated_bias_prior.accelerometerMS2());
      const PreintegrationResult plus = preintegrator.integrate(intervals[index], plus_bias);
      const PreintegrationResult minus = preintegrator.integrate(intervals[index], minus_bias);
      if (!plus.ok() || !minus.ok()) {
        return fail("finite-difference gyro-bias preintegration failed");
      }
      jacobian.col(axis) = (rotationResidual(*plus.value(), relative_rotation) -
                            rotationResidual(*minus.value(), relative_rotation)) /
                           (2.0 * difference_step);
    }
    const Eigen::Matrix3d covariance =
        prior_preintegrations[index].covariance().template block<3, 3>(0, 0);
    const auto whitening = inverseSquareRoot(covariance);
    if (!whitening.has_value()) {
      return fail("gyro preintegration covariance cannot be whitened");
    }
    const Eigen::Index row = static_cast<Eigen::Index>(3U * index);
    gyro_matrix.block<3, 3>(row, 0) = static_cast<const Eigen::MatrixXd&>(*whitening) * jacobian;
    gyro_rhs.segment<3>(row) = static_cast<const Eigen::MatrixXd&>(*whitening) * (-residual);
  }
  const auto gyro_solution = solveColumnScaled(gyro_matrix, gyro_rhs, options_);
  if (!gyro_solution.has_value()) {
    return fail("gyro-bias alignment is rank deficient or ill-conditioned");
  }
  const Eigen::Vector3d gyro_correction = gyro_solution->solution;
  quality.gyro_bias_min_singular_value = gyro_solution->minimum_singular_value;
  quality.gyro_bias_condition_number = gyro_solution->condition_number;
  quality.gyro_bias_correction_norm_rad_s = gyro_correction.norm();
  if (!gyro_correction.allFinite() ||
      gyro_correction.norm() > options_.maximum_gyro_bias_correction_rad_s) {
    return fail("gyro-bias correction exceeds the configured absolute bound");
  }
  const auto gyro_prior_mahalanobis =
      mahalanobisSquared(gyro_correction, options_.gyroscope_bias_prior_covariance);
  if (!gyro_prior_mahalanobis.has_value() ||
      *gyro_prior_mahalanobis > kGyroscopeBiasChiSquareThreeDof999) {
    return fail("gyro-bias correction is inconsistent with the calibrated prior");
  }
  const core::ImuBias estimated_bias(coreVector(prior_gyro + gyro_correction),
                                     options_.calibrated_bias_prior.accelerometerMS2());

  std::vector<CombinedPreintegration> preintegrations;
  preintegrations.reserve(intervals.size());
  for (const ImuInterval& interval : intervals) {
    const PreintegrationResult integrated = preintegrator.integrate(interval, estimated_bias);
    if (!integrated.ok()) {
      return fail("gyro-bias-corrected IMU reintegration failed");
    }
    preintegrations.push_back(*integrated.value());
  }

  const std::size_t fitted_state_count = accepted.size() - 1U;
  const Eigen::Index raw_column_count = static_cast<Eigen::Index>(3U * fitted_state_count + 3U);
  Eigen::MatrixXd gravity_matrix = Eigen::MatrixXd::Zero(
      static_cast<Eigen::Index>(6U * fitted_transition_count), raw_column_count);
  Eigen::VectorXd gravity_rhs =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(6U * fitted_transition_count));
  std::vector<Eigen::Matrix<double, 6, 6>> position_velocity_whitenings;
  position_velocity_whitenings.reserve(fitted_transition_count);
  for (std::size_t index = 0U; index < fitted_transition_count; ++index) {
    const Eigen::Matrix3d rotation_transpose =
        bootstrap_from_imu[index].rotationMatrix().transpose();
    const double dt = preintegrations[index].durationSeconds();
    Eigen::Matrix<double, 6, 6> covariance;
    covariance.topLeftCorner<3, 3>() =
        preintegrations[index].covariance().template block<3, 3>(3, 3);
    covariance.topRightCorner<3, 3>() =
        preintegrations[index].covariance().template block<3, 3>(3, 6);
    covariance.bottomLeftCorner<3, 3>() =
        preintegrations[index].covariance().template block<3, 3>(6, 3);
    covariance.bottomRightCorner<3, 3>() =
        preintegrations[index].covariance().template block<3, 3>(6, 6);
    const auto whitening = inverseSquareRoot(covariance);
    if (!whitening.has_value()) {
      return fail("position/velocity preintegration covariance cannot be whitened");
    }
    position_velocity_whitenings.push_back(*whitening);

    Eigen::MatrixXd local = Eigen::MatrixXd::Zero(6, raw_column_count);
    local.block<3, 3>(0, static_cast<Eigen::Index>(3U * index)) = dt * rotation_transpose;
    local.block<3, 3>(0, raw_column_count - 3) = 0.5 * dt * dt * rotation_transpose;
    local.block<3, 3>(3, static_cast<Eigen::Index>(3U * index)) = -rotation_transpose;
    local.block<3, 3>(3, static_cast<Eigen::Index>(3U * (index + 1U))) = rotation_transpose;
    local.block<3, 3>(3, raw_column_count - 3) = -dt * rotation_transpose;
    Eigen::Matrix<double, 6, 1> local_rhs;
    local_rhs.head<3>() = rotation_transpose * (bootstrap_from_imu[index + 1U].translation() -
                                                bootstrap_from_imu[index].translation()) -
                          eigen(preintegrations[index].deltaPosition());
    local_rhs.tail<3>() = eigen(preintegrations[index].deltaVelocity());
    const Eigen::Index row = static_cast<Eigen::Index>(6U * index);
    gravity_matrix.block(row, 0, 6, raw_column_count) =
        static_cast<const Eigen::MatrixXd&>(*whitening) * local;
    gravity_rhs.segment<6>(row) = static_cast<const Eigen::MatrixXd&>(*whitening) * local_rhs;
  }
  const auto raw_gravity_solution = solveColumnScaled(gravity_matrix, gravity_rhs, options_);
  if (!raw_gravity_solution.has_value()) {
    return fail("velocity/gravity alignment is rank deficient or ill-conditioned");
  }
  Eigen::Vector3d raw_gravity = raw_gravity_solution->solution.tail<3>();
  const double raw_gravity_magnitude = raw_gravity.norm();
  quality.gravity_min_singular_value = raw_gravity_solution->minimum_singular_value;
  quality.gravity_condition_number = raw_gravity_solution->condition_number;
  quality.raw_gravity_magnitude_m_s2 = raw_gravity_magnitude;
  if (!raw_gravity.allFinite() || !(raw_gravity_magnitude > 0.0) ||
      std::abs(raw_gravity_magnitude - options_.gravity_m_s2) >
          options_.maximum_gravity_magnitude_error_m_s2) {
    return fail("raw gravity magnitude rejects the dynamic alignment");
  }

  Eigen::Vector3d gravity = raw_gravity.normalized() * options_.gravity_m_s2;
  std::vector<Eigen::Vector3d> velocities(fitted_state_count, Eigen::Vector3d::Zero());
  for (std::size_t iteration = 0U; iteration < 4U; ++iteration) {
    const Eigen::Matrix<double, 3, 2> tangent = gravityTangentBasis(gravity);
    const Eigen::Index column_count = static_cast<Eigen::Index>(3U * fitted_state_count + 2U);
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(6U * fitted_transition_count), column_count);
    Eigen::VectorXd rhs =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(6U * fitted_transition_count));
    for (std::size_t index = 0U; index < fitted_transition_count; ++index) {
      const Eigen::Matrix3d rotation_transpose =
          bootstrap_from_imu[index].rotationMatrix().transpose();
      const double dt = preintegrations[index].durationSeconds();
      Eigen::MatrixXd local = Eigen::MatrixXd::Zero(6, column_count);
      local.block<3, 3>(0, static_cast<Eigen::Index>(3U * index)) = dt * rotation_transpose;
      local.block<3, 2>(0, column_count - 2) = 0.5 * dt * dt * rotation_transpose * tangent;
      local.block<3, 3>(3, static_cast<Eigen::Index>(3U * index)) = -rotation_transpose;
      local.block<3, 3>(3, static_cast<Eigen::Index>(3U * (index + 1U))) = rotation_transpose;
      local.block<3, 2>(3, column_count - 2) = -dt * rotation_transpose * tangent;
      Eigen::Matrix<double, 6, 1> local_rhs;
      local_rhs.head<3>() = rotation_transpose * (bootstrap_from_imu[index + 1U].translation() -
                                                  bootstrap_from_imu[index].translation()) -
                            eigen(preintegrations[index].deltaPosition()) -
                            0.5 * dt * dt * rotation_transpose * gravity;
      local_rhs.tail<3>() =
          eigen(preintegrations[index].deltaVelocity()) + dt * rotation_transpose * gravity;
      const Eigen::Index row = static_cast<Eigen::Index>(6U * index);
      matrix.block(row, 0, 6, column_count) = position_velocity_whitenings[index] * local;
      rhs.segment<6>(row) = position_velocity_whitenings[index] * local_rhs;
    }
    const auto refined = solveColumnScaled(matrix, rhs, options_);
    if (!refined.has_value()) {
      return fail("gravity tangent refinement is rank deficient or ill-conditioned");
    }
    quality.gravity_min_singular_value =
        std::min(*quality.gravity_min_singular_value, refined->minimum_singular_value);
    quality.gravity_condition_number =
        std::max(*quality.gravity_condition_number, refined->condition_number);
    for (std::size_t index = 0U; index < fitted_state_count; ++index) {
      velocities[index] = refined->solution.segment<3>(static_cast<Eigen::Index>(3U * index));
    }
    gravity =
        (gravity + tangent * refined->solution.tail<2>()).normalized() * options_.gravity_m_s2;
  }

  double whitened_residual_squared = 0.0;
  for (std::size_t index = 0U; index < fitted_transition_count; ++index) {
    const Eigen::Matrix3d rotation_transpose =
        bootstrap_from_imu[index].rotationMatrix().transpose();
    const double dt = preintegrations[index].durationSeconds();
    Eigen::Matrix<double, 6, 1> residual;
    residual.head<3>() = dt * rotation_transpose * velocities[index] +
                         0.5 * dt * dt * rotation_transpose * gravity -
                         rotation_transpose * (bootstrap_from_imu[index + 1U].translation() -
                                               bootstrap_from_imu[index].translation()) +
                         eigen(preintegrations[index].deltaPosition());
    residual.tail<3>() =
        -rotation_transpose * velocities[index] + rotation_transpose * velocities[index + 1U] -
        dt * rotation_transpose * gravity - eigen(preintegrations[index].deltaVelocity());
    whitened_residual_squared += (position_velocity_whitenings[index] * residual).squaredNorm();
  }
  const double alignment_residual_rms =
      std::sqrt(whitened_residual_squared / static_cast<double>(6U * fitted_transition_count));
  quality.alignment_residual_rms = alignment_residual_rms;
  if (!std::isfinite(alignment_residual_rms) ||
      alignment_residual_rms > options_.maximum_alignment_residual_rms) {
    return fail("whitened alignment residual exceeds the configured bound");
  }

  const std::size_t last_fitted_index = fitted_state_count - 1U;
  const CombinedPreintegration& held_out = preintegrations.back();
  const Eigen::Matrix3d fitted_rotation = bootstrap_from_imu[last_fitted_index].rotationMatrix();
  const Eigen::Matrix3d predicted_rotation =
      fitted_rotation * eigen(held_out.deltaRotation()).toRotationMatrix();
  const Eigen::Vector3d predicted_position =
      bootstrap_from_imu[last_fitted_index].translation() +
      held_out.durationSeconds() * velocities[last_fitted_index] +
      0.5 * held_out.durationSeconds() * held_out.durationSeconds() * gravity +
      fitted_rotation * eigen(held_out.deltaPosition());
  const Eigen::Vector3d held_out_position_residual =
      fitted_rotation.transpose() * (bootstrap_from_imu.back().translation() - predicted_position);
  const double held_out_rotation_error =
      Sophus::SO3d(predicted_rotation.transpose() * bootstrap_from_imu.back().rotationMatrix())
          .log()
          .norm();
  const double held_out_translation_error = held_out_position_residual.norm();
  quality.held_out_rotation_error_rad = held_out_rotation_error;
  quality.held_out_translation_error_m = held_out_translation_error;
  // This cross-modal residual includes temporary LiDAR-registration error.
  // Until that frontend exports a pose covariance, gating it with the much
  // smaller IMU-only covariance would be statistically invalid. Keep the two
  // explicit physical-unit gates instead.
  if (!std::isfinite(held_out_rotation_error) ||
      held_out_rotation_error > options_.maximum_held_out_rotation_error_rad ||
      !std::isfinite(held_out_translation_error) ||
      held_out_translation_error > options_.maximum_held_out_translation_error_m) {
    return fail("the reserved LiDAR transition rejects IMU propagation");
  }

  failure_quality = quality;
  failure_reason.clear();
  return AlignmentEstimate{
      .bias = estimated_bias,
      .gravity_bootstrap = gravity,
      .velocities_bootstrap = std::move(velocities),
      .quality = quality,
  };
}

DynamicInitializationUpdate DynamicInitializer::attempt(
    const ImuBuffer& imu_buffer, const GtsamCombinedPreintegrator& preintegrator) {
  const std::vector<AcceptedSweep> first_pass(accepted_sweeps_.begin(), accepted_sweeps_.end());
  core::InitializationQuality first_quality;
  std::string reason;
  const auto first_estimate = align(first_pass, imu_buffer, preintegrator, first_quality, reason);
  if (!first_estimate.has_value()) {
    DynamicInitializationUpdate update;
    update.status = core::InitializationStatus::kCollecting;
    update.reason = std::move(reason);
    update.quality = std::move(first_quality);
    return update;
  }

  bootstrap_.reset();
  std::vector<AcceptedSweep> refined;
  refined.reserve(first_pass.size());
  for (std::size_t index = 0U; index < first_pass.size(); ++index) {
    const auto deskewed =
        deskewWithEstimate(first_pass[index], *first_estimate, index, imu_buffer, preintegrator);
    if (!deskewed.has_value()) {
      DynamicInitializationUpdate update;
      update.status = core::InitializationStatus::kCollecting;
      update.reason = "full translational re-deskew lacks exact IMU support";
      update.quality = first_estimate->quality;
      return update;
    }
    BootstrapFrame frame = bootstrap_.add(first_pass[index].sweep, *deskewed);
    if (!frame.accepted) {
      DynamicInitializationUpdate update;
      update.status = core::InitializationStatus::kCollecting;
      update.reason = "full re-deskew changed LiDAR registration beyond its quality gates";
      update.quality = first_estimate->quality;
      return update;
    }
    refined.push_back(
        AcceptedSweep{.sweep = first_pass[index].sweep, .bootstrap = std::move(frame)});
  }

  core::InitializationQuality final_quality;
  const auto final_estimate = align(refined, imu_buffer, preintegrator, final_quality, reason);
  if (!final_estimate.has_value()) {
    DynamicInitializationUpdate update;
    update.status = core::InitializationStatus::kCollecting;
    update.reason = "refined pass: " + reason;
    update.quality = std::move(final_quality);
    return update;
  }

  double maximum_rotation_change = 0.0;
  double maximum_translation_change = 0.0;
  for (std::size_t index = 0U; index < refined.size(); ++index) {
    const Sophus::SE3d change = first_pass[index].bootstrap.odom_from_lidar.inverse() *
                                refined[index].bootstrap.odom_from_lidar;
    maximum_rotation_change = std::max(maximum_rotation_change, change.so3().log().norm());
    maximum_translation_change = std::max(maximum_translation_change, change.translation().norm());
  }
  final_quality = final_estimate->quality;
  final_quality.refinement_rotation_change_rad = maximum_rotation_change;
  final_quality.refinement_translation_change_m = maximum_translation_change;
  if (!std::isfinite(maximum_rotation_change) ||
      maximum_rotation_change > options_.maximum_refinement_rotation_change_rad ||
      !std::isfinite(maximum_translation_change) ||
      maximum_translation_change > options_.maximum_refinement_translation_change_m) {
    DynamicInitializationUpdate update;
    update.status = core::InitializationStatus::kCollecting;
    update.reason = "the two complete bootstrap passes are not stable";
    update.quality = final_quality;
    return update;
  }

  const std::size_t last_fitted_index = refined.size() - 2U;
  const auto last_fitted_time = shifted(refined[last_fitted_index].sweep.acquisitionEnd(),
                                        options_.lidar_time_offset_to_imu_ns);
  const auto anchor_time =
      shifted(refined.back().sweep.acquisitionEnd(), options_.lidar_time_offset_to_imu_ns);
  if (!last_fitted_time.has_value() || !anchor_time.has_value()) {
    DynamicInitializationUpdate update;
    update.status = core::InitializationStatus::kCollecting;
    update.reason = "final dynamic anchor overflows the measurement timeline";
    update.quality = final_quality;
    return update;
  }
  const ImuIntervalResult held_out_interval = imu_buffer.interval(*last_fitted_time, *anchor_time);
  if (!held_out_interval.ok()) {
    DynamicInitializationUpdate update;
    update.status = core::InitializationStatus::kCollecting;
    update.reason = "final held-out interval lost exact IMU support";
    update.quality = final_quality;
    return update;
  }
  const PreintegrationResult held_out =
      preintegrator.integrate(*held_out_interval.value(), final_estimate->bias);
  if (!held_out.ok()) {
    DynamicInitializationUpdate update;
    update.status = core::InitializationStatus::kCollecting;
    update.reason = "final held-out IMU reintegration failed";
    update.quality = final_quality;
    return update;
  }

  const Sophus::SE3d imu_from_lidar = sophus(options_.imu_from_lidar);
  const Sophus::SE3d bootstrap_from_imu_fitted =
      refined[last_fitted_index].bootstrap.odom_from_lidar * imu_from_lidar.inverse();
  const Eigen::Vector3d anchor_velocity_bootstrap =
      final_estimate->velocities_bootstrap.back() +
      final_estimate->gravity_bootstrap * held_out.value()->durationSeconds() +
      bootstrap_from_imu_fitted.rotationMatrix() * eigen(held_out.value()->deltaVelocity());
  const Sophus::SE3d bootstrap_from_imu_anchor =
      refined.back().bootstrap.odom_from_lidar * imu_from_lidar.inverse();
  const Sophus::SE3d imu_from_base = sophus(options_.base_from_imu).inverse();
  const Sophus::SE3d bootstrap_from_base_anchor = bootstrap_from_imu_anchor * imu_from_base;

  const Eigen::Quaterniond gravity_alignment = Eigen::Quaterniond::FromTwoVectors(
      final_estimate->gravity_bootstrap, -Eigen::Vector3d::UnitZ() * options_.gravity_m_s2);
  const Eigen::Matrix3d raw_odom_from_bootstrap = gravity_alignment.toRotationMatrix();
  const Eigen::Matrix3d raw_odom_from_base =
      raw_odom_from_bootstrap * bootstrap_from_base_anchor.rotationMatrix();
  const Eigen::Matrix3d odom_from_base_rotation = removeYaw(raw_odom_from_base);
  const Eigen::Matrix3d yaw_gauge = odom_from_base_rotation * raw_odom_from_base.transpose();
  const Eigen::Matrix3d odom_from_bootstrap = yaw_gauge * raw_odom_from_bootstrap;
  const Eigen::Vector3d odom_from_bootstrap_translation =
      -odom_from_bootstrap * bootstrap_from_base_anchor.translation();
  const Sophus::SE3d odom_from_bootstrap_pose(odom_from_bootstrap, odom_from_bootstrap_translation);
  const Sophus::SE3d odom_from_imu_anchor = odom_from_bootstrap_pose * bootstrap_from_imu_anchor;
  const Eigen::Vector3d anchor_velocity_odom = odom_from_bootstrap * anchor_velocity_bootstrap;

  const auto support_begin =
      shifted(refined.front().sweep.acquisitionBegin(), options_.lidar_time_offset_to_imu_ns);
  const auto support_end = core::TimeNs::checkedAdd(*anchor_time, 1);
  if (!support_begin.has_value() || !support_end.has_value()) {
    DynamicInitializationUpdate update;
    update.status = core::InitializationStatus::kCollecting;
    update.reason = "dynamic initialization cannot form half-open support";
    update.quality = final_quality;
    return update;
  }

  final_quality.all_required_gates_passed = true;
  core::NavigationState seed(core::StateId(1U), *anchor_time, corePose(odom_from_imu_anchor),
                             coreVector(anchor_velocity_odom), final_estimate->bias);
  core::InitializationResult result(core::InitializationMode::kDynamic, *anchor_time,
                                    std::move(seed), core::TimeRange(*support_begin, *support_end),
                                    final_quality);
  accepted_sweeps_.assign(refined.begin(), refined.end());
  accepted_result_ = result;
  accepted_anchor_points_ = refined.back().bootstrap.registration_points;
  return DynamicInitializationUpdate{
      .status = core::InitializationStatus::kAccepted,
      .reason = "two-pass staged LiDAR-IMU alignment and held-out validation accepted",
      .quality = final_quality,
      .bootstrap_pose = std::nullopt,
      .result = std::move(result),
  };
}

DynamicInitializationUpdate DynamicInitializer::add(
    const core::LidarSweep& sweep, const ImuBuffer& imu_buffer,
    const GtsamCombinedPreintegrator& preintegrator) {
  if (accepted_result_.has_value()) {
    return DynamicInitializationUpdate{
        .status = core::InitializationStatus::kAccepted,
        .reason = "dynamic initialization was already accepted",
        .quality = accepted_result_->quality(),
        .bootstrap_pose = std::nullopt,
        .result = accepted_result_,
    };
  }

  if (!accepted_sweeps_.empty()) {
    const auto elapsed = core::TimeNs::checkedDifference(
        sweep.acquisitionEnd(), accepted_sweeps_.front().sweep.acquisitionEnd());
    if (!elapsed.has_value() || *elapsed > options_.maximum_support_ns) {
      bootstrap_.reset();
      accepted_sweeps_.clear();
      rejected_transitions_ = 0U;
    }
  }

  const auto deskewed = deskewRotationOnly(sweep, imu_buffer, options_.calibrated_bias_prior);
  if (!deskewed.has_value()) {
    return DynamicInitializationUpdate{
        .status = core::InitializationStatus::kCollecting,
        .reason = "waiting for exact IMU support for LiDAR rotational deskew",
        .quality = {},
        .bootstrap_pose = std::nullopt,
        .result = std::nullopt,
    };
  }

  BootstrapFrame frame = bootstrap_.add(sweep, *deskewed);
  const BootstrapPoseSummary summary = summarize(sweep, frame);
  if (!frame.accepted) {
    ++rejected_transitions_;
    DynamicInitializationUpdate update;
    update.status = core::InitializationStatus::kBootstrapping;
    update.reason = "temporary LiDAR bootstrap rejected the current sweep";
    update.bootstrap_pose = summary;
    update.quality.lidar_sweep_count = static_cast<std::uint32_t>(accepted_sweeps_.size());
    update.quality.rejected_transition_count = rejected_transitions_;
    return update;
  }

  accepted_sweeps_.push_back(AcceptedSweep{.sweep = sweep, .bootstrap = std::move(frame)});
  if (accepted_sweeps_.size() < options_.target_sweeps) {
    DynamicInitializationUpdate update;
    update.status = core::InitializationStatus::kBootstrapping;
    update.reason = "collecting the configured LiDAR bootstrap window";
    update.bootstrap_pose = summary;
    update.quality.lidar_sweep_count = static_cast<std::uint32_t>(accepted_sweeps_.size());
    update.quality.rejected_transition_count = rejected_transitions_;
    return update;
  }

  DynamicInitializationUpdate update = attempt(imu_buffer, preintegrator);
  update.bootstrap_pose = summary;
  if (update.status != core::InitializationStatus::kAccepted) {
    // A failed target-sized attempt starts a fresh bounded window. Rebuilding
    // and re-registering an overlapping nineteen-sweep target on every new
    // scan is quadratic work and cannot sustain live bag playback.
    bootstrap_.reset();
    accepted_sweeps_.clear();
    rejected_transitions_ = 0U;
  }
  return update;
}

}  // namespace meridian::local_rt::initialization
