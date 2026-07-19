#include "meridian/local/imu.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <sophus/so3.hpp>
#include <utility>

namespace meridian::local {
namespace {

[[nodiscard]] ImuBufferError bufferError(ImuBufferErrorCode code, std::string detail) {
  return ImuBufferError{code, std::move(detail)};
}

[[nodiscard]] InitializationError initializationError(InitializationErrorCode code,
                                                      std::string detail) {
  return InitializationError{code, std::move(detail)};
}

[[nodiscard]] PropagationError propagationError(PropagationErrorCode code, std::string detail) {
  return PropagationError{code, std::move(detail)};
}

[[nodiscard]] InterpolatedImuSample interpolate(const core::ImuSample& left,
                                                const core::ImuSample& right,
                                                core::FusionTime time) {
  if (time == left.stamp.fusion_time) {
    return InterpolatedImuSample{time, left.specific_force_mps2, left.angular_velocity_radps,
                                 left.id, left.id};
  }
  if (time == right.stamp.fusion_time) {
    return InterpolatedImuSample{time, right.specific_force_mps2, right.angular_velocity_radps,
                                 right.id, right.id};
  }
  const auto numerator = static_cast<double>((time - left.stamp.fusion_time).nanoseconds);
  const auto denominator =
      static_cast<double>((right.stamp.fusion_time - left.stamp.fusion_time).nanoseconds);
  const double alpha = numerator / denominator;
  return InterpolatedImuSample{
      time, (1.0 - alpha) * left.specific_force_mps2 + alpha * right.specific_force_mps2,
      (1.0 - alpha) * left.angular_velocity_radps + alpha * right.angular_velocity_radps, left.id,
      right.id};
}

[[nodiscard]] bool finiteState(const core::NavStateEstimate& state) {
  return state.T_odom_imu.matrix().allFinite() && state.velocity_odom.allFinite() &&
         state.gyro_bias.allFinite() && state.accel_bias.allFinite();
}

}  // namespace

ImuBuffer::ImuBuffer(ImuBufferConfig config) : config_(config) {}

core::Result<ImuAppendReport, ImuBufferError> ImuBuffer::append(core::ImuSample sample) {
  if (!sample.id.valid() || !sample.stamp.source_epoch.valid() ||
      sample.stamp.status == core::TimeMappingStatus::Discontinuous ||
      !sample.specific_force_mps2.allFinite() || !sample.angular_velocity_radps.allFinite()) {
    return core::Result<ImuAppendReport, ImuBufferError>::failure(
        bufferError(ImuBufferErrorCode::InvalidSample,
                    "IMU sample identity, clock mapping, or values are invalid"));
  }

  core::Duration newest_gap{};
  if (!samples_.empty()) {
    if (sample.stamp.source_epoch != samples_.back().stamp.source_epoch) {
      return core::Result<ImuAppendReport, ImuBufferError>::failure(bufferError(
          ImuBufferErrorCode::SourceEpochChanged, "source epoch change requires a new IMU buffer"));
    }
    newest_gap = sample.stamp.fusion_time - samples_.back().stamp.fusion_time;
    if (newest_gap.nanoseconds <= 0) {
      return core::Result<ImuAppendReport, ImuBufferError>::failure(bufferError(
          ImuBufferErrorCode::NonMonotonicTime, "IMU fusion times must be strictly increasing"));
    }
  }

  samples_.push_back(std::move(sample));
  while (samples_.size() > config_.maximum_samples) {
    samples_.pop_front();
  }
  while (samples_.size() > 2U && (samples_.back().stamp.fusion_time -
                                  samples_.front().stamp.fusion_time) > config_.maximum_span) {
    samples_.pop_front();
  }
  return core::Result<ImuAppendReport, ImuBufferError>::success(
      ImuAppendReport{samples_.size(), newest_gap});
}

core::Result<ImuInterval, ImuBufferError> ImuBuffer::interval(core::TimeRange support,
                                                              core::Duration nominal_period) const {
  using Result = core::Result<ImuInterval, ImuBufferError>;
  if (!support.valid() || nominal_period.nanoseconds <= 0) {
    return Result::failure(bufferError(ImuBufferErrorCode::InvalidInterval,
                                       "IMU interval and nominal period must both be positive"));
  }
  if (samples_.size() < 2U || samples_.front().stamp.fusion_time > support.start) {
    return Result::failure(bufferError(ImuBufferErrorCode::MissingPastBracket,
                                       "no IMU sample brackets interval start"));
  }
  if (samples_.back().stamp.fusion_time < support.end) {
    return Result::failure(bufferError(ImuBufferErrorCode::MissingFutureBracket,
                                       "no IMU sample brackets interval end"));
  }

  const auto lowerAt = [&](core::FusionTime time) {
    return std::lower_bound(samples_.begin(), samples_.end(), time,
                            [](const core::ImuSample& sample, core::FusionTime query) {
                              return sample.stamp.fusion_time < query;
                            });
  };
  const auto start_right = lowerAt(support.start);
  const auto end_right = lowerAt(support.end);
  if (start_right == samples_.end() || end_right == samples_.end()) {
    return Result::failure(
        bufferError(ImuBufferErrorCode::MissingFutureBracket, "no future IMU bracket"));
  }
  const auto start_left =
      start_right->stamp.fusion_time == support.start ? start_right : std::prev(start_right);
  const auto end_left =
      end_right->stamp.fusion_time == support.end ? end_right : std::prev(end_right);

  ImuInterval result;
  result.support = support;
  result.knots.push_back(interpolate(*start_left, *start_right, support.start));

  for (auto iterator = std::next(start_left);
       iterator != samples_.end() && iterator->stamp.fusion_time < support.end; ++iterator) {
    if (iterator->stamp.fusion_time > support.start) {
      result.knots.push_back(
          InterpolatedImuSample{iterator->stamp.fusion_time, iterator->specific_force_mps2,
                                iterator->angular_velocity_radps, iterator->id, iterator->id});
    }
  }
  if (result.knots.back().time != support.end) {
    result.knots.push_back(interpolate(*end_left, *end_right, support.end));
  }

  const auto raw_begin = start_left;
  const auto raw_end = std::next(end_right);
  for (auto iterator = raw_begin; iterator != raw_end; ++iterator) {
    result.raw_measurements.push_back(iterator->id);
    result.contains_saturation = result.contains_saturation || iterator->saturated;
    result.maximum_time_uncertainty.nanoseconds = std::max(
        result.maximum_time_uncertainty.nanoseconds, iterator->stamp.uncertainty.nanoseconds);
    const auto next = std::next(iterator);
    if (next != raw_end) {
      const core::Duration gap = next->stamp.fusion_time - iterator->stamp.fusion_time;
      result.maximum_raw_gap.nanoseconds =
          std::max(result.maximum_raw_gap.nanoseconds, gap.nanoseconds);
      if (gap.nanoseconds > 2 * nominal_period.nanoseconds) {
        return Result::failure(bufferError(ImuBufferErrorCode::EpochBreakingGap,
                                           "raw IMU gap exceeds two nominal periods"));
      }
      if (gap.nanoseconds * 2 > 3 * nominal_period.nanoseconds) {
        const auto rounded_periods =
            static_cast<std::size_t>(std::llround(static_cast<double>(gap.nanoseconds) /
                                                  static_cast<double>(nominal_period.nanoseconds)));
        result.inferred_missing_ticks += rounded_periods > 0U ? rounded_periods - 1U : 0U;
      }
    }
  }
  return Result::success(std::move(result));
}

void ImuBuffer::discardBefore(core::FusionTime time) {
  // Keep the newest sample before the cutoff so a future exact boundary can be
  // interpolated without duplicating or inventing support.
  while (samples_.size() > 1U && samples_[1].stamp.fusion_time <= time) {
    samples_.pop_front();
  }
}

core::Result<StationaryInitialization, InitializationError> initializeStationary(
    const ImuInterval& interval, const StationaryInitializerConfig& config) {
  using Result = core::Result<StationaryInitialization, InitializationError>;
  if (!interval.support.valid() || interval.support.duration() < config.minimum_support ||
      interval.knots.size() < 2U) {
    return Result::failure(
        initializationError(InitializationErrorCode::InsufficientSupport,
                            "stationary initialization lacks the configured contiguous duration"));
  }
  if (interval.maximum_raw_gap.nanoseconds * 2 > 3 * config.nominal_period.nanoseconds) {
    return Result::failure(initializationError(
        InitializationErrorCode::CoverageGap,
        "stationary initialization forbids a raw gap over 1.5 nominal periods"));
  }
  if (interval.contains_saturation) {
    return Result::failure(initializationError(
        InitializationErrorCode::Saturated, "stationary interval contains a saturated IMU sample"));
  }
  if (interval.maximum_time_uncertainty > config.maximum_time_uncertainty) {
    return Result::failure(
        initializationError(InitializationErrorCode::TimeUncertain,
                            "stationary interval clock uncertainty exceeds the profile bound"));
  }

  Eigen::Vector3d mean_acceleration = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_angular_rate = Eigen::Vector3d::Zero();
  for (const auto& sample : interval.knots) {
    mean_acceleration += sample.specific_force_mps2;
    mean_angular_rate += sample.angular_velocity_radps;
  }
  const double count = static_cast<double>(interval.knots.size());
  mean_acceleration /= count;
  mean_angular_rate /= count;

  double acceleration_variation = 0.0;
  double angular_rate_variation = 0.0;
  for (const auto& sample : interval.knots) {
    acceleration_variation += (sample.specific_force_mps2 - mean_acceleration).squaredNorm();
    angular_rate_variation += (sample.angular_velocity_radps - mean_angular_rate).squaredNorm();
  }
  const double denominator =
      static_cast<double>(std::max<std::size_t>(1U, interval.knots.size() - 1U));
  const double acceleration_stddev = std::sqrt(acceleration_variation / denominator);
  const double angular_rate_stddev = std::sqrt(angular_rate_variation / denominator);

  if (mean_angular_rate.norm() > config.maximum_mean_angular_rate_radps ||
      angular_rate_stddev > config.maximum_gyro_stddev_radps ||
      acceleration_stddev > config.maximum_accel_stddev_mps2) {
    return Result::failure(
        initializationError(InitializationErrorCode::Moving,
                            "IMU mean/stddev tests reject the interval as non-stationary"));
  }
  if (std::abs(mean_acceleration.norm() - config.gravity_mps2) >
          config.maximum_gravity_norm_error_mps2 ||
      mean_acceleration.norm() < std::numeric_limits<double>::epsilon()) {
    return Result::failure(
        initializationError(InitializationErrorCode::GravityImplausible,
                            "specific-force norm is inconsistent with configured gravity"));
  }

  const Eigen::Quaterniond q_odom_imu =
      Eigen::Quaterniond::FromTwoVectors(mean_acceleration.normalized(), Eigen::Vector3d::UnitZ());
  StationaryInitialization initialization;
  initialization.state.T_odom_imu =
      core::Pose3d(Sophus::SO3d(q_odom_imu.normalized()), Eigen::Vector3d::Zero());
  initialization.state.velocity_odom.setZero();
  initialization.state.gyro_bias = mean_angular_rate;
  initialization.state.accel_bias.setZero();
  initialization.mean_specific_force = mean_acceleration;
  initialization.mean_angular_velocity = mean_angular_rate;
  initialization.acceleration_stddev_mps2 = acceleration_stddev;
  initialization.angular_rate_stddev_radps = angular_rate_stddev;
  initialization.samples = interval.knots.size();

  // Order: [R, v, p, bg, ba]. Position and yaw define the odom gauge and are
  // tight. Roll/pitch, velocity, and gyro bias come from sample statistics;
  // accelerometer bias remains deliberately broad because static data does not
  // observe it independently from gravity.
  auto& covariance = initialization.covariance.matrix;
  covariance.setZero();
  const double tilt_variance =
      std::pow(std::max(angular_rate_stddev, acceleration_stddev / config.gravity_mps2), 2);
  covariance.diagonal().segment<3>(0) << tilt_variance, tilt_variance, 1.0e-8;
  covariance.diagonal().segment<3>(3).setConstant(
      std::max(1.0e-6, acceleration_variation / (count * count)));
  covariance.diagonal().segment<3>(6).setConstant(1.0e-8);
  covariance.diagonal().segment<3>(9).setConstant(
      std::max(1.0e-10, angular_rate_variation / (count * count)));
  covariance.diagonal().segment<3>(12).setConstant(config.accelerometer_bias_prior_sigma_mps2 *
                                                   config.accelerometer_bias_prior_sigma_mps2);
  return Result::success(std::move(initialization));
}

MidpointImuPropagator::MidpointImuPropagator(Eigen::Vector3d gravity_odom)
    : gravity_odom_(std::move(gravity_odom)) {}

core::Result<PropagationResult, PropagationError> MidpointImuPropagator::propagate(
    core::FusionTime anchor_time, const core::NavStateEstimate& anchor,
    const ImuInterval& interval) const {
  using Result = core::Result<PropagationResult, PropagationError>;
  if (interval.knots.size() < 2U || !interval.support.valid()) {
    return Result::failure(
        propagationError(PropagationErrorCode::EmptySupport,
                         "midpoint propagation requires at least two supported knots"));
  }
  if (interval.support.start != anchor_time || interval.knots.front().time != anchor_time) {
    return Result::failure(
        propagationError(PropagationErrorCode::AnchorMismatch,
                         "propagation support must begin exactly at the anchor time"));
  }
  if (!finiteState(anchor) || !gravity_odom_.allFinite()) {
    return Result::failure(propagationError(PropagationErrorCode::NonFiniteState,
                                            "anchor state or gravity is non-finite"));
  }

  PropagationResult result;
  result.final_state = anchor;
  result.trajectory.push_back(TimedNavState{anchor_time, anchor});
  result.raw_measurements = interval.raw_measurements;
  result.inferred_missing_ticks = interval.inferred_missing_ticks;

  for (std::size_t index = 1U; index < interval.knots.size(); ++index) {
    const auto& previous = interval.knots[index - 1U];
    const auto& current = interval.knots[index];
    const double dt = static_cast<double>((current.time - previous.time).nanoseconds) * 1.0e-9;
    if (!std::isfinite(dt) || dt <= 0.0) {
      return Result::failure(
          propagationError(PropagationErrorCode::InvalidTimeStep,
                           "midpoint propagation encountered a non-positive time step"));
    }

    auto& state = result.final_state;
    const Eigen::Vector3d omega_mid =
        0.5 * (previous.angular_velocity_radps + current.angular_velocity_radps) - state.gyro_bias;
    const Sophus::SO3d rotation_previous = state.T_odom_imu.so3();
    const Sophus::SO3d rotation_current = rotation_previous * Sophus::SO3d::exp(omega_mid * dt);
    const Eigen::Vector3d acceleration_previous =
        rotation_previous * (previous.specific_force_mps2 - state.accel_bias) + gravity_odom_;
    const Eigen::Vector3d acceleration_current =
        rotation_current * (current.specific_force_mps2 - state.accel_bias) + gravity_odom_;
    const Eigen::Vector3d acceleration_mid = 0.5 * (acceleration_previous + acceleration_current);

    const Eigen::Vector3d position = state.T_odom_imu.translation() + state.velocity_odom * dt +
                                     0.5 * acceleration_mid * dt * dt;
    state.velocity_odom += acceleration_mid * dt;
    state.T_odom_imu = core::Pose3d(rotation_current, position);
    if (!finiteState(state)) {
      return Result::failure(propagationError(PropagationErrorCode::NonFiniteState,
                                              "midpoint propagation produced a non-finite state"));
    }
    result.trajectory.push_back(TimedNavState{current.time, state});
  }
  return Result::success(std::move(result));
}

core::Result<PropagationResult, PropagationError> MidpointImuPropagator::propagateBackwards(
    core::FusionTime anchor_time, const core::NavStateEstimate& anchor,
    const ImuInterval& before_anchor) const {
  using Result = core::Result<PropagationResult, PropagationError>;
  if (before_anchor.knots.size() < 2U || !before_anchor.support.valid()) {
    return Result::failure(
        propagationError(PropagationErrorCode::EmptySupport,
                         "backward midpoint propagation requires non-empty pre-anchor support"));
  }
  if (before_anchor.support.end != anchor_time || before_anchor.knots.back().time != anchor_time) {
    return Result::failure(
        propagationError(PropagationErrorCode::AnchorMismatch,
                         "backward propagation support must end exactly at the committed anchor"));
  }
  if (!finiteState(anchor) || !gravity_odom_.allFinite()) {
    return Result::failure(
        propagationError(PropagationErrorCode::NonFiniteState,
                         "backward propagation anchor state or gravity is non-finite"));
  }

  std::vector<TimedNavState> reversed;
  reversed.reserve(before_anchor.knots.size());
  core::NavStateEstimate earlier = anchor;
  reversed.push_back(TimedNavState{anchor_time, earlier});
  for (std::size_t right_index = before_anchor.knots.size() - 1U; right_index > 0U; --right_index) {
    const InterpolatedImuSample& previous = before_anchor.knots[right_index - 1U];
    const InterpolatedImuSample& current = before_anchor.knots[right_index];
    const double dt = static_cast<double>((current.time - previous.time).nanoseconds) * 1.0e-9;
    if (!std::isfinite(dt) || dt <= 0.0) {
      return Result::failure(
          propagationError(PropagationErrorCode::InvalidTimeStep,
                           "backward midpoint propagation encountered a non-positive time step"));
    }

    const Eigen::Vector3d omega_mid =
        0.5 * (previous.angular_velocity_radps + current.angular_velocity_radps) -
        earlier.gyro_bias;
    const Sophus::SO3d rotation_current = earlier.T_odom_imu.so3();
    const Sophus::SO3d rotation_previous = rotation_current * Sophus::SO3d::exp(-omega_mid * dt);
    const Eigen::Vector3d acceleration_previous =
        rotation_previous * (previous.specific_force_mps2 - earlier.accel_bias) + gravity_odom_;
    const Eigen::Vector3d acceleration_current =
        rotation_current * (current.specific_force_mps2 - earlier.accel_bias) + gravity_odom_;
    const Eigen::Vector3d acceleration_mid = 0.5 * (acceleration_previous + acceleration_current);
    const Eigen::Vector3d velocity_previous = earlier.velocity_odom - acceleration_mid * dt;
    const Eigen::Vector3d position_previous = earlier.T_odom_imu.translation() -
                                              velocity_previous * dt -
                                              0.5 * acceleration_mid * dt * dt;
    earlier.velocity_odom = velocity_previous;
    earlier.T_odom_imu = core::Pose3d(rotation_previous, position_previous);
    if (!finiteState(earlier)) {
      return Result::failure(
          propagationError(PropagationErrorCode::NonFiniteState,
                           "backward midpoint propagation produced a non-finite state"));
    }
    reversed.push_back(TimedNavState{previous.time, earlier});
  }

  PropagationResult result;
  result.final_state = anchor;
  result.trajectory.assign(reversed.rbegin(), reversed.rend());
  result.raw_measurements = before_anchor.raw_measurements;
  result.inferred_missing_ticks = before_anchor.inferred_missing_ticks;
  return Result::success(std::move(result));
}

core::Result<PropagationResult, PropagationError> MidpointImuPropagator::propagateAround(
    core::FusionTime anchor_time, const core::NavStateEstimate& anchor,
    const ImuInterval& before_anchor, const ImuInterval& after_anchor) const {
  using Result = core::Result<PropagationResult, PropagationError>;
  if (before_anchor.knots.size() < 2U || after_anchor.knots.size() < 2U ||
      !before_anchor.support.valid() || !after_anchor.support.valid()) {
    return Result::failure(propagationError(
        PropagationErrorCode::EmptySupport,
        "bidirectional midpoint propagation requires non-empty support on both sides"));
  }
  if (before_anchor.support.end != anchor_time || before_anchor.knots.back().time != anchor_time ||
      after_anchor.support.start != anchor_time || after_anchor.knots.front().time != anchor_time) {
    return Result::failure(propagationError(
        PropagationErrorCode::AnchorMismatch,
        "bidirectional propagation support must meet exactly at the committed anchor"));
  }
  if (!finiteState(anchor) || !gravity_odom_.allFinite()) {
    return Result::failure(
        propagationError(PropagationErrorCode::NonFiniteState,
                         "bidirectional propagation anchor state or gravity is non-finite"));
  }

  auto backward = propagateBackwards(anchor_time, anchor, before_anchor);
  if (!backward) {
    return Result::failure(backward.error());
  }
  auto forward = propagate(anchor_time, anchor, after_anchor);
  if (!forward) {
    return Result::failure(forward.error());
  }

  PropagationResult result;
  result.final_state = forward.value().final_state;
  result.trajectory = std::move(backward).value().trajectory;
  result.trajectory.reserve(result.trajectory.size() + forward.value().trajectory.size() - 1U);
  result.trajectory.insert(result.trajectory.end(), std::next(forward.value().trajectory.begin()),
                           forward.value().trajectory.end());
  result.raw_measurements = before_anchor.raw_measurements;
  for (const core::MeasurementId measurement : forward.value().raw_measurements) {
    if (std::find(result.raw_measurements.begin(), result.raw_measurements.end(), measurement) ==
        result.raw_measurements.end()) {
      result.raw_measurements.push_back(measurement);
    }
  }
  if (before_anchor.inferred_missing_ticks >
      std::numeric_limits<std::size_t>::max() - forward.value().inferred_missing_ticks) {
    return Result::failure(
        propagationError(PropagationErrorCode::InvalidTimeStep,
                         "bidirectional midpoint missing-tick accounting overflowed"));
  }
  result.inferred_missing_ticks =
      before_anchor.inferred_missing_ticks + forward.value().inferred_missing_ticks;
  return Result::success(std::move(result));
}

}  // namespace meridian::local
