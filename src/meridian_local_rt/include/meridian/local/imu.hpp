#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"

namespace meridian::local {

enum class ImuBufferErrorCode {
  InvalidSample,
  SourceEpochChanged,
  NonMonotonicTime,
  InvalidInterval,
  MissingPastBracket,
  MissingFutureBracket,
  EpochBreakingGap,
};

struct ImuBufferError {
  ImuBufferErrorCode code{};
  std::string detail;
};

struct ImuAppendReport {
  std::size_t retained_samples{};
  core::Duration newest_gap{};
};

struct InterpolatedImuSample {
  core::FusionTime time;
  Eigen::Vector3d specific_force_mps2{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_velocity_radps{Eigen::Vector3d::Zero()};
  core::MeasurementId left_source;
  core::MeasurementId right_source;
};

struct ImuInterval {
  core::TimeRange support;
  std::vector<InterpolatedImuSample> knots;
  std::vector<core::MeasurementId> raw_measurements;
  core::Duration maximum_raw_gap{};
  core::Duration maximum_time_uncertainty{};
  std::size_t inferred_missing_ticks{};
  bool contains_saturation{};
};

struct ImuBufferConfig {
  std::size_t maximum_samples{4'096};
  core::Duration maximum_span{20'000'000'000LL};
};

class ImuBuffer {
public:
  explicit ImuBuffer(ImuBufferConfig config = {});

  [[nodiscard]] core::Result<ImuAppendReport, ImuBufferError> append(core::ImuSample sample);

  // Extracts exact boundary-interpolated support. A raw gap above two nominal
  // periods is a hard failure; shorter inferred missing ticks remain explicit
  // in the returned report.
  [[nodiscard]] core::Result<ImuInterval, ImuBufferError> interval(
      core::TimeRange support, core::Duration nominal_period) const;

  void discardBefore(core::FusionTime time);
  [[nodiscard]] std::size_t size() const noexcept { return samples_.size(); }
  [[nodiscard]] bool empty() const noexcept { return samples_.empty(); }

private:
  ImuBufferConfig config_;
  std::deque<core::ImuSample> samples_;
};

struct StationaryInitializerConfig {
  core::Duration minimum_support{2'000'000'000LL};
  core::Duration nominal_period{5'000'000LL};
  core::Duration maximum_time_uncertainty{1'000'000LL};
  double gravity_mps2{9.80665};
  double maximum_mean_angular_rate_radps{0.05};
  double maximum_gyro_stddev_radps{0.01};
  double maximum_gravity_norm_error_mps2{0.5};
  double maximum_accel_stddev_mps2{0.2};
  double accelerometer_bias_prior_sigma_mps2{0.2};
};

enum class InitializationErrorCode {
  InsufficientSupport,
  CoverageGap,
  Saturated,
  TimeUncertain,
  Moving,
  GravityImplausible,
};

struct InitializationError {
  InitializationErrorCode code{};
  std::string detail;
};

enum class NavigationCovarianceOrder {
  RotationVelocityPositionGyroBiasAccelBias,
};

struct NavigationCovariance {
  Eigen::Matrix<double, 15, 15> matrix{Eigen::Matrix<double, 15, 15>::Zero()};
  NavigationCovarianceOrder order{
      NavigationCovarianceOrder::RotationVelocityPositionGyroBiasAccelBias};
};

struct StationaryInitialization {
  core::NavStateEstimate state;
  NavigationCovariance covariance;
  Eigen::Vector3d mean_specific_force{Eigen::Vector3d::Zero()};
  Eigen::Vector3d mean_angular_velocity{Eigen::Vector3d::Zero()};
  double acceleration_stddev_mps2{};
  double angular_rate_stddev_radps{};
  std::size_t samples{};
};

[[nodiscard]] core::Result<StationaryInitialization, InitializationError> initializeStationary(
    const ImuInterval& interval, const StationaryInitializerConfig& config = {});

struct TimedNavState {
  core::FusionTime time;
  core::NavStateEstimate state;
};

struct PropagationResult {
  core::NavStateEstimate final_state;
  std::vector<TimedNavState> trajectory;
  std::vector<core::MeasurementId> raw_measurements;
  std::size_t inferred_missing_ticks{};
};

enum class PropagationErrorCode {
  EmptySupport,
  AnchorMismatch,
  InvalidTimeStep,
  NonFiniteState,
};

struct PropagationError {
  PropagationErrorCode code{};
  std::string detail;
};

class MidpointImuPropagator {
public:
  explicit MidpointImuPropagator(Eigen::Vector3d gravity_odom = Eigen::Vector3d{0.0, 0.0,
                                                                                -9.80665});

  [[nodiscard]] core::Result<PropagationResult, PropagationError> propagate(
      core::FusionTime anchor_time, const core::NavStateEstimate& anchor,
      const ImuInterval& interval) const;

  // Reconstructs a strictly chronological trajectory ending at an already
  // committed state. This is the sensor-neutral late-factor path: a LiDAR
  // sweep whose reference time exactly shares a visual/guard state can still
  // be deskewed without creating or reinserting an IMU edge. `final_state`
  // remains the supplied state at `anchor_time`.
  [[nodiscard]] core::Result<PropagationResult, PropagationError> propagateBackwards(
      core::FusionTime anchor_time, const core::NavStateEstimate& anchor,
      const ImuInterval& before_anchor) const;

  // Builds one strictly chronological trajectory around an already committed
  // state inside a sensor acquisition. `before_anchor` must end exactly at
  // the anchor and `after_anchor` must begin there. Backward midpoint steps
  // reconstruct the earlier states; forward steps are identical to
  // propagate(). The returned final_state is at after_anchor.support.end.
  [[nodiscard]] core::Result<PropagationResult, PropagationError> propagateAround(
      core::FusionTime anchor_time, const core::NavStateEstimate& anchor,
      const ImuInterval& before_anchor, const ImuInterval& after_anchor) const;

private:
  Eigen::Vector3d gravity_odom_;
};

}  // namespace meridian::local
