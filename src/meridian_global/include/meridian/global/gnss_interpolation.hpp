#pragma once

#include <cstddef>
#include <string>

#include <Eigen/Core>

#include "meridian/core/api.hpp"
#include "meridian/global/gnss.hpp"

namespace meridian::global {

struct FinalTrajectoryKnot {
  core::StateId state;
  core::FusionTime exact_time;
  core::Pose3d T_submap_imu;
  Eigen::Vector3d velocity_submap{Eigen::Vector3d::Zero()};
};

enum class JointEndpointCovarianceOrder {
  RightLocalTranslationRotationVelocityPerEndpoint,
};

// Fixed block order:
// [dp0^I0, dtheta0^I0, dv0^S, dp1^I1, dtheta1^I1, dv1^S].
struct JointEndpointPoseVelocityCovariance {
  Eigen::Matrix<double, 18, 18> matrix{
      Eigen::Matrix<double, 18, 18>::Zero()};
  JointEndpointCovarianceOrder order{
      JointEndpointCovarianceOrder::
          RightLocalTranslationRotationVelocityPerEndpoint};
};

struct GnssInterpolationConfig {
  core::Duration maximum_segment_duration{500'000'000LL};
  double maximum_relative_rotation_rad{2.8};
  double maximum_covariance_condition{1.0e14};
};

struct GnssInterpolationInput {
  FinalTrajectoryKnot first;
  FinalTrajectoryKnot second;
  core::FusionTime observation_time;
  AntennaLeverArm lever_arm{AntennaLeverArm::fromImuToAntenna(
                               Eigen::Vector3d::Zero())
                               .value()};
  JointEndpointPoseVelocityCovariance endpoint_covariance;
  Eigen::Matrix3d lever_arm_covariance_imu{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d interpolation_model_covariance_submap{
      Eigen::Matrix3d::Zero()};
};

enum class GnssInterpolationErrorCode {
  InvalidConfiguration,
  InvalidIdentity,
  InvalidTimeOrder,
  ObservationOutsideSegment,
  SegmentTooLong,
  RelativeRotationUnsupported,
  NonFiniteInput,
  InvalidCovariance,
  IllConditionedCovariance,
  NumericalFailure,
};

struct GnssInterpolationError {
  GnssInterpolationErrorCode code{GnssInterpolationErrorCode::NonFiniteInput};
  std::string detail;
};

struct GnssInterpolationDiagnostics {
  double alpha{};
  double segment_duration_s{};
  double relative_rotation_rad{};
  double endpoint_covariance_condition{};
};

struct InterpolatedAntennaState {
  core::FusionTime exact_time;
  Eigen::Vector3d body_position_submap{Eigen::Vector3d::Zero()};
  Sophus::SO3d R_submap_imu;
  Eigen::Vector3d antenna_position_submap{Eigen::Vector3d::Zero()};
  // Columns follow JointEndpointPoseVelocityCovariance's declared order.
  Eigen::Matrix<double, 3, 18> antenna_jacobian_endpoints{
      Eigen::Matrix<double, 3, 18>::Zero()};
  Eigen::Matrix3d antenna_jacobian_lever_arm{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d antenna_covariance_submap{Eigen::Matrix3d::Zero()};
  GnssInterpolationDiagnostics diagnostics;
};

// Deterministic interpolation between finalized discrete states. Translation
// is cubic Hermite using endpoint velocities; rotation follows the shortest
// SO(3) geodesic. No continuous-time optimization variable is introduced.
[[nodiscard]] core::Result<InterpolatedAntennaState, GnssInterpolationError>
interpolateFinalAntennaState(
    const GnssInterpolationInput& input,
    const GnssInterpolationConfig& config = {});

}  // namespace meridian::global
