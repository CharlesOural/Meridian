#include "meridian/global/gnss_interpolation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include <Eigen/Eigenvalues>

namespace meridian::global {
namespace {

[[nodiscard]] GnssInterpolationError interpolationError(
    GnssInterpolationErrorCode code, std::string detail) {
  return GnssInterpolationError{code, std::move(detail)};
}

[[nodiscard]] Eigen::Matrix3d skew(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d result;
  result << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(),
      -vector.y(), vector.x(), 0.0;
  return result;
}

[[nodiscard]] Eigen::Matrix3d so3RightJacobian(
    const Eigen::Vector3d& tangent) {
  const double angle = tangent.norm();
  const Eigen::Matrix3d tangent_skew = skew(tangent);
  if (angle < 1.0e-7) {
    return Eigen::Matrix3d::Identity() - 0.5 * tangent_skew +
           (1.0 / 6.0) * tangent_skew * tangent_skew;
  }
  const double angle_squared = angle * angle;
  return Eigen::Matrix3d::Identity() -
         ((1.0 - std::cos(angle)) / angle_squared) * tangent_skew +
         ((angle - std::sin(angle)) / (angle_squared * angle)) *
             tangent_skew * tangent_skew;
}

[[nodiscard]] Eigen::Matrix3d so3RightJacobianInverse(
    const Eigen::Vector3d& tangent) {
  const double angle = tangent.norm();
  const Eigen::Matrix3d tangent_skew = skew(tangent);
  if (angle < 1.0e-7) {
    return Eigen::Matrix3d::Identity() + 0.5 * tangent_skew +
           (1.0 / 12.0) * tangent_skew * tangent_skew;
  }
  const double angle_squared = angle * angle;
  const double coefficient =
      1.0 / angle_squared -
      (1.0 + std::cos(angle)) /
          (2.0 * angle * std::sin(angle));
  return Eigen::Matrix3d::Identity() + 0.5 * tangent_skew +
         coefficient * tangent_skew * tangent_skew;
}

template <int Size>
[[nodiscard]] bool symmetricPositiveSemidefinite(
    const Eigen::Matrix<double, Size, Size>& covariance,
    bool require_positive_definite = false) {
  if (!covariance.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, covariance.cwiseAbs().maxCoeff());
  if ((covariance - covariance.transpose()).cwiseAbs().maxCoeff() >
      1.0e-10 * scale) {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Size, Size>> eigen(
      0.5 * (covariance + covariance.transpose()),
      Eigen::EigenvaluesOnly);
  if (eigen.info() != Eigen::Success || !eigen.eigenvalues().allFinite()) {
    return false;
  }
  const double tolerance = 1.0e-12 * scale;
  return require_positive_definite
             ? eigen.eigenvalues().minCoeff() > tolerance
             : eigen.eigenvalues().minCoeff() >= -tolerance;
}

[[nodiscard]] std::optional<double> positiveDefiniteCondition(
    const Eigen::Matrix<double, 18, 18>& covariance) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 18, 18>> eigen(
      0.5 * (covariance + covariance.transpose()),
      Eigen::EigenvaluesOnly);
  if (eigen.info() != Eigen::Success || !eigen.eigenvalues().allFinite() ||
      eigen.eigenvalues().minCoeff() <= 0.0) {
    return std::nullopt;
  }
  return eigen.eigenvalues().maxCoeff() /
         eigen.eigenvalues().minCoeff();
}

[[nodiscard]] bool validConfig(const GnssInterpolationConfig& config) {
  return config.maximum_segment_duration.nanoseconds > 0 &&
         std::isfinite(config.maximum_relative_rotation_rad) &&
         config.maximum_relative_rotation_rad > 0.0 &&
         config.maximum_relative_rotation_rad < 3.14159265358979323846 &&
         std::isfinite(config.maximum_covariance_condition) &&
         config.maximum_covariance_condition > 1.0;
}

}  // namespace

core::Result<InterpolatedAntennaState, GnssInterpolationError>
interpolateFinalAntennaState(const GnssInterpolationInput& input,
                             const GnssInterpolationConfig& config) {
  using Result =
      core::Result<InterpolatedAntennaState, GnssInterpolationError>;
  if (!validConfig(config)) {
    return Result::failure(interpolationError(
        GnssInterpolationErrorCode::InvalidConfiguration,
        "GNSS interpolation duration, rotation, or conditioning limit is invalid"));
  }
  if (!input.first.state.valid() || !input.second.state.valid() ||
      input.first.state >= input.second.state) {
    return Result::failure(interpolationError(
        GnssInterpolationErrorCode::InvalidIdentity,
        "final trajectory endpoint identities must be valid and increasing"));
  }
  if (input.first.exact_time >= input.second.exact_time) {
    return Result::failure(interpolationError(
        GnssInterpolationErrorCode::InvalidTimeOrder,
        "final trajectory endpoint times must increase strictly"));
  }
  if (input.observation_time < input.first.exact_time ||
      input.observation_time > input.second.exact_time) {
    return Result::failure(interpolationError(
        GnssInterpolationErrorCode::ObservationOutsideSegment,
        "GNSS observation is not bracketed by the finalized segment"));
  }
  const core::Duration segment_duration =
      input.second.exact_time - input.first.exact_time;
  if (segment_duration > config.maximum_segment_duration) {
    return Result::failure(interpolationError(
        GnssInterpolationErrorCode::SegmentTooLong,
        "finalized trajectory bracket exceeds the interpolation duration limit"));
  }
  if (!input.first.T_submap_imu.matrix().allFinite() ||
      !input.second.T_submap_imu.matrix().allFinite() ||
      !input.first.velocity_submap.allFinite() ||
      !input.second.velocity_submap.allFinite() ||
      !input.lever_arm.imuToAntennaInImu().allFinite()) {
    return Result::failure(interpolationError(
        GnssInterpolationErrorCode::NonFiniteInput,
        "trajectory endpoints or antenna lever arm contain non-finite values"));
  }
  if (input.endpoint_covariance.order !=
          JointEndpointCovarianceOrder::
              RightLocalTranslationRotationVelocityPerEndpoint ||
      !symmetricPositiveSemidefinite<18>(input.endpoint_covariance.matrix,
                                         true) ||
      !symmetricPositiveSemidefinite<3>(input.lever_arm_covariance_imu) ||
      !symmetricPositiveSemidefinite<3>(
          input.interpolation_model_covariance_submap)) {
    return Result::failure(interpolationError(
        GnssInterpolationErrorCode::InvalidCovariance,
        "endpoint, lever-arm, and interpolation covariances must follow the declared finite PSD order"));
  }
  const auto condition =
      positiveDefiniteCondition(input.endpoint_covariance.matrix);
  if (!condition || *condition > config.maximum_covariance_condition) {
    return Result::failure(interpolationError(
        GnssInterpolationErrorCode::IllConditionedCovariance,
        "joint endpoint pose/velocity covariance is singular or ill-conditioned"));
  }

  const double duration_s =
      static_cast<double>(segment_duration.nanoseconds) * 1.0e-9;
  const double alpha =
      static_cast<double>((input.observation_time - input.first.exact_time)
                              .nanoseconds) /
      static_cast<double>(segment_duration.nanoseconds);
  const Sophus::SO3d& R0 = input.first.T_submap_imu.so3();
  const Sophus::SO3d& R1 = input.second.T_submap_imu.so3();
  const Sophus::SO3d relative = R0.inverse() * R1;
  const Eigen::Vector3d relative_tangent = relative.log();
  const double rotation_angle = relative_tangent.norm();
  if (!relative_tangent.allFinite() ||
      rotation_angle > config.maximum_relative_rotation_rad) {
    return Result::failure(interpolationError(
        GnssInterpolationErrorCode::RelativeRotationUnsupported,
        "endpoint relative rotation exceeds the calibrated shortest-branch limit"));
  }

  const double alpha_squared = alpha * alpha;
  const double alpha_cubed = alpha_squared * alpha;
  const double h00 = 2.0 * alpha_cubed - 3.0 * alpha_squared + 1.0;
  const double h10 = alpha_cubed - 2.0 * alpha_squared + alpha;
  const double h01 = -2.0 * alpha_cubed + 3.0 * alpha_squared;
  const double h11 = alpha_cubed - alpha_squared;
  const Eigen::Vector3d p0 = input.first.T_submap_imu.translation();
  const Eigen::Vector3d p1 = input.second.T_submap_imu.translation();
  const Eigen::Vector3d body_position =
      h00 * p0 + h10 * duration_s * input.first.velocity_submap +
      h01 * p1 + h11 * duration_s * input.second.velocity_submap;

  const Eigen::Vector3d scaled_tangent = alpha * relative_tangent;
  const Sophus::SO3d R_alpha = R0 * Sophus::SO3d::exp(scaled_tangent);
  const Eigen::Matrix3d E_alpha =
      Sophus::SO3d::exp(scaled_tangent).matrix();
  const Eigen::Matrix3d relative_matrix = relative.matrix();
  const Eigen::Matrix3d differential =
      alpha * so3RightJacobian(scaled_tangent) *
      so3RightJacobianInverse(relative_tangent);
  const Eigen::Matrix3d orientation_from_first =
      E_alpha.transpose() - differential * relative_matrix.transpose();
  const Eigen::Matrix3d orientation_from_second = differential;

  const Eigen::Vector3d lever = input.lever_arm.imuToAntennaInImu();
  InterpolatedAntennaState result;
  result.exact_time = input.observation_time;
  result.body_position_submap = body_position;
  result.R_submap_imu = R_alpha;
  result.antenna_position_submap = body_position + R_alpha * lever;
  auto& jacobian = result.antenna_jacobian_endpoints;
  jacobian.block<3, 3>(0, 0) = h00 * R0.matrix();
  jacobian.block<3, 3>(0, 3) =
      -R_alpha.matrix() * skew(lever) * orientation_from_first;
  jacobian.block<3, 3>(0, 6) =
      h10 * duration_s * Eigen::Matrix3d::Identity();
  jacobian.block<3, 3>(0, 9) = h01 * R1.matrix();
  jacobian.block<3, 3>(0, 12) =
      -R_alpha.matrix() * skew(lever) * orientation_from_second;
  jacobian.block<3, 3>(0, 15) =
      h11 * duration_s * Eigen::Matrix3d::Identity();
  result.antenna_jacobian_lever_arm = R_alpha.matrix();
  result.antenna_covariance_submap =
      jacobian * input.endpoint_covariance.matrix * jacobian.transpose() +
      result.antenna_jacobian_lever_arm *
          input.lever_arm_covariance_imu *
          result.antenna_jacobian_lever_arm.transpose() +
      input.interpolation_model_covariance_submap;
  result.antenna_covariance_submap =
      0.5 * (result.antenna_covariance_submap +
             result.antenna_covariance_submap.transpose());
  result.diagnostics = GnssInterpolationDiagnostics{
      alpha, duration_s, rotation_angle, *condition};
  if (!result.body_position_submap.allFinite() ||
      !result.R_submap_imu.matrix().allFinite() ||
      !result.antenna_position_submap.allFinite() ||
      !result.antenna_jacobian_endpoints.allFinite() ||
      !result.antenna_covariance_submap.allFinite() ||
      !symmetricPositiveSemidefinite<3>(result.antenna_covariance_submap)) {
    return Result::failure(interpolationError(
        GnssInterpolationErrorCode::NumericalFailure,
        "Hermite/geodesic interpolation produced a non-finite or indefinite result"));
  }
  return Result::success(std::move(result));
}

}  // namespace meridian::global
