#include "meridian/local/equidistant_camera.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace meridian::local {
namespace {

constexpr double kProjectionEpsilon = 1.0e-12;
constexpr double kRadialEpsilon = 1.0e-8;
constexpr double kHalfPiMargin = 1.5707963267948966 - 1.0e-7;

[[nodiscard]] bool finiteParameters(const EquidistantCameraParameters& parameters) noexcept {
  return std::isfinite(parameters.fx) && std::isfinite(parameters.fy) &&
         std::isfinite(parameters.cx) && std::isfinite(parameters.cy) &&
         std::isfinite(parameters.k1) && std::isfinite(parameters.k2) &&
         std::isfinite(parameters.k3) && std::isfinite(parameters.k4);
}

}  // namespace

EquidistantCamera::EquidistantCamera(EquidistantCameraParameters parameters)
    : parameters_(parameters) {
  valid_ = parameters_.width > 1U && parameters_.height > 1U && parameters_.fx > 0.0 &&
           parameters_.fy > 0.0 && finiteParameters(parameters_);
  if (!valid_) {
    return;
  }

  // A radial inverse is only well-defined while theta_d is monotonic. Check
  // the complete front-facing hemisphere, not only the calibrated image rim.
  constexpr int kMonotonicitySamples = 128;
  for (int index = 0; index <= kMonotonicitySamples; ++index) {
    const double theta =
        kHalfPiMargin * static_cast<double>(index) / static_cast<double>(kMonotonicitySamples);
    if (!(distortedThetaDerivative(theta) > kProjectionEpsilon)) {
      valid_ = false;
      break;
    }
  }
}

double EquidistantCamera::distortedTheta(double theta) const noexcept {
  const double theta2 = theta * theta;
  const double theta4 = theta2 * theta2;
  const double theta6 = theta4 * theta2;
  const double theta8 = theta4 * theta4;
  return theta * (1.0 + parameters_.k1 * theta2 + parameters_.k2 * theta4 +
                  parameters_.k3 * theta6 + parameters_.k4 * theta8);
}

double EquidistantCamera::distortedThetaDerivative(double theta) const noexcept {
  const double theta2 = theta * theta;
  const double theta4 = theta2 * theta2;
  const double theta6 = theta4 * theta2;
  const double theta8 = theta4 * theta4;
  return 1.0 + 3.0 * parameters_.k1 * theta2 + 5.0 * parameters_.k2 * theta4 +
         7.0 * parameters_.k3 * theta6 + 9.0 * parameters_.k4 * theta8;
}

core::Result<CameraProjection, CameraModelError> EquidistantCamera::project(
    const Eigen::Vector3d& point_camera) const {
  if (!valid_) {
    return core::Result<CameraProjection, CameraModelError>::failure(
        {CameraModelErrorCode::InvalidParameters,
         "camera parameters are invalid or radial distortion is non-monotonic"});
  }
  if (!point_camera.allFinite()) {
    return core::Result<CameraProjection, CameraModelError>::failure(
        {CameraModelErrorCode::NonFiniteInput, "camera point is non-finite"});
  }
  if (std::abs(point_camera.z()) < kProjectionEpsilon) {
    return core::Result<CameraProjection, CameraModelError>::failure(
        {CameraModelErrorCode::PointOnProjectionPlane,
         "camera point lies on the projection plane"});
  }
  if (point_camera.z() < 0.0) {
    return core::Result<CameraProjection, CameraModelError>::failure(
        {CameraModelErrorCode::PointBehindCamera,
         "camera point is behind the front-facing image plane"});
  }

  const double inverse_z = 1.0 / point_camera.z();
  const Eigen::Vector2d undistorted = point_camera.head<2>() * inverse_z;
  const double radius = undistorted.norm();

  double scale = 1.0;
  Eigen::Matrix2d distortion_jacobian = Eigen::Matrix2d::Identity();
  if (radius > kRadialEpsilon) {
    const double theta = std::atan(radius);
    const double theta_distorted = distortedTheta(theta);
    scale = theta_distorted / radius;

    const double theta_radius_derivative = 1.0 / (1.0 + radius * radius);
    const double scale_radius_derivative =
        (distortedThetaDerivative(theta) * theta_radius_derivative * radius - theta_distorted) /
        (radius * radius);
    distortion_jacobian =
        scale * Eigen::Matrix2d::Identity() +
        (scale_radius_derivative / radius) * (undistorted * undistorted.transpose());
  }

  const Eigen::Vector2d distorted = scale * undistorted;
  CameraProjection projection;
  projection.pixel = Eigen::Vector2d{parameters_.fx * distorted.x() + parameters_.cx,
                                     parameters_.fy * distorted.y() + parameters_.cy};

  Eigen::Matrix<double, 2, 3> pinhole_jacobian;
  pinhole_jacobian << inverse_z, 0.0, -point_camera.x() * inverse_z * inverse_z, 0.0, inverse_z,
      -point_camera.y() * inverse_z * inverse_z;
  const Eigen::DiagonalMatrix<double, 2> focal(parameters_.fx, parameters_.fy);
  projection.point_jacobian = focal * distortion_jacobian * pinhole_jacobian;
  projection.domain = isInsideImage(projection.pixel) ? ProjectionDomain::InsideImage
                                                      : ProjectionDomain::OutsideImage;
  return core::Result<CameraProjection, CameraModelError>::success(projection);
}

core::Result<CameraRay, CameraModelError> EquidistantCamera::unproject(
    const Eigen::Vector2d& pixel) const {
  if (!valid_) {
    return core::Result<CameraRay, CameraModelError>::failure(
        {CameraModelErrorCode::InvalidParameters,
         "camera parameters are invalid or radial distortion is non-monotonic"});
  }
  if (!pixel.allFinite()) {
    return core::Result<CameraRay, CameraModelError>::failure(
        {CameraModelErrorCode::NonFiniteInput, "image point is non-finite"});
  }

  const Eigen::Vector2d distorted{(pixel.x() - parameters_.cx) / parameters_.fx,
                                  (pixel.y() - parameters_.cy) / parameters_.fy};
  const double distorted_radius = distorted.norm();

  CameraRay ray;
  const Eigen::DiagonalMatrix<double, 2> inverse_focal(1.0 / parameters_.fx, 1.0 / parameters_.fy);
  if (distorted_radius <= kRadialEpsilon) {
    // Preserve the derivative at the optical axis. Returning exactly UnitZ
    // throughout an epsilon disk would introduce a small but real Jacobian
    // discontinuity that is visible to an optimizer and to finite differences.
    Eigen::Vector3d projective{distorted.x(), distorted.y(), 1.0};
    const double inverse_norm = 1.0 / projective.norm();
    ray.unit_ray = projective * inverse_norm;
    Eigen::Matrix<double, 3, 2> normalized_jacobian;
    normalized_jacobian.topRows<2>() =
        inverse_norm * Eigen::Matrix2d::Identity() -
        std::pow(inverse_norm, 3) * (distorted * distorted.transpose());
    normalized_jacobian.bottomRows<1>() = -std::pow(inverse_norm, 3) * distorted.transpose();
    ray.pixel_jacobian = normalized_jacobian * inverse_focal;
    return core::Result<CameraRay, CameraModelError>::success(ray);
  }

  if (distorted_radius > distortedTheta(kHalfPiMargin)) {
    return core::Result<CameraRay, CameraModelError>::failure(
        {CameraModelErrorCode::DistortionNotInvertible,
         "image point maps beyond the front-facing distortion domain"});
  }

  // Safeguarded Newton solve of theta_d(theta)=r_d. OKVIS2-X solves the
  // equivalent normalized-plane inverse with Gauss--Newton; solving the radial
  // polynomial directly gives the same ray and an analytic inverse Jacobian.
  double lower = 0.0;
  double upper = kHalfPiMargin;
  double theta = std::min(distorted_radius, upper);
  bool converged = false;
  constexpr int kMaximumIterations = 20;
  for (int iteration = 0; iteration < kMaximumIterations; ++iteration) {
    const double residual = distortedTheta(theta) - distorted_radius;
    if (std::abs(residual) < 1.0e-12) {
      converged = true;
      break;
    }
    if (residual > 0.0) {
      upper = theta;
    } else {
      lower = theta;
    }
    const double derivative = distortedThetaDerivative(theta);
    if (!(derivative > kProjectionEpsilon)) {
      return core::Result<CameraRay, CameraModelError>::failure(
          {CameraModelErrorCode::DistortionNotInvertible,
           "distortion derivative became singular during inversion"});
    }
    const double newton = theta - residual / derivative;
    theta = (newton > lower && newton < upper) ? newton : 0.5 * (lower + upper);
  }
  if (!converged && std::abs(distortedTheta(theta) - distorted_radius) >= 1.0e-10) {
    return core::Result<CameraRay, CameraModelError>::failure(
        {CameraModelErrorCode::UnprojectionDidNotConverge,
         "radial equidistant inversion did not converge"});
  }

  const double sine = std::sin(theta);
  const double cosine = std::cos(theta);
  const double radial_scale = sine / distorted_radius;
  ray.unit_ray << radial_scale * distorted.x(), radial_scale * distorted.y(), cosine;

  const double theta_radius_derivative = 1.0 / distortedThetaDerivative(theta);
  const double radial_scale_derivative =
      (cosine * theta_radius_derivative * distorted_radius - sine) /
      (distorted_radius * distorted_radius);
  Eigen::Matrix<double, 3, 2> distorted_jacobian;
  distorted_jacobian.topRows<2>() =
      radial_scale * Eigen::Matrix2d::Identity() +
      (radial_scale_derivative / distorted_radius) * (distorted * distorted.transpose());
  distorted_jacobian.bottomRows<1>() =
      (-sine * theta_radius_derivative / distorted_radius) * distorted.transpose();
  ray.pixel_jacobian = distorted_jacobian * inverse_focal;

  return core::Result<CameraRay, CameraModelError>::success(ray);
}

bool EquidistantCamera::isInsideImage(const Eigen::Vector2d& pixel,
                                      double border_px) const noexcept {
  if (!valid_ || !pixel.allFinite() || border_px < 0.0) {
    return false;
  }
  return pixel.x() >= border_px && pixel.y() >= border_px &&
         pixel.x() < static_cast<double>(parameters_.width) - border_px &&
         pixel.y() < static_cast<double>(parameters_.height) - border_px;
}

}  // namespace meridian::local
