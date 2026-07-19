#pragma once

#include <Eigen/Core>
#include <cstdint>

#include "meridian/core/api.hpp"

namespace meridian::local {

// Pinhole projection followed by the four-coefficient Kannala--Brandt
// equidistant distortion used by the Newer College AlphaSense calibration.
// The implementation is adapted from the BSD-3-Clause OKVIS2-X camera math;
// see THIRD_PARTY_VISUAL.md for provenance and license text.
struct EquidistantCameraParameters {
  std::uint32_t width{};
  std::uint32_t height{};
  double fx{};
  double fy{};
  double cx{};
  double cy{};
  double k1{};
  double k2{};
  double k3{};
  double k4{};
};

enum class CameraModelErrorCode {
  InvalidParameters,
  NonFiniteInput,
  PointOnProjectionPlane,
  PointBehindCamera,
  DistortionNotInvertible,
  UnprojectionDidNotConverge,
};

struct CameraModelError {
  CameraModelErrorCode code{};
  const char* detail{};
};

enum class ProjectionDomain {
  InsideImage,
  OutsideImage,
};

struct CameraProjection {
  Eigen::Vector2d pixel{Eigen::Vector2d::Zero()};
  Eigen::Matrix<double, 2, 3> point_jacobian{Eigen::Matrix<double, 2, 3>::Zero()};
  ProjectionDomain domain{ProjectionDomain::OutsideImage};
};

struct CameraRay {
  Eigen::Vector3d unit_ray{Eigen::Vector3d::UnitZ()};
  Eigen::Matrix<double, 3, 2> pixel_jacobian{Eigen::Matrix<double, 3, 2>::Zero()};
};

class EquidistantCamera {
public:
  explicit EquidistantCamera(EquidistantCameraParameters parameters);

  [[nodiscard]] bool valid() const noexcept { return valid_; }
  [[nodiscard]] const EquidistantCameraParameters& parameters() const noexcept {
    return parameters_;
  }

  [[nodiscard]] core::Result<CameraProjection, CameraModelError> project(
      const Eigen::Vector3d& point_camera) const;

  [[nodiscard]] core::Result<CameraRay, CameraModelError> unproject(
      const Eigen::Vector2d& pixel) const;

  [[nodiscard]] bool isInsideImage(const Eigen::Vector2d& pixel,
                                   double border_px = 0.0) const noexcept;

private:
  [[nodiscard]] double distortedTheta(double theta) const noexcept;
  [[nodiscard]] double distortedThetaDerivative(double theta) const noexcept;

  EquidistantCameraParameters parameters_;
  bool valid_{};
};

}  // namespace meridian::local
