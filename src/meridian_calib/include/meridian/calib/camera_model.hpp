#pragma once

#include <Eigen/Core>

#include "meridian/calib/intrinsics.hpp"

namespace meridian {

// Pinhole projection with optional plumb-bob (radial-tangential) distortion. The
// coefficient layout is k1,k2,p1,p2,k3; all-zero coefficients reduce the math
// exactly to the pure pinhole equations (no distortion branch is taken). This
// projector covers only the rectify-to-pinhole path the front-end consumes; an
// equidistant/fisheye model has no usable mapping here and leaves the model
// invalid (valid()==false) rather than throwing.
//
// A default-constructed model is invalid (fx == 0); valid() gates the visual
// stage, which must stay disabled whenever the camera intrinsics are zero or
// otherwise unusable by this projector.
class CameraModel {
 public:
  CameraModel();
  explicit CameraModel(const IntrinsicsCamera& k);

  // True once fx,fy are both strictly positive. The visual front-end consults
  // this before building any photometric residual.
  bool valid() const { return valid_; }
  int width() const { return width_; }
  int height() const { return height_; }

  // Projects a camera-frame point to a distorted pixel. Returns false (and leaves
  // *uv untouched) when the point is at/behind the image plane (z <= 0), when the
  // resulting pixel falls outside [0,width)x[0,height), or when the model is
  // invalid. width/height == 0 disables the on-image check (projection still runs).
  bool project(const Eigen::Vector3d& p_cam, Eigen::Vector2d* uv) const;

  // 2x3 Jacobian d(uv)/d(p_cam) at p_cam, including the distortion terms. Valid
  // wherever project() returns true; meaningless for z <= 0 (returns zeros).
  Eigen::Matrix<double, 2, 3> projectJacobian(const Eigen::Vector3d& p_cam) const;

  // Distorted pixel -> unit-depth ray [x,y,1] in the camera frame. The radial-
  // tangential distortion has no closed-form inverse, so the undistorted
  // normalized coordinates are recovered by a fixed-point iteration (see .cpp).
  // For the pinhole (zero-coefficient) case this is exact in one step.
  Eigen::Vector3d unproject(const Eigen::Vector2d& uv) const;

 private:
  // Applies the plumb-bob distortion to normalized image coordinates (x,y) =
  // (X/Z, Y/Z), returning the distorted normalized coordinates.
  Eigen::Vector2d distortNormalized(const Eigen::Vector2d& xy) const;

  double fx_ = 0.0, fy_ = 0.0, cx_ = 0.0, cy_ = 0.0;
  double k1_ = 0.0, k2_ = 0.0, p1_ = 0.0, p2_ = 0.0, k3_ = 0.0;
  bool has_distortion_ = false;  // any coefficient non-zero
  int width_ = 0, height_ = 0;
  bool valid_ = false;
};

}  // namespace meridian
