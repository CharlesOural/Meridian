#include "meridian/calib/camera_model.hpp"

#include <cmath>

namespace meridian {

namespace {
// Fixed-point undistort: the plumb-bob map has no closed form inverse, so we
// recover undistorted normalized coordinates by the standard contraction
// x_{n+1} = (x_d - tangential(x_n)) / radial(x_n), seeded at x_d. It converges
// geometrically for the coefficient magnitudes of real lenses (the distortion is
// a small perturbation of the identity inside the image circle); these bounds
// stop the loop once the step is sub-micro-pixel or after a hard iteration cap.
constexpr int kUndistortMaxIters = 20;
// Squared-step tolerance: the iteration is linearly convergent, so a loose stop
// leaves residual error of the same order as the last step. 1e-16 squared (~1e-8
// linear) drives the recovered ray well below a hundredth of a pixel.
constexpr double kUndistortEps = 1e-16;
}  // namespace

CameraModel::CameraModel() = default;

CameraModel::CameraModel(const IntrinsicsCamera& k)
    : fx_(k.fx),
      fy_(k.fy),
      cx_(k.cx),
      cy_(k.cy),
      width_(k.width),
      height_(k.height) {
  // This projector implements only the pinhole + plumb-bob map; an equidistant
  // (fisheye) lens has no usable mapping here. Leave the model invalid so the
  // visual stage gates off (valid()==false) instead of crashing the pipeline.
  if (k.model == IntrinsicsCamera::Distortion::Equidistant) {
    valid_ = false;
    return;
  }
  // Distortion coefficients are consumed only for the RadTan model; the None
  // model leaves them at zero so the math degrades to pure pinhole.
  if (k.model == IntrinsicsCamera::Distortion::RadTan) {
    k1_ = k.coeffs[0];
    k2_ = k.coeffs[1];
    p1_ = k.coeffs[2];
    p2_ = k.coeffs[3];
    k3_ = k.coeffs[4];
  }
  has_distortion_ = (k1_ != 0.0) || (k2_ != 0.0) || (p1_ != 0.0) ||
                    (p2_ != 0.0) || (k3_ != 0.0);
  valid_ = (fx_ > 0.0) && (fy_ > 0.0);
}

Eigen::Vector2d CameraModel::distortNormalized(const Eigen::Vector2d& xy) const {
  const double x = xy.x();
  const double y = xy.y();
  const double r2 = x * x + y * y;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;
  const double radial = 1.0 + k1_ * r2 + k2_ * r4 + k3_ * r6;
  const double dx = 2.0 * p1_ * x * y + p2_ * (r2 + 2.0 * x * x);
  const double dy = p1_ * (r2 + 2.0 * y * y) + 2.0 * p2_ * x * y;
  return Eigen::Vector2d(x * radial + dx, y * radial + dy);
}

bool CameraModel::project(const Eigen::Vector3d& p_cam, Eigen::Vector2d* uv) const {
  if (!valid_ || p_cam.z() <= 0.0) {
    return false;
  }
  const double z_inv = 1.0 / p_cam.z();
  const Eigen::Vector2d xy(p_cam.x() * z_inv, p_cam.y() * z_inv);
  const Eigen::Vector2d xyd = has_distortion_ ? distortNormalized(xy) : xy;
  const Eigen::Vector2d px(fx_ * xyd.x() + cx_, fy_ * xyd.y() + cy_);

  // width/height == 0 means "unknown frame size"; skip the on-image gate then.
  if (width_ > 0 && height_ > 0) {
    if (px.x() < 0.0 || px.y() < 0.0 || px.x() >= static_cast<double>(width_) ||
        px.y() >= static_cast<double>(height_)) {
      return false;
    }
  }
  *uv = px;
  return true;
}

Eigen::Matrix<double, 2, 3> CameraModel::projectJacobian(
    const Eigen::Vector3d& p_cam) const {
  Eigen::Matrix<double, 2, 3> J = Eigen::Matrix<double, 2, 3>::Zero();
  if (!valid_ || p_cam.z() <= 0.0) {
    return J;
  }
  const double z_inv = 1.0 / p_cam.z();
  const double z_inv2 = z_inv * z_inv;
  const double x = p_cam.x() * z_inv;
  const double y = p_cam.y() * z_inv;

  // d(normalized)/d(p_cam): [ 1/Z, 0, -X/Z^2 ; 0, 1/Z, -Y/Z^2 ].
  Eigen::Matrix<double, 2, 3> dn_dp;
  dn_dp << z_inv, 0.0, -p_cam.x() * z_inv2, 0.0, z_inv, -p_cam.y() * z_inv2;

  // d(distorted normalized)/d(normalized): identity in the pinhole case, the full
  // plumb-bob Jacobian otherwise.
  Eigen::Matrix2d dd_dn = Eigen::Matrix2d::Identity();
  if (has_distortion_) {
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double radial = 1.0 + k1_ * r2 + k2_ * r4 + k3_ * r2 * r4;
    // d(radial)/d(r2) = k1 + 2 k2 r2 + 3 k3 r4, chained by d(r2)/dx = 2x, /dy = 2y.
    const double dradial_dr2 = k1_ + 2.0 * k2_ * r2 + 3.0 * k3_ * r4;
    const double dD_dx = dradial_dr2 * 2.0 * x;
    const double dD_dy = dradial_dr2 * 2.0 * y;
    dd_dn(0, 0) = radial + x * dD_dx + 2.0 * p1_ * y + 6.0 * p2_ * x;
    dd_dn(0, 1) = x * dD_dy + 2.0 * p1_ * x + 2.0 * p2_ * y;
    dd_dn(1, 0) = y * dD_dx + 2.0 * p1_ * x + 2.0 * p2_ * y;
    dd_dn(1, 1) = radial + y * dD_dy + 6.0 * p1_ * y + 2.0 * p2_ * x;
  }

  // d(uv)/d(distorted) = diag(fx, fy); compose right-to-left.
  Eigen::Matrix2d df_dd = Eigen::Matrix2d::Zero();
  df_dd(0, 0) = fx_;
  df_dd(1, 1) = fy_;
  J = df_dd * dd_dn * dn_dp;
  return J;
}

Eigen::Vector3d CameraModel::unproject(const Eigen::Vector2d& uv) const {
  if (!valid_) {
    return Eigen::Vector3d(0.0, 0.0, 1.0);
  }
  // Pixel -> distorted normalized coordinates (exact inverse of the affine K).
  const double xd = (uv.x() - cx_) / fx_;
  const double yd = (uv.y() - cy_) / fy_;
  if (!has_distortion_) {
    return Eigen::Vector3d(xd, yd, 1.0);
  }

  // Fixed-point inversion of the plumb-bob map: solve distort(x,y) = (xd,yd).
  double x = xd;
  double y = yd;
  for (int i = 0; i < kUndistortMaxIters; ++i) {
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double radial = 1.0 + k1_ * r2 + k2_ * r4 + k3_ * r6;
    const double dx = 2.0 * p1_ * x * y + p2_ * (r2 + 2.0 * x * x);
    const double dy = p1_ * (r2 + 2.0 * y * y) + 2.0 * p2_ * x * y;
    const double x_new = (xd - dx) / radial;
    const double y_new = (yd - dy) / radial;
    const double step = (x_new - x) * (x_new - x) + (y_new - y) * (y_new - y);
    x = x_new;
    y = y_new;
    if (step < kUndistortEps) {
      break;
    }
  }
  return Eigen::Vector3d(x, y, 1.0);
}

}  // namespace meridian
