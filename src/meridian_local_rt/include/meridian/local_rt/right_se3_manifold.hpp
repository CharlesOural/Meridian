#pragma once

#include <ceres/manifold.h>

namespace meridian::local_rt {

// Ambient pose: [px, py, pz, qx, qy, qz, qw].
// Right tangent: [rho_x, rho_y, rho_z, theta_x, theta_y, theta_z], with
// Plus(T, delta) = T * Exp(delta).
class RightSe3Manifold final : public ceres::Manifold {
public:
  [[nodiscard]] int AmbientSize() const override { return 7; }
  [[nodiscard]] int TangentSize() const override { return 6; }

  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;
  bool PlusJacobian(const double* x, double* jacobian) const override;
  bool Minus(const double* y, const double* x, double* y_minus_x) const override;
  bool MinusJacobian(const double* x, double* jacobian) const override;
};

}  // namespace meridian::local_rt
