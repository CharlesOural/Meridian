#pragma once

#include <array>

namespace meridian {

// Pinhole camera intrinsics plus the photometric (inverse-exposure) intrinsic
// the direct visual front-end refines per frame.
struct IntrinsicsCamera {
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;  // pinhole K_cam

  enum class Distortion { None, RadTan, Equidistant } model = Distortion::RadTan;
  // RadTan: k1,k2,p1,p2,k3 ; Equidistant/fisheye: k1..k4 (last entry unused).
  std::array<double, 5> coeffs = {0, 0, 0, 0, 0};
  int width = 0, height = 0;

  // Inverse exposure scale; brightness constancy is broken by auto-exposure, so
  // this is carried as an intrinsic with a prior and refined inside the front-end.
  double inv_expo_prior = 1.0;  // 1/exposure scale prior (1.0 = no scale)
  double inv_expo_std = 0.0;    // prior 1σ; 0 ⇒ fixed
  bool refine_photometric_online = true;
};

}  // namespace meridian
