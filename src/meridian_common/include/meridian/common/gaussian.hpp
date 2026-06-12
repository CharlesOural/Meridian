#pragma once

#include <Eigen/Core>

namespace meridian {

// Square symmetric PSD matrix tagged with whether it is a covariance or its inverse.
// The ordering of the N tangent dimensions is fixed by the type that owns the block
// (Pose order [rho; phi], NavState order [p|R|v|bg|ba|g]). The one exception is
// KeyframePacket::constraint_cov, which is ordered rotation-first [rx,ry,rz,tx,ty,tz]
// to match the GTSAM Pose3 boundary.
template <int N>
struct GaussianBlock {
  enum class Form { Covariance, Information } form = Form::Covariance;
  Eigen::Matrix<double, N, N> M = Eigen::Matrix<double, N, N>::Zero();
};

using PoseCov6 = GaussianBlock<6>;   // tangent order [rho; phi] (translation-first)
using NavCov18 = GaussianBlock<18>;  // tangent order [p|R|v|bg|ba|g]

}  // namespace meridian
