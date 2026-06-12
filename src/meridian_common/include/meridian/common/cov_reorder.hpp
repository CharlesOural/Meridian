#pragma once

#include <Eigen/Core>

namespace meridian {

// Reorders a 6x6 pose marginal between the internal translation-first [rho; phi]
// convention and the rotation-first [rx,ry,rz,tx,ty,tz] boundary convention. The
// permutation swaps the two diagonal 3-blocks and transposes the off-diagonal
// coupling across the swap; it is self-inverse, so applying it twice returns the
// original block.
inline Eigen::Matrix<double, 6, 6> reorderTransRotToRotTrans(
    const Eigen::Matrix<double, 6, 6>& trans_first) {
  // Build the permutation [3,4,5,0,1,2] (input [rho(0:2); phi(3:5)] -> [phi; rho])
  // and apply it symmetrically: P * M * P^T.
  Eigen::Matrix<double, 6, 6> perm = Eigen::Matrix<double, 6, 6>::Zero();
  for (int i = 0; i < 3; ++i) {
    perm(i, i + 3) = 1.0;  // rotation rows take the old rotation block
    perm(i + 3, i) = 1.0;  // translation rows take the old translation block
  }
  return perm * trans_first * perm.transpose();
}

}  // namespace meridian
