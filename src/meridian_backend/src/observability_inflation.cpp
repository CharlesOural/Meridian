#include "observability_inflation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace meridian::backend {
namespace {

// Rotation-first axis k reads its score at translation-first index kScoreIndex[k]. The axis
// convention is the same rotation-first order as KeyframePacket::constraint_cov.
constexpr std::array<std::size_t, 6> kScoreIndex = {3, 4, 5, 0, 1, 2};

// Permutation taking translation-first [tx,ty,tz,rx,ry,rz] coordinates to rotation-first.
Eigen::Matrix<double, 6, 6> permutationMatrix() {
  Eigen::Matrix<double, 6, 6> p = Eigen::Matrix<double, 6, 6>::Zero();
  for (std::size_t k = 0; k < 6; ++k) {
    p(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(kScoreIndex[k])) = 1.0;
  }
  return p;
}

}  // namespace

InflationResult inflate_by_observability(const GaussianBlock<6>& cov,
                                         const ObservabilityReport& obs, double rho_max,
                                         double gamma, double degenerate_thresh,
                                         bool degenerate_lock) {
  InflationResult out;
  Eigen::Matrix<double, 6, 6> lsqrt = Eigen::Matrix<double, 6, 6>::Zero();
  for (std::size_t k = 0; k < 6; ++k) {
    const double s = std::clamp(obs.score[kScoreIndex[k]], 0.0, 1.0);
    double lambda;
    if (degenerate_lock && s < degenerate_thresh) {
      lambda = rho_max;
      out.any_locked = true;
    } else {
      lambda = 1.0 + (rho_max - 1.0) * std::pow(1.0 - s, gamma);
    }
    out.lambda[k] = lambda;
    const auto ki = static_cast<Eigen::Index>(k);
    lsqrt(ki, ki) = std::sqrt(lambda);
  }

  if (obs.eigvecs.has_value()) {
    // Inflate in the reported eigvec basis. Permuting the columns onto rotation-first axes
    // keeps column k of R6 paired with lambda[k], which was read from the matching score.
    const Eigen::Matrix<double, 6, 6> p = permutationMatrix();
    const Eigen::Matrix<double, 6, 6> r6 = p * (*obs.eigvecs) * p.transpose();
    out.cov = r6 * lsqrt * (r6.transpose() * cov.M * r6) * lsqrt * r6.transpose();
  } else {
    out.cov = lsqrt * cov.M * lsqrt;
  }
  // The congruence preserves symmetry up to floating point; symmetrize to remove the residue.
  out.cov = 0.5 * (out.cov + out.cov.transpose()).eval();
  return out;
}

}  // namespace meridian::backend
