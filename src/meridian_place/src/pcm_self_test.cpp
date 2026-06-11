#include "pcm_self_test.hpp"

#include <cmath>
#include <limits>

namespace meridian {
namespace {

// chi-square (k=6) CDF.
double chi2_cdf6(double x) {
  if (x <= 0.0) return 0.0;
  const double h = 0.5 * x;
  return 1.0 - std::exp(-h) * (1.0 + h + 0.5 * h * h);
}

}  // namespace

double chi2InvDof6(double confidence) {
  if (confidence <= 0.0) return 0.0;
  if (confidence >= 1.0) return std::numeric_limits<double>::infinity();
  double lo = 0.0, hi = 200.0;
  for (int i = 0; i < 200; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (chi2_cdf6(mid) < confidence)
      lo = mid;
    else
      hi = mid;
  }
  return 0.5 * (lo + hi);
}

bool loopAgreesWithOdometry(const Pose& T_odom_from_to, const Pose& T_loop_from_to,
                            const Eigen::Matrix<double, 6, 6>& cov_combined,
                            double chi2_threshold, double* chi2_out) {
  // r = Log(T_odom^{-1} * T_loop), translation-first [rho; phi] to match cov_combined.
  const Eigen::Matrix<double, 6, 1> r = T_loop_from_to.boxminus(T_odom_from_to);
  const Eigen::Matrix<double, 6, 6> info =
      cov_combined.ldlt().solve(Eigen::Matrix<double, 6, 6>::Identity());
  const double chi2 = r.transpose() * info * r;
  if (chi2_out) *chi2_out = chi2;
  return chi2 <= chi2_threshold;
}

}  // namespace meridian
