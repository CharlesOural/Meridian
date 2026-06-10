#include "robust_kernels.hpp"

#include <boost/math/distributions/chi_squared.hpp>

namespace meridian::backend {

double chi2inv(double alpha, int dof) {
  return boost::math::quantile(boost::math::chi_squared(static_cast<double>(dof)), alpha);
}

gtsam::SharedNoiseModel make_huber_noise(const Eigen::MatrixXd& cov_gtsam, double huber_k) {
  const auto gaussian = gtsam::noiseModel::Gaussian::Covariance(cov_gtsam);
  return gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(huber_k),
                                           gaussian);
}

gtsam::SharedNoiseModel make_gnss_noise(const Eigen::Matrix3d& cov_enu, double huber_k) {
  return make_huber_noise(cov_enu, huber_k);
}

}  // namespace meridian::backend
