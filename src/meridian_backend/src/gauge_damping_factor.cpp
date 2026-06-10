#include "gauge_damping_factor.hpp"

#include <gtsam/linear/JacobianFactor.h>

#include <boost/make_shared.hpp>
#include <cmath>
#include <iostream>

namespace meridian::backend {

GaugeDampingFactor::GaugeDampingFactor(gtsam::Key key, double lambda)
    : gtsam::NonlinearFactor(gtsam::KeyVector{key}), lambda_(lambda) {}

double GaugeDampingFactor::error(const gtsam::Values& /*values*/) const {
  return 0.0;
}

std::size_t GaugeDampingFactor::dim() const {
  return 6;
}

boost::shared_ptr<gtsam::GaussianFactor> GaugeDampingFactor::linearize(
    const gtsam::Values& /*values*/) const {
  // J = sqrt(lambda)*I, b = 0 with a unit noise model: a pure quadratic lambda*|dx|^2
  // penalty on the local increment, independent of the current value.
  const gtsam::Matrix6 jacobian = std::sqrt(lambda_) * gtsam::Matrix6::Identity();
  return boost::make_shared<gtsam::JacobianFactor>(keys_.front(), jacobian, gtsam::Vector6::Zero());
}

gtsam::NonlinearFactor::shared_ptr GaugeDampingFactor::clone() const {
  return boost::make_shared<GaugeDampingFactor>(*this);
}

void GaugeDampingFactor::print(const std::string& s,
                               const gtsam::KeyFormatter& key_formatter) const {
  std::cout << s << "GaugeDampingFactor on " << key_formatter(keys_.front())
            << ", lambda = " << lambda_ << std::endl;
}

bool GaugeDampingFactor::equals(const gtsam::NonlinearFactor& other, double tol) const {
  const auto* rhs = dynamic_cast<const GaugeDampingFactor*>(&other);
  return rhs != nullptr && gtsam::NonlinearFactor::equals(other, tol) &&
         std::abs(lambda_ - rhs->lambda_) <= tol;
}

}  // namespace meridian::backend
