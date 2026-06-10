#pragma once

#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <boost/shared_ptr.hpp>
#include <cstddef>
#include <string>

namespace meridian::backend {

// Soft 6-DoF gauge anchor on one pose variable. It contributes zero cost at every
// linearization point but adds sqrt(lambda)*I to the variable's Jacobian, so the linear
// system stays full rank without any prior. Because the damping re-centers on the current
// estimate at each relinearization, the anchored pose still drifts freely once absolute
// information (GNSS, a tight prior) enters elsewhere in the graph.
class GaugeDampingFactor : public gtsam::NonlinearFactor {
public:
  using shared_ptr = boost::shared_ptr<GaugeDampingFactor>;

  GaugeDampingFactor(gtsam::Key key, double lambda);

  // Keep the base error overloads (HybridValues) visible alongside the override.
  using gtsam::NonlinearFactor::error;
  double error(const gtsam::Values& values) const override;
  std::size_t dim() const override;
  boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override;
  gtsam::NonlinearFactor::shared_ptr clone() const override;

  void print(const std::string& s = "",
             const gtsam::KeyFormatter& key_formatter = gtsam::DefaultKeyFormatter) const override;
  bool equals(const gtsam::NonlinearFactor& other, double tol = 1e-9) const override;

  double lambda() const { return lambda_; }

private:
  double lambda_ = 0.0;
};

}  // namespace meridian::backend
