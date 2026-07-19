#pragma once

// Private GTSAM seam. This header is intentionally not installed.

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <stdexcept>

#include "meridian/local/visual_factor.hpp"

namespace meridian::local::gtsam_api {

// The public visual API uses eta=log(inverse range).  The nonlinear solver
// instead owns an unconstrained scalar whose image is strictly inside the
// configured eta interval.  Keeping this seam private prevents solver
// coordinates from leaking into factors, diagnostics, or convergence APIs.
struct BoundedEtaValue {
  double eta{};
  double derivative_wrt_latent{};
};

[[nodiscard]] BoundedEtaValue decodeBoundedEta(double latent, double minimum_range_m,
                                                double maximum_range_m) noexcept;
[[nodiscard]] double encodeBoundedEta(double eta, double minimum_range_m,
                                      double maximum_range_m);

class VisualFactorEvaluationException final : public std::runtime_error {
public:
  explicit VisualFactorEvaluationException(VisualReprojectionError error);

  [[nodiscard]] VisualReprojectionErrorCode code() const noexcept { return code_; }

private:
  VisualReprojectionErrorCode code_;
};

class AnchoredInverseRangeFactor final
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, double> {
public:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, double>;

  AnchoredInverseRangeFactor(gtsam::Key anchor_pose_key, gtsam::Key observer_pose_key,
                             gtsam::Key eta_key, VisualReprojectionFactorSpec spec);

  [[nodiscard]] gtsam::Vector evaluateError(
      const gtsam::Pose3& T_odom_imu_anchor, const gtsam::Pose3& T_odom_imu_observer,
      const double& bounded_eta_latent,
      boost::optional<gtsam::Matrix&> anchor_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> observer_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> eta_jacobian = boost::none) const override;

  [[nodiscard]] const VisualReprojectionFactorSpec& spec() const noexcept { return spec_; }

private:
  VisualReprojectionFactorSpec spec_;
};

}  // namespace meridian::local::gtsam_api
