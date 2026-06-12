#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <boost/optional.hpp>

namespace meridian::backend {

// Online-extrinsic variants of the GNSS factors: the antenna lever arm is the translation of
// an extrinsic variable E = T_body_gnss rather than a constant, so a mission with enough
// rotational + translational excitation can refine the lever jointly with the trajectory.
// Geometry is identical to GnssFactor, with lever := E.translation():
//   ant_map = interpolate(Xi, Xj, beta) * E.translation()
//   r       = G^{-1} * ant_map - meas_enu
class GnssFactorRefined
    : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {
public:
  GnssFactorRefined(gtsam::Key xi, gtsam::Key xj, gtsam::Key g, gtsam::Key e, double beta,
                    const gtsam::Point3& meas_enu, const gtsam::SharedNoiseModel& noise);

  gtsam::Vector evaluateError(const gtsam::Pose3& Xi, const gtsam::Pose3& Xj, const gtsam::Pose3& G,
                              const gtsam::Pose3& E,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none,
                              boost::optional<gtsam::Matrix&> H3 = boost::none,
                              boost::optional<gtsam::Matrix&> H4 = boost::none) const override;

  gtsam::NonlinearFactor::shared_ptr clone() const override;

private:
  double beta_ = 0.0;
  gtsam::Point3 meas_enu_{0.0, 0.0, 0.0};
};

// Endpoint (beta in {0,1}) refined variant: one body pose X, the datum G, and the extrinsic E.
class GnssFactorRefinedEndpoint
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {
public:
  GnssFactorRefinedEndpoint(gtsam::Key x, gtsam::Key g, gtsam::Key e, const gtsam::Point3& meas_enu,
                            const gtsam::SharedNoiseModel& noise);

  gtsam::Vector evaluateError(const gtsam::Pose3& X, const gtsam::Pose3& G, const gtsam::Pose3& E,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none,
                              boost::optional<gtsam::Matrix&> H3 = boost::none) const override;

  gtsam::NonlinearFactor::shared_ptr clone() const override;

private:
  gtsam::Point3 meas_enu_{0.0, 0.0, 0.0};
};

}  // namespace meridian::backend
