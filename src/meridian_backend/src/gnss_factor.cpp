#include "gnss_factor.hpp"

#include <gtsam/base/Lie.h>

#include <boost/make_shared.hpp>

namespace meridian::backend {

GnssFactor::GnssFactor(gtsam::Key xi, gtsam::Key xj, gtsam::Key g, double beta,
                       const gtsam::Point3& lever, const gtsam::Point3& meas_enu,
                       const gtsam::SharedNoiseModel& noise)
    : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(noise, xi, xj, g),
      beta_(beta),
      lever_(lever),
      meas_enu_(meas_enu) {}

gtsam::Vector GnssFactor::evaluateError(const gtsam::Pose3& Xi, const gtsam::Pose3& Xj,
                                        const gtsam::Pose3& G, boost::optional<gtsam::Matrix&> H1,
                                        boost::optional<gtsam::Matrix&> H2,
                                        boost::optional<gtsam::Matrix&> H3) const {
  const bool need_pose = H1 || H2;

  // Body pose at the fix instant: SE(3) geodesic interpolation of the bracketing keyframes.
  gtsam::Matrix66 Hxi, Hxj;
  const gtsam::Pose3 Xb =
      gtsam::interpolate(Xi, Xj, beta_, need_pose ? &Hxi : nullptr, need_pose ? &Hxj : nullptr);

  // Antenna in the map frame, then mapped into the ENU tangent plane via the datum pose.
  gtsam::Matrix36 Ha;
  const gtsam::Point3 ant_map = Xb.transformFrom(lever_, need_pose ? &Ha : nullptr);

  gtsam::Matrix36 Hg;
  gtsam::Matrix33 Ham;
  const gtsam::Point3 ant_enu =
      G.transformTo(ant_map, H3 ? &Hg : nullptr, need_pose ? &Ham : nullptr);

  if (H1 || H2) {
    // d(ant_enu)/d(Xb) = Ham * Ha; push it through interpolation's pose Jacobians.
    const gtsam::Matrix36 d_enu_d_xb = Ham * Ha;
    if (H1) {
      *H1 = d_enu_d_xb * Hxi;
    }
    if (H2) {
      *H2 = d_enu_d_xb * Hxj;
    }
  }
  if (H3) {
    *H3 = Hg;
  }

  return ant_enu - meas_enu_;
}

gtsam::NonlinearFactor::shared_ptr GnssFactor::clone() const {
  return boost::make_shared<GnssFactor>(*this);
}

GnssFactorEndpoint::GnssFactorEndpoint(gtsam::Key x, gtsam::Key g, const gtsam::Point3& lever,
                                       const gtsam::Point3& meas_enu,
                                       const gtsam::SharedNoiseModel& noise)
    : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(noise, x, g),
      lever_(lever),
      meas_enu_(meas_enu) {}

gtsam::Vector GnssFactorEndpoint::evaluateError(const gtsam::Pose3& X, const gtsam::Pose3& G,
                                                boost::optional<gtsam::Matrix&> H1,
                                                boost::optional<gtsam::Matrix&> H2) const {
  gtsam::Matrix36 Ha;
  const gtsam::Point3 ant_map = X.transformFrom(lever_, H1 ? &Ha : nullptr);

  gtsam::Matrix36 Hg;
  gtsam::Matrix33 Ham;
  const gtsam::Point3 ant_enu = G.transformTo(ant_map, H2 ? &Hg : nullptr, H1 ? &Ham : nullptr);

  if (H1) {
    *H1 = Ham * Ha;
  }
  if (H2) {
    *H2 = Hg;
  }

  return ant_enu - meas_enu_;
}

gtsam::NonlinearFactor::shared_ptr GnssFactorEndpoint::clone() const {
  return boost::make_shared<GnssFactorEndpoint>(*this);
}

}  // namespace meridian::backend
