#include "gnss_factor_refined.hpp"

#include <gtsam/base/Lie.h>

#include <boost/make_shared.hpp>

namespace meridian::backend {

GnssFactorRefined::GnssFactorRefined(gtsam::Key xi, gtsam::Key xj, gtsam::Key g, gtsam::Key e,
                                     double beta, const gtsam::Point3& meas_enu,
                                     const gtsam::SharedNoiseModel& noise)
    : gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(noise, xi,
                                                                                       xj, g, e),
      beta_(beta),
      meas_enu_(meas_enu) {}

gtsam::Vector GnssFactorRefined::evaluateError(const gtsam::Pose3& Xi, const gtsam::Pose3& Xj,
                                               const gtsam::Pose3& G, const gtsam::Pose3& E,
                                               boost::optional<gtsam::Matrix&> H1,
                                               boost::optional<gtsam::Matrix&> H2,
                                               boost::optional<gtsam::Matrix&> H3,
                                               boost::optional<gtsam::Matrix&> H4) const {
  const bool need_pose = H1 || H2;
  const bool need_ham = H1 || H2 || H4;

  // The lever arm is the extrinsic's translation; its Jacobian feeds the E gradient.
  gtsam::Matrix36 Ht;
  const gtsam::Point3 lever = E.translation(H4 ? &Ht : nullptr);

  gtsam::Matrix66 Hxi, Hxj;
  const gtsam::Pose3 Xb =
      gtsam::interpolate(Xi, Xj, beta_, need_pose ? &Hxi : nullptr, need_pose ? &Hxj : nullptr);

  // transformFrom gives both the pose Jacobian (for Xb) and the point Jacobian (for the lever).
  gtsam::Matrix36 Ha;
  gtsam::Matrix33 Hlever;
  const gtsam::Point3 ant_map =
      Xb.transformFrom(lever, need_pose ? &Ha : nullptr, H4 ? &Hlever : nullptr);

  gtsam::Matrix36 Hg;
  gtsam::Matrix33 Ham;
  const gtsam::Point3 ant_enu =
      G.transformTo(ant_map, H3 ? &Hg : nullptr, need_ham ? &Ham : nullptr);

  if (need_pose) {
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
  if (H4) {
    *H4 = Ham * Hlever * Ht;
  }

  return ant_enu - meas_enu_;
}

gtsam::NonlinearFactor::shared_ptr GnssFactorRefined::clone() const {
  return boost::make_shared<GnssFactorRefined>(*this);
}

GnssFactorRefinedEndpoint::GnssFactorRefinedEndpoint(gtsam::Key x, gtsam::Key g, gtsam::Key e,
                                                     const gtsam::Point3& meas_enu,
                                                     const gtsam::SharedNoiseModel& noise)
    : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(noise, x, g, e),
      meas_enu_(meas_enu) {}

gtsam::Vector GnssFactorRefinedEndpoint::evaluateError(const gtsam::Pose3& X, const gtsam::Pose3& G,
                                                       const gtsam::Pose3& E,
                                                       boost::optional<gtsam::Matrix&> H1,
                                                       boost::optional<gtsam::Matrix&> H2,
                                                       boost::optional<gtsam::Matrix&> H3) const {
  const bool need_ham = H1 || H3;

  gtsam::Matrix36 Ht;
  const gtsam::Point3 lever = E.translation(H3 ? &Ht : nullptr);

  gtsam::Matrix36 Ha;
  gtsam::Matrix33 Hlever;
  const gtsam::Point3 ant_map = X.transformFrom(lever, H1 ? &Ha : nullptr, H3 ? &Hlever : nullptr);

  gtsam::Matrix36 Hg;
  gtsam::Matrix33 Ham;
  const gtsam::Point3 ant_enu =
      G.transformTo(ant_map, H2 ? &Hg : nullptr, need_ham ? &Ham : nullptr);

  if (H1) {
    *H1 = Ham * Ha;
  }
  if (H2) {
    *H2 = Hg;
  }
  if (H3) {
    *H3 = Ham * Hlever * Ht;
  }

  return ant_enu - meas_enu_;
}

gtsam::NonlinearFactor::shared_ptr GnssFactorRefinedEndpoint::clone() const {
  return boost::make_shared<GnssFactorRefinedEndpoint>(*this);
}

}  // namespace meridian::backend
