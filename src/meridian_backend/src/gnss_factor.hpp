#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <boost/optional.hpp>

namespace meridian::backend {

// 3-DoF position residual tying a GNSS antenna fix to the trajectory. The antenna sits at
// a fixed lever arm in the body; the body pose at the fix instant is the SE(3) interpolation
// of the two bracketing keyframes Xi, Xj at fraction beta. The datum pose G maps the world
// (map) frame into the local ENU tangent plane where the fix was measured:
//   ant_map = interpolate(Xi, Xj, beta) * lever
//   r       = G^{-1} * ant_map - meas_enu
// Estimating G jointly absorbs the unknown map->ENU yaw and origin offset.
class GnssFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {
public:
  GnssFactor(gtsam::Key xi, gtsam::Key xj, gtsam::Key g, double beta, const gtsam::Point3& lever,
             const gtsam::Point3& meas_enu, const gtsam::SharedNoiseModel& noise);

  gtsam::Vector evaluateError(const gtsam::Pose3& Xi, const gtsam::Pose3& Xj, const gtsam::Pose3& G,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none,
                              boost::optional<gtsam::Matrix&> H3 = boost::none) const override;

  gtsam::NonlinearFactor::shared_ptr clone() const override;

private:
  double beta_ = 0.0;
  gtsam::Point3 lever_{0.0, 0.0, 0.0};
  gtsam::Point3 meas_enu_{0.0, 0.0, 0.0};
};

// Degenerate GnssFactor for beta in {0, 1}: the fix coincides with a single keyframe X
// (the successor keyframe does not exist yet, or the fix landed exactly on a node), so the
// interpolation collapses and there is one body pose to differentiate against:
//   r = G^{-1} * (X * lever) - meas_enu
class GnssFactorEndpoint : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
public:
  GnssFactorEndpoint(gtsam::Key x, gtsam::Key g, const gtsam::Point3& lever,
                     const gtsam::Point3& meas_enu, const gtsam::SharedNoiseModel& noise);

  gtsam::Vector evaluateError(const gtsam::Pose3& X, const gtsam::Pose3& G,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none) const override;

  gtsam::NonlinearFactor::shared_ptr clone() const override;

private:
  gtsam::Point3 lever_{0.0, 0.0, 0.0};
  gtsam::Point3 meas_enu_{0.0, 0.0, 0.0};
};

}  // namespace meridian::backend
