#pragma once

#include <gtsam/navigation/CombinedImuFactor.h>

#include "meridian/common/imu_preintegration.hpp"
#include "meridian/config/config.hpp"

namespace meridian::backend {

// Rebuilds a GTSAM combined-measurement object from the boundary summary so the back-end
// can drop a CombinedImuFactor on a window restart. The summary's 9-vector and Jacobians
// are ordered [dR | dv | dp] (rotation, velocity, position); GTSAM's tangent storage and
// 15x15 measurement covariance are ordered [theta | p | v] (rotation, position, velocity),
// so the velocity and position blocks are swapped on the way in. The cross-correlation
// between the preintegrated increment and the bias is not carried by the summary and is
// left zero, a conservative (uncertainty-not-decreased) approximation. The continuous-time
// noise densities in `imu` only feed predict's gravity/Coriolis terms; the factor's
// measurement weight comes from the covariance set directly on the returned object.
gtsam::PreintegratedCombinedMeasurements pim_from_summary(const ImuPreintegrationSummary& s,
                                                          const BackendImu& imu);

}  // namespace meridian::backend
