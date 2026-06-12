#pragma once

#include <Eigen/Core>

namespace meridian::backend {

// Origin of the local ENU tangent plane. ENU coordinates are only meaningful once
// `set` is true; until then no fix can be projected.
struct GeodeticDatum {
  double lat0_deg = 0;
  double lon0_deg = 0;
  double alt0_m = 0;
  bool set = false;
};

// WGS84 geodetic (lat, lon in degrees; ellipsoidal height in metres) to ECEF [m].
Eigen::Vector3d lla_to_ecef(double lat_deg, double lon_deg, double alt_m);

// ENU position [m] of (lat, lon, alt) relative to the datum origin. Requires datum.set.
Eigen::Vector3d lla_to_enu(double lat_deg, double lon_deg, double alt_m,
                           const GeodeticDatum& datum);

}  // namespace meridian::backend
