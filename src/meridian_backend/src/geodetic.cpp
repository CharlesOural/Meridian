#include "geodetic.hpp"

#include <cmath>

namespace meridian::backend {

namespace {

constexpr double kWgs84A = 6378137.0;               // semi-major axis [m]
constexpr double kWgs84F = 1.0 / 298.257223563;     // flattening
const double kWgs84E2 = kWgs84F * (2.0 - kWgs84F);  // first eccentricity squared

constexpr double kDegToRad = M_PI / 180.0;

}  // namespace

Eigen::Vector3d lla_to_ecef(double lat_deg, double lon_deg, double alt_m) {
  const double lat = lat_deg * kDegToRad;
  const double lon = lon_deg * kDegToRad;
  const double sin_lat = std::sin(lat);
  const double cos_lat = std::cos(lat);
  const double sin_lon = std::sin(lon);
  const double cos_lon = std::cos(lon);

  // Radius of curvature in the prime vertical.
  const double n = kWgs84A / std::sqrt(1.0 - kWgs84E2 * sin_lat * sin_lat);

  return Eigen::Vector3d((n + alt_m) * cos_lat * cos_lon, (n + alt_m) * cos_lat * sin_lon,
                         (n * (1.0 - kWgs84E2) + alt_m) * sin_lat);
}

Eigen::Vector3d lla_to_enu(double lat_deg, double lon_deg, double alt_m,
                           const GeodeticDatum& datum) {
  const Eigen::Vector3d ecef = lla_to_ecef(lat_deg, lon_deg, alt_m);
  const Eigen::Vector3d ecef0 = lla_to_ecef(datum.lat0_deg, datum.lon0_deg, datum.alt0_m);
  const Eigen::Vector3d d = ecef - ecef0;

  const double lat0 = datum.lat0_deg * kDegToRad;
  const double lon0 = datum.lon0_deg * kDegToRad;
  const double sin_lat = std::sin(lat0);
  const double cos_lat = std::cos(lat0);
  const double sin_lon = std::sin(lon0);
  const double cos_lon = std::cos(lon0);

  // Rotation R_enu_ecef evaluated at the datum origin; rows are the E, N, U axes.
  Eigen::Matrix3d r_enu_ecef;
  r_enu_ecef << -sin_lon, cos_lon, 0.0, -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat,
      cos_lat * cos_lon, cos_lat * sin_lon, sin_lat;

  return r_enu_ecef * d;
}

}  // namespace meridian::backend
