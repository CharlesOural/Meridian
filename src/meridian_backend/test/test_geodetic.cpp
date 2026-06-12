#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>

#include "geodetic.hpp"

using meridian::backend::GeodeticDatum;
using meridian::backend::lla_to_ecef;
using meridian::backend::lla_to_enu;

namespace {

constexpr double kWgs84A = 6378137.0;
constexpr double kWgs84F = 1.0 / 298.257223563;
const double kWgs84B = kWgs84A * (1.0 - kWgs84F);  // semi-minor axis [m]

}  // namespace

TEST(Geodetic, EcefOfPrimeMeridianEquatorIsSemiMajorAxis) {
  const Eigen::Vector3d ecef = lla_to_ecef(0.0, 0.0, 0.0);
  EXPECT_NEAR(ecef.x(), kWgs84A, 1e-6);
  EXPECT_NEAR(ecef.y(), 0.0, 1e-6);
  EXPECT_NEAR(ecef.z(), 0.0, 1e-6);
}

TEST(Geodetic, EcefOfNinetyEastEquatorPointsAlongY) {
  const Eigen::Vector3d ecef = lla_to_ecef(0.0, 90.0, 0.0);
  EXPECT_NEAR(ecef.x(), 0.0, 1e-6);
  EXPECT_NEAR(ecef.y(), kWgs84A, 1e-6);
  EXPECT_NEAR(ecef.z(), 0.0, 1e-6);
}

TEST(Geodetic, EcefOfNorthPoleIsSemiMinorAxisOnZ) {
  const Eigen::Vector3d ecef = lla_to_ecef(90.0, 0.0, 0.0);
  EXPECT_NEAR(ecef.x(), 0.0, 1e-6);
  EXPECT_NEAR(ecef.y(), 0.0, 1e-6);
  EXPECT_NEAR(ecef.z(), kWgs84B, 1e-6);
}

TEST(Geodetic, AltitudeRaisesRadiallyAtEquator) {
  const Eigen::Vector3d ecef = lla_to_ecef(0.0, 0.0, 1000.0);
  EXPECT_NEAR(ecef.x(), kWgs84A + 1000.0, 1e-6);
  EXPECT_NEAR(ecef.y(), 0.0, 1e-6);
  EXPECT_NEAR(ecef.z(), 0.0, 1e-6);
}

TEST(Geodetic, EnuOfOriginIsZero) {
  GeodeticDatum datum;
  datum.lat0_deg = 51.5074;
  datum.lon0_deg = -0.1278;
  datum.alt0_m = 35.0;
  datum.set = true;

  const Eigen::Vector3d enu = lla_to_enu(datum.lat0_deg, datum.lon0_deg, datum.alt0_m, datum);
  EXPECT_NEAR(enu.norm(), 0.0, 1e-6);
}

TEST(Geodetic, EnuNorthStepLandsOnNorthAxis) {
  GeodeticDatum datum;
  datum.lat0_deg = 47.0;
  datum.lon0_deg = 8.0;
  datum.alt0_m = 400.0;
  datum.set = true;

  // 0.001 deg of latitude is ~111 m of northing, with east and up negligible.
  const Eigen::Vector3d enu =
      lla_to_enu(datum.lat0_deg + 0.001, datum.lon0_deg, datum.alt0_m, datum);
  EXPECT_NEAR(enu.x(), 0.0, 1e-3);   // east
  EXPECT_NEAR(enu.y(), 111.0, 1.0);  // north
  EXPECT_NEAR(enu.z(), 0.0, 0.5);    // up
}

TEST(Geodetic, EnuEastStepLandsOnEastAxis) {
  GeodeticDatum datum;
  datum.lat0_deg = 47.0;
  datum.lon0_deg = 8.0;
  datum.alt0_m = 400.0;
  datum.set = true;

  // At 47 deg latitude one deg of longitude is ~cos(lat) * 111 km of easting.
  const double expected_east = std::cos(47.0 * M_PI / 180.0) * 111319.0 * 0.001;
  const Eigen::Vector3d enu =
      lla_to_enu(datum.lat0_deg, datum.lon0_deg + 0.001, datum.alt0_m, datum);
  EXPECT_NEAR(enu.x(), expected_east, 1.0);  // east
  EXPECT_NEAR(enu.y(), 0.0, 1.0);            // north
  EXPECT_NEAR(enu.z(), 0.0, 0.5);            // up
}

TEST(Geodetic, EnuUpStepMatchesAltitudeDelta) {
  GeodeticDatum datum;
  datum.lat0_deg = 0.0;
  datum.lon0_deg = 0.0;
  datum.alt0_m = 0.0;
  datum.set = true;

  const Eigen::Vector3d enu = lla_to_enu(0.0, 0.0, 100.0, datum);
  EXPECT_NEAR(enu.x(), 0.0, 1e-6);
  EXPECT_NEAR(enu.y(), 0.0, 1e-6);
  EXPECT_NEAR(enu.z(), 100.0, 1e-6);
}
