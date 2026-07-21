#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "meridian/core/geometry.hpp"

namespace meridian::core {
namespace {

TEST(Quaterniond, DefaultsToIdentityAndAcceptsAUnitRotation) {
  const Quaterniond identity;
  EXPECT_DOUBLE_EQ(identity.w(), 1.0);
  EXPECT_DOUBLE_EQ(identity.x(), 0.0);
  EXPECT_DOUBLE_EQ(identity.squaredNorm(), 1.0);

  const double half_sqrt_two = std::sqrt(0.5);
  const Quaterniond rotation(half_sqrt_two, 0.0, 0.0, half_sqrt_two);
  EXPECT_TRUE(rotation.isFinite());
  EXPECT_NEAR(rotation.squaredNorm(), 1.0, 1.0e-15);
}

TEST(Quaterniond, RejectsNonFiniteAndNonUnitCoefficients) {
  EXPECT_THROW(
      static_cast<void>(Quaterniond(std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0)),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Quaterniond(2.0, 0.0, 0.0, 0.0)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Quaterniond(0.0, 0.0, 0.0, 0.0)), std::invalid_argument);
}

TEST(Pose3d, PreservesFrameTransformAndRejectsNonFiniteTranslation) {
  const Pose3d pose(Vec3d{1.0, 2.0, 3.0}, Quaterniond());
  EXPECT_EQ(pose.translation(), (Vec3d{1.0, 2.0, 3.0}));
  EXPECT_EQ(pose.rotation(), Quaterniond());

  EXPECT_THROW(static_cast<void>(Pose3d(Vec3d{1.0, std::numeric_limits<double>::quiet_NaN(), 3.0},
                                        Quaterniond())),
               std::invalid_argument);
}

}  // namespace
}  // namespace meridian::core
