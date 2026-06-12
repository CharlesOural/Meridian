#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "datum.hpp"

using meridian::Pose;
using meridian::Timestamp;
using meridian::backend::DatumInitializer;
using meridian::backend::DatumResult;

namespace {

// A fixed-seed Gaussian noise source so every fit is bit-reproducible.
struct Noise {
  std::mt19937 rng{12345};
  std::normal_distribution<double> n{0.0, 1.0};
  Eigen::Vector3d horiz(double sigma) {
    return Eigen::Vector3d(sigma * n(rng), sigma * n(rng), 0.0);
  }
};

// Build the map antenna position from an ENU position via the ground-truth similarity
// map_horiz = R_yaw(yaw_true) * enu_horiz + t_true (vertical passes through unrotated).
Eigen::Vector3d mapFromEnu(const Eigen::Vector3d& enu, double yaw_true,
                           const Eigen::Vector3d& t_true) {
  const Eigen::Matrix3d R(Eigen::AngleAxisd(yaw_true, Eigen::Vector3d::UnitZ()));
  return R * enu + t_true;
}

// Loose gates that pass for any non-degenerate track, so each test isolates one gate.
constexpr double kMinBaseline = 5.0;
constexpr double kMinExcitation = 3.0;
constexpr double kMinSpeed = 0.5;
constexpr int kMinMovingFixes = 5;
constexpr double kYawSigmaMax = 0.10;  // ~5.7 deg

}  // namespace

// The yaw Hessian is the planar moment of inertia of the de-meaned ENU track. A track
// that travels a long map baseline (gate (a) passes) yet stays spatially compact in ENU
// has a tiny moment of inertia, so yaw_sigma = 1/sqrt(H_yaw) is large and the Hessian
// gate rejects the lock. We accumulate baseline with many small back-and-forth steps so
// the arc length clears min_baseline while every ENU point stays within a few cm of the
// origin. The excitation gate is opened (min_excitation = 0) so the Hessian gate is the
// only one that can fire.
TEST(DatumInitializer, CompactTrackYawUnobservableNotLocked) {
  DatumInitializer datum;
  const double yaw_true = 0.3;
  const Eigen::Vector3d t_true(10.0, -4.0, 2.0);

  // 200 jitter steps of ~0.04 m each accumulate ~8 m of map arc length (> min_baseline),
  // yet all ENU points sit within ~0.02 m of the origin (negligible moment of inertia).
  for (int i = 0; i < 200; ++i) {
    const double s = 0.02 * static_cast<double>(i % 2 == 0 ? 1 : -1);
    const Eigen::Vector3d enu(s, 0.0, 0.0);
    datum.add(enu, mapFromEnu(enu, yaw_true, t_true), 2.0, static_cast<Timestamp>(i));
  }

  const DatumResult r = datum.try_lock(kMinBaseline, /*min_excitation=*/0.0, kMinSpeed,
                                       kMinMovingFixes, kYawSigmaMax);
  EXPECT_FALSE(r.locked);
}

// A stationary / jittering track: tiny horizontal excitation and (here) speeds below the
// moving threshold. The excitation/speed pre-gates fail before any fit runs.
TEST(DatumInitializer, StationaryJitterFailsExcitationAndSpeedPreGate) {
  DatumInitializer datum;
  Noise noise;
  const double yaw_true = -0.7;
  const Eigen::Vector3d t_true(3.0, 3.0, 0.0);

  for (int i = 0; i < 40; ++i) {
    const Eigen::Vector3d enu = noise.horiz(0.05);  // few-cm jitter
    const double speed = 0.05;                      // below kMinSpeed
    datum.add(enu, mapFromEnu(enu, yaw_true, t_true), speed, static_cast<Timestamp>(i));
  }

  const DatumResult r = datum.try_lock(/*min_baseline=*/0.0, kMinExcitation, kMinSpeed,
                                       kMinMovingFixes, kYawSigmaMax);
  EXPECT_FALSE(r.locked);
}

// An L-shaped track with strong two-axis excitation and a known ground-truth yaw: the fit
// locks, recovers the yaw to within a couple of degrees, and reports a small yaw sigma.
TEST(DatumInitializer, LShapedTrackLocksAndRecoversYaw) {
  DatumInitializer datum;
  Noise noise;
  const double yaw_true = 0.6;  // ~34 deg
  const Eigen::Vector3d t_true(-7.0, 12.0, 1.5);

  // Leg 1: along +E for 20 m. Leg 2: along +N for 20 m. 41 fixes, all moving.
  std::vector<Eigen::Vector3d> track;
  for (int i = 0; i <= 20; ++i) {
    track.emplace_back(static_cast<double>(i), 0.0, 0.0);
  }
  for (int i = 1; i <= 20; ++i) {
    track.emplace_back(20.0, static_cast<double>(i), 0.0);
  }

  Timestamp stamp = 0;
  for (const Eigen::Vector3d& enu : track) {
    const Eigen::Vector3d enu_noisy = enu + noise.horiz(0.05);
    const Eigen::Vector3d map = mapFromEnu(enu, yaw_true, t_true) + noise.horiz(0.05);
    datum.add(enu_noisy, map, 2.0, stamp++);
  }

  const DatumResult r =
      datum.try_lock(kMinBaseline, kMinExcitation, kMinSpeed, kMinMovingFixes, kYawSigmaMax);
  ASSERT_TRUE(r.locked);

  // Fitted yaw within a few degrees of truth.
  const double fitted_yaw = std::atan2(r.T_map_enu.R()(1, 0), r.T_map_enu.R()(0, 0));
  EXPECT_NEAR(fitted_yaw, yaw_true, 3.0 * M_PI / 180.0);

  // Yaw sigma is small (well-excited track) and consistent with the reported gate.
  EXPECT_LT(r.yaw_sigma_rad, kYawSigmaMax);
  EXPECT_GT(r.yaw_sigma_rad, 0.0);

  // Translation: T_map_enu should map the ENU mean onto the map mean, so applying it to a
  // representative ENU point lands near the ground-truth map point.
  const Eigen::Vector3d probe(5.0, 0.0, 0.0);
  const Eigen::Vector3d mapped = r.T_map_enu * probe;
  const Eigen::Vector3d expected = mapFromEnu(probe, yaw_true, t_true);
  EXPECT_LT((mapped - expected).norm(), 0.5);
}

// Below two buffered fixes there is nothing to fit; try_lock returns not-locked.
TEST(DatumInitializer, TooFewFixesNotLocked) {
  DatumInitializer datum;
  datum.add(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 2.0, 0);
  EXPECT_EQ(datum.size(), 1u);
  const DatumResult r =
      datum.try_lock(kMinBaseline, kMinExcitation, kMinSpeed, kMinMovingFixes, kYawSigmaMax);
  EXPECT_FALSE(r.locked);
}
