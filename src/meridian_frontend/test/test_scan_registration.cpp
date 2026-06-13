#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <random>
#include <sophus/se3.hpp>
#include <vector>

#include "lio/relative_cov.hpp"
#include "lio/scan_registration.hpp"
#include "meridian/common/cov_reorder.hpp"
#include "meridian/config/config.hpp"

namespace {

using meridian::FrontendLio;
using meridian::Pose;
using meridian::lio::composeRelativeCov;
using meridian::lio::MotionPrior;
using meridian::lio::NearestLookup;
using meridian::lio::RegistrationResult;
using meridian::lio::ScanRegistration;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat36 = Eigen::Matrix<double, 3, 6>;

// Exhaustive nearest neighbor over a plain point vector with the squared-distance
// gate, scanning in stored order with a strict '<' so ties resolve to the first point.
NearestLookup bruteForceNearest(const std::vector<Eigen::Vector3d>& pts, double gate) {
  return [&pts, gate](const Eigen::Vector3d& query, Eigen::Vector3d* neighbor) {
    double best = gate * gate;
    bool found = false;
    for (const Eigen::Vector3d& p : pts) {
      const double d2 = (p - query).squaredNorm();
      if (d2 < best) {
        best = d2;
        *neighbor = p;
        found = true;
      }
    }
    return found;
  };
}

Pose toPose(const Sophus::SE3d& T) {
  return Pose{T.unit_quaternion(), T.translation()};
}

Mat36 dataJacobian(const Eigen::Vector3d& p_world) {
  Mat36 J;
  J.leftCols<3>() = Eigen::Matrix3d::Identity();
  J.rightCols<3>() = -Sophus::SO3d::hat(p_world);
  return J;
}

// Three mutually orthogonal planes of a box corner, lattice-sampled; a full-rank
// scene for point-to-point alignment.
std::vector<Eigen::Vector3d> boxCornerCloud(double extent, double step) {
  std::vector<Eigen::Vector3d> pts;
  for (double a = 0.0; a <= extent + 1e-9; a += step) {
    for (double b = 0.0; b <= extent + 1e-9; b += step) {
      pts.emplace_back(a, b, 0.0);  // floor z = 0
      pts.emplace_back(0.0, a, b);  // wall x = 0
      pts.emplace_back(a, 0.0, b);  // wall y = 0
    }
  }
  return pts;
}

std::vector<Eigen::Vector3d> transformAll(const Sophus::SE3d& T,
                                          const std::vector<Eigen::Vector3d>& pts) {
  std::vector<Eigen::Vector3d> out;
  out.reserve(pts.size());
  for (const Eigen::Vector3d& p : pts) {
    out.push_back(T * p);
  }
  return out;
}

double rollOf(const RegistrationResult& res) {
  return Sophus::SO3d(res.pose.q).log().x();
}

// The data residual r(T) = T*p - q is linearized with the left/world perturbation
// T <- exp(dx)*T; the analytic [I | -hat(T*p)] must match central differences taken
// through that exact chart, column by column.
TEST(ScanRegistration, DataJacobianMatchesCentralDifference) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  const double h = 1e-6;
  for (int trial = 0; trial < 20; ++trial) {
    Vec6 xi;
    for (int i = 0; i < 6; ++i) {
      xi[i] = u(rng);
    }
    const Sophus::SE3d T = Sophus::SE3d::exp(xi);
    const Eigen::Vector3d p{5.0 * u(rng), 5.0 * u(rng), 5.0 * u(rng)};
    const Eigen::Vector3d q{5.0 * u(rng), 5.0 * u(rng), 5.0 * u(rng)};
    const Mat36 J = dataJacobian(T * p);
    for (int k = 0; k < 6; ++k) {
      Vec6 e = Vec6::Zero();
      e[k] = h;
      const Eigen::Vector3d r_plus = (Sophus::SE3d::exp(e) * T) * p - q;
      const Eigen::Vector3d r_minus = (Sophus::SE3d::exp(-e) * T) * p - q;
      const Eigen::Vector3d numeric = (r_plus - r_minus) / (2.0 * h);
      EXPECT_LT((numeric - J.col(k)).norm(), 1e-6) << "trial " << trial << " col " << k;
    }
  }
}

// A perturbed initial guess on a noiseless box-corner scene must be pulled back to the
// exact ground-truth transform, and a repeated solve must be bit-identical.
TEST(ScanRegistration, ConvergesOnBoxCorner) {
  const FrontendLio cfg;
  const std::vector<Eigen::Vector3d> map_pts = boxCornerCloud(4.0, 0.2);
  const NearestLookup nearest = bruteForceNearest(map_pts, cfg.max_corr_dist_m);

  Vec6 xi_true;
  xi_true << 0.4, -0.3, 0.25, 0.04, -0.03, 0.05;
  const Sophus::SE3d T_true = Sophus::SE3d::exp(xi_true);
  const std::vector<Eigen::Vector3d> keypoints = transformAll(T_true.inverse(), map_pts);

  Vec6 dxi;
  dxi << 0.08, -0.06, 0.05, 0.015, 0.02, -0.018;
  const Pose guess = toPose(Sophus::SE3d::exp(dxi) * T_true);

  const ScanRegistration reg{cfg};
  const RegistrationResult res = reg.registerScan(keypoints, nearest, guess, MotionPrior{});

  EXPECT_TRUE(res.converged);
  EXPECT_EQ(res.n_corr, static_cast<int>(keypoints.size()));
  EXPECT_GT(res.gn_iters, 0);
  EXPECT_LT(res.dx_norm, cfg.convergence_eps);
  const Sophus::SE3d T_hat(res.pose.q, res.pose.t);
  EXPECT_LT((T_hat.inverse() * T_true).log().norm(), 1e-4);
  EXPECT_LT(res.chi, 1e-6);

  // Re-running converges to the same solution within floating-point noise. Bit-identity
  // is not required: the association sums across threads, so the rounding order varies.
  const RegistrationResult res2 = reg.registerScan(keypoints, nearest, guess, MotionPrior{});
  const Sophus::SE3d T_hat2(res2.pose.q, res2.pose.t);
  EXPECT_LT((T_hat2.inverse() * T_hat).log().norm(), 1e-6);
  EXPECT_LT((res2.cov_right - res.cov_right).norm(), 1e-6 * (1.0 + res.cov_right.norm()));
}

// Every deskewed return must equal the analytic warp exp((t_i - t_end) * [v; w])
// applied to the extrinsic-transformed point; out-of-range and zero returns drop.
TEST(ScanRegistration, DeskewMatchesAnalyticScrew) {
  const FrontendLio cfg;
  const ScanRegistration reg{cfg};

  const Pose T_body_lidar{Eigen::Quaterniond{Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY())},
                          Eigen::Vector3d{0.1, 0.02, -0.05}};
  const Eigen::Vector3d omega{0.02, -0.3, 0.5};
  const Eigen::Vector3d vel{1.0, -0.4, 0.2};

  auto cloud = std::make_shared<meridian::PointCloud>();
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(-20.0, 20.0);
  const int n_valid = 50;
  for (int i = 0; i < n_valid; ++i) {
    meridian::LidarPoint pt;
    pt.xyz = Eigen::Vector3f(static_cast<float>(u(rng)), static_cast<float>(u(rng)),
                             static_cast<float>(u(rng)));
    pt.t_offset_ns = i * 2'000'000;
    cloud->push_back(pt);
  }
  meridian::LidarPoint far_pt;
  far_pt.xyz = Eigen::Vector3f(150.f, 0.f, 0.f);
  cloud->push_back(far_pt);
  cloud->push_back(meridian::LidarPoint{});  // zero return

  meridian::LidarScan scan;
  scan.stamp_start = 1'000'000'000;
  scan.sweep_duration = 100'000'000;
  scan.points = cloud;

  const meridian::Timestamp t_end = scan.stamp_start + scan.sweep_duration;
  const std::vector<Eigen::Vector3d> out = reg.deskew(scan, T_body_lidar, omega, vel, t_end);
  ASSERT_EQ(out.size(), static_cast<std::size_t>(n_valid));

  const Sophus::SE3d T_bl(T_body_lidar.q, T_body_lidar.t);
  Vec6 screw;
  screw.head<3>() = vel;
  screw.tail<3>() = omega;
  for (int i = 0; i < n_valid; ++i) {
    const meridian::LidarPoint& pt = (*cloud)[static_cast<std::size_t>(i)];
    const double dt = meridian::to_seconds(scan.stamp_start + pt.t_offset_ns - t_end);
    const Eigen::Vector3d expected = Sophus::SE3d::exp(dt * screw) * (T_bl * pt.xyz.cast<double>());
    EXPECT_LT((out[static_cast<std::size_t>(i)] - expected).norm(), 1e-12) << "point " << i;
  }
}

// A small fronto-parallel wall patch leaves rotation about its normal (the world
// x-axis, gravity-orthogonal) nearly unconstrained by the data. With the scan
// pre-rotated about that axis, the solve must follow the data when the regularizer is
// off, hold the prior orientation when it is on, and land in between when a large
// acceleration variance weakens the weight.
TEST(ScanRegistration, GravityRegularizerAnchorsDegenerateRoll) {
  const FrontendLio cfg;
  std::vector<Eigen::Vector3d> wall;
  for (double y = -0.15; y <= 0.15 + 1e-9; y += 0.03) {
    for (double z = -0.15; z <= 0.15 + 1e-9; z += 0.03) {
      wall.emplace_back(0.4, y, z);
    }
  }
  ASSERT_GE(static_cast<int>(wall.size()), cfg.min_keypoints);
  const NearestLookup nearest = bruteForceNearest(wall, cfg.max_corr_dist_m);

  const double droll = 0.06;
  const Sophus::SO3d R_corrupt = Sophus::SO3d::exp(-droll * Eigen::Vector3d::UnitX());
  std::vector<Eigen::Vector3d> keypoints;
  keypoints.reserve(wall.size());
  for (const Eigen::Vector3d& p : wall) {
    keypoints.push_back(R_corrupt * p);
  }

  const ScanRegistration reg{cfg};
  const Pose guess;  // identity, the prior pose the regularizer anchors to

  const MotionPrior off;  // imu_count = 0 disables the block
  MotionPrior strong;
  strong.imu_count = 10;
  strong.accel_mag_variance = 0.0;
  MotionPrior weak;
  weak.imu_count = 10;
  weak.accel_mag_variance = 50.0;

  const double roll_off = rollOf(reg.registerScan(keypoints, nearest, guess, off));
  const double roll_strong = rollOf(reg.registerScan(keypoints, nearest, guess, strong));
  const double roll_weak = rollOf(reg.registerScan(keypoints, nearest, guess, weak));

  EXPECT_NEAR(roll_off, droll, 0.01);        // pure data recovers the corruption
  EXPECT_LT(std::abs(roll_strong), 0.01);    // strong prior pins the degenerate axis
  EXPECT_GT(roll_weak, roll_strong + 0.01);  // larger variance weakens the pull
  EXPECT_LT(roll_weak, roll_off - 0.005);
}

// The reported information and covariance must replicate the exact chain at the solved
// pose: unaveraged data-only H, sigma2 = chi/(3N-6), Sigma_left = sigma2*(H + eps*I)^-1,
// cov_right = Ad(T^-1)*Sigma_left*Ad(T^-1)^T, info_body = Ad(T)^T*H*Ad(T) — with the
// gravity regularizer active, proving it never leaks into either output.
TEST(ScanRegistration, CovarianceChainReplicatesAtSolvedPose) {
  const FrontendLio cfg;
  const std::vector<Eigen::Vector3d> map_pts = boxCornerCloud(3.0, 0.25);
  const NearestLookup nearest = bruteForceNearest(map_pts, cfg.max_corr_dist_m);

  Vec6 xi_true;
  xi_true << 1.5, -0.8, 0.6, 0.2, -0.15, 0.25;
  const Sophus::SE3d T_true = Sophus::SE3d::exp(xi_true);
  std::vector<Eigen::Vector3d> keypoints = transformAll(T_true.inverse(), map_pts);
  // Small body-frame noise keeps chi (and so sigma2) strictly positive.
  std::mt19937 rng(21);
  std::normal_distribution<double> nd(0.0, 0.005);
  for (Eigen::Vector3d& p : keypoints) {
    p += Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
  }

  Vec6 dxi;
  dxi << 0.04, -0.03, 0.02, 0.01, -0.012, 0.008;
  const Pose guess = toPose(Sophus::SE3d::exp(dxi) * T_true);

  MotionPrior prior_on;
  prior_on.imu_count = 10;
  prior_on.accel_mag_variance = 0.0;

  const ScanRegistration reg{cfg};
  const RegistrationResult res = reg.registerScan(keypoints, nearest, guess, prior_on);
  ASSERT_TRUE(res.converged);
  ASSERT_GE(res.n_corr, cfg.min_keypoints);

  const Sophus::SE3d T_hat(res.pose.q, res.pose.t);
  Mat6 H = Mat6::Zero();
  double chi = 0.0;
  int n = 0;
  Eigen::Vector3d neighbor = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& p : keypoints) {
    const Eigen::Vector3d pw = T_hat * p;
    if (!nearest(pw, &neighbor)) {
      continue;
    }
    const Mat36 J = dataJacobian(pw);
    H += J.transpose() * J;
    chi += (pw - neighbor).squaredNorm();
    ++n;
  }
  ASSERT_GT(n, 2);
  EXPECT_EQ(n, res.n_corr);
  EXPECT_NEAR(chi, res.chi, 1e-9 * (1.0 + chi));

  const Mat6 adj_T = T_hat.Adj();
  const Mat6 info_expected = adj_T.transpose() * H * adj_T;
  EXPECT_LT((res.info_body - info_expected).norm() / info_expected.norm(), 1e-9);

  const double sigma2 = chi / static_cast<double>(3 * n - 6);
  const Mat6 cov_left = sigma2 * (H + 1e-6 * Mat6::Identity()).inverse();
  const Mat6 adj_T_inv = T_hat.inverse().Adj();
  const Mat6 cov_expected = adj_T_inv * cov_left * adj_T_inv.transpose();
  EXPECT_LT((res.cov_right - cov_expected).norm() / cov_expected.norm(), 1e-6);
}

// Repeated registrations of the same scene under fresh isotropic noise: the empirical
// right-chart errors log(T_hat^-1 * T_true) must be statistically consistent with the
// reported cov_right. The mean NEES of a calibrated 6-DoF estimate is 6; the envelope
// allows a factor-2 miscalibration either way. The large ground-truth offset makes the
// left-to-right adjoint transport load-bearing.
TEST(ScanRegistration, MonteCarloCovarianceIsConsistent) {
  const FrontendLio cfg;
  const std::vector<Eigen::Vector3d> map_pts = boxCornerCloud(3.0, 0.25);
  const NearestLookup nearest = bruteForceNearest(map_pts, cfg.max_corr_dist_m);

  Vec6 xi_true;
  xi_true << 6.0, -4.0, 2.5, 0.4, -0.3, 0.5;
  const Sophus::SE3d T_true = Sophus::SE3d::exp(xi_true);
  const std::vector<Eigen::Vector3d> clean = transformAll(T_true.inverse(), map_pts);

  const ScanRegistration reg{cfg};
  std::mt19937 rng(1234);
  std::normal_distribution<double> nd(0.0, 0.01);

  const int trials = 80;
  double nees_sum = 0.0;
  for (int t = 0; t < trials; ++t) {
    std::vector<Eigen::Vector3d> noisy = clean;
    for (Eigen::Vector3d& p : noisy) {
      p += Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
    }
    const RegistrationResult res = reg.registerScan(noisy, nearest, toPose(T_true), MotionPrior{});
    ASSERT_TRUE(res.converged);
    const Sophus::SE3d T_hat(res.pose.q, res.pose.t);
    const Vec6 err = (T_hat.inverse() * T_true).log();
    nees_sum += err.dot(res.cov_right.ldlt().solve(err));
  }
  const double mean_nees = nees_sum / trials;
  EXPECT_GT(mean_nees, 3.0);
  EXPECT_LT(mean_nees, 12.0);
}

// composeRelativeCov against brute force: sample both perturbations, compose the noisy
// poses, and compare the empirical second moment of log((X*D)^-1 * X*exp(xi1)*D*exp(xi2))
// = log(D^-1*exp(xi1)*D*exp(xi2)) with the transported-and-added prediction.
TEST(RelativeCov, ComposeMatchesTwoStepMonteCarlo) {
  Vec6 sig1;
  sig1 << 0.02, 0.015, 0.01, 0.006, 0.009, 0.012;
  Vec6 sig2;
  sig2 << 0.01, 0.02, 0.015, 0.012, 0.005, 0.008;
  const Mat6 cov1 = sig1.array().square().matrix().asDiagonal();
  const Mat6 cov2 = sig2.array().square().matrix().asDiagonal();

  Vec6 xi_d;
  xi_d << 0.4, -0.3, 0.9, 0.5, -0.2, 0.3;
  const Sophus::SE3d delta = Sophus::SE3d::exp(xi_d);

  const Mat6 predicted = composeRelativeCov(cov1, toPose(delta), cov2);

  std::mt19937 rng(99);
  std::normal_distribution<double> nd(0.0, 1.0);
  const int samples = 4000;
  Mat6 empirical = Mat6::Zero();
  for (int s = 0; s < samples; ++s) {
    Vec6 x1;
    Vec6 x2;
    for (int i = 0; i < 6; ++i) {
      x1[i] = sig1[i] * nd(rng);
      x2[i] = sig2[i] * nd(rng);
    }
    const Vec6 xi_tot =
        (delta.inverse() * Sophus::SE3d::exp(x1) * delta * Sophus::SE3d::exp(x2)).log();
    empirical += xi_tot * xi_tot.transpose();
  }
  empirical /= static_cast<double>(samples);
  EXPECT_LT((empirical - predicted).norm() / predicted.norm(), 0.15);
}

// The pack-time reorder is self-inverse and places blocks exactly: on a hand-built
// matrix the diagonal 3-blocks swap and the off-diagonal coupling transposes.
TEST(RelativeCov, ReorderTransRotRoundTripsOnHandBuiltMatrix) {
  Mat6 m;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      m(i, j) = 10.0 * i + j;
    }
  }
  const Mat6 r = meridian::reorderTransRotToRotTrans(m);
  EXPECT_TRUE((r.block<3, 3>(0, 0).array() == m.block<3, 3>(3, 3).array()).all());
  EXPECT_TRUE((r.block<3, 3>(3, 3).array() == m.block<3, 3>(0, 0).array()).all());
  EXPECT_TRUE((r.block<3, 3>(0, 3).array() == m.block<3, 3>(3, 0).array()).all());
  EXPECT_TRUE((r.block<3, 3>(3, 0).array() == m.block<3, 3>(0, 3).array()).all());

  const Mat6 back = meridian::reorderTransRotToRotTrans(r);
  EXPECT_TRUE((back.array() == m.array()).all());
}

}  // namespace
