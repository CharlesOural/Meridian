#pragma once

#include <array>
#include <vector>

#include <Eigen/Core>

#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"

#include "spline_window.hpp"

namespace ceres {
class Problem;
class Manifold;
}  // namespace ceres

namespace meridian::ct {

// Gravity is a fixed-magnitude direction: storage is a unit 3-vector held on the
// unit sphere by a SphereManifold, and the world gravity vector is recovered as
// magnitude * direction. Keeping the magnitude out of the optimised state is what
// makes |g_W| constant across solves.
struct GravityBlock {
  // Initialises the direction from any world gravity vector (its sign/orientation
  // are preserved; only the stored magnitude is replaced by `magnitude`).
  GravityBlock(const Eigen::Vector3d& g_world, double magnitude = 9.81);

  // World gravity vector g_W = magnitude * direction.
  Eigen::Vector3d gravityWorld() const;
  double magnitude() const { return magnitude_; }
  double* data() { return direction_.data(); }
  const double* data() const { return direction_.data(); }

  Eigen::Vector3d direction_ = Eigen::Vector3d(0.0, 0.0, -1.0);  // unit
  double magnitude_ = 9.81;                                      // fixed [m/s^2]
};

// Piecewise-linear bias trajectory sampled at a coarse cadence. Each knot owns a
// 3-vector for gyro bias and a 3-vector for accel bias; the bias at an arbitrary
// time is the linear interpolation between the bracketing knots. Consecutive knots
// are tied by a continuous-time random-walk residual so the bias drifts smoothly.
//
// Knot count is fixed at construction and the backing storage is never resized, so
// the pointers returned by gyroBlock()/accelBlock() stay stable for the lifetime of
// the object — they may be held as Ceres parameter blocks across a solve.
class BiasKnots {
 public:
  // Lays out `n_knots` (>= 1) knots uniformly over [t0, t0 + (n_knots-1)*dt_ns].
  // With a single knot the bias is constant over the whole window.
  BiasKnots(Timestamp t0, Duration dt_ns, int n_knots);

  int numKnots() const { return static_cast<int>(gyro_.size()); }
  Duration knotDt() const { return dt_ns_; }
  Timestamp knotTime(int k) const { return t0_ + static_cast<Timestamp>(k) * dt_ns_; }

  // Index of the knot at or before t, clamped to [0, numKnots()-2] when more than
  // one knot exists so that index+1 is always a valid right bracket.
  int leftIndex(Timestamp t) const;
  // Linear interpolation weight in [0,1] for the right knot of the bracket at t.
  double alpha(Timestamp t) const;

  Eigen::Vector3d gyroBiasAt(Timestamp t) const;
  Eigen::Vector3d accelBiasAt(Timestamp t) const;

  double* gyroBlock(int k) { return gyro_[k].data(); }
  double* accelBlock(int k) { return accel_[k].data(); }
  const Eigen::Vector3d& gyroKnot(int k) const { return gyro_[k]; }
  const Eigen::Vector3d& accelKnot(int k) const { return accel_[k]; }
  void setGyroKnot(int k, const Eigen::Vector3d& b) { gyro_[k] = b; }
  void setAccelKnot(int k, const Eigen::Vector3d& b) { accel_[k] = b; }

 private:
  Timestamp t0_ = 0;
  Duration dt_ns_ = 0;
  std::vector<Eigen::Vector3d> gyro_;
  std::vector<Eigen::Vector3d> accel_;
};

// Per-stream weights for the IMU residuals. The four values are the noise standard
// deviations; the assembler converts them to sqrt-information weights internally.
struct ImuWeights {
  double sigma_gyro = 1.0;      // gyro white-noise std [rad/s]
  double sigma_accel = 1.0;     // accel white-noise std [m/s^2]
  double sigma_bias_gyro = 1.0;   // gyro bias random-walk density [rad/(s*sqrt(s))]
  double sigma_bias_accel = 1.0;  // accel bias random-walk density [m/(s^2*sqrt(s))]
};

// Creates and returns the SphereManifold<3> that constrains the gravity direction
// to the unit sphere. Ownership transfers to the caller (hand it to the Problem).
ceres::Manifold* makeGravityManifold();

// Peak per-sample IMU excitation over a set of samples, the gating statistic for
// adaptive knot density. n_omega is the max bias-corrected body angular-rate magnitude
// ||omega_m - b_g||; n_accel is the max gravity-removed specific-force magnitude
// | ||a_m - b_a|| - |g| |. Both are frame-invariant per-sample magnitudes, so a fast
// pure rotation (whose world-frame vector sum cancels) still reads large.
struct ImuExcitation {
  double n_omega = 0.0;  // [rad/s]
  double n_accel = 0.0;  // [m/s^2]
};
ImuExcitation imuExcitation(const std::vector<ImuSample>& samples,
                            const Eigen::Vector3d& b_g, const Eigen::Vector3d& b_a,
                            double gravity_mag);

// Maps the excitation statistics to a control-point count in [1, n_cp_max] with
// hysteresis. Each rising threshold crossed in omega_thresh / accel_thresh adds one
// control point (n_cp = 1 + bands crossed); the two mappings are combined by the
// larger. The mapping is monotone and piecewise-constant: stepping DOWN a band
// requires the statistic to fall below that band edge by `hysteresis` (fractional), so
// density does not chatter at a threshold. `prev_n_cp` is the previous segment's count
// (1 on the first segment). Same inputs always yield the same output.
int knotDensityFromExcitation(const ImuExcitation& exc,
                              const std::vector<double>& omega_thresh,
                              const std::vector<double>& accel_thresh, double hysteresis,
                              int n_cp_max, int prev_n_cp);

// Adds one combined gyro+accel residual per IMU sample whose stamp lies inside the
// span, plus the random-walk ties between consecutive bias knots, to `problem`.
// The spline knot blocks are taken from `spline.segmentFor(sample.stamp)`, the bias
// blocks from `bias`, and the single gravity block from `gravity`. The gravity
// block is pinned to the unit sphere here (the SphereManifold is installed on it if
// it has none yet) so its magnitude stays fixed across solves. Returns the number
// of IMU residuals added.
int addImuResiduals(ceres::Problem& problem, SplineWindow& spline, BiasKnots& bias,
                    GravityBlock& gravity, const std::vector<ImuSample>& samples,
                    Timestamp t_begin, Timestamp t_end, const ImuWeights& weights);

// Adds only the bias random-walk ties (exposed separately for tests). Returns the
// number of tie residuals added (numKnots()-1, gyro and accel counted as one each).
int addBiasRandomWalk(ceres::Problem& problem, BiasKnots& bias,
                      const ImuWeights& weights);

// Adds tail anchors at each covered time: a world-velocity residual on the R^3 spline
// derivative toward v_pred and a body-rate residual on the SO(3) spline toward w_pred.
// The trailing control points past the newest measurement have (near-)zero basis
// weight over every measured time, so without these the solver may park arbitrary
// values there at no cost; the anchors tie that span to the IMU-predicted constant
// velocity / body rate with honest extrapolation sigmas, weak enough for the next
// sweep's measurements to override. Returns the number of anchored times.
int addTailAnchors(ceres::Problem& problem, SplineWindow& spline,
                   const std::vector<Timestamp>& times, const Eigen::Vector3d& v_pred,
                   const Eigen::Vector3d& w_pred, double sigma_vel, double sigma_rate);

// Adds the low-weight under-excitation regularizer at each supplied time: a jerk
// residual (third virtual-time derivative of the R^3 spline, scaled to real time) and
// a body angular-acceleration residual (second derivative of the SO(3) spline), each
// pulling the trajectory toward constant velocity / constant body rate. The weights are
// `weight` relative to `accel_weight` (jerk, m/s^3) and to `gyro_weight` (angular
// accel, rad/s^2); a tiny absolute scale keeps both unitless residuals bounded. Only
// times the spline covers are used. Returns the number of residual blocks added (one
// jerk + one angular-accel per covered time). The caller gates engagement (excitation
// below floor AND a degenerate axis) and supplies the knot-midpoint times.
int addMotionRegularizer(ceres::Problem& problem, SplineWindow& spline,
                         const std::vector<Timestamp>& times, double weight,
                         double accel_weight, double gyro_weight);

}  // namespace meridian::ct
