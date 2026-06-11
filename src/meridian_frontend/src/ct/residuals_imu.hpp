#pragma once

#include <Eigen/Core>
#include <array>
#include <deque>
#include <vector>

#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "spline_window.hpp"

namespace ceres {
class Problem;
class Manifold;
class CostFunction;
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
// The timeline grows via extendTo and slides via dropOldest; deque storage keeps
// the pointers returned by gyroBlock()/accelBlock() stable for the lifetime of each
// knot, so they may be held as Ceres parameter blocks and prior references.
class BiasKnots {
public:
  // Lays out `n_knots` (>= 1) knots uniformly over [t0, t0 + (n_knots-1)*dt_ns].
  // With a single knot the bias is constant over the whole window. Knot storage is a
  // deque: push_back/pop_front never move surviving elements, so Ceres parameter
  // blocks and marginalization-prior pointers into live knots stay valid as the
  // timeline grows and slides.
  BiasKnots(Timestamp t0, Duration dt_ns, int n_knots);

  // Deep copy of the knot timeline and every knot value into independent deque
  // storage, so a rebuilt residual set points at copy-owned blocks. Index i in the
  // copy matches index i in the original.
  BiasKnots clone() const;

  // Appends knots (each copying the last knot's value -- the random-walk mean) until
  // the final knot time is at or past t, so leftIndex(t)+1 is a real bracket.
  void extendTo(Timestamp t);

  // Drops the n oldest knots, advancing t0; clamped so at least two knots remain
  // (one bracket). The caller must ensure no live parameter-block pointer references
  // a dropped knot (see lowestKnotIndexOf).
  void dropOldest(int n);

  // Lowest knot index whose gyro or accel storage backs any of the given pointers
  // (max int when none does); bounds dropOldest the same way the spline guards its
  // front-trim against prior-held blocks.
  int lowestKnotIndexOf(const std::vector<const double*>& ptrs) const;

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
  std::deque<Eigen::Vector3d> gyro_;
  std::deque<Eigen::Vector3d> accel_;
};

// Per-stream weights for the IMU residuals. The four values are the noise standard
// deviations; the assembler converts them to sqrt-information weights internally.
struct ImuWeights {
  double sigma_gyro = 1.0;        // gyro white-noise std [rad/s]
  double sigma_accel = 1.0;       // accel white-noise std [m/s^2]
  double sigma_bias_gyro = 1.0;   // gyro bias random-walk density [rad/(s*sqrt(s))]
  double sigma_bias_accel = 1.0;  // accel bias random-walk density [m/(s^2*sqrt(s))]
};

// Creates and returns the SphereManifold<3> that constrains the gravity direction
// to the unit sphere. Ownership transfers to the caller (hand it to the Problem).
ceres::Manifold* makeGravityManifold();

// Peak per-sample IMU excitation over a set of samples, the gating statistic for the
// under-excitation regularizer. n_omega is the max bias-corrected body angular-rate
// magnitude ||omega_m - b_g||; n_accel is the max gravity-removed specific-force
// magnitude | ||a_m - b_a|| - |g| |. Both are frame-invariant per-sample magnitudes,
// so a fast pure rotation (whose world-frame vector sum cancels) still reads large.
struct ImuExcitation {
  double n_omega = 0.0;  // [rad/s]
  double n_accel = 0.0;  // [m/s^2]
};
ImuExcitation imuExcitation(const std::vector<ImuSample>& samples, const Eigen::Vector3d& b_g,
                            const Eigen::Vector3d& b_a, double gravity_mag);

// Mean per-sample IMU statistics over one segment's samples (t0 < stamp <= t1), the
// gearing statistic for adaptive knot density. mean_omega is the mean of the RAW
// per-sample gyro norms ||omega_m|| (no bias correction: the bias is orders below the
// band edges and a stale bias estimate must never gate density) -- a mean of NORMS,
// so a fast rotation that sweeps the sample vectors through every direction (whose
// vector mean cancels toward zero) still reads its full rate. mean_accel_dev is
// | ||mean of a_m|| - g |, the deviation of the mean specific-force VECTOR's norm
// from gravity: zero-mean vibration (step impacts while walking) cancels in the
// vector mean instead of inflating a norm mean (E||g+n|| > g for zero-mean n), while
// sustained acceleration shifts the mean vector and reads through. n is the sample
// count; n == 0 means the segment carries no data yet.
struct ImuMeanExcitation {
  double mean_omega = 0.0;      // [rad/s]
  double mean_accel_dev = 0.0;  // [m/s^2]
  int n = 0;
};
ImuMeanExcitation segmentMeanExcitation(const std::vector<ImuSample>& samples, Timestamp t0,
                                        Timestamp t1, double gravity_mag);

// Maps the segment-mean statistics to a control-point count ("gear") in [1, n_cp_max]
// with hysteresis. Each rising threshold crossed in omega_thresh / accel_thresh adds
// one control point (gear = 1 + bands crossed); the two mappings are combined by the
// larger. Stepping DOWN out of a band held last segment requires the statistic to
// fall below that band edge by `hysteresis` (fractional), so density does not chatter
// at a threshold. A segment with no samples (stats.n == 0) holds prev_gear: its data
// has not arrived, so there is no evidence to change on. Pure function of its inputs.
int gearFromMeanExcitation(const ImuMeanExcitation& stats, const std::vector<double>& omega_thresh,
                           const std::vector<double>& accel_thresh, double hysteresis,
                           int n_cp_max, int prev_gear);

// Analytic cost-function factories for the IMU factor family. Each returns a freshly
// allocated ceres::CostFunction whose ownership transfers to the AddResidualBlock
// caller. The `*Autodiff` twins build the equivalent autodiff functor and exist for
// the analytic-vs-autodiff derivative-correctness tests. Parameter-block order matches
// the assemblers: 4 SO(3) knots (size 4), 4 R^3 knots (size 3), then bias blocks, then
// the gravity direction (size 3). kSharedBias selects 2 bias blocks (b_g, b_a) vs 4
// (bg_left, bg_right, ba_left, ba_right) interpolated by alpha.
//
// `blend` / `cum_blend` are the per-interval (non-uniform) blending matrices of the
// segment the cost evaluates on; nullptr selects the uniform cardinal matrices and is
// bit-identical to a build without the parameters. The pointers are only read inside
// the factory call (analytic costs bake the blended coefficients; autodiff twins copy
// the matrices by value), so no lifetime extends past it.
template <bool kSharedBias>
ceres::CostFunction* makeImuCost(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel,
                                 double u, double inv_dt, double alpha, double gravity_mag,
                                 double w_gyro, double w_accel,
                                 const Eigen::Matrix4d* blend = nullptr,
                                 const Eigen::Matrix4d* cum_blend = nullptr);
template <bool kSharedBias>
ceres::CostFunction* makeImuCostAutodiff(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel,
                                         double u, double inv_dt, double alpha, double gravity_mag,
                                         double w_gyro, double w_accel,
                                         const Eigen::Matrix4d* blend = nullptr,
                                         const Eigen::Matrix4d* cum_blend = nullptr);

ceres::CostFunction* makeJerkCost(double u, double inv_dt, double weight,
                                  const Eigen::Matrix4d* blend = nullptr);
ceres::CostFunction* makeJerkCostAutodiff(double u, double inv_dt, double weight,
                                          const Eigen::Matrix4d* blend = nullptr);
ceres::CostFunction* makeVelocityAnchorCost(const Eigen::Vector3d& v_pred, double u, double inv_dt,
                                            double weight,
                                            const Eigen::Matrix4d* blend = nullptr);
ceres::CostFunction* makeVelocityAnchorCostAutodiff(const Eigen::Vector3d& v_pred, double u,
                                                    double inv_dt, double weight,
                                                    const Eigen::Matrix4d* blend = nullptr);
ceres::CostFunction* makeAngularAccelCost(double u, double inv_dt, double weight,
                                          const Eigen::Matrix4d* cum_blend = nullptr);
ceres::CostFunction* makeAngularAccelCostAutodiff(double u, double inv_dt, double weight,
                                                  const Eigen::Matrix4d* cum_blend = nullptr);
ceres::CostFunction* makeRateAnchorCost(const Eigen::Vector3d& w_pred, double u, double inv_dt,
                                        double weight,
                                        const Eigen::Matrix4d* cum_blend = nullptr);
ceres::CostFunction* makeRateAnchorCostAutodiff(const Eigen::Vector3d& w_pred, double u,
                                                double inv_dt, double weight,
                                                const Eigen::Matrix4d* cum_blend = nullptr);
ceres::CostFunction* makeBiasTieCost(double w_gyro, double w_accel);
ceres::CostFunction* makeBiasTieCostAutodiff(double w_gyro, double w_accel);

// Adds one combined gyro+accel residual per IMU sample whose stamp lies inside the
// span, plus the random-walk ties between consecutive bias knots, to `problem`.
// The spline knot blocks are taken from `spline.segmentFor(sample.stamp)`, the bias
// blocks from `bias`, and the single gravity block from `gravity`. The gravity
// block is pinned to the unit sphere here (the SphereManifold is installed on it if
// it has none yet) so its magnitude stays fixed across solves. Returns the number
// of IMU residuals added.
int addImuResiduals(ceres::Problem& problem, SplineWindow& spline, BiasKnots& bias,
                    GravityBlock& gravity, const std::vector<ImuSample>& samples, Timestamp t_begin,
                    Timestamp t_end, const ImuWeights& weights);

// Adds the random-walk tie between every pair of consecutive resident bias knots at
// full weight 1/(sigma*sqrt(knot_dt)). The tie is a model constraint re-stated per
// problem (it never enters the marginalization prior from a live solve; a departing
// knot's tie folds into the prior via the Schur drop in the window slide), so full
// weight per problem counts it exactly once. Returns the number of ties added.
int addBiasRandomWalk(ceres::Problem& problem, BiasKnots& bias, const ImuWeights& weights);

// Adds tail anchors at each covered time: a world-velocity residual on the R^3 spline
// derivative toward v_pred and a body-rate residual on the SO(3) spline toward w_pred.
// They brace the span past the newest measurement, where basis support fades to zero:
// the pinning rule removes the strict null directions, and the anchors additionally
// hold the velocity read-out to the IMU prediction when the seed carries model error
// (e.g. an unconverged bias) -- weak enough for real measurements to override.
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
                         const std::vector<Timestamp>& times, double weight, double accel_weight,
                         double gyro_weight);

}  // namespace meridian::ct
