#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"
#include "spline_window.hpp"

namespace ceres {
class Problem;
}  // namespace ceres

namespace meridian::ct {

// Local east-north-up tangent-plane frame anchored at the first accepted GNSS fix.
// The first fix's geodetic (lat, lon, alt) becomes the ENU origin; every later fix is
// converted to a metric ENU offset from it. The conversion is a first-order (small-
// area) tangent-plane approximation: a fix at displacement (de, dn, du) maps to local
// coordinates (de, dn, du) directly, with de/dn from the meridian/prime-vertical radii
// of curvature at the anchor latitude and du the ellipsoidal-height difference. This is
// exact only at the anchor and accumulates curvature error that grows quadratically
// with baseline; over the tens-of-metres-to-low-kilometres open-sky stretches L2 uses
// GNSS for, the error is sub-decimetre and well below the fix-quality floors. The
// authoritative global datum (the full LLA -> map transform with its own uncertainty)
// is L3's job; this anchor only gives the conservative L2 residual a metric frame.
//
// The anchor identifies the local ENU frame with the L2 odometry world frame W: the
// first accepted fix's antenna is taken to sit at the spline antenna position at the
// fix time, so z_W(fix) = local-ENU(fix). This is the L2 conservative convention (no
// datum heading/translation is estimated here); a real heading/translation alignment
// between ENU and W is L3's datum estimation.
class EnuAnchor {
public:
  EnuAnchor() = default;

  bool set() const { return set_; }

  // Anchors the ENU origin at this fix's geodetic position. The offset places the local
  // ENU origin so that this fix maps to `world_at_fix` (the spline antenna world
  // position at the fix time), tying the ENU frame to W at the first accepted fix.
  void anchor(const GnssFix& fix, const Eigen::Vector3d& world_at_fix);

  // Converts a fix's geodetic position to the local world frame W. Requires set().
  Eigen::Vector3d toWorld(const GnssFix& fix) const;

  double anchorLat() const { return lat0_deg_; }
  double anchorLon() const { return lon0_deg_; }
  double anchorAlt() const { return alt0_m_; }

private:
  bool set_ = false;
  double lat0_deg_ = 0.0;
  double lon0_deg_ = 0.0;
  double alt0_m_ = 0.0;
  // Metres-per-degree scale factors at the anchor latitude (north, east) and the
  // world-frame offset added to a raw ENU coordinate so the anchor fix maps to its
  // spline antenna position.
  double m_per_deg_lat_ = 0.0;
  double m_per_deg_lon_ = 0.0;
  Eigen::Vector3d enu_to_world_offset_ = Eigen::Vector3d::Zero();
};

// Per-fix-type position-std floor (horizontal, vertical) in metres, selected from the
// config by the fix's quality. `None` returns a zero floor (such a fix is rejected
// upstream before this is consulted).
Eigen::Vector2d fixTypeFloorStd(GnssFix::FixType fix, const FrontendGnss& cfg);

// The floored ENU position covariance: each diagonal element is raised to at least the
// squared per-fix-type floor std (horizontal floor on x/y, vertical floor on z), since a
// receiver's reported covariance is optimistic. Off-diagonal terms pass through. The
// floor is applied element-wise on the diagonal only.
Eigen::Matrix3d flooredCovEnu(const GnssFix& fix, const FrontendGnss& cfg);

// Outcome of gating one fix against the current spline pose. `innovation_m` is the
// Euclidean norm of the residual (fix minus interpolated antenna world position),
// surfaced as telemetry whether or not the fix is admitted.
struct GnssGateResult {
  enum class Reason { Accepted, Quality, Gate, ReacquirePersist };
  bool accepted = false;
  Reason reason = Reason::Accepted;
  double innovation_m = 0.0;
};

// Stateful gate enforcing the fix-quality floor, the Mahalanobis k-sigma innovation
// check, and the re-acquisition persistence rule across a fix stream. One instance is
// held by the front-end for the whole mission. The gate is consulted only on the
// active GNSS path (config use AND fixes present); on the inactive path it is never
// touched, so no GNSS state advances.
class GnssGate {
public:
  explicit GnssGate(const FrontendGnss& cfg) : cfg_(cfg) {}

  // Gates one fix. `r_world` is the residual z_W(fix) - p_W_ant(t_fix) at the current
  // spline pose; `cov_floored` the floored ENU covariance; `pose_marginal` the 3x3
  // position marginal of the spline pose at the fix time (added to the fix covariance so
  // the gate widens when odometry itself has drifted). A fix passing the Mahalanobis
  // gate still counts toward, and only after, the re-acquisition run is admitted.
  GnssGateResult gate(const Eigen::Vector3d& r_world, const Eigen::Matrix3d& cov_floored,
                      const Eigen::Matrix3d& pose_marginal);

  // Number of fixes accepted / rejected over the mission, for the accept-rate telemetry.
  int accepted() const { return accepted_; }
  int rejected() const { return rejected_; }

private:
  FrontendGnss cfg_;
  // Re-acquisition state: while `armed_` is false the window is in the re-acquire run
  // and `streak_` counts consecutive in-gate fixes; once the streak reaches the
  // configured count the gate arms and admits fixes immediately. A gated-out fix
  // disarms the gate and restarts the run, so a post-gap multipath burst cannot snap
  // the window before persistence is re-established.
  bool armed_ = false;
  int streak_ = 0;
  int accepted_ = 0;
  int rejected_ = 0;
};

// Adds one 3-DoF absolute-position residual binding the antenna world position at the
// fix's own time to the ENU-converted fix:
//   r = sqrt_info * ( z_W(fix) - ( p_W_Fe(t) + R_W_Fe(t) * t_fe_ant ) )
// DynamicAutoDiff over the 4 SO(3) + 4 R^3 knot blocks of t's segment, under a Huber
// loss. `sqrt_info` is the 3x3 upper-triangular square root of the inverse floored ENU
// covariance (the whitening matrix), so the residual is dimensionless and unit-variance.
// `z_world` is the fix already converted to W by the EnuAnchor. Returns true when the
// residual was added (the spline must cover t).
bool addGnssResidual(ceres::Problem& problem, SplineWindow& spline, const Eigen::Vector3d& z_world,
                     const Eigen::Vector3d& t_fe_ant, Timestamp t_fix,
                     const Eigen::Matrix3d& sqrt_info, double huber);

}  // namespace meridian::ct
