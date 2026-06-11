#include "residuals_gnss.hpp"

// The vendored coefficient helpers return through const-ref outputs (const_cast
// inside), which GCC's flow analysis flags as maybe-uninitialized once they inline
// into the Jet-instantiated functor below; every entry is in fact overwritten.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <basalt/spline/ceres_spline_helper.h>
#pragma GCC diagnostic pop

#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sophus/so3.hpp>
#include <vector>

#include "spline_analytic.hpp"

namespace meridian::ct {

namespace {

constexpr int kSplineOrder = 4;

// WGS84 ellipsoid parameters for the local tangent-plane radii of curvature.
constexpr double kWgs84A = 6378137.0;           // semi-major axis [m]
constexpr double kWgs84ESq = 0.00669437999013;  // first eccentricity squared
constexpr double kDeg2Rad = M_PI / 180.0;

// 3-DoF antenna-position residual at a single fix's true time. The 4 SO(3) and 4 R^3
// knot blocks of the fix's segment are the parameter blocks, in that order; the
// ENU-converted fix, the antenna lever-arm, the segment position u / inv_dt, and the
// 3x3 whitening matrix are fixed data.
struct GnssResidual {
  GnssResidual(const Eigen::Vector3d& z_world, const Eigen::Vector3d& t_fe_ant, double u,
               double inv_dt, const Eigen::Matrix3d& sqrt_info,
               const Eigen::Matrix4d* blend = nullptr, const Eigen::Matrix4d* cum_blend = nullptr)
      : z_world_(z_world),
        t_fe_ant_(t_fe_ant),
        u_(u),
        inv_dt_(inv_dt),
        sqrt_info_(sqrt_info),
        non_uniform_(blend != nullptr && cum_blend != nullptr) {
    if (non_uniform_) {
      blend_ = *blend;
      cum_blend_ = *cum_blend;
    }
  }

  template <class T>
  bool operator()(T const* const* params, T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;

    SO3T R_w_fe;
    Vec3T p_w_fe;
    if (non_uniform_) {
      evaluateLieWithMatrix<T, Sophus::SO3>(params, u_, inv_dt_, cum_blend_, &R_w_fe);
      evaluateRdWithMatrix<T, 3, 0>(params + kSplineOrder, u_, inv_dt_, blend_, &p_w_fe);
    } else {
      basalt::CeresSplineHelper<kSplineOrder>::template evaluate_lie<T, Sophus::SO3>(
          params, u_, inv_dt_, &R_w_fe);
      basalt::CeresSplineHelper<kSplineOrder>::template evaluate<T, 3, 0>(params + kSplineOrder, u_,
                                                                          inv_dt_, &p_w_fe);
    }

    const Vec3T p_w_ant = R_w_fe * t_fe_ant_.cast<T>() + p_w_fe;
    const Vec3T r = z_world_.cast<T>() - p_w_ant;
    Eigen::Map<Vec3T> out(residual_ptr);
    out = sqrt_info_.cast<T>() * r;
    return true;
  }

  Eigen::Vector3d z_world_;
  Eigen::Vector3d t_fe_ant_;
  double u_;
  double inv_dt_;
  Eigen::Matrix3d sqrt_info_;
  // Per-interval blending matrices copied by value (the functor outlives the segment
  // reference it was built from); false selects the uniform cardinal matrices.
  bool non_uniform_ = false;
  Eigen::Matrix4d blend_ = Eigen::Matrix4d::Zero();
  Eigen::Matrix4d cum_blend_ = Eigen::Matrix4d::Zero();
};

}  // namespace

void EnuAnchor::anchor(const GnssFix& fix, const Eigen::Vector3d& world_at_fix) {
  lat0_deg_ = fix.lat_deg;
  lon0_deg_ = fix.lon_deg;
  alt0_m_ = fix.alt_m;

  // Radii of curvature at the anchor latitude give the metres-per-degree scale for the
  // small-area tangent-plane map. M (meridian radius) scales latitude -> north;
  // N (prime-vertical radius) times cos(lat) scales longitude -> east.
  const double lat_rad = lat0_deg_ * kDeg2Rad;
  const double sin_lat = std::sin(lat_rad);
  const double denom = 1.0 - kWgs84ESq * sin_lat * sin_lat;
  const double sqrt_denom = std::sqrt(denom);
  const double m_radius = kWgs84A * (1.0 - kWgs84ESq) / (denom * sqrt_denom);
  const double n_radius = kWgs84A / sqrt_denom;
  m_per_deg_lat_ = m_radius * kDeg2Rad;
  m_per_deg_lon_ = n_radius * std::cos(lat_rad) * kDeg2Rad;

  // The anchor fix maps to its spline antenna world position: raw ENU at the anchor is
  // zero, so the offset is exactly that world position.
  enu_to_world_offset_ = world_at_fix;
  set_ = true;
}

Eigen::Vector3d EnuAnchor::toWorld(const GnssFix& fix) const {
  // First-order tangent-plane ENU offset from the anchor: east from the longitude
  // delta, north from the latitude delta, up from the ellipsoidal-height delta.
  const double east = (fix.lon_deg - lon0_deg_) * m_per_deg_lon_;
  const double north = (fix.lat_deg - lat0_deg_) * m_per_deg_lat_;
  const double up = fix.alt_m - alt0_m_;
  return enu_to_world_offset_ + Eigen::Vector3d(east, north, up);
}

Eigen::Vector2d fixTypeFloorStd(GnssFix::FixType fix, const FrontendGnss& cfg) {
  switch (fix) {
    case GnssFix::FixType::RTK_Fixed:
      return {cfg.floor_fixed_h, cfg.floor_fixed_v};
    case GnssFix::FixType::RTK_Float:
      return {cfg.floor_float_h, cfg.floor_float_v};
    case GnssFix::FixType::DGPS:
      return {cfg.floor_dgps_h, cfg.floor_dgps_v};
    case GnssFix::FixType::SPP:
      return {cfg.floor_spp_h, cfg.floor_spp_v};
    case GnssFix::FixType::None:
      break;
  }
  return {0.0, 0.0};
}

Eigen::Matrix3d flooredCovEnu(const GnssFix& fix, const FrontendGnss& cfg) {
  const Eigen::Vector2d floor = fixTypeFloorStd(fix.fix, cfg);
  const double var_h = floor.x() * floor.x();
  const double var_v = floor.y() * floor.y();
  Eigen::Matrix3d cov = fix.cov_enu;
  cov(0, 0) = std::max(cov(0, 0), var_h);
  cov(1, 1) = std::max(cov(1, 1), var_h);
  cov(2, 2) = std::max(cov(2, 2), var_v);
  return cov;
}

GnssGateResult GnssGate::gate(const Eigen::Vector3d& r_world, const Eigen::Matrix3d& cov_floored,
                              const Eigen::Matrix3d& pose_marginal) {
  GnssGateResult res;
  res.innovation_m = r_world.norm();

  // The gate covariance is the floored fix covariance plus the current pose marginal, so
  // a long GNSS gap during which odometry has itself drifted does not reject an otherwise
  // good fix; the symmetrised sum stays positive-definite.
  const Eigen::Matrix3d S = cov_floored + pose_marginal;
  const Eigen::Matrix3d S_sym = 0.5 * (S + S.transpose());
  const Eigen::LLT<Eigen::Matrix3d> llt(S_sym);
  double mahal = std::numeric_limits<double>::infinity();
  if (llt.info() == Eigen::Success) {
    // r^T S^-1 r via the Cholesky factor: solve L y = r, then ||y||^2.
    const Eigen::Vector3d y = llt.matrixL().solve(r_world);
    mahal = std::sqrt(y.squaredNorm());
  }

  if (mahal > cfg_.innovation_k) {
    // A gated-out fix disarms the gate so re-acquisition persistence restarts: the next
    // accepted run must again reach reacquire_count before fixes are admitted.
    armed_ = false;
    streak_ = 0;
    ++rejected_;
    res.accepted = false;
    res.reason = GnssGateResult::Reason::Gate;
    return res;
  }

  // In-gate. If already armed, admit immediately. Otherwise count toward the
  // re-acquisition run; the fix is held out (not admitted) until the run completes.
  if (armed_) {
    ++accepted_;
    res.accepted = true;
    res.reason = GnssGateResult::Reason::Accepted;
    return res;
  }
  ++streak_;
  const int need = std::max(cfg_.reacquire_count, 1);
  if (streak_ >= need) {
    armed_ = true;
    ++accepted_;
    res.accepted = true;
    res.reason = GnssGateResult::Reason::Accepted;
    return res;
  }
  ++rejected_;
  res.accepted = false;
  res.reason = GnssGateResult::Reason::ReacquirePersist;
  return res;
}

bool addGnssResidual(ceres::Problem& problem, SplineWindow& spline, const Eigen::Vector3d& z_world,
                     const Eigen::Vector3d& t_fe_ant, Timestamp t_fix,
                     const Eigen::Matrix3d& sqrt_info, double huber) {
  if (!spline.covers(t_fix)) {
    return false;
  }
  const SplineWindow::SegmentRef seg = spline.segmentFor(t_fix);

  auto* functor = new GnssResidual(z_world, t_fe_ant, seg.u, 1.0 / seg.dt_s, sqrt_info, seg.blend,
                                   seg.cum_blend);
  auto* cost = new ceres::DynamicAutoDiffCostFunction<GnssResidual, 4>(functor);
  for (int i = 0; i < 2 * kSplineOrder; ++i) {
    cost->AddParameterBlock(i < kSplineOrder ? 4 : 3);
  }
  cost->SetNumResiduals(3);

  std::vector<double*> blocks;
  blocks.reserve(2 * kSplineOrder);
  for (double* p : seg.so3_knots) {
    blocks.push_back(p);
  }
  for (double* p : seg.r3_knots) {
    blocks.push_back(p);
  }

  auto* loss = new ceres::HuberLoss(huber);
  problem.AddResidualBlock(cost, loss, blocks);
  return true;
}

}  // namespace meridian::ct
