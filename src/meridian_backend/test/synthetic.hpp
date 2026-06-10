#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "geodetic.hpp"
#include "meridian/common/imu_preintegration.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/debug/telemetry.hpp"

namespace meridian::backend::testing {

// Generator parameters for a contiguous keyframe chain with known ground truth.
// noise_* are 1-sigma magnitudes of the per-edge tangent perturbation ([m], [rad]);
// zero means the relatives reproduce the ground truth exactly.
struct SynthOptions {
  int n = 50;
  double step_m = 1.0;
  double yaw_step_rad = 0.05;
  double noise_trans = 0.0;
  double noise_rot = 0.0;
  std::uint32_t seed = 42;
  double cov_trans = 1e-4;
  double cov_rot = 1e-5;
};

struct SynthChain {
  std::vector<KeyframePacket> packets;
  std::vector<Pose> gt;  // gt[i] = T_odom_body of keyframe i
};

// Builds an n-keyframe chain: packet 0 is an AbsolutePrior at identity, packets 1..n-1
// are RelativeBetween edges along a constant-curvature arc (step_m forward, yaw_step_rad
// per step). T_ref_body is the composition of the (possibly noisy) relatives, so it plays
// the role of the front-end's drifting odometry hint.
inline SynthChain make_chain(const SynthOptions& opt) {
  SynthChain chain;
  chain.gt.reserve(static_cast<std::size_t>(opt.n));
  chain.packets.reserve(static_cast<std::size_t>(opt.n));

  const Pose increment{
      Eigen::Quaterniond(Eigen::AngleAxisd(opt.yaw_step_rad, Eigen::Vector3d::UnitZ())),
      Eigen::Vector3d(opt.step_m, 0.0, 0.0)};

  chain.gt.push_back(Pose{});
  for (int i = 1; i < opt.n; ++i) {
    chain.gt.push_back(chain.gt.back() * increment);
  }

  // Rotation-first [rx,ry,rz,tx,ty,tz] to match the packet's boundary convention.
  GaussianBlock<6> cov;
  cov.M.diagonal() << opt.cov_rot, opt.cov_rot, opt.cov_rot, opt.cov_trans, opt.cov_trans,
      opt.cov_trans;

  std::mt19937 rng(opt.seed);
  std::normal_distribution<double> gauss(0.0, 1.0);

  Pose odom;  // running composition of the noisy relatives
  for (int i = 0; i < opt.n; ++i) {
    KeyframePacket p;
    p.id = static_cast<std::uint64_t>(i);
    p.stamp = static_cast<Timestamp>(1'000'000'000LL) + static_cast<Timestamp>(i) * 100'000'000LL;
    p.ref_frame = Frame::Odom;
    p.constraint_cov = cov;
    p.calib_version = 1;
    p.frontend_kind = 1;

    if (i == 0) {
      p.constraint_kind = KeyframePacket::ConstraintKind::AbsolutePrior;
      p.T_ref_body = chain.gt[0];
    } else {
      p.constraint_kind = KeyframePacket::ConstraintKind::RelativeBetween;
      p.rel_to_id = static_cast<std::uint64_t>(i - 1);
      Pose rel = chain.gt[static_cast<std::size_t>(i - 1)].inverse() *
                 chain.gt[static_cast<std::size_t>(i)];
      Eigen::Matrix<double, 6, 1> xi = Eigen::Matrix<double, 6, 1>::Zero();
      if (opt.noise_trans > 0.0) {
        for (int k = 0; k < 3; ++k) xi[k] = opt.noise_trans * gauss(rng);
      }
      if (opt.noise_rot > 0.0) {
        for (int k = 3; k < 6; ++k) xi[k] = opt.noise_rot * gauss(rng);
      }
      rel = rel.boxplus(xi);
      p.T_relto_this = rel;
      odom = odom * rel;
      p.T_ref_body = odom;
    }
    chain.packets.push_back(std::move(p));
  }
  return chain;
}

// Telemetry sink that tallies events per tag and scalars per key so tests can assert
// "this path emitted that event" without parsing logs. Everything else is a no-op.
class CountingSink final : public TelemetrySink {
public:
  std::map<std::string, int> events;
  std::map<std::string, int> scalars;
  std::string last_event_msg;

  bool enabled(const char*) const override { return true; }

  void scalar(const char* key, double, Timestamp) override { ++scalars[key]; }
  void vec(const char*, const Eigen::Ref<const Eigen::VectorXd>&, Timestamp, const char*) override {
  }

  void cloud(const char*, const PointCloudView&, Frame, Timestamp) override {}
  void pose(const char*, const Pose&, Frame, Timestamp) override {}
  void marker(const Marker&, Timestamp) override {}
  void image(const char*, const ImageOverlay&, Timestamp) override {}

  void timing(const char*, double, Timestamp) override {}

  void event(Level, const char* tag, std::string_view msg, Timestamp) override {
    ++events[tag];
    last_event_msg.assign(msg.data(), msg.size());
  }

  int count(const char* tag) const {
    const auto it = events.find(tag);
    return it == events.end() ? 0 : it->second;
  }
};

// Builds a KeyframePacket carrying an ImuPreintegrationSummary (the window-restart
// fallback). The motion is a closed-form constant-rate integration of a fixed body-frame
// angular rate omega and specific force accel over N steps of dt seconds, so the summary
// is self-consistent in the [dR | dv | dp] order the boundary uses.
//
// The summary is geometrically valid in isolation but its pose/motion need NOT be
// physically consistent with the surrounding chain: these tests exercise graph
// bookkeeping (which variables/factors the bridge adds and marginalizes), not trajectory
// accuracy, so the bridge edge is allowed to disagree with T_ref_body.
inline KeyframePacket make_restart_packet(std::uint64_t prev_id, std::uint64_t id, Timestamp stamp,
                                          const Pose& T_ref_body, std::uint32_t seed = 7) {
  // Per-axis constant body rate and specific force; seeded so distinct restarts differ
  // but a fixed seed is reproducible. The magnitudes are deliberately mild.
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-0.3, 0.3);
  const Eigen::Vector3d omega(0.05 + u(rng), -0.04 + u(rng), 0.03 + u(rng));  // [rad/s]
  const Eigen::Vector3d accel(0.2 + u(rng), -0.1 + u(rng), 0.15 + u(rng));    // [m/s^2]

  constexpr int kN = 20;
  constexpr double kDt = 0.005;  // [s] per IMU step -> 0.1 s window
  const double total_s = kN * kDt;

  ImuPreintegrationSummary s;
  s.t_i = stamp - static_cast<Timestamp>(total_s * 1e9);
  s.t_j = stamp;
  s.gravity_mag = 9.81;
  s.bias_g_lin = Eigen::Vector3d::Zero();
  s.bias_a_lin = Eigen::Vector3d::Zero();

  // Forward Euler on the gravity-free preintegrated increment, accumulating the
  // first-order bias Jacobians along the way:
  //   dR_{k+1} = dR_k * Exp(omega*dt)
  //   dv_{k+1} = dv_k + dR_k * accel * dt
  //   dp_{k+1} = dp_k + dv_k * dt + 0.5 * dR_k * accel * dt^2
  // A perturbation of the gyro bias rotates every subsequent dR; an accel-bias
  // perturbation feeds through dR_k into dv and dp. Rotation carries no accel-bias
  // dependence, so dR_dba is identically zero and is not stored.
  Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
  Eigen::Vector3d dv = Eigen::Vector3d::Zero();
  Eigen::Vector3d dp = Eigen::Vector3d::Zero();
  Eigen::Matrix3d dR_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dba = Eigen::Matrix3d::Zero();

  const auto skew = [](const Eigen::Vector3d& w) {
    Eigen::Matrix3d m;
    m << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0;
    return m;
  };

  // Constant-rate step rotation Exp(omega*dt) and its transpose, formed once.
  const Eigen::Vector3d phi = omega * kDt;
  const double phi_norm = phi.norm();
  const Eigen::Matrix3d inc =
      (phi_norm > 0.0 ? Eigen::Matrix3d(Eigen::AngleAxisd(phi_norm, phi / phi_norm))
                      : Eigen::Matrix3d::Identity());
  const Eigen::Matrix3d inc_T = inc.transpose();

  for (int k = 0; k < kN; ++k) {
    const Eigen::Matrix3d dR_k = dR;
    const Eigen::Vector3d dv_k = dv;
    const Eigen::Matrix3d dR_dbg_k = dR_dbg;
    const Eigen::Vector3d acc_w = dR_k * accel;  // accel in the i-frame at step k

    dp += dv_k * kDt + 0.5 * acc_w * kDt * kDt;
    dv += acc_w * kDt;

    dp_dbg += dv_dbg * kDt - 0.5 * skew(acc_w) * dR_dbg_k * kDt * kDt;
    dp_dba += dv_dba * kDt + 0.5 * dR_k * kDt * kDt;
    dv_dbg += -skew(acc_w) * dR_dbg_k * kDt;
    dv_dba += dR_k * kDt;

    // d(dR)/d(bg): rotating one more step propagates the accumulated Jacobian through
    // the new increment and subtracts this step's right-Jacobian contribution.
    dR_dbg = inc_T * dR_dbg_k - dR_k * kDt;
    dR = dR_k * inc;
  }

  const Eigen::Quaterniond delta_R(dR);
  s.delta_R = delta_R.normalized();
  s.delta_v = dv;
  s.delta_p = dp;
  s.dR_dbg = dR_dbg;
  s.dv_dbg = dv_dbg;
  s.dv_dba = dv_dba;
  s.dp_dbg = dp_dbg;
  s.dp_dba = dp_dba;

  // Small SPD diagonal in [dR | dv | dp] order; rotation tightest, position loosest,
  // matching how preintegration uncertainty grows by integration order.
  s.preint_cov.form = GaussianBlock<9>::Form::Covariance;
  s.preint_cov.M.setZero();
  for (int k = 0; k < 3; ++k) s.preint_cov.M(k, k) = 1e-4;  // dR
  for (int k = 3; k < 6; ++k) s.preint_cov.M(k, k) = 1e-3;  // dv
  for (int k = 6; k < 9; ++k) s.preint_cov.M(k, k) = 1e-2;  // dp

  KeyframePacket p;
  p.id = id;
  p.stamp = stamp;
  p.ref_frame = Frame::Odom;
  p.T_ref_body = T_ref_body;
  p.constraint_kind = KeyframePacket::ConstraintKind::ImuPreintegration;
  p.rel_to_id = prev_id;
  // The restart fallback carries the kinematic state so the bridge can seed V and B.
  p.kinematics_included = true;
  p.v_ref = dv;  // a plausible body velocity in ref_frame; not chain-consistent
  p.b_g = Eigen::Vector3d::Zero();
  p.b_a = Eigen::Vector3d::Zero();
  p.imu_summary = std::move(s);
  p.calib_version = 1;
  p.frontend_kind = 1;
  return p;
}

// Builds an L-shaped keyframe chain with a known ground truth: n_before steps along +x,
// a single 90-degree left turn, then n_after steps along +y (n_before + n_after + 1
// keyframes total). The right-angle gives the buffered ENU track excitation along BOTH
// horizontal axes, so the datum-init yaw is well observable (a straight track leaves yaw
// near-singular and is rejected by the Hessian gate). Same packet layout as make_chain:
// packet 0 is an AbsolutePrior at the start pose, the rest are exact RelativeBetween edges,
// so the chain is its own optimum.
inline SynthChain make_l_chain(int n_before, int n_after, double step_m,
                               const SynthOptions& base = SynthOptions{}) {
  SynthChain chain;
  const int n = n_before + n_after + 1;
  chain.gt.reserve(static_cast<std::size_t>(n));
  chain.packets.reserve(static_cast<std::size_t>(n));

  const Eigen::Quaterniond q_turn(Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()));
  const Pose step_fwd{Eigen::Quaterniond::Identity(), Eigen::Vector3d(step_m, 0.0, 0.0)};
  // A single in-place 90-degree yaw at the corner, then a straight leg in the rotated
  // body frame keeps walking the world +y axis.
  const Pose turn{q_turn, Eigen::Vector3d::Zero()};

  chain.gt.push_back(Pose{});
  for (int i = 1; i < n; ++i) {
    if (i == n_before + 1) {
      chain.gt.push_back(chain.gt.back() * turn * step_fwd);
    } else {
      chain.gt.push_back(chain.gt.back() * step_fwd);
    }
  }

  GaussianBlock<6> cov;
  cov.M.diagonal() << base.cov_rot, base.cov_rot, base.cov_rot, base.cov_trans, base.cov_trans,
      base.cov_trans;

  for (int i = 0; i < n; ++i) {
    KeyframePacket p;
    p.id = static_cast<std::uint64_t>(i);
    p.stamp = static_cast<Timestamp>(1'000'000'000LL) + static_cast<Timestamp>(i) * 100'000'000LL;
    p.ref_frame = Frame::Odom;
    p.constraint_cov = cov;
    p.calib_version = 1;
    p.frontend_kind = 1;

    if (i == 0) {
      p.constraint_kind = KeyframePacket::ConstraintKind::AbsolutePrior;
      p.T_ref_body = chain.gt[0];
    } else {
      p.constraint_kind = KeyframePacket::ConstraintKind::RelativeBetween;
      p.rel_to_id = static_cast<std::uint64_t>(i - 1);
      p.T_relto_this = chain.gt[static_cast<std::size_t>(i - 1)].inverse() *
                       chain.gt[static_cast<std::size_t>(i)];
      p.T_ref_body = chain.gt[static_cast<std::size_t>(i)];
    }
    chain.packets.push_back(std::move(p));
  }
  return chain;
}

// Numerically inverts geodetic lla_to_enu: returns the (lat,lon,alt) whose ENU position
// about `origin` is `enu`. Using the back-end's OWN lla_to_enu as the forward model makes
// the round-trip exact to solver tolerance regardless of the WGS84 ellipsoid details, so
// a fix synthesized here reprojects back to exactly the intended p_enu inside the graph
// (no model-mismatch bias). The Jacobian is the near-constant local metres-per-degree
// scaling, so a few fixed-point steps converge; the altitude axis is decoupled (1 m / m).
inline void enu_to_lla(const Eigen::Vector3d& enu, const GeodeticDatum& origin, double* lat_deg,
                       double* lon_deg, double* alt_m) {
  // Local linear sensitivities d(enu)/d(deg) about the origin, estimated by finite
  // differences through the real forward map; ~1e-6 deg keeps the secant well-conditioned.
  constexpr double kDegEps = 1e-6;
  const Eigen::Vector3d e_lat =
      (lla_to_enu(origin.lat0_deg + kDegEps, origin.lon0_deg, origin.alt0_m, origin)) / kDegEps;
  const Eigen::Vector3d e_lon =
      (lla_to_enu(origin.lat0_deg, origin.lon0_deg + kDegEps, origin.alt0_m, origin)) / kDegEps;
  // 2x2 horizontal sensitivity (E,N) vs (lat,lon); alt maps 1:1 to Up by construction.
  Eigen::Matrix2d J;
  J << e_lat.x(), e_lon.x(), e_lat.y(), e_lon.y();
  const Eigen::Matrix2d Jinv = J.inverse();

  double lat = origin.lat0_deg, lon = origin.lon0_deg;
  double alt = origin.alt0_m + enu.z();
  for (int it = 0; it < 8; ++it) {
    const Eigen::Vector3d cur = lla_to_enu(lat, lon, alt, origin);
    const Eigen::Vector2d res(enu.x() - cur.x(), enu.y() - cur.y());
    if (res.norm() < 1e-9) break;
    const Eigen::Vector2d d = Jinv * res;
    lat += d.x();
    lon += d.y();
  }
  *lat_deg = lat;
  *lon_deg = lon;
  *alt_m = alt;
}

// Synthesizes a GNSS fix consistent with a ground-truth map pose, a chosen datum
// T_map_enu, and a geodetic origin. The antenna sits at gt_map_pose * lever in map; that
// point is pushed into ENU through T_map_enu^{-1}, then a zero-mean isotropic position
// error of sigma_m (in ENU metres) is added before converting back to lla. cov_enu is set
// isotropic at sigma_m^2 so the back-end weights the fix exactly as drawn. The same seed
// reproduces the same fix, so two deterministic runs see byte-identical input.
inline GnssFix make_fix(const Pose& gt_map_pose, const Eigen::Vector3d& lever,
                        const Pose& T_map_enu, const GeodeticDatum& origin, Timestamp stamp,
                        double sigma_m, GnssFix::FixType fix, std::uint32_t seed) {
  const Eigen::Vector3d antenna_map = gt_map_pose * lever;
  Eigen::Vector3d enu = T_map_enu.inverse() * antenna_map;

  if (sigma_m > 0.0) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, sigma_m);
    enu += Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng));
  }

  GnssFix f;
  f.stamp = stamp;
  f.sensor_id = 0;
  f.sensor_frame = Frame::GnssLink;
  enu_to_lla(enu, origin, &f.lat_deg, &f.lon_deg, &f.alt_m);
  const double var = (sigma_m > 0.0 ? sigma_m * sigma_m : 1e-4);
  f.cov_enu = var * Eigen::Matrix3d::Identity();
  f.fix = fix;
  f.num_sats = 12;
  return f;
}

}  // namespace meridian::backend::testing
