#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/pose.hpp"
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

}  // namespace meridian::backend::testing
