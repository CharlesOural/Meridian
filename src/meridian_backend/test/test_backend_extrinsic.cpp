#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "geodetic.hpp"
#include "meridian/backend/ibackend.hpp"
#include "synthetic.hpp"

using meridian::BackendConfig;
using meridian::CalibrationSet;
using meridian::Extrinsic;
using meridian::Frame;
using meridian::GnssFix;
using meridian::IBackEnd;
using meridian::KeyframePacket;
using meridian::Pose;
using meridian::Timestamp;
using meridian::backend::GeodeticDatum;
using meridian::backend::testing::CountingSink;
using meridian::backend::testing::make_chain;
using meridian::backend::testing::make_fix;
using meridian::backend::testing::SynthChain;
using meridian::backend::testing::SynthOptions;

namespace {

// GNSS-bearing config with the master extrinsic-refine switch on and decimation relaxed.
BackendConfig refineConfig() {
  BackendConfig cfg;
  cfg.gnss_enabled = true;
  cfg.gnss_min_spacing = 0.0;
  cfg.gnss_skip_if_confident = false;
  cfg.gnss_max_cov = 1e6;
  cfg.extrinsic_refine = true;
  // The production default prior (1e-2 m) is tight by design — it assumes a trusted calibration
  // that drifts only mm-cm. These tests exercise the refinement mechanism over larger, synthetic
  // lever errors, so the prior is loosened to let the fixes move the estimate.
  cfg.extrinsic_refine_sigma = 0.5;
  return cfg;
}

// Calibration carrying the GNSS lever at the (possibly wrong) offline value, flagged for
// refinement.
std::shared_ptr<CalibrationSet> calibRefine(const Eigen::Vector3d& offline_lever) {
  auto calib = std::make_shared<CalibrationSet>();
  Extrinsic e;
  e.child = Frame::GnssLink;
  e.parent = Frame::ImuLink;
  e.T_parent_child = Pose{Eigen::Quaterniond::Identity(), offline_lever};
  e.refine_online = true;
  calib->extrinsics.push_back(e);
  return calib;
}

GeodeticDatum hkOrigin() {
  GeodeticDatum o;
  o.lat0_deg = 22.337;
  o.lon0_deg = 114.263;
  o.alt0_m = 30.0;
  o.set = true;
  return o;
}

constexpr double kFixBeta = 0.6;
Timestamp fixStamp(const SynthChain& chain, std::size_t i) {
  const Timestamp a = chain.packets[i - 1].stamp;
  const Timestamp b = chain.packets[i].stamp;
  return a + (b - a) * 3 / 5;
}
Pose interpAtFix(const Pose& xi, const Pose& xj) {
  return xi.boxplus(xj.boxminus(xi) * kFixBeta);
}

void feed(IBackEnd& be, const KeyframePacket& p) {
  be.add_keyframe(KeyframePacket(p));
}

Eigen::Vector3d refinedLever(const IBackEnd& be) {
  return be.refined_calibration()->extrinsic(Frame::GnssLink).T_parent_child.t;
}

// Drives a constant-curvature arc (continuous yaw + translation) with GNSS fixes synthesized from
// `true_lever`, while the back-end starts from `offline_lever`. The arc excites both horizontal
// axes, so the datum yaw is observable and the extrinsic excitation gate opens.
std::unique_ptr<IBackEnd> drive(const Eigen::Vector3d& offline_lever,
                                const Eigen::Vector3d& true_lever, CountingSink* sink, int n = 90,
                                double max_dev = -1.0) {
  const Pose T_map_enu{Eigen::Quaterniond(Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitZ())),
                       Eigen::Vector3d(3.0, -2.0, 0.5)};
  const GeodeticDatum origin = hkOrigin();

  SynthOptions opt;
  opt.n = n;
  opt.step_m = 1.0;
  opt.yaw_step_rad = 0.10;  // continuous turn -> excitation on both axes
  const SynthChain chain = make_chain(opt);

  BackendConfig cfg = refineConfig();
  if (max_dev > 0.0) {
    cfg.extrinsic_max_dev = max_dev;
  }
  auto be = meridian::makeBackEnd(cfg, calibRefine(offline_lever), sink, /*deterministic=*/true);
  feed(*be, chain.packets[0]);
  be->optimize();
  std::uint32_t seed = 4000;
  for (std::size_t i = 1; i < chain.packets.size(); ++i) {
    feed(*be, chain.packets[i]);
    be->optimize();
    const Pose gt_at_fix = interpAtFix(chain.gt[i - 1], chain.gt[i]);
    const GnssFix f = make_fix(gt_at_fix, true_lever, T_map_enu, origin, fixStamp(chain, i),
                               /*sigma_m=*/0.02, GnssFix::FixType::RTK_Fixed, seed++);
    be->add_absolute(f, static_cast<std::uint64_t>(i));
    be->optimize();
  }
  return be;
}

}  // namespace

// With excitation, the online lever is refined from a wrong offline value toward the true one
// that generated the fixes, and the published calibration carries the refined value.
TEST(BackendExtrinsic, LeverConvergesUnderExcitation) {
  const Eigen::Vector3d true_lever(0.80, 0.00, 0.30);
  // A small drift, the regime this feature targets (a trusted calibration that flexes mm-cm).
  const Eigen::Vector3d offline_lever(0.82, -0.015, 0.31);

  CountingSink sink;
  auto be = drive(offline_lever, true_lever, &sink);

  const Eigen::Vector3d est = refinedLever(*be);
  const double err_after = (est - true_lever).norm();
  const double err_before = (offline_lever - true_lever).norm();
  EXPECT_GE(sink.count("backend/extrinsic_excited"), 1) << "excitation gate never opened";
  EXPECT_NE(est, offline_lever) << "lever was never refined / published";
  EXPECT_LT(err_after, err_before)
      << "refinement worsened the lever (got " << est.transpose() << ")";
}

// Off by default: with the master switch off, the lever is never graphed and the published
// calibration stays at the offline value regardless of excitation.
TEST(BackendExtrinsic, OffByDefaultLeavesLeverAtOffline) {
  const Eigen::Vector3d true_lever(0.80, 0.00, 0.30);
  const Eigen::Vector3d offline_lever(0.86, 0.04, 0.26);

  const Pose T_map_enu{Eigen::Quaterniond(Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitZ())),
                       Eigen::Vector3d(3.0, -2.0, 0.5)};
  const GeodeticDatum origin = hkOrigin();
  SynthOptions opt;
  opt.n = 60;
  opt.yaw_step_rad = 0.10;
  const SynthChain chain = make_chain(opt);

  BackendConfig cfg = refineConfig();
  cfg.extrinsic_refine = false;  // the default
  CountingSink sink;
  auto be = meridian::makeBackEnd(cfg, calibRefine(offline_lever), &sink, /*deterministic=*/true);
  feed(*be, chain.packets[0]);
  be->optimize();
  std::uint32_t seed = 5000;
  for (std::size_t i = 1; i < chain.packets.size(); ++i) {
    feed(*be, chain.packets[i]);
    be->optimize();
    const Pose gt_at_fix = interpAtFix(chain.gt[i - 1], chain.gt[i]);
    be->add_absolute(make_fix(gt_at_fix, true_lever, T_map_enu, origin, fixStamp(chain, i), 0.02,
                              GnssFix::FixType::RTK_Fixed, seed++),
                     static_cast<std::uint64_t>(i));
    be->optimize();
  }

  EXPECT_EQ(refinedLever(*be), offline_lever);
  EXPECT_EQ(sink.count("backend/extrinsic_excited"), 0);
}

// FM-5: a true lever far outside the offline box drives the estimate past the sanity bound, which
// must reject the refinement, revert to the offline lever, and warn.
TEST(BackendExtrinsic, ClampRejectsRunawayLever) {
  const Eigen::Vector3d offline_lever(0.80, 0.00, 0.30);
  const Eigen::Vector3d true_lever(0.80, 0.00, 0.55);  // a large, out-of-box discrepancy

  // A tight sanity box: as the estimate moves off the offline value it must trip the clamp.
  CountingSink sink;
  auto be = drive(offline_lever, true_lever, &sink, /*n=*/90, /*max_dev=*/0.003);

  EXPECT_GE(sink.count("backend/extrinsic_clamped"), 1) << "runaway lever was not clamped";
  // Reverted to the offline value; it never adopts the out-of-box estimate.
  EXPECT_EQ(refinedLever(*be), offline_lever);
}
