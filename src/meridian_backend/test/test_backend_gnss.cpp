#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

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
using meridian::StampedPose;
using meridian::Timestamp;
using meridian::backend::GeodeticDatum;
using meridian::backend::testing::CountingSink;
using meridian::backend::testing::make_chain;
using meridian::backend::testing::make_fix;
using meridian::backend::testing::make_l_chain;
using meridian::backend::testing::SynthChain;
using meridian::backend::testing::SynthOptions;

namespace {

constexpr const char* kDatumLockedEvent = "backend/datum_locked";

// A GNSS-bearing config: enable the datum path and relax decimation so the test can drive
// a dense, well-excited fix stream without fighting the spacing/confidence gates. The
// datum-init gates (baseline/excitation/speed/yaw-sigma) stay at their real defaults — the
// L-shaped track is built to pass them, the stationary track to fail them.
BackendConfig gnssConfig() {
  BackendConfig cfg;
  cfg.gnss_enabled = true;
  cfg.gnss_min_spacing = 0.0;          // admit every fix; spacing is exercised elsewhere
  cfg.gnss_skip_if_confident = false;  // don't let a tight marginal silence the stream
  cfg.gnss_max_cov = 1e6;              // never drop on quality in these synthetic streams
  return cfg;
}

// A calibration carrying only the GNSS lever arm the factor needs. The antenna sits ahead
// of and above the body origin so a body yaw rotates the lever materially — without a
// non-trivial lever the antenna track equals the body track and yaw stays unobservable.
std::shared_ptr<CalibrationSet> calibWithLever(const Eigen::Vector3d& lever) {
  auto calib = std::make_shared<CalibrationSet>();
  Extrinsic e;
  e.child = Frame::GnssLink;
  e.parent = Frame::ImuLink;
  e.T_parent_child = Pose{Eigen::Quaterniond::Identity(), lever};
  calib->extrinsics.push_back(e);
  return calib;
}

std::unique_ptr<IBackEnd> makeBackend(const BackendConfig& cfg,
                                      std::shared_ptr<const CalibrationSet> calib,
                                      meridian::TelemetrySink* sink) {
  return meridian::makeBackEnd(cfg, std::move(calib), sink, /*deterministic=*/true);
}

// add_keyframe consumes its argument, so feed a copy to keep the chain reusable.
void feed(IBackEnd& be, const KeyframePacket& p) {
  be.add_keyframe(KeyframePacket(p));
}

constexpr double kFixBeta = 0.6;  // fix-time fraction along the interval (i-1, i)

// Timestamp of a fix bound to the interval (i-1, i) at kFixBeta. With 0.1 s spacing and a
// 3/5 fraction the integer ns are exact, so the back-end's beta equals kFixBeta to the bit
// and the synthesized antenna position lands on the factor's interpolant. Both bracketing
// keyframes must already be in the graph; the interpolated GnssFactor path is exercised.
Timestamp fixStampInInterval(const SynthChain& chain, std::size_t i) {
  const Timestamp a = chain.packets[i - 1].stamp;
  const Timestamp b = chain.packets[i].stamp;
  return a + (b - a) * 3 / 5;
}

// The constant-velocity body pose at the fix time: Xi * Exp(beta * Log(Xi^-1 Xj)). Matches
// gtsam::interpolate, so make_fix places the antenna exactly where the factor evaluates it.
Pose interpAtFix(const Pose& xi, const Pose& xj) {
  return xi.boxplus(xj.boxminus(xi) * kFixBeta);
}

// Serializes a trajectory field-by-field for a bit-exact memcmp (mirrors test_determinism).
std::vector<std::byte> pack(const std::vector<StampedPose>& traj) {
  std::vector<std::byte> buf;
  const auto put = [&buf](const void* p, std::size_t n) {
    const auto* b = static_cast<const std::byte*>(p);
    buf.insert(buf.end(), b, b + n);
  };
  for (const StampedPose& sp : traj) {
    put(&sp.stamp, sizeof(sp.stamp));
    put(&sp.kf_id, sizeof(sp.kf_id));
    const double q[4] = {sp.T_map_body.q.w(), sp.T_map_body.q.x(), sp.T_map_body.q.y(),
                         sp.T_map_body.q.z()};
    put(q, sizeof(q));
    put(sp.T_map_body.t.data(), 3 * sizeof(double));
  }
  return buf;
}

// Mean translation error of a corrected trajectory against a ground-truth map track.
double ate(const std::vector<StampedPose>& traj, const std::vector<Pose>& gt) {
  double acc = 0.0;
  for (const StampedPose& sp : traj) {
    acc += (sp.T_map_body.t - gt[sp.kf_id].t).norm();
  }
  return traj.empty() ? 0.0 : acc / static_cast<double>(traj.size());
}

}  // namespace

// An L-shaped track gives the buffered ENU correspondences excitation on both horizontal
// axes, so the datum-init yaw is observable and the Hessian gate passes. After enough
// post-corner fixes the datum locks: the event fires, diagnostics report it, and the
// locking batch flags wants_immediate_optimize so the correction does not wait for cadence.
TEST(BackendGnss, DatumLocksOnExcitedTrack) {
  const Eigen::Vector3d lever(0.8, 0.0, 0.3);
  // A small yaw + translation map<-enu offset the datum fit must recover.
  const Pose T_map_enu{Eigen::Quaterniond(Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitZ())),
                       Eigen::Vector3d(3.0, -2.0, 0.5)};
  GeodeticDatum origin;
  origin.lat0_deg = 47.376;
  origin.lon0_deg = 8.548;
  origin.alt0_m = 400.0;
  origin.set = true;

  // 6 steps east, turn, 6 steps north; 2 m per step -> 12 m baseline on each leg, well past
  // the 5 m baseline / 3 m excitation gates.
  const SynthChain chain = make_l_chain(/*n_before=*/6, /*n_after=*/6, /*step_m=*/2.0);

  CountingSink sink;
  auto be = makeBackend(gnssConfig(), calibWithLever(lever), &sink);

  // Prime the first keyframe so every fix has both bracketing poses already in the graph.
  feed(*be, chain.packets[0]);
  be->optimize();

  bool saw_immediate_on_lock = false;
  std::uint32_t seed = 1000;
  for (std::size_t i = 1; i < chain.packets.size(); ++i) {
    feed(*be, chain.packets[i]);
    be->optimize();

    // One mid-interval fix bound to the just-completed interval (i-1, i).
    const bool was_locked = be->diagnostics().datum_locked;
    const Pose gt_at_fix = interpAtFix(chain.gt[i - 1], chain.gt[i]);
    const GnssFix f = make_fix(gt_at_fix, lever, T_map_enu, origin, fixStampInInterval(chain, i),
                               /*sigma_m=*/0.05, GnssFix::FixType::RTK_Fixed, seed++);
    be->add_absolute(f, /*nearest_kf_id=*/static_cast<std::uint64_t>(i));

    // The just-locked datum is a staged correction that must not wait for the cadence
    // timer, so the hint is sampled before optimize() consumes the batch. The lock itself
    // is confirmed only after that optimize() folds the seed + prior into the estimate.
    const bool hint_before_optimize = be->wants_immediate_optimize();
    be->optimize();
    if (!was_locked && be->diagnostics().datum_locked) {
      saw_immediate_on_lock = hint_before_optimize;
    }
  }

  EXPECT_GE(sink.count(kDatumLockedEvent), 1);
  EXPECT_TRUE(be->diagnostics().datum_locked);
  EXPECT_TRUE(saw_immediate_on_lock);
  EXPECT_GT(be->diagnostics().num_gnss_factors, 0u);
}

// A platform parked at one pose produces a buffered ENU track that is pure jitter: no
// baseline, no excitation, no moving fixes. The datum-init gates never pass, so the datum
// stays unlocked, GNSS contributes nothing, and the corrected trajectory still equals the
// odometry-only estimate (here, the anchored stationary pose).
TEST(BackendGnss, StationaryNeverLocks) {
  const Eigen::Vector3d lever(0.8, 0.0, 0.3);
  const Pose T_map_enu{};  // identity datum; stationarity is what we test, not alignment
  GeodeticDatum origin;
  origin.lat0_deg = 47.376;
  origin.lon0_deg = 8.548;
  origin.alt0_m = 400.0;
  origin.set = true;

  // A chain that does not move: zero step, zero yaw -> every gt pose is identity.
  SynthOptions opt;
  opt.n = 30;
  opt.step_m = 0.0;
  opt.yaw_step_rad = 0.0;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = makeBackend(gnssConfig(), calibWithLever(lever), &sink);

  feed(*be, chain.packets[0]);
  be->optimize();

  std::uint32_t seed = 2000;
  for (std::size_t i = 1; i < chain.packets.size(); ++i) {
    feed(*be, chain.packets[i]);
    be->optimize();
    // Jittery fixes about the stationary antenna: sigma 0.3 m, no real motion. The gt is
    // identity throughout, so the interpolant equals identity and the fix jitters in place.
    const Pose gt_at_fix = interpAtFix(chain.gt[i - 1], chain.gt[i]);
    const GnssFix f = make_fix(gt_at_fix, lever, T_map_enu, origin, fixStampInInterval(chain, i),
                               /*sigma_m=*/0.3, GnssFix::FixType::SPP, seed++);
    be->add_absolute(f, /*nearest_kf_id=*/static_cast<std::uint64_t>(i));
    be->optimize();
  }

  EXPECT_EQ(sink.count(kDatumLockedEvent), 0);
  EXPECT_FALSE(be->diagnostics().datum_locked);
  EXPECT_EQ(be->diagnostics().num_gnss_factors, 0u);

  // GNSS added nothing, so the estimate is still the anchored stationary pose.
  const auto traj = be->corrected_trajectory();
  for (const StampedPose& sp : traj) {
    EXPECT_LT(sp.T_map_body.t.norm(), 1e-6) << "kf " << sp.kf_id;
  }
}

// Odometry that drifts from ground truth, plus consistent post-lock GNSS fixes: after the
// datum locks and the fixes pull the later keyframes, the corrected trajectory is closer to
// the GNSS-implied (ground-truth) track than the pure-odometry estimate. Loose tolerance —
// an integration sanity check that GNSS information reaches the poses, not a precision test.
TEST(BackendGnss, GnssPullsDriftedTrajectory) {
  const Eigen::Vector3d lever(0.8, 0.0, 0.3);
  const Pose T_map_enu{Eigen::Quaterniond(Eigen::AngleAxisd(0.10, Eigen::Vector3d::UnitZ())),
                       Eigen::Vector3d(1.0, -1.0, 0.2)};
  GeodeticDatum origin;
  origin.lat0_deg = 47.376;
  origin.lon0_deg = 8.548;
  origin.alt0_m = 400.0;
  origin.set = true;

  // L-track ground truth; a noisy odometry copy that accumulates drift along the chain.
  SynthOptions base;
  base.noise_trans = 0.04;
  base.noise_rot = 0.01;
  base.seed = 9;
  const SynthChain gt_chain = make_l_chain(/*n_before=*/8, /*n_after=*/8, /*step_m=*/2.0);

  // Re-derive a drifted odometry by perturbing each exact relative edge; gt stays the truth.
  SynthChain drift = gt_chain;
  {
    std::mt19937 rng(base.seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    Pose odom;  // running composition of the perturbed relatives
    for (std::size_t i = 1; i < drift.packets.size(); ++i) {
      Eigen::Matrix<double, 6, 1> xi = Eigen::Matrix<double, 6, 1>::Zero();
      for (int k = 0; k < 3; ++k) xi[k] = base.noise_trans * gauss(rng);
      for (int k = 3; k < 6; ++k) xi[k] = base.noise_rot * gauss(rng);
      drift.packets[i].T_relto_this = drift.packets[i].T_relto_this.boxplus(xi);
      odom = odom * drift.packets[i].T_relto_this;
      drift.packets[i].T_ref_body = odom;
    }
  }

  // Odometry-only reference run: no GNSS at all.
  std::vector<StampedPose> odom_traj;
  {
    CountingSink sink;
    auto be = makeBackend(gnssConfig(), calibWithLever(lever), &sink);
    for (const KeyframePacket& p : drift.packets) {
      feed(*be, p);
      be->optimize();
    }
    odom_traj = be->corrected_trajectory();
  }

  // GNSS run: same drifted odometry, plus fixes consistent with the GT track + datum.
  std::vector<StampedPose> gnss_traj;
  {
    CountingSink sink;
    auto be = makeBackend(gnssConfig(), calibWithLever(lever), &sink);
    feed(*be, drift.packets[0]);
    be->optimize();
    std::uint32_t seed = 3000;
    for (std::size_t i = 1; i < drift.packets.size(); ++i) {
      feed(*be, drift.packets[i]);
      be->optimize();
      // Antenna from the GT interpolant; timestamp from the drifted chain (same stamps).
      const Pose gt_at_fix = interpAtFix(gt_chain.gt[i - 1], gt_chain.gt[i]);
      const GnssFix f = make_fix(gt_at_fix, lever, T_map_enu, origin, fixStampInInterval(drift, i),
                                 /*sigma_m=*/0.05, GnssFix::FixType::RTK_Fixed, seed++);
      be->add_absolute(f, /*nearest_kf_id=*/static_cast<std::uint64_t>(i));
      be->optimize();
    }
    ASSERT_TRUE(be->diagnostics().datum_locked);
    gnss_traj = be->corrected_trajectory();
  }

  // Compare only the post-lock tail, where the fixes have had a chance to pull the poses.
  const auto tail = [](const std::vector<StampedPose>& traj, std::size_t from) {
    std::vector<StampedPose> out;
    for (const StampedPose& sp : traj)
      if (sp.kf_id >= from) out.push_back(sp);
    return out;
  };
  const std::size_t lock_tail = drift.packets.size() / 2;
  const double odom_ate = ate(tail(odom_traj, lock_tail), gt_chain.gt);
  const double gnss_ate = ate(tail(gnss_traj, lock_tail), gt_chain.gt);

  EXPECT_LT(gnss_ate, odom_ate) << "odom ATE " << odom_ate << " gnss ATE " << gnss_ate;
}

// The full GNSS path is deterministic: two back-ends fed the byte-identical keyframe + fix
// sequence (optimize after every add, the replay cadence) produce bit-identical corrected
// trajectories. make_fix is seeded, so the input streams are themselves identical.
TEST(BackendGnss, GnssDeterministic) {
  const Eigen::Vector3d lever(0.8, 0.0, 0.3);
  const Pose T_map_enu{Eigen::Quaterniond(Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitZ())),
                       Eigen::Vector3d(3.0, -2.0, 0.5)};
  GeodeticDatum origin;
  origin.lat0_deg = 47.376;
  origin.lon0_deg = 8.548;
  origin.alt0_m = 400.0;
  origin.set = true;

  const SynthChain chain = make_l_chain(/*n_before=*/6, /*n_after=*/6, /*step_m=*/2.0);

  const auto run = [&]() {
    auto be = meridian::makeBackEnd(gnssConfig(), calibWithLever(lever),
                                    /*telemetry=*/nullptr, /*deterministic=*/true);
    be->add_keyframe(KeyframePacket(chain.packets[0]));
    be->optimize();
    std::uint32_t seed = 4000;
    for (std::size_t i = 1; i < chain.packets.size(); ++i) {
      be->add_keyframe(KeyframePacket(chain.packets[i]));
      be->optimize();
      const Pose gt_at_fix = interpAtFix(chain.gt[i - 1], chain.gt[i]);
      const GnssFix f = make_fix(gt_at_fix, lever, T_map_enu, origin, fixStampInInterval(chain, i),
                                 /*sigma_m=*/0.05, GnssFix::FixType::RTK_Fixed, seed++);
      be->add_absolute(f, /*nearest_kf_id=*/static_cast<std::uint64_t>(i));
      be->optimize();
    }
    return be->corrected_trajectory();
  };

  const auto traj_a = run();
  const auto traj_b = run();
  ASSERT_EQ(traj_a.size(), traj_b.size());

  const auto buf_a = pack(traj_a);
  const auto buf_b = pack(traj_b);
  ASSERT_EQ(buf_a.size(), buf_b.size());
  EXPECT_EQ(std::memcmp(buf_a.data(), buf_b.data(), buf_a.size()), 0);
}
