#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "meridian/backend/ibackend.hpp"
#include "synthetic.hpp"

using meridian::BackendConfig;
using meridian::CalibrationSet;
using meridian::GnssFix;
using meridian::GraphUpdate;
using meridian::IBackEnd;
using meridian::KeyframePacket;
using meridian::LoopConstraint;
using meridian::Pose;
using meridian::StampedPose;
using meridian::backend::testing::CountingSink;
using meridian::backend::testing::make_chain;
using meridian::backend::testing::SynthChain;
using meridian::backend::testing::SynthOptions;

namespace {

std::unique_ptr<IBackEnd> makeBackend(meridian::TelemetrySink* sink,
                                      const BackendConfig& cfg = BackendConfig{}) {
  return meridian::makeBackEnd(cfg, std::make_shared<CalibrationSet>(), sink,
                               /*deterministic=*/true);
}

std::vector<std::uint64_t> trajIds(const std::vector<StampedPose>& traj) {
  std::vector<std::uint64_t> ids;
  ids.reserve(traj.size());
  for (const StampedPose& p : traj) ids.push_back(p.kf_id);
  return ids;
}

// add_keyframe consumes its argument, so feed a copy to keep the chain reusable.
void feed(IBackEnd& be, const KeyframePacket& p) {
  be.add_keyframe(KeyframePacket(p));
}

}  // namespace

// A noise-free chain is its own graph optimum: with exact relatives and an anchored
// first pose, the corrected trajectory must reproduce the ground truth.
TEST(BackendSpine, IdentityProperty) {
  SynthOptions opt;
  opt.n = 60;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = makeBackend(&sink);
  for (const KeyframePacket& p : chain.packets) {
    feed(*be, p);
    be->optimize();
  }

  const auto traj = be->corrected_trajectory();
  ASSERT_EQ(traj.size(), static_cast<std::size_t>(opt.n));
  for (std::size_t i = 0; i < traj.size(); ++i) {
    EXPECT_EQ(traj[i].kf_id, i);
    const Pose& gt = chain.gt[i];
    const Pose& est = traj[i].T_map_body;
    EXPECT_LT((est.t - gt.t).norm(), 1e-9) << "kf " << i;
    EXPECT_LT(est.q.angularDistance(gt.q), 1e-10) << "kf " << i;
  }
}

// One variable per keyframe, and a no-op batch produces a no-op update: the second
// optimize() with nothing staged must report nothing moved.
TEST(BackendSpine, OneFactorPerEdge) {
  SynthOptions opt;
  opt.n = 20;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = makeBackend(&sink);
  for (const KeyframePacket& p : chain.packets) feed(*be, p);

  const GraphUpdate first = be->optimize();
  EXPECT_EQ(first.moved.size(), static_cast<std::size_t>(opt.n));
  EXPECT_EQ(be->diagnostics().num_keyframes, static_cast<std::uint64_t>(opt.n));

  const GraphUpdate second = be->optimize();
  EXPECT_TRUE(second.moved.empty());
  EXPECT_FALSE(second.loop_closed);
}

// An edge referencing anything but the last accepted keyframe is dropped whole, and the
// chain resumes from the last ACCEPTED id.
TEST(BackendSpine, ContiguityViolation) {
  SynthOptions opt;
  opt.n = 6;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = makeBackend(&sink);
  for (int i = 0; i <= 2; ++i) {
    feed(*be, chain.packets[static_cast<std::size_t>(i)]);
    be->optimize();
  }

  // Packet 4 references id 3, which was never accepted.
  feed(*be, chain.packets[4]);
  EXPECT_EQ(sink.count("backend/contiguity"), 1);
  be->optimize();
  EXPECT_EQ(trajIds(be->corrected_trajectory()), (std::vector<std::uint64_t>{0, 1, 2}));

  // Packet 3 is contiguous with the last accepted id (2), then 4 follows 3.
  feed(*be, chain.packets[3]);
  feed(*be, chain.packets[4]);
  be->optimize();
  EXPECT_EQ(sink.count("backend/contiguity"), 1);
  EXPECT_EQ(trajIds(be->corrected_trajectory()), (std::vector<std::uint64_t>{0, 1, 2, 3, 4}));
}

// A mid-stream AbsolutePrior is defensive territory: event + full drop, no variable.
TEST(BackendSpine, AbsoluteAfterFirstIgnored) {
  SynthOptions opt;
  opt.n = 5;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = makeBackend(&sink);
  for (int i = 0; i <= 3; ++i) {
    feed(*be, chain.packets[static_cast<std::size_t>(i)]);
    be->optimize();
  }

  KeyframePacket prior = chain.packets[4];
  prior.constraint_kind = KeyframePacket::ConstraintKind::AbsolutePrior;
  be->add_keyframe(std::move(prior));
  EXPECT_EQ(sink.count("backend/absolute_ignored"), 1);

  be->optimize();
  EXPECT_EQ(trajIds(be->corrected_trajectory()), (std::vector<std::uint64_t>{0, 1, 2, 3}));
}

// On a noise-free chain every optimize() after one new keyframe reports exactly that
// keyframe as moved (first appearance), and never re-lists settled old keyframes.
TEST(BackendSpine, MovedSemantics) {
  SynthOptions opt;
  opt.n = 30;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = makeBackend(&sink);
  for (const KeyframePacket& p : chain.packets) {
    const std::uint64_t id = p.id;
    feed(*be, p);
    const GraphUpdate up = be->optimize();
    ASSERT_EQ(up.moved.size(), 1u) << "kf " << id;
    EXPECT_EQ(up.moved[0].id, id);
    EXPECT_FALSE(up.loop_closed);
  }
}

TEST(BackendSpine, LoopGateRejectsLowFitness) {
  CountingSink sink;
  auto be = makeBackend(&sink);

  // A default LoopConstraint has fitness 0, below the floor, so it is rejected at the gate
  // before ever reaching PCM (the admit/evict path is exercised in test_backend_loops).
  be->add_loop_constraint(LoopConstraint{});
  EXPECT_EQ(sink.count("backend/loop_rejected_pcm"), 1);
  EXPECT_EQ(be->diagnostics().num_loops_rejected, 1u);
  EXPECT_EQ(be->diagnostics().num_loops, 0u);

  // A GNSS fix before any keyframe has no pose to anchor against; it must drop cleanly.
  EXPECT_NO_THROW(be->add_absolute(GnssFix{}, 0));
  EXPECT_EQ(be->diagnostics().num_gnss_factors, 0u);
}

// The latest-pose marginal exists after the first optimize, is finite, and its
// translation-block trace grows along the chain (uncertainty accumulates away from
// the anchored first pose).
TEST(BackendSpine, LatestMarginal) {
  SynthOptions opt;
  opt.n = 51;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = makeBackend(&sink);

  double trace5 = 0.0;
  double trace50 = 0.0;
  for (const KeyframePacket& p : chain.packets) {
    const std::uint64_t id = p.id;
    feed(*be, p);
    be->optimize();
    if (id == 5 || id == 50) {
      const auto marginal = be->latest_pose_marginal();
      ASSERT_TRUE(marginal.has_value()) << "kf " << id;
      EXPECT_EQ(marginal->form, meridian::PoseCov6::Form::Covariance);
      // Translation-first: the leading 3x3 block is the [m^2] translation covariance.
      const double tr = marginal->M.topLeftCorner<3, 3>().trace();
      EXPECT_TRUE(std::isfinite(tr));
      EXPECT_GT(tr, 0.0);
      (id == 5 ? trace5 : trace50) = tr;
    }
  }
  EXPECT_GT(trace50, trace5);
}

// A restart packet that lacks its IMU summary cannot form a bridge, so it is rejected
// rather than fabricated; the chain then continues from the last accepted keyframe. A
// well-formed restart bridge is exercised in test_restart_bridge.
TEST(BackendSpine, MalformedRestartRejected) {
  SynthOptions opt;
  opt.n = 5;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = makeBackend(&sink);
  for (int i = 0; i <= 2; ++i) {
    feed(*be, chain.packets[static_cast<std::size_t>(i)]);
    be->optimize();
  }

  // ImuPreintegration kind but no imu_summary / kinematics: the precondition fails.
  KeyframePacket restart = chain.packets[3];
  restart.constraint_kind = KeyframePacket::ConstraintKind::ImuPreintegration;
  EXPECT_NO_THROW(be->add_keyframe(std::move(restart)));
  be->optimize();
  // The malformed restart never enters the estimate.
  EXPECT_EQ(trajIds(be->corrected_trajectory()), (std::vector<std::uint64_t>{0, 1, 2}));

  // Packet 4 references the rejected id 3, so it too falls to the contiguity gate.
  feed(*be, chain.packets[4]);
  be->optimize();
  EXPECT_EQ(trajIds(be->corrected_trajectory()), (std::vector<std::uint64_t>{0, 1, 2}));
}

// The .g2o snapshot writes one SE3 vertex per estimated keyframe and is loadable text.
TEST(BackendSpine, G2oSnapshot) {
  SynthOptions opt;
  opt.n = 12;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = makeBackend(&sink);
  for (const KeyframePacket& p : chain.packets) {
    feed(*be, p);
    be->optimize();
  }

  const std::string path = "/tmp/meridian_spine_snapshot.g2o";
  be->write_g2o(path);

  std::ifstream f(path);
  ASSERT_TRUE(f.good());
  int vertices = 0;
  int edges = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VERTEX_SE3", 0) == 0) ++vertices;
    if (line.rfind("EDGE_SE3", 0) == 0) ++edges;
  }
  EXPECT_EQ(vertices, opt.n);
  EXPECT_EQ(edges, opt.n - 1);  // one between-edge per consecutive pair
}
