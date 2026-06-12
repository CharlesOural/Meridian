#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "meridian/backend/ibackend.hpp"
#include "synthetic.hpp"

using meridian::BackendConfig;
using meridian::CalibrationSet;
using meridian::GraphUpdate;
using meridian::IBackEnd;
using meridian::KeyframePacket;
using meridian::LoopConstraint;
using meridian::Pose;
using meridian::StampedPose;
using meridian::backend::testing::CountingSink;
using meridian::backend::testing::make_chain;
using meridian::backend::testing::make_loop;
using meridian::backend::testing::SynthChain;
using meridian::backend::testing::SynthOptions;

namespace {

std::unique_ptr<IBackEnd> makeBackend(meridian::TelemetrySink* sink,
                                      const BackendConfig& cfg = BackendConfig{}) {
  return meridian::makeBackEnd(cfg, std::make_shared<CalibrationSet>(), sink,
                               /*deterministic=*/true);
}

void feed(IBackEnd& be, const KeyframePacket& p) {
  be.add_keyframe(KeyframePacket(p));
}

// Drives a whole chain into the back-end with one optimize() per keyframe, the deterministic
// replay cadence, leaving the estimate ready for loop injection.
std::unique_ptr<IBackEnd> drive(const SynthChain& chain, CountingSink* sink,
                                const BackendConfig& cfg = BackendConfig{}) {
  auto be = makeBackend(sink, cfg);
  for (const KeyframePacket& p : chain.packets) {
    feed(*be, p);
    be->optimize();
  }
  return be;
}

// RMS of the per-keyframe translation error against ground truth over the whole trajectory.
double traj_rms(const IBackEnd& be, const SynthChain& chain) {
  double sq = 0.0;
  std::size_t n = 0;
  for (const StampedPose& sp : be.corrected_trajectory()) {
    sq += (sp.T_map_body.t - chain.gt[static_cast<std::size_t>(sp.kf_id)].t).squaredNorm();
    ++n;
  }
  return n > 0 ? std::sqrt(sq / static_cast<double>(n)) : -1.0;
}

}  // namespace

// A single loop is a size-1 clique and is always admitted (PCM only filters mutual inconsistency
// among several loops); the in-graph count rises and the fold reports a loop closure.
TEST(BackendLoops, SingleLoopAdmitted) {
  SynthOptions opt;
  opt.n = 40;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = drive(chain, &sink);

  be->add_loop_constraint(make_loop(chain, 5, 35, 0.05, 0.01));
  const GraphUpdate up = be->optimize();

  EXPECT_TRUE(up.loop_closed);
  EXPECT_EQ(be->diagnostics().num_loops, 1u);
  EXPECT_EQ(be->diagnostics().num_loops_rejected, 0u);
}

// A low-fitness loop never reaches PCM: it is rejected at the gate and the graph is untouched.
TEST(BackendLoops, LowFitnessRejectedAtGate) {
  SynthOptions opt;
  opt.n = 20;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = drive(chain, &sink);

  LoopConstraint weak = make_loop(chain, 2, 15, 0.05, 0.01);
  weak.fitness = 0.1;  // below the default loop_min_fitness (0.5)
  be->add_loop_constraint(weak);
  be->optimize();

  EXPECT_EQ(be->diagnostics().num_loops, 0u);
  EXPECT_EQ(be->diagnostics().num_loops_rejected, 1u);
}

// Four mutually consistent (truth-anchored) loops form the max clique and are admitted together;
// two loops with large, distinct translation errors are pairwise-inconsistent with that consensus
// and are rejected in the same pass.
TEST(BackendLoops, InliersAdmittedOutliersRejected) {
  SynthOptions opt;
  opt.n = 60;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = drive(chain, &sink);

  be->add_loop_constraint(make_loop(chain, 2, 20, 0.05, 0.01));
  be->add_loop_constraint(make_loop(chain, 5, 40, 0.05, 0.01));
  be->add_loop_constraint(make_loop(chain, 10, 50, 0.05, 0.01));
  be->add_loop_constraint(make_loop(chain, 15, 55, 0.05, 0.01));
  // Outliers pass the fitness gate but carry large, distinct position errors.
  be->add_loop_constraint(
      make_loop(chain, 3, 45, 0.05, 0.01, 0.85, Eigen::Vector3d(5.0, 0.0, 0.0)));
  be->add_loop_constraint(
      make_loop(chain, 8, 52, 0.05, 0.01, 0.85, Eigen::Vector3d(0.0, -6.0, 0.0)));

  be->optimize();

  EXPECT_EQ(be->diagnostics().num_loops, 4u);
  EXPECT_EQ(be->diagnostics().num_loops_rejected, 2u);
}

// On a drifting (noisy) chain, a truth-anchored loop tying the far end back to the gauge-anchored
// first keyframe injects correct global information: the whole-trajectory RMS error must shrink.
TEST(BackendLoops, LoopCorrectsDrift) {
  SynthOptions opt;
  opt.n = 80;
  opt.noise_trans = 0.04;
  opt.noise_rot = 0.006;
  opt.seed = 11;
  const SynthChain chain = make_chain(opt);

  CountingSink sink;
  auto be = drive(chain, &sink);

  const double before = traj_rms(*be, chain);
  ASSERT_GT(before, 0.0);

  be->add_loop_constraint(make_loop(chain, 0, 79, 0.1, 0.02));
  be->optimize();

  const double after = traj_rms(*be, chain);
  ASSERT_GT(after, 0.0);
  EXPECT_EQ(be->diagnostics().num_loops, 1u);
  EXPECT_LT(after, before) << "loop did not reduce trajectory drift";
}

// The whole admit/reject sequence is deterministic: two independent runs produce a byte-identical
// corrected trajectory.
TEST(BackendLoops, DeterministicWithLoops) {
  SynthOptions opt;
  opt.n = 60;
  const SynthChain chain = make_chain(opt);

  const auto run = [&chain]() {
    CountingSink sink;
    auto be = makeBackend(&sink);
    for (const KeyframePacket& p : chain.packets) {
      be->add_keyframe(KeyframePacket(p));
      be->optimize();
    }
    be->add_loop_constraint(make_loop(chain, 2, 20, 0.05, 0.01));
    be->add_loop_constraint(make_loop(chain, 10, 50, 0.05, 0.01));
    be->add_loop_constraint(
        make_loop(chain, 8, 52, 0.05, 0.01, 0.85, Eigen::Vector3d(0.0, -6.0, 0.0)));
    be->optimize();
    return be->corrected_trajectory();
  };

  const std::vector<StampedPose> a = run();
  const std::vector<StampedPose> b = run();
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].kf_id, b[i].kf_id);
    EXPECT_EQ(a[i].T_map_body.t.x(), b[i].T_map_body.t.x()) << "kf " << a[i].kf_id;
    EXPECT_EQ(a[i].T_map_body.t.y(), b[i].T_map_body.t.y()) << "kf " << a[i].kf_id;
    EXPECT_EQ(a[i].T_map_body.t.z(), b[i].T_map_body.t.z()) << "kf " << a[i].kf_id;
    EXPECT_EQ(a[i].T_map_body.q.w(), b[i].T_map_body.q.w()) << "kf " << a[i].kf_id;
  }
}
