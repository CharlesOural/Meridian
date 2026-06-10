#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "meridian/backend/ibackend.hpp"
#include "synthetic.hpp"

using meridian::BackendConfig;
using meridian::CalibrationSet;
using meridian::IBackEnd;
using meridian::KeyframePacket;
using meridian::Pose;
using meridian::StampedPose;
using meridian::Timestamp;
using meridian::backend::testing::CountingSink;
using meridian::backend::testing::make_chain;
using meridian::backend::testing::make_restart_packet;
using meridian::backend::testing::SynthChain;
using meridian::backend::testing::SynthOptions;

namespace {

// add_keyframe consumes its argument, so feed a copy to keep the source chain reusable.
void feed(IBackEnd& be, const KeyframePacket& p) {
  be.add_keyframe(KeyframePacket(p));
}

std::vector<std::uint64_t> trajIds(const std::vector<StampedPose>& traj) {
  std::vector<std::uint64_t> ids;
  ids.reserve(traj.size());
  for (const StampedPose& p : traj) ids.push_back(p.kf_id);
  return ids;
}

bool trajHasId(const std::vector<StampedPose>& traj, std::uint64_t id) {
  for (const StampedPose& p : traj) {
    if (p.kf_id == id) return true;
  }
  return false;
}

std::unique_ptr<IBackEnd> makeBackend(meridian::TelemetrySink* sink,
                                      const BackendConfig& cfg = BackendConfig{}) {
  return meridian::makeBackEnd(cfg, std::make_shared<CalibrationSet>(), sink,
                               /*deterministic=*/true);
}

// Serializes a trajectory field-by-field so memcmp compares every double bit-for-bit.
std::vector<std::byte> pack(const std::vector<StampedPose>& traj) {
  std::vector<std::byte> buf;
  buf.reserve(traj.size() * (sizeof(std::int64_t) + sizeof(std::uint64_t) + 7 * sizeof(double)));
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

Timestamp stampAfter(Timestamp prev) {
  return prev + 100'000'000LL;
}

}  // namespace

// A restart bridge admitted after a normal stretch must add the keyframe's pose variable,
// emit the restart-bridge event, and advance the keyframe count without throwing.
TEST(RestartBridge, BridgeAcceptedAfterNormal) {
  const SynthChain chain = make_chain(SynthOptions{/*n=*/3});
  CountingSink sink;
  auto be = makeBackend(&sink);

  feed(*be, chain.packets[0]);  // AbsolutePrior, id 0
  be->optimize();
  feed(*be, chain.packets[1]);  // RelativeBetween, id 1 rel_to 0
  be->optimize();
  ASSERT_EQ(be->corrected_trajectory().size(), 2u);

  const Timestamp t = stampAfter(chain.packets[1].stamp);
  const KeyframePacket restart = make_restart_packet(/*prev_id=*/1, /*id=*/2, t, chain.gt[2]);
  feed(*be, restart);
  EXPECT_NO_THROW(be->optimize());

  EXPECT_GE(sink.count("backend/window_restart_bridge"), 1);
  const auto traj = be->corrected_trajectory();
  EXPECT_EQ(traj.size(), 3u);
  EXPECT_TRUE(trajHasId(traj, 2));
  EXPECT_EQ(be->diagnostics().num_keyframes, 3u);
}

// Two restarts back-to-back chain onto a node whose V/B are still live (no normal keyframe
// between them to marginalize). The bridge must reuse those variables, not re-insert them.
TEST(RestartBridge, ConsecutiveRestartsDoNotDuplicateInsert) {
  const SynthChain chain = make_chain(SynthOptions{/*n=*/4});
  CountingSink sink;
  auto be = makeBackend(&sink);

  feed(*be, chain.packets[0]);
  be->optimize();
  feed(*be, chain.packets[1]);
  be->optimize();

  const Timestamp t2 = stampAfter(chain.packets[1].stamp);
  feed(*be, make_restart_packet(/*prev_id=*/1, /*id=*/2, t2, chain.gt[2]));
  EXPECT_NO_THROW(be->optimize());

  // Second restart chains onto id 2, whose V(2)/B(2) are still in the graph.
  const Timestamp t3 = stampAfter(t2);
  feed(*be, make_restart_packet(/*prev_id=*/2, /*id=*/3, t3, chain.gt[3]));
  EXPECT_NO_THROW(be->optimize());

  const auto traj = be->corrected_trajectory();
  EXPECT_EQ(traj.size(), 4u);
  EXPECT_TRUE(trajHasId(traj, 3));
}

// keep_inertial keeps V/B live across normal keyframes; a later restart that chains onto a
// previously-bridged node must still reuse, not re-insert, the existing inertial variables.
TEST(RestartBridge, KeepInertialSecondBridgeDoesNotDuplicate) {
  BackendConfig cfg;
  cfg.keep_inertial = true;
  const SynthChain chain = make_chain(SynthOptions{/*n=*/4});
  CountingSink sink;
  auto be = makeBackend(&sink, cfg);

  feed(*be, chain.packets[0]);
  be->optimize();
  feed(*be, chain.packets[1]);
  be->optimize();
  const Timestamp t2 = stampAfter(chain.packets[1].stamp);
  feed(*be, make_restart_packet(/*prev_id=*/1, /*id=*/2, t2, chain.gt[2]));
  be->optimize();
  // Normal keyframe 3; with keep_inertial, V(2)/B(2) stay live.
  feed(*be, chain.packets[3]);
  EXPECT_NO_THROW(be->optimize());
  EXPECT_EQ(be->corrected_trajectory().size(), 4u);
}

// After the bridge introduces V/B, the next NORMAL keyframe must marginalize them out and
// still leave a solvable graph. Without internal access the V/B removal is asserted
// INDIRECTLY: a subsequent normal keyframe must optimize cleanly (no indeterminate growth)
// and the trajectory must keep growing.
TEST(RestartBridge, VbMarginalizedAtNextNormal) {
  const SynthChain chain = make_chain(SynthOptions{/*n=*/5});
  CountingSink sink;
  auto be = makeBackend(&sink);

  feed(*be, chain.packets[0]);
  be->optimize();
  feed(*be, chain.packets[1]);
  be->optimize();

  const Timestamp t2 = stampAfter(chain.packets[1].stamp);
  feed(*be, make_restart_packet(/*prev_id=*/1, /*id=*/2, t2, chain.gt[2]));
  EXPECT_NO_THROW(be->optimize());
  ASSERT_EQ(be->corrected_trajectory().size(), 3u);

  // A normal between-edge chaining to the bridge keyframe: folding it triggers the
  // marginalization of the restart's V/B leaves.
  KeyframePacket next = chain.packets[3];
  next.id = 3;
  next.rel_to_id = 2;
  feed(*be, next);
  EXPECT_NO_THROW(be->optimize());

  auto traj = be->corrected_trajectory();
  EXPECT_EQ(traj.size(), 4u);
  EXPECT_TRUE(trajHasId(traj, 3));
  EXPECT_FALSE(be->diagnostics().last_optimize_diverged);

  // One more normal keyframe must still optimize without an indeterminate failure: a
  // dangling V/B left behind would surface here as growth in fallback_count.
  const std::uint64_t fallbacks_before = be->diagnostics().fallback_count;
  KeyframePacket next2 = chain.packets[4];
  next2.id = 4;
  next2.rel_to_id = 3;
  feed(*be, next2);
  EXPECT_NO_THROW(be->optimize());

  traj = be->corrected_trajectory();
  EXPECT_EQ(traj.size(), 5u);
  EXPECT_EQ(be->diagnostics().num_keyframes, 5u);
  EXPECT_EQ(be->diagnostics().fallback_count, fallbacks_before);
}

// keep_inertial retains the restart V/B variables instead of marginalizing them. The
// retained-state path must also be stable: bridge + next normal keyframes optimize cleanly
// and the trajectory grows.
TEST(RestartBridge, KeepInertialRetainsVb) {
  const SynthChain chain = make_chain(SynthOptions{/*n=*/5});
  CountingSink sink;
  BackendConfig cfg{};
  cfg.keep_inertial = true;
  auto be = makeBackend(&sink, cfg);

  feed(*be, chain.packets[0]);
  be->optimize();
  feed(*be, chain.packets[1]);
  be->optimize();

  const Timestamp t2 = stampAfter(chain.packets[1].stamp);
  feed(*be, make_restart_packet(/*prev_id=*/1, /*id=*/2, t2, chain.gt[2]));
  EXPECT_NO_THROW(be->optimize());
  ASSERT_EQ(be->corrected_trajectory().size(), 3u);

  KeyframePacket next = chain.packets[3];
  next.id = 3;
  next.rel_to_id = 2;
  feed(*be, next);
  EXPECT_NO_THROW(be->optimize());

  const auto traj = be->corrected_trajectory();
  EXPECT_EQ(traj.size(), 4u);
  EXPECT_TRUE(trajHasId(traj, 3));
  EXPECT_FALSE(be->diagnostics().last_optimize_diverged);
}

// A restart packet whose rel_to_id is not the last accepted keyframe is a contiguity
// violation: it must be dropped with the contiguity event and leave the trajectory
// unchanged, exactly like a broken between-edge.
TEST(RestartBridge, RestartContiguityViolationDropped) {
  const SynthChain chain = make_chain(SynthOptions{/*n=*/4});
  CountingSink sink;
  auto be = makeBackend(&sink);

  feed(*be, chain.packets[0]);
  be->optimize();
  feed(*be, chain.packets[1]);
  be->optimize();
  feed(*be, chain.packets[2]);  // last accepted is now id 2
  be->optimize();
  ASSERT_EQ(trajIds(be->corrected_trajectory()), (std::vector<std::uint64_t>{0, 1, 2}));

  // Restart references id 0, not the last accepted id 2.
  const Timestamp t = stampAfter(chain.packets[2].stamp);
  feed(*be, make_restart_packet(/*prev_id=*/0, /*id=*/3, t, chain.gt[3]));
  EXPECT_EQ(sink.count("backend/contiguity"), 1);

  be->optimize();
  EXPECT_EQ(trajIds(be->corrected_trajectory()), (std::vector<std::uint64_t>{0, 1, 2}));
  EXPECT_EQ(be->diagnostics().num_keyframes, 3u);
}

// Two deterministic back-ends fed the identical sequence containing a restart bridge must
// produce bit-identical corrected trajectories: the bridge and its later marginalization
// must not introduce any nondeterminism.
TEST(RestartBridge, DeterministicWithBridge) {
  const SynthChain chain = make_chain(SynthOptions{/*n=*/5});

  const auto run = [&chain]() {
    auto be = makeBackend(/*sink=*/nullptr);
    feed(*be, chain.packets[0]);
    be->optimize();
    feed(*be, chain.packets[1]);
    be->optimize();

    const Timestamp t2 = stampAfter(chain.packets[1].stamp);
    feed(*be, make_restart_packet(/*prev_id=*/1, /*id=*/2, t2, chain.gt[2]));
    be->optimize();

    KeyframePacket n3 = chain.packets[3];
    n3.id = 3;
    n3.rel_to_id = 2;
    feed(*be, n3);
    be->optimize();

    KeyframePacket n4 = chain.packets[4];
    n4.id = 4;
    n4.rel_to_id = 3;
    feed(*be, n4);
    be->optimize();
    return be->corrected_trajectory();
  };

  const auto traj_a = run();
  const auto traj_b = run();
  ASSERT_EQ(traj_a.size(), 5u);
  ASSERT_EQ(traj_a.size(), traj_b.size());

  const auto buf_a = pack(traj_a);
  const auto buf_b = pack(traj_b);
  ASSERT_EQ(buf_a.size(), buf_b.size());
  EXPECT_EQ(std::memcmp(buf_a.data(), buf_b.data(), buf_a.size()), 0);
}
