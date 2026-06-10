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
using meridian::KeyframePacket;
using meridian::StampedPose;
using meridian::backend::testing::make_chain;
using meridian::backend::testing::SynthChain;
using meridian::backend::testing::SynthOptions;

namespace {

// Serializes a trajectory field-by-field so memcmp compares every double bit-for-bit
// (stricter than operator==, which would let -0.0 alias +0.0).
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

std::uint64_t bits(double v) {
  std::uint64_t u = 0;
  std::memcpy(&u, &v, sizeof(u));
  return u;
}

}  // namespace

// Two independent back-ends fed the identical noisy chain (optimize after every add,
// the replay cadence) must produce bit-identical corrected trajectories and chi2.
TEST(BackendDeterminism, NoisyChainBitIdentical) {
  SynthOptions opt;
  opt.n = 80;
  opt.noise_trans = 0.02;
  opt.noise_rot = 0.005;
  opt.seed = 7;
  const SynthChain chain = make_chain(opt);

  const BackendConfig cfg{};
  const auto run = [&chain, &cfg]() {
    auto be = meridian::makeBackEnd(cfg, std::make_shared<CalibrationSet>(),
                                    /*telemetry=*/nullptr, /*deterministic=*/true);
    for (const KeyframePacket& p : chain.packets) {
      be->add_keyframe(KeyframePacket(p));
      be->optimize();
    }
    return be;
  };

  const auto a = run();
  const auto b = run();

  const auto traj_a = a->corrected_trajectory();
  const auto traj_b = b->corrected_trajectory();
  ASSERT_EQ(traj_a.size(), static_cast<std::size_t>(opt.n));
  ASSERT_EQ(traj_a.size(), traj_b.size());

  const auto buf_a = pack(traj_a);
  const auto buf_b = pack(traj_b);
  ASSERT_EQ(buf_a.size(), buf_b.size());
  EXPECT_EQ(std::memcmp(buf_a.data(), buf_b.data(), buf_a.size()), 0);

  EXPECT_EQ(bits(a->diagnostics().chi2), bits(b->diagnostics().chi2));
}
