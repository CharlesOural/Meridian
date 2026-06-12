#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

#include <cstdint>
#include <memory>

#include "isam2_backend.hpp"
#include "keys.hpp"
#include "synthetic.hpp"

namespace meridian::backend {
namespace {

// A pair of fresh keys joined by a single between-factor and nothing else: the subsystem
// has a 6-DoF null space, so folding it makes the iSAM2 update throw an indeterminate
// linear system on both the first attempt and the QR retry, driving the FM-3b path.
void stage_indeterminate(Isam2BackEnd& backend) {
  gtsam::NonlinearFactorGraph g;
  gtsam::Values v;
  // High-id pose keys in the 'x' namespace so the per-type relinearize thresholds apply;
  // they connect to nothing else, leaving a 6-DoF null space.
  const gtsam::Key a = keyX(9'000'000);
  const gtsam::Key b = keyX(9'000'001);
  g.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(a, b, gtsam::Pose3(),
                                                       gtsam::noiseModel::Isotropic::Sigma(6, 0.1));
  v.insert(a, gtsam::Pose3());
  v.insert(b, gtsam::Pose3());
  backend.stage_for_test(std::move(g), std::move(v));
}

// The packet is value-copyable (cloud/image are shared_ptr handles), so a copy lets a test
// feed the same keyframe twice without consuming the source chain.
KeyframePacket clone_packet(const KeyframePacket& src) {
  return src;
}

// FM-3b: when a staged batch cannot be folded even after the QR rebuild, the back-end
// must abandon it, flag divergence, and keep running — not throw out of optimize().
TEST(BackendRecovery, AbandonsUnrecoverableBatchAndSurvives) {
  const testing::SynthChain chain = testing::make_chain(testing::SynthOptions{/*n=*/4});
  testing::CountingSink sink;
  Isam2BackEnd backend(BackendConfig{}, nullptr, &sink, /*deterministic=*/true);

  // Two good keyframes commit normally.
  backend.add_keyframe(clone_packet(chain.packets[0]));
  backend.optimize();
  backend.add_keyframe(clone_packet(chain.packets[1]));
  backend.optimize();
  ASSERT_EQ(backend.corrected_trajectory().size(), 2u);

  // Stage an unconstrained subgraph alongside a third real keyframe, then fold.
  backend.add_keyframe(clone_packet(chain.packets[2]));  // id 2, rel_to 1
  stage_indeterminate(backend);
  EXPECT_NO_THROW(backend.optimize());

  EXPECT_TRUE(backend.diagnostics().last_optimize_diverged);
  EXPECT_EQ(backend.diagnostics().fallback_count, 1u);
  // The abandoned keyframe must not survive in the estimate.
  EXPECT_EQ(backend.corrected_trajectory().size(), 2u);
}

// F1 regression: after the rollback, the dropped keyframe's id is gone, so the naturally
// next packet (which chains to it) is cleanly contiguity-dropped rather than staged against
// a variable that was never linearized (which would throw on the following update).
TEST(BackendRecovery, RolledBackChainDropsCleanlyThenRecovers) {
  const testing::SynthChain chain = testing::make_chain(testing::SynthOptions{/*n=*/5});
  testing::CountingSink sink;
  Isam2BackEnd backend(BackendConfig{}, nullptr, &sink, /*deterministic=*/true);

  backend.add_keyframe(clone_packet(chain.packets[0]));
  backend.optimize();
  backend.add_keyframe(clone_packet(chain.packets[1]));
  backend.optimize();

  // Keyframe 2 is abandoned by FM-3b.
  backend.add_keyframe(clone_packet(chain.packets[2]));
  stage_indeterminate(backend);
  backend.optimize();
  ASSERT_EQ(backend.corrected_trajectory().size(), 2u);

  // Packet 3 chains to keyframe 2 (the dropped one): contiguity must reject it, and the
  // subsequent optimize must not throw on a dangling variable.
  backend.add_keyframe(clone_packet(chain.packets[3]));  // rel_to_id == 2
  EXPECT_NO_THROW(backend.optimize());
  EXPECT_GE(sink.events.count("backend/contiguity"), 1);
  EXPECT_EQ(backend.corrected_trajectory().size(), 2u);

  // A packet that chains to the surviving keyframe 1 commits again.
  KeyframePacket revive = clone_packet(chain.packets[3]);
  revive.id = 99;
  revive.rel_to_id = 1;
  backend.add_keyframe(std::move(revive));
  EXPECT_NO_THROW(backend.optimize());
  EXPECT_EQ(backend.corrected_trajectory().size(), 3u);
}

}  // namespace
}  // namespace meridian::backend
