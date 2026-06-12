#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "pcm.hpp"

using meridian::LoopConstraint;
using meridian::Pose;
using meridian::PoseCov6;
using meridian::backend::Pcm;
using meridian::backend::PcmDecision;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

namespace {

// chi^2_{6, 0.99}: the squared-Mahalanobis gate used by the consistency test.
constexpr double kChi2_6_099 = 16.8119;

Pose poseFrom(const Sophus::SE3d& T) {
  return Pose{T.unit_quaternion(), T.translation()};
}

PoseCov6 covOf(const Mat6& m) {
  PoseCov6 c;
  c.form = PoseCov6::Form::Covariance;
  c.M = m;
  return c;
}

Mat6 diagCov(double trans_var, double rot_var) {
  Vec6 d;
  d << trans_var, trans_var, trans_var, rot_var, rot_var, rot_var;
  return d.asDiagonal();
}

LoopConstraint makeLoop(std::uint64_t from, std::uint64_t to, const Sophus::SE3d& z,
                        const Mat6& cov, double fitness = 0.9) {
  LoopConstraint lc;
  lc.from_id = from;
  lc.to_id = to;
  lc.T_from_to = poseFrom(z);
  lc.cov = covOf(cov);
  lc.fitness = fitness;
  return lc;
}

// Draw a right-perturbed pose T * exp(L * n), n ~ N(0,I), so the sample has right-cov L L^T.
Sophus::SE3d sampleRight(const Sophus::SE3d& T, const Mat6& L, std::mt19937& rng) {
  std::normal_distribution<double> g(0.0, 1.0);
  Vec6 n;
  for (int i = 0; i < 6; ++i) n(i) = g(rng);
  return T * Sophus::SE3d::exp(L * n);
}

// Empirical right-cov of a SE(3) sample set about the mean Tbar: cov of Log(Tbar^{-1} Ti).
Mat6 empiricalRightCov(const std::vector<Sophus::SE3d>& samples, const Sophus::SE3d& Tbar) {
  Mat6 acc = Mat6::Zero();
  for (const auto& Ti : samples) {
    const Vec6 d = (Tbar.inverse() * Ti).log();
    acc += d * d.transpose();
  }
  return acc / static_cast<double>(samples.size());
}

// Pure-translation SE(3); avoids any dependence on a Sophus translation-factory helper.
Sophus::SE3d transSe3(const Eigen::Vector3d& v) {
  return Sophus::SE3d(Eigen::Quaterniond::Identity(), v);
}

Sophus::SE3d randomSe3(std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Vec6 xi;
  for (int i = 0; i < 6; ++i) xi(i) = u(rng);
  return Sophus::SE3d::exp(xi);
}

}  // namespace

// relative_cov is private; validate the exact adjoint-transport claim it relies on. For a
// relative T = Xlo^{-1} Xhi with right-cov C, the inverse T^{-1} has right-cov Ad_T C Ad_T^T,
// because (T exp(d))^{-1} = T^{-1} exp(-Ad_T d). Monte-Carlo both directions.
TEST(Pcm, RelativeCovForwardAndInverseAdjointTransport) {
  std::mt19937 rng(20240117u);
  const Sophus::SE3d Xlo = randomSe3(rng);
  const Sophus::SE3d Xhi = randomSe3(rng);
  const Sophus::SE3d T = Xlo.inverse() * Xhi;  // older->newer relative

  const Mat6 C = diagCov(0.02, 0.01);
  const Mat6 L = C.llt().matrixL();

  constexpr int kN = 40000;
  std::vector<Sophus::SE3d> fwd;
  std::vector<Sophus::SE3d> inv;
  fwd.reserve(kN);
  inv.reserve(kN);
  for (int i = 0; i < kN; ++i) {
    // Perturb the newer pose only: Xhi -> Xhi exp(L n). Then Xlo^{-1} (Xhi exp(L n)) =
    // T exp(L n) has right-cov C in hi's tangent, matching chain_cov(lo,hi).
    const Sophus::SE3d Xhi_s = sampleRight(Xhi, L, rng);
    const Sophus::SE3d rel = Xlo.inverse() * Xhi_s;
    fwd.push_back(rel);
    inv.push_back(rel.inverse());
  }

  const Mat6 emp_fwd = empiricalRightCov(fwd, T);
  EXPECT_LT((emp_fwd - C).norm() / C.norm(), 0.15);

  const Mat6 ad = T.Adj();
  const Mat6 expected_inv = ad * C * ad.transpose();
  const Mat6 emp_inv = empiricalRightCov(inv, T.inverse());
  EXPECT_LT((emp_inv - expected_inv).norm() / expected_inv.norm(), 0.15);
}

// A consistent two-loop cycle: d^2 of pair_distance_sq follows chi^2_6 (mean ~ 6) when the loops'
// measurements carry exactly the modelled covariance. Validates Sigma_eps propagation end-to-end.
TEST(Pcm, PairDistanceMonteCarloMeanSixWhenConsistent) {
  std::mt19937 rng(771u);

  // Four anchor poses p,q,r,s, ids 0,1,2,3, drawn once and fixed (the "estimate").
  std::map<std::uint64_t, Pose> poses;
  std::vector<Sophus::SE3d> X(4);
  for (int i = 0; i < 4; ++i) X[i] = randomSe3(rng);
  for (std::uint64_t i = 0; i < 4; ++i) poses[i] = poseFrom(X[i]);

  // Loop 1: p(0)->q(1). Loop 2: r(2)->s(3). The true measurements close the cycle exactly:
  // Z1 = Xp^{-1} Xq, Z2 = Xr^{-1} Xs. With B = Xp^{-1} Xr, D = Xs^{-1} Xq the product
  // Z1^{-1} B Z2 D = Xq^{-1} Xp Xp^{-1} Xr Xr^{-1} Xs Xs^{-1} Xq = I -> epsilon = 0.
  const Sophus::SE3d Z1_true = X[0].inverse() * X[1];
  const Sophus::SE3d Z2_true = X[2].inverse() * X[3];

  const Mat6 loop_cov = diagCov(4e-4, 4e-4);  // covariance of each loop measurement
  const Mat6 chain_c = diagCov(2e-4, 2e-4);   // chain cov for B and D relatives
  const Mat6 Lz = loop_cov.llt().matrixL();
  const Mat6 Lc = chain_c.llt().matrixL();

  // chain_cov(a,b), a<b, is constant chain_c for all probed pairs (B uses (0,2), D uses (1,3)).
  auto chain_cov = [&](std::uint64_t, std::uint64_t) -> std::optional<Mat6> { return chain_c; };

  // pose() draws fresh perturbed B/D relatives each sample by perturbing the newer endpoint, so
  // the relatives carry chain_c. The loop measurements are perturbed independently below.
  constexpr int kN = 30000;
  double sum = 0.0;
  int counted = 0;
  for (int i = 0; i < kN; ++i) {
    // Sample noisy loop measurements about the true (cycle-closing) values.
    Vec6 nz1, nz2;
    std::normal_distribution<double> g(0.0, 1.0);
    for (int k = 0; k < 6; ++k) {
      nz1(k) = g(rng);
      nz2(k) = g(rng);
    }
    const Sophus::SE3d Z1 = Z1_true * Sophus::SE3d::exp(Lz * nz1);
    const Sophus::SE3d Z2 = Z2_true * Sophus::SE3d::exp(Lz * nz2);

    // chain_cov(lo,hi) is the right-cov of X_lo^{-1} X_hi obtained by perturbing the NEWER (hi)
    // endpoint. B is relative 0->2 (lo=0,hi=2): perturb r=2 so B picks up chain_c directly. D is
    // relative 3->1 (lo=1,hi=3): perturb s=3 so X1^{-1}X3 picks up chain_c and D=(X1^{-1}X3)^{-1}
    // picks up the transported Ad_T chain_c Ad_T^T that relative_cov(3,1) reconstructs.
    std::map<std::uint64_t, Pose> sp = poses;
    const Sophus::SE3d Xr_s = sampleRight(X[2], Lc, rng);  // perturbs B's relative (newer = r)
    const Sophus::SE3d Xs_s = sampleRight(X[3], Lc, rng);  // perturbs D's relative (newer = s)
    sp[2] = poseFrom(Xr_s);
    sp[3] = poseFrom(Xs_s);
    auto pose = [&](std::uint64_t id) -> std::optional<Pose> { return sp.at(id); };

    Pcm pcm(kChi2_6_099, 16);
    const std::size_t h1 = pcm.add(makeLoop(0, 1, Z1, loop_cov));
    const std::size_t h2 = pcm.add(makeLoop(2, 3, Z2, loop_cov));
    const std::optional<double> d2 = pcm.pair_distance_sq(h1, h2, pose, chain_cov);
    ASSERT_TRUE(d2.has_value());
    sum += *d2;
    ++counted;
  }
  const double mean = sum / counted;
  // chi^2_6 has mean 6; first-order propagation + finite samples leave a modest band.
  EXPECT_NEAR(mean, 6.0, 0.6);

  // Corrupting one loop by a large translation drives d^2 far above the gate.
  auto pose0 = [&](std::uint64_t id) -> std::optional<Pose> { return poses.at(id); };
  Pcm pcm(kChi2_6_099, 16);
  const Sophus::SE3d Z1_bad = Z1_true * transSe3(Eigen::Vector3d(5.0, 0.0, 0.0));
  const std::size_t b1 = pcm.add(makeLoop(0, 1, Z1_bad, loop_cov));
  const std::size_t b2 = pcm.add(makeLoop(2, 3, Z2_true, loop_cov));
  const std::optional<double> dbad = pcm.pair_distance_sq(b1, b2, pose0, chain_cov);
  ASSERT_TRUE(dbad.has_value());
  EXPECT_GT(*dbad, 10.0 * kChi2_6_099);
}

namespace {

// A graph of N keyframes laid out so each loop's four endpoints have known poses, plus a constant
// diagonal chain cov for every pair. Loops here connect disjoint pairs; consistency is decided by
// whether the loop measurement closes the cycle with its partners.
struct Scene {
  std::map<std::uint64_t, Pose> poses;
  Mat6 chain_c = diagCov(1e-4, 1e-4);
  Mat6 loop_cov = diagCov(1e-3, 1e-3);

  Pcm::PoseFn pose_fn() const {
    return [this](std::uint64_t id) -> std::optional<Pose> {
      auto it = poses.find(id);
      if (it == poses.end()) return std::nullopt;
      return it->second;
    };
  }
  Pcm::ChainCovFn chain_fn() const {
    return [this](std::uint64_t, std::uint64_t) -> std::optional<Mat6> { return chain_c; };
  }
};

// A loop p->q whose true measurement is Xp^{-1} Xq (consistent), optionally corrupted.
LoopConstraint trueLoop(const Scene& sc, std::uint64_t p, std::uint64_t q,
                        const Sophus::SE3d& corrupt = Sophus::SE3d()) {
  const Sophus::SE3d Xp(sc.poses.at(p).q, sc.poses.at(p).t);
  const Sophus::SE3d Xq(sc.poses.at(q).q, sc.poses.at(q).t);
  return makeLoop(p, q, Xp.inverse() * Xq * corrupt, sc.loop_cov);
}

}  // namespace

// Four mutually consistent inliers admit; two corrupted outliers reject; nothing evicts.
TEST(Pcm, UpdateAdmitsInliersRejectsOutliers) {
  std::mt19937 rng(9090u);
  Scene sc;
  // Eight keyframes 0..7, random absolute poses.
  for (std::uint64_t i = 0; i < 8; ++i) sc.poses[i] = poseFrom(randomSe3(rng));

  Pcm pcm(kChi2_6_099, 16);
  // Four inliers on disjoint endpoint pairs; their cycle-consistency holds exactly.
  const std::size_t i0 = pcm.add(trueLoop(sc, 0, 1));
  const std::size_t i1 = pcm.add(trueLoop(sc, 2, 3));
  const std::size_t i2 = pcm.add(trueLoop(sc, 4, 5));
  const std::size_t i3 = pcm.add(trueLoop(sc, 6, 7));
  // Two outliers: same endpoints as an inlier-style pair but grossly corrupted translation.
  const Sophus::SE3d big = transSe3(Eigen::Vector3d(20.0, -15.0, 8.0));
  const std::size_t o0 = pcm.add(trueLoop(sc, 1, 2, big));
  const std::size_t o1 = pcm.add(trueLoop(sc, 3, 4, big));

  const PcmDecision d = pcm.update(sc.pose_fn(), sc.chain_fn());

  const std::vector<std::size_t> want_admit{i0, i1, i2, i3};
  std::vector<std::size_t> got_admit = d.to_admit;
  std::sort(got_admit.begin(), got_admit.end());
  EXPECT_EQ(got_admit, want_admit);

  EXPECT_TRUE(d.to_evict.empty());

  EXPECT_NE(std::find(d.to_reject.begin(), d.to_reject.end(), o0), d.to_reject.end());
  EXPECT_NE(std::find(d.to_reject.begin(), d.to_reject.end(), o1), d.to_reject.end());
}

// Two consistent loops that share an endpoint keyframe form a well-defined consistency cycle
// (the shared segment is identity with zero covariance), so they are testable and co-admit
// rather than each being treated as an isolated singleton.
TEST(Pcm, SharedEndpointLoopsAreTestable) {
  std::mt19937 rng(31337u);
  Scene sc;
  for (std::uint64_t i = 0; i < 3; ++i) sc.poses[i] = poseFrom(randomSe3(rng));

  Pcm pcm(kChi2_6_099, 16);
  // Both loops anchor at keyframe 0 (shared from_id): 0->1 and 0->2, each truth-consistent.
  const std::size_t a = pcm.add(trueLoop(sc, 0, 1));
  const std::size_t b = pcm.add(trueLoop(sc, 0, 2));

  const auto d2 = pcm.pair_distance_sq(a, b, sc.pose_fn(), sc.chain_fn());
  ASSERT_TRUE(d2.has_value());  // the shared endpoint no longer makes the pair untestable
  EXPECT_LT(*d2, kChi2_6_099);  // both truth-consistent -> the cycle closes

  const PcmDecision dec = pcm.update(sc.pose_fn(), sc.chain_fn());
  std::vector<std::size_t> got = dec.to_admit;
  std::sort(got.begin(), got.end());
  EXPECT_EQ(got, (std::vector<std::size_t>{a, b}));
}

// After admitting a consensus clique, a fresh consistent loop enlarges it and is admitted.
TEST(Pcm, UpdateAdmitsLaterEnlargingInlier) {
  std::mt19937 rng(313u);
  Scene sc;
  for (std::uint64_t i = 0; i < 10; ++i) sc.poses[i] = poseFrom(randomSe3(rng));

  Pcm pcm(kChi2_6_099, 16);
  const std::size_t i0 = pcm.add(trueLoop(sc, 0, 1));
  const std::size_t i1 = pcm.add(trueLoop(sc, 2, 3));
  const std::size_t i2 = pcm.add(trueLoop(sc, 4, 5));

  PcmDecision d1 = pcm.update(sc.pose_fn(), sc.chain_fn());
  std::sort(d1.to_admit.begin(), d1.to_admit.end());
  const std::vector<std::size_t> want1{i0, i1, i2};
  EXPECT_EQ(d1.to_admit, want1);
  for (std::size_t h : d1.to_admit) pcm.mark_admitted(h);
  EXPECT_TRUE(pcm.in_graph(i0));

  // A new consistent loop joins the consensus and is the only admit; nothing evicts.
  const std::size_t i3 = pcm.add(trueLoop(sc, 6, 7));
  const PcmDecision d2 = pcm.update(sc.pose_fn(), sc.chain_fn());
  EXPECT_EQ(d2.to_admit, std::vector<std::size_t>{i3});
  EXPECT_TRUE(d2.to_evict.empty());
}

// A larger, mutually consistent clique displaces an earlier admitted loop that no longer agrees
// with the new majority -> the displaced in-graph loop is evicted while the new inliers admit.
TEST(Pcm, UpdateEvictsDisplacedLoop) {
  std::mt19937 rng(5151u);
  Scene sc;
  for (std::uint64_t i = 0; i < 12; ++i) sc.poses[i] = poseFrom(randomSe3(rng));

  Pcm pcm(kChi2_6_099, 16);
  // A prior admit left two loops in the graph: a0 (a true inlier) and `odd`, whose measurement is
  // grossly corrupted so it is inconsistent with everything else.
  const std::size_t a0 = pcm.add(trueLoop(sc, 0, 1));
  const Sophus::SE3d big = transSe3(Eigen::Vector3d(30.0, 10.0, -5.0));
  const std::size_t odd = pcm.add(trueLoop(sc, 2, 3, big));
  pcm.mark_admitted(a0);
  pcm.mark_admitted(odd);

  // Four fresh inliers arrive. Together with a0 they form a consistent clique of five that
  // excludes `odd`.
  const std::size_t n0 = pcm.add(trueLoop(sc, 4, 5));
  const std::size_t n1 = pcm.add(trueLoop(sc, 6, 7));
  const std::size_t n2 = pcm.add(trueLoop(sc, 8, 9));
  const std::size_t n3 = pcm.add(trueLoop(sc, 10, 11));

  const PcmDecision d = pcm.update(sc.pose_fn(), sc.chain_fn());

  // `odd` is in-graph but outside the new clique -> evicted. a0 stays (in-graph and in clique).
  EXPECT_NE(std::find(d.to_evict.begin(), d.to_evict.end(), odd), d.to_evict.end());
  EXPECT_EQ(std::find(d.to_evict.begin(), d.to_evict.end(), a0), d.to_evict.end());
  // The four fresh inliers are admitted.
  for (std::size_t h : {n0, n1, n2, n3}) {
    EXPECT_NE(std::find(d.to_admit.begin(), d.to_admit.end(), h), d.to_admit.end());
  }
}

// A loop with a missing endpoint pose is neither admitted nor rejected; it stays pending.
TEST(Pcm, UpdateLeavesUnavailableLoopPending) {
  std::mt19937 rng(42u);
  Scene sc;
  for (std::uint64_t i = 0; i < 8; ++i) sc.poses[i] = poseFrom(randomSe3(rng));

  Pcm pcm(kChi2_6_099, 16);
  const std::size_t i0 = pcm.add(trueLoop(sc, 0, 1));
  const std::size_t i1 = pcm.add(trueLoop(sc, 2, 3));
  const std::size_t i2 = pcm.add(trueLoop(sc, 4, 5));
  // A loop referencing keyframe 99, which has no pose.
  const Sophus::SE3d Xp(sc.poses.at(6).q, sc.poses.at(6).t);
  const std::size_t missing = pcm.add(makeLoop(6, 99, Xp.inverse() * Xp, sc.loop_cov));

  const PcmDecision d = pcm.update(sc.pose_fn(), sc.chain_fn());

  // The three inliers admit; the missing-endpoint loop is neither admitted nor rejected.
  EXPECT_EQ(std::find(d.to_admit.begin(), d.to_admit.end(), missing), d.to_admit.end());
  EXPECT_EQ(std::find(d.to_reject.begin(), d.to_reject.end(), missing), d.to_reject.end());
  EXPECT_EQ(pcm.pending_count(), std::size_t{4});
  EXPECT_FALSE(pcm.in_graph(missing));

  std::vector<std::size_t> got_admit = d.to_admit;
  std::sort(got_admit.begin(), got_admit.end());
  const std::vector<std::size_t> want_admit{i0, i1, i2};
  EXPECT_EQ(got_admit, want_admit);
}

// pair_distance_sq is nullopt when any endpoint pose or chain cov is unavailable.
TEST(Pcm, PairDistanceNulloptOnMissingInputs) {
  std::mt19937 rng(7u);
  Scene sc;
  for (std::uint64_t i = 0; i < 4; ++i) sc.poses[i] = poseFrom(randomSe3(rng));

  Pcm pcm(kChi2_6_099, 16);
  const std::size_t h0 = pcm.add(trueLoop(sc, 0, 1));
  const std::size_t h1 = pcm.add(trueLoop(sc, 2, 3));

  // Missing pose for id 2.
  auto pose_missing = [&](std::uint64_t id) -> std::optional<Pose> {
    if (id == 2) return std::nullopt;
    return sc.poses.at(id);
  };
  EXPECT_FALSE(pcm.pair_distance_sq(h0, h1, pose_missing, sc.chain_fn()).has_value());

  // Missing chain cov.
  auto chain_missing = [&](std::uint64_t, std::uint64_t) -> std::optional<Mat6> {
    return std::nullopt;
  };
  EXPECT_FALSE(pcm.pair_distance_sq(h0, h1, sc.pose_fn(), chain_missing).has_value());

  // All present -> a value (a small, consistent distance).
  const std::optional<double> ok = pcm.pair_distance_sq(h0, h1, sc.pose_fn(), sc.chain_fn());
  ASSERT_TRUE(ok.has_value());
  EXPECT_LT(*ok, kChi2_6_099);
}

// State bookkeeping: add/admit/evict/reject and the pending counter.
TEST(Pcm, StateTransitionsAndPendingCount) {
  Scene sc;
  std::mt19937 rng(1u);
  for (std::uint64_t i = 0; i < 4; ++i) sc.poses[i] = poseFrom(randomSe3(rng));

  Pcm pcm(kChi2_6_099, 16);
  const std::size_t h0 = pcm.add(trueLoop(sc, 0, 1));
  const std::size_t h1 = pcm.add(trueLoop(sc, 2, 3));
  EXPECT_EQ(pcm.pending_count(), std::size_t{2});
  EXPECT_FALSE(pcm.in_graph(h0));

  pcm.mark_admitted(h0);
  EXPECT_TRUE(pcm.in_graph(h0));
  EXPECT_EQ(pcm.pending_count(), std::size_t{1});

  pcm.mark_evicted(h0);  // back to pending
  EXPECT_FALSE(pcm.in_graph(h0));
  EXPECT_EQ(pcm.pending_count(), std::size_t{2});

  pcm.mark_rejected(h1);
  EXPECT_EQ(pcm.pending_count(), std::size_t{1});
  EXPECT_EQ(pcm.at(h1).from_id, std::uint64_t{2});
}
