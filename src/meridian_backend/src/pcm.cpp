#include "pcm.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cstddef>

#include "max_clique.hpp"

namespace meridian::backend {

namespace {

using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

Sophus::SE3d se3(const Pose& p) {
  return Sophus::SE3d(p.q, p.t);
}

// Symmetrise and lift all eigenvalues to at least 1e-12 so the result is strictly PSD and
// safe to invert. Floors a covariance that first-order propagation can leave singular.
Mat6 psdGuard(const Mat6& s) {
  const Mat6 sym = 0.5 * (s + s.transpose());
  Eigen::SelfAdjointEigenSolver<Mat6> eig(sym);
  Vec6 d = eig.eigenvalues();
  for (int i = 0; i < 6; ++i) {
    if (d(i) < 1e-12) d(i) = 1e-12;
  }
  return eig.eigenvectors() * d.asDiagonal() * eig.eigenvectors().transpose();
}

// epsilon(g1,g2,g3,g4) = Log( g1^{-1} * g2 * g3 * g4 ), the SE(3) loop-consistency residual.
Vec6 consistencyError(const Sophus::SE3d& g1, const Sophus::SE3d& g2, const Sophus::SE3d& g3,
                      const Sophus::SE3d& g4) {
  return (g1.inverse() * g2 * g3 * g4).log();
}

// Numerical Jacobian d epsilon / d delta_k at delta=0, where the perturbed input is
// g_k -> g_k * exp(delta_k^) under central differences with step h = 1e-6.
Mat6 jacobianWrt(int which, const std::array<Sophus::SE3d, 4>& g) {
  constexpr double kH = 1e-6;
  Mat6 j;
  for (int axis = 0; axis < 6; ++axis) {
    Vec6 d = Vec6::Zero();
    d(axis) = kH;
    std::array<Sophus::SE3d, 4> gp = g;
    std::array<Sophus::SE3d, 4> gm = g;
    gp[which] = g[which] * Sophus::SE3d::exp(d);
    gm[which] = g[which] * Sophus::SE3d::exp(-d);
    const Vec6 ep = consistencyError(gp[0], gp[1], gp[2], gp[3]);
    const Vec6 em = consistencyError(gm[0], gm[1], gm[2], gm[3]);
    j.col(axis) = (ep - em) / (2.0 * kH);
  }
  return j;
}

}  // namespace

Pcm::Pcm(double chi2_thresh, int max_nodes) : chi2_thresh_(chi2_thresh), max_nodes_(max_nodes) {}

std::size_t Pcm::add(const LoopConstraint& lc) {
  entries_.push_back(Entry{lc, State::Pending});
  return entries_.size() - 1;
}

const LoopConstraint& Pcm::at(std::size_t handle) const {
  return entries_.at(handle).lc;
}

void Pcm::mark_admitted(std::size_t handle) {
  entries_.at(handle).state = State::InGraph;
}
void Pcm::mark_evicted(std::size_t handle) {
  entries_.at(handle).state = State::Pending;
}
void Pcm::mark_rejected(std::size_t handle) {
  entries_.at(handle).state = State::Rejected;
}

bool Pcm::in_graph(std::size_t handle) const {
  return entries_.at(handle).state == State::InGraph;
}

std::size_t Pcm::pending_count() const {
  std::size_t n = 0;
  for (const auto& e : entries_) {
    if (e.state == State::Pending) ++n;
  }
  return n;
}

std::optional<Mat6> Pcm::relative_cov(std::uint64_t a_id, std::uint64_t b_id, const PoseFn& pose,
                                      const ChainCovFn& chain_cov) const {
  // A keyframe relative to itself is the identity with no chain uncertainty, so two loops that
  // share an endpoint still form a well-defined consistency cycle (the shared segment is I with
  // zero covariance) instead of being dropped as untestable.
  if (a_id == b_id) {
    if (!pose(a_id)) return std::nullopt;
    return Mat6::Zero();
  }
  const std::uint64_t lo = std::min(a_id, b_id);
  const std::uint64_t hi = std::max(a_id, b_id);
  const std::optional<Mat6> c = chain_cov(lo, hi);
  if (!c.has_value()) return std::nullopt;
  // c is the right-cov of T = X_lo^{-1} X_hi. If the requested relative is exactly T, return it.
  if (a_id == lo && b_id == hi) return *c;
  // Otherwise the relative is T^{-1}. Right perturbation transports as (T exp(d))^{-1} =
  // T^{-1} exp(-Ad_T d), so the inverse's right tangent is -Ad_T d and its cov is Ad_T C Ad_T^T.
  const std::optional<Pose> Xlo = pose(lo);
  const std::optional<Pose> Xhi = pose(hi);
  if (!Xlo || !Xhi) return std::nullopt;
  const Sophus::SE3d T = se3(*Xlo).inverse() * se3(*Xhi);
  const Mat6 ad = T.Adj();
  return ad * (*c) * ad.transpose();
}

std::optional<double> Pcm::pair_distance_sq(std::size_t a, std::size_t b, const PoseFn& pose,
                                            const ChainCovFn& chain_cov) const {
  const LoopConstraint& l1 = entries_.at(a).lc;
  const LoopConstraint& l2 = entries_.at(b).lc;

  const std::uint64_t p = l1.from_id, q = l1.to_id;
  const std::uint64_t r = l2.from_id, s = l2.to_id;

  const std::optional<Pose> Xp = pose(p);
  const std::optional<Pose> Xq = pose(q);
  const std::optional<Pose> Xr = pose(r);
  const std::optional<Pose> Xs = pose(s);
  if (!Xp || !Xq || !Xr || !Xs) return std::nullopt;

  // Estimate-derived trusted relatives between the two loops' endpoints.
  const Sophus::SE3d sXp = se3(*Xp), sXq = se3(*Xq), sXr = se3(*Xr), sXs = se3(*Xs);
  const Sophus::SE3d B = sXp.inverse() * sXr;  // relative p->r
  const Sophus::SE3d D = sXs.inverse() * sXq;  // relative s->q

  // Covariances of B and D from the odometry chain, each in B's / D's own right tangent.
  // B is the relative p->r, D is the relative s->q. relative_cov handles the older->newer
  // orientation and the inverse-adjoint transport internally given the endpoint poses.
  const std::optional<Mat6> Sigma_B = relative_cov(p, r, pose, chain_cov);
  const std::optional<Mat6> Sigma_D = relative_cov(s, q, pose, chain_cov);
  if (!Sigma_B || !Sigma_D) return std::nullopt;

  const Sophus::SE3d Z1 = se3(l1.T_from_to);
  const Sophus::SE3d Z2 = se3(l2.T_from_to);
  const Mat6 Sigma1 = l1.cov.M;
  const Mat6 Sigma2 = l2.cov.M;

  // M = Z1^{-1} * B * Z2 * D, epsilon = Log(M). Inputs are the four independent uncertain
  // factors g1=Z1, g2=B, g3=Z2, g4=D, each under a right perturbation.
  const std::array<Sophus::SE3d, 4> g{Z1, B, Z2, D};
  const Vec6 eps = consistencyError(g[0], g[1], g[2], g[3]);

  const std::array<Mat6, 4> sig{Sigma1, *Sigma_B, Sigma2, *Sigma_D};
  Mat6 Sigma_eps = Mat6::Zero();
  for (int k = 0; k < 4; ++k) {
    const Mat6 Jk = jacobianWrt(k, g);
    Sigma_eps += Jk * sig[k] * Jk.transpose();
  }

  const Mat6 guarded = psdGuard(Sigma_eps);
  const double d2 = eps.dot(guarded.ldlt().solve(eps));
  return d2;
}

PcmDecision Pcm::update(const PoseFn& pose, const ChainCovFn& chain_cov) {
  // Active set: all loops not yet rejected, compacted to dense indices 0..K-1.
  std::vector<std::size_t> active;
  for (std::size_t h = 0; h < entries_.size(); ++h) {
    if (entries_[h].state != State::Rejected) active.push_back(h);
  }
  const std::size_t k = active.size();

  // Symmetric consistency matrix over the active set; diagonal true so an isolated present
  // loop still forms a size-1 clique. defined(i,j) tracks pairs whose distance was computable.
  std::vector<std::vector<bool>> consistent(k, std::vector<bool>(k, false));
  std::vector<std::vector<bool>> defined(k, std::vector<bool>(k, false));
  for (std::size_t i = 0; i < k; ++i) consistent[i][i] = true;
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = i + 1; j < k; ++j) {
      const std::optional<double> d2 = pair_distance_sq(active[i], active[j], pose, chain_cov);
      if (d2.has_value()) {
        defined[i][j] = defined[j][i] = true;
        const bool ok = *d2 <= chi2_thresh_;
        consistent[i][j] = consistent[j][i] = ok;
      }
    }
  }

  const std::vector<int> clique_idx = max_clique(consistent, max_nodes_);
  std::vector<bool> in_clique(k, false);
  for (const int ci : clique_idx) in_clique[static_cast<std::size_t>(ci)] = true;

  PcmDecision out;

  // Admit: clique members currently Pending. Evict: InGraph members no longer in the clique.
  for (std::size_t i = 0; i < k; ++i) {
    const std::size_t h = active[i];
    const State st = entries_[h].state;
    if (in_clique[i]) {
      if (st == State::Pending) out.to_admit.push_back(h);
    } else if (st == State::InGraph) {
      out.to_evict.push_back(h);
    }
  }

  // Reject only against an established consensus (clique size >= 2): a Pending loop that is
  // testable against some clique member, is outside the clique, and shares no consistency edge
  // with any clique member. Loops merely missing endpoints this pass are left pending.
  if (clique_idx.size() >= 2) {
    for (std::size_t i = 0; i < k; ++i) {
      if (in_clique[i]) continue;
      if (entries_[active[i]].state != State::Pending) continue;
      bool testable = false;
      bool shares_edge = false;
      for (const int ci : clique_idx) {
        const std::size_t c = static_cast<std::size_t>(ci);
        if (defined[i][c]) testable = true;
        if (consistent[i][c]) shares_edge = true;
      }
      if (testable && !shares_edge) out.to_reject.push_back(active[i]);
    }
  }

  return out;
}

}  // namespace meridian::backend
