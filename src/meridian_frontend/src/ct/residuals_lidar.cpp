#include "residuals_lidar.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Eigen/Eigenvalues>
#include <array>

#include <basalt/spline/ceres_spline_helper.h>
#include <basalt/utils/sophus_utils.hpp>

#include "ct/spline_analytic.hpp"
#include <ceres/cost_function.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <sophus/so3.hpp>

#include "ikd_Tree.h"

namespace meridian::ct {

namespace {

constexpr int kSplineOrder = 4;

// Fixed range/incidence inflation terms for the measurement variance model
//   sigma^2 = point_cov + kSigmaRange^2 |p_L|^2 + kSigmaIncidence^2 (1 - |n.r_hat|).
// They are intentionally small relative to the configured point_cov base: at 10 m
// range the range term adds ~1e-4 m^2 and a fully grazing hit adds 4e-4 m^2, so
// distant and oblique returns are down-weighted without swamping the base term.
constexpr double kSigmaRange = 1.0e-3;      // per-metre range std [1/sqrt(m^2)]
constexpr double kSigmaIncidence = 0.02;    // grazing-incidence std [m]

// Whitened-residual Huber threshold. The residual fed to Ceres is already divided
// by the per-point sigma, so it is dimensionless and unit-variance under the noise
// model; a threshold of one standard deviation switches to the linear (robust)
// regime once a correspondence disagrees by more than its own expected spread.
constexpr double kHuberWhitened = 1.0;

ikdTree_PointType toIkd(const Eigen::Vector3d& p) {
  return ikdTree_PointType(static_cast<float>(p.x()), static_cast<float>(p.y()),
                           static_cast<float>(p.z()));
}

// sqrt-information weight from the range/incidence variance model. `p_lidar` is the
// raw point (its norm is the range); `n` is the fitted plane normal in world; the
// incidence term uses the lidar-frame ray direction rotated into world by R_w_l.
double pointWeight(const Eigen::Vector3d& p_lidar, const Eigen::Vector3d& n,
                   const Eigen::Matrix3d& R_w_l, double point_cov) {
  const double range_sq = p_lidar.squaredNorm();
  double incidence = 0.0;
  if (range_sq > 0.0) {
    const Eigen::Vector3d r_hat_w = (R_w_l * (p_lidar / std::sqrt(range_sq)));
    incidence = 1.0 - std::abs(n.dot(r_hat_w));
  }
  const double var = point_cov + kSigmaRange * kSigmaRange * range_sq +
                     kSigmaIncidence * kSigmaIncidence * incidence;
  return 1.0 / std::sqrt(var > 0.0 ? var : point_cov);
}

// One point-to-plane residual at a single point's true time. The 4 SO(3) and 4 R^3
// knot blocks of the point's segment are the parameter blocks, in that order; the
// world plane (n, d), the lidar-frame point, the extrinsic R_fe_l/t_fe_l, the
// segment position u and inv_dt, and the sqrt-information weight are fixed data.
struct LidarResidual {
  LidarResidual(const Eigen::Vector3d& p_lidar, const Eigen::Vector3d& n, double d,
                const Eigen::Quaterniond& q_fe_l, const Eigen::Vector3d& t_fe_l,
                double u, double inv_dt, double weight)
      : p_lidar_(p_lidar),
        n_(n),
        d_(d),
        q_fe_l_(q_fe_l),
        t_fe_l_(t_fe_l),
        u_(u),
        inv_dt_(inv_dt),
        weight_(weight) {}

  template <class T>
  bool operator()(T const* const* params, T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;

    SO3T R_w_fe;
    basalt::CeresSplineHelper<kSplineOrder>::template evaluate_lie<T, Sophus::SO3>(
        params, u_, inv_dt_, &R_w_fe);

    Vec3T p_w_fe;
    basalt::CeresSplineHelper<kSplineOrder>::template evaluate<T, 3, 0>(
        params + kSplineOrder, u_, inv_dt_, &p_w_fe);

    const SO3T R_fe_l = SO3T(q_fe_l_.cast<T>());
    const Vec3T q_fe = R_fe_l * p_lidar_.cast<T>() + t_fe_l_.cast<T>();
    const Vec3T p_world = R_w_fe * q_fe + p_w_fe;

    residual_ptr[0] = T(weight_) * (n_.cast<T>().dot(p_world) + T(d_));
    return true;
  }

  Eigen::Vector3d p_lidar_;
  Eigen::Vector3d n_;
  double d_;
  Eigen::Quaterniond q_fe_l_;
  Eigen::Vector3d t_fe_l_;
  double u_;
  double inv_dt_;
  double weight_;
};

}  // namespace

std::vector<LidarPoint> voxelDownsample(const std::vector<LidarPoint>& pts,
                                        double voxel_m) {
  if (voxel_m <= 0.0 || pts.empty()) {
    return pts;
  }
  const double inv = 1.0 / voxel_m;
  // Pack the three signed voxel indices into one 64-bit key (21 bits each, biased
  // to stay non-negative). The first point landing in a cell wins, so a stable scan
  // ordering yields a deterministic representative per cell.
  const auto key = [inv](const Eigen::Vector3f& p) -> std::int64_t {
    const auto qx = static_cast<std::int64_t>(std::floor(p.x() * inv)) + (1 << 20);
    const auto qy = static_cast<std::int64_t>(std::floor(p.y() * inv)) + (1 << 20);
    const auto qz = static_cast<std::int64_t>(std::floor(p.z() * inv)) + (1 << 20);
    return (qx & 0x1FFFFF) | ((qy & 0x1FFFFF) << 21) | ((qz & 0x1FFFFF) << 42);
  };
  std::unordered_map<std::int64_t, std::size_t> seen;
  seen.reserve(pts.size());
  std::vector<LidarPoint> out;
  out.reserve(pts.size());
  for (const LidarPoint& p : pts) {
    const auto [it, inserted] = seen.try_emplace(key(p.xyz), out.size());
    if (inserted) {
      out.push_back(p);
    }
  }
  return out;
}

struct LidarLocalMap::Impl {
  explicit Impl(const FrontendLidar& cfg)
      : tree(std::make_unique<KD_TREE<ikdTree_PointType>>(0.3f, 0.6f, 0.2f)) {
    const double v = cfg.voxel_map_m > 1e-3 ? cfg.voxel_map_m : 1e-3;
    tree->set_downsample_param(static_cast<float>(v));
    num_match_points = cfg.num_match_points > 0 ? cfg.num_match_points : 5;
    max_match_dist_sq = cfg.max_match_dist_sq;
    plane_thresh = cfg.plane_thresh;
  }

  std::unique_ptr<KD_TREE<ikdTree_PointType>> tree;
  int num_match_points = 5;
  double max_match_dist_sq = 5.0;
  double plane_thresh = 0.1;

  // Current world-frame cubic local-map bounds and whether they have been seeded. The
  // cube is recentred only when the body comes within move_margin of an edge, so the
  // slide amount is bounded and the tree is not re-segmented on every sweep.
  BoxPointType cube{};
  bool cube_init = false;
};

LidarLocalMap::LidarLocalMap(const FrontendLidar& cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}

LidarLocalMap::~LidarLocalMap() = default;
LidarLocalMap::LidarLocalMap(LidarLocalMap&&) noexcept = default;
LidarLocalMap& LidarLocalMap::operator=(LidarLocalMap&&) noexcept = default;

bool LidarLocalMap::initialized() const {
  return impl_->tree->size() > 0;
}

void LidarLocalMap::insert(const std::vector<Eigen::Vector3d>& pts_world) {
  if (pts_world.empty()) {
    return;
  }
  KD_TREE<ikdTree_PointType>::PointVector add;
  add.reserve(pts_world.size());
  for (const Eigen::Vector3d& p : pts_world) {
    add.push_back(toIkd(p));
  }
  if (impl_->tree->size() == 0) {
    impl_->tree->Build(add);
  } else {
    impl_->tree->Add_Points(add, true);
  }
}

void LidarLocalMap::trimAround(const Eigen::Vector3d& center, double cube_m) {
  if (cube_m <= 0.0 || impl_->tree->size() == 0) {
    return;
  }
  const float half = static_cast<float>(cube_m) * 0.5f;
  const auto c = center.cast<float>();

  // First call just seeds the cube around the body; nothing has left it yet.
  if (!impl_->cube_init) {
    for (int i = 0; i < 3; ++i) {
      impl_->cube.vertex_min[i] = c(i) - half;
      impl_->cube.vertex_max[i] = c(i) + half;
    }
    impl_->cube_init = true;
    return;
  }

  // Recentre an axis only once the body is within move_margin of one of its faces; the
  // margin must exceed the nearest-neighbour search radius so points around the body are
  // never deleted. The face then slides by mov_dist (capped so the cube cannot overshoot
  // the body), and the rectangular slab between the old and new face is box-deleted.
  const float search_radius = std::sqrt(static_cast<float>(impl_->max_match_dist_sq));
  const float move_margin = std::max(search_radius, half * 0.2f);
  const float mov_dist = std::max(half - move_margin, move_margin);

  std::vector<BoxPointType> need_remove;
  BoxPointType updated = impl_->cube;
  for (int i = 0; i < 3; ++i) {
    const float dist_min = std::abs(c(i) - impl_->cube.vertex_min[i]);
    const float dist_max = std::abs(c(i) - impl_->cube.vertex_max[i]);
    if (dist_min <= move_margin) {
      // Body near the low face: slide the cube toward -i and drop the high-side slab
      // [new_max, old_max] that just left the cube.
      updated.vertex_min[i] -= mov_dist;
      updated.vertex_max[i] -= mov_dist;
      BoxPointType slab = impl_->cube;
      slab.vertex_min[i] = updated.vertex_max[i];
      need_remove.push_back(slab);
    } else if (dist_max <= move_margin) {
      updated.vertex_min[i] += mov_dist;
      updated.vertex_max[i] += mov_dist;
      BoxPointType slab = impl_->cube;
      slab.vertex_max[i] = updated.vertex_min[i];
      need_remove.push_back(slab);
    }
  }
  impl_->cube = updated;
  if (!need_remove.empty()) {
    impl_->tree->Delete_Point_Boxes(need_remove);
  }
}

bool LidarLocalMap::fitPlane(const Eigen::Vector3d& p_world, PlaneFit* out) const {
  if (out != nullptr) {
    *out = PlaneFit{};
  }
  const int k = impl_->num_match_points;
  KD_TREE<ikdTree_PointType>::PointVector nbrs;
  std::vector<float> dists;
  impl_->tree->Nearest_Search(toIkd(p_world), k, nbrs, dists);
  if (static_cast<int>(nbrs.size()) < k) {
    return false;
  }
  // Distances come back squared and sorted ascending, so the last is the farthest.
  if (dists.back() > impl_->max_match_dist_sq) {
    return false;
  }

  // Solve A u = -1 for the unnormalised plane coefficients, then normalise so
  // n = u/|u| is unit and d = 1/|u|; the plane is n.x + d = 0.
  const int m = static_cast<int>(nbrs.size());
  Eigen::MatrixXd A(m, 3);
  const Eigen::VectorXd b = -Eigen::VectorXd::Ones(m);
  for (int i = 0; i < m; ++i) {
    A(i, 0) = nbrs[i].x;
    A(i, 1) = nbrs[i].y;
    A(i, 2) = nbrs[i].z;
  }
  const Eigen::Vector3d u = A.colPivHouseholderQr().solve(b);
  const double norm = u.norm();
  if (!(norm > 0.0)) {
    return false;
  }
  const double inv = 1.0 / norm;
  const Eigen::Vector3d n = u * inv;
  const double d = inv;

  for (int i = 0; i < m; ++i) {
    const Eigen::Vector3d q(nbrs[i].x, nbrs[i].y, nbrs[i].z);
    if (std::abs(n.dot(q) + d) > impl_->plane_thresh) {
      return false;
    }
  }
  if (out != nullptr) {
    out->n = n;
    out->d = d;
    out->valid = true;
  }
  return true;
}

std::size_t LidarLocalMap::size() const {
  return static_cast<std::size_t>(impl_->tree->size());
}

LidarAssocStats associate(const SplineWindow& spline, const Pose& T_fe_lidar,
                          const std::vector<LidarPoint>& pts, Timestamp t0_scan,
                          const LidarLocalMap& map, const FrontendLidar& cfg,
                          std::vector<LidarHit>* hits) {
  LidarAssocStats stats;
  const Eigen::Quaterniond q_fe_l = T_fe_lidar.q;
  const Eigen::Vector3d t_fe_l = T_fe_lidar.t;

  for (const LidarPoint& lp : pts) {
    const Timestamp t = t0_scan + static_cast<Timestamp>(lp.t_offset_ns);
    if (!spline.covers(t)) {
      continue;
    }
    ++stats.attempted;

    const Eigen::Vector3d p_lidar = lp.xyz.cast<double>();
    const Pose T_w_fe = spline.pose(t);
    const Eigen::Vector3d p_world = T_w_fe * (q_fe_l * p_lidar + t_fe_l);

    PlaneFit plane;
    if (!map.fitPlane(p_world, &plane)) {
      continue;
    }
    ++stats.matched;

    const double r = plane.n.dot(p_world) + plane.d;
    const double range = p_lidar.norm();
    // A point at the sensor origin has no usable bearing and is rejected.
    if (range <= 0.0) {
      continue;
    }
    // Range-aware acceptance: the tolerance grows with sqrt(range), so distant
    // returns are gated more leniently than near ones.
    const double s = 1.0 - 0.9 * std::abs(r) / std::sqrt(range);
    if (!(s > 0.9)) {
      continue;
    }
    ++stats.accepted;

    LidarHit hit;
    hit.p_lidar = p_lidar;
    hit.p_world = p_world;
    hit.t = t;
    hit.plane = plane;
    hit.t_offset_ns = lp.t_offset_ns;
    hit.ring = lp.ring;
    const Eigen::Matrix3d R_w_l = (T_w_fe.q * q_fe_l).toRotationMatrix();
    hit.weight = pointWeight(p_lidar, plane.n, R_w_l, cfg.point_cov);
    if (hits != nullptr) {
      hits->push_back(hit);
    }
  }
  return stats;
}

namespace {

// Bin a unit plane normal into one of `n_strata` normal strata. With the default 7
// strata the first six are the axis-aligned hemispheres (+x,-x,+y,-y,+z,-z, indexed
// 2*axis + (negative?1:0)) and the seventh is the "oblique" catch-all: a normal whose
// dominant component does not clearly dominate the other two is not aligned with any
// axis and lands in the oblique bin so axis bins stay pure. With fewer strata the
// scheme degrades gracefully: 2*A+1 axis bins for A=floor((n-1)/2) usable axes plus an
// oblique bin; <2 strata collapses everything into bin 0.
int normalStratum(const Eigen::Vector3d& n, int n_strata) {
  if (n_strata <= 1) {
    return 0;
  }
  // Reserve the last bin for oblique; the rest are axis hemispheres, two per axis.
  const int oblique = n_strata - 1;
  const int n_axes = std::min(3, oblique / 2);
  if (n_axes <= 0) {
    return 0;
  }
  int axis = 0;
  double best = std::abs(n(0));
  for (int a = 1; a < n_axes; ++a) {
    if (std::abs(n(a)) > best) {
      best = std::abs(n(a));
      axis = a;
    }
  }
  // A normal counts as axis-aligned only when its dominant component clearly leads;
  // the cos(35 deg) ~ 0.82 cut keeps the axis hemispheres free of near-diagonal
  // normals, which the oblique bin then collects.
  constexpr double kAxisAlignedCos = 0.82;
  if (best < kAxisAlignedCos) {
    return oblique;
  }
  return 2 * axis + (n(axis) < 0.0 ? 1 : 0);
}

// True when a normal aligns (within weak_axis_cos) with any supplied weak axis, so the
// hit it belongs to is exempt from the factor cap.
bool alignsWithWeakAxis(const Eigen::Vector3d& n,
                        const std::vector<Eigen::Vector3d>& weak_axes,
                        double weak_axis_cos) {
  for (const Eigen::Vector3d& w : weak_axes) {
    if (std::abs(n.dot(w)) >= weak_axis_cos) {
      return true;
    }
  }
  return false;
}

}  // namespace

LidarCapStats capHitsByNormalStrata(const std::vector<LidarHit>& hits,
                                    const FrontendLidar& cfg, int outer_step,
                                    const std::vector<Eigen::Vector3d>& weak_axes,
                                    double weak_axis_cos,
                                    std::vector<LidarHit>* out) {
  LidarCapStats stats;
  if (out == nullptr) {
    return stats;
  }
  out->clear();

  const int cap = cfg.max_lidar_factors;
  // A non-positive or slack cap means every inlier is built; copy through unchanged so
  // the within-budget path costs nothing beyond the copy.
  if (cap <= 0 || static_cast<int>(hits.size()) <= cap) {
    *out = hits;
    stats.kept = static_cast<int>(out->size());
    stats.dropped = 0;
    return stats;
  }

  const int n_strata = std::max(1, cfg.normal_strata);
  const int floor_per = std::max(0, cfg.min_factors_per_normal);

  // Split inliers into the cap-exempt weak-axis set (kept whole) and the cappable
  // remainder, binned by normal stratum. Indices index back into `hits`.
  out->reserve(static_cast<std::size_t>(cap));
  std::vector<std::vector<int>> bins(static_cast<std::size_t>(n_strata));
  for (int i = 0; i < static_cast<int>(hits.size()); ++i) {
    const LidarHit& h = hits[i];
    if (!h.plane.valid) {
      continue;
    }
    if (alignsWithWeakAxis(h.plane.n, weak_axes, weak_axis_cos)) {
      out->push_back(h);  // weak-axis points bypass the budget entirely
      continue;
    }
    bins[static_cast<std::size_t>(normalStratum(h.plane.n, n_strata))].push_back(i);
  }
  const int exempt = static_cast<int>(out->size());

  // Order each bin by the stable per-point key (t_offset_ns, then ring) so the stride
  // selection is deterministic and replay-identical regardless of association order.
  for (auto& bin : bins) {
    std::sort(bin.begin(), bin.end(), [&hits](int a, int b) {
      const LidarHit& ha = hits[static_cast<std::size_t>(a)];
      const LidarHit& hb = hits[static_cast<std::size_t>(b)];
      if (ha.t_offset_ns != hb.t_offset_ns) {
        return ha.t_offset_ns < hb.t_offset_ns;
      }
      if (ha.ring != hb.ring) {
        return ha.ring < hb.ring;
      }
      return a < b;
    });
  }

  // Budget for the cappable bins is whatever the weak-axis exemptions left. If the
  // exemptions already exceed the cap, no further points are admitted (the weak axis
  // protection wins over the count target by design).
  int budget = cap - exempt;
  if (budget < 0) {
    budget = 0;
  }

  // Equal-then-proportional allocation: first floor each non-empty bin (clamped to its
  // population) so a sparsely-sampled normal is never starved, then hand the remainder
  // out in proportion to population. Allocations are clamped to bin population and to
  // the running budget so the totals never exceed it.
  std::vector<int> alloc(static_cast<std::size_t>(n_strata), 0);
  int total_pop = 0;
  int nonempty = 0;
  for (const auto& bin : bins) {
    total_pop += static_cast<int>(bin.size());
    if (!bin.empty()) {
      ++nonempty;
    }
  }

  int remaining = budget;
  for (int s = 0; s < n_strata && remaining > 0; ++s) {
    const int pop = static_cast<int>(bins[static_cast<std::size_t>(s)].size());
    if (pop == 0) {
      continue;
    }
    const int give = std::min({floor_per, pop, remaining});
    alloc[static_cast<std::size_t>(s)] = give;
    remaining -= give;
  }
  // Proportional remainder over residual population (population not yet allocated).
  int residual_pop = 0;
  for (int s = 0; s < n_strata; ++s) {
    residual_pop +=
        static_cast<int>(bins[static_cast<std::size_t>(s)].size()) - alloc[static_cast<std::size_t>(s)];
  }
  if (remaining > 0 && residual_pop > 0) {
    const int to_share = remaining;
    for (int s = 0; s < n_strata && remaining > 0; ++s) {
      const int pop = static_cast<int>(bins[static_cast<std::size_t>(s)].size());
      const int left = pop - alloc[static_cast<std::size_t>(s)];
      if (left <= 0) {
        continue;
      }
      // Floor of the proportional share; any rounding slack is mopped up below.
      long share = static_cast<long>(to_share) * left / residual_pop;
      int give = std::min<int>(static_cast<int>(share), std::min(left, remaining));
      alloc[static_cast<std::size_t>(s)] += give;
      remaining -= give;
    }
    // Distribute the rounding remainder one-by-one over bins with spare population, in
    // bin order, so the full budget is used deterministically.
    for (int s = 0; s < n_strata && remaining > 0; ++s) {
      const int pop = static_cast<int>(bins[static_cast<std::size_t>(s)].size());
      while (alloc[static_cast<std::size_t>(s)] < pop && remaining > 0) {
        ++alloc[static_cast<std::size_t>(s)];
        --remaining;
      }
    }
  }
  (void)total_pop;
  (void)nonempty;

  // Within each bin pick `alloc[s]` points by a uniform stride, phase-rotated by
  // outer_step so consecutive steps visit a moving subset of the same bin. The stride
  // is ceil(pop/alloc) so the picks span the whole bin rather than its head.
  for (int s = 0; s < n_strata; ++s) {
    const std::vector<int>& bin = bins[static_cast<std::size_t>(s)];
    const int pop = static_cast<int>(bin.size());
    const int take = alloc[static_cast<std::size_t>(s)];
    if (take <= 0 || pop == 0) {
      continue;
    }
    if (take >= pop) {
      for (int idx : bin) {
        out->push_back(hits[static_cast<std::size_t>(idx)]);
      }
      continue;
    }
    const int stride = (pop + take - 1) / take;  // >= 1
    const int phase = stride > 0 ? (outer_step % stride) : 0;
    int picked = 0;
    for (int j = 0; j < pop && picked < take; ++j) {
      // Walk the bin in stride steps starting at the phase offset; wrap is unnecessary
      // because stride*take >= pop covers the bin in one pass.
      const int pos = phase + j * stride;
      if (pos >= pop) {
        break;
      }
      out->push_back(hits[static_cast<std::size_t>(bin[static_cast<std::size_t>(pos)])]);
      ++picked;
    }
    // If the phase offset left fewer than `take` picks (the tail fell off the end),
    // backfill from the front of the bin so the per-stratum allocation is honoured.
    for (int j = 0; j < pop && picked < take; ++j) {
      const int pos = j;
      const bool on_stride = ((pos - phase) % stride == 0) && pos >= phase;
      if (on_stride) {
        continue;  // already taken in the first pass
      }
      out->push_back(hits[static_cast<std::size_t>(bin[static_cast<std::size_t>(pos)])]);
      ++picked;
    }
  }

  stats.kept = static_cast<int>(out->size());
  stats.dropped = static_cast<int>(hits.size()) - stats.kept;
  if (stats.dropped < 0) {
    stats.dropped = 0;
  }
  return stats;
}

std::vector<Eigen::Vector3d> weakTranslationAxes(const std::vector<LidarHit>& hits,
                                                 double degeneracy_thresh) {
  // A point-to-plane factor constrains translation along its plane normal n, so the
  // translation information of the association set is sum_j w_j^2 n_j n_j^T. Its
  // eigenvectors with a low conditioning score are the directions the LiDAR barely
  // sees (e.g. along a featureless corridor).
  Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
  for (const LidarHit& h : hits) {
    if (!h.plane.valid) {
      continue;
    }
    info += (h.weight * h.weight) * (h.plane.n * h.plane.n.transpose());
  }
  std::vector<Eigen::Vector3d> weak;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(info);
  if (es.info() != Eigen::Success) {
    return weak;
  }
  const double kappa = degeneracy_thresh > 0 ? degeneracy_thresh : 10.0;
  const Eigen::Vector3d lambdas = es.eigenvalues();
  const Eigen::Matrix3d V = es.eigenvectors();
  for (int e = 0; e < 3; ++e) {
    // lambda/(lambda+kappa) < 0.1 marks a weak axis, the same gate the observability
    // score uses, so the exemption tracks the degeneracy flag.
    const double lambda = std::max(lambdas(e), 0.0);
    if (lambda / (lambda + kappa) < 0.1) {
      weak.push_back(V.col(e));
    }
  }
  return weak;
}


// Analytic point-to-plane cost over the cumulative-form cubic SO(3) spline and the
// R^3 spline. Residual (identical to the autodiff LidarResidual):
//
//   r = w * ( n . ( R(u) * q_fe + p(u) ) + d ),   q_fe = R_fe_l * p_lidar + t_fe_l
//
// with R(u) = R_0 * prod_{j=1..3} exp( lambda_R[j] * delta_j ),
//      delta_j = Log( R_{j-1}^{-1} R_j ),   p(u) = sum_j lambda_P[j] * P_j.
//
// SO(3) knot Jacobians use the chain rule on the cumulative form. For a RIGHT
// perturbation R_j -> R_j * exp(theta^):
//
//   d r / d theta_j = ddis_dRt * dRt_dR_j,
//   ddis_dRt        = -w * n^T * R(u) * [q_fe]_x
//   dRt_dR_j        =  lambda_j   * P_j   * Jr(lambda_j   delta_j)   * Jr^-1(delta_j)
//                    - lambda_j+1 * P_j+1 * Jr(lambda_j+1 delta_j+1) * Jr^-1(delta_j+1)^T
//
// where P_j = A_3^-1 ... A_{j+1}^-1 (A_k = exp(lambda_k delta_k)) and the second
// term is absent for the last knot. d delta_j / d theta_j = Jr^-1(delta_j) and
// d delta_{j+1} / d theta_j = -Jr^-1(-delta_{j+1}) = -Jr^-1(delta_{j+1})^T account
// for knot j appearing on the right of delta_j and (inverted) on the left of
// delta_{j+1}. The R^3 knot Jacobians are the scalar blending weights:
// d r / d P_j = w * lambda_P[j] * n^T.
//
// Ceres consumes ambient-quaternion Jacobians on manifold blocks (the knots carry
// EigenQuaternionManifold), so the right-tangent rows are mapped to quaternion
// coordinates with d theta / d q~ at q~ = q for theta = Log(q^-1 q~):
//
//   J_q = J_theta * 2 * [ q_w I - [q_v]_x  |  -q_v ]        (3x4, (x,y,z,w) order)
//
// (vec(q^-1 (x) q~) = q_w q~_v - q~_w q_v - q_v x q~_v, so the skew enters negated)
//
// which is exact along the unit sphere's tangent; the off-manifold component is
// annihilated by the manifold's PlusJacobian.
class AnalyticLidarPlaneCost final : public ceres::CostFunction {
 public:
  AnalyticLidarPlaneCost(const Eigen::Vector3d& p_lidar, const Eigen::Vector3d& n, double d,
                         const Eigen::Quaterniond& q_fe_l, const Eigen::Vector3d& t_fe_l,
                         double u, double weight)
      : n_(n), d_(d), q_fe_(q_fe_l * p_lidar + t_fe_l), weight_(weight) {
    set_num_residuals(1);
    for (int j = 0; j < kSplineOrder; ++j) {
      mutable_parameter_block_sizes()->push_back(4);
    }
    for (int j = 0; j < kSplineOrder; ++j) {
      mutable_parameter_block_sizes()->push_back(3);
    }
    const Eigen::Vector4d up(1.0, u, u * u, u * u * u);
    lambda_r_ = basalt::CeresSplineHelper<kSplineOrder>::cumulative_blending_matrix_ * up;
    lambda_p_ = basalt::CeresSplineHelper<kSplineOrder>::blending_matrix_ * up;
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    using SO3 = Sophus::SO3d;
    std::array<SO3, kSplineOrder> R;
    for (int j = 0; j < kSplineOrder; ++j) {
      R[static_cast<std::size_t>(j)] = SO3(Eigen::Map<const Eigen::Quaterniond>(parameters[j]));
    }
    const So3SplineJac sj = evaluateSo3CumulativeJac(R, lambda_r_);
    const SO3& R_w_fe = sj.R;

    Eigen::Vector3d p_w_fe = Eigen::Vector3d::Zero();
    for (int j = 0; j < kSplineOrder; ++j) {
      p_w_fe += lambda_p_[j] * Eigen::Map<const Eigen::Vector3d>(parameters[kSplineOrder + j]);
    }

    residuals[0] = weight_ * (n_.dot(R_w_fe * q_fe_ + p_w_fe) + d_);

    if (jacobians == nullptr) {
      return true;
    }

    // d r / d theta_t for a right perturbation of the evaluated rotation.
    const Eigen::Matrix<double, 1, 3> ddis_dRt =
        -weight_ * n_.transpose() * R_w_fe.matrix() * SO3::hat(q_fe_);

    for (int j = 0; j < kSplineOrder; ++j) {
      if (jacobians[j] == nullptr) {
        continue;
      }
      const std::size_t k = static_cast<std::size_t>(j);
      const Eigen::Matrix<double, 1, 3> j_theta = ddis_dRt * sj.dRt_dR[k];
      Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> J(jacobians[j]);
      J = j_theta * tangentToQuatJac(R[k].unit_quaternion());
    }

    for (int j = 0; j < kSplineOrder; ++j) {
      if (jacobians[kSplineOrder + j] == nullptr) {
        continue;
      }
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> J(jacobians[kSplineOrder + j]);
      J = weight_ * lambda_p_[j] * n_.transpose();
    }
    return true;
  }

 private:
  Eigen::Vector3d n_;
  double d_;
  Eigen::Vector3d q_fe_;  // lidar point pre-composed with the extrinsic, body frame
  double weight_;
  Eigen::Vector4d lambda_r_;
  Eigen::Vector4d lambda_p_;
};

ceres::CostFunction* makeLidarPlaneCost(const Eigen::Vector3d& p_lidar,
                                        const Eigen::Vector3d& n, double d,
                                        const Eigen::Quaterniond& q_fe_l,
                                        const Eigen::Vector3d& t_fe_l, double u,
                                        double weight) {
  return new AnalyticLidarPlaneCost(p_lidar, n, d, q_fe_l, t_fe_l, u, weight);
}

int addLidarResiduals(ceres::Problem& problem, SplineWindow& spline,
                      const Pose& T_fe_lidar, const std::vector<LidarHit>& hits,
                      const FrontendLidar& cfg) {
  // The residual handed to Ceres is whitened by each hit's sqrt-information weight,
  // so it is dimensionless and unit-variance under the base noise model. The Huber
  // threshold is therefore a count of standard deviations, widened when the map
  // voxel is coarse relative to the base point std (sqrt(point_cov)) because a
  // coarser map yields noisier plane fits and should tolerate larger disagreements
  // before clamping. The widening is clamped so a fine map keeps the nominal band.
  const double base_std = std::sqrt(cfg.point_cov > 0.0 ? cfg.point_cov : 1.0);
  const double coarseness = cfg.voxel_map_m > 0.0 ? cfg.voxel_map_m / base_std : 1.0;
  const double huber = kHuberWhitened * std::max(1.0, std::min(coarseness, 4.0));
  int added = 0;
  for (const LidarHit& hit : hits) {
    if (!spline.covers(hit.t)) {
      continue;
    }
    const SplineWindow::SegmentRef seg = spline.segmentFor(hit.t);

    ceres::CostFunction* cost = makeLidarPlaneCost(hit.p_lidar, hit.plane.n, hit.plane.d,
                                                   T_fe_lidar.q, T_fe_lidar.t, seg.u,
                                                   hit.weight);

    std::vector<double*> blocks;
    blocks.reserve(2 * kSplineOrder);
    for (double* p : seg.so3_knots) {
      blocks.push_back(p);
    }
    for (double* p : seg.r3_knots) {
      blocks.push_back(p);
    }

    auto* loss = new ceres::HuberLoss(huber);
    problem.AddResidualBlock(cost, loss, blocks);
    ++added;
  }
  return added;
}

}  // namespace meridian::ct
