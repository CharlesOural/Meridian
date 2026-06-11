#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "meridian/common/point.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"

#include "spline_window.hpp"

namespace ceres {
class CostFunction;
class Problem;
}  // namespace ceres

namespace meridian::ct {

// Voxel-grid downsample keeping the first point seen per occupied cell. Cells are
// indexed by floor(p / voxel_m); a non-positive voxel size returns the input as-is.
std::vector<LidarPoint> voxelDownsample(const std::vector<LidarPoint>& pts,
                                        double voxel_m);

// A least-squares plane n.x + d = 0 with a unit normal. `valid` is false when the
// fit was rejected by the distance or planarity gates; `reject` then carries which
// gate fired. rms / nn_max_dist are output-only fit-quality diagnostics filled on
// success: the RMS point-to-fitted-plane distance over the k neighbours and the
// distance to the farthest neighbour used in the fit [m].
struct PlaneFit {
  enum class Reject : std::uint8_t { None = 0, NoNeighbors, TooFar, NotPlanar };
  Eigen::Vector3d n = Eigen::Vector3d::Zero();  // unit normal
  double d = 0.0;                               // offset: n.x + d = 0
  double rms = 0.0;                             // plane-fit RMS over the neighbours [m]
  double nn_max_dist = 0.0;                     // farthest fit neighbour [m]
  bool valid = false;
  Reject reject = Reject::None;
};

// ikd-Tree-backed local map of points expressed in the world/odom frame. The
// vendored tree type is hidden behind a pimpl so no PCL/ikd-Tree symbol leaks into
// this header; the move operations transfer the tree without rebuilding it.
class LidarLocalMap {
 public:
  explicit LidarLocalMap(const FrontendLidar& cfg);
  ~LidarLocalMap();
  LidarLocalMap(LidarLocalMap&&) noexcept;
  LidarLocalMap& operator=(LidarLocalMap&&) noexcept;
  LidarLocalMap(const LidarLocalMap&) = delete;
  LidarLocalMap& operator=(const LidarLocalMap&) = delete;

  bool initialized() const;

  // The first call builds the tree from `pts_world`; later calls add to it with
  // tree-side voxel downsampling on, so the map never grows denser than the voxel.
  void insert(const std::vector<Eigen::Vector3d>& pts_world);

  // Keeps only the cubic local map of side `cube_m` centred on `center` (world frame):
  // when the body nears an edge, the cube slides and the exited slab is box-deleted from
  // the ikd-Tree. Bounds map RAM and nearest-neighbour search depth on a long mission.
  // A non-positive cube_m is a no-op (trimming disabled).
  void trimAround(const Eigen::Vector3d& center, double cube_m);

  // Resolves every pending lazy push-down the tree carries, single-threaded, under the
  // exclusive lock. insert()/trimAround() mark interior nodes with deferred delete/
  // downsample flags that a later search would otherwise settle lazily by writing a
  // child node's flags while a concurrent searcher reads them. Settling them all here,
  // before the next parallel fitPlane round, means those searches only ever read
  // already-resolved flags and never take the in-search push-down branch, so concurrent
  // searches cannot race each other on those flags. Call once after the sweep's last
  // tree modification (insert + trimAround) and before the next association round.
  void flushPendingDeletes();

  // 5-NN plane fit at `p_world` with the distance and planarity gates from cfg:
  // rejects when fewer than cfg.num_match_points neighbours are found, when the
  // farthest neighbour's squared distance exceeds cfg.max_match_dist_sq, or when any
  // neighbour lies farther than cfg.plane_thresh from the fitted plane. Returns true
  // and fills *out (with out->valid == true) only when the fit passes every gate.
  bool fitPlane(const Eigen::Vector3d& p_world, PlaneFit* out) const;

  std::size_t size() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// One accepted point-to-plane association, retained so the solver can re-evaluate it
// and the front-end can extract observability at pack time.
struct LidarHit {
  Eigen::Vector3d p_lidar = Eigen::Vector3d::Zero();  // raw point in the lidar frame
  Eigen::Vector3d p_world = Eigen::Vector3d::Zero();  // point in world at its spline pose
  Timestamp t = 0;                                    // absolute point time
  PlaneFit plane;                                     // world-frame matched plane
  double weight = 1.0;  // sqrt-information from the range/incidence model
  std::int32_t t_offset_ns = 0;  // per-point sweep time, the stable subsample key
  std::uint16_t ring = 0;        // laser/row id, the first stride tie-break
};

// Outcome of the bounded-factor-count selection: how many hits were kept (built as
// residuals this step) and how many were dropped by the cap. kept + dropped equals
// the inlier count; both are zero when no cap applied. kept_per_stratum is the kept
// count binned by normal stratum (size = cfg.normal_strata; weak-axis-exempt hits
// are counted in their own normal's bin) — output-only, never read back.
struct LidarCapStats {
  int kept = 0;
  int dropped = 0;
  std::vector<int> kept_per_stratum;
};

// Caps the residual count at cfg.max_lidar_factors by normal-stratified subsampling
// (never uniform/spatial decimation, which thins the rarest normals first). Survivors
// are binned by plane normal into cfg.normal_strata bins (six axis-aligned hemispheres
// plus an oblique bin at the default of 7); the budget is allocated equal-then-
// proportional (each non-empty bin floored at cfg.min_factors_per_normal, remainder
// shared by population); within a bin points are picked by a deterministic stride over
// the (t_offset_ns, ring) key with the stride phase rotated by outer_step so the
// visited subset moves across steps yet is identical on replay. A hit whose normal
// aligns (|n.axis| >= weak_axis_cos) with any weak observability axis in `weak_axes`
// is exempt from the cap and always kept. Writes the kept subset to *out (cleared
// first) and returns the kept/dropped counts. When the inlier set is within budget the
// input is copied through unchanged and dropped == 0.
LidarCapStats capHitsByNormalStrata(const std::vector<LidarHit>& hits,
                                    const FrontendLidar& cfg, int outer_step,
                                    const std::vector<Eigen::Vector3d>& weak_axes,
                                    double weak_axis_cos,
                                    std::vector<LidarHit>* out);

// Weak translation axes (unit world vectors) of a LiDAR association set. A point-to-
// plane factor constrains translation along its plane normal n, so the translation
// information of the set is sum_j w_j^2 n_j n_j^T; an eigenvector of that block whose
// conditioning score lambda/(lambda+degeneracy_thresh) falls below 0.1 is a direction
// the LiDAR barely observes (e.g. the long axis of a corridor) and is returned as
// weak. A non-positive degeneracy_thresh falls back to 10.0. Invalid-plane hits are
// skipped; an eigen-decomposition failure yields an empty set.
std::vector<Eigen::Vector3d> weakTranslationAxes(const std::vector<LidarHit>& hits,
                                                 double degeneracy_thresh);

// Per-scan association bookkeeping. `attempted` counts points whose time the spline
// covers, `matched` those that passed the plane gates, `accepted` those that also
// passed the range-aware acceptance score and became hits. The reject_* counters
// break attempted - accepted down by reason: the three plane-fit gates plus the
// score gate (attempted == matched + reject_no_neighbors + reject_too_far +
// reject_not_planar, and matched == accepted + reject_score).
struct LidarAssocStats {
  int attempted = 0;
  int matched = 0;
  int accepted = 0;
  int reject_no_neighbors = 0;  // fewer than num_match_points neighbours found
  int reject_too_far = 0;       // farthest neighbour beyond max_match_dist_sq
  int reject_not_planar = 0;    // a neighbour beyond plane_thresh of the fit
  int reject_score = 0;         // plane ok, range-aware acceptance score failed
};

// Associates a (time-sorted, downsampled) scan against `map` at the current spline
// poses. Each point at absolute time t0_scan + t_offset is transformed by
// T_W_Fe(t) * T_fe_lidar, plane-fit by 5-NN, accepted iff the range-aware score
// s = 1 - 0.9|r|/sqrt(|p_L|) exceeds 0.9, and given a range/incidence-aware weight.
// Points whose time the spline does not cover are skipped. Appends accepted hits to
// *hits and returns the counts.
LidarAssocStats associate(const SplineWindow& spline, const Pose& T_fe_lidar,
                          const std::vector<LidarPoint>& pts, Timestamp t0_scan,
                          const LidarLocalMap& map, const FrontendLidar& cfg,
                          std::vector<LidarHit>* hits);

// Adds one point-to-plane residual per hit:
//   r = w * ( n . ( R_W_Fe(t)(R_fe_l p + t_fe_l) + p_W_Fe(t) ) + d )
// autodiff over the 4 SO(3) + 4 R^3 knot blocks of t's segment, with a Huber loss
// whose scale is derived from the weight model. Returns the number of residuals
// added (one per hit whose time the spline still covers).
// Builds one analytic point-to-plane cost (1 residual; parameter blocks: 4 SO(3)
// knots stored as Eigen quaternions, then 4 R^3 knots). Mathematically identical to
// the autodiff LidarResidual but with closed-form Jacobians via the cumulative-form
// chain rule, cutting the per-factor evaluation cost that dominates the solve.
// Exposed for the derivative-correctness tests; addLidarResiduals uses it
// internally. `u` is the normalized segment position; the SO(3) knot Jacobians are
// returned in ambient quaternion coordinates, compatible with
// ceres::EigenQuaternionManifold on the knot blocks. `blend` / `cum_blend` are the
// segment's per-interval (non-uniform) blending matrices; nullptr selects the uniform
// cardinal matrices bit-identically. They are read only inside the factory call (the
// blended coefficients are baked into the cost).
ceres::CostFunction* makeLidarPlaneCost(const Eigen::Vector3d& p_lidar,
                                        const Eigen::Vector3d& n, double d,
                                        const Eigen::Quaterniond& q_fe_l,
                                        const Eigen::Vector3d& t_fe_l, double u,
                                        double weight,
                                        const Eigen::Matrix4d* blend = nullptr,
                                        const Eigen::Matrix4d* cum_blend = nullptr);

int addLidarResiduals(ceres::Problem& problem, SplineWindow& spline,
                      const Pose& T_fe_lidar, const std::vector<LidarHit>& hits,
                      const FrontendLidar& cfg);

}  // namespace meridian::ct
