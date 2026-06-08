#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "ct/image_pyramid_view.hpp"
#include "ct/residuals_lidar.hpp"
#include "meridian/calib/camera_model.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"

namespace meridian::ct {

// One stored reference observation of a visual point: its three-level patch and the
// camera pose / inverse-exposure it was sampled under, plus the running score and a
// cached view direction (world-frame, point -> camera centre, unit) used by the
// add-gate, medoid selection, and re-scoring without recomputing it per pair.
struct VisualObservation {
  static constexpr int kPatch = 8;
  static constexpr int kLevels = 3;
  std::array<Eigen::Matrix<float, kPatch, kPatch>, kLevels> patches;
  Pose T_w_c;                                           // camera pose at this observation
  double inv_expo = 1.0;                                // inverse exposure at this observation
  Eigen::Vector3d view_dir = Eigen::Vector3d::UnitZ();  // unit, point -> cam centre
  float score = 0.f;                                    // w*NCC + (1-w)*cos(view-change)
};

// A visual map point: a world point with a LiDAR-fit plane normal, a bounded list of
// reference observations, and the lifecycle flags / reference patch the photometric
// residual reads. The flat reference fields (ref_patches, T_w_c_ref, inv_expo_ref)
// always mirror the current medoid observation, so the residual owner reads them
// directly without touching the internal observation list. Observations are RAII-
// owned through the enclosing VisualMap; ids stay stable across reorganisation.
struct VisualPoint {
  static constexpr int kPatch = VisualObservation::kPatch;
  static constexpr int kLevels = VisualObservation::kLevels;

  Eigen::Vector3d p_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d n_world = Eigen::Vector3d::Zero();  // plane normal (unit) or zero

  // Reference (medoid) observation surface — the pinned contract the residual reads.
  std::array<Eigen::Matrix<float, kPatch, kPatch>, kLevels> ref_patches;
  Pose T_w_c_ref;             // camera pose at the reference (medoid) observation
  double inv_expo_ref = 1.0;  // inverse exposure at the reference observation
  float score = 0.f;          // best observation score (selection / eviction)
  int obs_count = 0;          // number of stored observations

  // Internal lifecycle state (L2-only; not part of the residual contract surface).
  std::vector<VisualObservation> obs;  // bounded by ref_obs_cap
  int ref_index = 0;                   // medoid index into obs
  bool normal_initialized = false;
  bool converged = false;  // latched: obs list frozen, re-scoring still runs
  std::int64_t id = 0;     // stable creation order, deterministic tie-break
};

// Tunables for the visual-point map. Built from FrontendVisual where those fields
// exist; the remaining knobs keep their defaults until the config schema gains them.
struct VisualMapConfig {
  int patch = VisualObservation::kPatch;    // visual.patch_size
  int levels = VisualObservation::kLevels;  // visual.pyramid_levels
  int grid_cell_px = 32;                    // visual.grid_cell_px
  int ref_obs_cap = 30;                     // visual.ref_obs_cap
  double ref_add_angle_deg = 10.0;          // visual.ref_add_angle_deg
  double ref_add_score = 0.05;              // visual.ref_add_score
  double ref_score_w = 0.7;                 // visual.ref_score_w
  int ref_converged_obs = 8;                // visual.ref_converged_obs
  double ref_converged_angle_deg = 30.0;    // visual.ref_converged_angle_deg
  double active_box_m = 60.0;               // visual.active_box_m
  double voxel_m = 0.5;                     // shared LiDAR voxel hash pitch
  double depth_continuity_m = 0.5;          // depth-continuity gate
  double min_score_keep = 0.0;              // score-evict floor (<=0 disables)

  VisualMapConfig() = default;
  // Reads the fields FrontendVisual currently exposes; the rest keep their defaults.
  explicit VisualMapConfig(const FrontendVisual& v);
};

// The active visual-point map: promotes LiDAR-map points to visual
// points one-best-per-grid-cell, serves frustum/grid/depth-gated candidates to the
// photometric residual, runs the reference-patch lifecycle, and box-evicts points
// that leave the active box so memory stays bounded.
//
// Determinism: the point store is a contiguous vector keyed by a monotonically
// increasing id; the spatial index is an ordered std::map of voxel keys to
// id-sorted point-index lists. Every selection (promotion grid winner, candidate
// per cell, eviction order) is resolved by an explicit deterministic key, never by
// hash-table iteration order, so identical inputs yield byte-identical state.
class VisualMap {
public:
  explicit VisualMap(const VisualMapConfig& cfg);

  // Promote accepted LiDAR hits visible in the current frame to visual points.
  // Per grid cell the single highest-gradient-score candidate (with an initialized
  // plane normal) is kept; its kLevels reference patches are extracted from `img`
  // at the projected pixel and stored with the reference pose / inverse exposure.
  // Returns the number of points added.
  int promote(const ImagePyramidView& img, const CameraModel& cam, const Pose& T_w_c,
              double inv_expo, const std::vector<LidarHit>& hits);

  // Visible candidates for the current frame: frustum + grid one-best-per-cell
  // (nearest point wins the cell) + depth-continuity pre-filter. The photometric
  // NCC/SSD/warp gates are the residual owner's, not applied here. Result order is
  // deterministic (grid-cell index order).
  std::vector<const VisualPoint*> visibleCandidates(const CameraModel& cam,
                                                    const Pose& T_w_c) const;

  // Per-sweep visible-set cache for the CT hot loop. The visible selection (the
  // O(map) projection scan) is the per-sweep cost driver; computing it once at the
  // seed pose and reusing it across every outer re-association round AND
  // updateAfterSolve replaces up to (max_outer + 1) full scans with one. The warp /
  // gate / residual still use the current per-round pose -- only the selection of
  // WHICH points is fixed, the standard direct-method choice. Caller must
  // refreshVisibleCache() once before using cachedVisibleCandidates() /
  // updateAfterSolve() each sweep.
  void refreshVisibleCache(const CameraModel& cam, const Pose& T_w_c);
  std::vector<const VisualPoint*> cachedVisibleCandidates() const;

  // Reference-patch lifecycle after a solve: re-score observations against the
  // current view, recompute the medoid, append a new observation through the add
  // gate (respecting the obs cap with min-score eviction), and latch convergence.
  void updateAfterSolve(const ImagePyramidView& img, const CameraModel& cam, const Pose& T_w_c,
                        double inv_expo);

  // Bounded memory: drop points outside the active box centred on the current
  // position, and (if enabled) points whose best score fell below the keep floor.
  void evict(const Eigen::Vector3d& p_world_now);

  std::size_t size() const { return points_.size(); }

  // Deterministic state digest for replay-equivalence tests: order-independent of
  // storage, derived from the live points sorted by id.
  struct StateSummary {
    std::size_t count = 0;
    std::int64_t next_id = 0;
    std::uint64_t hash = 0;  // FNV-1a over rounded point fields, id-sorted
  };
  StateSummary summary() const;

private:
  using VoxelKey = std::array<std::int64_t, 3>;

  VoxelKey voxelKey(const Eigen::Vector3d& p) const;
  // Project a world point into the camera; returns true with *uv filled when the
  // point is in front of the camera and on the image.
  bool projectWorld(const CameraModel& cam, const Pose& T_w_c, const Eigen::Vector3d& p_world,
                    Eigen::Vector3d* p_cam, Eigen::Vector2d* uv) const;
  // True when a patch margin around `uv` fits inside the [0,w)x[0,h) image.
  bool inBoundsCam(const Eigen::Vector2d& uv, int w, int h) const;
  // Live point with the given id, or nullptr if it was evicted.
  const VisualPoint* pointById(std::int64_t id) const;
  VisualPoint* pointByIdMutable(std::int64_t id);
  // Ids of the per-cell nearest in-frustum points that pass the depth-continuity
  // gate -- the points the camera can actually use this frame. Shared by
  // visibleCandidates() and updateAfterSolve() so both bound their work to the
  // visible set (<= grid cells) instead of scanning the whole map.
  std::vector<std::int64_t> selectVisibleIds(const CameraModel& cam, const Pose& T_w_c) const;
  // Shi-Tomasi-like corner score from the local gradient structure tensor at `uv`.
  float gradientScore(const ImagePyramidView& img, const Eigen::Vector2d& uv) const;
  // Extract the kLevels reference patches centred at level-0 pixel `uv`.
  void extractPatches(const ImagePyramidView& img, const Eigen::Vector2d& uv,
                      VisualObservation* obs) const;
  // NCC of two observations' level-0 patches in [-1,1].
  static float patchNcc(const VisualObservation& a, const VisualObservation& b);
  void recomputeRefAndScore(VisualPoint* pt) const;
  void addPointAt(VoxelKey key, std::int64_t id);
  void removeFromIndex(VoxelKey key, std::int64_t id);

  VisualMapConfig cfg_;
  int half_patch_ = 4;
  int patch_margin_ = 5;  // patch half + 1 bilinear pad
  double cos_add_angle_ = 0.0;
  double cos_converged_angle_ = 0.0;

  // Point store keyed by stable id. An ordered map gives deterministic id-ascending
  // iteration for every selection while letting eviction erase the entry (and free
  // its RAII observation list) outright, so memory tracks the live set, not the
  // lifetime total of promotions.
  std::map<std::int64_t, std::unique_ptr<VisualPoint>> points_;
  std::map<VoxelKey, std::vector<std::int64_t>> index_;  // voxel -> sorted ids
  std::int64_t next_id_ = 0;
  // Visible-candidate ids selected at this sweep's seed pose (refreshVisibleCache).
  std::vector<std::int64_t> visible_cache_ids_;
};

}  // namespace meridian::ct
