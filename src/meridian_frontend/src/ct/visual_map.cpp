#include "ct/visual_map.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace meridian::ct {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Camera-frame point from a world point given T_w_c (camera pose in world).
Eigen::Vector3d worldToCam(const Pose& T_w_c, const Eigen::Vector3d& p_world) {
  return T_w_c.q.conjugate() * (p_world - T_w_c.t);
}

// A half-space in world coordinates: a point x is on the kept (inside) side when
// n.x + d >= 0. The view frustum is the intersection of five such half-spaces.
struct WorldPlane {
  Eigen::Vector3d n;
  double d;
};

// World-frame view-frustum planes for conservative voxel culling: the near plane
// (camera z >= 0) and four side planes through the camera centre, built from the
// image-corner rays so a voxel whose axis-aligned box lies wholly outside any one
// plane cannot contain a point that projects on-image. The corner rays are pushed
// out by a fixed angular margin so lens distortion (which bows the true frustum
// boundary outward between the sampled corners) can never carry an in-frustum
// point outside the cull volume -- the kept set stays a strict superset of the
// points the exact per-point projection test admits.
std::array<WorldPlane, 5> frustumPlanes(const CameraModel& cam, const Pose& T_w_c) {
  const double w = static_cast<double>(cam.width());
  const double h = static_cast<double>(cam.height());
  // Image corners, unprojected to unit-depth camera rays (undistort handles the
  // lens map), then widened from the optical axis by the distortion margin.
  const std::array<Eigen::Vector2d, 4> corners_px = {
      Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(w - 1.0, 0.0),
      Eigen::Vector2d(w - 1.0, h - 1.0), Eigen::Vector2d(0.0, h - 1.0)};
  constexpr double kMargin = 1.25;  // ray-spread inflation covering distortion bow
  std::array<Eigen::Vector3d, 4> rays_c;
  for (std::size_t i = 0; i < 4; ++i) {
    const Eigen::Vector3d r = cam.unproject(corners_px[i]);
    rays_c[i] = Eigen::Vector3d(r.x() * kMargin, r.y() * kMargin, 1.0);
  }

  const Eigen::Matrix3d R = T_w_c.q.toRotationMatrix();
  const Eigen::Vector3d c = T_w_c.t;  // camera centre in world

  std::array<WorldPlane, 5> planes;
  // Near plane: camera z >= 0 (in front of the optical centre).
  {
    const Eigen::Vector3d n_w = R * Eigen::Vector3d::UnitZ();
    planes[0] = {n_w, -n_w.dot(c)};
  }
  // Side planes: each contains the camera centre and two adjacent corner rays; the
  // inward normal is their cross product, sign-fixed against the optical axis.
  const Eigen::Vector3d axis_c(0.0, 0.0, 1.0);
  for (std::size_t i = 0; i < 4; ++i) {
    const Eigen::Vector3d& a = rays_c[i];
    const Eigen::Vector3d& b = rays_c[(i + 1) % 4];
    Eigen::Vector3d n_c = a.cross(b);
    if (n_c.dot(axis_c) < 0.0) n_c = -n_c;  // orient inward (toward the view axis)
    const Eigen::Vector3d n_w = R * n_c;
    planes[i + 1] = {n_w, -n_w.dot(c)};
  }
  return planes;
}

// True when the axis-aligned box [lo, hi] lies entirely on the outside of `pl`,
// i.e. its farthest-inward corner still has n.x + d < 0. Such a box holds no
// in-frustum point and may be skipped.
bool boxOutsidePlane(const Eigen::Vector3d& lo, const Eigen::Vector3d& hi,
                     const WorldPlane& pl) {
  Eigen::Vector3d far;
  for (int k = 0; k < 3; ++k) far[k] = pl.n[k] >= 0.0 ? hi[k] : lo[k];
  return pl.n.dot(far) + pl.d < 0.0;
}

}  // namespace

VisualMapConfig::VisualMapConfig(const FrontendVisual& v) {
  if (v.patch > 0) patch = v.patch;
  if (v.levels > 0) levels = v.levels;
  if (v.grid_cell_px > 0) grid_cell_px = v.grid_cell_px;
  if (v.active_box_m > 0) active_box_m = v.active_box_m;
}

VisualMap::VisualMap(const VisualMapConfig& cfg) : cfg_(cfg) {
  half_patch_ = VisualObservation::kPatch / 2;
  patch_margin_ = half_patch_ + 1;
  cos_add_angle_ = std::cos(cfg_.ref_add_angle_deg * kPi / 180.0);
  cos_converged_angle_ = std::cos(cfg_.ref_converged_angle_deg * kPi / 180.0);
}

VisualMap::VoxelKey VisualMap::voxelKey(const Eigen::Vector3d& p) const {
  const double inv = 1.0 / cfg_.voxel_m;
  VoxelKey k;
  for (int i = 0; i < 3; ++i) {
    k[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(std::floor(p[i] * inv));
  }
  return k;
}

void VisualMap::addPointAt(VoxelKey key, std::int64_t id) {
  auto& ids = index_[key];
  // Keep each cell's id list sorted so iteration is deterministic and erase is a
  // binary search.
  const auto it = std::lower_bound(ids.begin(), ids.end(), id);
  ids.insert(it, id);
}

void VisualMap::removeFromIndex(VoxelKey key, std::int64_t id) {
  const auto cell = index_.find(key);
  if (cell == index_.end()) return;
  auto& ids = cell->second;
  const auto it = std::lower_bound(ids.begin(), ids.end(), id);
  if (it != ids.end() && *it == id) ids.erase(it);
  if (ids.empty()) index_.erase(cell);
}

bool VisualMap::inBoundsCam(const Eigen::Vector2d& uv, int w, int h) const {
  const double mar = static_cast<double>(patch_margin_);
  return uv.x() >= mar && uv.y() >= mar && uv.x() <= static_cast<double>(w - 1) - mar &&
         uv.y() <= static_cast<double>(h - 1) - mar;
}

VisualPoint* VisualMap::pointByIdMutable(std::int64_t id) {
  auto it = points_.find(id);
  return it == points_.end() ? nullptr : it->second.get();
}

const VisualPoint* VisualMap::pointById(std::int64_t id) const {
  const auto it = points_.find(id);
  return it == points_.end() ? nullptr : it->second.get();
}

bool VisualMap::projectWorld(const CameraModel& cam, const Pose& T_w_c,
                             const Eigen::Vector3d& p_world, Eigen::Vector3d* p_cam,
                             Eigen::Vector2d* uv) const {
  const Eigen::Vector3d pc = worldToCam(T_w_c, p_world);
  if (pc.z() <= 0.0) return false;
  Eigen::Vector2d px;
  if (!cam.project(pc, &px)) return false;
  *p_cam = pc;
  *uv = px;
  return true;
}

float VisualMap::gradientScore(const ImagePyramidView& img, const Eigen::Vector2d& uv) const {
  // Shi-Tomasi corner score: the smaller eigenvalue of the gradient structure
  // tensor over the half-patch window. A high score means both image directions
  // carry gradient, i.e. the patch is well localisable for the photometric track.
  double dxx = 0.0;
  double dyy = 0.0;
  double dxy = 0.0;
  for (int x = -half_patch_; x <= half_patch_; ++x) {
    for (int y = -half_patch_; y <= half_patch_; ++y) {
      const Eigen::Vector2d g =
          img.gradient(0, uv + Eigen::Vector2d(static_cast<double>(x), static_cast<double>(y)));
      dxx += g.x() * g.x();
      dyy += g.y() * g.y();
      dxy += g.x() * g.y();
    }
  }
  const double tr = dxx + dyy;
  const double det = dxx * dyy - dxy * dxy;
  const double disc = std::sqrt(std::max(0.0, tr * tr - 4.0 * det));
  const double lambda_min = 0.5 * (tr - disc);
  return static_cast<float>(0.5 * lambda_min);
}

void VisualMap::extractPatches(const ImagePyramidView& img, const Eigen::Vector2d& uv,
                               VisualObservation* obs) const {
  const int levels = std::min(VisualObservation::kLevels, img.levels());
  for (int l = 0; l < VisualObservation::kLevels; ++l) {
    auto& patch = obs->patches[static_cast<std::size_t>(l)];
    if (l >= levels) {
      patch.setZero();
      continue;
    }
    // Patch pixels are spaced one level-pixel apart (2^level level-0 pixels), so a
    // higher level covers a wider scene area at coarser resolution.
    const double step = static_cast<double>(1 << l);
    for (int px = 0; px < VisualObservation::kPatch; ++px) {
      for (int py = 0; py < VisualObservation::kPatch; ++py) {
        const Eigen::Vector2d s =
            uv +
            Eigen::Vector2d((static_cast<double>(px) - static_cast<double>(half_patch_)) * step,
                            (static_cast<double>(py) - static_cast<double>(half_patch_)) * step);
        patch(px, py) = static_cast<float>(img.intensity(l, s));
      }
    }
  }
}

float VisualMap::patchNcc(const VisualObservation& a, const VisualObservation& b) {
  // Zero-mean normalised cross-correlation over the level-0 patches.
  const auto& pa = a.patches[0];
  const auto& pb = b.patches[0];
  const float ma = pa.mean();
  const float mb = pb.mean();
  double num = 0.0;
  double da = 0.0;
  double db = 0.0;
  for (int i = 0; i < VisualObservation::kPatch; ++i) {
    for (int j = 0; j < VisualObservation::kPatch; ++j) {
      const double ca = static_cast<double>(pa(i, j) - ma);
      const double cb = static_cast<double>(pb(i, j) - mb);
      num += ca * cb;
      da += ca * ca;
      db += cb * cb;
    }
  }
  const double denom = std::sqrt(da * db) + 1e-10;
  return static_cast<float>(num / denom);
}

void VisualMap::recomputeRefAndScore(VisualPoint* pt) const {
  // The medoid, per-observation scores, and reference surface are a pure function of
  // the observation set. While that set is unchanged the previous result is still
  // exact, so the O(n^2) re-score is skipped entirely -- the steady-state case where a
  // visible point adds no new view (and every converged point, which never adds one).
  if (!pt->obs_dirty) {
    return;
  }
  const std::size_t n = pt->obs.size();
  if (n == 0) {
    pt->score = 0.f;
    pt->obs_count = 0;
    pt->ncc_cache.clear();
    pt->obs_dirty = false;
    return;
  }
  // NCC of any fixed observation pair is constant (patches are immutable once stored),
  // so the pairwise matrix is rebuilt only here, when the obs set has changed. The
  // matrix is symmetric with a unit diagonal; storing it whole keeps the medoid loop a
  // table lookup instead of an O(n^2) patch correlation every sweep.
  pt->ncc_cache.assign(n * n, 0.f);
  for (std::size_t i = 0; i < n; ++i) {
    pt->ncc_cache[i * n + i] = 1.f;
    for (std::size_t j = i + 1; j < n; ++j) {
      const float c = patchNcc(pt->obs[i], pt->obs[j]);
      pt->ncc_cache[i * n + j] = c;
      pt->ncc_cache[j * n + i] = c;
    }
  }

  // Re-score each observation as w*NCC(to-others) + (1-w)*cos(view-change to set
  // centroid), then pick the medoid: the observation whose viewing direction is
  // most central (maximal summed cosine to the others), which is also the highest-
  // scoring reference. A single observation is trivially its own reference.
  const double w = cfg_.ref_score_w;
  Eigen::Vector3d mean_dir = Eigen::Vector3d::Zero();
  for (const auto& o : pt->obs) mean_dir += o.view_dir;
  if (mean_dir.norm() > 1e-9) mean_dir.normalize();

  float best_score = -std::numeric_limits<float>::max();
  std::size_t best_idx = 0;
  for (std::size_t i = 0; i < n; ++i) {
    double ncc_sum = 0.0;
    int cnt = 0;
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) continue;
      ncc_sum += static_cast<double>(pt->ncc_cache[i * n + j]);
      ++cnt;
    }
    const double ncc = cnt > 0 ? ncc_sum / cnt : 1.0;
    const double cos_central = pt->obs[i].view_dir.dot(mean_dir);
    const float score = static_cast<float>(w * ncc + (1.0 - w) * cos_central);
    pt->obs[i].score = score;
    // Medoid: largest centrality (summed cosine to others), tie-broken by score
    // then by index so the choice is deterministic.
    if (score > best_score) {
      best_score = score;
      best_idx = i;
    }
  }
  pt->ref_index = static_cast<int>(best_idx);
  pt->score = best_score;
  pt->obs_count = static_cast<int>(n);
  const VisualObservation& ref = pt->obs[best_idx];
  pt->ref_patches = ref.patches;
  pt->T_w_c_ref = ref.T_w_c;
  pt->inv_expo_ref = ref.inv_expo;
  pt->obs_dirty = false;
}

int VisualMap::promote(const ImagePyramidView& img, const CameraModel& cam, const Pose& T_w_c,
                       double inv_expo, const std::vector<LidarHit>& hits) {
  if (!cam.valid()) return 0;
  const int cw = cam.width();
  const int ch = cam.height();
  if (cw <= 0 || ch <= 0) return 0;
  const int cell = std::max(1, cfg_.grid_cell_px);
  const int gw = (cw + cell - 1) / cell;
  const int gh = (ch + cell - 1) / cell;
  const std::size_t ncells = static_cast<std::size_t>(gw) * static_cast<std::size_t>(gh);

  // One best candidate per grid cell, selected by gradient score. Storing the hit
  // index (not a pointer) and resolving ties by score then the stable per-point key
  // keeps the winner a pure function of the input, independent of hit order.
  struct Cell {
    int hit = -1;
    float score = -std::numeric_limits<float>::max();
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
  };
  std::vector<Cell> grid(ncells);

  for (std::size_t h = 0; h < hits.size(); ++h) {
    const LidarHit& hit = hits[h];
    // A normal is required for the warp; a hit whose plane fit was rejected has an
    // uninitialized (zero) normal and is not promotable this frame.
    if (!hit.plane.valid || hit.plane.n.squaredNorm() < 1e-9) continue;
    Eigen::Vector3d pc;
    Eigen::Vector2d uv;
    if (!projectWorld(cam, T_w_c, hit.p_world, &pc, &uv)) continue;
    if (!img.inBounds(0, uv, patch_margin_)) continue;
    const int cx = static_cast<int>(uv.x()) / cell;
    const int cy = static_cast<int>(uv.y()) / cell;
    if (cx < 0 || cy < 0 || cx >= gw || cy >= gh) continue;
    const std::size_t idx =
        static_cast<std::size_t>(cy) * static_cast<std::size_t>(gw) + static_cast<std::size_t>(cx);
    const float s = gradientScore(img, uv);
    Cell& c = grid[idx];
    const bool better =
        s > c.score || (s == c.score && c.hit >= 0 &&
                        hit.t_offset_ns < hits[static_cast<std::size_t>(c.hit)].t_offset_ns);
    if (c.hit < 0 || better) {
      c.hit = static_cast<int>(h);
      c.score = s;
      c.uv = uv;
    }
  }

  int added = 0;
  for (std::size_t i = 0; i < ncells; ++i) {
    const Cell& c = grid[i];
    if (c.hit < 0) continue;
    const LidarHit& hit = hits[static_cast<std::size_t>(c.hit)];

    auto pt = std::make_unique<VisualPoint>();
    pt->p_world = hit.p_world;
    // Orient the normal toward the camera so the warp does not mirror the patch.
    Eigen::Vector3d n = hit.plane.n.normalized();
    if (n.dot(hit.p_world - T_w_c.t) > 0.0) n = -n;
    pt->n_world = n;
    pt->normal_initialized = true;
    pt->id = next_id_++;

    VisualObservation obs;
    extractPatches(img, c.uv, &obs);
    obs.T_w_c = T_w_c;
    obs.inv_expo = inv_expo;
    Eigen::Vector3d dir = T_w_c.t - hit.p_world;
    obs.view_dir = dir.norm() > 1e-9 ? dir.normalized() : Eigen::Vector3d::UnitZ();
    obs.score = c.score;
    pt->obs.push_back(obs);
    recomputeRefAndScore(pt.get());

    const VoxelKey key = voxelKey(pt->p_world);
    const std::int64_t id = pt->id;
    points_.emplace(id, std::move(pt));
    addPointAt(key, id);
    ++added;
  }
  return added;
}

std::vector<std::int64_t> VisualMap::frustumCandidateIds(const CameraModel& cam,
                                                         const Pose& T_w_c) const {
  std::vector<std::int64_t> ids;
  const std::array<WorldPlane, 5> planes = frustumPlanes(cam, T_w_c);
  const double vm = cfg_.voxel_m;
  for (const auto& [key, cell_ids] : index_) {
    // World-space box of this voxel from its integer key and the index pitch.
    const Eigen::Vector3d lo(static_cast<double>(key[0]) * vm, static_cast<double>(key[1]) * vm,
                             static_cast<double>(key[2]) * vm);
    const Eigen::Vector3d hi = lo + Eigen::Vector3d::Constant(vm);
    bool outside = false;
    for (const auto& pl : planes) {
      if (boxOutsidePlane(lo, hi, pl)) {
        outside = true;
        break;
      }
    }
    if (outside) continue;
    ids.insert(ids.end(), cell_ids.begin(), cell_ids.end());
  }
  // Each cell's ids are already ascending and cells are visited key-ascending, but
  // different voxels interleave id ranges, so a global sort restores the exact
  // ascending-id order the whole-map scan relied on for its occlusion tie-break.
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::vector<std::int64_t> VisualMap::selectVisibleIds(const CameraModel& cam,
                                                      const Pose& T_w_c) const {
  return selectVisibleFromCandidates(cam, T_w_c, frustumCandidateIds(cam, T_w_c));
}

std::vector<std::int64_t> VisualMap::selectVisibleFromCandidates(
    const CameraModel& cam, const Pose& T_w_c,
    const std::vector<std::int64_t>& candidate_ids) const {
  std::vector<std::int64_t> out;
  if (!cam.valid()) return out;
  const int cw = cam.width();
  const int ch = cam.height();
  if (cw <= 0 || ch <= 0) return out;
  const int cell = std::max(1, cfg_.grid_cell_px);
  const int gw = (cw + cell - 1) / cell;
  const int gh = (ch + cell - 1) / cell;
  const std::size_t ncells = static_cast<std::size_t>(gw) * static_cast<std::size_t>(gh);

  // Per-cell winner is the nearest in-frustum point (mirrors the depth-map nearest
  // selection): the closest surface occludes the farther ones. We track the winning
  // id, its rounded projection pixel, and its camera-frame depth per cell.
  struct Cell {
    std::int64_t id = -1;
    double depth = std::numeric_limits<double>::max();
    int px = 0;
    int py = 0;
  };
  std::vector<Cell> grid(ncells);

  // Sparse per-pixel depth image: every in-frustum point splats its camera-frame z
  // into its rounded pixel (nearest wins so the closest surface owns the pixel). The
  // depth-continuity gate scans this buffer at patch granularity, detecting an
  // occlusion edge crossing a candidate's own 8-pixel footprint rather than only the
  // 32-pixel cell-grid boundaries. Zero marks an unfilled pixel.
  std::vector<float> depth_px(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch), 0.0f);

  // Per-cell nearest-point pass: each in-frustum point claims its cell only if it is
  // closer than the current occupant, so the closest surface wins (occlusion order).
  // Candidates are supplied ascending-id (the frustum subset in production), so the
  // winner is a deterministic function of the input -- identical to a whole-map scan
  // but bounded to what the camera can see rather than the full point count.
  for (const std::int64_t id : candidate_ids) {
    const VisualPoint* pt = pointById(id);
    if (pt == nullptr || !pt->normal_initialized) continue;
    Eigen::Vector3d pc;
    Eigen::Vector2d uv;
    if (!projectWorld(cam, T_w_c, pt->p_world, &pc, &uv)) continue;
    if (!inBoundsCam(uv, cw, ch)) continue;
    const int upx = static_cast<int>(uv.x());
    const int upy = static_cast<int>(uv.y());
    const std::size_t pidx =
        static_cast<std::size_t>(upy) * static_cast<std::size_t>(cw) + static_cast<std::size_t>(upx);
    const float z = static_cast<float>(pc.z());
    if (depth_px[pidx] == 0.0f || z < depth_px[pidx]) depth_px[pidx] = z;
    const int cx = upx / cell;
    const int cy = upy / cell;
    if (cx < 0 || cy < 0 || cx >= gw || cy >= gh) continue;
    const std::size_t idx =
        static_cast<std::size_t>(cy) * static_cast<std::size_t>(gw) + static_cast<std::size_t>(cx);
    Cell& c = grid[idx];
    if (pc.z() < c.depth) {
      c.depth = pc.z();
      c.id = pt->id;
      c.px = upx;
      c.py = upy;
    }
  }

  for (std::size_t i = 0; i < ncells; ++i) {
    if (grid[i].id < 0) continue;
    // Depth-continuity: scan the +/- half-patch pixel neighbourhood of the
    // candidate's own projection and reject if any filled pixel's depth differs from
    // the candidate's camera-frame z by more than the gate, i.e. the patch straddles
    // an occlusion edge.
    const Cell& c = grid[i];
    bool occluded = false;
    for (int v = -half_patch_; v <= half_patch_ && !occluded; ++v) {
      for (int u = -half_patch_; u <= half_patch_ && !occluded; ++u) {
        if (u == 0 && v == 0) continue;
        const int nx = c.px + u;
        const int ny = c.py + v;
        if (nx < 0 || ny < 0 || nx >= cw || ny >= ch) continue;
        const float d = depth_px[static_cast<std::size_t>(ny) * static_cast<std::size_t>(cw) +
                                 static_cast<std::size_t>(nx)];
        if (d == 0.0f) continue;
        if (std::abs(c.depth - static_cast<double>(d)) > cfg_.depth_continuity_m) {
          occluded = true;
        }
      }
    }
    if (occluded) continue;
    out.push_back(grid[i].id);
  }
  return out;
}

std::vector<const VisualPoint*> VisualMap::visibleCandidates(const CameraModel& cam,
                                                             const Pose& T_w_c) const {
  std::vector<const VisualPoint*> out;
  for (std::int64_t id : selectVisibleIds(cam, T_w_c)) {
    const VisualPoint* pt = pointById(id);
    if (pt != nullptr) out.push_back(pt);
  }
  return out;
}

void VisualMap::refreshVisibleCache(const CameraModel& cam, const Pose& T_w_c) {
  visible_cache_ids_ = selectVisibleIds(cam, T_w_c);
}

std::vector<const VisualPoint*> VisualMap::cachedVisibleCandidates() const {
  std::vector<const VisualPoint*> out;
  out.reserve(visible_cache_ids_.size());
  for (std::int64_t id : visible_cache_ids_) {
    const VisualPoint* pt = pointById(id);
    if (pt != nullptr) out.push_back(pt);
  }
  return out;
}

void VisualMap::updateAfterSolve(const ImagePyramidView& img, const CameraModel& cam,
                                 const Pose& T_w_c, double inv_expo) {
  if (!cam.valid()) return;
  const double cos_obs_add = cos_add_angle_;
  // Refine only the points the camera used this frame (the seed-pose visible cache,
  // shared with the residual build), not every in-frustum point: the heavy per-point
  // work (extractPatches, NCC, medoid re-score) must be bounded by the visible set,
  // and reusing the cache avoids a second O(map) selection scan this sweep.
  for (std::int64_t vid : visible_cache_ids_) {
    VisualPoint* pt = pointByIdMutable(vid);
    if (pt == nullptr || !pt->normal_initialized) continue;
    Eigen::Vector3d pc;
    Eigen::Vector2d uv;
    if (!projectWorld(cam, T_w_c, pt->p_world, &pc, &uv)) continue;
    if (!img.inBounds(0, uv, patch_margin_)) continue;

    Eigen::Vector3d dir = T_w_c.t - pt->p_world;
    if (dir.norm() < 1e-9) continue;
    dir.normalize();

    // Converged points never grow their observation list, but they are still
    // re-scored so a decorrelating reference can lose its medoid status.
    if (!pt->converged) {
      // Add gate: a new observation is stored only if it adds parallax (view angle
      // differs from every existing observation by >= ref_add_angle_deg) or its
      // score clearly improves on the current best.
      bool novel_view = true;
      for (const auto& o : pt->obs) {
        if (o.view_dir.dot(dir) > cos_obs_add) {
          novel_view = false;
          break;
        }
      }
      VisualObservation cand;
      extractPatches(img, uv, &cand);
      cand.T_w_c = T_w_c;
      cand.inv_expo = inv_expo;
      cand.view_dir = dir;
      // Provisional score vs the current reference for the score-improvement branch.
      cand.score =
          pt->obs.empty() ? 1.f : patchNcc(cand, pt->obs[static_cast<std::size_t>(pt->ref_index)]);
      const bool score_improves = cand.score > pt->score + static_cast<float>(cfg_.ref_add_score);
      if (novel_view || score_improves) {
        if (static_cast<int>(pt->obs.size()) >= cfg_.ref_obs_cap) {
          // Min-score eviction: drop the lowest-scoring observation so the cap
          // retains the most informative spread.
          std::size_t worst = 0;
          float worst_score = std::numeric_limits<float>::max();
          for (std::size_t k = 0; k < pt->obs.size(); ++k) {
            if (pt->obs[k].score < worst_score) {
              worst_score = pt->obs[k].score;
              worst = k;
            }
          }
          pt->obs.erase(pt->obs.begin() + static_cast<std::ptrdiff_t>(worst));
        }
        pt->obs.push_back(cand);
        // The obs set changed, so the cached NCC matrix and medoid are stale.
        pt->obs_dirty = true;
      }
    }

    recomputeRefAndScore(pt);

    // Converged latch: enough observations spanning enough view change, with a
    // stable medoid. View span is measured as the max pairwise angle: if any
    // observation pair differs by >= ref_converged_angle_deg the span is met.
    if (!pt->converged && static_cast<int>(pt->obs.size()) >= cfg_.ref_converged_obs) {
      bool span_met = false;
      for (std::size_t a = 0; a < pt->obs.size() && !span_met; ++a) {
        for (std::size_t b = a + 1; b < pt->obs.size(); ++b) {
          if (pt->obs[a].view_dir.dot(pt->obs[b].view_dir) < cos_converged_angle_) {
            span_met = true;
            break;
          }
        }
      }
      if (span_met) pt->converged = true;
    }
  }
}

void VisualMap::evict(const Eigen::Vector3d& p_world_now) {
  const double box = cfg_.active_box_m;
  const bool score_evict = cfg_.min_score_keep > 0.0;
  for (auto it = points_.begin(); it != points_.end();) {
    const VisualPoint* pt = it->second.get();
    const Eigen::Vector3d d = pt->p_world - p_world_now;
    const bool out_of_box = std::abs(d.x()) > box || std::abs(d.y()) > box || std::abs(d.z()) > box;
    const bool below_score = score_evict && pt->score < static_cast<float>(cfg_.min_score_keep);
    if (out_of_box || below_score) {
      removeFromIndex(voxelKey(pt->p_world), pt->id);
      // Erasing frees the RAII-owned observation list immediately, so memory tracks
      // the live set rather than the lifetime total of promotions.
      it = points_.erase(it);
    } else {
      ++it;
    }
  }
}

void VisualMap::transform(const Pose& delta) {
  const Eigen::Matrix3d dR = delta.q.toRotationMatrix();
  for (auto& [id, pt] : points_) {
    pt->p_world = dR * pt->p_world + delta.t;
    pt->n_world = dR * pt->n_world;  // unit normal rotates; a zero (uninitialised) stays zero
    pt->T_w_c_ref = delta * pt->T_w_c_ref;
    for (VisualObservation& ob : pt->obs) {
      ob.T_w_c = delta * ob.T_w_c;
      ob.view_dir = dR * ob.view_dir;
    }
  }
  // Voxel keys are derived from p_world, so the whole spatial index is stale: rebuild it
  // from the shifted positions. Iterating the id-ordered point map keeps each cell's id
  // list ascending, matching how the production path builds it.
  index_.clear();
  for (const auto& [id, pt] : points_) {
    addPointAt(voxelKey(pt->p_world), id);
  }
  // The per-sweep visible set was selected at the old poses; drop it so the next sweep
  // reselects against the shifted map.
  visible_cache_ids_.clear();
}

VisualMap::StateSummary VisualMap::summary() const {
  // Fold a stable digest over the live points (the ordered map yields them id-
  // ascending) using rounded fields, so the summary is a pure function of state and
  // independent of any insertion/eviction history that produced it.
  StateSummary s;
  s.next_id = next_id_;
  std::uint64_t hash = 1469598103934665603ULL;  // FNV-1a offset basis
  const auto mix = [&hash](std::uint64_t v) {
    hash ^= v;
    hash *= 1099511628211ULL;  // FNV-1a prime
  };
  const auto mix_d = [&mix](double v) {
    // Round to a fixed grid so floating-point noise below the grid never flips the
    // digest; identical inputs round identically.
    const std::int64_t q = static_cast<std::int64_t>(std::llround(v * 1e6));
    mix(static_cast<std::uint64_t>(q));
  };
  for (const auto& [id, uptr] : points_) {
    const VisualPoint* pt = uptr.get();
    ++s.count;
    mix(static_cast<std::uint64_t>(pt->id));
    mix_d(pt->p_world.x());
    mix_d(pt->p_world.y());
    mix_d(pt->p_world.z());
    mix_d(pt->n_world.x());
    mix_d(pt->n_world.y());
    mix_d(pt->n_world.z());
    mix(static_cast<std::uint64_t>(pt->obs_count));
    mix(static_cast<std::uint64_t>(pt->ref_index));
    mix(pt->converged ? 1ULL : 0ULL);
    mix_d(static_cast<double>(pt->score));
  }
  s.hash = hash;
  return s;
}

}  // namespace meridian::ct
