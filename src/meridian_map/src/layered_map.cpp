#include "layered_map.hpp"

#include <algorithm>
#include <utility>

namespace meridian::map {

LayeredMap::LayeredMap(const MapConfig& cfg, std::shared_ptr<const CalibrationSet> calib,
                       std::shared_ptr<KeyframeStore> store, std::unique_ptr<IRegistrationMap> reg,
                       std::unique_ptr<ISurfaceMap> surf, TelemetrySink* telemetry)
    : cfg_(cfg),
      calib_(std::move(calib)),
      store_(std::move(store)),
      reg_(std::move(reg)),
      surf_(std::move(surf)),
      sink_(telemetry) {}

void LayeredMap::integrate_id(std::uint64_t id, const Pose& T_map_body) {
  const PointCloudPtr cloud = store_->cloud(id);
  if (!cloud) return;
  reg_->integrate_keyframe(id, *cloud, T_map_body);
  const Pose tbc = store_->body_cam(id).value_or(Pose{});
  surf_->integrate(id, *cloud, store_->image(id), T_map_body, tbc);
}

void LayeredMap::integrate(const KeyframePacket& kf, const Pose& T_map_body) {
  // Store first (canonical), then the derived tiers (caches).
  store_->put(kf.id, kf.cloud_body, kf.image, T_map_body, kf.T_body_cam);
  if (!kf.cloud_body) return;
  reg_->integrate_keyframe(kf.id, *kf.cloud_body, T_map_body);
  surf_->integrate(kf.id, *kf.cloud_body, kf.image, T_map_body, kf.T_body_cam);
}

void LayeredMap::apply_graph_update(const GraphUpdate& update) {
  if (update.moved.empty()) return;

  // The region to rebuild spans the moved keyframes' OLD and NEW footprints: the old
  // location's voxels must be cleared, the new location's rebuilt.
  Aabb region;
  for (const auto& m : update.moved) {
    if (auto b = store_->bounds(m.id)) region.expand(*b);  // old footprint
  }
  for (const auto& m : update.moved) {
    store_->update_pose(m.id, m.new_T_map_body);
    if (auto b = store_->bounds(m.id)) region.expand(*b);  // new footprint
  }
  if (region.empty()) return;

  // Every keyframe overlapping the region is cleared and rebuilt, so each affected voxel
  // is re-fused from exactly its contributors (no double counting), at corrected poses.
  std::vector<std::uint64_t> rebuild = store_->in_region(region);
  std::sort(rebuild.begin(), rebuild.end());  // deterministic, ascending id

  reg_->clear_keyframes(rebuild);
  surf_->clear_keyframes(rebuild);
  for (std::uint64_t id : rebuild) {
    if (auto p = store_->pose(id)) integrate_id(id, *p);
  }
  last_rebuild_kf_ = rebuild.size();
}

std::optional<PlaneHit> LayeredMap::query_plane(const Eigen::Vector3f& p_map) const {
  return reg_->query_plane(p_map);
}

const ColorMesh& LayeredMap::extract_mesh() { return surf_->extract_mesh(); }

MapDiagnostics LayeredMap::diagnostics() const {
  MapDiagnostics d = reg_->diagnostics();
  const MapDiagnostics s = surf_->diagnostics();
  d.tsdf_blocks = s.tsdf_blocks;
  d.last_integrate_ms = s.last_integrate_ms;
  d.last_mesh_ms = s.last_mesh_ms;
  d.kf_count = store_->size();
  d.last_rebuild_kf = last_rebuild_kf_;
  return d;
}

}  // namespace meridian::map
