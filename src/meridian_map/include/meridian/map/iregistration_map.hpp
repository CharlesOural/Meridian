#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "meridian/common/cloud.hpp"
#include "meridian/common/map_types.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

// A span of map-frame points (the live registered scan fed back for the next sweep).
using PointCloudMap = std::vector<Eigen::Vector3f>;

// Tier R: the hot, local, CPU-resident registration map the front-end queries every
// scan. Cached per-voxel planes make the per-point query a hash lookup, not a k-NN +
// PCA. Safe for concurrent read (front-end) while the map thread inserts.
class IRegistrationMap {
 public:
  virtual ~IRegistrationMap() = default;

  // Best-effort scratch path: insert the just-registered live scan so the next sweep
  // has something dense to register against. Not recorded in the KeyframeStore.
  virtual void integrate_live(const PointCloudMap& pts_map, Timestamp t) = 0;

  // Canonical path: insert a keyframe's body-frame cloud at its map pose and record the
  // (id -> touched voxels) provenance a region rebuild needs.
  virtual void integrate_keyframe(std::uint64_t id, const PointCloud& cloud_body,
                                  const Pose& T_map_body) = 0;

  // Hot path (front-end thread, CPU only): cached plane of the voxel containing p_map,
  // or the best planar one-ring neighbour; nullopt if none.
  virtual std::optional<PlaneHit> query_plane(const Eigen::Vector3f& p_map) const = 0;

  // Erase the voxels the given keyframes contributed, and forget their provenance.
  virtual void clear_keyframes(const std::vector<std::uint64_t>& ids) = 0;

  // Bound the map to an AABB around the robot (FOV segmentation); prune voxels outside.
  virtual void set_hot_window(const Aabb& hot) = 0;

  virtual MapDiagnostics diagnostics() const = 0;
};

}  // namespace meridian
