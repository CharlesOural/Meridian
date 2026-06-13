#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "meridian/common/cloud.hpp"
#include "meridian/common/map_types.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"

namespace meridian {

// The retained per-keyframe point-cloud store: the canonical source of truth for L4.
// It owns nothing the KeyframePacket did not already own — it holds the SAME
// Shared-immutable cloud/image handles, keyed by id, for the mission's duration. The
// derived map tiers (registration + surface) are rebuildable from this store, so a
// loop correction clears the affected region and re-integrates these clouds at their
// corrected poses. L5 reads geometry from it; L4 re-integration reads it too. The pose
// is the single mutable field (update_pose, on a GraphUpdate); the cloud bytes are
// immutable. Each entry caches the map-frame AABB of its cloud so in_region() can find
// the keyframes overlapping a corrected region without rescanning every point.
class KeyframeStore {
 public:
  // First sight of a keyframe. bounds_map is derived from the cloud + pose here.
  void put(std::uint64_t id, PointCloudPtr cloud, std::shared_ptr<const CameraFrame> image,
           const Pose& T_init_map_body, const Pose& T_body_cam = Pose{});

  PointCloudPtr cloud(std::uint64_t id) const;
  std::shared_ptr<const CameraFrame> image(std::uint64_t id) const;
  std::optional<Pose> pose(std::uint64_t id) const;
  std::optional<Pose> body_cam(std::uint64_t id) const;
  std::optional<Aabb> bounds(std::uint64_t id) const;

  // Move a keyframe to its corrected pose (GraphUpdate); recomputes bounds_map so
  // in_region stays correct after a loop.
  void update_pose(std::uint64_t id, const Pose& T_map_body);

  // All stored keyframe ids, ascending.
  std::vector<std::uint64_t> ids() const;

  // Keyframe ids whose stored pose translation lies within `r` metres of `c`.
  std::vector<std::uint64_t> within_radius(const Eigen::Vector3f& c, float r) const;

  // Keyframe ids whose cached map-frame bounds intersect `region`, ascending. The
  // rebuild superset: every keyframe whose cloud could touch a cleared region.
  std::vector<std::uint64_t> in_region(const Aabb& region) const;

  std::size_t size() const;
  bool contains(std::uint64_t id) const;

 private:
  struct Entry {
    PointCloudPtr cloud;
    std::shared_ptr<const CameraFrame> image;
    Pose T_map_body;
    Pose T_body_cam;
    Aabb bounds_map;
  };

  // Map-frame AABB of a body-frame cloud placed at T_map_body.
  static Aabb compute_bounds(const PointCloudPtr& cloud, const Pose& T_map_body);

  mutable std::mutex mu_;
  std::map<std::uint64_t, Entry> entries_;  // ordered by id for deterministic iteration
};

}  // namespace meridian
