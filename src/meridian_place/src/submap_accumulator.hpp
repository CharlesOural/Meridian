#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/map/keyframe_store.hpp"

namespace meridian {

// Composes the last-N keyframe clouds ending at an anchor into the anchor's body frame
// using the back-end's corrected relative poses, voxel-downsampled, so retrieval and
// verification see dense local geometry instead of one thin sweep. Anchor-keyed LRU;
// an entry is dropped when a correction moves any keyframe inside its window, so the next
// use recomposes against the current corrected geometry. The submap is geometry only — the
// emitted loop stays between two single keyframe pose variables.
class SubmapAccumulator {
 public:
  using PoseLookup = std::function<std::optional<Pose>(std::uint64_t)>;

  SubmapAccumulator(std::shared_ptr<const KeyframeStore> store, PoseLookup pose_of,
                    int window, double voxel, int cache);

  // The submap anchored at `anchor` (cached). Null if the anchor has no corrected pose yet.
  std::shared_ptr<const PointCloud> submap(std::uint64_t anchor) const;

  // Drop cached submaps whose window includes any moved keyframe.
  void invalidate(const std::vector<std::uint64_t>& moved_ids);
  void clear();

 private:
  std::shared_ptr<const PointCloud> compose(std::uint64_t anchor,
                                            std::vector<std::uint64_t>& window_ids) const;
  void touch(std::uint64_t anchor) const;

  std::shared_ptr<const KeyframeStore> store_;
  PoseLookup pose_of_;
  int window_;
  double voxel_;
  std::size_t capacity_;  // max number of cached submaps

  struct CacheEntry {
    std::shared_ptr<const PointCloud> cloud;
    std::vector<std::uint64_t> window_ids;  // ids composed, for move-invalidation
  };
  mutable std::map<std::uint64_t, CacheEntry> cache_;  // anchor -> composed submap
  mutable std::list<std::uint64_t> lru_;               // front = most-recently used
};

}  // namespace meridian
