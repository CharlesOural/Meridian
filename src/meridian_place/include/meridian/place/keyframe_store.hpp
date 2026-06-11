#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "meridian/common/cloud.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"

namespace meridian {

// The retained per-keyframe point-cloud store. It owns nothing the KeyframePacket did
// not already own: it holds the SAME Shared-immutable cloud/image handles, keyed by id,
// for the mission's duration. L5 reads geometry from it; L4 re-integration will read it
// too. Pose snapshots index keyframes spatially so a moved region can find the clouds
// that touch it.
class KeyframeStore {
 public:
  void put(std::uint64_t id, PointCloudPtr cloud, std::shared_ptr<const CameraFrame> image,
           const Pose& T_init_map_body);

  PointCloudPtr cloud(std::uint64_t id) const;
  std::shared_ptr<const CameraFrame> image(std::uint64_t id) const;
  std::optional<Pose> pose(std::uint64_t id) const;

  // Keyframe ids whose stored pose translation lies within `r` metres of `c`.
  std::vector<std::uint64_t> within_radius(const Eigen::Vector3f& c, float r) const;

  std::size_t size() const;
  bool contains(std::uint64_t id) const;

 private:
  struct Entry {
    PointCloudPtr cloud;
    std::shared_ptr<const CameraFrame> image;
    Pose T_map_body;
  };

  mutable std::mutex mu_;
  std::map<std::uint64_t, Entry> entries_;  // ordered by id for deterministic iteration
};

}  // namespace meridian
