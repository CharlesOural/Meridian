#include "meridian/map/keyframe_store.hpp"

namespace meridian {

Aabb KeyframeStore::compute_bounds(const PointCloudPtr& cloud, const Pose& T_map_body) {
  Aabb b;
  if (!cloud) return b;
  for (const LidarPoint& p : *cloud) {
    b.expand((T_map_body * p.xyz.cast<double>()).cast<float>());
  }
  return b;
}

void KeyframeStore::put(std::uint64_t id, PointCloudPtr cloud,
                        std::shared_ptr<const CameraFrame> image, const Pose& T_init_map_body,
                        const Pose& T_body_cam) {
  Aabb bounds = compute_bounds(cloud, T_init_map_body);
  std::lock_guard<std::mutex> lock(mu_);
  entries_[id] =
      Entry{std::move(cloud), std::move(image), T_init_map_body, T_body_cam, bounds};
}

PointCloudPtr KeyframeStore::cloud(std::uint64_t id) const {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = entries_.find(id);
  return it == entries_.end() ? nullptr : it->second.cloud;
}

std::shared_ptr<const CameraFrame> KeyframeStore::image(std::uint64_t id) const {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = entries_.find(id);
  return it == entries_.end() ? nullptr : it->second.image;
}

std::optional<Pose> KeyframeStore::pose(std::uint64_t id) const {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = entries_.find(id);
  if (it == entries_.end()) return std::nullopt;
  return it->second.T_map_body;
}

std::optional<Pose> KeyframeStore::body_cam(std::uint64_t id) const {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = entries_.find(id);
  if (it == entries_.end()) return std::nullopt;
  return it->second.T_body_cam;
}

std::optional<Aabb> KeyframeStore::bounds(std::uint64_t id) const {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = entries_.find(id);
  if (it == entries_.end()) return std::nullopt;
  return it->second.bounds_map;
}

void KeyframeStore::update_pose(std::uint64_t id, const Pose& T_map_body) {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = entries_.find(id);
  if (it == entries_.end()) return;
  it->second.T_map_body = T_map_body;
  it->second.bounds_map = compute_bounds(it->second.cloud, T_map_body);
}

std::vector<std::uint64_t> KeyframeStore::ids() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::uint64_t> out;
  out.reserve(entries_.size());
  for (const auto& [id, e] : entries_) out.push_back(id);
  return out;  // ascending by std::map ordering
}

std::vector<std::uint64_t> KeyframeStore::within_radius(const Eigen::Vector3f& c, float r) const {
  std::lock_guard<std::mutex> lock(mu_);
  const float r2 = r * r;
  std::vector<std::uint64_t> out;
  for (const auto& [id, e] : entries_) {
    const Eigen::Vector3f d = e.T_map_body.t.cast<float>() - c;
    if (d.squaredNorm() <= r2) out.push_back(id);
  }
  return out;  // ascending id by std::map ordering
}

std::vector<std::uint64_t> KeyframeStore::in_region(const Aabb& region) const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::uint64_t> out;
  for (const auto& [id, e] : entries_) {
    if (region.intersects(e.bounds_map)) out.push_back(id);
  }
  return out;  // ascending id by std::map ordering
}

std::size_t KeyframeStore::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return entries_.size();
}

bool KeyframeStore::contains(std::uint64_t id) const {
  std::lock_guard<std::mutex> lock(mu_);
  return entries_.find(id) != entries_.end();
}

}  // namespace meridian
