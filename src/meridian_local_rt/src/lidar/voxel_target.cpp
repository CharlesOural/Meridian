#include "meridian/local_rt/lidar/voxel_target.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace meridian::local_rt::lidar {
namespace {

bool lexicographicallyLess(const Point3d& lhs, const Point3d& rhs) noexcept {
  if (lhs.x() != rhs.x()) {
    return lhs.x() < rhs.x();
  }
  if (lhs.y() != rhs.y()) {
    return lhs.y() < rhs.y();
  }
  return lhs.z() < rhs.z();
}

bool optionsAreValid(const VoxelTargetOptions& options) noexcept {
  return std::isfinite(options.voxel_size_m) && options.voxel_size_m > 0.0 &&
         std::isfinite(options.retention_radius_m) && options.retention_radius_m > 0.0 &&
         options.max_voxels > 0U && options.max_points_per_voxel > 0U &&
         std::isfinite(options.minimum_point_spacing_m) && options.minimum_point_spacing_m >= 0.0 &&
         options.max_neighbor_voxel_radius > 0U &&
         options.max_neighbor_voxel_radius <=
             static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
}

std::optional<std::int64_t> checkedOffset(std::int64_t value, std::int64_t offset) noexcept {
  if ((offset > 0 && value > std::numeric_limits<std::int64_t>::max() - offset) ||
      (offset < 0 && value < std::numeric_limits<std::int64_t>::min() - offset)) {
    return std::nullopt;
  }
  return value + offset;
}

double closestSquaredDistance(const PointCloud& points, const Point3d& location) noexcept {
  double closest = std::numeric_limits<double>::infinity();
  for (const Point3d& point : points) {
    closest = std::min(closest, (point - location).squaredNorm());
  }
  return closest;
}

struct CandidatePoint final {
  VoxelKey key;
  Point3d point{Point3d::Zero()};
  double squared_retention_distance{};
};

using CandidatePoints = std::vector<CandidatePoint, Eigen::aligned_allocator<CandidatePoint>>;

double minimumAxisDistanceVoxelUnits(std::int64_t offset) noexcept {
  if (offset > 1) {
    return static_cast<double>(offset - 1);
  }
  if (offset < -1) {
    return static_cast<double>(-offset - 1);
  }
  return 0.0;
}

long double squaredOffsetNorm(std::int64_t x, std::int64_t y, std::int64_t z) noexcept {
  const long double long_x = static_cast<long double>(x);
  const long double long_y = static_cast<long double>(y);
  const long double long_z = static_cast<long double>(z);
  return long_x * long_x + long_y * long_y + long_z * long_z;
}

}  // namespace

BoundedVoxelTarget::BoundedVoxelTarget(VoxelTargetOptions options) : options_(options) {
  if (!optionsAreValid(options_)) {
    throw std::invalid_argument(
        "VoxelTargetOptions require finite positive voxel/retention sizes, positive capacities "
        "and query radius, and finite non-negative point spacing");
  }
  const std::int64_t radius = static_cast<std::int64_t>(options_.max_neighbor_voxel_radius);
  const std::size_t side = 2U * options_.max_neighbor_voxel_radius + 1U;
  neighbor_offsets_.reserve(side * side * side);
  for (std::int64_t dx = -radius; dx <= radius; ++dx) {
    for (std::int64_t dy = -radius; dy <= radius; ++dy) {
      for (std::int64_t dz = -radius; dz <= radius; ++dz) {
        const double lower_x = minimumAxisDistanceVoxelUnits(dx);
        const double lower_y = minimumAxisDistanceVoxelUnits(dy);
        const double lower_z = minimumAxisDistanceVoxelUnits(dz);
        neighbor_offsets_.push_back(
            {dx, dy, dz, lower_x * lower_x + lower_y * lower_y + lower_z * lower_z});
      }
    }
  }
  std::sort(
      neighbor_offsets_.begin(), neighbor_offsets_.end(),
      [](const NeighborOffset& lhs, const NeighborOffset& rhs) {
        if (lhs.minimum_squared_distance_voxel_units != rhs.minimum_squared_distance_voxel_units) {
          return lhs.minimum_squared_distance_voxel_units <
                 rhs.minimum_squared_distance_voxel_units;
        }
        const long double lhs_norm = squaredOffsetNorm(lhs.x, lhs.y, lhs.z);
        const long double rhs_norm = squaredOffsetNorm(rhs.x, rhs.y, rhs.z);
        if (lhs_norm != rhs_norm) {
          return lhs_norm < rhs_norm;
        }
        if (lhs.x != rhs.x) {
          return lhs.x < rhs.x;
        }
        if (lhs.y != rhs.y) {
          return lhs.y < rhs.y;
        }
        return lhs.z < rhs.z;
      });
  voxels_.reserve(options_.max_voxels);
}

void BoundedVoxelTarget::clear() noexcept {
  voxels_.clear();
  point_count_ = 0U;
}

VoxelTargetUpdateStats BoundedVoxelTarget::update(std::span<const Point3d> points_source,
                                                  const Sophus::SE3d& T_target_source,
                                                  const Point3d& retention_center_target) {
  if (!T_target_source.matrix().allFinite()) {
    throw std::invalid_argument("target-source transform must be finite");
  }

  PointCloud transformed;
  transformed.reserve(points_source.size());
  std::size_t rejected_nonfinite = 0U;
  for (const Point3d& point : points_source) {
    if (!point.allFinite()) {
      ++rejected_nonfinite;
      continue;
    }
    Point3d transformed_point = T_target_source * point;
    if (!transformed_point.allFinite()) {
      ++rejected_nonfinite;
      continue;
    }
    transformed.push_back(std::move(transformed_point));
  }

  VoxelTargetUpdateStats stats = updateTargetFrame(transformed, retention_center_target);
  stats.input_points = points_source.size();
  stats.finite_points = transformed.size();
  stats.rejected_nonfinite_points += rejected_nonfinite;
  return stats;
}

VoxelTargetUpdateStats BoundedVoxelTarget::updateTargetFrame(
    std::span<const Point3d> points_target, const Point3d& retention_center_target) {
  if (!retention_center_target.allFinite()) {
    throw std::invalid_argument("retention centre must be finite");
  }

  VoxelTargetUpdateStats stats;
  stats.input_points = points_target.size();
  prune(retention_center_target, stats);

  CandidatePoints candidates;
  candidates.reserve(points_target.size());
  const double retention_radius_squared = options_.retention_radius_m * options_.retention_radius_m;
  for (const Point3d& point : points_target) {
    if (!point.allFinite()) {
      ++stats.rejected_nonfinite_points;
      continue;
    }
    ++stats.finite_points;
    const double squared_retention_distance = (point - retention_center_target).squaredNorm();
    if (squared_retention_distance > retention_radius_squared) {
      ++stats.rejected_retention_points;
      continue;
    }
    candidates.push_back(CandidatePoint{
        .key = pointToVoxelKey(point, options_.voxel_size_m),
        .point = point,
        .squared_retention_distance = squared_retention_distance,
    });
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const CandidatePoint& lhs, const CandidatePoint& rhs) {
              if (lhs.squared_retention_distance != rhs.squared_retention_distance) {
                return lhs.squared_retention_distance < rhs.squared_retention_distance;
              }
              if (lhs.key != rhs.key) {
                return lhs.key < rhs.key;
              }
              return lexicographicallyLess(lhs.point, rhs.point);
            });

  const double minimum_spacing_squared =
      options_.minimum_point_spacing_m * options_.minimum_point_spacing_m;
  for (const CandidatePoint& candidate : candidates) {
    auto found = voxels_.find(candidate.key);
    if (found == voxels_.end()) {
      if (!makeRoomFor(candidate.key, candidate.point, retention_center_target, stats)) {
        ++stats.rejected_target_capacity_points;
        continue;
      }
      PointCloud bucket;
      bucket.reserve(options_.max_points_per_voxel);
      found = voxels_.emplace(candidate.key, std::move(bucket)).first;
    }

    PointCloud& bucket = found->second;
    if (bucket.size() >= options_.max_points_per_voxel) {
      ++stats.rejected_voxel_capacity_points;
      continue;
    }
    if (minimum_spacing_squared > 0.0 &&
        std::any_of(bucket.cbegin(), bucket.cend(), [&](const Point3d& incumbent) {
          return (incumbent - candidate.point).squaredNorm() < minimum_spacing_squared;
        })) {
      ++stats.rejected_spacing_points;
      continue;
    }
    bucket.push_back(candidate.point);
    ++point_count_;
    ++stats.inserted_points;
  }
  return stats;
}

void BoundedVoxelTarget::prune(const Point3d& retention_center_target,
                               VoxelTargetUpdateStats& stats) {
  const double retention_radius_squared = options_.retention_radius_m * options_.retention_radius_m;
  for (auto iterator = voxels_.begin(); iterator != voxels_.end();) {
    PointCloud& bucket = iterator->second;
    const auto retained_end =
        std::remove_if(bucket.begin(), bucket.end(), [&](const Point3d& point) {
          return (point - retention_center_target).squaredNorm() > retention_radius_squared;
        });
    const std::size_t removed = static_cast<std::size_t>(std::distance(retained_end, bucket.end()));
    bucket.erase(retained_end, bucket.end());
    stats.pruned_points += removed;
    point_count_ -= removed;
    if (!bucket.empty()) {
      ++iterator;
      continue;
    }
    ++stats.pruned_voxels;
    iterator = voxels_.erase(iterator);
  }
}

bool BoundedVoxelTarget::makeRoomFor(const VoxelKey& candidate_key, const Point3d& candidate_point,
                                     const Point3d& retention_center_target,
                                     VoxelTargetUpdateStats& stats) {
  if (voxels_.size() < options_.max_voxels) {
    return true;
  }

  auto eviction = voxels_.end();
  double eviction_distance = -1.0;
  for (auto iterator = voxels_.begin(); iterator != voxels_.end(); ++iterator) {
    const double distance = closestSquaredDistance(iterator->second, retention_center_target);
    if (eviction == voxels_.end() || distance > eviction_distance ||
        (distance == eviction_distance && eviction->first < iterator->first)) {
      eviction = iterator;
      eviction_distance = distance;
    }
  }

  const double candidate_distance = (candidate_point - retention_center_target).squaredNorm();
  const bool candidate_preferred =
      candidate_distance < eviction_distance ||
      (candidate_distance == eviction_distance && candidate_key < eviction->first);
  if (!candidate_preferred) {
    return false;
  }

  stats.evicted_points += eviction->second.size();
  ++stats.evicted_voxels;
  point_count_ -= eviction->second.size();
  voxels_.erase(eviction);
  return true;
}

bool BoundedVoxelTarget::supportsQueryDistance(double max_distance_m) const noexcept {
  if (!std::isfinite(max_distance_m) || max_distance_m <= 0.0) {
    return false;
  }
  const double radius = std::ceil(max_distance_m / options_.voxel_size_m);
  return std::isfinite(radius) && radius >= 0.0 &&
         radius <= static_cast<double>(options_.max_neighbor_voxel_radius);
}

std::optional<NearestTargetPoint> BoundedVoxelTarget::nearestNeighbor(
    const Point3d& query_target, double max_distance_m, VoxelTargetQueryStats* query_stats) const {
  if (query_stats != nullptr) {
    *query_stats = {};
  }
  if (!query_target.allFinite()) {
    throw std::invalid_argument("nearest-neighbour query must be finite");
  }
  if (!supportsQueryDistance(max_distance_m)) {
    throw std::invalid_argument(
        "nearest-neighbour distance exceeds the configured voxel search radius");
  }
  if (voxels_.empty()) {
    return std::nullopt;
  }

  const VoxelKey centre = pointToVoxelKey(query_target, options_.voxel_size_m);
  const double maximum_squared_distance = max_distance_m * max_distance_m;
  const double voxel_size_squared = options_.voxel_size_m * options_.voxel_size_m;
  std::optional<NearestTargetPoint> nearest;
  VoxelTargetQueryStats accumulated_stats;

  // Coordinates inside the centre voxel. This lets us reject a neighboring
  // voxel before hashing it when that voxel's AABB cannot intersect the exact
  // Euclidean correspondence ball. The result remains an exact nearest-neighbor
  // query; this only removes impossible candidates from the old cube search.
  const Point3d scaled = query_target / options_.voxel_size_m;
  const Point3d fractional =
      scaled - Point3d{std::floor(scaled.x()), std::floor(scaled.y()), std::floor(scaled.z())};
  const auto axisLowerDistance = [&](std::int64_t offset, double fraction) noexcept {
    if (offset > 0) {
      return (static_cast<double>(offset) - fraction) * options_.voxel_size_m;
    }
    if (offset < 0) {
      return (fraction - static_cast<double>(offset + 1)) * options_.voxel_size_m;
    }
    return 0.0;
  };

  for (const NeighborOffset& offset : neighbor_offsets_) {
    const double incumbent_squared_distance =
        nearest.has_value() ? nearest->squared_distance_m2 : maximum_squared_distance;
    if (offset.minimum_squared_distance_voxel_units * voxel_size_squared >
        incumbent_squared_distance) {
      break;
    }
    const double lower_x = axisLowerDistance(offset.x, fractional.x());
    const double lower_y = axisLowerDistance(offset.y, fractional.y());
    const double lower_z = axisLowerDistance(offset.z, fractional.z());
    const double voxel_lower_bound_squared =
        lower_x * lower_x + lower_y * lower_y + lower_z * lower_z;
    if (voxel_lower_bound_squared > incumbent_squared_distance) {
      continue;
    }

    const auto x = checkedOffset(centre.x, offset.x);
    if (!x.has_value()) {
      continue;
    }
    const auto y = checkedOffset(centre.y, offset.y);
    const auto z = checkedOffset(centre.z, offset.z);
    if (!y.has_value() || !z.has_value()) {
      continue;
    }
    ++accumulated_stats.voxel_hash_probes;
    const auto bucket = voxels_.find(VoxelKey{*x, *y, *z});
    if (bucket == voxels_.end()) {
      continue;
    }
    ++accumulated_stats.occupied_buckets;
    accumulated_stats.point_candidates += bucket->second.size();
    for (const Point3d& point : bucket->second) {
      const double squared_distance = (point - query_target).squaredNorm();
      if (squared_distance > maximum_squared_distance) {
        continue;
      }
      if (!nearest.has_value() || squared_distance < nearest->squared_distance_m2 ||
          (squared_distance == nearest->squared_distance_m2 &&
           lexicographicallyLess(point, nearest->point))) {
        nearest = NearestTargetPoint{point, squared_distance};
      }
    }
  }
  if (query_stats != nullptr) {
    *query_stats = accumulated_stats;
  }
  return nearest;
}

PointCloud BoundedVoxelTarget::pointCloud() const {
  std::vector<VoxelKey> keys;
  keys.reserve(voxels_.size());
  for (const auto& [key, points] : voxels_) {
    static_cast<void>(points);
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());

  PointCloud result;
  result.reserve(point_count_);
  for (const VoxelKey& key : keys) {
    PointCloud bucket = voxels_.at(key);
    std::sort(bucket.begin(), bucket.end(), lexicographicallyLess);
    result.insert(result.end(), bucket.cbegin(), bucket.cend());
  }
  return result;
}

}  // namespace meridian::local_rt::lidar
