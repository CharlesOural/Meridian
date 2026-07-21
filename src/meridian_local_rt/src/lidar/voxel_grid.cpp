#include "meridian/local_rt/lidar/voxel_grid.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace meridian::local_rt::lidar {
namespace {

void validateVoxelSize(double voxel_size) {
  if (!std::isfinite(voxel_size) || voxel_size <= 0.0) {
    throw std::invalid_argument("voxel_size must be finite and positive");
  }
}

std::int64_t floorToVoxelCoordinate(double coordinate, double voxel_size) {
  const long double scaled = static_cast<long double>(coordinate) / voxel_size;
  const long double floored = std::floor(scaled);
  constexpr long double kMinimum =
      static_cast<long double>(std::numeric_limits<std::int64_t>::min());
  constexpr long double kMaximum =
      static_cast<long double>(std::numeric_limits<std::int64_t>::max());
  if (!std::isfinite(scaled) || floored < kMinimum || floored > kMaximum) {
    throw std::overflow_error("point coordinate is outside the signed 64-bit voxel domain");
  }
  return static_cast<std::int64_t>(floored);
}

bool lexicographicallyLess(const Point3d& lhs, const Point3d& rhs) noexcept {
  if (lhs.x() != rhs.x()) {
    return lhs.x() < rhs.x();
  }
  if (lhs.y() != rhs.y()) {
    return lhs.y() < rhs.y();
  }
  return lhs.z() < rhs.z();
}

long double squaredDistanceToVoxelCentre(const Point3d& point, const VoxelKey& key,
                                         double voxel_size) noexcept {
  const long double inverse_size = 1.0L / static_cast<long double>(voxel_size);
  const long double dx =
      static_cast<long double>(point.x()) * inverse_size - (static_cast<long double>(key.x) + 0.5L);
  const long double dy =
      static_cast<long double>(point.y()) * inverse_size - (static_cast<long double>(key.y) + 0.5L);
  const long double dz =
      static_cast<long double>(point.z()) * inverse_size - (static_cast<long double>(key.z) + 0.5L);
  return dx * dx + dy * dy + dz * dz;
}

std::uint64_t mix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

struct SelectedPoint final {
  Point3d point{Point3d::Zero()};
  long double squared_centre_distance{};
};

}  // namespace

std::size_t VoxelKeyHash::operator()(const VoxelKey& key) const noexcept {
  const std::uint64_t x = mix64(static_cast<std::uint64_t>(key.x));
  const std::uint64_t y = mix64(static_cast<std::uint64_t>(key.y));
  const std::uint64_t z = mix64(static_cast<std::uint64_t>(key.z));
  const std::uint64_t combined = x ^ std::rotl(y, 21) ^ std::rotl(z, 42);
  if constexpr (sizeof(std::size_t) >= sizeof(std::uint64_t)) {
    return static_cast<std::size_t>(combined);
  }
  return static_cast<std::size_t>(combined ^ (combined >> 32U));
}

VoxelKey pointToVoxelKey(const Point3d& point, double voxel_size) {
  validateVoxelSize(voxel_size);
  if (!point.allFinite()) {
    throw std::invalid_argument("pointToVoxelKey requires a finite point");
  }
  return VoxelKey{
      .x = floorToVoxelCoordinate(point.x(), voxel_size),
      .y = floorToVoxelCoordinate(point.y(), voxel_size),
      .z = floorToVoxelCoordinate(point.z(), voxel_size),
  };
}

VoxelDownsampleResult deterministicVoxelDownsample(std::span<const Point3d> points,
                                                   double voxel_size) {
  validateVoxelSize(voxel_size);

  VoxelDownsampleResult result;
  result.stats.input_points = points.size();
  std::map<VoxelKey, SelectedPoint> selected;

  for (const Point3d& point : points) {
    if (!point.allFinite()) {
      ++result.stats.rejected_nonfinite_points;
      continue;
    }
    ++result.stats.finite_points;
    const VoxelKey key = pointToVoxelKey(point, voxel_size);
    const long double squared_distance = squaredDistanceToVoxelCentre(point, key, voxel_size);
    const auto [iterator, inserted] =
        selected.try_emplace(key, SelectedPoint{point, squared_distance});
    if (!inserted) {
      SelectedPoint& incumbent = iterator->second;
      if (squared_distance < incumbent.squared_centre_distance ||
          (squared_distance == incumbent.squared_centre_distance &&
           lexicographicallyLess(point, incumbent.point))) {
        incumbent = SelectedPoint{point, squared_distance};
      }
    }
  }

  result.points.reserve(selected.size());
  for (const auto& [key, representative] : selected) {
    static_cast<void>(key);
    result.points.push_back(representative.point);
  }
  result.stats.output_points = result.points.size();
  return result;
}

}  // namespace meridian::local_rt::lidar
