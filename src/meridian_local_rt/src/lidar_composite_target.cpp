#include "meridian/local/lidar_composite_target.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "meridian/core/canonical_bytes.hpp"

namespace meridian::local {
namespace {

constexpr std::size_t kMaximumPointsPerVoxel{64U};
constexpr std::size_t kMaximumOwnerSlots{
    static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()) + 1U};

struct VoxelKey {
  std::int64_t x{};
  std::int64_t y{};
  std::int64_t z{};

  auto operator<=>(const VoxelKey&) const = default;
};

[[nodiscard]] LidarCompositeTargetError makeError(LidarCompositeTargetErrorCode code,
                                                  std::string detail) {
  return LidarCompositeTargetError{code, std::move(detail)};
}

[[nodiscard]] bool finitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool finitePose(const core::Pose3d& pose) noexcept {
  return pose.matrix().allFinite();
}

[[nodiscard]] bool validKind(LidarCompositeOwnerKind kind) noexcept {
  switch (kind) {
    case LidarCompositeOwnerKind::Live:
    case LidarCompositeOwnerKind::Finalized:
      return true;
  }
  return false;
}

[[nodiscard]] std::uint8_t kindRank(LidarCompositeOwnerKind kind) noexcept {
  return kind == LidarCompositeOwnerKind::Live ? 0U : 1U;
}

[[nodiscard]] auto ownerCanonicalKey(const LidarCompositeTargetOwnerInput& owner) noexcept {
  return std::tuple{kindRank(owner.kind), owner.state,         owner.time,
                    owner.sweep,          owner.pose_revision, owner.geometry_checksum};
}

[[nodiscard]] bool ownerCanonicalLess(const LidarCompositeTargetOwnerInput& lhs,
                                      const LidarCompositeTargetOwnerInput& rhs) noexcept {
  return ownerCanonicalKey(lhs) < ownerCanonicalKey(rhs);
}

[[nodiscard]] bool validConfig(const LidarCompositeTargetConfig& config) noexcept {
  return finitePositive(config.voxel_size_m) && config.maximum_points_per_voxel > 0U &&
         config.maximum_points_per_voxel <= kMaximumPointsPerVoxel &&
         config.maximum_total_points > 0U && config.maximum_owners > 0U &&
         config.maximum_owners <= kMaximumOwnerSlots &&
         finitePositive(config.maximum_query_distance_m) &&
         config.maximum_voxel_search_radius > 0U && config.maximum_voxel_search_radius <= 64U &&
         std::ceil(config.maximum_query_distance_m / config.voxel_size_m) <=
             static_cast<double>(config.maximum_voxel_search_radius);
}

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
  value ^= value >> 30U;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] std::size_t voxelKeyHash(const VoxelKey& key) noexcept {
  std::uint64_t hash = 0x6d6572696469616eULL;
  const auto combine = [&hash](std::uint64_t value) {
    hash = mix64(hash ^ mix64(value + 0x9e3779b97f4a7c15ULL));
  };
  combine(static_cast<std::uint64_t>(key.x));
  combine(static_cast<std::uint64_t>(key.y));
  combine(static_cast<std::uint64_t>(key.z));
  return static_cast<std::size_t>(hash);
}

struct VoxelKeyHash {
  [[nodiscard]] std::size_t operator()(const VoxelKey& key) const noexcept {
    return voxelKeyHash(key);
  }
};

[[nodiscard]] std::optional<VoxelKey> checkedVoxelKey(const Eigen::Vector3d& point,
                                                      double resolution_m) noexcept {
  if (!point.allFinite() || !finitePositive(resolution_m)) {
    return std::nullopt;
  }
  const Eigen::Array3d coordinate = (point / resolution_m).array().floor();
  constexpr double kInt64CoordinateLimit = 0x1p63;
  if (!coordinate.allFinite() || (coordinate < -kInt64CoordinateLimit).any() ||
      (coordinate >= kInt64CoordinateLimit).any()) {
    return std::nullopt;
  }
  return VoxelKey{static_cast<std::int64_t>(coordinate.x()),
                  static_cast<std::int64_t>(coordinate.y()),
                  static_cast<std::int64_t>(coordinate.z())};
}

[[nodiscard]] bool coordinateCanAdd(std::int64_t coordinate, std::int64_t offset) noexcept {
  return !((offset < 0 && coordinate < std::numeric_limits<std::int64_t>::min() - offset) ||
           (offset > 0 && coordinate > std::numeric_limits<std::int64_t>::max() - offset));
}

[[nodiscard]] std::optional<std::size_t> openAddressTableCapacity(
    std::size_t maximum_entries) noexcept {
  std::size_t capacity = 8U;
  while (maximum_entries > capacity / 2U) {
    if (capacity > std::numeric_limits<std::size_t>::max() / 2U) {
      return std::nullopt;
    }
    capacity *= 2U;
  }
  return capacity;
}

[[nodiscard]] Eigen::Vector3d voxelCenter(const VoxelKey& key, double voxel_size_m) noexcept {
  return Eigen::Vector3d{(static_cast<double>(key.x) + 0.5) * voxel_size_m,
                         (static_cast<double>(key.y) + 0.5) * voxel_size_m,
                         (static_cast<double>(key.z) + 0.5) * voxel_size_m};
}

struct VoxelCandidate {
  std::uint16_t owner_slot{};
  std::uint32_t local_point_index{};
  std::uint32_t source_index{};
  double center_distance_squared_m2{};
};

struct SelectedCandidate {
  VoxelKey key;
  std::uint16_t owner_slot{};
  std::uint32_t local_point_index{};
  std::uint32_t source_index{};
  std::uint64_t spatial_rank{};
};

using VoxelAccumulatorMap = std::unordered_map<VoxelKey, std::vector<VoxelCandidate>, VoxelKeyHash>;

[[nodiscard]] std::uint64_t candidateSpatialRank(const SelectedCandidate& candidate) noexcept {
  std::uint64_t rank = 0x636f6d706f736974ULL;
  const auto combine = [&rank](std::uint64_t value) {
    rank = mix64(rank ^ mix64(value + 0x9e3779b97f4a7c15ULL));
  };
  combine(static_cast<std::uint64_t>(candidate.key.x));
  combine(static_cast<std::uint64_t>(candidate.key.y));
  combine(static_cast<std::uint64_t>(candidate.key.z));
  combine(candidate.owner_slot);
  combine(candidate.source_index);
  combine(candidate.local_point_index);
  return rank;
}

template <typename Candidate>
[[nodiscard]] auto candidatePointIdentity(const Candidate& candidate) noexcept {
  return std::tuple{candidate.owner_slot, candidate.source_index, candidate.local_point_index};
}

[[nodiscard]] bool candidateWithinVoxelLess(const VoxelCandidate& lhs,
                                            const VoxelCandidate& rhs) noexcept {
  return std::tuple{lhs.center_distance_squared_m2, candidatePointIdentity(lhs)} <
         std::tuple{rhs.center_distance_squared_m2, candidatePointIdentity(rhs)};
}

[[nodiscard]] bool candidatePointIdentityLess(const VoxelCandidate& lhs,
                                              const VoxelCandidate& rhs) noexcept {
  return candidatePointIdentity(lhs) < candidatePointIdentity(rhs);
}

[[nodiscard]] bool candidateSpatialRankLess(const SelectedCandidate& lhs,
                                            const SelectedCandidate& rhs) noexcept {
  return std::tuple{lhs.spatial_rank, lhs.key, candidatePointIdentity(lhs)} <
         std::tuple{rhs.spatial_rank, rhs.key, candidatePointIdentity(rhs)};
}

[[nodiscard]] bool candidateIndexOrderLess(const SelectedCandidate& lhs,
                                           const SelectedCandidate& rhs) noexcept {
  return std::tuple{lhs.key, candidatePointIdentity(lhs)} <
         std::tuple{rhs.key, candidatePointIdentity(rhs)};
}

}  // namespace

struct LidarCompositeTarget::Impl {
  struct IndexedPoint {
    std::uint16_t owner_slot{};
    std::uint32_t local_point_index{};
  };

  struct Slot {
    bool occupied{};
    VoxelKey key;
    std::size_t point_begin{};
    std::size_t point_count{};
  };

  [[nodiscard]] const Slot* find(const VoxelKey& key) const noexcept {
    if (slots.empty()) {
      return nullptr;
    }
    const std::size_t mask = slots.size() - 1U;
    std::size_t slot_index = voxelKeyHash(key) & mask;
    for (std::size_t probe = 0U; probe < slots.size(); ++probe) {
      const Slot& slot = slots[slot_index];
      if (!slot.occupied) {
        return nullptr;
      }
      if (slot.key == key) {
        return &slot;
      }
      slot_index = (slot_index + 1U) & mask;
    }
    return nullptr;
  }

  LidarCompositeTargetConfig config;
  std::vector<LidarCompositeTargetOwnerInput> owners;
  LidarCompositeTargetBuildStats statistics;
  std::vector<Slot> slots;
  std::vector<IndexedPoint> points;
  core::ContentHash checksum{};
};

core::Result<core::ContentHash, LidarCompositeTargetError> LidarCompositeTarget::computeChecksum(
    const Impl& index) {
  using Result = core::Result<core::ContentHash, LidarCompositeTargetError>;
  constexpr std::uint64_t kFixedBytes = 1024U;
  constexpr std::uint64_t kBytesPerOwner = 512U;
  const std::uint64_t owner_count = static_cast<std::uint64_t>(index.owners.size());
  if (owner_count > (std::numeric_limits<std::uint64_t>::max() - kFixedBytes) / kBytesPerOwner) {
    return Result::failure(makeError(LidarCompositeTargetErrorCode::ChecksumFailure,
                                     "composite target is too large to checksum"));
  }
  auto encoder = core::CanonicalEncoder::create(kLidarCompositeTargetChecksumDomain,
                                                kLidarCompositeTargetChecksumSchemaVersion,
                                                kFixedBytes + owner_count * kBytesPerOwner);
  if (!encoder) {
    return Result::failure(makeError(LidarCompositeTargetErrorCode::ChecksumFailure,
                                     "composite target checksum encoder initialization failed"));
  }
  const auto write = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!write(encoder.value().writeDouble(index.config.voxel_size_m)) ||
      !write(encoder.value().writeU64(
          static_cast<std::uint64_t>(index.config.maximum_points_per_voxel))) ||
      !write(encoder.value().writeU64(
          static_cast<std::uint64_t>(index.config.maximum_total_points))) ||
      !write(encoder.value().writeU64(static_cast<std::uint64_t>(index.config.maximum_owners))) ||
      !write(encoder.value().writeDouble(index.config.maximum_query_distance_m)) ||
      !write(encoder.value().writeU64(
          static_cast<std::uint64_t>(index.config.maximum_voxel_search_radius))) ||
      !write(encoder.value().writeU64(owner_count))) {
    return Result::failure(makeError(LidarCompositeTargetErrorCode::ChecksumFailure,
                                     "composite target checksum header encoding failed"));
  }
  for (const LidarCompositeTargetOwnerInput& owner : index.owners) {
    if (!write(encoder.value().writeU8(kindRank(owner.kind))) ||
        !write(encoder.value().writeU64(owner.state.value())) ||
        !write(encoder.value().writeI64(owner.time.nanoseconds)) ||
        !write(encoder.value().writeU64(owner.sweep.value())) ||
        !write(encoder.value().writeU64(owner.pose_revision.value())) ||
        !write(encoder.value().writeHash(owner.geometry_checksum)) ||
        !write(encoder.value().writePose3(owner.T_odom_owner)) ||
        !write(encoder.value().writeU64(static_cast<std::uint64_t>(owner.points->size())))) {
      return Result::failure(makeError(LidarCompositeTargetErrorCode::ChecksumFailure,
                                       "composite target owner checksum encoding failed"));
    }
  }
  auto encoded = encoder.value().finish();
  if (!encoded) {
    return Result::failure(makeError(LidarCompositeTargetErrorCode::ChecksumFailure,
                                     "composite target checksum finalization failed"));
  }
  return Result::success(encoded.value().digest());
}

core::Result<LidarCompositeTarget, LidarCompositeTargetError> LidarCompositeTarget::create(
    LidarCompositeTargetConfig config, std::vector<LidarCompositeTargetOwnerInput> owners) {
  using Result = core::Result<LidarCompositeTarget, LidarCompositeTargetError>;
  if (!validConfig(config)) {
    return Result::failure(
        makeError(LidarCompositeTargetErrorCode::InvalidConfig,
                  "composite target voxel, query, owner, or point capacity is invalid"));
  }
  if (owners.empty()) {
    return Result::failure(makeError(LidarCompositeTargetErrorCode::EmptyOwners,
                                     "composite target requires at least one owner"));
  }
  if (owners.size() > config.maximum_owners || owners.size() > kMaximumOwnerSlots) {
    return Result::failure(makeError(LidarCompositeTargetErrorCode::Capacity,
                                     "composite target owner capacity is exceeded"));
  }

  std::sort(owners.begin(), owners.end(), ownerCanonicalLess);
  std::set<core::StateId> states;
  std::set<core::MeasurementId> sweeps;
  std::size_t input_points = 0U;
  for (const LidarCompositeTargetOwnerInput& owner : owners) {
    if (!validKind(owner.kind) || !owner.state.valid() || !owner.sweep.valid() ||
        !owner.pose_revision.valid() || !core::contentHashPresent(owner.geometry_checksum) ||
        !finitePose(owner.T_odom_owner) || !owner.points || owner.points->empty()) {
      return Result::failure(makeError(
          LidarCompositeTargetErrorCode::InvalidOwner,
          "composite target owner identity, revision, pose, checksum, or geometry is invalid"));
    }
    if (!states.insert(owner.state).second || !sweeps.insert(owner.sweep).second) {
      return Result::failure(makeError(LidarCompositeTargetErrorCode::DuplicateOwner,
                                       "composite target state and sweep owners must be unique"));
    }
    if (owner.points->size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      return Result::failure(makeError(LidarCompositeTargetErrorCode::Capacity,
                                       "owner-local geometry exceeds compact point identity"));
    }
    if (owner.points->size() > std::numeric_limits<std::size_t>::max() - input_points) {
      return Result::failure(makeError(LidarCompositeTargetErrorCode::Capacity,
                                       "composite target input point count overflowed"));
    }
    input_points += owner.points->size();
    std::vector<std::uint32_t> source_indices;
    source_indices.reserve(owner.points->size());
    for (const LidarCompositeTargetPoint& point : *owner.points) {
      if (!point.point_owner.allFinite()) {
        return Result::failure(
            makeError(LidarCompositeTargetErrorCode::InvalidGeometry,
                      "owner-local target geometry contains a non-finite point"));
      }
      source_indices.push_back(point.source_index);
    }
    std::sort(source_indices.begin(), source_indices.end());
    if (std::adjacent_find(source_indices.begin(), source_indices.end()) != source_indices.end()) {
      return Result::failure(
          makeError(LidarCompositeTargetErrorCode::InvalidGeometry,
                    "owner-local target geometry contains duplicate stable source indices"));
    }
  }

  // Keep only the K canonical center-nearest points in each voxel while the
  // input recipe is streamed. The heap root is the worst retained point, so
  // memory and comparison work per occupied voxel remain bounded by K. This
  // produces exactly the same set as sorting every input point by
  // (voxel, center distance, point identity) and taking the first K.
  VoxelAccumulatorMap voxel_accumulators;
  voxel_accumulators.reserve(std::min(input_points, config.maximum_total_points));
  std::size_t voxel_bounded_point_count = 0U;
  std::size_t per_voxel_discarded = 0U;
  for (std::size_t owner_index = 0U; owner_index < owners.size(); ++owner_index) {
    const LidarCompositeTargetOwnerInput& owner = owners[owner_index];
    const auto owner_slot = static_cast<std::uint16_t>(owner_index);
    for (std::size_t local_index = 0U; local_index < owner.points->size(); ++local_index) {
      const LidarCompositeTargetPoint& local = (*owner.points)[local_index];
      const Eigen::Vector3d point_odom = owner.T_odom_owner * local.point_owner;
      const auto key = checkedVoxelKey(point_odom, config.voxel_size_m);
      if (!key) {
        return Result::failure(
            makeError(LidarCompositeTargetErrorCode::SpatialIndexFailure,
                      "owner-local target point cannot be represented in the odom voxel index"));
      }
      const VoxelCandidate candidate{
          owner_slot, static_cast<std::uint32_t>(local_index), local.source_index,
          (point_odom - voxelCenter(*key, config.voxel_size_m)).squaredNorm()};
      auto& retained = voxel_accumulators[*key];
      if (retained.size() < config.maximum_points_per_voxel) {
        retained.push_back(candidate);
        std::push_heap(retained.begin(), retained.end(), candidateWithinVoxelLess);
        ++voxel_bounded_point_count;
        continue;
      }
      ++per_voxel_discarded;
      if (!candidateWithinVoxelLess(candidate, retained.front())) {
        continue;
      }
      std::pop_heap(retained.begin(), retained.end(), candidateWithinVoxelLess);
      retained.back() = candidate;
      std::push_heap(retained.begin(), retained.end(), candidateWithinVoxelLess);
    }
  }

  // Below the global cap, sorting only the occupied-voxel handles plus each
  // bounded bucket emits final index order and never computes the spatial
  // sampling rank.
  // Above it, rank only the already voxel-bounded set, select the exact same
  // globally ranked prefix as the reference algorithm, then canonicalize the
  // bounded final output.
  const std::size_t before_total_cap = voxel_bounded_point_count;
  std::vector<SelectedCandidate> selected;
  selected.reserve(std::min(before_total_cap, config.maximum_total_points));
  if (before_total_cap <= config.maximum_total_points) {
    std::vector<VoxelAccumulatorMap::value_type*> ordered_voxels;
    ordered_voxels.reserve(voxel_accumulators.size());
    for (auto& voxel : voxel_accumulators) {
      ordered_voxels.push_back(&voxel);
    }
    std::sort(ordered_voxels.begin(), ordered_voxels.end(),
              [](const auto* lhs, const auto* rhs) { return lhs->first < rhs->first; });
    for (auto* voxel : ordered_voxels) {
      const VoxelKey& key = voxel->first;
      std::vector<VoxelCandidate>& retained = voxel->second;
      std::sort(retained.begin(), retained.end(), candidatePointIdentityLess);
      for (const VoxelCandidate& candidate : retained) {
        selected.push_back(SelectedCandidate{key, candidate.owner_slot, candidate.local_point_index,
                                             candidate.source_index, 0U});
      }
    }
  } else {
    selected.reserve(before_total_cap);
    for (const auto& [key, retained] : voxel_accumulators) {
      for (const VoxelCandidate& candidate : retained) {
        SelectedCandidate output{key, candidate.owner_slot, candidate.local_point_index,
                                 candidate.source_index, 0U};
        output.spatial_rank = candidateSpatialRank(output);
        selected.push_back(output);
      }
    }
    std::nth_element(selected.begin(),
                     selected.begin() + static_cast<std::ptrdiff_t>(config.maximum_total_points),
                     selected.end(), candidateSpatialRankLess);
    selected.resize(config.maximum_total_points);
    std::sort(selected.begin(), selected.end(), candidateIndexOrderLess);
  }
  if (selected.empty()) {
    return Result::failure(makeError(LidarCompositeTargetErrorCode::Capacity,
                                     "composite target capacity retained no point"));
  }

  struct VoxelSpan {
    VoxelKey key;
    std::size_t point_begin{};
    std::size_t point_count{};
  };
  std::vector<VoxelSpan> voxel_spans;
  voxel_spans.reserve(std::min(voxel_accumulators.size(), selected.size()));
  auto implementation = std::make_unique<Impl>();
  implementation->config = config;
  implementation->owners = std::move(owners);
  implementation->points.reserve(selected.size());
  for (std::size_t begin = 0U; begin < selected.size();) {
    std::size_t end = begin + 1U;
    while (end < selected.size() && selected[end].key == selected[begin].key) {
      ++end;
    }
    const std::size_t point_begin = implementation->points.size();
    for (std::size_t index = begin; index < end; ++index) {
      implementation->points.push_back(
          Impl::IndexedPoint{selected[index].owner_slot, selected[index].local_point_index});
    }
    voxel_spans.push_back(VoxelSpan{selected[begin].key, point_begin, end - begin});
    begin = end;
  }

  const auto table_capacity = openAddressTableCapacity(voxel_spans.size());
  if (!table_capacity) {
    return Result::failure(makeError(LidarCompositeTargetErrorCode::SpatialIndexFailure,
                                     "composite target hash-table capacity overflowed"));
  }
  implementation->slots.resize(*table_capacity);
  const std::size_t slot_mask = implementation->slots.size() - 1U;
  for (const VoxelSpan& span : voxel_spans) {
    std::size_t slot_index = voxelKeyHash(span.key) & slot_mask;
    while (implementation->slots[slot_index].occupied) {
      slot_index = (slot_index + 1U) & slot_mask;
    }
    implementation->slots[slot_index] =
        Impl::Slot{true, span.key, span.point_begin, span.point_count};
  }
  implementation->statistics =
      LidarCompositeTargetBuildStats{implementation->owners.size(),
                                     input_points,
                                     implementation->owners.size(),
                                     voxel_spans.size(),
                                     implementation->points.size(),
                                     per_voxel_discarded,
                                     before_total_cap - implementation->points.size()};
  auto checksum = computeChecksum(*implementation);
  if (!checksum) {
    return Result::failure(checksum.error());
  }
  implementation->checksum = checksum.value();
  return Result::success(LidarCompositeTarget(std::move(implementation)));
}

LidarCompositeTarget::LidarCompositeTarget(std::unique_ptr<const Impl> implementation)
    : implementation_(std::move(implementation)) {}

LidarCompositeTarget::~LidarCompositeTarget() = default;
LidarCompositeTarget::LidarCompositeTarget(LidarCompositeTarget&&) noexcept = default;
LidarCompositeTarget& LidarCompositeTarget::operator=(LidarCompositeTarget&&) noexcept = default;

LidarCompositeTargetNeighbor LidarCompositeTarget::nearestExact(
    const Eigen::Vector3d& query_odom, double maximum_distance_m) const noexcept {
  LidarCompositeTargetNeighbor best;
  best.distance_squared_m2 = std::numeric_limits<double>::infinity();
  if (!query_odom.allFinite() || !finitePositive(maximum_distance_m) ||
      maximum_distance_m > implementation_->config.maximum_query_distance_m) {
    return best;
  }
  const auto center = checkedVoxelKey(query_odom, implementation_->config.voxel_size_m);
  if (!center) {
    return best;
  }
  const double maximum_squared = maximum_distance_m * maximum_distance_m;
  if (!std::isfinite(maximum_squared)) {
    return best;
  }
  best.query_valid = true;
  const double radius_value = std::ceil(maximum_distance_m / implementation_->config.voxel_size_m);
  if (!std::isfinite(radius_value) || radius_value < 1.0 ||
      radius_value > static_cast<double>(implementation_->config.maximum_voxel_search_radius)) {
    best.query_valid = false;
    return best;
  }
  const auto radius = static_cast<std::int64_t>(radius_value);
  for (std::int64_t dx = -radius; dx <= radius; ++dx) {
    for (std::int64_t dy = -radius; dy <= radius; ++dy) {
      for (std::int64_t dz = -radius; dz <= radius; ++dz) {
        ++best.voxel_lookups;
        if (!coordinateCanAdd(center->x, dx) || !coordinateCanAdd(center->y, dy) ||
            !coordinateCanAdd(center->z, dz)) {
          continue;
        }
        const Impl::Slot* slot =
            implementation_->find(VoxelKey{center->x + dx, center->y + dy, center->z + dz});
        if (slot == nullptr) {
          continue;
        }
        ++best.occupied_voxels;
        for (std::size_t offset = 0U; offset < slot->point_count; ++offset) {
          ++best.points_examined;
          const Impl::IndexedPoint& indexed = implementation_->points[slot->point_begin + offset];
          if (indexed.owner_slot >= implementation_->owners.size()) {
            continue;
          }
          const LidarCompositeTargetOwnerInput& owner = implementation_->owners[indexed.owner_slot];
          if (!owner.points || indexed.local_point_index >= owner.points->size()) {
            continue;
          }
          const LidarCompositeTargetPoint& local = (*owner.points)[indexed.local_point_index];
          const Eigen::Vector3d exact_point_odom = owner.T_odom_owner * local.point_owner;
          const double distance_squared = (exact_point_odom - query_odom).squaredNorm();
          if (!exact_point_odom.allFinite() || !std::isfinite(distance_squared) ||
              distance_squared > maximum_squared) {
            continue;
          }
          const bool better_distance = !best.found || distance_squared < best.distance_squared_m2;
          const bool equal_distance = best.found && distance_squared == best.distance_squared_m2;
          bool better_tie = false;
          if (equal_distance) {
            const LidarCompositeTargetOwnerInput& incumbent =
                implementation_->owners[best.owner_slot];
            better_tie =
                std::tuple{kindRank(owner.kind), indexed.owner_slot, local.source_index,
                           indexed.local_point_index} <
                std::tuple{kindRank(incumbent.kind), static_cast<std::uint16_t>(best.owner_slot),
                           best.source_index, static_cast<std::uint32_t>(best.local_point_index)};
          }
          if (!better_distance && !better_tie) {
            continue;
          }
          best.found = true;
          best.owner_slot = indexed.owner_slot;
          best.local_point_index = indexed.local_point_index;
          best.source_index = local.source_index;
          best.point_owner = local.point_owner;
          best.point_odom = exact_point_odom;
          best.distance_squared_m2 = distance_squared;
        }
      }
    }
  }
  return best;
}

const LidarCompositeTargetConfig& LidarCompositeTarget::config() const noexcept {
  return implementation_->config;
}

const LidarCompositeTargetBuildStats& LidarCompositeTarget::statistics() const noexcept {
  return implementation_->statistics;
}

std::span<const LidarCompositeTargetOwnerInput> LidarCompositeTarget::owners() const noexcept {
  return implementation_->owners;
}

const core::ContentHash& LidarCompositeTarget::checksum() const noexcept {
  return implementation_->checksum;
}

}  // namespace meridian::local
