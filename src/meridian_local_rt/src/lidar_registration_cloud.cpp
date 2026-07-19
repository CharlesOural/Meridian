#include "meridian/local/lidar_registration_cloud.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <compare>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

#include "meridian/core/canonical_bytes.hpp"
#include "meridian/core/canonical_records.hpp"

namespace meridian::local {
namespace {

struct VoxelKey {
  std::int64_t x{};
  std::int64_t y{};
  std::int64_t z{};
  auto operator<=>(const VoxelKey&) const = default;
};

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

[[nodiscard]] LidarPreprocessError preprocessError(LidarPreprocessErrorCode code,
                                                   std::string detail) {
  return LidarPreprocessError{code, std::move(detail)};
}

[[nodiscard]] std::optional<VoxelKey> checkedVoxelKey(const Eigen::Vector3d& point,
                                                      double resolution) noexcept {
  if (!point.allFinite() || !std::isfinite(resolution) || resolution <= 0.0) {
    return std::nullopt;
  }
  const Eigen::Array3d coordinate = (point / resolution).array().floor();
  // +2^63 is outside int64 while -2^63 is representable. Check before cast to
  // keep malformed construction input from invoking undefined behavior.
  constexpr double kInt64CoordinateLimit = 0x1p63;
  if (!coordinate.allFinite() || (coordinate < -kInt64CoordinateLimit).any() ||
      (coordinate >= kInt64CoordinateLimit).any()) {
    return std::nullopt;
  }
  return VoxelKey{static_cast<std::int64_t>(coordinate.x()),
                  static_cast<std::int64_t>(coordinate.y()),
                  static_cast<std::int64_t>(coordinate.z())};
}

[[nodiscard]] std::optional<std::size_t> checkedLayoutSourceDomain(
    const core::LidarLayout& layout, std::size_t payload_points) noexcept {
  if (layout.width == 0U || layout.height == 0U || payload_points == 0U) {
    return std::nullopt;
  }
  constexpr std::size_t kMaximumSealedSourceDomain = 100'000'000U;
  constexpr std::size_t kMaximumDomainToPayloadRatio = 16U;
  const std::size_t width = static_cast<std::size_t>(layout.width);
  const std::size_t height = static_cast<std::size_t>(layout.height);
  if (width > std::numeric_limits<std::size_t>::max() / height) {
    return std::nullopt;
  }
  const std::size_t source_domain = width * height;
  if (payload_points > source_domain || (layout.organized && payload_points != source_domain)) {
    return std::nullopt;
  }
  const std::size_t scaled_payload =
      payload_points > std::numeric_limits<std::size_t>::max() / kMaximumDomainToPayloadRatio
          ? std::numeric_limits<std::size_t>::max()
          : kMaximumDomainToPayloadRatio * payload_points;
  if (source_domain > kMaximumSealedSourceDomain || source_domain > scaled_payload) {
    return std::nullopt;
  }
  return source_domain;
}

[[nodiscard]] bool finiteRawPoint(const core::LidarPoint& point) noexcept {
  return std::isfinite(static_cast<double>(point.x)) &&
         std::isfinite(static_cast<double>(point.y)) &&
         std::isfinite(static_cast<double>(point.z)) &&
         std::isfinite(static_cast<double>(point.intensity));
}

[[nodiscard]] bool registrationPointExactlyBindsRaw(
    const LidarRegistrationPoint& registration_point, const core::LidarPoint& raw) noexcept {
  // Preserve signed zero and the exact float-to-double promotion as part of
  // the sealed row identity.
  return registration_point.source_index == raw.source_index &&
         std::bit_cast<std::uint64_t>(registration_point.point.x()) ==
             std::bit_cast<std::uint64_t>(static_cast<double>(raw.x)) &&
         std::bit_cast<std::uint64_t>(registration_point.point.y()) ==
             std::bit_cast<std::uint64_t>(static_cast<double>(raw.y)) &&
         std::bit_cast<std::uint64_t>(registration_point.point.z()) ==
             std::bit_cast<std::uint64_t>(static_cast<double>(raw.z)) &&
         std::bit_cast<std::uint32_t>(registration_point.intensity) ==
             std::bit_cast<std::uint32_t>(raw.intensity) &&
         registration_point.ring == raw.ring;
}

[[nodiscard]] bool validConfig(const LidarPreprocessConfig& config) noexcept {
  return config.parallel_worker_count >= 1U && config.parallel_worker_count <= 64U &&
         config.parallel_worker_count <=
             static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         std::isfinite(config.minimum_range_m) && std::isfinite(config.maximum_range_m) &&
         std::isfinite(config.voxel_size_m) && config.minimum_range_m >= 0.0 &&
         config.maximum_range_m > config.minimum_range_m && config.voxel_size_m > 0.0 &&
         config.maximum_output_points > 0U;
}

[[nodiscard]] bool finitePose(const core::Pose3d& pose) noexcept {
  return pose.matrix().allFinite();
}

[[nodiscard]] bool lineageReferencesSweep(const core::ObservationLineage& lineage,
                                          core::MeasurementId sweep) noexcept {
  return std::any_of(lineage.usage.begin(), lineage.usage.end(), [sweep](const auto& usage) {
    const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
    return measurement != nullptr && *measurement == sweep;
  });
}

[[nodiscard]] bool lineageExactlyConditionsOnImuSupport(
    const core::ObservationLineage& lineage,
    std::span<const core::MeasurementId> imu_support) noexcept {
  for (const core::MeasurementId measurement : imu_support) {
    std::size_t matching_usages = 0U;
    for (const core::ObservationUsage& usage : lineage.usage) {
      const auto* root = std::get_if<core::MeasurementId>(&usage.slice.root);
      if (root == nullptr || *root != measurement) {
        continue;
      }
      ++matching_usages;
      if (usage.role != core::ObservationRole::ConditioningOnly) {
        return false;
      }
    }
    if (matching_usages != 1U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] core::Result<core::ObservationLineage, LidarPreprocessError> sealLineage(
    core::ObservationLineage lineage, core::MeasurementId source) {
  using Result = core::Result<core::ObservationLineage, LidarPreprocessError>;
  if (!lineage.id.valid() || lineage.usage.empty() ||
      core::validateLineage(lineage) != core::LineageValidationError::None ||
      !lineageReferencesSweep(lineage, source)) {
    return Result::failure(preprocessError(
        LidarPreprocessErrorCode::InvalidLineage,
        "LiDAR registration lineage is invalid or does not reference its source sweep"));
  }
  const bool every_source_checksum_present = std::all_of(
      lineage.usage.begin(), lineage.usage.end(),
      [](const auto& usage) { return core::contentHashPresent(usage.slice.source_checksum); });
  if (!every_source_checksum_present) {
    lineage.checksum = {};
    return Result::success(std::move(lineage));
  }
  auto checksum = core::recomputeObservationLineageChecksum(lineage);
  if (!checksum) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::ChecksumFailure,
                        "LiDAR registration lineage checksum cannot be computed"));
  }
  if (core::contentHashPresent(lineage.checksum) && lineage.checksum != checksum.value()) {
    return Result::failure(preprocessError(LidarPreprocessErrorCode::InvalidLineage,
                                           "LiDAR registration lineage checksum does not match"));
  }
  lineage.checksum = checksum.value();
  return Result::success(std::move(lineage));
}

[[nodiscard]] bool writeLineage(core::CanonicalEncoder& encoder,
                                const core::ObservationLineage& lineage) {
  const auto write = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!write(encoder.writeU64(lineage.id.value())) ||
      !write(encoder.writeOptionalMarker(core::contentHashPresent(lineage.checksum))) ||
      (core::contentHashPresent(lineage.checksum) && !write(encoder.writeHash(lineage.checksum))) ||
      !write(encoder.writeU64(static_cast<std::uint64_t>(lineage.usage.size())))) {
    return false;
  }
  for (const core::ObservationUsage& usage : lineage.usage) {
    const bool measurement = std::holds_alternative<core::MeasurementId>(usage.slice.root);
    if (!write(encoder.writeU8(measurement ? 0U : 1U)) ||
        !write(encoder.writeU64(
            measurement ? std::get<core::MeasurementId>(usage.slice.root).value()
                        : std::get<core::GnssObservationId>(usage.slice.root).value())) ||
        !write(encoder.writeU8(static_cast<std::uint8_t>(usage.slice.kind))) ||
        !write(encoder.writeU64(usage.slice.begin)) || !write(encoder.writeU64(usage.slice.end)) ||
        !write(
            encoder.writeOptionalMarker(core::contentHashPresent(usage.slice.source_checksum))) ||
        (core::contentHashPresent(usage.slice.source_checksum) &&
         !write(encoder.writeHash(usage.slice.source_checksum))) ||
        !write(encoder.writeU64(usage.slice.calibration.value())) ||
        !write(encoder.writeU8(static_cast<std::uint8_t>(usage.role))) ||
        !write(encoder.writeU64(usage.consumer.value())) ||
        !write(encoder.writeOptionalMarker(usage.factor_group.has_value())) ||
        (usage.factor_group && !write(encoder.writeU64(usage.factor_group->value()))) ||
        !write(encoder.writeOptionalMarker(usage.correlation_group.has_value())) ||
        (usage.correlation_group && !write(encoder.writeU64(usage.correlation_group->value())))) {
      return false;
    }
  }
  if (!write(encoder.writeU64(static_cast<std::uint64_t>(lineage.correlations.size())))) {
    return false;
  }
  for (const core::CorrelationDeclaration& declaration : lineage.correlations) {
    if (!write(encoder.writeU64(declaration.group.value())) ||
        !write(encoder.writeU64(declaration.policy.value())) ||
        !write(encoder.writeU8(static_cast<std::uint8_t>(declaration.treatment))) ||
        !write(encoder.writeDouble(declaration.covariance_inflation)) ||
        !write(encoder.writeOptionalMarker(declaration.total_information_cap.has_value())) ||
        (declaration.total_information_cap &&
         !write(encoder.writeDouble(*declaration.total_information_cap)))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] core::Result<core::ContentHash, LidarPreprocessError> cloudChecksum(
    const LidarRegistrationCloudData& data, const core::ObservationLineage& lineage,
    double index_resolution) {
  using Result = core::Result<core::ContentHash, LidarPreprocessError>;
  constexpr std::uint64_t kFixedBytes = 2048U;
  constexpr std::uint64_t kBytesPerPoint = 128U;
  constexpr std::uint64_t kBytesPerLineageRow = 256U;
  constexpr std::uint64_t kBytesPerImuSupport = 16U;
  const auto point_count = static_cast<std::uint64_t>(data.points.size());
  const auto lineage_usage_count = static_cast<std::uint64_t>(lineage.usage.size());
  const auto lineage_correlation_count = static_cast<std::uint64_t>(lineage.correlations.size());
  const auto imu_support_count = static_cast<std::uint64_t>(data.imu_support.size());
  std::uint64_t maximum_bytes = kFixedBytes;
  const auto append_capacity = [&maximum_bytes](std::uint64_t rows, std::uint64_t bytes_per_row) {
    if (rows > (std::numeric_limits<std::uint64_t>::max() - maximum_bytes) / bytes_per_row) {
      return false;
    }
    maximum_bytes += rows * bytes_per_row;
    return true;
  };
  if (!append_capacity(point_count, kBytesPerPoint) ||
      !append_capacity(lineage_usage_count, kBytesPerLineageRow) ||
      !append_capacity(lineage_correlation_count, kBytesPerLineageRow) ||
      !append_capacity(imu_support_count, kBytesPerImuSupport)) {
    return Result::failure(preprocessError(LidarPreprocessErrorCode::ChecksumFailure,
                                           "LiDAR registration cloud is too large to checksum"));
  }
  auto encoder =
      core::CanonicalEncoder::create(kLidarRegistrationCloudChecksumDomain,
                                     kLidarRegistrationCloudChecksumSchemaVersion, maximum_bytes);
  if (!encoder) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::ChecksumFailure,
                        "LiDAR registration cloud checksum encoder initialization failed"));
  }
  const auto write = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!write(encoder.value().writeU64(data.source_sweep.value())) ||
      !write(encoder.value().writeI64(data.reference_time.nanoseconds)) ||
      !write(encoder.value().writePose3(data.T_odom_imu_seed)) ||
      !write(encoder.value().writeU32(data.layout.width)) ||
      !write(encoder.value().writeU32(data.layout.height)) ||
      !write(encoder.value().writeBool(data.layout.organized)) ||
      !write(encoder.value().writeDouble(index_resolution)) ||
      !write(encoder.value().writeU64(static_cast<std::uint64_t>(data.stats.input_points))) ||
      !write(encoder.value().writeU64(static_cast<std::uint64_t>(data.stats.valid_range_points))) ||
      !write(encoder.value().writeU64(
          static_cast<std::uint64_t>(data.stats.deterministic_voxel_points))) ||
      !write(encoder.value().writeU64(point_count)) ||
      !write(encoder.value().writeU64(imu_support_count))) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::ChecksumFailure,
                        "LiDAR registration cloud checksum header encoding failed"));
  }
  for (core::MeasurementId id : data.imu_support) {
    if (!write(encoder.value().writeU64(id.value()))) {
      return Result::failure(
          preprocessError(LidarPreprocessErrorCode::ChecksumFailure,
                          "LiDAR registration cloud IMU support encoding failed"));
    }
  }
  if (!writeLineage(encoder.value(), lineage)) {
    return Result::failure(preprocessError(LidarPreprocessErrorCode::ChecksumFailure,
                                           "LiDAR registration cloud lineage encoding failed"));
  }
  for (const LidarRegistrationPoint& point : data.points) {
    if (!write(encoder.value().writeU32(point.source_index)) ||
        !write(encoder.value().writeEigenVector(point.point)) ||
        !write(encoder.value().writeU32(std::bit_cast<std::uint32_t>(point.intensity))) ||
        !write(encoder.value().writeU16(point.ring))) {
      return Result::failure(
          preprocessError(LidarPreprocessErrorCode::ChecksumFailure,
                          "LiDAR registration cloud checksum row encoding failed"));
    }
  }
  auto encoded = encoder.value().finish();
  if (!encoded) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::ChecksumFailure,
                        "LiDAR registration cloud checksum finalization failed"));
  }
  return Result::success(encoded.value().digest());
}

struct Candidate {
  std::size_t point_index{};
  double center_distance_squared{};
};

struct RankedCandidate {
  VoxelKey key;
  Candidate candidate;
  std::uint32_t source_index{};
  std::uint64_t rank{};
};

[[nodiscard]] std::uint64_t spatialCandidateRank(const VoxelKey& key,
                                                 std::uint32_t source_index) noexcept {
  std::uint64_t rank = 0x6d6572696469616eULL;
  const auto combine = [&rank](std::uint64_t value) {
    rank = mix64(rank ^ mix64(value + 0x9e3779b97f4a7c15ULL));
  };
  combine(static_cast<std::uint64_t>(key.x));
  combine(static_cast<std::uint64_t>(key.y));
  combine(static_cast<std::uint64_t>(key.z));
  combine(static_cast<std::uint64_t>(source_index));
  return rank;
}

[[nodiscard]] std::optional<std::size_t> openAddressTableCapacity(
    std::size_t maximum_entries) noexcept {
  std::size_t capacity = 8U;
  // At most 0.5 load keeps probe chains bounded and makes an unsuccessful
  // lookup terminate at an empty slot without auxiliary allocations.
  while (maximum_entries > capacity / 2U) {
    if (capacity > std::numeric_limits<std::size_t>::max() / 2U) {
      return std::nullopt;
    }
    capacity *= 2U;
  }
  return capacity;
}

// One deterministic winner per preprocessing voxel. Hash-table layout and
// insertion order are intentionally not observable: winners are selected only
// by (distance to voxel center, source index), then copied into the existing
// canonical rank/key ordering before the artifact is sealed.
class PreprocessVoxelSelection final {
public:
  explicit PreprocessVoxelSelection(std::size_t table_capacity) : slots_(table_capacity) {}

  [[nodiscard]] bool consider(const VoxelKey& key, Candidate candidate,
                              std::uint32_t source_index) noexcept {
    const std::size_t mask = slots_.size() - 1U;
    std::size_t slot_index = voxelKeyHash(key) & mask;
    for (std::size_t probe = 0U; probe < slots_.size(); ++probe) {
      Slot& slot = slots_[slot_index];
      if (!slot.occupied) {
        slot = Slot{true, key, candidate, source_index};
        ++size_;
        return true;
      }
      if (slot.key == key) {
        if (candidate.center_distance_squared < slot.candidate.center_distance_squared ||
            (candidate.center_distance_squared == slot.candidate.center_distance_squared &&
             source_index < slot.source_index)) {
          slot.candidate = candidate;
          slot.source_index = source_index;
        }
        return true;
      }
      slot_index = (slot_index + 1U) & mask;
    }
    return false;
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  void appendRankedCandidates(std::vector<RankedCandidate>& output) const {
    for (const Slot& slot : slots_) {
      if (!slot.occupied) {
        continue;
      }
      output.push_back(RankedCandidate{slot.key, slot.candidate, slot.source_index,
                                       spatialCandidateRank(slot.key, slot.source_index)});
    }
  }

private:
  struct Slot {
    bool occupied{};
    VoxelKey key;
    Candidate candidate;
    std::uint32_t source_index{};
  };

  std::vector<Slot> slots_;
  std::size_t size_{};
};

[[nodiscard]] bool coordinateCanAdd(std::int64_t coordinate, std::int64_t offset) noexcept {
  return !((offset < 0 && coordinate < std::numeric_limits<std::int64_t>::min() - offset) ||
           (offset > 0 && coordinate > std::numeric_limits<std::int64_t>::max() - offset));
}

}  // namespace

struct LidarRegistrationCloud::ExactIndex {
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

  double voxel_resolution_m{};
  std::size_t voxel_count{};
  std::vector<Slot> slots;
  // Point storage indices for each occupied slot form one contiguous span.
  std::vector<std::size_t> voxel_point_storage_indices;
  std::vector<std::size_t> canonical_point_storage_indices;
};

LidarRegistrationCloud::LidarRegistrationCloud(LidarRegistrationCloudData data,
                                               core::ObservationLineage sealed_lineage,
                                               core::ContentHash cloud_checksum,
                                               std::shared_ptr<const LidarCompositeTargetPoints>
                                                   composite_points,
                                               std::unique_ptr<const ExactIndex> exact_index)
    : source_sweep(data.source_sweep),
      reference_time(data.reference_time),
      T_odom_imu_seed(std::move(data.T_odom_imu_seed)),
      layout(data.layout),
      points(std::move(data.points)),
      imu_support(std::move(data.imu_support)),
      stats(data.stats),
      lineage(std::move(sealed_lineage)),
      checksum(cloud_checksum),
      composite_points_(std::move(composite_points)),
      exact_index_(std::move(exact_index)) {}

LidarRegistrationCloud::~LidarRegistrationCloud() = default;

core::Result<std::shared_ptr<const LidarRegistrationCloud>, LidarPreprocessError>
LidarRegistrationCloud::create(LidarRegistrationCloudData data,
                               LidarRegistrationIndexConfig index_config) {
  using Result = core::Result<std::shared_ptr<const LidarRegistrationCloud>, LidarPreprocessError>;
  if (!data.source_sweep.valid() || !finitePose(data.T_odom_imu_seed) ||
      !std::isfinite(index_config.voxel_resolution_m) || index_config.voxel_resolution_m <= 0.0) {
    return Result::failure(preprocessError(
        LidarPreprocessErrorCode::InvalidConfig,
        "LiDAR registration identity, pose seed, or exact-index resolution is invalid"));
  }
  if (!data.points_in_reference_imu || data.points_in_reference_imu->empty()) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::InvalidLayout,
                        "LiDAR registration cloud requires a non-empty tracking payload"));
  }
  if (data.points.empty()) {
    return Result::failure(preprocessError(LidarPreprocessErrorCode::NoUsablePoints,
                                           "LiDAR registration cloud has no point rows"));
  }

  const auto source_domain =
      checkedLayoutSourceDomain(data.layout, data.points_in_reference_imu->size());
  if (!source_domain) {
    return Result::failure(preprocessError(
        LidarPreprocessErrorCode::InvalidLayout,
        "LiDAR registration source-index domain is invalid or unreasonably sparse"));
  }

  auto& raw_points = *data.points_in_reference_imu;
  for (const core::LidarPoint& point : raw_points) {
    if (!finiteRawPoint(point) || static_cast<std::size_t>(point.source_index) >= *source_domain) {
      return Result::failure(preprocessError(
          LidarPreprocessErrorCode::InvalidLayout,
          "tracking LiDAR rows must be finite and lie inside the declared source-index domain"));
    }
  }
  const auto raw_source_order = [](const core::LidarPoint& lhs, const core::LidarPoint& rhs) {
    return lhs.source_index < rhs.source_index;
  };
  if (!std::is_sorted(raw_points.begin(), raw_points.end(), raw_source_order)) {
    std::sort(raw_points.begin(), raw_points.end(), raw_source_order);
  }
  if (std::adjacent_find(raw_points.begin(), raw_points.end(),
                         [](const core::LidarPoint& lhs, const core::LidarPoint& rhs) {
                           return lhs.source_index == rhs.source_index;
                         }) != raw_points.end()) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::InvalidLayout,
                        "tracking LiDAR source indices must be unique within the declared layout"));
  }

  for (const LidarRegistrationPoint& point : data.points) {
    if (!point.point.allFinite() || !std::isfinite(static_cast<double>(point.intensity)) ||
        static_cast<std::size_t>(point.source_index) >= *source_domain) {
      return Result::failure(preprocessError(
          LidarPreprocessErrorCode::NoUsablePoints,
          "LiDAR registration rows must be finite and bind valid tracking source rows"));
    }
  }
  const auto registration_source_order = [](const LidarRegistrationPoint& lhs,
                                            const LidarRegistrationPoint& rhs) {
    return lhs.source_index < rhs.source_index;
  };
  if (!std::is_sorted(data.points.begin(), data.points.end(), registration_source_order)) {
    std::sort(data.points.begin(), data.points.end(), registration_source_order);
  }
  if (std::adjacent_find(data.points.begin(), data.points.end(),
                         [](const LidarRegistrationPoint& lhs, const LidarRegistrationPoint& rhs) {
                           return lhs.source_index == rhs.source_index;
                         }) != data.points.end()) {
    return Result::failure(preprocessError(
        LidarPreprocessErrorCode::NoUsablePoints,
        "LiDAR registration source indices must be unique within the declared layout"));
  }
  auto raw_iterator = raw_points.begin();
  for (const LidarRegistrationPoint& point : data.points) {
    raw_iterator = std::lower_bound(raw_iterator, raw_points.end(), point.source_index,
                                    [](const core::LidarPoint& raw, std::uint32_t source_index) {
                                      return raw.source_index < source_index;
                                    });
    if (raw_iterator == raw_points.end() || raw_iterator->source_index != point.source_index ||
        !registrationPointExactlyBindsRaw(point, *raw_iterator)) {
      return Result::failure(preprocessError(
          LidarPreprocessErrorCode::NoUsablePoints,
          "every registration row must exactly match one owned tracking LiDAR row"));
    }
  }

  if (std::any_of(data.imu_support.begin(), data.imu_support.end(),
                  [](core::MeasurementId id) { return !id.valid(); })) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::InvalidLineage,
                        "LiDAR registration IMU support contains an invalid measurement ID"));
  }
  if (!std::is_sorted(data.imu_support.begin(), data.imu_support.end())) {
    std::sort(data.imu_support.begin(), data.imu_support.end());
  }
  data.imu_support.erase(std::unique(data.imu_support.begin(), data.imu_support.end()),
                         data.imu_support.end());
  if (!lineageExactlyConditionsOnImuSupport(data.lineage, data.imu_support)) {
    return Result::failure(preprocessError(
        LidarPreprocessErrorCode::InvalidLineage,
        "every deskew IMU measurement must appear exactly once as conditioning-only ancestry"));
  }
  if (data.stats.input_points != data.points_in_reference_imu->size() ||
      data.stats.valid_range_points > data.stats.input_points ||
      data.stats.deterministic_voxel_points != data.points.size() ||
      data.points.size() > data.stats.valid_range_points) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::InvalidConfig,
                        "LiDAR preprocessing statistics do not match the sealed payload"));
  }

  auto sealed_lineage = sealLineage(std::move(data.lineage), data.source_sweep);
  if (!sealed_lineage) {
    return Result::failure(sealed_lineage.error());
  }
  auto cloud_checksum =
      cloudChecksum(data, sealed_lineage.value(), index_config.voxel_resolution_m);
  if (!cloud_checksum) {
    return Result::failure(cloud_checksum.error());
  }

  struct IndexedPoint {
    VoxelKey key;
    std::size_t storage_index{};
  };
  std::vector<IndexedPoint> indexed_points;
  indexed_points.reserve(data.points.size());
  for (std::size_t storage_index = 0U; storage_index < data.points.size(); ++storage_index) {
    const auto key =
        checkedVoxelKey(data.points[storage_index].point, index_config.voxel_resolution_m);
    if (!key) {
      return Result::failure(preprocessError(
          LidarPreprocessErrorCode::SpatialIndexFailure,
          "LiDAR registration row cannot be represented in the configured exact index"));
    }
    indexed_points.push_back(IndexedPoint{*key, storage_index});
  }
  std::sort(
      indexed_points.begin(), indexed_points.end(),
      [&data](const IndexedPoint& lhs, const IndexedPoint& rhs) {
        return std::tie(lhs.key, data.points[lhs.storage_index].source_index, lhs.storage_index) <
               std::tie(rhs.key, data.points[rhs.storage_index].source_index, rhs.storage_index);
      });

  struct VoxelSpan {
    VoxelKey key;
    std::size_t point_begin{};
    std::size_t point_count{};
  };
  std::vector<VoxelSpan> voxel_spans;
  voxel_spans.reserve(indexed_points.size());
  auto exact_index = std::make_unique<ExactIndex>();
  exact_index->voxel_resolution_m = index_config.voxel_resolution_m;
  exact_index->voxel_point_storage_indices.reserve(indexed_points.size());
  for (std::size_t begin = 0U; begin < indexed_points.size();) {
    std::size_t end = begin + 1U;
    while (end < indexed_points.size() && indexed_points[end].key == indexed_points[begin].key) {
      ++end;
    }
    const std::size_t point_begin = exact_index->voxel_point_storage_indices.size();
    for (std::size_t index = begin; index < end; ++index) {
      exact_index->voxel_point_storage_indices.push_back(indexed_points[index].storage_index);
    }
    voxel_spans.push_back(VoxelSpan{indexed_points[begin].key, point_begin, end - begin});
    begin = end;
  }

  const auto table_capacity = openAddressTableCapacity(voxel_spans.size());
  if (!table_capacity) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::SpatialIndexFailure,
                        "LiDAR registration exact-index table capacity overflowed"));
  }
  exact_index->voxel_count = voxel_spans.size();
  exact_index->slots.resize(*table_capacity);
  const std::size_t slot_mask = exact_index->slots.size() - 1U;
  for (const VoxelSpan& span : voxel_spans) {
    std::size_t slot_index = voxelKeyHash(span.key) & slot_mask;
    while (exact_index->slots[slot_index].occupied) {
      slot_index = (slot_index + 1U) & slot_mask;
    }
    exact_index->slots[slot_index] =
        ExactIndex::Slot{true, span.key, span.point_begin, span.point_count};
  }
  exact_index->canonical_point_storage_indices.resize(data.points.size());
  std::iota(exact_index->canonical_point_storage_indices.begin(),
            exact_index->canonical_point_storage_indices.end(), 0U);

  auto composite_points = std::make_shared<LidarCompositeTargetPoints>();
  composite_points->reserve(data.points.size());
  for (const LidarRegistrationPoint& point : data.points) {
    composite_points->push_back(LidarCompositeTargetPoint{point.point, point.source_index});
  }

  auto cloud = std::shared_ptr<const LidarRegistrationCloud>(
      new LidarRegistrationCloud(std::move(data), std::move(sealed_lineage).value(),
                                 cloud_checksum.value(), std::move(composite_points),
                                 std::move(exact_index)));
  return Result::success(std::move(cloud));
}

double LidarRegistrationCloud::exactIndexVoxelResolutionM() const noexcept {
  return exact_index_->voxel_resolution_m;
}

std::size_t LidarRegistrationCloud::exactIndexVoxelCount() const noexcept {
  return exact_index_->voxel_count;
}

std::span<const std::size_t> LidarRegistrationCloud::canonicalPointStorageIndices() const noexcept {
  return exact_index_->canonical_point_storage_indices;
}

const std::shared_ptr<const LidarCompositeTargetPoints>&
LidarRegistrationCloud::compositeTargetPoints() const noexcept {
  return composite_points_;
}

ExactLidarNeighbor LidarRegistrationCloud::nearestExact(const Eigen::Vector3d& query,
                                                        double maximum_distance_m) const noexcept {
  ExactLidarNeighbor best;
  best.distance_squared_m2 = std::numeric_limits<double>::infinity();
  if (!query.allFinite() || !std::isfinite(maximum_distance_m) || maximum_distance_m <= 0.0) {
    return best;
  }
  const auto center = checkedVoxelKey(query, exact_index_->voxel_resolution_m);
  if (!center) {
    return best;
  }
  const double maximum_squared = maximum_distance_m * maximum_distance_m;
  const double radius_value = std::ceil(maximum_distance_m / exact_index_->voxel_resolution_m);
  // Registration profiles keep this near one. Reject pathological public
  // queries before integer conversion or cubic work amplification.
  constexpr double kMaximumExactVoxelRadius = 64.0;
  if (!std::isfinite(maximum_squared) || !std::isfinite(radius_value) || radius_value < 0.0 ||
      radius_value > kMaximumExactVoxelRadius) {
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
        const ExactIndex::Slot* slot =
            exact_index_->find(VoxelKey{center->x + dx, center->y + dy, center->z + dz});
        if (slot == nullptr) {
          continue;
        }
        for (std::size_t offset = 0U; offset < slot->point_count; ++offset) {
          ++best.points_examined;
          const std::size_t point_index =
              exact_index_->voxel_point_storage_indices[slot->point_begin + offset];
          const LidarRegistrationPoint& point = points[point_index];
          const double distance_squared = (point.point - query).squaredNorm();
          if (distance_squared > maximum_squared) {
            continue;
          }
          if (!best.found || distance_squared < best.distance_squared_m2 ||
              (distance_squared == best.distance_squared_m2 &&
               std::tie(point.source_index, point_index) <
                   std::tie(best.source_index, best.point_storage_index))) {
            best.found = true;
            best.point_storage_index = point_index;
            best.source_index = point.source_index;
            best.distance_squared_m2 = distance_squared;
          }
        }
      }
    }
  }
  return best;
}

core::Result<std::shared_ptr<const LidarRegistrationCloud>, LidarPreprocessError>
buildLidarRegistrationCloud(DeskewedSweep deskewed, core::ObservationLineage lineage,
                            const LidarPreprocessConfig& config,
                            LidarRegistrationIndexConfig index_config) {
  using Result = core::Result<std::shared_ptr<const LidarRegistrationCloud>, LidarPreprocessError>;
  if (!validConfig(config) || !std::isfinite(index_config.voxel_resolution_m) ||
      index_config.voxel_resolution_m <= 0.0) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::InvalidConfig,
                        "LiDAR preprocess or exact-index configuration is invalid"));
  }
  if (!deskewed.points_in_reference_imu || deskewed.points_in_reference_imu->empty()) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::EmptyInput, "deskewed point set is empty"));
  }
  const auto source_domain =
      checkedLayoutSourceDomain(deskewed.layout, deskewed.points_in_reference_imu->size());
  if (!source_domain) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::InvalidLayout,
                        "LiDAR layout source-index domain is invalid or unreasonably sparse"));
  }

  const auto& input = *deskewed.points_in_reference_imu;
  const auto selection_capacity = openAddressTableCapacity(input.size());
  if (!selection_capacity) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::SpatialIndexFailure,
                        "LiDAR preprocessing voxel-selection table capacity overflowed"));
  }
  std::vector<bool> source_index_seen(*source_domain, false);
  PreprocessVoxelSelection selected(*selection_capacity);
  std::size_t valid_count = 0U;
  for (std::size_t index = 0U; index < input.size(); ++index) {
    const core::LidarPoint& point = input[index];
    if (static_cast<std::size_t>(point.source_index) >= *source_domain) {
      return Result::failure(
          preprocessError(LidarPreprocessErrorCode::InvalidLayout,
                          "point source index lies outside the declared LiDAR layout"));
    }
    if (!finiteRawPoint(point)) {
      return Result::failure(
          preprocessError(LidarPreprocessErrorCode::InvalidLayout,
                          "tracking-deskewed LiDAR payload contains a non-finite row"));
    }
    if (source_index_seen[point.source_index]) {
      return Result::failure(preprocessError(LidarPreprocessErrorCode::InvalidLayout,
                                             "LiDAR source indices must be unique within a sweep"));
    }
    source_index_seen[point.source_index] = true;

    const Eigen::Vector3d position{point.x, point.y, point.z};
    const double range_m = position.norm();
    if (!std::isfinite(range_m) || range_m < config.minimum_range_m ||
        range_m > config.maximum_range_m) {
      continue;
    }
    ++valid_count;
    const auto key = checkedVoxelKey(position, config.voxel_size_m);
    if (!key) {
      return Result::failure(preprocessError(
          LidarPreprocessErrorCode::SpatialIndexFailure,
          "range-valid LiDAR point cannot be represented in the preprocessing voxel domain"));
    }
    const Eigen::Vector3d center{(static_cast<double>(key->x) + 0.5) * config.voxel_size_m,
                                 (static_cast<double>(key->y) + 0.5) * config.voxel_size_m,
                                 (static_cast<double>(key->z) + 0.5) * config.voxel_size_m};
    const double center_distance_squared = (position - center).squaredNorm();
    if (!selected.consider(*key, Candidate{index, center_distance_squared}, point.source_index)) {
      return Result::failure(
          preprocessError(LidarPreprocessErrorCode::SpatialIndexFailure,
                          "LiDAR preprocessing voxel-selection table unexpectedly exhausted"));
    }
  }
  if (selected.size() == 0U) {
    return Result::failure(
        preprocessError(LidarPreprocessErrorCode::NoUsablePoints,
                        "range and deterministic voxel filters removed every point"));
  }

  std::vector<RankedCandidate> bounded_candidates;
  bounded_candidates.reserve(selected.size());
  selected.appendRankedCandidates(bounded_candidates);
  const auto rank_order = [](const RankedCandidate& lhs, const RankedCandidate& rhs) {
    return std::tie(lhs.rank, lhs.key, lhs.source_index) <
           std::tie(rhs.rank, rhs.key, rhs.source_index);
  };
  if (bounded_candidates.size() > config.maximum_output_points) {
    std::nth_element(
        bounded_candidates.begin(),
        bounded_candidates.begin() + static_cast<std::ptrdiff_t>(config.maximum_output_points),
        bounded_candidates.end(), rank_order);
    bounded_candidates.resize(config.maximum_output_points);
  }
  std::sort(bounded_candidates.begin(), bounded_candidates.end(),
            [](const RankedCandidate& lhs, const RankedCandidate& rhs) {
              return std::tie(lhs.key, lhs.source_index) < std::tie(rhs.key, rhs.source_index);
            });

  LidarRegistrationCloudData output;
  output.source_sweep = deskewed.source;
  output.reference_time = deskewed.reference_time;
  output.T_odom_imu_seed = deskewed.T_odom_imu_reference;
  output.layout = deskewed.layout;
  output.stats.input_points = input.size();
  output.stats.valid_range_points = valid_count;
  output.stats.deterministic_voxel_points = bounded_candidates.size();
  output.lineage = std::move(lineage);
  output.points.reserve(bounded_candidates.size());
  for (const RankedCandidate& ranked : bounded_candidates) {
    const core::LidarPoint& raw = input[ranked.candidate.point_index];
    output.points.push_back(LidarRegistrationPoint{Eigen::Vector3d{raw.x, raw.y, raw.z},
                                                   raw.source_index, raw.intensity, raw.ring});
  }
  output.imu_support = std::move(deskewed.imu_support);
  output.points_in_reference_imu = std::move(deskewed.points_in_reference_imu);
  return LidarRegistrationCloud::create(std::move(output), index_config);
}

}  // namespace meridian::local
