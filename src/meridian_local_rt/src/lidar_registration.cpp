#include "meridian/local/lidar_registration.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "meridian/core/canonical_bytes.hpp"
#include "meridian/local/finalized_lidar_target_map.hpp"
#include "meridian/local/lidar_composite_target.hpp"

namespace meridian::local {
namespace {

using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;

// This is the single local LiDAR registration implementation. Its direct P2P
// residual and open-addressed voxel lookup follow Meridian v1's proven
// frontend lineage. Pose ownership, immutable snapshots, bounded parallel
// work, rank projection, and atomic FactorBatch admission belong to v2.
constexpr std::size_t kMaximumTargetCount{8U};

struct TargetIndex {
  LidarRegistrationTarget record;
};

struct CompositeOwnerBinding {
  std::size_t record_index{};
};

struct CompositeTargetContext {
  explicit CompositeTargetContext(LidarCompositeTarget composite_target)
      : target(std::move(composite_target)) {}

  LidarCompositeTarget target;
  std::vector<CompositeOwnerBinding> bindings;
};

struct WorkingTarget {
  const TargetIndex* index{};
  std::vector<FrozenPointCorrespondence> rows;
  std::size_t candidate_voxel_lookups{};
  std::size_t candidate_points_examined{};
  std::size_t source_rows_excluded_by_ownership{};
};

struct WorkingFinalizedMap {
  core::OdomEpoch odom_epoch;
  core::SensorInstanceId sensor;
  std::uint64_t version{};
  core::ContentHash checksum{};
  core::Pose3d T_odom_source_seed;
  std::vector<std::shared_ptr<const FinalizedLidarTargetOwner>> owners;
  std::vector<FrozenFinalizedMapPointCorrespondence> rows;
  std::size_t candidate_voxel_lookups{};
  std::size_t candidate_occupied_voxels{};
  std::size_t candidate_points_examined{};
  std::size_t source_rows_excluded_by_ownership{};
};

struct AssociationSet {
  std::vector<WorkingTarget> targets;
  std::optional<WorkingFinalizedMap> finalized_map;
  std::size_t correspondences{};
  double huber_delta_m{};
};

enum class AssociationRowStatus : std::uint8_t {
  NoOwner,
  Owned,
  NumericalFailure,
};

enum class AssociationFailureKind : std::uint8_t {
  None,
  SourceTransform,
  TargetMetric,
};

struct Observability {
  Vector6 eigenvalues{Vector6::Zero()};
  Matrix6 eigenvectors{Matrix6::Identity()};
  double threshold{};
  std::size_t rank{};
};

struct Evaluation {
  double robust_cost{};
  double effective_correspondences{};
  double maximum_squared_residual_m2{};
  Matrix6 hessian{Matrix6::Zero()};
  Vector6 gradient{Vector6::Zero()};
  Observability observability;
};

struct EvaluationRowResult {
  bool valid{};
  double robust_cost{};
  double effective_correspondence{};
  double squared_residual_m2{};
  Matrix6 hessian{Matrix6::Zero()};
  Vector6 gradient{Vector6::Zero()};
};

struct TargetCandidate {
  bool present{};
  std::size_t target_point_storage_index{};
  double distance_squared{std::numeric_limits<double>::infinity()};
  std::uint32_t target_source_index{};
  std::size_t voxel_lookups{};
  std::size_t points_examined{};
};

struct FinalizedMapCandidate {
  bool present{};
  FinalizedLidarTargetPoint point;
  double distance_squared{std::numeric_limits<double>::infinity()};
  std::size_t voxel_lookups{};
  std::size_t occupied_voxels{};
  std::size_t points_examined{};
};

struct AssociationRowResult {
  AssociationRowStatus status{AssociationRowStatus::NoOwner};
  AssociationFailureKind failure_kind{AssociationFailureKind::None};
  std::size_t owner_target_index{};
  std::size_t failure_target_index{};
  std::array<TargetCandidate, kMaximumTargetCount> candidates;
  FinalizedMapCandidate finalized_map_candidate;
  std::size_t composite_voxel_lookups{};
  std::size_t composite_occupied_voxels{};
  std::size_t composite_points_examined{};
  bool finalized_map_stale{};
};

struct LidarRegistrationWorkspace {
  std::vector<AssociationRowResult> association_rows;
  std::vector<EvaluationRowResult> evaluation_rows;
};

[[nodiscard]] auto frozenRowCanonicalIdentity(const FrozenPointCorrespondence& row) noexcept {
  return std::tuple{row.source_index, row.target_source_index, row.source_point_storage_index,
                    row.target_point_storage_index};
}

[[nodiscard]] bool frozenRowCanonicalLess(const FrozenPointCorrespondence& lhs,
                                          const FrozenPointCorrespondence& rhs) noexcept {
  return frozenRowCanonicalIdentity(lhs) < frozenRowCanonicalIdentity(rhs);
}

[[nodiscard]] auto finalizedOwnerCanonicalIdentity(
    const FinalizedLidarTargetOwner& owner) noexcept {
  return std::tie(owner.finalized_state.odom_epoch, owner.finalized_state.state,
                  owner.finalized_state.exact_time, owner.sweep, owner.batch.sensor,
                  owner.batch.batch_id, owner.finalized_state.final_revision,
                  owner.admission.health.recovery_epoch, owner.calibration,
                  owner.cloud_checksum, owner.final_pose_covariance_checksum);
}

[[nodiscard]] bool finalizedOwnerCanonicalLess(
    const std::shared_ptr<const FinalizedLidarTargetOwner>& lhs,
    const std::shared_ptr<const FinalizedLidarTargetOwner>& rhs) noexcept {
  if (!lhs) {
    return static_cast<bool>(rhs);
  }
  if (!rhs) {
    return false;
  }
  return finalizedOwnerCanonicalIdentity(*lhs) < finalizedOwnerCanonicalIdentity(*rhs);
}

[[nodiscard]] auto frozenFinalizedMapRowCanonicalIdentity(
    const FrozenFinalizedMapPointCorrespondence& row) noexcept {
  return std::tuple{row.source_index, row.source_point_storage_index, row.owner_index,
                    row.target_source_index};
}

[[nodiscard]] bool frozenFinalizedMapRowCanonicalLess(
    const FrozenFinalizedMapPointCorrespondence& lhs,
    const FrozenFinalizedMapPointCorrespondence& rhs) noexcept {
  return frozenFinalizedMapRowCanonicalIdentity(lhs) < frozenFinalizedMapRowCanonicalIdentity(rhs);
}

[[nodiscard]] LidarRegistrationError makeError(LidarRegistrationErrorCode code, std::string detail,
                                               const LidarRegistrationWorkCounters& work) {
  return LidarRegistrationError{code, std::move(detail), work};
}

[[nodiscard]] bool checkedAdd(std::size_t increment, std::size_t* value) noexcept {
  if (increment > std::numeric_limits<std::size_t>::max() - *value) {
    return false;
  }
  *value += increment;
  return true;
}

[[nodiscard]] bool checkedAddDuration(const core::CpuWallDuration& increment,
                                      core::CpuWallDuration* value) noexcept {
  if (increment.wall.nanoseconds < 0 ||
      increment.wall.nanoseconds >
          std::numeric_limits<std::int64_t>::max() - value->wall.nanoseconds) {
    return false;
  }
  value->wall.nanoseconds += increment.wall.nanoseconds;
  if (!increment.thread_cpu) {
    return true;
  }
  if (!value->thread_cpu) {
    value->thread_cpu = core::Duration{0};
  }
  if (increment.thread_cpu->nanoseconds < 0 ||
      increment.thread_cpu->nanoseconds >
          std::numeric_limits<std::int64_t>::max() - value->thread_cpu->nanoseconds) {
    return false;
  }
  value->thread_cpu->nanoseconds += increment.thread_cpu->nanoseconds;
  return true;
}

[[nodiscard]] bool finitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool finitePose(const core::Pose3d& pose) noexcept {
  return pose.matrix().allFinite();
}

[[nodiscard]] bool validConfig(const LidarRegistrationConfig& config) noexcept {
  return finitePositive(config.target_voxel_resolution_m) &&
         finitePositive(config.source_voxel_size_m) &&
         finitePositive(config.maximum_correspondence_distance_m) &&
         std::isfinite(config.maximum_correspondence_distance_m *
                       config.maximum_correspondence_distance_m) &&
         config.maximum_voxel_search_radius > 0U && config.maximum_voxel_search_radius <= 64U &&
         std::ceil(config.maximum_correspondence_distance_m / config.target_voxel_resolution_m) <=
             static_cast<double>(config.maximum_voxel_search_radius) &&
         config.maximum_source_points > 0U && config.maximum_target_points_per_target > 0U &&
         config.maximum_targets > 0U && config.maximum_targets <= kMaximumTargetCount &&
         config.maximum_composite_owners > 0U &&
         config.maximum_composite_owners <=
             static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()) + 1U &&
         config.maximum_composite_indexed_points > 0U &&
         config.maximum_composite_points_per_voxel > 0U &&
         config.maximum_composite_points_per_voxel <= 64U &&
         config.parallel_worker_count > 0U && config.parallel_worker_count <= 64U &&
         config.parallel_worker_count <=
             static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         config.minimum_correspondences > 0U &&
         finitePositive(config.residual_standard_deviation_m) &&
         std::isfinite(config.residual_standard_deviation_m *
                       config.residual_standard_deviation_m) &&
         finitePositive(config.huber_delta_multiplier) &&
         finitePositive(config.minimum_huber_delta_m) &&
         finitePositive(config.maximum_huber_delta_m) &&
         config.maximum_huber_delta_m >= config.minimum_huber_delta_m &&
         config.maximum_huber_delta_m <= config.maximum_correspondence_distance_m &&
         config.maximum_outer_iterations > 0U && config.maximum_lm_damping_retries <= 16U &&
         finitePositive(config.initial_relative_damping) &&
         std::isfinite(config.damping_increase) && config.damping_increase > 1.0 &&
         finitePositive(config.translation_convergence_m) &&
         finitePositive(config.rotation_convergence_rad) &&
         finitePositive(config.maximum_correction_translation_m) &&
         finitePositive(config.maximum_correction_rotation_rad) &&
         config.minimum_observable_rank > 0U && config.minimum_observable_rank <= 6U &&
         finitePositive(config.absolute_normalized_observable_eigenvalue) &&
         std::isfinite(config.relative_normalized_observable_eigenvalue) &&
         config.relative_normalized_observable_eigenvalue >= 0.0 &&
         config.relative_normalized_observable_eigenvalue < 1.0 &&
         finitePositive(config.minimum_characteristic_length_m) &&
         finitePositive(config.maximum_characteristic_length_m) &&
         config.maximum_characteristic_length_m >= config.minimum_characteristic_length_m &&
         finitePositive(config.maximum_translation_information) &&
         finitePositive(config.seed_translation_consistency_tolerance_m) &&
         finitePositive(config.seed_rotation_consistency_tolerance_rad);
}

[[nodiscard]] Eigen::Matrix3d skew(const Eigen::Vector3d& vector) noexcept {
  Eigen::Matrix3d matrix;
  matrix << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(),
      0.0;
  return matrix;
}

[[nodiscard]] double huberWeight(double norm, double delta) noexcept {
  return norm <= delta || norm <= std::numeric_limits<double>::epsilon() ? 1.0 : delta / norm;
}

struct SourceVoxelKey {
  std::int64_t x{};
  std::int64_t y{};
  std::int64_t z{};

  bool operator==(const SourceVoxelKey&) const = default;
};

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] std::uint64_t sourceVoxelHash(const SourceVoxelKey& key) noexcept {
  return mix64(static_cast<std::uint64_t>(key.x)) ^
         (mix64(static_cast<std::uint64_t>(key.y)) << 1U) ^
         (mix64(static_cast<std::uint64_t>(key.z)) << 7U);
}

[[nodiscard]] std::optional<SourceVoxelKey> sourceVoxelKey(const Eigen::Vector3d& point,
                                                           double voxel_size_m) noexcept {
  if (!point.allFinite() || !finitePositive(voxel_size_m)) {
    return std::nullopt;
  }
  const Eigen::Array3d scaled = point.array() / voxel_size_m;
  constexpr double kMin = static_cast<double>(std::numeric_limits<std::int64_t>::min());
  constexpr double kMax = static_cast<double>(std::numeric_limits<std::int64_t>::max());
  if ((scaled < kMin).any() || (scaled > kMax).any()) {
    return std::nullopt;
  }
  return SourceVoxelKey{static_cast<std::int64_t>(std::floor(scaled.x())),
                        static_cast<std::int64_t>(std::floor(scaled.y())),
                        static_cast<std::int64_t>(std::floor(scaled.z()))};
}

class SourceVoxelSet {
public:
  explicit SourceVoxelSet(std::size_t expected) {
    std::size_t capacity = 16U;
    while (capacity < expected * 2U && capacity <= std::numeric_limits<std::size_t>::max() / 2U) {
      capacity *= 2U;
    }
    slots_.resize(capacity);
  }

  [[nodiscard]] bool insert(const SourceVoxelKey& key) noexcept {
    const std::size_t mask = slots_.size() - 1U;
    std::size_t index = static_cast<std::size_t>(sourceVoxelHash(key)) & mask;
    for (std::size_t probe = 0U; probe < slots_.size(); ++probe) {
      Slot& slot = slots_[index];
      if (!slot.occupied) {
        slot.occupied = true;
        slot.key = key;
        return true;
      }
      if (slot.key == key) {
        return false;
      }
      index = (index + 1U) & mask;
    }
    return false;
  }

private:
  struct Slot {
    bool occupied{};
    SourceVoxelKey key;
  };
  std::vector<Slot> slots_;
};

[[nodiscard]] double sourceCharacteristicLength(const LidarRegistrationCloud& source,
                                                std::span<const std::size_t> selected,
                                                const LidarRegistrationConfig& config) {
  std::vector<double> ranges;
  ranges.reserve(selected.size());
  for (const std::size_t index : selected) {
    ranges.push_back(source.points[index].point.norm());
  }
  const std::size_t middle = ranges.size() / 2U;
  std::nth_element(ranges.begin(), ranges.begin() + static_cast<std::ptrdiff_t>(middle),
                   ranges.end());
  return std::clamp(ranges[middle], config.minimum_characteristic_length_m,
                    config.maximum_characteristic_length_m);
}

[[nodiscard]] core::Result<std::vector<std::size_t>, LidarRegistrationError> selectSourcePoints(
    const std::shared_ptr<const LidarRegistrationCloud>& source,
    const LidarRegistrationConfig& config, LidarRegistrationWorkCounters* work) {
  using Result = core::Result<std::vector<std::size_t>, LidarRegistrationError>;
  if (!source || !source->source_sweep.valid() || source->points.empty() ||
      !finitePose(source->T_odom_imu_seed) ||
      source->exactIndexVoxelResolutionM() != config.target_voxel_resolution_m) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidSource,
        "direct point-to-point ICP source identity, seed, or sealed-index profile is invalid",
        *work));
  }
  work->source_points_considered = source->points.size();
  const std::span<const std::size_t> canonical = source->canonicalPointStorageIndices();
  std::vector<std::size_t> selected;
  selected.reserve(std::min(canonical.size(), config.maximum_source_points));
  SourceVoxelSet occupied(canonical.size());
  for (const std::size_t storage_index : canonical) {
    if (storage_index >= source->points.size()) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::InvalidSource,
          "direct point-to-point ICP source canonical index is outside the sealed cloud", *work));
    }
    const auto key =
        sourceVoxelKey(source->points[storage_index].point, config.source_voxel_size_m);
    if (!key) {
      ++work->invalid_source_points;
      continue;
    }
    if (occupied.insert(*key)) {
      selected.push_back(storage_index);
    }
  }
  if (selected.size() > config.maximum_source_points) {
    std::vector<std::size_t> bounded;
    bounded.reserve(config.maximum_source_points);
    if (config.maximum_source_points == 1U) {
      bounded.push_back(selected.front());
    } else {
      for (std::size_t output_index = 0U; output_index < config.maximum_source_points;
           ++output_index) {
        const std::size_t input_index =
            output_index * (selected.size() - 1U) / (config.maximum_source_points - 1U);
        bounded.push_back(selected[input_index]);
      }
    }
    selected = std::move(bounded);
  }
  if (selected.empty()) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidSource,
        "direct point-to-point ICP source voxel selection produced no usable point", *work));
  }
  work->source_points_selected = selected.size();
  work->source_points_omitted_by_capacity = canonical.size() - selected.size();
  return Result::success(std::move(selected));
}

[[nodiscard]] core::Result<std::vector<TargetIndex>, LidarRegistrationError> buildTargetIndices(
    core::StateId source_state, const LidarRegistrationCloud& source,
    std::span<const LidarRegistrationTarget> records, const LidarRegistrationConfig& config,
    LidarRegistrationWorkCounters* work) {
  using Result = core::Result<std::vector<TargetIndex>, LidarRegistrationError>;
  if (records.size() > config.maximum_targets) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidTarget,
        "pose-aware direct point-to-point ICP target count is outside its bounded profile", *work));
  }
  std::vector<LidarRegistrationTarget> canonical(records.begin(), records.end());
  std::sort(canonical.begin(), canonical.end(),
            [](const LidarRegistrationTarget& lhs, const LidarRegistrationTarget& rhs) {
              return std::tie(lhs.state, lhs.time) < std::tie(rhs.state, rhs.time);
            });

  std::vector<TargetIndex> output;
  output.reserve(canonical.size());
  std::set<core::StateId> target_states;
  std::set<core::FusionTime> target_times;
  std::set<core::MeasurementId> target_sweeps;
  std::set<const LidarRegistrationCloud*> target_clouds;
  for (const LidarRegistrationTarget& record : canonical) {
    if (!record.state.valid() || !record.cloud || !record.cloud->source_sweep.valid() ||
        record.cloud->points.empty() ||
        record.cloud->points.size() > config.maximum_target_points_per_target ||
        record.cloud->exactIndexVoxelResolutionM() != config.target_voxel_resolution_m ||
        record.time != record.cloud->reference_time || !finitePose(record.T_odom_imu_target_seed) ||
        !finitePose(record.T_target_source_seed)) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::InvalidTarget,
          "direct point-to-point ICP target identity, time, cloud, capacity, or seed is invalid",
          *work));
    }
    if (!(record.state < source_state) || !(record.time < source.reference_time) ||
        record.cloud.get() == std::addressof(source) ||
        record.cloud->source_sweep == source.source_sweep) {
      return Result::failure(makeError(LidarRegistrationErrorCode::InvalidTarget,
                                       "direct point-to-point ICP targets must be distinct from "
                                       "and temporally older than the source",
                                       *work));
    }
    if (!target_states.insert(record.state).second || !target_times.insert(record.time).second ||
        !target_sweeps.insert(record.cloud->source_sweep).second ||
        !target_clouds.insert(record.cloud.get()).second) {
      return Result::failure(makeError(LidarRegistrationErrorCode::InvalidTarget,
                                       "pose-aware direct point-to-point ICP target state, time, "
                                       "sweep, and cloud identities must be unique",
                                       *work));
    }
    const core::Pose3d expected = record.T_odom_imu_target_seed.inverse() * source.T_odom_imu_seed;
    const core::Pose3d discrepancy = record.T_target_source_seed.inverse() * expected;
    if (!finitePose(discrepancy) ||
        discrepancy.translation().norm() > config.seed_translation_consistency_tolerance_m ||
        discrepancy.so3().log().norm() > config.seed_rotation_consistency_tolerance_rad) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::InconsistentSeed,
          "T_target_source seed is inconsistent with immutable cloud odometry seeds", *work));
    }

    // The relative seed is a redundant API field. Validate it above, then
    // canonicalize the internal objective from the same two odometry poses
    // used by the live composite index. This prevents an allowed sub-micron
    // wire-rounding difference from defining a second registration geometry.
    LidarRegistrationTarget canonical_record = record;
    canonical_record.T_target_source_seed = expected;
    output.push_back(TargetIndex{std::move(canonical_record)});
  }
  return Result::success(std::move(output));
}

[[nodiscard]] LidarRegistrationErrorCode compositeErrorCode(
    LidarCompositeTargetErrorCode code) noexcept {
  switch (code) {
    case LidarCompositeTargetErrorCode::InvalidConfig:
      return LidarRegistrationErrorCode::InvalidConfig;
    case LidarCompositeTargetErrorCode::EmptyOwners:
    case LidarCompositeTargetErrorCode::InvalidOwner:
    case LidarCompositeTargetErrorCode::DuplicateOwner:
    case LidarCompositeTargetErrorCode::InvalidGeometry:
    case LidarCompositeTargetErrorCode::Capacity:
      return LidarRegistrationErrorCode::InvalidTarget;
    case LidarCompositeTargetErrorCode::SpatialIndexFailure:
    case LidarCompositeTargetErrorCode::ChecksumFailure:
      return LidarRegistrationErrorCode::NumericalFailure;
  }
  return LidarRegistrationErrorCode::NumericalFailure;
}

[[nodiscard]] bool recordCompositeBuild(
    const core::CpuWallDuration& duration, const LidarCompositeTargetBuildStats* statistics,
    LidarRegistrationWorkCounters* work) noexcept {
  LidarRegistrationWorkCounters updated = *work;
  if (!checkedAdd(1U, &updated.composite_index_builds) ||
      !checkedAddDuration(duration, &updated.composite_index_build_duration)) {
    return false;
  }
  if (statistics != nullptr &&
      (!checkedAdd(statistics->input_owners, &updated.composite_index_input_owners) ||
       !checkedAdd(statistics->input_points, &updated.composite_index_input_points) ||
       !checkedAdd(statistics->retained_owners, &updated.composite_index_retained_owners) ||
       !checkedAdd(statistics->retained_points, &updated.composite_index_retained_points) ||
       !checkedAdd(statistics->occupied_voxels, &updated.composite_index_occupied_voxels) ||
       !checkedAdd(statistics->occupied_voxels, &updated.target_index_voxels) ||
       !checkedAdd(statistics->per_voxel_capacity_discarded_points,
                   &updated.composite_index_per_voxel_discarded_points) ||
       !checkedAdd(statistics->total_capacity_discarded_points,
                   &updated.composite_index_total_discarded_points))) {
    return false;
  }
  *work = std::move(updated);
  return true;
}

[[nodiscard]] core::Result<CompositeTargetContext, LidarRegistrationError> buildCompositeTarget(
    const std::vector<TargetIndex>& targets, core::LocalGraphRevision live_pose_revision,
    const LidarRegistrationConfig& config, LidarRegistrationWorkCounters* work) {
  using Result = core::Result<CompositeTargetContext, LidarRegistrationError>;
  core::ThreadCpuWallTimer timer;
  std::vector<LidarCompositeTargetOwnerInput> owners;
  owners.reserve(targets.size());

  const auto failBeforeCreate = [&](std::string detail) {
    const core::CpuWallDuration duration = timer.elapsed();
    if (!recordCompositeBuild(duration, nullptr, work)) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "composite target build timing or work counter overflowed", *work));
    }
    return Result::failure(
        makeError(LidarRegistrationErrorCode::InvalidTarget, std::move(detail), *work));
  };

  if (targets.empty() || !live_pose_revision.valid()) {
    return failBeforeCreate("live composite target pose revision is invalid");
  }
  for (const TargetIndex& target : targets) {
    owners.push_back(LidarCompositeTargetOwnerInput{
        LidarCompositeOwnerKind::Live,
        target.record.state,
        target.record.time,
        target.record.cloud->source_sweep,
        live_pose_revision,
        target.record.cloud->checksum,
        target.record.T_odom_imu_target_seed,
        target.record.cloud->compositeTargetPoints(),
    });
  }
  LidarCompositeTargetConfig composite_config;
  composite_config.voxel_size_m = config.target_voxel_resolution_m;
  composite_config.maximum_points_per_voxel = config.maximum_composite_points_per_voxel;
  composite_config.maximum_total_points = config.maximum_composite_indexed_points;
  composite_config.maximum_owners = config.maximum_composite_owners;
  composite_config.maximum_query_distance_m = config.maximum_correspondence_distance_m;
  composite_config.maximum_voxel_search_radius = config.maximum_voxel_search_radius;
  auto built = LidarCompositeTarget::create(composite_config, std::move(owners));
  const core::CpuWallDuration duration = timer.elapsed();
  const LidarCompositeTargetBuildStats* statistics =
      built ? std::addressof(built.value().statistics()) : nullptr;
  if (!recordCompositeBuild(duration, statistics, work)) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "composite target build timing or work counter overflowed", *work));
  }
  if (!built) {
    return Result::failure(makeError(compositeErrorCode(built.error().code),
                                     "composite target build failed: " + built.error().detail,
                                     *work));
  }

  CompositeTargetContext context(std::move(built).value());
  context.bindings.reserve(context.target.owners().size());
  for (const LidarCompositeTargetOwnerInput& owner : context.target.owners()) {
    std::optional<std::size_t> binding;
    if (owner.kind != LidarCompositeOwnerKind::Live) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "live composite target unexpectedly contains finalized geometry", *work));
    }
    for (std::size_t index = 0U; index < targets.size(); ++index) {
      if (targets[index].record.state == owner.state &&
          targets[index].record.cloud->source_sweep == owner.sweep &&
          targets[index].record.cloud->checksum == owner.geometry_checksum) {
        if (binding) {
          return Result::failure(makeError(
              LidarRegistrationErrorCode::NumericalFailure,
              "composite target canonical live owner binding is ambiguous", *work));
        }
        binding = index;
      }
    }
    if (!binding) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "composite target canonical owner cannot be rebound to factor provenance", *work));
    }
    context.bindings.push_back(CompositeOwnerBinding{*binding});
  }
  return Result::success(std::move(context));
}

[[nodiscard]] AssociationRowResult associateSourceRow(
    const LidarRegistrationCloud& source, std::size_t source_point_index,
    const std::vector<TargetIndex>& targets, const FinalizedLidarTargetReadView* finalized_map,
    const CompositeTargetContext* live_composite, const core::Pose3d& T_odom_source,
    const LidarRegistrationConfig& config) {
  AssociationRowResult output;
  const LidarRegistrationPoint& source_point = source.points[source_point_index];
  const Eigen::Vector3d query_odom = T_odom_source * source_point.point;
  if (!query_odom.allFinite()) {
    output.status = AssociationRowStatus::NumericalFailure;
    output.failure_kind = AssociationFailureKind::SourceTransform;
    output.failure_target_index = targets.size();
    return output;
  }
  const auto targetFailure = [&]() {
    output.status = AssociationRowStatus::NumericalFailure;
    output.failure_kind = AssociationFailureKind::TargetMetric;
    output.failure_target_index = targets.size();
    return output;
  };

  LidarCompositeTargetNeighbor live_neighbor;
  if (live_composite != nullptr) {
    live_neighbor = live_composite->target.nearestExact(
        query_odom, config.maximum_correspondence_distance_m);
    output.composite_voxel_lookups = live_neighbor.voxel_lookups;
    output.composite_occupied_voxels = live_neighbor.occupied_voxels;
    output.composite_points_examined = live_neighbor.points_examined;
    if (!live_neighbor.query_valid ||
        (live_neighbor.found &&
         (live_neighbor.owner_slot >= live_composite->bindings.size() ||
          !live_neighbor.point_owner.allFinite() || !live_neighbor.point_odom.allFinite() ||
          !std::isfinite(live_neighbor.distance_squared_m2) ||
          live_neighbor.distance_squared_m2 < 0.0))) {
      return targetFailure();
    }
  }

  FinalizedLidarTargetNeighbor map_neighbor;
  if (finalized_map != nullptr) {
    map_neighbor = finalized_map->nearestExact(query_odom,
                                               config.maximum_correspondence_distance_m);
    output.finalized_map_candidate.voxel_lookups = map_neighbor.voxel_lookups;
    output.finalized_map_candidate.occupied_voxels = map_neighbor.occupied_voxels;
    output.finalized_map_candidate.points_examined = map_neighbor.points_examined;
    if (!checkedAdd(map_neighbor.voxel_lookups, &output.composite_voxel_lookups) ||
        !checkedAdd(map_neighbor.occupied_voxels, &output.composite_occupied_voxels) ||
        !checkedAdd(map_neighbor.points_examined, &output.composite_points_examined)) {
      return targetFailure();
    }
    if (!map_neighbor.view_current) {
      output.finalized_map_stale = true;
      return targetFailure();
    }
    if (!map_neighbor.query_valid ||
        (map_neighbor.found &&
         (!map_neighbor.point.owner || !map_neighbor.point.point_odom.allFinite() ||
          !std::isfinite(map_neighbor.distance_squared_m2) ||
          map_neighbor.distance_squared_m2 < 0.0))) {
      return targetFailure();
    }
  }

  // Both physical indexes are searched. The exact global winner owns this
  // source row; live wins the bit-exact distance tie so binary constraints
  // are never displaced by coincident fixed-map geometry.
  const bool choose_live =
      live_neighbor.found &&
      (!map_neighbor.found || live_neighbor.distance_squared_m2 <= map_neighbor.distance_squared_m2);
  if (choose_live) {
    const CompositeOwnerBinding& binding =
        live_composite->bindings[live_neighbor.owner_slot];
    if (binding.record_index >= targets.size() ||
        live_neighbor.local_point_index >=
            targets[binding.record_index].record.cloud->points.size()) {
      output.status = AssociationRowStatus::NumericalFailure;
      output.failure_kind = AssociationFailureKind::TargetMetric;
      output.failure_target_index = binding.record_index;
      return output;
    }
    const LidarRegistrationPoint& target_point =
        targets[binding.record_index].record.cloud->points[live_neighbor.local_point_index];
    if (target_point.source_index != live_neighbor.source_index ||
        !target_point.point.allFinite()) {
      output.status = AssociationRowStatus::NumericalFailure;
      output.failure_kind = AssociationFailureKind::TargetMetric;
      output.failure_target_index = binding.record_index;
      return output;
    }
    TargetCandidate& candidate = output.candidates[binding.record_index];
    candidate.present = true;
    candidate.target_point_storage_index = live_neighbor.local_point_index;
    candidate.distance_squared = live_neighbor.distance_squared_m2;
    candidate.target_source_index = live_neighbor.source_index;
    candidate.voxel_lookups = live_neighbor.voxel_lookups;
    candidate.points_examined = live_neighbor.points_examined;
    output.owner_target_index = binding.record_index;
    output.status = AssociationRowStatus::Owned;
    return output;
  }
  if (map_neighbor.found) {
    FinalizedMapCandidate& candidate = output.finalized_map_candidate;
    candidate.present = true;
    candidate.point = map_neighbor.point;
    candidate.distance_squared = map_neighbor.distance_squared_m2;
    output.owner_target_index = targets.size();
    output.status = AssociationRowStatus::Owned;
  }
  return output;
}

[[nodiscard]] core::Result<AssociationSet, LidarRegistrationError> buildAssociations(
    const LidarRegistrationCloud& source, std::span<const std::size_t> selected_source,
    const std::vector<TargetIndex>& targets, const core::Pose3d& source_right_correction,
    const FinalizedLidarTargetReadView* finalized_map,
    const CompositeTargetContext* live_composite, const LidarRegistrationConfig& config,
    LidarRegistrationWorkspace* workspace, LidarRegistrationWorkCounters* work) {
  using Result = core::Result<AssociationSet, LidarRegistrationError>;
  AssociationSet output;
  output.targets.reserve(targets.size());
  for (const TargetIndex& target : targets) {
    WorkingTarget working;
    working.index = &target;
    output.targets.push_back(std::move(working));
  }
  if (finalized_map != nullptr) {
    output.finalized_map.emplace();
    output.finalized_map->odom_epoch = finalized_map->odomEpoch();
    output.finalized_map->sensor = finalized_map->sensor();
    output.finalized_map->version = finalized_map->version();
    output.finalized_map->checksum = finalized_map->checksum();
    output.finalized_map->T_odom_source_seed = source.T_odom_imu_seed;
  }
  const std::size_t channel_count = targets.size() + (finalized_map != nullptr ? 1U : 0U);
  if (channel_count == 0U || channel_count > kMaximumTargetCount ||
      ((live_composite == nullptr) != targets.empty()) ||
      (live_composite != nullptr &&
       live_composite->bindings.size() != live_composite->target.owners().size()) ||
      selected_source.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      std::any_of(selected_source.begin(), selected_source.end(),
                  [&](std::size_t index) { return index >= source.points.size(); })) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::InvalidSource,
                  "direct point-to-point ICP association source-row domain is invalid", *work));
  }

  const core::Pose3d T_odom_source = source.T_odom_imu_seed * source_right_correction;
  if (!finitePose(T_odom_source)) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "direct point-to-point ICP odometry-frame association pose is non-finite", *work));
  }

  workspace->association_rows.resize(selected_source.size());
  const std::int64_t row_count = static_cast<std::int64_t>(selected_source.size());
  const int worker_count = static_cast<int>(config.parallel_worker_count);
#pragma omp parallel for schedule(static) num_threads(worker_count)
  for (std::int64_t row_index = 0; row_index < row_count; ++row_index) {
    const std::size_t canonical_index = static_cast<std::size_t>(row_index);
    workspace->association_rows[canonical_index] = associateSourceRow(
        source, selected_source[canonical_index], targets, finalized_map, live_composite,
        T_odom_source, config);
  }

  std::array<std::size_t, kMaximumTargetCount> owner_counts{};
  std::array<std::size_t, kMaximumTargetCount> excluded_counts{};
  std::array<std::size_t, kMaximumTargetCount> target_voxel_lookups{};
  std::array<std::size_t, kMaximumTargetCount> target_points_examined{};
  std::size_t map_owner_count = 0U;
  std::size_t map_excluded_count = 0U;
  std::size_t map_voxel_lookups = 0U;
  std::size_t map_occupied_voxels = 0U;
  std::size_t map_points_examined = 0U;
  std::size_t composite_voxel_lookups = 0U;
  std::size_t composite_occupied_voxels = 0U;
  std::size_t composite_points_examined = 0U;
  std::size_t invalid_source_points = 0U;
  std::size_t invalid_target_points = 0U;
  std::size_t map_stale_fallbacks = 0U;
  bool saw_numerical_failure = false;
  for (const AssociationRowResult& result : workspace->association_rows) {
    if (!checkedAdd(result.composite_voxel_lookups, &composite_voxel_lookups) ||
        !checkedAdd(result.composite_occupied_voxels, &composite_occupied_voxels) ||
        !checkedAdd(result.composite_points_examined, &composite_points_examined)) {
      return Result::failure(
          makeError(LidarRegistrationErrorCode::NumericalFailure,
                    "composite target correspondence work counter overflowed", *work));
    }
    if (finalized_map != nullptr &&
        (!checkedAdd(result.finalized_map_candidate.voxel_lookups, &map_voxel_lookups) ||
         !checkedAdd(result.finalized_map_candidate.occupied_voxels, &map_occupied_voxels) ||
         !checkedAdd(result.finalized_map_candidate.points_examined, &map_points_examined) ||
         (result.finalized_map_stale && !checkedAdd(1U, &map_stale_fallbacks)))) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "persistent finalized-map correspondence work counter overflowed", *work));
    }
    if (result.status == AssociationRowStatus::NumericalFailure) {
      saw_numerical_failure = true;
      if (result.failure_kind == AssociationFailureKind::SourceTransform) {
        if (!checkedAdd(1U, &invalid_source_points)) {
          return Result::failure(
              makeError(LidarRegistrationErrorCode::NumericalFailure,
                        "direct point-to-point ICP invalid-source counter overflowed", *work));
        }
      } else if (!checkedAdd(1U, &invalid_target_points)) {
        return Result::failure(
            makeError(LidarRegistrationErrorCode::NumericalFailure,
                      "direct point-to-point ICP invalid-target counter overflowed", *work));
      }
      continue;
    }
    if (result.status != AssociationRowStatus::Owned) {
      continue;
    }
    if (result.owner_target_index == targets.size()) {
      if (finalized_map == nullptr || !result.finalized_map_candidate.present ||
          !checkedAdd(1U, &map_owner_count)) {
        return Result::failure(makeError(LidarRegistrationErrorCode::NumericalFailure,
                                         "finalized-map direct point-to-point ICP owner index or "
                                         "correspondence count is invalid",
                                         *work));
      }
    } else if (result.owner_target_index >= targets.size() ||
               !result.candidates[result.owner_target_index].present ||
               !checkedAdd(1U, &owner_counts[result.owner_target_index]) ||
               !checkedAdd(result.candidates[result.owner_target_index].voxel_lookups,
                           &target_voxel_lookups[result.owner_target_index]) ||
               !checkedAdd(result.candidates[result.owner_target_index].points_examined,
                           &target_points_examined[result.owner_target_index])) {
      return Result::failure(
          makeError(LidarRegistrationErrorCode::NumericalFailure,
                    "direct point-to-point ICP correspondence owner or work is invalid", *work));
    }
  }

  LidarRegistrationWorkCounters updated_work = *work;
  for (std::size_t target_index = 0U; target_index < targets.size(); ++target_index) {
    output.targets[target_index].candidate_voxel_lookups = target_voxel_lookups[target_index];
    output.targets[target_index].candidate_points_examined = target_points_examined[target_index];
    output.targets[target_index].source_rows_excluded_by_ownership = excluded_counts[target_index];
  }
  if (output.finalized_map) {
    output.finalized_map->candidate_voxel_lookups = map_voxel_lookups;
    output.finalized_map->candidate_occupied_voxels = map_occupied_voxels;
    output.finalized_map->candidate_points_examined = map_points_examined;
    output.finalized_map->source_rows_excluded_by_ownership = map_excluded_count;
    if (!checkedAdd(map_voxel_lookups, &updated_work.finalized_map_candidate_voxel_lookups) ||
        !checkedAdd(map_occupied_voxels, &updated_work.finalized_map_candidate_occupied_voxels) ||
        !checkedAdd(map_points_examined, &updated_work.finalized_map_candidate_points_examined) ||
        !checkedAdd(map_stale_fallbacks, &updated_work.finalized_map_stale_fallbacks)) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "finalized-map direct point-to-point ICP accumulated work counter overflowed", *work));
    }
  }
  if (!checkedAdd(composite_voxel_lookups, &updated_work.candidate_voxel_lookups) ||
      !checkedAdd(composite_occupied_voxels, &updated_work.candidate_occupied_voxels) ||
      !checkedAdd(composite_points_examined, &updated_work.candidate_points_examined) ||
      !checkedAdd(invalid_source_points, &updated_work.invalid_source_points) ||
      !checkedAdd(invalid_target_points, &updated_work.invalid_target_points)) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "direct point-to-point ICP invalid-row work counter overflowed", *work));
  }
  if (saw_numerical_failure) {
    *work = updated_work;
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "direct point-to-point ICP association encountered a non-finite point or transform",
        *work));
  }
  for (std::size_t target_index = 0U; target_index < targets.size(); ++target_index) {
    output.targets[target_index].rows.reserve(owner_counts[target_index]);
  }
  if (output.finalized_map) {
    output.finalized_map->rows.reserve(map_owner_count);
    output.finalized_map->owners.reserve(map_owner_count);
  }
  std::vector<Eigen::Matrix3d> target_source_rotations;
  std::vector<Eigen::Vector3d> target_source_translations;
  target_source_rotations.reserve(targets.size());
  target_source_translations.reserve(targets.size());
  for (const TargetIndex& target : targets) {
    const core::Pose3d pose =
        target.record.T_target_source_seed * source_right_correction;
    if (!finitePose(pose)) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "direct point-to-point ICP frozen live-factor pose is non-finite", *work));
    }
    target_source_rotations.push_back(pose.rotationMatrix());
    target_source_translations.push_back(pose.translation());
  }
  const Eigen::Matrix3d odom_source_rotation = T_odom_source.rotationMatrix();
  if (!odom_source_rotation.allFinite()) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "direct point-to-point ICP frozen odometry-factor rotation is non-finite", *work));
  }
  std::unordered_map<const FinalizedLidarTargetOwner*, std::uint32_t> transient_owner_indices;
  transient_owner_indices.reserve(map_owner_count);
  for (std::size_t canonical_index = 0U; canonical_index < selected_source.size();
       ++canonical_index) {
    const AssociationRowResult& result = workspace->association_rows[canonical_index];
    if (result.status != AssociationRowStatus::Owned) {
      continue;
    }
    const std::size_t source_point_index = selected_source[canonical_index];
    const LidarRegistrationPoint& source_point = source.points[source_point_index];
    if (result.owner_target_index == targets.size()) {
      if (!output.finalized_map || !result.finalized_map_candidate.present ||
          !result.finalized_map_candidate.point.owner) {
        return Result::failure(
            makeError(LidarRegistrationErrorCode::NumericalFailure,
                      "finalized-map direct point-to-point ICP frozen owner is invalid", *work));
      }
      FrozenFinalizedMapPointCorrespondence row;
      row.source_point_storage_index = source_point_index;
      row.source_index = source_point.source_index;
      row.source_point = source_point.point;
      row.target_source_index = result.finalized_map_candidate.point.source_index;
      row.target_point_odom = result.finalized_map_candidate.point.point_odom;
      const auto& owner = result.finalized_map_candidate.point.owner;
      const auto existing_owner = transient_owner_indices.find(owner.get());
      if (existing_owner == transient_owner_indices.end()) {
        if (output.finalized_map->owners.size() >=
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
          return Result::failure(makeError(
              LidarRegistrationErrorCode::NumericalFailure,
              "finalized-map direct point-to-point ICP owner table exceeds its index domain",
              *work));
        }
        row.owner_index = static_cast<std::uint32_t>(output.finalized_map->owners.size());
        transient_owner_indices.emplace(owner.get(), row.owner_index);
        output.finalized_map->owners.push_back(owner);
      } else {
        row.owner_index = existing_owner->second;
      }
      const Eigen::Vector3d residual =
          odom_source_rotation * row.source_point + T_odom_source.translation() -
          row.target_point_odom;
      if (!residual.allFinite()) {
        return Result::failure(makeError(
            LidarRegistrationErrorCode::NumericalFailure,
            "finalized-map direct point-to-point ICP frozen residual is non-finite", *work));
      }
      row.association_distance_squared_m2 = std::max(0.0, residual.squaredNorm());
      output.finalized_map->rows.push_back(std::move(row));
      continue;
    }
    const TargetCandidate& selected = result.candidates[result.owner_target_index];
    const LidarRegistrationPoint& target_point =
        targets[result.owner_target_index]
            .record.cloud->points[selected.target_point_storage_index];
    FrozenPointCorrespondence row;
    row.source_point_storage_index = source_point_index;
    row.source_index = source_point.source_index;
    row.source_point = source_point.point;
    row.target_point_storage_index = selected.target_point_storage_index;
    row.target_source_index = target_point.source_index;
    row.target_point = target_point.point;
    const Eigen::Vector3d residual =
        target_source_rotations[result.owner_target_index] * row.source_point +
        target_source_translations[result.owner_target_index] - row.target_point;
    if (!residual.allFinite()) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "direct point-to-point ICP frozen live residual is non-finite", *work));
    }
    row.association_distance_squared_m2 = std::max(0.0, residual.squaredNorm());
    output.targets[result.owner_target_index].rows.push_back(std::move(row));
  }

  if (output.finalized_map) {
    const auto transient_owners = output.finalized_map->owners;
    std::sort(output.finalized_map->owners.begin(), output.finalized_map->owners.end(),
              finalizedOwnerCanonicalLess);
    output.finalized_map->owners.erase(
        std::unique(output.finalized_map->owners.begin(), output.finalized_map->owners.end(),
                    [](const auto& lhs, const auto& rhs) {
                      return lhs && rhs && finalizedOwnerCanonicalIdentity(*lhs) ==
                                               finalizedOwnerCanonicalIdentity(*rhs);
                    }),
        output.finalized_map->owners.end());
    for (FrozenFinalizedMapPointCorrespondence& row : output.finalized_map->rows) {
      if (row.owner_index >= transient_owners.size() || !transient_owners[row.owner_index]) {
        return Result::failure(makeError(
            LidarRegistrationErrorCode::NumericalFailure,
            "finalized-map direct point-to-point ICP transient owner index is invalid", *work));
      }
      const auto owner = transient_owners[row.owner_index];
      const auto canonical = std::lower_bound(output.finalized_map->owners.begin(),
                                              output.finalized_map->owners.end(), owner,
                                              finalizedOwnerCanonicalLess);
      if (canonical == output.finalized_map->owners.end() ||
          finalizedOwnerCanonicalIdentity(**canonical) != finalizedOwnerCanonicalIdentity(*owner)) {
        return Result::failure(makeError(
            LidarRegistrationErrorCode::NumericalFailure,
            "finalized-map direct point-to-point ICP owner cannot be canonicalized", *work));
      }
      row.owner_index = static_cast<std::uint32_t>(
          std::distance(output.finalized_map->owners.begin(), canonical));
    }
  }

  std::vector<double> residual_distances;
  residual_distances.reserve(selected_source.size());
  for (const WorkingTarget& target : output.targets) {
    for (const FrozenPointCorrespondence& row : target.rows) {
      residual_distances.push_back(std::sqrt(row.association_distance_squared_m2));
    }
  }
  if (output.finalized_map) {
    for (const FrozenFinalizedMapPointCorrespondence& row : output.finalized_map->rows) {
      residual_distances.push_back(std::sqrt(row.association_distance_squared_m2));
    }
  }
  if (residual_distances.empty()) {
    output.huber_delta_m = config.minimum_huber_delta_m;
  } else {
    const std::size_t middle = residual_distances.size() / 2U;
    std::nth_element(residual_distances.begin(),
                     residual_distances.begin() + static_cast<std::ptrdiff_t>(middle),
                     residual_distances.end());
    const double median = residual_distances[middle];
    for (double& distance : residual_distances) {
      distance = std::abs(distance - median);
    }
    std::nth_element(residual_distances.begin(),
                     residual_distances.begin() + static_cast<std::ptrdiff_t>(middle),
                     residual_distances.end());
    const double robust_sigma = 1.4826 * residual_distances[middle];
    output.huber_delta_m = std::clamp(config.huber_delta_multiplier * robust_sigma,
                                      config.minimum_huber_delta_m, config.maximum_huber_delta_m);
  }
  for (std::size_t target_index = 0U; target_index < targets.size(); ++target_index) {
    for (FrozenPointCorrespondence& row : output.targets[target_index].rows) {
      row.association_huber_weight =
          huberWeight(std::sqrt(row.association_distance_squared_m2), output.huber_delta_m);
    }
    std::sort(output.targets[target_index].rows.begin(), output.targets[target_index].rows.end(),
              frozenRowCanonicalLess);
    if (!checkedAdd(owner_counts[target_index], &output.correspondences)) {
      return Result::failure(makeError(LidarRegistrationErrorCode::NumericalFailure,
                                       "direct point-to-point ICP correspondence total overflowed",
                                       *work));
    }
  }
  if (output.finalized_map) {
    for (FrozenFinalizedMapPointCorrespondence& row : output.finalized_map->rows) {
      row.association_huber_weight =
          huberWeight(std::sqrt(row.association_distance_squared_m2), output.huber_delta_m);
    }
    std::sort(output.finalized_map->rows.begin(), output.finalized_map->rows.end(),
              frozenFinalizedMapRowCanonicalLess);
    if (!checkedAdd(map_owner_count, &output.correspondences)) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "finalized-map direct point-to-point ICP correspondence total overflowed", *work));
    }
  }
  if (!checkedAdd(1U, &updated_work.association_builds)) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "direct point-to-point ICP association-build counter overflowed", *work));
  }
  *work = updated_work;
  return Result::success(std::move(output));
}

[[nodiscard]] core::Result<Observability, LidarRegistrationError> computeObservability(
    const Matrix6& physical_hessian, double characteristic_length,
    const LidarRegistrationConfig& config, const LidarRegistrationWorkCounters& work) {
  using Result = core::Result<Observability, LidarRegistrationError>;
  Matrix6 normalization = Matrix6::Identity();
  normalization.diagonal().head<3>().setConstant(characteristic_length);
  const Matrix6 normalized = 0.5 * (normalization * physical_hessian * normalization +
                                    normalization * physical_hessian.transpose() * normalization);
  const Eigen::SelfAdjointEigenSolver<Matrix6> solver(normalized);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
      !solver.eigenvectors().allFinite()) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "normalized direct point-to-point ICP Hessian eigendecomposition failed", work));
  }
  const double scale = std::max(1.0, solver.eigenvalues().cwiseAbs().maxCoeff());
  if (solver.eigenvalues().minCoeff() < -1.0e-10 * scale) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "normalized direct point-to-point ICP Hessian is materially indefinite", work));
  }
  Observability output;
  output.eigenvalues = solver.eigenvalues().cwiseMax(0.0);
  output.eigenvectors = solver.eigenvectors();
  output.threshold =
      std::max(config.absolute_normalized_observable_eigenvalue,
               config.relative_normalized_observable_eigenvalue * output.eigenvalues.maxCoeff());
  for (Eigen::Index index = 0; index < 6; ++index) {
    if (output.eigenvalues(index) >= output.threshold) {
      ++output.rank;
    }
  }
  return Result::success(std::move(output));
}

[[nodiscard]] EvaluationRowResult evaluateRow(const FrozenPointCorrespondence& row,
                                              const core::Pose3d& T_target_source,
                                              const Eigen::Matrix3d& rotation,
                                              double inverse_variance, bool normal_equations) {
  EvaluationRowResult output;
  const Eigen::Vector3d transformed = rotation * row.source_point + T_target_source.translation();
  const Eigen::Vector3d residual = transformed - row.target_point;
  if (!transformed.allFinite() || !residual.allFinite() ||
      !finitePositive(row.association_huber_weight) || row.association_huber_weight > 1.0 ||
      !finitePositive(inverse_variance)) {
    return output;
  }
  output.squared_residual_m2 = std::max(0.0, residual.squaredNorm());
  output.effective_correspondence = row.association_huber_weight;
  const double row_scale = row.association_huber_weight * inverse_variance;
  output.robust_cost = 0.5 * row_scale * output.squared_residual_m2;
  if (!std::isfinite(output.squared_residual_m2) || !std::isfinite(output.robust_cost)) {
    return output;
  }
  if (normal_equations) {
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.leftCols<3>() = rotation;
    jacobian.rightCols<3>() = -rotation * skew(row.source_point);
    output.hessian.noalias() = row_scale * jacobian.transpose() * jacobian;
    output.gradient.noalias() = row_scale * jacobian.transpose() * residual;
    if (!output.hessian.allFinite() || !output.gradient.allFinite()) {
      return output;
    }
  }
  output.valid = true;
  return output;
}

[[nodiscard]] EvaluationRowResult evaluateRow(const FrozenFinalizedMapPointCorrespondence& row,
                                              const core::Pose3d& T_odom_source,
                                              const Eigen::Matrix3d& rotation,
                                              double inverse_variance, bool normal_equations) {
  EvaluationRowResult output;
  const Eigen::Vector3d transformed = rotation * row.source_point + T_odom_source.translation();
  const Eigen::Vector3d residual = transformed - row.target_point_odom;
  if (!transformed.allFinite() || !residual.allFinite() ||
      !finitePositive(row.association_huber_weight) || row.association_huber_weight > 1.0 ||
      !finitePositive(inverse_variance)) {
    return output;
  }
  output.squared_residual_m2 = std::max(0.0, residual.squaredNorm());
  output.effective_correspondence = row.association_huber_weight;
  const double row_scale = row.association_huber_weight * inverse_variance;
  output.robust_cost = 0.5 * row_scale * output.squared_residual_m2;
  if (!std::isfinite(output.squared_residual_m2) || !std::isfinite(output.robust_cost)) {
    return output;
  }
  if (normal_equations) {
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.leftCols<3>() = rotation;
    jacobian.rightCols<3>() = -rotation * skew(row.source_point);
    output.hessian.noalias() = row_scale * jacobian.transpose() * jacobian;
    output.gradient.noalias() = row_scale * jacobian.transpose() * residual;
    if (!output.hessian.allFinite() || !output.gradient.allFinite()) {
      return output;
    }
  }
  output.valid = true;
  return output;
}

[[nodiscard]] std::size_t evaluationTargetIndex(
    std::size_t row_index, std::size_t target_count,
    const std::array<std::size_t, kMaximumTargetCount + 1U>& target_offsets) noexcept {
  for (std::size_t target_index = 1U; target_index < target_count; ++target_index) {
    if (row_index < target_offsets[target_index]) {
      return target_index - 1U;
    }
  }
  return target_count - 1U;
}

[[nodiscard]] core::Result<Evaluation, LidarRegistrationError> evaluate(
    AssociationSet* associations, const core::Pose3d& source_right_correction,
    double characteristic_length, const LidarRegistrationConfig& config, bool normal_equations,
    LidarRegistrationWorkspace* workspace, LidarRegistrationWorkCounters* work) {
  using Result = core::Result<Evaluation, LidarRegistrationError>;
  Evaluation output;
  const double variance =
      config.residual_standard_deviation_m * config.residual_standard_deviation_m;
  const double inverse_variance = 1.0 / variance;
  const std::size_t channel_count =
      associations->targets.size() + (associations->finalized_map ? 1U : 0U);
  if (!finitePositive(inverse_variance) || channel_count == 0U ||
      channel_count > kMaximumTargetCount || !finitePositive(associations->huber_delta_m)) {
    return Result::failure(makeError(LidarRegistrationErrorCode::NumericalFailure,
                                     "direct point-to-point ICP evaluation domain is invalid",
                                     *work));
  }
  std::array<std::size_t, kMaximumTargetCount + 1U> target_offsets{};
  std::array<core::Pose3d, kMaximumTargetCount> T_target_source;
  std::array<Eigen::Matrix3d, kMaximumTargetCount> target_source_rotation;
  for (std::size_t target_index = 0U; target_index < associations->targets.size(); ++target_index) {
    target_offsets[target_index + 1U] = target_offsets[target_index];
    if (!checkedAdd(associations->targets[target_index].rows.size(),
                    &target_offsets[target_index + 1U])) {
      return Result::failure(makeError(LidarRegistrationErrorCode::NumericalFailure,
                                       "direct point-to-point ICP evaluation row count overflowed",
                                       *work));
    }
    T_target_source[target_index] =
        associations->targets[target_index].index->record.T_target_source_seed *
        source_right_correction;
    target_source_rotation[target_index] = T_target_source[target_index].rotationMatrix();
    if (!finitePose(T_target_source[target_index]) ||
        !target_source_rotation[target_index].allFinite()) {
      return Result::failure(makeError(LidarRegistrationErrorCode::NumericalFailure,
                                       "direct point-to-point ICP evaluation pose is non-finite",
                                       *work));
    }
  }
  const std::size_t live_rows = target_offsets[associations->targets.size()];
  std::size_t total_rows = live_rows;
  if (associations->finalized_map &&
      !checkedAdd(associations->finalized_map->rows.size(), &total_rows)) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "finalized-map direct point-to-point ICP evaluation row count overflowed", *work));
  }
  core::Pose3d T_odom_source;
  Eigen::Matrix3d odom_source_rotation = Eigen::Matrix3d::Identity();
  if (associations->finalized_map) {
    T_odom_source = associations->finalized_map->T_odom_source_seed * source_right_correction;
    odom_source_rotation = T_odom_source.rotationMatrix();
    if (!finitePose(T_odom_source) || !odom_source_rotation.allFinite()) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "finalized-map direct point-to-point ICP evaluation pose is non-finite", *work));
    }
  }
  if (total_rows == 0U ||
      total_rows > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return Result::failure(makeError(LidarRegistrationErrorCode::NumericalFailure,
                                     "direct point-to-point ICP evaluation row domain is invalid",
                                     *work));
  }
  workspace->evaluation_rows.resize(total_rows);
  const std::int64_t row_count = static_cast<std::int64_t>(total_rows);
  const int worker_count = static_cast<int>(config.parallel_worker_count);
#pragma omp parallel for schedule(static) num_threads(worker_count)
  for (std::int64_t row_index = 0; row_index < row_count; ++row_index) {
    const std::size_t canonical_index = static_cast<std::size_t>(row_index);
    if (canonical_index < live_rows) {
      const std::size_t target_index =
          evaluationTargetIndex(canonical_index, associations->targets.size(), target_offsets);
      const std::size_t target_row = canonical_index - target_offsets[target_index];
      workspace->evaluation_rows[canonical_index] = evaluateRow(
          associations->targets[target_index].rows[target_row], T_target_source[target_index],
          target_source_rotation[target_index], inverse_variance, normal_equations);
    } else {
      const std::size_t map_row = canonical_index - live_rows;
      workspace->evaluation_rows[canonical_index] =
          evaluateRow(associations->finalized_map->rows[map_row], T_odom_source,
                      odom_source_rotation, inverse_variance, normal_equations);
    }
  }

  if (std::any_of(workspace->evaluation_rows.begin(), workspace->evaluation_rows.end(),
                  [](const EvaluationRowResult& row) { return !row.valid; })) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "direct point-to-point ICP residual or row normal equation is non-finite", *work));
  }
  for (const EvaluationRowResult& row : workspace->evaluation_rows) {
    output.robust_cost += row.robust_cost;
    output.effective_correspondences += row.effective_correspondence;
    output.maximum_squared_residual_m2 =
        std::max(output.maximum_squared_residual_m2, row.squared_residual_m2);
    if (normal_equations) {
      output.hessian += row.hessian;
      output.gradient += row.gradient;
    }
  }
  if (!std::isfinite(output.robust_cost) || !std::isfinite(output.effective_correspondences) ||
      !std::isfinite(output.maximum_squared_residual_m2)) {
    return Result::failure(makeError(LidarRegistrationErrorCode::NumericalFailure,
                                     "direct point-to-point ICP robust objective is non-finite",
                                     *work));
  }
  LidarRegistrationWorkCounters updated_work = *work;
  if (!checkedAdd(1U, &updated_work.frozen_objective_evaluations)) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "direct point-to-point ICP objective-evaluation counter overflowed", *work));
  }
  if (!normal_equations) {
    *work = updated_work;
    return Result::success(std::move(output));
  }
  if (!checkedAdd(1U, &updated_work.normal_equation_evaluations)) {
    return Result::failure(makeError(LidarRegistrationErrorCode::NumericalFailure,
                                     "direct point-to-point ICP normal-equation counter overflowed",
                                     *work));
  }
  output.hessian = 0.5 * (output.hessian + output.hessian.transpose());
  if (!output.hessian.allFinite() || !output.gradient.allFinite()) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "direct point-to-point ICP physical normal equations are non-finite", *work));
  }
  *work = updated_work;
  auto observability = computeObservability(output.hessian, characteristic_length, config, *work);
  if (!observability) {
    return Result::failure(observability.error());
  }
  output.observability = observability.value();
  return Result::success(std::move(output));
}

[[nodiscard]] core::Result<core::ContentHash, LidarRegistrationError> snapshotChecksum(
    core::StateId target_state, core::FusionTime target_time, core::MeasurementId target_sweep,
    core::StateId source_state, core::FusionTime source_time, core::MeasurementId source_sweep,
    const core::ContentHash& target_cloud_checksum, const core::ContentHash& source_cloud_checksum,
    std::size_t source_point_count, const core::Pose3d& association_pose,
    std::size_t candidate_voxel_lookups, std::size_t candidate_points_examined,
    std::size_t source_rows_excluded_by_ownership, std::span<const FrozenPointCorrespondence> rows,
    double huber_delta_m, double characteristic_length_m, double geometric_information_scale,
    const LidarRegistrationConfig& config, const LidarRegistrationWorkCounters& work) {
  using Result = core::Result<core::ContentHash, LidarRegistrationError>;
  constexpr std::uint64_t kFixedBytes = 1024U;
  constexpr std::uint64_t kBytesPerRow = 512U;
  const std::uint64_t count = static_cast<std::uint64_t>(rows.size());
  if (count > (std::numeric_limits<std::uint64_t>::max() - kFixedBytes) / kBytesPerRow) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "direct point-to-point ICP snapshot is too large for canonical checksum encoding", work));
  }
  auto encoder = core::CanonicalEncoder::create(kLidarFactorSnapshotChecksumDomain,
                                                kLidarFactorSnapshotChecksumSchemaVersion,
                                                kFixedBytes + count * kBytesPerRow);
  if (!encoder) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "direct point-to-point ICP snapshot checksum encoder initialization failed", work));
  }
  const auto write = [](core::CanonicalEncodingError result) {
    return result == core::CanonicalEncodingError::None;
  };
  if (!write(encoder.value().writeU64(target_state.value())) ||
      !write(encoder.value().writeI64(target_time.nanoseconds)) ||
      !write(encoder.value().writeU64(target_sweep.value())) ||
      !write(encoder.value().writeU64(source_state.value())) ||
      !write(encoder.value().writeI64(source_time.nanoseconds)) ||
      !write(encoder.value().writeU64(source_sweep.value())) ||
      !write(encoder.value().writeHash(target_cloud_checksum)) ||
      !write(encoder.value().writeHash(source_cloud_checksum)) ||
      !write(encoder.value().writeU64(static_cast<std::uint64_t>(source_point_count))) ||
      !write(encoder.value().writePose3(association_pose)) ||
      !write(encoder.value().writeDouble(config.target_voxel_resolution_m)) ||
      !write(encoder.value().writeDouble(config.source_voxel_size_m)) ||
      !write(encoder.value().writeDouble(config.maximum_correspondence_distance_m)) ||
      !write(encoder.value().writeDouble(config.residual_standard_deviation_m)) ||
      !write(encoder.value().writeDouble(huber_delta_m)) ||
      !write(encoder.value().writeDouble(characteristic_length_m)) ||
      !write(encoder.value().writeDouble(geometric_information_scale)) ||
      !write(encoder.value().writeU64(static_cast<std::uint64_t>(candidate_voxel_lookups))) ||
      !write(encoder.value().writeU64(static_cast<std::uint64_t>(candidate_points_examined))) ||
      !write(encoder.value().writeU64(
          static_cast<std::uint64_t>(source_rows_excluded_by_ownership))) ||
      !write(encoder.value().writeU64(count))) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "direct point-to-point ICP snapshot checksum header encoding failed", work));
  }
  for (const FrozenPointCorrespondence& row : rows) {
    if (!write(
            encoder.value().writeU64(static_cast<std::uint64_t>(row.source_point_storage_index))) ||
        !write(encoder.value().writeU32(row.source_index)) ||
        !write(encoder.value().writeEigenVector(row.source_point)) ||
        !write(
            encoder.value().writeU64(static_cast<std::uint64_t>(row.target_point_storage_index))) ||
        !write(encoder.value().writeU32(row.target_source_index)) ||
        !write(encoder.value().writeEigenVector(row.target_point)) ||
        !write(encoder.value().writeDouble(row.association_distance_squared_m2)) ||
        !write(encoder.value().writeDouble(row.association_huber_weight))) {
      return Result::failure(
          makeError(LidarRegistrationErrorCode::NumericalFailure,
                    "direct point-to-point ICP snapshot checksum row encoding failed", work));
    }
  }
  auto finalized = encoder.value().finish();
  if (!finalized) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "direct point-to-point ICP snapshot checksum finalization failed", work));
  }
  return Result::success(finalized.value().digest());
}

[[nodiscard]] bool validFinalizedMapOwner(const FinalizedLidarTargetOwner& owner) {
  const FinalizedLidarAdmissionReceipt& admission = owner.admission;
  if (!owner.batch.sensor.valid() || !owner.batch.batch_id.valid() ||
      !owner.finalized_state.state.valid() || !owner.sweep.valid() ||
      !owner.finalized_state.odom_epoch.valid() || !owner.finalized_state.final_revision.valid() ||
      !owner.calibration.valid() ||
      admission.header.schema_version == 0U || !admission.header.trace.valid() ||
      !admission.header.producer.valid() || !admission.header.session.valid() ||
      !admission.header.config.valid() ||
      admission.header.direct_calibration != owner.calibration ||
      admission.odom_epoch != owner.finalized_state.odom_epoch ||
      admission.reference_time != owner.finalized_state.exact_time ||
      admission.health.sensor != owner.batch.sensor ||
      admission.health.state != core::SensorHealthState::Active ||
      !admission.health.recovery_epoch.valid() ||
      admission.health.transition_sequence == core::kInvalidSensorHealthTransitionSequence ||
      admission.health.assessed_at < admission.reference_time || !admission.map_eligible ||
      !admission.accepted_lineage.valid() ||
      !core::contentHashPresent(admission.accepted_lineage_checksum) ||
      !core::contentHashPresent(admission.accepted_batch_metadata_checksum) ||
      !core::contentHashPresent(owner.cloud_checksum) ||
      !core::contentHashPresent(owner.cloud_lineage.checksum) ||
      !core::contentHashPresent(owner.final_pose_covariance_checksum) ||
      owner.finalized_state.pose_covariance.tangent !=
          core::PoseTangentConvention::RightTranslationFirst ||
      !finitePose(owner.finalized_state.final_estimate.T_odom_imu) ||
      !owner.finalized_state.pose_covariance.matrix.allFinite() ||
      core::validateLineage(owner.cloud_lineage) != core::LineageValidationError::None) {
    return false;
  }
  const auto cloud_lineage_checksum = recomputeAcceptedLidarLineageChecksum(owner.cloud_lineage);
  if (!cloud_lineage_checksum || cloud_lineage_checksum.value() != owner.cloud_lineage.checksum) {
    return false;
  }
  std::set<core::MeasurementId> imu_support;
  for (const core::MeasurementId imu : owner.imu_support) {
    if (!imu.valid() || !imu_support.insert(imu).second) {
      return false;
    }
    const std::size_t uses = static_cast<std::size_t>(std::count_if(
        owner.cloud_lineage.usage.begin(), owner.cloud_lineage.usage.end(),
        [&](const core::ObservationUsage& usage) {
          const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
          return measurement != nullptr && *measurement == imu &&
                 usage.role == core::ObservationRole::ConditioningOnly;
        }));
    if (uses != 1U) {
      return false;
    }
  }
  const Matrix6& covariance = owner.finalized_state.pose_covariance.matrix;
  const double scale = std::max(1.0, covariance.cwiseAbs().maxCoeff());
  if ((covariance - covariance.transpose()).cwiseAbs().maxCoeff() > 1.0e-10 * scale) {
    return false;
  }
  const Eigen::SelfAdjointEigenSolver<Matrix6> solver(covariance);
  return solver.info() == Eigen::Success && solver.eigenvalues().allFinite() &&
         solver.eigenvalues().minCoeff() >= -1.0e-10 * scale;
}

[[nodiscard]] core::Result<double, LidarRegistrationError>
computeFinalizedMapOwnerPoseCovarianceInflation(
    std::span<const FrozenFinalizedMapPointCorrespondence> rows,
    std::span<const std::shared_ptr<const FinalizedLidarTargetOwner>> owners,
    double residual_standard_deviation_m, const LidarRegistrationWorkCounters& work) {
  using Result = core::Result<double, LidarRegistrationError>;
  if (rows.empty() || owners.empty() || !finitePositive(residual_standard_deviation_m) ||
      !std::is_sorted(owners.begin(), owners.end(), finalizedOwnerCanonicalLess) ||
      std::adjacent_find(owners.begin(), owners.end(), [](const auto& lhs, const auto& rhs) {
        return lhs && rhs && finalizedOwnerCanonicalIdentity(*lhs) ==
                                 finalizedOwnerCanonicalIdentity(*rhs);
      }) != owners.end() ||
      std::any_of(owners.begin(), owners.end(),
                  [](const auto& owner) { return !owner || !validFinalizedMapOwner(*owner); })) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidTarget,
        "finalized-map direct point-to-point ICP covariance inflation domain is invalid", work));
  }
  double maximum_point_variance = 0.0;
  for (const FrozenFinalizedMapPointCorrespondence& row : rows) {
    if (row.owner_index >= owners.size() || !row.target_point_odom.allFinite()) {
      return Result::failure(
          makeError(LidarRegistrationErrorCode::InvalidTarget,
                    "finalized-map direct point-to-point ICP covariance owner is invalid", work));
    }
    const FinalizedLidarTargetOwner& owner = *owners[row.owner_index];
    const core::Pose3d& T_odom_owner = owner.finalized_state.final_estimate.T_odom_imu;
    const Eigen::Vector3d point_owner = T_odom_owner.inverse() * row.target_point_odom;
    const Eigen::Matrix3d rotation = T_odom_owner.rotationMatrix();
    if (!point_owner.allFinite() || !rotation.allFinite()) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "finalized-map direct point-to-point ICP owner-frame point is non-finite", work));
    }
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.leftCols<3>() = rotation;
    jacobian.rightCols<3>() = -rotation * skew(point_owner);
    Eigen::Matrix3d point_covariance =
        jacobian * owner.finalized_state.pose_covariance.matrix * jacobian.transpose();
    point_covariance = 0.5 * (point_covariance + point_covariance.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(point_covariance);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "finalized-map direct point-to-point ICP point-covariance decomposition failed", work));
    }
    const double covariance_scale = std::max(1.0, solver.eigenvalues().cwiseAbs().maxCoeff());
    if (solver.eigenvalues().minCoeff() < -1.0e-10 * covariance_scale) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::InvalidTarget,
          "finalized-map direct point-to-point ICP point covariance is indefinite", work));
    }
    maximum_point_variance =
        std::max(maximum_point_variance, std::max(0.0, solver.eigenvalues().maxCoeff()));
  }
  const double variance = residual_standard_deviation_m * residual_standard_deviation_m;
  const double inflation = 1.0 + maximum_point_variance / variance;
  if (!std::isfinite(inflation) || inflation < 1.0) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "finalized-map direct point-to-point ICP owner covariance inflation is non-finite", work));
  }
  return Result::success(inflation);
}

[[nodiscard]] bool writeFinalizedMapOwner(core::CanonicalEncoder& encoder,
                                          const FinalizedLidarTargetOwner& owner) {
  const auto write = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!write(encoder.writeU8(static_cast<std::uint8_t>(owner.batch.sensor.modality))) ||
      !write(encoder.writeU64(owner.batch.sensor.instance)) ||
      !write(encoder.writeU64(owner.batch.batch_id.value())) ||
      !write(encoder.writeU32(owner.admission.header.schema_version)) ||
      !write(encoder.writeU64(owner.admission.header.trace.value())) ||
      !write(encoder.writeU64(owner.admission.header.producer.value())) ||
      !write(encoder.writeU64(owner.admission.header.session.value())) ||
      !write(encoder.writeI64(owner.admission.header.created_at.nanoseconds)) ||
      !write(encoder.writeU64(owner.admission.header.config.value())) ||
      !write(encoder.writeU64(owner.admission.odom_epoch.value())) ||
      !write(encoder.writeI64(owner.admission.reference_time.nanoseconds)) ||
      !write(encoder.writeHash(owner.admission.accepted_batch_metadata_checksum)) ||
      !write(encoder.writeU64(owner.admission.accepted_lineage.value())) ||
      !write(encoder.writeHash(owner.admission.accepted_lineage_checksum)) ||
      !write(encoder.writeU64(owner.admission_revision.value())) ||
      !write(encoder.writeU8(static_cast<std::uint8_t>(owner.admission_kind))) ||
      !write(encoder.writeU64(owner.finalized_state.state.value())) ||
      !write(encoder.writeI64(owner.finalized_state.exact_time.nanoseconds)) ||
      !write(encoder.writeU64(owner.finalized_state.odom_epoch.value())) ||
      !write(encoder.writeU64(owner.finalized_state.final_revision.value())) ||
      !write(encoder.writePose3(owner.finalized_state.final_estimate.T_odom_imu)) ||
      !write(encoder.writeEigenMatrix(owner.finalized_state.pose_covariance.matrix)) ||
      !write(encoder.writeHash(owner.final_pose_covariance_checksum)) ||
      !write(encoder.writeU64(owner.sweep.value())) ||
      !write(encoder.writeHash(owner.cloud_checksum)) ||
      !write(encoder.writeU64(owner.calibration.value())) ||
      !write(encoder.writeU64(owner.cloud_lineage.id.value())) ||
      !write(encoder.writeHash(owner.cloud_lineage.checksum)) ||
      !write(encoder.writeU8(static_cast<std::uint8_t>(owner.admission.health.state))) ||
      !write(encoder.writeU64(owner.admission.health.recovery_epoch.value())) ||
      !write(encoder.writeU64(owner.admission.health.transition_sequence)) ||
      !write(encoder.writeI64(owner.admission.health.assessed_at.nanoseconds)) ||
      !write(encoder.writeBool(owner.admission.map_eligible)) ||
      !write(encoder.writeU64(static_cast<std::uint64_t>(owner.imu_support.size())))) {
    return false;
  }
  for (const core::MeasurementId imu : owner.imu_support) {
    if (!write(encoder.writeU64(imu.value()))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] core::Result<core::ContentHash, LidarRegistrationError> finalizedMapSnapshotChecksum(
    core::StateId source_state, core::FusionTime source_time, core::MeasurementId source_sweep,
    const core::ContentHash& source_cloud_checksum, std::size_t source_point_count,
    core::OdomEpoch map_odom_epoch, core::SensorInstanceId map_sensor, std::uint64_t map_version,
    const core::ContentHash& map_checksum, const core::Pose3d& association_pose,
    std::span<const std::shared_ptr<const FinalizedLidarTargetOwner>> owners,
    std::span<const FrozenFinalizedMapPointCorrespondence> rows,
    std::size_t candidate_voxel_lookups, std::size_t candidate_occupied_voxels,
    std::size_t candidate_points_examined, std::size_t source_rows_excluded_by_ownership,
    double huber_delta_m, double characteristic_length_m, double geometric_information_scale,
    double owner_pose_covariance_inflation, const LidarRegistrationConfig& config,
    const LidarRegistrationWorkCounters& work) {
  using Result = core::Result<core::ContentHash, LidarRegistrationError>;
  if (owners.empty() || !std::is_sorted(owners.begin(), owners.end(), finalizedOwnerCanonicalLess) ||
      std::adjacent_find(owners.begin(), owners.end(), [](const auto& lhs, const auto& rhs) {
        return lhs && rhs && finalizedOwnerCanonicalIdentity(*lhs) ==
                                 finalizedOwnerCanonicalIdentity(*rhs);
      }) != owners.end() || std::any_of(owners.begin(), owners.end(),
                  [](const auto& owner) { return !owner || !validFinalizedMapOwner(*owner); })) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::InvalidTarget,
                  "finalized-map direct point-to-point ICP snapshot owner table is invalid", work));
  }
  constexpr std::uint64_t kFixedBytes = 1536U;
  constexpr std::uint64_t kBytesPerOwner = 2048U;
  constexpr std::uint64_t kBytesPerRow = 512U;
  const std::uint64_t owner_count = static_cast<std::uint64_t>(owners.size());
  const std::uint64_t row_count = static_cast<std::uint64_t>(rows.size());
  if (owner_count > (std::numeric_limits<std::uint64_t>::max() - kFixedBytes) / kBytesPerOwner ||
      row_count >
          (std::numeric_limits<std::uint64_t>::max() - kFixedBytes - owner_count * kBytesPerOwner) /
              kBytesPerRow) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "finalized-map direct point-to-point ICP snapshot checksum is too large", work));
  }
  auto encoder = core::CanonicalEncoder::create(
      kFinalizedMapLidarFactorSnapshotChecksumDomain,
      kFinalizedMapLidarFactorSnapshotChecksumSchemaVersion,
      kFixedBytes + owner_count * kBytesPerOwner + row_count * kBytesPerRow);
  if (!encoder) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "finalized-map direct point-to-point ICP checksum encoder initialization failed", work));
  }
  const auto write = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!write(encoder.value().writeU64(source_state.value())) ||
      !write(encoder.value().writeI64(source_time.nanoseconds)) ||
      !write(encoder.value().writeU64(source_sweep.value())) ||
      !write(encoder.value().writeHash(source_cloud_checksum)) ||
      !write(encoder.value().writeU64(static_cast<std::uint64_t>(source_point_count))) ||
      !write(encoder.value().writeU64(map_odom_epoch.value())) ||
      !write(encoder.value().writeU8(static_cast<std::uint8_t>(map_sensor.modality))) ||
      !write(encoder.value().writeU64(map_sensor.instance)) ||
      !write(encoder.value().writeU64(map_version)) ||
      !write(encoder.value().writeHash(map_checksum)) ||
      !write(encoder.value().writePose3(association_pose)) ||
      !write(encoder.value().writeDouble(config.target_voxel_resolution_m)) ||
      !write(encoder.value().writeDouble(config.source_voxel_size_m)) ||
      !write(encoder.value().writeDouble(config.maximum_correspondence_distance_m)) ||
      !write(encoder.value().writeDouble(config.residual_standard_deviation_m)) ||
      !write(encoder.value().writeDouble(huber_delta_m)) ||
      !write(encoder.value().writeDouble(characteristic_length_m)) ||
      !write(encoder.value().writeDouble(geometric_information_scale)) ||
      !write(encoder.value().writeDouble(owner_pose_covariance_inflation)) ||
      !write(encoder.value().writeU64(static_cast<std::uint64_t>(candidate_voxel_lookups))) ||
      !write(encoder.value().writeU64(static_cast<std::uint64_t>(candidate_occupied_voxels))) ||
      !write(encoder.value().writeU64(static_cast<std::uint64_t>(candidate_points_examined))) ||
      !write(encoder.value().writeU64(
          static_cast<std::uint64_t>(source_rows_excluded_by_ownership))) ||
      !write(encoder.value().writeU64(owner_count))) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "finalized-map direct point-to-point ICP snapshot header encoding failed", work));
  }
  for (const auto& owner : owners) {
    if (!writeFinalizedMapOwner(encoder.value(), *owner)) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "finalized-map direct point-to-point ICP snapshot owner encoding failed", work));
    }
  }
  if (!write(encoder.value().writeU64(row_count))) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "finalized-map direct point-to-point ICP snapshot row count encoding failed", work));
  }
  for (const FrozenFinalizedMapPointCorrespondence& row : rows) {
    if (row.owner_index >= owners.size()) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::InvalidTarget,
          "finalized-map direct point-to-point ICP row owner is absent from its canonical table",
          work));
    }
    if (!write(
            encoder.value().writeU64(static_cast<std::uint64_t>(row.source_point_storage_index))) ||
        !write(encoder.value().writeU32(row.source_index)) ||
        !write(encoder.value().writeEigenVector(row.source_point)) ||
        !write(encoder.value().writeU32(row.target_source_index)) ||
        !write(encoder.value().writeEigenVector(row.target_point_odom)) ||
        !write(encoder.value().writeU32(row.owner_index)) ||
        !write(encoder.value().writeDouble(row.association_distance_squared_m2)) ||
        !write(encoder.value().writeDouble(row.association_huber_weight))) {
      return Result::failure(
          makeError(LidarRegistrationErrorCode::NumericalFailure,
                    "finalized-map direct point-to-point ICP snapshot row encoding failed", work));
    }
  }
  auto finalized = encoder.value().finish();
  if (!finalized) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "finalized-map direct point-to-point ICP snapshot checksum finalization failed", work));
  }
  return Result::success(finalized.value().digest());
}

}  // namespace

namespace detail {

class LidarFactorSnapshotBuilder {
public:
  [[nodiscard]] static core::Result<std::shared_ptr<const LidarFactorSnapshot>,
                                    LidarRegistrationError>
  build(core::StateId source_state, const LidarRegistrationCloud& source,
        const WorkingTarget& working, const core::Pose3d& source_right_correction,
        double huber_delta_m, double characteristic_length_m, double geometric_information_scale,
        const LidarRegistrationConfig& config, const LidarRegistrationWorkCounters& work) {
    using Result = core::Result<std::shared_ptr<const LidarFactorSnapshot>, LidarRegistrationError>;
    const LidarRegistrationTarget& target = working.index->record;
    const core::Pose3d association_pose = target.T_target_source_seed * source_right_correction;
    auto checksum = snapshotChecksum(
        target.state, target.time, target.cloud->source_sweep, source_state, source.reference_time,
        source.source_sweep, target.cloud->checksum, source.checksum, work.source_points_selected,
        association_pose, working.candidate_voxel_lookups, working.candidate_points_examined,
        working.source_rows_excluded_by_ownership, working.rows, huber_delta_m,
        characteristic_length_m, geometric_information_scale, config, work);
    if (!checksum) {
      return Result::failure(checksum.error());
    }
    auto snapshot = std::shared_ptr<LidarFactorSnapshot>(new LidarFactorSnapshot());
    snapshot->target_state_ = target.state;
    snapshot->target_time_ = target.time;
    snapshot->target_sweep_ = target.cloud->source_sweep;
    snapshot->source_state_ = source_state;
    snapshot->source_time_ = source.reference_time;
    snapshot->source_sweep_ = source.source_sweep;
    snapshot->target_cloud_checksum_ = target.cloud->checksum;
    snapshot->source_cloud_checksum_ = source.checksum;
    snapshot->source_point_count_ = work.source_points_selected;
    snapshot->T_target_source_association_ = association_pose;
    snapshot->rows_ = working.rows;
    snapshot->candidate_voxel_lookups_ = working.candidate_voxel_lookups;
    snapshot->candidate_points_examined_ = working.candidate_points_examined;
    snapshot->source_rows_excluded_by_ownership_ = working.source_rows_excluded_by_ownership;
    snapshot->residual_standard_deviation_m_ = config.residual_standard_deviation_m;
    snapshot->huber_delta_m_ = huber_delta_m;
    snapshot->characteristic_length_m_ = characteristic_length_m;
    snapshot->geometric_information_scale_ = geometric_information_scale;
    snapshot->checksum_ = checksum.value();
    return Result::success(std::move(snapshot));
  }
};

class FinalizedMapLidarFactorSnapshotBuilder {
public:
  [[nodiscard]] static core::Result<std::shared_ptr<const FinalizedMapLidarFactorSnapshot>,
                                    LidarRegistrationError>
  build(core::StateId source_state, const LidarRegistrationCloud& source,
        const WorkingFinalizedMap& working, const core::Pose3d& source_right_correction,
        double huber_delta_m, double characteristic_length_m, double geometric_information_scale,
        const LidarRegistrationConfig& config, const LidarRegistrationWorkCounters& work) {
    using Result = core::Result<std::shared_ptr<const FinalizedMapLidarFactorSnapshot>,
                                LidarRegistrationError>;
    if (working.rows.empty() || !working.odom_epoch.valid() || !working.sensor.valid() ||
        working.sensor.modality != core::SensorModality::Lidar || working.version == 0U ||
        !core::contentHashPresent(working.checksum)) {
      return Result::failure(
          makeError(LidarRegistrationErrorCode::InvalidTarget,
                    "finalized-map direct point-to-point ICP snapshot view is invalid", work));
    }
    const core::Pose3d association_pose = source.T_odom_imu_seed * source_right_correction;
    if (!finitePose(association_pose)) {
      return Result::failure(
          makeError(LidarRegistrationErrorCode::NumericalFailure,
                    "finalized-map direct point-to-point ICP snapshot pose is non-finite", work));
    }
    auto covariance_inflation = computeFinalizedMapOwnerPoseCovarianceInflation(
        working.rows, working.owners, config.residual_standard_deviation_m, work);
    if (!covariance_inflation) {
      return Result::failure(covariance_inflation.error());
    }
    auto checksum = finalizedMapSnapshotChecksum(
        source_state, source.reference_time, source.source_sweep, source.checksum,
        work.source_points_selected, working.odom_epoch, working.sensor, working.version,
        working.checksum, association_pose, working.owners,
        working.rows, working.candidate_voxel_lookups, working.candidate_occupied_voxels,
        working.candidate_points_examined, working.source_rows_excluded_by_ownership, huber_delta_m,
        characteristic_length_m, geometric_information_scale, covariance_inflation.value(), config,
        work);
    if (!checksum) {
      return Result::failure(checksum.error());
    }
    auto snapshot =
        std::shared_ptr<FinalizedMapLidarFactorSnapshot>(new FinalizedMapLidarFactorSnapshot());
    snapshot->source_state_ = source_state;
    snapshot->source_time_ = source.reference_time;
    snapshot->source_sweep_ = source.source_sweep;
    snapshot->source_cloud_checksum_ = source.checksum;
    snapshot->source_point_count_ = work.source_points_selected;
    snapshot->map_odom_epoch_ = working.odom_epoch;
    snapshot->map_sensor_ = working.sensor;
    snapshot->map_version_ = working.version;
    snapshot->map_checksum_ = working.checksum;
    snapshot->T_odom_source_association_ = association_pose;
    snapshot->owners_ = working.owners;
    snapshot->rows_ = working.rows;
    snapshot->candidate_voxel_lookups_ = working.candidate_voxel_lookups;
    snapshot->candidate_occupied_voxels_ = working.candidate_occupied_voxels;
    snapshot->candidate_points_examined_ = working.candidate_points_examined;
    snapshot->source_rows_excluded_by_ownership_ = working.source_rows_excluded_by_ownership;
    snapshot->residual_standard_deviation_m_ = config.residual_standard_deviation_m;
    snapshot->huber_delta_m_ = huber_delta_m;
    snapshot->characteristic_length_m_ = characteristic_length_m;
    snapshot->geometric_information_scale_ = geometric_information_scale;
    snapshot->owner_pose_covariance_inflation_ = covariance_inflation.value();
    snapshot->checksum_ = checksum.value();
    return Result::success(std::move(snapshot));
  }
};

}  // namespace detail

bool isValidLidarRegistrationConfig(const LidarRegistrationConfig& config) noexcept {
  return validConfig(config);
}

core::Result<LidarRegistrationSourceSelection, LidarRegistrationError>
selectLidarRegistrationSourcePoints(const std::shared_ptr<const LidarRegistrationCloud>& source,
                                    const LidarRegistrationConfig& config) {
  using Result = core::Result<LidarRegistrationSourceSelection, LidarRegistrationError>;
  LidarRegistrationWorkCounters work;
  if (!validConfig(config)) {
    return Result::failure(makeError(LidarRegistrationErrorCode::InvalidConfig,
                                     "direct point ICP source-selection config is invalid", work));
  }
  auto selected = selectSourcePoints(source, config, &work);
  if (!selected) {
    return Result::failure(selected.error());
  }
  LidarRegistrationSourceSelection output;
  output.point_storage_indices = std::move(selected).value();
  output.points_considered = work.source_points_considered;
  output.points_omitted = work.source_points_omitted_by_capacity;
  return Result::success(std::move(output));
}

core::Result<LidarRegistrationSnapshotAggregate, LidarRegistrationError>
summarizeLidarFactorSnapshots(
    std::span<const std::shared_ptr<const LidarFactorSnapshot>> snapshots) {
  using Result = core::Result<LidarRegistrationSnapshotAggregate, LidarRegistrationError>;
  LidarRegistrationWorkCounters work;
  if (snapshots.empty()) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidTarget,
        "direct point-to-point ICP snapshot aggregate requires at least one target", work));
  }

  LidarRegistrationSnapshotAggregate output;
  output.target_count = snapshots.size();
  output.live_target_count = snapshots.size();
  std::optional<core::StateId> source_state;
  std::optional<core::FusionTime> source_time;
  std::optional<core::MeasurementId> source_sweep;
  std::optional<core::ContentHash> source_cloud_checksum;
  std::optional<std::size_t> source_point_count;
  std::optional<double> huber_delta_m;
  std::set<core::StateId> target_states;
  std::set<core::FusionTime> target_times;
  std::set<core::MeasurementId> target_sweeps;
  std::set<core::ContentHash> target_cloud_checksums;
  std::set<std::uint32_t> owned_source_indices;
  // Storage indices are process-local lookup handles, not observation
  // identity. Still require a one-to-one mapping inside one aggregate so a
  // malformed snapshot cannot alias two stable source rows to one payload row.
  std::set<std::size_t> owned_source_storage_rows;

  for (const auto& snapshot : snapshots) {
    if (!snapshot || !core::contentHashPresent(snapshot->checksum()) ||
        !core::contentHashPresent(snapshot->targetCloudChecksum()) ||
        !core::contentHashPresent(snapshot->sourceCloudChecksum()) ||
        !snapshot->targetState().valid() || !snapshot->targetSweep().valid() ||
        !snapshot->sourceState().valid() || !snapshot->sourceSweep().valid() ||
        !(snapshot->targetState() < snapshot->sourceState()) ||
        !(snapshot->targetTime() < snapshot->sourceTime()) ||
        !finitePose(snapshot->associationPose()) || snapshot->sourcePointCount() == 0U ||
        snapshot->rows().empty() || !finitePositive(snapshot->residualStandardDeviationM()) ||
        !finitePositive(snapshot->huberDeltaM()) ||
        !std::is_sorted(snapshot->rows().begin(), snapshot->rows().end(), frozenRowCanonicalLess) ||
        !target_states.insert(snapshot->targetState()).second ||
        !target_times.insert(snapshot->targetTime()).second ||
        !target_sweeps.insert(snapshot->targetSweep()).second ||
        !target_cloud_checksums.insert(snapshot->targetCloudChecksum()).second) {
      return Result::failure(makeError(LidarRegistrationErrorCode::InvalidTarget,
                                       "direct point-to-point ICP snapshot aggregate contains an "
                                       "invalid or duplicate target snapshot",
                                       work));
    }
    if ((source_state && *source_state != snapshot->sourceState()) ||
        (source_time && *source_time != snapshot->sourceTime()) ||
        (source_sweep && *source_sweep != snapshot->sourceSweep()) ||
        (source_cloud_checksum && *source_cloud_checksum != snapshot->sourceCloudChecksum()) ||
        (source_point_count && *source_point_count != snapshot->sourcePointCount()) ||
        (huber_delta_m && *huber_delta_m != snapshot->huberDeltaM())) {
      return Result::failure(makeError(LidarRegistrationErrorCode::InvalidTarget,
                                       "direct point-to-point ICP snapshot aggregate does not "
                                       "share one bit-identical source population",
                                       work));
    }
    source_state = snapshot->sourceState();
    source_time = snapshot->sourceTime();
    source_sweep = snapshot->sourceSweep();
    source_cloud_checksum = snapshot->sourceCloudChecksum();
    source_point_count = snapshot->sourcePointCount();
    huber_delta_m = snapshot->huberDeltaM();

    const Eigen::Matrix3d rotation = snapshot->associationPose().rotationMatrix();
    const Eigen::Vector3d translation = snapshot->associationPose().translation();
    const double variance =
        snapshot->residualStandardDeviationM() * snapshot->residualStandardDeviationM();

    for (const FrozenPointCorrespondence& row : snapshot->rows()) {
      // Stable source identity, rather than vector storage order, owns one
      // residual across every pose-aware target. Storage indices remain sparse
      // handles into the full cloud and may exceed sourcePointCount().
      if (!owned_source_indices.insert(row.source_index).second ||
          !owned_source_storage_rows.insert(row.source_point_storage_index).second ||
          !row.source_point.allFinite() || !row.target_point.allFinite() ||
          !std::isfinite(row.association_distance_squared_m2) ||
          row.association_distance_squared_m2 < 0.0 ||
          !finitePositive(row.association_huber_weight) || row.association_huber_weight > 1.0) {
        return Result::failure(makeError(LidarRegistrationErrorCode::InvalidTarget,
                                         "direct point-to-point ICP snapshot aggregate has invalid "
                                         "or overlapping frozen source-row ownership",
                                         work));
      }
      const Eigen::Vector3d residual = rotation * row.source_point + translation - row.target_point;
      const double squared_residual = residual.squaredNorm();
      const double scalar_tolerance =
          1.0e-9 * std::max({1.0, squared_residual, row.association_distance_squared_m2});
      const double expected_weight =
          huberWeight(std::sqrt(squared_residual), snapshot->huberDeltaM());
      if (!residual.allFinite() || !std::isfinite(squared_residual) ||
          std::abs(squared_residual - row.association_distance_squared_m2) > scalar_tolerance ||
          std::abs(expected_weight - row.association_huber_weight) > 1.0e-12) {
        return Result::failure(makeError(
            LidarRegistrationErrorCode::InvalidTarget,
            "direct point-to-point ICP snapshot aggregate contains a stale frozen robust weight",
            work));
      }
      output.effective_correspondences += row.association_huber_weight;
      output.maximum_squared_residual_m2 =
          std::max(output.maximum_squared_residual_m2, squared_residual);
      output.final_robust_cost += 0.5 * row.association_huber_weight * squared_residual / variance;
      ++output.correspondences;
    }
  }

  output.source_point_count = source_point_count.value_or(0U);
  output.huber_delta_m = huber_delta_m.value_or(0.0);
  output.overlap_fraction = output.source_point_count == 0U
                                ? 0.0
                                : static_cast<double>(output.correspondences) /
                                      static_cast<double>(output.source_point_count);
  if (output.correspondences > output.source_point_count ||
      !std::isfinite(output.overlap_fraction) || output.overlap_fraction < 0.0 ||
      output.overlap_fraction > 1.0 || !std::isfinite(output.effective_correspondences) ||
      !std::isfinite(output.maximum_squared_residual_m2) || !finitePositive(output.huber_delta_m) ||
      !std::isfinite(output.final_robust_cost)) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidTarget,
        "direct point-to-point ICP snapshot aggregate is non-finite or over-owned", work));
  }
  return Result::success(std::move(output));
}

core::Result<LidarRegistrationSnapshotAggregate, LidarRegistrationError>
summarizeLidarFactorSnapshots(
    std::span<const std::shared_ptr<const LidarFactorSnapshot>> snapshots,
    const std::shared_ptr<const FinalizedMapLidarFactorSnapshot>& finalized_map_snapshot) {
  using Result = core::Result<LidarRegistrationSnapshotAggregate, LidarRegistrationError>;
  if (!finalized_map_snapshot) {
    return summarizeLidarFactorSnapshots(snapshots);
  }
  LidarRegistrationWorkCounters work;
  LidarRegistrationSnapshotAggregate output;
  if (!snapshots.empty()) {
    auto live = summarizeLidarFactorSnapshots(snapshots);
    if (!live) {
      return Result::failure(live.error());
    }
    output = live.value();
  }
  const auto& map = *finalized_map_snapshot;
  if (!core::contentHashPresent(map.checksum()) ||
      !core::contentHashPresent(map.sourceCloudChecksum()) ||
      !core::contentHashPresent(map.mapChecksum()) || !map.sourceState().valid() ||
      !map.sourceSweep().valid() || !map.mapOdomEpoch().valid() || !map.mapSensor().valid() ||
      map.mapSensor().modality != core::SensorModality::Lidar || map.mapVersion() == 0U ||
      !finitePose(map.associationPose()) || map.sourcePointCount() == 0U || map.rows().empty() ||
      map.rows().size() > map.sourcePointCount() || map.owners().empty() ||
      !finitePositive(map.residualStandardDeviationM()) || !finitePositive(map.huberDeltaM()) ||
      !finitePositive(map.characteristicLengthM()) ||
      !finitePositive(map.geometricInformationScale()) || map.geometricInformationScale() > 1.0 ||
      !std::is_sorted(map.rows().begin(), map.rows().end(), frozenFinalizedMapRowCanonicalLess)) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidTarget,
        "mixed direct point-to-point ICP aggregate contains an invalid finalized-map snapshot",
        work));
  }
  if (!snapshots.empty()) {
    const auto& first = *snapshots.front();
    if (first.sourceState() != map.sourceState() || first.sourceTime() != map.sourceTime() ||
        first.sourceSweep() != map.sourceSweep() ||
        first.sourceCloudChecksum() != map.sourceCloudChecksum() ||
        first.sourcePointCount() != map.sourcePointCount() ||
        first.huberDeltaM() != map.huberDeltaM() ||
        first.residualStandardDeviationM() != map.residualStandardDeviationM() ||
        first.characteristicLengthM() != map.characteristicLengthM() ||
        first.geometricInformationScale() != map.geometricInformationScale()) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::InvalidTarget,
          "mixed direct point-to-point ICP snapshots do not share one sealed source objective",
          work));
    }
  } else {
    output.source_point_count = map.sourcePointCount();
    output.huber_delta_m = map.huberDeltaM();
  }

  std::set<std::uint32_t> owned_source_indices;
  std::set<std::size_t> owned_source_storage_rows;
  for (const auto& live : snapshots) {
    for (const FrozenPointCorrespondence& row : live->rows()) {
      owned_source_indices.insert(row.source_index);
      owned_source_storage_rows.insert(row.source_point_storage_index);
    }
  }
  const Eigen::Matrix3d rotation = map.associationPose().rotationMatrix();
  const Eigen::Vector3d translation = map.associationPose().translation();
  const double variance = map.residualStandardDeviationM() * map.residualStandardDeviationM();
  for (const FrozenFinalizedMapPointCorrespondence& row : map.rows()) {
    if (row.owner_index >= map.owners().size() ||
        !owned_source_indices.insert(row.source_index).second ||
        !owned_source_storage_rows.insert(row.source_point_storage_index).second ||
        !row.source_point.allFinite() || !row.target_point_odom.allFinite() ||
        !std::isfinite(row.association_distance_squared_m2) ||
        row.association_distance_squared_m2 < 0.0 ||
        !finitePositive(row.association_huber_weight) || row.association_huber_weight > 1.0) {
      return Result::failure(makeError(LidarRegistrationErrorCode::InvalidTarget,
                                       "mixed direct point-to-point ICP aggregate has invalid or "
                                       "overlapping finalized-map ownership",
                                       work));
    }
    const Eigen::Vector3d residual =
        rotation * row.source_point + translation - row.target_point_odom;
    const double squared_residual = residual.squaredNorm();
    const double tolerance =
        1.0e-9 * std::max({1.0, squared_residual, row.association_distance_squared_m2});
    const double expected_weight = huberWeight(std::sqrt(squared_residual), map.huberDeltaM());
    if (!residual.allFinite() || !std::isfinite(squared_residual) ||
        std::abs(squared_residual - row.association_distance_squared_m2) > tolerance ||
        std::abs(expected_weight - row.association_huber_weight) > 1.0e-12) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::InvalidTarget,
          "mixed direct point-to-point ICP finalized-map metric or robust weight is stale", work));
    }
    output.effective_correspondences += row.association_huber_weight;
    output.maximum_squared_residual_m2 =
        std::max(output.maximum_squared_residual_m2, squared_residual);
    output.final_robust_cost += 0.5 * row.association_huber_weight * squared_residual / variance;
    ++output.correspondences;
  }
  auto inflation = computeFinalizedMapOwnerPoseCovarianceInflation(
      map.rows(), map.owners(), map.residualStandardDeviationM(), work);
  if (!inflation || inflation.value() != map.ownerPoseCovarianceInflation()) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidTarget,
        "mixed direct point-to-point ICP finalized-map provenance summary is stale", work));
  }
  output.live_target_count = snapshots.size();
  output.finalized_map_target_count = 1U;
  output.target_count = snapshots.size() + 1U;
  output.overlap_fraction =
      static_cast<double>(output.correspondences) / static_cast<double>(output.source_point_count);
  if (output.correspondences > output.source_point_count ||
      !std::isfinite(output.overlap_fraction) || output.overlap_fraction < 0.0 ||
      output.overlap_fraction > 1.0 || !std::isfinite(output.final_robust_cost) ||
      !std::isfinite(output.effective_correspondences)) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::InvalidTarget,
                  "mixed direct point-to-point ICP aggregate is non-finite or over-owned", work));
  }
  return Result::success(std::move(output));
}

core::Result<core::RankAwareInformation, LidarRegistrationError> lidarFactorInformation(
    const std::shared_ptr<const LidarFactorSnapshot>& snapshot,
    const LidarRegistrationConfig& config, double information_scale) {
  using Result = core::Result<core::RankAwareInformation, LidarRegistrationError>;
  LidarRegistrationWorkCounters work;
  if (!validConfig(config) || !std::isfinite(information_scale) || information_scale <= 0.0 ||
      information_scale > 1.0) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::InvalidConfig,
                  "direct point-to-point ICP admission configuration or scale is invalid", work));
  }
  if (!snapshot || !core::contentHashPresent(snapshot->checksum()) ||
      !core::contentHashPresent(snapshot->targetCloudChecksum()) ||
      !core::contentHashPresent(snapshot->sourceCloudChecksum()) ||
      !snapshot->targetState().valid() || !snapshot->sourceState().valid() ||
      !(snapshot->targetState() < snapshot->sourceState()) ||
      !(snapshot->targetTime() < snapshot->sourceTime()) ||
      !finitePose(snapshot->associationPose()) ||
      snapshot->sourcePointCount() < snapshot->rows().size() ||
      snapshot->rows().size() < config.minimum_correspondences ||
      snapshot->residualStandardDeviationM() != config.residual_standard_deviation_m ||
      snapshot->huberDeltaM() < config.minimum_huber_delta_m ||
      snapshot->huberDeltaM() > config.maximum_huber_delta_m) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidTarget,
        "direct point-to-point ICP admission snapshot identity, support, or profile is invalid",
        work));
  }

  Matrix6 raw_hessian = Matrix6::Zero();
  const Eigen::Matrix3d association_rotation = snapshot->associationPose().rotationMatrix();
  const Eigen::Vector3d association_translation = snapshot->associationPose().translation();
  const double inverse_variance =
      1.0 / (snapshot->residualStandardDeviationM() * snapshot->residualStandardDeviationM());
  for (const FrozenPointCorrespondence& row : snapshot->rows()) {
    if (!row.source_point.allFinite() || !row.target_point.allFinite() ||
        !std::isfinite(row.association_distance_squared_m2) ||
        row.association_distance_squared_m2 < 0.0 ||
        !finitePositive(row.association_huber_weight) || row.association_huber_weight > 1.0) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::InvalidTarget,
          "direct point-to-point ICP admission snapshot contains an invalid frozen row", work));
    }
    const Eigen::Vector3d residual =
        association_rotation * row.source_point + association_translation - row.target_point;
    const double squared_residual = residual.squaredNorm();
    if (!residual.allFinite() || !std::isfinite(squared_residual)) {
      return Result::failure(
          makeError(LidarRegistrationErrorCode::NumericalFailure,
                    "direct point-to-point ICP admission residual metric is non-finite", work));
    }
    const double expected_weight =
        huberWeight(std::sqrt(squared_residual), snapshot->huberDeltaM());
    const double scalar_tolerance =
        1.0e-9 * std::max({1.0, squared_residual, row.association_distance_squared_m2});
    if (std::abs(squared_residual - row.association_distance_squared_m2) > scalar_tolerance ||
        std::abs(expected_weight - row.association_huber_weight) > 1.0e-12) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::InvalidTarget,
          "direct point-to-point ICP admission snapshot metric or robust weight is stale", work));
    }

    Eigen::Matrix<double, 3, 6> geometric_jacobian;
    geometric_jacobian.leftCols<3>() = association_rotation;
    geometric_jacobian.rightCols<3>() = -association_rotation * skew(row.source_point);
    raw_hessian.noalias() += row.association_huber_weight * inverse_variance *
                             geometric_jacobian.transpose() * geometric_jacobian;
  }
  raw_hessian = 0.5 * (raw_hessian + raw_hessian.transpose());
  if (!raw_hessian.allFinite()) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "direct point-to-point ICP admission raw information is non-finite", work));
  }

  const double characteristic_length = snapshot->characteristicLengthM();
  const double geometric_information_scale = snapshot->geometricInformationScale();
  if (!finitePositive(characteristic_length) ||
      characteristic_length < config.minimum_characteristic_length_m ||
      characteristic_length > config.maximum_characteristic_length_m ||
      !finitePositive(geometric_information_scale) || geometric_information_scale > 1.0) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidTarget,
        "direct point-to-point ICP snapshot has an invalid sealed information profile", work));
  }
  Matrix6 normalization = Matrix6::Identity();
  normalization.diagonal().head<3>().setConstant(characteristic_length);
  Matrix6 normalized_hessian = normalization * raw_hessian * normalization;
  normalized_hessian = 0.5 * (normalized_hessian + normalized_hessian.transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix6> normalized_solver(normalized_hessian);
  if (normalized_solver.info() != Eigen::Success || !normalized_solver.eigenvalues().allFinite() ||
      !normalized_solver.eigenvectors().allFinite()) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "direct point-to-point ICP admission normalized observability decomposition failed", work));
  }
  const double threshold = std::max(config.absolute_normalized_observable_eigenvalue,
                                    config.relative_normalized_observable_eigenvalue *
                                        normalized_solver.eigenvalues().cwiseMax(0.0).maxCoeff());
  Matrix6 normalized_projected = Matrix6::Zero();
  std::size_t rank = 0U;
  for (Eigen::Index index = 0; index < 6; ++index) {
    const double eigenvalue = std::max(0.0, normalized_solver.eigenvalues()(index));
    if (eigenvalue < threshold) {
      continue;
    }
    const Vector6 basis = normalized_solver.eigenvectors().col(index);
    normalized_projected.noalias() +=
        geometric_information_scale * eigenvalue * basis * basis.transpose();
    ++rank;
  }
  if (rank < config.minimum_observable_rank) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InsufficientObservableRank,
        "direct point-to-point ICP admission snapshot has insufficient observable rank", work));
  }

  Matrix6 inverse_normalization = Matrix6::Identity();
  inverse_normalization.diagonal().head<3>().setConstant(1.0 / characteristic_length);
  Matrix6 physical = inverse_normalization * normalized_projected * inverse_normalization;
  physical = 0.5 * (physical + physical.transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix6> physical_solver(physical);
  if (physical_solver.info() != Eigen::Success || !physical_solver.eigenvalues().allFinite() ||
      !physical_solver.eigenvectors().allFinite()) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "direct point-to-point ICP admission physical decomposition failed", work));
  }

  core::RankAwareInformation information;
  information.rank = rank;
  for (std::size_t mode = 0U; mode < 6U; ++mode) {
    const Eigen::Index input = 5 - static_cast<Eigen::Index>(mode);
    const Vector6 basis = physical_solver.eigenvectors().col(input);
    information.basis.col(static_cast<Eigen::Index>(mode)) = basis;
    if (mode >= rank) {
      continue;
    }
    const double shaped = std::max(0.0, physical_solver.eigenvalues()(input));
    if (!finitePositive(shaped)) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "direct point-to-point ICP admission supported information is non-positive", work));
    }
    information.eigenvalues(static_cast<Eigen::Index>(mode)) = information_scale * shaped;
  }
  return Result::success(std::move(information));
}

core::Result<double, LidarRegistrationError> lidarFinalizedMapOwnerPoseCovarianceInflation(
    const std::shared_ptr<const FinalizedMapLidarFactorSnapshot>& snapshot) {
  using Result = core::Result<double, LidarRegistrationError>;
  LidarRegistrationWorkCounters work;
  if (!snapshot || !core::contentHashPresent(snapshot->checksum()) || snapshot->rows().empty()) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::InvalidTarget,
                  "finalized-map direct point-to-point ICP covariance snapshot is invalid", work));
  }
  auto inflation = computeFinalizedMapOwnerPoseCovarianceInflation(
      snapshot->rows(), snapshot->owners(), snapshot->residualStandardDeviationM(), work);
  if (!inflation) {
    return Result::failure(inflation.error());
  }
  if (inflation.value() != snapshot->ownerPoseCovarianceInflation()) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::InvalidTarget,
                  "finalized-map direct point-to-point ICP covariance inflation is stale", work));
  }
  return Result::success(inflation.value());
}

core::Result<core::RankAwareInformation, LidarRegistrationError> lidarFinalizedMapFactorInformation(
    const std::shared_ptr<const FinalizedMapLidarFactorSnapshot>& snapshot,
    const LidarRegistrationConfig& config, double information_scale) {
  using Result = core::Result<core::RankAwareInformation, LidarRegistrationError>;
  LidarRegistrationWorkCounters work;
  if (!validConfig(config) || !std::isfinite(information_scale) || information_scale <= 0.0 ||
      information_scale > 1.0) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidConfig,
        "finalized-map direct point-to-point ICP admission configuration or scale is invalid",
        work));
  }
  if (!snapshot || !core::contentHashPresent(snapshot->checksum()) ||
      !core::contentHashPresent(snapshot->sourceCloudChecksum()) ||
      !core::contentHashPresent(snapshot->mapChecksum()) || !snapshot->sourceState().valid() ||
      !snapshot->sourceSweep().valid() || !snapshot->mapOdomEpoch().valid() ||
      !snapshot->mapSensor().valid() ||
      snapshot->mapSensor().modality != core::SensorModality::Lidar ||
      snapshot->mapVersion() == 0U || !finitePose(snapshot->associationPose()) ||
      snapshot->sourcePointCount() < snapshot->rows().size() ||
      snapshot->rows().size() < config.minimum_correspondences ||
      snapshot->uniqueOwnerCount() == 0U ||
      snapshot->residualStandardDeviationM() != config.residual_standard_deviation_m ||
      snapshot->huberDeltaM() < config.minimum_huber_delta_m ||
      snapshot->huberDeltaM() > config.maximum_huber_delta_m ||
      !std::is_sorted(snapshot->rows().begin(), snapshot->rows().end(),
                      frozenFinalizedMapRowCanonicalLess)) {
    return Result::failure(makeError(LidarRegistrationErrorCode::InvalidTarget,
                                     "finalized-map direct point-to-point ICP admission snapshot "
                                     "identity, support, or profile is invalid",
                                     work));
  }
  auto covariance_inflation = lidarFinalizedMapOwnerPoseCovarianceInflation(snapshot);
  if (!covariance_inflation) {
    return Result::failure(covariance_inflation.error());
  }
  auto checksum = finalizedMapSnapshotChecksum(
      snapshot->sourceState(), snapshot->sourceTime(), snapshot->sourceSweep(),
      snapshot->sourceCloudChecksum(), snapshot->sourcePointCount(), snapshot->mapOdomEpoch(),
      snapshot->mapSensor(), snapshot->mapVersion(), snapshot->mapChecksum(),
      snapshot->associationPose(), snapshot->owners(), snapshot->rows(),
      snapshot->candidateVoxelLookups(), snapshot->candidateOccupiedVoxels(),
      snapshot->candidatePointsExamined(), snapshot->sourceRowsExcludedByOwnership(),
      snapshot->huberDeltaM(), snapshot->characteristicLengthM(),
      snapshot->geometricInformationScale(), covariance_inflation.value(), config, work);
  if (!checksum || checksum.value() != snapshot->checksum()) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidTarget,
        "finalized-map direct point-to-point ICP admission snapshot checksum is stale", work));
  }

  Matrix6 raw_hessian = Matrix6::Zero();
  const Eigen::Matrix3d rotation = snapshot->associationPose().rotationMatrix();
  const Eigen::Vector3d translation = snapshot->associationPose().translation();
  const double inverse_variance =
      1.0 / (snapshot->residualStandardDeviationM() * snapshot->residualStandardDeviationM());
  for (const FrozenFinalizedMapPointCorrespondence& row : snapshot->rows()) {
    if (row.owner_index >= snapshot->owners().size() || !row.source_point.allFinite() ||
        !row.target_point_odom.allFinite() || !std::isfinite(row.association_distance_squared_m2) ||
        row.association_distance_squared_m2 < 0.0 ||
        !finitePositive(row.association_huber_weight) || row.association_huber_weight > 1.0) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::InvalidTarget,
          "finalized-map direct point-to-point ICP admission snapshot contains an invalid row",
          work));
    }
    const Eigen::Vector3d residual =
        rotation * row.source_point + translation - row.target_point_odom;
    const double squared_residual = residual.squaredNorm();
    const double tolerance =
        1.0e-9 * std::max({1.0, squared_residual, row.association_distance_squared_m2});
    const double expected_weight =
        huberWeight(std::sqrt(squared_residual), snapshot->huberDeltaM());
    if (!residual.allFinite() || !std::isfinite(squared_residual) ||
        std::abs(squared_residual - row.association_distance_squared_m2) > tolerance ||
        std::abs(expected_weight - row.association_huber_weight) > 1.0e-12) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::InvalidTarget,
          "finalized-map direct point-to-point ICP admission metric or robust weight is stale",
          work));
    }
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.leftCols<3>() = rotation;
    jacobian.rightCols<3>() = -rotation * skew(row.source_point);
    raw_hessian.noalias() +=
        row.association_huber_weight * inverse_variance * jacobian.transpose() * jacobian;
  }
  raw_hessian = 0.5 * (raw_hessian + raw_hessian.transpose());
  const double characteristic_length = snapshot->characteristicLengthM();
  const double geometric_information_scale = snapshot->geometricInformationScale();
  if (!raw_hessian.allFinite() || !finitePositive(characteristic_length) ||
      characteristic_length < config.minimum_characteristic_length_m ||
      characteristic_length > config.maximum_characteristic_length_m ||
      !finitePositive(geometric_information_scale) || geometric_information_scale > 1.0) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidTarget,
        "finalized-map direct point-to-point ICP sealed information profile is invalid", work));
  }
  Matrix6 normalization = Matrix6::Identity();
  normalization.diagonal().head<3>().setConstant(characteristic_length);
  Matrix6 normalized_hessian = normalization * raw_hessian * normalization;
  normalized_hessian = 0.5 * (normalized_hessian + normalized_hessian.transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix6> normalized_solver(normalized_hessian);
  if (normalized_solver.info() != Eigen::Success || !normalized_solver.eigenvalues().allFinite() ||
      !normalized_solver.eigenvectors().allFinite()) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "finalized-map direct point-to-point ICP normalized decomposition failed", work));
  }
  const double threshold = std::max(config.absolute_normalized_observable_eigenvalue,
                                    config.relative_normalized_observable_eigenvalue *
                                        normalized_solver.eigenvalues().cwiseMax(0.0).maxCoeff());
  Matrix6 normalized_projected = Matrix6::Zero();
  std::size_t rank = 0U;
  for (Eigen::Index index = 0; index < 6; ++index) {
    const double eigenvalue = std::max(0.0, normalized_solver.eigenvalues()(index));
    if (eigenvalue < threshold) {
      continue;
    }
    const Vector6 basis = normalized_solver.eigenvectors().col(index);
    normalized_projected.noalias() +=
        geometric_information_scale * eigenvalue * basis * basis.transpose();
    ++rank;
  }
  if (rank < config.minimum_observable_rank) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::InsufficientObservableRank,
                  "finalized-map direct point-to-point ICP snapshot has insufficient rank", work));
  }
  Matrix6 inverse_normalization = Matrix6::Identity();
  inverse_normalization.diagonal().head<3>().setConstant(1.0 / characteristic_length);
  Matrix6 physical = inverse_normalization * normalized_projected * inverse_normalization;
  physical = 0.5 * (physical + physical.transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix6> physical_solver(physical);
  if (physical_solver.info() != Eigen::Success || !physical_solver.eigenvalues().allFinite() ||
      !physical_solver.eigenvectors().allFinite()) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "finalized-map direct point-to-point ICP physical decomposition failed", work));
  }
  core::RankAwareInformation information;
  information.rank = rank;
  for (std::size_t mode = 0U; mode < 6U; ++mode) {
    const Eigen::Index input = 5 - static_cast<Eigen::Index>(mode);
    information.basis.col(static_cast<Eigen::Index>(mode)) =
        physical_solver.eigenvectors().col(input);
    if (mode >= rank) {
      continue;
    }
    const double shaped = std::max(0.0, physical_solver.eigenvalues()(input));
    if (!finitePositive(shaped)) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "finalized-map direct point-to-point ICP supported information is non-positive", work));
    }
    information.eigenvalues(static_cast<Eigen::Index>(mode)) = information_scale * shaped;
  }
  return Result::success(std::move(information));
}

[[nodiscard]] core::Result<LidarRegistrationResult, LidarRegistrationError> registerLidarScanImpl(
    core::StateId source_state, const std::shared_ptr<const LidarRegistrationCloud>& source,
    std::span<const LidarRegistrationTarget> targets,
    const FinalizedLidarTargetReadView* finalized_map,
    core::LocalGraphRevision live_pose_revision, const LidarRegistrationConfig& config) {
  using Result = core::Result<LidarRegistrationResult, LidarRegistrationError>;
  LidarRegistrationWorkCounters work;
  if (!validConfig(config)) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::InvalidConfig,
                  "pose-aware direct point-to-point ICP configuration is invalid", work));
  }
  if (!source_state.valid()) {
    return Result::failure(makeError(LidarRegistrationErrorCode::InvalidSource,
                                     "direct point ICP source state is invalid", work));
  }
  const bool has_finalized_map = finalized_map != nullptr;
  const std::size_t requested_channel_count = targets.size() + (has_finalized_map ? 1U : 0U);
  if (requested_channel_count == 0U || requested_channel_count > config.maximum_targets ||
      targets.size() > config.maximum_composite_owners ||
      (has_finalized_map &&
       (!finalized_map->odomEpoch().valid() || !finalized_map->sensor().valid() ||
        finalized_map->sensor().modality != core::SensorModality::Lidar ||
        finalized_map->version() == 0U ||
        !core::contentHashPresent(finalized_map->checksum()))) ||
      (!targets.empty() && !live_pose_revision.valid())) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidTarget,
        "direct point-to-point ICP live plus finalized-map target count or identity is invalid",
        work));
  }
  auto selected = selectSourcePoints(source, config, &work);
  if (!selected) {
    return Result::failure(selected.error());
  }
  if (selected.value().size() < config.minimum_correspondences) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InvalidSource,
        "direct point-to-point ICP source voxel selection has insufficient support", work));
  }
  auto target_indices = buildTargetIndices(source_state, *source, targets, config, &work);
  if (!target_indices) {
    return Result::failure(target_indices.error());
  }
  const double characteristic_length =
      sourceCharacteristicLength(*source, selected.value(), config);
  if (!finitePositive(characteristic_length)) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "direct point-to-point ICP source characteristic length is invalid", work));
  }

  core::Pose3d correction;
  LidarRegistrationWorkspace workspace;
  const FinalizedLidarTargetReadView* active_finalized_map = finalized_map;
  std::unique_ptr<CompositeTargetContext> composite;
  AssociationSet associations;
  const auto rebuildCompositeTarget = [&]() -> std::optional<LidarRegistrationError> {
    if (target_indices.value().empty()) {
      composite.reset();
      return std::nullopt;
    }
    auto rebuilt =
        buildCompositeTarget(target_indices.value(), live_pose_revision, config, &work);
    if (!rebuilt) {
      return rebuilt.error();
    }
    composite =
        std::make_unique<CompositeTargetContext>(std::move(rebuilt).value());
    return std::nullopt;
  };
  const auto rebuildAssociations = [&]() -> std::optional<LidarRegistrationError> {
    if (!composite && active_finalized_map == nullptr) {
      return makeError(LidarRegistrationErrorCode::NumericalFailure,
                       "direct point-to-point ICP has no live overlay or finalized-map base", work);
    }
    auto rebuilt = buildAssociations(*source, selected.value(), target_indices.value(), correction,
                                     active_finalized_map, composite.get(), config, &workspace,
                                     &work);
    if (!rebuilt) {
      return rebuilt.error();
    }
    associations = std::move(rebuilt).value();
    return std::nullopt;
  };
  if (auto error = rebuildCompositeTarget()) {
    return Result::failure(std::move(*error));
  }
  if (auto error = rebuildAssociations()) {
    return Result::failure(std::move(*error));
  }
  // Every retained channel must independently support a factor. Remove weak
  // live/map channels, then rebuild exclusive ownership from the complete
  // selected source population. If all channels are initially weak, retain
  // the strongest one; a live channel wins an exact row-count tie.
  const auto retainIndependentlySupportedTargets = [&]() -> std::optional<LidarRegistrationError> {
    const std::size_t channel_count =
        target_indices.value().size() + (active_finalized_map != nullptr ? 1U : 0U);
    if (channel_count <= 1U) {
      return std::nullopt;
    }
    std::vector<std::size_t> retained_indices;
    for (std::size_t index = 0U; index < associations.targets.size(); ++index) {
      if (associations.targets[index].rows.size() >= config.minimum_correspondences) {
        retained_indices.push_back(index);
      }
    }
    bool retain_map = associations.finalized_map &&
                      associations.finalized_map->rows.size() >= config.minimum_correspondences;
    if (retained_indices.empty() && !retain_map) {
      std::optional<std::size_t> best_live;
      for (std::size_t index = 0U; index < associations.targets.size(); ++index) {
        if (!best_live ||
            associations.targets[index].rows.size() >
                associations.targets[*best_live].rows.size() ||
            (associations.targets[index].rows.size() ==
                 associations.targets[*best_live].rows.size() &&
             target_indices.value()[*best_live].record.state <
                 target_indices.value()[index].record.state)) {
          best_live = index;
        }
      }
      const std::size_t best_live_rows =
          best_live ? associations.targets[*best_live].rows.size() : 0U;
      const std::size_t map_rows =
          associations.finalized_map ? associations.finalized_map->rows.size() : 0U;
      if (associations.finalized_map && map_rows > best_live_rows) {
        retain_map = true;
      } else if (best_live) {
        retained_indices.push_back(*best_live);
      }
    }
    const bool changed = retained_indices.size() != target_indices.value().size() ||
                         retain_map != (active_finalized_map != nullptr);
    if (changed) {
      std::vector<TargetIndex> retained_targets;
      retained_targets.reserve(retained_indices.size());
      for (const std::size_t index : retained_indices) {
        retained_targets.push_back(std::move(target_indices.value()[index]));
      }
      target_indices.value() = std::move(retained_targets);
      if (!retain_map) {
        active_finalized_map = nullptr;
      }
      if (auto error = rebuildCompositeTarget()) {
        return error;
      }
      if (auto error = rebuildAssociations()) {
        return error;
      }
    }
    return std::nullopt;
  };
  if (auto error = retainIndependentlySupportedTargets()) {
    return Result::failure(std::move(*error));
  }
  if (associations.correspondences < config.minimum_correspondences) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InsufficientCorrespondences,
        "initial pose-aware direct point-to-point ICP association has insufficient support", work));
  }
  if (std::any_of(associations.targets.begin(), associations.targets.end(),
                  [&](const WorkingTarget& target) {
                    return target.rows.size() < config.minimum_correspondences;
                  }) ||
      (associations.finalized_map &&
       associations.finalized_map->rows.size() < config.minimum_correspondences)) {
    return Result::failure(makeError(LidarRegistrationErrorCode::InsufficientCorrespondences,
                                     "pose-aware direct point-to-point ICP retained a target with "
                                     "insufficient independent factor support",
                                     work));
  }
  auto current =
      evaluate(&associations, correction, characteristic_length, config, true, &workspace, &work);
  if (!current) {
    return Result::failure(current.error());
  }
  if (current.value().observability.rank < config.minimum_observable_rank) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::InsufficientObservableRank,
        "initial pose-aware direct point-to-point ICP association has insufficient observable rank",
        work));
  }
  const double initial_cost = current.value().robust_cost;
  bool converged = false;

  for (std::size_t outer = 0U; outer < config.maximum_outer_iterations; ++outer) {
    if (!checkedAdd(1U, &work.outer_iterations)) {
      return Result::failure(
          makeError(LidarRegistrationErrorCode::NumericalFailure,
                    "direct point-to-point ICP outer-iteration counter overflowed", work));
    }
    Matrix6 normalization = Matrix6::Identity();
    normalization.diagonal().head<3>().setConstant(characteristic_length);
    const Vector6 normalized_gradient = normalization * current.value().gradient;
    const double damping_scale =
        std::max(1.0, current.value().observability.eigenvalues.maxCoeff());
    bool accepted = false;
    bool gn_converged = false;
    for (std::size_t trial = 0U; trial <= config.maximum_lm_damping_retries; ++trial) {
      double relative_damping = 0.0;
      if (trial == 0U) {
        if (!checkedAdd(1U, &work.gauss_newton_trials)) {
          return Result::failure(
              makeError(LidarRegistrationErrorCode::NumericalFailure,
                        "direct point-to-point ICP Gauss-Newton counter overflowed", work));
        }
      } else {
        if (!checkedAdd(1U, &work.lm_damping_trials)) {
          return Result::failure(
              makeError(LidarRegistrationErrorCode::NumericalFailure,
                        "direct point-to-point ICP LM-damping counter overflowed", work));
        }
        relative_damping = config.initial_relative_damping *
                           std::pow(config.damping_increase, static_cast<double>(trial - 1U));
      }
      Vector6 normalized_step = Vector6::Zero();
      for (Eigen::Index direction = 0; direction < 6; ++direction) {
        const double eigenvalue = current.value().observability.eigenvalues(direction);
        if (eigenvalue < current.value().observability.threshold) {
          continue;
        }
        const Vector6 basis = current.value().observability.eigenvectors.col(direction);
        const double denominator = eigenvalue + relative_damping * damping_scale;
        normalized_step.noalias() -= basis.dot(normalized_gradient) / denominator * basis;
      }
      const Vector6 step = normalization * normalized_step;
      if (!step.allFinite()) {
        continue;
      }
      if (trial == 0U && step.head<3>().norm() <= config.translation_convergence_m &&
          step.tail<3>().norm() <= config.rotation_convergence_rad) {
        gn_converged = true;
        break;
      }
      const core::Pose3d candidate = correction * core::Pose3d::exp(step);
      if (!finitePose(candidate) ||
          candidate.translation().norm() > config.maximum_correction_translation_m ||
          candidate.so3().log().norm() > config.maximum_correction_rotation_rad) {
        continue;
      }
      auto candidate_evaluation = evaluate(&associations, candidate, characteristic_length, config,
                                           false, &workspace, &work);
      if (!candidate_evaluation) {
        continue;
      }
      const double tolerance = 1.0e-12 * std::max(1.0, std::abs(current.value().robust_cost));
      if (candidate_evaluation.value().robust_cost >= current.value().robust_cost - tolerance) {
        if (!checkedAdd(1U, &work.rejected_frozen_cost_trials)) {
          return Result::failure(
              makeError(LidarRegistrationErrorCode::NumericalFailure,
                        "direct point-to-point ICP rejected-trial counter overflowed", work));
        }
        continue;
      }

      correction = candidate;
      if (!checkedAdd(1U, &work.accepted_steps)) {
        return Result::failure(
            makeError(LidarRegistrationErrorCode::NumericalFailure,
                      "direct point-to-point ICP accepted-step counter overflowed", work));
      }
      if (auto error = rebuildAssociations()) {
        return Result::failure(std::move(*error));
      }
      if (auto error = retainIndependentlySupportedTargets()) {
        return Result::failure(std::move(*error));
      }
      if (associations.correspondences < config.minimum_correspondences) {
        return Result::failure(makeError(LidarRegistrationErrorCode::InsufficientCorrespondences,
                                         "accepted direct point-to-point ICP step lost "
                                         "correspondence support after reassociation",
                                         work));
      }
      if (std::any_of(associations.targets.begin(), associations.targets.end(),
                      [&](const WorkingTarget& target) {
                        return target.rows.size() < config.minimum_correspondences;
                      }) ||
          (associations.finalized_map &&
           associations.finalized_map->rows.size() < config.minimum_correspondences)) {
        return Result::failure(makeError(LidarRegistrationErrorCode::InsufficientCorrespondences,
                                         "accepted direct point-to-point ICP step retained an "
                                         "independently under-supported target",
                                         work));
      }
      current = evaluate(&associations, correction, characteristic_length, config, true, &workspace,
                         &work);
      if (!current) {
        return Result::failure(current.error());
      }
      if (current.value().observability.rank < config.minimum_observable_rank) {
        return Result::failure(makeError(
            LidarRegistrationErrorCode::InsufficientObservableRank,
            "accepted direct point-to-point ICP step lost observable rank after reassociation",
            work));
      }
      accepted = true;
      if (step.head<3>().norm() <= config.translation_convergence_m &&
          step.tail<3>().norm() <= config.rotation_convergence_rad) {
        converged = true;
      }
      break;
    }
    if (gn_converged) {
      converged = true;
      break;
    }
    if (!accepted) {
      return Result::failure(makeError(LidarRegistrationErrorCode::NoDecreasingStep,
                                       "bounded direct point-to-point ICP GN/LM trials found no "
                                       "decreasing frozen-objective step",
                                       work));
    }
    if (converged) {
      break;
    }
  }

  LidarRegistrationResult output;
  output.source_state = source_state;
  output.source_time = source->reference_time;
  output.source_right_correction = correction;
  output.T_odom_source = source->T_odom_imu_seed * correction;
  output.initial_robust_cost = initial_cost;
  output.final_robust_cost = current.value().robust_cost;
  output.termination = converged ? LidarRegistrationTermination::Converged
                                 : LidarRegistrationTermination::IterationLimitReached;
  output.work = work;
  output.raw_physical_hessian = current.value().hessian;
  output.diagnostics.live_target_count = associations.targets.size();
  output.diagnostics.finalized_map_target_count = associations.finalized_map ? 1U : 0U;
  output.diagnostics.target_count =
      output.diagnostics.live_target_count + output.diagnostics.finalized_map_target_count;
  output.diagnostics.correspondences = associations.correspondences;
  for (const WorkingTarget& target : associations.targets) {
    output.diagnostics.live_correspondences += target.rows.size();
  }
  output.diagnostics.finalized_map_correspondences =
      associations.finalized_map ? associations.finalized_map->rows.size() : 0U;
  output.diagnostics.overlap_fraction = static_cast<double>(associations.correspondences) /
                                        static_cast<double>(selected.value().size());
  output.diagnostics.effective_correspondences = current.value().effective_correspondences;
  output.diagnostics.maximum_squared_residual_m2 = current.value().maximum_squared_residual_m2;
  output.diagnostics.huber_delta_m = associations.huber_delta_m;
  output.diagnostics.characteristic_length_m = characteristic_length;
  output.diagnostics.normalized_observable_eigenvalue_threshold =
      current.value().observability.threshold;
  output.diagnostics.observable_rank = current.value().observability.rank;

  Matrix6 normalized_projected = Matrix6::Zero();
  double maximum_supported_normalized_information = 0.0;
  for (Eigen::Index input_index = 0; input_index < 6; ++input_index) {
    const double eigenvalue = current.value().observability.eigenvalues(input_index);
    if (eigenvalue < current.value().observability.threshold) {
      continue;
    }
    const Vector6 basis = current.value().observability.eigenvectors.col(input_index);
    output.normalized_observability_projector.noalias() += basis * basis.transpose();
    normalized_projected.noalias() += eigenvalue * basis * basis.transpose();
    maximum_supported_normalized_information =
        std::max(maximum_supported_normalized_information, eigenvalue);
  }
  const double normalized_information_cap =
      config.maximum_translation_information * characteristic_length * characteristic_length;
  if (!finitePositive(maximum_supported_normalized_information) ||
      !finitePositive(normalized_information_cap)) {
    return Result::failure(
        makeError(LidarRegistrationErrorCode::NumericalFailure,
                  "direct point-to-point ICP normalized information scale is invalid", work));
  }
  const double geometric_information_scale =
      std::min(1.0, normalized_information_cap / maximum_supported_normalized_information);
  normalized_projected *= geometric_information_scale;
  output.diagnostics.maximum_supported_normalized_information =
      maximum_supported_normalized_information;
  output.diagnostics.normalized_information_cap = normalized_information_cap;
  output.diagnostics.geometric_information_scale = geometric_information_scale;
  for (Eigen::Index output_index = 0; output_index < 6; ++output_index) {
    const Eigen::Index input_index = 5 - output_index;
    output.diagnostics.raw_normalized_hessian_eigenvalues(output_index) =
        current.value().observability.eigenvalues(input_index);
    output.diagnostics.normalized_directional_information.basis.col(output_index) =
        current.value().observability.eigenvectors.col(input_index);
    output.diagnostics.normalized_directional_information.eigenvalues(output_index) =
        current.value().observability.eigenvalues(input_index) >=
                current.value().observability.threshold
            ? geometric_information_scale * current.value().observability.eigenvalues(input_index)
            : 0.0;
  }
  output.diagnostics.normalized_directional_information.rank = current.value().observability.rank;

  Matrix6 inverse_normalization = Matrix6::Identity();
  inverse_normalization.diagonal().head<3>().setConstant(1.0 / characteristic_length);
  Matrix6 projected_physical = inverse_normalization * normalized_projected * inverse_normalization;
  projected_physical = 0.5 * (projected_physical + projected_physical.transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix6> physical_solver(projected_physical);
  if (physical_solver.info() != Eigen::Success || !physical_solver.eigenvalues().allFinite() ||
      !physical_solver.eigenvectors().allFinite()) {
    return Result::failure(makeError(
        LidarRegistrationErrorCode::NumericalFailure,
        "projected physical direct point-to-point ICP information eigendecomposition failed",
        work));
  }
  const std::size_t rank = current.value().observability.rank;
  for (std::size_t direction = 0U; direction < rank; ++direction) {
    const Eigen::Index input_index = 5 - static_cast<Eigen::Index>(direction);
    const double shaped = std::max(0.0, physical_solver.eigenvalues()(input_index));
    if (!finitePositive(shaped)) {
      return Result::failure(makeError(
          LidarRegistrationErrorCode::NumericalFailure,
          "supported physical direct point-to-point ICP information direction is non-positive",
          work));
    }
    const Vector6 basis = physical_solver.eigenvectors().col(input_index);
    output.projected_physical_information.noalias() += shaped * basis * basis.transpose();
    output.supported_physical_covariance.noalias() += (1.0 / shaped) * basis * basis.transpose();
    output.diagnostics.physical_information.basis.col(static_cast<Eigen::Index>(direction)) = basis;
    output.diagnostics.physical_information.eigenvalues(static_cast<Eigen::Index>(direction)) =
        shaped;
  }
  output.diagnostics.physical_information.rank = rank;

  output.target_snapshots.reserve(associations.targets.size());
  for (const WorkingTarget& target : associations.targets) {
    auto snapshot = detail::LidarFactorSnapshotBuilder::build(
        source_state, *source, target, correction, associations.huber_delta_m,
        characteristic_length, geometric_information_scale, config, work);
    if (!snapshot) {
      return Result::failure(snapshot.error());
    }
    output.target_snapshots.push_back(std::move(snapshot.value()));
  }
  if (associations.finalized_map) {
    auto snapshot = detail::FinalizedMapLidarFactorSnapshotBuilder::build(
        source_state, *source, *associations.finalized_map, correction, associations.huber_delta_m,
        characteristic_length, geometric_information_scale, config, work);
    if (!snapshot) {
      return Result::failure(snapshot.error());
    }
    output.finalized_map_snapshot = std::move(snapshot.value());
  }
  return Result::success(std::move(output));
}

core::Result<LidarRegistrationResult, LidarRegistrationError> registerLidarScan(
    core::StateId source_state, const std::shared_ptr<const LidarRegistrationCloud>& source,
    std::span<const LidarRegistrationTarget> targets, const LidarRegistrationConfig& config) {
  return registerLidarScanImpl(source_state, source, targets, nullptr,
                               core::LocalGraphRevision{1U}, config);
}

core::Result<LidarRegistrationResult, LidarRegistrationError> registerLidarScan(
    core::StateId source_state, const std::shared_ptr<const LidarRegistrationCloud>& source,
    std::span<const LidarRegistrationTarget> targets,
    const FinalizedLidarTargetReadView& finalized_map,
    core::LocalGraphRevision live_pose_revision, const LidarRegistrationConfig& config) {
  return registerLidarScanImpl(source_state, source, targets, &finalized_map, live_pose_revision,
                               config);
}

}  // namespace meridian::local
