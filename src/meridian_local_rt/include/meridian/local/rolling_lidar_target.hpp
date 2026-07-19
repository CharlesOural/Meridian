#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/lidar_registration.hpp"

namespace meridian::local {

inline constexpr std::string_view kRollingLidarTargetBatchChecksumDomain{
    "meridian.local.rolling_lidar_target.batch"};
inline constexpr std::uint32_t kRollingLidarTargetBatchChecksumSchemaVersion{3U};

// One localization-accepted sweep. Geometry remains in the sweep reference
// IMU frame; T_odom_imu is the optimized pose owned by exactly this state.
// LidarRegistrationCloud has private validating construction and no mutable/copyable
// form, so the sole public admission path safely shares it by identity.
struct RegisteredLidarSweep {
  core::OdomEpoch odom_epoch;
  core::StateId state;
  // Graph admission provenance is retained with the immutable payload so a
  // later bounded sensor-failure rollback can remove exactly the clouds whose
  // localization evidence was removed.
  core::FactorBatchId admitting_batch_id;
  core::SensorRecoveryEpoch recovery_epoch;
  core::Pose3d T_odom_imu;
  std::shared_ptr<const LidarRegistrationCloud> cloud;
};

struct RollingLidarTargetConfig {
  core::OdomEpoch odom_epoch;
  std::size_t maximum_retained_sweeps{5U};
  std::size_t maximum_retained_points{200'000U};
  // Target selection is one deterministic policy: the newest retained sweep
  // plus targets distributed across the complete active history. It performs
  // no speculative nearest-neighbour pass; registerLidarScan is the sole owner
  // of association and overlap validation.
  LidarRegistrationConfig registration;
};

struct RollingLidarTargetEvictionStats {
  std::size_t sweeps{};
  std::size_t points{};
};

struct RollingLidarTargetAddStats {
  std::size_t input_points{};
  std::size_t retained_points_from_input{};
  std::size_t copied_raw_points{};
  std::size_t deterministic_cap_discarded_points{};
  RollingLidarTargetEvictionStats eviction;
  std::size_t retained_sweeps{};
  std::size_t retained_points{};
};

struct RollingLidarTargetPose {
  core::StateId state;
  core::Pose3d T_odom_imu;
};

struct RollingLidarTargetPoseSynchronizationStats {
  std::size_t supplied_poses{};
  std::size_t matched_retained_sweeps{};
  std::size_t finalized_sweeps_evicted{};
  std::size_t finalized_points_evicted{};
  std::size_t retained_sweeps{};
  std::size_t retained_points{};
  double maximum_translation_revision_m{};
  double maximum_rotation_revision_rad{};
};

struct RollingLidarTargetRemovalStats {
  std::size_t requested_batches{};
  std::size_t matched_batches{};
  std::size_t absent_batches{};
  std::size_t removed_sweeps{};
  std::size_t removed_points{};
  std::size_t retained_sweeps{};
  std::size_t retained_points{};
};

struct RollingLidarTargetBatchBuildStats {
  std::size_t retained_candidates{};
  std::size_t selected_targets{};
  std::uint64_t selected_state_span{};
  std::int64_t selected_time_span_ns{};
};

struct RollingLidarTargetStatistics {
  std::size_t add_attempts{};
  std::size_t rejected_adds{};
  std::size_t accepted_sweeps{};
  std::size_t input_points{};
  std::size_t retained_points_from_input{};
  std::size_t copied_raw_points{};
  std::size_t deterministic_cap_discarded_points{};
  RollingLidarTargetEvictionStats eviction;
  std::size_t pose_synchronization_attempts{};
  std::size_t rejected_pose_synchronizations{};
  std::size_t pose_synchronized_sweeps{};
  std::size_t finalized_sweeps_evicted{};
  std::size_t finalized_points_evicted{};
  std::size_t batch_removal_attempts{};
  std::size_t rejected_batch_removals{};
  std::size_t batch_removal_transactions{};
  std::size_t removed_sweeps{};
  std::size_t removed_points{};
  std::size_t batch_build_attempts{};
  std::size_t rejected_batch_builds{};
  std::size_t built_target_batches{};
  std::size_t batch_selected_targets{};
  std::size_t retained_sweeps{};
  std::size_t retained_points{};
};

// Read-only observability snapshot of the retained window. Each cloud belongs
// to one state only; there is deliberately no merged-map representation.
struct RollingLidarTargetSweep {
  core::StateId state;
  core::FusionTime reference_time;
  core::FactorBatchId admitting_batch_id;
  core::SensorRecoveryEpoch recovery_epoch;
  core::Pose3d T_odom_imu;
  std::shared_ptr<const LidarRegistrationCloud> cloud;
  core::ObservationLineage lineage;
  core::ContentHash checksum{};
};

struct RollingLidarTargetBatch {
  core::OdomEpoch odom_epoch;
  core::MeasurementId source_sweep;
  core::FusionTime source_reference_time;
  core::Pose3d T_odom_imu_source_seed;
  // Makes the batch checksum self-contained without retaining the potentially
  // large source cloud. The cloud checksum already covers its complete sealed
  // registration representation, lineage, and exact-index profile.
  core::ContentHash source_cloud_checksum{};
  // Canonical newest-state-first order. Each record is immediately consumable
  // by registerLidarScan: target geometry is scan-local, its cloud seed is
  // immutable preprocessing metadata, its explicit target seed is the
  // atomically synchronized optimized pose, and the relative seed is exact.
  // No record contains geometry from another retained state.
  std::vector<LidarRegistrationTarget> targets;
  core::ObservationLineage lineage;
  core::ContentHash checksum{};
  RollingLidarTargetBatchBuildStats build;
};

enum class RollingLidarTargetErrorCode {
  InvalidConfig,
  EpochMismatch,
  InvalidIdentity,
  DuplicateState,
  DuplicateSweep,
  DuplicateFactorBatch,
  NonMonotonicState,
  NonMonotonicSweep,
  NonMonotonicTime,
  NonMonotonicFactorBatch,
  NonMonotonicRecoveryEpoch,
  InvalidPose,
  MissingCommittedPose,
  InvalidCloud,
  InvalidLineage,
  EmptyTarget,
  TargetCapacity,
  RemovalCapacity,
  SourceAlreadyRetained,
  ChecksumFailure,
};

struct RollingLidarTargetError {
  RollingLidarTargetErrorCode code{};
  std::string detail;
};

// Recomputes the complete batch-local integrity checksum. Unlike the optional
// canonical raw-root lineage checksum, this digest always encodes the full
// merged lineage value (including its ID, usages, correlations, and absent
// source-checksum markers), so legacy inputs cannot leave mutable metadata
// outside the integrity envelope.
[[nodiscard]] core::Result<core::ContentHash, RollingLidarTargetError>
recomputeRollingLidarTargetBatchChecksum(const RollingLidarTargetBatch& batch);

// ROS-free bounded owner for pose-aware direct point ICP support. Admission, pose refresh,
// and batch publication are transactional: a rejected operation cannot expose
// a partially revised retained window.
class RollingLidarTargetBuilder {
public:
  [[nodiscard]] static core::Result<RollingLidarTargetBuilder, RollingLidarTargetError> create(
      RollingLidarTargetConfig config);

  ~RollingLidarTargetBuilder();
  RollingLidarTargetBuilder(RollingLidarTargetBuilder&&) noexcept;
  RollingLidarTargetBuilder& operator=(RollingLidarTargetBuilder&&) noexcept;
  RollingLidarTargetBuilder(const RollingLidarTargetBuilder&) = delete;
  RollingLidarTargetBuilder& operator=(const RollingLidarTargetBuilder&) = delete;

  [[nodiscard]] core::Result<RollingLidarTargetAddStats, RollingLidarTargetError> add(
      RegisteredLidarSweep sweep);

  [[nodiscard]] core::Result<RollingLidarTargetPoseSynchronizationStats, RollingLidarTargetError>
  synchronizeCommittedPoses(core::OdomEpoch odom_epoch,
                            std::span<const RollingLidarTargetPose> poses,
                            std::span<const core::StateId> finalized_states = {});

  // Removes only payloads admitted by the requested FactorBatch identities.
  // The request is bounded by maximum_retained_sweeps and is transactional:
  // invalid/duplicate identities reject without mutation, while valid absent
  // identities are inert. Last-seen admission identities are never rewound.
  [[nodiscard]] core::Result<RollingLidarTargetRemovalStats, RollingLidarTargetError>
  removeFactorBatches(core::OdomEpoch odom_epoch,
                      std::span<const core::FactorBatchId> batch_ids);

  // The source must not already be retained. Selection is deterministic and
  // distributed across the active retained history; no correspondence search
  // occurs here. A scan-local target_limit lets a second direct-registration
  // channel (for example the finalized map) reserve capacity without
  // publishing ancestry for live targets that the solve will not consume.
  // The limit must be positive and no larger than the configured maximum.
  [[nodiscard]] core::Result<RollingLidarTargetBatch, RollingLidarTargetError> buildBatch(
      std::shared_ptr<const LidarRegistrationCloud> source,
      core::ObservationLineageId output_lineage,
      std::optional<std::size_t> target_limit = std::nullopt);

  [[nodiscard]] std::vector<RollingLidarTargetSweep> retainedSweeps() const;
  [[nodiscard]] const RollingLidarTargetStatistics& statistics() const noexcept;
  [[nodiscard]] bool empty() const noexcept;

private:
  struct Impl;
  explicit RollingLidarTargetBuilder(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;

};

}  // namespace meridian::local
