#include "meridian/local/rolling_lidar_target.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

#include "meridian/core/canonical_bytes.hpp"

namespace meridian::local {
namespace {

struct RetainedSweep {
  core::StateId state;
  core::FusionTime reference_time;
  core::FactorBatchId admitting_batch_id;
  core::SensorRecoveryEpoch recovery_epoch;
  core::Pose3d T_odom_imu;
  std::shared_ptr<const LidarRegistrationCloud> cloud;
};

[[nodiscard]] RollingLidarTargetError targetError(RollingLidarTargetErrorCode code,
                                                  std::string detail) {
  return RollingLidarTargetError{code, std::move(detail)};
}

[[nodiscard]] bool finitePose(const core::Pose3d& pose) noexcept {
  return pose.matrix().allFinite();
}

[[nodiscard]] bool validConfig(const RollingLidarTargetConfig& config) noexcept {
  return config.odom_epoch.valid() && config.maximum_retained_sweeps > 0U &&
         config.maximum_retained_points > 0U &&
         isValidLidarRegistrationConfig(config.registration);
}

[[nodiscard]] bool sameSlice(const core::ObservationSlice& lhs,
                             const core::ObservationSlice& rhs) noexcept {
  return lhs.root == rhs.root && lhs.kind == rhs.kind && lhs.begin == rhs.begin &&
         lhs.end == rhs.end && lhs.source_checksum == rhs.source_checksum &&
         lhs.calibration == rhs.calibration;
}

[[nodiscard]] bool sameUsage(const core::ObservationUsage& lhs,
                             const core::ObservationUsage& rhs) noexcept {
  return sameSlice(lhs.slice, rhs.slice) && lhs.role == rhs.role && lhs.consumer == rhs.consumer &&
         lhs.factor_group == rhs.factor_group && lhs.correlation_group == rhs.correlation_group;
}

[[nodiscard]] bool sameDeclaration(const core::CorrelationDeclaration& lhs,
                                   const core::CorrelationDeclaration& rhs) noexcept {
  return lhs.group == rhs.group && lhs.policy == rhs.policy && lhs.treatment == rhs.treatment &&
         lhs.covariance_inflation == rhs.covariance_inflation &&
         lhs.total_information_cap == rhs.total_information_cap;
}

[[nodiscard]] auto usageCanonicalKey(const core::ObservationUsage& usage) {
  const bool measurement = std::holds_alternative<core::MeasurementId>(usage.slice.root);
  const std::uint64_t root = measurement
                                 ? std::get<core::MeasurementId>(usage.slice.root).value()
                                 : std::get<core::GnssObservationId>(usage.slice.root).value();
  return std::tuple{static_cast<std::uint8_t>(measurement ? 0U : 1U),
                    root,
                    static_cast<std::uint8_t>(usage.slice.kind),
                    usage.slice.begin,
                    usage.slice.end,
                    usage.slice.source_checksum,
                    usage.slice.calibration.value(),
                    static_cast<std::uint8_t>(usage.role),
                    usage.consumer.value(),
                    usage.factor_group.has_value(),
                    usage.factor_group ? usage.factor_group->value() : 0U,
                    usage.correlation_group.has_value(),
                    usage.correlation_group ? usage.correlation_group->value() : 0U};
}

[[nodiscard]] auto declarationCanonicalKey(const core::CorrelationDeclaration& declaration) {
  return std::tuple{declaration.group.value(), declaration.policy.value(),
                    static_cast<std::uint8_t>(declaration.treatment),
                    declaration.covariance_inflation,
                    declaration.total_information_cap.has_value(),
                    declaration.total_information_cap.value_or(0.0)};
}

void canonicalizeLineage(core::ObservationLineage* lineage) {
  std::sort(lineage->usage.begin(), lineage->usage.end(), [](const auto& lhs, const auto& rhs) {
    return usageCanonicalKey(lhs) < usageCanonicalKey(rhs);
  });
  std::sort(lineage->correlations.begin(), lineage->correlations.end(),
            [](const auto& lhs, const auto& rhs) {
              return declarationCanonicalKey(lhs) < declarationCanonicalKey(rhs);
            });
}

[[nodiscard]] core::Result<core::ObservationLineage, RollingLidarTargetError> mergedTargetLineage(
    const std::vector<const RetainedSweep*>& targets, core::ObservationLineageId output_id) {
  using Result = core::Result<core::ObservationLineage, RollingLidarTargetError>;
  if (!output_id.valid()) {
    return Result::failure(targetError(RollingLidarTargetErrorCode::InvalidIdentity,
                                       "target batch lineage identity is invalid"));
  }
  core::ObservationLineage merged;
  merged.id = output_id;
  const auto append =
      [&merged](const core::ObservationLineage& lineage) -> std::optional<RollingLidarTargetError> {
    for (const core::ObservationUsage& usage : lineage.usage) {
      const auto duplicate = std::find_if(merged.usage.begin(), merged.usage.end(),
                                          [&usage](const core::ObservationUsage& candidate) {
                                            return sameUsage(candidate, usage);
                                          });
      if (duplicate == merged.usage.end()) {
        merged.usage.push_back(usage);
      }
    }
    for (const core::CorrelationDeclaration& declaration : lineage.correlations) {
      const auto same_group =
          std::find_if(merged.correlations.begin(), merged.correlations.end(),
                       [&declaration](const core::CorrelationDeclaration& candidate) {
                         return candidate.group == declaration.group;
                       });
      if (same_group == merged.correlations.end()) {
        merged.correlations.push_back(declaration);
      } else if (!sameDeclaration(*same_group, declaration)) {
        return targetError(RollingLidarTargetErrorCode::InvalidLineage,
                           "selected sweeps disagree on a correlation declaration");
      }
    }
    return std::nullopt;
  };

  for (const RetainedSweep* target : targets) {
    if (target == nullptr) {
      return Result::failure(targetError(RollingLidarTargetErrorCode::InvalidIdentity,
                                         "selected target sweep is unavailable"));
    }
    if (auto error = append(target->cloud->lineage)) {
      return Result::failure(std::move(*error));
    }
  }
  canonicalizeLineage(&merged);
  if (core::validateLineage(merged) != core::LineageValidationError::None) {
    return Result::failure(targetError(RollingLidarTargetErrorCode::InvalidLineage,
                                       "target batch transitive lineage is invalid"));
  }
  auto checksum = core::recomputeObservationLineageChecksum(merged);
  if (!checksum) {
    // See sealedLineage(): target geometry and target selection are still
    // integrity-covered by the batch checksum while raw-ingress digests are
    // unavailable.
    return Result::success(std::move(merged));
  }
  merged.checksum = checksum.value();
  return Result::success(std::move(merged));
}

[[nodiscard]] bool writeCompleteLineage(core::CanonicalEncoder& encoder,
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
        !write(encoder.writeOptionalMarker(
            core::contentHashPresent(usage.slice.source_checksum))) ||
        (core::contentHashPresent(usage.slice.source_checksum) &&
         !write(encoder.writeHash(usage.slice.source_checksum))) ||
        !write(encoder.writeU64(usage.slice.calibration.value())) ||
        !write(encoder.writeU8(static_cast<std::uint8_t>(usage.role))) ||
        !write(encoder.writeU64(usage.consumer.value())) ||
        !write(encoder.writeOptionalMarker(usage.factor_group.has_value())) ||
        (usage.factor_group && !write(encoder.writeU64(usage.factor_group->value()))) ||
        !write(encoder.writeOptionalMarker(usage.correlation_group.has_value())) ||
        (usage.correlation_group &&
         !write(encoder.writeU64(usage.correlation_group->value())))) {
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

[[nodiscard]] bool addChecksumCapacity(std::uint64_t count, std::uint64_t bytes_per_record,
                                       std::uint64_t* capacity) noexcept {
  if (count > (std::numeric_limits<std::uint64_t>::max() - *capacity) / bytes_per_record) {
    return false;
  }
  *capacity += count * bytes_per_record;
  return true;
}

[[nodiscard]] core::Result<core::ContentHash, RollingLidarTargetError> batchChecksum(
    const RollingLidarTargetBatch& batch) {
  using Result = core::Result<core::ContentHash, RollingLidarTargetError>;
  constexpr std::uint64_t kFixedBytes = 2048U;
  constexpr std::uint64_t kBytesPerTarget = 512U;
  constexpr std::uint64_t kBytesPerLineageUsage = 256U;
  constexpr std::uint64_t kBytesPerCorrelation = 128U;
  const std::uint64_t count = static_cast<std::uint64_t>(batch.targets.size());
  if (!batch.odom_epoch.valid() || !batch.source_sweep.valid() ||
      !finitePose(batch.T_odom_imu_source_seed) ||
      !core::contentHashPresent(batch.source_cloud_checksum) || !batch.lineage.id.valid() ||
      batch.lineage.usage.empty() ||
      core::validateLineage(batch.lineage) != core::LineageValidationError::None ||
      !std::is_sorted(batch.lineage.usage.begin(), batch.lineage.usage.end(),
                      [](const auto& lhs, const auto& rhs) {
                        return usageCanonicalKey(lhs) < usageCanonicalKey(rhs);
                      }) ||
      !std::is_sorted(batch.lineage.correlations.begin(), batch.lineage.correlations.end(),
                      [](const auto& lhs, const auto& rhs) {
                        return declarationCanonicalKey(lhs) < declarationCanonicalKey(rhs);
                      }) ||
      batch.targets.empty()) {
    return Result::failure(targetError(RollingLidarTargetErrorCode::ChecksumFailure,
                                       "target batch is not canonical or checksumable"));
  }
  if (core::contentHashPresent(batch.lineage.checksum)) {
    const auto canonical_lineage_checksum =
        core::recomputeObservationLineageChecksum(batch.lineage);
    if (!canonical_lineage_checksum ||
        canonical_lineage_checksum.value() != batch.lineage.checksum) {
      return Result::failure(targetError(
          RollingLidarTargetErrorCode::ChecksumFailure,
          "target batch carries a stale canonical lineage checksum"));
    }
  }
  std::uint64_t capacity = kFixedBytes;
  if (!addChecksumCapacity(count, kBytesPerTarget, &capacity) ||
      !addChecksumCapacity(static_cast<std::uint64_t>(batch.lineage.usage.size()),
                           kBytesPerLineageUsage, &capacity) ||
      !addChecksumCapacity(static_cast<std::uint64_t>(batch.lineage.correlations.size()),
                           kBytesPerCorrelation, &capacity)) {
    return Result::failure(targetError(RollingLidarTargetErrorCode::ChecksumFailure,
                                       "target batch is too large to checksum"));
  }
  auto encoder = core::CanonicalEncoder::create(kRollingLidarTargetBatchChecksumDomain,
                                                kRollingLidarTargetBatchChecksumSchemaVersion,
                                                capacity);
  if (!encoder) {
    return Result::failure(targetError(RollingLidarTargetErrorCode::ChecksumFailure,
                                       "target batch checksum encoder initialization failed"));
  }
  const auto write = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!write(encoder.value().writeU64(batch.odom_epoch.value())) ||
      !write(encoder.value().writeU64(batch.source_sweep.value())) ||
      !write(encoder.value().writeI64(batch.source_reference_time.nanoseconds)) ||
      !write(encoder.value().writePose3(batch.T_odom_imu_source_seed)) ||
      !write(encoder.value().writeHash(batch.source_cloud_checksum)) ||
      !writeCompleteLineage(encoder.value(), batch.lineage) ||
      !write(encoder.value().writeU64(count))) {
    return Result::failure(targetError(RollingLidarTargetErrorCode::ChecksumFailure,
                                       "target batch checksum header encoding failed"));
  }
  for (std::size_t index = 0U; index < batch.targets.size(); ++index) {
    const LidarRegistrationTarget& target = batch.targets[index];
    if (!target.cloud || !core::contentHashPresent(target.cloud->checksum) ||
        !write(encoder.value().writeU64(target.state.value())) ||
        !write(encoder.value().writeI64(target.time.nanoseconds)) ||
        !write(encoder.value().writeU64(target.cloud->source_sweep.value())) ||
        !write(encoder.value().writeHash(target.cloud->checksum)) ||
        !write(encoder.value().writePose3(target.T_odom_imu_target_seed)) ||
        !write(encoder.value().writePose3(target.T_target_source_seed))) {
      return Result::failure(targetError(RollingLidarTargetErrorCode::ChecksumFailure,
                                         "target batch checksum record encoding failed"));
    }
  }
  auto encoded = encoder.value().finish();
  if (!encoded) {
    return Result::failure(targetError(RollingLidarTargetErrorCode::ChecksumFailure,
                                       "target batch checksum finalization failed"));
  }
  return Result::success(encoded.value().digest());
}

}  // namespace

core::Result<core::ContentHash, RollingLidarTargetError>
recomputeRollingLidarTargetBatchChecksum(const RollingLidarTargetBatch& batch) {
  return batchChecksum(batch);
}

struct RollingLidarTargetBuilder::Impl {
  explicit Impl(RollingLidarTargetConfig input_config) : config(std::move(input_config)) {}

  RollingLidarTargetConfig config;
  std::deque<RetainedSweep> retained;
  std::size_t retained_points{};
  std::optional<core::StateId> last_state;
  std::optional<core::MeasurementId> last_sweep;
  std::optional<core::FusionTime> last_time;
  std::optional<core::FactorBatchId> last_batch_id;
  std::optional<core::SensorRecoveryEpoch> last_recovery_epoch;
  RollingLidarTargetStatistics statistics;
};

core::Result<RollingLidarTargetBuilder, RollingLidarTargetError> RollingLidarTargetBuilder::create(
    RollingLidarTargetConfig config) {
  using Result = core::Result<RollingLidarTargetBuilder, RollingLidarTargetError>;
  if (!validConfig(config)) {
    return Result::failure(
        targetError(RollingLidarTargetErrorCode::InvalidConfig,
                    "rolling point target bounds or direct point ICP profile is invalid"));
  }
  return Result::success(RollingLidarTargetBuilder(std::make_unique<Impl>(std::move(config))));
}

RollingLidarTargetBuilder::RollingLidarTargetBuilder(std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

RollingLidarTargetBuilder::~RollingLidarTargetBuilder() = default;
RollingLidarTargetBuilder::RollingLidarTargetBuilder(RollingLidarTargetBuilder&&) noexcept =
    default;
RollingLidarTargetBuilder& RollingLidarTargetBuilder::operator=(
    RollingLidarTargetBuilder&&) noexcept = default;

core::Result<RollingLidarTargetAddStats, RollingLidarTargetError> RollingLidarTargetBuilder::add(
    RegisteredLidarSweep sweep) {
  using Result = core::Result<RollingLidarTargetAddStats, RollingLidarTargetError>;
  ++implementation_->statistics.add_attempts;
  const auto reject = [this](RollingLidarTargetErrorCode code, std::string detail) {
    ++implementation_->statistics.rejected_adds;
    return Result::failure(targetError(code, std::move(detail)));
  };

  if (sweep.odom_epoch != implementation_->config.odom_epoch) {
    return reject(RollingLidarTargetErrorCode::EpochMismatch,
                  "registered LiDAR sweep belongs to another odometry epoch");
  }
  if (!sweep.odom_epoch.valid() || !sweep.state.valid() || !sweep.admitting_batch_id.valid() ||
      !sweep.recovery_epoch.valid() || !sweep.cloud || !sweep.cloud->source_sweep.valid()) {
    return reject(RollingLidarTargetErrorCode::InvalidIdentity,
                  "registered LiDAR sweep has an invalid identity");
  }
  if (!finitePose(sweep.T_odom_imu)) {
    return reject(RollingLidarTargetErrorCode::InvalidPose,
                  "registered LiDAR sweep pose is non-finite");
  }
  if (sweep.cloud->exactIndexVoxelResolutionM() !=
      implementation_->config.registration.target_voxel_resolution_m) {
    return reject(RollingLidarTargetErrorCode::InvalidCloud,
                  "registered LiDAR scan exact index does not match the registration profile");
  }

  if (implementation_->last_state) {
    if (sweep.state == *implementation_->last_state) {
      return reject(RollingLidarTargetErrorCode::DuplicateState,
                    "state identity was already admitted");
    }
    if (sweep.cloud->source_sweep == *implementation_->last_sweep) {
      return reject(RollingLidarTargetErrorCode::DuplicateSweep,
                    "source sweep identity was already admitted");
    }
    if (sweep.admitting_batch_id == *implementation_->last_batch_id) {
      return reject(RollingLidarTargetErrorCode::DuplicateFactorBatch,
                    "FactorBatch identity was already admitted");
    }
    if (sweep.state < *implementation_->last_state) {
      return reject(RollingLidarTargetErrorCode::NonMonotonicState,
                    "state identities must increase strictly");
    }
    if (sweep.cloud->source_sweep < *implementation_->last_sweep) {
      return reject(RollingLidarTargetErrorCode::NonMonotonicSweep,
                    "source sweep identities must increase strictly");
    }
    if (sweep.cloud->reference_time <= *implementation_->last_time) {
      return reject(RollingLidarTargetErrorCode::NonMonotonicTime,
                    "sweep reference times must increase strictly");
    }
    if (sweep.admitting_batch_id < *implementation_->last_batch_id) {
      return reject(RollingLidarTargetErrorCode::NonMonotonicFactorBatch,
                    "admitting FactorBatch identities must increase strictly");
    }
    if (sweep.recovery_epoch < *implementation_->last_recovery_epoch) {
      return reject(RollingLidarTargetErrorCode::NonMonotonicRecoveryEpoch,
                    "sensor recovery epochs cannot move backwards");
    }
  }

  const std::size_t per_sweep_capacity =
      std::min(implementation_->config.maximum_retained_points,
               implementation_->config.registration.maximum_target_points_per_target);
  if (sweep.cloud->points.size() > per_sweep_capacity) {
    return reject(RollingLidarTargetErrorCode::TargetCapacity,
                  "sealed LiDAR scan exceeds configured target capacity");
  }
  RollingLidarTargetAddStats result;
  result.input_points = sweep.cloud->points.size();
  result.retained_points_from_input = sweep.cloud->points.size();

  RetainedSweep retained{sweep.state,
                         sweep.cloud->reference_time,
                         sweep.admitting_batch_id,
                         sweep.recovery_epoch,
                         sweep.T_odom_imu,
                         std::move(sweep.cloud)};
  implementation_->retained_points += retained.cloud->points.size();
  implementation_->retained.push_back(std::move(retained));
  while (implementation_->retained.size() > implementation_->config.maximum_retained_sweeps ||
         implementation_->retained_points > implementation_->config.maximum_retained_points) {
    result.eviction.points += implementation_->retained.front().cloud->points.size();
    implementation_->retained_points -= implementation_->retained.front().cloud->points.size();
    implementation_->retained.pop_front();
    ++result.eviction.sweeps;
  }
  implementation_->last_state = sweep.state;
  implementation_->last_sweep = implementation_->retained.back().cloud->source_sweep;
  implementation_->last_time = implementation_->retained.back().reference_time;
  implementation_->last_batch_id = sweep.admitting_batch_id;
  implementation_->last_recovery_epoch = sweep.recovery_epoch;
  result.retained_sweeps = implementation_->retained.size();
  result.retained_points = implementation_->retained_points;

  RollingLidarTargetStatistics& totals = implementation_->statistics;
  ++totals.accepted_sweeps;
  totals.input_points += result.input_points;
  totals.retained_points_from_input += result.retained_points_from_input;
  totals.copied_raw_points += result.copied_raw_points;
  totals.deterministic_cap_discarded_points += result.deterministic_cap_discarded_points;
  totals.eviction.sweeps += result.eviction.sweeps;
  totals.eviction.points += result.eviction.points;
  totals.retained_sweeps = result.retained_sweeps;
  totals.retained_points = result.retained_points;
  return Result::success(std::move(result));
}

core::Result<RollingLidarTargetRemovalStats, RollingLidarTargetError>
RollingLidarTargetBuilder::removeFactorBatches(
    core::OdomEpoch odom_epoch, std::span<const core::FactorBatchId> batch_ids) {
  using Result = core::Result<RollingLidarTargetRemovalStats, RollingLidarTargetError>;
  ++implementation_->statistics.batch_removal_attempts;
  const auto reject = [this](RollingLidarTargetErrorCode code, std::string detail) {
    ++implementation_->statistics.rejected_batch_removals;
    return Result::failure(targetError(code, std::move(detail)));
  };
  if (odom_epoch != implementation_->config.odom_epoch) {
    return reject(RollingLidarTargetErrorCode::EpochMismatch,
                  "FactorBatch removal belongs to another odometry epoch");
  }
  if (!odom_epoch.valid()) {
    return reject(RollingLidarTargetErrorCode::InvalidIdentity,
                  "FactorBatch removal odometry epoch is invalid");
  }
  if (batch_ids.size() > implementation_->config.maximum_retained_sweeps) {
    return reject(RollingLidarTargetErrorCode::RemovalCapacity,
                  "FactorBatch removal exceeds the bounded retained-sweep window");
  }

  std::vector<core::FactorBatchId> canonical_ids(batch_ids.begin(), batch_ids.end());
  for (const core::FactorBatchId batch_id : canonical_ids) {
    if (!batch_id.valid()) {
      return reject(RollingLidarTargetErrorCode::InvalidIdentity,
                    "FactorBatch removal contains an invalid batch identity");
    }
  }
  std::sort(canonical_ids.begin(), canonical_ids.end());
  if (std::adjacent_find(canonical_ids.begin(), canonical_ids.end()) != canonical_ids.end()) {
    return reject(RollingLidarTargetErrorCode::DuplicateFactorBatch,
                  "FactorBatch removal contains a duplicate batch identity");
  }

  // Stage the complete retained window before publication. Shared immutable
  // Immutable point payloads make this bounded copy cheap and guarantee that every
  // validation failure above leaves both geometry and accounting untouched.
  std::deque<RetainedSweep> staged;
  std::size_t staged_points = implementation_->retained_points;
  RollingLidarTargetRemovalStats result;
  result.requested_batches = canonical_ids.size();
  for (const RetainedSweep& sweep : implementation_->retained) {
    if (!std::binary_search(canonical_ids.begin(), canonical_ids.end(),
                            sweep.admitting_batch_id)) {
      staged.push_back(sweep);
      continue;
    }
    ++result.matched_batches;
    ++result.removed_sweeps;
    result.removed_points += sweep.cloud->points.size();
    staged_points -= sweep.cloud->points.size();
  }
  result.absent_batches = result.requested_batches - result.matched_batches;
  result.retained_sweeps = staged.size();
  result.retained_points = staged_points;

  implementation_->retained = std::move(staged);
  implementation_->retained_points = staged_points;
  RollingLidarTargetStatistics& totals = implementation_->statistics;
  ++totals.batch_removal_transactions;
  totals.removed_sweeps += result.removed_sweeps;
  totals.removed_points += result.removed_points;
  totals.retained_sweeps = result.retained_sweeps;
  totals.retained_points = result.retained_points;
  // Deliberately preserve every last_* marker: a rollback removes retained
  // registration targets, not the already-observed identity/time frontier.
  return Result::success(std::move(result));
}

core::Result<RollingLidarTargetPoseSynchronizationStats, RollingLidarTargetError>
RollingLidarTargetBuilder::synchronizeCommittedPoses(
    core::OdomEpoch odom_epoch, std::span<const RollingLidarTargetPose> poses,
    std::span<const core::StateId> finalized_states) {
  using Result = core::Result<RollingLidarTargetPoseSynchronizationStats, RollingLidarTargetError>;
  ++implementation_->statistics.pose_synchronization_attempts;
  const auto reject = [this](RollingLidarTargetErrorCode code, std::string detail) {
    ++implementation_->statistics.rejected_pose_synchronizations;
    return Result::failure(targetError(code, std::move(detail)));
  };
  if (odom_epoch != implementation_->config.odom_epoch) {
    return reject(RollingLidarTargetErrorCode::EpochMismatch,
                  "committed pose snapshot belongs to another odometry epoch");
  }
  for (std::size_t index = 0U; index < poses.size(); ++index) {
    if (!poses[index].state.valid()) {
      return reject(RollingLidarTargetErrorCode::InvalidIdentity,
                    "committed pose snapshot contains an invalid state identity");
    }
    if (!finitePose(poses[index].T_odom_imu)) {
      return reject(RollingLidarTargetErrorCode::InvalidPose,
                    "committed pose snapshot contains a non-finite pose");
    }
    if (index > 0U && poses[index].state == poses[index - 1U].state) {
      return reject(RollingLidarTargetErrorCode::DuplicateState,
                    "committed pose snapshot contains a duplicate state");
    }
    if (index > 0U && poses[index].state < poses[index - 1U].state) {
      return reject(RollingLidarTargetErrorCode::NonMonotonicState,
                    "committed pose snapshot states must increase strictly");
    }
  }
  for (std::size_t index = 0U; index < finalized_states.size(); ++index) {
    if (!finalized_states[index].valid()) {
      return reject(RollingLidarTargetErrorCode::InvalidIdentity,
                    "committed finality list contains an invalid state identity");
    }
    if (index > 0U && finalized_states[index] == finalized_states[index - 1U]) {
      return reject(RollingLidarTargetErrorCode::DuplicateState,
                    "committed finality list contains a duplicate state");
    }
    if (index > 0U && finalized_states[index] < finalized_states[index - 1U]) {
      return reject(RollingLidarTargetErrorCode::NonMonotonicState,
                    "committed finality list states must increase strictly");
    }
  }

  std::deque<RetainedSweep> staged;
  std::size_t staged_points = 0U;
  RollingLidarTargetPoseSynchronizationStats result;
  result.supplied_poses = poses.size();
  std::size_t pose_index = 0U;
  std::size_t finalized_index = 0U;
  for (const RetainedSweep& sweep : implementation_->retained) {
    // navigation_poses deliberately includes the pre-marginalization pose of
    // every state finalized by this same commit. Explicit finality therefore
    // takes precedence over a matching pose snapshot.
    while (finalized_index < finalized_states.size() &&
           finalized_states[finalized_index] < sweep.state) {
      ++finalized_index;
    }
    if (finalized_index < finalized_states.size() &&
        finalized_states[finalized_index] == sweep.state) {
      ++result.finalized_sweeps_evicted;
      result.finalized_points_evicted += sweep.cloud->points.size();
      continue;
    }
    while (pose_index < poses.size() && poses[pose_index].state < sweep.state) {
      ++pose_index;
    }
    if (pose_index == poses.size() || poses[pose_index].state != sweep.state) {
      return reject(RollingLidarTargetErrorCode::MissingCommittedPose,
                    "retained non-final LiDAR sweep is absent from the committed pose snapshot");
    }
    RetainedSweep revised = sweep;
    const core::Pose3d revision = revised.T_odom_imu.inverse() * poses[pose_index].T_odom_imu;
    const double translation_revision = revision.translation().norm();
    const double rotation_revision = revision.so3().log().norm();
    result.maximum_translation_revision_m =
        std::max(result.maximum_translation_revision_m, translation_revision);
    result.maximum_rotation_revision_rad =
        std::max(result.maximum_rotation_revision_rad, rotation_revision);
    ++result.matched_retained_sweeps;
    revised.T_odom_imu = poses[pose_index].T_odom_imu;
    staged_points += revised.cloud->points.size();
    staged.push_back(std::move(revised));
    ++pose_index;
  }

  implementation_->retained = std::move(staged);
  implementation_->retained_points = staged_points;
  result.retained_sweeps = implementation_->retained.size();
  result.retained_points = implementation_->retained_points;
  RollingLidarTargetStatistics& totals = implementation_->statistics;
  totals.pose_synchronized_sweeps += result.matched_retained_sweeps;
  totals.finalized_sweeps_evicted += result.finalized_sweeps_evicted;
  totals.finalized_points_evicted += result.finalized_points_evicted;
  totals.retained_sweeps = result.retained_sweeps;
  totals.retained_points = result.retained_points;
  return Result::success(std::move(result));
}

core::Result<RollingLidarTargetBatch, RollingLidarTargetError>
RollingLidarTargetBuilder::buildBatch(std::shared_ptr<const LidarRegistrationCloud> source,
                                      core::ObservationLineageId output_lineage,
                                      std::optional<std::size_t> target_limit) {
  using Result = core::Result<RollingLidarTargetBatch, RollingLidarTargetError>;
  ++implementation_->statistics.batch_build_attempts;
  const auto reject = [this](RollingLidarTargetErrorCode code, std::string detail) {
    ++implementation_->statistics.rejected_batch_builds;
    return Result::failure(targetError(code, std::move(detail)));
  };
  if (implementation_->retained.empty()) {
    return reject(RollingLidarTargetErrorCode::EmptyTarget,
                  "no localization-accepted LiDAR sweep is retained");
  }
  if (!source || !source->source_sweep.valid()) {
    return reject(RollingLidarTargetErrorCode::InvalidIdentity,
                  "target batch source identity is invalid");
  }
  if (source->exactIndexVoxelResolutionM() !=
      implementation_->config.registration.target_voxel_resolution_m) {
    return reject(RollingLidarTargetErrorCode::InvalidCloud,
                  "target batch source exact index does not match the registration profile");
  }
  if (!output_lineage.valid()) {
    return reject(RollingLidarTargetErrorCode::InvalidIdentity,
                  "target batch lineage identity is invalid");
  }
  if (target_limit &&
      (*target_limit == 0U ||
       *target_limit > implementation_->config.registration.maximum_targets)) {
    return reject(RollingLidarTargetErrorCode::TargetCapacity,
                  "scan-local rolling-target limit must be positive and no larger than the "
                  "configured registration target capacity");
  }
  if (std::any_of(implementation_->retained.begin(), implementation_->retained.end(),
                  [&source](const RetainedSweep& retained) {
                    return retained.cloud.get() == source.get() ||
                           retained.cloud->source_sweep == source->source_sweep;
                  })) {
    return reject(RollingLidarTargetErrorCode::SourceAlreadyRetained,
                  "direct point ICP source sweep is already retained as a target");
  }
  if (source->source_sweep == *implementation_->last_sweep) {
    return reject(RollingLidarTargetErrorCode::DuplicateSweep,
                  "direct point ICP source sweep identity was already admitted");
  }
  if (source->source_sweep < *implementation_->last_sweep) {
    return reject(RollingLidarTargetErrorCode::NonMonotonicSweep,
                  "direct point ICP source sweep identity predates the accepted rolling target");
  }
  if (source->reference_time <= *implementation_->last_time) {
    return reject(RollingLidarTargetErrorCode::NonMonotonicTime,
                  "direct point ICP source reference time must follow every retained target");
  }

  RollingLidarTargetBatchBuildStats build;
  build.retained_candidates = implementation_->retained.size();
  std::vector<const RetainedSweep*> selected_sweeps;
  const std::size_t target_capacity =
      target_limit.value_or(implementation_->config.registration.maximum_targets);
  const std::size_t selected_targets =
      std::min(target_capacity, implementation_->retained.size());
  selected_sweeps.reserve(selected_targets);
  if (selected_targets == 1U) {
    selected_sweeps.push_back(&implementation_->retained.back());
  } else {
    const std::size_t history_span = implementation_->retained.size() - 1U;
    const std::size_t selection_span = selected_targets - 1U;
    for (std::size_t slot = 0U; slot < selected_targets; ++slot) {
      const std::size_t rounded_age =
          (slot * history_span + selection_span / 2U) / selection_span;
      const std::size_t retained_index = history_span - rounded_age;
      selected_sweeps.push_back(&implementation_->retained[retained_index]);
    }
  }
  auto lineage = mergedTargetLineage(selected_sweeps, output_lineage);
  if (!lineage) {
    return reject(lineage.error().code, lineage.error().detail);
  }
  RollingLidarTargetBatch result;
  result.odom_epoch = implementation_->config.odom_epoch;
  result.source_sweep = source->source_sweep;
  result.source_reference_time = source->reference_time;
  result.T_odom_imu_source_seed = source->T_odom_imu_seed;
  result.source_cloud_checksum = source->checksum;
  result.lineage = std::move(lineage).value();
  result.targets.reserve(selected_sweeps.size());
  for (const RetainedSweep* target : selected_sweeps) {
    result.targets.push_back(LidarRegistrationTarget{
        target->state, target->reference_time, target->cloud, target->T_odom_imu,
        target->T_odom_imu.inverse() * source->T_odom_imu_seed});
  }
  build.selected_targets = selected_targets;
  build.selected_state_span = selected_sweeps.front()->state.value() -
                              selected_sweeps.back()->state.value();
  build.selected_time_span_ns = selected_sweeps.front()->reference_time.nanoseconds -
                                selected_sweeps.back()->reference_time.nanoseconds;
  result.build = build;
  auto checksum = batchChecksum(result);
  if (!checksum) {
    return reject(checksum.error().code, checksum.error().detail);
  }
  result.checksum = checksum.value();

  ++implementation_->statistics.built_target_batches;
  implementation_->statistics.batch_selected_targets += selected_targets;
  return Result::success(std::move(result));
}

std::vector<RollingLidarTargetSweep> RollingLidarTargetBuilder::retainedSweeps() const {
  std::vector<RollingLidarTargetSweep> output;
  output.reserve(implementation_->retained.size());
  for (const RetainedSweep& sweep : implementation_->retained) {
    output.push_back(RollingLidarTargetSweep{
        sweep.state, sweep.reference_time, sweep.admitting_batch_id, sweep.recovery_epoch,
        sweep.T_odom_imu, sweep.cloud, sweep.cloud->lineage, sweep.cloud->checksum});
  }
  return output;
}

const RollingLidarTargetStatistics& RollingLidarTargetBuilder::statistics() const noexcept {
  return implementation_->statistics;
}

bool RollingLidarTargetBuilder::empty() const noexcept {
  return implementation_->retained.empty();
}

}  // namespace meridian::local
