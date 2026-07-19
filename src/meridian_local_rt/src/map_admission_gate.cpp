#include "meridian/local/map_admission_gate.hpp"

#include <algorithm>
#include <utility>

namespace meridian::local {
namespace {

[[nodiscard]] MapAdmissionDecision decision(const core::FactorBatchMetadata& metadata,
                                            const MapAdmissionContext& context,
                                            MapAdmissionDisposition disposition,
                                            MapAdmissionReason reason) noexcept {
  return MapAdmissionDecision{
      MapAdmissionBatchRef{metadata.odom_epoch, metadata.sensor, metadata.batch_id},
      context.localization_revision, disposition, reason};
}

[[nodiscard]] bool sameHealthSnapshot(const core::SensorHealthSnapshot& left,
                                      const core::SensorHealthSnapshot& right) noexcept {
  return left.sensor == right.sensor && left.state == right.state &&
         left.recovery_epoch == right.recovery_epoch &&
         left.transition_sequence == right.transition_sequence &&
         left.assessed_at == right.assessed_at;
}

[[nodiscard]] bool validKind(MapAdmissionBatchKind kind) noexcept {
  switch (kind) {
    case MapAdmissionBatchKind::Regular:
    case MapAdmissionBatchKind::InitializationSeed:
      return true;
  }
  return false;
}

}  // namespace

MapAdmissionGate::MapAdmissionGate(MapAdmissionGateConfig config) : config_(config) {}

MapAdmissionDecision MapAdmissionGate::evaluate(const core::FactorBatchMetadata& metadata,
                                                const MapAdmissionContext& context) const noexcept {
  if (config_.maximum_recent_admitted_batches == 0U) {
    return decision(metadata, context, MapAdmissionDisposition::Rejected,
                    MapAdmissionReason::InvalidGateConfig);
  }
  if (!context.accepted_batch_revision) {
    return decision(metadata, context, MapAdmissionDisposition::Rejected,
                    MapAdmissionReason::BatchRejected);
  }
  if (!validKind(context.kind) || core::validateFactorBatchMetadata(metadata) !=
                                      core::FactorBatchMetadataValidationError::None) {
    return decision(metadata, context, MapAdmissionDisposition::Rejected,
                    MapAdmissionReason::InvalidMetadata);
  }
  if (core::validateSensorHealthSnapshot(context.health_before) !=
          core::SensorHealthValidationError::None ||
      core::validateSensorHealthSnapshot(context.health_after) !=
          core::SensorHealthValidationError::None) {
    return decision(metadata, context, MapAdmissionDisposition::Rejected,
                    MapAdmissionReason::InvalidHealthSnapshot);
  }
  if (!context.localization_revision.valid()) {
    return decision(metadata, context, MapAdmissionDisposition::Rejected,
                    MapAdmissionReason::InvalidLocalizationRevision);
  }
  if (*context.accepted_batch_revision != context.localization_revision) {
    return decision(metadata, context, MapAdmissionDisposition::Rejected,
                    MapAdmissionReason::AcceptedRevisionMismatch);
  }
  const MapAdmissionBatchRef batch_ref{metadata.odom_epoch, metadata.sensor, metadata.batch_id};
  if (contains(batch_ref)) {
    return decision(metadata, context, MapAdmissionDisposition::Rejected,
                    MapAdmissionReason::DuplicateBatch);
  }
  if (context.health_before.sensor != metadata.sensor ||
      context.health_after.sensor != metadata.sensor) {
    return decision(metadata, context, MapAdmissionDisposition::Rejected,
                    MapAdmissionReason::SensorMismatch);
  }
  if (context.health_before.recovery_epoch != metadata.health.recovery_epoch ||
      context.health_after.recovery_epoch != metadata.health.recovery_epoch) {
    return decision(metadata, context, MapAdmissionDisposition::Rejected,
                    MapAdmissionReason::RecoveryEpochMismatch);
  }
  if (!sameHealthSnapshot(context.health_before, metadata.health) ||
      context.health_after.transition_sequence < context.health_before.transition_sequence ||
      context.health_after.assessed_at < context.health_before.assessed_at) {
    return decision(metadata, context, MapAdmissionDisposition::Rejected,
                    MapAdmissionReason::HealthSnapshotMismatch);
  }
  if (!metadata.map_eligible) {
    return decision(metadata, context, MapAdmissionDisposition::Frozen,
                    MapAdmissionReason::MapIneligible);
  }
  if (context.health_before.state != core::SensorHealthState::Active ||
      context.health_after.state != core::SensorHealthState::Active) {
    return decision(metadata, context, MapAdmissionDisposition::Quarantined,
                    MapAdmissionReason::HealthNotActive);
  }
  if (context.kind == MapAdmissionBatchKind::InitializationSeed) {
    return decision(metadata, context, MapAdmissionDisposition::AcceptedInitializationSeed,
                    MapAdmissionReason::InitializationSeed);
  }
  return decision(metadata, context, MapAdmissionDisposition::Admitted,
                  MapAdmissionReason::Eligible);
}

MapAdmissionDecision MapAdmissionGate::admit(const core::FactorBatchMetadata& metadata,
                                             const MapAdmissionContext& context) {
  MapAdmissionDecision output = evaluate(metadata, context);
  if (!output.admitted()) {
    return output;
  }
  const MapAdmissionBatchRef batch_ref{metadata.odom_epoch, metadata.sensor, metadata.batch_id};
  const auto [unused, inserted] = admitted_batches_.insert(batch_ref);
  (void)unused;
  if (!inserted) {
    output.disposition = MapAdmissionDisposition::Rejected;
    output.reason = MapAdmissionReason::DuplicateBatch;
    return output;
  }
  try {
    admission_order_.push_back(batch_ref);
  } catch (...) {
    admitted_batches_.erase(batch_ref);
    throw;
  }
  ++total_admissions_;
  if (admission_order_.size() > config_.maximum_recent_admitted_batches) {
    const MapAdmissionBatchRef oldest = admission_order_.front();
    admission_order_.pop_front();
    admitted_batches_.erase(oldest);
    ++evicted_batches_;
  }
  history_high_water_mark_ = std::max(history_high_water_mark_, admitted_batches_.size());
  return output;
}

bool MapAdmissionGate::contains(MapAdmissionBatchRef batch) const noexcept {
  return admitted_batches_.contains(batch);
}

MapAdmissionGateStatistics MapAdmissionGate::statistics() const noexcept {
  return MapAdmissionGateStatistics{admitted_batches_.size(), total_admissions_, evicted_batches_,
                                    history_high_water_mark_};
}

}  // namespace meridian::local
