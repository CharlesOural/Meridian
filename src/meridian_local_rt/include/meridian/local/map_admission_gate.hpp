#pragma once

#include <compare>
#include <cstddef>
#include <deque>
#include <optional>
#include <set>

#include "meridian/core/factor_batch_api.hpp"
#include "meridian/core/local_graph_api.hpp"

namespace meridian::local {

struct MapAdmissionGateConfig {
  // Duplicate protection is intentionally a recent-history promise. Long
  // missions evict the oldest admitted identity instead of growing without
  // bound.
  std::size_t maximum_recent_admitted_batches{4'096U};
};

struct MapAdmissionGateStatistics {
  std::size_t remembered_batches{};
  std::size_t total_admissions{};
  std::size_t evicted_batches{};
  std::size_t history_high_water_mark{};

  bool operator==(const MapAdmissionGateStatistics&) const = default;
};

enum class MapAdmissionBatchKind {
  Regular,
  InitializationSeed,
};

enum class MapAdmissionDisposition {
  Admitted,
  AcceptedInitializationSeed,
  Frozen,
  Quarantined,
  Rejected,
};

enum class MapAdmissionReason {
  Eligible,
  InitializationSeed,
  InvalidGateConfig,
  BatchRejected,
  AcceptedRevisionMismatch,
  InvalidMetadata,
  InvalidHealthSnapshot,
  InvalidLocalizationRevision,
  DuplicateBatch,
  MapIneligible,
  SensorMismatch,
  RecoveryEpochMismatch,
  HealthSnapshotMismatch,
  HealthNotActive,
};

struct MapAdmissionContext {
  // Populated only from the accepted graph transition that owns this exact
  // batch. A missing or different revision is not map authority.
  std::optional<core::LocalGraphRevision> accepted_batch_revision;
  MapAdmissionBatchKind kind{MapAdmissionBatchKind::Regular};
  core::SensorHealthSnapshot health_before;
  core::SensorHealthSnapshot health_after;
  core::LocalGraphRevision localization_revision;
};

struct MapAdmissionBatchRef {
  core::OdomEpoch odom_epoch;
  core::SensorInstanceId sensor;
  core::FactorBatchId batch_id;

  auto operator<=>(const MapAdmissionBatchRef&) const = default;
};

struct MapAdmissionDecision {
  MapAdmissionBatchRef batch;
  core::LocalGraphRevision localization_revision;
  MapAdmissionDisposition disposition{MapAdmissionDisposition::Rejected};
  MapAdmissionReason reason{MapAdmissionReason::InvalidMetadata};

  [[nodiscard]] bool admitted() const noexcept {
    return disposition == MapAdmissionDisposition::Admitted ||
           disposition == MapAdmissionDisposition::AcceptedInitializationSeed;
  }
};

// The decision function is pure with respect to both the map and admission
// history. admit() uses that one decision path and records only successful
// batches; every deny disposition leaves the admitted set byte-for-byte
// unchanged.
class MapAdmissionGate {
public:
  explicit MapAdmissionGate(MapAdmissionGateConfig config = {});

  [[nodiscard]] MapAdmissionDecision evaluate(const core::FactorBatchMetadata& metadata,
                                              const MapAdmissionContext& context) const noexcept;
  [[nodiscard]] MapAdmissionDecision admit(const core::FactorBatchMetadata& metadata,
                                           const MapAdmissionContext& context);

  [[nodiscard]] bool contains(MapAdmissionBatchRef batch) const noexcept;
  [[nodiscard]] std::size_t admittedBatchCount() const noexcept { return admitted_batches_.size(); }
  [[nodiscard]] std::size_t historyCapacity() const noexcept {
    return config_.maximum_recent_admitted_batches;
  }
  [[nodiscard]] MapAdmissionGateStatistics statistics() const noexcept;

private:
  MapAdmissionGateConfig config_;
  std::set<MapAdmissionBatchRef> admitted_batches_;
  std::deque<MapAdmissionBatchRef> admission_order_;
  std::size_t total_admissions_{};
  std::size_t evicted_batches_{};
  std::size_t history_high_water_mark_{};
};

}  // namespace meridian::local
