#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "meridian/core/blob.hpp"
#include "meridian/core/strong_id.hpp"

namespace meridian::core {

using RawObservationKey = std::variant<MeasurementId, GnssObservationId>;

enum class ObservationRole {
  PrimaryResidual,
  ConditioningOnly,
  RetrievalSeedOnly,
  DerivedSummary,
};

enum class SliceKind {
  Whole,
  IndexRange,
};

struct ObservationSlice {
  RawObservationKey root;
  SliceKind kind{SliceKind::Whole};
  std::uint64_t begin{};
  std::uint64_t end{};
  ContentHash source_checksum{};
  CalibrationEpoch calibration;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool overlaps(const ObservationSlice& other) const noexcept;
};

enum class CorrelationTreatment {
  JointCompositeWhitening,
  CovarianceInflationAndInformationCap,
  NotIndependent,
};

struct ObservationUsage {
  ObservationSlice slice;
  ObservationRole role{ObservationRole::DerivedSummary};
  DerivedRecordId consumer;
  std::optional<FactorGroupId> factor_group;
  std::optional<CorrelationGroupId> correlation_group;
};

struct CorrelationDeclaration {
  CorrelationGroupId group;
  CorrelationPolicyRevision policy;
  CorrelationTreatment treatment{CorrelationTreatment::NotIndependent};
  double covariance_inflation{1.0};
  std::optional<double> total_information_cap;
};

// Transitive ancestry and information-use bookkeeping for a derived record.
// The lineage prevents accidental double counting and makes correlation policy
// part of the public API rather than an optimizer-side convention.
struct ObservationLineage {
  ObservationLineageId id;
  std::vector<ObservationUsage> usage;
  std::vector<CorrelationDeclaration> correlations;
  ContentHash checksum{};
};

enum class LineageValidationError {
  None,
  InvalidSlice,
  InvalidRole,
  InvalidConsumer,
  MissingFactorGroup,
  InvalidFactorGroup,
  InvalidCorrelationGroup,
  MissingCorrelationDeclaration,
  InvalidCorrelationDeclaration,
  InvalidCorrelationTreatment,
  DuplicateCorrelationDeclaration,
  InvalidCorrelationInflation,
  InvalidInformationCap,
  DuplicatePrimaryObservation,
};

[[nodiscard]] LineageValidationError validateLineage(const ObservationLineage& lineage) noexcept;
[[nodiscard]] bool lineagesAreIndependent(const ObservationLineage& lhs,
                                          const ObservationLineage& rhs) noexcept;

}  // namespace meridian::core
