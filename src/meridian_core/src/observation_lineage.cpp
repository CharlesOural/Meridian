#include "meridian/core/observation_lineage.hpp"

#include <algorithm>
#include <cmath>

namespace meridian::core {

bool ObservationSlice::valid() const noexcept {
  const bool valid_root =
      !root.valueless_by_exception() && std::visit([](const auto& id) { return id.valid(); }, root);
  if (!valid_root || !calibration.valid()) {
    return false;
  }
  switch (kind) {
    case SliceKind::Whole:
      return begin == 0U && end == 0U;
    case SliceKind::IndexRange:
      return begin < end;
  }
  return false;
}

bool ObservationSlice::overlaps(const ObservationSlice& other) const noexcept {
  // Invalid or unknown selectors are conservatively overlapping. This keeps a
  // malformed ancestry record from becoming an independence proof.
  if (!valid() || !other.valid()) {
    return true;
  }
  if (root != other.root) {
    return false;
  }
  if (kind == SliceKind::Whole || other.kind == SliceKind::Whole) {
    return true;
  }
  return begin < other.end && other.begin < end;
}

namespace {

bool validRole(ObservationRole role) noexcept {
  switch (role) {
    case ObservationRole::PrimaryResidual:
    case ObservationRole::ConditioningOnly:
    case ObservationRole::RetrievalSeedOnly:
    case ObservationRole::DerivedSummary:
      return true;
  }
  return false;
}

bool validCorrelationTreatment(CorrelationTreatment treatment) noexcept {
  switch (treatment) {
    case CorrelationTreatment::JointCompositeWhitening:
    case CorrelationTreatment::CovarianceInflationAndInformationCap:
    case CorrelationTreatment::NotIndependent:
      return true;
  }
  return false;
}

const CorrelationDeclaration* findDeclaration(const ObservationLineage& lineage,
                                              CorrelationGroupId group) {
  const auto match = std::find_if(
      lineage.correlations.begin(), lineage.correlations.end(),
      [group](const CorrelationDeclaration& declaration) { return declaration.group == group; });
  return match == lineage.correlations.end() ? nullptr : &*match;
}

bool duplicateIsDeclared(const ObservationLineage& lineage, const ObservationUsage& lhs,
                         const ObservationUsage& rhs) {
  if (!lhs.factor_group || !rhs.factor_group || lhs.factor_group != rhs.factor_group ||
      !lhs.correlation_group || lhs.correlation_group != rhs.correlation_group) {
    return false;
  }
  const CorrelationDeclaration* declaration = findDeclaration(lineage, *lhs.correlation_group);
  return declaration != nullptr && declaration->treatment != CorrelationTreatment::NotIndependent;
}

}  // namespace

LineageValidationError validateLineage(const ObservationLineage& lineage) noexcept {
  for (std::size_t index = 0U; index < lineage.correlations.size(); ++index) {
    const CorrelationDeclaration& declaration = lineage.correlations[index];
    if (!declaration.group.valid() || !declaration.policy.valid()) {
      return LineageValidationError::InvalidCorrelationDeclaration;
    }
    if (!validCorrelationTreatment(declaration.treatment)) {
      return LineageValidationError::InvalidCorrelationTreatment;
    }
    if (!std::isfinite(declaration.covariance_inflation) ||
        declaration.covariance_inflation < 1.0) {
      return LineageValidationError::InvalidCorrelationInflation;
    }
    if (declaration.total_information_cap.has_value() &&
        (!std::isfinite(*declaration.total_information_cap) ||
         *declaration.total_information_cap <= 0.0)) {
      return LineageValidationError::InvalidInformationCap;
    }
    for (std::size_t other = index + 1U; other < lineage.correlations.size(); ++other) {
      if (declaration.group == lineage.correlations[other].group) {
        return LineageValidationError::DuplicateCorrelationDeclaration;
      }
    }
  }

  for (std::size_t index = 0U; index < lineage.usage.size(); ++index) {
    const ObservationUsage& usage = lineage.usage[index];
    if (!usage.slice.valid()) {
      return LineageValidationError::InvalidSlice;
    }
    if (!validRole(usage.role)) {
      return LineageValidationError::InvalidRole;
    }
    if (!usage.consumer.valid()) {
      return LineageValidationError::InvalidConsumer;
    }
    if (usage.role == ObservationRole::PrimaryResidual && !usage.factor_group) {
      return LineageValidationError::MissingFactorGroup;
    }
    if (usage.factor_group.has_value() && !usage.factor_group->valid()) {
      return LineageValidationError::InvalidFactorGroup;
    }
    if (usage.correlation_group.has_value() && !usage.correlation_group->valid()) {
      return LineageValidationError::InvalidCorrelationGroup;
    }
    if (usage.correlation_group && findDeclaration(lineage, *usage.correlation_group) == nullptr) {
      return LineageValidationError::MissingCorrelationDeclaration;
    }
    if (usage.role != ObservationRole::PrimaryResidual) {
      continue;
    }
    for (std::size_t other_index = index + 1; other_index < lineage.usage.size(); ++other_index) {
      const ObservationUsage& other = lineage.usage[other_index];
      if (other.role == ObservationRole::PrimaryResidual && usage.slice.overlaps(other.slice) &&
          !duplicateIsDeclared(lineage, usage, other)) {
        return LineageValidationError::DuplicatePrimaryObservation;
      }
    }
  }
  return LineageValidationError::None;
}

bool lineagesAreIndependent(const ObservationLineage& lhs, const ObservationLineage& rhs) noexcept {
  if (validateLineage(lhs) != LineageValidationError::None ||
      validateLineage(rhs) != LineageValidationError::None) {
    return false;
  }
  for (const ObservationUsage& left : lhs.usage) {
    if (left.role == ObservationRole::RetrievalSeedOnly ||
        left.role == ObservationRole::DerivedSummary) {
      continue;
    }
    for (const ObservationUsage& right : rhs.usage) {
      if (right.role == ObservationRole::RetrievalSeedOnly ||
          right.role == ObservationRole::DerivedSummary) {
        continue;
      }
      if (left.slice.overlaps(right.slice)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace meridian::core
