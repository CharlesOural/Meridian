#include "meridian/core/canonical_verification.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>

namespace meridian::core {
namespace {

[[nodiscard]] CanonicalVerificationError semanticError(CanonicalRecordKind domain,
                                                       CanonicalSemanticValidationError error) {
  CanonicalVerificationError output;
  output.domain = domain;
  output.failure = CanonicalVerificationFailure::SemanticValidation;
  output.semantic_error = std::move(error);
  return output;
}

[[nodiscard]] CanonicalVerificationError recomputationError(CanonicalRecordKind domain,
                                                            CanonicalRecordError error) {
  CanonicalVerificationError output;
  output.domain = domain;
  output.failure = CanonicalVerificationFailure::Recomputation;
  output.recomputation_error = std::move(error);
  return output;
}

[[nodiscard]] CanonicalVerificationError mismatchError(CanonicalRecordKind domain,
                                                       const ContentHash& stored,
                                                       const ContentHash& recomputed) {
  CanonicalVerificationError output;
  output.domain = domain;
  output.failure = CanonicalVerificationFailure::DigestMismatch;
  output.stored_digest = stored;
  output.recomputed_digest = recomputed;
  return output;
}

[[nodiscard]] CanonicalVerificationReport report(CanonicalRecordKind domain,
                                                 const ContentHash& checksum,
                                                 std::uint32_t count = 1U) {
  return CanonicalVerificationReport{domain, count, checksum};
}

[[nodiscard]] std::optional<CanonicalRecordError> preflightCollections(
    CanonicalRecordKind domain, CanonicalRecordLimits limits,
    std::initializer_list<std::size_t> collections) {
  if (limits.maximum_output_bytes == 0U || limits.maximum_collection_entries == 0U) {
    return CanonicalRecordError{CanonicalRecordErrorCode::InvalidLimits, domain,
                                CanonicalEncodingError::None};
  }
  std::uint64_t remaining = limits.maximum_collection_entries;
  for (const std::size_t size : collections) {
    if (size > remaining) {
      return CanonicalRecordError{CanonicalRecordErrorCode::CollectionLimitExceeded, domain,
                                  CanonicalEncodingError::None};
    }
    remaining -= size;
  }
  return std::nullopt;
}

[[nodiscard]] CanonicalVerificationResult compareRecomputed(CanonicalRecordKind domain,
                                                            const ContentHash& stored,
                                                            CanonicalChecksumResult recomputed) {
  if (!recomputed) {
    return CanonicalVerificationResult::failure(recomputationError(domain, recomputed.error()));
  }
  if (stored != recomputed.value()) {
    return CanonicalVerificationResult::failure(mismatchError(domain, stored, recomputed.value()));
  }
  return CanonicalVerificationResult::success(report(domain, stored));
}

[[nodiscard]] bool finitePose(const Pose3d& pose) noexcept {
  return pose.matrix().allFinite();
}

[[nodiscard]] bool finiteBoundary(const BoundaryNavigationLinearization& boundary) noexcept {
  return boundary.state.valid() && boundary.final_revision.valid() &&
         finitePose(boundary.T_odom_imu) && boundary.velocity_odom.allFinite() &&
         boundary.gyro_bias.allFinite() && boundary.accel_bias.allFinite();
}

[[nodiscard]] CanonicalLocalContentValidationError validateLocalContent(
    const SparseSubmapSeal& seal) noexcept {
  if (seal.header.schema_version == 0U || !seal.header.producer.valid() ||
      !seal.header.session.valid() || !seal.header.config.valid() ||
      seal.header.session != seal.ref.session || !seal.header.direct_calibration.has_value() ||
      !seal.header.direct_calibration->valid() ||
      *seal.header.direct_calibration != seal.ref.calibration) {
    return CanonicalLocalContentValidationError::InvalidHeader;
  }
  if (validateSubmapRef(seal.ref) != SubmapRefValidationError::None) {
    return CanonicalLocalContentValidationError::InvalidRef;
  }
  if (!seal.final_local_revision.valid() || !seal.boundary_navigation.final_revision.valid() ||
      seal.boundary_navigation.final_revision > seal.final_local_revision) {
    return CanonicalLocalContentValidationError::InvalidFinalRevision;
  }
  if (!seal.support_time.valid() || seal.frame.boundary_time != seal.support_time.start) {
    return CanonicalLocalContentValidationError::InvalidSupport;
  }
  const double gravity_norm = seal.frame.gravity_up_odom.norm();
  if (!seal.frame.boundary_state.valid() || !seal.frame.gravity_up_odom.allFinite() ||
      !std::isfinite(gravity_norm) || std::abs(gravity_norm - 1.0) > 1.0e-6 ||
      !std::isfinite(seal.frame.boundary_yaw_odom) ||
      seal.frame.boundary_yaw_odom < -std::numbers::pi ||
      seal.frame.boundary_yaw_odom > std::numbers::pi) {
    return CanonicalLocalContentValidationError::InvalidFrame;
  }
  if (!finitePose(seal.T_odom_submap) || !finiteBoundary(seal.boundary_navigation)) {
    return CanonicalLocalContentValidationError::NonFiniteTransform;
  }
  if (seal.boundary_navigation.state != seal.frame.boundary_state ||
      seal.boundary_navigation.exact_time != seal.frame.boundary_time) {
    return CanonicalLocalContentValidationError::BoundaryIdentityMismatch;
  }
  if (validateSparsePayloadCatalog(seal.payloads) != SparsePayloadCatalogValidationError::None) {
    return CanonicalLocalContentValidationError::InvalidPayloadCatalog;
  }
  if (!seal.lineage.id.valid() || !contentHashPresent(seal.lineage.checksum) ||
      validateLineage(seal.lineage) != LineageValidationError::None) {
    return CanonicalLocalContentValidationError::InvalidLineage;
  }
  if (!contentHashPresent(seal.quality_checksum)) {
    return CanonicalLocalContentValidationError::MissingQualityChecksum;
  }
  return CanonicalLocalContentValidationError::None;
}

[[nodiscard]] std::optional<CanonicalVerificationError> lineagePrecondition(
    const ObservationLineage& lineage, CanonicalRecordKind domain) {
  if (!lineage.id.valid()) {
    return semanticError(domain, CanonicalVerificationPreconditionError::InvalidIdentity);
  }
  if (!contentHashPresent(lineage.checksum)) {
    return semanticError(domain, CanonicalVerificationPreconditionError::MissingStoredChecksum);
  }
  for (const ObservationUsage& usage : lineage.usage) {
    if (!contentHashPresent(usage.slice.source_checksum)) {
      return semanticError(domain,
                           CanonicalVerificationPreconditionError::MissingOpaqueChildChecksum);
    }
  }
  return std::nullopt;
}

[[nodiscard]] CanonicalVerificationResult propagate(CanonicalVerificationResult result,
                                                    std::uint32_t* verified_records) {
  if (result) {
    if (result.value().verified_records >
        std::numeric_limits<std::uint32_t>::max() - *verified_records) {
      CanonicalRecordError overflow{CanonicalRecordErrorCode::CollectionLimitExceeded,
                                    result.value().root_domain,
                                    CanonicalEncodingError::DimensionOverflow};
      return CanonicalVerificationResult::failure(
          recomputationError(result.value().root_domain, overflow));
    }
    *verified_records += result.value().verified_records;
  }
  return result;
}

}  // namespace

CanonicalVerificationResult verifyCanonicalObservationLineageChecksum(
    const ObservationLineage& lineage, CanonicalRecordLimits limits) {
  constexpr CanonicalRecordKind kDomain = CanonicalRecordKind::ObservationLineage;
  if (const auto bounds = preflightCollections(
          kDomain, limits, {lineage.usage.size(), lineage.correlations.size()})) {
    return CanonicalVerificationResult::failure(recomputationError(kDomain, *bounds));
  }
  if (const auto precondition = lineagePrecondition(lineage, kDomain)) {
    return CanonicalVerificationResult::failure(*precondition);
  }
  const LineageValidationError validation = validateLineage(lineage);
  if (validation != LineageValidationError::None) {
    return CanonicalVerificationResult::failure(semanticError(kDomain, validation));
  }
  return compareRecomputed(kDomain, lineage.checksum,
                           recomputeObservationLineageChecksum(lineage, limits));
}

CanonicalVerificationResult verifyCanonicalFrozenSquareRootFactorChecksum(
    const FrozenSquareRootFactor& factor, CanonicalRecordLimits limits) {
  constexpr CanonicalRecordKind kDomain = CanonicalRecordKind::FrozenSquareRootFactor;
  if (const auto bounds = preflightCollections(
          kDomain, limits, {factor.layout.size(), factor.row_major_A.size(), factor.rhs.size()})) {
    return CanonicalVerificationResult::failure(recomputationError(kDomain, *bounds));
  }
  const FrozenFactorValidationError validation = validateFrozenSquareRootFactor(factor);
  if (validation != FrozenFactorValidationError::None) {
    return CanonicalVerificationResult::failure(semanticError(kDomain, validation));
  }
  return compareRecomputed(kDomain, factor.checksum,
                           recomputeFrozenSquareRootFactorChecksum(factor, limits));
}

CanonicalVerificationResult verifyCanonicalCondensedBoundaryTransitionChecksum(
    const CondensedBoundaryTransition& transition, CanonicalRecordLimits limits) {
  constexpr CanonicalRecordKind kDomain = CanonicalRecordKind::CondensedBoundaryTransition;
  if (const auto bounds =
          preflightCollections(kDomain, limits, {transition.source_factors.size()})) {
    return CanonicalVerificationResult::failure(recomputationError(kDomain, *bounds));
  }
  const CondensedTransitionValidationError validation =
      validateCondensedBoundaryTransition(transition);
  if (validation != CondensedTransitionValidationError::None) {
    return CanonicalVerificationResult::failure(semanticError(kDomain, validation));
  }
  return compareRecomputed(kDomain, transition.checksum,
                           recomputeCondensedBoundaryTransitionChecksum(transition, limits));
}

CanonicalVerificationResult verifyCanonicalSparsePayloadCatalogChecksum(
    const SparsePayloadCatalog& catalog, CanonicalRecordLimits limits) {
  constexpr CanonicalRecordKind kDomain = CanonicalRecordKind::SparsePayloadCatalog;
  if (const auto bounds = preflightCollections(kDomain, limits, {catalog.entries.size()})) {
    return CanonicalVerificationResult::failure(recomputationError(kDomain, *bounds));
  }
  const SparsePayloadCatalogValidationError validation = validateSparsePayloadCatalog(catalog);
  if (validation != SparsePayloadCatalogValidationError::None) {
    return CanonicalVerificationResult::failure(semanticError(kDomain, validation));
  }
  return compareRecomputed(kDomain, catalog.checksum,
                           recomputeSparsePayloadCatalogChecksum(catalog, limits));
}

CanonicalVerificationResult verifyCanonicalSubmapLocalContentChecksum(
    const SparseSubmapSeal& seal, CanonicalRecordLimits limits) {
  constexpr CanonicalRecordKind kDomain = CanonicalRecordKind::SubmapLocalContent;
  if (const auto bounds = preflightCollections(kDomain, limits, {})) {
    return CanonicalVerificationResult::failure(recomputationError(kDomain, *bounds));
  }
  const CanonicalLocalContentValidationError validation = validateLocalContent(seal);
  if (validation != CanonicalLocalContentValidationError::None) {
    return CanonicalVerificationResult::failure(semanticError(kDomain, validation));
  }
  return compareRecomputed(kDomain, seal.ref.local_content_checksum,
                           recomputeSubmapLocalContentChecksum(seal, limits));
}

CanonicalVerificationResult verifyCanonicalSealedBoundaryTransitionChecksum(
    const SealedBoundaryTransition& transition, CanonicalRecordLimits limits) {
  constexpr CanonicalRecordKind kDomain = CanonicalRecordKind::SealedBoundaryTransition;
  if (const auto bounds = preflightCollections(kDomain, limits, {})) {
    return CanonicalVerificationResult::failure(recomputationError(kDomain, *bounds));
  }
  const SealedBoundaryTransitionValidationError validation =
      validateSealedBoundaryTransition(transition);
  if (validation != SealedBoundaryTransitionValidationError::None) {
    return CanonicalVerificationResult::failure(semanticError(kDomain, validation));
  }
  return compareRecomputed(kDomain, transition.checksum,
                           recomputeSealedBoundaryTransitionChecksum(transition, limits));
}

CanonicalVerificationResult verifyCanonicalSparseSubmapSealChecksum(const SparseSubmapSeal& seal,
                                                                    CanonicalRecordLimits limits) {
  constexpr CanonicalRecordKind kDomain = CanonicalRecordKind::SparseSubmapSeal;
  if (const auto bounds = preflightCollections(kDomain, limits, {})) {
    return CanonicalVerificationResult::failure(recomputationError(kDomain, *bounds));
  }
  const SparseSubmapSealValidationError validation = validateSparseSubmapSeal(seal);
  if (validation != SparseSubmapSealValidationError::None) {
    return CanonicalVerificationResult::failure(semanticError(kDomain, validation));
  }
  return compareRecomputed(kDomain, seal.seal_checksum,
                           recomputeSparseSubmapSealChecksum(seal, limits));
}

CanonicalVerificationResult verifyCanonicalSparseSubmapSeal(const SparseSubmapSeal& seal,
                                                            CanonicalRecordLimits limits) {
  std::uint32_t verified_records = 0U;

  auto result =
      propagate(verifyCanonicalObservationLineageChecksum(seal.lineage, limits), &verified_records);
  if (!result) {
    return result;
  }

  if (seal.from_previous) {
    const CondensedBoundaryTransition& local = seal.from_previous->local_transition;
    result = propagate(verifyCanonicalObservationLineageChecksum(local.lineage, limits),
                       &verified_records);
    if (!result) {
      return result;
    }
    result = propagate(verifyCanonicalFrozenSquareRootFactorChecksum(local.boundary_factor, limits),
                       &verified_records);
    if (!result) {
      return result;
    }
    result = propagate(verifyCanonicalCondensedBoundaryTransitionChecksum(local, limits),
                       &verified_records);
    if (!result) {
      return result;
    }
  }

  result = propagate(verifyCanonicalSparsePayloadCatalogChecksum(seal.payloads, limits),
                     &verified_records);
  if (!result) {
    return result;
  }
  result = propagate(verifyCanonicalSubmapLocalContentChecksum(seal, limits), &verified_records);
  if (!result) {
    return result;
  }

  if (seal.from_previous) {
    result = propagate(verifyCanonicalSealedBoundaryTransitionChecksum(*seal.from_previous, limits),
                       &verified_records);
    if (!result) {
      return result;
    }
  }

  result = propagate(verifyCanonicalSparseSubmapSealChecksum(seal, limits), &verified_records);
  if (!result) {
    return result;
  }
  return CanonicalVerificationResult::success(
      report(CanonicalRecordKind::SparseSubmapSeal, seal.seal_checksum, verified_records));
}

}  // namespace meridian::core
