#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "meridian/core/canonical_records.hpp"
#include "meridian/core/result.hpp"

namespace meridian::core {

enum class CanonicalVerificationFailure {
  SemanticValidation,
  Recomputation,
  DigestMismatch,
};

enum class CanonicalVerificationPreconditionError {
  InvalidIdentity,
  MissingStoredChecksum,
  MissingOpaqueChildChecksum,
};

// Semantic scope of the local-content checksum. Link and seal-envelope fields
// are intentionally absent because they belong to later checksum domains.
enum class CanonicalLocalContentValidationError {
  None,
  InvalidHeader,
  InvalidRef,
  InvalidFinalRevision,
  InvalidSupport,
  InvalidFrame,
  NonFiniteTransform,
  BoundaryIdentityMismatch,
  InvalidPayloadCatalog,
  InvalidLineage,
  MissingQualityChecksum,
};

using CanonicalSemanticValidationError =
    std::variant<std::monostate, CanonicalVerificationPreconditionError, LineageValidationError,
                 FrozenFactorValidationError, CondensedTransitionValidationError,
                 SparsePayloadCatalogValidationError, CanonicalLocalContentValidationError,
                 SealedBoundaryTransitionValidationError, SparseSubmapSealValidationError>;

struct CanonicalVerificationError {
  CanonicalRecordKind domain{CanonicalRecordKind::ObservationLineage};
  CanonicalVerificationFailure failure{CanonicalVerificationFailure::SemanticValidation};
  CanonicalSemanticValidationError semantic_error;
  std::optional<CanonicalRecordError> recomputation_error;
  std::optional<ContentHash> stored_digest;
  std::optional<ContentHash> recomputed_digest;
};

struct CanonicalVerificationReport {
  CanonicalRecordKind root_domain{CanonicalRecordKind::ObservationLineage};
  std::uint32_t verified_records{};
  ContentHash root_digest{};
};

using CanonicalVerificationResult = Result<CanonicalVerificationReport, CanonicalVerificationError>;

// Leaf verification is post-construction: semantic validation still permits
// producers to construct and validate pre-hash records through the existing
// APIs, while these functions require and compare the stored digest.
[[nodiscard]] CanonicalVerificationResult verifyCanonicalObservationLineageChecksum(
    const ObservationLineage& lineage, CanonicalRecordLimits limits = {});
[[nodiscard]] CanonicalVerificationResult verifyCanonicalFrozenSquareRootFactorChecksum(
    const FrozenSquareRootFactor& factor, CanonicalRecordLimits limits = {});
[[nodiscard]] CanonicalVerificationResult verifyCanonicalCondensedBoundaryTransitionChecksum(
    const CondensedBoundaryTransition& transition, CanonicalRecordLimits limits = {});
[[nodiscard]] CanonicalVerificationResult verifyCanonicalSparsePayloadCatalogChecksum(
    const SparsePayloadCatalog& catalog, CanonicalRecordLimits limits = {});
[[nodiscard]] CanonicalVerificationResult verifyCanonicalSubmapLocalContentChecksum(
    const SparseSubmapSeal& seal, CanonicalRecordLimits limits = {});
[[nodiscard]] CanonicalVerificationResult verifyCanonicalSealedBoundaryTransitionChecksum(
    const SealedBoundaryTransition& transition, CanonicalRecordLimits limits = {});
[[nodiscard]] CanonicalVerificationResult verifyCanonicalSparseSubmapSealChecksum(
    const SparseSubmapSeal& seal, CanonicalRecordLimits limits = {});

// Verifies the current seal's complete reachable canonical checksum closure in
// dependency order. A predecessor SubmapRef, opaque observation source hashes,
// blob payload hashes, input-partition hashes, and quality hashes are checked
// for presence but cannot be recomputed without their external bytes/records.
[[nodiscard]] CanonicalVerificationResult verifyCanonicalSparseSubmapSeal(
    const SparseSubmapSeal& seal, CanonicalRecordLimits limits = {});

}  // namespace meridian::core
