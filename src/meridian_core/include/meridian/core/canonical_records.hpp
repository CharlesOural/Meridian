#pragma once

#include <cstdint>
#include <string_view>

#include "meridian/core/canonical_bytes.hpp"
#include "meridian/core/local_graph_api.hpp"
#include "meridian/core/observation_lineage.hpp"
#include "meridian/core/result.hpp"
#include "meridian/core/sparse_map_api.hpp"

namespace meridian::core {

enum class CanonicalRecordKind {
  ObservationLineage,
  FrozenSquareRootFactor,
  CondensedBoundaryTransition,
  SparsePayloadCatalog,
  SubmapLocalContent,
  SealedBoundaryTransition,
  SparseSubmapSeal,
};

struct CanonicalRecordDomain {
  CanonicalRecordKind kind{CanonicalRecordKind::ObservationLineage};
  std::string_view tag;
  std::uint32_t schema_version{};
};

// These identifiers are wire constants. Changing a tag or schema version
// changes every digest in that domain and therefore requires an explicit
// producer/consumer migration.
inline constexpr CanonicalRecordDomain kObservationLineageChecksumDomain{
    CanonicalRecordKind::ObservationLineage, "meridian.core.checksum.observation_lineage", 1U};
inline constexpr CanonicalRecordDomain kFrozenSquareRootFactorChecksumDomain{
    CanonicalRecordKind::FrozenSquareRootFactor, "meridian.core.checksum.frozen_square_root_factor",
    1U};
inline constexpr CanonicalRecordDomain kCondensedBoundaryTransitionChecksumDomain{
    CanonicalRecordKind::CondensedBoundaryTransition,
    "meridian.core.checksum.condensed_boundary_transition", 1U};
inline constexpr CanonicalRecordDomain kSparsePayloadCatalogChecksumDomain{
    CanonicalRecordKind::SparsePayloadCatalog, "meridian.core.checksum.sparse_payload_catalog", 1U};
inline constexpr CanonicalRecordDomain kSubmapLocalContentChecksumDomain{
    CanonicalRecordKind::SubmapLocalContent, "meridian.core.checksum.submap_local_content", 1U};
inline constexpr CanonicalRecordDomain kSealedBoundaryTransitionChecksumDomain{
    CanonicalRecordKind::SealedBoundaryTransition,
    "meridian.core.checksum.sealed_boundary_transition", 1U};
inline constexpr CanonicalRecordDomain kSparseSubmapSealChecksumDomain{
    CanonicalRecordKind::SparseSubmapSeal, "meridian.core.checksum.sparse_submap_seal", 1U};

struct CanonicalRecordLimits {
  // The encoder never grows beyond this limit.
  std::uint64_t maximum_output_bytes{64ULL * 1024ULL * 1024ULL};
  // Sum of all variable-length collection counts in one record. This bounds
  // iteration even when individual collections are adversarially sized.
  std::uint64_t maximum_collection_entries{16ULL * 1024ULL * 1024ULL};
};

enum class CanonicalRecordErrorCode {
  InvalidLimits,
  CollectionLimitExceeded,
  MissingDependencyChecksum,
  UnsupportedValue,
  EncodingFailure,
};

struct CanonicalRecordError {
  CanonicalRecordErrorCode code{CanonicalRecordErrorCode::EncodingFailure};
  CanonicalRecordKind record{CanonicalRecordKind::ObservationLineage};
  CanonicalEncodingError encoding_error{CanonicalEncodingError::None};
};

using CanonicalChecksumResult = Result<ContentHash, CanonicalRecordError>;

// These functions compute the expected digest but deliberately do not mutate
// the record and do not make validation claims. Producers first compute and
// install child checksums, then compute parents in the order declared below;
// validators remain the authority for semantic validity. A record's own
// checksum field is always excluded from its digest.
[[nodiscard]] CanonicalChecksumResult recomputeObservationLineageChecksum(
    const ObservationLineage& lineage, CanonicalRecordLimits limits = {});
[[nodiscard]] CanonicalChecksumResult recomputeFrozenSquareRootFactorChecksum(
    const FrozenSquareRootFactor& factor, CanonicalRecordLimits limits = {});

// Embeds boundary_factor.checksum and lineage.checksum, not their raw child
// fields. input_partition_checksum is part of this domain.
[[nodiscard]] CanonicalChecksumResult recomputeCondensedBoundaryTransitionChecksum(
    const CondensedBoundaryTransition& transition, CanonicalRecordLimits limits = {});
[[nodiscard]] CanonicalChecksumResult recomputeSparsePayloadCatalogChecksum(
    const SparsePayloadCatalog& catalog, CanonicalRecordLimits limits = {});

// Covers local immutable content while excluding ref.local_content_checksum,
// previous, from_previous, and seal_checksum. Header trace/creation metadata is
// seal-envelope data; local content includes its schema, producer, session,
// config and direct calibration projection.
[[nodiscard]] CanonicalChecksumResult recomputeSubmapLocalContentChecksum(
    const SparseSubmapSeal& seal, CanonicalRecordLimits limits = {});

// Embeds both exact SubmapRefs and local_transition.checksum.
[[nodiscard]] CanonicalChecksumResult recomputeSealedBoundaryTransitionChecksum(
    const SealedBoundaryTransition& transition, CanonicalRecordLimits limits = {});

// Covers the complete seal envelope and exact stored child checksums, excluding
// only seal_checksum itself.
[[nodiscard]] CanonicalChecksumResult recomputeSparseSubmapSealChecksum(
    const SparseSubmapSeal& seal, CanonicalRecordLimits limits = {});

}  // namespace meridian::core
