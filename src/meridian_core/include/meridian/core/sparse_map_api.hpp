#pragma once

#include <compare>
#include <cstddef>
#include <optional>
#include <vector>

#include "meridian/core/blob.hpp"
#include "meridian/core/geometry.hpp"
#include "meridian/core/local_graph_api.hpp"
#include "meridian/core/observation_lineage.hpp"
#include "meridian/core/strong_id.hpp"
#include "meridian/core/time.hpp"

namespace meridian::core {

// Complete immutable sparse-content identity. `local_content_checksum` is
// computed before, and therefore never includes, an incoming or outgoing
// adjacent transition.
struct SubmapRef {
  SessionId session;
  OdomEpoch odom_epoch;
  SubmapId id;
  CalibrationEpoch calibration;
  SubmapContentRevision content_revision;
  ContentHash local_content_checksum{};

  auto operator<=>(const SubmapRef&) const = default;
};

struct SparseSubmapIdentityKey {
  SessionId session;
  OdomEpoch odom_epoch;
  SubmapId id;
  SubmapContentRevision content_revision;

  auto operator<=>(const SparseSubmapIdentityKey&) const = default;
};

[[nodiscard]] SparseSubmapIdentityKey sparseSubmapIdentityKey(const SubmapRef& ref) noexcept;

enum class SubmapRefValidationError {
  None,
  InvalidIdentity,
  MissingLocalContentChecksum,
};

[[nodiscard]] SubmapRefValidationError validateSubmapRef(const SubmapRef& ref) noexcept;

enum class SparsePayloadKind {
  InternalTrajectory,
  KeyframeIndex,
  RegistrationProxy,
  DenseInputIndex,
  VisualPlaceCatalog,
  LidarPlaceCatalog,
};

// Each entry names one immutable root record. Layout-specific records may in
// turn name child BlobRefs; their ordered child references are covered by the
// root record's checksum. Final seal entries themselves must already reside in
// durable storage.
struct SparsePayloadIndexEntry {
  SparsePayloadKind kind{SparsePayloadKind::InternalTrajectory};
  BlobRef root;
};

struct SparsePayloadCatalog {
  std::vector<SparsePayloadIndexEntry> entries;
  ContentHash checksum{};
};

enum class SparsePayloadCatalogValidationError {
  None,
  MissingRequiredEntry,
  NonCanonicalEntries,
  DuplicatePayload,
  InvalidDurableReference,
  MissingChecksum,
};

[[nodiscard]] SparsePayloadCatalogValidationError validateSparsePayloadCatalog(
    const SparsePayloadCatalog& catalog) noexcept;

struct SubmapFrameDefinition {
  StateId boundary_state;
  FusionTime boundary_time;
  Vector3d gravity_up_odom{Vector3d::UnitZ()};
  double boundary_yaw_odom{};
};

struct SealedBoundaryTransition {
  SubmapRef from;
  SubmapRef to;
  CondensedBoundaryTransition local_transition;
  ContentHash checksum{};
};

enum class SealedBoundaryTransitionValidationError {
  None,
  InvalidFromRef,
  InvalidToRef,
  SameEndpoint,
  SessionMismatch,
  OdomEpochMismatch,
  InvalidLocalTransition,
  LocalTransitionSessionMismatch,
  LocalTransitionEpochMismatch,
  EndpointIdentityMismatch,
  MissingChecksum,
  ChecksumDomainAliased,
};

[[nodiscard]] SealedBoundaryTransitionValidationError validateSealedBoundaryTransition(
    const SealedBoundaryTransition& transition) noexcept;

// Adds the exact finalized boundary state/time/revision and canonical chart
// value check that cannot be inferred from SubmapRef alone. Values are first
// validated for finiteness by validateCondensedBoundaryTransition(); equality
// here is bit-exact (with poses compared through their exact matrices).
[[nodiscard]] SealedBoundaryTransitionValidationError validateSealedBoundaryTransitionEndpoints(
    const SealedBoundaryTransition& transition,
    const BoundaryNavigationLinearization& expected_from,
    const BoundaryNavigationLinearization& expected_to) noexcept;

struct SparseSubmapSeal {
  RecordHeader header;
  SubmapRef ref;
  std::optional<SubmapRef> previous;
  LocalGraphRevision final_local_revision;
  TimeRange support_time;
  SubmapFrameDefinition frame;
  Pose3d T_odom_submap;
  BoundaryNavigationLinearization boundary_navigation;
  SparsePayloadCatalog payloads;
  std::optional<SealedBoundaryTransition> from_previous;
  ObservationLineage lineage;
  // The detailed quality schema is deliberately separate from this stable
  // envelope. This checksum pins that canonical record without committing the
  // core package to frontend-specific quality fields.
  ContentHash quality_checksum{};
  ContentHash seal_checksum{};
};

struct SparseSubmapSealIdentity {
  SubmapRef ref;
  ContentHash seal_checksum{};

  auto operator<=>(const SparseSubmapSealIdentity&) const = default;
};

[[nodiscard]] SparseSubmapSealIdentity sparseSubmapSealIdentity(
    const SparseSubmapSeal& seal) noexcept;

enum class SparseSealRedeliveryRelation {
  DistinctIdentity,
  Idempotent,
  IdentityConflict,
};

// `(session, odom epoch, submap, content revision)` selects an identity.
// Redelivery is idempotent only when the complete SubmapRef and independent
// seal checksum are bit-identical.
[[nodiscard]] SparseSealRedeliveryRelation classifySparseSealRedelivery(
    const SparseSubmapSealIdentity& accepted, const SparseSubmapSealIdentity& incoming) noexcept;

enum class SparseSubmapSealValidationError {
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
  MissingSealChecksum,
  ChecksumDomainAliased,
  IncompletePreviousLink,
  InvalidPreviousRef,
  InvalidBoundaryTransition,
  PreviousRefMismatch,
  CurrentRefMismatch,
  CurrentBoundaryMismatch,
};

// Validates the self-contained seal envelope. For a successor this proves the
// exact current endpoint, but the previous endpoint requires the predecessor
// seal and is checked by validateSparseSubmapLink().
[[nodiscard]] SparseSubmapSealValidationError validateSparseSubmapSeal(
    const SparseSubmapSeal& seal) noexcept;

enum class SparseSubmapLinkValidationError {
  None,
  InvalidPreviousSeal,
  InvalidCurrentSeal,
  MissingPreviousLink,
  PreviousRefMismatch,
  NonContiguousSupport,
  InvalidBoundaryTransition,
  EndpointIdentityMismatch,
};

[[nodiscard]] SparseSubmapLinkValidationError validateSparseSubmapLink(
    const SparseSubmapSeal& previous, const SparseSubmapSeal& current) noexcept;

}  // namespace meridian::core
