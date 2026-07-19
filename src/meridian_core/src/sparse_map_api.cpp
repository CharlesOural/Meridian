#include "meridian/core/sparse_map_api.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <set>
#include <utility>

namespace meridian::core {
namespace {

[[nodiscard]] bool finitePose(const Pose3d& pose) noexcept {
  return pose.matrix().allFinite();
}

[[nodiscard]] bool finiteBoundary(const BoundaryNavigationLinearization& boundary) noexcept {
  return boundary.state.valid() && boundary.final_revision.valid() &&
         finitePose(boundary.T_odom_imu) && boundary.velocity_odom.allFinite() &&
         boundary.gyro_bias.allFinite() && boundary.accel_bias.allFinite();
}

[[nodiscard]] bool sameBoundaryValue(const BoundaryNavigationLinearization& left,
                                     const BoundaryNavigationLinearization& right) noexcept {
  return left.state == right.state && left.exact_time == right.exact_time &&
         left.final_revision == right.final_revision &&
         (left.T_odom_imu.matrix().array() == right.T_odom_imu.matrix().array()).all() &&
         (left.velocity_odom.array() == right.velocity_odom.array()).all() &&
         (left.gyro_bias.array() == right.gyro_bias.array()).all() &&
         (left.accel_bias.array() == right.accel_bias.array()).all();
}

[[nodiscard]] bool validHeaderForSeal(const RecordHeader& header, const SubmapRef& ref) noexcept {
  return header.schema_version != 0U && header.trace.valid() && header.producer.valid() &&
         header.session.valid() && header.config.valid() && header.session == ref.session &&
         header.direct_calibration.has_value() && header.direct_calibration->valid() &&
         *header.direct_calibration == ref.calibration;
}

[[nodiscard]] bool validLineageRecord(const ObservationLineage& lineage) noexcept {
  return lineage.id.valid() && contentHashPresent(lineage.checksum) &&
         validateLineage(lineage) == LineageValidationError::None;
}

[[nodiscard]] std::optional<std::size_t> payloadKindIndex(SparsePayloadKind kind) noexcept {
  switch (kind) {
    case SparsePayloadKind::InternalTrajectory:
      return 0U;
    case SparsePayloadKind::KeyframeIndex:
      return 1U;
    case SparsePayloadKind::RegistrationProxy:
      return 2U;
    case SparsePayloadKind::DenseInputIndex:
      return 3U;
    case SparsePayloadKind::VisualPlaceCatalog:
      return 4U;
    case SparsePayloadKind::LidarPlaceCatalog:
      return 5U;
  }
  return std::nullopt;
}

}  // namespace

SparseSubmapIdentityKey sparseSubmapIdentityKey(const SubmapRef& ref) noexcept {
  return SparseSubmapIdentityKey{ref.session, ref.odom_epoch, ref.id, ref.content_revision};
}

SubmapRefValidationError validateSubmapRef(const SubmapRef& ref) noexcept {
  if (!ref.session.valid() || !ref.odom_epoch.valid() || !ref.id.valid() ||
      !ref.calibration.valid() || !ref.content_revision.valid()) {
    return SubmapRefValidationError::InvalidIdentity;
  }
  if (!contentHashPresent(ref.local_content_checksum)) {
    return SubmapRefValidationError::MissingLocalContentChecksum;
  }
  return SubmapRefValidationError::None;
}

SparsePayloadCatalogValidationError validateSparsePayloadCatalog(
    const SparsePayloadCatalog& catalog) noexcept {
  if (!contentHashPresent(catalog.checksum)) {
    return SparsePayloadCatalogValidationError::MissingChecksum;
  }
  if (catalog.entries.size() < 4U || catalog.entries.size() > 6U) {
    return SparsePayloadCatalogValidationError::MissingRequiredEntry;
  }

  std::array<bool, 6U> present{};
  std::set<std::pair<BlobStoreId, BlobId>> objects;
  std::optional<std::size_t> previous_kind;
  for (const SparsePayloadIndexEntry& entry : catalog.entries) {
    const std::optional<std::size_t> kind = payloadKindIndex(entry.kind);
    if (!kind.has_value()) {
      return SparsePayloadCatalogValidationError::NonCanonicalEntries;
    }
    if (previous_kind.has_value() && *kind <= *previous_kind) {
      return SparsePayloadCatalogValidationError::NonCanonicalEntries;
    }
    previous_kind = kind;
    if (present[*kind]) {
      return SparsePayloadCatalogValidationError::NonCanonicalEntries;
    }
    present[*kind] = true;
    if (validateDurableBlobRef(entry.root) != BlobRefValidationError::None) {
      return SparsePayloadCatalogValidationError::InvalidDurableReference;
    }
    if (!objects.emplace(entry.root.store, entry.root.id).second) {
      return SparsePayloadCatalogValidationError::DuplicatePayload;
    }
  }

  for (std::size_t required = 0U; required < 4U; ++required) {
    if (!present[required]) {
      return SparsePayloadCatalogValidationError::MissingRequiredEntry;
    }
  }
  return SparsePayloadCatalogValidationError::None;
}

SealedBoundaryTransitionValidationError validateSealedBoundaryTransition(
    const SealedBoundaryTransition& transition) noexcept {
  if (validateSubmapRef(transition.from) != SubmapRefValidationError::None) {
    return SealedBoundaryTransitionValidationError::InvalidFromRef;
  }
  if (validateSubmapRef(transition.to) != SubmapRefValidationError::None) {
    return SealedBoundaryTransitionValidationError::InvalidToRef;
  }
  if (transition.from.session == transition.to.session &&
      transition.from.odom_epoch == transition.to.odom_epoch &&
      transition.from.id == transition.to.id) {
    return SealedBoundaryTransitionValidationError::SameEndpoint;
  }
  if (transition.from.session != transition.to.session) {
    return SealedBoundaryTransitionValidationError::SessionMismatch;
  }
  if (transition.from.odom_epoch != transition.to.odom_epoch) {
    return SealedBoundaryTransitionValidationError::OdomEpochMismatch;
  }
  if (validateCondensedBoundaryTransition(transition.local_transition) !=
      CondensedTransitionValidationError::None) {
    return SealedBoundaryTransitionValidationError::InvalidLocalTransition;
  }
  if (transition.local_transition.header.session != transition.from.session) {
    return SealedBoundaryTransitionValidationError::LocalTransitionSessionMismatch;
  }
  if (transition.local_transition.odom_epoch != transition.from.odom_epoch) {
    return SealedBoundaryTransitionValidationError::LocalTransitionEpochMismatch;
  }
  if (!contentHashPresent(transition.checksum)) {
    return SealedBoundaryTransitionValidationError::MissingChecksum;
  }
  if (transition.checksum == transition.local_transition.checksum) {
    return SealedBoundaryTransitionValidationError::ChecksumDomainAliased;
  }
  return SealedBoundaryTransitionValidationError::None;
}

SealedBoundaryTransitionValidationError validateSealedBoundaryTransitionEndpoints(
    const SealedBoundaryTransition& transition,
    const BoundaryNavigationLinearization& expected_from,
    const BoundaryNavigationLinearization& expected_to) noexcept {
  const SealedBoundaryTransitionValidationError validation =
      validateSealedBoundaryTransition(transition);
  if (validation != SealedBoundaryTransitionValidationError::None) {
    return validation;
  }
  if (!finiteBoundary(expected_from) || !finiteBoundary(expected_to) ||
      !sameBoundaryValue(transition.local_transition.from, expected_from) ||
      !sameBoundaryValue(transition.local_transition.to, expected_to)) {
    return SealedBoundaryTransitionValidationError::EndpointIdentityMismatch;
  }
  return SealedBoundaryTransitionValidationError::None;
}

SparseSubmapSealIdentity sparseSubmapSealIdentity(const SparseSubmapSeal& seal) noexcept {
  return SparseSubmapSealIdentity{seal.ref, seal.seal_checksum};
}

SparseSealRedeliveryRelation classifySparseSealRedelivery(
    const SparseSubmapSealIdentity& accepted, const SparseSubmapSealIdentity& incoming) noexcept {
  if (sparseSubmapIdentityKey(accepted.ref) != sparseSubmapIdentityKey(incoming.ref)) {
    return SparseSealRedeliveryRelation::DistinctIdentity;
  }
  if (accepted == incoming) {
    return SparseSealRedeliveryRelation::Idempotent;
  }
  return SparseSealRedeliveryRelation::IdentityConflict;
}

SparseSubmapSealValidationError validateSparseSubmapSeal(const SparseSubmapSeal& seal) noexcept {
  if (!validHeaderForSeal(seal.header, seal.ref)) {
    return SparseSubmapSealValidationError::InvalidHeader;
  }
  if (validateSubmapRef(seal.ref) != SubmapRefValidationError::None) {
    return SparseSubmapSealValidationError::InvalidRef;
  }
  if (!seal.final_local_revision.valid() || !seal.boundary_navigation.final_revision.valid() ||
      seal.boundary_navigation.final_revision > seal.final_local_revision) {
    return SparseSubmapSealValidationError::InvalidFinalRevision;
  }
  if (!seal.support_time.valid() || seal.frame.boundary_time != seal.support_time.start) {
    return SparseSubmapSealValidationError::InvalidSupport;
  }
  const double gravity_norm = seal.frame.gravity_up_odom.norm();
  if (!seal.frame.boundary_state.valid() || !seal.frame.gravity_up_odom.allFinite() ||
      !std::isfinite(gravity_norm) || std::abs(gravity_norm - 1.0) > 1.0e-6 ||
      !std::isfinite(seal.frame.boundary_yaw_odom) ||
      seal.frame.boundary_yaw_odom < -std::numbers::pi ||
      seal.frame.boundary_yaw_odom > std::numbers::pi) {
    return SparseSubmapSealValidationError::InvalidFrame;
  }
  if (!finitePose(seal.T_odom_submap) || !finiteBoundary(seal.boundary_navigation)) {
    return SparseSubmapSealValidationError::NonFiniteTransform;
  }
  if (seal.boundary_navigation.state != seal.frame.boundary_state ||
      seal.boundary_navigation.exact_time != seal.frame.boundary_time) {
    return SparseSubmapSealValidationError::BoundaryIdentityMismatch;
  }
  if (validateSparsePayloadCatalog(seal.payloads) != SparsePayloadCatalogValidationError::None) {
    return SparseSubmapSealValidationError::InvalidPayloadCatalog;
  }
  if (!validLineageRecord(seal.lineage)) {
    return SparseSubmapSealValidationError::InvalidLineage;
  }
  if (!contentHashPresent(seal.quality_checksum)) {
    return SparseSubmapSealValidationError::MissingQualityChecksum;
  }
  if (!contentHashPresent(seal.seal_checksum)) {
    return SparseSubmapSealValidationError::MissingSealChecksum;
  }
  if (seal.seal_checksum == seal.ref.local_content_checksum) {
    return SparseSubmapSealValidationError::ChecksumDomainAliased;
  }
  if (seal.previous.has_value() != seal.from_previous.has_value()) {
    return SparseSubmapSealValidationError::IncompletePreviousLink;
  }
  if (!seal.previous.has_value()) {
    return SparseSubmapSealValidationError::None;
  }
  if (validateSubmapRef(*seal.previous) != SubmapRefValidationError::None) {
    return SparseSubmapSealValidationError::InvalidPreviousRef;
  }

  const SealedBoundaryTransition& transition = *seal.from_previous;
  if (validateSealedBoundaryTransition(transition) !=
      SealedBoundaryTransitionValidationError::None) {
    return SparseSubmapSealValidationError::InvalidBoundaryTransition;
  }
  if (transition.from != *seal.previous) {
    return SparseSubmapSealValidationError::PreviousRefMismatch;
  }
  if (transition.to != seal.ref) {
    return SparseSubmapSealValidationError::CurrentRefMismatch;
  }
  if (!sameBoundaryValue(transition.local_transition.to, seal.boundary_navigation) ||
      transition.local_transition.final_revision > seal.final_local_revision) {
    return SparseSubmapSealValidationError::CurrentBoundaryMismatch;
  }
  return SparseSubmapSealValidationError::None;
}

SparseSubmapLinkValidationError validateSparseSubmapLink(const SparseSubmapSeal& previous,
                                                         const SparseSubmapSeal& current) noexcept {
  if (validateSparseSubmapSeal(previous) != SparseSubmapSealValidationError::None) {
    return SparseSubmapLinkValidationError::InvalidPreviousSeal;
  }
  if (validateSparseSubmapSeal(current) != SparseSubmapSealValidationError::None) {
    return SparseSubmapLinkValidationError::InvalidCurrentSeal;
  }
  if (!current.previous.has_value() || !current.from_previous.has_value()) {
    return SparseSubmapLinkValidationError::MissingPreviousLink;
  }
  if (*current.previous != previous.ref || current.from_previous->from != previous.ref ||
      current.from_previous->to != current.ref) {
    return SparseSubmapLinkValidationError::PreviousRefMismatch;
  }
  if (previous.support_time.end != current.support_time.start) {
    return SparseSubmapLinkValidationError::NonContiguousSupport;
  }
  const SealedBoundaryTransitionValidationError transition_validation =
      validateSealedBoundaryTransitionEndpoints(
          *current.from_previous, previous.boundary_navigation, current.boundary_navigation);
  if (transition_validation == SealedBoundaryTransitionValidationError::EndpointIdentityMismatch) {
    return SparseSubmapLinkValidationError::EndpointIdentityMismatch;
  }
  if (transition_validation != SealedBoundaryTransitionValidationError::None) {
    return SparseSubmapLinkValidationError::InvalidBoundaryTransition;
  }
  return SparseSubmapLinkValidationError::None;
}

}  // namespace meridian::core
