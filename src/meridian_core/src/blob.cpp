#include "meridian/core/blob.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace meridian::core {
namespace {

[[nodiscard]] bool rangeInside(const ByteRange& range, std::uint64_t object_bytes) noexcept {
  return range.bytes > 0U && range.offset <= object_bytes &&
         range.bytes <= object_bytes - range.offset;
}

[[nodiscard]] bool fullObjectRange(const ByteRange& range, std::uint64_t object_bytes) noexcept {
  return range.offset == 0U && range.bytes == object_bytes;
}

[[nodiscard]] bool leaseMatchesStoreInstance(const BlobRef& object,
                                             StoreInstanceEpoch instance) noexcept {
  return object.storage != BlobStorage::SharedMemoryLease ||
         (object.lease_token.has_value() && object.lease_token->issuing_store_instance == instance);
}

[[nodiscard]] bool validTokenRecord(const AcquisitionToken& token) noexcept {
  return token.id.valid() && token.consumer.valid() &&
         validateDurableBlobRef(token.object) == BlobRefValidationError::None;
}

}  // namespace

bool contentHashPresent(const ContentHash& hash) noexcept {
  return std::any_of(hash.begin(), hash.end(), [](std::uint8_t byte) { return byte != 0U; });
}

BlobRefValidationError validateBlobRef(const BlobRef& ref) noexcept {
  if (!ref.store.valid() || !ref.id.valid() || !ref.layout.valid()) {
    return BlobRefValidationError::InvalidIdentity;
  }
  if (!contentHashPresent(ref.checksum)) {
    return BlobRefValidationError::MissingChecksum;
  }
  if (ref.bytes == 0U) {
    return BlobRefValidationError::EmptyObject;
  }

  switch (ref.storage) {
    case BlobStorage::InProcessPool:
    case BlobStorage::DurableSpool:
      if (ref.lease_token.has_value()) {
        return BlobRefValidationError::InvalidLease;
      }
      break;
    case BlobStorage::SharedMemoryLease:
      if (!ref.lease_token.has_value() || !ref.lease_token->id.valid() ||
          !ref.lease_token->issuing_store_instance.valid()) {
        return BlobRefValidationError::InvalidLease;
      }
      break;
    default:
      return BlobRefValidationError::InvalidStorage;
  }
  return BlobRefValidationError::None;
}

BlobRefValidationError validateDurableBlobRef(const BlobRef& ref) noexcept {
  const BlobRefValidationError validation = validateBlobRef(ref);
  if (validation != BlobRefValidationError::None) {
    return validation;
  }
  if (ref.storage != BlobStorage::DurableSpool) {
    return BlobRefValidationError::NonDurableStorage;
  }
  return BlobRefValidationError::None;
}

BlobApiValidationError validateBlobStoreDescriptor(const BlobStoreDescriptor& descriptor) noexcept {
  if (!descriptor.store.valid() || !descriptor.owner.valid() || !descriptor.instance.valid() ||
      !descriptor.endpoint.valid() || descriptor.capabilities.maximum_read_bytes == 0U) {
    return BlobApiValidationError::InvalidDescriptor;
  }
  return BlobApiValidationError::None;
}

BlobApiValidationError validateBlobStatRequest(const BlobStatRequest& request,
                                               const BlobStoreDescriptor& descriptor) noexcept {
  if (validateBlobStoreDescriptor(descriptor) != BlobApiValidationError::None) {
    return BlobApiValidationError::InvalidDescriptor;
  }
  if (validateBlobRef(request.object) != BlobRefValidationError::None) {
    return BlobApiValidationError::InvalidObject;
  }
  if (request.object.store != descriptor.store) {
    return BlobApiValidationError::StoreMismatch;
  }
  if (!request.expected_store_instance.valid() ||
      request.expected_store_instance != descriptor.instance) {
    return BlobApiValidationError::StoreInstanceMismatch;
  }
  if (!leaseMatchesStoreInstance(request.object, descriptor.instance)) {
    return BlobApiValidationError::StoreInstanceMismatch;
  }
  return BlobApiValidationError::None;
}

BlobApiValidationError validateBlobStatRecord(const BlobStatRecord& record,
                                              const BlobStatRequest& request) noexcept {
  if (validateBlobRef(record.object) != BlobRefValidationError::None) {
    return BlobApiValidationError::InvalidObject;
  }
  if (!record.store_instance.valid() || record.store_instance != request.expected_store_instance) {
    return BlobApiValidationError::StoreInstanceMismatch;
  }
  if (!leaseMatchesStoreInstance(record.object, record.store_instance)) {
    return BlobApiValidationError::StoreInstanceMismatch;
  }
  if (record.object != request.object) {
    return BlobApiValidationError::ResponseMismatch;
  }
  return BlobApiValidationError::None;
}

BlobApiValidationError validateBlobReadRequest(const BlobReadRequest& request,
                                               const BlobStoreDescriptor& descriptor) noexcept {
  const BlobStatRequest stat{request.object, request.expected_store_instance};
  const BlobApiValidationError stat_validation = validateBlobStatRequest(stat, descriptor);
  if (stat_validation != BlobApiValidationError::None) {
    return stat_validation;
  }
  if (!rangeInside(request.range, request.object.bytes)) {
    return BlobApiValidationError::InvalidRange;
  }
  if (request.range.bytes > descriptor.capabilities.maximum_read_bytes) {
    return BlobApiValidationError::ReadLimitExceeded;
  }
  if (!descriptor.capabilities.ranged_reads &&
      !fullObjectRange(request.range, request.object.bytes)) {
    return BlobApiValidationError::RangeUnsupported;
  }
  return BlobApiValidationError::None;
}

BlobApiValidationError validateBlobReadChunk(const BlobReadChunk& chunk,
                                             const BlobReadRequest& request) noexcept {
  if (validateBlobRef(chunk.object) != BlobRefValidationError::None) {
    return BlobApiValidationError::InvalidObject;
  }
  if (!chunk.store_instance.valid() || chunk.store_instance != request.expected_store_instance) {
    return BlobApiValidationError::StoreInstanceMismatch;
  }
  if (!leaseMatchesStoreInstance(chunk.object, chunk.store_instance)) {
    return BlobApiValidationError::StoreInstanceMismatch;
  }
  if (chunk.object != request.object || chunk.range != request.range) {
    return BlobApiValidationError::ResponseMismatch;
  }
  if (!chunk.payload || static_cast<std::uint64_t>(chunk.payload->size()) != chunk.range.bytes) {
    return BlobApiValidationError::InvalidPayload;
  }
  return BlobApiValidationError::None;
}

BlobApiValidationError validateFullBlobReadDigest(
    const BlobReadChunk& chunk, const BlobReadRequest& request,
    const ContentHash& computed_payload_digest) noexcept {
  const BlobApiValidationError chunk_validation = validateBlobReadChunk(chunk, request);
  if (chunk_validation != BlobApiValidationError::None) {
    return chunk_validation;
  }
  if (!fullObjectRange(request.range, request.object.bytes)) {
    return BlobApiValidationError::FullObjectRequired;
  }
  if (!contentHashPresent(computed_payload_digest) ||
      computed_payload_digest != request.object.checksum) {
    return BlobApiValidationError::PayloadDigestMismatch;
  }
  return BlobApiValidationError::None;
}

BlobApiValidationError validateBlobAcquireRequest(const BlobAcquireRequest& request,
                                                  const BlobStoreDescriptor& descriptor) noexcept {
  if (validateBlobStoreDescriptor(descriptor) != BlobApiValidationError::None) {
    return BlobApiValidationError::InvalidDescriptor;
  }
  if (!request.consumer.valid()) {
    return BlobApiValidationError::InvalidConsumer;
  }
  if (validateDurableBlobRef(request.object) != BlobRefValidationError::None) {
    return BlobApiValidationError::ObjectMustBeDurable;
  }
  if (request.object.store != descriptor.store) {
    return BlobApiValidationError::StoreMismatch;
  }
  if (!request.expected_store_instance.valid() ||
      request.expected_store_instance != descriptor.instance) {
    return BlobApiValidationError::StoreInstanceMismatch;
  }
  if (!descriptor.capabilities.durable_acquisitions) {
    return BlobApiValidationError::AcquisitionsUnsupported;
  }
  return BlobApiValidationError::None;
}

BlobApiValidationError validateAcquisitionToken(const AcquisitionToken& token,
                                                const BlobAcquireRequest& request) noexcept {
  if (!validTokenRecord(token)) {
    return BlobApiValidationError::InvalidToken;
  }
  if (token.consumer != request.consumer || token.object != request.object) {
    return BlobApiValidationError::ResponseMismatch;
  }
  return BlobApiValidationError::None;
}

BlobApiValidationError validateBlobReleaseRequest(const BlobReleaseRequest& request) noexcept {
  if (!validTokenRecord(request.token)) {
    return BlobApiValidationError::InvalidToken;
  }
  return BlobApiValidationError::None;
}

}  // namespace meridian::core
