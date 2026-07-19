#pragma once

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/strong_id.hpp"

namespace meridian::core {

using ContentHash = std::array<std::uint8_t, 32>;

[[nodiscard]] bool contentHashPresent(const ContentHash& hash) noexcept;

enum class BlobStorage {
  InProcessPool,
  SharedMemoryLease,
  DurableSpool,
};

// Shared-memory leases are scoped to the concrete store process that issued
// them. BlobStoreId remains stable across a restart; issuing_store_instance
// deliberately does not.
struct LeaseToken {
  LeaseTokenId id;
  StoreInstanceEpoch issuing_store_instance;

  auto operator<=>(const LeaseToken&) const = default;
};

struct BlobRef {
  BlobStoreId store;
  BlobId id;
  ContentHash checksum{};
  LayoutId layout;
  std::uint64_t bytes{};
  BlobStorage storage{BlobStorage::InProcessPool};
  std::optional<LeaseToken> lease_token;

  auto operator<=>(const BlobRef&) const = default;
};

using ImmutableBytes = std::shared_ptr<const std::vector<std::byte>>;

enum class BlobRefValidationError {
  None,
  InvalidIdentity,
  MissingChecksum,
  EmptyObject,
  InvalidLease,
  InvalidStorage,
  NonDurableStorage,
};

// Validates identity, immutable byte count, checksum, and storage/lease
// semantics. Shared-memory objects require a strong lease identity and the
// issuing store-process epoch. Ordinary in-process and durable objects must
// not carry a lease.
[[nodiscard]] BlobRefValidationError validateBlobRef(const BlobRef& ref) noexcept;

// Final sparse records may name only lease-free content owned by a durable
// spool. This function performs validateBlobRef() first.
[[nodiscard]] BlobRefValidationError validateDurableBlobRef(const BlobRef& ref) noexcept;

struct ByteRange {
  std::uint64_t offset{};
  std::uint64_t bytes{};

  auto operator<=>(const ByteRange&) const = default;
};

struct BlobTransportCapabilities {
  bool ranged_reads{false};
  bool durable_acquisitions{false};
  std::uint64_t maximum_read_bytes{};
};

// `endpoint` is an opaque deployment-resolved identity. It is deliberately
// not a path, URI, file descriptor, pointer, or middleware-specific handle.
struct BlobStoreDescriptor {
  BlobStoreId store;
  ProducerId owner;
  StoreInstanceEpoch instance;
  BlobTransportCapabilities capabilities;
  TransportEndpointId endpoint;
};

struct BlobStatRequest {
  BlobRef object;
  StoreInstanceEpoch expected_store_instance;
};

struct BlobStatRecord {
  BlobRef object;
  StoreInstanceEpoch store_instance;
};

struct BlobReadRequest {
  BlobRef object;
  StoreInstanceEpoch expected_store_instance;
  ByteRange range;
};

struct BlobReadChunk {
  BlobRef object;
  StoreInstanceEpoch store_instance;
  ByteRange range;
  ImmutableBytes payload;
};

struct BlobAcquireRequest {
  ConsumerId consumer;
  BlobRef object;
  StoreInstanceEpoch expected_store_instance;
};

// A successful repeated acquire for the same
// (consumer, store, object, checksum) returns the same logical token. The
// record carries the complete immutable object identity so release cannot
// accidentally unpin a conflicting revision.
struct AcquisitionToken {
  AcquisitionTokenId id;
  ConsumerId consumer;
  BlobRef object;
};

struct BlobReleaseRequest {
  AcquisitionToken token;
};

enum class BlobApiValidationError {
  None,
  InvalidDescriptor,
  InvalidObject,
  ObjectMustBeDurable,
  StoreMismatch,
  StoreInstanceMismatch,
  InvalidRange,
  RangeUnsupported,
  ReadLimitExceeded,
  InvalidConsumer,
  AcquisitionsUnsupported,
  InvalidToken,
  ResponseMismatch,
  InvalidPayload,
  FullObjectRequired,
  PayloadDigestMismatch,
};

[[nodiscard]] BlobApiValidationError validateBlobStoreDescriptor(
    const BlobStoreDescriptor& descriptor) noexcept;
[[nodiscard]] BlobApiValidationError validateBlobStatRequest(
    const BlobStatRequest& request, const BlobStoreDescriptor& descriptor) noexcept;
[[nodiscard]] BlobApiValidationError validateBlobStatRecord(
    const BlobStatRecord& record, const BlobStatRequest& request) noexcept;
[[nodiscard]] BlobApiValidationError validateBlobReadRequest(
    const BlobReadRequest& request, const BlobStoreDescriptor& descriptor) noexcept;
[[nodiscard]] BlobApiValidationError validateBlobReadChunk(const BlobReadChunk& chunk,
                                                           const BlobReadRequest& request) noexcept;
// Validates a complete immutable read and compares the digest computed over
// the returned payload with the BlobRef checksum. Call this after hashing the
// bytes and before any deserialization. Partial reads are intentionally not
// accepted by this pre-deserialization boundary.
[[nodiscard]] BlobApiValidationError validateFullBlobReadDigest(
    const BlobReadChunk& chunk, const BlobReadRequest& request,
    const ContentHash& computed_payload_digest) noexcept;
[[nodiscard]] BlobApiValidationError validateBlobAcquireRequest(
    const BlobAcquireRequest& request, const BlobStoreDescriptor& descriptor) noexcept;
[[nodiscard]] BlobApiValidationError validateAcquisitionToken(
    const AcquisitionToken& token, const BlobAcquireRequest& request) noexcept;
[[nodiscard]] BlobApiValidationError validateBlobReleaseRequest(
    const BlobReleaseRequest& request) noexcept;

enum class BlobAccessErrorCode {
  InvalidRequest,
  NotFound,
  StaleStoreInstance,
  LeaseExpired,
  IntegrityFailure,
  UnsupportedRange,
  ResourceLimit,
  TransportFailure,
};

struct BlobAccessError {
  BlobAccessErrorCode code{BlobAccessErrorCode::InvalidRequest};
  std::optional<BlobRef> object;
  std::string detail;
};

}  // namespace meridian::core
