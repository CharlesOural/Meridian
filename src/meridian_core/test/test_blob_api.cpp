#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "meridian/core/blob.hpp"

namespace meridian::core {
namespace {

ContentHash presentHash(std::uint8_t value) {
  ContentHash hash{};
  hash.front() = value;
  return hash;
}

BlobRef durableBlob(std::uint64_t id = 2U, std::uint64_t bytes = 16U) {
  BlobRef ref;
  ref.store = BlobStoreId{1U};
  ref.id = BlobId{id};
  ref.checksum = presentHash(static_cast<std::uint8_t>(id));
  ref.layout = LayoutId{3U};
  ref.bytes = bytes;
  ref.storage = BlobStorage::DurableSpool;
  return ref;
}

BlobStoreDescriptor descriptor() {
  BlobStoreDescriptor value;
  value.store = BlobStoreId{1U};
  value.owner = ProducerId{2U};
  value.instance = StoreInstanceEpoch{3U};
  value.capabilities.ranged_reads = true;
  value.capabilities.durable_acquisitions = true;
  value.capabilities.maximum_read_bytes = 16U;
  value.endpoint = TransportEndpointId{4U};
  return value;
}

TEST(BlobRefApi, EnforcesStorageAndLeaseSemantics) {
  BlobRef ref = durableBlob();
  EXPECT_EQ(validateBlobRef(ref), BlobRefValidationError::None);
  EXPECT_EQ(validateDurableBlobRef(ref), BlobRefValidationError::None);

  ref.storage = BlobStorage::SharedMemoryLease;
  EXPECT_EQ(validateBlobRef(ref), BlobRefValidationError::InvalidLease);
  ref.lease_token = LeaseToken{LeaseTokenId{9U}, StoreInstanceEpoch{3U}};
  EXPECT_EQ(validateBlobRef(ref), BlobRefValidationError::None);
  EXPECT_EQ(validateDurableBlobRef(ref), BlobRefValidationError::NonDurableStorage);

  ref.lease_token->id = LeaseTokenId{};
  EXPECT_EQ(validateBlobRef(ref), BlobRefValidationError::InvalidLease);
  ref.lease_token = LeaseToken{LeaseTokenId{9U}, StoreInstanceEpoch{}};
  EXPECT_EQ(validateBlobRef(ref), BlobRefValidationError::InvalidLease);
  ref.lease_token = LeaseToken{LeaseTokenId{9U}, StoreInstanceEpoch{3U}};

  ref.storage = static_cast<BlobStorage>(99);
  EXPECT_EQ(validateBlobRef(ref), BlobRefValidationError::InvalidStorage);

  ref.storage = BlobStorage::DurableSpool;
  EXPECT_EQ(validateBlobRef(ref), BlobRefValidationError::InvalidLease);
  ref.lease_token.reset();
  ref.checksum = {};
  EXPECT_EQ(validateBlobRef(ref), BlobRefValidationError::MissingChecksum);
}

TEST(BlobRefApi, RejectsLeaseFromAStaleStoreProcess) {
  BlobStoreDescriptor store = descriptor();
  BlobRef ref = durableBlob();
  ref.storage = BlobStorage::SharedMemoryLease;
  ref.lease_token = LeaseToken{LeaseTokenId{9U}, StoreInstanceEpoch{2U}};
  const BlobStatRequest request{ref, store.instance};

  EXPECT_EQ(validateBlobRef(ref), BlobRefValidationError::None);
  EXPECT_EQ(validateBlobStatRequest(request, store), BlobApiValidationError::StoreInstanceMismatch);

  const BlobStatRecord record{ref, store.instance};
  EXPECT_EQ(validateBlobStatRecord(record, request), BlobApiValidationError::StoreInstanceMismatch);
  const BlobReadRequest read_request{ref, store.instance, ByteRange{0U, ref.bytes}};
  const BlobReadChunk chunk{ref, store.instance, read_request.range,
                            std::make_shared<const std::vector<std::byte>>(ref.bytes)};
  EXPECT_EQ(validateBlobReadChunk(chunk, read_request),
            BlobApiValidationError::StoreInstanceMismatch);
}

TEST(BlobReadApi, BindsStoreInstanceRangeAndResponseExactly) {
  BlobStoreDescriptor store = descriptor();
  BlobReadRequest request{durableBlob(), store.instance, ByteRange{4U, 8U}};
  EXPECT_EQ(validateBlobReadRequest(request, store), BlobApiValidationError::None);

  BlobReadChunk chunk{request.object, request.expected_store_instance, request.range,
                      std::make_shared<const std::vector<std::byte>>(8U)};
  EXPECT_EQ(validateBlobReadChunk(chunk, request), BlobApiValidationError::None);

  chunk.payload = std::make_shared<const std::vector<std::byte>>(7U);
  EXPECT_EQ(validateBlobReadChunk(chunk, request), BlobApiValidationError::InvalidPayload);

  request.range = ByteRange{15U, 2U};
  EXPECT_EQ(validateBlobReadRequest(request, store), BlobApiValidationError::InvalidRange);

  request.range = ByteRange{0U, 16U};
  store.capabilities.maximum_read_bytes = 8U;
  EXPECT_EQ(validateBlobReadRequest(request, store), BlobApiValidationError::ReadLimitExceeded);
}

TEST(BlobReadApi, WholeObjectOnlyTransportRejectsPartialRange) {
  BlobStoreDescriptor store = descriptor();
  store.capabilities.ranged_reads = false;
  BlobReadRequest request{durableBlob(), store.instance, ByteRange{1U, 4U}};
  EXPECT_EQ(validateBlobReadRequest(request, store), BlobApiValidationError::RangeUnsupported);

  request.range = ByteRange{0U, request.object.bytes};
  EXPECT_EQ(validateBlobReadRequest(request, store), BlobApiValidationError::None);
}

TEST(BlobReadApi, VerifiesWholeObjectDigestBeforeDeserialization) {
  const BlobStoreDescriptor store = descriptor();
  BlobReadRequest request{durableBlob(), store.instance, ByteRange{0U, 16U}};
  BlobReadChunk chunk{request.object, request.expected_store_instance, request.range,
                      std::make_shared<const std::vector<std::byte>>(16U)};

  EXPECT_EQ(validateFullBlobReadDigest(chunk, request, request.object.checksum),
            BlobApiValidationError::None);
  EXPECT_EQ(validateFullBlobReadDigest(chunk, request, presentHash(99U)),
            BlobApiValidationError::PayloadDigestMismatch);

  request.range = ByteRange{0U, 8U};
  chunk.range = request.range;
  chunk.payload = std::make_shared<const std::vector<std::byte>>(8U);
  EXPECT_EQ(validateFullBlobReadDigest(chunk, request, request.object.checksum),
            BlobApiValidationError::FullObjectRequired);
}

TEST(BlobStatApi, RejectsStaleInstanceAndSubstitutedObject) {
  const BlobStoreDescriptor store = descriptor();
  BlobStatRequest request{durableBlob(), store.instance};
  EXPECT_EQ(validateBlobStatRequest(request, store), BlobApiValidationError::None);

  BlobStatRecord record{request.object, request.expected_store_instance};
  EXPECT_EQ(validateBlobStatRecord(record, request), BlobApiValidationError::None);
  record.object.id = BlobId{99U};
  EXPECT_EQ(validateBlobStatRecord(record, request), BlobApiValidationError::ResponseMismatch);

  request.expected_store_instance = StoreInstanceEpoch{8U};
  EXPECT_EQ(validateBlobStatRequest(request, store), BlobApiValidationError::StoreInstanceMismatch);
}

TEST(BlobAcquisitionApi, PinsOnlyExactDurableIdentityIdempotently) {
  const BlobStoreDescriptor store = descriptor();
  const BlobAcquireRequest request{ConsumerId{9U}, durableBlob(), store.instance};
  EXPECT_EQ(validateBlobAcquireRequest(request, store), BlobApiValidationError::None);

  const AcquisitionToken token{AcquisitionTokenId{10U}, request.consumer, request.object};
  EXPECT_EQ(validateAcquisitionToken(token, request), BlobApiValidationError::None);
  // A retry has the same logical request and accepts the same token.
  EXPECT_EQ(validateAcquisitionToken(token, request), BlobApiValidationError::None);
  EXPECT_EQ(validateBlobReleaseRequest(BlobReleaseRequest{token}), BlobApiValidationError::None);

  // Store-process freshness belongs to the transport request, not the stable
  // logical acquisition identity. A restart therefore accepts the same token
  // for the same consumer and immutable object.
  BlobStoreDescriptor restarted = store;
  restarted.instance = StoreInstanceEpoch{4U};
  BlobAcquireRequest retry = request;
  retry.expected_store_instance = restarted.instance;
  EXPECT_EQ(validateBlobAcquireRequest(retry, restarted), BlobApiValidationError::None);
  EXPECT_EQ(validateAcquisitionToken(token, retry), BlobApiValidationError::None);
  EXPECT_EQ(validateBlobAcquireRequest(request, restarted),
            BlobApiValidationError::StoreInstanceMismatch);

  BlobAcquireRequest conflicting = request;
  conflicting.consumer = ConsumerId{11U};
  EXPECT_EQ(validateAcquisitionToken(token, conflicting), BlobApiValidationError::ResponseMismatch);

  BlobAcquireRequest leased = request;
  leased.object.storage = BlobStorage::SharedMemoryLease;
  leased.object.lease_token = LeaseToken{LeaseTokenId{12U}, store.instance};
  EXPECT_EQ(validateBlobAcquireRequest(leased, store), BlobApiValidationError::ObjectMustBeDurable);
}

}  // namespace
}  // namespace meridian::core
