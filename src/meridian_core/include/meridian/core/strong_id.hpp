#pragma once

#include <compare>
#include <cstdint>
#include <functional>
#include <limits>

namespace meridian::core {

template <typename Tag>
class StrongId {
public:
  using Value = std::uint64_t;

  constexpr StrongId() = default;
  explicit constexpr StrongId(Value value) : value_(value) {}

  [[nodiscard]] constexpr Value value() const noexcept { return value_; }
  [[nodiscard]] constexpr bool valid() const noexcept { return value_ != kInvalidValue; }

  auto operator<=>(const StrongId&) const = default;

  static constexpr Value kInvalidValue = std::numeric_limits<Value>::max();

private:
  Value value_{kInvalidValue};
};

struct TraceIdTag;
struct ProducerIdTag;
struct SessionIdTag;
struct ConfigRevisionTag;
struct CalibrationEpochTag;
struct ClockRevisionTag;
struct SourceEpochTag;
struct SourceSequenceTag;
struct IngressSequenceTag;
struct MeasurementIdTag;
struct GnssObservationIdTag;
struct ObservationLineageIdTag;
struct CorrelationGroupIdTag;
struct CorrelationPolicyRevisionTag;
struct ResidualCalibrationRevisionTag;
struct FactorGroupIdTag;
struct FactorBatchIdTag;
struct SensorRecoveryEpochTag;
struct DerivedRecordIdTag;
struct BlobStoreIdTag;
struct BlobIdTag;
struct LayoutIdTag;
struct StoreInstanceEpochTag;
struct ConsumerIdTag;
struct LeaseTokenIdTag;
struct AcquisitionTokenIdTag;
struct TransportEndpointIdTag;
struct OdomEpochTag;
struct StateIdTag;
struct LocalGraphRevisionTag;
struct KnotRequestIdTag;
struct KnotResolutionIdTag;
struct FactorIdTag;
struct LandmarkSegmentIdTag;
struct SubmapIdTag;
struct SubmapContentRevisionTag;
struct CameraFrameIdTag;
struct SweepIdTag;
struct CameraIdTag;
struct LidarIdTag;

using TraceId = StrongId<TraceIdTag>;
using ProducerId = StrongId<ProducerIdTag>;
using SessionId = StrongId<SessionIdTag>;
using ConfigRevision = StrongId<ConfigRevisionTag>;
using CalibrationEpoch = StrongId<CalibrationEpochTag>;
using ClockRevision = StrongId<ClockRevisionTag>;
using SourceEpoch = StrongId<SourceEpochTag>;
using SourceSequence = StrongId<SourceSequenceTag>;
using IngressSequence = StrongId<IngressSequenceTag>;
using MeasurementId = StrongId<MeasurementIdTag>;
using GnssObservationId = StrongId<GnssObservationIdTag>;
using ObservationLineageId = StrongId<ObservationLineageIdTag>;
using CorrelationGroupId = StrongId<CorrelationGroupIdTag>;
using CorrelationPolicyRevision = StrongId<CorrelationPolicyRevisionTag>;
using ResidualCalibrationRevision = StrongId<ResidualCalibrationRevisionTag>;
using FactorGroupId = StrongId<FactorGroupIdTag>;
using FactorBatchId = StrongId<FactorBatchIdTag>;
using SensorRecoveryEpoch = StrongId<SensorRecoveryEpochTag>;
using DerivedRecordId = StrongId<DerivedRecordIdTag>;
using BlobStoreId = StrongId<BlobStoreIdTag>;
using BlobId = StrongId<BlobIdTag>;
using LayoutId = StrongId<LayoutIdTag>;
using StoreInstanceEpoch = StrongId<StoreInstanceEpochTag>;
using ConsumerId = StrongId<ConsumerIdTag>;
using LeaseTokenId = StrongId<LeaseTokenIdTag>;
using AcquisitionTokenId = StrongId<AcquisitionTokenIdTag>;
using TransportEndpointId = StrongId<TransportEndpointIdTag>;
using OdomEpoch = StrongId<OdomEpochTag>;
using StateId = StrongId<StateIdTag>;
using LocalGraphRevision = StrongId<LocalGraphRevisionTag>;
using KnotRequestId = StrongId<KnotRequestIdTag>;
using KnotResolutionId = StrongId<KnotResolutionIdTag>;
using FactorId = StrongId<FactorIdTag>;
using LandmarkSegmentId = StrongId<LandmarkSegmentIdTag>;
using SubmapId = StrongId<SubmapIdTag>;
using SubmapContentRevision = StrongId<SubmapContentRevisionTag>;
using CameraFrameId = StrongId<CameraFrameIdTag>;
using SweepId = StrongId<SweepIdTag>;
using CameraId = StrongId<CameraIdTag>;
using LidarId = StrongId<LidarIdTag>;

}  // namespace meridian::core

namespace std {

template <typename Tag>
struct hash<meridian::core::StrongId<Tag>> {
  std::size_t operator()(const meridian::core::StrongId<Tag>& id) const noexcept {
    return std::hash<std::uint64_t>{}(id.value());
  }
};

}  // namespace std
