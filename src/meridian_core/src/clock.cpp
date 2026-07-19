#include "meridian/core/clock.hpp"

#include <cmath>
#include <limits>
#include <utility>

namespace meridian::core {
namespace {

[[nodiscard]] ClockMappingError mappingError(ClockMappingErrorCode code,
                                             std::string detail) {
  return ClockMappingError{code, std::move(detail)};
}

[[nodiscard]] ClockGuardError guardError(ClockGuardErrorCode code,
                                         std::string detail) {
  return ClockGuardError{code, std::move(detail)};
}

}  // namespace

Result<SourceStamp, ClockMappingError> mapSourceStamp(
    const AffineClockModel& model, RawDeviceTime raw_time,
    ArrivalTime host_arrival_time,
    std::optional<SourceSequence> device_sequence,
    IngressSequence ingress_sequence) {
  using MappingResult = Result<SourceStamp, ClockMappingError>;
  if (!model.valid() || !ingress_sequence.valid()) {
    return MappingResult::failure(mappingError(
        ClockMappingErrorCode::InvalidModel,
        "clock model or generated ingress sequence is invalid"));
  }
  const long double raw_delta =
      static_cast<long double>(raw_time.nanoseconds) -
      static_cast<long double>(model.raw_reference.nanoseconds);
  const long double mapped =
      static_cast<long double>(model.fusion_reference.nanoseconds) +
      raw_delta * static_cast<long double>(model.rate);
  if (!std::isfinite(mapped) ||
      mapped < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
      mapped > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
    return MappingResult::failure(mappingError(
        ClockMappingErrorCode::Overflow,
        "affine raw-to-fusion clock mapping exceeds int64 nanoseconds"));
  }

  SourceStamp stamp;
  stamp.raw_time = raw_time;
  stamp.fusion_time = FusionTime{static_cast<std::int64_t>(std::llround(mapped))};
  stamp.host_arrival_time = host_arrival_time;
  stamp.clock_revision = model.revision;
  stamp.source_epoch = model.source_epoch;
  stamp.device_sequence = device_sequence;
  stamp.ingress_sequence = ingress_sequence;
  stamp.uncertainty = model.uncertainty;
  stamp.status = model.status;
  return MappingResult::success(std::move(stamp));
}

Result<bool, ClockGuardError> SourceClockGuard::admit(
    const SourceStamp& stamp) {
  using GuardResult = Result<bool, ClockGuardError>;
  if (!stamp.clock_revision.valid() || !stamp.source_epoch.valid() ||
      !stamp.ingress_sequence.valid() || stamp.uncertainty.nanoseconds < 0 ||
      stamp.status == TimeMappingStatus::Discontinuous) {
    return GuardResult::failure(guardError(
        ClockGuardErrorCode::InvalidStamp,
        "mapped source stamp has invalid identity, uncertainty, or status"));
  }
  if (last_) {
    if (stamp.source_epoch != last_->source_epoch) {
      return GuardResult::failure(guardError(
          ClockGuardErrorCode::SourceEpochChanged,
          "source epoch transition requires an explicit guard reset"));
    }
    if (stamp.raw_time.nanoseconds <= last_->raw_time.nanoseconds) {
      return GuardResult::failure(guardError(
          ClockGuardErrorCode::NonMonotonicRawTime,
          "raw device stamps must be strictly increasing"));
    }
    if (stamp.fusion_time <= last_->fusion_time) {
      return GuardResult::failure(guardError(
          ClockGuardErrorCode::NonMonotonicFusionTime,
          "mapped fusion stamps must be strictly increasing"));
    }
    if (stamp.ingress_sequence.value() <= last_->ingress_sequence.value()) {
      return GuardResult::failure(guardError(
          ClockGuardErrorCode::NonMonotonicIngressSequence,
          "adapter ingress sequence must be strictly increasing"));
    }
  }
  last_ = stamp;
  return GuardResult::success(true);
}

}  // namespace meridian::core
