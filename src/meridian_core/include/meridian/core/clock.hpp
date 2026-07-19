#pragma once

#include <cmath>
#include <optional>
#include <string>

#include "meridian/core/result.hpp"
#include "meridian/core/time.hpp"

namespace meridian::core {

// Immutable affine mapping from one raw device-clock epoch into fusion time.
// A rate of 1 maps nanoseconds one-for-one; calibrated offset lives in the two
// reference values rather than being hidden in an adapter callback.
struct AffineClockModel {
  ClockRevision revision;
  SourceEpoch source_epoch;
  RawDeviceTime raw_reference;
  FusionTime fusion_reference;
  double rate{1.0};
  Duration uncertainty{};
  TimeMappingStatus status{TimeMappingStatus::Valid};

  [[nodiscard]] bool valid() const noexcept {
    return revision.valid() && source_epoch.valid() && std::isfinite(rate) &&
           rate > 0.0 && uncertainty.nanoseconds >= 0 &&
           status != TimeMappingStatus::Discontinuous;
  }
};

enum class ClockMappingErrorCode {
  InvalidModel,
  Overflow,
};

struct ClockMappingError {
  ClockMappingErrorCode code{};
  std::string detail;
};

[[nodiscard]] Result<SourceStamp, ClockMappingError> mapSourceStamp(
    const AffineClockModel& model, RawDeviceTime raw_time,
    ArrivalTime host_arrival_time, std::optional<SourceSequence> device_sequence,
    IngressSequence ingress_sequence);

enum class ClockGuardErrorCode {
  InvalidStamp,
  SourceEpochChanged,
  NonMonotonicRawTime,
  NonMonotonicFusionTime,
  NonMonotonicIngressSequence,
};

struct ClockGuardError {
  ClockGuardErrorCode code{};
  std::string detail;
};

// Stateful adapter-side audit. It validates immutable mapped stamps before a
// record is admitted; it never clamps, sorts, or rewrites time.
class SourceClockGuard {
 public:
  [[nodiscard]] Result<bool, ClockGuardError> admit(const SourceStamp& stamp);
  void reset() noexcept { last_.reset(); }

 private:
  std::optional<SourceStamp> last_;
};

}  // namespace meridian::core
