#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <variant>

#include "meridian/core/observations.hpp"
#include "meridian/local_rt/config.hpp"
#include "meridian/local_rt/imu_types.hpp"

namespace meridian::local_rt {

enum class ImuInsertStatus : std::uint8_t {
  kInserted,
  kDuplicateTimestamp,
  kOutOfOrder,
  kNonFinite,
};

struct ImuInsertResult final {
  ImuInsertStatus status{ImuInsertStatus::kInserted};
  std::size_t evicted_samples{};

  [[nodiscard]] bool ok() const noexcept { return status == ImuInsertStatus::kInserted; }
};

enum class ImuIntervalErrorCode : std::uint8_t {
  kInvalidRange,
  kEmptyBuffer,
  kBeginNotBracketed,
  kEndNotBracketed,
  kSourceGapTooLarge,
};

struct ImuIntervalFailure final {
  ImuIntervalErrorCode code;
  std::string message;
  std::optional<core::TimeRange> offending_gap;
};

class ImuIntervalResult final {
public:
  explicit ImuIntervalResult(ImuInterval interval);
  explicit ImuIntervalResult(ImuIntervalFailure failure);

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] const ImuInterval* value() const noexcept;
  [[nodiscard]] const ImuIntervalFailure* error() const noexcept;

private:
  std::variant<ImuInterval, ImuIntervalFailure> result_;
};

// A bounded, strictly time-ordered IMU history. Intervals have exact endpoint
// support: measurements are linearly interpolated at both boundaries and each
// returned segment contains the trapezoidal midpoint measurement for its own
// disjoint integration duration.
class ImuBuffer final {
public:
  explicit ImuBuffer(ImuBufferConfig config);

  [[nodiscard]] ImuInsertResult insert(const core::ImuSample& sample);
  [[nodiscard]] ImuIntervalResult interval(core::TimeNs begin, core::TimeNs end) const;

  [[nodiscard]] std::size_t size() const noexcept { return samples_.size(); }
  [[nodiscard]] std::size_t capacity() const noexcept { return config_.capacity; }
  [[nodiscard]] bool empty() const noexcept { return samples_.empty(); }
  [[nodiscard]] std::optional<core::TimeNs> latestTime() const noexcept;
  void clear() noexcept { samples_.clear(); }

private:
  struct BufferedSample final {
    core::TimeNs time;
    Eigen::Vector3d angular_velocity_rad_s;
    Eigen::Vector3d specific_force_m_s2;
  };

  ImuBufferConfig config_;
  std::deque<BufferedSample> samples_;
};

}  // namespace meridian::local_rt
