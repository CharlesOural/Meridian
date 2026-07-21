#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>

#include "meridian/core/initialization.hpp"
#include "meridian/core/observations.hpp"

namespace meridian::local_rt::initialization {

struct StaticInitializerOptions final {
  std::int64_t window_duration_ns{};
  std::int64_t block_duration_ns{};
  std::size_t minimum_samples{};
  std::size_t minimum_blocks{};
  std::int64_t maximum_sample_gap_ns{};
  double gravity_m_s2{};
  double gyroscope_saturation_rad_s{};
  double accelerometer_saturation_m_s2{};
  double maximum_mean_angular_rate_rad_s{};
  double maximum_block_angular_dispersion_rad_s{};
  double maximum_specific_force_norm_error_m_s2{};
  double maximum_block_direction_dispersion_rad{};
  core::ImuBias calibrated_bias_prior;
  core::Pose3d base_from_imu;
};

struct StaticInitializationUpdate final {
  core::InitializationStatus status{core::InitializationStatus::kCollecting};
  std::string reason;
  core::InitializationQuality quality;
  std::optional<core::InitializationResult> result;
};

// Configured STATIC is an operational zero-motion assertion. This class only
// verifies that a rolling IMU window is consistent with that assertion; it
// never selects or falls back to another initialization mode.
class StaticInitializer final {
public:
  explicit StaticInitializer(StaticInitializerOptions options);

  [[nodiscard]] StaticInitializationUpdate add(const core::ImuSample& sample);
  [[nodiscard]] bool accepted() const noexcept { return accepted_result_.has_value(); }
  void reset() noexcept;

private:
  StaticInitializerOptions options_;
  std::deque<core::ImuSample> samples_;
  std::optional<core::InitializationResult> accepted_result_;
};

}  // namespace meridian::local_rt::initialization
