#pragma once

#include <cstdint>
#include <optional>

#include "meridian/core/navigation.hpp"
#include "meridian/core/time.hpp"

namespace meridian::core {

enum class InitializationMode : std::uint8_t {
  kStatic,
  kDynamic,
};

enum class InitializationStatus : std::uint8_t {
  kCollecting,
  kBootstrapping,
  kAligning,
  kValidating,
  kAccepted,
  kFailed,
};

[[nodiscard]] const char* toString(InitializationMode mode) noexcept;
[[nodiscard]] const char* toString(InitializationStatus status) noexcept;

// Diagnostics carried by an accepted result and reusable while reporting an
// initializer's progress. Algorithm-specific values remain optional so the
// same record can describe configured static and dynamic initialization.
struct InitializationQuality final {
  std::uint64_t imu_sample_count{};
  std::uint32_t lidar_sweep_count{};
  std::uint32_t fitted_transition_count{};
  std::uint32_t rejected_transition_count{};

  std::optional<double> registration_min_singular_value;
  std::optional<double> registration_condition_number;
  std::optional<double> gyro_bias_min_singular_value;
  std::optional<double> gyro_bias_condition_number;
  std::optional<double> gravity_min_singular_value;
  std::optional<double> gravity_condition_number;

  std::optional<double> raw_gravity_magnitude_m_s2;
  std::optional<double> gyro_bias_correction_norm_rad_s;
  std::optional<double> alignment_residual_rms;
  std::optional<double> held_out_rotation_error_rad;
  std::optional<double> held_out_translation_error_m;
  std::optional<double> refinement_rotation_change_rad;
  std::optional<double> refinement_translation_change_m;

  bool all_required_gates_passed{};

  [[nodiscard]] bool isValid() const noexcept;
};

// An InitializationResult represents an accepted warm start only. Collection,
// estimation, validation, and failure are exposed through InitializationStatus
// and do not create partially initialized navigation states.
class InitializationResult final {
public:
  InitializationResult(InitializationMode method, TimeNs anchor_time, NavigationState seed_state,
                       TimeRange support, InitializationQuality quality);

  [[nodiscard]] constexpr InitializationMode method() const noexcept { return method_; }
  [[nodiscard]] constexpr TimeNs anchorTime() const noexcept { return anchor_time_; }
  [[nodiscard]] constexpr const NavigationState& seedState() const noexcept { return seed_state_; }
  [[nodiscard]] constexpr const TimeRange& support() const noexcept { return support_; }
  [[nodiscard]] constexpr const InitializationQuality& quality() const noexcept { return quality_; }

private:
  InitializationMode method_;
  TimeNs anchor_time_;
  NavigationState seed_state_;
  TimeRange support_;
  InitializationQuality quality_;
};

}  // namespace meridian::core
