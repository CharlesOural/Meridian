#include "meridian/core/initialization.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace meridian::core {
namespace {

bool isNonnegativeFinite(const std::optional<double>& value) noexcept {
  return !value.has_value() || (std::isfinite(*value) && *value >= 0.0);
}

bool isValidConditionNumber(const std::optional<double>& value) noexcept {
  return !value.has_value() || (std::isfinite(*value) && *value >= 1.0);
}

bool hasCompleteDynamicQuality(const InitializationQuality& quality) noexcept {
  return quality.registration_min_singular_value.has_value() &&
         quality.registration_condition_number.has_value() &&
         quality.gyro_bias_min_singular_value.has_value() &&
         quality.gyro_bias_condition_number.has_value() &&
         quality.gravity_min_singular_value.has_value() &&
         quality.gravity_condition_number.has_value() &&
         quality.raw_gravity_magnitude_m_s2.has_value() &&
         quality.gyro_bias_correction_norm_rad_s.has_value() &&
         quality.alignment_residual_rms.has_value() &&
         quality.held_out_rotation_error_rad.has_value() &&
         quality.held_out_translation_error_m.has_value() &&
         quality.refinement_rotation_change_rad.has_value() &&
         quality.refinement_translation_change_m.has_value();
}

}  // namespace

const char* toString(InitializationMode mode) noexcept {
  switch (mode) {
    case InitializationMode::kStatic:
      return "static";
    case InitializationMode::kDynamic:
      return "dynamic";
  }
  return "unknown";
}

const char* toString(InitializationStatus status) noexcept {
  switch (status) {
    case InitializationStatus::kCollecting:
      return "collecting";
    case InitializationStatus::kBootstrapping:
      return "bootstrapping";
    case InitializationStatus::kAligning:
      return "aligning";
    case InitializationStatus::kValidating:
      return "validating";
    case InitializationStatus::kAccepted:
      return "accepted";
    case InitializationStatus::kFailed:
      return "failed";
  }
  return "unknown";
}

bool InitializationQuality::isValid() const noexcept {
  if (fitted_transition_count > 0U && lidar_sweep_count < 2U) {
    return false;
  }

  return isNonnegativeFinite(registration_min_singular_value) &&
         isValidConditionNumber(registration_condition_number) &&
         isNonnegativeFinite(gyro_bias_min_singular_value) &&
         isValidConditionNumber(gyro_bias_condition_number) &&
         isNonnegativeFinite(gravity_min_singular_value) &&
         isValidConditionNumber(gravity_condition_number) &&
         isNonnegativeFinite(raw_gravity_magnitude_m_s2) &&
         isNonnegativeFinite(gyro_bias_correction_norm_rad_s) &&
         isNonnegativeFinite(alignment_residual_rms) &&
         isNonnegativeFinite(held_out_rotation_error_rad) &&
         isNonnegativeFinite(held_out_translation_error_m) &&
         isNonnegativeFinite(refinement_rotation_change_rad) &&
         isNonnegativeFinite(refinement_translation_change_m);
}

InitializationResult::InitializationResult(InitializationMode method, TimeNs anchor_time,
                                           NavigationState seed_state, TimeRange support,
                                           InitializationQuality quality)
    : method_(method),
      anchor_time_(anchor_time),
      seed_state_(std::move(seed_state)),
      support_(support),
      quality_(std::move(quality)) {
  const std::optional<std::int64_t> support_duration = support_.durationNs();
  if (!support_duration.has_value() || *support_duration <= 0) {
    throw std::invalid_argument("InitializationResult support must have positive duration");
  }
  if (!support_.contains(anchor_time_)) {
    throw std::invalid_argument("InitializationResult anchor must lie inside support");
  }
  if (seed_state_.time() != anchor_time_) {
    throw std::invalid_argument("InitializationResult seed time must equal anchor time");
  }
  if (!quality_.isValid()) {
    throw std::invalid_argument("InitializationResult quality is invalid");
  }
  if (!quality_.all_required_gates_passed) {
    throw std::invalid_argument("InitializationResult requires all acceptance gates to pass");
  }
  if (quality_.imu_sample_count == 0U) {
    throw std::invalid_argument("InitializationResult requires supporting IMU samples");
  }
  if (!quality_.raw_gravity_magnitude_m_s2.has_value() ||
      *quality_.raw_gravity_magnitude_m_s2 <= 0.0) {
    throw std::invalid_argument("InitializationResult requires a positive raw gravity magnitude");
  }
  if (method_ == InitializationMode::kDynamic) {
    if (quality_.lidar_sweep_count < 3U || quality_.fitted_transition_count == 0U ||
        quality_.fitted_transition_count > quality_.lidar_sweep_count - 2U) {
      throw std::invalid_argument(
          "Dynamic InitializationResult requires fitted and held-out LiDAR transitions");
    }
    if (!hasCompleteDynamicQuality(quality_)) {
      throw std::invalid_argument(
          "Dynamic InitializationResult requires complete observability and validation quality");
    }
  }
}

}  // namespace meridian::core
