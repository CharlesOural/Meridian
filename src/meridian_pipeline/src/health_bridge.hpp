#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <optional>

#include "meridian/debug/telemetry.hpp"
#include "meridian/sensors/health.hpp"

namespace meridian {

// Bridges sensor health onto the telemetry bus: snapshots become per-sensor scalars,
// degradations become events. Sources call this from their (wrapper) threads and the
// aggregator from the stage thread, so the small state it keeps is lock-guarded.
class TelemetryHealthBridge final : public HealthSink {
 public:
  explicit TelemetryHealthBridge(TelemetrySink* sink) : sink_(sink) {}

  void update(const SensorHealth& h) override {
    if (sink_ == nullptr) return;
    const char* key = rate_key(h.modality);
    sink_->scalar(key, h.rate_hz, h.last_sample);
    {
      std::lock_guard<std::mutex> lock(m_);
      auto& prev = last_level_[static_cast<std::size_t>(h.modality)];
      if (prev == h.level) return;
      prev = h.level;
    }
    sink_->event(h.level == HealthLevel::Nominal ? Level::Info : Level::Warn,
                 level_tag(h.modality), level_name(h.level), h.last_sample);
  }

  void degrade(std::uint8_t /*id*/, HealthCode code) override {
    if (sink_ == nullptr) return;
    // A flapping or persistent condition re-raises its code on every occurrence;
    // repeats of the same code inside the window collapse into the first event. The
    // ongoing state stays visible through the level transitions and rate scalars.
    const auto now = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(m_);
      auto& last = last_degrade_[static_cast<std::size_t>(code)];
      if (last && now - *last < kDegradeRepeatWindow) return;
      last = now;
    }
    sink_->event(Level::Warn, "health/degrade", code_name(code), 0);
  }

 private:
  static const char* rate_key(Modality m) {
    switch (m) {
      case Modality::Imu: return "health/imu/rate_hz";
      case Modality::Lidar: return "health/lidar/rate_hz";
      case Modality::Camera: return "health/camera/rate_hz";
      case Modality::Gnss: return "health/gnss/rate_hz";
    }
    return "health/rate_hz";
  }
  static const char* level_tag(Modality m) {
    switch (m) {
      case Modality::Imu: return "health/imu";
      case Modality::Lidar: return "health/lidar";
      case Modality::Camera: return "health/camera";
      case Modality::Gnss: return "health/gnss";
    }
    return "health";
  }
  static const char* level_name(HealthLevel level) {
    switch (level) {
      case HealthLevel::Nominal: return "nominal";
      case HealthLevel::Degraded: return "degraded";
      case HealthLevel::Failed: return "failed";
    }
    return "unknown";
  }
  static const char* code_name(HealthCode code) {
    switch (code) {
      case HealthCode::None: return "none";
      case HealthCode::NoSync: return "no_sync";
      case HealthCode::SyncLost: return "sync_lost";
      case HealthCode::SyncResidualHigh: return "sync_residual_high";
      case HealthCode::ClockStepDetected: return "clock_step_detected";
      case HealthCode::SkewOutOfRange: return "skew_out_of_range";
      case HealthCode::Dropout: return "dropout";
      case HealthCode::RateLow: return "rate_low";
      case HealthCode::RateHigh: return "rate_high";
      case HealthCode::LateDrop: return "late_drop";
      case HealthCode::ImuLate: return "imu_late";
      case HealthCode::ImuNoDeviceClock: return "imu_no_device_clock";
      case HealthCode::CamNoExposure: return "cam_no_exposure";
      case HealthCode::CamTriggerMismatch: return "cam_trigger_mismatch";
      case HealthCode::GnssPpsLost: return "gnss_pps_lost";
      case HealthCode::GnssFixDropped: return "gnss_fix_dropped";
      case HealthCode::LidarNoPointTime: return "lidar_no_point_time";
      case HealthCode::LidarHighNanRatio: return "lidar_high_nan_ratio";
      case HealthCode::EmptyScan: return "empty_scan";
      case HealthCode::ImuNonFinite: return "imu_non_finite";
    }
    return "unknown";
  }

  static constexpr std::chrono::seconds kDegradeRepeatWindow{2};
  static constexpr std::size_t kCodeCount =
      static_cast<std::size_t>(HealthCode::ImuNonFinite) + 1;

  TelemetrySink* sink_;  // borrowed
  std::mutex m_;
  std::array<HealthLevel, 4> last_level_{HealthLevel::Nominal, HealthLevel::Nominal,
                                         HealthLevel::Nominal, HealthLevel::Nominal};
  std::array<std::optional<std::chrono::steady_clock::time_point>, kCodeCount>
      last_degrade_{};
};

}  // namespace meridian
