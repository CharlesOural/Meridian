#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

#include "meridian/local_rt/pipeline.hpp"

namespace meridian::local_rt {
namespace {

class CapturingSink final : public core::DebugSink {
public:
  [[nodiscard]] bool wantsLidarPreview() const noexcept override { return false; }
  void record(const core::ImuAcceptedEvent&) noexcept override {}
  void record(const core::LidarAcceptedEvent&) noexcept override {}
  void record(const core::LidarPreviewEvent&) noexcept override {}
  void record(const core::IngressFailureEvent&) noexcept override {}
  void record(const core::PreintegrationEvent&) noexcept override {
    preintegrations.fetch_add(1U, std::memory_order_relaxed);
  }
  void record(const core::InitializationEvent& event) noexcept override {
    if (event.status == core::InitializationStatus::kAccepted && event.accepted_seed.has_value()) {
      accepted.fetch_add(1U, std::memory_order_relaxed);
    }
  }
  void record(const core::BootstrapPoseEvent&) noexcept override {}
  void record(const core::LocalRegistrationMapEvent& event) noexcept override {
    registration_maps.fetch_add(1U, std::memory_order_relaxed);
    if (event.points) {
      registration_map_points.store(event.points->size(), std::memory_order_relaxed);
    }
  }
  [[nodiscard]] std::uint64_t droppedEvents() const noexcept override { return 0U; }
  [[nodiscard]] std::uint64_t logErrors() const noexcept override { return 0U; }

  std::atomic<std::uint64_t> accepted{0U};
  std::atomic<std::uint64_t> preintegrations{0U};
  std::atomic<std::uint64_t> registration_maps{0U};
  std::atomic<std::uint64_t> registration_map_points{0U};
};

core::ImuSample imu(std::uint64_t id, std::int64_t time_ns) {
  return core::ImuSample(
      core::ObservationHeader(core::SensorId("imu"), core::CalibrationId("imu-calibration"),
                              core::MeasurementId(id), core::TimeNs(time_ns), "imu"),
      {}, {.x = 0.0, .y = 0.0, .z = 9.80665});
}

core::LidarSweep lidarSweep() {
  return core::LidarSweep(
      core::ObservationHeader(core::SensorId("lidar"), core::CalibrationId("lidar-calibration"),
                              core::MeasurementId(10'000U), core::TimeNs(2'100'000'000), "lidar"),
      core::TimeNs(2'100'000'000), core::TimeNs(2'200'000'000),
      std::vector<core::LidarPoint>{{.x = 1.0F,
                                     .y = 0.0F,
                                     .z = 0.0F,
                                     .time_offset_ns = 100'000'000,
                                     .source_index = 0U,
                                     .intensity = std::nullopt,
                                     .ring = std::nullopt}});
}

TEST(LocalRtPipeline, ProcessesAReadyLidarBeforeLaterImuCanEvictItsSupport) {
  LocalRtPipelineConfig config;
  config.estimator.initialization.mode = core::InitializationMode::kStatic;
  config.estimator.imu_buffer = {
      .capacity = 64U,
      .maximum_gap = std::chrono::milliseconds(10),
  };
  config.static_initializer = initialization::StaticInitializerOptions{
      .window_duration_ns = 2'000'000'000,
      .block_duration_ns = 50'000'000,
      .minimum_samples = 300U,
      .minimum_blocks = 30U,
      .maximum_sample_gap_ns = 10'000'000,
      .gravity_m_s2 = 1.0,
      .gyroscope_saturation_rad_s = 8.0,
      .accelerometer_saturation_m_s2 = 80.0,
      .maximum_mean_angular_rate_rad_s = 0.1,
      .maximum_block_angular_dispersion_rad_s = 0.01,
      .maximum_specific_force_norm_error_m_s2 = 0.2,
      .maximum_block_direction_dispersion_rad = 0.02,
      .calibrated_bias_prior = {},
      .base_from_imu = {},
  };
  config.imu_queue_capacity = 1024U;
  config.lidar_queue_capacity = 4U;

  CapturingSink sink;
  LocalRtPipeline pipeline(std::move(config), sink);
  pipeline.submit(lidarSweep());
  for (std::uint64_t index = 0U; index <= 460U; ++index) {
    pipeline.submit(imu(index + 1U, static_cast<std::int64_t>(index) * 5'000'000));
  }
  pipeline.stopAndDrain();

  ASSERT_TRUE(pipeline.initializationResult().has_value());
  EXPECT_EQ(sink.accepted.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(sink.preintegrations.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(sink.registration_maps.load(std::memory_order_relaxed), 1U);
  EXPECT_GT(sink.registration_map_points.load(std::memory_order_relaxed), 0U);
}

}  // namespace
}  // namespace meridian::local_rt
