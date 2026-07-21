#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>

#include "meridian/core/debug_sink.hpp"
#include "meridian/core/initialization.hpp"
#include "meridian/core/observations.hpp"
#include "meridian/local_rt/combined_preintegration.hpp"
#include "meridian/local_rt/config.hpp"
#include "meridian/local_rt/estimator/fixed_lag_estimator.hpp"
#include "meridian/local_rt/imu_buffer.hpp"
#include "meridian/local_rt/imu_propagator.hpp"
#include "meridian/local_rt/initialization/dynamic_initializer.hpp"
#include "meridian/local_rt/initialization/static_initializer.hpp"
#include "meridian/local_rt/lidar/scan_preprocessor.hpp"

namespace meridian::local_rt {

struct LocalRtPipelineConfig final {
  Config estimator;
  initialization::StaticInitializerOptions static_initializer;
  initialization::DynamicInitializerOptions dynamic_initializer;
  estimator::FixedLagEstimatorOptions fixed_lag;
  lidar::ScanToMapOptions scan_to_map;
  lidar::ScanPreprocessorOptions scan_preprocessor;
  std::size_t imu_queue_capacity{};
  std::size_t lidar_queue_capacity{};
};

// The single ROS-free writer for this first local slice. ROS conversion threads
// transfer owned observations into bounded queues; this worker owns temporal
// ordering, initialization, and post-initialization preintegration records.
class LocalRtPipeline final {
public:
  LocalRtPipeline(LocalRtPipelineConfig config, core::DebugSink& debug_sink);
  ~LocalRtPipeline();

  LocalRtPipeline(const LocalRtPipeline&) = delete;
  LocalRtPipeline& operator=(const LocalRtPipeline&) = delete;

  void submit(core::ImuSample sample);
  void submit(core::LidarSweep sweep);

  // Stops admission and drains every observation that has complete IMU
  // support. Safe to call more than once.
  void stopAndDrain() noexcept;

  [[nodiscard]] std::optional<core::InitializationResult> initializationResult() const;

private:
  void workerLoop() noexcept;
  void processImu(core::ImuSample sample);
  [[nodiscard]] bool processLidar(const core::LidarSweep& sweep);
  void acceptInitialization(const core::InitializationResult& result, const std::string& reason,
                            std::span<const lidar::Point3d> anchor_points = {});
  void recordInitialization(core::TimeNs time, core::InitializationStatus status,
                            const std::string& reason, const core::InitializationQuality& quality,
                            const std::optional<core::NavigationState>& seed = std::nullopt);
  void recordBootstrap(const initialization::BootstrapPoseSummary& pose);
  void recordPreintegration(const ImuInterval& interval,
                            const CombinedPreintegration& preintegration, core::StateId from,
                            core::StateId to);
  void recordTiming(core::TimeNs time, std::optional<core::StateId> state_id, std::string stage,
                    std::string parent_stage, std::int64_t duration_ns);
  void recordTrajectory(const core::NavigationState& state, core::LocalTrajectoryKind kind,
                        std::uint64_t revision);
  void recordRegistrationMap(core::TimeNs time, core::StateId state_id);
  void recordEstimator(const estimator::FixedLagUpdate& update, core::StateId candidate_id,
                       core::TimeNs time, const lidar::ScanPreprocessorStats& preprocessing);
  void recordTerminalTrajectory() noexcept;

  LocalRtPipelineConfig config_;
  core::DebugSink& debug_sink_;
  ImuBuffer imu_buffer_;
  GtsamCombinedPreintegrator preintegrator_;
  ImuPropagator propagator_;
  lidar::ScanPreprocessor scan_preprocessor_;
  estimator::FixedLagEstimator estimator_;
  std::unique_ptr<initialization::StaticInitializer> static_initializer_;
  std::unique_ptr<initialization::DynamicInitializer> dynamic_initializer_;

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<core::ImuSample> imu_queue_;
  std::deque<core::LidarSweep> lidar_queue_;
  bool accepting_{true};
  bool stopped_{};
  std::thread worker_;

  std::optional<core::InitializationResult> initialization_result_;
  std::optional<core::TimeNs> last_preintegration_time_;
  core::StateId last_state_id_{1U};
  core::InitializationStatus last_reported_status_{core::InitializationStatus::kCollecting};
  std::string last_reported_reason_;
  std::uint64_t estimator_revision_{};
  bool terminal_trajectory_recorded_{};
};

}  // namespace meridian::local_rt
