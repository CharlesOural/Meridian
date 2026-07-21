#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/ids.hpp"
#include "meridian/core/initialization.hpp"
#include "meridian/core/observations.hpp"
#include "meridian/core/time.hpp"

namespace meridian::core {

struct ImuAcceptedEvent final {
  MeasurementId measurement_id;
  TimeNs measurement_time;
  Vec3d angular_velocity_rad_s;
  Vec3d specific_force_m_s2;
  std::int64_t arrival_steady_ns{};
  std::int64_t conversion_duration_ns{};
};

struct LidarAcceptedEvent final {
  MeasurementId measurement_id;
  TimeNs measurement_time;
  std::int64_t acquisition_duration_ns{};
  std::uint64_t source_points{};
  std::uint64_t accepted_points{};
  std::uint64_t nonfinite_xyz_points{};
  std::uint64_t zero_xyz_points{};
  std::uint64_t flattened_time_regressions{};
  std::int64_t arrival_steady_ns{};
  std::int64_t conversion_duration_ns{};
  std::uint64_t decode_queue_depth{};
};

struct LidarPreviewPoint final {
  float x{};
  float y{};
  float z{};
};

struct LidarPreviewEvent final {
  MeasurementId measurement_id;
  TimeNs measurement_time;
  std::shared_ptr<const std::vector<LidarPreviewPoint>> points;
};

struct IngressFailureEvent final {
  std::string sensor_id;
  MeasurementId measurement_id;
  std::optional<TimeNs> measurement_time;
  std::int64_t arrival_steady_ns{};
  std::int64_t conversion_duration_ns{};
  std::string error_code;
  std::string field;
  std::string detail;
};

struct PreintegrationEvent final {
  StateId from_state;
  StateId to_state;
  TimeRange support;
  std::uint64_t source_sample_count{};
  std::uint64_t integration_segment_count{};
  std::int64_t maximum_source_gap_ns{};
  Vec3d delta_rotation_vector_rad;
  Vec3d delta_velocity_m_s;
  Vec3d delta_position_m;
  std::string backend;
};

struct InitializationEvent final {
  TimeNs event_time;
  InitializationMode mode;
  InitializationStatus status;
  std::string reason;
  InitializationQuality quality;
  std::optional<NavigationState> accepted_seed;
};

struct BootstrapPoseEvent final {
  MeasurementId measurement_id;
  TimeNs measurement_time;
  Pose3d odom_from_lidar;
  std::uint64_t source_point_count{};
  std::uint64_t correspondence_count{};
  double point_rmse_m{};
  double hessian_condition_number{};
  bool accepted{};
};

enum class LocalTrajectoryKind : std::uint8_t {
  kOnline,
  kFinalized,
  kTerminal,
};

// One accepted local state. Online rows expose the causal newest estimate;
// finalized and terminal rows together form the non-duplicated trajectory used
// for post-run ATE/RPE evaluation.
struct LocalTrajectoryEvent final {
  NavigationState state;
  Pose3d odom_from_base;
  std::uint64_t estimator_revision{};
  LocalTrajectoryKind kind{LocalTrajectoryKind::kOnline};
};

// Observer-only, odom-frame snapshot of the complete registration map used by
// local tracking. The immutable shared payload keeps producer-side publication
// cheap while the asynchronous recorder owns all Rerun conversion and IO.
struct LocalRegistrationMapEvent final {
  TimeNs event_time;
  StateId state_id;
  std::uint64_t estimator_revision{};
  std::shared_ptr<const std::vector<LidarPreviewPoint>> points;
};

// Fixed stage names make timings directly plottable and aggregatable without
// giving debug IO any control over the estimator.
struct StageTimingEvent final {
  TimeNs event_time;
  std::optional<StateId> state_id;
  std::string stage;
  std::string parent_stage;
  std::int64_t duration_ns{};
};

struct LocalEstimatorEvent final {
  StateId state_id;
  TimeNs event_time;
  std::uint64_t estimator_revision{};
  std::string outcome;
  std::uint64_t active_state_count{};
  std::uint64_t imu_factor_count{};
  std::uint64_t lidar_batch_count{};
  std::uint64_t active_lidar_rows{};
  std::uint64_t finalized_lidar_rows{};
  std::uint64_t finalized_map_points{};
  std::uint64_t selected_active_owners{};
  std::uint64_t registration_correspondences{};
  std::uint64_t marginal_prior_rank{};
  double registration_rmse_m{};
  double initial_cost{};
  double final_cost{};
  double pose_correction_translation_m{};
  double pose_correction_rotation_rad{};
  bool accepted{};

  // Append-only scan-to-map telemetry. The RRD quality row keeps every
  // legacy scalar above in its original position so older analyzers can
  // continue reading indices [0, 14].
  std::uint64_t prepared_target_points{};
  std::uint64_t prepared_source_points{};
  std::uint64_t association_pass_count{};
  std::uint64_t association_input_points{};
  std::uint64_t association_rows_before_cap{};
  std::uint64_t registration_iterations{};
  std::uint64_t live_query_voxel_probes{};
  std::uint64_t finalized_query_voxel_probes{};
  std::uint64_t reassociated_rows{};
  std::uint64_t rejected_stale_rows{};
};

// Debug sinks are observers only: calls must not throw and must not mutate the
// localization path. Heavy preview geometry is built only when requested.
class DebugSink {
public:
  virtual ~DebugSink() = default;

  [[nodiscard]] virtual bool wantsLidarPreview() const noexcept = 0;
  virtual void record(const ImuAcceptedEvent& event) noexcept = 0;
  virtual void record(const LidarAcceptedEvent& event) noexcept = 0;
  virtual void record(const LidarPreviewEvent& event) noexcept = 0;
  virtual void record(const IngressFailureEvent& event) noexcept = 0;
  virtual void record(const PreintegrationEvent& event) noexcept = 0;
  virtual void record(const InitializationEvent& event) noexcept = 0;
  virtual void record(const BootstrapPoseEvent& event) noexcept = 0;
  virtual void record(const LocalTrajectoryEvent&) noexcept {}
  virtual void record(const LocalRegistrationMapEvent&) noexcept {}
  virtual void record(const StageTimingEvent&) noexcept {}
  virtual void record(const LocalEstimatorEvent&) noexcept {}
  [[nodiscard]] virtual std::uint64_t droppedEvents() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t logErrors() const noexcept = 0;
};

class NullDebugSink final : public DebugSink {
public:
  [[nodiscard]] bool wantsLidarPreview() const noexcept override { return false; }
  void record(const ImuAcceptedEvent&) noexcept override {}
  void record(const LidarAcceptedEvent&) noexcept override {}
  void record(const LidarPreviewEvent&) noexcept override {}
  void record(const IngressFailureEvent&) noexcept override {}
  void record(const PreintegrationEvent&) noexcept override {}
  void record(const InitializationEvent&) noexcept override {}
  void record(const BootstrapPoseEvent&) noexcept override {}
  void record(const LocalTrajectoryEvent&) noexcept override {}
  void record(const LocalRegistrationMapEvent&) noexcept override {}
  void record(const StageTimingEvent&) noexcept override {}
  void record(const LocalEstimatorEvent&) noexcept override {}
  [[nodiscard]] std::uint64_t droppedEvents() const noexcept override { return 0U; }
  [[nodiscard]] std::uint64_t logErrors() const noexcept override { return 0U; }
};

}  // namespace meridian::core
