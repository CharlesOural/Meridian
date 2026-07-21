#include "meridian/apps/rerun_debug_sink.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <rerun.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace meridian::apps {
namespace {

template <typename... Callables>
struct Overloaded : Callables... {
  using Callables::operator()...;
};
template <typename... Callables>
Overloaded(Callables...) -> Overloaded<Callables...>;

void requireSuccess(const rerun::Error& error, const std::string& operation) {
  if (error.is_err()) {
    throw std::runtime_error(operation + ": " + error.description);
  }
}

template <typename Id>
std::int64_t sequence(Id id) noexcept {
  constexpr auto kMaximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  return static_cast<std::int64_t>(std::min(id.value(), kMaximum));
}

rerun::Transform3D toTransform3D(const core::Pose3d& pose) {
  const auto& translation = pose.translation();
  const auto& rotation = pose.rotation();
  return rerun::Transform3D::from_translation_rotation(
      rerun::components::Translation3D(static_cast<float>(translation.x),
                                       static_cast<float>(translation.y),
                                       static_cast<float>(translation.z)),
      rerun::Rotation3D(rerun::datatypes::Quaternion::from_wxyz(
          static_cast<float>(rotation.w()), static_cast<float>(rotation.x()),
          static_cast<float>(rotation.y()), static_cast<float>(rotation.z()))));
}

rerun::Scalars poseScalars(const core::Pose3d& pose) {
  const auto& translation = pose.translation();
  const auto& rotation = pose.rotation();
  return rerun::Scalars({translation.x, translation.y, translation.z, rotation.x(), rotation.y(),
                         rotation.z(), rotation.w()});
}

std::string entityComponent(std::string value) {
  for (char& character : value) {
    const unsigned char byte = static_cast<unsigned char>(character);
    if (!std::isalnum(byte) && character != '_' && character != '-' && character != '.') {
      character = '_';
    }
  }
  return value.empty() ? std::string("unnamed") : value;
}

}  // namespace

RerunDebugSink::RerunDebugSink(Options options) : options_(std::move(options)) {
  if (options_.output_path.empty()) {
    throw std::invalid_argument("Rerun output path must not be empty");
  }
  if (options_.queue_capacity == 0U) {
    throw std::invalid_argument("Rerun queue capacity must be greater than zero");
  }
  if (options_.preintegration_debug_period.count() <= 0) {
    throw std::invalid_argument("Rerun preintegration debug period must be positive");
  }

  auto recording = std::make_unique<rerun::RecordingStream>("meridian");
  requireSuccess(recording->save(options_.output_path.string()), "open Rerun recording");
  worker_ = std::thread(
      [this, recording = std::move(recording)]() mutable { workerLoop(std::move(recording)); });
}

RerunDebugSink::~RerunDebugSink() {
  shutdown();
}

void RerunDebugSink::record(const core::ImuAcceptedEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::LidarAcceptedEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::LidarPreviewEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::IngressFailureEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::PreintegrationEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::InitializationEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::BootstrapPoseEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::LocalTrajectoryEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::LocalRegistrationMapEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::StageTimingEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::record(const core::LocalEstimatorEvent& event) noexcept {
  try {
    enqueue(Event(event));
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

std::uint64_t RerunDebugSink::droppedEvents() const noexcept {
  return dropped_events_.load(std::memory_order_relaxed);
}

std::uint64_t RerunDebugSink::logErrors() const noexcept {
  return log_errors_.load(std::memory_order_relaxed);
}

void RerunDebugSink::enqueue(Event event) noexcept {
  try {
    std::lock_guard lock(mutex_);
    if (closing_ || queue_.size() >= options_.queue_capacity) {
      dropped_events_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }
    queue_.push_back(std::move(event));
    queue_condition_.notify_one();
  } catch (...) {
    dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

void RerunDebugSink::shutdown() noexcept {
  {
    std::lock_guard lock(mutex_);
    closing_ = true;
  }
  queue_condition_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void RerunDebugSink::workerLoop(std::unique_ptr<rerun::RecordingStream> recording) noexcept {
  auto noteError = [this]() noexcept { log_errors_.fetch_add(1U, std::memory_order_relaxed); };
  auto note = [this](const rerun::Error& error) noexcept {
    if (error.is_err()) {
      log_errors_.fetch_add(1U, std::memory_order_relaxed);
    }
  };

  try {
    bool preintegration_backend_logged = false;
    std::optional<core::TimeNs> next_preintegration_debug_time;
    auto setMeasurementContext = [&recording](core::TimeNs time, core::MeasurementId id) {
      recording->disable_timeline("state_id");
      recording->disable_timeline("estimator_revision");
      recording->set_time_timestamp_nanos_since_epoch("sensor_time", time.count());
      recording->set_time_sequence("measurement_id", sequence(id));
    };
    auto setStateContext = [&recording](core::TimeNs time, core::StateId id) {
      recording->disable_timeline("measurement_id");
      recording->disable_timeline("estimator_revision");
      recording->set_time_timestamp_nanos_since_epoch("sensor_time", time.count());
      recording->set_time_sequence("state_id", sequence(id));
    };
    auto setEventTimeContext = [&recording](core::TimeNs time) {
      recording->disable_timeline("measurement_id");
      recording->disable_timeline("state_id");
      recording->disable_timeline("estimator_revision");
      recording->set_time_timestamp_nanos_since_epoch("sensor_time", time.count());
    };
    auto logOptionalScalar = [&recording, &note](const char* entity_path,
                                                 const std::optional<double>& value) {
      if (value.has_value()) {
        note(recording->try_log(entity_path, rerun::Scalars(*value)));
      }
    };

    while (true) {
      std::optional<Event> event;
      {
        std::unique_lock lock(mutex_);
        queue_condition_.wait(lock, [this] { return closing_ || !queue_.empty(); });
        if (queue_.empty()) {
          if (closing_) {
            break;
          }
          continue;
        }
        event.emplace(std::move(queue_.front()));
        queue_.pop_front();
      }

      try {
        std::visit(
            Overloaded{
                [&](const core::ImuAcceptedEvent& item) {
                  setMeasurementContext(item.measurement_time, item.measurement_id);
                  note(recording->try_log(
                      "/sensors/imu/sample",
                      rerun::Scalars({item.specific_force_m_s2.x, item.specific_force_m_s2.y,
                                      item.specific_force_m_s2.z, item.angular_velocity_rad_s.x,
                                      item.angular_velocity_rad_s.y, item.angular_velocity_rad_s.z,
                                      static_cast<double>(item.arrival_steady_ns),
                                      static_cast<double>(item.conversion_duration_ns)})));
                },
                [&](const core::LidarAcceptedEvent& item) {
                  setMeasurementContext(item.measurement_time, item.measurement_id);
                  note(recording->try_log(
                      "/sensors/lidar/metadata",
                      rerun::Scalars({static_cast<double>(item.source_points),
                                      static_cast<double>(item.accepted_points),
                                      static_cast<double>(item.nonfinite_xyz_points),
                                      static_cast<double>(item.zero_xyz_points),
                                      static_cast<double>(item.flattened_time_regressions),
                                      static_cast<double>(item.arrival_steady_ns),
                                      static_cast<double>(item.conversion_duration_ns),
                                      static_cast<double>(item.decode_queue_depth),
                                      static_cast<double>(item.acquisition_duration_ns)})));
                },
                [&](const core::LidarPreviewEvent& item) {
                  setMeasurementContext(item.measurement_time, item.measurement_id);
                  std::vector<std::array<float, 3>> positions;
                  if (item.points) {
                    positions.reserve(item.points->size());
                    for (const auto& point : *item.points) {
                      positions.push_back({point.x, point.y, point.z});
                    }
                  }
                  note(recording->try_log("/sensors/lidar/preview",
                                          rerun::Points3D(std::move(positions)).with_radii(0.02F)));
                },
                [&](const core::IngressFailureEvent& item) {
                  recording->disable_timeline("state_id");
                  recording->set_time_sequence("measurement_id", sequence(item.measurement_id));
                  if (item.measurement_time.has_value()) {
                    recording->set_time_timestamp_nanos_since_epoch("sensor_time",
                                                                    item.measurement_time->count());
                  } else {
                    recording->disable_timeline("sensor_time");
                  }
                  note(recording->try_log("/runtime/ingress_failure",
                                          rerun::TextLog(item.sensor_id + ":" + item.error_code +
                                                         ":" + item.field + ":" + item.detail)));
                },
                [&](const core::PreintegrationEvent& item) {
                  if (next_preintegration_debug_time.has_value() &&
                      item.support.end() < *next_preintegration_debug_time) {
                    return;
                  }
                  setStateContext(item.support.end(), item.to_state);
                  note(recording->try_log(
                      "/local_rt/preintegration/quality",
                      rerun::Scalars({static_cast<double>(item.from_state.value()),
                                      static_cast<double>(item.to_state.value()),
                                      static_cast<double>(item.support.durationNs().value_or(0)),
                                      static_cast<double>(item.source_sample_count),
                                      static_cast<double>(item.integration_segment_count),
                                      static_cast<double>(item.maximum_source_gap_ns),
                                      item.delta_rotation_vector_rad.x,
                                      item.delta_rotation_vector_rad.y,
                                      item.delta_rotation_vector_rad.z, item.delta_velocity_m_s.x,
                                      item.delta_velocity_m_s.y, item.delta_velocity_m_s.z,
                                      item.delta_position_m.x, item.delta_position_m.y,
                                      item.delta_position_m.z})));
                  if (!preintegration_backend_logged) {
                    note(recording->try_log("/local_rt/preintegration/backend",
                                            rerun::TextLog(item.backend)));
                    preintegration_backend_logged = true;
                  }
                  core::TimeNs deadline =
                      next_preintegration_debug_time.value_or(item.support.end());
                  do {
                    const auto advanced = core::TimeNs::checkedAdd(
                        deadline, options_.preintegration_debug_period.count());
                    if (!advanced.has_value()) {
                      next_preintegration_debug_time.reset();
                      return;
                    }
                    deadline = *advanced;
                    next_preintegration_debug_time = deadline;
                  } while (deadline <= item.support.end());
                },
                [&](const core::InitializationEvent& item) {
                  setEventTimeContext(item.event_time);
                  std::string status = "mode=";
                  status += core::toString(item.mode);
                  status += " status=";
                  status += core::toString(item.status);
                  if (!item.reason.empty()) {
                    status += " reason=";
                    status += item.reason;
                  }
                  note(recording->try_log("/local_rt/initialization/status",
                                          rerun::TextLog(std::move(status))));
                  note(recording->try_log(
                      "/local_rt/initialization/counts",
                      rerun::Scalars(
                          {static_cast<double>(item.quality.imu_sample_count),
                           static_cast<double>(item.quality.lidar_sweep_count),
                           static_cast<double>(item.quality.fitted_transition_count),
                           static_cast<double>(item.quality.rejected_transition_count)})));
                  note(recording->try_log(
                      "/local_rt/initialization/quality/all_required_gates_passed",
                      rerun::Scalars(item.quality.all_required_gates_passed ? 1.0 : 0.0)));
                  logOptionalScalar(
                      "/local_rt/initialization/quality/registration_min_singular_value",
                      item.quality.registration_min_singular_value);
                  logOptionalScalar(
                      "/local_rt/initialization/quality/registration_condition_number",
                      item.quality.registration_condition_number);
                  logOptionalScalar("/local_rt/initialization/quality/gyro_bias_min_singular_value",
                                    item.quality.gyro_bias_min_singular_value);
                  logOptionalScalar("/local_rt/initialization/quality/gyro_bias_condition_number",
                                    item.quality.gyro_bias_condition_number);
                  logOptionalScalar("/local_rt/initialization/quality/gravity_min_singular_value",
                                    item.quality.gravity_min_singular_value);
                  logOptionalScalar("/local_rt/initialization/quality/gravity_condition_number",
                                    item.quality.gravity_condition_number);
                  logOptionalScalar("/local_rt/initialization/quality/raw_gravity_magnitude_m_s2",
                                    item.quality.raw_gravity_magnitude_m_s2);
                  logOptionalScalar(
                      "/local_rt/initialization/quality/gyro_bias_correction_norm_rad_s",
                      item.quality.gyro_bias_correction_norm_rad_s);
                  logOptionalScalar("/local_rt/initialization/quality/alignment_residual_rms",
                                    item.quality.alignment_residual_rms);
                  logOptionalScalar("/local_rt/initialization/quality/held_out_rotation_error_rad",
                                    item.quality.held_out_rotation_error_rad);
                  logOptionalScalar("/local_rt/initialization/quality/held_out_translation_error_m",
                                    item.quality.held_out_translation_error_m);
                  logOptionalScalar(
                      "/local_rt/initialization/quality/refinement_rotation_change_rad",
                      item.quality.refinement_rotation_change_rad);
                  logOptionalScalar(
                      "/local_rt/initialization/quality/refinement_translation_change_m",
                      item.quality.refinement_translation_change_m);

                  if (item.accepted_seed.has_value()) {
                    const auto& seed = *item.accepted_seed;
                    setStateContext(seed.time(), seed.id());
                    note(recording->try_log("/local_rt/initialization/accepted_seed/odom_from_imu",
                                            toTransform3D(seed.odomFromImu())));
                    const auto& velocity = seed.velocityOdomMS();
                    note(recording->try_log(
                        "/local_rt/initialization/accepted_seed/velocity_odom_m_s",
                        rerun::Scalars({velocity.x, velocity.y, velocity.z})));
                    const auto& gyroscope_bias = seed.imuBias().gyroscopeRadS();
                    note(recording->try_log(
                        "/local_rt/initialization/accepted_seed/gyro_bias_rad_s",
                        rerun::Scalars({gyroscope_bias.x, gyroscope_bias.y, gyroscope_bias.z})));
                    const auto& accelerometer_bias = seed.imuBias().accelerometerMS2();
                    note(recording->try_log(
                        "/local_rt/initialization/accepted_seed/accel_bias_m_s2",
                        rerun::Scalars(
                            {accelerometer_bias.x, accelerometer_bias.y, accelerometer_bias.z})));
                  }
                },
                [&](const core::BootstrapPoseEvent& item) {
                  setMeasurementContext(item.measurement_time, item.measurement_id);
                  note(recording->try_log("/local_rt/bootstrap/odom_from_lidar",
                                          toTransform3D(item.odom_from_lidar)));
                  note(recording->try_log(
                      "/local_rt/bootstrap/quality",
                      rerun::Scalars({static_cast<double>(item.source_point_count),
                                      static_cast<double>(item.correspondence_count),
                                      item.point_rmse_m, item.hessian_condition_number,
                                      item.accepted ? 1.0 : 0.0})));
                },
                [&](const core::LocalTrajectoryEvent& item) {
                  setStateContext(item.state.time(), item.state.id());
                  recording->set_time_sequence(
                      "estimator_revision",
                      static_cast<std::int64_t>(std::min<std::uint64_t>(
                          item.estimator_revision,
                          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))));
                  const bool causal = item.kind == core::LocalTrajectoryKind::kOnline;
                  const std::string scalar_path = causal
                                                      ? "/local_rt/trajectory/causal/pose_odom_base"
                                                      : "/local_rt/trajectory/final/pose_odom_base";
                  note(recording->try_log(scalar_path, poseScalars(item.odom_from_base)));
                  note(recording->try_log(scalar_path + "_transform",
                                          toTransform3D(item.odom_from_base)));
                },
                [&](const core::LocalRegistrationMapEvent& item) {
                  setStateContext(item.event_time, item.state_id);
                  recording->set_time_sequence(
                      "estimator_revision",
                      static_cast<std::int64_t>(std::min<std::uint64_t>(
                          item.estimator_revision,
                          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))));
                  std::vector<std::array<float, 3>> positions;
                  if (item.points) {
                    positions.reserve(item.points->size());
                    for (const auto& point : *item.points) {
                      positions.push_back({point.x, point.y, point.z});
                    }
                  }
                  note(recording->try_log("/local_rt/map/registration",
                                          rerun::Points3D(std::move(positions))));
                },
                [&](const core::StageTimingEvent& item) {
                  if (item.state_id.has_value()) {
                    setStateContext(item.event_time, *item.state_id);
                  } else {
                    setEventTimeContext(item.event_time);
                  }
                  note(recording->try_log("/local_rt/timing/" + entityComponent(item.stage),
                                          rerun::Scalars(static_cast<double>(item.duration_ns))));
                },
                [&](const core::LocalEstimatorEvent& item) {
                  setStateContext(item.event_time, item.state_id);
                  recording->set_time_sequence(
                      "estimator_revision",
                      static_cast<std::int64_t>(std::min<std::uint64_t>(
                          item.estimator_revision,
                          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))));
                  note(recording->try_log("/local_rt/estimator/outcome",
                                          rerun::TextLog(item.outcome)));
                  note(recording->try_log(
                      "/local_rt/estimator/quality",
                      rerun::Scalars(
                          {static_cast<double>(item.active_state_count),
                           static_cast<double>(item.imu_factor_count),
                           static_cast<double>(item.lidar_batch_count),
                           static_cast<double>(item.active_lidar_rows),
                           static_cast<double>(item.finalized_lidar_rows),
                           static_cast<double>(item.finalized_map_points),
                           static_cast<double>(item.selected_active_owners),
                           static_cast<double>(item.registration_correspondences),
                           static_cast<double>(item.marginal_prior_rank), item.registration_rmse_m,
                           item.initial_cost, item.final_cost, item.pose_correction_translation_m,
                           item.pose_correction_rotation_rad, item.accepted ? 1.0 : 0.0,
                           static_cast<double>(item.prepared_target_points),
                           static_cast<double>(item.prepared_source_points),
                           static_cast<double>(item.association_pass_count),
                           static_cast<double>(item.association_input_points),
                           static_cast<double>(item.association_rows_before_cap),
                           static_cast<double>(item.registration_iterations),
                           static_cast<double>(item.live_query_voxel_probes),
                           static_cast<double>(item.finalized_query_voxel_probes),
                           static_cast<double>(item.reassociated_rows),
                           static_cast<double>(item.rejected_stale_rows)})));
                },
            },
            *event);
      } catch (...) {
        noteError();
      }
    }
  } catch (...) {
    noteError();
  }

  try {
    recording->disable_timeline("sensor_time");
    recording->disable_timeline("measurement_id");
    recording->disable_timeline("state_id");
    recording->disable_timeline("estimator_revision");
    note(recording->try_log(
        "/run/debug_events_dropped",
        rerun::Scalars(static_cast<double>(dropped_events_.load(std::memory_order_relaxed)))));
    // Log this last so it includes any error produced while writing the drop
    // counter. The process exit status remains authoritative if this write or
    // the following flush itself fails.
    note(recording->try_log(
        "/run/debug_log_errors",
        rerun::Scalars(static_cast<double>(log_errors_.load(std::memory_order_relaxed)))));
  } catch (...) {
    noteError();
  }

  try {
    note(recording->flush_blocking(30.0F));
  } catch (...) {
    noteError();
  }
}

}  // namespace meridian::apps
