#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/factor_batch_api.hpp"
#include "meridian/core/result.hpp"

namespace meridian::local {

inline constexpr std::size_t kMaximumConfiguredSensorHealthInstances = 32U;

struct SensorHealthPolicyConfig {
  std::size_t consecutive_failures_to_suspect{1U};
  std::size_t consecutive_failures_to_failed{3U};
  std::size_t recovery_good_shadow_results{3U};
  core::Duration suspect_after_no_result{500'000'000LL};
  core::Duration failed_after_no_result{2'000'000'000LL};

  bool operator==(const SensorHealthPolicyConfig&) const = default;
};

struct SensorHealthRegistryConfig {
  // The registry never discovers sensors dynamically. This bounded set is
  // canonicalized once at construction and every later update must name one
  // of its members.
  std::vector<core::SensorInstanceId> sensors;
  core::FusionTime initialized_at;
  SensorHealthPolicyConfig policy;
};

enum class SensorBatchHealthResult : std::uint8_t {
  Good,
  Failure,
};

enum class SensorBatchEvaluationMode : std::uint8_t {
  Primary,
  Shadow,
};

// One completed per-sensor batch assessment. The batch carries the recovery
// epoch under which it was evaluated. In particular, the first good shadow
// batch while Failed carries Failed/E; the registry alone owns the transition
// to Recovering/E+1.
struct SensorHealthBatchObservation {
  core::SensorInstanceId sensor;
  core::FactorBatchId batch_id;
  core::SensorRecoveryEpoch recovery_epoch;
  core::FusionTime assessed_at;
  SensorBatchHealthResult result{SensorBatchHealthResult::Failure};
  SensorBatchEvaluationMode mode{SensorBatchEvaluationMode::Primary};
};

struct SensorHealthRegistrySnapshot {
  core::SensorHealthSnapshot health;
  std::optional<core::FactorBatchId> last_batch;
  core::FusionTime last_observation_at;
  std::size_t consecutive_failures{};
  std::size_t recovery_good_shadow_results{};
};

struct SensorHealthUpdate {
  core::FactorBatchId batch_id;
  core::SensorHealthSnapshot before;
  core::SensorHealthSnapshot after;
  bool transitioned{};
  std::size_t consecutive_failures{};
  std::size_t recovery_good_shadow_results{};
};

struct SensorHealthTimeoutAssessment {
  core::SensorHealthSnapshot before;
  core::SensorHealthSnapshot after;
  bool transitioned{};
  bool timed_out{};
};

enum class SensorHealthRegistryErrorCode {
  InvalidConfig,
  InvalidConfiguredSensor,
  DuplicateConfiguredSensor,
  UnknownSensor,
  InvalidBatch,
  InvalidObservation,
  InvalidRecoveryEpoch,
  StaleBatch,
  StaleAssessmentTime,
  StaleRecoveryEpoch,
  FutureRecoveryEpoch,
  RecoveryEpochExhausted,
  TransitionSequenceExhausted,
};

struct SensorHealthRegistryError {
  SensorHealthRegistryErrorCode code{SensorHealthRegistryErrorCode::InvalidConfig};
  std::optional<core::SensorInstanceId> sensor;
  std::optional<core::FactorBatchId> batch_id;
  std::optional<core::SensorRecoveryEpoch> expected_recovery_epoch;
  std::optional<core::SensorRecoveryEpoch> received_recovery_epoch;
  std::string detail;
};

namespace detail {
struct SensorHealthRegistryState;
}  // namespace detail

// Single-writer, ROS-free health owner. Updates are validated and applied to
// a private candidate before commit, so stale identities and epoch overflow
// cannot partially mutate a sensor. No update ever touches another sensor.
class SensorHealthRegistry {
public:
  [[nodiscard]] static core::Result<SensorHealthRegistry, SensorHealthRegistryError> create(
      SensorHealthRegistryConfig config);

  ~SensorHealthRegistry();
  SensorHealthRegistry(SensorHealthRegistry&&) noexcept;
  SensorHealthRegistry& operator=(SensorHealthRegistry&&) noexcept;
  SensorHealthRegistry(const SensorHealthRegistry&) = delete;
  SensorHealthRegistry& operator=(const SensorHealthRegistry&) = delete;

  [[nodiscard]] core::Result<SensorHealthUpdate, SensorHealthRegistryError> observe(
      const SensorHealthBatchObservation& observation);

  // Timeouts are assessed explicitly per sensor. Active sensors first become
  // Suspect after the short timeout; Suspect and Recovering sensors become
  // Failed after the long timeout. A check never refreshes last_observation_at.
  [[nodiscard]] core::Result<SensorHealthTimeoutAssessment, SensorHealthRegistryError>
  assessTimeout(core::SensorInstanceId sensor, core::SensorRecoveryEpoch recovery_epoch,
                core::FusionTime assessed_at);

  [[nodiscard]] core::Result<SensorHealthRegistrySnapshot, SensorHealthRegistryError> snapshot(
      core::SensorInstanceId sensor) const;
  [[nodiscard]] std::vector<SensorHealthRegistrySnapshot> snapshots() const;
  [[nodiscard]] std::size_t size() const noexcept;

private:
  explicit SensorHealthRegistry(std::unique_ptr<detail::SensorHealthRegistryState> state);

  std::unique_ptr<detail::SensorHealthRegistryState> state_;
};

}  // namespace meridian::local
