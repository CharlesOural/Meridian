#include "meridian/local/sensor_health.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <utility>

namespace meridian::local {
namespace detail {

struct SensorHealthRegistryEntry {
  SensorHealthRegistrySnapshot snapshot;
};

struct SensorHealthRegistryState {
  explicit SensorHealthRegistryState(SensorHealthPolicyConfig configured_policy)
      : policy(std::move(configured_policy)) {}

  SensorHealthPolicyConfig policy;
  std::map<core::SensorInstanceId, SensorHealthRegistryEntry> sensors;
};

}  // namespace detail
namespace {

using Result = core::Result<SensorHealthRegistry, SensorHealthRegistryError>;

[[nodiscard]] SensorHealthRegistryError error(
    SensorHealthRegistryErrorCode code, std::string detail,
    std::optional<core::SensorInstanceId> sensor = std::nullopt,
    std::optional<core::FactorBatchId> batch_id = std::nullopt,
    std::optional<core::SensorRecoveryEpoch> expected_recovery_epoch = std::nullopt,
    std::optional<core::SensorRecoveryEpoch> received_recovery_epoch = std::nullopt) {
  return SensorHealthRegistryError{
      code, sensor, batch_id, expected_recovery_epoch, received_recovery_epoch, std::move(detail)};
}

[[nodiscard]] bool validPolicy(const SensorHealthPolicyConfig& policy) noexcept {
  return policy.consecutive_failures_to_suspect > 0U &&
         policy.consecutive_failures_to_failed > policy.consecutive_failures_to_suspect &&
         policy.recovery_good_shadow_results > 0U &&
         policy.suspect_after_no_result.nanoseconds > 0LL &&
         policy.failed_after_no_result.nanoseconds > policy.suspect_after_no_result.nanoseconds;
}

[[nodiscard]] bool validResult(SensorBatchHealthResult result) noexcept {
  switch (result) {
    case SensorBatchHealthResult::Good:
    case SensorBatchHealthResult::Failure:
      return true;
  }
  return false;
}

[[nodiscard]] bool validMode(SensorBatchEvaluationMode mode) noexcept {
  switch (mode) {
    case SensorBatchEvaluationMode::Primary:
    case SensorBatchEvaluationMode::Shadow:
      return true;
  }
  return false;
}

[[nodiscard]] bool transitionSequenceAvailable(const core::SensorHealthSnapshot& health) noexcept {
  return health.transition_sequence < core::kInvalidSensorHealthTransitionSequence - 1U;
}

[[nodiscard]] bool recoveryEpochAvailable(const core::SensorHealthSnapshot& health) noexcept {
  return health.recovery_epoch.value() < core::SensorRecoveryEpoch::kInvalidValue - 1U;
}

void applyStateTransition(core::SensorHealthSnapshot* health, core::SensorHealthState state) {
  health->state = state;
  ++health->transition_sequence;
}

[[nodiscard]] SensorHealthRegistrySnapshot makeInitialSnapshot(core::SensorInstanceId sensor,
                                                               core::FusionTime initialized_at) {
  SensorHealthRegistrySnapshot snapshot;
  snapshot.health.sensor = sensor;
  snapshot.health.state = core::SensorHealthState::Active;
  snapshot.health.recovery_epoch = core::SensorRecoveryEpoch{0U};
  snapshot.health.transition_sequence = 0U;
  snapshot.health.assessed_at = initialized_at;
  snapshot.last_observation_at = initialized_at;
  return snapshot;
}

[[nodiscard]] std::optional<SensorHealthRegistryError> validateEpoch(
    const SensorHealthRegistrySnapshot& current, core::SensorInstanceId sensor,
    std::optional<core::FactorBatchId> batch_id, core::SensorRecoveryEpoch received) {
  if (!received.valid()) {
    return error(SensorHealthRegistryErrorCode::InvalidRecoveryEpoch,
                 "sensor result carries an invalid recovery epoch", sensor, batch_id,
                 current.health.recovery_epoch, received);
  }
  if (received < current.health.recovery_epoch) {
    return error(SensorHealthRegistryErrorCode::StaleRecoveryEpoch,
                 "sensor result belongs to an older recovery epoch", sensor, batch_id,
                 current.health.recovery_epoch, received);
  }
  if (received > current.health.recovery_epoch) {
    return error(SensorHealthRegistryErrorCode::FutureRecoveryEpoch,
                 "sensor result claims a recovery epoch the registry has not opened", sensor,
                 batch_id, current.health.recovery_epoch, received);
  }
  return std::nullopt;
}

[[nodiscard]] long double elapsedNanoseconds(core::FusionTime later,
                                             core::FusionTime earlier) noexcept {
  return static_cast<long double>(later.nanoseconds) -
         static_cast<long double>(earlier.nanoseconds);
}

}  // namespace

core::Result<SensorHealthRegistry, SensorHealthRegistryError> SensorHealthRegistry::create(
    SensorHealthRegistryConfig config) {
  if (!validPolicy(config.policy) || config.sensors.empty() ||
      config.sensors.size() > kMaximumConfiguredSensorHealthInstances) {
    return Result::failure(
        error(SensorHealthRegistryErrorCode::InvalidConfig,
              "sensor health requires a non-empty bounded sensor set, ordered failure thresholds, "
              "and positive ordered timeouts"));
  }
  std::sort(config.sensors.begin(), config.sensors.end());
  for (std::size_t index = 0U; index < config.sensors.size(); ++index) {
    if (!config.sensors[index].valid()) {
      return Result::failure(error(SensorHealthRegistryErrorCode::InvalidConfiguredSensor,
                                   "configured sensor identity is invalid", config.sensors[index]));
    }
    if (index > 0U && config.sensors[index] == config.sensors[index - 1U]) {
      return Result::failure(error(SensorHealthRegistryErrorCode::DuplicateConfiguredSensor,
                                   "configured sensor identities must be unique",
                                   config.sensors[index]));
    }
  }

  auto state = std::make_unique<detail::SensorHealthRegistryState>(std::move(config.policy));
  for (const core::SensorInstanceId sensor : config.sensors) {
    state->sensors.emplace(sensor, detail::SensorHealthRegistryEntry{
                                       makeInitialSnapshot(sensor, config.initialized_at)});
  }
  return Result::success(SensorHealthRegistry{std::move(state)});
}

SensorHealthRegistry::SensorHealthRegistry(std::unique_ptr<detail::SensorHealthRegistryState> state)
    : state_(std::move(state)) {}

SensorHealthRegistry::~SensorHealthRegistry() = default;
SensorHealthRegistry::SensorHealthRegistry(SensorHealthRegistry&&) noexcept = default;
SensorHealthRegistry& SensorHealthRegistry::operator=(SensorHealthRegistry&&) noexcept = default;

core::Result<SensorHealthUpdate, SensorHealthRegistryError> SensorHealthRegistry::observe(
    const SensorHealthBatchObservation& observation) {
  using UpdateResult = core::Result<SensorHealthUpdate, SensorHealthRegistryError>;
  if (!state_) {
    return UpdateResult::failure(error(SensorHealthRegistryErrorCode::InvalidConfig,
                                       "sensor health registry is moved from"));
  }
  const auto found = state_->sensors.find(observation.sensor);
  if (found == state_->sensors.end()) {
    return UpdateResult::failure(error(SensorHealthRegistryErrorCode::UnknownSensor,
                                       "sensor is not in the configured registry set",
                                       observation.sensor, observation.batch_id));
  }
  if (!observation.batch_id.valid()) {
    return UpdateResult::failure(error(SensorHealthRegistryErrorCode::InvalidBatch,
                                       "sensor health observation requires a valid batch identity",
                                       observation.sensor, observation.batch_id));
  }
  if (!validResult(observation.result) || !validMode(observation.mode)) {
    return UpdateResult::failure(error(SensorHealthRegistryErrorCode::InvalidObservation,
                                       "sensor health result or evaluation mode is invalid",
                                       observation.sensor, observation.batch_id));
  }

  const SensorHealthRegistrySnapshot& current = found->second.snapshot;
  if (const auto epoch_error = validateEpoch(current, observation.sensor, observation.batch_id,
                                             observation.recovery_epoch)) {
    return UpdateResult::failure(*epoch_error);
  }
  if (current.last_batch && observation.batch_id <= *current.last_batch) {
    return UpdateResult::failure(error(SensorHealthRegistryErrorCode::StaleBatch,
                                       "sensor batch identity is not newer than its last result",
                                       observation.sensor, observation.batch_id));
  }
  if (observation.assessed_at <= current.health.assessed_at) {
    return UpdateResult::failure(
        error(SensorHealthRegistryErrorCode::StaleAssessmentTime,
              "sensor result assessment time must advance monotonically for that sensor",
              observation.sensor, observation.batch_id));
  }

  SensorHealthRegistrySnapshot candidate = current;
  const core::SensorHealthSnapshot before = current.health;
  const bool good = observation.result == SensorBatchHealthResult::Good;
  const bool shadow = observation.mode == SensorBatchEvaluationMode::Shadow;
  core::SensorHealthState next_state = current.health.state;
  bool increment_recovery_epoch = false;

  switch (current.health.state) {
    case core::SensorHealthState::Active:
    case core::SensorHealthState::Suspect:
      if (good) {
        candidate.consecutive_failures = 0U;
        candidate.recovery_good_shadow_results = 0U;
        next_state = core::SensorHealthState::Active;
      } else {
        candidate.consecutive_failures = std::min(candidate.consecutive_failures + 1U,
                                                  state_->policy.consecutive_failures_to_failed);
        if (candidate.consecutive_failures >= state_->policy.consecutive_failures_to_failed) {
          next_state = core::SensorHealthState::Failed;
        } else if (candidate.consecutive_failures >=
                   state_->policy.consecutive_failures_to_suspect) {
          next_state = core::SensorHealthState::Suspect;
        }
      }
      break;
    case core::SensorHealthState::Failed:
      candidate.recovery_good_shadow_results = 0U;
      if (good && shadow) {
        next_state = core::SensorHealthState::Recovering;
        increment_recovery_epoch = true;
        candidate.consecutive_failures = 0U;
      }
      break;
    case core::SensorHealthState::Recovering:
      if (good && shadow) {
        ++candidate.recovery_good_shadow_results;
        if (candidate.recovery_good_shadow_results >= state_->policy.recovery_good_shadow_results) {
          next_state = core::SensorHealthState::Active;
          candidate.recovery_good_shadow_results = 0U;
          candidate.consecutive_failures = 0U;
        }
      } else {
        next_state = core::SensorHealthState::Failed;
        candidate.recovery_good_shadow_results = 0U;
        candidate.consecutive_failures =
            std::min<std::size_t>(1U, state_->policy.consecutive_failures_to_failed);
      }
      break;
  }

  const bool transitioned = next_state != current.health.state;
  if (transitioned && !transitionSequenceAvailable(current.health)) {
    return UpdateResult::failure(error(
        SensorHealthRegistryErrorCode::TransitionSequenceExhausted,
        "sensor health transition sequence cannot advance without reaching its invalid sentinel",
        observation.sensor, observation.batch_id));
  }
  if (increment_recovery_epoch && !recoveryEpochAvailable(current.health)) {
    return UpdateResult::failure(
        error(SensorHealthRegistryErrorCode::RecoveryEpochExhausted,
              "sensor recovery epoch cannot advance without reaching its invalid sentinel",
              observation.sensor, observation.batch_id, current.health.recovery_epoch,
              observation.recovery_epoch));
  }
  if (increment_recovery_epoch) {
    candidate.health.recovery_epoch =
        core::SensorRecoveryEpoch{current.health.recovery_epoch.value() + 1U};
  }
  if (transitioned) {
    applyStateTransition(&candidate.health, next_state);
  }
  candidate.health.assessed_at = observation.assessed_at;
  candidate.last_batch = observation.batch_id;
  candidate.last_observation_at = observation.assessed_at;
  found->second.snapshot = candidate;

  return UpdateResult::success(SensorHealthUpdate{observation.batch_id, before, candidate.health,
                                                  transitioned, candidate.consecutive_failures,
                                                  candidate.recovery_good_shadow_results});
}

core::Result<SensorHealthTimeoutAssessment, SensorHealthRegistryError>
SensorHealthRegistry::assessTimeout(core::SensorInstanceId sensor,
                                    core::SensorRecoveryEpoch recovery_epoch,
                                    core::FusionTime assessed_at) {
  using TimeoutResult = core::Result<SensorHealthTimeoutAssessment, SensorHealthRegistryError>;
  if (!state_) {
    return TimeoutResult::failure(error(SensorHealthRegistryErrorCode::InvalidConfig,
                                        "sensor health registry is moved from"));
  }
  const auto found = state_->sensors.find(sensor);
  if (found == state_->sensors.end()) {
    return TimeoutResult::failure(error(SensorHealthRegistryErrorCode::UnknownSensor,
                                        "sensor is not in the configured registry set", sensor));
  }
  const SensorHealthRegistrySnapshot& current = found->second.snapshot;
  if (const auto epoch_error = validateEpoch(current, sensor, std::nullopt, recovery_epoch)) {
    return TimeoutResult::failure(*epoch_error);
  }
  if (assessed_at <= current.health.assessed_at) {
    return TimeoutResult::failure(
        error(SensorHealthRegistryErrorCode::StaleAssessmentTime,
              "sensor timeout assessment time must advance monotonically for that sensor", sensor));
  }

  SensorHealthRegistrySnapshot candidate = current;
  const core::SensorHealthSnapshot before = current.health;
  const long double elapsed = elapsedNanoseconds(assessed_at, current.last_observation_at);
  const long double suspect_timeout =
      static_cast<long double>(state_->policy.suspect_after_no_result.nanoseconds);
  const long double failed_timeout =
      static_cast<long double>(state_->policy.failed_after_no_result.nanoseconds);
  core::SensorHealthState next_state = current.health.state;
  bool timed_out = false;
  switch (current.health.state) {
    case core::SensorHealthState::Active:
      if (elapsed >= failed_timeout) {
        next_state = core::SensorHealthState::Failed;
        timed_out = true;
      } else if (elapsed >= suspect_timeout) {
        next_state = core::SensorHealthState::Suspect;
        timed_out = true;
      }
      break;
    case core::SensorHealthState::Suspect:
    case core::SensorHealthState::Recovering:
      if (elapsed >= failed_timeout) {
        next_state = core::SensorHealthState::Failed;
        timed_out = true;
      }
      break;
    case core::SensorHealthState::Failed:
      timed_out = elapsed >= failed_timeout;
      break;
  }

  const bool transitioned = next_state != current.health.state;
  if (transitioned && !transitionSequenceAvailable(current.health)) {
    return TimeoutResult::failure(error(
        SensorHealthRegistryErrorCode::TransitionSequenceExhausted,
        "sensor health transition sequence cannot advance without reaching its invalid sentinel",
        sensor));
  }
  if (transitioned) {
    applyStateTransition(&candidate.health, next_state);
    if (next_state == core::SensorHealthState::Failed) {
      candidate.recovery_good_shadow_results = 0U;
    }
  }
  candidate.health.assessed_at = assessed_at;
  found->second.snapshot = candidate;
  return TimeoutResult::success(
      SensorHealthTimeoutAssessment{before, candidate.health, transitioned, timed_out});
}

core::Result<SensorHealthRegistrySnapshot, SensorHealthRegistryError>
SensorHealthRegistry::snapshot(core::SensorInstanceId sensor) const {
  using SnapshotResult = core::Result<SensorHealthRegistrySnapshot, SensorHealthRegistryError>;
  if (!state_) {
    return SnapshotResult::failure(error(SensorHealthRegistryErrorCode::InvalidConfig,
                                         "sensor health registry is moved from"));
  }
  const auto found = state_->sensors.find(sensor);
  if (found == state_->sensors.end()) {
    return SnapshotResult::failure(error(SensorHealthRegistryErrorCode::UnknownSensor,
                                         "sensor is not in the configured registry set", sensor));
  }
  return SnapshotResult::success(found->second.snapshot);
}

std::vector<SensorHealthRegistrySnapshot> SensorHealthRegistry::snapshots() const {
  std::vector<SensorHealthRegistrySnapshot> output;
  if (!state_) {
    return output;
  }
  output.reserve(state_->sensors.size());
  for (const auto& [sensor, entry] : state_->sensors) {
    (void)sensor;
    output.push_back(entry.snapshot);
  }
  return output;
}

std::size_t SensorHealthRegistry::size() const noexcept {
  return state_ ? state_->sensors.size() : 0U;
}

}  // namespace meridian::local
