#pragma once

#include "meridian/global/graph.hpp"

namespace meridian::global::checkpoint_internal {

[[nodiscard]] GlobalGraphCheckpointLimits limitsForConfig(const GlobalGraphConfig& config) noexcept;

[[nodiscard]] core::Result<GlobalGraphConfigurationIdentity, GlobalGraphCheckpointError>
configurationIdentity(const GlobalGraphConfig& config, GlobalGraphCheckpointLimits limits);

[[nodiscard]] GlobalGraphRecoveryTolerances recoveryTolerances(
    const GlobalGraphConfig& config) noexcept;

// Verifies all typed/canonical structure and installs the checksum computed
// over every field except checksum itself.
[[nodiscard]] core::Result<GlobalGraphCheckpoint, GlobalGraphCheckpointError> finalize(
    GlobalGraphCheckpoint checkpoint, GlobalGraphCheckpointLimits limits);

// Object-level verification used before a restore. This does not trust a
// caller-supplied checksum and does not mutate the input.
[[nodiscard]] core::Result<bool, GlobalGraphCheckpointError> verify(
    const GlobalGraphCheckpoint& checkpoint, GlobalGraphCheckpointLimits limits);

}  // namespace meridian::global::checkpoint_internal
