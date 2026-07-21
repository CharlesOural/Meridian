#pragma once

#include <cstdint>
#include <string>
#include <variant>

#include "meridian/core/ids.hpp"
#include "meridian/core/navigation.hpp"
#include "meridian/local_rt/config.hpp"
#include "meridian/local_rt/imu_types.hpp"

namespace meridian::local_rt {

enum class PropagationErrorCode : std::uint8_t {
  kEmptyInterval,
  kSeedTimeMismatch,
  kNonFiniteResult,
};

struct PropagationFailure final {
  PropagationErrorCode code;
  std::string message;
};

class PropagationResult final {
public:
  explicit PropagationResult(DensePropagation propagation);
  explicit PropagationResult(PropagationFailure failure);

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] const DensePropagation* value() const noexcept;
  [[nodiscard]] const PropagationFailure* error() const noexcept;

private:
  std::variant<DensePropagation, PropagationFailure> result_;
};

// Bias-corrected midpoint propagation at every segment boundary. This is the
// live/debug trajectory path; optimizer state remains represented separately.
class ImuPropagator final {
public:
  explicit ImuPropagator(ImuModel model);

  [[nodiscard]] PropagationResult propagate(const core::NavigationState& seed,
                                            core::StateId endpoint_id,
                                            const ImuInterval& interval) const;

private:
  ImuModel model_;
};

}  // namespace meridian::local_rt
