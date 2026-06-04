#pragma once

#include <cstdint>

namespace meridian {

// Graded health of a sensor or pipeline stage — the graded replacement for a binary
// "ok/not-ok" flag. Richer per-sensor health detail lives in the L0 layer; this is the
// shared graded primitive other layers consult.
enum class HealthLevel : std::uint8_t {
  Nominal = 0,
  Degraded = 1,
  Failed = 2,
};

}  // namespace meridian
