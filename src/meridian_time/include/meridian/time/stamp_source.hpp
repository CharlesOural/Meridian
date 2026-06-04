#pragma once

#include <cstdint>

namespace meridian {

// How a sample's timestamp was placed on the Meridian timeline, ordered from
// most to least trustworthy. Reported per clock for health/telemetry.
enum class StampSource : std::uint8_t {
  HwPtp       = 0,  // device stamped from a PTP-disciplined hardware clock
  HwPps       = 1,  // device stamped against a PPS edge
  HwTrigger   = 2,  // device exposed/sampled on an external GPIO trigger
  SwOffset    = 3,  // host stamp corrected by an estimated software offset
  ArrivalOnly = 4,  // last resort: host arrival time, no correction (degraded)
  Replay      = 5,  // stamp came from a recorded bag (trusted as-is)
};

}  // namespace meridian
