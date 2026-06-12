#include "meridian/debug/telemetry.hpp"

#include "meridian/debug/log.hpp"

namespace meridian {

// Wall-clock reading on the Meridian timeline, in integer nanoseconds since the
// steady-clock epoch. Used as the default stamp for log lines emitted by the macros.
Timestamp now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

}  // namespace meridian
