#pragma once

#include <sstream>
#include <string>
#include <string_view>

#include "meridian/common/time.hpp"
#include "meridian/debug/telemetry.hpp"

namespace meridian {

// Sink for structured log lines. The core formats a "k=v k=v" body and hands it
// here; `enabled` is the cheap per-(level,module) gate so a disabled line costs
// only the gate check.
class LogSink {
 public:
  virtual ~LogSink() = default;
  virtual void log(Level, const char* module, std::string_view kvline, Timestamp) = 0;
  virtual bool enabled(Level, const char* module) const = 0;
};

// The process-global log sink, set once at startup. Null until bound.
LogSink* log_sink();
void set_log_sink(LogSink*);

// Current time on the Meridian timeline [ns].
Timestamp now();

namespace detail {

inline void fmt_kv_emit(std::ostringstream&) {}

// Consume the argument pack two at a time as (key, value) pairs, emitting
// "key=value" tokens separated by single spaces.
template <typename K, typename V, typename... Rest>
void fmt_kv_emit(std::ostringstream& os, const K& key, const V& value, const Rest&... rest) {
  if (os.tellp() != std::streamoff(0)) os << ' ';
  os << key << '=' << value;
  fmt_kv_emit(os, rest...);
}

}  // namespace detail

// Build a "k=v k=v ..." line from alternating key/value arguments. An empty pack
// yields an empty string; a trailing unpaired argument is a compile error.
template <typename... Args>
std::string fmt_kv(const Args&... args) {
  std::ostringstream os;
  detail::fmt_kv_emit(os, args...);
  return os.str();
}

}  // namespace meridian

#define MERIDIAN_LOG(lvl, mod, ...)                                       \
  do {                                                                    \
    auto* _s = ::meridian::log_sink();                                    \
    if (_s && _s->enabled(lvl, mod))                                      \
      _s->log(lvl, mod, ::meridian::fmt_kv(__VA_ARGS__), ::meridian::now()); \
  } while (0)

#define MERIDIAN_TRACE(mod, ...) MERIDIAN_LOG(::meridian::Level::Trace, mod, __VA_ARGS__)
#define MERIDIAN_DEBUG(mod, ...) MERIDIAN_LOG(::meridian::Level::Debug, mod, __VA_ARGS__)
#define MERIDIAN_INFO(mod, ...) MERIDIAN_LOG(::meridian::Level::Info, mod, __VA_ARGS__)
#define MERIDIAN_WARN(mod, ...) MERIDIAN_LOG(::meridian::Level::Warn, mod, __VA_ARGS__)
#define MERIDIAN_ERROR(mod, ...) MERIDIAN_LOG(::meridian::Level::Error, mod, __VA_ARGS__)
