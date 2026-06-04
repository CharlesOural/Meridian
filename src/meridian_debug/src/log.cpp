#include "meridian/debug/log.hpp"

#include <atomic>

namespace meridian {

namespace {
// Process-global log sink, set once at startup. Atomic so a late set_log_sink()
// never races a concurrent log_sink() read.
std::atomic<LogSink*> g_log_sink{nullptr};
}  // namespace

LogSink* log_sink() { return g_log_sink.load(std::memory_order_acquire); }

void set_log_sink(LogSink* sink) { g_log_sink.store(sink, std::memory_order_release); }

}  // namespace meridian
