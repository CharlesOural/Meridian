#pragma once

#include <functional>

#include "meridian/sensors/sensor_info.hpp"

namespace meridian {

// One per physical sensor stream, templated over the four concrete sample types
// {ImuSample, LidarScan, CameraFrame, GnssFix}. The source owns its acquisition
// thread and pushes a finished, time-synced sample to the callback.
//
// Obligations on every implementation:
//   - the sample handed to the callback carries a final Meridian-timeline stamp;
//     no downstream code re-stamps it,
//   - stamps are non-decreasing within one sensor_id (a regression is clamped and
//     reported, never silently emitted),
//   - the callback's caller never blocks on the consumer (the push is non-blocking),
//   - health is updated on every sample and on every state change,
//   - after stop() returns no further callbacks fire and the acquisition thread is
//     joined.
//
// Thread-confined: set_callback/start/stop are called from the owning thread; the
// callback fires on the source's acquisition thread.
template <class SampleT>
class ISensorSource {
 public:
  virtual ~ISensorSource() = default;

  // The source moves a single finished sample into this callback on its own thread.
  using Callback = std::function<void(SampleT&&)>;
  virtual void set_callback(Callback cb) = 0;

  virtual void start() = 0;  // begin streaming
  virtual void stop() = 0;   // stop; no callbacks fire after this returns

  virtual SensorInfo info() const = 0;
};

}  // namespace meridian
