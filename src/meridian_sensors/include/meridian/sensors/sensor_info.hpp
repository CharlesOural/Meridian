#pragma once

#include <cstdint>
#include <string>

#include "meridian/common/frame.hpp"
#include "meridian/time/stamp_source.hpp"

namespace meridian {

// The physical kind of a sensor stream. One source per modality in this rig.
enum class Modality : std::uint8_t {
  Imu = 0,
  Lidar = 1,
  Camera = 2,
  Gnss = 3,
};

// Static descriptor a source publishes through ISensorSource::info(). It is fixed
// for the source's lifetime; dynamic state (rate, sync, timing uncertainty) lives on
// the health channel instead.
struct SensorInfo {
  std::uint8_t id = 0;                            // unique within the rig
  Modality modality = Modality::Imu;              // Imu | Lidar | Camera | Gnss
  Frame sensor_frame = Frame::Unknown;            // os_sensor0 / cam_link / imu_link / gnss_link
  std::string model = "";                         // e.g. "ouster_os1_128", "vn100"
  double nominal_rate_hz = 0.0;                   // expected sample/scan rate [Hz], for dropout detection
  StampSource configured_stamp_source =           // best mechanism this sensor is wired for
      StampSource::ArrivalOnly;
};

}  // namespace meridian
