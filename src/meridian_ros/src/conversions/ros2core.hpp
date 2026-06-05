#pragma once

#include <optional>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "meridian/sensors/raw_frames.hpp"

namespace meridian {

// msg -> core translation. The ONLY place wire types become core types; the core
// never sees a ROS message. Stamps convert through from_ros() alone.

// ROS time -> Meridian timeline nanoseconds.
inline Timestamp from_ros(const builtin_interfaces::msg::Time& t) {
  return static_cast<Timestamp>(t.sec) * kNanosPerSecond + static_cast<Timestamp>(t.nanosec);
}

// Parses a PointCloud2 into a wire-free LiDAR frame. Requires x/y/z float fields and a
// per-point relative time field ("t" uint32 ns, or "time" float32 s); returns false
// when either is missing (the scan is unusable for deskew). `scratch` is reused across
// calls; `out->pts` views it and stays valid only until the next call.
bool to_raw_lidar(const sensor_msgs::msg::PointCloud2& msg, Timestamp host_arrival,
                  std::vector<RawPoint>* scratch, RawLidarFrame* out);

RawImuFrame to_raw_imu(const sensor_msgs::msg::Imu& msg, Timestamp host_arrival);

// Supported encodings: mono8, rgb8, bgr8 (swapped to RGB), bayer_rggb8. Returns
// nullopt for anything else.
std::optional<RawCameraFrame> to_raw_camera(const sensor_msgs::msg::Image& msg,
                                            Timestamp host_arrival);

RawGnssFrame to_raw_gnss(const sensor_msgs::msg::NavSatFix& msg, Timestamp host_arrival);

}  // namespace meridian
