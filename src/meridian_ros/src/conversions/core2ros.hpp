#pragma once

#include <string>

#include <builtin_interfaces/msg/time.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "meridian/common/frame.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"
#include "meridian/debug/telemetry.hpp"

namespace meridian {

// core -> msg translation. The single Timestamp -> ROS time conversion lives here.

inline builtin_interfaces::msg::Time to_ros(Timestamp t) {
  // A malformed upstream stamp must never become a negative ROS time.
  if (t < 0) t = 0;
  builtin_interfaces::msg::Time out;
  out.sec = static_cast<std::int32_t>(t / kNanosPerSecond);
  out.nanosec = static_cast<std::uint32_t>(t % kNanosPerSecond);
  return out;
}

// TF-style name for a core frame.
const char* frame_name(Frame f);

// Serializes a point view as x,y,z,intensity,rgb (float32), t (uint32 ns), ring (uint16).
// `rgb` carries a baked colour so a viewer renders the cloud without per-user colour-map
// setup: `color` selects the scalar (intensity reflectivity / map-frame height) the turbo
// map runs on, normalised over [color_min, color_max]; color_max <= color_min auto-normalises
// over the view's own range.
sensor_msgs::msg::PointCloud2 to_pointcloud2(const PointCloudView& view,
                                             const std::string& frame_id, Timestamp t,
                                             CloudColor color = CloudColor::Intensity,
                                             float color_min = 0.0F, float color_max = 0.0F);

nav_msgs::msg::Odometry to_odometry(const Pose& pose, const std::string& frame_id,
                                    Timestamp t);

visualization_msgs::msg::Marker to_marker(const Marker& m, Timestamp t);

// Publishes the overlay's base image (patch rasterisation arrives with the visual
// front-end).
sensor_msgs::msg::Image to_image(const ImageOverlay& overlay, Timestamp t);

}  // namespace meridian
