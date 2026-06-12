#include "conversions/core2ros.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sensor_msgs/msg/point_field.hpp>

namespace meridian {

namespace {
using sensor_msgs::msg::PointField;

void add_field(sensor_msgs::msg::PointCloud2& msg, const char* name, std::uint32_t offset,
               std::uint8_t datatype) {
  PointField f;
  f.name = name;
  f.offset = offset;
  f.datatype = datatype;
  f.count = 1;
  msg.fields.push_back(f);
}
}  // namespace

const char* frame_name(Frame f) {
  switch (f) {
    case Frame::Map:
      return "map";
    case Frame::Odom:
      return "odom";
    case Frame::BaseLink:
      return "base_link";
    case Frame::ImuLink:
      return "imu_link";
    case Frame::CamLink:
      return "cam_link";
    case Frame::GnssLink:
      return "gnss_link";
    case Frame::OsSensor0:
      return "os_sensor0";
    case Frame::Body:
      return "body";
    case Frame::Unknown:
      break;
  }
  return "unknown";
}

sensor_msgs::msg::PointCloud2 to_pointcloud2(const PointCloudView& view,
                                             const std::string& frame_id, Timestamp t) {
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp = to_ros(t);
  msg.header.frame_id = frame_id;
  msg.height = 1;
  msg.width = static_cast<std::uint32_t>(view.points.size());
  add_field(msg, "x", 0, PointField::FLOAT32);
  add_field(msg, "y", 4, PointField::FLOAT32);
  add_field(msg, "z", 8, PointField::FLOAT32);
  add_field(msg, "intensity", 12, PointField::FLOAT32);
  add_field(msg, "t", 16, PointField::UINT32);
  add_field(msg, "ring", 20, PointField::UINT16);
  msg.point_step = 24;
  msg.row_step = msg.point_step * msg.width;
  msg.is_bigendian = false;
  msg.is_dense = false;
  msg.data.resize(static_cast<std::size_t>(msg.row_step));

  std::uint8_t* dst = msg.data.data();
  for (const LidarPoint& p : view.points) {
    const float xyz[3] = {p.xyz.x(), p.xyz.y(), p.xyz.z()};
    std::memcpy(dst + 0, xyz, 12);
    std::memcpy(dst + 12, &p.intensity, 4);
    const std::uint32_t t_off = p.t_offset_ns >= 0 ? static_cast<std::uint32_t>(p.t_offset_ns) : 0u;
    std::memcpy(dst + 16, &t_off, 4);
    std::memcpy(dst + 20, &p.ring, 2);
    dst += msg.point_step;
  }
  return msg;
}

nav_msgs::msg::Odometry to_odometry(const Pose& pose, const std::string& frame_id, Timestamp t) {
  nav_msgs::msg::Odometry msg;
  msg.header.stamp = to_ros(t);
  msg.header.frame_id = frame_id;
  msg.child_frame_id = "body";
  msg.pose.pose.position.x = pose.t.x();
  msg.pose.pose.position.y = pose.t.y();
  msg.pose.pose.position.z = pose.t.z();
  msg.pose.pose.orientation.w = pose.q.w();
  msg.pose.pose.orientation.x = pose.q.x();
  msg.pose.pose.orientation.y = pose.q.y();
  msg.pose.pose.orientation.z = pose.q.z();
  return msg;
}

visualization_msgs::msg::Marker to_marker(const Marker& m, Timestamp t) {
  using RosMarker = visualization_msgs::msg::Marker;
  RosMarker msg;
  msg.header.stamp = to_ros(t);
  msg.header.frame_id = frame_name(m.frame);
  msg.ns = m.ns;
  msg.id = m.id;
  msg.action = RosMarker::ADD;
  switch (m.type) {
    case Marker::Type::Points:
      msg.type = RosMarker::POINTS;
      break;
    case Marker::Type::LineList:
      msg.type = RosMarker::LINE_LIST;
      break;
    case Marker::Type::LineStrip:
    case Marker::Type::Hexagon:
      msg.type = RosMarker::LINE_STRIP;
      break;
    case Marker::Type::Arrow:
      msg.type = RosMarker::ARROW;
      break;
    case Marker::Type::Cube:
      msg.type = RosMarker::CUBE;
      break;
    case Marker::Type::Sphere:
      msg.type = RosMarker::SPHERE;
      break;
    case Marker::Type::Text:
      msg.type = RosMarker::TEXT_VIEW_FACING;
      break;
  }
  msg.scale.x = m.scale;
  msg.scale.y = m.scale;
  msg.scale.z = m.scale;
  msg.color.r = m.color[0];
  msg.color.g = m.color[1];
  msg.color.b = m.color[2];
  msg.color.a = m.color[3];
  msg.text = m.text;
  msg.lifetime.sec = static_cast<std::int32_t>(m.lifetime_ns / kNanosPerSecond);
  msg.lifetime.nanosec = static_cast<std::uint32_t>(m.lifetime_ns % kNanosPerSecond);
  msg.points.reserve(m.points.size());
  for (const auto& p : m.points) {
    geometry_msgs::msg::Point q;
    q.x = p.x();
    q.y = p.y();
    q.z = p.z();
    msg.points.push_back(q);
  }
  msg.colors.reserve(m.colors.size());
  for (const auto& c : m.colors) {
    std_msgs::msg::ColorRGBA q;
    q.r = c[0];
    q.g = c[1];
    q.b = c[2];
    q.a = c[3];
    msg.colors.push_back(q);
  }
  return msg;
}

sensor_msgs::msg::Image to_image(const ImageOverlay& overlay, Timestamp t) {
  sensor_msgs::msg::Image msg;
  msg.header.stamp = to_ros(t);
  msg.header.frame_id = frame_name(overlay.frame);
  msg.width = static_cast<std::uint32_t>(overlay.width);
  msg.height = static_cast<std::uint32_t>(overlay.height);
  const bool src_rgb = overlay.encoding == ImageOverlay::Encoding::RGB8;
  const int w = overlay.width;
  const int h = overlay.height;

  // With no tracked patches the image passes through in its native encoding. With
  // patches we always emit rgb8 so the green annotations are visible over the grey
  // intensity image; the base bytes are expanded to RGB first, then the patch boxes
  // and centre dots are drawn directly into the buffer (no OpenCV dependency at the
  // ROS edge).
  if (overlay.patches.empty()) {
    msg.encoding = src_rgb ? "rgb8" : "mono8";
    msg.step = msg.width * (src_rgb ? 3 : 1);
    msg.is_bigendian = false;
    msg.data.assign(overlay.base.begin(), overlay.base.end());
    return msg;
  }

  msg.encoding = "rgb8";
  msg.step = msg.width * 3;
  msg.is_bigendian = false;
  msg.data.assign(static_cast<std::size_t>(w) * h * 3, 0);
  const std::size_t base_n = overlay.base.size();
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t di = (static_cast<std::size_t>(y) * w + x) * 3;
      if (src_rgb) {
        const std::size_t si = (static_cast<std::size_t>(y) * w + x) * 3;
        if (si + 2 < base_n) {
          msg.data[di] = overlay.base[si];
          msg.data[di + 1] = overlay.base[si + 1];
          msg.data[di + 2] = overlay.base[si + 2];
        }
      } else {
        const std::size_t si = static_cast<std::size_t>(y) * w + x;
        const std::uint8_t g = si < base_n ? overlay.base[si] : 0;
        msg.data[di] = g;
        msg.data[di + 1] = g;
        msg.data[di + 2] = g;
      }
    }
  }

  auto set_px = [&](int x, int y, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    const std::size_t di = (static_cast<std::size_t>(y) * w + x) * 3;
    msg.data[di] = r;
    msg.data[di + 1] = g;
    msg.data[di + 2] = b;
  };
  // Patch half-extent grows with the search level (each level halves resolution, so a
  // level-l patch covers 2^l level-0 pixels per cell). The box is drawn green; a small
  // red centre dot marks the tracked pixel.
  for (const ImageOverlay::Patch& p : overlay.patches) {
    const int cx = static_cast<int>(std::lround(p.uv.x()));
    const int cy = static_cast<int>(std::lround(p.uv.y()));
    const int half = 4 * (1 << std::max(p.level, 0));
    for (int d = -half; d <= half; ++d) {
      set_px(cx + d, cy - half, 0, 255, 0);
      set_px(cx + d, cy + half, 0, 255, 0);
      set_px(cx - half, cy + d, 0, 255, 0);
      set_px(cx + half, cy + d, 0, 255, 0);
    }
    set_px(cx, cy, 255, 0, 0);
  }
  return msg;
}

}  // namespace meridian
