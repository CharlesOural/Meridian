#include "conversions/core2ros.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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

// Turbo colour map (degree-5 polynomial approximation, x in [0,1] -> packed 0x00RRGGBB).
// Turbo is perceptually ordered dark-blue -> green -> yellow -> red, so a self-coloured
// cloud reads as a clear gradient in any viewer with no colour-map configuration.
std::uint32_t turbo_packed(float x) {
  x = std::clamp(x, 0.0F, 1.0F);
  const float x2 = x * x;
  const float x3 = x2 * x;
  const float x4 = x2 * x2;
  const float x5 = x4 * x;
  const auto chan = [&](float c0, float c1, float c2, float c3, float c4, float c5) {
    const float v = c0 + c1 * x + c2 * x2 + c3 * x3 + c4 * x4 + c5 * x5;
    return static_cast<std::uint32_t>(std::clamp(v, 0.0F, 1.0F) * 255.0F + 0.5F);
  };
  const std::uint32_t r =
      chan(0.13572138F, 4.61539260F, -42.66032258F, 132.13108234F, -152.94239396F, 59.28637943F);
  const std::uint32_t g =
      chan(0.09140261F, 2.19418839F, 4.84296658F, -14.18503333F, 4.27729857F, 2.82956604F);
  const std::uint32_t b =
      chan(0.10667330F, 12.64194608F, -60.58204836F, 110.36276771F, -89.90310912F, 27.34824973F);
  return (r << 16) | (g << 8) | b;
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
                                             const std::string& frame_id, Timestamp t,
                                             CloudColor color, float color_min, float color_max) {
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp = to_ros(t);
  msg.header.frame_id = frame_id;
  msg.height = 1;
  msg.width = static_cast<std::uint32_t>(view.points.size());
  add_field(msg, "x", 0, PointField::FLOAT32);
  add_field(msg, "y", 4, PointField::FLOAT32);
  add_field(msg, "z", 8, PointField::FLOAT32);
  add_field(msg, "intensity", 12, PointField::FLOAT32);
  // Packed colour: 0x00RRGGBB reinterpreted as float32, the convention RViz/Foxglove read
  // as point colour when a field is named "rgb".
  add_field(msg, "rgb", 16, PointField::FLOAT32);
  add_field(msg, "t", 20, PointField::UINT32);
  add_field(msg, "ring", 24, PointField::UINT16);
  msg.point_step = 28;
  msg.row_step = msg.point_step * msg.width;
  msg.is_bigendian = false;
  msg.is_dense = false;
  msg.data.resize(static_cast<std::size_t>(msg.row_step));

  // The scalar the turbo map runs on, per the selected mode.
  const auto scalar = [color](const LidarPoint& p) {
    return color == CloudColor::Height ? p.xyz.z() : p.intensity;
  };
  // Resolve the normalisation window: an explicit [min,max], else auto over the view's range
  // (so the gradient always spans the data; a degenerate range collapses to mid-scale).
  float lo = color_min;
  float hi = color_max;
  if (hi <= lo) {
    lo = std::numeric_limits<float>::max();
    hi = std::numeric_limits<float>::lowest();
    for (const LidarPoint& p : view.points) {
      const float s = scalar(p);
      lo = std::min(lo, s);
      hi = std::max(hi, s);
    }
  }
  const float span = hi - lo;

  std::uint8_t* dst = msg.data.data();
  for (const LidarPoint& p : view.points) {
    const float xyz[3] = {p.xyz.x(), p.xyz.y(), p.xyz.z()};
    std::memcpy(dst + 0, xyz, 12);
    std::memcpy(dst + 12, &p.intensity, 4);
    const float norm = span > 0.0F ? (scalar(p) - lo) / span : 0.5F;
    const std::uint32_t rgb = turbo_packed(norm);
    std::memcpy(dst + 16, &rgb, 4);
    const std::uint32_t t_off = p.t_offset_ns >= 0 ? static_cast<std::uint32_t>(p.t_offset_ns) : 0u;
    std::memcpy(dst + 20, &t_off, 4);
    std::memcpy(dst + 24, &p.ring, 2);
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
    case Marker::Type::Triangles:
      msg.type = RosMarker::TRIANGLE_LIST;
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
