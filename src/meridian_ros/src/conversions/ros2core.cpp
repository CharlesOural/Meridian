#include "conversions/ros2core.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#include <sensor_msgs/msg/nav_sat_status.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace meridian {

namespace {

using sensor_msgs::msg::PointField;

struct FieldRef {
  int offset = -1;
  std::uint8_t datatype = 0;
  bool valid() const { return offset >= 0; }
};

FieldRef find_field(const sensor_msgs::msg::PointCloud2& msg, const std::string& name) {
  for (const auto& f : msg.fields) {
    if (f.name == name) return FieldRef{static_cast<int>(f.offset), f.datatype};
  }
  return {};
}

template <typename T>
T read_unaligned(const std::uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

float read_float(const std::uint8_t* p, std::uint8_t datatype) {
  switch (datatype) {
    case PointField::FLOAT32: return read_unaligned<float>(p);
    case PointField::FLOAT64: return static_cast<float>(read_unaligned<double>(p));
    case PointField::UINT16: return static_cast<float>(read_unaligned<std::uint16_t>(p));
    case PointField::UINT8: return static_cast<float>(*p);
    case PointField::UINT32: return static_cast<float>(read_unaligned<std::uint32_t>(p));
    default: return 0.f;
  }
}

std::uint32_t read_u32(const std::uint8_t* p, std::uint8_t datatype) {
  switch (datatype) {
    case PointField::UINT32: return read_unaligned<std::uint32_t>(p);
    case PointField::UINT16: return read_unaligned<std::uint16_t>(p);
    case PointField::UINT8: return *p;
    default: return 0;
  }
}

}  // namespace

bool to_raw_lidar(const sensor_msgs::msg::PointCloud2& msg, Timestamp host_arrival,
                  std::vector<RawPoint>* scratch, RawLidarFrame* out) {
  const FieldRef fx = find_field(msg, "x");
  const FieldRef fy = find_field(msg, "y");
  const FieldRef fz = find_field(msg, "z");
  if (!fx.valid() || !fy.valid() || !fz.valid()) return false;

  // Per-point relative time is mandatory: "t" (uint32 ns, Ouster) or "time"
  // (float32 s). Both fields are relative offsets from the message stamp, not
  // absolute times; the "time" float branch interprets the value as relative
  // seconds. Offsets are normalized so the first return sits at 0.
  const FieldRef ft_ns = find_field(msg, "t");
  const FieldRef ft_s = find_field(msg, "time");
  const bool t_is_ns = ft_ns.valid() && ft_ns.datatype == PointField::UINT32;
  const bool t_is_s = !t_is_ns && ft_s.valid() && ft_s.datatype == PointField::FLOAT32;
  if (!t_is_ns && !t_is_s) return false;

  const FieldRef fi_a = find_field(msg, "intensity");
  const FieldRef fi_b = find_field(msg, "signal");
  const FieldRef fi_c = find_field(msg, "reflectivity");
  const FieldRef fi = fi_a.valid() ? fi_a : (fi_b.valid() ? fi_b : fi_c);
  FieldRef fring = find_field(msg, "ring");
  const FieldRef fambient_a = find_field(msg, "ambient");
  const FieldRef fambient_b = find_field(msg, "near_ir");
  const FieldRef fambient = fambient_a.valid() ? fambient_a : fambient_b;
  const FieldRef frange = find_field(msg, "range");

  const std::size_t n = static_cast<std::size_t>(msg.width) * msg.height;
  const std::uint8_t* base = msg.data.data();
  const auto point_at = [&](std::size_t row, std::size_t col) {
    return base + row * msg.row_step + col * msg.point_step;
  };

  if (n == 0) return false;

  // Pass 1: the smallest per-point time, so offsets can be normalized to it.
  std::int64_t t_min = std::numeric_limits<std::int64_t>::max();
  for (std::size_t r = 0; r < msg.height; ++r) {
    for (std::size_t c = 0; c < msg.width; ++c) {
      const std::uint8_t* p = point_at(r, c);
      const std::int64_t t =
          t_is_ns ? static_cast<std::int64_t>(read_u32(p + ft_ns.offset, ft_ns.datatype))
                  : static_cast<std::int64_t>(
                        static_cast<double>(read_unaligned<float>(p + ft_s.offset)) * 1e9);
      t_min = std::min(t_min, t);
    }
  }

  scratch->clear();
  scratch->reserve(n);
  for (std::size_t r = 0; r < msg.height; ++r) {
    for (std::size_t c = 0; c < msg.width; ++c) {
      const std::uint8_t* p = point_at(r, c);
      RawPoint pt;
      pt.x = read_unaligned<float>(p + fx.offset);
      pt.y = read_unaligned<float>(p + fy.offset);
      pt.z = read_unaligned<float>(p + fz.offset);
      if (fi.valid()) pt.intensity = read_float(p + fi.offset, fi.datatype);
      const std::int64_t t =
          t_is_ns ? static_cast<std::int64_t>(read_u32(p + ft_ns.offset, ft_ns.datatype))
                  : static_cast<std::int64_t>(
                        static_cast<double>(read_unaligned<float>(p + ft_s.offset)) * 1e9);
      pt.t = static_cast<std::uint32_t>(t - t_min);
      if (fring.valid())
        pt.ring = static_cast<std::uint16_t>(read_u32(p + fring.offset, fring.datatype));
      if (fambient.valid())
        pt.ambient =
            static_cast<std::uint16_t>(read_u32(p + fambient.offset, fambient.datatype));
      if (frange.valid()) {
        // Ouster reports range in millimetres as uint32.
        pt.range_m = frange.datatype == PointField::UINT32
                         ? static_cast<float>(read_u32(p + frange.offset, frange.datatype)) *
                               1e-3f
                         : read_float(p + frange.offset, frange.datatype);
      }
      scratch->push_back(pt);
    }
  }

  out->host_arrival = host_arrival;
  // The sweep reference is the message stamp shifted to the first return, so per-point
  // offsets are zero-based regardless of the driver's t origin.
  out->device_ns_first_column = from_ros(msg.header.stamp) + t_min;
  out->has_device_ns = true;
  out->pts = *scratch;
  return true;
}

RawImuFrame to_raw_imu(const sensor_msgs::msg::Imu& msg, Timestamp host_arrival) {
  RawImuFrame f;
  f.device_ns = from_ros(msg.header.stamp);
  f.has_device_ns = true;
  f.host_arrival = host_arrival;
  f.acc = Eigen::Vector3d(msg.linear_acceleration.x, msg.linear_acceleration.y,
                          msg.linear_acceleration.z);
  f.gyro = Eigen::Vector3d(msg.angular_velocity.x, msg.angular_velocity.y,
                           msg.angular_velocity.z);
  return f;
}

std::optional<RawCameraFrame> to_raw_camera(const sensor_msgs::msg::Image& msg,
                                            Timestamp host_arrival) {
  RawCameraFrame f;
  f.device_ns = from_ros(msg.header.stamp);
  f.has_device_ns = true;
  f.host_arrival = host_arrival;
  f.width = static_cast<int>(msg.width);
  f.height = static_cast<int>(msg.height);

  const std::size_t row = static_cast<std::size_t>(msg.step);
  // Copy row-by-row honouring msg.step so the tightly packed core buffer never
  // inherits the wire's per-row padding.
  const auto pack_rows = [&](std::size_t bytes_per_pixel) {
    const std::size_t dst_row = static_cast<std::size_t>(msg.width) * bytes_per_pixel;
    auto buf = std::make_shared<std::vector<std::uint8_t>>(dst_row * msg.height);
    for (std::size_t r = 0; r < msg.height; ++r) {
      const std::uint8_t* src = msg.data.data() + r * row;
      std::uint8_t* dst = buf->data() + r * dst_row;
      std::memcpy(dst, src, dst_row);
    }
    return buf;
  };

  if (msg.encoding == "mono8") {
    f.encoding = CameraFrame::Encoding::Mono8;
    f.data = pack_rows(1);
  } else if (msg.encoding == "rgb8") {
    f.encoding = CameraFrame::Encoding::RGB8;
    f.data = pack_rows(3);
  } else if (msg.encoding == "bgr8") {
    f.encoding = CameraFrame::Encoding::RGB8;
    auto rgb = std::make_shared<std::vector<std::uint8_t>>(
        static_cast<std::size_t>(msg.width) * msg.height * 3);
    for (std::size_t r = 0; r < msg.height; ++r) {
      const std::uint8_t* src = msg.data.data() + r * row;
      std::uint8_t* dst = rgb->data() + r * static_cast<std::size_t>(msg.width) * 3;
      for (std::size_t c = 0; c < msg.width; ++c) {
        dst[3 * c + 0] = src[3 * c + 2];
        dst[3 * c + 1] = src[3 * c + 1];
        dst[3 * c + 2] = src[3 * c + 0];
      }
    }
    f.data = std::move(rgb);
  } else if (msg.encoding == "bayer_rggb8") {
    f.encoding = CameraFrame::Encoding::Bayer_RGGB8;
    f.data = pack_rows(1);
  } else {
    return std::nullopt;
  }
  return f;
}

RawGnssFrame to_raw_gnss(const sensor_msgs::msg::NavSatFix& msg, Timestamp host_arrival) {
  using sensor_msgs::msg::NavSatStatus;
  RawGnssFrame f;
  f.device_ns = from_ros(msg.header.stamp);
  f.has_device_ns = true;
  f.host_arrival = host_arrival;
  f.lat_deg = msg.latitude;
  f.lon_deg = msg.longitude;
  f.alt_m = msg.altitude;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) f.cov_enu(r, c) = msg.position_covariance[3 * r + c];
  switch (msg.status.status) {
    case NavSatStatus::STATUS_FIX: f.fix = GnssFix::FixType::SPP; break;
    case NavSatStatus::STATUS_SBAS_FIX:
    case NavSatStatus::STATUS_GBAS_FIX: f.fix = GnssFix::FixType::DGPS; break;
    default: f.fix = GnssFix::FixType::None; break;
  }
  return f;
}

}  // namespace meridian
