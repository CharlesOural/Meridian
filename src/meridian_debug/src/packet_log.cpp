#include "meridian/debug/packet_log.hpp"

#include <bit>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace meridian {
namespace {

// Fixed-width ints are written in their native byte order; the format is defined as
// little-endian, so refuse to compile where the two would differ.
static_assert(std::endian::native == std::endian::little,
              "packet log assumes a little-endian host");

constexpr char kMagic[8] = {'M', 'E', 'R', 'I', 'D', 'P', 'K', 'T'};
constexpr std::uint16_t kVersion = 1;
constexpr std::uint16_t kFlagCloudsIncluded = 1u << 0;

// ---- payload serialization -------------------------------------------------------
// Scalars are memcpy'd at their exact width, so doubles/floats round-trip bit-exactly.
// Matrices are emitted element-by-element in row-major order regardless of Eigen's
// storage order; quaternions as (x, y, z, w); poses as 7 doubles (qx qy qz qw tx ty tz).

template <typename T>
void put(std::vector<std::uint8_t>* buf, T v) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto* b = reinterpret_cast<const std::uint8_t*>(&v);
  buf->insert(buf->end(), b, b + sizeof(v));
}

void put_vec3(std::vector<std::uint8_t>* buf, const Eigen::Vector3d& v) {
  put(buf, v.x());
  put(buf, v.y());
  put(buf, v.z());
}

void put_quat(std::vector<std::uint8_t>* buf, const Eigen::Quaterniond& q) {
  put(buf, q.x());
  put(buf, q.y());
  put(buf, q.z());
  put(buf, q.w());
}

void put_pose(std::vector<std::uint8_t>* buf, const Pose& T) {
  put_quat(buf, T.q);
  put_vec3(buf, T.t);
}

template <int R, int C>
void put_mat(std::vector<std::uint8_t>* buf, const Eigen::Matrix<double, R, C>& m) {
  for (int r = 0; r < R; ++r) {
    for (int c = 0; c < C; ++c) {
      put(buf, m(r, c));
    }
  }
}

// Bounds-checked forward cursor over a record payload; any overrun means the writer
// and reader disagree on layout (or the file is corrupt), so it throws.
class Cursor {
 public:
  Cursor(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

  template <typename T>
  T get() {
    static_assert(std::is_trivially_copyable_v<T>);
    if (off_ + sizeof(T) > size_) {
      throw std::runtime_error("packet log: truncated payload");
    }
    T v;
    std::memcpy(&v, data_ + off_, sizeof(T));
    off_ += sizeof(T);
    return v;
  }

  Eigen::Vector3d get_vec3() {
    const double x = get<double>(), y = get<double>(), z = get<double>();
    return {x, y, z};
  }

  Eigen::Quaterniond get_quat() {
    const double x = get<double>(), y = get<double>(), z = get<double>(), w = get<double>();
    // Direct coefficient construction: re-normalizing here would perturb the stored bits.
    return Eigen::Quaterniond(w, x, y, z);
  }

  Pose get_pose() {
    Pose T;
    // Assign members directly; the Pose(q, t) constructor normalizes, which would
    // break the bit-exact round-trip guarantee.
    T.q = get_quat();
    T.t = get_vec3();
    return T;
  }

  template <int R, int C>
  Eigen::Matrix<double, R, C> get_mat() {
    Eigen::Matrix<double, R, C> m;
    for (int r = 0; r < R; ++r) {
      for (int c = 0; c < C; ++c) {
        m(r, c) = get<double>();
      }
    }
    return m;
  }

  bool done() const { return off_ == size_; }

 private:
  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t off_ = 0;
};

const char* constraint_kind_name(KeyframePacket::ConstraintKind k) {
  switch (k) {
    case KeyframePacket::ConstraintKind::RelativeBetween:
      return "between";
    case KeyframePacket::ConstraintKind::AbsolutePrior:
      return "prior";
    case KeyframePacket::ConstraintKind::ImuPreintegration:
      return "imu";
  }
  return "between";
}

}  // namespace

// ---- writer ----------------------------------------------------------------------

PacketLogWriter::PacketLogWriter(const std::string& path, bool include_clouds)
    : bin_(path, std::ios::binary),
      index_(path + ".index.txt"),
      include_clouds_(include_clouds) {
  if (!bin_ || !index_) {
    throw std::runtime_error("packet log: cannot open " + path);
  }
  bin_.write(kMagic, sizeof(kMagic));
  const std::uint16_t version = kVersion;
  const std::uint16_t flags = include_clouds_ ? kFlagCloudsIncluded : 0;
  const std::uint32_t reserved = 0;
  bin_.write(reinterpret_cast<const char*>(&version), sizeof(version));
  bin_.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
  bin_.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
  index_ << "# meridian packet index v1\n";
}

PacketLogWriter::~PacketLogWriter() {
  bin_.flush();
  index_.flush();
}

void PacketLogWriter::write_record(std::uint8_t kind, const std::vector<std::uint8_t>& payload) {
  const auto len = static_cast<std::uint32_t>(payload.size());
  bin_.write(reinterpret_cast<const char*>(&kind), sizeof(kind));
  bin_.write(reinterpret_cast<const char*>(&len), sizeof(len));
  bin_.write(reinterpret_cast<const char*>(payload.data()),
             static_cast<std::streamsize>(payload.size()));
}

void PacketLogWriter::write_keyframe(const KeyframePacket& kf) {
  std::vector<std::uint8_t> p;
  put(&p, kf.id);
  put(&p, static_cast<std::int64_t>(kf.stamp));
  put(&p, static_cast<std::uint16_t>(kf.ref_frame));
  put_pose(&p, kf.T_ref_body);
  put(&p, static_cast<std::uint8_t>(kf.kinematics_included ? 1 : 0));
  put_vec3(&p, kf.v_ref);
  put_vec3(&p, kf.b_g);
  put_vec3(&p, kf.b_a);
  put(&p, static_cast<std::uint8_t>(kf.constraint_kind));
  put(&p, kf.rel_to_id);
  put_pose(&p, kf.T_relto_this);
  put(&p, static_cast<std::uint8_t>(kf.constraint_cov.form));
  put_mat<6, 6>(&p, kf.constraint_cov.M);
  put(&p, static_cast<std::uint16_t>(kf.observability.frame));
  for (const double s : kf.observability.score) {
    put(&p, s);
  }
  put(&p, static_cast<std::uint8_t>(kf.observability.eigvecs.has_value() ? 1 : 0));
  if (kf.observability.eigvecs) {
    put_mat<6, 6>(&p, *kf.observability.eigvecs);
  }
  put_pose(&p, kf.T_body_cam);
  put(&p, static_cast<std::uint8_t>(kf.imu_summary.has_value() ? 1 : 0));
  if (kf.imu_summary) {
    const ImuPreintegrationSummary& s = *kf.imu_summary;
    put(&p, static_cast<std::int64_t>(s.t_i));
    put(&p, static_cast<std::int64_t>(s.t_j));
    put_quat(&p, s.delta_R);
    put_vec3(&p, s.delta_v);
    put_vec3(&p, s.delta_p);
    put_vec3(&p, s.bias_g_lin);
    put_vec3(&p, s.bias_a_lin);
    put_mat<3, 3>(&p, s.dR_dbg);
    put_mat<3, 3>(&p, s.dv_dbg);
    put_mat<3, 3>(&p, s.dv_dba);
    put_mat<3, 3>(&p, s.dp_dbg);
    put_mat<3, 3>(&p, s.dp_dba);
    put(&p, static_cast<std::uint8_t>(s.preint_cov.form));
    put_mat<9, 9>(&p, s.preint_cov.M);
    put(&p, s.gravity_mag);
  }
  put(&p, kf.calib_version);
  put(&p, kf.frontend_kind);
  const bool with_cloud = include_clouds_ && kf.cloud_body != nullptr;
  put(&p, static_cast<std::uint8_t>(with_cloud ? 1 : 0));
  if (with_cloud) {
    put(&p, static_cast<std::uint32_t>(kf.cloud_body->size()));
    for (const LidarPoint& pt : *kf.cloud_body) {
      put(&p, pt.xyz.x());
      put(&p, pt.xyz.y());
      put(&p, pt.xyz.z());
      put(&p, pt.intensity);
      put(&p, pt.t_offset_ns);
      put(&p, pt.ring);
      put(&p, pt.ambient);
      put(&p, pt.range);
    }
  }
  write_record(static_cast<std::uint8_t>(PacketRecord::Kind::Keyframe), p);

  const Pose& T = kf.T_ref_body;
  index_ << "kf " << kf.id << ' ' << kf.stamp << ' ' << std::fixed;
  index_.precision(6);
  index_ << T.t.x() << ' ' << T.t.y() << ' ' << T.t.z() << ' ';
  index_.precision(9);
  index_ << T.q.x() << ' ' << T.q.y() << ' ' << T.q.z() << ' ' << T.q.w() << ' ' << kf.rel_to_id
         << ' ' << constraint_kind_name(kf.constraint_kind) << '\n';
}

void PacketLogWriter::write_gnss(const GnssFix& fix, std::uint64_t nearest_kf_id) {
  std::vector<std::uint8_t> p;
  put(&p, static_cast<std::int64_t>(fix.stamp));
  put(&p, fix.sensor_id);
  put(&p, static_cast<std::uint16_t>(fix.sensor_frame));
  put(&p, fix.lat_deg);
  put(&p, fix.lon_deg);
  put(&p, fix.alt_m);
  put_mat<3, 3>(&p, fix.cov_enu);
  put(&p, static_cast<std::uint8_t>(fix.fix));
  put(&p, fix.num_sats);
  put(&p, nearest_kf_id);
  write_record(static_cast<std::uint8_t>(PacketRecord::Kind::Gnss), p);

  index_ << "gnss " << fix.stamp << ' ' << std::fixed;
  index_.precision(8);
  index_ << fix.lat_deg << ' ' << fix.lon_deg << ' ';
  index_.precision(3);
  index_ << fix.alt_m << ' ' << static_cast<int>(fix.fix) << ' '
         << static_cast<unsigned>(fix.num_sats) << ' ' << nearest_kf_id << '\n';
}

void PacketLogWriter::write_loop(const LoopConstraint& loop) {
  std::vector<std::uint8_t> p;
  put(&p, loop.from_id);
  put(&p, loop.to_id);
  put_pose(&p, loop.T_from_to);
  put(&p, static_cast<std::uint8_t>(loop.cov.form));
  put_mat<6, 6>(&p, loop.cov.M);
  put(&p, loop.fitness);
  write_record(static_cast<std::uint8_t>(PacketRecord::Kind::Loop), p);

  index_ << "loop " << loop.from_id << ' ' << loop.to_id << '\n';
}

// ---- reader ----------------------------------------------------------------------

namespace {

KeyframePacket parse_keyframe(Cursor* c) {
  KeyframePacket kf;
  kf.id = c->get<std::uint64_t>();
  kf.stamp = c->get<std::int64_t>();
  kf.ref_frame = static_cast<Frame>(c->get<std::uint16_t>());
  kf.T_ref_body = c->get_pose();
  kf.kinematics_included = c->get<std::uint8_t>() != 0;
  kf.v_ref = c->get_vec3();
  kf.b_g = c->get_vec3();
  kf.b_a = c->get_vec3();
  kf.constraint_kind = static_cast<KeyframePacket::ConstraintKind>(c->get<std::uint8_t>());
  kf.rel_to_id = c->get<std::uint64_t>();
  kf.T_relto_this = c->get_pose();
  kf.constraint_cov.form = static_cast<GaussianBlock<6>::Form>(c->get<std::uint8_t>());
  kf.constraint_cov.M = c->get_mat<6, 6>();
  kf.observability.frame = static_cast<Frame>(c->get<std::uint16_t>());
  for (double& s : kf.observability.score) {
    s = c->get<double>();
  }
  if (c->get<std::uint8_t>() != 0) {
    kf.observability.eigvecs = c->get_mat<6, 6>();
  }
  kf.T_body_cam = c->get_pose();
  if (c->get<std::uint8_t>() != 0) {
    ImuPreintegrationSummary s;
    s.t_i = c->get<std::int64_t>();
    s.t_j = c->get<std::int64_t>();
    s.delta_R = c->get_quat();
    s.delta_v = c->get_vec3();
    s.delta_p = c->get_vec3();
    s.bias_g_lin = c->get_vec3();
    s.bias_a_lin = c->get_vec3();
    s.dR_dbg = c->get_mat<3, 3>();
    s.dv_dbg = c->get_mat<3, 3>();
    s.dv_dba = c->get_mat<3, 3>();
    s.dp_dbg = c->get_mat<3, 3>();
    s.dp_dba = c->get_mat<3, 3>();
    s.preint_cov.form = static_cast<GaussianBlock<9>::Form>(c->get<std::uint8_t>());
    s.preint_cov.M = c->get_mat<9, 9>();
    s.gravity_mag = c->get<double>();
    kf.imu_summary = s;
  }
  kf.calib_version = c->get<std::uint32_t>();
  kf.frontend_kind = c->get<std::uint32_t>();
  if (c->get<std::uint8_t>() != 0) {
    const auto n = c->get<std::uint32_t>();
    auto cloud = std::make_shared<PointCloud>();
    cloud->reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      LidarPoint pt;
      pt.xyz.x() = c->get<float>();
      pt.xyz.y() = c->get<float>();
      pt.xyz.z() = c->get<float>();
      pt.intensity = c->get<float>();
      pt.t_offset_ns = c->get<std::int32_t>();
      pt.ring = c->get<std::uint16_t>();
      pt.ambient = c->get<std::uint16_t>();
      pt.range = c->get<float>();
      cloud->push_back(pt);
    }
    kf.cloud_body = std::move(cloud);
  }
  // kf.image is intentionally never serialized and stays null.
  return kf;
}

void parse_gnss(Cursor* c, GnssFix* fix, std::uint64_t* nearest_kf_id) {
  fix->stamp = c->get<std::int64_t>();
  fix->sensor_id = c->get<std::uint8_t>();
  fix->sensor_frame = static_cast<Frame>(c->get<std::uint16_t>());
  fix->lat_deg = c->get<double>();
  fix->lon_deg = c->get<double>();
  fix->alt_m = c->get<double>();
  fix->cov_enu = c->get_mat<3, 3>();
  fix->fix = static_cast<GnssFix::FixType>(c->get<std::uint8_t>());
  fix->num_sats = c->get<std::uint8_t>();
  *nearest_kf_id = c->get<std::uint64_t>();
}

LoopConstraint parse_loop(Cursor* c) {
  LoopConstraint loop;
  loop.from_id = c->get<std::uint64_t>();
  loop.to_id = c->get<std::uint64_t>();
  loop.T_from_to = c->get_pose();
  loop.cov.form = static_cast<PoseCov6::Form>(c->get<std::uint8_t>());
  loop.cov.M = c->get_mat<6, 6>();
  loop.fitness = c->get<double>();
  return loop;
}

}  // namespace

PacketLogReader::PacketLogReader(const std::string& path) : bin_(path, std::ios::binary) {
  if (!bin_) {
    throw std::runtime_error("packet log: cannot open " + path);
  }
  char magic[sizeof(kMagic)] = {};
  std::uint16_t version = 0, flags = 0;
  std::uint32_t reserved = 0;
  bin_.read(magic, sizeof(magic));
  bin_.read(reinterpret_cast<char*>(&version), sizeof(version));
  bin_.read(reinterpret_cast<char*>(&flags), sizeof(flags));
  bin_.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
  if (!bin_ || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error("packet log: bad magic in " + path);
  }
  if (version != kVersion) {
    throw std::runtime_error("packet log: unsupported version " + std::to_string(version));
  }
  clouds_included_ = (flags & kFlagCloudsIncluded) != 0;
}

bool PacketLogReader::next(PacketRecord* out) {
  std::uint8_t kind = 0;
  bin_.read(reinterpret_cast<char*>(&kind), sizeof(kind));
  if (bin_.gcount() == 0) {
    return false;  // clean EOF: the previous record was the last one
  }
  std::uint32_t len = 0;
  bin_.read(reinterpret_cast<char*>(&len), sizeof(len));
  if (bin_.gcount() != static_cast<std::streamsize>(sizeof(len))) {
    throw std::runtime_error("packet log: truncated record header");
  }
  std::vector<std::uint8_t> payload(len);
  bin_.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(len));
  if (bin_.gcount() != static_cast<std::streamsize>(len)) {
    throw std::runtime_error("packet log: truncated record payload");
  }

  *out = PacketRecord{};
  out->kind = static_cast<PacketRecord::Kind>(kind);
  Cursor c(payload.data(), payload.size());
  switch (out->kind) {
    case PacketRecord::Kind::Keyframe:
      out->kf = parse_keyframe(&c);
      break;
    case PacketRecord::Kind::Gnss:
      parse_gnss(&c, &out->fix, &out->nearest_kf_id);
      break;
    case PacketRecord::Kind::Loop:
      out->loop = parse_loop(&c);
      break;
    default:
      throw std::runtime_error("packet log: unknown record kind " + std::to_string(kind));
  }
  if (!c.done()) {
    throw std::runtime_error("packet log: payload size mismatch");
  }
  return true;
}

}  // namespace meridian
