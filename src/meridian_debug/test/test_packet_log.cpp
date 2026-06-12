#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "meridian/debug/packet_log.hpp"

namespace meridian {
namespace {

double rand_double(std::mt19937& rng) {
  return std::uniform_real_distribution<double>(-10.0, 10.0)(rng);
}

Eigen::Vector3d rand_vec3(std::mt19937& rng) {
  return {rand_double(rng), rand_double(rng), rand_double(rng)};
}

template <int R, int C>
Eigen::Matrix<double, R, C> rand_mat(std::mt19937& rng) {
  Eigen::Matrix<double, R, C> m;
  for (int r = 0; r < R; ++r) {
    for (int c = 0; c < C; ++c) {
      m(r, c) = rand_double(rng);
    }
  }
  return m;
}

Pose rand_pose(std::mt19937& rng) {
  return Pose(Eigen::Quaterniond(rand_double(rng), rand_double(rng), rand_double(rng),
                                 rand_double(rng)),
              rand_vec3(rng));
}

PointCloudPtr rand_cloud(std::mt19937& rng, std::size_t n) {
  auto cloud = std::make_shared<PointCloud>();
  std::uniform_int_distribution<std::int32_t> ti(-50000000, 50000000);
  std::uniform_int_distribution<int> u16(0, 65535);
  for (std::size_t i = 0; i < n; ++i) {
    LidarPoint p;
    p.xyz = rand_vec3(rng).cast<float>();
    p.intensity = static_cast<float>(rand_double(rng));
    p.t_offset_ns = ti(rng);
    p.ring = static_cast<std::uint16_t>(u16(rng));
    p.ambient = static_cast<std::uint16_t>(u16(rng));
    p.range = static_cast<float>(rand_double(rng));
    cloud->push_back(p);
  }
  return cloud;
}

// A keyframe exercising every optional: eigvecs, IMU summary, and a cloud.
KeyframePacket make_full_keyframe(std::mt19937& rng) {
  KeyframePacket kf;
  kf.id = 42;
  kf.stamp = 1234567891234;
  kf.ref_frame = Frame::Odom;
  kf.T_ref_body = rand_pose(rng);
  kf.kinematics_included = true;
  kf.v_ref = rand_vec3(rng);
  kf.b_g = rand_vec3(rng);
  kf.b_a = rand_vec3(rng);
  kf.constraint_kind = KeyframePacket::ConstraintKind::ImuPreintegration;
  kf.rel_to_id = 41;
  kf.T_relto_this = rand_pose(rng);
  kf.constraint_cov.form = GaussianBlock<6>::Form::Information;
  kf.constraint_cov.M = rand_mat<6, 6>(rng);
  kf.observability.frame = Frame::Body;
  for (double& s : kf.observability.score) {
    s = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
  }
  kf.observability.eigvecs = rand_mat<6, 6>(rng);
  kf.cloud_body = rand_cloud(rng, 17);
  kf.T_body_cam = rand_pose(rng);
  ImuPreintegrationSummary s;
  s.t_i = 1234567000000;
  s.t_j = 1234567891234;
  s.delta_R = rand_pose(rng).q;
  s.delta_v = rand_vec3(rng);
  s.delta_p = rand_vec3(rng);
  s.bias_g_lin = rand_vec3(rng);
  s.bias_a_lin = rand_vec3(rng);
  s.dR_dbg = rand_mat<3, 3>(rng);
  s.dv_dbg = rand_mat<3, 3>(rng);
  s.dv_dba = rand_mat<3, 3>(rng);
  s.dp_dbg = rand_mat<3, 3>(rng);
  s.dp_dba = rand_mat<3, 3>(rng);
  s.preint_cov.form = GaussianBlock<9>::Form::Covariance;
  s.preint_cov.M = rand_mat<9, 9>(rng);
  s.gravity_mag = 9.80665;
  kf.imu_summary = s;
  kf.calib_version = 3;
  kf.frontend_kind = 1;
  return kf;
}

// A keyframe with every optional absent: no eigvecs, no IMU summary, no cloud.
KeyframePacket make_minimal_keyframe(std::mt19937& rng) {
  KeyframePacket kf;
  kf.id = 43;
  kf.stamp = 1234667891234;
  kf.ref_frame = Frame::Odom;
  kf.T_ref_body = rand_pose(rng);
  kf.constraint_kind = KeyframePacket::ConstraintKind::RelativeBetween;
  kf.rel_to_id = 42;
  kf.T_relto_this = rand_pose(rng);
  kf.constraint_cov.M = rand_mat<6, 6>(rng);
  kf.observability.frame = Frame::Body;
  kf.T_body_cam = rand_pose(rng);
  return kf;
}

GnssFix make_gnss(std::mt19937& rng) {
  GnssFix fix;
  fix.stamp = 1234600000000;
  fix.sensor_id = 2;
  fix.sensor_frame = Frame::GnssLink;
  fix.lat_deg = 51.0 + rand_double(rng) * 0.01;
  fix.lon_deg = -0.1 + rand_double(rng) * 0.01;
  fix.alt_m = 80.0 + rand_double(rng);
  fix.cov_enu = rand_mat<3, 3>(rng);
  fix.fix = GnssFix::FixType::RTK_Float;
  fix.num_sats = 14;
  return fix;
}

LoopConstraint make_loop(std::mt19937& rng) {
  LoopConstraint loop;
  loop.from_id = 43;
  loop.to_id = 7;
  loop.T_from_to = rand_pose(rng);
  loop.cov.form = PoseCov6::Form::Covariance;
  loop.cov.M = rand_mat<6, 6>(rng);
  loop.fitness = 0.83;
  return loop;
}

// Element-wise exact (bitwise for finite doubles) comparison.
template <typename DA, typename DB>
void expect_mat_eq(const Eigen::MatrixBase<DA>& a, const Eigen::MatrixBase<DB>& b) {
  ASSERT_EQ(a.rows(), b.rows());
  ASSERT_EQ(a.cols(), b.cols());
  for (Eigen::Index r = 0; r < a.rows(); ++r) {
    for (Eigen::Index c = 0; c < a.cols(); ++c) {
      EXPECT_EQ(a(r, c), b(r, c)) << "element (" << r << "," << c << ")";
    }
  }
}

void expect_pose_eq(const Pose& a, const Pose& b) {
  expect_mat_eq(a.q.coeffs(), b.q.coeffs());
  expect_mat_eq(a.t, b.t);
}

void expect_keyframe_eq(const KeyframePacket& a, const KeyframePacket& b, bool expect_cloud) {
  EXPECT_EQ(a.id, b.id);
  EXPECT_EQ(a.stamp, b.stamp);
  EXPECT_EQ(a.ref_frame, b.ref_frame);
  expect_pose_eq(a.T_ref_body, b.T_ref_body);
  EXPECT_EQ(a.kinematics_included, b.kinematics_included);
  expect_mat_eq(a.v_ref, b.v_ref);
  expect_mat_eq(a.b_g, b.b_g);
  expect_mat_eq(a.b_a, b.b_a);
  EXPECT_EQ(a.constraint_kind, b.constraint_kind);
  EXPECT_EQ(a.rel_to_id, b.rel_to_id);
  expect_pose_eq(a.T_relto_this, b.T_relto_this);
  EXPECT_EQ(a.constraint_cov.form, b.constraint_cov.form);
  expect_mat_eq(a.constraint_cov.M, b.constraint_cov.M);
  EXPECT_EQ(a.observability.frame, b.observability.frame);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(a.observability.score[i], b.observability.score[i]);
  }
  ASSERT_EQ(a.observability.eigvecs.has_value(), b.observability.eigvecs.has_value());
  if (a.observability.eigvecs) {
    expect_mat_eq(*a.observability.eigvecs, *b.observability.eigvecs);
  }
  expect_pose_eq(a.T_body_cam, b.T_body_cam);
  ASSERT_EQ(a.imu_summary.has_value(), b.imu_summary.has_value());
  if (a.imu_summary) {
    const auto& sa = *a.imu_summary;
    const auto& sb = *b.imu_summary;
    EXPECT_EQ(sa.t_i, sb.t_i);
    EXPECT_EQ(sa.t_j, sb.t_j);
    expect_mat_eq(sa.delta_R.coeffs(), sb.delta_R.coeffs());
    expect_mat_eq(sa.delta_v, sb.delta_v);
    expect_mat_eq(sa.delta_p, sb.delta_p);
    expect_mat_eq(sa.bias_g_lin, sb.bias_g_lin);
    expect_mat_eq(sa.bias_a_lin, sb.bias_a_lin);
    expect_mat_eq(sa.dR_dbg, sb.dR_dbg);
    expect_mat_eq(sa.dv_dbg, sb.dv_dbg);
    expect_mat_eq(sa.dv_dba, sb.dv_dba);
    expect_mat_eq(sa.dp_dbg, sb.dp_dbg);
    expect_mat_eq(sa.dp_dba, sb.dp_dba);
    EXPECT_EQ(sa.preint_cov.form, sb.preint_cov.form);
    expect_mat_eq(sa.preint_cov.M, sb.preint_cov.M);
    EXPECT_EQ(sa.gravity_mag, sb.gravity_mag);
  }
  EXPECT_EQ(a.calib_version, b.calib_version);
  EXPECT_EQ(a.frontend_kind, b.frontend_kind);
  EXPECT_EQ(b.image, nullptr);  // the image handle is never serialized
  if (!expect_cloud || a.cloud_body == nullptr) {
    EXPECT_EQ(b.cloud_body, nullptr);
    return;
  }
  ASSERT_NE(b.cloud_body, nullptr);
  ASSERT_EQ(a.cloud_body->size(), b.cloud_body->size());
  for (std::size_t i = 0; i < a.cloud_body->size(); ++i) {
    const LidarPoint& pa = (*a.cloud_body)[i];
    const LidarPoint& pb = (*b.cloud_body)[i];
    expect_mat_eq(pa.xyz, pb.xyz);
    EXPECT_EQ(pa.intensity, pb.intensity);
    EXPECT_EQ(pa.t_offset_ns, pb.t_offset_ns);
    EXPECT_EQ(pa.ring, pb.ring);
    EXPECT_EQ(pa.ambient, pb.ambient);
    EXPECT_EQ(pa.range, pb.range);
  }
}

std::filesystem::path fresh_dir(const std::string& name) {
  const auto dir = std::filesystem::temp_directory_path() / name;
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

// Writes the four representative records and returns them for comparison.
struct Records {
  KeyframePacket kf_full, kf_min;
  GnssFix fix;
  std::uint64_t nearest_kf_id = 0;
  LoopConstraint loop;
};

Records write_log(const std::string& path, bool include_clouds) {
  std::mt19937 rng(1234);
  Records r;
  r.kf_full = make_full_keyframe(rng);
  r.kf_min = make_minimal_keyframe(rng);
  r.fix = make_gnss(rng);
  r.nearest_kf_id = 42;
  r.loop = make_loop(rng);
  PacketLogWriter writer(path, include_clouds);
  writer.write_keyframe(r.kf_full);
  writer.write_keyframe(r.kf_min);
  writer.write_gnss(r.fix, r.nearest_kf_id);
  writer.write_loop(r.loop);
  return r;
}

void expect_round_trip(const std::string& path, const Records& r, bool clouds) {
  PacketLogReader reader(path);
  EXPECT_EQ(reader.clouds_included(), clouds);

  PacketRecord rec;
  ASSERT_TRUE(reader.next(&rec));
  ASSERT_EQ(rec.kind, PacketRecord::Kind::Keyframe);
  expect_keyframe_eq(r.kf_full, rec.kf, clouds);

  ASSERT_TRUE(reader.next(&rec));
  ASSERT_EQ(rec.kind, PacketRecord::Kind::Keyframe);
  expect_keyframe_eq(r.kf_min, rec.kf, clouds);

  ASSERT_TRUE(reader.next(&rec));
  ASSERT_EQ(rec.kind, PacketRecord::Kind::Gnss);
  EXPECT_EQ(rec.fix.stamp, r.fix.stamp);
  EXPECT_EQ(rec.fix.sensor_id, r.fix.sensor_id);
  EXPECT_EQ(rec.fix.sensor_frame, r.fix.sensor_frame);
  EXPECT_EQ(rec.fix.lat_deg, r.fix.lat_deg);
  EXPECT_EQ(rec.fix.lon_deg, r.fix.lon_deg);
  EXPECT_EQ(rec.fix.alt_m, r.fix.alt_m);
  expect_mat_eq(rec.fix.cov_enu, r.fix.cov_enu);
  EXPECT_EQ(rec.fix.fix, r.fix.fix);
  EXPECT_EQ(rec.fix.num_sats, r.fix.num_sats);
  EXPECT_EQ(rec.nearest_kf_id, r.nearest_kf_id);

  ASSERT_TRUE(reader.next(&rec));
  ASSERT_EQ(rec.kind, PacketRecord::Kind::Loop);
  EXPECT_EQ(rec.loop.from_id, r.loop.from_id);
  EXPECT_EQ(rec.loop.to_id, r.loop.to_id);
  expect_pose_eq(rec.loop.T_from_to, r.loop.T_from_to);
  EXPECT_EQ(rec.loop.cov.form, r.loop.cov.form);
  expect_mat_eq(rec.loop.cov.M, r.loop.cov.M);
  EXPECT_EQ(rec.loop.fitness, r.loop.fitness);

  EXPECT_FALSE(reader.next(&rec));  // clean EOF
  EXPECT_FALSE(reader.next(&rec));  // stays at EOF
}

TEST(PacketLog, RoundTripWithClouds) {
  const auto dir = fresh_dir("meridian_test_packet_log_clouds");
  const std::string path = (dir / "log.bin").string();
  const Records r = write_log(path, /*include_clouds=*/true);
  expect_round_trip(path, r, /*clouds=*/true);
}

TEST(PacketLog, RoundTripWithoutClouds) {
  const auto dir = fresh_dir("meridian_test_packet_log_noclouds");
  const std::string path = (dir / "log.bin").string();
  const Records r = write_log(path, /*include_clouds=*/false);
  expect_round_trip(path, r, /*clouds=*/false);
}

TEST(PacketLog, TruncatedFileThrowsMidRecord) {
  const auto dir = fresh_dir("meridian_test_packet_log_trunc");
  const std::string path = (dir / "log.bin").string();
  write_log(path, /*include_clouds=*/true);
  const auto size = std::filesystem::file_size(path);
  std::filesystem::resize_file(path, size - 4);

  PacketLogReader reader(path);
  PacketRecord rec;
  EXPECT_THROW(
      {
        while (reader.next(&rec)) {
        }
      },
      std::runtime_error);
}

TEST(PacketLog, BadMagicThrows) {
  const auto dir = fresh_dir("meridian_test_packet_log_badmagic");
  const std::string path = (dir / "log.bin").string();
  {
    std::ofstream f(path, std::ios::binary);
    f << "NOTAPKTLOG_____________";
  }
  EXPECT_THROW(PacketLogReader reader(path), std::runtime_error);
}

TEST(PacketLog, IndexHasHeaderPlusOneLinePerRecord) {
  const auto dir = fresh_dir("meridian_test_packet_log_index");
  const std::string path = (dir / "log.bin").string();
  write_log(path, /*include_clouds=*/true);

  std::ifstream idx(path + ".index.txt");
  ASSERT_TRUE(idx.is_open());
  std::vector<std::string> lines;
  for (std::string line; std::getline(idx, line);) {
    lines.push_back(line);
  }
  ASSERT_EQ(lines.size(), 5u);  // header + 2 keyframes + 1 gnss + 1 loop
  EXPECT_EQ(lines[0], "# meridian packet index v1");
  EXPECT_EQ(lines[1].rfind("kf 42 ", 0), 0u);
  EXPECT_EQ(lines[2].rfind("kf 43 ", 0), 0u);
  EXPECT_EQ(lines[3].rfind("gnss ", 0), 0u);
  EXPECT_EQ(lines[4], "loop 43 7");
}

}  // namespace
}  // namespace meridian
