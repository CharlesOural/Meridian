#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <memory>
#include <numbers>
#include <sophus/se3.hpp>
#include <vector>

#include "meridian/calib/calibration_set.hpp"
#include "meridian/common/cov_reorder.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/config/config.hpp"
#include "meridian/frontend/ifrontend.hpp"
#include "support/synthetic_world.hpp"

namespace {

using meridian::CalibrationSet;
using meridian::Frame;
using meridian::FrontendConfig;
using meridian::FrontEndDiagnostics;
using meridian::ImuSample;
using meridian::KeyframePacket;
using meridian::NavState;
using meridian::Pose;
using meridian::PreprocessedGroup;
using meridian::Timestamp;
using meridian::lio::test::SyntheticWorld;
using meridian::lio::test::WorldConfig;
using Mat6 = Eigen::Matrix<double, 6, 6>;

const Eigen::Vector3d kCamOffset{0.08, -0.02, 0.05};

std::shared_ptr<const CalibrationSet> makeCalib() {
  auto c = std::make_shared<CalibrationSet>();
  c->estimation_frame = Frame::ImuLink;
  c->imu_acc_noise = 5e-2;
  c->imu_gyr_noise = 5e-3;
  c->imu_acc_bias_rw = 1e-4;
  c->imu_gyr_bias_rw = 1e-5;
  c->version = 7;
  meridian::Extrinsic lidar;
  lidar.child = Frame::OsSensor0;
  lidar.parent = Frame::ImuLink;
  lidar.T_parent_child = Pose{};  // LiDAR coincident with the body for the synthetic rig
  c->extrinsics.push_back(lidar);
  meridian::Extrinsic cam;
  cam.child = Frame::CamLink;
  cam.parent = Frame::ImuLink;
  cam.T_parent_child = Pose{Eigen::Quaterniond::Identity(), kCamOffset};
  c->extrinsics.push_back(cam);
  return c;
}

Sophus::SE3d toSE3(const Pose& p) {
  return Sophus::SE3d{p.q, p.t};
}

// Odom -> world alignment fixed by the run's first keyframe: the estimator's origin is
// its init pose, so all ground-truth comparisons go through this constant transform.
Sophus::SE3d alignmentFrom(const SyntheticWorld& world, const KeyframePacket& first) {
  return toSE3(world.poseAt(first.stamp)) * toSE3(first.T_ref_body).inverse();
}

struct RunResult {
  std::vector<KeyframePacket> packets;
  std::vector<NavState> live_states;  // snapshot after each ingest
};

RunResult runStream(const std::vector<PreprocessedGroup>& stream,
                    const std::vector<std::vector<ImuSample>>& live_per_group = {}) {
  RunResult out;
  auto fe = meridian::makeFrontEnd(FrontendConfig{}, makeCalib(), nullptr, true);
  fe->set_keyframe_sink([&out](KeyframePacket&& kf) { out.packets.push_back(std::move(kf)); });
  for (std::size_t i = 0; i < stream.size(); ++i) {
    fe->ingest(stream[i]);
    if (i < live_per_group.size()) {
      for (const ImuSample& s : live_per_group[i]) {
        fe->ingest_imu_live(s);
      }
    }
    out.live_states.push_back(fe->live_state());
  }
  return out;
}

double minEigenvalue(const Mat6& m) {
  return Eigen::SelfAdjointEigenSolver<Mat6>(m).eigenvalues().minCoeff();
}

void expectBitEqual(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  EXPECT_TRUE((a.array() == b.array()).all()) << a.transpose() << " vs " << b.transpose();
}

}  // namespace

// A slow circuit through the box room: the final pose must stay within a loose bound
// of ground truth, and buffered live IMU must carry live_state() past the last solve.
TEST(LioFrontEnd, TracksSlowBoxRoomCircuit) {
  SyntheticWorld world{};
  const std::vector<PreprocessedGroup> stream = world.groups(120);

  std::vector<KeyframePacket> packets;
  auto fe = meridian::makeFrontEnd(FrontendConfig{}, makeCalib(), nullptr, true);
  fe->set_keyframe_sink([&packets](KeyframePacket&& kf) { packets.push_back(std::move(kf)); });
  for (const PreprocessedGroup& g : stream) {
    fe->ingest(g);
  }
  ASSERT_FALSE(packets.empty());

  const Sophus::SE3d T_align = alignmentFrom(world, packets.front());
  const NavState final_state = fe->live_state();
  EXPECT_EQ(final_state.stamp, world.groupEnd(119));
  const Sophus::SE3d err = (T_align * toSE3(final_state.T_world_body)).inverse() *
                           toSE3(world.poseAt(final_state.stamp));
  EXPECT_LT(err.translation().norm(), 0.30);
  EXPECT_LT(err.so3().log().norm() * 180.0 / std::numbers::pi, 5.0);

  const FrontEndDiagnostics diag = fe->diagnostics();
  EXPECT_GT(diag.iterations, 0);
  EXPECT_GE(diag.effective_points, FrontendConfig{}.lio.min_keypoints);
  EXPECT_FALSE(diag.deadline_hit);
  EXPECT_FALSE(diag.restarted);

  // Live runs ahead of the last registration on buffered IMU without a new solve.
  Timestamp t_live = 0;
  for (int i = 1; i <= 100; ++i) {
    t_live = world.groupEnd(119) + i * 5'000'000;
    fe->ingest_imu_live(world.imuAt(t_live));
  }
  const NavState live = fe->live_state();
  EXPECT_EQ(live.stamp, t_live);
  const Sophus::SE3d err_live =
      (T_align * toSE3(live.T_world_body)).inverse() * toSE3(world.poseAt(t_live));
  EXPECT_LT(err_live.translation().norm(), 0.5);
}

// First packet is the absolute anchor; every later packet chains a RelativeBetween to
// its predecessor with a positive-definite covariance whose rotation block leads.
TEST(LioFrontEnd, PacketStreamShapeAndCovariance) {
  SyntheticWorld world{};
  const RunResult run = runStream(world.groups(120));
  const std::vector<KeyframePacket>& pkts = run.packets;
  ASSERT_GE(pkts.size(), 5u);

  EXPECT_EQ(pkts[0].constraint_kind, KeyframePacket::ConstraintKind::AbsolutePrior);
  EXPECT_EQ(pkts[0].rel_to_id, 0u);
  EXPECT_GE(pkts[0].id, 1u);

  for (std::size_t i = 1; i < pkts.size(); ++i) {
    EXPECT_GT(pkts[i].id, pkts[i - 1].id);
    ASSERT_EQ(pkts[i].constraint_kind, KeyframePacket::ConstraintKind::RelativeBetween);
    EXPECT_EQ(pkts[i].rel_to_id, pkts[i - 1].id);
    // The chained relative transform must reproduce the absolute pose.
    const Sophus::SE3d predicted = toSE3(pkts[i - 1].T_ref_body) * toSE3(pkts[i].T_relto_this);
    EXPECT_LT((predicted.inverse() * toSE3(pkts[i].T_ref_body)).log().norm(), 1e-12);

    EXPECT_EQ(pkts[i].constraint_cov.form, meridian::GaussianBlock<6>::Form::Covariance);
    const Mat6& m = pkts[i].constraint_cov.M;
    EXPECT_TRUE((m - m.transpose()).norm() < 1e-15 * (1.0 + m.norm()));
    EXPECT_GT(minEigenvalue(m), 0.0);
    // Rotation-first layout check on physics: multi-meter lever arms make the rotation
    // marginal far tighter than translation, so the leading 3-block must be the small one.
    const double rot_trace = m.block<3, 3>(0, 0).trace();
    const double trans_trace = m.block<3, 3>(3, 3).trace();
    EXPECT_LT(rot_trace, trans_trace);
  }
}

// Hand-built sanity for the single pack-time reorder: a known trans-first diagonal
// must land with the rotation block first.
TEST(LioFrontEnd, PackTimeReorderPlacesRotationFirst) {
  Mat6 trans_first = Mat6::Zero();
  for (int i = 0; i < 6; ++i) {
    trans_first(i, i) = static_cast<double>(i + 1);  // [tx,ty,tz,rx,ry,rz] = 1..6
  }
  const Mat6 rot_first = meridian::reorderTransRotToRotTrans(trans_first);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(rot_first(i, i), static_cast<double>(i + 4));          // rx,ry,rz = 4,5,6
    EXPECT_EQ(rot_first(i + 3, i + 3), static_cast<double>(i + 1));  // tx,ty,tz = 1,2,3
  }
}

// A data hole beyond max_gap_s followed by a failed registration must reseed: the map
// restarts and the chain re-roots through one AbsolutePrior with an inflated covariance.
TEST(LioFrontEnd, GapReseedEmitsInflatedAbsolutePrior) {
  SyntheticWorld world{};
  const std::vector<PreprocessedGroup> all = world.groups(120);
  // 1 s hole (groups 40..49), then a blinded sweep so the post-gap registration
  // deterministically fails its keypoint gate.
  std::vector<PreprocessedGroup> stream(all.begin(), all.begin() + 40);
  stream.insert(stream.end(), all.begin() + 50, all.end());
  stream[40].group.scan.points = std::make_shared<meridian::PointCloud>();

  const RunResult run = runStream(stream);
  const std::vector<KeyframePacket>& pkts = run.packets;
  ASSERT_GE(pkts.size(), 5u);

  std::size_t anchor = 0;
  int n_anchors = 0;
  double max_rel_trace = 0.0;
  for (std::size_t i = 1; i < pkts.size(); ++i) {
    if (pkts[i].constraint_kind == KeyframePacket::ConstraintKind::AbsolutePrior) {
      anchor = i;
      ++n_anchors;
    } else {
      max_rel_trace = std::max(max_rel_trace, pkts[i].constraint_cov.M.trace());
    }
  }
  ASSERT_EQ(n_anchors, 1);
  ASSERT_GT(anchor, 0u);
  EXPECT_GT(pkts[anchor].stamp, stream[39].group.t_end);  // re-root happens after the hole

  EXPECT_EQ(pkts[anchor].rel_to_id, 0u);
  EXPECT_GT(minEigenvalue(pkts[anchor].constraint_cov.M), 0.0);
  // The anchor carries the first converged covariance scaled by reseed_cov_inflation,
  // which dominates every per-interval relative covariance of the run.
  EXPECT_GT(pkts[anchor].constraint_cov.M.trace(), 3.0 * max_rel_trace);

  // The relative chain re-roots at the anchor.
  ASSERT_LT(anchor + 1, pkts.size());
  EXPECT_EQ(pkts[anchor + 1].constraint_kind, KeyframePacket::ConstraintKind::RelativeBetween);
  EXPECT_EQ(pkts[anchor + 1].rel_to_id, pkts[anchor].id);
}

// Packet contract: deskewed body-frame cloud at the stamp (validated against ground
// truth), image passthrough with the extrinsic snapshot, and provenance fields.
TEST(LioFrontEnd, PacketContractCloudImageAndProvenance) {
  SyntheticWorld world{};
  const RunResult run = runStream(world.groups(40));
  const std::vector<KeyframePacket>& pkts = run.packets;
  ASSERT_GE(pkts.size(), 2u);

  for (const KeyframePacket& p : pkts) {
    EXPECT_EQ(p.frontend_kind, 2u);
    EXPECT_EQ(p.calib_version, 7u);
    EXPECT_EQ(p.ref_frame, Frame::Odom);
    EXPECT_FALSE(p.kinematics_included);
    ASSERT_TRUE(p.cloud_body);
    EXPECT_FALSE(p.cloud_body->empty());
    // Stamps are real measurement instants: sweep ends on the synthetic clock.
    EXPECT_EQ((p.stamp - meridian::lio::test::kWorldEpoch) % world.lidarPeriod(), 0);
    // Image passthrough: every synthetic group carries one.
    ASSERT_TRUE(p.image);
    expectBitEqual(p.T_body_cam.t, kCamOffset);
  }

  // cloud_body is the deskewed sweep in the body frame at the stamp: mapped through
  // the ground-truth pose, every point must land on a room surface (range noise plus
  // the constant-screw deskew approximation bound the residual).
  const KeyframePacket& p = pkts[1];
  const Sophus::SE3d T_gt = toSE3(world.poseAt(p.stamp));
  double sum = 0.0;
  double worst = 0.0;
  for (const meridian::LidarPoint& pt : *p.cloud_body) {
    const double d = world.surfaceDistance(T_gt * pt.xyz.cast<double>());
    sum += d;
    worst = std::max(worst, d);
  }
  EXPECT_LT(sum / static_cast<double>(p.cloud_body->size()), 0.02);
  EXPECT_LT(worst, 0.12);
}

// Observability scores: a closed room constrains every axis; a lone floor plane leaves
// in-plane translation and yaw about its normal unsupported.
TEST(LioFrontEnd, ObservabilityFullRoomVsSinglePlane) {
  SyntheticWorld room{};
  const RunResult room_run = runStream(room.groups(25));
  ASSERT_FALSE(room_run.packets.empty());
  const meridian::ObservabilityReport& full = room_run.packets[0].observability;
  EXPECT_EQ(full.frame, Frame::Body);
  for (const double s : full.score) {
    EXPECT_GT(s, 0.9);
  }

  WorldConfig floor_cfg;
  floor_cfg.floor_only = true;
  floor_cfg.radius_m = 0.0;  // hover at standstill above the plane
  SyntheticWorld floor{floor_cfg};
  const RunResult floor_run = runStream(floor.groups(25));
  ASSERT_FALSE(floor_run.packets.empty());
  const meridian::ObservabilityReport& plane = floor_run.packets[0].observability;
  EXPECT_LT(plane.score[0], 0.3);  // tx: sliding in the plane
  EXPECT_LT(plane.score[1], 0.3);  // ty: sliding in the plane
  EXPECT_GT(plane.score[2], 0.5);  // tz: pinned by the surface
  EXPECT_GT(plane.score[3], 0.5);  // rx
  EXPECT_GT(plane.score[4], 0.5);  // ry
  EXPECT_LT(plane.score[5], 0.3);  // rz: yaw about the plane normal
}
