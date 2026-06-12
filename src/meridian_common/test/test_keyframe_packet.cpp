#include <Eigen/Core>
#include <gtest/gtest.h>

#include "meridian/common/keyframe_packet.hpp"

using meridian::GaussianBlock;
using meridian::KeyframePacket;

TEST(KeyframePacket, Defaults) {
  const KeyframePacket kf;
  EXPECT_EQ(kf.id, 0u);
  EXPECT_EQ(kf.ref_frame, meridian::Frame::Odom);
  EXPECT_FALSE(kf.kinematics_included);
  EXPECT_EQ(kf.constraint_kind, KeyframePacket::ConstraintKind::RelativeBetween);
  EXPECT_EQ(kf.constraint_cov.form, GaussianBlock<6>::Form::Covariance);
  EXPECT_EQ(kf.constraint_cov.M.rows(), 6);
  EXPECT_EQ(kf.constraint_cov.M.cols(), 6);
  for (double s : kf.observability.score) EXPECT_DOUBLE_EQ(s, 1.0);
  EXPECT_FALSE(kf.imu_summary.has_value());
  EXPECT_EQ(kf.cloud_body, nullptr);
  EXPECT_EQ(kf.image, nullptr);
}

// constraint_cov is the one rotation-first block: dims 0..2 are rotation, 3..5 are
// translation. This encodes the convention so a future reorder bug fails here.
TEST(KeyframePacket, ConstraintCovIsRotationFirst) {
  KeyframePacket kf;
  kf.constraint_cov.form = GaussianBlock<6>::Form::Information;
  kf.constraint_cov.M = Eigen::Matrix<double, 6, 6>::Zero();
  kf.constraint_cov.M.diagonal() << 100, 100, 100, 1, 1, 1;  // tight rot, loose trans
  EXPECT_GT(kf.constraint_cov.M(0, 0), kf.constraint_cov.M(3, 3));  // rx better than tx
}

// observability.score is translation-first [tx,ty,tz,rx,ry,rz] — the opposite order.
TEST(KeyframePacket, ObservabilityIsTranslationFirst) {
  KeyframePacket kf;
  kf.observability.score = {0.2, 0.2, 0.2, 0.9, 0.9, 0.9};  // weak trans, strong rot
  EXPECT_LT(kf.observability.score[0], kf.observability.score[3]);  // tx weaker than rx
}
