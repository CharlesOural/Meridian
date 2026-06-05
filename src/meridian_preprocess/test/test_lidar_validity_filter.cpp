#include "meridian/preprocess/ilidar_preprocessor.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "meridian/calib/extrinsic.hpp"
#include "meridian/common/cloud.hpp"
#include "meridian/common/point.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"

using meridian::Extrinsic;
using meridian::LidarPoint;
using meridian::LidarScan;
using meridian::makeLidarPreprocessor;
using meridian::PointCloud;
using meridian::PreprocessConfig;

namespace {

LidarPoint mk(float x, float y, float z, float t_ns, float intensity = 1.f) {
  LidarPoint p;
  p.xyz = Eigen::Vector3f(x, y, z);
  p.t_offset_ns = static_cast<std::int32_t>(t_ns);
  p.intensity = intensity;
  return p;
}

LidarScan scanOf(std::vector<LidarPoint> pts) {
  LidarScan s;
  s.stamp_start = 1'000;
  s.points = std::make_shared<const PointCloud>(std::move(pts));
  return s;
}

PreprocessConfig defaultCfg() {
  PreprocessConfig c;
  c.lidar.blind = 0.5;
  c.lidar.det_range = 120.0;
  c.lidar.point_filter_num = 1;  // no decimation unless a test opts in
  c.lidar.intensity_gate = false;
  return c;
}

}  // namespace

TEST(LidarValidityFilter, DropsNanBlindFarInOrder) {
  PreprocessConfig cfg = defaultCfg();
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(std::nanf(""), 0, 0, 10));        // NaN -> drop
  pts.push_back(mk(0.1f, 0.0f, 0.0f, 20));           // blind (r=0.1 < 0.5) -> drop
  pts.push_back(mk(200.f, 0.f, 0.f, 30));            // far (r=200 > 120) -> drop
  pts.push_back(mk(3.f, 0.f, 0.f, 40));              // valid

  LidarScan out = pp->process(scanOf(pts));
  ASSERT_TRUE(out.points);
  EXPECT_EQ(out.points->size(), 1u);
  EXPECT_FLOAT_EQ((*out.points)[0].xyz.x(), 3.f);
}

TEST(LidarValidityFilter, IntensityGateOnlyWhenEnabled) {
  PreprocessConfig cfg = defaultCfg();
  cfg.lidar.intensity_gate = true;
  cfg.lidar.i_min = 1.0;
  cfg.lidar.i_max = 100.0;
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(3.f, 0.f, 0.f, 10, 0.f));    // below i_min -> drop
  pts.push_back(mk(3.f, 0.f, 0.f, 20, 50.f));   // in range -> keep
  pts.push_back(mk(3.f, 0.f, 0.f, 30, 500.f));  // above i_max -> drop

  LidarScan out = pp->process(scanOf(pts));
  ASSERT_TRUE(out.points);
  EXPECT_EQ(out.points->size(), 1u);
  EXPECT_FLOAT_EQ((*out.points)[0].intensity, 50.f);
}

TEST(LidarValidityFilter, DecimationOverValidPointsOnly) {
  PreprocessConfig cfg = defaultCfg();
  cfg.lidar.point_filter_num = 2;  // keep the 2nd, 4th, ... VALID point
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, nullptr);

  // Interleave invalids so index-based decimation would keep the wrong points.
  std::vector<LidarPoint> pts;
  pts.push_back(mk(0.1f, 0.f, 0.f, 1));   // blind (invalid, not counted)
  pts.push_back(mk(2.f, 0.f, 0.f, 2));    // valid #1 -> dropped (1 % 2 != 0)
  pts.push_back(mk(0.1f, 0.f, 0.f, 3));   // blind
  pts.push_back(mk(3.f, 0.f, 0.f, 4));    // valid #2 -> KEPT
  pts.push_back(mk(4.f, 0.f, 0.f, 5));    // valid #3 -> dropped
  pts.push_back(mk(5.f, 0.f, 0.f, 6));    // valid #4 -> KEPT

  LidarScan out = pp->process(scanOf(pts));
  ASSERT_TRUE(out.points);
  ASSERT_EQ(out.points->size(), 2u);
  EXPECT_FLOAT_EQ((*out.points)[0].xyz.x(), 3.f);
  EXPECT_FLOAT_EQ((*out.points)[1].xyz.x(), 5.f);
}

TEST(LidarValidityFilter, SortsByTimeOffset) {
  PreprocessConfig cfg = defaultCfg();
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(3.f, 0.f, 0.f, 30));
  pts.push_back(mk(3.f, 0.f, 0.f, 10));
  pts.push_back(mk(3.f, 0.f, 0.f, 20));

  LidarScan out = pp->process(scanOf(pts));
  ASSERT_EQ(out.points->size(), 3u);
  EXPECT_LT((*out.points)[0].t_offset_ns, (*out.points)[1].t_offset_ns);
  EXPECT_LT((*out.points)[1].t_offset_ns, (*out.points)[2].t_offset_ns);
}

TEST(LidarValidityFilter, ReusesBufferWhenNothingDroppedAndSorted) {
  PreprocessConfig cfg = defaultCfg();  // point_filter_num == 1
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(2.f, 0.f, 0.f, 10));
  pts.push_back(mk(3.f, 0.f, 0.f, 20));
  pts.push_back(mk(4.f, 0.f, 0.f, 30));
  LidarScan in = scanOf(pts);
  const PointCloud* before = in.points.get();

  LidarScan out = pp->process(in);
  EXPECT_EQ(out.points.get(), before);  // same Shared-immutable buffer, no copy
}

TEST(LidarValidityFilter, AllocatesNewBufferWhenSomethingDropped) {
  PreprocessConfig cfg = defaultCfg();
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(0.1f, 0.f, 0.f, 10));  // blind -> dropped
  pts.push_back(mk(3.f, 0.f, 0.f, 20));
  LidarScan in = scanOf(pts);
  const PointCloud* before = in.points.get();

  LidarScan out = pp->process(in);
  EXPECT_NE(out.points.get(), before);  // a fresh buffer was allocated
  EXPECT_EQ(out.points->size(), 1u);
}

TEST(LidarValidityFilter, AllocatesNewBufferWhenUnsortedEvenIfNothingDropped) {
  PreprocessConfig cfg = defaultCfg();
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(3.f, 0.f, 0.f, 30));
  pts.push_back(mk(2.f, 0.f, 0.f, 10));
  LidarScan in = scanOf(pts);
  const PointCloud* before = in.points.get();

  LidarScan out = pp->process(in);
  EXPECT_NE(out.points.get(), before);  // reordering forces a copy
  EXPECT_EQ(out.points->size(), 2u);
}
