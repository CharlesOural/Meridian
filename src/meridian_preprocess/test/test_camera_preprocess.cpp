#include "meridian/preprocess/camera_preprocess.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "meridian/calib/intrinsics.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"

using meridian::CameraFrame;
using meridian::CameraPreprocessor;
using meridian::IntrinsicsCamera;
using meridian::PreprocCamera;
using meridian::ProcessedCamera;

namespace {

CameraFrame monoFrame(int w, int h) {
  CameraFrame f;
  f.width = w;
  f.height = h;
  f.encoding = CameraFrame::Encoding::Mono8;
  f.exposure_s = 0.01f;
  f.gain = 1.f;
  auto buf = std::make_shared<std::vector<std::uint8_t>>(
      static_cast<std::size_t>(w * h), 128);
  f.data = buf;
  return f;
}

// Newer College Alphasense cam0: a wide fisheye (Kalibr equidistant).
IntrinsicsCamera ncdFisheye(int w, int h) {
  IntrinsicsCamera k;
  k.fx = 352.779;
  k.fy = 354.609;
  k.cx = 359.035;
  k.cy = 260.546;
  k.model = IntrinsicsCamera::Distortion::Equidistant;
  k.coeffs = {-0.04217, -0.00413, 0.00179, -0.00063, 0.0};
  k.width = w;
  k.height = h;
  return k;
}

}  // namespace

TEST(CameraPreprocessor, BuildsRequestedPyramidLevels) {
  PreprocCamera cfg;
  cfg.pyramid_levels = 3;
  cfg.photometric_calib = false;  // avoid scaling so we test only structure
  CameraPreprocessor pp(cfg, IntrinsicsCamera{}, nullptr);

  ProcessedCamera out = pp.process(monoFrame(64, 64));
  ASSERT_EQ(out.pyramid.size(), 3u);
  // Level 0 is full resolution; each subsequent level halves.
  EXPECT_EQ(out.pyramid[0].cols, 64);
  EXPECT_EQ(out.pyramid[0].rows, 64);
  EXPECT_EQ(out.pyramid[1].cols, 32);
  EXPECT_EQ(out.pyramid[2].cols, 16);
}

TEST(CameraPreprocessor, MonoProducesSingleChannelIntensityNoColour) {
  PreprocCamera cfg;
  cfg.pyramid_levels = 1;
  CameraPreprocessor pp(cfg, IntrinsicsCamera{}, nullptr);

  ProcessedCamera out = pp.process(monoFrame(32, 32));
  ASSERT_FALSE(out.intensity.empty());
  EXPECT_EQ(out.intensity.channels(), 1);
  EXPECT_TRUE(out.colour.empty());
  EXPECT_EQ(out.pyramid.size(), 1u);
}

TEST(CameraPreprocessor, ExposureKnownFlagTracksExposure) {
  PreprocCamera cfg;
  cfg.photometric_calib = true;
  CameraPreprocessor pp(cfg, IntrinsicsCamera{}, nullptr);

  CameraFrame known = monoFrame(16, 16);
  known.exposure_s = 0.02f;
  EXPECT_TRUE(pp.process(known).exposure_known);

  CameraFrame unknown = monoFrame(16, 16);
  unknown.exposure_s = 0.f;
  EXPECT_FALSE(pp.process(unknown).exposure_known);
}

TEST(CameraPreprocessor, NoDistortionLeavesIntensityUnchanged) {
  PreprocCamera cfg;
  cfg.pyramid_levels = 1;
  cfg.photometric_calib = false;
  IntrinsicsCamera k;
  k.model = IntrinsicsCamera::Distortion::None;
  k.width = 32;
  k.height = 32;
  CameraPreprocessor pp(cfg, k, nullptr);

  ProcessedCamera out = pp.process(monoFrame(32, 32));
  EXPECT_EQ(out.rectified.model, IntrinsicsCamera::Distortion::None);
  ASSERT_FALSE(out.intensity.empty());
  ASSERT_FALSE(out.intensity_raw.empty());
  // No rectification map is built, so the processed intensity is the decoded original.
  EXPECT_EQ(cv::countNonZero(out.intensity != out.intensity_raw), 0);
}

TEST(CameraPreprocessor, EquidistantRectifiesToPinhole) {
  PreprocCamera cfg;
  cfg.pyramid_levels = 1;
  cfg.photometric_calib = false;
  CameraPreprocessor pp(cfg, ncdFisheye(720, 540), nullptr);

  ProcessedCamera out = pp.process(monoFrame(720, 540));
  // The rectified output advertises a pinhole camera with a valid focal length.
  EXPECT_EQ(out.rectified.model, IntrinsicsCamera::Distortion::None);
  EXPECT_GT(out.rectified.fx, 0.0);
  EXPECT_GT(out.rectified.fy, 0.0);
  // Size is preserved and the un-rectified original is retained alongside it.
  EXPECT_EQ(out.intensity.cols, 720);
  EXPECT_EQ(out.intensity.rows, 540);
  EXPECT_EQ(out.intensity_raw.cols, 720);
  EXPECT_EQ(out.intensity_raw.rows, 540);
}

TEST(CameraPreprocessor, RectifyBalanceFeedsTheNewCameraMatrix) {
  PreprocCamera lo;
  lo.pyramid_levels = 1;
  lo.photometric_calib = false;
  lo.rectify_balance = 0.0;
  PreprocCamera hi = lo;
  hi.rectify_balance = 1.0;

  CameraPreprocessor pp_lo(lo, ncdFisheye(720, 540), nullptr);
  CameraPreprocessor pp_hi(hi, ncdFisheye(720, 540), nullptr);
  const double fx_lo = pp_lo.process(monoFrame(720, 540)).rectified.fx;
  const double fx_hi = pp_hi.process(monoFrame(720, 540)).rectified.fx;
  // Cropping (0) vs keeping full field of view (1) yields a different rectified focal.
  EXPECT_NE(fx_lo, fx_hi);
}
