#include "meridian/preprocess/camera_preprocess.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

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
