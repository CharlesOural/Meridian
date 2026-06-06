#include "ct/image_pyramid_view.hpp"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

using meridian::ct::ImagePyramidView;

namespace {

// A linear intensity ramp I(x,y) = round(ax + by + c), as CV_8UC1. Bilinear
// interpolation reproduces the underlying linear function exactly at integer-
// rounded sample nodes only up to the per-pixel rounding, so we keep a,b,c chosen
// to land on integers and stay within [0,255].
cv::Mat ramp8(int w, int h, double a, double b, double c) {
  cv::Mat m(h, w, CV_8UC1);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      m.at<std::uint8_t>(y, x) =
          static_cast<std::uint8_t>(a * x + b * y + c);
    }
  }
  return m;
}

cv::Mat ramp32f(int w, int h, double a, double b, double c) {
  cv::Mat m(h, w, CV_32FC1);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      m.at<float>(y, x) = static_cast<float>(a * x + b * y + c);
    }
  }
  return m;
}

}  // namespace

TEST(ImagePyramidView, ReportsLevelCount) {
  std::vector<cv::Mat> pyr{ramp8(32, 32, 1, 0, 0), ramp8(16, 16, 1, 0, 0),
                           ramp8(8, 8, 1, 0, 0)};
  ImagePyramidView v(pyr);
  EXPECT_EQ(v.levels(), 3);
}

// Bilinear interpolation of an exact integer-valued linear ramp must reproduce
// that linear function at sub-pixel locations.
TEST(ImagePyramidView, BilinearExactOnLinearRamp) {
  const double a = 2.0, b = 3.0, c = 5.0;
  std::vector<cv::Mat> pyr{ramp8(64, 64, a, b, c)};
  ImagePyramidView v(pyr);
  for (double x : {3.0, 10.25, 20.5, 30.75}) {
    for (double y : {4.0, 12.5, 25.25}) {
      const double expected = a * x + b * y + c;
      EXPECT_NEAR(v.intensity(0, {x, y}), expected, 1e-9)
          << "at (" << x << "," << y << ")";
    }
  }
}

// Integer-node sampling returns the stored pixel verbatim.
TEST(ImagePyramidView, IntegerNodesReturnStoredPixels) {
  cv::Mat m = ramp8(20, 20, 1, 4, 2);
  std::vector<cv::Mat> pyr{m};  // the view borrows this; keep it alive
  ImagePyramidView v(pyr);
  EXPECT_DOUBLE_EQ(v.intensity(0, {7.0, 5.0}),
                   static_cast<double>(m.at<std::uint8_t>(5, 7)));
}

// A linear ramp has a constant analytic gradient (a, b); central differences in
// level-0 units recover it exactly away from the border.
TEST(ImagePyramidView, GradientConstantOnLinearRamp) {
  const double a = 2.0, b = 3.0;
  std::vector<cv::Mat> pyr{ramp8(64, 64, a, b, 1.0)};
  ImagePyramidView v(pyr);
  for (double x : {10.0, 20.5, 33.25}) {
    for (double y : {12.0, 22.5}) {
      const Eigen::Vector2d g = v.gradient(0, {x, y});
      EXPECT_NEAR(g.x(), a, 1e-9);
      EXPECT_NEAR(g.y(), b, 1e-9);
    }
  }
}

// The level-0-units gradient on a coarser level is the per-level-pixel gradient
// scaled by 2^-level. A level-1 ramp with per-level slope (a,b) reports (a/2,b/2).
TEST(ImagePyramidView, GradientScaledByLevel) {
  const double a = 4.0, b = 2.0;
  std::vector<cv::Mat> pyr{ramp8(64, 64, a, b, 0.0),
                           ramp8(32, 32, a, b, 0.0)};
  ImagePyramidView v(pyr);
  // uv is in level-0 coords; (20,20) maps to (10,10) on level 1.
  const Eigen::Vector2d g = v.gradient(1, {20.0, 20.0});
  EXPECT_NEAR(g.x(), a / 2.0, 1e-9);
  EXPECT_NEAR(g.y(), b / 2.0, 1e-9);
}

TEST(ImagePyramidView, InBoundsRespectsMarginAndScale) {
  std::vector<cv::Mat> pyr{ramp8(64, 64, 1, 1, 0), ramp8(32, 32, 1, 1, 0)};
  ImagePyramidView v(pyr);
  // Centre of level 0 with a 2px margin is comfortably inside.
  EXPECT_TRUE(v.inBounds(0, {32.0, 32.0}, 2));
  // Near the level-0 edge: 1px from the right edge fails a 2px margin.
  EXPECT_FALSE(v.inBounds(0, {62.5, 32.0}, 2));
  // The same level-0 coord on level 1 is at 31 (= edge of a 32-wide level), so
  // any positive margin is out of bounds there.
  EXPECT_FALSE(v.inBounds(1, {62.0, 32.0}, 1));
  // Out-of-range level is never in bounds.
  EXPECT_FALSE(v.inBounds(5, {10.0, 10.0}, 0));
}

// Out-of-bounds reads clamp to the border instead of faulting.
TEST(ImagePyramidView, OutOfBoundsClampsToBorder) {
  cv::Mat m = ramp8(16, 16, 1, 1, 0);
  std::vector<cv::Mat> pyr{m};  // the view borrows this; keep it alive
  ImagePyramidView v(pyr);
  const double corner = static_cast<double>(m.at<std::uint8_t>(0, 0));
  EXPECT_DOUBLE_EQ(v.intensity(0, {-5.0, -5.0}), corner);
}

// The float pyramid path reads CV_32FC1 levels on the same scale as 8-bit.
TEST(ImagePyramidView, FloatLevelsSampledLikeEightBit) {
  const double a = 1.5, b = 0.5, c = 2.0;
  std::vector<cv::Mat> pyr{ramp32f(48, 48, a, b, c)};
  ImagePyramidView v(pyr);
  EXPECT_NEAR(v.intensity(0, {10.5, 7.25}), a * 10.5 + b * 7.25 + c, 1e-5);
  const Eigen::Vector2d g = v.gradient(0, {15.0, 15.0});
  EXPECT_NEAR(g.x(), a, 1e-5);
  EXPECT_NEAR(g.y(), b, 1e-5);
}
