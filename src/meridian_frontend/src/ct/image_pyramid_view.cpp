#include "ct/image_pyramid_view.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace meridian::ct {

namespace {
// Reads one pixel of a single-channel level as a double. The camera preprocessor
// builds 8-bit pyramids; a float pyramid is also valid (e.g. a photometrically
// linearised source). Any other element type is a wiring bug, not a value to
// guess at, so it aborts in a debug build and clamps to zero in release.
double readPixel(const cv::Mat& m, int x, int y) {
  switch (m.depth()) {
    case CV_8U:
      return static_cast<double>(m.at<std::uint8_t>(y, x));
    case CV_32F:
      return static_cast<double>(m.at<float>(y, x));
    default:
      assert(false && "ImagePyramidView: pyramid level must be CV_8UC1 or CV_32FC1");
      return 0.0;
  }
}
}  // namespace

ImagePyramidView::ImagePyramidView(const std::vector<cv::Mat>& pyramid)
    : pyramid_(pyramid) {
  // Single-channel 8-bit or float is the contract; catch a mis-typed pyramid at
  // construction rather than mis-reading bytes deep in a hot sampling loop.
  for (const cv::Mat& m : pyramid_) {
    if (m.empty()) continue;
    assert(m.channels() == 1 && (m.depth() == CV_8U || m.depth() == CV_32F) &&
           "ImagePyramidView: levels must be single-channel CV_8UC1 or CV_32FC1");
  }
}

double ImagePyramidView::sampleLevel(const cv::Mat& m, double x, double y) const {
  // Clamp the sample point so the 2x2 bilinear neighbourhood always lies inside
  // the image; an out-of-range request reads the border pixel rather than UB.
  const int w = m.cols;
  const int h = m.rows;
  if (w <= 0 || h <= 0) return 0.0;
  x = std::clamp(x, 0.0, static_cast<double>(w - 1));
  y = std::clamp(y, 0.0, static_cast<double>(h - 1));
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, w - 1);
  const int y1 = std::min(y0 + 1, h - 1);
  const double ax = x - x0;
  const double ay = y - y0;
  const double i00 = readPixel(m, x0, y0);
  const double i10 = readPixel(m, x1, y0);
  const double i01 = readPixel(m, x0, y1);
  const double i11 = readPixel(m, x1, y1);
  const double top = i00 * (1.0 - ax) + i10 * ax;
  const double bot = i01 * (1.0 - ax) + i11 * ax;
  return top * (1.0 - ay) + bot * ay;
}

bool ImagePyramidView::inBounds(int level, const Eigen::Vector2d& uv,
                                int margin) const {
  if (level < 0 || level >= levels()) return false;
  const cv::Mat& m = pyramid_[static_cast<std::size_t>(level)];
  if (m.empty()) return false;
  const double scale = static_cast<double>(1 << level);
  const double x = uv.x() / scale;
  const double y = uv.y() / scale;
  const double mar = static_cast<double>(margin);
  return x >= mar && y >= mar && x <= static_cast<double>(m.cols - 1) - mar &&
         y <= static_cast<double>(m.rows - 1) - mar;
}

double ImagePyramidView::intensity(int level, const Eigen::Vector2d& uv) const {
  if (level < 0 || level >= levels()) return 0.0;
  const cv::Mat& m = pyramid_[static_cast<std::size_t>(level)];
  const double scale = static_cast<double>(1 << level);
  return sampleLevel(m, uv.x() / scale, uv.y() / scale);
}

Eigen::Vector2d ImagePyramidView::gradient(int level,
                                           const Eigen::Vector2d& uv) const {
  if (level < 0 || level >= levels()) return Eigen::Vector2d::Zero();
  const cv::Mat& m = pyramid_[static_cast<std::size_t>(level)];
  const double scale = static_cast<double>(1 << level);
  const double x = uv.x() / scale;
  const double y = uv.y() / scale;
  // Central difference in the level's pixel units; dividing by `scale` converts a
  // one-level-pixel step back into level-0 units so gradients are comparable
  // across levels.
  const double gx = 0.5 * (sampleLevel(m, x + 1.0, y) - sampleLevel(m, x - 1.0, y));
  const double gy = 0.5 * (sampleLevel(m, x, y + 1.0) - sampleLevel(m, x, y - 1.0));
  return Eigen::Vector2d(gx / scale, gy / scale);
}

}  // namespace meridian::ct
