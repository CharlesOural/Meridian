#pragma once

#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace meridian::ct {

// Read-only bilinear sampler over a CameraPreprocessor intensity pyramid. Level 0
// is full resolution and each higher level is halved; a pixel coordinate `uv` is
// always expressed in level-0 pixels and divided down internally, so callers reason
// in one coordinate system regardless of which level they sample.
//
// Levels are single-channel and read on a common double scale; both the 8-bit
// (CV_8UC1) pyramid the preprocessor currently emits and a float (CV_32FC1)
// pyramid are accepted. Any other element type is rejected at construction.
//
// The pyramid is borrowed, not owned: the caller must keep the backing cv::Mat
// vector alive for the lifetime of the view.
class ImagePyramidView {
 public:
  explicit ImagePyramidView(const std::vector<cv::Mat>& pyramid);

  int levels() const { return static_cast<int>(pyramid_.size()); }

  // True when a `margin`-pixel patch centred at `uv` (level-0 coords) fits inside
  // the given level with room for the bilinear neighbours. A negative/too-large
  // level is out of bounds.
  bool inBounds(int level, const Eigen::Vector2d& uv, int margin) const;

  // Bilinearly interpolated intensity at `uv` (level-0 coords) on `level`. Out-of-
  // bounds reads are clamped to the border so a near-edge patch never reads past
  // the image; callers gate with inBounds() first when that matters.
  double intensity(int level, const Eigen::Vector2d& uv) const;

  // Central-difference image gradient [dI/du, dI/dv] in level-0 pixel units at
  // `uv` on `level`, each component bilinearly interpolated. The 2^-level factor
  // that converts a level gradient to level-0 units is folded in here.
  Eigen::Vector2d gradient(int level, const Eigen::Vector2d& uv) const;

 private:
  // Bilinear fetch in the level's own pixel coordinates (uv already downscaled).
  double sampleLevel(const cv::Mat& m, double x, double y) const;

  const std::vector<cv::Mat>& pyramid_;
};

}  // namespace meridian::ct
