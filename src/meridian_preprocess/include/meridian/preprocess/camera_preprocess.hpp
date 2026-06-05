#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "meridian/calib/calibration_set.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"

namespace meridian {

class TelemetrySink;

// The processed camera output handed to L2 (intensity + pyramid) and L4 (colour).
// Images are owned cv::Mat values; the pyramid is level 0 = full resolution, each
// subsequent level halved.
struct ProcessedCamera {
  CameraFrame frame;                  // stamp / sensor metadata carried through
  cv::Mat intensity;                  // single-channel, photometrically normalized
  cv::Mat colour;                     // RGB (empty for a mono source)
  std::vector<cv::Mat> pyramid;       // Gaussian pyramid of intensity, level 0 = full res
  bool photometric_calibrated = false;  // photometric normalization actually applied
  bool exposure_known = false;          // exposure_s was present (else exposure term skipped)
};

// Debayer / colour-convert to intensity (+ colour), apply photometric normalization
// where calibration data is available, and build the image pyramid. Geometric
// rectification depends on a per-camera undistort map that is not wired here; it is a
// documented seam (rectify() returns its input unchanged) rather than faked math.
//
// Thread-confined: driven from a single stage thread.
class CameraPreprocessor {
 public:
  // intrinsics carries the rectified K / distortion (used once rectification is wired).
  // telemetry may be nullptr (no-op).
  CameraPreprocessor(const PreprocCamera& cfg, const IntrinsicsCamera& intrinsics,
                     TelemetrySink* telemetry);

  // Decodes the frame to an intensity (+ colour) image and builds the pyramid.
  ProcessedCamera process(const CameraFrame& frame) const;

 private:
  // Decodes raw bytes to a single-channel intensity image and, for a colour source, the
  // RGB image. Mono passes through; Bayer is demosaiced.
  void decode(const CameraFrame& frame, cv::Mat* intensity, cv::Mat* colour) const;
  // Divides by the vignette gain map, linearizes via the CRF LUT, and normalizes
  // exposure/gain when known. No-op for the parts whose calibration data is absent.
  void photometric(const CameraFrame& frame, cv::Mat* intensity, bool* applied,
                   bool* exposure_known) const;
  // Identity seam: returns the input unchanged until a per-camera undistort-rectify map
  // is wired from calibration.
  cv::Mat rectify(const cv::Mat& img) const;
  // Builds a Gaussian pyramid with cfg_.pyramid_levels levels, halving each step.
  std::vector<cv::Mat> buildPyramid(const cv::Mat& intensity) const;

  PreprocCamera cfg_;
  IntrinsicsCamera intrinsics_;
  TelemetrySink* telemetry_ = nullptr;
};

}  // namespace meridian
