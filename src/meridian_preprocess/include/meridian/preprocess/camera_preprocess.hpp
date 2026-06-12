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
  cv::Mat intensity;                  // single-channel, rectified + photometrically normalized
  cv::Mat intensity_raw;              // single-channel, decoded only (pre-rectify, original feed)
  cv::Mat colour;                     // RGB (empty for a mono source), rectified
  std::vector<cv::Mat> pyramid;       // Gaussian pyramid of intensity, level 0 = full res
  IntrinsicsCamera rectified;         // pinhole K of the rectified images (distortion None)
  bool photometric_calibrated = false;  // photometric normalization actually applied
  bool exposure_known = false;          // exposure_s was present (else exposure term skipped)
};

// Debayer / colour-convert to intensity (+ colour), apply photometric normalization
// where calibration data is available, undistort to a rectified pinhole image, and build
// the image pyramid. The undistort-rectify map is built once at construction from the
// camera intrinsics and applied per frame.
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
  // Applies the prebuilt undistort-rectify remap; returns the input unchanged when no map
  // was built (no distortion, or intrinsics lacking a usable size).
  cv::Mat rectify(const cv::Mat& img) const;
  // Builds a Gaussian pyramid with cfg_.pyramid_levels levels, halving each step.
  std::vector<cv::Mat> buildPyramid(const cv::Mat& intensity) const;
  // Builds map1_/map2_ and the rectified pinhole intrinsics once from intrinsics_.
  void buildRectifyMap();

  PreprocCamera cfg_;
  IntrinsicsCamera intrinsics_;
  TelemetrySink* telemetry_ = nullptr;

  cv::Mat map1_, map2_;            // undistort-rectify remap (empty if no rectification)
  bool rectify_valid_ = false;    // a usable remap was built
  IntrinsicsCamera rectified_;    // pinhole K of the rectified output (== intrinsics_ if no map)
  mutable bool warned_size_mismatch_ = false;  // one-shot guard for the size-mismatch warning
};

}  // namespace meridian
