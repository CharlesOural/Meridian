#include "meridian/preprocess/camera_preprocess.hpp"

#include <algorithm>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "meridian/debug/log.hpp"
#include "meridian/debug/telemetry.hpp"

namespace meridian {

namespace {
constexpr const char* kLogModule = "preprocess.camera";
}

CameraPreprocessor::CameraPreprocessor(const PreprocCamera& cfg,
                                       const IntrinsicsCamera& intrinsics,
                                       TelemetrySink* telemetry)
    : cfg_(cfg), intrinsics_(intrinsics), telemetry_(telemetry) {
  buildRectifyMap();
}

void CameraPreprocessor::buildRectifyMap() {
  // Default: no rectification, so the rectified intrinsics equal the source intrinsics.
  rectified_ = intrinsics_;
  rectify_valid_ = false;

  const int w = intrinsics_.width;
  const int h = intrinsics_.height;
  if (w <= 0 || h <= 0 || intrinsics_.model == IntrinsicsCamera::Distortion::None) {
    return;
  }

  const cv::Matx33d K(intrinsics_.fx, 0.0, intrinsics_.cx, 0.0, intrinsics_.fy,
                      intrinsics_.cy, 0.0, 0.0, 1.0);
  const cv::Size sz(w, h);
  const double balance = std::clamp(cfg_.rectify_balance, 0.0, 1.0);
  cv::Mat new_K;

  if (intrinsics_.model == IntrinsicsCamera::Distortion::Equidistant) {
    const cv::Vec4d D(intrinsics_.coeffs[0], intrinsics_.coeffs[1], intrinsics_.coeffs[2],
                      intrinsics_.coeffs[3]);
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, sz, cv::Matx33d::eye(),
                                                            new_K, balance, sz);
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Matx33d::eye(), new_K, sz, CV_16SC2,
                                         map1_, map2_);
  } else {
    // RadTan / plumb-bob: alpha plays the same crop-vs-keep role balance does for fisheye.
    const cv::Mat D = (cv::Mat_<double>(1, 5) << intrinsics_.coeffs[0], intrinsics_.coeffs[1],
                       intrinsics_.coeffs[2], intrinsics_.coeffs[3], intrinsics_.coeffs[4]);
    new_K = cv::getOptimalNewCameraMatrix(K, D, sz, balance, sz);
    cv::initUndistortRectifyMap(K, D, cv::Mat(), new_K, sz, CV_16SC2, map1_, map2_);
  }

  rectified_.model = IntrinsicsCamera::Distortion::None;
  rectified_.coeffs = {0, 0, 0, 0, 0};
  rectified_.fx = new_K.at<double>(0, 0);
  rectified_.fy = new_K.at<double>(1, 1);
  rectified_.cx = new_K.at<double>(0, 2);
  rectified_.cy = new_K.at<double>(1, 2);
  rectify_valid_ = true;
}

void CameraPreprocessor::decode(const CameraFrame& frame, cv::Mat* intensity,
                                cv::Mat* colour) const {
  const int rows = frame.height;
  const int cols = frame.width;
  const std::uint8_t* bytes = frame.data ? frame.data->data() : nullptr;

  if (bytes == nullptr || rows <= 0 || cols <= 0) {
    *intensity = cv::Mat();
    *colour = cv::Mat();
    return;
  }

  // Wrap the borrowed bytes without copying, then clone where an owned image is needed.
  switch (frame.encoding) {
    case CameraFrame::Encoding::Mono8: {
      const cv::Mat view(rows, cols, CV_8UC1, const_cast<std::uint8_t*>(bytes));
      *intensity = view.clone();
      *colour = cv::Mat();
      break;
    }
    case CameraFrame::Encoding::Bayer_RGGB8: {
      const cv::Mat view(rows, cols, CV_8UC1, const_cast<std::uint8_t*>(bytes));
      cv::Mat rgb;
      cv::cvtColor(view, rgb, cv::COLOR_BayerRG2RGB);
      *colour = rgb;
      cv::cvtColor(rgb, *intensity, cv::COLOR_RGB2GRAY);
      break;
    }
    case CameraFrame::Encoding::RGB8: {
      const cv::Mat view(rows, cols, CV_8UC3, const_cast<std::uint8_t*>(bytes));
      *colour = view.clone();
      cv::cvtColor(view, *intensity, cv::COLOR_RGB2GRAY);
      break;
    }
  }
}

void CameraPreprocessor::photometric(const CameraFrame& frame, cv::Mat* intensity,
                                     bool* applied, bool* exposure_known) const {
  *applied = false;
  *exposure_known = frame.exposure_s > 0.f;

  if (!cfg_.photometric_calib) {
    MERIDIAN_WARN(kLogModule, "event", "camera/no_photometric_calib", "stamp",
                  frame.stamp);
    return;
  }

  // Vignetting (divide by V(u,v)) and CRF linearization need calibration maps that are
  // not wired through config yet; those remain seams. Exposure/gain normalization needs
  // no map and runs whenever the exposure is known.
  if (*exposure_known && frame.gain > 0.f) {
    const double e_scale = cfg_.ref_exposure_s / static_cast<double>(frame.exposure_s);
    const double g_scale = cfg_.ref_gain / static_cast<double>(frame.gain);
    intensity->convertTo(*intensity, intensity->type(), e_scale * g_scale);
    *applied = true;
  }
}

cv::Mat CameraPreprocessor::rectify(const cv::Mat& img) const {
  if (!rectify_valid_ || img.empty()) return img;
  // The map is sized to the configured intrinsics; a frame of a different size cannot be
  // remapped by it, so pass it through rather than warp against a mismatched grid.
  if (img.cols != map1_.cols || img.rows != map1_.rows) return img;
  cv::Mat out;
  cv::remap(img, out, map1_, map2_, cv::INTER_LINEAR);
  return out;
}

std::vector<cv::Mat> CameraPreprocessor::buildPyramid(const cv::Mat& intensity) const {
  std::vector<cv::Mat> pyr;
  const int levels = cfg_.pyramid_levels < 1 ? 1 : cfg_.pyramid_levels;
  pyr.reserve(static_cast<std::size_t>(levels));
  pyr.push_back(intensity);
  for (int l = 1; l < levels; ++l) {
    cv::Mat down;
    cv::pyrDown(pyr.back(), down);  // Gaussian blur + halve
    pyr.push_back(down);
  }
  return pyr;
}

ProcessedCamera CameraPreprocessor::process(const CameraFrame& frame) const {
  MERIDIAN_SCOPED_TIME(telemetry_, "preprocess.camera", frame.stamp);

  ProcessedCamera out;
  out.frame = frame;

  cv::Mat intensity;
  cv::Mat colour;
  decode(frame, &intensity, &colour);

  // The decoded, un-rectified intensity is the "original" feed surfaced for inspection.
  // Clone so the in-place photometric scaling below cannot reach back into it.
  out.intensity_raw = intensity.empty() ? cv::Mat() : intensity.clone();

  // The remap is sized to the configured intrinsics; a frame at any other resolution
  // stays distorted (rectify() passes it through), so it keeps its source intrinsics
  // rather than being mislabeled with the rectified pinhole K.
  const bool size_ok = !intensity.empty() && intensity.cols == intrinsics_.width &&
                       intensity.rows == intrinsics_.height;
  if (rectify_valid_ && !intensity.empty() && !size_ok && !warned_size_mismatch_) {
    MERIDIAN_WARN(kLogModule, "event", "camera/rectify_size_mismatch", "stamp", frame.stamp);
    warned_size_mismatch_ = true;
  }

  bool applied = false;
  bool exposure_known = false;
  if (!intensity.empty()) {
    photometric(frame, &intensity, &applied, &exposure_known);
    intensity = rectify(intensity);
  }
  if (!colour.empty()) {
    colour = rectify(colour);
  }

  out.intensity = intensity;
  out.colour = colour;
  out.rectified = (rectify_valid_ && size_ok) ? rectified_ : intrinsics_;
  out.photometric_calibrated = applied;
  out.exposure_known = exposure_known;
  out.pyramid = intensity.empty() ? std::vector<cv::Mat>{} : buildPyramid(intensity);
  return out;
}

}  // namespace meridian
