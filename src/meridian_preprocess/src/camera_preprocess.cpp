#include "meridian/preprocess/camera_preprocess.hpp"

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
    : cfg_(cfg), intrinsics_(intrinsics), telemetry_(telemetry) {}

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
  // Geometric rectification requires a per-camera undistort-rectify map built from
  // IntrinsicsCamera; that map is not wired here, so this is an identity seam that
  // returns the input unchanged rather than applying an unbuilt transform.
  return img;
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
  out.photometric_calibrated = applied;
  out.exposure_known = exposure_known;
  out.pyramid = intensity.empty() ? std::vector<cv::Mat>{} : buildPyramid(intensity);
  return out;
}

}  // namespace meridian
