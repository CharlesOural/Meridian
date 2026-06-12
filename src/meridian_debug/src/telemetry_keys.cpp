#include "meridian/debug/telemetry_keys.hpp"

#include <array>

namespace meridian::keys {

namespace {

constexpr std::array kAll{
    // lidar
    lidar::NIn, lidar::NOut, lidar::NVoxelOut, lidar::NNan, lidar::NBlind, lidar::NFar,
    lidar::NSelfhit, lidar::NIntensity, lidar::SelfhitFrac,
    // gnss
    gnss::Accepted, gnss::Rejected, gnss::Spoof,
    // preprocess
    preprocess::CameraPyramid, preprocess::CameraPyramidLevels, preprocess::CameraRaw,
    preprocess::CameraIntensity,
    // frontend
    frontend::OuterIters, frontend::SolveMs, frontend::KeyframeCount, frontend::Observability,
    frontend::ObsMin, frontend::BiasGyrNorm, frontend::BiasAccNorm, frontend::VelNorm,
    frontend::AssocNAttempted, frontend::AssocNMatched, frontend::SolverGnIters,
    frontend::SolverDxNorm, frontend::SolverChi, frontend::MapVoxels, frontend::MapPoints,
    frontend::MapSize, frontend::LidarInliers, frontend::LioBeta, frontend::LioAccelVar,
    frontend::LioNCorr, frontend::LioDeskewSpanTMs, frontend::PathSample, frontend::Keyframe,
    frontend::LioInitBacklog, frontend::LioInitDone, frontend::LioError, frontend::LioGap,
    frontend::LioReject, frontend::LioReseed,
    // backend
    backend::Chi2, backend::NFactors, backend::NLoops, backend::NGnss, backend::UpdateMs,
    backend::RelinCount, backend::QueueDepth, backend::OptimizeLag, backend::FallbackCount,
    backend::ObsMin, backend::NKeyframes, backend::GnssResidual, backend::ObservabilityPrefix,
    backend::LoopEdge, backend::GraphNodes, backend::GraphEdges, backend::Contiguity,
    backend::PsdClamp, backend::Indeterminate, backend::Degenerate, backend::InfoForm,
    backend::ObsFrame, backend::Relinearize, backend::RestartBridge, backend::MarginalizeSkip,
    backend::DatumLocked, backend::GnssDisabled, backend::GnssSkip, backend::AbsoluteIgnored,
    backend::LoopAccepted,
    backend::LoopRejectedPcm, backend::LoopRejectedGnc, backend::ExtrinsicExcited,
    backend::ExtrinsicClamped, backend::ExtrinsicFrozen, backend::G2oSnapshot,
    backend::QueueOverload,
    // place
    place::BestScDist, place::NEligible, place::NCandidates, place::BestFitness,
    place::Verified, place::SelfTestRejected, place::Emitted,
    // pipeline
    pipeline::QSensorsDepth, pipeline::QSensorsDropped, pipeline::QMeasDropped,
    pipeline::QMeasDroppedImu, pipeline::QMeasDroppedSweep, pipeline::GroupImuCount,
    pipeline::GroupPoints,
    // sensors
    sensors::ImuInGroup, sensors::LidarSweepDurationMs, sensors::LidarExtrinsicDefault,
    sensors::ValidatorPrefix, sensors::RatePrefix,
    // map / body / odom / wrapper
    map::Cloud, map::Registered, map::Keyframe, body::Scan, odom::Body, wrapper::LidarCbN,
    wrapper::LidarLostUpstreamN, wrapper::LidarConvertRejectedN,
    // health
    health::ImuRateHz, health::LidarRateHz, health::CameraRateHz, health::GnssRateHz,
    health::RateHz, health::Imu, health::Lidar, health::Camera, health::Gnss, health::Root,
    health::Degrade,
    // timing stages
    stage::Preprocess, stage::PreprocessDeskew, stage::PreprocessCamera,
    stage::PreprocessLidarValidity, stage::BackendOptimize, stage::LioIngest,
};

const char* to_str(Unit u) {
  switch (u) {
    case Unit::Meters: return "m";
    case Unit::Millimeters: return "mm";
    case Unit::Ms: return "ms";
    case Unit::Hz: return "hz";
    case Unit::Count: return "count";
    case Unit::Ratio: return "ratio";
    case Unit::Degrees: return "deg";
    case Unit::MetersPerSec: return "m/s";
    case Unit::MetersPerSec2: return "m/s^2";
    case Unit::RadPerSec: return "rad/s";
    case Unit::None: break;
  }
  return "";
}

// Keys whose runtime form is <prefix><suffix>; an exact-match miss falls back to these.
constexpr std::array kDynamicPrefixes{
    backend::ObservabilityPrefix,
    sensors::ValidatorPrefix,
    sensors::RatePrefix,
};

}  // namespace

std::span<const Key> catalog() { return kAll; }

const char* unit_string(std::string_view key) {
  for (const Key& k : kAll) {
    if (k.view() == key) return to_str(k.unit);
  }
  for (const Key& p : kDynamicPrefixes) {
    const std::string_view prefix = p.view();
    if (key.size() >= prefix.size() && key.substr(0, prefix.size()) == prefix)
      return to_str(p.unit);
  }
  return "";
}

}  // namespace meridian::keys
