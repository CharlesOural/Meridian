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
    preprocess::ImuInitProgress, preprocess::CameraPyramid, preprocess::CameraPyramidLevels,
    preprocess::CameraRaw, preprocess::CameraIntensity, preprocess::ImuInitDone,
    preprocess::ImuInitRetry, preprocess::BootstrapDrop, preprocess::DeskewHorizon,
    // frontend
    frontend::IterCount, frontend::OuterIters, frontend::SolveMs, frontend::DeadlineHit,
    frontend::BiasBounded, frontend::PriorDim, frontend::CtNCp, frontend::KeyframeCount,
    frontend::Observability, frontend::ObsMin, frontend::ObsMinScore, frontend::BiasGyrNorm,
    frontend::BiasAccNorm, frontend::VelNorm, frontend::InnovTransM, frontend::InnovRotDeg,
    frontend::LidarNInlier, frontend::LidarNFactorsKept, frontend::LidarNFactorsDropped,
    frontend::LidarInliers, frontend::VisualNTracked, frontend::VisualNCandidates,
    frontend::VisualNConverged, frontend::VisualMapPoints, frontend::VisualResMean,
    frontend::VisualExposureGain, frontend::VisualPatches, frontend::MapSize,
    frontend::MapInsertRejected, frontend::GnssAcceptRate, frontend::GnssInnovationM,
    frontend::SplineKnots, frontend::WindowBox, frontend::Keyframe, frontend::WindowRestart,
    frontend::SweepGapBridged, frontend::VisualDisabled, frontend::GnssInactive,
    frontend::GnssReject,
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
    stage::PreprocessLidarValidity, stage::BackendOptimize, stage::CtAssoc, stage::CtAssemble,
    stage::CtVisual, stage::CtVisualMap, stage::CtSolve, stage::CtSolveResidualEval,
    stage::CtSolveJacobianEval, stage::CtSolveLinearSolver, stage::CtPosecov,
    stage::CtPosecovWorker, stage::CtCovsnapshot, stage::CtCovsubmit, stage::CtMarg,
    stage::CtObs, stage::CtMap, stage::CtMapWarp, stage::CtMapInsert, stage::CtMapTrim,
    stage::CtVmapSelect, stage::CtVmapUpdate, stage::CtVmapPromote, stage::CtVmapEvict,
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
