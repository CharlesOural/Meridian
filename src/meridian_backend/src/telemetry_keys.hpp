#pragma once

namespace meridian::backend {

// Scalar series.
inline constexpr const char* kTeleChi2 = "backend/chi2";
inline constexpr const char* kTeleNFactors = "backend/n_factors";
inline constexpr const char* kTeleUpdateMs = "backend/update_ms";
inline constexpr const char* kTeleRelinCount = "backend/relin_count";
inline constexpr const char* kTeleQueueDepth = "backend/queue_depth";
inline constexpr const char* kTeleOptimizeLag = "backend/optimize_lag";
inline constexpr const char* kTeleFallbackCount = "backend/fallback_count";

// Event tags.
inline constexpr const char* kTeleContiguity = "backend/contiguity";
inline constexpr const char* kTelePsdClamp = "backend/psd_clamp";
inline constexpr const char* kTeleIndeterminate = "backend/indeterminate";

// Pose key prefix for corrected keyframe poses.
inline constexpr const char* kTeleMapKeyframe = "map/keyframe";

}  // namespace meridian::backend
