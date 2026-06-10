#pragma once

namespace meridian::backend {

// Scalar series.
inline constexpr const char* kTeleChi2 = "backend/chi2";
inline constexpr const char* kTeleNFactors = "backend/n_factors";
inline constexpr const char* kTeleNLoops = "backend/n_loops";  // loops currently in the graph
inline constexpr const char* kTeleNGnss = "backend/n_gnss";    // admitted GNSS factors
inline constexpr const char* kTeleUpdateMs = "backend/update_ms";
inline constexpr const char* kTeleRelinCount = "backend/relin_count";
inline constexpr const char* kTeleQueueDepth = "backend/queue_depth";
inline constexpr const char* kTeleOptimizeLag = "backend/optimize_lag";
inline constexpr const char* kTeleFallbackCount = "backend/fallback_count";

inline constexpr const char* kTeleObsMin = "backend/obs_min";

// GNSS residual vector series (ENU metres, per admitted fix).
inline constexpr const char* kTeleGnssResidual = "backend/gnss/residual";

// Event tags.
inline constexpr const char* kTeleContiguity = "backend/contiguity";
inline constexpr const char* kTelePsdClamp = "backend/psd_clamp";
inline constexpr const char* kTeleIndeterminate = "backend/indeterminate";
inline constexpr const char* kTeleDegenerate = "backend/degenerate";
inline constexpr const char* kTeleInfoForm = "backend/info_form";
inline constexpr const char* kTeleObsFrame = "backend/obs_frame";
inline constexpr const char* kTeleRelinearize = "backend/relinearize";
inline constexpr const char* kTeleRestartBridge = "backend/window_restart_bridge";
inline constexpr const char* kTeleMarginalizeSkip = "backend/marginalize_skip";
inline constexpr const char* kTeleDatumLocked = "backend/gnss/datum_locked";
inline constexpr const char* kTeleGnssDisabled = "backend/gnss_disabled";
inline constexpr const char* kTeleGnssSkip = "backend/gnss_skip";
inline constexpr const char* kTeleLoopAccepted = "backend/loop_accepted";
inline constexpr const char* kTeleLoopRejectedPcm = "backend/loop_rejected_pcm";
inline constexpr const char* kTeleLoopRejectedGnc = "backend/loop_rejected_gnc";

// Loop-closure edge marker (green = admitted, red = PCM/GNC-rejected); keyed by endpoint pair.
inline constexpr const char* kTeleLoopEdge = "backend/loop_edge";

// Per-keyframe vector series; the keyframe id is appended ("backend/observability/<id>").
inline constexpr const char* kTeleObservabilityPrefix = "backend/observability/";

// Pose key prefix for corrected keyframe poses.
inline constexpr const char* kTeleMapKeyframe = "map/keyframe";

}  // namespace meridian::backend
