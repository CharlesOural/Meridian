#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

class TelemetrySink;

// Publishes the pose sub-graph as two live markers (no GTSAM type appears here — the caller
// extracts the nodes/edges from the graph): `backend/graph_nodes` (a Points marker, one
// vertex at each keyframe's corrected map position) and `backend/graph_edges` (a LineList,
// the two endpoint positions of each relative/odometry edge). Both use id 0 so the whole
// graph redraws in place every fold rather than stacking. No-op when the sink is null or both
// keys are gated off.
void emitGraphMarkers(TelemetrySink* sink,
                      const std::vector<std::pair<std::uint64_t, Pose>>& nodes,
                      const std::vector<std::pair<std::uint64_t, std::uint64_t>>& edges,
                      Timestamp ts);

}  // namespace meridian
