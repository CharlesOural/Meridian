#include "graph_markers.hpp"

#include <unordered_map>

#include "meridian/debug/telemetry.hpp"
#include "meridian/debug/telemetry_keys.hpp"

namespace meridian {

void emitGraphMarkers(TelemetrySink* sink,
                      const std::vector<std::pair<std::uint64_t, Pose>>& nodes,
                      const std::vector<std::pair<std::uint64_t, std::uint64_t>>& edges,
                      Timestamp ts) {
  if (sink == nullptr) return;
  const bool want_nodes = sink->enabled(keys::backend::GraphNodes);
  const bool want_edges = sink->enabled(keys::backend::GraphEdges);
  if (!want_nodes && !want_edges) return;

  std::unordered_map<std::uint64_t, Eigen::Vector3f> pos;
  pos.reserve(nodes.size());

  Marker nm;
  nm.type = Marker::Type::Points;
  nm.frame = Frame::Map;
  nm.ns = keys::backend::GraphNodes;
  nm.id = 0;
  nm.color = {0.2f, 0.9f, 1.0f, 1.0f};  // cyan
  nm.scale = 0.15f;
  nm.points.reserve(nodes.size());
  for (const auto& [id, T] : nodes) {
    const Eigen::Vector3f p = T.t.cast<float>();
    pos[id] = p;
    if (want_nodes) nm.points.push_back(p);
  }
  if (want_nodes) sink->marker(nm, ts);

  if (want_edges) {
    Marker em;
    em.type = Marker::Type::LineList;
    em.frame = Frame::Map;
    em.ns = keys::backend::GraphEdges;
    em.id = 0;
    em.color = {0.6f, 0.6f, 0.6f, 0.8f};  // dim gray
    em.scale = 0.04f;
    em.points.reserve(edges.size() * 2);
    for (const auto& [a, b] : edges) {
      const auto ia = pos.find(a);
      const auto ib = pos.find(b);
      if (ia == pos.end() || ib == pos.end()) continue;
      em.points.push_back(ia->second);
      em.points.push_back(ib->second);
    }
    sink->marker(em, ts);
  }
}

}  // namespace meridian
