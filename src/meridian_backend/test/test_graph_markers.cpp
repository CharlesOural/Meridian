#include <gtest/gtest.h>

#include <memory>

#include "meridian/backend/ibackend.hpp"
#include "meridian/debug/recording_sink.hpp"
#include "synthetic.hpp"

using meridian::BackendConfig;
using meridian::CalibrationSet;
using meridian::KeyframePacket;
using meridian::LoopConstraint;
using meridian::Marker;
using meridian::RecordingSink;
using meridian::backend::testing::make_chain;
using meridian::backend::testing::make_loop;
using meridian::backend::testing::SynthChain;
using meridian::backend::testing::SynthOptions;

namespace {

const RecordingSink::MarkerRec* last_with_ns(const RecordingSink& sink, const char* ns) {
  const RecordingSink::MarkerRec* found = nullptr;
  for (const RecordingSink::MarkerRec& m : sink.markers) {
    if (m.ns == ns) found = &m;
  }
  return found;
}

}  // namespace

TEST(GraphMarkers, EmitsNodesAndOdometryEdges) {
  SynthOptions opt;
  opt.n = 6;
  const SynthChain chain = make_chain(opt);

  RecordingSink sink;
  auto be = meridian::makeBackEnd(BackendConfig{}, std::make_shared<CalibrationSet>(), &sink,
                                  /*deterministic=*/true);
  for (const KeyframePacket& p : chain.packets) {
    be->add_keyframe(KeyframePacket(p));
    be->optimize();
  }

  const RecordingSink::MarkerRec* nodes = last_with_ns(sink, "backend/graph_nodes");
  const RecordingSink::MarkerRec* edges = last_with_ns(sink, "backend/graph_edges");
  ASSERT_NE(nodes, nullptr);
  ASSERT_NE(edges, nullptr);
  EXPECT_EQ(nodes->type, Marker::Type::Points);
  EXPECT_EQ(nodes->n_points, 6u);  // one vertex per keyframe
  EXPECT_EQ(edges->type, Marker::Type::LineList);
  EXPECT_EQ(edges->n_points, 10u);  // 5 relative-between edges x 2 endpoints
}

TEST(GraphMarkers, LoopAddsAnEdge) {
  SynthOptions opt;
  opt.n = 20;
  const SynthChain chain = make_chain(opt);

  RecordingSink sink;
  auto be = meridian::makeBackEnd(BackendConfig{}, std::make_shared<CalibrationSet>(), &sink,
                                  /*deterministic=*/true);
  for (const KeyframePacket& p : chain.packets) {
    be->add_keyframe(KeyframePacket(p));
    be->optimize();
  }
  const std::size_t edges_before = last_with_ns(sink, "backend/graph_edges")->n_points;

  be->add_loop_constraint(make_loop(chain, 3, 17, 0.05, 0.01));
  be->optimize();

  const RecordingSink::MarkerRec* edges = last_with_ns(sink, "backend/graph_edges");
  ASSERT_NE(edges, nullptr);
  EXPECT_EQ(edges->n_points, edges_before + 2);  // the admitted loop is one more LineList edge
}
