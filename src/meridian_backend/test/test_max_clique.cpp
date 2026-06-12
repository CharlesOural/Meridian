#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "max_clique.hpp"

using meridian::backend::max_clique;

namespace {

// Build a symmetric adjacency matrix on n nodes from an undirected edge list. The diagonal
// is left false (it is ignored by the algorithm).
std::vector<std::vector<bool>> make_adjacency(int n,
                                              const std::vector<std::pair<int, int>>& edges) {
  std::vector<std::vector<bool>> adj(static_cast<std::size_t>(n),
                                     std::vector<bool>(static_cast<std::size_t>(n), false));
  for (const auto& [a, b] : edges) {
    adj[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = true;
    adj[static_cast<std::size_t>(b)][static_cast<std::size_t>(a)] = true;
  }
  return adj;
}

// Fully connect every pair within a vertex group, appending the edges to out.
void add_clique_edges(const std::vector<int>& group, std::vector<std::pair<int, int>>& out) {
  for (std::size_t i = 0; i < group.size(); ++i) {
    for (std::size_t j = i + 1; j < group.size(); ++j) {
      out.emplace_back(group[i], group[j]);
    }
  }
}

}  // namespace

TEST(MaxClique, EmptyGraphReturnsEmpty) {
  const std::vector<std::vector<bool>> adj;
  EXPECT_EQ(max_clique(adj, 64), std::vector<int>{});
}

TEST(MaxClique, SingleNodeReturnsThatNode) {
  const auto adj = make_adjacency(1, {});
  EXPECT_EQ(max_clique(adj, 64), (std::vector<int>{0}));
}

TEST(MaxClique, EdgelessGraphReturnsLowestNode) {
  const auto adj = make_adjacency(5, {});
  EXPECT_EQ(max_clique(adj, 64), (std::vector<int>{0}));
}

TEST(MaxClique, SingleTriangleWithIsolatedNodes) {
  // {0,1,2} mutually connected; 3 and 4 isolated.
  const auto adj = make_adjacency(5, {{0, 1}, {0, 2}, {1, 2}});
  EXPECT_EQ(max_clique(adj, 64), (std::vector<int>{0, 1, 2}));
}

TEST(MaxClique, TwoDisjointTrianglesPicksLowerIndexSet) {
  std::vector<std::pair<int, int>> edges;
  add_clique_edges({0, 1, 2}, edges);
  add_clique_edges({3, 4, 5}, edges);
  const auto adj = make_adjacency(6, edges);
  EXPECT_EQ(max_clique(adj, 64), (std::vector<int>{0, 1, 2}));
}

TEST(MaxClique, FourCliqueBeatsSeparateEdge) {
  std::vector<std::pair<int, int>> edges;
  add_clique_edges({0, 1, 2, 3}, edges);
  edges.emplace_back(4, 5);
  const auto adj = make_adjacency(6, edges);
  EXPECT_EQ(max_clique(adj, 64), (std::vector<int>{0, 1, 2, 3}));
}

TEST(MaxClique, LexicographicTieBreakPicksSmallerIndexSet) {
  // On {0,1,2,3} both {0,1,2} and {1,2,3} are triangles of equal size; the only missing
  // edge is (0,3). The smaller index set {0,1,2} must win.
  const auto adj = make_adjacency(4, {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 3}});
  EXPECT_EQ(max_clique(adj, 64), (std::vector<int>{0, 1, 2}));
}

TEST(MaxClique, GreedyPathFindsKnownLargeClique) {
  // n = 8 with a planted 5-clique {2,3,4,5,6} plus a few stray edges that cannot extend
  // it. exact_cap = 4 forces the greedy fallback (n > exact_cap).
  std::vector<std::pair<int, int>> edges;
  add_clique_edges({2, 3, 4, 5, 6}, edges);
  edges.emplace_back(0, 1);
  edges.emplace_back(0, 7);
  edges.emplace_back(1, 7);
  const auto adj = make_adjacency(8, edges);

  const int exact_cap = 4;
  ASSERT_GT(static_cast<int>(adj.size()), exact_cap);
  EXPECT_EQ(max_clique(adj, exact_cap), (std::vector<int>{2, 3, 4, 5, 6}));
}

TEST(MaxClique, GreedyAndExactAgreeOnSameGraph) {
  // The same planted 5-clique solved exactly (large cap) and greedily (cap below n) must
  // yield identical results here, where greedy is provably optimal.
  std::vector<std::pair<int, int>> edges;
  add_clique_edges({2, 3, 4, 5, 6}, edges);
  edges.emplace_back(0, 1);
  const auto adj = make_adjacency(8, edges);

  const auto exact = max_clique(adj, 64);
  const auto greedy = max_clique(adj, 4);
  EXPECT_EQ(exact, (std::vector<int>{2, 3, 4, 5, 6}));
  EXPECT_EQ(greedy, exact);
}

TEST(MaxClique, GreedyResultIsSortedRegardlessOfSeedOrder) {
  // Planted 4-clique {1,2,4,6} solved on the greedy path. A seed that is not the lowest
  // member (e.g. 6) grows {6,1,2,4}; the result must still come back sorted ascending.
  std::vector<std::pair<int, int>> edges;
  add_clique_edges({1, 2, 4, 6}, edges);
  edges.emplace_back(0, 3);
  edges.emplace_back(5, 7);
  const auto adj = make_adjacency(8, edges);

  const auto greedy = max_clique(adj, 4);
  EXPECT_EQ(greedy, (std::vector<int>{1, 2, 4, 6}));
}

TEST(MaxClique, DiagonalIsIgnored) {
  // Self-loops on the diagonal must not influence the result.
  auto adj = make_adjacency(3, {{0, 1}, {0, 2}, {1, 2}});
  adj[0][0] = true;
  adj[1][1] = true;
  adj[2][2] = true;
  EXPECT_EQ(max_clique(adj, 64), (std::vector<int>{0, 1, 2}));
}
