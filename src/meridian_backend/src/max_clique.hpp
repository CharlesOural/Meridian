#pragma once
#include <cstddef>
#include <vector>
namespace meridian::backend {
// Maximum-cardinality clique of an undirected graph on n nodes (0..n-1). adjacency is a
// symmetric n x n matrix: adjacency[i][j] == adjacency[j][i] == true iff i and j are
// connected. The diagonal is ignored. Returns the node indices of one maximum clique in
// ascending order; among equal-size maxima the lexicographically smallest index set is
// returned (fully deterministic). n == 0 -> {}; n >= 1 -> at least a 1-node clique.
// Exact Bron-Kerbosch with pivoting when n <= exact_cap; a deterministic greedy clique
// (seed each node in turn, extend by the lowest-index node adjacent to all chosen, keep the
// best over seeds) when n > exact_cap.
std::vector<int> max_clique(const std::vector<std::vector<bool>>& adjacency, int exact_cap);
}  // namespace meridian::backend
