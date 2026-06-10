#include "max_clique.hpp"

#include <algorithm>

namespace meridian::backend {

namespace {

// Deterministic work budget for the exact search. Bron-Kerbosch is worst-case exponential;
// on a pathological (loop-storm) consistency graph the exact path is abandoned once this many
// recursive expansions are spent and the caller falls back to the greedy clique. The bound is
// an expansion COUNT, not a wall-clock deadline, so replays stay byte-identical.
constexpr long long kMaxExpansions = 200000;

// A candidate clique is "better" than the incumbent when it is strictly larger, or the
// same size but its ascending-sorted vertex set is lexicographically smaller. Both inputs
// must already be sorted ascending.
bool is_better(const std::vector<int>& candidate, const std::vector<int>& best) {
  if (candidate.size() != best.size()) {
    return candidate.size() > best.size();
  }
  return std::lexicographical_compare(candidate.begin(), candidate.end(), best.begin(), best.end());
}

// Bron-Kerbosch with pivoting. r holds the clique being grown, p the candidates that can
// still extend it, x the vertices already explored from r. best tracks the running maximum.
// r is NOT in ascending index order (pivoting reorders the branches), so it is sorted
// before each comparison to honour the lexicographic tie-break on vertex sets.
//
// The pivot u is chosen from p u x; only candidates NOT adjacent to u are branched on,
// because any maximal clique either contains u or excludes one of u's non-neighbours. We
// pick the lowest-index pivot among those tying for the most neighbours in p so the branch
// order is fully deterministic.
// Returns false if the expansion budget was exhausted (the search was abandoned mid-way and
// its result must be discarded); true if it ran to completion.
bool bron_kerbosch(const std::vector<std::vector<bool>>& adj, std::vector<int>& r,
                   std::vector<int> p, std::vector<int> x, std::vector<int>& best,
                   long long& budget) {
  if (--budget < 0) {
    return false;
  }
  if (p.empty() && x.empty()) {
    std::vector<int> sorted_r = r;
    std::sort(sorted_r.begin(), sorted_r.end());
    if (is_better(sorted_r, best)) {
      best = std::move(sorted_r);
    }
    return true;
  }

  int pivot = -1;
  std::size_t pivot_score = 0;
  bool pivot_set = false;
  const auto count_neighbours_in_p = [&](int u) {
    std::size_t c = 0;
    for (const int v : p) {
      if (adj[u][v]) {
        ++c;
      }
    }
    return c;
  };
  for (const int u : p) {
    const std::size_t score = count_neighbours_in_p(u);
    if (!pivot_set || score > pivot_score) {
      pivot = u;
      pivot_score = score;
      pivot_set = true;
    }
  }
  for (const int u : x) {
    const std::size_t score = count_neighbours_in_p(u);
    if (!pivot_set || score > pivot_score) {
      pivot = u;
      pivot_score = score;
      pivot_set = true;
    }
  }

  // Branch on candidates not adjacent to the pivot, in ascending index order.
  std::vector<int> branch;
  branch.reserve(p.size());
  for (const int v : p) {
    if (v == pivot || !adj[pivot][v]) {
      branch.push_back(v);
    }
  }

  for (const int v : branch) {
    std::vector<int> p_next;
    std::vector<int> x_next;
    p_next.reserve(p.size());
    x_next.reserve(x.size());
    // Exclude w == v so a set diagonal (self-loop) cannot keep v in its own candidate set
    // and recurse forever; the diagonal is defined to carry no edge.
    for (const int w : p) {
      if (w != v && adj[v][w]) {
        p_next.push_back(w);
      }
    }
    for (const int w : x) {
      if (w != v && adj[v][w]) {
        x_next.push_back(w);
      }
    }

    r.push_back(v);
    const bool completed =
        bron_kerbosch(adj, r, std::move(p_next), std::move(x_next), best, budget);
    r.pop_back();
    if (!completed) {
      return false;  // budget exhausted; abandon the whole search
    }

    // Move v from the candidate set into the explored set for sibling branches.
    p.erase(std::remove(p.begin(), p.end(), v), p.end());
    x.push_back(v);
  }
  return true;
}

// Runs the exact search under the expansion budget. Sets completed=false (and the result is
// meaningless) if the budget was exhausted.
std::vector<int> exact_max_clique(const std::vector<std::vector<bool>>& adj, int n,
                                  bool* completed) {
  std::vector<int> r;
  std::vector<int> p(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    p[static_cast<std::size_t>(i)] = i;
  }
  std::vector<int> x;
  std::vector<int> best;
  long long budget = kMaxExpansions;
  *completed = bron_kerbosch(adj, r, std::move(p), std::move(x), best, budget);
  return best;
}

std::vector<int> greedy_max_clique(const std::vector<std::vector<bool>>& adj, int n) {
  std::vector<int> best;
  for (int seed = 0; seed < n; ++seed) {
    std::vector<int> clique{seed};
    // Repeatedly append the lowest-index vertex adjacent to every current member. The seed
    // need not be the smallest index, so the grown clique is sorted before comparison to
    // honour the lexicographic tie-break; the per-seed choice is fully deterministic.
    bool extended = true;
    while (extended) {
      extended = false;
      for (int cand = 0; cand < n; ++cand) {
        if (std::find(clique.begin(), clique.end(), cand) != clique.end()) {
          continue;
        }
        bool adjacent_to_all = true;
        for (const int m : clique) {
          if (!adj[cand][m]) {
            adjacent_to_all = false;
            break;
          }
        }
        if (adjacent_to_all) {
          clique.push_back(cand);
          extended = true;
          break;
        }
      }
    }
    std::sort(clique.begin(), clique.end());
    if (is_better(clique, best)) {
      best = std::move(clique);
    }
  }
  return best;
}

}  // namespace

std::vector<int> max_clique(const std::vector<std::vector<bool>>& adjacency, int exact_cap) {
  const int n = static_cast<int>(adjacency.size());
  if (n == 0) {
    return {};
  }
  if (n <= exact_cap) {
    bool completed = false;
    const std::vector<int> exact = exact_max_clique(adjacency, n, &completed);
    if (completed) {
      return exact;
    }
    // The exact search blew its budget on a pathological graph; fall back to the greedy clique.
  }
  return greedy_max_clique(adjacency, n);
}

}  // namespace meridian::backend
