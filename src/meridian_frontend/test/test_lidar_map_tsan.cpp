// ThreadSanitizer probe for the parallel-association data race in the ikd-Tree-backed
// LidarLocalMap. Each round mutates the map (a downsample-merging insert that leaves
// pending lazy push-down flags on interior tree nodes), then many threads call
// fitPlane() concurrently -- the exact pattern associate() uses. Without
// flushPendingDeletes(), a searcher that descends into a node carrying a pending
// push-down resolves it lazily (Push_Down) by WRITING a child node's
// need_push_down/deleted flags while another concurrent searcher READS those same flags
// under no shared lock: a data race TSan flags. With the flush run once after the
// mutation and before the parallel region, every flag is already settled, so the
// searches only ever read resolved flags and the race is gone by construction.
//
// The tree is intentionally kept small (well under the ikd-Tree's 1500-point async
// rebuild threshold) so the async rebuild thread never engages: that isolates the
// search-vs-search push-down flag race -- the one the flush addresses -- from the
// orthogonal rebuild-thread interactions, which are a separate ikd-Tree concern.
//
// This is a standalone main (not a GoogleTest binary) so it can be built and run under
// -fsanitize=thread without dragging the OpenMP runtime into the sanitized image; it
// drives the concurrency with std::thread, which exercises the same map search path.

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ct/residuals_lidar.hpp"
#include "meridian/config/config.hpp"

using meridian::FrontendLidar;
using meridian::ct::LidarLocalMap;
using meridian::ct::PlaneFit;

namespace {

FrontendLidar makeCfg() {
  FrontendLidar cfg;
  cfg.voxel_map_m = 0.05;
  cfg.num_match_points = 5;
  cfg.max_match_dist_sq = 5.0;
  cfg.plane_thresh = 0.1;
  cfg.point_cov = 1e-3;
  cfg.local_map_cube_m = 6.0;
  return cfg;
}

// Six axis-aligned walls on a coarse grid. Kept small on purpose so the populated tree
// stays below the async-rebuild threshold.
std::vector<Eigen::Vector3d> wallPoints(double hx, double hy, double hz, double step) {
  std::vector<Eigen::Vector3d> pts;
  for (double a = -hx; a <= hx + 1e-9; a += step) {
    for (double b = -hz; b <= hz + 1e-9; b += step) {
      pts.emplace_back(a, -hy, b);
      pts.emplace_back(a, hy, b);
    }
  }
  for (double a = -hy; a <= hy + 1e-9; a += step) {
    for (double b = -hz; b <= hz + 1e-9; b += step) {
      pts.emplace_back(-hx, a, b);
      pts.emplace_back(hx, a, b);
    }
  }
  for (double a = -hx; a <= hx + 1e-9; a += step) {
    for (double b = -hy; b <= hy + 1e-9; b += step) {
      pts.emplace_back(a, b, -hz);
      pts.emplace_back(a, b, hz);
    }
  }
  return pts;
}

}  // namespace

int main() {
  const FrontendLidar cfg = makeCfg();
  LidarLocalMap map(cfg);

  // The tree is kept under the ikd-Tree's 1500-point async rebuild threshold so the
  // rebuild worker never schedules work; that removes the rebuild-vs-search confound and
  // leaves the search-vs-search push-down flag race as the only possible report.
  const double step = std::getenv("TSAN_STEP") ? std::atof(std::getenv("TSAN_STEP")) : 0.6;
  const double half = std::getenv("TSAN_HALF") ? std::atof(std::getenv("TSAN_HALF")) : 4.0;
  const int rounds = std::getenv("TSAN_ROUNDS") ? std::atoi(std::getenv("TSAN_ROUNDS")) : 8;
  const auto base = wallPoints(half, half, half * 0.5, step);
  map.insert(base);

  // Query points spread over the interior of all six walls so the concurrent searches
  // descend through many distinct tree nodes (the ones the mutations re-mark each round).
  const double wall = half - 0.03;
  const double ztop = half * 0.5 - 0.03;
  std::vector<Eigen::Vector3d> queries;
  for (double x = -wall; x <= wall; x += 0.08) {
    for (double z = -ztop; z <= ztop; z += 0.08) {
      queries.emplace_back(x, -wall, z);
      queries.emplace_back(x, wall, z);
    }
  }
  for (double y = -wall; y <= wall; y += 0.08) {
    for (double z = -ztop; z <= ztop; z += 0.08) {
      queries.emplace_back(-wall, y, z);
      queries.emplace_back(wall, y, z);
    }
  }

  const unsigned hw = std::max(4u, std::thread::hardware_concurrency());
  const unsigned n_threads = std::min(16u, hw);

  // A pending push-down only exists in the window between a tree modification and the
  // first traversal that settles it, so each round re-creates that window: the worker
  // threads gather at the `pre` barrier, the main thread (the single thread the barrier
  // elects to run the completion) mutates the tree and -- on the fixed path -- flushes,
  // then all threads cross into one synchronized search wave. Starting every thread at
  // the same instant maximises the chance two searchers reach the same freshly-marked
  // node within the read-vs-Push_Down-write window; many rounds accumulate that
  // probability. The `post` barrier guarantees no search is still in flight when the next
  // mutation runs, mirroring the production rule that mutation never overlaps a search.
  //
  // Re-inserting the SAME base coordinates makes every point land in a voxel it already
  // occupies, so the tree-side downsample box-delete fires (setting need_push_down_to_*
  // on the nodes bounding each touched voxel) and re-adds the same representative: the
  // flags are re-stamped broadly across the interior without the tree growing. It thus
  // never reaches the 1500-point async-rebuild threshold and the rebuild worker stays
  // idle, isolating the search-vs-search push-down race as the only one that can fire.
  std::atomic<int> hits{0};
  auto mutate = [&] {
    map.insert(base);
#ifndef TSAN_PROBE_NO_FLUSH
    // The fix under test: settle every pending push-down single-threaded before the
    // parallel search wave so concurrent searches only ever read resolved flags.
    map.flushPendingDeletes();
#endif
  };
  std::barrier pre(static_cast<std::ptrdiff_t>(n_threads), mutate);
  std::barrier post(static_cast<std::ptrdiff_t>(n_threads), [] {});

  std::vector<std::thread> pool;
  pool.reserve(n_threads);
  for (unsigned t = 0; t < n_threads; ++t) {
    pool.emplace_back([&] {
      for (int r = 0; r < rounds; ++r) {
        pre.arrive_and_wait();  // mutation (+ flush) happens here, single-threaded
        // Every thread walks the FULL query set in the SAME order, so all threads
        // descend the same root-to-node paths at the same time: that drives many
        // searchers onto the same freshly-marked node within one searcher's
        // flag-read-vs-Push_Down-write window, which is exactly the race in question.
        for (const Eigen::Vector3d& q : queries) {
          PlaneFit fit;
          if (map.fitPlane(q, &fit)) {
            hits.fetch_add(1, std::memory_order_relaxed);
          }
        }
        post.arrive_and_wait();  // no search in flight when the next mutation runs
      }
    });
  }
  for (std::thread& th : pool) {
    th.join();
  }

  std::printf("tsan probe: %u threads, %d rounds, map size %zu, accepted fits %d\n", n_threads,
              rounds, map.size(), hits.load());
  return 0;
}
