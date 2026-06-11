#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include <Eigen/Core>

#include "meridian/common/cloud.hpp"

namespace meridian {

// One Stage-A retrieval candidate: an older keyframe whose Scan Context is close to the
// query's, with the coarse yaw (query expressed in the candidate frame) implied by the
// best column shift.
struct ScCandidate {
  std::uint64_t id = 0;
  double sc_dist = 0;        // Scan Context cosine distance in [0, 1] (0 = identical)
  double yaw_guess_rad = 0;  // coarse Delta-psi about +z that rotates candidate onto query
};

// Scan Context descriptor database (Stage A). Each keyframe cloud becomes a polar
// max-height image; retrieval ranks older keyframes by ring-key nearest-neighbour then by
// the column-shift cosine distance, recovering a coarse yaw. The descriptor math is the
// canonical Scan Context (rowwise ring key, columnwise sector key, circular-shift distance)
// operating on Meridian point clouds.
class ScanContextDb {
 public:
  ScanContextDb(int num_ring, int num_sector, double max_radius, int knn, double dist_thresh);

  void add(std::uint64_t id, const PointCloud& cloud_body);
  bool has(std::uint64_t id) const;
  std::size_t size() const;

  // Rank `eligible` keyframes against the query: ring-key KNN preselection, then full
  // column-shift distance; keep those under the distance threshold, sorted by (sc_dist, id),
  // at most `topK`. `eligible` is the already-gated candidate set. Deterministic.
  std::vector<ScCandidate> retrieve(std::uint64_t query_id,
                                    const std::vector<std::uint64_t>& eligible,
                                    int topK) const;

 private:
  Eigen::MatrixXd make_scan_context(const PointCloud& cloud) const;
  // (min cosine distance over column shifts, best shift in [0, Ns)).
  std::pair<double, int> distance_between(const Eigen::MatrixXd& a,
                                          const Eigen::MatrixXd& b) const;

  int nr_, ns_, knn_;
  double rmax_, dist_thresh_;

  struct Entry {
    Eigen::MatrixXd desc;       // nr_ x ns_ polar max-height image
    Eigen::VectorXd ring_key;   // rowwise mean (rotation-invariant), length nr_
  };
  std::map<std::uint64_t, Entry> db_;  // ordered by id for deterministic iteration
};

}  // namespace meridian
