#include "sc_descriptor.hpp"

#include <algorithm>
#include <cmath>

namespace meridian {
namespace {

// Shift z so that nearly all returns land above zero: the descriptor stores the maximum z
// per polar bin and leaves empty bins at zero, so a real return must read higher than an
// empty bin for the max to be meaningful.
constexpr double kHeightOffset = 2.0;

// Azimuth in [0, 360) degrees.
double azimuth_deg(double x, double y) {
  const double a = std::atan2(y, x) * 180.0 / M_PI;  // (-180, 180]
  return a < 0.0 ? a + 360.0 : a;
}

// Mean of each row -> a vector invariant to azimuthal rotation (the ring key).
Eigen::VectorXd ring_key_of(const Eigen::MatrixXd& desc) {
  return desc.rowwise().mean();
}

// Mean of each column -> a vector that shifts with yaw (the sector key), used to coarsely
// align two contexts before the finer per-shift search.
Eigen::RowVectorXd sector_key_of(const Eigen::MatrixXd& desc) {
  return desc.colwise().mean();
}

// Rotate columns right by `shift` (wrapping), i.e. a yaw rotation of the polar image.
Eigen::MatrixXd circshift_cols(const Eigen::MatrixXd& m, int shift) {
  const int n = static_cast<int>(m.cols());
  shift = ((shift % n) + n) % n;
  if (shift == 0) return m;
  Eigen::MatrixXd out(m.rows(), n);
  out.leftCols(shift) = m.rightCols(shift);
  out.rightCols(n - shift) = m.leftCols(n - shift);
  return out;
}

// 1 - mean cosine similarity over the non-empty column pairs.
double column_cosine_distance(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  double sum = 0.0;
  int eff = 0;
  for (int c = 0; c < a.cols(); ++c) {
    const double na = a.col(c).norm();
    const double nb = b.col(c).norm();
    if (na == 0.0 || nb == 0.0) continue;
    sum += a.col(c).dot(b.col(c)) / (na * nb);
    ++eff;
  }
  if (eff == 0) return 1.0;
  return 1.0 - sum / eff;
}

}  // namespace

ScanContextDb::ScanContextDb(int num_ring, int num_sector, double max_radius, int knn,
                             double dist_thresh)
    : nr_(num_ring), ns_(num_sector), knn_(knn), rmax_(max_radius), dist_thresh_(dist_thresh) {}

Eigen::MatrixXd ScanContextDb::make_scan_context(const PointCloud& cloud) const {
  constexpr double kNoPoint = -1000.0;
  Eigen::MatrixXd desc = Eigen::MatrixXd::Constant(nr_, ns_, kNoPoint);
  for (const auto& p : cloud) {
    const double x = p.xyz.x();
    const double y = p.xyz.y();
    const double z = static_cast<double>(p.xyz.z()) + kHeightOffset;
    const double range = std::sqrt(x * x + y * y);
    if (range > rmax_ || range < 1e-6) continue;
    const int ring =
        std::clamp(static_cast<int>(std::ceil(range / rmax_ * nr_)), 1, nr_);
    const int sector =
        std::clamp(static_cast<int>(std::ceil(azimuth_deg(x, y) / 360.0 * ns_)), 1, ns_);
    double& bin = desc(ring - 1, sector - 1);
    if (bin < z) bin = z;
  }
  for (int r = 0; r < nr_; ++r)
    for (int c = 0; c < ns_; ++c)
      if (desc(r, c) == kNoPoint) desc(r, c) = 0.0;
  return desc;
}

std::pair<double, int> ScanContextDb::distance_between(const Eigen::MatrixXd& a,
                                                       const Eigen::MatrixXd& b) const {
  // Coarse align by the sector key, then search a small column-shift band around it.
  const Eigen::RowVectorXd va = sector_key_of(a);
  const Eigen::RowVectorXd vb = sector_key_of(b);
  int coarse = 0;
  double best_vk = std::numeric_limits<double>::max();
  for (int s = 0; s < ns_; ++s) {
    const double d = (va - circshift_cols(vb, s)).norm();
    if (d < best_vk) {
      best_vk = d;
      coarse = s;
    }
  }
  const int radius = static_cast<int>(std::round(0.05 * ns_));  // 10% search, half each side
  std::vector<int> shifts{coarse};
  for (int i = 1; i <= radius; ++i) {
    shifts.push_back((coarse + i) % ns_);
    shifts.push_back((coarse - i + ns_) % ns_);
  }
  std::sort(shifts.begin(), shifts.end());
  shifts.erase(std::unique(shifts.begin(), shifts.end()), shifts.end());

  double best = std::numeric_limits<double>::max();
  int best_shift = 0;
  for (const int s : shifts) {
    const double d = column_cosine_distance(a, circshift_cols(b, s));
    if (d < best) {
      best = d;
      best_shift = s;
    }
  }
  return {best, best_shift};
}

void ScanContextDb::add(std::uint64_t id, const PointCloud& cloud_body) {
  Eigen::MatrixXd desc = make_scan_context(cloud_body);
  Eigen::VectorXd key = ring_key_of(desc);
  db_[id] = Entry{std::move(desc), std::move(key)};
}

bool ScanContextDb::has(std::uint64_t id) const { return db_.find(id) != db_.end(); }

std::size_t ScanContextDb::size() const { return db_.size(); }

std::vector<ScCandidate> ScanContextDb::retrieve(std::uint64_t query_id,
                                                 const std::vector<std::uint64_t>& eligible,
                                                 int topK) const {
  const auto q = db_.find(query_id);
  if (q == db_.end()) return {};

  // Step 1: ring-key nearest neighbours (exact brute force; the ring key is a length-nr_
  // vector and the candidate set is intra-session, so this matches a KD-tree result exactly
  // while staying obviously deterministic). Ties break by id.
  std::vector<std::pair<double, std::uint64_t>> ranked;
  ranked.reserve(eligible.size());
  for (const std::uint64_t id : eligible) {
    const auto it = db_.find(id);
    if (it == db_.end()) continue;
    const double d2 = (q->second.ring_key - it->second.ring_key).squaredNorm();
    ranked.emplace_back(d2, id);
  }
  std::sort(ranked.begin(), ranked.end());
  if (static_cast<int>(ranked.size()) > knn_) ranked.resize(knn_);

  // Step 2: full column-shift distance on the preselected candidates; keep the close ones.
  std::vector<ScCandidate> out;
  const double unit = 2.0 * M_PI / static_cast<double>(ns_);
  for (const auto& [d2, id] : ranked) {
    const auto& cand = db_.at(id);
    const auto [dist, shift] = distance_between(q->second.desc, cand.desc);
    if (dist >= dist_thresh_) continue;
    double yaw = static_cast<double>(shift) * unit;
    if (yaw > M_PI) yaw -= 2.0 * M_PI;  // wrap to (-pi, pi]
    out.push_back(ScCandidate{id, dist, yaw});
  }
  std::sort(out.begin(), out.end(), [](const ScCandidate& a, const ScCandidate& b) {
    return a.sc_dist != b.sc_dist ? a.sc_dist < b.sc_dist : a.id < b.id;
  });
  if (static_cast<int>(out.size()) > topK) out.resize(topK);
  return out;
}

}  // namespace meridian
