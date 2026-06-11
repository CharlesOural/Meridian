#include "gicp_verify.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

#include <small_gicp/registration/registration_helper.hpp>

namespace meridian {
namespace {

// Deterministic first-point-wins voxel downsample to a homogeneous-point vector. Owning the
// downsample (rather than letting small_gicp do it internally) lets the inlier ratio use the
// downsampled source size as its denominator, and keeps the result run-to-run reproducible.
std::vector<Eigen::Vector4d> downsample(const PointCloud& c, double res) {
  const double inv = res > 0.0 ? 1.0 / res : 0.0;
  std::map<std::array<long long, 3>, Eigen::Vector4d> grid;
  for (const LidarPoint& p : c) {
    const std::array<long long, 3> key{static_cast<long long>(std::floor(p.xyz.x() * inv)),
                                       static_cast<long long>(std::floor(p.xyz.y() * inv)),
                                       static_cast<long long>(std::floor(p.xyz.z() * inv))};
    grid.emplace(key, Eigen::Vector4d(p.xyz.x(), p.xyz.y(), p.xyz.z(), 1.0));
  }
  std::vector<Eigen::Vector4d> out;
  out.reserve(grid.size());
  for (const auto& [key, v] : grid) out.push_back(v);
  return out;
}

Eigen::Isometry3d to_iso(const Pose& p) {
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.linear() = p.R();
  t.translation() = p.t;
  return t;
}

}  // namespace

GicpVerifier::GicpVerifier(const PlaceConfig& cfg, bool deterministic)
    : cfg_(cfg), num_threads_(deterministic ? 1 : std::max(1, cfg.gicp_num_threads)) {}

VerifiedLoop GicpVerifier::verify(const PointCloud& from_target, const PointCloud& to_source,
                                  const Pose& init_from_to) const {
  VerifiedLoop v;
  if (from_target.empty() || to_source.empty()) return v;

  const std::vector<Eigen::Vector4d> target = downsample(from_target, cfg_.gicp_downsample);
  const std::vector<Eigen::Vector4d> source = downsample(to_source, cfg_.gicp_downsample);
  if (target.size() < 10 || source.size() < 10) return v;

  small_gicp::RegistrationSetting s;
  s.type = small_gicp::RegistrationSetting::GICP;
  s.num_threads = num_threads_;
  // The clouds are already downsampled; keep every point so num_inliers is counted over the
  // same source whose size is the overlap denominator.
  s.downsampling_resolution = 0.01;
  s.max_correspondence_distance = cfg_.gicp_max_corr_dist;
  s.max_iterations = 30;

  const small_gicp::RegistrationResult r =
      small_gicp::align(target, source, to_iso(init_from_to), s);

  v.T_from_to = Pose{Eigen::Quaterniond(r.T_target_source.linear()).normalized(),
                     r.T_target_source.translation()};
  v.info_rot_first = r.H;

  // Overlap = fraction of the (downsampled) source that found a correspondence: a metric
  // inlier ratio in [0, 1]. This is the GICP fitness; the spec's rmse-weighted refinement
  // needs a metric RMSE (small_gicp reports a Mahalanobis cost) and is deferred.
  const double n_src = static_cast<double>(source.size());
  v.overlap =
      n_src > 0.0 ? std::clamp(static_cast<double>(r.num_inliers) / n_src, 0.0, 1.0) : 0.0;
  v.rmse = r.num_inliers > 0 ? std::sqrt(r.error / static_cast<double>(r.num_inliers))
                             : std::numeric_limits<double>::max();
  v.fitness = v.overlap;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(0.5 * (r.H + r.H.transpose()));
  const double lo = es.eigenvalues()(0);
  const double hi = es.eigenvalues()(5);
  v.cond = lo > 1e-12 ? hi / lo : std::numeric_limits<double>::max();

  v.accepted = r.converged && v.fitness >= cfg_.gicp_fitness_min &&
               v.overlap >= cfg_.gicp_overlap_min && v.cond <= cfg_.gicp_cond_max;
  return v;
}

}  // namespace meridian
