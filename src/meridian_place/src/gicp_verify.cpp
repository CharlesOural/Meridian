#include "gicp_verify.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <small_gicp/registration/registration_helper.hpp>

namespace meridian {
namespace {

std::vector<Eigen::Vector4d> to_vec4(const PointCloud& c) {
  std::vector<Eigen::Vector4d> out;
  out.reserve(c.size());
  for (const auto& p : c)
    out.emplace_back(static_cast<double>(p.xyz.x()), static_cast<double>(p.xyz.y()),
                     static_cast<double>(p.xyz.z()), 1.0);
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

  small_gicp::RegistrationSetting s;
  s.type = small_gicp::RegistrationSetting::GICP;
  s.num_threads = num_threads_;
  s.downsampling_resolution = cfg_.gicp_downsample;
  s.max_correspondence_distance = cfg_.gicp_max_corr_dist;
  s.max_iterations = 30;

  const std::vector<Eigen::Vector4d> target = to_vec4(from_target);
  const std::vector<Eigen::Vector4d> source = to_vec4(to_source);
  const small_gicp::RegistrationResult r =
      small_gicp::align(target, source, to_iso(init_from_to), s);

  v.T_from_to = Pose{Eigen::Quaterniond(r.T_target_source.linear()).normalized(),
                     r.T_target_source.translation()};
  v.info_rot_first = r.H;

  const double n_src = static_cast<double>(to_source.size());
  v.overlap = n_src > 0.0
                  ? std::clamp(static_cast<double>(r.num_inliers) / n_src, 0.0, 1.0)
                  : 0.0;
  // small_gicp's `error` is the converged sum of inlier costs; the per-inlier root is a
  // residual proxy (not a strict metric RMSE) but is monotone in alignment quality, which is
  // all the fitness score and accept gates need.
  v.rmse = r.num_inliers > 0
               ? std::sqrt(r.error / static_cast<double>(r.num_inliers))
               : std::numeric_limits<double>::max();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(0.5 * (r.H + r.H.transpose()));
  const double lo = es.eigenvalues()(0);
  const double hi = es.eigenvalues()(5);
  v.cond = lo > 1e-12 ? hi / lo : std::numeric_limits<double>::max();

  v.fitness = v.overlap * std::exp(-v.rmse / std::max(cfg_.gicp_fit_sigma, 1e-6));
  v.accepted = r.converged && v.fitness >= cfg_.gicp_fitness_min &&
               v.overlap >= cfg_.gicp_overlap_min && v.rmse <= cfg_.gicp_rmse_max &&
               v.cond <= cfg_.gicp_cond_max;
  return v;
}

}  // namespace meridian
