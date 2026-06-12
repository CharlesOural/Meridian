#include "loop_cov.hpp"

#include <algorithm>
#include <cmath>

#include "meridian/common/cov_reorder.hpp"

namespace meridian {
namespace {

// Below this fitness the (fitness_ref/fitness)^2 scale is clamped, so a near-zero-fitness
// loop cannot blow the covariance to infinity (such loops are already rejected at Stage C).
constexpr double kFitnessFloor = 0.05;

// Smallest observability score allowed in the inflation divisor.
constexpr double kScoreFloor = 1e-3;

}  // namespace

PoseCov6 shapeLoopCov(const Eigen::Matrix<double, 6, 6>& info_rot_first, double fitness,
                      const PlaceConfig& cfg,
                      const std::optional<ObservabilityReport>& obs_from,
                      const std::optional<ObservabilityReport>& obs_to) {
  using M6 = Eigen::Matrix<double, 6, 6>;

  // 1) Sigma = s(fitness) * (H + lambda I)^-1, with degenerate eigen-directions inflated.
  const M6 hsym = 0.5 * (info_rot_first + info_rot_first.transpose());
  Eigen::SelfAdjointEigenSolver<M6> es(hsym);
  const Eigen::Matrix<double, 6, 1> eig = es.eigenvalues();  // ascending
  const M6 vecs = es.eigenvectors();

  const double s = std::pow(cfg.gicp_fitness_min / std::max(fitness, kFitnessFloor), 2.0);
  Eigen::Matrix<double, 6, 1> var;
  for (int i = 0; i < 6; ++i) {
    double v = s / (eig(i) + cfg.cov_lambda);
    if (eig(i) < cfg.cov_degenerate_eig) v *= cfg.cov_degenerate_mult;
    var(i) = v;
  }
  const M6 sigma_rf = vecs * var.asDiagonal() * vecs.transpose();  // rotation-first

  // 2) Permute rotation-first -> translation-first, exactly once, here (not in the back-end).
  M6 sigma_tf = reorderTransRotToRotTrans(sigma_rf);

  // 3) Loosen axes either endpoint reports as under-observed. Scores are translation-first
  //    [tx,ty,tz,rx,ry,rz]; D*Sigma*D with d_k = 1/score_k inflates axis-k variance by
  //    1/score_k^2 while staying PSD. (The report frame is taken as the factor frame here, a
  //    documented simplification until the Stage-B rotation into the factor frame lands.)
  Eigen::Matrix<double, 6, 1> d = Eigen::Matrix<double, 6, 1>::Ones();
  for (int k = 0; k < 6; ++k) {
    double score = 1.0;
    if (obs_from) score = std::min(score, obs_from->score[static_cast<std::size_t>(k)]);
    if (obs_to) score = std::min(score, obs_to->score[static_cast<std::size_t>(k)]);
    d(k) = 1.0 / std::clamp(score, kScoreFloor, 1.0);
  }
  sigma_tf = d.asDiagonal() * sigma_tf * d.asDiagonal();

  // 4) PSD floor so the covariance is strictly positive-definite.
  sigma_tf += cfg.cov_psd_floor * M6::Identity();

  PoseCov6 out;
  out.form = PoseCov6::Form::Covariance;
  out.M = sigma_tf;
  return out;
}

}  // namespace meridian
