#include "gnc_consolidation.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <cstddef>
#include <exception>

#include "gtsam_adapter.hpp"

namespace meridian::backend {

namespace {

gtsam::Key keyX(std::uint64_t id) {
  return gtsam::Symbol('x', id);
}

}  // namespace

std::vector<std::size_t> gnc_consolidate(const GncConsolidationInput& in) {
  if (in.keyframes.empty() || in.loops.empty()) {
    return {};
  }

  using gtsam::Pose3;
  using GncParams = gtsam::GncParams<gtsam::LevenbergMarquardtParams>;
  using GncOptimizer = gtsam::GncOptimizer<GncParams>;

  try {
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;

    // Loose isotropic anchor on every keyframe at its current estimate. Without these the
    // sub-graph is gauge-free (loops/odometry only fix relative geometry) and LM would be
    // singular; the prior is weak enough that consistent loops still pull the geometry.
    const auto prior_noise = gtsam::noiseModel::Isotropic::Sigma(6, in.prior_sigma);

    // Factor indices declared as certain inliers: all priors and all odometry edges. GNC only
    // re-weights the remaining (loop) factors. Indices are graph insertion order.
    GncParams::IndexVector known_inliers;

    for (const std::uint64_t id : in.keyframes) {
      const auto it = in.estimate.find(id);
      if (it == in.estimate.end()) {
        // A keyframe with no estimate cannot be anchored or linearized; bail rather than guess.
        return {};
      }
      const Pose3 anchor = to_gtsam(it->second);
      const std::size_t idx = graph.size();
      graph.emplace_shared<gtsam::PriorFactor<Pose3>>(keyX(id), anchor, prior_noise);
      known_inliers.push_back(idx);
      if (!values.exists(keyX(id))) {
        values.insert(keyX(id), anchor);
      }
    }

    for (const GncOdom& e : in.odom) {
      const std::size_t idx = graph.size();
      graph.emplace_shared<gtsam::BetweenFactor<Pose3>>(
          keyX(e.from_id), keyX(e.to_id), to_gtsam(e.T_from_to),
          gtsam::noiseModel::Gaussian::Covariance(reorder_meridian_to_gtsam(e.cov)));
      known_inliers.push_back(idx);
    }

    // Each loop's graph index, parallel to in.loops, so weights map back to handles.
    std::vector<std::size_t> loop_factor_index(in.loops.size());
    for (std::size_t i = 0; i < in.loops.size(); ++i) {
      const GncLoop& lp = in.loops[i];
      loop_factor_index[i] = graph.size();
      graph.emplace_shared<gtsam::BetweenFactor<Pose3>>(
          keyX(lp.from_id), keyX(lp.to_id), to_gtsam(lp.T_from_to),
          gtsam::noiseModel::Gaussian::Covariance(reorder_meridian_to_gtsam(lp.cov)));
    }

    GncParams params;
    params.setLossType(gtsam::GncLossType::TLS);
    params.setKnownInliers(known_inliers);

    GncOptimizer gnc(graph, values, params);
    // The inlier cost threshold (barc^2) is a property of the optimizer, not the params.
    gnc.setInlierCostThresholds(in.barc2);
    gnc.optimize();

    // Per-factor inlier weights in [0,1] over the full graph (known inliers stay at 1).
    const gtsam::Vector weights = gnc.getWeights();

    std::vector<std::size_t> rejected;
    for (std::size_t i = 0; i < in.loops.size(); ++i) {
      const auto idx = static_cast<Eigen::Index>(loop_factor_index[i]);
      if (idx < weights.size() && weights(idx) < in.reject_w) {
        rejected.push_back(in.loops[i].handle);
      }
    }
    return rejected;
  } catch (const std::exception&) {
    return {};
  }
}

}  // namespace meridian::backend
