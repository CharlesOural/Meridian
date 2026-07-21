#include "meridian/local_rt/estimator/fixed_lag_estimator.hpp"

#include <ceres/ceres.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

#include "meridian/local_rt/combined_imu_cost.hpp"
#include "meridian/local_rt/estimator/fixed_linearization_marginal_prior.hpp"
#include "meridian/local_rt/estimator/scan_to_map_cost.hpp"
#include "meridian/local_rt/right_se3_manifold.hpp"

namespace meridian::local_rt::estimator {
namespace {

using Clock = std::chrono::steady_clock;
using PoseArray = std::array<double, 7>;
using MotionArray = std::array<double, 9>;
using DynamicRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajor7x6 = Eigen::Matrix<double, 7, 6, Eigen::RowMajor>;

std::int64_t elapsedNs(const Clock::time_point begin) noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
}

bool positiveFinite(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

Sophus::SE3d sophus(const core::Pose3d& pose) {
  const core::Quaterniond& q = pose.rotation();
  const core::Vec3d& p = pose.translation();
  return Sophus::SE3d(Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z()),
                      Eigen::Vector3d(p.x, p.y, p.z));
}

core::Pose3d corePose(const PoseArray& pose) {
  Eigen::Quaterniond quaternion = Eigen::Map<const Eigen::Quaterniond>(pose.data() + 3);
  quaternion.normalize();
  return core::Pose3d(
      {.x = pose[0], .y = pose[1], .z = pose[2]},
      core::Quaterniond(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()));
}

PoseArray poseArray(const core::Pose3d& pose) {
  const core::Vec3d& p = pose.translation();
  const core::Quaterniond& q = pose.rotation();
  return {p.x, p.y, p.z, q.x(), q.y(), q.z(), q.w()};
}

MotionArray motionArray(const core::NavigationState& state) {
  const core::Vec3d& velocity = state.velocityOdomMS();
  const core::Vec3d& gyroscope = state.imuBias().gyroscopeRadS();
  const core::Vec3d& accelerometer = state.imuBias().accelerometerMS2();
  return {velocity.x,  velocity.y,      velocity.z,      gyroscope.x,    gyroscope.y,
          gyroscope.z, accelerometer.x, accelerometer.y, accelerometer.z};
}

core::NavigationState navigationState(core::StateId id, core::TimeNs time, const PoseArray& pose,
                                      const MotionArray& motion) {
  return core::NavigationState(id, time, corePose(pose),
                               {.x = motion[0], .y = motion[1], .z = motion[2]},
                               core::ImuBias({.x = motion[3], .y = motion[4], .z = motion[5]},
                                             {.x = motion[6], .y = motion[7], .z = motion[8]}));
}

void validate(const FixedLagEstimatorOptions& options) {
  if (options.maximum_states < 2U || options.maximum_lag_ns <= 0 ||
      options.maximum_solver_iterations == 0U || options.solver_threads == 0U ||
      !positiveFinite(options.maximum_solver_time_s) ||
      !positiveFinite(options.function_tolerance) || !positiveFinite(options.gradient_tolerance) ||
      !positiveFinite(options.parameter_tolerance) ||
      !positiveFinite(options.maximum_translation_correction_m) ||
      !positiveFinite(options.maximum_rotation_correction_rad) ||
      !positiveFinite(options.maximum_prior_translation_m) ||
      !positiveFinite(options.maximum_prior_rotation_rad) ||
      !positiveFinite(options.maximum_prior_motion_norm) ||
      !positiveFinite(options.initial_translation_sigma_m) ||
      !positiveFinite(options.initial_rotation_sigma_rad) ||
      !positiveFinite(options.initial_velocity_sigma_m_s) ||
      !positiveFinite(options.initial_gyro_bias_sigma_rad_s) ||
      !positiveFinite(options.initial_accel_bias_sigma_m_s2)) {
    throw std::invalid_argument("fixed-lag estimator options are incomplete or nonphysical");
  }
}

enum class BlockKind : std::uint8_t { kPose, kMotion };

struct BlockKey final {
  core::StateId state_id;
  BlockKind kind;
};

}  // namespace

struct FixedLagEstimator::Impl final {
  struct StateNode final {
    core::StateId id;
    core::TimeNs time;
    PoseArray pose;
    MotionArray motion;

    [[nodiscard]] core::NavigationState state() const {
      return navigationState(id, time, pose, motion);
    }
  };

  struct ImuFactor final {
    core::StateId from;
    core::StateId to;
    CombinedPreintegration preintegration;
  };

  struct ActiveLidarGroup final {
    std::uint64_t batch_id{};
    core::StateId source;
    core::StateId target;
    std::vector<ActiveLidarRow> rows;
  };

  struct FinalizedLidarGroup final {
    std::uint64_t batch_id{};
    core::StateId source;
    std::vector<FinalizedLidarRow> rows;
  };

  struct PriorData final {
    std::vector<MarginalPriorLinearizationBlock> blocks;
    Eigen::MatrixXd matrix;
    Eigen::VectorXd rhs;
  };

  struct EvaluatedFactor final {
    Eigen::VectorXd residual;
    std::vector<std::pair<BlockKey, Eigen::MatrixXd>> local_jacobians;
  };

  struct SolvePass final {
    ceres::Solver::Summary summary;
    std::int64_t problem_build_ns{};
    std::int64_t solve_ns{};
  };

  FixedLagEstimatorOptions options;
  core::Pose3d T_imu_lidar;
  lidar::ScanToMapTarget target;
  std::deque<StateNode> states;
  std::vector<ImuFactor> imu_factors;
  std::vector<ActiveLidarGroup> active_lidar_groups;
  std::vector<FinalizedLidarGroup> finalized_lidar_groups;
  std::optional<PriorData> prior;
  std::uint64_t next_batch_id{1U};

  Impl(FixedLagEstimatorOptions options_value, lidar::ScanToMapOptions scan_options,
       core::Pose3d extrinsic)
      : options(std::move(options_value)),
        T_imu_lidar(std::move(extrinsic)),
        target(std::move(scan_options), sophus(T_imu_lidar)) {}

  static StateNode* find(std::deque<StateNode>& nodes, core::StateId id) {
    const auto found = std::find_if(nodes.begin(), nodes.end(),
                                    [id](const StateNode& node) { return node.id == id; });
    return found == nodes.end() ? nullptr : &*found;
  }

  static const StateNode* find(const std::deque<StateNode>& nodes, core::StateId id) {
    const auto found = std::find_if(nodes.begin(), nodes.end(),
                                    [id](const StateNode& node) { return node.id == id; });
    return found == nodes.end() ? nullptr : &*found;
  }

  static double* parameter(StateNode& node, BlockKind kind) {
    return kind == BlockKind::kPose ? node.pose.data() : node.motion.data();
  }

  static const double* parameter(const StateNode& node, BlockKind kind) {
    return kind == BlockKind::kPose ? node.pose.data() : node.motion.data();
  }

  static std::vector<double*> priorParameters(const PriorData& value,
                                              std::deque<StateNode>& nodes) {
    std::vector<double*> parameters;
    parameters.reserve(value.blocks.size());
    for (const MarginalPriorLinearizationBlock& block : value.blocks) {
      const core::StateId id = std::holds_alternative<PosePriorLinearization>(block)
                                   ? std::get<PosePriorLinearization>(block).state_id
                                   : std::get<MotionPriorLinearization>(block).state_id;
      StateNode* node = find(nodes, id);
      if (node == nullptr) {
        throw std::logic_error("marginal prior references a missing active state");
      }
      parameters.push_back(std::holds_alternative<PosePriorLinearization>(block)
                               ? node->pose.data()
                               : node->motion.data());
    }
    return parameters;
  }

  static std::vector<BlockKey> priorKeys(const PriorData& value) {
    std::vector<BlockKey> keys;
    keys.reserve(value.blocks.size());
    for (const MarginalPriorLinearizationBlock& block : value.blocks) {
      if (const auto* pose = std::get_if<PosePriorLinearization>(&block)) {
        keys.push_back({pose->state_id, BlockKind::kPose});
      } else {
        keys.push_back({std::get<MotionPriorLinearization>(block).state_id, BlockKind::kMotion});
      }
    }
    return keys;
  }

  static EvaluatedFactor evaluate(std::unique_ptr<ceres::CostFunction> cost,
                                  const std::vector<BlockKey>& keys,
                                  const std::deque<StateNode>& nodes) {
    if (keys.size() != cost->parameter_block_sizes().size()) {
      throw std::logic_error("factor block identity count disagrees with its Ceres cost");
    }
    const int rows = cost->num_residuals();
    EvaluatedFactor evaluated;
    evaluated.residual.resize(rows);
    evaluated.local_jacobians.reserve(keys.size());
    std::vector<const double*> parameters;
    std::vector<std::vector<double>> ambient_storage;
    std::vector<double*> ambient_jacobians;
    parameters.reserve(keys.size());
    ambient_storage.reserve(keys.size());
    ambient_jacobians.reserve(keys.size());
    for (std::size_t index = 0U; index < keys.size(); ++index) {
      const StateNode* node = find(nodes, keys[index].state_id);
      if (node == nullptr) {
        throw std::logic_error("factor references a missing active state");
      }
      parameters.push_back(parameter(*node, keys[index].kind));
      const int ambient_size = cost->parameter_block_sizes()[index];
      ambient_storage.emplace_back(static_cast<std::size_t>(rows * ambient_size));
      ambient_jacobians.push_back(ambient_storage.back().data());
    }
    if (!cost->Evaluate(parameters.data(), evaluated.residual.data(), ambient_jacobians.data())) {
      throw std::runtime_error("factor evaluation failed during square-root marginalization");
    }

    RightSe3Manifold pose_manifold;
    for (std::size_t index = 0U; index < keys.size(); ++index) {
      const int ambient_size = cost->parameter_block_sizes()[index];
      const Eigen::Map<const DynamicRowMajor> ambient(ambient_storage[index].data(), rows,
                                                      ambient_size);
      Eigen::MatrixXd local;
      if (keys[index].kind == BlockKind::kPose) {
        std::array<double, 42> plus_storage{};
        if (!pose_manifold.PlusJacobian(parameters[index], plus_storage.data())) {
          throw std::runtime_error("pose lift failed during square-root marginalization");
        }
        const Eigen::Map<const RowMajor7x6> plus(plus_storage.data());
        local = ambient * plus;
      } else {
        local = ambient;
      }
      evaluated.local_jacobians.emplace_back(keys[index], std::move(local));
    }
    return evaluated;
  }

  void addProblemFactors(ceres::Problem& problem, std::deque<StateNode>& nodes,
                         const std::vector<ImuFactor>& imu,
                         const std::vector<ActiveLidarGroup>& active,
                         const std::vector<FinalizedLidarGroup>& finalized,
                         const std::optional<PriorData>& prior_value) const {
    for (StateNode& node : nodes) {
      problem.AddParameterBlock(node.pose.data(), 7, new RightSe3Manifold());
      problem.AddParameterBlock(node.motion.data(), 9);
    }
    if (prior_value.has_value()) {
      auto* cost = new FixedLinearizationMarginalPriorCost(prior_value->blocks, prior_value->matrix,
                                                           prior_value->rhs);
      problem.AddResidualBlock(cost, nullptr, priorParameters(*prior_value, nodes));
    }
    for (const ImuFactor& factor : imu) {
      StateNode* from = find(nodes, factor.from);
      StateNode* to = find(nodes, factor.to);
      if (from == nullptr || to == nullptr) {
        throw std::logic_error("IMU factor references a missing active state");
      }
      problem.AddResidualBlock(new CombinedImuCost(factor.preintegration), nullptr,
                               from->pose.data(), from->motion.data(), to->pose.data(),
                               to->motion.data());
    }
    for (const ActiveLidarGroup& group : active) {
      StateNode* source = find(nodes, group.source);
      StateNode* owner = find(nodes, group.target);
      if (source == nullptr || owner == nullptr) {
        throw std::logic_error("active LiDAR group references a missing active state");
      }
      problem.AddResidualBlock(new ActiveOwnerScanToMapCost(group.rows, T_imu_lidar), nullptr,
                               source->pose.data(), owner->pose.data());
    }
    for (const FinalizedLidarGroup& group : finalized) {
      StateNode* source = find(nodes, group.source);
      if (source == nullptr) {
        throw std::logic_error("finalized LiDAR group references a missing active state");
      }
      problem.AddResidualBlock(new FinalizedScanToMapCost(group.rows, T_imu_lidar), nullptr,
                               source->pose.data());
    }
  }

  void appendLidarBatch(const lidar::ScanToMapResult& association, core::StateId source,
                        std::uint64_t batch_id, std::vector<ActiveLidarGroup>& active,
                        std::vector<FinalizedLidarGroup>& finalized) const {
    std::map<core::StateId, std::vector<ActiveLidarRow>> by_owner;
    std::vector<FinalizedLidarRow> finalized_rows;
    for (const lidar::ScanToMapRow& row : association.rows) {
      if (row.active_target_state.has_value()) {
        by_owner[*row.active_target_state].push_back(
            {.source_lidar = row.source_lidar,
             .target_lidar = row.target,
             .sqrt_weight_over_sigma = row.sqrt_weight_over_sigma});
      } else {
        finalized_rows.push_back({.source_lidar = row.source_lidar,
                                  .target_odom = row.target,
                                  .sqrt_weight_over_sigma = row.sqrt_weight_over_sigma});
      }
    }
    for (auto& [owner, rows] : by_owner) {
      active.push_back(
          {.batch_id = batch_id, .source = source, .target = owner, .rows = std::move(rows)});
    }
    if (!finalized_rows.empty()) {
      finalized.push_back(
          {.batch_id = batch_id, .source = source, .rows = std::move(finalized_rows)});
    }
  }

  static void eraseLidarBatch(std::uint64_t batch_id, std::vector<ActiveLidarGroup>& active,
                              std::vector<FinalizedLidarGroup>& finalized) {
    std::erase_if(active,
                  [batch_id](const ActiveLidarGroup& group) { return group.batch_id == batch_id; });
    std::erase_if(finalized, [batch_id](const FinalizedLidarGroup& group) {
      return group.batch_id == batch_id;
    });
  }

  [[nodiscard]] std::optional<SolvePass> solveWindow(
      std::deque<StateNode>& nodes, const std::vector<ImuFactor>& imu,
      const std::vector<ActiveLidarGroup>& active,
      const std::vector<FinalizedLidarGroup>& finalized,
      const std::optional<PriorData>& prior_value, std::string& reason) const {
    SolvePass pass;
    try {
      // The Problem must die before any registered deque storage can be moved
      // or popped during the staged marginalization/commit below.
      ceres::Problem problem;
      const Clock::time_point problem_begin = Clock::now();
      addProblemFactors(problem, nodes, imu, active, finalized, prior_value);
      pass.problem_build_ns = elapsedNs(problem_begin);

      ceres::Solver::Options solver_options;
      solver_options.max_num_iterations = static_cast<int>(options.maximum_solver_iterations);
      solver_options.num_threads = static_cast<int>(options.solver_threads);
      solver_options.max_solver_time_in_seconds = options.maximum_solver_time_s;
      solver_options.function_tolerance = options.function_tolerance;
      solver_options.gradient_tolerance = options.gradient_tolerance;
      solver_options.parameter_tolerance = options.parameter_tolerance;
      solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      solver_options.minimizer_progress_to_stdout = false;
      solver_options.logging_type = ceres::SILENT;
      const Clock::time_point solve_begin = Clock::now();
      ceres::Solve(solver_options, &problem, &pass.summary);
      pass.solve_ns = elapsedNs(solve_begin);
    } catch (const std::exception& error) {
      reason = std::string("Ceres problem/solve failed: ") + error.what();
      return std::nullopt;
    }
    return pass;
  }

  [[nodiscard]] bool statesAreValid(const std::deque<StateNode>& baseline,
                                    const std::deque<StateNode>& optimized) const {
    if (baseline.size() != optimized.size()) {
      return false;
    }
    for (const StateNode& node : optimized) {
      const StateNode* original = find(baseline, node.id);
      if (original == nullptr || original->time != node.time ||
          !Eigen::Map<const Eigen::Matrix<double, 7, 1>>(node.pose.data())
               .array()
               .isFinite()
               .all() ||
          !Eigen::Map<const Eigen::Matrix<double, 9, 1>>(node.motion.data())
               .array()
               .isFinite()
               .all()) {
        return false;
      }
      const Eigen::Map<const Eigen::Quaterniond> quaternion(node.pose.data() + 3);
      if (std::abs(quaternion.squaredNorm() - 1.0) > 1.0e-8) {
        return false;
      }
      const Sophus::SE3d correction =
          sophus(original->state().odomFromImu()).inverse() * sophus(node.state().odomFromImu());
      if (!correction.matrix().allFinite() ||
          correction.translation().norm() > options.maximum_translation_correction_m ||
          correction.so3().log().norm() > options.maximum_rotation_correction_rad) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] std::optional<std::size_t> invalidLidarRowCount(
      const std::deque<StateNode>& nodes, const std::vector<ActiveLidarGroup>& active,
      const std::vector<FinalizedLidarGroup>& finalized) const {
    const Sophus::SE3d imu_from_lidar = sophus(T_imu_lidar);
    const double hard_gate = 2.0 * target.options().maximum_correspondence_distance_m;
    const double hard_gate_squared = hard_gate * hard_gate;
    std::size_t invalid = 0U;
    for (const ActiveLidarGroup& group : active) {
      const StateNode* source = find(nodes, group.source);
      const StateNode* owner = find(nodes, group.target);
      if (source == nullptr || owner == nullptr) {
        return std::nullopt;
      }
      const Sophus::SE3d owner_lidar_from_source_lidar =
          (sophus(owner->state().odomFromImu()) * imu_from_lidar).inverse() *
          sophus(source->state().odomFromImu()) * imu_from_lidar;
      for (const ActiveLidarRow& row : group.rows) {
        const Eigen::Vector3d error =
            owner_lidar_from_source_lidar * row.source_lidar - row.target_lidar;
        if (!error.allFinite()) {
          return std::nullopt;
        }
        invalid += static_cast<std::size_t>(error.squaredNorm() > hard_gate_squared);
      }
    }
    for (const FinalizedLidarGroup& group : finalized) {
      const StateNode* source = find(nodes, group.source);
      if (source == nullptr) {
        return std::nullopt;
      }
      const Sophus::SE3d odom_from_lidar =
          sophus(source->state().odomFromImu()) * imu_from_lidar;
      for (const FinalizedLidarRow& row : group.rows) {
        const Eigen::Vector3d error = odom_from_lidar * row.source_lidar - row.target_odom;
        if (!error.allFinite()) {
          return std::nullopt;
        }
        invalid += static_cast<std::size_t>(error.squaredNorm() > hard_gate_squared);
      }
    }
    return invalid;
  }

  bool priorChartIsValid(const std::optional<PriorData>& prior_value,
                         std::deque<StateNode>& nodes) const {
    if (!prior_value.has_value()) {
      return true;
    }
    FixedLinearizationMarginalPriorCost cost(prior_value->blocks, prior_value->matrix,
                                             prior_value->rhs);
    std::vector<double*> mutable_parameters = priorParameters(*prior_value, nodes);
    std::vector<const double*> parameters(mutable_parameters.begin(), mutable_parameters.end());
    const auto displacement = cost.chartDisplacement(parameters.data());
    return displacement.has_value() &&
           displacement->maximum_pose_translation_m <= options.maximum_prior_translation_m &&
           displacement->maximum_pose_rotation_rad <= options.maximum_prior_rotation_rad &&
           displacement->maximum_motion_tangent_norm <= options.maximum_prior_motion_norm;
  }

  bool shouldMarginalize(const std::deque<StateNode>& nodes) const {
    if (nodes.size() > options.maximum_states) {
      return true;
    }
    if (nodes.size() < 2U) {
      return false;
    }
    const auto duration = core::TimeNs::checkedDifference(nodes.back().time, nodes.front().time);
    return duration.has_value() && *duration > options.maximum_lag_ns;
  }

  bool marginalizeOldest(std::deque<StateNode>& nodes, std::vector<ImuFactor>& imu,
                         std::vector<ActiveLidarGroup>& active,
                         std::vector<FinalizedLidarGroup>& finalized,
                         std::optional<PriorData>& prior_value,
                         lidar::ScanToMapTarget& target_value, FixedLagUpdate& update) const {
    if (nodes.size() < 2U) {
      update.reason = "fixed-lag marginalization cannot remove the only active state";
      return false;
    }
    const core::StateId dropped = nodes.front().id;
    const Clock::time_point evaluate_begin = Clock::now();
    std::vector<EvaluatedFactor> evaluated;
    if (prior_value.has_value()) {
      evaluated.push_back(evaluate(std::make_unique<FixedLinearizationMarginalPriorCost>(
                                       prior_value->blocks, prior_value->matrix, prior_value->rhs),
                                   priorKeys(*prior_value), nodes));
    }
    for (const ImuFactor& factor : imu) {
      if (factor.from == dropped || factor.to == dropped) {
        evaluated.push_back(evaluate(std::make_unique<CombinedImuCost>(factor.preintegration),
                                     {{factor.from, BlockKind::kPose},
                                      {factor.from, BlockKind::kMotion},
                                      {factor.to, BlockKind::kPose},
                                      {factor.to, BlockKind::kMotion}},
                                     nodes));
      }
    }
    for (const ActiveLidarGroup& group : active) {
      if (group.source == dropped || group.target == dropped) {
        evaluated.push_back(
            evaluate(std::make_unique<ActiveOwnerScanToMapCost>(group.rows, T_imu_lidar),
                     {{group.source, BlockKind::kPose}, {group.target, BlockKind::kPose}}, nodes));
      }
    }
    for (const FinalizedLidarGroup& group : finalized) {
      if (group.source == dropped) {
        evaluated.push_back(
            evaluate(std::make_unique<FinalizedScanToMapCost>(group.rows, T_imu_lidar),
                     {{group.source, BlockKind::kPose}}, nodes));
      }
    }
    update.timing.marginalization_evaluate_ns += elapsedNs(evaluate_begin);
    if (evaluated.empty()) {
      update.reason = "oldest state has no factors to preserve during marginalization";
      return false;
    }

    Eigen::Index row_count = 0;
    for (const EvaluatedFactor& factor : evaluated) {
      row_count += factor.residual.size();
    }
    const Eigen::Index column_count = static_cast<Eigen::Index>(15U * nodes.size());
    Eigen::VectorXd residual(row_count);
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(row_count, column_count);
    auto columnOffset = [&](const BlockKey& key) -> Eigen::Index {
      const auto found = std::find_if(nodes.begin(), nodes.end(), [&](const StateNode& node) {
        return node.id == key.state_id;
      });
      if (found == nodes.end()) {
        throw std::logic_error("marginalized factor references an unknown state");
      }
      const Eigen::Index state_index = std::distance(nodes.begin(), found);
      return 15 * state_index + (key.kind == BlockKind::kPose ? 0 : 6);
    };
    Eigen::Index row_offset = 0;
    for (const EvaluatedFactor& factor : evaluated) {
      residual.segment(row_offset, factor.residual.size()) = factor.residual;
      for (const auto& [key, local] : factor.local_jacobians) {
        jacobian.block(row_offset, columnOffset(key), local.rows(), local.cols()) += local;
      }
      row_offset += factor.residual.size();
    }

    const Clock::time_point eliminate_begin = Clock::now();
    SquareRootMarginalizationResult reduced;
    try {
      reduced = squareRootMarginalizeFirstState(residual, jacobian, options.marginalization);
    } catch (const std::exception& error) {
      update.reason = std::string("square-root marginalization failed: ") + error.what();
      return false;
    }
    update.timing.marginalization_eliminate_ns += elapsedNs(eliminate_begin);
    if (reduced.retained_rank == 0U || reduced.square_root_matrix.rows() == 0) {
      update.reason = "square-root marginalization produced an empty retained prior";
      return false;
    }

    const Clock::time_point prior_begin = Clock::now();
    PriorData next_prior;
    next_prior.matrix = std::move(reduced.square_root_matrix);
    next_prior.rhs = std::move(reduced.right_hand_side);
    next_prior.blocks.reserve(2U * (nodes.size() - 1U));
    for (auto iterator = std::next(nodes.begin()); iterator != nodes.end(); ++iterator) {
      next_prior.blocks.emplace_back(
          PosePriorLinearization{.state_id = iterator->id, .parameters = iterator->pose});
      next_prior.blocks.emplace_back(
          MotionPriorLinearization{.state_id = iterator->id, .parameters = iterator->motion});
    }
    update.prior_rank = reduced.retained_rank;
    update.timing.marginalization_prior_build_ns += elapsedNs(prior_begin);

    const StateNode finalized_state = nodes.front();
    const Clock::time_point target_begin = Clock::now();
    if (target_value.hasActiveOwner(finalized_state.id)) {
      static_cast<void>(
          target_value.finalize(finalized_state.id, sophus(finalized_state.state().odomFromImu()),
                                sophus(nodes.back().state().odomFromImu()).translation()));
    }
    update.timing.target_finalize_ns += elapsedNs(target_begin);
    update.newly_finalized.push_back(finalized_state.state());

    nodes.pop_front();
    std::erase_if(imu, [dropped](const ImuFactor& factor) {
      return factor.from == dropped || factor.to == dropped;
    });
    std::erase_if(active, [dropped](const ActiveLidarGroup& group) {
      return group.source == dropped || group.target == dropped;
    });
    std::erase_if(finalized,
                  [dropped](const FinalizedLidarGroup& group) { return group.source == dropped; });
    prior_value = std::move(next_prior);
    ++update.marginalizations;
    return true;
  }
};

FixedLagEstimator::FixedLagEstimator(FixedLagEstimatorOptions options,
                                     lidar::ScanToMapOptions scan_options,
                                     core::Pose3d T_imu_lidar) {
  validate(options);
  impl_ =
      std::make_unique<Impl>(std::move(options), std::move(scan_options), std::move(T_imu_lidar));
}

FixedLagEstimator::~FixedLagEstimator() = default;

void FixedLagEstimator::initialize(const core::NavigationState& seed,
                                   std::optional<lidar::ScanFrame> anchor) {
  if (!impl_->states.empty()) {
    throw std::logic_error("fixed-lag estimator can only be initialized once");
  }
  if (anchor.has_value() && (anchor->state_id != seed.id() || anchor->time != seed.time() ||
                             anchor->target_points_lidar.empty())) {
    throw std::invalid_argument("fixed-lag anchor must match the initialization seed");
  }
  Impl::StateNode node{.id = seed.id(),
                       .time = seed.time(),
                       .pose = poseArray(seed.odomFromImu()),
                       .motion = motionArray(seed)};
  std::deque<Impl::StateNode> staged_states;
  staged_states.push_back(node);

  Impl::PriorData prior;
  prior.blocks.emplace_back(PosePriorLinearization{.state_id = node.id, .parameters = node.pose});
  prior.blocks.emplace_back(
      MotionPriorLinearization{.state_id = node.id, .parameters = node.motion});
  prior.matrix = Eigen::MatrixXd::Zero(15, 15);
  prior.matrix.diagonal().segment<3>(0).setConstant(1.0 /
                                                    impl_->options.initial_translation_sigma_m);
  prior.matrix.diagonal().segment<3>(3).setConstant(1.0 /
                                                    impl_->options.initial_rotation_sigma_rad);
  prior.matrix.diagonal().segment<3>(6).setConstant(1.0 /
                                                    impl_->options.initial_velocity_sigma_m_s);
  prior.matrix.diagonal().segment<3>(9).setConstant(1.0 /
                                                    impl_->options.initial_gyro_bias_sigma_rad_s);
  prior.matrix.diagonal().segment<3>(12).setConstant(1.0 /
                                                     impl_->options.initial_accel_bias_sigma_m_s2);
  prior.rhs = Eigen::VectorXd::Zero(15);
  lidar::ScanToMapTarget staged_target = impl_->target;
  if (anchor.has_value()) {
    staged_target.admit(std::move(*anchor), sophus(seed.odomFromImu()));
  }
  impl_->states = std::move(staged_states);
  impl_->prior = std::move(prior);
  impl_->target = std::move(staged_target);
}

bool FixedLagEstimator::initialized() const noexcept {
  return !impl_->states.empty();
}

core::NavigationState FixedLagEstimator::latestState() const {
  if (impl_->states.empty()) {
    throw std::logic_error("fixed-lag estimator is not initialized");
  }
  return impl_->states.back().state();
}

std::vector<core::NavigationState> FixedLagEstimator::activeStates() const {
  std::vector<core::NavigationState> result;
  result.reserve(impl_->states.size());
  for (const Impl::StateNode& node : impl_->states) {
    result.push_back(node.state());
  }
  return result;
}

lidar::PointCloud FixedLagEstimator::registrationMapPointCloud() const {
  if (impl_->states.empty()) {
    throw std::logic_error("fixed-lag estimator is not initialized");
  }
  return impl_->target.registrationMapPointCloud();
}

FixedLagUpdate FixedLagEstimator::addSweep(lidar::ScanFrame frame,
                                           const CombinedPreintegration& preintegration,
                                           const core::NavigationState& predicted) {
  const Clock::time_point total_begin = Clock::now();
  FixedLagUpdate update;
  update.predicted = predicted;
  if (impl_->states.empty() || frame.state_id != predicted.id() || frame.time != predicted.time() ||
      frame.target_points_lidar.empty() || frame.source_points_lidar.empty() ||
      predicted.time() <= impl_->states.back().time || preintegration.durationSeconds() <= 0.0) {
    update.reason = "candidate sweep/state/preintegration is inconsistent";
    update.timing.total_ns = elapsedNs(total_begin);
    return update;
  }

  const Clock::time_point first_association_begin = Clock::now();
  update.first_association =
      impl_->target.associateScan(frame.source_points_lidar, sophus(predicted.odomFromImu()));
  update.timing.association_first_ns = elapsedNs(first_association_begin);
  update.timing.registration_ns = update.timing.association_first_ns;
  update.association_passes = 1U;
  const bool first_target =
      update.first_association.status == lidar::ScanToMapStatus::kEmptyTarget;
  if (!first_target && !update.first_association.accepted()) {
    update.status = FixedLagStatus::kRegistrationRejected;
    update.reason = std::string("first scan-to-map association rejected: ") +
                    toString(update.first_association.status);
    update.timing.total_ns = elapsedNs(total_begin);
    return update;
  }
  update.registration = update.first_association;

  const Clock::time_point factor_begin = Clock::now();
  std::deque<Impl::StateNode> candidate_states = impl_->states;
  Impl::StateNode candidate{.id = predicted.id(),
                            .time = predicted.time(),
                            .pose = poseArray(predicted.odomFromImu()),
                            .motion = motionArray(predicted)};
  candidate_states.push_back(candidate);
  const std::deque<Impl::StateNode> baseline_states = candidate_states;
  std::vector<Impl::ImuFactor> candidate_imu = impl_->imu_factors;
  candidate_imu.push_back(
      {.from = impl_->states.back().id, .to = candidate.id, .preintegration = preintegration});
  std::vector<Impl::ActiveLidarGroup> candidate_active = impl_->active_lidar_groups;
  std::vector<Impl::FinalizedLidarGroup> candidate_finalized = impl_->finalized_lidar_groups;
  std::optional<Impl::PriorData> candidate_prior = impl_->prior;
  const std::uint64_t batch_id = impl_->next_batch_id;
  if (!first_target) {
    impl_->appendLidarBatch(update.first_association, candidate.id, batch_id, candidate_active,
                            candidate_finalized);
  }
  update.timing.factor_build_ns = elapsedNs(factor_begin);

  auto first_pass = impl_->solveWindow(candidate_states, candidate_imu, candidate_active,
                                       candidate_finalized, candidate_prior, update.reason);
  if (!first_pass.has_value()) {
    update.status = FixedLagStatus::kOptimizationRejected;
    update.timing.total_ns = elapsedNs(total_begin);
    return update;
  }
  update.timing.problem_build_first_ns = first_pass->problem_build_ns;
  update.timing.problem_build_ns = first_pass->problem_build_ns;
  update.timing.ceres_solve_first_ns = first_pass->solve_ns;
  update.timing.ceres_solve_ns = first_pass->solve_ns;
  update.initial_cost = first_pass->summary.initial_cost;
  update.final_cost = first_pass->summary.final_cost;
  update.solver_iterations = first_pass->summary.iterations.size();
  const bool first_solution_valid =
      first_pass->summary.IsSolutionUsable() && std::isfinite(first_pass->summary.final_cost) &&
      first_pass->summary.final_cost <= first_pass->summary.initial_cost &&
      impl_->statesAreValid(baseline_states, candidate_states) &&
      impl_->priorChartIsValid(candidate_prior, candidate_states);
  if (!first_solution_valid) {
    update.status = FixedLagStatus::kOptimizationRejected;
    update.reason = "first Ceres pass failed usability, descent, state, or prior-chart gates";
    update.timing.total_ns = elapsedNs(total_begin);
    return update;
  }

  lidar::ScanToMapTarget candidate_target = impl_->target;
  try {
    for (const Impl::StateNode& node : candidate_states) {
      if (Impl::find(impl_->states, node.id) != nullptr &&
          candidate_target.hasActiveOwner(node.id)) {
        candidate_target.updateOwnerPose(node.id, sophus(node.state().odomFromImu()));
      }
    }

    if (!first_target) {
      const Clock::time_point second_association_begin = Clock::now();
      update.registration = candidate_target.associateScan(
          frame.source_points_lidar, sophus(candidate_states.back().state().odomFromImu()));
      update.timing.association_second_ns = elapsedNs(second_association_begin);
      update.timing.registration_ns += update.timing.association_second_ns;
      update.association_passes = 2U;
      if (!update.registration.accepted()) {
        update.status = FixedLagStatus::kRegistrationRejected;
        update.reason = std::string("post-Ceres scan-to-map reassociation rejected: ") +
                        toString(update.registration.status);
        update.timing.total_ns = elapsedNs(total_begin);
        return update;
      }

      const Clock::time_point second_factor_begin = Clock::now();
      impl_->eraseLidarBatch(batch_id, candidate_active, candidate_finalized);
      impl_->appendLidarBatch(update.registration, candidate.id, batch_id, candidate_active,
                              candidate_finalized);
      update.reassociated_rows = update.registration.rows.size();
      update.timing.factor_build_ns += elapsedNs(second_factor_begin);

      auto second_pass = impl_->solveWindow(candidate_states, candidate_imu, candidate_active,
                                            candidate_finalized, candidate_prior, update.reason);
      if (!second_pass.has_value()) {
        update.status = FixedLagStatus::kOptimizationRejected;
        update.timing.total_ns = elapsedNs(total_begin);
        return update;
      }
      update.timing.problem_build_second_ns = second_pass->problem_build_ns;
      update.timing.problem_build_ns += second_pass->problem_build_ns;
      update.timing.ceres_solve_second_ns = second_pass->solve_ns;
      update.timing.ceres_solve_ns += second_pass->solve_ns;
      update.final_cost = second_pass->summary.final_cost;
      update.solver_iterations += second_pass->summary.iterations.size();
      const bool second_solution_valid =
          second_pass->summary.IsSolutionUsable() &&
          std::isfinite(second_pass->summary.final_cost) &&
          second_pass->summary.final_cost <= second_pass->summary.initial_cost;
      if (!second_solution_valid) {
        update.status = FixedLagStatus::kOptimizationRejected;
        update.reason = "second Ceres pass failed usability or descent gates";
        update.timing.total_ns = elapsedNs(total_begin);
        return update;
      }

      // The target view used by the next sweep must be at the final pass poses,
      // not the intermediate poses used for reassociation.
      for (const Impl::StateNode& node : candidate_states) {
        if (Impl::find(impl_->states, node.id) != nullptr &&
            candidate_target.hasActiveOwner(node.id)) {
          candidate_target.updateOwnerPose(node.id, sophus(node.state().odomFromImu()));
        }
      }
    }

    const Clock::time_point validation_begin = Clock::now();
    const Impl::StateNode& optimized_node = candidate_states.back();
    const Sophus::SE3d correction =
        sophus(predicted.odomFromImu()).inverse() * sophus(optimized_node.state().odomFromImu());
    update.correction_translation_m = correction.translation().norm();
    update.correction_rotation_rad = correction.so3().log().norm();
    const auto invalid_lidar_rows =
        impl_->invalidLidarRowCount(candidate_states, candidate_active, candidate_finalized);
    update.rejected_stale_rows = invalid_lidar_rows.value_or(0U);
    const bool valid = impl_->statesAreValid(baseline_states, candidate_states) &&
                       impl_->priorChartIsValid(candidate_prior, candidate_states) &&
                       invalid_lidar_rows.has_value() && *invalid_lidar_rows == 0U &&
                       update.correction_translation_m <=
                           impl_->options.maximum_translation_correction_m &&
                       update.correction_rotation_rad <=
                           impl_->options.maximum_rotation_correction_rad;
    update.timing.validation_ns = elapsedNs(validation_begin);
    if (!valid) {
      update.status = FixedLagStatus::kOptimizationRejected;
      update.reason =
          "optimized window failed state, correction, stale-row, or prior-chart gates";
      update.timing.total_ns = elapsedNs(total_begin);
      return update;
    }

    const Clock::time_point admit_begin = Clock::now();
    candidate_target.admit(std::move(frame),
                           sophus(candidate_states.back().state().odomFromImu()));
    update.timing.target_admit_ns = elapsedNs(admit_begin);

    while (impl_->shouldMarginalize(candidate_states)) {
      if (!impl_->marginalizeOldest(candidate_states, candidate_imu, candidate_active,
                                    candidate_finalized, candidate_prior, candidate_target,
                                    update)) {
        update.status = FixedLagStatus::kMarginalizationRejected;
        update.timing.total_ns = elapsedNs(total_begin);
        return update;
      }
    }
  } catch (const std::exception& error) {
    update.status = FixedLagStatus::kMarginalizationRejected;
    update.reason = std::string("staged target/marginalization update failed: ") + error.what();
    update.timing.total_ns = elapsedNs(total_begin);
    return update;
  }

  const Clock::time_point commit_begin = Clock::now();
  impl_->states = std::move(candidate_states);
  impl_->imu_factors = std::move(candidate_imu);
  impl_->active_lidar_groups = std::move(candidate_active);
  impl_->finalized_lidar_groups = std::move(candidate_finalized);
  impl_->prior = std::move(candidate_prior);
  impl_->target = std::move(candidate_target);
  ++impl_->next_batch_id;
  update.optimized = Impl::find(impl_->states, predicted.id())->state();
  update.active_states = impl_->states.size();
  update.imu_factors = impl_->imu_factors.size();
  update.active_lidar_groups = impl_->active_lidar_groups.size();
  update.finalized_lidar_groups = impl_->finalized_lidar_groups.size();
  update.lidar_rows = 0U;
  for (const Impl::ActiveLidarGroup& group : impl_->active_lidar_groups) {
    update.active_lidar_rows += group.rows.size();
  }
  for (const Impl::FinalizedLidarGroup& group : impl_->finalized_lidar_groups) {
    update.finalized_lidar_rows += group.rows.size();
  }
  update.lidar_rows = update.active_lidar_rows + update.finalized_lidar_rows;
  update.finalized_map_points = impl_->target.finalizedPointCount();
  update.status = FixedLagStatus::kAccepted;
  update.reason =
      first_target ? "accepted first production target"
                   : "accepted two-pass fixed-lag LiDAR-IMU update";
  update.timing.commit_ns = elapsedNs(commit_begin);
  update.timing.total_ns = elapsedNs(total_begin);
  return update;
}

const char* toString(FixedLagStatus status) noexcept {
  switch (status) {
    case FixedLagStatus::kAccepted:
      return "accepted";
    case FixedLagStatus::kRegistrationRejected:
      return "registration_rejected";
    case FixedLagStatus::kOptimizationRejected:
      return "optimization_rejected";
    case FixedLagStatus::kMarginalizationRejected:
      return "marginalization_rejected";
    case FixedLagStatus::kInvalidInput:
      return "invalid_input";
  }
  return "unknown";
}

}  // namespace meridian::local_rt::estimator
