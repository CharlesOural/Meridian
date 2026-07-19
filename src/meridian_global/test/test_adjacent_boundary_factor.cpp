#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "../src/adjacent_boundary_factor_internal.hpp"
#include "sparse_seal_test_utils.hpp"

namespace meridian::global::adjacent_internal {
namespace {

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct TestEndpointFrame {
  core::StateId state;
  core::Pose3d C_submap_imu;
};

struct SealPair {
  core::SparseSubmapSeal predecessor;
  core::SparseSubmapSeal current;
};

[[nodiscard]] core::ContentHash presentHash(std::uint8_t value) {
  core::ContentHash hash{};
  hash.front() = value;
  hash.back() = static_cast<std::uint8_t>(value ^ 0xa5U);
  return hash;
}

[[nodiscard]] core::LocalVariableRef stateVariable(core::LocalVariableKind kind,
                                                   core::StateId state) {
  core::LocalVariableRef variable;
  variable.kind = kind;
  variable.state = state;
  return variable;
}

[[nodiscard]] core::Pose3d pose(const Eigen::Vector3d& translation,
                                const Eigen::Vector3d& rotation_vector) {
  return core::Pose3d(Sophus::SO3d::exp(rotation_vector), translation);
}

[[nodiscard]] core::ObservationLineage lineage() {
  return test_support::lineage(91U);
}

[[nodiscard]] core::CondensedBoundaryTransition transition() {
  core::CondensedBoundaryTransition output;
  output.header.schema_version = 1U;
  output.header.trace = core::TraceId(1U);
  output.header.producer = core::ProducerId(2U);
  output.header.session = core::SessionId(3U);
  output.header.config = core::ConfigRevision(4U);
  output.header.direct_calibration = core::CalibrationEpoch(5U);
  output.odom_epoch = core::OdomEpoch(6U);
  output.from.state = core::StateId(10U);
  output.from.exact_time = core::FusionTime{100};
  output.from.final_revision = core::LocalGraphRevision(7U);
  output.from.T_odom_imu =
      pose(Eigen::Vector3d{1.2, -0.8, 0.4}, Eigen::Vector3d{0.17, -0.09, 0.22});
  output.from.velocity_odom = Eigen::Vector3d{2.1, -0.7, 0.35};
  output.from.gyro_bias = Eigen::Vector3d{0.01, -0.02, 0.03};
  output.from.accel_bias = Eigen::Vector3d{-0.11, 0.07, 0.04};
  output.to.state = core::StateId(20U);
  output.to.exact_time = core::FusionTime{200};
  output.to.final_revision = core::LocalGraphRevision(8U);
  output.to.T_odom_imu = pose(Eigen::Vector3d{3.4, 0.6, 0.9}, Eigen::Vector3d{-0.08, 0.13, 0.31});
  output.to.velocity_odom = Eigen::Vector3d{1.8, 0.4, -0.15};
  output.to.gyro_bias = Eigen::Vector3d{0.015, -0.018, 0.027};
  output.to.accel_bias = Eigen::Vector3d{-0.09, 0.08, 0.055};

  auto& factor = output.boundary_factor;
  factor.rows = 13U;
  factor.columns = 30U;
  factor.layout = {
      {stateVariable(core::LocalVariableKind::Pose, output.from.state), 0U, 6U},
      {stateVariable(core::LocalVariableKind::NavigationVelocity, output.from.state), 6U, 3U},
      {stateVariable(core::LocalVariableKind::GyroBias, output.from.state), 9U, 3U},
      {stateVariable(core::LocalVariableKind::AccelBias, output.from.state), 12U, 3U},
      {stateVariable(core::LocalVariableKind::Pose, output.to.state), 15U, 6U},
      {stateVariable(core::LocalVariableKind::NavigationVelocity, output.to.state), 21U, 3U},
      {stateVariable(core::LocalVariableKind::GyroBias, output.to.state), 24U, 3U},
      {stateVariable(core::LocalVariableKind::AccelBias, output.to.state), 27U, 3U},
  };
  factor.row_major_A.assign(static_cast<std::size_t>(factor.rows) * factor.columns, 0.0);
  for (std::size_t row = 0U; row < 12U; ++row) {
    for (std::size_t column = 0U; column < factor.columns; ++column) {
      factor.row_major_A[row * factor.columns + column] =
          0.01 * std::sin(static_cast<double>((row + 1U) * (column + 2U)));
    }
    factor.row_major_A[row * factor.columns + row] += 2.0;
  }
  factor.rhs.resize(factor.rows);
  for (std::size_t row = 0U; row < factor.rhs.size(); ++row) {
    factor.rhs[row] = 0.04 * static_cast<double>(row) - 0.2;
  }
  factor.constant_squared_error = 0.37;
  factor.numerical_rank = 12U;
  factor.absolute_rank_tolerance = 1.0e-12;
  factor.relative_rank_tolerance = 1.0e-9;
  factor.cost_statistics.source_residual_dof = 18U;
  factor.cost_statistics.eliminated_numerical_rank = 6U;
  factor.cost_statistics.effective_dof = 12U;
  factor.cost_statistics.calibration_revision = core::ResidualCalibrationRevision(1U);
  factor.cost_statistics.calibrated_total_cost_cutoff = 24.0;
  factor.checksum = test_support::required(core::recomputeFrozenSquareRootFactorChecksum(factor));
  output.source_factors = {{output.odom_epoch, core::FactorId(1U)},
                           {output.odom_epoch, core::FactorId(2U)}};
  output.lineage = lineage();
  output.final_revision = core::LocalGraphRevision(8U);
  output.input_partition_checksum = presentHash(9U);
  output.checksum =
      test_support::required(core::recomputeCondensedBoundaryTransitionChecksum(output));
  return output;
}

[[nodiscard]] TestEndpointFrame fromFrame(const core::CondensedBoundaryTransition& input) {
  return TestEndpointFrame{input.from.state, pose(Eigen::Vector3d{0.35, -0.18, 0.12},
                                                  Eigen::Vector3d{-0.12, 0.07, 0.09})};
}

[[nodiscard]] TestEndpointFrame toFrame(const core::CondensedBoundaryTransition& input) {
  return TestEndpointFrame{
      input.to.state, pose(Eigen::Vector3d{-0.21, 0.27, 0.08}, Eigen::Vector3d{0.06, -0.11, 0.14})};
}

[[nodiscard]] SealPair seals(const core::CondensedBoundaryTransition& input,
                             const TestEndpointFrame& from_frame,
                             const TestEndpointFrame& to_frame) {
  core::SparseSubmapSeal predecessor = test_support::firstSeal(
      1U, input.from.exact_time.nanoseconds,
      input.from.T_odom_imu * from_frame.C_submap_imu.inverse(), input.from.T_odom_imu);
  predecessor.header.session = input.header.session;
  predecessor.header.direct_calibration = input.header.direct_calibration;
  predecessor.ref.session = input.header.session;
  predecessor.ref.odom_epoch = input.odom_epoch;
  predecessor.ref.calibration = *input.header.direct_calibration;
  predecessor.frame.boundary_state = input.from.state;
  predecessor.frame.boundary_time = input.from.exact_time;
  predecessor.boundary_navigation = input.from;
  test_support::finalizeLocalContent(&predecessor);
  test_support::finalizeSealEnvelope(&predecessor);

  core::SparseSubmapSeal current = test_support::firstSeal(
      2U, input.to.exact_time.nanoseconds, input.to.T_odom_imu * to_frame.C_submap_imu.inverse(),
      input.to.T_odom_imu);
  current.header.session = input.header.session;
  current.header.direct_calibration = input.header.direct_calibration;
  current.ref.session = input.header.session;
  current.ref.odom_epoch = input.odom_epoch;
  current.ref.calibration = *input.header.direct_calibration;
  current.frame.boundary_state = input.to.state;
  current.frame.boundary_time = input.to.exact_time;
  current.boundary_navigation = input.to;
  test_support::finalizeLocalContent(&current);

  core::CondensedBoundaryTransition linked = input;
  linked.header.session = predecessor.ref.session;
  linked.header.direct_calibration = predecessor.ref.calibration;
  linked.odom_epoch = predecessor.ref.odom_epoch;
  linked.from = predecessor.boundary_navigation;
  linked.to = current.boundary_navigation;
  linked.checksum =
      test_support::required(core::recomputeCondensedBoundaryTransitionChecksum(linked));
  core::SealedBoundaryTransition sealed{predecessor.ref, current.ref, std::move(linked), {}};
  sealed.checksum = test_support::required(core::recomputeSealedBoundaryTransitionChecksum(sealed));
  current.previous = predecessor.ref;
  current.from_previous = std::move(sealed);
  test_support::finalizeSealEnvelope(&current);
  return SealPair{std::move(predecessor), std::move(current)};
}

[[nodiscard]] OdomEpochChartPlacement placement(const core::CondensedBoundaryTransition& input) {
  return OdomEpochChartPlacement{
      input.odom_epoch, pose(Eigen::Vector3d{8.0, -3.5, 1.1}, Eigen::Vector3d{0.0, 0.0, 0.43})};
}

[[nodiscard]] AdjacentBoundaryKeys keys() {
  return AdjacentBoundaryKeys{gtsam::Symbol('a', 0U), gtsam::Symbol('v', 0U),
                              gtsam::Symbol('g', 0U), gtsam::Symbol('b', 0U),
                              gtsam::Symbol('a', 1U), gtsam::Symbol('v', 1U),
                              gtsam::Symbol('g', 1U), gtsam::Symbol('b', 1U)};
}

[[nodiscard]] gtsam::Pose3 toGtsam(const core::Pose3d& input) {
  return gtsam::Pose3(gtsam::Rot3(input.so3().matrix()), input.translation());
}

[[nodiscard]] core::Pose3d fromGtsam(const gtsam::Pose3& input) {
  return core::Pose3d(Sophus::SO3d(input.rotation().matrix()), input.translation());
}

[[nodiscard]] std::array<gtsam::Key, 8> orderedKeys(const AdjacentBoundaryKeys& input) {
  return {input.anchor_from, input.velocity_from, input.gyro_bias_from, input.accel_bias_from,
          input.anchor_to,   input.velocity_to,   input.gyro_bias_to,   input.accel_bias_to};
}

[[nodiscard]] gtsam::Values valuesAwayFromCenters(const core::CondensedBoundaryTransition& input,
                                                  const TestEndpointFrame& from_frame,
                                                  const TestEndpointFrame& to_frame,
                                                  const OdomEpochChartPlacement& chart,
                                                  const AdjacentBoundaryKeys& factor_keys) {
  const gtsam::Pose3 anchor_from_center =
      toGtsam(chart.H_map_odom * input.from.T_odom_imu * from_frame.C_submap_imu.inverse());
  const gtsam::Pose3 anchor_to_center =
      toGtsam(chart.H_map_odom * input.to.T_odom_imu * to_frame.C_submap_imu.inverse());
  gtsam::Vector6 from_delta;
  from_delta << 0.08, -0.04, 0.03, 0.12, -0.07, 0.05;
  gtsam::Vector6 to_delta;
  to_delta << -0.05, 0.09, -0.02, -0.08, 0.11, 0.04;
  const Eigen::Matrix3d R_map_odom = chart.H_map_odom.so3().matrix();

  gtsam::Values values;
  values.insert(factor_keys.anchor_from,
                anchor_from_center.compose(gtsam::Pose3::Expmap(from_delta)));
  values.insert(
      factor_keys.velocity_from,
      (R_map_odom * input.from.velocity_odom + Eigen::Vector3d{0.12, -0.06, 0.03}).eval());
  values.insert(factor_keys.gyro_bias_from,
                (input.from.gyro_bias + Eigen::Vector3d{0.002, -0.001, 0.003}).eval());
  values.insert(factor_keys.accel_bias_from,
                (input.from.accel_bias + Eigen::Vector3d{-0.01, 0.02, -0.015}).eval());
  values.insert(factor_keys.anchor_to, anchor_to_center.compose(gtsam::Pose3::Expmap(to_delta)));
  values.insert(
      factor_keys.velocity_to,
      (R_map_odom * input.to.velocity_odom + Eigen::Vector3d{-0.07, 0.05, -0.025}).eval());
  values.insert(factor_keys.gyro_bias_to,
                (input.to.gyro_bias + Eigen::Vector3d{-0.003, 0.002, -0.001}).eval());
  values.insert(factor_keys.accel_bias_to,
                (input.to.accel_bias + Eigen::Vector3d{0.012, -0.008, 0.017}).eval());
  return values;
}

[[nodiscard]] Eigen::Matrix<double, 30, 1> canonicalCoordinates(
    const core::CondensedBoundaryTransition& input, const TestEndpointFrame& from_frame,
    const TestEndpointFrame& to_frame, const OdomEpochChartPlacement& chart,
    const AdjacentBoundaryKeys& factor_keys, const gtsam::Values& values) {
  Eigen::Matrix<double, 30, 1> delta;
  const core::Pose3d anchor_from = fromGtsam(values.at<gtsam::Pose3>(factor_keys.anchor_from));
  const core::Pose3d anchor_to = fromGtsam(values.at<gtsam::Pose3>(factor_keys.anchor_to));
  delta.segment<6>(0) =
      ((chart.H_map_odom * input.from.T_odom_imu).inverse() * anchor_from * from_frame.C_submap_imu)
          .log();
  delta.segment<3>(6) = chart.H_map_odom.so3().matrix().transpose() *
                            values.at<gtsam::Vector3>(factor_keys.velocity_from) -
                        input.from.velocity_odom;
  delta.segment<3>(9) =
      values.at<gtsam::Vector3>(factor_keys.gyro_bias_from) - input.from.gyro_bias;
  delta.segment<3>(12) =
      values.at<gtsam::Vector3>(factor_keys.accel_bias_from) - input.from.accel_bias;
  delta.segment<6>(15) =
      ((chart.H_map_odom * input.to.T_odom_imu).inverse() * anchor_to * to_frame.C_submap_imu)
          .log();
  delta.segment<3>(21) = chart.H_map_odom.so3().matrix().transpose() *
                             values.at<gtsam::Vector3>(factor_keys.velocity_to) -
                         input.to.velocity_odom;
  delta.segment<3>(24) = values.at<gtsam::Vector3>(factor_keys.gyro_bias_to) - input.to.gyro_bias;
  delta.segment<3>(27) = values.at<gtsam::Vector3>(factor_keys.accel_bias_to) - input.to.accel_bias;
  return delta;
}

[[nodiscard]] gtsam::Matrix numericalJacobian(const gtsam::NoiseModelFactor& factor,
                                              const gtsam::Values& values, gtsam::Key key,
                                              std::size_t dimension, bool pose_variable) {
  constexpr double kStep = 1.0e-6;
  gtsam::Matrix jacobian(factor.dim(), dimension);
  for (std::size_t column = 0U; column < dimension; ++column) {
    gtsam::Values plus = values;
    gtsam::Values minus = values;
    if (pose_variable) {
      gtsam::Vector6 perturbation = gtsam::Vector6::Zero();
      perturbation(static_cast<Eigen::Index>(column)) = kStep;
      const auto& center = values.at<gtsam::Pose3>(key);
      plus.update(key, center.compose(gtsam::Pose3::Expmap(perturbation)));
      minus.update(key, center.compose(gtsam::Pose3::Expmap(-perturbation)));
    } else {
      gtsam::Vector3 perturbation = gtsam::Vector3::Zero();
      perturbation(static_cast<Eigen::Index>(column)) = kStep;
      const auto& center = values.at<gtsam::Vector3>(key);
      plus.update(key, (center + perturbation).eval());
      minus.update(key, (center - perturbation).eval());
    }
    jacobian.col(static_cast<Eigen::Index>(column)) =
        (factor.unwhitenedError(plus) - factor.unwhitenedError(minus)) / (2.0 * kStep);
  }
  return jacobian;
}

TEST(AdjacentBoundaryFactor, EvaluatesExactCanonicalCoordinatesAndConstant) {
  const auto input = transition();
  ASSERT_EQ(core::validateCondensedBoundaryTransition(input),
            core::CondensedTransitionValidationError::None);
  const auto from_frame = fromFrame(input);
  const auto to_frame = toFrame(input);
  const auto chart = placement(input);
  const auto factor_keys = keys();
  const auto seal_pair = seals(input, from_frame, to_frame);
  const auto built =
      makeAdjacentBoundaryFactor(seal_pair.predecessor, seal_pair.current, chart, factor_keys);
  ASSERT_TRUE(built) << built.error().detail;
  const gtsam::Values values =
      valuesAwayFromCenters(input, from_frame, to_frame, chart, factor_keys);

  const Eigen::Map<const RowMajorMatrix> rows(input.boundary_factor.row_major_A.data(),
                                              input.boundary_factor.rows,
                                              input.boundary_factor.columns);
  const Eigen::Map<const gtsam::Vector> rhs(input.boundary_factor.rhs.data(),
                                            input.boundary_factor.rows);
  const gtsam::Vector expected =
      rows * canonicalCoordinates(input, from_frame, to_frame, chart, factor_keys, values) - rhs;
  const gtsam::Vector actual = built.value()->unwhitenedError(values);
  EXPECT_LT((actual - expected).norm(), 1.0e-10);
  EXPECT_NEAR(built.value()->error(values),
              0.5 * (expected.squaredNorm() + input.boundary_factor.constant_squared_error),
              1.0e-10);
}

TEST(AdjacentBoundaryFactor, AnalyticJacobiansMatchCentralDifferences) {
  const auto input = transition();
  const auto from_frame = fromFrame(input);
  const auto to_frame = toFrame(input);
  const auto chart = placement(input);
  const auto factor_keys = keys();
  const auto seal_pair = seals(input, from_frame, to_frame);
  const auto built =
      makeAdjacentBoundaryFactor(seal_pair.predecessor, seal_pair.current, chart, factor_keys);
  ASSERT_TRUE(built) << built.error().detail;
  const gtsam::Values values =
      valuesAwayFromCenters(input, from_frame, to_frame, chart, factor_keys);
  std::vector<gtsam::Matrix> analytic(8U);
  const gtsam::Vector residual = built.value()->unwhitenedError(values, analytic);
  EXPECT_EQ(residual.size(), input.boundary_factor.rows);
  const auto factor_keys_ordered = orderedKeys(factor_keys);
  constexpr std::array<std::size_t, 8> kDimensions{6U, 3U, 3U, 3U, 6U, 3U, 3U, 3U};
  for (std::size_t index = 0U; index < analytic.size(); ++index) {
    const gtsam::Matrix numeric =
        numericalJacobian(*built.value(), values, factor_keys_ordered[index], kDimensions[index],
                          index == 0U || index == 4U);
    EXPECT_LT((analytic[index] - numeric).cwiseAbs().maxCoeff(), 2.0e-6)
        << "variable index " << index;
  }
}

TEST(AdjacentBoundaryFactor, NullRowsAndCanonicalHessianArePreserved) {
  const auto input = transition();
  const auto from_frame = fromFrame(input);
  const auto to_frame = toFrame(input);
  const auto chart = placement(input);
  const auto factor_keys = keys();
  const auto seal_pair = seals(input, from_frame, to_frame);
  const auto built =
      makeAdjacentBoundaryFactor(seal_pair.predecessor, seal_pair.current, chart, factor_keys);
  ASSERT_TRUE(built) << built.error().detail;

  gtsam::Values center_values;
  center_values.insert(factor_keys.anchor_from, toGtsam(chart.H_map_odom * input.from.T_odom_imu *
                                                        from_frame.C_submap_imu.inverse()));
  center_values.insert(factor_keys.velocity_from,
                       (chart.H_map_odom.so3().matrix() * input.from.velocity_odom).eval());
  center_values.insert(factor_keys.gyro_bias_from, input.from.gyro_bias);
  center_values.insert(factor_keys.accel_bias_from, input.from.accel_bias);
  center_values.insert(factor_keys.anchor_to, toGtsam(chart.H_map_odom * input.to.T_odom_imu *
                                                      to_frame.C_submap_imu.inverse()));
  center_values.insert(factor_keys.velocity_to,
                       (chart.H_map_odom.so3().matrix() * input.to.velocity_odom).eval());
  center_values.insert(factor_keys.gyro_bias_to, input.to.gyro_bias);
  center_values.insert(factor_keys.accel_bias_to, input.to.accel_bias);

  std::vector<gtsam::Matrix> block_jacobians(8U);
  const gtsam::Vector residual = built.value()->unwhitenedError(center_values, block_jacobians);
  gtsam::Matrix actual_jacobian(input.boundary_factor.rows, 30);
  std::size_t offset = 0U;
  for (const gtsam::Matrix& block : block_jacobians) {
    actual_jacobian.block(0, static_cast<Eigen::Index>(offset), block.rows(), block.cols()) = block;
    offset += static_cast<std::size_t>(block.cols());
  }
  ASSERT_EQ(offset, 30U);

  gtsam::Matrix6 permutation = gtsam::Matrix6::Zero();
  permutation.topRightCorner<3, 3>().setIdentity();
  permutation.bottomLeftCorner<3, 3>().setIdentity();
  gtsam::Matrix coordinate_jacobian = gtsam::Matrix::Zero(30, 30);
  coordinate_jacobian.block<6, 6>(0, 0) =
      permutation * toGtsam(from_frame.C_submap_imu).inverse().AdjointMap();
  coordinate_jacobian.block<3, 3>(6, 6) = chart.H_map_odom.so3().matrix().transpose();
  coordinate_jacobian.block<3, 3>(9, 9).setIdentity();
  coordinate_jacobian.block<3, 3>(12, 12).setIdentity();
  coordinate_jacobian.block<6, 6>(15, 15) =
      permutation * toGtsam(to_frame.C_submap_imu).inverse().AdjointMap();
  coordinate_jacobian.block<3, 3>(21, 21) = chart.H_map_odom.so3().matrix().transpose();
  coordinate_jacobian.block<3, 3>(24, 24).setIdentity();
  coordinate_jacobian.block<3, 3>(27, 27).setIdentity();
  const Eigen::Map<const RowMajorMatrix> rows(input.boundary_factor.row_major_A.data(),
                                              input.boundary_factor.rows,
                                              input.boundary_factor.columns);
  const gtsam::Matrix expected_jacobian = rows * coordinate_jacobian;

  EXPECT_LT((actual_jacobian - expected_jacobian).cwiseAbs().maxCoeff(), 1.0e-10);
  EXPECT_LT((actual_jacobian.transpose() * actual_jacobian -
             expected_jacobian.transpose() * expected_jacobian)
                .cwiseAbs()
                .maxCoeff(),
            1.0e-10);
  EXPECT_LT(actual_jacobian.row(12).norm(), 1.0e-14);
  EXPECT_NEAR(residual(12), -input.boundary_factor.rhs[12], 1.0e-12);
  const Eigen::JacobiSVD<gtsam::Matrix> svd(actual_jacobian);
  EXPECT_EQ((svd.singularValues().array() > 1.0e-9).count(), 12);

  const auto linearized = built.value()->linearize(center_values);
  const auto jacobian_factor = boost::dynamic_pointer_cast<gtsam::JacobianFactor>(linearized);
  ASSERT_TRUE(jacobian_factor);
  const gtsam::Matrix augmented = jacobian_factor->augmentedJacobian();
  ASSERT_EQ(augmented.rows(), actual_jacobian.rows());
  ASSERT_EQ(augmented.cols(), actual_jacobian.cols() + 1);
  EXPECT_LT((augmented.leftCols(30) - actual_jacobian).cwiseAbs().maxCoeff(), 1.0e-12);
  EXPECT_LT((augmented.col(30) + residual).cwiseAbs().maxCoeff(), 1.0e-12);
}

TEST(AdjacentBoundaryFactor, RejectsMalformedIdentityLayoutChecksumAndPlacement) {
  const auto valid = transition();
  const auto valid_from = fromFrame(valid);
  const auto valid_to = toFrame(valid);
  const auto valid_placement = placement(valid);
  const auto valid_keys = keys();
  const auto valid_seals = seals(valid, valid_from, valid_to);

  auto malformed = valid_seals.current;
  std::swap(malformed.from_previous->local_transition.boundary_factor.layout[0],
            malformed.from_previous->local_transition.boundary_factor.layout[1]);
  auto rejected =
      makeAdjacentBoundaryFactor(valid_seals.predecessor, malformed, valid_placement, valid_keys);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, AdjacentBoundaryAdapterErrorCode::InvalidSparseLink);

  malformed = valid_seals.current;
  malformed.from_previous->local_transition.boundary_factor.checksum = {};
  rejected =
      makeAdjacentBoundaryFactor(valid_seals.predecessor, malformed, valid_placement, valid_keys);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, AdjacentBoundaryAdapterErrorCode::InvalidSparseLink);

  malformed = valid_seals.current;
  malformed.from_previous->local_transition.checksum = {};
  rejected =
      makeAdjacentBoundaryFactor(valid_seals.predecessor, malformed, valid_placement, valid_keys);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, AdjacentBoundaryAdapterErrorCode::InvalidSparseLink);

  auto tampered_predecessor = valid_seals.predecessor;
  tampered_predecessor.T_odom_submap.translation().x() += 1.0;
  rejected = makeAdjacentBoundaryFactor(tampered_predecessor, valid_seals.current, valid_placement,
                                        valid_keys);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, AdjacentBoundaryAdapterErrorCode::InvalidSparseLink);

  auto wrong_epoch = valid_placement;
  wrong_epoch.odom_epoch = core::OdomEpoch(999U);
  rejected = makeAdjacentBoundaryFactor(valid_seals.predecessor, valid_seals.current, wrong_epoch,
                                        valid_keys);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, AdjacentBoundaryAdapterErrorCode::OdomEpochMismatch);

  auto tilted = valid_placement;
  tilted.H_map_odom = pose(Eigen::Vector3d{1.0, 2.0, 3.0}, Eigen::Vector3d{0.02, 0.0, 0.1});
  rejected =
      makeAdjacentBoundaryFactor(valid_seals.predecessor, valid_seals.current, tilted, valid_keys);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, AdjacentBoundaryAdapterErrorCode::InvalidChartPlacement);

  auto duplicate_keys = valid_keys;
  duplicate_keys.accel_bias_to = duplicate_keys.anchor_from;
  rejected = makeAdjacentBoundaryFactor(valid_seals.predecessor, valid_seals.current,
                                        valid_placement, duplicate_keys);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, AdjacentBoundaryAdapterErrorCode::DuplicateKey);

  AdjacentBoundaryFactorLimits tiny_limits;
  tiny_limits.maximum_rows = 12U;
  tiny_limits.maximum_coefficients = 30U * 12U;
  rejected = makeAdjacentBoundaryFactor(valid_seals.predecessor, valid_seals.current,
                                        valid_placement, valid_keys, tiny_limits);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, AdjacentBoundaryAdapterErrorCode::CapacityExceeded);
}

}  // namespace
}  // namespace meridian::global::adjacent_internal
