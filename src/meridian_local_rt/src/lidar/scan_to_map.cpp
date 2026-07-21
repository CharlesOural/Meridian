#include "meridian/local_rt/lidar/scan_to_map.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>

namespace meridian::local_rt::lidar {
namespace {

using Clock = std::chrono::steady_clock;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix36d = Eigen::Matrix<double, 3, 6>;

std::int64_t elapsedNs(const Clock::time_point begin) noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
}

bool positiveFinite(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

bool lexicographicallyLess(const Point3d& lhs, const Point3d& rhs) noexcept {
  if (lhs.x() != rhs.x()) {
    return lhs.x() < rhs.x();
  }
  if (lhs.y() != rhs.y()) {
    return lhs.y() < rhs.y();
  }
  return lhs.z() < rhs.z();
}

std::optional<std::int64_t> checkedOffset(std::int64_t value, std::int64_t offset) noexcept {
  if ((offset > 0 && value > std::numeric_limits<std::int64_t>::max() - offset) ||
      (offset < 0 && value < std::numeric_limits<std::int64_t>::min() - offset)) {
    return std::nullopt;
  }
  return value + offset;
}

void validate(const ScanToMapOptions& options) {
  if (!positiveFinite(options.target_downsample_voxel_m) ||
      !positiveFinite(options.source_downsample_voxel_m) || options.maximum_active_owners == 0U ||
      options.maximum_factor_rows == 0U || options.minimum_correspondences < 3U ||
      options.minimum_correspondences > options.maximum_factor_rows ||
      options.maximum_icp_iterations == 0U || options.maximum_backtracking_steps == 0U ||
      !positiveFinite(options.maximum_correspondence_distance_m) ||
      !positiveFinite(options.point_sigma_m) || !positiveFinite(options.huber_mad_multiplier) ||
      !positiveFinite(options.minimum_huber_scale_m) ||
      !positiveFinite(options.maximum_huber_scale_m) ||
      options.minimum_huber_scale_m > options.maximum_huber_scale_m ||
      !positiveFinite(options.translation_convergence_m) ||
      !positiveFinite(options.rotation_convergence_rad) ||
      !positiveFinite(options.maximum_translation_step_m) ||
      !positiveFinite(options.maximum_rotation_step_rad) ||
      !positiveFinite(options.maximum_prediction_correction_m) ||
      !positiveFinite(options.maximum_prediction_correction_rad) ||
      !positiveFinite(options.relative_rank_tolerance) || options.relative_rank_tolerance >= 1.0 ||
      !positiveFinite(options.lm_damping)) {
    throw std::invalid_argument("scan-to-map options are incomplete or nonphysical");
  }
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const std::size_t middle = values.size() / 2U;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle),
                   values.end());
  const double upper = values[middle];
  if (values.size() % 2U != 0U) {
    return upper;
  }
  const double lower =
      *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle));
  return 0.5 * (lower + upper);
}

double robustScale(const std::vector<ScanToMapRow>& rows, const ScanToMapOptions& options) {
  std::vector<double> distances;
  distances.reserve(rows.size());
  for (const ScanToMapRow& row : rows) {
    distances.push_back(row.association_distance_m);
  }
  const double centre = median(distances);
  for (double& distance : distances) {
    distance = std::abs(distance - centre);
  }
  constexpr double kMadConsistency = 1.4826;
  return std::clamp(options.huber_mad_multiplier * kMadConsistency * median(std::move(distances)),
                    options.minimum_huber_scale_m, options.maximum_huber_scale_m);
}

double huberWeight(double norm, double scale) noexcept {
  return norm <= scale || norm <= 1.0e-15 ? 1.0 : scale / norm;
}

double huberCost(double squared_norm, double scale) noexcept {
  const double scale_squared = scale * scale;
  if (squared_norm <= scale_squared) {
    return squared_norm;
  }
  return 2.0 * scale * std::sqrt(squared_norm) - scale_squared;
}

bool limit(Eigen::Ref<Eigen::Vector3d> part, double maximum) noexcept {
  const double norm = part.norm();
  if (norm <= maximum) {
    return false;
  }
  part *= maximum / norm;
  return true;
}

struct Equations final {
  Matrix6d hessian{Matrix6d::Zero()};
  Vector6d gradient{Vector6d::Zero()};
  double cost{};
};

Matrix36d sourceJacobian(const Sophus::SE3d& odom_from_imu, const Sophus::SE3d& owner_odom_from_imu,
                         const Sophus::SE3d& imu_from_lidar, const Point3d& source_lidar,
                         bool finalized) {
  const Point3d source_imu = imu_from_lidar * source_lidar;
  Matrix36d result;
  if (finalized) {
    const Eigen::Matrix3d rotation = odom_from_imu.rotationMatrix();
    result.leftCols<3>() = rotation;
    result.rightCols<3>() = -rotation * Sophus::SO3d::hat(source_imu);
    return result;
  }
  const Sophus::SE3d owner_lidar_from_odom = (owner_odom_from_imu * imu_from_lidar).inverse();
  const Eigen::Matrix3d rotation =
      owner_lidar_from_odom.rotationMatrix() * odom_from_imu.rotationMatrix();
  result.leftCols<3>() = rotation;
  result.rightCols<3>() = -rotation * Sophus::SO3d::hat(source_imu);
  return result;
}

Point3d residual(const ScanToMapRow& row, const Sophus::SE3d& odom_from_imu,
                 const Sophus::SE3d& owner_odom_from_imu, const Sophus::SE3d& imu_from_lidar) {
  const Point3d source_odom = odom_from_imu * (imu_from_lidar * row.source_lidar);
  if (!row.active_target_state.has_value()) {
    return source_odom - row.target;
  }
  return (owner_odom_from_imu * imu_from_lidar).inverse() * source_odom - row.target;
}

void accumulateTiming(ScanToMapTiming& destination, const ScanToMapTiming& source) noexcept {
  destination.owner_selection_ns += source.owner_selection_ns;
  destination.live_composite_rebuild_ns += source.live_composite_rebuild_ns;
  destination.active_query_ns += source.active_query_ns;
  destination.finalized_query_ns += source.finalized_query_ns;
  destination.robust_scale_ns += source.robust_scale_ns;
  destination.linearization_ns += source.linearization_ns;
  destination.solve_ns += source.solve_ns;
  destination.active_queries += source.active_queries;
  destination.active_voxel_probes += source.active_voxel_probes;
  destination.active_candidate_points += source.active_candidate_points;
  destination.finalized_queries += source.finalized_queries;
  destination.finalized_voxel_probes += source.finalized_voxel_probes;
  destination.finalized_candidate_points += source.finalized_candidate_points;
}

template <typename OwnerPose>
bool evaluateQuality(ScanToMapResult& result, const Sophus::SE3d& odom_from_imu,
                     const Sophus::SE3d& imu_from_lidar, const ScanToMapOptions& options,
                     OwnerPose&& owner_pose, bool evaluate_spectrum) {
  const Clock::time_point scale_begin = Clock::now();
  result.huber_scale_m = robustScale(result.rows, options);
  result.timing.robust_scale_ns += elapsedNs(scale_begin);

  const Clock::time_point quality_begin = Clock::now();
  Matrix6d hessian = Matrix6d::Zero();
  double squared_error = 0.0;
  result.robust_cost = 0.0;
  result.active_rows = 0U;
  result.finalized_rows = 0U;
  for (ScanToMapRow& row : result.rows) {
    const Sophus::SE3d owner = owner_pose(row);
    const Point3d error = residual(row, odom_from_imu, owner, imu_from_lidar);
    const double norm = error.norm();
    const double weight = huberWeight(norm, result.huber_scale_m);
    squared_error += error.squaredNorm();
    result.robust_cost += huberCost(error.squaredNorm(), result.huber_scale_m);
    row.association_distance_m = norm;
    row.sqrt_weight_over_sigma = std::sqrt(weight) / options.point_sigma_m;
    if (row.active_target_state.has_value()) {
      ++result.active_rows;
    } else {
      ++result.finalized_rows;
    }
    if (evaluate_spectrum) {
      const Matrix36d jacobian =
          sourceJacobian(odom_from_imu, owner, imu_from_lidar, row.source_lidar,
                         !row.active_target_state.has_value());
      hessian.noalias() += weight * jacobian.transpose() * jacobian;
    }
  }
  result.rmse_m = std::sqrt(squared_error / static_cast<double>(result.rows.size()));

  if (evaluate_spectrum) {
    const Eigen::SelfAdjointEigenSolver<Matrix6d> spectrum(hessian);
    if (spectrum.info() != Eigen::Success || !spectrum.eigenvalues().allFinite()) {
      result.timing.linearization_ns += elapsedNs(quality_begin);
      return false;
    }
    const double maximum = spectrum.eigenvalues().maxCoeff();
    const double threshold = options.relative_rank_tolerance * maximum;
    result.observable_rank =
        maximum > 0.0 ? static_cast<std::size_t>(
                            (spectrum.eigenvalues().array() > threshold).template cast<int>().sum())
                      : 0U;
    const double minimum = spectrum.eigenvalues().minCoeff();
    result.condition_number =
        minimum > 0.0 ? maximum / minimum : std::numeric_limits<double>::infinity();
  }
  result.timing.linearization_ns += elapsedNs(quality_begin);
  return true;
}

void capRows(ScanToMapResult& result, std::size_t maximum_rows) {
  if (result.rows.size() <= maximum_rows) {
    return;
  }
  std::vector<ScanToMapRow> selected_rows;
  selected_rows.reserve(maximum_rows);
  for (std::size_t index = 0U; index < maximum_rows; ++index) {
    selected_rows.push_back(std::move(result.rows[index * result.rows.size() / maximum_rows]));
  }
  result.rows = std::move(selected_rows);
  result.active_rows = static_cast<std::size_t>(
      std::count_if(result.rows.begin(), result.rows.end(),
                    [](const ScanToMapRow& row) { return row.active_target_state.has_value(); }));
  result.finalized_rows = result.rows.size() - result.active_rows;
}

}  // namespace

struct ScanToMapTarget::Impl final {
  struct Owner final {
    core::StateId state_id;
    core::TimeNs time;
    PointCloud target_points_lidar;
    Sophus::SE3d odom_from_imu;
    std::unique_ptr<BoundedVoxelTarget> index;
  };

  struct NeighborOffset final {
    std::int64_t x{};
    std::int64_t y{};
    std::int64_t z{};
    double minimum_squared_distance_voxel_units{};
  };

  struct LivePoint final {
    Point3d point_odom{Point3d::Zero()};
    core::StateId owner_state;
    Point3d point_owner_lidar{Point3d::Zero()};
  };

  using LiveBucket = std::vector<LivePoint, Eigen::aligned_allocator<LivePoint>>;

  struct LiveVoxel final {
    LiveBucket points;
  };

  struct LiveHashSlot final {
    VoxelKey key;
    std::size_t voxel_index{};
    bool occupied{};
  };

  struct SelectedOwners final {
    std::vector<core::StateId> state_ids;
    std::int64_t selection_ns{};
  };

  struct LiveQueryStats final {
    std::uint64_t voxel_probes{};
    std::uint64_t candidate_points{};
  };

  struct LiveNearest final {
    core::StateId owner_state;
    Point3d point_owner_lidar{Point3d::Zero()};
    double squared_distance_m2{};
  };

  struct AssociationBatch final {
    std::vector<ScanToMapRow> rows;
    ScanToMapTiming timing;
  };

  explicit Impl(const ScanToMapOptions& options) : finalized(options.finalized_base) {
    const std::int64_t radius =
        static_cast<std::int64_t>(options.active_owner_index.max_neighbor_voxel_radius);
    const std::size_t side = 2U * options.active_owner_index.max_neighbor_voxel_radius + 1U;
    live_neighbor_offsets.reserve(side * side * side);
    for (std::int64_t dx = -radius; dx <= radius; ++dx) {
      for (std::int64_t dy = -radius; dy <= radius; ++dy) {
        for (std::int64_t dz = -radius; dz <= radius; ++dz) {
          const auto axis_minimum = [](std::int64_t offset) noexcept {
            return static_cast<double>(std::max<std::int64_t>(0, std::abs(offset) - 1));
          };
          const double minimum_x = axis_minimum(dx);
          const double minimum_y = axis_minimum(dy);
          const double minimum_z = axis_minimum(dz);
          live_neighbor_offsets.push_back(
              {dx, dy, dz, minimum_x * minimum_x + minimum_y * minimum_y + minimum_z * minimum_z});
        }
      }
    }
    std::sort(live_neighbor_offsets.begin(), live_neighbor_offsets.end(),
              [](const NeighborOffset& lhs, const NeighborOffset& rhs) {
                if (lhs.minimum_squared_distance_voxel_units !=
                    rhs.minimum_squared_distance_voxel_units) {
                  return lhs.minimum_squared_distance_voxel_units <
                         rhs.minimum_squared_distance_voxel_units;
                }
                if (lhs.x != rhs.x) {
                  return lhs.x < rhs.x;
                }
                if (lhs.y != rhs.y) {
                  return lhs.y < rhs.y;
                }
                return lhs.z < rhs.z;
              });
  }

  void invalidateLiveComposite() noexcept { live_composite_dirty = true; }

  [[nodiscard]] static std::size_t liveHashCapacity(std::size_t maximum_voxels) {
    constexpr std::size_t kMinimumCapacity = 2U;
    if (maximum_voxels > std::numeric_limits<std::size_t>::max() / 2U) {
      throw std::overflow_error("live composite voxel hash capacity overflow");
    }
    const std::size_t requested = std::max(kMinimumCapacity, maximum_voxels * 2U);
    std::size_t capacity = kMinimumCapacity;
    while (capacity < requested) {
      if (capacity > std::numeric_limits<std::size_t>::max() / 2U) {
        throw std::overflow_error("live composite voxel hash capacity overflow");
      }
      capacity *= 2U;
    }
    return capacity;
  }

  [[nodiscard]] const LiveBucket* findLiveBucket(const VoxelKey& key) const noexcept {
    if (live_hash_slots.empty()) {
      return nullptr;
    }
    const std::size_t mask = live_hash_slots.size() - 1U;
    std::size_t slot_index = VoxelKeyHash{}(key)&mask;
    for (;;) {
      const LiveHashSlot& slot = live_hash_slots[slot_index];
      if (!slot.occupied) {
        return nullptr;
      }
      if (slot.key == key) {
        return &live_voxels[slot.voxel_index].points;
      }
      slot_index = (slot_index + 1U) & mask;
    }
  }

  [[nodiscard]] SelectedOwners selectOwners(const Sophus::SE3d& odom_from_imu,
                                            std::size_t maximum_owners) const {
    const Clock::time_point begin = Clock::now();
    std::vector<std::pair<double, core::StateId>> ranked;
    ranked.reserve(owners.size());
    for (const auto& [state_id, owner] : owners) {
      ranked.emplace_back(
          (owner.odom_from_imu.translation() - odom_from_imu.translation()).squaredNorm(),
          state_id);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first != rhs.first ? lhs.first < rhs.first : lhs.second < rhs.second;
    });
    if (ranked.size() > maximum_owners) {
      ranked.erase(ranked.begin() + static_cast<std::ptrdiff_t>(maximum_owners), ranked.end());
    }

    SelectedOwners selected;
    selected.state_ids.reserve(ranked.size());
    for (const auto& [distance, state_id] : ranked) {
      static_cast<void>(distance);
      selected.state_ids.push_back(state_id);
    }
    // Query filtering is a binary search. Selection rank has already been
    // applied, so sorting by stable state identity does not change membership.
    std::sort(selected.state_ids.begin(), selected.state_ids.end());
    selected.selection_ns = elapsedNs(begin);
    return selected;
  }

  [[nodiscard]] std::int64_t ensureLiveComposite(const ScanToMapOptions& options,
                                                 const Sophus::SE3d& imu_from_lidar) const {
    if (!live_composite_dirty) {
      return 0;
    }
    const Clock::time_point begin = Clock::now();
    live_voxels.clear();
    live_hash_slots.clear();
    std::size_t expected_points = 0U;
    for (const auto& [state_id, owner] : owners) {
      static_cast<void>(state_id);
      expected_points += owner.index->pointCount();
    }
    const auto newest_owner = std::max_element(
        owners.begin(), owners.end(), [](const auto& lhs, const auto& rhs) {
          return lhs.second.time != rhs.second.time ? lhs.second.time < rhs.second.time
                                                    : lhs.first < rhs.first;
        });
    const Point3d retention_center_odom =
        newest_owner == owners.end() ? Point3d::Zero()
                                     : newest_owner->second.odom_from_imu.translation();
    const double retention_radius_squared = options.active_owner_index.retention_radius_m *
                                            options.active_owner_index.retention_radius_m;
    live_voxels.reserve(expected_points);
    live_hash_slots.resize(liveHashCapacity(expected_points));
    const std::size_t hash_mask = live_hash_slots.size() - 1U;

    for (const auto& [state_id, owner] : owners) {
      const Sophus::SE3d odom_from_lidar = owner.odom_from_imu * imu_from_lidar;
      const PointCloud indexed_points = owner.index->pointCloud();
      for (const Point3d& point_owner_lidar : indexed_points) {
        const Point3d point_odom = odom_from_lidar * point_owner_lidar;
        if ((point_odom - retention_center_odom).squaredNorm() > retention_radius_squared) {
          continue;
        }
        const VoxelKey key = pointToVoxelKey(point_odom, options.active_owner_index.voxel_size_m);
        std::size_t slot_index = VoxelKeyHash{}(key)&hash_mask;
        while (live_hash_slots[slot_index].occupied && live_hash_slots[slot_index].key != key) {
          slot_index = (slot_index + 1U) & hash_mask;
        }
        LiveHashSlot& slot = live_hash_slots[slot_index];
        if (!slot.occupied) {
          slot.key = key;
          slot.voxel_index = live_voxels.size();
          slot.occupied = true;
          live_voxels.push_back(LiveVoxel{.points = {}});
        }
        live_voxels[slot.voxel_index].points.push_back(
            LivePoint{.point_odom = point_odom,
                      .owner_state = state_id,
                      .point_owner_lidar = point_owner_lidar});
      }
    }
    for (LiveVoxel& voxel : live_voxels) {
      std::sort(voxel.points.begin(), voxel.points.end(),
                [](const LivePoint& lhs, const LivePoint& rhs) {
                  if (lhs.owner_state != rhs.owner_state) {
                    return lhs.owner_state < rhs.owner_state;
                  }
                  if (lhs.point_owner_lidar != rhs.point_owner_lidar) {
                    return lexicographicallyLess(lhs.point_owner_lidar, rhs.point_owner_lidar);
                  }
                  return lexicographicallyLess(lhs.point_odom, rhs.point_odom);
                });
    }
    live_composite_dirty = false;
    return elapsedNs(begin);
  }

  [[nodiscard]] std::optional<LiveNearest> nearestLive(
      const Point3d& query_odom, double maximum_distance_m,
      const std::vector<core::StateId>& selected_owners, const ScanToMapOptions& options,
      LiveQueryStats& stats) const {
    if (live_voxels.empty() || selected_owners.empty()) {
      return std::nullopt;
    }

    const double voxel_size = options.active_owner_index.voxel_size_m;
    const VoxelKey centre = pointToVoxelKey(query_odom, voxel_size);
    const std::int64_t radius =
        static_cast<std::int64_t>(std::ceil(maximum_distance_m / voxel_size));
    const double maximum_squared_distance = maximum_distance_m * maximum_distance_m;
    const double voxel_size_squared = voxel_size * voxel_size;
    const Point3d scaled = query_odom / voxel_size;
    const Point3d fractional =
        scaled - Point3d{std::floor(scaled.x()), std::floor(scaled.y()), std::floor(scaled.z())};
    const auto axisLowerDistance = [&](std::int64_t offset, double fraction) noexcept {
      if (offset > 0) {
        return (static_cast<double>(offset) - fraction) * voxel_size;
      }
      if (offset < 0) {
        return (fraction - static_cast<double>(offset + 1)) * voxel_size;
      }
      return 0.0;
    };

    std::optional<LiveNearest> nearest;
    for (const NeighborOffset& offset : live_neighbor_offsets) {
      const double incumbent_squared_distance =
          nearest.has_value() ? nearest->squared_distance_m2 : maximum_squared_distance;
      if (offset.minimum_squared_distance_voxel_units * voxel_size_squared >
          incumbent_squared_distance) {
        break;
      }
      if (std::abs(offset.x) > radius || std::abs(offset.y) > radius ||
          std::abs(offset.z) > radius) {
        continue;
      }
      const double lower_x = axisLowerDistance(offset.x, fractional.x());
      const double lower_y = axisLowerDistance(offset.y, fractional.y());
      const double lower_z = axisLowerDistance(offset.z, fractional.z());
      if (lower_x * lower_x + lower_y * lower_y + lower_z * lower_z > incumbent_squared_distance) {
        continue;
      }

      const auto x = checkedOffset(centre.x, offset.x);
      const auto y = checkedOffset(centre.y, offset.y);
      const auto z = checkedOffset(centre.z, offset.z);
      if (!x.has_value() || !y.has_value() || !z.has_value()) {
        continue;
      }
      ++stats.voxel_probes;
      const LiveBucket* bucket = findLiveBucket(VoxelKey{*x, *y, *z});
      if (bucket == nullptr) {
        continue;
      }
      for (const LivePoint& candidate : *bucket) {
        ++stats.candidate_points;
        if (!std::binary_search(selected_owners.begin(), selected_owners.end(),
                                candidate.owner_state)) {
          continue;
        }
        const double squared_distance = (candidate.point_odom - query_odom).squaredNorm();
        if (squared_distance > maximum_squared_distance) {
          continue;
        }
        const bool preferred =
            !nearest.has_value() || squared_distance < nearest->squared_distance_m2 ||
            (squared_distance == nearest->squared_distance_m2 &&
             (candidate.owner_state < nearest->owner_state ||
              (candidate.owner_state == nearest->owner_state &&
               lexicographicallyLess(candidate.point_owner_lidar, nearest->point_owner_lidar))));
        if (preferred) {
          nearest = LiveNearest{.owner_state = candidate.owner_state,
                                .point_owner_lidar = candidate.point_owner_lidar,
                                .squared_distance_m2 = squared_distance};
        }
      }
    }
    return nearest;
  }

  [[nodiscard]] AssociationBatch associate(std::span<const Point3d> source_points_lidar,
                                           const Sophus::SE3d& odom_from_imu,
                                           const std::vector<core::StateId>& selected_owners,
                                           const ScanToMapOptions& options,
                                           const Sophus::SE3d& imu_from_lidar) const {
    AssociationBatch batch;
    if (!selected_owners.empty()) {
      batch.timing.live_composite_rebuild_ns = ensureLiveComposite(options, imu_from_lidar);
    }

    std::vector<ScanToMapRow> rows(source_points_lidar.size());
    std::vector<std::uint8_t> valid(source_points_lidar.size(), std::uint8_t{0U});
    PointCloud source_points_odom(source_points_lidar.size());
    const Sophus::SE3d odom_from_lidar = odom_from_imu * imu_from_lidar;

    constexpr std::size_t kMaximumWorkers = 8U;
    constexpr std::size_t kMinimumPointsPerWorker = 512U;
    const unsigned hardware_workers = std::thread::hardware_concurrency();
    const std::size_t available_workers =
        hardware_workers == 0U ? 1U : static_cast<std::size_t>(hardware_workers);
    const std::size_t useful_workers =
        (source_points_lidar.size() + kMinimumPointsPerWorker - 1U) / kMinimumPointsPerWorker;
    const std::size_t worker_count =
        std::max<std::size_t>(1U, std::min({kMaximumWorkers, available_workers, useful_workers}));
    const auto parallel_for_sources = [&](auto&& operation) {
      std::vector<std::exception_ptr> worker_errors(worker_count);
      const auto run_worker = [&](std::size_t worker_index) noexcept {
        const std::size_t base_count = source_points_lidar.size() / worker_count;
        const std::size_t remainder = source_points_lidar.size() % worker_count;
        const std::size_t begin = worker_index * base_count + std::min(worker_index, remainder);
        const std::size_t end = begin + base_count + (worker_index < remainder ? 1U : 0U);
        try {
          for (std::size_t index = begin; index < end; ++index) {
            operation(worker_index, index);
          }
        } catch (...) {
          worker_errors[worker_index] = std::current_exception();
        }
      };

      std::vector<std::jthread> workers;
      workers.reserve(worker_count - 1U);
      for (std::size_t worker_index = 1U; worker_index < worker_count; ++worker_index) {
        workers.emplace_back([&, worker_index]() { run_worker(worker_index); });
      }
      run_worker(0U);
      for (std::jthread& worker : workers) {
        worker.join();
      }
      for (const std::exception_ptr& error : worker_errors) {
        if (error != nullptr) {
          std::rethrow_exception(error);
        }
      }
    };

    const Clock::time_point active_begin = Clock::now();
    std::vector<LiveQueryStats> live_worker_stats(worker_count);
    parallel_for_sources([&](std::size_t worker_index, std::size_t index) {
      const Point3d point_odom = odom_from_lidar * source_points_lidar[index];
      source_points_odom[index] = point_odom;
      if (selected_owners.empty()) {
        return;
      }
      const auto nearest = nearestLive(point_odom, options.maximum_correspondence_distance_m,
                                       selected_owners, options, live_worker_stats[worker_index]);
      if (!nearest.has_value()) {
        return;
      }
      rows[index] = ScanToMapRow{
          .source_lidar = source_points_lidar[index],
          .active_target_state = nearest->owner_state,
          .target = nearest->point_owner_lidar,
          .sqrt_weight_over_sigma = 0.0,
          .association_distance_m = std::sqrt(nearest->squared_distance_m2),
      };
      valid[index] = std::uint8_t{1U};
    });
    if (!selected_owners.empty()) {
      batch.timing.active_queries = source_points_lidar.size();
    }
    batch.timing.active_query_ns = elapsedNs(active_begin);
    for (const LiveQueryStats& worker_stats : live_worker_stats) {
      batch.timing.active_voxel_probes += worker_stats.voxel_probes;
      batch.timing.active_candidate_points += worker_stats.candidate_points;
    }

    const Clock::time_point finalized_begin = Clock::now();
    if (!finalized.empty()) {
      std::vector<VoxelTargetQueryStats> finalized_worker_stats(worker_count);
      parallel_for_sources([&](std::size_t worker_index, std::size_t index) {
        VoxelTargetQueryStats query_stats;
        const auto nearest = finalized.nearestNeighbor(
            source_points_odom[index], options.maximum_correspondence_distance_m, &query_stats);
        finalized_worker_stats[worker_index].voxel_hash_probes += query_stats.voxel_hash_probes;
        finalized_worker_stats[worker_index].occupied_buckets += query_stats.occupied_buckets;
        finalized_worker_stats[worker_index].point_candidates += query_stats.point_candidates;
        if (!nearest.has_value()) {
          return;
        }
        const double incumbent =
            valid[index] ? rows[index].association_distance_m * rows[index].association_distance_m
                         : std::numeric_limits<double>::infinity();
        // Exact distance ties remain attached to the live owner. This is
        // deterministic and preserves the relinearizable binary observation.
        if (nearest->squared_distance_m2 >= incumbent) {
          return;
        }
        rows[index] = ScanToMapRow{
            .source_lidar = source_points_lidar[index],
            .active_target_state = std::nullopt,
            .target = nearest->point,
            .sqrt_weight_over_sigma = 0.0,
            .association_distance_m = std::sqrt(nearest->squared_distance_m2),
        };
        valid[index] = std::uint8_t{1U};
      });
      batch.timing.finalized_queries = source_points_lidar.size();
      for (const VoxelTargetQueryStats& worker_stats : finalized_worker_stats) {
        batch.timing.finalized_voxel_probes += worker_stats.voxel_hash_probes;
        batch.timing.finalized_candidate_points += worker_stats.point_candidates;
      }
    }
    batch.timing.finalized_query_ns = elapsedNs(finalized_begin);

    batch.rows.reserve(source_points_lidar.size());
    for (std::size_t index = 0U; index < rows.size(); ++index) {
      if (valid[index]) {
        batch.rows.push_back(std::move(rows[index]));
      }
    }
    return batch;
  }

  std::map<core::StateId, Owner> owners;
  BoundedVoxelTarget finalized;
  std::vector<NeighborOffset> live_neighbor_offsets;
  mutable std::vector<LiveVoxel> live_voxels;
  mutable std::vector<LiveHashSlot> live_hash_slots;
  mutable bool live_composite_dirty{true};
};

ScanToMapTarget::ScanToMapTarget(ScanToMapOptions options, Sophus::SE3d imu_from_lidar)
    : options_(std::move(options)),
      imu_from_lidar_(std::move(imu_from_lidar)),
      impl_(std::make_unique<Impl>(options_)) {
  validate(options_);
  if (!imu_from_lidar_.matrix().allFinite()) {
    throw std::invalid_argument("scan-to-map extrinsic must be finite");
  }
  if (!options_.active_owner_index.max_voxels || !options_.finalized_base.max_voxels) {
    throw std::invalid_argument("scan-to-map target capacities must be positive");
  }
  BoundedVoxelTarget active_probe(options_.active_owner_index);
  if (!active_probe.supportsQueryDistance(options_.maximum_correspondence_distance_m) ||
      !impl_->finalized.supportsQueryDistance(options_.maximum_correspondence_distance_m)) {
    throw std::invalid_argument("scan-to-map voxel search radius cannot cover correspondence gate");
  }
}

ScanToMapTarget::~ScanToMapTarget() = default;
ScanToMapTarget::ScanToMapTarget(const ScanToMapTarget& other)
    : options_(other.options_),
      imu_from_lidar_(other.imu_from_lidar_),
      impl_(std::make_unique<Impl>(options_)) {
  impl_->finalized = other.impl_->finalized;
  for (const auto& [state_id, owner] : other.impl_->owners) {
    impl_->owners.emplace(state_id,
                          Impl::Owner{.state_id = owner.state_id,
                                      .time = owner.time,
                                      .target_points_lidar = owner.target_points_lidar,
                                      .odom_from_imu = owner.odom_from_imu,
                                      .index = std::make_unique<BoundedVoxelTarget>(*owner.index)});
  }
  impl_->invalidateLiveComposite();
}

ScanToMapTarget& ScanToMapTarget::operator=(const ScanToMapTarget& other) {
  if (this != &other) {
    ScanToMapTarget copy(other);
    *this = std::move(copy);
  }
  return *this;
}

ScanToMapTarget::ScanToMapTarget(ScanToMapTarget&&) noexcept = default;
ScanToMapTarget& ScanToMapTarget::operator=(ScanToMapTarget&&) noexcept = default;

bool ScanToMapTarget::empty() const noexcept {
  return impl_->owners.empty() && impl_->finalized.empty();
}

bool ScanToMapTarget::hasActiveOwner(core::StateId state_id) const noexcept {
  return impl_->owners.contains(state_id);
}

std::size_t ScanToMapTarget::activeOwnerCount() const noexcept {
  return impl_->owners.size();
}

std::size_t ScanToMapTarget::finalizedPointCount() const noexcept {
  return impl_->finalized.pointCount();
}

std::size_t ScanToMapTarget::finalizedVoxelCount() const noexcept {
  return impl_->finalized.voxelCount();
}

PointCloud ScanToMapTarget::registrationMapPointCloud() const {
  static_cast<void>(impl_->ensureLiveComposite(options_, imu_from_lidar_));
  PointCloud result = impl_->finalized.pointCloud();
  std::size_t live_points = 0U;
  for (const Impl::LiveVoxel& voxel : impl_->live_voxels) {
    live_points += voxel.points.size();
  }
  result.reserve(result.size() + live_points);
  for (const Impl::LiveVoxel& voxel : impl_->live_voxels) {
    for (const Impl::LivePoint& point : voxel.points) {
      result.push_back(point.point_odom);
    }
  }
  std::sort(result.begin(), result.end(), lexicographicallyLess);
  return result;
}

void ScanToMapTarget::admit(ScanFrame frame, const Sophus::SE3d& odom_from_imu) {
  if (!odom_from_imu.matrix().allFinite() || frame.target_points_lidar.empty() ||
      impl_->owners.contains(frame.state_id)) {
    throw std::invalid_argument(
        "active scan admission requires a unique state and finite geometry");
  }
  auto index = std::make_unique<BoundedVoxelTarget>(options_.active_owner_index);
  const VoxelTargetUpdateStats update =
      index->updateTargetFrame(frame.target_points_lidar, Point3d::Zero());
  if (update.inserted_points == 0U) {
    throw std::invalid_argument("active scan admission produced an empty voxel index");
  }
  const core::StateId id = frame.state_id;
  impl_->owners.emplace(id, Impl::Owner{.state_id = id,
                                        .time = frame.time,
                                        .target_points_lidar = std::move(frame.target_points_lidar),
                                        .odom_from_imu = odom_from_imu,
                                        .index = std::move(index)});
  impl_->invalidateLiveComposite();
}

void ScanToMapTarget::updateOwnerPose(core::StateId state_id, const Sophus::SE3d& odom_from_imu) {
  const auto found = impl_->owners.find(state_id);
  if (found == impl_->owners.end() || !odom_from_imu.matrix().allFinite()) {
    throw std::invalid_argument("active owner pose update is invalid");
  }
  found->second.odom_from_imu = odom_from_imu;
  impl_->invalidateLiveComposite();
}

FinalizedTargetStats ScanToMapTarget::finalize(core::StateId state_id,
                                               const Sophus::SE3d& odom_from_imu,
                                               const Point3d& retention_center_odom) {
  const auto found = impl_->owners.find(state_id);
  if (found == impl_->owners.end()) {
    throw std::invalid_argument("cannot finalize a state that does not own active geometry");
  }
  const Sophus::SE3d odom_from_lidar = odom_from_imu * imu_from_lidar_;
  const VoxelTargetUpdateStats update = impl_->finalized.update(
      found->second.target_points_lidar, odom_from_lidar, retention_center_odom);
  impl_->owners.erase(found);
  impl_->invalidateLiveComposite();
  return FinalizedTargetStats{.points = impl_->finalized.pointCount(),
                              .voxels = impl_->finalized.voxelCount(),
                              .update = update};
}

ScanToMapResult ScanToMapTarget::registerScan(std::span<const Point3d> source_points_lidar,
                                              const Sophus::SE3d& predicted_odom_from_imu) const {
  const Clock::time_point total_begin = Clock::now();
  ScanToMapResult result;
  result.odom_from_imu = predicted_odom_from_imu;
  if (empty()) {
    result.status = ScanToMapStatus::kEmptyTarget;
    result.timing.total_ns = elapsedNs(total_begin);
    return result;
  }
  if (source_points_lidar.empty() || !predicted_odom_from_imu.matrix().allFinite() ||
      std::any_of(source_points_lidar.begin(), source_points_lidar.end(),
                  [](const Point3d& point) { return !point.allFinite(); })) {
    result.status = ScanToMapStatus::kNumericalFailure;
    result.timing.total_ns = elapsedNs(total_begin);
    return result;
  }

  const Impl::SelectedOwners selected =
      impl_->selectOwners(predicted_odom_from_imu, options_.maximum_active_owners);
  result.selected_active_owners = selected.state_ids.size();
  result.timing.owner_selection_ns = selected.selection_ns;

  auto associate = [&](const Sophus::SE3d& pose) {
    Impl::AssociationBatch batch =
        impl_->associate(source_points_lidar, pose, selected.state_ids, options_, imu_from_lidar_);
    accumulateTiming(result.timing, batch.timing);
    return std::move(batch.rows);
  };

  auto ownerPose = [&](const ScanToMapRow& row) -> Sophus::SE3d {
    if (!row.active_target_state.has_value()) {
      return Sophus::SE3d();
    }
    return impl_->owners.at(*row.active_target_state).odom_from_imu;
  };

  auto equations = [&](const std::vector<ScanToMapRow>& rows, const Sophus::SE3d& pose,
                       double scale) {
    const Clock::time_point begin = Clock::now();
    Equations eq;
    for (const ScanToMapRow& row : rows) {
      const Sophus::SE3d owner = ownerPose(row);
      const Point3d error = residual(row, pose, owner, imu_from_lidar_);
      const double weight = huberWeight(error.norm(), scale);
      const Matrix36d jacobian = sourceJacobian(pose, owner, imu_from_lidar_, row.source_lidar,
                                                !row.active_target_state.has_value());
      eq.hessian.noalias() += weight * jacobian.transpose() * jacobian;
      eq.gradient.noalias() += weight * jacobian.transpose() * error;
      eq.cost += huberCost(error.squaredNorm(), scale);
    }
    result.timing.linearization_ns += elapsedNs(begin);
    return eq;
  };

  for (std::size_t iteration = 0U; iteration < options_.maximum_icp_iterations; ++iteration) {
    std::vector<ScanToMapRow> rows = associate(result.odom_from_imu);
    result.correspondences_before_cap = rows.size();
    if (rows.size() < options_.minimum_correspondences) {
      result.status = ScanToMapStatus::kInsufficientCorrespondences;
      result.rows = std::move(rows);
      result.timing.total_ns = elapsedNs(total_begin);
      return result;
    }
    const Clock::time_point scale_begin = Clock::now();
    const double scale = robustScale(rows, options_);
    result.timing.robust_scale_ns += elapsedNs(scale_begin);
    const Equations current = equations(rows, result.odom_from_imu, scale);
    const Eigen::SelfAdjointEigenSolver<Matrix6d> spectrum(current.hessian);
    if (spectrum.info() != Eigen::Success || !spectrum.eigenvalues().allFinite()) {
      result.status = ScanToMapStatus::kNumericalFailure;
      result.timing.total_ns = elapsedNs(total_begin);
      return result;
    }
    const double maximum = spectrum.eigenvalues().maxCoeff();
    const double threshold = options_.relative_rank_tolerance * maximum;
    result.observable_rank = static_cast<std::size_t>(
        (spectrum.eigenvalues().array() > threshold).template cast<int>().sum());
    if (!(maximum > 0.0) || result.observable_rank != 6U) {
      result.status = ScanToMapStatus::kDegenerate;
      result.rows = std::move(rows);
      result.timing.total_ns = elapsedNs(total_begin);
      return result;
    }
    result.condition_number = maximum / spectrum.eigenvalues().minCoeff();
    if (current.cost <= std::numeric_limits<double>::epsilon() * static_cast<double>(rows.size())) {
      result.iterations = iteration;
      break;
    }

    const Clock::time_point solve_begin = Clock::now();
    Matrix6d damped = current.hessian;
    damped.diagonal().array() +=
        options_.lm_damping * current.hessian.diagonal().cwiseAbs().array().max(1.0e-12);
    Eigen::LDLT<Matrix6d> decomposition(damped);
    Vector6d step = decomposition.solve(-current.gradient);
    result.timing.solve_ns += elapsedNs(solve_begin);
    if (decomposition.info() != Eigen::Success || !step.allFinite()) {
      result.status = ScanToMapStatus::kNumericalFailure;
      result.timing.total_ns = elapsedNs(total_begin);
      return result;
    }
    static_cast<void>(limit(step.head<3>(), options_.maximum_translation_step_m));
    static_cast<void>(limit(step.tail<3>(), options_.maximum_rotation_step_rad));
    bool decreased = false;
    Sophus::SE3d accepted_pose = result.odom_from_imu;
    Vector6d accepted_step = step;
    for (std::size_t backtrack = 0U; backtrack < options_.maximum_backtracking_steps; ++backtrack) {
      const Sophus::SE3d candidate = result.odom_from_imu * Sophus::SE3d::exp(accepted_step);
      if (candidate.matrix().allFinite() && equations(rows, candidate, scale).cost < current.cost) {
        accepted_pose = candidate;
        decreased = true;
        break;
      }
      accepted_step *= 0.5;
    }
    if (!decreased) {
      // A near-optimal finite step can round to the same cost at every
      // backtracking level. Once it is below the configured pose tolerance,
      // this is convergence rather than a registration failure.
      if (step.head<3>().norm() <= options_.translation_convergence_m &&
          step.tail<3>().norm() <= options_.rotation_convergence_rad) {
        result.iterations = iteration;
        break;
      }
      result.status = ScanToMapStatus::kNoDecreasingStep;
      result.rows = std::move(rows);
      result.timing.total_ns = elapsedNs(total_begin);
      return result;
    }
    result.odom_from_imu = accepted_pose;
    result.iterations = iteration + 1U;
    if (accepted_step.head<3>().norm() <= options_.translation_convergence_m &&
        accepted_step.tail<3>().norm() <= options_.rotation_convergence_rad) {
      break;
    }
  }

  result.rows = associate(result.odom_from_imu);
  result.correspondences_before_cap = result.rows.size();
  if (result.rows.size() < options_.minimum_correspondences) {
    result.status = ScanToMapStatus::kInsufficientCorrespondences;
    result.timing.total_ns = elapsedNs(total_begin);
    return result;
  }
  if (!evaluateQuality(result, result.odom_from_imu, imu_from_lidar_, options_, ownerPose, false)) {
    result.status = ScanToMapStatus::kNumericalFailure;
    result.timing.total_ns = elapsedNs(total_begin);
    return result;
  }
  capRows(result, options_.maximum_factor_rows);

  const Sophus::SE3d correction = predicted_odom_from_imu.inverse() * result.odom_from_imu;
  result.correction_translation_m = correction.translation().norm();
  result.correction_rotation_rad = correction.so3().log().norm();
  if (result.correction_translation_m > options_.maximum_prediction_correction_m ||
      result.correction_rotation_rad > options_.maximum_prediction_correction_rad) {
    result.status = ScanToMapStatus::kPredictionCorrectionExceeded;
  } else {
    result.status = ScanToMapStatus::kAccepted;
  }
  result.timing.total_ns = elapsedNs(total_begin);
  return result;
}

ScanToMapResult ScanToMapTarget::associateScan(std::span<const Point3d> source_points_lidar,
                                               const Sophus::SE3d& odom_from_imu) const {
  const Clock::time_point total_begin = Clock::now();
  ScanToMapResult result;
  result.odom_from_imu = odom_from_imu;
  if (empty()) {
    result.status = ScanToMapStatus::kEmptyTarget;
    result.timing.total_ns = elapsedNs(total_begin);
    return result;
  }
  if (source_points_lidar.empty() || !odom_from_imu.matrix().allFinite() ||
      std::any_of(source_points_lidar.begin(), source_points_lidar.end(),
                  [](const Point3d& point) { return !point.allFinite(); })) {
    result.status = ScanToMapStatus::kNumericalFailure;
    result.timing.total_ns = elapsedNs(total_begin);
    return result;
  }

  const Impl::SelectedOwners selected =
      impl_->selectOwners(odom_from_imu, options_.maximum_active_owners);
  result.selected_active_owners = selected.state_ids.size();
  result.timing.owner_selection_ns = selected.selection_ns;
  Impl::AssociationBatch batch = impl_->associate(source_points_lidar, odom_from_imu,
                                                  selected.state_ids, options_, imu_from_lidar_);
  accumulateTiming(result.timing, batch.timing);
  result.rows = std::move(batch.rows);
  result.correspondences_before_cap = result.rows.size();
  if (result.rows.size() < options_.minimum_correspondences) {
    result.status = ScanToMapStatus::kInsufficientCorrespondences;
    result.timing.total_ns = elapsedNs(total_begin);
    return result;
  }

  const auto owner_pose = [&](const ScanToMapRow& row) -> Sophus::SE3d {
    if (!row.active_target_state.has_value()) {
      return Sophus::SE3d();
    }
    return impl_->owners.at(*row.active_target_state).odom_from_imu;
  };
  if (!evaluateQuality(result, odom_from_imu, imu_from_lidar_, options_, owner_pose, true)) {
    result.status = ScanToMapStatus::kNumericalFailure;
    result.timing.total_ns = elapsedNs(total_begin);
    return result;
  }
  capRows(result, options_.maximum_factor_rows);
  result.status = ScanToMapStatus::kAccepted;
  result.timing.total_ns = elapsedNs(total_begin);
  return result;
}

const char* toString(ScanToMapStatus status) noexcept {
  switch (status) {
    case ScanToMapStatus::kAccepted:
      return "accepted";
    case ScanToMapStatus::kEmptyTarget:
      return "empty_target";
    case ScanToMapStatus::kInsufficientCorrespondences:
      return "insufficient_correspondences";
    case ScanToMapStatus::kDegenerate:
      return "degenerate";
    case ScanToMapStatus::kNoDecreasingStep:
      return "no_decreasing_step";
    case ScanToMapStatus::kPredictionCorrectionExceeded:
      return "prediction_correction_exceeded";
    case ScanToMapStatus::kNumericalFailure:
      return "numerical_failure";
  }
  return "unknown";
}

}  // namespace meridian::local_rt::lidar
