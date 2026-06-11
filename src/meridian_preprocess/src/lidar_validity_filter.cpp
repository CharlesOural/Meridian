#include "meridian/preprocess/ilidar_preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "meridian/common/point.hpp"
#include "meridian/debug/log.hpp"
#include "meridian/debug/telemetry.hpp"
#include "meridian/debug/telemetry_keys.hpp"

namespace meridian {

namespace {

constexpr const char* kLogModule = "preprocess.lidar";

// The self-hit mask: axis-aligned volumes in the sensor frame describing where the rig
// occludes the LiDAR. No geometry source is wired through config yet, so the mask is
// empty (no volumes) and contains() always returns false. The hook stays here so the
// gate order and telemetry have one home once a mask-volume set is loaded.
class SelfHitMask {
 public:
  bool empty() const { return true; }
  bool contains(const Eigen::Vector3f& /*p*/) const { return false; }
};

struct ValidityStats {
  std::uint32_t n_nan = 0;
  std::uint32_t n_blind = 0;
  std::uint32_t n_far = 0;
  std::uint32_t n_selfhit = 0;
  std::uint32_t n_intensity = 0;
  std::uint32_t n_in = 0;
  std::uint32_t n_out = 0;        // survivors after the validity gate
  std::uint32_t n_voxel_out = 0;  // survivors after the surf-voxel downsample
};

bool finite3(const Eigen::Vector3f& v) {
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

// A validity survivor paired with its decode index (position in the input cloud), the
// last documented tie-break for the deterministic per-voxel representative.
struct IndexedPoint {
  LidarPoint p;
  std::uint32_t decode_index = 0;
};

// Integer voxel coordinate, hashed for the downsample bucket map.
struct VoxelKey {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;
  bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey& k) const {
    // Three odd 64-bit multipliers folded together; collisions only cost a bucket walk.
    std::uint64_t h = static_cast<std::uint64_t>(k.x) * 0x9E3779B97F4A7C15ULL;
    h ^= static_cast<std::uint64_t>(k.y) * 0xC2B2AE3D27D4EB4FULL;
    h ^= static_cast<std::uint64_t>(k.z) * 0x165667B19E3779F9ULL;
    h ^= h >> 29;
    return static_cast<std::size_t>(h);
  }
};

VoxelKey voxelOf(const Eigen::Vector3f& xyz, double edge) {
  const double inv = 1.0 / edge;
  return VoxelKey{static_cast<std::int64_t>(std::floor(xyz.x() * inv)),
                  static_cast<std::int64_t>(std::floor(xyz.y() * inv)),
                  static_cast<std::int64_t>(std::floor(xyz.z() * inv))};
}

class LidarValidityFilter final : public ILidarPreprocessor {
 public:
  LidarValidityFilter(const PreprocLidar& cfg, const Extrinsic& T_imu_lidar,
                      double nominal_rate_hz, TelemetrySink* telemetry)
      : cfg_(cfg),
        T_imu_lidar_(T_imu_lidar),
        nominal_rate_hz_(nominal_rate_hz),
        telemetry_(telemetry) {}

  LidarScan process(const LidarScan& raw) const override {
    MERIDIAN_SCOPED_TIME(telemetry_, keys::stage::PreprocessLidarValidity, raw.stamp_start);

    LidarScan out = raw;  // copies metadata; points pointer shared until we rebuild it
    const PointCloud* in = raw.points.get();
    if (in == nullptr || in->empty()) {
      ValidityStats stats;
      stats.n_in = in == nullptr ? 0 : static_cast<std::uint32_t>(in->size());
      emit(stats, raw.stamp_start);
      return out;
    }

    const float blind2 = static_cast<float>(cfg_.blind * cfg_.blind);
    const float det2 = static_cast<float>(cfg_.det_range * cfg_.det_range);
    const float imin = static_cast<float>(cfg_.i_min);
    const float imax = static_cast<float>(cfg_.i_max);
    const int filter_num = cfg_.point_filter_num < 1 ? 1 : cfg_.point_filter_num;

    ValidityStats stats;
    stats.n_in = static_cast<std::uint32_t>(in->size());

    // Survivors carry their decode index for the voxel tie-break.
    std::vector<IndexedPoint> kept;
    kept.reserve(in->size());

    // Five checks, cheapest first, then decimation over valid points. valid counts
    // every survivor; a point is kept on the Nth, 2Nth, ... valid point.
    std::uint64_t valid = 0;
    std::uint32_t idx = 0;
    for (const LidarPoint& p : *in) {
      const std::uint32_t this_index = idx++;
      if (!finite3(p.xyz)) {
        ++stats.n_nan;
        continue;
      }
      const float r2 = p.xyz.squaredNorm();
      if (r2 < blind2) {
        ++stats.n_blind;
        continue;
      }
      if (r2 > det2) {
        ++stats.n_far;
        continue;
      }
      if (mask_.contains(p.xyz)) {
        ++stats.n_selfhit;
        continue;
      }
      if (cfg_.intensity_gate && (p.intensity < imin || p.intensity > imax)) {
        ++stats.n_intensity;
        continue;
      }
      ++valid;
      if (valid % static_cast<std::uint64_t>(filter_num) != 0) {
        continue;  // decimate by valid count
      }
      kept.push_back(IndexedPoint{p, this_index});
    }
    stats.n_out = static_cast<std::uint32_t>(kept.size());

    // Surf-voxel downsample: bound spatial density to one (or surf_max_pts) representative
    // per voxel. The representative is chosen deterministically so the survivor set is a
    // function of the input only, never of iteration order.
    PointCloud surf = downsample(kept);
    std::sort(surf.begin(), surf.end(), [](const LidarPoint& a, const LidarPoint& b) {
      return a.t_offset_ns < b.t_offset_ns;
    });
    stats.n_voxel_out = static_cast<std::uint32_t>(surf.size());
    emit(stats, raw.stamp_start);

    const Duration original_sweep = raw.sweep_duration;

    // Reuse the input buffer only when no point was dropped at any stage (validity kept
    // everything, no voxel cell exceeded the cap, decimation off) and the input is already
    // time-sorted, so the survivor order matches and no copy is needed.
    if (stats.n_voxel_out == stats.n_in && filter_num == 1) {
      const bool already_sorted = std::is_sorted(
          in->begin(), in->end(), [](const LidarPoint& a, const LidarPoint& b) {
            return a.t_offset_ns < b.t_offset_ns;
          });
      if (already_sorted) {
        out.points = raw.points;
        out.sweep_duration = sweepDuration(*in, original_sweep);
        return out;
      }
    }

    out.sweep_duration = sweepDuration(surf, original_sweep);
    out.points = std::make_shared<const PointCloud>(std::move(surf));
    return out;
  }

 private:
  // Reduces the validity survivors to at most surf_max_pts per voxel. Determinism: the
  // representative selection depends only on point coordinates / time / decode index, not
  // on traversal order, so a reordered input yields the identical survivor set.
  PointCloud downsample(const std::vector<IndexedPoint>& kept) const {
    PointCloud out;
    if (kept.empty()) {
      return out;
    }
    const double edge = cfg_.voxel_surf_m > 0.0 ? cfg_.voxel_surf_m : 0.5;
    const int cap = cfg_.surf_max_pts < 1 ? 1 : cfg_.surf_max_pts;

    if (cap == 1) {
      // Keep the point nearest the voxel centre; ties broken by ascending t_offset_ns then
      // by ascending decode index. Both tie-breaks are order-independent.
      struct Best {
        IndexedPoint pt;
        float dist2 = 0.f;  // squared distance to the voxel centre
      };
      std::unordered_map<VoxelKey, Best, VoxelKeyHash> cells;
      cells.reserve(kept.size());
      for (const IndexedPoint& ip : kept) {
        const VoxelKey key = voxelOf(ip.p.xyz, edge);
        const Eigen::Vector3f centre(
            (static_cast<float>(key.x) + 0.5f) * static_cast<float>(edge),
            (static_cast<float>(key.y) + 0.5f) * static_cast<float>(edge),
            (static_cast<float>(key.z) + 0.5f) * static_cast<float>(edge));
        const float d2 = (ip.p.xyz - centre).squaredNorm();
        auto it = cells.find(key);
        if (it == cells.end()) {
          cells.emplace(key, Best{ip, d2});
          continue;
        }
        Best& b = it->second;
        if (closerThan(d2, ip, b.dist2, b.pt)) {
          b.pt = ip;
          b.dist2 = d2;
        }
      }
      out.reserve(cells.size());
      for (const auto& kv : cells) {
        out.push_back(kv.second.pt.p);
      }
      return out;
    }

    // cap > 1: bucket points per voxel, then keep at most cap by reservoir sampling with a
    // seeded RNG so the draw is unbiased over the cell and identical across replays.
    std::unordered_map<VoxelKey, std::vector<IndexedPoint>, VoxelKeyHash> cells;
    cells.reserve(kept.size());
    for (const IndexedPoint& ip : kept) {
      cells[voxelOf(ip.p.xyz, edge)].push_back(ip);
    }
    std::mt19937_64 rng(cfg_.surf_seed);
    out.reserve(kept.size());
    for (auto& kv : cells) {
      std::vector<IndexedPoint>& bucket = kv.second;
      const std::size_t n = bucket.size();
      if (n <= static_cast<std::size_t>(cap)) {
        for (const IndexedPoint& ip : bucket) out.push_back(ip.p);
        continue;
      }
      // Sort the bucket into a canonical order first so the reservoir draw is a function of
      // the seed and the cell's content, not of which input order filled the bucket.
      std::sort(bucket.begin(), bucket.end(),
                [](const IndexedPoint& a, const IndexedPoint& b) {
                  return a.decode_index < b.decode_index;
                });
      std::vector<IndexedPoint> reservoir(bucket.begin(),
                                          bucket.begin() + static_cast<std::ptrdiff_t>(cap));
      for (std::size_t i = static_cast<std::size_t>(cap); i < n; ++i) {
        std::uniform_int_distribution<std::size_t> pick(0, i);
        const std::size_t j = pick(rng);
        if (j < static_cast<std::size_t>(cap)) {
          reservoir[j] = bucket[i];
        }
      }
      for (const IndexedPoint& ip : reservoir) out.push_back(ip.p);
    }
    return out;
  }

  // True when candidate c should replace the current best b under the documented order:
  // smaller distance-to-centre wins; on a tie, smaller t_offset_ns; then smaller decode
  // index.
  static bool closerThan(float c_d2, const IndexedPoint& c, float b_d2,
                         const IndexedPoint& b) {
    if (c_d2 != b_d2) {
      return c_d2 < b_d2;
    }
    if (c.p.t_offset_ns != b.p.t_offset_ns) {
      return c.p.t_offset_ns < b.p.t_offset_ns;
    }
    return c.decode_index < b.decode_index;
  }

  // Recomputes sweep_duration as the surviving sweep-end (max t_offset_ns), floored so a
  // sparse / over-filtered scan cannot collapse the deskew horizon. The floor only ever
  // raises the value toward one fraction of a nominal period and is itself capped at the
  // original (pre-filter) span, so no surviving point is pushed outside the horizon and
  // the anchor is never invented past the true sweep end.
  Duration sweepDuration(const PointCloud& pts, Duration original_sweep) const {
    std::int32_t hi = 0;
    for (const LidarPoint& p : pts) {
      hi = std::max(hi, p.t_offset_ns);
    }
    const Duration surviving_end = static_cast<Duration>(hi);
    if (nominal_rate_hz_ <= 0.0) {
      return surviving_end;
    }
    const double frac = cfg_.sweep_floor_frac;
    const Duration floor_ns = static_cast<Duration>(
        std::llround(frac / nominal_rate_hz_ * static_cast<double>(kNanosPerSecond)));
    return std::max(surviving_end, std::min(floor_ns, original_sweep));
  }

  void emit(const ValidityStats& s, Timestamp t) const {
    if (s.n_voxel_out == 0 && s.n_in > 0) {
      MERIDIAN_WARN(kLogModule, "event", "lidar/empty_after_filter", "n_in", s.n_in);
    }
    if (telemetry_ == nullptr) {
      return;
    }
    telemetry_->scalar(keys::lidar::NIn, static_cast<double>(s.n_in), t);
    telemetry_->scalar(keys::lidar::NOut, static_cast<double>(s.n_out), t);
    telemetry_->scalar(keys::lidar::NVoxelOut, static_cast<double>(s.n_voxel_out), t);
    telemetry_->scalar(keys::lidar::NNan, static_cast<double>(s.n_nan), t);
    telemetry_->scalar(keys::lidar::NBlind, static_cast<double>(s.n_blind), t);
    telemetry_->scalar(keys::lidar::NFar, static_cast<double>(s.n_far), t);
    telemetry_->scalar(keys::lidar::NSelfhit, static_cast<double>(s.n_selfhit), t);
    telemetry_->scalar(keys::lidar::NIntensity, static_cast<double>(s.n_intensity), t);
    const double sfrac =
        s.n_in > 0 ? static_cast<double>(s.n_selfhit) / static_cast<double>(s.n_in) : 0.0;
    telemetry_->scalar(keys::lidar::SelfhitFrac, sfrac, t);
  }

  PreprocLidar cfg_;
  Extrinsic T_imu_lidar_;
  double nominal_rate_hz_ = 0.0;
  TelemetrySink* telemetry_ = nullptr;
  SelfHitMask mask_{};
};

}  // namespace

std::unique_ptr<ILidarPreprocessor> makeLidarPreprocessor(const PreprocessConfig& cfg,
                                                          const Extrinsic& T_imu_lidar,
                                                          double nominal_rate_hz,
                                                          TelemetrySink* telemetry) {
  return std::make_unique<LidarValidityFilter>(cfg.lidar, T_imu_lidar, nominal_rate_hz,
                                               telemetry);
}

}  // namespace meridian
