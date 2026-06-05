#include "meridian/preprocess/ilidar_preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "meridian/common/point.hpp"
#include "meridian/debug/log.hpp"
#include "meridian/debug/telemetry.hpp"

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
  std::uint32_t n_out = 0;
};

bool finite3(const Eigen::Vector3f& v) {
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

class LidarValidityFilter final : public ILidarPreprocessor {
 public:
  LidarValidityFilter(const PreprocLidar& cfg, const Extrinsic& T_imu_lidar,
                      TelemetrySink* telemetry)
      : cfg_(cfg), T_imu_lidar_(T_imu_lidar), telemetry_(telemetry) {}

  LidarScan process(const LidarScan& raw) const override {
    MERIDIAN_SCOPED_TIME(telemetry_, "preprocess.lidar.validity", raw.stamp_start);

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

    PointCloud kept;
    kept.reserve(in->size());

    // Five checks, cheapest first, then decimation over valid points. valid counts
    // every survivor; a point is kept on the Nth, 2Nth, ... valid point.
    std::uint64_t valid = 0;
    for (const LidarPoint& p : *in) {
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
      kept.push_back(p);
    }

    std::sort(kept.begin(), kept.end(),
              [](const LidarPoint& a, const LidarPoint& b) {
                return a.t_offset_ns < b.t_offset_ns;
              });

    stats.n_out = static_cast<std::uint32_t>(kept.size());
    emit(stats, raw.stamp_start);

    // Reuse the input buffer only when no point was dropped, decimation kept everything
    // (filter_num == 1 so n_out == n_in implies every point passed), and the input is
    // already time-sorted (so the survivor order matches and no copy is needed).
    if (stats.n_out == stats.n_in && filter_num == 1) {
      const bool already_sorted = std::is_sorted(
          in->begin(), in->end(), [](const LidarPoint& a, const LidarPoint& b) {
            return a.t_offset_ns < b.t_offset_ns;
          });
      if (already_sorted) {
        out.points = raw.points;
        out.sweep_duration = sweepDuration(*in);
        return out;
      }
    }

    out.sweep_duration = sweepDuration(kept);
    out.points = std::make_shared<const PointCloud>(std::move(kept));
    return out;
  }

 private:
  static Duration sweepDuration(const PointCloud& pts) {
    if (pts.empty()) {
      return 0;
    }
    // t_offset_ns is relative to the scan's stamp_start, so the sweep ends at the
    // largest offset, not at the surviving span: filtering away the earliest columns
    // must not pull t_end below points still present in the cloud.
    std::int32_t hi = pts.front().t_offset_ns;
    for (const LidarPoint& p : pts) {
      hi = std::max(hi, p.t_offset_ns);
    }
    return static_cast<Duration>(hi);
  }

  void emit(const ValidityStats& s, Timestamp t) const {
    if (s.n_out == 0 && s.n_in > 0) {
      MERIDIAN_WARN(kLogModule, "event", "lidar/empty_after_filter", "n_in", s.n_in);
    }
    if (telemetry_ == nullptr) {
      return;
    }
    telemetry_->scalar("lidar/n_in", static_cast<double>(s.n_in), t);
    telemetry_->scalar("lidar/n_out", static_cast<double>(s.n_out), t);
    telemetry_->scalar("lidar/n_nan", static_cast<double>(s.n_nan), t);
    telemetry_->scalar("lidar/n_blind", static_cast<double>(s.n_blind), t);
    telemetry_->scalar("lidar/n_far", static_cast<double>(s.n_far), t);
    telemetry_->scalar("lidar/n_selfhit", static_cast<double>(s.n_selfhit), t);
    telemetry_->scalar("lidar/n_intensity", static_cast<double>(s.n_intensity), t);
    const double frac =
        s.n_in > 0 ? static_cast<double>(s.n_selfhit) / static_cast<double>(s.n_in) : 0.0;
    telemetry_->scalar("lidar/selfhit_frac", frac, t);
  }

  PreprocLidar cfg_;
  Extrinsic T_imu_lidar_;
  TelemetrySink* telemetry_ = nullptr;
  SelfHitMask mask_{};
};

}  // namespace

std::unique_ptr<ILidarPreprocessor> makeLidarPreprocessor(const PreprocessConfig& cfg,
                                                          const Extrinsic& T_imu_lidar,
                                                          TelemetrySink* telemetry) {
  return std::make_unique<LidarValidityFilter>(cfg.lidar, T_imu_lidar, telemetry);
}

}  // namespace meridian
