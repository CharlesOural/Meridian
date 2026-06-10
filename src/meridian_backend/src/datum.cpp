#include "datum.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace meridian::backend {

void DatumInitializer::add(const Eigen::Vector3d& p_enu, const Eigen::Vector3d& p_map_antenna,
                           double speed_mps, Timestamp stamp) {
  buf_.push_back(Corr{p_enu, p_map_antenna, speed_mps, stamp});
}

DatumResult DatumInitializer::try_lock(double min_baseline_m, double min_excitation_m,
                                       double min_speed_mps, int min_moving_fixes,
                                       double yaw_sigma_max_rad) {
  DatumResult result;
  const std::size_t n = buf_.size();
  if (n < 2) {
    return result;
  }

  // Pre-gate (a): cumulative arc length of the map antenna track since the first fix.
  double baseline = 0.0;
  for (std::size_t i = 1; i < n; ++i) {
    baseline += (buf_[i].map - buf_[i - 1].map).norm();
  }
  if (baseline <= min_baseline_m) {
    return result;
  }

  // Pre-gate (c): enough fixes captured while the body was actually moving.
  int moving = 0;
  for (const Corr& c : buf_) {
    if (c.speed > min_speed_mps) {
      ++moving;
    }
  }
  if (moving < min_moving_fixes) {
    return result;
  }

  // De-mean both tracks; only the horizontal (E, N) components drive the yaw fit.
  Eigen::Vector3d mean_enu = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_map = Eigen::Vector3d::Zero();
  for (const Corr& c : buf_) {
    mean_enu += c.enu;
    mean_map += c.map;
  }
  mean_enu /= static_cast<double>(n);
  mean_map /= static_cast<double>(n);

  // Pre-gate (b): horizontal excitation = span of the de-meaned ENU positions along
  // their dominant principal axis. The eigenvector of the 2x2 horizontal scatter for the
  // larger eigenvalue is the principal direction; the projected min-max span is the span.
  Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
  for (const Corr& c : buf_) {
    const Eigen::Vector2d d = c.enu.head<2>() - mean_enu.head<2>();
    scatter += d * d.transpose();
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(scatter);
  const Eigen::Vector2d principal = es.eigenvectors().col(1);  // larger eigenvalue
  double proj_min = std::numeric_limits<double>::max();
  double proj_max = std::numeric_limits<double>::lowest();
  for (const Corr& c : buf_) {
    const double p = (c.enu.head<2>() - mean_enu.head<2>()).dot(principal);
    proj_min = std::min(proj_min, p);
    proj_max = std::max(proj_max, p);
  }
  const double excitation = proj_max - proj_min;
  if (excitation <= min_excitation_m) {
    return result;
  }

  // Planar Umeyama with scale = 1: fit map_horiz ~= R_yaw * enu_horiz + t. The optimal
  // yaw comes from the 2x2 horizontal cross-covariance H = sum (map-mean)(enu-mean)^T;
  // for a pure rotation the maximizer of trace(R^T H) is yaw = atan2(H10-H01, H00+H11).
  Eigen::Matrix2d cross = Eigen::Matrix2d::Zero();
  for (const Corr& c : buf_) {
    const Eigen::Vector2d dm = c.map.head<2>() - mean_map.head<2>();
    const Eigen::Vector2d de = c.enu.head<2>() - mean_enu.head<2>();
    cross += dm * de.transpose();
  }
  const double yaw = std::atan2(cross(1, 0) - cross(0, 1), cross(0, 0) + cross(1, 1));

  // Yaw uncertainty from the Gauss-Newton Hessian of the planar yaw residual: the
  // information in the heading is the planar moment of inertia of the de-meaned ENU
  // track, H_yaw = sum |enu_horiz - mean|^2. Weakly excited tracks (collinear, short)
  // give a small Hessian and a large sigma. sigma_yaw = 1/sqrt(H_yaw).
  double h_yaw = 0.0;
  for (const Corr& c : buf_) {
    h_yaw += (c.enu.head<2>() - mean_enu.head<2>()).squaredNorm();
  }
  if (h_yaw <= 0.0) {
    return result;
  }
  const double yaw_sigma = 1.0 / std::sqrt(h_yaw);
  if (yaw_sigma >= yaw_sigma_max_rad) {
    return result;
  }

  // T_map_enu = (R_yaw about +z, t), t = mean_map - R_yaw * mean_enu so the means align.
  // The yaw acts on the full 3D ENU vector; vertical (up) passes through unrotated.
  const Eigen::Quaterniond q(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
  const Eigen::Matrix3d R = q.toRotationMatrix();
  const Eigen::Vector3d t = mean_map - R * mean_enu;

  result.locked = true;
  result.T_map_enu = Pose{q, t};
  result.yaw_sigma_rad = yaw_sigma;
  return result;
}

}  // namespace meridian::backend
