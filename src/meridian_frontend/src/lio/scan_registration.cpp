#include "lio/scan_registration.hpp"

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <chrono>
#include <sophus/se3.hpp>

namespace meridian::lio {
namespace {

using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat36 = Eigen::Matrix<double, 3, 6>;

using SteadyClock = std::chrono::steady_clock;
double ms_since(SteadyClock::time_point t0) {
  return std::chrono::duration<double, std::milli>(SteadyClock::now() - t0).count();
}

// World up-gravity vector [m/s^2]; static initialization aligns the world z-axis
// with the measured up direction.
const Eigen::Vector3d kGravityUpWorld{0.0, 0.0, ImuTracker::kGravityMagnitude};

struct NormalEq {
  Mat6 H = Mat6::Zero();
  Vec6 b = Vec6::Zero();
  double chi = 0.0;
  int n = 0;
};

// Unaveraged normal equations of the data term at pose T. r = exp(dx)*T*p - q gives, at
// dx = 0 in the left/world chart, dr/ddx = [I | -hat(T*p)] with translation-first columns.
// The per-correspondence terms are summed across threads (the nearest lookup is a const
// read of the map, so concurrent probing is safe); the reduction order is the thread
// schedule, which shifts the sum by floating-point rounding only — below the ATE noise.
NormalEq accumulateData(const std::vector<Eigen::Vector3d>& keypoints_body,
                        const NearestLookup& nearest, const Sophus::SE3d& T) {
  NormalEq eq;
  const int n = static_cast<int>(keypoints_body.size());
#pragma omp declare reduction(mergeEq:NormalEq                                            \
        : omp_out.H += omp_in.H, omp_out.b += omp_in.b, omp_out.chi += omp_in.chi,        \
          omp_out.n += omp_in.n) initializer(omp_priv = NormalEq{})
#pragma omp parallel for reduction(mergeEq : eq) schedule(static)
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d& p = keypoints_body[static_cast<std::size_t>(i)];
    const Eigen::Vector3d pw = T * p;
    Eigen::Vector3d neighbor;
    if (!nearest(pw, &neighbor)) {
      continue;
    }
    const Eigen::Vector3d r = pw - neighbor;
    Mat36 J;
    J.leftCols<3>() = Eigen::Matrix3d::Identity();
    J.rightCols<3>() = -Sophus::SO3d::hat(pw);
    eq.H.noalias() += J.transpose() * J;
    eq.b.noalias() += J.transpose() * r;
    eq.chi += r.squaredNorm();
    ++eq.n;
  }
  return eq;
}

}  // namespace

ScanRegistration::ScanRegistration(const FrontendLio& cfg) : cfg_(cfg) {}

std::vector<Eigen::Vector3d> ScanRegistration::deskew(const LidarScan& scan,
                                                      const Pose& T_body_lidar,
                                                      const Eigen::Vector3d& omega_body,
                                                      const Eigen::Vector3d& vel_body,
                                                      Timestamp t_end) const {
  std::vector<Eigen::Vector3d> out;
  if (!scan.points) {
    return out;
  }
  out.reserve(scan.points->size());
  const Sophus::SE3d T_bl(T_body_lidar.q, T_body_lidar.t);
  Vec6 screw;
  screw.head<3>() = vel_body;
  screw.tail<3>() = omega_body;
  for (const LidarPoint& pt : *scan.points) {
    const Eigen::Vector3d p_lidar = pt.xyz.cast<double>();
    const double range = p_lidar.norm();
    if (!(range > 0.0) || range > cfg_.max_range_m) {
      continue;
    }
    // Under the constant screw xi = [v; w], the body frame at t_i relates to the frame
    // at t_end by exp((t_i - t_end) * xi); applying it re-expresses the return in the
    // sweep-end body frame.
    const double dt = to_seconds(scan.stamp_start + pt.t_offset_ns - t_end);
    out.push_back(Sophus::SE3d::exp(dt * screw) * (T_bl * p_lidar));
  }
  return out;
}

RegistrationResult ScanRegistration::registerScan(
    const std::vector<Eigen::Vector3d>& keypoints_body, const VoxelGridMap& map,
    const Pose& T_world_body_guess, const MotionPrior& prior) const {
  return registerScan(
      keypoints_body,
      [&map](const Eigen::Vector3d& query, Eigen::Vector3d* neighbor) {
        return map.nearest(query, neighbor);
      },
      T_world_body_guess, prior);
}

RegistrationResult ScanRegistration::registerScan(
    const std::vector<Eigen::Vector3d>& keypoints_body, const NearestLookup& nearest,
    const Pose& T_world_body_guess, const MotionPrior& prior) const {
  RegistrationResult out;
  Sophus::SE3d T(T_world_body_guess.q, T_world_body_guess.t);

  // The regularizer weight 1/beta fades with acceleration excitation; a single sample
  // yields no variance, so the block is disabled below two samples.
  const double beta = (cfg_.min_beta > 0.0 && prior.imu_count > 1)
                          ? cfg_.min_beta * (1.0 + prior.accel_mag_variance)
                          : -1.0;
  // Up-gravity in the guess body frame, frozen across the iteration: the regularizer
  // penalizes roll/pitch drift away from the prior orientation.
  const Eigen::Vector3d g_anchor = T.so3().inverse() * kGravityUpWorld;

  for (int iter = 0; iter < cfg_.icp_max_iterations; ++iter) {
    const auto t_assoc0 = SteadyClock::now();
    const NormalEq eq = accumulateData(keypoints_body, nearest, T);
    out.assoc_ms += ms_since(t_assoc0);
    out.gn_iters = iter + 1;
    if (eq.n == 0) {
      break;  // association lost; report the failure from the current pose
    }
    const double inv_n = 1.0 / static_cast<double>(eq.n);
    Mat6 H = eq.H * inv_n;
    Vec6 b = eq.b * inv_n;
    if (beta > 0.0) {
      // r_ori = R^T*g_up - g_anchor; with R <- exp(phi^)*R the predicted gravity is
      // R^T*exp(-phi^)*g_up ~= r_ori + g_anchor + R^T*hat(g_up)*phi, so
      // J_ori = [0 | R^T*hat(g_up)] in the same left/world chart as the data term.
      const Eigen::Matrix3d Rt = T.so3().inverse().matrix();
      const Eigen::Vector3d r_ori = Rt * kGravityUpWorld - g_anchor;
      Mat36 J_ori = Mat36::Zero();
      J_ori.rightCols<3>() = Rt * Sophus::SO3d::hat(kGravityUpWorld);
      H.noalias() += J_ori.transpose() * J_ori / beta;
      b.noalias() += J_ori.transpose() * r_ori / beta;
    }
    const auto t_ldlt0 = SteadyClock::now();
    const Vec6 dx = H.ldlt().solve(-b);
    out.ldlt_ms += ms_since(t_ldlt0);
    T = Sophus::SE3d::exp(dx) * T;
    out.dx_norm = dx.norm();
    if (out.dx_norm < cfg_.convergence_eps) {
      out.converged = true;
      break;
    }
  }

  out.pose = Pose{T.unit_quaternion(), T.translation()};

  const auto t_cov0 = SteadyClock::now();
  // Re-associate at the solved pose: the unaveraged data-term normal equations are the
  // Laplace information. The heuristic gravity prior carries no measurement
  // information, so it is excluded and unobserved axes stay reported as unobserved.
  const NormalEq fin = accumulateData(keypoints_body, nearest, T);
  out.n_corr = fin.n;
  out.chi = fin.chi;

  // Information is a quadratic form: with dx_left = Ad(T)*dx_right, the left-chart
  // H becomes Ad(T)^T * H * Ad(T) in the right/body chart.
  const Mat6 adj_T = T.Adj();
  out.info_body = adj_T.transpose() * fin.H * adj_T;

  const int dof = 3 * fin.n - 6;
  if (fin.n >= cfg_.min_keypoints && dof > 0) {
    // Each correspondence contributes a 3-vector residual and the fit absorbs the 6
    // pose DoF, so chi / (3N - 6) estimates the isotropic measurement variance.
    const double sigma2 = fin.chi / static_cast<double>(dof);
    const Mat6 cov_left = sigma2 * (fin.H + 1e-6 * Mat6::Identity()).inverse();
    // exp(dx_left)*T = T*exp(dx_right) gives dx_right = Ad(T^-1)*dx_left to first
    // order, so the left-chart covariance transports by Ad(T^-1) on both sides.
    const Mat6 adj_T_inv = T.inverse().Adj();
    out.cov_right = adj_T_inv * cov_left * adj_T_inv.transpose();
  } else {
    out.converged = false;
  }
  out.cov_ms = ms_since(t_cov0);
  return out;
}

}  // namespace meridian::lio
