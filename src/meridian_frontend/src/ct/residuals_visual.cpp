#include "ct/residuals_visual.hpp"

#include <basalt/spline/ceres_spline_helper.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sophus/so3.hpp>
#include <utility>
#include <vector>

namespace meridian::ct {

namespace {

constexpr int kSplineOrder = 4;
constexpr int kPatch = VisualObservation::kPatch;  // 8
constexpr int kHalf = kPatch / 2;                  // 4
constexpr int kPatchArea = kPatch * kPatch;        // 64
constexpr int kMaxSearchLevel = VisualObservation::kLevels - 1;

// Whitened-residual Huber threshold: the photometric residual fed to Ceres is
// divided by sqrt(img_point_cov), so it is unit-variance under the noise model and a
// one-sigma threshold switches to the robust regime once a pixel disagrees by more
// than its own expected spread.
constexpr double kHuberWhitened = 1.0;

// Reference camera pose -> point/normal in that camera frame.
Eigen::Vector3d toCam(const Pose& T_w_c, const Eigen::Vector3d& p_world) {
  return T_w_c.inverse() * p_world;
}

}  // namespace

ExposureChain::ExposureChain(const FrontendVisual& cfg) : cfg_(cfg) {}

std::size_t ExposureChain::addFrame(Timestamp t, double inv_expo) {
  const double lo = cfg_.inv_expo_min;
  taus_.push_back(inv_expo > lo ? inv_expo : lo);
  times_.push_back(t);
  return taus_.size() - 1;
}

namespace {

// tau_k - tau_{k-1}, whitened by 1/sqrt(inv_expo_cov * dt): a longer gap admits a
// larger exposure step, a shorter gap pins consecutive frames tighter together.
struct ExposureRandomWalk {
  explicit ExposureRandomWalk(double weight) : weight_(weight) {}
  template <class T>
  bool operator()(const T* prev, const T* cur, T* residual) const {
    residual[0] = T(weight_) * (cur[0] - prev[0]);
    return true;
  }
  double weight_;
};

// (tau_0 - prior) / prior_std: anchors the first in-window exposure.
struct ExposurePrior {
  ExposurePrior(double prior, double weight) : prior_(prior), weight_(weight) {}
  template <class T>
  bool operator()(const T* tau, T* residual) const {
    residual[0] = T(weight_) * (tau[0] - T(prior_));
    return true;
  }
  double prior_;
  double weight_;
};

}  // namespace

void ExposureChain::addTo(ceres::Problem& problem, double prior, double prior_std) {
  if (taus_.empty()) {
    return;
  }
  for (std::size_t k = 0; k < taus_.size(); ++k) {
    double* b = &taus_[k];
    if (!problem.HasParameterBlock(b)) {
      problem.AddParameterBlock(b, 1);
    }
    // Strict positivity: a non-positive inverse exposure would flip the photometric
    // residual sign, so clamp every block above inv_expo_min.
    problem.SetParameterLowerBound(b, 0, cfg_.inv_expo_min);
  }

  for (std::size_t k = 1; k < taus_.size(); ++k) {
    const double dt = std::max(1e-3, to_seconds(times_[k] - times_[k - 1]));
    const double var = std::max(1e-12, cfg_.inv_expo_cov * dt);
    const double weight = 1.0 / std::sqrt(var);
    auto* cost = new ceres::AutoDiffCostFunction<ExposureRandomWalk, 1, 1, 1>(
        new ExposureRandomWalk(weight));
    problem.AddResidualBlock(cost, nullptr, &taus_[k - 1], &taus_[k]);
  }

  // First-frame prior: a positive std pulls tau_0 toward the prior; a zero std means
  // that frame's exposure is held fixed at the prior value.
  double* first = &taus_[0];
  if (prior_std > 0.0) {
    const double weight = 1.0 / prior_std;
    auto* cost =
        new ceres::AutoDiffCostFunction<ExposurePrior, 1, 1>(new ExposurePrior(prior, weight));
    problem.AddResidualBlock(cost, nullptr, first);
  } else {
    taus_[0] = prior > cfg_.inv_expo_min ? prior : cfg_.inv_expo_min;
    if (problem.HasParameterBlock(first)) {
      problem.SetParameterBlockConstant(first);
    }
  }
}

bool warpMatrixAffineHomography(const CameraModel& cam, const Eigen::Vector2d& px_ref,
                                const Eigen::Vector3d& xyz_ref_cam,
                                const Eigen::Vector3d& n_ref_cam, const Pose& T_cur_ref,
                                int level_ref, Eigen::Matrix2d* A_cur_ref) {
  if (A_cur_ref == nullptr) {
    return false;
  }
  // Plane-induced homography mapping reference-camera rays to current-camera rays:
  //   H = R_cur_ref ( (n . x_ref) I - t n^T ),   t = T_cur_ref^{-1}.translation()
  // which equals (n.x_ref)(R_cur_ref + t_cur_ref n^T / (n.x_ref)) up to the positive
  // projective scale (n.x_ref); the scale cancels in the world2cam differences below.
  const Eigen::Vector3d t = T_cur_ref.inverse().t;
  const double nx = n_ref_cam.dot(xyz_ref_cam);
  const Eigen::Matrix3d H =
      T_cur_ref.R() * (nx * Eigen::Matrix3d::Identity() - t * n_ref_cam.transpose());

  // Probe two reference rays one half-patch apart (in ref level-0 pixels) and push
  // them through H into the current image; finite differences of the projected
  // pixels give the 2x2 affine warp.
  const double step = static_cast<double>(kHalf) * static_cast<double>(1 << level_ref);
  const Eigen::Vector3d f_ref = cam.unproject(px_ref);
  const Eigen::Vector3d f_du_ref = cam.unproject(px_ref + Eigen::Vector2d(step, 0.0));
  const Eigen::Vector3d f_dv_ref = cam.unproject(px_ref + Eigen::Vector2d(0.0, step));

  // Scale the reference rays to lie on the plane (so H maps them consistently); H is
  // linear so the overall ray scale is irrelevant to the projected differences.
  const Eigen::Vector3d f_cur = H * (f_ref * xyz_ref_cam.z());
  const Eigen::Vector3d f_du_cur = H * (f_du_ref * xyz_ref_cam.z());
  const Eigen::Vector3d f_dv_cur = H * (f_dv_ref * xyz_ref_cam.z());

  Eigen::Vector2d px_cur, px_du_cur, px_dv_cur;
  // The warp is meaningful only when all three probes land in front of the camera
  // and on the image; a probe falling off the edge means the patch straddles the
  // border, which the inBounds gate would reject anyway.
  if (f_cur.z() <= 0.0 || f_du_cur.z() <= 0.0 || f_dv_cur.z() <= 0.0) {
    return false;
  }
  if (!cam.project(f_cur, &px_cur) || !cam.project(f_du_cur, &px_du_cur) ||
      !cam.project(f_dv_cur, &px_dv_cur)) {
    return false;
  }
  A_cur_ref->col(0) = (px_du_cur - px_cur) / static_cast<double>(kHalf);
  A_cur_ref->col(1) = (px_dv_cur - px_cur) / static_cast<double>(kHalf);
  return A_cur_ref->allFinite();
}

int bestSearchLevel(const Eigen::Matrix2d& A_cur_ref, int max_level) {
  int level = 0;
  double d = std::abs(A_cur_ref.determinant());
  while (d > 3.0 && level < max_level) {
    ++level;
    d *= 0.25;
  }
  return level;
}

bool warpWellConditioned(const Eigen::Matrix2d& A, double det_min, double det_max,
                         double cond_max) {
  if (!A.allFinite()) {
    return false;
  }
  const double det = std::abs(A.determinant());
  if (det < det_min || det > det_max) {
    return false;
  }
  // Condition number = ratio of singular values; a near-singular warp stretches the
  // patch unboundedly along one axis even when |det| is in band.
  Eigen::JacobiSVD<Eigen::Matrix2d> svd(A);
  const double s_max = svd.singularValues()(0);
  const double s_min = svd.singularValues()(1);
  if (!(s_min > 0.0)) {
    return false;
  }
  return (s_max / s_min) <= cond_max;
}

double patchNCC(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t n = std::min(a.size(), b.size());
  if (n == 0) {
    return 0.0;
  }
  const double mean_a = std::accumulate(a.begin(), a.begin() + n, 0.0) / static_cast<double>(n);
  const double mean_b = std::accumulate(b.begin(), b.begin() + n, 0.0) / static_cast<double>(n);
  double num = 0.0, da = 0.0, db = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double xa = a[i] - mean_a;
    const double xb = b[i] - mean_b;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  return num / std::sqrt(da * db + 1e-10);
}

namespace {

// One photometric patch residual at a single camera frame, autodiff over the 4 SO(3)
// + 4 R^3 knot blocks of t_mid's segment plus the current frame's inverse-exposure
// block (the reference exposure is a fixed constant). Image sampling is linearized at
// the projected pixel: I_cur(u) ~= I0 + grad_I . (u - u0), so only the smooth spline
// / projection / exposure chain runs through autodiff. Every patch pixel shares the
// same projected-centre displacement (the patch grid is a fixed regular offset, not
// re-chained through the pose), so a single delta-pixel feeds all kPatchArea
// residuals.
//
// Parameter block layout (handed to DynamicAutoDiffCostFunction):
//   [0..3]   SO(3) knots (quaternion, xyzw)
//   [4..7]   R^3 knots
//   [8]      current-frame inverse exposure tau_cur
struct VisualPatchResidual {
  VisualPatchResidual(const Eigen::Quaterniond& q_fe_c, const Eigen::Vector3d& t_fe_c,
                      const Eigen::Vector3d& p_world, const Eigen::Matrix<double, 2, 3>& Jpi,
                      double u, double inv_dt, double tau_ref, double weight,
                      std::vector<double> ref_patch, std::vector<double> cur_I0,
                      std::vector<Eigen::Vector2d> cur_grad)
      : q_fe_c_(q_fe_c),
        t_fe_c_(t_fe_c),
        p_world_(p_world),
        Jpi_(Jpi),
        u_(u),
        inv_dt_(inv_dt),
        tau_ref_(tau_ref),
        weight_(weight),
        ref_patch_(std::move(ref_patch)),
        cur_I0_(std::move(cur_I0)),
        cur_grad_(std::move(cur_grad)) {}

  template <class T>
  bool operator()(T const* const* params, T* residual) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec2T = Eigen::Matrix<T, 2, 1>;
    using SO3T = Sophus::SO3<T>;

    SO3T R_w_fe;
    basalt::CeresSplineHelper<kSplineOrder>::template evaluate_lie<T, Sophus::SO3>(
        params, u_, inv_dt_, &R_w_fe);
    Vec3T p_w_fe;
    basalt::CeresSplineHelper<kSplineOrder>::template evaluate<T, 3, 0>(params + kSplineOrder, u_,
                                                                        inv_dt_, &p_w_fe);

    // World point -> current camera frame: P_c = R_cf (R_wfe^T (P_w - p_wfe)) + t_cf.
    const SO3T R_fe_c = SO3T(q_fe_c_.cast<T>());
    const Vec3T p_fe = R_w_fe.inverse() * (p_world_.cast<T>() - p_w_fe);
    const Vec3T p_c = R_fe_c.inverse() * (p_fe - t_fe_c_.cast<T>());

    // First-order pixel displacement of the patch centre. The 2x3 projection
    // Jacobian and the linearization point p_c0 are fixed; the autodiff variable is
    // p_c through the spline, giving the pixel shift du = Jpi (p_c - p_c0).
    const Vec3T dpc = p_c - p_c0_.cast<T>();
    const Vec2T du = Jpi_.cast<T>() * dpc;

    const T tau_cur = params[2 * kSplineOrder][0];
    for (int i = 0; i < kPatchArea; ++i) {
      // I_cur(u) ~= I0 + grad_I . (u - u0); since u - u0 == du is shared, only the
      // per-pixel I0 and gradient differ.
      const T i_cur = T(cur_I0_[static_cast<std::size_t>(i)]) +
                      cur_grad_[static_cast<std::size_t>(i)].cast<T>().dot(du);
      residual[i] =
          T(weight_) * (tau_cur * i_cur - T(tau_ref_) * T(ref_patch_[static_cast<std::size_t>(i)]));
    }
    return true;
  }

  // Set by the builder once the linearization point is known.
  Eigen::Vector3d p_c0_ = Eigen::Vector3d::Zero();

  Eigen::Quaterniond q_fe_c_;
  Eigen::Vector3d t_fe_c_;
  Eigen::Vector3d p_world_;
  Eigen::Matrix<double, 2, 3> Jpi_;
  double u_;
  double inv_dt_;
  double tau_ref_;
  double weight_;
  std::vector<double> ref_patch_;
  std::vector<double> cur_I0_;
  std::vector<Eigen::Vector2d> cur_grad_;
};

}  // namespace

VisualAssocStats addVisualResiduals(ceres::Problem& problem, SplineWindow& spline,
                                    const CameraModel& cam, const Pose& T_fe_cam,
                                    const ImagePyramidView& img, Timestamp t_mid_expo,
                                    ExposureChain& expo, std::size_t expo_index,
                                    const VisualMap& vmap, const FrontendVisual& cfg,
                                    std::vector<VisualUsedPoint>* used) {
  VisualAssocStats stats;
  if (!cam.valid() || expo.empty() || expo_index >= expo.size() || !spline.covers(t_mid_expo)) {
    return stats;
  }

  const Pose T_w_fe = spline.pose(t_mid_expo);
  const Pose T_w_c = T_w_fe * T_fe_cam;
  const std::vector<const VisualPoint*> candidates = vmap.visibleCandidates(cam, T_w_c);
  stats.candidates = static_cast<int>(candidates.size());

  const double tau_cur = expo.value(expo_index);
  const double weight = 1.0 / std::sqrt(cfg.img_point_cov > 0.0 ? cfg.img_point_cov : 1.0);
  const double huber = kHuberWhitened;

  const SplineWindow::SegmentRef seg = spline.segmentFor(t_mid_expo);

  for (const VisualPoint* pt : candidates) {
    // Gate 1 (grid occupancy) and gate 2 (depth-continuity) are applied by the map
    // in visibleCandidates(); here we own gates 3..5 (warp/level, NCC, SSD).
    if (pt == nullptr || !pt->normal_initialized) {
      continue;  // no fitted normal -> excluded, reconsidered next frame
    }
    const Pose& T_w_cref = pt->T_w_c_ref;

    // Project the point centre into the current camera at the linearization pose.
    const Eigen::Vector3d p_c0 = toCam(T_w_c, pt->p_world);
    if (p_c0.z() <= 0.0) {
      continue;
    }
    Eigen::Vector2d u0;
    if (!cam.project(p_c0, &u0)) {
      continue;
    }

    // Reference-frame geometry for the warp, with the normal-sign guard.
    const Eigen::Vector3d xyz_ref_cam = toCam(T_w_cref, pt->p_world);
    if (xyz_ref_cam.z() <= 0.0) {
      continue;
    }
    // Normal-sign guard: orient the plane normal so n.xyz_ref > 0 (the plane depth
    // along the normal is positive). A back-facing normal would negate the plane-
    // induced homography, mapping the warped rays behind the current camera and
    // mirroring the patch; this orientation keeps H.xyz_ref a positive multiple of
    // the forward current-frame point.
    Eigen::Vector3d n_ref_cam = T_w_cref.q.conjugate() * pt->n_world;
    if (n_ref_cam.dot(xyz_ref_cam) < 0.0) {
      n_ref_cam = -n_ref_cam;
    }
    const Eigen::Vector2d px_ref = [&] {
      Eigen::Vector2d p;
      cam.project(xyz_ref_cam, &p);
      return p;
    }();

    const Pose T_cur_ref = T_w_c.inverse() * T_w_cref;

    // Gate 3: warp / level. Compute the affine warp at ref level 0, pick the best
    // pyramid level, then verify the warp is well-conditioned.
    Eigen::Matrix2d A_cur_ref;
    if (!warpMatrixAffineHomography(cam, px_ref, xyz_ref_cam, n_ref_cam, T_cur_ref, 0,
                                    &A_cur_ref)) {
      continue;
    }
    const int level = bestSearchLevel(A_cur_ref, kMaxSearchLevel);
    if (level < 0 || level >= img.levels() || level >= VisualObservation::kLevels) {
      continue;
    }
    if (!warpWellConditioned(A_cur_ref, cfg.warp_det_min, cfg.warp_det_max, cfg.warp_cond_max)) {
      continue;
    }
    ++stats.warped;

    // The patch lives at `level`; offsets step by 2^level level-0 pixels. The current
    // patch is read on a regular grid around u0; the warped reference patch is read by
    // mapping each current offset back through A_ref_cur into the reference image.
    const double scale = static_cast<double>(1 << level);
    const Eigen::Matrix2d A_ref_cur = A_cur_ref.inverse();
    if (!A_ref_cur.allFinite()) {
      continue;
    }

    // Margin check: every current-patch tap must stay in bounds at this level.
    if (!img.inBounds(level, u0, kHalf + 1)) {
      continue;
    }

    std::vector<double> ref_warp(kPatchArea, 0.0);
    std::vector<double> cur_I0(kPatchArea, 0.0);
    std::vector<Eigen::Vector2d> cur_grad(kPatchArea, Eigen::Vector2d::Zero());
    bool ref_ok = true;
    for (int yy = 0; yy < kPatch; ++yy) {
      for (int xx = 0; xx < kPatch; ++xx) {
        const int idx = yy * kPatch + xx;
        const Eigen::Vector2d off(static_cast<double>(xx - kHalf), static_cast<double>(yy - kHalf));
        // Current patch sample (level-0 coordinates, offset scaled to this level).
        const Eigen::Vector2d u_cur = u0 + off * scale;
        cur_I0[static_cast<std::size_t>(idx)] = img.intensity(level, u_cur);
        cur_grad[static_cast<std::size_t>(idx)] = img.gradient(level, u_cur);
        // Warp the same current offset back into the reference image and sample the
        // stored reference patch by bilinear lookup inside the kPatch x kPatch grid.
        const Eigen::Vector2d off_ref = A_ref_cur * off;
        const double rx = off_ref.x() + static_cast<double>(kHalf);
        const double ry = off_ref.y() + static_cast<double>(kHalf);
        if (rx < 0.0 || ry < 0.0 || rx > static_cast<double>(kPatch - 1) ||
            ry > static_cast<double>(kPatch - 1)) {
          ref_ok = false;
          break;
        }
        const int rx0 = static_cast<int>(std::floor(rx));
        const int ry0 = static_cast<int>(std::floor(ry));
        const int rx1 = std::min(rx0 + 1, kPatch - 1);
        const int ry1 = std::min(ry0 + 1, kPatch - 1);
        const double ax = rx - rx0;
        const double ay = ry - ry0;
        // The map stores patches as patch(x_index, y_index), so the first index is
        // the column (x) offset and the second the row (y) offset.
        const auto& P = pt->ref_patches[static_cast<std::size_t>(level)];
        const double v00 = static_cast<double>(P(rx0, ry0));
        const double v01 = static_cast<double>(P(rx1, ry0));
        const double v10 = static_cast<double>(P(rx0, ry1));
        const double v11 = static_cast<double>(P(rx1, ry1));
        const double top = v00 * (1.0 - ax) + v01 * ax;
        const double bot = v10 * (1.0 - ax) + v11 * ax;
        ref_warp[static_cast<std::size_t>(idx)] = top * (1.0 - ay) + bot * ay;
      }
      if (!ref_ok) {
        break;
      }
    }
    if (!ref_ok) {
      continue;
    }

    // Gate 4: NCC against the warped reference patch.
    const double ncc = patchNCC(ref_warp, cur_I0);
    if (ncc < cfg.ncc_thre) {
      continue;
    }

    // Gate 5: SSD outlier rejection on the exposure-compensated patch.
    double ssd = 0.0;
    for (int i = 0; i < kPatchArea; ++i) {
      const double r = tau_cur * cur_I0[static_cast<std::size_t>(i)] -
                       pt->inv_expo_ref * ref_warp[static_cast<std::size_t>(i)];
      ssd += r * r;
    }
    if (ssd > cfg.outlier_threshold * static_cast<double>(kPatchArea)) {
      continue;
    }

    // Build the residual. The projection Jacobian is taken at the linearization
    // point; the reference patch is folded with the reference exposure into ref_warp.
    const Eigen::Matrix<double, 2, 3> Jpi = cam.projectJacobian(p_c0);

    auto* functor =
        new VisualPatchResidual(T_fe_cam.q, T_fe_cam.t, pt->p_world, Jpi, seg.u, 1.0 / seg.dt_s,
                                pt->inv_expo_ref, weight, ref_warp, cur_I0, cur_grad);
    functor->p_c0_ = p_c0;

    auto* cost = new ceres::DynamicAutoDiffCostFunction<VisualPatchResidual, 4>(functor);
    for (int i = 0; i < 2 * kSplineOrder; ++i) {
      cost->AddParameterBlock(i < kSplineOrder ? 4 : 3);
    }
    cost->AddParameterBlock(1);  // tau_cur
    cost->SetNumResiduals(kPatchArea);

    std::vector<double*> blocks;
    blocks.reserve(2 * kSplineOrder + 1);
    for (double* p : seg.so3_knots) {
      blocks.push_back(p);
    }
    for (double* p : seg.r3_knots) {
      blocks.push_back(p);
    }
    blocks.push_back(expo.block(expo_index));

    auto* loss = new ceres::HuberLoss(huber);
    problem.AddResidualBlock(cost, loss, blocks);
    ++stats.accepted;
    // ssd already holds sum_i (tau_cur I_cur - inv_expo_ref I_ref)^2 over the patch;
    // its per-pixel RMS is this patch's photometric residual magnitude.
    const double rms = std::sqrt(ssd / static_cast<double>(kPatchArea));
    stats.res_sum += rms;
    if (used != nullptr) {
      VisualUsedPoint u;
      u.uv = u0;
      u.depth = static_cast<float>(p_c0.z());
      u.level = level;
      u.residual = static_cast<float>(rms);
      used->push_back(u);
    }
  }
  return stats;
}

}  // namespace meridian::ct
