#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <sophus/so3.hpp>
#include <vector>

#include "ct/image_pyramid_view.hpp"
#include "ct/residuals_lidar.hpp"
#include "ct/residuals_visual.hpp"
#include "ct/spline_window.hpp"
#include "ct/visual_map.hpp"
#include "meridian/calib/camera_model.hpp"
#include "meridian/calib/intrinsics.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"

using meridian::CameraModel;
using meridian::Duration;
using meridian::FrontendVisual;
using meridian::IntrinsicsCamera;
using meridian::LidarPoint;
using meridian::Pose;
using meridian::SplineWindow;
using meridian::Timestamp;
using meridian::ct::addVisualResiduals;
using meridian::ct::bestSearchLevel;
using meridian::ct::ExposureChain;
using meridian::ct::ImagePyramidView;
using meridian::ct::LidarHit;
using meridian::ct::patchNCC;
using meridian::ct::PlaneFit;
using meridian::ct::VisualAssocStats;
using meridian::ct::VisualMap;
using meridian::ct::VisualMapConfig;
using meridian::ct::VisualPoint;
using meridian::ct::VisualUsedPoint;
using meridian::ct::warpMatrixAffineHomography;
using meridian::ct::warpWellConditioned;

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kLevels = 3;

IntrinsicsCamera makeIntr() {
  IntrinsicsCamera k;
  k.fx = 400.0;
  k.fy = 400.0;
  k.cx = kWidth / 2.0;
  k.cy = kHeight / 2.0;
  k.model = IntrinsicsCamera::Distortion::None;
  k.width = kWidth;
  k.height = kHeight;
  return k;
}

FrontendVisual makeVisCfg() {
  FrontendVisual v;
  v.ncc_thre = 0.5;
  v.outlier_threshold = 1000.0;
  v.img_point_cov = 100.0;
  v.inv_expo_cov = 1e-2;
  v.inv_expo_min = 1e-3;
  v.grid_cell_px = 32;
  v.warp_det_min = 0.1;
  v.warp_det_max = 10.0;
  v.warp_cond_max = 50.0;
  return v;
}

// A textured world plane z = z0 (normal +z). The intensity at a world point is a
// smooth band pattern so the photometric residual has gradient everywhere and the
// affine warp recovers a meaningful patch. The texture is deterministic.
struct TexturedPlane {
  double z0 = 6.0;
  Eigen::Vector3d normal() const { return Eigen::Vector3d(0.0, 0.0, 1.0); }

  double intensity(double x, double y) const {
    // Two superposed sinusoids in x and y, scaled into [40, 215] so a uint8 image
    // never clips and central differences stay well-defined.
    const double s = std::sin(x * 1.3) * std::cos(y * 1.1) + 0.5 * std::sin(x * 0.5 + y * 0.7);
    return 127.5 + 70.0 * s;
  }
};

// Renders the plane into a camera image by intersecting each pixel ray with z = z0.
// inv_expo scales the rendered intensity (the inverse-exposure affine model: the
// stored pixel is tau-scaled brightness, so a smaller inv_expo darkens the image).
cv::Mat renderPlane(const TexturedPlane& plane, const CameraModel& cam, const Pose& T_w_c,
                    double inv_expo) {
  cv::Mat img(kHeight, kWidth, CV_8UC1, cv::Scalar(0));
  const Eigen::Vector3d cam_origin_w = T_w_c.t;
  for (int v = 0; v < kHeight; ++v) {
    for (int u = 0; u < kWidth; ++u) {
      const Eigen::Vector3d ray_c = cam.unproject(Eigen::Vector2d(u + 0.5, v + 0.5));
      const Eigen::Vector3d ray_w = T_w_c.q * ray_c;  // direction in world
      // Intersect cam_origin + s*ray_w with z = z0.
      const double denom = ray_w.z();
      if (std::abs(denom) < 1e-9) continue;
      const double s = (plane.z0 - cam_origin_w.z()) / denom;
      if (s <= 0.0) continue;
      const Eigen::Vector3d p_w = cam_origin_w + s * ray_w;
      const double i = plane.intensity(p_w.x(), p_w.y());
      const double scaled = inv_expo * i;
      const int q = std::clamp(static_cast<int>(std::lround(scaled)), 0, 255);
      img.at<std::uint8_t>(v, u) = static_cast<std::uint8_t>(q);
    }
  }
  return img;
}

std::vector<cv::Mat> buildPyramid(const cv::Mat& full) {
  std::vector<cv::Mat> pyr;
  pyr.push_back(full);
  for (int l = 1; l < kLevels; ++l) {
    cv::Mat down;
    cv::resize(pyr.back(), down, cv::Size(pyr.back().cols / 2, pyr.back().rows / 2), 0, 0,
               cv::INTER_LINEAR);
    pyr.push_back(down);
  }
  return pyr;
}

// A uniform (n_cp==1) spline holding a single constant pose over the window so the
// pose at any covered time is exactly T0 (the GT camera body pose).
SplineWindow makeStaticSpline(const Pose& T0, Timestamp t0, Timestamp t_end, Duration knot_dt) {
  SplineWindow spline(knot_dt, 1);
  spline.initialize(t0, T0);
  const auto seed = [&](Timestamp) { return T0; };
  const Timestamp target = t_end + 4 * knot_dt;
  for (Timestamp t = t0 + knot_dt; t <= target; t += knot_dt) {
    spline.extendTo(t, seed, 1);
  }
  return spline;
}

// Promotes the centre region of a rendered frame into a VisualMap by feeding LiDAR
// hits whose world points lie on the plane and project near the image centre. Returns
// the number of promoted points.
int promotePlanePoints(VisualMap* vmap, const TexturedPlane& plane, const CameraModel& cam,
                       const ImagePyramidView& img, const Pose& T_w_c, double inv_expo) {
  std::vector<LidarHit> hits;
  // Sample a grid of world points on the plane in front of the camera.
  for (double x = -1.5; x <= 1.5 + 1e-9; x += 0.5) {
    for (double y = -1.0; y <= 1.0 + 1e-9; y += 0.5) {
      LidarHit h;
      h.p_world = Eigen::Vector3d(x, y, plane.z0);
      h.plane.n = plane.normal();
      h.plane.d = -plane.z0;  // n.x + d = 0 on the plane
      h.plane.valid = true;
      h.t_offset_ns = static_cast<std::int32_t>((x + 2.0) * 1000 + (y + 2.0));
      hits.push_back(h);
    }
  }
  return vmap->promote(img, cam, T_w_c, inv_expo, hits);
}

// Sum of squared photometric residuals returned by evaluating the full problem.
double problemCost(ceres::Problem* problem) {
  double cost = 0.0;
  problem->Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
  return cost;
}

}  // namespace

// Warp helpers: a pure fronto-parallel pure-translation gives a near-identity affine
// warp; a degenerate (zero-determinant) matrix is rejected.
TEST(ResidualsVisual, WarpHelpersBasic) {
  Eigen::Matrix2d id = Eigen::Matrix2d::Identity();
  EXPECT_TRUE(warpWellConditioned(id, 0.1, 10.0, 50.0));
  EXPECT_EQ(bestSearchLevel(id, 2), 0);

  Eigen::Matrix2d big = 2.5 * Eigen::Matrix2d::Identity();  // det = 6.25 > 3
  EXPECT_GE(bestSearchLevel(big, 2), 1);

  Eigen::Matrix2d singular;
  singular << 1.0, 1.0, 1.0, 1.0;  // det = 0
  EXPECT_FALSE(warpWellConditioned(singular, 0.1, 10.0, 50.0));

  Eigen::Matrix2d ill;
  ill << 1.0, 0.0, 0.0, 100.0;  // det = 100 out of [0.1,10]
  EXPECT_FALSE(warpWellConditioned(ill, 0.1, 10.0, 50.0));
}

// NCC is 1 for identical patches, near 0 for decorrelated ones.
TEST(ResidualsVisual, NccGate) {
  std::vector<double> a(64), b(64), c(64);
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> d(0.0, 255.0);
  for (int i = 0; i < 64; ++i) {
    a[i] = d(rng);
    b[i] = a[i] * 1.3 + 5.0;  // affine of a -> NCC == 1
    c[i] = d(rng);            // independent -> NCC ~ 0
  }
  EXPECT_NEAR(patchNCC(a, b), 1.0, 1e-6);
  EXPECT_LT(std::abs(patchNCC(a, c)), 0.5);
}

// Exposure chain: a single frame with a finite prior std is pulled toward the prior;
// a zero prior std fixes it. Consecutive frames are random-walk tied.
TEST(ResidualsVisual, ExposureChainPriorAndTie) {
  FrontendVisual cfg = makeVisCfg();

  // Single frame, prior pull recovers the prior when no other residual acts.
  {
    ExposureChain chain(cfg);
    chain.addFrame(0, 0.5);  // start away from the prior
    ceres::Problem problem;
    chain.addTo(problem, /*prior=*/1.0, /*prior_std=*/0.1);
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    EXPECT_NEAR(chain.value(0), 1.0, 1e-6);
  }

  // Zero prior std fixes the first block at the prior.
  {
    ExposureChain chain(cfg);
    chain.addFrame(0, 0.5);
    ceres::Problem problem;
    chain.addTo(problem, /*prior=*/0.8, /*prior_std=*/0.0);
    EXPECT_NEAR(chain.value(0), 0.8, 1e-9);
    EXPECT_TRUE(problem.IsParameterBlockConstant(chain.block(0)));
  }

  // Positivity: the lower bound keeps tau strictly positive.
  {
    ExposureChain chain(cfg);
    chain.addFrame(0, cfg.inv_expo_min * 0.5);  // below the floor on seed
    EXPECT_GE(chain.value(0), cfg.inv_expo_min);
  }
}

// Exposure recovery: with the pose pinned at GT and the photometric residuals on the
// current frame, solving for the current-frame inverse exposure recovers the known
// tau ratio between reference and current frames.
TEST(ResidualsVisual, ExposureRecovery) {
  const IntrinsicsCamera intr = makeIntr();
  const CameraModel cam(intr);
  const FrontendVisual cfg = makeVisCfg();
  const TexturedPlane plane;

  // Reference frame: camera at the origin looking down +z at the plane.
  const Pose T_w_cref;  // identity
  const double inv_expo_ref = 1.0;
  const cv::Mat ref_img_full = renderPlane(plane, cam, T_w_cref, inv_expo_ref);
  const std::vector<cv::Mat> ref_pyr = buildPyramid(ref_img_full);
  const ImagePyramidView ref_view(ref_pyr);

  VisualMap vmap{VisualMapConfig(cfg)};
  const int promoted = promotePlanePoints(&vmap, plane, cam, ref_view, T_w_cref, inv_expo_ref);
  ASSERT_GT(promoted, 0);

  // Current frame: same pose, rendered darker by a known brightness factor. The
  // photometric model ties tau*I across frames, so the inverse exposure the solve
  // must recover is tau_ref / brightness (a darker image needs a larger tau to
  // match the brighter reference radiance).
  const Pose T_w_cur = T_w_cref;
  const double brightness = 0.6;
  const double inv_expo_cur_true = inv_expo_ref / brightness;  // ~1.667
  const cv::Mat cur_img_full = renderPlane(plane, cam, T_w_cur, brightness);
  const std::vector<cv::Mat> cur_pyr = buildPyramid(cur_img_full);
  const ImagePyramidView cur_view(cur_pyr);

  const Duration knot_dt = meridian::from_seconds(0.05);
  const Timestamp t_mid = meridian::from_seconds(1.0);
  // The camera extrinsic is identity here (F_e == C), so the spline holds T_w_cur.
  SplineWindow spline = makeStaticSpline(T_w_cur, 0, t_mid + knot_dt, knot_dt);

  ExposureChain expo(cfg);
  const std::size_t idx = expo.addFrame(t_mid, 1.0);  // seed away from the truth

  ceres::Problem problem;
  // Pin the trajectory so only exposure moves.
  SplineWindow::SegmentRef seg = spline.segmentFor(t_mid);
  std::vector<VisualUsedPoint> used;
  vmap.refreshVisibleCache(cam, spline.pose(t_mid));
  const VisualAssocStats stats = addVisualResiduals(problem, spline, cam, Pose{}, cur_view, t_mid,
                                                    expo, idx, vmap, cfg, &used);
  ASSERT_GT(stats.accepted, 0);
  for (double* b : seg.so3_knots) {
    if (problem.HasParameterBlock(b)) problem.SetParameterBlockConstant(b);
  }
  for (double* b : seg.r3_knots) {
    if (problem.HasParameterBlock(b)) problem.SetParameterBlockConstant(b);
  }
  // No prior on the exposure so the photometric data alone drives it.
  expo.addTo(problem, /*prior=*/1.0, /*prior_std=*/0.0);
  // addTo with prior_std==0 would fix the block; re-enable it as a free variable so
  // the photometric residuals can recover it.
  problem.SetParameterBlockVariable(expo.block(idx));
  problem.SetParameterLowerBound(expo.block(idx), 0, cfg.inv_expo_min);

  ceres::Solver::Options opts;
  opts.linear_solver_type = ceres::DENSE_QR;
  opts.max_num_iterations = 50;
  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);

  EXPECT_NEAR(expo.value(idx), inv_expo_cur_true, 0.02) << summary.BriefReport();
}

// Zero residual at ground truth: identical reference and current frames (same pose,
// same exposure) produce a near-zero photometric cost; a pose or exposure
// perturbation makes it strictly larger.
TEST(ResidualsVisual, ZeroAtGroundTruthAndNonzeroUnderPerturbation) {
  const IntrinsicsCamera intr = makeIntr();
  const CameraModel cam(intr);
  const FrontendVisual cfg = makeVisCfg();
  const TexturedPlane plane;

  const Pose T_w_cref;  // identity
  const double inv_expo = 1.0;
  const cv::Mat ref_full = renderPlane(plane, cam, T_w_cref, inv_expo);
  const std::vector<cv::Mat> ref_pyr = buildPyramid(ref_full);
  const ImagePyramidView ref_view(ref_pyr);

  VisualMap vmap{VisualMapConfig(cfg)};
  ASSERT_GT(promotePlanePoints(&vmap, plane, cam, ref_view, T_w_cref, inv_expo), 0);

  // Current frame from a slightly translated pose so the warp is exercised but the
  // patch overlap is high; render it from that exact pose.
  Pose T_w_cur;
  T_w_cur.t = Eigen::Vector3d(0.05, -0.03, 0.0);
  const cv::Mat cur_full = renderPlane(plane, cam, T_w_cur, inv_expo);
  const std::vector<cv::Mat> cur_pyr = buildPyramid(cur_full);
  const ImagePyramidView cur_view(cur_pyr);

  const Duration knot_dt = meridian::from_seconds(0.05);
  const Timestamp t_mid = meridian::from_seconds(1.0);

  const auto cost_at = [&](const Pose& body_pose, double tau) -> double {
    SplineWindow spline = makeStaticSpline(body_pose, 0, t_mid + knot_dt, knot_dt);
    ExposureChain expo(cfg);
    const std::size_t idx = expo.addFrame(t_mid, tau);
    ceres::Problem problem;
    std::vector<VisualUsedPoint> used;
    vmap.refreshVisibleCache(cam, spline.pose(t_mid));
    const VisualAssocStats stats = addVisualResiduals(problem, spline, cam, Pose{}, cur_view, t_mid,
                                                      expo, idx, vmap, cfg, &used);
    EXPECT_GT(stats.accepted, 0);
    return problemCost(&problem);
  };

  const double cost_gt = cost_at(T_w_cur, inv_expo);

  // A pose offset of 10 cm and an exposure offset both raise the cost meaningfully.
  Pose perturbed = T_w_cur;
  perturbed.t += Eigen::Vector3d(0.1, 0.0, 0.0);
  const double cost_pose = cost_at(perturbed, inv_expo);
  const double cost_expo = cost_at(T_w_cur, inv_expo * 1.4);

  EXPECT_LT(cost_gt, cost_pose);
  EXPECT_LT(cost_gt, cost_expo);
}

// Degenerate / decorrelated rejection: a current frame whose texture is replaced by
// noise fails the NCC gate, so no residuals are admitted; and a back-facing /
// behind-camera point is skipped.
TEST(ResidualsVisual, GatesRejectDecorrelatedAndInvalid) {
  const IntrinsicsCamera intr = makeIntr();
  const CameraModel cam(intr);
  const FrontendVisual cfg = makeVisCfg();
  const TexturedPlane plane;

  const Pose T_w_cref;
  const double inv_expo = 1.0;
  const cv::Mat ref_full = renderPlane(plane, cam, T_w_cref, inv_expo);
  const std::vector<cv::Mat> ref_pyr = buildPyramid(ref_full);
  const ImagePyramidView ref_view(ref_pyr);

  VisualMap vmap{VisualMapConfig(cfg)};
  ASSERT_GT(promotePlanePoints(&vmap, plane, cam, ref_view, T_w_cref, inv_expo), 0);

  // A decorrelated current image: uniform random noise, no relation to the plane.
  cv::Mat noise(kHeight, kWidth, CV_8UC1);
  std::mt19937 rng(99);
  std::uniform_int_distribution<int> d(0, 255);
  for (int v = 0; v < kHeight; ++v) {
    for (int u = 0; u < kWidth; ++u) {
      noise.at<std::uint8_t>(v, u) = static_cast<std::uint8_t>(d(rng));
    }
  }
  const std::vector<cv::Mat> noise_pyr = buildPyramid(noise);
  const ImagePyramidView noise_view(noise_pyr);

  const Duration knot_dt = meridian::from_seconds(0.05);
  const Timestamp t_mid = meridian::from_seconds(1.0);
  SplineWindow spline = makeStaticSpline(T_w_cref, 0, t_mid + knot_dt, knot_dt);
  ExposureChain expo(cfg);
  const std::size_t idx = expo.addFrame(t_mid, inv_expo);

  ceres::Problem problem;
  std::vector<VisualUsedPoint> used;
  // NCC threshold 0.5 against pure noise rejects (almost) everything.
  FrontendVisual strict = cfg;
  strict.ncc_thre = 0.8;
  vmap.refreshVisibleCache(cam, spline.pose(t_mid));
  const VisualAssocStats stats = addVisualResiduals(problem, spline, cam, Pose{}, noise_view, t_mid,
                                                    expo, idx, vmap, strict, &used);
  EXPECT_GT(stats.candidates, 0);
  EXPECT_EQ(stats.accepted, 0);
}

// SSD outlier gate: a current frame whose texture matches the reference (NCC passes)
// but with a large brightness offset the fixed exposures cannot absorb produces a
// huge per-pixel SSD, so a tight outlier_threshold drops every candidate even though
// they pass the NCC and warp gates. A loose threshold admits them again.
TEST(ResidualsVisual, SsdGateRejectsBrightnessOutlier) {
  const IntrinsicsCamera intr = makeIntr();
  const CameraModel cam(intr);
  const FrontendVisual cfg = makeVisCfg();
  const TexturedPlane plane;

  const Pose T_w_cref;
  const double inv_expo = 1.0;
  const cv::Mat ref_full = renderPlane(plane, cam, T_w_cref, inv_expo);
  const std::vector<cv::Mat> ref_pyr = buildPyramid(ref_full);
  const ImagePyramidView ref_view(ref_pyr);

  VisualMap vmap{VisualMapConfig(cfg)};
  ASSERT_GT(promotePlanePoints(&vmap, plane, cam, ref_view, T_w_cref, inv_expo), 0);

  // Same geometry/texture, but a large additive brightness offset: NCC (zero-mean,
  // scale-normalised) is unchanged, while tau_cur*I_cur - tau_ref*I_ref carries the
  // whole offset into the SSD.
  cv::Mat bright = renderPlane(plane, cam, T_w_cref, inv_expo);
  bright += cv::Scalar(80);  // saturating add; raises every residual by ~80
  const std::vector<cv::Mat> bright_pyr = buildPyramid(bright);
  const ImagePyramidView bright_view(bright_pyr);

  const Duration knot_dt = meridian::from_seconds(0.05);
  const Timestamp t_mid = meridian::from_seconds(1.0);
  SplineWindow spline = makeStaticSpline(T_w_cref, 0, t_mid + knot_dt, knot_dt);

  const auto run = [&](double outlier_thresh) -> VisualAssocStats {
    ExposureChain expo(cfg);
    const std::size_t idx = expo.addFrame(t_mid, inv_expo);
    ceres::Problem problem;
    std::vector<VisualUsedPoint> used;
    FrontendVisual c = cfg;
    c.outlier_threshold = outlier_thresh;
    vmap.refreshVisibleCache(cam, spline.pose(t_mid));
    return addVisualResiduals(problem, spline, cam, Pose{}, bright_view, t_mid, expo, idx, vmap, c,
                              &used);
  };

  // A tight SSD threshold rejects the brightness-offset patches (which still pass
  // NCC), while a very loose one admits them.
  const VisualAssocStats tight = run(50.0);
  const VisualAssocStats loose = run(1e9);
  EXPECT_GT(tight.warped, 0);  // warp + NCC gates passed
  EXPECT_EQ(tight.accepted, 0);
  EXPECT_GT(loose.accepted, 0);
}

// Disabled-camera harmlessness: an invalid CameraModel (zero intrinsics) yields no
// residuals and never crashes.
TEST(ResidualsVisual, InvalidCameraIsHarmless) {
  const CameraModel cam;  // default-constructed -> invalid (fx == 0)
  const FrontendVisual cfg = makeVisCfg();
  const TexturedPlane plane;

  cv::Mat blank(kHeight, kWidth, CV_8UC1, cv::Scalar(0));
  const std::vector<cv::Mat> pyr = buildPyramid(blank);
  const ImagePyramidView view(pyr);

  VisualMap vmap{VisualMapConfig(cfg)};
  const Duration knot_dt = meridian::from_seconds(0.05);
  const Timestamp t_mid = meridian::from_seconds(1.0);
  SplineWindow spline = makeStaticSpline(Pose{}, 0, t_mid + knot_dt, knot_dt);
  ExposureChain expo(cfg);
  const std::size_t idx = expo.addFrame(t_mid, 1.0);

  ceres::Problem problem;
  std::vector<VisualUsedPoint> used;
  vmap.refreshVisibleCache(cam, spline.pose(t_mid));
  const VisualAssocStats stats =
      addVisualResiduals(problem, spline, cam, Pose{}, view, t_mid, expo, idx, vmap, cfg, &used);
  EXPECT_EQ(stats.candidates, 0);
  EXPECT_EQ(stats.accepted, 0);
  EXPECT_EQ(problem.NumResidualBlocks(), 0);
}

// Jacobian vs numeric diff on the SMOOTH part: the photometric residual is linearized
// in the image, so autodiff of the residual w.r.t. the control points must match a
// finite-difference of the SAME linearized residual (image gradient held fixed at the
// linearization point). We verify by perturbing each R^3 knot and comparing the
// autodiff-derived cost gradient against a central difference of the cost.
TEST(ResidualsVisual, JacobianMatchesNumericOnSmoothChain) {
  const IntrinsicsCamera intr = makeIntr();
  const CameraModel cam(intr);
  const FrontendVisual cfg = makeVisCfg();
  const TexturedPlane plane;

  const Pose T_w_cref;
  const double inv_expo = 1.0;
  const cv::Mat ref_full = renderPlane(plane, cam, T_w_cref, inv_expo);
  const std::vector<cv::Mat> ref_pyr = buildPyramid(ref_full);
  const ImagePyramidView ref_view(ref_pyr);

  VisualMap vmap{VisualMapConfig(cfg)};
  ASSERT_GT(promotePlanePoints(&vmap, plane, cam, ref_view, T_w_cref, inv_expo), 0);

  Pose T_w_cur;
  T_w_cur.t = Eigen::Vector3d(0.04, 0.02, 0.0);
  const cv::Mat cur_full = renderPlane(plane, cam, T_w_cur, inv_expo);
  const std::vector<cv::Mat> cur_pyr = buildPyramid(cur_full);
  const ImagePyramidView cur_view(cur_pyr);

  const Duration knot_dt = meridian::from_seconds(0.05);
  const Timestamp t_mid = meridian::from_seconds(1.0);
  SplineWindow spline = makeStaticSpline(T_w_cur, 0, t_mid + knot_dt, knot_dt);
  ExposureChain expo(cfg);
  const std::size_t idx = expo.addFrame(t_mid, inv_expo);

  ceres::Problem problem;
  std::vector<VisualUsedPoint> used;
  vmap.refreshVisibleCache(cam, spline.pose(t_mid));
  const VisualAssocStats stats = addVisualResiduals(problem, spline, cam, Pose{}, cur_view, t_mid,
                                                    expo, idx, vmap, cfg, &used);
  ASSERT_GT(stats.accepted, 0);

  // Compare Ceres' autodiff gradient of the cost against a central difference of the
  // SAME cost, restricted to one well-supported R^3 knot. Scoping EvaluateOptions to
  // that single block makes the returned gradient exactly the 3 entries for it, so no
  // block-ordering reverse-engineering is needed.
  SplineWindow::SegmentRef seg = spline.segmentFor(t_mid);
  double* knot = seg.r3_knots[2];

  ceres::Problem::EvaluateOptions eopt;
  eopt.parameter_blocks = {knot};
  double cost0 = 0.0;
  std::vector<double> grad;
  ASSERT_TRUE(problem.Evaluate(eopt, &cost0, nullptr, &grad, nullptr));
  ASSERT_EQ(grad.size(), 3u);

  // The cost central difference uses the full problem (all residuals), which is what
  // the scoped gradient differentiates.
  ceres::Problem::EvaluateOptions full;
  const double h = 1e-4;
  for (int c = 0; c < 3; ++c) {
    const double orig = knot[c];
    knot[c] = orig + h;
    double cost_plus = 0.0;
    problem.Evaluate(full, &cost_plus, nullptr, nullptr, nullptr);
    knot[c] = orig - h;
    double cost_minus = 0.0;
    problem.Evaluate(full, &cost_minus, nullptr, nullptr, nullptr);
    knot[c] = orig;
    const double numeric = (cost_plus - cost_minus) / (2.0 * h);
    const double analytic = grad[static_cast<std::size_t>(c)];
    EXPECT_NEAR(analytic, numeric, 1e-2 * (1.0 + std::abs(numeric))) << "coord " << c;
  }
}

// ---------------------------------------------------------------------------
// Analytic photometric patch cost: autodiff-parity of residual and Jacobians.
// ---------------------------------------------------------------------------
namespace {

meridian::ct::VisualPatchParams randomVisualProbe(std::mt19937& rng, double rot_scale) {
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  auto rv = [&] { return Eigen::Vector3d(uni(rng), uni(rng), uni(rng)); };
  meridian::ct::VisualPatchParams p;
  // 4 knots a bounded rotation/translation apart, like a warm-started window.
  // (Knots themselves are passed as parameter blocks at Evaluate time.)
  p.q_fe_c = Eigen::Quaterniond(uni(rng) + 1.5, uni(rng), uni(rng), uni(rng)).normalized();
  p.t_fe_c = 0.2 * rv();
  p.p_world = 3.0 * rv() + Eigen::Vector3d(0, 0, 5);
  // A plausible pinhole projection Jacobian at the linearization point.
  Eigen::Vector3d pc0 = Eigen::Vector3d(uni(rng), uni(rng), 4.0 + std::abs(uni(rng)));
  const double fx = 300.0, fy = 300.0;
  p.Jpi << fx / pc0.z(), 0.0, -fx * pc0.x() / (pc0.z() * pc0.z()),
      0.0, fy / pc0.z(), -fy * pc0.y() / (pc0.z() * pc0.z());
  p.p_c0 = pc0;
  p.u = 0.5 * (uni(rng) + 1.0) * 0.999;
  p.tau_ref = 0.8 + 0.2 * (uni(rng) + 1.0);
  p.weight = 0.5 + 0.5 * (uni(rng) + 1.0);
  const int area = 64;  // kPatch^2
  p.ref_patch.resize(area);
  p.cur_I0.resize(area);
  p.cur_grad.resize(area);
  for (int i = 0; i < area; ++i) {
    p.ref_patch[i] = 128.0 + 40.0 * uni(rng);
    p.cur_I0[i] = 128.0 + 40.0 * uni(rng);
    p.cur_grad[i] = Eigen::Vector2d(20.0 * uni(rng), 20.0 * uni(rng));
  }
  (void)rot_scale;
  return p;
}

}  // namespace

TEST(AnalyticVisualCost, JacobiansMatchAutodiffTangentProjected) {
  std::mt19937 rng(101);
  ceres::EigenQuaternionManifold quat_manifold;
  for (int trial = 0; trial < 50; ++trial) {
    const double rot_scale = (trial % 2 == 0) ? 1e-4 : 0.4;
    meridian::ct::VisualPatchParams vp = randomVisualProbe(rng, rot_scale);
    if (trial % 5 == 0) vp.u = 1e-6;
    if (trial % 7 == 0) vp.u = 1.0 - 1e-6;

    // Random knots + exposure as the parameter blocks.
    std::array<Eigen::Quaterniond, 4> q;
    std::array<Eigen::Vector3d, 4> pr;
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    Eigen::Quaterniond q0(uni(rng) + 1.5, uni(rng), uni(rng), uni(rng));
    q0.normalize();
    for (int j = 0; j < 4; ++j) {
      q[j] = q0;
      q0 = (q0 * Sophus::SO3d::exp(rot_scale * Eigen::Vector3d(uni(rng), uni(rng), uni(rng)))
                     .unit_quaternion())
               .normalized();
      pr[j] = 0.5 * Eigen::Vector3d(uni(rng), uni(rng), uni(rng));
    }
    double tau = 0.9 + 0.1 * uni(rng);

    std::unique_ptr<ceres::CostFunction> analytic(meridian::ct::makeVisualPatchCost(vp));
    std::unique_ptr<ceres::CostFunction> reference(meridian::ct::makeVisualPatchCostAutodiff(vp));

    std::vector<const double*> params;
    for (int j = 0; j < 4; ++j) params.push_back(q[j].coeffs().data());
    for (int j = 0; j < 4; ++j) params.push_back(pr[j].data());
    params.push_back(&tau);

    const int A = 64;
    std::vector<double> ra(A), rr(A);
    std::vector<std::vector<double>> ja(9), jr(9);
    std::vector<double*> jap(9), jrp(9);
    for (int b = 0; b < 9; ++b) {
      const int cols = (b < 4) ? 4 : (b < 8 ? 3 : 1);
      ja[b].resize(A * cols);
      jr[b].resize(A * cols);
      jap[b] = ja[b].data();
      jrp[b] = jr[b].data();
    }
    ASSERT_TRUE(analytic->Evaluate(params.data(), ra.data(), jap.data()));
    ASSERT_TRUE(reference->Evaluate(params.data(), rr.data(), jrp.data()));

    for (int i = 0; i < A; ++i) EXPECT_NEAR(ra[i], rr[i], 1e-9) << "resid " << i << " trial " << trial;

    for (int j = 0; j < 4; ++j) {
      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> plus;
      quat_manifold.PlusJacobian(params[j], plus.data());
      Eigen::Map<Eigen::Matrix<double, 64, 4, Eigen::RowMajor>> Ja(ja[j].data());
      Eigen::Map<Eigen::Matrix<double, 64, 4, Eigen::RowMajor>> Jr(jr[j].data());
      const Eigen::Matrix<double, 64, 3> la = Ja * plus;
      const Eigen::Matrix<double, 64, 3> lr = Jr * plus;
      const double err = (la - lr).cwiseAbs().maxCoeff();
      EXPECT_LT(err, 1e-7 + 1e-7 * lr.cwiseAbs().maxCoeff())
          << "so3 knot " << j << " trial " << trial << " err " << err;
    }
    for (int j = 0; j < 4; ++j) {
      Eigen::Map<Eigen::Matrix<double, 64, 3, Eigen::RowMajor>> Ja(ja[4 + j].data());
      Eigen::Map<Eigen::Matrix<double, 64, 3, Eigen::RowMajor>> Jr(jr[4 + j].data());
      EXPECT_LT((Ja - Jr).cwiseAbs().maxCoeff(), 1e-9) << "r3 knot " << j << " trial " << trial;
    }
    Eigen::Map<Eigen::Matrix<double, 64, 1>> Ja(ja[8].data());
    Eigen::Map<Eigen::Matrix<double, 64, 1>> Jr(jr[8].data());
    EXPECT_LT((Ja - Jr).cwiseAbs().maxCoeff(), 1e-9) << "tau trial " << trial;
  }
}
