#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstddef>
#include <vector>

#include "ct/image_pyramid_view.hpp"
#include "ct/spline_window.hpp"
#include "ct/visual_map.hpp"
#include "meridian/calib/camera_model.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"

namespace ceres {
class Problem;
}  // namespace ceres

namespace meridian::ct {

// Per-frame inverse-exposure chain. One scalar inverse-exposure block (tau) is held
// per camera frame in the window. Consecutive blocks are tied by a random-walk
// residual r = tau_k - tau_{k-1} weighted by 1/sqrt(inv_expo_cov * dt), and each
// block is held strictly positive by a Ceres lower bound at inv_expo_min, so a tau
// can never flip the sign of the photometric residual. The first block is anchored
// toward inv_expo_prior with std inv_expo_std (a zero std fixes that block at the
// prior); a block with no photometric evidence inherits its neighbour's exposure
// through the tie rather than floating.
//
// The chain owns the scalar storage so a block pointer stays a stable Ceres
// parameter block for the chain's lifetime. addFrame() appends a frame (returning
// its index); the chain replays the tie + prior residuals into the problem on
// addTo(). Blocks are never freed mid-window; window slide is handled by the caller
// rebuilding the chain for the active frames.
class ExposureChain {
public:
  explicit ExposureChain(const FrontendVisual& cfg);

  // Appends a camera frame at absolute time t, seeded with inv_expo. Returns the new
  // frame's index. The first frame added becomes the prior-anchored reference.
  std::size_t addFrame(Timestamp t, double inv_expo);

  std::size_t size() const { return taus_.size(); }
  bool empty() const { return taus_.empty(); }

  // Mutable scalar storage for frame k (the Ceres parameter block).
  double* block(std::size_t k) { return &taus_[k]; }
  const double* block(std::size_t k) const { return &taus_[k]; }
  double value(std::size_t k) const { return taus_[k]; }
  Timestamp time(std::size_t k) const { return times_[k]; }

  // Registers every frame block as a parameter block with the inv_expo_min lower
  // bound, adds the random-walk ties between consecutive frames (weighted by the
  // real time gap), and the first-frame prior pull when prior_std > 0 (fixing the
  // block when prior_std == 0). Idempotent given a fixed frame set; call once after
  // all frames are added and before the solve. prior / prior_std come from the
  // camera intrinsics (spec 01 §5.1).
  void addTo(ceres::Problem& problem, double prior, double prior_std);

private:
  FrontendVisual cfg_;
  std::vector<double> taus_;
  std::vector<Timestamp> times_;
};

// Outcome of the candidate -> residual association for one frame.
struct VisualAssocStats {
  int candidates = 0;  // returned by VisualMap::visibleCandidates
  int warped = 0;      // passed warp/level + normal-sign + degenerate-warp guards
  int accepted = 0;    // also passed NCC + SSD, residuals added to the problem
  // Sum over accepted patches of the per-patch root-mean-square exposure-compensated
  // photometric residual (intensity units) at the linearization pose. The mean over
  // accepted patches is res_sum / accepted.
  double res_sum = 0.0;
};

// A patch that contributed a photometric residual this frame, captured at residual
// build time so the overlay reflects what actually tracked (no later re-selection):
// the current-frame projection pixel, its LiDAR depth, the chosen pyramid level, and
// the per-patch RMS exposure-compensated photometric residual. Self-contained — holds
// no VisualPoint pointer, so it stays valid after the map is mutated post-solve.
struct VisualUsedPoint {
  Eigen::Vector2d uv = Eigen::Vector2d::Zero();
  float depth = 0.f;
  int level = 0;
  float residual = 0.f;
};

// Adds the sparse-direct photometric residuals for one camera frame. For each
// visible candidate (one-best-per-grid-cell from the map), the reference patch is
// affine-warped into the current view via the LiDAR-plane homography
// H = R_cur_ref * (n.x_ref I - t n^T) (FAST-LIVO2 form; equivalent to R + t n^T/d up
// to projective scale), the best pyramid level is chosen by the warp determinant,
// and the candidate is run through the fixed gate order grid -> depth -> warp/level
// -> NCC -> SSD before any residual is built. Survivors enter a DynamicAutoDiff cost
// over the 4 SO(3) + 4 R^3 knot blocks of t_mid_expo's segment plus the frame's
// inverse-exposure block (the reference exposure enters as a constant), with one
// scalar residual per patch pixel
//   r = tau_cur I_cur(warp(px)) - tau_ref P_ref[px]
// under a Huber loss and the cfg.img_point_cov weight.
//
// Image sampling is not analytically differentiable through the pyramid, so each
// pixel's current intensity is linearized at the projected pixel: I_cur(u) ~=
// I0 + grad_I . (u - u0) with I0, grad_I (level-0 units) fixed data, so the autodiff
// runs only through the smooth spline / projection / exposure chain — the exact
// sparse-direct pattern (image-gradient row x projection Jacobian x pose chain).
//
// `expo_index` selects which ExposureChain frame this image belongs to (its tau is
// the current-frame block). Candidates whose normal is uninitialized, whose warp is
// degenerate, or which fail a gate are skipped (never warped against a guessed
// plane). Accepted candidates are appended to *used when non-null, each carrying its
// projection pixel, depth, chosen level, and per-patch residual for the overlay.
VisualAssocStats addVisualResiduals(ceres::Problem& problem, SplineWindow& spline,
                                    const CameraModel& cam, const Pose& T_fe_cam,
                                    const ImagePyramidView& img, Timestamp t_mid_expo,
                                    ExposureChain& expo, std::size_t expo_index,
                                    const VisualMap& vmap, const FrontendVisual& cfg,
                                    std::vector<VisualUsedPoint>* used);

// --- exposed for unit testing the smooth (autodiff) part in isolation ---

// The 2x2 affine warp A_cur_ref mapping a reference-patch pixel offset to a current-
// frame pixel offset, from the LiDAR-plane homography. `n_ref_cam` is the plane
// normal expressed in the reference camera frame (already sign-oriented toward the
// camera). `xyz_ref_cam` is the point in the reference camera frame. Returns false
// when the projection of any probe point fails (point behind a camera).
bool warpMatrixAffineHomography(const CameraModel& cam, const Eigen::Vector2d& px_ref,
                                const Eigen::Vector3d& xyz_ref_cam,
                                const Eigen::Vector3d& n_ref_cam, const Pose& T_cur_ref,
                                int level_ref, Eigen::Matrix2d* A_cur_ref);

// Picks the pyramid level whose warped patch is closest to unit pixel density:
// climbs while det(A) > 3, capped at max_level. Mirrors FAST-LIVO2 getBestSearchLevel.
int bestSearchLevel(const Eigen::Matrix2d& A_cur_ref, int max_level);

// Degenerate-warp guard: |det A| in [det_min, det_max] and condition number (ratio of
// singular values) <= cond_max. A warp failing this is dropped for the frame.
bool warpWellConditioned(const Eigen::Matrix2d& A, double det_min, double det_max, double cond_max);

// Normalized cross-correlation of two equal-length intensity patches in [-1, 1].
double patchNCC(const std::vector<double>& a, const std::vector<double>& b);

}  // namespace meridian::ct
