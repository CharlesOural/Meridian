# Grounding Dossier — FAST-LIVO2: Sparse-Direct Visual Residual & LiDAR-Inertial-Visual Fusion

> Scope: how FAST-LIVO2 attaches a **direct photometric (patch) residual** to the *same* error-state iterated Kalman filter state used by the LiDAR update, using **LiDAR-provided depth** so no triangulation/feature-matching is ever needed. Grounded in a full read of the cloned source; line numbers are exact against `C:/Users/charl/Sources/slam-reference/FAST-LIVO2/` and were re-read in a stable session. Paper references are to FAST-LIVO2 (arXiv 2408.14035) by section.

---

## 0. The one-paragraph thesis

FAST-LIVO2 maintains **one** state (`StatesGroup`) and updates it **sequentially**: first by the LiDAR measurements (point-to-plane against a voxel-plane map, in `voxel_map.cpp`), then — without re-propagating — by the **image** measurements (photometric patch alignment, in `vio.cpp`). Both are error-state iterated Kalman filter (ESIKF) updates against the same covariance. The visual side never estimates depth from images: every visual map point gets its 3D position from the LiDAR map, so the camera contributes a pure brightness-consistency constraint. This is the structural reason LIVO is robust: in a geometrically degenerate scene the LiDAR block adds ~no information to the unobservable axis, but the photometric block does, and they sum in one Hessian before inversion.

The per-frame visual pipeline is `VIOManager::processFrame` (`src/vio.cpp:1786`):

```
processFrame (vio.cpp:1786)
  ├─ new Frame(cam,img); updateFrameState()          // set camera pose from current state   (:1799-1800)
  ├─ resetGrid()                                                                               (:1802)
  ├─ retrieveFromVisualSparseMap(img, pg, plane_map) // build the visual submap (assoc.)       (:1806)
  ├─ computeJacobianAndUpdateEKF(img)                // the ESIKF photometric update           (:1810)
  ├─ generateVisualMapPoints(img, pg)                // promote new LiDAR pts to visual pts     (:1814)
  ├─ updateVisualMapPoints(img)                      // add observations to existing pts        (:1824)
  └─ updateReferencePatch(plane_map)                 // re-pick best ref patch + normal refine  (:1828)
```

---

## 1. Data structures

### 1.1 `VisualPoint` (`include/visual_point.h:23`, impl `src/visual_point.cpp`)
A 3-D surface point that the camera observes. Key members (`visual_point.h:28-37`):
- `Vector3d pos_` — 3-D position in **world** frame (came from the LiDAR map; never triangulated).
- `Vector3d normal_`, `previous_normal_`, `Matrix3d normal_information_` — surface normal + its information (refined from the voxel-map plane).
- `Eigen::Matrix3d covariance_` — positional covariance (copied from the LiDAR point's `var`).
- `list<Feature*> obs_` — the set of **reference patches** (observations) of this point from past keyframes.
- `Feature* ref_patch; bool has_ref_patch_` — the currently selected best reference patch.
- `bool is_converged_, is_normal_initialized_`.

Observation management: `addFrameRef` (push_front, `visual_point.cpp:34`), `getCloseViewObs` (pick the observation whose viewing ray is most parallel to the current ray; rejects > 60°, `visual_point.cpp:57-95`), `findMinScoreFeature` / `deleteNonRefPatchFeatures` (cap obs list, `:97-127`).

### 1.2 `Feature` (a reference patch) — used throughout `vio.cpp`
Holds: the pixel patch buffer `patch_` (multi-level), pixel coord `px_`, bearing `f_`, the **frame pose at capture** `T_f_w_`, the pyramid `level_`, the per-patch mean `mean_` and score `score_`, the image `img_`, the frame `id_`, and crucially `inv_expo_time_` — the inverse exposure time at capture (for photometric/affine exposure compensation).

### 1.3 `Frame` (`include/frame.h:26`)
`SE3 T_f_w_` (camera-from-world), `T_f_w_prior_` (IMU-prior pose), the grayscale `img_`. Provides the projection helpers used everywhere: `w2c` (world→pixel, `frame.h:49`), `w2f` (world→camera, `:61`), `f2w`, `pos()` (camera centre in world, `:70`). Image pyramid built by `frame_utils::createImgPyramid` (half-sampling, `frame.cpp:54-63`).

### 1.4 The visual map: a **shared 0.5 m voxel hash**
`VIOManager::feat_map : unordered_map<VOXEL_LOCATION, VOXEL_POINTS*>` (`vio.h:126`) keyed at `voxel_size = 0.5 m`. `insertPointIntoVoxelMap` (`vio.cpp:227-250`) hashes a `VisualPoint` by `floor(pos/0.5)`. The same `VOXEL_LOCATION`/octree machinery is the LiDAR `plane_map` (`voxel_map.cpp`) — **one spatial index serves both modalities** (paper §V, "unified voxel map").

### 1.5 `SubSparseMap` — the per-frame visual measurement set (`vio.h:26-57`)
Filled by association, consumed by the EKF. Parallel arrays over the selected points: `voxel_points` (the `VisualPoint*`), `warp_patch` (the warped reference patch per point), `search_levels`, `errors`, `inv_expo_list` (reference inverse-exposure), `add_from_voxel_map`.

---

## 2. Association: `retrieveFromVisualSparseMap` (`vio.cpp:352-782`)

Goal: pick, for each image grid cell, the single best visual map point and prepare its warped reference patch. Steps:

1. **Depth image from the current LiDAR scan** (`:386-428`): project every LiDAR point `pg[i].point_w` into the image, write its depth into `depth_img`. Also tag each occupied 0.5 m voxel in `sub_feat_map`.
2. **Gather in-FOV visual points** (`:440-484`): for each tagged voxel, take its `VisualPoint`s, project (`new_frame_->w2c`), and into a grid of size `grid_size`, keep the **closest** point per cell (`map_dist[index]`, `grid_num[index]=TYPE_MAP`, `retrieve_voxel_points[index]=pt`).
3. **Raycasting fallback** (`:487-591`, `raycast_en`): for empty cells, march precomputed rays (`rays_with_sample_points`, built in `initializeVIO` over depths 0.1–3.0 m step 0.2, `:91-118`) until they hit an occupied visual voxel or a LiDAR plane voxel — pulls in points just entering the FOV.
4. **Depth-continuity gate** (`:618-640`): reject a candidate if any neighbour pixel in the patch footprint has a depth discontinuity > 0.5 m (avoids patches straddling occlusion edges).
5. **Reference-patch + warp** (`:645-742`):
   - choose `ref_ftr`: if `normal_en`, pick the obs with min photometric disagreement to the others (`:663-692`), else `getCloseViewObs` (most parallel view).
   - compute the **affine warp** `A_cur_ref` from reference patch to current view. With a surface normal, use the **homography** form `getWarpMatrixAffineHomography` (`:252-273`): `H = R_cur_ref·(nᵀx·I − t·nᵀ)`. Otherwise the depth-based affine `getWarpMatrixAffine` (`:275-290`). Pick pyramid `search_level` by the warp determinant (`getBestSearchLevel`, `:320-331`: while det>3 go coarser).
   - `warpAffine` (`:292-318`) samples the warped reference patch into `patch_wrap`.
6. **Photometric gate** (`:744-763`): compute current patch (`getImagePatch`, bilinear, `:203-225`), the exposure-compensated SSD error (eq. below), optional **NCC** check (`calculateNCC`, `:333-350`; threshold `ncc_thre`), and reject if `error > outlier_threshold * patch_size_total`.
7. Push survivors into `visual_submap`. `total_points = visual_submap->voxel_points.size()` (`:775`).

---

## 3. The photometric residual (the heart)

For visual point *i*, reference patch pixel *k*, the residual is **exposure-compensated brightness difference** (`vio.cpp:1619-1621`, association-side mirror at `:749-750`):

$$
r_{i,k} \;=\; \underbrace{\tau_{\text{cur}}}_{\text{state->inv\_expo\_time}}\, I_{\text{cur}}\!\big(\pi(p^c_i)+\Delta_k\big)\;-\;\underbrace{\tau_{\text{ref}}}_{\text{inv\_expo\_list}[i]}\, P^{\text{warp}}_{i}[k]
$$

where:
- $p^c_i = R_{cw} p^w_i + P_{cw}$ is the point in the current camera frame, with $R_{cw}=R_{ci}R_{wi}^\top$, $P_{cw}=-R_{ci}R_{wi}^\top P_{wi}+P_{ci}$ (`vio.cpp:1542-1543`); $R_{ci},P_{ci}$ are the composed camera-IMU extrinsic built in `initializeVIO` from $R_{cl},R_{li}$ (`:57-58`).
- $\pi(\cdot)$ = `cam->world2cam` (pinhole projection), $I_{\text{cur}}$ bilinearly sampled.
- $P^{\text{warp}}_i$ = the reference patch affinely warped into the current view.
- $\tau = $ inverse exposure time → **affine photometric (exposure) compensation**; if `exposure_estimate_en`, $\tau_{\text{cur}}$ (`inv_expo_time`) is itself a state variable estimated online (paper §VI-B).

The patch is `patch_size × patch_size` (config; `patch_size_total = patch_size²`), evaluated over a 3-level pyramid, coarse→fine (`computeJacobianAndUpdateEKF`, `:790-799`).

---

## 4. The Jacobian (derived in `updateState`, `vio.cpp:1520-1688`)

The measurement Jacobian of one patch pixel w.r.t. the error state is built by the chain rule (`:1611-1617`). With image gradient $\nabla I=[d_u,d_v]$ (central differences, `:1600-1609`), projection Jacobian $J_{\pi}=\partial \pi/\partial p^c$ (2×3, `computeProjectionJacobian`, `:189-201`):

$$
J_{\text{img}} = \tau_{\text{cur}}\,\tfrac{1}{2^{\text{level}}}\,[d_u,\;d_v]\quad(1\times2)
$$
$$
J_{\delta\phi} = J_{\text{img}}\,J_{\pi}\,\lfloor p^c\rfloor_\times,\qquad
J_{\delta p} = -\,J_{\text{img}}\,J_{\pi}
$$
$$
\boxed{\;J_R = J_{\delta\phi}\,\frac{\partial(\delta\phi)}{\partial(\delta\theta_{wi})} + J_{\delta p}\,\frac{\partial p^c}{\partial(\delta\theta_{wi})},\qquad
J_t = J_{\delta p}\,\frac{\partial p^c}{\partial(\delta p_{wi})}\;}
$$

In code (`:1614-1617`): `Jdphi = Jimg*Jdpi*p_hat`, `Jdp = -Jimg*Jdpi`, `JdR = Jdphi*Jdphi_dR + Jdp*Jdp_dR`, `Jdt = Jdp*Jdp_dt`. The constant blocks `Jdphi_dR=Rci`, `Jdp_dR=-Rci·⌊Pic⌋` are precomputed in `initializeVIO` (`:62-65`); `Jdp_dt=Rci·Rwi^T` is per-iteration (`:1544`). Each pixel contributes one row of `H_sub` (1×6, or 1×7 if exposure is estimated — the 7th column is `cur_value`, `:1628-1629`).

---

## 5. The sequential ESIKF update (`updateState`, `vio.cpp:1648-1675`)

Stacking all pixel rows gives `H_sub` (`H_DIM × {6,7}`, `H_DIM = total_points·patch_size_total`) and residual `z`. The update is the **iterated** error-state KF in the compact form FAST-LIO uses (note: only the top-left 7×7 of the state Hessian is touched because the photometric measurement only sees pose + exposure):

```cpp
H_T_H.block<7,7>(0,0) = H_subᵀ H_sub;                                   // :1660
K_1 = ( H_T_H + (state->cov / img_point_cov).inverse() ).inverse();     // :1661   (= (HᵀR⁻¹H + P⁻¹)⁻¹ )
vec = state_propagat − state;                                            // prior error (boxminus)   :1664
G.block<DIM,7>(0,0) = K_1.block<DIM,7>(0,0) * H_T_H.block<7,7>(0,0);     // :1665
solution = −K_1.block<DIM,7>(0,0)·(H_subᵀ z) + vec − G·vec.block<7,1>;   // :1667
state += solution;                                                       // boxplus update           :1669
```
- `img_point_cov` is the scalar photometric measurement variance $R$ (per pixel).
- The prior `(state->cov)` here is the covariance **after the LiDAR update of this same frame** — that's what makes it *sequential* tight fusion, not two independent filters.
- Convergence: stop when `‖δrot‖·57.3 < 1e-3°` and `‖δt‖·100 < 1e-3 cm` (`:1675`), else iterate (coarse→fine over pyramid levels and `max_iterations`). A line-search guard reverts the state if error increased (`:1677-1681`).
- After all levels: `state->cov -= G*state->cov` (`computeJacobianAndUpdateEKF:800`), then `updateFrameState` writes the corrected camera pose (`:1690-1696`).

There is also an **inverse-compositional** variant `updateStateInverse` (`:1398`) that precomputes reference-patch Jacobians once (`precomputeReferencePatches`, `:1327`) for speed — same residual, Jacobian evaluated at the reference.

---

## 6. Map maintenance after the update

- `generateVisualMapPoints` (`:804-906`): for grid cells with no map point, take the strongest LiDAR point (Shi-Tomasi score, `:822`), make a new `VisualPoint`, attach its first `Feature` patch, copy the LiDAR `var` into `covariance_`, set the normal — **this is where LiDAR depth becomes a visual point**.
- `updateVisualMapPoints` (`:908-967`): add a new observation to an existing point when the view changed enough (Δpos>0.5 m or Δθ>0.3 rad or pixel-shift>40 px, `:939-944`); cap obs list at 30 (`:947`).
- `updateReferencePatch` (`:969-1100`): refine the point normal from the voxel-map plane (with a 3σ plane-consistency test, `:1005-1011`), mark converged, and re-select the reference patch by an NCC+view-angle score (`:1036-1097`).

---

## 7. What Meridian should take from this (design implications)

1. **One state, sequential modality updates.** Keep a single state on the manifold; apply LiDAR then camera updates against the same covariance. This is the cleanest tight-fusion structure and it composes with a CT/iEKF front-end behind `IFrontEnd`.
2. **LiDAR-provided depth for vision** removes the entire VIO initialization/triangulation problem — adopt it. The camera contributes only photometric residuals.
3. **Unified voxel hash** for LiDAR planes and visual points (one spatial index) — fewer structures, cache-friendly, and it is exactly the registration layer Meridian already plans.
4. **Affine/exposure compensation (`inv_expo_time`)** is essential for direct methods under auto-exposure; make it a first-class (optionally estimated) state — required for the tactical day/night robustness goal.
5. **Patch + pyramid + NCC + depth-continuity gates** are the practical robustifiers; expose their thresholds as parameters and as debug overlays (FAST-LIVO2 prints a per-stage timing table, `:1851-1868` — Meridian's debug spec should publish these as ROS 2 diagnostics).
6. **Caveat to improve on:** FAST-LIVO2 uses heavy raw-pointer ownership (`new`/`delete` of `VisualPoint`/`Feature`, manual `feat_map` teardown in `~VIOManager`, `:20-27`). Meridian should use RAII/smart pointers and a clear ownership model in the visual map (consistent with the clean-architecture requirement).

---

## 8. Key citations (verified line numbers)

- Pipeline order: `src/vio.cpp:1786-1834` (`processFrame`).
- Association: `src/vio.cpp:352-782` (`retrieveFromVisualSparseMap`); depth image `:386-428`; raycast `:487-591`; depth-continuity `:618-640`; warp+gate `:645-763`.
- Residual: `src/vio.cpp:1619-1621` (update), `:746-751` (association).
- Jacobian: `src/vio.cpp:189-201` (proj.), `:1600-1617` (gradient + chain), constants `:62-65`.
- ESIKF: `src/vio.cpp:1648-1686`; cov update `:800`.
- Extrinsic composition: `src/vio.cpp:29-39`, `:57-58`.
- Map points: `visual_point.{h,cpp}`; `src/vio.cpp:227-250` (insert), `:804-906` (generate), `:969-1100` (ref-patch/normal).
- Structures: `include/vio.h:26-184`, `include/frame.h`, `include/visual_point.h`.
- Paper: FAST-LIVO2 arXiv 2408.14035 (ESIKF §IV; unified voxel map §V; exposure/photometric §VI).
