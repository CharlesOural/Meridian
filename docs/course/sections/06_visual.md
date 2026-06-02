## 06. Visual residual: sparse-direct photometric + LiDAR depth

> **Reading dependencies.** This section assumes the manifold machinery of §02 (the error state, $\boxplus/\boxminus$, $\mathrm{Exp}/\mathrm{Log}$, the hat operator $(\cdot)^\wedge$), the MAP/least-squares view of §03, the IMU propagation and prior of §04, and the LiDAR point-to-plane residual and the unified voxel map of §05. The recursive solver that consumes this residual — the iterated EKF / ESIKF and the sequential update — is the subject of §09; we give just enough of it here to show *where* the visual residual plugs in. The full single-step estimator, mapped end-to-end onto FAST-LIVO2, is §12.
>
> **Code grounding.** Every claim below is tied to FAST-LIVO2 source (`hku-mars/FAST-LIVO2`) at the paths under `slam-reference/FAST-LIVO2/`, and to the FAST-LIVO2 paper (Zheng et al., *T-RO* 2024, arXiv:2408.14035, in `papers/2408.14035.txt`). Citations are `file:line` and `eq.(n)` / `§n`. The classical sparse-direct method this builds on is SVO (Forster et al. 2014); the unified voxel map is from VoxelMap (Yuan et al. 2022, ref [14] in the paper).

---

### 06.1 Why a visual residual at all, and why *this* one

A LiDAR-inertial estimator (§04, §05) is metrically excellent but blind in two ways. First, it has **no colour**: the deliverable for Meridian is a *colourised* mesh (see the project map stack, L4), and colour must come from a camera. Second, and more importantly for the estimator, **geometry degenerates**: a long corridor, a tunnel, a single flat wall, a snowfield — these leave the point-to-plane Hessian rank-deficient along the unconstrained translation direction (this is the degeneracy of §05 and §11). In exactly those scenes a camera, which constrains pose through *appearance gradients* rather than geometry, can carry the axes LiDAR has lost. The two modalities fail on disjoint scene types, so fusing them is the whole robustness argument (paper §I; design rationale echoed in `arc-slam/docs/SOTA.md` §2.6).

The question is *how* to turn an image into a residual on the state. There is a spectrum:

- **Indirect / feature-based** (ORB-SLAM, LVI-SAM's VIO): extract keypoints, match descriptors, minimise *reprojection* error $\lVert \pi(T,{}^G\mathbf p) - \mathbf u_{\text{obs}}\rVert$. Requires a feature detector, a descriptor, a matcher, and — for a monocular camera — *triangulation* to recover the depth of each landmark.
- **Dense direct** (DTAM, and R3LIVE's VIO operate near this end): use *every* pixel, minimise photometric error over the whole image. Maximally informative, ruinously expensive, and tied to the resolution of the point map.
- **Sparse direct** (SVO; FAST-LIVO2): use a *few hundred* well-chosen points, but compare raw image *patches* (no descriptors), minimising photometric error. No feature extraction, no descriptor matching.

FAST-LIVO2 is sparse-direct, with one decisive twist that removes the worst cost of monocular direct VO. In ordinary direct VO you must *estimate the depth* of every map point (triangulation or depth filters) before you can project it and form a residual — a whole back-end of its own. Here, **the LiDAR already measured the depth.** The visual map points *are* LiDAR points (paper §I, contribution; abstract: "the visual module attaches image patches to the LiDAR points"). The camera never triangulates anything. This is the single most important idea in this section, and it is what makes the fusion *tight at the map level*, not just at the state level (paper §II-B contrasts this with DEMO/LIMO/LVI-SAM which keep separate maps and interpolate depth).

The consequences cascade:

1. No feature extraction, no descriptors, no matching → cheap and robust in low-texture (paper §II-A: "minimizes direct photometric errors without extracting ORB or FAST corner features").
2. No triangulation, no depth filter, no visual sliding-window BA → the visual module is *stateless* beyond the shared filter state.
3. One unified voxel map (§05) serves *both* LiDAR registration and visual alignment, and the LiDAR's *plane normals* serve the visual module's patch warping (§06.5).
4. The residual is "frame-to-map" in one shot, not frame-to-frame-then-frame-to-map (paper §II-B contrasts this with the two-stage R3LIVE/DVL-style trackers; robustness comes from not depending on a fragile optical-flow initial guess).

The rest of this section derives the residual (§06.4), the affine patch warp that makes it geometrically honest (§06.5), the exposure compensation that makes it photometrically honest (§06.6), the photometric Jacobian via the image-gradient chain rule (§06.7), and finally how the visual update is sequenced *after* the LiDAR update on one shared state (§06.8). We close with worked numbers (§06.9), the introspection a Meridian operator needs (§06.10), and the IFrontEnd contract (§06.11).

---

### 06.2 Notation and frames specific to this section

We use the shared notation of §02. Beyond it:

| Symbol | Meaning |
|---|---|
| $I(\mathbf u)$ | scalar image intensity (grey level, $0\!-\!255$) at pixel $\mathbf u=(u,v)^\top$, bilinearly interpolated |
| $\nabla I = (I_u, I_v)$ | image gradient (1×2 row), $I_u=\partial I/\partial u$, $I_v=\partial I/\partial v$ |
| $\pi(\mathbf p)$ | camera projection $\mathbb R^3 \to \mathbb R^2$, ${}^C\mathbf p \mapsto \mathbf u$ |
| $\pi^{-1}(\mathbf u)$ | back-projection (bearing) $\mathbb R^2 \to \mathbb S^2 \subset \mathbb R^3$ |
| $K_{\text{cam}}$ | pinhole intrinsics $(f_x,f_y,c_x,c_y)$ |
| ${}^G\mathbf p_i$ | a visual map point (a LiDAR point) in the global frame |
| ${}^C\mathbf p_i = (x,y,z)^\top$ | the same point in the *current* camera frame |
| $P_i$ (size $N{\times}N$) | the reference patch attached to map point $i$ |
| $\mathbf n$ | the local plane normal at the map point (from §05's voxel map) |
| $\tau$ | inverse exposure time $1/t_{\exp}$, scalar state component (`inv_expo_time`) |
| ${}^I_C T$, ${}^C_I T$ | camera↔IMU extrinsic (rigid); ${}^I_L T$ camera↔LiDAR is composed through these |
| $A_{c\,r}$ | $2{\times}2$ affine warp, reference patch → current patch |

**Frame chain.** The body frame is the IMU $I$ (paper §IV-A). The camera frame $C$ is fixed to $I$ by the extrinsic ${}^C_I T=({}^C_I R,{}^C_I \mathbf p)$. FAST-LIVO2 composes it from the LiDAR↔IMU and camera↔LiDAR extrinsics at startup:

```
Rci = Rcl * Rli;            // R_{C<-I} = R_{C<-L} R_{L<-I}
Pci = Rcl * Pli + Pcl;      // p_{C<-I}
```
— `vio.cpp:57-58`, with `setLidarToCameraExtrinsic` / `setImuToLidarExtrinsic` at `vio.cpp:29-39`. A global point reaches the camera by
$$
{}^C\mathbf p \;=\; {}^C_I R \,\big({}^G_I R^\top ({}^G\mathbf p - {}^G\mathbf p_I)\big) + {}^C_I\mathbf p
\;=\; {}^C_I R \,{}^I_G R\,{}^G\mathbf p + (\dots),
$$
which is exactly what `Frame::w2c` / `w2f` wrap (the world-to-camera and world-to-camera-frame helpers used throughout `retrieveFromVisualSparseMap`, e.g. `vio.cpp:409,416,466`). The pose being estimated is ${}^G_I T$ (the IMU pose, the LiDAR and camera ride on it through fixed extrinsics — which are themselves refineable as graph variables in the Meridian back-end, see L3).

---

### 06.3 The visual map point and its reference patch

The data structure is the heart of "LiDAR supplies depth, camera supplies appearance." A **visual map point** (`VisualPoint`, `visual_point.h`) is:

- `pos_` — ${}^G\mathbf p_i$, the 3-D position, *fixed by LiDAR* (it is a LiDAR point on a converged plane);
- `normal_`, `previous_normal_` — the local plane normal $\mathbf n$, seeded from the voxel-map plane (§05) and optionally refined (§06.5.3);
- `covariance_` — the $3{\times}3$ positional covariance carried over from the LiDAR point (`vio.cpp:886`, `pt_new->covariance_ = pt_var.var`);
- `obs_` — a list of **observations**, each a `Feature`;
- `ref_patch`, `has_ref_patch_`, `is_converged_`, `is_normal_initialized_` — bookkeeping for which observation is the *reference* and whether the point's normal has settled.

A **`Feature`** (`feature.h`) is one observed patch of this point:

- `patch_` — the raw pixel intensities, a *patch pyramid* of an $N{\times}N$ patch (paper §V-C: "three layers… each layer is half sampled from the previous"; default $N=8$ via `patch_size: 8`, with `patch_pyrimid_level: 4` levels in the avia config, `config/avia.yaml:33-34`; the buffer length is `patch_size_total * patch_pyrimid_level`, `vio.cpp:153`);
- `T_f_w_` — the camera pose ${}^C_G T$ at the frame where this patch was grabbed (so the patch "remembers" its viewpoint);
- `px_` — the pixel coordinate of the point in that frame; `f_` — its bearing $\pi^{-1}(\text{px})$;
- `inv_expo_time_` — the inverse exposure $\tau$ at that frame (§06.6);
- `level_`, `score_`, `mean_` — pyramid level, reference-selection score, patch mean.

So a single 3-D point accumulates a small album of patches from different viewpoints and exposures. Exactly one of them is elected the **reference patch** (§06.5.2); the alignment residual compares the *current* image against that reference, warped into the current view.

**How patches are sampled.** `getImagePatch` (`vio.cpp:203-225`) reads an $N{\times}N$ block around a sub-pixel centre `pc`, with bilinear interpolation (the four corner weights `w_ref_tl…w_ref_br` at `vio.cpp:212-215`) and a per-level stride `scale = 1<<level`. This is the only place raw pixels enter the residual, and the bilinearity is what makes $\nabla I$ well-defined for the Jacobian (§06.7).

**How a point is born.** `generateVisualMapPoints` (`vio.cpp:804-906`) walks the LiDAR scan's plane points `pg`, projects each into the image (`w2c`, `vio.cpp:814`), bins them into a $30{\times}30$-pixel grid, and in each empty grid cell keeps the candidate with the **highest Shi-Tomasi corner score** (`vk::shiTomasiScore`, `vio.cpp:822,845`). The winner becomes a new `VisualPoint`, its patch grabbed at the current pose and exposure (`vio.cpp:874-883`), its normal seeded from the LiDAR plane with a sign chosen to face the camera (`vio.cpp:889-890`), and it is inserted into the *same* voxel hash the LiDAR uses (`insertPointIntoVoxelMap`, `vio.cpp:227-250`, voxel size 0.5 m matching §05). High-gradient selection is what makes the method *sparse*: only points whose neighbourhood actually constrains pose are kept (a textureless wall point has zero gradient and contributes nothing — §06.7 makes this precise).

**Patch album maintenance.** `updateVisualMapPoints` (`vio.cpp:908-967`) adds a *new* observation to an existing point only when the view has changed enough to be worth it: translation > 0.5 m **or** rotation > 0.3 rad relative to the last patch's pose (`vio.cpp:937-939`), or pixel motion > 40 px (`vio.cpp:943-944`). The album is capped at 30 (`vio.cpp:947`), evicting the lowest-scoring patch (`findMinScoreFeature`, `visual_point.cpp:97-111`). Converged points keep only their reference patch (`deleteNonRefPatchFeatures`, `visual_point.cpp:113-126`). This bounded-memory album of multi-view patches is what lets the reference-selection (§06.5.2) prefer *inlier, high-parallax* references rather than the nearest (and therefore least-constraining) view.

---

### 06.4 The sparse-direct photometric residual

#### 06.4.1 Ideal residual (paper eq. 21)

Take a visual map point ${}^G\mathbf p_i$ with reference patch in frame $C_r$ (pose ${}^{C_r}_G T$, known and fixed since that frame was processed). For the *true* current state $\mathbf x_k$ — and hence the true camera pose ${}^C_I(\mathbf x_k)$ — the intensity the current image $I_k$ shows at the projection of the point should equal the intensity the reference image $I_r$ showed, for every pixel of the patch. Paper eq. (21):
$$
0 \;=\; \tau_k\, I_k\!\Big(\pi\big({}^C_I({}^G_I T)^{-1}\,{}^G\mathbf p_i\big) + \delta\mathbf u\Big)
\;-\; \tau_r\, I_r\!\Big(\pi\big({}^{C_r}_G T\,{}^G\mathbf p_i\big) + A_{c\,r}\,\delta\mathbf u\Big),
$$
where $\delta\mathbf u$ ranges over the patch offsets relative to the centre, $A_{c\,r}$ is the affine warp (§06.5) that relates patch coordinates in the reference to patch coordinates in the current view, and $\tau_k,\tau_r$ are the inverse exposures (§06.6). The pixel values are *measured* — $I_k=I_k^{gt}+\delta I_k$, $I_r=I_r^{gt}+\delta I_r$ — so the realised residual carries photometric measurement noise $\mathbf v_c=(\delta I_k,\delta I_r)$ (paper §VII-B, around eq. 21).

#### 06.4.2 The residual a programmer actually forms

For one patch, stack the per-pixel differences into a vector. With $\mathbf u_i = \pi({}^C\mathbf p_i)$ the projected centre,
$$
\boxed{\;\mathbf r_i(\mathbf x)\;=\;\Big[\,\tau_k\, I_k(\mathbf u_i+\delta\mathbf u_j)\;-\;\tau_r\, I_r(\mathbf u_{r,i}+A_{c\,r}\,\delta\mathbf u_j)\,\Big]_{j=1\ldots N^2}\;\in\;\mathbb R^{N^2}.\;}
$$
The second term is *precomputed once per iteration sweep*: the reference patch is affine-warped into the current geometry and frozen as `warp_patch`. In code, `retrieveFromVisualSparseMap` builds it via `warpAffine` over all pyramid levels (`vio.cpp:739-742`) and stores it in `visual_submap->warp_patch` (`vio.cpp:769`) along with the reference exposure (`vio.cpp:770`). The first term — the *current* patch $\tau_k I_k(\cdot)$ — is resampled fresh at the current pose each iteration via `getImagePatch(img, pc, patch_buffer, …)` (`vio.cpp:744`).

The scalar **photometric error** the code monitors per point is the sum of squares of this vector (`vio.cpp:746-751`):
```cpp
float error = 0.0;
for (int ind = 0; ind < patch_size_total; ind++)
{
  error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]) *
           (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]);
}
```
Read that line carefully: it is *exactly* $\sum_j\big(\tau_r\,P_r^{\text{warp}}[j] - \tau_k\,P_k[j]\big)^2$, the squared norm of $\mathbf r_i$. (Note the code subtracts current from reference, i.e. it stores $-\mathbf r_i$; a global sign on the residual is absorbed by the Kalman gain and is immaterial.) A point whose error exceeds `outlier_threshold * patch_size_total` is dropped before it ever enters the filter (`vio.cpp:763`) — a coarse but effective per-point gate (§06.10 returns to robust gating).

#### 06.4.3 Total cost and the MAP connection

The visual update minimises, over the error state $\widetilde{\mathbf x}$ (§02), the Mahalanobis sum of all inlier patches plus the prior carried from the LiDAR update (§06.8):
$$
\widetilde{\mathbf x}^\star \;=\; \arg\min_{\widetilde{\mathbf x}}\;
\underbrace{\big\lVert \widetilde{\mathbf x} \boxminus (\widehat{\mathbf x}\boxminus \overline{\mathbf x})\big\rVert^2_{\Sigma^{-1}}}_{\text{prior from LiDAR-updated state}}
\;+\;
\sum_{i\in\text{submap}} \big\lVert \mathbf r_i(\overline{\mathbf x}\boxplus\widetilde{\mathbf x})\big\rVert^2_{R_i^{-1}},
\qquad R_i=\sigma_I^2\, \mathbf I_{N^2} .
$$
This is the same NLLS-as-MAP picture as §03; the recursive way to solve it is §09's iterated EKF, and §06.8 shows the per-iteration update line. The measurement noise $R_i$ is isotropic per pixel with $\sigma_I$ the image intensity noise (paper §VII-B; configurable, related to `IMG_POINT_COV` in the YAML).

> **Worked micro-example.** $N=8 \Rightarrow N^2=64$ residuals per patch. A typical submap holds a few hundred points (the code prints `Retrieve %d points`, `vio.cpp:781`); say 300. Then one visual update assembles $300\times64 \approx 1.9\times10^4$ scalar residuals against a 7-DOF visual error state (6 pose + 1 exposure). Compare to LiDAR's one scalar per point (§05): the patch gives $64\times$ the constraints per point, which is why a *sparse* set of points suffices.

---

### 06.5 Affine warping with a LiDAR plane prior

A patch is a little flat window of texture. If the current camera views the surface from a different angle or distance than the reference did, the patch is sheared and scaled. Comparing raw patches without correcting for that would inject geometric error into the photometric residual. So before differencing, the reference patch is **affine-warped** into the current geometry. The warp is the $2{\times}2$ matrix $A_{c\,r}$ in eq. (21).

#### 06.5.1 The homography behind the affine warp (paper eq. 13)

If every pixel of the patch lay at the *same depth* (the assumption FAST-LIVO and SVO make), the warp would follow from that constant depth alone — and it would be wrong on any slanted surface. FAST-LIVO2's key accuracy gain (paper §I, contribution 2: "FAST-LIVO assumes all pixels in a patch share the same depth, a wild assumption") is to use the **LiDAR plane** the point lies on. With local plane normal ${}^{I_r}\mathbf n$ and point ${}^{I_r}\mathbf p$ in the reference frame, the relative pose $({}^{I_i}_{I_r}R,{}^{I_i}_{I_r}\mathbf t)$ induces a homography
$$
A_{i}^{\,r} \;=\; P\Big({}^{I_i}_{I_r}R + {}^{I_i}_{I_r}\mathbf t\,\frac{{}^{I_r}\mathbf n^\top}{{}^{I_r}\mathbf n^\top\,{}^{I_r}\mathbf p}\Big)P^{-1}\quad\text{(paper eq. 13)},
$$
with $P$ the projection (intrinsics for a pinhole). The corresponding code is `getWarpMatrixAffineHomography` (`vio.cpp:252-273`):
```cpp
const Eigen::Matrix3d H_cur_ref =
    T_cur_ref.rotation_matrix() * (normal_ref.dot(xyz_ref) * Matrix3d::Identity()
                                   - t * normal_ref.transpose());
```
— literally $R\,(\mathbf n^\top\mathbf p\,\mathbf I - \mathbf t\,\mathbf n^\top)$, the plane-induced homography. The $2{\times}2$ affine $A_{c\,r}$ is then read off by warping two unit pixel steps through the homography and differencing (`vio.cpp:261-272`). When the plane prior is disabled (`normal_en=false`), the code falls back to the constant-depth affine `getWarpMatrixAffine` (`vio.cpp:275-290`) — useful to *see* the accuracy difference in an ablation, and a good Meridian debug toggle.

`warpAffine` (`vio.cpp:292-318`) inverts $A_{c\,r}$ and resamples the reference image at the warped patch locations with bilinear interpolation (`vk::interpolateMat_8u`, `vio.cpp:315`), filling `warp_patch`. The best pyramid level to sample from is chosen by `getBestSearchLevel` (`vio.cpp:320-331`): it climbs levels while $\det A_{c\,r} > 3$, i.e. coarsens the reference when the current view is much *lower* resolution than the reference, to avoid aliasing.

#### 06.5.2 Choosing the reference patch (paper eq. 12)

Of the album of patches on a point, which is *the* reference? Choosing the nearest view (FAST-LIVO, SVO) is tempting but wrong: a patch from almost the current viewpoint imposes a *weak* constraint (little parallax) (paper §V-D). FAST-LIVO2 scores each candidate patch $f$ by photometric agreement with the others **and** by how square-on it views the plane:
$$
\text{NCC}(f,g)=\frac{\sum (f-\bar f)(g-\bar g)}{\sqrt{\sum(f-\bar f)^2\sum(g-\bar g)^2}},\qquad
c=\frac{\mathbf n^\top \mathbf p}{\lVert\mathbf p\rVert},\qquad
S=(1-\alpha)\,\tfrac1n\sum_i\text{NCC}(f,g_i)+\alpha\,c,
$$
(paper eq. 12). The patch with the highest $S$ wins. In code, `updateReferencePatch` (`vio.cpp:1036-1097`) computes exactly this: `score = NCC + cos_angle` (`vio.cpp:1087`), tracks the max (`vio.cpp:1091-1096`), and sets `pt->ref_patch`. The NCC term, borrowed from multi-view stereo, suppresses patches that fell on a moving object (they disagree with the rest); the cosine term keeps texture at high resolution. This is contribution 3 in the paper.

#### 06.5.3 Refining the normal (paper eqs. 14-16)

The plane normal can be *refined* photometrically: pick the ${}^{I_r}\mathbf n$ that minimises the photometric error between the reference patch and the others (paper eq. 14). Because $\mathbf n$ only enters through the 3-vector $M={}^{I_r}\mathbf n/({}^{I_r}\mathbf n^\top {}^{I_r}\mathbf p)$ subject to ${}^{I_r}\mathbf p^\top M=1$, the optimisation is reparameterised onto a 2-vector $\mathbf m\in\mathbb R^2$ (paper eqs. 15-16, the $M=B\mathbf m+\mathbf b$ chart of Fig. 5b), making it an unconstrained 2-D least-squares run **in a separate thread** so it never blocks odometry. The convergence test and sign handling live in `updateReferencePatch` (`vio.cpp:1019-1030`): once the normal stops moving (`normal_update < 1e-4`) and enough observations exist, the point is marked `is_converged_` and frozen. For Meridian this is an L2 background refinement, not on the critical path — but it is what gives pixel-level map colour later.

---

### 06.6 Affine photometric (exposure) compensation

Auto-exposure is the silent killer of direct VO: walk from a dim corridor into sunlight and the *same* surface reports wildly different intensities, so a raw photometric residual explodes even when the pose is perfect (paper §I, contribution 4; the dramatic indoor→outdoor sequence in Fig. 1 e5-e6). FAST-LIVO2 makes the **inverse exposure time $\tau$ a state variable** and estimates it online.

It enters multiplicatively: a patch grabbed under exposure $t_{\exp}$ has intensities scaled by $t_{\exp}$, so multiplying by $\tau=1/t_{\exp}$ normalises to a common radiance scale. That is precisely the $\tau_k,\tau_r$ factors in eq. (21) and in the `error` accumulation (`vio.cpp:749-750`): the *current* patch is scaled by `state->inv_expo_time` (the live state $\tau_k$) and the *reference* patch by `ref_ftr->inv_expo_time_` (the stored $\tau_r$ from when that patch was taken). Each `Feature` records the exposure under which it was captured (`ftr_new->inv_expo_time_ = state->inv_expo_time`, `vio.cpp:883,962`), so the album is exposure-consistent across illumination changes.

In the state-transition model $\tau$ is driven by a random walk (paper eq. 2: "$\tau$ is the inverse camera exposure time… $n_\tau$ models $\tau$ as a random walk"), so it is slowly time-varying, not constant — correct for an auto-exposing camera. Because $\tau$ multiplies the intensity, its Jacobian column is just the patch intensity itself (§06.7). This is the model R3LIVE++ also uses (paper §II-B); the difference is FAST-LIVO2 does it at *patch* level inside the same ESIKF.

> **Design note for Meridian.** Make exposure estimation a *switchable* feature behind the IFrontEnd config (the YAML flag `exposure_estimate_en` exists in FAST-LIVO2). In darkness, where the camera should be dropped entirely (open question 3 in `arc-slam/docs/NEXT_GEN_DESIGN.md` §14), you also drop $\tau$ from the state. When the camera is healthy but the scene is photometrically benign (constant indoor lighting), you may freeze $\tau$ to a constant to save a state dimension. The right place to decide is the L1 camera health channel feeding L2.

---

### 06.7 The photometric Jacobian (the image-gradient chain rule)

This is the technical core. We need $H_i=\partial \mathbf r_i/\partial\widetilde{\mathbf x}$, the $N^2\times 7$ block (6 pose + 1 exposure) that the ESIKF needs. Only the *current* term of $\mathbf r_i$ depends on the live state (the reference term is frozen warp), so we differentiate $\tau_k\,I_k(\pi({}^C\mathbf p_i + \delta\mathbf u_j))$.

#### 06.7.1 The chain

The dependence is a chain of four maps:
$$
\widetilde{\mathbf x}\;\xrightarrow{\;\text{pose action}\;}\;{}^C\mathbf p\;\xrightarrow{\;\pi\;}\;\mathbf u\;\xrightarrow{\;I_k\;}\;\text{intensity}\;\xrightarrow{\;\times\tau_k\;}\;\text{residual.}
$$
Differentiate one pixel of one patch and apply the chain rule:
$$
\frac{\partial r_{ij}}{\partial \widetilde{\mathbf x}_{\text{pose}}}
\;=\;
\tau_k \;\underbrace{\nabla I_k(\mathbf u_{ij})}_{1\times 2}\;
\underbrace{\frac{\partial \pi}{\partial\, {}^C\mathbf p}}_{2\times 3}\;
\underbrace{\frac{\partial\, {}^C\mathbf p}{\partial \widetilde{\mathbf x}_{\text{pose}}}}_{3\times 6}.
$$

**(a) Image gradient $\nabla I_k$ (1×2).** The grey-level derivatives $(I_u,I_v)$ at the sub-pixel projection, from finite differences on the bilinearly-interpolated image. This is the *only* place pixel content enters the Jacobian; it is also *why sparse-direct picks high-gradient points* — where $\nabla I=0$ (a blank wall) the whole row is zero and the point constrains nothing (§06.3, the Shi-Tomasi selection). The code samples the current patch with `getImagePatch` and forms gradients from neighbouring samples; this term is what makes the residual sensitive to *texture*.

**(b) Projection Jacobian $\partial\pi/\partial\,{}^C\mathbf p$ (2×3).** For a pinhole, with ${}^C\mathbf p=(x,y,z)^\top$, $u=f_x x/z+c_x$, $v=f_y y/z+c_y$:
$$
\frac{\partial\pi}{\partial\, {}^C\mathbf p}=
\begin{bmatrix}
\dfrac{f_x}{z} & 0 & -\dfrac{f_x x}{z^2}\\[2.2ex]
0 & \dfrac{f_y}{z} & -\dfrac{f_y y}{z^2}
\end{bmatrix}.
$$
This is implemented verbatim in `computeProjectionJacobian` (`vio.cpp:189-201`):
```cpp
J(0,0)=fx*z_inv;  J(0,2)=-fx*x*z_inv_2;
J(1,1)=fy*z_inv;  J(1,2)=-fy*y*z_inv_2;
```
(`z_inv=1/z`, `z_inv_2=1/z²`). Note the $1/z$ and $1/z^2$: nearby points (small $z$) give large pixel motion per metre of camera motion — close, textured surfaces are the most informative, which matches intuition.

**(c) Pose Jacobian $\partial\,{}^C\mathbf p/\partial\widetilde{\mathbf x}_{\text{pose}}$ (3×6).** The point in the camera frame is ${}^C\mathbf p = {}^C_I R\,{}^I_G R({}^G\mathbf p-{}^G\mathbf p_I)+{}^C_I\mathbf p$. Perturb the IMU pose on the manifold (§02) — attitude error $\delta\boldsymbol\theta$ and position error $\delta\mathbf p$ — and use $\frac{\partial}{\partial\delta\boldsymbol\theta}(R\,\mathbf a) = -R\,\mathbf a^\wedge$ and the position term:
$$
\frac{\partial\,{}^C\mathbf p}{\partial\delta\boldsymbol\theta}= {}^C_I R\,\big({}^I_G R({}^G\mathbf p-{}^G\mathbf p_I)\big)^{\wedge}\quad(\text{up to convention sign}),\qquad
\frac{\partial\,{}^C\mathbf p}{\partial\delta\mathbf p}= -\,{}^C_I R\,{}^I_G R .
$$
FAST-LIVO2 **precomputes the constant parts of this chain at startup**, which is exactly why the two members `Jdphi_dR` and `Jdp_dR` are set once in `initializeVIO` (`vio.cpp:62-65`):
```cpp
Jdphi_dR = Rci;                       // = R_{C<-I}
Pic = -Rci.transpose() * Pci;
tmp << SKEW_SYM_MATRX(Pic);
Jdp_dR = -Rci * tmp;                  // = -R_{C<-I} (p_{I<-C})^
```
`Jdphi_dR = ${}^C_I R$` rotates an attitude perturbation from the IMU tangent space into the camera frame; `Jdp_dR = $-{}^C_I R\,({}^I_C\mathbf p)^\wedge$` couples IMU attitude error into camera-frame position through the lever arm — the standard rigid-extrinsic lever-arm Jacobian. The third constant `Jdp_dt = Rci * Rwi.transpose()` $= {}^C_I R\,{}^I_G R = {}^C_G R$ is rebuilt each iteration (`vio.cpp:1544`). These constants are composed at runtime with the (b) projection block and the point-dependent skew to build each patch row — and the actual assembly in `updateState` (`vio.cpp:1611-1617`) is *verbatim* the chain rule above:
```cpp
Jimg << du, dv;                       // image gradient (a), finite-diff, vio.cpp:1600-1611
Jimg = Jimg * state->inv_expo_time;   // x tau_k  (the exposure factor, §06.6)
Jimg = Jimg * inv_scale;              // pyramid-level normalisation
Jdphi = Jimg * Jdpi * p_hat;          // p_hat = (^C p)^  (skew of CAMERA-frame point)
Jdp   = -Jimg * Jdpi;
JdR   = Jdphi * Jdphi_dR + Jdp * Jdp_dR;   // -> attitude columns (1x3)
Jdt   = Jdp * Jdp_dt;                       // -> position columns (1x3)
```
Note `p_hat << SKEW_SYM_MATRX(pf)` (`vio.cpp:1578`) is the skew of the *camera-frame* point ${}^C\mathbf p$, and the code factors the world-point form into the (precomputed) extrinsic constants `Jdphi_dR`/`Jdp_dR` plus the live `Jdp_dt` — algebraically the same 3×6 block, arranged for cache reuse. FAST-LIVO2 also offers an *inverse-compositional* variant (`updateStateInverse`, gated by `inverse_composition_en`, `vio.cpp:792-798`) that evaluates the gradient on the **reference** patch once in `precomputeReferencePatches` (`vio.cpp:1327-1396`, the reference-image `du,dv` and `JdR = Jimg*Jdpi*R_ref_w*p_w_hat`, `Jdt = -Jimg*Jdpi*R_ref_w` stored in `H_sub_inv`) and merely re-rotates it into the current frame each iteration (`vio.cpp:1470-1474`). This is the Baker-Matthews speed trick; it is part of why FAST-LIVO2 needs only ~3 iterations per pyramid level vs. FAST-LIVO's ~10 (paper §IX-E). The inverse-compositional path drops the exposure column (6-wide `H_sub`, `vio.cpp:1418`), so exposure estimation requires the forward `updateState` path.

**(d) Exposure column (×1).** Differentiating $\tau_k\,I_k$ w.r.t. the exposure error gives simply
$$
\frac{\partial r_{ij}}{\partial\widetilde\tau}\;=\;I_k(\mathbf u_{ij}),
$$
the raw current intensity — cheap, and the reason exposure is almost free to estimate. In code this is the literal `cur_value` (the bilinearly-interpolated current intensity) appended as the 7th column when `exposure_estimate_en` (`vio.cpp:1628`):
```cpp
if (exposure_estimate_en) { H_sub.block<1,7>(...,0) << JdR, Jdt, cur_value; }
else                      { H_sub.block<1,6>(...,0) << JdR, Jdt; }
```

#### 06.7.2 Assembling the row

Stacking (a)-(d), one pixel contributes a $1\times7$ Jacobian row
$$
H_{ij}=\Big[\;\underbrace{\tau_k\,\nabla I_k\,\tfrac{\partial\pi}{\partial {}^C\mathbf p}\,\big(\text{pose }3\times6\big)}_{1\times 6}\;\;\Big|\;\;\underbrace{I_k(\mathbf u_{ij})}_{1\times1}\;\Big],
$$
and the patch's block $H_i\in\mathbb R^{N^2\times 7}$ stacks $N^2$ such rows; the full visual Jacobian `H_sub` is $\big(\text{total\_points}\cdot N^2\big)\times 7$ (`vio.cpp:1533`), exactly the $H_c$ of paper Algorithm 1 line 19 ("Compute residual $\mathbf z_c$ and Jacobian $H_c$"). The per-pixel residual stored alongside it is `res = inv_expo_time*cur_value - inv_ref_expo*P[...]` (`vio.cpp:1621`) — current minus warped reference, i.e. $-\mathbf r_i$ relative to the eq.(21) sign of §06.4; the consistent sign through `K` and the update line makes this immaterial. Because every pose row factors through the *same* tiny per-point chain, the assembly is $O(N^2)$ multiply-adds per point and is OpenMP-parallelised over points (`#pragma omp parallel for`, `vio.cpp:1554`) — fast enough for 10-50 Hz on a CPU (paper §I, §IX-E).

> **Intuition recap.** The Jacobian says: a pixel constrains the pose only where (i) it has image gradient (texture), (ii) it is close (small $z$ ⇒ large $\partial\pi$), and (iii) the camera motion moves it across that gradient. Degenerate visual scenes (no texture) → zero rows → the visual update simply doesn't fight the LiDAR there; degenerate LiDAR scenes (textured corridor) → strong visual rows on the axis LiDAR lost. The two Jacobians are complementary by construction, which is the formal statement of the robustness claim in §06.1.

---

### 06.8 The sequential ESIKF update: LiDAR then image, one state

#### 06.8.1 Why sequential, not stacked

LiDAR and camera produce measurements of *different dimensions* (one scalar per LiDAR point vs. $N^2$ per patch) and live on different pyramid levels; stacking them into one update is awkward and couples their iteration counts. FAST-LIVO2's answer (paper §IV-D, the central contribution 1) is a **sequential update**: given the IMU-propagated prior $p(\mathbf x)$, fuse LiDAR first to get $p(\mathbf x\mid\mathbf y_l)$, then fuse the image against *that* posterior. The factorisation that makes this exactly equivalent to a joint update (paper eq. 5), assuming LiDAR and image noise are independent given the state:
$$
p(\mathbf x\mid \mathbf y_l,\mathbf y_c)\;\propto\; p(\mathbf y_c\mid \mathbf x)\,\underbrace{p(\mathbf y_l\mid \mathbf x)\,p(\mathbf x)}_{\propto\;p(\mathbf x\mid \mathbf y_l)}.
$$
Both fusions have the identical form $q(\mathbf x\mid\mathbf y)\propto q(\mathbf y\mid\mathbf x)\,q(\mathbf x)$ (paper eq. 8). For the **visual** step the "prior" $q(\mathbf x)$ is the *converged LiDAR-updated* state and covariance — i.e. the camera refines what LiDAR already settled (paper §IV-D, immediately after eq. 8: "In case of the visual update… $(\overline{\mathbf x},\overline P)$ is the converged state and covariance obtained from the LiDAR update").

#### 06.8.2 The shared iterated-EKF update (paper eq. 11)

Each fusion is an *iterated* EKF step (§09): linearise the measurement at the current iterate $\mathbf x^\kappa$, solve, $\boxplus$, repeat to convergence. The update line (paper eq. 11), specialised to the visual residual $\mathbf z_c$, Jacobian $H_c$, noise $R_c$:
$$
K=(H_c^\top R_c^{-1}H_c + \overline P^{-1})^{-1}H_c^\top R_c^{-1},\qquad
\mathbf x^{\kappa+1}=\mathbf x^{\kappa}\boxplus\big(-K\mathbf z_c-(\mathbf I-KH_c)(\mathbf x^{\kappa}\boxminus\overline{\mathbf x})\big).
$$
The iterated update is mathematically equivalent to Gauss-Newton on the MAP cost of §06.4.3 (Bell & Cathey; this equivalence is the backbone of §09). FAST-LIVO2 uses the FAST-LIO Woodbury form so the inverted matrix is state-sized, not measurement-sized, with the isotropic image noise $R_c=\sigma_I^2\mathbf I$ entering as the scalar `img_point_cov` (`IMG_POINT_COV`, default 100, = paper's "camera photometric noise = 100", §IX-A). The exact solve (`vio.cpp:1657-1667`) is:
```cpp
H_T_H.block<7,7>(0,0) = H_sub_T * H_sub;                         // = H^T H
K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse(); // (H^T H + (P/sigma_I^2)^-1)^-1
HTz = H_sub_T * z;                                                // = H^T z
vec = (*state_propagat) - (*state);                              // = x_prior boxminus x^kappa
G.block<DIM_STATE,7>(0,0) = K_1.block<DIM_STATE,7>(0,0) * H_T_H.block<7,7>(0,0);
solution = -K_1.block<DIM_STATE,7>(0,0)*HTz + vec - G.block<DIM_STATE,7>(0,0)*vec.block<7,1>(0,0);
(*state) += solution;                                            // boxplus
```
This is eq. (11) line-for-line: `solution` $=-K\mathbf z_c-(\mathbf I-KH_c)(\mathbf x^\kappa\boxminus\overline{\mathbf x})$, with `state_propagat` the prior $\overline{\mathbf x}$ (the LiDAR-updated state) and the $\boxplus$ implemented by `StatesGroup::operator+=` (which `Exp`-maps the rotation increment and adds the rest, `common_lib.h:182-192`). The convergence test is $\lVert\delta\mathbf R\rVert<0.001°$ and $\lVert\delta\mathbf t\rVert<0.001$ cm (`vio.cpp:1675`); the iteration also rolls back if the photometric error rises (`error <= last_error` guard, `vio.cpp:1648`) — a Levenberg-like safeguard. After convergence the covariance is updated $\overline P\leftarrow(\mathbf I-G)\overline P$ via `state->cov -= G*state->cov` (`vio.cpp:800`). `computeJacobianAndUpdateEKF` (`vio.cpp:784-802`) runs this **from the coarsest pyramid level down to the finest** (`for (int level = patch_pyrimid_level-1; level>=0; level--)`, `vio.cpp:790`) — a coarse-to-fine cascade that widens the convergence basin (the classic image-alignment trick) — then writes the refined pose back into the frame (`updateFrameState`, `vio.cpp:801`). The outer pyramid loop matches paper Algorithm 1 lines 13-23 (the `until level >= 2` loop).

#### 06.8.3 Full per-frame order (paper Algorithm 1)

The end-to-end sequence for one synchronized LiDAR+image frame, with the visual pieces in **bold**:

1. Scan recombination + IMU forward propagation → prior $(\overline{\mathbf x},\overline P)$ (§04; paper Alg.1 L1-2).
2. Backward propagation to deskew the LiDAR scan (§04/§05; Alg.1 L3).
3. **LiDAR ESIKF update** to convergence (§05; Alg.1 L4-11). Posterior $(\overline{\mathbf x},\overline P)$.
4. Update the voxel-map *geometry* with the registered scan (§05; paper §V-B).
5. **Build the visual submap** for this image: `retrieveFromVisualSparseMap` (`vio.cpp:352-782`) — project map points, build the LiDAR depth image, reject occluded/depth-discontinuous points (§06.5.4 below), pick references, warp patches, gate by photometric error.
6. **Visual ESIKF update** to convergence, coarse-to-fine over pyramid levels: `computeJacobianAndUpdateEKF` (§06.8.2; Alg.1 L13-23). Posterior $(\widehat{\mathbf x},\widehat P)$ — the optimal state.
7. **Append/maintain visual map points** at $\widehat{\mathbf x}$: `generateVisualMapPoints` (`vio.cpp:804`), `updateVisualMapPoints` (`vio.cpp:908`).
8. **Refine reference patches / normals** in the background: `updateReferencePatch` (`vio.cpp:969`) (§06.5.2-3).
9. $\widehat{\mathbf x}$ propagates the next IMU interval; emit keyframe to L3 (§07, §08, §09).

#### 06.8.4 Submap selection and outlier rejection (paper §VII-A)

Step 5 deserves its own note because it is where "LiDAR finds the visible points" happens. `retrieveFromVisualSparseMap`:

- **Visible-voxel query** — poll the voxel-hash cells hit by the *current LiDAR scan* (`sub_feat_map`, `vio.cpp:386-404`); map points in the camera FoV almost surely lie in those cells (paper §VII-A1). This is far cheaper than scanning the whole map.
- **On-demand raycasting** — for image grid cells the voxel query left empty (e.g. LiDAR blind zone, or camera FoV beyond LiDAR FoV), cast a ray through the cell centre and sample voxels along it (`raycast_en` block, `vio.cpp:486-591`; pre-tabulated ray samples from `initializeVIO`, `vio.cpp:80-126`) (paper §VII-A2, Fig. 7).
- **Depth-image occlusion rejection** — render the LiDAR scan to a depth image (`vio.cpp:371-425`) and discard map points whose patch neighbourhood straddles a depth discontinuity > 0.5 m (`depth_continous` test, `vio.cpp:619-640`) (paper §VII-A3, Fig. 8). This kills patches that span an occluding edge, which would otherwise produce nonsense photometric residuals.
- **Per-point gates** — keep only points with an initialised normal (`vio.cpp:651`), optionally an NCC check vs. the current patch (`vio.cpp:753-761`), and the photometric-error threshold (`vio.cpp:763`).

Only the survivors enter `visual_submap` and thus the Jacobian. For Meridian this whole stage is the natural home of *per-point* introspection (§06.10).

---

### 06.9 A small worked example end-to-end

Take one textured point on a slanted wall, 4 m ahead, seen by a 640×512 pinhole ($f_x=f_y=400$, $c_x=320$, $c_y=256$), $N=8$.

1. **Birth.** A LiDAR point ${}^G\mathbf p$ lands on a voxel plane with normal $\mathbf n$ (§05). It projects to $\mathbf u_r=(360,250)$ in the reference image with strong Shi-Tomasi score, so `generateVisualMapPoints` keeps it, grabs its $8{\times}8{\times}3$ patch, stores $\tau_r$ and ${}^{C_r}_G T$.
2. **Next frame.** Camera has moved 0.3 m and yawed 2°. IMU+LiDAR ESIKF settles the pose to $\overline{\mathbf x}$. The point projects (with the *predicted* pose) to $\mathbf u_i\approx(372,251)$; depth $z\approx3.7$ m.
3. **Warp.** Relative pose + plane normal → homography (`getWarpMatrixAffineHomography`) → $A_{c\,r}$ (slight horizontal shear from the yaw and the slant). `warpAffine` resamples the reference into the current geometry → 64-pixel `warp_patch`. `getBestSearchLevel` says level 0 ($\det A<3$).
4. **Residual.** Resample the *current* 64-pixel patch; form $r_j=\tau_r P_r^{\text{warp}}[j]-\tau_k P_k[j]$. Suppose the predicted pose is 1.5 px off along the gradient; the patch mean residual is, say, 6 grey-levels.
5. **Jacobian.** At one bright-edged pixel $\nabla I_k=(40,5)$ grey/px. Projection block at $z=3.7$: $\partial u/\partial x=f_x/z=108$ /m. The position column for camera-x ≈ $\tau_k\cdot 40\cdot 108 \approx 4.3\times10^3\,\tau_k$ residual-per-metre — a stiff constraint along the gradient direction. The exposure column is just the intensity ($\sim$180).
6. **Update.** Stack 300 such patches; one ESIKF iteration nudges the pose ~1.4 px worth along the visual gradients and tweaks $\tau_k$ by the average bright/dark mismatch. Coarse-to-fine repeats; converges in 2-4 iterations.
7. **Maintain.** At the new $\widehat{\mathbf x}$, view changed > thresholds? `updateVisualMapPoints` appends a fresh patch+exposure. Background thread re-scores references and refines $\mathbf n$.

Numbers are illustrative, but the *shape* — large, well-conditioned Jacobian entries on close textured points, near-zero on blank or far ones — is exactly what the math predicts.

---

### 06.10 Introspection and debug topics (the Meridian requirement)

Meridian's non-negotiable principle is that an operator/developer can *see* what the estimator is doing (project brief; `arc-slam/docs/NEXT_GEN_DESIGN.md` §10.3). For the visual residual specifically, expose:

- **Submap overlay** — the projected visual map points coloured by residual magnitude (green inlier → red rejected), the depth-image occlusion mask, and the raycast-recovered points. FAST-LIVO2 already builds the ingredients in `projectPatchFromRefToCur` (`vio.cpp:1102+`, the ref/cur side-by-side image dump) and the `printf("[ VIO ] Retrieve %d points…")` counters (`vio.cpp:781,902,966`); promote these to ROS image/marker topics.
- **Per-stage timing** — `retrieveFromVisualSparseMap` time, `computeJacobianAndUpdateEKF` time (the code already separates `compute_jacobian_time` / `update_ekf_time`, `vio.cpp:788`), per pyramid level. Publish as a structured timing message.
- **Effective points & convergence** — number of inlier patches actually in $H_c$, iterations to convergence per level, final mean photometric error. These are the visual analogues of LiDAR's "effective feature num" (§05).
- **Exposure trace** — plot $\tau_k$ over time; a sudden jump flags an illumination transition (and validates the exposure estimator). 
- **Visual observability** — eigen-spectrum of $H_c^\top R_c^{-1}H_c$ restricted to the pose block, so the operator sees which axes the *camera* is constraining vs. LiDAR. This is the per-modality, per-axis observability that flows into the back-end noise (§11; design §6.5).
- **Reference-patch gallery** — for a clicked map point, show its album and which patch is the current reference and why (its $S$ score). Invaluable when alignment misbehaves.

Robustness hooks beyond the coarse `outlier_threshold` gate (§06.4.2): wrap the per-patch residual in a robust kernel (Huber) so a few bad patches (specular highlight, dynamic object that slipped past the NCC filter) cannot dominate the update; this is the front-end counterpart of the GNC/switchable machinery the back-end uses (§11).

---

### 06.11 The IFrontEnd contract for Meridian

Meridian builds the front-end behind a swappable `IFrontEnd` interface (project brief; an iEKF v1 now, a continuous-time v2 later, §10). The visual residual must therefore be a *module*, not a hard-wired step. Concretely:

- **Inputs:** a synchronized image (L1, photometrically calibrated and undistorted — see design §5.2), the LiDAR scan's plane points with normals (from §05's voxel map), the IMU-propagated prior (§04), and the *shared* state object (pose + velocity + biases + gravity + exposure $\tau$).
- **State coupling:** the visual update mutates the *same* state the LiDAR update did (sequential update, §06.8). The interface must expose the state's error-space (§02) and its covariance so the camera step can fuse against the LiDAR posterior. Do **not** give the camera a private state — that would be loose coupling and would forfeit the whole argument of §06.1.
- **Map coupling:** the visual module reads and writes the *same* voxel hash as LiDAR (§05). Visual map points are LiDAR points with a patch album bolted on; this must be a property attached to the existing map structure, not a parallel map.
- **Swap points:** `IFrontEnd` should let the photometric residual be (i) enabled/disabled (LiDAR-inertial-only fallback when the camera is dead), (ii) configured for exposure on/off, plane-prior warp on/off, inverse-composition on/off, pyramid depth, patch size — every one of these is a real FAST-LIVO2 toggle and a legitimate ablation/operating mode.
- **v2 forward-compatibility:** in the continuous-time front-end (§10), the *same* photometric residual and Jacobian (§06.4, §06.7) attach to the B-spline control points whose support overlaps the image timestamp, instead of to a single discrete pose. The residual's measurement-side math (image gradient × projection × exposure) is **identical**; only the pose-side $\partial\,{}^C\mathbf p/\partial(\cdot)$ changes from a single-pose Jacobian to a sum over control-point Jacobians. Designing the residual to take the camera pose (and its tangent map) as an argument — rather than reaching into a fixed state layout — is what makes the iEKF→CT swap a drop-in.

---

### 06.12 Summary

The visual residual in FAST-LIVO2 is **sparse-direct photometric patch alignment with LiDAR-supplied depth**. Its defining moves:

1. **LiDAR gives depth** → the visual map point *is* a LiDAR point; no triangulation, no depth filter, no visual BA (§06.1, §06.3).
2. **Raw patches, not features** → no detector, descriptor, or matcher; high-gradient point selection keeps it sparse (§06.3, §06.4).
3. **Plane-prior affine warp** → patches are warped through the LiDAR plane's homography, not a flat constant-depth assumption (§06.5, paper eq. 13).
4. **Reference patch chosen for parallax + inlier agreement** (NCC + view angle), not proximity (§06.5.2, eq. 12).
5. **Online inverse-exposure $\tau$ in the state** → multiplicative photometric compensation survives auto-exposure swings (§06.6).
6. **Jacobian = image-gradient chain** $\tau\,\nabla I\cdot\partial\pi/\partial{}^C\mathbf p\cdot\partial{}^C\mathbf p/\partial\widetilde{\mathbf x}$, with the constant lever-arm parts (`Jdphi_dR`, `Jdp_dR`) precomputed (§06.7).
7. **Sequential ESIKF** → fuse LiDAR, then fuse the image against the LiDAR posterior, on **one shared state**, coarse-to-fine over the image pyramid (§06.8, paper eq. 5, eq. 11, Alg. 1).

The result is a tightly-coupled, ROS-agnostic, swappable visual front-end module whose constraints are *complementary* to LiDAR's: it carries the pose axes that geometry loses, and it paints the map with colour for the operator. Continue to §07 (GNSS/absolute residuals), §09 (the ESIKF in full), §11 (degeneracy/robustness that consumes the per-axis observability of §06.7), and §12 (the whole estimator step assembled).
