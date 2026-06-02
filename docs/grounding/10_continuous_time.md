# Grounding Dossier 10 — Continuous-Time Estimation: SE(3) B-spline Trajectories for LiDAR-Inertial(-Visual) Odometry

**Scope.** This dossier grounds the *continuous-time (CT) front-end* option for Meridian: representing the sensor trajectory as a cubic B-spline on SE(3) (or split SO(3)×ℝ³), attaching IMU / LiDAR / camera measurements at their *true timestamps*, and solving a sliding-window non-linear least squares over the control points. It captures the exact spline value/derivative equations, residual formulations, knot-placement strategies, marginalization, the canonical reference implementations, and a concrete library recommendation for Meridian. It contrasts the CT approach with the discrete iterated-EKF front-end (FAST-LIO2, Point-LIO), which is the *other* branch behind Meridian's single front-end interface.

**Primary sources (read for this dossier).**
- **CLINS** — Lv, Hu, Xu, Liu, Ma, Zuo, *"CLINS: Continuous-Time Trajectory Estimation for LiDAR-Inertial System,"* IROS 2021. arXiv:2109.04687. Repo: `github.com/APRIL-ZJU/clins`. PDF: `april.zju.edu.cn/core/papercite-data/pdf/lv2021clins.pdf`.
- **Coco-LIC** — Lang, Chen, Tang, Ma, Lv, Liu, Zuo, *"Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry using Non-Uniform B-spline,"* IEEE RA-L 8(11):7074–7081, 2023. arXiv:2309.09808. Repo: `github.com/APRIL-ZJU/Coco-LIC`.
- **Sommer / Usenko / Schubert / Demmel / Cremers**, *"Efficient Derivative Computation for Cumulative B-Splines on Lie Groups,"* CVPR 2020. arXiv:1911.08860. (The mathematical foundation; "the continuous-time trajectory is taken from basalt," per the CLINS README.)
- **basalt-headers** — `VladyslavUsenko/basalt-headers`; docs `vladyslavusenko.gitlab.io/basalt-headers/`. Classes `RdSpline`, `So3Spline`, `Se3Spline`. The de-facto reference C++ implementation of the above math.
- **Contrast:** FAST-LIO2 (Xu, Cai, He, Zhu, Lin, Liu, Zhang), arXiv:2107.06829; Point-LIO (He, Xu, Zhang et al.), *Adv. Intelligent Systems* 2023.
- Background derivations: Kim et al. "spline fusion" (cumulative cubic B-spline on SE(3) for SLAM, the original); Haarbach/Usenko et al. comparison of SE(3) vs split.

> **Confidence note.** Equation numbers below are quoted from machine reads of the arXiv HTML (ar5iv) and the basalt-headers Doxygen. Where the source paper does not state a numeric value (e.g., some covariance weights, the exact Coco-LIC control-point lookup thresholds), this dossier says so explicitly rather than inventing it. Downstream authors should verify the two or three flagged items against the PDFs before freezing a spec.

---

## 1. Why continuous-time at all (the problem it solves)

A modern multi-sensor rig produces measurements that are (a) *asynchronous* (LiDAR points stream continuously over a ~100 ms sweep; IMU at 200–400 Hz; camera at 10–30 Hz) and (b) sampled *during* motion. A discrete-time estimator parameterizes the trajectory by a pose **at each scan/keyframe** and must:

1. **Undistort** the LiDAR sweep — assume/interpolate intra-scan motion to deskew points to a single reference time (motion compensation). Errors here directly corrupt the map.
2. **Pre-integrate** the IMU between two discrete poses (a separate, approximation-laden machinery, e.g. Forster et al.).
3. Pick a single timestamp for the camera and tolerate the residual time mismatch.

A **continuous-time** estimator instead represents the trajectory as a *single smooth function* `T(t) ∈ SE(3)` parameterized by a small set of control points. Then **any** measurement, at **its own exact timestamp t**, is a function of `T(t)` and its analytic time-derivatives. Consequences:

- **Sub-scan motion is absorbed for free.** A LiDAR point at time `t` is transformed by `T(t)` — no separate deskew step; deskew *is* the trajectory. (CLINS abstract: "simultaneously removes the motion distortion in LiDAR scans.")
- **Asynchrony is native.** IMU, LiDAR, camera all evaluate the *same* `T(·)` at different `t`. No pre-integration, no timestamp snapping.
- **Velocity / acceleration / angular velocity are analytic** (derivatives of the spline), so the IMU is a *direct* residual on those derivatives — no integration.
- **High-rate / arbitrary-rate sensors** fuse cleanly; the trajectory can be queried at any rate (Coco-LIC reports 100 Hz pose queries from the spline).

**Crucial limitation (state it loudly).** A CT spline absorbs *known* timing — every measurement's residual uses its *recorded* timestamp. It does **NOT** correct an *unknown clock offset / time-delay* `t_d` between sensors (e.g., LiDAR-to-camera latency, IMU triggering skew). If sensor A's clock is biased by `t_d` relative to B's, the spline will faithfully fit the *wrong* time and bake the offset into the trajectory. Removing `t_d` requires *explicitly adding it as an estimated parameter* and evaluating that sensor's residual at `T(t + t_d)` (which is straightforward in CT precisely because `T(·)` is differentiable in `t` — see §9). CT does not magically self-synchronize clocks. This is the single most important caveat for Meridian.

---

## 2. Cumulative B-splines on Lie groups (the math foundation)

### 2.1 Uniform cubic B-spline, scalar/Euclidean form

A B-spline of order `k` (degree `k−1`) is `C^{k−2}`-continuous. **Cubic** means `k = 4` (degree 3), giving `C²` continuity — position, velocity, *and* acceleration are continuous, which is exactly what an IMU acceleration residual needs. A point on the curve at time `t` depends on exactly **`k = 4` control points** (the "local support" property): the 4 control points whose knot interval brackets `t`. This is why a LiDAR point's residual touches "~4 overlapping control points."

For a uniform spline with knot spacing `Δt`, define the normalized time within segment `i`:

```
u(t) = (t − t_i) / Δt ,  u ∈ [0, 1)
```

where `t_i` is the knot at the start of the active segment. The **cumulative** (a.k.a. blending) representation expresses the value relative to the first active control point plus weighted *increments*:

```
p(u) = p_i + Σ_{j=1}^{k-1} λ_j(u) · d_j ,   d_j = p_{i+j} − p_{i+j-1}        (Euclidean)
```

with the cumulative blending weights packed as a matrix product (Sommer et al. Eq. 21):

```
λ(u) = M̃^(k) · u ,   u = [1, u, u², …, u^{k-1}]ᵀ
```

`M̃^(k)` is the **cumulative basis matrix**: `m̃_{j,n} = Σ_{s=j}^{k-1} m_{s,n}` where `m` is the ordinary B-spline basis matrix (Sommer Eq. 18, cumulative property Eq. 20: `m̃_{0,n} = δ_{n,0}`).

**Numeric cubic (k=4) matrices (uniform), the ones basalt/CLINS/Coco-LIC use** — verify against `RdSpline::computeBlendingMatrix<k,…>` in basalt-headers:

Ordinary cubic basis matrix:
```
M^(4) = (1/6) · [ 1  4  1  0
                 -3  0  3  0
                  3 -6  3  0
                 -1  3 -3  1 ]
```
Cumulative cubic basis matrix (column-cumulative sums, the one actually multiplied by `u`):
```
M̃^(4) = (1/6) · [ 6  5  1  0
                  0  3  3  0
                  0 -3  3  0
                  0  1 -2  1 ]
```
so `λ(u) = M̃^(4)·[1,u,u²,u³]ᵀ` and `λ_0 ≡ 1` (the constant term, hence "value relative to `p_i`"). This is the classic Kim "spline fusion" matrix.

### 2.2 Lie-group (SE(3) / SO(3)) form — the actual trajectory

On a matrix Lie group `𝒢` with exp/log `Exp: 𝔤 → 𝒢`, the cumulative form replaces "+ increment" with "right-multiply by `Exp(increment)`" (Sommer Eq. 23–24; identical in CLINS Table I and Coco-LIC Eq. 2):

```
T(u) = T_i · Π_{j=1}^{k-1} Exp( λ_j(u) · d_j ) ,     d_j = Log( T_{i+j-1}^{-1} · T_{i+j} ) ∈ 𝔤      (Eq. 23/24)
```

For the **rotation** spline on SO(3) with control rotations `R_·`:
```
R(u) = R_i · Π_{j=1}^{3} Exp( λ_j(u) · d_j ) ,   d_j = Log( R_{i+j-1}^{-1} R_{i+j} ) ∈ 𝔰𝔬(3)
```
For **translation** on ℝ³, use the Euclidean form in §2.1.

### 2.3 Analytic derivatives (velocity, acceleration, angular velocity)

This is the payoff: the IMU residual needs `a(t)` (2nd derivative of position) and `ω(t)` (body angular velocity), and both are **closed-form** functions of the control points and `u`. Sommer et al.'s key contribution is an **O(k)** recurrence (vs O(k²) naive) — basalt implements exactly this.

Define `A_j(u) = Exp(λ_j(u)·d_j)`. Velocity (Sommer Thm 5.2, Eq. 33–35):
```
Ṫ = T · (ω^(k))^∧
ω^(j)   = Ad_{A_{j-1}^{-1}} · ω^(j-1) + λ̇_{j-1} · d_{j-1} ,    ω^(1) = 0
```
Acceleration (Sommer Thm 5.3, Eq. 40–42):
```
T̈ = T · ( (ω^(k))^∧ (ω^(k))^∧ + (ω̇^(k))^∧ )
ω̇^(j) = λ̇_{j-1}·[ω^(j)^∧, d_{j-1}^∧]_∨ + Ad_{A_{j-1}^{-1}}·ω̇^(j-1) + λ̈_{j-1}·d_{j-1} ,   ω̇^(1) = 0
```
For **SO(3)** the adjoint becomes a rotation transpose, simplifying to (Sommer Eq. 38, 46):
```
ω^(j)  = A_{j-1}ᵀ · ω^(j-1) + λ̇_{j-1} · d_{j-1}
ω̇^(j) = λ̇_{j-1} · (ω^(j) × d_{j-1}) + A_{j-1}ᵀ · ω̇^(j-1) + λ̈_{j-1} · d_{j-1}
```
The **body-frame angular velocity** the gyro sees is `ω = (Rᵀ Ṙ)_∨ = ω^(k)` from the SO(3) recurrence (Sommer notes `T^{-1}Ṫ = (ω^(k))^∧`). The `λ̇, λ̈` are obtained by differentiating `λ(u) = M̃·u`: `λ̇ = M̃·u̇·(1/Δt)`, `λ̈ = M̃·ü·(1/Δt²)` where `u̇ = [0,1,2u,3u²]ᵀ`, `ü = [0,0,2,6u]ᵀ`.

### 2.4 Split SO(3)×ℝ³ vs full SE(3) — the representation choice

Two ways to model `T(t)`:

- **Full SE(3) spline.** One cumulative spline directly on SE(3); control points are full poses; `d_j ∈ 𝔰𝔢(3)`. Rotation and translation are *coupled* through the SE(3) exp.
- **Split SO(3)×ℝ³.** A separate SO(3) spline (rotation) and a separate ℝ³ spline (translation). Translation control points and rotation control points are independent.

Sommer et al. §2 surveys the literature and concludes: *"on average, using the split representation is better both in terms of trajectory representation and in terms of computation time."* The intuition: an SE(3) spline couples rotation and translation in a way that produces a slightly unnatural "screw-like" interpolation and makes the translation depend on rotation control points; the split version is simpler, faster (no SE(3) adjoints in the cross terms), and empirically as-or-more accurate for VIO/LIO. **basalt's `Se3Spline` is itself implemented as a split** — a `So3Spline<N>` plus an `RdSpline<3,N>` (the class name says SE(3) but the internals are split). **CLINS and Coco-LIC both use the split representation** (separate `R(t)` and `p(t)` splines, per their Table I / Eq. 2).

**Takeaway for Meridian:** use the **split SO(3)×ℝ³** cubic spline. It is the consensus choice, it is what every reference impl ships, and it makes the IMU residual decomposition clean (gyro ↔ SO(3) spline's `ω`; accel ↔ ℝ³ spline's 2nd derivative rotated into body frame).

---

## 3. The basalt-headers reference implementation (data structures & API)

`basalt-headers` is header-only (Eigen + Sophus) and is the implementation CLINS explicitly reuses. Key classes (`include/basalt/spline/`):

- **`RdSpline<_DIM, _N, _Scalar>`** — uniform B-spline for ℝ^DIM, order `_N` (so cubic = `_N=4`).
  - Storage: control points (knots) in an `Eigen::aligned_deque` `knots_`; segment timing via `start_t_ns_`, `dt_ns_`, `inv_dt_`.
  - `computeBlendingMatrix<_N, _Scalar, _Cumulative>()` returns the static `M` or `M̃` (the §2.1 matrices) at compile time.
  - `evaluate<Derivative>(time_ns, J*)`: returns value (Derivative=0), velocity (=1), acceleration (=2); fills the analytic Jacobian w.r.t. the `_N` active control points if requested.
  - Time→segment: `s = (t − start_t)/dt`, integer part = first active knot index `i`, fractional part = `u`.
- **`So3Spline<_N, _Scalar>`** — cumulative SO(3) spline. Control points are `Sophus::SO3`.
  - `evaluate(time_ns)` → `SO3` rotation; `velocityBody(time_ns)` → body angular velocity `ω` (the §2.3 SO(3) recurrence); `accelerationBody(time_ns)`; all with optional analytic Jacobians w.r.t. control points.
- **`Se3Spline<_N, _Scalar>`** — the trajectory facade = `So3Spline<_N>` (rotation) + `RdSpline<3,_N>` (translation), i.e. **split**.
  - `pose(time_ns)` → `Sophus::SE3`; `transVelWorld`, `transAccelWorld` (translation spline derivatives in world frame); `rotVelBody` (gyro); plus helpers that directly build IMU/pose residuals.
  - `knots_push_back(...)` appends a control point; constructor takes the knot spacing `dt`.

The basalt classes are **uniform** (constant `dt`). Non-uniform behaviour (Coco-LIC) is layered on top via virtual time (see §5).

---

## 4. Residual formulations (the factors that touch the spline)

All three reference systems use **Ceres** (Levenberg–Marquardt) with **analytic** spline Jacobians; states are control points (SO(3) via `LocalParameterization`/manifold) plus IMU biases (+ gravity, + time offset, + extrinsics if calibrating).

### 4.1 IMU residual — direct, no pre-integration

A raw IMU sample `(ω_m, a_m)` at time `t_m` (CLINS Eq. 8–9; Coco-LIC Eq. 15). The spline gives `ω(t_m)` (body angular velocity from the SO(3) spline) and `a(t_m)` (2nd derivative of the ℝ³ spline = world acceleration). Residuals:

```
r_ω = ω(t_m) + b_g − ω_m
r_a = R(t_m)ᵀ · ( a(t_m) − g^G ) + b_a − a_m
```

(`g^G` world gravity, optionally estimated as a 2-DOF direction on S²; `b_g, b_a` gyro/accel biases.) **Bias random-walk** factor between window steps (Coco-LIC Eq. 16):
```
r_bg = b_g^k − b_g^{k-1} ,   r_ba = b_a^k − b_a^{k-1}
```
Each `r_ω, r_a` factor connects to exactly the **`k=4` control points** active at `t_m` (4 SO(3) for `r_ω`; 4 SO(3)+4 ℝ³ for `r_a` because `a` is rotated into body frame). This *is* the IMU's role: a residual on the spline's 2nd derivative + angular velocity. Compare with discrete LIO where the IMU is a separately pre-integrated factor between poses.

### 4.2 LiDAR point-to-plane residual — attaches a point at time t to ~4 control points

For a raw LiDAR point `p^L` measured at *its own* timestamp `t` (CLINS Eq. 5–7; Coco-LIC Eq. 12–13):

1. **Timestamp → spline → world.** Evaluate the trajectory at the point's exact time: `R(t), p(t)` (each from the 4 overlapping control points active at `t`). Transform to world (or to the scan-reference frame for undistortion):
   ```
   p̂^G = R_{LG}(t) · p^L + p_{LG}(t)        (Coco-LIC Eq. 12; LiDAR↔IMU extrinsic folded in)
   ```
   Because every point uses *its own* `t`, this **is** deskew — no separate motion-compensation pass.
2. **Plane association.** Fit a plane `(n^G, d^G)` to the 5 nearest map neighbors (same as FAST-LIO2 — no feature extraction needed, though CLINS also keeps edge features with a point-to-line variant).
3. **Residual** (point-to-plane signed distance):
   ```
   r_L = n^Gᵀ · p̂^G + d^G        (Coco-LIC Eq. 13)
   ```

Jacobian of `r_L` flows through `R(t), p(t)` to the 4 active SO(3) + 4 active ℝ³ control points. A whole scan contributes hundreds–thousands of such factors, each touching a (time-dependent) different set of 4 control points; the *overlap* of these sets across the scan is what stitches the trajectory together.

### 4.3 Camera / visual residual (Coco-LIC)

Coco-LIC's signature trick: **avoid optimizing pixel depth** (which would need a long sliding window / many control points). Instead, **borrow depth from the LiDAR map**: project map points into the image, associate to tracked features (KLT sparse optical flow), and form a **frame-to-map reprojection** factor (Coco-LIC Eq. 14):
```
r_C = π_c( C_G T(t) · p_s^G ) − [u_s, v_s]ᵀ     (Cauchy robust kernel)
```
where `p_s^G` is a LiDAR-map 3D point with a known image observation `(u_s,v_s)`. The camera factor evaluates `T(t)` at the *image* timestamp → again the 4 active control points. This keeps the window short (no visual structure to triangulate), which is what makes a 3-sensor CT system real-time.

> **Meridian relevance.** Meridian's target output is a colourised mesh, so the camera is for *colour + extra constraint*, not dense VO. Coco-LIC's "depth-from-LiDAR-map, frame-to-map reprojection" pattern is an excellent fit: it constrains the trajectory and gives per-point colour without the cost of full visual SLAM structure in the window.

---

## 5. Knot placement: uniform vs non-uniform / adaptive

### 5.1 CLINS (essentially uniform, dynamically grown)

CLINS uses a **uniform** cubic spline with constant `Δt`, adding new control points as scans arrive (initialized by IMU integration to match pose/velocity/orientation). Reported `Δt`: **0.05 s** for fast room-scale motion, **0.1 s** for slower large-scale. The trade-off is global: small `Δt` → more control points → captures aggressive motion but more compute and risk of under-constraint; large `Δt` → cheaper but over-smooths fast motion. CLINS picks one `Δt` per scenario; it has *no* per-segment adaptivity. (CLINS' "two-stage" method is for *loop-closure correction*, §7 — not knot density.)

### 5.2 Coco-LIC (non-uniform / adaptive — the contribution)

Coco-LIC keeps a **fixed outer cadence of `Δt = 0.1 s`** (one optimization step per 0.1 s window) but **adaptively chooses how many control points `n_cp` to place inside each 0.1 s segment** based on motion dynamics measured from IMU (Coco-LIC §III-B/C, Eq. 11–12):
```
N_g = (1/n)·‖ Σ_i R_i · ω_i ‖        (mean angular rate over the segment)
N_a = (1/n)·‖ Σ_i (R_i · a_i − g) ‖  (mean linear acceleration over the segment)
```
A lookup (paper Fig. 2; **exact thresholds are given graphically, not as text numbers — verify in the PDF/code before freezing**) maps `(N_g, N_a)` → a discrete `n_cp` (and "we choose the larger one" if the two disagree). The chosen control points are then placed **uniformly within** the 0.1 s segment (interval `0.1/n_cp`). Net effect: **more control points during aggressive motion, fewer during smooth motion** — best of both: accuracy when needed, cheap when not.

**Implementation insight (important for Meridian).** Coco-LIC realizes "non-uniform" by, in effect, running a *uniform* cubic B-spline machinery over a **re-mapped / virtual time** so each adaptively-placed control point is equally spaced in virtual time. (The paper presents the non-uniform cumulative matrix `M̃^(k)(i)` as *segment-dependent* — Eq. 2/8 — which is the rigorous formulation; the code uses the time-remapping to reuse uniform-spline kernels.) The cumulative blending matrix is no longer a single constant: `M̃^(k+1)(i)` depends on the (non-uniform) knot positions of segment `i`. Confidence note: the read indicates Coco-LIC uses *genuine* non-uniform matrices (Eq. 2 caption: "`M̃^(k+1)(i)` is non-constant — the main difference between uniform and non-uniform B-splines"), with the virtual-time uniform reuse as the engineering realization; downstream authors should confirm which path the released code takes (`trajectory` / spline classes in the Coco-LIC repo, which themselves vendor basalt-style headers).

### 5.3 Recommendation on knots for Meridian

Adopt **Coco-LIC's adaptive scheme**: fixed outer window cadence (~0.1 s), IMU-driven `n_cp` per segment. It is the SOTA accuracy/efficiency trade-off and directly handles the "aggressive motion vs idle" range a survey/handheld rig sees. If schedule forces a v1 shortcut, ship a **uniform** spline first (CLINS-style, `Δt ≈ 0.05–0.1 s`) behind the same interface and add adaptivity later — the residual/Jacobian code is identical; only knot insertion changes.

---

## 6. Sliding window + marginalization

Both CT systems run a **fixed-lag / sliding-window** optimization (not full batch), so cost stays bounded:

- **Active set.** Control points spanning the current window (the new scan/segment) plus a small overlap of recent ones are *optimized*. Older control points outside the window are either held fixed or marginalized.
- **CLINS local window (§IV-B).** Control points in the active segment `Φ(t_k, t_k+ΔT)` are optimized; older "static" control points are kept *fixed* (included in residuals but not updated). Cost = LiDAR + IMU(accel+gyro) residuals over the window. Converges in 4–5 iterations, ~200 ms/optimization.
- **Coco-LIC marginalization (§III-D4, Eq. 17–18).** Joint cost over the window:
  ```
  argmin_{X^k}  Σ‖r_L‖²_ΣL + Σ‖r_C‖²_ΣC + Σ‖r_I‖²_ΣI + Σ‖r_Ib‖²_ΣIb + Σ‖r_prior‖²_Σprior
  ```
  On sliding the window forward, states leaving the window are **marginalized via a Schur-complement prior** (`r_prior`). The retained prior set `X_prior` = {control points *shared* between the marginalized and remaining windows} ∪ {current IMU biases `b_g, b_a`}. This is the standard VINS-Mono-style linearized prior carried as a dense factor on the boundary control points. Marginalizing (vs fixing) keeps information from old measurements without re-optimizing them — better consistency than CLINS' "hold fixed," at the cost of prior bookkeeping.

**Recommendation.** Meridian's CT front-end should marginalize (Schur-complement prior on boundary control points + biases), Coco-LIC-style, for consistency; "hold-fixed" is acceptable for a first cut but biases the estimate as the window slides. Note: this front-end window is *separate* from the global GTSAM/iSAM2 back-end (Meridian's back-end); the CT window outputs relative-pose / keyframe factors into iSAM2.

---

## 7. Loop closure with a CT trajectory (CLINS two-stage)

A naive global re-optimization of all control points after a loop closure is intractable. CLINS' **two-stage correction (§IV-C, Fig. 3)**:
1. **Stage 1 — discrete pose-graph.** Run a standard pose-graph optimization over *key-scan poses* only (cheap), to get globally consistent corrected poses `(R̂_k, p̂_k)`.
2. **Stage 2 — control-point update.** Re-fit the continuous spline to honor the corrected anchor poses while *preserving local shape* via the original local velocities/angular velocities:
   ```
   argmin_{Φ_update}  Σ ( ‖Log(R̂_kᵀ R(t_k))‖ + ‖p(t_k) − p̂_k‖ )
                     + Σ ( ‖R(t_j)ᵀ v(t_j) − v̂_j‖ + ‖ω(t_j) − ω̂_j‖ )
   ```
For Meridian, loop closure is the back-end's job (Scan Context++/STD → GICP → PCM → iSAM2). The CLINS two-stage idea is still useful as the mechanism to *deform the CT front-end trajectory* to agree with a back-end loop correction without re-running the front-end window.

---

## 8. Contrast with discrete iEKF front-ends (the other Meridian branch)

Meridian's front-end interface admits *either* CT spline *or* a FAST-LIO2-style iEKF. Side-by-side:

### 8.1 FAST-LIO2 (iterated error-state EKF) — arXiv:2107.06829
- **State** on manifold `ℳ = SO(3)×ℝ¹⁵×SO(3)×ℝ³` (24-D): pose, velocity, biases, gravity, LiDAR–IMU extrinsic (Eq. 3).
- **Propagation:** IMU drives forward propagation; **backward propagation** deskews each point to scan-end using its exact sample time (Algorithm 1, §IV-A2) — this is the *discrete* analogue of "evaluate spline at point time," but it relies on the *propagated* (not jointly optimized) intra-scan trajectory.
- **Update:** iterated update; Kalman gain computed at *state* dimension (Eq. 14):
  ```
  K = (Hᵀ R⁻¹ H + P⁻¹)⁻¹ Hᵀ R⁻¹
  x̂^{κ+1} = x̂^κ ⊞ ( −K z^κ − (I−KH)(J^κ)⁻¹ (x̂^κ ⊟ x̂) )
  ```
- **Map:** **ikd-Tree** — incremental k-d tree with point-wise insert/delete, **box-wise delete** (lazy labels), on-tree downsampling, parallel re-balancing (α-balanced/α-deleted). Point-to-plane residual on raw points vs 5 nearest neighbors (Eq. 4–5, 9). No feature extraction.
- **Strengths:** extremely fast (filter, no NLLS window), robust, mature. **Weaknesses:** intra-scan motion handled by *propagation* (approximate) not joint optimization; pre-integration-style IMU handling baked in; harder to fuse a camera tightly at arbitrary timestamps; deskew accuracy degrades under very aggressive motion.

### 8.2 Point-LIO — *Adv. Intelligent Systems* 2023
- **Point-by-point update** (not frame-by-frame): the filter updates at *each LiDAR point's* timestamp, so there is **no intra-scan distortion to compensate at all** — the estimate advances continuously.
- **Stochastic-process IMU model:** IMU `ω, a` are treated as **outputs of the state** (state *augmented with* angular velocity and linear acceleration, driven by a random-walk/white-noise process) rather than as *inputs* that must be integrated. Consequence: handles **very high angular velocity** and **saturated IMU** (the IMU dropout is just a missing measurement of an augmented state, not a propagation failure).
- Still an EKF; conceptually the *filter-side* analogue of CT's "trajectory is a smooth function of time, IMU is a measurement of its derivatives." Point-LIO is the strongest discrete competitor to CT for aggressive-motion robustness.

### 8.3 CT vs iEKF — decision table

| Dimension | CT B-spline (CLINS/Coco-LIC) | iEKF (FAST-LIO2 / Point-LIO) |
|---|---|---|
| Intra-scan motion | Absorbed natively (deskew = trajectory) | Backward-propagation deskew (FAST-LIO2) / per-point update (Point-LIO) |
| Sensor asynchrony | Native (any sensor at its `t`) | Needs per-sensor handling; camera fusion awkward |
| IMU handling | Direct residual on spline 2nd-deriv + ω | Propagation / augmented-state |
| Multi-camera tight fusion | Clean (Coco-LIC) | Harder (need MSCKF-style add-ons) |
| Compute | Heavier (NLLS window, Ceres) ~tens–200 ms | Lighter (filter), real-time on CPU |
| Maturity / robustness | Newer, fewer deployments | Very mature, battle-tested |
| Unknown clock offset `t_d` | Must add `t_d` as parameter (easy: `T(t+t_d)` differentiable) | Must add `t_d`; less natural |
| Loop-closure deformation | Two-stage CT correction | Re-localize / pose-graph |

---

## 9. Time-offset / temporal calibration (the caveat, operationalized)

Because `T(·)` is analytically differentiable in `t`, adding an unknown per-sensor time offset `t_d` is clean: evaluate that sensor's residual at `T(t_meas + t_d)` and let Ceres compute `∂r/∂t_d` via `∂r/∂T · Ṫ(t)`. This is exactly how CT calibration toolboxes (Kalibr-style, and the APRIL-ZJU lab's own calibration work, e.g. LI-Calib/OA-LICalib) estimate LiDAR–IMU and camera–IMU time offsets. **Meridian should expose `t_d` per non-reference sensor as an optional optimized parameter** (online or one-shot calibration), defaulting to fixed-zero once calibrated. Without this, the "CT absorbs asynchrony" benefit is real only for *known* timestamps and silently degrades under clock skew.

---

## 10. Failure modes (catalog)

1. **Knot spacing too large** → spline cannot represent fast motion → systematic LiDAR/IMU residual the optimizer cannot null → drift/blur. (Mitigation: Coco-LIC adaptive `n_cp`.)
2. **Knot spacing too small** → control points under-constrained between sparse measurements → overfitting/jitter, ill-conditioning, slow optimization. (Mitigation: IMU residuals densely constrain; cap `n_cp`.)
3. **Pure rotation / degenerate geometry** → translation control points unobservable from LiDAR → relies entirely on IMU; bias errors leak into trajectory.
4. **Unknown time offset `t_d`** → silently baked into the trajectory (§1, §9). The headline trap.
5. **Gravity / accel-bias ambiguity** at low excitation → poor `b_a`/gravity separation (shared with all VIO/LIO).
6. **Marginalization linearization error** → stale prior inconsistent after large corrections (fix-vs-marginalize trade-off, §6).
7. **SE(3) (non-split) spline coupling** → unnatural screw interpolation, worse and slower than split (§2.4).
8. **Compute blow-up** in 3-sensor windows if depth is optimized per pixel → Coco-LIC avoids via LiDAR-map depth (§4.3).
9. **Boundary/extrapolation** — querying `T(t)` outside the spanned knots is undefined; the window must always cover the newest measurement before adding factors.

---

## 11. Recommendation for Meridian's CT front-end

**Representation.** Split **SO(3)×ℝ³ cubic (order 4, `C²`) cumulative B-spline.** Rationale: consensus best accuracy/speed (Sommer §2; used by CLINS, Coco-LIC, basalt); cubic gives the `C²` continuity the IMU acceleration residual requires; split decouples gyro↔SO(3) and accel↔ℝ³ Jacobians.

**Knots.** **Adaptive (Coco-LIC-style)**: fixed ~0.1 s outer cadence, IMU-driven `n_cp ∈ {1..N_max}` per segment from `(N_g, N_a)`. Ship a uniform `Δt≈0.05–0.1 s` (CLINS-style) as the v1 fallback behind the same interface.

**Factors.** Direct IMU (`r_ω`, `r_a`) + bias random-walk; LiDAR point-to-plane on raw points vs 5-NN planes (deskew-free); optional camera **frame-to-map reprojection with LiDAR-map depth** (Coco-LIC) for colour + constraint. Add per-sensor `t_d` as an optimizable parameter.

**Window / solver.** Fixed-lag sliding window with **Schur-complement marginalization** (prior on boundary control points + biases). Output keyframe relative-pose factors to the GTSAM/iSAM2 back-end.

**Library choice — recommended: `basalt-headers` (vendored) as the spline kernel, Ceres as the solver.**
- *Why basalt-headers:* it is the **canonical, battle-tested C++ implementation** of exactly the §2 math — `RdSpline`/`So3Spline`/`Se3Spline` with the O(k) analytic derivatives *and analytic Jacobians w.r.t. control points* (the hard, error-prone part). Header-only (Eigen + Sophus), permissive BSD-3, and it is precisely what CLINS reuses ("the continuous-time trajectory is taken from basalt"). Reusing it removes the single biggest implementation risk (hand-deriving and debugging spline Jacobians).
- *Adaptation needed:* basalt ships **uniform** splines. For Coco-LIC adaptivity, wrap basalt with the **virtual-time remapping** (place adaptive knots, run uniform kernels over re-scaled time) — this is exactly the Coco-LIC engineering pattern and avoids re-deriving non-uniform Jacobians. Alternatively, **start from the Coco-LIC repo's spline module directly** (it already vendors basalt-style headers and adds the non-uniform handling, IMU/LiDAR/camera Ceres factors, and marginalization) — fastest path to a working CT front-end, though it carries Coco-LIC's code style/license (inspect before vendoring).
- *Solver:* **Ceres** (LM, analytic cost functions, manifold/`LocalParameterization` for SO(3), built-in Schur for marginalization) — what all three references use. (GTSAM is reserved for Meridian's *global* iSAM2 back-end; the CT window is its own Ceres problem feeding the back-end.)
- *Avoid:* writing the spline + Jacobians from scratch (months of subtle bugs), and avoid a full-SE(3) (non-split) spline.

**Trade-offs accepted:** CT is heavier than an iEKF and newer/less battle-tested operationally. Hence keep the **iEKF (FAST-LIO2/Point-LIO) branch behind the same front-end interface** as the robust/real-time default, and use the CT branch where its strengths pay off: tight camera fusion for colourisation, very aggressive motion, and asynchronous multi-LiDAR. This is exactly the dual-branch design Meridian already specifies.

---

## 12. Source index (for downstream spec authors)

- CLINS — arXiv:2109.04687 ; PDF `april.zju.edu.cn/core/papercite-data/pdf/lv2021clins.pdf` ; repo `github.com/APRIL-ZJU/clins` (Ceres; spline from basalt; split SO(3)+ℝ³ cubic; Δt 0.05/0.1 s; two-stage loop correction).
- Coco-LIC — arXiv:2309.09808 ; RA-L 8(11):7074–7081 ; repo `github.com/APRIL-ZJU/Coco-LIC` (non-uniform/adaptive `n_cp`; Eq. 2/8 non-uniform `M̃`; Eq. 11–12 `N_g,N_a`; Eq. 12–13 LiDAR; Eq. 14 visual frame-to-map; Eq. 15–16 IMU; Eq. 17–18 window+marginalization; deps Ceres 2.0, Eigen 3.3.7, Sophus, PCL, OpenCV 4).
- Sommer et al. — arXiv:1911.08860 ; CVPR 2020 (Eq. 21 `λ=M̃u`; Eq. 23–24 Lie cumulative; Thm 5.2/5.3 velocity/accel recurrences; split vs SE(3) §2). 
- basalt-headers — `vladyslavusenko.gitlab.io/basalt-headers/` ; repo `github.com/VladyslavUsenko/basalt-headers` (`RdSpline`,`So3Spline`,`Se3Spline`; `computeBlendingMatrix`; `evaluate<Derivative>`; analytic Jacobians).
- FAST-LIO2 — arXiv:2107.06829 (24-D state Eq. 3; iEKF update Eq. 14; backward-propagation deskew; ikd-Tree; point-to-plane Eq. 4–5,9).
- Point-LIO — *Adv. Intelligent Systems* 2023 ; repo `github.com/hku-mars/Point-LIO` (point-by-point update; IMU-as-output stochastic model; high-rate/saturated-IMU robustness).
- Background: Kim et al. "spline fusion" (original cubic SE(3) cumulative spline for SLAM); Haarbach/Usenko SE(3)-vs-split comparison.

### Items to verify against PDFs/code before freezing a formal spec
1. Coco-LIC Fig. 2 exact `(N_g, N_a) → n_cp` thresholds (graphical in paper) and whether the released code uses genuine non-uniform `M̃(i)` vs virtual-time uniform reuse.
2. Exact numeric cubic cumulative matrix sign convention as used by basalt `computeBlendingMatrix` (basis vs cumulative; ensure consistency with the recurrence). The §2.1 matrices follow the Kim/basalt convention but cross-check the code.
3. Covariance/weight values `Σ_L, Σ_I, Σ_C` — not numerically given in the papers' main text; tune empirically.
