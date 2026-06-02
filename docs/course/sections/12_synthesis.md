## 12. Synthesis: one full estimator step, mapped to FAST-LIO2/LIVO2

> **Where we are.** Sections 02–11 built the pieces of a tightly-coupled
> multi-sensor estimator one at a time: the state on a manifold (§02), the
> probabilistic MAP / factor-graph view (§03), the inertial residual (§04), the
> LiDAR point-to-plane residual (§05), the sparse-direct photometric residual
> (§06), the GNSS / absolute residual (§07), batch Gauss–Newton (§08), the
> recursive iterated-EKF and its equivalence to batch MAP (§09), continuous-time
> trajectories (§10), and robustness / observability machinery (§11). This final
> section is the **capstone**: it walks *one complete estimator step* end to end
> and pins each abstract idea to the exact line of code that implements it in two
> reference systems we keep on disk — FAST-LIO2
> (`FAST_LIO/src/laserMapping.cpp`, the LiDAR-inertial baseline) and FAST-LIVO2
> (`FAST-LIVO2/src/LIVMapper.cpp`, the LiDAR-inertial-visual extension).
>
> Papers describe the *math*, but the code reveals the *order of operations*, the
> *approximations actually shipped*, and the *introspection hooks* — exactly the
> three things Meridian's design principles (modularity, simplicity,
> debug-in-the-right-places) demand we understand before rebuilding from scratch.

Throughout, we reuse the notation defined in §02: state $x$ (rotation
$R\in SO(3)$, position $p$, velocity $v$, gyro bias $b_g$, accel bias $b_a$,
gravity $g$, and the LiDAR↔IMU extrinsic $({}^{I}R_L,{}^{I}t_L)$ — in code
`offset_R_L_I`, `offset_T_L_I`); boxplus $\boxplus$ / boxminus $\boxminus$; the
capitalised Lie maps $\mathrm{Exp}/\mathrm{Log}$ and lowercase $\exp/\log$; hat
$(\cdot)^\wedge$; residual $r$, measurement $z$, prediction $h(x)$, Jacobian $H$,
covariance $\Sigma$, information $\Omega=\Sigma^{-1}$, Kalman gain $K$; LiDAR
point $p_L$, plane $(n,d)$ with $n\cdot x + d = 0$; camera intensity $I$,
projection $\pi$, intrinsics $K_{\text{cam}}$.

In the FAST-LIO2 code the state $x$ is the 23-dimensional manifold built by the
IKFoM macro `MTK_BUILD_MANIFOLD(state_ikfom, …)`
(`FAST_LIO/include/use-ikfom.hpp:12`–`21`), in this order:

$$
x = \big(\underbrace{p}_{3},\;\underbrace{R}_{SO(3)},\;
\underbrace{{}^{I}R_L}_{SO(3)},\;\underbrace{{}^{I}t_L}_{3},\;
\underbrace{v}_{3},\;\underbrace{b_g}_{3},\;\underbrace{b_a}_{3},\;
\underbrace{g}_{S^2}\big),
$$

with the error state living in $\mathbb{R}^{23}$ (two SO(3) blocks contribute 3
each, the $S^2$ gravity contributes 2). The control input is
$u=(a,\omega)$ — `input_ikfom` with members `acc`, `gyro`
(`use-ikfom.hpp:23`–`26`). Keep this layout in mind: it is what makes the
measurement Jacobian's "12 nonzero columns" in §12.6 concrete (columns for
$p, R, {}^{I}R_L, {}^{I}t_L$).

---

### 12.1 The step, in one sentence and one diagram

A single FAST-LIO2 estimator step consumes one **sweep** of LiDAR points plus all
IMU samples spanning it, and produces an updated state and an updated map. It is,
almost exactly, **one iterated-EKF measurement update wrapped around an
IMU-driven prediction**, which §09 showed is algebraically the same object as one
Gauss–Newton solve of the MAP problem of §03. The whole thing is the body of the
`while` loop in `main()` (`FAST_LIO/src/laserMapping.cpp:865`–`1019`):

```
   ┌─────────────────────────── one estimator step ───────────────────────────┐
   │                                                                           │
 [§02] sync_packages ──► p_imu->Process ──► (a) IMU forward-propagate x, Σ     │
   │   (lidar+imu pkg)    (IMU_Processing)      over the sweep   (predict)      │
   │   laserMapping:869   laserMapping:888  (b) backward-pass deskew points     │
   │                                                                           │
 [§05] lasermap_fov_segment ──► downSizeFilterSurf ──► feats_down_body          │
   │   laserMapping:901          laserMapping:904       (the scan we register)  │
   │                                                                           │
 [§08/09] kf.update_iterated_dyn_share_modified(LASER_POINT_COV, …)            │
   │        laserMapping:960   (body: esekfom.hpp:1619)                         │
   │        repeat i=-1..max_iter:                                              │
   │          h_share_model:  associate (ikd-Tree kNN) → plane fit → residual   │
   │          laserMapping:638   r_j = n_j·(R(R_L p_L + t_L)+p) + d_j   [§05]    │
   │                             Jacobian row H_j (1×12)               [§05]     │
   │          solve δx (iEKF normal equations)  esekfom.hpp:1782      [§08/09]   │
   │          x ← x ⊞ δx ; check ‖δx‖<limit     esekfom.hpp:1817,1821 [§02/09]   │
   │        final covariance  Σ ← (I−KH)Σ       esekfom.hpp:1834+     [§09]      │
   │                                                                           │
 [§05] map_incremental ──► ikd-Tree.Add_Points   laserMapping:976,427          │
   │                                                                           │
 [§06] (FAST-LIVO2 only) interleave a VIO update: photometric residual on       │
   │     LiDAR-depth map points, same iEKF machinery (LIVMapper.cpp)            │
   │                                                                           │
       publish_odometry / publish_path / publish_frame_world  (debug topics)    │
       laserMapping:972, 980, 981                                               │
   └───────────────────────────────────────────────────────────────────────────┘
```

We use FAST-LIO2 as the spine because its single-residual structure is the
cleanest instance of the general estimator; §12.7 shows FAST-LIVO2 reusing the
*identical* iterated-update skeleton, only swapping/adding the residual.

---

### 12.2 Stage 0 — packaging a measurement group (the clock)

`sync_packages(MeasureGroup &meas)` (`laserMapping.cpp:368`) assembles the unit of
work: one LiDAR sweep plus every IMU sample whose timestamp falls inside
$[t_{\text{lidar,beg}},\,t_{\text{lidar,end}}]$. This is the discrete "frame" of
§03's factor graph — between consecutive frames lives the IMU preintegration /
propagation factor (§04); at the frame live the LiDAR factors (§05). The first
sweep only sets the time origin and is skipped (`flg_first_scan`,
`laserMapping.cpp:871`–`877`) because there is no prior motion to propagate from.

The sweep end time is set from the last point's offset (`lidar_end_time =
meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000)`,
`laserMapping.cpp:393`), and the function blocks (returns `false`) until the IMU
buffer covers `lidar_end_time` (`laserMapping.cpp:404`–`407`) — so the update
never runs on an interval the IMU has not yet bracketed.

Two timing details the code makes explicit and the papers gloss:

1. **Per-point timestamps are relative.** Each raw point carries its offset from
   sweep start in its `curvature` field, in milliseconds; the deskew loop reads it
   as `it_pcl->curvature / double(1000)`
   (`IMU_Processing.hpp:323`, `:325`). This is the per-point time $\tau$ that
   §10's B-spline trajectory $T(\tau)$ would consume if we swapped in a CT
   front-end.
2. **The IMU stream is stitched across frames.** `UndistortPcl` prepends the
   *previous* frame's last IMU sample (`v_imu.push_front(last_imu_)`,
   `IMU_Processing.hpp:220`) so the integration interval has no gap at the seam —
   the implementation correlate of the preintegration interval boundaries in §04.

For Meridian this is L0/L1 (sensor abstraction + per-sensor preprocessing); the
lesson is that *time, not data, is the contract between modules* — PTP-disciplined
per-point stamps are what let L2's estimator be sensor-agnostic.

---

### 12.3 Stage 1 — IMU forward propagation (the prediction / prior)

`p_imu->Process(...)` (`laserMapping.cpp:888`) calls `ImuProcess::UndistortPcl`
(`IMU_Processing.hpp:216`), which in one pass **forward-propagates the state and
covariance** (this stage) and **backward-deskews the points** (§12.4).

#### The continuous-time kinematics, discretised

FAST-LIO2 uses the IKFoM "input" formulation: the gyro and accel are *inputs*,
not states, and the state derivative $f(x,u)$ is integrated. The derivative is the
function `get_f` (`use-ikfom.hpp:47`–`59`), whose nonzero blocks are *exactly*
the kinematics of §04 (with $\omega = \mathrm{gyro}\boxminus b_g$ and
$a_{\text{inertial}} = R(\mathrm{acc}-b_a)$):

$$
\dot p = v,\qquad
\dot R \;\leftrightarrow\; \omega = \omega_m - b_g,\qquad
\dot v = R(a_m - b_a) + g,\qquad
\dot b_g=\dot b_a=\dot g=0,
$$

shipped as `res(i)=s.vel[i]` ($\dot p$), `res(i+3)=omega[i]` (the SO(3) tangent
rate), `res(i+12)=a_inertial[i]+s.grav[i]` ($\dot v$)
(`use-ikfom.hpp:54`–`56`). Per-interval the inputs are the **midpoint averages**
of consecutive IMU samples (`IMU_Processing.hpp:257`–`262`),
$\bar\omega = \tfrac12(\omega_k+\omega_{k+1})$,
$\bar a = \tfrac12(a_k+a_{k+1})$, with the raw accel rescaled to metric gravity
units, `acc_avr * G_m_s2 / mean_acc.norm()` (`IMU_Processing.hpp:266`). The step
is `kf_state.predict(dt, Q, in)` (`IMU_Processing.hpp:284`), implemented in the
IKFoM toolkit (`esekfom.hpp:279`).

**State mean** (manifold-aware Euler step). §04/§09 wrote
$x_{k+1} = x_k \boxplus (f(x_k,u_k)\,\Delta t)$; the toolkit applies it as
`x_.oplus(f_, dt)` (`esekfom.hpp:287`). For the SO(3) and $S^2$ blocks this is a
genuine retraction, not vector addition: the rotation increment is applied through
the matrix exponential `MTK::exp<scalar_type,3>(...)` with the right Jacobian
absorbed (`esekfom.hpp:312`, `:321` set `F_x1.block<3,3>(idx,idx) =
res.toRotationMatrix()`) — the right-Jacobian-of-$\boxplus$ of §02.

**Error-state covariance** (the prior $\Omega$ carried into the update),
$\Sigma_{k+1} = F_x\,\Sigma_k\,F_x^\top + F_w\,Q\,F_w^\top$, shipped as

```cpp
F_x1 += f_x_final * dt;                                   // esekfom.hpp:380
P_ = (F_x1)*P_*(F_x1).transpose()
   + (dt*f_w_final)*Q*(dt*f_w_final).transpose();         // esekfom.hpp:381
```

with $F_x \approx I + \tfrac{\partial f}{\partial x}\Delta t$ — first-order
discretisation of the error-state transition. The continuous Jacobians
$\tfrac{\partial f}{\partial x}$ (`df_dx`) and $\tfrac{\partial f}{\partial w}$
(`df_dw`) are spelled out in `use-ikfom.hpp:61`–`88`; the load-bearing blocks
match §04 exactly:

$$
\frac{\partial \dot v}{\partial \delta R} = -R\,(a_m-b_a)^\wedge
\;\;\texttt{(df\_dx:69)},\qquad
\frac{\partial \dot v}{\partial b_a} = -R \;\;\texttt{(df\_dx:70)},\qquad
\frac{\partial \dot R}{\partial b_g} = -I \;\;\texttt{(df\_dx:75)},
$$

and the noise mapping $\partial\dot v/\partial n_a=-R$,
$\partial\dot R/\partial n_g=-I$, $\partial\dot b_g/\partial n_{bg}=I$,
$\partial\dot b_a/\partial n_{ba}=I$ (`df_dw:83`–`86`). $Q$ is the diagonal IMU
noise, filled just before the call from the `gyr_cov`, `acc_cov`, `b_gyr_cov`,
`b_acc_cov` params (`IMU_Processing.hpp:280`–`283`; params at
`laserMapping.cpp:777`–`780`).

After the loop a final partial step propagates to the exact sweep end
(`dt = note*(pcl_end_time - imu_end_time); kf_state.predict(dt, Q, in);`,
`IMU_Processing.hpp:299`–`301`). At this point $x$ is the **prior mean
$\hat x$** and $\Sigma$ the **prior covariance** for the iterated update — i.e. the
prediction $h_{\text{prior}}(x)$ around which §09's iEKF linearises. Back in
`main()`, `state_point = kf.get_x()` reads it (`laserMapping.cpp:889`).

> **Debug hook (carry into Meridian).** `main()` logs the full predicted state to
> `fout_pre` (`laserMapping.cpp:939`–`940`: euler, pos, extrinsic, vel, bg, ba,
> grav). Meridian's L2 should publish this every step. Seeing the *prior* before the
> *posterior* is the single most useful introspection for diagnosing divergence,
> because it separates "IMU drift" from "bad LiDAR association."

---

### 12.4 Stage 2 — motion compensation (deskew)

A spinning / solid-state LiDAR samples points at different times across one
sweep, so each was observed from a slightly different pose. §05 said we must warp
every point into one reference frame (here, the sweep-end IMU frame) before
association. FAST-LIO2 does a **backward pass** over the saved propagation poses
`IMUpose` (pushed during forward integration at `IMU_Processing.hpp:295`).

For a point with relative time $\tau$, the code reconstructs the pose at $\tau$ by
a short constant-rate extrapolation from the bracketing IMU pose
(`IMU_Processing.hpp:331`, `:334`):

$$
R_i = R_{\text{imu}}\,\mathrm{Exp}(\bar\omega\,\Delta t),\qquad
T_{ei} = p_{\text{imu}} + v_{\text{imu}}\,\Delta t
        + \tfrac12 a_{\text{imu}}\,\Delta t^2 - {}^{W}p_e,
$$

then transforms the point into the sweep-end LiDAR frame
(`IMU_Processing.hpp:335`):

$$
p_L^{\text{comp}} =
{}^{I}R_L^{\top}\!\Big({}^{W}\!R_e^{\top}\big(R_i({}^{I}R_L\,p_L + {}^{I}t_L)+T_{ei}\big) - {}^{I}t_L\Big),
$$

shipped as
`P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i*(imu_state.offset_R_L_I*P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I)`,
where $({}^{W}\!R_e,{}^{W}p_e)$ is the sweep-end pose (`imu_state`). The inline
comment `// not accurate!` (`IMU_Processing.hpp:335`) is itself instructive: only
rotation + IMU-position is compensated, the small-approximation discussed in §05.
The output overwrites the point in place (`it_pcl->x/y/z = P_compensate(...)`,
`:338`–`:340`).

The result, `feats_undistort`, is one rigid scan as if captured instantaneously
at the sweep end — the cloud the rest of the step treats as $\{p_L\}$.

> **CT swap point (§10, Phase 5 of Meridian's plan).** This `R_i,T_ei`
> extrapolation is exactly where a continuous-time front-end differs: instead of
> piecewise-constant-rate extrapolation between IMU poses, §10's B-spline
> $T(\tau)$ is evaluated analytically at each point's $\tau$. Because the deskew is
> already parameterised by per-point $\tau$, the interface boundary is clean —
> which is why Meridian keeps the front-end behind `IFrontEnd`.

---

### 12.5 Stage 3 — map management and downsampling

Back in `main()`, three cheap-but-essential operations bracket the update:

1. **`lasermap_fov_segment()`** (`laserMapping.cpp:231`, called at `:901`) keeps
   the ikd-Tree's spatial extent centred on the sensor: when the FOV cube would
   leave the local map it slides the cube and deletes far boxes
   (`ikdtree.Delete_Point_Boxes(cub_needrm)`, `:275`). This bounds memory and
   query cost — the "local map" of §05, the moving-window map of the FAST-LIO2
   paper (`papers/2107.06829.txt`, §III).
2. **`downSizeFilterSurf`** voxel-downsamples the deskewed scan to
   `feats_down_body` (`laserMapping.cpp:904`–`905`) at `filter_size_surf`
   resolution. `feats_down_size` (`:907`) is the number of residuals we will build
   — the effective measurement count before association gating.
3. **First-frame bootstrap**: if the ikd-Tree is empty, build it from the first
   downsampled world-frame scan and `continue` (`laserMapping.cpp:909`–`922`):
   nothing to register against yet. A scan with `feats_down_size < 5` is also
   skipped (`:929`).

For Meridian's layered map (L4) this is the **registration layer** (adaptive
voxel-hash replacing ikd-Tree); the TSDF/mesh layers sit *downstream* and never
participate in the estimator's inner loop — a separation FAST-LIO2 enforces by
having `h_share_model` query only the ikd-Tree.

---

### 12.6 Stage 4 — the iterated update (the heart of the step)

`kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time)`
(`laserMapping.cpp:960`; body `esekfom.hpp:1619`) is the iterated-EKF update of
§09. "Dyn-share" means the user supplies the measurement model *and its Jacobian*
through one callback, `h_share_model` (`laserMapping.cpp:638`), registered via
`kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi)`
(`laserMapping.cpp:828`). The toolkit loops `for(int i=-1; i<maximum_iter; i++)`
(`esekfom.hpp:1633`); `maximum_iter` is the `max_iteration` param (default 4,
`laserMapping.cpp:765`). Below is one iteration, mapping §05/§08/§09 onto code.

#### (a) Data association + residual — `h_share_model`, first loop

For each downsampled point $p_L$ (`feats_down_body`, `laserMapping.cpp:650`):

1. **Transform to world** with the *current iterate* $x=(R,p,{}^{I}R_L,{}^{I}t_L)$
   (`laserMapping.cpp:657`):
   $$ p_W = R\,({}^{I}R_L\,p_L + {}^{I}t_L) + p . $$
2. **k-NN search** the ikd-Tree for `NUM_MATCH_POINTS` (=5) nearest map points,
   **only when `ekfom_data.converge` is true** (`laserMapping.cpp:667`–`672`).
   This is the efficiency trick the paper states and the code makes literal:
   re-associate only when the previous iteration's $\delta x$ was small; while the
   state still moves a lot, reuse the existing neighbours. Validity gate: reject if
   fewer than 5 neighbours or the 5th is farther than 5 m
   (`pointSearchSqDis[NUM_MATCH_POINTS-1] > 5`, `:671`).
3. **Plane fit** via `esti_plane(pabcd, points_near, 0.1f)`
   (`laserMapping.cpp:678`): least-squares fit of $n\cdot x + d = 0$ to the 5
   neighbours (the $(n,d)$ of §05), threshold $0.1$ m. `pabcd = [n_x,n_y,n_z,d]`.
4. **Point-to-plane residual** (`laserMapping.cpp:680`):
   $$ r = n\cdot p_W + d
        = \texttt{pabcd}(0)\,p_W.x+\texttt{pabcd}(1)\,p_W.y+\texttt{pabcd}(2)\,p_W.z+\texttt{pabcd}(3), $$
   stored as `pd2` — §05's signed point-to-plane distance.
5. **Robust validity weight** (`laserMapping.cpp:681`):
   $$ s = 1 - 0.9\,\frac{|r|}{\sqrt{\lVert p_L\rVert}},\qquad \text{accept iff } s>0.9, $$
   a range-aware gate (a far point's plane fit is trusted less) — a *hard* robust
   kernel, the simplest member of the §11 robust-loss family. Accepted points set
   `point_selected_surf[i]=true`, store the normal in `normvec`, keep
   `res_last[i]=|r|` (`:685`–`690`).
6. **Compaction**: a second loop packs accepted points into `laserCloudOri` /
   `corr_normvect` and counts `effct_feat_num` (`laserMapping.cpp:697`–`706`). If
   `effct_feat_num < 1`, the update is flagged invalid (`ekfom_data.valid=false`,
   `:710`) — the degeneracy escape hatch (§11). The mean residual `res_mean_last`
   (`:715`) is the headline health number to publish.

#### (b) Measurement Jacobian — `h_share_model`, second loop

For each effective point the code builds the $1\times 12$ row of $H$ (12 =
$\{\delta R(3),\delta p(3),\delta{}^{I}R_L(3),\delta{}^{I}t_L(3)\}$; bias,
velocity, gravity columns are zero for the LiDAR residual, so $H$ is declared
`MatrixXd::Zero(effct_feat_num, 12)`, `laserMapping.cpp:720`). With
$p_{Le} = {}^{I}R_L\,p_L + {}^{I}t_L$ (point in IMU frame, `point_this`,
`laserMapping.cpp:729`) and world normal $n$ (`norm_vec`, `:735`), the shipped
pieces are (`laserMapping.cpp:738`–`743`):

```cpp
V3D C(s.rot.conjugate() * norm_vec);                       // C = R^T n
V3D A(point_crossmat * C);                                 // A = [p_Le]^x R^T n
V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // extrinsic-rot term
ekfom_data.h_x.block<1,12>(i,0)
    << norm_p.x, norm_p.y, norm_p.z,   // ∂r/∂p  = n^T
       VEC_FROM_ARRAY(A),              // ∂r/∂R  (rotation block)
       VEC_FROM_ARRAY(B),              // ∂r/∂(R_L)
       VEC_FROM_ARRAY(C);             // ∂r/∂(t_L) = n^T R
```

These are §05's derivatives of $r = n\cdot(R\,p_{Le}+p)+d$ on the manifold with
the right-perturbation convention of §02:

$$
\frac{\partial r}{\partial \delta p} = n^{\top},\qquad
\frac{\partial r}{\partial \delta R}
   = -\,n^{\top} R\,(p_{Le})^{\wedge}
   = \big((p_{Le})^{\wedge} R^{\top} n\big)^{\top} = A^{\top},\qquad
\frac{\partial r}{\partial \delta\, {}^{I}t_L} = n^{\top} R = C^{\top}.
$$

(Sign conventions are absorbed by IKFoM's $\boxplus$ definition; what matters is
the column placement.) When `extrinsic_est_en` is false the $\{R_L,t_L\}$ columns
are hard-zeroed (`laserMapping.cpp:747`), i.e. the extrinsic is *frozen* — Meridian's
"online extrinsic refinement as graph variables" is the same idea promoted from
these EKF columns to L3 graph variables. The measurement entry is
`ekfom_data.h(i) = -norm_p.intensity = -r` (`laserMapping.cpp:751`), the negative
residual the toolkit minimises.

#### (c) Solving the normal equations and the $\boxplus$ correction

Inside `update_iterated_dyn_share_modified` the per-iteration solve is the
§08/§09 normal-equation step. With $R=\texttt{LASER\_POINT\_COV}$ the scalar
measurement variance ($=0.001$, `laserMapping.cpp:64`), the toolkit forms (the
"measurement dimension $\ge$ state" branch, `esekfom.hpp:1781`–`1810`):

```cpp
cov P_temp = (P_/R).inverse();                       // = Σ^{-1} · R   (prior info)
Eigen::Matrix<scalar_type,12,12> HTH = h_x_.transpose()*h_x_;  // Hᵀ R⁻¹ H  (×R)
P_temp.block<12,12>(0,0) += HTH;                     // (Σ⁻¹ + Hᵀ R⁻¹ H)·R
cov P_inv = P_temp.inverse();
K_h = P_inv.block<n,12>(0,0) * h_x_.transpose() * dyn_share.h;   // = K·(-r)
K_x.block<n,12>(0,0) = P_inv.block<n,12>(0,0) * HTH;            // = K·H
Matrix<scalar_type,n,1> dx_ = K_h + (K_x - I) * dx_new;          // esekfom.hpp:1815
x_.boxplus(dx_);                                                 // esekfom.hpp:1817
```

which is precisely

$$
\big(H^{\top} R^{-1} H + \Sigma^{-1}\big)\,\delta x
  = H^{\top} R^{-1}\,(-r) - \Sigma^{-1}\,(x_i \boxminus \hat x),
\qquad x_{i+1} = x_i \boxplus \delta x .
$$

The term $(x_i \boxminus \hat x)$ is `x_.boxminus(dx, x_propagated)`
(`esekfom.hpp:1652`) — the running offset from the prior mean, the prior-pull of
the iterated form. Crucially the toolkit **never forms a full $23\times 23$
measurement system**: `HTH` is $12\times 12$ and is added only to the top-left
$12\times 12$ block (`esekfom.hpp:1785`), the §08 sparsity / partitioning idea in
miniature (LiDAR touches only the geometric columns). Convergence is the
`fabs(dx_[i]) > limit[i]` test over all $n$ components (`esekfom.hpp:1819`–`1826`,
with `epsi=0.001` set at `laserMapping.cpp:826`–`827`); the resulting flag is the
`ekfom_data.converge` that gates re-association in (a). On the final iteration the
posterior covariance is rebuilt, $\Sigma \leftarrow (I-KH)\,\Sigma$ with the SO(3)
projection re-applied (`esekfom.hpp:1834`+, `L_ = P_;` then the
$\mathrm{A\_matrix}$ correction loops).

This is the concrete realisation of §09's theorem: **the converged iterated-EKF
fixed point equals the Gauss–Newton MAP estimate** of the factor graph
{IMU-prior factor ⊕ all point-to-plane factors}. FAST-LIO2 just runs that GN to a
handful of iterations recursively rather than as a sliding-window batch.

After the call, `main()` reads back the posterior (`state_point = kf.get_x()`,
`laserMapping.cpp:961`) and converts to Euler/quaternion for publishing
(`:962`–`:967`).

> **Observability / degeneracy (§11) — where Meridian diverges deliberately.**
> FAST-LIO2's only degeneracy guards are `effct_feat_num < 1` and the per-point
> $s$-gate. Meridian's L2 must instead inspect the eigen-structure of
> $H^{\top}R^{-1}H$ (the `HTH` matrix above, `esekfom.hpp:1784`) per axis (X-ICP /
> D2-LIO style) and feed weak directions as inflated noise into the L3 back-end.
> The clean place to compute it is right where `HTH` is built, before the inverse
> — that information matrix is already in hand.

---

### 12.7 Stage 4′ — adding the visual residual: FAST-LIVO2

FAST-LIVO2 (`FAST-LIVO2/src/LIVMapper.cpp`, with `vio.cpp`, `voxel_map.cpp`,
`visual_point.cpp`, `frame.cpp`) keeps the *entire* skeleton above and changes one
thing: at a frame it runs **two** iterated updates against the same state — a LIO
update (LiDAR point-to-plane) and a **VIO update** (sparse-direct photometric).
The main loop `LIVMapper::run()` (`LIVMapper.cpp:534`) is the FAST-LIO2 `main`
loop's twin: `sync_packages(LidarMeasures)` → `handleFirstFrame()` →
`processImu()` → `stateEstimationAndMapping()` (`:540`–`:551`). The IMU stage
`processImu()` (`:248`) forward-propagates and snapshots the prior into
`state_propagat` (`:256`) — the same predict-then-correct structure as §12.3.
`stateEstimationAndMapping()` (`:267`) then *dispatches on the packet type*
`LidarMeasures.lio_vio_flg`: a `VIO` packet routes to `handleVIO()` (`:281`), a
`LIO`/`LO` packet to `handleLIO()` (`:336`). So the estimator step is
"propagate → deskew → {LIO update **or** VIO update for this packet} → map
update," with consecutive packets alternating modality, all reusing the one
error-state (`_state`, a `StatesGroup`), one covariance, one $\boxplus$ — the
textbook tightly-coupled fusion of §01/§03 with two heterogeneous residuals
folded into the same MAP problem. The LIO update is delegated to
`voxelmap_manager->StateEstimation(state_propagat)` (`handleLIO`, `:370`) — the
voxel-map analogue of §12.6's iterated point-to-plane solve, started from the
propagated prior; the VIO update is delegated to
`vio_manager->processFrame(img, _pv_list, voxel_map_, …)` (`handleVIO`, `:305`).

The **photometric residual** is §06's: for a map point $P_W$ (a sparse "visual
point" with an attached reference patch), project into the current image and
compare intensities,

$$
r_{\text{photo}} = I\big(\pi(K_{\text{cam}},\, {}^{C}R_W(P_W - {}^{C}t_W))\big) - I_{\text{ref}},
$$

with an affine photometric correction over a *patch* (not a single pixel). The
Meridian-relevant point: FAST-LIVO2 obtains the **depth of visual points from the
LiDAR map**, not from triangulation — so the visual residual is *direct* and
*depth-known*, removing the scale/initialisation fragility of pure VIO. That is
exactly why we chose this style for L2 (§06). The voxel / visual-point map is
maintained in `voxel_map.cpp` / `visual_point.cpp` and is shared with the LiDAR
layer, so a single update of $x$ improves both modalities' future associations.

Two structural contracts to confirm against the source before extending Meridian
(stated at the level the code enforces them):

- **Shared state, separate per-modality updates.** `handleLIO` and `handleVIO`
  both correct the *same* `_state`/`StatesGroup` and the same covariance; they are
  not stacked into one big $H$ but dispatched per packet (`stateEstimationAnd-
  Mapping`'s `switch`, `:269`). Algebraically, for Gaussian factors, sequential
  iterated updates and a single stacked update converge to the same MAP point
  (§03/§09), and the per-modality form is simpler to make modular: each sensor
  owns its own state-estimation entry point (`voxelmap_manager->StateEstimation`
  for LiDAR, `vio_manager->processFrame` for the camera). This is the pattern
  Meridian's `IFrontEnd` should expose — a list of residual providers each
  contributing $(r,H,R)$ against one shared state.
- **One IMU pipeline, many consumers.** Both updates start from the *same*
  propagated prior `state_propagat` (snapshotted once in `processImu`, `:256`) and
  consume the *same* deskewed geometry; the IMU processing is unchanged in spirit
  from FAST-LIO. This confirms the L0/L1↔L2 boundary: preprocessing and
  propagation are sensor-generic, only the residual is sensor-specific.

For Meridian's phasing this is the literal justification for "build the iEKF v1
first": FAST-LIO2 and FAST-LIVO2 are the *same estimator* with one vs. two
residual callbacks. Once the LiDAR callback and the iterated-solve harness exist,
adding the camera (Phase 6 sensor extension) is "register another residual
provider," not "rewrite the estimator." GNSS (§07) slots in identically as a third
provider with an absolute-position residual and a switchable constraint — though
in Meridian GNSS lives more naturally in the L3 graph (§07/§11, PCM), the front-end
interface is the same shape.

---

### 12.8 Stage 5 — committing the scan to the map

With the posterior $x$ in hand, `map_incremental()` (`laserMapping.cpp:427`,
called at `:976`) inserts the now-correctly-registered scan into the ikd-Tree. The
interesting part is that it is **downsample-aware**: for each new world point it
computes the centre of the voxel it falls in (`mid_point`,
`laserMapping.cpp:444`–`446`) and only adds the point if no existing neighbour is
already closer to that voxel centre (`:452`–`:460`), pushing "needs downsample" and
"no downsample" points to separate `Add_Points` calls (`:470`–`:471`). This keeps
map density bounded — the incremental analogue of §05's voxel-grid map, and the
reason the ikd-Tree paper (`papers/2102.10808.txt`) emphasises incremental,
balance-maintaining insertion/deletion over periodic rebuilds.

For Meridian's L4 this insertion is the **registration-layer** update only. The TSDF
integration (NVBlox) and mesh extraction happen on a *separate* cadence from the
posterior pose plus the retained per-keyframe cloud, so the estimator's hot loop
is never blocked by surface reconstruction — a separation FAST-LIO2 enforces
simply by not having a TSDF in the loop at all.

---

### 12.9 Stage 6 — outputs and introspection (don't skip this)

Finally `main()` publishes (`laserMapping.cpp:972`–`982`): `publish_odometry`
(`/Odometry`, with the pose covariance copied out of $\Sigma$,
`laserMapping.cpp:596`–`606`), `publish_path` (`/path`), `publish_frame_world`
(`/cloud_registered`), `publish_frame_body` (`/cloud_registered_body`). The
advertised but commented-out `publish_effect_world` (`/cloud_effected`,
`laserMapping.cpp:551`, `:983`) is the *effective points actually used* in the
update — the one most worth re-enabling. These, plus the staged timers
`t0..t5` / `match_time` / `solve_time` / `solve_H_time` (`laserMapping.cpp:879`,
`:958`–`:959`, `:991`–`:996`) and the per-step state logs `fout_pre` / `fout_out`
(`:939`, `:1011`), are the model for Meridian's "debug in the right places"
principle:

| What to expose | FAST-LIO2 source | Meridian L2 debug topic |
|---|---|---|
| prior state (pre-update) | `fout_pre`, `laserMapping.cpp:939` | `~/debug/state_prior` |
| posterior state | `kf.get_x()`, `:961` | `~/debug/state_posterior` |
| residual health (`res_mean_last`, `effct_feat_num`) | `h_share_model`, `:715`,`:695` | `~/debug/residuals` |
| effective points | `/cloud_effected`, `:551` | `~/debug/cloud_effected` (rviz) |
| pose covariance | `kf.get_P()`, `:596` | `~/debug/pose_cov` |
| per-stage timing | `t0..t5`, `match_time`, `solve_time`, `:991`–`996` | `~/debug/timing` |
| observability (Meridian-new) | — (compute from `HTH`, `esekfom.hpp:1784`) | `~/debug/observability` |

A developer who can see the prior, the posterior, the effective points, the mean
residual, and the per-stage timing can *watch the estimator think* — which is the
whole reason §11's robustness machinery (degeneracy detection, GNC, PCM) has
somewhere to plug in.

---

### 12.10 The step, reassembled (and how every section maps to it)

One estimator step is the following composition, each arrow a function you can
open in the reference code:

$$
x_{k}\;\xrightarrow[\text{IMU, §04}]{\texttt{predict}}\;\hat x,\Sigma
\;\xrightarrow[\text{§05}]{\texttt{deskew + downsample}}\;\{p_L\}
\;\xrightarrow[\text{§08/09}]{\texttt{update\_iterated\_dyn\_share}}\;x_{k}^{+}
\;\xrightarrow[\text{§05}]{\texttt{map\_incremental}}\;\mathcal M^{+}.
$$

- **§01 (coupling spectrum):** the single shared $x,\Sigma$ updated by all
  residuals is the *tight* end — one `update_iterated_dyn_share_modified` call, not
  a cascade of filters.
- **§02 (manifolds):** every correction is $x\boxplus\delta x$
  (`esekfom.hpp:287`,`:1817`), every error is $x_i\boxminus\hat x$
  (`:1652`); the SO(3) exponential appears in propagation (`MTK::exp`,
  `:312`) and deskew (`Exp(angvel,dt)`, `IMU_Processing.hpp:331`). State layout
  from `MTK_BUILD_MANIFOLD` (`use-ikfom.hpp:12`).
- **§03 (MAP / factor graph):** the per-step problem is {IMU-prior ⊕
  point-to-plane (⊕ photometric)}; the iEKF computes its MAP point.
- **§04 (IMU):** `get_f` dynamics (`use-ikfom.hpp:47`) + `predict` covariance prop
  (`esekfom.hpp:380`–`381`) = the inertial prior; input-model integration rather
  than explicit preintegration (equivalent within the frame).
- **§05 (LiDAR):** `h_share_model` = association (ikd-Tree kNN), plane fit
  (`esti_plane`), residual $n\cdot p_W + d$, Jacobian $A,B,C$
  (`laserMapping.cpp:638`).
- **§06 (visual):** the FAST-LIVO2 VIO update — same harness, photometric residual
  with LiDAR-provided depth.
- **§07 (GNSS):** an additional absolute residual provider (in Meridian, mainly an L3
  factor) using the same $(r,H,R)$ contract.
- **§08 (batch GN):** the normal equations
  $(H^\top R^{-1}H+\Sigma^{-1})\,\delta x=\cdots$ solved per iteration, exploiting
  $H$'s 12 nonzero columns (`HTH` is $12\times 12$, `esekfom.hpp:1784`–`1785`).
- **§09 (iEKF):** the `for(i=-1; i<maximum_iter; i++)` loop with re-association
  gating (`:1633`, `:667`); its fixed point = §08's GN solution.
- **§10 (CT):** the per-point $\tau$ deskew is the seam where a B-spline $T(\tau)$
  replaces piecewise-constant extrapolation — clean because of `IFrontEnd`.
- **§11 (robustness):** the $s$-gate and `effct_feat_num` check are the shipped
  guards; Meridian adds per-axis observability from the very `HTH` this step builds,
  plus GNC/PCM in L3.

That is the whole chapter, executing once. Everything earlier was a derivation of
one of these arrows; everything later in Meridian's build plan (back-end graph, loop
closure, mesh, robustness, CT swap, sensor extensions) either consumes the
keyframes this step emits or replaces one arrow with a more capable implementation
behind the same interface.

---

> **Source-fidelity note.** All line citations into
> `FAST_LIO/src/laserMapping.cpp`, `FAST_LIO/src/IMU_Processing.hpp`,
> `FAST_LIO/include/use-ikfom.hpp`, and
> `FAST_LIO/include/IKFoM_toolkit/esekfom/esekfom.hpp` (`predict` at line 279,
> `update_iterated_dyn_share_modified` at line 1619, the `HTH`/`boxplus`/`boxminus`
> markers, `get_f`/`df_dx`/`df_dw`) were read verbatim from the on-disk files and
> are exact. The FAST-LIVO2 *control flow* in §12.7 — `run()` (line 534),
> `processImu()` (248, `state_propagat` at 256), `stateEstimationAndMapping()`
> dispatch (267–279), `handleVIO`/`processFrame` (281, 305),
> `handleLIO`/`StateEstimation` (336, 370) — was likewise read verbatim and is
> exact. The *internals* of the photometric residual (patch projection, affine
> exposure, depth-from-LiDAR) live in `vio.cpp` / `visual_point.cpp` / `frame.cpp`,
> and the voxel point-to-plane internals in `voxel_map.cpp`; those were referenced
> at file granularity, so re-open them and the paper `papers/2408.14035.txt` before
> copying their exact line numbers into derived Meridian code. None of the equations
> above depend on those internals — they are grounded in the verbatim FAST-LIO
> residual/Jacobian/propagation code and the FAST-LIO2 paper.
