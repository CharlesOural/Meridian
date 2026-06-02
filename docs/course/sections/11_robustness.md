## 11. Robustness: degeneracy/observability, robust kernels, GNC, switchable, PCM

> **Where this sits.** Sections 04–07 built the individual residuals (IMU, LiDAR point-to-plane, sparse-direct photometric, GNSS); Section 08 turned a stack of residuals into a Gauss–Newton/Levenberg–Marquardt normal-equation solve; Section 09 showed the recursive (iterated-EKF / fixed-lag) form and its equivalence to the batch MAP; Section 10 lifted all of this onto a continuous-time B-spline trajectory. **Every one of those steps assumed the data was good.** This section removes that assumption. We treat two distinct failure modes that a tactical multi-sensor SLAM must survive:
>
> 1. **The geometry is uninformative** — the scene does not constrain some directions of motion (a tunnel, a long featureless corridor, a flat field seen by the dome LiDAR, a wall filling the camera). The residuals are individually *correct* but collectively *rank-deficient*. This is **degeneracy / observability**, and it is a property of the **Hessian**.
> 2. **The data is wrong** — a measurement is associated to the wrong plane, a pixel lands on a moving object or an occlusion edge, a GNSS fix is multipath-corrupted, a loop closure is a perceptual alias. The residual is large not because the state is wrong but because the *model* is wrong for that datum. This is **outlier rejection**, handled by **robust kernels (M-estimation)**, **Graduated Non-Convexity (GNC)**, **switchable constraints**, and — for loop closures specifically — **Pairwise Consistency Maximization (PCM)**.
>
> The unifying object is the cost from Section 03/08:
> $$ \mathcal{C}(x) \;=\; \tfrac12\,\big\lVert x \boxminus \hat{x}_0 \big\rVert^2_{\Sigma_{\text{prior}}^{-1}} \;+\; \tfrac12 \sum_k \rho_k\!\Big( \big\lVert r_k(x) \big\rVert^2_{\Sigma_k^{-1}} \Big), $$
> where the prior term is the IMU forward-propagation prior of Section 04/09. Degeneracy is about the **curvature** of $\mathcal{C}$ (its Hessian $\mathbf{H}$); outliers are about the **shape of each $\rho_k$**. We will see that these are not independent: a robust kernel that down-weights inliers in a poorly-observed direction can *manufacture* degeneracy, so the two must be designed together.

---

### 11.1 Degeneracy and observability: it lives in the Hessian eigen-spectrum

#### 11.1.1 The geometric setup

Recall the LiDAR point-to-plane residual (Section 05). For the $j$-th source point $p_{L,j}$ associated to a map plane with unit normal $n_j$ passing through point $q_j$, with the body-to-world pose $T = (R,p) \in SE(3)$,
$$ r_j(x) \;=\; n_j^\top\!\big( R\,(R_{LI}\,p_{L,j} + p_{LI}) + p \;-\; q_j \big), \qquad r_j \in \mathbb{R}. $$
This is exactly the scalar residual computed in FAST-LIO's measurement model: the predicted signed distance `pd2 = pabcd(0)*point_world.x + pabcd(1)*point_world.y + pabcd(2)*point_world.z + pabcd(3)` and the residual fed to the filter is `ekfom_data.h(i) = -norm_p.intensity` (the stored `pd2`) — see `FAST_LIO/src/laserMapping.cpp:680` and `:751`. The plane $(n,d)$ itself comes from the least-squares fit `esti_plane` (`FAST_LIO/include/common_lib.h:226`), which solves $A\,n = -\mathbf{1}$ by `colPivHouseholderQr` (`:241`) and rejects the fit if any of the `NUM_MATCH_POINTS` points deviates from the plane by more than a planarity threshold (`:251`).

Linearising $r_j$ about the current estimate gives the row Jacobian $H_j = \partial r_j / \partial \delta x \in \mathbb{R}^{1\times \dim x}$. Stacking the informative state block (rotation $\delta\theta$, position $\delta p$, and optionally the LiDAR–IMU extrinsic), FAST-LIO writes each row as
$$ H_j \;=\; \big[\; \underbrace{n_j^\top}_{\partial/\partial p}\;\; \underbrace{(\,[p_j]_\times R^\top n_j)^\top}_{\partial/\partial \delta\theta}\;\; \cdots \big], $$
which is the literal line `ekfom_data.h_x.block<1,12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), ...` with `C = s.rot.conjugate()*norm_vec` and `A = point_crossmat * C` (`laserMapping.cpp:738-743`). The crucial structural fact for degeneracy: **the position part of every LiDAR row is just the plane normal $n_j$.** Therefore the position-block of the Gauss–Newton Hessian is
$$ \mathbf{H}_{pp} \;=\; \sum_j \frac{1}{\sigma_j^2}\, n_j\, n_j^\top \;\in\; \mathbb{R}^{3\times3}. $$
This is a sum of rank-1 outer products of plane normals. **If all observed planes share a normal direction** — e.g. a long straight corridor where every wall normal is horizontal and perpendicular to the corridor axis — then $\mathbf{H}_{pp}$ has (near-)zero eigenvalues along the unconstrained axes (the corridor's length direction and possibly height). Translation along the corridor is **unobservable from LiDAR**. This is precisely the Zhang–Singh 2016 picture: degeneracy is rank-deficiency of the registration optimisation, diagnosed by the eigenvalues of the optimisation matrix (their "degeneracy factor" = the smallest eigenvalue of $\mathbf{H}$).

#### 11.1.2 The full Hessian and where the prior saves us

In the tightly-coupled estimator the per-step normal equations are not just LiDAR. From Sections 08–09, the iterated-EKF update FAST-LIO actually forms in `update_iterated_dyn_share_modified` (the routine `laserMapping.cpp:960` calls), in the branch that solves the state dimension directly, is
$$ \mathbf{H}_{\text{post}} \;=\; \underbrace{(P/R)^{-1}}_{\text{IMU prior, full rank}} \;+\; \underbrace{H^\top H}_{\text{LiDAR, possibly rank-deficient}}, \qquad K_x = \mathbf{H}_{\text{post}}^{-1}\big|_{\,\cdot,\,1:12}\; H^\top H, $$
matching the code verbatim (`esekfom.hpp:1782-1809`):
```cpp
cov P_temp = (P_/R).inverse();
Eigen::Matrix<scalar_type,12,12> HTH = h_x_.transpose() * h_x_;   // H^T H; scalar R folded into P_temp
P_temp.template block<12,12>(0,0) += HTH;
cov P_inv = P_temp.inverse();
K_h = P_inv.template block<n,12>(0,0) * h_x_.transpose() * dyn_share.h;  // K * residual
K_x.template block<n,12>(0,0) = P_inv.template block<n,12>(0,0) * HTH;
```
Here $R$ is the scalar LiDAR measurement variance (`LASER_POINT_COV = 0.001`, `laserMapping.cpp:64`, passed at `:960`) and $H$ is the stacked `h_x` over all effective points (`effct_feat_num`); folding $R$ into the prior as $(P/R)^{-1}$ is algebraically identical to the information form $H^\top R^{-1} H + P^{-1}$ above. **This is the single most important robustness mechanism in a tightly-coupled LIO and it is *structural*, not a heuristic.** Because $P^{-1}$ is full rank (the IMU constrains all six DOF over the propagation interval), $\mathbf{H}_{\text{post}}$ is invertible *even when $H^\top H$ is rank-2*. In the unobservable corridor direction the update simply **falls back to the IMU prediction**: the gain in that direction is (near-)zero, so $\delta x$ there is whatever the IMU said, and the covariance in that direction stays large.

It is worth being precise about what the reference systems do and do not do. The FAST-LIO2 paper contains **no explicit degeneracy handling** — the word "degenerate" does not appear in it (2107.06829) — and the robustness to degeneracy is *entirely* this silent Bayesian regularization. The paper does cite the loosely-coupled precedent: LION's "observability awareness check ... to lower the point number and hence save the computation" in vision-denied environments (2107.06829, line 26). FAST-LIVO2, by contrast, is built and tested explicitly for degeneracy: it documents "Long Corridor" and single-wall sequences where "due to the absence of geometrical constraints since only one wall plane is being observed by the LiDAR, LIO methods would fail" (2408.14035, line 849), and survives them by adding the *visual* constraint (§11.2.3).

> **Worked micro-example (corridor).** Suppose after a scan, the LiDAR information $H^\top H/R$ has eigenvalues $\{4000, 3800, 5\}$ in the position block (m$^{-2}$), the small one along the corridor axis $\hat{a}$. The IMU prior over a 100 ms interval might give $P^{-1}_{pp}\approx \text{diag}(50,50,50)$ m$^{-2}$ (decent, because integrated acceleration is informative). Then $\mathbf{H}_{\text{post}}$ has eigenvalues $\approx\{4050,3850,55\}$. Translation along $\hat a$ is estimated with std $\approx 1/\sqrt{55}\approx 0.13$ m — dominated by inertial dead-reckoning — while the cross-corridor directions have std $\approx 0.016$ m. The filter does not crash or "snap"; it gracefully *degrades to inertial odometry along the degenerate axis*. This is the behaviour we want, and it is automatic.

#### 11.1.3 Per-axis observability scoring (not a binary flag)

The Bayesian fallback above prevents a singular solve, but it does **not** tell the back-end (Section 03/09 factor graph, GTSAM iSAM2) how much to *trust* the resulting keyframe pose along each axis. A keyframe born in a corridor has an honest, anisotropic uncertainty: tight across, loose along. Meridian's design principle is that **observability must flow into the back-end as a noise model, not as a binary "skip this constraint" flag.** A binary flag throws away a perfectly good 2-DOF constraint; an anisotropic covariance keeps the informative directions and lets loop closures / GNSS fill in the rest.

The eigen-decomposition gives us exactly the right object. Diagonalise the per-keyframe information block
$$ \mathbf{H} \;=\; U\,\Lambda\,U^\top, \qquad \Lambda = \operatorname{diag}(\lambda_1 \ge \dots \ge \lambda_d), $$
with $\lambda_i$ the curvature and $u_i$ the direction. We define a **continuous per-axis observability score** rather than a threshold. Three families, in increasing sophistication:

1. **Zhang–Singh degeneracy factor (absolute).** $\;o_i = \lambda_i$ directly, compared against a physical floor $\lambda_{\min}$. Their original test is "if $\lambda_i < \lambda_{\min}$, the $i$-th eigen-direction is degenerate; project the update to remove it." We *keep* the value rather than thresholding: the publishable scalar is $\lambda_{\min}(\mathbf{H})$, the worst-observed direction.
2. **Condition-number / relative score.** $\;o_i = \lambda_i / \lambda_1 \in (0,1]$. This is scale-invariant and what an operator overlay can colour-map directly (Section 06 / L6 confidence overlay): green where $o_i\to 1$, red where $o_i\to 0$.
3. **X-ICP localizability (directional, sample-based).** X-ICP (Tuna et al.) argues the raw eigenvalue is contaminated by the *number* of points and noise scale, so it instead samples, per eigen-direction $u_i$, the **contribution strength** of each point's constraint and classifies the direction as *localizable / partially / non-localizable*. Concretely, for direction $u_i$ it forms the projected constraint contributions $\{(H_j u_i)\}_j$ and looks at how many points contribute meaningfully and how aligned they are, yielding a normalized score in $[0,1]$ per axis. The D²-LIO line of work uses a similar per-axis observability derived from the LiDAR Hessian to *blend* the LiDAR update with the inertial prediction direction-by-direction.

For Meridian we adopt a **soft, eigen-spectrum-based information injection**. Let $\Sigma_{kf} = \mathbf{H}_{\text{post}}^{-1}$ be the raw posterior covariance of a keyframe pose. Form its eigen-decomposition $\Sigma_{kf} = U \Sigma_\lambda U^\top$ and **inflate the weakly-observed directions** before handing the constraint to the back-end:
$$ \tilde{\Sigma}_{kf} \;=\; U\, \operatorname{diag}\!\Big( \sigma_i^2 \big/ g(o_i) \Big)\, U^\top, \qquad g(o) = \min\!\Big(1,\; \tfrac{o}{o_\star}\Big)^{\gamma}, $$
where $o_i$ is the relative observability of direction $i$, $o_\star$ a target (e.g. 0.1) and $\gamma\!\in\![1,2]$ controls how aggressively under-observed axes are deflated. $g(o)\to 1$ leaves well-observed axes untouched; $g(o)\to 0$ blows up the variance along the corridor axis so the back-end barely trusts that DOF of the odometry factor. **Nothing is discarded; the constraint is simply honest about which directions it knows.** This is the antithesis of the Zhang–Singh hard projection and is what makes loop closures and GNSS able to *correct* the dead-reckoned axis later.

> **Debug/introspection (Meridian L2).** Per the project's non-negotiable principle of "seeing what the estimator is doing," the front-end must publish, every step: the three position-block eigenvalues $\lambda_{1,2,3}$ of `HTH` (`esekfom.hpp:1784`), their eigenvectors as rviz arrows at the body origin (length $\propto 1/\sqrt{\lambda_i}$ so a long red arrow literally *points down the corridor*), the relative score $o_i$, and a scalar "degeneracy alarm" $= \mathbb{1}[\lambda_{\min} < \lambda_{\min,\text{floor}}]$. FAST-LIO publishes only the raw $6\times6$ pose covariance into the odometry message (`laserMapping.cpp:596-606`) and nothing about the eigen-spectrum — it relies entirely on the silent Bayesian fallback of §11.1.2 — which is exactly the introspection gap Meridian closes.

#### 11.1.4 Planarity degeneracy is also detected upstream

Degeneracy can be caught *before* the Hessian, at the data-association stage, and FAST-LIVO2 does this in its voxel map. When building a map plane it forms the point covariance $\Sigma = \frac1N\sum p\,p^\top - \bar p\,\bar p^\top$ and eigen-decomposes it (`FAST-LIVO2/src/voxel_map.cpp:65-86`):
```cpp
plane->covariance_ = plane->covariance_ / N - plane->center_ * plane->center_.transpose();
Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance_);
...
if (evalsReal(evalsMin) < planer_threshold_) { ... plane->is_plane_ = true; }   // :86, :121
```
where `planer_threshold_` is the tunable `lio/min_eigen_value` (default `0.01`, `voxel_map.cpp:42`). The smallest eigenvalue $\lambda_{\min}$ of the *point* covariance is the planarity measure: small $\Rightarrow$ points lie on a plane $\Rightarrow$ the normal $u_{\min}$ (eigenvector of $\lambda_{\min}$) is well-defined and the plane is admitted as a constraint (`plane->normal_ = evecs.col(evalsMin)`, `:113`). A cluster that is too "fat" ($\lambda_{\min}$ large) is rejected — it would contribute a *mis-oriented* normal and inject error. The min/mid/max eigenvalues are retained (`:116-118`) and the analytic Jacobian of $(n,q)$ w.r.t. each contributing point (`:90-110`) propagates point noise into the plane's $6\times6$ uncertainty `plane_var_`. That uncertainty then becomes a *per-residual, uncertainty-scaled gate* in the LiDAR update: FAST-LIVO2 computes $\sigma_l = J_{nq}\,\Sigma_{\text{plane}}\,J_{nq}^\top + n^\top \Sigma_p\, n$ and accepts the point-to-plane match only if `dis_to_plane_abs < 3*sqrt(sigma_l)` (`vio.cpp:1005-1011`) — a 3-$\sigma$ Mahalanobis test, strictly better than FAST-LIO's hand-tuned constant. This is the same eigen-spectrum idea as §11.1.1, applied to the *map* rather than the *estimator*: **degeneracy is checked at every level where a least-squares is solved.** Meridian's adaptive voxel-hash (L4) inherits this — a voxel only emits a registration plane if its $\lambda_{\min}$ planarity passes, and the surviving plane carries an anisotropic normal covariance into the point-to-plane $\Sigma_k$ via exactly such a $\sigma_l$.

---

### 11.2 Robust kernels (M-estimation) and IRLS

We now switch from "the geometry is uninformative" to "the datum is wrong." The least-squares cost $\tfrac12 r^2$ has unbounded influence: a single gross outlier with $r=100\sigma$ contributes $5000$ to the cost and its gradient drags the solution arbitrarily far. **M-estimation** replaces the quadratic with a sub-quadratic loss $\rho(\cdot)$ that saturates.

#### 11.2.1 Definition and the influence function

Per residual we minimise $\rho(r)$ where, writing the whitened (Mahalanobis) residual $u_k = \lVert r_k \rVert_{\Sigma_k^{-1}}$, the robust cost is $\rho(u_k)$. The **influence** of a residual on the solution is governed by $\psi(u) = \rho'(u)$ (the derivative); a bounded $\psi$ means a single datum cannot dominate. Common kernels:

| Kernel | $\rho(u)$ | $\psi(u)=\rho'(u)$ | behaviour |
|---|---|---|---|
| L2 (quadratic) | $\tfrac12 u^2$ | $u$ | influence unbounded |
| Huber($\delta$) | $\tfrac12 u^2$ if $|u|\le\delta$; $\;\delta(|u|-\tfrac12\delta)$ else | $u$ then $\delta\,\mathrm{sgn}(u)$ | influence *bounded*, never zero |
| Cauchy($c$) | $\tfrac{c^2}{2}\log(1+u^2/c^2)$ | $u/(1+u^2/c^2)$ | influence $\to 0$ (redescending) |
| Geman–McClure | $\tfrac{u^2/2}{1+u^2}$ | $u/(1+u^2)^2$ | strongly redescending |
| Truncated L2 (TLS) | $\tfrac12\min(u^2,\bar c^2)$ | $u$ then $0$ | hard reject beyond $\bar c$ |

**Huber** is the conservative default: it is convex (so it does not introduce extra local minima — important for the GN/LM solver of Section 08) and bounds influence, but because $\psi$ never returns to zero, a far outlier still exerts a constant pull. **Redescending** kernels (Cauchy, Geman–McClure, truncated) drive $\psi\to0$ so a far outlier is *ignored*, at the cost of non-convexity (more local minima — this is what GNC, §11.3, exists to tame).

#### 11.2.2 IRLS: the weight is the kernel in disguise

The Gauss–Newton machinery of Section 08 only knows how to solve weighted *quadratic* problems. We recover that form via **Iteratively Reweighted Least Squares (IRLS)**. The key identity: minimising $\sum_k \rho(u_k)$ has the same stationarity condition as a weighted quadratic with weights
$$ \boxed{\,w_k \;=\; \frac{\psi(u_k)}{u_k} \;=\; \frac{\rho'(u_k)}{u_k}\,} $$
because $\frac{d}{dx}\rho(u_k) = \psi(u_k)\,\frac{du_k}{dx} = \underbrace{\tfrac{\psi(u_k)}{u_k}}_{w_k}\, u_k \,\frac{du_k}{dx}$, which is exactly the gradient of $\tfrac12 w_k u_k^2$ when $w_k$ is held fixed. So the IRLS loop is:

1. At the current $x$, compute residuals $u_k$ and weights $w_k = \psi(u_k)/u_k$.
2. Solve the **weighted** normal equations of Section 08, $\big(\sum_k w_k H_k^\top \Sigma_k^{-1} H_k + \Sigma_{\text{prior}}^{-1}\big)\,\delta x = -\sum_k w_k H_k^\top \Sigma_k^{-1} r_k + \dots$
3. Update $x \boxplus \delta x$, repeat until convergence.

For Huber, $w_k = 1$ for inliers and $w_k = \delta/|u_k| < 1$ for outliers — the far outlier is linearly down-weighted. For Cauchy, $w_k = 1/(1+u_k^2/c^2)$. **Notice that IRLS reweighting and the per-point Hessian $\mathbf H = \sum_k w_k H_k^\top\Sigma_k^{-1}H_k$ couple back to §11.1:** down-weighting points reduces the effective information, which can *deepen* a degeneracy. A robust kernel applied carelessly along a barely-observed axis can zero out the only points constraining it. Hence in Meridian the robust weight and the observability score are computed together and the weight is *clamped* in directions flagged weakly-observed.

#### 11.2.3 What FAST-LIO and FAST-LIVO2 actually do (and what we add)

FAST-LIO uses **no explicit M-estimator on the LiDAR residual**; it uses a hard *gate* at association time. In `h_share_model` (`laserMapping.cpp:680-685`):
```cpp
float pd2 = pabcd(0)*point_world.x + pabcd(1)*point_world.y + pabcd(2)*point_world.z + pabcd(3);
float s   = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
if (s > 0.9) { point_selected_surf[i] = true; ... }      // else the point is dropped
```
This is a **binary truncated kernel keyed to point-to-plane distance, range-normalised**: a point is accepted only if its plane distance is small relative to $\sqrt{\text{range}}$ (far points are allowed slightly larger residuals because their depth noise is larger). It is a crude 0/1 weight — effectively $w_k\in\{0,1\}$ — combined with the planarity check from `esti_plane` (§11.1.4) and a gross-association gate `pointSearchSqDis[NUM_MATCH_POINTS-1] > 5 ? false` (`:671`). Point-LIO uses the same idea in its per-point sequential model but writes the gate as a range/residual product, `if (p_body.norm() > match_s * pd2 * pd2)` after `esti_plane` (`Point-LIO/src/Estimator.cpp:138-160`). Notably, the Point-LIO source carries a **commented-out continuous robust weight** right above that gate — `weight = sqrt(2*epsilon - 1)/epsilon` with `epsilon = pd2/sqrt(noise_state)` (`Estimator.cpp:149-158`) — a square-root (Huber-like) reweighting that the authors prototyped and disabled; it is precisely the continuous kernel Meridian should turn back on.

FAST-LIVO2 *does* use a robust treatment, on the **visual** side. Its sparse-direct photometric residual is the inverse-exposure-compensated patch error $e=\sum_{\text{patch}}\big(\tau_{\text{ref}}\,I_{\text{ref}} - \tau_{\text{cur}}\,I_{\text{cur}}\big)^2$ (`vio.cpp:746-751`, with $\tau=\,$`inv_expo_time`), and it **truncates** the patch when that error exceeds a threshold: `if (error > outlier_threshold * patch_size_total) continue;` (`vio.cpp:763`). The paper's §"Outlier Rejection" motivates this directly: visual map points "could be occluded in the current frame, have discontinuous depth, ... or have large view angles, all of which can severely degrade image alignment accuracy" and are therefore rejected (FAST-LIVO2, 2408.14035, line 686). That is a truncated-least-squares kernel on the photometric residual of Section 06. It is exactly this *visual* constraint — added on top of a degenerate LiDAR update — that lets FAST-LIVO2 survive the single-wall corridor where pure LIO "would fail" (line 849, §11.1.2).

**Meridian's upgrade.** Replace the binary gates with *continuous* robust kernels so the transition inlier→outlier is smooth (no chattering as a point flickers across the `s>0.9` boundary):
- LiDAR point-to-plane: **Huber** with $\delta$ set from the plane's own normal-covariance — exactly the $\sigma_l$ of `vio.cpp:1008` — so the gate scales with map uncertainty instead of FAST-LIO's hand-tuned `0.9`. (FAST-LIVO2 already does the *3-$\sigma$ truncation*; Meridian softens it to a continuous Huber so the transition does not chatter, and re-enables Point-LIO's disabled square-root weight.)
- Photometric: **Huber/Cauchy** on the patch residual with the affine/inverse-exposure-compensated intensity model of Section 06, exposing the per-patch weight as a debug image overlay.
- GNSS: **switchable constraint** (§11.4) rather than a kernel, because GNSS outliers are often *biased* (multipath) not just heavy-tailed, and a switch handles a persistently-wrong fix better than a redescending kernel.

---

### 11.3 Graduated Non-Convexity (GNC)

Redescending kernels reject outliers cleanly but are **non-convex**: started far from the truth, IRLS can converge to a local minimum where it has labelled inliers as outliers and vice-versa. **GNC (Yang, Antonante, Tzoumas, Carlone 2020)** solves this by *homotopy*: optimise an easy (convex) surrogate first, then gradually deform it back to the true non-convex robust cost, tracking the minimiser. It needs no initial guess of the inlier set — its headline result is that GNC recovers correct estimates with up to ~80–90% outliers in problems like point-cloud registration and pose-graph SLAM, where a single bad initialisation would otherwise be fatal.

#### 11.3.1 The surrogate and the control parameter

GNC introduces a **control parameter** $\mu>0$ and a surrogate $\rho_\mu(u)$ such that:
- at one extreme of $\mu$, $\rho_\mu$ is convex (often $\approx$ pure least-squares — every datum trusted);
- at the other extreme, $\rho_\mu \to \rho$, the target robust kernel (e.g. truncated least squares / Geman–McClure).

GNC is most cleanly stated through the **Black–Rangarajan duality**: any robust $\rho$ can be written as a joint minimisation over the state $x$ *and* per-residual weights $w_k\in[0,1]$,
$$ \min_x \sum_k \rho(u_k) \;=\; \min_{x}\;\min_{w_k\in[0,1]} \sum_k \Big( w_k\, u_k^2 \;+\; \Phi_\rho(w_k) \Big), $$
where $\Phi_\rho$ is the *outlier-process penalty* determined by $\rho$. For **Truncated Least Squares** with inlier threshold $\bar c$, the GNC surrogate weight update (closed form, given $\mu$) is the Yang-2020 result:
$$ w_k \;=\; \begin{cases} 0, & u_k^2 \;>\; \dfrac{\mu+1}{\mu}\,\bar c^2 \quad(\text{confident outlier}) \\[2mm] 1, & u_k^2 \;<\; \dfrac{\mu}{\mu+1}\,\bar c^2 \quad(\text{confident inlier}) \\[2mm] \dfrac{\bar c}{u_k}\sqrt{\mu(\mu+1)} \;-\; \mu, & \text{otherwise (graduated, the only branch that depends on }\mu) \end{cases} $$
and for **Geman–McClure**,
$$ w_k \;=\; \left( \frac{\mu\,\bar c^2}{u_k^2 + \mu\,\bar c^2} \right)^{2}. $$
The algorithm alternates (this *is* the GNC loop):

1. **Initialise** $\mu$ at the convex extreme (so every $w_k\approx 1$) and $x$ at any guess.
2. **Variable update (state):** with $w_k$ fixed, solve the weighted least squares for $x$ — one or a few GN/LM steps from Section 08.
3. **Weight update:** with $x$ fixed, recompute $w_k$ from the closed forms above using the current $\mu$.
4. **Graduate:** anneal $\mu$ one notch toward the non-convex extreme (e.g. tighten by a constant factor each outer iteration).
5. Repeat 2–4 until $\mu$ reaches the target and the weights stop changing.

Because step 2 is a standard weighted GN solve, **GNC reuses the entire Section 08 solver** — it is an outer loop wrapping the normal equations, identical in form to IRLS (§11.2.2) but with the weight update *parameterised by a homotopy that starts convex*. That single difference is what buys robustness to a bad initial guess.

#### 11.3.2 Where Meridian uses GNC (and where it must not)

GNC is expensive (an IRLS loop per $\mu$ value) and is therefore **not** appropriate inside the real-time L2 front-end per-scan update (which already gets its robustness from the IMU prior §11.1.2 + per-point Huber §11.2). Meridian places GNC where the initial guess is genuinely poor and the time budget is relaxed:
- **Loop-closure relative-pose estimation (L5).** After Scan-Context++/STD-BTC propose a candidate, GICP aligns the two submaps; GNC-GICP (truncated LS over correspondences) makes that alignment robust to the gross outliers typical of a perceptually-aliased candidate, *before* the candidate is passed to PCM (§11.5).
- **Back-end batch re-linearisation / loss-of-track recovery.** When the front-end signals tracking loss (high $\lambda_{\min}$ degeneracy + large residuals), the back-end can run a GNC-robustified batch solve over the recent window to re-converge without a good prior.

---

### 11.4 Switchable constraints

**Switchable constraints (Sünderhauf & Protzel 2012)** attack the loop-closure / GNSS outlier problem from a different angle than M-estimation: instead of *down-weighting* a bad constraint by a fixed nonlinearity, they let the **optimiser itself decide whether to keep each constraint**, by promoting the on/off decision to a *continuous optimisation variable*.

For each suspect constraint $k$ (a loop closure, a GNSS factor) introduce a **switch variable** $s_k \in \mathbb{R}$ (later mapped through $\Psi(\cdot)$ to $[0,1]$) and rewrite the cost as
$$ \mathcal{C} \;=\; \underbrace{\sum_{\text{odom}} \lVert r_i \rVert^2_{\Sigma_i^{-1}}}_{\text{trusted (never switched)}} \;+\; \sum_{k\in\text{suspect}} \Big( \underbrace{\Psi(s_k)^2\, \lVert r_k \rVert^2_{\Sigma_k^{-1}}}_{\text{switched constraint}} \;+\; \underbrace{\lVert\, 1 - s_k \,\rVert^2_{\Sigma_s^{-1}}}_{\text{switch prior}} \Big), $$
with $\Psi$ a sigmoid-like clamp ($\Psi(s)=\operatorname{clamp}(s,0,1)$ in the original; a logistic in later variants). The mechanics:
- The factor's effective information is **scaled by $\Psi(s_k)^2$.** If the optimiser drives $s_k\to0$, the constraint is *switched off* — it contributes nothing — and the pose is free to ignore it.
- The **switch prior** $\lVert 1 - s_k\rVert^2_{\Sigma_s^{-1}}$ pulls every switch toward $1$ (on). Turning a constraint off costs $\approx 1/\Sigma_s$. So the optimiser only switches off a constraint when the residual it would otherwise pay, $\Psi(s_k)^2\lVert r_k\rVert^2$, *exceeds* the price of disabling it. $\Sigma_s$ sets that price: small $\Sigma_s$ (strong prior) = reluctant to switch off = trust constraints; large $\Sigma_s$ = quick to disable.

This is elegant because the on/off decision is made **jointly with the trajectory** inside the same Gauss–Newton solve — $s_k$ is just another variable with its own Jacobian columns. Compared to a robust kernel: switchable constraints handle a *persistently biased* outlier (a multipath GNSS fix that is consistently 5 m off) better than a redescending kernel, because the switch can fully disable it, whereas a kernel keeps reintroducing a residual each iteration.

**Relation to GNC.** Dynamic Covariance Scaling (DCS) is the closed-form solution of the switch sub-problem; and switchable constraints, DCS, and GNC are all instances of the same Black–Rangarajan outlier-process (§11.3.1) — they differ in the penalty $\Phi$ and in whether the weight is solved in closed form, annealed, or co-optimised. Meridian uses **switchable constraints for GNSS** (Section 07: each absolute-position factor carries a switch, so a bad-fixed-quality epoch is silently disabled and the trajectory rides on LIO until GNSS recovers) and reserves **PCM** for loop closures (next), because PCM's batch consistency test is stronger than per-edge switching when many loop candidates arrive at once.

> **Meridian debug.** Publish each switch value $\Psi(s_k)$ on a ROS topic and colour the corresponding GNSS/loop marker green→red as it switches off. An operator watching the overlay can literally *see* the system reject a multipath GNSS epoch.

---

### 11.5 Pairwise Consistency Maximization (PCM) for loop closures

A single false loop closure is catastrophic: it welds two unrelated places together and folds the entire map. Robust kernels and switchable constraints help, but they reason about each loop **in isolation**. **PCM (Mangelson, Rosen, ... 2018 — "Pairwise Consistent Measurement Set Maximization")** instead exploits a powerful collective signal: *true* loop closures are **mutually geometrically consistent**, while false ones are (almost surely) inconsistent with the true ones and with each other.

#### 11.5.1 The consistency test

Consider two loop closures: $z_{ab}$ (a relative pose between keyframes $a,b$) and $z_{cd}$ (between $c,d$). The trajectory already gives odometry-chained relative transforms $\hat T_{ac}$ (from $a$ to $c$) and $\hat T_{bd}$. If *both* loops are correct, then composing
$$ T_{\text{loop}} \;=\; z_{ab}^{-1} \cdot \hat T_{ac} \cdot z_{cd} \cdot \hat T_{db} $$
should compose back to the identity (a closed cycle through the two loops and the odometry between their endpoints). Define the **pairwise consistency residual** and its Mahalanobis norm
$$ \epsilon_{ij} \;=\; \log\!\big( T_{\text{loop}} \big)^\vee, \qquad d^2_{ij} \;=\; \epsilon_{ij}^\top\, \Sigma_{ij}^{-1}\, \epsilon_{ij}, $$
using $\log/\vee$ (Section 02) to map the residual transform to the tangent space, with $\Sigma_{ij}$ propagated from the loop and odometry covariances. Two loop closures $i,j$ are declared **pairwise consistent** iff
$$ d^2_{ij} \;\le\; \chi^2_{d,\,1-\alpha}, $$
the chi-squared threshold at significance $\alpha$ for $d$ DOF (e.g. $\chi^2_{6,0.95}$ for a full SE(3) loop). Note that consistency uses the **odometry**, which §11.1–11.4 have already made trustworthy along its observable axes — PCM is only as good as the relative-pose backbone feeding it, which is why observability (§11.1) and per-edge robustness (§11.2–11.4) come first.

#### 11.5.2 Max-clique: pick the largest mutually-consistent set

Build a **consistency graph** $G$: one vertex per candidate loop closure, an edge between $i$ and $j$ iff they pass the pairwise test above. A set of loops that are *all* mutually consistent is a **clique** in $G$. PCM then selects the **maximum clique** — the largest set of loop closures that are pairwise consistent — and accepts only those; everything outside the clique is rejected as inconsistent (likely a perceptual alias).
$$ \mathcal{S}^\star \;=\; \arg\max_{\mathcal{S}\subseteq V}\, |\mathcal{S}| \quad \text{s.t. } (i,j)\in E\;\; \forall\, i,j\in\mathcal{S}. $$
Maximum-clique is NP-hard in general, but the consistency graphs in SLAM are small (tens of candidate loops) and PCM uses an exact branch-and-bound / heuristic max-clique that runs in milliseconds. The intuition for why this works: a *random* false loop is consistent with the true cluster only by coincidence (a measure-zero event in continuous pose space, made finite only by noise), so false loops form at most small, scattered cliques while the true loops form one large clique. Picking the largest clique recovers the true set even under a high false-positive rate from the place-recognition front-end.

#### 11.5.3 PCM in the Meridian pipeline

PCM is the **last gate before a loop closure becomes a factor** (L5 → L3):
1. Place recognition (Scan-Context++ → STD/BTC) proposes candidate matches — *high recall, low precision* by design.
2. GICP (optionally GNC-GICP, §11.3.2) computes each candidate's relative pose $z_{ab}$ and covariance.
3. **PCM** filters the candidate set to the maximum pairwise-consistent clique.
4. Survivors enter the GTSAM iSAM2 factor graph (Section 03/09), *additionally* wrapped in a switchable constraint or robust (DCS/Cauchy) kernel as a second line of defence against the rare false loop that slips through PCM.

This layering — PCM (set-level) **then** robust kernel/switch (edge-level) — is deliberate: PCM removes the structured, gross errors that would defeat a per-edge kernel, and the kernel mops up the residual heavy tail. Neither alone is sufficient for a tactical system.

> **Meridian debug.** Publish the consistency graph (vertices = candidate loops, edges = pairwise-consistent) as an rviz marker array, with the selected max-clique highlighted, so an operator can audit *why* a loop was accepted or rejected.

---

### 11.6 Synthesis: the robustness stack, end to end

Putting §§11.1–11.5 in the order a measurement experiences them, from raw point to global map:

```
 raw LiDAR point / pixel / GNSS fix
        │
   [L1 preprocess]                         (Section 05/06)
        │
   PLANARITY EIGEN-TEST  ───────────────►  reject fat voxels   (§11.1.4, voxel_map.cpp:86)
        │  (admit plane + anisotropic Σ_k)
   ASSOCIATION GATE / ROBUST KERNEL ─────► down-weight outliers (§11.2; laserMapping.cpp:681, vio.cpp:763/1011)
        │  (w_k = ψ(u_k)/u_k)
   IRLS-WEIGHTED GN/iEKF STEP             (Section 08/09; esekfom.hpp:1782-1809)
        │   H_post = (P/R)^-1 + Σ_k w_k H_kᵀ Σ_k^-1 H_k
   HESSIAN EIGEN-SPECTRUM ───────────────► per-axis observability o_i (§11.1.3, Zhang/X-ICP/D²-LIO)
        │   IMU prior P^-1 regularizes degenerate axes (§11.1.2; FAST-LIO2 has no explicit degeneracy code)
        │
   keyframe pose + ANISOTROPIC Σ̃_kf  ────► flows into back-end as noise, NOT a flag
        │
   ════════════════ L3 BACK-END (GTSAM iSAM2) ════════════════
        │
   GNSS factor ──► SWITCHABLE CONSTRAINT  (§11.4, Sünderhauf 2012)
   loop candidates ──► GNC-GICP (§11.3) ──► PCM max-clique (§11.5, Mangelson 2018)
                                               │ + switch/DCS/Cauchy as 2nd defence
        │
   globally consistent, robust trajectory  ──►  L4 TSDF mesh + L6 confidence overlay
```

The three ideas you should leave this section with:

1. **Degeneracy is curvature, outliers are residuals — diagnose them separately.** Degeneracy is read from the Hessian eigen-spectrum ($\sum_j n_j n_j^\top$ for LiDAR position, `esekfom.hpp:1784` `HTH`); outliers are read from individual residual magnitudes (`vio.cpp:763`, `laserMapping.cpp:681`). Conflating them (e.g. rejecting a point because the axis it constrains is poorly observed) is a classic bug.
2. **Never use a binary flag where a noise model will do.** The IMU prior turns degeneracy into a *graceful, automatic fallback to inertial odometry* (§11.1.2); per-axis observability turns into an *anisotropic keyframe covariance* (§11.1.3); a suspect GNSS fix turns into a *continuous switch* (§11.4). Binary flags throw away usable partial information and chatter at thresholds.
3. **Outlier robustness is layered, and the layers are the same math.** Hard gate (FAST-LIO `s>0.9`, `laserMapping.cpp:681`; FAST-LIVO2 3-$\sigma$, `vio.cpp:1011`) ⊂ Huber/Cauchy IRLS ⊂ GNC homotopy ⊂ switchable constraints ⊂ PCM are all instances of the Black–Rangarajan outlier process, differing only in (a) whether the weight is binary/continuous, (b) whether it is convexified by a homotopy, and (c) whether the decision is per-edge or per-set. Meridian uses the *cheapest sufficient* layer at each stage: prior+gate in the real-time front-end, GNC+PCM+switches in the relaxed-budget back-end.

> **Cross-references.** State manifold and $\log/\vee$ used in §11.5: Section 02. MAP cost and information form: Section 03. IMU prior $P^{-1}$ that regularizes degeneracy: Sections 04, 09. Point-to-plane residual and its Jacobian: Section 05. Photometric residual and affine/inverse-exposure compensation: Section 06. GNSS/ENU frames for switchable constraints: Section 07. The weighted normal equations that IRLS/GNC reuse: Section 08. iEKF≡GN equivalence behind the Bayesian fallback: Section 09. Lifting robust weights onto B-spline control points: Section 10. The concrete FAST-LIO2/FAST-LIVO2 step that wires all of this together: Section 12.
