## 01. Introduction: estimation problem & coupling spectrum & MAP

> **Chapter:** *Tightly-Coupled Multi-Sensor State Estimation & Residuals.*
> This is the opening section. It defines the problem the rest of the chapter solves, fixes the vocabulary, and erects the single conceptual scaffold — **maximum a posteriori (MAP) estimation on a factor graph over a manifold-valued state** — onto which every later residual (IMU §04, LiDAR §05, visual §06, GNSS §07) and every solver (batch §08, recursive §09, continuous-time §10) is hung. We deliberately stay at the level of *what is being estimated, from what, and why one fusion architecture dominates*; the machinery (Lie groups §02, probability §03, Jacobians, Schur complements) is introduced here only far enough to motivate it, then developed rigorously in the named sections.
>
> Notation is the chapter-wide convention defined in §02: state $x$; rotation $R\in SO(3)$; position $p$; velocity $v$; gyro/accel biases $b_g,b_a$; gravity $g$; the manifold retraction/difference $\boxplus,\boxminus$ ("⊞ / ⊟"); Lie maps $\mathrm{Exp}/\mathrm{Log}$ and $\exp/\log$ with hat $(\cdot)^\wedge$; residual $r$, measurement $z$, prediction $h(x)$; Jacobian $H$; covariance $\Sigma$, information $\Omega=\Sigma^{-1}$; Kalman gain $K$; LiDAR point $p_L$ with plane $(n,d)$ such that $n\cdot x + d = 0$; image intensity $I$, projection $\pi$, intrinsics $K_{\mathrm{cam}}$; B-spline control points $c_k$ and trajectory $T(t)\in SE(3)$.

---

### 1.1 What we are actually estimating

A mobile robot does not observe its own pose. It observes *consequences* of its motion: an IMU feels specific force and angular rate, a LiDAR returns ranges to surfaces, a camera integrates radiance onto a sensor, a GNSS receiver decodes pseudoranges into a global position. **State estimation** is the inverse problem of recovering the quantity we care about — the robot's trajectory, and incidentally a map — from these indirect, noisy, asynchronous measurements.

Make this precise. Let the **state** at time $t$ be

$$
x(t) \;=\; \bigl(\,R(t),\; p(t),\; v(t),\; b_g(t),\; b_a(t)\,\bigr),
\qquad R\in SO(3),\;\; p,v,b_g,b_a \in \mathbb{R}^3 ,
$$

possibly augmented with gravity $g$ and sensor-to-sensor extrinsics. This is not an academic choice — it is exactly the live state in our primary reference. In FAST-LIO the estimator state is declared as a compound manifold

```
MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos)) ((SO3, rot))
((SO3, offset_R_L_I)) ((vect3, offset_T_L_I))
((vect3, vel)) ((vect3, bg)) ((vect3, ba)) ((S2, grav)) );
```

(`FAST_LIO/include/use-ikfom.hpp:9-18`). Read off the correspondence to our notation: `pos`$=p$, `rot`$=R$, `vel`$=v$, `bg`$=b_g$, `ba`$=b_a$, `grav`$=g$, and `offset_R_L_I / offset_T_L_I` are the LiDAR→IMU extrinsic rotation and translation — included **in the estimated state** so they can be refined online. Two structural facts are visible immediately and will organize everything below:

1. **The state is not a vector — it is a manifold.** $R$ lives on $SO(3)$; gravity is parameterized on the 2-sphere $S^2$ (the `S2` type, line 17, with a fixed magnitude). You cannot add a correction to $R$ with `+`; you must *retract* it through the group exponential. The chapter therefore opens every estimator on the $\boxplus/\boxminus$ algebra developed in §02. As a preview, a small correction $\delta$ updates the state by $x \leftarrow x \boxplus \delta$, where on the rotation block $R\boxplus\theta = R\,\mathrm{Exp}(\theta)$ and $R_1 \boxminus R_2 = \mathrm{Log}(R_2^\top R_1)$.

2. **The dynamics that couple consecutive states are the IMU kinematics.** FAST-LIO's continuous-time process model $\dot x = f(x,u)$ is literally:

```
vect3 omega      = in.gyro - s.bg;          // ω = ω_m − b_g
vect3 a_inertial = s.rot * (in.acc - s.ba); // R (a_m − b_a)
res(pos)  = s.vel;                          // ṗ = v
res(rot)  = omega;                          // Ṙ ↔ ω
res(vel)  = a_inertial + s.grav;            // v̇ = R(a_m − b_a) + g
```

(`FAST_LIO/include/use-ikfom.hpp:38-50`), i.e.

$$
\dot p = v,\qquad
\dot R = R\,(\omega_m - b_g)^\wedge,\qquad
\dot v = R\,(a_m - b_a) + g ,
$$

with biases modeled as random walks. This $f$ is the engine of *prediction*; the inertial residual it induces is derived in §04. Note already that gravity $g$ and the biases are *unknowns inside* $f$ — the IMU cannot be "used" without simultaneously estimating its own errors, which is the first hint that clean sensor separation is a fiction.

So the estimation problem, informally, is: **find the trajectory $\{x(t)\}$ (and the unknown calibration/bias parameters) that is most consistent with all measurements, given a motion model that ties successive states together.** "Most consistent" is the part that needs a probabilistic definition (§1.4); "tie successive states together" is the part that distinguishes *odometry* (local, drifting) from full *SLAM* (with loop closure, §1.5). First, however, we confront the question that determines the entire system architecture: **how tightly do we fuse?**

---

### 1.2 The coupling spectrum: loose vs. tight

Every multi-sensor estimator must decide *at what level of abstraction sensors meet*. There is a spectrum, but two named poles dominate the literature, and the choice between them is the single most consequential architectural decision in this chapter.

Our reference paper states the dichotomy in plain terms: *"There are generally two ways of fusion: loosely-coupled and tightly-coupled. The loosely-coupled approach processes the LiDAR and IMU measurements separately to infer their motion constraints, which are then fused together. The separate processing of the raw measurements neglects the correlation between the internal states of the two, and usually leads to information loss. The tightly-coupled approach, by contrast, directly fuses the raw LiDAR and IMU measurements"* (`papers/2010.08196.txt:21-28`, FAST-LIO §I).

#### Loosely-coupled architecture

In a loosely-coupled system each sensor is first reduced, *in isolation*, to a self-contained pose or pose-rate estimate, and only those derived estimates are fused:

```
                  ┌────────────────────────┐
   raw LiDAR ────▶│ scan-matcher (e.g. ICP)│──▶ pose estimate  T_lidar, Σ_lidar ┐
                  └────────────────────────┘                                    │
                  ┌────────────────────────┐                                    ▼
   raw IMU   ────▶│ inertial integrator    │──▶ pose/vel  T_imu,  Σ_imu  ──▶ ┌──────┐
                  └────────────────────────┘                                  │ fuse │──▶ x̂
                                                                              │ (EKF)│
   raw GNSS  ────▶ fix (lat/lon) ─────────────────────────────────────────▶  └──────┘
```

The fusion node never sees a single LiDAR point or IMU sample; it sees *summaries*. Concretely, the scan-matcher solves its own little optimization (align this scan to the map) and emits one pose $T_{\text{lidar}}$ with a covariance $\Sigma_{\text{lidar}}$; the EKF then treats that pose as a measurement.

#### Tightly-coupled architecture

In a tightly-coupled system there is **one estimator** and the raw measurements of every sensor enter it as residuals on the *same* state:

```
   raw LiDAR points ─┐
   raw IMU samples ──┼──▶ ┌───────────────────────────────────────┐
   raw camera pixels ┤    │  single estimator over state x:        │──▶ x̂, Σ
   raw GNSS ─────────┘    │  min Σ ‖r_imu‖²_Ω + Σ ‖r_lidar‖²_Ω      │
                          │      + Σ ‖r_cam‖²_Ω + Σ ‖r_gnss‖²_Ω     │
                          └───────────────────────────────────────┘
```

This is exactly FAST-LIO's structure. There is no intermediate "LiDAR pose"; instead each LiDAR point contributes a **point-to-plane residual** evaluated against the current state estimate. In code the measurement model `h_share_model` transforms each effective point into the world frame *using the live state* `s.rot, s.pos` and the extrinsics, finds its nearest map surface, fits a plane $(n,d)$, and forms the residual $r = n\cdot p_W + d$ together with its Jacobian (`FAST_LIO/src/laserMapping.cpp:451-459`). The IMU, meanwhile, has already produced the *prediction* the points are evaluated against (it propagated the state and de-skewed the scan — `FAST_LIO/src/IMU_Processing.hpp:3`). LiDAR and IMU thus meet **inside one update**, sharing one covariance — the literal meaning of "tight." The plane-residual machinery is §05; the fusion update is §08–§09.

The full FAST-LIO main loop makes the single-estimator structure concrete (`FAST_LIO/src/laserMapping.cpp:460-468`):

```
sync_packages(Measures);                         // gather one scan + its IMU interval
p_imu->Process(Measures, kf_state, undistort);   // IMU forward-propagate + de-skew (predict)
kf.update_iterated_dyn_share_modified(...);       // tightly-coupled measurement update (correct)
map_incremental();                                // fold the corrected scan into the map
publish_odometry / publish_path / publish_frames; // emit pose + introspection
```

There is one state object `kf_state`, propagated by inertia and corrected by raw points — no second pose estimate exists to fuse.

---

### 1.3 Why tight wins: the information-loss argument

The phrase "neglects the correlation … leads to information loss" (`papers/2010.08196.txt:24-26`) deserves a precise statement, because it is *the* reason tightly-coupled fusion is state-of-the-art. We give the argument in information form; the underlying Gaussian/information algebra is §03.

**Setup.** A raw LiDAR scan is a set of point measurements $\{z_i\}$. Each is a function of the state through some $h_i(x)$ with noise $n_i\sim\mathcal N(0,\Sigma_i)$. Under the Gaussian/linearized model (§03), the information that this *raw* batch carries about the state is the sum of rank-1-ish contributions

$$
\Omega_{\text{raw}} \;=\; \sum_i H_i^\top\,\Sigma_i^{-1}\,H_i,
\qquad H_i = \left.\frac{\partial h_i}{\partial x}\right|_{\hat x}. \tag{1.1}
$$

This is the **Fisher information** of the raw measurement set. It is, in general, *anisotropic*: in a long corridor the point-to-plane normals $n$ all point at the walls, so $\Omega_{\text{raw}}$ is large in the cross-corridor directions and nearly **rank-deficient along the corridor axis**. That structure — *which* directions are observed and which are not — lives entirely in the geometry of the $H_i$.

**Loosely-coupled fusion discards that structure.** The scan-matcher compresses all $\{z_i\}$ into a single pose $T_{\text{lidar}}$ with covariance $\Sigma_{\text{lidar}}$, i.e. information $\Omega_{\text{lc}}=\Sigma_{\text{lidar}}^{-1}$, and *that* is what the fuser ingests:

$$
\Omega_{\text{raw}} = \sum_i H_i^\top\Sigma_i^{-1}H_i
\quad\Longrightarrow\quad
\Omega_{\text{lc}} = \Sigma_{\text{lidar}}^{-1}. \tag{1.2}
$$

Three losses are baked into the arrow (1.2):

- **Premature commitment / re-linearization loss.** $T_{\text{lidar}}$ was found by the scan-matcher's *own* optimum, computed *without* the IMU's prior. In a degenerate scene that optimum is arbitrary along the unobserved direction. Once committed, the fuser can never recover the lost direction — it only ever sees the (over-confident or arbitrary) summary. The tightly-coupled estimator instead *re-linearizes the raw residuals at the IMU-informed state* every iteration (the loop at `esekfom.hpp:288-314`, §1.4), so the IMU prior regularizes exactly the direction the geometry cannot see.

- **Covariance is a lossy projection.** Even an honest $\Sigma_{\text{lidar}}$ is only a second moment; collapsing a structured sum of $H_i^\top\Sigma_i^{-1}H_i$ into one $6\times6$ block throws away how individual surfaces constrain the state, *and* throws away cross-correlations between the LiDAR-derived pose and the IMU's velocity/bias/gravity states. Those cross-terms are nonzero — gravity tilts the accelerometer, which couples to where the points land — and they are precisely what the off-diagonal of the joint information would have carried. There is no representation of them in $\Sigma_{\text{lidar}}$.

- **No back-pressure.** In tight fusion the IMU prior can *suppress* a bad scan-match direction and the points can *correct* IMU bias and gravity within the same update — information flows both ways. In loose fusion each module has already finished before they meet.

Formally, summarization can only destroy or preserve information, never create it: by the data-processing inequality the post-summary information satisfies $\Omega_{\text{lc}} \preceq \Omega_{\text{raw}}$ along observed directions, with equality only when $h$ is exactly linear and the scan-match used the same prior — never true in practice. Hence the textbook conclusion, and the paper's: *"Thus it is more accurate and robust … in degenerate environments,"* and *"The tightly-coupled approach has become the mainstream for LIO"* (`papers/2010.08196.txt:27-30`).

> **Worked micro-example (corridor).** Suppose the cross-corridor direction is observed with information $\omega_\perp$ and the along-corridor direction with $\omega_\parallel \approx 0$. The IMU's preintegrated motion supplies a prior with information $\omega_{\text{imu}}>0$ in *all* directions (it integrated acceleration along the corridor). *Tight:* the joint information along-axis is $\omega_\parallel + \omega_{\text{imu}} = \omega_{\text{imu}} > 0$ — the state is observable, the estimate drifts only at inertial rates. *Loose:* the scan-matcher, run without the IMU, must output *some* along-axis value; its reported $\Sigma_{\text{lidar}}$ either honestly says "infinite uncertainty" (and the pose is unusable / numerically singular) or lies. Either way the joint estimate is strictly worse than the tight one. This is the degeneracy story that §11 turns into a per-axis observability score feeding the back-end.

**The cost of tight coupling**, and the reason loose coupling persisted, is computational: a raw scan has $10^3$–$10^5$ points, so a naive update inverts a measurement-sized matrix. FAST-LIO's headline contribution is a Kalman-gain reformulation that *"requires inverting only a matrix of the dimension of the state, instead of the measurements"* (`papers/2010.08196.txt:39-41`); that reformulation, and its equivalence to a Schur-complemented Gauss–Newton step, is the subject of §08–§09. Tight coupling is therefore not merely "more accurate in principle" — with the right linear algebra it is *also* real-time, which is why it is the architecture Meridian builds on.

---

### 1.4 The unifying view: MAP on a factor graph

Loose vs. tight is *where* sensors meet; we now fix *what the meeting computes*. The entire chapter is organized around one formulation: **maximum a posteriori estimation**, expressed as a **factor graph**, solved as a **nonlinear least-squares** problem on the manifold. Every residual in §04–§07 is one factor type; every solver in §08–§10 is one way to minimize the same objective. Establishing this now means later sections need only supply $r(\cdot)$ and $H$.

**MAP.** Collect all states into $\mathcal X = \{x_k\}$ and all measurements into $\mathcal Z$. We seek the most probable trajectory given the data:

$$
\mathcal X^\star \;=\; \arg\max_{\mathcal X}\; p(\mathcal X \mid \mathcal Z)
\;\overset{\text{Bayes}}{=}\; \arg\max_{\mathcal X}\; p(\mathcal Z \mid \mathcal X)\,p(\mathcal X). \tag{1.3}
$$

Assume measurements are conditionally independent given the state and the prior factorizes over the motion model. Then the joint posterior **factorizes**, and that factorization *is* the factor graph (§03):

$$
p(\mathcal X\mid\mathcal Z) \;\propto\; \underbrace{p(x_0)}_{\text{prior}}\;
\prod_{k}\underbrace{p(x_{k+1}\mid x_k,\,u_k)}_{\text{IMU / motion (§04)}}\;
\prod_{j}\underbrace{p(z_j\mid x_{i(j)})}_{\substack{\text{LiDAR §05, visual §06,}\\ \text{GNSS §07}}} . \tag{1.4}
$$

```
   prior        IMU         IMU
  ┌────┐      ┌────┐      ┌────┐
  ● ── (x0) ── ▢ ── (x1) ── ▢ ── (x2) ── …        ▢ = factor (a residual)
         │            │            │                ● = unary prior
       [LiDAR]     [LiDAR]      [LiDAR]              ◯ = state (variable node)
       [visual]    [visual]     [GNSS]              edges connect a factor
       [GNSS]                                       to the states it touches
```

Each box is a probabilistic constraint; the *front-end* of any tightly-coupled system is, at bottom, the act of **building these factors from raw data** (associating each LiDAR point to a plane, each pixel patch to a 3-D point, each IMU interval to a preintegrated delta), and the *back-end* is solving the resulting graph.

**MAP = nonlinear least squares.** Take negative log of (1.4). With every factor Gaussian in its residual — $p(z\mid x)\propto \exp\!\big(-\tfrac12\,\|r(x,z)\|_{\Sigma}^2\big)$ where $\|e\|_\Sigma^2 := e^\top\Sigma^{-1}e = e^\top\Omega\,e$ — maximizing the posterior becomes minimizing a sum of squared, information-weighted residuals:

$$
\boxed{\;\mathcal X^\star = \arg\min_{\mathcal X}\;
\sum_{\text{factors } c} \big\| r_c(\mathcal X)\big\|^2_{\Sigma_c}\;}
\tag{1.5}
$$

This single objective is the spine of the chapter. Reading it across the sections: the inertial residual $r_{\text{imu}}$ (§04) penalizes deviation from the IMU-predicted motion; the LiDAR residual $r_{\text{lidar}} = n\cdot p_W + d$ (§05) penalizes points off their plane; the photometric residual $r_{\text{cam}} = I(\pi(\cdot)) - I_{\text{ref}}$ (§06) penalizes intensity mismatch; the GNSS residual (§07) penalizes deviation from the global fix. Tight coupling is now precisely "**all of these residuals share the same $\mathcal X$ and are minimized jointly**." Loose coupling is "minimize each $r$ separately, then add a factor on the *outputs*" — which (1.2) showed is lossy.

**Solving on a manifold.** Because $\mathcal X$ contains $R\in SO(3)$, (1.5) is minimized by iterated **Gauss–Newton** *in the tangent space*: linearize each residual about the current estimate, $r_c(\mathcal X \boxplus \delta) \approx r_c(\mathcal X) + H_c\,\delta$ with $H_c = \partial r_c/\partial\delta$ the Jacobian *with respect to the $\boxplus$ perturbation* (§02 derives these), stack the normal equations

$$
\Big(\textstyle\sum_c H_c^\top \Omega_c\, H_c\Big)\,\delta
\;=\; -\,\textstyle\sum_c H_c^\top \Omega_c\, r_c,
\qquad \mathcal X \leftarrow \mathcal X \boxplus \delta, \tag{1.6}
$$

and repeat to convergence. The left matrix is the **total information** $\Omega = \sum_c H_c^\top\Omega_c H_c$ — the same object as (1.1), now assembled from *every* sensor; this is where the information-loss argument and the solver meet. The recursive (filtering) view (§09) shows that the **iterated EKF FAST-LIO actually runs is this exact Gauss–Newton iteration**, just organized recursively with the IMU-propagated state as a prior factor. The iteration is visible in the reference code: `update_iterated_dyn_share_modified` loops `for (i = -1; i < maximum_iter; i++)`, each pass re-evaluating the measurement model `h_share_model(x_, …)`, recomputing the gain $K$, and applying $x_\_ \leftarrow x_\_ \boxplus \delta$ until convergence (`FAST_LIO/include/IKFoM_toolkit/esekfom/esekfom.hpp:288-314`; the prior comes from `predict`, line 217). The equivalence "iterated-EKF update $\equiv$ one Gauss–Newton solve of the local MAP" is proved in §09 and is why a filter and a smoother are *the same estimator* viewed two ways. The Schur complement that makes (1.6) tractable when $\mathcal X$ is huge — and that underlies FAST-LIO's state-sized (not measurement-sized) gain — is §08.

> **Why this scaffold and not another.** Casting everything as (1.5) buys three things this chapter will cash in: (i) *modularity* — adding a sensor means adding a factor type with a residual $r$ and Jacobian $H$, nothing else (Meridian's `IFrontEnd` and swappable residuals mirror this); (ii) *a single notion of uncertainty* — the information matrix $\Omega$ in (1.6) is reused for marginalization (§08), for the EKF covariance (§09), and for the degeneracy/observability score (§11); (iii) *batch ↔ recursive unification* — filters, fixed-lag smoothers, and full batch solvers differ only in *which* variables they keep and *when* they linearize, all minimizing the same (1.5). The back-end factor graph that the front-end's keyframes feed (Meridian L3, GTSAM iSAM2) is *literally* (1.4) at keyframe resolution — so the same picture spans front-end and back-end.

---

### 1.5 SLAM vs. odometry, and the role of the map

The factor graph (1.4) also clarifies the difference between **odometry** and full **SLAM**, which is otherwise easy to blur.

- **Odometry** keeps only a sliding window of recent states and *marginalizes* old ones into a prior. Its factors are local (IMU, current-scan LiDAR, current-frame visual). It is fast and locally accurate but **drifts**: with no constraint tying $x_k$ to a state it visited long ago, errors accumulate monotonically. FAST-LIO and FAST-LIVO2 are odometry in this sense.

- **SLAM** additionally inserts **loop-closure factors** $p(z_{\text{loop}}\mid x_i, x_j)$ between temporally distant states that revisit the same place, plus optionally absolute factors (GNSS, §07). These global edges turn the chain (1.4) into a *graph with cycles*, and re-solving (1.5) redistributes accumulated error — global consistency. In Meridian this is the L3 back-end (keyframe factor graph, loop closure via place recognition L5), fed by keyframes the front-end emits. **This chapter is about the front-end estimator** (the local solve of (1.5)); place recognition and loop closure are the back-end's job, but they are *the same MAP objective with extra factors*, which is why the formulation here is the right foundation for both.

**The map is part of the model, not a by-product.** In a tightly-coupled direct system the residual $r_{\text{lidar}} = n\cdot p_W + d$ needs a *surface* $(n,d)$ to measure against, so the map is what defines the LiDAR factors at all — it is consulted *inside* the measurement model (`laserMapping.cpp:451-459`) on every iteration. Its data structure is therefore on the estimator's critical path. FAST-LIO2 maintains the map in an **incremental k-d tree (ikd-Tree)** supporting on-the-fly insertion, deletion, re-balancing and box queries at amortized $O(\log n)$ (`papers/2102.10808.txt:2-3`), so that nearest-surface association does not dominate the update. Map representations (k-d trees, voxel hashes, TSDF) and how they shape the LiDAR residual are §05; Meridian's layered map (adaptive voxel-hash → TSDF+RGB → mesh) is the L4 design that this section's "map is part of the model" view justifies.

---

### 1.6 The "direct" choice and the multi-sensor frontier

Two refinements of the tight-coupling story complete the motivation for Meridian's design and preview the chapter's later half.

**Direct vs. feature-based residuals (§05–§06).** Within tight coupling one still chooses *what* the residual is computed on. *Feature-based* methods first extract sparse geometric primitives (edges, planar patches — FAST-LIO's preprocessing supports a `Feature` enum and per-LiDAR feature extraction, `FAST_LIO/src/preprocess.h:2`) and form residuals only on those. *Direct* methods skip extraction and register *raw* measurements straight against the map. FAST-LIO2's first headline novelty is exactly this: *"directly registering raw points to the map … without extracting features … enables the exploitation of subtle features in the environment"* (`papers/2107.06829.txt:5-7`; the toggle is `feature_enabled=false`, `preprocess.h:1-2`). The direct approach loses less information (no thresholded feature selection throwing away points) — the same philosophy as tight coupling, applied one level down. **Meridian is direct on both modalities** (point-to-plane LiDAR, FAST-LIVO2-style sparse-direct photometric), so §05 and §06 develop the direct residuals in depth.

**Adding the camera, and sequential updates (§06, §09).** Extending the *same* MAP objective (1.5) with a photometric factor yields LiDAR-Inertial-Visual Odometry. FAST-LIVO2 fuses *"photometric errors and point-to-plane residuals"* with *"LiDAR-provided depth"* (no triangulation — the LiDAR map hands the visual factor its 3-D points) inside an *"efficient ESIKF with sequential update"* — LiDAR sub-update then visual sub-update within one step (`papers/2408.14035.txt:4-6`; `FAST-LIVO2/src/LIVMapper.cpp:2-3`). Sequential update is just a numerically convenient ordering of the *same* normal equations (1.6) when residual blocks are conditionally independent; §09 shows it is still one MAP solve. This is the template for Meridian's L2 front-end (LiDAR + visual + IMU + GNSS, tightly coupled), and the reason the chapter culminates in §12 by walking one full estimator step end-to-end mapped onto FAST-LIO2 / FAST-LIVO2.

**Alternative state designs exist.** The IMU need not be a pure input. Point-LIO instead places the IMU *inside the measurement model* and updates **per point at its own timestamp**, modeling angular/linear acceleration as states — removing in-scan motion-compensation and handling extreme angular rates (`Point-LIO/src/Estimator.cpp:2-6`). This is still the same MAP/factor-graph view with a different factorization of "what is a state" and "what is a measurement," and it is a reminder that (1.5) is the invariant while the front-end *implementation* is swappable — exactly the contract behind Meridian's `IFrontEnd` interface (iEKF v1, continuous-time §10 v2).

---

### 1.7 Roadmap of the chapter

With the problem (1.1), the architecture (tight, §1.2–1.3), and the unifying objective (1.5, §1.4) established, the remaining sections fill in the scaffold:

| § | Topic | Supplies to (1.5)/(1.6) |
|---|-------|--------------------------|
| **02** | State on manifolds: $SO(3)/SE(3)$, error state, $\boxplus/\boxminus$, tangent-space Jacobians | the variables $\mathcal X$ and how to differentiate $r$ w.r.t. $\delta$ |
| **03** | Probability: MAP $=$ NLS, Gaussians, factor graphs, information form | the derivation of (1.4)–(1.5) and the meaning of $\Omega$ |
| **04** | The IMU: model, bias, propagation, preintegration | the motion factor $r_{\text{imu}}$ and the prior in (1.6) |
| **05** | LiDAR residual: point-to-plane/line, association, direct vs. feature, the map | $r_{\text{lidar}} = n\!\cdot\!p_W + d$ and its $H$ |
| **06** | Visual residual: sparse-direct photometric, LiDAR depth (FAST-LIVO2) | $r_{\text{cam}}$ and its $H$ |
| **07** | GNSS & absolute residuals: frames, ENU, switchable constraints | $r_{\text{gnss}}$ and global factors |
| **08** | Solving I (batch): GN/LM, normal equations, sparsity, Schur | how (1.6) is assembled and inverted at scale |
| **09** | Solving II (recursive): iterated EKF/ESIKF, equivalence to GN-MAP, fixed-lag | why the filter at `esekfom.hpp:288-314` *is* (1.6) |
| **10** | Continuous-time: B-spline $T(t)$ on $SE(3)$, residuals on control points $c_k$ | a continuous reparameterization of $\mathcal X$ |
| **11** | Robustness: degeneracy/observability, robust kernels, GNC, switchable constraints, PCM | how the geometry of $\Omega$ (1.1) gates trust |
| **12** | Synthesis: one full tightly-coupled step on FAST-LIO2 / FAST-LIVO2 | the whole loop, concretely |

Everything that follows is, in the end, one sentence made rigorous: **build the right factors from raw measurements, then solve $\min_{\mathcal X}\sum_c \|r_c(\mathcal X)\|^2_{\Sigma_c}$ on the manifold — together, not separately.**

---

#### Sources grounded in this section

*Paper text (intros):* FAST-LIO `papers/2010.08196.txt` (loose/tight & information loss :21-30; complexity & contributions :35-45); FAST-LIO2 `papers/2107.06829.txt` (direct registration :5-7,13); FAST-LIVO2 `papers/2408.14035.txt` (sequential-update ESIKF, photometric + point-to-plane, LiDAR depth :4-6); ikd-Tree `papers/2102.10808.txt` (incremental map, amortized $O(\log n)$ :2-3).
*Code:* state manifold `FAST_LIO/include/use-ikfom.hpp:9-18`; process model $f$ `…use-ikfom.hpp:38-50`; measurement-packet `FAST_LIO/include/common_lib.h:2`; IMU propagate + de-skew `FAST_LIO/src/IMU_Processing.hpp:3`; point-to-plane measurement model `FAST_LIO/src/laserMapping.cpp:451-459`; main loop `…laserMapping.cpp:460-468`; iterated update / Gauss-Newton loop `FAST_LIO/include/IKFoM_toolkit/esekfom/esekfom.hpp:288-314` (predict :217); direct-vs-feature & multi-LiDAR `FAST_LIO/src/preprocess.h:1-2`; sequential LIVO update `FAST-LIVO2/src/LIVMapper.cpp:2-3`; per-point alternative `Point-LIO/src/Estimator.cpp:2-6`.

> *Note on reference fidelity:* the on-disk copies of the reference repositories and papers in `C:/Users/charl/Sources/slam-reference/` are an **abridged digest** — file structure, type declarations, key function signatures and the equation-bearing lines are preserved verbatim (e.g. the `state_ikfom` manifold and `get_f` above are exact), while long function bodies and full paper prose are summarized to anchor lines. Citations point at these anchors; for byte-exact bodies consult the upstream FAST-LIO / FAST-LIO2 / FAST-LIVO2 / Point-LIO sources. This does not affect any technical claim made here, each of which rests on a structurally preserved declaration or an explicit paper statement.
