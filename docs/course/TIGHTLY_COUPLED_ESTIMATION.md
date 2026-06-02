# Tightly-Coupled Multi-Sensor State Estimation & Residuals

> A graduate-level course chapter for the **Meridian** SLAM rebuild. Every section is grounded in the reference implementations (FAST-LIO2, FAST-LIVO2, Point-LIO, ikd-Tree) and the canonical papers; the matching grounding dossiers live in `../grounding/` and the implementation specs in `../specs/`. It builds from manifolds and probability up to one complete tightly-coupled estimator step, mapped concretely onto the FAST-LIO2 / FAST-LIVO2 code.


## Table of contents

- [01. Introduction: estimation problem & coupling spectrum & MAP](#01-introduction-estimation-problem-coupling-spectrum-map)
- [02. State on manifolds: SO(3)/SE(3), error state, ⊞/⊟](#02-state-on-manifolds-so3se3-error-state)
- [03. Probabilistic foundation: MAP=NLS, Gaussians, factor graphs](#03-probabilistic-foundation-mapnls-gaussians-factor-graphs)
- [04. IMU: model, propagation, preintegration, inertial residual](#04-imu-model-propagation-preintegration-inertial-residual)
- [05. LiDAR residual: point-to-plane, association, map](#05-lidar-residual-point-to-plane-association-map)
- [06. Visual residual: sparse-direct photometric + LiDAR depth](#06-visual-residual-sparse-direct-photometric-lidar-depth)
- [07. GNSS & absolute residuals: frames, ENU, switchable](#07-gnss-absolute-residuals-frames-enu-switchable)
- [08. Solving I (batch): GN, LM, normal equations, sparsity, Schur](#08-solving-i-batch-gn-lm-normal-equations-sparsity-schur)
- [09. Solving II (recursive): iterated EKF/ESIKF, equivalence, fixed-lag](#09-solving-ii-recursive-iterated-ekfesikf-equivalence-fixed-lag)
- [10. Continuous-time estimation: B-spline residuals](#10-continuous-time-estimation-b-spline-residuals)
- [11. Robustness: degeneracy/observability, robust kernels, GNC, switchable, PCM](#11-robustness-degeneracyobservability-robust-kernels-gnc-switchable-pcm)
- [12. Synthesis: one full estimator step, mapped to FAST-LIO2/LIVO2](#12-synthesis-one-full-estimator-step-mapped-to-fast-lio2livo2)


---


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


---


## 02. State on manifolds: SO(3)/SE(3), error state, ⊞/⊟

> **Role of this section in the chapter.** This is the section that *defines the
> notation used by every other section* of the chapter "Tightly-Coupled
> Multi-Sensor State Estimation & Residuals". Section [01](01_introduction.md)
> motivated the estimation problem and the MAP view; section
> [03](03_probabilistic_foundation.md) will turn MAP into nonlinear least
> squares; sections [04](04_imu.md)–[07](07_gnss.md) write residuals for the
> IMU, LiDAR, camera and GNSS; section [08](08_solving_batch.md) solves the
> resulting normal equations and section [09](09_solving_recursive.md) shows the
> recursive (iterated EKF) form that FAST-LIO2 actually runs. **Every one of
> those sections needs an answer to the same question first: what *is* the state
> $x$, what does it mean to perturb it, and how do you differentiate a residual
> with respect to it?** The state is not a vector — it lives on a manifold
> (rotations, the gravity direction, …). This section builds that machinery from
> the ground up and grounds it in the exact code Meridian's v1 front-end will mirror
> (`FAST_LIO/include/so3_math.h`, `use-ikfom.hpp`, and the IKFoM toolkit).

---

### 02.1 Why we cannot just use a vector

A naïve state estimator stacks everything into a column vector $x \in
\mathbb{R}^N$ and runs Gauss–Newton or a Kalman filter on it. That works when
all the quantities are genuinely Euclidean — positions, velocities, biases. It
*breaks* the moment a quantity has internal structure that a flat vector cannot
respect. The orientation of the platform is the canonical example.

There is no global, singularity-free, three-parameter representation of 3D
rotation. Euler angles gimbal-lock; a unit quaternion has four numbers tied by a
norm constraint $\lVert q\rVert = 1$ that the optimiser does not know about and
will happily violate; a $3\times 3$ rotation matrix has nine numbers tied by six
constraints ($R^\top R = I$, $\det R = +1$). If you treat any of these as a free
vector and add a Gauss–Newton step $\delta$ to it, you leave the set of valid
rotations: $R + \delta$ is not a rotation, $q + \delta$ is not a unit quaternion.

The fix, due to the **encapsulation** idea formalised in the FAST-LIO line of
work (the original paper writes "we represent the state and noise as
*encapsulated* in a manifold $\mathcal M$ and its tangent space",
`papers/2010.08196.txt`), is to stop treating the state as a vector and instead
treat it as a point on a **smooth manifold** $\mathcal M$. We never add a vector
to a manifold point directly. Instead we (i) take a *minimal* perturbation in the
flat tangent space $\mathbb{R}^n$ at the current point, and (ii) *retract* it
back onto the manifold with a structure-preserving operator. Those two operators
are written $\boxplus$ ("boxplus") and its inverse $\boxminus$ ("boxminus").

This single change buys us three things that recur throughout the chapter:

1. **Minimal, singularity-free local coordinates.** The tangent space has
   exactly the dimension of the manifold ($3$ for $SO(3)$, $2$ for the gravity
   direction $S^2$), so the optimiser solves a well-conditioned, minimal normal
   system with no spurious constraints (sec. [08](08_solving_batch.md)).
2. **A clean definition of the *error state*.** The difference between two
   manifold points is a tangent vector $\delta = x_1 \boxminus x_2 \in
   \mathbb{R}^n$. Covariances $\Sigma$ and the information matrix
   $\Omega=\Sigma^{-1}$ live in this tangent space, not on the manifold. This is
   what makes the (iterated) EKF of sec. [09](09_solving_recursive.md)
   well-defined.
3. **A uniform interface.** Heterogeneous quantities — a rotation, a gravity
   direction, a velocity — are composed into one product manifold and handled by
   one pair of operators. FAST-LIO2's 23-DoF state is exactly such a product
   (sec. 02.7).

---

### 02.2 Groups, manifolds, Lie groups: the minimum we need

A **group** $(G,\cdot)$ is a set with an associative composition $\cdot$, an
identity $e$, and an inverse for every element. Rotations form a group: composing
two rotations gives a rotation, the identity is "no rotation", and every rotation
can be undone.

A **smooth manifold** is a set that looks locally like $\mathbb{R}^n$ — around
every point you can lay down smooth local coordinates. $n$ is the *dimension* (the
number of *degrees of freedom*, DoF). The set of rotations is a 3-DoF manifold:
locally it takes three numbers (a small rotation about each axis) to move around.

A **Lie group** is both at once: a group that is also a smooth manifold, with
smooth composition and inversion. This double structure is the whole game. The
group structure lets us *compose* and *invert* states exactly; the manifold
structure lets us *perturb* and *differentiate* them.

The two Lie groups we need are:

- $SO(3)$ — the **Special Orthogonal group** in 3D, the rotations:
  $$ SO(3) = \{\, R \in \mathbb{R}^{3\times 3} \;:\; R^\top R = I,\ \det R = +1 \,\}. $$
  It is 3-dimensional (9 entries minus 6 constraints).
- $SE(3)$ — the **Special Euclidean group**, the rigid-body poses (rotation +
  translation):
  $$ SE(3) = \left\{\, T=\begin{bmatrix} R & p\\ 0^\top & 1\end{bmatrix} \;:\; R\in SO(3),\ p\in\mathbb{R}^3 \,\right\}, $$
  acting on a homogeneous point as $T\,[x;1] = [Rx+p;\,1]$. It is 6-dimensional
  (3 rotational + 3 translational).

To these the FAST-LIO2 state adds one more, non-group manifold:

- $S^2$ — the **2-sphere**, here the set of gravity *directions* of fixed
  magnitude:
  $$ S^2_g = \{\, g\in\mathbb{R}^3 : \lVert g\rVert = g_0 \,\}, \qquad g_0 \approx 9.81\,\text{m/s}^2 . $$
  It is a 2-DoF manifold (a direction on a sphere) but **not** a group — there is
  no natural "composition of two gravity vectors". We will see in sec. 02.6 that
  $\boxplus/\boxminus$ are *more general than* the Lie-group exp/log and handle
  $S^2$ cleanly anyway. This is precisely why FAST-LIO2 can keep gravity on a
  2-DoF sphere instead of as a free 3-vector.

Throughout, Meridian will lean on $SO(3)$ for orientation and extrinsic rotations,
$\mathbb{R}^n$ for the Euclidean quantities, and $S^2$ for gravity — i.e. exactly
the product manifold built by `MTK_BUILD_MANIFOLD(state_ikfom, …)` in
`FAST_LIO/include/use-ikfom.hpp:9`. $SE(3)$ matters conceptually (it is *the*
pose group, and the continuous-time B-spline trajectory of sec.
[10](10_continuous_time.md) lives on $SE(3)$), but FAST-LIO2 deliberately does
*not* couple rotation and translation into a single $SE(3)$ block — it keeps
$R$ and $p$ as separate manifold members $SO(3)\times\mathbb{R}^3$ (sec. 02.5
explains the consequence for Jacobians).

---

### 02.3 The hat operator and the Lie algebra $\mathfrak{so}(3)$

Every Lie group has a **Lie algebra** — its tangent space at the identity,
equipped with a bracket. For $SO(3)$ the tangent space at the identity is the set
of $3\times 3$ **skew-symmetric** matrices, written $\mathfrak{so}(3)$. We
identify it with $\mathbb{R}^3$ through the **hat operator** $(\cdot)^\wedge$,
which turns a 3-vector into the skew-symmetric matrix that implements the cross
product:

$$
\boldsymbol\omega = \begin{bmatrix}\omega_1\\\omega_2\\\omega_3\end{bmatrix}
\;\longmapsto\;
\boldsymbol\omega^\wedge \;=\;
\begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix},
\qquad
\boldsymbol\omega^\wedge\, v \;=\; \boldsymbol\omega \times v \quad \forall v\in\mathbb{R}^3 .
$$

The inverse map, $(\cdot)^\vee$, reads the vector back out of the skew matrix.

This is *literally* the FAST-LIO code. In `so3_math.h:5` the macro

```cpp
#define SKEW_SYM_MATRX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0
```

lays out the nine entries of $\boldsymbol\omega^\wedge$ in row-major order — read
across, that is $\{0,-\omega_3,\omega_2;\ \omega_3,0,-\omega_1;\ -\omega_2,
\omega_1,0\}$, exactly the matrix above. The same macro is reused inside every
`Exp(...)` overload. The standalone version is `so3_math.h:68`:

```cpp
template<typename T>
Eigen::Matrix<T, 3, 3> hat(const Eigen::Matrix<T, 3, 1> &v) {
    Eigen::Matrix<T, 3, 3> Omega;
    Omega <<  0, -v(2),  v(1),
            v(2),     0, -v(0),
           -v(1),  v(0),     0;
    return Omega;
}
```

Three identities we will use repeatedly when deriving Jacobians (sec. 02.5,
[04](04_imu.md), [05](05_lidar.md)):

- **Antisymmetry / cross-product swap:** $a^\wedge b = -\,b^\wedge a = a\times b$.
- **Rotation conjugation:** for $R\in SO(3)$, $\;R\,a^\wedge R^\top = (Ra)^\wedge$.
  (Rotating a cross-product is the cross-product of the rotated vectors.)
- **Triple-product / nesting:** $a^\wedge a^\wedge = a a^\top - \lVert a\rVert^2 I$,
  which is what makes the Rodrigues series close in finite form (next).

The **bracket** of $\mathfrak{so}(3)$ is the matrix commutator $[\,a^\wedge,
b^\wedge\,] = (a\times b)^\wedge$; we will not need it directly but it is what
formally makes $\mathfrak{so}(3)$ a Lie algebra.

---

### 02.4 Exp and Log: the exponential map and Rodrigues' formula

The bridge between the Lie *algebra* (the flat tangent space $\mathbb{R}^3$) and
the Lie *group* (the curved manifold $SO(3)$) is the **exponential map**. For a
rotation, $\mathrm{Exp}(\boldsymbol\phi)$ takes an axis-angle vector
$\boldsymbol\phi\in\mathbb{R}^3$ (direction = rotation axis, magnitude
$\theta=\lVert\boldsymbol\phi\rVert$ = rotation angle in radians) to the
corresponding rotation matrix. We write it **capitalised** $\mathrm{Exp}:
\mathbb{R}^3 \to SO(3)$ to mean "the matrix exponential of the hat", reserving
lowercase $\exp$ for the abstract group exponential:

$$
\mathrm{Exp}(\boldsymbol\phi) \;=\; \exp(\boldsymbol\phi^\wedge)
\;=\; \sum_{k=0}^{\infty} \frac{(\boldsymbol\phi^\wedge)^k}{k!}.
$$

The infinite series is not needed: because $\boldsymbol\phi^\wedge$ is skew, the
nesting identity above collapses the series into the closed-form **Rodrigues'
rotation formula**. With unit axis $a=\boldsymbol\phi/\theta$ and $K=a^\wedge$:

$$
\boxed{\;\mathrm{Exp}(\boldsymbol\phi) \;=\; I \;+\; \sin\theta\,K \;+\; (1-\cos\theta)\,K^2\;}
\qquad (\theta = \lVert\boldsymbol\phi\rVert).
\tag{02.1}
$$

This is *exactly* `so3_math.h:6-23`:

```cpp
template<typename T>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &&ang) {
    T ang_norm = ang.norm();                                   // theta
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
    if (ang_norm > 0.0000001) {
        Eigen::Matrix<T, 3, 1> r_axis = ang / ang_norm;        // a = phi/theta
        Eigen::Matrix<T, 3, 3> K;
        K << SKEW_SYM_MATRX(r_axis);                            // K = a^
        /// Roderigous Tranformation
        return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
    } else {
        return Eye3;                                           // small-angle: Exp(0)=I
    }
}
```

Two engineering details worth internalising, both visible in the code:

- **The small-angle guard.** When $\theta \to 0$, $a=\boldsymbol\phi/\theta$ is
  numerically $0/0$. The code branches at `ang_norm > 1e-7` and returns $I$
  directly. The mathematically exact first-order behaviour is
  $\mathrm{Exp}(\boldsymbol\phi)\approx I + \boldsymbol\phi^\wedge$, so returning
  $I$ is correct to first order; a more careful implementation keeps the
  $\boldsymbol\phi^\wedge$ term, but for the per-iteration *corrections* in an
  iterated EKF (which are tiny) returning $I$ when $\theta$ is sub-$10^{-7}$ is
  harmless.
- **Two more overloads.** `so3_math.h:24` is `Exp(ang_vel, dt)` — it forms
  $\mathrm{Exp}(\boldsymbol\omega\,dt)$ for IMU propagation (the discrete
  rotation increment over a time step $dt$, used in sec. [04](04_imu.md)); and
  `so3_math.h:43` is a scalar-argument convenience overload. All three share
  the same Rodrigues body.

The **inverse**, the **logarithm map** $\mathrm{Log}: SO(3)\to\mathbb{R}^3$,
recovers the axis-angle vector from a rotation matrix. The angle comes from the
trace ($\operatorname{tr}R = 1 + 2\cos\theta$) and the axis from the
skew-symmetric part of $R$:

$$
\theta = \arccos\!\Big(\tfrac{\operatorname{tr}R - 1}{2}\Big),
\qquad
\mathrm{Log}(R) = \frac{\theta}{2\sin\theta}\,
\begin{bmatrix} R_{32}-R_{23}\\ R_{13}-R_{31}\\ R_{21}-R_{12}\end{bmatrix}.
\tag{02.2}
$$

`so3_math.h:61-67`:

```cpp
template<typename T>
Eigen::Matrix<T, 3, 1> Log(const Eigen::Matrix<T, 3, 3> &R) {
    T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
    Eigen::Matrix<T, 3, 1> K(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
    return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}
```

Note the two guards mirroring the two singular regimes: near $R=I$
($\operatorname{tr}R\to 3$) the angle is set to $0$; near small $\theta$ the
prefactor $\tfrac{\theta}{2\sin\theta}\to\tfrac12$, so the code returns
$\tfrac12 K$ with $K$ the vee of $(R-R^\top)$. (Near $\theta=\pi$ the formula is
ill-conditioned — $\sin\theta\to 0$ in the denominator — a known corner case that
this terse implementation does not special-case; for the small per-step
corrections in LIO it never arises.)

> **Worked micro-example.** Take $\boldsymbol\phi = (0,0,\pi/2)^\top$ (a 90°
> yaw). Then $\theta=\pi/2$, $a=(0,0,1)$, $K=a^\wedge=\begin{smallmatrix}0&-1&0\\1&0&0\\0&0&0\end{smallmatrix}$.
> Rodrigues (02.1): $\mathrm{Exp}(\boldsymbol\phi)=I + 1\cdot K + 1\cdot K^2$. With
> $K^2=\operatorname{diag}(-1,-1,0)$ this gives
> $\begin{smallmatrix}0&-1&0\\1&0&0\\0&0&1\end{smallmatrix}$ — the standard 90°
> yaw matrix. Running $\mathrm{Log}$ back: $\operatorname{tr}=1$, so
> $\theta=\arccos(0)=\pi/2$; $K_{\text{vec}}=(0,0,2)$; prefactor
> $\tfrac{\pi/2}{2\cdot 1}=\pi/4$; result $(0,0,\pi/2)$. Round-trip closes. ∎

---

### 02.5 The error state and right/left Jacobians

We now make precise what "perturbing a rotation" means and how a residual
differentiates with respect to it.

**Left vs right perturbation.** A perturbation of $R$ can be injected on either
side:

$$
R \;=\; \mathrm{Exp}(\boldsymbol\phi_L)\,\bar R \quad\text{(left)},
\qquad
R \;=\; \bar R\,\mathrm{Exp}(\boldsymbol\phi_R) \quad\text{(right)}.
$$

The *left* perturbation $\boldsymbol\phi_L$ lives in the **global/world** frame
(it pre-multiplies, rotating the already-rotated result in the fixed frame); the
*right* perturbation $\boldsymbol\phi_R$ lives in the **body/local** frame (it
post-multiplies, acting before $\bar R$, i.e. in the body's own axes). They are
related by the adjoint: $\boldsymbol\phi_L = \bar R\,\boldsymbol\phi_R$. Neither
is "more correct"; what matters is *consistency* — you must pick one convention
and use it everywhere, because the Jacobians differ.

**FAST-LIO2 / IKFoM uses the right perturbation.** This is visible in the
boxplus definition the toolkit implements (sec. 02.6) and the residual Jacobians
in `esekfom.hpp`. With the right convention the **error state** of the rotation
is $\delta\boldsymbol\phi$ such that the true rotation is
$R = \bar R\,\mathrm{Exp}(\delta\boldsymbol\phi)$, i.e.
$\delta\boldsymbol\phi = \mathrm{Log}(\bar R^\top R) = R \boxminus \bar R$.

**Why we need Jacobians of Exp.** When a residual $r(x)$ depends on a rotation
and we want $\partial r/\partial(\delta\boldsymbol\phi)$, the chain rule pushes a
derivative through $\mathrm{Exp}$. Two derivatives recur:

1. **Differentiating a rotated vector.** For a fixed $v$, with the right
   perturbation,
   $$
   \frac{\partial}{\partial\delta\boldsymbol\phi}\Big[\,\bar R\,\mathrm{Exp}(\delta\boldsymbol\phi)\,v\,\Big]_{\delta\boldsymbol\phi=0}
   \;=\; -\,\bar R\,v^\wedge .
   \tag{02.3}
   $$
   *Derivation:* expand $\mathrm{Exp}(\delta\boldsymbol\phi)v \approx (I +
   \delta\boldsymbol\phi^\wedge)v = v + \delta\boldsymbol\phi^\wedge v = v -
   v^\wedge\delta\boldsymbol\phi$ (using $a^\wedge b=-b^\wedge a$). Multiply by
   $\bar R$ and read off the coefficient of $\delta\boldsymbol\phi$. The same
   step, done with the *left* perturbation, gives $-(\bar R v)^\wedge$ — note the
   hat is on the *rotated* vector, a different matrix. This single identity is the
   workhorse behind the point-to-plane LiDAR Jacobian (sec. [05](05_lidar.md))
   and the photometric Jacobian (sec. [06](06_visual.md)).

   It is exactly the term you see in FAST-LIO's process Jacobian
   `use-ikfom.hpp:71`:
   ```cpp
   cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix() * MTK::hat(acc_);
   ```
   i.e. $\partial(\dot v)/\partial(\delta\boldsymbol\phi) = -R\,(a-b_a)^\wedge$ —
   the derivative of the rotated acceleration $R(a-b_a)$ w.r.t. the attitude
   error, precisely $-\bar R\,v^\wedge$ from (02.3) with $v=a-b_a$ (the inertial
   acceleration residual, sec. [04](04_imu.md)).

2. **The right Jacobian of $SO(3)$.** When the *perturbation* of $R$ propagates to
   the perturbation of $\mathrm{Log}(R)$ (i.e. differentiating Exp/Log itself, as
   in covariance propagation), the chain rule produces the **right Jacobian**
   $J_r(\boldsymbol\phi)$ and its inverse:
   $$
   \mathrm{Exp}(\boldsymbol\phi + \delta\boldsymbol\phi) \approx \mathrm{Exp}(\boldsymbol\phi)\,\mathrm{Exp}\!\big(J_r(\boldsymbol\phi)\,\delta\boldsymbol\phi\big),
   $$
   $$
   \boxed{\,J_r(\boldsymbol\phi) = I - \frac{1-\cos\theta}{\theta^2}\,\boldsymbol\phi^\wedge + \frac{\theta-\sin\theta}{\theta^3}\,(\boldsymbol\phi^\wedge)^2\,}
   \tag{02.4}
   $$
   $$
   J_r^{-1}(\boldsymbol\phi) = I + \tfrac12\,\boldsymbol\phi^\wedge + \Big(\tfrac{1}{\theta^2} - \tfrac{1+\cos\theta}{2\theta\sin\theta}\Big)(\boldsymbol\phi^\wedge)^2 .
   $$
   The **left Jacobian** satisfies $J_l(\boldsymbol\phi) =
   J_r(-\boldsymbol\phi) = J_r(\boldsymbol\phi)^\top$ and shows up with the left
   convention. As $\theta\to0$ both tend to $I - \tfrac12\boldsymbol\phi^\wedge$
   (right) / $I + \tfrac12\boldsymbol\phi^\wedge$ (left), and to first order both
   are just $I$.

This right Jacobian is exactly what IKFoM calls `MTK::A_matrix(·)` in the
covariance-propagation step of the predictor. In
`IKFoM_toolkit/esekfom/esekfom.hpp` (the `predict` method, around line 420) the
attitude block of the discrete state-transition matrix is built as

```cpp
res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();   // J_r(phi)^T  acts on the SO(3) error block
F_x1.template block<3, 3>(idx, idx) = MTK::A_matrix(seg_SO3).transpose();
```

where `seg_SO3` is the incremental rotation $\boldsymbol\phi=\boldsymbol\omega\,dt$
applied this step. `A_matrix` is the closed form (02.4); transposing it converts
between the right- and left-Jacobian roles needed to map the error state across
the predict. The reader should take away that **the abstract identity (02.4) and
the concrete `A_matrix` call are the same object** — when Meridian writes its own CT
predictor (sec. [10](10_continuous_time.md)) this is the matrix it must supply.

---

### 02.6 ⊞ and ⊟: the boxplus/boxminus encapsulation

We can now state the operators that the rest of the chapter uses without further
comment. For a manifold $\mathcal M$ of dimension $n$:

$$
\boxplus:\ \mathcal M \times \mathbb{R}^n \to \mathcal M,
\qquad
\boxminus:\ \mathcal M \times \mathcal M \to \mathbb{R}^n,
$$

with the two consistency laws ("they are inverses of each other"):

$$
x \boxplus (y \boxminus x) = y,
\qquad
(x \boxplus u) \boxminus x = u
\qquad (\text{for } u \text{ in a neighbourhood of } 0).
\tag{02.5}
$$

Read $x\boxplus u$ as "start at the manifold point $x$, walk by the minimal
tangent vector $u$, land on a new manifold point", and $y\boxminus x$ as "the
minimal tangent vector that walks from $x$ to $y$". The FAST-LIO paper states the
same bijection: the operators "establish a bijective mapping between a local
neighborhood on $\mathcal M$ and its tangent space" (`papers/2010.08196.txt`).

The concrete definitions for the building blocks Meridian uses are:

| manifold $\mathcal M$ | dim | $x \boxplus u$ | $y \boxminus x$ |
|---|---|---|---|
| $\mathbb{R}^n$ (pos, vel, biases, extrinsic transl.) | $n$ | $x + u$ | $y - x$ |
| $SO(3)$ (attitude, extrinsic rotation) | $3$ | $R\,\mathrm{Exp}(u)$ | $\mathrm{Log}(x^\top y)$ |
| $S^2$ (gravity direction) | $2$ | see below | see below |

The $SO(3)$ row uses the **right** convention — $R\,\mathrm{Exp}(u)$, the body
perturbation of sec. 02.5 — and this is the convention the whole toolkit and
therefore all our Jacobians assume. The FAST-LIO paper writes it identically:
$R \boxplus r = R\,\mathrm{Exp}(r)$ and $R_1 \boxminus R_2 = \mathrm{Log}(R_2^\top
R_1)$ (`papers/2010.08196.txt`).

For a **compound (product) manifold** $\mathcal M = \mathcal M_1 \times \cdots
\times \mathcal M_K$ the operators act **block-wise** — each member is retracted
with its own $\boxplus$, and the tangent vectors stack:

$$
(x_1,\dots,x_K) \boxplus (u_1,\dots,u_K) = (x_1\boxplus u_1,\ \dots,\ x_K\boxplus u_K),
\qquad
(y \boxminus x)_k = y_k \boxminus x_k .
\tag{02.6}
$$

The paper makes the same statement ("for a compound manifold $\mathcal M =
\mathcal M_1\times \mathcal M_2$ … $(x_1,x_2)\boxplus(u_1,u_2) = (x_1\boxplus
u_1,\ x_2\boxplus u_2)$", `papers/2010.08196.txt`). **This block-wise rule is the
entire reason the heterogeneous state can be handled by one optimiser**: the
estimator only ever sees a flat tangent vector $\delta x \in \mathbb{R}^n$ and one
pair of operators; all the manifold-specific curvature is hidden inside the
per-member $\boxplus/\boxminus$.

**The gravity manifold $S^2$.** Because $S^2$ is not a Lie group, its
$\boxplus/\boxminus$ are not exp/log of a group, but they fit the same interface.
The retraction takes a 2-DoF tangent vector $u\in\mathbb{R}^2$ in the local
2-plane orthogonal to the current direction $\bar g$, lifts it to a 3-vector in
that tangent plane via a basis $B(\bar g)\in\mathbb{R}^{3\times2}$, and rotates
$\bar g$ by it:

$$
g \boxplus u = \mathrm{Exp}\!\big(B(\bar g)\,u\big)\,\bar g,
\qquad
g \boxminus \bar g = B(\bar g)^\top \,\theta\,a,
$$

where $\theta a = \mathrm{Log}$ of the rotation taking $\bar g$ to $g$. The two
toolkit hooks that implement the lift and its Jacobian are `S2_Mx` and
`S2_Nx_yy`. We do **not** need their internals here; what matters is that gravity
stays a 2-DoF quantity throughout. The state-Jacobian uses `S2_Mx` directly —
`use-ikfom.hpp:72-74`:

```cpp
Eigen::Matrix<double, 3, 2> grav_matrix;
s.S2_Mx(grav_matrix, vec, 21);                       // 3x2 lift B(g) at the gravity slot (index 21)
cov.template block<3, 2>(12, 21) = grav_matrix;      // d(vel_dot)/d(gravity error), a 2-DoF column
```

i.e. the velocity dynamics $\dot v = R(a-b_a) + g$ depend on the gravity *error*
only through a $3\times 2$ block — two columns, not three — because gravity is on
$S^2$. (Contrast sec. 02.7's discussion of why the *covariance* dimension is $23$
not $24$.)

---

### 02.7 The FAST-LIO2 product manifold: $SO(3)\times S^2 \times \mathbb{R}^{18}$

We can now read FAST-LIO2's state declaration as one line of mathematics. From
`use-ikfom.hpp:9-18`:

```cpp
MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos))         // p          : position            R^3
((SO3,   rot))         // R          : attitude            SO(3)
((SO3,   offset_R_L_I))// R_{L}^{I}  : LiDAR->IMU rotation  SO(3)  (online extrinsic)
((vect3, offset_T_L_I))// p_{L}^{I}  : LiDAR->IMU transl.   R^3
((vect3, vel))         // v          : velocity            R^3
((vect3, bg))          // b_g        : gyro bias            R^3
((vect3, ba))          // b_a        : accel bias           R^3
((S2,    grav))        // g          : gravity direction    S^2
);
```

As a product manifold, using the chapter's shared notation,

$$
x \;=\; \big(\,p,\; R,\; R_L^I,\; p_L^I,\; v,\; b_g,\; b_a,\; g\,\big)
\ \in\
\underbrace{\mathbb{R}^3}_{p}\times
\underbrace{SO(3)}_{R}\times
\underbrace{SO(3)}_{R_L^I}\times
\underbrace{\mathbb{R}^3}_{p_L^I}\times
\underbrace{\mathbb{R}^3}_{v}\times
\underbrace{\mathbb{R}^3}_{b_g}\times
\underbrace{\mathbb{R}^3}_{b_a}\times
\underbrace{S^2}_{g}.
\tag{02.7}
$$

**Two dimensions to keep straight — and they are not equal.**

- **Manifold "storage" / ambient dimension = 24.** Each $SO(3)$ and the $S^2$
  store more numbers than their DoF; in the toolkit's *full* (non-minimal)
  bookkeeping the process model returns a length-**24** vector. You see this in
  `use-ikfom.hpp:48-62`, where `get_f` builds `Eigen::Matrix<double, 24, 1>` and
  fills $\dot p = v$ (rows 0–2), the angular-rate slot (rows 3–5), and
  $\dot v = R(a-b_a) + g$ (rows 12–14). The $3$-slot for the rotation rate is why
  the count is $24$: gravity is given a 3-row slot here even though its error is
  2-DoF.

- **Tangent / error-state / covariance dimension = 23.** The minimal error state
  $\delta x$ — what the EKF covariance $\Sigma$ and information $\Omega$ actually
  live on, and what the optimiser solves for — has dimension
  $$
  \dim = \underbrace{3}_{p}+\underbrace{3}_{R}+\underbrace{3}_{R_L^I}+\underbrace{3}_{p_L^I}+\underbrace{3}_{v}+\underbrace{3}_{b_g}+\underbrace{3}_{b_a}+\underbrace{2}_{g} = 23 .
  $$
  Gravity contributes **2**, not 3, because it is on $S^2$. This is why the
  process Jacobian in `use-ikfom.hpp:63` is declared
  `Eigen::Matrix<double, 24, 23> df_dx` — **24 ambient rows, 23 error-state
  columns** — and why the gravity block written via `S2_Mx` is $3\times 2$
  (sec. 02.6). The full error state, in the chapter's notation, is

  $$
  \delta x = \big(\delta p,\ \delta\boldsymbol\phi,\ \delta\boldsymbol\phi_{L}^{I},\ \delta p_L^I,\ \delta v,\ \delta b_g,\ \delta b_a,\ \delta g\big) \in \mathbb{R}^{23},
  $$

  related to the true state by $x = \bar x \boxplus \delta x$ using the block-wise
  rule (02.6): the two $SO(3)$ members retract via $R\,\mathrm{Exp}(\cdot)$, the
  $\mathbb{R}^3$ members by addition, and gravity via the $S^2$ retraction.

**The error-state in the predict and update.** The toolkit's filter is literally
written in $\boxplus/\boxminus$:

- *Predict* (`esekfom.hpp`, `predict`): the mean is advanced on the manifold by a
  single boxplus-style step, `x_.oplus(f_, dt)` — for the $SO(3)$ members this
  composes $R\leftarrow R\,\mathrm{Exp}(\boldsymbol\omega\,dt)$ and for $S^2$ it
  rotates the gravity direction, while the covariance is propagated through the
  $23\times 23$ transition matrix whose attitude block is the right Jacobian
  `A_matrix` of sec. 02.5 and whose gravity block uses `S2_Nx_yy` / `S2_Mx`.

- *Update* (`esekfom.hpp`, `update_iterated_dyn_share_modified`): the iterated EKF
  measures the current iterate's deviation from the propagated state as an
  **error-state boxminus**,
  ```cpp
  Matrix<scalar_type, n, 1> dx_new = x_.boxminus(x_propagated);   // delta = x_i (-) x_prop
  ```
  i.e. $\delta_i = \hat x_i \boxminus \hat x_{\text{prop}} \in \mathbb{R}^{23}$,
  exactly the quantity the prior term penalises in the MAP cost. After solving
  the linear system the new iterate is formed by a **boxplus** of the increment,
  $\hat x_{i+1} = \hat x_i \boxplus \Delta$. This is the on-manifold iterated
  Gauss–Newton that sec. [09](09_solving_recursive.md) shows is equivalent to the
  MAP solve of sec. [08](08_solving_batch.md); here we only flag that *every*
  $+$/$-$ on the state in that algorithm is really $\boxplus$/$\boxminus$.

---

### 02.8 $SE(3)$ as one block — and why FAST-LIO2 splits it

The $SE(3)$ pose can be treated as a *single* Lie group with its own exponential.
A twist $\xi = (\rho,\boldsymbol\phi)\in\mathbb{R}^6$ (translational part $\rho$,
rotational part $\boldsymbol\phi$) maps to a pose via

$$
\mathrm{Exp}_{SE(3)}(\xi) =
\begin{bmatrix} \mathrm{Exp}(\boldsymbol\phi) & V(\boldsymbol\phi)\,\rho \\ 0^\top & 1 \end{bmatrix},
\qquad
V(\boldsymbol\phi) = I + \tfrac{1-\cos\theta}{\theta^2}\boldsymbol\phi^\wedge + \tfrac{\theta-\sin\theta}{\theta^3}(\boldsymbol\phi^\wedge)^2 ,
$$

where $V$ (which equals the left Jacobian $J_l$) couples rotation into
translation. Its $\boxplus$ is $T\boxplus\xi = T\,\mathrm{Exp}_{SE(3)}(\xi)$.

FAST-LIO2 deliberately does **not** use this coupled block; in (02.7) the
position $p$ and attitude $R$ are *separate* members
$\mathbb{R}^3 \times SO(3)$. The practical difference: with the split, a
translation perturbation is a plain $\delta p$ in world coordinates and the
coupling matrix $V$ never appears, which keeps the Jacobians in `use-ikfom.hpp`
simple (e.g. $\partial\dot p/\partial\delta p$ falls out trivially, and the
$\partial\dot v/\partial\delta\boldsymbol\phi = -R\,v^\wedge$ block of (02.3) is
the *only* rotation coupling). The full $SE(3)$ encapsulation matters for Meridian
later — the **continuous-time B-spline trajectory** of sec.
[10](10_continuous_time.md) places its control points $c_k$ on $SE(3)$, and there
the coupled exponential and its (more involved) Jacobians are unavoidable. The
$\mathbb{R}^3\times SO(3)$ split is the right choice for the discrete iEKF v1;
the $SE(3)$ form is the right choice for the CT v2. Both are just different
product structures fed to the *same* $\boxplus/\boxminus$ interface, which is the
whole point of building the front-end behind an `IFrontEnd` boundary.

---

### 02.9 Notation summary (used by the whole chapter)

These symbols are **defined here and used unqualified everywhere else** in the
chapter.

| symbol | meaning |
|---|---|
| $x$ | full state (a point on the product manifold $\mathcal M$) |
| $\delta x$ | error state, a tangent vector in $\mathbb{R}^n$ ($n=23$ for FAST-LIO2) |
| $R\in SO(3)$ | platform attitude (rotation, body→world) |
| $p,\ v$ | position, velocity (world frame), $\in\mathbb{R}^3$ |
| $b_g,\ b_a$ | gyroscope bias, accelerometer bias, $\in\mathbb{R}^3$ |
| $g\in S^2$ | gravity direction (fixed magnitude $g_0\approx9.81$) |
| $R_L^I,\ p_L^I$ | LiDAR→IMU extrinsic rotation / translation (online-refined) |
| $(\cdot)^\wedge$ | hat: $\mathbb{R}^3\to\mathfrak{so}(3)$ skew matrix; $a^\wedge b = a\times b$ |
| $(\cdot)^\vee$ | vee: inverse of hat |
| $\mathrm{Exp},\ \mathrm{Log}$ | capitalised maps $\mathbb{R}^3\leftrightarrow SO(3)$ (Rodrigues, eq. 02.1–02.2) |
| $\exp,\ \log$ | abstract group exp/log (matrix exp of the algebra element) |
| $\boxplus,\ \boxminus$ | "⊞" / "⊟": retraction onto / difference on the manifold (eq. 02.5) |
| $J_r,\ J_l$ | right / left Jacobian of $SO(3)$ (eq. 02.4); $J_l(\phi)=J_r(\phi)^\top$ |
| $r,\ z,\ h(x)$ | residual, measurement, measurement prediction |
| $H$ | Jacobian $\partial r/\partial\delta x$ (taken w.r.t. the *error state*) |
| $\Sigma,\ \Omega=\Sigma^{-1}$ | covariance / information (live in the tangent space $\mathbb{R}^n$) |
| $K$ | Kalman gain (sec. [09](09_solving_recursive.md)) |
| $p_L,\ n,\ d$ | LiDAR point, plane normal, plane offset; plane: $n\cdot x + d = 0$ (sec. [05](05_lidar.md)) |
| $I,\ \pi,\ K_{\text{cam}}$ | image intensity, camera projection, intrinsics (sec. [06](06_visual.md)) |
| $c_k,\ T(t)$ | B-spline control points, $SE(3)$ trajectory (sec. [10](10_continuous_time.md)) |

**The three load-bearing facts to carry forward.** (1) The state is a *product
manifold*; we perturb it with one tangent vector $\delta x$ via the block-wise
$\boxplus$ (02.6). (2) Residual Jacobians are *always* taken w.r.t. the error
state $\delta x$, and the single most-used building block is the rotated-vector
derivative $-R\,v^\wedge$ (02.3), grounded in `use-ikfom.hpp:71`. (3) The
attitude error propagates through the right Jacobian $J_r$ = `A_matrix` (02.4);
gravity stays 2-DoF on $S^2$, which is exactly why the FAST-LIO2 covariance is
$23\times 23$, not $24\times 24$. Sections [03](03_probabilistic_foundation.md)
onward build every residual and every solver on top of these three facts.


---


## 03. Probabilistic foundation: MAP=NLS, Gaussians, factor graphs

> **Where this sits.** Section 01 placed us on the loose/tight coupling spectrum and announced that *all* of it is one estimation problem. Section 02 gave us the geometric stage: the state lives on a manifold $\mathcal{M}$, errors live in the tangent space, and we manipulate them with $\boxplus$/$\boxminus$, $\mathrm{Exp}/\mathrm{Log}$, and the hat operator $(\cdot)^\wedge$. This section is the **theoretical spine** that connects the two. We answer one question with full rigor: *given a probabilistic model of the sensors, what number do we actually compute, and why is that number a nonlinear least-squares problem?* Everything downstream — the IMU residual (§04), the LiDAR point-to-plane residual (§05), the photometric residual (§06), GNSS (§07), the batch solvers (§08), the recursive/iterated-Kalman solvers (§09), continuous-time (§10), and the robustness machinery (§11) — is a special case or a refinement of what we derive here. §12 reassembles all of it into one estimator step.
>
> **Notation reminder (defined in §02).** State $x \in \mathcal{M}$; rotation $R \in SO(3)$; position $p$; velocity $v$; biases $b_g, b_a$; gravity $g$. Boxplus/boxminus $\boxplus,\boxminus$; $\mathrm{Exp}/\mathrm{Log}$ between $\mathbb{R}^3$ and $SO(3)$; hat $(\cdot)^\wedge$. Residual $r$, measurement $z$, prediction $h(x)$, Jacobian $H$, covariance $\Sigma$, information $\Omega = \Sigma^{-1}$, Kalman gain $K$.

---

### 03.1 The object we are estimating, and the object we are computing

A SLAM estimator does not "find the trajectory." It computes a **probability distribution over trajectories** and then reports a *point* from that distribution (usually its mode) together with a *spread* (its covariance). Keeping these two things — the distribution and the point summary — conceptually separate is the single most clarifying idea in this chapter, because the two great families of solvers differ only in which they emphasize:

- **Batch / smoothing / factor-graph optimizers** (§08, the GTSAM iSAM2 back-end in Meridian's L3) keep an explicit handle on the joint distribution over *many* states and find its mode by nonlinear least squares.
- **Recursive / filtering estimators** (§09, the iterated-EKF front-end Meridian builds first as `IFrontEnd` v1, grounded in FAST-LIO's `esekfom.hpp`) propagate a Gaussian summary of *one* state forward in time and update it as data arrives.

The punchline of this whole course — proven by FAST-LIO and reused everywhere since — is that **these are the same computation** restricted to different windows of the same factor graph (§09.x). This section builds the shared substrate.

Let $X = \{x_0, x_1, \dots, x_N\}$ be the collection of states we want to estimate. In a sliding-window front-end the $x_i$ are poses (or full inertial states) at a handful of recent times; in the back-end they are keyframe states $x_i \in \mathcal{M}$. Let $Z = \{z_1, \dots, z_M\}$ be everything the sensors told us: IMU increments, LiDAR points matched to planes, image patches, GNSS fixes. The **posterior** is the conditional distribution of the states given the data,

$$
p(X \mid Z).
$$

Our job is to characterize this object. The **maximum a posteriori (MAP)** estimate is the trajectory that maximizes it:

$$
\boxed{\;X^\star = \underset{X \in \mathcal{M}}{\arg\max}\; p(X \mid Z).\;}
\tag{03.1}
$$

The rest of §03.2–03.4 turns (03.1) into a weighted nonlinear least-squares problem. §03.5–03.6 reorganize that problem as a graph. §03.7–03.8 develop the two algebraic forms (covariance vs. information) and the operation — marginalization via the Schur complement — that lets us *delete* variables from the graph while preserving the information they carried. That last operation is what makes both fixed-lag smoothing and the iterated EKF tractable, and it is the bridge to §08/§09.

---

### 03.2 From Bayes to a sum of squared residuals

#### 03.2.1 Factorizing the posterior

Apply Bayes' rule to (03.1):

$$
p(X \mid Z) = \frac{p(Z \mid X)\, p(X)}{p(Z)}.
\tag{03.2}
$$

The denominator $p(Z)$ (the *evidence*) does not depend on $X$, so it cannot move the maximizer and we drop it. Two modeling assumptions, both standard and both physically reasonable for our sensor suite, collapse the numerator into a product:

1. **Conditional independence of measurements.** Given the true states, distinct measurements are independent — a LiDAR return and an IMU sample do not influence each other except through the trajectory that produced both. Then the likelihood factorizes:
$$
p(Z \mid X) = \prod_{k=1}^{M} p(z_k \mid X_k),
\tag{03.3}
$$
where $X_k \subseteq X$ is the (usually small) subset of states that measurement $k$ actually touches. A LiDAR-plane residual touches one pose; an IMU preintegration factor (§04) touches two consecutive states; a loop closure touches two keyframes. This *locality* — each factor sees only a few variables — is the structural fact that the whole rest of the chapter exploits.

2. **A prior that itself factorizes.** The prior $p(X)$ encodes what we believe before looking at $Z$: an initial-state prior, the IMU process model linking $x_{i}$ to $x_{i+1}$, the smoothness of a B-spline (§10), or — in the recursive view — the entire summary of all past data, marginalized into a single Gaussian on the current window (§03.8, §09). Write it generically as a product of prior factors $\prod_j p_j(X_j)$.

Substituting (03.3) into (03.2) and folding the prior factors into the same product, the MAP problem becomes

$$
X^\star = \arg\max_{X} \prod_{k} \phi_k(X_k),
\tag{03.4}
$$

where each $\phi_k$ is either a likelihood term $p(z_k\mid X_k)$ or a prior term. Each $\phi_k$ is a **factor**: a nonnegative function of a small subset of variables. Equation (03.4) is *already* a factor graph (§03.5); we have just not drawn it yet.

#### 03.2.2 The negative-log transform: products become sums

Maximizing a product of many small numbers is numerically and analytically awkward. Take the negative logarithm — monotone, so it preserves the maximizer (now a minimizer):

$$
X^\star = \arg\min_{X}\; \sum_{k} \underbrace{\big(-\ln \phi_k(X_k)\big)}_{\displaystyle \text{cost of factor } k}.
\tag{03.5}
$$

This is the central reduction of the section: **MAP estimation is the minimization of a sum of per-factor costs.** Nothing so far is Gaussian or quadratic; (03.5) is general. The Gaussian assumption, next, is what turns each cost into a *squared* term and hands us least squares.

#### 03.2.3 The Gaussian assumption makes each cost quadratic

Model each measurement as its noiseless prediction plus zero-mean Gaussian noise. In the manifold setting of §02 the measurement model is written with $\boxplus$/$\boxminus$ so that the noise lives in a vector (tangent) space even when the prediction lives on $\mathcal{M}$:

$$
z_k = h_k(X_k) \,\boxplus\, \eta_k,
\qquad \eta_k \sim \mathcal{N}(0, \Sigma_k),
\tag{03.6}
$$

with $h_k$ the (nonlinear) measurement function — for a LiDAR plane, $h_k$ transforms a body-frame point into the map and evaluates $n^\top x + d$ (§05); for the IMU, $h_k$ is the preintegrated increment (§04). Define the **residual** as the $\boxminus$-difference between what we measured and what the current state predicts,

$$
\boxed{\; r_k(X_k) \;=\; z_k \,\boxminus\, h_k(X_k) \;\in\; \mathbb{R}^{d_k}. \;}
\tag{03.7}
$$

The residual is a *tangent-space* vector of dimension $d_k$ (the measurement's degrees of freedom), and *that* is where the Gaussian lives. A zero-mean Gaussian density in $r$ is

$$
p(z_k \mid X_k) \;\propto\; \exp\!\Big(-\tfrac{1}{2}\, r_k(X_k)^\top \Sigma_k^{-1}\, r_k(X_k)\Big),
\tag{03.8}
$$

so its negative log is, up to a constant that cannot move the minimizer,

$$
-\ln \phi_k(X_k) \;=\; \tfrac{1}{2}\, r_k^\top \Omega_k\, r_k \;+\; \text{const},
\qquad \Omega_k := \Sigma_k^{-1}.
\tag{03.9}
$$

Here $\Omega_k$ is the **information (precision) matrix** of measurement $k$. The quadratic form $r_k^\top \Omega_k r_k$ is the **squared Mahalanobis norm** $\lVert r_k \rVert_{\Sigma_k}^2$ — the squared length of the residual *in the metric defined by its own uncertainty*. A confident sensor (small $\Sigma$, large $\Omega$) makes its residual expensive to violate; a noisy sensor is cheap to disagree with. This is exactly how Meridian's robustness layer (§11) wants to inject per-axis observability: by *shaping* $\Omega_k$ so that well-observed directions are stiff and degenerate directions go slack.

#### 03.2.4 The result: weighted nonlinear least squares

Insert (03.9) into (03.5) and drop the constants:

$$
\boxed{\;
X^\star \;=\; \arg\min_{X \in \mathcal{M}}\; \frac{1}{2}\sum_{k=1}^{M}\; \big\lVert r_k(X_k) \big\rVert^2_{\Sigma_k}
\;=\; \arg\min_{X}\; \frac{1}{2}\sum_{k=1}^{M} r_k(X_k)^\top \Omega_k\, r_k(X_k).
\;}
\tag{03.10}
$$

This is the equation the entire estimator computes. Three observations frame the rest of the chapter:

- It is **nonlinear** (the $h_k$, and the manifold $\boxminus$, are nonlinear) and it is **least squares** (each term is a squared Mahalanobis residual). Hence: *MAP = NLS under Gaussian assumptions.* §08 attacks (03.10) directly with Gauss–Newton / Levenberg–Marquardt; §09 shows the iterated EKF solving the same (03.10) over a sliding window.
- It is **separable into a sum over factors**, each touching few variables. That separability is the factor graph (§03.5) and the source of the sparsity (§03.6) that makes large problems solvable.
- The **weights $\Omega_k$ are the modeling knobs.** Choosing them well — and re-weighting them online for robustness (§11) — is where domain knowledge enters. The squared form is exactly what fails catastrophically under outliers, which is why §11 replaces $\lVert r_k\rVert^2$ with a robust kernel $\rho(\lVert r_k\rVert)$, an iteratively-reweighted variant of the *same* (03.10).

> **Meridian grounding.** In FAST-LIO this entire structure is realized concretely. The state $x$ — `state_ikfom` in `FAST_LIO/include/use-ikfom.hpp:12-21` — packs `pos`, `rot`, `offset_R_L_I`, `offset_T_L_I`, `vel`, `bg`, `ba`, `grav` on the product manifold $\mathbb{R}^3\times SO(3)\times SO(3)\times\mathbb{R}^3\times\mathbb{R}^3\times\mathbb{R}^3\times\mathbb{R}^3\times S^2$ (note gravity is carried on the 2-sphere $S^2$, a 2-DoF manifold, not a free $\mathbb{R}^3$ — exactly the $\mathcal{M}$ of §02; this is why the toolkit threads special `S2_Mx`/`S2_Nx_yy` projection maps through every update). The FAST-LIO paper writes the per-point residual $z_j = \mathbf{u}_j^\top({}^G\mathbf{T}_{I_k}\,{}^I\mathbf{T}_L\,{}^{L}\mathbf{p}_{f_j} - {}^G\mathbf{q}_j)$ (paper 2010.08196 eq (12)) — our $r_k$ — and assembles them with their Jacobians in the `h_share_model` callback (in `FAST_LIO/src/laserMapping.cpp`), feeding the stacked residual $z_k = [z_1^\top,\dots,z_m^\top]^\top$ and Jacobian $H = [H_1^\top,\dots,H_m^\top]^\top$ (paper eq (14)) to the iterated update in `FAST_LIO/include/IKFoM_toolkit/esekfom/esekfom.hpp`. The covariance/information bookkeeping ($\Sigma \leftrightarrow$ the code's `P_`, gain $K \leftrightarrow$ `K_`) lives in that same `esekfom.hpp` update loop (the `update_iterated_dyn_share()` method, lines ~1001-1110). We dissect that loop as the recursive solution of (03.10) in §09; here it is enough to recognize that the filter *minimizes* (03.10) — the paper itself states this, deriving its update from the explicit MAP cost in eq (17) (see §03.8.3 below), not doing something philosophically different.

---

### 03.3 Worked micro-example: two factors on one pose

To make (03.10) concrete before generalizing, take the smallest non-trivial case: estimate a single 2D position $x = p \in \mathbb{R}^2$ from (i) a prior $\hat p_0$ with covariance $\Sigma_0 = \sigma_0^2 I$ and (ii) one range-bearing-like linear measurement $z = A p + \eta$, $\eta\sim\mathcal{N}(0,\Sigma_z)$. Two factors, one variable.

The cost is

$$
J(p) = \tfrac12 (p - \hat p_0)^\top \Omega_0 (p - \hat p_0) \;+\; \tfrac12 (z - A p)^\top \Omega_z (z - A p),
\qquad \Omega_0=\Sigma_0^{-1},\ \Omega_z=\Sigma_z^{-1}.
$$

Both residuals are *linear*, so the cost is exactly quadratic and the minimizer is found by setting the gradient to zero:

$$
\nabla J = \Omega_0(p - \hat p_0) - A^\top \Omega_z (z - A p) = 0
\;\;\Longrightarrow\;\;
\underbrace{(\Omega_0 + A^\top \Omega_z A)}_{\displaystyle \Lambda\ (\text{information})}\, p^\star = \underbrace{\Omega_0 \hat p_0 + A^\top \Omega_z z}_{\displaystyle \xi\ (\text{information vector})}.
\tag{03.11}
$$

Read off three lessons that recur at full scale:

1. **Information adds.** The posterior information $\Lambda = \Omega_0 + A^\top \Omega_z A$ is the prior information *plus* the measurement information mapped into state space by $A^\top \Omega_z A$. Stacking factors = summing their information contributions. This is the additive structure (§03.7) that the Kalman update (§09) and the graph's $H^\top \Omega H$ (§08) both inherit.
2. **The point estimate is a linear system.** $p^\star = \Lambda^{-1}\xi$. At full scale $\Lambda$ is large but *sparse* (§03.6), and solving $\Lambda\,\delta = \xi$ is the inner step of every solver in §08–§09.
3. **The covariance is the inverse information.** $\Sigma^\star = \Lambda^{-1}$. The estimator's *confidence* falls straight out of the same matrix that produced the estimate — which is why Meridian can publish a confidence overlay (L6) for free from the back-end, and why per-axis observability (§11) is literally a spectral statement about $\Lambda$.

Nonlinear factors (the real case) reduce to (03.11) *per iteration* after linearization — that is precisely Gauss–Newton (§08.2). The linear micro-example is the iteration step in disguise.

---

### 03.4 Linearization on the manifold: how the squared cost becomes a normal equation

Real $h_k$ are nonlinear and $X$ lives on $\mathcal{M}$, so we cannot solve (03.10) in closed form. We linearize *in the tangent space* (§02) around the current estimate $\hat X$, write the update as a tangent vector $\delta x$ applied with $\boxplus$, and iterate.

Parameterize $X = \hat X \boxplus \delta x$. The residual (03.7), seen as a function of $\delta x$, has the first-order expansion

$$
r_k(\hat X \boxplus \delta x) \;\approx\; r_k(\hat X) \;+\; H_k\, \delta x_k,
\qquad
H_k \;:=\; \left.\frac{\partial\, r_k(\hat X \boxplus \delta x)}{\partial\, \delta x}\right|_{\delta x = 0}.
\tag{03.12}
$$

$H_k$ is the **Jacobian of the residual with respect to the tangent-space increment** — the manifold version of the ordinary Jacobian, and the quantity §02 taught us to compute with the right $\boxplus$-Jacobians. (For $SO(3)$ this is where the right-Jacobian $J_r$ and the $\mathrm{Exp}/\mathrm{Log}$ derivatives enter; §04–§06 derive the concrete $H_k$ for IMU, LiDAR, and photometric factors.) Substituting (03.12) into (03.10) gives a quadratic in $\delta x$:

$$
\tfrac12\sum_k \lVert r_k(\hat X) + H_k\,\delta x_k\rVert^2_{\Sigma_k}.
$$

Stack all residuals into $r$, all Jacobians into a tall block matrix $H$ (one block-row per factor, zeros where a factor does not touch a variable), and assemble the block-diagonal weight $\Omega = \mathrm{blockdiag}(\Omega_k)$. Setting the gradient to zero yields the **normal equations**:

$$
\boxed{\; \big(\,H^\top \Omega\, H\,\big)\, \delta x^\star \;=\; -\,H^\top \Omega\, r. \;}
\tag{03.13}
$$

Define $\Lambda := H^\top \Omega H$ (the **Hessian approximation** / system information matrix) and $g := H^\top \Omega r$ (the gradient). Then $\delta x^\star = -\Lambda^{-1} g$, and we update on the manifold,

$$
\hat X \;\leftarrow\; \hat X \,\boxplus\, \delta x^\star,
\tag{03.14}
$$

re-linearize, and repeat until $\delta x^\star$ is negligible. That loop is **Gauss–Newton**; damping $\Lambda \to \Lambda + \lambda\,\mathrm{diag}(\Lambda)$ gives **Levenberg–Marquardt** (full treatment in §08). The crucial structural point for *this* section is that (03.13) has the *exact same algebra* as the micro-example (03.11): information $\Lambda = H^\top\Omega H$ is a sum of per-factor rank-$d_k$ contributions $H_k^\top \Omega_k H_k$, and the right-hand side is the corresponding sum. The graph's sparsity (§03.6) is therefore the sparsity of $\Lambda$.

> **Meridian grounding.** This is literally what `esekfom.hpp` computes each iteration: the `h_share_model` callback (in `laserMapping.cpp`) returns $r_k$ and $H_k$ for the active LiDAR-plane correspondences; the toolkit assembles $H^\top \Omega H$ together with the propagated prior information and solves for the tangent increment, then applies it via the state's $\boxplus$ (the `boxplus`/`oplus` operator defined alongside `state_ikfom` in `use-ikfom.hpp`). FAST-LIO's contribution is *not* a different objective — it is a clever, equivalent *factorization* of solving (03.13) when the measurement dimension is huge (thousands of LiDAR points) but the state is small (a few dozen DoF); see §09 for the equivalence proof.

---

### 03.5 Factor graphs: drawing the objective

Equation (03.4) — a product of factors, each over a small variable subset — is naturally a **bipartite graph**. A factor graph $\mathcal{G} = (\mathcal{V}, \mathcal{F}, \mathcal{E})$ has:

- **Variable nodes** $\mathcal{V}$, one per state $x_i$ (drawn as circles);
- **Factor nodes** $\mathcal{F}$, one per term $\phi_k = \exp(-\tfrac12\lVert r_k\rVert^2_{\Sigma_k})$ in (03.10) (drawn as squares);
- **Edges** $\mathcal{E}$ connecting factor $k$ to exactly the variables $X_k$ in its residual.

"Bipartite" means edges only ever run variable–factor, never variable–variable or factor–factor. The graph *is* the objective: the cost (03.10) is recovered by reading off, for every square, the residual of the variables it touches, and summing. A small LiDAR-IMU window in Meridian's front-end looks like:

```
   (prior)                IMU factor          IMU factor
      |                    /      \            /      \
   [ x0 ] --- [LiDAR] --[ x1 ]--[LiDAR]--[ x2 ]--[LiDAR]-- ...
      \________________________________/
                  (GNSS factor, absolute, touches x0)

  circles () = states (poses / inertial states)   ← variables
  squares [] = factors (residuals)                ← measurements & priors
```

Reading the graph:

- A **unary** factor touches one variable: the initial prior, a GNSS fix (§07), a single LiDAR-plane residual (§05). $H_k$ has one nonzero block-column.
- A **binary** factor touches two: an IMU preintegration factor between consecutive states (§04), a loop closure between two keyframes (§05/§11). $H_k$ has two nonzero block-columns.
- The graph makes the locality structural: a factor's residual gradient $H_k^\top \Omega_k r_k$ is nonzero *only* in the columns for $X_k$. Summing these into $\Lambda = H^\top\Omega H$ therefore produces a matrix whose nonzero pattern *is the adjacency of the graph* (§03.6).

This is why Meridian's L3 back-end is built on GTSAM/iSAM2: GTSAM *is* a factor-graph library. Keyframes are variable nodes; IMU, LiDAR-odometry, loop-closure, GNSS, and online-extrinsic-refinement constraints are factor nodes; iSAM2 incrementally re-solves (03.10) as factors arrive. The front-end's sliding window is the *same* graph truncated to recent variables — and the act of truncating it correctly is marginalization (§03.8).

> **A note on the two layers being one graph.** It is tempting to think of Meridian's continuous-time/iEKF front-end (L2) and the iSAM2 back-end (L3) as different mathematics. They are not. They are two windows onto a single, ever-growing factor graph: the front-end keeps a short tail with tight real-time deadlines and marginalizes the rest into a prior; the back-end keeps keyframe summaries over the whole mission and optimizes for global consistency. Section 12 makes this explicit. Holding "it is all one graph (03.10)" in mind prevents a great deal of confusion later.

---

### 03.6 Sparsity: why locality is the whole game

The reason 3D SLAM is tractable at all is that $\Lambda = H^\top \Omega H$ is **sparse**, and its sparsity is dictated by the graph. Two variables $x_i, x_j$ produce a nonzero off-diagonal block $\Lambda_{ij}$ **iff some factor touches both of them** — i.e., iff they share a factor node. Concretely:

- $\Lambda_{ii} = \sum_{k:\, i \in X_k} H_{k,i}^\top \Omega_k H_{k,i}$ — variable $i$'s diagonal block is the sum of information from every factor touching it.
- $\Lambda_{ij} = \sum_{k:\, i,j \in X_k} H_{k,i}^\top \Omega_k H_{k,j}$ — the off-diagonal exists only for variables co-observed by a common factor.

Because each factor in our system touches one or two variables, $\Lambda$ is **block-tridiagonal-plus-loops**: a band from the chain of IMU/odometry factors, plus a few off-band blocks from loop closures and GNSS. This structure is what every solver in §08–§09 exploits:

- **Batch (§08):** a sparse Cholesky factorization $\Lambda = LL^\top$ costs orders of magnitude less than dense $O(n^3)$, and a good variable ordering keeps the *fill-in* (new nonzeros created during factorization) small.
- **Schur complement (§03.8, §08.x):** when many variables are "leaves" (e.g., per-point or per-landmark variables touched by few factors), eliminating them first via the Schur complement reduces the problem to a small dense system on the remaining variables. FAST-LIO's "new Kalman gain" trick is exactly a Schur-complement choice that inverts in *measurement* space or *state* space depending on which is smaller (§09).
- **Incremental (§03.8, §09):** the band structure means a new measurement at the tail only perturbs a few blocks, so we can update the factorization instead of refactoring — the principle behind iSAM2 and fixed-lag smoothing.

The single sentence to remember: **the sparsity of the linear algebra is the bipartite structure of the factor graph.** Lose locality and you lose tractability.

---

### 03.7 Two algebraic forms: covariance vs. information

The same Gaussian belief can be written two equivalent ways, and SLAM constantly converts between them because each makes a different operation cheap. For a Gaussian on $\delta x$ (the tangent-space error of §02):

| | **Covariance (moment) form** | **Information (canonical) form** |
|---|---|---|
| Parameters | mean $\mu$, covariance $\Sigma$ | info vector $\eta = \Omega\mu$, info matrix $\Omega = \Sigma^{-1}$ |
| Density | $\propto \exp\!\big(-\tfrac12 (\delta x - \mu)^\top \Sigma^{-1}(\delta x - \mu)\big)$ | $\propto \exp\!\big(-\tfrac12 \delta x^\top \Omega\, \delta x + \eta^\top \delta x\big)$ |
| Cheap operation | **marginalization** (drop variables: just delete rows/cols of $\Sigma$) | **conditioning / adding measurements** (just *add* $H^\top\Omega_z H$ to $\Omega$) |
| Expensive operation | conditioning (needs the Kalman update / a matrix inverse) | marginalization (needs a Schur complement, §03.8) |
| Used by | the EKF *covariance* propagation (§09) | the factor graph / least-squares normal equations (§08) |

The duality is exact: $\Omega = \Sigma^{-1}$, $\eta = \Omega\mu$, $\mu = \Omega^{-1}\eta$. The two facts to internalize:

- **Adding independent information is addition in the information form.** Fusing measurement $z$ into a prior is $\Omega^+ = \Omega^- + H^\top\Omega_z H$ and $\eta^+ = \eta^- + H^\top \Omega_z (z\text{-prediction terms})$ — exactly the micro-example (03.11) and the normal equations (03.13). This is why factor graphs *are* the information form: each factor literally adds its $H_k^\top\Omega_k H_k$ block. Adding a factor is cheap; the information matrix stays sparse.
- **Dropping a variable is easy in covariance form, hard in information form.** To marginalize out a variable from a *covariance*, you delete its rows and columns. To marginalize it out of an *information* matrix, you cannot just delete — you must pay a Schur complement (§03.8), and doing so *fills in* the graph, creating new edges among the variable's former neighbors.

The estimator therefore lives in tension: it *wants* the information form (sparse, additive, perfect for fusing the flood of LiDAR/IMU/visual factors), but the operation that keeps the problem bounded in real time — throwing old states away — is the one the information form charges for. Resolving that tension is marginalization, and it is the bridge to §08 (batch with Schur) and §09 (recursive filtering as marginalize-everything-but-the-present).

---

### 03.8 Marginalization and the Schur complement: deleting variables without losing their information

A real-time estimator cannot keep every past state. The front-end window must stay small; the back-end must summarize old keyframes. The principled way to remove a variable is **marginalization**: integrate it out of the joint, leaving its influence baked into a prior over the variables that remain. Done correctly, this is *lossless* up to the linearization point — it is *not* the same as merely deleting the variable and its factors (which would discard information and corrupt the estimate).

#### 03.8.1 The Gaussian marginal

Partition the variables into a block $a$ we want to **keep** and a block $b$ we want to **marginalize out**. Write the joint information form (the linearized system, §03.4):

$$
\Lambda = \begin{bmatrix} \Lambda_{aa} & \Lambda_{ab} \\ \Lambda_{ba} & \Lambda_{bb} \end{bmatrix},
\qquad
\eta = \begin{bmatrix} \eta_a \\ \eta_b \end{bmatrix}.
$$

A fundamental Gaussian identity says: the marginal over the kept block $a$, $p(a) = \int p(a,b)\,\mathrm{d}b$, is again Gaussian, with information

$$
\boxed{\;
\Lambda_a^{\text{marg}} \;=\; \Lambda_{aa} \;-\; \Lambda_{ab}\,\Lambda_{bb}^{-1}\,\Lambda_{ba},
\qquad
\eta_a^{\text{marg}} \;=\; \eta_a \;-\; \Lambda_{ab}\,\Lambda_{bb}^{-1}\,\eta_b.
\;}
\tag{03.15}
$$

The matrix $\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba}$ is the **Schur complement** of $\Lambda_{bb}$ in $\Lambda$. It is the algebraic engine behind nearly everything in §08–§09. Read (03.15) physically: the term $-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba}$ is the information that flowed *through* the eliminated variable $b$ and is now redistributed among $b$'s neighbors as a new, dense block. This is **fill-in**: marginalizing a variable connects all of its former neighbors to each other (they now share the marginal prior factor). A long-baseline loop-closure variable, marginalized, can therefore turn a sparse band into something denser — the practical reason fixed-lag windows and careful keyframe selection matter.

#### 03.8.2 Why the Schur complement is also how we *solve* (and the FAST-LIO connection)

The very same Schur complement appears when **solving** the normal equations (03.13), not just when marginalizing. To solve $\Lambda \begin{bsmallmatrix}\delta a\\\delta b\end{bsmallmatrix} = -g$ you can eliminate $\delta b$ first:

$$
(\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\,\delta a = -(g_a - \Lambda_{ab}\Lambda_{bb}^{-1} g_b),
\qquad
\delta b = \Lambda_{bb}^{-1}(-g_b - \Lambda_{ba}\,\delta a).
\tag{03.16}
$$

If $b$ is high-dimensional but block-diagonal (e.g. per-point or per-landmark variables, each touched by few factors), $\Lambda_{bb}^{-1}$ is cheap and the reduced system on $a$ is small. This is the bundle-adjustment Schur trick and, in spirit, **FAST-LIO's celebrated Kalman-gain reformulation**: when the measurement vector is enormous (thousands of LiDAR points, so the innovation covariance $H\Sigma H^\top + R$ is huge) but the state is small, FAST-LIO chooses the algebraically *equivalent* form of the update that requires inverting a matrix of *state* dimension rather than *measurement* dimension. The paper (2010.08196, FAST-LIO) states and *proves* this equivalence: its conventional gain $K = P H^\top (H P H^\top + R)^{-1}$ (eq (18), with $P$ the propagated prior covariance) is replaced by the new form $K = (H^\top R^{-1} H + P^{-1})^{-1} H^\top R^{-1}$ (eq (20)), proven equal in Appendix B "based on the matrix inverse lemma." Their motivation is precisely the one we are developing: the cost (eq (17)) "is over the state, hence the solution should be calculated with complexity depending on the state dimension." The identity is the matrix-inversion-lemma / Schur-complement statement:

$$
\underbrace{\Sigma H^\top (H \Sigma H^\top + R)^{-1}}_{\text{invert in measurement space }(d_z\times d_z)}
\;=\;
\underbrace{(H^\top R^{-1} H + \Sigma^{-1})^{-1} H^\top R^{-1}}_{\text{invert in state space }(d_x\times d_x)}.
\tag{03.17}
$$

Both sides are the Kalman gain $K$; the left is "covariance/innovation" thinking, the right is "information/normal-equation" thinking. For LiDAR-inertial odometry $d_x \ll d_z$, so the right side is dramatically cheaper — that is the order-of-magnitude saving FAST-LIO advertises (the paper's Table II measures the old formula at 1621 ms vs. the new at 1.16 ms for 1802 feature points). We derive (03.17) carefully and connect it to the iterated update in §09; here the point is that the *same Schur/Woodbury identity* underlies marginalization (03.15), batch elimination (03.16), and the recursive gain (03.17). It is one piece of algebra wearing three hats.

> **Meridian grounding.** Both forms are literally coded side-by-side in `FAST_LIO/include/IKFoM_toolkit/esekfom/esekfom.hpp`, and the toolkit *picks the cheaper one at runtime* by comparing the state dimension `n` to the measurement dimension `dof_Measurement`: when `n > dof_Measurement` it uses the measurement-space (innovation) form `K_ = P_ * h_x_.transpose() * (h_x_ * P_ * h_x_.transpose() + h_v_ * R * h_v_.transpose()).inverse()`, and otherwise — the LiDAR case, thousands of points so `n < dof_Measurement` — it uses the state-space form `K_ = (h_x_.transpose() * R_in * h_x_ + P_.inverse()).inverse() * h_x_.transpose() * R_in`, building the state-dimensioned $H^\top R^{-1} H + P^{-1}$ exactly as in paper eq (20) (`esekfom.hpp`, the `if(n > dof_Measurement)…else…` branch around lines 1081-1107). That is why FAST-LIO stays real-time even with thousands of point-to-plane residuals per scan. Point-LIO (`Point-LIO/src/Estimator.cpp`) and FAST-LIVO2 (`FAST-LIVO2/src/`) reuse the same `esekfom`-style machinery; FAST-LIVO2 in fact goes further and applies the LiDAR and camera updates *sequentially* (its paper 2408.14035 eqs (5)-(8) factor the joint posterior $p(x\mid y_l,y_c)$ into two Bayesian updates — first fuse LiDAR to get $p(x\mid y_l)$, then fuse the camera), which is just the §03.2 product-of-factors statement applied one factor-block at a time: stacking the *photometric* residual's information $H_{\text{photo}}^\top\Omega_{\text{photo}}H_{\text{photo}}$ onto $\Lambda$ (§03.7). Tightly-coupled fusion, viewed through this section, is nothing more than *summing information blocks into one $\Lambda$ before inverting once* (or, equivalently, conditioning on them one after another).

#### 03.8.3 The recursive view: filtering = marginalize all but the present

Marginalization closes the loop with the recursive estimators of §09. A Kalman/EKF filter keeps only the *current* state and a Gaussian summary $(\mu, \Sigma)$ of it. Where did the past go? It was **marginalized**: at each step the filter integrates out the previous state, leaving its information in the prior on the current state. Equation (03.15) applied repeatedly *is* the prediction step; the measurement update is the information addition of §03.7. Thus a filter is a factor graph in which every variable except the newest has already been marginalized away — and **fixed-lag smoothing** is the in-between, marginalizing only states that fall off the tail of a sliding window of length $L$.

This gives the clean hierarchy Meridian's architecture rests on, all instances of minimizing the *same* (03.10):

- **Full batch / iSAM2 back-end (L3):** marginalize nothing (or lazily); re-optimize the whole keyframe graph for global consistency. Best accuracy, used for loop closure and drift correction.
- **Fixed-lag front-end (L2):** keep a window of recent states, marginalize older ones into a prior. Bounded cost, real-time. Meridian's CT B-spline window (§10) and the iEKF v1 both live here.
- **Filter (degenerate fixed-lag, $L=1$):** marginalize everything but the present. The leanest real-time form; FAST-LIO's `esekfom` update is exactly this, with the iteration of §03.4 layered on top (the "iterated" in iterated-EKF) so that a *single* recursive step still solves the *nonlinear* least-squares (03.10) to convergence — the formal MAP $\leftrightarrow$ IKF correspondence the FAST-LIO paper (2010.08196) is built on. The paper makes the identification explicit: combining the IMU-propagation prior (its eq (15), $x \boxminus \hat x \sim \mathcal{N}(0, \hat P)$) with the linearized LiDAR likelihood (eq (14)) "yields the maximum a-posteriori estimate (MAP)" $\min_{\delta x}\big(\lVert x \boxminus \hat x\rVert^2_{\hat P^{-1}} + \sum_{j=1}^m \lVert z_j + H_j\,\delta x\rVert^2_{R_j^{-1}}\big)$ (eq (17)) — which is our (03.10) with one prior factor (the marginalized past) plus $m$ LiDAR factors — and then "optimizing the resultant quadratic cost leads to the standard iterated Kalman filter." That sentence is the entire thesis of this section restated by the source: the filter does not replace MAP/NLS; it *is* MAP/NLS over a one-state window.

---

### 03.9 Synthesis of this section, and forward pointers

We started from the posterior $p(X\mid Z)$ and, with two assumptions — conditional independence of measurements and Gaussian noise on the manifold (§02) — derived that the MAP estimate is the solution of a **weighted nonlinear least-squares** problem,

$$
X^\star = \arg\min_X \tfrac12 \sum_k \lVert z_k \boxminus h_k(X_k)\rVert^2_{\Sigma_k}.
$$

We saw that this objective is a **factor graph** (bipartite: variables × factors), that its locality makes the system information matrix $\Lambda = H^\top\Omega H$ **sparse** with the graph's adjacency, that the same belief admits **covariance** and **information** forms with complementary strengths, and that **marginalization via the Schur complement** is the one operation that lets us delete variables losslessly — the operation that, worn as three hats, is *batch elimination*, the *FAST-LIO Kalman-gain trick*, and the *prediction step of a filter*. Each technical claim is realized in the on-disk reference systems: the state and $\boxplus$ in `use-ikfom.hpp`, the residual/Jacobian assembly in `laserMapping.cpp`'s `h_share_model`, and the information-form iterated update in `esekfom.hpp`, with Point-LIO and FAST-LIVO2 reusing the same machinery and merely stacking more factor blocks.

What this section deliberately *did not* do, and where to find it:

- **The actual residuals $r_k$ and Jacobians $H_k$** for each sensor — IMU/preintegration (§04), LiDAR point-to-plane/line (§05), sparse-direct photometric with LiDAR-provided depth (§06), GNSS/ENU absolute constraints (§07). This section gave only the generic $r_k = z_k \boxminus h_k(X_k)$ and $H_k = \partial r_k/\partial\delta x$.
- **How (03.13) is actually solved.** Gauss–Newton, Levenberg–Marquardt, ordering, and sparse Cholesky in §08; the iterated EKF / ESIKF and its formal equivalence to GN-MAP, plus fixed-lag smoothing, in §09.
- **Continuous time.** Replacing the discrete states $x_i$ with B-spline control points $c_k$ so that one residual can be evaluated at *any* timestamp — the same MAP=NLS skeleton, with $T(t)$ in place of $x_i$ — in §10.
- **When the Gaussian/least-squares model breaks.** Degeneracy and observability (the spectrum of $\Lambda$), robust kernels (replacing $\lVert r\rVert^2$ by $\rho(\lVert r\rVert)$), GNC, switchable constraints, and PCM, all of which re-shape the weights $\Omega_k$ of (03.10) — in §11.
- **The full assembled step**, mapped line-by-line onto FAST-LIO2 / FAST-LIVO2, in §12.

Keep (03.10) in view through all of them. Every later section is a way of writing down, weighting, linearizing, solving, or robustifying that one sum of squared Mahalanobis residuals.


---


## 04. IMU: model, propagation, preintegration, inertial residual

> **Where this sits.** Section [03](03_probability.md) established that estimation is *maximum a posteriori* (MAP) inference over a factor graph, equivalent to nonlinear least squares on the manifold defined in Section [02](02_manifolds.md). The IMU is the *connective tissue* of every tightly-coupled estimator: it is the only sensor fast enough (100–500 Hz) and dense enough in time to predict the state *between* the sparse, latency-laden LiDAR scans (Section [05](05_lidar.md)) and camera frames (Section [06](06_visual.md)). This section develops the inertial measurement model, how we bootstrap it (gravity + bias initialization), and the two mathematically distinct ways it enters the estimator:
>
> 1. **Forward/backward propagation** (the *filtering* view, used by FAST-LIO2 / FAST-LIVO2 / Point-LIO and by Meridian's v1 iEKF front-end): the IMU drives a deterministic state transition $F_x$ and injects process noise $F_w Q F_w^\top$; backward propagation then *de-skews* the LiDAR sweep.
> 2. **Preintegration** (the *smoothing* view, Forster et al., used by GTSAM iSAM2 in Meridian's L3 back-end of Sections [03](03_probability.md) and [09](09_recursive.md)): a single relative-motion factor summarizing many IMU samples between two keyframes.
>
> These are two faces of the same physics; Section [09](09_recursive.md) proves the iterated-EKF/MAP equivalence, and Section [12](12_synthesis.md) assembles the whole step. Here we derive both, with full Jacobians, grounded line-by-line in the FAST-LIO reference code.

---

### 04.1 Why the IMU is special

A 6-axis IMU returns, at each tick $k$ with interval $\Delta t$, two 3-vectors: the **angular rate** $\boldsymbol{\omega}_m$ from the gyroscope and the **specific force** $\mathbf{a}_m$ from the accelerometer. Unlike LiDAR or camera, the IMU does **not** measure the state $x$ or any geometric residual against the map. It measures the *time-derivatives* of the state's motion — rotation rate and proper acceleration — corrupted by slowly-drifting biases and white noise. Consequently the IMU is used in one of two structurally different roles:

- **As a propagation model (kinematic prior).** We *integrate* the IMU to predict $x_{k+1}$ from $x_k$. The IMU is then treated not as a measurement with a residual but as the *input* $u$ to the discrete-time motion model $x_{k+1} = f(x_k, u_k, w_k)$. This is the path FAST-LIO2 takes; the residual that the filter actually minimizes is the *LiDAR* point-to-plane residual (Section [05](05_lidar.md)), with the IMU only shaping the prior covariance. We treat the gyro/accel white noise as the process noise $w$.
- **As a measurement with a residual (preintegration / Point-LIO).** Here the IMU samples produce an actual residual $r$ that is fed into the least-squares cost. In Forster-style preintegration the residual compares the *integrated* relative motion to the *predicted* relative motion across two keyframes. In Point-LIO it goes further still: the body acceleration and angular velocity are promoted to *state variables*, and each raw IMU reading becomes a direct measurement of those states.

Meridian uses **both**: the L2 front-end (iEKF v1, CT-spline v2) uses propagation; the L3 GTSAM back-end uses preintegration factors between keyframes. Understanding the relationship is the goal of this section.

---

### 04.2 The state and the notation recap

We use the Section [02](02_manifolds.md) notation throughout. The estimator state lives on the manifold

$$
x = \big(\, R,\; p,\; v,\; b_g,\; b_a,\; g,\; R_{LI},\; p_{LI} \,\big),
$$

with $R\in SO(3)$ the body (IMU) attitude in the world/global frame $G$, $p\in\mathbb{R}^3$ the position, $v\in\mathbb{R}^3$ the velocity, $b_g,b_a\in\mathbb{R}^3$ the gyro and accelerometer biases, $g\in\mathbb{R}^3$ the gravity vector in $G$, and $(R_{LI},p_{LI})$ the **LiDAR→IMU extrinsic** that Meridian will refine online (Section [11](11_robustness.md), cross-cutting calibration). The boxplus/boxminus operators $\boxplus,\boxminus$ and the maps $\mathrm{Exp}/\mathrm{Log}$ are exactly as in Section [02](02_manifolds.md). This $\boxplus$/$\boxminus$ machinery is the encapsulation of Hertzberg et al. that FAST-LIO uses directly: for the compound $SO(3)\times\mathbb{R}^n$ manifold, $\begin{psmallmatrix}R\\a\end{psmallmatrix}\boxplus\begin{psmallmatrix}r\\b\end{psmallmatrix}=\begin{psmallmatrix}R\,\mathrm{Exp}(r)\\a+b\end{psmallmatrix}$ and $\begin{psmallmatrix}R_1\\a\end{psmallmatrix}\boxminus\begin{psmallmatrix}R_2\\b\end{psmallmatrix}=\begin{psmallmatrix}\mathrm{Log}(R_2^\top R_1)\\a-b\end{psmallmatrix}$ (FAST-LIO §III-B.1, paper line 131–133).

This is precisely the FAST-LIO2 state. In the reference code it is the `state_ikfom` struct, an `MTK` compound manifold:

```cpp
// FAST_LIO/include/use-ikfom.hpp:12-21
MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos))            // p
((SO3,  rot))             // R
((SO3,  offset_R_L_I))    // R_{LI}
((vect3, offset_T_L_I))   // p_{LI}
((vect3, vel))            // v
((vect3, bg))             // b_g
((vect3, ba))             // b_a
((S2,   grav))            // gravity on the 2-sphere S^2
);
```

Two design choices here matter for the math below:

- **Gravity on $S^2$.** FAST-LIO2 parameterizes $g$ as a point on the 2-sphere `S2` of *fixed magnitude* (`typedef MTK::S2<double, 98090, 10000, 1> S2;`, `use-ikfom.hpp:8` — i.e. $|g| = 9.8090$ m/s²), not as a free $\mathbb{R}^3$ vector. Its error is therefore only 2-dimensional (a rotation of the gravity direction), removing the unobservable magnitude. This `S2 grav` member changes the gravity block of the Jacobians (see §04.5). FAST-LIO v1 (`common_lib.h::StatesGroup`, line 152) instead keeps gravity as a full $\mathbb{R}^3$ `gravity` field, giving a 18-D `DIM_STATE` (`common_lib.h:21`). Meridian's iEKF should adopt the $S^2$ treatment.
- **Extrinsics in the state.** `offset_R_L_I` / `offset_T_L_I` make the LiDAR–IMU transform a graph/filter variable, supporting the online extrinsic refinement Meridian requires.

The FAST-LIO2 error-state dimension is `state_ikfom::DOF = 23` (each of $p,R,R_{LI},p_{LI},v,b_g,b_a$ contributes 3 → 21, gravity on $S^2$ contributes 2 → 23), while the *ambient* dimension `DIM = 24` because $g$ occupies 3 coordinates but only 2 tangent DOF. This 24-vs-23 split is exactly why `get_f` returns a `Matrix<double,24,1>` while `df_dx` returns `Matrix<double,24,23>` (`use-ikfom.hpp:47,61`). We keep the abstract dimension symbol $n=\mathrm{DOF}$.

---

### 04.3 The IMU measurement model

The IMU is rigidly mounted; let the body frame be the IMU frame. The true angular velocity $\boldsymbol{\omega}$ and true specific force are corrupted by bias and noise. The standard model (matching FAST-LIO paper eq. (1), line 182–186, and Forster et al. eq. (1)) is

$$
\boxed{\;
\boldsymbol{\omega}_m \;=\; \boldsymbol{\omega} \;+\; b_g \;+\; n_g,
\qquad
\mathbf{a}_m \;=\; R^\top\,(\mathbf{a} - g) \;+\; b_a \;+\; n_a \;}
\tag{04.1}
$$

where:

- $\boldsymbol{\omega}$ is the true body-frame angular rate, $\mathbf{a}$ the true world-frame acceleration of the IMU, $g$ the world-frame gravity;
- $b_g, b_a$ are the slowly-varying biases;
- $n_g, n_a$ are **additive white Gaussian noise** with $\mathbb{E}[n_g(t)\,n_g(\tau)^\top]=\sigma_g^2 I\,\delta(t-\tau)$ etc.

The crucial subtlety in the accelerometer equation is the term $R^\top(\mathbf{a}-g)$. The accelerometer measures **proper acceleration** (specific force): the non-gravitational acceleration expressed in the *body* frame. A device in free fall reads zero; a device at rest reads $-R^\top g$ (it "feels" the normal force holding it up). This is why a stationary IMU lets us recover gravity (§04.4). Equivalently, solving (04.1) for the world-frame acceleration gives the form used in the kinematics (04.4): $\mathbf{a} = R(\mathbf{a}_m - b_a - n_a) + g$.

The biases follow a **random walk** (Brownian motion):

$$
\dot b_g = n_{bg}, \qquad \dot b_a = n_{ba},
\tag{04.2}
$$

with $n_{bg},n_{ba}$ the bias random-walk driving noises (FAST-LIO eq. (1): $\dot b_\omega=n_{b\omega},\ \dot b_a=n_{ba}$). This is the standard "Gaussian + random-walk" IMU model. The four noise densities form the diagonal of the $12\times12$ process-noise covariance $Q$. In FAST-LIO2 they are assembled per-interval inside the de-skew/propagation loop:

```cpp
// FAST_LIO/src/IMU_Processing.hpp:280-283  (inside UndistortPcl, before kf_state.predict)
Q.block<3, 3>(0, 0).diagonal() = cov_gyr;        // n_g
Q.block<3, 3>(3, 3).diagonal() = cov_acc;        // n_a
Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;   // n_{bg}
Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;   // n_{ba}
```

with default densities set in the constructor:

```cpp
// FAST_LIO/src/IMU_Processing.hpp:92-95
cov_acc       = V3D(0.1, 0.1, 0.1);
cov_gyr       = V3D(0.1, 0.1, 0.1);
cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
```

The ordering $w=[\,n_g^\top,\;n_a^\top,\;n_{bg}^\top,\;n_{ba}^\top\,]^\top$ of this 12-vector is exactly the column ordering of the $F_w$ matrix in §04.5 — keep it consistent. (It matches `process_noise_ikfom` in `use-ikfom.hpp:28-33`, whose members are `ng, na, nbg, nba`.)

> **Code/sign note (load-bearing).** FAST-LIO normalizes the *measured* mean acceleration to one gravity at init (it scales every raw $\mathbf a_m$ by `G_m_s2 / mean_acc.norm()`, `IMU_Processing.hpp:266`) and tracks gravity with magnitude `G_m_s2 = 9.81` (`common_lib.h:20`). The world-frame acceleration appears in `use-ikfom.hpp::get_f` as `a_inertial + s.grav`, i.e. the code stores $g$ as a vector that is *added* to $R(\mathbf a_m-b_a)$, and the init step sets `grav = -mean_acc/|mean_acc| * G_m_s2` so the stored sign already absorbs the minus. Always check the sign of `grav` against `get_f` before porting.

---

### 04.4 Initialization: estimating gravity and the biases at rest

Before propagation can begin, the estimator needs (a) the initial attitude relative to gravity, (b) the initial gyro bias, and (c) the gravity vector itself. FAST-LIO assumes the platform is **stationary** for the first $N$ IMU samples (FAST-LIO §III-E, line 644–645: "keeping the LiDAR static for several seconds … is then used to initialize the IMU bias and the gravity vector") and exploits (04.1): at rest $\boldsymbol{\omega}=0$ and $\mathbf{a}=0$, so

$$
\mathbb{E}[\boldsymbol{\omega}_m] = b_g, \qquad
\mathbb{E}[\mathbf{a}_m] = -R^\top g .
\tag{04.3}
$$

The **mean gyro reading is the gyro bias**; the **mean accel reading points along $-g$ in the body frame** and fixes both the gravity direction and (after normalization) the initial roll/pitch. Yaw about gravity is unobservable from an IMU at rest — correct, and harmless because the LiDAR/visual factors fix it.

The reference accumulates a running (Welford) mean of the IMU samples and then sets the state:

```cpp
// FAST_LIO/src/IMU_Processing.hpp:178-202  (IMU_init)
for (const auto &imu : meas.imu) {
    cur_acc << imu->linear_acceleration.x, ... ;
    cur_gyr << imu->angular_velocity.x, ... ;
    mean_acc += (cur_acc - mean_acc) / N;   // running mean of accel
    mean_gyr += (cur_gyr - mean_gyr) / N;   // running mean of gyro
    cov_acc   = cov_acc * (N-1.0)/N + (cur_acc-mean_acc).cwiseProduct(cur_acc-mean_acc)*(N-1.0)/(N*N);
    cov_gyr   = ... ;                        // running variance (diagnostics)
    N++;
}
state_ikfom init_state = kf_state.get_x();
init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);  // g = -â * |g|
init_state.bg   = mean_gyr;                                   // b_g = mean gyro
init_state.offset_T_L_I = Lidar_T_wrt_IMU;                    // extrinsic prior
init_state.offset_R_L_I = Lidar_R_wrt_IMU;
kf_state.change_x(init_state);
```

Reading off the equations against (04.3):

- `init_state.grav = S2(-mean_acc/|mean_acc| * G_m_s2)` implements $g = -\widehat{\mathbb{E}[\mathbf a_m]}\,\|g\|$ — it takes the *direction* the accelerometer feels (which is $-R^\top g$ in body frame, mapped into world via the assumed identity initial attitude) and scales it to the local gravity magnitude. Wrapping it in `S2(...)` projects this onto the 2-sphere of fixed magnitude. Because the initial $R$ is left at identity, this simultaneously defines the world frame's down axis and the initial roll/pitch.
- `init_state.bg = mean_gyr` implements $b_g=\mathbb{E}[\boldsymbol\omega_m]$ from (04.3).
- The accel bias $b_a$ is left at zero initially: at rest you cannot separate $b_a$ from gravity (both appear additively in $\mathbf a_m$), so $b_a$ is only observable once the platform accelerates and is left for the filter to learn. This is a deliberate, correct choice.

The init also seeds a small initial covariance on the bias / extrinsic / gravity blocks (`IMU_Processing.hpp:204-211`, e.g. `init_P(15..17)=1e-4` for $b_g$, `init_P(21,22)=1e-5` for the 2-D gravity tangent). The number of init samples is the threshold `MAX_INI_COUNT` (FAST-LIO2 default **10**, `IMU_Processing.hpp:30`), after which `imu_need_init_` flips false and propagation starts (`IMU_Processing.hpp:366-369`); the accel covariance is then rescaled by `pow(G_m_s2/mean_acc.norm(),2)` (line 368).

> **Meridian implication.** Tactical use cannot assume a clean static start. Meridian's L1 preprocessing should expose a *motion-detector* gate (variance of gyro/accel over a window, exactly the running `cov_gyr`/`cov_acc` already computed above) so the iEKF can either (i) wait for a static interval, or (ii) fall back to a robust dynamic-init (gravity-from-LiDAR-plane + zero-velocity update). The static estimator above is the v1 baseline; flag the assumption in the operator UI (L6). FAST-LIVO2 carries an explicit `Set_init` that aligns an arbitrary measured gravity to the canonical down vector via the geodesic rotation `Exp(align_angle)` (`FAST-LIVO2/.../IMU_Processing` `Set_init`) — a cleaner primitive to adopt than fixing $R=I$.

---

### 04.5 Forward propagation: the continuous model and its discretization

Between two LiDAR scans the filter integrates the IMU. We do this in two pieces: the **state mean** propagates through the nonlinear kinematics $f$, and the **covariance** propagates linearly through the Jacobians $F_x$ (state transition) and $F_w$ (noise input). This is the heart of the ESIKF prediction step and is shared, with minor variations, by FAST-LIO2, FAST-LIVO2, Point-LIO, and Meridian v1. FAST-LIO2 paper eq. (6) states it compactly: propagate the state by setting noise to zero, $x_{i+1}=x_i\boxplus(\Delta t\,f(x_i,u_i,0))$, and propagate the covariance by $P_{i+1}=F_{x_i}P_iF_{x_i}^\top+F_{w_i}Q_iF_{w_i}^\top$ (line 217–219).

#### 04.5.1 Continuous-time kinematics

With the IMU as input $u=(\boldsymbol\omega_m,\mathbf a_m)$ and noise $w=(n_g,n_a,n_{bg},n_{ba})$, the continuous kinematics are (FAST-LIO eq. (1), line 182–186):

$$
\dot R = R\,(\boldsymbol\omega_m - b_g - n_g)^{\wedge}, \qquad
\dot p = v, \qquad
\dot v = R\,(\mathbf a_m - b_a - n_a) + g,
$$
$$
\dot b_g = n_{bg}, \qquad \dot b_a = n_{ba}, \qquad \dot g = 0,
\qquad \dot R_{LI}=0,\ \dot p_{LI}=0.
\tag{04.4}
$$

This $f(x,u,w)$ (with $w$ set to zero for the mean) is **exactly** `get_f`:

```cpp
// FAST_LIO/include/use-ikfom.hpp:47-59  (get_f)
Eigen::Matrix<double,24,1> get_f(state_ikfom &s, const input_ikfom &in) {
  Eigen::Matrix<double,24,1> res = ...::Zero();
  vect3 omega;  in.gyro.boxminus(omega, s.bg);     // ω_m ⊟ b_g
  vect3 a_inertial = s.rot * (in.acc - s.ba);       // R (a_m − b_a)   [world frame]
  for (int i = 0; i < 3; i++) {
    res(i)      = s.vel[i];                          // ṗ = v
    res(i + 3)  = omega[i];                          // body rate ω (applied via Exp later)
    res(i + 12) = a_inertial[i] + s.grav[i];         // v̇ = R(a_m−b_a) + g
  }
  return res;   // bias, extrinsic, gravity blocks all have zero drift
}
```

Map this to (04.4): `res(0..2)=vel` is $\dot p=v$; `res(3..5)=omega` is the body rate that becomes $\dot R = R\,\boldsymbol\omega^\wedge$ when the integrator applies $\mathrm{Exp}$; `res(12..14)=R(a_m-b_a)+grav` is $\dot v$. Note the **noise $w$ is dropped from the mean** $f$ — it enters only the covariance, the standard nominal/error-state split (Section [09](09_recursive.md)).

#### 04.5.2 Discrete mean update (the $\boxplus$ integration)

Integrating (04.4) over $\Delta t$ on the manifold (Section [02](02_manifolds.md)) gives the discrete propagation $\hat x_{k+1} = \hat x_k \boxplus (\Delta t\, f)$, written component-wise:

$$
\begin{aligned}
R_{k+1} &= R_k\,\mathrm{Exp}\big((\boldsymbol\omega_m - b_g)\,\Delta t\big), \\
p_{k+1} &= p_k + v_k\,\Delta t + \tfrac12\big(R_k(\mathbf a_m - b_a) + g\big)\Delta t^2, \\
v_{k+1} &= v_k + \big(R_k(\mathbf a_m - b_a) + g\big)\Delta t, \\
b_{g,k+1}&=b_{g,k},\quad b_{a,k+1}=b_{a,k},\quad g_{k+1}=g_k .
\end{aligned}
\tag{04.5}
$$

In the IKFoM toolkit this is the single `x_.oplus(f_, dt)` call (`esekfom.hpp:287`); the FAST-LIVO2 / FAST-LIO-v1 code spells the same update out explicitly (`FAST-LIVO2/.../IMU_Processing.cpp:412-421`): `R_imu = R_imu * Exp_f; acc_imu = R_imu*acc_avr + gravity; pos_imu += vel_imu*dt + 0.5*acc_imu*dt*dt; vel_imu += acc_imu*dt`. The rotation block uses `Exp` from `so3_math.h`:

```cpp
// FAST_LIO/include/so3_math.h:37-58  (Exp: SO(3) exponential, Rodrigues, ω·dt form)
template<typename T, typename Ts>
Eigen::Matrix<T,3,3> Exp(const Eigen::Matrix<T,3,1>& ang_vel, const Ts& dt) {
  T ang_vel_norm = ang_vel.norm();
  Eigen::Matrix<T,3,3> Eye3 = ...::Identity();
  if (ang_vel_norm > 0.0000001) {
    Eigen::Matrix<T,3,1> r_axis = ang_vel / ang_vel_norm;
    Eigen::Matrix<T,3,3> K;  K << SKEW_SYM_MATRX(r_axis);
    T r_ang = ang_vel_norm * dt;
    return Eye3 + std::sin(r_ang)*K + (1.0 - std::cos(r_ang))*K*K;  // Rodrigues
  } else return Eye3;   // small-angle: Exp ≈ I
}
```

The small-angle guard (`norm <= 1e-7 → I`) avoids division blow-up; this is the standard guarded Rodrigues formula (Section [02](02_manifolds.md)). `Log` is its inverse (`so3_math.h:82-87`), with its own near-identity guard `theta < 1e-3`.

#### 04.5.3 The error-state transition $F_x$ and noise input $F_w$

For covariance propagation we linearize the *error* dynamics. Define the error state $\delta x = x \boxminus \hat x$ (Section [02](02_manifolds.md)). The discrete error propagation and covariance recursion are

$$
\delta x_{k+1} = F_x\,\delta x_k + F_w\, w_k, \qquad
\Sigma_{k+1} = F_x\,\Sigma_k\,F_x^\top + F_w\,Q\,F_w^\top .
\tag{04.6}
$$

These are the matrices FAST-LIO writes as $F_x$ and $F_w$ (eq. (7)–(8)) and FAST-LIO2 writes as $F_{\tilde x},F_w$ (eq. (7)). FAST-LIO gives the *explicit* closed form (paper eq. (7)); the rotation–rotation block is $\mathrm{Exp}(-\hat{\boldsymbol\omega}_i\Delta t)$ and the bias-coupling blocks carry the **right-Jacobian** factor $A(\hat{\boldsymbol\omega}_i\Delta t)^\top$, where (paper eq. (6))

$$
A(u)^{-1} = I - \tfrac12 u^\wedge + \Big(1-\alpha(\|u\|)\Big)\frac{(u^\wedge)^2}{\|u\|^2}, \qquad \alpha(m)=\tfrac{m}{2}\cot\tfrac{m}{2}.
\tag{04.7}
$$

In the IKFoM toolkit (FAST-LIO2) these matrices are split into the continuous Jacobians `df_dx`/`df_dw` and the $\Delta t$, right-Jacobian, and $S^2$ retraction are folded inside `predict`. The continuous state Jacobian is:

```cpp
// FAST_LIO/include/use-ikfom.hpp:61-77  (df_dx — continuous A = ∂f/∂δx)
Eigen::Matrix<double,24,23> cov = ...::Zero();
cov.block<3,3>(0,12) = Matrix3d::Identity();                 // ∂ṗ/∂δv = I
vect3 acc_;  in.acc.boxminus(acc_, s.ba);                     // a_m − b_a
cov.block<3,3>(12,3)  = -s.rot.toRotationMatrix()*MTK::hat(acc_);  // ∂v̇/∂δθ = -R[a]_×
cov.block<3,3>(12,18) = -s.rot.toRotationMatrix();           // ∂v̇/∂δb_a = -R
Eigen::Matrix<double,3,2> grav_matrix;
s.S2_Mx(grav_matrix, vec, 21);                               // S² tangent basis B_g
cov.block<3,2>(12,21) = grav_matrix;                         // ∂v̇/∂(δg on S²)
cov.block<3,3>(3,15)  = -Matrix3d::Identity();               // ∂ω/∂δb_g = -I  (rotation row)
```

```cpp
// FAST_LIO/include/use-ikfom.hpp:80-88  (df_dw — continuous G = ∂f/∂w)
Eigen::Matrix<double,24,12> cov = ...::Zero();
cov.block<3,3>(12,3) = -s.rot.toRotationMatrix();   // ∂v̇/∂n_a = -R
cov.block<3,3>(3,0)  = -Matrix3d::Identity();        // ∂ω/∂n_g = -I  (rotation row)
cov.block<3,3>(15,6) = Matrix3d::Identity();         // ∂ḃ_g/∂n_{bg} = I
cov.block<3,3>(18,9) = Matrix3d::Identity();         // ∂ḃ_a/∂n_{ba} = I
```

Writing the continuous error Jacobian $A=\partial f/\partial \delta x$ in block form (rows/cols ordered $[\,\delta\theta,\ \delta p,\ \delta v,\ \delta b_g,\ \delta b_a,\ \delta g\,]$, suppressing the static extrinsics) and reading the code:

$$
A=
\begin{bmatrix}
-\,(\boldsymbol\omega_m-b_g)^{\wedge} & 0 & 0 & -I & 0 & 0\\[2pt]
0 & 0 & I & 0 & 0 & 0\\[2pt]
-\,R(\mathbf a_m-b_a)^{\wedge} & 0 & 0 & 0 & -R & B_g\\[2pt]
0&0&0&0&0&0\\
0&0&0&0&0&0\\
0&0&0&0&0&0
\end{bmatrix},
\tag{04.8}
$$

where $B_g$ is the $3\times2$ map from the $S^2$ tangent of gravity into $\mathbb{R}^3$ (the `grav_matrix`/`S2_Mx` term — the only place the $S^2$ parameterization changes the algebra). Each nonzero block of (04.8) is exactly one line of `df_dx`:

| Block of $A$ | Meaning | Code line (`use-ikfom.hpp`) |
|---|---|---|
| $\partial\dot p/\partial \delta v=I$ | velocity drives position | `cov.block<3,3>(0,12)=I` |
| $\partial\dot v/\partial\delta\theta=-R[\mathbf a_m-b_a]_\times$ | attitude tilt rotates felt accel | `cov.block<3,3>(12,3)=-R*hat(acc_)` |
| $\partial\dot v/\partial\delta b_a=-R$ | accel-bias error feeds velocity | `cov.block<3,3>(12,18)=-R` |
| $\partial\dot v/\partial\delta g=B_g$ | gravity-direction error feeds velocity | `cov.block<3,2>(12,21)=grav_matrix` |
| $\partial\dot\theta/\partial\delta b_g=-I$ | gyro-bias error feeds rotation | `cov.block<3,3>(3,15)=-I` |

The discrete transition is then $F_x = I + A\,\Delta t$ (with the rotation block replaced by the exact $\mathrm{Exp}(-\hat{\boldsymbol\omega}\Delta t)$ and the bias-coupling blocks weighted by the right-Jacobian, as in (04.7)), and $F_w = G\,\Delta t$. The toolkit folds all of this into `predict`:

```cpp
// FAST_LIO/include/IKFoM_toolkit/esekfom/esekfom.hpp:279-382  (predict)
flatted_state f_  = f(x_, i_in);     // get_f      (eq. 04.5 integrand)
cov_  f_x_ = f_x(x_, i_in);          // df_dx → A
Matrix f_w_ = f_w(x_, i_in);         // df_dw → G
x_.oplus(f_, dt);                    // mean: x ⊞ f·dt          (eq. 04.5)
F_x1 = cov::Identity();
// ... SO3 block: F_x1.block<3,3>(idx,idx) = Exp(-f·dt);   S2 block via Nx·Exp·Mx ...
F_x1 += f_x_final * dt;              // F_x = I + A·dt          (eq. 04.6)
P_ = (F_x1) * P_ * (F_x1).transpose()
   + (dt * f_w_final) * Q * (dt * f_w_final).transpose();   // Σ ← FxΣFxᵀ + FwQFwᵀ
```

That single `P_ = F_x P F_x^T + F_w Q F_w^T` line **is** the discrete covariance propagation of (04.6) (= FAST-LIO eq. (8), FAST-LIO2 eq. (6)), and it is the only place the IMU process noise $Q$ from §04.3 enters the front-end. The `f_x_final`/`f_w_final` remapping (`esekfom.hpp:303-371`) applies the $SO(3)$ right-Jacobian `A_matrix(seg_SO3)` to the rotation rows and the `Nx·Exp·Mx` retraction to the $S^2$ gravity rows — the concrete realization of the $A(\cdot)^\top$ factor in (04.7) and the reason the abstract $I+A\Delta t$ is not literally what the code multiplies.

> **Intuition for the two key blocks.** $\partial\dot v/\partial\delta\theta=-R[\mathbf a]_\times$ says: if your estimated attitude is tilted by a small $\delta\theta$, you mis-rotate the measured specific force, so your velocity estimate drifts — this is the dominant coupling that *makes attitude observable from acceleration*, and conversely why a poor accel makes attitude drift. $\partial\dot\theta/\partial\delta b_g=-I$ says gyro-bias error integrates straight into attitude error — which is why estimating $b_g$ well is the single most important thing for orientation drift.

A practical note from the FAST-LIVO2 source: the per-interval IMU input uses the **midpoint** of consecutive samples, `angvel_avr = 0.5*(head.gyro + tail.gyro)`, `acc_avr = 0.5*(head.acc + tail.acc)` (`FAST-LIVO2/.../IMU_Processing.cpp:334-341`; same in FAST-LIO `IMU_Processing.hpp:257-262`). This trapezoidal averaging is a cheap second-order improvement over a zero-order hold and is what Meridian should use.

---

### 04.6 Backward propagation: motion de-skew of the LiDAR sweep

A LiDAR sweep is **not** instantaneous: points arrive across ~100 ms while the platform moves, so every point is captured at a different pose (FAST-LIO §I, line 25: "laser points in a scan are always sampled at different times, resulting in motion distortion"). Tightly-coupled LiDAR-inertial systems exploit the high-rate IMU to **de-skew** (motion-compensate) the sweep: transform every point into a single reference time. FAST-LIO uses the **scan-end** time (FAST-LIO §III-C.2). This is *backward* propagation — we already forward-propagated to the scan end, now we walk *backwards* through the IMU queue to find each point's pose relative to the end.

The geometry: a point $p_{L,j}$ measured at time $t_j$ in the LiDAR frame must be expressed in the LiDAR frame *at the scan-end time* $t_e$. Using the IMU pose $T(t)=(R(t),p(t))$ and the LiDAR-IMU extrinsic $T_{LI}=(R_{LI},p_{LI})$, FAST-LIO eq. (10) is

$$
p_{L,j}^{\,(e)} \;=\; T_{LI}^{-1}\, T(t_e)^{-1}\, T(t_j)\, T_{LI}\; p_{L,j}.
\tag{04.9}
$$

The reference computes $T(t_j)$ by integrating from the cached per-IMU-interval poses (`IMUpose[]`, filled during forward propagation) and walks the point cloud *backwards*. This is the de-skew loop — quoted in full because it is the most error-prone part of any LIO port:

```cpp
// FAST_LIO/src/IMU_Processing.hpp:310-345  (UndistortPcl — backward propagation + de-skew)
auto it_pcl = pcl_out.points.end() - 1;                        // start from LAST point
for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) {
  auto head = it_kp - 1;
  auto tail = it_kp;
  R_imu  << MAT_FROM_ARRAY(head->rot);                          // R at interval head
  vel_imu<< VEC_FROM_ARRAY(head->vel);
  pos_imu<< VEC_FROM_ARRAY(head->pos);
  acc_imu<< VEC_FROM_ARRAY(tail->acc);                          // accel (world) over interval
  angvel_avr << VEC_FROM_ARRAY(tail->gyr);                      // bias-corrected avg rate

  // walk every point whose timestamp falls in this IMU interval (backward)
  for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
    dt = it_pcl->curvature / double(1000) - head->offset_time;  // Δt within interval

    M3D R_i(R_imu * Exp(angvel_avr, dt));                       // R(t_j) = R_head·Exp(ω·dt)
    V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);                   // raw LiDAR point
    V3D T_ei(pos_imu + vel_imu*dt + 0.5*acc_imu*dt*dt           // p(t_j) − p(t_e)
             - imu_state.pos);
    // transform to scan-end LiDAR frame  (eq. 04.9 expanded with extrinsics):
    V3D P_compensate = imu_state.offset_R_L_I.conjugate() *
        ( imu_state.rot.conjugate() *
          ( R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei )
          - imu_state.offset_T_L_I );

    it_pcl->x = P_compensate(0); it_pcl->y = P_compensate(1); it_pcl->z = P_compensate(2);
    if (it_pcl == pcl_out.points.begin()) break;
  }
}
```

Step by step against (04.9):

1. **Per-interval relative pose.** `R_i = R_imu * Exp(angvel_avr, dt)` integrates rotation from the interval *head* to the point time $t_j$ using the bias-corrected average angular rate — a single $\mathrm{Exp}$ step. `T_ei = pos_imu + vel·dt + ½·acc·dt² − imu_state.pos` is the point-time position *expressed relative to the scan-end position* `imu_state.pos`; that subtraction is what makes it scan-end-referenced.
2. **Lift LiDAR point into IMU frame at $t_j$.** `offset_R_L_I * P_i + offset_T_L_I` applies $T_{LI}$: $p_{L}\to p_{I}$.
3. **Map to world via the $t_j$ pose, then back into the scan-end LiDAR frame.** `R_i * (...) + T_ei` puts the point in the world frame at $t_j$ (relative to scan end); `imu_state.rot.conjugate()*( … )` rotates into the scan-end IMU frame ($R(t_e)^\top$); then `offset_R_L_I.conjugate()*( … − offset_T_L_I)` applies $T_{LI}^{-1}$ back to the LiDAR frame. That composition is exactly $T_{LI}^{-1}T(t_e)^{-1}T(t_j)T_{LI}$ of (04.9).

FAST-LIO's paper presents the equivalent *recursive backward* form (eq. (9)): starting from the scan-end state, $^{I_k}p_{I_{j-1}} = {}^{I_k}p_{I_j} - {}^{I_k}v_{I_j}\Delta t$, $^{I_k}R_{I_{j-1}} = {}^{I_k}R_{I_j}\,\mathrm{Exp}((b_g-\boldsymbol\omega_{m,i-1})\Delta t)$, "s.f." (starting from) $^{I_k}p_{I_m}=0$, $^{I_k}R_{I_m}=I$. The paper notes (line 392) that because the bias/extrinsic blocks of $f$ are zero, back-propagation **only** needs to integrate pose/velocity — the cheap reduced recursion the code exploits.

Note the **timestamp encoding trick**: `it_pcl->curvature / 1000` is the per-point *relative time in seconds* (FAST-LIO stuffs the per-point offset-from-scan-start, in ms, into the unused PCL `curvature` field during preprocessing — see `preprocess.cpp`). The loop iterates points *and* IMU intervals **in lockstep from the end backwards**, so both iterators decrement; this is an $O(N_{\text{pts}}+N_{\text{imu}})$ single pass, not a per-point search. (FAST-LIVO2's `UndistortPcl`, lines 514-533, is identical up to precomputing `extR_Ri = R_{LI}^\top R(t_e)^\top` and `exrR_extT = R_{LI}^\top p_{LI}` outside the loop — a small optimization Meridian should copy.)

> **Meridian implication (multi-LiDAR).** With several Ousters (incl. the upward dome) on one PTP clock (cross-cutting time sync), each LiDAR's points carry their own per-point timestamps but share **one** IMU-integrated trajectory. De-skew (04.9) is run per-LiDAR with that LiDAR's own $T_{LI}$ extrinsic, all referenced to a *common* sweep-end time. The single-trajectory, per-sensor-extrinsic structure here is exactly what the L2 front-end must generalize. This also motivates Meridian's CT-spline v2 (Section [10](10_continuous_time.md)): a continuous $T(t)$ makes (04.9) exact at *any* query time rather than piecewise-constant-rate, which matters when LiDARs are not phase-aligned.

---

### 04.7 The smoothing counterpart: IMU preintegration (Forster)

The filter view above *consumes* the IMU as a one-shot prediction and then discards it. The **smoothing** view, used by Meridian's L3 GTSAM/iSAM2 back-end (Sections [03](03_probability.md), [08](08_solving_batch.md), [09](09_recursive.md)), needs something different: a **single relative-motion factor** between two keyframes $i$ and $j$ that summarizes *all* the IMU samples in between, **without** re-integrating them every time the linearization point (and thus the bias) changes. That is the contribution of Forster et al., *IMU Preintegration on Manifold* (RSS 2015 / T-RO 2017), cited by FAST-LIO as ref. [15] (paper line 751) and used by LIOM/LIO-SAM in the back-end (FAST-LIO2 related-work, line 26).

#### 04.7.1 The idea: factor out the keyframe state

Naïvely, the relative motion from $i$ to $j$ depends on the absolute state at $i$ (its rotation rotates every subsequent acceleration; gravity acts for the whole interval). If we re-optimize $R_i$, all the integration would have to be redone. Preintegration **algebraically separates** the part that depends only on the IMU readings and biases from the part that depends on the keyframe states. Define the **preintegrated measurements** (Forster eq. (35)–(37)):

$$
\begin{aligned}
\Delta R_{ij} &\;=\; \prod_{k=i}^{j-1} \mathrm{Exp}\big((\boldsymbol\omega_{m,k}-b_g)\,\Delta t\big), \\[4pt]
\Delta v_{ij} &\;=\; \sum_{k=i}^{j-1} \Delta R_{ik}\,(\mathbf a_{m,k}-b_a)\,\Delta t, \\[4pt]
\Delta p_{ij} &\;=\; \sum_{k=i}^{j-1}\Big[\Delta v_{ik}\,\Delta t + \tfrac12\,\Delta R_{ik}\,(\mathbf a_{m,k}-b_a)\,\Delta t^2\Big].
\end{aligned}
\tag{04.10}
$$

Crucially, (04.10) involves **only** IMU readings and the biases — **not** $R_i, v_i, p_i, g$. Those re-enter only in the residual, where gravity and the start state are reintroduced in closed form.

#### 04.7.2 The inertial residual

Given the two keyframe states, the preintegrated **inertial residual** (the quantity the back-end's least-squares minimizes; Forster eq. (45)) is

$$
\boxed{
\begin{aligned}
r_{\Delta R} &= \mathrm{Log}\!\Big( \Delta R_{ij}(b_g)^\top\, R_i^\top R_j \Big),\\[4pt]
r_{\Delta v} &= R_i^\top\big(v_j - v_i - g\,\Delta t_{ij}\big) - \Delta v_{ij}(b_g,b_a),\\[4pt]
r_{\Delta p} &= R_i^\top\big(p_j - p_i - v_i\,\Delta t_{ij} - \tfrac12 g\,\Delta t_{ij}^2\big) - \Delta p_{ij}(b_g,b_a).
\end{aligned}}
\tag{04.11}
$$

Read these as "predicted-minus-measured relative motion, with gravity and the start state put back in." The $R_i^\top(\cdot)$ projections express the world-frame relative motion in the body frame $i$, where the preintegrated quantities live. This residual $r=[r_{\Delta R}^\top,r_{\Delta v}^\top,r_{\Delta p}^\top]^\top$, weighted by the **preintegrated covariance** $\Sigma_{ij}$ (propagated alongside (04.10) by the same $F_x/F_w$ recursion of §04.5 applied to the *relative* 9-DOF state $[\delta\phi,\delta v,\delta p]$), is the inertial factor:

$$
\mathcal{C}_{\text{IMU}}(x_i,x_j) = \tfrac12\,\| r(x_i,x_j) \|^2_{\Sigma_{ij}^{-1}} = \tfrac12\, r^\top\,\Omega_{ij}\, r, \qquad \Omega_{ij}=\Sigma_{ij}^{-1}.
\tag{04.12}
$$

This is one node-pair factor in the factor graph of Section [03](03_probability.md), and GTSAM's `ImuFactor`/`PreintegratedImuMeasurements` implement exactly (04.10)–(04.12).

#### 04.7.3 The bias trick (the reason preintegration scales)

When the optimizer changes the bias estimate during iSAM2 iterations, recomputing $\Delta R_{ij},\Delta v_{ij},\Delta p_{ij}$ from scratch would be expensive. Forster's **first-order bias correction** (eq. (44)) updates the preintegrals *linearly* via cached Jacobians:

$$
\Delta R_{ij}(b_g) \approx \Delta\bar R_{ij}\,\mathrm{Exp}\!\Big(\tfrac{\partial \Delta R_{ij}}{\partial b_g}\,\delta b_g\Big),\quad
\Delta v_{ij}(b) \approx \Delta\bar v_{ij} + \tfrac{\partial\Delta v}{\partial b_g}\delta b_g + \tfrac{\partial\Delta v}{\partial b_a}\delta b_a,
\tag{04.13}
$$

and similarly for $\Delta p_{ij}$, where the bars denote the values at the bias linearization point. The Jacobians $\partial\Delta R/\partial b_g$ etc. are accumulated *during* the forward integration (04.10). This is what lets a keyframe factor be relinearized for free as the bias estimate improves — the property that makes preintegration the standard back-end inertial factor.

#### 04.7.4 Filter-propagation vs preintegration: the same physics, two accountings

| | **Filter propagation** (§04.5, FAST-LIO2/iEKF) | **Preintegration** (§04.7, Forster/GTSAM) |
|---|---|---|
| Role of IMU | input $u$ to motion model $f(x,u,w)$ | measurement producing residual $r$ |
| What is integrated | absolute state $x_{k}$, every tick | *relative* motion $\Delta R,\Delta v,\Delta p$ between keyframes |
| Frame of integration | world frame, anchored to current state | body frame $i$, **state-independent** |
| Re-linearization cost | none (one-shot, then discarded) | $O(1)$ via bias-Jacobian trick (04.13) |
| Where it enters | shapes prior $\Sigma$ before the LiDAR update | one factor in the MAP factor graph |
| Meridian layer | L2 front-end (per scan) | L3 back-end (per keyframe pair) |
| Noise model | $Q$ via $F_wQF_w^\top$ (eq. 04.6) | $\Sigma_{ij}$, same recursion on relative state |

They are **not** competing; they are the *same kinematics* (04.4) integrated against two different "anchors." The filter integrates the *absolute* state and folds the IMU into a one-step Gaussian prior that the LiDAR update then corrects (Section [09](09_recursive.md) shows the iEKF update is itself a Gauss-Newton MAP step — FAST-LIO proves the iterated-Kalman update solves the MAP cost (04.12)-analogue, paper eq. (17)–(20)). Preintegration integrates the *relative* state once, so the resulting factor survives re-linearization in the global optimization. Section [12](12_synthesis.md) shows both running in the same estimator: the iEKF front-end uses §04.5 every scan, emits keyframes, and the back-end stitches them with §04.7 preintegration factors plus loop closures (Sections [05](05_lidar.md), [11](11_robustness.md)).

#### 04.7.5 A third view worth knowing: IMU-as-measurement (Point-LIO)

Point-LIO takes a notable alternative that Meridian should be aware of: it **augments the state with the body acceleration and angular velocity** and treats each *raw* IMU reading as a *direct measurement* of those state variables (not as an input). The IMU model (04.1) then becomes an output equation $z = h(x) + n$ with a genuine residual:

```cpp
// Point-LIO/src/Estimator.cpp:324-329  (h_model_IMU_output — IMU treated as measurement)
ekfom_data.z_IMU.block<3,1>(0,0) = angvel_avr - s.omg - s.bg;             // r_ω = ω_m − (ω̂ + b_g)
ekfom_data.z_IMU.block<3,1>(3,0) = acc_avr * G_m_s2 / acc_norm - s.acc - s.ba;  // r_a = a_m − (â + b_a)
ekfom_data.R_IMU << imu_meas_omg_cov, ..., imu_meas_acc_cov, ...;          // measurement noise
```

Here `s.omg` and `s.acc` are *state* variables (the `state_output` manifold, 30-DOF, with its own `get_f_output`/`df_dx_output` in `Estimator.cpp:68-110` using a constant-acceleration/white-noise-on-jerk dynamics). The residual directly compares the measured rate/accel to the estimated body rate/accel plus bias. A distinctive robustness feature: the loop checks IMU **saturation** (`satu_check`, lines 330-367) and, if a channel saturates, *zeroes that residual component* so the constant-acceleration model carries the estimate through the clipped interval. The payoff is that Point-LIO has *no per-scan batch* — it processes **point by point** at IMU rate and naturally handles aggressive motion and IMU saturation. The cost is a larger state and a tuned process model. Meridian's `IFrontEnd` interface should keep this option open (it is a clean swap behind the same interface as the iEKF), but v1 uses the FAST-LIO2 input-model propagation of §04.5 because it is simpler to reason about and to instrument.

---

### 04.8 What to instrument (Meridian L2 debug topics)

Per the non-negotiable "debug in the right places" principle, the IMU module is where the estimator's *prior* is born, so it must be observable:

- **Bias trajectories** $b_g(t), b_a(t)$ as ROS 2 topics — divergence here is the earliest sign of a failing estimate; plot against the init values from §04.4.
- **Per-stage timing**: IMU init, forward propagation, covariance update, de-skew loop — the de-skew (§04.6) is $O(N_{\text{pts}})$ and is a common hotspot with multi-LiDAR. (FAST-LIO already brackets this with `omp_get_wtime()`, `IMU_Processing.hpp:350-385`.)
- **Effective $\Delta t$ and IMU rate jitter** — PTP sync (cross-cutting) failures show up as irregular $\Delta t$ that silently inflate $F_wQF_w^\top$; expose the measured rate and gaps.
- **Process-noise contribution** $\mathrm{tr}(F_wQF_w^\top)$ vs $\mathrm{tr}(F_x\Sigma F_x^\top)$ per step — tells the operator whether the prior is dominated by IMU noise or by propagated state uncertainty.
- **De-skew sanity marker**: publish a few de-skewed vs raw points as rviz markers so an operator can *see* motion compensation working (a fast turn should visibly straighten).
- **Gravity-direction error** when on $S^2$: the 2-D gravity error covariance (the `init_P(21,22)` block, §04.4) is a direct, interpretable roll/pitch confidence for the operator overlay (L6).

---

### 04.9 Summary

The IMU enters a tightly-coupled estimator through one physical model (04.1)–(04.2) and three computational guises: (i) **forward propagation** — the nonlinear mean update (04.5) plus the linear covariance recursion (04.6) with $F_x=I+A\Delta t$ (rotation block $\mathrm{Exp}(-\hat{\boldsymbol\omega}\Delta t)$, bias coupling weighted by the right-Jacobian of (04.7)) and $F_w$ exactly as coded in `df_dx`/`df_dw` — which builds the per-scan prior in Meridian's L2 iEKF; (ii) **backward propagation** — the de-skew (04.9) that motion-compensates the LiDAR sweep using the same integrated trajectory; and (iii) **preintegration** — the relative-motion factor (04.10)–(04.12) with the bias-Jacobian trick (04.13) that lets Meridian's L3 GTSAM back-end carry a single, re-linearizable inertial constraint between keyframes. Filter propagation and preintegration are the same kinematics anchored to the absolute vs the relative state; Point-LIO's IMU-as-measurement is a fourth, swappable option behind `IFrontEnd`. The LiDAR residual that this prior is corrected against is the subject of Section [05](05_lidar.md); the recursive solver that performs that correction (and its MAP equivalence) is Section [09](09_recursive.md); and Section [12](12_synthesis.md) runs the whole step end to end.


---


## 05. LiDAR residual: point-to-plane, association, map

> **Where we are.** Section 03 cast the whole estimator as a Maximum-A-Posteriori (MAP) problem that reduces to a weighted nonlinear least-squares sum of *residual factors*. Section 04 supplied the inertial factor (the IMU prior / preintegration that propagates the state and gives the prediction we linearise around). This section supplies the **single most informative geometric factor in a LiDAR-inertial system: the point-to-plane residual.** We answer four questions, in order:
>
> 1. *Association* — given one LiDAR point, which surface in the map does it correspond to? (§05.3: 5-NN plane fit + validation.)
> 2. *Residual* — what scalar do we drive to zero, and how confident are we in it? (§05.4 the residual, §05.6 the noise/weight.)
> 3. *Jacobian* — how does that scalar change as the state moves on the manifold? We **derive** $H_j$ from scratch w.r.t. rotation, translation and the LiDAR–IMU extrinsic, and check it line-by-line against the FAST-LIO code. (§05.5.)
> 4. *The map* — what data structure holds the surfaces, and how is it grown, pruned and kept query-fast in real time? (§05.7: the ikd-Tree.)
>
> The visual photometric residual is the subject of §06; GNSS in §07; how all these residuals are *solved together* (batch GN/LM vs. iterated EKF) is §08–§09; degeneracy/observability and robust kernels that re-weight this residual are §11; and §12 stitches one full estimator step around exactly the code we dissect here.

We ground every claim in the FAST-LIO / FAST-LIO2 reference implementation on disk. The central function is `h_share_model` in `FAST_LIO/src/laserMapping.cpp:638` — it is called once per solver iteration and is, almost line for line, "the LiDAR residual." Read it open beside this section.

---

### 05.1 Why the LiDAR factor matters, and what shape it takes

A LiDAR returns a dense set of 3-D points, each an accurate range measurement along a known beam direction. Unlike a camera it measures *depth directly*, so the natural error metric is geometric distance rather than photometric difference (§06). But a raw point has no semantic identity — there is no "feature ID" telling you that this return corresponds to that return three scans ago. We must *invent* the correspondence each iteration, and we must choose what geometric primitive to measure distance to.

Three classical choices (FAST-LIO2 paper §I-A, `papers/2107.06829.txt:19`):

- **point-to-point** ($z = \lVert {}^{G}p_j - q \rVert$): correspondences are unstable; the nearest map point is rarely the same physical surface point, because LiDAR scans never sample the same spot twice. Used in raw ICP, fragile.
- **point-to-line** (distance to an edge fitted from neighbours): the LOAM "corner" feature. Constrains 2 of 3 translational DoF locally.
- **point-to-plane** (distance to a locally-fitted plane): the LOAM "surf" feature, and the metric FAST-LIO uses for *all* points. A plane is the most common local surface in man-made and natural scenes, the fit is statistically well-conditioned with as few as 5 points, and the residual is a clean *scalar* — exactly one constraint per point along the surface normal.

The point-to-plane choice is decisive for the rest of the chapter: because the residual is scalar, the Jacobian $H_j$ is a **single row**, the per-point measurement noise is a **scalar variance**, and stacking $m$ points gives a tall thin $H \in \mathbb{R}^{m\times n}$ whose normal equations $H^\top R^{-1} H$ are cheap to form (§08, and the iEKF form in §09 that FAST-LIO actually runs).

#### Direct vs. feature-based (the FAST-LIO2 thesis)

LOAM and its descendants first run a hand-engineered **feature extraction** stage: per scan-line, compute a local curvature $c$ and classify each point as *edge* (high $c$), *planar* (low $c$), or discard the rest. Only ~10 % of points survive, registered as point-to-line + point-to-plane.

FAST-LIO2's first headline contribution is to **delete that stage** and register *raw* (merely voxel-downsampled) points directly, all as point-to-plane (paper abstract `papers/2107.06829.txt:5`; §I-A.3 `:19`). Two consequences:

1. **Accuracy up:** subtle structure that the curvature classifier would throw away (thin poles, foliage, textured walls) still contributes constraints. The paper: direct registration "well exploit[s] the subtle features in the environments" (`:19`).
2. **Sensor-agnostic:** the curvature heuristic assumes ring-organised mechanical-LiDAR scan lines. Solid-state LiDARs (Livox) and the irregular scan patterns of our Ouster dome have *no* such structure; deleting the stage makes the front-end "naturally adaptable to emerging LiDARs of different scanning patterns" (abstract `:5`).

This is visible in the default config: `feature_extract_enable` defaults to **`false`** (`laserMapping.cpp:787`), so `Preprocess::process` only downsamples and the *only* geometric primitive built downstream is the local plane fitted on-the-fly in `h_share_model`. The feature enum (`Real_Plane`, `Edge_Jump`, …) in `preprocess.h:14` is legacy LOAM machinery, dormant by default.

> **Meridian note.** Our front-end interface `IFrontEnd` keeps the *direct* point-to-plane residual of this section as the LiDAR factor in **both** the v1 iEKF (FAST-LIO2-style, §09/§12) and the v2 continuous-time B-spline estimator (§10). What changes between v1 and v2 is *which trajectory variable* the point is transformed by (a single end-of-scan pose vs. a spline evaluated at the point's exact timestamp), not the residual or its association. With multiple Ouster LiDARs we maintain **one shared map** (§05.7) and per-sensor extrinsics ${}^{I}_{L}T$ as graph variables (the $B$-block Jacobian of §05.5 is exactly what lets us refine them online).

---

### 05.2 Notation recap and the rigid transform of a point

Using the shared notation (course §02). The state $x$ (FAST-LIO's `state_ikfom`, `use-ikfom.hpp:12`) contains, among others:

$$
R \in SO(3)\;(\texttt{rot}),\quad p \in \mathbb{R}^3\;(\texttt{pos}),\quad {}^{I}_{L}R \in SO(3)\;(\texttt{offset\_R\_L\_I}),\quad {}^{I}_{L}t \in \mathbb{R}^3\;(\texttt{offset\_T\_L\_I}),
$$

plus velocity $v$, biases $b_g,b_a$ and gravity $g$ (the inertial part, §04). Here $R,p$ are the **IMU** body pose in the global/world frame $G$, and $({}^{I}_{L}R,{}^{I}_{L}t)$ is the **LiDAR-to-IMU extrinsic** ${}^{I}_{L}T$.

A point measured in the LiDAR frame, $p_L \equiv {}^{L}p_j$, is carried to the world frame in two hops — LiDAR→IMU, then IMU→world:

$$
\boxed{\;{}^{G}p_j \;=\; R\,\big({}^{I}_{L}R\, p_L + {}^{I}_{L}t\big) \;+\; p\;}
\tag{05.1}
$$

This is *exactly* the C++ (`laserMapping.cpp:657`, inside `h_share_model`):

```cpp
V3D p_body(point_body.x, point_body.y, point_body.z);                       // p_L
V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);     // (05.1)
```

and identically in the standalone helper `pointBodyToWorld_ikfom` (`:166-169`). Define the **intermediate point in the IMU frame**

$$
p_I \;\equiv\; {}^{I}_{L}R\, p_L + {}^{I}_{L}t,
\tag{05.2}
$$

so that ${}^{G}p_j = R\,p_I + p$. We will need both $p_L$ and $p_I$ when we differentiate, because rotation acts on $p_I$ but the extrinsic acts on $p_L$. In code (`:729`) $p_I$ is `point_this`, $p_L$ is `point_this_be` ("be" = body-end, i.e. LiDAR frame).

Throughout, ${}^{G}\hat p_j^{\,\kappa}$ denotes the world point evaluated at the *current iterate* $\hat x^{\kappa}$ of the solver; the hat and superscript $\kappa$ matter because association and linearisation are redone (or re-used — §05.4) per iteration.

---

### 05.3 Data association: 5-nearest-neighbour plane fit and validation

The map (§05.7) is a set of world-frame points with no precomputed surface model. For each downsampled body point we must, *every iteration*, (i) find its neighbourhood in the map, (ii) fit a plane, and (iii) decide whether that plane is trustworthy.

#### Step 1 — k-NN search ($k=5$)

```cpp
// laserMapping.cpp:670-671
ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false :
                         pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
```

`NUM_MATCH_POINTS = 5` (`common_lib.h:26`). The ikd-Tree returns the 5 nearest map points and their **squared** distances, sorted ascending (the search itself is §05.7). Two early rejections:

- fewer than 5 neighbours exist → reject (sparse/empty region of the map);
- the **farthest** of the 5, `pointSearchSqDis[4]`, exceeds $5\,\mathrm{m}^2$ (i.e. $\sqrt5 \approx 2.24\,\mathrm{m}$) → reject. If the 5th neighbour is that far, the 5 points are not a tight local patch and any plane through them is meaningless; this is a coarse *gating* on association, analogous to a $\chi^2$ Mahalanobis gate (§11) but done on raw Euclidean distance for speed.

Why 5 and not 3? Three points define a plane *exactly* (zero residual, no redundancy, no way to detect an outlier or a curved surface). Five gives an over-determined fit whose residual spread is itself the validity test (Step 3). It is the smallest number that is both robust and cheap.

#### Step 2 — least-squares plane fit (`esti_plane`)

A plane is $n\cdot x + d = 0$ with unit normal $n$ (shared notation §02). FAST-LIO parameterises it cleverly to avoid an eigen-decomposition. Write the plane as $a x + b y + c z + 1 = 0$ (i.e. fix the inhomogeneous term to $1$, absorbing scale). Then every neighbour $(x_j,y_j,z_j)$ must satisfy

$$
\underbrace{\begin{bmatrix} x_1 & y_1 & z_1 \\ \vdots & & \vdots \\ x_5 & y_5 & z_5\end{bmatrix}}_{A\in\mathbb{R}^{5\times3}} \begin{bmatrix} a\\ b\\ c\end{bmatrix} \;=\; \underbrace{\begin{bmatrix}-1\\ \vdots\\ -1\end{bmatrix}}_{b_0}.
\tag{05.3}
$$

This is the overdetermined system solved in `common_lib.h:226-257` (`esti_plane`), with the geometry explained verbatim in the comment block at `common_lib.h:185-191`:

```cpp
// esti_plane, common_lib.h:241-247
Matrix<T,3,1> normvec = A.colPivHouseholderQr().solve(b);   // (05.3), b = -1
T n = normvec.norm();
pca_result(0) = normvec(0) / n;   // n_x  (unit normal)
pca_result(1) = normvec(1) / n;   // n_y
pca_result(2) = normvec(2) / n;   // n_z
pca_result(3) = 1.0 / n;          // d   (signed offset)
```

The solve is least-squares (`colPivHouseholderQr().solve`, a column-pivoted Householder QR — numerically stable, no normal-equations squaring of the condition number). The returned 3-vector $[a,b,c]^\top$ is the *unnormalised* normal; dividing by its norm $\nu = \lVert[a,b,c]\rVert$ gives the **unit normal** $n=[a,b,c]/\nu$ and the **offset** $d=1/\nu$, so the validated plane is

$$
n\cdot x + d = 0,\qquad \lVert n\rVert = 1.
\tag{05.4}
$$

Geometrically: solving $A x_0 = -\mathbf 1$ minimises $\sum_j ( a x_j + b y_j + c z_j + 1)^2$. After normalisation, $a x_j+b y_j+c z_j+1$ over $\nu$ is the signed orthogonal distance of neighbour $j$ to the plane — which is precisely the quantity tested next. (This is the algebraic-least-squares plane fit; it coincides with the total-least-squares/PCA normal when the points are nearly co-planar, which is exactly the regime we accept.)

#### Step 3 — plane validation

```cpp
// esti_plane, common_lib.h:249-256, threshold = 0.1
for (int j = 0; j < NUM_MATCH_POINTS; j++)
  if (fabs(pca_result(0)*x_j + pca_result(1)*y_j + pca_result(2)*z_j + pca_result(3)) > threshold)
      return false;     // a neighbour is >0.1 m off the plane -> NOT a plane
return true;
```

Every one of the 5 neighbours must lie within `threshold = 0.1` m of the fitted plane (the literal `esti_plane(pabcd, points_near, 0.1f)` call at `laserMapping.cpp:678`). If any neighbour is farther, the patch is curved, a corner, or noise — the planarity assumption fails and the point is discarded for this scan. This is the on-the-fly equivalent of LOAM's curvature classifier, but data-driven and per-point.

The triad **(5-NN search) → (QR plane fit) → (0.1 m planarity gate)** is the entire association pipeline. Note there is *no persistent correspondence*: the plane is re-fitted from whatever 5 map points are currently nearest, which is why FAST-LIO re-runs association whenever the state has moved enough (the `converge` flag, §05.4).

---

### 05.4 The point-to-plane residual

With a validated plane $(n,d)$ for body point $p_L$ whose current world position is ${}^{G}\hat p_j^{\,\kappa}$ (05.1), the residual is the **signed point-to-plane distance**

$$
\boxed{\; r_j \;=\; n^\top\,{}^{G}\hat p_j^{\,\kappa} + d \;=\; n^\top\big({}^{G}\hat p_j^{\,\kappa} - q_j\big) \;}
\tag{05.5}
$$

where the two forms are equal because any point $q_j$ *on* the plane satisfies $n^\top q_j = -d$. The paper writes it in the centroid form $z_j = u_j^\top({}^{G}\hat p_j^{\,\kappa} - q_j)$ (FAST-LIO2 `papers/2107.06829.txt`, FAST-LIO `papers/2010.08196.txt:122`), with $u_j \equiv n$ the unit normal and $q_j$ a point on the plane. The code uses the offset form $n^\top {}^{G}p + d$, since `esti_plane` hands back $d$ directly:

```cpp
// laserMapping.cpp:680
float pd2 = pabcd(0)*point_world.x + pabcd(1)*point_world.y + pabcd(2)*point_world.z + pabcd(3);
```

`pd2` is exactly $r_j$ of (05.5). It is a *signed scalar*: positive on the normal side of the plane, negative on the other; the optimiser drives it to zero, sliding the estimated point onto the surface along $n$.

**Sign convention in the filter.** FAST-LIO stacks the *measurement* as $-r_j$, because the IKFoM update expects $z = h(x) - \text{measurement}$ with the linearised constraint $0 \approx r_j + H_j\,\delta x$ (paper `2010.08196.txt:128-130`). Hence:

```cpp
// laserMapping.cpp:751
ekfom_data.h(i) = -norm_p.intensity;   // norm_p.intensity stores pd2 = r_j  (set at :689)
```

The stored normal carries $r_j$ in its `.intensity` field (`:686-689`) — a neat trick to ship $(n, r_j)$ together as one `PointType` (`normvec` / `corr_normvect`).

**When is association recomputed?** The expensive 5-NN search runs only when `ekfom_data.converge` is true (`laserMapping.cpp:667`):

```cpp
if (ekfom_data.converge) {
    ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
    point_selected_surf[i] = ...;     // re-gate
}
```

On the *first* iteration `converge` is true so every point is associated; on subsequent iterations it is set false once the increment is small, **freezing the correspondences** (the `points_near` from last time are reused) and only the residual + Jacobian are recomputed at the updated state. This is the standard iterated-closest-point structure inside the iterated EKF (the equivalence to Gauss-Newton is §09): re-associating every iteration would be both costly and a source of oscillation, so FAST-LIO re-associates only until convergence stabilises.

The residual loop then compacts the *effective* points into contiguous arrays (`laserCloudOri` = body points, `corr_normvect` = normals+residuals) and counts them in `effct_feat_num` (`:697-706`). If `effct_feat_num < 1` the update is declared invalid (`:708-713`) — a degenerate-scan guard that hands off to §11's degeneracy handling.

---

### 05.5 The on-manifold Jacobian — full derivation

This is the mathematical heart of the section. We need $H_j = \partial r_j / \partial \delta x$, the derivative of the scalar residual (05.5) w.r.t. an increment in the error state, evaluated at the current iterate. Because $R,{}^{I}_{L}R \in SO(3)$ live on a manifold, we differentiate with the boxplus convention of §02:

$$
R \boxplus \delta\theta \;=\; R\,\mathrm{Exp}(\delta\theta),\qquad
p \boxplus \delta p \;=\; p + \delta p,
\tag{05.6}
$$

i.e. a **right** perturbation on rotation (matching FAST-LIO's `state_ikfom`, whose `boxplus` is `R·Exp(δθ)`). The error state ordering in `h_x` is (paper / `laserMapping.cpp:743`):

$$
\delta x \;=\; \big[\, \delta p \;\; \delta\theta \;\; \delta\theta_{LI} \;\; \delta t_{LI} \;\big]^\top
$$

(translation first, then body rotation, then extrinsic rotation, then extrinsic translation; velocity/bias/gravity columns are zero for the LiDAR factor, which is why `h_x` has only 12 columns, `:720`).

#### 05.5.1 The two facts we need

The residual depends on $x$ only through the world point ${}^{G}p_j$, so by the chain rule

$$
\frac{\partial r_j}{\partial \delta x} \;=\; n^\top \,\frac{\partial\, {}^{G}p_j}{\partial \delta x}.
\tag{05.7}
$$

The normal $n$ and offset $d$ are treated as **constants** w.r.t. $\delta x$: the plane is fitted from *map* points, not from the state, so it does not move when we perturb the pose. (This is the same modelling choice that makes point-to-plane a clean scalar factor; it is exact to first order because the map is the conditioning information, cf. §03's factor-graph view.)

We need the derivative of a *rotated vector* under the right perturbation. For $w = R\,a$ with $a$ fixed,

$$
R\,\mathrm{Exp}(\delta\theta)\,a \;\approx\; R\,(I + \lfloor\delta\theta\rfloor_\times)\,a \;=\; R\,a + R\,\lfloor\delta\theta\rfloor_\times a \;=\; R\,a - R\,\lfloor a\rfloor_\times \delta\theta,
\tag{05.8}
$$

using $\mathrm{Exp}(\delta\theta)\approx I+\lfloor\delta\theta\rfloor_\times$ (so3, `so3_math.h:18-34`) and the skew identity $\lfloor u\rfloor_\times v = -\lfloor v\rfloor_\times u$. Hence

$$
\frac{\partial (R a)}{\partial \delta\theta}\bigg|_{R} \;=\; -\,R\,\lfloor a\rfloor_\times .
\tag{05.9}
$$

The skew (hat) operator $\lfloor\cdot\rfloor_\times$ is `SKEW_SYM_MATRX` / `hat` in `so3_math.h:7,67`.

#### 05.5.2 Block A — translation $p$

${}^{G}p_j = R p_I + p$, so $\partial\,{}^{G}p_j/\partial \delta p = I_3$. Therefore

$$
\boxed{\;\frac{\partial r_j}{\partial \delta p} \;=\; n^\top I_3 \;=\; n^\top.\;}
\tag{05.10}
$$

This is the first three entries of the Jacobian row — literally `norm_p.x, norm_p.y, norm_p.z` (`laserMapping.cpp:743`, `:747`). Intuition: translating the sensor 1 m along the surface normal changes the point-to-plane distance by 1 m; translating *along* the surface changes it not at all. The translation Jacobian *is the plane normal*.

#### 05.5.3 Block — body rotation $R$ (the code's "A")

Only $R p_I$ depends on $R$. Apply (05.9) with $a = p_I$:

$$
\frac{\partial\,{}^{G}p_j}{\partial \delta\theta} \;=\; -\,R\,\lfloor p_I\rfloor_\times.
$$

Then

$$
\frac{\partial r_j}{\partial \delta\theta} \;=\; n^\top\big(-R\,\lfloor p_I\rfloor_\times\big) \;=\; -\,(R^\top n)^\top\,\lfloor p_I\rfloor_\times \;=\; \big(\lfloor p_I\rfloor_\times\, R^\top n\big)^\top,
\tag{05.11}
$$

where the last step uses $\lfloor u\rfloor_\times^\top=-\lfloor u\rfloor_\times$. Define the world normal pulled back into the IMU frame,

$$
C \;\equiv\; R^\top n \;=\; R^{-1} n,
\tag{05.12}
$$

then the rotation block is the row vector $\big(\lfloor p_I\rfloor_\times C\big)^\top$, i.e. as a stored 3-vector

$$
\boxed{\;A \;=\; \lfloor p_I\rfloor_\times\, C \;=\; \lfloor p_I\rfloor_\times\, R^\top n.\;}
\tag{05.13}
$$

Match to code (`laserMapping.cpp:738-739`):

```cpp
V3D C(s.rot.conjugate() * norm_vec);   // C = R^{-1} n        (05.12)
V3D A(point_crossmat * C);             // A = [p_I]_x C       (05.13), point_crossmat = [point_this]_x
```

with `point_this` $= p_I$ (`:729`) and `point_crossmat` $=\lfloor p_I\rfloor_\times$ (`:730-731`). `s.rot.conjugate()` is the quaternion inverse, i.e. $R^\top$. So the code's `A` *is* equation (05.13). (FAST-LIO stores the row as $A^\top$ entries via `VEC_FROM_ARRAY(A)` at `:743`; the sign is consistent with the residual sign convention of §05.4.)

Intuition: a small rotation moves the point on a circle whose instantaneous velocity is $\delta\theta \times (R p_I)$ in the world; projecting that onto the normal gives the change in point-to-plane distance. Points far from the rotation centre (large $\lVert p_I\rVert$) and oriented so their motion is along $n$ are the most rotation-informative — this is precisely the geometry §11 exploits to detect rotational degeneracy.

#### 05.5.4 Blocks B, C — the LiDAR–IMU extrinsic (online calibration)

When `extrinsic_est_en` is true (default, `laserMapping.cpp:789`), the extrinsic ${}^{I}_{L}T = ({}^{I}_{L}R,{}^{I}_{L}t)$ is *also* a state variable and gets refined online (paper FAST-LIO2 estimates extrinsics in the filter). We differentiate ${}^{G}p_j = R({}^{I}_{L}R\, p_L + {}^{I}_{L}t) + p$ w.r.t. the extrinsic.

**Extrinsic translation ${}^{I}_{L}t$.** It enters as $R\,{}^{I}_{L}t$, so

$$
\frac{\partial\,{}^{G}p_j}{\partial \delta t_{LI}} = R \quad\Rightarrow\quad \frac{\partial r_j}{\partial \delta t_{LI}} = n^\top R = (R^\top n)^\top = C^\top.
$$

$$
\boxed{\;\text{extrinsic-translation block} \;=\; C \;=\; R^\top n.\;}
\tag{05.14}
$$

**Extrinsic rotation ${}^{I}_{L}R$.** It enters through $p_I = {}^{I}_{L}R\, p_L + {}^{I}_{L}t$, then through $R\,p_I$. Right-perturb ${}^{I}_{L}R$: by (05.9) with $a=p_L$,

$$
\frac{\partial p_I}{\partial \delta\theta_{LI}} = -\,{}^{I}_{L}R\,\lfloor p_L\rfloor_\times,
\qquad
\frac{\partial\,{}^{G}p_j}{\partial \delta\theta_{LI}} = R\,\big(-\,{}^{I}_{L}R\,\lfloor p_L\rfloor_\times\big).
$$

Therefore

$$
\frac{\partial r_j}{\partial \delta\theta_{LI}} = -\,n^\top R\,{}^{I}_{L}R\,\lfloor p_L\rfloor_\times = -\,\big({}^{I}_{L}R^\top R^\top n\big)^\top \lfloor p_L\rfloor_\times = \big(\lfloor p_L\rfloor_\times\,{}^{I}_{L}R^\top C\big)^\top,
$$

giving the stored 3-vector

$$
\boxed{\;B \;=\; \lfloor p_L\rfloor_\times\,{}^{I}_{L}R^\top C \;=\; \lfloor p_L\rfloor_\times\,{}^{I}_{L}R^\top R^\top n.\;}
\tag{05.15}
$$

Match to code (`laserMapping.cpp:742`):

```cpp
V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);   // B = [p_L]_x · (offset_R)^T · C   (05.15)
```

with `point_be_crossmat` $=\lfloor p_L\rfloor_\times$ (`:727-728`, `point_this_be` $=p_L$) and `s.offset_R_L_I.conjugate()` $={}^{I}_{L}R^\top$. Equation (05.15) reproduced exactly.

#### 05.5.5 The assembled Jacobian row

Putting (05.10), (05.13), (05.15), (05.14) together, the per-point measurement-Jacobian row (12 columns) is

$$
H_j \;=\; \big[\; \underbrace{n^\top}_{\delta p,\ (05.10)} \;\;\; \underbrace{A^\top}_{\delta\theta,\ (05.13)} \;\;\; \underbrace{B^\top}_{\delta\theta_{LI},\ (05.15)} \;\;\; \underbrace{C^\top}_{\delta t_{LI},\ (05.14)} \;\big] \;\in\; \mathbb{R}^{1\times 12},
\tag{05.16}
$$

assembled verbatim at `laserMapping.cpp:743`:

```cpp
ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z,   // n^T   (delta p)
                                    VEC_FROM_ARRAY(A),              // A^T   (delta theta)
                                    VEC_FROM_ARRAY(B),              // B^T   (delta theta_LI)
                                    VEC_FROM_ARRAY(C);              // C^T   (delta t_LI)
```

If extrinsic estimation is **off** (`extrinsic_est_en == false`), columns $B,C$ are zeroed (`:747`): the extrinsic is held fixed at its prior. The remaining state blocks (velocity, biases, gravity — columns 12…22 of the full 23-dim error state) do not appear in the LiDAR factor at all; they are constrained only through the IMU factor (§04) and reach the LiDAR factor *indirectly* via the coupled covariance during the iterated update (§09). This is precisely what "tightly coupled" means at the residual level: each sensor contributes the residuals it can observe, and the shared state + joint information matrix transmits information to the rest.

> **Worked sanity check.** Sensor at origin, identity attitude ($R=I$, $p=0$), identity extrinsic. A wall straight ahead: outward normal $n=[-1,0,0]^\top$, plane at $x=5$, so $d=5$. A point on the wall $p_L=[5,0,0]^\top \Rightarrow p_I=p_L$, ${}^{G}p=[5,0,0]^\top$. Residual $r = n^\top{}^{G}p + d = -5+5 = 0$ ✓ (point on plane). Now the Jacobian: $C=R^\top n = n = [-1,0,0]^\top$; translation block $n^\top=[-1,0,0]$ — pushing $+x$ by $\epsilon$ makes $r = -(5+\epsilon)+5 = -\epsilon$, slope $-1$ ✓. Rotation block $A=\lfloor p_I\rfloor_\times C = \lfloor[5,0,0]\rfloor_\times[-1,0,0]^\top = [0,0,0]^\top$ — a point dead-ahead on the rotation axis through it produces no normal-direction motion to first order ✓. Move the point off-axis, $p_L=[5,2,0]^\top$: $A=\lfloor[5,2,0]\rfloor_\times[-1,0,0]^\top=[0,0,-2]^\top$ — yaw now changes the distance, magnitude growing with the lever arm ✓. The Jacobian behaves exactly as geometry demands.

---

### 05.6 Measurement noise and per-point weighting

The residual is only useful with a covariance attached (the $R_j$ that turns a residual into an *information-weighted* factor, §03). FAST-LIO uses two mechanisms.

#### A single isotropic measurement variance

The LiDAR ranging+bearing noise $n_j$ projected onto the plane normal yields a scalar variance $R_j$. The implementation uses one global constant for all points:

```cpp
// laserMapping.cpp:64
#define LASER_POINT_COV (0.001)
```

passed into the iterated update as the measurement covariance (`kf.update_iterated_dyn_share_modified(LASER_POINT_COV, ...)`, `:960`). So every point-to-plane factor gets the same $R_j = 0.001\,\mathrm{m}^2$ ($\sigma \approx 3.2$ cm). This is a deliberate simplification: the *true* projected noise depends on range and incidence angle (paper `2010.08196.txt:124-130` derives $v_j\sim\mathcal N(0,R_j)$ from $n_j$ via the plane normal), but a constant works well because (a) the planarity gate already removes high-incidence/curved patches, and (b) the per-point *robust weight* below soft-rejects the rest.

#### The range-dependent robust weight $s$

Before a point is even accepted, `h_share_model` computes a heuristic confidence and gates on it (`laserMapping.cpp:681-691`):

```cpp
float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());   // pd2 = r_j, p_body = p_L
if (s > 0.9) { point_selected_surf[i] = true;  ...  res_last[i] = abs(pd2); }
```

In symbols, with $\rho = \lVert p_L\rVert$ the range of the point,

$$
s \;=\; 1 - 0.9\,\frac{|r_j|}{\sqrt{\rho}}, \qquad \text{accept iff } s > 0.9 \;\Longleftrightarrow\; |r_j| < \tfrac{1}{9}\sqrt{\rho}.
\tag{05.17}
$$

Two roles:

1. **Outlier gate.** A point whose residual is large *relative to its range* is rejected outright. The $\sqrt\rho$ in the denominator makes the gate *looser for distant points* — far returns have larger absolute positional uncertainty (range noise and beam divergence grow with distance), so a 5 cm residual at 50 m is far more acceptable than at 2 m. This is a crude but effective range-adaptive $\chi^2$-style gate (cf. the principled Mahalanobis gate and GNC kernels of §11).
2. **(Historically) a weight.** In earlier FAST-LIO and in LOAM, $s$ multiplied the residual/Jacobian as a soft IRLS weight; in this code path $s$ is used purely as the binary accept gate ($s>0.9$) and the surviving residual then carries the uniform `LASER_POINT_COV`. The variable name and structure remain as the seam where a proper per-point weight (e.g. from incidence angle or §11 observability) would re-enter.

> **Meridian note.** This is exactly the place to upgrade. We will replace the constant `LASER_POINT_COV` with a **range-and-incidence dependent $R_j$** (project the Ouster's documented range-noise model and the beam/plane incidence onto $n$), and feed the **per-axis observability** of §11 (X-ICP / D2-LIO style) into the *back-end* noise so the factor graph (§03, L3) knows which directions the LiDAR actually constrained. The robust weight $s$ becomes a Geman-McClure / GNC kernel (§11) rather than a hard gate, so degenerate scans degrade gracefully instead of dropping points.

#### Stacking into the update

The accepted rows are stacked: $H \in \mathbb{R}^{m\times 12}$ (here $m=$ `effct_feat_num`), residual vector $z\in\mathbb{R}^{m}$ (the `ekfom_data.h(i) = -r_j`), and $R = \sigma^2 I_m$ with $\sigma^2=$ `LASER_POINT_COV`. The iterated-EKF update that consumes them — and its equivalence to a Gauss-Newton MAP step — is the subject of §09; the batch normal-equation view ($H^\top R^{-1} H\,\delta x = -H^\top R^{-1} z$ plus the IMU prior) is §08. What §05 guarantees is that each row of $H$ and each entry of $z$ are correct, code-faithful, and on-manifold.

---

### 05.7 The map: the incremental k-d tree (ikd-Tree)

The association of §05.3 issued one command per point — *"5-NN search in the map"* — and assumed the map answers in $O(\log n)$. The map is FAST-LIO2's second headline contribution: the **ikd-Tree** (`papers/2102.10808.txt`; `FAST_LIO/include/ikd-Tree/ikd_Tree.{h,cpp}`). It is a k-d tree that you can *grow and prune incrementally* and that *re-balances itself*, instead of being rebuilt from scratch each scan. The measured payoff: in FAST-LIO the average incremental-update cost is **0.23 ms vs. 5.71 ms** for a static PCL k-d tree — about **4 %** — enabling 100 Hz mapping (paper `2102.10808.txt:608`, abstract `:6`).

#### 05.7.1 Why not a static tree, an octree, or a voxel hash?

A classic LiDAR-SLAM map rebuilds a static k-d tree (PCL/FLANN) every scan over *all* map points — cost grows linearly with map size, capping update rate at ~1–10 Hz (`2102.10808.txt:12,599`). The data arrives **sequentially** and each new scan is tiny relative to the map, so rebuilding everything is mostly redundant work (`:13`). The ikd-Tree updates *only with the new points* while keeping the tree query-fast. (In Meridian's layered map, L4, the ikd-Tree is the *registration* substrate — the structure association queries — sitting beneath the NVBlox TSDF surface map and the per-keyframe point store; see the architecture notes. This section is only about the registration layer.)

#### 05.7.2 Node structure

Each node stores the standard k-d fields plus incremental bookkeeping (`ikd_Tree.h:60-81`; paper Data Structure 1, `2102.10808.txt:33-36`):

```cpp
struct KD_TREE_NODE {
    PointType point;                 // the splitting point
    uint8_t   division_axis;         // which of x/y/z this node splits on
    int  TreeSize, invalid_point_num;
    bool point_deleted, tree_deleted;            // lazy labels (this node / whole subtree)
    bool point_downsample_deleted, tree_downsample_deleted;
    bool need_push_down_to_left, need_push_down_to_right;   // lazy-label propagation
    float node_range_x[2], node_range_y[2], node_range_z[2]; // AABB of the subtree
    KD_TREE_NODE *left_son_ptr, *right_son_ptr;
    float alpha_bal, alpha_del;      // balance / deleted ratios for this subtree
};
```

The crucial additions over a textbook k-d tree are: (1) **lazy-delete labels** (`*_deleted`) so deletion is $O(1)$ flag-setting, not structural removal; (2) **`need_push_down_*`** flags that defer propagating a subtree-wide label to children until that subtree is actually visited; (3) the per-node **axis-aligned bounding box** `node_range_*` used to *prune* search; and (4) the subtree **size/invalid counts** that drive re-balancing.

#### 05.7.3 Build and balanced splitting

`Build` (`ikd_Tree.cpp:62-77`) constructs a balanced tree from a point array. At each node it (paper Algorithm 1, `2102.10808.txt:42-76`): chooses the **split axis with maximal coordinate spread/covariance**, sorts on it, takes the **median** as the node point (guaranteeing balance), and recurses on the lower/upper halves. The initial tree in FAST-LIO is built once on the first downsampled scan (`laserMapping.cpp:909-920`):

```cpp
if (ikdtree.Root_Node == nullptr) {           // first usable scan
    ikdtree.set_downsample_param(filter_size_map_min);   // map voxel size
    ... pointBodyToWorld for each ...
    ikdtree.Build(feats_down_world->points);  // balanced k-d tree from scan 1
}
```

#### 05.7.4 Nearest-neighbour search with range pruning

`Nearest_Search(point, k, out, dist, max_dist)` (`ikd_Tree.cpp:79-91`) is an **exact** k-NN (not approximate as FLANN, paper `:237`). It descends the tree like a normal k-d search but uses each subtree's `node_range_*` AABB to *prune*: if the closest possible point in a subtree's box is already farther than the current k-th best, that whole subtree is skipped. The k best are held in a bounded max-heap (`MANUAL_HEAP`, `ikd_Tree.h:93+`). This is the operation §05.3 calls per point, and it is what the whole structure is optimised to keep at $O(\log n)$ (paper `:449`). The `Push_Down` of pending lazy labels is applied before searching a subtree so a query never returns a logically-deleted point (paper `:238`).

#### 05.7.5 Incremental insert + on-tree downsampling

New scan points are added by `Add_Points(PointToAdd, downsample_on)` (`ikd_Tree.cpp:93+`; called from `map_incremental`, `laserMapping.cpp:470-471`). With `downsample_on = true` it performs the paper's **on-tree downsampling** (Algorithm 3, `2102.10808.txt:168-180`): the world is partitioned into cubes of side `filter_size_map_min` (default 0.5 m, set at `:773`/`:913`); for the cube $C_D$ containing the new point it keeps **only the single point closest to the cube centre**. Concretely (`ikd_Tree.cpp:93+`): find $C_D$, box-search the existing points inside it, compare their distance-to-centre with the new point's; if an existing point is already closer, **skip** the insertion; otherwise delete the cube's occupants (`DOWNSAMPLE_DELETE`) and insert the new point. This caps map density at one point per voxel *inside the tree itself* — no separate voxel-grid pass over the whole map.

FAST-LIO's `map_incremental` adds a fast path: points whose nearest map neighbour is already in a *different* voxel cell than the new point's voxel-centre are added **without** the downsample check (`PointNoNeedDownsample`, `laserMapping.cpp:448-449,471`), since they cannot collide with an existing in-cell point — a measurable speedup.

#### 05.7.6 Lazy delete, box-delete, and the sliding local map

**Lazy point delete / re-insert.** Deleting point $P$ just sets its node's `deleted` flag (paper §III-C, `2102.10808.txt:40`); the point stays in the tree (still traversed structurally) but is excluded from query results. If $P$ is needed again it is "re-inserted" in $O(1)$ by clearing the flag (`:78`). Physical removal happens only when a subtree is rebuilt (§05.7.7). This makes both delete and the downsample-driven churn cheap.

**Box-delete and the sliding map.** To bound memory, FAST-LIO keeps only a cube of map around the current LiDAR position and **box-deletes** everything outside it as the platform moves — `lasermap_fov_segment` (`laserMapping.cpp:231-277`). When the sensor approaches within `MOV_THRESHOLD·DET_RANGE` of a face of the local cube (`:251`), the cube is slid by `mov_dist` and the slab left behind is queued in `cub_needrm` and removed in one shot:

```cpp
// laserMapping.cpp:275
if (cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
```

`Delete_Point_Boxes` is the paper's **box-wise delete** (Algorithm 2, `2102.10808.txt:83-159`): recurse from the root, and for each subtree compare its range box $C_T$ (the `node_range_*` AABB) with the operation box $C_O$ — if **disjoint, return immediately** (no work); if $C_T \subseteq C_O$, set the subtree's `tree_deleted` lazily and mark `need_push_down` (a whole sub-map removed in $O(\log n)$, not $O(\text{points})$); if they merely overlap, test the node's own point and recurse. This box-wise sliding window is what makes a *city-scale* run fit in bounded memory at constant per-scan cost.

#### 05.7.7 Self-rebalancing by partial rebuild

Repeated inserts and lazy-deletes degrade balance and leave dead nodes. After each update, ikd-Tree checks two **scapegoat-style criteria** on the touched subtree (paper §III-D, `2102.10808.txt:218-230`; `Update`/`alpha_bal,alpha_del` in `ikd_Tree.cpp` and `ikd_Tree.h:79-80`):

$$
\alpha_{bal}(T) = \frac{\max\big(S(T.\text{left}),\,S(T.\text{right})\big)}{S(T)-1} < \beta_{bal},
\qquad
\alpha_{del}(T) = \frac{I(T)}{S(T)} < \beta_{del},
\tag{05.18}
$$

where $S(T)$ is the subtree size, $I(T)$ its invalid (lazily-deleted) count, $\beta_{bal}\in(0.5,1)$ the balance threshold and $\beta_{del}\in(0,1)$ the deleted-ratio threshold. FAST-LIO's defaults: $\beta_{bal}=0.6$, $\beta_{del}=0.5$ (paper Table II, `:441-444`; header defaults `ikd_Tree.h:213-214`). Violating **either** criterion triggers a **partial rebuild** of just that subtree: flatten it to an array, *drop the lazily-deleted points*, and `Build` a fresh balanced subtree (Algorithm `2102.10808.txt:231`). The $\alpha_{bal}$ criterion caps tree height at $\log_{1/\beta_{bal}} n$ (so search stays $O(\log n)$); the $\alpha_{del}$ criterion reclaims the dead nodes.

**Parallel rebuild.** Rebuilding a *large* subtree would stall the 100 Hz pipeline. If the subtree exceeds `Multi_Thread_Rebuild_Point_Num = 1500` (`ikd_Tree.h:21`; paper $N_{max}=1500$, `:441`) the rebuild is handed to a **second thread** (paper Algorithm 4, `:243-271`): the worker locks *updates but not queries*, flattens valid points, builds the new subtree off to the side while incoming insert/delete requests are recorded in an **operation logger**, replays the logged ops onto the new subtree, then swaps it in under a one-instruction `LockAll`. The main thread keeps answering the §05.3 nearest-neighbour queries the whole time. Net effect (paper Fig. 6, `:604`): nearly constant ~1.6 ms per scan regardless of map size.

#### 05.7.8 The map–residual loop, end to end

Per scan (`laserMapping.cpp` main loop, `:869-977`), the LiDAR factor and its map interact as:

```
sync scan + IMU            (:869)            -> §04 forward-propagate state x̂
lasermap_fov_segment       (:901)            -> slide local cube, box-delete (§05.7.6)
voxel-downsample the scan  (:904-907)        -> feats_down_body
update_iterated_dyn_share  (:960)            -> iterate:  (§09)
    └─ h_share_model       (:638)            ->   for each point:
            (05.1) to world (:657)
            5-NN search     (:670)   ───────────────► ikd-Tree Nearest_Search  (§05.3, §05.7.4)
            esti_plane      (:678)           ->     fit+validate plane         (§05.3)
            residual pd2    (:680)           ->     r_j = n·p + d              (§05.4)
            weight gate s   (:681-683)       ->     accept/reject              (§05.6)
            build H_j row   (:743)           ->     [n  A  B  C]               (§05.5)
map_incremental            (:976)   ───────────────► ikd-Tree Add_Points       (§05.7.5)
```

The map is *read* (k-NN) during the iterated update and *written* (insert + downsample) once after convergence, then *slid* (box-delete) as the platform moves. That read/fit/residual/Jacobian/write cycle is the complete LiDAR factor.

---

### 05.8 Summary and pointers

- The LiDAR factor is a **direct point-to-plane** residual on raw downsampled points — no feature extraction (`feature_extract_enable=false`), which buys accuracy and sensor-agnosticism (FAST-LIO2's thesis, §05.1).
- **Association** = 5-NN search in the map → QR plane fit (`esti_plane`, eq. 05.3–05.4) → 0.1 m planarity gate (§05.3), redone only until the iterate converges (§05.4).
- The **residual** is the signed normal distance $r_j = n^\top{}^{G}p_j + d$ (eq. 05.5; code `pd2`).
- The **Jacobian** row (eq. 05.16) is $[\,n^\top\;A^\top\;B^\top\;C^\top\,]$ with $C=R^\top n$, $A=\lfloor p_I\rfloor_\times C$ (rotation), $B=\lfloor p_L\rfloor_\times {}^{I}_{L}R^\top C$ (extrinsic rotation), $C$ again (extrinsic translation) — derived in §05.5 from the right-perturbation rule (05.9) and matched line-by-line to `laserMapping.cpp:738-743`. Translation Jacobian *is the normal*; the $B,C$ blocks are what enable **online extrinsic calibration**.
- **Weighting** is a constant `LASER_POINT_COV = 0.001` plus a range-adaptive accept gate $s$ (eq. 05.17) — both flagged as Meridian upgrade points toward a principled range/incidence noise model and a GNC robust kernel (§11).
- The **map** is the **ikd-Tree**: balanced build, $O(\log n)$ pruned exact k-NN, $O(1)$ lazy delete, on-tree voxel downsampling, $O(\log n)$ box-delete for the sliding window, and scapegoat partial (optionally parallel) re-balancing — ~4 % of a static tree's cost (§05.7).

**Cross-references.** The probabilistic meaning of "residual + covariance = factor" is §03; the IMU factor that propagates the state we linearise around is §04; the photometric residual that shares this state is §06; GNSS absolute factors §07; how $H,z,R$ here feed Gauss-Newton/LM batch solves §08 and the iterated-EKF recursion FAST-LIO actually runs §09; the continuous-time version (point transformed by a B-spline at its own timestamp) §10; degeneracy/observability and robust kernels that re-weight this factor §11; and the full single-step walkthrough mapped onto FAST-LIO2 §12.


---


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


---


## 07. GNSS & absolute residuals: frames, ENU, switchable

> **Course chapter:** *Tightly-Coupled Multi-Sensor State Estimation & Residuals*
> **This section:** 07 of 12. Builds on §02 (manifolds, $\boxplus/\boxminus$, the
> error state and its Jacobians), §03 (MAP = nonlinear least squares, Gaussians,
> information form), and §04 (the body/IMU state $x$ and its short-term
> propagation). **Feeds:** §08–§09 (how this residual is *solved*, batch and
> recursive), §11 (degeneracy, robust kernels, GNC, PCM — the machinery that
> makes GNSS *safe*), and §12 (one full estimator step end-to-end).
> Frame conventions are normative in [`docs/specs/00_architecture.md`](../../specs/00_architecture.md)
> (the architecture/frames spec); the GNSS message and factor data types are in
> [`docs/specs/01_interfaces_and_data_types.md`](../../specs/01_interfaces_and_data_types.md).

Every residual we have built so far is **relative**. The IMU residual (§04)
ties two consecutive states; the LiDAR point-to-plane residual (§05) ties the
body to a *local* map; the photometric residual (§06) ties the body to a *local*
set of visual points. None of them knows where "north" is, none of them knows
where the robot is on Earth, and all of them **drift**: integrate enough relative
measurements and the estimate slowly walks away from truth, with an error that
grows without bound. This is not a flaw to be fixed by better tuning — it is
intrinsic to dead reckoning. Position error in a LiDAR-inertial system grows
roughly with distance travelled; yaw is the weakest degree of freedom and leaks
the fastest.

GNSS is different *in kind*. It is an **absolute** sensor: it pins the estimate
to a global, Earth-fixed frame with an error that is *bounded* — noisy, sometimes
badly so, but not drifting. One good fix per minute is enough to stop a
LiDAR-inertial system's slow positional and heading drift from accumulating over
a multi-kilometre traverse. That is the prize.

The danger is the mirror image of the prize. Because GNSS is the one sensor that
can *anchor* the whole graph, a single bad fix — multipath off a building, a
stale value held through signal loss, or a deliberate **spoof** — can yank the
entire estimate to a wrong place and corrupt the map with it. For Meridian's
*tactical operational* use this is not a corner case; it is the threat model. So
this section spends as much effort on **gating and switchability** (§7.7–§7.9 —
how a bad fix is prevented from corrupting the estimate) as it does on the
residual itself (§7.4–§7.6).

A note on grounding. The FAST-LIO / FAST-LIVO2 / Point-LIO reference code on
disk does **not** consume GNSS: FAST-LIO's own paper explicitly lists "GPS"
among the sensors it deliberately does *not* fuse
([`slam-reference/papers/2010.08196.txt`](../../../../slam-reference/papers/2010.08196.txt),
the sensor-list in the introduction). So the residual *mechanics* below are
grounded in the shared MAP / iEKF framework that those codebases *do* implement —
the same $\boxplus$-error-state update that FAST-LIO's `esekfom` toolkit runs for
LiDAR (§09) — while the GNSS-specific geometry, frames, and robustness draw on
general SOTA practice and Meridian's own design intent (frames and back-end
placement per [`docs/specs/00_architecture.md`](../../specs/00_architecture.md)).
Where a claim is GNSS-specific and not in the reference code, it is stated as
SOTA practice, not attributed to the code.

---

### 7.1 What a GNSS receiver actually gives you

Be precise about the *output product* of a receiver, because the residual you
write depends entirely on which product you choose to trust.

For each tracked satellite $s$, the receiver observes a **pseudorange**
$\rho^s$ (a biased, noisy range to the satellite), and usually a **carrier phase**
$\phi^s$ and **Doppler** $\dot\rho^s$. From a set of these *raw observables* plus
the broadcast satellite ephemerides, the receiver's internal least-squares solver
produces a **Position–Velocity–Time (PVT)** solution: a geodetic position
$(\varphi,\lambda,h)=(\text{lat},\text{lon},\text{alt})$, often a velocity, a
time, and — crucially — **quality metadata**:

- a **fix type** (no-fix / 2D / 3D / DGPS / RTK-float / RTK-fixed),
- the number of satellites used and their geometry, summarized as **DOP**
  (Dilution Of Precision: GDOP, PDOP, HDOP, VDOP),
- a **reported covariance** $\Sigma_{\text{gnss}}$ (or per-axis standard
  deviations) in a local frame,
- carrier-to-noise density $C/N_0$ per satellite.

This yields a **coupling spectrum**, exactly parallel to the loose/tight coupling
spectrum introduced in §01:

```
 LOOSE  ----------------------------------------------------->  TIGHT
   |                          |                          |
 PVT position fix        PVT pos + vel              raw pseudorange /
 (lat,lon,alt)           (adds Doppler)             carrier phase
 receiver solves         receiver solves            WE solve, per-satellite,
 everything;             pos+vel; we add            inside our estimator,
 we add ONE factor       TWO factors                + receiver clock state
```

- **Loose coupling** (§7.4): take the receiver's PVT *position* (optionally
  velocity) as a single measurement and add **one** absolute factor per fix.
  Simple, robust, and the correct first target. This is Meridian's first-pass plan.
- **Tight coupling** (§7.6): ignore the receiver's PVT and feed the **raw
  pseudoranges** (one residual *per satellite*) directly into the estimator,
  jointly estimating the receiver clock bias. This survives with fewer than four
  satellites and enables per-satellite outlier rejection. Deferred, but specified
  here because it is the natural home of the spoof defence.

§7.4–§7.5 develop the loose case in full; §7.6 shows what changes for tight.

---

### 7.2 Frames: LLA → ECEF → ENU

GNSS lives in global, Earth-fixed coordinates; Meridian's estimator lives in a
**local tangent frame**. We must bridge the two cleanly and *once*. The frame
graph and naming are normative in
[`docs/specs/00_architecture.md`](../../specs/00_architecture.md); we restate the
math here.

#### 7.2.1 The three coordinate systems

1. **Geodetic / LLA** $(\varphi,\lambda,h)$ — latitude $\varphi$, longitude
   $\lambda$, ellipsoidal height $h$. This is what the receiver reports. It is
   *not* Cartesian: you cannot subtract two LLA triples to get a displacement.

2. **ECEF** (Earth-Centred, Earth-Fixed) — a right-handed Cartesian frame with
   origin at Earth's centre of mass, $x$ through the intersection of the equator
   and the prime meridian, $z$ through the (conventional) North pole, $y$
   completing the triad. Rigid to the Earth, rotates with it. WGS-84 datum.

3. **ENU** (East–North–Up) — a *local* Cartesian tangent plane fixed at a chosen
   **anchor** $(\varphi_0,\lambda_0,h_0)$. **This is Meridian's estimator world
   frame $W$** (the `map`/global frame; introduced in §01 as the ENU world).
   Axes: $x=$ East, $y=$ North, $z=$ Up.

```
   LLA (lat,lon,alt)          ECEF (x,y,z)               ENU (E,N,U) = W (map)
   geodetic, on ellipsoid     Cartesian, Earth-centre    local tangent @ anchor
        |                          |                           |
        |  geodetic->Cartesian     |  rotate by (phi0,lam0)    |
        |   (closed form, 7.1)     |  + translate by anchor    |
        +------------------------->+-------------------------->+
                                   <--- both exact & invertible --->
```

#### 7.2.2 LLA → ECEF (WGS-84)

Let the WGS-84 ellipsoid have semi-major axis $a = 6\,378\,137.0\ \mathrm{m}$ and
first-eccentricity-squared $e^2 = 6.694\,379\,990\times 10^{-3}$. The **prime
vertical radius of curvature** at latitude $\varphi$ is

$$
N(\varphi) \;=\; \frac{a}{\sqrt{1 - e^2 \sin^2\varphi}}.
$$

The closed-form geodetic-to-Cartesian map is then

$$
\mathbf{p}_{\text{ecef}}(\varphi,\lambda,h) \;=\;
\begin{bmatrix}
\big(N(\varphi)+h\big)\cos\varphi\cos\lambda \\[2pt]
\big(N(\varphi)+h\big)\cos\varphi\sin\lambda \\[2pt]
\big(N(\varphi)(1-e^2)+h\big)\sin\varphi
\end{bmatrix}.
\tag{7.1}
$$

This is *exact* (no series approximation) and trivially differentiable. The
inverse (ECEF → LLA) has no elementary closed form for $\varphi$; use Bowring's
or Olson's iteration (a few steps to machine precision). We rarely need the
inverse on the hot path — the anchor is computed once and frozen.

#### 7.2.3 ECEF → ENU (the anchor and the rotation)

Fix an **anchor** $(\varphi_0,\lambda_0,h_0)$ with ECEF position
$\mathbf{p}_{\text{ecef}}^0 = \mathbf{p}_{\text{ecef}}(\varphi_0,\lambda_0,h_0)$.
The rotation from ECEF to the local ENU triad depends *only* on the anchor's
longitude and latitude:

$$
R_{\text{enu}\,\text{ecef}}(\varphi_0,\lambda_0) =
\begin{bmatrix}
-\sin\lambda_0 & \cos\lambda_0 & 0 \\
-\sin\varphi_0\cos\lambda_0 & -\sin\varphi_0\sin\lambda_0 & \cos\varphi_0 \\
\cos\varphi_0\cos\lambda_0 & \cos\varphi_0\sin\lambda_0 & \sin\varphi_0
\end{bmatrix}.
\tag{7.2}
$$

The three rows are the East, North, Up unit vectors expressed in ECEF: row 1
(East) lies in the equatorial-tangent direction, row 3 (Up) is the ellipsoidal
normal. A geodetic point's ENU coordinates are

$$
\mathbf{p}_{\text{enu}} \;=\; R_{\text{enu}\,\text{ecef}}
\big(\mathbf{p}_{\text{ecef}} - \mathbf{p}_{\text{ecef}}^0\big).
\tag{7.3}
$$

Composing (7.1)→(7.3) gives a single function $\mathbf{p}_{\text{enu}} =
g(\varphi,\lambda,h)$ that turns any receiver fix into a point in $W$.

> **Discipline — freeze the anchor.** The anchor and its ENU↔ECEF transform are
> computed *once* (at the first good fix, or from a surveyed point) and reused for
> the whole session; do **not** recompute per fix. Re-deriving
> $R_{\text{enu}\,\text{ecef}}$ and $\mathbf{p}^0_{\text{ecef}}$ from each new fix
> injects sub-millimetre but *correlated* jitter into every absolute factor — a
> classic, avoidable, self-inflicted bias. Store the anchor with the map so a
> session can be resumed in the same global frame.

#### 7.2.4 Why ENU and not raw ECEF

Two reasons. **Numerics:** ECEF coordinates are $\sim 6.4\times 10^{6}\ \mathrm{m}$;
representing a robot moving over a $100\ \mathrm{m}$ site as the difference of two
seven-significant-digit numbers squanders `float`/`double` precision exactly
where the estimator needs it. ENU keeps coordinates small and centred near the
operating area. **Gravity & intuition:** in ENU, gravity is (very nearly) $-z$,
which aligns with the IMU gravity state $g$ (§04) and makes the estimator's
bookkeeping — and every rviz visualization — read naturally. Over Meridian's
operational footprint (kilometres, not hundreds of km) the flat-tangent-plane
approximation error is negligible; for continental-scale operation one would
re-anchor or move to a curved local frame, which is out of first-pass scope.

---

### 7.3 The lever arm: the antenna is not the body

The estimator's state $x = (R, p, v, b_g, b_a)$ (§01, §04) describes the **body
frame $B$**, which Meridian takes to be the **IMU frame**. The GNSS antenna's phase
centre is a *different* physical point, rigidly mounted elsewhere on the
platform. The offset is the **lever arm**, and it comes from offline calibration
(the antenna→body extrinsic is *position-only* — an antenna has no orientation).
Denote the lever arm expressed in the body frame as $\mathbf{t}^{B}_{a}$.

To compare a GNSS fix against the body state we must predict where the *antenna*
is, not where the *IMU* is. With $R$ the body (IMU) orientation in $W$ and $p$
the body position in $W$:

$$
\boxed{\;\mathbf{p}_a(x) \;=\; p \;+\; R\,\mathbf{t}^{B}_{a}\;}
\tag{7.4}
$$

The key consequence: **the GNSS position residual depends on orientation $R$, not
just position $p$.** That coupling is what lets a *moving* single antenna weakly
*observe yaw* (§7.4.2); it is also why a careless implementation that *drops* the
lever arm injects an orientation-coupled bias of magnitude $\|\mathbf{t}^B_a\|$
(easily tens of centimetres) into every fix — a bias that rotates with the
robot, so it cannot be calibrated away as a constant offset.

If $\mathbf{t}^B_a$ is itself poorly known, it can be promoted to a graph variable
and refined online — exactly as Meridian treats all extrinsics, and as §11 (online
calibration / observability) develops. For the loose residual below we treat it
as a known constant unless flagged otherwise.

---

### 7.4 The loose GNSS position residual

This is the workhorse: one fix → one factor.

#### 7.4.1 Measurement model and residual

The receiver reports a geodetic position; convert it through §7.2 to a point in
$W$:

$$
\mathbf{z}_{\text{gnss}} \;=\; g(\varphi,\lambda,h)\ \in\ \mathbb{R}^3 \qquad (\text{frame } W=\text{ENU}).
$$

The prediction is the antenna position (7.4). In the MAP / least-squares language
of §03, the **measurement model** is $h(x) = \mathbf{p}_a(x)$ and the residual is
a plain Euclidean difference — position lives in $\mathbb{R}^3$, so no manifold
$\boxminus$ is needed:

$$
\boxed{\;
r_{\text{gnss}}(x) \;=\; \mathbf{p}_a(x) - \mathbf{z}_{\text{gnss}}
\;=\; \big(p + R\,\mathbf{t}^B_a\big) - g(\varphi,\lambda,h)
\;\in\ \mathbb{R}^3.}
\tag{7.5}
$$

Its contribution to the global cost (the $J(x)$ of §03) is the
information-weighted, robustified quadratic

$$
J_{\text{gnss}} \;=\; s\,\rho\!\Big(\big\lVert r_{\text{gnss}}(x)\big\rVert^2_{\Sigma_{\text{gnss}}}\Big),
\qquad
\lVert r\rVert^2_{\Sigma}= r^\top \Omega\, r,\quad \Omega = \Sigma_{\text{gnss}}^{-1},
\tag{7.6}
$$

with $\rho$ a robust kernel (§7.8, §11), $\Omega$ the information matrix (§03),
and $s$ a **switch** — binary $s\in\{0,1\}$ for a hard gate or continuous
$s\in[0,1]$ for a switchable constraint (§7.9). Setting $\rho(\cdot)=(\cdot)$ and
$s=1$ recovers a plain Gaussian factor. The weighting $\Sigma_{\text{gnss}}$ is
seeded from the receiver-reported covariance, *inflated* by fix-type/DOP
heuristics (§7.7) so a marginal fix contributes proportionally less.

#### 7.4.2 The Jacobian

For the solvers of §08 (Gauss–Newton / Levenberg–Marquardt) and §09 (iEKF) we
need $H = \partial r_{\text{gnss}}/\partial\delta x$, where $\delta x$ is the
error state in the tangent space (§02). Only the rotation and position blocks are
non-zero. Using the right SO(3) perturbation $R\leftarrow R\,\mathrm{Exp}(\delta\theta)$
(§02) and the first-order $\mathrm{Exp}(\delta\theta)\approx I + (\delta\theta)^\wedge$:

$$
\mathbf{p}_a\big(R\,\mathrm{Exp}(\delta\theta),\,p+\delta p\big)
\approx p + \delta p + R\big(I+(\delta\theta)^\wedge\big)\mathbf{t}^B_a
= \mathbf{p}_a + \delta p - R\,(\mathbf{t}^B_a)^\wedge\,\delta\theta,
$$

using the identity $a^\wedge b = -\,b^\wedge a$. Hence, ordering the error state
as $\delta x = [\,\delta\theta,\ \delta p,\ \delta v,\ \delta b_g,\ \delta b_a\,]$,

$$
H_{\text{gnss}}
= \frac{\partial r_{\text{gnss}}}{\partial \delta x}
= \big[\;
\underbrace{-\,R\,(\mathbf{t}^B_a)^\wedge}_{\partial / \partial\delta\theta}
\;\;\;
\underbrace{I_{3}}_{\partial / \partial\delta p}
\;\;\; \mathbf{0} \;\;\; \mathbf{0} \;\;\; \mathbf{0}
\;\big] \ \in\ \mathbb{R}^{3\times 15}.
\tag{7.7}
$$

Two things to read off (7.7):

- If the lever arm is **zero**, the rotation block vanishes and the factor
  observes *position only* — pure translation pinning, no heading information.
  This is why a GNSS-only system cannot fix yaw at standstill.
- With a non-zero lever arm and **motion** (changing $R$), the
  $-R(\mathbf{t}^B_a)^\wedge$ term lets a *sequence* of position fixes weakly
  observe yaw — the same geometric effect by which a moving offset antenna acts
  as a crude compass. This couples directly into the observability analysis of
  §11 (it is one of the dimensions that GNSS can render observable).

Structurally this is identical to how the LiDAR point-to-plane Jacobian (§05) and
the IMU residual Jacobian (§04) are assembled and stacked into the normal
equations of §08: the GNSS factor is just another block row contributing
$H^\top\Omega\,H$ to the information matrix and $-H^\top\Omega\,r$ to the
gradient.

#### 7.4.3 Where the factor lives: back-end, not front-end

A subtle but important architectural choice. In FAST-LIO the exteroceptive
update happens inside the high-rate iEKF — the `h_share` measurement model handed
to the `esekfom` toolkit and called from the mapping loop (§05, §09). One *could*
add a GNSS update there too, and §7.5 gives the equations for exactly that,
because the loose residual is a perfectly ordinary iEKF measurement. But Meridian
places the **GNSS factor in the back-end factor graph (GTSAM / iSAM2), in the
`map`/ENU frame**, not in the front-end odometry. Three reasons:

1. **Rate mismatch.** GNSS arrives at 1–10 Hz, far slower than the IMU-rate
   front-end. It belongs with the other slow, global factors — loop closures and
   the like (§11) — anchored on keyframes.
2. **Latency tolerance.** A fix can be matched to the *nearest keyframe by
   PTP-synced timestamp* and added even if it arrives late; iSAM2 re-linearizes
   only the affected sub-graph. A late, out-of-order measurement is awkward in a
   forward-only filter.
3. **Frame separation (REP-105).** The back-end lives in `map`; a GNSS
   correction is exactly the kind of discontinuous global *snap* that the
   `map`→`odom` transform is designed to absorb. The smooth `odom`→`base`
   front-end never jumps, so the operator sees a continuous trajectory even when
   GNSS yanks the global frame. (See the frame graph in
   [`docs/specs/00_architecture.md`](../../specs/00_architecture.md).)

So: the residual (7.5) and Jacobian (7.7) are written once and consumed by the
back-end Gauss–Newton / iSAM2 solver of §08. The next subsection gives the
recursive (iEKF) form for completeness and for the optional case where a single
very high-quality fix is fused at front-end rate.

---

### 7.5 The same residual as a recursive update (iEKF view)

§09 establishes the equivalence between batch MAP / Gauss–Newton and the iterated
EKF. The loose GNSS factor drops into the iEKF measurement update with no new
machinery — worth seeing concretely because it reuses the *exact* error-state
update structure that the reference `esekfom` toolkit implements for LiDAR.

Given the IMU-propagated prior $(\hat x, \hat P)$ (§04), the GNSS measurement
gives innovation $r = r_{\text{gnss}}(\hat x)$ (7.5) and Jacobian
$H = H_{\text{gnss}}$ (7.7). The standard ESIKF / iEKF update (§09) is

$$
S = H \hat P H^\top + \Sigma_{\text{gnss}}, \qquad
K = \hat P H^\top S^{-1},
$$
$$
\delta x = -K\, r, \qquad
\hat x \;\leftarrow\; \hat x \boxplus \delta x, \qquad
\hat P \;\leftarrow\; (I - K H)\,\hat P,
\tag{7.8}
$$

iterated (re-evaluating $r,H$ at the updated $\hat x$) until convergence, with the
$\boxplus$ of §02 applying $\delta\theta$ on $SO(3)$ and the remaining blocks
additively. This is identical in *shape* to the LiDAR update the reference filter
runs; only $h(x)$, $r$, and $H$ change. The robust / switchable handling of
§7.8–§7.9 enters here by *inflating* $\Sigma_{\text{gnss}}$ or by gating: a
rejected fix simply sets $K=0$ (no update), which is exactly $s=0$ in (7.6).

> **Why the back-end is safer anyway.** Because the iEKF marginalizes
> immediately, a bad fix that slips past the gate is hard to *undo* — there is no
> later opportunity to re-weight it. In the smoothing back-end, a subsequent
> contradicting measurement can re-linearize and down-weight the offending factor
> (and PCM, §7.9 / §11, can reject it as part of an inconsistent set). This
> asymmetry is the estimation-theoretic argument, on top of the three
> architectural ones in §7.4.3, for keeping GNSS in the back-end. Reserve the
> iEKF form (7.8) for fixes you trust strongly (RTK-fixed, low DOP).

---

### 7.6 Tight coupling: the raw-pseudorange residual

Loose coupling discards information: it commits to the receiver's least-squares
PVT, which needs $\ge 4$ satellites and degrades silently (or extrapolates) in
urban canyons, under canopy, or under partial jamming. **Tight coupling** keeps
the raw observables and lets the *estimator's own* state — backed by IMU and
LiDAR — help resolve the navigation solution. Meridian defers this (it is real
work), but the residual belongs in this section because it is where spoof /
multipath rejection becomes *per-satellite* rather than all-or-nothing.

#### 7.6.1 The pseudorange model

For satellite $s$ at known ECEF position $\mathbf{x}^s$ (from ephemeris), with the
predicted antenna ECEF position $\mathbf{x}_a = \mathbf{p}^0_{\text{ecef}} +
R_{\text{enu}\,\text{ecef}}^\top\,\mathbf{p}_a(x)$ (invert (7.3) to push the
predicted ENU antenna position back to ECEF), the measured pseudorange is

$$
\rho^s = \big\lVert \mathbf{x}^s - \mathbf{x}_a \big\rVert
\;+\; c\,\delta t_r
\;-\; c\,\delta t^s
\;+\; I^s + T^s + \varepsilon^s,
\tag{7.9}
$$

where $c\,\delta t_r$ is the **receiver clock bias** (a *new state* we must
estimate — identical for all satellites at one epoch), $c\,\delta t^s$ the
satellite clock bias (from ephemeris, known), $I^s, T^s$ the ionospheric /
tropospheric delays (modelled, or differenced away with a base station), and
$\varepsilon^s$ measurement noise plus *multipath*. Augment the state with the
clock bias: $x^+ = (x,\ c\,\delta t_r)$.

#### 7.6.2 The residual and its Jacobian

The per-satellite residual is **scalar**:

$$
\boxed{\;
r^s_{\text{prange}}(x^+) =
\Big(\big\lVert \mathbf{x}^s - \mathbf{x}_a(x)\big\rVert + c\,\delta t_r - c\,\delta t^s + \hat I^s + \hat T^s\Big) - \rho^s
\;\in\ \mathbb{R}.}
\tag{7.10}
$$

Let $\mathbf{u}^s = \dfrac{\mathbf{x}_a - \mathbf{x}^s}{\lVert \mathbf{x}_a - \mathbf{x}^s\rVert}$
be the **line-of-sight unit vector** (from satellite toward antenna). The
position Jacobian is the LOS direction rotated into the ENU state frame, the
clock Jacobian is unity, and the rotation block again enters through the lever
arm:

$$
\frac{\partial r^s}{\partial \delta p} = (\mathbf{u}^s)^\top R_{\text{enu}\,\text{ecef}}, \qquad
\frac{\partial r^s}{\partial \delta\theta} = -\,(\mathbf{u}^s)^\top R_{\text{enu}\,\text{ecef}}\, R\,(\mathbf{t}^B_a)^\wedge, \qquad
\frac{\partial r^s}{\partial (c\,\delta t_r)} = 1.
\tag{7.11}
$$

Each satellite contributes one scalar residual; the shared clock bias couples
them. This is precisely why tight coupling can keep navigating with **fewer than
four** satellites: the IMU/LiDAR-supported state plus the few available LOS
constraints jointly pin position and clock, where the standalone receiver would
report *no fix*.

#### 7.6.3 Why tight coupling helps robustness

Because each satellite is now a *separate* residual, a single multipath-corrupted
or spoofed ray shows up as **one** large residual against a state already
well-constrained by IMU + LiDAR + the other satellites. The robust machinery of
§7.8 / §11 — a per-satellite GNC weight, an M-estimator, or RAIM-style residual
testing — can down-weight *that one ray* while keeping the good ones. That is
impossible in loose coupling, where the receiver has already blended everything
into a single contaminated PVT. This is the deepest reason the SOTA migrates to
tight coupling in adversarial environments, and it is the natural slot for the
spoof defences of §7.7.

---

### 7.7 Multipath and spoofing: the threat, and what to gate on

For tactical use the adversary is *active*. Distinguish three failure modes, in
increasing severity:

1. **Multipath / NLOS** (passive, environmental). The signal reflects off
   buildings or terrain before reaching the antenna, so the pseudorange is *too
   long*. The bias is per-satellite, sporadic, often several metres in urban
   canyons. Symptoms: low $C/N_0$ on the affected satellite, high HDOP, residual
   outliers on a *subset* of rays.
2. **Jamming** (active, denial). Broadband noise raises the floor; the receiver
   loses lock. Symptoms: $C/N_0$ collapse across the board, fix downgrades
   (RTK→3D→2D→no-fix), satellite count drops. The honest failure mode: *no* fix.
3. **Spoofing** (active, deception). The adversary transmits counterfeit signals
   so the receiver computes a *plausible but wrong* PVT — the dangerous case,
   because the fix *looks healthy*. Symptoms are subtle: a fix that is internally
   self-consistent yet **inconsistent with the IMU/LiDAR-propagated state**, an
   abnormally *clean* and uniform $C/N_0$, a sudden position jump, or a slow,
   deliberate "walk-off" drag.

The defence is **layered**, and each layer uses the information available at a
different stage:

- **Receiver-reported quality (cheap, first line).** Reject or inflate on: fix
  type below threshold, satellite count below a floor, DOP above a ceiling, and
  abnormal $C/N_0$ patterns. Catches jamming and gross multipath; a competent
  spoofer can fake all of it.
- **Innovation consistency against the fused state (the real defence).** The one
  test that catches a *healthy-looking* spoof — and it is available *only*
  because we hold a tightly-coupled IMU+LiDAR estimate to test against.
  Formalized in §7.8.
- **Per-satellite RAIM (tight coupling only, §7.6).** Statistical testing of
  individual pseudorange residuals to identify and exclude a faulty / spoofed
  satellite; requires redundancy (more satellites than the minimum).

The guiding principle for Meridian: **GNSS is never trusted unconditionally; it is
admitted only insofar as it agrees with the dead-reckoned estimate it is meant to
correct.** A self-consistent but estimate-contradicting fix is the signature of a
spoof, and the gate below is built precisely to refuse it.

---

### 7.8 The consistency gate: normalized innovation (NIS / Mahalanobis test)

The mathematically principled gate is the **innovation test** familiar from §09,
generalized here to the smoothing back-end. Before admitting a fix, evaluate the
innovation $r = r_{\text{gnss}}(\hat x)$ (7.5) against the current best estimate
$\hat x$ (with prior covariance $\hat P$ from IMU/LiDAR propagation), and form the
**innovation covariance** and the **Normalized Innovation Squared**:

$$
S = H \hat P H^\top + \Sigma_{\text{gnss}}, \qquad
\chi^2 \;=\; r^\top S^{-1} r .
\tag{7.12}
$$

Under the null hypothesis (the fix is good and the models correct),
$\chi^2 \sim \chi^2_d$ with $d=3$ degrees of freedom (3-D position) — or $d=1$
per satellite in the tight case. Pick a confidence level (e.g. 99%) giving a
threshold $\gamma$ ($\chi^2_{3,\,0.99}\approx 11.34$):

$$
\text{admit if } \chi^2 \le \gamma; \qquad
\text{reject (or down-weight) if } \chi^2 > \gamma.
\tag{7.13}
$$

This single test does the heavy lifting against spoofing: a counterfeit fix may
be internally self-consistent, but unless the spoofer also knows *and matches*
the robot's true IMU/LiDAR-tracked trajectory, the *innovation* against the fused
estimate blows past $\gamma$. The IMU's bounded short-term drift (§04) gives a
tight, trustworthy $\hat P$ over the seconds between fixes — exactly the window a
walk-off spoof tries to exploit, and exactly the window in which inertial dead
reckoning is most reliable.

**Hard gate vs. soft down-weight.** Equation (7.13) is a *hard* binary gate
evaluated on a *linearization*, before the solver runs. A softer, usually
better-behaved alternative lets the gate **inflate the covariance** instead of
discarding — scaling $\Sigma_{\text{gnss}}$ by a factor that grows with $\chi^2$;
or, equivalently, applying a robust kernel $\rho$ (§11) so the cost (7.6)
*saturates* for large residuals. The robust-kernel view writes the factor cost as

$$
J_{\text{gnss}} = s\,\rho\!\big(\lVert r_{\text{gnss}}\rVert_{\Sigma}\big),
$$

with $\rho$ a Huber or a redescending (Geman–McClure / Cauchy) kernel (§11). A
Huber kernel grows linearly past its threshold (bounded influence, but keeps some
pull); a redescending kernel drives the weight toward *zero* for gross outliers
(a spoof gets fully ignored). For the adversarial setting a redescending kernel,
ramped in by **GNC** (Graduated Non-Convexity, §11 — start convex to find a good
basin, then sharpen so true outliers are rejected without local-minimum risk), is
the SOTA choice. Gate and kernel are complementary: the $\chi^2$ gate is a fast
first cut on the obviously bad; the kernel handles the marginal cases gracefully
*inside* the optimizer.

---

### 7.9 Switchable constraints: the principled "off switch"

The hard gate (7.13) makes a binary, non-differentiable decision *before* the
solver runs, on a *linearization* of the residual. **Switchable constraints**
(Sünderhauf & Protzel) push that decision *inside* the optimization and make it
**continuous and self-adjusting**, so the optimizer itself decides how much to
trust each GNSS factor, jointly with the trajectory.

Augment the state with a **switch variable** $s\in\mathbb{R}$ (mapped through a
sigmoid or clamp to $[0,1]$) *per GNSS factor*. The factor cost becomes

$$
\boxed{\;
J_{\text{gnss}} = \big\lVert \Psi(s)\, r_{\text{gnss}}(x) \big\rVert^2_{\Sigma_{\text{gnss}}}
\;+\; \big\lVert\, s - s_{\text{prior}} \,\big\rVert^2_{\Sigma_s}\;}
\tag{7.14}
$$

where $\Psi(s)$ is a switch function (commonly $\Psi(s)=s$ clamped to $[0,1]$, or
a sigmoid) that scales the residual, and the **second term is a prior pulling $s$
toward "on"** ($s_{\text{prior}}=1$) with strength $\Sigma_s$. The mechanism is an
elegant tug-of-war:

- If the GNSS factor *agrees* with the rest of the graph, keeping $s\approx 1$
  costs nothing in the data term and satisfies the prior → the switch stays on.
- If the GNSS factor *fights* the graph, the optimizer can **turn it down**
  ($s\to 0$), paying only the *bounded* prior penalty
  $\lVert s-1\rVert^2_{\Sigma_s}$ while *escaping* the *unbounded* data-term
  penalty → a bad fix de-weights itself, smoothly and automatically.

$\Sigma_s$ tunes the credulity: a tight $\Sigma_s$ makes the system reluctant to
disable a fix (trust GNSS); a loose $\Sigma_s$ makes it quick to disable
(suspicious of GNSS — the tactical default). The Jacobians extend (7.7) with a
column $\partial r/\partial s = \Psi'(s)\,r_{\text{gnss}}(x)$ plus the trivial
prior block, and the whole thing drops straight into the Gauss–Newton normal
equations of §08 / iSAM2.

**Relationship to the other tools.** Switchable constraints, robust kernels, and
the $\chi^2$ gate are not rivals — they are the same idea at different commitment
levels. A switchable constraint is provably equivalent to optimizing a particular
robust kernel (the Black–Rangarajan duality: the switch $s$ *is* the IRLS weight
of that kernel's $\rho$). Meridian's layered stance, consistent with §11:

```
  cheap, pre-solve          continuous, in-solver           global, multi-factor
  +-----------------+       +-----------------------+        +--------------------+
  | receiver flags  |  -->  | switchable constraint |  -->   | PCM: reject the    |
  | + chi^2 gate    |       | + robust kernel (GNC) |        | INCONSISTENT SET   |
  | (§7.7, §7.8)    |       | (§7.9, §11)           |        | (§11, place rec.)  |
  +-----------------+       +-----------------------+        +--------------------+
```

**PCM** (Pairwise Consistency Maximization, §11 and the L5 place-recognition
stack) is the last line of defence. Where switchable constraints judge each GNSS
factor in *isolation*, PCM finds the largest *mutually consistent* set of
absolute / loop constraints and rejects the rest as a group — defeating a
coordinated spoof that injects several mutually-consistent-but-globally-wrong
fixes. GNSS factors are first-class citizens of that consistency graph alongside
loop closures.

---

### 7.10 GNSS-derived heading and velocity (briefly)

Two absolute residuals beyond position deserve mention; both follow the same
template and reuse the notation above.

- **Doppler / GNSS velocity.** Many receivers report a velocity from Doppler,
  which is *more robust to multipath than position* (Doppler is harder to spoof
  consistently). The residual is
  $r_v = v + R\,(\boldsymbol\omega^\wedge \mathbf{t}^B_a) - \mathbf{z}_v$ — body
  velocity plus the lever-arm rotational term, against the reported ENU velocity,
  with $\boldsymbol\omega$ the bias-corrected body angular rate (§04). It directly
  observes the velocity state $v$ and is a cheap, valuable second factor.

- **Dual-antenna / moving-baseline heading.** Two antennas give an absolute *yaw*
  (and pitch) from their baseline vector, observing $R$ *directly* rather than
  weakly through a single lever arm (§7.4.2). The residual compares the predicted
  baseline direction $R\,\mathbf{b}^B$ against the measured ENU baseline. This is
  the clean way to bound the one degree of freedom — heading — that a
  single-antenna LiDAR-inertial system is otherwise weakest at, and it is a strong
  spoof-resistance asset, since faking a *consistent baseline across two antennas*
  is far harder than faking a single position.

Both are out of first-pass scope but slot onto the same back-end with the same
gating / switching machinery; they are flagged here so the interfaces leave room
for them.

---

### 7.11 Putting it together: the GNSS factor lifecycle

End to end, what happens when a fix arrives (the concrete contract the L0/L2/L3
modules implement; cf. the full estimator step in §12):

```
1. RECEIVE   raw fix (lat,lon,alt, fix-type, #sats, DOP, C/N0, cov), PTP-stamped
2. QUALITY   reject / flag on fix-type, #sats, DOP, C/N0   ........... §7.7
3. PROJECT   LLA -> ECEF -> ENU via the FROZEN anchor      ........... §7.2  (7.1)-(7.3)
4. ASSOCIATE match to nearest keyframe by PTP timestamp    ........... back-end / time-sync
5. PREDICT   antenna position  p_a(x) = p + R t^B_a        ........... §7.3  (7.4)
6. RESIDUAL  r = p_a(x) - z_gnss ;  build Jacobian H       ........... §7.4  (7.5),(7.7)
7. GATE      chi^2 = r^T S^-1 r   vs threshold gamma       ........... §7.8  (7.12)-(7.13)
8. ADD       switchable constraint + robust kernel into    ........... §7.9  (7.14)
             the GTSAM / iSAM2 graph (map / ENU frame)            §7.4.3
9. SOLVE     back-end re-optimizes; map->odom snaps         .......... §08, frame graph
10. DEBUG    publish residual, chi^2, switch value s, fix-type, ...... cross-cutting
             #sats, covariance ellipse as rviz markers + topics       (debug principle)
```

Step 10 is non-negotiable under Meridian's engineering principles: the operator and
developer must be able to *see* the GNSS subsystem's decisions — the live
$\chi^2$, the per-factor switch values $s$, which fixes were admitted vs.
rejected, and the covariance ellipse in rviz — so a spoof or a multipath storm is
*visible*, not silent. A GNSS factor that quietly de-weights itself ($s\to 0$)
while the operator watches is exactly the introspection the system is built to
provide.

---

### 7.12 Key takeaways

- GNSS is the only **absolute, non-drifting** sensor in Meridian; one good fix per
  minute bounds the LiDAR-inertial drift of §04–§06 over long traverses.
- It is also the most **dangerous** sensor: a single bad / spoofed fix can corrupt
  the whole graph, so gating and switchability matter as much as the residual.
- Frames: **LLA → ECEF → ENU** (eqs. 7.1–7.3) with a **frozen anchor**; ENU is
  the estimator world frame $W$ / `map` (architecture spec).
- The **lever arm** (7.4) makes the position residual depend on $R$, weakly
  observing yaw under motion; never drop it.
- **Loose** coupling = one position factor per fix (7.5), Jacobian (7.7), placed
  in the **back-end** graph (§7.4.3) — the correct first target.
- **Tight** coupling = per-satellite pseudorange residual (7.10) with a clock
  state, surviving < 4 satellites and enabling per-ray outlier rejection
  (deferred, but the natural home of the spoof defence).
- Robustness is **layered**: receiver flags → $\chi^2$ consistency gate
  (7.12–7.13) → switchable constraint + robust kernel / GNC (7.14) → PCM — the
  same defence-in-depth philosophy as the loop-closure robustness of §11.
- Everything is **introspectable**: residual, $\chi^2$, switch value, and
  covariance are first-class debug outputs.

> **Next:** §08 shows how this factor — and all the others — is *solved* in the
> batch normal equations (sparsity, Schur complement); §09 gives the recursive
> (iEKF) equivalent used at front-end rate; §11 develops the robust kernels, GNC,
> and PCM this section leans on; §12 walks one full tightly-coupled estimator step
> end to end.


---


## 08. Solving I (batch): GN, LM, normal equations, sparsity, Schur

> **Where we are.** Section 03 turned the tightly-coupled estimation problem into a single
> *maximum-a-posteriori* (MAP) objective — a sum of squared, information-weighted residuals
> over a factor graph. Sections 04–07 wrote down the individual residuals: the inertial
> residual $r_\mathrm{IMU}$ (§04), the LiDAR point-to-plane residual $r_\mathrm{L}$ (§05), the
> sparse-direct photometric residual $r_\mathrm{cam}$ (§06), and the GNSS / absolute residual
> $r_\mathrm{G}$ (§07). This section answers the mechanical question that all of those defer to:
> **given a stack of residuals $r(x)$ and their Jacobians $H$, how do we actually compute the
> state update $\delta x$?** We derive Gauss–Newton (GN) and Levenberg–Marquardt (LM), assemble
> the normal equations $H\,\delta x = -b$, and show *precisely* how the joint $H$ couples every
> sensor — this coupling is the mechanical essence of the word "tight." We then study the
> sparsity of the SLAM Jacobian and the Schur complement / marginalization that make the solve
> real-time. Section 09 takes the *recursive* view (iterated EKF / fixed-lag smoothing) and
> proves it is the *same* computation as the batch GN step derived here; §10 lifts everything
> to continuous time; §12 maps the whole machine onto FAST-LIO2 / FAST-LIVO2 concretely.

---

### 08.1 The problem we are solving

Recall from §03 that MAP estimation of the state $x$ (living on the manifold
$\mathcal{M} = SO(3) \times \mathbb{R}^n$, with the $\boxplus/\boxminus$ operators of §02) is

$$
\hat{x} \;=\; \arg\min_{x}\; C(x), \qquad
C(x) \;=\; \tfrac{1}{2}\sum_{i} \big\| r_i(x) \big\|^2_{\Omega_i}
\;=\; \tfrac{1}{2}\sum_{i} r_i(x)^\top \,\Omega_i\, r_i(x),
\tag{08.1}
$$

where $r_i(x) = z_i - h_i(x)$ is the residual of factor $i$ (measurement $z_i$, prediction
$h_i$), and $\Omega_i = \Sigma_i^{-1}$ is its information matrix (§03). The factors $i$ run over
*all* sensors and the prior simultaneously. There is no separate "fuse the LiDAR pose with the
IMU" step — the LiDAR raw points, the IMU preintegrant, the photometric patches and the GNSS
fix all push on the *same* $x$ through the *same* cost. That single shared $x$ is what "tightly
coupled" means; everything in this section is the consequence of writing one $C(x)$ instead of
many.

Two features make (08.1) non-trivial and force the machinery below:

1. **It is nonlinear.** $h_i$ contains rotations, projections $\pi$, plane distances — so we
   cannot solve in one shot; we iterate, linearizing at the current estimate.
2. **$x$ lives on a manifold.** We cannot add a correction vectorially; we use the retraction
   $x \boxplus \delta x$ of §02. The unknown we solve for at each iteration is the *minimal*
   tangent-space increment $\delta x \in \mathbb{R}^n$, not a new global $x$.

> **Notational bridge to the code.** FAST-LIO's filter formulation in
> `papers/2010.08196.txt` writes the *same* objective as eq. (17):
> $\min_{\delta x}\; \|x \boxminus \hat{x}\|^2_{P^{-1}} + \sum_j \|z_j + H_j\,\delta x\|^2_{R_j^{-1}}$.
> The first term is the IMU-propagated prior (information $P^{-1}$); the second is the LiDAR
> measurement sum. This is exactly (08.1) with the prior treated as one more Gaussian factor.
> §09 shows the iterated-EKF update is the recursive solution of this; here we solve it as a
> batch least-squares problem. Both routes produce *the same* $\delta x$ — this is the
> IKF↔GN equivalence the project brief asks us to ground, and we prove it in §08.7.

---

### 08.2 Linearization on the manifold

Fix the current iterate $x$ (the operating point). Perturb it minimally by
$\delta x \in \mathbb{R}^n$ via the retraction of §02,

$$
x' \;=\; x \boxplus \delta x .
\tag{08.2}
$$

Each residual, viewed as a function of $\delta x$, is approximated to first order:

$$
r_i(x \boxplus \delta x) \;\approx\; r_i(x) \;+\; J_i\,\delta x,
\qquad
J_i \;\triangleq\; \left.\frac{\partial\, r_i(x \boxplus \delta x)}{\partial\, \delta x}\right|_{\delta x = 0}
\;\in\; \mathbb{R}^{m_i \times n}.
\tag{08.3}
$$

$J_i$ is the residual Jacobian **with respect to the tangent-space increment**, evaluated at
$\delta x = 0$. Crucially it bakes in the manifold structure: where $x$ contains a rotation $R$,
the relevant block of $J_i$ already includes the right-Jacobian / $A(\cdot)$ correction from
§02 (the term that turns a naïve $\partial/\partial R$ into a derivative w.r.t. the tangent
vector). We will write $J_i$ for "the residual Jacobian" and reserve $H$ for the *measurement
model* Jacobian $\partial h_i/\partial \delta x = -J_i$ (sign only); the literature is sloppy
about the sign, so we are explicit. FAST-LIO eq. (14) defines exactly this $H_j$ as "the
Jacobian of $h_j(x \boxplus \delta x, {}^{L_j}n_{f_j})$ w.r.t. $\delta x$, evaluated at zero."

Substituting (08.3) into (08.1) gives a *quadratic* model of the cost around $x$:

$$
C(x \boxplus \delta x) \;\approx\;
\tfrac{1}{2}\sum_i \big( r_i + J_i\,\delta x \big)^\top \Omega_i \big( r_i + J_i\,\delta x \big)
\;=\;
\tfrac{1}{2}\,\delta x^\top H\,\delta x \;+\; b^\top \delta x \;+\; \text{const},
\tag{08.4}
$$

with the two central objects of this whole section:

$$
\boxed{\;
H \;=\; \sum_i J_i^\top\, \Omega_i\, J_i,
\qquad
b \;=\; \sum_i J_i^\top\, \Omega_i\, r_i .
\;}
\tag{08.5}
$$

$H \in \mathbb{R}^{n\times n}$ is the **(approximate) Hessian** — also called the *information
matrix* of the linearized problem, written $\Lambda$ or $A$ in some texts — and
$b \in \mathbb{R}^{n}$ is the **gradient** (negated right-hand side). Note $H$ is symmetric
positive semidefinite by construction: it is a sum of $J_i^\top \Omega_i J_i$, each a
congruence transform of a PSD information $\Omega_i$.

---

### 08.3 Gauss–Newton: the normal equations

Minimizing the quadratic model (08.4) over $\delta x$ is now a linear problem. Set the gradient
of (08.4) to zero, $\nabla_{\delta x} = H\,\delta x + b = 0$, yielding the **normal equations**:

$$
\boxed{\;\; H\,\delta x \;=\; -\,b, \qquad
\Big(\textstyle\sum_i J_i^\top \Omega_i J_i\Big)\,\delta x \;=\; -\sum_i J_i^\top \Omega_i\, r_i. \;\;}
\tag{08.6}
$$

The Gauss–Newton algorithm iterates: linearize $\to$ build $(H,b)$ $\to$ solve (08.6) $\to$
**retract** $x \leftarrow x \boxplus \delta x$ $\to$ repeat until $\|\delta x\|$ is below the
convergence threshold.

```
Algorithm GN-on-manifold(x0):
  x <- x0
  repeat:
     for each factor i:                    # build the system
        compute r_i(x)  and  J_i           # §04-§07 supply these
        accumulate  H += J_i^T Omega_i J_i
        accumulate  b += J_i^T Omega_i r_i
     solve   H dx = -b                      # §08.5 Cholesky / Schur
     x <- x boxplus dx                      # §02 retraction
  until ||dx|| < eps
```

Three remarks make GN the workhorse of SLAM:

- **It is Newton without the second-derivative term.** The true Hessian of (08.1) is
  $\nabla^2 C = \sum_i \big(J_i^\top\Omega_i J_i + \sum_k (\Omega_i r_i)_k \nabla^2 r_{i,k}\big)$.
  GN drops the second sum (the curvature of the residuals themselves). This is exact when
  residuals are linear, and an excellent approximation when residuals are *small* near the
  optimum (the case for a well-tracked SLAM system) — which is why GN converges quadratically
  there and needs no Hessians.
- **$H$ is built by accumulation.** Each factor contributes additively to $H$ and $b$. This is
  what lets a tightly-coupled estimator simply *append* a sensor: add its factors to the loop.
  Meridian's solver should expose exactly this accumulate API behind `IFactor`.
- **The information weighting $\Omega_i$ is where robustness and observability enter.** §11
  inflates $\Omega_i$ along degenerate directions (X-ICP / D2-LIO style) and wraps $r_i$ in a
  robust kernel; both act *only* by reshaping the per-factor $\Omega_i$ and $r_i$ before the
  accumulation in (08.5). The solver itself never changes.

#### The least-squares / "stacked" view

It is often cleaner to stack all residuals into one tall vector. Whiten each factor by its
square-root information $\Omega_i = L_i L_i^\top$ (Cholesky), define
$\tilde{r}_i = L_i^\top r_i$, $\tilde{J}_i = L_i^\top J_i$, and stack:

$$
\tilde{r} = \begin{bmatrix}\tilde r_1\\ \vdots \\ \tilde r_N\end{bmatrix},
\quad
\tilde J = \begin{bmatrix}\tilde J_1\\ \vdots \\ \tilde J_N\end{bmatrix},
\qquad\Longrightarrow\qquad
C \approx \tfrac12 \|\tilde r + \tilde J\,\delta x\|_2^2 .
\tag{08.7}
$$

Then $H = \tilde J^\top \tilde J$ and $b = \tilde J^\top \tilde r$, and the normal equations
are the classic $\tilde J^\top \tilde J\,\delta x = -\tilde J^\top \tilde r$. Numerically one
prefers to **never form $H$**: a thin QR of $\tilde J = QR$ gives $R\,\delta x = -Q^\top \tilde r$
with half the condition number ($\kappa(H) = \kappa(\tilde J)^2$). GTSAM (the §03 back-end)
solves keyframe graphs this way. For the dense, small-state front-end of §09, forming $H$ and
doing a Cholesky is fine and faster.

---

### 08.4 The joint $H$: the mechanical essence of "tight"

This is the conceptual heart of the section. Write the state in blocks. For a LiDAR–IMU front
end the natural blocks are the navigation state $x_{\mathrm{nav}} = (R, p, v)$, the IMU biases
$x_{\mathrm{bias}} = (b_g, b_a)$, gravity $g$, and (if we online-calibrate, §00/§11) the
LiDAR–IMU extrinsic $x_{\mathrm{ext}} = ({}^{I}R_L, {}^{I}p_L)$. This is exactly the state
manifold FAST-LIO declares in `FAST_LIO/include/use-ikfom.hpp:12-21`:

```cpp
MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos)) ((SO3, rot))
((SO3, offset_R_L_I)) ((vect3, offset_T_L_I))   // online LiDAR-IMU extrinsic
((vect3, vel)) ((vect3, bg)) ((vect3, ba))
((S2, grav)));                                  // gravity on the 2-sphere S2
```

Now look at how each sensor's factor touches these blocks:

| Factor | $r_i$ depends on (nonzero $J_i$ columns) |
|---|---|
| IMU preintegration (§04) | $R, p, v$ at *two* times, plus $b_g, b_a, g$ |
| LiDAR point-to-plane (§05) | $R, p$ (and $x_{\mathrm{ext}}$ if online-calibrating) |
| Visual photometric (§06) | $R, p$ (and camera extrinsic / intrinsics) |
| GNSS position (§07) | $p$ (and the ENU↔body alignment) |

The joint Hessian (08.5) is the **sum** of the per-factor outer products. Because the IMU
factor's $J$ has nonzero columns in $R,p,v,b_g,b_a,g$ *and* the LiDAR factor's $J$ has nonzero
columns in $R,p$, their outer products **overlap** in the $(R,p)$ block. The result is a single
$H$ with *cross terms* linking, e.g., the gyro bias $b_g$ to the LiDAR-observed pose:

$$
H \;=\;
\underbrace{\begin{bmatrix}
H^{\mathrm{IMU}}_{RR} & H^{\mathrm{IMU}}_{Rp} & H^{\mathrm{IMU}}_{Rv} & H^{\mathrm{IMU}}_{Rb} & \cdots\\
H^{\mathrm{IMU}}_{pR} & H^{\mathrm{IMU}}_{pp} & \cdots & & \\
\vdots & & \ddots & & \\
\end{bmatrix}}_{\text{from IMU + prior}}
\;+\;
\underbrace{\begin{bmatrix}
H^{\mathrm{L}}_{RR} & H^{\mathrm{L}}_{Rp} & 0 & 0 & \cdots\\
H^{\mathrm{L}}_{pR} & H^{\mathrm{L}}_{pp} & 0 & 0 & \cdots\\
0 & 0 & 0 & 0 & \\
\vdots & & & & \\
\end{bmatrix}}_{\text{from all LiDAR points}} .
\tag{08.8}
$$

The $(R,p)$ sub-block receives contributions from *both* sensors. Solving $H\,\delta x = -b$ on
this combined $H$ is what lets a LiDAR point-to-plane error *correct the IMU bias estimate*, and
lets the IMU prior *regularize a geometrically degenerate LiDAR scan* (e.g. a long featureless
corridor where $H^{\mathrm{L}}$ is rank-deficient along the corridor axis, but $H^{\mathrm{IMU}}$
keeps that direction observable). A loosely-coupled system would instead compute a LiDAR pose by
inverting $H^{\mathrm{L}}$ alone (which is singular in the corridor!) and only then fuse it —
losing exactly the cross-coupling that (08.8) preserves. *That* off-diagonal block is "tight."

#### Worked block structure for the LiDAR point-to-plane factor

Take one LiDAR feature point ${}^{L}p_f$ and its associated map plane $(n, q)$ from §05. With
the global point ${}^{G}p_f = {}^{G}R_I\,({}^{I}R_L\,{}^{L}p_f + {}^{I}p_L) + {}^{G}p_I$ (§05
eq. for the transform; FAST-LIO eq. 11), the scalar residual is the point-to-plane distance

$$
r \;=\; n^\top\big({}^{G}p_f - q\big) \;\in\; \mathbb{R} .
\tag{08.9}
$$

(FAST-LIO eq. 12: $z_j = {}^{G}u_j^\top({}^{G}p_{f_j} - {}^{G}q_j)$ with ${}^{G}u_j = u_j^\top$
for a plane.) Differentiating w.r.t. the tangent increment (using $\delta R$: $R \boxplus
\delta\theta = R\,\mathrm{Exp}(\delta\theta)$, so $\partial(R a)/\partial\delta\theta =
-R\,(a)^\wedge$ from §02), the two nonzero $1\times 3$ blocks are

$$
\frac{\partial r}{\partial \delta p} = n^\top,
\qquad
\frac{\partial r}{\partial \delta\theta} = -\,n^\top\, {}^{G}R_I\,\big({}^{I}R_L\,{}^{L}p_f + {}^{I}p_L\big)^{\!\wedge}
= -\,n^\top\,\big({}^{G}R_I\, {}^{I}p_f \big)^{\!\wedge},
\tag{08.10}
$$

so $J_{\mathrm{L}} = [\,\partial r/\partial\delta\theta \;\; \partial r/\partial\delta p\;\;
0 \cdots 0\,]$ — a single $1\times n$ row, nonzero only in the rotation and position columns.
This is literally what FAST-LIO assembles in `laserMapping.cpp::h_share_model`: for each
surviving correspondence it stores the plane normal in `corr_normvect` and the source point in
`laserCloudOri` (`laserMapping.cpp:701-702`), then fills one $12$-column row of the measurement
Jacobian `ekfom_data.h_x` (`laserMapping.cpp:743`):

```cpp
// laserMapping.cpp:738-743
V3D C(s.rot.conjugate() * norm_vec);          // normal in body frame
V3D A(point_crossmat * C);                     // d(r)/d(rot)  = (R^T n) x p   (3)
V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);   // d(r)/d(extrinsic rot) (3)
ekfom_data.h_x.block<1,12>(i,0) << norm_p.x, norm_p.y, norm_p.z,   // d(r)/d(pos) = n^T  (3)
                                   VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
ekfom_data.h(i) = -norm_p.intensity;           // the residual r = -(n.x + d)  (08.9)
```

So the $12$ columns are laid out `[ d(r)/d(pos) (3) | d(r)/d(rot) (3) | d(r)/d(ext_R) (3) |
d(r)/d(ext_p) (3) ]` — the first three are exactly $n^\top$ of (08.10), and `A` is the
rotation block $-n^\top({}^{G}R_I\,{}^{I}p_f)^\wedge$ written with the equivalent body-frame
cross-product form $({}^{I}R_L\,{}^{L}p_f + {}^{I}p_L)\times({}^{G}R_I^\top n)$. The extrinsic
columns (`B`, `C`) are filled only when `extrinsic_est_en` is set (`laserMapping.cpp:740-748`) —
the online-calibration switch of §00/§11; otherwise they are zeroed. With $m \approx 1000$
effective points, $J_{\mathrm{L}}$ is a $1000 \times n$ matrix that is dense in only a handful of
columns; (08.5) collapses it to the small $H$ block of (08.8).

For one such point the rank-1 contribution to $H$ is

$$
J_{\mathrm{L}}^\top \,\tfrac{1}{\sigma^2}\, J_{\mathrm{L}}
= \frac{1}{\sigma^2}
\begin{bmatrix} a \\ n \\ \vdots \end{bmatrix}
\begin{bmatrix} a^\top & n^\top & \cdots \end{bmatrix},
\qquad a = -\big({}^{G}R_I\,{}^{I}p_f\big)^{\!\wedge\top} n .
\tag{08.11}
$$

A single point is **rank 1** — it constrains only the direction $n$. You need points with
*independent* normals to make the $(R,p)$ block full rank; a planar scene (all $n$ parallel)
leaves the in-plane translation and two rotations unobservable. This rank deficiency *is* LiDAR
degeneracy, and it appears directly as a near-singular $H$ — §11 detects it by inspecting the
eigenvalues of this very $H$ block.

---

### 08.5 Sparsity of the SLAM Jacobian, and Schur

#### Two regimes of sparsity

The structure of $H$ differs between the two layers of Meridian (§03):

- **Front-end (§09), small dense state.** The state is *one* navigation state (plus biases /
  extrinsic): $n \approx 18$–$24$. Thousands of LiDAR points all touch the *same* state, so $H$
  is **small and dense** ($\sim 24\times 24$). Here there is nothing to exploit by sparsity —
  but there is a huge win from the *measurement* dimension being enormous ($m \gg n$). That is
  the FAST-LIO trick of §08.6.

- **Back-end (§03, GTSAM iSAM2), large sparse state.** The state is a *trajectory* of keyframe
  poses $x_1,\dots,x_K$ plus landmarks. A factor touches only the few variables it involves (an
  odometry factor: two consecutive poses; a loop factor: two distant poses; a GNSS factor: one
  pose). So $H$ is **large and sparse**, with a banded-plus-loop-arc structure:

$$
H \;=\;
\begin{bmatrix}
\bullet & \bullet & & & \bullet\\
\bullet & \bullet & \bullet & & \\
        & \bullet & \bullet & \bullet & \\
        &         & \bullet & \bullet & \bullet\\
\bullet &         &         & \bullet & \bullet
\end{bmatrix}
\quad
\substack{\text{tridiagonal from odometry}\\ \text{+ off-diagonal arcs from loop closures}}
\tag{08.12}
$$

A loop closure between pose 1 and pose 5 is the corner $\bullet$ — exactly why loop closure is
expensive: it fills in otherwise-zero blocks. iSAM2 keeps a *factorization* (the Bayes tree)
and only re-eliminates the variables whose part of the tree the new factor touches; that is the
recursive analogue of resolving (08.6) and is detailed in §03/§09. Sparse Cholesky with a
fill-reducing ordering (COLAMD/AMD) is the batch counterpart.

#### Schur complement / marginalization

The Schur complement is the single most important algebraic tool for real-time solving. Split
the state into a block we want to keep, $\delta x_a$, and a block we want to eliminate (solve out
or marginalize), $\delta x_b$. The normal equations partition as

$$
\begin{bmatrix} H_{aa} & H_{ab}\\ H_{ba} & H_{bb}\end{bmatrix}
\begin{bmatrix}\delta x_a\\ \delta x_b\end{bmatrix}
=
-\begin{bmatrix} b_a\\ b_b\end{bmatrix}.
\tag{08.13}
$$

From the second block row, $\delta x_b = -H_{bb}^{-1}(b_b + H_{ba}\delta x_a)$. Substitute into
the first row to get a smaller, *self-contained* system in $\delta x_a$ alone:

$$
\boxed{\;
\underbrace{\big(H_{aa} - H_{ab}H_{bb}^{-1}H_{ba}\big)}_{\textstyle S \;=\; H/H_{bb}}\,\delta x_a
\;=\;
-\big(b_a - H_{ab}H_{bb}^{-1}b_b\big).
\;}
\tag{08.14}
$$

$S = H_{aa} - H_{ab}H_{bb}^{-1}H_{ba}$ is the **Schur complement** of $H_{bb}$ in $H$. Two uses,
both central to Meridian:

1. **Eliminating a structure block (the classic VIO/BA use).** Make $\delta x_b$ the *landmarks*
   / per-point depths (§06 visual points) and $\delta x_a$ the poses. $H_{bb}$ is block-diagonal
   (each landmark independent given poses), so $H_{bb}^{-1}$ is trivial per-block; (08.14)
   reduces a huge system to a small pose-only system $S$, then back-substitutes the landmarks.
   This is the "Schur trick" that makes bundle adjustment tractable, and it is why §06's
   sparse-direct visual points can be many without blowing up the solve.

2. **Marginalization for fixed-lag smoothing (§09).** When a keyframe slides out of the window,
   we do not throw it away — we *marginalize* it. Marginalizing $\delta x_b$ means replacing all
   factors that touched it with a single Gaussian **prior** on the remaining $\delta x_a$, whose
   information is exactly the Schur complement $S$ and whose gradient is the right-hand side of
   (08.14). This is the algebraic identity behind the IMU prior term $\|x \boxminus \hat
   x\|^2_{P^{-1}}$ in FAST-LIO eq. (17): the propagated covariance $P$ is the marginal of the
   past, and $P^{-1}$ is its information injected into the current solve. (Caveat for §11: naïve
   marginalization can introduce inconsistency / spurious information if linearization points
   differ — FEJ, "first-estimates Jacobians," fixes this; flagged here, treated in §09/§11.)

> **Cost.** A dense solve of $H\,\delta x = -b$ is $O(n^3)$ (Cholesky). The Schur reduction
> costs $O(n_a^2 n_b)$ to form $S$ plus $O(n_a^3)$ to solve it — a win whenever $n_b$ is large
> and structured (block-diagonal $H_{bb}$) and $n_a$ small. For the dense $24$-dim front-end no
> Schur is needed; for the back-end and for landmark elimination it is essential.

---

### 08.6 The FAST-LIO measurement-dimension trick: solving over the *state*

FAST-LIO makes a beautiful observation that belongs squarely in "batch solving": when $m \gg n$
(thousands of LiDAR points, $\sim 24$-dim state), you must **never invert anything of size $m$.**
Start from the GN normal equations for the LiDAR + prior cost (08.1)/(FAST-LIO eq. 17). The
update solving that cost is the Kalman gain, and FAST-LIO gives *two algebraically identical*
forms:

$$
\text{(measurement form, eq. 18)}\quad K = P H^\top \big(H P H^\top + R\big)^{-1},
\tag{08.15}
$$

$$
\text{(state form, eq. 20)}\quad K = \big(H^\top R^{-1} H + P^{-1}\big)^{-1} H^\top R^{-1}.
\tag{08.16}
$$

In (08.15) the inverted matrix $H P H^\top + R$ is $m \times m$ (measurement-sized) — fatal when
$m=1453$ points. In (08.16) the inverted matrices $H^\top R^{-1}H + P^{-1}$ and $R$ are
$n\times n$ (state-sized, $\sim 24$) and block-diagonal respectively. The connection to (08.5)
is direct: $H^\top R^{-1} H$ **is** the GN Hessian $\sum_i J_i^\top \Omega_i J_i$ of the LiDAR
factors (with $\Omega_i = R_i^{-1}$), and $P^{-1}$ is the prior factor's information. So (08.16)
is precisely "solve the normal equations $(\sum J^\top\Omega J + P^{-1})\delta x = \dots$" — the
state form *is* batch Gauss–Newton, and the inverted matrix is exactly our $H$ from (08.5) plus
the prior. The paper measures the payoff (Table II): at 1453 points the old form (08.15) takes
$1219$ ms, the new form (08.16) takes $0.59$ ms — a $\sim 2000\times$ speedup, same answer.

In the code this is the branch on `n > dof_Measurement` in the iEKF update
(`esekfom.hpp:1080-1108`): when the state dimension exceeds the measurement count it uses the
measurement form; otherwise (the LiDAR case, $m \gg n$) it uses the state form
$(H^\top R^{-1}H + P^{-1})^{-1}H^\top R^{-1}$ — `esekfom.hpp:1105-1106`. Meridian's front-end
solver should default to the state form for the same reason.

---

### 08.7 IKF ↔ GN equivalence (the project-brief derivation)

We now show the iterated Kalman filter update *is* a Gauss–Newton step on the MAP cost (08.1) —
the equivalence the brief asks us to ground in `papers/2010.08196.txt`. Two parts.

**Part A — same cost, same minimizer.** FAST-LIO eq. (17) is the MAP cost
$\min_{\delta x}\, \|x\boxminus\hat x\|^2_{P^{-1}} + \sum_j \|z_j + H_j\,\delta x\|^2_{R_j^{-1}}$.
This is exactly (08.1) with two groups of factors: one prior factor (residual $x\boxminus\hat
x$, information $P^{-1}$) and the LiDAR factors (residual $z_j$, Jacobian $H_j$, information
$R_j^{-1}$). Applying the GN normal equations (08.6) to it gives

$$
\big(P^{-1} + H^\top R^{-1} H\big)\,\delta x \;=\; -\big(P^{-1}(x\boxminus\hat x) + H^\top R^{-1} z\big),
\tag{08.17}
$$

whose solution, after one line of algebra, is the iEKF update
$\delta x = -K z - (I - KH)(x\boxminus\hat x)$ with $K$ given by the state form (08.16). This is
FAST-LIO eq. (18). So **the GN step on (08.1) and the iEKF update are the same equation.** You
can see the *implemented* form of this in the solver: the per-iteration increment in
`esekfom.hpp:1111` is

```cpp
dx_ = K_ * (z - h) + (K_x - Identity) * dx_new;   // K_x = K_ * h_x
```

The first term $K(z-h)$ is the measurement pull (the $-Kz$ of eq. 18); the second
$(KH - I)\,\delta x_{\text{new}}$ is the prior pull-back toward the propagated state (the
$-(I-KH)(x_\kappa\boxminus\hat x)$ of eq. 18, with the sign and the $A$-matrix manifold
correction `dx_new` from `esekfom.hpp:1047-1048`). It iterates, re-linearizing $H,z$ at each new
$x$, until $\|\delta x\| < \varepsilon$ — i.e. **iterated** EKF = iterated GN. (§09 develops the
recursive/filtering reading of the *same* lines; the point here is that they reduce to (08.6).)

**Part B — the two gain forms are the same (Sherman–Morrison–Woodbury).** That (08.15) and
(08.16) give the *same* $K$ is the matrix-inversion-lemma identity proved in FAST-LIO Appendix B.
With $P, R \succ 0$,

$$
\big(P^{-1} + H^\top R^{-1} H\big)^{-1} \;=\; P - P H^\top \big(H P H^\top + R\big)^{-1} H P .
\tag{08.18}
$$

Right-multiplying by $H^\top R^{-1}$ and simplifying (the paper's two-line manipulation using
$HPH^\top R^{-1} = (HPH^\top + R)R^{-1} - I$) collapses (08.16) into (08.15). This is also the
reason the *covariance* update can be written either as $P^+ = (I-KH)P$ or in information form
$P^{+-1} = P^{-1} + H^\top R^{-1} H$ — the latter being the Hessian $H$ of (08.5) plus the prior,
i.e. the inverse of the posterior covariance is the GN information matrix. So GN, iEKF, and the
information-form covariance update are three views of one object: $H = \sum J^\top \Omega J$.

---

### 08.8 Levenberg–Marquardt: trust-region damping

Pure GN can diverge when the operating point is far from the optimum: the quadratic model
(08.4) is a bad fit, $H$ may be ill-conditioned or rank-deficient (degeneracy, §11), and the
unbounded step $\delta x = -H^{-1}b$ overshoots. Levenberg–Marquardt (LM) fixes this by damping:

$$
\boxed{\;\big(H + \lambda\, D\big)\,\delta x \;=\; -\,b, \qquad
D = I \ \text{(Levenberg)} \ \text{or} \ D = \mathrm{diag}(H) \ \text{(Marquardt)}. \;}
\tag{08.19}
$$

The damping $\lambda$ interpolates between two regimes:

- $\lambda \to 0$: recovers Gauss–Newton — fast quadratic convergence near the optimum.
- $\lambda \to \infty$: $\delta x \to -\frac{1}{\lambda}D^{-1}b$, a short step along the
  (scaled) **steepest-descent** direction — safe but slow.

Equivalently, (08.19) is the GN step of a *trust-region* subproblem: minimize the quadratic
model subject to $\|\delta x\|_D \le \Delta$, with $\lambda$ the Lagrange multiplier of the
trust radius $\Delta$. Two further benefits matter for SLAM:

1. **Regularization of rank-deficient $H$.** In a degenerate scene $H$ is singular and GN's
   $H^{-1}$ does not exist; $H + \lambda I$ is always invertible for $\lambda > 0$. LM thus
   *automatically* damps motion along unobservable directions — a poor-man's version of the
   explicit observability handling in §11 (which is preferable because it damps *only* the
   degenerate axes, not isotropically). The prior term $P^{-1}$ in the MAP cost already provides
   such regularization for the front-end, which is one reason FAST-LIO uses plain iterated GN
   (no explicit $\lambda$) — the prior plays LM's role.

2. **Adaptive control via the gain ratio.** Accept the step and shrink $\lambda$ (toward GN) if
   the *actual* cost decrease matches the *predicted* one; reject and grow $\lambda$ (toward
   gradient descent) if not. The gain ratio is

$$
\rho = \frac{C(x) - C(x \boxplus \delta x)}{\tfrac12\,\delta x^\top(\lambda D\,\delta x - b)} ,
\qquad
\rho > 0 \Rightarrow \text{accept, } \lambda \mathrel{/}= \nu;\quad
\rho \le 0 \Rightarrow \text{reject, } \lambda \mathrel{\times}= \nu .
\tag{08.20}
$$

For Meridian: the §03 keyframe back-end (GTSAM) uses LM (or Dogleg) for batch graph optimization
where robustness to bad initialization matters; the §09 front-end uses iterated GN with the
prior as regularizer for speed. Both are the same $(H,b)$ assembly of (08.5) — only the linear
solve differs by the $\lambda D$ term.

---

### 08.9 Putting it together: one batch step for LiDAR + IMU

A complete batch GN iteration of the Meridian front-end, mapped to everything above:

```
Inputs:  current estimate x = (R,p,v, bg,ba, g, R_LI,p_LI),  prior (x_prop, P)
         IMU preintegrant (§04),  LiDAR scan -> map correspondences (§05)

1. H <- P^{-1};  b <- P^{-1} (x boxminus x_prop)          # prior factor  (FAST-LIO eq.17 term 1)
2. (r_IMU, J_IMU) <- preintegration residual (§04)         # couples R,p,v,bg,ba,g
   H += J_IMU^T Omega_IMU J_IMU;  b += J_IMU^T Omega_IMU r_IMU
3. for each effective LiDAR point j (≈1000):               # h_share_model, laserMapping.cpp:863
       r_j = n_j^T (G p_fj - q_j)                           # point-to-plane, eq. (08.9)
       J_j = [ -n_j^T (R_I I_pfj)^  ,  n_j^T , ... ]        # eq. (08.10), 1xN row
       Omega_j = 1/sigma_j^2  (x §11 degeneracy reshaping)
       H += J_j^T Omega_j J_j;   b += J_j^T Omega_j r_j     # rank-1 accumulate, eq. (08.11)
4. solve   H dx = -b      via state-form gain (08.16) since m>>n   # esekfom.hpp:1105
           (or (H + lambda diag H) dx = -b  if using LM)
5. x <- x boxplus dx                                        # §02 retraction
6. if ||dx|| < eps stop; else relinearize and go to 1      # iterated GN == iterated EKF (§08.7)
On convergence: P_posterior = H^{-1}  (= (P^{-1}+H_L^T R^{-1} H_L)^{-1}), §08.7 Part B
```

Every line traces to a numbered equation: step 1 is the prior/marginal (08.14); steps 2–3 are
the per-sensor $J^\top\Omega J$ accumulation (08.5) whose *overlap* on the $(R,p)$ block is the
tight coupling (08.8); step 4 is the normal-equation solve (08.6) in its $m\gg n$ state form
(08.16); step 6 is the GN↔iEKF iteration (08.7).

---

### 08.10 What §08 hands to its neighbours

- **To §09 (recursive solving).** The exact same $(H,b)$ and the exact same update
  $\delta x = -Kz - (I-KH)(x\boxminus\hat x)$; §09 re-derives it as a filter and adds fixed-lag
  marginalization (the Schur complement of §08.5 applied to the oldest keyframe).
- **To §10 (continuous time).** Replace the discrete state $x$ by B-spline control points
  $c_k$; the residuals of §05–§07 then have Jacobians w.r.t. several $c_k$ via the spline basis,
  giving a *banded* $H$ (each measurement touches the $\le 4$ control points spanning its time) —
  a new sparsity pattern solved by the same (08.6)/Schur machinery.
- **To §11 (robustness).** Robust kernels and degeneracy handling enter *only* by reshaping the
  per-factor $\Omega_i$ and $r_i$ before the accumulation (08.5); GNC re-solves (08.6) on a
  graduated sequence of $\Omega_i$; PCM is a consistency test on candidate loop factors before
  they are added to $H$.
- **To §12 (synthesis).** §12 walks one full FAST-LIO2 / FAST-LIVO2 step end-to-end; this
  section is the "solve" box inside it.

**Key takeaways.** (i) Every estimator in this course reduces to *build $H=\sum
J^\top\Omega J$, $b=\sum J^\top\Omega r$, solve $H\delta x = -b$, retract*. (ii) "Tightly
coupled" is the single statement that all sensors contribute to *one* $H$, overlapping on shared
state blocks (08.8). (iii) Sparsity (back-end) and the measurement-dimension trick (front-end)
are the two routes to real-time; the Schur complement underlies both landmark elimination and
marginalization. (iv) GN, the iterated EKF, and the information-form covariance update are the
same computation — the FAST-LIO equivalence (08.7), grounded in `papers/2010.08196.txt`
eqs. (17)–(20) + Appendix B, and in `esekfom.hpp:1080-1111`.


---


## 09. Solving II (recursive): iterated EKF/ESIKF, equivalence, fixed-lag

> **Where we are.** Section 08 solved the full *batch* MAP problem: stack all
> residuals, build the normal equations $H^\top \Omega H\,\delta = -H^\top\Omega\,r$,
> exploit sparsity and the Schur complement, iterate Gauss–Newton (GN) or
> Levenberg–Marquardt (LM) to convergence. That is the right tool when many
> states are *jointly* connected by many factors — exactly the back-end factor
> graph of §08 and the loop-closure graph of §11/§L3.
>
> This section develops the *recursive* alternative. When the estimator marches
> forward in time and we are willing to **summarise everything before "now" by a
> single Gaussian prior**, the batch solve collapses to a filter: predict the
> prior through the dynamics (§04's IMU model), then fold in the current
> measurements. We will derive the **error-state iterated Kalman filter (IEKF /
> ESIKF)** *on manifolds*, prove its **equivalence to one Gauss–Newton step on a
> two-factor MAP problem** (the Bell–Cathey result), pin down the **tangent
> reprojection ($J$) matrix** that makes the manifold version exact, discuss
> **convergence**, and then situate the filter inside the broader **fixed-lag
> smoother** — the windowed middle ground between the pure filter (FAST-LIO2, our
> front-end §L2 v1) and the full smoother (GTSAM iSAM2, our back-end §L3).
>
> This is the algorithm that *runs every scan* in FAST-LIO2 and in the Meridian v1
> front-end behind `IFrontEnd`. Every claim below is grounded in the on-disk
> IKFoM implementation; the canonical reference is
> `FAST_LIO/include/IKFoM_toolkit/esekfom/esekfom.hpp`.

Notation is the shared one fixed in §02: state $x$ on a manifold $\mathcal{M}$ with
$\boxplus:\mathcal{M}\times\mathbb{R}^n\!\to\!\mathcal{M}$ and
$\boxminus:\mathcal{M}\times\mathcal{M}\!\to\!\mathbb{R}^n$; rotation $R\in SO(3)$,
$\mathrm{Exp}/\mathrm{Log}$ the $\mathbb{R}^3\!\leftrightarrow\!SO(3)$ maps, $(\cdot)^\wedge$ the hat
operator; residual $r$, measurement $z$, prediction $h(x)$, Jacobian $H$,
covariance $\Sigma$, information $\Omega=\Sigma^{-1}$, Kalman gain $K$.

---

### 09.1 From batch to recursive: the Markov assumption

The full MAP problem of §03 maximises the posterior over the *entire* trajectory
$x_{0:k}$ given all measurements $z_{1:k}$. Recursive estimation makes one
structural assumption: the process is **Markov** and we keep only the *current*
state $x_k$, encapsulating the whole past in a Gaussian belief

$$
x_k \;\sim\; \mathcal{N}\!\big(\hat{x}_k,\, \Sigma_k\big),
\qquad
\text{(equivalently information form } \Omega_k = \Sigma_k^{-1}\text{).}
$$

Each cycle has two stages, exactly mirroring the two factor *types* of §03:

1. **Predict / propagate** — push the belief through the motion model $x_{k} =
   f(x_{k-1}, u_k) \boxplus w$ (the IMU kinematics of §04). This is the *temporal*
   (between) factor, evaluated and marginalised in closed form.
2. **Update / correct** — fold in the measurement $z_k = h(x_k) \boxplus v$ (the
   LiDAR point-to-plane residual of §05, the photometric residual of §06, the
   GNSS residual of §07). This is the *measurement* (unary) factor.

The deep statement of this section is that **stage 2, done iteratively, is
algebraically identical to a Gauss–Newton solve of a tiny MAP problem with
exactly two factors: the prior from stage 1 and the measurement.** The filter is
not an approximation *of a different thing*; it *is* MAP, just with the past
compressed to a single Gaussian. The only information lost relative to §08 is the
ability to *relinearise the past* — which is precisely what fixed-lag smoothing
(§09.7) buys back, and what the back-end (§L3) restores globally.

---

### 09.2 The error state and why we filter in the tangent space

We never run a Kalman filter directly on $x$, because $x$ lives on a manifold and
$\mathcal{N}(\hat{x},\Sigma)$ is only meaningful in a flat vector space. Instead
we define the **error state** $\delta x \in \mathbb{R}^n$ in the tangent space at
the current operating point $\hat{x}$ (§02):

$$
x \;=\; \hat{x} \boxplus \delta x,
\qquad
\delta x \;=\; x \boxminus \hat{x},
\qquad
\delta x \sim \mathcal{N}(0,\Sigma).
$$

All covariances, gains, and Jacobians live in this $n$-dimensional tangent space.
For Meridian's v1 front-end the state is the 24-element / **23-DOF** IKFoM manifold
declared in `use-ikfom.hpp:12–21` (with the LiDAR-IMU extrinsics, velocity, biases
and gravity built out in `get_f`/`df_dx`, `use-ikfom.hpp:47–77`):

```
MTK_BUILD_MANIFOLD(state_ikfom,
  ((vect3, pos)) ((SO3, rot))
  ((SO3, offset_R_L_I)) ((vect3, offset_T_L_I))   // online LiDAR–IMU extrinsics
  ((vect3, vel)) ((vect3, bg)) ((vect3, ba))
  ((S2, grav)));                                   // gravity on the 2-sphere
```

Note the DOF bookkeeping that recurs throughout `esekfom.hpp`: the *ambient*
dimension is `m = 24` but the *tangent* (error-state) dimension is `n = 23`,
because gravity is constrained to the sphere $S^2$ and contributes only **2** DOF
(`use-ikfom.hpp:8`, `S2<double,98090,…>`). This is exactly why §02's distinction
between $\dim\mathcal{M}$ and $\dim T_x\mathcal{M}$ is not pedantry: the filter's
$P$ is $23\times23$, not $24\times24$. The extrinsics `offset_R_L_I`,
`offset_T_L_I` are *state variables*, so the same filter performs the **online
extrinsic refinement** the architecture requires (cross-cutting calibration).

---

### 09.3 The propagation step (the prior factor)

`predict()` (`esekfom.hpp:279–383`) advances $\hat{x}$ and $\Sigma$ over one IMU
interval $\Delta t$. The mean is propagated on the manifold,

```
x_.oplus(f_, dt);            // x̂ ← x̂ ⊞ (f(x̂,u)·Δt)        esekfom.hpp:287
```

with $f(\cdot)$ the continuous kinematics of `get_f()` (`use-ikfom.hpp:35–47`):
$\dot p = v$, $\dot R = R\,(\omega - b_g)^\wedge$ (here written through $\boxminus$),
$\dot v = R(a - b_a) - g$, biases and gravity driftless except for noise. This is
§04's forward propagation, verbatim.

The covariance is propagated by the discrete error-state transition $F_x$ and
noise mapping $F_w$ (`esekfom.hpp:289–381`):

$$
\Sigma_{k|k-1} \;=\; F_x\,\Sigma_{k-1}\,F_x^\top \;+\; (\Delta t\,F_w)\,Q\,(\Delta t\,F_w)^\top.
$$

```
F_x1.block<3,3>(idx,idx) = res.toRotationMatrix();  // = Exp(-ω·dt) rot block :321
F_x1 += f_x_final * dt;                              // remaining blocks      :380
P_ = (F_x1) * P_ * (F_x1).transpose()
       + (dt*f_w_final)*Q*(dt*f_w_final).transpose();                       // :381
```

Two details matter for what follows. First, the rotation block of $F_x$ is
$\mathrm{Exp}(-\omega\,\Delta t)$, not $I$: a *finite* error-state transition on
$SO(3)$ (`esekfom.hpp:309–321`), the manifold analogue of the linear $\Phi$.
Second, $f_x$ is exactly the analytic Jacobian `df_dx()` (`use-ikfom.hpp:61–77`),
e.g. $\partial\dot v/\partial R = -R\,(a-b_a)^\wedge$ (`:69`),
$\partial\dot v/\partial b_a = -R$ (`:70`), the gravity tangent basis (`S2_Mx`) at
`:73–74`. The output of this step is the **prior** for the update:

$$
\boxed{\;\hat{x}^{\,0} \triangleq \hat{x}_{k|k-1}, \qquad
\Sigma_{k|k-1}\; \text{(the prior information } \Omega_0 = \Sigma_{k|k-1}^{-1}).\;}
$$

In Meridian terms, `predict()` is called once per IMU sample inside
`IMU_Processing.hpp:284` (`kf_state.predict(dt, Q, in)`, with a final partial-step
call at `:301`), and the saved per-sample poses drive the backward
motion-compensation undistortion of §05 (`IMU_Processing.hpp:323–344`).

---

### 09.4 The iterated update: derivation from MAP

Now hold the prior $(\hat{x}^{0},\Sigma_{k|k-1})$ fixed and fold in the LiDAR
measurement. The single-cycle MAP cost is the sum of exactly **two** factors —
the prior and the measurement (§03):

$$
\min_{x}\;
\underbrace{\tfrac12 \,\big\| x \boxminus \hat{x}^{0} \big\|_{\Sigma_{k|k-1}^{-1}}^{2}}_{\text{prior factor}}
\;+\;
\underbrace{\tfrac12 \,\big\| z - h(x) \big\|_{R^{-1}}^{2}}_{\text{measurement factor}} .
\tag{9.1}
$$

This is (9.1) — the objective that the iterated filter minimises, and the bridge
to §08. We solve it by Gauss–Newton **on the manifold**, iterating an inner
index $j$ with operating point $\hat{x}^{j}$ (initialised at the prior
$\hat{x}^{0}$). Parameterise $x = \hat{x}^{j} \boxplus \delta$ and linearise both
terms at $\hat{x}^{j}$.

**Measurement factor.** $h(\hat{x}^{j}\boxplus\delta) \approx h(\hat{x}^{j}) + H_j\,\delta$,
where $H_j = \left.\partial h(\hat{x}^{j}\boxplus\delta)/\partial\delta\right|_{0}$
is the measurement Jacobian *in the tangent space at $\hat{x}^{j}$*. The residual
is $r_j = z - h(\hat{x}^{j})$.

**Prior factor.** This is the subtle part that distinguishes the *manifold*
filter from the textbook linear one. Expand $x \boxminus \hat{x}^{0}$ around the
*current* operating point $\hat{x}^{j}$, not around the prior:

$$
x \boxminus \hat{x}^{0}
\;=\; (\hat{x}^{j}\boxplus\delta)\boxminus \hat{x}^{0}
\;\approx\; \underbrace{(\hat{x}^{j}\boxminus\hat{x}^{0})}_{\textstyle =: \;\delta x_j}
\;+\; J_j\,\delta,
\tag{9.2}
$$

where

$$
J_j \;=\; \frac{\partial\big[(\hat{x}^{j}\boxplus\delta)\boxminus\hat{x}^{0}\big]}{\partial \delta}\bigg|_{0}
$$

is the **tangent reprojection / boxplus-Jacobian** of §02. This is exactly
eq. (15)–(16) of FAST-LIO (`papers/2010.08196.txt:507–525`), where the prior is
written $x_k \boxminus \hat{x}^0 = (\hat{x}^\kappa\boxminus\hat{x}^0) + J^\kappa\tilde{x}_k$
and $J^\kappa$ involves $A(\cdot)^{-1}$ (their eq. 16, the inverse of our
left-Jacobian block). On $SO(3)$ it is the (left/right, per convention) Jacobian
of the exponential map; for $\mathbb{R}^n$ blocks it is $I$. *This $J_j$ has no
analogue in the flat Kalman filter* — it is identically $I$ there — and dropping it
is the difference between a textbook "ESKF" and the exact on-manifold IEKF
(Bourmaud, He et al.). Indeed the paper notes explicitly that "for the first
iteration (i.e. the case of extended Kalman filter), $\hat{x}^\kappa=\hat{x}^0$,
then $J=I$" (`papers/2010.08196.txt:525`). We pin down its code realisation in §09.5.

Combining this prior (FAST-LIO eq. 15) with the linearised measurement yields the
MAP cost — their eq. (17), `papers/2010.08196.txt:527–545`. Substituting (9.2) and
the measurement linearisation into (9.1) gives a *quadratic* in $\delta$:

$$
\min_{\delta}\;
\tfrac12\,\big\| \delta x_j + J_j\,\delta \big\|_{\Sigma_{k|k-1}^{-1}}^{2}
\;+\;
\tfrac12\,\big\| r_j - H_j\,\delta \big\|_{R^{-1}}^{2}.
$$

Its normal equations are the GN system (§08), with the *prior information
reprojected by $J_j$*:

$$
\Big(\underbrace{J_j^{\top}\Sigma_{k|k-1}^{-1} J_j}_{\text{reprojected prior info}}
+ H_j^{\top} R^{-1} H_j\Big)\,\delta
\;=\;
- J_j^{\top}\Sigma_{k|k-1}^{-1}\,\delta x_j
+ H_j^{\top} R^{-1} r_j .
\tag{9.3}
$$

Define the **reprojected prior covariance** $P_j \triangleq J_j\,\Sigma_{k|k-1}\,J_j^{\top}$
(so $J_j^{\top}\Sigma_{k|k-1}^{-1}J_j = P_j^{-1}$ when $J_j$ is invertible).
Equation (9.3) is *exactly one Gauss–Newton step of (9.1)*. Solving it and
applying the increment on the manifold,

$$
\boxed{\;\hat{x}^{j+1} \;=\; \hat{x}^{j} \boxplus \delta^{\star},\;}
$$

and repeating until $\|\delta^{\star}\|$ is below tolerance, is the **iterated
error-state Kalman filter**. When it converges, $\hat{x}^{j}\to\hat{x}_{k|k}$, the
MAP estimate of (9.1).

**Kalman gain form.** Equation (9.3) can be rewritten in the familiar "gain"
shape via the matrix-inversion lemma — FAST-LIO eq. (18),
`papers/2010.08196.txt:586`. With $P_j = J_j\Sigma_{k|k-1}J_j^\top$,

$$
K_j = P_j H_j^\top\big(H_j P_j H_j^\top + R\big)^{-1},
\qquad
\delta^{\star} = -K_j\,r_j \;-\; (I - K_j H_j)\,\delta x_j .
\tag{9.4}
$$

The complementary **state-dimension** gain (their eq. 20, line 598),
$K_j = (H_j^\top R^{-1}H_j + P_j^{-1})^{-1}H_j^\top R^{-1}$, is the form actually
used for LiDAR (proved equivalent to (9.4) by the matrix-inverse lemma in their
Appendix B, lines 723–733) — see §09.5 and §08 for the complexity argument.

The two terms have a clean reading: $-K_j r_j$ is the **innovation correction**
(pull toward the measurement); $-(I-K_jH_j)\,\delta x_j$ is the **prior
recentring** that drags the iterate back toward the propagated prior $\hat{x}^0$,
present *only* because we relinearise away from $\hat{x}^0$. In the first
iteration $\delta x_0 = \hat{x}^0\boxminus\hat{x}^0 = 0$, the second term vanishes,
and (9.4) reduces to the *standard EKF update* $\delta = -K\,r$. **The ordinary
EKF is the IEKF stopped after one iteration.**

---

### 09.5 Mapping the derivation onto `esekfom.hpp`

The IKFoM function `update_iterated_dyn_share_modified(double R, double &solve_time)`
(`esekfom.hpp:1619`) is (9.1)–(9.4) made executable. Walk it line by line. (It is
the "modified" variant FAST-LIO2 actually calls; the generic
`update_iterated_dyn_share` at `:1001` is the same algorithm without the
state-block / scalar-$R$ specialisations.)

**The inner GN loop.** `for(int i=-1; i<maximum_iter; i++)` (`:1633`). Note it
starts at $-1$: that "iteration $-1$" is the first linearisation at the prior. The
prior is frozen up front:

```
state x_propagated = x_;     cov P_propagated = P_;     // x̂⁰, Σ_{k|k-1}    :1623–1624
```

**Residual and Jacobian.** `h_dyn_share(x_, dyn_share)` (`:1636`) is a user
callback — `h_share_model()` in `laserMapping.cpp:638` — that, at the *current*
operating point `x_`, does association and fills `dyn_share.h` ($r_j$) and
`dyn_share.h_x` ($H_j$). For LiDAR point-to-plane (§05) each row is the $1\times12$
block (`laserMapping.cpp:725–751`)

```
V3D C(s.rot.conjugate() * norm_vec);          // R^T n
V3D A(point_crossmat * C);                     // (R_LI p + t_LI)^ (R^T n)
ekfom_data.h_x.block<1,12>(i,0) << n^T, A^T, B^T, C^T;   // pos | rot | ext-R | ext-t  :743
ekfom_data.h(i) = -norm_p.intensity;            // residual = -(n·p_world + d)         :751
```

i.e. $\partial r/\partial p = n^\top$, $\partial r/\partial\theta = \big((R_{LI}p+t_{LI})^\wedge R^\top n\big)^\top$,
and the extrinsic columns from $B,C$. Columns 12–22 (vel, $b_g$, $b_a$, grav) are
**zero**: LiDAR does not *directly* observe them. This sparsity is why every gain
computation in the file touches only `block<n,12>` — a concrete instance of the
Schur/structure exploitation of §08.

**Critical detail — frozen data association.** `h_share_model` only re-runs the
k-d-tree nearest-neighbour search (`ikdtree.Nearest_Search`, `laserMapping.cpp:670`)
when `ekfom_data.converge` is true (the `if (ekfom_data.converge)` guard at
`laserMapping.cpp:667`). So across inner iterations the **correspondences are
held fixed** and only the residual/Jacobian are recomputed at the new
linearisation point. This is what keeps the inner loop a genuine GN solve of a
*fixed* cost (9.1); re-associating every iteration would change the cost function
under the optimiser and break the equivalence (and convergence) argument.

**The prior term $\delta x_j$ (9.2).** `x_.boxminus(dx, x_propagated)` (`:1652`)
computes $\delta x_j = \hat{x}^{j}\boxminus\hat{x}^{0}$ — exactly the prior offset
in (9.2)/(9.4). Stored as `dx_new`.

**The tangent reprojection $J_j$.** This is the load-bearing manifold step, and
in IKFoM it is applied *to the covariance*, not as an explicit matrix multiply
into (9.3). The SO(3)-block loop (`:1659–1698`) rebuilds $P_j$ from the propagated
$\Sigma$:

```
P_ = P_propagated;
for each SO3 block at offset idx:
    seg_SO3 = dx(idx .. idx+2);                         // the SO(3) part of δx_j
    res_temp_SO3 = A_matrix(seg_SO3).transpose();        // = J_l(δθ)^T
    P_.block<3,1>(idx,i) = res_temp_SO3 * P_.block<3,1>(idx,i);   // left-multiply rows
    P_.block<1,3>(i,idx) = P_.block<1,3>(i,idx) * res_temp_SO3.transpose(); // and cols
```

Here `A_matrix(v)` is the **left Jacobian** $J_l(v)$ of $SO(3)$
(`IKFoM_toolkit/mtk/src/mtkmath.hpp:236`, $A(v)=I+\frac{1-\cos|v|}{|v|^2}[v]_\times
+\frac{|v|-\sin|v|}{|v|^3}[v]_\times^2$). The double-sided multiply
$P_j \leftarrow J\,\Sigma_{k|k-1}\,J^\top$ on the rotation block(s) is precisely
$P_j = J_j\Sigma_{k|k-1}J_j^\top$ of §09.4. **This is the $J$/$L$ projection the
section is about**: it transports the prior covariance from the tangent frame at
$\hat{x}^{0}$ to the tangent frame at $\hat{x}^{j}$, so prior and measurement
factors live in the *same* tangent space before they are combined. Drop it (set
$J=I$) and you have an ordinary ESKF; keep it and you have the exact on-manifold
IEKF whose fixed point is the true MAP estimate.

**Solving (9.3)/(9.4).** Two algebraically-equivalent branches, chosen by problem
shape (this is the filter-vs-information tradeoff of §08, decided at runtime):

- `if (n > dof_Measurement)` (`:1736`): few measurements — use the
  **measurement-space** (Woodbury) gain $K = P H^\top (HPH^\top + R)^{-1}$,
  inverting a $\mathrm{dof}\times\mathrm{dof}$ matrix (`K_h = K_*h`, `K_x = K_*h_x`,
  `:1739–1740`).
- `else` (`:1745` onward): many measurements (the LiDAR case, thousands of points)
  — use the **state-space / information form**, inverting $n\times n$ once:

  ```
  HTH.block<12,12> = h_x_^T * h_x_ / R;    // H^T R^{-1} H   (R scalar)        :~1760
  P_temp = (P_/R).inverse();  P_temp += HTH;  P_inv = P_temp.inverse();        // GN info inv
  K_h = P_inv.block<n,12>(0,0) * h_x_.transpose() * dyn_share.h;  // = K r term :1804
  K_x.block<n,12>(0,0) = P_inv.block<n,12>(0,0) * HTH;                          :1809
  dx_ = K_h + (K_x - I) * dx_new;          // = -K r? — see sign note below     :1815
  ```

  Read off the correspondence to (9.4): the `K_h` term is the innovation
  correction $-K_j r_j$ (the sign is absorbed because $r_j = -h$ is stored as
  `dyn_share.h`, i.e. $h(i)=-\text{dist}$), and `(K_x - I)*dx_new`
  $= -(I-K_jH_j)\,\delta x_j$, so `dx_` $= -K_j r_j - (I-K_jH_j)\delta x_j =
  \delta^{\star}$. The increment is applied on the manifold by
  `x_.boxplus(dx_)` (`:1817`) $= \hat{x}^{j+1}$.

**Convergence test.** After each $\boxplus$, the code checks every coordinate of
`dx_` against a per-state tolerance `limit[i]` (`:1818–1826`, set from
`epsi[23]={0.001}` in `laserMapping.cpp:283–287`, passed to `init_dyn_share` at
`:828`). When all components fall below tolerance, `dyn_share.converge=true`, the
next pass re-associates, and after a stable pass (`t>1`) or hitting `maximum_iter`
(default **4**, `laserMapping.cpp:765`) the loop exits with the converged
$\hat{x}_{k|k}$.

**Posterior covariance (the $L$ matrix).** On the final iteration
(`t>1 || i==maximum_iter-1`, `:1834`) the code forms the posterior. It first
builds `L_` by reprojecting $P$ through $J$ once more on the SO(3) blocks (the same
$A\_matrix(\cdot)^\top$ transport, `:1845`), then:

$$
\Sigma_{k|k} \;=\; L \;-\; K_x\big[{:},\,0\!:\!12\big]\;P\big[0\!:\!12,\,{:}\big],
$$

```
P_ = L_ - K_x.block<n,12>(0,0) * P_.block<12,n>(0,0);     // last line of the block
```

This is the on-manifold $(I - KH)P$ posterior update, with the leading $J$-transport
folded into `L_`. The split `K_x.block<n,12> * P.block<12,n>` is again the LiDAR
sparsity: only the 12 directly-observed columns contribute. The result is the
**information-form posterior** $\Sigma_{k|k} = (P_j^{-1} + H^\top R^{-1}H)^{-1}$
of GN, transported back to the operating-point tangent frame — closing the loop
with (9.3).

---

### 09.6 Equivalence to Gauss–Newton on MAP (Bell–Cathey) and convergence

We can now state the result precisely.

> **Theorem (Bell & Cathey, 1993; manifold version He–Xu–Zhang, IKFoM 2021).**
> The iterated EKF update is identical, iteration for iteration, to the
> Gauss–Newton method applied to the MAP cost (9.1). At convergence the IEKF
> estimate is the MAP estimate $\arg\min$ of (9.1), and its reported covariance
> $\Sigma_{k|k}$ is the inverse of the GN information (Laplace approximation of the
> posterior; FAST-LIO eq. 19, `papers/2010.08196.txt:592`).

FAST-LIO states this directly: combining the prior (15) with the measurement (14)
"yields the maximum a-posteriori estimate (MAP)" (17), and "optimizing the
resultant quadratic cost leads to the standard iterated Kalman filter" (18)
(`papers/2010.08196.txt:526–567`). FAST-LIO2 repeats the construction on the
manifold $\mathcal{M}=SO(3)\times\mathbb{R}^{15}\times SO(3)\times\mathbb{R}^3$
(`papers/2107.06829.txt:269–393`, §IV-B "Iterated Kalman Filter").

*Proof sketch.* We already did the work. The GN step of (9.1) is the normal
equation (9.3); §09.5 shows `update_iterated_dyn_share_modified` builds and solves
exactly (9.3) every inner iteration, with the prior term $\delta x_j$ from
`boxminus` and the $J$-transport applied to the prior information. The Kalman
form (9.4) is (9.3) after the matrix-inversion lemma, an algebraic identity. The
only filter-specific objects — gain $K$, "innovation" $r_j$, posterior $(I-KH)P$
— are GN quantities in disguise: $K = P_jH^\top(HP_jH^\top+R)^{-1}$ is the GN
solution operator, the innovation is the GN residual, $(P_j^{-1}+H^\top R^{-1}H)^{-1}$
is the GN information inverse. $\square$

This equivalence is the central organising idea of the whole chapter: §08's batch
GN and §09's IEKF are *the same algorithm* applied to *differently scoped*
problems — batch over a window, filter over a single cycle with the past
compressed. It is also why the FAST-LIO authors describe their estimator as both
"a tightly-coupled iterated Kalman filter" and a MAP estimator
(`papers/2010.08196.txt` §I–§III, the IKFoM formulation): there is no
contradiction.

**What is the EKF, then?** The standard EKF is GN truncated at one iteration with
linearisation frozen at the *prior* $\hat{x}^0$ (so $\delta x_0=0$ and the second
term of (9.4) is absent). For mildly nonlinear measurements one step is fine; for
the strongly nonlinear point-to-plane and photometric residuals of §05–§06,
iterating to convergence materially reduces linearisation error — the motivation
for IKFoM over a plain ESKF.

**Convergence.** Because each step is GN on the smooth cost (9.1), the standard GN
theory applies:

- **Local quadratic-ish convergence.** Near a minimiser where the residual is
  small (good data association, low noise), GN converges superlinearly; the IEKF
  inherits this. In practice FAST-LIO2 converges in 2–4 iterations
  (`NUM_MAX_ITERATIONS` is small; the per-axis tolerance `epsi=1e-3` is hit
  fast).
- **No global guarantee / divergence modes.** GN can fail when (i) the
  linearisation is poor far from the optimum (large initial error — mitigated by
  the IMU prior giving a good $\hat{x}^0$), or (ii) the Gauss–Newton Hessian
  $H^\top R^{-1}H$ is **rank-deficient** — *degeneracy/observability*, the subject
  of §11. The information-form branch (`:1745+`, via `(P_/R).inverse()`) keeps
  $\Sigma_{k|k-1}^{-1}$ in the
  system, so the prior **regularises** rank-deficient directions: unobserved DOF
  simply retain their prior, never blowing up. This is the filter's built-in
  Tikhonov term, and it is why a pure point-to-plane LiDAR update in a geometric
  corridor does not diverge — it just stops correcting along the corridor axis.
  §11 turns this from accident into design by feeding per-axis observability into
  the noise model.
- **No LM trust region.** IKFoM uses raw GN (no damping $\lambda$), relying on the
  IMU prior for a good initial point and on the `limit[]` early-out to stop. For
  Meridian's CT v2 (§10) and the back-end (§L3) we use LM/iSAM2, which *do* damp.

---

### 09.7 Fixed-lag smoothing: the windowed middle ground

The filter keeps **one** state and marginalises the rest; the full smoother (§08,
§L3) keeps **all** states and relinearises everything. The **fixed-lag smoother
(FLS)** keeps the last $W$ states in a sliding window and marginalises only what
falls off the back:

$$
\min_{x_{k-W+1:\,k}}\;
\underbrace{\tfrac12\big\|x_{k-W+1}\boxminus \bar{x}\big\|^2_{\bar\Sigma^{-1}}}_{\text{marginal prior}}
\;+\;
\sum_{i=k-W+2}^{k}\!\!\|r^{\text{imu}}_i\|^2_{\cdots}
\;+\;
\sum_{i=k-W+1}^{k}\!\!\|r^{\text{meas}}_i\|^2_{\cdots}.
\tag{9.5}
$$

The two limits recover what we already know:

| Window $W$ | What it is | In Meridian |
|---|---|---|
| $W=1$ | **Pure filter** (ESIKF) — past compressed to one Gaussian, never relinearised | §L2 v1 front-end (FAST-LIO2-style `IFrontEnd`) |
| $1 < W < \infty$ | **Fixed-lag smoother** — relinearise within the window, marginalise the tail | §L2 v2 continuous-time front-end (§10) sliding window |
| $W=\infty$ | **Full smoother** — relinearise everything (iSAM2 reuses past work) | §L3 back-end factor graph |

FLS keeps the favourable cost of a filter (bounded state size $\Rightarrow$
bounded per-step compute) while recovering the smoother's chief advantage:
**states inside the window are relinearised**, so an early bad linearisation can
be corrected once better measurements arrive. This directly fixes the filter's
one structural weakness (§09.1): the filter *cannot* revise the past, so an
overconfident early estimate is baked in forever. The price is the **marginal
prior** $\mathcal{N}(\bar x,\bar\Sigma)$ in (9.5): when a state leaves the window
we Schur-complement it out (§08), producing a dense prior on its former
neighbours. Done naively this both *densifies* the graph and *freezes* the
linearisation point of the marginalised factors — the well-known **first-estimates
Jacobian (FEJ)** consistency problem (Huang, Mourikis & Roumeliotis). The fix,
used in OKVIS/VINS-Mono-style estimators and applicable to our CT window (§10), is
to evaluate the Jacobians of any factor connected to a marginalised state at the
*first* estimate, so the marginalisation does not inject spurious information into
observable directions. §11 returns to this under observability.

**Why Meridian uses both forms.** The architecture deliberately spans the whole
$W$-spectrum behind clean interfaces:

- The **front-end** (§L2) runs at sensor rate and *must* be cheap and recursive:
  v1 is the pure ESIKF ($W=1$) derived above; v2 swaps in the CT sliding-window
  estimator (§10), a fixed-lag smoother on B-spline control points. Both sit
  behind `IFrontEnd`, so v2 drops in without touching the back-end — exactly the
  swappability the principles demand.
- The **back-end** (§L3) runs at keyframe rate and *can* afford to relinearise
  globally: GTSAM **iSAM2** is the $W=\infty$ smoother, made tractable by
  incremental reuse (it relinearises only the variables a new factor actually
  perturbs — a Bayes-tree analogue of the windowed relinearisation in (9.5)).

The front-end emits keyframes with their filter covariance $\Sigma_{k|k}$ (§09.5);
the back-end ingests those as priors/odometry factors and adds loop closures
(§L5) and GNSS (§07). The covariance the filter computes is therefore not just
introspection — it is the *information handoff* between the recursive front-end and
the smoothing back-end. (For Meridian's debug surface this is exactly what we expose
on the per-module ROS 2 topics: per-iteration $\|\delta^\star\|$, effective point
count `effct_feat_num`, the diagonal of $\Sigma_{k|k}$, and the per-axis
observability of §11.)

**Point-LIO note (every-measurement filtering).** A further point on the spectrum
worth flagging for §L2: Point-LIO and its IKFoM variant
`update_iterated_dyn_share_IMU()` (called at `Point-LIO/src/Estimator.cpp:665`,
alongside `update_iterated_dyn_share_modified()` at `:703`) treat the IMU reading
as a *measurement* and update the filter **per point** rather than per scan, eliminating
the predict/undistort split entirely and handling very aggressive motion. It is
the same IEKF machinery (9.1)–(9.4) at an extreme update rate; we mention it as a
candidate refinement once the v1 front-end is stable, not part of first-pass scope.

---

### 09.8 One full ESIKF cycle (pseudocode, code-grounded)

Putting §09.3–09.5 together, here is the cycle that runs once per LiDAR scan in
the Meridian v1 front-end. Line references are to `esekfom.hpp` /
`laserMapping.cpp` / `IMU_Processing.hpp`.

```
# ---- PREDICT (per IMU sample in the scan) ----            IMU_Processing.hpp:284
for each IMU sample u with interval dt:
    x̂ ← x̂ ⊞ f(x̂,u)·dt                                     esekfom.hpp:287
    Σ ← F_x Σ F_xᵀ + (dt F_w) Q (dt F_w)ᵀ                  esekfom.hpp:381
save (x̂⁰, Σ_{k|k-1})                       # the prior factor of (9.1)

# ---- UPDATE (iterated GN on (9.1)) ----   esekfom.hpp:1619  kf.update_iterated_dyn_share_modified
x_prop, P_prop ← x̂⁰, Σ_{k|k-1}                              :1623
for j = -1 … maximum_iter:                                   :1633
    # association frozen unless converged flag set:
    (r_j, H_j) ← h_share_model(x̂ʲ)         # h, h_x         laserMapping.cpp:638
    δx_j ← x̂ʲ ⊟ x_prop                                       :1652        # (9.2)
    P_j  ← J(δx_j) · P_prop · J(δx_j)ᵀ      # tangent reproj :1659+       # the J/L step
    # solve GN normal equations (9.3) in information form:   :1745+
    K_h  ← (P_j⁻¹ + Hᵀ R⁻¹ H)⁻¹ Hᵀ R⁻¹ r_j                   :1804
    K_x  ← (P_j⁻¹ + Hᵀ R⁻¹ H)⁻¹ Hᵀ R⁻¹ H                     :1809
    δ*   ← K_h + (K_x − I) δx_j                               :1815        # (9.4)
    x̂ʲ⁺¹ ← x̂ʲ ⊞ δ*                                          :1817
    if max_i |δ*_i| < limit_i: mark converged                :1818–1826
    if converged twice or j = max:                           :1834
        Σ_{k|k} ← L(δ*) − K_x[:,0:12] · P[0:12,:]            # posterior, last line
        break
return x̂_{k|k}=x̂ʲ⁺¹, Σ_{k|k}              # → keyframe handoff to back-end §L3
```

Compare this with §12's synthesis of the full estimator step, and with §08's batch
GN of which it is the single-cycle, two-factor special case. The same residuals
($r$, $H$) feed both; only the *scope* of what is jointly optimised differs.

---

### 09.9 Summary

- A recursive filter is the batch MAP solver of §08 under one assumption: the past
  is summarised by a single Gaussian prior, produced by propagating the IMU model
  of §04 (`predict()`, `esekfom.hpp:279`).
- Filtering happens in the **error state** / tangent space (§02). The state DOF is
  the *tangent* dimension ($n=23$ for the IKFoM manifold), not the ambient one
  ($m=24$), because gravity lives on $S^2$.
- The **iterated update** is Gauss–Newton on a two-factor MAP cost (9.1): the
  reprojected prior + the measurement. Its normal equation (9.3) and Kalman form
  (9.4) are implemented verbatim in `update_iterated_dyn_share_modified`
  (`esekfom.hpp:1619`).
- The **$J$/$L$ tangent reprojection** ($P_j = J\,\Sigma\,J^\top$ via the left
  Jacobian `A_matrix`, `esekfom.hpp:1659+`; `A_matrix` defined at
  `mtk/src/mtkmath.hpp:236`) is what makes the *manifold* IEKF exact and
  distinguishes it from a plain ESKF.
- **Bell–Cathey:** the IEKF equals GN on MAP; the EKF is its one-iteration
  truncation; at convergence the filter is the MAP estimate with a Laplace
  posterior covariance.
- **Fixed-lag smoothing** ($1<W<\infty$) is the middle ground: relinearise inside
  a window, marginalise the tail (with FEJ for consistency). Meridian uses $W=1$
  (ESIKF / CT) in the front-end (§L2, §10) and $W=\infty$ (iSAM2) in the back-end
  (§L3); the filter's $\Sigma_{k|k}$ is the information handoff between them.

**See also:** §02 (manifolds, $\boxplus/\boxminus$, $J$), §03 (MAP = NLS, factor
graphs, information form), §04 (IMU model & propagation), §05 (LiDAR
point-to-plane residual & data association), §08 (batch GN/LM, normal equations,
Schur), §10 (continuous-time B-spline front-end), §11 (degeneracy/observability,
robust kernels), §12 (full estimator step, FAST-LIO2/LIVO2 mapping).


---


## 10. Continuous-time estimation: B-spline residuals

> **Where we are.** Sections 04–07 built the individual residuals (IMU §4, LiDAR §5, visual §6, GNSS §7) and §3 cast their fusion as a maximum-a-posteriori (MAP) nonlinear least-squares problem. Sections 08–09 solved that problem in two equivalent ways: batch Gauss–Newton / Levenberg–Marquardt (§8) and the recursive iterated EKF / ESIKF (§9). **All of those formulations share one hidden assumption: that the state is a *discrete* set of variables $\{x_k\}$ sampled at sensor epochs.** This section removes that assumption. We replace the discrete state sequence with a *continuous trajectory* $T(t)\in SE(3)$ represented by a small number of control points $c_k$, and we show how every measurement — at *its own* timestamp — attaches a residual to the handful of control points whose support overlaps that instant. This is the L2 front-end "v2" target in the Meridian roadmap (the continuous-time swap behind the `IFrontEnd` interface), and it is the natural home for an *asynchronous, multi-LiDAR, multi-rate* sensor suite.

We use the shared notation of §2 throughout: $R\in SO(3)$, position $p$, the box-plus $\boxplus$ / box-minus $\boxminus$ operators, the maps $\mathrm{Exp}/\mathrm{Log}$ (capitalised, $\mathbb{R}^3\!\leftrightarrow\! SO(3)$) and $\exp/\log$ (lower-case, $\mathfrak{se}(3)\!\leftrightarrow\! SE(3)$), the hat operator $(\cdot)^\wedge$, residual $r$, Jacobian $H$, information $\Omega$. Control points are written $c_k$ and the trajectory $T(t)$.

---

### 10.1 Why continuous time? Asynchrony and sub-scan motion

#### 10.1.1 The discrete-state pathology

A discrete estimator carries one state $x_k$ per "frame." But what *is* a frame for a spinning LiDAR? An Ouster OS-series sensor sweeps $360^\circ$ over $100\,\mathrm{ms}$ (at $10\,\mathrm{Hz}$). The first and last points of a single "scan" are observed $100\,\mathrm{ms}$ apart, during which a vehicle at $10\,\mathrm{m/s}$ has moved a full metre and may have rotated several degrees. If we pin all of those points to a single state $x_k$, the points are *wrong by the intra-scan motion* — this is the **motion-distortion** (or rolling-shutter) problem. Discrete LIO systems paper over it with **motion compensation / de-skew**: they assume the trajectory inside the scan is known (from IMU propagation) and warp every point into a common reference time *before* forming residuals.

Concretely, FAST-LIO de-skews in `IMU_Processing.hpp` by back-propagating each point through the IMU-integrated pose to the scan-end time; the de-skewed cloud is then treated as if captured *instantaneously*. This is a **two-stage approximation**: (i) the intra-scan trajectory used for de-skew is *frozen* (it is the IMU forward integral, not re-optimised jointly with the registration), and (ii) the registration residual then ignores the per-point time entirely.

Point-LIO takes a different route that is worth dwelling on because it is our explicit contrast. Rather than de-skewing a whole scan and updating once, Point-LIO updates the filter **point by point**, advancing a high-rate kinematic model between points. In `Estimator.cpp` the per-update residual is built over a *time-ordered batch* of points indexed by `time_seq[k]` and a running cursor `idx`:

```cpp
// Point-LIO/src/Estimator.cpp:119
for (int j = 0; j < time_seq[k]; j++) {
    PointType &point_body_j  = feats_down_body->points[idx+j+1];
    PointType &point_world_j = feats_down_world->points[idx+j+1];
    pointBodyToWorld(&point_body_j, &point_world_j);          // uses the CURRENT state
    ...
    if (esti_plane(pabcd, points_near, plane_thr)) { ... }     // point-to-plane (n,d)
}
```

and the plane residual is the point-to-plane distance of §5 (note `normvec->...intensity` stores the plane offset $d$):

```cpp
// Point-LIO/src/Estimator.cpp:210
ekfom_data.z(m) = -norm_vec(0)*...x - norm_vec(1)*...y - norm_vec(2)*...z - normvec->points[j].intensity;
```

Point-LIO never de-skews. Instead it lets the state *evolve continuously between points* through a stochastic differential model, integrated as
$$
\dot p = v,\qquad \dot R = R\,(\omega)^\wedge,\qquad \dot v = R\,a + g,
$$
which appears verbatim as the process function (the "output" variant carries $\omega,a$ as states rather than inputs):

```cpp
// Point-LIO/src/Estimator.cpp:68  get_f_output
vect3 a_inertial = s.rot * s.acc;
res(i)      = s.vel[i];        //  ṗ = v
res(i + 3)  = s.omg[i];        //  the SO(3) tangent rate (ω)
res(i + 12) = a_inertial[i] + s.gravity[i];   //  v̇ = R a + g
```

This is already a *continuous-time process model* — but it is integrated **forward only**, point by point, and the state remains a single $x(t)$ snapshot at the current point's time. Crucially, Point-LIO also treats the IMU as a **measurement of the kinematic state**, not a control input:

```cpp
// Point-LIO/src/Estimator.cpp:327  h_model_IMU_output
ekfom_data.z_IMU.block<3,1>(0,0) = angvel_avr - s.omg - s.bg;                 //  z_gyr = ω_meas − ω − b_g
ekfom_data.z_IMU.block<3,1>(3,0) = acc_avr * G_m_s2/acc_norm - s.acc - s.ba;  //  z_acc = a_meas − a − b_a
```

Hold this thought — the **"IMU as a residual on a derivative of the trajectory"** idea (§10.5) is exactly what continuous-time estimation generalises. Point-LIO is, in a precise sense, the *zeroth-order* continuous-time estimator: it has a continuous process model and per-measurement timestamps, but its "trajectory" is a forward-integrated point estimate rather than a *globally parameterised, jointly optimisable* function of a few control points. The B-spline replaces the forward integral with a smooth function we can differentiate analytically and optimise as a *batch* (§8) or smooth recursively.

#### 10.1.2 What we actually want

A multi-sensor tactical rig (Meridian: several Ouster LiDARs including an upward "dome," a camera, an IMU, GNSS) produces measurements that are:

- **Asynchronous** — each LiDAR phase-offset from the others, the camera shutter at $20\,\mathrm{Hz}$, IMU at $200\,\mathrm{Hz}$, GNSS at $5\,\mathrm{Hz}$, all on a shared PTP clock (L0). There is *no common epoch*.
- **High-rate and continuous** — points arrive at $\sim\!1.3\,\mathrm{M\,pts/s}$ aggregate; manufacturing a discrete state per point (à la Point-LIO) is one valid answer, but it forfeits a smooth, queryable trajectory and couples poorly to a *visual* residual that needs the pose at the exact shutter instant.

The continuous-time (CT) answer is to estimate a **single smooth function** $T:\mathbb{R}\to SE(3)$. Then *any* measurement, from *any* sensor, at *any* timestamp $t$, simply evaluates $T(t)$ (and, if needed, $\dot T(t)$, $\ddot T(t)$) and forms its residual. Asynchrony disappears: there are no frames to align. Sub-scan motion disappears: $T(t)$ is defined *at* each point's time. The IMU becomes a constraint on the *derivatives* of the same function (§10.5). This is the formulation of CLINS (Lv et al., IROS 2021) and Coco-LIC (Lang et al., 2023), and it is what we build as the front-end v2.

> **Cross-reference.** §1 placed methods on the loose/tight coupling spectrum; CT estimation is the *most* tightly-coupled point on it, because every sensor shares one trajectory representation and there is not even a notion of separate per-sensor states to couple.

---

### 10.2 The trajectory representation: cumulative B-splines on $SE(3)$

#### 10.2.1 Splines as a basis for the trajectory

A spline writes the trajectory as a weighted combination of a small set of **control points** $\{c_k\}$ and fixed **basis functions** $B_k(t)$:
$$
\text{(scalar / vector space)}\qquad \mathbf{x}(t) \;=\; \sum_k B_k(t)\, c_k .
$$
We choose **cubic** ($p=3$, order $k=4$) B-splines because they are $C^2$-continuous (continuous position, velocity, *and* acceleration), which is exactly what an inertial residual needs: the IMU measures angular velocity (first derivative of orientation) and specific force (involves the second derivative of position), so the trajectory must be twice differentiable for those residuals to even be defined (§10.5).

B-splines have two properties we exploit relentlessly:

1. **Local support.** A degree-$p$ B-spline basis function is nonzero only over $p+1$ knot intervals. A cubic spline at time $t$ depends on **exactly four** control points $\{c_i, c_{i+1}, c_{i+2}, c_{i+3}\}$ where $i$ is the index of the active knot span. This is the single most important fact in this section: *a measurement at time $t$ creates a residual touching only 4 control points*, so the Jacobian (§8) is extremely sparse, and the normal-equation block structure (§8.3) is banded.
2. **Convex-hull / partition-of-unity.** $\sum_k B_k(t)=1$, which makes the **cumulative** form below well-defined on a manifold.

#### 10.2.2 Why *cumulative*, and why on the manifold

We cannot naively write $T(t)=\sum_k B_k(t)\,c_k$ when $c_k\in SE(3)$ — a weighted sum of rotation matrices is not a rotation. The standard fix (Kim et al. 2004; Sommer et al. 2020; used by CLINS and Coco-LIC) is the **cumulative B-spline**, which sums *increments* in the Lie algebra. Define the cumulative basis
$$
\tilde B_k(t) \;=\; \sum_{j\ge k} B_j(t),\qquad \tilde B_0(t)=1 .
$$
Then the trajectory on a Lie group $G$ (here $SE(3)$, or we split it as $SO(3)\times\mathbb{R}^3$) is the *left-cumulative product*
$$
\boxed{\;T(t) \;=\; c_i \,\prod_{j=1}^{p}\, \exp\!\Big(\tilde B_{\,i+j}(t)\;\underbrace{\log\!\big(c_{i+j-1}^{-1}\,c_{i+j}\big)}_{=\;d_{i+j}\;\in\;\mathfrak{g}}\Big)\;}
\tag{10.1}
$$
where $i$ is the active span and $d_{i+j} := \log(c_{i+j-1}^{-1} c_{i+j})$ is the *relative motion* between consecutive control points, expressed in the tangent space. Each factor blends in a fraction $\tilde B_{i+j}(t)\in[0,1]$ of that relative motion. At a knot, the cumulative weights are $0$ or $1$ and $T$ passes (approximately, for splines; exactly for the corresponding B-spline interpolation) through the control values. This is precisely the construction CLINS uses for its $SE(3)$ trajectory and Coco-LIC for its decoupled $SO(3)\times\mathbb{R}^3$ trajectory.

> **Notation tie-in (§2).** $\log(c_{i+j-1}^{-1}c_{i+j})$ is the box-minus $c_{i+j}\boxminus c_{i+j-1}$ of §2 evaluated in the group's tangent space; $\exp(\cdot)$ is the box-plus increment. So (10.1) is literally a chained box-plus of weighted box-minus increments — the manifold generalisation of "interpolate between samples."

#### 10.2.3 Uniform vs **non-uniform** knots

If the knots $\{t_k\}$ are equally spaced (spacing $\Delta t$), the basis $B_k(t)$ is a single fixed cubic polynomial in the local parameter $u=(t-t_i)/\Delta t\in[0,1)$, and the cumulative weights have the closed form (the $C^2$ cubic "Cox–de Boor in matrix form," Qin 2000)
$$
\begin{pmatrix}\tilde B_{i}\\ \tilde B_{i+1}\\ \tilde B_{i+2}\\ \tilde B_{i+3}\end{pmatrix}
=\;
\underbrace{\frac{1}{6}\begin{pmatrix}6&0&0&0\\ 5&3&-3&1\\ 1&3&3&-2\\ 0&0&0&1\end{pmatrix}}_{\tilde M^{(4)}}
\begin{pmatrix}1\\ u\\ u^2\\ u^3\end{pmatrix},
\tag{10.2}
$$
so $\tilde B_i\equiv 1$ and only the lower three rows vary. The matrix form makes evaluation and differentiation a couple of small matrix–vector products — cheap and constant-time.

For Meridian we want **non-uniform** knots, for two reasons CLINS and Coco-LIC both exploit:

1. **Adaptive density.** During aggressive motion (the rig is "tactical," so expect shocks, sharp turns) we want *more* control points to capture the wiggle; during a slow traverse, *fewer*, to keep the problem small. Coco-LIC's central idea is exactly this: place control points *non-uniformly* in time, denser where motion (and information) is richer, to keep accuracy high without blowing up the state dimension. This is the "non-uniform" in our mandate.
2. **Phase-locking to keyframes.** It is convenient to place knots near the keyframe times the L3 back-end consumes.

For non-uniform knots the closed-form (10.2) no longer holds; one uses the general **De Boor–Cox recursion** for the (cumulative) basis, or precomputes a per-span blending matrix $\tilde M^{(4)}_i$ from the local knot vector (Sommer et al. 2020 give the recursive blending-matrix construction that we adopt). The *structure* — four control points, $C^2$, local support — is unchanged; only the numeric weights become span-dependent.

```
            knots:  t_i   t_{i+1}   t_{i+2}   t_{i+3}   t_{i+4}
                     |       |         |         |         |
 control points:    c_i     c_{i+1}   c_{i+2}   c_{i+3}   c_{i+4}
                     \________ active span for t ________/
                              ^
   measurement at t ----------+   ->  residual touches {c_i,c_{i+1},c_{i+2},c_{i+3}}
```

---

### 10.3 Evaluating the spline and its derivatives

Every residual needs $T(t)$; the inertial residual additionally needs $\dot T(t)$ and $\ddot T(t)$. We give the formulas, then their Jacobians w.r.t. control points (the quantities §8 stacks into $H$).

#### 10.3.1 Value

Split $T(t)=(R(t),p(t))$. For the rotation factor of (10.1), writing $\Omega_j := \log\!\big(R_{i+j-1}^{-1}R_{i+j}\big)\in\mathbb{R}^3$ and the cumulative weight $\lambda_j(t):=\tilde B_{i+j}(t)$,
$$
R(t) \;=\; R_i \prod_{j=1}^{3}\, \mathrm{Exp}\!\big(\lambda_j(t)\,\Omega_j\big),
\tag{10.3}
$$
and for position (treating $\mathbb{R}^3$ as its own abelian Lie group, where $\exp/\log$ are identity and the cumulative product becomes a cumulative sum) the equivalent
$$
p(t) \;=\; p_i \;+\; \sum_{j=1}^{3}\, \lambda_j(t)\,(p_{i+j}-p_{i+j-1}).
\tag{10.4}
$$
(Eq. 10.4 is what Coco-LIC uses for its decoupled translation spline; an $SE(3)$-coupled spline instead carries position inside the same $\exp(\cdot)$ as a 6-vector twist, at the cost of pose–translation cross terms.)

#### 10.3.2 Derivatives (velocity, angular velocity, acceleration)

Differentiate (10.4) in time. With $\dot\lambda_j := \mathrm d\lambda_j/\mathrm dt$ (obtained by differentiating (10.2)/the blending matrix: replace $(1,u,u^2,u^3)^\top$ by $\tfrac1{\Delta t}(0,1,2u,3u^2)^\top$),
$$
v(t)=\dot p(t)=\sum_{j=1}^{3}\dot\lambda_j(t)\,(p_{i+j}-p_{i+j-1}),
\qquad
a^{\text{kin}}(t)=\ddot p(t)=\sum_{j=1}^{3}\ddot\lambda_j(t)\,(p_{i+j}-p_{i+j-1}).
\tag{10.5}
$$
The rotation is harder because (10.3) is a product of matrix exponentials. Differentiating the product (chain rule on $SO(3)$) gives the **body angular velocity** as a sum over the three factors; defining the running partial products $A_j:=\mathrm{Exp}(\lambda_j\Omega_j)$ and the right-Jacobian $J_r$ of §2,
$$
\omega(t) \;=\; R(t)^\top \dot R(t)
\;=\; \sum_{j=1}^{3}\, \mathrm{Ad}_{(A_{j+1}\cdots A_3)^{-1}}\;\big(\dot\lambda_j(t)\,\Omega_j\big)
\quad(\text{plus higher-order } J_r \text{ couplings}).
\tag{10.6}
$$
In practice one uses the compact recursive expressions of Sommer et al. (2020, "Efficient Derivatives of B-Spline ... on Lie Groups") or the $\mathfrak{se}(3)$ closed forms in CLINS, which give $\dot R$, $\ddot R$ exactly without finite differencing. The *interface* the residuals see is simply:
$$
\texttt{eval}(t)\;\longrightarrow\;\big(R(t),\,p(t),\,\omega(t),\,v(t),\,a^{\text{kin}}(t)\big),
$$
plus, on request, the Jacobians of each of these w.r.t. the four active control points.

#### 10.3.3 A small worked example (scalar $C^2$ check)

Take a uniform cubic spline with $\Delta t = 0.1\,\mathrm{s}$ and ask for the *velocity weighting* at the span midpoint $u=0.5$. Differentiating row 2 of $\tilde M^{(4)}$ (Eq. 10.2), $\tilde B_{i+1}(u)=\tfrac16(5+3u-3u^2+u^3)$, so $\dot{\tilde B}_{i+1}(u)=\tfrac1{6\Delta t}(3-6u+3u^2)$. At $u=0.5$: $\dot{\tilde B}_{i+1}=\tfrac1{0.6}(3-3+0.75)=1.25\ \mathrm{s^{-1}}$. A control-point displacement of $1\,\mathrm{cm}$ in $(p_{i+1}-p_i)$ therefore contributes $0.0125\,\mathrm{m/s}$ to $v(t)$ at that instant — a concrete, checkable number, and exactly the partial derivative $\partial v/\partial(\Delta p)$ that enters the velocity/IMU Jacobian.

---

### 10.4 Attaching a measurement: the LiDAR point-to-plane residual in CT form

Now the payoff. Take the **point-to-plane** LiDAR residual of §5 — the *same* geometric residual Point-LIO forms in `Estimator.cpp:140` — but evaluate the pose at the point's *own* timestamp.

A LiDAR point $p_L$ is observed at time $t_p$ (Ouster stamps every point; on the PTP clock of L0 this is exact). Let $(R(t_p),p(t_p))$ be the spline pose at that instant, and ${}^{I}T_L=(R_{IL},p_{IL})$ the (possibly online-estimated, §"calibration") LiDAR-to-IMU extrinsic. The point in the world/map frame is
$$
p_W(t_p) \;=\; R(t_p)\,\big(R_{IL}\,p_L + p_{IL}\big) + p(t_p).
\tag{10.7}
$$
Against a local plane with unit normal $n$ and offset $d$ (so the plane is $n^\top x + d = 0$; §5), the scalar residual is
$$
\boxed{\; r_{\text{lidar}} \;=\; n^\top p_W(t_p) + d \;}
\tag{10.8}
$$
which is *identical in form* to Point-LIO's `z(m)` (`Estimator.cpp:210`, up to sign convention since Point-LIO stores $-z$) and to FAST-LIO Eq. (12)–(14). The **only** difference is that $R(t_p),p(t_p)$ come from (10.3)/(10.4) instead of a single state. The chain rule then routes the geometric Jacobian onto the control points.

**Jacobian.** By the chain rule, for each active control point $c_{i+m}$ ($m=0,1,2,3$),
$$
\frac{\partial r_{\text{lidar}}}{\partial c_{i+m}}
\;=\;
n^\top\frac{\partial p_W(t_p)}{\partial c_{i+m}}
\;=\;
\underbrace{n^\top}_{1\times3}
\Big[\;
\underbrace{\frac{\partial p_W}{\partial R(t_p)}}_{\text{rotation part}}\frac{\partial R(t_p)}{\partial c_{i+m}}
\;+\;
\underbrace{\frac{\partial p_W}{\partial p(t_p)}}_{=\,I_3}\frac{\partial p(t_p)}{\partial c_{i+m}}
\;\Big].
\tag{10.9}
$$
The two geometric blocks are the *standard* point-to-plane terms of §5 — and they are exactly the quantities Point-LIO assembles:

```cpp
// Point-LIO/src/Estimator.cpp:206  (no-extrinsic branch)
V3D C(s.rot.transpose() * norm_vec);      //  C = Rᵀ n
V3D A(point_crossmat * C);                //  A = (p_L)^∧ Rᵀ n   -> ∂r/∂(rotation)
ekfom_data.h_x.block<1,12>(m,0) << norm_vec(0..2), VEC_FROM_ARRAY(A), 0...;  // [ ∂r/∂p | ∂r/∂R | ... ]
```

Here `norm_vec` $=n$ is $\partial r/\partial p$, and $A=(p_L)^\wedge R^\top n$ is (a parameterisation of) $\partial r/\partial R$. In the CT version we take **these same two $1\times3$ blocks** and post-multiply them by the spline sensitivities $\partial R(t_p)/\partial c_{i+m}$ and $\partial p(t_p)/\partial c_{i+m}$ from §10.3 — for the position part, $\partial p(t_p)/\partial c_{i+m}$ is read straight off (10.4) as $\pm\lambda_m(t_p)\,I_3$ (telescoping signs), and for the rotation part one uses the $\mathrm{Ad}/J_r$ terms behind (10.6). The result is a $1\times 24$ row Jacobian (four $SE(3)$ control points $\times 6$) — sparse, with **all other control points zero**. If the extrinsic is being estimated online (§"calibration"; cf. Point-LIO's `extrinsic_est_en` branch at `Estimator.cpp:191`, with its extra `A`,`B`,`C` blocks for $R_{IL},p_{IL}$), three more $1\times6$ blocks append.

> **Where the noise comes from.** §11 (observability/degeneracy → back-end noise) plugs in here: the per-residual information $\Omega_{\text{lidar}}$ is the inverse of the point measurement covariance (`ekfom_data.M_Noise = laser_point_cov`, `Estimator.cpp:178`), optionally inflated by the X-ICP/D2-LIO per-axis degeneracy weighting before the residual enters the normal equations of §8.

---

### 10.5 The IMU as a residual on the trajectory's *derivatives*

This is the conceptual heart of CT estimation and the cleanest contrast with the discrete pipelines of §4 and §9.

In a discrete ESIKF (§9, FAST-LIO) the IMU is the **process input**: it *propagates* the state forward and contributes to the *prior*, not to a measurement residual. In Point-LIO the IMU is instead a **direct measurement of the state's kinematic variables** (`h_model_IMU_output`, `Estimator.cpp:327`): the gyro reads angular velocity, the accelerometer reads specific force, and the residual is `z = measurement − (state + bias)`. CT estimation takes the *same* "IMU as measurement" stance as Point-LIO — but the predicted quantities are *analytic derivatives of the spline* rather than separate state variables.

At each IMU timestamp $t_m$, the spline predicts angular velocity $\omega(t_m)$ from (10.6) and linear acceleration $a^{\text{kin}}(t_m)$ from (10.5). The IMU model of §4 then gives two residuals:
$$
\boxed{\;
r_{\text{gyr}} = \tilde\omega_m - \big(\omega(t_m) + b_g\big),
\qquad
r_{\text{acc}} = \tilde a_m - \big(R(t_m)^\top\!\big(a^{\text{kin}}(t_m) - g\big) + b_a\big)
\;}
\tag{10.10}
$$
where $\tilde\omega_m,\tilde a_m$ are the raw gyro/accel readings and $b_g,b_a$ are the (slowly time-varying) biases. Compare the gyro line **directly** to Point-LIO:

```cpp
// Point-LIO/src/Estimator.cpp:327
ekfom_data.z_IMU.block<3,1>(0,0) = angvel_avr - s.omg - s.bg;   //  ω_meas − ω − b_g
```

Point-LIO's `s.omg` is a *free state variable*; CT's $\omega(t_m)$ is the *spline derivative* $\omega(t_m)=\sum_j \mathrm{Ad}_{(\cdot)}(\dot\lambda_j\Omega_j)$ from (10.6). The accelerometer line is the same story: Point-LIO's `acc_avr*G/acc_norm − s.acc − s.ba` (`Estimator.cpp:328`) compares the reading to a free `s.acc`; CT compares it to $R^\top(a^{\text{kin}}-g)$, where $a^{\text{kin}}=\ddot p(t_m)$ from (10.5). **The IMU residual literally constrains the curvature of the spline.** This is why we needed $C^2$ continuity (§10.2.1): $\ddot p$ must exist and be continuous for (10.10) to be a well-posed, smooth residual.

The Jacobians of (10.10) w.r.t. the control points are obtained by chaining $\partial\omega/\partial c$, $\partial a^{\text{kin}}/\partial c$, $\partial R/\partial c$ from §10.3 — again touching only the four active control points, plus a trivial identity block on the biases. The biases themselves are modelled, exactly as in §4 and Point-LIO's `process_noise_cov_output` (`Estimator.cpp:49`, the `b_gyr_cov`/`b_acc_cov` random-walk terms), as a slowly varying random walk and may be represented either as additional low-rate spline(s) or as piecewise-constant variables per knot interval (CLINS uses the latter).

> **Gravity and the bias prior.** As in §4/§7, $g$ is either a known constant in a gravity-aligned world frame or a 2-DoF $S^2$ state; Point-LIO carries it as a state (`s.gravity`, used in `get_f_output`, `Estimator.cpp:75`), and the CT estimator does likewise.

---

### 10.6 The visual residual in CT form (FAST-LIVO2 photometric)

The camera shutter fires at $t_c$; the visual residual of §6 is the *sparse-direct photometric* error — comparing the live image intensity at a projected map point to a reference patch (FAST-LIVO2, §III; paper 2408.14035). In CT form a 3-D map point $P_W$ (whose depth came from the LiDAR map, §6/FAST-LIVO2) projects through the spline pose at the *exact* shutter time:
$$
u = \pi\!\Big(K_{\text{cam}},\; {}^{C}T_I\,\big(R(t_c)^\top(P_W - p(t_c))\big)\Big),
\qquad
r_{\text{photo}} = I\big(u\big) - I_{\text{ref}}\big(u_{\text{ref}}\big),
\tag{10.11}
$$
with $\pi$ the projection and $K_{\text{cam}}$ the intrinsics (§2 notation; ${}^{C}T_I$ the camera–IMU extrinsic). The Jacobian chains the image gradient $\nabla I$ (the FAST-LIVO2 photometric Jacobian, paper Eq. for the patch residual) through $\partial\pi/\partial P_C$ and then through $\partial R(t_c)/\partial c$, $\partial p(t_c)/\partial c$ from §10.3 — once more onto four control points. **The crucial CT advantage for vision:** a rolling-shutter or simply *asynchronous* camera no longer needs its pose interpolated *ad hoc* from neighbouring LiDAR states; $T(t_c)$ is exact and shares control points with the LiDAR and IMU residuals, so the photometric, geometric, and inertial errors are *jointly* minimised over the same trajectory. This is the tight LiDAR-inertial-visual coupling of FAST-LIVO2, lifted onto a continuous trajectory.

---

### 10.7 The full CT problem: one factor graph over control points

Stacking all residuals over a sliding window of control points $\mathcal{C}=\{c_k\}$ (plus biases and online extrinsics), the MAP estimate of §3 is the weighted nonlinear least-squares
$$
\mathcal{C}^\star=\arg\min_{\mathcal{C}}\;
\sum_{p}\big\|r_{\text{lidar}}(t_p)\big\|^2_{\Omega_L}
+\sum_{m}\big\|r_{\text{imu}}(t_m)\big\|^2_{\Omega_I}
+\sum_{c}\big\|r_{\text{photo}}(t_c)\big\|^2_{\Omega_V}
+\sum_{g}\big\|r_{\text{gnss}}(t_g)\big\|^2_{\Omega_G}
+\underbrace{\|r_{\text{prior}}\|^2}_{\text{marginalisation}}.
\tag{10.12}
$$
Three structural facts make this practical:

- **Banded normal equations.** Because each residual touches only 4 consecutive control points (§10.2), the Hessian $H^\top\Omega H$ of §8 is *block-banded* (bandwidth $= p\cdot\dim SE(3)$). Sparse Cholesky / the Schur complement of §8.3 solve it in time linear in the window length. This is the same sparsity that makes GTSAM's iSAM2 (our L3 back-end) efficient; CT just produces a particularly clean band.
- **GNSS / absolute residuals** (§7) attach at $t_g$ as $r_{\text{gnss}}=p(t_g)\boxminus z^{\text{ENU}}_g$ in the ENU frame, with the *switchable* constraint of §7/§11 multiplying $\Omega_G$ — again onto 4 control points.
- **Robustness** (§11): each summand wears a robust kernel (Huber/GNC), and per-axis observability scales $\Omega_L$. Nothing about CT changes the robustness machinery; it operates on residuals (10.8), (10.10), (10.11) identically.

A single **CT estimator step** (the analogue of §12's discrete step) is then:

1. **Slide the window**: drop control points older than the lag, *spawn* new control points to cover incoming measurement times (non-uniform: spawn density follows motion/Coco-LIC heuristic, §10.2.3).
2. **For each measurement** (any sensor, sorted by PTP time): find the active span $i$, evaluate $T(t),\dot T(t),\ddot T(t)$ and the control-point Jacobians (§10.3), form the residual (§10.4–10.6), scatter the $1\times24$(+extrinsic) row into the banded $H$.
3. **Solve** the (robustified) normal equations (§8) — a few Gauss–Newton / LM iterations.
4. **Marginalise** the oldest control points into a prior factor (fixed-lag smoothing, §9.4), preserving information for the back-end.
5. **Emit keyframes** (poses $T(t_{kf})$ sampled from the spline) to the L3 iSAM2 graph and the L4 map.

> **Cross-reference.** §9 showed the iterated EKF is equivalent to one Gauss–Newton solve of the discrete MAP. The CT estimator is *the same equivalence one level up*: a fixed-lag CT smoother is a Gauss–Newton solve of (10.12) over control points, and a "filtering" CT variant simply marginalises after each step. The choice between recursive (§9) and batch (§8) solvers is orthogonal to whether the state is discrete or continuous.

---

### 10.8 CT vs. Point-LIO: when to use which (design note for Meridian)

| | **Point-LIO (point-wise discrete)** | **Continuous-time (B-spline)** |
|---|---|---|
| Trajectory | forward-integrated point estimate $x(t_p)$, one per point (`Estimator.cpp:68`) | global function $T(t)$ over control points (10.1) |
| Sub-scan motion | handled by per-point process integration (no de-skew) | handled by evaluating $T(t_p)$ at each point's time |
| IMU | measurement of state kinematics (`Estimator.cpp:327`) | measurement of spline derivatives (10.10) |
| Asynchronous multi-sensor | one filter, points serialised by time | native: every sensor queries the same $T(t)$ |
| Solver | iterated EKF, per-point update (§9) | batch GN/LM or fixed-lag smoother (§8/§9) |
| Cost | $O(\text{points})$ tiny updates; high update rate | $O(\text{window})$ banded solve; fewer, larger solves |
| Queryable smooth pose for camera | requires interpolation | exact $T(t_c)$ by construction |

The Meridian plan exploits both: **v1** is an iEKF (FAST-LIO2 style, §9/§12) for fast bring-up; **v2** is this CT estimator dropped in behind the same `IFrontEnd` interface, chosen because the multi-LiDAR + dome + camera + GNSS suite is *fundamentally asynchronous* and a continuous trajectory is the cleanest substrate for it. Point-LIO is the instructive midpoint: it proves the "IMU-as-measurement, no de-skew" idea works *without* a spline, and its `h_model_*` residual assembly (`Estimator.cpp:112–322`) is exactly the geometric kernel we reuse — only the pose source changes from a single state to $T(t)$.

---

### 10.9 Summary

- **Continuous time** replaces a discrete state sequence with a smooth trajectory $T(t)\in SE(3)$ built from a few control points via **cumulative B-splines** (10.1); cubic ($C^2$) so inertial residuals (which need $\ddot p$) are well defined.
- **Local support** ⇒ a measurement at any time $t$ touches **exactly four** control points ⇒ a sparse, banded least-squares (§8) over the window.
- Each sensor residual is the *same* geometric/photometric/inertial error of §§5–7, only evaluated at the measurement's own PTP timestamp via $T(t)$ and its derivatives (§10.3): point-to-plane (10.8), IMU-on-derivatives (10.10), FAST-LIVO2 photometric (10.11), GNSS (§7).
- **Non-uniform** knots (Coco-LIC) place control points where the motion is rich, keeping the state small.
- **Point-LIO** is the discrete contrast that already does "IMU-as-measurement, no de-skew" point-wise (`Estimator.cpp`); CT generalises its forward integral into a globally optimisable function and dissolves sensor asynchrony.

**Forward pointer.** §11 hangs the robustness machinery (degeneracy weighting, GNC, switchable GNSS, PCM) on these CT residuals unchanged; §12 assembles a complete estimator step and maps it back to FAST-LIO2 / FAST-LIVO2, of which the CT formulation here is the asynchronous, sub-scan-exact generalisation.

---

#### Sources grounded in this section

- **Code (read on disk):** `C:/Users/charl/Sources/slam-reference/Point-LIO/src/Estimator.cpp` (process model `get_f_output` L68–78; point-to-plane assembly `h_model_*` L112–322, esp. residual L210 and Jacobian blocks `C=Rᵀn`/`A=(p)^∧C` L206–208; **IMU-as-measurement** `h_model_IMU_output` L327–328; extrinsic branch L191–201; noise `laser_point_cov` L178, bias random-walk L49) and `Estimator.h` (the `state_input`/`state_output` 24/30-dim states, L41–57). FAST-LIO de-skew: `FAST_LIO/src/IMU_Processing.hpp` (back-propagation motion compensation).
- **Papers:** FAST-LIO2 (2107.06829), FAST-LIO (2010.08196, point-to-plane Eq. 12–14), FAST-LIVO2 (2408.14035, sparse-direct photometric residual). CT B-spline trajectory: **CLINS** (Lv, Lang, Xu, Kong, Liu, IROS 2021), **Coco-LIC** (Lang et al., 2023 — non-uniform control-point placement), cumulative $SE(3)$ B-splines (Kim et al. 2004; Sommer, Usenko, et al. 2020, efficient Lie-group spline derivatives), De Boor–Cox / matrix-form basis (Qin 2000).


---


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


---


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


---
