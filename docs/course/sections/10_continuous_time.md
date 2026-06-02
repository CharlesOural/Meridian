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
