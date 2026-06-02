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
