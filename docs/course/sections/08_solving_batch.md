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
