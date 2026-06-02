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
