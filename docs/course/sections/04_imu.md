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
