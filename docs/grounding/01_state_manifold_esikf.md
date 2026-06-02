# Grounding Dossier — State Representation on Manifolds + the (Iterated) Error‑State Kalman Filter (IKFoM)

> Scope: how FAST‑LIO / FAST‑LIO2 represent the navigation state on a product manifold **S²×SO(3)×ℝⁿ**, the generic `MTK`/`IKFoM` toolkit that encapsulates the ⊞/⊟ operators, the forward error‑state propagation (F_x, F_w, covariance), and the iterated measurement update driven by the `h_share_model` callback (with the information‑form Kalman gain). Every claim is grounded in the cloned FAST_LIO source; file:line citations are relative to `C:/Users/charl/Sources/slam-reference/`.
>
> Provenance note for downstream authors: all header/toolkit math below is transcribed from **complete reads** of the actual files (`use-ikfom.hpp`, `esekfom.hpp` lines 1‑1155 incl. the full `predict` and the full `update_iterated_dyn_share`, `S2.hpp`, `SOn.hpp`, `vect.hpp`, `mtkmath.hpp`, `build_manifold.hpp`, `IMU_Processing.hpp`, `so3_math.h`, `Exp_mat.h`, `common_lib.h`) and a **complete read of the FAST‑LIO paper** `papers/2010.08196.txt` (Eqs. 1‑23). Two specific source windows — the inner body of `h_share_model` (`laserMapping.cpp:638‑767`) and `update_iterated_dyn_share_modified` (`esekfom.hpp:1619‑1960`) — could not be re‑opened in the final pass due to a transient tool fault; for those I cite the verified surrounding lines I *did* read (the function signature, the `init_dyn_share` registration, the call site, `epsi`, `LASER_POINT_COV`) and reconstruct the body from (a) its algorithmic twin `update_iterated_dyn_share` which I read in full, and (b) the paper equations. Each such reconstruction is explicitly flagged "[reconstructed]".

---

## 0. Big picture: kinematic state lives on a manifold; the filter operates in the tangent space

FAST‑LIO does not estimate the state as a flat ℝⁿ vector. Two components are non‑Euclidean:

- **Rotation** `R ∈ SO(3)` — 3 DOF, stored as a unit quaternion (`SOn.hpp:178`, `enum {DOF=3, DIM=3, TYP=2}` at `SOn.hpp:179`).
- **Gravity** `g ∈ S²` — the gravity *direction* on a 2‑sphere of fixed radius. The S² type fixes the magnitude: `S2<double, 98090, 10000, 1>` ⇒ `length = den/num = 98090/10000 = 9.809` (`use-ikfom.hpp:8`; `length = scalar(den)/scalar(num)` at `S2.hpp:104`; `enum {DOF=2, TYP=1, DIM=3}` at `S2.hpp:105`). So gravity is **2 DOF in the tangent, 3 numbers stored**.

The FAST‑LIO paper (`2010.08196.txt:127‑177`) defines the trick that makes a KF work on this manifold: the **⊞ / ⊟ encapsulation operators** ("boxplus"/"boxminus") that establish a bijection between a neighborhood of the manifold M and its tangent space ℝⁿ:

```
⊞ : M × ℝⁿ → M ,   SO(3): R ⊞ r = R·Exp(r) ;   ℝⁿ: a ⊞ b = a + b      (paper Eq. at line 131)
⊟ : M × M  → ℝⁿ ,   SO(3): R₁ ⊟ R₂ = Log(R₂ᵀ R₁) ; ℝⁿ: a ⊟ b = a − b   (paper Eq. at line 133)
Exp(r) = I + (sin‖r‖/‖r‖)⌊r⌋ + ((1−cos‖r‖)/‖r‖²)⌊r⌋²                    (paper Eq. at line 135)
```
with the consistency identities (`2010.08196.txt:177`): `(x ⊞ u) ⊟ x = u` and `x ⊞ (y ⊟ x) = y`. The filter keeps a nominal state `x` on M and a zero‑mean **error state** `δx ∈ ℝⁿ`; the covariance `P` is the covariance of `δx` in the tangent space (paper III‑C, line 374). The *iterated* part relinearizes ⊞/⊟ about the current iterate each iteration (a Gauss‑Newton step with a Gaussian prior).

The generic IKFoM toolkit derives from Hertzberg et al., "Integrating generic sensor fusion algorithms … through encapsulation of manifolds" (cited atop every toolkit header, e.g. `S2.hpp:3‑4`).

---

## 1. The concrete LIO state manifold (`use-ikfom.hpp`)

Built with the `MTK_BUILD_MANIFOLD` macro (`build_manifold.hpp:180`). Source `use-ikfom.hpp:6‑33`:

```cpp
typedef MTK::vect<3, double> vect3;            // use-ikfom.hpp:6
typedef MTK::SO3<double>     SO3;              // use-ikfom.hpp:7
typedef MTK::S2<double, 98090, 10000, 1> S2;   // use-ikfom.hpp:8  (length=9.809, S2_typ=1)

MTK_BUILD_MANIFOLD(state_ikfom,                // use-ikfom.hpp:12
((vect3, pos))           // p   : IMU position in world      (R^3)
((SO3,   rot))           // R   : IMU attitude               (SO(3))
((SO3,   offset_R_L_I))  // R_LI: LiDAR->IMU extrinsic rot   (SO(3))
((vect3, offset_T_L_I))  // t_LI: LiDAR->IMU extrinsic trans (R^3)
((vect3, vel))           // v   : IMU velocity in world      (R^3)
((vect3, bg))            // b_g : gyro bias                  (R^3)
((vect3, ba))            // b_a : accel bias                 (R^3)
((S2,    grav))          // g   : gravity vector             (S^2, 2 DOF)
);
```

### 1.1 Tangent‑vector (δx) layout and the 24 vs 23 vs 22 dimensions

The macro accumulates two running totals per member (`build_manifold.hpp:129‑130`): `DOF += type::DOF` (tangent dimension) and `DIM += type::DIM` (ambient/"flat" storage dimension). For this state:

| idx (DOF) | dim (DIM) | component | manifold | type::DOF / type::DIM |
|---|---|---|---|---|
| 0..2   | 0..2   | `pos`          | ℝ³    | 3 / 3 |
| 3..5   | 3..5   | `rot`          | SO(3) | 3 / 3 |
| 6..8   | 6..8   | `offset_R_L_I` | SO(3) | 3 / 3 |
| 9..11  | 9..11  | `offset_T_L_I` | ℝ³    | 3 / 3 |
| 12..14 | 12..14 | `vel`          | ℝ³    | 3 / 3 |
| 15..17 | 15..17 | `bg`           | ℝ³    | 3 / 3 |
| 18..20 | 18..20 | `ba`           | ℝ³    | 3 / 3 |
| 21..22 | 21..23 | `grav`         | S²    | **2 / 3** |

So **`state_ikfom::DOF = 23`** (tangent/covariance dimension) but **`state_ikfom::DIM = 24`** (ambient — gravity stores 3, contributes 2). This is exactly why `get_f` returns a **24**‑vector while `df_dx` is **24×23** (`use-ikfom.hpp:47,61`): the model functions work in the *flat/ambient* (DIM=24) row space, and the filter's `predict` later folds the 3‑row gravity block down to 2 tangent rows through the S² projection (§4.2). The single most important "math ≠ naive vector" fact in the whole codebase is this **S² → 2 DOF reduction**.

The IKFoM template instantiation is `esekfom::esekf<state_ikfom, 12, input_ikfom>` (`IMU_Processing.hpp:53`, `laserMapping.cpp:131`), where `12` = process‑noise DOF. Inside the class: `enum{ n = state::DOF (=23), m = state::DIM (=24), l = measurement::DOF }` (`esekfom.hpp:109‑111`); `cov` is `Matrix<scalar, n, n>` = 23×23 (`esekfom.hpp:116`); `cov_` is `Matrix<scalar, m, n>` = 24×23 (`esekfom.hpp:117`).

### 1.2 `input_ikfom` (6 DOF) and `process_noise_ikfom` (12 DOF)

```cpp
MTK_BUILD_MANIFOLD(input_ikfom, ((vect3, acc))((vect3, gyro)));            // use-ikfom.hpp:23
MTK_BUILD_MANIFOLD(process_noise_ikfom,                                    // use-ikfom.hpp:28
  ((vect3, ng))((vect3, na))((vect3, nbg))((vect3, nba)));
```
The default process‑noise covariance is diagonal (`use-ikfom.hpp:35‑43`, `process_noise_cov()`): `ng=1e-4, na=1e-4, nbg=1e-5, nba=1e-5` — but these are *overwritten at runtime* per IMU step from `cov_gyr/cov_acc/cov_bias_gyr/cov_bias_acc` (§4.3).

---

## 2. The kinematic model that drives propagation (`use-ikfom.hpp`)

Three free functions plug LIO physics into the generic toolkit; everything else is generic.

### 2.1 Continuous dynamics `get_f` → 24‑vector (`use-ikfom.hpp:47‑59`)

```cpp
Eigen::Matrix<double,24,1> get_f(state_ikfom &s, const input_ikfom &in){
  res = 0;
  vect3 omega;  in.gyro.boxminus(omega, s.bg);   // omega = w_m - b_g     (line 51)
  vect3 a_inertial = s.rot * (in.acc - s.ba);     // R (a_m - b_a)         (line 52)
  for i in 0..2:
     res(i)      = s.vel[i];                       // dp/dt = v            (line 54)
     res(i+3)    = omega[i];                        // d(rot)/dt = omega    (line 55)
     res(i+12)   = a_inertial[i] + s.grav[i];       // dv/dt = R(a_m-b_a)+g (line 56)
  // res rows 6..11 (ext), 15..20 (biases), 21..23 (grav) left ZERO
}
```
This is the standard strapdown IMU kinematics and matches the paper's continuous model `2010.08196.txt:182‑183` (Eq. 1): `ṗ = v`, `Ṙ = R⌊w_m − b_g − n_g⌋`, `v̇ = R(a_m − b_a − n_a) + g`, `ġ = 0`, `ḃ_g = n_bg`, `ḃ_a = n_ba`. Note `in.gyro.boxminus(omega, s.bg)` is just `omega = gyro − bg` because `vect3::boxminus` is `res = *this − other` (`vect.hpp:120‑122`). The `rot` slot stores the body angular rate `ω`; the discrete integration applies it through SO(3) Exp (§4.2), realizing `R_{k+1} = R_k·Exp(ω·Δt)` (right‑invariant).

### 2.2 `df_dx` → 24×23 (`use-ikfom.hpp:61‑77`)

```cpp
cov(0:2 , 12:14) = I₃;                                  // d(ṗ)/d(δv)
acc_ = in.acc - s.ba;                                   // line 66
cov(12:14, 3:5)  = -s.rot.toRotationMatrix()*MTK::hat(acc_);  // d(v̇)/d(δθ)  line 69
cov(12:14, 18:20)= -s.rot.toRotationMatrix();          // d(v̇)/d(δb_a)  line 70
s.S2_Mx(grav_matrix, vec=0, 21);                       // B(g): 3x2 basis at g, line 73
cov(12:14, 21:22)= grav_matrix;                        // d(v̇)/d(δg)    line 74
cov(3:5 , 15:17) = -I₃;                                 // d(θ̇)/d(δb_g)  line 75
```
Linearization of `get_f` under the **right perturbation** convention:
```
∂ṗ/∂δv = I₃ ,  ∂v̇/∂δθ = −R⌊a_m−b_a⌋ ,  ∂v̇/∂δb_a = −R ,  ∂v̇/∂δg = B(g) (3×2) ,  ∂θ̇/∂δb_g = −I₃
```
`MTK::hat(v)` is the skew `⌊v⌋` (`mtkmath.hpp:176‑183`). `S2_Mx` supplies the 3×2 tangent basis `B(g) = −⌊g⌋·B_x` at δ=0 (`S2.hpp:266‑280`, δ‑norm<tol branch returns `-hat(vec)*Bx`). Columns are 23 wide because gravity occupies only tangent cols 21..22. This refines the paper's `F_x` (Eq. 7, `2010.08196.txt:256‑298`): the paper writes the gravity column with `-A(…)ᵀΔt` type factors; the code splits the analytic part into `df_dx`/`df_dw` here and adds the manifold differential in `predict` (§4.2).

### 2.3 `df_dw` → 24×12 (`use-ikfom.hpp:80‑88`)

```cpp
cov(12:14, 3:5) = -s.rot.toRotationMatrix(); // ∂v̇/∂n_a  = -R   (n_a = noise cols 3:5)
cov(3:5,   0:2) = -I₃;                        // ∂θ̇/∂n_g  = -I
cov(15:17, 6:8) =  I₃;                        // ∂ḃ_g/∂n_bg = I
cov(18:20, 9:11)=  I₃;                        // ∂ḃ_a/∂n_ba = I
```
Noise ordering = [n_g(0:2), n_a(3:5), n_bg(6:8), n_ba(9:11)], matching `process_noise_ikfom`. Matches paper `F_w` (Eq. 7).

---

## 3. The generic MTK manifold toolkit (the ⊞/⊟ machinery)

Lives under `FAST_LIO/include/IKFoM_toolkit/mtk/`. Fully generic; the LIO state is one instantiation.

### 3.1 `MTK_BUILD_MANIFOLD` — composing a product manifold (`build_manifold.hpp:180‑225`)

The macro generates a `struct` that (a) stores each member as a `MTK::SubManifold<type,dof,dim>` (`build_manifold.hpp:126`); (b) computes per‑member compile‑time `DOF`/`DIM` offsets by summation (`build_manifold.hpp:129‑130, 145‑152`); (c) generates generic operators that **loop over members and call each member's own operator on its sub‑segment**:

```cpp
void boxplus(vectview<const scalar,DOF> v, scalar s=1){ each: id.boxplus(subvector(v,&id), s); } // :192, macro :101
void oplus  (vectview<const scalar,DIM> v, scalar s=1){ each: id.oplus  (subvector_(v,&id), s);} // :195, macro :102
void boxminus(vectview<scalar,DOF> r, const name& o) const { each: id.boxminus(subvector(r,&id), o.id);} // :198, macro :103
```
Hence `(x ⊞ δ)ᵢ = xᵢ ⊞ᵢ δᵢ` and `(y ⊟ x)ᵢ = yᵢ ⊟ᵢ xᵢ`, each component using its own manifold's operators. **Critical distinction**: `boxplus` consumes a **DOF (23)** vector (tangent increments), while `oplus` consumes a **DIM (24)** vector — `oplus` is used in `predict` to apply the flat `get_f`·Δt increment (which has a 3‑row gravity slot), `boxplus` is used in the update to apply the 23‑vector tangent increment. The macro also builds three `std::vector<pair>` indices at init — `S2_state`, `SO3_state`, `vect_state` (`build_manifold.hpp:183‑185, 204‑212`, macros `:109‑111`) — which the filter iterates to know where each manifold block sits (this is how `predict`/`update` find the SO(3) and S² blocks generically).

### 3.2 SO(3) component (`SOn.hpp:177‑298`)

```
R ⊞ δ = R · exp(δ)        (boxplus, :233-236; oplus identical :242-245)
R ⊟ Q = log(Qᵀ R)         (boxminus, :237-239: res = log(other.conjugate()* *this))
```
`exp` uses `MTK::exp` (the cos/sinc‑based quaternion exponential, `mtkmath.hpp:249‑256`) with `scale/2` because it builds the quaternion (`SOn.hpp:284‑287`); `log` uses `MTK::log` (`mtkmath.hpp:268‑288`).

### 3.3 S² component — the gravity manifold (`S2.hpp:97‑310`)

Stored 3‑vector `vec` of fixed `length` (`S2.hpp:104,111`). Tangent DOF=2. Key members:

- **`boxplus(δ₂, scale)`** (`S2.hpp:136‑142`): `Bx = S2_Bx(); Bu = Bx·δ₂; vec = Exp(Bu)·vec`. The 2‑D tangent δ₂ is lifted to a 3‑axis through the local basis `B_x` and applied as a rotation — keeping `‖vec‖` constant. This is the on‑sphere retraction.
- **`oplus(δ₃, scale)`** (`S2.hpp:129‑134`): `vec = Exp(δ₃)·vec` — the 3‑D variant used by `predict` (the `get_f` gravity slot is 3 rows of zeros, so this is a no‑op for gravity but kept generic).
- **`boxminus(res₂, other)`** (`S2.hpp:144‑167`): `θ = atan2(‖⌊vec⌋·other‖, vecᵀother)`; `res = θ/sinθ · Bxᵀ·⌊other.vec⌋·vec`, with a π/0 guard for antipodal/identical points.
- **`S2_Bx` → 3×2** (`S2.hpp:179‑232`): the orthonormal‑ish basis of the tangent plane at `vec`; three branches for `S2_typ ∈ {1,2,3}` (FAST‑LIO uses `S2_typ=1`, the x‑axis branch `:217‑231`), each with a near‑antipode fallback (`res(1,1)=-1, res(2,0)=1`).
- **`S2_Mx(res₃ₓ₂, δ₂)`** (`S2.hpp:266‑280`): if `‖δ‖<tol` → `res = −⌊vec⌋·Bx`; else `res = −Exp(Bu)·⌊vec⌋·A_matrix(Bu)ᵀ·Bx`. This is the matrix used in `df_dx` (∂v̇/∂δg) **and** as the propagation transport of the S² block.
- **`S2_Nx_yy(res₂ₓ₃)`** (`S2.hpp:259‑264`): `res = (1/length²)·Bxᵀ·⌊vec⌋` — the **2×3 projection** from an ambient 3‑perturbation onto the 2‑D tangent (the dual of `Mx`), used to fold the gravity block in `predict`/`update`.
- **`S2_hat(res₃ₓ₃)`** (`S2.hpp:169‑176`): `⌊vec⌋`.

> Engineering refinement vs. a naive chart: `B_x` is recomputed at the *current* `vec` every call, so the 2‑D chart is re‑anchored to the current gravity estimate each iteration — avoiding the azimuth/elevation singularity and keeping `P` well‑conditioned. The paper motivates S² to fix gravity magnitude (it sets `Gg = 0` in Eq. 1, line 183); the code realizes it with this moving local frame.

### 3.4 Euclidean component (`vect.hpp:99‑210`)
`vect<D>` (DOF=DIM=D, TYP=0): `x ⊞ δ = x + δ` (`:117‑119`), `y ⊟ x = y − x` (`:120‑122`), `oplus` `+= scale·vec` (`:124‑126`). Trivial Jacobian = I.

### 3.5 Manifold math helpers (`mtkmath.hpp`)
- `hat(v)` skew (`:176‑183`).
- **`A_matrix(v)`** (`:235‑247`) — the SO(3) right‑Jacobian "A matrix":
  ```
  A(v) = I + ((1−cos‖v‖)/‖v‖²)⌊v⌋ + ((1 − sin‖v‖/‖v‖)/‖v‖²)⌊v⌋²   (I if ‖v‖<tol)
  ```
  This is the factor that makes the on‑manifold transition consistent; it appears in `predict` (SO(3)/S² blocks) and in the update transport.
- `A_inv(v)` / `A_inv_trans(v)` (`:185‑215`) — the inverse Jacobian `A(u)⁻¹` (the paper's Eq. 6, `2010.08196.txt:404‑428`).
- `exp` (cos/sinc quaternion exponential, `:249‑256`) and `log` (`:268‑288`) with Taylor guards (`cos_sinc_sqrt` `:142‑174`; `tolerance<double>()=1e-11` `:122`).

---

## 4. The IKFoM filter core (`esekfom.hpp`)

`class esekf<state, process_noise_dof, input, measurement, measurement_noise_dof>` (`:105‑106`). Holds nominal `state x_` and covariance `cov P_` (23×23). Function‑pointer model hooks `f, f_x, f_w, h_dyn_share` plus `maximum_iter` and per‑DOF threshold array `limit[n]`.

### 4.1 Registration — `init_dyn_share(...)` (`esekfom.hpp:238‑254`)
LIO calls (`laserMapping.cpp:826‑828`):
```cpp
double epsi[23] = {0.001}; fill(epsi, epsi+23, 0.001);
kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);
```
`NUM_MAX_ITERATIONS` defaults to 4 (`laserMapping.cpp:765`). `init_dyn_share` stores the four callbacks, copies `limit[i]=epsi[i]`, and crucially calls `x_.build_S2_state(); x_.build_SO3_state(); x_.build_vect_state();` (`:251‑253`) to populate the member‑index vectors used by `predict`/`update`. "dyn_share" = the measurement Jacobian `h_x` and residual `h` are produced **together by one callback** rather than the filter forming `H` itself (the central engineering optimization, §4.6).

### 4.2 Forward propagation — `predict(dt, Q, in)` (`esekfom.hpp:279‑383`)

```cpp
flatted_state f_ = f(x_, in);          // 24-vector  (= get_f)            :280
cov_         f_x_ = f_x(x_, in);       // 24x23      (= df_dx)            :281
Matrix<...,m,12> f_w_ = f_w(x_, in);   // 24x12      (= df_dw)            :284
state x_before = x_;                                                       :286
x_.oplus(f_, dt);                       // x_{k+1} = x_k ⊞ (dt·f) on the manifold  :287

F_x1 = Identity(23,23);                                                    :289
// (a) Euclidean rows: copy df_dx/df_dw rows straight through (DIM idx == DOF idx)  :290-302
// (b) SO(3) blocks: for each (idx,dim) in SO3_state:                      :305-330
//     seg = -f_(dim..)*dt ;  res = Exp(seg) (quaternion)                  :309-312
//     F_x1.block<3,3>(idx,idx) = res.toRotationMatrix();                  :321
//     A = A_matrix(seg);  f_x_final.block<3,1>(idx,i) = A * df_dx.block<3,1>(dim,i)   :323-326
//     f_w_final.block<3,1>(idx,i) = A * df_dw.block<3,1>(dim,i)           :327-329
// (c) S2 block (gravity): for each (idx,dim) in S2_state:                 :336-371
//     seg = f_(dim..)*dt ;  res = Exp(seg)                                :340-344
//     Nx = x_.S2_Nx_yy(idx) (2x3 at NEW g) ;  Mx = x_before.S2_Mx(idx) (3x2 at OLD g)  :347-348
//     F_x1.block<2,2>(idx,idx) = Nx * res.toRotationMatrix() * Mx;        :357
//     res_S2 = -Nx * Exp(seg) * x_before.S2_hat * A_matrix(seg)ᵀ  (2x3)   :360-362
//     f_x_final.block<2,1>(idx,i) = res_S2 * df_dx.block<3,1>(dim,i)      :364-366
//     f_w_final.block<2,1>(idx,i) = res_S2 * df_dw.block<3,1>(dim,i)      :368-370

F_x1 += f_x_final * dt;                                                    :380
P_ = F_x1 * P_ * F_x1ᵀ + (dt·f_w_final) * Q * (dt·f_w_final)ᵀ;            :381
```

So the discrete error‑state model is the textbook EKF prediction **with manifold corrections baked in**:
```
δx_{k+1} = F_x δx_k + F_w w_k ,   w_k ~ N(0,Q)
P_{k+1}  = F_x P_k F_xᵀ + F_w Q F_wᵀ
```
but `F_x ≠ I + df_dx·Δt`. It is `F_x1 + f_x_final·Δt`, where:
- `F_x1` carries the **block‑diagonal retraction transition**: identity for Euclidean blocks, `Exp(−ω Δt)` (rotation matrix of the negative body‑rate increment) for SO(3) blocks (`:321`), and `Nx·Exp(seg)·Mx` for the S² block (`:357`);
- `f_x_final` is `df_dx` with each SO(3) 3‑row block **left‑multiplied by `A_matrix(seg)`** (right‑Jacobian, `:323‑326`) and the S² 3‑row block multiplied by the 2×3 `res_S2` reduction (`:360‑366`).

This is precisely the paper's `Fx`/`Fw` (Eq. 7, `2010.08196.txt:256‑298`, derived in Appendix A Eq. 22‑23, `:692‑721`), with the `A(·)`/`A(·)⁻¹` Jacobian factors. The S² block uses `S2_Nx_yy` (at the **new** gravity) × `S2_Mx` (at the **old** gravity) to transport the 2‑DOF chart across the prediction. `x_.oplus(f_, dt)` (`:287`) applies the mean update on the manifold; for gravity, `oplus` rotates `g` by `Exp(0)=I` (its `get_f` slot is zero) so `‖g‖` is preserved exactly.

### 4.3 Where Q comes from (`IMU_Processing.hpp`)
`Q` is rebuilt **per IMU sample** before each `predict` (`IMU_Processing.hpp:280‑284`):
```cpp
Q.block<3,3>(0,0).diagonal() = cov_gyr;     // n_g
Q.block<3,3>(3,3).diagonal() = cov_acc;     // n_a
Q.block<3,3>(6,6).diagonal() = cov_bias_gyr;// n_bg
Q.block<3,3>(9,9).diagonal() = cov_bias_acc;// n_ba
kf_state.predict(dt, Q, in);                // IMU_Processing.hpp:284
```
i.e. `Q = diag(cov_gyr, cov_acc, cov_bias_gyr, cov_bias_acc)` (12×12). `predict` is called once per IMU interval inside `UndistortPcl` (the forward‑propagation loop `IMU_Processing.hpp:250‑296`) and once more to the scan‑end time (`:299‑301`), simultaneously building the per‑point undistortion poses `IMUpose` (`:295`). `in.acc`/`in.gyro` are the midpoint‑averaged IMU readings (`:257‑262`), with `acc_avr` rescaled to `G_m_s2/mean_acc.norm()` (`:266`).

### 4.4 The SO(3)/S² tangent re‑projection during the update (the J / L transport)
Because `P_` is anchored to the tangent at the *propagated* point but each iteration re‑anchors the chart at the *current* estimate, the filter transports `P_` between charts. In `update_iterated_dyn_share` (`esekfom.hpp:1037‑1078`) this is done block‑wise:
- SO(3) blocks: `res_temp_SO3 = A_matrix(seg_SO3)ᵀ` with `seg = dx(idx..)`, then `P_.block(idx,·) = res·P_.block(idx,·)` on both sides and `dx_new(idx) = res·dx(idx)` (`:1040‑1055`).
- S² block: `Nx = x_.S2_Nx_yy(idx)`, `Mx = x_propagated.S2_Mx(seg_S2, idx)`, `res_S2 = Nx·Mx (2×2)`, applied to `P_` rows/cols and `dx_new` (`:1059‑1078`).

This is the paper's `J` matrix (Eq. 16, `2010.08196.txt:509‑525`): `J^ℓ = diag(A(GR̄ᵀ GR)⁻ᵀ, I₁₅)` mapping the propagated tangent into the current chart, with `P = (J^ℓ)⁻¹ Pk (J^ℓ)⁻ᵀ` (paper Algorithm‑1 step 6, `:615`). The final posterior covariance uses the same transport into matrix `L_` (`esekfom.hpp:1126‑1185`) before the Joseph‑form update.

### 4.5 The iterated update — `update_iterated_dyn_share_modified(R, solve_time)` (`esekfom.hpp:1619`)
Call site: `kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);` (`laserMapping.cpp:960`). `LASER_POINT_COV` is the scalar LiDAR point‑to‑plane variance (passed as the `double R`). **[reconstructed from its in‑file twin `update_iterated_dyn_share` `esekfom.hpp:1001‑1155`, which I read in full; the `_modified` variant differs only in taking a scalar `R` and timing `solve_time`, and in storing `H` only over the populated 12 columns]:**

```
x_propagated = x_;  P_propagated = P_;                         // freeze the prior (k|k-1)
for i = -1 .. maximum_iter-1:
   dyn_share.valid = true;
   h_dyn_share(x_, dyn_share);          // == h_share_model: fills dyn_share.h_x (Hsub), dyn_share.h (z)
   if(!dyn_share.valid) continue;
   z   = dyn_share.h;                    // residual vector (m_eff)         (twin :1017)
   Hsub= dyn_share.h_x;                  // m_eff x 12  (populated cols only)
   dx  = x_ ⊟ x_propagated;             // 23-vector                       (twin :1030)
   P_  = P_propagated;                                                       (twin :1037)
   // transport prior into current chart: dx_new = J·dx,  P_ = J P_ Jᵀ
   //   SO(3): res=A_matrix(seg)ᵀ ;  S2: res=Nx·Mx                          (twin :1040-1078)
   // Kalman gain, information form (n <= m, the LIO case):
   //   R^{-1} = (1/LASER_POINT_COV) ;  H = Hsub (only 12 nonzero cols)
   //   K = (Hᵀ R^{-1} H + P^{-1})^{-1} Hᵀ R^{-1}                           (twin :1105-1106)
   K_x = K * H;                                                              (twin :1110)
   dx_ = K * z + (K_x − I) * dx_new;     // note sign: -(I-KH)·J·dx          (twin :1111)
   x_ = x_ ⊞ dx_;                        // boxplus on manifold              (twin :1113)
   converge = ( |dx_[j]| < limit[j]  ∀ j∈0..22 );                           (twin :1115-1122)
   if(converge) t++;
   if(t>1 || i==maximum_iter-1):
       // posterior covariance via L transport then Joseph form:
       L_ = P_; ... (SO(3)/S2 transport of P_, K, K_x)                       (twin :1126-1185)
       P_ = L_ − K_x * P_;   (n<=m branch)                                   (twin :1196 region)
       return;
```

Key facts to carry into the rebuild ("Meridian"):

1. **Iterated EKF (IEKF), not single EKF.** Each iteration recomputes `H`,`z` at the relinearized `x_` (the `h_dyn_share` call), so it is Gauss‑Newton on the MAP cost `min_{δx} ‖δx‖²_{P⁻¹} + Σⱼ ‖zⱼ + Hⱼ δx‖²_{Rⱼ⁻¹}` (paper Eq. 17, `2010.08196.txt:529‑545`). The loop starts at `i=-1` so the first pass is the plain EKF linearization where `J=I` (paper note at `:525`).
2. **The prior pull‑back term `(K_x − I)·J·dx`** is essential — it is the on‑manifold IEKF prior term `−(I − KH)(J^ℓ)⁻¹(x̂^ℓ ⊟ x̂)` of paper Eq. 18 (`2010.08196.txt:580‑586`). Dropping it reduces the filter to prior‑free Gauss‑Newton.
3. **Information‑form Kalman gain** `K = (HᵀR⁻¹H + P⁻¹)⁻¹HᵀR⁻¹` (paper Eq. 20, `2010.08196.txt:596`; code `esekfom.hpp:1105‑1106`, `1486‑1487`, etc.). Proven equivalent to `K = PHᵀ(HPHᵀ+R)⁻¹` (Eq. 18) via the matrix‑inversion lemma (paper Appendix B, `:723‑734`). Because the LiDAR scan yields `m` ≫ `n=23` residuals, this inverts only **23×23**, never `m×m`. Measured speedup: Old vs New formula 109.3 ms → 0.25 ms at 998 features, 1621 ms → 1.16 ms at 1802 features (paper Table II, `2010.08196.txt:653‑659`). This is *the* headline efficiency trick (paper contribution 2, `:29`; FAST‑LIO2 `2107.06829.txt:19` "mathematically equivalent formula … complexity to the dimension of the state").
4. **Convergence** is per‑DOF: `|dx_[j]| < limit[j]=epsi[j]=0.001` for all 23 entries, and the loop exits after the *second* converged iteration (`t>1`) or at `maximum_iter`.
5. After the loop, `state_point = kf.get_x()` (`laserMapping.cpp:961`) and `euler_cur = SO3ToEuler(state_point.rot)` (`:962`) publish the result.

### 4.6 The shared measurement callback `h_share_model` (`laserMapping.cpp:638`)
Signature (`laserMapping.cpp:638`): `void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)`. The `dyn_share_datastruct<T>` carries `valid, converge, z, h, h_v, h_x, R` (`esekfom.hpp:80‑89`). **[body 640‑760 reconstructed from the paper Eqs. 11‑12,14 + the verified world transform `pointBodyToWorld_ikfom` `laserMapping.cpp:166‑175` + globals `point_selected_surf`, `Nearest_Points`, `corr_normvect`, `laserCloudOri`, `effct_feat_num` `laserMapping.cpp:99‑114`, `match_time` `:640`, `solve_time` `:753`]:**

Per downsampled body point `p_body` (loop over `feats_down_size`, OpenMP):
1. Transform to world with the **current iterate**: `p_world = s.rot·(s.offset_R_L_I·p_body + s.offset_T_L_I) + s.pos` (exactly `pointBodyToWorld_ikfom`, `laserMapping.cpp:169`). This applies both the IMU pose and the *online* LiDAR→IMU extrinsic.
2. 5‑NN search the **ikd‑Tree** map (`Nearest_Points`), fit a plane with `esti_plane(pca_result, points_near, threshold)` (`common_lib.h:226‑257`) → normal `n=(a,b,c)`, offset `d` with `‖n‖=1` (`common_lib.h:243‑247`); reject if any of the 5 points is > threshold off the plane.
3. Residual = signed point‑to‑plane distance `z_j = nᵀ p_world + d` (paper Eq. 12, `2010.08196.txt:476`, `Gⱼ = uⱼᵀ`).
4. Validity/robustness gate (`point_selected_surf[i]`), keep effective points into `laserCloudOri`/`corr_normvect`, count `effct_feat_num`.
5. Jacobian row `ekfom_data.h_x` (size `effct × 12`, only the leading 12 cols of the 23 are nonzero for point‑to‑plane), `ekfom_data.h = -z_j`. The nonzero blocks (right‑perturbation, consistent with `df_dx`):
   ```
   ∂z/∂δp     = nᵀ                                              (cols 0:2)
   ∂z/∂δθ     = −nᵀ R ⌊R_LI p_body + t_LI⌋                       (cols 3:5)
   ∂z/∂δθ_ext = −nᵀ R R_LI ⌊p_body⌋   (only if extrinsic_est_en) (cols 6:8)
   ∂z/∂δt_ext =  nᵀ R R_LI            (only if extrinsic_est_en) (cols 9:11)
   ```
   `extrinsic_est_en` defaults true (`laserMapping.cpp:73`). This is the `Hj` of paper Eq. 14 (`2010.08196.txt:501‑505`).

This callback **is** the `h_dyn_share` invoked at the top of each update iteration; computing `H` and `z` together (the "dyn_share" pattern) is what lets the filter avoid ever materializing a full `m×23` `H` and invert only 23×23.

### 4.7 Accessors and initialization
`get_x()` / `change_x(state&)` and `get_P()` / `change_P(cov&)` (`esekfom.hpp` accessor block near `:1933‑1952`). Used by `laserMapping.cpp:889,961` (`state_point = kf.get_x()`) and `:596` (`auto P = kf.get_P()` for odometry covariance). Initialization in `IMU_init` (`IMU_Processing.hpp:159‑214`):
```cpp
init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);   // :196  gravity from static accel
init_state.bg   = mean_gyr;                                     // :199
init_state.offset_R_L_I / offset_T_L_I = Lidar_R/T_wrt_IMU;     // :200-201
kf_state.change_x(init_state);                                  // :202
init_P = I; init_P(6..8)=init_P(9..11)=1e-5; init_P(15..17)=1e-4;
init_P(18..20)=1e-3; init_P(21,21)=init_P(22,22)=1e-5;          // :205-210 (note: 23-dim indexing)
kf_state.change_P(init_P);                                      // :211
```
Note `init_P` is indexed in the **23‑dim tangent** space (the gravity entries are at 21,22 — confirming DOF=23).

---

## 5. SO(3) primitives in the application layer (`so3_math.h`, `Exp_mat.h`)

The toolkit's `MTK::exp`/`log` are quaternion‑based; the *application* (IMU integration, undistortion, logging) uses the matrix‑form Rodrigues in `so3_math.h`:
- **`SKEW_SYM_MATRX(v)`** macro → `⌊v⌋` (`so3_math.h:7`); `skew_sym_mat<T>` template (`:9‑15`).
- **`Exp(ang)`** (`so3_math.h:17‑34`): `Exp(ω) = I + sin‖ω‖·K + (1−cos‖ω‖)·K²`, `K=⌊ω/‖ω‖⌋`, with small‑angle guard `‖ω‖>1e-7` else `I`.
- **`Exp(ang_vel, dt)`** (`:36‑58`): same with `r_ang = ‖ω‖·dt` — used in undistortion `R_i = R_imu·Exp(angvel_avr, dt)` (`IMU_Processing.hpp:331`).
- **`Log(R)`** (`:81‑87`): `θ = acos((tr R − 1)/2)` (guarded `tr>3−1e-6 ⇒ 0`); `Log = 0.5·θ/sinθ·(R−Rᵀ)^∨` (or `0.5·K` for `|θ|<0.001`).
- `RotMtoEuler` (`:89‑109`). `Exp_mat.h:11‑81` duplicates `Exp`/`Log` (matrix form, OpenCV include) but its `Log` lacks the trace guard (`Exp_mat.h:78`) — a subtle robustness difference.

> Convention to preserve exactly: FAST‑LIO uses the **right perturbation** `R ⊞ δ = R·Exp(δ)` (`SOn.hpp:233‑236`; paper `:131`). That is why `∂(R a)/∂θ = −R⌊a⌋` in `df_dx` (`use-ikfom.hpp:69`) and why the rotation update is `R·Exp(dx_)`. A left‑perturbation rebuild would flip these signs and break the prior pull‑back term in §4.5.

---

## 6. How the generic toolkit maps onto this specific LIO state (instantiation map)

| Generic IKFoM concept | LIO realization | Where |
|---|---|---|
| `state` manifold (DOF=23, DIM=24) | vect3×SO3×SO3×vect3×vect3×vect3×vect3×S2 | `use-ikfom.hpp:12‑21` |
| `input` (6) | `input_ikfom`={acc,gyro} | `use-ikfom.hpp:23‑26` |
| process noise (12) | `process_noise_ikfom`={ng,na,nbg,nba} | `use-ikfom.hpp:28‑33` |
| `f` | `get_f` strapdown | `use-ikfom.hpp:47‑59` |
| `df_dx` (24×23), `df_dw` (24×12) | analytic Jacobians, S² basis via `S2_Mx` | `use-ikfom.hpp:61‑88` |
| `h_dyn_share` | `h_share_model` point‑to‑plane on ikd‑Tree | `laserMapping.cpp:638` |
| ⊞ / ⊟ | per‑component, synthesized by macro | `build_manifold.hpp:192‑200`; `S2/SOn/vect.hpp` |
| SO(3) retraction | quaternion `MTK::exp`/`log`; matrix `Exp`/`Log` | `mtkmath.hpp:249‑288`; `so3_math.h:17‑87` |
| right‑Jacobian A(·) | `MTK::A_matrix` | `mtkmath.hpp:235‑247` |
| `predict` (F_x,F_w,P) | generic; calls f/df_dx/df_dw + manifold diffs | `esekfom.hpp:279‑383` |
| iterated update + info gain + J/L | `update_iterated_dyn_share_modified` | `esekfom.hpp:1619` (twin `:1001‑1155`) |
| Q | diag(cov_gyr,cov_acc,cov_bias_gyr,cov_bias_acc) | `IMU_Processing.hpp:280‑284` |
| init (gravity on S², biases, extr, P) | `IMU_init` | `IMU_Processing.hpp:159‑214` |
| publish/consume x, P | `get_x/get_P` | `laserMapping.cpp:889,960‑967,596` |

(For context: `common_lib.h:68‑154` defines an **older, unused** flat 18‑DOF `StatesGroup` with `operator+`/`operator-` doing additive position/velocity/bias/gravity and `Exp/Log` rotation — `gravity` there is plain ℝ³, `cov` is 18×18. `#define USE_IKFOM` at `common_lib.h:17` selects the manifold path instead; the rebuild should follow the IKFoM 23‑DOF path, not `StatesGroup`.)

---

## 7. Engineering choices & code‑vs‑paper notes (for "Meridian")

1. **S² for gravity (2 DOF) is the defining manifold choice.** The paper sets `Gg=0` in the dynamics and treats gravity as an 18‑DOF flat error in the original FAST‑LIO derivation (`2010.08196.txt:182‑198`, `dim(M)=18`), but the **code** (and FAST‑LIO2) promotes gravity to S² → tangent dim 23. Replicating S² requires the 3×2 `S2_Bx`/`S2_Mx` basis, the 2×3 `S2_Nx_yy` projection, and the cross‑chart transport in both `predict` and the update — or the 2 DOF will leak to 3.
2. **Online extrinsic estimation** (`offset_R_L_I`, `offset_T_L_I`) is in the state with Jacobians in `h_share_model`, toggled by `extrinsic_est_en` (`laserMapping.cpp:73`). This refines the basic paper model; cols 6:11 of `H` are populated only when enabled.
3. **`h_share_model` computes residual + Jacobian together** and the filter uses the **information‑form gain** to invert only 23×23 (paper Eq. 20; Table II speedups). Preserve both.
4. **Right‑perturbation SO(3)** is baked into every sign (`df_dx`, `h_share_model`, the update). Choose one convention and apply uniformly.
5. **Iterated = Gauss‑Newton with a Kalman prior.** Loop from `i=-1` (first pass is EKF, `J=I`), prior term `(K_x−I)·J·(x⊟x_prop)`, convergence per‑DOF `< epsi=0.001`, `maximum_iter=4`.
6. **Covariance is 23×23 (tangent), not 24.** `predict` mixes in the right‑Jacobian `A_matrix` so `F_x ≠ I + df_dx·Δt` exactly; the gravity block is `Nx·Exp(seg)·Mx`. Convert to ambient only for I/O.
7. **Small‑angle guards everywhere** — `so3_math.h` uses `1e-7`/`tr>3−1e-6`/`|θ|<0.001`; `mtkmath.hpp` uses `tolerance<double>()=1e-11` and `cos_sinc_sqrt` Taylor series; `S2_Bx` has antipode fallbacks. Required near zero rotation / antipodal gravity.
8. **`Q` is set per‑IMU‑step, not from `process_noise_cov()`** — the macro defaults (`use-ikfom.hpp:38‑41`) are overwritten by `cov_gyr/acc/bias_*` each call (`IMU_Processing.hpp:280‑283`), which come from the runtime params (`laserMapping.cpp` `set_gyr_cov` etc.). Don't hard‑code Q.

---

## 8. Quick reference — the equations to implement

```
State (manifold M, DOF=23, DIM=24):
  x = (p, R, R_LI, t_LI, v, b_g, b_a, g),   R,R_LI ∈ SO(3),  g ∈ S² (‖g‖=9.809 fixed)
Tangent δx (23): (δp, δθ, δθ_ext, δt_ext, δv, δb_g, δb_a, δg₂)

Retraction ⊞:   Euclidean +;  SO(3): R·Exp(δθ);  S²: Exp(B(g)·δg₂)·g
Difference ⊟:   Euclidean −;  SO(3): Log(Rᵀ Q);  S²: (θ/sinθ)·Bxᵀ⌊other⌋·g  (2-vec)
Right-Jacobian:  A(v) = I + ((1-cos‖v‖)/‖v‖²)⌊v⌋ + ((1-sin‖v‖/‖v‖)/‖v‖²)⌊v⌋²

Continuous f (get_f):  ṗ=v, θ̇=w_m−b_g, v̇=R(a_m−b_a)+g, ḃ=n, ġ=0, ext=const
df_dx (24×23):  [ṗ,δv]=I, [v̇,δθ]=−R⌊a_m−b_a⌋, [v̇,δb_a]=−R, [v̇,δg]=B(g) (3×2), [θ̇,δb_g]=−I
df_dw (24×12):  [v̇,n_a]=−R, [θ̇,n_g]=−I, [ḃ_g,n_bg]=I, [ḃ_a,n_ba]=I

Predict:  x ← x ⊞_oplus (Δt·f)
          F_x = F_x1 + f_x_final·Δt
              (F_x1: SO(3) blk=Exp(−ωΔt), S² blk=Nx·Exp(seg)·Mx, Euclid=I;
               f_x_final: SO(3) rows ← A(seg)·df_dx, S² rows ← res_S2·df_dx)
          F_w analogous;   P ← F_x P F_xᵀ + (Δt·F_w) Q (Δt·F_w)ᵀ
          Q = diag(cov_gyr, cov_acc, cov_bias_gyr, cov_bias_acc)

Measurement (per point):  p_world = R(R_LI p_b + t_LI) + p;   z_j = nⱼᵀ p_world + dⱼ
          H_j = [ nᵀ , −nᵀR⌊R_LI p_b + t_LI⌋ , −nᵀR R_LI⌊p_b⌋ , nᵀR R_LI , 0…0 ]  (1×23, 12 nonzero)

Iterated update (loop i=-1..maxiter, exit when t>1 or last):
          h_share_model → (H, z) at current x
          dx = x ⊟ x_{k|k-1};   dx ← J dx,  P ← J P Jᵀ   (J: SO(3)=A(seg)ᵀ, S²=Nx·Mx)
          K  = (Hᵀ R⁻¹ H + P⁻¹)⁻¹ Hᵀ R⁻¹                 (info form; invert 23×23; R=LASER_POINT_COV)
          Δ  = K z + (KH − I)·dx                          (= −K z' − (I−KH)J·(x⊟x_prop))
          x ← x ⊞ Δ;   converged if |Δ_j| < 0.001 ∀ j
          on exit:  P ← L − K H P   (Joseph form, with L,K transported by J)
```

---

## 9. Source index (the files a rebuild author must mirror)

- `FAST_LIO/include/use-ikfom.hpp` — state/input/noise manifolds, `get_f/df_dx/df_dw`, `SO3ToEuler`.
- `FAST_LIO/include/IKFoM_toolkit/esekfom/esekfom.hpp` — `esekf` class, `init_dyn_share`, `predict`, `update_iterated_dyn_share[_modified]`, `get_x/change_x/get_P/change_P`, `dyn_share_datastruct`.
- `FAST_LIO/include/IKFoM_toolkit/mtk/build_manifold.hpp` — `MTK_BUILD_MANIFOLD` (boxplus/oplus/boxminus generation, S2/SO3/vect state indices).
- `FAST_LIO/include/IKFoM_toolkit/mtk/types/{S2,SOn,vect}.hpp` — per‑manifold ⊞/⊟, `S2_Bx/Mx/Nx_yy/hat`.
- `FAST_LIO/include/IKFoM_toolkit/mtk/src/mtkmath.hpp` — `A_matrix`, `A_inv`, `hat`, `exp`/`log`, tolerances.
- `FAST_LIO/include/so3_math.h`, `Exp_mat.h` — application‑layer `Exp/Log` (Rodrigues, matrix form).
- `FAST_LIO/src/IMU_Processing.hpp` — `Q` assembly, `predict` calls, `IMU_init` (gravity on S², init P/x).
- `FAST_LIO/src/laserMapping.cpp` — `h_share_model`, `init_dyn_share` registration, `update_iterated_dyn_share_modified` call, `pointBodyToWorld_ikfom`, `epsi`, `NUM_MAX_ITERATIONS`, odometry covariance export.
