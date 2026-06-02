# Grounding Dossier — Point-LIO: Point-Wise Update & Stochastic IMU-as-Measurement Model

**Scope.** This dossier grounds, line by line, how Point-LIO (a) updates the filter per LiDAR point (or per "same-timestamp" mini-batch) at the *true sampling time* rather than once per scan; (b) treats the IMU gyro/accel readings as **measurements** of an *extended state* in which angular velocity `omg` and acceleration `acc` are themselves state variables driven by a random-walk/white-noise process, so there is **no explicit de-skew / motion-undistortion step**; (c) the on-manifold iterated-Kalman-filter (IKFoM/MTK) machinery that executes both the point update and the IMU update; and (d) how this realizes a discrete sampling of a continuous-time stochastic trajectory. It contrasts each point with FAST-LIO2.

All Point-LIO paths are under `C:/Users/charl/Sources/slam-reference/Point-LIO/`. FAST-LIO contrast paths are under `C:/Users/charl/Sources/slam-reference/FAST_LIO/`. FAST-LIO2 paper text is `papers/2107.06829.txt`; FAST-LIO paper text is `papers/2010.08196.txt`. (The Point-LIO paper itself is *not* on disk; Point-LIO claims are grounded in code + `Point-LIO/README.md`. Published as *He, Xu, Cai, Zhang, "Point-LIO: Robust High-Bandwidth Lidar-Inertial Odometry," Advanced Intelligent Systems*, DOI 10.1002/aisy.202200459, per `README.md:46-47`.) **Every code claim below was read directly and is line-cited; where I note the code refines/differs from the paper I say so.**

---

## 0. Executive summary (the four ideas, and where they live)

1. **Two modes, two states, two filters.** Point-LIO maintains *two* alternative state manifolds and *two* Kalman filters, selected by the global flag `use_imu_as_input` (default `false`, `parameters.cpp:16`):
   - **"input" mode** (`use_imu_as_input = true`): a 24-DOF `state_input` (`common_lib.h:25-34`) where the IMU is the *control input* `input_ikfom = {acc, gyro}` (`common_lib.h:49-52`). This essentially reproduces the FAST-LIO2 formulation inside the same codebase, filtered by `kf_input` (`Estimator.cpp:17`).
   - **"output" mode** (default, `use_imu_as_input = false`): a 30-DOF `state_output` (`common_lib.h:36-47`) that **adds angular velocity `omg` and acceleration `acc` to the state** and treats the IMU as a *measurement* of those blocks plus their biases. This is the novel Point-LIO model, filtered by `kf_output` (`Estimator.cpp:18`).
2. **Per-point / per-batch update at the true sample time.** The main loop iterates over a *time-compressed* list `time_seq` of the down-sampled cloud (`laserMapping.cpp:496`; impl `common_lib.h:139-165`). For each batch `k` it computes the absolute sample time `time_current = point_body.curvature/1000 + pcl_beg_time` (`laserMapping.cpp:601`, `:819`), **propagates the filter forward by the exact `dt` to that instant**, then runs a measurement update. This replaces FAST-LIO2's single per-scan iterated update.
3. **IMU as a stochastic measurement, removing de-skew.** In output mode each IMU sample triggers `h_model_IMU_output` (`Estimator.cpp:324-368`) whose residual is `z_IMU = [ω_meas − omg − bg ; a_meas·g/‖a‖ − acc − ba]` (`Estimator.cpp:327-328`). Because `omg`/`acc` are *states* whose mean derivative is zero plus process noise (`get_f_output`, `Estimator.cpp:68-78`; cov `Estimator.cpp:42-52`), each point is registered with the *propagated* pose at its own timestamp — there is **no backward-propagation undistortion pass** anywhere in Point-LIO. Contrast: FAST-LIO's `UndistortPcl` (`FAST_LIO/src/IMU_Processing.hpp:216-...`, "undistort each lidar point (backward propagation)" at `:307`) explicitly back-propagates every point into the scan-end frame.
4. **On-manifold IKFoM with three measurement hooks.** The filter is MTK/IKFoM `esekfom::esekf<state, DIM, input>` (`common_lib.h:69-70`), extended with a *modified* `dyn_share_modified` struct (`esekfom.hpp:61-72`) and three entry points: `update_iterated_dyn_share_modified()` for LiDAR (`esekfom.hpp:184-240`), `update_iterated_dyn_share_IMU()` for IMU-as-measurement (`esekfom.hpp:242-282`), and `predict()` for on-manifold covariance propagation (`esekfom.hpp:137-182`). All are wired in `main`: `kf_input.init_dyn_share_modified_2h(...)` and `kf_output.init_dyn_share_modified_3h(...)` (`laserMapping.cpp:357-358`).

---

## 1. State manifolds and the extended ("output") state

### 1.1 The two state definitions (MTK manifolds)
`common_lib.h:19-23`:
```
typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double>     SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2;   // declared; gravity is kept as vect3 in these states
```

**Input state** — 24 DOF (`common_lib.h:25-34`):
```
MTK_BUILD_MANIFOLD(state_input,
  ((vect3, pos)) ((SO3, rot)) ((SO3, offset_R_L_I)) ((vect3, offset_T_L_I))
  ((vect3, vel)) ((vect3, bg)) ((vect3, ba)) ((vect3, gravity)) );
```
Error-state index layout (each vect3 = 3 DOF, SO3 = 3 DOF): `pos`[0:3], `rot`[3:6], `offset_R_L_I`[6:9], `offset_T_L_I`[9:12], `vel`[12:15], `bg`[15:18], `ba`[18:21], `gravity`[21:24]. Confirmed by `df_dx_input` block indices (`Estimator.cpp:80-96`) and `process_noise_cov_input` (`Estimator.cpp:26-40`).

**Output state** — 30 DOF (`common_lib.h:36-47`):
```
MTK_BUILD_MANIFOLD(state_output,
  ((vect3, pos)) ((SO3, rot)) ((SO3, offset_R_L_I)) ((vect3, offset_T_L_I))
  ((vect3, vel)) ((vect3, omg)) ((vect3, acc)) ((vect3, gravity))
  ((vect3, bg)) ((vect3, ba)) );
```
Error-state index layout: `pos`[0:3], `rot`[3:6], `offset_R_L_I`[6:9], `offset_T_L_I`[9:12], `vel`[12:15], **`omg`[15:18]**, **`acc`[18:21]**, `gravity`[21:24], **`bg`[24:27]**, **`ba`[27:30]**. The two *new* blocks at **15** and **18**, plus the biases at **24** and **27**, are exactly what the IMU measurement update touches: `update_iterated_dyn_share_IMU` reads `P_.col(15+l) + P_.col(24+l)` (`esekfom.hpp:262-263`), i.e. it observes `omg`/`acc` *and* `bg`/`ba` together because the residual contains both (§4.2).

**Why this matters (the core idea).** In FAST-LIO2 angular velocity and acceleration are *inputs* `u={a_m, ω_m}` consumed during propagation. In Point-LIO output mode they are promoted to **state** variables `omg, acc`; the IMU readings then become *noisy observations* of these states (plus biases). This single move (a) removes the need to de-skew and (b) makes the system robust to IMU saturation (a saturated reading is a measurement to be dropped — §4.3).

### 1.2 Input vector and process-noise manifolds
```
MTK_BUILD_MANIFOLD(input_ikfom, ((vect3, acc)) ((vect3, gyro)) );                                 // common_lib.h:49-52
MTK_BUILD_MANIFOLD(process_noise_input,  ((vect3,ng))((vect3,na))((vect3,nbg))((vect3,nba));        // :54-59
MTK_BUILD_MANIFOLD(process_noise_output, ((vect3,vel))((vect3,ng))((vect3,na))((vect3,nbg))((vect3,nba)); // :61-67
```
`process_noise_output` carries a `vel` noise term — in output mode the *velocity* equation gets injected noise too, because acceleration is now a (noisy) state rather than a directly-integrated input.

---

## 2. Continuous-time process model (the `f` functions)

`ẋ = f(x, u)` is returned (flattened to the manifold's *embedding* dimension `m = state::DIM`) by `get_f_*`. The IKFoM `predict()` integrates the mean by `x_.oplus(f_, dt)` (`esekfom.hpp:140-141`).

### 2.1 Input-mode dynamics — `get_f_input` (`Estimator.cpp:54-66`)
```
omega      = gyro ⊟ bg            // in.gyro.boxminus(omega, s.bg)
a_inertial = R · (acc − ba)
res(0:3)   = vel                  // ṗ
res(3:6)   = omega                // ṙot
res(12:15) = a_inertial + gravity // v̇el
```
Classical strap-down INS with IMU as input: ṗ=v, Ṙ=R⌊ω_m−b_g⌋, v̇=R(a_m−b_a)+g; biases/gravity constant in the mean.

### 2.2 Output-mode dynamics — `get_f_output` (`Estimator.cpp:68-78`)
```
a_inertial = R · acc              // acc is a STATE, not (a_m − ba)
res(0:3)   = vel                  // ṗ
res(3:6)   = omg                  // ṙot  ← driven by the STATE omg
res(12:15) = a_inertial + gravity // v̇el  ← driven by the STATE acc
```
**Crucially, `get_f_output` writes nothing into the `omg` (rows 15:18) or `acc` (rows 18:21) derivative slots — they stay zero.** So the *mean* continuous model is:
- ṗ = v
- Ṙ = R·⌊omg⌋
- v̇ = R·acc + g
- **ω̇ = 0 (+ n_ω, random walk)**, **ȧ = 0 (+ n_a, random walk)**, ġ=0, ḃg=0, ḃa=0.

This is the **stochastic process** driving the kinematics: angular velocity and acceleration are modeled as Brownian-like signals (zero-mean derivative + white-noise drive), with intensities `gyr_cov_output`, `acc_cov_output` (`Estimator.cpp:47-48`). The IMU then *measures* those signals (§4). This is the continuous-time view: the trajectory is an SDE in `(p,R,v,omg,acc,g,bg,ba)`, and both LiDAR points and IMU samples are irregular-rate observations of it.

### 2.3 Process-noise covariances
`process_noise_cov_input()` (`Estimator.cpp:26-40`): `[3:6]`(gyro→ṙot)=`gyr_cov_input`, `[12:15]`(accel→v̇el)=`acc_cov_input`, `[15:18]`(bg)=`b_gyr_cov`, `[18:21]`(ba)=`b_acc_cov`.

`process_noise_cov_output()` (`Estimator.cpp:42-52`): `[12:15]`(vel)=`vel_cov`, `[15:18]`(omg)=`gyr_cov_output`, `[18:21]`(acc)=`acc_cov_output`, `[24:27]`(bg)=`b_gyr_cov`, `[27:30]`(ba)=`b_acc_cov`.

Defaults (`parameters.cpp`): `use_imu_as_input=false`/`prop_at_freq_of_imu=true`/`check_satu=true` (`:15-16`); `vel_cov` default 20 (`:78`); `imu_meas_acc_cov`/`imu_meas_omg_cov` (`:84-85`); `laser_point_cov` default 0.01 (`:25`, overridden by yaml `lidar_meas_cov`, `:76`/`:105`); `match_s` default 81 (`:18`); `plane_thr` default 0.1 (`:19`); `imu_time_inte=0.005` / `lidar_time_inte=0.1` (`:24`,`:38`).

### 2.4 Linearized transition Jacobians (`df_dx_*`)
`df_dx_input` (`Estimator.cpp:80-96`):
```
∂ṗ/∂v    : block(0,12)  = I
∂v̇/∂R    : block(12,3)  = −R·⌊a_m ⊟ ba⌋
∂v̇/∂ba   : block(12,18) = −R
∂v̇/∂g    : block(12,21) = I
∂ṙot/∂bg : block(3,15)  = −I
```
`df_dx_output` (`Estimator.cpp:98-109`):
```
∂ṗ/∂v    : block(0,12)  = I
∂v̇/∂R    : block(12,3)  = −R·⌊acc⌋   (acc is a state)
∂v̇/∂acc  : block(12,18) = +R
∂v̇/∂g    : block(12,21) = I
∂ṙot/∂omg: block(3,15)  = +I          (omg is a state)
```
The sign/structure differences (`(12,18)=+R` vs `−R`; `(3,15)=+I` vs `−I`) reflect that the drivers are *states* in output mode (positive partials w.r.t. the state) versus *input−bias* in input mode (negative partials w.r.t. the bias). Note `df_dx_output` does NOT place anything in columns 24/27 (bias) for the `omg`/`acc` rows — biases couple to the state only through the IMU measurement update, not the propagation Jacobian.

---

## 3. On-manifold covariance propagation (`predict`)

`esekf::predict(double &dt, Q, const input &i_in, bool predict_state, bool prop_cov)` (`esekfom.hpp:137-182`). The two booleans **decouple mean propagation from covariance propagation** — the key lever that lets Point-LIO move the mean at every point but the covariance only at IMU rate (§5.3).

**Mean step** (`predict_state==true`, `esekfom.hpp:138-142`):
```
flatted_state f_ = f(x_, i_in);
x_.oplus(f_, dt);                 // x ⊞ (f·dt) on the manifold
```

**Covariance step** (`prop_cov==true`, `esekfom.hpp:144-181`):
1. `f_ = f(x_, i_in)` again; `f_x_ = f_x(x_, i_in)` (the `df_dx_*`); `F_x1 = I` (`:146-151`).
2. Scatter the vect-state rows of `f_x_` from embedding indices (`dim`) into DOF indices (`idx`): `f_x_final(idx+j,i)=f_x_(dim+j,i)` (`:152-160`).
3. For each SO3 sub-state: `seg_SO3 = −f_(dim:dim+3)·dt`; set the rotational transition block `F_x1.block<3,3>(idx,idx)=SO3::exp(seg_SO3)` (`:172`), and left-multiply the SO3 rows of `f_x_final` by the right-Jacobian `A_matrix(seg_SO3)` (`:173-176`).
4. `F_x1 += f_x_final · dt` (`:179`).
5. **Discrete covariance update:** `P_ = F_x1 · P_ · F_x1ᵀ + Q·(dt·dt)` (`esekfom.hpp:180`).

Two engineering notes worth flagging: (i) the noise term is `Q·dt²` (not `F·Q·Fᵀ` Van-Loan discretization); (ii) the `f_w` (noise-input Jacobian) path of stock IKFoM is **bypassed** — `f_w` is declared (`esekfom.hpp:321`) but its assignment is commented out in both `init_dyn_share_modified_2h/_3h` (`:111`,`:125`) and the propagation uses `Q·dt²` directly. These are deliberate Point-LIO simplifications of the IKFoM toolkit.

---

## 4. The IMU-as-measurement update (heart of Point-LIO output mode)

### 4.1 Residual / measurement model — `h_model_IMU_output` (`Estimator.cpp:324-368`)
```
z_IMU[0:3] = angvel_avr − omg − bg                 // gyro residual  (Estimator.cpp:327)
z_IMU[3:6] = acc_avr · G_m_s2/acc_norm − acc − ba  // accel residual (Estimator.cpp:328)
R_IMU      = [imu_meas_omg_cov×3, imu_meas_acc_cov×3]                 // (Estimator.cpp:329)
```
Model: `ω_m = omg + bg + n_g`, and (unit-normalized) `a_m·g/‖a‖ = acc + ba + n_a`. So the IMU directly observes `omg`[15:18], `acc`[18:21], `bg`[24:27], `ba`[27:30]. `angvel_avr`/`acc_avr` are the current raw IMU sample, set in the main loop from `imu_next` (`laserMapping.cpp:645-646`). The accel is normalized to gravity units by `G_m_s2/acc_norm` (`G_m_s2 = ‖gravity‖`, `laserMapping.cpp:471`).

### 4.2 The KF math — `update_iterated_dyn_share_IMU` (`esekfom.hpp:242-282`)
The observation Jacobian is **implicit and sparse**: the residual depends on `omg`/`bg` (rows 0:3) and `acc`/`ba` (rows 3:6), so `H` has `+(−I)` on columns 15..20 (omg/acc) **and** columns 24..29 (bg/ba). The code never forms `H` explicitly; it builds `PHᵀ` and `HP` by *summing the corresponding covariance columns/rows*:
```
for l in 0..5 (skip if satu_check[l]):
    PHT.col(l) = P_.col(15+l) + P_.col(24+l);    // P·Hᵀ  : col(omg/acc) + col(bg/ba)   (esekfom.hpp:262)
    HP.row(l)  = P_.row(15+l) + P_.row(24+l);    // H·P                                  (esekfom.hpp:263)
for l in 0..5:
    if !satu_check[l]: HPHT.col(l) = HP.col(15+l) + HP.col(24+l);    // H·P·Hᵀ           (esekfom.hpp:270)
    HPHT(l,l) += R_IMU(l);                                            // + R              (esekfom.hpp:272)
K   = PHT · HPHT.inverse();      // 30×6                                                 (esekfom.hpp:274)
dx_ = K · z_IMU;                                                                          (esekfom.hpp:276)
P_ -= K · HP;                    // covariance update                                     (esekfom.hpp:278)
x_.boxplus(dx_);                 // on-manifold correction                               (esekfom.hpp:279)
```
This is the standard EKF correction `K=PHᵀ(HPHᵀ+R)⁻¹`, `x⊞Kz`, `P←P−K·HP`, but exploiting the sparse `H` (only 6 measured rows touching 12 specific state columns) so that `PHᵀ` and `HPHᵀ` are assembled by column/row additions — extremely cheap, enabling kHz-rate IMU updates. The 6×6 `HPHT.inverse()` is the only matrix inversion.

> **Refinement vs my earlier reading:** the IMU update observes the *biases too* (the `+P_.col(24+l)` term), because the residual subtracts `bg`/`ba`. A naive "H = [0 I 0]" picture is incomplete; the implementation effectively uses `H` columns on both the (omg,acc) and (bg,ba) blocks.

### 4.3 Saturation handling (robustness to IMU clipping)
`h_model_IMU_output` checks each axis against `0.99·satu_gyro` / `0.99·satu_acc` (`Estimator.cpp:330-367`); if saturated it sets `satu_check[axis]=true` and zeros that residual entry. `update_iterated_dyn_share_IMU` then **skips that row** when assembling `PHT`/`HP`/`HPHT` (`esekfom.hpp:258-271`), so a saturated axis contributes *no* update — the state coasts on its random-walk prediction. This is the mechanism behind README's "robust to IMU saturation … (75 rad/s in our test)" (`README.md:16`).

---

## 5. The LiDAR per-point / per-batch update path

### 5.1 Time compression: grouping points that share a timestamp
`time_compressing<int>(feats_down_body)` (`laserMapping.cpp:496`; impl `common_lib.h:139-165`) walks the *time-sorted* down-sampled cloud and produces `time_seq`, a vector of run-lengths: each entry counts how many consecutive points share the *same* `curvature` (timestamp). Points are time-sorted via `sort(..., time_list)` first (`laserMapping.cpp:488`/`:493`; `time_list` compares `curvature`, `IMU_Processing.cpp:3`). A "batch" `k` = all points captured at the *same* sampling instant; for LiDARs with unique per-point times, each batch is one point. **`curvature` carries the per-point relative timestamp in milliseconds** (`preprocess.cpp:116` Livox `offset_time/1e6`; `:195`,`:244`,`:268`,`:394`,`:455` Velodyne/Ouster/Hesai `t·time_unit_scale`).

### 5.2 The point-by-point loop (output mode, `laserMapping.cpp:588-736`)
For `k = 0 … time_seq.size()-1` (with running `idx`):
1. **Exact sample time** of the batch: `point_body = feats_down_body[idx+time_seq[k]]; time_current = point_body.curvature/1000.0 + pcl_beg_time;` (`laserMapping.cpp:599-601`).
2. **Drain IMU samples up to `time_current`** (`laserMapping.cpp:622-675`): for each IMU sample with timestamp `< time_current`, load `angvel_avr`/`acc_avr` (`:645-646`); **mean-predict** to the IMU time `kf_output.predict(dt, Q_output, input_in, true, false)` (`:650`); then if `dt_cov>0` **cov-predict** `kf_output.predict(dt_cov, ..., false, true)` (`:661`) and **run the IMU measurement update** `kf_output.update_iterated_dyn_share_IMU()` (`:665`). This folds the IMU in as a measurement at *its own* rate.
3. **Propagate to the point's sample time** (`laserMapping.cpp:681-694`): `dt = time_current − time_predict_last_const`; if `!prop_at_freq_of_imu` optionally cov-predict (`:683-691`); then mean-predict `kf_output.predict(dt, Q_output, input_in, true, false)` (`:692`); set `time_predict_last_const = time_current` (`:694`).
4. **LiDAR update at the point time**: `kf_output.update_iterated_dyn_share_modified()` (`laserMapping.cpp:703`). On failure (no valid plane) skip the batch (`:703-707`).
5. **Re-register the batch's points to world** with the *just-updated* pose (`laserMapping.cpp:723-728` → `pointBodyToWorld`); `idx += time_seq[k]` (`:733`).

The **input-mode** loop (`laserMapping.cpp:808-927`) is structurally identical but sets `input_in = {acc, gyro}` from the IMU (`:835-837`,`:845-847`, with `input_in.acc *= G_m_s2/acc_norm`), uses `kf_input.predict(...)`, and has **no** `update_iterated_dyn_share_IMU` call (the IMU is the input, not a measurement).

### 5.3 Decoupled mean/covariance timestamps
Three "last-time" cursors: `time_predict_last_const` (mean), `time_update_last` (covariance), `t_last` (input-mode). The `predict` flag pair `(predict_state, prop_cov)` advances the **mean at every point** while advancing the **covariance only at IMU rate** when `prop_at_freq_of_imu` is true (default). This keeps the kHz point-rate affordable: cheap O(point-rate) mean propagation + sparse LiDAR/IMU updates, with the expensive 30×30 covariance propagation only at O(IMU-rate).

### 5.4 The LiDAR residual & Jacobian — `h_model_output` (`Estimator.cpp:218-322`) / `h_model_input` (`:112-216`)
For each point `j` in batch `k`: transform to world via `pointBodyToWorld` (uses *current* `kf_*.x_` pose, `Estimator.cpp:370-401`); 5-NN search in the iVox map `ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS=5)` (`Estimator.cpp:237`; `common_lib.h:82`); fit a plane `esti_plane(pabcd, points_near, plane_thr)` (`common_lib.h:202-234`: solves `A·x=−1`, normal `n=x/‖x‖`, `d=1/‖x‖`, rejects if any residual `>plane_thr`); point-to-plane distance `pd2=|n·p_world+d|`; accept iff `p_norm > match_s·pd2²` (range-adaptive gate, `Estimator.cpp:267`).

Residual & Jacobian per accepted point (`Estimator.cpp:291-319`):
```
z(m) = −(n·p_world + d)                                  // signed plane distance (Estimator.cpp:316)
C = Rᵀ·n
A = ⌊p_body⌋·C                  (or p_imu_crossmat·C with extrinsic est)
B = ⌊p_body⌋·offset_R_L_Iᵀ·C   (extrinsic only)
H row (1×12) = [ n(0:3), A(0:3), (B or 0)(0:3), (C or 0)(0:3) ]      // Estimator.cpp:307 / :314
```
The 12 Jacobian columns map to `[pos(0:3), rot(3:6), offset_T(6:9 or 0), offset_R(9:12 or 0)]`; only the first **12** error-state dims are touched, which is why the KF uses `P_.block<n,12>(0,0)` (§5.5). Per-point measurement noise is scalar `M_Noise = laser_point_cov` (`Estimator.cpp:286`). The commented-out block (`Estimator.cpp:249-266`) shows an *abandoned* per-point adaptive plane-noise weighting `noise_state = nᵀ(cov_p + R⌊p⌋cov_R⌊p⌋ᵀRᵀ)n + …`; the shipped code uses fixed `laser_point_cov` (refinement-vs-paper note). `cov_p`,`cov_R` are still passed in (`P_.block<3,3>(0,0)`, `P_.block<3,3>(3,3)`, `esekfom.hpp:192`) but only used by that dead code.

### 5.5 The LiDAR KF math — `update_iterated_dyn_share_modified` (`esekfom.hpp:184-240`)
```
h_dyn_share_modified_1(x_, P_.block<3,3>(0,0), P_.block<3,3>(3,3), dyn_share);  // → h_model_output/_input
if(!dyn_share.valid) return false;
z = dyn_share.z (dof×1);  h_x = dyn_share.h_x (dof×12);  m_noise = dyn_share.M_Noise;
if (n > dof_Measurement):                                    // few points: measurement-space gain
    PHT  = P_.block<n,12>(0,0) · h_xᵀ;                       // n×dof   (esekfom.hpp:215)
    HPHT = h_x · PHT.topRows(12);                            // dof×dof (esekfom.hpp:216)
    for m: HPHT(m,m) += m_noise;                              // + R     (esekfom.hpp:217-220)
    K_ = PHT · HPHT.inverse();                               // n×dof   (esekfom.hpp:221)
else:                                                         // many points: state-space gain (FAST-LIO2 trick)
    HTH = m_noise · h_xᵀ·h_x;                                // 12×12   (esekfom.hpp:225)
    P_inv = P_.inverse(); P_inv.block<12,12>(0,0) += HTH; P_inv = P_inv.inverse();   (esekfom.hpp:226-228)
    K_ = P_inv.block<n,12>(0,0) · h_xᵀ · m_noise;            (esekfom.hpp:229)
dx_ = K_ · z;  x_.boxplus(dx_);                              (esekfom.hpp:231-234)
P_  = P_ − K_·h_x·P_.block<12,n>(0,0);                       (esekfom.hpp:236)
```
`maximum_iter = 1` (set in both `init_dyn_share_modified_2h/_3h`, `esekfom.hpp:114`,`:129`) — so despite the "iterated" name, **each update is a single Gauss-Newton step per point**; the per-point cadence substitutes for per-scan iteration. The dual-branch gain is the FAST-LIO/2 "Kalman gain in state dimension" optimization (paper `2107.06829.txt:30`, "new Kalman gain formula … complexity from the dimension of the measurements to the dimension of the state"): for a single-point batch `dof=1 < n`, so the cheap measurement-space branch (one scalar `HPHT` inverse) is taken.

---

## 6. IMU init, sync, and the absence of de-skew

### 6.1 IMU init is stationary averaging only
`ImuProcess::IMU_init` (`IMU_Processing.cpp:70-100`) accumulates running means `mean_acc`,`mean_gyr` over `MAX_INI_COUNT=100` samples (`IMU_Processing.h:25`,`:117`). `Set_init` aligns gravity to the mean-acc direction via a single rotation (`IMU_Processing.cpp:40-68`). `ImuProcess::Process` (`:102-135`) **does not undistort** — it merely copies `*cur_pcl_un_ = *(meas.lidar)` (`:121`,`:128`,`:133`). No forward/backward propagation, no per-point compensation. This is the structural proof Point-LIO has no de-skew stage; the raw (distorted) cloud flows straight to the per-point estimator. Gravity alignment uses `Exp` of the angle between measured and reference gravity (`IMU_Processing.cpp:45-67`).

### 6.2 Packet sync preserves per-point times
`sync_packages` (`li_initialization.cpp:212-354`) bundles a LiDAR scan with the IMU samples spanning it; `lidar_end_time = lidar_beg_time + max(curvature)/1000` (`:281-291`). Each point keeps its own `curvature`, which the main loop converts to absolute time (`laserMapping.cpp:601`). `imu_cbk` applies `timediff_imu_wrt_lidar + time_lag_IMU_wtr_lidar` alignment (`li_initialization.cpp:190`), underscoring README notes A/C: IMU and LiDAR must be hardware-synchronized and per-point timestamps must be present ("Point-LIO processes at the sampling time of each LiDAR point", `README.md:23`,`:27`).

---

## 7. Continuous-time trajectory interpretation

Point-LIO is *not* a B-spline/Gaussian-process continuous-time estimator, but it realizes a **discrete-sample continuous-time SDE filter**:
- The state evolves by an SDE: `dp=v dt`, `dR=R⌊omg⌋dt`, `dv=(R·acc+g)dt+dW_v`, `d(omg)=dW_ω`, `d(acc)=dW_a`, `d(bg)=dW_bg`, `d(ba)=dW_ba` (from `get_f_output` zero-derivatives for omg/acc + `process_noise_cov_output`). The white-noise drives on `omg`/`acc` mean the *kinematic inputs themselves are continuous stochastic processes* — this is the "continuous-time / high-rate" angle.
- **Observations are irregular point processes** on that SDE: a LiDAR point at `t_L` (a plane constraint) and an IMU sample at `t_I` (a direct `omg`/`acc`/`bg`/`ba` observation) are each handled at *their own* time by predicting forward exactly `dt` and applying the appropriate sparse update. The trajectory is *sampled* at the true event times, not snapped to a fixed scan boundary.
- Each point is registered with the pose propagated to its own timestamp (not a single scan-end pose), so in-scan motion is captured by the filter dynamics — **this is the "no motion distortion" property** (`README.md:17`), achieved *implicitly* by per-point propagation rather than an explicit interpolation/undistortion model.
- Output rate is decoupled from scan rate: with `publish_odometry_without_downsample=true` odometry publishes at point rate (`laserMapping.cpp:710-714`,`:903-907`, stamped with `time_current`, `:277`), giving the README's "4k–8kHz" output (`README.md:15`).

---

## 8. Explicit contrast with FAST-LIO2

| Aspect | FAST-LIO2 | Point-LIO (output mode) |
|---|---|---|
| **State** | ~17/18-DOF; IMU is the *input* `u={a_m,ω_m}` | 30-DOF; **adds `omg`,`acc` as states** (`common_lib.h:36-47`) |
| **IMU role** | Control input during propagation | **Measurement** of `omg`/`acc`/`bg`/`ba` (`Estimator.cpp:324-368`, `esekfom.hpp:242-282`) |
| **Update granularity** | One iterated update **per scan** (`2107.06829.txt:214`,`:276`) | One single-step update **per point/batch** (`laserMapping.cpp:588-736`) |
| **Motion distortion** | Explicit **backward-propagation de-skew** into scan-end frame (`2107.06829.txt:196`; `FAST_LIO/src/IMU_Processing.hpp:307-...`, "undistort each lidar point (backward propagation)") | **None** — `ImuProcess::Process` just copies the cloud (`IMU_Processing.cpp:121,128`); per-point propagation handles motion implicitly |
| **Propagation** | Forward-integrate IMU over scan span, then de-skew (`2107.06829.txt:215`) | `predict()` to each event time with decoupled mean/cov flags (`esekfom.hpp:137-182`; `laserMapping.cpp:650,661,692`) |
| **Iteration** | Multiple Gauss-Newton iters per scan | `maximum_iter = 1` per point (`esekfom.hpp:114,129`) — cadence replaces iteration |
| **Kalman gain** | "State-dimension" gain formula (`2107.06829.txt:30`) | Same trick available (dual-branch `n>dof` in `esekfom.hpp:213-230`); per-point usually takes the cheap measurement-space branch |
| **IMU saturation** | Corrupts the input/propagation | Saturated axis is **dropped** from the measurement (`Estimator.cpp:330-367`; `esekfom.hpp:258-271`) |
| **Output rate** | Per-scan (~10–100 Hz, `2107.06829.txt:7`) | Per-point (4k–8kHz, `README.md:15`) |
| **Degeneracy** | Can drift under LiDAR degeneration | IMU-as-measurement keeps `omg`/`acc` observable; README: "would not fly under degeneration" (`README.md:14`) |
| **Map** | ikd-Tree (`2107.06829.txt:19`,`:39`) | iVox incremental voxel map (`Estimator.cpp:10`; `MapIncremental` `laserMapping.cpp:119-150`) |

**FAST-LIO de-skew detail (the step Point-LIO removes), grounded:** `UndistortPcl` (`FAST_LIO/src/IMU_Processing.hpp:216`) forward-propagates the EKF across each IMU interval, saving per-IMU poses in `IMUpose` (`:241`,`:295`), then iterates points and for each computes a per-point rotation `R_i = R_imu·Exp(angvel_avr, dt)` and position `T_ei = pos_imu + vel·dt + 0.5·acc·dt² − imu_state.pos`, transforming the point into the scan-end frame `P_compensate = R_imu_eᵀ(R_i·P_i + T_ei)` (`:307-334`). Point-LIO has **no analogue** of this loop anywhere in `IMU_Processing.cpp` or `laserMapping.cpp`. The FAST-LIO2 paper formalizes this as projecting every point to the scan-end time so the scan "can be viewed as all sampled simultaneously" (`2107.06829.txt:196`); Point-LIO instead never aggregates to a scan-end time at all.

**Note — Point-LIO also ships FAST-LIO2:** *input mode* (`use_imu_as_input=true`, 24-DOF `state_input`, `kf_input`) reproduces the FAST-LIO2 input formulation, but still **per-point** (`laserMapping.cpp:808-927`) and still **without de-skew**. So even in input mode the per-point cadence already removes explicit undistortion; the IMU-as-measurement model is the *additional* output-mode innovation that also handles saturation/degeneration.

---

## 9. Data structures (quick reference)

- `esekfom::dyn_share_modified<T>` (`esekfom.hpp:61-72`): `{bool valid, converge; T M_Noise; VectorX z; MatrixX h_x; Vector6 z_IMU; Vector6 R_IMU; bool satu_check[6];}` — one struct carrying *both* LiDAR (`z`,`h_x`,`M_Noise`) and IMU (`z_IMU`,`R_IMU`,`satu_check`) payloads.
- `esekfom::esekf<state, process_noise_dof, input>` (`esekfom.hpp:74-340`): public `state x_`, `cov P_` (`:308-309`); function pointers `f`,`f_x`,`f_w` (`:319-321`); three measurement hooks `h_dyn_share_modified_1/_2/_3` (`:329-333`); `maximum_iter` (`:335`). Template enum `n=state::DOF, m=state::DIM, l=measurement::DOF` (`:78-80`).
- `MeasureGroup` (`common_lib.h:113-125`): `{double lidar_beg_time, lidar_last_time; PointCloudXYZI::Ptr lidar; deque<Imu::ConstPtr> imu;}`.
- Global per-batch scratch (`Estimator.h:14-33`, defined `Estimator.cpp:4-24`): `time_seq`, `feats_down_body/world`, `pbody_list`, `Nearest_Points`, `crossmat_list`, `point_selected_surf[100000]`, `k`, `idx`, `angvel_avr`, `acc_avr`, `input_in`; the two filters `kf_input`/`kf_output` are globals (`Estimator.cpp:17-18`, declared `common_lib.h:69-70`).
- Map: `ivox_` (`Estimator.cpp:10`), an incremental voxel map (faster_lio iVox), queried by `GetClosestPoint(..., 5)` and grown by `MapIncremental()` (`laserMapping.cpp:119-150`, voxel-grid dedup by `filter_size_map_min`).
- Per-point timestamp carrier: `PointType.curvature` = relative time in **ms** (`preprocess.cpp:116` etc.); `PointType = pcl::PointXYZINormal` (`common_lib.h:93`).

---

## 10. Engineering choices to flag for the Meridian rebuild

1. **Two filters / two states behind one flag** is heavyweight: `get_f_input`/`get_f_output`, `df_dx_input`/`df_dx_output`, `h_model_input`/`h_model_output` are near-duplicated (`Estimator.cpp`). A unified extended state with switchable measurement models would be cleaner.
2. **`maximum_iter = 1`**: the "IKFoM iterated" filter is effectively a single-step EKF per event; correctness relies on the high event rate keeping linearization errors small. Decide whether Meridian keeps true iteration or relies on cadence.
3. **Decoupled mean/cov propagation flags** (`predict(..., predict_state, prop_cov)`, `esekfom.hpp:137`) plus `prop_at_freq_of_imu` are the key efficiency lever — preserve them.
4. **`Q·dt²` noise discretization** (`esekfom.hpp:180`) and the **bypassed `f_w`** path (`:111`,`:125`,`:321`) are simplifications vs a rigorous Van-Loan / `Fw·Q·Fwᵀ` discretization — candidates for tightening.
5. **Abandoned adaptive plane-noise** (`Estimator.cpp:249-266`, commented; `cov_p`,`cov_R` plumbed but unused) shows the authors considered per-point measurement-covariance weighting; the ship uses fixed `laser_point_cov`. Meridian could revive this.
6. **Saturation dropout via row-skipping** (`esekfom.hpp:258-271`) is elegant and cheap — keep it; also keep the bias-aware IMU `H` (cols 15+l AND 24+l).
7. **Hard sync requirement** (README A/C): per-point timing is meaningful only with synchronized, per-point-timestamped LiDAR; Meridian must validate the `time`/`curvature` field and reject clouds missing it.
8. **iVox vs ikd-Tree**: Point-LIO swapped FAST-LIO2's ikd-Tree for iVox (`Estimator.cpp:10,20`, `parameters.cpp:107-120`); a deliberate map-structure change for the high-rate regime.

---

## Appendix A — Output-state error vector (30-D) index map

| idx | block | meaning | touched by |
|----|------|---------|-----------|
| 0:3 | pos | position | LiDAR H col 0:3 (`Estimator.cpp:307/314`) |
| 3:6 | rot | attitude (SO3) | LiDAR H col 3:6; `df_dx_output`(3,15) |
| 6:9 | offset_R_L_I | extrinsic rot | LiDAR H col 9:12 if `extrinsic_est_en` |
| 9:12 | offset_T_L_I | extrinsic trans | LiDAR H col 6:9 if `extrinsic_est_en` |
| 12:15 | vel | velocity | `df_dx_output`(0,12),(12,3); process noise `vel_cov` |
| **15:18** | **omg** | **angular velocity (state)** | **IMU update col 15+l** (`esekfom.hpp:262`) |
| **18:21** | **acc** | **acceleration (state)** | **IMU update col 18+l (=15+3)** |
| 21:24 | gravity | gravity | `df_dx_output`(12,21)=I |
| **24:27** | **bg** | gyro bias | IMU residual `−bg` (`Estimator.cpp:327`); **IMU update col 24+l** (`esekfom.hpp:262`) |
| **27:30** | **ba** | accel bias | IMU residual `−ba` (`Estimator.cpp:328`); **IMU update col 27+l** |

---

## Appendix B — One-glance control flow (output mode, per scan)

```
sync_packages → p_imu->Process (copy cloud, NO undistort)        // li_initialization.cpp:212 ; IMU_Processing.cpp:102
  → downsample + sort by curvature + time_compressing(time_seq)  // laserMapping.cpp:486-497
  → for k in time_seq (running idx):                             // laserMapping.cpp:597
      time_current = curvature/1000 + pcl_beg_time               // :601
      while IMU sample < time_current:                           // :642
        predict(dt, mean=true,  cov=false)                       // :650
        predict(dt_cov, mean=false, cov=true)                    // :661
        update_iterated_dyn_share_IMU()    [IMU as measurement]  // :665
      predict(dt = time_current − last, mean=true, cov=false)    // :692
      update_iterated_dyn_share_modified() [LiDAR point update]  // :703
      re-register points to world with updated pose              // :723-728
  → MapIncremental() (iVox add) if feats_down_size>4             // :1013-1016
  → publish_odometry / path / frame                              // :1005-1022
```
