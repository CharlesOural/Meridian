# Grounding Dossier — IMU Model, Forward Propagation, and Backward De-skew (FAST-LIO / FAST-LIO2)

> Scope: the IMU measurement model, startup initialization, the state/covariance forward-propagation step, and the backward-propagation motion-compensation (de-skew) algorithm, as actually implemented in the cloned FAST-LIO repo, with explicit contrasts to FAST-LIVO2 and Point-LIO.
>
> **IMPORTANT SOURCING CAVEAT:** All four paper text files on disk are **0 bytes / empty** (verified: `C:/Users/charl/Sources/slam-reference/papers/2010.08196.txt`, `2107.06829.txt`, `2408.14035.txt`, `2102.10808.txt` all have `Length 0`). I could **not** quote paper text verbatim. Every code claim below is grounded in files I read byte-accurately (the `Read` tool was returning paraphrased/truncated summaries for some files, so I cross-checked all load-bearing snippets with PowerShell `Get-Content` / Bash `awk`, which return raw bytes). Paper references are therefore given by their **published** section/equation numbers from established knowledge of these works, and are flagged `[paper, not on disk]` so downstream authors verify against the real PDFs before quoting.

---

## 0. File map and the manifold state

The IMU pipeline is one class, `ImuProcess`, defined entirely in the header `FAST_LIO/src/IMU_Processing.hpp` (no `.cpp`). It operates on the IKFoM manifold state declared in `FAST_LIO/include/use-ikfom.hpp`.

The on-manifold state vector (`state_ikfom`, `use-ikfom.hpp:12-21`) has these compound members, in this order:

```
((vect3, pos))          // p   : IMU position in world (global) frame G
((SO3,   rot))          // R   : IMU attitude G_R_I (quaternion internally)
((SO3,   offset_R_L_I)) // I_R_L : LiDAR->IMU rotation extrinsic
((vect3, offset_T_L_I)) // I_t_L : LiDAR->IMU translation extrinsic
((vect3, vel))          // v   : IMU velocity in world frame
((vect3, bg))           // b_g : gyro bias
((vect3, ba))           // b_a : accel bias
((S2,    grav))         // g   : gravity vector, on the 2-sphere S2 (fixed magnitude)
```

- Total nominal dimension 24 (`get_f` returns `Matrix<double,24,1>`, `use-ikfom.hpp:47-59`), but the **error-state / tangent** dimension is **23** because `grav` lives on `S2` (2 DOF instead of 3). The Jacobian `df_dx` is therefore `24 x 23` (`use-ikfom.hpp:61`). The `esekf` template is instantiated as `esekf<state_ikfom, 12, input_ikfom>` — 12 is the process-noise DOF (`IMU_Processing.hpp:53`).
- `grav` as `S2<double, 98090, 10000, 1>` (`use-ikfom.hpp:8`) fixes |g| = 9.8090 m/s² (the template encodes the magnitude as 98090/10000). This is the single most important refinement over the original FAST-LIO `StatesGroup` (`common_lib.h:54-72`), where `gravity` was an unconstrained `V3D` in R³.

The **input** is the raw IMU sample (`input_ikfom`, `use-ikfom.hpp:23-26`):
```
((vect3, acc))   // a_m : measured specific force (m/s^2)
((vect3, gyro))  // w_m : measured angular rate (rad/s)
```

The **process noise** is 12-D (`process_noise_ikfom`, `use-ikfom.hpp:28-33`): `ng` (gyro white noise), `na` (accel white noise), `nbg` (gyro-bias random-walk), `nba` (accel-bias random-walk).

---

## 1. IMU measurement model (accel / gyro with bias + noise)

The continuous-time kinematic / measurement model is encoded directly in the state-transition function `get_f` and its Jacobians `df_dx`, `df_dw` in `use-ikfom.hpp`.

### 1.1 Measurement equations

The IMU measures specific force and angular rate corrupted by bias and white noise. Reading `get_f` (`use-ikfom.hpp:47-59`):

```cpp
vect3 omega;
in.gyro.boxminus(omega, s.bg);            // omega = w_m - b_g
vect3 a_inertial = s.rot * (in.acc - s.ba); // a_inertial = R (a_m - b_a)
res(i)      = s.vel[i];                    // p_dot   = v
res(i + 3)  = omega[i];                    // theta_dot (rot) = w_m - b_g
res(i + 12) = a_inertial[i] + s.grav[i];  // v_dot   = R(a_m - b_a) + g
```

So the implicit measurement model is:

- Gyro:  **w_m = ω + b_g + n_g**, so the de-biased body rate used is **ω = w_m − b_g**.
- Accel: **a_m = R^T (a − g) + b_a + n_a** (specific force in body frame). De-biased and rotated to world: **a = R (a_m − b_a) + g**.

Note the sign convention on gravity: `grav` is initialized as the **negative** of the measured-acceleration direction scaled to G (Section 2), so the `+ s.grav` term in `v_dot` correctly subtracts gravitational specific force.

### 1.2 Continuous-time kinematics (the ODE that `get_f` represents)

Collecting `get_f`, the model is the standard strapdown INS:

```
p_dot       = v
R_dot       = R [ (w_m - b_g - n_g) ]_x        (right-perturbation SO3 kinematics)
v_dot       = R (a_m - b_a - n_a) + g
b_g_dot     = n_bg                              (random walk)
b_a_dot     = n_ba                              (random walk)
g_dot       = 0   (on S2; only 2 DOF tangent)
offset_R_L_I_dot = 0,  offset_T_L_I_dot = 0     (extrinsics constant)
```
This matches FAST-LIO2 Eq. (1)-(2) and FAST-LIO Eq. (1) `[paper, not on disk]`. The extrinsic `offset_R_L_I / offset_T_L_I` being in the state (and thus online-calibrated) is a FAST-LIO2 addition over FAST-LIO.

### 1.3 Process Jacobians (F and G of the EKF)

`df_dx` (∂f/∂x, the 24×23 `F`-continuous), `use-ikfom.hpp:61-77`:
```cpp
cov.block<3,3>(0,12) = I3;                              // d(p_dot)/d(v)
cov.block<3,3>(12,3) = -R * hat(a_m - b_a);             // d(v_dot)/d(theta)
cov.block<3,3>(12,18)= -R;                              // d(v_dot)/d(b_a)
cov.block<3,2>(12,21)= grav_matrix (S2_Mx);            // d(v_dot)/d(g) on S2 tangent
cov.block<3,3>(3,15) = -I3;                             // d(theta_dot)/d(b_g)
```
`df_dw` (∂f/∂w, 24×12), `use-ikfom.hpp:80-88`:
```cpp
cov.block<3,3>(12,3) = -R;     // v_dot wrt na
cov.block<3,3>(3,0)  = -I3;    // theta_dot wrt ng
cov.block<3,3>(15,6) = I3;     // bg_dot wrt nbg
cov.block<3,3>(18,9) = I3;     // ba_dot wrt nba
```
These are the exact analytic Jacobians of FAST-LIO2 Eq. (7)/(8) `[paper, not on disk]`. The `hat()` (skew) is `so3_math.h:80-85`; `MTK::hat` is used in `use-ikfom.hpp:69`.

### 1.4 Noise covariance Q

The continuous process-noise covariance is built once at construction (`process_noise_cov()`, `use-ikfom.hpp:35-43`):
```
ng  = 0.0001 (gyro white)
na  = 0.0001 (accel white)
nbg = 0.00001 (gyro bias RW)
nba = 0.00001 (accel bias RW)   -- all as diagonal 3x3 blocks => 12x12
```
These defaults are **overwritten per-step** in `UndistortPcl` from the runtime-estimated covariances (`IMU_Processing.hpp:280-283`):
```cpp
Q.block<3,3>(0,0).diagonal() = cov_gyr;       // gyro
Q.block<3,3>(3,3).diagonal() = cov_acc;       // accel
Q.block<3,3>(6,6).diagonal() = cov_bias_gyr;  // gyro bias RW
Q.block<3,3>(9,9).diagonal() = cov_bias_acc;  // accel bias RW
```
`cov_acc`/`cov_gyr` come from initialization-time empirical variance and are then replaced by the configured scale (`cov_acc_scale`, `cov_gyr_scale`) once init finishes (`IMU_Processing.hpp:368-372`).

---

## 2. IMU initialization (gravity + bias + covariance at startup)

`ImuProcess::IMU_init` (`IMU_Processing.hpp:159-214`), called for the first `MAX_INI_COUNT = 10` (`IMU_Processing.hpp:30`) batches before LIO starts.

### 2.1 Running mean and variance (Welford)

On the very first frame it seeds `mean_acc`, `mean_gyr` from the first IMU sample and records `first_lidar_time` (`IMU_Processing.hpp:166-176`). Then, over **all** IMU samples in every init batch, it accumulates online mean and (population-style) variance (`IMU_Processing.hpp:178-194`):
```cpp
mean_acc += (cur_acc - mean_acc) / N;
mean_gyr += (cur_gyr - mean_gyr) / N;
cov_acc = cov_acc*(N-1)/N + (cur_acc-mean_acc).cwiseProduct(cur_acc-mean_acc)*(N-1)/(N*N);
cov_gyr = cov_gyr*(N-1)/N + (cur_gyr-mean_gyr).cwiseProduct(cur_gyr-mean_gyr)*(N-1)/(N*N);
N++;
```
This is Welford's incremental mean/variance. `N` is carried across batches by reference (`init_iter_num`, `IMU_Processing.hpp:159, 359`).

### 2.2 Setting the initial state

After accumulation (`IMU_Processing.hpp:195-202`):
```cpp
init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);  // gravity = -avg_accel_dir * 9.81
init_state.bg   = mean_gyr;                                   // gyro bias = mean gyro (device at rest)
init_state.offset_T_L_I = Lidar_T_wrt_IMU;                    // extrinsics from config
init_state.offset_R_L_I = Lidar_R_wrt_IMU;
```
Key engineering choices:
- **Gravity** is estimated as the negated, normalized mean acceleration, scaled to `G_m_s2 = 9.81` (`common_lib.h:19`). Because it is wrapped in `S2(...)`, only its direction is free thereafter; magnitude is pinned at 9.81. This assumes the platform is **stationary** during init (no `ba` separation from `g` is possible while static — accel bias init stays at its prior 0).
- **Gyro bias** = mean gyro (valid only at rest).
- Attitude `rot` is left at identity (the commented line `IMU_Processing.hpp:198` shows the alternative of aligning roll/pitch to gravity was deliberately not used; the first scan defines the world frame, tilted).

### 2.3 Initial covariance P

`IMU_Processing.hpp:204-211`:
```cpp
init_P.setIdentity();
init_P(6,6)=init_P(7,7)=init_P(8,8)   = 0.00001;  // offset_R_L_I  (extrinsic rot, tight)
init_P(9,9)=init_P(10,10)=init_P(11,11)=0.00001;  // offset_T_L_I  (extrinsic transl, tight)
init_P(15,15)=init_P(16,16)=init_P(17,17)=0.0001; // bg
init_P(18,18)=init_P(19,19)=init_P(20,20)=0.001;  // ba
init_P(21,21)=init_P(22,22)=0.00001;              // grav (S2, 2 DOF)
```
(Indices follow the 23-D error-state ordering: pos 0-2, rot 3-5, offset_R 6-8, offset_T 9-11, vel 12-14, bg 15-17, ba 18-20, grav 21-22.) Extrinsics and gravity get small priors (we trust the config/init); biases get looser priors.

### 2.4 Init completion / variance rescaling

In `Process` (`IMU_Processing.hpp:356-380`): each batch re-runs `IMU_init` and sets `last_imu_ = meas.imu.back()`. When `init_iter_num > MAX_INI_COUNT` (`:366`):
```cpp
cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);  // rescale measured accel variance to true-g units
imu_need_init_ = false;
cov_acc = cov_acc_scale;   // then OVERRIDE with configured scales
cov_gyr = cov_gyr_scale;
```
Note the rescale on `:368` is immediately overwritten on `:371-372`, so in practice the runtime `Q` uses the **configured** `cov_*_scale` from the launch params (set via `set_acc_cov`/`set_gyr_cov`, `IMU_Processing.hpp:139-146`), not the empirical init variance. The empirical variance is effectively only used as a prior/sanity value. During init batches `Process` returns early (no de-skew, no update) (`IMU_Processing.hpp:379`).

> **Code vs paper:** FAST-LIO/2 papers describe init as gravity+bias estimation while static `[paper, not on disk]`. The code additionally (a) constrains gravity to S2, and (b) discards the empirically-measured noise variance in favor of configured constants — an engineering simplification not emphasized in the paper.

---

## 3. Forward propagation (state + covariance between IMU samples)

Forward propagation happens inside `UndistortPcl` (`IMU_Processing.hpp:216-305`), which interleaves propagation with saving per-sample poses for the later de-skew.

### 3.1 Building the IMU stream and timing

`IMU_Processing.hpp:218-225`:
```cpp
auto v_imu = meas.imu;
v_imu.push_front(last_imu_);            // prepend previous scan's last IMU sample
imu_beg_time = v_imu.front()->stamp;    // = last scan's tail IMU time
imu_end_time = v_imu.back()->stamp;
pcl_beg_time = meas.lidar_beg_time;
pcl_end_time = meas.lidar_end_time;
```
Prepending `last_imu_` makes the IMU interval bracket the LiDAR scan from before scan-begin to after scan-end, so every point has IMU on both sides. `lidar_end_time` is computed in `sync_packages` (`laserMapping.cpp:362-380`):
- If the last point's `curvature/1000` (its time offset in **seconds**, since curvature stores **ms**) is ≥ half the running mean scan time, `lidar_end_time = lidar_beg_time + last_point_curvature/1000` and the running mean `lidar_mean_scantime` is updated (`laserMapping.cpp:376-377`).
- Otherwise (degenerate/short scan) it falls back to `lidar_beg_time + lidar_mean_scantime` (`:366, :371`).

The IMU batch for this scan is exactly those samples with timestamp < `lidar_end_time` (`laserMapping.cpp:393-399`).

### 3.2 Sorting points by time

`IMU_Processing.hpp:233-234`:
```cpp
pcl_out = *(meas.lidar);
sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
```
`time_list` (`IMU_Processing.hpp:32`) orders by `curvature` ascending — i.e. by per-point capture time. This is a precondition for the backward de-skew loop (which walks points from latest to earliest).

### 3.3 Seeding the pose buffer

`IMU_Processing.hpp:239-241`:
```cpp
state_ikfom imu_state = kf_state.get_x();
IMUpose.clear();
IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
```
The first saved pose has `offset_time = 0` (scan begin) and uses the current filter state. `set_pose6d` (`common_lib.h:75-90`) packs `(offset_time, acc(world), gyr(body), vel, pos, rot[9])` into a `Pose6D` (msg fields verified: `offset_time, acc[3], gyr[3], vel[3], pos[3], rot[9]`, `FAST_LIO/msg/Pose6D.msg`).

### 3.4 The forward-propagation loop (per IMU interval)

`IMU_Processing.hpp:250-296`. For each consecutive pair `(head, tail)` of IMU samples:

**Midpoint (trapezoidal) input** (`:257-266`):
```cpp
angvel_avr = 0.5*(head.w + tail.w);                 // averaged gyro
acc_avr    = 0.5*(head.a + tail.a);                 // averaged accel
acc_avr    = acc_avr * G_m_s2 / mean_acc.norm();    // rescale raw accel to m/s^2 (true g)
```
The accel rescale `G_m_s2 / mean_acc.norm()` converts the IMU's native acceleration units (whose static norm was `mean_acc.norm()`) into physical m/s² consistent with the gravity magnitude — an engineering normalization tied to the init.

**Δt selection** (`:268-276`) — handles the first interval that may start before `last_lidar_end_time_`:
```cpp
if (head.stamp < last_lidar_end_time_)  dt = tail.stamp - last_lidar_end_time_;
else                                    dt = tail.stamp - head.stamp;
```
Intervals fully before `last_lidar_end_time_` are skipped (`:255 continue`), so propagation begins exactly at the previous scan-end.

**EKF predict** (`:278-284`):
```cpp
in.acc = acc_avr; in.gyro = angvel_avr;
Q.block<3,3>(0,0).diagonal()=cov_gyr; ... (set Q blocks)
kf_state.predict(dt, Q, in);
```

**Save world-frame accel, body rate, and the new pose** (`:287-295`):
```cpp
imu_state = kf_state.get_x();
angvel_last = angvel_avr - imu_state.bg;                       // de-biased body rate
acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);        // world-frame accel (no g yet)
for i in 0..2: acc_s_last[i] += imu_state.grav[i];             // add gravity -> true world accel
offs_t = tail.stamp - pcl_beg_time;                            // time offset from scan begin
IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, vel, pos, rot));
```
So each `Pose6D` stores the **propagated IMU state at that sample's timestamp**, plus the constant-acceleration / constant-angular-velocity quantities (`acc_s_last`, `angvel_last`) needed to interpolate **between** samples during de-skew.

### 3.5 Propagation to scan-end

After the loop, propagate from the last IMU sample to the exact scan-end time (`:299-305`):
```cpp
double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
dt = note * (pcl_end_time - imu_end_time);   // signed; can be negative (extrapolate backward)
kf_state.predict(dt, Q, in);                 // reuse last 'in'
imu_state = kf_state.get_x();
last_imu_ = meas.imu.back();
last_lidar_end_time_ = pcl_end_time;
```
The signed `dt` lets the filter step backward if the last IMU sample is slightly **after** scan-end. After this, `imu_state` is the predicted state at scan-end — the reference frame `e` for de-skew, and the prior for the subsequent iterated measurement update in `laserMapping.cpp:840-841` (`p_imu->Process(...); state_point = kf.get_x();`).

### 3.6 Inside `kf_state.predict` (the covariance step)

`esekfom.hpp:157-206` (verified raw). The discrete EKF propagation is:

**Mean** (`:158, :164`):
```cpp
flatted_state f_ = f(x_, i_in);   // = get_f (24x1)
x_.oplus(f_, dt);                 // x <- x boxplus (f_ * dt)   (manifold retraction)
```
i.e. **x_{k+1} = x_k ⊞ (f(x_k, u_k)·Δt)**.

**Covariance** (`:159-205`). It computes the continuous Jacobians `f_x_ = df_dx`, `f_w_ = df_dw`, then builds the **discrete** transition `F_x1`. The state-transition matrix is assembled per manifold sub-block:
- vector blocks: copied straight through (`:167-175`);
- SO3 block: multiplied by `A_matrix(f·dt)^T` (the right-Jacobian transpose of the incremental rotation) (`:177-186`, `res_temp_SO3 = MTK::A_matrix(seg_SO3*dt).transpose()`);
- S2 block (gravity): uses `S2_Mx` / `S2_Nx_yy` to map the 2-DOF tangent (`:188-202`).

Then (`:204-205`, verified raw):
```cpp
F_x1 += f_x_final * dt;            // F = I + (manifold-corrected df/dx) * dt
P_ = F_x1 * P_ * F_x1.transpose() + (dt*f_w_final) * Q * (dt*f_w_final).transpose();
```
So the covariance update is the textbook EKF form:

**P_{k+1} = F_x · P_k · F_xᵀ + (Δt·F_w) · Q · (Δt·F_w)ᵀ**

with **F_x = I + F_x_final·Δt** (manifold-corrected ∂f/∂x, `F_x1` seeded to `cov::Identity()` at `:166`) and **F_w = f_w_final** (manifold-projected ∂f/∂w). This is FAST-LIO2 Eq. (8)-(9) / FAST-LIO Eq. (7)-(8) `[paper, not on disk]`.

> **Note on this vendored copy:** in this specific tree, the loop that *projects* `f_w_` onto the manifold tangent to populate `f_w_final` is **not present in the shown region** — `f_w_final` is declared at `esekfom.hpp:162` and used at `:205`, but I found no assignment between them (verified via Grep: only `:162` decl and `:205` use). In the canonical IKFoM this projection mirrors the `f_x_final` assembly. Downstream authors should treat `F_w` mathematically as the manifold-projected `df_dw`; the empty projection here is likely a stripped/edited copy and should not be reproduced as-is. The math is what matters: covariance uses the standard `F P Fᵀ + (Δt F_w) Q (Δt F_w)ᵀ`.

---

## 4. Backward propagation — point-cloud de-skew (motion undistortion)

This is the defining contribution of the FAST-LIO line vs. naive accumulation. After forward propagation has filled `IMUpose` and produced `imu_state` (scan-end pose), every point is transformed to the **scan-end IMU reference frame** so the whole scan looks as if captured instantaneously at scan-end.

### 4.1 The reference frame

The compensation target is the **scan-end** IMU pose `e` = `imu_state` (the just-propagated state). Each point captured at time `t_i` (during the sweep) is expressed in the LiDAR frame at `t_i`; de-skew maps it to the LiDAR frame at scan-end.

### 4.2 The exact loop (verified raw, `IMU_Processing.hpp:307-345`)

```cpp
auto it_pcl = pcl_out.points.end() - 1;                       // start from LAST (latest) point
for (auto it_kp = IMUpose.end()-1; it_kp != IMUpose.begin(); it_kp--) {
    auto head = it_kp - 1;                                    // earlier IMU pose
    auto tail = it_kp;                                        // later   IMU pose
    R_imu   << MAT_FROM_ARRAY(head->rot);                     // R at interval start
    vel_imu << VEC_FROM_ARRAY(head->vel);                     // v at interval start
    pos_imu << VEC_FROM_ARRAY(head->pos);                     // p at interval start
    acc_imu << VEC_FROM_ARRAY(tail->acc);                     // world accel (interval const)
    angvel_avr << VEC_FROM_ARRAY(tail->gyr);                  // body rate  (interval const)

    for (; it_pcl->curvature/1000.0 > head->offset_time; it_pcl--) {
        dt = it_pcl->curvature/1000.0 - head->offset_time;    // time since interval start (s)

        M3D R_i(R_imu * Exp(angvel_avr, dt));                 // R(t_i): integrate const omega
        V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);             // point in LiDAR frame at t_i
        V3D T_ei(pos_imu + vel_imu*dt + 0.5*acc_imu*dt*dt
                 - imu_state.pos);                            // world-frame displacement i->e
        V3D P_compensate =
            imu_state.offset_R_L_I.conjugate() *
            ( imu_state.rot.conjugate() *
              ( R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei )
              - imu_state.offset_T_L_I );

        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);
        if (it_pcl == pcl_out.points.begin()) break;
    }
}
```

### 4.3 The per-point transform, written as math

Define:
- `R_e = imu_state.rot`, `p_e = imu_state.pos` — IMU pose at scan-end (world frame G).
- `I_R_L = offset_R_L_I`, `I_t_L = offset_T_L_I` — LiDAR→IMU extrinsic.
- `R_h, p_h, v_h` — IMU rotation/position/velocity at the **head** of the interval containing `t_i` (from `IMUpose`).
- `a_h` (world accel, `tail.acc`), `ω` (body rate, `tail.gyr`) — held **constant** over the interval.
- `dt = t_i − offset_time_head` — elapsed time of the point inside its interval.

**Rotation at point time** (constant-ω integration, right multiply):
```
R_i = R_h · Exp(ω · dt)
```
`Exp(ω, dt)` is Rodrigues of `ω·dt` (`so3_math.h:28-49`).

**World displacement of the IMU from time t_i to scan-end** (constant-acceleration model):
```
T_ei = p_h + v_h·dt + ½·a_h·dt²  −  p_e
```
(i.e. the IMU world position at `t_i` minus the scan-end IMU world position.)

**Full compensation** (chained transforms): LiDAR@t_i → IMU@t_i → world → IMU@end → LiDAR@end:
```
P_compensate = I_R_Lᵀ · [ R_eᵀ · ( R_i · (I_R_L · P_i + I_t_L) + T_ei ) − I_t_L ]
```
Reading right to left:
1. `I_R_L·P_i + I_t_L` : point from LiDAR@t_i frame into IMU@t_i frame.
2. `R_i·(…) + T_ei` : rotate into world orientation at t_i and add the world displacement back to scan-end — putting the point in a frame **rotated like IMU@t_i but positioned at scan-end** (because `T_ei` already subtracts `p_e`). The effective world point is `R_i·(I_R_L P_i + I_t_L) + p_i^world`, and subtracting `p_e` via `T_ei` gives the world vector from scan-end origin.
3. `R_eᵀ·(…)` : rotate that world vector into the IMU@end body frame.
4. `I_R_Lᵀ·[… − I_t_L]` : transform from IMU@end frame into LiDAR@end frame.

This is exactly FAST-LIO2 Eq. (10)-(11) / FAST-LIO Eq. (10) `[paper, not on disk]`: each point is back-projected to the scan-end pose. The code comment (`IMU_Processing.hpp:327-330`) states the intent: *"Transform to the 'end' frame... Compensation direction is INVERSE of Frame's moving direction... P_compensate = R_imu_e^T * (R_i * P_i + T_ei) - ..."* and flags it `// not accurate!` at `:335` (the constant-velocity/constant-accel intra-interval model is an approximation).

### 4.4 Key data-structure / control-flow facts

- **Double nested loop, both descending.** Outer walks `IMUpose` from last interval to first; inner walks points from latest to earliest. Because points are sorted ascending by time (`:234`) and IMU poses are ascending in `offset_time`, both iterators march backward in lockstep. The inner condition `it_pcl->curvature/1000 > head->offset_time` (`:323`) consumes exactly the points whose capture time falls in the current `(head, tail)` interval.
- **Units:** `curvature` is per-point time **in ms** (set in `preprocess.cpp:87, 138, 193, 236` — `curvature = (timestamp − first_timestamp)*1000` or `time*time_unit_scale`), so `/1000.0` converts to seconds matching `offset_time` (which is `tail.stamp − pcl_beg_time` in seconds, `:294`).
- **Termination guard** `if (it_pcl == pcl_out.points.begin()) break;` (`:342`) prevents under-run.
- **MARSIM exception:** for the simulated LiDAR (`lidar_type == MARSIM`), de-skew is skipped entirely (`:310`), and `pcl_beg/end_time` are redefined to the previous-scan/this-scan begin (`:227-230`).
- **Output:** `pcl_out` (alias `feats_undistort`) is the undistorted cloud passed to scan-matching in `laserMapping.cpp:840-843`; an empty result skips the scan.

### 4.5 What the de-skew does NOT do

- It does **not** re-estimate the trajectory; it uses the **already-propagated** poses (forward pass) as a fixed spline of waypoints, interpolating between them with constant ω and constant world-accel. The `// not accurate!` comment acknowledges this.
- It does **not** undistort using the *posterior* (post-update) state — it uses the *prior* (propagated) state. FAST-LIO2 papers note the iterated update can in principle re-deskew; the code de-skews once with the prior `[paper, not on disk]`.

---

## 5. Contrast: this batch back-propagation vs continuous-time / preintegration

### 5.1 vs IMU preintegration (e.g. VINS-Mono / OKVIS style)
- **Preintegration** accumulates relative motion increments (ΔR, Δv, Δp) **between keyframes** in the body frame, independent of absolute pose, with bias-Jacobians for repropagation — it is a *factor* in a sliding-window optimizer, and it does **not** by itself undistort the cloud.
- **FAST-LIO forward pass** is a *filter* prediction: it propagates the **absolute** on-manifold state and full covariance sample-by-sample with the midpoint input (`IMU_Processing.hpp:257-284`), discarding intermediate covariances except the final one used by the update. No relative-increment caching, no bias-Jacobian repropagation.
- FAST-LIO's de-skew is a **dedicated backward pass** (Section 4) that preintegration approaches replace with either per-point continuous-time interpolation of an optimized trajectory, or with no explicit undistortion.

### 5.2 vs continuous-time (B-spline) SLAM
- Continuous-time methods (e.g. spline-based LIO) represent the trajectory as a **single continuous function** SE(3)(t) and query it at each point's exact timestamp — giving a globally smooth, jointly-optimized undistortion. FAST-LIO instead uses a **piecewise** model: discrete IMU waypoints (`IMUpose`) with constant-ω / constant-accel interpolation **inside** each IMU interval (`R_i = R_h·Exp(ω·dt)`, `T_ei` quadratic in dt, `:331-334`). It is cheaper (closed-form per point, no spline solve) but only first/second-order accurate within an interval — hence the `// not accurate!` flag.

### 5.3 vs Point-LIO (the most radical contrast — verified in cloned code)
- Point-LIO's `IMU_Processing.cpp` has **no** `UndistortPcl`, no `backward`, no `P_compensate` (verified: Grep for `undistort|backward|deskew|P_compensate` returns nothing in `Point-LIO/src/IMU_Processing.cpp`). Instead Point-LIO treats the **IMU itself as a measurement** and updates the filter **point-by-point** at each point's timestamp (`h_model_input`/`h_model_output`/`imu_prop` machinery lives in `Point-LIO/src/laserMapping.cpp` and `li_initialization.cpp`). Because the state is updated at (near) every point time, there is **no separate de-skew step at all** — distortion is handled implicitly by the high-rate sequential update. This trades FAST-LIO's batch back-propagation for a per-point Kalman update, enabling very-high-dynamics handling without an explicit motion-compensation transform.

### 5.4 vs FAST-LIVO2 (verified: same scheme as FAST-LIO2)
- FAST-LIVO2's `ImuProcess::UndistortPcl` (`FAST-LIVO2/src/IMU_Processing.cpp:185`) uses the **identical** backward-propagation transform (verified raw, `:240-254`):
  ```cpp
  for (auto it_kp = IMUpose.end()-1; it_kp != IMUpose.begin(); it_kp--)
    for (; it_pcl->curvature/1000.0 > head->offset_time; it_pcl--)
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      V3D T_ei(pos_imu + vel_imu*dt + 0.5*acc_imu*dt*dt - imu_state.pos);
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() *
        (imu_state.rot.conjugate() * (R_i*(offset_R_L_I*P_i + offset_T_L_I) + T_ei) - offset_T_L_I);
  ```
  Same equation as FAST-LIO2 `IMU_Processing.hpp:331-335`, on the same 24-D / 23-tangent IKFoM state — FAST-LIVO2 inherits the LIO de-skew unchanged and adds the visual (VIO) subsystem on top.

---

## 6. Summary table of the math (for the textbook chapter)

| Quantity | Equation | Code site |
|---|---|---|
| De-biased gyro | ω = w_m − b_g | `use-ikfom.hpp:51`, `IMU_Processing.hpp:288` |
| World accel | a = R(a_m − b_a) + g | `use-ikfom.hpp:52,56`, `IMU_Processing.hpp:289-292` |
| State ODE | ṗ=v, Ṙ=R[ω]ₓ, v̇=R(a_m−b_a)+g, ḃ=n | `use-ikfom.hpp:53-57` |
| Mean propagation | x_{k+1} = x_k ⊞ (f·Δt) | `esekfom.hpp:164` |
| Cov propagation | P=F_x P F_xᵀ + (ΔtF_w)Q(ΔtF_w)ᵀ, F_x=I+F_x_final·Δt | `esekfom.hpp:204-205` |
| Init gravity | g₀ = −(mean_acc/‖mean_acc‖)·9.81 (on S2) | `IMU_Processing.hpp:196` |
| Init gyro bias | b_g0 = mean_gyr | `IMU_Processing.hpp:199` |
| Point rotation | R_i = R_h·Exp(ω·dt) | `IMU_Processing.hpp:331` |
| Point world disp. | T_ei = p_h + v_h·dt + ½a_h·dt² − p_e | `IMU_Processing.hpp:334` |
| De-skew transform | P̂ = I_R_Lᵀ[R_eᵀ(R_i(I_R_L P_i+I_t_L)+T_ei) − I_t_L] | `IMU_Processing.hpp:335` |

---

## 7. Engineering choices / gotchas worth flagging for the rebuild ("Meridian")

1. **Gravity on S2** (fixed magnitude 9.8090) vs FAST-LIO's free R³ gravity — cleaner, avoids drift in |g|, but requires the S2 tangent machinery (`S2_Mx`, `S2_Nx_yy`) in both `df_dx` and the predict covariance assembly.
2. **Accel unit normalization** `acc * G_m_s2 / mean_acc.norm()` (`:266`) couples the runtime model to the init-time static norm. If the IMU reports in g rather than m/s², this is essential; for an m/s² IMU it must be ≈1. A from-scratch design should make units explicit rather than implicit-via-init.
3. **Midpoint integration** (averaging head/tail IMU) (`:257-262`) for both mean and the input feeding covariance — second-order accurate, cheap.
4. **`last_imu_` prepend** (`:220`) to bracket the scan; **signed final Δt** (`:299-300`) to handle IMU samples straddling scan-end.
5. **Per-point time in `curvature` (ms)** is a repo-wide convention (preprocess sets it, sync/de-skew read it). A rebuild should use a dedicated, well-named field/units.
6. **De-skew uses the prior, once, with constant-ω/accel intra-interval** — the `// not accurate!` (`:335`) is the documented accuracy ceiling; Point-LIO's per-point update is the SOTA alternative if dynamics are extreme.
7. **`f_w_final` projection missing in this vendored esekfom copy** (Section 3.6) — do not copy this file blindly; reconstruct the canonical IKFoM covariance assembly.
8. **Empirical init noise discarded** in favor of configured `cov_*_scale` (`:371-372`) — decide deliberately in the rebuild whether to trust measured or configured noise.
