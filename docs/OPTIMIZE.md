# Tuning ledger

Every constant that trades **compute vs accuracy/robustness** in the front-end, so it
can be retuned on the Jetson without re-deriving why it has its value. When you change
a tuning number (config or hardcoded), add/update a one-line row here: value, what it
trades, measured effect. Baselines are dev-PC, canteen_day full sequence, 10 Hz.

> **⚠ STREAM-COMPLETENESS CAVEAT (2026-06-10).** Every benchmark before this date ran
> on an input stream silently thinned ~14% by best-effort transport loss (see
> REALTIME_DEBUGGING.md "intake audit") — effectively ~8.6 Hz with random IMU-bridged
> single-scan holes. The thinning acted as a regularizer the tuning absorbed: after the
> QoS fix the true complete 10 Hz stream degraded full-mission ATE from ~0.15-0.22 m to
> ~1.2-1.3 m. **Accuracy rows measured pre-fix carry that caveat; compute/timing rows
> remain valid** (compute is stream-rate honest).
>
> **RESOLVED (same day):** attribution matrix traced the degradation to
> **marginalization-prior overconfidence** (see `marg_prior_scale` row) — the holes had
> been breaking the prior chain, masking it. With `marg_prior_scale: 0.5` the complete
> stream achieves **0.083 m rmse / 0.064 m mean** — better than every thinned-era number
> — at full real-time, 0 bridges, 0 restarts. Baseline re-established on truth.

Budget: one sweep must finish in **~100 ms**. Validate only on a **full sequence**
(a short clip never fills the Q_meas buffer so it hides drops); the rows below were
measured on the 267 s canteen_day run unless a row says otherwise.
The LIVO path now runs **≈85 ms/sweep, ~7 dropped sweeps, 0 restarts, GT ATE ~0.10 m**
— effectively real-time on the dev PC (session journey: 462 drops / 114 ms → 7 / 85 ms,
via analytic Jacobians → Ceres-solve threading → **parallel association**, the last
being decisive: drops 205→7). The Jetson is ~3-4× slower per core; these levers buy the
headroom to repeat this there. With drops ~gone, the residual ~0.10 m ATE is almost pure
open-loop drift — closing it to SOTA (~0.05 m) is the **L3 backend (loop closure)**,
not more front-end tuning.

**Tested-and-rejected (do not re-try without addressing the cause):**
- `solver.time_limit_ms` 90→**60**: REJECTED. Starves convergence — `deadline_hit`
  ~0.05→~1.0, ATE 0.10→**0.56 m**. After parallel association the solve finishes well
  inside 90 ms, so lowering the cap buys nothing. Keep 90.
- `spline.n_cp_max` **2 and 3** (legacy peak-geared path): DIVERGED (ATE **1173 m** /
  **413 m**, visual map collapses). Diagnosis (adversarially verified): the analytic
  Jacobians ARE n_cp-correct (parity ~3e-11); the failure was the **dense-knot warm
  start** — a wiggly dense-knot seed corrupts data association on real data, with
  per-round re-association already present. RESOLVED by #72: the redesign shipped as
  `frontend.ncp.*` (true non-uniform basis, mean-of-norms gearing, tail re-knot, and
  the `warmstart_iters` IMU-only mini-solve that irons fresh dense tails before
  association). The legacy peak-geared path is deleted; do not re-grow it. `ncp.enabled`
  stays false until the validation campaign passes.
- **LiDAR densification on Newer College quad-easy**: REJECTED across the board.
  voxel 0.5 = neutral (0.197 vs 0.16-0.19 baseline, +11 ms); voxel 0.5 + factors 1500 and
  + point_filter 1 = budget blown (104-138 ms/sweep, 262-468 Q_meas drops, divergence);
  voxel 0.4 + factors 2000 = clean delivery (0 bridges, 88 ms) yet **worse accuracy
  (1.165 m)** — finer voxels give noisier plane normals. The LiDAR term is not the
  binding error source on this sequence; the levers are prior/bias consistency and the
  L3 backend.


## System behavior learnings (meta — how this estimator responds to its knobs)

Dataset-agnostic behavior established by controlled experiments; the *why* behind the
rows below.

- **Prior forgetting is a robustness knob, not a cliff.** A fixed-lag CT window with
  per-sweep Schur marginalization accumulates prior confidence faster than its
  linearization stays valid. Symptoms: prior residual RMS grows monotonically over a
  mission, and a weakly-observable state (accel bias) pins at a wrong value. Deflating
  the carried information per slide (`marg_prior_scale`) fixes it; the system plateaus
  across 0.5–0.1 — tune coarsely, it is not fragile.
- **Input integrity dominates accuracy comparisons — and the estimator hides loss.**
  IMU gap-bridging is so effective that double-digit input loss barely moves the ATE
  *median*; it shows up as restart lotteries and rmse tails. Never read an ATE without
  the bridge/`lost_upstream` counters next to it.
- **Complete data exposes miscalibration that thinned data masks.** Random input holes
  act as regularization (they break the prior chain). Fixing a data-loss bug can
  *worsen* ATE until re-tuned on the truth — and a config that re-introduces loss via
  compute pressure (Q_meas evictions) can *fake* an accuracy win. Cross-check bridges
  on every A/B.
- **QoS reliability must match the publisher per deployment.** Best-effort delivery of
  multi-MB messages silently loses a double-digit fraction under host load (no
  retransmission, invisible to application counters). Queue *depth* does not protect
  against this — depth only absorbs executor burstiness.
- **Output-only work belongs off the deadline thread, at low priority.** Anything with
  next-keyframe (~1 s) slack rather than next-sweep (100 ms) slack — covariance
  recovery, future descriptors — runs on a deprioritized worker fed by a snapshot;
  the deadline thread pays only the value-copy.
- **A deadline-bound solver converts Jacobian savings into iterations, not wall-time.**
  Cheaper derivatives improve accuracy at fixed budget; lowering the budget to cash
  them as speed starves convergence unless `deadline_hit` is already ~0.

## Overnight accuracy campaign (2026-06-12, deterministic replay, full sequences)

Per-sequence optima found by coordinate descent on the deterministic harness
(bit-exact; one run per point). Headline: quad-easy 0.16→**0.073 m**; math-medium
first-ever working runs 291→**~1.0 m**; park unblocked (moving-start init).

- **Retention re-tuned offline**: `marg_prior_scale` sweep on quad is monotonic to 1.0
  (0.5=0.219 → 1.0=0.086) — the 0.5 default was medicine for LIVE-timing nondeterminism,
  not the estimator. SEQUENCE-DEPENDENT: math-medium prefers 0.5 (fast-motion
  linearization error); keep per-sequence values, pursue per-state forgetting later.
- **`max_outer_iters` 2→4**: quad 0.086→0.075; saturates at 4 (6/reassoc3/inner8 all ≤1 mm).
- **`huber_clamp_max` 4→1**: quad 0.075→0.073 BUT math diverges (763 m) — the tight knee
  starves the solve during fast turns. Default stays 4.0; 1.0 only on gentle sequences.
- **`gravity_refine` (free gravity block): REJECTED — measurement CONFOUNDED, flag now
  hard-blocked by `validate()`** — quad 0.073→1.70 m, but the run went through a broken
  prior path: the marginalization prior keeps the freed gravity block with no
  sphere-manifold tangent lift (Euclidean fallback linearizes the wrong tangent) AND the
  keyframe worker's prior clone aliases live gravity storage across threads. 1.70 m
  indicts that path, not per-window gravity observability. Config validation rejects the
  flag outright until the prior cost grows the sphere lift and the capture remap covers
  gravity; re-measure only after both. The init-tilt mechanism wants a SOFT prior (future).
- **THE ROTATION-RATE WALL**: math-medium's 291 m failure = a 2.6 Hz yaw SWAY (not a
  turn) at t≈10-14 s exceeding the 100 ms-knot spline bandwidth (deskew/sweep_trans_m
  exploded to 3.9 m/sweep on clean IMU data; bias pegged its box corner; map poisoned
  permanently — n_matched halved). `knot_dt_ms` 100→50 (+window_knots 8→16) was the
  stopgap (291→5.8→1.0 m with scale 0.5) at ~2x solve cost on fast sequences;
  **STOPGAP RETIRED** — the YAMLs are back at 100/8 and the fix is excitation-geared
  knot density (`frontend.ncp`, #72), which densifies only the excited segments. 33 ms
  global knots still diverge (dense seeding everywhere); do not re-try global densification.
- **Moving-start init (`init_force_after_s`)**: structural parity gap vs FAST-LIO2 —
  our bootstrap demanded a static window forever and silently discarded ALL of park
  (5,711 groups, zero telemetry). The fallback (sweep-mean init after N s) unblocks
  park; pair with the static gate, never free gravity to compensate.
- **IMU noise on quad: SATURATED at current values** (everything ±1 mm); datasheet-tight
  accel (cov 2.25e-06) DIVERGES (28 km) — over-trusting the IMU is fatal in both
  directions (cf. the legged ×40 row). The corrected NC values stand.

## Real-time budget levers (tuned this session)

| knob (config `meridian.frontend.*`) | now | trades | effect (measured) |
|---|---|---|---|
| `lidar.voxel_map_m` | 0.6 | ↑ = fewer downsampled scan points → cheaper assoc+map+visual_map; coarser map | 0.5→0.6: 77→67 ms, GT 0.042→0.054 m |
| `lidar.max_lidar_factors` | 1000 | ↑ = more LiDAR residuals → bigger solve; cap applied *after* assoc | 1500→1000: solve −6 ms, GT unchanged |
| (structural) visible-candidate cache | on | select visible visual points once/sweep vs per outer round | 117→84 ms, no accuracy cost |
| (structural) analytic Jacobians (LiDAR, visual, IMU) | on | analytic vs Ceres autodiff; whole factor family analytic (autodiff kept behind `*Autodiff` factories as parity twins) | LiDAR: faster solve. IMU: deadline-bound solve cashes cheaper Jacobians as *iterations* → GT ATE 0.123→0.100 m, solve_ms unchanged |
| (structural) keyframe-gated pose covariance | on | recover the Ceres pose marginal only on keyframe sweeps (~12%); GNSS gate tolerates the stale value | solve_ms 114→105, drops 462→321, ATE unchanged (0.123 m) |
| (structural) gap-bridge horizon `max_bridge` | 5 sweeps | bridge a dropped-sweep hole by IMU extrapolation + solve vs restart-and-coast; too high and a real long hole extrapolates garbage knots | LIO 8 m→0.27 m full sequence, restarts→0 |
| (structural) Ceres window-solve `num_threads` (ct_frontend.cpp) | live: `min(8, cores)`; replay: 1 | parallel per-residual evaluation; deterministic/replay path stays 1 for bit-exact replay==live | solve reaches tolerance before deadline (`deadline_hit` ≈1.0→~0.7) → drops 306→205, solve_ms 105→~98, ATE within run variance (~0.10–0.13 m). 120/120 tests green |
| (structural) **parallel association** (OpenMP, residuals_lidar.cpp `associate()`) | live: `min(8, cores)` | `schedule(static)` over points + per-thread buffers merged in thread order → BIT-IDENTICAL to serial (parity-tested), no determinism cost; ikd-Tree guarded by `shared_mutex` (shared on fitPlane, exclusive on insert/trim) | **the headline win:** assoc ~21→~7 ms/pass (~28 ms/sweep saved); solve_ms ~98→~85, deadline_hit 0.7→~0.05, drops 205→7, 0 restarts, ATE held (mean 0.098 m) |
| (structural) visual NCC-matrix cache (visual_map.cpp) | on | recompute per-point pairwise NCC only on viewpoint change (`obs_dirty`) vs every sweep | NEUTRAL: visual_map stayed ~11 ms — NCC re-score was not the dominant cost (projection/eviction is). No harm; behaviour-equivalent (digest test green) |
| (structural) frustum-bounded visible scan (`VisualMap::selectVisibleIds` via `frustumCandidateIds`) | on | walk only voxels intersecting the view frustum (spatial index + 5 world planes, side rays widened ×1.25 for distortion) instead of projecting every map point; per-point test unchanged so the candidate set is identical | `vmap.select` **9.06→3.43 ms** at ~55k map points / ~480 candidates (scales with frustum, not map size); visual_map total 13.5→12.6 ms. The ×1.25 margin is pinned by gtest (`VisualMapFrustum.*`): byte-identical to a full-map scan for pinhole + RadTan up to k1≈1.2; ×1.0 breaks at k1=0.4 pincushion, ×1.25 holds to k1≈3.0. Sub-timers `frontend.ct.vmap.{select,update,promote,evict}` are permanent telemetry |
| `debug.telemetry_rate_hz` (newer-college-quad debug posture) | 50 (was 10) | scalar token-bucket rate on /meridian/telemetry; ↑ = more wrapper publish load (debug path only, estimator untouched). At 10 the limit sits exactly on the 10 Hz sweep cadence and beats against per-key jitter | measured: per-key live capture 1–98% (chaotic aliasing) at 10 → every-sweep capture at 50; replay/FileSink path unaffected (always exact) |

## Other compute levers (available, not yet pulled)

| knob | now | trades |
|---|---|---|
| `visual.enable` | **false** everywhere deployed (newer-college: equidistant cams + vignetting, see DATASET.md; legged_underground: dark-corridor divergence) | false → drop the whole photometric stage (~20 ms), clean LIO ~7.5 cm. Biggest cut. On the legged bag the stage is also an accuracy hazard: in the dark corridor it diverges the estimate (full-bag replay 340–2834 m vs 4.5 m off; the well-lit first 120 s are fine at 1.3 m). Low-light gating needed before re-enabling there. |
| camera (`sensors.camera`) | FLIR 1024×768 | DAVIS `event_cam00` 346×260 ≈ 8× fewer pixels in the visual stage |
| `lidar.num_match_points` | 5 | ↓ = fewer NN neighbours per assoc query |
| (settled) iVox map backend — REMOVED | n/a | A/B benchmarked then removed (don't re-attempt). Exact stencil diverged (1331 voxels at radius 2.236 m / voxel 0.6 m); approx stencil=1 (27 vox) *tied* ikd-Tree (~22 vs ~21 ms, ATE 0.108 vs 0.103). iVox only wins at radius≈voxel (Faster-LIO regime), not our ratio 3.7. The map structure was never the lever — parallel association was. |
| `solver.time_limit_ms` | 90 | hard per-sweep solve cap (safety, not tuning); ↓ bounds latency but shallows the solve |
| `max_outer_iters` / `reassoc_steps` | 2 / 2 (newer-college yamls; code default 4 / 2); **6 / 2 + `solver_max_iterations` 12 (legged_underground)** | ↓ = fewer re-association passes → less assoc+solve, less refinement. On the legged bag the 5/4 caps under-converge the corridor walk and diverge at t≈188 s even with the deadline off (replay ATE 340 m vs 4.5 m at 12/6); live, `time_limit_ms` gates the schedule either way. |
| ceres `num_threads` — marginalization (marginalization.cpp) | 1 | the window solve is threaded (above); the marginalization solve could follow the same determinism-gated pattern if it shows up in profiling |

## Accuracy / robustness constants (change only with GT in hand)

| knob | now | role |
|---|---|---|
| `sensors.imu.cov_*` (legged_underground) | ≈×40 the calib Allan densities | IMU-vs-LiDAR trust. Static-calib Allan floors over-trust the IMU under locomotion vibration: calib values diverge (448 m ATE / 120 s), ×10 still diverges (116 m), ×20 holds (0.5 m), ×40 holds (0.6 m; shipped values 0.2 m, full bag 4.5 m / 161 m). Set per platform from a GT replay sweep, not from the calib file. |
| `ncp.enabled` / `ncp.n_cp_max` | **false** / 3 (all YAMLs) | adaptive knot density (#72): per-outer-segment control-point count geared from segment IMU stats; off = uniform spline, bit-identical to the pre-ncp build (merge gate: quad-easy replay diff — PASSED, byte-identical at both the outer-4 and deployed parameterizations). Flip `enabled` for the validation campaign (knot_dt 100 / window 8, n_cp_max 3); a gear-g segment multiplies that segment's knot count (solve cost) by g only where excitation demands it. **Validation campaign #1 FAILED — keep false**: quad-easy 0.084→**206 m**, math-medium **4980 m** (vs 0.30 gate). Mechanism (telemetry + raw-bag recompute agree): the *accel* axis mis-gears plain walking — step-impact vibration folds into the norm so mean‖a‖ sits ~1 m/s² above g (quad p50 0.98, p90 1.78; math p50 1.75) and crosses both 0.5/1.0 edges ⇒ gear 3 on 53-77% of segments = the dense-knots-everywhere divergence class. The gearing *concept* works: gear 3 held through the math sway and moved the rotation-wall onset 10.3 s→~17 s (max err 0.49 m through the sway vs instant 240 m before); solve cost stayed flat (12.4 vs 11.7 ms/pass; warmstart ~1 ms). Fix is the accel statistic/edges, not the spline machinery — accel statistic since reworked to the vector mean (see `ncp.accel_thresh`); re-run the campaign before flipping |
| `ncp.omega_thresh` | [0.5, 1.0, 5.0] rad/s | mean-of-norms gyro band edges raising the gear to 2/3/4 (Coco-LIC's mean-rate ladder, retained: math-medium's mean rate straddles the 0.5 edge so the sway gears up while quad-easy stays at 1); RAW gyro, no bias correction |
| `ncp.accel_thresh` | [0.5, 1.0, 5.0] m/s² (edges unchanged; **statistic re-based post-campaign-#1**) | accel band edges, same ladder on the accel axis; the larger of the two axes wins. Statistic is now ‖vector-mean(a)‖ deviation from g — was \|mean‖a‖−g\|, **MEASURED BROKEN for handheld walking**: impact vibration inflates a mean of norms (Jensen: E‖g+n‖>g), p50 ~1.0-1.8 m/s² mid-walk on quad/math (raw 200 Hz bag windows, independent of estimator) ⇒ permanent gear 3 and the campaign-#1 divergence. The vector mean cancels zero-mean vibration (gtest-pinned: alternating ±n reads 0, sustained 2 m/s² reads 2.0) so unaccelerated walking should read ~0 — bag-level floor NOT yet re-measured; confirm in campaign #2 before trusting the edges. Trades: perpendicular-to-g sustained accel under-reads (~a²/2g) under either statistic; in-segment attitude change shrinks the mean by O(θ²)≈0.1 at 5 rad/s — both invisible at these edges, and rotation is the omega axis's job (omega measured exactly as designed: quad p50 0.24/p90 0.41 stays gear 1; math sway crosses to gear 3) |
| `ncp.hysteresis` | 0.15 (frac) | band-down hysteresis: a held band's drop edge is `edge*(1−h)`, so the gear cannot chatter at a threshold; trades a slightly sticky high gear for solve-cost stability |
| `ncp.warmstart_iters` | 4 | LM iterations of the IMU-only mini-solve smoothing freshly laid dense tails before association (1 thread, no deadline; timer `frontend.ct.warmstart`). 0 disables = the recorded 1173/413 m dense-seed divergence class; more buys little (the main solve re-fits) |
| `marg_prior_scale` | **legacy fallback** — superseded by `prior_forgetting.*` (next row), which ignores it while enabled; still 0.5 on `newer-college-park.yaml` (forgetting unvalidated there; code default 1.0) | multiplier in (0,1] on the marginalization-prior information at build time (sqrt(scale) on J0/r0: GN step direction preserved, confidence deflated = exponential forgetting per slide; 1.0 skips the multiply, bit-identical to pre-knob builds). **THE complete-stream fix.** Attribution matrix (complete 10 Hz, 0 bridges): 1.0 = 0.894 m rmse, prior RMS growing 3.7→7.4, accel bias locked ~0.16 m/s² (unbroken prior chain bakes in early bias/tilt error); **0.5 = 0.083 rmse / 0.064 mean**, prior RMS flat ~1.3-2.5, accel bias relaxes to 0.01-0.09; 0.1 = 0.087 (plateau). Cross-checks: `max_lidar_factors` 1500 = nil (0.951); LIO-only = partial (0.498); **`voxel_map_m` 0.5's apparent win (0.089) is CONFOUNDED — its load re-thinned the stream (313 bridges); do not re-tune voxel from that run.** Newer College full-bag A/B (quad-easy, LIO-only, corrected IMU noise): 0.5 = consistent {0.193, 0.159} rmse; 1.0 = **bimodal lottery {0.078, 1.172}** across identical runs — the overconfident chain makes accuracy depend on nondeterministic live-replay timing. Keep 0.5: forgetting buys run-to-run consistency, the property a benchmark baseline needs. Short-clip A/Bs of this knob are invalid (effects are cumulative; a 90 s A/B inverted the verdict twice). |
| `prior_forgetting.*` | **enabled** (`newer-college-quad/math.yaml`; park still legacy scalar — unvalidated there; code default off) | **Structured per-state replacement for `marg_prior_scale`** (which is ignored while enabled): each slide relaxes the carried prior's *covariance* per block class — bias blocks by the configured IMU random-walk variance over the slide (`use_imu_walk: true` ⇒ `Q = b_*_cov * dt`, NC math/quad: acc 1.85e-6, gyr 1.6e-12 per 100 ms), kept gravity by `q_gravity` 1e-10/slide, pose/velocity knots by `q_knot * dt` (default **0** — knot uncertainty belongs to the measurement model). Woodbury form in information space: null directions preserved, info capped at 1/q (EKF steady state), prior mean preserved; `enabled: false` is bit-identical legacy. The accumulated forgetting dt (spans since the last *successful* prior build, so slides that skipped a rebuild accumulate) is **clamped to 1.0 s** (`ct_frontend.cpp` slide path): trades outage-proportional forgetting (exact up to 1 s of skipped builds) against one-step prior erasure — unclamped, a long bridge/outage would inflate the bias blocks by the whole gap and wipe the carried prior in a single build. Measured (deterministic suite, knot-100, outer4): quad-easy **0.075 rmse** = scalar-best 1.0 (vs deployed 0.5 = 0.219); math-medium pre-wall (12 s) **12.5 m** vs 12.1 scalar-0.7 / 14.7 scalar-1.0 with deeper accel-bias relaxation; math full-bag **4691 m** vs 16666 deployed-0.5 but > the lucky scalar-0.7 261 — the t≈10.2 s rotation wall poisons the prior's knot blocks and the post-wall runaway magnitude is chaotic in every arm (scalar family spans 261→16666). One config now serves both collections where no single scalar did (0.5: 0.219/16666; 0.7: 0.125/261; 1.0: 0.075/4690-class). `q_knot` **1e-4 and 1e-3 tested-and-rejected** on math (4405 / 17837 — post-wall lottery, no systematic gain); the wall itself is knot-density (#72) + prior-feed outlier gating (#74) work, not forgetting. |
| `spline.window_knots` / `knot_dt_ms` | 8 / 100 | window length (knots) / one knot per 10 Hz sweep |
| `bias.gyr_max` / `acc_max` / `knot_dt_ms` | 0.5 / 5.0 / 500 | bias box bounds [rad/s, m/s²] + random-walk knot cadence [ms] |
| `init_time_s` | 1.0 | static-init window; longer tightens gravity/bias init (frozen accel bias can't absorb its error) |
| `lidar.plane_thresh` / `max_match_dist_sq` / `point_cov` | 0.1 / 5.0 / 1e-3 | plane-fit planarity gate / assoc gate radius² / LiDAR point variance |
| `lidar.normal_strata` / `min_factors_per_normal` | 7 / 50 | factor-cap stratification (weak-axis observability protection) |
| `visual.ncc_thre` / `img_point_cov` / `outlier_threshold` | 0.5 / 100 / 1000 | photometric correlation gate / residual weight / SSD reject |
| `visual.active_box_m` | 20 | visual-map memory bound (NOT per-sweep cost — that's the cache); set to camera re-observability range, not LiDAR scale |
| `preprocess.voxel_surf_m` / `point_filter_num` / `blind` / `det_range` | 0.5 / 2 / 0.5 / 120 | L1 scan downsample / decimation / min-max range [m] |
| `preprocess.blind` (newer-college-*.yaml) | 1.0 | NC rig is carried: 0.5 admits operator body/leg returns; 1.0 cuts them. Untuned beyond that — revisit with GT if close structure matters |
| `preprocess.det_range` (newer-college-*.yaml) | 50.0 | OS0-128 usable range; FP's 120 m only keeps empty bins on this sensor. Range cap, not a perf lever |
| `imu.cov_gyr` (newer-college-*.yaml) | 4e-06 (σ=2e-3 rad/s/√Hz) | the dataset's Kalibr file duplicates the accel density (0.019) into the gyro field → cov 3.6e-4 → gyro residual ~90× under-weighted → rotation floats. Measured: quad-easy 90 s ATE **5018 m → 0.066 m**. σ from OKVIS2's BMI085 (Hilti-2022) config; do not re-import the dataset value |
| `imu.b_gyr_cov` (newer-college-*.yaml + import_ncd default) | 1.6e-11 (σ=4e-6 rad/s²/√Hz; dataset Kalibr ships 7.08e-08) | the dataset's gyro RANDOM WALK is corrupted like its density (row above): the loose value lets the gyro bias absorb LiDAR misfit — bias pegs the 0.5 rad/s `bias.gyr_max` box at t≈83 s, heading diverges, phantom 3–4 m/s velocity. A 90 s clip never sees it (why the 0.066 m row missed it) — full-sequence validation only | quad-easy full bag **15.99 → 0.193 m rmse** (0.231 m with only this knob changed). Meta-lesson: sanity-check EVERY imported IMU-noise field (density AND random walk) against the datasheet; the symptom of a corrupted one is a bias state pegging its box while absorbing another sensor's misfit |

## Hardcoded code constants (need recompile)

| const (file) | value | role / retune note |
|---|---|---|
| `kObsKappa` (lio_frontend.cpp) | 0.01 | knee of the per-axis observability score s = h/(h+κ) on the per-correspondence average of the normal-projected information; 0.01 puts any axis carrying ≥ ~1/10 of the normal energy above 0.9 and an unsupported axis (plane sliding / yaw about a lone normal) near 0 (synthetic box room > 0.9 all axes, single floor plane < 0.3 on tx/ty/rz — test-pinned) |
| `kAnchorPriorVar` (lio_frontend.cpp) | 1e-8 | per-axis variance of the run's first AbsolutePrior keyframe (the odom-origin anchor, σ = 1e-4 matching backend `anchor_sigma`); tighter buys nothing, looser lets the backend wander the origin |
| `kMaxHeldGroups` / `kMaxLiveImu` (lio_frontend.cpp) | 64 / 4096 | pre-init group hold and live-IMU buffer bounds (~6 s scans / ~20 s IMU); memory caps only — past them the oldest data drops and init/live re-anchors later |
| `kWeakAxisCos` (ct_frontend.cpp) | 0.966 (~15°) | normal-vs-weak-axis alignment for factor-cap exemption |
| `kMinSupportWeight` (ct_frontend.cpp) | 0.02 | tail-knot pinning threshold (u³/6 basis support); below → pin the knot |
| `kTailSigmaVel` / `kTailSigmaRate` (ct_frontend.cpp) | 0.2 m/s / 0.1 rad/s | tail-anchor strengths over the unmeasured span |
| `kMaxInsertRangeM` (ct_frontend.cpp) | 200 | map-insert sanity bound [m] from anchor |
| `kHuberWhitened` (residuals_lidar/visual.cpp) | 1.0 | robust-kernel knee on the whitened (unit-variance) residual |
| `VisualMapConfig` defaults (visual_map.hpp) | ref_obs_cap 30, ref_add_angle 10°, ref_converged_obs 8 / 30°, depth_continuity_m 0.5 | visual-point reference-patch lifecycle; not yet config-exposed (`grid_cell_px` 32 now is, via `visual.grid_cell_px`) |
| KeyframeFinalizer queue capacity (ct_frontend.cpp `set_keyframe_sink`) | 4 | async keyframe-cov worker depth (live path): the per-keyframe pose marginal (was 30-80 ms, p90 ~103 ms — over budget) runs OFF T2; T2 pays only the snapshot copy (`frontend.ct.covsnapshot` ~0.03 ms). constraint_cov is ready by the next keyframe (~1 s slack); GNSS gate reads it stale-tolerant (1-keyframe lag). Deeper queue costs RAM, no latency benefit; a full queue blocks T2 one finalize (never worse than the old synchronous cost). Replay path: posecov stays inline on T2. |
| KeyframeFinalizer worker nice (keyframe_finalizer.cpp `worker_loop`) | 19 | worker has ~1 s slack vs T2's 100 ms deadline → lowest CFS weight: full speed on idle cores, yields contended cycles. Hot-box A/B (`MERIDIAN_SYNC_KEYFRAME_COV`): sync-inline 577 drops / **25 restarts** / 3.8 m ATE; async 464-513 drops / 0-1 restarts / 0.14-1.3 m. `frontend.ct.covsubmit` ~0.005 ms (push never blocks). |
| Sensor subscription QoS depths (odometry_node.cpp `create_subscriptions`) | cloud 20, image 40, imu 400 | deeper absorbs bursty delivery while the executor drains serially; costs only burst-peak memory. cloud 5→20: NO effect on the 14% scan loss (that was in-flight best-effort loss, see next row); kept as cheap burst insurance. |
| LiDAR subscription reliability (`sensors.lidar.qos_reliable` + tools/replay_qos_overrides.yaml) | replay: reliable both sides; robot: match the driver | reliable = retransmission, zero loss, slight backpressure under overload; best-effort = ~14% measured in-flight loss of the 8 MB scans under full-pipeline host load (uniform, immune to kernel buffers/depths). The actual replay-loss fix: 2671 published → 2671 received reliable (a reliable probe got 100% in the same run a best-effort reader lost 465). Wrapper logs `TRANSPORT LOSS` (gate) if upstream loss reappears. |
