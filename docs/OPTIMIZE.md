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
- `spline.n_cp_max` **2 and 3**: DIVERGES (ATE **1173 m** / **413 m**, visual map
  collapses). Diagnosis (adversarially verified): the analytic Jacobians ARE
  n_cp-correct (parity ~3e-11); the failure is the **dense-knot warm start** — a wiggly
  dense-knot seed corrupts data association on real data. **DEFERRED ACCURACY LEVER
  (task #62)**: adaptive knot density is mathematically validated and the principal
  remaining front-end lever for fast-motion accuracy, but needs a warm-start redesign
  (smooth dense-segment seeding + selectKnotDensity hysteresis), not Jacobian work.
  Do not flip n_cp_max > 1 before that lands.
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
| `backend.anchor_sigma` | 0.1 (λ=100) | gauge-anchor damping: every pose marginal floors at σ² (↓σ tightens loop-gate marginals) vs rigid corrections propagate at ~chain_info/(chain_info+λ) per solver iteration (↑σ floats faster). Measured (10-edge chain + conflicting absolute prior): λ=100 absorbs >99.9% of a forced rigid shift; λ=1e4 stalls at 38% after 5000 iters even in batch GN; λ=1e8 ≈ hard prior (0.02%) and the conflict permanently distorts relative geometry instead. Floor at default = (10 cm)². |
| `spline.n_cp_max` | 1 | adaptive knot density; >1 DIVERGES until the warm-start redesign lands (task #62) — see tested-and-rejected above |
| `marg_prior_scale` | **0.5** (all deployed configs, `newer-college-*.yaml`; code default 1.0) | multiplier in (0,1] on the marginalization-prior information at build time (sqrt(scale) on J0/r0: GN step direction preserved, confidence deflated = exponential forgetting per slide; 1.0 skips the multiply, bit-identical to pre-knob builds). **THE complete-stream fix.** Attribution matrix (complete 10 Hz, 0 bridges): 1.0 = 0.894 m rmse, prior RMS growing 3.7→7.4, accel bias locked ~0.16 m/s² (unbroken prior chain bakes in early bias/tilt error); **0.5 = 0.083 rmse / 0.064 mean**, prior RMS flat ~1.3-2.5, accel bias relaxes to 0.01-0.09; 0.1 = 0.087 (plateau). Cross-checks: `max_lidar_factors` 1500 = nil (0.951); LIO-only = partial (0.498); **`voxel_map_m` 0.5's apparent win (0.089) is CONFOUNDED — its load re-thinned the stream (313 bridges); do not re-tune voxel from that run.** Newer College full-bag A/B (quad-easy, LIO-only, corrected IMU noise): 0.5 = consistent {0.193, 0.159} rmse; 1.0 = **bimodal lottery {0.078, 1.172}** across identical runs — the overconfident chain makes accuracy depend on nondeterministic live-replay timing. Keep 0.5: forgetting buys run-to-run consistency, the property a benchmark baseline needs. Short-clip A/Bs of this knob are invalid (effects are cumulative; a 90 s A/B inverted the verdict twice). |
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
| `kWeakAxisCos` (ct_frontend.cpp) | 0.966 (~15°) | normal-vs-weak-axis alignment for factor-cap exemption |
| `kMinSupportWeight` (ct_frontend.cpp) | 0.02 | tail-knot pinning threshold (u³/6 basis support); below → pin the knot |
| `kTailSigmaVel` / `kTailSigmaRate` (ct_frontend.cpp) | 0.2 m/s / 0.1 rad/s | tail-anchor strengths over the unmeasured span |
| `kMaxInsertRangeM` (ct_frontend.cpp) | 200 | map-insert sanity bound [m] from anchor |
| `kHuberWhitened` (residuals_lidar/visual.cpp) | 1.0 | robust-kernel knee on the whitened (unit-variance) residual |
| `VisualMapConfig` defaults (visual_map.hpp) | ref_obs_cap 30, ref_add_angle 10°, ref_converged_obs 8 / 30°, depth_continuity_m 0.5 | visual-point reference-patch lifecycle; not yet config-exposed (`grid_cell_px` 32 now is, via `visual.grid_cell_px`) |
| `kMaxExpansions` (backend/max_clique.cpp) | 200000 | PCM exact max-clique work budget; above it the bounded Bron–Kerbosch abandons the exact search and falls back to the greedy clique (FM-4). Deterministic (expansion count, not wall-clock). Raise only if a real loop set needs a larger exact search. |
| KeyframeFinalizer queue capacity (ct_frontend.cpp `set_keyframe_sink`) | 4 | async keyframe-cov worker depth (live path): the per-keyframe pose marginal (was 30-80 ms, p90 ~103 ms — over budget) runs OFF T2; T2 pays only the snapshot copy (`frontend.ct.covsnapshot` ~0.03 ms). constraint_cov is ready by the next keyframe (~1 s slack); GNSS gate reads it stale-tolerant (1-keyframe lag). Deeper queue costs RAM, no latency benefit; a full queue blocks T2 one finalize (never worse than the old synchronous cost). Replay path: posecov stays inline on T2. |
| KeyframeFinalizer worker nice (keyframe_finalizer.cpp `worker_loop`) | 19 | worker has ~1 s slack vs T2's 100 ms deadline → lowest CFS weight: full speed on idle cores, yields contended cycles. Hot-box A/B (`MERIDIAN_SYNC_KEYFRAME_COV`): sync-inline 577 drops / **25 restarts** / 3.8 m ATE; async 464-513 drops / 0-1 restarts / 0.14-1.3 m. `frontend.ct.covsubmit` ~0.005 ms (push never blocks). |
| Sensor subscription QoS depths (odometry_node.cpp `create_subscriptions`) | cloud 20, image 40, imu 400 | deeper absorbs bursty delivery while the executor drains serially; costs only burst-peak memory. cloud 5→20: NO effect on the 14% scan loss (that was in-flight best-effort loss, see next row); kept as cheap burst insurance. |
| LiDAR subscription reliability (`sensors.lidar.qos_reliable` + tools/replay_qos_overrides.yaml) | replay: reliable both sides; robot: match the driver | reliable = retransmission, zero loss, slight backpressure under overload; best-effort = ~14% measured in-flight loss of the 8 MB scans under full-pipeline host load (uniform, immune to kernel buffers/depths). The actual replay-loss fix: 2671 published → 2671 received reliable (a reliable probe got 100% in the same run a best-effort reader lost 465). Wrapper logs `TRANSPORT LOSS` (gate) if upstream loss reappears. |

## L5 loop closure (place.*) — tuned on NCD quad-easy

| key | value | trade/effect |
|---|---|---|
| `place.enable` (newer-college-quad.yaml) | false | master switch; OFF by default because enabling the detector adds caller-thread work that shifts the vendored ikd-Tree async-rebuild timing and breaks bit-exact replay at the float floor late in the run (pre-existing front-end property). On quad-easy it auto-detects ~7 revisit loops (1 self-test-rejected); ATE 0.124→0.132 m (low-drift, ~neutral). |
| `place.sc_dist_thresh` (Scan Context accept) | 0.13 | retrieval precision. 0.5 (loose) admits co-visible-but-not-co-located pairs → GICP aligns them on the rigid scene → 45 loops → ATE 6.6 m (corrupted). 0.13 → 7 high-confidence loops, ATE stable. SC is high-recall/low-precision; this is the main precision gate alongside GICP fitness. |
| `place.gicp_fitness_min` / `gicp_overlap_min` | 0.8 / 0.6 | accept gate on the GICP inlier ratio (fitness == overlap). Together with sc_dist_thresh keeps only excellent geometric matches; lowering them re-admits the corrupting loops. |
| GICP fitness = inlier ratio (gicp_verify.cpp) | overlap = num_inliers / **own-downsampled** source size | the inlier-ratio denominator MUST be the cloud actually registered. Dividing by the full keyframe cloud (small_gicp downsamples internally) made overlap ~100× too small → fitness capped at 0.14 → zero loops admitted. Own the downsample → correct ratio. The spec's rmse-weighted fitness term is deferred (small_gicp reports a Mahalanobis cost, not metric rmse). |
| `place.cov_lambda` (loop covariance regulariser) | 0.1 | loop-factor softness. quad-easy barely drifts, so a stiff loop (tight cov from fitness≈1) adds cm-level noise to an already-accurate chain; a large λ keeps loops gentle here while a drifting sequence still accumulates correction through the chain. |
| `kMapCloudFoldPeriod` (meridian_pipeline.cpp) | 10 | assembled `map/cloud` rebuild cadence: re-composes all keyframe clouds at corrected poses every N folds (forced on a loop fold so the de-warp shows instantly). Rebuild is O(total points); lower = fresher map, more cost. Viz-only — the assembled cloud feeds no estimator decision. |
| `kMapCloudVoxel` (meridian_pipeline.cpp) | 0.3 m | voxel pitch the assembled `map/cloud` is collapsed to (first return per cell wins). Bounds the published cloud size as the trajectory grows; smaller = denser map, larger payload. Viz-only. |
