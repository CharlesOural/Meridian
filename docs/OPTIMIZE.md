# Tuning ledger

Every constant that trades **compute vs accuracy/robustness** in the front-end, so
it can be retuned on the Jetson without re-deriving why it has its value. The
convention (also in CLAUDE.md): when you change a tuning number — config key or
hardcoded constant — add/update its one-line row here with **value | what it
trades | measured effect**. A constant changed without an entry is a future
mystery.

State of the ledger: the LIO front-end replaced the CT estimator on this branch,
so **every CT-era row is gone** (they described deleted machinery; git history on
the pre-LIO front-end branch keeps them). The rows below are the LIO ledger. Where the
effect column says **initial value, untuned (pre-Jetson)** the number is the
shipped default with no measured A/B behind it yet — the first Newer College
eval pass and then the Jetson port will fill these in.

Budget: one sweep must finish in **~100 ms** (10 Hz). Validate only on **full
sequences** (a short clip never fills Q_meas, so it hides drops), with the
deterministic replay harness (`replay_runner`) for accuracy numbers — the LIO
front-end is bit-identical run-to-run, so a single replay per point is valid.

## `frontend.lio.*` (config keys; spec 04 §14 owns the semantics)

| key | value | trades | measured effect |
|---|---|---|---|
| `voxel_size_m` | 1.0 | ↓ = finer map → denser correspondences, better detail, more voxels/memory and more NN probes; also scales the keypoint voxel and the observability neighbourhood | initial value, untuned (pre-Jetson) |
| `max_points_per_voxel` | 20 | ↑ = denser in-cell sampling → better NN quality, linear in-cell scan cost; also tightens the min-spacing rule `v/sqrt(cap)` | initial value, untuned (pre-Jetson) |
| `max_range_m` | 50 (newer-college yamls) / 100 (default) | ↑ = longer-range returns kept + larger resident map → more constraint at range, more memory/clip cost | set to match L1 `det_range` (OS0 usable range); no perf measurement yet |
| `keypoint_voxel_factor` | 1.5 | ↓ = more keypoints per sweep → more correspondences (accuracy) at linear assoc+GN cost | initial value, untuned (pre-Jetson) |
| `min_keypoints` | 30 | ↑ = stricter sweep-usability gate → fewer garbage solves, more skipped sweeps in sparse scenes | initial value, untuned (pre-Jetson) |
| `icp_max_iterations` | 100 | ↓ = bounded worst-case solve latency, risk of non-convergence counting as failure | initial value, untuned (pre-Jetson); typical converged iters ≪ cap |
| `convergence_eps` | 1e-5 | ↑ = earlier GN exit → cheaper solve, coarser pose | initial value, untuned (pre-Jetson) |
| `max_corr_dist_m` | 0.5 | ↑ = wider association gate → survives worse predictions, admits worse matches | initial value, untuned (pre-Jetson) |
| `min_beta` | 200.0 | ↓ = stronger gravity/attitude anchor at rest → roll-pitch stability vs fighting real slow tilt | initial value, untuned (pre-Jetson) |
| `max_expected_jerk` | 3.0 m/s³ | ↑ = accel-magnitude filter tracks faster excitation changes, noisier diagnostic | initial value, untuned (pre-Jetson); diagnostics-only consumer |
| `init_stationary_s` | 1.0 | ↑ = tighter gravity/bias init (frozen biases can't absorb error later) vs longer mandatory standstill | initial value, untuned (pre-Jetson); 0 disables static init — bench only |
| `max_gap_s` | 0.5 | ↓ = reseeds armed on smaller holes → safer but more chain breaks (AbsolutePrior packets) | initial value, untuned (pre-Jetson) |
| `reseed_cov_inflation` | 100.0 | ↓ = post-reseed anchor trusted more → backend snaps to it, wrong-landing risk | initial value, untuned (pre-Jetson) |
| `estimate_gyro_bias` | true (newer-college yamls) | online gyro-bias observer: each scan folds the IMU-vs-ICP attitude discrepancy into b_g, so the rest-fixed bias tracks slow drift on long runs | **park (571 s) 0.64→0.52 m (−19 %, max 1.46→0.95), quad-hard 0.123→0.118, quad-easy/math neutral** — replay-validated on NCD; off = the pre-existing static-init bias |
| `gyro_bias_gain` | 0.05 | per-scan filter gain on the observed rate error. ↑ tracks bias faster but absorbs ICP rotation noise; swept on NCD: 0.05 beats 0.1 (park 0.52 vs 0.58, and 0.1 regresses quad-hard back to 0.124) | **swept; 0.05 optimal of {0.05, 0.1}** |
| `gyro_bias_max` | 0.1 rad/s | per-axis clamp on \|b_g\| — runaway guard against a bad interval; well above any real MEMS gyro bias | safety clamp, not hit in NCD runs |

## `frontend.keyframe.*` (cadence — packet rate × per-packet info trade)

| key | value | trades | measured effect |
|---|---|---|---|
| `dist_m` | 1.0 | ↓ = denser keyframes → more backend factors/cost, shorter (smaller-Σ) relative constraints | initial value, untuned (pre-Jetson) |
| `rot_deg` | 10.0 | same trade on the rotation axis | initial value, untuned (pre-Jetson) |
| `time_s` | 1.0 | ceiling on packet latency when stationary/slow | initial value, untuned (pre-Jetson) |

`frontend.debug_path_sample_hz` (default 30) is deliberately **not** in this
ledger: it gates no estimator computation (debug stream sampling only).

## Back-end constants

| key | value | role |
|---|---|---|
| `backend.anchor_sigma` | 0.1 (λ=100) | gauge-anchor damping: every pose marginal floors at σ² (↓σ tightens loop-gate marginals) vs rigid corrections propagate at ~chain_info/(chain_info+λ) per solver iteration (↑σ floats faster). Measured (10-edge chain + conflicting absolute prior): λ=100 absorbs >99.9% of a forced rigid shift; λ=1e4 stalls at 38% after 5000 iters even in batch GN; λ=1e8 ≈ hard prior (0.02%) and the conflict permanently distorts relative geometry instead. Floor at default = (10 cm)². |

## Hardcoded code constants (need recompile)

| const (file) | value | role / retune note |
|---|---|---|
| `kObsKappa` (lio_frontend.cpp) | 0.01 | knee of the per-axis observability score s = h/(h+κ) on the per-correspondence average of the normal-projected information; 0.01 puts any axis carrying ≥ ~1/10 of the normal energy above 0.9 and an unsupported axis (plane sliding / yaw about a lone normal) near 0 (synthetic box room > 0.9 all axes, single floor plane < 0.3 on tx/ty/rz — test-pinned) |
| `kObsMinNeighbors` (lio_frontend.cpp) | 5 | minimum map neighbours for a usable surface normal; fewer = more correspondences scored, noisier normals |
| `kAnchorPriorVar` (lio_frontend.cpp) | 1e-8 | per-axis variance of the run's first AbsolutePrior keyframe (the odom-origin anchor, σ = 1e-4 matching backend `anchor_sigma`); tighter buys nothing, looser lets the backend wander the origin |
| `kMaxHeldGroups` / `kMaxLiveImu` (lio_frontend.cpp) | 64 / 4096 | pre-init group hold and live-IMU buffer bounds (~6 s scans / ~20 s IMU); memory caps only — past them the oldest data drops and init/live re-anchors later |
| `kUninformativeAccelVariance` (imu_tracker.hpp) | 1e9 | variance reported for a <2-sample interval, large enough that β suppresses the gravity anchor to numerical irrelevance |
| H ridge (scan_registration.cpp) | 1e-6·I | regularizes the per-scan information inverse for Σ; pure numerical floor, not a tuning lever |

## Surviving non-front-end rows (keys still deployed; findings frontend-agnostic)

| knob | now | role |
|---|---|---|
| `preprocess.blind` (newer-college-*.yaml) | 1.0 | NC rig is carried: 0.5 admits operator body/leg returns; 1.0 cuts them. Untuned beyond that — revisit with GT if close structure matters |
| `preprocess.det_range` (newer-college-*.yaml) | 50.0 | OS0-128 usable range; 120 m only keeps empty bins on this sensor. Range cap, not a perf lever |
| `sensors.imu.cov_gyr` / `b_gyr_cov` (newer-college-*.yaml) | 4e-06 / 1.6e-11 | the dataset's Kalibr file ships a corrupted gyro density (duplicated accel value) AND a corrupted gyro random walk; the corrected values stand for **any** consumer of these fields. Meta-lesson: sanity-check every imported IMU-noise field (density and random walk) against the datasheet. (The headline ATE numbers behind this correction were measured on the retired CT front-end; the file-corruption finding is what carries over.) |
| `debug.telemetry_rate_hz` (newer-college-quad debug posture) | 50 (was 10) | scalar token-bucket rate on /meridian/telemetry; ↑ = more wrapper publish load (debug path only). At 10 the limit sits exactly on the 10 Hz sweep cadence and beats against per-key jitter — measured: per-key live capture 1–98% (chaotic aliasing) at 10 → every-sweep capture at 50; replay/FileSink path unaffected (always exact) |
| Sensor subscription QoS depths (odometry_node.cpp) | cloud 20, image 40, imu 400 | deeper absorbs bursty delivery while the executor drains serially; costs only burst-peak memory. Depth does NOT fix in-flight best-effort loss (next row) |
| LiDAR subscription reliability (`sensors.lidar.qos_reliable` + tools/replay_qos_overrides.yaml) | replay: reliable both sides; robot: match the driver | best-effort delivery of the ~8 MB scans silently loses ~14% under host load (uniform, invisible to app counters); reliable = retransmission, zero loss. Reliability is per-pairing — match the publisher. Wrapper logs `TRANSPORT LOSS` (standing gate) if upstream loss reappears |

## Carried-over learnings (measured once — still binding)

- **Input integrity dominates accuracy comparisons.** Never read an ATE without
  the transport-loss / drop counters next to it; a config that re-introduces loss
  via compute pressure can fake an accuracy win. Run the intake waterfall
  (REALTIME_DEBUGGING.md) on every A/B.
- **QoS reliability must match the publisher per deployment.** Best-effort
  multi-MB messages lose a double-digit fraction under load with no
  retransmission and no application-visible counter; queue depth only absorbs
  executor burstiness.
- **Validate on full sequences only.** Short clips hide queue-pressure drops and
  slow failure modes (a bias-driven divergence at t≈83 s was invisible in every
  90 s clip).

## Back-end & loop closure constants

| const (file) | value | role / retune note |
|---|---|---|
| `kMaxExpansions` (backend/max_clique.cpp) | 200000 | PCM exact max-clique work budget; above it the bounded Bron–Kerbosch abandons the exact search and falls back to the greedy clique (FM-4). Deterministic (expansion count, not wall-clock). Raise only if a real loop set needs a larger exact search. |

## L5 loop closure (place.*) — tuned on NCD quad-easy

| key | value | trade/effect |
|---|---|---|
| `place.enable` (newer-college-quad.yaml) | true | master switch; ON by default now that the discrete LIO front-end carries no async map rebuild, so sync replay is byte-identical with the detector enabled (verified: two loop-closure replays md5-identical). On quad-easy it auto-detects 9 revisit loops; ATE 0.088 m with or without (low-drift, ~neutral — the loop path runs but the LIO odometry barely drifts here). |
| `place.sc_dist_thresh` (Scan Context accept) | 0.13 | retrieval precision. 0.5 (loose) admits co-visible-but-not-co-located pairs → GICP aligns them on the rigid scene → 45 loops → ATE 6.6 m (corrupted). 0.13 → 7 high-confidence loops, ATE stable. SC is high-recall/low-precision; this is the main precision gate alongside GICP fitness. |
| `place.gicp_fitness_min` / `gicp_overlap_min` | 0.8 / 0.6 | accept gate on the GICP inlier ratio (fitness == overlap). Together with sc_dist_thresh keeps only excellent geometric matches; lowering them re-admits the corrupting loops. |
| GICP fitness = inlier ratio (gicp_verify.cpp) | overlap = num_inliers / **own-downsampled** source size | the inlier-ratio denominator MUST be the cloud actually registered. Dividing by the full keyframe cloud (small_gicp downsamples internally) made overlap ~100× too small → fitness capped at 0.14 → zero loops admitted. Own the downsample → correct ratio. The spec's rmse-weighted fitness term is deferred (small_gicp reports a Mahalanobis cost, not metric rmse). |
| `place.cov_lambda` (loop covariance regulariser) | 0.1 | loop-factor softness. quad-easy barely drifts, so a stiff loop (tight cov from fitness≈1) adds cm-level noise to an already-accurate chain; a large λ keeps loops gentle here while a drifting sequence still accumulates correction through the chain. |
| `kMapCloudFoldPeriod` (meridian_pipeline.cpp) | 5 | assembled `map/cloud` rebuild cadence: re-composes all keyframe clouds at corrected poses every N folds (forced on a loop fold so the de-warp shows instantly). Rebuild is O(total points); lower = fresher map, more cost. 5 (~5 s refresh) balances live-viz freshness against the rebuild cost; it runs off the back-end driver so it never blocks the estimator. Viz-only — the assembled cloud feeds no estimator decision. |
| `kMapCloudVoxel` (meridian_pipeline.cpp) | 0.3 m | voxel pitch the assembled `map/cloud` is collapsed to (first return per cell wins). Bounds the published cloud size as the trajectory grows; smaller = denser map, larger payload. Viz-only. |

## L1 camera preprocessing (preprocess.camera.*)

| key | value | trade/effect |
|---|---|---|
| `preprocess.camera.rectify_balance` (newer-college-quad.yaml) | 0.0 | fisheye→pinhole undistort framing. 0 crops the rectified image to the all-valid region (no black border, drops peripheral field of view) → cleanest input for direct photometric matching; 1 keeps the full field of view (black corners, heavier edge stretch). Feeds OpenCV's new-camera-matrix estimate (`balance` for equidistant, `alpha` for radtan), which sets the rectified focal length. |
