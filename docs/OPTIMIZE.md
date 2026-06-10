# Tuning ledger

Every constant that trades **compute vs accuracy/robustness** in the front-end, so it
can be retuned on the Jetson without re-deriving why it has its value. When you change
a tuning number (config or hardcoded), add/update a one-line row here: value, what it
trades, measured effect. Baselines are dev-PC, canteen_day full sequence, 10 Hz.

Budget: one sweep must finish in **~100 ms**. Current critical path **≈ 67 ms**
(assoc 21 + visual 3 + solve 17 + map 8 + visual_map 17), 0 drops, GT ATE 0.054 m.
The Jetson is slower per core — these are the dials to recover headroom there.

## Real-time budget levers (tuned this session)

| knob (config `meridian.frontend.*`) | now | trades | effect (measured) |
|---|---|---|---|
| `lidar.voxel_map_m` | 0.6 | ↑ = fewer downsampled scan points → cheaper assoc+map+visual_map; coarser map | 0.5→0.6: 77→67 ms, GT 0.042→0.054 m |
| `lidar.max_lidar_factors` | 1000 | ↑ = more LiDAR residuals → bigger solve; cap applied *after* assoc (no assoc effect) | 1500→1000: solve −6 ms, GT unchanged |
| (structural) visible-candidate cache | on | select visible visual points once/sweep vs per outer round | 117→84 ms, no accuracy cost |
| (structural) analytic Jacobians (LiDAR, visual) | on | analytic vs Ceres autodiff in the solve | faster solve; revert commit `df9316d` → autodiff |

## Other compute levers (available, not yet pulled)

| knob | now | trades |
|---|---|---|
| `visual.enable` | true (garden_day); **false (legged_underground)** | false → drop the whole photometric stage (~20 ms), clean LIO ~7.5 cm. Biggest cut. On the legged bag the stage is also an accuracy hazard: in the dark corridor it diverges the estimate (full-bag replay 340–2834 m vs 4.5 m off; the well-lit first 120 s are fine at 1.3 m). Low-light gating needed before re-enabling there. |
| camera (`sensors.camera`) | FLIR 1024×768 | DAVIS `event_cam00` 346×260 ≈ 8× fewer pixels in the visual stage |
| `lidar.num_match_points` | 5 | ↓ = fewer NN neighbours per assoc query |
| `solver.time_limit_ms` | 90 | hard per-sweep solve cap (safety, not tuning); ↓ bounds latency but shallows the solve |
| `max_outer_iters` / `reassoc_steps` | 4 / 2 (garden_day); **6 / 2 + `solver_max_iterations` 12 (legged_underground)** | ↓ = fewer re-association passes → less assoc+solve, less refinement. On the legged bag the 5/4 caps under-converge the corridor walk and diverge at t≈188 s even with the deadline off (replay ATE 340 m vs 4.5 m at 12/6); live, `time_limit_ms` gates the schedule either way. |
| analytic IMU factor | not built | ~5 ms off solve, no accuracy cost (last autodiff residual; spec-04 banner item) |
| ceres `num_threads` (ct_frontend.cpp:621, marginalization.cpp:241) | 1 | >1 helps only on large problems; ours is small. Must stay deterministic (replay==live). |

## Accuracy / robustness constants (change only with GT in hand)

| knob | now | role |
|---|---|---|
| `sensors.imu.cov_*` (legged_underground) | ≈×40 the calib Allan densities | IMU-vs-LiDAR trust. Static-calib Allan floors over-trust the IMU under locomotion vibration: calib values diverge (448 m ATE / 120 s), ×10 still diverges (116 m), ×20 holds (0.5 m), ×40 holds (0.6 m; shipped values 0.2 m, full bag 4.5 m / 161 m). Set per platform from a GT replay sweep, not from the calib file. |
| `backend.anchor_sigma` | 0.1 (λ=100) | gauge-anchor damping: every pose marginal floors at σ² (↓σ tightens loop-gate marginals) vs rigid corrections propagate at ~chain_info/(chain_info+λ) per solver iteration (↑σ floats faster). Measured (10-edge chain + conflicting absolute prior): λ=100 absorbs >99.9% of a forced rigid shift; λ=1e4 stalls at 38% after 5000 iters even in batch GN; λ=1e8 ≈ hard prior (0.02%) and the conflict permanently distorts relative geometry instead. Floor at default = (10 cm)². |
| `spline.n_cp_max` | 1 | adaptive knot density; >1 is math-validated but inflates the solve past budget — flip to 3 only after the analytic IMU lands |
| `spline.window_knots` / `knot_dt_ms` | 8 / 100 | window length (knots) / one knot per 10 Hz sweep |
| `bias.gyr_max` / `acc_max` / `knot_dt_ms` | 0.5 / 5.0 / 500 | bias box bounds [rad/s, m/s²] + random-walk knot cadence [ms] |
| `init_time_s` | 1.0 | static-init window; longer tightens gravity/bias init (frozen accel bias can't absorb its error) |
| `lidar.plane_thresh` / `max_match_dist_sq` / `point_cov` | 0.1 / 5.0 / 1e-3 | plane-fit planarity gate / assoc gate radius² / LiDAR point variance |
| `lidar.normal_strata` / `min_factors_per_normal` | 7 / 50 | factor-cap stratification (weak-axis observability protection) |
| `visual.ncc_thre` / `img_point_cov` / `outlier_threshold` | 0.5 / 100 / 1000 | photometric correlation gate / residual weight / SSD reject |
| `visual.active_box_m` | 20 | visual-map memory bound (NOT per-sweep cost — that's the cache); set to camera re-observability range, not LiDAR scale |
| `preprocess.voxel_surf_m` / `point_filter_num` / `blind` / `det_range` | 0.5 / 2 / 0.5 / 120 | L1 scan downsample / decimation / min-max range [m] |

## Hardcoded code constants (need recompile)

| const (file) | value | role / retune note |
|---|---|---|
| `kWeakAxisCos` (ct_frontend.cpp:40) | 0.966 (~15°) | normal-vs-weak-axis alignment for factor-cap exemption |
| `kMinSupportWeight` (ct_frontend.cpp:812) | 0.02 | tail-knot pinning threshold (u³/6 basis support); below → pin the knot |
| `kTailSigmaVel` / `kTailSigmaRate` (ct_frontend.cpp:713) | 0.2 m/s / 0.1 rad/s | tail-anchor strengths over the unmeasured span |
| `kMaxInsertRangeM` (ct_frontend.cpp:1160) | 200 | map-insert sanity bound [m] from anchor |
| `kHuberWhitened` (residuals_lidar/visual.cpp) | 1.0 | robust-kernel knee on the whitened (unit-variance) residual |
| `VisualMapConfig` defaults (visual_map.hpp) | ref_obs_cap 30, ref_add_angle 10°, ref_converged_obs 8 / 30°, grid_cell_px 32, depth_continuity_m 0.5 | visual-point reference-patch lifecycle; not yet config-exposed |
