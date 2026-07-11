# Meridian L3 Back-End — Engineering Spec Sheet

Dense at-a-glance reference for the `meridian_backend` global optimizer. For the pedagogical
walk-through see `docs/BACKEND_COURSE.md`; the authoritative design contract is
`docs/specs/05_backend_graph.md`. Config defaults are quoted from
`src/meridian_config/include/meridian/config/config.hpp` (the code is the ground truth for values).

---

## 1. Overview

L3 is a **GTSAM 4.2 iSAM2 incremental pose-graph global optimizer** (`Isam2BackEnd`,
`BackEndKind::Isam2`). It owns one `gtsam::ISAM2` instance and the canonical **map-frame** estimate
of every keyframe pose `X(i) = T_map_body`. It stitches three input streams into one factor graph
and broadcasts corrections:

- **Inputs:** `KeyframePacket` from L2 (`add_keyframe`), `LoopConstraint` from L5/`meridian_place`
  (`add_loop_constraint`), `GnssFix` from L0 (`add_absolute`).
- **Outputs:** `GraphUpdate` (moved keyframes + `loop_closed`) → L4/L2; `corrected_trajectory()`
  (map-frame `StampedPose`); `T_map_odom` correction → L2; latest pose marginal cov → L5 gate;
  versioned `CalibrationSet` (refined GNSS lever) → L2; `.g2o` snapshot for offline inspection.
- **Thread:** **Live** — its own back-end thread (T3), shared with L5; `add_*` only **stage**,
  `optimize()` folds on a decoupled cadence (`backend_loop()`, `meridian_pipeline.cpp:436`).
  **Replay** — no thread/queue/clock; methods run inline on the caller's thread and `optimize()`
  folds once after every keyframe (`backend_runner.cpp:177`).
- **Boundary invariant:** everything entering a GTSAM noise model is **rotation-first**
  `[rx,ry,rz,tx,ty,tz]`; everything leaving toward Meridian types is **translation-first**
  `[ρ;φ] = [tx,ty,tz,rx,ry,rz]` (`isam2_backend.hpp:31-34`). Reordering happens **exactly once**
  per covariance via `reorder_meridian_to_gtsam` / `reorder_gtsam_to_meridian`.

---

## 2. Interface surface — `IBackEnd`

`include/meridian/backend/ibackend.hpp`. All methods run on the single serial driver thread.

| Method                     | Signature                                         | Does                                                                                                     | When called                              |
| -------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `add_keyframe`             | `void(KeyframePacket&&)`                          | Stages an odometry edge (between / restart-IMU / first-keyframe anchor) per `constraint_kind`. No solve. | Each keyframe from L2.                   |
| `add_loop_constraint`      | `void(const LoopConstraint&)`                     | Fitness-gates, then buffers the loop in PCM. No solve.                                                   | Each loop from L5.                       |
| `add_absolute`             | `void(const GnssFix&, uint64_t nearest_kf_id)`    | Quality-gates a fix; buffers into datum init, or (post-lock) stages a GNSS factor.                       | Each fix from L0.                        |
| `optimize`                 | `GraphUpdate()`                                   | Runs PCM, folds the whole staged batch into iSAM2 in one incremental update, returns moved keyframes.    | On cadence / immediate request.          |
| `corrected_trajectory`     | `vector<StampedPose>() const`                     | Map-frame pose of every estimated keyframe, in `kf_order_`.                                              | Pull, output writers.                    |
| `refined_calibration`      | `shared_ptr<const CalibrationSet>() const`        | Refined-lever snapshot once frozen, else offline calib.                                                  | L2 calib feedback.                       |
| `diagnostics`              | `BackEndDiagnostics() const`                      | Counters + last-fold state (§8).                                                                         | Telemetry/introspection.                 |
| `wants_immediate_optimize` | `bool() const`                                    | `batch_has_loop_ \|\| datum_just_locked_` — bypass the cadence timer.                                    | Pipeline cadence driver.                 |
| `latest_pose_marginal`     | `optional<PoseCov6>() const`                      | Marginal cov of the latest keyframe, translation-first; `nullopt` pre-first-fold.                        | L5 loop gate, GNSS skip gate.            |
| `pose_of`                  | `optional<Pose>(uint64_t) const`                  | Corrected map pose of a keyframe.                                                                        | Read-only view fed to L5.                |
| `chain_cov_between`        | `optional<Matrix6>(uint64_t a, uint64_t b) const` | Odometry-chain covariance a→b, translation-first.                                                        | Read-only view fed to L5 + internal PCM. |
| `map_odom`                 | `Pose() const`                                    | `T_map_odom_`; identity until first fold, jumps on a loop.                                               | L2 rebase, viz TF.                       |
| `write_g2o`                | `void(const string&) const`                       | Best-effort pose sub-graph snapshot (X vertices + between/loop edges only).                              | Debug.                                   |

Factory: `makeBackEnd(const BackendConfig&, shared_ptr<const CalibrationSet>, TelemetrySink*, bool deterministic=false)`.

---

## 3. Factor types in the graph

Variable key namespaces (`keys.hpp`): `x`=keyframe pose, `v`=velocity, `b`=IMU bias,
`e`=sensor extrinsic, `g`=single GNSS datum node. Steady-state graph is **pose-only**; V/B are
transient (restart-edge endpoints, marginalized at the next normal keyframe unless `keep_inertial`).

| Factor                 | GTSAM class                                                        | Connects                               | Source / `constraint_kind`                   | Noise model & tangent order                                                                                       | Config keys                     |
| ---------------------- | ------------------------------------------------------------------ | -------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| **Gauge anchor**       | `GaugeDampingFactor` (custom)                                      | `X(first)`                             | first keyframe / `AbsolutePrior`             | `error()≡0`; `linearize→√λ·I₆`, λ=1/σ². Adds isotropic damping, no bias.                                          | `anchor_sigma`                  |
| **Odometry (normal)**  | `BetweenFactor<Pose3>`                                             | `X(rel_to_id), X(id)`                  | `RelativeBetween`                            | `Gaussian::Covariance(Σ)`; Σ=`constraint_cov` already **rotation-first**, after observability inflation.          | (obs-inflation keys)            |
| **Odometry (restart)** | `CombinedImuFactor`                                                | `X,V,B(i)`, `X,V,B(j)`                 | `ImuPreintegration`                          | PIM rebuilt from `imu_summary` via `pim_from_summary`. Loose V(i)/B(i) priors (σ=10, σ=1).                        | `imu.*`                         |
| **Loop closure**       | `BetweenFactor<Pose3>`                                             | `X(from_id), X(to_id)`                 | `LoopConstraint`, after PCM clique           | `Robust(Huber(k), Gaussian::Cov(Σ))`; Σ reordered **translation→rotation-first** via `reorder_meridian_to_gtsam`. | `loop_huber_k`, `robust_kernel` |
| **GNSS position**      | `GnssFactor` / `GnssFactorEndpoint` (custom `NoiseModelFactor3/2`) | `X(i),X(j),G` (interp β) or `X(end),G` | `GnssFix` post-lock                          | `Robust(Huber(k), Gaussian::Cov(cov_enu))`, 3-DoF ENU.                                                            | `gnss_huber_k`                  |
| **GNSS-refined**       | `GnssFactorRefined(Endpoint)`                                      | `…, E(GnssLink)`                       | as above when online lever refinement active | same Huber; adds `E` Pose3 variable.                                                                              | `extrinsic_*`                   |
| **GNSS-origin prior**  | `PriorFactor<Pose3>`                                               | `G`                                    | first datum lock                             | `Diagonal::Sigmas` rotation-first `[r;t]`: roll/pitch 1e-3, yaw=fitted σ, trans 0.5 m.                            | (datum keys)                    |
| **Extrinsic prior**    | `PriorFactor<Pose3>`                                               | `E(GnssLink)`                          | online refinement seed / re-pin              | `Isotropic::Sigma(6, extrinsic_refine_sigma)` seed; `extrinsic_prior_sigma` on re-pin.                            | `extrinsic_*`                   |

**Tangent-order rule (the classic silent bug):** Meridian core types are translation-first
`[ρ;φ]` (`PoseCov6`, `LoopConstraint.cov`, GNSS/observability, `chain_cov`). GTSAM `Pose3` is
rotation-first. The **one exception** is `KeyframePacket.constraint_cov`, which the front-end already
ships **rotation-first** — it enters the between-factor directly with **no** reorder
(`isam2_backend.cpp:139-177`). Information-form `constraint_cov` is audited and inverted at ingest
with a `backend/info_form` warning.

---

## 4. The robustness pipeline

Layered **in series**: L5 single-loop self-test → L3 cross-loop PCM max-clique → committed-Huber
kernel → off-thread batch-GNC. Each stage is independent and most heavy layers are off by default.

| Layer                       | Scope           | What it does                                                                                                                                                                                                                                              | Where                                                | Gating config (value)                                                     | Default                    |
| --------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------- |
| **L5 self-test (Stage D)**  | single loop     | Whitened χ² of one GICP loop vs the corrected odometry chain between _its own_ two endpoints. Rejects loops inconsistent with plausible drift.                                                                                                            | `meridian_place` `pcm_self_test.cpp`                 | `place.pcm_chi2_conf` (0.99)                                              | on (`place.pcm`=true)      |
| **Fitness gate**            | single loop     | `lc.fitness < loop_min_fitness` → dropped before PCM.                                                                                                                                                                                                     | `add_loop_constraint`, `isam2_backend.cpp:290`       | `loop_min_fitness` (0.5)                                                  | on                         |
| **L3 cross-loop PCM**       | loop set        | Pairwise SE(3) consistency `ε=Log(Z₁⁻¹·B·Z₂·D)`, `d²=εᵀΣ_ε⁻¹ε ≤ χ²₆,α`; consistency graph → bounded **max-clique**. Admits clique, evicts displaced, rejects inconsistent (clique≥2). Chain cov composed incrementally, never from the graph.             | `pcm.cpp`, `max_clique.cpp`, `process_pending_loops` | `pcm_chi2_alpha` (0.99 → χ²₆≈16.81), `pcm_max_nodes` (64 exact cap)       | on                         |
| **Committed Huber**         | per loop factor | Admitted loops enter iSAM2 wrapped in `Robust(Huber(k))`; convex, safe inside one `ISAM2::update`; kernel fixed at admission (not annealed).                                                                                                              | `robust_kernels.cpp`, `make_huber_noise`             | `robust_kernel` (Huber), `loop_huber_k` (1.345), `gnss_huber_k` (1.345)   | on                         |
| **Batch-GNC consolidation** | loop sub-graph  | Off-thread/off-live TLS-GNC re-judgement of the in-graph loop sub-graph on a **copy** (loops robust, odometry known-inlier, keyframes loosely pinned). Loops driven below `gnc_reject_w` are removed + PCM-rejected. **Only runs when `deterministic_`.** | `gnc_consolidation.cpp`, `run_gnc_consolidation`     | `gnc_consolidate_interval` (10; 0=off), `gnc_reject_w` (0.1), barc²=χ²₆,α | runs (replay) per interval |
| **In-graph amortised GNC**  | live graph      | Anneal μ across iSAM2 updates, reweight loops in place.                                                                                                                                                                                                   | (gated by flag)                                      | `gnc_enabled` (false), `gnc_anneal_steps` (5)                             | **off / experimental**     |
| **Loop eviction**           | loop set        | PCM `to_evict`: a larger clique displaces an in-graph loop → factor removed via `remove_indices_`, loop returned to `Pending` (re-judgeable, not rejected).                                                                                               | `finalize_pending_loops`                             | —                                                                         | on                         |
| **Extra Dogleg passes**     | solve           | After a loop fold, run `extra_iters_loop` extra `isam2_->update()` passes to settle the rigid correction.                                                                                                                                                 | `optimize`, `isam2_backend.cpp:828`                  | `extra_iters_loop` (4), `extra_iters_normal` (0)                          | on                         |

Indeterminate-system recovery: `run_update_with_recovery` retries once on QR-factorization, then
abandons the batch and rolls back uncommitted keyframes; increments `fallback_count`,
`last_optimize_diverged=true`.

Max-clique is **exact Bron-Kerbosch with pivoting** for n≤`pcm_max_nodes`, with a deterministic
expansion budget `kMaxExpansions=200000` (`max_clique.cpp:13`); on budget exhaustion or n>cap it
falls back to a deterministic greedy clique (lexicographic tie-break throughout).

---

## 5. `BackendConfig` — full field table

Every field of `struct BackendConfig` (`config.hpp:418-495`), grouped by concern. Values are the
**code defaults**; deployment YAML may override.

### Bring-up

| Field              | Default | Unit | Meaning                                                                                                                                  |
| ------------------ | ------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `enable`           | `true`  | —    | Off runs the pipeline without the back-end.                                                                                              |
| `kind`             | `Isam2` | enum | Only back-end implementation.                                                                                                            |
| `correct_frontend` | `true`  | —    | Feed admitted loop/GNSS corrections back into L2 (re-anchor spline + local map). Off decouples L2 as pure odometry; A/B + safe fallback. |

### Gauge / anchor

| Field          | Default | Unit  | Meaning                                                                                                                                               |
| -------------- | ------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `anchor_sigma` | `0.1`   | m/rad | Gauge damping on first pose, λ=1/σ²=100. Two-sided: every pose marginal floors at σ²; rigid corrections propagate at ~chain_info/(chain_info+λ)/iter. |

### iSAM2

| Field                      | Default | Unit  | Meaning                                                                                     |
| -------------------------- | ------- | ----- | ------------------------------------------------------------------------------------------- |
| `isam2_relinearize_skip`   | `1`     | count | `ISAM2Params.relinearizeSkip`.                                                              |
| `isam2_relinearize_thresh` | `0.1`   | —     | (struct field; per-variable-class thresholds are set explicitly in `make_isam2`, see note). |
| `extra_iters_normal`       | `0`     | count | Extra Dogleg passes on a normal batch.                                                      |
| `extra_iters_loop`         | `4`     | count | Extra Dogleg passes when a loop was admitted.                                               |
| `isam2_use_qr`             | `false` | —     | QR vs Cholesky factorization (QR is the recovery fallback).                                 |
| `optimize_interval_ms`     | `100.0` | ms    | Min wall-clock between folds (live cadence). Ignored in replay.                             |
| `queue_warn_depth`         | `32`    | count | Input queue depth raising a warning.                                                        |

> `make_isam2` overrides relinearize thresholds per variable class (rotation-first): `x`=[0.05·3, 0.10·3], `v`=0.10, `b`=[1e-3·3,1e-4·3], `e`=1e-3, `g`=[1e-3·3,5e-3·3]; Dogleg, Cholesky (or QR), `cacheLinearizedFactors`, `findUnusedFactorSlots`, `enableDetailedResults`.

### Observability → noise inflation

| Field                 | Default | Unit  | Meaning                                         |
| --------------------- | ------- | ----- | ----------------------------------------------- |
| `obs_inflation_max`   | `1e4`   | ×     | ρ_max: max per-axis variance multiplier.        |
| `obs_inflation_gamma` | `2.0`   | —     | γ in λ_k = 1+(ρ_max−1)(1−s_k)^γ.                |
| `degenerate_thresh`   | `0.05`  | score | Axis with score below this is degenerate.       |
| `degenerate_lock`     | `true`  | —     | Hard-lock a degenerate axis to ρ_max inflation. |

### Loops + PCM

| Field              | Default | Unit  | Meaning                                          |
| ------------------ | ------- | ----- | ------------------------------------------------ |
| `loop_min_fitness` | `0.5`   | —     | Reject loops below this GICP fitness before PCM. |
| `pcm_chi2_alpha`   | `0.99`  | —     | PCM consistency confidence → χ²₆ gate (≈16.81).  |
| `pcm_max_nodes`    | `64`    | count | Exact max-clique node cap; greedy above it.      |

### Robust kernels / GNC

| Field                      | Default | Unit   | Meaning                                                  |
| -------------------------- | ------- | ------ | -------------------------------------------------------- |
| `robust_kernel`            | `Huber` | enum   | Committed incremental loop kernel (Huber/Cauchy/Gm/Tls). |
| `loop_huber_k`             | `1.345` | —      | Huber knee for loop factors.                             |
| `gnss_huber_k`             | `1.345` | —      | Huber knee for GNSS factors.                             |
| `gnc_reject_w`             | `0.1`   | weight | Loop GNC weight below which it is removed.               |
| `gnc_enabled`              | `false` | —      | **Experimental** amortised in-graph GNC.                 |
| `gnc_anneal_steps`         | `5`     | count  | μ anneal steps for in-graph GNC.                         |
| `gnc_consolidate_interval` | `10`    | count  | Admitted loops between batch consolidations (0=off).     |

### GNSS

| Field                        | Default | Unit  | Meaning                                                                         |
| ---------------------------- | ------- | ----- | ------------------------------------------------------------------------------- |
| `gnss_enabled`               | `true`  | —     | Master GNSS gate.                                                               |
| `gnss_max_cov`               | `25.0`  | m²    | Drop fix if `trace(cov_enu)` above.                                             |
| `gnss_lock_yaw`              | `false` | —     | External heading available (yaw lock).                                          |
| `gnss_min_baseline`          | `5.0`   | m     | Min travelled baseline before datum fit.                                        |
| `gnss_min_excitation`        | `3.0`   | m     | Min dominant-axis span of buffered ENU track.                                   |
| `gnss_min_speed`             | `0.5`   | m/s   | Speed a fix must exceed to count as moving.                                     |
| `gnss_min_moving_fixes`      | `5`     | count | Moving fixes required before datum fit.                                         |
| `gnss_datum_yaw_sigma_max`   | `5.0`   | deg   | Reject datum fit above this yaw uncertainty.                                    |
| `gnss_skip_if_confident`     | `true`  | —     | Skip a fix no tighter than the back-end marginal.                               |
| `gnss_skip_confidence_k`     | `1.0`   | κ     | Confidence multiplier in the skip test.                                         |
| `gnss_min_spacing`           | `1.0`   | m     | Min travel between admitted fixes (decimation).                                 |
| `gnss_redistribute`          | `false` | —     | **Deferred** (§6.6) drift redistribution across an outage span.                 |
| `gnss_reacq_fix`             | `1`     | enum  | Min fix-quality to count as re-acquired (also used as chi2-health persistence). |
| `gnss_reacq_persist`         | `5`     | count | Consecutive fixes to declare re-acquisition / auto-disable.                     |
| `gnss_redistribute_span_max` | `200`   | kf    | Max keyframes back redistribution reaches (deferred).                           |

### Online extrinsics (§10, off by default)

| Field                    | Default | Unit | Meaning                                                 |
| ------------------------ | ------- | ---- | ------------------------------------------------------- |
| `extrinsic_refine`       | `false` | —    | Master switch for online GNSS lever-arm refinement.     |
| `extrinsic_prior_sigma`  | `1e-3`  | m    | Tight prior σ on re-pin (clamp/freeze).                 |
| `extrinsic_refine_sigma` | `1e-2`  | m    | Loose seed prior σ when E enters.                       |
| `extrinsic_excite_rot`   | `0.5`   | rad  | Cumulative rotation before refinement engages.          |
| `extrinsic_excite_trans` | `2.0`   | m    | Cumulative translation before refinement engages.       |
| `extrinsic_freeze_cov`   | `1e-6`  | m²   | Freeze when lever marginal trace below.                 |
| `extrinsic_max_dev`      | `0.1`   | m    | FM-5 sanity clamp; revert + freeze if lever leaves box. |

### Marginalization / cadence / debug

| Field                  | Default                   | Unit | Meaning                                                                      |
| ---------------------- | ------------------------- | ---- | ---------------------------------------------------------------------------- |
| `keep_inertial`        | `false`                   | —    | Keep restart V/B variables live (else marginalized).                         |
| `reintegrate_thresh`   | `0.1`                     | m    | Pose-move threshold to report a keyframe moved (gates both `‖ρ‖` and `‖φ‖`). |
| `emit_moved_cov`       | `false`                   | —    | Fill moved-set covariances (costly).                                         |
| `loop_gate_k`          | `3.0`                     | σ    | k·σ_pos search radius for the loop pre-filter marginal.                      |
| `imu` (`BackendImu`)   | acc/gyr 0.1, bias_rw 1e-4 | —    | PIM noise for restart edges.                                                 |
| `debug_dump_residuals` | `false`                   | —    | Debug residual dump.                                                         |
| `snapshot_on_request`  | `false`                   | —    | Enable on-request snapshot.                                                  |
| `snapshot_dir`         | `/tmp/meridian`           | path | Snapshot output dir.                                                         |

---

## 6. Loop detector feed — `meridian_place` (L5)

`HierarchicalLoopDetector` (`ILoopDetector`) produces `LoopConstraint`s consumed by
`add_loop_constraint`. **The detector never links L3** — the pipeline supplies read-only
`std::function` views (`pose(id)`, `chain_cov(a,b)`, `stamp`, `obs`) from the back-end.

Cascade per `detect()`: eligibility → **ScanContext++** ring-key retrieval (STD/BTC re-rank seam
skipped) → **submap accumulation** (last-N keyframes composed/voxelized, LRU-cached) → **small_gicp
verify** → **Stage D single-loop self-test** (χ² of the GICP loop vs the odometry chain between its
own endpoints). This single-loop self-test is **distinct** from the L3 cross-loop max-clique PCM
(§4).

`LoopConstraint` fields (`loop_constraint.hpp`): `from_id` (candidate), `to_id` (query/newest),
`T_from_to` (GICP-refined), `cov` (**PoseCov6, Covariance, translation-first** `[ρ;φ]`), `fitness`
(GICP inlier ratio → drives the L3 robust kernel). Covariance is shaped in L5 (`shapeLoopCov`):
fitness-scaled GICP information + λ regularizer, rotation→translation-first reorder, per-axis
observability loosening, PSD floor. The L3 boundary does the translation→rotation-first reorder.

Key `PlaceConfig` values (`config.hpp:521`). `place.enable` default `false`; deployment YAML may
override (newer-college-quad keeps it false: enabling it perturbs the L2 ikd-Tree async-rebuild
timing).

| Field                                                                         | Default                 | Meaning                                                              |
| ----------------------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------- |
| `enable` / `pcm`                                                              | false / true            | L5 master switch / single-loop self-test on.                         |
| `submap_window`/`submap_voxel`/`submap_cache`                                 | 5 / 0.25 m / 32         | Anchor submap granularity + LRU.                                     |
| `detect_period_kf` / `cooldown_kf`                                            | 1 / 5                   | Run cadence / suppress after an emitted loop.                        |
| `min_time_gap` / `min_kf_gap`                                                 | 30 s / 30               | Revisit eligibility floors.                                          |
| `sc_Nr`/`sc_Ns`/`sc_rmax`                                                     | 20 / 60 / 80 m          | ScanContext descriptor geometry.                                     |
| `sc_knn`/`sc_topK`/`sc_dist_thresh`                                           | 15 / 5 / 0.13           | Retrieval breadth / accept (precision gate).                         |
| `gicp_fitness_min`/`gicp_overlap_min`/`gicp_rmse_max`                         | 0.6 / 0.4 / 0.3 m       | GICP accept gates.                                                   |
| `gicp_max_corr_dist`/`gicp_cond_max`/`gicp_num_threads`                       | 1.0 m / 1e4 / 4         | GICP correspondence/condition/threads (forced 1 when deterministic). |
| `pcm_chi2_conf`                                                               | 0.99                    | Self-test χ² confidence.                                             |
| `cov_lambda` / `cov_degenerate_eig` / `cov_degenerate_mult` / `cov_psd_floor` | 1e-3 / 1.0 / 100 / 1e-9 | Loop covariance shaping.                                             |

---

## 7. GNSS feed

Fixes enter via `add_absolute(fix, nearest_kf_id)`. Path (`isam2_backend.cpp:546`):

1. **Master gate:** `gnss_enabled && !gnss_auto_disabled_`; needs a `GnssLink` antenna lever in calib
   (else dropped). Quality floor: `fix==None || trace(cov_enu) > gnss_max_cov` → dropped.
2. **ENU origin:** first accepted fix fixes the tangent-plane origin (`gnss_origin_`). Origin set ≠
   datum locked.
3. **Bracket:** `find_bracket` locates the keyframe pair straddling the fix stamp (or a single
   endpoint); body pose at the fix is the SE(3) interpolation across the bracket.
4. **Datum init (pre-lock):** buffer (ENU, map-antenna, speed) correspondences; `datum_.try_lock`
   gates on `gnss_min_baseline`, `gnss_min_excitation`, `gnss_min_speed`, `gnss_min_moving_fixes`,
   `gnss_datum_yaw_sigma_max`. On lock: insert `G=T_map_enu` with a weak `PriorFactor<Pose3>`
   (tight trans/roll/pitch, yaw = fitted σ); sets `datum_just_locked_` → immediate optimize.
5. **Post-lock admit** (`admit_gnss_fix`): `GnssGate::evaluate` (`gnss_gate.cpp`) →
   RejectQuality / SkipConfident (vs back-end marginal × κ²) / SkipSpacing (`gnss_min_spacing`) /
   Accept. Then **FM-6 chi2 health**: whitened ENU residual against `cov_enu`, gate χ²₃,₀.₉₉; a
   sustained run (`gnss_reacq_persist`) of failures trips `gnss_auto_disabled_`. Accepted →
   interpolated `GnssFactor` or endpoint `GnssFactorEndpoint` (refined variants when online lever
   refinement is active).

**Deferred:** §6.6 drift redistribution on re-acquisition (`gnss_redistribute=false`; span capped at
`gnss_redistribute_span_max`). Until built, re-acquired GNSS re-enters as ordinary factors.

---

## 8. Outputs & telemetry

**`GraphUpdate`** (`graph_update.hpp`): `moved[]` (id, `new_T_map_body`, optional translation-first
cov), `loop_closed`. A keyframe is "moved" on first emission, or when `‖ρ‖ > reintegrate_thresh` or
`‖φ‖ > reintegrate_thresh` since last emit. Covariance filled only if `emit_moved_cov`.

**Other outputs:** `corrected_trajectory()` (map-frame `StampedPose` over `kf_order_`);
`T_map_odom_ = X_last · T_ref_body(last)⁻¹` (recomputed each fold, jumps on a loop);
`latest_pose_marginal()` (cached, translation-first); `refined_calibration()` (versioned snapshot
once the lever freezes); `.g2o` pose sub-graph.

**`BackEndDiagnostics`** (`backend_diagnostics.hpp`): `num_keyframes`, `num_loops`,
`num_loops_rejected`, `num_gnss_factors`, `isam_update_ms`, `last_optimize_diverged`, `chi2`,
`variables_relinearized`, `optimize_lag`, `datum_locked`, `fallback_count`.

**Telemetry keys emitted** (`telemetry_keys.hpp`):

| Kind    | Keys                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scalars | `backend/chi2`, `backend/n_factors`, `backend/n_loops`, `backend/n_gnss`, `backend/update_ms`, `backend/relin_count`, `backend/optimize_lag`, `backend/fallback_count`, `backend/obs_min`, `backend/queue_depth` (pipeline)                                                                                                                                                                                                                                    |
| Vectors | `backend/gnss/residual` (e,n,u), `backend/observability/<id>` (rx..tz)                                                                                                                                                                                                                                                                                                                                                                                         |
| Poses   | `map/keyframe` (latest corrected)                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Markers | `backend/loop_edge` (green admitted / red rejected), `backend/graph_nodes`, `backend/graph_edges`                                                                                                                                                                                                                                                                                                                                                              |
| Events  | `backend/contiguity`, `backend/psd_clamp`, `backend/indeterminate`, `backend/degenerate`, `backend/info_form`, `backend/obs_frame`, `backend/relinearize`, `backend/window_restart_bridge`, `backend/marginalize_skip`, `backend/gnss/datum_locked`, `backend/gnss_disabled`, `backend/gnss_skip`, `backend/loop_accepted`, `backend/loop_rejected_pcm`, `backend/loop_rejected_gnc`, `backend/extrinsic_excited`/`_frozen`/`_clamped`, `backend/g2o_snapshot` |
| Timing  | `backend.optimize`                                                                                                                                                                                                                                                                                                                                                                                                                                             |

---

## 9. Threading

- **Serial driver model:** all `IBackEnd` methods run on one thread; nothing is locked. iSAM2 is
  owned exclusively.
- **Live cadence** (`backend_loop()`, `meridian_pipeline.cpp:436`): the back-end thread drains the
  lossless queue, stages the whole burst in one wake, and folds at most once per
  `optimize_interval_ms` (100 ms) — **unless** `wants_immediate_optimize()` (`batch_has_loop_ ||
datum_just_locked_`) forces it. Loops detected by L5 re-enter via the queue and fold next wake.
- **Replay cadence** (`backend_runner.cpp:177`): `optimize()` runs after **every** keyframe; no clock
  read, no timer. Fixed cadence + fixed factor-insertion order ⇒ COLAMD ordering and relinearization
  are pure functions of the graph.
- **Max-clique budget:** an expansion **count**, not wall-clock.

---

## 10. Status / deferred

| Item                                                                      | Where                                                    | Status                                                                           |
| ------------------------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------- |
| iSAM2 pose spine, gauge anchor, between/restart-IMU edges                 | `isam2_backend.cpp`                                      | **Implemented & validated** (NCD quad-easy: backend identity exact).             |
| Restart-IMU bridge + transient V/B marginalization                        | `add_restart_imu_edge`, `perform_bridge_marginalization` | **Implemented**.                                                                 |
| Observability→noise inflation, degenerate-axis lock                       | `observability_inflation.cpp`                            | **Implemented**.                                                                 |
| Cross-loop PCM max-clique + committed Huber                               | `pcm.cpp`, `max_clique.cpp`, `robust_kernels.cpp`        | **Implemented & validated** (quad-easy ~5/1 admit/reject).                       |
| Indeterminate-system QR recovery + batch rollback                         | `run_update_with_recovery`                               | **Implemented**.                                                                 |
| GNSS datum init, interpolated/endpoint factors, gating, FM-6 auto-disable | `datum.cpp`, `gnss_factor.cpp`, `gnss_gate.cpp`          | **Implemented & validated** (clean-portion).                                     |
| Batch-GNC consolidation (off-thread/replay)                               | `gnc_consolidation.cpp`                                  | **Implemented**, runs only when `deterministic_`, interval-gated.                |
| `correct_frontend` feedback                                               | pipeline + `frontend.*.rebase_*`                         | **Implemented**, on by default.                                                  |
| Online GNSS lever-arm extrinsic refinement (§10)                          | `update_extrinsic`, `gnss_factor_refined.cpp`            | **Implemented, off by default** (`extrinsic_refine=false`); enable per-platform. |
| In-graph amortised GNC                                                    | gated by `gnc_enabled`                                   | **Experimental, off by default**.                                                |
| GNSS drift redistribution on re-acquisition (§6.6)                        | `gnss_redistribute`                                      | **Deferred** (designed, not built; default false).                               |
| Keyframe culling / graph sparsification (§11.2)                           | —                                                        | **Deferred** utility.                                                            |
