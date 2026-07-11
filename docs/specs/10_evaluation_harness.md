# Meridian — Evaluation & Test Harness (Spec 10)

> Status: DRAFT for build. Cross-cutting spec. Authoritative for: the offline replay harness, accuracy metrics (ATE/RPE via evo), full-system + module-integration acceptance criteria, the unit/integration/regression test strategy, and map-quality (cloud-to-cloud / cloud-to-mesh) metrics. Consumes the `RunManifest` (Spec 00 §6 / `specs/00_architecture.md`), the telemetry bus (Spec 00 §10), and the core interfaces (Spec 01 / `specs/01_interfaces_and_data_types.md`). References — does **not** duplicate — the dataset decision in `docs/DATASET.md`. Frame/unit conventions inherited from Spec 01 §2. Logging/timing primitives (`ScopedTimer`, `TimingRegistry`, the `ParquetTelemetrySink` and its fixed columnar schema, and the `Config::dump`/`RunManifest` provenance) are defined in Spec 09 (`specs/09_debug_introspection.md`) and consumed here.
>
> **What Meridian is, for the purposes of this spec.** A single, complete **discrete, tightly-coupled LiDAR-Inertial** estimator (Spec 00): one per-sweep LIO front-end (`lio` — internal constant-screw deskew, voxel-hash local map, Gauss-Newton ICP with an interval-averaged IMU prior); an iSAM2 factor-graph back-end; and a GPU **nvblox** TSDF+colour+Marching-Cubes map. Camera and GNSS streams ride through the pipeline unfused. There is **one** system to evaluate — not a sequence of shippable versions. The harness therefore measures one estimator against ground truth. (The FAST-LIO2-style iEKF oracle and the continuous-time estimator that previously sat behind the same `IFrontEnd` are **removed** — Spec 00 §5.4. The front-end cross-check is direct: the synthetic-world GT-tracking suite, Jacobian-vs-numeric parity, and the Monte-Carlo covariance-chain test; §8.)
>
> **Deployment target.** NVIDIA Jetson Orin; a CUDA GPU is **always present**. Mapping is nvblox, GPU-only — there is no CPU map path to evaluate and no CPU-fallback gate. **Single LiDAR + single IMU + single camera + GNSS** — there is no multi-LiDAR merge to test.

---

## 0. Why this spec exists, and the one idea it is built on

A SLAM estimator is a long pipeline of approximations — deskew, linearization, data association, marginalization, robust kernels — and each approximation can be individually defensible while the composition silently drifts. The only defence is **measurement against ground truth on the live code path, reproducibly, every commit**. This spec defines the machinery for that.

The single load-bearing idea, inherited from the architecture, is:

> **The offline replay harness drives the *same* core objects, through the *same* `IFrontEnd::ingest` / `IBackEnd::add_keyframe` / `IMapLayer::integrate` calls, in the same order, that the live ROS 2 node would.** It differs from the live system only in (a) where samples come from (a bag instead of DDS), and (b) that it owns the clock (the replay clock is driven by bag time, Spec 00 §10.1 — no core module reads the wall clock). Nothing in `meridian_core` can tell whether it is being driven live (`meridian_ros`) or replayed (`meridian_tools`); both feed the *same* `MeridianPipeline` via the *same* `ISensorSource` boundary (Spec 00 §9.3, Spec 01 §7.1).

Everything below is a consequence of this. If a metric can only be computed in the harness and not in the node, or a code path runs only under test, we have violated the principle and the number is worthless.

This is why Spec 00 §11 confines each stage to a single thread and Spec 01 makes every interface a clock-free, side-effect-explicit call: it is precisely the property that makes replay *be* the live path rather than a parallel re-implementation of it. The harness is the payoff for that discipline.

### 0.1 Scope boundary

In scope: trajectory accuracy (ATE/RPE), map geometric quality (cloud-to-cloud, cloud-to-mesh, surface coverage), per-stage timing/throughput against the Jetson Orin real-time budget, the test pyramid (unit→integration→regression→system), CI smoke gating, and the debug hooks specific to evaluation.

Out of scope, deferred with the system (Spec 00 §12): semantic/object-detection metrics and path-planning (ESDF) metrics. The harness is structured so these slot in as additional `Metric` implementations (§5.6) onto the same `RunManifest`/result substrate later, attaching to the same retained keyframe store and nvblox TSDF the surface metrics already use. We build the geometry and trajectory metrics now; the first-pass deliverable that the map metrics judge is the **colourised nvblox mesh** (Spec 00 §0, Spec 01 §7.7).

---

## 1. Architecture of the harness

### 1.1 Components

```
                       ┌──────────────────────────────────────────────┐
                       │                meridian_eval (lib)               │
                       │  (ROS-agnostic, links meridian_core, no ROS)     │
  bag (ros2bag/mcap)──▶│  BagSource ─▶ Replayer ─▶ Orchestrator        │
  dataset.yaml ───────▶│      │            │           │  drives core  │
  config.yaml ────────▶│      │            │           ▼               │
  calib.yaml ─────────▶│      │            │     MeridianPipeline (L0..L6) │
                       │      │            │           │               │
                       │      │            ▼           ▼               │
                       │      │      ReplayClock  TelemetrySink (Parquet)│
                       │      │                         │               │
                       │      └────────▶ RunManifest ◀──┘               │
                       │                       │                        │
                       │   TrajectoryWriter (TUM) ─▶ run_dir/           │
                       │   CloudWriter / MeshWriter ─▶ run_dir/         │
                       └───────────────────────────────────────────────┘
                                               │
                  ┌────────────────────────────┴───────────────────────────┐
                  │              meridian_eval_metrics (post-hoc)               │
                  │  evo wrapper (ATE/RPE) | C2C/C2M | timing                │
                  │              ▼ writes results.json + plots               │
                  └──────────────────────────────────────────────────────────┘
```

Two stages, deliberately decoupled by files on disk:

1. **Replay stage** (`meridian_eval`, built under `meridian_tools` per Spec 00 §9.3): drives the core, produces *artifacts* (estimated trajectory in TUM, optional exported cloud/mesh, the per-module telemetry Parquet, and the `RunManifest`). This stage links `meridian_core` only and is the part that must run the identical live call path. It constructs **one** `MeridianPipeline` — the identical object the ROS node owns.
2. **Metrics stage** (`meridian_eval_metrics`): pure post-processing of artifacts vs ground truth. It links nothing of the estimator; it is Python (evo) + a thin C++/Open3D tool for C2C/C2M. Decoupling means a metric bug never invalidates a (re-runnable) replay, and metrics can be recomputed or added retroactively against archived runs.

### 1.2 Directory layout

```
eval/
  datasets/<dataset>/<sequence>.yaml      # provenance, GT path, extrinsics ref, expected metrics
  configs/<profile>.yaml                  # estimator config profiles used by eval (the full meridian:: tree, Spec 00 §8.2)
  baselines/<sequence>/<profile>.json     # frozen regression baselines (committed)
  smoke/<clip>.mcap                        # tiny CI clips (LFS) — the only bags committed; full bags stay external (DATASET.md)
runs/<run_id>/                             # produced; gitignored
  manifest.json                            # RunManifest (Spec 00 §6)
  trajectory_odom.tum                      # front-end (L2) live odometry trajectory
  trajectory_world.tum                     # back-end (L3) optimized map-frame trajectory
  keyframes.tum                            # per-keyframe poses (join key = keyframe id)
  debug/frontend.parquet  backend.parquet  map.parquet  timing.parquet
  cloud.ply  mesh.ply                      # optional map artifacts (flagged)
  results.json                             # metrics stage output
  plots/*.pdf
```

`run_id` = `<sequence>__<profile>__<git_sha8>__<utc_timestamp>`; it is also stored inside `manifest.json` so a result is never separated from its conditions. The single production profile is the LIO estimator (`frontend.kind: lio`, `backend.kind: isam2`, `map.backend: nvblox`, Spec 00 §8.2). Any other profile in `eval/configs/` is a *diagnostic ablation* of the same `lio` tree (a `frontend.lio.*` knob change), never a different estimator.

### 1.3 The dataset descriptor (`eval/datasets/<dataset>/<sequence>.yaml`)

The harness never hard-codes dataset facts. `DATASET.md` decides *which* sequences and the GT/format conventions; this YAML is the machine-readable instance, one per sequence:

```yaml
dataset: newer_college_2021
sequence: quad-easy
bag:
  uri: "bags/newer-college/quad-easy"                # prepared ros2 bag (DATASET.md pipeline)
  sha256: "…"
  format: ros2bag                                    # ros2bag | mcap | ros1(converted via rosbags-convert)
topics:                                              # bag topic → core sensor (the L0 ISensorSource mapping)
  /os_cloud_node/points:       { sensor: lidar, type: lidar }   # the single LiDAR (OS0-128, 10 Hz, per-point t)
  /alphasense_driver_ros/imu:  { sensor: imu,   type: imu }     # 200 Hz
  /alphasense_driver_ros/cam0/compressed: { sensor: cam, type: image }  # visual stage disabled on this set
  # no gnss entry — the Newer College rig carries no GNSS
groundtruth:
  path: "gt/tum_asimu/gt-nc-quad-easy.csv"           # TUM, re-expressed in the IMU frame (DATASET.md, evo-readable)
  frame: "gt_world"
  body_frame: "imu"                                  # frame the GT pose refers to (matches estimation_frame)
calibration: "src/meridian_ros/config/newer-college-quad.yaml"  # per-collection; hashed into manifest (Spec 00 §6, Spec 01 §5)
characteristics: [outdoor, revisit_loop]             # free-form tags used to select acceptance gates
role: tuning                                          # tuning | holdout (quad-hard is holdout: milestone evals only)
expected:                                             # per-sequence acceptance (drives the §6.2 system test)
  ate_rmse_m_max: 0.30
  rpe_trans_pct_max: 1.5
  drift_pct_max: 0.8
```

There is exactly **one** `lidar` entry — Meridian is single-LiDAR (Spec 00 §2, Spec 01 §4.2). A bag carrying surplus streams maps only its chosen topics; the rest are simply not subscribed (Newer College's secondary Ouster-internal IMU `/os_cloud_node/imu` is an example, as would be the extra cloud topics of a multi-LiDAR rig). The `body_frame` field is essential: ATE compares like-with-like, and GT is in each dataset's own world frame at a specific body frame. The harness composes the estimator output (the estimation frame `imu_link`, Spec 01 §2.3) with the GT body frame before alignment (§3.2). `role: holdout` marks a sequence as **blind validation only**: the system runner accepts it solely for milestone evaluations (§6.2), and the regression tiers (§7.2, §8.3) must never select it — running a holdout routinely turns it into a tuning input.

---

## 2. The replay stage in detail

### 2.1 `BagSource` — boundary conversion (the L0 wrapper analogue, ROS-free)

`BagSource` is the harness's analogue of the ROS subscriber callbacks, and it is the only file in the eval lib allowed to know bag/serialization formats. It reads a bag and yields a *time-ordered* stream of typed core samples (`ImuSample`, `LidarScan`, `CameraFrame`, `GnssFix`; Spec 01 §4), tagged with the core `sensor` from the descriptor and pushed through the same `ISensorSource` callback contract (Spec 01 §7.1) the live driver uses.

```cpp
struct StampedSample {
  Timestamp t;                      // int64 ns, AFTER applying the same L0/PTP sync the node uses
  std::string sensor;               // resolved from the topic map ("lidar" | "imu" | "cam" | "gnss")
  std::variant<ImuSample, LidarScan, CameraFrame, GnssFix> data;
};

class BagSource {
public:
  explicit BagSource(const DatasetDescriptor&);
  // Strictly non-decreasing in t; ties broken by a stable (sensor, seq) order.
  std::optional<StampedSample> next();
  Timestamp first_stamp() const; Timestamp last_stamp() const;
};
```

Critical subtlety — **time sync must match the live path**: a hardware-synced rig takes the identity offset (Newer College: Ouster + alphasense share one synced clock, DATASET.md), an unsynced one takes the live L0 estimated offset. Either way `BagSource` applies *exactly the same* L0 sync transform the live `ISensorSource` applies, because time sync is core code (Spec 00 L0, `specs/02_sensors_timesync.md`), not a harness convenience. The harness additionally *records* the measured per-sensor offset as a diagnostic into `timing.parquet`. It must never silently "fix up" timestamps in a way the live node wouldn't.

### 2.2 Replay clock injection

Per Spec 00 §10.1, no core module reads the wall clock — time is always passed in as `meridian::Timestamp` (int64 ns, Spec 01 §2.1). The harness constructs a replay clock whose `now()` returns the timestamp of the sample currently being processed, and binds it where the node would bind the system clock. The custom clang-tidy check (`meridian-no-stdchrono`, enforced by the §9.4 CI gate of Spec 00) forbids `std::chrono::*_clock::now()` in `meridian_core`; this is what guarantees the replay clock is authoritative and that replay's time base is reproducible.

### 2.3 `Replayer` + `Orchestrator` — driving the core

The orchestrator owns the L0→L6 object graph **as a `MeridianPipeline`** (Spec 00 §11.3 — the same core application object the node drives) and reproduces the node's call order. In the live node this order is enforced by message arrival + the single-threaded replay executor; in replay it is enforced by `BagSource` ordering. The two MUST be the same call sequence.

```
loop:
  s = bag.next();  if (!s) break
  clock.set(s.t)
  pipeline.ingest(s.raw)                           # L0 stamping → L1 validity/aggregation (Spec 02 §8)
  # The pipeline forwards each IMU sample to frontend.ingest_imu_live (live state
  # only) and, once per sweep, hands the assembled bundle to the front-end —
  # the same call order the live node produces:
  on PreprocessedGroup g:                          # Spec 01 §7.3 — THE L1→L2 value
      frontend.ingest(g)                           # LIO: internal deskew + per-sweep GN registration
      traj_odom.append(clock.now(), frontend.live_state())   # live odom trace (Spec 01 §7.3)
      # keyframe handoff fires via the keyframe sink the orchestrator registered:
      on KeyframePacket kf:                          # Spec 01 §6 — the ONE L2→L3 value
        store.put(kf.id, kf.cloud_body, kf.image, kf.T_ref_body)   # KeyframeStore (Spec 01 §7.5)
        backend.add_keyframe(std::move(kf))          # builds ONE factor per kf.constraint_kind (§6.4 of Spec 01)
        sink.write_backend(...)
      loops = loopdetector.detect()                  # ILoopDetector (Spec 01 §7.6): SC++→STD/BTC→GICP→PCM
      for lc in loops: backend.add_loop_constraint(lc)
      upd = backend.optimize()                       # GraphUpdate: which keyframes moved (Spec 01 §7.4)
      map.integrate(kf_latest, upd.pose_of(kf_latest))         # nvblox TSDF+colour (GPU)
      if upd.has_moves(): map.apply_graph_update(upd, store)   # clear-and-rebuild moved region (GPU)
      sink.write_frontend(clock.now(), frontend.diagnostics()) # Spec 01 Appendix A FrontEndDiagnostics
finalize:
  traj_world = backend.corrected_trajectory()        # final optimized, map frame (Spec 01 §7.4)
  write traj_world.tum, keyframes.tum
  if export_map: cloud.ply = map.export_cloud(); mesh.ply = mesher.extract(map)  # Spec 01 §7.7 ColorMesh
  manifest.finalize(input_hashes, stamps); write manifest.json
```

The harness adds **no** factors of its own: it feeds `KeyframePacket`s and the back-end derives exactly one constraint per interval from `constraint_kind` — a single `BetweenFactor` on the normal `RelativeBetween` path, a prior on the `AbsolutePrior` anchor/chain-break, a GTSAM `CombinedImuFactor` *only* on the reserved `ImuPreintegration` kind (never emitted by the current front-end), per Spec 01 §6.4 and Spec 00 §6.3. Because the harness has no API to inject a factor, it *cannot* accidentally double-count information.

### 2.4 Two trajectory products, two purposes

- `trajectory_odom.tum` — the **front-end** live estimate, `IFrontEnd::live_state()` sampled at every scan (Spec 01 §7.3). This isolates L2 (the LIO estimator) quality without back-end help. Used to diagnose whether drift is a front-end or a back-end/loop problem.
- `trajectory_world.tum` — the **back-end** optimized estimate, `IBackEnd::corrected_trajectory()` at keyframe rate (Spec 01 §7.4), optionally densified by interpolating front-end relative motion between keyframes. This is the headline trajectory.

Reporting both is non-negotiable, and it is a *diagnostic* split, not a milestone split: if `trajectory_odom` is good but `trajectory_world` is worse, the back-end (or a bad loop) is hurting; if both are bad, the front-end is the cause. This mirrors the architecture's L2/L3 seam and gives the operator a direct causal read on a single, complete system.

### 2.5 Replay modes

```cpp
enum class ReplayMode {
  AsFastAsPossible,   // default for CI/metrics: ignore real time, max throughput
  RealTime,           // pace to bag wall-rate; used for the Jetson real-time budget check (§5.5)
  Stepped             // single-step for debugging; dumps state each step
};
```

`AsFastAsPossible` is the metrics mode. `RealTime` is *only* for measuring whether per-frame compute fits the sensor period on the Orin; it does not change any output, only the pacing, and timing is always measured from `ScopedTimer` (Spec 09) on compute, never from pacing.

---

## 3. Ground-truth handling

### 3.1 Ingestion

All GT enters as TUM (DATASET.md fixes evo/TUM as the convention). The metrics stage loads it via evo's TUM reader; the harness's only GT responsibility in the *replay* stage is to copy/normalize the path into the run dir so a run is self-contained. GT for the **GNSS prior fed *into* the estimator** (when a sequence exercises GNSS factors — none of the current Newer College set does; the rig has no GNSS) comes from the bag's GNSS topic, **never** from the evaluation GT — otherwise we leak the answer into the estimate and ATE becomes meaningless. The harness asserts the eval GT file path is not the input GNSS topic (§9.1).

### 3.2 Frame and body-frame reconciliation

The estimator reports poses of the estimation frame `imu_link` (Spec 01 §2.3). GT may report a different body frame (`<sequence>.yaml: groundtruth.body_frame`). Before any metric:

$$ \hat{\mathbf{T}}_{\text{gt-body}}(t) \;=\; \mathbf{T}^{\text{gt-body}}_{\,F_e}\cdot \mathbf{T}_{\text{est}}(t), $$

where $\mathbf{T}^{\text{gt-body}}_{\,F_e}$ is the static extrinsic from the calibration file (Spec 01 §5) mapping the estimation frame $F_e$ into the GT body frame. We transform the *estimate* into the GT body frame (not vice-versa) so the trajectory we judge is what the rest of the world consumes from us. The world-frame gauge difference between the estimator world and `gt_world` is then absorbed by Umeyama alignment (§4.4).

A constant body offset does **not** wash out under that alignment — Umeyama removes one rigid world transform, not a per-pose lever arm that rotates with the body — so skipping this composition biases ATE by up to the lever-arm length. Newer College is the worked instance: the dataset's native GT is in the *base* frame, ~7.6 cm from the estimation (alphasense-IMU) frame, so DATASET.md re-expresses GT offline into the IMU frame (`gt/tum_asimu/`); for those sequences `body_frame: imu` and the composition above is identity. ATE is scored against `tum_asimu`, never the base-frame files.

---

## 4. Trajectory accuracy metrics (evo)

We standardize on **evo** (Grupp, 2017) as the trajectory metric engine, called from the metrics stage. Rationale: it is the de-facto community standard, so our numbers are directly comparable to FAST-LIO2/Point-LIO/FAST-LIVO2 papers, and it removes any temptation to hand-roll alignment. We wrap it (`evo_ape`, `evo_rpe` Python API) and persist both the scalar summary and the per-pose error series into `results.json`.

### 4.1 Association

Estimate and GT are associated by timestamp with `max_diff = 0.01 s` (10 ms) nearest-neighbour matching (evo `--t_max_diff`). Unmatched estimate poses are dropped from error stats but counted (`assoc.matched`, `assoc.dropped`) — a high drop count is itself a failure signal (timestamp problem).

### 4.2 Absolute Trajectory Error (ATE)

After a rigid (SE(3)) alignment $(\mathbf{R}^\star,\mathbf{t}^\star)$ (and optional scale $s^\star$, §4.4), with associated pairs $\{(\hat{\mathbf{T}}_i,\,\mathbf{T}^{\text{gt}}_i)\}_{i=1}^{N}$, the per-pose absolute error pose is

$$ \mathbf{E}_i \;=\; \big(\mathbf{S}\,\hat{\mathbf{T}}_i\big)^{-1}\,\mathbf{T}^{\text{gt}}_i, \qquad \mathbf{S}=\begin{bmatrix}s^\star\mathbf{R}^\star & \mathbf{t}^\star\\ \mathbf{0} & 1\end{bmatrix}. $$

We report the translational ATE as RMSE (the headline number), plus mean/median/std/min/max:

$$ \text{ATE}_{\text{rmse}} \;=\; \sqrt{\frac{1}{N}\sum_{i=1}^{N}\big\lVert \operatorname{trans}(\mathbf{E}_i)\big\rVert_2^2}. $$

Rotational ATE uses the geodesic angle $\theta_i=\lVert\log(\operatorname{rot}(\mathbf{E}_i))^\vee\rVert$, reported as RMSE in degrees.

### 4.3 Relative Pose Error (RPE)

RPE measures local consistency (drift rate), insensitive to a single global misalignment. For a fixed delta $\Delta$ (we use both a metric delta of 1 m and a temporal/frame delta), with $\mathbf{Q}_{i}=\big(\mathbf{T}^{\text{gt}}_i{}^{-1}\mathbf{T}^{\text{gt}}_{i+\Delta}\big)$ and $\hat{\mathbf{Q}}_{i}=\big(\hat{\mathbf{T}}_i{}^{-1}\hat{\mathbf{T}}_{i+\Delta}\big)$:

$$ \mathbf{F}_i = \hat{\mathbf{Q}}_i^{-1}\,\mathbf{Q}_i, \qquad \text{RPE}_{\text{trans}}=\sqrt{\tfrac{1}{M}\sum_i \lVert\operatorname{trans}(\mathbf{F}_i)\rVert^2}. $$

We additionally report **drift as % of distance travelled** (KITTI-style): RPE translational error over $\{100,200,\dots,800\}$ m sub-paths, averaged — this is the number directly comparable to the LIO literature and is the one in `<sequence>.yaml: expected.drift_pct_max`. The rotational RPE is reported in deg/m.

> Worked example. On the ~570 s Newer College `park` sequence, suppose a ~480 m walked path with $\text{ATE}_{\text{rmse}}=0.21$ m and KITTI-drift $=0.55\%$. The drift figure ($0.0055 \times$ path) implies an *expected* end-to-end translational gap of $\sim2.6$ m before loop closure; if `trajectory_world` (post loop) ATE is 0.21 m, loop closure is recovering $\sim92\%$ of the open-loop drift — a healthy back-end. If instead post-loop ATE were *also* $\sim2.5$ m, loop closure is not firing or PCM (Spec 01 §7.6 `fitness`/PCM) is rejecting good loops. This is the causal reasoning the two-trajectory + drift% reporting is designed to enable.

### 4.4 Alignment policy (and the scale trap)

- **Default**: SE(3) Umeyama alignment, **no scale** (`evo --align`, not `--align --correct_scale`). LiDAR-inertial SLAM is metric; allowing scale would mask a real metric error.
- **Diagnostic only**: we *also* compute the Sim(3) scale factor $s^\star$ and report it. $|s^\star-1|>0.01$ is a red flag (IMU scale / extrinsic / unit bug) raised as a warning in `results.json`, never silently corrected.
- **Yaw-only / origin alignment** variants are computed for visualization but never for the headline gate.

### 4.5 `results.json` schema (metrics stage output)

```json
{
  "run_id": "...", "manifest_sha": "...",
  "trajectory": {
    "odom":  { "ate_rmse_m": 0.27, "ate_rot_rmse_deg": 1.1, "rpe_trans_pct": 0.81,
               "rpe_rot_deg_per_m": 0.012, "scale_est": 1.001, "n_poses": 11873,
               "assoc": {"matched": 11860, "dropped": 13} },
    "world": { "ate_rmse_m": 0.19, "ate_rot_rmse_deg": 0.7, "rpe_trans_pct": 0.42, "...": "..." }
  },
  "map": { "c2c_mean_m": 0.041, "c2c_rmse_m": 0.069, "completeness_pct": 93.4,
           "fscore_at_0p1": 0.88, "accuracy_m_p95": 0.12, "c2m_signed_std_m": 0.018, "...": "..." },
  "timing": { "frontend_ms_p50": 18.4, "frontend_ms_p99": 41.2, "rt_factor": 0.42, "...": "..." },
  "gates": [ {"name":"ate_rmse","value":0.19,"limit":0.30,"pass":true}, "..." ],
  "verdict": "PASS"
}
```

---

## 5. Map-quality metrics (cloud-to-cloud / cloud-to-mesh)

Trajectory accuracy is necessary but not sufficient: a low-ATE run can still produce a smeared, double-walled map (bad deskew, bad extrinsics, bad loop re-integration). We therefore evaluate the **map artifact** geometrically. The first-pass scope stops at a colourised mesh (Spec 00 §0), so map metrics target the exported point cloud (`cloud.ply`) and the **nvblox Marching-Cubes mesh** (`mesh.ply`, Spec 01 §7.7) — the single map backend; there is no alternative map representation to evaluate.

### 5.1 Reference geometry

Map ground truth is the relevant dataset's reference: a survey/TLS reference cloud or prior map where one exists (Newer College ships the TLS-derived `prior_map/maths-institute.ply` covering the `math-medium` collection; DATASET.md records provenance per sequence). When no metric reference map exists, map quality is reported only as **self-consistency** metrics (§5.4), clearly labelled, and never used as an absolute gate.

### 5.2 Cloud-to-Cloud (C2C) — accuracy and completeness

Let $\mathcal{M}$ be our exported map cloud and $\mathcal{G}$ the reference cloud, both expressed in the same frame after the *trajectory* SE(3) alignment from §4.4 is applied (we reuse the trajectory alignment so map and trajectory errors are in one consistent gauge), followed by a fine ICP refine (max 0.2 m correspondence, to remove residual bias not attributable to the map itself; the pre-ICP and post-ICP numbers are both reported).

- **Accuracy** (how wrong are the points we produced): for each $\mathbf{m}\in\mathcal{M}$, $d(\mathbf{m})=\min_{\mathbf{g}\in\mathcal{G}}\lVert\mathbf{m}-\mathbf{g}\rVert$. Report mean, RMSE, and the 95th percentile $d_{p95}$ (robust to a few fliers; the headline accuracy number).
- **Completeness** (how much of the scene did we cover): for each $\mathbf{g}\in\mathcal{G}$, $d'(\mathbf{g})=\min_{\mathbf{m}\in\mathcal{M}}\lVert\mathbf{g}-\mathbf{m}\rVert$; *completeness* = fraction of $\mathcal{G}$ with $d'(\mathbf{g})<\tau$ (default $\tau=0.10$ m).

Both directions are needed: accuracy alone rewards a sparse, conservative map; completeness alone rewards a dense, smeared one. We report the **F-score at $\tau$** (harmonic mean of accuracy-precision and completeness-recall) as the single map scalar, following the Tanks-and-Temples convention.

### 5.3 Cloud-to-Mesh (C2M)

For mesh quality we measure point-to-surface distance from $\mathcal{G}$ to the nvblox triangle mesh (signed, using vertex normals to detect inside/outside), surfacing **double-wall** artifacts (bimodal signed-distance histogram) that C2C can hide. We report the signed-distance std and the fraction of points with $|d|>0.05$ m. A clean clear-and-rebuild de-integration on loop closure (Spec 01 §7.5) should keep this unimodal.

### 5.4 Self-consistency (no reference map)

When no GT map exists we quantify internal sharpness directly from the retained per-keyframe clouds in the `KeyframeStore` (Spec 01 §7.5 — the same store L4 uses for loop re-integration):

- **Map entropy / planarity residual**: for overlapping keyframe-pair clouds at their optimized poses, fit local planes and report the RMS point-to-plane residual. A smeared map has high residual even with low ATE.
- **Re-integration stability**: after a loop closure triggers `IMapLayer::apply_graph_update` (Spec 01 §7.5), recompute the planarity residual in the affected region; it must *decrease or hold*. An increase means the clear-and-rebuild de-integration corrupted the region.

### 5.5 Throughput / real-time budget (Jetson Orin)

From `timing.parquet` (fed by `ScopedTimer`/`TimingRegistry`, Spec 09), measured **on the Orin target**:

- Per-stage latency percentiles (p50/p99): L1 preprocess, L2 per-sweep solve (`frontend.lio.ingest`, with the deskew/solve split in `FrontEndDiagnostics`), L3 `optimize`, L4 nvblox `integrate`, L5 detect, mesh extract (the stage keys are the Spec 00 §10.2 telemetry names).
- **Real-time factor** `rt_factor` = total compute time / bag duration (must be < 1 for the front-end path to keep up; the back-end optimize, GPU map integration, and meshing run on separate threads, Spec 00 §11.1, with their own budgets and may lag behind the front-end).
- Per-frame budget gate: the L2 front-end `step` p99 must be below the LiDAR period (100 ms at 10 Hz Ouster) for the production profile. Measured in `RealTime` mode (§2.5) and cross-checked against `AsFastAsPossible` ScopedTimer numbers (the two must agree to within scheduling noise). nvblox GPU integration and Marching-Cubes timing are measured the same way; there is no CPU map path to compare against.

### 5.6 The `Metric` interface (extensibility seam)

All metrics — trajectory, map, timing, and the deferred semantic/ESDF ones (Spec 00 §12) — implement one interface so adding a metric never touches the harness:

```cpp
struct MetricResult { std::string name; double value; std::optional<double> limit; bool gate; bool pass; nlohmann::json detail; };
class Metric {
public:
  virtual ~Metric() = default;
  virtual std::string name() const = 0;
  // Pure post-processing on run artifacts + descriptor; no estimator deps.
  virtual std::vector<MetricResult> evaluate(const RunArtifacts&, const DatasetDescriptor&) = 0;
};
```

`RunArtifacts` is a thin reader over a `run_dir/` (trajectories, clouds, mesh, telemetry Parquet, manifest). Registered metrics are run in sequence; their `MetricResult`s populate `results.json.gates`. The deferred ESDF and semantics metrics are *new `Metric` registrations* consuming the same nvblox TSDF and keyframe store — never edits to the existing gates.

---

## 6. Acceptance criteria — full system, gated by module-integration milestones

Meridian is **one** system (Spec 00 §13): the LIO estimator with nvblox mapping. There is no "v1 we ship then a v2 we swap." Acceptance is therefore organised around **bring-up / module-integration milestones** — the compile-and-integrate order of Spec 00 §13 for standing the *one* system up — and, once the full system is integrated, around **full-system accuracy and map quality** on the evaluation sequences.

Each milestone has (a) a *smoke* gate that must pass in CI on every commit (tiny clips, fast, deterministic) and (b) a *system* gate run on the full sequences before the milestone is declared met. Numeric limits live in `<sequence>.yaml: expected` and `eval/baselines/`; the table gives the intent and target sequence. Limits below are initial targets, to be ratified against first measured baselines (§7.3) — they are placeholders for the *shape* of the gate, not hand-tuned truth.

The milestone order mirrors Spec 00 §13 (a convenience for integration), not a roadmap of shippable variants; every interface (Spec 01) is in place from the first milestone.

Scope note on the current benchmark set (DATASET.md: Newer College 2021): it exercises the **LiDAR-inertial core** — the estimator fuses no camera or GNSS (both ride through unfused) — so the visual/GNSS tags below are dormant until those stages exist, and published ATE numbers are LIO-only. `quad-hard` is the **holdout** (`role: holdout`, §1.3): blind milestone validation only, never a tuning or regression input.

| Milestone (Spec 00 §13 step) | What is being integrated into the one system | Primary eval sequence(s) (DATASET.md) | Acceptance metric (gate) |
|---|---|---|---|
| **M0 — contracts + skeleton** (steps 1) | Cross-cutting types + the full interface set compile; no-op module bodies; harness runs the empty pipeline end-to-end | smoke clip | Harness runs end-to-end; `manifest.json` valid. No accuracy gate. |
| **M1 — sensing path** (steps 2) | L0/L1: real measurements onto the monotonic timeline, time-synced, preprocessed | NC `quad-easy` (sensing only) | Per-sensor sync offset within sequence spec; preprocessed scan rate/density sane; no dropped-sample storm. |
| **M2 — LIO front-end** (step 3) | L2 `lio`: internal screw deskew + voxel-hash map + GN ICP with the IMU prior, one solve per sweep. | NC `quad-easy`, `math-medium` | `trajectory_odom` ATE RMSE ≤ 0.30 m; KITTI-drift ≤ 0.8%; front-end p99 < LiDAR period; rt_factor < 0.7. Synthetic GT-tracking, Jacobian-parity, and covariance-chain checks green (§8.1–§8.2). |
| **M3 — nvblox map** (step 4) | L4: GPU TSDF+colour+Marching-Cubes mesh consuming keyframe clouds; KeyframeStore | NC `math-medium` (+ `prior_map/maths-institute.ply` as reference) | Mesh extracts without holes in covered region; C2C F-score@0.1 ≥ 0.85 vs ref where available; C2M signed-distance unimodal. |
| **M4 — back-end** (step 5) | L3: iSAM2 graph consuming `KeyframePacket`s; `GraphUpdate` feedback to L2/L4; GNC robust kernels; online extrinsic refinement | NC `quad-easy` | `trajectory_world` ATE ≤ `trajectory_odom` ATE (back-end helps, never hurts); ATE ≤ 0.20 m on quad-easy; online extrinsic converges to within 1° / 2 cm of the calib prior. |
| **M5 — loop closure** (step 6) | L5: SC++→STD/BTC→GICP→PCM feeding loop constraints to L3; closing the correction loop into L4's clear-and-rebuild | NC revisit sequences (`quad-easy` quad circuits, `park`) | Loop precision = 1.0 (zero false loops via PCM); ATE drops or holds after loop; re-integration stability (§5.4) holds; no ATE spike at loop time. |
| **M6 — full-system robustness & global** (step 7 + system) | The complete system under degeneracy, outliers, GNSS, and global consistency | Fault-injected variants of the tuning set (§6.3); NC `quad-hard` **HOLDOUT** (blind, milestone-only) | On (injected) degeneracy: no divergence (ATE finite, < 2× nominal) and the unobservable axis's `observability.score` drops (Spec 01 §3.4); GNSS factor on synthetic injection until a GNSS-carrying sequence is added (target: global ENU ≤ 0.5 m where the fix is good, no jump > 0.3 m at re-acquisition); GNC drives injected outliers' weight → 0 (§6.3); quad-hard ATE within its `expected` bounds without any retuning. |

There is no milestone for "swap the front-end." The LIO estimator is THE front-end from M2 onward; its cross-check is the direct GT-tracking test (§8.2 — the retired oracles have no role here). Multi-LiDAR, ESDF, and semantics are deferred (Spec 00 §12) and have no milestone here.

### 6.1 Mapping of dataset *characteristics* → which gate applies

The harness selects extra gates from `<sequence>.yaml: characteristics` (so we do not, e.g., assert a GNSS gate on a GNSS-less sequence):

- `degeneracy_corridor` ⇒ assert at least one `observability.score[axis]` drops below threshold during the degenerate span AND ATE finite (no divergence). (No current sequence carries the tag — the NC `stairs` extra is the candidate once imported; until then the gate runs on injected degeneracy, §6.3.)
- `low_illumination` ⇒ no estimator gate today (the front-end is LiDAR-inertial; the camera is passthrough). Tag retained for the future visual stage's gates. (Dormant.)
- `gnss_denied_transition` ⇒ continuity gate at the GNSS-denied transition (M6). (Tag retained for a future GNSS-carrying sequence; none in the current set.)
- `fast_motion` ⇒ the deskew and prior must be measurably working: assert `frontend/lio/accel_var` rose with the motion, the gravity regulariser engaged (`frontend/lio/beta` > its floor), and no `frontend/lio/reject` streak appeared. (Candidate carrier: `quad-hard` — holdout, so this fires at milestone evals only.)

### 6.2 The system test runner

`meridian_eval_system` iterates the sequences for a milestone, runs replay+metrics, compares against `<sequence>.yaml: expected` and `eval/baselines/`, and emits a single PASS/FAIL plus a Markdown report (per-sequence rows, plots embedded). This is what a developer runs before declaring a milestone met and what the nightly job runs.

### 6.3 Fault-injection tests (M6 robustness)

Robustness gates require *controlled* faults, injected at the harness boundary (not in core), so the same core code is exercised:

- **Outlier injection**: corrupt a configurable fraction of LiDAR returns (range spikes) or add spurious loop candidates with wrong relative poses; assert GNC (Spec 01 robust kernels / `fitness`) drives their effective weight → 0 and ATE stays within tolerance.
- **Dropout**: drop all input for longer than `lio.max_gap_s` (assert the gap/reseed fallback fires, Spec 00 §7.4 → §6.4 below), or drop single sweeps (assert the bridge integrates across and the chain stays `RelativeBetween`).
- **Time-jitter**: perturb per-sensor timestamps within a bound; assert the L0 sync diagnostic flags it and ATE degrades gracefully.

These are implemented as `BagSource` decorators (`OutlierInjector`, `Dropper`, `Jitterer`) so they compose and never alter core code.

### 6.4 Restart/fallback path coverage

The gap/reseed fallback (Spec 00 §7.4, Spec 01 §6.4 chain break) is a code path that must be tested *deliberately* because it rarely triggers naturally on clean public data. A dedicated integration test forces a reseed (via the dropout injector plus a registration-failure trigger) and asserts: (a) the state bridges the hole on constant velocity and the reseed arms (`frontend/lio/gap`), (b) on the post-gap failure the map clears and the *next* `KeyframePacket` carries `constraint_kind == AbsolutePrior` with the inflated covariance, and **no** `BetweenFactor` spans the hole (mutual exclusivity, Spec 01 §6.4), (c) the relative chain resumes off the anchor with no discontinuity beyond tolerance, (d) the map re-integrates the post-reseed region cleanly.

---

## 7. Regression baselines

### 7.1 What a baseline is

A **baseline** is a frozen `results.json` (and the `manifest.json` that produced it) for a `(sequence, profile)` pair, committed under `eval/baselines/<sequence>/<profile>.json`. It is the authoritative "this is how good we were" record. The `RunManifest` (Spec 00 §6: git SHA, config hash, calib hash, dataset id, input content hash) is what makes a baseline meaningful — a metric number without its conditions is noise. The tracked profile is the `lio` production system; the baseline records exactly that config. A diagnostic ablation profile may carry its own baseline as a reference, never as the headline.

### 7.2 The regression check

On every PR (CI) and nightly (full), for each tracked `(sequence, profile)`. The holdout sequence (`role: holdout`, §1.3) is **never** in the tracked set — comparing against it per-PR would make it a tuning input; it is exercised only by the §6.2 milestone runs:

```
new = run_replay_and_metrics(sequence, profile)
base = load_baseline(sequence, profile)
for metric in tracked_metrics:
    if new[metric] worse than base[metric] by more than tolerance[metric]:
        FAIL("regression in {metric}: {base}->{new}")
    if new[metric] better than base[metric] by more than improve_threshold:
        WARN("improvement — consider rebaselining")
assert new.manifest.calib_hash == base.manifest.calib_hash or flagged   # calib change must be intentional
```

Tolerances (initial): ATE RMSE +5% or +0.02 m (whichever larger), drift% +10%, timing p99 +15%, C2C F-score −0.02 absolute. A regression *fails the build*; an improvement *warns* (forces a human to consciously rebaseline so we never silently lower the bar).

### 7.3 Establishing and updating baselines

- First baseline per `(sequence, profile)` is set the moment a milestone gate is first met, by an explicit `meridian_baseline --bless <run_id>` command that copies the run's `results.json`+`manifest.json` into `eval/baselines/`. Blessing is a committed, reviewed change — never automatic.
- Re-baselining (after an intended improvement or a deliberate config/calib change) is likewise an explicit, reviewed commit, with the WARN from §7.2 as the trigger. The PR description must state *why* the baseline moved (this is the audit trail the manifest enables).
- Cross-machine note: ATE/map metrics are hardware-independent, so accuracy baselines are portable; *timing* baselines are host-tagged (`manifest.host_info`) and only compared within the same host class (the Jetson Orin target), else compared as a soft warn.

---

## 8. Test pyramid (unit → integration → regression → system)

The milestone gates above are the *system* tier. Beneath them:

### 8.1 Unit tests (fast, per-module, no bag)

Per core module, table-driven tests on each interface contract (Spec 01) with synthetic inputs and known answers. The clock-free, side-effect-explicit interfaces (Spec 01) are exactly what makes these trivial — each module is a near-pure function of its inputs.

- **Math kernels**: SO(3)/SE(3) exp/log round-trips, box-plus/box-minus on `Pose`/`NavState`, Jacobian checks by finite difference (each analytic Jacobian in L2/L3 has an `EXPECT_NEAR(analytic, numeric, 1e-6)` test — non-negotiable given the rigour of the derivations in `specs/04_frontend_estimation.md`), including the GN registration Jacobian and the deskew screw.
- **LIO unit suite**: the voxel map's neighbour search against a brute-force reference and its order-independence/determinism properties; deskew against the analytic constant-screw warp; static-init recovery of gravity/biases from synthetic standstill data (and refusal while moving); IMU straddle-sample dedup; β gravity-regulariser engagement; the **Monte-Carlo covariance chain** (noisy synthetic scans → per-scan Σ → composed relative Σ → empirical scatter of the relative-pose error within χ² bounds) plus the rotation-first reorder round-trip regression.
- **Data association / voxel map**: insertion/query correctness, plane fit on synthetic planes (`query_plane`, Spec 01 §7.5).
- **KeyframePacket round-trip**: serialize/deserialize equality; `kinematics_included` flag honoured (Spec 01 §6).
- **iSAM2 factor builder**: given two `KeyframePacket`s with `RelativeBetween`, exactly one `BetweenFactor` is produced; on a forced `ImuPreintegration` restart, exactly one `CombinedImuFactor` and zero between-factors (Spec 01 §6.4). These are asserted **white-box** by binding a recording `IntrospectionHooks` consumer (Spec 09 §2.4) to read the live `gtsam::NonlinearFactorGraph`/`ISAM2ResultExt` directly — factor count by type, the gauge-anchor presence, the relinearised set — with no message schema between the test and the real object. The hook is `const`-ref, producer-thread-synchronous, and unbound (`NullHooks`) everywhere but these tests.

### 8.2 Integration tests (multi-module, tiny synthetic or smoke clip)

- **L0→L2 on a synthetic bag**: a programmatically generated bag (known trajectory, simulated planar-room single LiDAR + IMU) where the *exact* answer is known; assert ATE ≈ 0 (within numerical tolerance). This catches frame/convention bugs that real-data ATE (with its own GT error) cannot.
- **Direct GT-tracking check (replaced the retired differential oracles)**: on synthetic box-room worlds (raycast scans + IMU derived from the ground-truth trajectory) and on smoke clips carrying a reference trajectory, drive the production `lio` front-end and assert `live_state()` tracks the reference pose within an absolute position/orientation bound, and that the packet stream has the contractual shape (first keyframe `AbsolutePrior`, chained ids, PD covariances). A direct absolute-error bound is stronger and more interpretable than a "two estimators agree" comparison.
- **Loop closure + re-integration**: a synthetic loop where the true loop constraint is known; assert PCM accepts it, the back-end corrects, and `apply_graph_update` (Spec 01 §7.5) reduces the planarity residual (§5.4).
- **Restart/fallback** (§6.4).
- **End-to-end smoke**: the M0 gate — full L0→L6 on a 5–15 s clip, asserting it runs and produces a valid manifest.

### 8.3 Tiers, triggers, runtime budget

| Tier | When | Data | Budget |
|------|------|------|--------|
| Unit | every commit (pre-push hook + CI) | synthetic | < 60 s total |
| Integration | every commit (CI) | synthetic + smoke clips (LFS) | < 5 min total |
| Regression | every PR (CI) | smoke clips + 1–2 short full seqs | < 20 min |
| System | nightly + pre-milestone-gate | full sequences (fetched by hash) | hours (parallel over sequences) |

The split honours DATASET.md: only tiny clips live in the repo (LFS); the full benchmark bags are not committed — they are fetched and prepared per DATASET.md (content-hashed in the descriptor, §1.3) and are therefore *not* in the per-commit path. The holdout (`role: holdout`) is excluded from every tier here including nightly; it runs only in pre-milestone system evals (§6.2, §7.2).

---

## 9. Debug hooks (this component's introspection)

Per Spec 00 §10 every module already emits structured telemetry through the `TelemetrySink`; the harness binds a recording sink that *consumes* it into Parquet and adds its own evaluation-specific introspection. Mandatory eval debug outputs:

- **Per-frame error trace**: after metrics, an `error_series.parquet` joining each estimate pose to its matched GT pose and per-pose ATE/RPE — lets a developer scrub the trajectory and see *where* error accumulates (the live node can overlay the same series against a GT topic when present).
- **Alignment report**: the Umeyama $\mathbf{R}^\star,\mathbf{t}^\star,s^\star$, the residual after alignment, and the scale-warning flag (§4.4).
- **Association histogram**: distribution of estimate↔GT time offsets (catches sync drift).
- **Gate ledger**: every gate's name/value/limit/pass into `results.json.gates`, mirrored to stdout as a table so a CI log is self-explanatory.
- **Per-stage timing waterfall**: `timing.parquet` rolled up per stage (uses Spec 09 `TimingRegistry`; stage keys are the Spec 00 §10.2 telemetry names).
- **Map-error colouring**: when a reference map exists, export `cloud_error.ply` with per-point C2C distance as a colour scalar — direct visual of *where* the map is wrong, viewable in CloudCompare/rviz.

### 9.1 Failure modes the harness must detect and name

The harness's job is to fail *informatively*. Named, asserted failure modes:

| Failure | Symptom the harness detects | Where flagged |
|---------|------------------------------|---------------|
| Sync/timestamp bug | high `assoc.dropped`; association-offset histogram skewed | §4.1, §9 |
| Scale/unit/extrinsic bug | $|s^\star-1|>0.01$ | §4.4 |
| Frame/body-frame bug | huge ATE on synthetic-known integration test (8.2) but trajectory "looks" plausible | §8.2 |
| Back-end hurts | `trajectory_world` ATE > `trajectory_odom` ATE | §2.4, M4 gate |
| Bad loop accepted | PCM precision < 1.0 on labelled loops; ATE spike at loop time | §6 (M5), §6.3 |
| Smeared map / double wall | bimodal C2M signed-distance; high planarity residual at low ATE | §5.3–§5.4 |
| Re-integration corruption | planarity residual *increases* after loop clear-and-rebuild | §5.4 |
| Real-time miss (Orin) | front-end `step` p99 > sensor period; rt_factor ≥ 1 | §5.5 |
| Restart path broken | forced-restart test: discontinuity, or wrong factor emitted | §6.4 |
| Init/recovery paths not exercised | no `frontend/lio/init_done` on a run, or the §6.4 reseed test not firing its events | §6.1, §6.4 |
| Front-end estimator bug | direct GT-tracking bound exceeded on synthetic/smoke data; or an analytic Jacobian diverges from its numeric check | §8.1–§8.2 |
| GT leakage | GNSS GT used as both input prior and eval reference | §3.1 (asserted: eval GT path ≠ input GNSS topic) |

---

## 10. Tooling, dependencies, CLI

- **evo** (Python) for ATE/RPE — pinned version, wrapped by `meridian_eval_metrics`.
- **Open3D** for C2C/C2M (KD-tree NN, ICP refine, mesh signed-distance) — single choice; the `Metric` interface (§5.6) hides it.
- **Arrow/Parquet** for telemetry logs (Spec 09 sink) — columnar, fast to query in pandas/duckdb.
- **GoogleTest** for unit/integration; **CTest** labels (`unit`, `integration`, `regression`, `system`) drive the tiers (§8.3).
- Bag IO: `rosbag2` reader (primary — the prepared benchmark bags are ros2/sqlite3), `mcap` C++ reader for the smoke clips; ROS 1 originals are converted offline with `rosbags-convert` (DATASET.md).

CLI surface (all ROS-free, built under `meridian_tools`):
```
meridian_replay   --dataset <d> --sequence <s> --profile <p> [--mode afap|rt|step] [--export-map] -> run_dir/
meridian_metrics  --run <run_dir> [--ref-map <ply>]                                              -> results.json
meridian_system   --milestone <Mn>                     # run all milestone sequences, gate, report
meridian_baseline --bless <run_id>                     # promote a run to baseline (reviewed commit)
meridian_regress  --pr                                 # compare tracked runs vs baselines
```

---

## 11. Open dependencies on other specs

- **Spec 09** (`specs/09_debug_introspection.md`, logging/timing/provenance): `ScopedTimer`, `TimingRegistry`, the `ParquetTelemetrySink` and its fixed columnar schema (Spec 09 §2.5) that this spec's `run_dir/debug/*.parquet` artifacts and metrics stage read, the `IntrospectionHooks` slot (Spec 09 §2.4) the white-box back-end integration tests (§8.1–§8.2) bind a recording consumer to, the `Config::dump(run_dir)` resolved-config snapshot and the canonical `RunManifest` serializer (Spec 09 §9.6) that anchors every baseline. This spec *consumes* them; if their field names change, §4.5/§9 schemas follow.
- **Spec 11** (`specs/11_build_system_libraries.md`, build & libraries): pins evo, Open3D, mcap, GoogleTest and the eval CLI targets; it is also where any *considered-then-rejected* alternative (e.g. PCL vs Open3D for C2C, VDBFusion as a map backend) is recorded — this spec names only the single chosen tool per job.
- **Spec 08** (`specs/08_calibration.md`, calibration): the static extrinsics file hashed into the manifest and used in §3.2 body-frame reconciliation and the online-extrinsic convergence gate (M4).
- **Spec 00 / 01** are authoritative for all core types used here; this spec adds **no** new core types — only harness-side types (`StampedSample`, `DatasetDescriptor`, `Metric`, `RunArtifacts`, the injector decorators), which live in `meridian_eval` / `meridian_tools`, never in `meridian_core`.
