# Meridian real-time debugging runbook

This is the operator workflow for live sensors and test bags. It intentionally contains no v1 topic/key names or expected dataset numbers. Fill executable commands and concrete message fields as each implementation slice lands. Architecture and required reports are defined in [SYSTEM_SPECS.md](SYSTEM_SPECS.md).

Current status: runbook skeleton; no v2 estimator executable exists yet.

## 1. Preserve evidence first

Before restarting or retuning, capture:

- run/build/container/config/calibration/model identifiers;
- lifecycle and capability states;
- the last typed health/runtime reports;
- sensor publisher/topic/QoS information;
- queue depth/bytes/oldest age/drop reasons;
- CPU/GPU/RAM/VRAM/power/temperature and throttling;
- the triggered flight-recorder bundle when available.

Do not diagnose from ATE or RViz alone. Do not rebuild symlink-installed code while a run is active.

## 2. Preflight

- [ ] Correct container, repository revision, resolved deployment, and rig artifact.
- [ ] No stale Meridian nodes or duplicate TF authorities.
- [ ] System clock/PTP/driver status matches the deployment assumption.
- [ ] Sensor publishers exist at expected rates and report compatible QoS.
- [ ] LiDAR contains mandatory valid per-point time; image layout and calibration dimensions agree.
- [ ] Host and Jetson are not already CPU/GPU/memory/thermal constrained.
- [ ] Bag replay clock mode is consistent for every node.
- [ ] Output directory has space and is not on the local real-time thread path.

Record preflight as part of the run manifest.

## 3. Intake waterfall

For each source, reconcile counts in order:

```text
publisher/record count
  → ROS callback count and inferred upstream sequence/stamp gaps
  → wire-schema accepted/rejected
  → clock/time-model accepted/reordered/too-old
  → core ingress queue accepted/dropped/coalesced/expired
  → frontend consumed
  → graph factor accepted/deferred/rejected
  → local/global commit manifest
```

The first mismatch owns the fault. Queue depth cannot repair best-effort transport loss; reliable QoS cannot repair a blocked callback or stale-work policy. Always inspect count, bytes, and oldest age together.

## 4. Time and calibration triage

Check before algorithm residuals:

- raw device, fusion, and host-arrival stamp monotonicity;
- source/clock/calibration epoch changes;
- IMU sample gaps and exact support around every knot/sweep;
- LiDAR header convention, raw acquisition support, and point-offset extrema;
- camera exposure midpoint/offset and image/calibration revision;
- GNSS receiver time/status and antenna lever revision;
- transform direction and sole TF authority.

A clean-looking point cloud with wrong time or extrinsics is not valid evidence. Do not tune robust thresholds to compensate.

## 5. Local estimator waterfall

### 5.1 IMU backbone

Inspect initialization tests, sample count/support, mean/std/saturation, gravity/bias seeds, preintegration `dt`, covariance/rank, boundary interpolation, gap inflation, reintegration count, and propagation-anchor revision.

Typical actions:

- initialization remains pending: fix motion/clock/noise/rig issue or supply a reviewed seed;
- unsupported gap: repair transport/scheduling; do not add a pseudo-factor;
- bias/reintegration storm: inspect calibration, vibration, time alignment, and modality residuals before changing thresholds.

### 5.2 Visual lane

Inspect raw/valid images, blur/saturation, tracked/new/recovered counts, grid coverage, parallax, keyframe causes, descriptor/RANSAC inliers, reprojection quantiles/NIS, triangulation rejects, factor dispositions, and visual FSM.

Use gated overlays: tracks by age/status, cell occupancy, rejected geometry, and residual vectors. A high feature count with poor spatial/parallax support is not healthy tracking.

### 5.3 LiDAR lane

Inspect raw/valid/deskewed/downsampled counts, original sweep support, pose-table and bias revisions, re-deskew displacement, target/source revisions, overlap/inliers, residual quantiles, robust weights, eigenvalues/rank/classifier, source-exclusion audit, and LiDAR FSM.

Use gated debug clouds for raw versus deskewed, associations, target provenance, residual coloring, and supported/null directions. Rank loss in a corridor may be correct behavior; stale deskew or source leakage is not.

### 5.4 Local graph

Inspect knots/landmarks/factors by type, optional-knot suppression, costs/NIS, state deltas, relinearization, marginal ordering/rank, solve/rebuild/commit times, journal/replay, checkpoint age, and current capability. A solver recovery that adds hidden strong priors is a correctness failure.

## 6. Sparse/global waterfall

### 6.1 Submaps

Check core/context intervals, split/seal cause, finalization lag, final deskew, proxy/keyframe bytes, condensed-edge rank, manifest uniqueness, checksum, outbox/journal state, and exact content revision.

### 6.2 Loop closure

Follow one proposal end to end:

```text
descriptor/query
  → candidate/exclusion/voting
  → independent visual or LiDAR seed
  → geometric matches/support/residual/rank
  → LoopMeasurement revisions/manifest
  → PCM pending/clique decision
  → GNC weight in full shadow graph
  → validation/journal/CAS
  → durable GlobalCommit or typed rejection
```

Never accept a loop because RViz looks aligned. Capture both endpoint blobs, seed/final transform, match support, information basis, PCM neighborhood, GNC weights, and before/after graph reports.

### 6.3 GNSS

Inspect actual receiver solution type, covariance/status/correction age, ENU origin source, alignment baseline/excitation/yaw uncertainty, interpolation/lever covariance, spacing/information cap, innovation/NIS, quarantine, FSM cause, and reacquisition shadow result. GNSS may correct `map -> odom`; any discontinuity in local `odom -> base_link` is an ownership bug.

### 6.4 Global commit/TF

Check parent/current revision, journal durability, changed anchors, factor dispositions, trusted-factor residual change, robust weights, reference submap/odom epoch, `T_map_odom` step, and TF authority. Readers must never see mixed revisions.

## 7. Runtime and deadline triage

Use queue wait + compute + publish/serialization to account for end-to-end age. Check p50/p95/p99/max, not only average. Correlate misses with CPU affinity, GPU stream contention, memory allocation/copies/synchronization, I/O, debug publication, global verification, dense work, power mode, temperature, and throttling.

The degradation governor should shed debug/global/dense/optional work in its declared order. If it silently changes estimator math or consumes stale data, stop and file a correctness bug.

## 8. Symptom routing

| Symptom | First checks | Do not do |
|---|---|---|
| smeared LiDAR sweep | point time, raw support, IMU brackets, extrinsic, deskew revision | tune ICP gates first |
| stable scan but drifting pose | modality ranks/residuals, bias/time calibration, target provenance | trust raw Hessian covariance |
| visual jumps | time/exposure, coverage/parallax, track identity, recovery matching | reconnect IDs by descriptor alone |
| corridor uncertainty grows | LiDAR nullspace plus visual/IMU health | force six-DoF LiDAR precision |
| local pose jumps on loop/GNSS | TF/state ownership and odom epoch | smooth feedback into local graph |
| false global correction | verifier support/rank, manifests, PCM/GNC/validation | lower gates until it “works” |
| increasing output age | intake waterfall, queue oldest age, stage p99 | increase every queue depth |
| non-deterministic replay | ordering, seeds, unordered iteration, GPU reduction, async commit | average multiple unexplained runs |
| memory growth | active caps, blob/outbox retention, snapshots, DDS histories | treat 24 h leak as normal caching |

## 9. Forensic bundle checklist

- [ ] Trigger and typed cause.
- [ ] Resolved run manifest and all revisions/hashes.
- [ ] Bounded raw/domain references around the event.
- [ ] Relevant frontend and solver reports.
- [ ] Graph/submap snapshots before and after.
- [ ] Queue/resource timeline.
- [ ] Deterministic replay command.
- [ ] Human note describing observed versus expected capability.
