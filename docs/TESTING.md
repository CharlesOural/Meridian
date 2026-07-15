# Meridian testing and benchmark runbook

This is the operational test skeleton for Meridian v2. It intentionally removes all v1 executables, configurations, topics, and reference numbers. The binding architecture, factors, failure behavior, and implementation slices are in [SYSTEM_SPECS.md](SYSTEM_SPECS.md).

Current status: pre-Slice-0. Only workspace infrastructure can be built; fill commands and acceptance thresholds as real v2 targets land.

## 1. Test principles

- Test contracts and failure semantics before end-to-end accuracy.
- Compare analytic Jacobians against numerical derivatives.
- Compare incremental/fixed-lag results against small batch or analytic oracles.
- Use one domain implementation for live and replay.
- Make ordering, random seeds, threads, and GPU reductions deterministic in replay mode.
- Validate input counts/time/calibration before scoring trajectories.
- Tune only on development data; preserve independent holdouts.
- Use full sequences and fault/endurance campaigns for promotion.
- A run must record capability/availability and catastrophic failures, not only ATE.

## 2. Test pyramid

### 2.1 Static/build gates

- format, warnings-as-errors, clang-tidy;
- package dependency direction and cycle check;
- no ROS includes/dependencies in core/local/global/dense/store;
- no GTSAM/CUDA/nvblox type in public headers;
- public-header self-containment;
- CPU sanitizers and external-code license/source scan;
- clean build without an old install overlay.

### 2.2 Unit/property tests

- IDs, revisions, time intervals, source epochs, serialization compatibility;
- SE(3), frame direction, perturbation/tangent conversion;
- covariance/information PSD/rank/units/frame guards;
- queues, manifests, duplicate and stale revision behavior;
- IMU integration/bias/covariance and exact segment ownership;
- camera/LiDAR/GNSS factor Jacobians;
- submap seals, checksums, idempotency and half-open boundary ownership.

### 2.3 Component tests

- sensor wire conversions with malformed and golden messages;
- IMU initialization/propagation/gaps;
- visual tracking/recovery/triangulation/factors;
- LiDAR deskew/registration/degeneracy;
- local graph/marginalization/rebuild/rollback;
- sparse submap finalization/condensation/outbox;
- visual/LiDAR retrieval and verification;
- PCM/GNC transactions and global revisions;
- GNSS datum/alignment/factor/FSM/reacquisition;
- lifecycle/QoS/TF/restart and missing-revision recovery;
- later dense nvblox analytic GPU behavior.

### 2.4 System tests

- deterministic full-bag replay;
- live-driver soak with intake reconciliation;
- dropout/delay/reorder/duplicate/time-jump/corruption injection;
- darkness/blur/repeated texture, corridor/floor/open field, vegetation/dynamics;
- GNSS multipath/outage/return;
- false-loop storms and mutually consistent false clusters;
- local/global/store/dense process kill/restart;
- CPU/GPU contention, thermal/power modes, storage backpressure;
- 24-hour bounded memory and journal/checkpoint recovery.

## 3. Benchmark manifest

Every bag/dataset has a committed manifest; raw bags remain unversioned. A manifest contains:

```text
id and role: development | regression | holdout | adversarial
source/license/citation
bag/dataset checksums and time interval
sensor models/rates/resolutions
topic mapping and publisher QoS
clock/stamp/per-point-time conventions
calibration artifact + trust/provenance
ground-truth source, frame, uncertainty and scoring transform
known gaps/dynamics/environment labels
fault-injection profile and seeds
permitted decisions/tunables
```

### Dataset registry

| Manifest | Sensors | Environment/failures | Ground truth | Role | Calibration trust | Status |
|---|---|---|---|---|---|---|
| pending | — | — | — | — | — | no v2 manifests committed |

Newer College and existing Barracuda recordings may be migrated into manifests after their topic/time/calibration facts are audited. Old configuration values and measured v1 outcomes are not v2 baselines.

## 4. Run manifest and artifacts

Every run records:

- run ID, UTC start, repository/build/container IDs;
- resolved config/calibration/model hashes and random seeds;
- dataset manifest/checksum and scoring interval;
- machine/Jetson/JetPack/CUDA, power mode, clocks, affinity, ambient/concurrent load;
- source/QoS discovery and intake waterfall;
- capability/FSM timeline and local/global revision history;
- trajectory and covariance outputs;
- typed reports/events/stage timing/resource trace;
- loop/GNSS dispositions and forensic bundles;
- scoring tool/version/alignment policy;
- exit status and test result summary.

Artifacts live under a run-ID directory and are immutable after scoring.

## 5. Standard execution skeleton

Commands will be implemented in `meridian_tools`; this is the required shape:

```bash
# Clean build of affected packages
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to <package>
colcon test --packages-select <package> --event-handlers console_cohesion+
colcon test-result --verbose

# Deterministic domain replay (future command)
meridian_replay --manifest <manifest.yaml> --out <run_dir> --deterministic

# Fault campaign (future command)
meridian_replay --manifest <manifest.yaml> --faults <faults.yaml> --out <run_dir>

# Scoring
python3 tools/eval_ate.py <ground_truth.tum> <run_dir/local_or_global.tum>
```

The runner must fail if a stale node exists, manifests/hashes do not match, expected input loss is unexplained, outputs are incomplete, or child processes survive teardown.

## 6. Accuracy and consistency metrics

Record as applicable:

- local and global ATE/RPE with explicit alignment and frame;
- translational/rotational drift per time/distance;
- pose/velocity/bias NEES and per-factor NIS/coverage;
- availability by capability state and recovery latency;
- deskew error and registration residual/rank calibration;
- visual track survival/coverage/recovery and loop recall/precision;
- GNSS innovation/alignment/reacquisition consistency;
- false global commits—catastrophic count must be zero in the release adversarial suite;
- map/submap consistency later.

Never rank systems on one aggregate number. Report sequence-level distributions and failures.

## 7. Real-time/resource metrics

On target hardware record p50/p95/p99/max for ingress age, visual, deskew, LiDAR match/factor, local solve/marginalize/commit, seal, retrieval, verification, global transaction, publication, and end-to-end output age. Also record deadline misses, queue count/bytes/age, CPU, GPU, RAM, VRAM, allocations/copies/synchronization, storage I/O, power, temperature, clocks, and throttling.

Global/loop/GNSS/dense stress must run concurrently while checking that the local deadline remains valid.

## 8. Decision benchmark fairness

For residual/frontend comparisons:

- share admitted raw IDs, calibration, deskew and preprocessing unless under test;
- share knot/keyframe/submap schedule unless under test;
- share target content and source-exclusion policy;
- share initial guesses, robust gates, and scoring timestamps;
- cap candidates by comparable compute/memory budgets;
- run declared failure cases, not only nominal sequences.

Record decisions D001–D017 and their artifacts in [OPTIMIZE.md](OPTIMIZE.md).

## 9. Slice acceptance checklist

- [ ] Clean build/static gates.
- [ ] Unit/Jacobian/property suite.
- [ ] Component deterministic replay and oracle comparison.
- [ ] Relevant failure-injection suite.
- [ ] No unexplained intake/drop/revision mismatch.
- [ ] Accuracy and consistency thresholds reviewed.
- [ ] Target p99/memory/power/thermal thresholds reviewed.
- [ ] Forensic trigger and recovery paths tested.
- [ ] Documentation and tuning ledger updated.
- [ ] Holdout used only under its declared policy.
