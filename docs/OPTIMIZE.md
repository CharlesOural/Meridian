# Meridian optimization and decision ledger

This ledger records every parameter, algorithm decision, and resource trade used to optimize Meridian. Architecture and provisional choices live in [SYSTEM_SPECS.md](SYSTEM_SPECS.md); this file records experiments and accepted values without rewriting the design.

No v2 performance result exists yet. Values in the specification are benchmark seeds, not measurements.

## 1. Rules

- Change one causal variable or a declared coupled set per A/B.
- Use identical admitted inputs, timestamps, calibration, graph schedule, seeds, and scoring interval.
- Record intake loss and capability/failure state next to every accuracy number.
- Tune on development sequences; evaluate holdouts only at declared milestones.
- Use full sequences for promotion. Short clips are diagnostic only.
- Report p50/p95/p99 and worst deadline, not only mean runtime.
- Record target Jetson power mode, clocks, temperature, throttling, CPU affinity, GPU load/memory, and competing workloads.
- Keep failed/negative experiments; they prevent repeated mistakes.
- A hard-coded constant follows the same process as a YAML/config value.

## 2. Tunable registry

Add a row before the first experiment that changes a parameter.

| ID | Owner/module | Key or constant | Units/type | Seed/current | Allowed range | Accuracy/robustness trade | Resource trade | Safety constraint | Status |
|---|---|---|---|---|---|---|---|---|---|
| T-PENDING | — | — | — | — | — | — | — | — | no v2 entries yet |

Status is one of `SEED`, `IN_EXPERIMENT`, `ACCEPTED`, `REJECTED`, or `RETIRED`. Accepted rows link to an experiment and artifact.

Required registry groups as they are implemented:

- clocks, reorder/age windows, queues, and source epochs;
- IMU initialization, noise, gap policy, bias reintegration, guard knots;
- visual detector/tracker, grid, keyframe, triangulation, robust gates, recovery;
- LiDAR validity/downsample/covariance, deskew, VGICP/p2plane, targets, robust gates, degeneracy hysteresis;
- local lag, state/factor/landmark caps, relinearization, marginalization and rebuild;
- sparse-submap split, overlap/context, proxy and seal caps;
- visual/LiDAR retrieval and verification;
- PCM/GNC limits, validation gates, graph/checkpoint cadence;
- GNSS quality, alignment excitation, spacing, information cap, NIS and reacquisition;
- ROS QoS/history, publisher/debug rates, blob/outbox limits;
- runtime governor, CPU/GPU/memory/power/thermal budgets;
- later dense voxel/truncation/color/occupancy/ESDF/mesh settings.

## 3. Algorithm decision register

The binding open decisions are D001–D017 in section 17 of `SYSTEM_SPECS.md`. Create one row when its benchmark begins.

| Decision | Candidates/versions | Benchmark manifest | Primary metrics | Failure suite | Jetson profile | Result artifact | Chosen | Review/date |
|---|---|---|---|---|---|---|---|---|
| D001–D017 | see specification | pending | pending | pending | pending | pending | provisional only | pending |

## 4. Experiment record template

Copy this block for each experiment. Never overwrite an old result.

```text
Experiment: EXP-YYYY-NNN
Hypothesis:
Owner/reviewer:
Decision/tunable IDs:
Candidate commits and model hashes:
Container/build ID:
Calibration/config revisions:
Dataset manifests and checksums:
Development vs holdout role:
Fault injections and random seeds:
Target hardware / JetPack / CUDA / power mode / clocks / ambient:
Concurrent workloads:
Baseline:
Variant:
Admission integrity (input counts, gaps, rejects, queue drops):
Accuracy/consistency (ATE, RPE, drift, NEES/NIS, availability):
Robustness (failure/recovery latency, false commits, capability time):
Resources (p50/p95/p99/max latency, CPU/GPU/RAM/VRAM/power/temp):
Artifacts (reports, trajectory, forensic bundles, plots):
Result:
Decision and rationale:
Follow-up:
```

## 5. Run summary table

| Experiment | Date | Change | Development datasets | Holdout used? | Accuracy delta | Robustness delta | p99/deadline delta | Power/thermal delta | Decision |
|---|---|---|---|---|---|---|---|---|---|
| — | — | no v2 experiments yet | — | no | — | — | — | — | — |

## 6. Standing validity checks

An experiment is invalid if any of these differs unintentionally between candidates:

- admitted measurement IDs or transport loss;
- calibration/time model, topic mapping, or ground-truth frame;
- local knot/submap schedule when not itself under test;
- random seed/determinism mode;
- robust gate, target content, or initial guesses in a residual comparison;
- CPU/GPU power/thermal/concurrent-load posture;
- scoring timestamps or alignment method.

Input integrity dominates comparisons: an algorithm that appears faster because it silently drops old work has failed, not won.
