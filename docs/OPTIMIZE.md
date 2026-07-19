# Meridian optimization ledger

This file records V3 parameter tuning, algorithm comparisons, and resource
trade-offs. Provisional architecture belongs in
[SYSTEM_SPECS.md](SYSTEM_SPECS.md); this ledger contains measured evidence and
the resulting implementation decisions.

No V3 experiment has been recorded yet.

## Experiment rules

- Change one causal variable, or one explicitly coupled set, per comparison.
- Keep inputs, calibration, initialization, scoring interval, and random seeds
  identical between candidates.
- Record coverage, rejected inputs, queue loss, and health state with every
  accuracy result.
- Use short clips for diagnosis and complete sequences for promotion.
- Keep negative results so failed approaches are not repeated.
- Report distribution tails and resource use, not only mean runtime.

## Tunable registry

Add a row before the first experiment that changes a value.

| ID | Module | Parameter | Units/type | Seed | Tested range | Current value | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-PENDING | — | — | — | — | — | — | SEED | — |

Status is `SEED`, `IN_EXPERIMENT`, `ACCEPTED`, `REJECTED`, or `RETIRED`.

## Algorithm decision registry

| ID | Decision | Candidates | Dataset/profile | Primary metrics | Selected | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| D-PENDING | — | — | — | — | — | — |

## Experiment template

Copy this section for each run and keep completed records immutable.

```text
Experiment: EXP-YYYY-NNN
Date:
Hypothesis:
Owner/reviewer:
Decision and tunable IDs:
Source commit:
Container/build identity:
Configuration and calibration identity:
Dataset, profile, and checksums:
Baseline:
Variant:
Input integrity (counts, gaps, rejects, queue drops):
Accuracy (ATE, RPE, coverage, drift):
Robustness (failures, recovery, invalid outputs):
Runtime (RTF and stage p50/p95/p99/max):
Resources (CPU, RAM, power, temperature, throttling):
Artifacts:
Result:
Decision and rationale:
Follow-up:
```

## Run summary

| Experiment | Date | Change | Dataset | Accuracy effect | Robustness effect | Runtime/resource effect | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| — | — | no V3 experiments yet | — | — | — | — | — |

## Validity checklist

Before comparing two candidates, verify that these remain equivalent unless
they are the tested variable:

- admitted measurement IDs and timestamp support;
- calibration, frames, and time conventions;
- initialization and estimator schedule;
- target geometry and pose seed;
- scoring timestamps and alignment protocol;
- host workload and thermal/power state.

