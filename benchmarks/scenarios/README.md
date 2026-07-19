# Benchmark scenarios

## Trajectory pairing

Newer College's base-frame ground truth is paired only with
`trajectory_base.tum`. Both sides name `base_link` independently so the
evaluator can reject a mismatch before association or alignment. An IMU-frame
trajectory must instead use the matching IMU-frame ground truth; renaming a
frame or relying on SE(3) alignment is not a valid conversion.

Manual evaluator mode deliberately requires both frame arguments. Acceptance
runs instead select the checked-in scenario and profile, which own both frame
names and reject conflicting CLI overrides:

```bash
python3 tools/eval_ate.py \
  bags/newer-college/gt/tum/gt-nc-quad-easy.csv \
  trajectory_base.tum \
  --scenario benchmarks/scenarios/newer_college_quad_easy.yaml \
  --profile internal_interpolated \
  --track-label local
```
