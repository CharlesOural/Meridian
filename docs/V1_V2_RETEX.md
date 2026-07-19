# Meridian v1/v2 — Engineering Retrospective

## Purpose

This document preserves the implementation evidence, measured results, failed
experiments, and cross-module lessons that must survive the v3 rewrite. It is
not the current architecture or a parameter source. [`SYSTEM_SPECS.md`](SYSTEM_SPECS.md)
defines v3. The relevant V2 experiment ledger evidence is distilled here; new
optimization records start with the V3 implementation.

A specification states intent; source, tests, and complete run artifacts state
what existed. Short-prefix scores below remain diagnostic. Only explicitly
named complete-sequence runs support full-run accuracy or runtime conclusions.

## Version and evidence boundary

### Exact version vocabulary

The repository contains four distinct historical boundaries that must not be
collapsed into an ambiguous “v1” or “v2” label.

| Name used here                 | Exact object                                                                                                                                                                           | Meaning                                                                                                                                                                                                                         |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tagged v1                      | commit [`73a795edb5fd4ddc2e46f79d19fb4b0bb045820f`](https://github.com/CharlesOural/Meridian/tree/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f), target of annotated tag `v1`              | The released discrete-LIO frontend, keyframe backend, place/GNSS stack, and telemetry snapshot.                                                                                                                                 |
| Post-v1 hardening and mapping  | commits after the tag through extraction baseline [`f5ca513158c95aaf88223486ec481c1d42730a21`](https://github.com/CharlesOural/Meridian/tree/f5ca513158c95aaf88223486ec481c1d42730a21) | Includes `9d99b94` convergence tuning, `b73e897` gyro-bias observation, `1cfc208` open-addressed/parallel association, and the CPU/nvblox map work. These mechanisms are not present at tag `v1`.                               |
| Cleanup base                   | commit `aaa145da744d5a35079b3f008892341e03622853`                                                                                                                                      | Deletes the legacy implementation before the rewrite. It is not a v2 implementation snapshot.                                                                                                                                   |
| V2 implementation audited here | commit [`1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44`](https://github.com/CharlesOural/Meridian/tree/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44), branch `v2` | The complete `meridian_core/local_rt/global/ros/tools/apps` sources, tests, benchmark scenarios, and final V2 documentation snapshot committed on 2026-07-19. |

Future readers should use the full object IDs above. Deleted paths remain
readable with, for example:

```bash
git show 73a795edb5fd4ddc2e46f79d19fb4b0bb045820f:src/meridian_frontend/src/lio/scan_registration.cpp
git show f5ca513158c95aaf88223486ec481c1d42730a21:src/meridian_map/src/layered_map.cpp
```

The V2 source links below target its immutable commit so this retrospective
remains navigable after the clean V3 rewrite.

### Evidence strength and claim rules

- Source plus focused tests establishes an implemented component contract.
  It does not establish that an executable composed the component.
- A complete run establishes only the configured modalities, dataset,
  evaluation profile, replay mode, machine, and code snapshot that it names.
- The v1 exact results below are durable ledger/ADR statements, but their raw
  run directories are absent. Treat them as historical measured evidence, not
  independently reproducible publication results.
- The decisive v2 run directories still existed under `/tmp` during this
  audit. Their identities are recorded below, but the bytes, source commit,
  build/container identity, and resolved bag checksum are not in Git.
- Values reported only by prose or an external `/usr/bin/time` invocation are
  labelled ledger-reported. A hash identifies recovered bytes; it does not
  archive them.
- Historical design prose is not proof that an integration path ran.

## V1 and post-v1 lessons

### What actually shipped

Tagged v1 was a keyframe-oriented pipeline:

```text
LiDAR + grouped IMU
  -> heuristic discrete LIO / world-frame voxel map
  -> relative or absolute keyframe packet
  -> pose-oriented iSAM2 backend
  -> GNSS and loop constraints
  -> optional correction fed back into frontend state (map correction was incomplete)
```

Useful algorithms existed across the frontend, backend, place-recognition,
mapping, calibration, ROS, and telemetry packages. The package split also
created ambiguous ownership: a local frontend owned an `odom`-like world map
while the backend could move its pose into `map`. V3 should extract narrow
algorithms and tests, not restore that runtime architecture.

Primary source map:

- tagged [LIO coordinator](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/lio_frontend.cpp),
  [IMU tracker](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/imu_tracker.cpp),
  [registration](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/scan_registration.cpp),
  and [voxel map](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/voxel_grid_map.cpp);
- tagged [backend](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_backend/src/isam2_backend.cpp)
  and [place detector](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_place/src/hierarchical_loop_detector.cpp);
- post-v1 [engineering ADR](https://github.com/CharlesOural/Meridian/blob/b73e897fd62f70db8f3d1cb07e533e673deeb6e1/docs/ARCHITECTURE_DECISION.md)
  and final [v1-lineage tuning ledger](https://github.com/CharlesOural/Meridian/blob/f5ca513158c95aaf88223486ec481c1d42730a21/docs/OPTIMIZE.md);
- final [map layer](https://github.com/CharlesOural/Meridian/tree/f5ca513158c95aaf88223486ec481c1d42730a21/src/meridian_map).

### Initialization, propagation, and bias

The tagged `ImuTracker` accumulated IMU samples for a configured duration
and then assumed standstill. It used mean gyro as `b_g`, mean
specific-force direction for roll/pitch, zero local yaw and velocity, and the
remaining gravity-reaction error as `b_a`. No independent zero-motion
assertion, stationarity hypothesis test, complete covariance, or moving
initializer existed.

The useful part is a fast supervised branch when zero motion is independently
known. The limitation is observability: one static orientation cannot
independently recover transverse accelerometer bias and tilt, and low IMU
variance alone cannot assert zero velocity.

The tagged propagation and deskew must be described precisely:

- propagation integrated every accepted IMU sample with the current sample held
  over `dt`, exact SO(3) exponential rotation, and simple constant-
  acceleration position/velocity kinematics;
- scan deskew used one interval-mean angular rate and body velocity, applying an
  exact SE(3) exponential under that constant-screw model.

They were not a shared midpoint/preintegration model, but neither was deskew
merely a first-order pose update. V3 retains the simple ownership boundary and
uses exact raw support, midpoint propagation, and graph preintegration.

Several source-level limits matter when reusing the initializer or tracker:

- `rebase()` changed stamp, pose, and velocity but retained the originally
  estimated biases; there was no normal-path online accelerometer-bias state;
- configured IMU density/random-walk fields did not drive the final discrete
  LIO path, and translated IMU extrinsics omitted lever-arm transport terms;
- pre-initialization storage was capped at 64 groups. Once initialization
  completed, held scans drained even though their IMU support had already been
  consumed, so their poses were not reconstructed from raw history;
- the first usable sweep populated the map without registration. Later large
  IMU gaps armed a constant-velocity reseed, while an ordinary sparse or
  non-converged sweep advanced dead reckoning and retained the existing map;
- solved velocity was finite-differenced from consecutive registration poses,
  not estimated as part of an inertial optimization.

See the tagged [tracker](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/imu_tracker.cpp)
and [frontend state machine](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/lio_frontend.cpp).

Post-v1 commit
[`b73e897fd62f70db8f3d1cb07e533e673deeb6e1`](https://github.com/CharlesOural/Meridian/commit/b73e897fd62f70db8f3d1cb07e533e673deeb6e1)
added a slow registration-driven gyro-bias observer. With gain `0.05`,
the historical ledger reports Park ATE `0.640 -> 0.517 m` and maximum
error `1.46 -> 0.95 m`, Quad Hard `0.123 -> 0.118 m`, and
neutral changes on Quad Easy and Math Medium. Gain `0.1` over-corrected.
The reusable lesson is that a low-frequency external orientation observer can
control long-run bias; the exact filter and gain are not v3 calibration
results, and accelerometer bias was not similarly validated.

### Direct registration and model selection

Tagged v1 provided a compact sequential point-to-point Gauss-Newton frontend:

- deterministic voxel sampling;
- a spatially clipped voxel map with a per-cell point cap and minimum spacing;
- fixed-order probing of the 27 neighbouring cells;
- a constant-screw deskew seed;
- a seed-orientation roll/pitch regularizer, heuristically weighted using IMU
  acceleration statistics and used only inside the frontend proposal;
- a final data-only Hessian and heuristic covariance.

The 27-cell search is exact only within its one-cell envelope. The header
explicitly warns that radii larger than one voxel edge are silently truncated,
and configuration validation did not require
`max_corr_dist_m <= voxel_size_m`. The shipped Quad profile was safe at
`0.5 <= 1.0 m`; the mechanism was not exact for arbitrary configuration.
The map stored the full deskewed sweep in world coordinates even though
registration used sampled keypoints. Clipping scanned every occupied voxel and
tested cell centres. Sequential association work was roughly keypoints times
candidate points in the 27 cells times Gauss-Newton iterations; clipping was
linear in occupied voxels.

The orientation term was not an inertial measurement residual. It froze a
gravity anchor from the propagated pose seed, so its residual started at zero,
and used acceleration magnitude variance/sample count only to choose a
heuristic weight. Mean acceleration did not enter as a measurement. The
`max_expected_jerk` path was a persistent diagnostic scalar filter, not a
registration constraint. Sources: tagged
[registration](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/scan_registration.cpp)
and [voxel-map contract](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/voxel_grid_map.hpp#L47-L63).

At the tag, the voxel store used the original map implementation, association
was sequential, and the Newer College convergence threshold was `1e-5`.
Post-tag commit
[`1cfc208d587707a290be00babb4a7bb43c1558fd`](https://github.com/CharlesOural/Meridian/commit/1cfc208d587707a290be00babb4a7bb43c1558fd)
introduced the power-of-two open-addressed index and bounded OpenMP
association. Commit
[`9d99b940017225b6d6dc4c6c469218267fe46290`](https://github.com/CharlesOural/Meridian/commit/9d99b940017225b6d6dc4c6c469218267fe46290)
changed the evaluated Newer College convergence threshold to `1e-3`.

The post-v1 ADR records these full-sequence historical results:

| Registration                        | Quad Easy | Math Medium | Quad Hard |       Park |
| ----------------------------------- | --------: | ----------: | --------: | ---------: |
| Point-to-point baseline             | `0.085 m` |   `0.144 m` | `0.123 m` |  `0.640 m` |
| Pure point-to-plane                 | `0.081 m` |    `1537 m` |  `6091 m` | `129017 m` |
| P2P warm-up 8, then point-to-plane  | `0.083 m` |      `87 m` |   `5.8 m` |   `1013 m` |
| P2P warm-up 20, then point-to-plane | `0.086 m` |    `29.7 m` |  `0.30 m` |   diverged |

Point-to-plane was about `23 ms` versus `73 ms` registration
on gentle Quad Easy. It failed the aggressive/vegetated stressors because of a
narrower association basin, sensitivity to residual deskew through fitted
normals, and false planes in vegetation. This is evidence for the off-road
baseline, not a universal claim against point-to-plane or distributional
registration.

The same campaign established two other durable lessons:

- loosening natural convergence from `1e-5` to `1e-3` changed
  roughly `32 -> 20` iterations and `116 -> 77 ms` frontend
  time with no recorded ATE change;
- a hard cap of 10 iterations diverged on every evaluated sequence. A fault
  ceiling and a natural convergence condition are different contracts.

A fixed-scale robust kernel also changed Park from `0.64` to
`7.8 m`. Robust scale must follow a declared residual population and
be stress-tested; “add a robust kernel” is not automatically safer.

The annotated tag records a deterministic Quad Easy system headline of
`0.088 m`. Its release merge message additionally records 14 packages,
529 passing tests, 242 keyframes, nine admitted loops, and two byte-identical
sync replays with MD5 prefix `bd3772ba`. That score, the frontend-only table above, and later
loop-enabled rows use different scopes. No raw v1 run directory or complete
run manifest survives, so do not silently combine them into one reproducible
baseline.

### Information and covariance

Tagged registration formed the unaveraged point-row Hessian, estimated an
isotropic variance as `chi / (3N - 6)`, added a numerical ridge for
inversion, and chained relative covariances between keyframes. The
gravity/IMU proposal regularizer was correctly excluded from the reported
LiDAR information.

The chained relative covariance adjoint-transported and added successive
increments as if their map errors were independent. Gap-bridge uncertainty and
scan-to-map correlation were omitted. Synthetic Monte Carlo used a fixed map
and independent isotropic point noise, so it did not validate the field scale.
The frontend's separate observability proxy fitted local point-to-plane rows
and reported only body-axis diagonal scores `h / (h + 0.01)`; it did not expose
the eigenbasis of the point-to-point Hessian. The backend could therefore
inflate named axes, not the true unsupported directions.

The directional shape is useful. The absolute scale is not calibrated because
point rows, target history, deskew, calibration, and environment modelling are
correlated. V2’s large accuracy swing under a physical information cap
confirmed this. V3 may retain the supported subspace and an independently
benchmarked information ceiling; it must not treat raw point count or a
Hessian inverse as independent measurement covariance.

Sources: tagged
[registration covariance](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/scan_registration.cpp#L139-L163),
[relative covariance transport](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/relative_cov.hpp),
and
[backend observability inflation](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_backend/src/observability_inflation.cpp).

### Three cross-component contract failures

Tagged v1 contained three severe seams that local unit tests did not catch.

First, after a gap/reseed the LIO frontend emitted a mid-run
`AbsolutePrior` keyframe, but the backend accepted
`AbsolutePrior` only for the first keyframe and dropped every later one.
The frontend recovery contract and backend admission contract were mutually
incompatible. Relevant sources are tagged
[frontend keyframe emission](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/lio_frontend.cpp#L434-L478)
and
[backend admission](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_backend/src/isam2_backend.cpp#L103-L121).
Recovery paths require a composed contract test, not two locally valid APIs.

Second, backend correction shifted the discrete LIO state, tracker, and live
pose while explicitly leaving the world-frame voxel map in the old frame. It
did not arm the only failure branch that could clear/reseed that map. The same
bug had already been diagnosed in the earlier CT frontend by commit
[`055473061dcbd4b55b790844133cb5defd28bd13`](https://github.com/CharlesOural/Meridian/commit/055473061dcbd4b55b790844133cb5defd28bd13):
a loop correction produced `24 m` divergence; rigid map
transform/rebuild restored `0.10 m` backend and `0.16 m`
frontend ATE. The discrete rewrite deleted that implementation and reintroduced
the invariant violation in tagged
[`LioFrontEnd::apply_correction`](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/lio_frontend.cpp#L519-L540).

Third, an indeterminate backend update retried with QR and then restored the
last-good graph while deleting the staged keyframes. The next real frontend
packet still referenced the deleted predecessor and failed the backend's
strict contiguity gate. The recovery test only progressed by manually
fabricating a packet rooted at the last surviving ID; the live frontend had no
resynchronization handshake, and its alternative mid-run `AbsolutePrior` was
also rejected. See tagged
[rollback implementation](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_backend/src/isam2_backend.cpp#L795-L950)
and [recovery test](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_backend/test/test_backend_recovery.cpp#L65-L98).

A related correction gap was semantic rather than numerical: the pipeline
parked corrections only when `GraphUpdate.loop_closed` was true, while that
flag meant only “this batch contained a loop.” GNSS-driven realignment was not
fed back to the frontend despite comments describing loop or GNSS correction.

V3 avoids these failure classes: local geometry stays owner-local or fixed in
`odom`, global revision is expressed only through `map -> odom`, and
recovery/finality are common transactional contracts with an explicit
re-root/epoch transition when a live chain is abandoned.

### Backend, loop closure, and GNSS

The v1 backend supplies useful component algorithms, but its keyframe-oriented
pose graph, duplicated restart IMU summary, and global-correction feedback are
not a v3 runtime template.

Reusable findings include:

- explicit translation-first Meridian versus rotation-first GTSAM adapters and
  finite-difference/Jacobian tests;
- exact-time GNSS interpolation and a rotated antenna lever arm;
- staged candidate/rebuild mechanics, provided the missing live-chain resync
  contract above is supplied;
- chain covariance, PCM, and deterministic maximum-clique primitives;
- separating retrieval score from geometric verification.

The immutable `KeyframePacket` was a valuable L2-to-L3 seam: exactly one
mutually exclusive geometric constraint, shared immutable cloud/image handles,
calibration provenance, named-frame observability, and an explicit covariance
reorder at the GTSAM boundary. Its shipped limits are equally important:

- the final frontend never emitted `ImuPreintegration`; ordinary L3 operation
  was pose-only with one odometry `BetweenFactor` per keyframe, while the
  combined-IMU path was exercised synthetically;
- the covariance advertised by the first packet was ignored. The backend used
  `backend.anchor_sigma` through a `GaugeDampingFactor` instead;
- that factor's nonlinear error was zero at every pose while its linearization
  injected `sqrt(lambda) I`, making it numerical gauge regularization rather
  than a calibrated measurement.

The tuning ledger exposed the mismatch directly: its
`kAnchorPriorVar = 1e-8` implied sigma `1e-4`, while the configured backend
`anchor_sigma` was `0.1` and the packet value was ignored.

Sources: tagged
[`KeyframePacket`](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_common/include/meridian/common/keyframe_packet.hpp),
[backend packet admission](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_backend/src/isam2_backend.cpp#L103-L285),
and
[`GaugeDampingFactor`](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_backend/src/gauge_damping_factor.cpp).

Gauge damping was measured, not merely selected. With
`lambda = 100` it absorbed more than `99.9%` of a forced rigid
shift in the recorded chain test. `lambda = 1e4` reached only
`38%` after 5000 batch-GN iterations, and `lambda = 1e8` behaved
like a hard prior and distorted relative geometry. The lesson is to test gauge
handling against the corrections it must permit.

Loop closure exposed the cost of weak retrieval precision:

- a Scan Context distance threshold of `0.5` admitted 45
  co-visible-but-not-co-located loops and produced `6.6 m` ATE;
- `0.13` plus strict GICP fitness/overlap admitted seven
  high-confidence loops in the earlier campaign and changed Quad Easy ATE
  `0.124 -> 0.132 m`, essentially neutral/worse on an already low-drift
  trajectory. The later tagged profile reported about nine loops and
  `0.088 m` with or without closure; those are different run scopes;
- commit
  [`458c63979362f63298ee5a069b5241187d8a1f54`](https://github.com/CharlesOural/Meridian/commit/458c63979362f63298ee5a069b5241187d8a1f54)
  fixed overlap normalization. The buggy path divided inliers by the full input
  cloud rather than the exact downsampled cloud being registered, reducing
  overlap by about 100x and rejecting every loop.

The v1 bounded max-clique fell back to a deterministic greedy clique when its
work bound was exhausted. V3 should preserve deterministic bounded work but
defer a complete component rather than silently substitute a partial heuristic
when the robust admission policy requires an exact set.

The shipped loop cascade was Scan Context retrieval, submap GICP, then an
odometry-chain chi-square self-test. STD/BTC existed only as configuration/spec
surface. Candidate eligibility partly used distance in the drifted corrected
estimate; GICP started from odometry rather than the retrieved Scan Context
yaw; and the configured `gicp_rmse_max` was not used by admission. The
single-loop self-test could also reject a true loop when the odometry-chain
covariance was optimistic. Deterministic replay forced one GICP thread, while
live mode used configured threading. Loop covariance was an
inverse-Hessian/fitness/degeneracy heuristic that simplified endpoint
observability into the factor frame; batch GNC supplied an additional layer
only in deterministic mode. These details make replay parity and live loop
throughput separate claims. The Quad profile enabled place closure even though
the global `PlaceConfig` default was false, so it was profile-specific behavior.

Restart/GNSS commits
`d6ca9932cdd17e6131813204a65a56913b087e33`,
`2c23b2706e8bd9b852ea01544c0f30ccc7c7763a`, and
`8425155345075a33466aafacfbe3df4ce9186955` caught gravity ENU/NED
sign, PIM tangent-order, key-lifecycle, and roughly `0.5 m`
datum/factor antenna-geometry errors. These are source/test extraction targets,
not proof of complete GNSS/global field performance.

The implemented GNSS path did include first-fix ENU origin, baseline/excitation
gates for a 4-DoF datum, exact-time interpolation, antenna lever arm, robust
factors, marginal/spacing gates, and chi-square auto-disable. Buffered
pre-lock fixes helped initialize the datum but were never replayed as factors;
datum yaw sigma was a geometry heuristic without measurement-noise scaling;
online lever refinement was disabled by default. Quad Easy carried no GNSS, so
none of this has release-sequence field validation.

### Input integrity, determinism, and telemetry

Input integrity was one of v1’s strongest durable results:

- best-effort delivery of fragmented roughly 8 MB scans lost 465 messages in
  the observed loaded run while the reliable probe received
  `2671/2671`, about `14%` silent loss;
- publisher and subscriber reliability must match; queue depth cannot repair
  messages lost before the callback;
- the permanent intake waterfall distinguished bag publication, callback
  delivery, conversion rejection, preprocessing, queue eviction, and frontend
  ingest;
- the observed cloud/image/IMU subscription depths were `20/40/400`, but
  capacity did not repair in-flight best-effort loss;
- a 10 Hz per-key telemetry token bucket beating against 10 Hz sweeps captured
  between `1%` and `98%` depending on the key; 50 Hz removed
  the aliasing;
- an unrelated encoder using about `65%` CPU doubled drops;
- deterministic direct replay was byte-identical for identical admitted input,
  while live ROS replay was not;
- short clips hid queue pressure and a bias failure that appeared after roughly
  90 seconds.

The standing rule survives unchanged: validate admitted IDs, holes, queue
loss, coverage, and machine load before reading ATE or timing. Never build into
a symlinked install overlay while a replay is running.

The live and replay schedulers did not exercise the same overload surface. Live
mode used three stage threads: a lossy 512-item sensor queue; a 192-item
measurement queue that preferentially evicted live-state IMU before a whole
sweep group; and a backend queue configured at 64 but fed with `push_always`,
so it could grow without bound and only warned after depth 32. The newest
frontend correction occupied one overwritable slot. Direct replay bypassed all
of those threads and queues and folded every keyframe/loop immediately. Its
byte determinism proved admitted computation/order, not live queue loss,
batching cadence, correction overwrite, GICP threading, or RTF.

The aggregator was LiDAR-centric: one sweep with spanning IMU, one interval
image, and interval GNSS. It retained a straddling IMU sample, allowed 20 ms
reordering, closed after IMU coverage or 150 ms timeout, and capped the
IMU/image/GNSS/sweep buffers at `8192/16/64/8`. This is evidence for replacing
the bundle with a sensor-neutral event timeline, not for copying its capacities.

Sources:
[`cdff23b131ea6e2ba990c8fad73bc47986717345`](https://github.com/CharlesOural/Meridian/commit/cdff23b131ea6e2ba990c8fad73bc47986717345),
the tagged [debugging runbook](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/docs/REALTIME_DEBUGGING.md),
[`MeridianPipeline`](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_pipeline/src/meridian_pipeline.cpp),
[aggregator](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_sensors/src/aggregator.cpp),
and the final [v1-lineage ledger](https://github.com/CharlesOural/Meridian/blob/f5ca513158c95aaf88223486ec481c1d42730a21/docs/OPTIMIZE.md).

### Camera, map visualization, and evaluator boundary

Tagged v1 did not fuse camera measurements. The frontend attached the latest
raw image to a keyframe; image undistortion ran only for debug
`CameraPyramid` telemetry, and the pipeline explicitly passed the raw image to
the estimator. This was an inspection path, not visual-inertial odometry.

There was also no `meridian_map` package at the tag. `map/cloud` was a
visualization product that periodically recomposed retained keyframe clouds,
every five backend folds and on loop closure, then first-point voxelized them.
That publication was O(total retained history). Frontend `toCloud()` preserved
only XYZ and
zeroed intensity, point time, ring, ambient, and range, reintroducing the
payload-loss defect previously fixed in the CT path by
[`c414c06f6c53941508870d7019a94e63f4227565`](https://github.com/CharlesOural/Meridian/commit/c414c06f6c53941508870d7019a94e63f4227565).
The post-v1 TSDF/nvblox work below is therefore a separate lineage, not shipped
V1 dense mapping.

Finally, `tools/eval_ate.py` used nearest timestamps within 50 ms and rigid
SE(3) position alignment. It reported position ATE summaries but no support
coverage, full-span gate, orientation/RPE, runtime, resource, or input-integrity
check; an advertised time-offset option was never read. Thus the tag headline
`0.088 m` is a useful historical release result, but its trajectory, exact
scoring support, timings, and input manifest are not recoverable. Quad Easy was
one low-drift sequence where loops were neutral; GNSS was absent, camera was
unfused, mapping was visualization-only, and no held-out Quad Hard or Jetson
result is preserved.

Sources: tagged
[frontend image/cloud emission](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_frontend/src/lio/lio_frontend.cpp),
[camera debug path](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/src/meridian_pipeline/src/meridian_pipeline.cpp#L543-L650),
and [ATE evaluator](https://github.com/CharlesOural/Meridian/blob/73a795edb5fd4ddc2e46f79d19fb4b0bb045820f/tools/eval_ate.py).

### Calibration import

The Newer College Kalibr file duplicated accelerometer density
`0.019` into the gyroscope-density field. The historical retired
estimator changed from `5018 m` to `0.066 m` after using
roughly `0.002 rad/s/sqrt(Hz)`. Its gyro random walk
`2.66e-4 rad/s^2/sqrt(Hz)` also allowed bias to absorb LiDAR error and
failed after roughly 90 seconds; the historical comparison was
`15.99 -> 0.193 m` at `4e-6`.

Those ATE values came from a retired estimator and are not v3 calibration
results. The durable lesson is to validate every imported density and random
walk independently, retain physical units, and test the complete sequence and
the exact sensor/calibration identity. See the final
[dataset caveat](https://github.com/CharlesOural/Meridian/blob/f5ca513158c95aaf88223486ec481c1d42730a21/docs/DATASET.md#L143-L159).

### Dense mapping and provenance

Post-v1 mapping work is extraction material even though it is not present at
the tag:

- commit `30f9cfd1d736ac80c7ef86be1f6b6623c7ddcc2b` added the CPU
  TSDF/mesh oracle and keyframe pose/calibration provenance;
- `a5bb69a54f026ecdeaa4dfd658a95b45908a3778` added camera colouring
  and provenance-derived dirty/rebuild bounds;
- `610d596e9bc1e1afbf1a5f05d48766ebab5f7ba4` repaired the pinned
  nvblox v0.0.7 GPU build/capability seam;
- [`f9072553635b018a71f0c16a4eb109ef7d8ec89c`](https://github.com/CharlesOural/Meridian/commit/f9072553635b018a71f0c16a4eb109ef7d8ec89c)
  added CPU/GPU conformance and deterministic clear/rebuild;
- `c6096520da5587f80e124967933ce5bc834a0243` fixed a camera
  extrinsic that had silently stayed identity and moved rectification to the
  typed boundary.

The recorded GPU resource result matters: `0.10 m` voxels with a
`40 m` integration range exhausted an 8 GB GPU because projective
free-space carving allocated blocks along full rays. `0.15 m / 30 m`
completed the evaluated sequence. A hidden `7 m` occlusion-tracer
default also truncated colour.

The architecture lesson is broader than those values. Retained owner-local
keyframes and their pose/calibration provenance are authoritative; fused
TSDF/colour/mesh products are disposable caches. Running-average fusion cannot
be exactly “subtracted.” A correction must identify every co-contributor,
clear the complete affected block superset, and deterministically rebuild it.
Backend selection must be explicit and fail-fast; CPU is the oracle and GPU
conformance is tested, not a silent fallback. Mapping and mesh publication
remain off the localization critical path.

Sources:
[`layered_map.cpp`](https://github.com/CharlesOural/Meridian/blob/f5ca513158c95aaf88223486ec481c1d42730a21/src/meridian_map/src/layered_map.cpp),
[CPU conformance test](https://github.com/CharlesOural/Meridian/blob/f5ca513158c95aaf88223486ec481c1d42730a21/src/meridian_map/test/test_layered_map.cpp),
and
[nvblox backend](https://github.com/CharlesOural/Meridian/blob/f5ca513158c95aaf88223486ec481c1d42730a21/src/meridian_map/src/nvblox/nvblox_surface_map.cpp).

### Earlier continuous-time campaign: evidence, not an architecture to restore

The pre-release continuous-time estimator was deleted before tagged v1, but
its complete-sequence failures contain lessons needed if v3 again uses
fixed-lag estimation:

- [`fbafa394a57598d0f877c19facdf12d78ae3e850`](https://github.com/CharlesOural/Meridian/commit/fbafa394a57598d0f877c19facdf12d78ae3e850)
  showed why short clips are misleading. On the full 267 s sequence,
  visual-map work hidden by a 21 s clip cost 162 ms and 118 drops; bounded
  adaptive density changed that to 32 ms and 20 drops. Marginalization had to
  eliminate the whole departing knot group; trimming raw history discarded
  information rather than bounding it correctly.
- [`251b2444357766cf5cb26dc4a1c644dd61f47d1e`](https://github.com/CharlesOural/Meridian/commit/251b2444357766cf5cb26dc4a1c644dd61f47d1e)
  combined deterministic thread-local association with an ikd-tree race fix:
  association changed `21 -> 7 ms`, sweep time `98 -> 85 ms`, and
  drops `205 -> 7` at about `0.098 m` ATE.
- [`da76dd0fe242f55e6e7283bcb261c0bad6a56d48`](https://github.com/CharlesOural/Meridian/commit/da76dd0fe242f55e6e7283bcb261c0bad6a56d48)
  found keyframe covariance taking 30–80 ms, with p90 about 103 ms. A
  nice-19 worker with about one second of slack reduced inline restart pressure
  from 25 to 0–1. In that campaign, prior scale `0.5` also changed Canteen ATE
  `0.894 -> 0.083 m`.
- [`fcd1866fc8768a8198f2751bdf899011d2358e6d`](https://github.com/CharlesOural/Meridian/commit/fcd1866fc8768a8198f2751bdf899011d2358e6d)
  demonstrated sequence dependence: prior scale `0.5` produced
  `{0.193, 0.159} m`, while `1.0` produced `{0.078, 1.172} m` on the two
  full bags. A short-prefix optimum was not a general knob.
- [`df9316d3a4c3fcd28c33b031f6fc6dce135eb1a0`](https://github.com/CharlesOural/Meridian/commit/df9316d3a4c3fcd28c33b031f6fc6dce135eb1a0)
  and [`9fe03139b170ba1f6e071bfd16befc149f907310`](https://github.com/CharlesOural/Meridian/commit/9fe03139b170ba1f6e071bfd16befc149f907310)
  parity-tested analytic LiDAR, visual, and IMU Jacobians against autodiff near
  `1e-7` to `1e-9`. Cheaper IMU derivatives did not lower wall time under the
  same deadline; they enabled more iterations and improved ATE
  `0.123 -> 0.100 m`.
- [`2af22878bbe686b31eefdd741cc5906685aa621f`](https://github.com/CharlesOural/Meridian/commit/2af22878bbe686b31eefdd741cc5906685aa621f)
  preserves the spline-specific traps: unsupported future knots create
  nullspaces and must be pinned; leaving knot/bias groups must be
  Schur-marginalized together; prior storage/pointers must survive; and prior
  information deflation must be explicit.
- [`cba0ee2a2825be54059561b18cf0cae08db0cf76`](https://github.com/CharlesOural/Meridian/commit/cba0ee2a2825be54059561b18cf0cae08db0cf76)
  showed that Allan-floor IMU noise overtrusted a vibrating legged platform:
  roughly `448 m / 120 s` became `0.2 m / 120 s` and
  `4.5 m / 161 m` after about 40x covariance inflation. It also recorded a
  measured 26 ms IMU/LiDAR offset and visual failure in a dark corridor.
- [`055473061dcbd4b55b790844133cb5defd28bd13`](https://github.com/CharlesOural/Meridian/commit/055473061dcbd4b55b790844133cb5defd28bd13)
  established the state-plus-map correction invariant described above, and
  [`44a088d7c599afbd95745ded63eb5f273fa205b9`](https://github.com/CharlesOural/Meridian/commit/44a088d7c599afbd95745ded63eb5f273fa205b9)
  replaced a second-estimator oracle with direct ground-truth tests: two
  estimators agreeing is weaker evidence than either satisfying truth gates.

These are extraction lessons, not a request to resurrect the implementation.
Commit [`34d311a8788143cfb450218258aa2430191a8d63`](https://github.com/CharlesOural/Meridian/commit/34d311a8788143cfb450218258aa2430191a8d63)
records that the B-spline approach was not working; commits
[`42475f65f06a827e18dae43db695032f56afe04e`](https://github.com/CharlesOural/Meridian/commit/42475f65f06a827e18dae43db695032f56afe04e),
[`cd365a4b30f34fe9cca2fbabdaaec8f6af97df73`](https://github.com/CharlesOural/Meridian/commit/cd365a4b30f34fe9cca2fbabdaaec8f6af97df73),
and [`8993eb4441331203ad3ec768c09e79d594aafd0f`](https://github.com/CharlesOural/Meridian/commit/8993eb4441331203ad3ec768c09e79d594aafd0f)
then purged its configuration, estimator, and pipeline machinery before the
discrete v1 release.

## V2 implementation and validation boundary

V2 was a substantial offline local estimator and a separately implemented
global component library. It was not an integrated local-plus-global runtime.

| Capability                                                      | Implemented evidence                                                        | End-to-end evidence                                              |
| --------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Core IDs, time, calibration, lineage, canonical records         | Source and focused tests in [`meridian_core`](https://github.com/CharlesOural/Meridian/tree/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_core)         | Used by the offline local application                            |
| Local IMU/LiDAR, visual components, fixed-lag graph, health     | Source and focused tests in [`meridian_local_rt`](https://github.com/CharlesOural/Meridian/tree/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt) | Complete runs H–L exercised local LiDAR–IMU only                 |
| Offline bag replay, conversion, reports, trajectories           | [`meridian_bag_localize`](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_apps/src/bag_localize.cpp)        | Sole runnable composition used by Run L                          |
| Visual fusion                                                   | KLT/BRISK, landmark/factor, lane and graph tests                            | Prefix diagnostics only; no complete scored visual-fusion result |
| Sparse submaps, place verification, PCM/GNC, GNSS, global graph | Source and component tests in [`meridian_global`](https://github.com/CharlesOural/Meridian/tree/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_global)   | No executable connected local finality to this library           |
| Persistence                                                     | Seal spool and graph-journal component tests                                | No composed local outbox/global cache/optimizer resume           |
| Live ROS local/global nodes and `map -> odom`                   | Conversion and diagnostics libraries only                                   | Not runnable                                                     |
| Dense mapping                                                   | No active v2 mapping package                                                | Not exercised                                                    |
| Jetson, Quad Hard, Park                                         | No final-path run                                                           | Not established                                                  |

The sole executable is defined by
[`meridian_apps/CMakeLists.txt`](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_apps/CMakeLists.txt)
and does not depend on `meridian_global`. Run L’s report records
`lidar-imu` mode, `39,763` IMU events, `1,991` LiDAR
events, and zero camera events or visual factors. Its accuracy cannot be cited
for vision, GNSS, loop closure, sparse sealing, global consistency, dense
mapping, live ROS, or deployment hardware.

The strongest V2 assets are conventions and failure semantics:

- `T_A_B` maps coordinates from `B` into `A`;
- public right perturbations and translation-first tangents, with explicit
  private GTSAM reordering;
- signed integer-nanosecond fusion time, half-open intervals, source/clock
  epochs, and exact support ownership;
- immutable calibration/config revisions and strong typed IDs;
- observation lineage distinguishing residual, conditioning, and retrieval
  use;
- rank-aware information with exact zero unsupported directions;
- one writer per mutable owner, candidate/shadow commit, hard queue/memory
  bounds, and typed degradation.

Source entry points:
[core factor-batch metadata](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_core/include/meridian/core/factor_batch_api.hpp),
[local coordinator](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/include/meridian/local/local_estimator.hpp),
[local graph](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/include/meridian/local/graph.hpp), and
[offline app README](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_apps/README.md).

## V2 IMU and initialization lessons

### Observability before optimization

An unconstrained moving bootstrap allowed inertial bias to absorb motion. On a
short Quad Easy diagnostic it produced a single-step divergence of
`28.3 m/s` and `39.2 deg`. One shared calibrated bias prior
plus a five-sigma plausibility gate restored a physically plausible solution.
A prior belongs to the shared quantity once; duplicating it at every state
changes its information.

A historical 550/1700-event prefix then reported expected sensor/calibrated/full
behavior as data/calibrated/full ranks `189 / 191 / 195`, with two
tilt/accelerometer-bias modes resolved by calibration. This is experiment
`BRINGUP-2026-07-16-L`, not Run L’s final initialization.

Run L’s accepted final two-pass batch had scalar dimension `195`,
expected data rank `191`, actual data/calibrated/full ranks
`191 / 191 / 195`, zero prior-resolved modes,
`sensor_observable` classification, 20 segments, 440 IMU knots, and
four solver iterations. This distinction prevents a diagnostic prefix rank
from becoming the final-run claim.

Changing the Jacobi-scaled numerical rank tolerance from `1e-10` to
`1e-11` stopped three genuine roughly `4e-11` scaled modes
from being discarded. It is a classifier setting for that scaled problem, not
a physical observability constant.

V2 also raised the motion-bootstrap fault limit to 1000 LM iterations during
bring-up. Run L required four. The limit is evidence of a formerly weakly
seeded all-at-once problem, not a solver budget to preserve. V3 decomposes
dynamic initialization into gyro-bias, velocity/gravity, gravity-sphere
refinement, then one normal production replay.

Sources:
[motion initializer](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/motion_initializer.cpp),
[motion tests](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/test/test_motion_initializer.cpp), and
the resolved profile in
[`bag_localize.cpp`](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_apps/src/bag_localize.cpp).

### Static authority and startup coverage

V2 correctly made dynamic LiDAR–IMU initialization the unsupervised default.
Static initialization required an explicit, epoch-matched
`ZeroMotionPrior` from an operator, supervisor, or scenario; IMU
statistics verified but did not invent that assertion.

Run L’s first estimate was `2.000515938 s` after the first ground-truth
stamp. This is an evaluation-timeline startup delay, not measured wall-clock
initialization CPU. The internal fixed-startup-window profile excluded the
declared 2.2 s cold-start region and achieved pose/time coverage
`1.0 / 0.999997`. The publication-style full-reference profile allowed
no trim and achieved only `0.9899396 / 0.9899306`, narrowly below
`0.99`.

V3 must retain immutable startup measurements and replay them through the
production path after initialization acceptance. It must report startup
latency separately from compute time and from the chosen evaluator’s coverage
denominator.

### Calibration and units

The application composition once supplied gyro-bias random walk
`2.66e-4 rad/s^2/sqrt(Hz)` where the audited Newer College profile
used `4e-6`. Restoring `4e-6` repaired configuration
consistency, but the controlled comparison covered only short prefixes and the
historical fault appeared much later. It is not a v3 or SBG calibration result.

The V2 implementation correctly squared continuous noise densities before
passing covariance densities to GTSAM. The boundary between density, random
walk, variance, and discrete covariance remains unit-explicit and tested.
Each `SensorId / CalibrationId` resolves its own profile; Newer
College Alphasense values never transfer silently to the deployment SBG IMU.

Sources:
[Newer College calibration](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_ros/src/newer_college_calibration.cpp),
[calibration tests](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_ros/test/test_newer_college_calibration.cpp),
and [local graph config](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/include/meridian/local/graph.hpp).

### Timeline and unsupported IMU conditions

An accidental 100 ms state-request limiter suppressed the intended sensor
cadence until it was replaced by a 1 ms numerical guard. Exact sensor times,
not LiDAR arrival or a nominal knot rate, must drive the sensor-neutral
timeline. Splitting an active interval requires raw IMU reintegration, and
adjacent intervals may share a boundary sample but never elapsed duration.

V2 represented saturation, inferred missing ticks, and timestamp uncertainty,
but the graph returned typed “not implemented” errors for those intervals.
V3 needs an explicit inflated bridge, quarantine, or odometry-epoch break; it
must not silently inherit V2’s complete rejection as field recovery.

Sources:
[event scheduler](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/event_scheduler.cpp),
[IMU support](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/imu.cpp), and
[graph interval admission](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/graph.cpp).

## V2 LiDAR and local-map lessons

### Registration model history

The GICP/VGICP path accumulated normals, point covariances, association caches,
solver-specific state, and multiple representations. A diagnostic Dogleg
change reported `23/23` registrations while moving poses by less than
`5 micrometres`; proposal status and graph objective were not proving
the same thing. Prefixes later exposed hundreds of milliseconds in target,
GICP, and graph stages. No complete, scored sole-GICP trajectory established a
V2 benefit.

A provisional VGICP information ceiling of `1e3` changed the same
short-prefix result from `1.869 m / 7.307 m` ATE/RPE to
`0.167 m / 0.281 m`, proving overconfidence in the old
`1e6` scale but not calibrating `1e3`. Stateful
per-linearization caches were first isolated per candidate and then retired.
Their correctness cost was real, but shared mutable caches were not an
acceptable optimization.

The T038 path removed that surface and selected one immutable point-to-point
implementation:

- deterministic `0.30 m` registration artifacts;
- deterministic `1.50 m` source-row selection;
- bounded open-addressed exact-neighbour indices and four-way association;
- a `0.50 m` correspondence gate;
- adaptive `2.5 x MAD` Huber scale clamped to
  `[0.15, 0.50] m`;
- separate natural translation/rotation convergence at `1e-3`;
- one exclusive target winner for each source row;
- live binary factors and at most one finalized-map unary channel;
- rank projection and a physical information cap.

IMU could seed and deskew registration but remained conditioning-only in the
LiDAR batch. The graph factor did not receive a synthetic scan-matching pose.
It sealed point identities, fixed Huber weights, target geometry, supported
directions, and information scale, then compressed the accepted weighted point
objective into at most six exact sufficient-statistic residual modes. Graph
probes were O(6), stateless, and did not reassociate or reweight.

Sources:
[registration](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/lidar_registration.cpp),
[cloud/index](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/lidar_registration_cloud.cpp),
[private factor](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/direct_lidar_factor.cpp), and
[sufficient-statistic test](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/test/test_direct_lidar_factor_sufficient_statistics.cpp).

### Target-topology experiments

The first two rows below used the retired direct-surfel factor and are a
controlled topology lesson only with respect to each other. Runs H–L used
T038 direct point-to-point. The Q/R-to-H transition changed the residual,
factor, source selection, robust scale, and runtime path in addition to target
topology.

All rows used complete Quad Easy LiDAR–IMU replay and the internal evaluation
protocol.

| Experiment           | Model and target                                       | Fixed-lag ATE | Runtime evidence                                                                      | Causal lesson                                                                         |
| -------------------- | ------------------------------------------------------ | ------------: | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| BRINGUP-Q            | retired surfel, three pose-synchronized sweeps         |  `0.840655 m` | `307.427 s`, RTF `0.6473`                                                             | Correct synchronization alone did not supply enough historical geometry               |
| BRINGUP-R            | retired surfel, FIFO 15 still attributed to one pose   |  `0.511941 m` | report `467.909 s`, RTF `0.4253`; ledger separately says `468.14 s`                   | More geometry helped, but the wrong one-pose Jacobian caused 377 objective rejections |
| Run H / FOUNDATION-G | T038 pose-aware live target only                       |  `0.559663 m` | `198.087 s` for `198.998 s`, RTF `1.004599`                                           | Loss-free unpaced throughput, but active history alone did not bound drift            |
| Run I / FOUNDATION-I | broad persistent finalized base                        |  `0.072653 m` | `1159.687 s`, RTF `0.171597`; ledger-reported `1,896,024 KB` RSS; `9.798B` candidates | Broad fixed history anchored drift; the query/factor schedule was unusable            |
| Run J / FOUNDATION-J | nearest 12 finalized owners                            |  `0.118963 m` | `378.790 s`, RTF `0.525352`                                                           | A centre-biased owner bank lost spatial diversity                                     |
| Run K / FOUNDATION-K | same owners, restored scalar information               |  `0.130191 m` | `295.459 s`, RTF `0.673521`                                                           | Scalar weight cannot recreate missing geometry                                        |
| Run L / FOUNDATION-L | persistent finalized base plus pose-aware live overlay |  `0.071944 m` | `298.797685 s`, RTF `0.665997`; ledger-reported `407,836 KB` RSS                      | The two-layer topology recovered accuracy and cost, but remained slower than input    |

RTF is `sensor duration / wall time`. Every H–L report says
`pipeline_runtime.mode = unpaced`. H therefore demonstrates average
compute throughput and input completeness, not scheduled recorded-rate queue
behaviour or a per-frame deadline.

Run L also produced:

- internal online ATE `0.083586733 m` and fixed-lag ATE
  `0.071944237 m`;
- internal fixed RPE at 10 m `0.151843118 m / 0.578973795 deg`;
- publication-style online/fixed ATE
  `0.083604726 / 0.072033464 m`;
- `1,968/1,968` converged registrations, split into 984
  factor-due and 984 tracking-only scans;
- zero process, input, registration, graph, degradation, or health-transition
  loss.

It passed the historical project objective of less than `0.08 m`.
The current checked-in Quad Easy scenario instead names `0.07 m` as
its competitive gate, so `0.071944 m` does not pass that newer
threshold. The evaluator JSON’s `quality.passed` concerns coverage and
trajectory-health gates, not ATE; even inaccurate Run H passes those quality
checks. Run L is strong single-sequence internal evidence, not a SOTA, Jetson,
Quad Hard, Park, scheduled-realtime, visual, global, or release result.

### Revision and frame consistency

Immutable scan-local clouds initially retained poses from their admission
revision. After a graph update, the next target mixed new state poses with
stale scan poses. Synchronizing every active owner pose after every successful
transaction removed that defect. A rejected candidate never changes the
target.

Run L0 stopped after 50.1785 s of bag time with 480 states, 479 registrations,
and 720 graph commits. Its typed failure was:

```text
accepted direct point ICP snapshot has no canonical admission information:
direct point-to-point ICP admission snapshot metric or robust weight is stale
```

Association had frozen distance in `odom` while a live factor
revalidated the row in its target frame through a redundant relative seed. A
physically irrelevant rounding difference created two definitions of one
geometry. The repair canonicalized the seed, computed every metric in its
actual factor frame before shared MAD/Huber scaling, and advanced both snapshot
checksum schemas. V3 stores one canonical point pair and robust scale per batch
revision and rebuilds association explicitly.

Sources:
[rolling target](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/rolling_lidar_target.cpp),
[composite target](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/lidar_composite_target.cpp), and
[finalized target](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/finalized_lidar_target_map.cpp).

### Information, correlation, and finality

One failed persistent-map run applied finalized-owner covariance inflation to
both finalized and live rows. The first mixed batch scaled live LiDAR by about
`1/212` instead of `1/6`, making the graph nearly IMU-only
and ending with an ineligible partial `7.151966 m` ATE. Factor
channels require independent, testable scaling.

Raw point count is not independent information because source scans, target
points, map history, deskew, and calibration are correlated. V3 retains the
full directional shape, one independently benchmarked information ceiling for
each lineage, and exclusive source ownership. V2’s per-owner covariance
inflation machinery is not required in the first v3 build.

Active binary factors remain relinearizable while both owners are live. When
a state is marginalized, existing information enters the square-root prior and
its accepted scan may migrate to fixed `odom` geometry. The old binary
factor is not rewritten as unary; only future scans form unary rows against the
fixed base.

V2’s active and terminal factor-batch journals were bounded. Once an old
terminal record fell out of the local journal, overlap checking could no longer
inspect primary measurements whose information remained in the marginal
prior. “Marginalization seals it” is incomplete: durable lineage must be
exported before bounded local provenance is evicted.

## V2 graph, software, and runtime lessons

### Atomic candidate state and objective consistency

Candidate isolation prevented a failed registration, visual candidate, or
optimization from mutating the accepted graph, IDs, factor slots, finality, or
maps. Sensor ID, batch ID, exact times, owner states, health epoch,
covariance/information, directional observability, lineage, checksum, and
`map_eligible` are durable batch fields. A recent batch is removable
only while all of its rows remain explicit; marginalization seals it.

The graph’s safety gates are worth preserving; the custom iSAM2 machinery is
not automatically worth preserving. V2 learned that:

- the complete true nonlinear objective, not an iSAM2 status field, decides
  descent;
- physical innovation must use the newest relative transition and IMU
  repropagation, not the largest common-frame smoothing revision anywhere in
  the active window;
- tightening relinearization thresholds did not fix a stale objective
  baseline;
- a stock Dogleg switch did not fix a factor whose `error()` and
  `linearize()` represented different association snapshots;
- every accepted factor must be stateless or carry an explicit candidate-local
  cache revision protocol.

Run L nevertheless performed `2,953` graph transactions for
`1,968` scans: one navigation append per scan plus a second LiDAR
factor transaction for 984 keyframes. It recorded `2,910` rejected
full steps, `20,593` Gauss-Newton backtracking trials,
`2,513` Cauchy attempts, zero accepted Cauchy steps, and
`2,660` zero-step terminations, with no rejected transactions. This is
strong evidence to assemble one ready event into one v3 candidate transaction
and simplify no-op globalization work.

Sources:
[candidate-isolated adapter](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/candidate_isolated_isam2.cpp),
[graph implementation](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/graph.cpp), and
[graph tests](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/test/test_graph.cpp).

### Dependency capabilities and exact-preserving optimization

GTSAM 4.2a9 appended replacement marginal factors even with unused-slot reuse
enabled, so factor storage/work grew after every marginalization. Requiring
GTSAM 4.2.1 and testing 32 successive public-API leaf marginalizations bounded
the factor work. On the 5000-event diagnostic, consumer CPU changed
`93.623 -> 78.942 s` and graph mean/p95
`368.4/530.2 -> 300.2/410.7 ms` with only roundoff-scale trajectory
changes. Dependency versions should be selected by a tested capability, not an
unchecked version string or a private workaround.

Other exact-preserving changes were valuable:

- evaluating one pose per distinct LiDAR time offset changed deskew pose
  evaluations by exactly 128x and reduced the compared 1700-event consumer CPU
  from `64.765` to `20.628 s` with byte-identical trajectory;
- cost-only rejected trials, accepted-snapshot reuse, final-only sealing,
  affected-descendant re-elimination, and previous-objective reuse changed the
  compared 5000-event CPU from `63.532` to `33.632 s` within
  explicit nanometre/microdegree equivalence gates;
- disabling duplicate GTSAM nonlinear-error traversal was safe only because
  Meridian already evaluated the complete objective used by its gates.

Performance work must state the equivalence invariant before optimization and
measure mature end-to-end CPU, not only a microbenchmark.

### Keep dense payloads off the localization critical path

Replacing synchronous full raw-sweep sealing with an O(1) accepted-map ticket
changed complete wall time from `259.346 s` to `198.087 s`
with trajectory differences at numerical roundoff. Registration artifacts
remain in the local path. A downstream consumer re-deskews and seals the full
raw sweep only after localization acceptance/finality.

Sources:
[registration-cloud boundary](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/lidar_registration_cloud.cpp)
and [map payload](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/lidar_map_payload.cpp).

### Sensor health, rollback, and safe map admission

The generic registry used independent
`ACTIVE / SUSPECT / FAILED / RECOVERING` state and a monotonic
recovery epoch per sensor. V2’s complete LiDAR path:

1. evaluated failed/quarantined LiDAR in shadow mode;
2. stopped new primary factors and froze map admission;
3. could remove only a bounded recent set of still-explicit batches;
4. removed the matching rolling-target payloads in the same recovery action;
5. required good shadow results before reactivation;
6. never seeded a target from a rejected or unlocalized scan.

This is a strong contract. Only LiDAR was wired end to end. Visual and GNSS
health parity is unproven.

Sources:
[health registry](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/include/meridian/local/sensor_health.hpp),
[map admission](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/map_admission_gate.cpp), and
[coordinator recovery](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/local_estimator.cpp).

### Visual negative evidence

V2 implemented independent KLT/BRISK camera lanes, anchored log-inverse-range
landmarks, reprojection factors, identity reconciliation, and component tests.
The reusable results were failure semantics, not a complete visual accuracy
claim:

- a bounded private inverse-range latent prevented a fatal physical-range
  escape;
- a rejected visual candidate discarded its dependent seeds/factors and
  preserved the accepted graph and unrelated lanes;
- the corresponding 5000-event prefix committed zero visual factors, so
  isolation was not fusion success;
- a “visual graph disabled” experiment still decoded and tracked all cameras
  and then suffered a LiDAR target-recovery cascade, so it was neither a visual
  compute isolation nor a valid visual-value A/B;
- factors created by a just-resolved visual keyframe could not join that same
  state transaction. They waited for an unrelated later navigation append
  because no visual factor-only batch path existed;
- visual health/removal was not wired through the complete sensor-health path.

There is no complete scored visual-fusion result to carry forward. V3 should
retain per-camera lane isolation, bounded coordinates, dependency-closure
rollback, exact state ownership, and a sensor-pure same-event batch
transaction; it should not retain delayed attachment.

Sources:
[visual lane](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/visual_lane.cpp),
[visual factor](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/visual_factor.cpp),
[local visual integration tests](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/test/test_local_estimator_visual.cpp),
and the historical experiment evidence summarized in this retrospective.

### Measure complete work and input integrity

Run L stage measurements below are from the final rolling window of at most
1024 samples, except factor preparation, which contains all 984 samples. The
stages are nested or have different event populations and must not be summed.

| Stage                                  |        Mean |         p95 |          p99 |      Maximum |
| -------------------------------------- | ----------: | ----------: | -----------: | -----------: |
| Registration, including live composite | `54.674 ms` | `90.881 ms` | `111.186 ms` | `127.393 ms` |
| Live composite build                   |  `9.766 ms` | `11.973 ms` |  `12.283 ms` |  `13.285 ms` |
| Graph transaction                      | `40.130 ms` | `85.152 ms` |  `89.277 ms` |  `95.119 ms` |
| Registration view                      | `19.085 ms` | `20.457 ms` |  `21.617 ms` |  `23.760 ms` |
| Deskew                                 |  `6.180 ms` |  `7.222 ms` |   `7.432 ms` |   `7.801 ms` |
| Finalized-base update                  |  `1.561 ms` |  `7.976 ms` |   `8.367 ms` |   `9.619 ms` |
| Factor preparation                     | `13.529 ms` | `20.235 ms` |  `21.350 ms` |  `22.548 ms` |

The driver’s inclusive `processReady` p99/max were
`195.207/225.314 ms`. The report does not turn nested timings into an
exact CPU attribution. The ledger-reported process result was 174% CPU and
407,836 KB peak RSS; the Run L directory does not contain the original
`/usr/bin/time` output, so those two figures are weaker evidence than
the report timings.

Every benchmark must preserve queue depth/age, drops, rejects, skips, input
IDs, coverage, initialization, CPU, RSS, and scoring support. An apparently
faster algorithm that loses work has failed. Unpaced throughput and scheduled
queue/deadline tests are distinct gates.

Sources:
[typed timing](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_local_rt/src/pipeline_timing_internal.hpp),
[run-report writer](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_apps/src/bag_localize.cpp), and
[benchmark scenarios](../benchmarks/scenarios/README.md).

## V2 global and persistence component lessons

V2’s global source is real component code, but no application constructed it.
The implemented library included sparse-submap types, visual and LiDAR
retrieval/verification, PCM, exact maximum clique, GNC-TLS, GNSS alignment/FSM,
a bounded full-shadow global graph, graph journal, and sparse-seal spool.

The intended and implemented boundaries diverged:

- there was no authoritative local-finality-to-condensation/seal producer;
- no process composed the local outbox, transport, global seal cache, global
  coordinator, and `map -> odom` publisher;
- only the first `OdomEpoch` could enter the connected graph; later
  epochs remained `PendingUnconnected` because the verified connector
  transaction was absent;
- the spool lived in `meridian_global` even though the producer-side
  outbox belongs beside local finality;
- the journal recovered immutable checkpoints for inspection/replay but had no
  graph restore/install API for resuming active optimizer state;
- child durable-blob ownership and the global seal cache were absent;
- the full-boundary adjacent navigation adapter existed privately but was not
  connected to the public pose-oriented graph.

The reusable persistence implementation did establish versioned canonical
encoding, SHA-256 identity, bounded reads, file and directory `fsync`,
same-directory temporary files, atomic no-replace rename, prepared/committed
records, prefix recovery, and explicit corruption/capacity failures. Durable
state claims require the complete ownership and resume path, not only correct
file primitives.

The global algorithms remain candidates for extraction behind v3 contracts.
Their component tests do not authorize statements about global ATE, loop
precision/recall, GNSS field robustness, crash-resumed SLAM, or live
`map -> odom`.

Sources:
[global coordinator](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_global/include/meridian/global/global_coordinator.hpp),
[global graph](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_global/src/graph.cpp),
[loop consensus](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_global/src/loop_consensus.cpp),
[GNSS FSM](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_global/src/gnss_fsm.cpp),
[seal spool](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_global/src/seal_spool.cpp),
[graph journal](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_global/src/graph_journal.cpp), and
[implementation provenance](https://github.com/CharlesOural/Meridian/blob/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44/src/meridian_global/REFERENCE_PROVENANCE.md).

## V2 artifact identity

These are the exact bytes inspected during this retrospective. The paths are
ephemeral. The table identifies evidence but does not make it durable.

| Run | Artifact directory | `run_report.json` SHA-256 | Evaluation SHA-256 |
| --- | --- | --- | --- |
| BRINGUP-Q | `/tmp/meridian_qe_target_sync_full` | `2e4d057b692bc9be8317ae73e52aa63ff4e84dda3f44b64c45811735eee6998e` | internal: `da1b6beb37fc1b02f0d06d36928f6d881863dcdaf3b3e027ce8cd1f6599e8d23` |
| BRINGUP-R | `/tmp/meridian_qe_target15_full` | `05341626178764be1e8e3de95319d7897185edc64af2a41b5baf71d07589416f` | internal: `15f5d146752c29454a6a3b71b44aeea82f5329cdaf3a53a8b3efcb0e42a6150e` |
| H | `/tmp/meridian-qe-direct-p2p-full-20260717-h-runtime-refactor` | `106f6c94ca6221391015900f74f44b37a450ed9ef00ca8b9b9ea744e0c5da082` | online `12e5feea5e8ea8b746f83040ce454a78e36bdb3f3e0470d8e8652337181c90ea`; fixed `8f4f6ec50c1c7ca0504d9ca41c1604bc201d4789d642378214e76632ee110e93` |
| I | `/tmp/meridian-qe-t039-rev3-full-20260717-b` | `e7f3ae4449e61cb2dfc2719a47ddb15d6fa2493a2aebc7f11d3c6a4d976a9dd7` | online `5967e1c9c552ad125f2ebed5c05925abbf11bef00fb383cd6b00278392735f27`; fixed `447800e15b7f0069402174eb7577ac21f0b379565ff42f3489f1cfeff0d961b8` |
| J | `/tmp/meridian-qe-t040-composite-full-20260717-a` | `ec93ea82da9fdb4117505d651306231c6e5768d2d55041a6db3a98a55da1ab69` | online `efa92a887bd9bdac2f62227761cbebd3125678216a462efe3be8fbadb7bfc3b2`; fixed `f3855aee63d5477805bf8a9f872e12f8c46dda5c8710dff266678ead9b32fda4` |
| K | `/tmp/meridian-qe-t041-correlation-streaming-full-20260718-a` | `3902410b68d52a4e26efc6b5dee6a21ae9448bc02748b283f265cd668bfc5212` | online `f283561a0a9ff2fe7b34847ba5c50c5ce6b45b038519e718aa21cb2fa1bae1cc`; fixed `80689351d740eb681a57d348abaf76a63f1c93259396d7b972f373a978dfe676` |
| L0 | `/tmp/meridian-qe-t042-persistent-overlay-full-20260718-a` | `79e80008f0fee85d0bd13aa02f4243b08fc5791fa1b1134191b371ebd493dd07` | incomplete; no evaluation |
| L | `/tmp/meridian-qe-t042-persistent-overlay-full-20260718-b` | `3bc8d6e9565a8735f3fb99ba9ecb5566b442ab4b04e63668d66081c842b8d3c9` | internal online `14c8263a3010c77bd1cdbed78a31d5c0070a5f84611d7085727ba88c1b7fbcd6`; internal fixed `09c6c84c7e38410d49b943041145d9e0994f06c285fa6738c8384810927b6d5c`; published online `f6d0e44dd28106f8b46cb40bd164f9ff877f7b3c933331c42a856fd586753290`; published fixed `56c82edaed0231e723251e6f1a1b1ff98059f75b48508480478ab1faaad339eb` |

The H directory additionally contains the only preserved resource record among
these final runs. Runs I/J/K/L RSS and CPU percentages in the decision ledger
must remain labelled ledger-reported unless their original resource output is
recovered.

## Extraction register for v3

### Keep as contracts or source-reading material

- exact frame direction, tangent order, unit, calibration, and time semantics;
- independent sensor queues and exact-time state ownership;
- explicit supervised static authority and bounded dynamic initialization;
- midpoint propagation/preintegration with raw interval ownership;
- v1/post-v1 deterministic bounded voxel search and direct-registration
  stress-test lessons;
- scan-local geometry with synchronized owner poses;
- exclusive source-row ownership and canonical factor-frame metrics;
- accepted-localization map admission and finality as the only live-to-fixed
  transition;
- sensor-pure factor batches, complete lineage, per-channel information, and
  per-sensor recovery epochs;
- complete-objective/physical gates and candidate isolation;
- bounded queues, maps, factor journals, persistence, and exact failure
  dispositions;
- V1 PCM/GNSS/tangent tests and V2 global algorithms as narrow component
  extraction candidates;
- retained-source provenance and deterministic rebuild for non-invertible dense
  products;
- one live/replay core, explicit replay mode, complete input reconciliation,
  fixed scenario/evaluation profile, and full-run artifacts.

### Do not port as hidden compatibility

- V1 LiDAR-grouped scheduling, heuristic Euler/ZOH tracker, constant-screw
  production deskew, world-frame mutable registration map, relative-pose
  keyframe filter, correction feedback, or mid-run absolute-prior mismatch;
- raw Hessian-as-covariance, fixed robust scale, hard iteration truncation, or
  single-pose attribution of multi-owner geometry;
- V1 pose-only global backend, duplicated restart IMU summary, corrected-XY
  loop eligibility, greedy fallback presented as exact PCM, or globally
  mutable mission-length keyframe/dense map;
- V2 GICP/VGICP/mutable-cache compatibility paths;
- V2 two graph transactions per LiDAR keyframe, delayed visual attachment,
  per-owner inflation machinery, no-op globalization burden, or uncalibrated
  scalar information values;
- component-only global/persistence code described as integrated runtime;
- any v1/v2 threshold promoted without a resolved sensor profile and a fresh
  complete-sequence benchmark.

## V3 invariants carried from the evidence

1. Version claims name an exact source object; run claims name source, resolved
   configuration/calibration, input checksum, replay mode, evaluator profile,
   and artifact identity.
2. `StateTimeline` is sensor-neutral and owns exact adjacent IMU
   support. Frontend queues are independent; a synchronized all-sensor bundle
   is not a scheduling primitive.
3. Static initialization requires explicit same-epoch zero-motion authority.
   Unsupervised startup is a bounded observable dynamic solve whose accepted
   raw measurements are replayed once through the normal path.
4. Optional frontend batches are sensor-pure. Conditioning evidence is explicit
   lineage, not a duplicate residual or prior.
5. One ready event creates one complete candidate transaction containing its
   requested state, exact IMU interval, and optional sensor batch.
6. Candidate estimator state, complete sensor batch, and staged target delta
   commit atomically or leave the accepted state and registration target
   unchanged.
7. Active LiDAR geometry retains immutable owner-local points and resolves
   their current owner poses at use time. Finalization is the only transition
   to fixed `odom` geometry.
8. Every source point has one target winner, one canonical factor-frame metric,
   and one residual lineage. Association and factor evaluation never define
   nominally equivalent geometry twice.
9. Directional information, robust scale, and correlation treatment have one
   tested owner. Live and finalized channels never share an accidental scalar.
10. Rejected or unhealthy measurements never seed localization or map
    geometry. Recent rollback removes explicit factors and their payloads
    together; marginalized evidence remains sealed and durably traceable.
11. Local `odom` stays continuous. Global correction is expressed
    through `map -> odom` and never rewrites local deskew, matching, or
    finalized geometry.
12. Non-invertible fused map products are disposable, provenance-derived
    caches; authoritative owner-local source data is retained until the rebuild
    policy permits deletion.
13. Component tests, offline local replay, scheduled realtime, live ROS,
    global-SLAM, restart recovery, dense mapping, Jetson, and publication
    benchmarks are separate capabilities and are never implied by one another.
14. Full-sequence ATE/RPE and coverage, queues and input IDs, stage
    p50/p95/p99/max, CPU, RSS, health transitions, and durable artifacts select
    a production decision. Short prefixes diagnose only.
15. Paper and legacy code are implementation context. Meridian owns frame,
    time, noise, API, provenance, failure, and validation semantics.
