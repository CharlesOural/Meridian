# Meridian v3 — System Specification

## Mission and evaluation context

Meridian v3 is a localization and mapping framework for the Shark Robotics
[Barakuda](https://www.shark-robotics.com/fr/barakuda-mule-robot/), a heavy
four-wheel-drive ground robot operating off road. The target environment
includes vegetation, mud, rain, rough terrain, wheel slip, shocks, and sustained
mechanical vibration.

The target sensor and compute configuration is:

- Ouster OS1-128 LiDAR at 10 Hz;
- SBG Ellipse-D six-degree-of-freedom inertial measurements at 200 Hz; and
- NVIDIA Jetson AGX Orin 64 GB with 12 CPU cores.

The first implementation is optimized for correctness, robustness, and clarity
on the development machine while sustaining the 10 Hz LiDAR input. Jetson
profiling and hardware-specific optimization follow once the algorithms are
validated.

A mission lasts at least three hours and may cover more than 10 km. Local pose
continuity, bounded long-distance drift, recovery from degraded sensing, and a
globally georeferenced result are therefore important.

Initial evaluation uses the complete datasets and calibration data under
`bags/newer-college/`. Full-sequence ATE and RPE are the primary localization
metrics, accompanied by runtime and input-queue integrity. Runtime factor is
`RTF = sensor duration / wall duration`; `RTF >= 1` sustains recorded rate.

Dataset evaluation deliberately exercises the deployed ROS input path. A
generic launch starts Meridian and Foxglove Bridge and invokes the standard
`ros2 bag play` process. Meridian records a lightweight Rerun `.rrd` containing
converted input evidence and stage telemetry, extended with the estimated
trajectory as estimator layers land. The Newer College ground truth under
`bags/newer-college/gt/` remains the accuracy reference and the original bag
remains the raw sensor authority.

Evaluation is repeatable by launching the bag again; bit-for-bit scheduling is
not a requirement. Small run-to-run uncertainty is acceptable and is reported
alongside the metrics. There is no DDS-bypassing bag reader or separate replay
executable: normal ROS transport, conversion, queues, and estimator behavior
are exercised together because that is the system that must work live. Topic
delivery/count comparison is an optional post-run check against rosbag2
metadata, not an in-process acceptance mechanism.

Sensor parameters are selected by explicit `SensorId` and `CalibrationId`.
Newer College evaluation uses its OS0-128 and Alphasense IMU calibration
profile; deployment uses the OS1-128 and SBG Ellipse-D profile. Timing, noise,
extrinsics, and vibration gates are validated independently for each profile;
no value measured on one platform is silently transferred to the other.

## Coordinate frames

`T_A_B` transforms coordinates expressed in frame `B` into frame `A`. Meridian
uses the frame tree

```text
earth -> map -> odom -> base_link -> sensor frames
```

- `earth` is the Earth-centered, Earth-fixed WGS84 frame. It is a geodetic
  parent, not a working frame for point clouds or local optimization.
- `map` is a local WGS84 tangent frame with East-North-Up axes. Its datum and
  axes remain fixed for one map epoch. Global corrections may change
  `map -> odom`.
- `odom` is the continuous local-estimation frame. `odom -> base_link` remains
  smooth through global corrections.
- `base_link` is the rigid robot reference with x forward, y left, and z up.
- `imu_link`, `lidar_link`, `gnss_link`, and future sensor frames use calibrated
  static transforms from `base_link`.
- `base_footprint` may be published as a ground-projected convenience frame; it
  is not an estimator state.

The composed global robot pose is

```text
T_earth_base = T_earth_map * T_map_odom * T_odom_base
```

The datum fixes `earth -> map` for one map epoch. This transform is published
only after datum commitment and is then immutable. A new datum starts a new map
epoch. `map -> odom` establishes the initial geographic alignment and later
carries global corrections. Local odometry is not reset when that estimate
changes.

One running localization session owns one immutable datum. Changing it requires
a new session; a process that must expose both epochs uses an epoch-qualified
child frame such as `map_2`, rather than publishing a second static transform
for the same `map` child.

Before geographic alignment is valid, Meridian publishes the local
`odom -> base_link` tree without an identity placeholder for `earth -> map` or
`map -> odom`. A configured geographic fallback ensures that operation without
GNSS can still become georeferenced after startup validation.

### ROS localization outputs

ROS topics and TF expose both the continuous local result and its georeferenced
composition. One authority owns each transform edge.

| Output | ROS type | Semantics |
| --- | --- | --- |
| `/tf` | `tf2_msgs/msg/TFMessage` | Dynamic `odom -> base_link` and valid `map -> odom` transforms |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | Sensor extrinsics and committed `earth -> map` |
| `/meridian/local/odometry` | `nav_msgs/msg/Odometry` | Causal local estimate in `odom`, child `base_link` |
| `/meridian/global/odometry` | `nav_msgs/msg/Odometry` | Composed estimate in `map`, published after geographic alignment |
| `/meridian/global/fix` | `sensor_msgs/msg/NavSatFix` | Georeferenced antenna estimate in WGS84 with ENU covariance |
| `/meridian/localization/status` | Meridian typed message | Local and geographic validity, initialization source, map epoch, estimate revision, and sensor states |

`/meridian/local/odometry` is the causal high-rate propagation from the newest
accepted estimator state. Each accepted graph revision restarts propagation
from that optimized state; output timestamps never move backward. Its
covariance combines the optimized anchor covariance with subsequent IMU
propagation uncertainty, while localization status carries the source revision.

Foxglove derives a trajectory trail from odometry. Mission-length
`nav_msgs/msg/Path` accumulation is therefore unnecessary; a later bounded and
downsampled path may be added only as a visualization output.

### Pose ownership and future global correction

`meridian_local_rt` is the sole source of the continuous local pose
`T_odom_base`. A separate revisioned `MapOdomEstimate` supplies
`T_map_odom`. `meridian_ros` is the sole TF publisher and composes the global
pose as

\[
T^M_B = T^M_O T^O_B.
\]

In the first implementation, the startup georeference coordinator produces the
`MapOdomEstimate`. A future global backend will replace that producer behind the
same typed API; the two producers never run concurrently. A correction changes
`map -> odom`, while `odom -> base_link` remains continuous. Consequently,
`/meridian/global/odometry` is the current composition at its publication time,
not a republished globally smoothed history.

A future backend will keep accepted geometry in immutable local submaps. For a
reference submap `s`, it derives

\[
T^M_O = T^M_{S_s}(T^O_{S_s})^{-1}.
\]

Loop closure or GNSS may revise `T_map_submap`; the points remain in the submap
frame and move globally through composition rather than being rewritten. A
geometrically verified match against an existing submap is a loop closure
during normal operation and a relocalization measurement after loss or restart.
It updates the global alignment and does not inject global points into the
bounded local registration target.

## Operational visibility and deep debugging

Standard ROS topics and TF are the operational system API and the source used
by other robot components and Foxglove. Algorithm instrumentation is recorded
in Rerun `.rrd` format. Debug recording has no authority over estimation, map
state, or ROS publication, and Meridian does not publish Rerun-only previews or
diagnostics as ROS topics.

### Evaluation record and flow

The RRD is a compact evidence record, not a repackaged bag and not an estimator
checkpoint. The first ingress slice stores full-rate scalar IMU evidence,
per-scan LiDAR metadata, conversion failures, recorder health, and a 1 Hz
point-cloud preview capped by its point count. The preview is never published
on ROS, and raw LiDAR payloads stay only in the source bag. Later slices add
timestamped accepted local/global poses,
covariance, estimator revision, initialization state, stage timing, and queue
health without changing this ownership boundary.

```text
ros2 bag play
  -> ROS 2 subscriptions
  -> meridian_ros conversion
  -> owned ROS-free observations
  -> ObservationCallbacks -> future LocalPipeline
  -> future ROS odometry, TF, and status
  -> neutral debug sink -> compact run.rrd

run.rrd estimated trajectory + integrity counters --+
Newer College timestamped ground truth -------------+-> evaluation tool
                                                       -> ATE / RPE / coverage report
                                                       -> Rerun comparison view
```

The current trajectory evaluator consumes explicit estimate and ground-truth
TUM files through a direct CLI. It associates timestamps, aligns only the
permitted gauge, computes ATE and distance-based RPE, and reports coverage
before accuracy. When trajectory entities are added to the RRD, a small
extractor may query them through supported Rerun APIs; the metric implementation
remains independent of Rerun's private file layout.

Post-run tooling may compare accepted RRD rows with rosbag2 topic counts and
report conversion/queue failures, trajectory coverage, and runtime factor.
These checks remain outside the online pipeline. Repeating the standard bag
launch provides the desired practical repeatability; measured run-to-run
dispersion is reported when material, without requiring bit-for-bit execution.

## Software structure

This section fixes ownership and data flow. The named record semantics are
stable; private layouts may evolve while implementing and benchmarking them.

```text
src/
  meridian_cmake/       shared build and test helpers
  meridian_core/        ROS-free frames, time, math, IDs, and common records
  meridian_local_rt/    planned ROS-free local localization pipeline
  meridian_ros/         ROS-to-core conversion and subscriptions; later ROS outputs
  meridian_apps/        live composition, bag-session lifecycle, and Rerun debug IO

tools/                  dataset inspection and evaluation utilities
```

Additional packages are introduced only with their complete vertical slice.
The implemented first slice contains `meridian_core`, `meridian_ros`, and
`meridian_apps`; `meridian_local_rt` and georeferencing components are planned,
not current code.

The current `meridian_apps` executable loads ROS parameters, constructs the
Rerun sink and generic ingress node, runs the executor, and drains both workers
at shutdown. Owned observations cross `ObservationCallbacks`, the narrow seam
where the future ROS-free local pipeline will attach.

The Meridian executable always consumes normal ROS subscriptions. Recorded
datasets use standard `ros2 bag play` and therefore exercise normal ROS 2
transport, the executor, conversion, queues, and the local pipeline as a live
sensor session does. The session launch owns process lifecycle and the RRD
records what reaches Meridian's converted input boundary. Post-run comparison
with bag metadata exposes transport or scheduling variation without maintaining
a second ingestion path.

### Local package organization

`meridian_local_rt` is one package with focused internal libraries and folders:

```text
initialization/
imu/
lidar/preprocess/
lidar/deskew/
lidar/registration/
lidar/target/
vision/preprocess/       added with the visual frontend
vision/tracking/         added with the visual frontend
estimator/timeline/
estimator/factors/
estimator/marginalization/
reliability/
runtime/
```

ROS image decoding belongs to `meridian_ros`; photometric processing such as
vignetting correction, rectification, and image preparation belongs to
`meridian_local_rt/vision/preprocess` when the selected visual method requires
it.

### Data flow and ownership

```text
ROS message
  -> meridian_ros conversion
  -> typed observation
  -> LocalPipeline ingestion and independent bounded sensor queue
  -> initialization / preprocessing / deskew / frontend
  -> local estimator update
  -> accepted pose commit
  -> accepted payload admission into the active registration overlay
  -> state finalization and immutable finalized-state/scan stream
  -> active-owner removal and accepted payload insertion into the fixed voxel base
  -> immutable local outputs
  -> ROS publication and optional Rerun debug recording
```

`meridian_ros` validates ROS message representation. `meridian_local_rt` owns
the authoritative timestamp, calibration, physical-value, queue, and algorithm
validation. It owns all sensor queues, the scheduler, estimator state, and local
registration map. Raw observations become immutable after admission; large
buffers transfer ownership without avoidable copies. One local writer commits
estimator and map state.

The scheduler advances from measurement timestamps and per-queue watermarks.
IMU samples are admitted continuously. A LiDAR sweep becomes ready only when
valid IMU support brackets every time needed by deskew and its requested state;
it never waits for camera, GNSS, or any unrelated queue. The earliest ready
frontend event is processed deterministically. A late event inside the active
lag may request a state and exact interval reintegration; one older than the
sealed boundary is rejected as `TooLateForActiveLag`.

The current ingress has a count-bounded LiDAR conversion queue and a separate
count-bounded Rerun writer queue; IMU conversion runs directly in its callback.
Queue rejection is recorded as an ingress failure. When bag playback exits,
the session stops admission, drains both queues, writes recorder drop/error
counters, and flushes the RRD. Byte/age bounds and algorithm-support tail
handling belong to later estimator queues and are not claimed by this slice.

The initial stable record set is:

- `ObservationHeader`: sensor identity, measurement time, sequence, frame, and
  the configuration/calibration identity needed to interpret the observation;
- `ImuSample`;
- `LidarSweep`, including the sweep interval and per-point time semantics;
- `GnssFix`, including antenna-frame time, WGS84 position, ENU covariance,
  quality, and datum/receiver provenance;
- `NavigationState`: `T_odom_imu`, velocity in `odom`, and IMU biases at one
  time;
- `PreintegratedImu`: exact support, preintegrated motion, bias Jacobians,
  covariance, quality, and sample provenance for one adjacent state interval;
- `InitializationResult`: one validated local initial state and its support,
  uncertainty, method, observability, and provenance;
- `LidarRegistrationFrame`: one immutable deskewed scan representation tied to
  a requested state;
- `FinalizedState`: one state leaving the active lag at an estimator revision;
- `FinalizedLidarFrame`: accepted scan-local geometry paired with its finalized
  state, calibration, odom epoch, quality, and provenance; the bounded local
  voxel base consumes it now and a future submap builder may consume the same
  immutable payload;
- `TerminalTrajectorySnapshot`: the accepted states still active when an
  evaluation session shuts down cleanly, used only to complete evaluation
  output;
- `LocalEstimate`: current causal estimate and validity;
- `LocalStatus`: initialization, input, queue, and localization health;
- `GeographicDatum`, `MapOdomEstimate`, and `GeoreferenceStatus`: immutable
  map-epoch origin plus the revisioned source, uncertainty, and validity of the
  geographic alignment; and
- typed frontend results such as `LidarFrontendResult` and, later,
  `VisualFrontendResult`, carrying that frontend's result, uncertainty,
  directional observability, quality, and outcome.

An accepted frontend result creates an immutable, sensor-pure `FactorBatch`.
It carries sensor and batch IDs, exact measurement and support timestamps, the
complete covariance or square-root information representation, directional
observability, sensor health and recovery epoch, measurement provenance, and
`map_eligible`. IMU may condition a frontend's deskew or tracking, while the
resulting optional sensor batch contains only that frontend's measurement
residuals. Adjacent IMU preintegration is represented once on the common
navigation timeline.

The initial public facade is deliberately small:

```cpp
class LocalPipeline {
 public:
  Admission ingest(ImuSample sample);
  Admission ingest(LidarSweep sweep);
  ProcessResult processReady();

  LocalEstimate latestEstimate() const;
  std::vector<LocalUpdate> takeUpdates();
  LocalStatus status() const;
};
```

Future sensor inputs extend this facade with typed `ingest` overloads and
independent queues. Internal frontends remain explicit typed modules rather
than runtime plugins.

### Local estimator choice

Fixed lag describes how much state remains active, a factor graph describes the
objective, and Levenberg-Marquardt describes how that objective is solved. A
batch sliding window is therefore a fixed-lag factor graph; `batch` means that
the complete active objective is solved on an accepted transaction, not that
sensor messages are synchronized into an all-modalities bundle.

| Estimator architecture | Representative systems | Main strength | Meridian tradeoff | v3 use |
| --- | --- | --- | --- | --- |
| Iterated error-state filter | [FAST-LIO2](https://arxiv.org/abs/2107.06829), [FAST-LIVO2](https://arxiv.org/abs/2408.14035) | Very high-rate direct fusion with bounded state dimension | Measurement ordering, immediate information absorption, and map mutation make delayed optional factors, recent-factor removal, and multi-pose relinearization more stateful | Accuracy and runtime reference |
| Incremental fixed-lag smoother with iSAM2 | [GLIM](https://arxiv.org/abs/2407.10344) | Reuses Bayes-tree elimination and selected linearizations as new factors arrive | Factor deletion, candidate isolation, marginalization, relinearization policy, and recovery remain coupled to persistent solver state | Solver challenger if profiling later identifies the active solve as the bottleneck |
| Batch sliding-window nonlinear least squares | [OKVIS](https://doi.org/10.1177/0278364914554813), [VINS-Mono](https://arxiv.org/abs/1708.03852), [GVINS](https://arxiv.org/abs/2103.07899), and the unified window in [Ultra-Fusion](https://arxiv.org/abs/2606.21223) | One inspectable objective, complete relinearization of explicit active factors, mature trust-region LM, and simple atomic admission/removal | Rebuilds and resolves the bounded active problem around a fixed linear marginal prior, so state and residual budgets are explicit | **Selected production architecture** |
| Full-trajectory batch optimization | Offline smoothing and mapping systems | Retrospective relinearization of the complete history | Compute and memory grow with mission length | Evaluation or a future global layer |
| Continuous-time trajectory | SLICT, CLINS, Coco-LIC | Native arbitrary-time trajectory evaluation | Additional knot, support, calibration, convergence, and marginalization complexity | Outside the first implementation |

GLIM is a reference for discrete fixed-lag state ownership, asynchronous factor
insertion, and pose-aware multi-scan constraints. Meridian does not inherit its
GPU VGICP implementation or iSAM2 choice. Ultra-Fusion is a reference for a
unified asynchronous window and factor-wise reliability; its current public
release does not provide source code and therefore cannot be the sole
implementation reference.

### Selected batch sliding window

The ROS-free estimator owns a sensor-neutral `StateTimeline`. LiDAR scans,
visual keyframes, and future modalities may request discrete navigation states
`x_k = {X_k, V_k, B_k}` at their own timestamps. Consecutive states are joined
by one manifold IMU-preintegration interval with exact sample support, following
[Forster et al.](https://arxiv.org/abs/1512.02363). LiDAR arrival is not the
clock or authority of the timeline.

The active objective is assembled from the marginal or initialization prior,
adjacent IMU residuals, and every accepted optional sensor batch whose states
remain live. Each estimator update is a candidate transaction against an
expected revision: add requested states and ready batches, remove a bounded set
of recent faulty batches when required, warm-start from the last accepted
solution, solve the complete active objective with bounded trust-region LM, and
validate the result. When an event also changes the LiDAR target, its bounded
`TargetDelta` is staged against the same expected estimator and target
revisions after the candidate solve. A successful transaction commits states,
factor provenance, target ownership/finalization, and both revisions through
the single local writer. A failed solve or unstaged delta leaves the accepted
estimator and registration target unchanged; a separately valid IMU-only
transition may then be proposed.

Window duration, state count, LiDAR residual count, nonlinear iteration bound,
and time budget are benchmark decisions. They are fixed only after
full-sequence accuracy, p95/p99 latency, memory, and queue-integrity
measurements. Meridian carries one production estimator path at a time.

After an accepted solve, states outside the selected lag are marginalized into
a square-root prior on the retained boundary, following the numerical lessons
of [square-root marginalization](https://arxiv.org/abs/2109.02182). Their final
local estimates are appended to an immutable trajectory stream, allowing
full-sequence ATE and RPE without growing the optimization window. A batch is
removable while all of its information is still explicit in the active window;
after it enters the marginal prior, its provenance becomes sealed.

The marginal prior is an immutable typed record containing retained `StateId`s,
the fixed linearization states and tangent charts, square-root matrix `A`,
right-hand side `b`, rank/nullspace diagnostics, and absorbed batch provenance.
Its residual is evaluated as `A * delta(x, x0) - b`. Explicit active factors are
relinearized on every batch solve; absorbed factors are not reconstructed or
silently relinearized. This fixed-linearization, first-estimate-Jacobian rule
preserves the gauge and the Schur complement that were actually marginalized.
Prior-chart displacement is monitored and can reject a candidate that leaves
the validated local chart.

### Ceres implementation references

The bounded batch window is solved with Ceres LM. Its concrete parameter-block,
manifold, linear-solver, loss, and stopping choices are made when this slice is
implemented and benchmarked. Before writing it, the implementation work starts
by reading these paper-code seams:

| Reference code | Files to inspect | What Meridian checks there |
| --- | --- | --- |
| [GTSAM navigation](https://borglab.github.io/gtsam/combinedimufactor/) | `PreintegratedCombinedMeasurements`, `CombinedImuFactor`, and their tests in the selected GTSAM source | Combined 15-D preintegration, bias correction, covariance, residuals, and Jacobians exposed through the private Ceres adapter |
| [OKVIS2-X](https://github.com/ethz-mrl/OKVIS2-X) | `../slam-reference/OKVIS2-X/okvis_ceres/src/{ViGraph,ViSlamBackend,ImuError,PoseLocalParameterization}.cpp` and `test/{TestImuError,TestMarginalization}.cpp` | Ceres problem ownership, sliding-window lifecycle, manifolds, IMU residuals, marginalization, and tests |
| [GVINS](https://github.com/HKUST-Aerial-Robotics/GVINS) | `../slam-reference/GVINS/estimator/src/estimator.cpp` and `factor/{imu_factor.h,integration_base.h,marginalization_factor.cpp,pose_local_parameterization.cpp}` | Window assembly, preintegration residuals, fixed-linearization marginal prior, and solve/marginalize ordering |
| [Super Odometry](https://github.com/superxslam/SuperOdom) | `../slam-reference/SuperOdom/super_odometry/src/LidarProcess/{LidarSlam,pose_local_parameterization}.cpp` and `include/super_odometry/LidarProcess/factor/lidarOptimization.h` | Direct LiDAR residual wiring and a bounded Ceres registration solve |
| [D-LI-Init](https://arxiv.org/abs/2504.01451) | `../slam-reference/D-LI-Init/include/Dynamic_init/Dynamic_init.cpp` | Dynamic gyro-bias, velocity, and gravity alignment stages used by the selected initializer |
| [LI-Init](https://github.com/hku-mars/LiDAR_IMU_Init) | `../slam-reference/LiDAR_IMU_Init/include/LI_init/{LI_init.cpp,LI_init.h}` | Commissioning reference for staged spatiotemporal LiDAR-IMU calibration |

The first entry points are `ViGraph::optimise`,
`ViSlamBackend::optimiseRealtimeGraph`, and `ImuError::EvaluateWithMinimalJacobians`
in OKVIS2-X; `Estimator::optimization`, `MarginalizationInfo::marginalize`, and
`MarginalizationFactor::Evaluate` in GVINS;
`LidarSLAM::solveOptimizationProblem` and `EdgeAnalyticCostFunction` in Super
Odometry; `Dynamic_init::{solve_Rot_bias_gyro,LinearAlignment_withoutba}` in
D-LI-Init; and `LI_Init::{solve_Rotation_only,solve_Rot_bias_gyro,
solve_trans_biasacc_grav}` in LI-Init for commissioning context.

These repositories are implementation context, not templates to copy blindly:
OKVIS2-X and GVINS contain visual-landmark graph structures, Super Odometry is
feature based, and several still use Ceres' older local-parameterization API.
Meridian rederives frame and perturbation conventions, uses the current public
Ceres API, and validates every residual and marginal prior with the tests
specified in its corresponding section.

### Canonical navigation state and inertial model

The optimized pose is the IMU pose. At timestamp `t_k`, the named state is

```text
NavigationState
  id
  time
  T_odom_imu
  velocity_odom
  gyro_bias_imu
  accel_bias_imu
```

`odom -> base_link` is derived from this state and the calibrated static
`T_imu_base`; frontend residuals use the corresponding calibrated sensor
transform. No positional array ordering crosses a module API.

The initialization gauge is defined on the robot frame, not on the displaced
IMU. At anchor time, `T_odom_base` has zero translation and zero yaw, with roll
and pitch aligned to gravity; the optimized seed is then
`T_odom_imu = T_odom_base * T_base_imu`. This preserves the calibrated lever arm
while keeping ROS and estimator frames consistent.

The internal pose update is the documented right perturbation

\[
T^O_I(\delta\xi)=T^O_I\operatorname{Exp}(\delta\xi),
\qquad
\delta\xi=\begin{bmatrix}\delta\rho_I&\delta\theta_I\end{bmatrix}^{T},
\]

with translation first. `NavigationCovariance`, `PoseInformation`, and bias
covariance types expose named blocks and perform explicit chart conversion;
raw `Matrix15` or `Matrix6` objects do not cross an API.

With `R_odom_imu` rotating vectors from IMU into `odom`, the measurement model
is

\[
\tilde{\omega}=\omega+b_g+n_g,
\qquad
\tilde{a}=R_{O I}^{T}(a_O-g_O)+b_a+n_a,
\]

and, after local initialization,

\[
g_O=\begin{bmatrix}0&0&-g\end{bmatrix}^{T}.
\]

Local position and yaw are gauges fixed at initialization. Gravity magnitude
comes from the configured physical model; initialization estimates its
direction and validates its magnitude. Geospatial heading changes
`map -> odom`, not this local gauge.

### IMU ingestion, propagation, and preintegration

#### Selected model

Each adjacent state pair is joined by one combined 15-D inertial constraint,
following the preintegration model of
[Forster et al.](https://arxiv.org/abs/1512.02363). The first implementation
reuses `gtsam::PreintegratedCombinedMeasurements` and the residual/Jacobian
semantics of `gtsam::CombinedImuFactor` behind a private adapter. GTSAM provides
the preintegrated deltas, first-order bias correction, full covariance including
motion-bias correlations, and factor Jacobians. There is no additional bias
`BetweenFactor`.

Measurement noise and bias random-walk densities enter every interval. The
independent initial bias uncertainty remains one initialization prior and is
not copied into each preintegration edge.

In GTSAM parameter terms, accelerometer and gyroscope bias random-walk
covariances describe the interval process. `biasAccOmegaInt`, when exposed by
the selected GTSAM version, is the separate uncertainty of the bias used during
integration; it is not the initial-state bias prior. The adapter names, units,
and resolved values for these quantities explicitly.

The local nonlinear problem remains a Ceres batch window. A private
`ceres::CostFunction` converts the named Meridian pose, velocity, and bias
blocks to the GTSAM navigation types, evaluates the combined factor, applies its
full whitening, and converts the local Jacobians to the Meridian Ceres
manifold. GTSAM keys, `Values`, factor-graph objects, and matrix orderings never
cross the module API. Rotation/translation tangent order and GTSAM's
accelerometer/gyroscope bias order are converted explicitly and covered by
round-trip tests.

The concrete GTSAM preintegration realization is explicit in the resolved build
and runtime diagnostics; it is never inherited silently from a library compile
default. The initial target is GTSAM's manifold realization corresponding to
the Forster model, using its public templated API when available. Any supported
version substitution must first pass the same residual, Jacobian, covariance,
and trajectory tests.

For interval `i -> j`, let the stored deltas be linearized at
`bar_b = {bar_b_g, bar_b_a}` and let `delta_b = b_i - bar_b`. The corrected
motion residual is

\[
\begin{aligned}
r_R &= \operatorname{Log}\!\left(
  (\Delta\bar R_{ij}\operatorname{Exp}(J_{R b_g}\delta b_g))^T
  R_i^T R_j\right),\\
r_v &= R_i^T(v_j-v_i-g_O\Delta t)
  -(\Delta\bar v_{ij}+J_{v b_g}\delta b_g+J_{v b_a}\delta b_a),\\
r_p &= R_i^T(p_j-p_i-v_i\Delta t-\tfrac12 g_O\Delta t^2)
  -(\Delta\bar p_{ij}+J_{p b_g}\delta b_g+J_{p b_a}\delta b_a),
\end{aligned}
\]

and the bias residual is

\[
r_b=\begin{bmatrix}b^g_j-b^g_i & b^a_j-b^a_i\end{bmatrix}^{T}.
\]

GTSAM defines the combined motion/bias error and its complete `15 x 15`
covariance. `CombinedImuFactor::evaluateError()` supplies the unwhitened error
and Jacobians; the Ceres adapter applies the GTSAM noise model exactly once.
The adapter exposes named Meridian blocks rather than the library's positional
ordering. The IMU residual uses its Gaussian noise model directly; a robust
loss does not hide a timestamp gap, saturation, or invalid sample.

The ROS-free Meridian API remains independent of GTSAM:

```cpp
struct ImuBias {
  Vec3 gyro;
  Vec3 accel;
};

struct ImuInterval {
  StateId from;
  StateId to;
  TimeRange support;
  SampleSpan samples;
  ImuIntervalQuality quality;
};

struct PreintegratedImu {
  TimeRange support;
  ImuBias linearization_bias;
  SO3 delta_rotation;
  Vec3 delta_velocity;
  Vec3 delta_position;
  BiasJacobians bias_jacobians;
  CombinedPreintegrationCovariance covariance;
  ObservationProvenance provenance;
};

class ImuBuffer {
 public:
  IntervalResult interval(TimeNs begin, TimeNs end) const;
};

class ImuPreintegrator {
 public:
  PreintegrationResult integrate(const ImuInterval&, ImuBias bias) const;
};

class ImuPropagator {
 public:
  PropagationResult propagate(const NavigationState&,
                              const ImuInterval&) const;
};
```

`ImuPreintegrator` privately owns the GTSAM PIM. `ImuPropagator` owns the dense
discrete trajectory needed to deskew a LiDAR sweep. They share calibrated
measurements, exact interval support, frame conventions, and noise units, and
their endpoint predictions must agree within a tested tolerance. Dense
propagation conditions LiDAR processing and is not added as a second source of
IMU information.

#### Exact temporal support

Time is signed integer nanoseconds at every API boundary. An interval is
integrated on exact support `[t_i, t_j)`. The buffer retains a bracketing sample
on each side and linearly interpolates accelerometer and gyroscope measurements
at exact interval boundaries. Adjacent intervals own disjoint durations; a
boundary sample may interpolate both sides but no `dt` enters two constraints.

When a new sensor timestamp requests a state inside an existing interval, the
timeline re-integrates the retained raw samples into the two exact child
intervals. It never algebraically splits a preintegrated delta. Bias changes use
the analytic first-order Jacobians while they remain inside a validated
correction range; a larger change causes raw-sample repropagation.

Admission checks cover monotonicity, duplicate timestamps, finite values,
device status, range/saturation, sample rate, endpoint brackets, and maximum
gap. A failed interval produces a typed health event and no inertial
constraint. The first implementation requires complete intervals; covariance
inflation across missing ticks is introduced only if a later controlled
benchmark validates a physical gap model.

Each IMU adapter preserves the selected measurement timestamp, raw device
timestamp when available, sequence, clock epoch, temperature, and device status
in `ImuSample`. For deployment, the SBG adapter unwraps its finite-width device
counter before it becomes `TimeNs`; wrap, reboot, PPS/UTC lock change, or
time-source change is an observable epoch event. This is mandatory for a
three-hour mission. ROS callback-arrival time remains transport metadata rather
than measurement time.

The inertial observation is calibrated six-axis angular rate and specific
force. Vendor navigation attitude, velocity, position, or GNSS-aided INS output
is not folded into this factor under an IMU label.

Noise inputs are continuous-time densities with explicit units. Newer College
uses its Alphasense dataset calibration. The deployment profile is seeded from
the exact SBG Ellipse-D configuration and its
[official specifications](https://support.sbg-systems.com/sc/el/latest/ellipse-documentation/performance-specifications),
then measured with Allan-variance and on-vehicle vibration logs. The resolved
`SensorId`, `CalibrationId`, values, and units are recorded at startup.

#### Preintegration decision and benchmark

| Realization | Strength | Cost or risk | v3 decision |
| --- | --- | --- | --- |
| GTSAM combined 15-D preintegration through a private Ceres adapter | Mature deltas, bias correction, full covariance, residuals, and Jacobians | Requires explicit convention conversion and retains both GTSAM and Ceres | **Production** |
| Meridian-native combined Ceres factor | Complete control over integration and layouts | Reimplements sensitive navigation mathematics and its validation surface | Later alternative only if the adapter proves limiting |
| GTSAM `ImuFactor` plus bias walk | Simpler split covariance and established implementation | Drops motion-bias cross-covariance | Numerical comparison |
| Closed-form or higher-order preintegration | Can improve low-rate, high-dynamic integration | More mathematics and validation for an uncertain 200 Hz benefit | Later research alternative |

The adapter must match the native GTSAM combined factor on randomized states for
the unwhitened and whitened residuals and every Jacobian. Further gates cover a
zero residual at `predict()`, finite differences in the Ceres chart, covariance
symmetry/PSD and named block order, exact-boundary and state-split integration,
bias correction versus raw reintegration, analytic motion, Monte-Carlo NEES,
and a small Ceres solve compared with GTSAM LM. Gap, duplicate, saturation,
unit, and clock-epoch tests fail observably. Runtime reports integration,
factor evaluation, and repropagation p50/p95/p99 at 200 Hz.

Primary implementation context is GTSAM's official
[`PreintegratedCombinedMeasurements`](https://borglab.github.io/gtsam/preintegratedcombinedmeasurements/)
and [`CombinedImuFactor`](https://borglab.github.io/gtsam/combinedimufactor/).
`../slam-reference/OKVIS2-X/okvis_ceres/src/ImuError.cpp` and
`../slam-reference/GVINS/estimator/src/factor/{integration_base.h,imu_factor.h}`
remain independent Ceres references for convention and Jacobian audits.

### Local inertial initialization

Initialization is a bounded, ROS-free coordinator that owns startup IMU and
LiDAR buffers plus a temporary registration target. It has no authority over
the production estimator or registration map until one result passes all
gates. Geospatial datum selection runs in parallel and is independent.

```cpp
enum class InitializationMode {
  Static,
  Dynamic,
};

struct InitializationResult {
  InitializationMode method;
  TimeNs anchor_time;
  NavigationState seed_state;
  NavigationCovariance seed_uncertainty;
  IndependentInitialPrior initial_prior;
  TimeRange support;
  InitializationQuality quality;
  ObservationProvenance provenance;
};
```

`seed_state` and `seed_uncertainty` are warm-start diagnostics derived from the
bootstrap observations. They never become estimator residuals when those
observations are replayed. `initial_prior` contains only independently sourced
terms: local position/yaw gauge, calibrated bias prior, and, in configured
`Static` mode, the operator's zero-motion assumption with its named support and
uncertainty. Each term carries its own provenance.

The YAML selects exactly `STATIC` or `DYNAMIC`; there is no automatic
classification or fallback between algorithms. A failed mode reports a typed
initialization failure. A six-axis IMU alone cannot distinguish rest from
constant-velocity motion, so selecting `Static` is an operational assertion,
not an inference from low IMU variance.

#### Static mode

A contiguous IMU window is checked for sample count, gaps, saturation, mean
angular rate, gyro dispersion, specific-force norm, and direction stability
under real vehicle vibration. Robust statistics are formed from short
time-block means so high-frequency chassis vibration cannot dominate the
estimate. These checks reject inconsistent input but do not switch to the
dynamic method. On acceptance:

- mean gyro seeds `b_g`;
- the direction of mean specific force minus the calibrated `b_a` prior seeds
  roll and pitch;
- the configured zero-motion assumption supplies the zero-velocity
  observation;
- `base_link` yaw and position initialize their `odom` gauges to zero; and
- `b_a` is seeded from the calibrated sensor prior with conservative
  covariance.

Full accelerometer bias and tilt are coupled in one static orientation. The
implementation therefore retains the calibrated `b_a` prior rather than the
overconfident full accelerometer-bias estimate used in Meridian v1.

#### Dynamic LiDAR-IMU branch

The selected dynamic initializer follows the staged observability structure of
[D-LI-Init](https://arxiv.org/abs/2504.01451) and VINS/GVINS while keeping the
calibrated extrinsics and time offset fixed:

1. Collect a bounded contiguous LiDAR-IMU window; the initial benchmark seed is
   20 Ouster sweeps, approximately two seconds, with a five-second hard support
   ceiling for weak but improving excitation.
2. Rotation-deskew each sweep with the calibrated gyro-bias prior.
3. Run a private bootstrap LiDAR odometry using the same direct point-to-point
   registration algorithm as normal tracking and a temporary target. Its metric
   adjacent motions are initialization observations only; reserve the newest
   accepted transition from the alignment solves for validation.
4. Solve gyro bias from LiDAR relative rotations and IMU rotation Jacobians.
5. Reintegrate all IMU intervals at the corrected gyro bias.
6. Solve window velocities and gravity linearly from LiDAR translations and
   preintegrated position/velocity deltas, with `b_a` fixed to its calibrated
   prior.
7. Refine the gravity direction on the two-dimensional tangent space of the
   gravity sphere, apply the resulting alignment to every bootstrap pose and
   velocity, then set the anchor `base_link` translation and yaw gauges to zero
   and derive its IMU pose through the calibrated extrinsic.
8. Perform one full translational deskew and one registration/alignment rebuild;
   this second pass is the refinement ceiling.
9. Validate excitation, linear-system rank and condition, gravity magnitude,
   bias plausibility, residual consistency, and the newest held-out LiDAR
   transition.
10. Return one `InitializationResult`; insufficient excitation or geometry
    remains in `Collecting` rather than committing a guessed state.

The temporary LiDAR odometry owns no production factor, state, or map payload
and is discarded after initialization. Its poses only warm-start the result.

The accelerometer bias begins at one calibration prior and is estimated by the
normal sliding window as informative motion accumulates. This removes the
poorly conditioned all-at-once tilt/gravity/accelerometer-bias solve that made
v2 require dataset-specific rank thresholds and up to 1000 LM iterations.

Bootstrap processing is isolated from production map state. After acceptance,
the coordinator places the anchor at the first retained support time and
replays the buffered immutable observations once through the normal ingestion,
deskew, frontend, and estimator path. Static IMU statistics and dynamic
LiDAR-IMU alignment supply warm starts only: their temporary objectives are not
retained as estimator factors or priors. The original observations contribute
likelihood information exactly once, through production IMU/LiDAR factors, or
are explicitly marked as target-only. This reconstructs the startup trajectory
and target using production semantics, rather than losing the first two seconds
from ATE/RPE coverage.

The first replayed, map-eligible LiDAR frame is admitted once as the
`BootstrapAnchorPayload` after its anchored state is accepted. It seeds the
empty active target but creates no LiDAR residual and contributes no second
initialization information. Normal registration begins with the next sweep;
the one-time admission and its odom epoch are explicit in status and debug data.

| Initialization family | Representative reference | Strength | Limitation | v3 use |
| --- | --- | --- | --- | --- |
| Configured static IMU mean | FAST-LIO2, RKO-LIO, Meridian v1 | Fast, simple, and appropriate when standstill is operationally guaranteed | Invalid for a moving start | **Static mode** |
| IMU mean without motion verification | GLIM naive initializer | Immediate | Moving starts bias gravity, attitude, and velocity | Negative oracle |
| One large joint nonlinear bootstrap | GLIM loose initializer, Meridian v2 | Can represent all variables | Seed, gauge, and weak-mode sensitivity | Offline comparison only |
| Staged metric LiDAR-IMU alignment then normal window | D-LI-Init and VINS/GVINS lineage | Observable subproblems, metric scale, and inspectable failure gates | Requires a short LiDAR motion window | **Dynamic mode** |
| Online spatial-temporal calibration | [LI-Init](https://arxiv.org/abs/2202.11006), Ultra-Fusion | Also estimates time and extrinsics | Richer excitation and convergence burden | Commissioning or later extension |

Initialization is benchmarked at multiple start offsets in every bag: verified
rest, vibration at rest, translation, rotation, and weak excitation. Reports
include success and false-success rates, time to result, gravity-direction
error, initial velocity and bias plausibility, rank/condition metrics, held-out
motion error, ATE/RPE over the first 10 s and 30 s, total trajectory coverage,
CPU, memory, and p95 latency. The runtime contains one dynamic algorithm; the
other rows are literature and test oracles.

The primary code references are
`../slam-reference/D-LI-Init/include/Dynamic_init/Dynamic_init.cpp` and
`../slam-reference/GVINS/estimator/src/initial/initial_aligment.cpp`.
`../slam-reference/LiDAR_IMU_Init` is commissioning context, while GLIM's naive
and loose initializers are alternatives to inspect. GPL reference repositories
supply published mathematics and implementation lessons without source copying.

### Direct LiDAR frontend and local registration target

#### Selected first implementation

Meridian has one production LiDAR path:

- preintegrator-consistent discrete IMU propagation with exact timestamp
  support for sweep-end deskew and the pose seed;
- deterministic robust direct point-to-point registration;
- an owner-aware active overlay plus a spatially bounded finalized voxel base;
- pose-linked binary residuals for active target owners and unary residuals for
  fixed finalized geometry; and
- atomic, sensor-pure `FactorBatch` admission.

The IMU prediction conditions deskew, correspondence capture, and candidate
initialization. The LiDAR batch contains no gravity regularizer, preintegration
row, or other inertial measurement. The ICP source-pose correction seeds the
common estimator candidate; it is not inserted again as a relative-pose
measurement.

| Reference | Mechanism used as context | Meridian use |
| --- | --- | --- |
| [KISS-ICP](https://arxiv.org/abs/2209.15397) | Robust direct point-to-point ICP, voxel sampling, adaptive thresholding | Registration mathematics and robust-loss challenger |
| [RKO-LIO](https://arxiv.org/abs/2509.06593) | Parallel direct point-to-point association with IMU deskew and prediction | Runtime and capture-range context |
| Meridian v1 | Direct residual, bounded voxel hash, clipping, and natural-convergence behavior | Implementation starting point |
| [FAST-LIO2](https://arxiv.org/abs/2107.06829) | Direct raw-point frontend and bounded incremental map | Map-maintenance and point-to-plane challenger |
| [Voxel-SLAM](https://arxiv.org/abs/2410.08935) | Adaptive voxel planes, cluster sufficient statistics, sliding-window LiDAR–IMU BA, and hierarchical global geometry BA | Plane-BA and multi-timescale association challenger; see the [full comparison](VOXEL_SLAM_COMPARISON.md) |
| [Faster-LIO](https://github.com/gaoxiang12/faster-lio) | Sparse incremental voxel indexing and bounded local queries | Data-layout and query-pruning context only |
| [GLIM](https://arxiv.org/abs/2407.10344) | Pose-local scans, binary active constraints, unary fixed-target constraints | Ownership and finalization topology |
| Meridian v2 | Atomic batches, owner revisions, MAD-Huber, information cap, finalized support | Validated correctness and failure lessons |

This first implementation needs no normals, local-plane fitting, per-point
covariances, GICP/VGICP, learned features, or fallback registrar. Point-to-plane
and VGICP remain controlled research challengers rather than runtime code paths.

| LiDAR design | Main strength | Off-road and framework tradeoff | v3 decision |
| --- | --- | --- | --- |
| Robust direct point-to-point | No normal/covariance neighbourhoods, exact residuals, and strong KISS-ICP/RKO/v1 context | Requires balanced voxel support, robust rejection, and conservative information | **Production** |
| Local point-to-plane | Strong convergence on stable planar surfaces; FAST-LIO2 reference | Plane fitting and normals become fragile on foliage, mixed surfaces, and irregular ground | Controlled challenger after the first pass |
| GICP/VGICP | Rich local surface uncertainty; GLIM reference | Covariance construction, mutable association, and evaluation complexity dominated v2 bring-up | Outside the first implementation |
| ICP pose plus covariance factor | Very small graph | Discards direct directional geometry and may count the ICP proposal twice | Outside the first implementation |
| One mutable accumulated cloud | Simple sequential odometry | Live geometry cannot follow sliding-window pose revisions | Used only for finalized geometry |

#### Immutable scan representation

After physical filtering and exact deskew to the sweep-end reference frame, the
frontend creates

```cpp
struct LidarRegistrationFrame {
  ObservationId observation;
  SensorId sensor;
  TimeRange sweep_support;
  TimeNs reference_time;
  CalibrationId calibration;
  StateId state;
  DeskewProvenance deskew;
  PointBuffer points_lidar_reference;
  PointSelection target_points;
  PointSelection source_rows;
  LidarPreprocessQuality quality;
  ContentHash content;
};
```

Input validation covers calibrated Ouster per-point time, return/status policy,
and finite coordinates. Each sensor profile then supplies finite
`minimum_registration_range_m` and `maximum_registration_range_m`, with
`0 <= minimum < maximum` and a maximum compatible with the sensor metadata.
The frontend keeps a return exactly when

\[
r_{min} \leq \lVert p_L(t_{acquisition})\rVert \leq r_{max}.
\]

This physical range is evaluated in the return's acquisition-time LiDAR frame
before robot-body masking, deskew, voxel selection, registration, factor
construction, or local-target admission. Every downstream SLAM artifact
therefore contains only the filtered geometry; filtering is performed once,
not repeated at map insertion. The immutable raw sweep remains available to
offline inspection/debug and to future non-SLAM consumers that may require a
different near-field policy.

Initial benchmark seeds are `[1.5 m, 50 m]` for the Newer College OS0-128 and
`[1.5 m, 150 m]` for the Barakuda OS1-128. The OS1 upper bound is a registration
ROI, not a quality guarantee, and is accepted only when the deployed hardware,
firmware, and signal-multiplier configuration give an unambiguous representable
range beyond it. This distinction follows the published
[Newer College platform specification](https://ori-drs.github.io/newer-college-dataset/multi-cam/platform-multi/),
[OS1 specifications](https://ouster.com/products/hardware/os1-lidar-sensor),
and [Ouster signal-multiplier limits](https://static.ouster.dev/sensor-docs/image_route1/image_route3/sensor_operations/sensor-operations.html).
The `1.5 m` near bound is validated against `1.0 m` on complete evaluation
sequences before promotion to a deployment default.

The raw accepted sweep and its registration artifact are immutable.
Deterministic voxel selection makes a denser target representation and a
sparser, spatially balanced source-row representation; these are two densities
of the same algorithm. Preprocessing reports `invalid`, `below_minimum`,
`above_maximum`, `body_masked`, and `retained` counts plus a bounded range
histogram. A sweep empty after filtering produces a typed frontend rejection,
never an empty ICP call. Unit tests retain both exact endpoints, prove that
range is evaluated before deskew, preserve observation/time provenance, show
that rejected points reach no SLAM artifact, and verify deterministic output
under the configured worker count.

`DeskewProvenance` records the anchor-state revision, bias linearization value,
exact IMU sample support, calibration identity, and clock epoch. Geometry is
frozen for that `FactorBatch` revision. The frontend reports a conservative
bound on point displacement induced by later pose and bias corrections, but the
first implementation does not retrospectively re-deskew active scans. A
replacement policy is introduced only if full-sequence measurements show that
this approximation materially affects registration.

#### Registration target and ownership

```text
LocalLidarTarget
├── ActiveOverlay
│   └── one immutable scan-local SparseVoxelIndex per live owner StateId
└── FinalizedVoxelBase
    └── one bounded SparseVoxelIndex of geometry fixed in odom
```

The active overlay contains accepted, map-eligible scans whose owners remain in
the estimator window. Each owner stores immutable target points and a compact
voxel index in that scan's own LiDAR reference frame, plus a coarse occupancy
envelope and its current pose snapshot. The index is built once when the owner
is admitted. A later estimator correction changes only the owner pose; it does
not transform its points, change voxel keys, or rebuild a composite cloud.

Before registration, a deterministic selector scores active owners by predicted
geometric overlap and field of view, preserves spatial diversity, and freezes a
finite owner set for the proposal. Timestamp or `StateId` only breaks equal
scores; newest/middle/oldest is not a target policy. The first profile uses at
most four active owners. This scan-local ownership and overlap selection follow
the useful topology of [GLIM](https://arxiv.org/abs/2407.10344), without taking
its VGICP residual or solver.

The finalized base is an incremental voxel hash clipped to a bounded radius
around the robot. It stores points transformed by their owners' final local
poses and minimal owner provenance. Density, per-voxel capacity, and spatial
eviction are deterministic. This base supplies broad historical local geometry
without becoming the future mission-length global map.

The base is one consumer of `FinalizedLidarFrame`, not the mission-length owner
of its scan-local payload. Finalization exposes that immutable payload before
local voxelization and radius eviction. A future submap builder can consume the
same stream without reading, copying, or depending on the mutable registration
target.

The active and finalized layers remain separate. For each source row, the
predicted point is transformed into every selected owner frame and queries that
owner's scan-local index; its `odom` position queries the finalized base once.
Rigid transforms preserve metric distance, so all candidates are compared by
one exact squared-distance rule and the single closest valid winner is kept. It
therefore creates one active-owner row or one finalized-base row, never both.
The owner set, poses, finalized-base version, and index handles form one pinned
immutable target snapshot for the complete association round.

#### Voxel index mechanics

Both target layers use the same ROS-free `SparseVoxelIndex` primitive with
different ownership and lifecycle. It is a compact sparse voxel hash grounded
in Meridian v1 and [KISS-ICP](https://arxiv.org/abs/2209.15397), and informed by
the incremental voxel layout of
[Faster-LIO](https://github.com/gaoxiang12/faster-lio). It is not an ikd-tree or
an imported iVox runtime.

For point `p` in the index's declared frame and cell width `v`, the cell key is

\[
c(p)=\left\lfloor p/v\right\rfloor.
\]

Each occupied cell stores a bounded, deterministically selected contiguous set
of points with a minimum spacing. A query with correspondence gate `d_max`
visits only

\[
\left(2\left\lceil d_{max}/v\right\rceil+1\right)^3
\]

neighbouring cells and then applies the exact Euclidean gate. The `1.0 m` cell
and `0.5 m` gate seed therefore visit 27 cells per queried index. Cells are
visited in a deterministic order; the point-to-cell-AABB lower bound skips any
cell that cannot beat the current winner or gate. Query work is bounded by
source rows, selected active owners, visited cells, candidates per cell, and
outer iterations rather than total map history.

Hot-path cells and point records use compact integer owner/point handles. Full
provenance is interned once in the pinned owner table and later in the
`FactorBatch`; it is not copied through candidate queries. Queries allocate no
memory. Source transforms, candidate buffers, and per-worker scratch persist
across scans. The finalized base inserts through touched fine cells and groups
them into coarse spatial chunks; radius eviction occurs only after configured
robot motion and prunes chunk AABBs with hysteresis instead of walking every
fine voxel at 10 Hz.

An accumulated cloud without ownership can register a sequential odometry
frontend, but after the sliding-window solver changes an old pose its points
remain at the old placement. Attaching that cloud to one newest pose also gives
the solver the wrong Jacobian. The per-owner indices preserve exact
relinearizable geometry without rebuilding active points; finalization is the
explicit moment at which old geometry is transformed once and becomes fixed.

#### Point-to-point proposal

The optimized state is the IMU pose, so the LiDAR pose is

\[
T^O_{L_k}(X_k)=T^O_{I_k}(X_k)T^I_L.
\]

For source point `p_j` and active target point `q_i`, both stored in their
owning LiDAR frames, the residual is

\[
r_{ji}(X_j,X_i)=
  (T^O_{L_i}(X_i))^{-1}T^O_{L_j}(X_j)p_j-q_i.
\]

It is a binary residual on the two true owner poses. For a finalized point
`q_m^O`,

\[
r_{jm}(X_j)=T^O_{L_j}(X_j)p_j-q_m^O,
\]

which is unary on the live source pose.

The frontend first solves a six-degree-of-freedom ICP proposal while the target
snapshot remains immutable. Each outer iteration:

1. transforms the deterministic source rows from the IMU pose seed;
2. searches the selected owner indices and finalized base and assigns one
   globally nearest target winner per row;
3. computes the robust objective and normal equations in a declared tangent
   convention;
4. solves only numerically observable directions with bounded LM damping;
5. accepts a step only when the frozen-correspondence objective decreases; and
6. reassociates for the next outer iteration.

Translation and rotation have separate convergence thresholds. Iteration count
is a fault ceiling. Non-finite math, insufficient spatial support, excessive
correction from the IMU seed, zero observable rank, or failure to find a
decreasing bounded step rejects the proposal. Registration never mutates either
target layer.

The ICP evaluator uses a persistent structure-of-arrays workspace and a fixed
source-row partition. Each worker accumulates one local `{H, b, cost, count}`
without constructing a 6-by-6 object per correspondence; worker results are
reduced in a fixed order. Source transforms and candidate buffers are reused
within the iteration. Configuration supplies finite caps for source rows,
selected owners, candidate examinations, outer iterations, and live wall time;
exhaustion is a typed failed proposal rather than a partial association.

During normal initialized operation, every complete ICP has an explicit
consumer: it supplies both the warm-start and the `LidarFactorBatch` of the same
candidate transaction. The first implementation requests one navigation state
and one possible LiDAR batch for every eligible 10 Hz sweep. It has no complete
tracking-only registration. Bootstrap and recovery registrations are separate
declared consumers; if factor cadence is later reduced, intermediate sweeps do
not run the full ICP or enter the target unless their result is assigned a
formal estimator or predictor role.

#### Robust weights and directional information

For final correspondence norms `d_j = ||r_j||`, the selected robust scale is

\[
\hat\sigma_r=1.4826\operatorname{median}_j
  |d_j-\operatorname{median}(d)|,
\qquad
\delta=\operatorname{clamp}(2.5\hat\sigma_r,0.15\text{ m},0.50\text{ m}).
\]

Each 3-D point residual receives its own Huber loss with this fixed scale. This
MAD-Huber model is simple to inspect and produced the complete loss-free Quad
Easy direct-path runs and the best v2 hybrid result. It remains to be validated
on Park vegetation; KISS-ICP's Geman-McClure weighting is the first controlled
challenger after the initial implementation.

For channel `c`, the declared metric objective is

\[
F_c=\frac{\alpha_c}{2\sigma_c^2}
\sum_j\rho_{\delta_c}(\|r_j\|^2),
\]

where `delta_c` is in metres. An implementation that whitens before applying
the loss must therefore use threshold `delta_c / sigma_c`. Correspondences and
MAD scale remain frozen during each LM decrease test; association and scale are
recomputed only between bounded outer rounds.

The rule applies at two levels. Frontend ICP reassociates between accepted
six-degree-of-freedom outer steps. The complete window solve then uses immutable
correspondence revisions. Before commit, the estimator checks the incoming and
every older explicit live LiDAR batch against its own connected-pose revision
and benchmarked association-validity bound. One bounded repair round may
atomically rebuild/replace affected batches, or remove a recent invalid batch
under the reliability policy, before resolving the complete candidate. A
remaining stale batch or exceeded repair budget rejects the candidate. Existing
factors never query nearest neighbours from inside Ceres.

At the accepted association pose, with every target pose held at the immutable
association snapshot, the frontend computes the robust weighted conditional
source-pose normal matrix

\[
H_{S\mid T}=\sum_j \frac{w_j}{\sigma_j^2}J_{S,j}^T J_{S,j}.
\]

Translation and rotation are normalized with a characteristic scene length
before eigendecomposition. `FactorBatch` reports this typed
`SourceConditionalObservability`: matrix, eigenvalues, observable basis, rank,
and condition. It is a frontend quality view, not the full joint information of
binary multi-pose factors and is not inserted as a pose prior. Complete factor
information is carried by every row's typed covariance, its source and target
`StateId`s, and its residual definition; the solver forms the joint block
Jacobian from those rows.

Active and finalized channels compute separate conditional matrices and
separate scalar information ceilings. `alpha_active` scales only active binary
rows and `alpha_finalized` scales only fixed-base unary rows. Their scaled sum
produces the reported source-conditional view. Each scalar preserves its
channel's matrix coupling and rank, null directions remain null, and correlated
point count cannot make either lineage arbitrarily certain. This separation is
the direct guard against v2's accidental application of finalized-map
inflation to live factors.

The two typed effective scales are `sigma_active` and `sigma_finalized`.
`sigma_active` includes source sampling, live-target representation, and deskew
uncertainty; it is not advertised as raw Ouster point covariance. Finalized
geometry is also derived from measurements already absorbed by the marginal
prior, so future unary rows are a conditionally correlated approximation.
`sigma_finalized` adds the fixed-map representation/correlation floor and has
its own information ceiling. Both are calibrated with consistency and NEES
tests; the initial `0.10 m` value below is only a benchmark seed.

#### LiDAR FactorBatch

One accepted registration proposal builds one immutable batch:

```text
LidarFactorBatch
  metadata
    sensor_id, batch_id, source observation and StateId
    exact sweep and reference timestamps
    calibration_id and recovery_epoch
    target estimator revision and finalized-base version
    active and finalized row-noise models
    source-conditional directional observability
    frontend health, provenance, and map_eligible

  active groups[]
    target StateId and observation ID
    source-local / target-local point pairs
    row noise and fixed robust scale

  optional finalized group
    source-local / fixed-odom point pairs
    finalized owner provenance
    row noise and fixed robust scale
```

One source row belongs to exactly one group. The batch objective keeps a bounded
number of explicit 3-D point residuals selected deterministically from the
accepted association. Ceres receives a small number of analytic batch
evaluators—one per connected active owner and one for the finalized channel—not
one heap-allocated `CostFunction` per correspondence. Each evaluator owns
contiguous structure-of-arrays row data and applies the declared Huber objective
independently to every 3-D row. Scalar reference tests verify objective,
residual, and Jacobian parity over nominal, boundary, and outlier cases. v2's
compressed sufficient-statistic factor returns only as a separately benchmarked
replacement if this bounded evaluator remains the measured bottleneck. The
complete batch is inserted, rejected, replaced, removed, or sealed atomically.

Ceres relinearizes these fixed geometric pairs against current active poses but
never queries an index inside `CostFunction::Evaluate`. The post-solve
all-live-batch validation and bounded repair policy above applies before every
commit, not only when the newest batch is created. Recent LiDAR batches remain
removable while all their rows are explicit in the active window. If a repaired
candidate still fails, the complete LiDAR transaction is rejected; a separately
valid IMU state transition may then be proposed without it.

#### Commit and state finalization

A LiDAR event uses one candidate transaction:

1. request the source state and exact IMU interval;
2. create a frontend proposal against immutable target versions;
3. assemble the state, IMU interval, and optional LiDAR batch;
4. solve and validate the complete active objective;
5. stage a bounded `TargetDelta` containing changed owner-pose snapshots,
   admitted payload, finalization migrations, and evictions; and
6. atomically commit the estimator revision, factor provenance, and target
   delta, then publish the resulting immutable updates.

Payload enters the active overlay only when localization, `map_eligible`, and
sensor health all permit it. If the target delta cannot reserve and validate
its bounded work or expected revision, the complete LiDAR candidate remains
uncommitted and the target stays unchanged; the scheduler may propose the
corresponding IMU-only transition. Owner removal and finalized-base insertion
are two sides of the same target delta and cannot become separately visible.

When a state leaves the lag:

```cpp
struct FinalizedState {
  StateId id;
  NavigationState state;
  NavigationCovariance covariance;
  EstimatorRevision revision;
};
```

the candidate square-root marginal prior, sealed batch dispositions, final
state, and live-to-finalized target migration are staged together. After their
common commit, the final state is appended to the full-length ATE/RPE trajectory
and the system emits `FinalizedState` plus the immutable scan-local
`FinalizedLidarFrame`. A future submap builder may consume the same finalized
stream independently.

An old binary residual is not converted to a unary residual when its target
owner finalizes: its information is already represented in the marginal prior.
Only future source scans create new unary rows against fixed geometry. During a
LiDAR failure, existing good geometry remains readable and frozen while failed
payloads are excluded.

Clean evaluation shutdown performs one terminal accepted solve and emits the
still-active tail as a `TerminalTrajectorySnapshot` in timestamp order. This
completes ATE/RPE evaluation without sealing those states, migrating their
geometry, or inventing future measurements.

#### Initial registration profile and benchmark

The selected topology is grounded in complete Quad Easy v2 experiments. These
are internal causal results, not SOTA or deployment claims:

| Historical target | Fixed-lag ATE | Runtime result | Decision lesson |
| --- | ---: | --- | --- |
| Three pose-synchronized sweeps | `0.840655 m` | RTF `0.647` | Correct ownership with too little history still drifts |
| Fifteen sweeps attributed to one newest pose | `0.511941 m` | RTF `0.420` | More geometry cannot repair the wrong factor Jacobian |
| Pose-aware active target only | `0.559663 m` | RTF `1.005` | Realtime and loss-free, but active history alone does not bound drift |
| Broad persistent finalized base | `0.072653 m` | RTF `0.172`, `1.896 GB`, `9.798B` candidate examinations | Historical spatial support restores accuracy, while the old query/factor schedule is unusable |
| Nearest 12 finalized owners | `0.118963–0.130191 m` | RTF `0.525–0.674` | A small centre-biased owner set loses spatial diversity |
| Persistent finalized base plus pose-aware active overlay | `0.071944 m` | RTF `0.666`, about `408 MB` RSS | **Selected target lifecycle; query and factor cost still require fresh profiling** |

No durable v1 artifact establishes a comparable full-run ATE or RTF. v1 is the
voxel/P2P implementation reference. v2 GICP/VGICP bring-up did not produce a
comparable accepted full-run result, so it motivates a simpler first path but
does not prove those methods universally inferior.

These are reproducible v1/v2-derived starting values, not promoted universal
constants:

| Parameter | Initial benchmark seed |
| --- | ---: |
| Target sampling | `0.30 m` |
| Source-row sampling | `1.50 m` |
| Target index voxel | `1.0 m` |
| Target capacity | `20 points/voxel` |
| Correspondence gate | `0.50 m` |
| Point residual standard deviation | `0.10 m` |
| Huber scale | `2.5 x MAD`, clamped to `[0.15, 0.50] m` |
| Minimum correspondences | `30` |
| Maximum active target owners | `4`, selected by geometric overlap |
| Translation/rotation convergence | `1e-3 m / 1e-3 rad` |
| ICP outer-iteration ceiling | `10` |
| Finalized-base radius | `100 m` |

Source-row and retained-factor-row caps, candidate-examination cap, ICP and
Ceres wall deadlines, active/finalized memory caps, finalized density and
radius, information ceiling, and window size are selected by controlled
standalone and full-sequence benchmarks. Every launch profile must contain
finite nonzero values for these bounds before its first complete bag; missing
or unbounded values are configuration errors. Minimum support uses raw count,
robust effective count, occupied source-voxel coverage, and selected-owner
overlap. The initial runtime contains only the selected P2P/Huber path. Later
challengers change one decision at a time—FAST-LIO2-style local point-to-plane
or KISS-ICP Geman-McClure—using identical preprocessing, deskew, pose seed,
target content, input IDs, and scoring support.

Every complete Quad Easy, Quad Hard, and Park run reports online and finalized
ATE/RPE and coverage; residual and overlap statistics; directional spectrum;
correction from the IMU seed; registrations, batch gates, and recovery epochs;
map admissions, freezes, points, voxels, and evictions; candidate examinations;
range-filter dispositions, preprocessing, deskew, active-owner selection/query,
base query, ICP, factor construction, solve, and marginalization
p50/p95/p99/max; queue depth/age/loss; RTF, CPU, and peak RSS. Quad Easy gates
basic correctness, Quad Hard gates difficult motion, and Park gates vegetation,
duration, repeated structure, target growth, and memory. Short prefixes
diagnose a failure but do not select a model.

Implementation context is pinned by function and file in the local reference
repositories: KISS-ICP `cpp/kiss_icp/core/Registration.cpp` and
`VoxelHashMap.cpp`; RKO-LIO `rko_lio/core/lio.cpp`; FAST-LIO2
`src/laserMapping.cpp`; Faster-LIO `include/ivox3d/`; and GLIM
`src/glim/odometry/odometry_estimation_{gpu,imu}.cpp`. Meridian v1's direct
registration and `voxel_grid_map.cpp` are read from the `v1` tag; its linear
sparse-hash lookup is useful, while its whole-map `clipFarFrom` traversal is not
the finalized-base eviction design. The exact v1/v2 experiments and causal
lessons are preserved in `V1_V2_RETEX.md`.

### Sensor extension and reliability seam

Every modality follows the same narrow flow:

```text
independent bounded queue
  -> modality frontend
  -> typed FrontendResult
  -> per-sensor reliability gate
  -> optional FactorBatch
  -> common StateTimeline and estimator transaction
  -> accepted-localization map gate
```

The initial reliability policy uses per-sensor
`ACTIVE / SUSPECT / FAILED / RECOVERING` states and explicit recovery epochs.
Transport validity, frontend quality and directional observability, estimator
innovation, and post-solve acceptance are separate checks. The gate admits,
downweights, or rejects the complete batch. Registration or mapping payload is
inserted only after localization acceptance and `map_eligible`; a failed sensor
keeps its payload frozen while other queues and the IMU timeline continue.

### Startup coordination

Local IMU/LiDAR initialization and startup datum selection run independently.
Local odometry becomes available as soon as local initialization succeeds. The
startup `GeoreferenceCoordinator` in `meridian_apps` consumes typed `GnssFix`
records from `meridian_ros` together with a time-aligned accepted
`T_odom_base`, the calibrated GNSS lever arm, and the configured or measured
heading with its support time. It prefers a quality-accepted GNSS origin during
a bounded startup interval and otherwise uses the configured geographic
fallback. From the resulting `T_map_base` at the common support time it computes
`T_map_odom = T_map_base * inverse(T_odom_base)`. It emits one immutable
`GeographicDatum` and the initial revisioned `MapOdomEstimate` through
`meridian_core` geodetic functions and has no authority over the local
estimator.

The initialization YAML provides fallback latitude, longitude, WGS84 ellipsoid
height, and true heading of the `base_link` +x axis, clockwise from North.
Geographic heading does not gate local-odometry startup; it establishes the
initial yaw relationship between the ENU `map` frame and `odom`. A single
position fix does not observe heading; a quality-accepted dual-antenna or
motion-derived heading may replace the fallback before map-epoch commitment.
Later GNSS observations are reserved for a future estimator of `map -> odom`;
they never redefine `earth -> map`.

A later global backend consumes immutable local submaps built from the
finalized scan stream, estimates revisioned `T_map_submap` anchors from local
transitions, verified loop/relocalization constraints, and GNSS, then produces
`MapOdomEstimate` through the same API. Retrieval, geometric verification, and
global-graph algorithms are deliberately deferred until that implementation
stage.

### Debug implementation

There is no separate Rerun package. A neutral debug sink API allows pipeline
stages to emit timestamped records. Its private implementation lives under
`meridian_apps`; the current ingress executable always writes one `.rrd`
without starting or connecting to a Rerun viewer. A `NullDebugSink` keeps the
ROS-free boundary usable in focused tests and future compositions.

Rerun records what the process observes; it cannot reconstruct a sample lost in
DDS before a subscription callback or an executor decision that was never
exposed to the application. Standard bag playback intentionally validates that
same live path. The offline analyzer compares configured bag topic counts with
accepted rows and reports timing, gaps, ingress failures, and recorder health;
none of this accounting exists as runtime acceptance policy.

The initial record is deliberately small: accepted IMU scalars, one metadata
row per accepted LiDAR scan, ingress failures, recorder health, and a decimated
1 Hz LiDAR preview. Later algorithm slices add only the stage records needed to
explain their behavior. Heavy geometry is constructed only when requested by
the neutral sink.

## First implementation sequence

The first vertical slice is built in this order:

1. Implement core time/ID/observation records, generic ROS conversion, the
   owned-observation callback seam, bounded LiDAR/Rerun queues, standard bag
   playback, lightweight RRD telemetry, and the offline analyzer. **Complete.**
2. Implement the exact-support IMU buffer, GTSAM combined-preintegration Ceres
   adapter, dense propagator, and their numerical parity tests.
3. Implement LiDAR preprocessing, IMU deskew, deterministic voxel selection,
   the two-layer target, and standalone direct P2P registration. The same
   module supplies the temporary bootstrap odometry.
4. Implement configured `STATIC` and selected `DYNAMIC` initialization, with
   Newer College using `DYNAMIC` for the primary end-to-end benchmark.
5. Implement the sensor-neutral state timeline, Ceres batch window, combined
   IMU and direct LiDAR factors, fixed-linearization square-root
   marginalization, atomic commit, and finalization stream.
6. Publish local/global-composed poses, TF, and status; record diagnostics and
   debug evidence in Rerun; then run complete Quad Easy, Quad Hard, and Park
   evaluations in that order.

Lag/state limits, LiDAR row/candidate/memory budgets, association validity,
information ceilings, and ICP/Ceres iteration and time limits are required
resolved configuration values. Finite conservative seeds are established on
standalone fixtures before the first complete bag run and then changed only by
recorded full-sequence experiments. LiDAR factor cadence is initially every
eligible 10 Hz sweep. A lower ATE obtained by lost input, reduced coverage,
stale geometry, or missed deadlines is invalid. Global-backend, visual, and
dense-map implementation begins only after this local LiDAR-IMU slice is
correct and its full-sequence limitations are measured.
