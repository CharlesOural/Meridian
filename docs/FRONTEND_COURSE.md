# Meridian Front-End — A Course

This course is for the project owner: someone who knows the Meridian codebase but is new to the internals of SLAM front-ends. It builds intuition from first principles — what real-time odometry is and why it is hard, the continuous-time B-spline that represents the trajectory, the four families of measurement residuals, the sliding-window solve and marginalization, the real-time/deskew/observability machinery and how it fails, and finally a glossary plus an end-to-end walkthrough. A companion `docs/BACKEND_COURSE.md` covers L3 (the back-end factor graph, loop closure, and global consistency). Read this one front to back the first time; afterward, Chapter 6 is your reference desk.

## Table of Contents

- [Chapter 1 — What the Front-End Is and Why It's Hard](#chapter-1--what-the-front-end-is-and-why-its-hard)
- [Chapter 2 — Continuous-Time Trajectories and B-Splines](#chapter-2--continuous-time-trajectories-and-b-splines)
- [Chapter 3 — The Measurements: Residuals That Pull the Spline](#chapter-3--the-measurements-residuals-that-pull-the-spline)
- [Chapter 4 — The Sliding-Window Solve and Marginalization](#chapter-4--the-sliding-window-solve-and-marginalization)
- [Chapter 5 — Real-Time, Deskew, Observability, and How It Fails](#chapter-5--real-time-deskew-observability-and-how-it-fails)
- [Chapter 6 — Glossary & How to Navigate the Front-End](#chapter-6--glossary--how-to-navigate-the-front-end)

---

## Chapter 1 — What the Front-End Is and Why It's Hard

Before we dive into the math that makes Meridian's front-end work, let's build intuition about *what* it does and *why* it's so difficult. The front-end is the real-time odometry engine: it watches the robot move and tries to answer, a hundred times per second, "where am I right now?" But unlike offline processing, it has a strict deadline—roughly 100 milliseconds per LiDAR sweep—and it must fuse four different sensors that arrive at different rates, with different noise characteristics, and at different times.

### 1.1 Odometry vs Full SLAM: The Contract

Let's start with the big picture. **SLAM** (Simultaneous Localization And Mapping) has two parts:

1. **Localization**: figure out where the robot is.
2. **Mapping**: build a map of what the robot sees.

These two problems are circular—you need a good map to know where you are, and you need to know where you are to build a good map. Full SLAM solves both globally, correcting drift over long missions (that's the back-end, L3).

The **front-end** (L2) is not full SLAM. It is **odometry**—a local, real-time estimate of the trajectory in a short sliding window. Think of it as "how am I moving *right now*?" It does not try to correct months-old drift. It is smooth, continuous, and updated a hundred times per second. But alone, it is drift-prone: over a kilometer of travel, typical LiDAR+IMU odometry drifts 1–3 meters in directions where features are sparse (long corridors, tunnels, snow).

The **back-end** (L3) later asks "where have I *actually* been globally and consistently?"—reconciling that drifty odometry with loop closures (moments when the robot returns to a previously visited place) and GNSS anchors. The back-end can then broadcast corrections, and the front-end keeps publishing smooth local motion on top of the corrected global frame. This split—front-end fast and drifty, back-end slow and drift-free—is deliberate and powerful.

### 1.2 Four Sensors, Four Different Stories

Meridian's front-end fuses four streams of measurement. Each tells a different story and has its own failure modes.

#### LiDAR: Dense, Direct, Short-Range

A **LiDAR** (Light Detection and Ranging) scanner fires millions of laser pulses and measures how long the reflections take to return. The result is a dense 3D point cloud—millions of points showing the exact shape of the world. LiDAR is:

- **Direct**: the points are actual 3D coordinates in space; there is no correspondence or feature-extraction ambiguity.
- **Accurate**: ranges are precise (typically 1–2 cm error) over ~30 meters.
- **Dense**: you get millions of points per scan, so you can fit planes and surfaces even in featureless terrain (a hallway, an open field).

But it has limits:

- **Reflectivity-dependent**: shiny surfaces (glass, wet asphalt, water) reflect poorly.
- **Range-limited**: beyond ~30 m, the signal fades and measurements become noisier.
- **Motion distortion**: if the robot is moving during the scan, different points arrive at different times but the scanner collects them all as if the robot were still. This is **deskewing** — a central challenge we'll return to (§2.3).

Meridian registers LiDAR points by matching them to local planes in the 3D world: each point's distance to the nearest fitted plane is a "residual" that the optimizer tries to minimize. This is called **point-to-plane** matching.

#### IMU: Continuous Rate and Acceleration, No Absolute Reference

An **IMU** (Inertial Measurement Unit) has two sensors: a **gyroscope** (measures rotation rate, $\omega$ in rad/s) and an **accelerometer** (measures specific force, $a$ in m/s²). Meridian's IMU runs at ~200 Hz, so you get a dense stream of measurements.

IMU is:

- **Fast**: 200 Hz vs 10 Hz LiDAR, so the front-end gets motion information between scans.
- **Unbiased on average**: over short timescales (seconds), the mean of the IMU reads is very accurate.

But critically:

- **Has no absolute reference**: an IMU alone tells you how fast you're accelerating and rotating, but not where you are. If you blindfold yourself and someone accelerates you in a car, you can feel the acceleration, but you cannot tell if you are 1 meter or 1 kilometer from your starting point.
- **Biased**: the IMU drifts. Gyro bias (the "null reading" when the IMU is still) is ~0.01–0.1 rad/s, and it changes slowly with temperature. Accelerometer bias is ~0.01–0.1 m/s². These biases, if not estimated and removed, integrate into huge position/velocity errors.
- **Noise accumulates**: even white noise, when integrated twice to get position, grows without bound. A 0.1 m/s² noise density integrated over 1000 seconds gives drift on the order of 50 meters—severe but slower than gyro drift.

The front-end **estimates IMU biases** in real time as part of the sliding window, using LiDAR/camera constraints to ground truth. It also models IMU noise and integrates the measurements to predict motion between scans.

#### Camera: Texture and Features, but Fragile

A **camera** captures a 2D image. The front-end uses a **sparse-direct** photometric approach (FAST-LIVO2 style): it does not extract traditional features (corners, SIFT) but instead correlates small patches of pixel brightness against previous frames and the LiDAR map. This approach:

- **Works in low-texture areas** where traditional feature detectors fail (featureless walls, snow).
- **Is fast**: no expensive feature detection, just local patch correlation.

But:

- **Requires LiDAR depth**: without range data, the camera cannot estimate the 3D positions of matched patches. The camera's depth comes *directly* from the LiDAR map, so there is no separate VIO (visual-inertial odometry) initialization problem.
- **Fails in darkness**: no light, no image. LiDAR works at night.
- **Fails in dynamic scenes**: if people are moving in the image, the correlations get confused.
- **Is expensive if unconstrained**: without LiDAR constraining depth, visual bundle adjustment becomes a separate hard problem.

Camera integration here is **auxiliary**—it helps in textured scenes and refines extrinsics (the LiDAR-to-camera alignment) but does not drive the odometry alone.

#### GNSS: Absolute Position, but Coarse and Intermittent

**GNSS** (Global Navigation Satellite System, e.g. GPS) provides an absolute position fix (latitude, longitude, altitude) from satellites. On a clear day in open sky:

- **Accurate**: 1–2 meters of error in good conditions.
- **Absolute reference**: unlike IMU, it tells you where you are on Earth, not just how fast you're moving.

But:

- **Coarse**: 1–2 meters is 10–100× worse than LiDAR point accuracy.
- **Multipath-prone**: reflections off buildings corrupt the signal in urban canyons, giving fixes that are 10+ meters off.
- **Intermittent**: unlike LiDAR or IMU, GNSS might not arrive for seconds, or might be completely unavailable underground or in dense forest.
- **No heading**: a GPS receiver cannot tell which way it is facing—only its position.

The front-end **gates** GNSS fixes—it accepts them only when they are consistent with the odometry, using them to correct absolute drift but not overreacting to noise.

### 1.3 The Real-Time Constraint: ~100 ms per Sweep

Meridian targets a **Jetson Orin** GPU running at 10 Hz LiDAR (100 ms per sweep). Each sweep must:

1. Be registered against the map (LiDAR point-to-plane).
2. Have its motion estimated (deskew the sweep).
3. Be fused with the last N seconds of IMU, camera, and GNSS.
4. Produce a new pose estimate and any new keyframes.
5. **All in under 100 ms**.

This is **hard real-time**, not best-effort. Missing a deadline means:

- The operator gets stale odometry (a jumpy, discontinuous trajectory).
- The mapping layer falls behind and the map becomes outdated.
- In a real system (navigation, collision avoidance), this is **hazardous**.

The classic failure mode is **overload, not divergence**. FAST-LIO (the reference system) has been known to "fall behind"—the solve takes longer than 100 ms, so scans queue up, and by the time a pose is published it is already 200 ms old. The system is technically still working, but the latency has become unacceptable for control.

Meridian addresses this with:

- **Bounded solve**: the linear system solver has a fixed cost per control point, not per point cloud density.
- **Deterministic factor count**: the number of LiDAR residuals is capped (§3.1 in spec 04) by stratified resampling, not random or spatial subsampling.
- **Pipelined architecture**: the back-end solves in parallel on a separate thread, never blocking the front-end.

### 1.4 Continuous-Time Deskew: Why This Is Not Like FAST-LIO

The deepest difference between Meridian and FAST-LIO is the **deskew model**.

**FAST-LIO** works in discrete time: at each scan, it assumes the robot was stationary during the sweep, then deskews the cloud backward in time using IMU integration (rotating each point based on the cumulative rotation that happened during that point's true sample time). The trajectory between scans is a piecewise-linear interpolation of velocity, so orientation changes (gyro integration) add a nonlinear term. This works, and works well, but it requires a separate "deskew pass."

**Meridian** is **continuous-time (CT)**: the trajectory is a smooth spline $T(t)$ evaluated at each point's true sample time. There is no deskew pass; each point's residual is computed at its own time. As the optimizer iterates and moves the control points, every point is implicitly re-deskewed because the control points it depends on have moved. Deskew is free—it *is* the trajectory evaluation.

Why does this matter?

1. **Exact multi-rate fusion**: IMU at 200 Hz, LiDAR at 10 Hz, camera at 20 Hz, GNSS at 1 Hz all evaluate the same spline at different times. No resampling, no phase errors.
2. **True per-point pose**: for colourization (painting RGB onto the 3D map), you need the true pose at each point's instant. A discrete-time filter cannot give you that between scans; a CT spline gives it exactly and analytically.
3. **Analytic derivatives**: the spline's analytic derivatives (velocity, acceleration) feed the IMU residual with no numerical differentiation, which keeps the condition number good.

But there is a cost: **control points are not on-trajectory poses**. A B-spline control point is a latent variable; the curve passes nearby but not through it. Seeding the spline from an IMU-only prediction requires careful placement (one interval back, not at the grid time) so the seeded curve lands on the predicted trajectory, not a delayed copy of it (§1.2 in spec 04).

### 1.5 Tightly-Coupled vs Loosely-Coupled Fusion

There is a fundamental choice in sensor fusion: how many things do you optimize jointly?

**Loosely-coupled**: estimate odometry from LiDAR alone, separately estimate from visual/IMU, then fuse the two estimates. Each sub-system is simple, but information is lost at the boundary.

**Tightly-coupled**: one nonlinear least-squares problem with all four sensors. LiDAR residuals, camera residuals, and IMU residuals all compete in the same cost function, so the optimizer can trade off between them and correct for extrinsic (sensor mounting) errors. This is harder but more powerful.

Meridian is **tightly-coupled**: all four sensor streams live in one nonlinear least-squares window. The advantage:

- **Online extrinsic refinement**: the camera-to-LiDAR and GNSS-antenna-to-body transforms are estimated as part of the solve, correcting for mount flex or thermal drift.
- **Unified observability**: the optimizer knows which directions are weakly constrained (e.g., sideways motion in a hallway) and can inflate uncertainty on those axes so the back-end knows not to over-trust them (§4 in spec 04).
- **Consistent uncertainty**: the output covariance reflects all four sensors' contributions, not a post-hoc fusion of two separate covariances.

The cost is complexity: four residual functions to implement, four Jacobians to derive, four noise models to tune.

### 1.6 The Data Pipeline L0 → L1 → L2 → KeyframePacket → L3

Here is how data flows through Meridian in real time:

```
    [Raw sensors: LiDAR 10 Hz, IMU 200 Hz, camera 20 Hz, GNSS 1 Hz]
                        ↓
             L0: Sensor Abstraction + Time Sync
                  (PTP/PPS clock, one monotonic timeline)
                        ↓
          L1: Preprocessing (deskew, filter, pyramid)
            (LiDAR: range/intensity gate, voxel downsample)
         (Camera: build 3-level pyramid, global-shutter assume)
           (IMU: cold-start deskew provider if no spline yet)
                        ↓
       L2: Front-End Continuous-Time Estimator (THIS COURSE)
          (sliding-window B-spline, 8 control points ≈ 200 ms)
        (LiDAR point-to-plane, camera photometric, IMU, GNSS)
             (Live pose estimate every 100 ms via spline eval)
                        ↓ (on keyframe trigger: dist > 1 m or rot > 10°)
                  KeyframePacket
        (pose, relative covariance, observability, cloud, image)
                        ↓
        L3: Back-End (iSAM2 factor graph, global scope)
             L4: Map (GPU nvblox TSDF + colour mesh)
             L5: Loop closure (place recognition, GICP)
```

**L2 owns the odom frame** and publishes high-rate smooth odometry (no jumps). **L3 owns the map frame** and gradually corrects drift via loop closure and GNSS. They are decoupled: the front-end never blocks waiting for the back-end, and the back-end never makes the front-end jump—it broadcasts corrected poses that L2 picks up in stride.

### 1.7 Failure Modes and Mitigation

The front-end can fail in several ways. Understanding these clarifies the design:

#### Degeneracy: Weak Observability in Some Directions

Imagine a robot in a **hallway**:
- **Forward (down the hallway)**: well-observed; you see the hallway receding.
- **Sideways (perpendicular to walls)**: poorly observed; the walls are featureless.
- **Up**: unobserved; you see the ceiling but motion up is essentially invisible.

The front-end estimates this per-axis (§4 in spec 04) as an **observability score** (0 = unobservable, 1 = fully observable). When a direction is weak (score ≈ 0.1), the estimated covariance is inflated so the back-end does not over-trust it.

Mitigation: **observability inflation** (degeneracy handling). The back-end receives the per-axis scores and scales the covariance accordingly, rather than guessing or silently dropping factors.

#### Motion Blur: Fast Motion During a Sweep

If the robot is moving fast (e.g. on a quadcopter at 10 m/s), a 100 ms LiDAR sweep captures motion over 1 meter. The first points in the sweep are 1 meter behind the last points. **Without deskew**, these are treated as if they were all at the same instant, corrupting the plane fits.

Mitigation: **continuous-time deskew** (each point at its true time), plus **IMU warm-start** (if motion is so fast the early trajectory guess is bad, re-seed from IMU prediction).

#### Loss of Tracking: Observability Collapse or Matching Failure

The front-end can "lose lock" when:
- **Features disappear**: a long tunnel with no variation.
- **Observability collapses**: the modes the sensors constrain narrow to a subspace (e.g., only forward motion, no sideways or rotation).
- **Map degrades**: the local map has aged and no longer matches new scans.

Mitigation: **window restart fallback** (§7.4 in spec 04). When the solver's residuals blow up or observability degrades, freeze the current good pose, re-initialize the sliding window from IMU-only integration, and resume. The next keyframe emitted carries an IMU preintegration factor instead of a between-factor, telling the back-end "the odometry was broken here, use raw IMU." The back-end can then use loop closure or GNSS to correct if needed.

#### Latency Deadline Miss

The solver takes longer than 100 ms, so old scans queue up.

Mitigation: **bounded factor count** (never more than ~1500 LiDAR factors per sweep, even if 10,000 points match the map) and **deterministic subsampling by plane-normal stratum** so that factor selection does not change across reorderings.

### 1.8 What Happens on a Keyframe

Not every sweep triggers a keyframe. The front-end publishes a **smooth, continuous pose** at 100 Hz (via spline evaluation), but only emits a **KeyframePacket** (the boundary handoff to the back-end) when:

1. **Distance**: the robot has moved > 1 m from the last keyframe, OR
2. **Rotation**: the robot has rotated > 10° from the last keyframe, OR
3. **Time**: 1 second has elapsed since the last keyframe.

(These thresholds are configurable, spec 00 §8.2.)

When a keyframe is born:

1. The spline is **evaluated** at the keyframe's timestamp to get a pose.
2. The **relative transform** to the previous keyframe is computed (with covariance).
3. The **observability scores** are extracted from the window Hessian (per-axis Fisher information).
4. The **retained cloud** is deskewed by evaluating the spline at each point's true time, giving a true per-point pose at that instant.
5. The **image** (if present) is saved for colourization.
6. All of this is packed into a `KeyframePacket` and handed to L3.

The `KeyframePacket` is the **only** thing that crosses from L2 to L3. It contains pose, covariance, observability, and data, but *no* internal front-end state (biases, spline control points, map). This enforces a clean seam: the back-end is fully decoupled from how the front-end estimated the pose.

### 1.9 Reading Guide for the Rest of the Course

Now that you have the big picture, here is what comes next:

- **Chapter 2**: The spline representation (split SO(3)×ℝ³ B-spline), why it is continuous-time, and how to seed it from IMU.
- **Chapter 3**: The four residuals (LiDAR point-to-plane, camera photometric, IMU derivative, GNSS), plus per-axis observability and degeneracy detection.
- **Chapter 4**: The sliding-window solve and marginalization (how old information is summarized), threading, and the `IFrontEnd` interface.
- **Chapter 5**: Real-time budgeting, the deskew feedback loop, and how the system fails and recovers.
- **Chapter 6**: A glossary and an end-to-end walkthrough of a single LiDAR sweep through the code.

Each section assumes you understand the layer contracts from spec 00 (§6: the KeyframePacket; §7: deskew feedback; §11: threading).

**This is the front-end's job: estimate a smooth, continuous trajectory in real time, fusing four sensors with tight coupling and per-axis observability awareness, all while hitting a hard 100 ms deadline and handing off drift-free, well-characterized uncertainty to the back-end.**

---

## Chapter 2 — Continuous-Time Trajectories and B-Splines

### 2.1 Why continuous-time matters: the spinning LiDAR problem

The frontend (L2) sits inside a **continuous** stream of sensor data arriving at different rates and different times. The LiDAR spins at, say, 10 Hz, but each point in a sweep has a *different timestamp* — the first point at the start of the spin, the last point at the very end. An IMU samples at 200 Hz. The camera fires at 15 Hz. They are all trying to tell you something about the same robot's motion, but they are all looking at *slightly different moments in time*.

Here's the classical discrete-time problem. Older systems (like FAST-LIO, which Meridian is built on) would do this: run the optimizer at fixed intervals, say every 10 Hz. At each solve, pick a single pose estimate for that instant and use it for all the points in the LiDAR sweep that arrived during that period. This is called the "piecewise-constant-pose" approximation.

```
LiDAR sweep [t=0.00s to t=0.10s]
  Point 1: t=0.00s  ← use T(0.05s) for all points
  Point 2: t=0.02s  ← same T
  Point 3: t=0.04s  ← same T  (ERROR: point 3 is 40ms earlier than point 1)
  ...
  Point n: t=0.10s  ← same T
```

Each point is stamped at its own time, but you register it at a *wrong* time. Over a 100 ms spin, a fast-moving robot can rotate noticeably. The farther a point is from the chosen "keyframe time," the more the registration error — this distortion is called **motion blur** or **motion distortion**.

The **continuous-time** solution: instead of one pose per interval, compute a **smooth function** T(t) that gives you a pose at *any* time $t$. Now every LiDAR point can be registered at its *own true time*:

```
LiDAR sweep [t=0.00s to t=0.10s]
  Point 1: t=0.00s  ← query T(t=0.00s) from the smooth function
  Point 2: t=0.02s  ← query T(t=0.02s)  ✓ exact
  Point 3: t=0.04s  ← query T(t=0.04s)  ✓ exact
  ...
  Point n: t=0.10s  ← query T(t=0.10s)  ✓ exact
```

This is **deskewing** — compensating for motion during the sweep — and in the continuous-time world it is free. You evaluate the trajectory at each point's timestamp. No separate motion-compensation pass, no approximation. This is the central advantage Meridian gains over the piecewise-constant approach.

### 2.2 What is a B-spline, intuitively?

A **B-spline** is a smooth curve built from a small number of control points. Think of it like a flexible ruler: you push on a few control points (the "knots"), and the curve bends smoothly through and around them.

```
Control points (knots):
  K₀ ──o
       │  \
  K₁ ──o   \___
       │        \  [the actual smooth curve]
  K₂ ──o        /
       │       /
  K₃ ──o──o
```

The magic of a B-spline is **local control**. If you nudge K₁, only the curve *near* K₁ changes. Distant parts of the curve are unaffected. This is different from, say, a polynomial fit through all the points, where moving one point can cause wild oscillations everywhere.

In Meridian's case, instead of one control point per time interval, we have a **deque of control points**, each roughly one knot spacing apart. The trajectory T(t) is defined as a smooth blend of exactly **four** nearby control points at any queried time t.

### 2.3 Why exactly four control points? The cubic basis.

Meridian uses a **cubic** B-spline — degree 3, which means the curve is a polynomial of degree 3 (a cubic) on each interval between knots. Why cubic?

- **Degree 1** (linear): piecewise-straight segments. Deskew would be piecewise-constant angular velocity $\omega$ — still motion blur.
- **Degree 2** (quadratic): smoother, but the second derivative (acceleration) has discontinuities. The IMU constraint (which depends on second derivatives) would be jagged.
- **Degree 3** (cubic): the second derivative is continuous (smooth acceleration). The IMU sees smooth acceleration, which is physical. This is $C^2$ continuity (continuous up to the second derivative).
- **Degree 4+**: overkill. The computation cost grows with degree. A cubic is the sweet spot.

**The local-support property:** for a cubic B-spline, each point on the curve depends on exactly **four** local control points. No more, no less. If you query T(t) for some time t, only the four knots bracketing that interval matter. The other thousand control points in your deque are irrelevant to that evaluation.

This sparsity is crucial for the solver. When you optimize the control points, the gradient of the objective at one point touches only four nearby control points. The resulting Hessian (the matrix you invert) is **banded** — only entries near the diagonal are nonzero — and solving a banded system is *fast*, even for thousands of unknowns.

### 2.4 Meridian's split SO(3) × ℝ³ form

Meridian does **not** use a unified SE(3) spline (a single smooth curve in the 6D Lie group of rigid motions). Instead, it splits the trajectory into two independent splines:

- **An SO(3) spline** for rotation: $R_{W\,F_e}(t)$, represented by a cubic B-spline over orientation space.
- **An ℝ³ spline** for translation: $p_{W\,F_e}(t)$, represented by a cubic B-spline over 3D Euclidean space.

Why split? Three reasons.

1. **Computation:** the mathematics is cleaner. The rotation spline naturally works with angular velocity (the first derivative), and the translation spline with velocity and acceleration (first and second derivatives). There are no cross-terms — no SE(3) adjoints needed.

2. **Physics:** the gyroscope measures angular velocity, which should be the derivative of the rotation spline. The accelerometer measures linear acceleration (after subtracting gravity and IMU bias), which should be the second derivative of the translation spline. Each sensor cleanly couples to one spline.

3. **Numerics:** SOTA SLAM systems (CLINS, Coco-LIC, basalt) find that split splines are faster and more stable than unified SE(3) splines. The decoupling avoids the screw-coupled covariance of a unified representation.

The downside: you carry more state (two splines instead of one). But the upside — clean sensor coupling, better numerical properties — wins. Meridian inherits the basalt-headers `So3Spline` and `RdSpline` from the open-source libraries, so the battle-tested implementation is already there.

### 2.5 Control points are NOT poses: the warm-start offset

Here is a subtle but critical idea: **a control point is not a pose on the trajectory**. This trips up many people reading the code.

Because the blending is cumulative (a mathematical detail we'll get to), a control point with grid time $t_j$ *peaks in influence* not at time $t_j$ but roughly one knot interval *earlier*, at time $t_j - \Delta t$.

Concretely, if your knot spacing is 50 ms:

```
Control point C₅ at grid index 5 ──o  (time 250 ms)
                                   │
                        peaks in influence at
                                (250 - 50) = 200 ms
```

When you warm-start the optimizer from IMU integration (the "seed" guess of the trajectory from integrating accelerometer and gyroscope), you must account for this. If you naively set $C_j = T_{\text{seed}}(t_j)$ — the seed pose at grid time — the resulting spline curve will be *delayed* by one interval relative to your seed. You'd be optimizing a trajectory that lags the measurements.

**The fix:** seed each control point from the *earlier* time: $C_j \leftarrow T_{\text{seed}}(t_j - \Delta t_{\text{local}})$. Now when the spline blends them together, the blending "shifts" the influence forward, and the final curve lands on the seed trajectory rather than a delayed copy.

This is a property of the *curve*, not the control points. It's why Meridian's documentation says: "Knot seeding by IMU integration is a property of the *curve*, never $C_j = T(t_j)$."

### 2.6 A worked example: evaluating the spline at one instant

Let's say you have four consecutive control points (rotations and translations) at knot times 100 ms, 150 ms, 200 ms, 250 ms, and you want to query the pose at time **t = 175 ms** — a moment *between* the 150 ms and 200 ms knots.

**Step 1: Which knot interval?**

Interval 1 contains times [100, 150).  
Interval 2 contains times [150, 200). ← **175 ms is here**.  
Interval 3 contains times [200, 250).

So you're in interval 2, bracketed by control points at indices i, i+1, i+2, i+3 (say, knot indices 1, 2, 3, 4).

**Step 2: Normalize time to the interval.**

The interval spans [150 ms, 200 ms), a width of 50 ms. The time 175 ms is 25 ms into the interval. Normalized:

$$u = \frac{175 - 150}{200 - 150} = \frac{25}{50} = 0.5$$

So $u \in [0, 1)$, where $u=0$ is the start (150 ms) and $u=1$ would be the end (200 ms, exclusive).

**Step 3: Evaluate the cubic basis (blending weights).**

A cubic B-spline uses cumulative blending weights — that's the "cumulative" part of "cumulative cubic." The weights are given by a matrix multiplication:

$$\lambda(u) = \tilde M^{(4)} \, [1, u, u^2, u^3]^T$$

where $\tilde M^{(4)}$ is the **cumulative matrix** for cubics (the Kim/basalt convention):

$$\tilde M^{(4)} = \frac{1}{6} \begin{bmatrix} 6 & 5 & 1 & 0 \\ 0 & 3 & 3 & 0 \\ 0 & -3 & 3 & 0 \\ 0 & 1 & -2 & 1 \end{bmatrix}$$

With $u = 0.5$, compute:

$$[1, u, u^2, u^3]^T = [1, 0.5, 0.25, 0.125]^T$$

Multiply by the matrix:

$$\lambda(0.5) = \frac{1}{6} \begin{bmatrix} 6 & 5 & 1 & 0 \\ 0 & 3 & 3 & 0 \\ 0 & -3 & 3 & 0 \\ 0 & 1 & -2 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 0.5 \\ 0.25 \\ 0.125 \end{bmatrix}$$

Row 1: $\frac{1}{6}(6 \cdot 1 + 5 \cdot 0.5 + 1 \cdot 0.25 + 0 \cdot 0.125) = \frac{1}{6}(6 + 2.5 + 0.25) = \frac{8.75}{6} \approx 0.458$

Row 2: $\frac{1}{6}(0 + 3 \cdot 0.5 + 3 \cdot 0.25 + 0) = \frac{1}{6}(1.5 + 0.75) = \frac{2.25}{6} \approx 0.375$

Row 3: $\frac{1}{6}(0 - 3 \cdot 0.5 + 3 \cdot 0.25 + 0) = \frac{1}{6}(-1.5 + 0.75) = \frac{-0.75}{6} \approx -0.125$

Row 4: $\frac{1}{6}(0 + 1 \cdot 0.5 - 2 \cdot 0.25 + 1 \cdot 0.125) = \frac{1}{6}(0.5 - 0.5 + 0.125) = \frac{0.125}{6} \approx 0.021$

So $\lambda \approx [0.458, 0.375, -0.125, 0.021]^T$. (Check: they don't sum to 1, but that's okay for cumulative weights — they sum to 1 *in the context of the blending formula below*.)

**Step 4: Blend the four control points.**

For translation, the blended position is:

$$p_{W\,F_e}(0.5) = p_i + \lambda_1 \, d^p_1 + \lambda_2 \, d^p_2 + \lambda_3 \, d^p_3$$

where $d^p_j = p_{i+j} - p_{i+j-1}$ are **differences** between consecutive control points.

If the control points are $p_1 = [0, 0, 0]$, $p_2 = [1, 0, 0]$, $p_3 = [1.5, 0.5, 0]$, $p_4 = [2, 1, 0]$:

$$d^p_1 = p_2 - p_1 = [1, 0, 0]$$
$$d^p_2 = p_3 - p_2 = [0.5, 0.5, 0]$$
$$d^p_3 = p_4 - p_3 = [0.5, 0.5, 0]$$

Blend:

$$p(0.5) = [0, 0, 0] + 0.458 [1, 0, 0] + 0.375 [0.5, 0.5, 0] - 0.125 [0.5, 0.5, 0]$$
$$= [0, 0, 0] + [0.458, 0, 0] + [0.188, 0.188, 0] - [0.062, 0.062, 0]$$
$$= [0.584, 0.126, 0]$$

**Rotation** works similarly, but on the manifold SO(3) — you blend the *difference operators* (log of relative rotations) and then exponentiate back. The details are in Appendix R.1 of spec 04; the idea is the same: four control rotations, blend them, get a rotation.

**Step 5: Get velocity and acceleration for free.**

Because the position $p(u)$ is a cubic polynomial in $u$, you can differentiate it analytically:

$$\dot p = \frac{dp}{du} = \frac{1}{\Delta t} \, \lambda'(u) \cdot [\text{same blends}]$$

where $\lambda'(u)$ is the derivative of the blending function (computed from the matrix's derivative). No numerical differentiation, no finite differences — just chain the derivative through the basis. This gives you velocity.

Similarly, the second derivative gives acceleration:

$$\ddot p = \frac{d^2 p}{du^2} \cdot \ldots$$

This is why **analytic derivatives** matter: the IMU residual reads $\ddot p$ directly from the spline (after rotating into the body frame and subtracting gravity). The solver has exact derivatives, so it converges faster and more reliably than if you had to approximate derivatives numerically.

### 2.7 Knots, windows, and adaptive spacing

The control points live in a **deque** (double-ended queue), one of the engineering choices Meridian makes. The deque grows as new LiDAR sweeps arrive (extending forward in time) and shrinks as old sweeps drop out of the optimization window (trimming from the back).

**Non-uniform spacing (adaptive knots):** the time gaps between knots are *not* constant. Early in a slow motion, you might have knot spacing of 50 ms. In a high-speed maneuver, you might pack them tighter (say 25 ms) to capture curvature. Too coarse and the cubic basis cannot fit a tight motion; too fine and you have too many variables.

Meridian manages this by:

1. **Growing:** when a new sweep arrives, extend the spline forward with new control points at times predicted from IMU integration (the "seed"), spaced according to motion intensity.
2. **Sliding:** every ~100 ms, the window slides forward, and old control points whose data is fully processed drop off the back.
3. **Virtual time remapping:** the non-uniform spacing is *realized* internally by remapping the normalized parameter $u$ — under the hood, the basalt kernels still assume uniform spacing in "$u$-space," but a non-linear time transformation handles the real-world variable spacing. This is Coco-LIC's engineering trick; it means you get the analytic Jacobians of the uniform basalt code without hand-deriving non-uniform ones.

### 2.8 The tail-knot problem: what to do past the last measurement

Here's a practical problem: your newest measurement arrived at time $t = 1.5$ s, but you have control points extending to $t = 2.0$ s (seeded from IMU integration guessing future motion). Those tail knots at $t \geq 1.6$ s have *no measurement data reaching them*. The cost function is completely flat in those variables — there's nothing pulling them in any direction.

If you leave them free, the optimizer is indifferent. A bad seed can poison them, and they'll drag neighboring (measured) control points astray. If you fix them at their seed values, they can diverge from the true trajectory as motion changes.

Meridian's solution uses **three coordinated mechanisms** (spec 04 §1.3.1):

1. **Tail-knot pinning:** any control point whose basis weight over all measured times is exactly zero is locked constant at its IMU seed value for this solve. Once real data covers it in a later sweep, it unfreezes automatically.

2. **Tail re-seeding:** before each solve, re-integrate the IMU forward from the latest solved time and update the pinned tail knots with the freshest seed. This ensures the extrapolation always uses the newest inertial prediction, not a stale guess.

3. **Tail anchors:** weak residual terms (soft priors) tying the *velocity* and *angular rate* of the tail to the seed's predictions, with low weight so any real measurement overrides them. These smooth the tail without fixing it to point values.

Together, these prevent null-space pollution while letting the tail respond to measured data the instant it arrives.

### 2.9 Derivatives: velocity and angular velocity from the spline

One of the big wins of continuous-time is that **velocity and angular velocity are analytically available**.

From the SO(3) spline $R_{W\,F_e}(t)$, the body angular velocity (what the gyroscope measures) is:

$$\omega_{W\,F_e}(t) = (R_{W\,F_e}(t)^T \, \dot R_{W\,F_e}(t))^\vee$$

where $(\cdot)^\vee$ is the "skew inverse" (extract the 3D vector from a skew-symmetric matrix). The derivative $\dot R$ is computed by differentiating the blending weights, not by finite differencing.

From the ℝ³ spline $p_{W\,F_e}(t)$, the velocity is:

$$v_{W\,F_e}(t) = \dot p_{W\,F_e}(t)$$

and acceleration is:

$$a_{W\,F_e}(t) = \ddot p_{W\,F_e}(t)$$

All three are **closed-form expressions** in the control points and basis derivatives. No numerical approximation. This is fed directly to the IMU residual (§3.3), which compares:

- *Expected* angular velocity (from the spline): $\omega_{W\,F_e}(t_i)$
- *Measured* angular velocity (from the gyroscope): $\omega_m(t_i) - b_g - n_g$

and similarly for acceleration. Because the expectation is analytic, the residual's Jacobian w.r.t. the control points is exact, and the solver has high-quality gradients.

### 2.10 The global picture: from measurement to trajectory to output

Let's trace the flow:

**Input:** Raw measurements arrive at L1.

- LiDAR sweep: raw points $(x, y, z)$ in *sensor frame*, each with timestamp $t_{\text{offset}}$ relative to sweep start.
- IMU samples: $(a_m, \omega_m)$ at full rate.
- Camera frame: image at time $t$.
- GNSS fix: (lat, lon, alt) at time $t$.

**L2 frontend wakes up at ~10 Hz** and assembles a batch.

- Extend the spline forward, seeding new control points from IMU integration.
- For each LiDAR point $\mathbf{p}^L_j$ with offset $t_j$, query the spline at time $t_j$ to get $T_{W\,F_e}(t_j)$, transform the point to world frame, and match it against a local map.
- For each IMU sample at time $t_i$, query the spline's $\omega(t_i)$ and $\ddot p(t_i)$ and build a residual.
- For the camera image at time $t$, query the spline's $T_{W\,F_e}(t)$ and project map points.
- For GNSS, use it as a global position anchor.

**Solve:** Run a windowed non-linear least-squares optimization (Ceres) over the active control points.

- LiDAR point-to-plane residuals, IMU residuals, visual residuals, and GNSS residuals all compete to explain the measurements.
- The solver adjusts the control points to minimize the total error.
- Because each residual touches only 4 control points, the Hessian is banded and solving is fast.

**Output:** evaluate the spline to produce a pose and keyframe decision.

- At sweep times, evaluate $T_{W\,F_e}(t_{\text{sweep}})$ to get the estimated pose and emit a `KeyframePacket` if motion exceeds thresholds.
- Velocity $\dot p(t_{\text{sweep}})$ is emitted as telemetry (not used by the backend, just diagnostic).
- Retained per-keyframe point cloud: for each LiDAR point in this sweep, evaluate the spline at *that point's* time to deskew it into the world frame. This is the true per-point deskew.

**Visual map:** as points arrive, the solver maintains a voxel-hashed map of LiDAR points, each enriched with photometric reference patches (observed from multiple camera views). This unified map is used for both LiDAR and visual association.

### 2.11 Continuous-time LIVO: synthesis

The front-end's beauty is that **continuity is free**. There is no separate deskewing step. No "batch-process the sweep into one pose first, then deskew." Every point is implicitly deskewed because it's evaluated at its own time, and as the solver iterates and the control points move, every point is re-deskewed instantly.

This stands in sharp contrast to iEKF or discrete-time filters, where you must choose *one* representative time and approximate. Here, you get the *exact* per-point timestamp and the *exact* derivative of the trajectory at that instant. The spline is the trajectory, and querying it at any time gives you the exact pose and kinematics.

For a robot spinning a LiDAR while taking photos and accelerating, this is a massive advantage: you fuse four sensors, each at its own rate and timestamp, into a single smooth estimate, and every single measurement is evaluated at its true time.

**Chapter summary.** This chapter covered the continuous-time trajectory representation that makes Meridian's frontend special: **why CT matters** (every measurement arrives at a different instant, and CT lets you register each at its true instant without approximation); **B-splines** (smooth curves with local support — only 4 knots affect any point — and $C^2$ continuity for the IMU); **split SO(3) × ℝ³** (rotation and translation as separate splines, cleanly coupled to gyro and accelerometer); **analytic derivatives** (velocity and acceleration are closed-form — exact Jacobians for the solver); **control points vs. poses** (a control point is not a pose; warm-starting must account for its temporal offset); **adaptive knots and tail management** (the window grows and slides, with tail knots pinned and re-seeded to prevent null-space pollution); and the **LIVO synthesis** (one smooth function of time fusing LiDAR, IMU, and camera, with per-point deskew happening automatically at evaluation). The next chapter builds on this: given this trajectory, what residuals drive the optimization?

---

## Chapter 3 — The Measurements: Residuals That Pull the Spline

Every measurement the front-end receives — a LiDAR point, a pixel patch, an IMU sample, a GPS fix — becomes a **residual**: a scalar or vector that says "the spline's pose should satisfy this constraint." The least-squares solver (Ceres) collects thousands of residuals, asks "which trajectory makes all these residuals as small as possible?", and solves the minimization problem. You'll hear jargon like "photometric residual," "point-to-plane," "IMU derivative," and "robust kernel." This chapter explains what each one measures, why that measurement matters, and how a covariance becomes a weight.

**Three principles before we dive in:**

1. **Each residual is evaluated at the measurement's true time.** The spline $T_{W\,F_e}(t)$ is a continuous function, so a LiDAR point with sample time $t_j$ is registered using the pose $T(t_j)$ — exact, not approximated. This is the payoff of continuous-time: deskew (motion compensation) is intrinsic, not bolted on.

2. **Different measurements constrain different things.** LiDAR points anchor the trajectory to the geometric map. IMU measurements constrain the derivatives (velocity, acceleration) of the trajectory. Camera measurements align pixel intensities. GPS fixes anchor to absolute coordinates. One solve running simultaneously means all four sources tug on the spline at once.

3. **Weights come from uncertainty.** A GPS fix with high noise is trusted less (down-weighted). A LiDAR point hitting a grazing incidence (hard to match) is weighted less. The weight usually comes from the measurement's reported covariance (uncertainty). Outliers are handled by **robust kernels** that smooth out hard spikes.

### 3.1 LiDAR point-to-plane: direct, per-point registration

#### Intuition

Imagine you have a 3D scan of a wall. You want to align that scan against a map of the world. One simple idea: *each point should lie on a plane in the map*. The wall is a plane, so each point's distance to the nearest plane should be zero (or very small).

Here's the registration idea:
1. Take a LiDAR point from the scan.
2. Find its position in the world using the spline at that point's sample time.
3. Query the local map (a voxel grid) to find the 5 nearest neighbors and fit a plane to them.
4. Measure the **signed distance** from the point to that plane.
5. Make that distance small.

In continuous-time SLAM, *each point knows its own time*, so the transformation is exact. You don't "deskew the whole scan first" — you evaluate the spline at each point's individual time.

#### The measurement model

A LiDAR point $\mathbf{p}^L_j$ arrives with a sensor-frame position and a per-point time offset $t_j$ (nanoseconds from the scan start). The spline gives you the pose $T_{W\,F_e}(t_j) = (R(t_j), p(t_j))$ at that point's true time. The extrinsic calibration $T_{F_e\,L}$ maps the sensor frame to the estimation body frame. So the point's world position is:

$$\mathbf{p}^W_j = R(t_j)\big(R_{F_e\,L}\,\mathbf{p}^L_j + t_{F_e\,L}\big) + p(t_j)$$

(Read right-to-left: sensor-frame point → body frame → world frame.)

Now you find a plane in the local map. The plane is represented as a unit normal $\mathbf{n}$ and a distance $d$, so the plane equation is $\mathbf{n}^\top \mathbf{x} + d = 0$. The **signed distance** from the point to the plane is:

$$r^{\text{lid}}_j = \mathbf{n}^\top \mathbf{p}^W_j + d$$

This is a **scalar residual**. When it's zero, the point lies exactly on the plane. When it's nonzero, the point is above or below the plane.

#### How planes are found: the ikd-Tree and plane fitting

Each LiDAR sweep queries a **spatial acceleration structure** (an ikd-Tree, an adaptive-KD-Tree variant used in FAST-LIO) to find the `num_match_points = 5` nearest neighbors in the local map. Once you have 5 neighbors, fit a plane by least-squares: solve $A\mathbf{u} = -\mathbf{1}$ using QR decomposition, where $A$ is a $5 \times 3$ matrix of neighbor coordinates centered at their centroid. The solution gives the plane normal and intercept.

**Quality gates:** The plane is accepted only if all 5 neighbors lie within `plane_thresh = 0.1` m of the fitted plane — if they scatter too much, the fit is unreliable. The farthest neighbor must be within `max_match_dist_sq = 5` m² of the query point. These gates reject noisy correspondences.

#### The Jacobian and chain rule

The residual depends on the pose at time $t_j$. When we minimize the least-squares problem, the solver needs the derivative with respect to tiny perturbations of the pose:

$$\frac{\partial r_j}{\partial \delta p(t_j)} = \mathbf{n}^\top$$

(The normal tells you how sensitive the residual is to position changes.)

$$\frac{\partial r_j}{\partial \delta R(t_j)} = -\mathbf{n}^\top R(t_j) [\mathbf{q}_j]_\times$$

where $\mathbf{q}_j = R_{F_e\,L}\mathbf{p}^L_j + t_{F_e\,L}$ is the point in the body frame, and $[\cdot]_\times$ denotes the skew-symmetric matrix. (Rotating the point changes which plane it hits.)

These Jacobians are then chained through the **spline Jacobian** — the derivative of the spline with respect to its control points. Because the spline is a cubic polynomial in normalized time, its derivatives are analytic (no finite-difference approximation needed).

#### Weighting and robust kernels

A **covariance** is the uncertainty: how much do we trust this measurement? For LiDAR, the covariance is typically modeled as:

$$\sigma^2_{\text{lid}}(j) = \sigma_0^2 + \sigma_r^2 \|\mathbf{p}^L_j\|^2 + \sigma_\theta^2 (1 - |\mathbf{n}^\top \hat{\mathbf{r}}_j|)$$

- **$\sigma_0^2$**: baseline per-point noise (e.g., $0.01$ m²).
- **$\sigma_r^2 \|\mathbf{p}^L_j\|^2$**: noise scales with range — distant points are noisier.
- **$\sigma_\theta^2 (1 - |\mathbf{n}^\top \hat{\mathbf{r}}_j|)$**: **grazing-incidence inflation**. When the plane is nearly parallel to the ray ($\mathbf{n} \perp \hat{\mathbf{r}}$), the plane's tangent direction is poorly constrained, so we down-weight it.

The weight is $w = 1/\sigma^2$.

**Robust kernels** (Huber, Cauchy) smooth out outliers. Imagine a plane fit is bad and a point is 10 m away (a wild outlier). In a pure quadratic least-squares, that single point contributes $(10 \text{ m})^2 = 100$ to the cost, dominating the solve. A **Huber kernel** is quadratic near zero (trusting inliers) but linear at large residuals (ignoring huge outliers). A **Cauchy kernel** is even more aggressive, effectively capping the influence of any one point.

#### Bounded factor count

A dense LiDAR sweep can produce thousands of inlier point-to-plane residuals. Adding all of them makes the Hessian huge and the per-step cost unbounded — bad for real-time performance. So the front-end **subsamples by plane-normal stratification**: bin points by their fitted plane normal (six axis-aligned hemispheres plus an oblique bin), allocate a factor budget across bins (each bin gets a minimum floor, remainder distributed proportionally), and sample uniformly within each bin. This preserves constraints on weak axes (e.g., a corridor's vertical constraint) rather than naively decimating and losing rare normals.

### 3.2 Visual photometric residual: sparse-direct, LiDAR-depth

#### Intuition: brightness constancy

Imagine looking at a textured wall and photographing it from two different positions. The wall's surface reflectance doesn't change, so the same pixel patch should look the same in both images — just warped geometrically because of the viewpoint change.

In traditional structure-from-motion, you'd find corner features (SIFT, ORB) and match them across frames — **feature-based**. FAST-LIVO2 (and now Meridian) does something different: **direct visual registration** using raw pixel intensity.

The idea: take a reference patch (e.g., an $8 \times 8$ grid of pixels) from the LiDAR map. Project it into the current camera image using the current pose. Compare the pixel values. If they match (same brightness), the pose is probably right. If they don't match, adjust the pose until they do.

The LiDAR map provides **depth**, so you don't need stereo or triangulation — depth is free from the 3D scan.

#### The measurement model and photometric chain rule

The residual is per **map point** $\mathbf{P}^W$ (a 3D position from the LiDAR map). Project it into the camera:

$$\mathbf{P}^C = R_{C\,F_e} \big( R_{W\,F_e}(t)^\top (\mathbf{P}^W - p(t)) \big) + t_{C\,F_e}$$

where $(R(t), p(t))$ is the spline pose at the image's mid-exposure time $t$. Then project to pixel coordinates: $\mathbf{u} = \pi(\mathbf{P}^C)$ using the camera's intrinsics.

The reference patch is warped into the current view using a **homography** (affine map) based on the LiDAR plane normal — if the patch lies on a plane, the warp is exact. Then the **photometric residual** over pixel offsets $\Delta$ is:

$$r^{\text{vis}}_\Delta = \tau_{\text{cur}} \, I_{\text{cur}}(\mathbf{u} + \Delta) - \tau_{\text{ref}} \, P^{\text{warp}}[\Delta]$$

where:
- $\tau$ is the **inverse exposure** (a nuisance parameter; changing exposure changes brightness).
- $I_{\text{cur}}$ is the current image intensity.
- $P^{\text{warp}}$ is the reference patch, geometrically warped and brightness-normalized.

**Brightness normalization:** The inverse exposure model is FAST-LIVO2's trick to handle auto-exposure. If the camera auto-adjusts exposure (or vignetting changes), the brightness scales. By including $\tau_{\text{cur}}$ and $\tau_{\text{ref}}$ as optimization variables tied by a weak random-walk prior, the solver adjusts exposure to match the data rather than treating the change as a residual.

The Jacobian chain rule links this to pose:
$$\frac{\partial r}{\partial \delta R(t)} = J_{\text{img}} J_\pi \, R_{C\,F_e} [R(t)^\top (\mathbf{P}^W - p(t))]_\times$$
$$\frac{\partial r}{\partial \delta p(t)} = -J_{\text{img}} J_\pi \, R_{C\,F_e} R(t)^\top$$

where $J_\pi$ is the projection Jacobian and $J_{\text{img}}$ is the image gradient scaled by exposure. (It's the standard photometric chain rule.)

#### Reference patches and the active visual-point map

The LiDAR map and visual map are **unified** — the same 3D points, but the visual map adds **reference patches**: small $8 \times 8$ grayscale images taken when the point was first observed, plus metadata about viewpoints and brightness.

A point accumulates observations: as the robot revisits it from different angles, new patches are added (if they offer enough parallax). The observation list is capped at 30 patches (to save memory) and evicted by least-informative-first. The reference patch actually used is the **medoid** — the observation whose viewpoint is most central to all others (resistant to drift when the platform moves far away).

Spatial management: points are stored in an ordered voxel-keyed map. As the platform moves away, points outside a $\pm 20$ m box are deleted from the active set — but retained in the persistent per-keyframe store so loop closure can re-initialize them.

#### Candidate selection and gating

Not every map point projects into the current frame. The visual residual is expensive, so the front-end **limits to one candidate per image grid cell** (32×32 pixel grid, say). For each cell, it picks the single best-scoring candidate by running each through a **deterministic fixed gate sequence**:
1. Grid occupancy (cell has room).
2. Depth continuity (patch doesn't straddle an occlusion).
3. Warp validity (homography determinant is well-conditioned; normal is front-facing).
4. Normalized Cross-Correlation (NCC ≥ threshold).
5. Photometric SSD (sum-of-squared-difference < outlier threshold).
6. Robust kernel (Huber or Cauchy).

This deterministic gate order ensures reproducibility and avoids data-dependent selection order bugs.

#### Degenerate-warp guards

The homography warp is valid only when the patch lies on a well-defined plane. Two guards prevent blowup:
- If the $2 \times 2$ affine part of the warp has determinant outside $[0.1, 10]$ or condition number above 50, the warp is **degenerate** (near-rotation, extreme scale, grazing view) and the point is dropped for this frame.
- If the plane normal is back-facing (pointing away from the camera), it's flipped to face the camera.

These guards ensure a bad geometry silently drops a point rather than corrupting the solve with a NaN or nonsensical residual.

### 3.3 IMU derivative residual and bias random-walk

#### Intuition: the IMU tells you about derivatives

An IMU measures specific force (acceleration) and angular rate (gyro). In a continuous-time system, these are tied to the **derivatives** of the trajectory:

- The spline's body angular velocity $\omega_{W\,F_e}(t)$ should match the gyro reading (minus the gyro bias).
- The spline's linear acceleration should match the accelerometer reading (minus accel bias, minus the projection of gravity).

The IMU residual is **not** an integration step or a prediction — it's a direct constraint on the spline's derivatives. This is the power of analytic derivatives: the cubic B-spline has a closed-form expression for velocity and acceleration, so you can evaluate them at the IMU's exact sample time (not rounded to a keyframe).

#### The kinematic model and IMU measurement

The reference IMU dynamics (strapdown INS in the world frame $W$):

$$\dot R_{W\,F_e} = R_{W\,F_e} [\omega_m - b_g - n_g]_\times$$
$$\dot v = R_{W\,F_e} (a_m - b_a - n_a) + g_W$$

where:
- $\omega_m, a_m$ are the raw gyro and accel readings.
- $b_g, b_a$ are **biases** (slowly drifting offsets).
- $n_g, n_a$ are white noise.
- $g_W$ is gravity (a state variable optimized on the 2-DoF sphere $S^2$).

The front-end **does not integrate** this ODE. Instead, it reads off the spline's derivatives and enforces the constraint:

For the gyro:
$$r_i^\omega = (\omega_{W\,F_e}(t_i) + b_g) - \omega_m$$

(The spline's body rate plus the gyro bias should match the gyro reading.)

For the accel:
$$r_i^a = (R_{W\,F_e}(t_i)^\top (\ddot p(t_i) - g_W) + b_a) - a_m$$

(The world acceleration projected back to the body frame, minus gravity, plus the accel bias, should match the accel reading.)

These are **vector residuals** (3D each). They're weighted by the IMU noise densities (Allan-variance parameters from the IMU spec).

#### Biases as a state: the random-walk timeline

Biases don't jump discontinuously — they **drift slowly**. The front-end models this with a **multi-knot bias timeline**: a piecewise-linear interpolation between bias knots spaced 500 ms apart. At an arbitrary time, the bias is interpolated between the bracketing knots.

To prevent the bias from drifting away, consecutive bias knots are tied by a **random-walk residual**:

$$r_{k}^{b_g} = b_g^k - b_g^{k-1}, \quad r_k^{b_a} = b_a^k - b_a^{k-1}$$

weighted by the bias covariance $\sigma_{bg}^2, \sigma_{ba}^2$ scaled by the time interval. (If a knot is 500 ms away, the variance is 500 ms times the rate — more time = more drift allowed.)

#### Gauge deficiency and observability

Here's an important point: **the IMU alone is gauge-deficient**. The gyro residual depends only on the *body-frame* angular velocity (invariant to global rotation) and the accel residual depends on acceleration (invariant to any constant velocity or position offset). That's **9 unobservable degrees of freedom** (3 for rotation, 6 for affine position/velocity).

In a real system, the **LiDAR point-to-plane constraints** and **GNSS fixes** anchor the trajectory to the map and absolute coordinates. The IMU fills in between, densely constraining the spline's shape wherever LiDAR and visual measurements are sparse.

### 3.4 GNSS absolute-position residual (conservative)

#### Intuition: ground truth from satellites

GPS (or RTK-GNSS) gives you an absolute position on Earth in geodetic coordinates (latitude, longitude, altitude). The signal is coarse (meters of noise) and often unreliable (urban canyon, multipath), but when you have open sky and a good receiver, it's solid ground truth.

Meridian fuses GNSS **conservatively**: it uses the fix to anchor the drift but doesn't let a bad fix corrupt the local estimate.

#### The measurement model

A GNSS fix provides the **antenna position** in ENU (East-North-Up) coordinates. You have a calibrated lever-arm (the offset from the IMU to the antenna), so the world antenna position is:

$$\mathbf{p}_{W,\text{ant}} = p(t) + R(t) \, t_{F_e,\text{ant}}$$

The GNSS measurement gives you:
$$\mathbf{z}^W = T_{MW}(\text{LLA} \to \text{ENU}) \quad \text{(convert fix to world)}$$

The residual is:
$$r^{\text{gnss}} = \mathbf{z}^W - \mathbf{p}_{W,\text{ant}}$$

**Bind to the spline pose at the fix time, not the nearest keyframe.** This is critical: because the trajectory is continuous, $p(t)$ is the spline **interpolated** to the fix's exact timestamp. Snapping to the nearest keyframe would bake in the antenna's displacement over that interval as a systematic error — a classic GNSS bug. Continuous-time fixes this for free.

#### Gating and robustness

GNSS is unreliable, so several guards protect against bad fixes:

1. **Covariance floors**: The receiver reports covariance (uncertainty), but it's often optimistic. The front-end floors it by fix type:
   - RTK Fixed: 0.05 m (horizontal), 0.10 m (vertical)
   - RTK Float: 0.50 m (horizontal), 1.00 m (vertical)
   - DGPS: 1.50 m, 3.00 m
   - SPP: 3.00 m, 6.00 m

2. **Innovation gating (Mahalanobis)**: A fix is admitted only if its normalized innovation $\|r^{\text{gnss}}\|_\Sigma \leq 3.0$ (a 3-sigma gate). The gate includes the current odometry uncertainty, so after a long GNSS gap when odometry has drifted, the threshold loosens.

3. **Re-acquisition persistence**: After a gap or a run of gated-out fixes, GNSS is re-admitted only once 5 consecutive fixes pass the gate. This prevents a single multipath spike from jerking the window.

4. **Robust kernel**: Survivors enter the cost under a Huber or Cauchy kernel, so one bad-but-accepted fix can't dominate.

#### Policy: L2 conservative, L3 authoritative

L2 uses GNSS to constrain drift. **L3 (the backend)** is where the real GNSS fusion happens: it solves for the datum (the $T_{M\text{W}}$ transform from odometry to map), detects and rejects multipath, and integrates GNSS over long arcs. L2 is deliberately lightweight.

### 3.5 The joint window cost and robust kernels

#### Weighting by covariance

Every residual is scaled by its inverse covariance (the information weight). If a measurement has high uncertainty $\sigma^2$, its weight is $w = 1/\sigma^2$ (low). If it's precise, the weight is high. This **maximum-likelihood** weighting is optimal: measurements with more information get more influence.

In matrix form, a residual's contribution to the least-squares cost is:
$$\|r\|_\Sigma^2 = r^\top \Sigma^{-1} r$$

Diagonal covariances are simple; dense blocks (e.g., the IMU's coupled accel/rotation constraints) use the full matrix inversion.

#### Robust kernels: smooth out outliers

A **robust kernel** $\rho(s)$ is a smooth function that is quadratic near zero (trusting good inliers) but sublinear at large values (ignoring outliers). Common choices:

**Huber kernel:**
$$\rho(s) = \begin{cases} s^2 & \text{if } s \leq \delta \\ 2\delta s - \delta^2 & \text{if } s > \delta \end{cases}$$

Quadratic inside $[-\delta, \delta]$, linear outside. Smooth transition at the threshold.

**Cauchy kernel:**
$$\rho(s) = \ln(1 + s^2)$$

Even more aggressive: outliers contribute logarithmically (slowly).

The least-squares cost becomes:
$$\mathcal{C}(X) = \sum_j \rho(\|r_j\|^2_\Sigma)$$

The solver minimizes this, automatically down-weighting bad residuals.

#### The joint window cost

Over an entire solve, the window minimizes:
$$\mathcal{C}(X) = \sum_{\text{lidar}} \rho(\|r^{\text{lid}}\|^2) + \sum_{\text{visual}} \rho(\|r^{\text{vis}}\|^2) + \sum_{\text{imu}} \|r^\omega\|^2 + \sum_{\text{imu}} \|r^a\|^2 + \sum_{\text{bias}} \|r^b\|^2 + \sum_{\text{exposure}} \|r^\tau\|^2 + \sum_{\text{gnss}} \rho(\|r^{\text{gnss}}\|^2) + \text{(marginalization prior)} + \text{(tail anchors)}$$

The IMU residuals don't have a robust kernel (they're well-behaved and outliers are rare). LiDAR, visual, and GNSS residuals all have robust kernels to handle occasional bad matches, multipath, or occlusion.

### 3.6 Extrinsic and gravity refinement (gated by observability)

#### Extrinsic calibration as a variable

The **extrinsic transforms** $T_{F_e\,L}$ (IMU→LiDAR) and $T_{F_e\,C}$ (IMU→camera) are normally calibrated offline and held fixed. But in practice, mechanical vibration or thermal drift can shift them. Meridian **refines extrinsics online** — they're optimization variables in the window.

The refinement is **gated by observability**: if the window is executing a smooth straight-line motion (low rotation, no acceleration), there's no excitation to constrain the extrinsic. Forcing it to refine would just add noise. So the front-end checks per-axis excitation (stored in the observability report) and **freezes the Jacobian columns** (zeroes the gradients) when a DoF is unexcited. As soon as motion excites that DoF, refinement resumes.

#### Gravity refinement

Gravity $g_W$ (the 3D acceleration due to Earth's gravitational field) is also a state variable. It's **stored on the 2-DoF sphere** $S^2$ with fixed magnitude 9.81 m/s² — only direction varies. This prevents the magnitude from drifting (a known iEKF failure mode). Refinement is gated: gravity is refined only when there's sufficient translational excitation (long enough motion with diverse accelerations). In short-range deployments it's held constant at its initialization.

#### Inverse exposure refinement

The inverse exposure $\tau = 1/t_{\text{exp}}$ (a nuisance parameter for photometric brightness) is refined per frame with a weak random-walk tie to the previous frame. If there's not enough photometric evidence, the tie pulls it back to the neighbor's value. This handles auto-exposure steps smoothly.

### 3.7 Per-axis observability: telling which directions are weak

#### The observability problem

Not all directions are equally observable. Imagine a corridor:
- **Longitudinal motion** (down the corridor): LiDAR points constrain range → well-observed.
- **Lateral motion** (across the corridor): wall on both sides gives parallel planes → cross-range is **weak**.
- **Vertical motion**: floor and ceiling → observed if you're looking straight, otherwise weak.

A binary "degenerate/not-degenerate" flag is too coarse. Meridian computes a **6-DoF observability score** (one per axis: $t_x, t_y, t_z, r_x, r_y, r_z$) using the **information matrix** (the Hessian of the LiDAR residuals alone):

$$\Lambda_{\text{pose}} = \sum_j w_j J_j^\top J_j$$

where $w_j = 1/\sigma_j^2$ and $J_j$ is the Jacobian of the LiDAR residual at the current pose.

Eigendecompose $\Lambda$: directions with small eigenvalues are weakly observed. Map each eigenvalue to a $[0, 1]$ score using:
$$\text{score} = \frac{\lambda}{\lambda + \kappa_{\text{deg}}}$$

where $\kappa_{\text{deg}}$ is a degeneracy threshold. A score near 1 means well-observed; near 0 means weak.

#### Why this matters

The observability scores ship in the **KeyframePacket** to the backend. The backend uses them to **inflate covariance** on weak axes:
$$\Sigma_{\text{inflated}} = \text{diag}(\text{inflation factors}) \cdot \Sigma_{\text{marginal}}$$

This tells the graph optimizer "don't trust the pose estimate in this direction." Weak axes are then anchored by visual or GNSS constraints downstream.

### Summary: residuals as constraints

Each residual family tightens a different aspect:

| Residual | Constrains | Measurement | Weighting |
|---|---|---|---|
| **LiDAR point-to-plane** | Position (range to map) | Signed distance to plane | Range-aware, grazing-inflation |
| **Visual photometric** | Pose via pixel brightness | Patch SSD / NCC | NCC score; grid one-per-cell |
| **IMU gyro** | Rotation and angular velocity | Body-frame rate | Allan-variance white noise |
| **IMU accel** | Translation and linear acceleration | Specific force | Allan-variance white noise |
| **IMU bias RW** | Bias drift | Bias difference (knot-to-knot) | Bias random-walk variance |
| **GNSS position** | Absolute location | ENU antenna position | Covariance floor + Mahalanobis gate |
| **Tail anchors** | Unmeasured span near window end | Velocity/rate | Weak (seed carries them) |

All four sensors feed into **one** window minimization problem. The continuous-time spline acts as the glue, evaluated at each measurement's true time, giving exact deskew and multi-rate fusion for free. The resulting trajectory is **drift-free locally, consistent globally** — ready to hand to the backend. The next chapter explains *how* that one minimization problem is actually solved in real time, and how old data leaves the window without being forgotten.

---

## Chapter 4 — The Sliding-Window Solve and Marginalization

Welcome to the engine of the front-end. You now understand the spline's shape, the four streams of measurements, and how they become residuals in a cost function. This chapter explains **how Meridian actually estimates** — the bounded solve that runs 10 times a second, what happens when old data leaves the window, and how the front-end keeps information from the past without keeping the full history.

### 4.1 The Real-Time Window: Why Not Solve Everything?

Imagine you could solve for the entire trajectory of your robot from startup to now. That sounds ideal — you would use *all* the data, and the estimate should be optimal. But it is impossible in a live system:

**The naive approach breaks real-time:**

```
Solve the entire trajectory T(t) from t=0 to t=now
  using ALL LiDAR points, ALL IMU samples, ALL camera frames, ALL GNSS fixes.

On a 10-minute mission:
  - LiDAR at 10 Hz         → 6,000 scans
  - IMU at 200 Hz          → 120,000 samples
  - Camera at 20 Hz        → 12,000 frames
  - GNSS at 1 Hz           → 600 fixes

Total measurements: ~138,600.
Each point to a variable, each creates a constraint...
The Hessian becomes 138,600 × 138,600 (or thereabouts).
Inverting it: seconds. Minutes.
Your robot is long gone while you are still thinking about where it was.
```

**Real time demands a bounded window:**

Instead, Meridian solves for only a **recent window** of the trajectory — typically **~0.8 seconds** of control points. This window slides forward as new measurements arrive. It is the same strategy as FAST-LIO, iSAM2, and most working real-time SLAM systems: **fixed-lag smoothing**.

Here is the high-level picture:

```
Old data (t < t_window_start)
 └─→ marginalized into a dense PRIOR on the boundary
                                    ↓
┌─────────────────────────────────────────────────────┐
│ ACTIVE WINDOW (solve for these control points)      │
│ t_window_start ──────────────────────→ t_window_end │
├─────────────────────────────────────────────────────┤
│ New measurements:                                    │
│ • LiDAR points at their true times                  │
│ • IMU samples at their times                        │
│ • Camera frames                                     │
│ • GNSS fixes                                        │
│ All fused in ONE non-linear least-squares solve     │
└─────────────────────────────────────────────────────┘
                         ↓
       Solve (Levenberg–Marquardt on Ceres)
                         ↓
┌─────────────────────────────────────────────────────┐
│ Updated window estimate                             │
└─────────────────────────────────────────────────────┘
```

### 4.2 The Outer Loop: One Sweep Cycle

Each time a LiDAR sweep (or IMU batch, or image) arrives, the front-end does **not** solve immediately. Instead, it buffers the measurement, and every ~100 milliseconds (the `knot_dt`), one **outer step** runs. Here is what happens:

**Step 1: Extend the spline**

The spline already covers some trajectory up to time $t_{\text{end}}$. A new bundle of measurements pushes that end forward. Meridian adds new control points to the spline to cover the new time span. How many control points? That depends on **motion energy** — a fast spin or hard acceleration gets more control points; smooth motion gets fewer (adaptive knots, covered in Chapter 2). The new control points are seeded by **IMU-only integration** (the inertial dead-reckoner) so the warm start is already in the right ballpark.

**Step 2: Associate LiDAR points to the map**

Every LiDAR point has a time tag (its offset within the sweep). To use it as a constraint, you must know what it is constrained *to* — i.e., find the nearest surface in the local map and fit a plane to it. This is the **association step** and happens at the *current* spline pose. The plane becomes the target for the point-to-plane residual (Chapter 3.1).

Similarly, camera pixels are matched against the LiDAR map's points viewed under the current pose.

**Step 3: Build and solve the Ceres problem**

Assemble all the residuals (LiDAR point-to-plane, IMU derivative, visual photometric, GNSS, biases, etc.) into one non-linear least-squares problem. Add the marginalization prior from the previous window (the memory of old data; see §4.3). **Solve** using Levenberg–Marquardt (a dampened Gauss–Newton method) under a hard wall-clock deadline — typically 90 milliseconds to meet the 10 Hz cadence.

The solve is a series of **inner iterations**. After a few iterations the poses have shifted, and the associations (which point matches which plane) may no longer be optimal. So the solution branches into an **outer loop** that re-associates after each inner solve, typically running 3–4 times per sweep, until associations stop changing.

**Step 4: Marginalize old control points**

As the window slides forward, the oldest control points leave the window. Rather than discard them, they are **summarized** into a dense linear constraint on the boundary knots (Schur complement; see §4.3). This is how information from the past survives without keeping the variables.

**Step 5: Update the visual map**

New LiDAR points become candidates for visual tracking. Reference patches are added, observations are re-scored, and points far from the robot are dropped to keep memory bounded.

**Step 6: Emit a keyframe**

Every so often (10 Hz IMU / 0.1 s, or after sufficient motion), a **keyframe** is emitted — a `KeyframePacket` containing the current pose, covariance, and a snapshot of the map. This is the **only** thing the front-end hands to the back-end.

### 4.3 Marginalization: Summarizing the Past

**The core idea:** When a control point leaves the window, throw away the variable, but keep a **summary of its influence**.

Imagine a control point $c_i$ that is now too old to stay in the active window. It was constrained by measurements (IMU samples, LiDAR points, etc.). Those measurements also constrain younger control points $c_j, c_k$ that are still in the window. If you simply delete $c_i$, you lose the information it carried about the relationship between $c_j$ and $c_k$.

**Solution:** Use the **Schur complement** to eliminate $c_i$ from the problem. The Schur complement produces a **prior** — a quadratic cost term that encodes: "*if you want $c_j$ and $c_k$ to be good, they should satisfy this constraint.*" This prior is derived from all the measurements that touched the old point.

**A small worked example:**

Suppose you have a Gauss–Newton normal equation (the linearized least-squares problem):

$$
\begin{bmatrix} H_{ii} & H_{ij} \\ H_{ji} & H_{jj} \end{bmatrix} \begin{bmatrix} \delta c_i \\ \delta c_j \end{bmatrix} = \begin{bmatrix} g_i \\ g_j \end{bmatrix}
$$

where $c_i$ is the old point, $c_j$ is a boundary point (still in the window), and $H$ and $g$ are blocks of the Hessian and gradient.

You want to eliminate $c_i$. Solve the first equation for $\delta c_i$:

$$
\delta c_i = H_{ii}^{-1} (g_i - H_{ij} \delta c_j)
$$

Substitute into the second equation:

$$
H_{ji} H_{ii}^{-1} (g_i - H_{ij} \delta c_j) + H_{jj} \delta c_j = g_j
$$

Rearrange:

$$
(H_{jj} - H_{ji} H_{ii}^{-1} H_{ij}) \delta c_j = g_j - H_{ji} H_{ii}^{-1} g_i
$$

The matrix $(H_{jj} - H_{ji} H_{ii}^{-1} H_{ij})$ is called the **Schur complement**. It is a new "prior Hessian" on $c_j$ that encapsulates all the constraints that involved $c_i$. The right-hand side is the corresponding "prior gradient."

When you marginalize a control point, you:

1. Invert its local Hessian block $H_{ii}$.
2. Compute the Schur complement — the new information matrix $\Lambda_{\text{prior}}$.
3. Compute the prior gradient $\mathbf g_{\text{prior}}$.
4. On the next solve, add a **prior residual** to the cost:

$$
r_{\text{prior}} = c_j \boxminus c_{j}^{\text{prior}}
$$

weighted by $\Lambda_{\text{prior}}$ (in information form, not covariance).

**Two crucial properties:**

* **No double-counting:** Measurements are represented **once** — either as residuals (in the current window) or as the marginalization prior (summarizing old measurements). Never both.
* **Exact in the linearization:** The Schur complement is exact *at the linearization point*, meaning if you re-expand the prior as if all the old data were present, you get the same cost. But the linearization point drifts as you iterate, so the prior gradually becomes stale — a known risk (§4.3).

**Meridian's implementation:**

* Control points and bias knots are marginalized as the window slides using the built-in Ceres Schur-complement machinery.
* The prior is stored as a dense $6 \times 6$ (or larger) block on the boundary control points, plus gravity and the active bias knots.
* A **forgetting factor** `marg_prior_scale` (typically 0.5–0.1) is applied at marginalization time: the prior's confidence is deflated by multiplying its square-root information by $\sqrt{\text{scale}}$. This is exponential forgetting — the further back a measurement is, the less confident we are about its linearization (an unfamiliar technique, but the reference systems CLINS and Coco-LIC do the same).

### 4.4 Fixed-Lag Continuous-Time Smoothing

The window constantly slides forward. At any given moment, Meridian solves for:

* **Active control points:** roughly 8–12 points, spanning ~0.8 seconds.
* **Bias knots:** one every 500 ms (sliding timeline; see Chapter 3.3).
* **Gravity direction:** one 2-DoF variable.
* **In-window exposure terms:** one per camera frame.
* **Marginalization prior:** a dense constraint on the boundary points.

**Why is this "fixed-lag"?**

Classical SLAM distinguishes:

* **Filtering:** estimate the *current* state given all past data. ("Where am I *now*?")
* **Smoothing:** estimate the *entire past trajectory* given all the data. ("What is my full history?")

Meridian does **fixed-lag smoothing** — it estimates the trajectory over a fixed time window in the past, and slides that window forward. It is not a filter (which only cares about the present pose), and it is not a global smoother (which re-optimizes the entire past, which is too slow). It is the sweet spot: you get accurate, smooth motion estimates over the window, and the marginalization prior anchors the next window to the past without keeping all the variables.

### 4.5 The Ceres Solver and Levenberg–Marquardt

**What is Levenberg–Marquardt?**

Non-linear least squares is the problem: minimize $\sum_j r_j(x)^2$ where $r_j$ is a residual (a scalar measure of how badly the current state $x$ fits measurement $j$). The Gauss–Newton method linearizes the residuals and solves a quadratic approximation:

$$
H \delta x = -\mathbf g, \quad H = \sum_j J_j^\top J_j, \quad \mathbf g = \sum_j J_j^\top r_j,
$$

where $J_j = \partial r_j / \partial x$ is the Jacobian.

**The problem:** This step can be too aggressive — $\delta x$ can be huge, the linearization breaks down, and you step into a worse solution.

**Levenberg–Marquardt's solution:** Add a **damping term** to the Hessian:

$$
(H + \lambda I) \delta x = -\mathbf g.
$$

When $\lambda$ is small, you trust the Gauss–Newton step. When $\lambda$ is large, you move toward a tiny gradient step. An adaptive rule adjusts $\lambda$ after each iteration: if the step improved the cost, decrease $\lambda$ (trust the curvature more); if it made things worse, increase $\lambda$ (be more conservative). Over iterations, the method converges to a local minimum.

**Why Ceres, not GTSAM?**

Both are industrial-strength nonlinear optimizers. The spec (04 §11) and Appendix R.1 explain the choice:

* **Ceres** is a general-purpose NLLS solver, ideal for the fixed-lag window. It has efficient sparse Cholesky / Schur-complement solvers, out-of-the-box SO(3) manifold support, and band-sparse structure (each measurement touches only 4 control points → banded Hessian → fast solve).
* **GTSAM** excels at incremental iSAM2 updates on factor graphs. L2 doesn't need incrementalism (it re-solves the window from scratch each sweep); GTSAM is reserved for L3 (the backend).

**Analytic Jacobians vs Autodiff:**

Meridian **hand-derives Jacobians** for all residuals (LiDAR point-to-plane, visual photometric, IMU factor). These are given to Ceres as analytic cost functions. Why hand-code?

* **Speed:** Analytic Jacobians are faster to evaluate than automatic differentiation.
* **Clarity:** The code is a direct translation of the math in Chapter 3, making it auditable.
* **Parity testing:** Meridian retains equivalent autodiff versions of every residual solely for testing — every unit test asserts that the analytic Jacobian and autodiff Jacobian agree to machine precision (~1e-9).

### 4.6 Band-Sparse Structure and Real-Time Complexity

Here is why a CT spline with fixed-lag smoothing can run in real time on a Jetson.

**Each measurement touches only 4 control points:**

A cubic B-spline basis has **local support** — the curve in a given time interval depends on exactly 4 consecutive control points (the left and right neighbors of the interval). An IMU sample at time $t$ constrains the spline at $t$, which depends on 4 points. A LiDAR point at time $t_j$ depends on 4 points. All residuals have this property.

**Sparse Hessian structure:**

The Hessian $H = \sum_j J_j^\top J_j$ is the sum of outer products of Jacobians. Since each residual's Jacobian has columns for only 4 control points, $H$ is **banded**:

```
Control points (time → right)
0  1  2  3  4  5  6  7  8  9  10 ...
X  X  X  X  0  0  0  0  0  0  0     Residual 1
0  X  X  X  X  0  0  0  0  0  0     Residual 2
0  0  X  X  X  X  0  0  0  0  0     Residual 3
...
```

Rather than a dense 6000 × 6000 matrix for a 10-minute solve, you have a sparse band — only the diagonals and a few off-diagonals are non-zero. Sparse Cholesky factorization on this band runs in O(n) time instead of O(n³).

**Bounded solve cost:**

The window has a fixed number of control points (~12) × 3 DoF per point = ~36 degrees of freedom, plus biases (~3 knots × 6 DoF = 18), plus gravity (2) and exposure (a few). **The total optimization problem size is constant.** Doubling the number of LiDAR points does not double the solve time — it adds residuals, but the Hessian size does not grow (it has the same band structure). The cost is proportional to the number of residuals times the solve iterations.

**Real-time deadline:**

Ceres is given a hard wall-clock budget of ~90 ms per sweep. The solver runs Levenberg–Marquardt iterations until it converges or the deadline hits, whichever comes first. Hitting the deadline is *not* a failure — it publishes the best iterate so far and logs it as telemetry. A sustained deadline-hit rate is the signal to an operator that the window is too large for the platform (a graceful degradation, not a crash).

### 4.7 Re-association: Outer Loop

The association step (finding which LiDAR point corresponds to which plane, which pixel corresponds to which map point) depends on the **current spline pose**. As the solve iterates, the control points move, the poses at each measurement time change, and the correspondences may no longer be optimal.

**Fixed-cadence re-association:**

Rather than re-associate on every Ceres iteration (expensive and a source of non-determinism), Meridian uses a **nested loop**:

* **Outer loop (association):** At most `max_outer_iters` times (default 4):
  1. Evaluate the current spline at every measurement time.
  2. Re-associate all LiDAR points: kNN + plane fit.
  3. Re-select all visual patches: grid cell, gates, NCC scoring.
  4. Hand the fresh residual set to the inner loop.
  5. Check if the poses have moved enough since the last association; if not, stop (associations are stable).

* **Inner loop (solve):** `reassoc_steps` Ceres iterations (default 2) on a *fixed* residual set.

This decoupling is crucial:

* **Determinism:** The residual set is a function of the pose, not the iteration order.
* **Efficiency:** Association is expensive (kNN lookup, QR plane fit), but you only do it a few times per sweep.
* **Correctness:** By the time you solve again, associations have had time to adjust to the new pose, not thrash around.

**Early termination:**

If the keyframe pose has shifted by less than `assoc_shift_thresh_m` (0.02 m) and `assoc_shift_thresh_deg` (0.2°) since the last association, no new correspondences will be found, so the outer loop stops. On smooth motion, this often means 1–2 outer iterations; on a hard maneuver, 3–4.

### 4.8 Keyframe Emission and the KeyframePacket Hand-Off

The front-end does not emit a keyframe on every measurement. Instead, a **keyframe selection policy** fires based on:

* Translation > 0.5–1.0 m since the last keyframe.
* Rotation > 10–15° since the last keyframe.
* Elapsed time > 1–2 s (heartbeat — keep the graph connected even if the robot is stationary).
* Information-driven criteria (a surge in visual or LiDAR information, or an excitation change).

When a keyframe is due, the front-end packs the following into a `KeyframePacket`:

| Field | Meaning |
|-------|---------|
| `id` | Monotonic integer; unique to this keyframe. |
| `stamp` | The PTP timestamp (nanoseconds) of this keyframe pose. A real measurement instant, not an arbitrary grid time. |
| `T_ref_body` | The spline evaluated at `stamp`, expressed in the reference frame (`Odom`). This is the keyframe pose in the local odometry frame. |
| `constraint_kind` | Normally `RelativeBetween` — a relative pose to the previous keyframe. On a window restart (§4.10), switches to `ImuPreintegration` with the raw IMU summary. |
| `rel_to_id`, `T_relto_this` | The previous keyframe's id and the **relative** transform from that frame to this one (both evaluated from the spline). |
| `constraint_cov` | The covariance of the relative transform — a 6×6 block in the order `[rx, ry, rz, tx, ty, tz]` (rotation-first, to match GTSAM `Pose3`). This is the **window-posterior pose marginal**, recovered by sparse back-substitution over the entire window. |
| `observability` | Per-axis degeneracy scores: 6 numbers in $[0, 1]$ (one per axis: tx, ty, tz, rx, ry, rz) indicating how well-constrained that axis is. Weak axes are a red flag for odometry-only stretches. |
| `cloud_body` | A smart-pointer to the LiDAR scan, deskewed to the keyframe frame $F_{e,\text{stamp}}$ by spline evaluation. This is the data the loop-closure detector and GPU map use. |
| `image` | The camera frame at this keyframe (nullable). |
| `calib_version`, `frontend_kind` | Metadata: the `CalibrationSet` version used, and `1` for CT. |

**The constraint covariance:**

Computing the full window-posterior covariance is expensive — a sparse back-substitution over the whole Ceres problem. To avoid stalling the front-end thread, this computation runs **asynchronously on a low-priority worker thread**. The front-end thread captures a snapshot of the final solve's Ceres problem (knots, biases, gravity, residuals), submits a `KeyframeJob`, and continues. The worker rebuilds a bit-identical Ceres problem from the snapshot, solves for the marginal, chains it into the relative-covariance calculation, and delivers the finished packet to the backend. Because keyframes are emitted at ~0.1–1 s intervals (much slower than the 100 ms solve cadence), the worker has ample time to finish before the next keyframe.

### 4.9 Two Constraint Types: RelativeBetween vs. ImuPreintegration

**Default (steady state): `RelativeBetween`**

On the normal path, Meridian emits a single relative `BetweenFactor(prev_id → this_id)` with the relative covariance. This factor encodes: "the measurement says the relative pose is $T_{\text{prev,cur}}$ with covariance $\Sigma$; the solver should minimize the error of this constraint." The absolute poses of the keyframes are then determined by stitching these relative constraints together in the backend graph.

This design **fuses all sensors internally** (LiDAR + IMU + visual + GNSS) into one window, and summarizes them as a single relative edge. No separate IMU factor, no separate LiDAR factor — just one edge per keyframe pair.

**Why?** Because the IMU information is already baked into the relative covariance. Emitting both an IMU factor and a relative factor would double-count the IMU. The "single clean constraint contract" in spec 01 §6.4 enforces this.

**Fallback (after a window restart): `ImuPreintegration`**

If the window restarts (§4.10) — typically after a divergence or a long sensor gap — the relative pose is no longer reliable. Instead, Meridian falls back to a raw IMU summary:

$$
\Delta R, \quad \Delta v = \int_{t_{\text{prev}}}^{t_{\text{cur}}} v(t) \, dt, \quad \Delta p = \int_{t_{\text{prev}}}^{t_{\text{cur}}} v(t) \, dt
$$

plus the 9×9 covariance and Jacobians w.r.t. biases. The backend then builds a GTSAM `CombinedImuFactor` with its own velocity and bias nodes. This is the classic VIO preintegration trick: **trust the dead-reckoning IMU, not the pose difference across a restart.**

### 4.10 Failure Mode: Window Restart

**When does a window restart happen?**

The window is the fixed-lag estimate. Normally it converges, marginalization keeps the memory consistent, and life is good. But three failure modes can break it:

1. **A series of bad data associations** (a featureless region, strong specular reflection, or odometry-only mode) causes the spline to drift, and the solver hits a local minimum far from ground truth.
2. **Unbounded bias growth** — the IMU bias estimate drifts so far that the bias box bounds are hit (§5.2 in the spec), signaling a broken model.
3. **A long sensor gap** — the IMU cannot bridge from `last_solved_t_` to the new sweep's start time, so there is no continuous trajectory to solve for.

When this happens, Meridian **restarts** the window:

1. Clear the current window (all control points and bias knots).
2. Declare the platform's location and velocity **unknown** (reset the live state).
3. Re-seed the window from the IMU dead-reckoning of the newest measurements.
4. Require a short static start to re-initialize gravity and bias.
5. Re-emit data (without solving) as IMU-only window restarts until the new window has real measurement data.

The keyframes emitted during restart use `constraint_kind = ImuPreintegration` to signal to the backend: "*these are connected by raw IMU, not a solved pose — be conservative.*"

### 4.11 From Solve to Backend: The Single Clean Handoff

The front-end window solve produces:

* A smooth, locally-accurate 6-DoF trajectory over ~0.8 seconds.
* A covariance that captures the geometric and IMU information inside that window.
* Sparse residuals (LiDAR points, visual patches) that anchor the trajectory to the map.
* A marginalization prior that ties the next window to this one.

The backend (L3) receives **only** the `KeyframePacket`:

```cpp
struct KeyframePacket {
  uint64_t id;                         // Unique identifier
  Timestamp stamp;                     // Real measurement time
  Pose T_ref_body;                     // Spline-evaluated pose
  enum ConstraintKind constraint_kind; // RelativeBetween or ImuPreintegration
  uint64_t rel_to_id;                  // Previous keyframe id
  Pose T_relto_this;                   // Relative transform
  GaussianBlock<6> constraint_cov;     // Relative covariance
  ObservabilityReport observability;   // Per-axis degeneracy scores
  shared_ptr<const vector<LidarPoint>> cloud_body;  // Deskewed LiDAR
  // ... plus image, exposure, calib version, etc.
};
```

The backend does **not** see:

* The spline control points (it does not know about B-splines).
* The IMU samples or biases (except on restart).
* The Ceres problem or the Hessian.
* The marginalization prior (it is the frontend's internal memory).

This clean boundary allows the frontend to be replaced — if a different estimator (iEKF, particle filter, neural net) can produce a `KeyframePacket`, it plugs in without the backend changing a line. The spec enforces this (spec 01 §7 defines `IFrontEnd`).

### 4.12 Real-Time Execution and Telemetry

The frontend runs a **hard real-time** loop at ~10 Hz:

```
For each PreprocessedGroup (a LiDAR sweep + bundled IMU/camera/GNSS):
  1. Buffer the measurements.
  2. On outer cadence (~0.1 s): [CRITICAL SECTION]
     - Extend spline (IMU-seeded).
     - Associate (LiDAR + visual).
     - Build Ceres problem.
     - Solve under 90 ms deadline (Levenberg–Marquardt).
     - Marginalize old knots.
     - Update visual map.
     - Check keyframe condition; if true, submit KeyframeJob.
  3. Return; next ingest blocks until this completes.
```

The **deadline** is enforced at the Ceres solver level: `max_solver_time_in_seconds = 0.090`. The solver completes at least `min_iterations` (2) and stops at `max_iterations` (5) or the deadline, whichever comes first. If the deadline is hit before convergence, `deadline_hit` is set to true and logged. A sustained deadline-hit rate indicates the window or knot budget exceeds the platform's compute — a signal to degrade (reduce knots, tighten the window, simplify residuals).

**Telemetry:**

The frontend publishes:
* `live_state()` — the current pose (smoothly interpolated from the last-solved window, with IMU extrapolation for times past that window).
* `diagnostics()` — a `FrontEndDiagnostics` struct containing solver iterations, deadline hits, bias estimates, observability, and per-sweep timings.

The backend and operator use this telemetry to monitor health and detect failure modes.

**Chapter summary.** The sliding-window solve is the heart of the frontend. It (1) **bounds compute** by solving for only ~12 control points at a time, enabling real-time execution; (2) **marginalizes the past** via Schur complement, keeping information without variables; (3) **re-solves the window from scratch** each sweep, so there is no filter recursion (simpler, more robust); (4) **solves tightly** — all sensors in one non-linear least-squares problem under one Hessian; and (5) **emits a single clean keyframe** to the backend, with no double-counting. The marginalization prior is the glue: old measurements are summarized as a dense quadratic constraint on the boundary knots, so each new window "remembers" what the old windows learned. The forgetting factor deflates old data over time, resisting the danger that a stale linearization runs away, and the covariance recovery is deferred to a background thread so the critical path stays fast. The next chapter zooms in on the systems reality — the real-time budget, the deskew feedback loop, observability, and how the whole thing fails and recovers.

---

## Chapter 5 — Real-Time, Deskew, Observability, and How It Fails

The previous chapters built the estimator from the inside out: the spline, the residuals, the solve. This chapter steps back to the **systems reality** — where the 100 ms actually goes, how to tell overload apart from divergence, what deskew really is in a continuous-time system, why observability is a first-class output, and how to read telemetry to diagnose and recover when the front-end fails. Throughout, we move from intuition to the measured numbers on real hardware.

### 5.1 The 100 ms budget: where the time actually goes

Every LiDAR sweep gives the front-end ~100 ms to produce a pose. The instinct of a newcomer is that the math — the Ceres solve, the matrix factorization — eats the budget. The reality is the opposite: **association dominates, not arithmetic.**

Here is the plain-language picture of one sweep cycle:

```
  [ingest + buffer]      ~1 ms    cheap: copy measurements into the window
  [extend spline]        ~1 ms    cheap: add 1 knot, seed from IMU
  [LiDAR association]   ~7-21 ms   EXPENSIVE: kNN per point + plane fit
  [visual association]  ~3-8 ms    moderate: project, warp, NCC per candidate
  [Ceres solve]         ~10-25 ms  moderate: banded, bounded by knot count
  [marginalize]          ~1 ms    cheap: Schur on the boundary block
  [visual map update]   ~2-4 ms    moderate: patch insert, medoid, eviction
  [keyframe submit]     <1 ms      cheap: snapshot + handoff to worker thread
```

The single most important fact: **finding correspondences (which point matches which plane) costs more than solving for the trajectory once you have them.** A kNN query into the ikd-Tree, repeated for every one of thousands of points, repeated across 3–4 outer re-association iterations, is where the wall-clock goes. The solve is fast because the Hessian is banded and bounded in size (Chapter 4.6); association is slow because it scales with the point count and the map density.

This is why the two biggest real-time levers are (1) **bounding the LiDAR factor count** (the stratified subsample of Chapter 3.1, default 1500) and (2) **parallelizing association** (§5.6 below). Tuning the solver iterations barely moves the needle; tuning association moves it a lot.

### 5.2 Overload vs. divergence: the critical misdiagnosis

When the trajectory output looks bad — jumpy, lagging, drifting — there are two completely different root causes, and confusing them is the most common operational mistake.

**Divergence** is an *estimation* failure: the math found the wrong answer. The associations were bad (featureless tunnel), observability collapsed (Chapter 3.7), or the solver fell into a local minimum. The pose is wrong because the *estimate* is wrong.

**Overload** is a *scheduling* failure: the math was fine, but it arrived too late. The solve took 130 ms when the budget was 100 ms, so the next sweep queued up behind it. Sweeps start dropping. The pose you see is correct but **stale** — and because it is stale, it looks exactly like drift on a dashboard.

The two demand opposite responses. Divergence needs *better constraints* (more factors, visual fusion, a window restart). Overload needs *less work* (fewer factors, a smaller window, a coarser map) — adding constraints to fix "drift" that is actually overload makes it strictly worse.

The discriminator is telemetry, not the trajectory shape. The single most diagnostic signal is **`q_meas_dropped`**: if measurements are being dropped from the ingest queue, you are overloaded, full stop. A healthy system on the right hardware drops nothing. The moment it starts dropping, latency climbs and the published pose ages, and any apparent "drift" is an artifact of staleness.

**Cold-start and restart are not overload.** When the front-end has no spline yet (startup) or has just restarted the window (Chapter 4.10), it deliberately does **not** solve. It re-emits IMU-dead-reckoned state, rebuilds the window from a constant-velocity prediction, and waits for enough real measurements to cover the knots before solving again. This looks like a stall on a naive dashboard but is the correct, designed behavior — distinguishable from overload because `q_meas_dropped` stays at zero.

### 5.3 The deskew feedback loop: resolving a circular dependency

Deskew has a chicken-and-egg problem. To deskew a sweep (place each point at its true-time pose), you need the trajectory during the sweep. But to estimate the trajectory, you want to register the sweep — which needs it deskewed. The dependency is circular.

Meridian resolves it in two regimes, mediated by a small interface contract, the `IDeskewProvider` (spec 04 §2.4, spec 00 §7.3):

**Cold-start (no spline yet).** Before the first window solve, there is no trajectory to query. L1 falls back to **IMU-only forward integration**: integrate gyro and accel from the sweep start to each point's time, producing a rough per-point pose. This is good enough to seed the first association. The deskew provider here is the IMU dead-reckoner.

**Steady-state (spline exists).** Once the window is running, the circular dependency *dissolves* — because **the spline is both the estimate and the deskew provider.** Each point's residual is evaluated at $T(t_{\text{offset}})$, its own true time (Chapter 2.1). There is no separate deskew pass and no separate deskew edge: as the solver iterates and the control points move, every point is implicitly re-deskewed because the pose it is registered at has changed. Deskew is not a step that runs before the solve; it *is* the solve, queried per-point.

This is the deep payoff of continuous-time. In a discrete filter (FAST-LIO) deskew is an explicit pre-pass that moves all points to one epoch using the latest IMU estimate, and that estimate is frozen for the registration. In Meridian, deskew refines *with* the trajectory inside the solve. The `IDeskewProvider` interface exists precisely so that L1 can request the best-available per-point pose without knowing or caring whether the answer comes from the IMU dead-reckoner (cold start) or the spline (steady state).

### 5.4 Observability and degeneracy: a first-class output

We met observability as a residual-side concept in Chapter 3.7. Here we treat it as what it really is operationally: **a first-class output of the front-end, not an afterthought.** It is the front-end's honest self-report of which directions it can and cannot trust.

Start with the concrete corridor. A robot drives down a long, straight, featureless hallway. The walls constrain lateral motion. The floor and ceiling constrain vertical motion. But **forward motion is invisible** — every LiDAR scan looks identical as you slide forward, so the geometry provides *no* constraint on how far along the corridor you are. The forward axis is degenerate. A naive system either drifts silently forward or, worse, snaps to a wrong position when a stray feature appears.

Meridian computes a **per-axis observability score** $s \in [0,1]^6$, one number for each of the six pose DoF (tx, ty, tz, rx, ry, rz). It is built from the LiDAR information matrix $\Lambda_{\text{pose}} = \sum_j w_j J_j^\top J_j$ (Chapter 3.7), eigendecomposed and mapped to $[0,1]$ via $s = \lambda/(\lambda + \kappa_{\text{deg}})$ — the Zhang / X-ICP style score. A score near 1 means "well-constrained, trust me"; near 0 means "I'm guessing on this axis."

A useful way to picture it is the **observability hexagon** (spec 09 §7.1): a radar plot with six spokes, one per axis, each filled to its score. A healthy open environment fills the hexagon nearly full. The corridor collapses one spoke (forward translation) to near zero while the others stay high — an instant visual read of *which* DoF is degenerate, not just *that* something is wrong.

```
        rz
         *           full hexagon = healthy
   rx  *   *  ry
      *  +  *        corridor: tx spoke collapses
   tz  *   *  ty          inward to ~0.1
         *
        tx  (collapsed ~0.1)
```

The scores ship in the `KeyframePacket`, and the backend uses them to inflate covariance on weak axes (Chapter 3.7) so the global graph does not over-trust an odometry-only stretch. This is the difference between a system that *knows* it is uncertain and one that confidently reports garbage.

Crucially, the **bounded factor count interacts with observability.** When the front-end caps LiDAR factors at 1500 (Chapter 3.1), it does *not* decimate uniformly — that would preferentially throw away the rare normals that constrain weak axes (the single far wall in the corridor that gives you the one lateral constraint you have). Instead it **stratifies by plane normal**, guaranteeing each normal-direction bin a floor of factors. Subsampling is thus observability-preserving by construction: it protects the weak axes precisely because those are the ones a naive subsample would starve.

### 5.5 The deskew–observability feedback

There is a subtle coupling between deskew quality and observability worth making explicit, because it explains some failures that look mysterious.

Deskew quality depends on the trajectory estimate during the sweep. Under hard motion, two things happen at once: the IMU seed gets noisier (more integration over a faster-changing rate), and the spline has to bend harder to fit. If deskew is poor, the LiDAR points land at slightly wrong poses, the plane fits get noisier, and the information matrix $\Lambda_{\text{pose}}$ — which is built from those very Jacobians — degrades. A degraded $\Lambda$ lowers observability scores.

So observability is not a static property of the *scene* alone; it is a property of the scene *and* how well the trajectory was estimated through it. A geometrically rich environment traversed under violent motion with a poor seed can report lower observability than the same environment traversed smoothly. This is correct behavior — the front-end genuinely is less certain in that moment — but it means "low observability" is sometimes a symptom of a deskew/seed problem (fixable with better IMU handling or more knots), not only of scene degeneracy.

### 5.6 Parallel association: the decisive lever

If §5.1 said association dominates the budget, this is the lever that tames it. LiDAR association — kNN + plane fit, per point — is **embarrassingly parallel**: each point's correspondence is independent of every other point's. Parallelizing it across cores is the single largest real-time win available.

On the reference workload the measured effect is roughly **21 ms serial → 7 ms parallel**, a ~14 ms saving on a 100 ms budget — the difference between comfortably making the deadline and dropping sweeps. Nothing else in the pipeline buys that much.

The catch is **determinism.** A parallel reduction that accumulates in thread-completion order gives bitwise-different results run to run, which breaks the replay==live guarantee. Meridian preserves determinism **by construction**: each thread writes its hits into a per-thread buffer, and the buffers are merged back **in fixed index order**, not completion order. The arithmetic is identical regardless of how the threads were scheduled. You get the 14 ms and keep reproducibility — you do not trade one for the other.

### 5.7 Standing real-time gates: what the operator watches

A handful of telemetry signals are the standing health gates (spec 09 §5, and `docs/REALTIME_DEBUGGING.md`). An operator watches these, not the raw trajectory, to judge health:

| Signal | Healthy | Meaning when it trips |
|---|---|---|
| `q_meas_dropped` | 0 | **Overload.** Measurements dropped from ingest — latency is climbing, pose is going stale. |
| `deadline_hit` | rare | Solve hit the 90 ms wall. Occasional is fine; sustained means the window/knot budget exceeds the platform. |
| `scan_to_odom_ms` | < 100 | End-to-end latency from scan arrival to published pose. Rising = falling behind. |
| observability scores | high | A collapsed spoke flags an odometry-only / degenerate stretch the backend must down-weight. |
| `window_restart` | 0 | A restart fired (divergence, bias blow-up, or sensor gap). Each one is a seam the backend bridges with IMU preintegration. |
| transport loss | 0 | Packets lost between layers — a plumbing problem, not an estimator problem. |

The reading discipline: **`q_meas_dropped` and `deadline_hit` first** (is it overloaded?), **then observability and `window_restart`** (is it diverging?). That order resolves the overload-vs-divergence question of §5.2 before you reach for a fix.

### 5.8 Failure modes and recovery

Three failure modes cover almost everything, each with a distinct signature and a distinct fix:

**Case 1 — Overload (too much work).** *Symptom:* `q_meas_dropped > 0`, `scan_to_odom_ms` rising, `deadline_hit` sustained; trajectory lags and looks drifty. *Diagnosis:* the per-stage timing shows association or solve over budget. *Recovery:* reduce work — lower `max_lidar_factors`, coarsen the map voxel (`voxel_surf_m` up), shrink the window, or enable parallel association if it is not already on. Do **not** add constraints.

**Case 2 — Starvation / cold-start stall (too little to solve).** *Symptom:* no pose updates for a stretch, but `q_meas_dropped == 0`. *Diagnosis:* the window has no spline yet (startup) or just restarted; it is IMU-dead-reckoning and waiting for knot coverage. *Recovery:* none needed if transient — it is designed behavior. If persistent, the system cannot get a static start to initialize gravity/bias (check for constant motion at startup) or a sensor stream is missing.

**Case 3 — Pure degeneracy (good compute, no constraint).** *Symptom:* timings healthy, `q_meas_dropped == 0`, but an observability spoke is collapsed and that axis drifts. *Diagnosis:* the scene genuinely does not constrain that DoF (corridor, tunnel, open field, glass). *Recovery:* the front-end already does the right thing — it reports low observability so the backend down-weights the axis and leans on visual/GNSS. The operational fix is upstream (route, sensing) or downstream (loop closure), not in the front-end solve.

**Per-stage tuning knobs, ranked by measured impact:**

| Knob | Trades | Impact |
|---|---|---|
| parallel association | determinism (preserved by design) for ~14 ms | largest |
| `max_lidar_factors` | accuracy on weak axes for solve+assoc time | large |
| `voxel_surf_m` (map voxel) | map density / kNN cost for resolution | large |
| window length / knot count | solve time for trajectory fidelity | moderate |
| solver `max_iterations` | convergence for time | small |

The ordering matters: reach for the top of the table first. Shaving solver iterations (bottom) almost never fixes an overload; cutting association cost (top) almost always does.

### 5.9 A worked debugging session

Put it together with a realistic session. The trajectory on the dashboard looks bad — visibly drifting after a couple of minutes.

1. **Preflight: overload or divergence?** Check `q_meas_dropped`. It is **non-zero and climbing**. This is overload, not drift. Stop looking at the trajectory shape.
2. **Real-time gates.** `scan_to_odom_ms` is at 140 ms (budget 100), `deadline_hit` is sustained. Confirmed overloaded.
3. **Per-stage timings.** Association is eating 60 ms of the sweep; the solve is a healthy 18 ms. The cost is in correspondence-finding.
4. **Diagnosis.** Association cost that high points at map density — too many neighbors per kNN query because the surface voxel is too fine, inflating the local map point count.
5. **Fix.** Raise `voxel_surf_m` (coarsen the map). Per the tuning discipline (CLAUDE.md), this is a tuning constant, so it gets a one-line row in `docs/OPTIMIZE.md` recording the value, what it trades (map resolution for kNN cost), and the measured effect.
6. **Verify.** Re-run. Association drops to ~20 ms, `scan_to_odom_ms` falls under 80, `q_meas_dropped` returns to 0. The "drift" is gone — because it was never drift, it was staleness. The estimate was correct all along; it was just late.

That last point is the lesson of the chapter: **the trajectory shape is a symptom, the telemetry is the diagnosis.** Read the gates in order — overload before divergence — and the fix follows directly.

**Chapter summary.** Real time is a scheduling problem, and in this front-end the schedule is dominated by *association*, not arithmetic — so the biggest levers are bounding the factor count and parallelizing correspondence (deterministically, by construction). The single most important diagnostic discipline is separating **overload** (work arrives late; `q_meas_dropped > 0`) from **divergence** (work is wrong; observability collapses), because they demand opposite fixes. Deskew is free in continuous time — the spline is its own deskew provider in steady state, with an IMU dead-reckoner only at cold start — and observability is a first-class per-axis output that the backend trusts to down-weight degenerate stretches. When it fails, the recovery is dictated by which of three signatures you see, and the telemetry gates, read in order, tell you which.

---

## Chapter 6 — Glossary & How to Navigate the Front-End

This final chapter is your reference desk. Part A is an alphabetical glossary of every term used in the course. Part B maps concepts to source files. Part C walks a single LiDAR sweep through the front-end end to end, naming the function at each step.

### Part A: Alphabetical Glossary

**Association** — the process of matching each LiDAR point to the nearest plane in the local map (via kNN + least-squares plane fit, §3.1) and each visual patch to a reference observation (via grid-cell occupancy + photometric gates, §3.2). Associations are **linearisation-point dependent** and are re-computed on every outer loop iteration so that as the pose estimate moves, stale correspondences are corrected before the solver diverges.

**Autodiff** — automatic differentiation via code generation. Meridian's residual functors ship in pairs: **analytic** Jacobians (hand-derived, the normative path) and **autodiff** twins (generated by Ceres AD framework, §4.5). The two are bit-compared in unit tests to guard against derivation errors; new residual types must supply both.

**B-spline** — a smooth parametric curve whose value at any time $t$ depends on exactly **four local control points** (the cubic degree-3 property). Meridian uses a **split** representation: separate SO(3) spline for rotation $R(t)$ and separate ℝ³ spline for translation $p(t)$, avoiding the coupling of a unified SE(3) spline. The trajectory $T_{W\,F_e}(t)$ is evaluated **at each measurement's own time** — IMU samples at 200 Hz, LiDAR points at their per-point offsets, camera frames at image capture, GNSS fixes at reception — without batching, giving native asynchrony (§2.2, Appendix R.1).

**Bias** — systematic errors in the IMU (gyroscope bias $b_g$, accelerometer bias $b_a$). Estimated as a sliding timeline of **bias knots** in the window, each pair piecewise-linearly interpolated to any timestamp. Biases are always free (no excitation gate), held inside box bounds (§4.10), and tied by random-walk residuals (§3.3) so they drift slowly rather than freely.

**Box bounds** — hard parameter constraints enforced by the solver. Biases are clamped: $|b_g| \le 0.5$ rad/s, $|b_a| \le 5.0$ m/s² (resolved defaults). These prevent a transient outlier burst from driving biases to absurd values before the solver diverges; the bounds are the **pre-restart stability guard** (§4.10).

**Ceres** — the nonlinear least-squares solver (Levenberg-Marquardt). Meridian's frontend uses Ceres, not GTSAM; the backend (L3) uses GTSAM's iSAM2 (§4.5, Appendix R.1). Ceres is chosen for the banded-sparse window structure: each measurement touches exactly 4 control points per spline ⇒ the Hessian is banded ⇒ sparse Cholesky/Schur marginalizes efficiently.

**Control point** — a knot of the B-spline trajectory, stored as a $(R, p) \in SO(3) \times \mathbb{R}^3$ pair. **Not an on-trajectory pose** — because the blending is cumulative, the control point at grid time $t_j$ dominates the curve roughly **one knot interval earlier**, near $t_j - \Delta t$ (§2.5). Warm-starts seed control points from IMU integration **one interval back**, not at their grid time, to land the fitted curve on the seed trajectory rather than a delayed copy.

**Continuous-time (CT)** — the trajectory is a smooth function $T(t)$ that can be queried and differentiated at any real time, not just at discrete keyframe epochs. This allows **true per-point deskew** (each LiDAR point uses $T$ evaluated at its own $t_{\text{offset}}$), **per-row rolling-shutter handling** (designed, deferred, §3.2), and **exact GNSS residuals** (evaluated at the fix's own time, not snapped to the nearest keyframe, §3.4). The continuous queryability is also what enables the colourised mesh goal: any point can be rendered at its true pose and with the true camera pose under which it was observed.

**CT spline** — see B-spline and continuous-time above. The Meridian frontend is not a filter (no predict-update cycle); it is a **trajectory estimator** that minimizes the joint likelihood of all sensor residuals over a **sliding window** of control points, subject to a marginalization prior from previous windows. This is the Coco-LIC / basalt approach (Appendix R.1).

**Degeneracy** — a direction (axis or mode) along which the geometry provides no constraint, so the estimator cannot observe motion in that direction. Classic example: a narrow corridor with only parallel walls provides no lateral constraint; an elevator with smooth walls provides no orientation constraint. Meridian detects per-axis degeneracy via the information matrix $\Lambda_{\text{pose}}$ (the LiDAR Jacobian outer-product sum) and reports a score per axis: [tx, ty, tz, rx, ry, rz] (§3.7, §5.4). Weak axes are down-weighted in the backend (L3) and may be remapped (X-ICP "solution remapping", not yet implemented).

**Deskew** — motion compensation of the LiDAR sweep, accounting for the robot's motion during the scan. In a discrete-time filter (FAST-LIO), deskew is a separate pre-processing pass that moves points backward to a single epoch. In CT, **deskew is intrinsic to the spline**: each point's residual is evaluated at $T(t_{\text{offset}})$, its own time, so "deskewing" is just querying the trajectory at the right time. As the solver iterates and the spline moves, every point is implicitly re-deskewed (§2.1, §5.3). This is why there is **no separate deskew feedback edge** at steady state — the spline is both the estimate and the deskew provider.

**DoF** — degrees of freedom. SE(3) poses have 6 DoF (3 translation, 3 rotation). Biases have 6 DoF (3 gyro, 3 accel). Gravity is 2 DoF (direction on a sphere, magnitude fixed at 9.81 m/s²), stored as a 3D vector on a Ceres manifold.

**Extrinsic** — a rigid transform from the estimation frame $F_e$ (imu_link) to a sensor frame: $T_{F_e\,L}$ (LiDAR), $T_{F_e\,C}$ (camera), or the GNSS antenna lever-arm $t_{F_e,\text{ant}}$. Extrinsics are **online-refined when motion is exciting enough** (gated by observability, §3.6). Refinement happens on the window; the L3 backend ratifies a refined extrinsic and returns a versioned `CalibrationSet` snapshot, which L2 reads (the "feedback edge" of spec 00 §7.3).

**Exposure (inverse)** — the FAST-LIVO2 photometric nuisance $\tau$ (inverse exposure time), one scalar per camera frame in the window. Used to absorb brightness/gain changes in the photometric residual (§3.2). The inverse-exposure model $r = \tau_{\text{cur}}\,I_{\text{cur}} - \tau_{\text{ref}}\,P_{\text{ref}}$ handles auto-exposure steps without absorbing them into the pose. Consecutive exposures are tied by a random-walk residual so frames with little photometric evidence inherit the neighbour's value (§3.6).

**Factor** — a residual term in the cost function, each associated with a measurement (LiDAR point, visual patch, IMU sample, GNSS fix). Ceres builds the Hessian from outer-product sums of factor Jacobians; the window cost (§3.5) is the sum of all factors' squared, robustly-weighted residuals.

**Gauge** — the unobservable symmetry of a system. The **IMU alone has a 9-DoF null space**: global left-rotation (3 DoF) + arbitrary affine position drift (6 DoF, constant offset + constant velocity). This null space is **anchored by the other residuals and the marginalization prior**, not by the IMU (§3.3): LiDAR point-to-plane ties the map, GNSS provides absolute position, and the marginalization prior locks the boundary. The first keyframe uses an `AbsolutePrior` to pin the odom origin.

**Gravity** — the local gravity vector $g_W \in \mathbb{R}^3$, part of the continuous-time IMU model (§3.3). Magnitude is fixed at 9.81 m/s²; direction is a 2-DoF state (a point on the unit sphere $S^2$) refined when the platform has enough rotational and translational excitation (§3.6). Initialized from a short static start via averaging the de-biased accelerometer.

**Gravity alignment** — the process of orienting the coordinate frame so that gravity is vertical (aligned with the negative z-axis in the world frame). For a LiDAR-inertial system, gravity alignment is essential: the accelerometer residual is $R^\top(\ddot p - g) - a_m$, so the direction of $g_W$ couples directly to the orientation estimate. A misaligned gravity vector corrupts the pose.

**GNSS** — Global Navigation Satellite System (GPS, GLONASS, Galileo, etc.). Provides absolute position but is coarse (~1–10 m) and noisy in multipath-rich environments. L2 fuses GNSS **conservatively** with covariance floors and innovation gating (§3.4); the **authoritative GNSS fusion and datum management are L3's job** (spec 05). GNSS is optional; L2 runs seamlessly without it (GNSS-denied / tactical mode).

**GNSS fix** — a single position measurement from the GNSS receiver, carrying latitude/longitude/altitude, covariance, and fix type (RTK fixed, RTK float, DGPS, SPP). The covariance is floored by fix type before use (RTK fixed: 0.05 m horizontal, 0.10 m vertical; SPP: 3 m / 6 m; §3.4).

**Gravity-removed acceleration** — in the adaptive-knot scheme (§2.7), the statistic $N_a = \max_i \| a_{m,i} - b_a \| - 9.81$ (bias-corrected, gravity magnitude subtracted). This is used to gate knot density without requiring the current orientation estimate; it correctly identifies high motion even in attitudes where acceleration is orthogonal to the local gravity vector.

**ikd-Tree** — an **incremental, dynamic k-d tree** for nearest-neighbor queries. Used to find the 5 nearest LiDAR map points to each incoming scan point, enabling fast plane-fitting (§3.1). The ikd-Tree is vendored (FAST-LIO's implementation) and wrapped behind a pimpl (`LidarLocalMap`) so the tree's PCL symbols don't leak into the header. The tree supports lazy-evaluation delete and insert (flags set at interior nodes, pushed down on search); `flushPendingDeletes()` clears all pending flags before the next association round to prevent concurrent-search races (§3.1).

**IMU** — Inertial Measurement Unit (accelerometer + gyroscope). Provides dense (200–400 Hz), relative motion measurements at low latency but with noise (white noise on acceleration/angular velocity) and bias. IMU is the **process model in a filter** (EKF, iEKF); in CT it is a **direct residual** on the spline's analytic derivatives (§3.3, Appendix R.1). The IMU provides **dense constraint** between sparse LiDAR/visual measurements; it also enables deskew at per-point granularity (§2.1).

**IMU preintegration** — the `ImuPreintegration` constraint mode in the L2→L3 handoff (§4.9, spec 01 §6.4–6.5). This is the fallback when the window has been restarted (§4.10). The front-end integrates the IMU across the new window and emits an `ImuPreintegrationSummary` instead of relying on the internal spline to tie to the previous keyframe. Not the normal path; used only after a reset.

**Jacobian** — the derivative of a residual or cost function w.r.t. the state variables. A **measurement Jacobian** is the derivative of a residual w.r.t. the pose at the measurement time; the **spline Jacobian** is the derivative of the spline pose w.r.t. the control points. The two are chained via the chain rule: $\frac{\partial r}{\partial \delta c} = \frac{\partial r}{\partial \delta T(t)} \cdot \frac{\partial T(t)}{\partial \delta c}$ (§3.1). Meridian's analytic Jacobians are **hand-derived** (not autodiff) for speed; autodiff twins are retained for parity testing (§4.5).

**Keyframe** — a snapshot pose + metadata emitted at a fixed-lag trigger (translation, rotation, elapsed time, or information-driven, §4.8). Keyframes are the **only communication between L2 (frontend) and L3 (backend)**. Each keyframe carries a `KeyframePacket` (spec 01 §6) with the pose, covariance, observability, retained point cloud, camera image, and a constraint (relative-between, absolute-prior, or IMU-preintegration mode, §4.9).

**Keyframe packet** — see spec 01 §6. The one aggregate value L2 hands L3. Contains: pose + covariance, observability, constraint type + relative transform to previous keyframe, point cloud, camera frame, biases/velocity as seeds, calib version. Never reshaped or re-parsed inside L3; the boundary contract is rigorous (spec 01 R1, R2).

**Knot** — a control point of the B-spline. The spline is stored as a deque of $(R_i, p_i) \in SO(3) \times \mathbb{R}^3$ pairs. The deque is used so that `pop_front` and `push_back` remain real-time and addresses of surviving knots stay stable (important for the Ceres parameter blocks and marginalization prior pointers, §4.3). Knot storage is managed via **virtual time** so that adaptive knot density is supported while the underlying basalt kernels see a uniform grid (§2.7).

**Knot cadence (outer cadence)** — the fixed time interval $\Delta t_{\text{knot}} \approx 0.1$ s between consecutive outer-segment starts. The number of control points placed within each outer segment varies adaptively (§2.7), but the outer segments themselves are spaced uniformly in real time.

**Knot density** — the number of control points within one outer segment ($n_{cp} \in \{1, \ldots, n_{cp,\max}\}$, default $n_{cp,\max} = 1$). Higher density gives finer trajectory representation at the cost of more parameters to optimize. Density is chosen based on IMU excitation (peak body rate and gravity-removed acceleration) so that fast motion gets more knots and smooth motion stays cheap (§2.7).

**LIVO** — LiDAR-Inertial-Visual Odometry. The integration of three sensor streams (LiDAR, IMU, camera) into one joint estimate. The GNSS extension is LIVO+G. Meridian's front-end is a CT LIVO estimator (visual stage is FAST-LIVO2 sparse-direct photometric, §3.2).

**LiDAR** — Light Detection and Ranging, a 3D laser scanner. Returns a point cloud (millions of points per scan) with 3D position, intensity, ring/row id, and time offset. LiDAR is **the primary geometric sensor** — its point-to-plane factors provide the strong translational and rotational constraints that anchor the IMU's gauge. LiDAR has high spatial resolution (~2 cm) and range (~30 m) but is ego-centric (map is built around the robot, not absolute).

**Linearization point** — the current estimate of the state (the spline control points, biases, etc.) around which the residuals and Jacobians are computed. As the solver iterates, the linearization point moves, and residuals are re-evaluated. Associations (LiDAR plane matches, visual patch matches) are **linearisation-dependent** — a different pose may match a point to a different plane — so associations are re-computed on every outer-loop iteration (§4.7).

**Manifold** — the mathematical space in which the state lives. SO(3) is a manifold (rotations cannot be added; they compose via the group law $R \cdot R'$). SE(3) is a manifold. Tangent vectors to a manifold are perturbations in the local tangent space. Ceres uses `LocalParameterization` / `Manifold` to respect the structure of non-Euclidean state spaces.

**Map (local)** — the voxel-based 3D point cloud built in the frontend's **odom frame** from the deskewed LiDAR scans. Implemented as an **ikd-Tree** with a sliding window (points farther than `cube_m` from the current pose are dropped). Used for plane-fitting (nearest-neighbor queries) and as the reference for photometric residuals (visual point map, §3.2).

**Marginalization** — the process of removing old control points and biases from the active state and summarizing their information as a prior. Done via **Schur complement** to keep the prior linear in the remaining variables (§4.3). The prior is the **only memory of past data** inside L2 and is mutually exclusive with re-introducing those measurements (preventing double-counting). Biases, gravity, and the boundary control points (the ones still in the window) are retained in the prior.

**Measurement covariance** — the uncertainty (noise power) of a sensor measurement. LiDAR covariance is range- and incidence-aware (§3.1); visual covariance is photometric-quality dependent (§3.2); IMU covariance is from the accelerometer and gyroscope Allan deviation (§3.3). Measurement covariances are inverse-weighted in the cost: high confidence → low weight (high influence). They are **not Kalman gains**; Ceres uses them to form the normal equations, not as per-measurement weighting in a sequential filter.

**NavState** — the live navigation state (pose, velocity, biases) output by `IFrontEnd::live_state()` for telemetry and control. It is **not the primary output** (keyframes are); it is a real-time snapshot for dashboards and feedback to the platform (e.g., velocity for motion planning).

**Normal-sign guard** — in the photometric warp (§3.2), the plane normal is flipped if it points away from the camera. A back-facing normal would mirror the patch and yield a spurious low residual. The guard ensures the warped patch is always front-facing before correlation.

**Observability** — the degree to which the geometry constrains each DoF of the pose. Measured via the information matrix $\Lambda_{\text{pose}} = \sum_j w_j J_j^\top \Sigma_j^{-1} J_j$ (sum of weighted LiDAR point-to-plane Jacobian outer products, §3.7). Per-axis scores are mapped from eigenvalues: degenerate directions (small eigenvalue) get low scores. Reported in the `KeyframePacket` so the backend can down-weight weak axes (spec 05 §4.3).

**Odometry frame (odom)** — the smooth, drift-prone, continuous world frame owned by the frontend. It is gravity-aligned and starts at the first keyframe. The backend (L3) owns the **map frame** and manages the `odom→map` transformation when loop closure triggers; the frontend only reads this snapshot (holds it fixed in L2, L3 owns updates).

**Outer loop** — the fixed-cadence re-association-and-solve loop in the window solve (§4.7). Each outer iteration rebuilds the LiDAR and visual correspondences against the current pose estimate, then feeds a fresh Ceres problem to the inner solver. The outer loop runs at most `max_outer_iters` times (default 4) and stops early when associations stabilize (pose shift < threshold). Deterministic mode runs exactly `max_outer_iters` to ensure reproducibility.

**Outlier** — a measurement that does not fit the model (a point matching the wrong plane, a feature match at a wrong location, a GNSS multipath hit). Detected and down-weighted via **robust kernels** (Huber, Cauchy, or GNC). A robust kernel makes the cost grow subquadratically with large residuals, capping their influence (§3.5).

**Photometric residual** — the difference between a reference patch (from a previous frame) and the warped patch in the current frame, under an affine brightness model $r = \tau_{\text{cur}}\,I - \tau_{\text{ref}}\,P_{\text{warp}}$ (§3.2). Sparse-direct methods compute residuals at a sparse set of pixels (e.g., high-gradient corners) rather than all pixels, making the method fast and robust to large regions with uniform intensity.

**Plane fitting** — least-squares fit of a plane $(n, d)$ (unit normal + offset) to the 5 nearest LiDAR neighbors. Used to define the point-to-plane residual (distance from point to plane). Rejected if any neighbor lies too far from the plane (planarity gate) or if fewer than 5 neighbors are found within a distance threshold (§3.1).

**Point-to-plane** — the residual for a LiDAR point: $r = n^\top p + d$ (signed distance from the point to the fitted plane). This is **direct registration** (no feature extraction), and is the geometric basis of FAST-LIO. Per-point true-time registration (via spline evaluation at $t_{\text{offset}}$) ensures the residual sees the point at the pose it was captured under (§3.1).

**Preintegration (IMU)** — integration of the IMU trajectory (pose, velocity, bias, gravity) over an interval, on an assumed constant-bias assumption, to produce a *relative* motion summary. Used in the IMU-preintegration constraint mode (the fallback when the window is restarted, §4.9, spec 01 §6.5). **Not the normal path**; internal spline provides the measurement of relative motion in steady state.

**Prior** — a term in the cost function that encodes a belief about the state, independent of new measurements. In the sliding-window frontend, the **marginalization prior** (§4.3) is the compressed information from old control points, summarizing all past data in the form of a dense quadratic function on the boundary knots. It is the **only memory** carried between window slides and is exclusive with re-introducing those measurements (preventing double counting).

**Robust kernel (robust weighting)** — a function $\rho(x)$ that grows sublinearly with squared error, capping the influence of outliers. Examples: Huber (quadratic near zero, linear in the tails), Cauchy (always sublinear), Geman-McClure / GNC (gradually tightens as iterations proceed). Applied to LiDAR point-to-plane, visual photometric, and GNSS residuals to prevent a single bad correspondence from dominating the solve (§3.5).

**Rolling shutter** — a camera exposure mode where image rows are captured sequentially over the readout interval. The top row is captured first; the bottom row captures last, potentially at a different robot pose. Handling rolling shutter exactly requires evaluating each row's patch at its own row-time pose $T(t_{\text{row}})$ rather than the frame-average time (§3.2, designed but deferred pending deployment camera verification).

**Schur complement** — a matrix operation that eliminates intermediate variables from a linear system, leaving a reduced system over the remaining variables. Used for marginalization: when control points leave the window, their pointers are eliminated via Schur complement of the Hessian, leaving a dense prior on the boundary knots. The Schur complement is **exact** (not an approximation) and preserves the information loss from marginalization, ensuring proper information flow (§4.3).

**SE(3)** — the special Euclidean group, the set of rigid transformations (rotation + translation) in 3D. Written $SE(3) = SO(3) \ltimes \mathbb{R}^3$ (a rotation composed with a translation). A pose $T_{A\,B} \in SE(3)$ maps points from frame B to frame A. Not stored as a 4×4 matrix in Meridian; instead, as a quaternion + translation vector (more compact, avoids gimbal lock).

**Sliding window** — the active set of control points spanned by the current window (typically ~0.8 s deployed, i.e., ~8 knots at 10 Hz cadence). Older control points are marginalized (not held fixed); the window size is fixed, so per-step cost is **bounded**. As new measurements arrive, the window extends and old knots are marginalized. This is a **fixed-lag** estimator, not a fixed-memory filter; the lag is the window duration (spec 00 §6).

**SO(3)** — the special orthogonal group, the set of rotation matrices in 3D. $SO(3)$ is a 3-DoF manifold (every orientation can be uniquely parameterized by an angle-axis vector $\phi \in \mathbb{R}^3$, with magnitude = angle and direction = axis). Exponential map: $\mathrm{Exp}: \mathfrak{so}(3) \to SO(3)$, $\mathrm{Exp}(\phi) = I + \frac{\sin\|\phi\|}{\|\phi\|}\,[\phi]_\times + \frac{1-\cos\|\phi\|}{\|\phi\|^2}\,[\phi]_\times^2$ (Rodrigues' formula).

**Sparse direct** — a visual odometry method that computes photometric residuals at sparse pixel locations (high-gradient corners, edges) rather than every pixel, yielding speed and robustness to textureless regions. FAST-LIVO2's approach (Appendix R.3); the alternative is "dense direct" (all pixels, slower but uses all available information).

**Spline** — see B-spline and CT spline above. In Meridian, the spline is the **trajectory representation** and is queried for pose, velocity, and angular velocity at any time via closed-form expressions (the basis polynomial derivatives, §2.9). This is the core advantage over piecewise-constant or discrete-pose models: no interpolation error, analytic derivatives, and true per-point deskew.

**Spline Jacobian** — the derivative of the spline pose w.r.t. the control points: $\frac{\partial T(t)}{\partial \delta c_i}$. Computed via the chain rule through the B-spline basis (analytic; the basalt kernels provide these). Chained with measurement Jacobians to form the full residual Jacobian w.r.t. the state (§3.1, Appendix R.1).

**Strapdown** — continuous-time integration of IMU (accelerometer + gyroscope) to propagate pose, velocity, and biases. Solved via ODE $\dot R = R[\omega_m - b_g]_\times$, $\dot p = v$, $\dot v = R(a_m - b_a) + g$ (§3.3). In a filter, strapdown is the **process model**. In CT, it is the **residual equation** the spline derivatives must satisfy, turned into the IMU factor (§3.3).

**Tail anchors** — weak residuals pinned to the tail end of the spline (past $t_{\text{end}}$, the last measurement time) to tie the unmeasured span's velocity and rate to the IMU seed's constant extrapolation (§2.8). They prevent null-space float in the unsupported knots without blocking recovery from a seed error.

**Tail knots** — the trailing knots in the spline (indices ≥ first unsupported knot) whose basis weight over all measured times is zero or near-zero. These knots lie past $t_{\text{end}}$ and have no measurement support; they are **pinned constant** at their IMU warm-start value each sweep and **re-seeded** from the fresh IMU integration every new sweep (§2.8).

**Tangent space** — the vector space of infinitesimal perturbations to a manifold element. For SO(3), the tangent space is $\mathfrak{so}(3) \cong \mathbb{R}^3$ (angle-axis vectors). For SE(3), the tangent space is $\mathbb{R}^6$ (translation first, rotation second, in Meridian: [ρ; φ]). Tangent vectors are used for optimization: the solver computes updates in the tangent space and applies them via the box-plus operation.

**Timestamp** — absolute time in nanoseconds since epoch (`int64_t`), the ROS 2 convention. Per-point offsets within a LiDAR sweep are relative offsets in nanoseconds (`int32_t`, ±2.1 s relative to `stamp_start`, §2.1).

**Voxel** — a cubic cell in 3D space. Voxel downsampling (decimation by occupancy) reduces point-cloud density: each cell stores only the first point seen. Used to limit the local map growth and to balance LiDAR and visual constraints (dense clouds would bias the solve toward geometry far from the camera).

**Voxel hash** — a spatial data structure mapping voxel grid indices to lists of points or features inside that voxel. Fast spatial lookup and insertion (O(1) average). Used for the visual map (separate from the LiDAR ikd-Tree) to track active map points and evict out-of-box points deterministically (§3.2).

**Warp (affine warp)** — the transformation of a reference patch into the current camera view under a homography model using the **LiDAR plane normal as the depth proxy** (§3.2). Exact under planar geometry; invalid if the plane is degenerate (near-singular affine matrix, extreme scale, near-singular condition number). A bad warp is detected (determinant check, condition-number check, §3.2) and the point is dropped for that frame rather than warped against a bad homography.

### Part B: How to Read the Code — Concept-to-File Mapping

The Meridian frontend lives in `/src/meridian_frontend/src/ct/` and is organized as follows:

| Concept | File | Key Classes / Functions | Notes |
|---------|------|-------------------------|-------|
| **Core estimator** | `ct_frontend.hpp/.cpp` | `CtFrontEnd` | Main entry point; implements `IFrontEnd`. Drives the per-sweep window solve (solveWindow, slideWindow) and keyframe emission. |
| **B-spline trajectory** | `spline_window.hpp/.cpp` | `SplineWindow` | Stores control points; supports initialization, extension, re-seeding, sliding. Owns the real→virtual time map for adaptive knots. Delegates pose/velocity/acceleration evaluation to basalt-headers. |
| **Spline analytic Jacobians** | `spline_analytic.hpp` | Templated polynomial basis; derivatives; thin wrapper over basalt. | Hand-coded derivatives (not autodiff). Chained with measurement Jacobians. |
| **Window optimization** | `window_problem.hpp/.cpp` | `WindowProblemInputs`, `LidarLocalMap`, `PlaneFit`, `LidarHit` | Manages the Ceres problem per sweep: LiDAR association, visual association, factor building, solver setup. The ikd-Tree map is wrapped here. |
| **LiDAR residuals** | `residuals_lidar.hpp/.cpp` | `LidarPointToPlaneFactorAnalytic`, `LidarPointToPlaneFactorAutodiff` | Point-to-plane residual + Jacobian. Association (plane fitting) logic. Robust weighting and bounded-factor subsampling. |
| **IMU residuals** | `residuals_imu.hpp/.cpp` | `ImuGyroFactorAnalytic`, `ImuAccelFactorAnalytic`, `BiasRandomWalkFactor`, `VelocityAnchorResidual`, `RateAnchorResidual` | Gyro/accel residuals on spline derivatives. Bias random-walk ties. Tail anchors. |
| **Visual residuals** | `residuals_visual.hpp/.cpp` | `PhotometricResidualAnalytic`, `VisualPoint`, `Feature`, `ExposureChain` | Photometric patch residual, warp logic, reference-patch lifecycle. Per-frame inverse exposure state. |
| **Visual map** | `visual_map.hpp/.cpp` | `VisualMap`, `VisualPoint` | Active visual point map (separate from LiDAR ikd-Tree). Reference-patch storage, observation cap, medoid selection, box eviction. |
| **GNSS residuals** | `residuals_gnss.hpp/.cpp` | `GnssAbsolutePositionFactor` | Absolute position residual. Fix covariance floors and innovation gating. |
| **Marginalization** | `marginalization.hpp/.cpp` | Schur-complement elimination of old control points. | Builds the dense prior; ensures use-after-free safety via knot-index bounds on front-trim. |
| **Pose marginal** | `pose_marginal.hpp/.cpp` | `poseMarginalFromProblem`, `poseMarginal` (fallback) | Extracts 6×6 covariance of the keyframe pose from the solved problem (Schur back-subst) or from LiDAR hits only (fallback). |
| **Keyframe finalization** | `keyframe_finalizer.hpp/.cpp` | `KeyframePacket` assembly, async covariance worker. | Async worker thread that emits keyframe packets. Two-path covariance: inline (deterministic) or async (live). |
| **Image pyramid** | `image_pyramid_view.hpp/.cpp` | 3-level image pyramid for coarse-to-fine patch search. | FAST-LIVO2 scheme. |
| **Bias knots** | (in `ct_frontend.hpp`) | `BiasKnots` | Sliding timeline of (gyro bias, accel bias) pairs. Piecewise-linear interpolation. Stored in deque for stable addresses. |
| **Observability** | (in `ct_frontend.hpp`, `residuals_lidar.cpp`) | `computeObservability` | Information matrix from LiDAR Jacobians. Per-axis scores and eigenvector modes. |

**Key entry points to understand the flow:**

1. **Ingest and window solve** — `CtFrontEnd::ingest()` is called per-sweep. It calls `solveWindow()`, which is the inner guts: spline extension, association, Ceres problem build, fixed-cadence re-association loop, and window-slide marginalization.

2. **Residual building** — Inside `solveWindow()`, look for the loops that iterate over LiDAR hits, visual patches, and IMU samples, calling `problem.AddResidualBlock()` for each. Each residual class (e.g., `LidarPointToPlaneFactorAnalytic`) is a `ceres::CostFunction` subclass.

3. **Spline control** — `SplineWindow` is the trajectory store. Its `pose()`, `angularVelocityBody()`, `linearVelocityWorld()` methods are called to evaluate the trajectory at measurement times. The spline is extended (`extendTo`) at the start of each sweep, seeded by IMU integration.

4. **Keyframe packing** — When a keyframe is due (§4.8 triggers), `maybeEmitKeyframe()` fills a `KeyframePacket` and hands it to the `KeyframePacket` finalizer, which computes the covariance (either inline on deterministic path or async on the live path).

### Part C: Follow One LiDAR Sweep Through the Front-End

To understand the entire front-end, trace a single LiDAR sweep from raw data to keyframe emission. Here's the narrative:

#### 1. **Ingest** (L2 boundary, L1→L2)

A LiDAR scan arrives: `LidarScan{stamp_start, sweep_duration, points[]}` where each point carries `(xyz_sensor, t_offset_ns, ring, intensity, range)`. The points are **not deskewed** — they are raw in the sensor frame at their per-point time offsets.

**Where it enters L2:** `CtFrontEnd::ingest(PreprocessedGroup group)`. The group bundles the LiDAR scan with the latest IMU samples and camera frame (if present).

#### 2. **Spline seeding** (IMU integration)

Before solving, the spline must be extended to cover the new sweep's time span. The frontend calls `buildSeed()`, which integrates the group's IMU forward from the last-solved pose using the current bias and gravity estimates, with **constant-rate and constant-accel intra-sample**. This produces a seed trajectory $T_{\text{seed}}(t)$.

**Code:** `ct_frontend.cpp::buildSeed()`. The seed is passed as a `std::function<Pose(Timestamp)>` closure to `SplineWindow::extendTo()`.

**Why:** The spline is a trajectory representation; it must be initialized somewhere. IMU integration provides a good **warm start** (a pose near the true solution) so the solver doesn't have to search far. **Critically**, the warm start seeds each new control point from the seed trajectory **one local knot interval back** ($C_j \leftarrow T_{\text{seed}}(t_j - \Delta t)$), not at $t_j$ itself, because the control point dominates the curve one interval earlier (§2.5, a subtle detail).

#### 3. **Spline extension** (adding knots)

The spline is extended to cover $[\text{stamp\_start}, \text{stamp\_start} + \text{sweep\_duration}]$ via `SplineWindow::extendTo(t_end, seed, n_cp)`. The number of control points `n_cp` is chosen adaptively based on IMU excitation (peak body rate and gravity-removed acceleration, §2.7), but deployed it is typically 1 (uniform spline).

**Code:** `spline_window.cpp::extendTo()`. New knots are placed at uniform real-time intervals and seeded from the IMU seed.

#### 4. **Tail-knot pinning and re-seeding** (unmeasured span governance)

Knots whose basis support over $[\text{stamp\_start}, t_{\text{end}}]$ is zero (or near-zero) are **pinned constant** at their IMU warm-start value (§2.8). Their stale values from the previous sweep are **overwritten** (`reseedFrom`) with the fresh IMU prediction, ensuring the tail is never poisoned by old extrapolation.

**Code:** `spline_window.cpp::reseedFrom()`, called in `solveWindow()` after `extendTo()`.

#### 5. **LiDAR point-to-plane association** (plane fitting)

For each point in the scan, the frontend:
1. Queries its time: $t_j = \text{stamp\_start} + t\_\text{offset}\_\text{ns}$.
2. Evaluates the current spline at $t_j$ to get the pose $T_{W\,F_e}(t_j)$.
3. Transforms the point from sensor to world: $p_j^W = T_{W\,F_e}(t_j) \cdot [T_{F_e\,L} \cdot p_j^L]$.
4. Queries the ikd-Tree for the 5 nearest neighbors.
5. Fits a plane $(n, d)$ via QR least-squares.
6. Rejects if planarity is poor or neighbors are too far (§3.1 gates).
7. Accepts and stores the hit in a `LidarHit` structure.

**Code:** `window_problem.cpp::LidarLocalMap::fitPlane()`. Association is parallelized over points (OpenMP).

**Why this time?** Each point is evaluated at its own $t_j$. As the solver moves the control points, every point is implicitly **re-deskewed** because $T(t_j)$ changes. There is **no separate deskew pass**.

#### 6. **Bounded-factor subsampling** (observability-aware decimation)

If the inlier point count exceeds `max_lidar_factors` (default 1500), the frontend subsamples by **plane-normal stratification** (§3.1), not uniform decimation. This preserves the rare normal directions that constrain weak axes, protecting observability.

**Code:** `window_problem.cpp` or `residuals_lidar.cpp`. Points are binned by their fitted normal into strata (six axis-aligned hemispheres + oblique); each stratum is allocated a fair share of the budget, with a floor to guarantee sparse normals aren't starved.

#### 7. **Visual association (photometric)** (camera frame processing)

If a camera frame is present and visual fusion is enabled:
1. Build an image pyramid (3 levels, §3.2).
2. For each active visual map point (from previous sweeps), project it into the current camera using the current spline pose and camera extrinsic $T_{F_e\,C}$.
3. Search for the best-matching reference patch in the point's observation list using NCC (normalized cross-correlation) and photometric SSD (sum-of-squared-difference).
4. Compute the affine warp using the LiDAR plane normal (if the point has a fitted plane) or reject the point.
5. Gate: grid occupancy (at most one point per cell), depth continuity (no occlusion jumps), warp validity, NCC threshold, SSD threshold, robust kernel.
6. Build the photometric residual: $r = \tau_{\text{cur}} I(\mathbf u) - \tau_{\text{ref}} P_{\text{warp}}$.

**Code:** `visual_map.cpp` (association) + `residuals_visual.cpp` (residual). Association happens at fixed cadence (every outer loop iteration, §4.7).

#### 8. **Ceres problem assembly** (building the least-squares cost)

The frontend constructs a `ceres::Problem` containing:
- **LiDAR point-to-plane residuals** (one per hit): $r_j = n^\top p_j^W + d$.
- **IMU residuals** (one per sample): gyro and accel, evaluated at the spline derivatives.
- **Bias random-walk residuals** (ties across bias knots).
- **Visual photometric residuals** (one per active patch match).
- **GNSS absolute-position residuals** (if GNSS is enabled and a fix is in-gate).
- **Tail anchors** (weak velocity/rate residuals on the unsupported knots).
- **Marginalization prior** (from the previous window, §4.3).
- **Gravity** (2-DoF manifold).
- **Exposure inverse** (one per camera frame, with random-walk ties).

**Code:** `window_problem.cpp::solveWindow()`, calling Ceres `AddResidualBlock()` for each residual type.

#### 9. **Fixed-cadence solve** (re-association loop)

The solve proceeds in **nested loops** (§4.7):

```
for outer_iter in 0..max_outer_iters:
  (a) Re-associate LiDAR and visual (rebuild residuals)
  (b) Run Ceres LM solve for reassoc_steps inner iterations
  (c) If pose shift since last assoc < threshold: break (early stop)
```

Each outer loop re-associates because the poses have moved. Each inner solve refines the poses under the fixed correspondences. This avoids churning the correspondence set (non-determinism) and avoids baking in stale matches (divergence).

**Code:** `solveWindow()` contains the outer loop; Ceres is invoked with `solver.Solve()`.

#### 10. **Observability extraction** (per-axis geometric constraint)

After the solve converges, compute the information matrix $\Lambda_{\text{pose}}$ from the LiDAR point-to-plane Jacobians:
$$\Lambda = \sum_j w_j J_j^\top \Sigma_j^{-1} J_j$$

Eigendecompose to extract per-axis scores (§3.7). Store in the `ObservabilityReport` for the keyframe.

**Code:** `ct_frontend.cpp::computeObservability()`.

#### 11. **Window slide and marginalization** (removing old data)

Once the spline has grown past the window length (`window_knots` × `knot_dt`), the oldest control point(s) and associated bias knots that leave the window are marginalized via **Schur complement** (§4.3). The marginalization computes a dense prior on the **boundary** control points (the ones still in the window) summarizing all past measurements. This prior is the **only memory** L2 carries between sweeps.

**Code:** `ct_frontend.cpp::slideWindow()`, calling `marginalization.cpp` for the Schur elimination.

#### 12. **Keyframe decision** (emission policy)

Check if a keyframe should be emitted based on the triggers: translation > `d_kf`, rotation > `theta_kf`, elapsed time > `dt_kf`, or information-driven criteria (§4.8). Never emit while the window is uninitialised (first ~1 s).

**Code:** `ct_frontend.cpp::maybeEmitKeyframe()`.

#### 13. **Keyframe packet assembly** (L2→L3 handoff)

If a keyframe is due:
1. **Pose** — Evaluate the spline at the keyframe timestamp: $T_{\text{odom}\,F_e}(\text{stamp})$.
2. **Covariance** — Recover the 6×6 pose marginal from the solved window (joint LiDAR + IMU information, §3.7; fallback to LiDAR-only if rank-deficient).
3. **Relative constraint** — Compute the relative transform to the previous keyframe: $\hat T_{\text{prev}}^{-1} \hat T_{\text{cur}}$. The relative covariance is the **sum of two consecutive pose marginals** (conservative upper bound, drops the cross-covariance).
4. **Observability** — Attach the `ObservabilityReport` (per-axis scores, §3.7).
5. **Retained cloud** — Deskew the full scan to the keyframe pose: for each point, evaluate $T_{W\,F_e}(t_{\text{offset}})$ at its true time and express it in the keyframe's estimation frame $F_{e,\text{stamp}}$. This is a **true per-point deskew**, not a motion-compensation pass.
6. **Camera frame** — Attach the current camera frame + the extrinsic snapshot used.
7. **Constraint kind** — Default: `RelativeBetween` (relative pose + covariance). Special cases: `AbsolutePrior` (first keyframe) or `ImuPreintegration` (after a window restart).

**Code:** `keyframe_finalizer.cpp` (async covariance worker) and `ct_frontend.cpp` (packet assembly).

#### 14. **Async finalization** (optional on the live path)

On the live (non-deterministic) path, the covariance is computed async by a worker thread so the main thread doesn't stall on matrix inversion. The problem snapshot is captured at solve time and the worker reconstructs the Hessian on a cloned spline window, then inverts the information block. The deterministic path computes covariance inline (synchronous).

**Code:** `keyframe_finalizer.cpp::KeyframeJob`.

#### 15. **Visual map update** (reference-patch maintenance)

After the solve, update the visual map (§3.2):
- Promote new LiDAR points to visual points (add to the map).
- Add new observations to existing visual points (add-gate, §3.2).
- Recompute medoid references (most central observation).
- Box-delete visual points that fell outside the active box.

**Code:** `visual_map.cpp`.

#### 16. **Emission** (sending the keyframe to L3)

Submit the `KeyframePacket` to the backend via the `KeyframeSink` callback. **This is the only interface between L2 and L3.**

**Code:** `ct_frontend.cpp::ingest()`, calling `keyframe_sink_(packet)`.

### Synthesis

The frontend is a **continuous-time, trajectory-based sliding-window estimator**:

1. **Trajectory**: Stored as a split SO(3)×ℝ³ cubic B-spline, extended and re-associated per sweep.
2. **Residuals**: Four types — LiDAR point-to-plane, IMU derivatives, visual photometric, GNSS absolute position — form a joint nonlinear least-squares cost.
3. **Measurement asynchrony**: Native — each point is evaluated at its true time, no synchronization.
4. **Deskew**: Intrinsic to the spline; each point's residual queries the trajectory at its own time.
5. **Observability**: Measured and reported per-axis via the information matrix, protecting against geometric degeneracy.
6. **Marginalization**: Schur-complement, maintaining a dense prior on boundary knots as the window slides.
7. **Keyframes**: Emitted at fixed-lag triggers with absolute pose, relative constraint, covariance, observability, retained point cloud, and camera frame.

The frontend is **production-ready**, not a research prototype. It has deterministic and live modes, deadline-bounded solving, and comprehensive observability/degeneracy mitigation. Its **contract with the backend** is the `KeyframePacket` — clean, versioned, and sufficient for L3 to build a globally consistent map and trajectory without re-implementing the measurement models. For that downstream story — the iSAM2 factor graph, loop closure, and global consistency — see the companion `docs/BACKEND_COURSE.md`.
