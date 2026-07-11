# Meridian Backend — A Course

This course is for the project owner: someone who knows the Meridian codebase but is new to the internals of SLAM back-ends. It builds intuition from first principles — why a back-end exists, what a factor graph is, the geometry-and-uncertainty math that underpins everything, how Meridian's iSAM2 graph is actually wired, how loop closure stays robust, and finally a glossary plus an end-to-end walkthrough. Read it front to back the first time; afterward, Chapter 6 is your reference desk. Each chapter is self-contained enough to revisit on its own, and the worked examples are meant to be re-read whenever a concept stops feeling concrete.

## Table of Contents

- [Chapter 1 — Why a SLAM Backend Exists](#chapter-1--why-a-slam-backend-exists)
- [Chapter 2 — Factor Graphs from First Principles](#chapter-2--factor-graphs-from-first-principles)
- [Chapter 3 — The Geometry & Uncertainty Toolkit](#chapter-3--the-geometry--uncertainty-toolkit)
- [Chapter 4 — Inside Meridian's Backend Graph: The iSAM2 Factor Graph](#chapter-4--inside-meridians-backend-graph-the-isam2-factor-graph)
- [Chapter 5 — Loop Closure and Robustness](#chapter-5--loop-closure-and-robustness)
- [Chapter 6 — Glossary & How to Navigate](#chapter-6--glossary--how-to-navigate)

---

## Chapter 1 — Why a SLAM Backend Exists

Welcome to the **backend**. The frontend (that's L2 in Meridian-speak) is doing something hard: it's watching a robot move through space and trying to estimate *where it is* by looking at what it sees — LiDAR scans, camera images, IMU vibrations, and GPS signals. But it can only see a sliding window of the recent past (think of it as a short-term memory). The backend's job is to connect that local, short-term estimate to a **globally consistent** long-term view of the world.

Let's build intuition before the math.

### 1.1 What is SLAM and why is it hard?

**SLAM** stands for **Simultaneous Localization And Mapping**. It means: as you move through a space, build a map of that space *and* figure out where you are in it, **at the same time**. This is hard because the two jobs are circular:

- To build a good map, you need to know where the robot is when each measurement arrives.
- To know where the robot is, you need a good map to match against.

So neither is "done first" — they're estimated together.

#### The sensor story: three threads of evidence

Meridian fuses four streams of measurement:

1. **LiDAR**: a 3D laser scanner that returns millions of points, showing the 3D shape of the world around you. It's direct and accurate over short ranges (~10–30 m).

2. **Inertial Measurement Unit (IMU)**: an accelerometer and gyroscope that feel the robot's motion. An IMU can tell you *how fast you're rotating and accelerating*, but not *where you are* — it has no absolute reference.

3. **Camera**: a visual sensor that can recognize textures and edges. Meridian uses sparse photometric matching (correlating pixel brightness between frames) to register against the environment.

4. **GNSS** (e.g. GPS): an absolute positioning signal from satellites. On a good day, it can tell you your position on Earth. On a bad day (urban canyon, multipath), it's corrupted by meters of error.

Each stream is **drift-prone** on its own:

```
LiDAR at 10 Hz  ──┐
IMU at 200 Hz    │
Camera at 20 Hz  ├──→ [Frontend window L2]  → smooth local odometry ("odom" frame)
GNSS at 1 Hz     │    (continuous trajectory)
                 │
          (drift accumulates over time)
```

### 1.2 The frontend produces drifting odometry

The **frontend** (L2 in Meridian) does something remarkable: inside a sliding window, it tightly fuses all four sensors into one continuous-time B-spline trajectory. Think of the spline as a smooth curve through space-time, evaluated at each point's true sample instant — this is why Meridian is "continuous-time."

At each new LiDAR scan, the frontend:

1. Deskews the scan (corrects for motion during the sweep)
2. Registers the points against a local, voxel-based map
3. Fuses IMU / camera / GNSS measurements
4. Solves for the best spline that fits all the data

The result is a very **smooth, locally accurate** estimate of the trajectory. But here's the problem: **the frontend's map is local** — it only remembers a small region around the current pose. Once you move far away, that local map becomes stale. So when you come back to a place you visited before, the frontend has *no idea* it's a place you've seen — it treats it as brand new.

That means distances accumulate:

```
Start → move 100 m north → lose visual features → 
move 100 m east (IMU drifts ~1% per second) → 
return to start location after 200 m total motion.

Frontend estimate: "You're at [100.5 m, 98.2 m]"  
Reality:           "You're back at [0, 0]"
Error:             ~140 m! (drift of 1–3% over 200 m is typical for odometry)
```

This drift happens because:
- **LiDAR matching** drifts when you traverse repetitive or featureless zones (long corridors, flat plains, snow).
- **IMU integration** is unbiased on average, but has a white-noise process noise that accumulates — a random walk. Over seconds it stays tight; over minutes to hours it explodes.
- **Camera features** disappear in dark or low-texture areas.
- **GNSS** is coarse and noisy, often unusable near buildings.

The frontend's local estimate is **drift-prone**. It's not wrong — it's just local.

### 1.3 Enter the backend: global consistency

The **backend** (L3) solves a different problem: *given a history of short-term local estimates, find the globally consistent, drift-free trajectory*.

Here's the idea: imagine you have a string of estimates, one from each scan:

```
Scan 1 (t=0 s)   : "I'm at [0, 0]"     — solid (just started)
Scan 2 (t=1 s)   : "I'm at [1.2, 0.1]" — drifted slightly from Scan 1
Scan 3 (t=2 s)   : "I'm at [2.4, 0.3]" — drifted more
  ...
Scan 50 (t=50 s) : "I'm at [50.5, 4.2]" — drifted 4.2 m perpendicular!
Scan 51 (t=51 s) : "I'm at [50.8, 4.5]" — ... and suddenly I recognize
                     this place — I saw it at Scan 1!
```

At Scan 51, you could **loop closure**: a constraint that says "the pose at Scan 51 should be *almost identical* to the pose at Scan 1 (same location, different time)."

Now you have a conflict: your odometry says "I drifted 4.5 m sideways," but a loop closure says "I'm back where I started." The backend's job is to resolve this **optimally**, typically by:

1. Trusting the loop closure (it's a direct geometric observation)
2. Adjusting your entire trajectory to be **globally consistent**

One way to think about this: imagine you have a chain of springs connecting consecutive poses, each spring representing "this is how far I moved according to odometry." Now imagine looping a rubber band from pose 1 to pose 51 that says "these should be the same pose." When you release the system, the springs and rubber band will settle into a new configuration that keeps odometry *roughly* right (the springs pull) while satisfying the loop closure *exactly* (the rubber band's tension). The backend is that spring-and-rubber-band system, solved exactly.

### 1.4 Keyframes and the interface

The frontend *doesn't* send the backend every single measurement. Instead, it groups measurements into **keyframes** — snapshots taken at moments when "enough new information has accrued."

A keyframe might be triggered when you've moved more than 1 m or rotated more than 10°, or it's been 1 second since the last keyframe. The frontend bundles this into a **`KeyframePacket`**, a single message containing:

```
KeyframePacket:
  id: 42
  stamp: 1234567890 ns     (time of this keyframe)
  T_rel_prev: [relative pose to keyframe 41]
  cov: [uncertainty of that relative pose]
  observability: [per-axis quality]
  cloud_body: [the LiDAR points at this moment]
  image: [RGB image for colourization]
  ...
```

This packet is the **contract** between frontend and backend: it says "here's what I observed locally, and here's how uncertain it is."

### 1.5 Two frames: odom and map

This is crucial conceptually. Meridian works in two frames:

**Odom frame** ("`Frame::Odom`"):
- Owned by the frontend (L2)
- Smooth and continuous
- Drifts over time
- Every frontend output is in odom

**Map frame** ("`Frame::Map`"):
- Owned by the backend (L3)
- Global and drift-free
- Anchor point for loop closures and GNSS

The relationship between them is `T_map_odom`, a 6-DoF transform. When you first boot up, `T_map_odom ≈ identity` (they're close). But as the backend detects loop closures, it refines `T_map_odom` to correct the frontend's drift.

In code-speak:
```
p_map = T_map_odom * p_odom    (transform from odom to map)
```

The frontend lives in odom and publishes smooth odometry — it never jumps. The backend lives in map and publishes global consistency. The pipeline translates between them.

### 1.6 The factor graph: encoding belief as springs

The backend uses a **factor graph** — a graph where:

- **Nodes** are poses (one per keyframe): `X(1), X(2), ..., X(n)`
- **Edges** are constraints (one per piece of information)

Visually:

```
X(1) ──between──→ X(2) ──between──→ X(3) ──between──→ X(4)
  ↑                                                      ↓
  └────────────── loop closure ─────────────────────────┘
           ("I'm back where I started")

        +   GNSS anchor at X(2)
        +   Gyro bias drift model
        +   ...all adding constraints
```

Each edge represents a **belief** or **observation**:

- **"Between" edges** (odometry): "X(i+1) should be about 1.2 m northeast of X(i), ± some uncertainty (covariance)"
- **Loop closure edges**: "X(1) and X(4) should be very close together (same physical location)"
- **GNSS edges**: "X(2)'s absolute position is lat/lon/alt, ± some GPS uncertainty"

To solve the graph means: find poses `X(1) ... X(n)` that:
1. Minimize the sum of all constraint violations (the "spring energy")
2. Keep odometry mostly right
3. Satisfy loop closures
4. Match GNSS where available
5. Stay consistent with gravity and inertial physics

This is a **nonlinear least-squares problem**: 

$$\text{minimize} \quad \sum_i \| h_i(X) - z_i \|_{\Sigma_i}^2$$

where `h_i(X)` is the prediction of measurement `z_i` given poses `X`, and `$\Sigma_i$` is the uncertainty (covariance).

**Intuitively**: springs with stiffness determined by covariance. A tight covariance (low uncertainty) = a stiff spring. A loose covariance = a floppy spring. The solver finds the configuration with minimum potential energy.

### 1.7 How the problem arrives at the backend

Let's trace a keyframe through the system:

```
[LiDAR scan at t=5.0 s]
        ↓
[L0 sensor abstraction: raw timestamp + sync]
        ↓
[L1 preprocessing: deskew, filter, pyramid]
        ↓
[L2 frontend: sliding-window continuous-time LIVO+GNSS]
    - Optimize B-spline trajectory inside a window
    - Estimate T_odom_body, v, biases, gravity
    - Evaluate at t=5.0 s to get the keyframe pose
        ↓
[Frontend decides: "This is a keyframe!" (moved >1 m or rotated >10°)]
        ↓
[Pack a KeyframePacket]
    - Pose: T_odom_body at t=5.0 s
    - Relative to previous: T_prev_this (with covariance)
    - Per-axis observability (degeneracy)
    - The deskewed cloud (shared immutable)
    - The image for colourization
        ↓
[L3 backend receives KeyframePacket]
    - Lift pose from odom → map frame
    - Create a graph node X(id=42)
    - Add an odometry "between" edge from X(41) to X(42)
    - Store the keyframe info (stamp, cloud, image)
```

Now, later, if L5 (loop detector) notices that keyframe 42 looks very similar to keyframe 5 (both at the same place), it sends a **loop constraint** to L3:

```
[L5 loop detector: "Keyframe 42 matches Keyframe 5!"]
    - Run GICP to refine the relative transform
    - Compute loop covariance
        ↓
[L3 receives LoopConstraint(from=5, to=42, T_5_42)]
    - Add a loop edge to the graph
    - Run incremental optimization (iSAM2)
        ↓
[iSAM2 finds: poses have drifted. Adjust them all to satisfy loop.]
        ↓
[L3 broadcasts GraphUpdate: "Keyframes 1–42 moved as follows:"]
        ↓
[L4 map layer re-integrates changed keyframes with corrected poses]
[L2 frontend picks up refined calibration and continues smoothly]
```

### 1.8 Observability: handling degeneracy

Odometry is not equally good in all directions. Imagine driving down a long hallway:

```
  ↑ Z (up) — unobservable (can't see ceiling/floor well)
  │
  ├─→ X (forward) — well observable (hallway recedes)
  │
  └─→ Y (sideways) — POORLY observable (white walls, no features)
```

Your motion forward is clear (you see the hallway recede). Your rotation around Z is clear (you see walls). But your drift sideways is invisible — could you have drifted 10 cm left? The walls don't tell you.

The frontend computes **observability scores** (per-axis, 0 to 1) for each keyframe. The backend receives these and **inflates the covariance** on poorly-observed axes:

```
Original covariance:    [small, small, small, ...]
Sideways (Y) is bad (score=0.1):  inflate by ~100×
Result:                 [small, LARGE, small, ...]
```

Now the optimizer **trusts the sideways direction less** and won't try to correct it as hard. A loop closure later can still correct it (loop closures are direct geometric observations), but weak odometry won't.

### 1.9 Two types of odometry: normal and restart

The frontend normally sends **relative pose** constraints ("I moved this far from the last keyframe, with this covariance"). This is the common case, and it encodes all the fusion of LiDAR+IMU+camera.

But sometimes the frontend **loses tracking** — it can't estimate a good relative pose (observability collapsed, matches degraded, etc.). In that case, it falls back to a window restart: throw away the old window, re-initialize from IMU-only integration, and start fresh. On restart, it sends a different constraint type: **IMU preintegration** (ΔR, Δv, Δp, and bias Jacobians) — the raw cumulative effect of IMU measurements.

The backend treats these two mutually exclusively: **normal path** → one `BetweenFactor`; **restart path** → one `CombinedImuFactor`. Never both for the same edge. This is MUST-FIX #1 in Meridian's design docs: **exactly one constraint per edge, no double-counting**.

### 1.10 Layers L0–L6 and where L3 and L5 sit

Meridian is organized in **7 layers**:

```
L0  Sensor abstraction + time sync
    ↓
L1  Preprocessing (filter, downsample, pyramid)
    ↓
L2  Frontend (continuous-time LIVO+GNSS) — sliding window, odom frame
    ↓   (KeyframePacket)
L3  Backend (iSAM2 factor graph) — global, map frame  ← YOU ARE HERE
    ↓   (GraphUpdate + LoopConstraint from L5)
L4  nvblox map (GPU TSDF + colour mesh)
    ↓
L5  Loop closure / place recognition  ← feeds back to L3
    ↓   (LoopConstraint)
L6  Operator surface (colour mesh + confidence)
```

**L3** (the backend, this chapter's focus):
- Owns the global `map` frame
- Consumes `KeyframePacket` from L2
- Consumes `LoopConstraint` from L5 and `GnssFix` from L0
- Emits `GraphUpdate` (which keyframes moved) → L4, L2
- Emits refined calibration snapshots → L2

**L5** (place recognition):
- Recognizes when two keyframes are at the same location
- Verifies the match with GICP (ICP with scale & Gaussian kernels)
- Sends verified loop constraints back to L3
- Runs on the backend thread, so it can afford to be slow

The loop from L5 → L3 is what lets the system correct its drift.

### 1.11 Why factor graphs over filters

The frontend is a **filter** (Kalman-like): it estimates the state `[pose, velocity, bias]` and maintains a covariance of the *most recent* state. It's fast and works well locally.

The backend is a **factor graph** / **optimizer** (iSAM2, Levenberg-Marquardt). It re-estimates *all keyframes in the graph*, not just the latest. This is slower, but it:

1. **Finds globally consistent solutions** — loop closures can "reshape" the entire trajectory
2. **Handles non-Gaussian outliers** with robust kernels (Huber, GNC) — loop closures can be wrong
3. **Re-optimizes incrementally** — iSAM2 is SOTA for real-time backend optimization
4. **Decouples online extrinsic refinement** — camera-IMU / LiDAR-IMU transforms are graph variables, refined globally

### 1.12 A worked example: closing a loop

Let's say Meridian explores a building:

**Phase 1: Exploration (t = 0 to 100 s)**
- Starts at the entrance (X0 = identity)
- Walks down a hallway (5 keyframes, 10 m forward, some sideways drift from odometry)
- Turns a corner
- Explores a room (10 more keyframes)
- Walks back toward the entrance

```
  Entrance   Room
    (X0)─────(X8)
     │        │
    X1      X9
     │        │
    X2      X10
     │        │
    ...........─ Loop closure detected! X15 ≈ X2
```

**Phase 2: Loop closure detected (t = 100 s)**
- L5 recognizes: "Keyframe 15 (current pose) looks like Keyframe 2 (a room corner I saw before)"
- GICP refines the alignment: `T(X2) → T(X15)` is not identity, it's small but non-zero
- L5 sends `LoopConstraint(from=2, to=15, T_2_15, covariance)` to L3

**Phase 3: Backend optimizes (milliseconds)**
- L3 adds a loop edge to the graph
- iSAM2 re-linearizes and updates
- Result: the entire path from X0 to X15 is adjusted to satisfy the loop closure
- The hallway odometry (X0 → X2), which had some sideways drift, is corrected
- All 15 poses move slightly to make the graph internally consistent

```
Before loop closure:
  Entrance──────→ Room
    (slight sideways drift in hallway)

After loop closure:
  Entrance══════→ Room
    (hallway straightened, drift removed, path is now reversible)
```

**Phase 4: Broadcast and re-map**
- L3 emits `GraphUpdate` saying "X0 moved by +0.05 m, X1 by +0.08 m, ..., X15 by -0.03 m"
- L4 (nvblox map) receives this, **de-integrates** the affected voxels (undoes the old pose), **re-integrates** with corrected poses
- The 3D mesh is now globally consistent
- L2 (frontend) picks up the refined trajectory and rebases its odometry: it keeps publishing smooth odom-frame motion, but the underlying `T_map_odom` transform is corrected

### 1.13 Conclusion: the backend's role

The **frontend** asks: *"How am I moving right now?"* (local, fast, drifty)

The **backend** asks: *"Where have I actually been, globally and consistently?"* (global, slower, drift-free)

They work together:
- The frontend feeds raw local estimates (keyframes)
- Loop closure and GNSS add global constraints
- The backend solves for the best global trajectory that respects odometry while satisfying global observations
- The result feeds back to the frontend as a pose correction and refined calibration

This is why you need a backend for real SLAM. Odometry alone will always drift. A backend with loop closure is the feedback mechanism that brings the estimate back to reality.

In the next chapters, we'll see how the backend *builds* and *solves* the factor graph (what variables and factors go into it), how it handles loops and GNSS, and how it refines extrinsics online.

---

## Chapter 2 — Factor Graphs from First Principles

This chapter builds up the factor-graph formalism step by step, grounding every concept in Meridian's actual design, so you can understand not just *what* the back-end does, but *why*.

### 2.1 Variables, Measurements, and the Intuition

Imagine a robot rolling through a corridor, collecting sensor readings. At certain moments (keyframes), we want to lock down "where was the robot?" — we call these **unknown robot poses** our **variables**. Let's call the first pose `X₁` and the second pose `X₂`.

We also have **measurements**: the IMU and LiDAR tell us something about how the robot moved between these two moments. The front-end (L2) fuses these sensors and gives us a relative motion estimate — "from `X₁` to `X₂`, the robot moved forward by about 2 meters and rotated 5 degrees." But measurements are never perfect; they come with **uncertainty** (a covariance matrix).

**The core insight:** instead of running a filter that maintains *only* the latest state, we want to ask *all the measurements at once* what the best trajectory was, given their uncertainties. If one measurement disagrees with the others, we want to resolve that disagreement fairly.

This is an **inverse problem**: given noisy measurements, infer the true poses. It turns out this problem has a beautiful mathematical structure when measurements are Gaussian-distributed (which they approximately are, by the Central Limit Theorem, especially after front-end filtering). That structure is the **factor graph**.

### 2.2 The Spring Analogy: Energy and Least Squares

Think of the unknown poses as beads on a wire, and each measurement as a spring connecting two beads. A spring wants to pull the beads into a certain relative configuration, and the stiffer the spring (the more confident the measurement), the harder it pulls.

If you push the beads around and release them, they settle into a configuration where the total potential energy (summed spring stretch) is minimized. **This is exactly what a maximum-likelihood (MAP) estimator does with Gaussian noise.** The "spring stiffness" is the inverse of the measurement covariance — tighter covariance means stiffer spring.

**Mathematically:** if we have a measurement `z` (what we observed) of a true quantity `h(X)` (a function of the poses), and the measurement error has covariance `Σ`, then the **negative log-likelihood** (energy contribution) is:

```
error = ||h(X) - z||²_Σ
```

where `||·||_Σ` is the Mahalanobis norm (weighted by inverse covariance). The total graph cost is the sum of all these error terms over all measurements. **Minimizing the total error is the same as finding the MAP estimate.**

### 2.3 A Two-Pose, One-Loop Toy Example

Let's make this concrete. Suppose we have two keyframes:

```
Keyframe 1: time=0,   pose X₁ = [rot=0°, trans=(0,0)]
Keyframe 2: time=1s,  pose X₂ = [rot=?, trans=(?,?)]
```

**Measurement 1 (Odometry between poses):** The front-end says "I fused my LiDAR and IMU, and computed the relative motion from X₁ to X₂." It outputs:

```
z_odom = {relative_rotation = 5°, relative_translation = (2m, 0m), cov = 0.01 m² I₆}
```

This goes into a `BetweenFactor` (spec 05 §4) that constrains the relative pose:

```
error_odom = ||T_{1→2} - z_odom||² / (0.01)
```

**Measurement 2 (Loop closure):** After the robot moves around and comes back, a place-recognition system recognizes that keyframe 10 (far in the future) is actually looking at the same scene as keyframe 1. It measures a relative transform:

```
z_loop = {relative_rotation = -5°, relative_translation = (-2m, 0m), cov = 0.1 m² I₆}
```

But wait — between keyframe 2 and keyframe 10, there are 8 intermediate poses. So the loop constraint connects X₂ and X₁₀ through the entire chain, and the loop "closes" when the chain of relative motions is consistent.

**The optimization problem:** minimize

```
error_total = error_odom + error_loop + (penalty for gauge freedom)
```

over all unknown poses `X₂, X₃, ..., X₁₀`. The optimization moves the poses to satisfy both constraints *simultaneously* — odometry wants to drive the chain forward, and the loop pulls it back into consistency.

### 2.4 Variables: the Unknowns the Back-End Estimates

In Meridian, the back-end (L3) estimates:

1. **`X(i)` = keyframe poses** — one per keyframe. `X(i)` is a `Pose3` in GTSAM lingo (rotation + translation in the `map` frame). This is the **primary variable**. (Spec 05 §2.1)
2. **`V(i)`, `B(i)` = velocity and IMU bias** — only on restart edges (rare events). These are transient, introduced only when the front-end window collapses and must bridge using raw IMU preintegration. They are marginalized away at the next normal keyframe. (Spec 05 §5, §11)
3. **`G` = GNSS-origin transform** — the transformation from the SLAM `map` frame to the ENU (East-North-Up) geodetic frame, estimated once GNSS measurements arrive and the robot moves enough to make yaw observable. (Spec 05 §6)
4. **`E(s)` = extrinsic calibration** — the camera↔body or GNSS antenna↔body transforms, if online refinement is enabled. Off by default (spec 05 §10).

**Why not keep velocity and bias on every keyframe?** The front-end (L2) already estimated these inside its sliding window and folded their information into the relative covariance matrix `constraint_cov` (spec 01 §6.4, spec 05 §2.2). Carrying `V` and `B` on every keyframe *and* using the relative covariance would **double-count the evidence** — a fatal mistake that collapses marginal covariances and silently breaks loop closure. The transient-inertial design (spec 05 §2.2, Appendix R.2) prevents this.

### 2.5 Factors: the Constraints

A **factor** is a measurement or prior that connects one or more variables. Each factor contributes an error term to the total cost. Here are the six types Meridian uses (spec 05 §3):

| Factor | Connects | Source | Interpretation |
|--------|----------|--------|-----------------|
| **Gauge anchor** | `X(first)` | bootstrap | Fixes the absolute position/orientation of the first pose so the graph is not free-floating in space. Uses **damping** (not a hard prior) so corrections can shift the whole trajectory. (Spec 05 §3) |
| **Odometry (normal)** | `X(i), X(i+1)` | front-end `KeyframePacket` | "The front-end measured this relative motion with this covariance." One per keyframe interval. (Spec 05 §4) |
| **Odometry (restart)** | `X(i), V(i), B(i), X(j), V(j), B(j)` | front-end IMU preintegration summary | Used when the front-end window collapses and must bridge via raw IMU. Includes bias random-walk. (Spec 05 §5) |
| **Loop closure** | `X(i), X(j)` (far apart) | place-recognition (L5) | "We detected that keyframe `i` and keyframe `j` see the same place, with this relative transform." Switched on/off (not always admitted) and robustified. (Spec 05 §7, §8) |
| **GNSS position** | `X(i), X(j), G` | satellite receiver | "At the given time, the antenna was at this ENU position." Interpolated to the fix timestamp between two bracketing keyframes. Switched and robustified. (Spec 05 §6) |
| **GNSS-origin prior** | `G` | first valid GNSS fix | Weak prior seeding the origin estimate. (Spec 05 §6.1) |
| **Extrinsic prior** | `E(s)` | offline calibration | Tight prior pinning the camera/GNSS lever arm. (Spec 05 §10) |

**Why exactly one odometry factor per interval?** This is *the* load-bearing rule (spec 05 §3.1). The front-end shipped you the marginal covariance of the relative motion — not the conditional, not the information matrix, just the final uncertainty of "pose `j` relative to pose `i`" after fusing all sensors in the window. Using that marginal *once*, as a `BetweenFactor`, is correct. Adding an absolute prior *and* an IMU preintegration factor on the same edge would re-inject the same evidence three times, overconfidently. The single-factor-per-edge rule is enforced by the `KeyframePacket.constraint_kind` enum (spec 01 §6.4), which is never both `RelativeBetween` and `ImuPreintegration` in a single packet.

### 2.6 The Nonlinear Least-Squares (NLS) Problem

Putting it together, the back-end solves:

```
X* = argmin_X Σ_i || h_i(X) - z_i ||²_Σ_i
```

where:
- `X` is the vector of all unknowns (all poses, and the origin/extrinsics if present).
- `h_i(X)` is the function that predicts what measurement `i` should be, given the state `X`.
- `z_i` is what we actually observed.
- `Σ_i` is the covariance (uncertainty) of measurement `i`; `||·||_Σ` is the Mahalanobis norm.

For an odometry factor connecting `X(i)` and `X(j)`:

```
h(X) = inverse(X(i)) ∘ X(j)    [the relative pose if both keyframes are at the true position]
z = T_relto_j                  [what the front-end measured]
error = || h(X) - z ||²_Σ
```

This is a **nonlinear** problem because `h()` involves group operations (matrix inverses, exponentials/logs on the rotation group `SO(3)`). We cannot solve it in closed form; we use **iterative optimization**.

### 2.7 Nonlinear Optimization: Gauss-Newton and Dogleg

iSAM2 uses a **trust-region method** called **Dogleg** (a variant of the Gauss-Newton algorithm, spec 05 §9.1). Here's the intuition:

1. **Linearize** around a current guess `X_k`. Expand each error term `h_i(X)` as a Taylor series:

```
h_i(X) ≈ h_i(X_k) + J_i(X_k) · (X - X_k)
```

where `J_i` is the Jacobian (derivative) of `h_i` with respect to `X`.

2. **Solve the linear system** to find the step `δX` that minimizes the quadratic approximation:

```
H · δX = -g
```

where `H = Σ_i J_i^T Σ_i^{-1} J_i` is the **Hessian** and `g = Σ_i J_i^T Σ_i^{-1} (h_i(X_k) - z_i)` is the gradient.

3. **Update** `X_{k+1} = X_k + δX`. If the error decreased, take the step and repeat. If it increased, the linear approximation was bad — trust it less, take a smaller step (trust-region radius).

**Why Dogleg?** The linear system `H · δX = -g` is most reliable when the linearization point is already good. Visual and GNSS factors can sit far from their true solution at init; a trust region (Dogleg) bounds the step conservatively so we don't overshoot. Gauss-Newton alone might diverge. Meridian's choice is standard in modern SLAM (LIO-SAM also uses Dogleg, spec 05 Appendix R.1).

### 2.8 The Covariance and the Information Form

After solving, the optimization's output is not just the poses `X*` but also the **uncertainty** of each estimate — the covariance matrix `Σ` of the posterior. For a variable `X(i)`, `Σ(i,i)` tells you how much you trust the estimate: a small covariance means "I'm confident"; a large one means "I'm uncertain."

**The information form** inverts the covariance: `Ω = Σ^{-1}` is the **information matrix**. It is a way of writing the problem that is numerically efficient — instead of storing dense covariances, we can store the sparse information (the Hessian `H` above) and recover just the marginals we need. This is one of the tricks that makes iSAM2 efficient.

### 2.9 Marginalization: Summarizing and Dropping Old Variables

A multi-hour SLAM session can accumulate thousands of keyframes. Keeping all of them in memory and reoptimizing from scratch is expensive. **Marginalization** is the trick to keep the graph bounded.

**Intuition:** Suppose we have an old restart-bridge's velocity and bias (`V`, `B`) that are no longer used. We can remove them from the graph by computing the **Schur complement** — a dense Gaussian factor on the remaining variables that encodes the information the removed variables carried.

**Mathematically:** if the Hessian of the linearized problem is block-structured:

```
H = | H_M,M   H_M,S |     [M = marginalized, S = separator/retained]
    | H_S,M   H_S,S |
```

Then the Schur complement on `S` is:

```
H_S^+ = H_S,S - H_S,M · H_M,M^{-1} · H_M,S
```

This becomes a new dense factor on `S`, added to the graph. The variables in `M` are then deleted. **The result is exact at the moment of marginalization** (the factor encodes all the information those deleted variables carried), **but frozen there** — if the separator variables move far from the linearization point later, the factor cannot be re-linearized, so the approximation error grows (spec 05 §11.2).

Meridian uses a conservative strategy: marginalize **only** transient velocity/bias from restart bridges (which are rare), **never** poses. Poses are cheap (one `Pose3` per ~1 meter of travel, so even a 10 km mission is only ~10k keyframes, well within modern computers), and keeping them means loops and future corrections can still attach to old parts of the trajectory (spec 05 §11).

### 2.10 Gaussian Factors and Weighted Least Squares

The bridge between covariance and the cost function is **noise models**. When you tell GTSAM "this measurement has covariance `Σ`," you are saying:

```
P(z | h(X)) ∝ exp(-½ || h(X) - z ||²_Σ)
```

This is a **Gaussian likelihood**. Maximizing it (MAP estimation) is equivalent to minimizing the squared error. The **information** (inverse covariance) acts as a weight: measurements with small covariance (high information) get weighted more heavily in the optimization.

### 2.11 iSAM2: Incremental Smoothing and Mapping

Finally, the **engine** — **iSAM2** — is an **incremental** algorithm for solving pose-graph problems. Instead of re-solving the entire system from scratch every time a new keyframe arrives, iSAM2:

1. **Maintains a Bayes tree** — a sparse factorization of the Hessian of the graph, organized as a tree.
2. **When a new factor is added:** find the part of the tree it touches (the variables it connects to), re-eliminate only those cliques on the path to the root, and reuse the rest.
3. **Fluid relinearization:** if a variable's estimate moved past a threshold `relinearizeThreshold`, mark it for re-elimination on the next update; otherwise reuse its linear factor from last time.
4. **Recover only what you need:** to get the latest pose for publishing, do a cheap partial back-substitution in the tree instead of recovering the entire state.

**Why is this fast?** In a typical SLAM problem, a new keyframe's odometry factor connects only to the previous keyframe. The two variables and their neighbors form a small region of the graph. Re-eliminating that region (maybe 10–50 variables) is much cheaper than eliminating all 10,000 keyframes. And because the graph grows roughly linearly, the per-update cost stays *nearly constant* — the iSAM2 win.

**Why a Bayes tree?** Gaussian elimination on a matrix is not inherently incremental — once you factor it, a new variable on the right side requires refactoring. The **Bayes tree** is a graphical representation of the Cholesky factor that *does* support incremental updates. Relinearization in the tree corresponds to updating only the affected cliques. It's the data structure that makes iSAM2 work (Kaess et al., 2012, cited in spec 05 Appendix R.7).

### 2.12 Loop Closures and Robust Kernels

When a loop closes (the robot returns to a previously visited place), a `LoopConstraint` connects two distant keyframes `X(i)` and `X(j)`. This is powerful — it corrects drift — but risky: a wrong loop can weld two unrelated places and ruin the map.

Meridian defends with three layers (spec 05 §7–8):

1. **PCM (Pairwise Consistent Measurement) pre-filter:** Before a loop enters the graph, check: "Does this loop agree with all the odometry constraints between `i` and `j`?" Build a consistency graph of pending loops, find the maximum clique of mutually consistent loops, and admit only that clique.

2. **Committed robust kernel:** Wrap the loop factor in a **Huber** M-estimator (convex, safe in the incremental solve). If the loop's residual is large (outlier), the Huber curve soft-downweights it. Huber is convex, so it plays nicely with Dogleg's trust region.

3. **Batch GNC consolidation (optional):** Periodically, run a heavier **Graduated Non-Convexity (GNC)** optimization off-thread on just the loop sub-graph, re-judging each loop with redescending kernels. Feed the verdict back to the live graph: remove loops that GNC rejects as outliers.

**The intuition:** PCM is combinatorial (pick a mutually consistent set), the Huber kernel is continuous (smooth the residuals), and batch GNC is the heavyweight (offline statistical verification). Together they make loop closure robust without destabilizing the incremental solve.

### 2.13 Observability and Degeneracy Inflation

Not all directions are equally observable. If the robot moves in a straight corridor with featureless walls on both sides, rotation around the vertical axis is unobservable — the LiDAR cannot see it. The front-end computes an `ObservabilityReport` with per-axis scores `s_k ∈ [0,1]` (1 = fully observable) in `[tx, ty, tz, rx, ry, rz]` order.

L3 uses this signal (spec 05 §4.3): if a direction is poorly observable (`s_k ≈ 0`), **inflate the covariance along that axis** so the optimizer does not over-trust it:

```
λ_k = 1 + (ρ_max - 1) · (1 - s_k)^γ
```

with `ρ_max = 1e4` (default). A fully degenerate axis gets inflated by `1e4`; a fully observable one stays at 1. This is the **degeneracy contract** of spec 05 §4.3: never silently drop a factor, only inflate its uncertainty to match reality.

### 2.14 GNSS and Datum Alignment

GNSS provides absolute position in the WGS84 geodetic datum (lat/lon/alt), but it is weak (meter-level), multipath-prone, and **switched** (you cannot assume every fix is correct). GNSS also arrives at unpredictable times, not at keyframe moments.

Meridian's GNSS handler (spec 05 §6):

1. **Seeds an origin variable `G = T_map_enu`** — the transformation from the SLAM `map` frame to ENU. This is estimated, not assumed; residual misalignment between the SLAM trajectory and the geodetic datum is therefore observable and correctable.

2. **Buffers fixes until datum lock:** Before admitting any GNSS factor, collect a few fixes until the robot moves enough (baseline > 5 m) and along non-collinear directions (yaw becomes observable). Then fit the origin using **Umeyama alignment** — a least-squares point-cloud registration — and check that the fitted yaw has acceptable uncertainty (< 5°). Only then lock the datum and admit fixes as factors.

3. **Interpolates each fix to the fix timestamp** (not the nearest keyframe), connecting two bracketing poses and the origin variable. This avoids injecting the keyframe interval's residual motion into the GNSS residual.

4. **Robustifies with Huber**, decimates by distance (not time, so multipath jitter does not create redundant factors), and gates: skip a fix if it is less certain than the back-end's own marginal estimate (spec 05 §6.4).

### 2.15 Extrinsics: Calibration as Variables

The **extrinsic** transforms (LiDAR↔body, camera↔body, GNSS antenna↔body) are *usually* calibrated offline and held fixed. But in a field rig, mounts flex with temperature or impact, and the offline calibration can drift.

If `extrinsic_refine = true` (per-platform, default off), L3 holds `E(s) = T_body_sensor(s)` as graph variables, starting from the offline calibration prior. Observability safeguards (spec 05 §10):

- **Excitation gate:** `E(s)` stays pinned until the robot accumulates enough rotation *and* translation (not just linear motion). Extrinsics are weakly observable without full 6-DoF excitation.
- **Convergence freeze:** Once `E(s)`'s marginal covariance falls below a threshold, re-tighten its prior and stop relinearizing (save cost).
- **Sanity clamp:** If `E(s)` strays too far from the offline prior, reject the update and warn.

### 2.16 Putting It Together: The Update Cycle

Every ~100 ms (default, spec 05 §9.2), the back-end:

1. **Drains the input queue:** accumulates all keyframes, loops, and GNSS fixes arrived since the last update into one batch.
2. **Stages mutations:** inserts the batch into the pending graph and values (does not solve yet).
3. **Calls `ISAM2::update`:** solves the factor graph once, using fluid relinearization. Runs `extra_iters_loop` extra passes if a loop was admitted, to help the solution converge despite the large correction.
4. **Extracts outputs:**
   - **`GraphUpdate`** → L4 (which keyframes moved, rebuild which TSDF regions).
   - **`corrected_trajectory()`** → TF tree / evaluation (the `map`-frame poses).
   - **`refined_calibration()`** → L2 (if extrinsics changed).
   - Publishes telemetry: chi-square, relinearization count, timing, loop accept/reject verdicts.

**Why batch and not one-per-keyframe?** A burst of keyframes (e.g., the robot accelerating) creates one batch, so iSAM2 folds them in one elimination pass. Batching amortizes relinearization cost and lets the solver amortize its work. If a keyframe arrives every 100 ms and `optimize_interval_ms = 100`, you might batch 1–2 keyframes per update in steady state; during a loop closure (rare event), you might batch a loop, a GNSS fix, and a keyframe in one shot, and iSAM2 handles it gracefully (spec 05 §9.2).

### 2.17 Determinism and Replay

For evaluation (spec 10), the **replay harness** runs the back-end off a fixed sequence of measurement files, with **no wall-clock timers** and **no threads**. The cadence is deterministic: optimize after every keyframe, folding in all loops/GNSS staged since the previous keyframe (spec 05 §17.1).

Given a fixed packet stream the back-end solve is order-independent (GTSAM TBB-off, COLAMD a pure function of the graph).

---

**You now understand the core of Meridian's back-end.** The factor graph is a probabilistic inference engine: it asks "what poses and calibrations best explain all the measurements, given their uncertainties?" iSAM2 solves it incrementally, staying fast even as the graph grows. Robustness comes from layered outlier rejection (PCM + robust kernels), and observability inflation prevents overconfidence in degenerate directions. The single-factor-per-edge contract keeps measurements from being double-counted, and the Bayes tree keeps computation incremental. The rest is engineering: threading, config, telemetry, and the careful hand-off between the latency-sensitive front-end and the globally consistent back-end.

---

## Chapter 3 — The Geometry & Uncertainty Toolkit

### Why we need a special toolkit

Before a SLAM system can *optimize* anything, it must first have a language to talk about it. How do we represent a robot's position and orientation? How do we add two small rotations without gimbal lock? What does it mean for a covariance matrix to be "good" or "bad"? And why do the mathematics of rotation differ fundamentally from the mathematics of translation?

This chapter builds that language from the ground up, grounded in how Meridian actually implements it. You will learn not just the concepts but the **actual ordering conventions** used in the code, because a single mistake there—mixing translation-first with rotation-first—silently corrupts estimates. Let's start with the simplest object: a pose.

### 3.1 The Pose: Position + Orientation as a Rigid Transform

A **pose** is the answer to one question: *where and how is the robot right now?* It combines:
- **Position**: a 3D point (three numbers: *x, y, z* in meters)
- **Orientation**: a 3D rotation (how the robot is facing)

We write it mathematically as a **rigid transform** $T$, which maps a point in the robot's own local coordinate system ("body" frame) into a global coordinate system ("map" frame). The equation is:

$$p_{\text{map}} = T \, p_{\text{body}} = R \cdot p_{\text{body}} + t$$

where $R$ is a 3×3 rotation matrix and $t$ is a 3-vector translation.

In code, Meridian stores a pose as two simple pieces:

```cpp
struct Pose {
  Eigen::Quaterniond q;      // rotation: a unit quaternion (normalized 4-tuple)
  Eigen::Vector3d    t;      // translation: a 3D vector in meters
};
```

**Why a quaternion, not a matrix or Euler angles?**

- A **3×3 rotation matrix** drifts: repeated multiplications introduce rounding errors that push it off the set of true rotations. A quaternion is one normalization away from perfect.
- **Euler angles** (roll, pitch, yaw) suffer from *gimbal lock*: at certain angles (e.g. pitch = 90°), two of the three axes become parallel, and the system loses a degree of freedom. A quaternion avoids this entirely.
- A **quaternion** is a 4-tuple $q = [w, x, y, z]$ with the constraint $w^2 + x^2 + y^2 + z^2 = 1$. It is the minimal, singularity-free way to store a rotation.

The quaternion $q$ encodes the rotation $R$ via the standard formula; the code has a `.toRotationMatrix()` method. In practice, you almost never work with that matrix directly—the manifold math (explained below) handles perturbations in a cleaner space.

**Example: a robot moves 1 meter forward and turns 45 degrees.**

```
Pose p1 = { q: identity,     t: [0, 0, 0] }       // start at origin, facing forward
Pose p2 = { q: from(45°),    t: [1, 0, 0] }       // move 1m, turn 45°
```

The quaternion `from(45°)` is actually a 4-tuple that, when applied, rotates by that angle around the axis you specify (here, the *z*-axis, the "yaw" in a robot's frame).

### 3.2 Composing Poses: The Chain Rule for Transforms

Poses form a **group**. If robot A is at pose $T_{\text{map} \leftarrow \text{A}}$ relative to the map, and robot B is at pose $T_{\text{A} \leftarrow \text{B}}$ relative to A, then robot B's pose relative to the map is:

$$T_{\text{map} \leftarrow \text{B}} = T_{\text{map} \leftarrow \text{A}} \cdot T_{\text{A} \leftarrow \text{B}}$$

The $\cdot$ is group composition (matrix multiplication under the hood). Order matters: this is not commutative.

```cpp
Pose T_map_robot_old = ...;
Pose T_robot_old_robot_new = ...;   // delta (relative motion over one time step)
Pose T_map_robot_new = T_map_robot_old * T_robot_old_robot_new;  // composition
```

This is how keyframes chain together. Each `BetweenFactor` in the graph (Chapter 4 on the backend) says: "the relative motion from keyframe *i* to keyframe *i+1* is this pose; compose it with X(i) to get X(i+1)."

### 3.3 Small Perturbations and the Tangent Space: Why Rotations Aren't Vectors

Here is the first deep insight: **you cannot add two small rotations the way you add two vectors.**

Imagine a robot's yaw angle is 90°. You want to perturb it by 5°. Naively, you might think: "add 5 to 90, get 95." This works. But now imagine the yaw is in a quaternion, and you want to add a small rotation vector $[\text{roll}, \text{pitch}, \text{yaw}] = [0, 0, 0.087]$ (5 degrees in radians). You *cannot* just add 0.087 to the yaw component of the quaternion and renormalize; the result is not a valid rotation.

**The fix: use the exponential map.**

The key insight is that *near the identity*, we can approximate a small rotation $\phi = [\phi_x, \phi_y, \phi_z]$ (a 3-vector) as:

$$R_{\text{approx}} = \begin{bmatrix} 1 & -\phi_z & \phi_y \\ \phi_z & 1 & -\phi_x \\ -\phi_y & \phi_x & 1 \end{bmatrix}$$

This is the **skew-symmetric matrix**, often written $[\phi]_\times$. And as the perturbation gets smaller, this approximation becomes exact in the limit. Formally, the map from the 3D vector $\phi$ to the rotation group is the **exponential map**:

$$R = \exp([\phi]_\times), \quad \text{or briefly} \quad R = \mathrm{Exp}(\phi).$$

And there is an inverse, the **logarithm**:

$$\phi = \mathrm{Log}(R),$$

which takes a rotation matrix back to a 3-vector.

**Intuition via analogy:** Imagine a flat map of the Earth. You can add two 2D vectors (displacements on the map) directly. But the Earth is a sphere. For small regions, the flat approximation works—this is what a **tangent space** is. For rotations, the tangent space is $\mathbb{R}^3$ (the 3-vector $\phi$), and the manifold is $SO(3)$ (the group of all rotations).

To add two rotations perturbed from a base $R_0$ by small vectors $\phi_1$ and $\phi_2$:

1. Convert to rotation matrices: $R_1 = R_0 \cdot \mathrm{Exp}(\phi_1)$, $R_2 = R_1 \cdot \mathrm{Exp}(\phi_2)$.
2. Do NOT add $\phi_1 + \phi_2$ and then exponentiate—that loses information.
3. Instead, compose in the group: $R_{\text{final}} = R_0 \cdot \mathrm{Exp}(\phi_1) \cdot \mathrm{Exp}(\phi_2)$.

**For a full pose (position + rotation)**, we now have two pieces:
- **Translation perturbation** $\rho$ (a 3-vector, m)
- **Rotation perturbation** $\phi$ (a 3-vector, rad)

Together, the **6-DoF tangent vector is** $\xi = [\rho; \phi]$. This is the fundamental representation for *small changes* to a pose. In Meridian, this is written **translation-first**, then rotation:

$$\xi = \begin{bmatrix} \rho_x \\ \rho_y \\ \rho_z \\ \phi_x \\ \phi_y \\ \phi_z \end{bmatrix} \in \mathbb{R}^6.$$

And the operation to apply a perturbation to a pose is:

$$T' = T \,\boxplus\, \xi = T \cdot \mathrm{Exp}(\xi),$$

called **box-plus**. The inverse is **box-minus**:

$$\xi = T_1 \ominus T_2 = \mathrm{Log}(T_2^{-1} \cdot T_1).$$

### 3.4 The Tangent Ordering Pitfall (Translation-First vs Rotation-First)

**This is a load-bearing bug detector. Read this carefully.**

Meridian uses **translation-first** ordering: $[\rho; \phi]$ where $\rho$ is the perturbation to position (indices 0–2) and $\phi$ is the perturbation to rotation (indices 3–5). This is what the code at `01 §3.1` shows:

```cpp
// Meridian internal representation (translation-first)
Pose boxplus(const Eigen::Matrix<double,6,1>& xi) const;
// xi = [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]
```

But **GTSAM** (the optimization library used in the backend) uses **rotation-first** ordering: $[\phi; \rho]$ with rotation perturbations (indices 0–2) and translation (indices 3–5).

If you mix these without converting, the math is silently wrong. A Jacobian computed in Meridian order is not compatible with GTSAM's order. The backend spec (05 §12, "Tangent-ordering adapter") is entirely devoted to handling this:

```cpp
// Adapter at the L2→L3 boundary (spec 05 §12):
// Convert from Meridian order [rho; phi] to GTSAM order [phi; rho]
Eigen::Matrix<double,6,6> reorder_meridian_to_gtsam(const Eigen::Matrix<double,6,6>& M_meridian) {
  // Permutation: [0,1,2,3,4,5] (meridian: trans|rot) -> [3,4,5,0,1,2] (gtsam: rot|trans)
  static constexpr std::array<int,6> perm = {3,4,5, 0,1,2};
  Eigen::Matrix<double,6,6> M_gtsam;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      M_gtsam(i,j) = M_meridian(perm[i], perm[j]);
  return M_gtsam;
}
```

**Why is this so dangerous?** Because if you forget the reordering:

- A Jacobian shaped like `[H_rho_rho, H_rho_phi; H_phi_rho, H_phi_phi]` in Meridian order gets fed to GTSAM as if it were rotation-first—and GTSAM silently treats the top-left block as if it's the rotation sensitivity, not translation. The optimizer then over-trusts or under-trusts different directions.
- A loop closure with covariance in Meridian order, if not reordered, causes the loop rejection algorithm to gate on the wrong axes.

**Golden rule:** The backend adapter in `05 §12` is the only place in the system where Meridian-order data is reordered to GTSAM order. Never do it elsewhere. Never assume an order without checking the comment. The specs mark every covariance with its order: `[rx,ry,rz,tx,ty,tz]` for GTSAM, `[tx,ty,tz,rx,ry,rz]` for Meridian.

### 3.5 Lie Groups, Manifolds, and Why You Need Them

The mathematics you just learned—exponential maps, tangent spaces, box-plus and box-minus—is the language of **Lie groups**. A Lie group is a smooth manifold where group operations (like composing rotations) behave nicely.

- **$SO(3)$**: the group of 3D rotations (special orthogonal group). It is a 3-manifold embedded in 4D quaternion space (or 9D matrix space). It is *not* a vector space—you cannot add rotations like vectors.
- **$SE(3)$**: the group of rigid transforms in 3D (special Euclidean group). It is a 6-manifold: 3 dimensions of translation + 3 of rotation.

The **tangent space** at the identity of a Lie group is called the **Lie algebra**. For $SE(3)$, the Lie algebra is $\mathfrak{se}(3) \cong \mathbb{R}^6$, which is exactly where our $\xi = [\rho; \phi]$ lives.

Why does this matter?

1. **Optimization**: Nonlinear least squares solvers (like GTSAM's iSAM2) linearize around the current estimate. They assume they can perturb variables in a *linear* space (the tangent space), compute gradients, and step. If you try to treat $SO(3)$ as a vector space (like early EKF-SLAM papers sometimes did), the linear approximation breaks down, and the solver diverges.

2. **Correct Jacobians**: A Jacobian $\frac{\partial h}{\partial x}$ where $x$ is a rotation must differentiate along the tangent space (the $\phi$ vector), not the quaternion components. If you differentiate w.r.t. the 4 quaternion entries, you get the wrong derivatives (they are not orthogonal to the constraint $\|q\|=1$). The correct Jacobian is computed in the tangent space and has shape $n \times 3$ for a function $h : SO(3) \to \mathbb{R}^n$.

3. **Gauge freedom** (§3.8 below): a relative-only pose graph has a 6-DoF null space. The tangent space framework makes this precise: there is a 6-dimensional subspace of perturbations that do not change any relative pose (the "rigid transformation" of the entire trajectory). Solvers must be aware of and handle this.

### 3.6 What is a Covariance Matrix, and Why 6×6 for Poses?

A **covariance matrix** $\Sigma \in \mathbb{R}^{n \times n}$ is a symmetric, positive-semidefinite (PSD) matrix that summarizes the uncertainty in an $n$-dimensional quantity.

For a pose estimate with 6-DoF tangent vector $\xi = [\rho; \phi]$, the covariance is a 6×6 matrix:

$$\Sigma = \begin{bmatrix}
\Sigma_{\rho\rho} & \Sigma_{\rho\phi} \\
\Sigma_{\phi\rho} & \Sigma_{\phi\phi}
\end{bmatrix} \in \mathbb{R}^{6 \times 6}.$$

The diagonal blocks are:
- **$\Sigma_{\rho\rho}$** (3×3): variance of the position estimate. The diagonal entries are $\sigma_x^2, \sigma_y^2, \sigma_z^2$ (in m²). If $\sigma_x = 0.1$ m, then we're about 68% confident the true position is within ±0.1 m of the estimate (1-sigma).
- **$\Sigma_{\phi\phi}$** (3×3): variance of the rotation estimate. The diagonal entries are $\sigma_{\text{roll}}^2, \sigma_{\text{pitch}}^2, \sigma_{\text{yaw}}^2$ (in rad²).

The off-diagonal blocks $\Sigma_{\rho\phi}$ encode **correlation**: if you're uncertain about the position and the rotation simultaneously, those blocks are nonzero. For example, a poorly localized camera might couple position and yaw uncertainty.

**Intuition**: The covariance matrix is the Hessian of the log-likelihood. If your estimate is normally distributed, the covariance tells you the level sets (ellipsoids) of equal probability density. A small covariance means high confidence; a large one means high uncertainty.

**Why symmetric and PSD?** Symmetry ($\Sigma = \Sigma^\top$) is required for a real covariance. PSD means all eigenvalues are ≥ 0 (in practice, we clamp them to avoid numerical issues). If a covariance matrix is not PSD, something is wrong (e.g. a coding bug, or a measurement noise model that violates Gaussian assumptions).

### 3.7 The Information Matrix: the Inverse Covariance

The **information matrix** is defined as:

$$\Omega = \Sigma^{-1}.$$

It has the same size and symmetry as the covariance, but its interpretation is inverted: **large entries mean high confidence, small entries mean low confidence**.

Why have both?

- **Filters** (like EKF) naturally produce *covariance* matrices. A filter maintains an estimate and its spread around that estimate.
- **Optimizers** (like iSAM2) naturally work with *information* matrices. When you compose measurements, you *add* information matrices (in a common basis), not covariances. This is more efficient numerically.

In Meridian, the boundary type `GaussianBlock<6>` carries a **tag** to say which form it is:

```cpp
struct GaussianBlock<6> {
  enum class Form { Covariance, Information } form;
  Eigen::Matrix<double,6,6> M;
};
```

The front-end ships covariance; the backend converts to information via inversion (or uses the covariance form directly in GTSAM's noise models). **The tag prevents silent errors.**

**Example**: Two independent measurements contribute *additive* information:

```
Sigma_total = (Sigma_1^{-1} + Sigma_2^{-1})^{-1}   (covariance composition - slower)
Omega_total = Omega_1 + Omega_2                     (information composition - faster)
```

### 3.8 Gauge Freedom: Why the Whole Map Can Float

Here is a subtle but crucial idea: **a pose graph built from *relative* measurements alone does not uniquely determine the poses. The whole map can translate or rotate rigidly without changing any relative pose.**

**Example**: You have three keyframes connected in a chain by relative odometry factors:

```
X(0) --[delta_0_1]--> X(1) --[delta_1_2]--> X(2)
```

The relative factors say: "X(1) is delta_0_1 away from X(0), and X(2) is delta_1_2 away from X(1)." These *relative* constraints have no opinion about where X(0) is in the world. You could:
1. Translate all three poses by 1 km in the *x* direction. All relative deltas stay the same.
2. Rotate all three poses around the origin by 45°. All relative deltas stay the same (in the local frame).

This 6-DoF flexibility is called **gauge freedom** (or sometimes the "null space" of the problem). Without something to anchor the map, the optimizer has multiple solutions that are all equally good for the relative constraints.

**The fix:** Add a **gauge anchor**—one constraint that breaks the freedom. In Meridian, this is not a hard prior (which would bias the global estimate toward an arbitrary initial seed). Instead, spec 05 §3 describes a custom `GaugeDampingFactor` that:
- Allows the first pose to move (so global corrections can shift the whole trajectory),
- But adds a soft damping term to the Hessian that prevents pathological drift.

This is much cleaner than a hard `PriorFactor`, which would fight loop closures and GNSS corrections that try to realign the map.

**Deeper perspective:** The null space of the pose-graph Hessian is precisely the 6-dimensional space of rigid motions: a 3-vector for translation + a 3-vector for rotation angles. The damping factor adds just enough information (λ I₆) to make the system well-posed, without pulling the solution toward a remembered value.

### 3.9 The Mahalanobis Distance and Gating

When a measurement arrives, you need to decide: is it an outlier or inlier? The **squared Mahalanobis distance** is the canonical test.

Given a residual $r \in \mathbb{R}^n$ (the difference between what you expected and what you measured) with covariance $\Sigma$, the squared Mahalanobis distance is:

$$d^2 = r^\top \Sigma^{-1} r = r^\top \Omega r.$$

This is a **scalar**. Under the assumption that the residual is normally distributed, $d^2$ follows a chi-square distribution with $n$ degrees of freedom: $d^2 \sim \chi^2_n$.

**Gating rule:** Accept the measurement if $d^2 \le \chi^2_{n,\alpha}$, where $\chi^2_{n,\alpha}$ is the **inverse CDF (quantile)** of the chi-square distribution. For example:
- $\chi^2_{6,0.99} \approx 16.81$ (a 6-DoF Pose residual at 99% confidence)
- $\chi^2_{3,0.99} \approx 11.34$ (a 3-DoF position residual at 99% confidence)

**Critically important (from spec 05 §3.2):** The reference confidence in Meridian is **always α = 0.99**. This is the binding convention across loop closure (Chapter 5) and GNSS. Use the wrong quantile or confidence level, and your system will silently reject valid measurements or accept outliers.

**Intuition**: Mahalanobis distance is the residual scaled by confidence. A residual of 1 m is huge if your position uncertainty is ±1 cm (Mahalanobis ~ 100, likely an outlier), but tiny if your uncertainty is ±10 m (Mahalanobis ~ 0.1, likely inlier).

### 3.10 Jacobians: Local Sensitivity

A **Jacobian** is how a function's output changes when you perturb its input. For a function $h(x) : \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is an $m \times n$ matrix:

$$J = \frac{\partial h}{\partial x} = \begin{bmatrix}
\frac{\partial h_1}{\partial x_1} & \cdots & \frac{\partial h_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial h_m}{\partial x_1} & \cdots & \frac{\partial h_m}{\partial x_n}
\end{bmatrix}.$$

Each column tells you how much the output changes per unit input along one axis.

**In SLAM**, Jacobians appear everywhere:
- **Measurement Jacobians**: "How does the predicted measurement change if I perturb the pose?" For example, in a LiDAR point-to-plane residual, the Jacobian is the derivative of the plane-distance with respect to the 6-DoF pose perturbation.
- **Information matrices**: The Hessian $H = J^\top J$ (or more generally, $J^\top \Sigma^{-1} J$ for weighted least squares) is the information matrix. It tells you which directions have strong gradients and which are flat.

**Critical point for rotations**: When you compute a Jacobian w.r.t. a rotation, you differentiate in the **tangent space** (the 3-vector $\phi$), not the quaternion coefficients. The Jacobian has shape $m \times 3$, not $m \times 4$.

### 3.11 Observability: Which Directions are Constrained?

Sometimes a measurement constrains only *some* of a pose's 6 DoF. A LiDAR scanning a long featureless corridor can measure *forward* motion precisely (the walls are sharp features perpendicular to the corridor), but *sideways* motion not at all (no walls on the sides). This is an **observability** problem.

Meridian quantifies this with the `ObservabilityReport` (spec 01 §3.4):

```cpp
struct ObservabilityReport {
  Frame frame;                        // which frame the scores are expressed in
  std::array<double, 6> score;        // [tx, ty, tz, rx, ry, rz], each in [0,1]
  std::optional<Eigen::Matrix<double,6,6>> eigvecs;
};
```

Each axis has a **score in [0, 1]**: 1 means fully observable, 0 means degenerate (not constrained at all). The front-end computes this from the registration Hessian (e.g., the eigenvalue ratio in ICP-style methods).

**What does L3 (the backend) do with this?** It inflates the measurement covariance along degenerate axes (spec 05 §4.3):

$$\lambda_k = 1 + (\rho_{\max} - 1) \cdot (1 - s_k)^\gamma.$$

If the score $s_k$ is 1 (fully observable), no inflation; if it is 0, inflate by $\rho_{\max}$ (default 10,000×). This ensures the optimizer does not over-trust directions that were not actually measured.

**Example**: A robot scans a corridor. The front-end's registration Hessian has large eigenvalues for forward motion, tiny eigenvalues for sideways motion. The observability report marks sideways translation with score ≈ 0. L3 inflates the covariance in that direction by 10,000×, so the optimizer down-weights the (unreliable) sideways constraint and only trusts forward motion.

### 3.12 Composing Covariances: The 6-DoF Case

When two measurements contribute to one estimate, you need to compose their covariances. The rule is:

**In a common coordinate frame, add the *information matrices* (or equivalently, add the inverted covariances):**

$$\Sigma_{\text{total}}^{-1} = \Sigma_1^{-1} + \Sigma_2^{-1}.$$

But there is a subtlety for 6-DoF poses: **the tangent ordering must be consistent**. If one covariance is in Meridian order `[rho; phi]` and the other is in GTSAM order `[phi; rho]`, you must reorder one before adding.

**Example (loop closure)**: The loop-detector provides a relative pose between two keyframes, with covariance in Meridian order. The odometry chain between those keyframes has covariance in Meridian order. To apply the loop gate (spec 05 §7), you add them:

```
Sigma_loop = Sigma_loop_measurement
Sigma_odom_chain = sum of the odometry covariances on the path
Sigma_composed = Sigma_loop + Sigma_odom_chain
d^2 = residual^T * Sigma_composed^{-1} * residual
if d^2 < chi2(6, 0.99): accept the loop
```

If you mix orderings, $d^2$ will be computed incorrectly, and the loop will be accepted or rejected based on garbage.

### 3.13 Summary: Building a Mental Model

Here is the intuition you should carry away:

1. **A pose is a rigid transform**: position + orientation, stored as a quaternion + vector.

2. **Poses form a group**: you can compose them, but they obey non-commutative algebra.

3. **Small changes to poses live in a 6D tangent space**: two 3-vectors, translation first, then rotation. This is where perturbations, Jacobians, and covariances live.

4. **Rotations are not vectors**: you cannot add rotation perturbations like vector components. Use the exponential map (tangent space ↔ manifold).

5. **Tangent ordering is load-bearing**: translation-first (`[rho; phi]`) is Meridian; rotation-first (`[phi; rho]`) is GTSAM. Mix them, and math silently breaks.

6. **Covariance matrices are 6×6 for poses**, symmetric and PSD, with blocks for position, rotation, and their correlation. Information matrices are their inverses.

7. **Mahalanobis distance** gates measurements: $d^2 = r^\top \Sigma^{-1} r$ compared against chi-square quantiles. Always use α = 0.99 in Meridian.

8. **Jacobians** relate perturbations to residuals. For rotations, differentiate in the tangent space, not the quaternion.

9. **Observability** tells you which axes are constrained by measurements. The backend inflates covariance along degenerate axes.

10. **Gauge freedom** means a relative-only pose graph does not anchor globally. The gauge damping factor breaks this symmetry without biasing the estimate.

With this toolkit, you are ready to understand how the front-end estimates keyframe poses, how the backend optimizes them globally (Chapter 4), and how loop closures and GNSS corrections feed into the graph (Chapter 5).

---

## Chapter 4 — Inside Meridian's Backend Graph: The iSAM2 Factor Graph

### Purpose and Position

The **back-end (L3)** is Meridian's global, drift-free layer. While the front-end (L2) solves a continuous-time trajectory inside a small sliding window, the back-end stitches together every keyframe the front-end emits, fuses loop closures and GNSS, and computes a single consistent **map**-frame pose for each keyframe. It does this using GTSAM's **iSAM2** — incremental Smoothing And Mapping v2 — an incremental factor-graph optimizer that solves nonlinear least squares and updates the solution efficiently as new measurements (keyframes, loops, GNSS) arrive.

Think of the back-end as the "global conscience" of the system: the front-end sees only what is within arm's reach (one small window), makes good local decisions, but cannot see if it drifted or went in circles. The back-end, by contrast, watches the *entire history* and corrects the front-end whenever it spots an inconsistency (a loop that closes, or a GNSS fix that says "actually, you are here").

### The State Being Estimated: Graph Variables

An iSAM2 factor graph is a collection of **variables** (unknowns to estimate) connected by **factors** (measurements and constraints). Meridian's graph variables are:

| Variable | Meaning | Type | When Present |
|----------|---------|------|--------------|
| **X(i)** | Pose of keyframe `i` in the `map` frame | `Pose3` (rotation + translation) | Every keyframe, always |
| **V(i)** | Velocity at keyframe `i` | 3D vector (m/s) | **Only** on restart-bridge edges (rare) |
| **B(i)** | IMU bias at keyframe `i` | 6D vector (accel bias + gyro bias) | **Only** on restart-bridge edges (rare) |
| **E(s)** | Extrinsic calibration of sensor `s` (e.g., LiDAR-to-body) | `Pose3` | When online extrinsic refinement is enabled |
| **G** | GNSS-to-map alignment (the datum transform) | `Pose3` | When GNSS is used |

The key design principle: **the steady-state graph is pose-only**. Velocity and bias, which contain information from the front-end's sliding window, are created only as a bridge when the front-end restarts and must be reset. Once the next normal keyframe arrives, those transient variables are marginalized away, keeping the graph lean and well-conditioned.

### Plain-English Picture: What the Graph Looks Like

Imagine a chain of keyframe poses: pose 1, pose 2, pose 3, …, pose 100. Between adjacent poses is a **relative constraint** (odometry) that says "I know how pose 2 relates to pose 1, because my front-end fused the LiDAR, IMU, and camera over that interval." 

Now, much later, pose 80 looks like a place you have seen before — pose 5. A loop-closure detector (L5, the place recognizer) compares their point clouds, verifies they match with geometric alignment, and says "poses 80 and 5 are actually the same place." The back-end receives this loop constraint and asks: "does this loop close consistently with the odometry chain connecting them?" If yes, the back-end adjusts the entire chain to make the loop consistent, spreading the correction naturally (old poses move less, recent poses more). If no — if the loop is so far off that it contradicts the existing trajectory — the back-end can reject it or downweight it.

GNSS adds another layer: global absolute positions that tell the map "you are at latitude X, longitude Y." The back-end fuses these too, but carefully: GNSS is often noisy and can outlier, so it uses a robust-fitting technique to keep outliers from ruining the estimate.

### The Factors: What Constraints Tie the Graph Together

Every factor is a measurement or constraint, represented as a squared error: if I predict a measurement and it differs from what I observe, I square that difference and minimize the sum. In Bayesian language, this is maximum a posteriori (MAP) estimation under Gaussian noise.

#### 1. The Between-Factor (Odometry, the Backbone)

The most common factor, and the one you will see in every edge:

```
BetweenFactor( X(i), X(j), T_measured, noise_model )
```

This says: "I measured the relative pose from keyframe i to keyframe j; it is T_measured, and I believe it with the certainty encoded in noise_model." The noise model is a covariance matrix — the larger the covariance, the more uncertain you are. The front-end hands over this constraint via the `KeyframePacket`, which carries a relative transform `T_relto_this` and its covariance `constraint_cov`.

**Key rule:** There is exactly **one** such factor per keyframe interval, ever. Not two, not three. This is called the **hand-off contract**. If you added both a relative odometry factor *and* an absolute prior *and* an IMU factor, you would be counting the same LiDAR/IMU evidence three times — your uncertainty estimates would shrink artificially, you would trust odometry too much, and you would miss loop closures later. Meridian's contract prevents that: the `constraint_kind` field in the packet is an enum that forces the choice: either `RelativeBetween` (normal, common case) or `ImuPreintegration` (fallback on restart, rare).

**Noise inflation for degeneracy:** If the front-end observes some directions poorly (e.g., motion along a featureless corridor means you do not know how far left-right you moved), the keyframe packet carries an **observability report** with six scores per axis `[0, 1]`. A score of 1 means "fully observed"; a score of 0 means "invisible." The back-end **inflates** the covariance along poorly-observed axes. The formula is:

$$\lambda_k = 1 + (\rho_{\max}-1)\,(1 - s_k)^{\gamma}$$

If you have a score of 0.1 (mostly unobserved) and ρ_max = 10000, you inflate that axis's variance by a factor of ~10000. This is honest: it says "I really do not trust this direction," so the optimizer does not falsely lock it down and later loop closures can correct it.

#### 2. The Gauge Anchor (Fixing the Gauge, Not Biasing the Solution)

A factor-graph of only relative constraints has a problem: it has six degrees of freedom in the absolute pose of the *first* keyframe. If you rotate or translate all keyframes rigidly together, the relative constraints are still satisfied — they do not care where pose 1 is in the absolute map frame. GTSAM will complain with an "indeterminate linear system" exception.

To fix this, you need an "anchor." A naive approach is a hard prior: "pose 1 is at location (0, 0, 0) with rotation identity." But this backfires: if a loop later says "actually, the whole map is rotated 10°," that hard prior fights the loop, and the map ends up twisted instead of globally corrected.

Meridian uses a **`GaugeDampingFactor`**, a custom factor that says: "add a light damping term to the first pose's optimization gradient, just enough to remove the null space, but not so much that it acts like a hard prior." It contributes no residual (no measurement), only to the Hessian's regularization. At every relinearization, it re-centers on the current linearization point, so it never pulls toward a remembered seed value. The strength is set to λ = 1 / σ², where σ is small (default 0.1 m), so the marginal covariance floors at (0.1 m)² — negligible compared to loop closures, which move poses by meters.

#### 3. The Combined IMU Factor (The Only IMU Factor, on Restarts)

Normally, the front-end fuses IMU inside its sliding window and hands a between-factor to L3. But sometimes the front-end's window diverges or loses observability, and it must restart: it stops the old window, re-initializes from the last good pose, and starts a new window. During the restart, there is no continuous relative pose; instead, the front-end integrates raw IMU samples to estimate the relative motion.

When this happens, the keyframe packet carries `constraint_kind == ImuPreintegration` and an `ImuPreintegrationSummary`: the integrated rotation, velocity, and position changes, plus Jacobians with respect to IMU bias. L3 builds a `CombinedImuFactor` — GTSAM's standard IMU factor — which also includes a random-walk assumption on bias evolution (bias does not change instantly; it drifts slowly).

**Crucial:** The IMU factor is **mutually exclusive** with the between-factor. You get one or the other per interval, never both. When a restart happens, the two endpoints of the restart bridge get velocity and bias variables, but these are **temporary**. Once the window recovers and the next normal keyframe arrives, those variables are marginalized away. The graph returns to pose-only. This keeps the back-end lean: velocity and bias are expensive (they are not constrained by loops or GNSS), so you do not want them lingering indefinitely.

#### 4. The Loop Closure Factor (Another Between-Factor, but Global)

When the place recognizer (L5) detects a loop — "keyframe 80 and keyframe 5 are the same place" — it outputs a `LoopConstraint`: the relative pose from keyframe 5 to keyframe 80, plus its covariance. L3 receives this and must decide: is it a real loop, or a false positive?

L3 runs two checks before adding a loop to the graph:

- **Pairwise Consistency (PCM):** Take the loop constraint, the current graph estimate, and the odometry chain connecting the two keyframes. Compute the cycle error: "if I go from pose 5 to pose 80 via the odometry chain, and also via the loop, do I end up at the same place?" If the cycle closes within the noise budget, the loop is consistent. If not, it is probably wrong.
  
- **Maximum-clique set:** It is rare, but you can have *multiple* pending loops. They must all be mutually consistent with each other — if two loops contradict, at least one is wrong. L3 finds the largest subset of loops that all agree with each other (a "maximum clique") and admits only those.

Once a loop passes these checks, it enters the graph as a `BetweenFactor<Pose3>` **with a robust kernel** — a statistical down-weighting if the residual is large. The kernel is **Huber**, which is smooth and convex, so it does not cause numerical instability in iSAM2. If it is an outlier, the kernel gently reduces its weight; if it is a true inlier, it pulls fully.

#### 5. The GNSS Factor (Position Measurements on the Map)

GNSS (GPS/RTK) gives absolute position in the ENU (East-North-Up) frame. But the `map` frame is a local tangent plane. The relationship between them is unknown and drifts during long missions — that is the whole reason you need GNSS.

So GNSS is modeled as a transform **G = T_map_enu**, a pose variable in the graph. The first GNSS fix is not enough to determine G: latitude and longitude constrain translation, but yaw (rotation about the vertical axis) is unobserved until the platform moves. L3 **defers** activating G until the platform has traveled a baseline (e.g., 5 m) and rotated enough that yaw becomes observable. It then runs an **Umeyama alignment** to fit G to all the buffered fixes, checking that the fitted yaw uncertainty is below a threshold (e.g., 5°). Only then does G lock into the graph.

Once G is locked, each subsequent GNSS fix becomes a factor: it says "the antenna position at keyframe i, transformed via G into map, should match my measured ENU position." The factor depends on three things:

1. The pose of keyframe i
2. The GNSS datum G
3. If the fix arrives between two keyframes, the interpolated pose at the fix time (a smooth Exp/Log interpolation, not a jump to the nearest keyframe)

Like loops, GNSS factors use a robust kernel (Huber) to downweight outliers. Also like loops, there is a "confidence gate": if the GNSS fix is less certain than the back-end's own estimate, it is skipped — it cannot improve the estimate and only adds noise.

#### 6. The Extrinsic Prior (Calibration, Pinned by Default)

If online extrinsic refinement is enabled, each sensor (LiDAR, camera) gets an `E(s)` variable: its pose relative to the body. A tight prior factor pins each to the offline-calibrated value, holding it steady unless there is evidence it drifted. Once the platform rotates and translates enough to observe the extrinsic (it needs both types of motion to separate "I moved" from "my sensor moved"), the prior is loosened and E can float. When E's covariance shrinks below a freeze threshold, it is re-tightened to the converged value and marked frozen.

### The Optimize Cycle: Keyframe In → Correction Out

Here is the rhythm of the back-end:

1. **L2 sends a keyframe:** The front-end emits a `KeyframePacket` with pose, relative odometry, and observability. L3 extracts:
   - The keyframe id and pose estimate (used as initialization)
   - The between-factor and its noise model (with observability inflation)
   - Bookkeeping: cloud handle, image, observability scores

2. **L3 stages the keyframe:** It does *not* optimize yet. It inserts the keyframe id `X(i)` as a new variable, adds the between-factor to a temporary batch, and records the packet's details (cloud handle, observability) in a per-keyframe record.

3. **Batched optimize:** On a timer (e.g., every 100 ms), or when a loop or GNSS datum locks in, the back-end flushes all staged items and calls `ISAM2::update()`. The optimizer:
   - Adds the new variables and factors to the Bayes tree (a compact factorization of the information matrix)
   - Re-eliminates only the part of the tree affected by the new factors
   - Re-linearizes any variable whose estimate moved too far (fluid relinearization)
   - Solves for the new estimates

4. **Extra passes on loops:** A loop closure can move many poses, so iSAM2 makes extra Dogleg solver passes (typically 4–5) when a loop is present in the batch, giving the nonlinear solver more iterations to converge.

5. **Extract corrections:** The optimizer returns the new estimates. L3 compares each moved keyframe to its last-published pose. If it moved by more than a threshold (e.g., 0.1 m translation), it is marked "moved" and sent to L4 (the mapper) for re-integration.

6. **Compute `map→odom` feedback:** The front-end publishes odometry in its local `odom` frame. The back-end's global `map` frame is different (especially after loops or GNSS corrections). L3 computes the rigid transform between them:

   $$T_{\text{map} \leftarrow \text{odom}} = X(n) \cdot (T_{\text{odom} \leftarrow \text{body}}(n))^{-1}$$

   where `X(n)` is the latest keyframe's global pose and `T_odom_body(n)` is the front-end's odometry estimate. This transform is sent to L2 only on genuine jumps (loops, GNSS datum lock), not on the routine gauge float.

7. **Publish telemetry:** The back-end emits timing (how long the optimize took), chi-square residuals (is the solution healthy?), relinearization counts, loop accept/reject decisions, and GNSS events.

### Observability-Aware Noise: Degeneracy Handling

Not all directions are equally observable. In a corridor, you know you are moving forward but not how far left-right. In a featureless plane, you do not know heading. Meridian's front-end computes per-axis observability scores `[0, 1]` for six directions: three translation, three rotation.

When L3 builds the between-factor, it inflates the covariance of poorly-observed axes. The formula is power-law:

$$\lambda_k = 1 + (\rho_{\max}-1)\,(1-s_k)^2$$

with default ρ_max = 10000 (inflation cap) and exponent γ = 2. If a direction is fully unobserved (`s_k = 0`), the variance inflates by 10000×. If fully observed (`s_k = 1`), no inflation. In between, smooth curve.

Why this matters: if you naively kept small covariances on unobservable axes, the optimizer would over-trust them. Later, when a loop or GNSS tries to correct in that direction, the optimizer would resist, saying "but the odometry is very sure!" and the loop correction would be wasted. By inflating the covariance upfront, you say "I am not sure about this direction," so the optimizer is free to correct.

**Example:** Imagine a straight corridor (forward motion, no turning). Lateral motion is unobservable. The scores might be `[0.9, 0.1, 0.9, 1.0, 1.0, 1.0]` (translate-first) — forward good, left-right bad, rotation fine. L3 inflates the lateral axis by ~2000×, and if a loop later says "you are actually 1 m to the left of where you thought," the optimizer accepts it because the odometry covariance is now large in that direction.

### The Window-Restart Bridge: The Only IMU Factor

Normally, L2 (the front-end) keeps a sliding window of poses and fuses all measurements inside. It outputs keyframes with clean relative odometry.

But sometimes things go wrong:
- The window loses observability (e.g., featureless motion).
- The estimator diverges (residuals explode).
- The IMU saturates (acceleration too large).

When this happens, L2 stops, freezes the last good pose, and restarts the window from scratch. But now there is a gap: no continuous window spans from the old pose to the new one, so no relative covariance.

Instead, L2 integrates raw IMU samples across the gap. It outputs a packet with `constraint_kind == ImuPreintegration`, carrying:
- Integrated rotation (exponential map)
- Integrated velocity change
- Integrated position change
- Jacobians of these w.r.t. accelerometer and gyro bias
- The preintegration covariance

L3 builds this into a `CombinedImuFactor`. The factor also includes bias evolution: biases do not jump instantly, they random-walk. So the factor links six variables: pose and bias before, pose and bias after, plus the factor itself encoding the IMU preintegration.

**Transient variables:** V (velocity) and B (bias) are created only for the restart bridge. Once the next normal keyframe arrives, they are marginalized away via the Schur complement — a well-defined operation that folds their information into a dense prior on their neighbors and removes the variables. The graph returns to pose-only.

**Why this is clean:** Restart edges are rare (divergence is not the normal case). If they happen, you pay the cost of transient variables for one factor. Then they are gone. You never accumulate hours of V/B variables in a long mission.

### GNSS Datum Alignment: Umeyama Fit with Yaw Observability Gate

GNSS is absolute, but the map frame is relative. When the first fix arrives, it says "you are at some latitude/longitude/altitude," but you do not know how the local map is oriented relative to ENU. You cannot solve for that rotation from a single fix.

L3 buffers fixes until the platform moves enough:

1. **Baseline gate:** Have you traveled at least 5 m? (Pure translation plus jitter cannot separate yaw from translation.)
2. **Velocity gate:** Have you moved fast enough (0.5 m/s) for at least 5 fixes? (Stationary multipath cannot be trusted.)
3. **Span gate:** The fixes spread over at least 3 m along the dominant axis? (Tiny motion is noise.)

When these gates pass, L3 runs a **Umeyama similarity transform** to align the SLAM-estimated antenna trajectory to the ENU fix positions. Umeyama finds the rotation, translation, and scale that best overlap two point clouds. L3 restricts it to 4 DoF: yaw (heading) + 3D translation; roll and pitch are locked to gravity, scale is 1 (we do not expect the map to shrink).

L3 also computes the Hessian of the yaw parameter from the Gauss-Newton approximation. If yaw uncertainty σ_yaw is larger than 5°, L3 rejects the fit and keeps buffering: the baseline is not long enough or collinear enough to observe yaw. Once σ_yaw < 5°, the datum G locks into the graph.

**Why this is important:** A poor yaw initialization bakes a fixed heading error into every GNSS residual thereafter. By insisting on good observability before locking, you avoid a slow, insidious drift.

### Factor Removal and Commitment: Why Loops Are Doubly Checked

Once a factor enters iSAM2, removing it is expensive — the Bayes tree must be re-factored. But loops can be wrong, so L3 needs to change its mind sometimes.

**PCM pre-filter (before adding):** L3 checks consistency *before* a loop enters the graph. If the loop contradicts the odometry chain, it is rejected outright. If multiple loops arrive, L3 finds the maximum clique (largest mutually consistent subset) and admits only those.

**Robust kernel (inside the graph):** Once a loop is in, it is wrapped in a Huber robust kernel. This is a smooth function that downweights large residuals without driving them to zero. It absorbs borderline fits gracefully.

**Off-thread batch GNC (optional, experimental):** For extra safety, L3 can periodically extract the loop sub-graph and run a heavier nonlinear optimization (Graduated Non-Convexity, which anneals toward a redescending robust loss). Loops that end up with low weight (`w < 0.1`) are removed and returned to the pending buffer. This happens off the live optimization thread, so it does not stall the front-end.

### Relinearization and the Bayes Tree

iSAM2 is "incremental" because it does not re-solve the whole graph every time. Instead, it maintains a factorization of the information matrix as a **Bayes tree** — a tree of cliques, each representing a block of variables. When new factors arrive:

1. Insert the new variables and factors into the tree.
2. Only the cliques containing new variables or variables whose estimates moved too far are re-eliminated (re-factored).
3. The rest of the tree stays cached.

This is much faster than a batch solve, but it has a cost: if estimates move a lot, relinearization can become expensive. L3 configures this carefully:

- **Pose thresholds:** Re-eliminate a pose if its estimate moves more than 0.05 rad (rotation) or 0.1 m (translation).
- **Velocity/bias thresholds:** Tighter, because they are smaller variables.
- **GNSS origin threshold:** Tight (1 mm translation, 5 mm rotation), because it is freshly locked and settling.
- **Relinearize skip:** Always relinearize (skip = 1), because L2 keyframes arrive slowly enough that batching saves cost.

**Extra passes on loops:** When a loop is present, iSAM2 makes extra Dogleg passes (the solver tries multiple step sizes in the trust region). This helps the nonlinear solver converge when poses move significantly.

### Implicit Correction Feedback to the Front-End

When the back-end optimizes and poses move, L2 needs to know. The pipeline calls `IFrontEnd::apply_correction()` with a `GraphUpdate` saying which keyframes moved and by how much. But L2 does not re-anchor on *every* keyframe motion — it re-anchors only on genuine jumps: loops, GNSS datum lock, or a cumulative drift exceeding a threshold. Why? Because re-anchoring means shifting the CT spline's control points; if the old marginalization prior is not transported along with the shift, it fights the new estimate and the next window optimization becomes expensive.

### Online Extrinsics (Off by Default)

The LiDAR-to-body and camera-to-body transforms are usually calibrated offline, but they can drift (thermal expansion, mechanical flex). L3 *can* refine them online by keeping them as graph variables.

**Why off by default:** An extrinsic is **weakly observable**. The platform must both rotate *and* translate to separate "the body moved" from "the sensor moved relative to the body." If refinement is on but the motion is purely translational (driving straight), the extrinsic is unconstrained and becomes a source of indeterminacy.

**When on:** Only if the offline calibration is trusted but suspected to drift, *and* the mission promises sufficient excitation (e.g., navigating an indoor space with turns and loops). Even then, L3 gates extrinsic updates until the excitation gates open, holds it with a loose prior, and freezes it once the marginal covariance is small.

### Marginalization: Keeping the Graph Lean

A multi-hour mission produces thousands of keyframes. The graph stays full: L3 never removes poses (loops can target old keyframes, so you cannot marginalize them away). But **transient inertial variables** (velocity and bias from restart bridges) are marginalized the moment they are no longer needed.

Marginalization via **Schur complement** is the standard technique: when you remove a variable from the optimization, you fold its information into the variables it touches, leaving a dense linear prior on them. This is exact at the moment of marginalization but frozen afterward (does not relinearize). Since only transient V/B are marginalized — never poses — this is fine: the pose graph stays fully flexible.

### Tangent-Space Ordering: A Silent Bug Avoided

GTSAM represents poses as `Pose3` with rotation and translation, and computes in rotation-first tangent order: `[rx, ry, rz, tx, ty, tz]`. Meridian's core uses translation-first: `[tx, ty, tz, rx, ry, rz]`. A permutation matrix must convert covariances and Jacobians when they cross into GTSAM. (See Chapter 3 §3.4 for the full reorder adapter and why the permutation matters.)

**Why this matters:** If you forget to reorder, the noise model is silently wrong. Say the front-end says "I trust translation but not rotation." If covariances are reordered wrong, GTSAM sees "I trust rotation but not translation" — backward. The optimizer would then over-trust noisy rotation and under-trust good translation. Subtle, deadly.

L3 reorders exactly once at every boundary: when building a between-factor, when accepting a loop, when processing a GNSS fix. The reorder is audited in a unit test that perturbs a pose and checks the residual grows along the right covariance axis.

### Debug and Observability

L3 emits rich telemetry:

- **Chi-square health:** After each optimize, compute the nonlinear error and chi-square. Large chi-square (> 2× the DoF) signals unmodeled outliers or poor initialization.
- **Per-axis observability inflation:** Publish the actual per-axis variance multipliers, so you can see which directions are degenerate.
- **Loop and GNSS events:** When a loop is accepted or rejected, log the PCM distance, fitness, and reason. When GNSS is skipped, logged why (quality gate, confidence gate, deferred datum).
- **Relinearization counts:** How many variables relinearized this step? If it is high, the graph might be far from the linearization point.
- **Timing:** How long did `ISAM2::update` take? Useful for load budgeting.

All of this flows through the `TelemetrySink` to the wrapper, which publishes it as ROS topics, logs it, or feeds it to rviz for visualization.

### Summary: The Complete Contract

The back-end's contract, in brief:

1. **Take one keyframe packet at a time.** Extract the pose guess, the between-factor, and observability scores. Inflate covariance for degenerate axes.

2. **Stage it (do not solve yet).** Let them batch.

3. **On a timer or on a major event (loop, GNSS lock), call optimize().** iSAM2 updates the Bayes tree and returns new estimates.

4. **Extract movers.** Keyframes that moved > threshold go to L4 for re-integration. The map→odom transform goes to L2 for correction feedback.

5. **Handle loops and GNSS carefully.** Pre-filter loops for consistency before adding. Use robust kernels to downweight outliers. Defer GNSS until yaw is observable.

6. **Publish corrections and telemetry.** Let the rest of the system know what moved, and emit debug signals so an operator sees health and observability.

The result: a globally consistent, loop-corrected, GNSS-aligned estimate that improves as loops close and measurements accumulate.

---

## Chapter 5 — Loop Closure and Robustness

### The problem: Drift, revisits, and why they matter

Imagine you're piloting a robot through a large building. The onboard sensors — LiDAR, camera, IMU — feed a real-time odometry estimator that whispers "you're here" a hundred times per second. The odometry is locally smooth and accurate: over the next few meters, it's reliable. But over an hour of continuous operation, small errors accumulate. A camera might mis-track a featureless wall; the IMU might slowly bias; an unfortunate dust cloud might fool the point-cloud alignment. After enough time, the map drifts. The robot thinks it is somewhere it is not.

Then the robot rounds a corner and recognizes a place it visited earlier — a distinctive doorway, a parking lot with a certain configuration of bollards, a conference room with specific furniture. The odometry is wrong about where that place is *now*, but the robot's memory is right about where the same place was *before*. If the system can detect that match, it can emit a **loop constraint**: "I was here before (at keyframe 412) and I'm here again now (at keyframe 2025); the two places are the *same*, separated by a known rigid transform." That constraint is then fed into a global optimization engine that bends the entire recorded trajectory to make both occurrences of that place line up. The map stops drifting and becomes globally consistent.

This is **loop closure** — the mechanism Meridian uses to anchor the trajectory and correct drift.

### Why loop closure is risky

But here's the catch: a **false loop is catastrophic**. If the system incorrectly matches two *different* places that merely *look* similar (two identical conference rooms, two parking rows that are structurally the same), injecting that false constraint as fact will weld the wrong parts of the map together. The optimization engine will then bend the trajectory in a way that contradicts all the odometry and real loops, corrupting the entire global estimate. The damage is often unrecoverable — the system cannot "undo" a false loop without human intervention. (This is why Appendix R.6 of the spec says: "a missed loop costs accuracy; a **false loop can be unrecoverable.**")

So the system trades detection rate for precision: it is extremely conservative. It would rather miss a loop (lower recall, less correction) than inject one false match (which corrupts the map). Meridian achieves this conservatism through a **cascade of increasingly expensive filters**, each one more discriminative than the last, arranged in a funnel: the cheap stages run on everything and reject the vast majority of false positives quickly; the expensive stages see only a small, high-quality candidate set.

### The cascade: Four stages of escalating scrutiny

Meridian's loop detector runs four stages in sequence:

1. **Stage A — Scan Context++ retrieval** (milliseconds): A fast, global rotation-invariant "fingerprint" of a LiDAR scan, used to propose candidate places. High recall, mediocre precision — a coarse semantic question: "does this place *feel* like a place I've seen before?"

2. **Stage B — STD/BTC re-ranking** (milliseconds): Local geometric descriptors that extract stable keypoints and triangles from the scan. Far more discriminative than global shape, they yield a **6-DoF initial guess** for the alignment. Answers: "if these two places *do* match, how would they be oriented relative to each other?"

3. **Stage C — GICP geometric verification** (10–100 ms): Generalized Iterative Closest Point — a plane-to-plane point-cloud registration algorithm that refines the alignment and produces a **metric transform and a fitness score** (how many inlier points agree with the match). This is the first stage that produces a number the optimizer can trust.

4. **Stage D — PCM batch consistency** (milliseconds): Pairwise Consistent Measurement set maximization — a graph-based outlier filter that asks a purely *geometric* question: is this loop mutually consistent with other loops we already trust and the odometry chain between the same two keyframes? Rejects loops that are individually plausible but globally inconsistent (the perceptual-aliasing catch-all).

If any stage rejects a candidate, it does not proceed to the next. By the time a loop reaches Stage D, it has survived three increasingly expensive filters. And even *after* a loop is admitted (passes all four stages), the back-end applies two more layers of robustness (a Huber kernel and off-thread GNC) to catch anything the cascade missed.

---

### Stage A — Scan Context++: The rotation-invariant fingerprint

#### Intuition first: Why rotation-invariance matters

A LiDAR scan is a point cloud captured from one viewpoint. If you rotate the robot 180° and scan the same room again, you get the same 3D structure but with points in completely different directions. A naive feature matcher would not recognize them as the same place because "forward" is now "backward."

Scan Context solves this by encoding a scan as a **2D image** in polar coordinates (angle and radius from the sensor), where each pixel's value captures how *tall* the tallest point is in that direction. Because the image is indexed by angle (azimuth, yaw), rotating the robot simply **shifts the image columns** — a circular shift. So the algorithm can compute distance not just once, but for every possible rotation offset, and find the rotation that makes the two images most similar. The rotation that minimizes distance is the robot's heading offset.

#### The math: Building and comparing the descriptor

For each body-frame point $\mathbf{p} = (x, y, z)$, compute the cylindrical coordinates:
$$\rho = \sqrt{x^2 + y^2}, \quad \theta = \operatorname{atan2}(y, x) \in [0, 2\pi).$$

Divide the radial and azimuth ranges into bins (default: 20 radial "rings," 60 azimuth "sectors," out to 80 meters). Bin $[r, s]$ contains all points where radius falls in ring $r$ and azimuth falls in sector $s$. The **descriptor value** at that bin is the **maximum z-height** of all points in it (empty bins are 0):

$$\mathbf{I}[r, s] = \max_{\mathbf{p} \in \text{bin}(r, s)} z(\mathbf{p}).$$

This is the **Scan Context matrix** $\mathbf{I} \in \mathbb{R}^{20 \times 60}$.

To find matches, compute two summary keys:

- **Ring key** $\mathbf{k}_{\text{ring}} \in \mathbb{R}^{20}$: For each ring $r$, count what fraction of sectors have non-empty bins. This is **rotation-invariant** because rotating the image shifts columns within each row, leaving the row occupancy unchanged. Use this to do a fast KD-tree search.

- **Sector key** $\mathbf{k}_{\text{sec}} \in \mathbb{R}^{60}$: For each sector (column), the mean z-height. Use this to align the yaw quickly.

To compare two scans, compute a **column-wise cosine distance** under every possible circular shift (column offset) $n$:

$$d(\mathbf{I}_q, \mathbf{I}_c) = \min_{n \in [0,60)} \frac{1}{60} \sum_s \left( 1 - \frac{\mathbf{c}_q^s \cdot \mathbf{c}_c^{(s+n) \bmod 60}}{\lVert \mathbf{c}_q^s \rVert \lVert \mathbf{c}_c^{(s+n) \bmod 60} \rVert} \right),$$

where $\mathbf{c}^s$ is column $s$ of the matrix. The minimizing $n^\star$ tells you the yaw offset: $\Delta\psi = 2\pi n^\star / 60$ (each sector is 6°). Accept the match if the distance is below a dataset-dependent threshold (e.g. 0.13).

#### Scan Context++ adds lateral invariance

The original Scan Context handles yaw well. But if the robot translates sideways (perpendicular to its heading), the polar image distorts. Scan Context++ fixes this by *also* building a **Cartesian context** (regular x-y bird's-eye view) and computing a lateral offset the same way. This gives not just a yaw but also a coarse translation guess.

#### Stage A algorithm and complexity

```
For each new keyframe:
  1. Accumulate the latest 5 keyframes into a body-frame submap
  2. Build the Scan Context matrix from the submap
  3. Compute ring and sector keys
  4. Insert into a KD-tree indexed by ring key
  
When querying:
  1. Retrieve ~15 keyframes from the KD-tree (ring-key NN search)
  2. For each candidate, compute SC distance and coarse yaw
  3. Keep only top-5 by distance
```

**Complexity:** The KD-tree search is $O(\log N)$; comparing distances is $O(N_r N_s \cdot \text{shift-band})$, roughly $O(20 \times 60 \times 10) = O(12k)$ per candidate — sub-millisecond. Stage A is the fast sieve.

#### Critical guardrail: Never accept on distance alone

Scan Context is a *recall* device: it has high recall (catches most real revisits) but is precision-limited (can confuse similar-looking places). **Stage A never accepts a loop.** It only *proposes candidates* that Stage B and beyond must confirm. This is the crucial design point that makes the cascade work.

---

### Stage B — STD/BTC: Local geometric structure and descriptor-derived initialization

#### The perceptual aliasing problem

Two corridors, architecturally identical, will have nearly identical Scan Context. Two parking rows of bollards will look the same to Scan Context. Two enclosed courtyards with the same geometry. The global descriptor cannot distinguish them.

But their **local structure differs**: one corridor has an electrical panel on one wall and a doorway on the other; the other has a fire extinguisher and a window. A local geometry descriptor can capture these details and tell them apart.

#### Stable Triangle Descriptor (STD) intuition

Extract stable **keypoints** from the scan — corners, boundary extrema, plane-pair intersections — that repeat across different viewpoints of the same place. Form all **triangles** from keypoint triples. Describe each triangle by its **sorted side lengths** $(ℓ_1 \leq ℓ_2 \leq ℓ_3)$ and the **angles between the plane normals** at the three vertices: a 6-D descriptor invariant to both rotation and translation.

Two triangles from the same scene should have similar sides and angles, even if the viewpoint changed. Use a hash table keyed by quantized side lengths to vote for matching triangle pairs. Triangles that collect votes are the real correspondences. Use **RANSAC** on these correspondences to solve for the rigid transform between the two scans (keypoint sets).

#### Binary Triangle Combined (BTC): Adding local appearance

STD triangles can have the same shape but sit in different parts of the environment (same geometry, different context). BTC augments each keypoint with a **binary occupancy code** — a bit-string capturing whether a small neighbourhood around the keypoint is occupied. Two keypoints are appearance-compatible only if their codes have high Hamming similarity (e.g., > 60%). This is a cheap pre-filter before the full triangle hash vote, and it **breaks perceptual aliasing**: identical triangles in different corridors will have different binary codes.

#### Stage B algorithm

```
For each candidate from Stage A:
  1. Extract keypoints from both query and candidate submaps
  2. Build BTC descriptors: side-length triples + binary codes
  3. Hash-vote on side-length matches, gated by Hamming similarity
  4. If enough matches (e.g., >= 4):
       - Use RANSAC to solve the keypoint rigid transform
       - Gate the solution by yaw (can't deviate far from Stage A's yaw)
       - Compute a combined score from SC distance, inlier ratio, residual mean
  5. Keep top-2 by score
```

**Output:** A **6-DoF pose guess** $\mathbf{T}_{\text{guess}}$ (rotation and translation) good enough to initialize a nonlinear optimizer. This is the key innovation: Stages A and B together produce not just a yes/no but a concrete *hypothesis* for how the two scans align.

---

### Stage C — GICP: Metric verification and trustworthy fitness

#### Why GICP?

Stage B produces a 6-DoF guess. But guesses can be wrong — the RANSAC inlier count might be high by chance, the geometry might be degenerate, or the initial pose might be outside the basin of convergence for point-cloud registration.

**GICP (Generalized ICP)** is a classical plane-to-plane registration algorithm. Unlike point-to-point ICP, which assumes each point matches to the nearest point, GICP assumes each point in the source cloud lies on a plane whose covariance is estimated from the local neighbourhood. It solves for the rigid transform that minimizes the **squared Mahalanobis distance** from each source point to the target plane:

$$\hat{\mathbf{T}} = \arg\min_{\mathbf{T}} \sum_i \mathbf{d}_i^\top \left( C^Q_i + \mathbf{R} C^P_i \mathbf{R}^\top \right)^{-1} \mathbf{d}_i,$$

where $\mathbf{d}_i = \mathbf{q}_i - \mathbf{T} \mathbf{p}_i$ is the residual, and $C^P_i, C^Q_i$ are the covariances of the source and target point neighbourhoods.

This is more robust to partial overlap and varying point density than point-to-point ICP. And critically, GICP returns the **Hessian matrix $\mathbf{H}$ at convergence**, which Meridian inverts to get the **loop's information matrix** (the inverse of covariance) — exactly what is needed to scale how much the loop should influence the global optimization.

#### Stage C inputs and outputs

- **Source cloud:** The query keyframe's points (downsampled, ~1000 points)
- **Target cloud:** The candidate keyframe's submap (last 5 frames composed, ~10k points after downsampling)
- **Initial guess:** $\mathbf{T}_{\text{guess}}$ from Stage B
- **Outputs:**
  - **Refined transform** $\mathbf{T}$ (maps source into target frame)
  - **Fitness** $\in [0, 1]$ — a trust score combining overlap and residual mean:
    $$\text{fitness} = \text{overlap} \cdot \exp\left( -\frac{\text{rmse}}{\sigma_{\text{fit}}} \right),$$
    where overlap is the inlier ratio and rmse is the mean squared error.
  - **Condition number** (ratio of largest to smallest eigenvalue of $\mathbf{H}$) — high cond means the alignment is geometrically degenerate (e.g., a tunnel constrains yaw but not along-axis translation).

#### Acceptance gates

```
if converged AND fitness >= 0.6 AND overlap >= 0.4 AND rmse <= 0.3 m AND cond <= 10000:
  accept loop
else:
  reject (log reason)
```

Each gate has a **physical meaning**: overlap $\geq 0.4$ means at least 40% of source points found correspondences; fitness ties the two acceptance criteria together; condition number rejects degenerate geometries that cannot constrain all DoF equally.

---

### Stage D — PCM: Global consistency and the defense against perceptual aliasing

#### The loop consistency problem

Suppose Stage C produces two loops: one between keyframes 100 ↔ 500 (a genuine closure), and another between keyframes 100 ↔ 1000 (a false positive on a perceptually aliased corridor). Both pass GICP gating. Both are, individually, geometrically plausible. But they are **mutually inconsistent**: if 100 ↔ 500 is correct, then the odometry chain 500 → 1000 should carry you from one place to another, and the two loops imply contradictory positions for keyframe 1000.

PCM rejects such inconsistent sets by asking: "which subset of proposed loops are *all mutually consistent with each other AND consistent with the odometry chain?*" The answer is a **maximum clique** in a consistency graph — a discrete, combinatorial yes/no gate applied *before* any loop touches the optimizer.

#### Pairwise consistency test (the math)

Two loops $z_{ik}$ (from keyframe $i$ to keyframe $k$) and $z_{jl}$ (from $j$ to $l$) are consistent if you can traverse loop → odom → loop → odom and close the loop (return to identity) within noise:

$$\mathbf{r} = \log\left( \hat{\mathbf{T}}_{ij}^{\text{odom}} \cdot \mathbf{T}_{jl}^{\text{loop}} \cdot \hat{\mathbf{T}}_{lk}^{\text{odom}} \cdot \mathbf{T}_{ki}^{\text{loop}} \right)^\vee \in \mathbb{R}^6,$$

where $\log$ is the matrix logarithm (maps back to the tangent space), the $\vee$ operator extracts the 6-D vector, $\hat{\mathbf{T}}^{\text{odom}}$ is the odometry chain (corrected, from iSAM2), and $\mathbf{T}^{\text{loop}}$ is the proposed loop. Gate the squared Mahalanobis distance:

$$\mathbf{r}^\top \Sigma^{-1} \mathbf{r} \leq \chi^2_{6, 0.99} \approx 16.81,$$

where $\Sigma$ is the combined covariance of the two loops and the odometry chain. If satisfied, the loops are pairwise consistent.

**Intuition:** The residual $\mathbf{r}$ measures how much the cycle closes. The covariance scales the test — if you're far from home (large accumulated odometry error), the test is loose; if you're still confident, it's strict.

#### Single-loop self-test (crucial for the first loop)

The max-clique problem requires at least two loops to have edges. But what if this is the *first* loop in the session? It would form a 1-node clique and trivially pass.

To prevent this, every loop is first tested against the odometry chain it claims to close, independently:

$$\mathbf{r}^{\text{self}} = \log\left( \hat{\mathbf{T}}_{ik}^{\text{odom}} {}^{-1} \mathbf{T}_{ik}^{\text{loop}} \right)^\vee,$$

gated on the same chi-square quantile. This rejects a lone loop that contradicts a still-confident odometry chain. Later, when more loops accrue, the pairwise max-clique adds *cross-loop* consistency on top.

#### Max-clique selection and incremental operation

Build a graph: nodes = proposed loops; edge = pairwise consistent. Find the **largest clique** (subset where every pair is consistent). Run this incrementally:

1. Maintain a "trusted set" of loops already in iSAM2.
2. For each new loop, test it against all trusted loops (pairwise checks).
3. Run max-clique on the new+trusted loops.
4. Emit only the *new* loops in the clique (ones that were proposed this iteration).
5. Update the trusted set for next time.

This avoids stale inconsistencies: if a loop later proves inconsistent, it is excluded from the clique and will not return unless new evidence arrives.

---

### The result: A LoopConstraint ready for optimization

After passing all four stages, a loop is packaged as a **`LoopConstraint`**:

```cpp
struct LoopConstraint {
  uint64_t from_id, to_id;       // which keyframes (from = older, to = newer)
  Pose T_from_to;                // rigid transform: maps 'to' body into 'from' body
  PoseCov6 cov;                  // 6×6 covariance (scales how much the loop influences)
  double fitness;                // GICP fitness [0,1] (used by the robust kernel)
};
```

**The covariance scaling rule** (fitness-driven loosening):
$$\Sigma_{\text{loop}} = s(\text{fitness}) \cdot \mathbf{H}^{-1}_{\text{gicp}},$$

where $s(\text{fitness}) = \left( \frac{\text{fitness}_{\text{ref}}}{\max(\text{fitness}, 0.5)} \right)^2$, with $\text{fitness}_{\text{ref}} = 0.6$ (the GICP min threshold). A barely-passing loop (fitness ≈ 0.6) gets a covariance inflation factor of ~1; an excellent loop (fitness → 1) gets a tight covariance. This encodes the principle: **a marginal loop barely tugs the graph; a confident loop steers it strongly.**

Per-axis degeneracy inflation is also applied here: if the GICP information matrix has a near-zero eigenvalue along a certain direction (e.g., a tunnel loop that constraints yaw but not along-tunnel translation), that axis's covariance is inflated 100×, so the weak direction barely contributes to the optimization.

---

### Back-end robustness: The two-layer safety net

After L5 emits a `LoopConstraint`, L3 (the back-end) applies two more layers of robustness to catch anything the cascade missed:

#### Layer 1: Committed Huber kernel

The loop factor enters iSAM2 immediately, wrapped in a **Huber robust kernel**. Huber is convex, so its reweighting does not introduce instability in the incremental solver. It is "committed" — the kernel is fixed at admission, not annealed. Its role: absorb a borderline residual (e.g., a point cloud with a few outlier clusters) without letting one bad correspondence dominate the whole alignment.

#### Layer 2: Off-thread batch-GNC

Graduated Non-Convexity (GNC) is a **batch** annealing schedule that starts convex and gradually becomes redescending (outlier-preferring). Because it is inherently batch and non-convex, it runs **off the live odometry thread** over a **loop sub-graph** (just the loop factors and the odometry chain bridging them) to a convergence decision. Loops that GNC drives to a near-zero weight are marked for removal and returned to the PCM quarantine buffer (maybe they are real, maybe they'll be consistent once more loops arrive).

**Why both?** PCM is a discrete *pre-admission* gate (consistent-set selection before the factor exists); Huber is continuous *incremental* robustness (per-factor weighting during solving); GNC is *batch* outlier judgment that also catches cases where the *odometry* (not the loop) was wrong. Complementary layers, not redundant.

---

### Putting it together: A worked example

Imagine a robot exploring a building. At keyframe 27, it scans a distinctive entrance area. At keyframe 412 (much later), it re-enters from a different door but recognizes the same area. Here's what happens:

1. **Stage A (Scan Context++):** Retrieves ~15 candidates from KD-tree, including keyframe 27. Computes SC distance and coarse yaw. Returns top-5.

2. **Stage B (STD/BTC):** Extracts keypoints and triangles from submaps (kf 25–29 and 410–414). Matches triangles, rejects by binary appearance, solves RANSAC. Returns top-2, one of which is kf 27.

3. **Stage C (GICP):** Aligns kf 412's cloud to kf 27's submap using the Stage B guess. Converges with fitness=0.8, overlap=0.65, rmse=0.15 m. All gates pass. Outputs $\mathbf{T}_{27 \leftarrow 412}$ and the Hessian.

4. **Stage D (PCM):** Tests against the odometry chain from 27 to 412 (Δ of 385 keyframes, large accumulated uncertainty). The loop is consistent. If other loops exist, tests pairwise consistency. Included in the clique. Emitted.

5. **Back-end Layer 1 (Huber):** Loop enters iSAM2 as a `BetweenFactor`, wrapped in Huber. Each `ISAM2::update` relinearizes from this new factor backward up the Bayes tree, propagating the correction. The trajectory from 27 to 412 bends to align the two occurrences of the entrance.

6. **Back-end Layer 2 (GNC):** Off-thread consolidation extracts the sub-graph, re-optimizes with GNC annealing. Loop confirms as an inlier. Committed weight remains.

7. **L4 re-integration:** The back-end's `GraphUpdate` lists moved keyframes (27–412). L4 clears the TSDF voxels in those regions and re-integrates the retained keyframe clouds at corrected poses. The map corrects.

The result: **the trajectory is globally consistent; the map is no longer drifted**.

---

### Why synthetic/injected loops for testing?

Early in development and testing, before an automatic place recognizer is fully tuned, Meridian supports **injecting synthetic loops by hand** — specifying in test data that keyframes $A$ and $B$ should be constrained with a known relative transform. This lets the team:

- Test the back-end's loop optimization in isolation (does it bend the trajectory correctly?).
- Verify that de-integration works (does L4 rebuild the right regions?).
- Tune the robust kernels without waiting for a real loop detector.
- Validate the covariance inflation logic (does a degenerate loop barely contribute?).

Once the automatic detector is working, injected loops become a regression test: if the detector misses an obvious revisit that an injected loop would catch, it is a tuning opportunity.

---

### Conclusion: Defense in depth

Loop closure in Meridian is a **layered defense**:

1. A cheap, global **Scan Context** proposer (high recall, low cost).
2. A local **STD/BTC** verifier that yields a 6-DoF guess (precision + initialization).
3. A metric **GICP** refiner with fitness scaling (trustworthy transform + covariance).
4. A combinatorial **PCM** gate that enforces global consistency (discrete outlier rejection).
5. A **Huber kernel** inside iSAM2 (incremental robustness).
6. A **batch-GNC** consolidation off-thread (heavy outlier judgment).

Each rung catches a different failure mode. A false loop has to slip through *all* of them — unlikely, and even if it somehow does, the cascading robustness makes it easy for the optimizer to de-weight it. The result is a system that **tolerates the real world**: it closes loops when it is confident, it rejects when it is unsure, and when it does accept a marginal loop, the global estimate stays robust.

---

## Chapter 6 — Glossary & How to Navigate

### (A) Alphabetical Glossary of Terms & Acronyms

#### ATE
**Absolute Trajectory Error.** A metric for ground-truth evaluation: the RMS difference between estimated and ground-truth keyframe poses in the global frame, after alignment by rigid Procrustes fit. Used in `DATASET.md` and the evaluation harness (spec 10).

#### BTC
**Binary + Triangle Combined.** A local geometric descriptor that extends the Stable Triangle Descriptor (STD) with a per-keypoint binary occupancy code, enabling discrimination between self-similar places (e.g., identical corridor geometry in different locations). Implemented in L5 place recognition, stage B. See *STD* and Chapter 5.

#### Calibration (online refinement)
The process of refining extrinsic transforms (sensor-to-body) and, optionally, temporal offsets *inside the factor graph* as the system runs. Controlled by `Extrinsic.refine_online` in the calibration set. Enabled by default; refined values are broadcast back to L2 as a versioned `CalibrationSet` snapshot, maintaining causal consistency. See spec 08 §3 and spec 05 §10.

#### Continuous-Time (CT)
The front-end's trajectory representation: a B-spline $T(t) \in SE(3)$ parameterised by cubic Hermite segments over fixed knot intervals, enabling per-point registration at true sample times without intermediate "warp to reference time" steps. Eliminates rolling-exposure and multi-rate fusion distortions by design. See spec 04 §2 and spec 00 §7.5.

#### CT vs. discrete-time
CT fuses all measurements on a single spline trajectory evaluated at each measurement's true time. Discrete-time (e.g., iEKF) estimates a pose at a chosen reference time and separately deskews/compensates per-sensor observations. Meridian is CT; the retired iEKF oracle showed both satisfy `IFrontEnd`.

#### Degeneracy / Observability
A directional loss of sensitivity in the front-end's sliding window: e.g., translation unobservable in a long corridor, or rotation weak in a featureless tunnel. Quantified by `ObservabilityReport.score[6]` per axis (order: tx, ty, tz, rx, ry, rz) in [0,1], where 1 = fully observable. Low scores trigger noise inflation in the back-end (spec 05 §4.3) and looser loop gates (spec 07 §9.2). See Chapter 3 §3.11.

#### Deskew
Motion compensation: rotating and translating each LiDAR point from its sensor-frame position at sample time $t_i$ to the body frame at a reference time (cold-start: keyframe stamp; steady-state: implicit in CT evaluation). Requires knowledge of the trajectory; bootstrapped by IMU-only integration before the CT window initialises (spec 00 §7.2–§7.3).

#### DoF
**Degrees of freedom.** For a pose/transform (6-DoF: 3 translation + 3 rotation), for motion (18-DoF: the `NavState` error dimension comprising position, rotation, velocity, gyro bias, accel bias, and gravity), or for a graph quantity (number of independent constraints).

#### ESDF
**Euclidean Signed Distance Field.** A volumetric map where each voxel stores the signed distance to the nearest surface (positive in free space, negative in occupancy). Used for path planning and collision avoidance. Designed-for in spec 06 §10 but deferred: not built in the first-pass system.

#### ENU
**East-North-Up.** A local rectilinear coordinate system for geodetic data, with the origin at a fixed latitude/longitude/altitude (the GNSS datum). The `map` frame is gravity-aligned ENU. See spec 01 §2.2.

#### Factor Graph
A probabilistic graphical model where nodes are variables (poses, velocities, biases, extrinsics, GNSS origin) and edges are constraints (odometry, GNSS, loop closures, priors). The posterior is the product of all factors' negative log-likelihoods. L3 (iSAM2) solves the MAP estimate incrementally. See Chapter 2.

#### Gauge (gauge ambiguity / fixing the gauge)
The 6-DoF translation + rotation freedom of a relative-pose-only graph: without an absolute anchor, the entire trajectory can be rigidly moved/rotated and the relative constraints remain satisfied. Fixed by a `GaugeDampingFactor` on the first keyframe (spec 05 §3). See Chapter 3 §3.8.

#### GICP
**Generalized ICP.** A point-cloud registration algorithm that treats each point's uncertainty as a covariance (the "surfel" or distribution-to-distribution model), making it robust to partial overlap and differing sampling compared to point-to-point ICP. Used in L5 stage C for loop-closure geometric verification. See spec 07 §7 and Chapter 5.

#### GNC
**Graduated Non-Convexity.** A robust estimation framework for outlier rejection: gradually transition from a convex relaxation (easy to optimise) to a sharp outlier penalty (strong down-weighting). Applied to loop-closure and GNSS outliers in the back-end to prevent false loops from distorting the estimate. See spec 05 §8 and `Appendix R.5`.

#### GNSS (GNSS anchoring, GNSS alignment)
Global Navigation Satellite System fixes: latitude/longitude/altitude measurements with covariance. L2 uses GNSS as weak anchors for drift containment; L3 estimates the datum alignment (`T_map_enu`) as a graph variable, so GNSS can realign the map post-hoc (spec 06 §6).

#### GTSAM
**Georgia Tech Smoothing and Mapping.** An open-source C++ library implementing factor graphs and iSAM2 (incremental Smoothing And Mapping). Meridian uses GTSAM 4.2 as the back-end engine (L3); all pose variables are GTSAM `Pose3` objects. See spec 05.

#### Huber
A robust loss function (M-estimator) that behaves like squared-loss for small residuals (convex) and linear for large residuals (outlier-down-weighting). Meridian's default robust kernel for GNSS and loop factors in iSAM2 (spec 05 §8.4), chosen because it is convex and safe inside the incremental Gauss-Newton solver.

#### iEKF (iterated Extended Kalman Filter)
A discrete-time nonlinear filter that iteratively refines a pose estimate at a reference time by linearising residuals. Meridian's retired test oracle; both it and the CT front-end satisfy `IFrontEnd` (spec 01 §8, now deprecated). See also **discrete-time**.

#### iSAM2
**Incremental Smoothing And Mapping version 2** (GTSAM). An algorithm that maintains a Bayes tree of the smoothing problem's square-root information matrix, re-eliminating only the cliques on the path to the root when new factors arrive, so incremental updates are faster than batch re-solve. Meridian's back-end (L3) uses iSAM2 to handle loop closures and GNSS factors as they arrive. See Chapters 2 and 4.

#### KD-tree (k-d tree)
A binary space-partition data structure for nearest-neighbour queries and range searches in $\mathbb{R}^k$. FAST-LIO uses an **ikd-Tree** (incremental k-d tree with lazy deletion and background rebalance). Meridian replaces it with an adaptive voxel hash (spec 06 §3) for O(1) region-clear and to avoid rebalancing pathologies during loop-closure rebuilds.

#### Keyframe
A selected pose snapshot from the continuous trajectory, emitted by the front-end when motion, time, or observability thresholds are crossed. The keyframe is the unit of graph topology and retained-cloud storage; loop closures are between keyframes, not between raw scans.

#### KeyframePacket
The sole boundary value L2 (front-end) hands to L3 (back-end). Packages a keyframe's `id`, `stamp`, estimated pose, a relative/absolute/IMU constraint with covariance, observability report, deskewed cloud (Shared-immutable), and optional image + colour extrinsic. The contract ensures exactly one factor per edge (no double-counting). See spec 01 §6.

#### LIVO, LIVO2
**LiDAR-Inertial-Visual Odometry.** FAST-LIVO2 is a tightly-coupled filter fusing LiDAR (point-to-plane), IMU (preintegration), and camera (sparse-direct photometric) in one sliding window. Meridian is a **CT LIVO+GNSS** system (also includes GNSS anchoring) with an iSAM2 back-end. See spec 04 intro.

#### LiDAR (LiDAR point, LiDAR scan)
Light Detection and Ranging: a rotating or scanning laser range sensor. A scan is a set of 3D points with per-point intensity, azimuth/elevation coordinates, and time offset. Meridian assumes a single LiDAR with per-point timestamps for correct deskew. See spec 01 §4.2.

#### Lie group (SE(3), SO(3), Lie algebra)
Mathematical structures for rotations and rigid transforms. $SO(3)$ = special orthogonal group of 3×3 rotation matrices; $SE(3)$ = special Euclidean group of 4×4 homogeneous rigid transforms (rotation + translation). Meridian uses Sophus for Lie-group operations and exponential maps. See spec 01 §3.1 and Chapter 3 §3.5.

#### LIO
**LiDAR-Inertial Odometry.** Tightly-coupled fusion of LiDAR and IMU only (no camera). FAST-LIO2 is the canonical reference. Meridian is a superset: LIO + camera + GNSS.

#### MAP (Maximum A-Posteriori)
The most likely parameter values under a Bayesian model: $\hat{X} = \arg\max_X P(X|Z)$. For Gaussian noise this reduces to nonlinear least squares: $\hat{X} = \arg\min_X \sum_i \|h_i(X) \ominus z_i\|^2_{\Sigma_i^{-1}}$. L3's iSAM2 solves the MAP pose estimate (spec 05 §1).

#### Marginalization
The process of eliminating low-priority variables from a graph by summing out their factors, leaving a marginal factor on the remaining variables. Meridian marginalizes transient velocity/bias variables after restart-fallback edges to keep the steady-state graph pose-only (spec 05 §11). See Chapter 2 §2.9.

#### Mahalanobis distance
The normalised distance $d^2 = r^\top \Sigma^{-1} r$ of a residual $r$ under covariance $\Sigma$. Used for statistical gating: under Gaussian noise, $d^2$ is distributed $\chi^2_n$ (n = residual dimension). Binding convention: spec 05 §3.2. See Chapter 3 §3.9.

#### Mesh (Marching Cubes)
A triangulated surface extracted from the TSDF by the Marching-Cubes algorithm. nvblox computes the mesh on the GPU; Meridian's L4 applies per-vertex colour from retained keyframe images. See spec 06 §5.

#### Odometry (odometry frame, odom)
Dead-reckoning from IMU + LiDAR fusion: smooth, drift-prone, and locally accurate over short horizons. The front-end publishes pose in the `odom` frame; the back-end owns the `map` frame and estimates the `map→odom` transform via loop closures and GNSS.

#### PCM
**Pairwise Consistent Measurement** set maximization. A batch consistency filter: given a set of candidate loop closures, reject those globally inconsistent with the odometry chain and the trusted loops, keeping the maximum mutually-consistent subset (a maximum-clique problem). Applied in L5 stage D (spec 07 §8) and as a front-gate before the back-end's robust kernels. See Chapter 5.

#### PSD (positive semi-definite)
A matrix $M$ is PSD if $x^\top M x \ge 0$ for all vectors $x$. Covariances must be PSD; a composed covariance may violate this due to numerical rounding and must be guarded by eigenvalue-clamping before inversion (spec 07 §8.2). See Chapter 3 §3.6.

#### Point-to-plane (point-to-plane residual)
A LiDAR registration residual measuring the signed distance of a deskewed point to the nearest surface plane: $r = \hat{n}^\top (R p_i + t - q_i)$ where $\hat{n}$ is the plane normal and $(q_i, \hat{n})$ is the correspondence. More stable than point-to-point for sparse/angled points. See spec 04 §2.2.

#### Pose (Pose type, Pose3)
A rigid body transform in $SE(3)$, consisting of a rotation (unit quaternion) and translation (3×1 vector). Meridian's `Pose` type stores $T_{target\_source}$ mapping points from source frame to target frame. GTSAM's `Pose3` uses the same convention. See spec 01 §3.1 and Chapter 3 §3.1.

#### RTK
**Real-Time Kinematic** GNSS positioning: centimetre-level accuracy using a local base station and real-time integer-ambiguity resolution. Meridian gates GNSS by `fix` type (RTK_Fixed >> SPP) so RTK fixes are weighted heavily relative to single-point positioning. See spec 01 §4.4, spec 05 §6.4.

#### Scan Context (Scan Context++)
A rotation-invariant, global loop-closure descriptor: a 2-D polar occupancy image (radial × azimuth bins, value = max height). Scan Context++ augments the polar context with a Cartesian context and a fast two-key KD-tree search plus coarse yaw/lateral offset recovery. Used in L5 stage A for high-recall candidate retrieval. See spec 07 §5, `Appendix R.1`, and Chapter 5.

#### SE(3), SO(3)
**SE(3)** = Special Euclidean group, 6-DoF rigid transforms (3 translation + 3 rotation). **SO(3)** = Special Orthogonal group, 3×3 rotation matrices. Both are Lie groups with tangent algebras for perturbation. See **Lie group** and Chapter 3 §3.5.

#### SLAM (Simultaneous Localization and Mapping)
The joint problem of estimating a robot's pose trajectory (`localization`) while building a map (`mapping`) from sensor observations. Meridian is a tightly-coupled, multi-sensor (LiDAR + IMU + camera + GNSS) SLAM system. See Chapter 1.

#### STD
**Stable Triangle Descriptor.** A local geometric descriptor encoding triangles of stable keypoints by their sorted edge lengths and inter-normal angles, invariant to rigid transforms. More discriminative than global descriptors; used in L5 stage B for re-ranking and 6-DoF initial-guess extraction. See spec 07 §6, `Appendix R.2`, and Chapter 5.

#### TLS (Total Least Squares)
An orthogonal regression method minimising perpendicular distance to a line/plane, symmetric in all variables (unlike ordinary least squares). Used in plane fitting when all coordinates are noisy. The PCA plane fit (spec 06 §3.3) is the TLS solution.

#### TSDF (Truncated Signed Distance Field)
A volumetric representation where each voxel stores a signed distance to the nearest surface, truncated at a threshold to bound per-voxel memory. Running-average fusion allows incremental updates; nvblox implements TSDF on the GPU. See spec 06 §4.

#### Twist
The spatial velocity of a body: linear velocity (m/s) and angular velocity (rad/s) in a chosen frame. Meridian's `Twist` type (spec 01 §3.2) is an optional companion to a pose for trajectory interpolation.

#### Umeyama
A closed-form algorithm for rigid-alignment (Procrustes) between two point sets by solving the orthogonal Procrustes problem via SVD. Used in GNSS datum alignment (spec 05 §6.2) and STD/BTC RANSAC (spec 07 §6.3) for 6-DoF pose recovery.

#### Voxel (voxel grid, voxel hash, voxel map)
A 3×3×3 cell in a spatial grid (volumetric pixel). Meridian's map uses an adaptive voxel hash (Tier R, spec 06 §3) for registration and GPU-resident TSDF voxels (Tier S, spec 06 §4) for surface fusion. See also **Tier R, Tier S**.

#### Voxel Hash
An O(1) spatial index using a flat hash table keyed by quantised integer coordinates (e.g., x//res, y//res, z//res). Meridian's registration map (Tier R) uses a voxel hash to replace FAST-LIO's ikd-Tree, enabling O(1) region clears for loop-closure rebuilds (spec 06 §3.1–§3.2).

---

### (B) How to Read the Code & Specs

#### Starting Points by Interest

**"I want to understand how pose estimation works."**
1. Read spec 00 §0 for the one-paragraph mental model.
2. Read spec 01 §2 (conventions: units, frames, time) and §3 (core math types: `Pose`, `NavState`, `Twist`).
3. Read spec 04 intro + §2 (the CT B-spline, LiDAR point-to-plane, IMU preintegration, visual photometry).
4. Code entry point: `meridian_frontend/include/meridian/frontend/ifrontend.hpp` and `src/ct/ct_frontend.cpp`.

**"I want to understand the back-end."**
1. Read spec 00 §6 (the `KeyframePacket` as the sole L2→L3 boundary).
2. Read spec 01 §6 (the `KeyframePacket` full contract and the no-double-counting rule).
3. Read spec 05 §1 (scope and responsibilities) and §2–§5 (variables, factors, odometry edges, restart edges).
4. Code entry point: `meridian_backend/include/meridian/backend/ibackend.hpp` and `src/isam2_backend.cpp`.

**"I want to understand loop closure."**
1. Read spec 00 §3 (layer responsibilities) and the description of L5 place recognition.
2. Read spec 07 §1–§4 (purpose, cascade overview, four-stage hierarchy).
3. Dive into stage A (Scan Context), B (STD/BTC), C (GICP), D (PCM consistency).
4. Code entry point: `meridian_place/include/meridian/place/iplace_recognizer.hpp` and `src/place_detector.cpp`.

**"I want to understand the map."**
1. Read spec 00 §3 (layer roles) and §9.5 (GPU / no CPU fallback).
2. Read spec 06 §1 (the two-tier mental model: Tier R registration hash, Tier S nvblox GPU TSDF, KeyframeStore).
3. Read spec 06 §6 (the retained cloud store — MUST-FIX #4).
4. Code entry point: `meridian_map/include/meridian/map/imaplayer.hpp` and `src/nvblox/nvblox_map.cpp`.

**"How does configuration work?"**
1. Read spec 00 §8 (one typed tree, YAML loader, ROS param loader).
2. See `meridian_config/include/meridian/config/config.hpp` for the full `Config` struct.
3. See `meridian_ros/config/*.yaml` for example YAML files (the ground truth).

**"How does telemetry / debug work?"**
1. Read spec 00 §10 (debug as a first-class subsystem).
2. See `meridian_debug/include/meridian/debug/telemetry.hpp` for the `TelemetrySink` interface.
3. See `meridian_ros/src/telemetry/ros_telemetry_sink.cpp` for how the wrapper converts core telemetry to ROS topics.

**"I want to understand time / synchronisation."**
1. Read spec 01 §2.1 (int64 nanoseconds) and §2.3 (estimation frame).
2. Read spec 02 (sensors and time sync) for per-sensor timestamp model and PTP / PPS discipline.
3. Code: `meridian_time/include/meridian/time/time_model.hpp`.

**"I want to understand the online calibration / extrinsic refinement."**
1. Read spec 01 §5 (calibration types and the `refine_online` flag).
2. Read spec 05 §10 (how L3 holds extrinsics as graph variables and publishes refined snapshots to L2).
3. Read spec 08 (full calibration spec, offline prior + online refinement).

---

#### Code Organisation Quick Map

| Concept | Where it lives |
|---------|---------------|
| Core types (Pose, NavState, KeyframePacket, GnssFix, …) | `meridian_common/include/meridian/common/` (header-only) |
| Configuration schema & YAML loader | `meridian_config/include/meridian/config/` + `src/` |
| Time model, PTP, clock discipline | `meridian_time/include/meridian/time/` |
| Calibration / extrinsic types | `meridian_calib/include/meridian/calib/` |
| Telemetry / debug bus | `meridian_debug/include/meridian/debug/telemetry.hpp` |
| Sensor abstraction (ISensorSource) | `meridian_sensors/include/meridian/sensors/` |
| Preprocessing (deskew, validity, downsample) | `meridian_preprocess/include/meridian/preprocess/` |
| **Front-end (CT LIVO+GNSS)** | `meridian_frontend/include/meridian/frontend/ifrontend.hpp` + `src/ct/` |
| **Back-end (iSAM2 graph)** | `meridian_backend/include/meridian/backend/ibackend.hpp` + `src/isam2_backend.cpp` |
| **Map (registration hash + nvblox TSDF)** | `meridian_map/include/meridian/map/imaplayer.hpp` + `src/nvblox/` |
| **Place recognition (loop closure cascade)** | `meridian_place/include/meridian/place/iplace_recognizer.hpp` + `src/` |
| Pipeline orchestration & threading | `meridian_pipeline/include/meridian/pipeline/pipeline.hpp` + `src/` |
| ROS wrapper & message converters | `meridian_ros/src/` |

---

### (C) Following One Keyframe End-to-End

A narrative walkthrough of a single keyframe as it flows through the system. **Assumptions:** the system is running in steady state (CT window is initialised, odometry is locking); a new LiDAR sweep arrives with IMU and optional camera data.

#### Stage 1: Sensor Input (L0)

1. **Raw LiDAR scan arrives** with `stamp_start`, `sweep_duration`, and per-point `t_offset_ns` (time relative to start). Example: Ouster `OS0_128` at 10 Hz produces ~20 million points over ~100 ms (spec 02 for rates).
   - **File:** `meridian_sensors/src/lidar_source.cpp` → `ISensorSource::onSample` callback.

2. **IMU samples** stream in at ~200 Hz, bracketing the scan interval. Each carries `acc`, `gyro`, and `stamp`.
   - **File:** `meridian_sensors/src/imu_source.cpp`.

3. **Synchronised measurement group** (`MeasureGroup`) is assembled by L0 timing logic: one scan + the IMU samples spanning its interval (spec 02 §8).
   - **File:** `meridian_sensors/include/meridian/sensors/measure_group.hpp`.

#### Stage 2: Preprocessing (L1)

4. **Raw scan is preprocessed:** points outside blind radius, invalid NaNs, low intensity, and voxel downsampling (spec 03). Result is a `LidarScan` with fewer points and metadata.
   - **File:** `meridian_preprocess/src/lidar_preprocessor.cpp` → `ILidarPreprocessor::process`.

5. **Deskew.** At **cold-start** (window not yet init): IMU-only forward-integrate gyro + accel to estimate gravity and biases; deskew the scan by dead-reckoning integration.
   - At **steady-state** (window running): deskew is **implicit** — the CT front-end will evaluate the spline at each point's true time, so no explicit deskew here (but the `IDeskewProvider` interface is prepared, spec 00 §7.3).
   - **File:** `meridian_preprocess/include/meridian/preprocess/ideskew_provider.hpp`.

6. **Image preprocessing** (if camera present): rectify, build Gaussian pyramid, compute edge map for photometric keypoint selection.
   - **File:** `meridian_preprocess/src/camera_preprocessor.cpp`.

7. **PreprocessedGroup** bundles the deskewed/downsampled cloud, pyramid image (if any), and IMU samples; pushed to the front-end queue (spec 02 §9).
   - **File:** `meridian_preprocess/include/meridian/preprocess/preprocessed_group.hpp`.

#### Stage 3: Front-End CT Windowing & Optimisation (L2)

8. **Front-end ingests the `PreprocessedGroup`** via `IFrontEnd::ingest()`.
   - **File:** `meridian_frontend/include/meridian/frontend/ifrontend.hpp::ingest`.

9. **Spline window setup.** Current B-spline knots are checked; new control points are added if the sweep time spans beyond the trailing edge (typically every 2–5 sweeps at 10 Hz with 25 ms knot spacing). Each control point represents an SE(3) pose to be optimised.
   - **File:** `meridian_frontend/src/ct/spline_window.cpp`.

10. **Residuals built.**
    - **Point-to-plane LiDAR:** each deskewed body-frame point `p_body` at time `t_i` is registered against the map. Query the map: `plane = map.query_plane(T(t_i)^{-1} * map.query_plane_point)` where $T(t_i)$ is the spline at the point's time. Residual: $r = \hat{n}^\top (T(t_i) p_{\text{body}} - q)$, where $(q, \hat{n})$ is the nearest plane (Tier R of the map, § 6 §3).
    - **IMU preintegration:** derivatives of the control-point poses w.r.t. the spline knots, integrated against IMU-rate expectations (no discrete EKF; instead, the relative transform between knots is a manifold Jacobian).
    - **Sparse photometric (camera):** FAST-LIVO2 style: warp high-gradient pixel locations from the camera image into a reference frame via the spline, compute photometric residuals (normalised SSD), and accumulate Jacobians.
    - **GNSS (optional, weak):** position fixes in the window's time span are loose anchors (large covariance by design, spec 04 §3.4); they do not strongly constrain the window but provide global-drift feedback.
    - **File:** `meridian_frontend/src/ct/residuals_*.cpp`.

11. **Nonlinear solve.** A Ceres solver (Levenberg-Marquardt) optimises the spline control points w.r.t. the residuals (point-to-plane, IMU, photometric, GNSS). Typical window size: 8 knots, ~500 ms of motion. Window marginalisation drops old knots (spec 04 §5.5) so the optimisation size stays bounded.
    - **File:** `meridian_frontend/src/ct/ct_frontend.cpp::optimize_window`.

12. **Observability check.** After the solve, the Hessian block-diagonal (per-axis, per-point-to-plane residuals) is analysed to compute an `ObservabilityReport` — six scores (tx, ty, tz, rx, ry, rz) quantifying per-axis sensitivity. Low scores in a corridor (e.g., tz weak) signal the back-end to inflate covariance along that axis (spec 05 §4.3).
    - **File:** `meridian_frontend/src/ct/observability.cpp`.

13. **Keyframe decision.** After the solve, check if this sweep warrants a keyframe: distance moved > `frontend.keyframe.dist_m` (e.g., 1 m), rotation > `frontend.keyframe.rot_deg` (e.g., 10°), or time > `frontend.keyframe.time_s` (e.g., 1 s). If yes, proceed to step 14; else, just publish a live `NavState` (step 33) and return.
    - **File:** `meridian_frontend/src/ct/ct_frontend.cpp::decide_keyframe`.

#### Stage 4: Keyframe Handoff to Back-End (L2→L3 Boundary)

14. **Keyframe is registered.** The front-end samples the optimised spline at the keyframe `stamp` to get $T_{\text{odom}\_\text{body}}$. The last keyframe from the previous window is $T_{\text{odom}\_\text{prev}}$. Compute the **relative pose** $T_{\text{prev}\_\text{this}} = T_{\text{prev}}^{-1} \cdot T_{\text{this}}$ and its **marginal covariance** (by Schur complement over the window-marginal; conservative: sum of endpoint marginals minus cross-cov, spec 01 §6.4).
    - **File:** `meridian_frontend/src/ct/keyframe_packer.cpp`.

15. **Build `KeyframePacket`.**
    - `id`: monotonic keyframe sequence number (uint64).
    - `stamp`: the keyframe's real sensor time (int64 ns).
    - `ref_frame = Odom`, `T_ref_body`: the sampled odom-frame pose.
    - `constraint_kind = RelativeBetween` (normal path, not restart).
    - `rel_to_id`, `T_relto_this`, `constraint_cov`: the relative transform + covariance (rotation-first GTSAM order).
    - `observability`: the report from step 12.
    - `cloud_body`: shared_ptr to the **deskewed body-frame cloud** (immutable, reusable).
    - `image`, `T_body_cam`: the camera frame + snapshot extrinsic for later colourisation.
    - `calib_version`: the version of the `CalibrationSet` used to produce this packet.
    - **File:** `meridian_frontend/src/ct/keyframe_packer.cpp`.

16. **Emit via the keyframe sink.** The front-end calls the sink callback (`set_keyframe_sink`) passing the packet by move semantics. The packet is pushed into the back-end's thread-safe bounded queue `Q_kf`.
    - **File:** `meridian_frontend/include/meridian/frontend/ifrontend.hpp::set_keyframe_sink`.

#### Stage 5: Back-End Graph Ingest & Optimisation (L3)

17. **Back-end consumes from `Q_kf`.** The pipeline's back-end thread pops the keyframe packet and calls `IBackEnd::add_keyframe(packet)`.
    - **File:** `meridian_backend/src/isam2_backend.cpp::add_keyframe`.

18. **Lift pose into map frame.** The packet's pose is in `odom` frame. The back-end knows the current `map←odom` transform (from previous loop corrections or identity at start); lift: $T_{\text{map}\_\text{body}} = T_{\text{map}\_\text{odom}} \cdot T_{\text{odom}\_\text{body}}$. Use this as the initial value for the new GTSAM variable `X(id)`.
    - **File:** `meridian_backend/src/isam2_backend.cpp::add_between_edge` step 1.

19. **Add the between-factor.** The packet's `constraint_cov` is inflated by observability (spec 05 §4.3): low tx-score → inflate tx variance, etc. Build a GTSAM `BetweenFactor<Pose3>(X(prev_id), X(id), T_relto_this, noise_model)` with that inflated covariance. This is the **only** odometry factor for this edge (no absolute prior, no IMU factor — the no-double-counting contract).
    - **File:** `meridian_backend/src/isam2_backend.cpp::add_between_edge` step 2.

20. **Bookkeeping.** Record the keyframe's `id`, `stamp`, cloud handle, observability, and odom-hint pose in an internal `kf_record` map. Forward the cloud and pose to the `KeyframeStore` (L4) and the place recogniser (L5).
    - **File:** `meridian_backend/src/isam2_backend.cpp::add_between_edge` step 3.

#### Stage 6: Map Integration (L4)

21. **Map layer ingest.** The pipeline forwards the `KeyframePacket` (now with `T_map_body` filled by L3's lifting) to `IMapLayer::integrate(packet)`.
    - **File:** `meridian_map/include/meridian/map/imaplayer.hpp::integrate`.

22. **Tier R (registration voxel hash).** Deskewed body-frame cloud is transformed by `T_map_body` into map frame. Each point is hashed to a voxel coordinate; the voxel accumulates representative points (bounded buffer, reservoir sampling for eviction). Plane fits are recomputed lazily when the voxel is queried. The `(id → [voxelkeys])` mapping is recorded for later loop-closure rebuilds (step 31).
    - **File:** `meridian_map/src/nvblox/voxel_hash_map.cpp::insert_points`.

23. **Tier S (nvblox GPU TSDF+RGB).** The map-frame cloud is sent to nvblox; nvblox integrates points into its GPU TSDF grid (running-average fusion, spec 06 §4.1), updating the signed-distance values in the affected blocks. If the keyframe has an image, the RGB is projected onto the surface and fused per voxel (Appendix R of spec 06).
    - **File:** `meridian_map/src/nvblox/nvblox_map.cpp::integrate`.

24. **KeyframeStore.** The `cloud_body` (Shared-immutable) is registered with `store.put(id, cloud_body, image, T_map_body)` — the canonical retained storage for loop-closure re-integration (spec 06 §6).
    - **File:** `meridian_map/src/keyframe_store.cpp::put`.

#### Stage 7: Place Recognition & Loop Closure (L5)

25. **Descriptor build.** The place recogniser's `add_keyframe(id, cloud, T_map_body)` is called (on the back-end thread, inline with L3).
    - Compose a **submap** from the last 5 keyframes ending at `id` (spec 07 §3.4), each transformed by the back-end's current corrected relative poses into keyframe `id`'s body frame.
    - Build a **Scan Context** matrix from the submap (radial occupancy histogram in polar coords, spec 07 §5.1). Compute ring key and sector key.
    - Insert into the KD-tree descriptor DB; BTC triangle descriptors are built lazily on demand (spec 07 §6.1–§6.2).
    - **File:** `meridian_place/src/place_detector.cpp::add_keyframe`.

26. **Loop detection** (periodically, e.g., every 5 keyframes).
    - **Stage A (Scan Context retrieval):** KD-tree nearest-neighbour search on the newest keyframe's ring key; return top-K candidates (e.g., K=5) with coarse yaw delta (spec 07 §5.4).
    - **Stage B (STD/BTC re-ranking):** for each Stage-A candidate, extract and match stable triangles; score by geometry + appearance; RANSAC the top 1–2 candidates to get a 6-DoF `T_guess` (spec 07 §6.3–§6.4).
    - **Stage C (GICP verification):** align the newest keyframe's cloud against each re-ranked candidate's submap using `small_gicp`, starting from `T_guess`. Compute fitness (overlap × exp(−rmse/σ)), overlap, condition number. Accept iff fitness ≥ 0.3, overlap ≥ 0.5, cond ≤ threshold (spec 07 §7.4–§7.5).
    - **Stage D (PCM batch consistency):** for each verified loop, test pairwise consistency with all trusted loops using the current corrected odometry chain. Run max-clique to select the largest mutually-consistent subset. Reject loops that fail the odometry self-consistency test or are inconsistent with the clique (spec 07 §8.2–§8.3).
    - **File:** `meridian_place/src/place_detector.cpp::detect`.

27. **Emit `LoopConstraint`.** For each accepted loop (say, `from_id=100, to_id=150`), build a `LoopConstraint` with:
    - `from_id`, `to_id`: the two keyframe ids.
    - `T_from_to`: the GICP-refined relative pose (maps `to`'s body into `from`'s body frame).
    - `cov`: the GICP Hessian, inflated by observability (spec 07 §9.2).
    - `fitness`: the GICP fitness (spec 07 §7.5).
    - **File:** `meridian_place/src/place_detector.cpp::build_loop_constraint`.

#### Stage 8: Back-End Loop Integration & Global Optimisation (L3)

28. **Back-end ingest loop.** For each `LoopConstraint`, call `IBackEnd::add_loop_constraint(lc)`.
    - Add a `BetweenFactor<Pose3>(X(from_id), X(to_id), T_from_to, robust_noise_model)` with a **Huber** robust kernel and optional GNC (spec 05 §8). The loop does NOT increase the pose-variable count (the keyframes are already in the graph).
    - **File:** `meridian_backend/src/isam2_backend.cpp::add_loop_constraint`.

29. **Back-end optimisation.** Call `IBackEnd::optimize()` (on a throttled cadence, e.g., every 5 keyframes or 1 s).
    - Run `iSAM2::update(new_factors, new_values)`. iSAM2 increments the Bayes tree, re-eliminating cliques on the path from the newest pose to the root. Relinearization is fluid: if a factor's linearisation point moved too far, relinearise its clique.
    - Return a `GraphUpdate` listing moved keyframes: `{(id_1, new_T_map_body_1), (id_2, new_T_map_body_2), ...}`.
    - **File:** `meridian_backend/src/isam2_backend.cpp::optimize`.

#### Stage 9: Map De-integration & Region Rebuild (L4)

30. **Back-end broadcasts `GraphUpdate`.** The pipeline receives the list of moved keyframes (typically a chain of 10–50 poses if the loop was large).
    - **File:** `meridian_backend/src/isam2_backend.cpp::optimize` (return value).

31. **Map rebuilds affected regions.** `IMapLayer::apply_graph_update(update, store)` is called:
    - For each moved keyframe `id`, look up the `(id → [voxelkeys])` mapping from step 22. Mark those voxels as dirty (invalidate plane caches in Tier R) or remove them entirely.
    - Query `KeyframeStore` for all clouds in the affected AABB; re-transform them by their *new* corrected poses; re-insert them into Tier R and nvblox (spec 06 §7.2–§7.3).
    - This is the **clear-and-rebuild** contract: the TSDF is non-invertible (running average), so we cannot subtract old contributions; instead, we rebuild from the canonical cloud store at corrected poses.
    - **File:** `meridian_map/src/nvblox/nvblox_map.cpp::apply_graph_update`.

32. **Submap cache invalidation** (L5). If the back-end's `GraphUpdate` moves any keyframe within a cached L5 submap's window, that submap is invalidated and will be recomposed on the next descriptor query (step 25).
    - **File:** `meridian_place/src/place_detector.cpp::on_graph_update`.

#### Stage 10: Live Output & Telemetry

33. **Live `NavState` publication.** Between keyframes (at every IMU sample or every scan), the front-end publishes a smooth `NavState` in the `odom` frame:
    - Sample the CT spline at the current time $t$.
    - Propagate through IMU-only integration if the time is beyond the spline's trailing knot (motion prediction).
    - Emit via `live_state()` (L2 boundary, spec 01 §7.3).
    - The pipeline converts this to `nav_msgs/Odometry` and publishes `/odom`.
    - **File:** `meridian_frontend/src/ct/ct_frontend.cpp::live_state`.

34. **Correction feedback.** When the back-end's `optimize()` returns, L3 computes the `map←odom` transform (the difference between the corrected poses and the live odometry) and publishes it as a TF frame. L2 picks this up on its next loop iteration and re-anchors its odom frame (spec 01 §7.3, spec 05 §9.3).
    - **File:** `meridian_backend/src/isam2_backend.cpp::corrected_trajectory`.

35. **Telemetry.** Every module emits rich debug data on the `TelemetrySink`:
    - Front-end: per-axis observability scores, sparse-direct photometric residual count, spline knot poses, window restart events.
    - Back-end: chi-square (consistency check), number of factors, relinearization events, loop-closure accept/reject reasons.
    - Map: integration timing, voxel counts, region-rebuild events.
    - Place recognizer: per-stage timing and scores (SC distance, STD inlier count, GICP fitness, PCM consistency).
    - The wrapper (`meridian_ros`) subscribes to these sinks and publishes to ROS topics (PointCloud2, TF, custom Meridian message types) and writes to the debug bus.
    - **File:** `meridian_debug/include/meridian/debug/telemetry.hpp`.

36. **Mesh extraction.** On a throttled thread (T5, not the critical path), nvblox marches the dirty TSDF blocks and extracts triangles (spec 06 §5). Per-vertex colour is assigned from the retained keyframe images (via `T_body_cam`). The mesh is published as a `visualization_msgs/Marker` or written to a file.
    - **File:** `meridian_map/src/nvblox/nvblox_map.cpp::extract_mesh`.

---

#### Summary Diagram: One Keyframe's Journey

```
┌─────────────────────────────────────────────────────────────────────────┐
│  L0: Raw sensor                                                         │
│   (LiDAR scan + IMU stream + camera frame + GNSS fix)                  │
└────────────────┬────────────────────────────────────────────────────────┘
                 │ (steps 1–3)
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  L1: Preprocessing                                                       │
│   (deskew, downsample, validity, pyramid build)                         │
│  → PreprocessedGroup                                                    │
└────────────────┬────────────────────────────────────────────────────────┘
                 │ (step 4)
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  L2: Front-end (CT LIVO+GNSS)                                           │
│   (spline window, residual build, nonlinear solve)                      │
│   [Keyframe decision: should we emit?] ──no──→ emit live NavState (33) │
│                         │                                               │
│                        yes                                              │
│                         ▼                                               │
│   compute relative pose + covariance + observability                    │
│   → KeyframePacket (id, T_relto_this, cov, cloud_body, image, …)      │
└────────────────┬────────────────────────────────────────────────────────┘
                 │ (step 5)
          [Q_kf queue]
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  L3: Back-end (iSAM2)                                                   │
│   (lift to map frame, add between-factor, integrate loop closures)      │
│   [Optimisation cadence] ──trigger──→ iSAM2::update()                 │
│                                   → GraphUpdate(moved keyframes)        │
└────────────────┬────────────────────────────────────────────────────────┘
                 │ (steps 6–7)
                 ├──────────────────────────┬────────────────────────┐
                 │                          │                        │
                 ▼                          ▼                        ▼
        ┌─────────────────┐      ┌──────────────────┐    ┌──────────────────┐
        │ L4: Map         │      │ L5: Place        │    │ L6: Output &     │
        │ (Tier R hash +  │      │ Recognition      │    │ Telemetry (33)   │
        │  Tier S nvblox) │      │ (add descriptor) │    │                  │
        │                 │      │                  │    │ publish /odom,   │
        │ (step 8)        │      │ [detect cadence] │    │ /tf, /mesh,      │
        │ insert cloud    │      │    ↓             │    │ debug topics      │
        │                 │      │ detect loop      │    │                  │
        │ [correction]    │      │ → LoopConstraint │    │ (emit live       │
        │    ↓            │      │ → back to L3     │    │  NavState & map  │
        │ apply_graph_    │      │                  │    │  poses)          │
        │ update (step 9) │      │ (steps 10–12)    │    └──────────────────┘
        │ rebuild from    │      └──────────────────┘
        │ KeyframeStore   │
        │ at corrected    │
        │ poses           │
        └─────────────────┘

Time axis →
```

---

### (D) Key File Locations for Each Component

| Component | Primary Header | Primary Implementation | Test |
|-----------|---|---|---|
| Core types | `meridian_common/include/meridian/common/pose.hpp`, `keyframe_packet.hpp` | (header-only) | `meridian_common/test/` |
| Config | `meridian_config/include/meridian/config/config.hpp` | `meridian_config/src/config.cpp` | `meridian_config/test/` |
| Time sync | `meridian_time/include/meridian/time/time_model.hpp` | `meridian_time/src/` | `meridian_time/test/` |
| Sensors | `meridian_sensors/include/meridian/sensors/isensor_source.hpp` | `meridian_sensors/src/lidar_source.cpp`, etc. | `meridian_sensors/test/` |
| Preprocessing | `meridian_preprocess/include/meridian/preprocess/ideskew_provider.hpp` | `meridian_preprocess/src/lidar_preprocessor.cpp` | `meridian_preprocess/test/` |
| **Front-end (CT)** | `meridian_frontend/include/meridian/frontend/ifrontend.hpp` | `meridian_frontend/src/ct/ct_frontend.cpp` | `meridian_frontend/test/ct/` |
| **Back-end (iSAM2)** | `meridian_backend/include/meridian/backend/ibackend.hpp` | `meridian_backend/src/isam2_backend.cpp` | `meridian_backend/test/` |
| **Map (Tier R+S)** | `meridian_map/include/meridian/map/imaplayer.hpp` | `meridian_map/src/nvblox/nvblox_map.cpp` | `meridian_map/test/` |
| **Place (loop)** | `meridian_place/include/meridian/place/iplace_recognizer.hpp` | `meridian_place/src/place_detector.cpp` | `meridian_place/test/` |
| Pipeline | `meridian_pipeline/include/meridian/pipeline/pipeline.hpp` | `meridian_pipeline/src/pipeline.cpp` | `meridian_pipeline/test/` |
| ROS wrapper | (N/A) | `meridian_ros/src/odometry_node.cpp`, `mapping_node.cpp` | `meridian_ros/test/` |
| Telemetry | `meridian_debug/include/meridian/debug/telemetry.hpp` | `meridian_debug/src/null_sink.cpp`, `meridian_ros/src/telemetry/ros_telemetry_sink.cpp` | (N/A) |

---

### (E) Important Boundary Contracts (Quick Reference)

| Boundary | Spec | One Thing It Passes |
|----------|------|---------------------|
| L0→L1 | 01 §4.1–§4.4 | `ImuSample`, `LidarScan`, `CameraFrame`, `GnssFix` (value types) |
| L1→L2 | spec 02 §9 | `MeasureGroup` / `PreprocessedGroup` (bundled measurements) |
| L2→L3 | 01 §6 | **`KeyframePacket`** (the sole currency; no others cross here) |
| L3→L4 | 01 §7.4–§7.5 | `KeyframePacket` (for cloud store) + `GraphUpdate` (for de-integration) |
| L3→L5 | 01 §7.6 | `KeyframePacket` (for descriptor) + `LoopConstraint` (returned) |
| L2 ↔ L3 feedback | spec 05 §9.3 | refined `CalibrationSet` snapshot (versioned) |
| L4 to L2 | spec 06 §3.5 | `PlaneHit` (nearest plane query, front-end reads from map) |
| (All) → debug | spec 00 §10, 10.1 | `TelemetrySink` (timing, scalars, clouds, poses, markers, events) |

---

This concludes the course. Chapter 6 is your reference desk: the glossary defines 60+ terms, the "How to Read" guide routes you to entry points by interest, the code-to-spec cross-reference shows where each component lives, the end-to-end narrative ties Chapters 1–5 together by following one keyframe through the whole system, and the quick-reference tables pin down file locations and boundary contracts.
