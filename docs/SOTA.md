# Real‑Time Large‑Scale Multi‑Sensor 3D SLAM — State of the Art (2026)

> Prepared 2026‑05‑28. This document is the SOTA reference; the system we're designing is in `NEXT_GEN_DESIGN.md`. The 2020‑era baseline (LIO‑SAM / your arc‑slam) is the comparison anchor throughout.

Scope: large‑scale (kilometres, hours), real‑time (≥ scan rate), multi‑sensor (LiDAR + IMU + camera + GNSS, optionally radar/UWB), robust to degeneracy and sensor dropout. Pure visual SLAM and pure mono‑LiDAR systems are out of scope.

---

## 1. The seven shifts since LIO‑SAM (2020 → 2026)

Each is a structural change in how the community builds systems, not just a faster algorithm. Everything that follows in this document expands one of these.

1. **Direct registration replaced feature extraction for the high‑rate front‑end.** FAST‑LIO2 (2022) showed that registering raw points to an ikd‑Tree is faster *and* more accurate than computing LOAM smoothness, edge points, and surface points. Every serious 2024–2026 LIO is direct.
2. **The local map became a voxel‑hashed structure** (ikd‑Tree, iVox, Voxel‑SLAM hash, AVK‑SLAM density‑adaptive voxel). KdTree‑per‑keyframe submaps — your current arc‑slam mechanism — are a 2020 artifact.
3. **Tight visual–LiDAR–inertial fusion is the new default**, not a bolt‑on. FAST‑LIVO2 (T‑RO 2024), Coco‑LIC (RA‑L 2023), Gaussian‑LIC (ICRA 2025), GS‑LIVO (RA‑L 2025) all fuse all three at the front‑end, not just feed visual loop closures into a LiDAR back‑end.
4. **Continuous‑time trajectories** (B‑spline, Gaussian Process) graduated from research to production with Wildcat (CSIRO, deployed underground 2022–24) and Coco‑LIC. The motivation isn't "look, splines" — it's that LiDAR, IMU, and camera measurements occur at fundamentally different and asynchronous timestamps, and discrete‑time pose graphs paper over that with deskewing approximations.
5. **Map representation diversified into a layered stack**: a fast geometric layer for registration (voxel hash), a dense geometric layer for planning (TSDF/ESDF), an optional photometric layer for rendering (3DGS or per‑voxel RGB), and an emerging semantic layer (INF‑SLiM 2026, PIN‑SLAM 2024). No 2026 system uses a single representation for everything.
6. **Place recognition replaced geometric loop detection.** Scan Context (2018), Scan Context++ (2021), STD (2023), BTC (2024), learned descriptors (MinkLoc3D, Logg3dNet, BoW3D) all do what your `detectLoopClosureDistance` cannot: recognise a previously visited place after large drift, in seconds, without an initial guess.
7. **Robustness became a first‑class concept**, not a side check. Per‑axis degeneracy (X‑ICP 2023, D²‑LIO 2025), switchable constraints (Sünderhauf 2012, now standard), Graduated Non‑Convexity (Yang T‑RO 2020), sensor health monitoring (INAF Fusion 2025) — modern systems instrument every estimator step with a confidence channel that flows into the back‑end.

---

## 2. State estimation: front‑end families

### 2.1 The three estimator paradigms

| Paradigm | Representative | Latency / freq | Global consistency | Loop closure | When to use |
|---|---|---|---|---|---|
| **Iterated EKF on manifold** | FAST‑LIO2, Point‑LIO, D²‑LIO | < 5 ms / 100‑500 Hz | None | External only | High‑rate odometry, embedded platforms |
| **Factor graph + iSAM2** | LIO‑SAM, arc‑slam, LVI‑SAM | 20‑100 ms / 5‑10 Hz | Yes | Native | Mapping, multi‑session, GPS fusion |
| **Continuous‑time sliding window + factor graph back‑end** | Wildcat, Coco‑LIC | 30‑80 ms / 5‑20 Hz | Yes | Native | Asynchronous heterogeneous rigs, fast motion |

The 2026 trend is the third: **CT local window with a discrete keyframe back‑end**. You get IEKF‑class robustness to fast motion AND graph‑class global consistency. The two pioneers (Wildcat, Coco‑LIC) converged on this architecture independently.

### 2.2 Iterated EKF (iEKF) front‑ends

**The pattern** (FAST‑LIO 2021, T‑RO):
- State: 15‑DOF position/velocity/attitude/IMU bias on $SO(3) \times \mathbb{R}^{12}$.
- IMU propagates the state at gyro rate.
- LiDAR scan registers to the local map via Newton iteration on the manifold; the iterated Kalman update is mathematically equivalent to Gauss‑Newton on the maximum a posteriori problem (Bell & Cathey 1993). This is the trick that makes iEKF competitive with non‑linear optimization.

**FAST‑LIO2 (2022)**: removes feature extraction entirely. Every LiDAR point registers directly to the map via point‑to‑plane residual. The map is an **ikd‑Tree** (incremental k‑d tree, Cai et al. 2021 RA‑L) — supports point insertion in $O(\log n)$ amortized via lazy rebalancing, and box deletion for sliding the active region. This is the structural insight that made direct registration tractable.

**Point‑LIO (2023, AISY)**: each LiDAR point is its own EKF update at its true sampling timestamp, instead of accumulating a 100 ms scan. Effectively a discrete approximation of continuous‑time fusion. Removes scan deskewing entirely. Handles motion up to ~30 m/s where FAST‑LIO2 fails. Cost: more update overhead per second.

**D²‑LIO (Aug 2025)**: dual‑degeneracy‑aware iEKF. Tracks two degeneracy directions independently (translation along sensor axis, rotation around scene axis) and feeds per‑axis observability into the noise model rather than the binary "degenerate/not" switch that LIO‑SAM and your arc‑slam use. On the new degradation‑robust benchmark (arXiv 2507.20516, 2025), beats Point‑LIO by ~30 % ATE in symmetry‑heavy scenes (warehouse corridors, parking structures).

**Limits of pure iEKF**: no loop closure (the linearization horizon is one scan), so they drift over kilometres. Standard fix is to publish odometry and let a separate graph back‑end consume it — this is the "FAST‑LIO‑SAM" pattern that several 2023–24 repos implement and that your next system should adopt.

### 2.3 Direct vs feature‑based registration

Direct point‑to‑plane residual (FAST‑LIO2 style):

$$r_i \;=\; n_i^T (R p_i + t - q_i)$$

where $q_i$ is the closest map point and $n_i$ is the local plane normal from a kNN PCA. No edge/plane classification, no smoothness sort, no per‑sector picking. The argument is: LOAM features were designed for a world where matching was expensive; with ikd‑Tree it isn't. Skip the abstraction and match everything.

**Feature‑based** (LOAM / LIO‑SAM / your code) still wins when:
- The sensor is intrinsically sparse (4‑line lidars, MEMS lidars with bad ranging on smooth surfaces).
- The downstream consumer needs labelled features (e.g., feature‑level loop closure).
- Compute is tightly constrained and you can't afford full‑cloud kNN.

**Direct** wins when:
- Plenty of points (32‑128 line spinning lidars).
- Need robustness in feature‑poor scenes (snow, sand, water, tunnels).
- The map data structure supports fast neighborhood queries (ikd‑Tree, iVox).

Modern verdict: **direct for the front‑end, optional feature layer for back‑end recognition only**. Multi‑lidar rigs (your roadmap) make the direct case even stronger because you have so many points that picking 200 edge features per scan throws information away.

### 2.4 Continuous‑time fusion

**Why CT matters concretely**:

- A Velodyne VLP‑128 fires one beam every ~2 μs. An IMU samples at 200‑500 Hz, every 2‑5 ms. A camera shutters at 30 Hz, exposure ~5 ms. These are three different clocks with three different jitters. Discrete pose graphs assume each measurement happens at one canonical timestamp per keyframe — wrong.
- A spinning LiDAR + a vehicle at 20 m/s traverses 2 m per scan. Your translation‑deskew‑disabled code falls off a cliff here. A CT trajectory absorbs this without an extra deskewing step — the trajectory *is* the deskew.
- Asynchronous measurements (e.g., GPS arrives at 10 Hz but is fused into the graph at keyframe rate) get the linearization wrong on the gap. CT closes the gap by evaluating the trajectory at the actual measurement timestamp.

**The B‑spline approach** (CLINS, Coco‑LIC, Wildcat):
- Cubic or quintic B‑spline control points $\{c_k\}$ define the SE(3) trajectory $T(t)$.
- Any measurement at time $t$ contributes a factor that depends on the few control points whose support overlaps $t$ — typically 4 (cubic) or 6 (quintic).
- The factor graph is over control points, not poses. Loop closure factors interpolate the trajectory at the closure times.

**Non‑uniform B‑splines** (Coco‑LIC 2023, RA‑L): place control points denser where motion is fast or where measurements are dense. Adapts spline complexity to the data. Crucial for vehicles that alternate stop‑and‑go.

**Gaussian Process trajectories** (Anderson, Barfoot et al.): replace the spline with a GP prior on the trajectory. Theoretically more elegant (any inter‑sample query has a closed‑form mean+covariance) but heavier than splines. Used in research, rare in deployed systems.

**Wildcat (CSIRO, 2022‑24)**: production CT SLAM for the DARPA SubT challenges. Sliding window CT front‑end fuses LiDAR + IMU + (optionally) camera. Window slides at ~1 Hz; each window discharges a single keyframe pose into a discrete back‑end with iSAM2‑class optimization. Currently the reference architecture for fast, robust, async multi‑sensor SLAM.

### 2.5 Multi‑LiDAR fusion

Your planned rig (multiple Ousters + dome) puts you in territory where there are roughly three patterns:

**Common‑frame fusion** (M‑LOAM, Jiao et al., 2022): each LiDAR's scan is deskewed with the shared IMU, then transformed into a common base frame, then concatenated. One big cloud, one front‑end. Simple. Requires good extrinsic calibration; sensitive to small calibration errors when the LiDARs see overlapping regions.

**Per‑sensor front‑end, joint back‑end** (some 2024 papers, no canonical reference): each LiDAR runs its own registration against its own local map (or a shared one with view‑dependent association). The poses share IMU and bias factors at the back‑end. Better for handling per‑sensor failures (one Ouster fogs; the others carry the system).

**Hierarchical fusion** (Wildcat, multi‑lidar variants): merge into a common point pool but tag each point with its sensor. The CT trajectory is the same; the residual covariance is per‑sensor. Best of both — single‑pose state, per‑sensor robustness.

For a dome Ouster (upward) + horizontal Ousters, the dome contributes mostly to roll/pitch (it sees the sky and overhangs) while horizontals contribute to yaw and translation. Joint observability is excellent — practically immune to corridor‑axis degeneracy. M‑LOAM evaluations on multi‑LiDAR vehicles show 2–4× ATE improvement over single‑LiDAR baselines.

### 2.6 Visual fusion (LIVO and beyond)

**The case for the camera**: in geometrically degenerate scenes (corridors, tunnels, snow, sand) LiDAR loses translation observability. A camera contributes direct photometric constraints that are completely independent of geometry. The combined system is robust to single‑modality failure.

**The case against the camera**: calibration cost (millimetre‑level extrinsics, sub‑millisecond timestamp alignment), photometric calibration (auto‑exposure plays havoc with direct VO), and a whole new failure mode (darkness, glare).

**Three integration patterns**:

1. **Loosely coupled**: visual VO/SLAM runs independently, publishes pose; LiDAR back‑end consumes as a factor. Simple, robust to per‑system failure. Loses information at the boundary. LVI‑SAM (2021, IROS) is the canonical example.
2. **Tightly coupled, factor graph**: visual reprojection/photometric factors enter the same iSAM2 graph as LiDAR factors. R3LIVE (2021), LVI‑SAM. Best accuracy on synchronized rigs.
3. **Tightly coupled, iEKF**: visual residuals fold into the same iEKF update step as LiDAR. FAST‑LIVO (2022), **FAST‑LIVO2 (T‑RO 2024)**. Currently the speed/accuracy frontier; FAST‑LIVO2 runs at ~50 Hz on a desktop CPU with a full LiDAR + global shutter camera + IMU stack.

**FAST‑LIVO2** specifically: sparse direct visual residuals (photometric, no descriptors), patches around LiDAR projections (so it never needs to do depth from vision — the LiDAR provides it). Resilient to feature‑poor visual scenes because it doesn't extract features. The reference for camera fusion in 2025–26.

**Coco‑LIC** (RA‑L 2023): continuous‑time tight LIVO. Non‑uniform B‑spline. Adds a camera reprojection factor at each visual frame time, evaluated against the spline. Handles rolling‑shutter cameras gracefully.

### 2.7 GNSS / GPS fusion

The arc‑slam pattern (`addGPSFactor`) is sound and still represents 80 % of deployed practice. SOTA upgrades:

- **Switchable constraints** on the GPS factor (Sünderhauf 2012) so multipath / spoofing can be optimized away by a latent switch variable.
- **GNSS RTK / PPP** instead of single‑point fixes: centimetre accuracy, much tighter covariances. Cost: a base station or correction stream.
- **Anti‑spoofing checks**: GPS velocity vs IMU‑derived velocity disagreement → drop factor. Critical for tactical use where spoofing is a real adversary capability.
- **Tightly coupled raw GNSS** (pseudoranges, carrier phase): bring measurements pre‑PVT into the factor graph. Robust to bad fixes; gives you fault‑detection via measurement‑level residuals. See GVINS (Cao et al. 2022, T‑RO).

---

## 3. Back‑end optimization

### 3.1 iSAM2 — still the right answer, with caveats

iSAM2 (Kaess et al. 2012) has not been displaced. The Bayes tree + variable‑local relinearization with `relinearizeThreshold` is still the most efficient way to do incremental MAP inference on a SLAM graph.

What has changed since 2012:
- **GTSAM has gained CT factors** (B‑spline trajectory factors), making CT + iSAM2 a single‑library solution.
- **GPU‑accelerated Schur complement** (some 2024 work) for very dense graphs — relevant for 1000+ keyframe sessions.
- **SymForce (Skydio, open‑sourced 2022)**: symbolic differentiation of factors generates Jacobians as straight‑line C++ code, 5–10× faster than GTSAM's runtime expressions. Not a replacement for iSAM2, but a code‑generation layer on top.
- **GTSAM hybrid factor graphs (≥ 4.2)**: mix continuous and discrete variables in one graph. Lets you model loop closure inlier/outlier as a discrete switch directly, without the soft Sünderhauf approximation.

The honest answer to "is iSAM2 the best for robust global optimization": **yes, for any system that needs incremental loop closure**. The alternatives are:
- Sliding‑window bundle adjustment (VINS‑Mono): local only, you'd lose your kilometre‑scale consistency.
- Pose graph optimization (g2o, Ceres): batch or fixed‑lag; iSAM2 dominates incrementally.
- Distributed factor graph (DSGO, DC2SLAM): for multi‑robot, not single platform.

### 3.2 Robustness in the back‑end

Three layers of defence against bad factors:

**Layer 1: Robust kernels.** Wrap each suspect factor (loop closure, GPS, late visual) in a robust noise model. Huber is the safe default; Geman‑McClure is more aggressive; **Graduated Non‑Convexity** (GNC, Yang et al. T‑RO 2020) is the 2020s replacement that re‑runs the optimization with a slowly tightening kernel — provably converges to the global optimum under bounded outlier ratios.

**Layer 2: Switchable constraints.** Each suspect factor gets a latent switch variable $s_i \in [0,1]$ that scales its information matrix. The optimizer turns off bad factors automatically by driving $s_i \to 0$. Sünderhauf & Protzel 2012; GTSAM has a contrib implementation. With hybrid factor graphs in GTSAM ≥ 4.2 you can make $s_i \in \{0,1\}$ discrete, which is cleaner.

**Layer 3: Pre‑optimization outlier rejection.** PCM (Pairwise Consistency Maximization, Mangelson 2018) checks that loop closures are mutually consistent before adding them — a clique‑finding problem on the loop graph. Reject any factor not in the maximum consistent clique.

A 2026 production system uses all three: GNC kernel on individual factors, switchable constraints on classes of suspect factors (loops, late GPS), PCM as a pre‑filter on loop closure batches.

### 3.3 Degeneracy as a first‑class concept

Your code does Zhang 2016: eigendecompose the Hessian, project out under‑constrained directions. This is a 2016 idea. The 2023–2025 successors treat degeneracy as a per‑axis observability score that propagates into the back‑end noise model.

**X‑ICP** (Tuna et al., T‑RO 2023): instead of projecting away the bad direction, compute six per‑axis localizability scores (one per DOF), publish them with every odometry message. The back‑end inflates the corresponding diagonal entries of the BetweenFactor noise — so a corridor odometry edge gets a huge sigma on the along‑axis translation, and iSAM2 weights it down naturally. No discrete switch, no information loss.

**D²‑LIO** (2025): generalises X‑ICP by also detecting rotational degeneracy (e.g., a long flat ceiling that loses yaw observability). Two degeneracy modes tracked independently.

**Multi‑modal redundancy**: when one sensor degrades, others must carry the system. The principled way is: each sensor's contribution to the joint Hessian is observable. If LiDAR contributes nothing to the $z$ translation axis but the camera does, the camera carries that axis. With per‑sensor per‑axis observability tracked through the front‑end, the system gracefully degrades modality‑by‑modality. This is what makes a tactical system robust to smoke, dust, fog, darkness, GPS denial — each removes one sensor; the others observe the missing axes.

---

## 4. Map representations

### 4.1 The layered‑map principle

No 2026 system uses a single representation for everything. The compute and storage profile of a registration map (must support fast nearest‑neighbour queries) differs from a planning map (must support distance queries) differs from a rendering map (must encode appearance) differs from a recognition map (must encode semantics or descriptors).

The pattern is a **stack of registered layers**, all keyed off the same keyframe poses from the back‑end:

| Layer | Purpose | Typical structure |
|---|---|---|
| Registration | Sub‑scan front‑end matching | Voxel hash (ikd‑Tree, iVox) |
| Planning | Path planning, collision | TSDF + derived ESDF |
| Photometric | Visualization, photometric loop closure | Per‑voxel RGB or 3DGS |
| Semantic | Object detection, semantic loop closure | Per‑voxel class label |

Loop closure rewrites the poses; each layer re‑integrates its affected keyframes. Voxel‑hashed layers handle this trivially (clear and re‑integrate touched voxels). 3DGS does not handle it well — current systems re‑optimize the Gaussians in the affected window, which is expensive.

### 4.2 Voxel‑hashed structures (registration + planning)

**ikd‑Tree** (Cai et al., RA‑L 2021): incremental k‑d tree with lazy rebalancing. Insertion $O(\log n)$ amortized. Used by FAST‑LIO2, Point‑LIO, iG‑LIO. Best for kNN queries; the global tree handles bounded regions of interest via box deletion.

**iVox** (Bai et al., 2022): incremental voxel structure with intra‑voxel point lists. $O(1)$ insertion. Used by Faster‑LIO. Slightly less accurate kNN than ikd‑Tree but ~3× faster.

**Voxel hash** (Voxel‑SLAM 2024, AVK‑SLAM 2026): hash table over $(i,j,k)$ voxel indices, each cell storing per‑voxel statistics (mean, covariance, normal, intensity). Used for both registration *and* serves as a base for TSDF. AVK‑SLAM (2026) adds **density‑adaptive voxel size**: high‑density regions sub‑divide, sparse regions coarsen.

**OpenVDB / NanoVDB** (NVIDIA): hierarchical sparse voxel grid originally from VFX. Two to three levels of internal nodes, with hash‑table leaves. Used in NVBlox (NVIDIA, 2023) and the Foundry/Cesium ecosystems. Best when the map is much bigger than the active region (kilometre‑scale outdoor mapping).

The 2026 trend is **VDB‑class hierarchical sparse grids for the global map, flat voxel hashes for the active local region**. Lets you stream parts of the global map to disk and back without disturbing the front‑end's local data structure.

### 4.3 TSDF + ESDF for planning

**TSDF** (Truncated Signed Distance Field): each voxel stores a signed distance to the nearest surface (truncated to ± μ) and a weight. Surfaces are extracted by Marching Cubes at the zero crossing. Originally from real‑time RGB‑D fusion (KinectFusion 2011).

**Voxblox** (Oleynikova et al., 2017, IROS): the canonical CPU TSDF for robotics. ROS 1, integrates with ROS planning stacks. Has a colored variant that integrates RGB from camera projection.

**NVBlox** (NVIDIA, 2023): GPU‑accelerated TSDF on NanoVDB. ROS 2. ~10‑20× faster than Voxblox on a modest GPU. Cost: CUDA dependency.

**Voxfield** (2022): extension with per‑voxel surface normals. Cleaner meshes; faster Marching Cubes.

**ESDF** (Euclidean Signed Distance Field): for every voxel in free space, the distance to the nearest occupied voxel. Computed from the TSDF by Brushfire/wavefront propagation. The single most useful structure for planning — A*/RRT/MPC all benefit from O(1) clearance queries.

Voxblox and NVBlox both maintain ESDF incrementally as the TSDF updates. This is the production planning map.

### 4.4 Implicit neural maps

The 2023–2026 trajectory:

**NeRF‑SLAM family** (NICE‑SLAM 2022, Co‑SLAM 2023): visual SLAM with neural radiance field map. Mostly RGB‑D, not LiDAR. Demonstrated photoreal rendering but not real‑time at large scale.

**NeRF‑LOAM** (Deng et al., ICCV 2023): first attempt at neural implicit for LiDAR. Octree of latent features decoded to SDF by an MLP. Works but limited to ~100 m scenes.

**PIN‑SLAM** (Pan et al., T‑RO 2024): **point‑based implicit neural** SDF. Each point carries a small latent feature; SDF queried by aggregating nearest features. Scales to kilometre maps. Real‑time at LiDAR rate on a modest GPU. Currently the best unified geometry+semantics representation for LiDAR SLAM.

**Hi‑LOAM** (2026): hierarchical neural implicit fields. Multi‑resolution latent grid; self‑supervised. Better scaling than NeRF‑LOAM; reasonable real‑time on a workstation GPU.

**INF‑SLiM** (Zhang et al., J. Field Robotics 2026): adds **semantic labels** to the implicit field. Same MLP decodes both SDF and class probabilities from the latent. Octree + latent features + radial window self‑attention for semantic prediction. Currently the highest‑expressiveness LiDAR map (geometry + appearance + semantics from one structure).

**XGrid‑Mapping** (Dec 2025): hybrid **explicit + implicit** grid. Explicit voxels carry coarse geometry for fast queries; implicit MLP refines local detail. Pragmatic compromise: 80 % of the neural fidelity at 20 % of the cost.

When to use implicit neural maps: when you need a unified representation that gives geometry, appearance, and semantics together, and you can afford a GPU at runtime. When *not* to: when the downstream consumer is a planner that just needs an ESDF (then a TSDF is simpler and faster) or a tactical visualizer that just wants a coloured mesh (TSDF wins again).

### 4.5 3D Gaussian Splatting

**3DGS** (Kerbl et al., SIGGRAPH 2023): each point is a 3D anisotropic Gaussian with view‑dependent SH colour coefficients. Rasterized to image plane via differentiable splatting. Trained by photometric loss.

For SLAM:

- **Gaussian‑LIC** (ICRA 2025): LiDAR‑Inertial‑Camera with Gaussian map. CT trajectory feeds both registration and Gaussian densification.
- **GS‑LIVO** (RA‑L 2025): the embedded‑target reference. Hash‑indexed octree of Gaussians with a sliding window. IESKF front‑end. Runs on a Jetson Orin NX at ~10 Hz indoor, ~3 Hz outdoor. The first practically deployable 3DGS SLAM.
- **LiV‑GS** (Nov 2024): outdoor LiDAR‑vision GS. Larger scenes; lower fidelity per Gaussian.
- **MM3DGS** (IROS 2024): multi‑modal GS with vision + depth + inertial. Mostly indoor.

**Verdict for large‑scale tactical work**: 3DGS is the wrong layer. The Gaussians don't define a surface (they smear across them) — useless for planning. They are excellent for "you‑are‑there" replay or photo‑realistic playback, but for a tactical 3D overview, a coloured TSDF mesh is faster, more readable, and renders anywhere.

### 4.6 Mesh and texture

For the **tactical render** layer specifically:

- **Marching Cubes** on TSDF: produces a triangle mesh with per‑vertex colour (from per‑voxel RGB). Standard pipeline.
- **Dual Contouring** (Voxfield, some custom systems): produces sharper edges than Marching Cubes — buildings come out crisper.
- **Screened Poisson reconstruction** (Kazhdan & Hoppe 2013): offline, takes the full point cloud, produces watertight smooth mesh. Best for final deliverable maps; not for live SLAM.
- **Per‑face texture projection**: for crisper colour than per‑voxel allows, project camera keyframes onto the mesh faces post hoc. Standard photogrammetry; cheap because keyframes already have poses.

---

## 5. Place recognition and loop closure

### 5.1 Why geometric proximity fails

Your arc‑slam detector uses `kdtreeHistoryKeyPoses->radiusSearch(15 m)` — find any keyframe within 15 m of current. This fails the moment drift exceeds 15 m. After kilometres of LIO with no closures, drift is routinely 1–10 m per 100 m. So you can drive past your own start point and miss the loop because your KdTree says you're 30 m away.

Place recognition removes this dependency: identify "this is the same place" from the *appearance* of the scan (or scan + image), independent of where you think you are. Then run geometric verification.

### 5.2 The descriptor zoo

| Descriptor | Year | Input | Strength | Weakness |
|---|---|---|---|---|
| **Scan Context** | 2018 | Polar bird's‑eye height histogram | Rotation invariant by ring shift; fast (KdTree on ring keys) | Sensitive to perspective change in narrow places |
| **Scan Context++** | 2021 | Adds sector key + per‑ring weights | More robust to occlusion | Same locality bias |
| **Intensity SC** | 2020 | Adds LiDAR intensity channel | Disambiguates similar geometry by reflectance | Requires intensity calibration |
| **Semantic SC** | 2022 | Per‑cell class histogram | Robust across viewpoint changes | Needs a semantic segmenter |
| **STD** | 2023 ICRA | Stable Triangle Descriptor — triplets of stable keypoints | Strong on structured scenes; small descriptor | Needs robust keypoint detection |
| **BTC** | 2024 | Binary Triangle Code — binarized STD | 10× faster than STD | Small accuracy loss |
| **OverlapNet** | 2020 | Learned, predicts pairwise scan overlap | Direct loop candidate scoring | Network needs domain training |
| **MinkLoc3D** | 2021 | Sparse 3D convolution, learned global desc | SOTA accuracy on benchmarks | GPU required for inference |
| **Logg3dNet** | 2022 | Local + global learned descriptors | Best on large‑scale outdoor | Heavier than MinkLoc3D |
| **BoW3D** | 2022 | Bag‑of‑words on LinK3D local features | Real‑time, no GPU | Needs vocabulary training |

For 2026 deployments the pragmatic stack is:
- **Scan Context++** for fast candidate retrieval (CPU, 100+ Hz).
- **STD or BTC** for geometric verification (CPU, milliseconds per candidate).
- **GICP or NDT** for final pose alignment.
- **Optional semantic descriptor** when you have a segmenter, for cross‑season robustness.

For research / GPU‑equipped systems:
- **MinkLoc3D or Logg3dNet** for retrieval.
- Same geometric verification + alignment.

### 5.3 Hierarchical recognition

Modern systems run **multiple descriptors in series**:

1. Coarse global descriptor returns the top‑k candidates (k = 50–100), fast.
2. Mid descriptor (e.g., STD) re‑ranks to the top‑5.
3. Geometric verification (GICP) on the top‑5; accept the first that converges below a fitness threshold.

This pipeline gracefully degrades — if step 1 is noisy, step 2 and 3 filter. If step 3 fails, the closure is rejected, no bad factor added.

### 5.4 Semantic loop closure

Once you have a per‑voxel/per‑keyframe semantic layer (INF‑SLiM, PIN‑SLAM, or a separate 3D segmenter), you get cheap loop closure for free: match the histogram of class counts in the current keyframe to historical ones. Invariant to viewpoint, season, lighting. Most useful as a *secondary* signal — high‑recall, low‑precision, paired with geometric verification.

---

## 6. Benchmarks and evaluation

What the field actually competes on in 2026:

| Benchmark | Focus | Why it matters |
|---|---|---|
| **KITTI** | Urban driving, single LiDAR + camera | Legacy; almost saturated by 2023 |
| **KITTI‑360** | Larger urban | Still active for visual SLAM |
| **MulRan, Oxford Radar** | Long‑term outdoor, weather | Cross‑condition robustness |
| **Newer College** | Hand‑carried indoor‑outdoor | Tight LIVO benchmarks (FAST‑LIVO2, Coco‑LIC) |
| **Hilti SLAM Challenge** | Construction sites, tunnels, degenerate | Robustness; Wildcat winners 2022/23 |
| **SubT** | Underground, GPS‑denied, multi‑robot | The hardest mainstream benchmark; ended 2021 but datasets remain |
| **M2DGR** | Multi‑modal ground robot, urban | Multi‑sensor fusion focus |
| **Boreas** | Long‑term, weather, radar+camera+LiDAR | Cross‑season; radar inclusion |
| **Degradation‑robust LiDAR‑Inertial Dataset (arXiv 2507.20516, 2025)** | Specifically designed to break existing LIO | The new hard bar; expect 2026–27 papers to cluster here |

The 2026 SOTA papers all report on Newer College + Hilti + the 2025 degradation dataset as their robustness story. KITTI numbers alone are no longer publishable.

---

## 7. Where the field is going (genuinely uncertain)

Honest assessment of open problems and active research directions in 2026:

- **Foundation models for SLAM**: large vision‑language models embedded in the loop for semantic grounding ("the robot is in a kitchen, expect counters and appliances"). Two or three preprint efforts in 2025; nothing production yet.
- **Differentiable SLAM end‑to‑end**: PIN‑SLAM is the closest; full back‑prop through factor graph optimization is a research topic.
- **Multi‑robot collaborative mapping**: DiSCo‑SLAM, Kimera‑Multi, DC2SLAM. Real systems exist; integration with neural maps unresolved.
- **Long‑term map maintenance**: how to update a year‑old map when the world has changed. Active research; no canonical solution.
- **Adversarial robustness**: GPS spoofing, LiDAR jamming (real, demonstrated in 2024). Tactical systems need defences; the academic literature is light here.
- **On‑device 3DGS at scale**: GS‑LIVO showed embedded feasibility, but km‑scale outdoor 3DGS in real time is still aspirational.

---

## 8. Reading list, organized

Foundations (read once, reference forever):
- **Solà, Deray, Atchuthan 2018** — *Micro Lie theory for state estimation*, arXiv:1812.01537.
- **Forster, Carlone, Dellaert, Scaramuzza 2017** — *On‑Manifold Preintegration*, IEEE T‑RO.
- **Kaess et al. 2012** — *iSAM2*, IJRR.

Front‑end systems:
- **Zhang & Singh 2014** — *LOAM*, RSS.
- **Shan et al. 2020** — *LIO‑SAM*, IROS. (your direct ancestor)
- **Xu et al. 2021/2022** — *FAST‑LIO / FAST‑LIO2*, T‑RO.
- **He et al. 2023** — *Point‑LIO*, AISY.
- **Cai et al. 2021** — *ikd‑Tree*, RA‑L.
- **Jiao et al. 2022** — *M‑LOAM*, IEEE T‑RO. (multi‑LiDAR)

Visual+LiDAR fusion:
- **Lin et al. 2022** — *R3LIVE*.
- **Zheng et al. 2022/2024** — *FAST‑LIVO / FAST‑LIVO2*, T‑RO.
- **Lang et al. 2023** — *Coco‑LIC*, RA‑L. (continuous‑time)

Maps:
- **Oleynikova et al. 2017** — *Voxblox*, IROS.
- **Kerbl et al. 2023** — *3D Gaussian Splatting*, SIGGRAPH.
- **Pan et al. 2024** — *PIN‑SLAM*, T‑RO.
- **Zhang et al. 2026** — *INF‑SLiM*, J. Field Robotics.

Robustness:
- **Zhang et al. 2016** — *Degeneracy of Optimization‑based State Estimation*, ICRA.
- **Tuna et al. 2023** — *X‑ICP*, IEEE T‑RO.
- **Sünderhauf & Protzel 2012** — *Switchable Constraints*, IROS.
- **Yang et al. 2020** — *Graduated Non‑Convexity*, IEEE T‑RO.
- **Mangelson et al. 2018** — *Pairwise Consistency Maximization*, ICRA.

Continuous time:
- **Lv et al. 2021** — *CLINS*.
- **Lang et al. 2023** — *Coco‑LIC*, RA‑L.
- **Bosse, Zlot et al.** — *Wildcat / Zebedee* (multi‑year CSIRO line; see SubT publications).

Recent (2025–2026):
- **D²‑LIO** (Aug 2025) — arXiv 2508.14355.
- **GS‑LIVO** (Jan 2025) — arXiv 2501.08672.
- **INAF Fusion** (Oct 2025) — arXiv 2510.15803.
- **AVK‑SLAM** (2026) — Springer ISR.
- **Hi‑LOAM** (2026) — arXiv 2604.01720.
- **XGrid‑Mapping** (Dec 2025) — arXiv 2512.20976.
- **INF‑SLiM** (2026) — J. Field Robotics 10.1002/rob.70058.
- **GPS‑Denied LiDAR‑SLAM Survey** (2025) — IET Cyber‑Systems and Robotics.
- **Degradation‑Robust LiDAR‑Inertial Dataset** (2025) — arXiv 2507.20516.

*End of SOTA.*
