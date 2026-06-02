# Grounding Dossier 07 — Dense Mapping: TSDF + ESDF, NVBlox, Voxblox, and Surface Reconstruction

**Scope:** The dense-mapping layer of Meridian. How range/RGB measurements become a volumetric Truncated Signed Distance Field (TSDF), how that becomes a colourised triangle mesh (Marching Cubes) and, optionally, a Euclidean Signed Distance Field (ESDF) for planning, and how loop-closure pose corrections force a clear-and-rebuild. Covers Voxblox (CPU, ESDF), NVIDIA nvblox (GPU), VDBFusion / OpenVDB / NanoVDB (sparse km-scale), Marching Cubes vs Dual Contouring, and Screened Poisson for the final archival watertight mesh.

**Status of sources.** All technical content below is grounded in the primary papers and the canonical open-source repos (Voxblox `ethz-asl/voxblox`, nvblox `nvidia-isaac/nvblox`, VDBFusion `PRBonn/vdbfusion`, OpenVDB/NanoVDB `AcademySoftwareFoundation/openvdb`, PoissonRecon `mkazhdan/PoissonRecon`). Where a specific equation number, default value, or arXiv ID could not be verified against the primary text during research, it is **explicitly flagged as approximate / to-verify** rather than asserted. Treat flagged items as "confirm before writing the formal spec".

---

## 0. Executive summary and recommendation (read this first)

Meridian's pipeline target is: tightly-coupled state estimator → per-keyframe deskewed point cloud + pose → **volumetric fusion (TSDF + RGB)** → **Marching Cubes** → colourised mesh, with loop-closure-driven rebuilds, and an **offline Screened-Poisson** pass for the final archival watertight mesh.

The four candidate fusion libraries and the one-line verdict:

| Library | Compute | Storage | ESDF? | Mesh | Scale sweet-spot | Verdict for Meridian |
|---|---|---|---|---|---|---|
| **Voxblox** (`ethz-asl/voxblox`) | CPU | block-hash, 16³ voxel blocks | **Yes** (incremental) | per-block MC | indoor / MAV | Reference for ESDF + bundled raycasting; ageing codebase. |
| **nvblox** (`nvidia-isaac/nvblox`) | **GPU/CUDA** | block-hash, 8³ blocks, CUDA unified memory | **Yes** (GPU) | GPU MC | room → building, real-time | **Recommended if a CUDA GPU is in the deployment target.** |
| **VDBFusion** (`PRBonn/vdbfusion`) | CPU | **OpenVDB** sparse tree (8³ leaves) | No | OpenVDB MC (`volumeToMesh`) | **km-scale outdoor LiDAR** | **Recommended CPU fallback / km-scale path.** |
| **OpenVDB/NanoVDB** (`AcademySoftwareFoundation/openvdb`) | CPU (+GPU read via NanoVDB) | the storage substrate itself | No (DIY) | `tools::volumeToMesh` | foundational | Use as the *storage layer*, not a turnkey fusion lib. |

**Primary recommendation for Meridian:** **nvblox** as the online fusion + meshing engine *iff* a CUDA-capable GPU is guaranteed in the target platform (Jetson Orin or x86+RTX). It gives GPU TSDF + GPU ESDF + GPU Marching Cubes + colour in one library, has a maintained ROS 2 wrapper (`isaac_ros_nvblox`), and is the natural fit for a system that already wants GPU for the camera/iEKF front-end. **CPU fallback / km-scale outdoor path:** **VDBFusion on OpenVDB**, because OpenVDB's hierarchical sparsity scales to kilometre trajectories with memory proportional to observed surface area, and its `tools::volumeToMesh` gives a clean MC mesh. **Keep Voxblox only as the ESDF algorithm reference** (its incremental raise/lower wavefront is the cleanest published description) — do not adopt its ageing code wholesale.

**Archival mesh:** run **Screened Poisson Surface Reconstruction** (Kazhdan & Hoppe 2013, `mkazhdan/PoissonRecon`, or CGAL's `poisson_surface_reconstruction`) *offline* on the retained per-keyframe oriented point store after the final globally-optimised trajectory is known. MC-on-TSDF is the online/interactive mesh; Poisson is the watertight deliverable.

**The non-negotiable architectural consequence** (Section 8): a running-weighted-average TSDF is **not per-keyframe reversible**. When a loop closure shifts historical poses, you **cannot** "subtract" old observations from the voxel grid. Meridian **must** retain a **per-keyframe deskewed cloud store** (cloud + the pose that placed it) and, on a significant pose correction, **clear and re-integrate** the affected region (or the whole map) from that store using the corrected poses. This requirement dictates that the keyframe cloud store is a first-class, persistent data structure, not a transient buffer.

---

## 1. The TSDF: definition, update equations, colour

### 1.1 What a TSDF is

A **Signed Distance Field (SDF)** assigns to every point in space the signed distance to the nearest surface: negative inside an object, positive in free space (the sign convention used by KinectFusion/Voxblox/nvblox is **positive in front of the surface toward the sensor, negative behind it**). The **zero-level-set** (D = 0) *is* the surface; meshing extracts it.

A **Truncated** SDF only stores distances within a band ±δ ("truncation distance") of the surface. Outside the band the value is clamped to ±δ and the voxel carries no useful gradient. Truncation is what makes the representation tractable: only a thin shell of voxels around observed surfaces is ever updated, and conflicting measurements average cleanly near the surface instead of fighting over far-away free space.

Lineage: Curless & Levoy (SIGGRAPH 1996, "A volumetric method for building complex models from range images") introduced the cumulative weighted-average TSDF; KinectFusion (Newcombe et al., ISMAR 2011) made it real-time on GPU; Voxblox/nvblox/VDBFusion are direct descendants.

### 1.2 Signed distance from a measurement

For a measured surface point **p** (range return / back-projected depth pixel), sensor origin **o**, and a voxel centre **v**, the **projective** signed distance used along the ray is:

```
sdf(v) = ‖p − o‖ − ‖v − o‖            (Eq. TSDF-1, projective/along-ray form)
```

i.e. how much further along the ray the surface is than this voxel. `sdf > 0` ⇒ voxel is in front of the surface (free); `sdf < 0` ⇒ behind it (occluded interior). This is the form used by VDBFusion and the simple projective integrators.

A more accurate **non-projective / Euclidean** variant uses the true Euclidean distance from the voxel to the measured point (Voxblox supports this), which reduces the systematic error projective distance incurs at grazing angles, at extra cost.

Truncation:

```
d(v) = clamp( sdf(v), −δ, +δ )         (Eq. TSDF-2)
```

Voxels are only updated where `sdf(v) ≥ −δ` (within the band on the near side, and one truncation band behind the surface). With **space carving / voxel carving** enabled, the entire ray from the sensor to `p − δ` is also marked free (large positive distance), which removes dynamic obstacles and trailing artefacts at the cost of more updates.

### 1.3 Running weighted-average update (the core fusion equation)

Each voxel stores a distance `D` and an accumulated weight `W`. A new measurement contributes `(d, w)`:

```
D_new(v) = ( W(v)·D(v) + w·d(v) ) / ( W(v) + w )       (Eq. TSDF-3)
W_new(v) = min( W(v) + w, W_max )                       (Eq. TSDF-4)
```

This is the Curless–Levoy cumulative average. `W_max` (a.k.a. `max_weight`) caps the weight so the map stays adaptive to change — without it, an old surface observed thousands of times can never be moved by new evidence. Capping `W` turns the average into an exponential-forgetting filter once saturated.

(These are Eqs. 1–2 of the Voxblox paper, Section III; the exact in-paper numbering should be confirmed against the PDF — **flagged**.)

### 1.4 The measurement weight w(x)

The per-measurement weight encodes sensor confidence and shapes the surface. Voxblox analyses several weighting schemes (Section III, weighting analysis):

- **Constant** weight `w = 1` (config `use_const_weight`): simplest; used by default in VDBFusion. Good for LiDAR where return confidence is roughly range-independent.
- **Inverse-square depth** `w ∝ 1/z²`: models depth-camera (stereo/structured-light) noise that grows quadratically with range — far measurements contribute less.
- **Behind-surface drop-off:** within the truncation band, weight is held roughly constant from the sensor up to the surface, then **drops off linearly and then exponentially** behind the surface toward `−δ`, and is **zero beyond `−δ`**. The drop-off keeps occluded/interior voxels from being over-confidently written by a single grazing ray.

The weight function is the main knob that trades **sharpness** (small band, high near-surface weight) against **robustness to noise** (wider averaging).

### 1.5 Colour integration

Colour is fused with its **own weighted average**, separate from the distance, so a colour observed at a grazing angle does not corrupt geometry:

```
C_new(v) = ( W_c(v)·C(v) + w_c·c ) / ( W_c(v) + w_c )   (per-channel, Eq. TSDF-5)
W_c_new(v) = min( W_c(v) + w_c, W_c_max )
```

In practice colour is only blended in the thin band very close to the surface (|D| small), because colour far from the zero-crossing is meaningless. nvblox does this in a dedicated `ProjectiveColorIntegrator` writing a `ColorLayer`; Voxblox stores `Color` inside the `TsdfVoxel`. At mesh time the per-vertex colour is sampled/interpolated from the colour field — this is how the "colourised mesh" deliverable is produced.

### 1.6 Voxel / weight numeric type note

Voxblox and nvblox store `distance` and `weight` as `float`. Be aware of weight saturation precision and of colour stored as 8-bit RGBA with a separate float/uint16 colour-weight. Keep this in the formal spec when defining the voxel struct.

---

## 2. TSDF integrators: projective vs raycasting vs bundled

How measurements are turned into voxel updates is the single biggest performance lever. Three families (Voxblox names them `simple`, `merged`, `fast`):

### 2.1 Simple / projective integrator (`simple`)

For every measured point, **raycast** from `o` toward `p` and update every voxel in the truncation band along the ray. Accurate (it touches every voxel a real ray passes through) but **slow**: voxels near the sensor are hit by enormous numbers of overlapping rays and updated redundantly. Voxblox calls the naive per-point raycaster `simple`; nvblox's `ProjectiveTsdfIntegrator` is the *projective* variant that instead iterates voxels in the frustum and looks each up in the depth image (one update per voxel, GPU-friendly).

**Projective (voxel→image) vs raycasting (ray→voxels)** is the key distinction:
- **Projective:** iterate candidate voxels, project each into the depth image, read the depth, compute `sdf`. One pass, embarrassingly parallel ⇒ ideal on GPU (nvblox). Artefacts at depth discontinuities/silhouettes because a voxel may project onto a foreground pixel that isn't on its true ray.
- **Raycasting:** iterate measurements, walk voxels along each ray. More faithful near edges; redundant near the origin.

### 2.2 Merged / bundled raycasting (`merged`) — Voxblox's key efficiency contribution

The Voxblox contribution (Section III) for **real-time CPU** operation: **group all measured points that fall into the same voxel** at the ray endpoints ("bundling"), replace each group by a single representative point with the **combined weight** (and merged colour), then cast **one bundled ray per group**. This collapses the redundant near-sensor updates while preserving accuracy, giving a large speedup over `simple` with minimal quality loss. This is the integrator to study for any CPU raycasting design.

### 2.3 Fast integrator (`fast`)

Voxblox's `fast` integrator targets hard real-time: it subsamples start voxels (`start_voxel_subsampling_factor`), limits redundant work via per-voxel "already updated this frame" checks and a cap on consecutive ray collisions (`max_consecutive_ray_collisions`), and bounds the number of integrated rays. Trades a little accuracy for predictable latency.

### 2.4 Recommendation for Meridian

- **GPU path (nvblox):** use the **projective** TSDF integrator — it is what nvblox is built around and is the reason it hits real-time at small voxel sizes. Accept that projective edges are slightly noisier than raycast; the colour/mesh quality is excellent in practice.
- **CPU path (VDBFusion):** VDBFusion uses **raycasting** along each LiDAR ray on the OpenVDB grid (with `ValueAccessor` caching to make neighbour walks ~O(1)). For dense rotating-LiDAR scans this is fine; enable `space_carving` only if you need dynamic-object removal (it roughly doubles the work).
- If you ever write a bespoke CPU integrator, **copy the Voxblox `merged` bundling strategy** — it is the best-documented way to get raycast quality at near-projective cost.

---

## 3. Storage: voxel hashing, OpenVDB, NanoVDB

### 3.1 Block-hash (Voxblox, nvblox)

Both use a two-level **block hash**:

```
Layer  →  hash_map< BlockIndex(Vec3i), Block* >
Block  →  dense [N×N×N] array of Voxels  (+ origin, block index)
Voxel  →  { float distance; float weight; Color color; … }
```

- **Voxblox:** `Layer<VoxelType>` (e.g. `TsdfLayer`), `Block<VoxelType>`, `TsdfVoxel{distance, weight, color}`. **`voxels_per_side` defaults to 16** ⇒ blocks are **16×16×16 = 4096 voxels**. `block_size = voxels_per_side × voxel_size`.
- **nvblox:** `VoxelBlock<VoxelType>` with **8×8×8 = 512 voxels** per block, allocated in **CUDA unified memory** so the same pointer is valid on host and device; a `GPUHashMap` / `GPULayerView` exposes blocks to CUDA kernels.

Index math (both):
```
BlockIndex = floor( p / block_size )
VoxelIndex = floor( (p − block_origin) / voxel_size )
```
Blocks are **allocated lazily** only where measurements land ⇒ memory ∝ observed volume shell. Sparse, but the per-block hash has overhead and the 3D bookkeeping is non-trivial.

### 3.2 OpenVDB tree (VDBFusion)

OpenVDB stores a grid as a shallow, wide **B+tree-like "VDB" tree**: **Root** (a sparse hash map, effectively unbounded extent) → two levels of **Internal nodes** → **Leaf nodes** holding dense voxel grids. The default `5-4-3` configuration means log₂ dims of `32 / 16 / 8`, so **leaf nodes are 8×8×8 = 512 voxels** — same block granularity as nvblox.

Key properties:
- **Background value + tiles:** unallocated space is represented by a single background value; large uniform regions are stored as constant "tiles" on internal nodes ⇒ **memory ∝ surface area, not bounding volume** ⇒ ideal for **unbounded km-scale** maps.
- **`tree::ValueAccessor`** caches the node path from the last accessed leaf up the tree; spatially coherent accesses (voxels along a ray) reuse the cache for ~O(1) neighbour access. This is what makes raycast TSDF fusion on OpenVDB fast.
- **Meshing:** `openvdb::tools::volumeToMesh(grid, points, quads, tris, isovalue, adaptivity)` runs MC-style polygonisation directly on the grid; `adaptivity` merges coplanar regions to cut triangle count.

### 3.3 NanoVDB

**NanoVDB** is a **read-optimised, pointerless, statically-laid-out** linearisation of a VDB tree into a **single contiguous buffer** (offsets instead of pointers) that can be `memcpy`'d to the **GPU** and traversed inside CUDA kernels. Same logical tree (leaf 8³), **immutable topology** once built. It is the bridge between OpenVDB's authoring/streaming and GPU consumption. (Note: nvblox's *primary* on-device structure is its own unified-memory block hash; NanoVDB is the ASWF-standard GPU VDB format and the natural interchange/streaming format if Meridian wants OpenVDB authoring + GPU rendering.)

### 3.4 Recommendation

- **Online, bounded/room-to-building, GPU available →** nvblox block-hash (8³, unified memory).
- **Online, km-scale outdoor, CPU →** OpenVDB grid via VDBFusion. The hierarchical sparsity + tiles is the only one of these that gracefully reaches kilometres without a tiling/active-window scheme bolted on.
- **Streaming a large OpenVDB map to a GPU renderer →** export to **NanoVDB**.

---

## 4. ESDF: propagating Euclidean distance from the TSDF

Meridian's mesh deliverable does **not** require an ESDF. Build the ESDF **only** if a planning/collision-avoidance consumer needs it. It is documented here because both Voxblox and nvblox can produce one and it is the historically headline Voxblox feature.

### 4.1 TSDF vs ESDF

- **TSDF** distance is only valid **within ±δ** of the surface (projective, truncated). Useless for "how far is the nearest obstacle" beyond the band.
- **ESDF** gives the **true Euclidean distance to the nearest surface at every voxel**, plus a usable gradient ⇒ what trajectory optimisers (CHOMP, etc.) want.

### 4.2 Incremental wavefront / brushfire (Voxblox, Section IV)

Voxblox builds the ESDF **incrementally** from the TSDF using a **two-queue wavefront (brushfire)**:

1. **Fixed band:** voxels within the TSDF truncation are copied directly: `ESDF.distance = TSDF.distance`, marked `fixed`. These are the seeds / boundary conditions.
2. **Lower queue (distances decreasing — new/closer obstacle):** for each voxel, relax neighbours:
   ```
   for n in neighbours(v):
     if dist(v) + edge(v,n) < dist(n):
       dist(n) = dist(v) + edge(v,n);  parent(n) = v;  push n to lower_queue
   ```
3. **Raise queue (distances increasing — obstacle removed/farther):**
   ```
   for n in neighbours(v):
     if parent(n) == v:            # n depended on the now-invalid v
        invalidate n; push n to raise_queue
     else:
        push n to lower_queue       # n can re-seed a correct, smaller value
   ```
   Process the raise queue first (invalidate stale regions), then the lower queue (re-propagate correct values).

When the TSDF integrator reports **changed voxels** each frame, those seed the queues — newly observed ⇒ lower, newly cleared ⇒ raise — so cost is **O(changed voxels)**, not a full recompute.

### 4.3 Quasi-Euclidean distance

Exact Euclidean ESDF is expensive. Voxblox uses a **quasi-Euclidean** approximation propagating to 26-connected neighbours with edge weights `1` (face), `√2` (edge), `√3` (corner), tracking a `parent` direction. Near-Euclidean accuracy at a fraction of the cost.

### 4.4 nvblox ESDF

nvblox's `EsdfIntegrator` computes the ESDF on the **GPU** (parallel wavefront) from the TSDF/occupancy layer and can emit a **2D ESDF slice** at a chosen height for a Nav2 costmap (`isaac_ros_nvblox`).

### 4.5 Failure modes

- **Stale ESDF after pose correction:** an ESDF derived from a TSDF that was just cleared-and-rebuilt must also be rebuilt; the incremental queues only handle *local* changes, not a global pose shift.
- **Truncation too small** ⇒ fixed band too thin ⇒ the wavefront seeds from a noisy boundary ⇒ ESDF artefacts. Keep δ ≥ a few voxels.

---

## 5. Surface extraction: Marching Cubes (online) vs Poisson (archival)

### 5.1 Marching Cubes (Lorensen & Cline, SIGGRAPH 1987)

The standard way to extract the TSDF zero-level-set. March over grid **cells (cubes)**, each with **8 corners** carrying scalar values; classify each corner inside/outside vs the iso-value (here iso = 0):

```
corner is "inside" if value ≤ iso, else "outside"
```

The 8 inside/outside bits form an **8-bit cube index ∈ [0,255]**. The **256 configurations** reduce by rotational, reflective and complementary symmetry to **15 unique base cases**. The cube index looks up:
- an **edge table** (256 entries, 12-bit mask): which of the cube's **12 edges** the surface crosses;
- a **triangle table** (256 entries): how the edge-crossing points connect into **0–5 triangles**.

Each crossing vertex is placed by **linear interpolation** along the edge between corners `(P₀,v₀)` and `(P₁,v₁)`:

```
t = (iso − v₀) / (v₁ − v₀)
P = P₀ + t · (P₁ − P₀)            (Eq. MC-1)
```

(with `iso = 0` for a TSDF). Vertex **normals** come from interpolating the field gradient (central differences) at corners, or from the TSDF gradient ∇D; vertex **colour** from sampling the colour field at `P`.

**Ambiguity:** on a cube face with two diagonal "inside" corners and two "outside" (face ambiguity), and analogous interior ambiguities, the original 15-case table can connect adjacent cubes inconsistently ⇒ **holes/cracks**. Fixes:
- **Asymptotic decider** (Nielson & Hamann 1991): use the bilinear saddle value on the ambiguous face to choose connectivity consistently.
- **Marching Cubes 33** (Chernyaev 1995): extended 33-case table + asymptotic decider ⇒ guaranteed **topologically correct, crack-free, manifold** output. Prefer an MC33 implementation if watertightness of the *online* mesh matters; otherwise plain MC is fine for visualisation.

**Properties for Meridian:** local, incremental (only re-mesh blocks whose TSDF changed — both Voxblox `MeshIntegrator` and nvblox do this), GPU-parallel (nvblox runs MC on GPU), produces an **open** mesh at unobserved boundaries (no hole-filling). This is exactly what you want for the **online, interactive, colourised** mesh.

### 5.2 Dual Contouring (sharp-edge alternative)

**Dual Contouring** (Ju, Losasso, Schaefer, Warren 2002) places **one vertex per cell** (the dual of MC's on-edge vertices) using **Hermite data** — for each sign-changing edge it stores the **crossing point and the surface normal** — and solves a **Quadratic Error Function**:

```
E(x) = Σ_i [ n_i · (x − p_i) ]²       (Eq. DC-1)
```

minimised over the cell's Hermite samples `(p_i, n_i)`. This places the vertex at the intersection of tangent planes, **reproducing sharp edges and corners** that MC rounds off; it also adapts naturally to octrees. Downsides: needs reliable normals, can produce **non-manifold** vertices without extra handling, and is more complex.

| | Marching Cubes | Dual Contouring |
|---|---|---|
| Vertex placement | on edges (interpolated) | one per cell (QEF) |
| Inputs | scalar field | Hermite (point+normal) |
| Sharp features | rounded | preserved |
| Manifold | yes (with MC33) | needs care |
| Octree adaptivity | harder | natural |
| Used by TSDF libs | **yes** (Voxblox/nvblox/VDBFusion) | rare |

**Recommendation:** **MC (ideally MC33) for Meridian's online mesh.** TSDFs are noisy and organic-scene-dominated; DC's sharp-feature advantage doesn't pay off and its normal sensitivity hurts. Consider DC only for a specialised CAD/structured-scene mode.

### 5.3 Screened Poisson Surface Reconstruction (Kazhdan & Hoppe, ToG 2013) — the archival mesh

Operates on **oriented points** (positions + normals), not on a grid, and produces a **globally smooth, watertight** mesh — the right tool for the **final archival deliverable**.

Core idea: the indicator function χ (1 inside, 0 outside) has gradient zero except at the surface, where it equals the surface normal. Build a vector field **V** from the oriented samples and solve for χ whose gradient matches it ⇒ a **Poisson equation**:

```
min_χ ‖∇χ − V‖²   ⇒   Δχ = ∇·V        (Eq. POIS-1)
```

The surface is the iso-surface of χ at the average χ over the samples.

**Screening (the 2013 contribution):** add a **positional interpolation constraint** pulling the iso-surface to actually pass through the input points, fixing the original (2006) method's over-smoothing/drift:

```
E(χ) = ∫ ‖∇χ − V‖² dx  +  α · (1/Σw_i) · Σ_i w_i (χ(p_i) − iso)²     (Eq. POIS-2)
```

α is the **point/screening weight** (default ≈ 4 in the released code — **verify**). χ is represented in **octree-adaptive B-spline** bases (refined near samples, coarse elsewhere ⇒ memory ∝ surface area); discretising E gives a **sparse SPD linear system** solved with a **multigrid** solver in near-linear time. Because χ is a single globally-defined smooth function, its iso-surface is a **closed manifold** even across data gaps ⇒ watertight.

**Key parameters (`mkazhdan/PoissonRecon` CLI):**
- `--depth` (octree depth, ~8–11): detail vs memory. The dominant quality/cost knob.
- `--pointWeight` (α, screening strength; default ~4): higher ⇒ closer to data, less smoothing.
- `--samplesPerNode` (min samples per node, ~1–5; raise for noisy data ⇒ smoother).
- `--density` + a **trim** post-step: cut low-density (extrapolated) triangles to remove hallucinated surface in unobserved regions.
- Boundary condition (Neumann/Dirichlet), scale.

### 5.4 MC-on-TSDF vs Poisson — decision

**Use Marching Cubes on the TSDF** when: online/incremental/real-time; you already maintain a TSDF; GPU-friendly; open meshes at unobserved boundaries are acceptable. ⇒ **Meridian's live mesh.**

**Use Screened Poisson** when: final archival pass; need **watertight** + graceful hole-filling; have good oriented normals; offline global solve is acceptable. Downside: not incremental; can hallucinate surface in unsupported regions (mitigate with density trimming). ⇒ **Meridian's deliverable mesh.**

Normals for Poisson can come from the **TSDF gradient** (∇D, already smooth and consistent) sampled at the mesh vertices, **or** computed per-point on the retained keyframe clouds (PCA on k-NN, oriented toward the sensor origin). The TSDF-gradient route is usually cleaner because fusion has already denoised it.

---

## 6. Parameters that matter (and sane starting values)

| Parameter | Meaning | Effect | Indoor/room start | Outdoor LiDAR start |
|---|---|---|---|---|
| `voxel_size` | edge length of a voxel | the master detail/cost/memory knob; cost ∝ ~1/voxel³ | 0.02–0.05 m | 0.10–0.20 m |
| `truncation_distance` δ | band half-width | small δ = sharp but noise-sensitive & thin ESDF band; large δ = smooth, robust, more updates | 3–5 × voxel | 3 × voxel (e.g. 0.3 m) |
| `max_weight` (`W_max`) | weight cap | small = adapts fast to change (good for dynamics), large = stable/low-noise but stale | moderate | moderate–high |
| `max_integration_distance` / `max_ray_length_m` | depth cutoff | ignore noisy far returns; bound work | ~5–7 m (depth cam) | sensor max (e.g. 50–100 m) |
| weight model | const vs 1/z² vs drop-off | shapes sharpness vs noise | 1/z² (depth cam) | const (LiDAR) |
| `voxels_per_side` / block size | hash granularity | 16³ (Voxblox) / 8³ (nvblox/OpenVDB); smaller = finer alloc, more overhead | lib default | lib default |
| `space_carving` / `voxel_carving_enabled` | clear full ray | removes dynamics/trailing artefacts; ~2× cost | off unless dynamics | on for dynamic scenes |
| colour weight / band | colour blend region | restrict to |D|≪δ near surface | near-surface only | near-surface only |
| Poisson `--depth`, `--pointWeight`, trim | archival mesh | detail / data-fidelity / hole-trimming | 9–10, ~4, trim | 10–11, ~4, trim |

**Rule of thumb:** set `voxel_size` first from the smallest feature you must capture and your compute budget, then `δ = (3–5)·voxel_size`, then tune the weight model and `max_weight` for your dynamics.

---

## 7. Memory behaviour at km scale

- **Block-hash (Voxblox, nvblox):** memory ∝ allocated blocks ∝ observed-surface shell. A voxel ≈ a few–dozen bytes (float distance + float weight + colour + colour weight ≈ 12–16 B). A 16³ block ≈ 4096 voxels ≈ tens of KB; an 8³ block ≈ 512 voxels. At km scale the number of surface blocks grows large; **nvblox is bounded by GPU VRAM** (a hard ceiling) and **Voxblox by host RAM**. Neither was designed for unbounded outdoor extent without an **active-window / map-tiling** scheme that swaps far blocks to disk.
- **OpenVDB (VDBFusion):** memory ∝ surface area, with empty space collapsed to background + tiles. This is the structure that **actually reaches kilometres** on CPU RAM; it is why VDBFusion is the LiDAR-mapping choice on large sequences (KITTI-scale).
- **NanoVDB:** compact contiguous buffer; the format for **streaming** a large OpenVDB map to GPU for rendering, and for shipping a finished grid.
- **Mitigations regardless of library:** (a) coarser `voxel_size` outdoors; (b) **submapping** — many small bounded TSDF submaps anchored to keyframe poses, meshed independently and stitched, so a loop closure rebuilds only affected submaps (see §8); (c) stream finished/old regions to disk and keep only an active window resident.

---

## 8. Loop closure ⇒ CLEAR-AND-REBUILD (the load-bearing constraint)

### 8.1 Why the TSDF cannot be incrementally corrected

The TSDF voxel value is a **running weighted average** (Eq. TSDF-3) of *all* measurements that ever hit it, accumulated in the **global frame using the poses that were current at integration time**. It is a lossy sum: the individual contributions are **not stored**, so the fusion is **not per-keyframe reversible**. When the back-end (GTSAM iSAM2) applies a loop closure and **shifts historical keyframe poses**, every voxel that was written using a now-wrong pose is wrong, and there is **no operation to "subtract" the stale observations**. You cannot patch the grid in place.

### 8.2 The required architecture: retained per-keyframe cloud store + re-integration

Therefore Meridian **must** keep a **persistent per-keyframe store**:

```
Keyframe k : { deskewed cloud C_k (in the keyframe/body frame),
               pose T_k (a handle into the iSAM2 pose graph),
               optional per-point colour, timestamp }
```

The TSDF/mesh is a **pure function** of `{(C_k, T_k)}`. On a pose correction:

1. **Detect** which keyframe poses changed beyond a threshold (translation/rotation) after the iSAM2 update.
2. **Clear** the affected region (the union of the touched submaps, or — if poses shifted globally — the whole grid).
3. **Re-integrate** the affected keyframes' clouds at their **corrected** poses `T_k`, using the same integrator and parameters.
4. **Re-mesh** the dirtied blocks (MC) and, if used, **rebuild the ESDF** for them.

This is exactly why the deliverable's spec says "loop-closure pose corrections force a CLEAR-AND-REBUILD". It is not a limitation of a particular library — it is intrinsic to running-average TSDF.

### 8.3 Submapping to make rebuilds cheap (strong recommendation)

A **full** global-grid clear-and-rebuild after every loop closure is too expensive for a live system. The standard fix (Voxgraph, c-blox, and the nvblox mapper pattern) is **submapping**:

- Maintain many **small bounded TSDF submaps**, each anchored to a keyframe pose `T_k` and integrating only the clouds near it.
- A submap's *internal* voxels are fixed relative to its anchor pose; a loop closure moves the **anchor poses** (rigid transforms of whole submaps), which is cheap and needs **no re-integration** for unaffected submaps.
- Only submaps whose **anchor pose changed** (or that must be re-fused because their constituent keyframes moved relative to each other) are cleared and rebuilt; the rest are just re-placed.
- The display/archival mesh is the union of submap meshes; overlapping regions are blended or the higher-confidence submap wins.

This bounds rebuild cost to "a handful of submaps" instead of "the whole map", and is the recommended structure for Meridian given iSAM2 will issue frequent pose updates.

### 8.4 How the keyframe store feeds each library

- **nvblox:** call `clear()` on the affected layers (or recreate the submap mapper), then re-run `integrateDepth/integrateColor` for the corrected keyframes from the retained clouds (project clouds to depth or use the point-cloud integrator), then `updateMesh()`/`updateEsdf()`. All on GPU; with submapping only dirty submaps are touched.
- **VDBFusion:** clear/recreate the affected `openvdb::FloatGrid`(s) and re-run `integrate(points, origin)` for the corrected keyframes, then `extract_triangle_mesh()`. OpenVDB's cheap allocation/clearing makes per-submap rebuilds light.
- **Voxblox:** clear the `TsdfLayer` (or per-submap layers as in Voxgraph/c-blox), re-`integratePointCloud` at corrected poses, then re-run the `MeshIntegrator` and `EsdfIntegrator` on dirty blocks.

### 8.5 Consequences for the keyframe store design (spec these explicitly)

- The store is **first-class and persistent**, not a transient ring buffer; it is the only authoritative record of geometry.
- Store clouds in **body/keyframe frame** so they re-project correctly under any corrected `T_k`.
- Key its lifetime to the pose graph: a keyframe's cloud lives as long as its node in iSAM2.
- For km-scale, the store itself must be **disk-backed / streamable** (it can exceed RAM); load on demand during rebuilds of an active window.
- Decide **rebuild granularity** (whole-map vs submap vs touched-blocks) and the **pose-change threshold** that triggers a rebuild — these are tuning parameters with real latency impact.

---

## 9. Failure modes catalogue

- **Pose drift / mis-registration** → "double walls", ghosting, smeared geometry. Largely the front-end/back-end's fault, but exposed as TSDF artefacts; mitigated by clear-and-rebuild after loop closure (§8).
- **Projective-integrator silhouette artefacts** → spurious surface at depth discontinuities (voxel projects onto wrong pixel). Mitigate: raycasting/bundled integrator, or per-pixel depth-gradient rejection.
- **Truncation too small** → noisy/holey surface, thin ESDF fixed band; **too large** → over-smoothed, bridged thin gaps, more compute. Keep δ = 3–5 voxels.
- **`max_weight` too high** → stale map, dynamic objects "burned in"; **too low** → flickering, noisy surface. Tune to scene dynamics.
- **MC topological ambiguity** → cracks/holes; use **MC33 + asymptotic decider** if watertight live mesh matters.
- **Colour bleeding** → grazing-angle colour written far from surface; restrict colour blend to |D| ≪ δ and use a colour weight.
- **VRAM exhaustion (nvblox)** → unbounded outdoor extent overruns GPU memory; use coarser voxels, an active window, or fall back to VDBFusion for km-scale.
- **Poisson hallucination** → watertight surface invented in unobserved regions; always **density-trim** and supply oriented normals.
- **Stale ESDF after rebuild** → ESDF must be rebuilt whenever its TSDF is cleared-and-rebuilt; the incremental queues do not catch global pose shifts.
- **Dynamic objects** → leave trailing "tube" artefacts without space carving; enable carving or a freespace/dynamics layer (nvblox supports a freespace/dynamic-detection pipeline).

---

## 10. Concrete recommendations for Meridian

1. **Online fusion + meshing: nvblox** if a CUDA GPU is in the target (Jetson Orin / x86+RTX). One library covers GPU TSDF (projective) + GPU colour + GPU Marching Cubes + GPU ESDF, with `isaac_ros_nvblox` as a maintained ROS 2 wrapper and `nvblox_torch` for any ML/torch integration. 8³ blocks in CUDA unified memory, SQLite serialisation. Best real-time colourised-mesh quality/effort ratio. Trade-off: **VRAM-bounded extent**, NVIDIA/CUDA lock-in.
2. **CPU fallback / km-scale outdoor: VDBFusion on OpenVDB.** OpenVDB hierarchical sparsity (8³ leaves, background+tiles) reaches kilometres with RAM ∝ surface; raycasting integrator with `ValueAccessor` caching; `tools::volumeToMesh` for MC. No ESDF (add separately if needed). Trade-off: CPU-bound throughput, no built-in colour fusion (extend the voxel type or carry colour on the keyframe clouds and bake at mesh time).
3. **ESDF (only if a planner needs it):** prefer nvblox's GPU ESDF; otherwise implement the **Voxblox incremental raise/lower wavefront** (the cleanest reference, §4). Don't build an ESDF you don't consume.
4. **Online mesh: Marching Cubes, ideally an MC33 implementation**, per-block/incremental, colourised from the colour field. Skip Dual Contouring unless a CAD/structured-scene mode is needed.
5. **Archival watertight mesh: Screened Poisson** (`mkazhdan/PoissonRecon` or CGAL `poisson_surface_reconstruction`) run **offline** on the retained keyframe oriented-point store after final trajectory optimisation. Use `--depth 10–11`, `--pointWeight ~4`, and **density-trim**. Source normals from the TSDF gradient or per-point PCA oriented to the sensor.
6. **Keyframe cloud store is first-class, persistent, body-frame, disk-backable** (§8.5). It is the authoritative geometry record and the input to both the live rebuild and the Poisson pass.
7. **Adopt submapping** (Voxgraph/c-blox/nvblox-mapper pattern, §8.3) so loop-closure rebuilds touch only affected submaps, not the whole map. Define the **rebuild granularity** and **pose-change trigger threshold** as explicit, tunable parameters.
8. **Wrap the fusion engine behind a narrow interface** (`integrate(cloud, pose)`, `clearRegion(...)`, `getMesh()`, `getEsdfSlice()`), mirroring the front-end's "iEKF or B-spline behind one interface" pattern, so nvblox (GPU) and VDBFusion (CPU/km-scale) are swappable per deployment.

---

## 11. Source list

**Papers**
- Oleynikova, Taylor, Fehr, Siegwart, Nieto — *Voxblox: Incremental 3D Euclidean Signed Distance Fields for On-Board MAV Planning*, IROS 2017. arXiv:1611.03631. (TSDF integrators §III; incremental ESDF wavefront §IV; Eqs. 1–2 weighted average — **exact eq. numbering to verify against PDF**.)
- Curless & Levoy — *A Volumetric Method for Building Complex Models from Range Images*, SIGGRAPH 1996. (Origin of cumulative weighted-average TSDF.)
- Newcombe et al. — *KinectFusion*, ISMAR 2011. (Real-time GPU TSDF + projective integration + MC.)
- Lorensen & Cline — *Marching Cubes: A High Resolution 3D Surface Construction Algorithm*, SIGGRAPH 1987. (256→15 cases, edge/triangle tables, edge interpolation Eq. MC-1.)
- Nielson & Hamann — *The Asymptotic Decider*, IEEE Vis 1991; Chernyaev — *Marching Cubes 33*, 1995. (Ambiguity fixes.)
- Ju, Losasso, Schaefer, Warren — *Dual Contouring of Hermite Data*, SIGGRAPH 2002. (DC + QEF Eq. DC-1.)
- Kazhdan & Hoppe — *Screened Poisson Surface Reconstruction*, ACM ToG 2013. (Poisson Eq. POIS-1, screening Eq. POIS-2, octree B-spline multigrid; `--depth`/`--pointWeight`/trim.)
- Vizzo et al. — *VDBFusion: Flexible and Efficient TSDF Integration of Range Sensor Data*, Sensors 2022 (MDPI 1424-8220/22/3/1296). (OpenVDB-based raycast TSDF, km-scale LiDAR.)
- Millane, Oleynikova, Wirbel, Steiner, Ramasamy, Tingdahl, Siegwart (NVIDIA / ETH ASL) — *nvblox: GPU-Accelerated Incremental Signed Distance Field Mapping*, **ICRA 2024, arXiv:2311.00626** (v2 2024-03-15). Reports up to **177× speed-up in surface reconstruction** and **31× in distance-field (ESDF) computation** vs CPU state-of-the-art; open-source in ROS 1 and ROS 2. (Verified.) Detailed defaults still best read from the `nvidia-isaac/nvblox` repo config headers, which evolve per release.

**Repositories / docs**
- `ethz-asl/voxblox` (+ `voxblox_ros`) — `simple`/`merged`/`fast` integrators; `tsdf_voxel_size`, `tsdf_voxels_per_side`=16, `truncation_distance`, `max_weight`, `voxel_carving_enabled`, `use_const_weight`; `TsdfMap/Layer/Block/TsdfVoxel`; `EsdfIntegrator`, `MeshIntegrator`.
- `nvidia-isaac/nvblox` (+ `isaac_ros_nvblox`, `nvblox_torch`) — GPU TSDF/ESDF/colour/mesh/occupancy/freespace; 8³ `VoxelBlock`; CUDA unified memory; SQLite serialisation; `ProjectiveTsdfIntegrator`, `ProjectiveColorIntegrator`, `EsdfIntegrator`, `MeshIntegrator`.
- `PRBonn/vdbfusion` — OpenVDB raycast TSDF; `voxel_size`, `sdf_trunc`, `space_carving`; `extract_triangle_mesh`.
- `AcademySoftwareFoundation/openvdb` — VDB tree (5-4-3 ⇒ 8³ leaves), background+tiles, `tree::ValueAccessor`, `tools::volumeToMesh`; **NanoVDB** (GPU, pointerless contiguous buffer).
- `mkazhdan/PoissonRecon` — reference Screened Poisson; CGAL `Poisson_reconstruction_function` / `poisson_surface_reconstruction` as a library alternative.

**Verification flags (confirm before formal spec):**
- Voxblox exact equation numbers (Eqs. 1–2) and the precise weight-function constants (PDF was image-only during research; read from `helenol.github.io/publications/iros_2017_voxblox.pdf`).
- nvblox paper citation is **verified** (arXiv:2311.00626, ICRA 2024); current default `voxel_size`/`truncation_distance_vox`/`max_integration_distance_m`/`max_weight` (read from repo config headers — they evolve per release; latest tag seen during research was v0.0.10).
- Screened-Poisson default `--pointWeight` (≈4) and `--samplesPerNode` against the current `PoissonRecon` release.
- VDBFusion default `voxel_size`/`sdf_trunc` per dataset config.
