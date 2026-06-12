# 06 — Map Layer: Registration Voxel-Hash + nvblox GPU TSDF/Colour/Mesh (L4)

> **Spec status:** normative *implementation* spec. This document fixes the
> internals of layer **L4** — the map. It is consistent with, and subordinate to,
> `00_architecture.md` (layer roles, threading, debug bus, deferred seams) and
> `01_interfaces_and_data_types.md` (the boundary value types: `KeyframePacket`,
> `GraphUpdate`, `KeyframeStore`, `IMapLayer`, `Pose`, `LidarPoint`). Where this
> spec re-states a type from `01`, that re-statement is *descriptive*; the
> authoritative declaration lives in `01`. `PlaneHit` and `ColorMesh` are the
> canonical spec 01 boundary types (used, not redefined, here); new types
> introduced here (`Aabb`, `MapDiagnostics`, the voxel-hash / nvblox-wrapper
> internals) are declared here and, if they ever cross an L4 boundary, are
> mirrored into `01`.
>
> **System framing (supersedes any earlier phased plan).** Meridian is **one
> complete system** — a discrete, tightly-coupled LiDAR-Inertial estimator
> feeding an iSAM2 back-end and a **GPU nvblox** map. There is no
> "ship a simpler map first" milestone and no v1/v2 split anywhere in L4. The
> deployment target is an **NVIDIA Jetson Orin with a CUDA GPU always present**;
> the surface map is **nvblox, GPU-only, with no CPU fallback path** (spec 00 §0,
> §9.5). The bring-up order in `00_architecture.md §13` is module integration
> order for this one system, not a feature roadmap.
>
> **Scope.** First-pass scope ends at a **colourised triangle mesh** (nvblox
> Marching Cubes). The ESDF (path planning) and semantic/label channels are
> *designed-for* and called out as **deferred hooks** (§11), but are **not built
> now**. Nothing in this spec may require an ESDF or a semantic label to function.
>
> **Grounding.** The surface/mesh tier is grounded in **nvblox** (GPU TSDF + GPU
> colour + GPU Marching Cubes); the fusion-equation, Marching-Cubes and ESDF
> reference digests are in Appendix R. The clear-and-rebuild contract is the
> load-bearing constraint of any running-average TSDF: it is not per-keyframe
> reversible, so loop correction is *clear and re-integrate from retained clouds at
> corrected poses*, never per-voxel subtraction (the argument is in §7.4). The
> registration tier is grounded in FAST-LIO's incremental
> k-d tree, `slam-reference/FAST_LIO/include/ikd-Tree/ikd_Tree.{h,cpp}`, and the
> ikd-Tree paper `slam-reference/papers/2102.10808.txt`; we keep its incremental
> insert / box-delete / nearest-neighbour *behaviour* and replace its data layout
> with an adaptive voxel hash for the reasons in §3.1.
>
> **Notation** follows the shared block (spec 01 §1): transform $T_{A\_B}\in
> SE(3)$ maps a point from frame $B$ to frame $A$ ($p_A=T_{A\_B}\,p_B$); time is
> `Timestamp = int64` ns; ownership categories *Value / Owned-unique /
> Shared-immutable / Borrowed* and the thread-crossing marker **[TS]** are as in
> spec 01 §2.4.

---

## Table of contents

1. [What L4 is, and the two-tier mental model](#1-what-l4-is-and-the-two-tier-mental-model)
2. [Inputs, outputs, threading, lifetime](#2-inputs-outputs-threading-lifetime)
3. [Tier R — adaptive voxel-hash registration map](#3-tier-r--adaptive-voxel-hash-registration-map)
4. [Tier S — nvblox GPU TSDF + RGB surface map](#4-tier-s--nvblox-gpu-tsdf--rgb-surface-map)
5. [The mesh stage — nvblox Marching Cubes + colour](#5-the-mesh-stage--nvblox-marching-cubes--colour)
6. [The retained per-keyframe cloud store (MUST-FIX #4)](#6-the-retained-per-keyframe-cloud-store-must-fix-4)
7. [Loop-closure de-integration: clear-and-rebuild (MUST-FIX #4)](#7-loop-closure-de-integration-clear-and-rebuild-must-fix-4)
8. [Interfaces, factories, configuration](#8-interfaces-factories-configuration)
9. [Failure modes & recovery](#9-failure-modes--recovery)
10. [Deferred-but-designed hooks: ESDF, semantics, Poisson export](#10-deferred-but-designed-hooks-esdf-semantics-poisson-export)
11. [Debug / introspection for L4](#11-debug--introspection-for-l4)
12. [Parameter reference](#12-parameter-reference)

---

## 1. What L4 is, and the two-tier mental model

L4 turns a stream of *pose-tagged, deskewed point clouds* (plus RGB) into (a) a
structure the front-end can register against in real time and (b) a colourised
triangle mesh for the operator. Crucially, it must do this while the back-end
keeps **moving keyframe poses underneath it** (loop closure, global
optimisation): the map is not write-once, it is *correctable*.

A single structure cannot serve both of L4's jobs well, because they have
contradictory requirements:

| Job | Needs | Hurt by |
|---|---|---|
| Front-end registration (point-to-plane) | fast local k-NN / plane query, cheap incremental insert near the robot | a fixed global grid, full rebuilds, GPU round-trips on the hot path |
| Surface reconstruction | dense signed-distance + colour fusion, watertight extraction | sparse point sets with no free-space model |
| Loop correction | ability to *move* large regions of map to corrected poses | running-average fusion that cannot be un-mixed |

Meridian therefore splits L4 into **two cooperating tiers plus one store**, all
behind a single `IMapLayer` family of interfaces (§8):

```
                         KeyframePacket (pose + deskewed cloud_body + rgb)            GraphUpdate
                                   │                                                      │
            ┌──────────────────────┼──────────────────────────────────────────┐         │
            ▼                       ▼                                           ▼         ▼
   ┌─────────────────┐   ┌──────────────────────────┐            ┌────────────────────────────────┐
   │ Tier R          │   │ Tier S  (GPU)             │            │ KeyframeStore (§6)             │
   │ adaptive        │   │ nvblox TSDF + RGB          │            │ retained deskew clouds +       │
   │ VOXEL-HASH      │   │ running-average fusion      │◀──rebuild──│ poses + rgb, keyed by id       │
   │ (registration)  │   │  -> Marching Cubes mesh §5  │   region   │ (Shared-immutable; canonical)  │
   │ CPU, queried/scan│  └─────────────┬─────────────┘            └────────────────┬───────────────┘
   └────────┬────────┘                 │ ColorMesh                                │ in_region()
            │ plane query              ▼                                          ▼
            ▼                       L6 operator surface                    L5 GICP / re-integration
        L2 front-end
```

* **Tier R — registration voxel-hash** (§3): the hot, local, *queryable* CPU
  structure the front-end hits every scan. It replaces FAST-LIO's ikd-Tree (we
  keep its incremental-insert / box-delete / k-NN *behaviour*, change its
  *layout*). It carries cached per-voxel planes so the front-end's per-point query
  is a hash lookup, not a 5-NN + PCA.
* **Tier S — nvblox GPU TSDF+RGB surface** (§4): the dense signed-distance +
  colour field, running-average fused on the GPU by nvblox, from which nvblox
  extracts the mesh (§5). This is the **only** surface/map backend — there is no
  CPU fallback, no VDBFusion, no OpenVDB/NanoVDB archive, no second `IMapLayer`
  surface implementation (spec 00 §2.1, §9.5; spec 11 §7).
* **`KeyframeStore`** (§6): the *canonical retained cloud store* — the data
  structure MUST-FIX #4 requires. Both tiers are *derived* and *rebuildable* from
  this store; the store is the single source of truth for re-integration after a
  loop closes.

The discipline that makes loop closure tractable: **Tiers R and S are disposable
caches of the `KeyframeStore`.** Any region of either tier can be discarded and
rebuilt from the retained clouds at corrected poses (§7). This is the
architectural choice behind MUST-FIX #4, and it is *forced* on Tier S because
nvblox's running-average TSDF is mathematically non-invertible (§4.4 below).

> **Why no out-of-core archive tier.** An earlier design carried a third
> "global NanoVDB archive" tier for evicted registration voxels. That is removed:
> it is a defensive dual-path the simplicity mandate forbids (spec 00 §0, spec 11
> §11). For the benchmark sequences (Newer College, `DATASET.md`) Tier R's
> hot window plus the retained `KeyframeStore` are sufficient; long-mission RAM
> growth is handled by the store's deferred mmap hook (§6.4), not by a second
> voxel container with its own clear-region contract.

---

## 2. Inputs, outputs, threading, lifetime

### 2.1 Inputs

L4 consumes exactly two boundary values (spec 01 §6, §7.4):

1. **`KeyframePacket`** — but L4 only reads a *subset* of it:
   `id`, `stamp`, the corrected map-frame pose, `cloud_body`
   (Shared-immutable deskewed body-frame cloud), `image` + `T_body_cam` (for
   colour). L4 ignores the front-end's constraint/observability fields (those are
   L3's). **L4 never sees a raw scan and never deskews** — the cloud arrives
   already deskewed in the body frame at `stamp` (spec 01 §6.2, §6.4 item 5), so
   integrating it at a corrected pose is a single `Pose` multiply.

2. **`GraphUpdate`** (spec 01 §7.4) — the list of keyframes whose poses changed
   after an `IBackEnd::optimize()`, with their new `map`-frame poses. This is the
   *only* trigger for de-integration / rebuild (§7).

There is also a high-rate, **non-keyframe** path: the live `NavState`
the front-end emits every scan. L4 uses the live registered scan to
keep Tier R fresh *between* keyframes (so the front-end has something to register
against immediately), but **only `KeyframePacket` clouds enter the
`KeyframeStore` and Tier S** — see §3.4 for why the live path is "best-effort
scratch" and the keyframe path is "canonical".

### 2.2 Outputs

| Consumer | Value | Mechanism |
|---|---|---|
| L2 front-end | `PlaneHit` (nearest plane to a query point) | `IRegistrationMap::query_plane` (§3.5), synchronous, hot, CPU |
| L6 operator surface | `ColorMesh` (colourised triangles) | `ISurfaceMap::extract_mesh` (§5), on-demand/throttled; the standing host mesh is updated by streaming only changed blocks, and the L4→L6 `IMeshExtractor` boundary hands L6 a moved snapshot of it (spec 01 §7.7, §9) |
| L5 place recognition | retained clouds in a region | `KeyframeStore::in_region` (§6), Shared-immutable |
| Debug bus | timing/scalars/markers | `TelemetrySink` (§11) |

### 2.3 Threading & lifetime (consistent with arch §11)

* **Map/integrate thread (T4)** owns Tier R, Tier S (the host side of nvblox)
  and the `KeyframeStore`. `integrate`, `apply_graph_update`, the live-scan
  refresh, and all nvblox kernel *launches* run here, single-threaded internally
  (thread-confined, arch §11.2). nvblox kernels execute on the GPU; the host stage
  remains thread-confined and launches them from T4.
* **Mesh thread (T5)** runs `extract_mesh` off the critical path. nvblox meshes
  the dirty blocks on the GPU; T5 pulls the result back and hands a host-side
  `ColorMesh` to L6, so meshing never blocks integration (arch §11.1 diagram, §11.2).
* **`Q_map` is lossless** (arch §11.2): losing a keyframe cloud corrupts the map
  permanently, so back-pressure is correct here even though `Q_meas` is lossy.
* **`query_plane` is called from the front-end thread (T2)**, *not* T4. Tier R
  must therefore be safe for concurrent read (front-end) while T4 inserts. The
  concrete concurrency design is in §3.6 (per-bucket sharded locks + an
  insert-only fast path), chosen so the front-end's hot query never blocks on a
  map insert. **The front-end never touches the GPU on its hot path** — Tier R is
  CPU-resident precisely so per-point plane queries are not GPU round-trips.
* **Lifetime / ownership.** `cloud_body` arrives Shared-immutable; the
  `KeyframeStore` is the *retention owner* (it deliberately keeps clouds alive for
  the mission). Tier R holds only derived voxels (plus weak references into the
  store's clouds for re-query); Tier S holds only nvblox's GPU TSDF/colour blocks.
  Both are independent of any single store cloud once integrated, so the rebuild
  in §7 is the only coupling.

---

## 3. Tier R — adaptive voxel-hash registration map

### 3.1 Why a voxel-hash, not the ikd-Tree

FAST-LIO registers against an **incremental k-d tree** (`ikd_Tree.h`:`KD_TREE`),
which gives it: incremental point insertion with on-insert voxel downsampling
(`KD_TREE::Add_Points` with the `downsample_switch` / `downsample_size`
parameter), box deletion of the region leaving the local map
(`KD_TREE::Delete_Point_Boxes`, used by FAST-LIO's `lasermap_fov_segment` sliding
local map), k-NN search for the plane fit (`KD_TREE::Nearest_Search`), and
amortised rebalancing of subtrees that grow too unbalanced or accumulate too many
lazily-deleted nodes (the paper's *scapegoat*-style criterion on
`balance_criterion` / `delete_criterion`, ikd-Tree paper §III, `2102.10808.txt`).
The lazy delete uses per-node `deleted` / `tree_deleted` flags and pushes the
heavy rebuild onto a background thread (`KD_TREE`:`Rebuild_Pointcloud_Thread`).

The ikd-Tree is excellent and we **keep its operational contract** — incremental
insert, *box delete*, k-NN, background rebalance, voxel downsampling on insert.
We change the **underlying container** to a **flat spatial hash of voxels**
because, for *our* workload (large point counts per scan, online extrinsics, and
— decisively — **loop-closure region rebuilds**), a hash beats the tree on two
axes:

1. **O(1) region clear.** MUST-FIX #4 needs "clear all voxels in an AABB" to be
   cheap and exact. In a hash that is a bucket scan + erase; in a k-d tree it is a
   `Delete_Point_Boxes` that marks nodes deleted and *defers* the structural
   reclaim to a rebuild — fine for a sliding FOV, wasteful when we clear-and-
   rebuild a large loop region (§7).
2. **No rebalancing pathology.** The tree's `criterion_param` /
   `balance_criterion` rebuilds are amortised-cheap on a *streaming* FOV but can
   spike when a loop dumps a corrected region back in; a hash has no global
   structural invariant to repair.

We pay for this with slightly worse worst-case k-NN locality than a balanced
tree, which §3.3 mitigates with an *adaptive* voxel resolution and a small
per-voxel plane cache so the common query is O(1) and never touches raw points.

> **Single LiDAR.** Tier R indexes one LiDAR's deskewed points; there is no
> per-sensor residual stream and no merge-of-N-LiDARs. A second LiDAR, if ever
> added, would attach behind the same `ISensorSource`/extrinsic machinery as one
> more measurement stream into the front-end (spec 00 §12) — a future extension,
> not a current contract.

### 3.2 Data structure

```cpp
// ---- voxel coordinate key ----
struct VoxelKey {              // integer voxel coordinate at the voxel's OWN level
  std::int32_t x, y, z;
  bool operator==(const VoxelKey&) const noexcept = default;
};
struct VoxelKeyHash {          // spatial hash (the FAST-LIVO2 / Loam-style mix)
  std::size_t operator()(const VoxelKey& k) const noexcept {
    // Teschner et al. (2003) spatial hash (three large primes, XOR-mixed).
    // NB: this is NOT FAST-LIVO2's hash — voxel_map.h:30,115 uses a single-prime
    // polynomial form (P=116101: ((s.z*P)%N + s.y)*P%N + s.x). We use Teschner's
    // because nvblox owns the registration voxel index; this hash is for our own
    // CPU-side query cache only.
    return (std::size_t(k.x) * 73856093) ^ (std::size_t(k.y) * 19349663)
         ^ (std::size_t(k.z) * 83492791);
  }
};

// ---- per-voxel payload ----
struct RegVoxel {
  // Adaptive occupancy: a voxel holds a SMALL bounded set of representative
  // points (capacity Nv, default 20) rather than every point, like ikd-Tree's
  // on-insert downsample (Add_Points/downsample_size). Once full, points are
  // replaced by deterministic reservoir sampling (§3.2a), not newest-wins.
  std::array<Eigen::Vector3f, kMaxPtsPerVoxel> pts;  // map-frame points
  std::uint8_t  n = 0;                                // valid count
  std::uint32_t seen = 0;                             // total points offered (reservoir counter)

  // Cached local plane (n·x + d = 0), refit lazily when `dirty` && n>=kMinPlanePts.
  Eigen::Vector3f plane_n {0,0,0};
  float           plane_d = 0.f;
  float           plane_rms = std::numeric_limits<float>::infinity(); // fit residual
  bool            is_plane = false;   // passed planarity test (§3.3)
  bool            dirty = true;       // needs refit

  std::uint8_t    level = 0;          // adaptive subdivision level (§3.3)
  Timestamp       last_touch = 0;     // for hot-window pruning (§3.4)
};

class VoxelHashMap final : public IRegistrationMap {
  // Hot, local, queryable, CPU-resident. Sharded for concurrent read/insert (§3.6).
  struct Shard {
    std::unordered_map<VoxelKey, RegVoxel, VoxelKeyHash> voxels;
    mutable std::shared_mutex mtx;     // many readers (front-end) / one writer (T4)
  };
  std::array<Shard, kNShards> shards_;   // kNShards = 64, key.x low bits select
  float base_res_;                       // map.reg_voxel_m (e.g. 0.2 m)
  // ... config, telemetry, hot-window AABB ...
};
```

`kMaxPtsPerVoxel` (default 20) bounds memory per voxel and makes plane fitting
O(1); it is the direct analogue of ikd-Tree's per-voxel downsample cap
(`Add_Points` with `downsample_size` — one representative survives per downsample
cell). `kNShards` keeps the front-end's read lock contention low (§3.6).

### 3.2a Bounded-voxel eviction is deterministic (reservoir, not newest-wins)

Once a voxel holds `kMaxPtsPerVoxel` points, a further insert must drop one. A
**newest-wins** policy (overwrite the oldest slot) makes the surviving point set —
and therefore the cached plane — depend on the *order* points were offered, which
the multi-threaded live path (§3.4) does not fix; the same bag would yield a
different plane cache run to run, breaking the determinism mode (§9). Eviction is
therefore **reservoir sampling** with a per-process **seeded** RNG: the voxel keeps
a `seen` counter, and the $k$-th offered point ($k>$`kMaxPtsPerVoxel`) replaces a
uniformly-chosen existing slot with probability `kMaxPtsPerVoxel`$/k$, else is
dropped. This keeps a uniform sample of all points ever offered to the voxel
(unbiased toward early *or* late returns) and, because the RNG is seeded from a
fixed config seed (`map.reg_seed`, default 0) and advanced in keyframe-insert order
on T4, the retained set is **reproducible** across runs of the same bag in
single-thread replay. The reservoir draw is the only randomness in Tier R and is
confined to the eviction decision, never to the query path.

### 3.3 Adaptive resolution & the plane cache

"Adaptive" means the effective voxel size **varies with local geometry**, the
idea FAST-LIVO2's `voxel_map` uses (`voxel_map.cpp`: octree subdivision of a root
voxel when the points inside are not well-explained by a single plane):

* A root voxel at `base_res_` (e.g. 0.2 m) holds up to `kMaxPtsPerVoxel` points.
* When a refit gives `plane_rms > planarity_thresh` *and* `n` is large, the voxel
  is **subdivided** one `level` deeper (half the edge), redistributing its points
  into 8 child keys at `level+1`. This captures curved/cluttered geometry (foliage,
  rubble) at finer resolution while keeping large flat surfaces (walls, road) at
  coarse resolution — fewer voxels, better planes.
* Subdivision is capped at `max_level` (default 3 → finest 0.025 m at base 0.2 m)
  to bound depth. A voxel that cannot be made planar by `max_level` is flagged
  `is_plane = false` and is *excluded from plane queries* (the front-end falls
  back to point-to-point or simply gets no hit there).

**Plane fit** (lazy, on first query after `dirty`): given the voxel's points
$\{x_i\}_{i=1}^{n}$, compute the centroid $\bar{x}$ and the scatter matrix
$$
\Sigma = \frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})(x_i-\bar{x})^{\top}.
$$
The plane normal $\hat{n}$ is the eigenvector of $\Sigma$ with the smallest
eigenvalue $\lambda_0$; $d = -\hat{n}^\top\bar{x}$. Planarity is accepted iff
$$
\frac{\lambda_0}{\lambda_1} < \text{planarity\_thresh}\quad(\text{default }0.1),
\qquad \text{plane\_rms} = \sqrt{\lambda_0}.
$$
This is the standard PCA plane that FAST-LIO computes per-neighbourhood in
`esti_plane` (`laserMapping.cpp`:`esti_plane`); the difference is **we cache it
per voxel** so consecutive scans reuse the fit instead of re-running PCA on a
fresh k-NN set every point, every scan. The cache is invalidated (`dirty=true`)
whenever a point is inserted into or removed from the voxel.

### 3.4 The hot window, the live path, and what is canonical

Two write paths feed Tier R:

* **Live scratch path (best-effort, between keyframes).** Each registered scan's
  points (in map frame at the live `NavState`) are inserted to keep the local map
  dense for the *next* scan's registration — exactly FAST-LIO's behaviour of
  feeding the just-registered cloud back into the ikd-Tree
  (`KD_TREE::Add_Points` after the update). These points are **scratch**: they
  are *not* recorded in the `KeyframeStore`, and on a loop correction the affected
  region is rebuilt from canonical keyframe clouds (§7), discarding any scratch.
* **Canonical keyframe path.** When a `KeyframePacket` arrives, its
  `cloud_body` (already in the store, §6) is transformed by `T_map_body` and
  inserted, *and* its `(id → contributing VoxelKeys)` provenance is recorded
  (§3.7) so the region rebuild knows which voxels a moved keyframe touched.

**Hot window / FOV segmentation.** Tier R is bounded to an AABB around the robot
(`reg_hot_radius_m`, default 100 m), the analogue of FAST-LIO's
`lasermap_fov_segment` + `Delete_Point_Boxes` sliding local map. Voxels whose
center leaves the hot AABB are **pruned** from Tier R (their canonical geometry
remains in the `KeyframeStore`, the single source of truth, and is re-fetched and
re-inserted if the robot revisits). Pruning runs on T4 after each integrate,
batched, with an `event` telemetry (`map/prune`, §11). There is no separate
out-of-core voxel archive: the retained `KeyframeStore` *is* the durable record,
so a revisit re-integrates `store.in_region(...)` rather than thawing a second
serialised grid.

### 3.5 The query the front-end calls

```cpp
struct PlaneHit {
  Eigen::Vector3f n;       // unit normal, MAP frame
  float           d;       // plane offset: n·x + d = 0
  float           rms;     // fit residual (confidence)
  Eigen::Vector3f centroid;// voxel centroid (for residual weighting)
  std::uint8_t    n_pts;   // support
};

// Hot path: front-end thread (T2), CPU only. Returns the cached plane of the
// voxel containing p_map (and, if empty/non-planar, optionally the best of the
// 26 neighbours within one ring). NO k-NN over raw points in the common case;
// NO GPU access.
std::optional<PlaneHit> query_plane(const Eigen::Vector3f& p_map) const override;
```

The front-end forms the point-to-plane residual $r = \hat n^\top p_{\text{map}}+d$
(spec 04), the same residual FAST-LIO assembles in `h_share_model`. The win over
ikd-Tree: the plane is **precomputed and cached** per voxel, so the per-point
query is a hash lookup, not a 5-NN search + PCA. When the containing voxel is
empty or non-planar we optionally probe the 26 one-ring neighbours (config
`reg_neighbor_ring`, default 1) and return the lowest-`rms` planar hit; if none,
return `nullopt` and the front-end treats the point as unmatched (FAST-LIO's
`point_selected_surf=false` path).

### 3.6 Concurrency: front-end reads while T4 writes

The front-end queries Tier R at full scan rate on T2 while T4 inserts/prunes.
Design:

* **Sharded `shared_mutex`** (`kNShards=64`): a query takes a *read* lock on the
  one shard owning the key; an insert takes a *write* lock on its shard. Cross-
  shard operations (region clear, prune sweep) lock shards one at a time, never
  globally, so a query elsewhere is never blocked.
* **Read-mostly fast path.** `query_plane` reads `plane_n/plane_d/is_plane`; the
  refit-on-dirty is the *only* place a query would need a write lock. To avoid
  promoting the hot query to a writer, refits are done **eagerly on T4 at insert
  time** (insert sets `dirty`, then T4 refits before releasing the write lock), so
  the front-end always sees an up-to-date cached plane under a read lock.
* **No allocation under the front-end lock**: voxel storage is `std::array`
  inline (`kMaxPtsPerVoxel`), so insertion never reallocates a per-voxel buffer
  while a reader holds it.

### 3.7 Per-keyframe → voxel provenance (needed by §7)

For the region rebuild to be exact and cheap, Tier R records, per keyframe id, a
compact list of the `VoxelKey`s that keyframe's canonical cloud contributed to:

```cpp
// id -> the keys that keyframe touched (canonical path only). Roaring-bitmap or
// flat sorted vector of packed keys; ~tens of KB per keyframe at 0.2 m.
std::unordered_map<std::uint64_t, std::vector<VoxelKey>> kf_voxels_;
```

This lets §7 compute the *exact* affected voxel set as the union of the moved
keyframes' touched voxels, instead of clearing a loose bounding AABB and
re-inserting everything in it.

---

## 4. Tier S — nvblox GPU TSDF + RGB surface map

### 4.1 What it is

Tier S is the **dense surface representation**, and it is **nvblox on the GPU,
full stop** — the only surface/map backend in Meridian (spec 00 §2.1, §9.5; spec 11
§7). nvblox maintains a sparse, block-allocated grid of
**Truncated Signed Distance Function (TSDF)** voxels (signed distance + fusion
weight) and a separate **colour** layer, fused with **running-average** semantics
(§4.3), from which nvblox extracts the mesh (§5). Unlike Tier R (which
stores points + cached planes for *registration*), Tier S models **free space and
surface** so Marching Cubes can produce a watertight, colourised mesh.

There is **no CPU fallback and no second implementation.** nvblox runs on the
guaranteed-present Jetson Orin CUDA GPU; a missing GPU at runtime is a hard,
fail-fast configuration error, not a degraded mode (spec 00 §9.5). VDBFusion,
Voxblox, OpenVDB and NanoVDB were considered and rejected (spec 11 §11) — they do
not appear in the build or this spec except as the named alternatives we declined.

### 4.2 How Meridian drives nvblox

`meridian_map` links the nvblox **C++ core library** directly (not the
`isaac_ros_nvblox` ROS node — the core is ROS-agnostic, spec 00 R1, spec 11 §7.2).
The host-side wrapper class owns nvblox's mapper and its layers, and the only CUDA
Meridian *writes* is the cloud→depth projection that feeds nvblox's integrators
(spec 11 §7.3, `nvblox_integrate.cu`):

```cpp
// meridian/map/nvblox_surface_map.hpp  — Tier S host wrapper (consumed by L6)
class NvbloxSurfaceMap final : public ISurfaceMap {
  // Owns the nvblox layers + integrators (TSDF, colour, mesh) on the GPU device.
  // Mirrors the per-keyframe -> touched block set for §7 rebuild provenance.
  std::unordered_map<std::uint64_t, std::vector<BlockKey>> kf_blocks_;
  float voxel_m_, trunc_m_;  // truncation distance trunc_m = trunc_voxels * voxel_m
  // nvblox::Mapper / TsdfLayer / ColorLayer / MeshLayer handles ...
  // (8^3 voxel blocks in CUDA unified memory)
};
```

The constraint `tsdf_voxel_m ≤ reg_voxel_m` (arch §8.3, validated on load) holds:
surface detail (0.05 m) is finer than registration (0.2 m).

### 4.3 Integration: projective TSDF fusion with running average

Per `integrate(kf, T_map_body)`, the host wrapper projects the keyframe's
deskewed cloud into a depth image (and carries the RGB image), then launches
nvblox's **projective** integrators (the GPU-native one-update-per-voxel pass
nvblox is built around; projective vs raycast contrast in Appendix R.1). The math
nvblox applies, stated for reference:

**Geometric (TSDF) update.** For each measured point $p$ (range $\rho$ along the
ray from sensor origin $o$ in map frame), nvblox updates voxels in the truncation
band $[\rho-\tau,\ \rho+\tau]$ ($\tau=$`trunc_m`). For a voxel at map position
$x$, the projective signed distance is
$$
\mathrm{sdf}(x) = \rho - \lVert x - o\rVert,\qquad
d(x) = \mathrm{clamp}\big(\mathrm{sdf}(x),\,-\tau,\,+\tau\big),
$$
fused by the running average with measurement weight $w$:
$$
d \leftarrow \frac{W\,d + w\,d(x)}{W + w},\qquad W \leftarrow \min(W + w,\ W_{\max}).
$$
$W_{\max}$ (`tsdf_w_max`, default **8**) is deliberately **small** — a handful of
observations, not a hundred. Two forces set it low. First, **scene
responsiveness**: once $W$ reaches $W_{\max}$ each new observation carries weight
$w/(W_{\max}+w)$, so a small cap keeps a stale voxel correctable by fresh geometry
— a transient or slowly-displaced surface (a parked car that leaves, a door that
opens) fades within a few sweeps instead of being burned in for the mission. A
large cap (the rejected $W_{\max}=100$) makes a fused voxel almost immovable: each
new measurement moves it by under 1 %, so dynamics ghost and never clear. Second,
**loop correction is owned by clear-and-rebuild, not by re-fusion** (§7): the map
does not depend on a high $W_{\max}$ to absorb a pose snap, because the affected
blocks are cleared and re-integrated from scratch at corrected poses. The weight
cap therefore exists only to damp per-sweep range noise on a *static* surface
(roughly $1/\sqrt{W_{\max}}$ noise reduction at saturation), and ~5–10 is enough
for that; pushing it higher trades responsiveness for a noise-reduction floor the
LiDAR does not need. The first pass uses a constant LiDAR measurement weight $w=1$
inside the band, $0$ outside (the constant-weight choice for range-independent
LiDAR returns; weight models in Appendix R.4).

**Integration distance is the map's own cutoff, decoupled from preprocessing.**
Fusion ignores any point whose range exceeds `tsdf_max_integration_dist_m`
(default **50 m**). This is a **distinct** parameter from preprocessing's
`det_range` (spec 03, default 120 m): `det_range` governs which returns the
*front-end* registers against (it wants long-range structure for odometry), while
`tsdf_max_integration_dist_m` governs how far the *dense surface* is fused — and
the surface must be cut much closer because TSDF VRAM scales with the integrated
volume. Allocated 8³ blocks grow roughly with the swept volume out to the cutoff
($\propto r^2$ along the trajectory for a ground vehicle), so a 120 m surface
cutoff would allocate ~5–6× the blocks of a 50 m one for geometry the operator
mesh does not need at that range and that the LiDAR resolves too sparsely to mesh
well. Setting the surface cutoff to ~40–60 m keeps the dense map dense where it is
useful and is the **primary VRAM control** alongside `tsdf_voxel_m` and
`reg_hot_radius_m` (the GPU-OOM remedy, §9). Tier R is unaffected — it indexes the
full `det_range` cloud, since CPU voxels are cheap relative to GPU TSDF blocks.

**Colour update.** nvblox's colour integrator projects each surface voxel into the
keyframe image using `T_body_cam` (carried in the packet so colourisation is
reproducible after online extrinsic refinement, spec 01 §6.2 item 6) and the
camera intrinsics: $u = \pi(K_{cam}, T_{cam\_body}\,T_{body\_map}\,x)$. A voxel is
coloured only when three conditions hold: $u$ is in-bounds, the voxel is near the
surface ($|d| \le$ truncation), and the voxel is **not occluded** from the camera.
Occlusion is rejected by a **sphere trace** from the camera origin toward $x$
through the TSDF — if the trace crosses the zero level set before reaching $x$,
some nearer surface lies between camera and voxel, so the sampled pixel belongs to
that nearer surface and must not be painted onto $x$. Skipping the occlusion test
bleeds foreground colour onto background geometry at every depth discontinuity (a
wall behind a pole takes the pole's colour), so the test is on by default
(`color_occlusion_check`, default `true`).

Colour fusion is an **exponentially-weighted moving average (EWMA)** toward the
newest accepted observation, not the running mean used for geometry:
$$
\mathbf{c} \leftarrow (1-\alpha)\,\mathbf{c} + \alpha\,\mathbf{c}_{\text{obs}},
\qquad \alpha = \texttt{color\_ewma\_alpha}\ (\text{default } 0.8),
$$
applied per channel in linear-light space. The colour channel is therefore weighted
**independently of the TSDF weight** — geometry may be well-observed from many
sweeps while colour is seen from few, and a high $\alpha$ keeps the surface tint
tracking the most recent good view (correct exposure, no glare) rather than
averaging toward a muddy mean over a long revisit. Before fusion the image has
exposure/gain normalisation applied (spec 01 §4.3); the first observation of a
voxel ($\mathbf{c}$ unset) is taken verbatim ($\alpha=1$ effectively).

**Provenance.** Every nvblox block touched by keyframe `kf.id` is appended to
`kf_blocks_[kf.id]` (§7 needs this). We read the touched-block set from nvblox's
per-integration block-update report.

### 4.4 Why running-average forces clear-and-rebuild

The running average $d \leftarrow (Wd + w d')/(W+w)$ is **not invertible**: once a
voxel has fused observations from several keyframes you cannot exactly *subtract*
one keyframe's contribution (nvblox does not retain the per-keyframe terms, and
the order/weights are entangled). This is the precise technical reason MUST-FIX #4
mandates **clear-and-rebuild**, not per-voxel de-integration — detailed in §7.4.

### 4.5 Designed-but-deferred: free-space decay of stale geometry (post-MVP)

A long mission accumulates two classes of stale TSDF: **far-range range outliers**
(atmospheric returns, rain, a momentary spurious echo that fused a thin sliver of
surface in free space) and **slowly-displaced scene geometry** the small $W_{\max}$
(§4.3) does not fully clear because the voxel stops being re-observed at all. Two
optional, off-by-default mechanisms address these; both are **specified now, built
post-MVP**, and neither runs in the first pass.

* **`invalid_depth_decay`** (config, default `false`). When enabled, a voxel that
  projects **inside** a later keyframe's depth image but with **no return at or
  beyond its range** (i.e. the ray passed through where the voxel claims surface)
  has its weight decremented by a fixed step each such observation, and is freed
  when $W$ reaches zero. This is a **reproducible** prune: the decrement is driven
  only by the keyframe stream, applied in keyframe order, so a `--single-thread`
  replay produces the identical pruned set (the determinism requirement of §9). It
  prunes the far-range outlier class — geometry that free-space evidence
  contradicts — without touching well-observed surface. It is **never** a substitute
  for clear-and-rebuild on a loop correction (§7); it is a maintenance prune that
  rides the ordinary `integrate` path.

* **`decayTsdfExcludeLastView`** stays **optional and never-default.** nvblox offers
  a global TSDF decay that uniformly fades every voxel each step (excluding the most
  recent view). Meridian does **not** enable it: a uniform global decay mutates the
  canonical-looking surface in a way the `KeyframeStore` cannot reproduce by
  re-integration (a rebuilt region would not match a decayed one), which violates
  the invariant that Tiers R and S are pure caches of the store (§1). It is exposed
  only as an explicit long-mission operator tool, never wired into the default
  pipeline and never part of the rebuild path.

### 5.1 Output type

`ColorMesh` is the canonical L4→L6 boundary type (spec 01 §7.7); `IMeshExtractor::extract`
returns it (moved to L6). It is the host-side mirror of nvblox's `MeshLayer`. The
host mirror is a **standing** mesh maintained incrementally — each cycle streams in
only the changed blocks (§5.4) rather than re-copying the whole `MeshLayer` out of
GPU memory — and the L4→L6 boundary hands L6 a snapshot of it:

```cpp
// Canonical declaration in spec 01 §7.7 / Appendix A — repeated for reference.
struct ColorMesh {
  std::vector<Eigen::Vector3f> vertices;     // map frame
  std::vector<Eigen::Vector3f> normals;      // per-vertex
  std::vector<std::array<std::uint8_t,3>> colors;  // per-vertex RGB (from colour layer)
  std::vector<std::uint32_t>   indices;      // triangle list (3 per tri)
  std::vector<float>           confidence;   // per-vertex [0,1] (§5.3, L6 overlay)
  Timestamp                    stamp = 0;    // extraction time
};
```

### 5.2 Algorithm

nvblox runs **Marching Cubes over the TSDF on the GPU**; we do not reimplement it.
For reference, nvblox's mesher (the MC ambiguity → MC33 background is in
Appendix R.3):

1. For each allocated TSDF block (and its + neighbours, so cube corners that span
   a block boundary are handled), iterates cubes of 8 adjacent voxels.
2. Classifies each corner by the sign of `distance` (skipping cubes with any
   zero-weight corner — unobserved). The 8 signs index the Marching Cubes case
   table (256 configurations → 15 symmetry-reduced base cases).
3. For each edge crossing the zero level set, places a vertex by **linear
   interpolation** of the SDF: for edge endpoints $x_a,x_b$ with distances
   $d_a,d_b$ of opposite sign, $x_v = x_a + \frac{d_a}{d_a-d_b}(x_b-x_a)$.
4. Normal = normalised TSDF gradient $\nabla d$ (central differences), or the
   triangle normal as fallback.
5. Colours the vertex by sampling nvblox's colour layer at $x_v$.

The mesh is **open** at unobserved boundaries (no hole-filling) — exactly what is
wanted for the online, interactive, colourised mesh. Watertight hole-filling is the
deferred offline Poisson export utility (§10), not this path.

### 5.3 Per-vertex confidence (feeds L6 overlay)

The operator surface (L6) tints the mesh by confidence. Confidence per vertex is
a monotone map of the local **TSDF weight** $W$ (more observations → more
confident) and, optionally, the **pose covariance** of the contributing
keyframes (a keyframe with loose covariance taints its surface). First pass:
$$
c = \mathrm{sat}\!\left(\frac{W}{W_{\text{conf}}}\right),
$$
clamped to $[0,1]$, with $W_{\text{conf}}=$ `mesh_conf_w` (default **8**) set equal
to `tsdf_w_max` so a fully-fused (saturated) voxel reads as full confidence and an
under-observed one is tinted down in proportion to its support. The pose-covariance
term is a deferred refinement hook (the data — `GraphUpdate::Moved::cov` — is
available, spec 01 §7.4).

### 5.4 Incremental meshing, streaming & threading

* **Block-incremental.** nvblox re-meshes only blocks whose TSDF changed since the
  last extraction (it tracks dirty blocks internally); a single re-integrated
  region re-meshes only its blocks. We mark a keyframe's blocks dirty in `§7`'s
  rebuild and let nvblox's mesher pick them up.
* **Streamed, not re-copied wholesale.** The runtime `extract_mesh` path returns a
  **delta**: it serialises only the changed mesh blocks via nvblox's
  `serializeSelectedLayers` over the dirty/cleared block set (the union of this
  cycle's integrated blocks and the `getClearedBlocks` set from any rebuild,
  §7.3a), and a host-side **layer streamer** applies that delta onto L6's standing
  `ColorMesh` block-for-block (new/updated block meshes replace their slots, cleared
  blocks drop out). Export cost therefore scales with the **changed** surface per
  cycle, not the total map size — a multi-kilometre map costs the same per cycle as
  a small one as long as the per-cycle change is bounded. This is what keeps mesh
  export affordable on the Orin over a long mission; a **full** mesh serialisation
  over every allocated block is **offline-only** (the §10 export utility / a
  one-shot end-of-mission dump), never the per-cycle runtime path.
* **Off the critical path.** The mesh path runs on T5: nvblox meshes the dirty
  blocks on the GPU and the streamer applies the delta to the host `ColorMesh`, so
  T4 integration is never stalled by meshing (arch §11.1, §11.2: meshing is off the
  front-end critical path).
* **On-demand / throttled.** Triggered by L6 request or a rate limit
  (`mesh.max_rate_hz`), never per-scan.

---

## 6. The retained per-keyframe cloud store (MUST-FIX #4)

### 6.1 Role and why it exists

This is the data structure the earlier design was *missing*. Three consumers need
the *same* per-keyframe clouds:

* **L4 re-integration** — rebuild map regions (Tier R + nvblox Tier S) at
  corrected poses after a loop (§7).
* **L5 GICP** — geometric verification of a loop candidate runs on the two
  keyframes' clouds (spec 01 §7.6, `IPlaceRecognizer::verify(..., store)`).
* **Optional offline Poisson export** — reads canonical clouds for the watertight
  archival mesh utility (§10).

A single retained store with `shared_ptr<const>` clouds gives all three the same
bytes and one lifetime owner (spec 01 §2.4, R3, §7.5). It is also the durable
geometry record that lets Tier R prune its hot window without an out-of-core
voxel archive (§3.4).

### 6.2 Declaration (mirrors spec 01 §7.5)

```cpp
class KeyframeStore : public IKeyframeStore {
public:
  struct Entry {
    std::uint64_t id;
    Pose          T_map_body;     // CURRENT best pose; updated on GraphUpdate
    std::shared_ptr<const std::vector<LidarPoint>> cloud_body;   // deskewed, body frame @ stamp
    std::shared_ptr<const CameraFrame>             image;        // optional RGB
    Pose          T_body_cam;     // extrinsic snapshot used at this KF (spec 01 §6.2)
    Aabb          bounds_map;     // cached map-frame AABB of the cloud at T_map_body
  };
  void put(Entry e) override;                                    // T4, on first integrate
  std::optional<Entry> get(std::uint64_t id) const override;
  std::vector<Entry> in_region(const Aabb& region) const override;  // L5 + rebuild
  void update_pose(std::uint64_t id, const Pose& T_map_body) override; // on GraphUpdate
  std::size_t size() const override;
private:
  std::unordered_map<std::uint64_t, Entry> entries_;
  // spatial index over bounds_map for in_region (grid of id-lists), refreshed on update_pose
  mutable std::shared_mutex mtx_;   // T4 writes, L5 reads concurrently
};
```

### 6.3 Invariants

* **The cloud is stored once.** `cloud_body` is the *same* `shared_ptr` the
  front-end created during deskew and put in the `KeyframePacket`; the store
  retains it, the graph node references it, L5 references it — no copy (spec 01
  §6.3).
* **`T_map_body` is the single mutable field.** It is updated *only* by
  `update_pose` on a `GraphUpdate`. The cloud bytes are immutable for the mission.
* **`bounds_map` is derived** from `T_map_body` and the cloud's body-frame AABB;
  recompute on `update_pose` so `in_region` stays correct after a loop.

### 6.4 Memory budget (first-pass scope)

First pass **retains all keyframe clouds in RAM**. Budget sanity: a keyframe at
`keyframe.dist_m = 1.0` m spacing, deskewed+downsampled to ~10–20k points × 16 B
(`LidarPoint` xyz+intensity+meta) ≈ 0.2–0.3 MB/keyframe; a 2 km trajectory ≈
2000 keyframes ≈ 0.4–0.6 GB. The Newer College benchmark sequences (`DATASET.md`)
are all shorter than that, so first-pass RAM retention is comfortably in budget.

This is the one component of L4 that is **deliberately unbounded** — the store is
the canonical source of truth, so it cannot evict clouds the way the derived tiers
prune voxels (§3.4) or cap fusion weight (§4.3). Growth is therefore linear in
trajectory length and **must be watched**: `store.size()` and the resident byte
total are exported every cycle (`map/kf_count`, `map/store_bytes`, §11), and the
failure-mode table (§9) routes "RAM pressure from retained clouds" to the on-disk
store. The growth is **bounded only by keyframe spacing** — a denser
`keyframe.dist_m` (spec 05) multiplies store RAM proportionally, which is the knob
to widen if a long mission approaches the host budget. The `IKeyframeStore`
interface allows an **mmap'd on-disk store** swap (arch §5.2 roster) for longer
missions without touching consumers — a deferred hook, not first-pass work, and the
*single* place long-mission RAM is addressed (there is no voxel-level archive tier).
Every other L4 store is bounded by design: Tier R by its hot window
(`reg_hot_radius_m`), Tier S by `tsdf_max_integration_dist_m` and the GPU VRAM cap,
the host mesh by the streamed delta (§5.4). The retained-cloud store is the
intentional exception, made safe by being watched and swappable rather than
unbounded-and-silent.

---

## 7. Loop-closure de-integration: clear-and-rebuild (MUST-FIX #4)

### 7.1 The trigger

After `IBackEnd::optimize()`, L3 emits a `GraphUpdate` (spec 01 §7.4) listing the
keyframes whose `map`-frame poses changed and their new poses (and optional cov).
L4 consumes it on T4 via:

```cpp
void apply_graph_update(const GraphUpdate& update, const KeyframeStore& store) override;
```

`GraphUpdate::loop_closed == true` flags a large rigid snap (many keyframes
moved); small relinearisation deltas (a handful of keyframes nudged) take the same
code path with a smaller affected set.

### 7.2 The algorithm (exact, provenance-driven)

```text
apply_graph_update(update, store):
  if update.moved is empty: return

  # 1. Update canonical poses FIRST (store is the source of truth).
  for m in update.moved:
      store.update_pose(m.id, m.new_T_map_body)     # also refreshes bounds_map

  # 2. Compute the EXACT affected derived-voxel/block set from provenance (§3.7,§4.3),
  #    NOT a loose AABB. Affected = union over moved ids of touched keys.
  dirtyVox  = ∪_{m in update.moved} kf_voxels_[m.id]      # Tier R
  dirtyBlk  = ∪_{m in update.moved} kf_blocks_[m.id]      # Tier S (nvblox blocks)
  regionAabb = bounding_aabb(dirtyVox ∪ dirtyBlk)

  # 3. CLEAR those derived cells. Tier R erases at voxel granularity (its own hash);
  #    Tier S clears at nvblox's native 8^3-BLOCK granularity across ALL of its
  #    layers (TSDF + colour + mesh) for the touched blocks (§7.3a).
  for k in dirtyVox: tierR.erase(k);   tierR.forget_provenance(moved ids)
  clearedBlk = tierS.clear_blocks(dirtyBlk)   # clears TSDF+colour+mesh; returns blocks actually cleared
  tierS.forget_provenance(moved ids)

  # 4. Determine the FULL rebuild set: every keyframe whose cloud overlaps the
  #    cleared region, not just the moved ones (a stationary KF may have co-fused
  #    voxels with a moved KF; those voxels were cleared and must be rebuilt).
  rebuildIds = { e.id for e in store.in_region(regionAabb) }   # superset of moved

  # 5. RE-INTEGRATE from the retained store at CORRECTED poses, in TWO passes.
  emit telemetry event "map/region_rebuild" with regionAabb, |rebuildIds|
  order = stamp-ascending(rebuildIds)             # deterministic, oldest->newest
  #  5a. GEOMETRY PASS: rebuild all TSDF + Tier R voxels first, so the truncation
  #      band and zero level set in the region are correct before any colouring.
  for id in order:
      e = store.get(id)
      reintegrate_tierR(e.cloud_body, e.T_map_body)        # voxels + plane cache + provenance
      reintegrate_tierS_geometry(e.cloud_body, e.T_map_body)   # nvblox TSDF only
  #  5b. COLOUR PASS: replay colour over the now-correct geometry, oldest->newest,
  #      so the EWMA's newest-wins bias lands on the freshest view and the
  #      sphere-traced occlusion test sees the completed surface.
  for id in order:
      e = store.get(id)
      reintegrate_tierS_colour(e.cloud_body, e.image, e.T_map_body, e.T_body_cam)
      # nvblox marks the re-fused blocks dirty; the mesher (§5.4) picks them up.

  # 6. nvblox re-meshes only the dirty blocks (incremental, on T5); the streamed
  #    mesh delta (§5.4) covers the re-fused blocks plus `clearedBlk` (blocks that
  #    must drop out of L6's standing mesh), and L6 swaps them block-for-block.
```

### 7.3 Why provenance + region superset (steps 2 & 4)

* **Step 2 (provenance, not loose AABB):** clearing the *exact* union of moved
  keyframes' touched cells avoids needlessly destroying correct map far from the
  correction. This is why Tier R keeps `kf_voxels_` (§3.7) and the nvblox wrapper
  keeps `kf_blocks_` (§4.2).
* **Step 4 (rebuild superset):** because TSDF voxels are *co-fused* by multiple
  keyframes, clearing a moved keyframe's voxels also destroys a *stationary*
  keyframe's contribution to those same voxels. Correctness requires
  re-integrating **every** keyframe overlapping the cleared region, found by
  `store.in_region(regionAabb)`. Missing this is the classic "ghost surface after
  loop closure" bug — re-integrating only the moved keyframes leaves the cleared
  co-fused voxels permanently missing a stationary keyframe's contribution.
* **Step 5 (two passes, geometry then colour):** colour fusion is order-dependent
  — the EWMA (§4.3) biases toward the newest observation, and the sphere-traced
  occlusion test reads the TSDF zero level set to decide whether a voxel is visible.
  Both require the region's *geometry* to already be correct. Interleaving colour
  with geometry per keyframe would colour against a half-built surface (spurious
  occlusion rejections, foreground bleed) and would let an early keyframe's colour
  dominate. The geometry pass therefore rebuilds the whole region's TSDF first; the
  colour pass then replays oldest→newest over the completed surface, so occlusion is
  tested against final geometry and the freshest accepted view wins the EWMA. The
  stamp-ascending order makes the rebuilt colour deterministic (the determinism
  requirement of §9).

### 7.3a Clear granularity: nvblox clears whole 8³ blocks, not voxels

nvblox stores all of its layers as sparse, block-allocated grids of **8³-voxel
blocks** in CUDA unified memory; its public clear API deallocates or zeroes **whole
blocks**, not individual voxels. Tier S therefore clears at block granularity: the
provenance the wrapper keeps is a **block** set (`kf_blocks_`, §4.2 — `BlockKey`,
not `VoxelKey`), and `clear_blocks(dirtyBlk)` clears the matching blocks across the
**TSDF, colour, and mesh layers together** so the three never desynchronise (a
cleared TSDF block with a stale colour or mesh block would re-mesh into a coloured
ghost). Because a block spans 8³ voxels, the cleared set is the block-rounded
super-set of the moved keyframes' touched voxels — clearing is slightly coarser
than the exact voxel footprint, which is harmless (the rebuild superset of step 4
re-fuses every keyframe overlapping the region anyway) and is the only granularity
nvblox actually offers. Tier R, being our own CPU hash, still clears at its finer
voxel granularity.

**Mesh swap is driven by the cleared-block report.** `clear_blocks` returns the set
of blocks it actually cleared (`getClearedBlocks` on the nvblox mapper). The mesher
(§5.4) treats those blocks as dirty and re-meshes exactly them; L6 keeps the
previously extracted host `ColorMesh` until the re-meshed blocks are ready, then the
new mesh is swapped in block-for-block (§7.5). The operator therefore never sees a
hole where a cleared region's mesh has been removed but not yet rebuilt.

### 7.4 Why clear-and-rebuild, not subtraction (the running-average argument)

nvblox's TSDF fusion is the running average $d\leftarrow(Wd+wd')/(W+w)$ (§4.3). To
*subtract* keyframe $k$'s contribution you would need to invert this for one
term, which requires retaining every per-keyframe $(d'_k, w_k)$ ever fused into
each voxel and replaying the exact arithmetic — orders of magnitude more memory
than the clouds themselves, and numerically fragile.
**Clear-and-rebuild from the retained clouds is exact, bounded, and matches nvblox
semantics** (it reproduces the same running average the voxel would have had if
the corrected poses had been used from the start). This is the explicit resolution
of MUST-FIX #4, consistent with spec 01 §7.5 ("we do NOT do per-voxel
subtraction").

### 7.5 Consistency during rebuild (no torn map)

The front-end keeps querying Tier R on T2 while T4 rebuilds. To avoid serving a
half-cleared region:

* Tier R rebuild is **region-local and staged**: cleared shards are write-locked
  per shard during their own clear+reinsert (§3.6), so a front-end query in an
  unaffected shard never blocks and a query in an affected shard either sees the
  pre-clear state or the post-rebuild state, never a torn intermediate (the
  clear+reinsert of a given shard happens under one write-lock acquisition where
  feasible; for large regions it is chunked, and an in-progress chunk returns
  `nullopt` rather than a partial plane).
* Tier S / mesh: nvblox meshing reads the GPU mesh layer on T5; L6 keeps the
  previously extracted host `ColorMesh` until the cleared blocks (the
  `getClearedBlocks` set, §7.3a) are re-meshed, then swaps the new block meshes in
  block-for-block (§5.4) — the front-end never queries Tier S, so no hot-path
  consistency concern exists there.
* The whole operation emits `event("map/region_rebuild", ...)` with the AABB and
  keyframe count, and a `marker("map/dirty_region", aabb)` so an operator *sees*
  the correction (arch §10.2, §10.4).

### 7.6 Cost & cadence

Rebuild cost ∝ (#points in `rebuildIds`). For a typical mid-mission loop snapping
~100 keyframes over a ~50 m region, that is ~1–3 M points re-integrated — the
nvblox GPU re-fusion is tens of ms; the Tier R CPU rebuild is the larger term,
tens to low-hundreds of ms on T4, off the front-end's critical path. Large global
snaps are chunked across several T4 cycles with `dirty_region` markers updated per
chunk; the front-end never stalls (it registers against the unaffected hot window
meanwhile). `Q_map` back-pressure (lossless) ensures no keyframe is dropped while
T4 is busy rebuilding.

---

## 8. Interfaces, factories, configuration

### 8.1 The interface decomposition

`IMapLayer` (spec 01 §7.5) is realised as a small family so the registration and
surface tiers stay independently testable (arch §5), all constructed by
`meridian_pipeline` and wired together. **The surface tier has exactly one
implementation — nvblox** (spec 00 §2.1, §9.5); there is no second `ISurfaceMap`
impl and no CPU-fallback selector.

```cpp
// meridian/map/iregistration_map.hpp  — Tier R (queried by L2, CPU)
class IRegistrationMap {
public:
  virtual ~IRegistrationMap() = default;
  virtual void integrate_live(const PointCloudMap& pts_map, Timestamp t) = 0;  // scratch path
  virtual void integrate_keyframe(std::uint64_t id,
                                  const std::vector<LidarPoint>& cloud_body,
                                  const Pose& T_map_body) = 0;                  // canonical
  virtual std::optional<PlaneHit> query_plane(const Eigen::Vector3f& p_map) const = 0;
  virtual void clear_keyframes(const std::vector<std::uint64_t>& ids) = 0;      // §7 step 3
  virtual void set_hot_window(const Aabb& hot) = 0;                             // FOV segment
  virtual MapDiagnostics diagnostics() const = 0;
};

// meridian/map/isurface_map.hpp  — Tier S + mesh (consumed by L6). ONE impl: nvblox.
class ISurfaceMap {
public:
  virtual ~ISurfaceMap() = default;
  // Live path: lay this keyframe's TSDF geometry then its colour in one call.
  // (The §7 rebuild splits geometry and colour into two region-wide passes
  // internally — see §7.2 — because co-fused colour is order-sensitive.)
  virtual void integrate(std::uint64_t id, const std::vector<LidarPoint>& cloud_body,
                         const std::shared_ptr<const CameraFrame>& image,
                         const Pose& T_map_body, const Pose& T_body_cam) = 0;
  // Clears the blocks the given keyframes touched (kf_blocks_ -> 8^3 blocks),
  // across the TSDF+colour+mesh layers together (§7.3a). The cleared-block set
  // feeds the next mesh delta so L6 drops the removed blocks.
  virtual void clear_keyframes(const std::vector<std::uint64_t>& ids) = 0;      // §7 step 3
  // Updates the standing host ColorMesh by streaming ONLY the changed/cleared
  // blocks (serializeSelectedLayers + layer streamer, §5.4) and returns a const
  // reference to it; it never re-serialises the whole map (that is offline-only).
  virtual const ColorMesh& extract_mesh() = 0;                                  // §5 (spec 01 §7.7 type)
  virtual MapDiagnostics diagnostics() const = 0;
};
// The only ISurfaceMap implementation: NvbloxSurfaceMap (meridian/map/src/nvblox/).

// meridian/map/imap.hpp — the façade IMapLayer (spec 01 §7.5) the pipeline holds.
// Owns IRegistrationMap (VoxelHashMap) + ISurfaceMap (NvbloxSurfaceMap) +
// KeyframeStore and implements integrate()/apply_graph_update()/query_plane()/
// extract_mesh() by delegation, enforcing the "store first, derived tiers are
// caches" discipline.
class LayeredMap final : public IMapLayer { /* ... composition ... */ };

IMapLayer::Ptr makeMapLayer(const MapConfig&, const CalibrationSet&);  // factory (arch §5.1)
```

The façade `LayeredMap::integrate(kf, T_map_body)` does, in order: (1)
`store.put` if first sight of the id; (2) `regMap.integrate_keyframe`; (3)
`surfaceMap.integrate` (nvblox). `apply_graph_update` runs the §7.2 algorithm
across both tiers. `query_plane` delegates to Tier R; `extract_mesh` to nvblox.

> **`makeMapLayer` does not select a backend.** There is no map backend choice:
> `makeMapLayer` always constructs `VoxelHashMap` + `NvbloxSurfaceMap`. The
> factory exists for dependency injection and test mocking, not to offer a
> CPU/GPU menu (spec 00 §5, §8.3).

### 8.2 Supporting value types declared here

```cpp
struct Aabb { Eigen::Vector3f min, max; bool intersects(const Aabb&) const; };
struct MapDiagnostics {
  std::size_t reg_voxels, tsdf_blocks, kf_count;
  double      last_integrate_ms, last_rebuild_ms, last_mesh_ms;
  std::size_t last_rebuild_kf, last_rebuild_voxels;
  std::size_t hot_window_voxels;
};
```

`PlaneHit` (§3.5, canonical in spec 01) and `ColorMesh` (§5.1, spec 01 §7.7) are used above. `PointCloudMap` is a thin
alias for a span/vector of map-frame `Eigen::Vector3f`.

### 8.3 Configuration (extends arch §8.2 `map:`)

The architecture schema has `map: { backend: nvblox, reg_voxel_m, tsdf_voxel_m,
mesh, colour }`. L4 expands it (validated on load, arch §8.3; **must** keep
`tsdf_voxel_m ≤ reg_voxel_m`). `backend` has exactly one valid value, `nvblox`;
there is no CPU/VDB alternative to select (spec 00 §8.3):

```yaml
map:
  backend:           nvblox        # the ONLY valid value (GPU; no fallback)
  reg_voxel_m:       0.2           # Tier R base voxel (== arch reg_voxel_m)
  reg_max_pts:       20            # kMaxPtsPerVoxel
  reg_seed:          0             # seed for deterministic reservoir voxel eviction (§3.2a)
  reg_max_level:     3             # adaptive subdivision cap
  reg_planarity:     0.1           # lambda0/lambda1 plane-accept threshold
  reg_min_plane_pts: 5             # kMinPlanePts
  reg_neighbor_ring: 1             # 0 = containing voxel only; 1 = 26 neighbours
  reg_hot_radius_m:  100.0         # hot-window half-extent (FOV segment)
  tsdf_voxel_m:      0.05          # Tier S voxel (== arch tsdf_voxel_m)
  tsdf_trunc_voxels: 4             # truncation = 4 * tsdf_voxel_m = 0.2 m
  tsdf_w_max:        8.0           # fusion weight cap (small: keeps surface responsive; §4.3)
  tsdf_max_integration_dist_m: 50.0  # depth cutoff for fusion; decoupled from preproc det_range (§4.3)
  color_enable:      true
  color_ewma_alpha:  0.8           # colour EWMA blend toward newest in-band observation (§4.3)
  color_occlusion_check: true      # sphere-trace before colouring a voxel (§4.3)
  invalid_depth_decay: false       # optional reproducible far-outlier prune (§4.5, post-MVP)
  mesh:              marching_cubes
  mesh_max_rate_hz:  2.0           # extract throttle
  mesh_conf_w:       8.0           # W_conf for per-vertex confidence (= tsdf_w_max)
  store:
    backend:         ram           # ram | mmap (deferred)
```

There is no `archive:` block (the NanoVDB out-of-core archive is removed) and no
fallback/`cpu:` key anywhere in the map schema.

---

## 9. Failure modes & recovery

| Failure | Detection | Response |
|---|---|---|
| Keyframe arrives with empty/degenerate `cloud_body` | `cloud_body->size() < min_pts` | skip integrate, `event(WARN,"map/empty_kf")`; still `store.put` (pose node valid) |
| `query_plane` finds no planar voxel | `nullopt` returned | front-end treats point unmatched (FAST-LIO `point_selected_surf=false`); not an L4 error |
| `GraphUpdate` references unknown id | `store.get(id)` empty | skip that id, `event(ERROR,"map/unknown_kf_in_update")`; rebuild the rest (degraded but safe) |
| Rebuild set too large (global snap) | `|rebuildIds|` > `rebuild_chunk_max` | chunk across T4 cycles (§7.6); front-end runs on hot window meanwhile |
| `Q_map` back-pressure (T4 behind) | queue near capacity | lossless: front-end keyframe creation slows (arch §11.2); **never drop** a keyframe cloud |
| GPU out of memory (nvblox, unbounded extent) | nvblox allocation failure | `event(ERROR,"map/gpu_oom")`; operator remedy is to lower `tsdf_max_integration_dist_m` (the primary VRAM control, §4.3), coarsen `tsdf_voxel_m`, or tighten `reg_hot_radius_m`. Fail-fast, **not** a silent CPU fallback (spec 00 §9.5; VRAM-exhaustion sharp edge, Appendix R.4) |
| CUDA GPU absent at startup | nvblox/CUDA init fails | hard fail-fast configuration error (spec 00 §9.5) — Meridian does not run mapping without a GPU |
| TSDF voxel saturated, can't correct | weight at `tsdf_w_max` yet pose moved | small `w_max` cap + clear-and-rebuild (§7) guarantees corrected re-fuse; saturation can't lock a wrong surface |
| Dynamic object burned into surface | stale surface persists where scene changed | small `tsdf_w_max` (default 8) lets fresh free-space/geometry fade it within a few sweeps (§4.3); the optional `invalid_depth_decay` (§4.5) prunes the far-outlier class |
| Foreground colour bled onto background | wrong tint at a depth discontinuity | `color_occlusion_check` sphere-traces before colouring; occluded voxels are not painted (§4.3) |
| RAM pressure from retained clouds | `store.size()` × avg over budget | enable `store.backend: mmap` (deferred hook §6.4) |
| Projective-integrator silhouette artefact | spurious surface at depth discontinuity (Appendix R.1) | accepted for the online mesh; the offline Poisson export (§10) is the clean archival path |
| NaN/inf in fused distance | non-finite `distance` after fuse | nvblox rejects the measurement; `event(WARN,"map/nonfinite_sdf")` from the host wrapper |

**Determinism mode** (arch §11.2): in `--single-thread` replay, integrate / mesh
/ rebuild run inline on the replay thread in the same order; nvblox kernels are run
in their deterministic variant where available, so the map (and mesh) is
reproducible across runs of the same bag to the extent the GPU reductions allow
(arch §11.2 notes bit-reproducibility is a test-only guarantee). The two
order-sensitive CPU/host decisions are pinned to be reproducible: Tier R voxel
eviction is seeded reservoir sampling (`reg_seed`, §3.2a), and the rebuild colour
pass replays keyframes in stamp-ascending order (§7.2), so neither the retained
point set nor the rebuilt surface colour depends on thread interleaving.

---

## 10. Deferred-but-designed hooks: ESDF, semantics, Poisson export, ground-alignment

First-pass scope **stops at the colour mesh**. The following are *not built now*
but the substrate is shaped so they attach without changing any L4 boundary
(arch §12: "deferred features attach as new interface implementations or new
consumers of existing value types — never as edits to existing layer contracts").

* **ESDF (path planning).** nvblox already computes a GPU ESDF from the same TSDF
  layer (`EsdfIntegrator`; the incremental raise/lower wavefront is digested in
  Appendix R.2). A deferred `EsdfMap` registers as one more derived consumer fed by
  the same `integrate` / `apply_graph_update` events; clear-and-rebuild applies
  unchanged (the ESDF is rebuilt whenever its TSDF region is — the incremental
  queues catch only local changes, not a global pose shift). **No change** to
  L0–L3, the `KeyframeStore`,
  the registration tier, or the mesh. Hook: a config slot keyed `map.esdf.enable`
  (absent in first pass).
* **Semantic / label channel.** nvblox's voxel can carry an extra label channel
  (running-max / running-histogram over class ids), populated by a future
  semantics module that already has what it needs: `KeyframePacket` carries
  `image` + `T_body_cam`, and the `KeyframeStore` retains cloud + RGB
  (spec 01 §6, arch §12). Mesh vertices would gain a per-vertex label; the `ColorMesh`
  type would grow a `labels` field (an additive amendment to spec 01, not a
  breaking change). **No change** to fusion or rebuild logic — labels fuse and
  rebuild on the same paths as colour.
* **Offline Screened-Poisson export (utility, not a runtime path).** The nvblox
  Marching-Cubes mesh **is** Meridian's deliverable mesh (spec 00 §12). *Optionally*,
  a one-shot offline export utility in `meridian_tools` may run **Screened-Poisson
  Surface Reconstruction** (Appendix R.3) over the retained
  `KeyframeStore` oriented-point clouds *after* the final globally-optimised
  trajectory is known, to produce a watertight archival mesh with density
  trimming. This is an export pass, **never** a core runtime tier, never an
  alternative online mesher, and it consumes the existing store without changing
  any boundary.
* **MapClosures ground-alignment hook (interface only).** A place-recognition
  variant (MapClosures-style, off the dense local map rather than per-scan
  descriptors) can detect a revisit and propose a **ground-aligned** relative
  transform between two map regions — a constraint whose vertical / roll-pitch
  components are pinned by a shared ground plane. L4 specifies only the **shape of
  the hook**, not the detector: a single function that hands L4 the candidate's
  region clouds and receives a ground-plane estimate,

  ```cpp
  // meridian/map/iground_align.hpp  — interface only; no first-pass implementation.
  struct GroundPlane { Eigen::Vector3f n; float d; float rms; };  // map frame, n·x+d=0
  class IGroundAligner {
  public:
    virtual ~IGroundAligner() = default;
    // Fit the dominant ground plane over the retained clouds in a region. Pure
    // read of the KeyframeStore; produces NO map mutation of its own.
    virtual std::optional<GroundPlane> fit_ground(const Aabb& region,
                                                  const KeyframeStore& store) const = 0;
  };
  ```

  The hook is **read-only against L4**: it consumes `KeyframeStore::in_region`
  (§6) and returns a `GroundPlane`. It does **not** move the map. Any correction it
  implies still reaches L4 through the one and only de-integration trigger — the
  back-end's `GraphUpdate` (§7) — after L5 turns the ground-aligned candidate into a
  `LoopConstraint` and the back-end optimises. This keeps the L4 contract intact
  (the map is corrected only by clear-and-rebuild on a `GraphUpdate`) and confines
  the new code to a read-only consumer of the store. First pass ships without it
  (`map.ground_align.enable`, absent in first pass).

The rule, restated: ESDF, semantics, Poisson, and ground-alignment are *new derived
consumers / optional channels / an offline utility / a read-only store consumer*,
riding the existing `integrate` / `apply_graph_update` / `KeyframeStore` machinery.
The first-pass code must compile and run with all of them absent.

---

## 11. Debug / introspection for L4

L4 emits through the ROS-agnostic `TelemetrySink` (arch §10); the wrapper maps it
to topics/markers. The L4-specific signals (extending arch §10.2 Appendix C):

| Signal | Channel & key | Purpose |
|---|---|---|
| Registration map cloud | `cloud("map/registered")` | the Tier R hot voxels (centroids) — the `/cloud_registered` analogue |
| Surface mesh | `mesh`/`cloud("map/mesh")` → wrapper `ColorMesh` msg | the nvblox colour mesh for L6 (streamed block delta, §5.4) |
| Per-stage timing | `timing("map.integrate")`, `timing("map.tsdf_fuse")`, `timing("map.mesh_extract")`, `timing("map.region_rebuild")` | live breakdown (replaces FAST-LIO's CSV `aver_time_*`) |
| Plane-query stats | `scalar("map/plane_hit_rate")`, `scalar("map/empty_voxel_rate")` | how well Tier R is serving the front-end |
| Voxel / block counts | `scalar("map/reg_voxels")`, `scalar("map/tsdf_blocks")` | growth / memory watch |
| GPU memory | `scalar("map/gpu_mem_mb")` | nvblox VRAM watch (the OOM early-warning, §9) |
| **Region rebuild** | `event("map/region_rebuild")` + `marker("map/dirty_region", aabb)` | **MUST-FIX #4 made visible** — operator sees the loop correction's footprint (arch §10.2) |
| Hot-window prune | `event("map/prune")` + `scalar("map/pruned_voxels")` | hot-window shrink (Tier R FOV segmentation) |
| Hot window | `marker("map/hot_window", aabb)` | the FOV-segment box (FAST-LIO `LocalMap_Points` analogue) |
| Confidence overlay | per-vertex `ColorMesh::confidence` → L6 tint | low-confidence (thin TSDF weight) surface flagged to operator (§5.3) |
| Store size | `scalar("map/kf_count")`, `scalar("map/store_bytes")` | retained-store budget watch (§6.4) |

`DebugControl` (arch §10.5) toggles the heavy ones (mesh, registered cloud) at
runtime without restart; `NullSink` makes them zero-cost when off (arch §10.6).

---

## 12. Parameter reference

| Param | Default | Meaning | Cross-ref |
|---|---|---|---|
| `map.backend` | nvblox | surface map backend (the only valid value) | §4, arch §8.2 |
| `map.reg_voxel_m` | 0.2 | Tier R base voxel edge [m] | §3.2, arch §8.2 |
| `map.reg_max_pts` | 20 | points kept per voxel (downsample cap) | §3.2 (ikd-Tree `downsample_size` analogue) |
| `map.reg_seed` | 0 | seed for deterministic reservoir voxel eviction | §3.2a |
| `map.reg_max_level` | 3 | adaptive subdivision depth cap | §3.3 |
| `map.reg_planarity` | 0.1 | $\lambda_0/\lambda_1$ plane-accept threshold | §3.3 |
| `map.reg_min_plane_pts` | 5 | min support to fit a plane | §3.3 |
| `map.reg_neighbor_ring` | 1 | neighbour probe radius on empty/non-planar voxel | §3.5 |
| `map.reg_hot_radius_m` | 100 | hot-window half-extent (FOV segment) | §3.4 |
| `map.tsdf_voxel_m` | 0.05 | nvblox voxel edge [m] (≤ `reg_voxel_m`) | §4.2, arch §8.3 |
| `map.tsdf_trunc_voxels` | 4 | truncation band in voxels ($\tau = 4\cdot v$) | §4.3, Appendix R.4 |
| `map.tsdf_w_max` | 8 | fusion weight cap [obs]; small ⇒ surface stays responsive to change | §4.3, §9 |
| `map.tsdf_max_integration_dist_m` | 50.0 | surface-fusion depth cutoff [m]; decoupled from preproc `det_range`; primary VRAM control | §4.3, §9 |
| `map.color_enable` | true | fuse RGB into nvblox colour layer | §4.3 |
| `map.color_ewma_alpha` | 0.8 | colour EWMA blend toward newest in-band observation | §4.3 |
| `map.color_occlusion_check` | true | sphere-trace occlusion test before colouring a voxel | §4.3 |
| `map.invalid_depth_decay` | false | optional reproducible far-outlier prune (post-MVP) | §4.5 |
| `map.mesh` | marching_cubes | mesher kind (nvblox MC) | §5, arch §8.2 |
| `map.mesh_max_rate_hz` | 2.0 | mesh extraction throttle | §5.4 |
| `map.mesh_conf_w` | 8 | $W_{\text{conf}}$ for per-vertex confidence (= `tsdf_w_max`) | §5.3 |
| `map.store.backend` | ram | retained store impl (`ram`/`mmap`) | §6.4 |

---

## Summary of L4 contracts (do not break)

1. **The `KeyframeStore` is the single source of truth**; Tiers R and S are
   disposable caches rebuildable from it (§1, §6). There is no out-of-core voxel
   archive — the store is the durable geometry record.
2. **L4 never deskews**; clouds arrive deskewed in the body frame at `stamp` and
   are integrated by one `Pose` multiply (§2.1, spec 01 §6).
3. **Registration (Tier R) uses an adaptive voxel-hash on the CPU** with cached
   per-voxel planes, keeping ikd-Tree's incremental-insert/box-delete/k-NN
   *behaviour* (§3). The front-end's hot query never touches the GPU.
4. **The surface map is nvblox, GPU-only** (TSDF + colour + Marching-Cubes mesh),
   with a **small** fusion-weight cap (`tsdf_w_max` ≈ 8) for scene responsiveness —
   correctability is owned by clear-and-rebuild, not the cap (§4.3). Colour is fused
   by EWMA with a sphere-traced occlusion test (§4.3); the runtime mesh is streamed
   as a per-cycle block delta, full extraction is offline-only (§5.4). No CPU
   fallback, no VDBFusion/OpenVDB, one `ISurfaceMap` implementation (§4, §5;
   spec 00 §9.5).
5. **Loop correction is clear-and-rebuild from retained clouds at corrected
   poses** — never per-voxel subtraction (MUST-FIX #4, §7); exact
   via per-keyframe provenance + region-overlap superset, cleared at nvblox's 8³
   block granularity (§7.3a) and replayed geometry-pass-then-colour-pass (§7.2).
6. **Scope stops at the colour mesh**; ESDF, semantics, the Screened-Poisson
   export, and the ground-alignment hook are deferred consumers / optional channels
   / an offline utility / a read-only store consumer that ride existing machinery
   and change no boundary (§10).
7. **Region rebuild and pruning are visible** on the debug bus (§11).

---

## Appendix R — SOTA reference grounding (non-normative)

This appendix is evidence, not contract: curated digests of the reference systems
this spec's design was validated against. Nothing here binds Meridian's behavior —
the normative sections above own the design. Each block names the reference
checkout it was verified against; the clones live in `/home/user/slam-reference`.

> **Source-of-record note for this appendix.** The dense-mapping reference systems
> (Voxblox, nvblox, VDBFusion, OpenVDB/NanoVDB, PoissonRecon) are **not** cloned
> into `/home/user/slam-reference` — only the LIO/LIVO/back-end/loop systems are.
> These blocks are therefore digests of the **primary papers + canonical repos**
> and are the **local source of record**; values flagged *verify against repo* must
> be confirmed against the upstream config headers (they drift per release) before
> they are ever treated as load-bearing.

### R.1 TSDF integrators — Voxblox `simple` / `merged` / `fast` contrast

Verified against the Voxblox paper (Oleynikova et al., IROS 2017, arXiv:1611.03631,
§III) + `ethz-asl/voxblox` — clone of 2026-06 (not in slam-reference).

| Integrator | Strategy | Cost driver | When |
|---|---|---|---|
| `simple` (raycast) | one ray per measured point; update every band voxel along it | redundant near-sensor updates (many rays overlap) | reference / accuracy baseline |
| `merged` (bundled raycast) | group points landing in the same endpoint voxel, cast **one bundled ray per group** with combined weight + merged colour | bundling cost; near-raycast quality at near-projective cost | the CPU-raycast design to copy if ever writing a bespoke integrator |
| `fast` | subsample start voxels, per-voxel "already-updated-this-frame" guard, cap consecutive ray collisions, bound integrated rays | tuned for predictable latency, trades accuracy | hard real-time CPU |

**Projective (voxel→image) vs raycasting (ray→voxels):** projective iterates
candidate frustum voxels and looks each up in the depth image — one pass,
embarrassingly parallel ⇒ the GPU-native form nvblox is built around
(`ProjectiveTsdfIntegrator`); artefacts at depth discontinuities (a voxel may
project onto a foreground pixel off its true ray). Raycasting is more faithful near
edges but redundant near the origin. Meridian's Tier S uses the projective form
(§4.3).

### R.2 ESDF — incremental raise/lower wavefront digest

Verified against Voxblox paper §IV + `ethz-asl/voxblox` `EsdfIntegrator` — clone of
2026-06 (not in slam-reference). (ESDF is a deferred hook for Meridian, §10.)

Two-queue brushfire seeded by the TSDF integrator's per-frame **changed-voxel**
report ⇒ cost O(changed voxels), not a full recompute:

1. **Fixed band:** voxels within TSDF truncation copy `ESDF.dist = TSDF.dist`,
   marked `fixed` (seeds / boundary conditions).
2. **Lower queue** (a closer obstacle appeared): relax neighbours —
   `if dist(v)+edge(v,n) < dist(n): dist(n)=dist(v)+edge(v,n); parent(n)=v; push n`.
3. **Raise queue** (obstacle removed/farther): for each neighbour, if `parent(n)==v`
   invalidate `n` and push to raise; else push `n` to lower to re-seed a correct
   smaller value. **Process raise first** (invalidate stale), then lower
   (re-propagate).

**Quasi-Euclidean** propagation to 26-connected neighbours with edge weights `1`
(face), `√2` (edge), `√3` (corner), tracking a `parent` direction — near-Euclidean
at a fraction of the exact cost. nvblox computes the same wavefront on GPU
(`EsdfIntegrator`, can emit a 2D height slice for a Nav2 costmap). Sharp edge: an
ESDF derived from a TSDF that was cleared-and-rebuilt (§7) must also be rebuilt —
the incremental queues catch only *local* changes, not a global pose shift.

### R.3 Surface extraction — MC33, Dual Contouring, Screened Poisson

Verified against the source papers + canonical repos (Lorensen & Cline 1987;
Nielson & Hamann 1991; Chernyaev 1995; Ju et al. 2002; Kazhdan & Hoppe ToG 2013,
`mkazhdan/PoissonRecon`) — clones of 2026-06 (not in slam-reference).

**Marching Cubes ambiguity → MC33.** The 256 corner-sign configurations reduce by
symmetry to 15 base cases; face/interior ambiguities can connect adjacent cubes
inconsistently ⇒ cracks/holes. Fixes: the **asymptotic decider** (bilinear saddle
value on the ambiguous face chooses connectivity consistently) and **Marching
Cubes 33** (33-case table + asymptotic decider ⇒ topologically correct, crack-free,
manifold). nvblox's online MC (§5) does not need watertightness; MC33 is the
reference if a watertight *live* mesh is ever wanted.

**Dual Contouring (sharp-edge alternative, not used).** One vertex per cell (dual
of MC's on-edge vertices) from **Hermite data** `(p_i, n_i)` per sign-changing
edge, minimising the QEF
$$E(x)=\textstyle\sum_i\big[\,n_i\cdot(x-p_i)\,\big]^2.$$
Reproduces sharp edges/corners MC rounds off and adapts to octrees, but needs
reliable normals and can be non-manifold without extra care.

| | Marching Cubes | Dual Contouring |
|---|---|---|
| Vertex placement | on edges (interpolated) | one per cell (QEF) |
| Inputs | scalar field | Hermite (point + normal) |
| Sharp features | rounded | preserved |
| Manifold | yes (with MC33) | needs care |
| Octree adaptivity | harder | natural |
| Used by TSDF libs | **yes** (Voxblox/nvblox) | rare |

TSDFs are noisy and organic-scene-dominated, so DC's sharp-feature advantage does
not pay off; Meridian uses MC (§5).

**Screened Poisson (offline archival mesh, §10).** Operates on **oriented points**,
not a grid; the indicator function χ (1 inside, 0 outside) has gradient equal to
the surface normal at the surface ⇒ solve a Poisson equation, screened by a
positional constraint pulling the iso-surface through the input points:
$$
E(\chi)=\int\lVert\nabla\chi-V\rVert^2\,dx
       +\alpha\,\frac{1}{\sum_i w_i}\sum_i w_i\big(\chi(p_i)-\text{iso}\big)^2.
$$
χ is on an octree-adaptive B-spline basis ⇒ memory ∝ surface area; discretising
gives a sparse SPD system solved by multigrid in near-linear time; the single
globally-defined smooth χ yields a watertight closed manifold across data gaps.

Key `mkazhdan/PoissonRecon` CLI parameters (*verify defaults against current
release*):

| Flag | Meaning | Note |
|---|---|---|
| `--depth` | octree depth (~8–11) | dominant detail/cost knob; 10–11 for archival |
| `--pointWeight` (α) | screening strength | default ≈ 4; higher ⇒ closer to data, less smoothing |
| `--samplesPerNode` | min samples per node (~1–5) | raise for noisy data ⇒ smoother |
| `--density` + **trim** | density-trim post-step | cut low-density extrapolated triangles to kill hallucinated surface in unobserved regions |

Normals for Poisson: TSDF gradient ∇D sampled at vertices (already denoised by
fusion) or per-point PCA on k-NN oriented toward the sensor.

### R.4 Parameter sharp-edges (Tier S relevant subset)

Verified against Voxblox/nvblox/PoissonRecon papers + repos — clones of 2026-06
(not in slam-reference). Starting values are *room-indoor / outdoor-LiDAR*.

| Parameter | Effect | Indoor start | Outdoor LiDAR start |
|---|---|---|---|
| `voxel_size` | master detail/cost knob; cost ∝ ~1/voxel³ | 0.02–0.05 m | 0.10–0.20 m |
| `truncation_distance` δ | small = sharp but noise-sensitive & thin ESDF band; large = smooth, robust, more updates | 3–5 × voxel | 3 × voxel (≈0.3 m) |
| `max_weight` (W_max) | small = adapts fast to change; large = stable but stale | moderate | moderate–high |
| `max_integration_distance` | depth cutoff; bound work, drop noisy far returns | ~5–7 m (depth cam) | sensor max (50–100 m) |
| weight model | const vs 1/z² vs behind-surface drop-off | 1/z² (depth cam) | const (LiDAR) |
| colour weight / band | restrict colour blend to \|D\|≪δ near surface | near-surface | near-surface |
| Poisson `--depth`/`--pointWeight`/trim | archival detail / data-fidelity / hole-trim | 9–10, ≈4, trim | 10–11, ≈4, trim |

Rule of thumb: set `voxel_size` from the smallest feature you must capture and the
compute budget, then `δ = (3–5)·voxel_size`, then tune the weight model and
`max_weight` to scene dynamics. Failure-mode pairings carried into §9:
projective-integrator silhouette artefacts at depth discontinuities; W_max too high
⇒ dynamic objects "burned in"; truncation too small ⇒ noisy/holey surface and a
thin ESDF fixed band; Poisson hallucination in unobserved regions ⇒ always
density-trim.

### R.5 Source index (papers + repos, none cloned locally)

The dense-mapping reference systems below are **not in `/home/user/slam-reference`**;
this index is the local source of record. Verified against the cited papers/repos —
clones/PDFs of 2026-06.

**Papers**
- Oleynikova et al. — *Voxblox*, IROS 2017, arXiv:1611.03631 (integrators §III;
  incremental ESDF wavefront §IV; weighted-average update). *Exact in-paper eq.
  numbering unverified — confirm against PDF.*
- Curless & Levoy — *A Volumetric Method for Building Complex Models from Range
  Images*, SIGGRAPH 1996 (origin of cumulative weighted-average TSDF).
- Newcombe et al. — *KinectFusion*, ISMAR 2011 (real-time GPU TSDF + projective
  integration + MC).
- Lorensen & Cline — *Marching Cubes*, SIGGRAPH 1987 (256→15 cases; edge/triangle
  tables; edge interpolation).
- Nielson & Hamann — *The Asymptotic Decider*, IEEE Vis 1991; Chernyaev —
  *Marching Cubes 33*, 1995 (ambiguity fixes).
- Ju, Losasso, Schaefer, Warren — *Dual Contouring of Hermite Data*, SIGGRAPH 2002
  (DC + QEF).
- Kazhdan & Hoppe — *Screened Poisson Surface Reconstruction*, ACM ToG 2013
  (Poisson + screening; octree B-spline multigrid; `--depth`/`--pointWeight`/trim).
- Millane et al. (NVIDIA/ETH ASL) — *nvblox: GPU-Accelerated Incremental Signed
  Distance Field Mapping*, ICRA 2024, arXiv:2311.00626 (reports ~177× surface-recon
  and ~31× ESDF speed-up vs CPU SOTA; defaults best read from repo config headers).

**Repositories**
- `ethz-asl/voxblox` (+ `voxblox_ros`) — `simple`/`merged`/`fast` integrators;
  `tsdf_voxel_size`, `tsdf_voxels_per_side`=16, `truncation_distance`, `max_weight`,
  `voxel_carving_enabled`, `use_const_weight`; `TsdfMap/Layer/Block/TsdfVoxel`;
  `EsdfIntegrator`, `MeshIntegrator`.
- `nvidia-isaac/nvblox` (+ `isaac_ros_nvblox`, `nvblox_torch`) — GPU
  TSDF/ESDF/colour/mesh; 8³ `VoxelBlock` in CUDA unified memory; SQLite
  serialisation; `ProjectiveTsdfIntegrator`, `ProjectiveColorIntegrator`,
  `EsdfIntegrator`, `MeshIntegrator`. *Current default
  `voxel_size`/`truncation_distance_vox`/`max_integration_distance_m`/`max_weight`
  drift per release — read from repo config headers.*
- `mkazhdan/PoissonRecon` — reference Screened Poisson; CGAL
  `poisson_surface_reconstruction` as a library alternative.
