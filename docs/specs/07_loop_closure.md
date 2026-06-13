# 07 — L5: Place Recognition & Loop Closure

> **Spec status:** normative implementation spec. This document specifies L5 — the
> place-recognition / loop-closure layer of Meridian. It is an *implementation* spec
> (data structures, interfaces, algorithms in pseudocode with real equations,
> parameters, failure modes, debug hooks), not a tutorial.
>
> **Consistency contract.** This spec is subordinate to `00_architecture.md`
> (layer boundaries, threading, telemetry subsystem, library canon) and
> `01_interfaces_and_data_types.md` (the boundary types). It **does not redefine**
> any boundary type. In particular it consumes `KeyframePacket` (`01` §6), reads
> geometry through the retained `KeyframeStore` (`01` §7.5), implements
> `ILoopDetector` (`01` §7.6), and emits `LoopConstraint` (`01` §7.6). It triggers
> map de-integration **only** through the back-end's `GraphUpdate` →
> `IMapLayer::apply_graph_update` path (`01` §7.5); L5 never mutates the map
> directly. Where this spec needs a field that `01` does not declare, it keeps that
> field in an **L5-private** type and says so explicitly — it never silently widens
> a boundary type. Two genuinely useful additions to `01` are surfaced as
> **[PROPOSE-TO-01]** for the human to decide.
>
> **One system, no phasing.** Meridian is one complete system: a discrete,
> tightly-coupled LiDAR-Inertial estimator (`00` §0). L5 is a permanent
> part of that system, not a "later phase." There is no "v1 without loops, v2 with
> loops" rollout: the cascade below is the design from the start. The only
> bring-up note is module *integration order* — L5 compiles and links against the
> stable `01` boundary types, so it can be built and unit-tested before the
> front-end is fully tuned — but that is a build convenience, not a product
> milestone.
>
> **Grounding honesty.** L5 is the one Meridian layer whose core algorithms are *not*
> present in the reference odometry code. The FAST-LIO2 paper text
> (`papers/2107.06829.txt`) and FAST-LIVO2 paper text (`papers/2408.14035.txt`)
> contain **no** place-recognition or loop-closure method — both are explicitly
> odometry-without-loop-closure (FAST-LIO2 paper l.1221: *"FAST-LIO2 is an
> odometry without any loop detection or correction"*; FAST-LIVO2 paper l.1103:
> *"As an odometry, FAST-LIVO2 may have drifts over long distances. In the future,
> we could integrate loop closure"*). So L5's **descriptors and consistency logic
> are grounded in external SOTA** (Scan Context++, STD/BTC, GICP, PCM, GNC,
> switchable constraints), digested in **Appendix R** (cited as `Appendix R.N`) and
> cross-checked against the back-end spec's robust-factor grounding. The
> **geometric sub-steps L5 reuses** — nearest-neighbour search, plane/surfel
> fitting, body→world point transform — *are* in the reference code and are cited
> to `file:line`. Both classes of citation appear in §16.
>
> Notation follows `01` §2: $T_{A\_B}\in SE(3)$ maps a point in frame $B$ to frame
> $A$ ($p_A = T_{A\_B}\,p_B$); `Pose` tangent order is $\xi=[\rho;\phi]$
> (translation-first, right perturbation, `01` §3.1); the per-axis observability
> order is $[t_x,t_y,t_z,r_x,r_y,r_z]$, scores in $[0,1]$, 1 = well observed
> (`01` §3.4). Time is `int64` nanoseconds (`01` §2.1).

---

## Table of contents

1. [Purpose, scope, non-goals](#1-purpose-scope-non-goals)
2. [Position in the system: data flow and threading](#2-position-in-the-system-data-flow-and-threading)
3. [Inputs and outputs (the contracts)](#3-inputs-and-outputs-the-contracts)
4. [The hierarchical cascade (overview)](#4-the-hierarchical-cascade-overview)
5. [Stage A — Scan Context++ retrieval](#5-stage-a--scan-context-retrieval)
6. [Stage B — STD / BTC re-ranking](#6-stage-b--std--btc-re-ranking)
7. [Stage C — GICP geometric verification (small_gicp)](#7-stage-c--gicp-geometric-verification-small_gicp)
8. [Stage D — PCM batch consistency](#8-stage-d--pcm-batch-consistency)
9. [Building the LoopConstraint (fitness-scaled, GNC+switchable)](#9-building-the-loopconstraint-fitness-scaled-gncswitchable)
10. [Interaction with L4: retained clouds and de-integration](#10-interaction-with-l4-retained-clouds-and-de-integration)
11. [Data structures and interfaces (C++)](#11-data-structures-and-interfaces-c)
12. [Parameters](#12-parameters)
13. [Failure modes and mitigations](#13-failure-modes-and-mitigations)
14. [Debug / introspection hooks](#14-debug--introspection-hooks)
15. [Test plan](#15-test-plan)
16. [References](#16-references)

---

## 1. Purpose, scope, non-goals

### 1.1 Purpose

L5 answers one question, repeatedly and conservatively: *"Has the platform been
**here** before, and if so, what is the rigid transform $T_{\text{from}\_\text{to}}$
that re-aligns the current keyframe to the past one?"* It packages each accepted
answer as a single `LoopConstraint` (`01` §7.6) and hands it to the L3 back-end via
`IBackEnd::add_loop_constraint` (`01` §7.4), which decides — globally, with iSAM2
and all other factors — whether and how far to bend the trajectory.

The asymmetry that drives the whole design: **a missed loop costs accuracy; a
false loop corrupts the global map and can be unrecoverable.** L5 is therefore a
*cascade of increasingly expensive, increasingly discriminative filters*: cheap
stages reject the overwhelming majority of non-matches, so the expensive
geometric and consistency stages see only a small, high-quality candidate set.
The final defences (PCM pre-admission in L5, then committed-Huber and batch-GNC in
the back-end — the robustness series of §9.3) ensure that even a wrong loop that
survives verification cannot, by itself, break the graph. This is the
defence-in-depth hierarchy of `Appendix R.6, §8`.

Place recognition recognises a place from the *appearance* of the scan,
independent of where odometry thinks it is — which is why it works after the drift
that a pure radius-search proximity detector cannot survive (`Appendix R.6`).

### 1.2 In scope

- **Intra-session loop closure** on the live trajectory.
- The four-stage cascade: **Scan Context++ retrieval → STD/BTC re-rank → GICP
  verify (small_gicp) → PCM batch check** — the hierarchical coarse→fine→batch
  pattern of `Appendix R.6, §8`.
- Synthesising the `LoopConstraint`: GICP-fitness-scaled, observability-aware
  covariance. The back-end then applies the committed-Huber-then-batch-GNC
  robustness series, with switchable held in reserve (this spec sets the `cov` and
  `fitness` fields those mechanisms consume; it does not implement the GTSAM kernel
  — that is the back-end spec, `05_backend_graph.md`, grounded in `Appendix R.5`
  and the back-end's own reference grounding).
- Reading geometry from the retained per-keyframe `KeyframeStore` and *being the
  cause of* L4 de-integration (via the back-end's `GraphUpdate`).

### 1.3 Non-goals (deferred)

- **Multi-session / map-reuse relocalization** ("kidnapped robot" against a saved
  map). The descriptor DB is built so it *could* support it (Stages A/B query a
  DB, not "the live trajectory"), but the session-bootstrap-against-prior-map
  control flow is **not built now**.
- **Semantic / object-level loop cues.** Deferred with all semantics (the system's
  first-pass scope stops at the colourised mesh, `00` §0). A high-recall
  semantic-histogram proposer would compose cleanly in parallel with Stage A
  later, but is out of scope now.
- **Pure appearance (visual) place recognition** (NetVLAD/BoW). L5 is
  LiDAR-descriptor-primary; the camera contributes only an *optional* photometric
  tie-break inside Stage C (off by default, §7.6). The image-assisted iBTC variant
  (`Appendix R.2`) is a documented future extension behind the same Stage-A/B
  interface, not built now.

---

## 2. Position in the system: data flow and threading

```
        L2 LIO front-end                                  L4 nvblox map
            │ KeyframePacket                               (GPU TSDF + colour
            ▼                                               + Marching-Cubes mesh)
   ┌──────────────────────┐                                      ▲
   │  L3 back-end (iSAM2)  │   add_keyframe → KeyframeStore.put   │ apply_graph_update
   │  keyframe pose graph  │ ───────────────────────────────────▶│ (clear & rebuild)
   └──────────────────────┘   add_keyframe → ILoopDetector       │
        ▲            │ GraphUpdate (moved keyframes)              │
   LoopConstraint    └────────────────────────────┐             │
        │  add_loop_constraint                     ▼             │
   ┌────┴───────────────────────────────────────────────────────┴───┐
   │                        L5  (this spec)                           │
   │  add_keyframe(id,cloud,T): build+insert descriptor               │
   │  detect(): run cascade                                           │
   │  ┌─────────┐  ┌──────────┐  ┌────────────┐  ┌──────┐             │
   │  │ A SC++  │─▶│ B STD/BTC │─▶│ C small_gicp│─▶│ D PCM│─▶LoopConstraint│
   │  │ retrieve│  │  re-rank  │  │  verify    │  │ batch│    → L3      │
   │  └─────────┘  └──────────┘  └────────────┘  └──────┘             │
   │   descriptor DB (Scan-Context KD-tree + BTC triangle/binary hash)│
   │   over per-anchor submaps (last-N composed keyframes, §3.4)      │
   │       geometry read-only via  KeyframeStore.cloud(id)            │
   └─────────────────────────────────────────────────────────────────┘
```

**Threading** (per `00_architecture.md` §11 and `01` §2.4 / §7.6). L5 runs on the
**back-end thread (best-effort)**, *not* the front-end thread — loop closure must
never stall the real-time odometry. Concretely:

- **`add_keyframe(id, cloud, T_map_body)`** (the `ILoopDetector` method, `01` §7.6)
  is called when the back-end admits a keyframe. It composes the anchor submap
  (§3.4), builds the keyframe's descriptors, and inserts them into L5's
  privately-owned DB. It is cheap (the SC build is sub-millisecond, §5.4; submap
  composition is a small-window union + one downsample) so it runs inline in the
  back-end's keyframe ingest. BTC triangles are built lazily on first use, not here.
- **`detect()`** is called on the back-end thread (throttled, §12
  `lc.detect_period_kf`). It runs the cascade and returns the batch of verified,
  PCM-filtered `LoopConstraint`s; the back-end calls `add_loop_constraint` for
  each, robustifies it through the committed-Huber-then-batch-GNC series (§9.3),
  and folds it into iSAM2.
- **Geometry access:** L5 reads clouds via `KeyframeStore::cloud(id)` (`01` §7.5),
  which returns `shared_ptr<const std::vector<LidarPoint>>` (Shared-immutable, `01`
  §2.4). Concurrent reads need no lock; a cloud L5 holds a handle to cannot be
  freed mid-GICP (reference-counted). L5 **never** copies a cloud and **never**
  mutates one.
- **De-integration:** L5 does **not** send a region-rebuild message. The loop it
  emits changes keyframe poses *inside iSAM2*; the back-end's `optimize()`
  returns a `GraphUpdate` listing moved keyframes (`01` §7.4), and the back-end
  forwards that to `IMapLayer::apply_graph_update` (`01` §7.5). L5's only role in
  de-integration is *causing the loop* and (for debug) *observing the resulting
  `GraphUpdate`*. See §10.

Because L5's geometry comes from the same `KeyframeStore` that the back-end fills
from the `KeyframePacket`, L5 is independent of the estimator internals: the
packet's `cloud_body` is a Shared-immutable handle the back-end forwards to the
store; L5 dereferences it through the store, never touching an L2-internal buffer.
(The front-end sits behind the `IFrontEnd` interface, `00` §5.4; L5 is identical
regardless of which estimator produced the keyframe, because it only ever sees
`01` boundary types.)

---

## 3. Inputs and outputs (the contracts)

### 3.1 Input — the keyframe (from L3, geometry from `KeyframeStore`)

L5's `add_keyframe(id, cloud, T_map_body)` (`01` §7.6) receives the three things it
needs directly. For richer gating it also reads, from the `KeyframePacket` the
back-end holds (`01` §6) and the store:

| Source | Field | L5 use |
|---|---|---|
| `add_keyframe` arg | `id : uint64` | DB key; `LoopConstraint.from_id`/`to_id`; graph node id. |
| `add_keyframe` arg | `cloud : shared_ptr<const vector<LidarPoint>>` | the deskewed **body-frame** cloud — geometry source for Stages A/B/C. |
| `add_keyframe` arg | `T_map_body : Pose` | stored snapshot for spatial gating + GICP initial-guess seed. |
| `KeyframePacket` | `stamp : Timestamp` | temporal gating (reject too-recent candidates, §5.4). |
| `KeyframePacket` | `observability : ObservabilityReport` | per-axis loosening of the loop covariance (§9.2). |
| `KeyframePacket` | `image`, `T_body_cam` | optional Stage-C photometric tie-break only (§7.6). |

> **Clarification (consistent with `01` §6.5 / §7.5).** L5 gets geometry
> **exclusively** through the retained store, *not* from any transient L2 buffer.
> The packet's `cloud_body` and the store's `cloud(id)` resolve to the *same*
> Shared-immutable bytes. This is what makes loop re-integration cheap (§10) and
> keeps L5 estimator-agnostic.

**Frame contract (critical).** Clouds live in a **body frame** (the estimation
frame $F_e$, normally `imu_link`, `01` §2.3), never world. The same immutable
cloud is reusable at *any* corrected pose because placement is
$p_{\text{map}} = T_{\text{map}\_\text{body}}\cdot p_{\text{body}}$. This is the
exact operation the reference odometry performs in `pointBodyToWorld`
(`FAST_LIO/src/laserMapping.cpp:178`, computing `R*(R_LI*p_body + t_LI) + t`). L5
applies the same transform when it needs world coordinates (descriptors/markers);
it never stores world points.

Per-point covariances / normals for GICP (§7.2) are **not** in the `LidarPoint`
(`01` §4.2); they are computed on demand by `small_gicp` from each point's k-NN and
cached in an L5-private side table keyed by `id` (§11) — L5 does **not** mutate the
Shared-immutable cloud.

The `KeyframeStore` accessors L5 uses (`01` §7.5): `cloud(id)`, `image(id)`,
`within_radius(center, radius)` (the latter as a loose spatial pre-gate beside
descriptor retrieval). The stored pose snapshot needed for gating is the
`T_map_body` L5 was handed in `add_keyframe` and kept in its own DB entry.

### 3.2 Output — `LoopConstraint` → L3 (defined in `01` §7.6)

`detect()` returns `std::vector<LoopConstraint>`; the back-end ingests each via
`add_loop_constraint`. The struct is **fixed by `01` §7.6** and L5 must not widen it:

```cpp
struct LoopConstraint {          // 01 §7.6 — DO NOT redefine
  std::uint64_t from_id = 0, to_id = 0; // from = older (target), to = newer (source)
  Pose          T_from_to;              // maps `to` body into `from` body frame
  PoseCov6      cov;                    // 6-DoF covariance (Σ), fitness-scaled   ← §9.2
  double        fitness = 0;            // GICP fitness / inlier ratio in [0,1]    ← §7.5
};
```

L5's responsibilities per field: set `from_id`/`to_id`/`T_from_to` from Stage C;
set `cov` per §9.2 (already in `PoseCov6` tangent order $[\rho;\phi]$); set
`fitness` from §7.5. The back-end reads `fitness` to drive its robustness series —
the committed Huber kernel and the off-thread batch GNC (`05_backend_graph.md` §8;
§9.3/§9.4) — so a marginal loop is easier to down-weight and remove (`01` §7.6 note:
*"a low-fitness GICP match yields a loose `cov`, so a marginal loop barely tugs the
graph"*; the same fitness-scaled noise model is `Appendix R.5`). Meridian uses **no**
in-graph switch variable (`05_backend_graph.md` §8.3); §9.4.

> **Single relative constraint, no double-counting (consistent with `01` §6.4 /
> MUST-FIX #3).** A loop is **one** relative pose constraint between two existing
> keyframe pose variables, with marginal covariance — exactly like an odometry
> between-factor, but non-sequential. It carries **no** velocity/bias and **no**
> IMU preintegration; velocity/bias never cross the loop edge. The back-end turns
> `T_from_to` + `cov` into a single `BetweenFactor(from_id, to_id)` under a robust
> kernel (`01` §7.6; `Appendix R.5`). This keeps the information budget correct.

**[PROPOSE-TO-01] (a)** `LoopConstraint` carries no provenance. L5 keeps per-stage
scores/timings in an L5-private `LoopProvenance` table (§11) for telemetry, keyed
by `(from_id,to_id)` — nothing extra crosses the boundary. If `01` ever wants
provenance on the wire, add an opaque `uint64_t provenance_id` rather than
inlining fields into the graph-boundary value type.
**[PROPOSE-TO-01] (b)** The back-end's robust-kernel weighting is derived from
`fitness` inside L3 (committed Huber + off-thread GNC, `05_backend_graph.md` §8 — no
in-graph switch variable, §8.3); if instead L5 should set an explicit per-loop initial
weight, add a `double init_weight = 1.0` to `LoopConstraint`. The default derives it
in L3 from `fitness` (so no change needed).

### 3.3 Output (indirect) — de-integration via the back-end

L5 emits no map message. After the back-end optimises with the new loop, its
`GraphUpdate` (`01` §7.4) drives `IMapLayer::apply_graph_update(update, store)`
(`01` §7.5), which **clears and rebuilds** the affected region from `KeyframeStore`
clouds at corrected poses. §10 documents how this resolves MUST-FIX #4 at the L5
boundary and why L5 must *not* own the rebuild.

### 3.4 The submap accumulator (geometry granularity for A/B/C)

A single keyframe's deskewed cloud is thin: one viewpoint, sparse at range, with
little structure for a global descriptor or a stable triangle set. Both descriptor
retrieval (Stages A/B) and geometric verification (Stage C) therefore operate over
an **accumulated submap** — the composition of the **last `lc.submap_window` (N,
default 5)** keyframes ending at a given anchor keyframe, each transformed into the
anchor's body frame by the *relative* poses from the back-end's corrected estimate:
$$
\text{submap}(a)=\bigcup_{j=a-N+1}^{a}\;T_{a\_j}\cdot\text{store.cloud}(j),
\qquad T_{a\_j}=\hat T^{\text{odom}}_{a}{}^{-1}\hat T^{\text{odom}}_{j}.
$$
Composition uses the *corrected* relative pose (so a prior loop that already bent
the local chain is reflected) and is body-frame throughout, so the accumulator is
itself a body-frame cloud anchored at $a$ — the same immutability and re-placement
properties as a single keyframe (§3.1). The submap is voxel-downsampled at
`lc.submap_voxel` (default 0.25 m) after composition to bound point count and
deduplicate overlap.

> **Invariant — the keyframe stays the correction unit.** Submaps are a *geometry
> aggregation for recall and registration only*. The `LoopConstraint` L5 emits is
> still between **two single keyframe pose variables** ($a$ and the candidate
> anchor $b$), never between submap nodes. The submap exists only to build a richer
> descriptor and a denser GICP target; the relative transform Stage C produces is
> immediately expressed as the anchor-to-anchor keyframe constraint (§7.4). This
> keeps the L3 graph topology, the `KeyframePacket` boundary, and the L4
> clear-and-rebuild unit (per-keyframe clouds) all unchanged — no submap node ever
> enters iSAM2 or the `KeyframeStore`.

The accumulator is L5-private, built on demand and cached per anchor in a small LRU
sized by `lc.submap_cache`; it composes existing Shared-immutable store clouds and
never copies or mutates them. When the back-end reports a `GraphUpdate` that moves
any keyframe inside a cached submap's window, that submap entry is invalidated and
recomposed on next use, so descriptors and GICP targets always reflect the current
corrected geometry. This same accumulator is reused as the Stage-C **target** (§7).

---

## 4. The hierarchical cascade (overview)

```
add_keyframe(id,cloud,T):  compose anchor submap (§3.4); build Scan-Context (+ keys), insert into KD-tree; BTC built lazily on demand.

detect():   for the newest keyframe k —
  [A] Scan Context++ retrieval     ring-key KD-tree KNN + column-shift distance   ~µs–ms
        │ top-K candidates, each with a coarse yaw Δψ (+ SC++ lateral offset)
        ▼
  [B] STD / BTC re-rank            stable-triangle matching → 6-DoF coarse T_guess  ~ms
        │ best 1–2 candidates with an SE(3) initial guess
        ▼
  [C] GICP verification (small_gicp)  plane-to-plane GICP → refined T + fitness      ~10–100 ms
        │ accept iff fitness≥τ_fit ∧ overlap≥τ_ovl ∧ rmse≤τ_rmse ∧ cond≤τ_cond
        ▼
  [D] PCM batch consistency        max-clique on pairwise-consistent loops          ~ms
        │ keep only loops mutually consistent with the trusted set + odometry
        ▼
  return LoopConstraint[] → L3 (add_loop_constraint per loop)
```

**Why four stages and not two.** Scan Context++ is a *global* rotation-invariant
descriptor: high recall, mediocre precision — it confuses self-similar places
(corridors, parking rows; `Appendix R.1`, §6.1). STD/BTC add *local* geometric
structure (triangles of stable keypoints) that is far more discriminative *and*
yields a **6-DoF transform hypothesis** (not merely a yaw), which lands GICP inside
its convergence basin (`Appendix R.2`). GICP converts the hypothesis into a
metric transform plus a **fitness we can trust** (`Appendix R.3`). PCM is the
last line: it rejects geometrically-plausible-but-globally-inconsistent loops
(perceptual aliasing that survived A–C; `Appendix R.4`). This is the
defence-in-depth hierarchy of `Appendix R.6`. Dropping any stage measurably
degrades precision/recall on the Newer College benchmark sequences (`DATASET.md`).

---

## 5. Stage A — Scan Context++ retrieval

### 5.1 The descriptor

Scan Context (Kim & Kim, IROS 2018) encodes one scan as a 2-D image
$\mathbf I\in\mathbb R^{N_r\times N_s}$ in **polar** coordinates centred at the
sensor: rows = radial bins, columns = azimuth (yaw) sectors. **Scan Context++**
(Kim, Choi & Kim, T-RO 2021) keeps the polar context (PC) and adds a cartesian
context (CC) for lateral invariance, plus a fast two-key search and a 1-DoF
semi-metric offset (`Appendix R.1`). L5 implements the **polar (rotation-
invariant)** descriptor as the retrieval path, using SC++'s augmented search to
recover a coarse **yaw** and (CC path) a coarse **lateral** offset as a better
initial guess for Stage C.

For a body-frame point $\mathbf p=(x,y,z)$, with $\rho=\sqrt{x^2+y^2}$ and
$\theta=\operatorname{atan2}(y,x)\in[0,2\pi)$:

$$
r=\Big\lfloor \tfrac{\rho}{\rho_{\max}}N_r\Big\rfloor,\quad
s=\Big\lfloor \tfrac{\theta}{2\pi}N_s\Big\rfloor,\quad
\mathbf I[r,s]=\max_{\mathbf p\in\text{bin}(r,s)} z(\mathbf p).
$$

Bin value = **max z-height** (the original SC max-z encoding, `Appendix R.1`).
Points with $\rho>\rho_{\max}$ are dropped; empty bins are 0. The KITTI-tuned
defaults ($N_r{=}20$, $N_s{=}60$, $\rho_{\max}{=}80\text{ m}$,
i.e. 4 m/ring and 6°/sector; `Appendix R.1`) are the §12 starting points.

### 5.2 The two keys

**Ring key** $\mathbf k_{\text{ring}}\in\mathbb R^{N_r}$ — rotation-*invariant*
(row-wise occupancy ratio, the reference-impl reduction, `Appendix R.1`);
populates a KD-tree for fast candidate retrieval:
$$\mathbf k_{\text{ring}}[r]=\tfrac1{N_s}\sum_{s}\mathbb 1[\mathbf I[r,s]\neq 0].$$
**Sector key** $\mathbf k_{\text{sec}}\in\mathbb R^{N_s}$ — column-wise mean
height; the column analogue used to align yaw quickly before the full distance
(`Appendix R.1`).

### 5.3 Distance + coarse yaw

Column-wise cosine distance under a circular column shift $n$ (= yaw bin offset;
one column $=2\pi/N_s = 6°$ of azimuth, `Appendix R.1`):
$$
d(\mathbf I_q,\mathbf I_c)=\min_{n\in[0,N_s)}\;\tfrac1{N_s}\sum_{s}
\Big(1-\frac{\mathbf c_q^{s}\cdot\mathbf c_c^{(s+n)\bmod N_s}}
{\lVert\mathbf c_q^{s}\rVert\,\lVert\mathbf c_c^{(s+n)\bmod N_s}\rVert}\Big),
$$
$\mathbf c^s$ = column $s$. The minimiser $n^\star$ gives coarse yaw
$\Delta\psi=2\pi n^\star/N_s$. SC++'s fast path first aligns via the sector key's
circular cross-correlation, then refines $n$ in a small band
(`lc.sc_yaw_search_band`) instead of brute-forcing all $N_s$ shifts
(`Appendix R.1, §3.2`).

### 5.4 Algorithm

```text
function STAGE_A_build(id, cloud, T_map_body):    # inside add_keyframe
    sm    = submap(id)                             # last-N composed kfs, body frame at id (§3.4)
    I     = make_scan_context(sm)                  # Nr×Ns, max-z bins (§5.1) over the submap
    kring = ring_key(I);  ksec = sector_key(I)
    DB.put(id, {I, kring, ksec, pose=T_map_body, stamp=packet_stamp(id)})
    DB.ringtree.insert(kring -> id)                # KD-tree over ring keys

function STAGE_A_query(k) -> ScCandidate[]:        # k = newest id
    cand = DB.ringtree.knn(DB[k].kring, lc.sc_knn)            # ~15
    out = []
    for id in cand:
        if DB[k].stamp - DB[id].stamp < lc.min_time_gap: continue # anti self-loop
        if k - id < lc.min_kf_gap:                  continue
        if ‖DB[k].pose.t - DB[id].pose.t‖ > lc.sc_max_xy: continue # loose spatial sanity
        d, dpsi = sc_distance_and_yaw(DB[k].I, DB[id].I)          # §5.3
        if d < lc.sc_dist_thresh:                                  # ~0.13
            out.push({id, sc_dist=d, yaw_guess=dpsi})
    sort out by sc_dist ascending
    return out[0 : lc.sc_topK]                                     # ~5
```

The descriptor is built over the **submap** (§3.4), not the bare keyframe cloud: a
denser, multi-viewpoint polar image is more repeatable across a revisit and far
less sparse at range, which is the standard recall fix for single-scan descriptors
on long-range / sparse data. The descriptor matrix and keys are still anchored at
keyframe `id` and keyed by `id`, so retrieval, gating, and the eventual constraint
remain per-keyframe (§3.4 invariant).

**Complexity.** Build: $O(P)$ in submap points + $O(\log N)$ KD-tree insert (submap
composition is $O(N\cdot P_{\text{kf}})$ over the small window, then one
downsample). Query: one KNN ($O(\log N)$) + `sc_knn` cosine alignments, each
$O(N_r N_s\cdot\text{band})$. With $N_r{=}20,N_s{=}60$ this is sub-millisecond
(`Appendix R.1`, two-stage hierarchical search). If Stage A returns empty (the
common case) the cascade stops — this must be the fast path.

> **Never accept on SC distance alone.** The SC distance threshold is
> dataset-dependent and SC is precision-limited; `Appendix R.1` is explicit
> that a loop is *never* accepted on SC distance — Stage A only proposes
> candidates, which Stages C and D must confirm.

---

## 6. Stage B — STD / BTC re-ranking

### 6.1 Why a second, local descriptor

Scan Context cannot tell "this corridor" from "that identical corridor." **STD**
(Stable Triangle Descriptor, Yuan et al., ICRA 2023) and its successor **BTC**
(Binary + Triangle Combined, Yuan et al., 2024) encode *local* geometry: extract
stable keypoints (plane intersections / boundary extrema), form triangles from
keypoint triples, describe each by its **sorted side lengths**
$(\ell_1\le\ell_2\le\ell_3)$ plus the **angles between the vertex-plane normals**
— a 6-D signature invariant to rotation *and* translation (`Appendix R.2`).
Triangle matching both **re-ranks** Stage A's candidates *and* yields a **6-DoF
coarse transform** via the keypoint correspondences (`Appendix R.2`).

L5 implements **BTC** in the first build: each keypoint carries the STD triangle
descriptor **plus** a binary appearance code (a local occupancy bit-string), and
BTC is a strict **superset** of STD — set the appearance code to a constant and the
re-rank degrades exactly to STD triangle matching. The binary code is a cheap
Hamming-distance pre-filter ahead of the triangle hash vote, and it is precisely
the discriminator that breaks corridor/parking-row-style aliasing (two
self-similar places with identical triangle geometry but different local
occupancy). The interface (§11) is descriptor-agnostic; this is one descriptor
pipeline (SC++ retrieve → BTC re-rank), not two competing pipelines.

**The binary appearance code.** For each retained keypoint, voxelise a fixed-radius
local neighbourhood (`lc.btc_local_radius`, `lc.btc_bin_count` bins along the
keypoint's dominant plane-normal axis) and set bit $b$ iff bin $b$ is occupied
above `lc.btc_occupancy_min` points; a packed summary byte (popcount band) gates
the full Hamming compare. Two keypoints are appearance-compatible iff their codes'
Hamming similarity $\ge$ `lc.btc_hamming_min`; only appearance-compatible keypoints
are allowed to vote in the triangle hash (§6.3). Because the code is derived from
the same local plane fit already computed for keypoint extraction (§6.2), it adds
no new neighbourhood search.

### 6.2 Keypoint and triangle extraction

Keypoint and triangle extraction runs over the **submap** anchored at the keyframe
(§3.4), the same denser geometry the Stage-A descriptor used; this yields more
stable keypoints and more repeatable triangles than a single sparse scan, while the
descriptor and the eventual constraint stay keyed to the anchor keyframe.

1. **Plane extraction:** voxelise the anchor's submap cloud; per voxel fit a plane
   by the same procedure the reference odometry uses for point-to-plane —
   `esti_plane(pabcd, points_near, 0.1)` (`FAST_LIO/src/laserMapping.cpp:678`),
   which solves the normal equation $A\,\mathbf n=-\mathbf 1$ by QR over the
   neighbourhood and gates on planar residual. (Reference fits over exactly
   `NUM_MATCH_POINTS=5` neighbours, `common_lib.h:26`; here we fit over a voxel's
   full membership — same math, more support.) An equivalent eigen-decomposition
   of the per-voxel scatter matrix is the FAST-LIVO2 `init_plane` route
   (`FAST-LIVO2/src/voxel_map.cpp:55–86`).
2. **Keypoints** = stable extrema along plane-pair intersection lines / boundary
   points, repeatably detectable across viewpoint — the "stable" criterion of
   `Appendix R.2`. Keep the strongest `lc.std_max_keypoints`.
3. **Triangles:** connect each keypoint to its `lc.std_knn_kp` nearest keypoints;
   keep triangles with all sides in `[lc.std_side_min, lc.std_side_max]` (rejects
   degenerate slivers and over-large triangles).
4. **Descriptor** = sorted side triple $(\ell_1,\ell_2,\ell_3)$ + the three
   inter-normal angles $(\alpha_1,\alpha_2,\alpha_3)$ — the 6-D vector of
   `Appendix R.2` — **plus the per-vertex binary appearance code** (§6.1). The
   triangle's vertices fix a local reference frame, so a single matched triangle
   pair already yields a 6-DoF pose hypothesis; the appearance codes disambiguate
   triangle pairs that are geometrically identical but sit in different places.

### 6.3 Matching → coarse transform

```text
function STAGE_B(k, scCandidates) -> RankedCand[]:
    Tq = STD_DB.get_or_build(k)                   # triangles for query
    ranked = []
    for c in scCandidates:
        Tc = STD_DB.get_or_build(c.id)
        matches = match_triangles(Tq, Tc, side_tol = lc.std_side_tol,   # hash by quantised triple
                                  hamming_min = lc.btc_hamming_min)      # BTC appearance pre-filter
        if matches.size < lc.std_min_matches: continue                  # ~4
        (T_guess, inliers) = ransac_rigid(matches,
                                thresh = lc.std_ransac_eps,
                                iters  = lc.std_ransac_iters,
                                yaw_prior = c.yaw_guess)                 # seed from Stage A
        if inliers < lc.std_min_inliers: continue                       # ~5
        scoreB = combine(c.sc_dist, inliers/matches.size, mean_residual)# §6.4
        ranked.push({id=c.id, T_guess, inliers, scoreB})
    sort ranked by scoreB (best first)
    return ranked[0 : lc.std_topK]                                      # ~2
```

Matching uses a **hash table keyed by the quantised side-length triple** with
secondary angle/attribute checks, exactly the hash-voting of `Appendix R.2`
(`key = (round(ℓ1/Δ), round(ℓ2/Δ), round(ℓ3/Δ))`), so it is *not* an all-pairs
descriptor comparison. The **BTC appearance pre-filter** runs inside the vote: a
hashed triangle pair is admitted only if each matched vertex pair's binary codes
clear `lc.btc_hamming_min` Hamming similarity (the cheap summary-byte popcount band
rejects most pairs before the full code compare). This is the cheap, high-precision
discriminator against self-similar geometry; setting `lc.btc_hamming_min = 0`
disables it and recovers pure STD matching. `ransac_rigid` solves the 3-point absolute-orientation
(Umeyama/Horn closed form) per RANSAC sample over the voted correspondences and
counts keypoint-residual inliers; the **Stage-A yaw prior** rejects samples whose
yaw deviates from $\Delta\psi$ by more than `lc.std_yaw_gate`, which cuts
iterations and false alignments dramatically (yaw is the most ambiguous DoF for
ground vehicles).

### 6.4 Combined score

$$
\text{score}_B=w_{sc}(1-d_{sc})+w_{in}\frac{n_{\text{in}}}{n_{\text{match}}}
+w_{res}\exp\!\Big(-\frac{\bar r}{\sigma_r}\Big),\qquad
\mathbf w=\texttt{lc.scoreB\_w}=\{0.2,0.5,0.3\}.
$$

Stage B is a *re-ranker + initialiser*, not the final arbiter: it prunes to the
best 1–2 and hands Stage C a 6-DoF `T_guess` good enough to converge
(`Appendix R.2`: STD as the high-precision verifier/initialiser feeding the
same GICP + PCM back end).

---

## 7. Stage C — GICP geometric verification (small_gicp)

### 7.1 Role

A–B propose *where* and *roughly how*; C produces a **metric, trustworthy**
relative transform plus a **fitness** the rest of the system scales covariance by.
We use **GICP** (Generalized-ICP, Segal, Haehnel & Thrun, RSS 2009) —
plane-to-plane / distribution-to-distribution — far more robust to partial overlap
and differing sampling than point-to-point ICP, with no explicit normal
computation (`Appendix R.3`). The descriptor stage is not metrically accurate
enough to be a graph constraint and SC can produce false positives; GICP both
refines the pose to `BetweenFactor` accuracy and produces the fitness that gates
acceptance (`Appendix R.3`).

**Library (single choice, no hedging).** Stage C uses **`small_gicp`**
(`koide3/small_gicp`) — the library mandated for this component by the Meridian
library canon (`00` §0 LIBRARY CANON; `11_build_system_libraries.md` §3, "Fine
registration (loop-closure GICP verify) → small_gicp"). It is header-light,
multi-threaded (OpenMP/TBB), takes Eigen point buffers directly (no PCL
registration stack in the hot path), supports GICP/VGICP, and — critically —
returns the linearised Hessian $\mathbf H$ at convergence, which §9.2 turns into
the loop covariance (`Appendix R.3`). PCL GICP and libpointmatcher were
considered and rejected for dragging the full PCL registration stack / being
slower (`11` §10 rejected-alternatives table). L5 builds the target's spatial
index with `small_gicp`'s bundled nanoflann KD-tree; the front-end's own
voxel-hash map (`11` §1, spec 06) is unrelated to Stage C's
index.

### 7.2 GICP cost

Source point $\mathbf p_i$ (query kf), correspondence $\mathbf q_i$ (candidate kf),
each with a surface covariance $C^P_i,C^Q_i$ estimated from its k-NN. For
$\mathbf T=(\mathbf R,\mathbf t)$, residual $\mathbf d_i=\mathbf q_i-\mathbf T\mathbf p_i$,
the maximum-likelihood estimate weights each residual by the combined covariance
(`Appendix R.3`):
$$
\hat{\mathbf T}=\arg\min_{\mathbf T}\sum_i \mathbf d_i^\top
\big(C^Q_i+\mathbf R\,C^P_i\,\mathbf R^\top\big)^{-1}\mathbf d_i.
$$

**Surfel covariance.** `small_gicp` fits a plane to each point's neighbourhood and
sets $C_i=\mathbf U\,\mathrm{diag}(1,1,\epsilon)\,\mathbf U^\top$ with $\mathbf U$
the plane eigenbasis and $\epsilon$ small (the GICP "$\epsilon$-trick" that makes
the cost behave like point-to-plane — the same residual family as the reference
odometry's $r_i=n_i^\top(R p_i+t-q_i)$). The per-kf covariances are computed once
and cached lazily in L5's per-kf side table (§11) so re-verification is cheap;
this is the same plane/scatter machinery cited in §6.2.

### 7.3 Correspondence search

Correspondences are nearest neighbours of the *transformed* source in the target's
spatial index, the standard GICP inner loop. `small_gicp` builds the KD-tree
(bundled nanoflann) over the target cloud and re-queries each iteration after
applying the current $\mathbf T$, gated by `lc.gicp_max_corr_dist` (the library's
`max_correspondence_distance`, `Appendix R.3`). This is the same
nearest-neighbour correspondence step the reference odometry runs in
`h_share_model` (`FAST_LIO/src/laserMapping.cpp:638`, `Nearest_Search` at `:670`)
— L5 just lets `small_gicp` own the index rather than hand-rolling it. Because the
target is a multi-keyframe submap (§3.4) rather than a single scan, the target
point count is larger; the VGICP variant (`setting.type = VGICP`,
`lc.gicp_voxel_res`) is the speed path when the downsampled submap is still large,
while plain GICP remains fine for a modest submap window (`Appendix R.3`).

### 7.4 Algorithm

```text
function STAGE_C(k, rankedB) -> VerifiedLoop[]:
    src = downsample(store.cloud(k), lc.gicp_downsample)          # §7.5 voxel grid; single query kf
    out = []
    for rc in rankedB:
        tgt   = downsample(submap(rc.id), lc.gicp_downsample)     # §3.4 candidate submap (reuse accumulator)
        # small_gicp builds the target KD-tree + per-point surfel covariances internally,
        # cached in surfels[] (§7.2). init = T_guess (6-DoF) from Stage B / yaw from Stage A.
        res = small_gicp::align(tgt, src, init = rc.T_guess, setting)   # §7.2 cost
        T   = res.T_target_source                                       # = T_from_to (from=rc.id, to=k)
        (fitness, overlap, rmse, cond) = score(res, src, tgt, T)        # §7.5
        if res.converged
           and fitness >= lc.gicp_fitness_min  and overlap >= lc.gicp_overlap_min
           and rmse <= lc.gicp_rmse_max        and cond   <= lc.gicp_cond_max:
            out.push(VerifiedLoop{from=rc.id, to=k, T, fitness, overlap, rmse, cond,
                                  info6 = res.H, descriptor_kind = rc.kind})
        else:
            log_reject(rc.id, reason)                          # §14
    return out
```

**GICP target = candidate submap.** The registration *target* is the candidate
keyframe's **submap** (§3.4) — the same accumulator already composed and cached for
that anchor in Stages A/B, so building the GICP target is free of new composition.
A dense, multi-viewpoint target gives the source far more surface to find
correspondences against, which is the dominant recall win for sparse / long-range
revisits. The *source* may remain the single query keyframe cloud `k`: it keeps the
correspondence search cheap and the source needs only to lie inside the target's
coverage. (A symmetric submap source is available behind `lc.gicp_source_submap`,
default off, when the query scan is itself too sparse to register.)

Because both submaps are body-frame clouds **anchored at their respective
keyframes** (`rc.id` and `k`), `res.T_target_source` is already the
keyframe-`rc.id`-to-keyframe-`k` body transform — there is no submap-node frame to
unwind. Note `from = rc.id` (older) and `to = k` (newer), and `T` (`small_gicp`'s
`res.T_target_source`, target = older candidate's submap, anchored at `rc.id`) maps
`to`'s body into `from`'s body frame, matching `LoopConstraint` semantics
(`01` §7.6; `Appendix R.5` inverts the helper's convention to the `BetweenFactor`
one — here `target` is already the older `from`, so the mapping is direct). This is
the §3.4 invariant in force: the submap densifies the registration but the emitted
constraint is the keyframe-to-keyframe transform. `res.H` is the registration
Hessian (`Appendix R.3`), the base information for §9.2.

### 7.5 Fitness, overlap, condition number (these feed §9)

- **overlap** $=\dfrac{\#\{\text{src points with a corr.}<d_{\max}\}}{\#\text{src}}$
  (the inlier ratio of `Appendix R.3`).
- **rmse** $=\sqrt{\tfrac1M\sum_i\lVert\mathbf d_i\rVert^2}$ over inliers
  (the mean-error gate of `Appendix R.3`).
- **fitness** $\in[0,1]$ — single trust scalar (the field `LoopConstraint.fitness`):
  $$\text{fitness}=\text{overlap}\cdot\exp\!\Big(-\tfrac{\text{rmse}}{\sigma_{\text{fit}}}\Big),\quad
  \sigma_{\text{fit}}=\texttt{lc.gicp\_fit\_sigma}\ (0.1\text{ m}).$$
  Acceptance is `inlier_ratio ≥ τ` **and** `mean_error ≤ τ` (`Appendix R.3`),
  expressed here as the `gicp_overlap_min` / `gicp_rmse_max` gates of §7.4.
- **cond** = condition number of the GICP information matrix
  $\mathbf H=\sum_i J_i^\top\Omega_i J_i$ (`res.H`, largest/smallest eigenvalue).
  High cond ⇒ geometrically degenerate alignment (e.g. a featureless tunnel
  constrains rotation but not along-axis translation). This is the loop-level
  analogue of L2's per-axis observability (`01` §3.4) and drives the per-axis
  covariance inflation in §9.2.

### 7.6 Optional photometric tie-break (OFF by default)

If `lc.use_photo_tiebreak` and both keyframes have an `image` + `T_body_cam`
(`01` §6.1, §4.3), project a sparse set of high-gradient pixels from the candidate
into the query at $\mathbf T$ and compute a normalised photometric residual
(FAST-LIVO2-style sparse-direct patch). Used **only to break ties / reject** when
two candidates pass §7.4 similarly — it never *creates* a loop. Off by default
because exposure/illumination differs across revisits; enabled only for
camera-rich, stable-lighting deployments. (The full LiDAR+camera fusion path is
iBTC, `Appendix R.2` — a future extension, not this tie-break.)

### 7.7 Output

`VerifiedLoop{from, to, T, fitness, overlap, rmse, cond, info6, descriptor_kind}`
(L5-internal, §11). `info6` ($=\mathbf H$ at the optimum) is the *base* covariance
source before fitness scaling. Empty ⇒ no loop this round (stop).

---

## 8. Stage D — PCM batch consistency

### 8.1 Why PCM

A GICP-verified loop can still be *wrong* under perceptual aliasing (two genuinely
similar places that each match the query but imply contradictory drift,
`Appendix R.4`). **PCM** — Pairwise Consistent Measurement set maximization
(Mangelson et al., ICRA 2018) — ignores descriptors entirely and asks a purely
*geometric* question: **is this loop mutually consistent with the loops we already
trust and the odometry chain between their endpoints?** It then selects the
**largest mutually-consistent subset** (a maximum-clique problem on a consistency
graph), discarding the rest (`Appendix R.4`). PCM is the *front gate*; the
back-end GNC/Huber/switchable is the *safety net* — the same front-gate / safety-net
division of labour as Appendix R.4, R.5 and the back-end spec (`05_backend_graph.md`).

### 8.2 Pairwise consistency test

Two loops $z_{ik}$ (between kfs $i,k$) and $z_{jl}$ (between $j,l$) are consistent
if traversing loop → odom → loop⁻¹ → odom returns to identity. On $SE(3)$
(`Appendix R.4`):
$$
\mathbf r_{ik,jl}=\log\!\Big(\hat T_{ij}^{\text{odom}}\cdot
\hat T_{jl}^{\text{loop}}\cdot \hat T_{lk}^{\text{odom}}\cdot
\hat T_{ki}^{\text{loop}}\Big)^{\vee}\in\mathbb R^6,
$$
with the Mahalanobis gate
$$
\mathbf r^\top\Sigma^{-1}\mathbf r\le\chi^2_{6,\alpha},\quad
\alpha=\texttt{lc.pcm\_chi2\_conf}=0.99\ \Rightarrow\ \chi^2_{6,0.99}\approx16.81.
$$
The odometry composites $\hat T^{\text{odom}}$ come from the current L3 estimate
(chaining between-factors / iSAM2 marginals via the back-end's
`corrected_trajectory()` query, `01` §7.4); $\Sigma$ propagates the two loops'
`cov` and the odometry-chain covariance. The keyframes'
`KeyframePacket.observability` and the back-end covariance enter here — long
odometry chains between far-apart loop endpoints are appropriately uncertain, so
PCM is *not* over-strict on distant loops.

The odometry-chain covariance $\Sigma_{ij}^{\text{odom}}$ between two endpoints is
the **chained corrected-odometry covariance**: the right-composition of the
per-edge between-factor covariances along the keyframe path $i\to j$, each Adjoint-
transported into the accumulating frame, sourced from the back-end's corrected
estimate so it reflects the current (post-prior-loop) linearisation rather than the
raw front-end stream. $\Sigma$ in the gate above is then
$\Sigma_{ik}^{\text{loop}}+\Sigma_{ij}^{\text{odom}}+\Sigma_{jl}^{\text{loop}}+\Sigma_{lk}^{\text{odom}}$,
all in `PoseCov6` order; it is PSD-guarded (clamp eigenvalues to
`lc.cov_psd_floor` before inverting) so a degenerate composite never produces a
negative or singular information matrix that would silently admit or reject the
loop. The same chained covariance is what makes the single-loop gate of §8.2.1
meaningful.

#### 8.2.1 Single-loop odometry-consistency gate (before max-clique)

A lone new loop forms a 1-node clique and so passes max-clique unconditionally
(§8.3). To give PCM teeth from the **first** loop in a session, every candidate is
first tested against the trajectory it claims to close, *independently of any other
loop*. A single loop $z_{ik}$ asserts that the relative pose between endpoints $i$
and $k$ equals $T^{\text{loop}}_{ik}$; the corrected odometry chain between the same
endpoints already asserts $\hat T^{\text{odom}}_{ik}$. The two must agree within
noise:
$$
\mathbf r^{\text{self}}_{ik}=\log\!\Big(\hat T^{\text{odom}}_{ik}{}^{-1}\,
T^{\text{loop}}_{ik}\Big)^{\vee}\in\mathbb R^6,\qquad
\mathbf r^{\text{self}\top}\,\big(\Sigma^{\text{loop}}_{ik}+\Sigma^{\text{odom}}_{ik}\big)^{-1}\,
\mathbf r^{\text{self}}\le\chi^2_{6,\alpha}.
$$
$\Sigma^{\text{odom}}_{ik}$ is the chained corrected-odom covariance of §8.2, so the
gate is **loose where it should be loose** — after long drift the accumulated odom
covariance is large, the gate widens, and a genuine large-correction loop is *not*
rejected merely for disagreeing with badly-drifted dead-reckoning; conversely a
loop that contradicts a still-confident short chain is rejected immediately. A
candidate that fails this self-test is dropped before it ever enters the
consistency graph and is logged with reason `PCM_ODOM_INCONSISTENT` (§14). This is
the gate that closes the §13.10 "first session, no loops yet" gap.

> **Keep the χ² quantile in lock-step with the back-end (binding).** The PCM gate
> here, the single-loop self-test, and the back-end's GNC inlier threshold must all
> use the **same squared-Mahalanobis convention and the same χ² quantile**. That
> convention — residual whitened as $d^2=\epsilon^\top\Omega\epsilon$ gated on
> $\chi^2_{6,\alpha}$, and the GNC `setInlierCostThresholds(barc2=\chi^2_{6,0.99})`
> derived from it — is defined canonically in `05_backend_graph.md` §3.2 (the binding
> system-wide clause, applied there in §7.1 and §8); this spec does **not** restate
> it. `lc.pcm_chi2_conf` is bound equal to the
> back-end's `backend.pcm_chi2_alpha`; a mismatch means the front gate and the
> safety net judge the same loop on different scales, a silent statistical bug.

### 8.3 Max-clique selection

```text
function STAGE_D(new_loops, trusted) -> accepted:
    survivors = [z in new_loops if odom_consistent(z)]      # §8.2.1 single-loop self-test
    L = trusted ∪ survivors
    graph G: node per loop; edge(a,b) iff pairwise_consistent(a,b)        # §8.2
    require self-consistency (each loop's own GICP cond/fitness gate, §7.4)
    C = max_clique(G, budget = lc.pcm_maxclique_ms)   # Pattabiraman/PMC heuristic, §16
    accepted = C ∩ survivors          # emit only the new loops that joined the clique
    trusted  = C                      # update the trusted set
    return accepted
```

**Notes.** Exact max-clique is NP-hard but the consistency graph is small (number
of loops, not keyframes) and sparse; a fast heuristic (PMC, as used by
Kimera-RPGO, `Appendix R.4`) capped at `lc.pcm_maxclique_ms` suffices. Run PCM
**incrementally** (Kimera-RPGO style, `Appendix R.4`): keep the trusted clique,
test only new loops against it + each other, so a loop that later proves
inconsistent is excluded before it pollutes iSAM2. **Single-loop case:** the
odometry-consistency self-test (§8.2.1) is what filters a lone loop — the 1-node
clique it would otherwise form is no longer a free pass. The pairwise max-clique
then adds *cross-loop* mutual consistency on top, and its discriminative power
still grows with loop count (§13.10). A loop rejected once (by either gate) is held
in a `lc.pcm_quarantine` ring buffer and may be re-tested when more evidence
accrues.

### 8.4 Output

Loops in the consistent clique become `LoopConstraint`s emitted to the back-end.
Rejected loops are logged with reason `PCM_INCONSISTENT` (§14). (Because
`LoopConstraint` carries no PCM flag, L5 only emits loops it has *already* vetted;
the back-end's batch GNC additionally validates the loop subset over the affected
sub-graph before commit — `Appendix R.5` — treating L5's output as candidates.)

---

## 9. Building the LoopConstraint (fitness-scaled, robustness series)

This is the L5→L3 deliverable. A loop passes through a series of robustness
mechanisms — PCM pre-admission, then committed-Huber, then batch-GNC (§9.3) —
arranged as defence in depth; L5 owns the first (covariance shaping and PCM) and
*feeds* the rest with a trustworthy `cov`/`fitness` (`Appendix R.5`).

### 9.1 What L5 emits vs what L3 does

L5 emits a single `LoopConstraint` (`01` §7.6): `from_id, to_id, T_from_to, cov,
fitness`. The back-end (`05_backend_graph.md`) turns it into one
`BetweenFactor(from_id, to_id)` with `T_from_to` + `cov`, robustified by the
committed-Huber-then-batch-GNC series (§9.3), with a **switchable constraint** as a
reserved alternative whose prior it would derive from `fitness` (`Appendix R.5`).
L5's job is to make `cov` and `fitness` *trustworthy inputs* to those mechanisms.
No velocity/bias, no preintegration cross the loop (§3.2).

### 9.2 Fitness-scaled, observability-aware covariance (`cov`)

Build the $6\times6$ covariance from the `small_gicp` information matrix
($\mathbf H=$ `res.H`, `Appendix R.3`), then inflate by (a) inverse fitness and
(b) per-axis degeneracy:
$$
\Sigma_{\text{loop}}=s(\text{fitness})\cdot\big(\mathbf H_{\text{gicp}}+\lambda\mathbf I\big)^{-1},
\qquad
s(\text{fitness})=\Big(\frac{\text{fitness}_{\text{ref}}}{\max(\text{fitness},\text{fit}_{\min})}\Big)^2 .
$$

This is the $\Sigma_{\text{loop}} = H^{-1}/\max(f,f_{\min})$ rule of
`Appendix R.5` (registration-Hessian start, fitness-scaled), made
observability-aware:

- $\mathbf H_{\text{gicp}}$ at convergence (§7.5); $\lambda\mathbf I$
  (`lc.cov_lambda`) regularises degenerate directions so $\Sigma$ is finite. The
  `f_min` clamp stops a near-zero-fitness loop from blowing up the covariance
  (`Appendix R.5`) — such loops should already be rejected at §7.4.
- **Per-axis degeneracy inflation.** Along eigen-directions of
  $\mathbf H_{\text{gicp}}$ with eigenvalue below `lc.cov_degenerate_eig`, multiply
  that axis's variance by `lc.cov_degenerate_mult` (e.g. 100×). This is the
  loop-level analogue of L2's per-axis observability *and* folds in the endpoints'
  `KeyframePacket.observability` (a loop touching an under-observed axis on either
  endpoint is loosened on that axis, after rotating each report into the factor
  frame — `01` §3.4 mandates a named frame precisely so this rotation is correct).
  Net effect: a tunnel loop confident in rotation but not along-axis translation
  bends yaw yet barely tugs along the tunnel — exactly right.
- **Tangent order.** $\Sigma$ is laid out in `PoseCov6` order (the `Pose` tangent
  $[\rho;\phi]$, `01` §3.1/§3.3). The GICP/`small_gicp` $\mathbf H$ is permuted into
  that order at the boundary; do this *once*, here, not in the back-end (a classic
  sign/ordering trap — note GTSAM `Pose3` uses $[\text{rot};\text{trans}]$,
  `Appendix R.5`; the adapter that converts must be deliberate).
- Lower fitness ⇒ larger $s$ ⇒ looser factor. With $\text{fitness}_{\text{ref}}=
  \texttt{lc.gicp\_fitness\_min}$, a barely-passing loop is ≈unit-scaled; an
  excellent loop ($\to1$) gets tighter. This is exactly the `01` §7.6 / `Appendix R.5`
  promise: "a low-fitness GICP match yields a loose `cov`, so a marginal loop
  barely tugs the graph."

### 9.3 The robustness series: PCM → committed-Huber → batch-GNC

The three robustness mechanisms a loop passes through are arranged **in series**,
in the order the loop reaches them along the L5→L3 path. L5 owns the first rung;
the back-end (`05_backend_graph.md`) owns the latter two. L5's only obligation to
both is a sane `cov` (§9.2) and a trustworthy `fitness`, so the whitened residual
the back-end forms is meaningful.

1. **PCM pre-admission (L5, §8).** A discrete consistent-set gate: a loop that is
   self-inconsistent with the corrected odometry (§8.2.1) or mutually inconsistent
   with the trusted clique (§8.2/§8.3) never becomes a factor at all.
2. **Committed Huber, incremental (back-end).** A loop that PCM admits enters
   iSAM2 immediately under an always-on **Huber** kernel, so it can begin
   correcting the trajectory at keyframe rate without waiting for a batch solve and
   without any single residual dominating. This is the cheap incremental safety net
   that runs inside the normal `ISAM2::update` path.
3. **Batch-GNC consolidation, off-thread (back-end).** Graduated Non-Convexity is a
   *batch* schedule (it re-anneals over the whole sub-problem) and is therefore run
   off the incremental thread, over the **loop sub-graph** the new loop touches,
   to reach the hard inlier/outlier decision. A loop driven to a near-zero GNC
   weight is removed and returned to the PCM quarantine; a confirmed inlier is
   re-committed at its consolidated weight.

L5 does not implement, schedule, or sequence any of rungs 2–3 — the kernel
mechanics (Huber tuning, the GNC anneal schedule and weight update, the
`setKnownInliers(odometry)` declaration, and the incremental-vs-batch split) are
defined in `05_backend_graph.md` §8 and must not be restated or contradicted here.
What this spec fixes is the **placement on the loop side**: PCM is the front gate
*before* the factor exists; Huber-then-GNC is the safety net *after* it does; and
the statistical scale on which all of them judge a loop is the single canonical χ²
convention of `05_backend_graph.md` §3.2 (§8.2.1). The converged GNC weight is
surfaced as a debug/audit signal (§14), read back from the back-end since L5 does
not run the optimiser.

> **Why all three.** PCM is a *discrete pre-admission* gate (consistent-set
> selection before the factor enters the graph); committed Huber gives *immediate,
> incremental* robustness so a freshly admitted loop is useful at once without one
> residual running away; batch-GNC is the *deliberate* outlier decision that also
> catches the case where the *odometry* (not the loop) was wrong. Complementary,
> not redundant — the front-gate / safety-net split of `Appendix R.4, §7.3`.

### 9.4 Post-admission disabling is GNC, not an in-graph switch variable

A loop that PCM admits can still turn out wrong; the back-end must be able to **undo**
it after admission. Meridian does this through the §9.3 series — the committed Huber
kernel down-weights a borderline residual incrementally, and the off-thread batch GNC
(`05_backend_graph.md` §8.3) drives a confirmed outlier's weight to zero and removes
the factor. `LoopConstraint.fitness` feeds that machinery: a confident loop ships a
tight `cov` (§9.2) and a high `fitness` so GNC keeps it; a marginal loop ships a loose
`cov` and a low `fitness` so a small disagreement is enough for GNC to shed it.

Sünderhauf-style **switchable constraints** (Sünderhauf & Protzel, IROS 2012) — one
real switch variable $\sigma\in[0,1]$ per loop, pulled toward a prior — are the
classic alternative for this "turn a loop off smoothly after admission" job, but they
are **deliberately not used** in Meridian: a switch variable per loop grows the iSAM2
Bayes tree and makes that region non-convex, exactly the relinearisation churn the
incremental solver must avoid. `05_backend_graph.md` §8.3 makes this the binding
decision (GNC subsumes the switch; no switch variable enters the graph), and this spec
does not contradict it — L5 only supplies `cov` + `fitness`; it neither adds a switch
variable nor assumes the back-end does. See [PROPOSE-TO-01] (b) in §3.2 if the
fitness-derived weighting choice
should move into L5.

### 9.5 Submission

```text
function EMIT(v: VerifiedLoop) -> LoopConstraint:
    cov = shape_cov(v.info6, v.fitness,                       # §9.2 (perm to PoseCov6 order)
                    obs[v.from], obs[v.to], lc)               # endpoint observability
    record provenance{v.from, v.to, sc_dist, std_inliers,     # L5-private telemetry (§14)
                      v.fitness, v.overlap, v.rmse, v.cond, stage_ms[4]}
    return LoopConstraint{ from_id=v.from, to_id=v.to,
                           T_from_to=v.T, cov=cov, fitness=v.fitness }
```

---

## 10. Interaction with L4: retained clouds and de-integration

This section resolves **MUST-FIX #4** at the L5 boundary, consistently with
`01` §7.5 and `06_mapping.md`.

### 10.1 The retained store (consumer view)

L5 is one of **three consumers** of the `KeyframeStore` (`01` §7.5 names them):
(i) L4 nvblox TSDF re-integration, (ii) L5 GICP, (iii) optional final mesh export.
Each keyframe holds an **immutable, body-frame, deskewed** cloud. Immutability +
body-frame storage is the entire trick: a loop correction changes *poses*, not
clouds, so re-integration is "re-place and re-fuse," not "edit points."

L5 reads via `store.cloud(id)` / `store.image(id)` / `store.within_radius(c, r)`
(`01` §7.5); clouds are Shared-immutable so a handle L5 holds during GICP cannot be
evicted out from under it (ref-counted, `01` §2.4). L5 caches *its own* surfel
covariances in a side table (§7.2); it never writes the store.

### 10.2 What L5 does (and does not) do

L5 **does not** clear or rebuild the map and **does not** send a region message.
The chain is:

```
L5.detect() → LoopConstraint  →  L3.add_loop_constraint(lc) ; L3.optimize() → GraphUpdate
                                                          │
                                                          ▼
                       L3 forwards GraphUpdate → IMapLayer.apply_graph_update(update, store)
                                                          │  (clear & rebuild, 01 §7.5)
                                                          ▼
                       L4 asks store.within_radius for affected kfs, clears those voxels,
                          re-integrates store.cloud(id) at corrected poses:
                          p_map = T_map_body^new · p_body
```

The body→world placement is the exact `pointBodyToWorld` operation
(`laserMapping.cpp:178`). L5's only de-integration responsibilities are: (1) being
the *cause* (emitting the loop), and (2) for debug, observing the `GraphUpdate`
the back-end exposes and visualising the dirty region (§14). This matches the
worked example in `01` §10 (loop on kf 412↔27 → `GraphUpdate` → `apply_graph_update`).

### 10.3 Why CLEAR-AND-REBUILD, not per-voxel subtraction (the MUST-FIX #4 decision)

This decision lives in `01` §7.5 and `06_mapping.md`; restated for L5 context:
nvblox TSDF voxels are **running weighted averages** with no exact inverse —
per-measurement "subtraction" would require storing every contribution per voxel
(huge) and is numerically unstable. The principled operation matching
running-average semantics is to **discard affected voxels and re-integrate the
retained clouds at corrected poses** (`00` §9.5; the same logic runs on whichever
`ISurfaceMap` backend is selected — nvblox GPU or cpu host). Because clouds are
immutable + body-frame, this is mechanical. L5 must therefore *not* attempt its own
incremental map edit — that
would duplicate and contradict L4.

### 10.4 Burst coalescing (advisory)

A loop burst after long drift can move many keyframes at once. Coalescing repeated
rebuilds into one over the union of affected regions is an **L4 concern**
(`apply_graph_update` rate-limiting / double-buffering, `06_mapping.md`). L5's
contribution is upstream: PCM admits only the mutually-consistent subset in one
batch (§8), so the back-end sees one coherent `GraphUpdate` rather than a thrash of
contradictory loops.

---

## 11. Data structures and interfaces (C++)

L5 lives in `meridian_place` (`00_architecture.md` §2). The **only** types crossing
L5's boundaries are the `01` types (`KeyframePacket` is read indirectly;
`LoopConstraint`, `Pose`, `PoseCov6`, `ObservabilityReport`, `LidarPoint`,
`KeyframeStore` cross or are queried). Everything below is L5-**internal**.

```cpp
// ---- Descriptor DB entry (internal) -------------------------------------
struct ScanContext {
  Eigen::MatrixXf I;                 // Nr x Ns (max-z bins)
  Eigen::VectorXf ring_key;          // Nr   (rotation-invariant, KD-tree key)
  Eigen::VectorXf sector_key;        // Ns   (yaw/lateral alignment, SC++)
};
struct BtcCode {                     // per-keypoint binary appearance code (§6.1)
  std::array<uint64_t, BTC_WORDS> bits;  // packed local-occupancy bit-string
  uint8_t  summary;                  // popcount band; gates the full Hamming compare
};
struct BtcTriangle {                 // one BTC descriptor: STD triangle (6-D) + per-vertex code
  std::array<float,3> sides;         // sorted side lengths (l1<=l2<=l3)
  std::array<float,3> n_angles;      // angles between the three vertex-plane normals
  std::array<Eigen::Vector3f,3> kp;  // keypoints (body frame, submap-anchored)
  std::array<BtcCode,3>          code;   // appearance code per vertex (set const => pure STD)
};
struct KeyframeDescriptor {
  uint64_t      kf_id;               // anchor keyframe; submap window ends here (§3.4)
  ScanContext   sc;                  // built over the anchor's submap
  std::vector<BtcTriangle> btc;      // lazily built (get_or_build) over the anchor's submap
  meridian::Pose   pose;                // T_map_body snapshot at insert (gating only)
  meridian::Timestamp stamp;
};

// ---- Stage outputs (internal) -------------------------------------------
struct ScCandidate { uint64_t id; float sc_dist; double yaw_guess; };
struct RankedCand  { uint64_t id; meridian::Pose T_guess; int inliers; float scoreB; uint32_t kind; };
struct VerifiedLoop {
  uint64_t from, to;  meridian::Pose T;
  float fitness, overlap, rmse, cond;
  Eigen::Matrix<double,6,6> info6;   // small_gicp res.H at optimum (observability order [t,r])
  uint32_t descriptor_kind;          // 0=SC++, 1=STD, 2=BTC
};

// ---- Provenance (telemetry only; NOT on the L3 boundary, see §3.2) -------
struct LoopProvenance {
  uint64_t from, to;
  float sc_dist; int std_inliers;
  float gicp_fitness, gicp_overlap, gicp_rmse, gicp_cond;
  bool  pcm_passed; float gnc_weight_final;   // gnc_weight_final filled from L3 readback
  std::array<double,4> stage_ms;              // A,B,C,D timings
};

// ---- The concrete L5 engine (implements ILoopDetector, 01 §7.6) ----------
class HierarchicalLoopDetector final : public meridian::ILoopDetector {
public:
  HierarchicalLoopDetector(const LoopClosureConfig& cfg,
                           std::shared_ptr<const meridian::KeyframeStore> store,  // geometry
                           meridian::IBackEnd* backend);                         // odom composites for PCM
  // 01 §7.6 interface:
  void add_keyframe(std::uint64_t id,
                    std::shared_ptr<const std::vector<meridian::LidarPoint>> cloud,
                    const meridian::Pose& T_map_body) override;                  // Stage A build (§5.4)
  std::vector<meridian::LoopConstraint> detect() override;                       // A→B→C→D, emit (§4)
  meridian::LoopDiagnostics diagnostics() const override;                        // 01 Appendix A; §14
private:
  SubmapCache     submaps_;   // anchor-keyed last-N composed clouds, recompute-on-miss (§3.4)
  ScanContextDb   sc_db_;     // owns descriptors + ring-key KD-tree
  BtcDb           btc_db_;    // owns BTC triangle hash + binary codes
  GicpVerifier    gicp_;      // Stage C — wraps small_gicp (§7.1)
  PcmFilter       pcm_;       // Stage D (odom self-test + trusted clique + quarantine)
  SurfelCache     surfels_;   // per-anchor covariance side table (§7.2), L5-private
  std::vector<LoopProvenance> provenance_;
};
```

`detect()` needs each keyframe's `stamp` and `observability` (for §5.4 gating and
§9.2 inflation). It obtains them via the back-end's keyframe accessor (the
back-end holds the `KeyframePacket`); L5 receives a back-end pointer at
construction and uses only its **read-only** queries (`corrected_trajectory()` for
PCM odometry composites, and a per-id keyframe lookup for `stamp`/`observability`).
This keeps L5 from `#include`-ing any back-end *implementation* (dependency rule
R2/R3, `00_architecture.md` §4).

**Sub-component interfaces** (the *cascade* is fixed; each *algorithm* is an
interface so it is unit-testable and a future descriptor (e.g. iBTC, §6.1) can
drop in behind Stage B, per `00_architecture.md` §5 — these are seams for
testing/extension, not a menu of shipping variants):

```cpp
struct ISceneDescriptor {            // Stage A/B descriptor abstraction
  virtual void insert(uint64_t id, const std::vector<meridian::LidarPoint>& body_cloud) = 0;
  virtual std::vector<ScCandidate> retrieve(uint64_t query_id) = 0;     // SC++ KD-tree
  virtual std::vector<RankedCand>  rerank(uint64_t query_id,
                                          const std::vector<ScCandidate>&) = 0; // STD/BTC
};
struct IGeometricVerifier {          // Stage C — the one impl wraps small_gicp (§7.1)
  virtual std::optional<VerifiedLoop>
    verify(const std::vector<meridian::LidarPoint>& src,
           const std::vector<meridian::LidarPoint>& tgt,
           const meridian::Pose& T_guess) = 0;  // GicpVerifier (small_gicp)
};
struct IConsistencyFilter {          // Stage D
  virtual std::vector<VerifiedLoop> select(const std::vector<VerifiedLoop>& fresh) = 0; // PcmFilter
};
```

---

## 12. Parameters

All under `meridian.place` in the typed `Config` tree (`00_architecture.md` §8;
file `config/place.yaml`). Defaults are the KITTI-tuned Scan Context starting
points (`Appendix R.1`); they are *starting points* to be swept on the Newer
College replay sequences with the Ouster OS0-128 (`DATASET.md`), never trusted
as-is (`Appendix R.1`).

| Param | Default | Meaning |
|---|---|---|
| **Cadence / gating** | | |
| `lc.detect_period_kf` | 1 | run `detect()` every N keyframes |
| `lc.min_time_gap` | 30 s | reject candidates newer than this (anti self-loop) |
| `lc.min_kf_gap` | 30 | …or fewer than this many keyframes apart |
| `lc.cooldown_kf` | 5 | per-candidate cooldown after a match |
| **Submap accumulator (§3.4)** | | |
| `lc.submap_window` | 5 | N keyframes composed into each anchor's submap |
| `lc.submap_voxel` | 0.25 m | voxel downsample after composing a submap |
| `lc.submap_cache` | 32 | anchor-keyed submap LRU entries (recompute-on-miss) |
| **Scan Context++ (A)** | | |
| `lc.sc_Nr` / `lc.sc_Ns` | 20 / 60 | radial / sector bins (4 m/ring, 6°/sector, `Appendix R.1`) |
| `lc.sc_rmax` | 80 m | max radius ($N_r\times4$ m, `Appendix R.1`) |
| `lc.sc_knn` | 15 | ring-key KD-tree candidates |
| `lc.sc_dist_thresh` | 0.13 | SC cosine-distance accept (`Appendix R.1`: 0.13–0.20) |
| `lc.sc_max_xy` | 50 m | loose spatial sanity gate |
| `lc.sc_yaw_search_band` | 10 | sectors searched around sector-key alignment |
| `lc.sc_topK` | 5 | candidates passed to Stage B |
| **STD/BTC (B)** | | |
| `lc.std_max_keypoints` | 500 | keypoints per kf |
| `lc.std_knn_kp` | 10 | neighbours per keypoint for triangles |
| `lc.std_side_min/max` | 0.5 / 50 m | triangle side-length bounds |
| `lc.std_side_tol` | 0.2 m | side-length match tolerance (hash quantisation, `Appendix R.2`) |
| `lc.std_min_matches` | 4 | min triangle matches |
| `lc.std_ransac_eps` | 0.5 m | keypoint inlier threshold |
| `lc.std_ransac_iters` | 200 | RANSAC iterations |
| `lc.std_min_inliers` | 5 | min keypoint inliers |
| `lc.std_yaw_gate` | 30° | reject RANSAC samples far from Stage-A yaw |
| `lc.scoreB_w` | {0.2,0.5,0.3} | score weights (sc, inliers, residual) |
| `lc.std_topK` | 2 | candidates passed to Stage C |
| `lc.btc_local_radius` | 2.0 m | radius of the per-keypoint binary appearance neighbourhood |
| `lc.btc_bin_count` | 40 | occupancy bins along the keypoint normal axis (binary-code length) |
| `lc.btc_occupancy_min` | 1 | points-in-bin to set an occupancy bit |
| `lc.btc_hamming_min` | 0.6 | min Hamming similarity to admit a keypoint vote (0 ⇒ pure STD) |
| **GICP (C) — small_gicp** | | |
| `lc.gicp_downsample` | 0.25 m | voxel downsample of both clouds (`Appendix R.3`) |
| `lc.gicp_max_corr_dist` | 1.0 m | correspondence gate (`small_gicp max_correspondence_distance`) |
| `lc.gicp_num_threads` | 4 | small_gicp OpenMP/TBB threads |
| `lc.gicp_voxel_res` | 1.0 m | VGICP voxel resolution (large/submap clouds) |
| `lc.gicp_source_submap` | false | also use the query submap as source (very sparse query, §7.4) |
| `lc.gicp_fitness_min` | 0.6 | accept fitness (also `fitness_ref`, §9.2) |
| `lc.gicp_overlap_min` | 0.4 | accept overlap (inlier ratio) |
| `lc.gicp_rmse_max` | 0.3 m | accept rmse (mean error) |
| `lc.gicp_cond_max` | 1e4 | reject if alignment too degenerate |
| `lc.gicp_fit_sigma` | 0.1 m | fitness rmse scale |
| `lc.use_photo_tiebreak` | false | optional RGB tie-break (§7.6) |
| **PCM (D)** | | |
| `lc.pcm_chi2_conf` | 0.99 | χ² confidence, **bound equal to** `backend.pcm_chi2_alpha`; convention in `05_backend_graph.md` §3.2 (§8.2.1) |
| `lc.pcm_maxclique_ms` | 20 ms | max-clique time budget |
| `lc.pcm_quarantine` | 50 | rejected-loop retest ring buffer |
| **Covariance shaping (→ L3)** | | |
| `lc.cov_lambda` | 1e-3 | Hessian regularisation |
| `lc.cov_degenerate_eig` | 1.0 | eigenvalue below ⇒ degenerate axis |
| `lc.cov_degenerate_mult` | 100 | variance inflation on a degenerate axis |
| `lc.cov_psd_floor` | 1e-9 | eigenvalue clamp on composed covariances before inversion (§8.2 PSD guard) |

(GNC kernel type, switchable on/off and its prior, and re-integration thresholds
belong to the back-end / map specs, not L5.)

---

## 13. Failure modes and mitigations

| # | Failure mode | Symptom | Mitigation |
|---|---|---|---|
| 13.1 | **Perceptual aliasing** (self-similar corridors/rows) | confident GICP on the *wrong* place | BTC local structure + per-keypoint binary appearance code (B, §6.1) + PCM odom self-test (8.2.1) + PCM clique (D) + GNC down-weight (9.3) + `sc_max_xy` sanity gate. Layered by design (`Appendix R.6`). |
| 13.2 | **GICP local minimum** (bad initial guess) | high fitness on a wrong alignment, or non-convergence | Stage-B 6-DoF `T_guess` (+ Stage-A yaw, SC++ lateral) seeds small_gicp in-basin (`Appendix R.3`); `cond_max` rejects degenerate fits; try the top-2 RANSAC hypotheses if the first fails. |
| 13.3 | **Degenerate geometry** (tunnel / open field) | GICP converges but along-axis translation unconstrained | `cond` gate rejects, *or* admit with per-axis-inflated `cov` (§9.2) so the weak axis barely contributes. Never a tight factor on an unobservable axis. |
| 13.4 | **Sparse / low-overlap revisit** | few correspondences, low fitness | Submap-accumulated descriptors (§3.4) and a submap GICP target (§7.4) densify both retrieval and registration, the primary recall fix; `gicp_min`-overlap gate then rejects what remains; descriptor stage already requires a structural match, so few survive to C. Enable `gicp_source_submap` and loosen `gicp_downsample` for very sparse sensors during the dataset sweep. |
| 13.5 | **Reverse-direction revisit** | same place, opposite heading | SC ring key is yaw-invariant (still retrieved); STD triangles are rotation+translation invariant (`Appendix R.2`); GICP resolves the 180° via `T_guess`. This is the headline SC property (`Appendix R.1`). |
| 13.6 | **Loop burst after long drift** | many loops fire at once | PCM admits only the mutually-consistent subset (one coherent batch); coalescing of the resulting rebuild is L4's job (§10.4). |
| 13.7 | **Stale cloud eviction race** | store evicts a cloud mid-GICP | Shared-immutable ref-counted handle pins it for the duration (`01` §2.4 / §7.5); eviction defers. |
| 13.8 | **Descriptor DB growth** | memory on long missions | SC images + ring keys are tiny (~5 KB/kf); BTC keypoints+binary codes bounded by `std_max_keypoints` (the binary code is a few bytes/keypoint); clouds live in `KeyframeStore` (its eviction/disk-spill policy applies — L5 holds only ids/descriptors). **Submaps and per-keypoint plane/triangle clouds are recomputed on demand from the store, not cached for the mission**: the submap LRU (`lc.submap_cache`) and surfel side-table hold only a small working set and are recompute-on-miss, so memory is bounded by the working set, not the trajectory length. A speculative long-lived LRU triangle/plane-cloud cache is **not** the default; it is available only as a very-long-mission tuning option when recomputation is measured to cost detection latency. |
| 13.9 | **Wrong loop already admitted** (slipped PCM) | global map kink | the §9.3 series down-weights then removes it post-admission — committed Huber absorbs a borderline residual, off-thread batch GNC drives a confirmed outlier's weight → 0 and removes the factor (9.3/9.4); a periodic PCM re-audit can flag it (operator-visible, §14). |
| 13.10 | **First session, no loops yet** | PCM 1-node clique would be a free pass | The single-loop **odometry-consistency self-test** (§8.2.1) filters a lone loop against the chained corrected-odom covariance *before* max-clique, so the very first loop is gated even with no trusted set; §7 GICP gates remain the geometric filter, and the cross-loop max-clique power still grows with loop count. |
| 13.11 | **Front-end reseed** (`01` §6.4 / MUST-FIX #2) | a keyframe whose constraint is `AbsolutePrior` (chain break), not relative | L5 is unaffected: it keys on `id` and reads geometry from the store; PCM's odometry composites use whatever the back-end's `corrected_trajectory()` exposes across the restart, so consistency tests still close. |

---

## 14. Debug / introspection hooks

Per `00_architecture.md` §10 (telemetry is a first-class subsystem), L5 emits
through the ROS-agnostic `TelemetrySink` (`00` §10.1); the wrapper maps these to
topics/markers. The *most valuable* output is the **rejection reason** at each
stage — a developer must be able to see *why* a candidate died. The
`LoopDiagnostics` struct (`01` Appendix A: `candidates, verified, rejected_pcm,
sc_query_ms, gicp_ms`) is the aggregate `diagnostics()` returns; the richer
per-loop detail rides in the L5-private `LoopProvenance` table the wrapper reads.

**Telemetry keys (core)** → bound by the wrapper under `/meridian/place/…`:

| Key (`TelemetrySink`) | Kind | Content |
|---|---|---|
| `place/sc_image` | cloud/image | the current keyframe's scan-context heatmap. |
| `place/candidates` | marker | per Stage-A candidate, a line current→candidate, coloured by `sc_dist`, thickness by furthest stage reached (A/B/C/D). |
| `place/verified_loop` | marker | accepted loop: src cloud aligned at GICP `T` over the target, with a text label `fitness/overlap/rmse/cond`. |
| `place/rejected_loop` | marker + event | rejected candidate in red with **reason**: `SC_DIST`, `BTC_HAMMING`, `STD_INLIERS`, `GICP_FITNESS`, `GICP_OVERLAP`, `GICP_COND`, `PCM_ODOM_INCONSISTENT`, `PCM_INCONSISTENT`. |
| `place/pcm_graph` | marker | the consistency graph: nodes=loops, edges=consistent, clique highlighted. |
| `place/loop_constraint` | scalar/vec | for each emitted `LoopConstraint`: `from_id,to_id,fitness`, `cov` diagonal. |
| `place/dirty_region` | marker | the `GraphUpdate` AABB the back-end will rebuild (read back, §10.2). |
| `place/timing` | timing | per-stage ms (A/B/C/D) via `ScopedTimer` (`00` §10.1), rolling p50/p95. |
| `place/stats` | scalar | counters: candidates seen / surviving each stage; loops emitted; rejections by reason; DB size; mean GNC weight of live loops (read back from L3). |

**Structured logging** (`meridian::log`, `00` §10.3): one JSON-lines record per
`detect()`: `{kf_id, n_sc_cand, n_after_B, n_verified, n_emitted, stage_ms[4],
rejection_histogram}`.

The condition number `cond` and the per-axis `cov` inflation are surfaced so
loop-level degeneracy is *visible*, mirroring L2's observability reporting (`00`
§10.4 observability marker).

---

## 15. Test plan

Per `00_architecture.md` §9.4, against the Newer College benchmark set
(`DATASET.md`). The L5 acceptance criterion: ATE drops after loop closure; no
false-loop corruption. Runs under `colcon test` with the deterministic
single-thread mode (`00` §11.2) for reproducibility.

1. **Unit — Scan Context invariance.** Synthetic cloud rotated by a known yaw ⇒
   recovered `Δψ` within one sector; translated within tolerance ⇒ SC distance
   below threshold. Empty/degenerate cloud ⇒ no crash, empty candidates.
2. **Unit — BTC triangle matching.** Two overlapping crops of one cloud with a
   known SE(3) offset ⇒ `ransac_rigid` recovers it within `std_ransac_eps`;
   non-overlapping ⇒ rejected. Verify the 6-D triangle descriptor (3 sides + 3
   inter-normal angles) is invariant to vertex ordering and to a rigid transform of
   the crop. Verify the binary appearance code: identical geometry with *different*
   local occupancy ⇒ codes diverge and `btc_hamming_min` blocks the vote (the
   anti-aliasing property); `btc_hamming_min = 0` recovers pure-STD behaviour
   (superset check, §6.1).
3. **Unit — submap accumulator.** Composing N keyframes with known relative poses
   ⇒ a body-frame cloud anchored at the latest keyframe matching the hand-composed
   ground truth; a `GraphUpdate` moving a windowed keyframe invalidates and
   recomposes the cached entry (§3.4). Assert no submap node ever appears in an
   emitted `LoopConstraint` (the §3.4 invariant: constraints stay keyframe-to-
   keyframe).
4. **Unit — GICP (small_gicp).** Known-offset pair ⇒ recovers transform; report
   fitness/overlap/cond and the returned `res.H`. Planar/tunnel input ⇒ high
   `cond`, correct gating. Pin the small_gicp version and assert the
   `RegistrationResult` members the spec relies on (`T_target_source`, `converged`,
   `num_inliers`, `H`) — pins the `small_gicp` API surface (Appendix R.3).
5. **Unit — PCM.** Hand-built loop sets with one injected outlier ⇒ outlier
   excluded from the clique; all-consistent ⇒ all kept. **Single-loop self-test
   (§8.2.1):** a lone loop that agrees with a still-confident corrected-odom chain
   is accepted; the same loop contradicting that chain is rejected with
   `PCM_ODOM_INCONSISTENT`; and a large-correction loop closing a heavily-drifted
   chain is *not* rejected (the chained-covariance widening — 13.10).
6. **Unit — covariance shaping.** `cov` fitness-scaling monotonic; degenerate-axis
   inflation applied along the correct eigen-direction; chained corrected-odom
   covariance composed with Adjoint transport and PSD-guarded (§8.2); tangent
   permutation ($[t,r]\!\to\![\rho;\phi]$, and the GTSAM $[\text{rot};\text{trans}]$
   adapter) verified against a hand-computed case (`Appendix R.5`).
7. **Integration — replay.** Newer College tuning sequences with revisits
   (quad-easy, math-medium, park; `DATASET.md`). Metrics: precision/recall of
   detected loops vs the GT revisit set (revisit pairs derived from the
   `gt/tum_asimu` trajectories); **ATE before/after** loop closure; target
   **zero false loops** on the tuning set. quad-hard is the holdout: milestone
   evaluations only, never tuned on.
8. **Integration — de-integration.** Force a loop on a drifted trajectory; assert
   the back-end's `GraphUpdate` lists the correct moved keyframes and that
   `apply_graph_update` is invoked; assert the nvblox map RMSE-to-GT drops in the
   region after rebuild (cross-checked with `06_mapping.md`'s test). For
   math-medium the dataset's `prior_map/maths-institute.ply` serves as the map
   ground truth for a map-vs-map check.
9. **Stress — perceptual aliasing.** Synthetic sequence of repeated identical
   corridors; assert the BTC appearance code, PCM odom self-test, and GNC
   reject aliased loops; inspect `place/rejected_loop` reasons.
10. **Stress — degenerate revisit.** A tunnel/corridor revisit; assert high `cond`,
    per-axis covariance inflation, and that the loop bends yaw but not along-axis
    translation (13.3).

---

## 16. References

### Project documents (authoritative; this spec is subordinate)

- `00_architecture.md` — layers, threading (back-end/loop thread), telemetry
  subsystem, config tree, dependency rules, library canon (small_gicp for Stage C).
- `01_interfaces_and_data_types.md` — **the boundary types**: `KeyframePacket`
  (§6), `KeyframeStore` (§7.5), `IMapLayer`/`GraphUpdate` + `apply_graph_update`
  (§7.5), `ILoopDetector` + **`LoopConstraint`** (§7.6), `IBackEnd`
  (`add_loop_constraint`, `corrected_trajectory`, §7.4), `Pose`/`PoseCov6`
  (§3.1/§3.3), `ObservabilityReport` (§3.4), `LidarPoint` (§4.2), `LoopDiagnostics`
  (Appendix A), worked example (§10).
- `05_backend_graph.md` — iSAM2 graph, `BetweenFactor` ingestion, the
  committed-Huber-then-batch-GNC robustness series and switchable reserve, and the
  **canonical squared-Mahalanobis / χ² convention** (§7.1, §8) this spec binds its
  PCM gates to (§8.2.1). Consumes L5's `LoopConstraint`.
- `06_mapping.md` — L4 nvblox map, retained-cloud store, clear-and-rebuild
  re-integration (executor side of §10).
- `11_build_system_libraries.md` — library canon: small_gicp for the L5 GICP
  verify (§3), scancontext vendored.
- **Appendix R** (this spec) — non-normative SOTA reference grounding: SC/SC++
  equation+defaults digest (R.1), STD/BTC/iBTC repo pointers (R.2), GICP +
  `small_gicp` verified-API block (R.3), PCM + Kimera-RPGO repo pointers (R.4),
  robust `BetweenFactor` injection digest (R.5), defence-in-depth hierarchy (R.6).
- `05_backend_graph.md` (again) — the back-end the loop factor is handed to (robust
  noise model, GNC/Huber/switchable, marginal-covariance gate); kept in lock-step on
  tangent order and the PCM-front-gate / GNC-safety-net split.
- `DATASET.md` — the Newer College benchmark set (quad-easy / math-medium / park
  tuning, quad-hard holdout; `prior_map/maths-institute.ply` for map-vs-map
  checks): evaluation data and loop-closure acceptance (ATE drop, zero false loops).

### Reference code (geometric primitives L5 reuses; cited `file:line`)

- Surfel / plane / normal fit: `esti_plane(pabcd, points_near, 0.1)` —
  `FAST_LIO/src/laserMapping.cpp:678` (normal equation $A\mathbf n=-\mathbf 1$ over
  `NUM_MATCH_POINTS=5`, `common_lib.h:26`; planar-residual gate). Eigen-decomp
  variant: `init_plane` — `FAST-LIVO2/src/voxel_map.cpp:55–86`. Reused in §6.2,
  §7.2.
- Nearest-neighbour / radius search (the correspondence step small_gicp performs
  internally; reference analogue): `KD_TREE::Nearest_Search` —
  `FAST_LIO/include/ikd-Tree/ikd_Tree.h:328`; `::Radius_Search` — `:330`; used by
  the odometry correspondence step `h_share_model` — `laserMapping.cpp:638`
  (`Nearest_Search` call at `:670`). Reused in §7.3.
- Body→world point placement (for descriptors/markers + the L4 rebuild):
  `pointBodyToWorld` — `FAST_LIO/src/laserMapping.cpp:178` ($R(R_{LI}p+t_{LI})+t$).
  Reused in §3.1, §10.2.
- Reference incremental-map insert/delete/re-balance (the behaviour spec 06's
  Tier R keeps, not Stage C's index): paper `papers/2102.10808.txt`; impl
  `FAST_LIO/include/ikd-Tree/ikd_Tree.{h,cpp}`.

> **Grounding caveat (header, restated):** the FAST-LIO2 (`papers/2107.06829.txt`,
> l.1221) and FAST-LIVO2 (`papers/2408.14035.txt`, l.1103) texts contain **no**
> place-recognition / loop-closure method — they ground only the geometric
> sub-steps above. The descriptor and consistency algorithms are grounded in the
> external SOTA below, digested in Appendix R.

### External SOTA (algorithmic grounding for the descriptor / consistency stages)

- G. Kim, A. Kim, *"Scan Context: Egocentric Spatial Descriptor for Place
  Recognition,"* IROS 2018 — Stage A descriptor, ring/sector keys, column-shift
  distance (§5; `Appendix R.1`). Repo `irapkaist/scancontext`.
- G. Kim, S. Choi, A. Kim, *"Scan Context++: Structural Place Recognition Robust
  to Rotation and Lateral Variations,"* IEEE T-RO 2021 — PC/CC sub-descriptors,
  two-key fast search, 1-DoF semi-metric yaw+lateral (§5.2–5.3; `Appendix R.1`).
- C. Yuan et al., *"STD: Stable Triangle Descriptor for 3D Place Recognition,"*
  ICRA 2023 — Stage B keypoints/triangles, 6-D side+angle descriptor, hash voting
  (§6; `Appendix R.2`). Repo `hku-mars/STD`.
- C. Yuan et al., *"BTC: A Binary and Triangle Combined Descriptor for 3D Place
  Recognition,"* 2024 (and iBTC, image-assisted) — binary pre-filter extension
  (future extension behind Stage B; §6.1; `Appendix R.2`).
- A. Segal, D. Haehnel, S. Thrun, *"Generalized-ICP,"* RSS 2009 — Stage C
  plane-to-plane cost, surfel covariances, $\epsilon$-trick (§7.2; `Appendix R.3`).
  Library `koide3/small_gicp` (the Meridian Stage-C implementation).
- J. Mangelson, D. Dominic, R. Eustice, R. Vasudevan, *"Pairwise Consistent
  Measurement Set Maximization for Robust Multi-Robot Map Merging,"* ICRA 2018 —
  Stage D consistency test + max-clique (§8; `Appendix R.4`). Impl
  `MIT-SPARK/Kimera-RPGO` (incremental PCM + PMC max-clique).
- H. Yang, P. Antonante, V. Tzoumas, L. Carlone, *"Graduated Non-Convexity for
  Robust Spatial Perception,"* IEEE RA-L 2020 — GNC kernel/weights (§9.3; GTSAM
  `GncOptimizer`; `Appendix R.5`).
- N. Sünderhauf, P. Protzel, *"Switchable Constraints for Robust Pose Graph
  SLAM,"* IROS 2012 — switchable constraint variable (§9.4; `Appendix R.5`).
- B. Pattabiraman et al., *"Fast Algorithms for the Maximum Clique Problem on
  Massive Sparse Graphs,"* WAW 2013 — max-clique heuristic for PCM (§8.3).

---

## Appendix R — SOTA reference grounding (non-normative)

This appendix is evidence, not contract: curated digests of the reference systems
this spec's design was validated against. Nothing here binds Meridian's behavior —
the normative sections above own the design. Each block names the reference checkout
it was verified against; the clones live in `/home/user/slam-reference`.

### R.1 Scan Context / Scan Context++ — equation + defaults digest

*Source of record (local).* The Scan Context (`irapkaist/scancontext`) and Scan
Context++ (`gisbi-kim/scancontext_tro`) repositories are **not** in
`/home/user/slam-reference`; this block is the digest of record. Verified against the
papers (Kim & Kim, IROS 2018, pp. 4802–4809; Kim, Choi & Kim, IEEE T-RO 2021,
arXiv:2109.13494) — clone of 2026-06.

Descriptor: polar BEV matrix $\mathbf I\in\mathbb R^{N_r\times N_s}$, bin value =
**max z-height** of points in ring $r$, sector $s$; empty bins = 0; $\rho>\rho_{\max}$
dropped.

| Symbol | Meaning | KITTI-tuned default |
|---|---|---|
| $N_r$ | radial (ring) bins | 20 (4 m/ring) |
| $N_s$ | azimuth (sector) bins | 60 (6°/sector) |
| $\rho_{\max}$ | max radius $=N_r\times 4\,$m | 80 m |
| $K$ | ring-key KNN candidates | 10–25 |
| $\tau_{SC}$ | accept iff SC dist $<\tau_{SC}$ | ~0.13–0.20 (dataset-dependent) |

- **Ring key** $\mathbf k_{\text{ring}}[r]=\tfrac1{N_s}\sum_s\mathbb 1[\mathbf I[r,s]\neq0]$
  (row occupancy ratio in the reference impl) — rotation-invariant (yaw permutes
  columns within a row, so the row aggregate is unchanged); KD-tree retrieval key.
- **Sector key** = column reduction (column-wise mean height); column analogue used to
  pre-align the shift search.
- **Column-shift cosine distance** $d(\mathbf I_q,\mathbf I_c)=\min_n\tfrac1{N_s}\sum_s
  (1-\frac{\mathbf c_q^s\cdot\mathbf c_c^{(s+n)\bmod N_s}}{\lVert\mathbf c_q^s\rVert\lVert\mathbf c_c^{(s+n)}\rVert})$;
  one column $=2\pi/N_s=6°$. Minimiser $n^\star$ gives coarse yaw $\Delta\psi=2\pi n^\star/N_s$.
- **SC++** adds a cartesian context (CC, regular x/y BEV) alongside the polar context
  (PC); a lateral translation becomes a 1-D shift in CC just as a yaw is a 1-D shift in
  PC. Two complementary keys (ring for yaw, sector for lateral) drive a two-key fast
  search; the winning shifts give a **1-DoF semi-metric** (yaw + lateral) offset.
- **Sharp edge:** $\tau_{SC}$ is dataset-dependent; SC is precision-limited. *Never*
  accept a loop on SC distance alone — Stage A only proposes candidates.

### R.2 STD / BTC / iBTC — triangle descriptors (repo pointers)

The triangle-descriptor math the spec body owns (6-D = sorted side lengths + three
inter-normal angles, hash-voting, geometric verification) is reproduced here only as
file pointers into the clones. Verified against the live checkouts below — clone of
2026-06.

| Topic | Repo@sha — path:line |
|---|---|
| STD descriptor struct (6-D: `side_length_` + `angle_`, 3 vertices) | `STD@6e5903c` — `include/STDesc.h:61–88` |
| STD build (`build_stdesc`) | `STD@6e5903c` — `src/STDesc.cpp:929` |
| STD hash-voting candidate select (`candidate_selector`) | `STD@6e5903c` — `src/STDesc.cpp:1079` |
| STD geometric verify (`candidate_verify`, `triangle_solver`, `plane_geometric_verify`) | `STD@6e5903c` — `src/STDesc.cpp:1217 / 1302 / 1326` |
| BTC `BinaryDescriptor{occupy_array_, summary_}` + `BTC{triangle_, angle_, binary_A/B/C}` | `btc_descriptor@742af15` — `include/btc.h:73–88` |
| BTC binary appearance extract + descriptor build (`binary_extractor`, `generate_btc`) | `btc_descriptor@742af15` — `src/btc.cpp:1017 / 1411` |
| iBTC LiDAR+camera fusion (image-assisted, the §7.6 future extension) | `iBTC@9f3bf49` — `iBTC/src/ibtc.cpp`, `iBTC/include/ibtc.h` |

Note: the standalone BTC repo **is** present as `btc_descriptor@742af15` (it was
historically unconfirmed and could not be named at design time); BTC = STD-triangle + a per-keypoint binary
occupancy code (`occupy_array_` bit-string + `summary_` byte) compared by Hamming
similarity.

### R.3 GICP + `small_gicp` — verified-API block

*Source of record (local).* `koide3/small_gicp` is **not** in
`/home/user/slam-reference`; this block is the digest of record (the library mandated
for Stage C by the library canon, `11_build_system_libraries.md` §3). Verified against
its README + `src/example/01_basic_registration.cpp` — clone of 2026-06. Pin the
version and re-check member names against the headers before relying on them.

GICP cost (Segal, Haehnel, Thrun, RSS 2009), residual $\mathbf d_i=\mathbf q_i-\mathbf T\mathbf p_i$:
$\hat{\mathbf T}=\arg\min_{\mathbf T}\sum_i\mathbf d_i^\top(C^Q_i+\mathbf R\,C^P_i\,\mathbf R^\top)^{-1}\mathbf d_i$.
Surfel covariance $C_i=\mathbf U\,\mathrm{diag}(1,1,\epsilon)\,\mathbf U^\top$ (plane
eigenbasis, $\epsilon$ small) makes the cost behave like point-to-plane. Acceptance:
`inlier_ratio ≥ τ_inlier` **and** `mean_error ≤ τ_err`.

```cpp
#include <small_gicp/registration/registration_helper.hpp>
using namespace small_gicp;
RegistrationSetting setting;
setting.type            = RegistrationSetting::GICP;  // {ICP, PLANE_ICP, GICP, VGICP}
setting.num_threads     = 4;                          // OpenMP/TBB
setting.downsampling_resolution     = 0.25;           // m, voxel downsample
setting.max_correspondence_distance = 1.0;            // m  (acts as d_max)
setting.voxel_resolution            = 1.0;            // m  (VGICP only)
// target = older candidate kf, source = newer query kf; init = T_target_source guess
RegistrationResult result = align(target_points, source_points, init, setting);
//   result.T_target_source : Eigen::Isometry3d  -> = T_from_to (target = older from)
//   result.converged       : bool
//   result.num_inliers     : int            -> overlap = num_inliers / N_source
//   result.H               : Eigen::Matrix<double,6,6>  -> ~information; Σ ≈ H^{-1}
//   result.b               : Eigen::Matrix<double,6,1>
//   result.error           : double          -> final cost (mean_error gating)
```

- Deps: **Eigen** only (+ bundled **nanoflann**, **Sophus**); C++17; no PCL in the hot
  path (PCL wrappers optional). small_gicp builds the target KD-tree and per-point
  surfel covariances internally.
- `result.H` (final linearised Hessian) is the principled per-loop information; $\Sigma\approx H^{-1}$,
  fitness-scaled in §9.2.
- VGICP (`setting.type=VGICP`, set `voxel_resolution`) for large clouds; plain GICP for
  two single keyframes.
- **Sharp edge:** the `align()` helper surface, the `RegistrationSetting` /
  `RegistrationType` enum, and the `RegistrationResult` members
  (`T_target_source`/`converged`/`num_inliers`/`H`/`b`/`error`) are stable, but exact
  member spelling has shifted across versions — pin and re-verify.

### R.4 PCM — pairwise consistency + max-clique (repo pointers)

The PCM consistency residual and max-clique selection the spec body owns map to the
reference incremental gate below (Mangelson et al., ICRA 2018; PMC max-clique).
Verified against the live checkout — clone of 2026-06.

| Topic | Repo@sha — path:line |
|---|---|
| Pairwise loop consistency (Mahalanobis / chi-squared) `areLoopsConsistent` | `Kimera-RPGO@d28b4df` — `include/KimeraRPGO/outlier/Pcm.h:670` |
| Odometry-chain consistency `isOdomConsistent` | `Kimera-RPGO@d28b4df` — `include/KimeraRPGO/outlier/Pcm.h:604` |
| Incremental clique update (`findInliers` / `findInliersIncremental`) | `Kimera-RPGO@d28b4df` — `include/KimeraRPGO/outlier/Pcm.h:851 / 906` |
| PMC parallel max-clique solver (`maxClique` / `maxCliqueHeu`) | `Kimera-RPGO@d28b4df` — `include/KimeraRPGO/max_clique_finder/findClique.h:47 / 54` |

Consistency residual (SE(3)): $\mathbf r=\log(\hat T^{\text{odom}}_{ij}\hat T^{\text{loop}}_{jl}\hat T^{\text{odom}}_{lk}\hat T^{\text{loop}}_{ki})^\vee$,
gated $\mathbf r^\top\Sigma^{-1}\mathbf r\le\chi^2_{6,\alpha}$ ($\alpha=0.99\Rightarrow\approx16.81$).
**Keep this quantile in lock-step with the back-end GNC's `setInlierCostThresholds(barc2=chi2inv(0.99,6))`**
so the front gate and the safety net judge a loop on the same statistical scale.

### R.5 Accepted loop → robust BetweenFactor (digest)

The injection mechanics (fitness-scaled covariance, Huber/GNC/switchable) the spec
body owns (§9). Cross-checked against the back-end grounding consumed by
`05_backend_graph.md`. GNC and switchable are GTSAM mechanisms applied by the back-end,
not L5.

- **Fitness scaling:** $\Sigma_{\text{loop}}=H^{-1}/\max(f,f_{\min})$ — start from the
  GICP Hessian, loosen a marginal loop. `f_min` clamps near-zero fitness (such loops
  should already be rejected at Stage C).
- **Tangent-order trap:** GTSAM `Pose3` covariance is ordered $[\text{rot};\text{trans}]$,
  whereas Meridian `PoseCov6` is $[\rho;\phi]$ (translation-first, `01` §3.1). Permute
  once, at the L5 boundary.
- **Robust kernels:** Huber $k\approx1.345$ (95% Gaussian efficiency) as always-on
  baseline; **GNC** (TLS/GM, `barc2=chi2inv(0.99,6)`, `setKnownInliers(odometry)`) in
  batch over the affected sub-graph before commit; **switchable** (Sünderhauf & Protzel,
  IROS 2012) only when the switch posterior must stay queryable online. GTSAM has no
  first-class switchable factor — GNC is the default.

### R.6 Defence-in-depth hierarchy (rationale digest)

Why the cascade has four stages, and what each rung catches (the asymmetry: a missed
loop costs accuracy; a false loop can be unrecoverable).

| Rung | Catches | Cost |
|---|---|---|
| spatial/marginal-cov pre-gate | geometrically impossible candidates | very low |
| Scan Context++ ring-key KNN | most non-places (high recall, fast) | very low |
| column-shift / SC distance | wrong-yaw / dissimilar BEV | low |
| STD/BTC re-rank | self-similar places (corridors, rows); yields 6-DoF guess | low–medium |
| GICP fitness/overlap/cond | geometrically non-overlapping (false positive) | medium |
| PCM max-clique | mutually-inconsistent loops (perceptual aliasing) | low (small M) |
| Huber / GNC / switchable | residual outliers at optimisation time | back-end |

Place recognition recognises a place from scan *appearance*, independent of where
odometry thinks it is — which is why it survives drift that a pure radius-search
proximity detector cannot. PCM is the discrete front gate; back-end GNC/switchable is
the continuous in-graph safety net (complementary, not redundant).
