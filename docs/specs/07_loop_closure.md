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
> **One system, no phasing.** Meridian is one complete system: a continuous-time (CT),
> tightly-coupled LiDAR-Inertial-Visual-GNSS estimator (`00` §0). L5 is a permanent
> part of that system, not a "later phase." There is no "v1 without loops, v2 with
> loops" rollout: the cascade below is the design from the start. The only
> bring-up note is module *integration order* — L5 compiles and links against the
> stable `01` boundary types, so it can be built and unit-tested before the CT
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
> switchable constraints), assembled per the grounding dossier
> `docs/grounding/08_loop_closure_placerec.md` (cited as `dossier-08 §N`) and
> cross-checked against the back-end dossier `docs/grounding/09_backend_isam2.md`.
> The **geometric sub-steps L5 reuses** — nearest-neighbour search, plane/surfel
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
The final defences (PCM pre-admission in L5 + GNC + switchable in the back-end)
ensure that even a wrong loop that survives verification cannot, by itself, break
the graph. This is the defence-in-depth hierarchy of `dossier-08 §0, §8`.

Place recognition recognises a place from the *appearance* of the scan,
independent of where odometry thinks it is — which is why it works after the drift
that a pure radius-search proximity detector cannot survive (`dossier-08 §1`).

### 1.2 In scope

- **Intra-session loop closure** on the live trajectory.
- The four-stage cascade: **Scan Context++ retrieval → STD/BTC re-rank → GICP
  verify (small_gicp) → PCM batch check** — the hierarchical coarse→fine→batch
  pattern of `dossier-08 §0, §8`.
- Synthesising the `LoopConstraint`: GICP-fitness-scaled, observability-aware
  covariance. The back-end then applies GNC + switchable (this spec sets the
  `cov` and `fitness` fields those mechanisms consume; it does not implement the
  GTSAM kernel — that is the back-end spec, `05_backend_graph.md`, grounded in
  `dossier-08 §7` / `dossier-09`).
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
  (`dossier-08 §4.4`) is a documented future extension behind the same Stage-A/B
  interface, not built now.

---

## 2. Position in the system: data flow and threading

```
        L2 CT LIVO+GNSS front-end                         L4 nvblox map
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
   │       descriptor DB (Scan-Context KD-tree + STD triangle hash)   │
   │       geometry read-only via  KeyframeStore.cloud(id)            │
   └─────────────────────────────────────────────────────────────────┘
```

**Threading** (per `00_architecture.md` §11 and `01` §2.4 / §7.6). L5 runs on the
**back-end thread (best-effort)**, *not* the front-end thread — loop closure must
never stall the real-time odometry. Concretely:

- **`add_keyframe(id, cloud, T_map_body)`** (the `ILoopDetector` method, `01` §7.6)
  is called when the back-end admits a keyframe. It builds the keyframe's
  descriptors and inserts them into L5's privately-owned DB. It is cheap
  (sub-millisecond, §5.4) so it runs inline in the back-end's keyframe ingest.
- **`detect()`** is called on the back-end thread (throttled, §12
  `lc.detect_period_kf`). It runs the cascade and returns the batch of verified,
  PCM-filtered `LoopConstraint`s; the back-end calls `add_loop_constraint` for
  each, wraps it in GNC + switchable, and folds it into iSAM2.
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
(The architecture keeps the CT front-end and the test-only iEKF reference oracle
behind the same `IFrontEnd` interface, `00` §5.4; L5 is identical regardless of
which produced the keyframe, because it only ever sees `01` boundary types.)

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
`fitness` from §7.5. The back-end reads `fitness` to drive GNC and to scale a
switchable-constraint prior (`01` §7.6 note: *"a low-fitness GICP match yields a
loose `cov`, so a marginal loop barely tugs the graph"*; the same fitness-scaled
noise model is `dossier-08 §7.2`).

> **Single relative constraint, no double-counting (consistent with `01` §6.4 /
> MUST-FIX #3).** A loop is **one** relative pose constraint between two existing
> keyframe pose variables, with marginal covariance — exactly like an odometry
> between-factor, but non-sequential. It carries **no** velocity/bias and **no**
> IMU preintegration; velocity/bias never cross the loop edge. The back-end turns
> `T_from_to` + `cov` into a single `BetweenFactor(from_id, to_id)` under a robust
> kernel (`01` §7.6; `dossier-08 §7.1`). This keeps the information budget correct.

**[PROPOSE-TO-01] (a)** `LoopConstraint` carries no provenance. L5 keeps per-stage
scores/timings in an L5-private `LoopProvenance` table (§11) for telemetry, keyed
by `(from_id,to_id)` — nothing extra crosses the boundary. If `01` ever wants
provenance on the wire, add an opaque `uint64_t provenance_id` rather than
inlining fields into the graph-boundary value type.
**[PROPOSE-TO-01] (b)** The back-end's switchable prior is derived from `fitness`
inside L3; if instead L5 should choose it, add a `double switch_prior = 1.0` to
`LoopConstraint`. The default derives it in L3 from `fitness` (so no change needed).

### 3.3 Output (indirect) — de-integration via the back-end

L5 emits no map message. After the back-end optimises with the new loop, its
`GraphUpdate` (`01` §7.4) drives `IMapLayer::apply_graph_update(update, store)`
(`01` §7.5), which **clears and rebuilds** the affected region from `KeyframeStore`
clouds at corrected poses. §10 documents how this resolves MUST-FIX #4 at the L5
boundary and why L5 must *not* own the rebuild.

---

## 4. The hierarchical cascade (overview)

```
add_keyframe(id,cloud,T):  build Scan-Context (+ keys), insert into KD-tree; STD built lazily on demand.

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
(corridors, parking rows; `dossier-08 §2.4`, §6.1). STD/BTC add *local* geometric
structure (triangles of stable keypoints) that is far more discriminative *and*
yields a **6-DoF transform hypothesis** (not merely a yaw), which lands GICP inside
its convergence basin (`dossier-08 §4.2`). GICP converts the hypothesis into a
metric transform plus a **fitness we can trust** (`dossier-08 §5.1`). PCM is the
last line: it rejects geometrically-plausible-but-globally-inconsistent loops
(perceptual aliasing that survived A–C; `dossier-08 §6.1`). This is the
defence-in-depth hierarchy of `dossier-08 §8`. Dropping any stage measurably
degrades precision/recall on the FusionPortable / M2DGR targets (`DATASET.md`).

---

## 5. Stage A — Scan Context++ retrieval

### 5.1 The descriptor

Scan Context (Kim & Kim, IROS 2018) encodes one scan as a 2-D image
$\mathbf I\in\mathbb R^{N_r\times N_s}$ in **polar** coordinates centred at the
sensor: rows = radial bins, columns = azimuth (yaw) sectors. **Scan Context++**
(Kim, Choi & Kim, T-RO 2021) keeps the polar context (PC) and adds a cartesian
context (CC) for lateral invariance, plus a fast two-key search and a 1-DoF
semi-metric offset (`dossier-08 §3`). L5 implements the **polar (rotation-
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

Bin value = **max z-height** (the original SC max-z encoding, `dossier-08 §2.2`).
Points with $\rho>\rho_{\max}$ are dropped; empty bins are 0. The KITTI-tuned
defaults from the dossier ($N_r{=}20$, $N_s{=}60$, $\rho_{\max}{=}80\text{ m}$,
i.e. 4 m/ring and 6°/sector; `dossier-08 §2.2, §2.5`) are the §12 starting points.

### 5.2 The two keys

**Ring key** $\mathbf k_{\text{ring}}\in\mathbb R^{N_r}$ — rotation-*invariant*
(row-wise occupancy ratio, the reference-impl reduction, `dossier-08 §2.3`);
populates a KD-tree for fast candidate retrieval:
$$\mathbf k_{\text{ring}}[r]=\tfrac1{N_s}\sum_{s}\mathbb 1[\mathbf I[r,s]\neq 0].$$
**Sector key** $\mathbf k_{\text{sec}}\in\mathbb R^{N_s}$ — column-wise mean
height; the column analogue used to align yaw quickly before the full distance
(`dossier-08 §3.2`).

### 5.3 Distance + coarse yaw

Column-wise cosine distance under a circular column shift $n$ (= yaw bin offset;
one column $=2\pi/N_s = 6°$ of azimuth, `dossier-08 §2.3`):
$$
d(\mathbf I_q,\mathbf I_c)=\min_{n\in[0,N_s)}\;\tfrac1{N_s}\sum_{s}
\Big(1-\frac{\mathbf c_q^{s}\cdot\mathbf c_c^{(s+n)\bmod N_s}}
{\lVert\mathbf c_q^{s}\rVert\,\lVert\mathbf c_c^{(s+n)\bmod N_s}\rVert}\Big),
$$
$\mathbf c^s$ = column $s$. The minimiser $n^\star$ gives coarse yaw
$\Delta\psi=2\pi n^\star/N_s$. SC++'s fast path first aligns via the sector key's
circular cross-correlation, then refines $n$ in a small band
(`lc.sc_yaw_search_band`) instead of brute-forcing all $N_s$ shifts
(`dossier-08 §2.3, §3.2`).

### 5.4 Algorithm

```text
function STAGE_A_build(id, cloud, T_map_body):    # inside add_keyframe
    I     = make_scan_context(cloud)               # Nr×Ns, max-z bins (§5.1)
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

**Complexity.** Build: $O(P)$ in points + $O(\log N)$ KD-tree insert. Query: one
KNN ($O(\log N)$) + `sc_knn` cosine alignments, each $O(N_r N_s\cdot\text{band})$.
With $N_r{=}20,N_s{=}60$ this is sub-millisecond (`dossier-08 §2.3`, two-stage
hierarchical search). If Stage A returns empty (the common case) the cascade
stops — this must be the fast path.

> **Never accept on SC distance alone.** The SC distance threshold is
> dataset-dependent and SC is precision-limited; `dossier-08 §2.5` is explicit
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
— a 6-D signature invariant to rotation *and* translation (`dossier-08 §4.2`).
Triangle matching both **re-ranks** Stage A's candidates *and* yields a **6-DoF
coarse transform** via the keypoint correspondences (`dossier-08 §4.3`).

L5 implements the **STD triangle descriptor**; the data structures are laid out so
the **BTC binary occupancy code can be added later as a pre-filter** behind the
same interface (BTC = STD-triangle + a binary appearance code per keypoint,
`dossier-08 §4.4`). The interface (§11) is descriptor-agnostic; this is one
descriptor pipeline (SC++ retrieve → STD re-rank), not two competing pipelines.

### 6.2 Keypoint and triangle extraction

1. **Plane extraction:** voxelise the retained cloud; per voxel fit a plane by the
   same procedure the reference odometry uses for point-to-plane —
   `esti_plane(pabcd, points_near, 0.1)` (`FAST_LIO/src/laserMapping.cpp:678`),
   which solves the normal equation $A\,\mathbf n=-\mathbf 1$ by QR over the
   neighbourhood and gates on planar residual. (Reference fits over exactly
   `NUM_MATCH_POINTS=5` neighbours, `common_lib.h:26`; here we fit over a voxel's
   full membership — same math, more support.) An equivalent eigen-decomposition
   of the per-voxel scatter matrix is the FAST-LIVO2 `init_plane` route
   (`FAST-LIVO2/src/voxel_map.cpp:55–86`).
2. **Keypoints** = stable extrema along plane-pair intersection lines / boundary
   points, repeatably detectable across viewpoint — the "stable" criterion of
   `dossier-08 §4.2`. Keep the strongest `lc.std_max_keypoints`.
3. **Triangles:** connect each keypoint to its `lc.std_knn_kp` nearest keypoints;
   keep triangles with all sides in `[lc.std_side_min, lc.std_side_max]` (rejects
   degenerate slivers and over-large triangles).
4. **Descriptor** = sorted side triple $(\ell_1,\ell_2,\ell_3)$ + the three
   inter-normal angles $(\alpha_1,\alpha_2,\alpha_3)$ — the 6-D vector of
   `dossier-08 §4.2`. The triangle's vertices fix a local reference frame, so a
   single matched triangle pair already yields a 6-DoF pose hypothesis.

### 6.3 Matching → coarse transform

```text
function STAGE_B(k, scCandidates) -> RankedCand[]:
    Tq = STD_DB.get_or_build(k)                   # triangles for query
    ranked = []
    for c in scCandidates:
        Tc = STD_DB.get_or_build(c.id)
        matches = match_triangles(Tq, Tc, side_tol = lc.std_side_tol)  # hash by quantised triple
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
secondary angle/attribute checks, exactly the hash-voting of `dossier-08 §4.3`
(`key = (round(ℓ1/Δ), round(ℓ2/Δ), round(ℓ3/Δ))`), so it is *not* an all-pairs
descriptor comparison. `ransac_rigid` solves the 3-point absolute-orientation
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
(`dossier-08 §4.5`: STD as the high-precision verifier/initialiser feeding the
same GICP + PCM back end).

---

## 7. Stage C — GICP geometric verification (small_gicp)

### 7.1 Role

A–B propose *where* and *roughly how*; C produces a **metric, trustworthy**
relative transform plus a **fitness** the rest of the system scales covariance by.
We use **GICP** (Generalized-ICP, Segal, Haehnel & Thrun, RSS 2009) —
plane-to-plane / distribution-to-distribution — far more robust to partial overlap
and differing sampling than point-to-point ICP, with no explicit normal
computation (`dossier-08 §5.2`). The descriptor stage is not metrically accurate
enough to be a graph constraint and SC can produce false positives; GICP both
refines the pose to `BetweenFactor` accuracy and produces the fitness that gates
acceptance (`dossier-08 §5.1`).

**Library (single choice, no hedging).** Stage C uses **`small_gicp`**
(`koide3/small_gicp`) — the library mandated for this component by the Meridian
library canon (`00` §0 LIBRARY CANON; `11_build_system_libraries.md` §3, "Fine
registration (loop-closure GICP verify) → small_gicp"). It is header-light,
multi-threaded (OpenMP/TBB), takes Eigen point buffers directly (no PCL
registration stack in the hot path), supports GICP/VGICP, and — critically —
returns the linearised Hessian $\mathbf H$ at convergence, which §9.2 turns into
the loop covariance (`dossier-08 §5.4`). PCL GICP and libpointmatcher were
considered and rejected for dragging the full PCL registration stack / being
slower (`11` §10 rejected-alternatives table). L5 builds the target's spatial
index with `small_gicp`'s bundled nanoflann KD-tree; the vendored ikd-Tree is the
registration *oracle* for the front-end map (`11` §3, spec 06), not Stage C's
index.

### 7.2 GICP cost

Source point $\mathbf p_i$ (query kf), correspondence $\mathbf q_i$ (candidate kf),
each with a surface covariance $C^P_i,C^Q_i$ estimated from its k-NN. For
$\mathbf T=(\mathbf R,\mathbf t)$, residual $\mathbf d_i=\mathbf q_i-\mathbf T\mathbf p_i$,
the maximum-likelihood estimate weights each residual by the combined covariance
(`dossier-08 §5.2`):
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
`max_correspondence_distance`, `dossier-08 §5.4`). This is the same
nearest-neighbour correspondence step the reference odometry runs in
`h_share_model` (`FAST_LIO/src/laserMapping.cpp:638`, `Nearest_Search` at `:670`)
— L5 just lets `small_gicp` own the index rather than hand-rolling it. For very
large keyframe clouds the VGICP variant (`setting.type = VGICP`,
`lc.gicp_voxel_res`) is the speed path; for two single keyframes plain GICP is
fine (`dossier-08 §5.4`).

### 7.4 Algorithm

```text
function STAGE_C(k, rankedB) -> VerifiedLoop[]:
    src = downsample(store.cloud(k), lc.gicp_downsample)          # §7.5 voxel grid
    out = []
    for rc in rankedB:
        tgt   = downsample(store.cloud(rc.id), lc.gicp_downsample)
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

Note `from = rc.id` (older) and `to = k` (newer), and `T` (`small_gicp`'s
`res.T_target_source`, target = older candidate) maps `to`'s body into `from`'s
body frame, matching `LoopConstraint` semantics (`01` §7.6; `dossier-08 §7.1`
inverts the helper's convention to the `BetweenFactor` one — here `target` is
already the older `from`, so the mapping is direct). `res.H` is the registration
Hessian (`dossier-08 §5.4`), the base information for §9.2.

### 7.5 Fitness, overlap, condition number (these feed §9)

- **overlap** $=\dfrac{\#\{\text{src points with a corr.}<d_{\max}\}}{\#\text{src}}$
  (the inlier ratio of `dossier-08 §5.3`).
- **rmse** $=\sqrt{\tfrac1M\sum_i\lVert\mathbf d_i\rVert^2}$ over inliers
  (the mean-error gate of `dossier-08 §5.3`).
- **fitness** $\in[0,1]$ — single trust scalar (the field `LoopConstraint.fitness`):
  $$\text{fitness}=\text{overlap}\cdot\exp\!\Big(-\tfrac{\text{rmse}}{\sigma_{\text{fit}}}\Big),\quad
  \sigma_{\text{fit}}=\texttt{lc.gicp\_fit\_sigma}\ (0.1\text{ m}).$$
  Acceptance is `inlier_ratio ≥ τ` **and** `mean_error ≤ τ` (`dossier-08 §5.3`),
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
iBTC, `dossier-08 §4.4` — a future extension, not this tie-break.)

### 7.7 Output

`VerifiedLoop{from, to, T, fitness, overlap, rmse, cond, info6, descriptor_kind}`
(L5-internal, §11). `info6` ($=\mathbf H$ at the optimum) is the *base* covariance
source before fitness scaling. Empty ⇒ no loop this round (stop).

---

## 8. Stage D — PCM batch consistency

### 8.1 Why PCM

A GICP-verified loop can still be *wrong* under perceptual aliasing (two genuinely
similar places that each match the query but imply contradictory drift,
`dossier-08 §6.1`). **PCM** — Pairwise Consistent Measurement set maximization
(Mangelson et al., ICRA 2018) — ignores descriptors entirely and asks a purely
*geometric* question: **is this loop mutually consistent with the loops we already
trust and the odometry chain between their endpoints?** It then selects the
**largest mutually-consistent subset** (a maximum-clique problem on a consistency
graph), discarding the rest (`dossier-08 §6.3`). PCM is the *front gate*; the
back-end GNC/Huber/switchable is the *safety net* — the same division of labour as
`dossier-08 §6.3`, §7.3 and `dossier-09`.

### 8.2 Pairwise consistency test

Two loops $z_{ik}$ (between kfs $i,k$) and $z_{jl}$ (between $j,l$) are consistent
if traversing loop → odom → loop⁻¹ → odom returns to identity. On $SE(3)$
(`dossier-08 §6.2`):
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

> **Keep the χ² quantile in lock-step with GNC.** `dossier-08 §6.2` / §7.3 require
> the PCM gate and the back-end GNC's `setInlierCostThresholds(barc2 =
> chi2inv(0.99, 6))` to use the *same* quantile, so a loop the front gate admits is
> judged on a consistent statistical scale by the safety net. `lc.pcm_chi2_conf`
> and the back-end's GNC confidence (`05_backend_graph.md`) must match.

### 8.3 Max-clique selection

```text
function STAGE_D(new_loops, trusted) -> accepted:
    L = trusted ∪ new_loops
    graph G: node per loop; edge(a,b) iff pairwise_consistent(a,b)        # §8.2
    require self-consistency (each loop's own GICP cond/fitness gate, §7.4)
    C = max_clique(G, budget = lc.pcm_maxclique_ms)   # Pattabiraman/PMC heuristic, §16
    accepted = C ∩ new_loops          # emit only the new loops that joined the clique
    trusted  = C                      # update the trusted set
    return accepted
```

**Notes.** Exact max-clique is NP-hard but the consistency graph is small (number
of loops, not keyframes) and sparse; a fast heuristic (PMC, as used by
Kimera-RPGO, `dossier-08 §6.3`) capped at `lc.pcm_maxclique_ms` suffices. Run PCM
**incrementally** (Kimera-RPGO style, `dossier-08 §6.3`): keep the trusted clique,
test only new loops against it + each other, so a loop that later proves
inconsistent is excluded before it pollutes iSAM2. **Single-loop case:** a lone new
loop trivially "passes" PCM (a 1-node clique), so the §7 GICP gates are the real
filter early in a session — PCM's power grows with loop count (§13.10). A loop
rejected once is held in a `lc.pcm_quarantine` ring buffer and may be re-tested
when more evidence accrues.

### 8.4 Output

Loops in the consistent clique become `LoopConstraint`s emitted to the back-end.
Rejected loops are logged with reason `PCM_INCONSISTENT` (§14). (Because
`LoopConstraint` carries no PCM flag, L5 only emits loops it has *already* vetted;
the back-end's batch GNC additionally validates the loop subset over the affected
sub-graph before commit — `dossier-08 §7.3` — treating L5's output as candidates.)

---

## 9. Building the LoopConstraint (fitness-scaled, GNC+switchable)

This is the L5→L3 deliverable. Three robustness layers stack (defence in depth);
L5 owns the first (covariance shaping) and *feeds* the other two (`dossier-08 §7`).

### 9.1 What L5 emits vs what L3 does

L5 emits a single `LoopConstraint` (`01` §7.6): `from_id, to_id, T_from_to, cov,
fitness`. The back-end (`05_backend_graph.md`) turns it into one
`BetweenFactor(from_id, to_id)` with `T_from_to` + `cov`, **wrapped in GNC** and
optionally a **switchable constraint** whose prior it derives from `fitness`
(`dossier-08 §7.4`). L5's job is to make `cov` and `fitness` *trustworthy inputs*
to those mechanisms. No velocity/bias, no preintegration cross the loop (§3.2).

### 9.2 Fitness-scaled, observability-aware covariance (`cov`)

Build the $6\times6$ covariance from the `small_gicp` information matrix
($\mathbf H=$ `res.H`, `dossier-08 §5.4`), then inflate by (a) inverse fitness and
(b) per-axis degeneracy:
$$
\Sigma_{\text{loop}}=s(\text{fitness})\cdot\big(\mathbf H_{\text{gicp}}+\lambda\mathbf I\big)^{-1},
\qquad
s(\text{fitness})=\Big(\frac{\text{fitness}_{\text{ref}}}{\max(\text{fitness},\text{fit}_{\min})}\Big)^2 .
$$

This is the $\Sigma_{\text{loop}} = H^{-1}/\max(f,f_{\min})$ rule of
`dossier-08 §7.2` (registration-Hessian start, fitness-scaled), made
observability-aware:

- $\mathbf H_{\text{gicp}}$ at convergence (§7.5); $\lambda\mathbf I$
  (`lc.cov_lambda`) regularises degenerate directions so $\Sigma$ is finite. The
  `f_min` clamp stops a near-zero-fitness loop from blowing up the covariance
  (`dossier-08 §7.2`) — such loops should already be rejected at §7.4.
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
  `dossier-08 §7.2`; the adapter that converts must be deliberate).
- Lower fitness ⇒ larger $s$ ⇒ looser factor. With $\text{fitness}_{\text{ref}}=
  \texttt{lc.gicp\_fitness\_min}$, a barely-passing loop is ≈unit-scaled; an
  excellent loop ($\to1$) gets tighter. This is exactly the `01` §7.6 / `dossier-08
  §7.2` promise: "a low-fitness GICP match yields a loose `cov`, so a marginal loop
  barely tugs the graph."

### 9.3 GNC (applied by the back-end)

The back-end wraps the factor in a **Graduated Non-Convexity** kernel (Yang et
al., RA-L 2020, GTSAM `GncOptimizer`) so that even an outlier loop surviving PCM
cannot dominate. GNC anneals the convexity parameter $\mu$ from convex toward the
target shape (Geman-McClure / TLS), re-weighting the loop each iteration
$w_i=\big(\tfrac{\mu\bar c}{\bar c+\mu r_i^2}\big)^2$; a loop driven to $w_i\to0$ is
auto-rejected. GNC is run in **batch** over the affected sub-graph before commit,
with `setKnownInliers(odometry)` so only loop factors are subject to it
(`dossier-08 §7.3`). L5's job is only to provide a sane `cov` so the whitened
residual $r_i$ is meaningful. The converged $w_i$ is surfaced as a debug/audit
signal (§14), read back from the back-end since L5 does not run the optimiser.

> **Why both PCM and GNC.** PCM is a *discrete pre-admission* gate (consistent-set
> selection before the factor enters the graph); GNC is *continuous in-graph*
> down-weighting that catches residual outliers and the case where the *odometry*
> (not the loop) was wrong. Complementary, not redundant — the front-gate /
> safety-net split of `dossier-08 §6.3, §7.3`.

### 9.4 Switchable constraint (applied by the back-end, fed by `fitness`)

The back-end may also encode the loop as a switchable constraint (Sünderhauf &
Protzel, IROS 2012): a scalar switch $\sigma\in[0,1]$ scales the factor weight,
pulled toward a prior. If the loop disagrees with the graph the optimiser drives
$\sigma\to0$, turning it off smoothly even after admission. The switch prior is
derived in L3 from `LoopConstraint.fitness` (confident loops → prior 1.0; marginal
loops → a lower prior so they are easier to switch off). GTSAM has no first-class
switchable factor, so `dossier-08 §7.3` recommends GNC as the default and reserves
switchable for when the switch posterior must stay queryable online; this spec
only *feeds* whichever L3 uses. See [PROPOSE-TO-01] (b) in §3.2 if the prior choice
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
retained clouds at corrected poses** (GPU, `00` §9.5 — nvblox is the only map
backend and there is no CPU path). Because clouds are immutable + body-frame, this
is mechanical. L5 must therefore *not* attempt its own incremental map edit — that
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
struct StdTriangle {                 // one stable-triangle descriptor (6-D, dossier-08 §4.2)
  std::array<float,3> sides;         // sorted side lengths (l1<=l2<=l3)
  std::array<float,3> n_angles;      // angles between the three vertex-plane normals
  std::array<Eigen::Vector3f,3> kp;  // keypoints (body frame)
};
struct KeyframeDescriptor {
  uint64_t      kf_id;
  ScanContext   sc;
  std::vector<StdTriangle> std;      // lazily built (get_or_build); BTC binary code later
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
  ScanContextDb   sc_db_;     // owns descriptors + ring-key KD-tree
  StdDb           std_db_;    // owns STD triangle hash
  GicpVerifier    gicp_;      // Stage C — wraps small_gicp (§7.1)
  PcmFilter       pcm_;       // Stage D (trusted clique + quarantine)
  SurfelCache     surfels_;   // per-kf covariance side table (§7.2), L5-private
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
interface so it is unit-testable and the BTC code can drop in behind Stage B, per
`00_architecture.md` §5 — these are seams for testing/extension, not a menu of
shipping variants):

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
points (`dossier-08 §2.5`) adapted for Ouster on FusionPortable / M2DGR-like data
(`DATASET.md`); they are *starting points* to be swept on the replay sets, never
trusted as-is (`dossier-08 §2.5`).

| Param | Default | Meaning |
|---|---|---|
| **Cadence / gating** | | |
| `lc.detect_period_kf` | 1 | run `detect()` every N keyframes |
| `lc.min_time_gap` | 30 s | reject candidates newer than this (anti self-loop) |
| `lc.min_kf_gap` | 30 | …or fewer than this many keyframes apart |
| `lc.cooldown_kf` | 5 | per-candidate cooldown after a match |
| **Scan Context++ (A)** | | |
| `lc.sc_Nr` / `lc.sc_Ns` | 20 / 60 | radial / sector bins (4 m/ring, 6°/sector, `dossier-08 §2.2`) |
| `lc.sc_rmax` | 80 m | max radius ($N_r\times4$ m, `dossier-08 §2.2`) |
| `lc.sc_knn` | 15 | ring-key KD-tree candidates |
| `lc.sc_dist_thresh` | 0.13 | SC cosine-distance accept (`dossier-08 §2.5`: 0.13–0.20) |
| `lc.sc_max_xy` | 50 m | loose spatial sanity gate |
| `lc.sc_yaw_search_band` | 10 | sectors searched around sector-key alignment |
| `lc.sc_topK` | 5 | candidates passed to Stage B |
| **STD/BTC (B)** | | |
| `lc.std_max_keypoints` | 500 | keypoints per kf |
| `lc.std_knn_kp` | 10 | neighbours per keypoint for triangles |
| `lc.std_side_min/max` | 0.5 / 50 m | triangle side-length bounds |
| `lc.std_side_tol` | 0.2 m | side-length match tolerance (hash quantisation, `dossier-08 §4.3`) |
| `lc.std_min_matches` | 4 | min triangle matches |
| `lc.std_ransac_eps` | 0.5 m | keypoint inlier threshold |
| `lc.std_ransac_iters` | 200 | RANSAC iterations |
| `lc.std_min_inliers` | 5 | min keypoint inliers |
| `lc.std_yaw_gate` | 30° | reject RANSAC samples far from Stage-A yaw |
| `lc.scoreB_w` | {0.2,0.5,0.3} | score weights (sc, inliers, residual) |
| `lc.std_topK` | 2 | candidates passed to Stage C |
| **GICP (C) — small_gicp** | | |
| `lc.gicp_downsample` | 0.25 m | voxel downsample of both clouds (`dossier-08 §5.4`) |
| `lc.gicp_max_corr_dist` | 1.0 m | correspondence gate (`small_gicp max_correspondence_distance`) |
| `lc.gicp_num_threads` | 4 | small_gicp OpenMP/TBB threads |
| `lc.gicp_voxel_res` | 1.0 m | VGICP voxel resolution (large clouds only) |
| `lc.gicp_fitness_min` | 0.6 | accept fitness (also `fitness_ref`, §9.2) |
| `lc.gicp_overlap_min` | 0.4 | accept overlap (inlier ratio) |
| `lc.gicp_rmse_max` | 0.3 m | accept rmse (mean error) |
| `lc.gicp_cond_max` | 1e4 | reject if alignment too degenerate |
| `lc.gicp_fit_sigma` | 0.1 m | fitness rmse scale |
| `lc.use_photo_tiebreak` | false | optional RGB tie-break (§7.6) |
| **PCM (D)** | | |
| `lc.pcm_chi2_conf` | 0.99 | χ² confidence ⇒ `χ²₆,₀.₉₉≈16.81` (match GNC, §8.2) |
| `lc.pcm_maxclique_ms` | 20 ms | max-clique time budget |
| `lc.pcm_quarantine` | 50 | rejected-loop retest ring buffer |
| **Covariance shaping (→ L3)** | | |
| `lc.cov_lambda` | 1e-3 | Hessian regularisation |
| `lc.cov_degenerate_eig` | 1.0 | eigenvalue below ⇒ degenerate axis |
| `lc.cov_degenerate_mult` | 100 | variance inflation on a degenerate axis |

(GNC kernel type, switchable on/off and its prior, and re-integration thresholds
belong to the back-end / map specs, not L5.)

---

## 13. Failure modes and mitigations

| # | Failure mode | Symptom | Mitigation |
|---|---|---|---|
| 13.1 | **Perceptual aliasing** (self-similar corridors/rows) | confident GICP on the *wrong* place | STD/BTC local structure (B) + PCM clique (D) + GNC down-weight (9.3) + `sc_max_xy` sanity gate. Layered by design (`dossier-08 §8`). |
| 13.2 | **GICP local minimum** (bad initial guess) | high fitness on a wrong alignment, or non-convergence | Stage-B 6-DoF `T_guess` (+ Stage-A yaw, SC++ lateral) seeds small_gicp in-basin (`dossier-08 §5.1`); `cond_max` rejects degenerate fits; try the top-2 RANSAC hypotheses if the first fails. |
| 13.3 | **Degenerate geometry** (tunnel / open field) | GICP converges but along-axis translation unconstrained | `cond` gate rejects, *or* admit with per-axis-inflated `cov` (§9.2) so the weak axis barely contributes. Never a tight factor on an unobservable axis. |
| 13.4 | **Sparse / low-overlap revisit** | few correspondences, low fitness | `gicp_min`-overlap gate rejects; descriptor stage already requires a structural match, so few survive to C. Loosen `gicp_downsample` for sparse sensors during the dataset sweep. |
| 13.5 | **Reverse-direction revisit** | same place, opposite heading | SC ring key is yaw-invariant (still retrieved); STD triangles are rotation+translation invariant (`dossier-08 §4.1`); GICP resolves the 180° via `T_guess`. This is the headline SC property (`dossier-08 §2.4`). |
| 13.6 | **Loop burst after long drift** | many loops fire at once | PCM admits only the mutually-consistent subset (one coherent batch); coalescing of the resulting rebuild is L4's job (§10.4). |
| 13.7 | **Stale cloud eviction race** | store evicts a cloud mid-GICP | Shared-immutable ref-counted handle pins it for the duration (`01` §2.4 / §7.5); eviction defers. |
| 13.8 | **Descriptor DB growth** | memory on long missions | SC images + ring keys are tiny (~5 KB/kf); STD bounded by `std_max_keypoints`; clouds live in `KeyframeStore` (its eviction/disk-spill policy applies — L5 holds only ids/descriptors). |
| 13.9 | **Wrong loop already admitted** (slipped PCM) | global map kink | switchable can turn it off post-hoc (9.4); GNC weight → 0 (9.3); a periodic PCM re-audit can flag it (operator-visible, §14). |
| 13.10 | **First session, no loops yet** | PCM trivial (1-node clique) | §7 GICP gates are the real early filter; PCM strength grows with loop count. Documented, expected. |
| 13.11 | **Front-end window restart** (`01` §6.4 / MUST-FIX #2) | a keyframe whose constraint is `ImuPreintegration`, not relative | L5 is unaffected: it keys on `id` and reads geometry from the store; PCM's odometry composites use whatever the back-end's `corrected_trajectory()` exposes across the restart, so consistency tests still close. |

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
| `place/rejected_loop` | marker + event | rejected candidate in red with **reason**: `SC_DIST`, `STD_INLIERS`, `GICP_FITNESS`, `GICP_OVERLAP`, `GICP_COND`, `PCM_INCONSISTENT`. |
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

Per `00_architecture.md` §9.4 and the dataset acceptance mapping (`DATASET.md`:
"back-end + loop closure → ATE drop after loop closure; no false-loop
corruption"). Runs under `colcon test` with the deterministic single-thread mode
(`00` §11.2) for reproducibility.

1. **Unit — Scan Context invariance.** Synthetic cloud rotated by a known yaw ⇒
   recovered `Δψ` within one sector; translated within tolerance ⇒ SC distance
   below threshold. Empty/degenerate cloud ⇒ no crash, empty candidates.
2. **Unit — STD triangle matching.** Two overlapping crops of one cloud with a
   known SE(3) offset ⇒ `ransac_rigid` recovers it within `std_ransac_eps`;
   non-overlapping ⇒ rejected. Verify the 6-D descriptor (3 sides + 3 inter-normal
   angles) is invariant to vertex ordering and to a rigid transform of the crop.
3. **Unit — GICP (small_gicp).** Known-offset pair ⇒ recovers transform; report
   fitness/overlap/cond and the returned `res.H`. Planar/tunnel input ⇒ high
   `cond`, correct gating. Pin the small_gicp version and assert the
   `RegistrationResult` members the spec relies on (`T_target_source`, `converged`,
   `num_inliers`, `H`) — closes `dossier-08 §9` open question 5.
4. **Unit — PCM.** Hand-built loop sets with one injected outlier ⇒ outlier
   excluded from the clique; all-consistent ⇒ all kept; lone loop ⇒ trivially
   accepted (13.10).
5. **Unit — covariance shaping.** `cov` fitness-scaling monotonic; degenerate-axis
   inflation applied along the correct eigen-direction; tangent permutation
   ($[t,r]\!\to\![\rho;\phi]$, and the GTSAM $[\text{rot};\text{trans}]$ adapter)
   verified against a hand-computed case (`dossier-08 §7.2`).
6. **Integration — replay.** FusionPortable (primary) + M2DGR (co-primary)
   sequences with revisits (`DATASET.md`). Metrics: precision/recall of detected
   loops vs the GT revisit set; **ATE before/after** loop closure; target **zero
   false loops** on the primary set.
7. **Integration — de-integration.** Force a loop on a drifted trajectory; assert
   the back-end's `GraphUpdate` lists the correct moved keyframes and that
   `apply_graph_update` is invoked; assert the nvblox map RMSE-to-GT drops in the
   region after rebuild (cross-checked with `06_mapping.md`'s test).
8. **Stress — perceptual aliasing.** A sequence with repeated identical corridors
   (M2DGR indoor); assert PCM/GNC reject aliased loops; inspect
   `place/rejected_loop` reasons.
9. **Stress — degenerate revisit.** A tunnel/corridor revisit; assert high `cond`,
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
- `05_backend_graph.md` — iSAM2 graph, `BetweenFactor` ingestion, GNC + switchable
  wiring (consumes L5's `LoopConstraint`).
- `06_mapping.md` — L4 nvblox map, retained-cloud store, clear-and-rebuild
  re-integration (executor side of §10).
- `11_build_system_libraries.md` — library canon: small_gicp for the L5 GICP
  verify (§3), vendored ikd-Tree as the registration oracle, scancontext vendored.
- `docs/grounding/08_loop_closure_placerec.md` — **the grounding dossier** for this
  spec: SC/SC++ (§2–§3), STD/BTC (§4), GICP + small_gicp (§5), PCM (§6),
  fitness-scaled robust `BetweenFactor` (§7), the end-to-end hierarchy (§8), and
  the open-questions list (§9) this spec's tests close.
- `docs/grounding/09_backend_isam2.md` — the back-end dossier the loop factor is
  handed to (robust noise model, GNC/Huber/switchable, marginal-covariance gate);
  kept in lock-step on tangent order and the PCM-front-gate / GNC-safety-net split.
- `DATASET.md` — FusionPortable (primary), M2DGR (co-primary): evaluation data and
  loop-closure acceptance (ATE drop, zero false loops).

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
- ikd-Tree incremental insert/delete/re-balance (the vendored registration oracle,
  not Stage C's index): paper `papers/2102.10808.txt`; impl
  `include/ikd-Tree/ikd_Tree.{h,cpp}`.

> **Grounding caveat (header, restated):** the FAST-LIO2 (`papers/2107.06829.txt`,
> l.1221) and FAST-LIVO2 (`papers/2408.14035.txt`, l.1103) texts contain **no**
> place-recognition / loop-closure method — they ground only the geometric
> sub-steps above. The descriptor and consistency algorithms are grounded in the
> external SOTA below, via dossier-08.

### External SOTA (algorithmic grounding for the descriptor / consistency stages)

- G. Kim, A. Kim, *"Scan Context: Egocentric Spatial Descriptor for Place
  Recognition,"* IROS 2018 — Stage A descriptor, ring/sector keys, column-shift
  distance (§5; `dossier-08 §2`). Repo `irapkaist/scancontext`.
- G. Kim, S. Choi, A. Kim, *"Scan Context++: Structural Place Recognition Robust
  to Rotation and Lateral Variations,"* IEEE T-RO 2021 — PC/CC sub-descriptors,
  two-key fast search, 1-DoF semi-metric yaw+lateral (§5.2–5.3; `dossier-08 §3`).
- C. Yuan et al., *"STD: Stable Triangle Descriptor for 3D Place Recognition,"*
  ICRA 2023 — Stage B keypoints/triangles, 6-D side+angle descriptor, hash voting
  (§6; `dossier-08 §4`). Repo `hku-mars/STD`.
- C. Yuan et al., *"BTC: A Binary and Triangle Combined Descriptor for 3D Place
  Recognition,"* 2024 (and iBTC, image-assisted) — binary pre-filter extension
  (future extension behind Stage B; §6.1; `dossier-08 §4.4`).
- A. Segal, D. Haehnel, S. Thrun, *"Generalized-ICP,"* RSS 2009 — Stage C
  plane-to-plane cost, surfel covariances, $\epsilon$-trick (§7.2; `dossier-08
  §5`). Library `koide3/small_gicp` (the Meridian Stage-C implementation).
- J. Mangelson, D. Dominic, R. Eustice, R. Vasudevan, *"Pairwise Consistent
  Measurement Set Maximization for Robust Multi-Robot Map Merging,"* ICRA 2018 —
  Stage D consistency test + max-clique (§8; `dossier-08 §6`). Impl
  `MIT-SPARK/Kimera-RPGO` (incremental PCM + PMC max-clique).
- H. Yang, P. Antonante, V. Tzoumas, L. Carlone, *"Graduated Non-Convexity for
  Robust Spatial Perception,"* IEEE RA-L 2020 — GNC kernel/weights (§9.3; GTSAM
  `GncOptimizer`; `dossier-08 §7.3`).
- N. Sünderhauf, P. Protzel, *"Switchable Constraints for Robust Pose Graph
  SLAM,"* IROS 2012 — switchable constraint variable (§9.4; `dossier-08 §7.3`).
- B. Pattabiraman et al., *"Fast Algorithms for the Maximum Clique Problem on
  Massive Sparse Graphs,"* WAW 2013 — max-clique heuristic for PCM (§8.3).
