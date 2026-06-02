# 08 — Place Recognition & Loop Closure (Grounding Dossier)

> Meridian SLAM rebuild — grounding dossier for the **place-recognition → loop-closure** subsystem.
> Scope: LiDAR-centric global descriptors (Scan Context family, STD/BTC), fine geometric
> verification (GICP / small_gicp), batch outlier rejection (PCM), and how an accepted loop
> is injected into the factor graph as a robust `BetweenFactor` for iSAM2.
>
> Companion doc: [`09_backend_isam2.md`](./09_backend_isam2.md) — this dossier produces the
> loop factors that `09` consumes (relative pose + switchable / GNC / Huber robust noise model)
> and is the consumer of the marginal-covariance pre-filter `09 §13` exposes. The two docs are
> kept in lock-step: noise-model tangent order, robust-kernel choice, and the PCM-then-GNC
> division of labour all match `09`.

## Sourcing note (read this first)

Primary sources were fetched live during authoring and the claims below are grounded against
them:

- **Scan Context** paper PDF (`gisbi-kim.github.io/publications/gkim-2018-iros.pdf`) and the
  canonical repo `irapkaist/scancontext` (mirror `gisbi-kim/scancontext`). Confirmed:
  `N_r = 20` rings, `N_s = 60` sectors (**6° per sector, 4 m per ring** → `L_max = 80 m`),
  max-z bin encoding, ring-key rotation-invariant row encoding, and the two-stage search
  (ring-key KD-tree → pairwise column-shift). IROS 2018, pp. 4802–4809.
- **Scan Context++** paper (`gkim-2021-tro.pdf`, arXiv:2109.13494, IEEE T-RO 2021). Confirmed:
  two sub-descriptors (polar context PC, cartesian context CC), robustness to rotation **and**
  lateral translation, ring key + sector key, topological retrieval + **1-DOF semi-metric
  localization**.
- **STD** paper PDF (`arxiv.org/pdf/2209.12435`, `jiaronglin.com/uploads/std.pdf`) and repo
  `hku-mars/STD`. Confirmed: triangle descriptor is a **6-D vector = 3 side lengths + 3 angles
  between the normal vectors of the planes attached to each vertex**, completely
  rotation/translation invariant; matching by side-length **hash + voting**; descriptor
  correspondences feed geometric verification. ICRA 2023.
- **BTC** (Binary and Triangle Combined Descriptor, 2024) and **iBTC** (image-assisting,
  2024). Repo lineage: `hku-mars/iBTC` is confirmed live; the standalone BTC code ships in the
  HKU-MARS lineage (the iBTC repo supersedes/contains it). **[VERIFY: exact standalone BTC repo
  name — `hku-mars/btc_descriptor` could not be confirmed; use `hku-mars/iBTC` or the STD repo's
  BTC branch].**
- **GICP** — Segal, Haehnel, Thrun, "Generalized-ICP," RSS 2009; library **`koide3/small_gicp`**
  (README + `src/example/01_basic_registration.cpp`). Confirmed: header-only, deps = Eigen +
  bundled nanoflann + Sophus, C++17, OpenMP/TBB, registration types ICP / Plane-ICP / GICP /
  VGICP, ~2× faster than fast_gicp and ~2.4× single-thread vs PCL GICP.
- **PCM** — Mangelson, Dominic, Eustice, Vasudevan, "Pairwise Consistent Measurement Set
  Maximization for Robust Multi-robot Map Merging," ICRA 2018; reference impl in
  `MIT-SPARK/Kimera-RPGO`. Pairwise consistency = odometry-composed loop residual under a
  Mahalanobis / chi-squared test; consistent set = **maximum clique** of the consistency graph.

A few exact equation **numbers** and the standalone BTC repo name remain flagged **[VERIFY]**
inline (the search channel returned prose summaries of those PDFs rather than the typeset
equation numbers). The equations themselves are reproduced faithfully; only the cross-reference
numerals are unconfirmed. See **§9 Open questions**.

---

## 0. TL;DR / Recommendation

**Recommended Meridian pipeline (hierarchical, coarse → fine → batch):**

1. **Global descriptor retrieval (coarse).** Build a **Scan Context** descriptor per keyframe.
   Retrieve top-K candidates via the **ring-key** (rotation-invariant) KD-tree. Cheap, gives
   rotation invariance for free.
2. **Descriptor re-rank + yaw estimate.** For each candidate run **column-shift (circular)
   alignment** on the full SC matrix → a distance score *and* an initial **yaw** `n*·(2π/N_s)`.
   Keep candidates below a distance threshold.
   *(Optional upgrade: **STD/BTC** triangle descriptors for full 6-DoF / viewpoint robustness
   in unstructured or reverse-loop scenes; STD additionally hands you a 6-DoF pose guess.)*
3. **Fine geometric verification (GICP).** Align candidate ↔ query scan with **GICP** seeded by
   step-2's initial guess, using **`small_gicp`**. Accept only if the **fitness / inlier ratio**
   passes a threshold and residual is low. Output: relative pose `T_i_j` + fitness `f`.
4. **Batch consistency (PCM).** Buffer accepted loops; run **Pairwise Consistency Maximization**
   (max-clique on the consistency graph) to reject mutually-inconsistent loops (aliasing).
5. **Factor injection.** Convert each surviving loop to a `BetweenFactor<Pose3>` with a
   **fitness-scaled** covariance wrapped in **Huber**, with **GNC** (batch) / **switchable**
   constraints as the back-end safety net, then hand to **iSAM2** (see `09 §9–§11, §14`).

**Concrete library choices for Meridian:**

| Stage | Library / Impl | Why |
|---|---|---|
| Scan Context (+ ring key) | C++ from `irapkaist/scancontext`, as integrated in `gisbi-kim/SC-LIO-SAM` | battle-tested, KITTI-tuned defaults, MIT-style license |
| STD / BTC (optional) | `hku-mars/STD`; BTC via `hku-mars/iBTC` lineage **[VERIFY standalone repo]** | full 6-DoF, no-GPS, viewpoint-robust, gives pose guess |
| Nearest-neighbor | `nanoflann` KD-tree (bundled in small_gicp; also used by SC) | header-only |
| GICP fine | **`koide3/small_gicp`** | fast, multithreaded, GICP/VGICP, Eigen-native, returns H for covariance |
| PCM | port from `MIT-SPARK/Kimera-RPGO` (uses PMC max-clique) | reference incremental PCM gate |
| Backend robust | **GTSAM** `GncOptimizer` + `noiseModel::Robust(Huber)` (per `09`) | matches `09 §9–§10` |

---

## 1. Problem statement & terminology

**Place recognition (loop detection):** given the current keyframe's point cloud (or a global
descriptor of it), find a *previously visited* keyframe observing the same physical place,
**without** relying on the (drifted) pose estimate.

**Loop closure (the constraint):** once a match is verified, compute the **relative rigid
transform** between the two keyframe poses and add it as a constraint to the factor graph,
which the optimizer uses to cancel accumulated drift.

Two failure modes dominate and motivate the hierarchical design:

- **False positives (perceptual aliasing):** different places that look similar (parallel
  corridors, repetitive facades). A single false loop can catastrophically warp the map.
  → handled by GICP fitness gating + PCM + back-end robustifiers.
- **False negatives (viewpoint / direction change):** the same place revisited from a different
  heading or lateral offset (reverse revisit, adjacent lane). → handled by descriptor
  invariances (SC ring-key rotation invariance, SC++ lateral invariance, STD/BTC 6-DoF
  invariance). Scan Context was explicitly designed so loops are detectable at **reverse
  revisits and corners** (Kim & Kim 2018, abstract).

Coordinate convention: `T_a_b` maps a point expressed in frame `b` into frame `a`
(`p_a = T_a_b · p_b`). A loop between query keyframe `i` and matched keyframe `j` produces a
measurement `Z_ij ≈ T_i_j` (relative pose of `j` in frame `i`), which is exactly what
`BetweenFactor<Pose3>(X(i), X(j), Z_ij, ·)` expects (`09 §6.3, §8`). GTSAM `Pose3` tangent
order is **[rot(3); trans(3)]** — `09 §6.2` — and the loop covariance must be assembled in that
order.

---

## 2. Scan Context (Kim & Kim, IROS 2018)

> G. Kim and A. Kim, *"Scan Context: Egocentric Spatial Descriptor for Place Recognition within
> 3D Point Cloud Map,"* IROS 2018, pp. 4802–4809.
> Canonical repo: **`https://github.com/irapkaist/scancontext`** (MATLAB + C++; mirror
> `gisbi-kim/scancontext`). C++ integration most useful to Meridian: **`gisbi-kim/SC-LIO-SAM`** and
> the original LIO-SAM loop-closure module.

### 2.1 Idea

Encode a single 3D LiDAR scan as a **polar bird's-eye-view (BEV) image** that is *egocentric*
(sensor at the origin) and summarizes 3D structure by the **maximum height (max-z)** of points
in each polar bin. It is **non-histogram-based** and requires **no prior training** — it
directly records the 3D structure of the visible space (paper abstract / §III). Place
recognition becomes image matching with a rotation-handling trick.

### 2.2 Descriptor construction (the N_r × N_s matrix)

Partition the ground region around the sensor (radius ≤ `L_max`) into a polar grid:

- **N_r** *ring* bins along radius (concentric annuli). **Default `N_r = 20`, 4 m per ring.**
- **N_s** *sector* bins along azimuth (angular wedges). **Default `N_s = 60`, 6° per sector.**
- Together → **`L_max = N_r × 4 m = 80 m`** (confirmed from the paper's stated 4 m ring gap and
  6° sector resolution).

Each 3D point `p = (x, y, z)` is assigned to bin `(i, j)` by radius and azimuth:

```
ρ        = sqrt(x^2 + y^2)                     # radial distance
θ        = atan2(y, x)            ∈ [0, 2π)     # azimuth
ring i   = floor( ρ / (L_max / N_r) )          # i ∈ {0 .. N_r-1}, gap = 4 m
sector j = floor( θ / (2π / N_s) )             # j ∈ {0 .. N_s-1}, width = 6°
```

The **bin value is the maximum z (height)** of all points in that bin — the "max-z encoding"
(paper Eq. for the bin/point-encoding function `φ`, **[VERIFY eq. number]**):

```
SC(i, j) = φ( P_{ij} ) = max_{ p ∈ P_{ij} } z(p)     # 0 if bin empty
```

`P_{ij}` = points in ring `i`, sector `j`. Result: the **N_r × N_s** Scan Context matrix
`I ∈ ℝ^{N_r × N_s}`. Rationale for max-z: tall structures (buildings, poles, trees) are stable
and discriminative; the egocentric height signature is robust to dynamic ground objects and to
point density.

```
                 sectors j (azimuth, 0..2π)  →  N_s = 60 columns (6° each)
              +----+----+----+----+----+----+
   rings i  0 | z  | z  | z  | .. |    |    |
 (radius) ↓ 1 |    |    | z  |    | z  |    |   4 m per ring → 80 m out
            2 | z  |    |    | z  |    |    |
           .. |    |    |    |    |    |    |
       N_r-1  |    | z  |    |    |    | z  |   (N_r = 20)
              +----+----+----+----+----+----+
   each cell = max height (max-z) of points in that polar bin
```

### 2.3 Two-step search: ring-key retrieval + column-shift alignment

A direct all-pairs matrix comparison is O(N) over the database and rotation-dependent. Scan
Context uses a **two-stage hierarchical search** (paper §IV; confirmed: "1) ring key-based KD
tree for fast candidate proposal, 2) candidate-to-query pairwise comparison-based nearest
search"):

**Stage 1 — ring-key KD-tree retrieval (fast, rotation-invariant).**
Collapse each SC matrix to a **ring key**: a vector with one scalar per ring, computed by a
**ring encoding function** applied row-wise (paper §IV-A: "each row of a scan context is encoded
into a single real value via a ring encoding function"; the reference impl uses the **occupancy
ratio** of the row — fraction of non-empty sectors):

```
k(i) = ψ( row_i of SC )                         # ring encoding (occupancy ratio in ref impl)
ringkey = [ k(0), k(1), ..., k(N_r-1) ] ∈ ℝ^{N_r}
```

Because each entry aggregates over **all sectors of a ring**, a yaw rotation only **permutes the
columns** (sectors) within each row → the row aggregate is unchanged → the ring key is
**rotation-invariant**. Store ring keys in a **KD-tree** (`nanoflann` /
`KDTreeVectorOfVectorsAdaptor`) and retrieve the top-K nearest by L2.
**[VERIFY: ring-key reduction is occupancy ratio vs row mean across paper vs repo — repo uses
occupancy ratio].**

**Stage 2 — pairwise column-shift (circular) alignment (precise, yields yaw).**
For each of the K candidates, compute a **column-wise cosine distance** between query SC and
candidate SC, minimized over all **circular column shifts** `n` (a column shift = a yaw step,
one column = `2π/N_s = 6°` of azimuth).

Per-column cosine distance (paper Eq., **[VERIFY eq. number]**):

```
                      1      N_s-1        c_j^q · c_j^c
d(I^q, I^c)  =  1 - ----- ·  Σ      -------------------------
                     N_s     j=0      ‖c_j^q‖ · ‖c_j^c‖
```

`c_j` = column `j` (length-`N_r` vector) of the SC matrix; empty-correspondence columns are
skipped. Minimize over circular shifts `n ∈ {0 .. N_s-1}`:

```
D(I^q, I^c) = min_{ n ∈ [0, N_s) }  d( I^q ,  shift(I^c, n) )
n*          = argmin_n  d( I^q ,  shift(I^c, n) )
```

The optimal shift gives the **relative yaw**:

```
Δψ ≈ n* · (2π / N_s) = n* · 6°       # initial yaw guess for fine registration
```

Optimization: the reference impl narrows the shift search around the **sector key** (column
analog of the ring key — see SC++ §3.2) rather than brute-forcing all `N_s`.

### 2.4 Invariances (and the deliberate *non*-invariances)

- **Rotation (yaw):** invariant in retrieval via the ring key, and *recovered* (not merely
  tolerated) in alignment via `n* → Δψ`. Headline property; enables **reverse-revisit and
  corner** loop detection.
- **Translation:** SC is **egocentric** (sensor at origin) → *robust* to small translation but
  **not translation-invariant**. Large lateral offsets (adjacent lane, offset reverse) degrade
  matching — exactly what **Scan Context++** (§3) fixes.
- Intentionally **not** scale/elevation normalized beyond the max-z choice; height is kept to
  stay discriminative.

### 2.5 Practical defaults (KITTI-tuned, from the paper / reference impl)

```
N_r (rings)        = 20          # 4 m per ring
N_s (sectors)      = 60          # 6° per sector
L_max (max radius) = 80 m
num candidates K   = 10..25
SC distance thresh = ~0.13 .. 0.20  # accept loop if D < thresh  [VERIFY exact paper value]
```

> ⚠️ Thresholds are dataset-dependent. For Meridian, treat as starting points and tune on
> representative data; **never** accept a loop on SC distance alone — always gate with GICP
> fitness (§5) and PCM (§6). Bound the candidate set by the back-end's 3σ marginal-covariance
> gate (`09 §13`).

---

## 3. Scan Context++ (Kim, Choi, Kim, IEEE T-RO 2021)

> G. Kim, S. Choi, A. Kim, *"Scan Context++: Structural Place Recognition Robust to Rotation and
> Lateral Variations in Urban Environments,"* IEEE T-RO 2021 (arXiv:2109.13494).
> Repo lineage: `gisbi-kim/scancontext_tro`.

SC++ keeps the polar-BEV idea but adds **two sub-descriptors** so the system is robust to both
**rotation (heading)** and **lateral translation** when roll/pitch are not severe, and it
bridges **topological place retrieval** and **metric localization** by recovering a **1-DOF
semi-metric** offset (paper abstract / §I, confirmed).

### 3.1 Two augmentations: polar context (PC) and cartesian context (CC)

- **Polar Context (PC):** essentially the original Scan Context (polar ring/sector grid, max-z).
  Naturally suited to **rotational** invariance — a yaw becomes a **column shift** in the polar
  grid. This is the **rotation-robust** matching path.
- **Cartesian Context (CC):** a **cartesian** BEV grid (regular x/y cells instead of polar
  ring/sector), still max-z per cell. A **lateral translation** (e.g. adjacent lane) becomes a
  **row/column shift** in the cartesian grid — the same 1-D shift-search machinery as PC but now
  recovering **lateral** rather than rotational misalignment. This gives **lateral invariance**.
  **[VERIFY which cartesian axis encodes the lateral shift].**

The key conceptual move: *both* rotation and lateral translation reduce to a **1-D shift search
over a structured BEV matrix** — same algorithm, two complementary parameterizations.

### 3.2 Ring key and sector key (the search analogs)

- **Ring key** (row reduction, one scalar per ring) — invariant to a **column shift** → used for
  **rotation-invariant** retrieval (as in original SC, §2.3).
- **Sector key** (column reduction, one scalar per sector/column) — invariant to a **row shift**
  → the complementary key used to accelerate / pre-align the **lateral (CC)** shift search.

```
ring key   (row reduction)    → invariant to shift along columns  (yaw)
sector key (column reduction) → invariant to shift along rows     (lateral)
```

The appropriate key is chosen per matching path (PC for rotation, CC for lateral).
**[VERIFY: precise equation for the sector key and the augmented two-key distance].**

### 3.3 1-DOF semi-metric localization

Because the winning shift in PC is a yaw and the winning shift in CC is a lateral offset, SC++
returns a **1-DOF semi-metric** alignment (the dominant misalignment axis) on top of the
topological match — a better initial guess for the GICP fine step (§5) than plain SC's yaw
alone.

### 3.4 Why this matters for Meridian

Urban / structured driving with **reverse-direction** or **adjacent-lane** revisits is SC's main
failure mode and SC++'s target. If Meridian operates in such environments, prefer SC++ (CC path);
otherwise plain SC + GICP is simpler and adequate.

---

## 4. STD / BTC: triangle descriptors (Yuan et al.)

> C. Yuan, J. Lin, Z. Zou, X. Hong, F. Zhang, *"STD: Stable Triangle Descriptor for 3D Place
> Recognition,"* ICRA 2023 (arXiv:2209.12435). Repo: **`https://github.com/hku-mars/STD`**.
> Follow-ups: **BTC** ("Binary and Triangle Combined Descriptor," 2024) adds a *binary*
> appearance descriptor; **iBTC** ("image-assisting BTC," 2024) fuses LiDAR + camera
> (repo `https://github.com/hku-mars/iBTC`).

### 4.1 Motivation vs Scan Context

Scan Context is a *global, egocentric* BEV descriptor tied to the sensor origin → sensitive to
large translation and viewpoint. **STD** instead builds **local geometric descriptors from
triangles formed by stable keypoints**. A triangle is **uniquely determined by its side lengths
(or included angles) and is completely invariant to rigid (6-DoF) transformation** (paper
abstract / §III), so STD is strong for **viewpoint changes, reverse loops, and unstructured
scenes**.

### 4.2 STD construction

**Step 1 — Stable keypoint extraction.** Fit planes to the point cloud (region-growing /
voxel-based plane segmentation), then extract **boundary / projection keypoints** that are
**repeatably detectable across viewpoints** — the "stable" criterion: a keypoint must reappear
under viewpoint change (paper §III-A). Stability is the crux (false negatives come from unstable
keypoints).

**Step 2 — Triangle descriptor (6-D).** For triples of nearby stable keypoints `(p1, p2, p3)`,
form a triangle and describe it by the **three side lengths** plus the **angles between the
normal vectors of the planes attached to each vertex** — a **6-dimensional vector** (confirmed
from the paper: "the length of three triangle sides and the angles between the normal vectors of
the adjacent plane attached to each triangle vertex"):

```
sides   = sort( ‖p1-p2‖, ‖p2-p3‖, ‖p3-p1‖ )  =  (l1 ≤ l2 ≤ l3)
angles  = ( ∠(n_a, n_b), ∠(n_b, n_c), ∠(n_c, n_a) )   # between vertex-plane normals
STD     = ( l1, l2, l3,  α1, α2, α3 )  ∈ ℝ^6
```

Sorting side lengths makes the descriptor **invariant to vertex ordering**; lengths and
inter-normal angles are **isometry-invariant** → full rotation + translation invariance. The
triangle's three vertices also fix a **local reference frame**, so a single matched triangle
pair yields a **6-DoF relative-pose hypothesis**.

### 4.3 Hash voting (matching)

STD does **not** brute-force nearest-neighbor over all descriptors. It uses a **hash table keyed
by the (quantized) side lengths** (paper §III-C):

```
key   = ( round(l1/Δ), round(l2/Δ), round(l3/Δ) )     # Δ = quantization step
table[ key ].append( triangle )                        # build phase (database)
```

Query phase — for each query triangle, hash to its bucket, retrieve candidate triangles with the
same/near key (with secondary checks on the angles/attributes), and each match **votes** for the
frame it came from:

```
for each query triangle t_q:
    for each candidate t_c in table[ hash(t_q) ]:
        if attributes_match(t_q, t_c):                 # angle/side tolerance
            vote[ frame_of(t_c) ] += 1
loop_candidate = argmax_frame  vote[frame]
```

The top-voted frame is the loop candidate; its matched triangle pairs give **point
correspondences** → solve a **6-DoF transform** (SVD / RANSAC over voted correspondences) →
**geometric verification** by plane-overlap / inlier count (paper §III-D: "The point
correspondence obtained from the descriptor matching pair can be further used in geometric
verification, which greatly improves the accuracy of place recognition"). This built-in
correspondence + transform + verification is STD's major advantage: it yields a relative pose
*and* a verification score in one shot.

### 4.4 BTC (2024): + binary descriptor

BTC augments the **triangle (geometric)** descriptor with a **binary appearance descriptor** — a
compact bit-string summarizing the local point distribution / projected occupancy around each
keypoint — combining *geometry* (triangle side lengths, transform-invariant) with *appearance*
(binary, fast Hamming compare) for "full pose invariance and high performance across diverse
scenes." This improves discrimination in geometrically-ambiguous scenes while keeping matching
cheap (hash + Hamming). **iBTC** further fuses camera measurements for LiDAR-degenerate scenes
(long corridors). **[VERIFY exact binary construction and the standalone BTC repo name].**

### 4.5 When to prefer STD/BTC over Scan Context for Meridian

| Scenario | Prefer |
|---|---|
| Structured urban driving, mostly same-direction revisits | Scan Context / SC++ (simpler) |
| Large viewpoint / reverse loops, unstructured, no-GPS indoor | **STD / BTC** |
| Need relative pose directly from the descriptor stage | **STD / BTC** (triangle correspondences) |
| Tight compute budget, single spinning LiDAR, KITTI-like | Scan Context |
| LiDAR-degenerate long corridors with a camera available | **iBTC** |

Pragmatic Meridian design: **Scan Context for cheap recall**, **STD/BTC as an optional
high-precision verifier / alternate path**, both feeding the same GICP + PCM back end.

---

## 5. GICP — fine geometric verification (Segal et al., RSS 2009; `small_gicp`)

> A. Segal, D. Haehnel, S. Thrun, *"Generalized-ICP,"* RSS 2009.
> Library: **`koide3/small_gicp`** — header-only (Eigen + bundled nanoflann + Sophus), C++17,
> OpenMP/TBB, ICP / Plane-ICP / GICP / VGICP; ~2× faster than fast_gicp, ~2.4× single-thread vs
> PCL GICP (confirmed from README).

### 5.1 Why a fine step at all

The descriptor stage (SC or STD) gives a **candidate** + a **coarse** relative pose (yaw from
SC's `n*`, lateral from SC++ CC, or full 6-DoF from STD). It is *not* metrically accurate enough
to be a graph constraint, and SC can produce false positives. **GICP** (a) refines the relative
pose to the accuracy a `BetweenFactor` needs, and (b) produces a **fitness score** that gates
acceptance (false-positive rejection).

### 5.2 GICP formulation (plane-to-plane / distribution-to-distribution)

Standard ICP minimizes point-to-point distance. **GICP** models each point as a **Gaussian**
with a local covariance `C_i` (from its k nearest neighbors; for surfaces the covariance is
flattened toward the plane → "plane-to-plane"). For corresponding points `a_i` (source) and
`b_i` (target) with covariances `C_i^A`, `C_i^B`, the residual `d_i = b_i − T a_i` is itself
Gaussian, and GICP performs **maximum-likelihood** estimation of `T` (Segal §3,
**[VERIFY eq. number]**):

```
T* = argmin_T  Σ_i  d_i^T ( C_i^B + T C_i^A T^T )^{-1} d_i
                       with   d_i = b_i − T a_i
```

The Mahalanobis weight `(C_i^B + T C_i^A T^T)^{-1}` down-weights directions where surfaces are
uncertain (along a plane) and up-weights the surface-normal direction → robust **plane-to-plane**
behaviour and far better convergence than vanilla ICP on structured scenes, with no explicit
normal computation. Special cases: `C_i = I` → point-to-point ICP; rank-deficient-toward-normal
`C_i` → point-to-plane ICP. GICP is the general case.

### 5.3 Fitness / acceptance score

After convergence, compute a **fitness** measure to accept/reject:

```
inlier set   = { i : ‖b_i − T* a_i‖ < d_max }
inlier_ratio = |inlier set| / N
mean_error   = (1/|inliers|) Σ_{inliers} ‖b_i − T* a_i‖
fitness      = inlier_ratio   (and/or  1 / (1 + mean_error))
```

**Accept iff** `inlier_ratio ≥ τ_inlier` **and** `mean_error ≤ τ_err`. This `fitness` is reused
in §7 to **scale the factor covariance** (better fit → tighter covariance).

```
ACCEPT-LOOP(query, candidate, T_init):
    src, tgt  = downsample(query), downsample(candidate)   # voxel grid
    result    = small_gicp::align(tgt, src, tree_tgt, T_init, setting)
    if result.converged and inlier_ratio(result) >= τ_inlier and result.error <= τ_err:
        return ACCEPT, result.T_target_source, inlier_ratio(result)  # → T_candidate_query
    else:
        return REJECT
```

### 5.4 `small_gicp` for Meridian — concrete usage (grounded against README + example)

`small_gicp` is the recommended impl: faster than PCL's GICP, multithreaded (OpenMP/TBB),
supports **GICP** and **VGICP** (voxelized, faster for larger clouds), Eigen-native with optional
PCL wrappers, and **returns the linearized Hessian `H`** so you can derive a registration
covariance for the noise model. High-level helper API:

```cpp
#include <small_gicp/registration/registration_helper.hpp>
using namespace small_gicp;

// RegistrationSetting (high-level helper) — confirmed fields:
RegistrationSetting setting;
setting.type            = RegistrationSetting::GICP;   // {ICP, PLANE_ICP, GICP, VGICP}
setting.num_threads     = 4;                           // OpenMP/TBB
setting.downsampling_resolution      = 0.25;           // m, voxel downsample
setting.max_correspondence_distance  = 1.0;            // m  (acts as d_max)
setting.voxel_resolution             = 1.0;            // m  (VGICP only)

// target = matched keyframe cloud, source = current query keyframe cloud.
// init = T_target_source initial guess (yaw from SC / lateral from SC++ / 6-DoF from STD)
Eigen::Isometry3d init = T_init;

// align(target_points, source_points, init, setting) → RegistrationResult
RegistrationResult result = align(target_points, source_points, init, setting);

// RegistrationResult fields (confirmed): 
//   result.T_target_source : Eigen::Isometry3d   -> this is T_candidate_query (invert for T_i_j)
//   result.converged       : bool
//   result.iterations      : int
//   result.num_inliers     : int           -> inlier_ratio = num_inliers / N_source
//   result.H               : Eigen::Matrix<double,6,6>   -> ~ information; Σ ≈ H^{-1}
//   result.b               : Eigen::Matrix<double,6,1>
//   result.error           : double         -> final cost (use for mean_error gating)
```

Notes confirmed from the README/example:

- Deps are only **Eigen** (+ bundled **nanoflann**, **Sophus**); C++17; no PCL required (PCL
  wrappers optional). Easy to vendor into Meridian's tree.
- The **registration result exposes `H`** (final linearized Hessian); `Σ_reg ≈ H^{-1}` is a
  principled per-loop covariance you can feed (after the fitness scaling of §7) into the
  `BetweenFactor` noise model — preferable to a hand-tuned constant when available.
- For larger keyframe clouds prefer **VGICP** (`setting.type = VGICP`, set `voxel_resolution`)
  for speed; for the loop-verification of two single keyframes GICP is fine.

> ⚠️ **[VERIFY]** the precise field naming has shifted across small_gicp versions; the *stable*
> surface (the `align()` helper, the `RegistrationSetting` type + `RegistrationType` enum,
> `num_threads`, `downsampling_resolution`, `max_correspondence_distance`, and a
> `RegistrationResult` carrying `T_target_source`/`converged`/`num_inliers`/`H`/`b`/`error`) is
> confirmed; pin the version and re-check exact members for the formal spec. The low-level
> `Registration<...>` class template gives finer control if the helper is too coarse.

---

## 6. PCM — Pairwise Consistency Maximization (Mangelson et al., ICRA 2018)

> J. Mangelson, D. Dominic, R. Eustice, R. Vasudevan, *"Pairwise Consistent Measurement Set
> Maximization for Robust Multi-robot Map Merging,"* ICRA 2018.
> Reference impls: **`MIT-SPARK/Kimera-RPGO`** (incremental PCM gate, uses a **PMC** parallel
> max-clique solver), `lajoiepy/robust_distributed_mapper`.

### 6.1 Why batch consistency (after GICP already gated)

GICP fitness rejects *geometrically bad* loops, but **perceptual aliasing** can produce loops
that each look individually plausible yet are **mutually inconsistent** (e.g. two corridors that
both match the same query but imply contradictory drift). PCM rejects the **inconsistent subset**
by demanding accepted loops agree *with each other* through the odometry chain.

### 6.2 Pairwise consistency test

Two loop measurements `z_ik` (between poses `i,k`) and `z_jl` (between poses `j,l`) are
**pairwise consistent** if traversing odometry + both loops returns (near) identity. Using the
(drift-prone but locally-OK) odometry estimates `x̂`, define the consistency residual (Mangelson
§III, **[VERIFY eq. number]**):

```
C( z_ik , z_jl ) =  ‖  x̂_ij  ⊕  z_jl  ⊕  x̂_lk  ⊖  z_ik  ‖_Σ
```

(pose-composition notation: compose odometry leg `i→j`, loop `j→l`, odometry leg `l→k`, then
compare against loop `i→k`; closer to identity = more consistent). The norm is **Mahalanobis**
under the combined covariance `Σ`, so the test is **chi-squared**:

```
z_ik , z_jl  consistent   ⇔   C(z_ik, z_jl)^2  ≤  γ
                                where γ = chi2inv(p, dof)     # e.g. p = 0.99
```

This is the same chi-squared family GNC uses for `setInlierCostThresholds` in `09 §10.3`
(`barc2 = chi2inv(0.99, 6)` for a 6-DoF factor) — keep the quantile consistent between the PCM
gate and the back-end GNC.

### 6.3 Max-clique on the consistency graph

Build a graph `G`: one **node per candidate loop**, an **edge** iff the pair passes the test
(§6.2). The **largest mutually-consistent set** is the **maximum clique** of `G`:

```
PCM(candidate_loops, odometry, γ):
    A = zeros(M, M)                                   # consistency adjacency, M loops
    for each pair (a, b) of candidate loops:
        if C(z_a, z_b)^2 <= γ:                        # pairwise consistent
            A[a,b] = A[b,a] = 1
    clique = MAX_CLIQUE(A)                            # NP-hard; PMC heuristic/exact
    return loops in clique                            # consistent set; reject the rest
```

Max-clique is NP-hard, but per-update candidate counts are small and fast solvers (PMC —
Parallel Maximum Clique, as used by Kimera-RPGO) run in real time. Only clique members are
committed.

> Design choice for Meridian: run PCM as an **incremental gate** (Kimera-RPGO style) — maintain the
> consistency graph across keyframes and re-evaluate the clique as loops arrive, so a loop that
> *later* proves inconsistent is excluded before it pollutes iSAM2. This is the **front gate**;
> the back-end GNC/Huber/switchable (`09 §9–§11`) is the *safety net* for residual outliers that
> slip through. `09 §11` makes the same PCM-then-GNC division explicit.

---

## 7. From accepted loop → robust `BetweenFactor` for iSAM2

This section is the contract with [`09_backend_isam2.md`](./09_backend_isam2.md) (§6, §8.5, §9–§11, §14).

### 7.1 The measurement

A surviving loop (SC/STD retrieval → GICP fitness → PCM) provides:

- matched keyframe indices `i` (query) and `j` (candidate),
- relative pose `Z_ij = T_i_j ∈ SE(3)` (from `small_gicp` `result.T_target_source`, inverted to
  the `BetweenFactor` convention, `09 §8.3`),
- a scalar **fitness** `f ∈ [0,1]` (GICP inlier ratio), residual `e` (mean error), and
  optionally the registration Hessian `H` (`Σ_reg ≈ H^{-1}`).

### 7.2 Fitness-scaled noise model

Base loop covariance **scaled by fitness** — a high-fitness (high inlier ratio, low error)
registration is trusted more (tighter covariance); a marginal one is loosened so a borderline
loop perturbs the graph less:

```
σ_trans(f) = σ_trans0 / max(f, f_min)            # translation std (m), 3 DoF
σ_rot(f)   = σ_rot0   / max(f, f_min)            # rotation std (rad), 3 DoF
Σ_loop     = diag( σ_rot^2 ×3 ,  σ_trans^2 ×3 )  # GTSAM Pose3 order: [rot; trans]
```

`f_min` clamps so near-zero fitness doesn't blow up the covariance (such loops should already be
rejected at §5). Optionally fold the residual in (`σ ∝ e / f`) or, when available, start from the
GICP Hessian: `Σ_loop = H^{-1} / max(f, f_min)`.

> ⚠️ GTSAM `Pose3` tangent order is **[rx, ry, rz, tx, ty, tz]** (rotation first) — `09 §6.2`.
> `Σ_loop`'s diagonal must match; getting this wrong silently mis-weights rotation vs
> translation. Feed it via `noiseModel::Gaussian::Covariance(Σ_loop)` (`09 §6.4`).

### 7.3 Robust kernel: Huber + GNC + switchable

Even after PCM, treat loop factors as **potentially wrong** at the back end. Three compatible
mechanisms (use the one `09` standardizes on; combinable):

1. **Huber M-estimator (always-on, convex, safe in incremental iSAM2).** Wrap the Gaussian model
   so a single mis-registration can't dominate:

   ```
   ρ_Huber(r) = { ½ r²                  if |r| ≤ k
                { k(|r| − ½ k)          if |r| >  k
   ```

   GTSAM: `noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(k), base)` with
   `k ≈ 1.345` (95% Gaussian efficiency) — `09 §6.4`.

2. **GNC — Graduated Non-Convexity (batch global outlier rejection).** Anneal a control parameter
   `μ` from convex toward the true non-convex robust cost (Geman-McClure / truncated least
   squares) so the optimizer escapes outlier-induced local minima and zero-weights bad loops.
   GTSAM: `GncOptimizer<GncParams<LevenbergMarquardtParams>>`, `setLossType(TLS|GM)`,
   `setInlierCostThresholds(barc2)` from a chi-square quantile, `setKnownInliers(odometry)` so
   **only loop factors** are subject to GNC. **GNC is batch, not incremental** — `09 §10–§11`
   runs it over the affected sub-graph *before* committing loops to iSAM2.

3. **Switchable constraints (per-factor soft on/off).** Each loop gets a switch `s ∈ [0,1]`
   optimized jointly; the residual is scaled by `s` with a prior pulling `s→1`, so the optimizer
   can turn off an inconsistent loop (Sünderhauf & Protzel, IROS 2012). GTSAM has no first-class
   switchable factor — implement as a custom `(Pose_i, Pose_j, Switch)` factor + `PriorFactor<double>`
   on the switch, or prefer GNC (`09 §9.3, §11`). In-graph switches add a variable per loop and
   non-convexity → `09` recommends GNC unless the switch posterior must stay queryable online.

**Recommendation for Meridian (matching `09 §11, §14`):** PCM (§6) is the **front gate**;
**Huber-wrapped fitness-scaled covariance on every committed loop** is the always-on baseline;
**batch GNC** (TLS, chi-square `barc2`) validates the loop subset over the affected sub-graph
*before* commit. Use switchable variables only if the switch state must persist online.

### 7.4 Pseudocode — full injection

```
ON ACCEPTED LOOP (i, j, T_i_j, fitness f, residual e [, Hessian H]):

    # 1. fitness-scaled covariance (Pose3 order: [rot(3); trans(3)])
    if H available:   Σ = (H^{-1}) / max(f, f_min)
    else:             σr = σr0/max(f,f_min);  σt = σt0/max(f,f_min)
                      Σ  = diag(σr², σr², σr², σt², σt², σt²)
    base   = noiseModel::Gaussian::Covariance(Σ)

    # 2. robust wrap (Huber, k≈1.345)
    robust = noiseModel::Robust::Create(
                 noiseModel::mEstimator::Huber::Create(1.345), base)

    # 3. between factor (added to the candidate buffer, not directly to iSAM2)
    loopFactor = BetweenFactor<Pose3>( X(i), X(j), Pose3(T_i_j), robust )

    # 4. batch GNC validation over the affected sub-graph (09 §11); commit inliers only
    if GNC_inlier(loopFactor):
        graph.add(loopFactor)
        isam2.update(graph, /*no new values*/)   # 1–2 extra empty update() after big loops, 09 §14
```

---

## 8. End-to-end pipeline (the recommended hierarchy)

```
                         ┌────────────────────────────────────────────┐
   new keyframe  ─────►  │ 1. Build descriptor                         │
   (point cloud)         │    Scan Context (N_r×N_s, max-z)            │
                         │    [optional: STD/BTC triangles]            │
                         └───────────────┬────────────────────────────┘
                            (09 §13: 3σ marginal-cov gate bounds search)
                         ┌───────────────▼────────────────────────────┐
                         │ 2. COARSE retrieval (ring-key KD-tree)      │
                         │    rotation-invariant, top-K candidates     │
                         └───────────────┬────────────────────────────┘
                                  column-shift align → SC dist + yaw n*
                                  (SC++ CC path → lateral offset too)
                         ┌───────────────▼────────────────────────────┐
                         │ 3. RE-RANK + reject by descriptor distance  │
                         │    keep candidates with D < τ_SC            │
                         └───────────────┬────────────────────────────┘
                                  T_init (yaw n* / SC++ lateral / STD 6-DoF)
                         ┌───────────────▼────────────────────────────┐
                         │ 4. FINE verify: GICP (small_gicp)           │
                         │    accept iff inlier_ratio≥τ, err≤τ         │
                         │    → relative pose T_i_j + fitness f (+H)    │
                         └───────────────┬────────────────────────────┘
                                         │ candidate loops (buffer)
                         ┌───────────────▼────────────────────────────┐
                         │ 5. BATCH consistency: PCM (max-clique)      │
                         │    keep mutually-consistent loop set        │
                         └───────────────┬────────────────────────────┘
                                         │ surviving loops
                         ┌───────────────▼────────────────────────────┐
                         │ 6. INJECT: BetweenFactor<Pose3>             │
                         │    fitness-scaled Σ, Huber; batch-GNC vet   │
                         │    → iSAM2.update()   (see 09 §9–§11, §14)  │
                         └────────────────────────────────────────────┘
```

**Why each rung exists (defense in depth against false loops):**

| Rung | Catches | Cost |
|---|---|---|
| 3σ marginal-cov gate (`09 §13`) | geometrically impossible candidates | very low |
| ring-key KD-tree | most non-places (recall, fast) | very low |
| column-shift / SC dist | wrong-yaw / dissimilar BEV | low |
| GICP fitness | geometrically non-overlapping (false positive) | medium |
| PCM | mutually-inconsistent (perceptual aliasing) | low (small M) |
| Huber / GNC / switchable | residual outliers at optimization time | back-end (`09`) |

---

## 9. Open questions / things to re-verify before the formal spec

1. **Scan Context exact equation numbers** (IROS 2018): the max-z encoding `φ` eq. number, the
   cosine column-distance eq. number, and the precise reference distance threshold
   (~0.13–0.20). N_r=20 / N_s=60 / 6°-sector / 4 m-ring / 80 m are **confirmed**.
2. **Scan Context++** (T-RO 2021): which cartesian axis encodes the lateral shift, the precise
   sector-key equation, and the augmented two-key distance. PC/CC, ring-vs-sector key, and 1-DOF
   semi-metric localization are **confirmed**.
3. **STD/BTC**: exact keypoint-stability criterion and the angle/attribute match tolerances, the
   hash quantization step Δ default, BTC's binary descriptor construction, and the **standalone
   BTC repo name** (`hku-mars/iBTC` confirmed; standalone `btc_descriptor` unconfirmed). The 6-D
   triangle descriptor (3 sides + 3 inter-normal angles) and hash-voting + geometric
   verification are **confirmed**.
4. **GICP**: the exact ML-objective equation number in Segal RSS 2009.
5. **`small_gicp` API**: pin the version and confirm exact `RegistrationSetting` /
   `RegistrationResult` member names (esp. whether `H`/`b`/`error` vs `num_inliers` are spelled
   as above) against the headers; the helper `align()`, the `RegistrationType` enum
   (ICP/PLANE_ICP/GICP/VGICP), threading, downsample/voxel resolution, and Eigen-only deps are
   **confirmed**.
6. **PCM**: the exact consistency-metric equation number (Mangelson ICRA 2018) and confirm
   Kimera-RPGO's incremental PCM + PMC behaviour to port; the chi-squared gate and max-clique
   formulation are **confirmed**.
7. **Backend robustifiers** — already cross-checked against `09`: keep `Σ_loop` order
   **[rot; trans]**, Huber `k ≈ 1.345`, GNC TLS with `barc2 = chi2inv(0.99, 6)`,
   `setKnownInliers(odometry)`, and the **PCM-front-gate / GNC-safety-net** split all match
   `09 §6, §9–§11, §14`. (Verified: `09` exists and is consistent.)

---

## 10. Primary sources (canonical references & repos)

- **Scan Context** — G. Kim, A. Kim, "Scan Context: Egocentric Spatial Descriptor for Place
  Recognition within 3D Point Cloud Map," *IROS 2018*, pp. 4802–4809.
  Paper: `https://gisbi-kim.github.io/publications/gkim-2018-iros.pdf`.
  Repo: `https://github.com/irapkaist/scancontext` (mirror `gisbi-kim/scancontext`).
  Integration: `https://github.com/gisbi-kim/SC-LIO-SAM`.
- **Scan Context++** — G. Kim, S. Choi, A. Kim, "Scan Context++: Structural Place Recognition
  Robust to Rotation and Lateral Variations in Urban Environments," *IEEE T-RO 2021*
  (arXiv:2109.13494). Paper: `https://gisbi-kim.github.io/publications/gkim-2021-tro.pdf`.
  Repo: `https://github.com/gisbi-kim/scancontext_tro`.
- **STD** — C. Yuan, J. Lin, Z. Zou, X. Hong, F. Zhang, "STD: Stable Triangle Descriptor for 3D
  Place Recognition," *ICRA 2023* (arXiv:2209.12435). Repo: `https://github.com/hku-mars/STD`.
- **BTC / iBTC** — "BTC: A Binary and Triangle Combined Descriptor for 3D Place Recognition,"
  2024; "iBTC: Image-assisting BTC … by Fusing LiDAR and Camera Measurements," 2024 (HKU-MARS).
  Repo: `https://github.com/hku-mars/iBTC` (standalone BTC repo name **[VERIFY]**).
- **GICP** — A. Segal, D. Haehnel, S. Thrun, "Generalized-ICP," *RSS 2009*.
  Library: `https://github.com/koide3/small_gicp` (predecessor `koide3/fast_gicp`).
- **PCM** — J. Mangelson, D. Dominic, R. Eustice, R. Vasudevan, "Pairwise Consistent Measurement
  Set Maximization for Robust Multi-robot Map Merging," *ICRA 2018*.
  Impl: `https://github.com/MIT-SPARK/Kimera-RPGO`.
- **Switchable constraints** — N. Sünderhauf, P. Protzel, "Switchable Constraints for Robust Pose
  Graph SLAM," *IROS 2012*.
- **GNC** — H. Yang, P. Antonante, V. Tzoumas, L. Carlone, "Graduated Non-Convexity for Robust
  Spatial Perception," *RA-L 2020* (arXiv:1909.08605). Implemented in GTSAM `GncOptimizer`.

> Companion: [`09_backend_isam2.md`](./09_backend_isam2.md) consumes the robust `BetweenFactor`
> produced here and exposes the marginal-covariance pre-filter (`09 §13`) this pipeline uses to
> bound its candidate search. Keep noise-model ordering, robust-kernel choice, and the
> PCM/GNC split in lock-step between the two documents.
