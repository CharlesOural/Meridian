# Grounding Dossier — Direct LiDAR Residual, Data Association, and the ikd-Tree Incremental Map

Scope: FAST-LIO2 reference implementation. All file/line citations are against the local checkout at `C:/Users/charl/Sources/slam-reference/`. The point-to-plane residual + Jacobian live in `FAST_LIO/src/laserMapping.cpp::h_share_model` (lines 638–754); plane fitting in `FAST_LIO/include/common_lib.h::esti_plane` (lines 225–257); map management in `laserMapping.cpp` (`lasermap_fov_segment` 231–277, `map_incremental` 427–474, `points_cache_collect` 222–227); the ikd-Tree in `FAST_LIO/include/ikd-Tree/ikd_Tree.{h,cpp}`. Paper grounding: ikd-Tree paper `papers/2102.10808.txt` (Sec. III, Algorithm 1, Data Structure 1) and FAST-LIO2 paper `papers/2107.06829.txt`.

> Verification note. The following were read in full and every quoted line was confirmed in-session: `laserMapping.cpp` (1–1056), `common_lib.h` (1–259), `ikd_Tree.h` (1–345), `ikd_Tree.cpp` (1–~250), and `2102.10808.txt` (1–~60, incl. Algorithm 1 / Data Structure 1). For the **algorithm bodies inside `ikd_Tree.cpp`** (e.g. `BuildTree`, `Search`, `Add_by_point`, `Delete_by_range`, `Criterion_Check`, `Push_Down`, `Update`) the re-reads in this session intermittently failed to return; those are documented from (a) the verified `ikd_Tree.h` declarations + constants, (b) the verified ikd-Tree paper text, and (c) the verified call sites in `laserMapping.cpp`. Wherever a `ikd_Tree.cpp` body line number is given without a quoted snippet, treat it as "to re-confirm against the local file"; the header/paper/call-site facts around it are confirmed. Each such spot is flagged inline with **[ikd.cpp: re-confirm]**.

---

## 0. Where this sits in the LIO update

Per scan, after IMU back-propagation/undistortion, the main loop (`laserMapping.cpp:865`–`1019`) does:

1. `p_imu->Process(...)` → undistorted cloud `feats_undistort`; `state_point = kf.get_x()` (888–889).
2. `lasermap_fov_segment()` — slide the bounded local map (901, body 231–277).
3. Voxel-grid downsample the scan into `feats_down_body` (904–905), `feats_down_size` (907).
4. If the tree is empty, `ikdtree.Build(feats_down_world->points)` and `continue` (909–921).
5. `Nearest_Points.resize(feats_down_size)` (951); the per-point neighbor cache.
6. `kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time)` (960) — the IEKF that calls `h_share_model` once per inner iteration. `LASER_POINT_COV = 0.001` (`laserMapping.cpp:64`).
7. `map_incremental()` — insert the registered, downsample-aware scan into the tree (976, body 427–474).

`h_share_model` is wired into the IKFoM filter at init: `kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi)` (`laserMapping.cpp:828`). It is the **shared** residual+Jacobian callback: one pass produces both `ekfom_data.h` (residual vector) and `ekfom_data.h_x` (measurement Jacobian).

State manifold (FAST-LIO2 `use-ikfom.hpp`, referenced here): `state_ikfom` = (`pos`, `rot`∈SO(3), `offset_R_L_I`∈SO(3), `offset_T_L_I`, `vel`, `bg`, `ba`, `grav`∈S²), error-state dim 23; the LiDAR measurement only touches the first 12 error dims (pos, rot, offset_R, offset_T) — see `h_x` width 12 below. (Note `common_lib.h:21` `DIM_STATE (18)` and the `StatesGroup` struct, lines 68–154, are the *legacy non-IKFoM* state; FAST-LIO2 uses `state_ikfom` via `#define USE_IKFOM`, `common_lib.h:17`. The legacy `StatesGroup::operator+` at 103–114 documents the boxplus ordering rot/pos/vel/bg/ba/grav for the old path only.)

---

## 1. Data association — finding the 5 nearest map points

### 1.1 Transform query point to world

In `h_share_model`, for each downsampled body point `i` (`laserMapping.cpp:650`–661):

```
V3D p_body(point_body.x, point_body.y, point_body.z);
V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);   // line 657
```

i.e. world point
$$ p^W = R_{WI}\,(R_{IL}\,p^L + t_{IL}) + t_{WI} $$
with `s.rot`=$R_{WI}$, `s.offset_R_L_I`=$R_{IL}$, `s.offset_T_L_I`=$t_{IL}$, `s.pos`=$t_{WI}$. The same transform is reused everywhere (`pointBodyToWorld_ikfom` 166–175, `pointBodyToWorld` 178–187, used by `map_incremental` 436).

### 1.2 kNN query, only on convergence

```
auto &points_near = Nearest_Points[i];                      // 665  (persistent per-point cache)
if (ekfom_data.converge) {
    ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);  // 670
    point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false
                           : pointSearchSqDis[NUM_MATCH_POINTS-1] > 5 ? false : true;       // 671
}
if (!point_selected_surf[i]) continue;                       // 674
```

Key engineering choices:
- **`NUM_MATCH_POINTS = 5`** (`common_lib.h:26`). Exactly 5 NN are requested.
- **Search only when `ekfom_data.converge` is true.** The IEKF sets `converge=true` on the first inner iteration and whenever the previous increment was small; while it is true the neighbor set is *re-searched*, otherwise the cached `Nearest_Points[i]` from the previous iteration is reused. This re-association policy ("rematch on convergence") avoids re-running kNN every inner iteration. (`Nearest_Points` is `vector<PointVector>`, `laserMapping.cpp:101`; resized to scan size at 951.)
- **Two rejection gates at association time** (671):
  1. fewer than 5 neighbors returned → reject;
  2. the **farthest** of the 5 squared distances `pointSearchSqDis[4] > 5` m² (≈ 2.24 m) → reject (point is in empty space / map hole).
- `pointSearchSqDis` are **squared** Euclidean distances (the tree returns squared dist; see `calc_dist` returns squared, `common_lib.h:220`–223, and `ikd_Tree` `PointType_CMP.dist` is squared).

`Nearest_Search` signature (`ikd_Tree.h:328`): `void Nearest_Search(PointType point, int k_nearest, PointVector &Nearest_Points, vector<float> &Point_Distance, float max_dist = INFINITY)`. Internally it runs the private recursive `Search(root, k, point, MANUAL_HEAP &q, max_dist)` (`ikd_Tree.h:293`) and pops the heap into the output ordered nearest-first. The result is **sorted ascending by distance** (so `[0]` is closest, `[4]` is farthest), which both the `>5` test (671) and `map_incremental` (use of `points_near[0]`, 448) rely on.

### 1.3 The kNN search algorithm (ikd-Tree)

The bounded best-first kNN uses a fixed-capacity max-heap `MANUAL_HEAP` (`ikd_Tree.h:111`–201) keyed by squared distance via `PointType_CMP` (`ikd_Tree.h:93`–109). `PointType_CMP::operator<` (102–108): primary key `dist`, tie-break by `point.x` (deterministic ordering when distances are within `1e-10`). The heap keeps the *k smallest* by evicting the top (largest) when full.

`Search(root, k, point, q, max_dist)` **[ikd.cpp: re-confirm body line numbers]** does standard k-d tree pruned descent:
- Skip a subtree if `tree_deleted` (lazy-deleted whole subtree) — see §3.4.
- Visit current node's `point` if not `point_deleted`; push `(point, dist²)` into the heap if heap not full or `dist² < q.top().dist`.
- Choose near/far child by comparing `point[axis]` to `node->point[axis]` (`division_axis`, `ikd_Tree.h:62`).
- Prune the far child using the **axis-aligned bounding box** stored per node (`node_range_x/y/z[2]`, `ikd_Tree.h:74`) via `calc_box_dist(node, point)` (declared `ikd_Tree.h:303`): if the minimum possible squared distance from `point` to the child's bounding box exceeds the current k-th distance (`q.top().dist`), the child is skipped. This box-pruning (rather than single-axis hyperplane distance) is the ikd-Tree's main search accelerator and is described in the paper Sec. III (the per-node `range[k][2]` of Data Structure 1, paper line 36).

Concurrency: search must coexist with the parallel rebuild thread. The tree increments a `search_mutex_counter` under `search_flag_mutex` so the rebuild swap waits for in-flight searches (declared `ikd_Tree.h:262`, 268). For a subtree currently being rebuilt (`*Rebuild_Ptr == &node`), the search reads the frozen copy / falls back appropriately (paper Sec. III-E "parallel rebuild"; `size()`/`validnum()` show the same trylock pattern, `ikd_Tree.cpp:69`–163).

OpenMP: the entire per-point search+residual loop is parallelized — `#pragma omp parallel for` over `feats_down_size` (`laserMapping.cpp:646`–650, guarded by `#ifdef MP_EN`). Each thread writes only its own slot `i` in `point_selected_surf`, `normvec->points[i]`, `res_last[i]`, `Nearest_Points[i]` — no contention. The compaction into `laserCloudOri`/`corr_normvect` is a serial second pass (695–706).

---

## 2. Plane fitting, validation, residual, weight, and Jacobian

### 2.1 Plane fit — `esti_plane` (`common_lib.h:225`–257)

Called as `esti_plane(pabcd, points_near, 0.1f)` (`laserMapping.cpp:678`) with `pabcd` a `VF(4)` (= `Matrix<float,4,1>`, macro `common_lib.h:48`). The fit solves, over the 5 neighbors:

```
Matrix<T,5,3> A;  Matrix<T,5,1> b;  b.setOnes(); b *= -1.0f;   // 229–232
A(j,:) = (x_j, y_j, z_j)                                        // 234–239
Matrix<T,3,1> normvec = A.colPivHouseholderQr().solve(b);      // 241
```

Math (documented in the comment block `common_lib.h:185`–191). Plane $a x + b y + c z + d = 0$ rescaled by $1/d$ to $\tfrac{a}{d}x+\tfrac{b}{d}y+\tfrac{c}{d}z = -1$. Stacking the 5 points gives the over-determined system
$$ A\,\mathbf{u} = -\mathbf{1}, \qquad A_j = (x_j,y_j,z_j), \quad \mathbf{u} = \big(\tfrac{a}{d},\tfrac{b}{d},\tfrac{c}{d}\big)^\top, $$
solved in least squares by **column-pivoted Householder QR** (`colPivHouseholderQr`). Then normalize so the stored normal is unit and the 4th entry is the signed plane offset (243–247):
$$ n = \|\mathbf{u}\|,\quad (n_x,n_y,n_z) = \mathbf{u}/n,\quad d = 1/n. $$
So `pca_result = [n_x, n_y, n_z, d]` with $\|(n_x,n_y,n_z)\|=1$, and the plane is
$$ n_x x + n_y y + n_z z + d = 0. $$

**Validation** (249–255): the fit is accepted only if **every** one of the 5 neighbors lies within `threshold = 0.1` of the plane:
$$ |\,n_x x_j + n_y y_j + n_z z_j + d\,| \le 0.1\ \text{m},\quad j=0..4. $$
If any neighbor fails, `esti_plane` returns `false` and the point is dropped (`point_selected_surf[i]` stays `false`, set at 677). This is a planarity/inlier test on the *map* points — it rejects edges, corners, and noisy neighborhoods so only locally-planar patches contribute. (A sibling routine `esti_normvector`, `common_lib.h:192`–218, does the same with a runtime `point_num` and tests against the un-normalized `+1.0f`; it is **not** used by `h_share_model`.)

Refinement vs. paper: FAST-LIO2 paper (2107.06829) describes per-point-to-plane residual where the plane normal $u_j$ and a point $q_j$ on the plane are estimated from the **nearest neighbors in the map**. The code's concrete realization is this 5-point QR-fit + 0.1 m planarity gate; the paper does not pin the exact solver or the 0.1/5/`>5` constants — those are engineering choices in the implementation.

### 2.2 Point-to-plane residual and the per-correspondence weight gate

After a valid plane (`laserMapping.cpp:680`–691):

```
float pd2 = pabcd(0)*pw.x + pabcd(1)*pw.y + pabcd(2)*pw.z + pabcd(3);   // 680  signed distance
float s   = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());                  // 681  acceptance score
if (s > 0.9) {                                                          // 683
    point_selected_surf[i] = true;
    normvec->points[i] = (pabcd(0),pabcd(1),pabcd(2));                  // 686–688  store plane normal
    normvec->points[i].intensity = pd2;                                 // 689  store signed residual
    res_last[i] = abs(pd2);                                             // 690  store |residual|
}
```

- **Residual** `pd2` = signed point-to-plane distance of the *world-frame* query point:
  $$ r_i \;=\; n_i^\top p^W_i + d_i. $$
- **Acceptance score** (681): $ s = 1 - 0.9\,\dfrac{|r_i|}{\sqrt{\|p^L_i\|}} $, accept iff $s>0.9$, i.e.
  $$ |r_i| < \tfrac{1}{9}\,\sqrt{\|p^L_i\|}. $$
  `p_body.norm()` is the body-frame range $\|p^L_i\|$ (meters). The tolerance on $|r_i|$ **grows with $\sqrt{\text{range}}$** — far points are allowed a looser fit (they are sparser, so a larger plane-distance is still plausibly an inlier), while near points must fit tightly. Note `sqrt(p_body.norm())` is `sqrt(range)` (not `range`), a deliberately gentle range dependence.
- The plane normal is cached in `normvec->points[i]` (x,y,z) and the **signed** residual in its `.intensity` (689); `res_last[i]` holds `|r_i|` for the mean-residual statistic (690, used at 703/715).

### 2.3 Compaction into the effective set

A serial second pass packs the surviving correspondences (`laserMapping.cpp:695`–706):

```
effct_feat_num = 0;
for i in [0, feats_down_size):
  if point_selected_surf[i]:
    laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];   // body point
    corr_normvect->points[effct_feat_num] = normvec->points[i];           // normal + signed residual
    total_residual += res_last[i];
    effct_feat_num++;
if (effct_feat_num < 1) { ekfom_data.valid = false; return; }             // 708–713
res_mean_last = total_residual / effct_feat_num;                          // 715
```

`laserCloudOri`/`corr_normvect`/`normvec` are pre-allocated to 100000 (`laserMapping.cpp:112`–114); the boolean mask and `res_last` arrays are sized 100000 (94, 76), and the `>5`/distance arrays are per-thread `vector<float>(NUM_MATCH_POINTS)` (663). `ekfom_data.valid=false` (710) tells the IEKF to abort this update if no correspondences survived.

### 2.4 Measurement Jacobian $H = \partial r / \partial \delta x$ (on manifold)

`ekfom_data.h_x` is `effct_feat_num × 12` (`laserMapping.cpp:720`, `//23` comment marks that the full error-state is 23 but only the first 12 dims are non-zero). Per correspondence (723–752):

```
V3D point_this_be = laserCloudOri[i];                  // p^L  (body/lidar frame point)
M3D point_be_crossmat = SKEW(point_this_be);           // [p^L]_x         (728)
V3D point_this = s.offset_R_L_I * p^L + s.offset_T_L_I; // q = R_IL p^L + t_IL  (729)
M3D point_crossmat = SKEW(point_this);                  // [q]_x           (731)
V3D norm_vec = corr_normvect[i].(x,y,z);               // n  (world-frame plane normal)
V3D C = s.rot.conjugate() * norm_vec;                  // C = R_WI^T n    (738)
V3D A = point_crossmat * C;                            // A = [q]_x (R_WI^T n)   (739)
if (extrinsic_est_en) {
    V3D B = point_be_crossmat * s.offset_R_L_I.conjugate() * C;   // B = [p^L]_x R_IL^T C   (742)
    h_x.block<1,12>(i,0) << n.x, n.y, n.z, A, B, C;              // 743  (3+3+3+3 = 12)
} else {
    h_x.block<1,12>(i,0) << n.x, n.y, n.z, A, 0,0,0, 0,0,0;      // 747
}
h(i) = -norm_p.intensity;                              // 751  measurement = -r_i
```

**Column layout of the 12-wide Jacobian row** (order matches `state_ikfom` error-state dims pos, rot, offset_R_L_I, offset_T_L_I):

| cols | block | value | meaning |
|------|-------|-------|---------|
| 0–2  | `n`   | $n^\top$ | $\partial r/\partial \delta t_{WI}$ |
| 3–5  | `A`   | $\big([q]_\times R_{WI}^\top n\big)^\top$ | $\partial r/\partial \delta\theta_{WI}$ |
| 6–8  | `B`   | $\big([p^L]_\times R_{IL}^\top R_{WI}^\top n\big)^\top$ | $\partial r/\partial \delta\phi_{IL}$ (extrinsic rotation) |
| 9–11 | `C`   | $\big(R_{WI}^\top n\big)^\top$ | $\partial r/\partial \delta t_{IL}$ (extrinsic translation) |

Derivation (right-perturbation convention, $R\leftarrow R\,\mathrm{Exp}(\delta)$; $r = n^\top p^W + d$, $p^W = R_{WI}(R_{IL}p^L + t_{IL}) + t_{WI}$, $q := R_{IL}p^L + t_{IL}$):

- $\dfrac{\partial r}{\partial \delta t_{WI}} = n^\top$  → cols 0–2 = `n`. ✔
- $\delta(R_{WI} q) = -R_{WI}[q]_\times\,\delta\theta$, so $\dfrac{\partial r}{\partial\delta\theta_{WI}} = -n^\top R_{WI}[q]_\times = -C^\top[q]_\times$. As a row vector this equals $\big([q]_\times C\big)^\top$ since $[q]_\times^\top=-[q]_\times$ → cols 3–5 = `A = [q]_x C`. ✔
- Extrinsic rotation: $\delta q = -R_{IL}[p^L]_\times\,\delta\phi$, so $\dfrac{\partial r}{\partial\delta\phi_{IL}} = -n^\top R_{WI}R_{IL}[p^L]_\times = -C^\top R_{IL}[p^L]_\times$, transposing → $[p^L]_\times R_{IL}^\top C$ = `B`. ✔
- Extrinsic translation: $\partial q/\partial t_{IL} = I$, so $\dfrac{\partial r}{\partial \delta t_{IL}} = n^\top R_{WI} = C^\top$ → cols 9–11 = `C`. ✔

So the extrinsic-on row is `[n, A, B, C]` (3+3+3+3=12); extrinsic-off zeroes cols 6–11 (only base pose pos/rot are corrected). `extrinsic_est_en` defaults `true` (`laserMapping.cpp:73`, param `mapping/extrinsic_est_en` 789).

**Measurement vector** (751): `h(i) = -norm_p.intensity = -r_i`. The IKFoM update forms the increment from $H$, the residual $-r$, and measurement noise $R$; storing $-r$ matches the linearization $0 \approx r_i + H_i\,\delta x$ ⇒ $H_i\,\delta x = -r_i$.

> Note: `SKEW_SYM_MATRX` (used 728, 731) is the skew/cross-product-matrix macro from `so3_math.h`; `s.rot.conjugate()` is the inverse rotation (Eigen quaternion conjugate = $R^\top$). `VEC_FROM_ARRAY(v)` expands to `v[0],v[1],v[2]` (`common_lib.h:29`), which is how `A,B,C` each fill 3 columns of the comma-initializer.

### 2.5 Measurement noise / weight $R$

The per-point measurement variance is **a single global scalar** passed into the filter: `kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time)` with `LASER_POINT_COV = 0.001` (`laserMapping.cpp:64`, 960). Inside IKFoM this becomes $R = \sigma^2 I$ with $\sigma^2 = 10^{-3}$ applied uniformly to every point-to-plane residual. There is **no** per-correspondence covariance weighting in this residual model — instead, robustness comes from the *gating* (the planarity check §2.1 and the range-scaled score §2.2) that decides which residuals enter the stack at all, not from down-weighting. (Contrast: Point-LIO/voxel-map variants compute per-point $R$ from range/bearing + plane uncertainty; FAST-LIO2's `laserMapping` does not.)

This matches the FAST-LIO2 paper's tightly-coupled IEKF measurement model where the LiDAR point noise is modeled as a small isotropic term; the implementation collapses it to the constant `LASER_POINT_COV`.

---

## 3. ikd-Tree — incremental map data structure

### 3.1 Node structure (`ikd_Tree.h:59`–82)

```
struct KD_TREE_NODE {
  PointType point;                              // the stored map point (61)
  int division_axis;                            // split axis (62)
  int TreeSize = 1;                             // # nodes in subtree incl. deleted (63)
  int invalid_point_num = 0;                    // # lazy-deleted points in subtree (64)
  int down_del_num = 0;                         // # points deleted by downsampling (65)
  bool point_deleted = false;                   // this node lazy-deleted (66)
  bool tree_deleted = false;                    // whole subtree lazy-deleted (67)
  bool point_downsample_deleted = false;        // this node removed by downsample (68)
  bool tree_downsample_deleted = false;         // (69)
  bool need_push_down_to_left, need_push_down_to_right; // lazy-label propagation flags (70–71)
  bool working_flag;                            // node touched by rebuild thread (72)
  pthread_mutex_t push_down_mutex_lock;         // per-node lock for push-down (73)
  float node_range_x[2], node_range_y[2], node_range_z[2];  // axis-aligned bbox of subtree (74)
  float radius_sq;                              // bounding-sphere radius² (75)
  KD_TREE_NODE *left_son_ptr, *right_son_ptr, *father_ptr;  // (76–78)
  float alpha_del, alpha_bal;                   // recorded balance/delete ratios (80–81)
};
```

Mapping to paper Data Structure 1 (paper lines 33–36): `point`/`leftson,rightson`/`axis` are the standard fields (DS1 line 2–4); `treesize, invalidnum`, `deleted, treedeleted, pushdown`, and `range[k][2]` are the ikd-Tree additions (DS1 line 5–7). The implementation **splits the paper's single `deleted` into `point_deleted` (a point) and `tree_deleted` (a subtree)**, and the paper's `pushdown` into two directional flags `need_push_down_to_left/right`. It also adds `down_del_num`/`*_downsample_deleted` for box downsampling, `radius_sq` for sphere pruning, `father_ptr` for in-place subtree replacement, and `alpha_del/alpha_bal` for the re-balance criteria (§3.5).

### 3.2 Module constants (`ikd_Tree.h:13`–18) and tunables (`ikd_Tree.h:277`–279, 309)

```
#define Minimal_Unbalanced_Tree_Size 10        // don't rebuild subtrees smaller than this (14)
#define Multi_Thread_Rebuild_Point_Num 1500    // subtrees larger than this rebuild on the 2nd thread (15)
#define DOWNSAMPLE_SWITCH true                  // (16)
#define ForceRebuildPercentage 0.2             // (17)
#define Q_LEN 1000000                           // op-logger ring buffer length (18)
delete_criterion_param  = 0.5f;   // alpha_del threshold (277)
balance_criterion_param = 0.7f;   // alpha_bal threshold (278)   [ctor default 0.6, line 309]
downsample_size         = 0.2f;   // default box length (279)
```

In FAST-LIO2 the downsample size is overridden to the map filter size: `ikdtree.set_downsample_param(filter_size_map_min)` before `Build` (`laserMapping.cpp:913`; setter `ikd_Tree.h:319`–322). `filter_size_map_min` defaults 0.5 m (`laserMapping.cpp:773`).

### 3.3 Build (`Build` `ikd_Tree.h:327`; `BuildTree` recursion `ikd_Tree.h:287` — body **[ikd.cpp: re-confirm]**)

Public `Build(PointVector)` resets the tree and recursively constructs via `BuildTree(&root, l, r, Storage)`. Per paper Algorithm 1 (paper lines 42–60):
- `mid = (l+r)/2`; choose the **division axis with maximal covariance/spread** among the points in `[l,r]` (paper line 56 "Axis with Maximal Covariance"; impl compares ranges and uses `point_cmp_x/y/z`, `ikd_Tree.h:304`–306);
- `nth_element`/sort by that axis so the **median** point becomes node `T->point` (paper line 57–58);
- left subtree ← points below median, right subtree ← above (paper line 60+, recursive);
- then `LazyLabelInit` + `Pullup` (paper line 38, "Line 11-12") — in code this is `Update(node)` (§3.6) which recomputes `TreeSize`, `invalid_point_num`, the bbox `node_range_*`, and `alpha_*`.

The maximal-spread axis (instead of round-robin depth%k) keeps the tree shallow/balanced for the anisotropic LiDAR point distribution.

### 3.4 Lazy delete and box-wise delete

**Point lazy delete** (`Delete_by_point`, `ikd_Tree.h:290`; `Delete_Points`, 333 — body **[ikd.cpp: re-confirm]**): a deleted point is **not** removed; its node's `point_deleted=true` and ancestors' `invalid_point_num` increment (paper Sec. III-C, paper line 40 "lazy delete ... only labeled as deleted"). Searches skip `point_deleted` nodes and `tree_deleted` subtrees. Re-insertion of a previously-deleted point just flips the flag back (paper "re-insertion").

**Box-wise delete** (`Delete_by_range`, `ikd_Tree.h:289`; `Delete_Point_Boxes`, 334 — body **[ikd.cpp: re-confirm]**): given a `BoxPointType` (axis-aligned min/max, `ikd_Tree.h:25`–29), recursion uses each node's bbox `node_range_*`:
- if the node's subtree bbox is **fully inside** the delete box → set `tree_deleted=true` and lazy-label the whole subtree (O(1), the cheap case);
- if **disjoint** → return immediately (prune);
- if **partially overlapping** → recurse into children and test the node's own point.
This box delete is what the sliding-map uses (§3.7). It returns the count of deleted points (`kdtree_delete_counter`, `laserMapping.cpp:275`).

Lazy labels are pushed lazily: `Push_Down` (`ikd_Tree.h:297`) propagates a parent's `tree_deleted`/downsample flag to children only when that child is next visited, using `need_push_down_to_left/right` and the per-node `push_down_mutex_lock` (so it is safe against the rebuild thread).

### 3.5 Re-balancing criteria (scapegoat-style)

The tree monitors two ratios per node (recorded in `alpha_bal`, `alpha_del`, `ikd_Tree.h:80`–81; computed in `Update`, §3.6) and decides to partially rebuild in `Criterion_Check` (`ikd_Tree.h:296` — body **[ikd.cpp: re-confirm]**):

$$ \alpha_{bal}(T) = \frac{\max(\#\text{left}, \#\text{right})}{\text{TreeSize}-1}, \qquad \alpha_{del}(T) = \frac{\text{invalid\_point\_num}}{\text{TreeSize}}. $$

A subtree at $T$ is rebuilt when it is **unbalanced** ($\alpha_{bal} > $ `balance_criterion_param` = 0.7, i.e. one side holds >70% of the nodes) **or** **too garbage-laden** ($\alpha_{del} > $ `delete_criterion_param` = 0.5, i.e. >50% of nodes are lazy-deleted). Subtrees with `TreeSize < Minimal_Unbalanced_Tree_Size` (10) are never rebuilt (too small to matter). This is the scapegoat-tree concept the paper attributes to Galperin et al. and adopts (paper line 27, Sec. III-D). On rebuild, the offending subtree is **flattened** to a point array (skipping deleted points, which physically reclaims them) and re-`BuildTree`'d balanced.

### 3.6 Pull-up / Update (`Update`, `ikd_Tree.h:298` — body **[ikd.cpp: re-confirm]**)

After any structural change, `Update(node)` recomputes from its (already-updated) children:
- `TreeSize = left.TreeSize + right.TreeSize + 1`;
- `invalid_point_num = left.invalid + right.invalid + point_deleted`;
- `down_del_num` similarly;
- the subtree bbox `node_range_{x,y,z}` = union of children boxes and own point;
- `alpha_bal`, `alpha_del` per §3.5.
This is the paper's "Pullup" (paper line 38). The bbox so maintained is exactly what `Search` and `Delete_by_range` use for pruning.

### 3.7 Incremental insert with on-the-fly downsampling

**`Add_Points(PointToAdd, downsample_on)`** (`ikd_Tree.h:331`; called twice in `map_incremental`, `laserMapping.cpp:470`–471). Returns the number actually inserted.

The map-management policy lives in `map_incremental` (`laserMapping.cpp:427`–474). For each registered scan point it decides **whether** and **how** to add, using the voxel grid of size `filter_size_map_min` (`L`):

```
mid_point = voxel-center of the cell containing the world point:
  mid.x = floor(pw.x / L)*L + 0.5*L     // 444 (and y,z 445–446)
dist = ||pw - mid||²                     // 447  (squared, calc_dist)
// (a) point's own cell is far from the nearest existing map point's cell → definitely add, no downsample
if |near[0].x - mid.x| > 0.5L AND |near[0].y - mid.y| > 0.5L AND |near[0].z - mid.z| > 0.5L:
     PointNoNeedDownsample.push_back(pw); continue;     // 448–451
// (b) otherwise add only if this point is closer to its voxel center than every existing neighbor
need_add = true;
for readd_i in [0,5):
   if near.size() < 5: break;
   if calc_dist(near[readd_i], mid) < dist: need_add = false; break;   // 455–459
if need_add: PointToAdd.push_back(pw);                 // 461
// (init phase / no neighbors) just add
else: PointToAdd.push_back(pw);                        // 463–465
...
ikdtree.Add_Points(PointToAdd, true);                  // 470  downsample ON
ikdtree.Add_Points(PointNoNeedDownsample, false);      // 471  downsample OFF
```

Semantics:
- **One point per voxel cell, the one nearest the cell center.** Branch (b) keeps the map at the target resolution: a new point replaces/avoids existing ones only if it is the best (closest-to-center) representative of its voxel. This is the "simultaneous downsampling" of the ikd-Tree paper realized at the application layer + the tree's own `downsample`.
- Branch (a) is a fast path: if even the nearest existing map point is in a *different* voxel along all three axes, the new point's voxel is empty, so add it directly with `downsample_on=false` (cheaper — no in-tree downsample search).
- `Add_Points(..., true)` triggers the tree's box-downsample (`downsample`, `ikd_Tree.h:300`; `Add_by_point`/`Add_by_range`, 291–292 — bodies **[ikd.cpp: re-confirm]**): when inserting into a voxel that already holds points, the tree keeps only the one closest to the voxel center and lazy-deletes the rest (`DOWNSAMPLE_DELETE` op, `ikd_Tree.h:37`; `down_del_num` accounting).
- New points are inserted by ordinary k-d descent appending a leaf, then `Update` pulls up sizes/boxes; if `Criterion_Check` trips on an ancestor, that subtree is scheduled for (possibly threaded) rebuild.

### 3.8 Parallel rebuild thread (`ikd_Tree.cpp:192`–250+ — verified through 250)

A background `pthread` (`rebuild_thread`, `ikd_Tree.h:261`) runs `multi_thread_rebuild()` (`ikd_Tree.cpp:230`). When a subtree to rebuild exceeds `Multi_Thread_Rebuild_Point_Num` (1500), the main thread hands it to this worker via `Rebuild_Ptr` (`ikd_Tree.h:267`) under `rebuild_ptr_mutex_lock`; concurrent insert/delete/downsample ops on that frozen subtree are appended to the **operation logger** `Rebuild_Logger` (a `MANUAL_Q` ring of `Operation_Logger_Type{point, boxpoint, flags, op}`, `ikd_Tree.h:84`–90, 203–255) and **replayed** onto the freshly built balanced subtree before it is swapped in. Smaller subtrees rebuild inline on the calling thread. The mutexes `working_flag_mutex`/`search_flag_mutex` (262) coordinate with concurrent `Nearest_Search`. The accessors `size()/validnum()/tree_range()/root_alpha()` (`ikd_Tree.cpp:69`–190) all use `pthread_mutex_trylock(&working_flag_mutex)` and fall back to cached `Treesize_tmp/Validnum_tmp/alpha_*_tmp` (275–276) when the root is mid-rebuild — confirming the lock-free read-during-rebuild design (paper Sec. III-E). This is the FAST-LIO2 paper's claim that map updates run "in parallel" without stalling the odometry.

### 3.9 Removed-point cache (`acquire_removed_points`, `points_cache_collect`)

`acquire_removed_points(removed_points)` (`ikd_Tree.h:336`) drains `Points_deleted`/`Multithread_Points_deleted` (`ikd_Tree.h:282`, 284) — points physically reclaimed during rebuild/box-delete. In FAST-LIO2 this is called by `points_cache_collect()` (`laserMapping.cpp:222`–227) right before each box-delete (273), but the body that would re-add them to `_featsArray` is **commented out** (226) — so in this configuration removed points are simply discarded (the map is purely a bounded sliding window, no long-term history buffer). `flatten(root, Storage, storage_type)` (`ikd_Tree.h:335`) is the traversal used both for rebuild and (optionally) to dump all live points for visualization (`laserMapping.cpp:944`–947, behind `if(0)`).

---

## 4. The sliding local map — `lasermap_fov_segment` (`laserMapping.cpp:231`–277)

This keeps the ikd-Tree bounded to a cube of side `cube_len` (param `cube_side_length`, default 200 m, `laserMapping.cpp:774`) centered roughly on the LiDAR, moving it as the sensor drives out, using box-deletes.

Constants: `DET_RANGE = 300.0f` (sensor max range, `laserMapping.cpp:77`, param `mapping/det_range` 775), `MOV_THRESHOLD = 1.5f` (`laserMapping.cpp:78`).

Algorithm:
1. **First call** — initialize the cube centered at the current LiDAR world position `pos_LiD = pos_lid` (= `state.pos + state.rot*state.offset_T_L_I`, set at 890/963): `vertex_min/max[i] = pos ± cube_len/2` (240–241), set `Localmap_Initialized`, return (238–245).
2. **Each later call** — compute distance from the LiDAR to each of the 6 cube faces (249–250):
   `dist_to_map_edge[i][0/1] = |pos_LiD(i) - vertex_min/max[i]|`.
   If **any** face is within `MOV_THRESHOLD * DET_RANGE = 1.5 × 300 = 450 m` of the sensor, `need_move=true` (251). If no face is that close, return (253) — no map update needed.
3. **Move distance** (256):
   $$ \text{mov\_dist} = \max\!\Big( (\text{cube\_len} - 2\,\tau\,D)\cdot 0.5\cdot 0.9,\; D\,(\tau-1) \Big),\quad \tau=\text{MOV\_THRESHOLD}=1.5,\ D=\text{DET\_RANGE}. $$
4. For each axis whose near/far face is too close, **shift the cube** by `mov_dist` toward the sensor and record the **slab that falls outside** the new cube as a `BoxPointType` to delete (259–269):
   - near face close: `New.vertex_{min,max}[i] -= mov_dist`; the vacated slab `tmp.vertex_min[i] = old.vertex_max[i] - mov_dist` (i.e. the far slab now outside) is pushed to `cub_needrm` (260–263);
   - far face close: shift `+= mov_dist`; vacated near slab `tmp.vertex_max[i] = old.vertex_min[i] + mov_dist` pushed (264–268).
5. Commit the new cube (`LocalMap_Points = New_LocalMap_Points`, 271), `points_cache_collect()` (273), then **box-delete** all out-of-bounds slabs: `kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm)` (275). Timed into `kdtree_delete_time` (274–276).

Net effect: the tree never grows beyond ~`cube_len³` of map; box-delete (§3.4) makes the eviction O(log n + #evicted) by labeling whole interior subtrees `tree_deleted`, and the rebuild thread later physically reclaims them. This is FAST-LIO2's bounded "local map" that the paper credits the ikd-Tree for enabling at high rate.

---

## 5. Quick reference — constants & their sources

| Symbol | Value | Where |
|---|---|---|
| `NUM_MATCH_POINTS` | 5 | `common_lib.h:26` |
| plane planarity threshold | 0.1 m | `laserMapping.cpp:678` (arg to `esti_plane`) |
| kNN far-distance reject | `sqDis[4] > 5` (m²) | `laserMapping.cpp:671` |
| acceptance score | `s = 1 − 0.9·|r|/√‖p^L‖`, accept `s>0.9` | `laserMapping.cpp:681`–683 |
| `LASER_POINT_COV` (R) | 0.001 | `laserMapping.cpp:64`, 960 |
| `extrinsic_est_en` | true (default) | `laserMapping.cpp:73`, 789 |
| `h_x` width | 12 (of 23 error dims) | `laserMapping.cpp:720` |
| `filter_size_map_min` (voxel/downsample) | 0.5 m | `laserMapping.cpp:773`, 913 |
| `cube_len` | 200 m | `laserMapping.cpp:774` |
| `DET_RANGE` | 300 m | `laserMapping.cpp:77`, 775 |
| `MOV_THRESHOLD` | 1.5 | `laserMapping.cpp:78` |
| balance/delete criteria | α_bal>0.7, α_del>0.5 | `ikd_Tree.h:278`, 277 |
| min rebuild subtree | 10 | `ikd_Tree.h:14` |
| threaded-rebuild threshold | 1500 pts | `ikd_Tree.h:15` |
| op-logger ring length | 1,000,000 | `ikd_Tree.h:18` |

## 6. Differences/refinements vs. the papers (explicit)

- **Plane solver.** Paper (2107.06829) states "point-to-plane" with a normal from nearest map points; the code fixes this to a **5-point QR least-squares fit of `Au=−1`** then unit-normalizes (`common_lib.h:241`–247), with a **per-neighbor 0.1 m planarity gate** (251). These exact constants/solver are implementation choices, not in the paper.
- **Robustness = gating, not weighting.** The paper's measurement noise is a small isotropic term; the code uses a **single global** `LASER_POINT_COV=0.001` and instead rejects bad correspondences via the planarity test and the **range-scaled score** `s>0.9` (`laserMapping.cpp:681`). No per-point $R$.
- **Re-association cadence.** kNN is only re-run when `ekfom_data.converge` (`laserMapping.cpp:667`); otherwise the cached `Nearest_Points[i]` is reused — an efficiency refinement beyond the paper's per-iteration association.
- **ikd-Tree node fields are richer than DS1.** The single paper `deleted/pushdown` flags are split into point/tree and left/right variants, plus downsample bookkeeping (`down_del_num`, `*_downsample_deleted`), `radius_sq`, `father_ptr`, and recorded `alpha_*` (`ikd_Tree.h:59`–82) — needed for box-downsample and lock-safe parallel rebuild.
- **Removed-point history is discarded** in FAST-LIO2's wiring: `points_cache_collect`'s re-append is commented out (`laserMapping.cpp:226`), so the map is a pure bounded sliding window.

## 7. Open items for downstream authors (to re-confirm against the local file)

The following `ikd_Tree.cpp` **bodies** were not re-readable in this session (header decls, paper, and call sites are confirmed). Re-open and capture exact line ranges before writing formal specs: `BuildTree`, `Search`, `Search_by_range`, `calc_box_dist`, `Add_by_point`, `Add_by_range`, `Delete_by_point`, `Delete_by_range`, `Criterion_Check`, `Push_Down`, `Update`, `downsample`, `run_operation`, `flatten`, `acquire_removed_points`, `Add_Points`, `Delete_Point_Boxes`, `Nearest_Search`. The verified facts (node fields, constants, criteria thresholds, the box-prune/lazy-delete/scapegoat design, and all `laserMapping.cpp` call sites) constrain these bodies tightly.