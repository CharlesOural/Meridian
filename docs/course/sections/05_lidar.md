## 05. LiDAR residual: point-to-plane, association, map

> **Where we are.** Section 03 cast the whole estimator as a Maximum-A-Posteriori (MAP) problem that reduces to a weighted nonlinear least-squares sum of *residual factors*. Section 04 supplied the inertial factor (the IMU prior / preintegration that propagates the state and gives the prediction we linearise around). This section supplies the **single most informative geometric factor in a LiDAR-inertial system: the point-to-plane residual.** We answer four questions, in order:
>
> 1. *Association* — given one LiDAR point, which surface in the map does it correspond to? (§05.3: 5-NN plane fit + validation.)
> 2. *Residual* — what scalar do we drive to zero, and how confident are we in it? (§05.4 the residual, §05.6 the noise/weight.)
> 3. *Jacobian* — how does that scalar change as the state moves on the manifold? We **derive** $H_j$ from scratch w.r.t. rotation, translation and the LiDAR–IMU extrinsic, and check it line-by-line against the FAST-LIO code. (§05.5.)
> 4. *The map* — what data structure holds the surfaces, and how is it grown, pruned and kept query-fast in real time? (§05.7: the ikd-Tree.)
>
> The visual photometric residual is the subject of §06; GNSS in §07; how all these residuals are *solved together* (batch GN/LM vs. iterated EKF) is §08–§09; degeneracy/observability and robust kernels that re-weight this residual are §11; and §12 stitches one full estimator step around exactly the code we dissect here.

We ground every claim in the FAST-LIO / FAST-LIO2 reference implementation on disk. The central function is `h_share_model` in `FAST_LIO/src/laserMapping.cpp:638` — it is called once per solver iteration and is, almost line for line, "the LiDAR residual." Read it open beside this section.

---

### 05.1 Why the LiDAR factor matters, and what shape it takes

A LiDAR returns a dense set of 3-D points, each an accurate range measurement along a known beam direction. Unlike a camera it measures *depth directly*, so the natural error metric is geometric distance rather than photometric difference (§06). But a raw point has no semantic identity — there is no "feature ID" telling you that this return corresponds to that return three scans ago. We must *invent* the correspondence each iteration, and we must choose what geometric primitive to measure distance to.

Three classical choices (FAST-LIO2 paper §I-A, `papers/2107.06829.txt:19`):

- **point-to-point** ($z = \lVert {}^{G}p_j - q \rVert$): correspondences are unstable; the nearest map point is rarely the same physical surface point, because LiDAR scans never sample the same spot twice. Used in raw ICP, fragile.
- **point-to-line** (distance to an edge fitted from neighbours): the LOAM "corner" feature. Constrains 2 of 3 translational DoF locally.
- **point-to-plane** (distance to a locally-fitted plane): the LOAM "surf" feature, and the metric FAST-LIO uses for *all* points. A plane is the most common local surface in man-made and natural scenes, the fit is statistically well-conditioned with as few as 5 points, and the residual is a clean *scalar* — exactly one constraint per point along the surface normal.

The point-to-plane choice is decisive for the rest of the chapter: because the residual is scalar, the Jacobian $H_j$ is a **single row**, the per-point measurement noise is a **scalar variance**, and stacking $m$ points gives a tall thin $H \in \mathbb{R}^{m\times n}$ whose normal equations $H^\top R^{-1} H$ are cheap to form (§08, and the iEKF form in §09 that FAST-LIO actually runs).

#### Direct vs. feature-based (the FAST-LIO2 thesis)

LOAM and its descendants first run a hand-engineered **feature extraction** stage: per scan-line, compute a local curvature $c$ and classify each point as *edge* (high $c$), *planar* (low $c$), or discard the rest. Only ~10 % of points survive, registered as point-to-line + point-to-plane.

FAST-LIO2's first headline contribution is to **delete that stage** and register *raw* (merely voxel-downsampled) points directly, all as point-to-plane (paper abstract `papers/2107.06829.txt:5`; §I-A.3 `:19`). Two consequences:

1. **Accuracy up:** subtle structure that the curvature classifier would throw away (thin poles, foliage, textured walls) still contributes constraints. The paper: direct registration "well exploit[s] the subtle features in the environments" (`:19`).
2. **Sensor-agnostic:** the curvature heuristic assumes ring-organised mechanical-LiDAR scan lines. Solid-state LiDARs (Livox) and the irregular scan patterns of our Ouster dome have *no* such structure; deleting the stage makes the front-end "naturally adaptable to emerging LiDARs of different scanning patterns" (abstract `:5`).

This is visible in the default config: `feature_extract_enable` defaults to **`false`** (`laserMapping.cpp:787`), so `Preprocess::process` only downsamples and the *only* geometric primitive built downstream is the local plane fitted on-the-fly in `h_share_model`. The feature enum (`Real_Plane`, `Edge_Jump`, …) in `preprocess.h:14` is legacy LOAM machinery, dormant by default.

> **Meridian note.** Our front-end interface `IFrontEnd` keeps the *direct* point-to-plane residual of this section as the LiDAR factor in **both** the v1 iEKF (FAST-LIO2-style, §09/§12) and the v2 continuous-time B-spline estimator (§10). What changes between v1 and v2 is *which trajectory variable* the point is transformed by (a single end-of-scan pose vs. a spline evaluated at the point's exact timestamp), not the residual or its association. With multiple Ouster LiDARs we maintain **one shared map** (§05.7) and per-sensor extrinsics ${}^{I}_{L}T$ as graph variables (the $B$-block Jacobian of §05.5 is exactly what lets us refine them online).

---

### 05.2 Notation recap and the rigid transform of a point

Using the shared notation (course §02). The state $x$ (FAST-LIO's `state_ikfom`, `use-ikfom.hpp:12`) contains, among others:

$$
R \in SO(3)\;(\texttt{rot}),\quad p \in \mathbb{R}^3\;(\texttt{pos}),\quad {}^{I}_{L}R \in SO(3)\;(\texttt{offset\_R\_L\_I}),\quad {}^{I}_{L}t \in \mathbb{R}^3\;(\texttt{offset\_T\_L\_I}),
$$

plus velocity $v$, biases $b_g,b_a$ and gravity $g$ (the inertial part, §04). Here $R,p$ are the **IMU** body pose in the global/world frame $G$, and $({}^{I}_{L}R,{}^{I}_{L}t)$ is the **LiDAR-to-IMU extrinsic** ${}^{I}_{L}T$.

A point measured in the LiDAR frame, $p_L \equiv {}^{L}p_j$, is carried to the world frame in two hops — LiDAR→IMU, then IMU→world:

$$
\boxed{\;{}^{G}p_j \;=\; R\,\big({}^{I}_{L}R\, p_L + {}^{I}_{L}t\big) \;+\; p\;}
\tag{05.1}
$$

This is *exactly* the C++ (`laserMapping.cpp:657`, inside `h_share_model`):

```cpp
V3D p_body(point_body.x, point_body.y, point_body.z);                       // p_L
V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);     // (05.1)
```

and identically in the standalone helper `pointBodyToWorld_ikfom` (`:166-169`). Define the **intermediate point in the IMU frame**

$$
p_I \;\equiv\; {}^{I}_{L}R\, p_L + {}^{I}_{L}t,
\tag{05.2}
$$

so that ${}^{G}p_j = R\,p_I + p$. We will need both $p_L$ and $p_I$ when we differentiate, because rotation acts on $p_I$ but the extrinsic acts on $p_L$. In code (`:729`) $p_I$ is `point_this`, $p_L$ is `point_this_be` ("be" = body-end, i.e. LiDAR frame).

Throughout, ${}^{G}\hat p_j^{\,\kappa}$ denotes the world point evaluated at the *current iterate* $\hat x^{\kappa}$ of the solver; the hat and superscript $\kappa$ matter because association and linearisation are redone (or re-used — §05.4) per iteration.

---

### 05.3 Data association: 5-nearest-neighbour plane fit and validation

The map (§05.7) is a set of world-frame points with no precomputed surface model. For each downsampled body point we must, *every iteration*, (i) find its neighbourhood in the map, (ii) fit a plane, and (iii) decide whether that plane is trustworthy.

#### Step 1 — k-NN search ($k=5$)

```cpp
// laserMapping.cpp:670-671
ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false :
                         pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
```

`NUM_MATCH_POINTS = 5` (`common_lib.h:26`). The ikd-Tree returns the 5 nearest map points and their **squared** distances, sorted ascending (the search itself is §05.7). Two early rejections:

- fewer than 5 neighbours exist → reject (sparse/empty region of the map);
- the **farthest** of the 5, `pointSearchSqDis[4]`, exceeds $5\,\mathrm{m}^2$ (i.e. $\sqrt5 \approx 2.24\,\mathrm{m}$) → reject. If the 5th neighbour is that far, the 5 points are not a tight local patch and any plane through them is meaningless; this is a coarse *gating* on association, analogous to a $\chi^2$ Mahalanobis gate (§11) but done on raw Euclidean distance for speed.

Why 5 and not 3? Three points define a plane *exactly* (zero residual, no redundancy, no way to detect an outlier or a curved surface). Five gives an over-determined fit whose residual spread is itself the validity test (Step 3). It is the smallest number that is both robust and cheap.

#### Step 2 — least-squares plane fit (`esti_plane`)

A plane is $n\cdot x + d = 0$ with unit normal $n$ (shared notation §02). FAST-LIO parameterises it cleverly to avoid an eigen-decomposition. Write the plane as $a x + b y + c z + 1 = 0$ (i.e. fix the inhomogeneous term to $1$, absorbing scale). Then every neighbour $(x_j,y_j,z_j)$ must satisfy

$$
\underbrace{\begin{bmatrix} x_1 & y_1 & z_1 \\ \vdots & & \vdots \\ x_5 & y_5 & z_5\end{bmatrix}}_{A\in\mathbb{R}^{5\times3}} \begin{bmatrix} a\\ b\\ c\end{bmatrix} \;=\; \underbrace{\begin{bmatrix}-1\\ \vdots\\ -1\end{bmatrix}}_{b_0}.
\tag{05.3}
$$

This is the overdetermined system solved in `common_lib.h:226-257` (`esti_plane`), with the geometry explained verbatim in the comment block at `common_lib.h:185-191`:

```cpp
// esti_plane, common_lib.h:241-247
Matrix<T,3,1> normvec = A.colPivHouseholderQr().solve(b);   // (05.3), b = -1
T n = normvec.norm();
pca_result(0) = normvec(0) / n;   // n_x  (unit normal)
pca_result(1) = normvec(1) / n;   // n_y
pca_result(2) = normvec(2) / n;   // n_z
pca_result(3) = 1.0 / n;          // d   (signed offset)
```

The solve is least-squares (`colPivHouseholderQr().solve`, a column-pivoted Householder QR — numerically stable, no normal-equations squaring of the condition number). The returned 3-vector $[a,b,c]^\top$ is the *unnormalised* normal; dividing by its norm $\nu = \lVert[a,b,c]\rVert$ gives the **unit normal** $n=[a,b,c]/\nu$ and the **offset** $d=1/\nu$, so the validated plane is

$$
n\cdot x + d = 0,\qquad \lVert n\rVert = 1.
\tag{05.4}
$$

Geometrically: solving $A x_0 = -\mathbf 1$ minimises $\sum_j ( a x_j + b y_j + c z_j + 1)^2$. After normalisation, $a x_j+b y_j+c z_j+1$ over $\nu$ is the signed orthogonal distance of neighbour $j$ to the plane — which is precisely the quantity tested next. (This is the algebraic-least-squares plane fit; it coincides with the total-least-squares/PCA normal when the points are nearly co-planar, which is exactly the regime we accept.)

#### Step 3 — plane validation

```cpp
// esti_plane, common_lib.h:249-256, threshold = 0.1
for (int j = 0; j < NUM_MATCH_POINTS; j++)
  if (fabs(pca_result(0)*x_j + pca_result(1)*y_j + pca_result(2)*z_j + pca_result(3)) > threshold)
      return false;     // a neighbour is >0.1 m off the plane -> NOT a plane
return true;
```

Every one of the 5 neighbours must lie within `threshold = 0.1` m of the fitted plane (the literal `esti_plane(pabcd, points_near, 0.1f)` call at `laserMapping.cpp:678`). If any neighbour is farther, the patch is curved, a corner, or noise — the planarity assumption fails and the point is discarded for this scan. This is the on-the-fly equivalent of LOAM's curvature classifier, but data-driven and per-point.

The triad **(5-NN search) → (QR plane fit) → (0.1 m planarity gate)** is the entire association pipeline. Note there is *no persistent correspondence*: the plane is re-fitted from whatever 5 map points are currently nearest, which is why FAST-LIO re-runs association whenever the state has moved enough (the `converge` flag, §05.4).

---

### 05.4 The point-to-plane residual

With a validated plane $(n,d)$ for body point $p_L$ whose current world position is ${}^{G}\hat p_j^{\,\kappa}$ (05.1), the residual is the **signed point-to-plane distance**

$$
\boxed{\; r_j \;=\; n^\top\,{}^{G}\hat p_j^{\,\kappa} + d \;=\; n^\top\big({}^{G}\hat p_j^{\,\kappa} - q_j\big) \;}
\tag{05.5}
$$

where the two forms are equal because any point $q_j$ *on* the plane satisfies $n^\top q_j = -d$. The paper writes it in the centroid form $z_j = u_j^\top({}^{G}\hat p_j^{\,\kappa} - q_j)$ (FAST-LIO2 `papers/2107.06829.txt`, FAST-LIO `papers/2010.08196.txt:122`), with $u_j \equiv n$ the unit normal and $q_j$ a point on the plane. The code uses the offset form $n^\top {}^{G}p + d$, since `esti_plane` hands back $d$ directly:

```cpp
// laserMapping.cpp:680
float pd2 = pabcd(0)*point_world.x + pabcd(1)*point_world.y + pabcd(2)*point_world.z + pabcd(3);
```

`pd2` is exactly $r_j$ of (05.5). It is a *signed scalar*: positive on the normal side of the plane, negative on the other; the optimiser drives it to zero, sliding the estimated point onto the surface along $n$.

**Sign convention in the filter.** FAST-LIO stacks the *measurement* as $-r_j$, because the IKFoM update expects $z = h(x) - \text{measurement}$ with the linearised constraint $0 \approx r_j + H_j\,\delta x$ (paper `2010.08196.txt:128-130`). Hence:

```cpp
// laserMapping.cpp:751
ekfom_data.h(i) = -norm_p.intensity;   // norm_p.intensity stores pd2 = r_j  (set at :689)
```

The stored normal carries $r_j$ in its `.intensity` field (`:686-689`) — a neat trick to ship $(n, r_j)$ together as one `PointType` (`normvec` / `corr_normvect`).

**When is association recomputed?** The expensive 5-NN search runs only when `ekfom_data.converge` is true (`laserMapping.cpp:667`):

```cpp
if (ekfom_data.converge) {
    ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
    point_selected_surf[i] = ...;     // re-gate
}
```

On the *first* iteration `converge` is true so every point is associated; on subsequent iterations it is set false once the increment is small, **freezing the correspondences** (the `points_near` from last time are reused) and only the residual + Jacobian are recomputed at the updated state. This is the standard iterated-closest-point structure inside the iterated EKF (the equivalence to Gauss-Newton is §09): re-associating every iteration would be both costly and a source of oscillation, so FAST-LIO re-associates only until convergence stabilises.

The residual loop then compacts the *effective* points into contiguous arrays (`laserCloudOri` = body points, `corr_normvect` = normals+residuals) and counts them in `effct_feat_num` (`:697-706`). If `effct_feat_num < 1` the update is declared invalid (`:708-713`) — a degenerate-scan guard that hands off to §11's degeneracy handling.

---

### 05.5 The on-manifold Jacobian — full derivation

This is the mathematical heart of the section. We need $H_j = \partial r_j / \partial \delta x$, the derivative of the scalar residual (05.5) w.r.t. an increment in the error state, evaluated at the current iterate. Because $R,{}^{I}_{L}R \in SO(3)$ live on a manifold, we differentiate with the boxplus convention of §02:

$$
R \boxplus \delta\theta \;=\; R\,\mathrm{Exp}(\delta\theta),\qquad
p \boxplus \delta p \;=\; p + \delta p,
\tag{05.6}
$$

i.e. a **right** perturbation on rotation (matching FAST-LIO's `state_ikfom`, whose `boxplus` is `R·Exp(δθ)`). The error state ordering in `h_x` is (paper / `laserMapping.cpp:743`):

$$
\delta x \;=\; \big[\, \delta p \;\; \delta\theta \;\; \delta\theta_{LI} \;\; \delta t_{LI} \;\big]^\top
$$

(translation first, then body rotation, then extrinsic rotation, then extrinsic translation; velocity/bias/gravity columns are zero for the LiDAR factor, which is why `h_x` has only 12 columns, `:720`).

#### 05.5.1 The two facts we need

The residual depends on $x$ only through the world point ${}^{G}p_j$, so by the chain rule

$$
\frac{\partial r_j}{\partial \delta x} \;=\; n^\top \,\frac{\partial\, {}^{G}p_j}{\partial \delta x}.
\tag{05.7}
$$

The normal $n$ and offset $d$ are treated as **constants** w.r.t. $\delta x$: the plane is fitted from *map* points, not from the state, so it does not move when we perturb the pose. (This is the same modelling choice that makes point-to-plane a clean scalar factor; it is exact to first order because the map is the conditioning information, cf. §03's factor-graph view.)

We need the derivative of a *rotated vector* under the right perturbation. For $w = R\,a$ with $a$ fixed,

$$
R\,\mathrm{Exp}(\delta\theta)\,a \;\approx\; R\,(I + \lfloor\delta\theta\rfloor_\times)\,a \;=\; R\,a + R\,\lfloor\delta\theta\rfloor_\times a \;=\; R\,a - R\,\lfloor a\rfloor_\times \delta\theta,
\tag{05.8}
$$

using $\mathrm{Exp}(\delta\theta)\approx I+\lfloor\delta\theta\rfloor_\times$ (so3, `so3_math.h:18-34`) and the skew identity $\lfloor u\rfloor_\times v = -\lfloor v\rfloor_\times u$. Hence

$$
\frac{\partial (R a)}{\partial \delta\theta}\bigg|_{R} \;=\; -\,R\,\lfloor a\rfloor_\times .
\tag{05.9}
$$

The skew (hat) operator $\lfloor\cdot\rfloor_\times$ is `SKEW_SYM_MATRX` / `hat` in `so3_math.h:7,67`.

#### 05.5.2 Block A — translation $p$

${}^{G}p_j = R p_I + p$, so $\partial\,{}^{G}p_j/\partial \delta p = I_3$. Therefore

$$
\boxed{\;\frac{\partial r_j}{\partial \delta p} \;=\; n^\top I_3 \;=\; n^\top.\;}
\tag{05.10}
$$

This is the first three entries of the Jacobian row — literally `norm_p.x, norm_p.y, norm_p.z` (`laserMapping.cpp:743`, `:747`). Intuition: translating the sensor 1 m along the surface normal changes the point-to-plane distance by 1 m; translating *along* the surface changes it not at all. The translation Jacobian *is the plane normal*.

#### 05.5.3 Block — body rotation $R$ (the code's "A")

Only $R p_I$ depends on $R$. Apply (05.9) with $a = p_I$:

$$
\frac{\partial\,{}^{G}p_j}{\partial \delta\theta} \;=\; -\,R\,\lfloor p_I\rfloor_\times.
$$

Then

$$
\frac{\partial r_j}{\partial \delta\theta} \;=\; n^\top\big(-R\,\lfloor p_I\rfloor_\times\big) \;=\; -\,(R^\top n)^\top\,\lfloor p_I\rfloor_\times \;=\; \big(\lfloor p_I\rfloor_\times\, R^\top n\big)^\top,
\tag{05.11}
$$

where the last step uses $\lfloor u\rfloor_\times^\top=-\lfloor u\rfloor_\times$. Define the world normal pulled back into the IMU frame,

$$
C \;\equiv\; R^\top n \;=\; R^{-1} n,
\tag{05.12}
$$

then the rotation block is the row vector $\big(\lfloor p_I\rfloor_\times C\big)^\top$, i.e. as a stored 3-vector

$$
\boxed{\;A \;=\; \lfloor p_I\rfloor_\times\, C \;=\; \lfloor p_I\rfloor_\times\, R^\top n.\;}
\tag{05.13}
$$

Match to code (`laserMapping.cpp:738-739`):

```cpp
V3D C(s.rot.conjugate() * norm_vec);   // C = R^{-1} n        (05.12)
V3D A(point_crossmat * C);             // A = [p_I]_x C       (05.13), point_crossmat = [point_this]_x
```

with `point_this` $= p_I$ (`:729`) and `point_crossmat` $=\lfloor p_I\rfloor_\times$ (`:730-731`). `s.rot.conjugate()` is the quaternion inverse, i.e. $R^\top$. So the code's `A` *is* equation (05.13). (FAST-LIO stores the row as $A^\top$ entries via `VEC_FROM_ARRAY(A)` at `:743`; the sign is consistent with the residual sign convention of §05.4.)

Intuition: a small rotation moves the point on a circle whose instantaneous velocity is $\delta\theta \times (R p_I)$ in the world; projecting that onto the normal gives the change in point-to-plane distance. Points far from the rotation centre (large $\lVert p_I\rVert$) and oriented so their motion is along $n$ are the most rotation-informative — this is precisely the geometry §11 exploits to detect rotational degeneracy.

#### 05.5.4 Blocks B, C — the LiDAR–IMU extrinsic (online calibration)

When `extrinsic_est_en` is true (default, `laserMapping.cpp:789`), the extrinsic ${}^{I}_{L}T = ({}^{I}_{L}R,{}^{I}_{L}t)$ is *also* a state variable and gets refined online (paper FAST-LIO2 estimates extrinsics in the filter). We differentiate ${}^{G}p_j = R({}^{I}_{L}R\, p_L + {}^{I}_{L}t) + p$ w.r.t. the extrinsic.

**Extrinsic translation ${}^{I}_{L}t$.** It enters as $R\,{}^{I}_{L}t$, so

$$
\frac{\partial\,{}^{G}p_j}{\partial \delta t_{LI}} = R \quad\Rightarrow\quad \frac{\partial r_j}{\partial \delta t_{LI}} = n^\top R = (R^\top n)^\top = C^\top.
$$

$$
\boxed{\;\text{extrinsic-translation block} \;=\; C \;=\; R^\top n.\;}
\tag{05.14}
$$

**Extrinsic rotation ${}^{I}_{L}R$.** It enters through $p_I = {}^{I}_{L}R\, p_L + {}^{I}_{L}t$, then through $R\,p_I$. Right-perturb ${}^{I}_{L}R$: by (05.9) with $a=p_L$,

$$
\frac{\partial p_I}{\partial \delta\theta_{LI}} = -\,{}^{I}_{L}R\,\lfloor p_L\rfloor_\times,
\qquad
\frac{\partial\,{}^{G}p_j}{\partial \delta\theta_{LI}} = R\,\big(-\,{}^{I}_{L}R\,\lfloor p_L\rfloor_\times\big).
$$

Therefore

$$
\frac{\partial r_j}{\partial \delta\theta_{LI}} = -\,n^\top R\,{}^{I}_{L}R\,\lfloor p_L\rfloor_\times = -\,\big({}^{I}_{L}R^\top R^\top n\big)^\top \lfloor p_L\rfloor_\times = \big(\lfloor p_L\rfloor_\times\,{}^{I}_{L}R^\top C\big)^\top,
$$

giving the stored 3-vector

$$
\boxed{\;B \;=\; \lfloor p_L\rfloor_\times\,{}^{I}_{L}R^\top C \;=\; \lfloor p_L\rfloor_\times\,{}^{I}_{L}R^\top R^\top n.\;}
\tag{05.15}
$$

Match to code (`laserMapping.cpp:742`):

```cpp
V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);   // B = [p_L]_x · (offset_R)^T · C   (05.15)
```

with `point_be_crossmat` $=\lfloor p_L\rfloor_\times$ (`:727-728`, `point_this_be` $=p_L$) and `s.offset_R_L_I.conjugate()` $={}^{I}_{L}R^\top$. Equation (05.15) reproduced exactly.

#### 05.5.5 The assembled Jacobian row

Putting (05.10), (05.13), (05.15), (05.14) together, the per-point measurement-Jacobian row (12 columns) is

$$
H_j \;=\; \big[\; \underbrace{n^\top}_{\delta p,\ (05.10)} \;\;\; \underbrace{A^\top}_{\delta\theta,\ (05.13)} \;\;\; \underbrace{B^\top}_{\delta\theta_{LI},\ (05.15)} \;\;\; \underbrace{C^\top}_{\delta t_{LI},\ (05.14)} \;\big] \;\in\; \mathbb{R}^{1\times 12},
\tag{05.16}
$$

assembled verbatim at `laserMapping.cpp:743`:

```cpp
ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z,   // n^T   (delta p)
                                    VEC_FROM_ARRAY(A),              // A^T   (delta theta)
                                    VEC_FROM_ARRAY(B),              // B^T   (delta theta_LI)
                                    VEC_FROM_ARRAY(C);              // C^T   (delta t_LI)
```

If extrinsic estimation is **off** (`extrinsic_est_en == false`), columns $B,C$ are zeroed (`:747`): the extrinsic is held fixed at its prior. The remaining state blocks (velocity, biases, gravity — columns 12…22 of the full 23-dim error state) do not appear in the LiDAR factor at all; they are constrained only through the IMU factor (§04) and reach the LiDAR factor *indirectly* via the coupled covariance during the iterated update (§09). This is precisely what "tightly coupled" means at the residual level: each sensor contributes the residuals it can observe, and the shared state + joint information matrix transmits information to the rest.

> **Worked sanity check.** Sensor at origin, identity attitude ($R=I$, $p=0$), identity extrinsic. A wall straight ahead: outward normal $n=[-1,0,0]^\top$, plane at $x=5$, so $d=5$. A point on the wall $p_L=[5,0,0]^\top \Rightarrow p_I=p_L$, ${}^{G}p=[5,0,0]^\top$. Residual $r = n^\top{}^{G}p + d = -5+5 = 0$ ✓ (point on plane). Now the Jacobian: $C=R^\top n = n = [-1,0,0]^\top$; translation block $n^\top=[-1,0,0]$ — pushing $+x$ by $\epsilon$ makes $r = -(5+\epsilon)+5 = -\epsilon$, slope $-1$ ✓. Rotation block $A=\lfloor p_I\rfloor_\times C = \lfloor[5,0,0]\rfloor_\times[-1,0,0]^\top = [0,0,0]^\top$ — a point dead-ahead on the rotation axis through it produces no normal-direction motion to first order ✓. Move the point off-axis, $p_L=[5,2,0]^\top$: $A=\lfloor[5,2,0]\rfloor_\times[-1,0,0]^\top=[0,0,-2]^\top$ — yaw now changes the distance, magnitude growing with the lever arm ✓. The Jacobian behaves exactly as geometry demands.

---

### 05.6 Measurement noise and per-point weighting

The residual is only useful with a covariance attached (the $R_j$ that turns a residual into an *information-weighted* factor, §03). FAST-LIO uses two mechanisms.

#### A single isotropic measurement variance

The LiDAR ranging+bearing noise $n_j$ projected onto the plane normal yields a scalar variance $R_j$. The implementation uses one global constant for all points:

```cpp
// laserMapping.cpp:64
#define LASER_POINT_COV (0.001)
```

passed into the iterated update as the measurement covariance (`kf.update_iterated_dyn_share_modified(LASER_POINT_COV, ...)`, `:960`). So every point-to-plane factor gets the same $R_j = 0.001\,\mathrm{m}^2$ ($\sigma \approx 3.2$ cm). This is a deliberate simplification: the *true* projected noise depends on range and incidence angle (paper `2010.08196.txt:124-130` derives $v_j\sim\mathcal N(0,R_j)$ from $n_j$ via the plane normal), but a constant works well because (a) the planarity gate already removes high-incidence/curved patches, and (b) the per-point *robust weight* below soft-rejects the rest.

#### The range-dependent robust weight $s$

Before a point is even accepted, `h_share_model` computes a heuristic confidence and gates on it (`laserMapping.cpp:681-691`):

```cpp
float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());   // pd2 = r_j, p_body = p_L
if (s > 0.9) { point_selected_surf[i] = true;  ...  res_last[i] = abs(pd2); }
```

In symbols, with $\rho = \lVert p_L\rVert$ the range of the point,

$$
s \;=\; 1 - 0.9\,\frac{|r_j|}{\sqrt{\rho}}, \qquad \text{accept iff } s > 0.9 \;\Longleftrightarrow\; |r_j| < \tfrac{1}{9}\sqrt{\rho}.
\tag{05.17}
$$

Two roles:

1. **Outlier gate.** A point whose residual is large *relative to its range* is rejected outright. The $\sqrt\rho$ in the denominator makes the gate *looser for distant points* — far returns have larger absolute positional uncertainty (range noise and beam divergence grow with distance), so a 5 cm residual at 50 m is far more acceptable than at 2 m. This is a crude but effective range-adaptive $\chi^2$-style gate (cf. the principled Mahalanobis gate and GNC kernels of §11).
2. **(Historically) a weight.** In earlier FAST-LIO and in LOAM, $s$ multiplied the residual/Jacobian as a soft IRLS weight; in this code path $s$ is used purely as the binary accept gate ($s>0.9$) and the surviving residual then carries the uniform `LASER_POINT_COV`. The variable name and structure remain as the seam where a proper per-point weight (e.g. from incidence angle or §11 observability) would re-enter.

> **Meridian note.** This is exactly the place to upgrade. We will replace the constant `LASER_POINT_COV` with a **range-and-incidence dependent $R_j$** (project the Ouster's documented range-noise model and the beam/plane incidence onto $n$), and feed the **per-axis observability** of §11 (X-ICP / D2-LIO style) into the *back-end* noise so the factor graph (§03, L3) knows which directions the LiDAR actually constrained. The robust weight $s$ becomes a Geman-McClure / GNC kernel (§11) rather than a hard gate, so degenerate scans degrade gracefully instead of dropping points.

#### Stacking into the update

The accepted rows are stacked: $H \in \mathbb{R}^{m\times 12}$ (here $m=$ `effct_feat_num`), residual vector $z\in\mathbb{R}^{m}$ (the `ekfom_data.h(i) = -r_j`), and $R = \sigma^2 I_m$ with $\sigma^2=$ `LASER_POINT_COV`. The iterated-EKF update that consumes them — and its equivalence to a Gauss-Newton MAP step — is the subject of §09; the batch normal-equation view ($H^\top R^{-1} H\,\delta x = -H^\top R^{-1} z$ plus the IMU prior) is §08. What §05 guarantees is that each row of $H$ and each entry of $z$ are correct, code-faithful, and on-manifold.

---

### 05.7 The map: the incremental k-d tree (ikd-Tree)

The association of §05.3 issued one command per point — *"5-NN search in the map"* — and assumed the map answers in $O(\log n)$. The map is FAST-LIO2's second headline contribution: the **ikd-Tree** (`papers/2102.10808.txt`; `FAST_LIO/include/ikd-Tree/ikd_Tree.{h,cpp}`). It is a k-d tree that you can *grow and prune incrementally* and that *re-balances itself*, instead of being rebuilt from scratch each scan. The measured payoff: in FAST-LIO the average incremental-update cost is **0.23 ms vs. 5.71 ms** for a static PCL k-d tree — about **4 %** — enabling 100 Hz mapping (paper `2102.10808.txt:608`, abstract `:6`).

#### 05.7.1 Why not a static tree, an octree, or a voxel hash?

A classic LiDAR-SLAM map rebuilds a static k-d tree (PCL/FLANN) every scan over *all* map points — cost grows linearly with map size, capping update rate at ~1–10 Hz (`2102.10808.txt:12,599`). The data arrives **sequentially** and each new scan is tiny relative to the map, so rebuilding everything is mostly redundant work (`:13`). The ikd-Tree updates *only with the new points* while keeping the tree query-fast. (In Meridian's layered map, L4, the ikd-Tree is the *registration* substrate — the structure association queries — sitting beneath the NVBlox TSDF surface map and the per-keyframe point store; see the architecture notes. This section is only about the registration layer.)

#### 05.7.2 Node structure

Each node stores the standard k-d fields plus incremental bookkeeping (`ikd_Tree.h:60-81`; paper Data Structure 1, `2102.10808.txt:33-36`):

```cpp
struct KD_TREE_NODE {
    PointType point;                 // the splitting point
    uint8_t   division_axis;         // which of x/y/z this node splits on
    int  TreeSize, invalid_point_num;
    bool point_deleted, tree_deleted;            // lazy labels (this node / whole subtree)
    bool point_downsample_deleted, tree_downsample_deleted;
    bool need_push_down_to_left, need_push_down_to_right;   // lazy-label propagation
    float node_range_x[2], node_range_y[2], node_range_z[2]; // AABB of the subtree
    KD_TREE_NODE *left_son_ptr, *right_son_ptr;
    float alpha_bal, alpha_del;      // balance / deleted ratios for this subtree
};
```

The crucial additions over a textbook k-d tree are: (1) **lazy-delete labels** (`*_deleted`) so deletion is $O(1)$ flag-setting, not structural removal; (2) **`need_push_down_*`** flags that defer propagating a subtree-wide label to children until that subtree is actually visited; (3) the per-node **axis-aligned bounding box** `node_range_*` used to *prune* search; and (4) the subtree **size/invalid counts** that drive re-balancing.

#### 05.7.3 Build and balanced splitting

`Build` (`ikd_Tree.cpp:62-77`) constructs a balanced tree from a point array. At each node it (paper Algorithm 1, `2102.10808.txt:42-76`): chooses the **split axis with maximal coordinate spread/covariance**, sorts on it, takes the **median** as the node point (guaranteeing balance), and recurses on the lower/upper halves. The initial tree in FAST-LIO is built once on the first downsampled scan (`laserMapping.cpp:909-920`):

```cpp
if (ikdtree.Root_Node == nullptr) {           // first usable scan
    ikdtree.set_downsample_param(filter_size_map_min);   // map voxel size
    ... pointBodyToWorld for each ...
    ikdtree.Build(feats_down_world->points);  // balanced k-d tree from scan 1
}
```

#### 05.7.4 Nearest-neighbour search with range pruning

`Nearest_Search(point, k, out, dist, max_dist)` (`ikd_Tree.cpp:79-91`) is an **exact** k-NN (not approximate as FLANN, paper `:237`). It descends the tree like a normal k-d search but uses each subtree's `node_range_*` AABB to *prune*: if the closest possible point in a subtree's box is already farther than the current k-th best, that whole subtree is skipped. The k best are held in a bounded max-heap (`MANUAL_HEAP`, `ikd_Tree.h:93+`). This is the operation §05.3 calls per point, and it is what the whole structure is optimised to keep at $O(\log n)$ (paper `:449`). The `Push_Down` of pending lazy labels is applied before searching a subtree so a query never returns a logically-deleted point (paper `:238`).

#### 05.7.5 Incremental insert + on-tree downsampling

New scan points are added by `Add_Points(PointToAdd, downsample_on)` (`ikd_Tree.cpp:93+`; called from `map_incremental`, `laserMapping.cpp:470-471`). With `downsample_on = true` it performs the paper's **on-tree downsampling** (Algorithm 3, `2102.10808.txt:168-180`): the world is partitioned into cubes of side `filter_size_map_min` (default 0.5 m, set at `:773`/`:913`); for the cube $C_D$ containing the new point it keeps **only the single point closest to the cube centre**. Concretely (`ikd_Tree.cpp:93+`): find $C_D$, box-search the existing points inside it, compare their distance-to-centre with the new point's; if an existing point is already closer, **skip** the insertion; otherwise delete the cube's occupants (`DOWNSAMPLE_DELETE`) and insert the new point. This caps map density at one point per voxel *inside the tree itself* — no separate voxel-grid pass over the whole map.

FAST-LIO's `map_incremental` adds a fast path: points whose nearest map neighbour is already in a *different* voxel cell than the new point's voxel-centre are added **without** the downsample check (`PointNoNeedDownsample`, `laserMapping.cpp:448-449,471`), since they cannot collide with an existing in-cell point — a measurable speedup.

#### 05.7.6 Lazy delete, box-delete, and the sliding local map

**Lazy point delete / re-insert.** Deleting point $P$ just sets its node's `deleted` flag (paper §III-C, `2102.10808.txt:40`); the point stays in the tree (still traversed structurally) but is excluded from query results. If $P$ is needed again it is "re-inserted" in $O(1)$ by clearing the flag (`:78`). Physical removal happens only when a subtree is rebuilt (§05.7.7). This makes both delete and the downsample-driven churn cheap.

**Box-delete and the sliding map.** To bound memory, FAST-LIO keeps only a cube of map around the current LiDAR position and **box-deletes** everything outside it as the platform moves — `lasermap_fov_segment` (`laserMapping.cpp:231-277`). When the sensor approaches within `MOV_THRESHOLD·DET_RANGE` of a face of the local cube (`:251`), the cube is slid by `mov_dist` and the slab left behind is queued in `cub_needrm` and removed in one shot:

```cpp
// laserMapping.cpp:275
if (cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
```

`Delete_Point_Boxes` is the paper's **box-wise delete** (Algorithm 2, `2102.10808.txt:83-159`): recurse from the root, and for each subtree compare its range box $C_T$ (the `node_range_*` AABB) with the operation box $C_O$ — if **disjoint, return immediately** (no work); if $C_T \subseteq C_O$, set the subtree's `tree_deleted` lazily and mark `need_push_down` (a whole sub-map removed in $O(\log n)$, not $O(\text{points})$); if they merely overlap, test the node's own point and recurse. This box-wise sliding window is what makes a *city-scale* run fit in bounded memory at constant per-scan cost.

#### 05.7.7 Self-rebalancing by partial rebuild

Repeated inserts and lazy-deletes degrade balance and leave dead nodes. After each update, ikd-Tree checks two **scapegoat-style criteria** on the touched subtree (paper §III-D, `2102.10808.txt:218-230`; `Update`/`alpha_bal,alpha_del` in `ikd_Tree.cpp` and `ikd_Tree.h:79-80`):

$$
\alpha_{bal}(T) = \frac{\max\big(S(T.\text{left}),\,S(T.\text{right})\big)}{S(T)-1} < \beta_{bal},
\qquad
\alpha_{del}(T) = \frac{I(T)}{S(T)} < \beta_{del},
\tag{05.18}
$$

where $S(T)$ is the subtree size, $I(T)$ its invalid (lazily-deleted) count, $\beta_{bal}\in(0.5,1)$ the balance threshold and $\beta_{del}\in(0,1)$ the deleted-ratio threshold. FAST-LIO's defaults: $\beta_{bal}=0.6$, $\beta_{del}=0.5$ (paper Table II, `:441-444`; header defaults `ikd_Tree.h:213-214`). Violating **either** criterion triggers a **partial rebuild** of just that subtree: flatten it to an array, *drop the lazily-deleted points*, and `Build` a fresh balanced subtree (Algorithm `2102.10808.txt:231`). The $\alpha_{bal}$ criterion caps tree height at $\log_{1/\beta_{bal}} n$ (so search stays $O(\log n)$); the $\alpha_{del}$ criterion reclaims the dead nodes.

**Parallel rebuild.** Rebuilding a *large* subtree would stall the 100 Hz pipeline. If the subtree exceeds `Multi_Thread_Rebuild_Point_Num = 1500` (`ikd_Tree.h:21`; paper $N_{max}=1500$, `:441`) the rebuild is handed to a **second thread** (paper Algorithm 4, `:243-271`): the worker locks *updates but not queries*, flattens valid points, builds the new subtree off to the side while incoming insert/delete requests are recorded in an **operation logger**, replays the logged ops onto the new subtree, then swaps it in under a one-instruction `LockAll`. The main thread keeps answering the §05.3 nearest-neighbour queries the whole time. Net effect (paper Fig. 6, `:604`): nearly constant ~1.6 ms per scan regardless of map size.

#### 05.7.8 The map–residual loop, end to end

Per scan (`laserMapping.cpp` main loop, `:869-977`), the LiDAR factor and its map interact as:

```
sync scan + IMU            (:869)            -> §04 forward-propagate state x̂
lasermap_fov_segment       (:901)            -> slide local cube, box-delete (§05.7.6)
voxel-downsample the scan  (:904-907)        -> feats_down_body
update_iterated_dyn_share  (:960)            -> iterate:  (§09)
    └─ h_share_model       (:638)            ->   for each point:
            (05.1) to world (:657)
            5-NN search     (:670)   ───────────────► ikd-Tree Nearest_Search  (§05.3, §05.7.4)
            esti_plane      (:678)           ->     fit+validate plane         (§05.3)
            residual pd2    (:680)           ->     r_j = n·p + d              (§05.4)
            weight gate s   (:681-683)       ->     accept/reject              (§05.6)
            build H_j row   (:743)           ->     [n  A  B  C]               (§05.5)
map_incremental            (:976)   ───────────────► ikd-Tree Add_Points       (§05.7.5)
```

The map is *read* (k-NN) during the iterated update and *written* (insert + downsample) once after convergence, then *slid* (box-delete) as the platform moves. That read/fit/residual/Jacobian/write cycle is the complete LiDAR factor.

---

### 05.8 Summary and pointers

- The LiDAR factor is a **direct point-to-plane** residual on raw downsampled points — no feature extraction (`feature_extract_enable=false`), which buys accuracy and sensor-agnosticism (FAST-LIO2's thesis, §05.1).
- **Association** = 5-NN search in the map → QR plane fit (`esti_plane`, eq. 05.3–05.4) → 0.1 m planarity gate (§05.3), redone only until the iterate converges (§05.4).
- The **residual** is the signed normal distance $r_j = n^\top{}^{G}p_j + d$ (eq. 05.5; code `pd2`).
- The **Jacobian** row (eq. 05.16) is $[\,n^\top\;A^\top\;B^\top\;C^\top\,]$ with $C=R^\top n$, $A=\lfloor p_I\rfloor_\times C$ (rotation), $B=\lfloor p_L\rfloor_\times {}^{I}_{L}R^\top C$ (extrinsic rotation), $C$ again (extrinsic translation) — derived in §05.5 from the right-perturbation rule (05.9) and matched line-by-line to `laserMapping.cpp:738-743`. Translation Jacobian *is the normal*; the $B,C$ blocks are what enable **online extrinsic calibration**.
- **Weighting** is a constant `LASER_POINT_COV = 0.001` plus a range-adaptive accept gate $s$ (eq. 05.17) — both flagged as Meridian upgrade points toward a principled range/incidence noise model and a GNC robust kernel (§11).
- The **map** is the **ikd-Tree**: balanced build, $O(\log n)$ pruned exact k-NN, $O(1)$ lazy delete, on-tree voxel downsampling, $O(\log n)$ box-delete for the sliding window, and scapegoat partial (optionally parallel) re-balancing — ~4 % of a static tree's cost (§05.7).

**Cross-references.** The probabilistic meaning of "residual + covariance = factor" is §03; the IMU factor that propagates the state we linearise around is §04; the photometric residual that shares this state is §06; GNSS absolute factors §07; how $H,z,R$ here feed Gauss-Newton/LM batch solves §08 and the iterated-EKF recursion FAST-LIO actually runs §09; the continuous-time version (point transformed by a B-spline at its own timestamp) §10; degeneracy/observability and robust kernels that re-weight this factor §11; and the full single-step walkthrough mapped onto FAST-LIO2 §12.
