## 03. Probabilistic foundation: MAP=NLS, Gaussians, factor graphs

> **Where this sits.** Section 01 placed us on the loose/tight coupling spectrum and announced that *all* of it is one estimation problem. Section 02 gave us the geometric stage: the state lives on a manifold $\mathcal{M}$, errors live in the tangent space, and we manipulate them with $\boxplus$/$\boxminus$, $\mathrm{Exp}/\mathrm{Log}$, and the hat operator $(\cdot)^\wedge$. This section is the **theoretical spine** that connects the two. We answer one question with full rigor: *given a probabilistic model of the sensors, what number do we actually compute, and why is that number a nonlinear least-squares problem?* Everything downstream — the IMU residual (§04), the LiDAR point-to-plane residual (§05), the photometric residual (§06), GNSS (§07), the batch solvers (§08), the recursive/iterated-Kalman solvers (§09), continuous-time (§10), and the robustness machinery (§11) — is a special case or a refinement of what we derive here. §12 reassembles all of it into one estimator step.
>
> **Notation reminder (defined in §02).** State $x \in \mathcal{M}$; rotation $R \in SO(3)$; position $p$; velocity $v$; biases $b_g, b_a$; gravity $g$. Boxplus/boxminus $\boxplus,\boxminus$; $\mathrm{Exp}/\mathrm{Log}$ between $\mathbb{R}^3$ and $SO(3)$; hat $(\cdot)^\wedge$. Residual $r$, measurement $z$, prediction $h(x)$, Jacobian $H$, covariance $\Sigma$, information $\Omega = \Sigma^{-1}$, Kalman gain $K$.

---

### 03.1 The object we are estimating, and the object we are computing

A SLAM estimator does not "find the trajectory." It computes a **probability distribution over trajectories** and then reports a *point* from that distribution (usually its mode) together with a *spread* (its covariance). Keeping these two things — the distribution and the point summary — conceptually separate is the single most clarifying idea in this chapter, because the two great families of solvers differ only in which they emphasize:

- **Batch / smoothing / factor-graph optimizers** (§08, the GTSAM iSAM2 back-end in Meridian's L3) keep an explicit handle on the joint distribution over *many* states and find its mode by nonlinear least squares.
- **Recursive / filtering estimators** (§09, the iterated-EKF front-end Meridian builds first as `IFrontEnd` v1, grounded in FAST-LIO's `esekfom.hpp`) propagate a Gaussian summary of *one* state forward in time and update it as data arrives.

The punchline of this whole course — proven by FAST-LIO and reused everywhere since — is that **these are the same computation** restricted to different windows of the same factor graph (§09.x). This section builds the shared substrate.

Let $X = \{x_0, x_1, \dots, x_N\}$ be the collection of states we want to estimate. In a sliding-window front-end the $x_i$ are poses (or full inertial states) at a handful of recent times; in the back-end they are keyframe states $x_i \in \mathcal{M}$. Let $Z = \{z_1, \dots, z_M\}$ be everything the sensors told us: IMU increments, LiDAR points matched to planes, image patches, GNSS fixes. The **posterior** is the conditional distribution of the states given the data,

$$
p(X \mid Z).
$$

Our job is to characterize this object. The **maximum a posteriori (MAP)** estimate is the trajectory that maximizes it:

$$
\boxed{\;X^\star = \underset{X \in \mathcal{M}}{\arg\max}\; p(X \mid Z).\;}
\tag{03.1}
$$

The rest of §03.2–03.4 turns (03.1) into a weighted nonlinear least-squares problem. §03.5–03.6 reorganize that problem as a graph. §03.7–03.8 develop the two algebraic forms (covariance vs. information) and the operation — marginalization via the Schur complement — that lets us *delete* variables from the graph while preserving the information they carried. That last operation is what makes both fixed-lag smoothing and the iterated EKF tractable, and it is the bridge to §08/§09.

---

### 03.2 From Bayes to a sum of squared residuals

#### 03.2.1 Factorizing the posterior

Apply Bayes' rule to (03.1):

$$
p(X \mid Z) = \frac{p(Z \mid X)\, p(X)}{p(Z)}.
\tag{03.2}
$$

The denominator $p(Z)$ (the *evidence*) does not depend on $X$, so it cannot move the maximizer and we drop it. Two modeling assumptions, both standard and both physically reasonable for our sensor suite, collapse the numerator into a product:

1. **Conditional independence of measurements.** Given the true states, distinct measurements are independent — a LiDAR return and an IMU sample do not influence each other except through the trajectory that produced both. Then the likelihood factorizes:
$$
p(Z \mid X) = \prod_{k=1}^{M} p(z_k \mid X_k),
\tag{03.3}
$$
where $X_k \subseteq X$ is the (usually small) subset of states that measurement $k$ actually touches. A LiDAR-plane residual touches one pose; an IMU preintegration factor (§04) touches two consecutive states; a loop closure touches two keyframes. This *locality* — each factor sees only a few variables — is the structural fact that the whole rest of the chapter exploits.

2. **A prior that itself factorizes.** The prior $p(X)$ encodes what we believe before looking at $Z$: an initial-state prior, the IMU process model linking $x_{i}$ to $x_{i+1}$, the smoothness of a B-spline (§10), or — in the recursive view — the entire summary of all past data, marginalized into a single Gaussian on the current window (§03.8, §09). Write it generically as a product of prior factors $\prod_j p_j(X_j)$.

Substituting (03.3) into (03.2) and folding the prior factors into the same product, the MAP problem becomes

$$
X^\star = \arg\max_{X} \prod_{k} \phi_k(X_k),
\tag{03.4}
$$

where each $\phi_k$ is either a likelihood term $p(z_k\mid X_k)$ or a prior term. Each $\phi_k$ is a **factor**: a nonnegative function of a small subset of variables. Equation (03.4) is *already* a factor graph (§03.5); we have just not drawn it yet.

#### 03.2.2 The negative-log transform: products become sums

Maximizing a product of many small numbers is numerically and analytically awkward. Take the negative logarithm — monotone, so it preserves the maximizer (now a minimizer):

$$
X^\star = \arg\min_{X}\; \sum_{k} \underbrace{\big(-\ln \phi_k(X_k)\big)}_{\displaystyle \text{cost of factor } k}.
\tag{03.5}
$$

This is the central reduction of the section: **MAP estimation is the minimization of a sum of per-factor costs.** Nothing so far is Gaussian or quadratic; (03.5) is general. The Gaussian assumption, next, is what turns each cost into a *squared* term and hands us least squares.

#### 03.2.3 The Gaussian assumption makes each cost quadratic

Model each measurement as its noiseless prediction plus zero-mean Gaussian noise. In the manifold setting of §02 the measurement model is written with $\boxplus$/$\boxminus$ so that the noise lives in a vector (tangent) space even when the prediction lives on $\mathcal{M}$:

$$
z_k = h_k(X_k) \,\boxplus\, \eta_k,
\qquad \eta_k \sim \mathcal{N}(0, \Sigma_k),
\tag{03.6}
$$

with $h_k$ the (nonlinear) measurement function — for a LiDAR plane, $h_k$ transforms a body-frame point into the map and evaluates $n^\top x + d$ (§05); for the IMU, $h_k$ is the preintegrated increment (§04). Define the **residual** as the $\boxminus$-difference between what we measured and what the current state predicts,

$$
\boxed{\; r_k(X_k) \;=\; z_k \,\boxminus\, h_k(X_k) \;\in\; \mathbb{R}^{d_k}. \;}
\tag{03.7}
$$

The residual is a *tangent-space* vector of dimension $d_k$ (the measurement's degrees of freedom), and *that* is where the Gaussian lives. A zero-mean Gaussian density in $r$ is

$$
p(z_k \mid X_k) \;\propto\; \exp\!\Big(-\tfrac{1}{2}\, r_k(X_k)^\top \Sigma_k^{-1}\, r_k(X_k)\Big),
\tag{03.8}
$$

so its negative log is, up to a constant that cannot move the minimizer,

$$
-\ln \phi_k(X_k) \;=\; \tfrac{1}{2}\, r_k^\top \Omega_k\, r_k \;+\; \text{const},
\qquad \Omega_k := \Sigma_k^{-1}.
\tag{03.9}
$$

Here $\Omega_k$ is the **information (precision) matrix** of measurement $k$. The quadratic form $r_k^\top \Omega_k r_k$ is the **squared Mahalanobis norm** $\lVert r_k \rVert_{\Sigma_k}^2$ — the squared length of the residual *in the metric defined by its own uncertainty*. A confident sensor (small $\Sigma$, large $\Omega$) makes its residual expensive to violate; a noisy sensor is cheap to disagree with. This is exactly how Meridian's robustness layer (§11) wants to inject per-axis observability: by *shaping* $\Omega_k$ so that well-observed directions are stiff and degenerate directions go slack.

#### 03.2.4 The result: weighted nonlinear least squares

Insert (03.9) into (03.5) and drop the constants:

$$
\boxed{\;
X^\star \;=\; \arg\min_{X \in \mathcal{M}}\; \frac{1}{2}\sum_{k=1}^{M}\; \big\lVert r_k(X_k) \big\rVert^2_{\Sigma_k}
\;=\; \arg\min_{X}\; \frac{1}{2}\sum_{k=1}^{M} r_k(X_k)^\top \Omega_k\, r_k(X_k).
\;}
\tag{03.10}
$$

This is the equation the entire estimator computes. Three observations frame the rest of the chapter:

- It is **nonlinear** (the $h_k$, and the manifold $\boxminus$, are nonlinear) and it is **least squares** (each term is a squared Mahalanobis residual). Hence: *MAP = NLS under Gaussian assumptions.* §08 attacks (03.10) directly with Gauss–Newton / Levenberg–Marquardt; §09 shows the iterated EKF solving the same (03.10) over a sliding window.
- It is **separable into a sum over factors**, each touching few variables. That separability is the factor graph (§03.5) and the source of the sparsity (§03.6) that makes large problems solvable.
- The **weights $\Omega_k$ are the modeling knobs.** Choosing them well — and re-weighting them online for robustness (§11) — is where domain knowledge enters. The squared form is exactly what fails catastrophically under outliers, which is why §11 replaces $\lVert r_k\rVert^2$ with a robust kernel $\rho(\lVert r_k\rVert)$, an iteratively-reweighted variant of the *same* (03.10).

> **Meridian grounding.** In FAST-LIO this entire structure is realized concretely. The state $x$ — `state_ikfom` in `FAST_LIO/include/use-ikfom.hpp:12-21` — packs `pos`, `rot`, `offset_R_L_I`, `offset_T_L_I`, `vel`, `bg`, `ba`, `grav` on the product manifold $\mathbb{R}^3\times SO(3)\times SO(3)\times\mathbb{R}^3\times\mathbb{R}^3\times\mathbb{R}^3\times\mathbb{R}^3\times S^2$ (note gravity is carried on the 2-sphere $S^2$, a 2-DoF manifold, not a free $\mathbb{R}^3$ — exactly the $\mathcal{M}$ of §02; this is why the toolkit threads special `S2_Mx`/`S2_Nx_yy` projection maps through every update). The FAST-LIO paper writes the per-point residual $z_j = \mathbf{u}_j^\top({}^G\mathbf{T}_{I_k}\,{}^I\mathbf{T}_L\,{}^{L}\mathbf{p}_{f_j} - {}^G\mathbf{q}_j)$ (paper 2010.08196 eq (12)) — our $r_k$ — and assembles them with their Jacobians in the `h_share_model` callback (in `FAST_LIO/src/laserMapping.cpp`), feeding the stacked residual $z_k = [z_1^\top,\dots,z_m^\top]^\top$ and Jacobian $H = [H_1^\top,\dots,H_m^\top]^\top$ (paper eq (14)) to the iterated update in `FAST_LIO/include/IKFoM_toolkit/esekfom/esekfom.hpp`. The covariance/information bookkeeping ($\Sigma \leftrightarrow$ the code's `P_`, gain $K \leftrightarrow$ `K_`) lives in that same `esekfom.hpp` update loop (the `update_iterated_dyn_share()` method, lines ~1001-1110). We dissect that loop as the recursive solution of (03.10) in §09; here it is enough to recognize that the filter *minimizes* (03.10) — the paper itself states this, deriving its update from the explicit MAP cost in eq (17) (see §03.8.3 below), not doing something philosophically different.

---

### 03.3 Worked micro-example: two factors on one pose

To make (03.10) concrete before generalizing, take the smallest non-trivial case: estimate a single 2D position $x = p \in \mathbb{R}^2$ from (i) a prior $\hat p_0$ with covariance $\Sigma_0 = \sigma_0^2 I$ and (ii) one range-bearing-like linear measurement $z = A p + \eta$, $\eta\sim\mathcal{N}(0,\Sigma_z)$. Two factors, one variable.

The cost is

$$
J(p) = \tfrac12 (p - \hat p_0)^\top \Omega_0 (p - \hat p_0) \;+\; \tfrac12 (z - A p)^\top \Omega_z (z - A p),
\qquad \Omega_0=\Sigma_0^{-1},\ \Omega_z=\Sigma_z^{-1}.
$$

Both residuals are *linear*, so the cost is exactly quadratic and the minimizer is found by setting the gradient to zero:

$$
\nabla J = \Omega_0(p - \hat p_0) - A^\top \Omega_z (z - A p) = 0
\;\;\Longrightarrow\;\;
\underbrace{(\Omega_0 + A^\top \Omega_z A)}_{\displaystyle \Lambda\ (\text{information})}\, p^\star = \underbrace{\Omega_0 \hat p_0 + A^\top \Omega_z z}_{\displaystyle \xi\ (\text{information vector})}.
\tag{03.11}
$$

Read off three lessons that recur at full scale:

1. **Information adds.** The posterior information $\Lambda = \Omega_0 + A^\top \Omega_z A$ is the prior information *plus* the measurement information mapped into state space by $A^\top \Omega_z A$. Stacking factors = summing their information contributions. This is the additive structure (§03.7) that the Kalman update (§09) and the graph's $H^\top \Omega H$ (§08) both inherit.
2. **The point estimate is a linear system.** $p^\star = \Lambda^{-1}\xi$. At full scale $\Lambda$ is large but *sparse* (§03.6), and solving $\Lambda\,\delta = \xi$ is the inner step of every solver in §08–§09.
3. **The covariance is the inverse information.** $\Sigma^\star = \Lambda^{-1}$. The estimator's *confidence* falls straight out of the same matrix that produced the estimate — which is why Meridian can publish a confidence overlay (L6) for free from the back-end, and why per-axis observability (§11) is literally a spectral statement about $\Lambda$.

Nonlinear factors (the real case) reduce to (03.11) *per iteration* after linearization — that is precisely Gauss–Newton (§08.2). The linear micro-example is the iteration step in disguise.

---

### 03.4 Linearization on the manifold: how the squared cost becomes a normal equation

Real $h_k$ are nonlinear and $X$ lives on $\mathcal{M}$, so we cannot solve (03.10) in closed form. We linearize *in the tangent space* (§02) around the current estimate $\hat X$, write the update as a tangent vector $\delta x$ applied with $\boxplus$, and iterate.

Parameterize $X = \hat X \boxplus \delta x$. The residual (03.7), seen as a function of $\delta x$, has the first-order expansion

$$
r_k(\hat X \boxplus \delta x) \;\approx\; r_k(\hat X) \;+\; H_k\, \delta x_k,
\qquad
H_k \;:=\; \left.\frac{\partial\, r_k(\hat X \boxplus \delta x)}{\partial\, \delta x}\right|_{\delta x = 0}.
\tag{03.12}
$$

$H_k$ is the **Jacobian of the residual with respect to the tangent-space increment** — the manifold version of the ordinary Jacobian, and the quantity §02 taught us to compute with the right $\boxplus$-Jacobians. (For $SO(3)$ this is where the right-Jacobian $J_r$ and the $\mathrm{Exp}/\mathrm{Log}$ derivatives enter; §04–§06 derive the concrete $H_k$ for IMU, LiDAR, and photometric factors.) Substituting (03.12) into (03.10) gives a quadratic in $\delta x$:

$$
\tfrac12\sum_k \lVert r_k(\hat X) + H_k\,\delta x_k\rVert^2_{\Sigma_k}.
$$

Stack all residuals into $r$, all Jacobians into a tall block matrix $H$ (one block-row per factor, zeros where a factor does not touch a variable), and assemble the block-diagonal weight $\Omega = \mathrm{blockdiag}(\Omega_k)$. Setting the gradient to zero yields the **normal equations**:

$$
\boxed{\; \big(\,H^\top \Omega\, H\,\big)\, \delta x^\star \;=\; -\,H^\top \Omega\, r. \;}
\tag{03.13}
$$

Define $\Lambda := H^\top \Omega H$ (the **Hessian approximation** / system information matrix) and $g := H^\top \Omega r$ (the gradient). Then $\delta x^\star = -\Lambda^{-1} g$, and we update on the manifold,

$$
\hat X \;\leftarrow\; \hat X \,\boxplus\, \delta x^\star,
\tag{03.14}
$$

re-linearize, and repeat until $\delta x^\star$ is negligible. That loop is **Gauss–Newton**; damping $\Lambda \to \Lambda + \lambda\,\mathrm{diag}(\Lambda)$ gives **Levenberg–Marquardt** (full treatment in §08). The crucial structural point for *this* section is that (03.13) has the *exact same algebra* as the micro-example (03.11): information $\Lambda = H^\top\Omega H$ is a sum of per-factor rank-$d_k$ contributions $H_k^\top \Omega_k H_k$, and the right-hand side is the corresponding sum. The graph's sparsity (§03.6) is therefore the sparsity of $\Lambda$.

> **Meridian grounding.** This is literally what `esekfom.hpp` computes each iteration: the `h_share_model` callback (in `laserMapping.cpp`) returns $r_k$ and $H_k$ for the active LiDAR-plane correspondences; the toolkit assembles $H^\top \Omega H$ together with the propagated prior information and solves for the tangent increment, then applies it via the state's $\boxplus$ (the `boxplus`/`oplus` operator defined alongside `state_ikfom` in `use-ikfom.hpp`). FAST-LIO's contribution is *not* a different objective — it is a clever, equivalent *factorization* of solving (03.13) when the measurement dimension is huge (thousands of LiDAR points) but the state is small (a few dozen DoF); see §09 for the equivalence proof.

---

### 03.5 Factor graphs: drawing the objective

Equation (03.4) — a product of factors, each over a small variable subset — is naturally a **bipartite graph**. A factor graph $\mathcal{G} = (\mathcal{V}, \mathcal{F}, \mathcal{E})$ has:

- **Variable nodes** $\mathcal{V}$, one per state $x_i$ (drawn as circles);
- **Factor nodes** $\mathcal{F}$, one per term $\phi_k = \exp(-\tfrac12\lVert r_k\rVert^2_{\Sigma_k})$ in (03.10) (drawn as squares);
- **Edges** $\mathcal{E}$ connecting factor $k$ to exactly the variables $X_k$ in its residual.

"Bipartite" means edges only ever run variable–factor, never variable–variable or factor–factor. The graph *is* the objective: the cost (03.10) is recovered by reading off, for every square, the residual of the variables it touches, and summing. A small LiDAR-IMU window in Meridian's front-end looks like:

```
   (prior)                IMU factor          IMU factor
      |                    /      \            /      \
   [ x0 ] --- [LiDAR] --[ x1 ]--[LiDAR]--[ x2 ]--[LiDAR]-- ...
      \________________________________/
                  (GNSS factor, absolute, touches x0)

  circles () = states (poses / inertial states)   ← variables
  squares [] = factors (residuals)                ← measurements & priors
```

Reading the graph:

- A **unary** factor touches one variable: the initial prior, a GNSS fix (§07), a single LiDAR-plane residual (§05). $H_k$ has one nonzero block-column.
- A **binary** factor touches two: an IMU preintegration factor between consecutive states (§04), a loop closure between two keyframes (§05/§11). $H_k$ has two nonzero block-columns.
- The graph makes the locality structural: a factor's residual gradient $H_k^\top \Omega_k r_k$ is nonzero *only* in the columns for $X_k$. Summing these into $\Lambda = H^\top\Omega H$ therefore produces a matrix whose nonzero pattern *is the adjacency of the graph* (§03.6).

This is why Meridian's L3 back-end is built on GTSAM/iSAM2: GTSAM *is* a factor-graph library. Keyframes are variable nodes; IMU, LiDAR-odometry, loop-closure, GNSS, and online-extrinsic-refinement constraints are factor nodes; iSAM2 incrementally re-solves (03.10) as factors arrive. The front-end's sliding window is the *same* graph truncated to recent variables — and the act of truncating it correctly is marginalization (§03.8).

> **A note on the two layers being one graph.** It is tempting to think of Meridian's continuous-time/iEKF front-end (L2) and the iSAM2 back-end (L3) as different mathematics. They are not. They are two windows onto a single, ever-growing factor graph: the front-end keeps a short tail with tight real-time deadlines and marginalizes the rest into a prior; the back-end keeps keyframe summaries over the whole mission and optimizes for global consistency. Section 12 makes this explicit. Holding "it is all one graph (03.10)" in mind prevents a great deal of confusion later.

---

### 03.6 Sparsity: why locality is the whole game

The reason 3D SLAM is tractable at all is that $\Lambda = H^\top \Omega H$ is **sparse**, and its sparsity is dictated by the graph. Two variables $x_i, x_j$ produce a nonzero off-diagonal block $\Lambda_{ij}$ **iff some factor touches both of them** — i.e., iff they share a factor node. Concretely:

- $\Lambda_{ii} = \sum_{k:\, i \in X_k} H_{k,i}^\top \Omega_k H_{k,i}$ — variable $i$'s diagonal block is the sum of information from every factor touching it.
- $\Lambda_{ij} = \sum_{k:\, i,j \in X_k} H_{k,i}^\top \Omega_k H_{k,j}$ — the off-diagonal exists only for variables co-observed by a common factor.

Because each factor in our system touches one or two variables, $\Lambda$ is **block-tridiagonal-plus-loops**: a band from the chain of IMU/odometry factors, plus a few off-band blocks from loop closures and GNSS. This structure is what every solver in §08–§09 exploits:

- **Batch (§08):** a sparse Cholesky factorization $\Lambda = LL^\top$ costs orders of magnitude less than dense $O(n^3)$, and a good variable ordering keeps the *fill-in* (new nonzeros created during factorization) small.
- **Schur complement (§03.8, §08.x):** when many variables are "leaves" (e.g., per-point or per-landmark variables touched by few factors), eliminating them first via the Schur complement reduces the problem to a small dense system on the remaining variables. FAST-LIO's "new Kalman gain" trick is exactly a Schur-complement choice that inverts in *measurement* space or *state* space depending on which is smaller (§09).
- **Incremental (§03.8, §09):** the band structure means a new measurement at the tail only perturbs a few blocks, so we can update the factorization instead of refactoring — the principle behind iSAM2 and fixed-lag smoothing.

The single sentence to remember: **the sparsity of the linear algebra is the bipartite structure of the factor graph.** Lose locality and you lose tractability.

---

### 03.7 Two algebraic forms: covariance vs. information

The same Gaussian belief can be written two equivalent ways, and SLAM constantly converts between them because each makes a different operation cheap. For a Gaussian on $\delta x$ (the tangent-space error of §02):

| | **Covariance (moment) form** | **Information (canonical) form** |
|---|---|---|
| Parameters | mean $\mu$, covariance $\Sigma$ | info vector $\eta = \Omega\mu$, info matrix $\Omega = \Sigma^{-1}$ |
| Density | $\propto \exp\!\big(-\tfrac12 (\delta x - \mu)^\top \Sigma^{-1}(\delta x - \mu)\big)$ | $\propto \exp\!\big(-\tfrac12 \delta x^\top \Omega\, \delta x + \eta^\top \delta x\big)$ |
| Cheap operation | **marginalization** (drop variables: just delete rows/cols of $\Sigma$) | **conditioning / adding measurements** (just *add* $H^\top\Omega_z H$ to $\Omega$) |
| Expensive operation | conditioning (needs the Kalman update / a matrix inverse) | marginalization (needs a Schur complement, §03.8) |
| Used by | the EKF *covariance* propagation (§09) | the factor graph / least-squares normal equations (§08) |

The duality is exact: $\Omega = \Sigma^{-1}$, $\eta = \Omega\mu$, $\mu = \Omega^{-1}\eta$. The two facts to internalize:

- **Adding independent information is addition in the information form.** Fusing measurement $z$ into a prior is $\Omega^+ = \Omega^- + H^\top\Omega_z H$ and $\eta^+ = \eta^- + H^\top \Omega_z (z\text{-prediction terms})$ — exactly the micro-example (03.11) and the normal equations (03.13). This is why factor graphs *are* the information form: each factor literally adds its $H_k^\top\Omega_k H_k$ block. Adding a factor is cheap; the information matrix stays sparse.
- **Dropping a variable is easy in covariance form, hard in information form.** To marginalize out a variable from a *covariance*, you delete its rows and columns. To marginalize it out of an *information* matrix, you cannot just delete — you must pay a Schur complement (§03.8), and doing so *fills in* the graph, creating new edges among the variable's former neighbors.

The estimator therefore lives in tension: it *wants* the information form (sparse, additive, perfect for fusing the flood of LiDAR/IMU/visual factors), but the operation that keeps the problem bounded in real time — throwing old states away — is the one the information form charges for. Resolving that tension is marginalization, and it is the bridge to §08 (batch with Schur) and §09 (recursive filtering as marginalize-everything-but-the-present).

---

### 03.8 Marginalization and the Schur complement: deleting variables without losing their information

A real-time estimator cannot keep every past state. The front-end window must stay small; the back-end must summarize old keyframes. The principled way to remove a variable is **marginalization**: integrate it out of the joint, leaving its influence baked into a prior over the variables that remain. Done correctly, this is *lossless* up to the linearization point — it is *not* the same as merely deleting the variable and its factors (which would discard information and corrupt the estimate).

#### 03.8.1 The Gaussian marginal

Partition the variables into a block $a$ we want to **keep** and a block $b$ we want to **marginalize out**. Write the joint information form (the linearized system, §03.4):

$$
\Lambda = \begin{bmatrix} \Lambda_{aa} & \Lambda_{ab} \\ \Lambda_{ba} & \Lambda_{bb} \end{bmatrix},
\qquad
\eta = \begin{bmatrix} \eta_a \\ \eta_b \end{bmatrix}.
$$

A fundamental Gaussian identity says: the marginal over the kept block $a$, $p(a) = \int p(a,b)\,\mathrm{d}b$, is again Gaussian, with information

$$
\boxed{\;
\Lambda_a^{\text{marg}} \;=\; \Lambda_{aa} \;-\; \Lambda_{ab}\,\Lambda_{bb}^{-1}\,\Lambda_{ba},
\qquad
\eta_a^{\text{marg}} \;=\; \eta_a \;-\; \Lambda_{ab}\,\Lambda_{bb}^{-1}\,\eta_b.
\;}
\tag{03.15}
$$

The matrix $\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba}$ is the **Schur complement** of $\Lambda_{bb}$ in $\Lambda$. It is the algebraic engine behind nearly everything in §08–§09. Read (03.15) physically: the term $-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba}$ is the information that flowed *through* the eliminated variable $b$ and is now redistributed among $b$'s neighbors as a new, dense block. This is **fill-in**: marginalizing a variable connects all of its former neighbors to each other (they now share the marginal prior factor). A long-baseline loop-closure variable, marginalized, can therefore turn a sparse band into something denser — the practical reason fixed-lag windows and careful keyframe selection matter.

#### 03.8.2 Why the Schur complement is also how we *solve* (and the FAST-LIO connection)

The very same Schur complement appears when **solving** the normal equations (03.13), not just when marginalizing. To solve $\Lambda \begin{bsmallmatrix}\delta a\\\delta b\end{bsmallmatrix} = -g$ you can eliminate $\delta b$ first:

$$
(\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\,\delta a = -(g_a - \Lambda_{ab}\Lambda_{bb}^{-1} g_b),
\qquad
\delta b = \Lambda_{bb}^{-1}(-g_b - \Lambda_{ba}\,\delta a).
\tag{03.16}
$$

If $b$ is high-dimensional but block-diagonal (e.g. per-point or per-landmark variables, each touched by few factors), $\Lambda_{bb}^{-1}$ is cheap and the reduced system on $a$ is small. This is the bundle-adjustment Schur trick and, in spirit, **FAST-LIO's celebrated Kalman-gain reformulation**: when the measurement vector is enormous (thousands of LiDAR points, so the innovation covariance $H\Sigma H^\top + R$ is huge) but the state is small, FAST-LIO chooses the algebraically *equivalent* form of the update that requires inverting a matrix of *state* dimension rather than *measurement* dimension. The paper (2010.08196, FAST-LIO) states and *proves* this equivalence: its conventional gain $K = P H^\top (H P H^\top + R)^{-1}$ (eq (18), with $P$ the propagated prior covariance) is replaced by the new form $K = (H^\top R^{-1} H + P^{-1})^{-1} H^\top R^{-1}$ (eq (20)), proven equal in Appendix B "based on the matrix inverse lemma." Their motivation is precisely the one we are developing: the cost (eq (17)) "is over the state, hence the solution should be calculated with complexity depending on the state dimension." The identity is the matrix-inversion-lemma / Schur-complement statement:

$$
\underbrace{\Sigma H^\top (H \Sigma H^\top + R)^{-1}}_{\text{invert in measurement space }(d_z\times d_z)}
\;=\;
\underbrace{(H^\top R^{-1} H + \Sigma^{-1})^{-1} H^\top R^{-1}}_{\text{invert in state space }(d_x\times d_x)}.
\tag{03.17}
$$

Both sides are the Kalman gain $K$; the left is "covariance/innovation" thinking, the right is "information/normal-equation" thinking. For LiDAR-inertial odometry $d_x \ll d_z$, so the right side is dramatically cheaper — that is the order-of-magnitude saving FAST-LIO advertises (the paper's Table II measures the old formula at 1621 ms vs. the new at 1.16 ms for 1802 feature points). We derive (03.17) carefully and connect it to the iterated update in §09; here the point is that the *same Schur/Woodbury identity* underlies marginalization (03.15), batch elimination (03.16), and the recursive gain (03.17). It is one piece of algebra wearing three hats.

> **Meridian grounding.** Both forms are literally coded side-by-side in `FAST_LIO/include/IKFoM_toolkit/esekfom/esekfom.hpp`, and the toolkit *picks the cheaper one at runtime* by comparing the state dimension `n` to the measurement dimension `dof_Measurement`: when `n > dof_Measurement` it uses the measurement-space (innovation) form `K_ = P_ * h_x_.transpose() * (h_x_ * P_ * h_x_.transpose() + h_v_ * R * h_v_.transpose()).inverse()`, and otherwise — the LiDAR case, thousands of points so `n < dof_Measurement` — it uses the state-space form `K_ = (h_x_.transpose() * R_in * h_x_ + P_.inverse()).inverse() * h_x_.transpose() * R_in`, building the state-dimensioned $H^\top R^{-1} H + P^{-1}$ exactly as in paper eq (20) (`esekfom.hpp`, the `if(n > dof_Measurement)…else…` branch around lines 1081-1107). That is why FAST-LIO stays real-time even with thousands of point-to-plane residuals per scan. Point-LIO (`Point-LIO/src/Estimator.cpp`) and FAST-LIVO2 (`FAST-LIVO2/src/`) reuse the same `esekfom`-style machinery; FAST-LIVO2 in fact goes further and applies the LiDAR and camera updates *sequentially* (its paper 2408.14035 eqs (5)-(8) factor the joint posterior $p(x\mid y_l,y_c)$ into two Bayesian updates — first fuse LiDAR to get $p(x\mid y_l)$, then fuse the camera), which is just the §03.2 product-of-factors statement applied one factor-block at a time: stacking the *photometric* residual's information $H_{\text{photo}}^\top\Omega_{\text{photo}}H_{\text{photo}}$ onto $\Lambda$ (§03.7). Tightly-coupled fusion, viewed through this section, is nothing more than *summing information blocks into one $\Lambda$ before inverting once* (or, equivalently, conditioning on them one after another).

#### 03.8.3 The recursive view: filtering = marginalize all but the present

Marginalization closes the loop with the recursive estimators of §09. A Kalman/EKF filter keeps only the *current* state and a Gaussian summary $(\mu, \Sigma)$ of it. Where did the past go? It was **marginalized**: at each step the filter integrates out the previous state, leaving its information in the prior on the current state. Equation (03.15) applied repeatedly *is* the prediction step; the measurement update is the information addition of §03.7. Thus a filter is a factor graph in which every variable except the newest has already been marginalized away — and **fixed-lag smoothing** is the in-between, marginalizing only states that fall off the tail of a sliding window of length $L$.

This gives the clean hierarchy Meridian's architecture rests on, all instances of minimizing the *same* (03.10):

- **Full batch / iSAM2 back-end (L3):** marginalize nothing (or lazily); re-optimize the whole keyframe graph for global consistency. Best accuracy, used for loop closure and drift correction.
- **Fixed-lag front-end (L2):** keep a window of recent states, marginalize older ones into a prior. Bounded cost, real-time. Meridian's CT B-spline window (§10) and the iEKF v1 both live here.
- **Filter (degenerate fixed-lag, $L=1$):** marginalize everything but the present. The leanest real-time form; FAST-LIO's `esekfom` update is exactly this, with the iteration of §03.4 layered on top (the "iterated" in iterated-EKF) so that a *single* recursive step still solves the *nonlinear* least-squares (03.10) to convergence — the formal MAP $\leftrightarrow$ IKF correspondence the FAST-LIO paper (2010.08196) is built on. The paper makes the identification explicit: combining the IMU-propagation prior (its eq (15), $x \boxminus \hat x \sim \mathcal{N}(0, \hat P)$) with the linearized LiDAR likelihood (eq (14)) "yields the maximum a-posteriori estimate (MAP)" $\min_{\delta x}\big(\lVert x \boxminus \hat x\rVert^2_{\hat P^{-1}} + \sum_{j=1}^m \lVert z_j + H_j\,\delta x\rVert^2_{R_j^{-1}}\big)$ (eq (17)) — which is our (03.10) with one prior factor (the marginalized past) plus $m$ LiDAR factors — and then "optimizing the resultant quadratic cost leads to the standard iterated Kalman filter." That sentence is the entire thesis of this section restated by the source: the filter does not replace MAP/NLS; it *is* MAP/NLS over a one-state window.

---

### 03.9 Synthesis of this section, and forward pointers

We started from the posterior $p(X\mid Z)$ and, with two assumptions — conditional independence of measurements and Gaussian noise on the manifold (§02) — derived that the MAP estimate is the solution of a **weighted nonlinear least-squares** problem,

$$
X^\star = \arg\min_X \tfrac12 \sum_k \lVert z_k \boxminus h_k(X_k)\rVert^2_{\Sigma_k}.
$$

We saw that this objective is a **factor graph** (bipartite: variables × factors), that its locality makes the system information matrix $\Lambda = H^\top\Omega H$ **sparse** with the graph's adjacency, that the same belief admits **covariance** and **information** forms with complementary strengths, and that **marginalization via the Schur complement** is the one operation that lets us delete variables losslessly — the operation that, worn as three hats, is *batch elimination*, the *FAST-LIO Kalman-gain trick*, and the *prediction step of a filter*. Each technical claim is realized in the on-disk reference systems: the state and $\boxplus$ in `use-ikfom.hpp`, the residual/Jacobian assembly in `laserMapping.cpp`'s `h_share_model`, and the information-form iterated update in `esekfom.hpp`, with Point-LIO and FAST-LIVO2 reusing the same machinery and merely stacking more factor blocks.

What this section deliberately *did not* do, and where to find it:

- **The actual residuals $r_k$ and Jacobians $H_k$** for each sensor — IMU/preintegration (§04), LiDAR point-to-plane/line (§05), sparse-direct photometric with LiDAR-provided depth (§06), GNSS/ENU absolute constraints (§07). This section gave only the generic $r_k = z_k \boxminus h_k(X_k)$ and $H_k = \partial r_k/\partial\delta x$.
- **How (03.13) is actually solved.** Gauss–Newton, Levenberg–Marquardt, ordering, and sparse Cholesky in §08; the iterated EKF / ESIKF and its formal equivalence to GN-MAP, plus fixed-lag smoothing, in §09.
- **Continuous time.** Replacing the discrete states $x_i$ with B-spline control points $c_k$ so that one residual can be evaluated at *any* timestamp — the same MAP=NLS skeleton, with $T(t)$ in place of $x_i$ — in §10.
- **When the Gaussian/least-squares model breaks.** Degeneracy and observability (the spectrum of $\Lambda$), robust kernels (replacing $\lVert r\rVert^2$ by $\rho(\lVert r\rVert)$), GNC, switchable constraints, and PCM, all of which re-shape the weights $\Omega_k$ of (03.10) — in §11.
- **The full assembled step**, mapped line-by-line onto FAST-LIO2 / FAST-LIVO2, in §12.

Keep (03.10) in view through all of them. Every later section is a way of writing down, weighting, linearizing, solving, or robustifying that one sum of squared Mahalanobis residuals.
