# meridian_local_rt

`meridian_local_rt` is Meridian's ROS-free real-time estimator. It owns the
discrete IMU state backbone, independent modality queues, bounded frontend
proposals, and transactional local factor-graph updates. Public headers expose
neither ROS nor GTSAM types.

## One production LiDAR path

The package deliberately carries one LiDAR registration implementation:

1. midpoint IMU propagation provides the discrete deskew trajectory and pose
   seed without contributing IMU information to the returned LiDAR factor;
2. `buildLidarRegistrationCloud` retains an immutable 0.30 m deskewed scan
   artifact and constructs its exact target index;
3. `RollingLidarTargetBuilder` retains at most 15 accepted scan-local clouds or
   300,000 points with separate optimized pose metadata—accepted scans are never
   merged under the newest pose;
4. `registerLidarScan` deterministically selects 1.50 m source rows and runs one
   bounded, reassociating, robust direct point-to-point ICP proposal against at
   most five explicit target poses using immutable open-addressed exact-neighbor
   indices and exclusive source-row ownership; and
5. immutable checksum-bearing snapshots become stateless binary point-to-point
   factors through an atomic `FactorBatch` transaction.

The complete tracking-deskewed return and layout remain attached to each
`LidarRegistrationCloud`; canonical dense mapping must re-deskew the raw sweep
at its accepted/finalized pose. Graph pose revisions update only target pose
metadata; they do not copy scan geometry.
There is no alternate GICP, VGICP, surfel, point-to-plane, or mutable
registration factor path in this package.

Registration uses a 0.50 m Euclidean correspondence gate, bounded OpenMP
association, adaptive MAD Huber scale clamped to 0.15–0.50 m, separate natural
translation/rotation convergence at `1e-3`, and a 100-iteration fault ceiling.
GN/LM trials must decrease the frozen objective. The 5 s / 64-state local graph
lag and the rolling target's 15-sweep / 300,000-point geometry horizon are
separate bounds; neither defines the other.

## Research context

The implementation was independently derived after studying:

- Meridian's v1 direct point-to-point lineage (tag `v1`, plus later commits
  `1cfc208` and `9d99b94`) for its bounded voxel index, parallel association,
  and practical convergence lessons;
- KISS-ICP's robust direct-ICP and adaptive-threshold concepts (MIT), RKO-LIO's
  simple direct LiDAR–inertial registration lessons, and FAST-LIO's bounded
  incremental-map search patterns; no filter-state architecture is copied;
- GLIM's IMU integration, cloud deskew, fixed-lag ownership, and pose-local
  scan storage (BSD-3-Clause);
- gtsam_points' fixed-lag ordering and leaf marginalization mechanics (MIT);
- GTSAM's combined IMU preintegration, iSAM2, and marginal covariance APIs
  (BSD-3-Clause); and
- OKVIS2-X camera/frontend code for anchored inverse-range visual geometry
  and keyframe lifecycle context (BSD-3-Clause).

Meridian adds deterministic bounded work, signed-nanosecond support ownership,
explicit target/source pose dependencies, normalized directional-observability
projection, a physical information cap, robust-weight sealing, provenance,
and transactional rejection. Reference source code is not copied into this
package. See `REFERENCE_PROVENANCE.md` for the implementation-level mapping.

`LocalEstimator` is the composition API. `DynamicOnly` is the unsupervised IMU
initialization default. Static initialization requires an explicit,
epoch-matched `ZeroMotionPrior`; IMU statistics may validate that prior but do
not invent it. A rejected LiDAR batch leaves both graph and rolling-map state
unchanged and permits an explicitly reported IMU-only transition.
