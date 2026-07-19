# Local real-time implementation provenance

This package is an original Meridian implementation. The checked-in reference
repositories below were read to identify proven LiDAR registration behavior,
numerical conventions, and common failure modes. No GPL source was copied,
translated, or linked.

## Deterministic direct-ICP cloud preparation

`include/meridian/local/lidar_registration_cloud.hpp`,
`src/lidar_registration_cloud.cpp`, and their focused tests use algorithmic
context from:

- Meridian v1 tag `v1`, especially
  `src/meridian_frontend/src/lio/voxel_grid_map.{hpp,cpp}` and
  `scan_registration.{hpp,cpp}`;
- Meridian v1 commits `1cfc208` (parallel association) and `9d99b94`
  (`1e-3` convergence seed);
- `../slam-reference/kiss-icp` for robust direct-ICP and adaptive-threshold
  concepts; and
- `../slam-reference/FAST_LIO` and `../slam-reference/rko_lio` for bounded
  incremental spatial-search and LiDAR–inertial conditioning lessons.

Meridian independently keeps one immutable, deterministic 0.30 m
registration-only scan artifact; registration then selects source rows on a
1.50 m voxel grid. Source indices remain intact, but the provisional tracking
deskew is discarded after the selected rows and exact index are sealed. An
accepted localization instead emits an `AcceptedLidarMapInput` that shares the
immutable raw sweep. A downstream map worker explicitly re-deskews that raw
sweep from accepted/final localization and calls `LidarMapPayload::seal()` for
full row validation and hashing. Consequently dense-payload work is not part
of the local-estimator critical path. No Gaussian covariance or normal
estimation is part of the active local registration path.

## Pose-aware direct point-to-point ICP

`include/meridian/local/lidar_registration.hpp`, `src/lidar_registration.cpp`,
`src/direct_lidar_factor.{hpp,cpp}`, `src/rolling_lidar_target.cpp`, and their
focused tests use algorithmic context from:

- the Meridian v1 files and commits above for its open-addressed, power-of-two,
  linear-probe voxel index, fixed 27-cell exact search, bounded per-voxel
  support, indexed OpenMP association, and convergence lessons;
- `../slam-reference/kiss-icp` for robust direct-registration and
  adaptive-scale concepts;
- `../slam-reference/rko_lio` and `../slam-reference/FAST_LIO` for direct
  registration conditioned by IMU prediction and bounded local-map search; and
- GLIM for scan-local geometry with independently revisioned smoother poses.

The production path keeps one point cloud per accepted scan and one explicit
optimized pose per target record. The rolling window never merges multiple
live scans under one pose. Registration deterministically distributes its
configured target count across the complete retained live history, including
the newest target. Sweep count and total retained points are independently
bounded by configuration.

Meridian independently adds exact target-pose ownership, exclusive source-row
assignment across target poses, an exact-radius deterministic point index,
adaptive MAD Huber weighting clamped to 0.15–0.50 m, normalized observability
projection, a physical information cap, and canonical checksum-bearing
snapshots. Each accepted proposal seals its point correspondence identities,
scalar row information, robust weights, and supported row space.
`lidarFactorInformation` is the single ROS/GTSAM-free
calculation used both by factor-batch admission and the private graph factor.

The graph factor is stateless: one binary factor owns one target/source pose
pair and evaluates the same frozen residual model in `error()` and
`linearize()`. Reassociation creates a new atomic factor-batch revision; it
does not mutate an optimizer-side cache. IMU propagation may provide deskew
and the source seed, but a returned LiDAR factor contains only LiDAR-derived
information.

## Deliberate one-path boundary

Historical surfel point-to-plane, GICP/VGICP, and mutable per-linearization
implementations were removed after pose-aware direct point-to-point ICP became
the integrated frontend and graph-factor path.
Keeping those alternatives compiled would create divergent covariance,
association, observability, and ownership semantics. Comparative registration
experiments belong in benchmark branches or dedicated research tools, not in
the production local estimator library.

Candidate-isolated iSAM2 remains because graph transactions still use it to
guarantee candidate/committed solver separation. Current point-to-point factors
own no mutable association cache, but the graph-level isolation mechanism is
broader than LiDAR and is therefore not legacy registration code.

No ATE, realtime, Jetson, or SOTA-performance claim follows from this source
mapping alone; those claims require the repository's full-sequence benchmark
and timing gates.
