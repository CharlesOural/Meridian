# Grounding dossiers

Deep technical extractions from the SOTA reference implementations + papers, produced by reading the actual cloned code and paper text. These are the **evidence base** the course chapter (`../course/`) and specs (`../specs/`) are written from — when a spec makes a claim, the grounding for it lives here with `file:line` and paper eq/section citations.

Reference code (read-only, not committed; under `C:/Users/charl/Sources/slam-reference/`):
- **FAST_LIO** (FAST-LIO2) — `src/laserMapping.cpp`, `src/IMU_Processing.hpp`, `include/use-ikfom.hpp`, `include/IKFoM_toolkit/esekfom/esekfom.hpp`, `include/ikd-Tree/`
- **FAST-LIVO2** — `src/{vio,voxel_map,visual_point,frame,LIVMapper}.cpp`
- **Point-LIO** — `src/{Estimator,IMU_Processing,li_initialization}.{cpp,h}`
- Papers: FAST-LIO2 (arXiv 2107.06829), FAST-LIO (2010.08196), FAST-LIVO2 (2408.14035), ikd-Tree (2102.10808)

| Dossier | Topic | Status |
|---|---|---|
| `01_state_manifold_esikf.md` | State on S²×SO(3)×ℝⁿ, IKFoM, ⊞/⊟, error-state propagation, iterated update | ✅ solid, line-cited |
| `02_imu_propagation_deskew.md` | IMU model, init, forward propagation, backward-propagation de-skew | ✅ code line-cited (paper refs by memory) |
| `03_lidar_residual_ikdtree.md` | Point-to-plane residual + Jacobian, plane fit, ikd-Tree map mgmt | ✅ solid, line-cited |
| `04_livo2_visual.md` | Sparse-direct photometric residual, LiDAR depth, sequential ESIKF | ✅ rewritten from full code read, exact line numbers |
| `05_point_lio_contrast.md` | Point-wise update, IMU-as-measurement, CT relation, vs FAST-LIO2 | ✅ solid, line-cited |
| `06_engineering_ros_debug.md` | Code org, ROS integration, debug/viz patterns to emulate/improve | ✅ re-extracted from completed run, line-cited |

### Map / back-end / loop-closure grounding (paper-based, no local code)
| `07_mapping_tsdf_mesh.md` | NVBlox/Voxblox TSDF+ESDF, Marching Cubes, Poisson surface reconstruction | research dossier |
| `08_loop_closure_placerec.md` | Scan Context / Scan Context++ / STD / BTC place recognition + GICP | research dossier |
| `09_backend_isam2.md` | iSAM2 / GTSAM factor graph, Bayes tree, marginalization, robust kernels | research dossier |
| `10_continuous_time.md` | B-spline SE(3) continuous-time LIO (CLINS / Coco-LIC) for the CT front-end | research dossier |

> Provenance note: dossiers were recovered from a multi-agent grounding run during which the file-read tooling failed intermittently for some agents. 01/02/03/05 are substantively correct; 04 and 06 are flagged for regeneration now that the environment is stable. **Always confirm a `file:line` against the live file before quoting it in a spec.**
