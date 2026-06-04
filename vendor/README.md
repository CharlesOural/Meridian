# vendor/ — pinned upstream submodules

These three upstreams are vendored as git submodules (see `../.gitmodules`) and
consumed in-tree via `MeridianVendored.cmake` (which wraps each as a
`meridian::vendor_*` CMake target). They are pinned by submodule SHA, not fetched
at build time, so a clean checkout is reproducible and offline-buildable.

| Submodule | Used by | Role |
|---|---|---|
| `basalt-headers` | `meridian_frontend` | continuous-time SE(3) B-spline kernel (analytic Jacobians) |
| `ikd-Tree` | `meridian_frontend` / `meridian_map` | incremental k-d tree — registration correctness oracle |
| `scancontext` | `meridian_place` | rotation-invariant loop-closure descriptor |

Fetch them with:

```bash
git submodule update --init --recursive
```