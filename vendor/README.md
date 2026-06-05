# vendor/ — pinned upstream submodules

> `COLCON_IGNORE` keeps colcon out of this tree: the upstreams ship their own
> CMake projects (basalt-headers even pins its bundled Eigen), but they are
> consumed as header sources through `MeridianVendored.cmake`, never built
> standalone.

These three upstreams are vendored as git submodules (see `../.gitmodules`) and
consumed in-tree via `MeridianVendored.cmake` (which wraps each as a
`meridian::vendor_*` CMake target).

| Submodule | Used by | Role |
|---|---|---|
| `basalt-headers` | `meridian_frontend` | continuous-time SE(3) B-spline kernel (analytic Jacobians) |
| `ikd-Tree` | `meridian_frontend` / `meridian_map` | incremental k-d tree — registration correctness oracle |
| `scancontext` | `meridian_place` | rotation-invariant loop-closure descriptor |

Fetch them with:

```bash
git submodule update --init
```