# vendor/ — pinned upstream submodules

> `COLCON_IGNORE` keeps colcon out of this tree: the upstreams ship their own
> CMake projects, but they are consumed as header sources through
> `MeridianVendored.cmake`, never built standalone.

These upstreams are vendored as git submodules (see `../.gitmodules`) and
consumed in-tree via `MeridianVendored.cmake` (which wraps each as a
`meridian::vendor_*` CMake target).

| Submodule | Used by | Role |
|---|---|---|
| `scancontext` | `meridian_place` | rotation-invariant loop-closure descriptor |

Fetch them with:

```bash
git submodule update --init
```
