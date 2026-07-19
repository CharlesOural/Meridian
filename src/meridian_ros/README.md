# meridian_ros

ROS 2 types stop at this package. The estimator packages consume only the
ROS-free records from `meridian_core`.

The initial PointCloud2/image conversion seam was re-derived from Meridian's
legacy `src/meridian_ros/src/conversions/ros2core.cpp` at commit
`f5ca513158c95aaf88223486ec481c1d42730a21`. The v2 implementation adds typed
failures, field-bound and payload validation, endian handling, half-open sweep
times, immutable context metadata, and regression tests; it does not copy the
legacy domain types or node runtime.

Algorithmic estimation code is intentionally forbidden here.

## Pipeline timing bridge

`meridian_core/pipeline_observability.hpp` defines the canonical ROS-free
runtime records:

- fixed-capacity stage, queue, and disposition text;
- steady-wall and Linux per-thread CPU duration;
- optional measurement, state, local-revision, and global-revision identity;
- queue count/bytes/capacity/oldest-age plus cumulative reject, drop, and skip
  counters;
- a preallocated rolling-window accumulator with deterministic nearest-rank
  p50/p95/p99/max snapshots.

`PipelineTimingPublisher` converts samples and rolling snapshots to
`diagnostic_msgs/msg/DiagnosticArray`. Its default topic is
`/meridian/local/pipeline_timing`; QoS is reliable, volatile, and bounded
KeepLast(32). The diagnostic key names carry unit suffixes such as `_ns`.
`DiagnosticArray` is only a dashboard/ROS transport view—the typed core record
is authoritative.

The publisher contains no stage selection or estimator logic. Composition
roots must wire timers and queue snapshots explicitly, and must account for
the publication/serialization stage separately so debug traffic cannot hide
its own deadline cost.
