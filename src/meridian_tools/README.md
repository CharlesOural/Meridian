# meridian_tools

`meridian_tools` owns deterministic, offline ROS 2 bag replay. It contains no
estimator algorithm and does not expose ROS messages to estimator code. Each
configured message is deserialized, converted through `meridian_ros`, audited
by a per-stream clock guard, then delivered as a move-only `DomainEvent`.

The replay profile is the complete run input: topic/wire type, source identity,
affine clock model, sensor ID, lidar time origin, camera exposure-midpoint
offset, calibration/config/session identities, and explicit event/message
limits. Bag-record time is retained both as `RecordedBagTimestamp` and
`ArrivalTime`; sensor header time remains the raw clock input.

## Integration

Create a `CalibrationBundle`, build the current NCD ROS 2 profile, and supply a
visitor. The visitor may continue, stop successfully, or return a typed error:

```cpp
NewerCollegeReplayOptions options;
options.bounds = {50'000, 500'000};
auto profile = makeNewerCollegeRos2ReplayProfile(calibration, options);

auto report = replayRos2Bag(
    "/data/newer-college/quad-easy", profile.value(),
    [](DomainEvent&& event) {
      // Move event into the application dispatcher, or inspect it here.
      return ReplayVisitResult::success(ReplayVisitAction::Continue);
    });
```

Generic profiles can configure `ImuReplaySource`, `LidarReplaySource`, raw or
compressed `CameraReplaySource`, and `GnssReplaySource`. NCD has no GNSS topic,
so its convenience factory deliberately does not invent one.

## Scheduled producer/consumer replay

`replayRos2BagScheduled()` preserves the synchronous reader/conversion path but
delivers valid `DomainEvent`s through one producer and one consumer. The
producer anchors the first recorded-arrival timestamp to steady wall time and
applies `recorded_delta / playback_rate`; `1.0` is real time. The consumer is
the only thread that invokes the caller's existing visitor.

The event queue has explicit count and logical-byte capacities. Logical bytes
come from `estimateDomainEventBytes()`: the immutable event object, metadata
text, and retained LiDAR/image payload. Queue overflow rejects one event and
terminates replay; it never silently drops the oldest item or skips accepted
work. Visitor stop/failure cancellation accounts any still-pending accepted
items as `skipped_policy`.

The scheduled result includes terminal and maximum queue count/bytes,
oldest-age, accept/reject/drop/skip counters, producer/consumer total
steady-wall plus Linux thread-CPU time, and bounded p50/p95/p99/max event-stage
snapshots. Both worker threads are stopped and joined before return. The
original `replayRos2Bag()` remains the deterministic no-scheduling API.

## Verification

The smoke test opens the existing Quad Easy ROS 2 bag and stops after 12 domain
events (with a 1,024-record hard read limit); it never copies or scans the full
bag:

```bash
colcon test --packages-select meridian_tools --event-handlers console_direct+
```

ROS 1 bags are intentionally outside this package. They can be converted and
profiled later when the extra benchmark variety becomes useful.
