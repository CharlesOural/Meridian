#!/usr/bin/env python3
"""Repair a converted ROS 2 bag whose LiDAR per-point time field carries mid-sweep
clock re-anchoring (e.g. FusionPortable's hardware-synced Ouster recordings, where
the column clock jumps by hundreds of milliseconds between columns that actually
fired ~50 us apart).

Columns are stored in firing order at a constant cadence, so the median adjacent
step is the true cadence: any step that is negative or far beyond it is replaced by
the median, rebuilding a contiguous zero-based timeline. A healthy scan passes
through byte-identical (offsets rebased to start at 0).

All topics are copied through unchanged except the LiDAR topic (repaired) and
topics whose message type the Humble typestore cannot register (e.g. dvs_msgs) —
those are skipped and reported. The output metadata is patched to the
Humble-compatible schema (version 5, string offered_qos_profiles).

Usage (inside the dev container, where `rosbags` is installed):
    python3 tools/repair_bag_column_clock.py SRC_BAG_DIR DST_BAG_DIR \
        [--lidar-topic /os_cloud_node/points] \
        [--topics /a /b ...] [--exclude /c /d ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

UINT32 = 6  # sensor_msgs/msg/PointField datatype constant


def repair_scan(msg) -> bool:
    """Rebuilds the per-point `t` field of one PointCloud2 in place.

    Returns True when at least one cadence outlier was repaired.
    """
    t_field = next((f for f in msg.fields if f.name == 't'), None)
    if t_field is None or t_field.datatype != UINT32:
        return False

    h, w, step = int(msg.height), int(msg.width), int(msg.point_step)
    if w < 2 or h < 1:
        return False
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, step).copy()
    off = int(t_field.offset)

    # Columns share their stamp across rows: row 0 defines the column timeline.
    t = arr[0, :, off:off + 4].copy().view('<u4').reshape(-1).astype(np.int64)
    d = np.diff(t)
    med = np.int64(np.median(d))
    cap = med * 16 + 1_000_000  # generous cadence bound [ns]
    bad = (d < 0) | (d > cap)
    if bad.any():
        d = d.copy()
        d[bad] = med
        rel = np.concatenate(([0], np.cumsum(d)))
        repaired = True
    else:
        rel = t - t[0]
        repaired = bool(t[0] != 0)

    new_t = rel.astype('<u4').view(np.uint8).reshape(1, w, 4)
    arr[:, :, off:off + 4] = np.broadcast_to(new_t, (h, w, 4))
    msg.data = arr.reshape(-1)
    return repaired


def make_writer(dst: Path) -> Writer:
    """Newer rosbags requires (and honours) an explicit metadata version; older ones
    accept no version argument and get the post-hoc metadata patch instead."""
    try:
        return Writer(dst, version=5)
    except TypeError:
        return Writer(dst)


def patch_metadata_for_humble(bag_dir: Path) -> None:
    meta = bag_dir / 'metadata.yaml'
    text = meta.read_text()
    text = text.replace('offered_qos_profiles: []', 'offered_qos_profiles: ""')
    text = text.replace('  version: 9', '  version: 5')
    text = text.replace('  version: 8', '  version: 5')
    meta.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    ap.add_argument('--lidar-topic', default='/os_cloud_node/points')
    ap.add_argument('--topics', nargs='*', default=None,
                    help='restrict output to exactly these topics '
                         '(plus the LiDAR topic); default is to keep everything')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='topics to drop from the output')
    args = ap.parse_args()

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    n_scans = n_repaired = n_copied = 0
    with Reader(args.src) as reader, make_writer(args.dst) as writer:
        conn_out = {}
        skipped = []
        for conn in reader.connections:
            if conn.topic in args.exclude:
                skipped.append((conn.topic, 'excluded'))
            elif args.topics is not None and conn.topic not in args.topics \
                    and conn.topic != args.lidar_topic:
                skipped.append((conn.topic, 'not in --topics'))
            elif conn.msgtype not in typestore.types:
                # Registering a connection writes its message definition into the
                # bag metadata, which needs the type in the typestore even though
                # pass-through topics are never deserialized.
                skipped.append((conn.topic, f'no typesupport for {conn.msgtype}'))
            else:
                conn_out[conn.id] = writer.add_connection(
                    conn.topic, conn.msgtype, typestore=typestore)
        for topic, reason in skipped:
            print(f'skipping {topic}: {reason}', file=sys.stderr)
        if not conn_out:
            print(f'error: no topics left to write from {args.src}', file=sys.stderr)
            return 1
        if not any(c.topic == args.lidar_topic for c in reader.connections):
            print(f'warning: LiDAR topic {args.lidar_topic} not in {args.src}; '
                  'copying without repair', file=sys.stderr)

        for conn, timestamp, data in reader.messages():
            if conn.id not in conn_out:
                continue
            if conn.topic == args.lidar_topic:
                msg = typestore.deserialize_cdr(data, conn.msgtype)
                n_scans += 1
                if repair_scan(msg):
                    n_repaired += 1
                data = typestore.serialize_cdr(msg, conn.msgtype)
            writer.write(conn_out[conn.id], timestamp, data)
            n_copied += 1

    patch_metadata_for_humble(args.dst)
    print(f'{args.dst}: {n_copied} messages, {n_scans} scans, '
          f'{n_repaired} scans repaired, {len(skipped)} topics skipped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
