#!/usr/bin/env python3
"""Record the Meridian odometry track to a TUM-format trajectory file.

Subscribes to a configurable nav_msgs/Odometry topic and appends one TUM line
per message:

    timestamp tx ty tz qx qy qz qw

with the timestamp in seconds (the header stamp, sim-time when the node runs with
use_sim_time). The file is flushed on every write and again on SIGINT, so a track
survives a Ctrl-C at the end of a bag.

Usage (inside the dev container, after sourcing the workspace):
    python3 tools/record_tum.py OUT.tum [--topic /meridian/local/odometry]

Then play the bag as usual; Ctrl-C this script when the bag finishes. Compare
against the sequence's ground truth with the trajectory evaluator:
    python3 tools/trajectory_eval.py GT.txt OUT.tum \
        --reference-body-frame base_link --estimate-body-frame base_link
"""
import argparse
import sys

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class TumRecorder(Node):
    def __init__(self, out_path: str, topic: str):
        super().__init__("meridian_record_tum")
        self._file = open(out_path, "w")
        self._count = 0
        # A best-effort reader is compatible with either reliable or best-effort
        # odometry publishers. Evaluation still audits received timestamps/counts.
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self._sub = self.create_subscription(Odometry, topic, self._on_odom, qos)
        self.get_logger().info(f"recording {topic} -> {out_path}")

    def _on_odom(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._file.write(
            f"{t:.9f} {p.x:.6f} {p.y:.6f} {p.z:.6f} "
            f"{q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n"
        )
        self._file.flush()
        self._count += 1

    def close(self):
        self._file.flush()
        self._file.close()
        self.get_logger().info(f"wrote {self._count} poses")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", help="output TUM trajectory file")
    parser.add_argument("--topic", default="/meridian/local/odometry")
    args = parser.parse_args()

    rclpy.init()
    node = TumRecorder(args.out, args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
