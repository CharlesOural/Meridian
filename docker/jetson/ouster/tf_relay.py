#!/usr/bin/env python3
"""Re-publish latched /tf_static transforms onto /tf at a fixed rate.

foxglove_bridge forwards a latched (TRANSIENT_LOCAL) message only to the clients
subscribed at the instant it arrives; a client connecting later never sees the
one-shot /tf_static and ends up with an empty transform tree. Mirroring those
transforms onto /tf on a timer makes them continuously available, so any viewer
connecting at any time gets the frames. Viz aid for raw-driver preview — the real
pipeline publishes its own TF.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from tf2_msgs.msg import TFMessage


class TfStaticRelay(Node):
    def __init__(self):
        super().__init__("tf_static_relay")
        latched = QoSProfile(
            depth=100,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._transforms = {}
        self.create_subscription(TFMessage, "/tf_static", self._on_static, latched)
        self._pub = self.create_publisher(TFMessage, "/tf", 20)
        self.create_timer(0.05, self._tick)

    def _on_static(self, msg):
        for t in msg.transforms:
            self._transforms[(t.header.frame_id, t.child_frame_id)] = t

    def _tick(self):
        if not self._transforms:
            return
        out = TFMessage()
        now = self.get_clock().now().to_msg()
        for t in self._transforms.values():
            t.header.stamp = now
            out.transforms.append(t)
        self._pub.publish(out)


def main():
    rclpy.init()
    node = TfStaticRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
