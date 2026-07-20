from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _fail_if_process_failed(process_name):
    def _handle_exit(event, _context):
        if event.returncode != 0:
            raise RuntimeError(
                f"{process_name} failed with exit code {event.returncode}"
            )
        return None

    return _handle_exit


def generate_launch_description() -> LaunchDescription:
    config = LaunchConfiguration("config")
    rrd = LaunchConfiguration("rrd")

    ingress = Node(
        package="meridian_apps",
        executable="meridian_ingress",
        name="meridian_ingress",
        output="screen",
        parameters=[config],
        arguments=["--rrd", rrd],
    )
    ingress_failure = RegisterEventHandler(
        OnProcessExit(
            target_action=ingress,
            on_exit=_fail_if_process_failed("meridian_ingress"),
        )
    )

    # Compose forwards its configurable host port to this fixed listener.
    foxglove = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        parameters=[{"port": 8765, "address": "0.0.0.0"}],
    )
    foxglove_failure = RegisterEventHandler(
        OnProcessExit(
            target_action=foxglove,
            on_exit=_fail_if_process_failed("foxglove_bridge"),
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                description="ROS parameter YAML describing the input topics and message layout",
            ),
            DeclareLaunchArgument("rrd", default_value="/workspace/out/meridian_ingress.rrd"),
            ingress_failure,
            foxglove_failure,
            ingress,
            foxglove,
        ]
    )
