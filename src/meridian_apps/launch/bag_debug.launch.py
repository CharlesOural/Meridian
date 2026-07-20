"""Play any ROS 2 bag through Meridian's generic ingress and Rerun recorder."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _stop_after_player(event, _context):
    if event.returncode != 0:
        raise RuntimeError(f"ros2 bag play failed with exit code {event.returncode}")
    reason = f"ros2 bag play exited with code {event.returncode}"
    # Give executor callbacks a short grace period before shutdown. The ingress
    # node drains its own bounded decode and Rerun queues during shutdown.
    return [TimerAction(period=1.0, actions=[EmitEvent(event=Shutdown(reason=reason))])]


def generate_launch_description() -> LaunchDescription:
    bag = LaunchConfiguration("bag")
    config = LaunchConfiguration("config")
    rate = LaunchConfiguration("rate")
    rrd = LaunchConfiguration("rrd")

    debug_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("meridian_apps"), "launch", "input_debug.launch.py"]
            )
        ),
        launch_arguments={"config": config, "rrd": rrd}.items(),
    )

    player = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            bag,
            "--clock",
            "100.0",
            "--rate",
            rate,
            # rosbag2 creates its publishers before this delay. It gives DDS
            # discovery time to match without adding delivery policy to ingress.
            "--delay",
            "1.0",
            "--disable-keyboard-controls",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("bag", description="Path to a rosbag2 directory"),
            DeclareLaunchArgument(
                "config", description="ROS parameter YAML for Meridian ingress"
            ),
            DeclareLaunchArgument("rate", default_value="1.0"),
            DeclareLaunchArgument(
                "rrd", default_value="/workspace/out/meridian_ingress.rrd"
            ),
            debug_stack,
            # This is startup sequencing, not a delivery or count guarantee.
            TimerAction(period=1.0, actions=[player]),
            RegisterEventHandler(
                OnProcessExit(target_action=player, on_exit=_stop_after_player)
            ),
        ]
    )
