"""Launches the Meridian odometry node (and optionally rviz) with a config file.

    ros2 launch meridian_ros meridian.launch.py \
        config_file:=/path/to/config.yaml use_sim_time:=true rviz:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('meridian_ros')
    default_config = os.path.join(share, 'config', 'newer-college-quad.yaml')
    default_rviz = os.path.join(share, 'rviz', 'meridian.rviz')

    config_file = LaunchConfiguration('config_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz = LaunchConfiguration('rviz')

    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=default_config),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        Node(
            package='meridian_ros',
            executable='odometry_node',
            name='meridian_odometry',
            output='screen',
            parameters=[{
                'config_file': config_file,
                'use_sim_time': use_sim_time,
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', default_rviz],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(rviz),
        ),
    ])
