from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("episode", default_value="hospital_001"),
            Node(package="dynanav_bridge", executable="sensor_bridge", output="screen"),
            Node(
                package="dynanav_bridge",
                executable="episode_manager",
                parameters=[{"episode_id": LaunchConfiguration("episode")}],
                output="screen",
            ),
        ]
    )
