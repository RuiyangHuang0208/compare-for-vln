from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("profile", default_value="shared_pure_pursuit"),
            Node(
                package="vln_evaluation",
                executable="controller_probe",
                parameters=[{"profile": LaunchConfiguration("profile")}],
                output="screen",
            ),
        ]
    )
