from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config = PathJoinSubstitution([FindPackageShare("navigation_bridge"), "config", "navigation.yaml"])
    return LaunchDescription([
        Node(package="navigation_bridge", executable="navigation_bridge", parameters=[config], output="screen")
    ])
