from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="straight"),
        DeclareLaunchArgument("wait_for_episode_start", default_value="false"),
        DeclareLaunchArgument("turn_radius", default_value="1.0"),
        DeclareLaunchArgument("turn_degrees", default_value="45.0"),
        Node(
            package="dummy_adapter",
            executable="dummy_adapter",
            parameters=[
                {
                    "mode": LaunchConfiguration("mode"),
                    "wait_for_episode_start": ParameterValue(
                        LaunchConfiguration("wait_for_episode_start"), value_type=bool
                    ),
                    "trajectory.turn_radius": ParameterValue(
                        LaunchConfiguration("turn_radius"), value_type=float
                    ),
                    "trajectory.turn_degrees": ParameterValue(
                        LaunchConfiguration("turn_degrees"), value_type=float
                    ),
                }
            ],
            output="screen",
        ),
    ])
