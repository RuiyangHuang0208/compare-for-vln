import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory("omnivla_adapter"), "config", "omnivla.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("goal_profile", default_value="language_only"),
            DeclareLaunchArgument("allow_stub_server", default_value="false"),
            DeclareLaunchArgument("evaluation_mode", default_value="trajectory_normalized"),
            DeclareLaunchArgument("direct_velocity", default_value="false"),
            Node(
                package="omnivla_adapter",
                executable="omnivla_adapter",
                parameters=[
                    config,
                    {
                        "goal.profile": LaunchConfiguration("goal_profile"),
                        "runtime.allow_stub_server": ParameterValue(
                            LaunchConfiguration("allow_stub_server"), value_type=bool
                        ),
                        "evaluation.mode": LaunchConfiguration("evaluation_mode"),
                        "evaluation.direct_velocity": ParameterValue(
                            LaunchConfiguration("direct_velocity"), value_type=bool
                        ),
                    },
                ],
                output="screen",
            ),
        ]
    )
