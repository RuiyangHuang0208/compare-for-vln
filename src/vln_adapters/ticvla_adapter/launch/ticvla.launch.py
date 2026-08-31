import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory("ticvla_adapter"), "config", "ticvla.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("evaluation_mode", default_value="trajectory_normalized"),
        DeclareLaunchArgument("direct_velocity", default_value="false"),
        DeclareLaunchArgument("model_image_aspect_ratio", default_value="0.0"),
        Node(
            package="ticvla_adapter",
            executable="ticvla_adapter",
            parameters=[
                config,
                {
                    "evaluation.mode": LaunchConfiguration("evaluation_mode"),
                    "evaluation.direct_velocity": ParameterValue(
                        LaunchConfiguration("direct_velocity"), value_type=bool
                    ),
                    "input.model_image_aspect_ratio": ParameterValue(
                        LaunchConfiguration("model_image_aspect_ratio"), value_type=float
                    ),
                },
            ],
            output="screen",
        ),
    ])
