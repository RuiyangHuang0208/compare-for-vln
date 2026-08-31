import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory("mobilevla_r1_adapter"), "config", "mobilevla_r1.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("allow_stub_server", default_value="false"),
            DeclareLaunchArgument("command_duration_s", default_value="0.0"),
            DeclareLaunchArgument("velocity_units_confirmed", default_value="false"),
            DeclareLaunchArgument("coordinate_signs_confirmed", default_value="false"),
            Node(
                package="mobilevla_r1_adapter",
                executable="mobilevla_r1_adapter",
                parameters=[
                    config,
                    {
                        "runtime.allow_stub_server": ParameterValue(
                            LaunchConfiguration("allow_stub_server"), value_type=bool
                        ),
                        "control.command_duration_s": ParameterValue(
                            LaunchConfiguration("command_duration_s"), value_type=float
                        ),
                        "control.velocity_units_confirmed": ParameterValue(
                            LaunchConfiguration("velocity_units_confirmed"), value_type=bool
                        ),
                        "control.coordinate_signs_confirmed": ParameterValue(
                            LaunchConfiguration("coordinate_signs_confirmed"), value_type=bool
                        ),
                    },
                ],
                output="screen",
            ),
        ]
    )

