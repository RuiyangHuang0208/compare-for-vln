import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    workspace_root = LaunchConfiguration("workspace_root")
    mode = LaunchConfiguration("mode")
    script = PathJoinSubstitution(
        [workspace_root, "src", "robot_controller", "scripts", "run_b2w_hospital.sh"]
    )
    navigation_config = os.path.join(get_package_share_directory("navigation_bridge"), "config", "navigation.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "workspace_root",
                default_value=EnvironmentVariable("ROBOT_VLN_WS", default_value=os.getcwd()),
                description="robot_vln_ws root; set ROBOT_VLN_WS when launching outside the workspace",
            ),
            DeclareLaunchArgument("mode", default_value="straight"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("turn_radius", default_value="1.0"),
            DeclareLaunchArgument("turn_degrees", default_value="45.0"),
            Node(
                package="robot_controller",
                executable="udp_velocity_bridge",
                parameters=[{"command_source": "vln"}],
                output="screen",
            ),
            Node(
                package="navigation_bridge",
                executable="navigation_bridge",
                parameters=[navigation_config],
                output="screen",
            ),
            Node(
                package="dummy_adapter",
                executable="dummy_adapter",
                parameters=[
                    {
                        "mode": mode,
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
            ExecuteProcess(
                cmd=[script, "ros-vln"],
                condition=UnlessCondition(LaunchConfiguration("headless")),
                output="screen",
            ),
            ExecuteProcess(
                cmd=[script, "ros-vln", "--headless"],
                condition=IfCondition(LaunchConfiguration("headless")),
                output="screen",
            ),
        ]
    )
