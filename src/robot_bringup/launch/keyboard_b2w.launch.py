import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    workspace_root = LaunchConfiguration("workspace_root")
    script = PathJoinSubstitution(
        [workspace_root, "src", "robot_controller", "scripts", "run_b2w_hospital.sh"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "workspace_root",
                default_value=EnvironmentVariable("ROBOT_VLN_WS", default_value=os.getcwd()),
                description="robot_vln_ws root; set ROBOT_VLN_WS when launching outside the workspace",
            ),
            ExecuteProcess(
                cmd=[script, "keyboard"],
                output="screen",
            ),
        ]
    )
