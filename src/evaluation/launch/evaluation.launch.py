from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("model", default_value="dummy"),
            DeclareLaunchArgument("experiment", default_value="single_episode"),
            DeclareLaunchArgument("output_root", default_value="outputs"),
            Node(package="vln_evaluation", executable="goal_monitor", output="screen"),
            Node(
                package="vln_evaluation",
                executable="evaluator",
                parameters=[
                    {
                        "model_name": LaunchConfiguration("model"),
                        "experiment_name": LaunchConfiguration("experiment"),
                        "output_root": LaunchConfiguration("output_root"),
                    }
                ],
                output="screen",
            ),
        ]
    )
