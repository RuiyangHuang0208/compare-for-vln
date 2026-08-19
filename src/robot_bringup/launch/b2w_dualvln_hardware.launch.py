"""Start only the upstream B2 nodes required by the DualVLN control loop."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    robot_namespace = LaunchConfiguration("robot_namespace")
    platform_config = PathJoinSubstitution(
        [FindPackageShare("b2_platform"), "config", "b2_platform.yaml"]
    )
    mux_config = PathJoinSubstitution(
        [FindPackageShare("b2_control"), "config", "twist_mux.yaml"]
    )
    tf_remaps = [("/tf", "tf"), ("/tf_static", "tf_static")]

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_namespace", default_value="b2_366"),
            SetEnvironmentVariable("B2_NS", robot_namespace),
            SetEnvironmentVariable("ROS_DOMAIN_ID", "10"),
            SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
            Node(
                package="b2_platform",
                executable="b2_video",
                name="b2_front_video",
                namespace=robot_namespace,
                parameters=[platform_config],
                remappings=tf_remaps,
                output="screen",
            ),
            Node(
                package="b2_platform",
                executable="b2_statepublisher",
                name="b2_statepublisher",
                namespace=robot_namespace,
                parameters=[platform_config],
                remappings=tf_remaps,
                output="screen",
            ),
            Node(
                package="b2_platform",
                executable="b2_highroscontrol",
                name="b2_hardware",
                namespace=robot_namespace,
                parameters=[platform_config],
                remappings=tf_remaps,
                output="screen",
            ),
            Node(
                package="twist_mux",
                executable="twist_mux",
                name="twist_mux_node",
                namespace=robot_namespace,
                parameters=[mux_config],
                remappings=[
                    ("/diagnostics", "diagnostics"),
                    ("/tf", "tf"),
                    ("/tf_static", "tf_static"),
                    ("cmd_vel_out", "hardware/cmd_vel"),
                ],
                output="screen",
            ),
        ]
    )
