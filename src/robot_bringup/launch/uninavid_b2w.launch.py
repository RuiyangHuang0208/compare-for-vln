from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    source = PythonLaunchDescriptionSource(
        get_package_share_directory("robot_bringup") + "/launch/dynanav_single_episode.launch.py"
    )
    return LaunchDescription([IncludeLaunchDescription(source, launch_arguments={"model": "uninavid"}.items())])
