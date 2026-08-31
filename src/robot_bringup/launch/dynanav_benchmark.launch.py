from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def launch_benchmark(context):
    command = [
        "ros2",
        "run",
        "vln_evaluation",
        "benchmark_runner",
        "--model",
        LaunchConfiguration("model").perform(context),
        "--experiment",
        LaunchConfiguration("experiment").perform(context),
        "--episodes",
        LaunchConfiguration("episodes").perform(context),
    ]
    if LaunchConfiguration("headless").perform(context).lower() in {"1", "true", "yes"}:
        command.append("--headless")
    if LaunchConfiguration("dry_run").perform(context).lower() in {"1", "true", "yes"}:
        command.append("--dry-run")
    if LaunchConfiguration("resume").perform(context).lower() in {"1", "true", "yes"}:
        command.append("--resume")
    if LaunchConfiguration("continue_on_error").perform(context).lower() in {"1", "true", "yes"}:
        command.append("--continue-on-error")
    execution_profile = LaunchConfiguration("execution_profile").perform(context)
    if execution_profile != "fair":
        command.extend(["--execution-profile", execution_profile])
    command.extend(
        [
            "--max-attempts",
            LaunchConfiguration("max_attempts").perform(context),
            "--timeout-scale",
            LaunchConfiguration("timeout_scale").perform(context),
            "--startup-grace",
            LaunchConfiguration("startup_grace").perform(context),
        ]
    )
    return [ExecuteProcess(cmd=command, output="screen")]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("model", default_value="dummy"),
            DeclareLaunchArgument("experiment", default_value="benchmark"),
            DeclareLaunchArgument("episodes", default_value="hospital_001"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument("resume", default_value="false"),
            DeclareLaunchArgument("continue_on_error", default_value="false"),
            DeclareLaunchArgument("execution_profile", default_value="fair"),
            DeclareLaunchArgument("max_attempts", default_value="1"),
            DeclareLaunchArgument("timeout_scale", default_value="2.5"),
            DeclareLaunchArgument("startup_grace", default_value="120.0"),
            OpaqueFunction(function=launch_benchmark),
        ]
    )
