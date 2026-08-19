# Navigation Bridge

`navigation_bridge` 是统一 VLN 命令与固定 B2W locomotion 执行器之间的唯一控制层。

```text
/vln/command
    -> path_follower
    -> safety_filter
    -> /nav_vel
    -> robot_controller
```

行为约定：

- `STOP` 立即输出零速度。
- `WAYPOINT` 和 `TRAJECTORY` 由 `path_follower` 根据 odometry 转为速度。
- `VELOCITY` 跳过路径跟随，但仍必须经过超时检查和速度限幅。
- 命令超时、NaN、Inf 或类型错误都必须输出零速度。
- 输出使用 `geometry_msgs/Twist`，映射为 `[linear.x, linear.y, angular.z]`。

当前仿真闭环直接调用 `path_follower.py` 和 `safety_filter.py`，不经过 ROS 2。DualVLN 默认期望速度为 `0.3 m/s`，运行时可通过 `speed MPS` 在 `0.05-5.0 m/s` 内调整；横移为零，角速度上限为 `0.4 rad/s`。B2W policy 的原始训练范围只到 `1.0 m/s`，超过该值不保证稳定。ROS 2 话题接口仍是后续真机边界，本阶段不运行。
