# 导航桥

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

ROS 2 评测入口由 `navigation_bridge_node.py` 实现。公平模式下 TRAJECTORY 与 WAYPOINT
都先转换为世界路径并使用同一个 Pure Pursuit；原生模式由模型 adapter 输出官方高层
速度（有官方连续控制器时）或使用离散动作路径适配。VELOCITY 只经过限速、加减速度平滑和 0.5 秒 command
timeout；STOP 清除全部状态并立即发布零速度。轨迹有独立 2.0 秒 timeout，模型 inference
期间不会错误地按 velocity timeout 处理。所有参数位于 `config/navigation.yaml`，公共镜像
位于工作区 `configs/navigation.yaml`。

对于 NaVILA、Uni-NaVid 等离散转向 adapter，`Pose2D.theta` 会随圆弧轨迹显式给出。共享
follower 在达到 XY 终点后继续校正最终航向，航向误差小于 5 度才发布轨迹完成。未设置非零
`theta` 的 DualVLN/TIC-VLA 连续 XY 轨迹保持原有行为。
离散转向进入终点 `0.10 m` 捕获区后线速度归零，以免 B2-W 围绕很短的圆弧终点打圈。

原生控制器由 `path_follower.controller` 选择：

- `shared_pure_pursuit`：公平轨道；
- TIC-VLA 原生模式在 `ticvla_adapter` 内按官方 Spot 曲率公式直接发布速度；
- DualVLN 官方 MPC 依赖 CasADi/IPOPT。当前未安装该依赖，因此使用共享
  Pure Pursuit，并明确记录为 B2-W adapter，不再用自写求解器冒充官方 MPC；
- `discrete_action_path`：NaVILA/Uni-NaVid 官方离散动作语义；
- OmniVLA 原生 PD 在 adapter 内输出 VELOCITY，因此跳过路径跟踪器。

所有模式仍经过统一 STOP、超时、速度/加速度限幅，并输出 `/nav_vel` 给 SRU-ONNX。

滚动短轨迹的终点阈值为 `0.05 m`。该阈值只决定当前短轨迹是否执行完，不代替独立
`goal_monitor` 的 `0.5 m` episode 成功判定。

当前明确使用 SRU ONNX 控制器。公平模式默认期望速度为 `1.0 m/s`；原生模式从
`configs/models.yaml` 读取模型参数，并强制不超过已验证的 SRU-ONNX
`|vx|,|vy|,|wz| <= 1.0` 输入范围。手动启动的 `2.0 m/s` 仅是历史实验上限，
不属于可审计 benchmark 参数。

启动时可通过 `desired_speed:=1.0` 设置，运行中通过 ROS 2 修改：

```bash
ros2 topic pub --once /navigation/desired_speed std_msgs/msg/Float32 "{data: 2.0}"
ros2 topic echo /navigation/desired_speed/current --once
```

节点拒绝非有限值、零、负数以及高于 `2.0 m/s` 的设置。该速度只作用于 WAYPOINT 和
TRAJECTORY 的共享 path follower；模型直接输出的 VELOCITY 仍只经过自身限速路径。
