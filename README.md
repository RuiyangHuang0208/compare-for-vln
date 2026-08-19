# Robot VLN Workspace

这个工作区用于把不同 VLN 模型接到同一台机器人上。目录骨架已经建立，并已迁入当前验证过的 B2W locomotion 运行资产。

```text
robot_vln_ws/
├── src/
│   ├── robot_bringup/       # 机器狗与整套系统启动
│   ├── robot_description/   # B2-W、Lite3、机械臂描述
│   ├── robot_controller/    # 最底层运动控制
│   ├── vln_interface/       # 统一不同 VLN 模型的输出
│   ├── vln_models/          # 每个 VLN 模型独立封装
│   ├── navigation_bridge/   # VLN 输出到机器人控制
│   ├── sensors/             # 相机、深度与雷达
│   └── evaluation/          # 统一测试场景与指标
├── third_party/             # RobotLab 和原始 VLN 开源仓库
├── checkpoints/             # 模型权重，按模型隔离
├── configs/                 # 跨模块公共配置
└── README.md
```

## 固定架构

```text
VLN/VLA model
      |
      | model-specific output
      v
vln_interface adapter
      |
      | /vln/command (NavigationCommand)
      v
navigation_bridge
      |
      | /nav_vel (Twist: vx, vy, yaw_rate)
      v
robot_controller
      |
      | leg + wheel actions
      v
Unitree B2W
```

模型与机器人控制必须解耦：

- VLN/VLA 模型不得导入 RobotLab、加载 B2W policy 或发送关节动作。
- 每个模型只增加薄 `ros_node.py`，模型输出差异由 `vln_interface` adapter 处理。
- 所有模型统一发布 `/vln/command`。
- `navigation_bridge` 是生成安全速度命令的唯一位置。
- `robot_controller` 永远只接收 `[vx, vy, yaw_rate]`，不识别模型类型。

统一消息支持：

```text
STOP
WAYPOINT
TRAJECTORY
VELOCITY
```

消息定义见 [src/vln_interface/msg/NavigationCommand.msg](src/vln_interface/msg/NavigationCommand.msg)，字段和 topic 约定见 [src/vln_interface/README.md](src/vln_interface/README.md)。

目录职责：

- `vln_models/<model>/` 只处理该模型的加载、推理和原始输出。
- `vln_interface/` 把 waypoint、trajectory 或 velocity 转成统一接口。
- `navigation_bridge/` 负责路径跟随和安全过滤，不放模型代码。
- `robot_controller/` 负责把统一速度命令接到 B2W locomotion policy。
- `third_party/robot_lab/` 保存完整 RobotLab 仓库及其 Git 历史。
- `third_party/unitree_ros/` 保存 Unitree 官方仓库，用于 B2W 几何和传感器 frame 参考。
- `third_party/InternNav/`、`third_party/NaVILA/` 等目录供后续放入原始 VLN 仓库。
- `checkpoints/b2w_locomotion/` 保存已经验证过的 B2W locomotion policy。

## 接入状态

| 模块 | 状态 |
|---|---|
| B2W RobotLab locomotion | READY |
| Hospital 键盘/目标点/移动验收 | READY |
| `NavigationCommand.msg` 契约 | READY |
| DualVLN 模拟 RGB-D/odometry 输入 | READY |
| DualVLN HTTP inference service | READY |
| trajectory adapter | READY |
| path follower + safety filter | READY |
| `[vx, vy, yaw_rate]` 到 B2W policy bridge | READY |
| DualVLN Hospital 40 秒闭环 | READY |
| NaVILA / Evolve-Nav / Aware-VLN | PENDING |

`READY` 只表示已经有实际代码或验证结果；占位脚本不会标为完成。

DualVLN 仿真闭环已在 Isaac Sim 5.1 Hospital 中实际验证：输入为 `640x480` 对齐 RGB-D、相机内参和采集时刻 `[x,y,yaw]`；模型返回 `33x2` ego trajectory，经 adapter 与安全跟踪器写入 B2W policy 的 `[vx,0,yaw_rate]`。40 秒测试位移 `8.081 m`、upright `100%`、最低高度 `0.606 m`，模型最终返回 `STOP` 并持续输出零速度。

## 已迁入的 B2W 内容

```text
src/robot_controller/scripts/rsl_rl/play.py  Hospital、键盘、目标点和移动验收入口
src/robot_controller/scripts/rl_utils.py     Isaac Sim 跟随相机辅助函数
src/robot_controller/config/b2w.yaml         task、维度、命令范围和关节清单
src/robot_description/urdf/                  B2W URDF
src/robot_description/meshes/                B2W mesh
checkpoints/b2w_locomotion/model_2600.pt      已验证的 locomotion checkpoint
```

快速启动：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
./src/robot_controller/scripts/run_b2w_hospital.sh goal
```

启动脚本会在需要时自动从 `(base)` 激活 `isaaclab232`，并加载 Isaac Sim 环境变量。
可以用 `./src/robot_controller/scripts/run_b2w_hospital.sh env-check` 单独检查环境。

详细操作见 [src/robot_controller/README.md](src/robot_controller/README.md)。

完整 `robot_lab` 已移动到工作区内部的 `third_party/robot_lab`，作为原始 task 注册和 Isaac Lab 环境依赖。B2W 启动脚本会自动把它的 `source/robot_lab` 加入 `PYTHONPATH`。

Unitree 官方 `unitree_ros` 已克隆到 `third_party/unitree_ros`，参考 commit 为 `daadf41ee9afce8f90fdc09a98506012691fa122`。官方 B2W URDF 与 RobotLab 当前使用的 B2W URDF 哈希一致；第三方版本与更新规则见 `third_party/README.md`。

## DualVLN 仿真闭环

当前阶段只运行 Isaac Sim，不启动 ROS2、Unitree SDK、RealSense 驱动、LiDAR、SLAM、Nav2 或机械臂：

```text
Isaac RGB-D + instruction + capture odometry
                    -> independent DualVLN service
                    -> 33x2 ego trajectory
                    -> trajectory_adapter
                    -> path_follower + safety_filter
                    -> [vx, 0, yaw_rate]
                    -> RobotLab B2W locomotion policy
```

运行命令见 `src/vln_models/dualvln/README.md`。完整 40 秒视频位于 `checkpoints/b2w_locomotion/videos/play/rl-video-step-0.mp4`。

## DualVLN 实机接口（当前不运行）

下面保留的硬件映射只供以后参考，不属于当前仿真启动链。

| 数据/接口 | 工作区配置 | 用途 |
|---|---|---|
| 前向 RGB | `src/sensors/camera/config/b2_front_rgb.yaml` | DualVLN observation |
| 自然语言指令 | `/vln/instruction` | DualVLN instruction |
| B2W odometry | `/b2_366/base/odom` | 执行模型轨迹 |
| 自动速度输入 | `/b2_366/autonomous_mid_priority/cmd_vel` | 接入 B2 twist mux |
| 急停 | `/b2_366/e_stop` | 最高优先级停车 |

官方 real-world async agent 的连续轨迹分支需要对齐 RGB-D；当前仿真已满足。以后若转实机，需要补齐对齐 depth，不能直接把现有 RGB-only 硬件入口标为完成。LiDAR、GNSS、后置相机、本体 IMU 和机械臂状态仍不属于该模型的最小输入。

构建工作区后，最小实机硬件入口为：

```bash
export B2_NS=b2_366
export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch robot_bringup b2w_dualvln_hardware.launch.py
```

这个入口只组合上游 `b2_platform`/`b2_control` 的可执行节点和参数文件，不改写官方源码或 launch。它启动前向相机、状态发布、B2 高层速度控制和带急停的 `twist_mux`；不启动后置相机、RealSense、LiDAR、GNSS 或 Z1。

## 上游文件保护与迁移记录

`../Unitree_B2_MybotShop_Software-main` 作为官方硬件软件上游，只读使用。本次没有修改、移动或删除其中任何文件，也没有复制整套驱动；工作区配置只记录其已发布的 ROS topic 和必要运行参数。

如果后续确实必须修复上游，必须先在本节记录原文件、原因、补丁和验证结果。目前已知但尚未修改的上游问题：

- 模拟 RGB 使用 Unitree 官方 B2W URDF 的 `base_link -> f_oc_link` 标称外参；MybotShop 定制实机仍需核对实际安装。
- 前后内置相机共用 `sensor/camera_info`，同时运行时需要在工作区 remap。
- 前后相机标定文件数值相同，启用 rectification 前必须核验。
- `b2_statepublisher` 的 odometry 坐标变换含上游 TODO，闭环运行前必须实机验证轴向。

本次新增或修改的工作区文件：

```text
README.md
configs/sensors.yaml
src/robot_bringup/CMakeLists.txt
src/robot_bringup/config/b2w_dualvln.yaml
src/robot_bringup/launch/b2w_dualvln_hardware.launch.py
src/robot_bringup/package.xml
src/robot_controller/README.md
src/robot_controller/scripts/rsl_rl/play.py
src/robot_controller/scripts/run_b2w_hospital.sh
src/vln_models/dualvln/sim_server.py
src/vln_models/dualvln/sim_client.py
src/vln_models/dualvln/run_sim_server.sh
src/vln_interface/scripts/trajectory_adapter.py
src/navigation_bridge/scripts/path_follower.py
src/navigation_bridge/scripts/safety_filter.py
src/robot_description/config/b2w_sensor_frames_official.yaml
src/sensors/camera/README.md
src/sensors/camera/config/b2_front_rgb.yaml
src/sensors/camera/config/b2_front_rgb_sim.yaml
src/vln_models/dualvln/README.md
src/vln_models/dualvln/config/model.yaml
third_party/COLCON_IGNORE
third_party/README.md
third_party/unitree_ros/  # official clone, commit daadf41e
```
