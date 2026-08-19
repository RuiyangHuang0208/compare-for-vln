# Unitree B2W Controller

这里保存从 RobotLab 迁入的新工作区运行入口。完整 RobotLab 仓库位于工作区内的 `third_party/robot_lab`，负责注册 Isaac Lab task 和提供完整 locomotion 环境；本目录负责 Hospital 场景、键盘、目标点和后续 VLN 对接入口。

键盘、目标点、40 秒移动验收和 DualVLN 仿真闭环都直接修改 policy observation 中的 `[vx, vy, yaw_rate]`。B2W policy、57 维 observation 和 16 维 action 结构保持不变；当前仿真不经过 ROS2 `/nav_vel`。

## 前置条件

启动脚本会自动检测 Isaac Lab；从 `(base)` 运行时会自动激活 `isaaclab232` 并加载
`/home/mifcom2/isaacsim/setup_conda_env.sh`，不需要手动执行 `conda activate`。进入工作区：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
```

可先做不启动仿真窗口的环境检查：

```bash
./src/robot_controller/scripts/run_b2w_hospital.sh env-check
```

## 运行中发送目标点

```bash
./src/robot_controller/scripts/run_b2w_hospital.sh goal
```

程序启动后，在同一个终端输入：

```text
goal 2.0 1.0
status
goal 0.0 0.0
stop
quit
```

`goal X Y` 使用 Hospital 世界坐标。当前控制器只把目标方向转换为 `[vx, 0, yaw_rate]`，不包含路径规划或避障，因此目标之间需要有可通行的直线路径。

## 键盘控制

```bash
./src/robot_controller/scripts/run_b2w_hospital.sh keyboard
```

```text
W      前进
S      后退
A      左转
D      右转
SPACE  停止
```

## 40 秒移动验收

```bash
./src/robot_controller/scripts/run_b2w_hospital.sh test
```

该模式依次测试前进、停止、后退、停止、左转、停止、右转和停止。

## DualVLN 模拟传感器

在 Hospital 中加载 B2W 前向 RGB-D 相机和模拟 odometry：

```bash
./src/robot_controller/scripts/run_b2w_hospital.sh dualvln-sensors
```

该模式仍使用交互目标点控制并原地等待命令，但同时可从 Isaac Lab scene 读取：

```text
front_rgb/rgb                 640 x 480 RGB, 10 Hz
front_rgb/distance_to_image_plane 640 x 480 depth (m), 10 Hz
robot.root_link_pos_w         世界位置
robot.root_link_quat_w        世界姿态，wxyz
robot.root_lin_vel_b          机体系线速度
robot.root_ang_vel_b          机体系角速度
```

自动检查相机和 odometry：

```bash
./src/robot_controller/scripts/run_b2w_hospital.sh sensor-test
```

输出保存在 `outputs/dualvln_sensor/front_rgb.png` 和 `front_depth.png`。传感器不会加入 locomotion policy observation，policy 维度保持 57/16。

## DualVLN 仿真闭环

先在独立终端启动推理服务，再启动 Isaac Hospital：

```bash
./src/vln_models/dualvln/run_sim_server.sh
./src/robot_controller/scripts/run_b2w_hospital.sh dualvln
```

第二个进程启动后默认原地停车，必须输入 `instruction TEXT` 才会开始规划和移动。其余命令都是可选的：`speed 0.6` 可在任务前或运行中调整期望前进速度，`home` 立即重置环境并把机器人刷新到启动状态，`status` 查询状态，`stop` 停车，`quit` 退出。速度允许范围为 `0.05-5.0 m/s`，默认 `0.3 m/s`；RobotLab policy 的训练范围只到 `1.0 m/s`，更高数值仅用于超出训练分布的仿真实验。`home` 后保持停车，必须输入新的 `instruction TEXT` 才会继续导航。`stop`、HTTP 错误、非法轨迹和结果超时都会清除轨迹并立即写入零速度。

## 直接调用 play.py

入口位于 `scripts/rsl_rl/play.py`，默认 checkpoint 位于工作区根目录的 `checkpoints/b2w_locomotion/model_2600.pt`。完整 task、关节和命令信息见 `config/b2w.yaml`。
