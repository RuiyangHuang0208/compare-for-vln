# Unitree B2W 控制器

这里保存 B2W 在 Isaac Sim 中的统一运行入口。默认使用 SRU Gazebo deployment 的
`policy_force_new.onnx` 和 SRU Isaac B2W USD；原来的 `isaac-pt` 与 RobotLab policy 均完整
保留，可随时切换对照。完整 RobotLab 仓库位于 `third_party/robot_lab`。

键盘、目标点和旧 DualVLN 直接入口仍可直接修改 policy observation 中的
`[vx, vy, yaw_rate]`。新统一评测入口则由 `udp_velocity_bridge` 把 ROS 2 `/nav_vel` 送入
同一 observation；低层 policy、关节映射与动作缩放没有改变。

| 选择 | Observation | Action | Robot asset |
|---|---:|---:|---|
| `sru-onnx`（默认） | 60 | 16 | SRU `b2w_rsl.usd` |
| `isaac-pt`（保留） | 60 | 16 | SRU `b2w_rsl.usd` |
| `robotlab`（保留） | 57 | 16 | RobotLab B2W asset |

`sru-onnx` 与 `isaac-pt` 的 60 维输入均为 base linear velocity、base angular velocity、
projected gravity、3 维 command、16 维 joint offset、16 维 joint velocity 和 16 维上一步
raw action。输出前 12 维以 `default + 0.5 * action` 写入腿关节位置，后 4 维以
`5.0 * action` 写入轮关节速度。`sru-onnx` 使用与 Gazebo 相同的显式 256 维 LSTM
hidden/cell，并在 50 Hz 更新；Isaac 的 200 Hz physics 子步保持最新动作。

## 前置条件

启动脚本会自动检测 Isaac Lab；从 `(base)` 运行时会自动激活 `isaaclab232` 并加载
`${ISAAC_SIM_ROOT:-$HOME/isaacsim}/setup_conda_env.sh`，不需要手动执行 `conda activate`。进入工作区：

```bash
cd "$ROBOT_VLN_WS"
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

ROS 2 统一入口使用：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_bringup keyboard_b2w.launch.py
```

`udp_velocity_bridge` 支持 `keyboard` 与 `vln` 两个互斥 source。运行中向
`/robot_controller/source` 发布新 source 时会先发送零速度，并等待该 source 的新命令；
非活动 source 的消息不能控制机器人。VLN 统一输出始终是 `/nav_vel`，不需要模型知道
Isaac UDP 或 locomotion policy。

## Isaac Sim viewport 轨迹

非 headless 的统一 VLN 评测会自动显示两条轨迹：青色为 `/vln/debug_path` 的模型预测，
黄色为机器人实际运动轨迹。ROS 节点通过独立 UDP `5823` 发送世界坐标路径；速度仍使用
UDP `5820`，因此调试显示不会修改 locomotion 命令。实际轨迹按 `0.08 m` 间距采样，最多
保留 500 点。STOP 或 reset 会清除相应 marker。

## 40 秒移动验收

```bash
./src/robot_controller/scripts/run_b2w_hospital.sh test
```

该模式依次测试前进、停止、后退、停止、左转、停止、右转和停止。

2026-08-21 在 Isaac Sim 5.1 Hospital 中对默认 `sru-onnx` 实际运行 40 秒：前进
`0.520 m/s`、后退 `-0.493 m/s`、左转 `0.534 rad/s`、右转 `-0.516 rad/s`，停止与全部
移动项目 PASS；upright `100%`，最低高度 `0.704 m`，最大腿关节位置 `1.327 rad`。
保留的 `isaac-pt` 已于 2026-08-20 通过同一测试。

## 切换控制器

原 policy 和 checkpoint 没有删除。只对当前命令设置环境变量：

```bash
B2W_LOCOMOTION_POLICY=isaac-pt ./src/robot_controller/scripts/run_b2w_hospital.sh goal
B2W_LOCOMOTION_POLICY=robotlab ./src/robot_controller/scripts/run_b2w_hospital.sh goal
```

键盘、测试和 DualVLN 模式同样适用。取消该环境变量后恢复默认 `sru-onnx`：

```bash
unset B2W_LOCOMOTION_POLICY
```

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

输出保存在 `outputs/dualvln_sensor/front_rgb.png` 和 `front_depth.png`。传感器不会加入 locomotion policy observation，因此两套 locomotion 契约仍分别保持 60/16 和 57/16。

## DualVLN 仿真闭环

先在独立终端启动推理服务，再启动 Isaac Hospital：

```bash
./src/vln_adapters/dualvln_adapter/run_inference_server.sh
./src/robot_controller/scripts/run_b2w_hospital.sh dualvln
```

第二个进程启动后默认原地停车，必须输入 `instruction TEXT` 才会开始规划和移动。其余命令都是可选的：`speed 0.6` 可在任务前或运行中调整期望前进速度，`home` 立即重置环境并把机器人刷新到启动状态，`status` 查询状态，`stop` 停车，`quit` 退出。速度允许范围为 `0.05-5.0 m/s`，默认 `0.3 m/s`；RobotLab policy 的训练范围只到 `1.0 m/s`，更高数值仅用于超出训练分布的仿真实验。`home` 后保持停车，必须输入新的 `instruction TEXT` 才会继续导航。`stop`、HTTP 错误、非法轨迹和结果超时都会清除轨迹并立即写入零速度。

## 直接调用 play.py

入口位于 `scripts/rsl_rl/play.py`。默认 ONNX 位于
`checkpoints/b2w_locomotion/sru_onnx/policy_force_new.onnx`，保留的 TorchScript 位于
`checkpoints/b2w_locomotion/isaac_pt/policy_b2w_new_2.pt`，RobotLab checkpoint 位于
`checkpoints/b2w_locomotion/model_2600.pt`。完整 task、关节和命令信息见 `config/b2w.yaml`。
