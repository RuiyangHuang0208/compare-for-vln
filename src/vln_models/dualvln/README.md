# DualVLN Integration Boundary

这里接入官方 `InternVLA-N1-DualVLN` 的 real-world async agent。虽然基准表把 DualVLN 标为 RGB，官方 `InternVLAN1AsyncAgent.step_s1()` 实际使用当前 RGB、目标帧 RGB 以及与两帧对齐的 depth 生成连续轨迹，因此本仿真闭环提供 RGB-D。

## 最小闭环

```text
language instruction + aligned front RGB-D
                 -> DualVLN
                 -> ego-centric trajectory
                 -> trajectory_adapter
                 -> NavigationCommand.TRAJECTORY
                 -> path_follower + B2 odometry
                 -> safety_filter
                 -> [vx, 0, yaw_rate] policy observation
                 -> B2W locomotion
```

必需输入：

- `/vln/instruction`: 自然语言导航指令。
- Isaac `front_rgb/rgb`: `640x480` 前向 RGB。
- Isaac `front_rgb/distance_to_image_plane`: 同相机、同像素的 depth（米）。
- Isaac B2W `[x, y, yaw]`: 不是模型 observation，但把采集时刻的 ego trajectory 转到世界坐标时必须使用。

不加入仿真的设备：RealSense 驱动、LiDAR、GNSS、后置相机、机械臂、Unitree SDK 和 ROS2 硬件桥。深度直接由 Isaac 相机产生。

## 仿真运行

两个进程使用独立 Conda 环境，但共享 GPU：

```bash
# Terminal 1
cd /home/mifcom2/b2w/robot_vln_ws
./src/vln_models/dualvln/run_sim_server.sh

# Terminal 2: 先激活现有 isaaclab232 并 source Isaac Sim 环境
cd /home/mifcom2/b2w/robot_vln_ws
./src/robot_controller/scripts/run_b2w_hospital.sh dualvln
```

运行中在 Terminal 2 输入命令。它们不需要全部依次输入，只有开始导航时需要
`instruction TEXT`：

```text
instruction Walk forward to the vending machine and stop in front of it.
```

可选运行时命令：

| 命令 | 作用 |
|---|---|
| `speed 0.6` | 把期望前进速度改为 `0.6 m/s`，任务前或运行中均可输入 |
| `home` | 立即重置整个仿真环境，将机器人刷新到本次启动位置并停车 |
| `status` | 只查看位置、当前指令、路径状态、期望速度和当前速度命令 |
| `stop` | 取消当前任务、清除轨迹并立即停车 |
| `quit` | 停车并退出 Isaac Sim 进程 |

默认启动状态是等待指令并保持 `[0,0,0]`。只有收到 `instruction TEXT` 后才开始提交 RGB-D 和驱动机器人。自动化测试仍可显式传入 `--instruction "..."`，此时会在启动后自动执行。

`home` 不调用 DualVLN，也不执行返航轨迹。它会立即调用 Isaac Lab environment reset，
恢复机器人的初始 base 位姿、朝向、速度和关节状态，同时重置 policy 状态、清除旧轨迹、
离散动作和可视化，并使尚未完成的模型结果失效。reset 后保持停车，必须再次输入
`instruction TEXT` 才会继续导航。

默认速度是 `0.3 m/s`。需要更快时建议先输入 `speed 0.6`，稳定后可尝试
`speed 1.0`；运行时允许范围为 `0.05-5.0 m/s`，最高可输入 `speed 5.0`。
RobotLab B2W policy 原始 `vx` 训练范围只有 `[-1.0,1.0] m/s`，因此 `1.0 m/s`
以上属于超出训练分布的仿真实验命令：不保证实际达到设定速度，也不保证机器人稳定。
超过 `0.4 m/s` 同时高于 InternNav 官方实机控制器的保守上限。横向速度固定为
`vy=0`，角速度仍限制为 `0.4 rad/s`。

非 headless 模式会在 Isaac Sim 中打开 `DualVLN Monitor` 窗口，实时显示：

- 当前自然语言指令。
- 等待、推理、轨迹跟踪、离散动作、停止、超时和错误状态。
- System 1 轨迹点数与像素目标，或 System 2 的 `FORWARD`、`TURN`、`STOP` 动作。
- 请求帧编号、推理耗时、期望速度、实际 command 和局部轨迹剩余距离。

System 1 返回的世界坐标轨迹还会以小球显示在 Hospital 地面上。这些是模型和控制器
实际输出的可观察状态，不是额外生成的隐藏思维文本。使用 `--headless` 时不会创建窗口或
轨迹标记，但终端日志保持不变。

推理服务使用 PyTorch SDPA，以支持 RTX 5090；模型和 policy 结构均未改变。path follower 默认采用官方控制器相同的 `0.3 m/s` 期望速度；仿真运行时上限放宽到 `5.0 m/s`，角速度上限保持 `0.4 rad/s`。

## 所有权边界

- 官方 InternNav 源码放在 `third_party/InternNav`，不复制到本目录。
- 官方模型权重放在 `checkpoints/dualvln`，不提交到源码目录。
- `ros_node.py` 只负责消息收发和调用官方推理 API，不包含 B2W 控制代码。
- checkpoint-specific 图像预处理和轨迹张量解释必须以实际安装的 InternNav commit 为准。

官方源码固定在 `third_party/InternNav`，模型权重位于 `checkpoints/dualvln`。仿真入口不会启动任何 ROS2 或真机驱动。

## 上游最小兼容补丁

固定源码 commit：`7a5c62400ac45b313d9b709c740b64191556a242`。工作区只修改以下四个上游文件，并保留在 third-party Git diff 中：

- `agent/__init__.py`: `INTERNNAV_MINIMAL_IMPORT=1` 时只加载 real-world agent，避免引入 Habitat 评测依赖。
- `internvla_n1_agent_realworld.py`: attention backend 可由服务设置为 `sdpa`，适配 RTX 5090。
- `internvla_n1_arch.py`: DepthAnything 路径可配置，并按公开 checkpoint 的 `[1024,384]` FFN 形状设置 `2/3` multiplier。
- `nextdit_traj.py`: 兼容 Diffusers 0.33 的 gradient-checkpointing 方法签名。

此外，模型调用由工作区自己的 `sim_server.py` 统一放在 `torch.inference_mode()` 中。

没有修改模型 observation、policy 架构、权重或轨迹归一化。

## 实测结果

2026-08-19 在 Isaac Sim 5.1 Hospital、RTX 5090 上运行 40 秒：

```text
instruction:      Walk forward to the vending machine and stop in front of it.
model output:     33x2 ego trajectory; final System 2 output STOP
displacement:     8.081 m
upright:          100.000%
minimum height:   0.606 m
maximum command:  0.363
stop command:     [0.00, 0.00, 0.00]
video:            checkpoints/b2w_locomotion/videos/play/rl-video-step-0.mp4
```
