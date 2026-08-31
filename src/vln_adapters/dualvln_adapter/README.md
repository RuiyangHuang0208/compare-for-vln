# DualVLN ROS 2 适配器

本目录是 DualVLN 的完整工作区接入边界，布局与 `ticvla_adapter` 一致：

```text
dualvln_adapter/
├── config/dualvln.yaml
├── launch/dualvln.launch.py
├── dualvln_adapter/
│   ├── dualvln_node.py       # ROS 2 统一接口
│   ├── inference_server.py   # 独立 Python 3.11 模型服务
│   ├── service_client.py     # 直接 Isaac 模式兼容客户端
│   └── coordinates.py        # 轨迹坐标变换
├── run_inference_server.sh
└── test/
```

官方 InternNav 源码只保存在 `third_party/InternNav`，权重只保存在
`checkpoints/vln/dualvln`。本目录不复制模型源码、权重或 B2-W 控制代码。

## 输入与输出

官方 `InternVLAN1AsyncAgent.step_s1()` 实际使用当前 RGB、目标帧 RGB 和与两帧对齐的
depth，因此仿真提供 RGB-D，而不是只提供 RGB。ROS adapter 订阅：

- `/vln/instruction`
- `/camera/rgb/image_raw`
- `/camera/depth/image_raw`
- `/camera/rgb/camera_info`
- `/odom`
- `/episode/state`

连续输出是米制、本体坐标系下已经累计完成的 `Nx2` 轨迹。适配器不能再次 `cumsum`。
它使用 RGB-D 拍摄时刻的 pose 补偿推理延迟，并发布
`vln_interfaces/NavigationCommand.TRAJECTORY`。离散 FORWARD/LEFT/RIGHT 转换为局部短轨迹，
仍进入共享 path follower；STOP 直接发布统一 STOP。模型不会直接控制 B2-W 关节。

## 运行

终端 1，加载 DualVLN：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
./src/vln_adapters/dualvln_adapter/run_inference_server.sh
```

终端 2，启动共享 Isaac Sim、SRU ONNX locomotion 和导航链路：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=dualvln episode:=hospital_001 \
  evaluation_mode:=trajectory_normalized sensor_profile:=rgb_d
```

运行中替换语言指令：

```bash
ros2 topic pub --once /vln/instruction std_msgs/msg/String \
  "{data: 'Go straight ahead and stop at the vending machine.'}"
```

运行中修改共享轨迹跟踪速度：

```bash
ros2 topic pub --once /navigation/desired_speed std_msgs/msg/Float32 "{data: 1.0}"
```

配置位于 `config/dualvln.yaml`，参数和 TIC-VLA 一样分为 `model.*`、`input.*`、
`output.*` 和 `runtime.*`。episode reset、FINISH、FAILED 和 SUCCESS 都会清除历史图像、
过期 future 和旧轨迹。

## 兼容直接模式

旧的无 ROS 2 直接模式仍可用于交互调试，但不是统一评测入口：

```bash
# 终端 1 仍使用上面的 run_inference_server.sh
./src/robot_controller/scripts/run_b2w_hospital.sh dualvln
```

该模式在 Isaac 终端中接受 `instruction TEXT`、`speed VALUE`、`home`、`status`、`stop`
和 `quit`。`service_client.py` 只为该兼容入口提供非阻塞 HTTP 客户端。

## 上游兼容边界

InternNav 固定 commit 为 `7a5c62400ac45b313d9b709c740b64191556a242`。工作区保留的
最小兼容补丁包括：

- 最小导入模式，避免 Habitat 评测依赖。
- SDPA attention backend，适配 RTX 5090。
- DepthAnything checkpoint 路径和公开 checkpoint FFN 形状兼容。
- Diffusers 0.33 gradient-checkpointing 签名兼容。

`inference_server.py` 使用 `torch.inference_mode()`。没有修改模型 observation、policy
结构、权重或轨迹归一化。

2026-08-19 的直接模式实测输出为 `33x2` ego trajectory，运行 40 秒、位移
`8.081 m`、upright `100%`，最终官方 System 2 输出 STOP。统一 ROS adapter 另有坐标、
STOP、reset 和超时集成测试。
