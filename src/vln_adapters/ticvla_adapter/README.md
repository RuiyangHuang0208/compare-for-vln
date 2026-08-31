# TIC-VLA ROS 2 适配器

本包使用官方 DynaNav benchmark 实际调用的
`third_party/TIC-VLA/DynaNav/ticvla.py::TICVLA.predict_async`，不使用
`rl/fine_tuning/ticvla.py`。checkpoint 会先去掉 Lightning 的 `model.` 前缀，再以
`strict=True` 加载到 DynaNav 推理类，防止混用不兼容的 ActionExpert 权重。

独立的 Python 3.11 推理服务加载 InternVL3-1B 和 TIC-VLA checkpoint。服务会核对
`vlm` 与 `action_expert` 的权重键映射，不兼容的权重会直接报错。ROS 节点维护 4 帧
RGB 历史、由 odom 构造的官方 6 值运动状态、每秒实际位移历史和异步请求；VLM 的 KV
cache 跨控制周期保留，episode reset 时由服务完整清除。首个有效结果产生前以及 reset
之后始终发布 STOP。

动作解码按上游 DynaNav 的 `10 Hz` 节奏运行；高层 prompt 使用其 benchmark 的
`wheeled robot` 表述，底层执行器仍固定为 B2-W SRU-ONNX。

正式输出契约为 `(1, 30, 2)` 的米制本体局部 XY 点。这些点已经是轨迹位置，不是
`(dx,dy,dtheta)`，因此适配器不执行 `cumsum`。`predict_async` 使用当前 RGB 解码的
轨迹已经位于当前 `base_link`，适配器不再对它做第二次位姿补偿，然后发布
`NavigationCommand.TRAJECTORY`；总延迟、VLM 估计延迟和
ActionExpert 延迟分别记录。

原生高层模式使用官方 Spot DynaNav 的曲率公式，直接把局部 `30x2` 轨迹转换成
`NavigationCommand.VELOCITY`。官方 Spot 参数是 `L=1.0`、角度增益 `0.8`、滤波
`0.35`、前馈 `0.5`、线/角加速度 `2.0/3.0`、减速度 `2.5/3.5`。Spot 的
`1.5 m/s` 最大线速度超过已验证 B2-W SRU-ONNX 输入范围，因此仅在最终接口裁到
`1.0 m/s`；`wz` 保持官方 Spot 的 `1.0 rad/s`。公平模式仍发布 TRAJECTORY，使用共享
Pure Pursuit。

原生模式命令：

```bash
./scripts/run_dynanav_full85_all.sh --native --episodes hospital_001 ticvla
```

本地文件：

```text
checkpoints/vln/ticvla/InternVL3-1B/
checkpoints/vln/ticvla/TIC-VLA-model.ckpt
```

先运行离线 `30x2` 推理：

```bash
./src/vln_adapters/ticvla_adapter/run_offline_inference.sh
```

输出和延迟保存在 `outputs/ticvla/offline_30x2.json`。

闭环运行：

```bash
# 终端 1
./src/vln_adapters/ticvla_adapter/run_inference_server.sh

# 终端 2
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=ticvla episode:=dynanav_001 \
  desired_speed:=1.0 shutdown_after_finish:=false
```

带窗口启动时，30 点预测轨迹会以青色点直接显示在 Isaac Sim viewport，B2-W 实际轨迹
以黄色点显示。运行中可从任意已加载工作区的 ROS 2 终端替换指令：

```bash
ros2 topic pub --once /vln/instruction std_msgs/msg/String \
  "{data: 'Go straight ahead and stop at the vending machine.'}"
```

发布空字符串会立即停车，并清除旧预测轨迹。

Office 单回合检查建议使用下面的命令。它会自动启动 TIC-VLA 服务、等待 `5802/health`
就绪，再启动 Isaac Sim；直接运行 `ros2 launch` 而没有先启动服务会得到
`mean_inference_latency: null`，不能作为模型失败结果。

```bash
cd /home/mifcom2/b2w/robot_vln_ws
export DYNANAV_NO_PEDESTRIANS=1
export DYNANAV_EXPERIMENT=ticvla_office_success_check
./scripts/run_dynanav_full85_all.sh --episodes dynanav_026 --no-resume ticvla
```

运行中可修改共享轨迹跟踪速度：

```bash
ros2 topic pub --once /navigation/desired_speed std_msgs/msg/Float32 "{data: 2.0}"
```

到达 episode 目标后会保存结果并停车，但默认保留 Isaac Sim 窗口，可继续发新指令。
默认速度为 `1.0 m/s`；`2.0 m/s` 是允许的实验上限。

2026-08-21 实际验证：

- 官方 checkpoint 成功映射 637 个 VLM tensor 和 50 个 ActionExpert tensor；
- 离线输出为有限数值 `(1, 30, 2)`，推理耗时 `1.162 s`；
- `hospital_001` 连续闭环运行 `60.12 s`，路径长度 `16.478 m`，最终误差
  `18.355 m`；
- 记录到 88 次小腿 link 原始接触事件，但当前 API 无法识别接触对象，尚不能解释为
  88 次障碍碰撞；
- 未进入 `0.5 m` 成功半径，episode 结果为 `FAIL`。

失败结果属于模型在当前指令和场景中的真实行为，不是 checkpoint 加载失败。适配器没有
注入目标点、交换坐标轴或添加第二次累积求和。
