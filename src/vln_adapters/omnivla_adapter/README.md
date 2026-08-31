# OmniVLA 适配器

本包接入的是 `NHirose/OmniVLA` 的完整 `omnivla-original`，不是 OmniVLA-edge。
提供两个互相独立的执行模式：`omnivla` 是公平比较用的共享轨迹模式；
`omnivla_native` 是复现官方 waypoint-4 速度换算的模式。两者都继续使用本工程的
B2-W SRU-ONNX 低层执行器。
公平评测链路固定为：

```text
当前 RGB + 原始语言指令
  -> 独立 OmniVLA 服务（modality_id=7）
  -> 8x4 [x,y,heading_x,heading_y]
  -> 米制 base_link 轨迹
  -> /vln/command
  -> 共享 Pure Pursuit -> /nav_vel -> SRU ONNX B2-W
```

官方模式的数据流为：

```text
8x4 输出 -> waypoint[4] -> 官方 waypoint-to-velocity -> NavigationCommand.VELOCITY
           -> /nav_vel -> SRU ONNX B2-W
```

ROS 节点只订阅 `/camera/rgb/image_raw`、`/vln/instruction`、`/episode/state` 和
`/episode/id`。它不订阅 Depth、LiDAR、odom、GPS、goal image 或 ground-truth goal，
也不直接发布 `/nav_vel`。

回合切换时，`/episode/id`、`/vln/instruction` 和 `/episode/state` 可能因 DDS
跨话题调度而乱序。适配器收到 `START` 后会先保持 STOP，等待真实 episode ID、指令和
0.5 秒稳定窗口，再向服务端发送 `/reset`；旧回合的迟到请求会被丢弃，不会污染新回合。
该等待时间由 `runtime.episode_start_settle_time` 配置。

## 权重与环境

源码固定为 `third_party/OmniVLA`。完整权重放到：

```text
checkpoints/vln/omnivla/omnivla-original
```

官方环境文档以 Python 3.10、NumPy 1.26.4、Torch 2.2.0、TorchVision 0.17.0、
TorchAudio 2.2.0 为基线，并将 FlashAttention 2.5.5 标为训练依赖。不要安装进 ROS、
Isaac Sim 或其他模型环境。
官方当前 `pyproject.toml` 的 `av` 与 `openai-clip` 依赖项缺少逗号，当前 pip 会直接报
`TOMLDecodeError: Unclosed array`；该问题属于上游源码，本项目没有修改。推理服务通过
`PYTHONPATH` 直接导入官方源码，不执行 editable install，也不加载 TensorFlow/RLDS 训练栈。

本机是 RTX 5090（Blackwell）。PyTorch 官方从 2.7/CUDA 12.8 才提供 Blackwell wheel，
所以官方 Torch 2.2.0/CUDA 12.1 基线与本机不兼容。当前独立环境已经实测通过
Torch 2.7.1+cu128、TorchVision 0.22.1+cu128 和上游 Transformers fork 4.40.1。
官方 `define_model()` 推理路径没有启用 FlashAttention，因此 Blackwell 推理模式不要求
安装仅训练使用的旧 FlashAttention 2.5.5。该兼容方式不修改第三方模型代码。

使用 RTX 5090 兼容环境时必须显式开关启动：

```bash
OMNIVLA_ALLOW_BLACKWELL_COMPAT=1 \
OMNIVLA_PYTHON=/home/mifcom2/miniconda3/envs/omnivla/bin/python \
  ./src/vln_adapters/omnivla_adapter/run_inference_server.sh
```

OmniVLA 加载前会检查可用显存（默认至少 14 GiB），并自动设置
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。如果提示显存不足，通常是上一次
Isaac Sim 或模型服务仍在占用显存。先查看：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

只结束确认属于本次评测的残留 PID，再重新启动；不要为了绕过检查强行降低阈值。

不开该开关时，脚本继续严格校验上游 Torch 2.2.0/FlashAttention 2.5.5 基线。

启动真实服务：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
OMNIVLA_ALLOW_BLACKWELL_COMPAT=1 \
OMNIVLA_PYTHON=/home/mifcom2/miniconda3/envs/omnivla/bin/python \
  ./src/vln_adapters/omnivla_adapter/run_inference_server.sh
```

接口测试服务（只生成固定直线，不能用于模型评测）：

```bash
./src/vln_adapters/omnivla_adapter/run_inference_server.sh --stub
```

stub 模式启动 ROS 时需显式增加 adapter 参数 `allow_stub_server:=true`；统一 benchmark
默认拒绝 stub，避免把假轨迹记录为 OmniVLA 结果。

## 单回合

服务显示 `[OMNIVLA SERVER] Ready` 后，在另一个终端执行：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=omnivla episode:=dynanav_001 \
  comparison_track:=rgb_only sensor_profile:=rgb_only \
  goal_profile:=language_only evaluation_mode:=trajectory_normalized \
  desired_speed:=1.0 shutdown_after_finish:=false
```

调试话题：

```bash
ros2 topic echo /vln/omnivla/raw_trajectory
ros2 topic echo /vln/omnivla/metadata
ros2 topic echo /vln/debug_path
```

官方 native 模式（单独评测，不与公平轨迹结果合并）：

```bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=omnivla_native episode:=hospital_001 \
  comparison_track:=omnivla_native sensor_profile:=rgb_only \
  goal_profile:=language_only evaluation_mode:=native_output \
  desired_speed:=1.0 shutdown_after_finish:=false
```

这里的 `omnivla_native` 仍会发布 `NavigationCommand.VELOCITY`，但不会绕过
`navigation_bridge` 或 SRU-ONNX；桥接层继续负责 `/nav_vel` 的消息检查、加速度平滑和
超时停车。native 模式只改变“模型输出到速度”的转换方式。

`NavigationCommand.dt=1/3 s`，项目既有约定将 `horizon` 记为 `dt * points_count`，
因此 8 点为 `8/3 s`。位置只执行 `raw_xy * 0.1`，不会执行 `cumsum`、交换坐标或翻转 y。

## 当前限制

- `omnivla` 使用 `trajectory_normalized`；`omnivla_native` 使用 `native_output`。
- 官方模式的速度限制保持上游示例的 `max_v=0.3 m/s`、`max_w=0.3 rad/s`，不会修改
  SRU-ONNX 网络或关节控制。
- 模型本身没有可靠 STOP，成功由共享 `goal_monitor` 判断。
- 权重缺失、shape 错误、NaN/Inf、零轨迹、超时、OOM、服务断开或迟到响应均停车。
- 真实模态泄漏测试必须在完整权重存在后执行；未通过前不得运行正式 benchmark。

## 2026-08-24 实测

- 权重完整并通过 `SHA256SUMS` 全文件校验。
- 官方静态 RGB 输出 `8x4`，普通单次延迟 0.118 秒，峰值分配显存
  16069447680 bytes，language-only 泄漏差异 0.0。
- `dynanav_001` 无行人回合完成完整技术闭环；平均推理延迟 0.122 秒，控制频率
  20.02 Hz，运行轨迹已转换为 world 坐标绘制到 Isaac viewport。
- 导航任务未成功：40.02 秒后误差 6.961 m，路径 7.456 m，静态接触 61 次；因此未运行
  动态行人回合。
