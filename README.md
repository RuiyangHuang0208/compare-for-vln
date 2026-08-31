# 机器人视觉语言导航工作区

这个工作区用于把不同 VLN 模型接到同一台机器人上。目录骨架已经建立，并已迁入当前验证过的 B2W 移动控制运行资产。

新机器部署请先阅读 [`DOWNLOAD_AND_RUN.md`](DOWNLOAD_AND_RUN.md)：其中包含五个活动模型的
下载来源、固定目录、SHA-256 校验、子模块初始化、ROS 2 构建和最小运行命令。

## ROS 2 统一评测框架（当前权威入口）

当前可构建的数据流是：

```text
Isaac Sim 5.1 医院场景 + 评测回合
  -> 共享 RGB-D / CameraInfo / odom / TF / 时钟
  -> dummy | DualVLN | TIC-VLA | NaVILA | Uni-NaVid | OmniVLA 适配器
  -> /vln/command (vln_interfaces/NavigationCommand)
  -> 共享纯追踪器 / 速度滤波器
  -> /nav_vel
  -> robot_controller 独占控制源选择器
  -> SRU ONNX B2-W 移动控制器（默认）
```

`src/vln_interfaces` 是 ROS 2 权威消息包。模型源码只在 `third_party` 保存一份，权重
统一位于 `checkpoints/vln`，每个模型的 ROS 节点、独立推理服务、配置、launch 和测试
全部位于 `src/vln_adapters/<model>_adapter`。原单数目录 `src/vln_interface` 仅保留直接
DualVLN 兼容模式使用的轨迹工具，不作为新模型接入位置。

环境实测为 ROS 2 Jazzy、Isaac Sim 5.1、Isaac Lab 2.3.2（Python 包版本 0.54.2）、
Conda `isaaclab232`、Python 3.11.15、Torch 2.7.0+cu128。ROS 2 使用系统 Python 3.12，
因此大模型服务与 ROS 节点通过 localhost HTTP 隔离，避免把模型依赖安装进 Isaac 环境。

### 统一环境策略

这里的“统一环境”指所有模型使用完全相同的仿真、传感器、消息接口和机器人控制链路，
而不是把所有 Python 依赖强行安装到同一个 Conda 环境。当前版本必须保持下面的边界：

```text
系统 Python 3.12：ROS 2 Jazzy 节点、NavigationCommand、评估器
isaaclab232/Python 3.11：Isaac Sim 5.1、Isaac Lab、B2-W SRU-ONNX 控制器
模型服务环境：各模型官方推理依赖，通过 localhost HTTP 接入
```

模型服务环境对应关系如下：

| 模型 | 推理环境 | 原因 |
| --- | --- | --- |
| DualVLN | `internnav` | InternNav/DualVLN 官方依赖 |
| TIC-VLA | `internnav` | 当前适配器复用 InternNav 依赖 |
| NaVILA | `navila` | Python 3.10、LLaVA/FlashAttention 版本固定 |
| Uni-NaVid | `uninavid` | Python 3.10、官方 Transformers 版本固定 |
| OmniVLA | `omnivla` | 官方模型 fork、FlashAttention 和 Blackwell 兼容配置 |

这些环境都只负责加载一个模型并发布统一的 `/vln/command`，不允许直接控制关节。
所有模型实际共享：Isaac Sim 5.1 场景、B2-W USD、RGB/RGB-D 传感器、DynaNav 回合、
`navigation_bridge`、`/nav_vel` 和 SRU-ONNX locomotion。批量脚本会按模型顺序启动和清理
服务，避免多个大模型同时占用 GPU。

不要在 `isaaclab232` 中安装 ROS Jazzy 的 `rclpy`，也不要把 NaVILA/Uni-NaVid/OmniVLA
的 Transformers 或 FlashAttention 降级到同一版本。当前检查结果显示各模型需要的
Python、Torch 和 Transformers 版本不同；强行合并会破坏 Isaac Lab 或模型加载。启动前
可检查统一控制环境：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
./src/robot_controller/scripts/run_b2w_hospital.sh env-check
```

若以后确实需要“单一 Python 进程”，应另建兼容环境并重新构建 ROS 2 Python 绑定；这不是
当前 `isaaclab232` 环境内的安全改动，因此本项目不自动执行。

构建：

```bash
export ROBOT_VLN_WS="$(pwd)"
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 运行 VLN 模型

DualVLN、TIC-VLA、NaVILA、Uni-NaVid 和 OmniVLA 都采用独立模型服务。每次运行需要两个终端：终端 1 常驻加载模型，
终端 2 启动 ROS 2、Isaac Sim、B2-W 和统一导航链路。不要在模型加载完成前关闭终端 1。

### 已验证成功的简单回合

以下结果均为真实模型推理、共享 `navigation_bridge` 和 SRU ONNX B2-W 闭环结果，不是
dummy 或 stub。`simple_vending_machine` 使用 DynaNav 售货机起终点；
`simple_forward_3m_standard` 要求机器人直行 3 m，并使用 DynaNav 官方的 1.5 m 成功阈值。

```bash
# DualVLN：Go straight ahead and stop at the vending machine.
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=dualvln episode:=simple_vending_machine \
  comparison_track:=rgb_d sensor_profile:=rgb_d \
  evaluation_mode:=trajectory_normalized desired_speed:=1.0 \
  experiment:=simple_success shutdown_after_finish:=true

# NaVILA：Move straight forward for 3 meters, then stop.
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=navila episode:=simple_forward_3m_standard \
  comparison_track:=rgb_only sensor_profile:=rgb_only \
  evaluation_mode:=trajectory_normalized desired_speed:=1.0 \
  experiment:=simple_success shutdown_after_finish:=true

# Uni-NaVid：与 NaVILA 使用同一简单回合
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=uninavid episode:=simple_forward_3m_standard \
  comparison_track:=rgb_only sensor_profile:=rgb_only \
  evaluation_mode:=trajectory_normalized desired_speed:=1.0 \
  experiment:=simple_success shutdown_after_finish:=true
```

三次实测结果分别为：DualVLN `1.495 m`、NaVILA `1.494 m`、Uni-NaVid
`1.488 m`，均为 `success=true`、`SPL=1.0`、零碰撞。模型服务仍按下面各模型的
“终端 1”命令单独启动；三个大模型必须依次运行，不能同时占用同一块 GPU。

### DualVLN

终端 1，启动 DualVLN 推理服务：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
./src/vln_adapters/dualvln_adapter/run_inference_server.sh
```

终端 2，使用统一接口启动 Hospital 单回合：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=dualvln episode:=dynanav_001 \
  comparison_track:=rgb_d \
  evaluation_mode:=trajectory_normalized sensor_profile:=rgb_d \
  desired_speed:=1.0
```

### TIC-VLA

终端 1，启动 TIC-VLA 推理服务：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
./src/vln_adapters/ticvla_adapter/run_inference_server.sh
```

脚本默认选择 `internnav` 环境；只有需要替换 Python 时才设置
`TICVLA_PYTHON=/自定义环境/bin/python`。

终端 2，使用相同 B2-W、SRU ONNX 控制器和共享 path follower 启动：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=ticvla episode:=dynanav_001 \
  comparison_track:=rgb_only sensor_profile:=rgb_only \
  evaluation_mode:=trajectory_normalized \
  desired_speed:=1.0 shutdown_after_finish:=false
```

Office 单回合验证不要只启动 `ros2 launch`，必须先启动 TIC-VLA 推理服务。最不容易漏步骤的
方式是让批量脚本只选择一个回合：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
export DYNANAV_NO_PEDESTRIANS=1
export DYNANAV_EXPERIMENT=ticvla_office_success_check
./scripts/run_dynanav_full85_all.sh --episodes dynanav_026 --no-resume ticvla
```

结果中必须同时满足 `success: true` 和 `mean_inference_latency` 非空；若延迟为 `null`，
说明服务没有连接，不应归因于模型或场景。

### NaVILA

终端 1，在独立 Python 3.10 环境中启动官方 8 帧 NaVILA 推理服务：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
./src/vln_adapters/navila_adapter/run_inference_server.sh
```

终端 2，仅向 NaVILA 提供 RGB 和语言指令，并复用同一个 SRU ONNX B2-W controller 与
Pure Pursuit：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=navila episode:=dynanav_001 \
  comparison_track:=rgb_only \
  sensor_profile:=rgb_only evaluation_mode:=trajectory_normalized \
  desired_speed:=1.0 shutdown_after_finish:=false
```

当前 `1.0 m` 转弯半径已按
[NaVILA adapter 文档](src/vln_adapters/navila_adapter/README.md)完成左右 dummy 圆弧实测；
更换 locomotion 或 path follower 后必须重新标定。设为 `0.0` 时节点会主动拒绝启动。

### Uni-NaVid

终端 1，启动独立 Python 3.10 在线缓存推理服务：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
UNINAVID_PYTHON=/home/mifcom2/miniconda3/envs/uninavid/bin/python \
  ./src/vln_adapters/uninavid_adapter/run_inference_server.sh
```

看到 `[UNINAVID SERVER] Ready` 后，在终端 2 启动相同 B2-W、SRU ONNX、相机和共享 Pure Pursuit：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=uninavid episode:=dynanav_001 \
  comparison_track:=rgb_only sensor_profile:=rgb_only \
  evaluation_mode:=trajectory_normalized desired_speed:=1.0 \
  shutdown_after_finish:=false
```

Uni-NaVid 只接收 1 Hz RGB 新帧和语言指令。每次最多预测四个离散动作，执行前两个；动作轨迹结束由
`/navigation/trajectory_finished` 反馈触发下一次推理；轨迹超时由
`/navigation/trajectory_failed` 触发重新观察，不使用固定等待时间。原始输出可查看：

Uni-NaVid 的 30 度离散转向使用 `0.25 m` 圆弧，且共享 follower 会检查最终航向。使用 SRU ONNX
B2-W 实测单次左转/右转达到 `+25.49/-25.06` 度，连续两次达到 `+57.09/-55.33` 度，
均满足 5 度完成容差。

```bash
ros2 topic echo /vln/uninavid/raw_action
```

2026-08-24 最终复测：官方 36 帧样例推理成功；Hospital 无行人 `dynanav_001` 技术闭环运行
40.04 秒，但该较长回合为 `success=false`（最终误差 11.197 m）。同日新增的
`simple_forward_3m_standard` 真实闭环为 `success=true`（最终误差 1.488 m、SPL 1.0、
零碰撞）。首帧 RGB 已检查为方向正常的 640x480、HFOV 120 度图像。简单回合成功不能
替代完整 DynaNav 回合成绩，两类结果必须分别报告。

### OmniVLA

这里的 OmniVLA 固定指完整 `NHirose/OmniVLA` `omnivla-original` step 120000，
不是 OmniVLA-edge。模型只接收当前 RGB 和原始语言指令，强制使用 language-only
`modality_id=7`，输出完整 `8x4 [x,y,heading_x,heading_y]` 局部轨迹。

终端 1，在独立 Python 3.10 环境中启动模型服务：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
OMNIVLA_ALLOW_BLACKWELL_COMPAT=1 \
OMNIVLA_PYTHON=/home/mifcom2/miniconda3/envs/omnivla/bin/python \
  ./src/vln_adapters/omnivla_adapter/run_inference_server.sh
```

终端 2，启动与其他 RGB-only 模型相同的 DynaNav、相机、Pure Pursuit 和 SRU ONNX：

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

本机已下载完整 `omnivla-original` 权重到 `checkpoints/vln/omnivla/omnivla-original`，
SHA256 清单位于该目录的 `SHA256SUMS`。独立 `omnivla` 环境采用 Python 3.10.20、
NumPy 1.26.4、Torch 2.7.1+cu128、TorchVision 0.22.1+cu128 和上游指定的
Transformers fork 4.40.1；没有污染 Isaac 或 ROS 环境。上游固定 Torch 2.2.0 不支持
RTX 5090，因此本机启动时必须显式设置 `OMNIVLA_ALLOW_BLACKWELL_COMPAT=1`。

2026-08-24 实测：官方静态 RGB 真推理输出有效 `8x4`，普通单次推理 0.118 秒，峰值分配
显存 16069447680 bytes；language-only 伪目标泄漏检查最大差异 0.0，通过。结果保存在
`outputs/omnivla/static_inference/static_001.json`。同一 `dynanav_001` 无行人回合技术闭环
运行 40.02 秒，模型持续以约 2.86 Hz 输出轨迹，但导航结果为 `success=false`：最终误差
6.961 m、路径 7.456 m、静态接触 61 次。因此没有继续动态行人回合，不能把本次结果写成
导航成功。完整结果见 `outputs/omnivla/single_episode/dynanav_001.json`。

下载方法、完整文件检查、stub 接口测试和调试话题见
[OmniVLA adapter 文档](src/vln_adapters/omnivla_adapter/README.md)。stub 只验证统一链路，
不能作为 OmniVLA 评测结果。

### 公平对比配置

权威清单位于 `configs/fair_comparison.yaml`。统一 launch 默认
`comparison_track:=auto`，会根据 `configs/models.yaml` 自动选择并校验：

```text
rgb_only:                  TIC-VLA、NaVILA、Uni-NaVid、OmniVLA
rgb_d:                     DualVLN
debug:                     dummy（不进入正式统计）
```

公平轨道固定同一 B2-W、DynaNav episode、相机安装位姿、`640x480@10 Hz`、共享
SRU ONNX 和 `1.0 m/s`。轨迹模型共享 Pure Pursuit 与 `trajectory_normalized`。运行中修改速度会被拒绝。
每条结果会保存 `comparison_track` 和 `model_inputs`，RGB-D 结果不得与 RGB-only 合并。
episode 的起点、目标、指令、种子、人物数量、最大时长和成功阈值始终以
`src/simulation_bridge/dynanav_bridge/config/episodes.yaml` 为准；同一 episode 对所有模型相同。

### TIC-VLA 官方 benchmark 模式

85 个 DynaNav 回合已按 TIC-VLA 官方 `benchmark_full.yaml` 导入。成功标准与官方一致：
使用 ground-truth XY 距离，严格满足 `distance < success_threshold`（默认 `1.5 m`），并使用
每回合自己的仿真超时；不会因模型主动 STOP、碰撞或接口状态而放宽目标判定。工程仍使用
B2-W + SRU-ONNX 执行器，官方配置中的 Nova Carter 仅作为上游参考机器人。

五个模型的执行参数直接保存在 `configs/models.yaml` 各模型条目的 `execution` 中，
没有额外的执行配置文件。`official` 子项记录上游代码能够直接核对的参数，
`desired_speed` 与 `navigation` 是当前 B2-W 实际采用的折中参数：

模型原生高层模式使用 `--native`。它保留统一的场景、成功判定、评估器、`/nav_vel`
接口和 B2-W SRU-ONNX 低层执行器，只替换模型输出到速度之间的高层执行方法：

| 模型 | 原生高层执行 |
|---|---|
| TIC-VLA | 官方 Spot DynaNav 曲率公式，由 adapter 直接输出速度；B2-W 限幅 1.0 m/s |
| OmniVLA | 官方 waypoint-4 PD 速度换算 |
| DualVLN | 官方模型轨迹 + B2-W 共享 Pure Pursuit；官方 MPC 因缺少 CasADi/IPOPT 暂不可用 |
| NaVILA | 官方离散距离/转角语义，经 B2-W 局部路径执行 |
| Uni-NaVid | 官方多步离散动作语义，经 B2-W 局部路径执行 |

因此 `--native` 是“原生高层 + 统一 B2-W 低层”，不是重新启用 Carter、Go2 或 Spot
的车轮/关节控制器。结果会记录 `high_level_controller`、`controller_fidelity`、
`model_native_high_level` 和 `uses_shared_path_follower`，不得与公平轨道混合统计。

各模型官方源码、固定 commit、文件 SHA256 以及 B2-W 适配差异见
[`docs/OFFICIAL_MODEL_AUDIT.md`](docs/OFFICIAL_MODEL_AUDIT.md)。其中没有官方连续控制器的
模型会明确标记，不能把适配参数称作官方参数。

正式模型评测前，分别验证五条控制器链路（直行 + 左转，共 10 回合）：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
./scripts/run_controller_acceptance.sh
```

这个测试不加载 VLN 大模型，只验证高层转换、navigation bridge、B2-W SRU-ONNX 和
Isaac Sim 物理执行。结果保存在 `outputs/controller_probe_*/controller_acceptance/`，
不会写入正式模型 benchmark 目录。

| 模型 | 期望速度 | 线速度上限 | 角速度上限 | 线/角加速度上限 | HFOV | 依据 |
|---|---:|---:|---:|---:|---:|---|
| TIC-VLA | 1.0 m/s | 1.0 m/s | 1.0 rad/s | 2.0 / 3.0 | 90° | 官方 Spot 为 1.5/1.0，B2-W 仅将线速度裁到已验证 1.0；减速度 2.5/3.5 |
| OmniVLA | 0.3 m/s | 0.3 m/s | 0.3 rad/s | 2.0 / 3.0 | 90° | 官方 waypoint-4 PD 最终限幅 |
| DualVLN | 0.3 m/s | 0.4 m/s | 0.4 rad/s | 0.8 / 1.5 | 79° | 官方实机 MPC 默认值 |
| NaVILA | 0.5 m/s | 0.5 m/s | 0.6 rad/s | 0.8 / 1.5 | 90° | 官方 0.25 m/15° 离散动作；连续速度为 B2-W 执行值 |
| Uni-NaVid | 0.5 m/s | 0.5 m/s | 1.0 rad/s | 0.8 / 1.5 | 90° | 官方 0.5 m/30° 离散动作；连续速度为 B2-W 执行值 |

仿真发布分辨率仍统一为 `640x480@10 Hz`。TIC-VLA 上游 DynaNav 的
`1920x1080@90°`、NaVILA Habitat 的 `512x512@90°` 以及模型内部 224/384 输入尺寸
仅作为 `execution` 中的参考元数据记录，不会冒充当前仿真实际分辨率。

运行完整官方条件（包含官方行人数量和 Outdoor）：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
./scripts/run_dynanav_full85_all.sh --official --no-resume \
  ticvla omnivla dualvln navila uninavid
```

官方模式会自动覆盖本地快捷默认值：`DYNANAV_NO_PEDESTRIANS=0`、
`DYNANAV_PEDESTRIAN_CAP=0`、不排除任何场景。要显示 Isaac Sim 窗口，在命令中加入
`--gui`；不加时沿用 `DYNANAV_HEADLESS`（默认无界面）。默认脚本仍是无行人、排除
Outdoor 的快速 smoke 模式，两种结果必须分开报告。

如果暂不运行 Outdoor、也不加载行人，但仍希望应用上述每模型参数：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
export DYNANAV_NO_PEDESTRIANS=1
export DYNANAV_EXPERIMENT=native_nonoutdoor
./scripts/run_dynanav_full85_all.sh --native --no-outdoor --no-resume \
  ticvla omnivla dualvln navila uninavid
```

### 其他入口

```bash
# 键盘 B2-W
ros2 launch robot_bringup keyboard_b2w.launch.py

# 已完整验证的 dummy 单回合
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=dummy episode:=hospital_001 \
  evaluation_mode:=trajectory_normalized sensor_profile:=rgb_only

# 先只检查 85 个回合的配置和启动命令，不启动 Isaac Sim
ros2 launch robot_bringup dynanav_benchmark.launch.py \
  model:=dummy experiment:=dry_validation episodes:=all dry_run:=true

# 确认单回合无误后，才顺序运行 85 个回合（每条使用独立子进程）
ros2 launch robot_bringup dynanav_benchmark.launch.py \
  model:=dummy experiment:=benchmark episodes:=all

# 五个 VLN 模型自动依次运行；默认排除 Outdoor，因此每个模型运行
# Hospital 25 + Office 25 + Warehouse 25 = 75 个官方回合
# 顺序：TIC-VLA、OmniVLA、DualVLN、NaVILA、Uni-NaVid
./scripts/run_dynanav_full85_all.sh

# 只测试其他四个模型，不重复 TIC-VLA（同样默认排除 Outdoor）
./scripts/run_dynanav_full85_all.sh omnivla dualvln navila uninavid

# 先用同一个简单无行人回合比较其他模型，确认推理/控制链路后再跑75回合
DYNANAV_EPISODES=simple_forward_3m_standard \
  DYNANAV_EXPERIMENT=simple_nonoutdoor_20260829 \
  DYNANAV_NO_PEDESTRIANS=1 \
  ./scripts/run_dynanav_full85_all.sh --no-resume omnivla dualvln navila uninavid

# 只运行指定模型（示例）
./scripts/run_dynanav_full85_all.sh dualvln navila

# 从已有 JSON 续跑；默认会跳过已经完成的回合
DYNANAV_EXPERIMENT=dynanav_full85_final \
  ./scripts/run_dynanav_full85_all.sh

# 五个模型全部重新续跑，但所有回合强制关闭动态行人
DYNANAV_NO_PEDESTRIANS=1 \
  ./scripts/run_dynanav_full85_all.sh

# 需要恢复 Outdoor 时显式清空筛选（不建议与本轮结果混用）
DYNANAV_EXCLUDE_SCENES="" ./scripts/run_dynanav_full85_all.sh

# 查看脚本参数
./scripts/run_dynanav_full85_all.sh --help
```

全量脚本默认启用 Isaac Sim 显存保护：将单回合动态行人数量限制为 40，避免
OmniVLA/TIC-VLA 与 Isaac Sim 渲染同时占满 32GB GPU。限制值会写入回合日志；如需
严格使用 `episodes.yaml` 中的官方行人数，可设置 `DYNANAV_PEDESTRIAN_CAP=0`，但
高行人密度场景可能再次触发 `ERROR_DEVICE_LOST`。

脚本不再使用全局锁文件；每次启动都由操作者自行确认没有其它 Isaac Sim 或模型服务占用
GPU。若 Isaac Sim 发生 GPU 崩溃，benchmark 会检测回合日志中的 `ERROR_DEVICE_LOST`
或显存错误并立即回收该回合的整个进程组，不再等待完整超时。不要并行启动多个模型服务，
否则可能发生显存冲突。

脚本每次只在 GPU 上保留一个 VLN 推理服务和一个 Isaac Sim 进程。每个模型完成所选回合后，
脚本会发送中断并释放模型显存，再启动下一个模型。单回合结果位于
`outputs/<model>/<experiment>/`，模型和 benchmark 日志位于 `logs/`。脚本默认开启续跑和
单模型内部错误后继续；若某个模型的服务无法启动，会记录为基础设施失败并继续其他模型。

### 2026-08-29 三场景运行前 smoke 基线

### 显示 Isaac Sim 界面

批量脚本默认使用无界面模式。若要在测试时看到 Isaac Sim 窗口，可在启动时设置
`DYNANAV_HEADLESS=0`（或传入 `--gui`/`--no-headless`）。脚本仍会按顺序运行模型，
每次只打开一个 Isaac Sim 实例：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
export DYNANAV_NO_PEDESTRIANS=1
export DYNANAV_EXCLUDE_SCENES=outdoor
export DYNANAV_EXPERIMENT=simple_vending_gui_20260829

./scripts/run_dynanav_full85_all.sh \
  --episodes simple_vending_machine \
  --no-resume \
  --gui \
  ticvla omnivla dualvln navila uninavid
```

如果只想观察一个模型，建议先运行单模型，窗口更容易观察：

```bash
./scripts/run_dynanav_full85_all.sh \
  --episodes simple_vending_machine \
  --no-resume \
  --gui \
  ticvla
```

图形界面需要在本机桌面会话中运行；远程 SSH 或无 DISPLAY 环境仍然无法显示窗口。

已使用同一组无行人回合验证五个模型：Hospital=`simple_forward_3m_standard`、
Office=`dynanav_026`、Warehouse=`dynanav_061`，命令如下：

```bash
DYNANAV_NO_PEDESTRIANS=1 \
DYNANAV_EPISODES=simple_forward_3m_standard,dynanav_026,dynanav_061 \
DYNANAV_EXCLUDE_SCENES=outdoor \
DYNANAV_ASSET_VERSION=5.0 \
DYNANAV_EXPERIMENT=scene_smoke_20260829 \
./scripts/run_dynanav_full85_all.sh --no-resume
```

本次 15 个回合全部启动并写入结果，`infrastructure_failures=0`；因此五个模型均达到
“可运行”标准。导航成功情况如下，失败原因保留在每个 JSON 的
`failure_attribution` 字段中：

| 模型 | Hospital | Office | Warehouse |
| --- | --- | --- | --- |
| TIC-VLA | 失败（超时/接触） | 失败（超时/接触） | 失败（超时/接触） |
| OmniVLA | 成功 | 失败（超时/接触） | 失败（超时/接触） |
| DualVLN | 成功 | 失败（超时） | 失败（超时/接触） |
| NaVILA | 成功 | 失败（超时/接触） | 失败（超时） |
| Uni-NaVid | 失败（超时/接触） | 失败（超时/接触） | 失败（超时/接触） |

结果目录为 `outputs/<model>/scene_smoke_20260829/`，汇总文件为
`outputs/scene_smoke_20260829/summary.csv`。这组结果证明的是模型服务、相机、统一接口、
路径跟踪器和 SRU-ONNX 控制器均可运行；Office/Warehouse 的导航失败仍属于模型视觉域、
场景语义和轨迹控制表现，不能在未单独定位前直接宣称模型已通过完整 benchmark。

本轮“模型泛化”判断规则：JSON 中 `success=false` 只有在回合完整启动、模型服务就绪、
RGB/控制话题持续、SRU-ONNX 实际执行且没有基础设施错误时，才作为模型视觉域/语义泛化
失败分析。服务启动失败、CUDA/OOM、Isaac 崩溃、话题断开、控制器 stuck 或回合未落盘，
只记入 benchmark 日志的 `infrastructure_failures`，不计入模型成功率。所有模型使用相同的
Hospital/Office/Warehouse 回合、RGB-only 640x480 相机、1.0 m/s 期望速度、共享 Pure Pursuit、
SRU-ONNX 和无行人设置；Outdoor 暂不纳入本轮比较。

### 2026-08-29 其他模型无行人 smoke test

在关闭原 TIC-VLA 服务后，使用同一个 Hospital 回合 `simple_forward_3m_standard`、
RGB-only `640x480`、1.0 m/s、共享 Pure Pursuit、SRU-ONNX 和无行人设置，依次实际测试了
OmniVLA、DualVLN、NaVILA、Uni-NaVid。Outdoor 没有参与本轮测试。

| 模型 | 结果 | 最终误差 | 原因归类 | 备注 |
|---|---:|---:|---|---|
| OmniVLA | SUCCESS | 1.499 m | success | 4.30 s，0 碰撞，平均推理 0.144 s |
| NaVILA | SUCCESS | 1.491 m | success | 3.60 s，0 碰撞，平均推理 0.608 s |
| DualVLN | FAIL | 2.101 m | inconclusive_timeout | 45.16 s，0 碰撞，未收到有效推理延迟 |
| Uni-NaVid | FAIL | 4.226 m | inconclusive_timeout | 45.18 s，4 次静态接触 |

四个 benchmark 子进程均正常落盘，日志均为 `infrastructure_failures=0`。DualVLN 需要
单独排查服务请求/响应，Uni-NaVid 需要排查静态接触后的轨迹恢复；这两项暂时不能算作
纯视觉域泛化失败。只有无碰撞、无 stuck、服务持续返回动作并在目标外主动 STOP，才会
在新 JSON 中标记为 `failure_attribution=model_behavior`。

结果目录：`outputs/<model>/simple_nonoutdoor_20260829/`；汇总日志：
`logs/<model>_simple_nonoutdoor_20260829.log`。

### CUDA 显存不足

DualVLN 常驻约 `16-18 GiB` 显存，Hospital Isaac Sim 约需 `8-9 GiB`。同一时刻只运行一个
模型服务和一个 Isaac Sim。若出现 `Failed to create primary CUDA context`、
`Warp CUDA error 2` 或 `ERROR_OUT_OF_DEVICE_MEMORY`，先在另一个终端检查：

```bash
nvidia-smi
pgrep -af 'inference_server.py|play.py|isaac-sim'
```

先在旧 Isaac Sim/测试终端按 `Ctrl+C` 正常退出，再重新执行本节启动命令。不要重复启动
模型服务。`play.py` 已使用 `finally` 关闭 SimulationApp，即使场景初始化异常也会释放
Kit 后台线程和 CUDA 显存。

### Isaac Sim 中显示轨迹并手动发送指令

launch 中的 `episode:=...` 决定模型收到哪条语言指令。指令定义在
`src/simulation_bridge/dynanav_bridge/config/episodes.yaml`。例如：

```text
episode:=hospital_001
instruction="Walk forward along the corridor and stop at the target."

episode:=dynanav_001
instruction="Go straight ahead and stop at the vending machine at front."
```

启动终端会立即打印 `Selected <episode>: instruction='...'`，场景就绪后还会打印一次
`Started <episode>: instruction='...'`。运行中也可在已加载 ROS 2 工作区的新终端查询当前
模型实际收到的指令：

```bash
ros2 topic echo /vln/instruction std_msgs/msg/String \
  --qos-durability transient_local --once
```

带窗口启动时，统一 ROS 2 入口会自动在 Isaac Sim viewport 绘制：

```text
青色大点：VLN/VLA 当前预测的世界坐标轨迹
黄色小点：B2-W 实际走过的轨迹，最多保留 500 个点
```

预测路径从 `/vln/debug_path` 经独立 UDP `5823` 进入 Isaac 进程，不与 `/nav_vel`
控制通道混用。STOP、切换到 keyboard、episode reset/finish 会清除预测路径；reset 同时
清除实际轨迹。`headless:=true` 时不创建 marker。

运行中在另一个 ROS 2 终端发送自己的指令：

```bash
source /opt/ros/jazzy/setup.bash
source /home/mifcom2/b2w/robot_vln_ws/install/setup.bash

ros2 topic pub --once /vln/instruction std_msgs/msg/String \
  "{data: 'Go straight ahead and stop at the vending machine.'}"

# 空指令会清除模型历史和旧轨迹并立即停车
ros2 topic pub --once /vln/instruction std_msgs/msg/String "{data: ''}"
```

手动指令只改变模型行为，不改变 episode 的评测目标坐标；两者目标不一致时，最终
`success` 没有比较意义。

当单回合已经保存为 `SUCCESS` 或 `FAILED`、但使用了 `shutdown_after_finish:=false` 时，
DualVLN/TIC-VLA/NaVILA/Uni-NaVid 的具体行为由各 adapter 决定。NaVILA 和 Uni-NaVid 收到新的非空指令会自动恢复
手动演示模式，无需再次发布 `/episode/state START`，且不会覆盖已经保存的回合结果。

运行中修改轨迹跟踪速度：

```bash
# 设置 2.0 m/s；允许范围为 0 < speed <= 2.0
ros2 topic pub --once /navigation/desired_speed std_msgs/msg/Float32 "{data: 2.0}"

# 查询当前设置
ros2 topic echo /navigation/desired_speed/current --once
```

公平对比默认锁定 `1.0 m/s`，上述修改会被拒绝。需要调速或手动演示时，应在启动命令增加：

```text
comparison_track:=none
```

此类结果会标记为 `untracked`，不能进入公平对比汇总。

单回合 launch 明确使用 `sru-onnx`，默认 `desired_speed:=1.0`、
`shutdown_after_finish:=false`。到达评测目标后
仍会立即停车并保存 JSON/CSV，但不会关闭 Isaac Sim。此后可继续发布新指令或新速度做
演示；已经保存的回合成绩不会被覆盖。需要恢复自动退出时添加
`shutdown_after_finish:=true`。批量 benchmark 始终强制自动退出。

`2.0 m/s` 是运行时可选上限，超出当前 locomotion 已验证的 `1.0 m/s` 默认速度，属于仿真实验设置。Pure Pursuit
转弯时会自动降低前进速度，线加速度仍限制为 `0.8 m/s^2`，所以 `/nav_vel` 和机器人实际
速度不会在每个时刻都等于 `2.0 m/s`。

2026-08-21 实际 dummy 闭环结果：起点 `(8,0)`、目标 `(11,0)`，路径 `2.482 m`、
仿真时长 `5.44 s`、最终误差 `0.490 m`、SPL `1.0`、碰撞和卡住为 0。结果见
`outputs/dummy/acceptance_v3/hospital_001.json`。DualVLN 真实服务和统一适配器已完成连续
HTTP 推理与结果落盘；当前 `hospital_001` 的泛化指令使模型主要输出右转，60 秒未到目标，
因此该次结果正确记录为失败。使用明确售货机目标的 `simple_vending_machine` 已成功，最终
误差 `1.495 m`、SPL `1.0`、零碰撞。TIC-VLA 的两个官方权重已加载，并完成真实离线 `30x2`
推理和同一个 `hospital_001` 连续滚动轨迹闭环。闭环移动 `16.478 m`、运行 `60.12 s`、
最终误差 `18.355 m`，未达到成功阈值，按真实表现记录为失败；结果见
`outputs/ticvla/ticvla_fixed/hospital_001.json`。适配器严格采用官方 FLU 本体坐标系累计
XY 轨迹，不做轴交换或二次 `cumsum`，也没有注入 episode 目标坐标。该回合记录了
88 次小腿 link 原始接触事件；当前 net-force API 无法识别对方物体，必须先校准接触过滤，
不能直接把这个计数解释为 88 次障碍碰撞。

DynaNav 官方 85 个回合已完整转换为 Isaac Sim 5.1 配置：医院 25、办公室 25、
室外 10、仓库 25。四类场景均已实际加载动态人物并验证 RGB-D/odom，1 名人物在各场景
2.4 秒测试中均移动 `1.920 m`。动态医院完整 dummy 回合成功，最终误差 `0.497 m`。
Isaac Sim 5.0 的 People/NavMesh 扩展会破坏当前 5.1 引用式场景的渲染 stage，因此默认
使用确定性的轻量运动学兼容人物，并保留 DynaNav 的随机种子、目标点和命令生成规则；
只有显式设置 `ROBOT_VLN_TRY_ISAAC_PEOPLE=1` 才尝试旧扩展。室外场景仍有若干旧版
Rivermark 装饰资产引用缺失，但核心场景、RGB-D、odom 和动态人物均通过实际测试。

85 回合 dry-run 已验证总数和场景分布，并验证了一个由 benchmark runner 启动、结束、
汇总的真实子进程。完整 85 回合实验尚未执行，避免在模型单回合尚未成功时消耗大量时间。

NaVILA 已完成官方 8 帧 checkpoint 的真实离线推理和无行人闭环。离线原始文本为
`The next action is move forward 75 cm.`，推理耗时 `0.880 s`，CUDA 峰值
`18,722,249,728 bytes`。较长的 `dynanav_001` 仍为失败；简单回合
`simple_forward_3m_standard` 已成功，最终误差 `1.494 m`、SPL `1.0`、零碰撞。结果分别
保存在 `outputs/navila/single_episode` 和 `outputs/navila/simple_success`，两类回合不得混合
汇总。尚未继续运行动态行人回合。

```text
robot_vln_ws/
├── src/
│   ├── robot_bringup/       # 机器狗与整套系统启动
│   ├── robot_description/   # B2-W、Lite3、机械臂描述
│   ├── robot_controller/    # 最底层运动控制
│   ├── vln_interfaces/      # ROS 2 统一导航消息（权威接口）
│   ├── vln_adapters/        # dummy、DualVLN、TIC-VLA、NaVILA 适配器
│   ├── navigation_bridge/   # VLN 输出到机器人控制
│   ├── simulation_bridge/   # DynaNav / Isaac Sim 5.1 兼容层
│   ├── sensors/             # 相机、深度与雷达
│   └── evaluation/          # 统一测试场景与指标
├── third_party/             # RobotLab 和原始 VLN 开源仓库
├── checkpoints/             # 模型权重，按模型隔离
├── configs/                 # 跨模块公共配置
└── README.md
```

## 固定架构

```text
VLN/VLA 模型
      |
      | 模型专用输出
      v
vln_adapters/<模型>
      |
      | /vln/command (NavigationCommand)
      v
navigation_bridge（导航桥）
      |
      | /nav_vel (Twist: vx, vy, yaw_rate)
      v
robot_controller（机器人控制器）
      |
      | 腿部 + 轮子动作
      v
Unitree B2W
```

模型与机器人控制必须解耦：

- VLN/VLA 模型不得导入 RobotLab、加载 B2W policy 或发送关节动作。
- 每个模型只增加轻量 ROS 适配器，模型输出差异由 `src/vln_adapters` 处理。
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

消息定义见 [src/vln_interfaces/msg/NavigationCommand.msg](src/vln_interfaces/msg/NavigationCommand.msg)，字段和话题约定见 [src/vln_interfaces/README.md](src/vln_interfaces/README.md)。

目录职责：

- `vln_adapters/<model>_adapter/` 只处理模型输入、推理服务通信和统一消息封装。
- `vln_interfaces/` 定义所有模型共同使用的 ROS 2 消息契约。
- `navigation_bridge/` 负责路径跟随和安全过滤，不放模型代码。
- `robot_controller/` 负责把统一速度命令接到 B2W 移动控制策略。
- `third_party/robot_lab/` 保存完整 RobotLab 仓库及其 Git 历史。
- `third_party/unitree_ros/` 保存 Unitree 官方仓库，用于 B2W 几何和传感器 frame 参考。
- `third_party/InternNav/`、`third_party/NaVILA/` 等目录供后续放入原始 VLN 仓库。
- `checkpoints/b2w_locomotion/` 保存已经验证过的 B2W 移动控制策略。

## 接入状态

| 模块 | 状态 |
|---|---|
| B2W SRU Gazebo `sru-onnx` 移动控制器 | 可用（默认） |
| B2W SRU `isaac-pt` 移动控制器 | 可用（保留） |
| B2W RobotLab 移动控制器 | 可用（保留） |
| 医院场景键盘/目标点/移动验收 | 可用 |
| ROS 2 `vln_interfaces/NavigationCommand.msg` 契约 | 可用 |
| dummy -> 共享跟踪器 -> SRU ONNX B2-W | 可用（已实际运行单回合） |
| DualVLN ROS 适配器 | 可用（真实推理链路；`simple_vending_machine` 成功） |
| TIC-VLA ROS 适配器 | 可用（官方 DynaNav async/KV cache，真实 30x2；当前 B2-W 场景泛化仍失败） |
| DynaNav 医院场景 5.1 兼容层 | 可用（动态 dummy 完整回合通过） |
| DynaNav 办公室/室外/仓库场景 | 可用（动态人物、RGB-D、odom 实测） |
| DynaNav 官方 85 回合导入 | 可用（25/25/10/25，dry-run 通过，未完整执行） |
| DualVLN 模拟 RGB-D/里程计输入 | 可用 |
| DualVLN HTTP 推理服务 | 可用 |
| 轨迹适配器 | 可用 |
| 路径跟踪器 + 安全过滤器 | 可用 |
| `[vx, vy, yaw_rate]` 到 B2W 策略的桥接 | 可用 |
| DualVLN 医院场景 40 秒闭环 | 可用 |
| NaVILA ROS 适配器 | 可用（真实 8 帧推理；`simple_forward_3m_standard` 成功） |
| Uni-NaVid ROS 适配器 | 可用（真实在线缓存推理；`simple_forward_3m_standard` 成功） |

### 最近闭环修复（2026-08-29）

- Isaac Sim RGB、Depth、CameraInfo 和 B2-W odometry 统一使用仿真时间戳；TIC-VLA 的
  `sim_step`、历史帧和运动状态不再混入墙钟时间。
- 评测 watchdog 只有在仿真时钟真正停止时才使用墙钟兜底，不会因为 Isaac Sim 低于实时速度
  提前结束回合。
- 共享 Pure Pursuit 的连续轨迹取点方式与官方 TIC-VLA DynaNav（1 m 弧长、`2..T-3`）一致，
  仍统一输出到 `/nav_vel`，底层固定为 B2-W SRU-ONNX。
- `colcon build --symlink-install` 六个相关包通过，ROS/适配器/导航回归测试 `32 passed`。
- 实测 `dynanav_026` Office 已完整运行 50 仿真秒后才记录失败（最终误差 11.442 m）；
  `hospital_001` 已完整运行 60.04 仿真秒后记录失败（最终误差 6.502 m）。两次均无提前
  watchdog、无碰撞，推理延迟 P95 约 0.10 s。失败原因是模型在当前 B2-W 相机视角下选择了
  错误走廊，不能通过修改目标点或把模型 STOP 伪装成成功来规避。
- 对照回合 `simple_forward_3m_standard` 使用完全相同的 TIC-VLA、相机、共享跟踪器和
  SRU-ONNX 执行器成功：仿真时长 `11.28 s`、最终误差 `1.500 m`、结果 `SUCCESS`。
  这证明控制链路可用；Office/Hospital 剩余问题属于官方复杂指令在 B2-W 视觉域中的
  泛化/相机输入问题。应先完成相机视场与图像纵横比校准，再重测官方回合，不应修改
  ground-truth 目标或放宽成功判定。
- TIC-VLA adapter 现支持 `input.model_image_aspect_ratio` 中心裁剪实验。实测将
  `640x480` 直接裁为 `640x360` 后，简单直行回合反而失败（误差 `2.455 m`），因此默认
保持原生 `640x480`，避免破坏已验证链路；要测试 16:9 请显式设置该参数并单独记录结果。

原生配置的官方依据不再依赖人工说明：`configs/models.yaml` 保存每个上游的完整 commit、
源文件 SHA256、控制器可用性和 B2-W 适配差异；benchmark 启动前会校验这些值。官网或官方
GitHub 没有公布的连续速度不会标成官方参数。NaVILA 与 Uni-NaVid 的 `0.5 m/s` 是 B2-W
执行参数，官方只定义了离散距离/转角。DualVLN 当前也不会把任何自写求解器标成官方 MPC。
- 批量评测脚本现默认跳过 Outdoor，只选择官方 85 条中的 Hospital/Office/Warehouse
  共 75 条；可用 `DYNANAV_EXCLUDE_SCENES=""` 恢复完整 85 条，或用
  `DYNANAV_EPISODES=simple_forward_3m_standard` 先做所有模型的同回合 smoke test。
- 评估 JSON 新增 `failure_attribution`。`success` 表示到达目标；模型在目标外主动 STOP
  标记为 `model_behavior`；碰撞或 stuck 分别标记为 `collision_or_scene_contact`、
  `controller_stuck`；墙钟兜底超时标记为 `inconclusive_timeout`。只有前两类之外的
  `model_behavior` 才能作为视觉域/语义泛化失败分析，基础设施失败仍只计入 benchmark
  日志的 `infrastructure_failures`。
| Evolve-Nav / Aware-VLN | 仅保留扩展接口 |

“可用”只表示已经有实际代码或验证结果；占位脚本不会标为完成。

默认控制器是从官方 Gazebo 部署迁入的 SRU `sru-onnx`。SRU `isaac-pt` 和原 RobotLab
控制器均未删除。切换命令：

```bash
# 默认：Gazebo SRU ONNX，60 维观测 / 16 维动作 / LSTM 256
./src/robot_controller/scripts/run_b2w_hospital.sh goal

# 保留方案：SRU Isaac TorchScript，60 维观测 / 16 维动作
B2W_LOCOMOTION_POLICY=isaac-pt ./src/robot_controller/scripts/run_b2w_hospital.sh goal

# 保留方案：RobotLab，57 维观测 / 16 维动作
B2W_LOCOMOTION_POLICY=robotlab ./src/robot_controller/scripts/run_b2w_hospital.sh goal
```

`sru-onnx` 与 Gazebo 控制器保持相同的 50 Hz 推理、60 维输入顺序、16 维输出、
256 维 hidden/cell、关节顺序以及 `0.5/5.0` action scaling。Isaac Sim 与 Gazebo 的物理
引擎不同，因此状态轨迹不会逐帧完全一致。2026-08-21 的 Hospital 40 秒实测中，前进
`0.520 m/s`、后退 `-0.493 m/s`、左转 `0.534 rad/s`、右转 `-0.516 rad/s`、停止和连续
稳定均通过；直立率 `100%`，最低高度 `0.704 m`。

DualVLN 仿真闭环已在 Isaac Sim 5.1 医院场景中实际验证：输入为 `640x480` 对齐 RGB-D、相机内参和采集时刻 `[x,y,yaw]`；模型返回 `33x2` 本体坐标系轨迹，经适配器与安全跟踪器写入 B2W 策略的 `[vx,0,yaw_rate]`。40 秒测试位移 `8.081 m`、直立率 `100%`、最低高度 `0.606 m`，模型最终返回 `STOP` 并持续输出零速度。

## 已迁入的 B2W 内容

```text
src/robot_controller/scripts/rsl_rl/play.py  医院场景、键盘、目标点和移动验收入口
src/robot_controller/scripts/rl_utils.py     Isaac Sim 跟随相机辅助函数
src/robot_controller/config/b2w.yaml         task、维度、命令范围和关节清单
src/robot_description/urdf/                  B2W URDF
src/robot_description/meshes/                B2W 网格模型
checkpoints/b2w_locomotion/model_2600.pt      已验证的移动控制权重
checkpoints/b2w_locomotion/sru_onnx/          Gazebo 部署 ONNX（默认）
checkpoints/b2w_locomotion/isaac_pt/          SRU Isaac 循环策略
third_party/sru-navigation-sim/                SRU B2W USD（未修改）
```

快速启动：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
./src/robot_controller/scripts/run_b2w_hospital.sh goal
```

启动脚本会在需要时自动从 `(base)` 激活 `isaaclab232`，并加载 Isaac Sim 环境变量。
可以用 `./src/robot_controller/scripts/run_b2w_hospital.sh env-check` 单独检查环境。

如果启动前看到 `NVIDIA GPU is unavailable in this shell`，表示当前终端看不到 NVIDIA
设备，不是模型或 B2-W 控制器错误。先在宿主机终端确认：

```bash
nvidia-smi
ls -l /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm
```

宿主机若 `nvidia-smi` 也失败，需要由管理员恢复 NVIDIA 驱动/设备节点（例如重启后再检查，
或按本机驱动维护流程重新加载 `nvidia`、`nvidia_uvm` 模块）。如果宿主机正常而 VS Code/Codex
终端失败，则是该终端的容器或沙箱没有映射 GPU；请在能运行 `nvidia-smi` 的宿主终端启动，
不要重装 CUDA、PyTorch、Isaac Sim 或 Isaac Lab。

详细操作见 [src/robot_controller/README.md](src/robot_controller/README.md)。

完整 `robot_lab` 已移动到工作区内部的 `third_party/robot_lab`，作为原始 task 注册和 Isaac Lab 环境依赖。B2W 启动脚本会自动把它的 `source/robot_lab` 加入 `PYTHONPATH`。

Unitree 官方 `unitree_ros` 已克隆到 `third_party/unitree_ros`，参考 commit 为 `daadf41ee9afce8f90fdc09a98506012691fa122`。官方 B2W URDF 与 RobotLab 当前使用的 B2W URDF 哈希一致；第三方版本与更新规则见 `third_party/README.md`。

## DualVLN 仿真闭环

当前阶段只运行 Isaac Sim，不启动 ROS2、Unitree SDK、RealSense 驱动、LiDAR、SLAM、Nav2 或机械臂：

```text
Isaac RGB-D + 指令 + 采集时刻里程计
                    -> 独立 DualVLN 服务
                    -> 33x2 本体坐标系轨迹
                    -> 轨迹适配器
                    -> 路径跟踪器 + 安全过滤器
                    -> [vx, 0, yaw_rate]
                    -> RobotLab B2W 移动控制策略
```

运行命令见 `src/vln_adapters/dualvln_adapter/README.md`。完整 40 秒视频位于 `checkpoints/b2w_locomotion/videos/play/rl-video-step-0.mp4`。

## OmniVLA 两种执行模式

`omnivla` 保留公平评测所需的 `trajectory_normalized`：8 个模型轨迹点交给共享
Pure Pursuit；`omnivla_native` 单独复现官网的 waypoint-4 速度换算（`0.3 m/s`、
`0.3 rad/s`），之后仍通过同一个 `/nav_vel` 和 SRU-ONNX B2-W 低层执行器。两种结果
不能合并统计。

公平轨迹模式：

```bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=omnivla episode:=hospital_001 comparison_track:=rgb_only \
  sensor_profile:=rgb_only evaluation_mode:=trajectory_normalized \
  goal_profile:=language_only desired_speed:=1.0 shutdown_after_finish:=false
```

官方 native 模式：

```bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=omnivla_native episode:=hospital_001 comparison_track:=none \
  sensor_profile:=rgb_only evaluation_mode:=native_output \
  goal_profile:=language_only desired_speed:=0.3 execution_profile:=native \
  shutdown_after_finish:=false
```

两种模式都只使用 RGB 和语言指令作为 OmniVLA 输入；`RGB-only` 表示模型输入为
`RGB + instruction`，不是取消系统中供控制器/评估器使用的 odom。

## DualVLN 实机接口（当前不运行）

下面保留的硬件映射只供以后参考，不属于当前仿真启动链。

| 数据/接口 | 工作区配置 | 用途 |
|---|---|---|
| 前向 RGB | `src/sensors/camera/config/b2_front_rgb.yaml` | DualVLN 观测输入 |
| 自然语言指令 | `/vln/instruction` | DualVLN 指令输入 |
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
src/vln_adapters/dualvln_adapter/dualvln_adapter/inference_server.py
src/vln_adapters/dualvln_adapter/dualvln_adapter/service_client.py
src/vln_adapters/dualvln_adapter/run_inference_server.sh
src/vln_interface/scripts/trajectory_adapter.py
src/navigation_bridge/scripts/path_follower.py
src/navigation_bridge/scripts/safety_filter.py
src/robot_description/config/b2w_sensor_frames_official.yaml
src/sensors/camera/README.md
src/sensors/camera/config/b2_front_rgb.yaml
src/sensors/camera/config/b2_front_rgb_sim.yaml
src/vln_adapters/dualvln_adapter/README.md
src/vln_adapters/dualvln_adapter/config/dualvln.yaml
third_party/COLCON_IGNORE
third_party/README.md
third_party/unitree_ros/  # 官方克隆，提交 daadf41e
```

## DynaNav 场景兼容修复（2026-08-29）

Office/Warehouse 的官方起点和朝向原本针对 Nova Carter。当前 B2-W 评测仍使用官方
episode 的世界坐标，但在 Isaac Sim 创建环境时会明确关闭 locomotion 的 reset 随机化，
并在日志打印实际出生位姿，便于检查坐标是否被覆盖。Hospital/Warehouse 默认加载
DynaNav benchmark 指定的 Isaac 5.0 资产，可用 `DYNANAV_ASSET_VERSION=5.1` 回退。

另外，接触事件按官方 DynaNav 的持续接触原则去抖：默认连续 100 个 B2-W 仿真周期
（50 Hz，约 2 秒）超过 100 N 才触发 `/simulation/collision` 和恢复动作；单帧落地冲击
不会再清除模型轨迹。可通过 `DYNANAV_CONTACT_DEBOUNCE_STEPS` 调整，但公平比较时应
固定该值。修改位于 `src/robot_controller/scripts/rsl_rl/play.py`，未修改任何上游
DynaNav/TIC-VLA 文件。

已在主机 Isaac Sim/NVIDIA 环境复测 Office `dynanav_026`：实际出生位姿为
`(-2.850, 9.599, -2.304 rad)`，确认朝向覆盖问题已解决；该 dummy 直线路径仍会因其
不包含转弯且与 B2-W 车体发生持续静态接触而超时，这属于测试轨迹/车体几何与原始
Nova Carter 回合不匹配，不能解释为模型语义成功。纯逻辑回归测试通过：`11 passed`，其中包含
旋转出生朝向下长轨迹的坐标系回归测试。
