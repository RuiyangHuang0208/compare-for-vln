# NaVILA ROS 2 适配器

本包是官方 NaVILA 与工作区统一导航接口之间的唯一边界：

```text
DynaNav RGB + /vln/instruction
  -> 独立 NaVILA Python 3.10 推理服务
  -> 原始文本动作
  -> navila_adapter
  -> /vln/command (TRAJECTORY 或 STOP)
  -> 共享 navigation_bridge
  -> /nav_vel
  -> SRU ONNX B2-W locomotion
```

官方源码只位于 `third_party/NaVILA`，权重只位于
`checkpoints/vln/navila`。本包不复制或修改模型核心代码，也不包含 Go2 locomotion、
关节控制、Depth/LiDAR 避障或目标坐标输入。

## 输入与帧采样

ROS 节点只订阅：

- `/camera/rgb/image_raw`
- `/vln/instruction`
- `/episode/state`
- `/episode/id`

相机可以继续发布 Depth 和 CameraInfo，但 NaVILA 不订阅它们。每次决策保存一张当前
RGB；当前帧始终放在最后，从整个 episode 决策历史均匀抽取前
`model.config.num_video_frames - 1` 帧。历史不足时在最前面补黑帧，RESET 后完全清空。
实现不会退化成“最近 8 帧”滑动窗口。

## 官方推理契约

服务从官方仓库导入：

```text
load_pretrained_model
process_images
tokenizer_image_token
conv_templates["llama_3"]
```

生成固定为 `do_sample=false`、`temperature=0`、`max_new_tokens=32`、
`use_cache=true`。模型返回的原始文本发布到 `/vln/navila/raw_action`，支持：

```text
stop
move forward 25|50|75 cm
turn left 15|30|45 degree
turn right 15|30|45 degree
```

完整句式如 `The next action is move forward 50 cm` 也可解析。未知文本、空输出、
非法步长、NaN/Inf、超时、HTTP 错误或 CUDA OOM 一律发布 STOP；不会采用官方评测代码
中“解析失败默认前进 25 cm”的回退。

前进动作转换为 `base_link` 下的直线 XY 轨迹；左右转转换为带 XY 位移的圆弧轨迹，
然后进入所有轨迹模型共用的 Pure Pursuit。模型 STOP 会立即停车并发布
`/episode/model_stop`；独立 `goal_monitor` 根据 ground-truth 目标距离判定 episode
SUCCESS 或 FAILED，模型自身不能直接宣告成功。

## 安装与运行

模型服务使用独立 Conda `navila`（Python 3.10），不要安装进 ROS Jazzy 或 Isaac Lab：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
./src/vln_adapters/navila_adapter/run_inference_server.sh
```

看到 `[NAVILA SERVER] Ready` 后，在第二个终端启动同一套 DynaNav/B2-W 链路：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=navila episode:=dynanav_001 \
  sensor_profile:=rgb_only evaluation_mode:=trajectory_normalized \
  desired_speed:=1.0 shutdown_after_finish:=false
```

运行中可替换语言指令：

```bash
ros2 topic pub --once /vln/instruction std_msgs/msg/String \
  "{data: 'Go straight ahead and stop at the vending machine at front.'}"
```

回合进入 `SUCCESS`、`FAILED` 或模型 STOP 后，Isaac Sim 在
`shutdown_after_finish:=false` 下继续保持运行。此时发送新的非空指令会自动重新激活
NaVILA 的手动演示模式，不需要额外发布 `START`；这不会重新开启或覆盖已经保存的评测结果。
空指令仍会立即停车。

## 转弯半径标定

`config/navila.yaml` 当前使用已经实测的 `conversion.turn_radius: 1.0`。标定使用完全相同的
B2-W、locomotion 和 path follower，命令为：

```bash
ros2 launch robot_bringup dummy_b2w.launch.py \
  mode:=left_turn turn_radius:=1.0 turn_degrees:=45.0

ros2 launch robot_bringup dummy_b2w.launch.py \
  mode:=right_turn turn_radius:=1.0 turn_degrees:=45.0
```

2026-08-24 实测中，左转连续运行后 `y=+1.063 m`、机身高度 `0.749 m`；右转为
`y=-2.467 m`、机身高度 `0.796 m`，两个方向均保持直立并正确累计 yaw。若以后更换
locomotion、速度限制或 path follower，必须重新执行标定。把半径改回 `0.0` 会使节点
拒绝启动，防止静默使用未标定值。

## 测试

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  src/vln_adapters/navila_adapter/test
```

测试覆盖全历史帧采样、严格文本解析、直线/左右圆弧、STOP、超时、RGB-only 订阅以及
reset 后丢弃上一回合迟到响应。

## 2026-08-24 实测结果

使用 Hospital 前置 RGB 静态图和指令
`Go straight ahead and stop at the vending machine at front.` 完成了真实 8 帧推理。模型原始
输出为：

```text
The next action is move forward 75 cm.
```

该输出转换成 `base_link` 下 8 个直线轨迹点，终点为 `(0.75, 0.0, 0.0)`。单次推理耗时
`0.880 s`，服务报告 CUDA 峰值 `18,722,249,728 bytes`；结果保存在
`outputs/navila/offline_text_action.json`。

随后实际运行了无行人 `dynanav_001`。NaVILA 先连续输出 `move forward 75 cm`，接近后改为
`move forward 25 cm`，最后自行输出 `STOP`。B2-W 通过共享 Pure Pursuit 和 SRU ONNX
locomotion 行走 `7.563 m`，无碰撞、无 stuck，平均推理耗时 `0.575 s`，但停止时距官方
ground-truth 目标仍有 `6.151 m`，因此独立 goal monitor 正确判定为 `FAILED`。结果保存在
`outputs/navila/single_episode/dynanav_001.json`。

这不是服务、文本解析或 ROS 控制链断开：动作持续发布并实际驱动了 B2-W；失败原因是模型
在目标阈值外过早输出 STOP。按照“无行人回合成功后才运行动态行人回合”的验收顺序，
`dynanav_002` 尚未运行。不得通过修改目标坐标或把模型 STOP 直接当作成功来掩盖该结果。
