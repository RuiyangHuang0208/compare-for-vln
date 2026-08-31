# MobileVLA-R1 适配器

本包把官方 MobileVLA-R1 接到现有统一导航接口，第三方模型源码保持在
`third_party/MobileVLA-R1`，不会复制或修改模型核心代码。

```text
同步 RGB + Depth + CameraInfo
  -> Depth 投影并采样为 2048x3 Point Cloud
  -> 独立 Python 3.10 推理服务
  -> 严格解析 <answer>[12 个有限数值]</answer>
  -> 仅转发 index 0/1/2
  -> NavigationCommand.VELOCITY
  -> 共享 navigation_bridge/VelocityFilter
  -> /nav_vel -> B2-W sru_onnx
```

MobileVLA-R1 不进入 Pure Pursuit，不直接发布 `/nav_vel`，不订阅 LiDAR，也不发送 gait、
body pose 或关节动作。完整 12 维向量只保存在原始输出和状态审计消息中。

## 官方契约审查

- 源码 commit：`ff9062c143d6b50c05d913e6be15d36634a7a460`，Apache-2.0。
- 权重许可证：CC-BY-SA-4.0。
- 官方训练脚本和 dataset loader 使用 8 帧 RGB、1 帧 Depth、2048 点。
- 当前帧位于 RGB 历史末尾；历史不足时重复最后一帧，历史较长时均匀采样。
- `llama_3` conversation，官方采样参数为 `do_sample=true`、`temperature=0.7`、
  `top_p=0.9`、`max_new_tokens=512`。
- 12 个字段依次为 `x_vel_cmd, y_vel_cmd, yaw_vel_cmd, body_height_cmd,
  step_frequency_cmd, gait1, gait2, gait3, footswing_height_cmd, pitch_cmd,
  roll_cmd, stance_width_cmd`。
- 上游 `llava/model/multimodal_encoder/builder.py` 使用 `Optional` 却未导入。adapter 仅在加载
  `inference.py` 的瞬间临时注入 `typing.Optional` 并立即恢复，不修改第三方文件。

官方公开材料没有一致给出前三项的完整物理语义（尤其横向/yaw 符号）和一条命令的执行窗口。
因而默认：

```yaml
control.command_duration_s: 0.0
control.velocity_units_confirmed: false
control.coordinate_signs_confirmed: false
```

真实节点会拒绝启动。这是安全保护，不是安装错误；在找到可引用的官方执行语义前，不要
凭经验填写这些值，也不要把 stub 结果计入评测。

官方 MobileVLA-CoT Step 数据的说明文本支持 `x_vel_cmd` 使用 m/s、`yaw_vel_cmd` 使用
rad/s，但抽样记录中的 yaw 正负方向描述互相矛盾，且没有足够的非零 `y_vel_cmd` 样本确认
横向符号。论文补充材料只称输出为“短时域连续命令”，没有提供数值执行窗口；论文中的约
10 秒推理和约 15 秒闭环延迟不能当成命令持续时间。因此真实运行的单位/符号总开关仍保持
关闭，`command_duration_s` 仍为 `0.0`。

## Depth 与点云

Isaac bridge 发布对齐的 `32FC1` 米制深度。adapter 使用同一帧 CameraInfo 的针孔内参生成
camera-optical XYZ（x 向右、y 向下、z 向前），过滤零值、NaN 和 Inf，以固定 seed 无放回
采样 2048 点并按轴标准化。官方公共 `depth_to_point_cloud()` 使用归一化图像坐标且不读取
CameraInfo；这里采用实际内参是为了满足仿真传感器契约，差异必须随实验结果记录。

## 权重与环境

官方约 26.9 GB、26 个连续分卷已于 2026-08-24 下载并通过逐卷 SHA256、合并包 CRC、
危险路径和符号链接检查：

```text
checkpoints/vln/mobilevla_r1/archives/weight.zip.part-aa
...
checkpoints/vln/mobilevla_r1/archives/weight.zip.part-az
```

实际归档只有 `weight/sft` 和 `weight/rl` 两套 RGB VILA/NaVILA 结构，配置未启用 Depth/Point
tower，也没有对应 encoder 或 bridge 权重。主要哈希与上游 issue #9 报告完全一致。因此当前
官方权重不能运行论文描述的 RGB-D+Point MobileVLA-R1，服务会拒绝加载，不能用 RGB-only
NaVILA 冒充。完整校验结果见 `checkpoints/vln/mobilevla_r1/README.md`。真实推理环境固定为
独立 `mobilevla_r1` Conda 环境和 Python 3.10；官方脚本基线是 Torch 2.3、CUDA 12.2、
Transformers 4.37.2、FlashAttention 2.5.8。RTX 5090 当前环境采用已验证可识别该 GPU 的
Torch 2.7.1/CUDA 12.8、Transformers 4.37.2、FlashAttention 2.8.4 兼容栈，并只在该独立
环境应用官方 Transformers/DeepSpeed 补丁。不要安装到 ROS、Isaac Sim 或其他模型环境。

## 构建与测试

```bash
cd /home/mifcom2/b2w/robot_vln_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash

ROS_LOG_DIR=/tmp/mobilevla_r1_ros_logs \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/usr/bin/python3 -m pytest -q src/vln_adapters/mobilevla_r1_adapter/test
```

stub 服务只验证协议和 STOP 生命周期：

```bash
./src/vln_adapters/mobilevla_r1_adapter/run_inference_server.sh --stub
```

另一个终端可启动 adapter 接口测试模式：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch mobilevla_r1_adapter mobilevla_r1.launch.py \
  allow_stub_server:=true command_duration_s:=0.15
```

真实权重、单位、符号和执行窗口全部确认后，服务和单回合命令形式为：

```bash
./src/vln_adapters/mobilevla_r1_adapter/run_inference_server.sh
```

```bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=mobilevla_r1 episode:=dynanav_001 \
  comparison_track:=rgb_d_pointcloud_native \
  sensor_profile:=rgb_d_pointcloud_from_depth \
  evaluation_mode:=native_output desired_speed:=1.0
```

## 调试话题

```text
/vln/mobilevla_r1/raw_response       完整模型文本
/vln/mobilevla_r1/parsed_velocity    只含 vx/vy/wz
/vln/mobilevla_r1/inference_latency  模型推理耗时
/vln/mobilevla_r1/status             STOP 原因、完整12维审计信息、服务耗时和显存峰值
/vln/command                          统一 NavigationCommand
```

成功仍只由独立 `goal_monitor` 的 ground-truth 距离判断。模型零速度仅表示
`model_requested_stop`，不能直接记为成功。
