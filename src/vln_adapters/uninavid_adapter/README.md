# Uni-NaVid 接入

本包只订阅 RGB、语言指令和 episode 状态。模型输出经严格解析后转成 `base_link` 局部轨迹，交给共享
`navigation_bridge`；它不订阅 Depth、LiDAR、地图或目标真值，也不发布 `/nav_vel` 或关节动作。

官方 VLN-CE 在线策略每次预测最多四个 `forward/left/right/stop` 动作，本接入执行前两个。前进为
0.5 m，转向为 30 度；转向会转换成 B2-W 可由 Pure Pursuit 执行的 0.25 m 半径圆弧。共享
跟踪器还会检查轨迹中显式给出的最终航向，不会只到达 XY 后提前结束。只有收到
`/navigation/trajectory_finished=true` 后才会请求下一次推理，不使用固定 sleep。共享跟踪器若发布
`/navigation/trajectory_failed=true`，adapter 会丢弃未执行完的离散动作并使用最新 RGB 重新推理，
不会继续执行过期轨迹。

## checkpoint 加载 warning 说明

官方发布的 `config.json` 将 `model_type` 写为基础 LLaMA，但实际导航类是
`LlavaLlamaAttForCausalLM`。adapter 只在 `/tmp` 的临时配置中将其重标记为 `llava`，不修改
官方 checkpoint 或第三方源码。这样可以消除 Transformers 的 LLaMA/LLaVA 类型不匹配提示。

EVA 视觉编码器仍按官方 `strict=False` 加载。EVA 预训练文件包含分类头、归一化层和额外
block，这些键不属于 Uni-NaVid 的导航视觉塔，因此出现 `unexpected_keys` 是正常的；真正的
加载错误、语言模型权重缺失或推理 HTTP 5xx 才需要处理。

## 缓存同步修复

ROS 2 的 `START` 和锁存指令消息可能乱序到达。adapter 现在把 `START` 后收到的第一条非空指令视为
当前回合指令，只有已有指令被替换时才增加 generation。服务端对过期 generation 返回 HTTP 409，adapter
会自动采用最新 generation、重新初始化在线缓存并继续请求，不会用旧缓存无限重试。重新启动服务后即可
验证该修复；旧的 `scene_gate_20260829` 结果不需要删除。

模型服务：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
UNINAVID_PYTHON=/home/mifcom2/miniconda3/envs/uninavid/bin/python \
  ./src/vln_adapters/uninavid_adapter/run_inference_server.sh
```

另一个终端启动同一公平评测场景：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=uninavid episode:=dynanav_001 sensor_profile:=rgb_only \
  evaluation_mode:=trajectory_normalized comparison_track:=rgb_only
```

## 当前实测结果（2026-08-24）

- SRU ONNX 实机仿真圆弧校准通过：请求左转/右转 30 度，实测分别为 +25.49/-25.06 度；
- 连续两个转向动作校准通过：请求左转/右转 60 度，实测分别为 +57.09/-55.33 度；
- ROS episode RESET 已修复并实际验证，不再触发 PyTorch inference tensor 原地更新错误；
- 官方测试图像 36 帧在线缓存推理可运行，最后输出 `stop stop stop stop`；
- Isaac Sim 5.1 Hospital、无行人 `dynanav_001` 最终复测运行 40.04 秒；
- 该回合模型未到达目标：`success=false`，最终误差 11.197 m，不能算作导航成功；
- 路径长度 5.494 m，记录 2 个静态几何接触事件；
- 所有 0.25 m 圆弧与最终航向校正均正常完成，本轮没有 trajectory timeout；
- 端到端推理延迟平均 0.173 秒，P95 为 0.199 秒，最大 0.221 秒；
- 首帧 RGB 已人工检查：640x480、HFOV 120 度，方向正常且目标售货机位于左前方；
- 因静态回合未成功，尚未运行动态行人回合和完整 benchmark。
