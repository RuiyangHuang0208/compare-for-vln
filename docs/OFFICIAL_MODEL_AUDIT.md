# VLN 模型官方实现审计

本文件区分三件容易混淆的内容：模型官方推理、模型官方高层控制、B2-W
低层执行。所有模型最终仍通过统一 ROS 2 接口和 B2-W SRU-ONNX 执行器运行。

`configs/models.yaml` 中每个已接入模型都记录：

- 官方仓库中的依据文件；
- 审查时的完整 Git commit；
- 依据文件的 SHA256；
- 官方参数；
- 为适配 B2-W 而产生的差异。

`benchmark_runner` 在原生高层模式启动前会校验 commit 和 SHA256。官方文件发生
变化时会拒绝运行，避免把未经复核的新代码当成已验证实现。

## 模型结论

| 模型 | 官方依据 | 当前实现 | 必要差异 |
|---|---|---|---|
| TIC-VLA | `DynaNav/behavior/spot_test_ticvla.py` | 原生模式复用官方 Spot 曲率、滤波、限速和加减速公式 | 官方 Spot 线速度上限 1.5 m/s，按已验证的 SRU-ONNX 指令范围裁剪为 1.0 m/s；低层仍为 B2-W |
| OmniVLA | `inference/run_omnivla.py` | 原生模式复用第 4 个 waypoint、`dt=1/3`、比例限幅至 0.3 m/s 和 0.3 rad/s 的官方换算 | 只替换机器人低层为 B2-W SRU-ONNX |
| DualVLN | `InternNav/scripts/realworld/controllers.py` | 复用官方模型输出；当前由共享 Pure Pursuit 执行 | 官方 MPC 依赖 CasADi/IPOPT，当前环境未安装，不能称为官方 MPC |
| NaVILA | `evaluation/vlnce_baselines/navila_trainer.py` | 复用官方 prompt、8 帧全历史采样、模型生成和离散动作语义 | Habitat 离散动作被转换为可供 B2-W 跟踪的局部圆弧；解析失败采用安全 STOP，不使用上游“默认前进”回退 |
| Uni-NaVid | `offline_eval_uninavid.py` | 复用官方在线缓存、生成设置以及 0.5 m/30° 动作语义 | 上游没有连续机器人控制器；动作按本工程 `base_link`（x 前、y 左、yaw 逆时针为正）转换为 B2-W 圆弧 |

核对入口：[TIC-VLA 官方仓库](https://github.com/ucla-mobility/TIC-VLA)、
[OmniVLA 官方仓库](https://github.com/NHirose/OmniVLA)、
[InternNav/DualVLN 官方仓库](https://github.com/InternRobotics/InternNav)、
[NaVILA 官方仓库](https://github.com/AnjieCheng/NaVILA)、
[Uni-NaVid 官方仓库](https://github.com/jzhzhang/Uni-NaVid)。本地 `third_party`
用于固定可复现 commit；参数依据同时与这些上游入口核对，而不是只相信本地副本。

## 两种评测模式

公平模式：所有轨迹/离散动作模型进入共享 Pure Pursuit。该模式适合比较模型输出，
但不是各模型官方控制器复现。

```bash
./scripts/run_dynanav_full85_all.sh --no-outdoor --no-resume \
  ticvla omnivla dualvln navila uninavid
```

原生高层模式：有官方公式的模型使用官方公式；没有官方连续控制器的模型使用已明确
标注的 B2-W 适配。场景、成功判定、评估器和 SRU-ONNX 低层保持统一。

```bash
./scripts/run_dynanav_full85_all.sh --native --no-outdoor --no-resume \
  ticvla omnivla dualvln navila uninavid
```

注意：`--official` 表示使用 DynaNav 官方 85 回合、seed、起终点、指令、超时和成功
阈值；它不等于“每个模型都存在官方 B2-W 控制器”。`--native` 才选择模型原生高层
执行配置。

## 控制器独立验收

在正式模型评测前，可用确定性输入分别验证五条控制链路。该测试不加载 VLN 大模型，
因此失败可归因于高层转换、navigation bridge、SRU-ONNX、场景或仿真基础设施，不能
归因于模型视觉理解。每个控制器分别测试直行和左转，共 10 个独立 Isaac Sim 回合：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
./scripts/run_controller_acceptance.sh
```

结果位于 `outputs/controller_probe_*/controller_acceptance/`。只有 `success=true` 且
`physical_collision_count=0` 才判定 PASS。脚本等待每个 Isaac 进程树退出后才启动下一项，
不同时占用多份显存。

## 不能从官网确认的参数

NaVILA 和 Uni-NaVid 没有发布统一的连续速度、角速度和加速度。当前 0.5 m/s 是
B2-W 执行适配参数，不是官方声明值。相机位置继续统一使用 B2-W 官方传感器安装位姿；
模型内部 resize 使用各 checkpoint/官方预处理代码，不能把模型 resize 尺寸误认为
仿真相机原始分辨率。

NaVILA 的 VLN-CE 配置明确给出 512×512、90° HFOV；TIC-VLA 的 DynaNav Spot 代码
明确给出 1920×1080、90° HFOV。OmniVLA 和 Uni-NaVid 没有发布可直接对应本评测的
HFOV，二者当前使用统一 DynaNav 90° 相机，并在 `camera_parameter_source` 中标为评测
配置而非模型官方参数。
