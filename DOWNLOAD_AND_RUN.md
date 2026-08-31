# 下载模型并运行工作区

GitHub 仓库包含 ROS 2/Isaac Sim 代码、配置、实验 JSON/CSV、B2-W 小型策略文件和第三方
源码的 Git 子模块引用。VLN 大模型权重、日志、构建目录和测试采集数据不在 Git 中。

本指南说明在一台新机器上恢复当前五模型评测环境。五个活动模型的权重合计约
66.3 GiB；考虑 Hugging Face 缓存和构建文件，建议至少预留 100 GiB 可用空间。

## 1. 克隆代码和子模块

```bash
git clone --recurse-submodules \
  https://github.com/RuiyangHuang0208/compare-for-vln.git
cd compare-for-vln

# 如果克隆时没有使用 --recurse-submodules：
git submodule update --init --recursive

# 应用本仓库验证过的四项 InternNav 兼容修改。
git -C third_party/InternNav apply --check ../../patches/internnav-local.patch
git -C third_party/InternNav apply ../../patches/internnav-local.patch
```

`git submodule status` 每行开头不应为 `-`。InternNav 应用补丁后显示为 modified，这是
预期状态；不要在主仓库执行 `git add third_party/InternNav` 来替换固定的上游 commit。

## 2. 安装下载工具

下面的命令要求系统已有 Python 3 和足够磁盘空间：

```bash
python3 -m pip install --user -U "huggingface_hub[cli]"
hf --help
```

部分 Hugging Face 模型可能要求先在网页接受许可证并登录：

```bash
hf auth login
```

所有命令都应在仓库根目录执行：

```bash
export ROBOT_VLN_WS="$(pwd)"
mkdir -p checkpoints/vln
```

## 3. 下载五个活动模型

### DualVLN

```bash
hf download InternRobotics/InternVLA-N1-DualVLN \
  --local-dir checkpoints/vln/dualvln

curl -fL --retry 3 \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth \
  -o checkpoints/vln/dualvln/depth_anything_v2_metric_hypersim_vits.pth
```

辅助深度模型的预期 SHA-256：

```text
b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545
```

检查：

```bash
test -f checkpoints/vln/dualvln/model.safetensors.index.json
test -f checkpoints/vln/dualvln/model-00004-of-00004.safetensors
echo 'b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545  checkpoints/vln/dualvln/depth_anything_v2_metric_hypersim_vits.pth' | sha256sum -c -
```

### TIC-VLA

```bash
hf download OpenGVLab/InternVL3-1B \
  --local-dir checkpoints/vln/ticvla/InternVL3-1B

hf download handsomeYun/TIC-VLA TIC-VLA-model.ckpt \
  --repo-type dataset \
  --local-dir checkpoints/vln/ticvla
```

检查：

```bash
echo 'a8b67c54568417f3631723e6b3e120720eaa638e03e62dc25666c70e3ae3e484  checkpoints/vln/ticvla/InternVL3-1B/model.safetensors' | sha256sum -c -
echo '376263f89fad0f42c267d85655019232edc91d36e214e23424804dd4cd42e036  checkpoints/vln/ticvla/TIC-VLA-model.ckpt' | sha256sum -c -
```

### NaVILA

必须使用 8 帧模型 `a8cheng/navila-llama3-8b-8f`：

```bash
hf download a8cheng/navila-llama3-8b-8f \
  --revision b2294e96581454468d6b94f38201f4f965ef48b7 \
  --local-dir checkpoints/vln/navila
```

检查六个主要权重：

```bash
(cd checkpoints/vln/navila && sha256sum -c SHA256SUMS)
```

`SHA256SUMS` 已由 Git 提供；服务启动脚本还会检查所有必需文件。

### OmniVLA

只使用 `omnivla-original`，不要换成 balance、edge、CAST 或量化版本：

```bash
hf download NHirose/omnivla-original \
  --local-dir checkpoints/vln/omnivla/omnivla-original

(cd checkpoints/vln/omnivla/omnivla-original && sha256sum -c SHA256SUMS)
```

`SHA256SUMS` 已由 Git 提供，下载命令不会覆盖它。若 Hugging Face 下载了不同版本，
校验会失败；此时不要继续评测，应先确认上游 revision 是否发生变化。

### Uni-NaVid

```bash
mkdir -p checkpoints/vln/uninavid
hf download Jzzhang/Uni-NaVid \
  --include 'uninavid-7b-full-224-video-fps-1-grid-2/**' \
  --local-dir checkpoints/vln/uninavid

curl -fL --retry 3 \
  https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth \
  -o checkpoints/vln/uninavid/eva_vit_g.pth
```

EVA 权重检查：

```bash
echo '99d2bb36c6b52c94fe6e2e12373afb27de57ae81378c3d8c53bf0e83b0f4275f  checkpoints/vln/uninavid/eva_vit_g.pth' | sha256sum -c -
test -f checkpoints/vln/uninavid/uninavid-7b-full-224-video-fps-1-grid-2/pytorch_model.bin.index.json
```

## 4. 不需要下载的内容

- `logs/`、`log/`：旧运行日志，不影响重新运行。
- `build/`、`install/`：ROS 2 构建产物，应在新机器重新生成。
- `test_data/`：调试相机帧和官方离线样例，不是闭环 DynaNav 必需项。
- `checkpoints/vln/mobilevla_r1/`：约 75 GiB 的本地归档。官方发布包缺少论文所述的
  Depth/Point tower，当前服务会拒绝静默降级，因此不属于五模型正式比较。
- Hugging Face `.cache/`：可重新生成。

B2-W SRU-ONNX 和 Isaac `.pt` 小型策略已提交到 Git，无需另行下载。模型权重不得用
`git add -f` 上传到 GitHub。

## 5. 软件环境

经过验证的基础环境是 Ubuntu、NVIDIA GPU、ROS 2 Jazzy、Isaac Sim 5.1 和 Isaac Lab
2.3.2。模型服务依赖互相冲突，必须使用独立环境：

| 服务 | 默认 Python | 要求 |
|---|---|---|
| DualVLN | `~/miniconda3/envs/internnav/bin/python` | InternNav 依赖、CUDA PyTorch、Flask |
| TIC-VLA | 同 `internnav` | Python 3.11、Torch、Transformers、Flask |
| NaVILA | `~/miniconda3/envs/navila/bin/python` | Python 3.10、官方 NaVILA/LLaVA 依赖 |
| OmniVLA | `~/miniconda3/envs/omnivla/bin/python` | Python 3.10；版本检查写在启动脚本中 |
| Uni-NaVid | `~/miniconda3/envs/uninavid/bin/python` | Python 3.10、官方 Uni-NaVid 依赖 |

各上游依赖入口分别为：

```text
third_party/InternNav/pyproject.toml
third_party/TIC-VLA/pyproject.toml
third_party/NaVILA/pyproject.toml
third_party/OmniVLA/pyproject.toml
third_party/Uni-NaVid/pyproject.toml
```

不要把这些依赖全部安装进 ROS 或 Isaac Lab 环境。启动脚本会检查实际 Python 版本、
关键包、checkpoint 文件和 OmniVLA 可用显存，并在条件不满足时给出错误。

## 6. 构建 ROS 2 工作区

```bash
cd "${ROBOT_VLN_WS}"
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

先验证不加载大模型的控制链路：

```bash
./src/robot_controller/scripts/run_b2w_hospital.sh env-check
./scripts/run_controller_acceptance.sh
```

## 7. 运行模型

单模型调试需要两个终端。以 TIC-VLA 为例：

终端 1：

```bash
cd "${ROBOT_VLN_WS}"
./src/vln_adapters/ticvla_adapter/run_inference_server.sh
```

终端 2：

```bash
cd "${ROBOT_VLN_WS}"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_bringup dynanav_single_episode.launch.py \
  model:=ticvla episode:=dynanav_001 \
  evaluation_mode:=trajectory_normalized sensor_profile:=rgb_only
```

其他模型只需替换终端 1 的服务路径和终端 2 的 `model`：

```text
dualvln  -> src/vln_adapters/dualvln_adapter/run_inference_server.sh
navila   -> src/vln_adapters/navila_adapter/run_inference_server.sh
omnivla  -> src/vln_adapters/omnivla_adapter/run_inference_server.sh
uninavid -> src/vln_adapters/uninavid_adapter/run_inference_server.sh
```

DualVLN 使用 `sensor_profile:=rgb_d`；其余四个活动模型使用 `rgb_only`。更稳妥的批量入口
会自动启动、等待和清理每个模型服务，避免同时占用 GPU：

```bash
# 先跑一个简单、无行人的 smoke 回合
DYNANAV_NO_PEDESTRIANS=1 \
  ./scripts/run_dynanav_full85_all.sh --episodes simple_forward_3m_standard \
  --no-resume omnivla dualvln navila uninavid

# 五模型非 Outdoor 比较
./scripts/run_dynanav_full85_all.sh --no-outdoor --no-resume \
  ticvla omnivla dualvln navila uninavid

# 官方 85 回合（包含 Outdoor 和官方行人配置）
./scripts/run_dynanav_full85_all.sh --official --no-resume \
  ticvla omnivla dualvln navila uninavid
```

运行结果保存在 `outputs/<model>/<experiment>/`，运行日志保存在 `logs/`。完整评测模式、
控制器差异和已验证环境见主 [`README.md`](README.md) 与
[`docs/OFFICIAL_MODEL_AUDIT.md`](docs/OFFICIAL_MODEL_AUDIT.md)。

## 8. 常见错误

- `Missing ... checkpoint`：下载目录层级不正确；对照 `configs/models.yaml` 的路径。
- `ModuleNotFoundError`：模型服务使用了错误的 Conda 环境，不要在 ROS 系统 Python 中补包。
- `CUDA unavailable`：确认 `nvidia-smi` 和对应环境中的 `torch.cuda.is_available()`；若从
  Codex 运行，还要允许宿主机/GPU权限。
- OmniVLA 报显存不足：关闭旧的 Isaac Sim 或其他模型服务；不要同时加载五个模型。
- 子模块目录为空：执行 `git submodule update --init --recursive`。
- InternNav 补丁已经应用：`git apply --check` 会失败；用 `git -C third_party/InternNav status`
  确认四个预期修改即可，不要重复应用。
