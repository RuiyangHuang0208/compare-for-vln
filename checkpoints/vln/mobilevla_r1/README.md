# MobileVLA-R1 权重

2026-08-24 已下载官方 Hugging Face 仓库发布的 26 个连续分卷：
`weight.zip.part-aa` 至 `weight.zip.part-az`，总计 `26,871,863,199` bytes。

官方 Hugging Face LFS 元数据中的每卷大小和 SHA256 已固化在：

```text
src/vln_adapters/mobilevla_r1_adapter/config/official_archive_manifest.json
```

预期目录：

```text
checkpoints/vln/mobilevla_r1/
├── archives/             # 原始分卷、weight.zip 及 SHA256SUMS
└── MobileVLA-R1/         # 安全检查后解压的官方归档
    └── weight/
        ├── sft/
        └── rl/
```

全部分卷 SHA256 与官方 LFS metadata 一致。合并归档 SHA256：

```text
90a356b75dada4be196462926082a5e36a3d9a77c438ff412bb1e7db98e5a966
```

ZIP CRC、归档路径和符号链接检查通过，共 87 个条目；安全解压后 72 个文件，完整清单位于
`MobileVLA-R1/CHECKPOINT_INVENTORY.json`。

## 官方发布包阻塞

本地配置和权重清单确认，`weight/sft` 与 `weight/rl` 都没有：

```text
use_depth_tower: true
use_point_tower: true
Depth encoder/tower weights
Point encoder/tower weights
modality bridge weights
non_lora_trainables.bin
```

`weight/rl/config.json` 的 `_name_or_path` 指向
`vila-long-8b-8f-scanqa-rxr-real-v1-seed10-bs10-1e4`，实际是 RGB NaVILA 结构。六个主要
RL 文件 SHA256 与上游 issue #9 报告完全相同。当前官方发布包无法作为论文所述的
RGB-D+Point MobileVLA-R1 checkpoint，服务会拒绝 RGB-only 静默回退。

主要 RL 文件 SHA256：

```text
d4df84977665ab9be02f987a4eccd56abb0ac7273f12f6906f50c13bc7110371  llm/model-00001-of-00004.safetensors
42c210a04fc723675f858b383d546d2d5e0d4aa857d4fb321d3e57da241796f9  llm/model-00002-of-00004.safetensors
d53f6cab87861720f85f22af13e756cd7cccf569c2f8c096a7ec38d513be43e4  llm/model-00003-of-00004.safetensors
f35c5720effc40db734ac67b232e0b97389f802251d5d4d2480987f91f6dc937  llm/model-00004-of-00004.safetensors
1ff7d080a98a3f014b057dca741d7be358b700565493a15dfd82cb98903b0c4b  vision_tower/model.safetensors
0921455072f43bfe67637a6d69faa7248b830c15dc67a6e9798b50e2aed6d376  mm_projector/model.safetensors
```

重新验证分卷：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run mobilevla_r1_adapter mobilevla_r1_checkpoint \
  --archives checkpoints/vln/mobilevla_r1/archives
```

如未来官方替换了归档，移走现有输出后再合并、检查并解压；工具不会覆盖现有文件：

```bash
ros2 run mobilevla_r1_adapter mobilevla_r1_checkpoint \
  --archives checkpoints/vln/mobilevla_r1/archives \
  --merged checkpoints/vln/mobilevla_r1/archives/weight.zip \
  --extract-to checkpoints/vln/mobilevla_r1/MobileVLA-R1
```

工具拒绝缺卷、多余卷、大小或 SHA256 不符、危险相对路径、绝对路径、符号链接、已有合并包，
以及非空 checkpoint 目标。解压成功后会生成 `CHECKPOINT_INVENTORY.json`，其中包含每个文件的
大小和 SHA256。合并包 SHA256 会打印到 JSON 报告中。
