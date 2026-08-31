# VLN Checkpoints

Model weights are local runtime data and are not committed. Expected new layout:

```text
dualvln/
ticvla/
navila/
aware_vln/
evolve_nav/
```

`mobilevla_r1/` 已退出活动模型集合；若目录仍存在，仅作为本机归档，不会被构建或评测脚本加载。

DualVLN is stored under `dualvln/` together with the other VLN model checkpoints. Its 16 GB of
weights were moved in place from the previous `checkpoints/dualvln` location; no duplicate copy is kept.

The official TIC-VLA files are present under `ticvla/`:

```text
InternVL3-1B/model.safetensors
  size:   1876463472 bytes
  sha256: a8b67c54568417f3631723e6b3e120720eaa638e03e62dc25666c70e3ae3e484
TIC-VLA-model.ckpt
  size:   1938501547 bytes
  sha256: 376263f89fad0f42c267d85655019232edc91d36e214e23424804dd4cd42e036
```

Weights are runtime data and remain ignored by Git.

NaVILA 的 8 帧官方 checkpoint 位于 `navila/`。模型类型、revision 和六个主权重文件的
SHA256 见 `navila/README.md`。
