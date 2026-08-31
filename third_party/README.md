# Third-Party Sources

第三方仓库保持原始 Git 历史，不在仓库内部直接修改。适配代码和补丁只能放在 `robot_vln_ws/src` 或 `robot_vln_ws/configs`。

| Directory | Upstream | Pinned reference | Purpose |
|---|---|---|---|
| `robot_lab` | `fan-ziqi/robot_lab` | `dda1510` | B2W locomotion task and policy runtime |
| `InternNav` | `InternRobotics/InternNav` | `7a5c624` | DualVLN inference runtime |
| `unitree_ros` | `unitreerobotics/unitree_ros` | `daadf41ee9afce8f90fdc09a98506012691fa122` | Official B2W geometry and nominal sensor frames |
| `sru-navigation-sim` | `leggedrobotics/sru-navigation-sim` | `da6d5d68e46122967ab82ea9077f8304b5af47d7` | Official SRU Isaac B2W USD reference |
| `TIC-VLA` | `ucla-mobility/TIC-VLA` | `9fa6f8b66b9e121d5df5df071297bba8e5353ebb` | TIC-VLA model source and DynaNav compatibility reference |
| `NaVILA` | `AnjieCheng/NaVILA` | `76b98f233dd0fff05dfcd69435eec6740febff9d` | NaVILA official RGB text-action inference source |
| `OmniVLA` | `NHirose/OmniVLA` | `5182600cb4a9ee07684e17cdd2a6cbafc56b8a68` | Full OmniVLA 8x4 continuous trajectory inference source |
| `MobileVLA-R1` | `AIGeeksGroup/MobileVLA-R1` | `ff9062c143d6b50c05d913e6be15d36634a7a460` | 已退出活动模型集合；仅保留未修改的上游归档 |

TIC-VLA uses UCLA Mobility Lab's Academic Software License. It permits academic/nonprofit research use,
prohibits further transfer, and requires commercial users to obtain a separate license. For that reason it
is retained as an upstream git submodule; no TIC-VLA source, model weight, or DynaNav scene asset is copied
into a workspace package. Review `third_party/TIC-VLA/LICENSE.md` before redistributing this workspace.

NaVILA 使用 Apache-2.0 许可证，并作为上游 git submodule 保持未修改。工作区适配代码只在
`src/vln_adapters/navila_adapter`，checkpoint 只在 `checkpoints/vln/navila`。

OmniVLA 使用 MIT 许可证，并作为上游 git submodule 保持未修改。主评测只使用完整
`omnivla-original`，不会把 `OmniVLA-edge` 混入同一个模型名称。

MobileVLA-R1 已退出活动模型、构建和 benchmark 集合。源码和本机权重暂时只作为可恢复归档，
不会由五模型运行脚本启动；其 adapter 目录带有 `COLCON_IGNORE`。

The default locomotion binary is copied from `leggedrobotics/sru-robot-deployment`
commit `568a96c6c704d9dede4d7293fa09b98d9cbff4e0` into
`checkpoints/b2w_locomotion/sru_onnx/`. Its exact source path and checksum are
recorded in that directory's README; upstream source files are not modified here.

Clone all pinned upstream repositories with:

```bash
git clone --recurse-submodules https://github.com/RuiyangHuang0208/compare-for-vln.git
cd compare-for-vln
git -C third_party/InternNav apply ../../patches/internnav-local.patch
```

The InternNav patch contains the four simulation compatibility changes documented in
`src/vln_adapters/dualvln_adapter/README.md`. Model weights are not stored in Git.

Unitree B2W sensor-frame source:

```text
third_party/unitree_ros/robots/b2w_description/urdf/b2w_description.urdf
SHA-256: cb70693d5bdf98d7c9c402a1dacb6c74cb64518fdb4daf47fa068534c8e10d90
```

更新参考版本时必须同时更新：

```text
src/robot_description/config/b2w_sensor_frames_official.yaml
src/sensors/camera/config/b2_front_rgb_sim.yaml
src/sensors/camera/config/b2_front_rgb.yaml
```
