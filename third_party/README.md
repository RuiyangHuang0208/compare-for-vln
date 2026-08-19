# Third-Party Sources

第三方仓库保持原始 Git 历史，不在仓库内部直接修改。适配代码和补丁只能放在 `robot_vln_ws/src` 或 `robot_vln_ws/configs`。

| Directory | Upstream | Pinned reference | Purpose |
|---|---|---|---|
| `robot_lab` | `fan-ziqi/robot_lab` | `dda1510` | B2W locomotion task and policy runtime |
| `InternNav` | `InternRobotics/InternNav` | `7a5c624` | DualVLN inference runtime |
| `unitree_ros` | `unitreerobotics/unitree_ros` | `daadf41ee9afce8f90fdc09a98506012691fa122` | Official B2W geometry and nominal sensor frames |

Clone all pinned upstream repositories with:

```bash
git clone --recurse-submodules https://github.com/RuiyangHuang0208/VLNmodel_B2W.git
cd VLNmodel_B2W
git -C third_party/InternNav apply ../patches/internnav_sim_compat.patch
```

The InternNav patch contains the four simulation compatibility changes documented in
`src/vln_models/dualvln/README.md`. Model weights are not stored in Git.

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
