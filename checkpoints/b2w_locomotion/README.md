# B2W Locomotion Checkpoint

`model_2600.pt` 是从 `third_party/robot_lab` 迁入的 Unitree B2W flat locomotion policy。

```text
Source RobotLab commit: dda1510
Training config: UnitreeB2WFlatPPORunnerCfg
Official pretrained: no
SHA-256: f9c1ce4400b9b3286f89a31e76a770a3e8c9a0ecea23666ac6e312a43a05f075
```

该权重接收 `[vx, vy, yaw_rate]`，输出 12 个腿关节位置动作和 4 个轮关节速度动作。
