# B2W Locomotion Checkpoint

`model_2600.pt` 是从 `third_party/robot_lab` 迁入的 Unitree B2W flat locomotion policy。

```text
Source RobotLab commit: dda1510
Training config: UnitreeB2WFlatPPORunnerCfg
Official pretrained: no
SHA-256: f9c1ce4400b9b3286f89a31e76a770a3e8c9a0ecea23666ac6e312a43a05f075
```

该权重接收 `[vx, vy, yaw_rate]`，输出 12 个腿关节位置动作和 4 个轮关节速度动作。

原 RobotLab 和 Isaac TorchScript 权重继续保留用于对照。当前默认权重是
Gazebo deployment controller 的 `sru_onnx/policy_force_new.onnx`：

```text
Source: leggedrobotics/sru-robot-deployment
Source commit: 568a96c6c704d9dede4d7293fa09b98d9cbff4e0
Observation/action: 60/16, recurrent LSTM 256, fixed batch 1
SHA-256: 3117407a6d984a06d489d4b55b1f8962492686b7e8ebdb8a33bf838bae46d44e
```

保留的 Isaac TorchScript：

```text
Source: leggedrobotics/sru-navigation-sim
Source commit: da6d5d68e46122967ab82ea9077f8304b5af47d7
Observation/action: 60/16, recurrent LSTM
SHA-256: d2cece212a16ad53ebea4a8b096090b440b5d22ed6fdbc6800ddffddfb393a3b
```
