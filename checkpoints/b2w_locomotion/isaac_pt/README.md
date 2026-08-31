# SRU Isaac B2W Policy

```text
Upstream: https://github.com/leggedrobotics/sru-navigation-sim.git
Commit: da6d5d68e46122967ab82ea9077f8304b5af47d7
File: policy_b2w_new_2.pt
SHA-256: d2cece212a16ad53ebea4a8b096090b440b5d22ed6fdbc6800ddffddfb393a3b
Contract: 60 observation / 16 action / recurrent LSTM
```

This policy remains available with `B2W_LOCOMOTION_POLICY=isaac-pt`. The default
is now the Gazebo deployment ONNX at `../sru_onnx/policy_force_new.onnx`; the
RobotLab checkpoint remains at `../model_2600.pt`.
