# SRU Gazebo B2W ONNX Policy

```text
Upstream: https://github.com/leggedrobotics/sru-robot-deployment.git
Commit: 568a96c6c704d9dede4d7293fa09b98d9cbff4e0
Source: b2w_sim/b2w_controllers/src/policy/policy_force_new.onnx
SHA-256: 3117407a6d984a06d489d4b55b1f8962492686b7e8ebdb8a33bf838bae46d44e
Contract: 60 observation / 16 action / recurrent LSTM 256 / fixed batch 1
Inference: 50 Hz, ONNX Runtime CPU
```

This is the default `robot_vln_ws` locomotion backend. Its observation order,
recurrent state, joint order and action scaling are copied from the official
Gazebo controller. `isaac-pt` and `robotlab` remain available for comparison.
