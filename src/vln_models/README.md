# VLN Model Wrappers

每个模型目录只负责三件事：加载该模型、接收统一传感器输入、发布模型原始预测。`ros_node.py` 必须保持薄，不得包含 B2W 控制逻辑。

建议映射：

| 模型 | 原始输出 | Adapter |
|---|---|---|
| DualVLN | trajectory | `trajectory_adapter.py` |
| NaVILA | waypoint | `waypoint_adapter.py` |
| Evolve-Nav | 按实际模型接口确认 | 对应 adapter |
| Aware-VLN | 按实际模型接口确认 | 对应 adapter |

模型尚未接入前，不要猜测其 observation、action 或 checkpoint 格式。原始仓库放在 `third_party/`，权重放在 `checkpoints/<model>/`，两者都不应安装进 B2W 的 Isaac Lab 环境。
