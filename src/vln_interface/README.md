# VLN Interface

所有 VLN/VLA 模型通过 adapter 发布同一种消息：

```text
/vln/command  vln_interface/msg/NavigationCommand
```

支持四种命令：

| `command_type` | 有效载荷 | 用途 |
|---|---|---|
| `STOP` | 无 | 立即请求停车 |
| `WAYPOINT` | `waypoints` | 一个或多个目标位姿 |
| `TRAJECTORY` | `trajectory` | 已排序路径 |
| `VELOCITY` | `velocity` | 直接速度命令 |

速度字段固定映射：

```text
velocity.linear.x  -> vx
velocity.linear.y  -> vy
velocity.angular.z -> yaw_rate
```

`waypoint_adapter.py`、`trajectory_adapter.py` 和 `velocity_adapter.py` 只处理模型输出差异。它们不得加载 B2W policy，也不得直接发布机器人关节动作。

坐标约定和 topic 名称见 `config/interface.yaml`。
