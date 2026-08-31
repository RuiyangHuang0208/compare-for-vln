# VLN 统一消息接口

所有模型适配器都在 `/vln/command` 发布
`vln_interfaces/msg/NavigationCommand`。消息坐标系固定为 `base_link`：x 向前、y 向左、
yaw 逆时针为正；距离单位为米，时间单位为秒，角度单位为弧度。

每条命令必须带非零时间戳。导航桥会拒绝过期、来自未来、包含 NaN/Inf、空轨迹、
`valid=false` 或 frame 不正确的运动命令。`STOP` 不受这些运动数据检查影响，始终立即
清空活动路径并输出零速度。
