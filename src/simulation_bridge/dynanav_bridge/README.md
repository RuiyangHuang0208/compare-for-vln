# DynaNav 兼容桥

本包把 DynaNav 的场景/episode 职责与 TIC-VLA 模型解耦。Isaac Sim 5.1 路径支持
Hospital、Office、Outdoor 和 Warehouse，统一发布对齐 RGB-D、相机内参、odom、TF、
仿真时钟、语言指令、目标点和 episode 元数据。`/nav_vel` 只能通过
`robot_controller` 进入当前选择的 B2-W locomotion backend。

`episode_manager` 读取 `config/episodes.yaml`；launch 层把选中 episode 的
`[x,y,yaw]` 出生位姿传给 Isaac Sim。只有 odom 和 RGB 都可用后才发布 START。RESET
会使模型历史、异步请求和旧轨迹在整条链路上失效。

`scripts/import_benchmark.py` 把上游 DynaNav 的角度制 yaw 和 85 个 episode 转成当前
配置。正式集合包括 Hospital 25、Office 25、Outdoor 10、Warehouse 25；
`episodes:=all` 只选择这 85 条，不包含本地 smoke episode。

批量比较时可以在不改动官方配置的前提下排除场景。例如当前阶段不使用 Outdoor：

```bash
cd /home/mifcom2/b2w/robot_vln_ws
DYNANAV_EXCLUDE_SCENES=outdoor \
  ./scripts/run_dynanav_full85_all.sh omnivla dualvln navila uninavid
```

此时每个模型计划运行 75 条回合（Hospital/Office/Warehouse 各 25 条）。先做相同的简单
无行人 smoke test：

```bash
DYNANAV_EPISODES=simple_forward_3m_standard \
  DYNANAV_NO_PEDESTRIANS=1 \
  ./scripts/run_dynanav_full85_all.sh --no-resume omnivla dualvln navila uninavid
```

`--exclude-scenes outdoor` 和 `DYNANAV_EXCLUDE_SCENES=outdoor` 只影响本次选择，不删除或
修改 Outdoor episode。评测器仍会先验证官方套件有完整 85 条，再应用筛选。

评测协议与 TIC-VLA 官方 DynaNav 保持一致：每回合独立启动 Isaac Sim，使用 episode
配置中的起点、目标、语言指令、seed、最大时长和每回合成功阈值，按 success、
navigation error、SPL、轨迹长度、时长和碰撞字段写入 JSON/CSV。当前 B2-W 版本将
官方 Nova Carter/Spot 执行器替换为固定的 SRU-ONNX B2-W locomotion；这只改变执行
平台，不改变 85 个 episode 的场景和任务定义。

接触判定沿用 DynaNav 的“持续接触”原则：Isaac PhysX 接触力必须连续超过 100 N
才会送入 `/simulation/collision`，默认去抖长度为 100 个仿真控制周期（B2-W 的 50 Hz 下约 2 秒）。
这会过滤 B2-W 腿/底盘落地时的单帧冲击，避免导航桥错误触发倒车恢复；真实持续碰撞仍会
记录到 episode JSON/CSV 并触发安全停车。可用
`DYNANAV_CONTACT_DEBOUNCE_STEPS` 调整，仅用于控制器标定，不建议在公平比较中改变。

跟踪器同时修正了一个坐标系错误：长轨迹的 Pure Pursuit 目标已经转换到 `base_link`，
因此不会再次减去世界坐标 yaw。这样 Office/Warehouse 的旋转出生朝向不会把直线路径误判
成大角度转向。该修复位于 `src/navigation_bridge/navigation_bridge/core.py`，并由旋转朝向
回归测试覆盖。

为适配 Isaac Sim 5.1，Outdoor 默认使用仓库中与官方 benchmark 同名的
`assets/outdoor_small.usd`。该 USD 仍可能引用 NVIDIA 在线 Rivermark 资产；如果
这些在线资产不可访问，Isaac Sim 会在日志中报告缺失引用，不能把该回合当作与官方
环境等价的结果。运行时可用 `DYNANAV_OUTDOOR_ASSET` 覆盖资产文件名。

旧实验目录 `outputs/ticvla/dynanav_full85_noped_fixed` 生成于本次修正前，使用了
`outdoor_new.usd`、统一 `1.5 m` 阈值和旧的短历史采样，不能与修正后的结果混合统计。

`DYNANAV_NO_PEDESTRIANS=1` 只关闭实际仿真人物，结果同时记录
`official_pedestrian_count` 和 `effective_pedestrian_count`。评估器以仿真时钟为主，
并带有 `max_duration * 2.0` 的 wall-clock 兜底，防止重型 USD 场景因 `/clock` 暂停而
永不落盘。

2026-08-21 已在 Isaac Sim 5.1 实际验证：

- 四类场景都能输出有效的 `640x480` RGB-D 和 odom；
- 四类场景各生成 1 个动态人物，2.4 秒内位移均为 `1.920 m`；
- `hospital_dynamic_001` dummy 完整闭环成功，最终误差 `0.497 m`；
- Office 和 Outdoor 复用 DynaNav 本地 USD，Hospital 与 Warehouse 默认使用 DynaNav
  官方 benchmark 的 Isaac 5.0 Nucleus 资产；可通过 `DYNANAV_ASSET_VERSION=5.1` 回退到
  Isaac 5.1 资产；
- Outdoor 的旧 Rivermark 装饰资源存在部分缺失警告，但核心场景、传感器、odom 和人物
  运动均正常。

上游 People/NavMesh 面向 Isaac Sim 5.0。在当前 IsaacLab 2.3.2 的引用式 stage 中启用
这些扩展会导致 schema 和 RenderSettings 错误，所以 5.1 默认使用确定性的轻量运动学
兼容人物。兼容实现沿用 DynaNav 的 seed、目标点集合和命令生成器，可被 RGB 相机看到，
并参与人物距离碰撞统计。仅在重新验证旧扩展时设置：

```bash
export ROBOT_VLN_TRY_ISAAC_PEOPLE=1
```

首次加载 NVIDIA 在线场景资产时必须保证资产服务器可访问；无法解析的资产会在 Isaac
进程日志中明确显示，不会被静默忽略。
