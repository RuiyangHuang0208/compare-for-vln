# 模型无关评测

`goal_monitor` 使用 `/ground_truth/odom` 和 `/episode/goal` 判断成功，VLN 模型本身不参与
成功判定。到达目标后会发布 STOP、把 `/nav_vel` 归零并标记 SUCCESS。`evaluator` 使用
`/clock` 统计仿真时长，并记录路径长度、SPL、最终误差、碰撞、卡住/恢复、模型推理延迟
和控制频率。

### TIC-VLA 官方 benchmark 对齐

官方 DynaNav 的成功条件是：机器人真实 XY 位置到目标点的距离严格小于该回合的
`success_threshold`（默认 `1.5 m`）；超时使用每个回合配置的仿真秒数。工作区的
`goal_monitor` 和 `episodes.yaml` 已按此规则执行，边界值等于阈值不会判定成功。85 条
回合的 `seed`、场景、起点、朝向、目标、指令、`timeout` 和 `num_people` 均来自
`third_party/TIC-VLA/DynaNav/configs/benchmark_full.yaml`；只将官方角度从度转换为 ROS
约定的弧度。

工作区仍固定使用 B2-W 和 SRU-ONNX 低层执行器，因此与官方示例中的 Nova Carter 机器人
不同；这属于机器人执行器差异，不会改变成功判定或回合参数。需要严格使用官方人数和
全部场景时运行：

```bash
./scripts/run_dynanav_full85_all.sh --official --no-resume \
  ticvla omnivla dualvln navila uninavid
```

`--official` 会关闭“无行人”和“排除 Outdoor”的本地快捷默认值，并恢复官方每回合人数。
它同时启用模型原生高层执行；如果只想在本地筛选场景/行人的情况下启用原生高层，使用
`--native`。两种情况下底层均固定为 B2-W SRU-ONNX。
若只想先验证官方单回合调度而不启动 Isaac Sim：

```bash
ros2 launch robot_bringup dynanav_benchmark.launch.py \
  model:=dummy episodes:=all dry_run:=true headless:=false
```

日常快速回归仍可使用脚本默认的无行人、排除 Outdoor 模式；该结果应标记为 smoke，不能
与 `--official` 结果混合统计。

单回合结果：

```text
outputs/<model>/<experiment>/<episode_id>.json
```

实验汇总：

```text
outputs/<experiment>/summary.csv
```

新生成的 JSON 使用严格标准格式：缺失指标写为 `null`，不会写非标准的 `NaN` 或
`Infinity`。

公平对比由工作区 `configs/fair_comparison.yaml` 定义。结果中的 `comparison_track` 区分
`rgb_only`、`rgb_d`、`debug` 和 `untracked`，`model_inputs` 保存模型实际高层输入。只有同一
track、同一 episode、同一共享配置的结果才能合并统计。公平运行锁定 `1.0 m/s`；使用
`comparison_track:=none` 的手动调速结果不得进入正式汇总。

碰撞输入为 `/simulation/collision`。当前 Isaac contact telemetry 会保存接触时间、力和
机器人本体 link。net-force API 无法识别静态接触物体，因此对方对象记录为
`unknown_static_geometry`；动态人物由 DynaNav bridge 根据几何距离单独识别。原始接触
记录仍保留，便于后处理区分真实障碍碰撞和持续地面接触误报。

`benchmark_runner` 为每个 episode 启动独立 launch 子进程。`episodes:=all` 必须精确
解析为 85 条：Hospital 25、Office 25、Outdoor 10、Warehouse 25。正式运行前先验证：

```bash
ros2 launch robot_bringup dynanav_benchmark.launch.py \
  model:=dummy experiment:=dry_validation episodes:=all dry_run:=true
```

dry-run 只校验场景、出生点、目标点、资产和最终启动命令，不启动 Isaac Sim。当前已验证
85 条调度和一个真实子回合的结束、结果发现及 CSV 汇总；完整 85 回合实验尚未执行。
