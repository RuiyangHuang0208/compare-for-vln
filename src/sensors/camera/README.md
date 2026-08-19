# Camera Input for DualVLN

仿真闭环使用同一 Isaac 相机输出的对齐前向 RGB-D。这里不安装 RealSense 驱动；depth 由 `distance_to_image_plane` 直接产生。

本目录只保存工作区自己的输入契约，不复制或修改 MybotShop 相机驱动：

```text
/b2_366/sensor/front/camera_raw
  sensor_msgs/msg/Image
  1280 x 720, rgb8
  frame_id: front_camera
```

驱动仍由外部官方软件提供：

```text
../Unitree_B2_MybotShop_Software-main/
  src/mybotshop/b2_platform/src/b2_video.cpp
  src/mybotshop/b2_platform/config/b2_platform.yaml
```

DualVLN 的 resize、crop、normalize 和图像历史必须由 `third_party/InternNav` 的官方预处理完成，不能在相机驱动中重复实现。

## 接入前必须确认

- 实机 namespace 当前记录为 `b2_366`，启动前仍需用 `echo $B2_NS` 和 `ros2 topic list` 核对。
- 标称外参以 Unitree 官方 B2W URDF 的 `base_link -> f_oc_link` 为准；实机若有 MybotShop 定制安装仍需核对。
- 上游前后相机共用 `sensor/camera_info`。DualVLN 不消费 CameraInfo，因此当前不修改上游；若以后同时使用前后标定信息，需在工作区通过 remap 分离话题。
- 上游前后标定文件数值相同。启用图像矫正前需要逐相机复核。

配置见 [config/b2_front_rgb.yaml](config/b2_front_rgb.yaml)。

## Isaac Sim

模拟相机由工作区自己的 `play.py` 动态挂载，不修改 RobotLab task 或 B2W URDF：

```bash
./src/robot_controller/scripts/run_b2w_hospital.sh dualvln-sensors
```

运行中可从以下位置读取输入：

```python
rgb = env.unwrapped.scene["front_rgb"].data.output["rgb"]
depth = env.unwrapped.scene["front_rgb"].data.output["distance_to_image_plane"]
position = env.unwrapped.scene["robot"].data.root_link_pos_w
orientation = env.unwrapped.scene["robot"].data.root_link_quat_w
linear_velocity = env.unwrapped.scene["robot"].data.root_lin_vel_b
angular_velocity = env.unwrapped.scene["robot"].data.root_ang_vel_b
```

相机 smoke test：

```bash
./src/robot_controller/scripts/run_b2w_hospital.sh sensor-test
```

它会检查 RGB、对齐 depth、相机内参和 B2W odometry，并分别保存 `front_rgb.png` 与 `front_depth.png`。模拟参数见 [config/b2_front_rgb_sim.yaml](config/b2_front_rgb_sim.yaml)。

模拟 RGB 使用 Unitree 官方 B2W URDF 的 `f_oc_link` 标称变换：

```text
parent: base_link
xyz:    0.3993 0 -0.01576
rpy:   -1.5708 0 -1.5708
```

官方文件中的 LiDAR、IMU、前后 DC/OC 原始 frame 汇总在 `src/robot_description/config/b2w_sensor_frames_official.yaml`。

2026-08-19 在 Isaac Sim 5.1 Hospital 场景的实际 smoke test：

```text
RGB shape:       (480, 640, 3) PASS
RGB pixel std:   41.382        PASS
Depth shape:     (480, 640)    PASS
Depth valid:     100%          PASS
Depth range:     0.958-20.547m PASS
Intrinsics:      fx=fy=386.5   PASS
Camera forward:  PASS
Robot occlusion: PASS
Odometry finite: PASS
Policy obs/action dimensions unchanged: 57 / 16
```
