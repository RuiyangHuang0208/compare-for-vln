# OmniVLA 权重

完整模型必须放在 `omnivla-original/`，固定使用官方 `NHirose/omnivla-original` 的
step 120000 文件。启动脚本会逐项检查四个 safetensors shard、索引、配置、
`action_head--120000_checkpoint.pt` 和 `proprio_projector--120000_checkpoint.pt`。

2026-08-24 已下载完整权重到 `omnivla-original/`，总模型 shard 大小
15082474368 bytes，动作头 201513842 bytes，姿态投影器 67209720 bytes。
逐文件校验清单位于 `omnivla-original/SHA256SUMS`，验证命令：

```bash
cd /home/mifcom2/b2w/robot_vln_ws/checkpoints/vln/omnivla/omnivla-original
sha256sum -c SHA256SUMS
```

需要在其他机器重新下载时明确执行：

```bash
hf download NHirose/omnivla-original \
  --local-dir /home/mifcom2/b2w/robot_vln_ws/checkpoints/vln/omnivla/omnivla-original
```

不要用 `omnivla-original-balance`、CAST、edge 或量化权重替换本目录文件。
