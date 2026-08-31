# Uni-NaVid 权重

本目录只保存本机权重，不提交大文件：

```text
uninavid/
├── uninavid-7b-full-224-video-fps-1-grid-2/  # 官方 7B checkpoint
└── eva_vit_g.pth     # 官方 EVA-CLIP 视觉编码器
```

推理服务在 `/tmp` 创建只含符号链接和解析后 `config.json` 的运行时目录，因此不会修改官方 checkpoint。
