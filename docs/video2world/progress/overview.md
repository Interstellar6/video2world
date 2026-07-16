---
title: 进展与实测
id: video2world-progress-overview
category: 进展
visibility: public
updated: 2026-07-16
summary: Video2World 真实场景运行、资产门禁和浏览器验证记录入口。
tags:
  - Progress
  - QA
---

# 进展与实测

![Pillow scene query](../assets/pipeline/14-web-pillow-location.png "真实 bedroom_4 Web 运行时把问答解析、相机 focus、包围框与可旋转枕头视觉组件绑定到同一个对象 ID")

- [bedroom_4 真实资产与门禁记录（2026-07-16）](bedroom4-20260716.md)

当前完成了独立仓库、10-stage DAG、强类型 manifest、离线中英 scene query、Qwen2.5-VL-3B evidence caption、Web GLB/BVH/object-group runtime 与 pillow SAM3 delta。bedroom_4 production 大资产浏览器 QA 已通过；准确计数、交互、性能、截图及 visual-only 枕头边界见实测记录。
