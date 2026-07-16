---
title: Video2World
id: video2world-home
category: Video2World
visibility: public
updated: 2026-07-16
summary: 从扫描视频构建分层、可交互、可查询 3D 世界的独立 pipeline 与实测文档库。
tags:
  - Video2World
  - 3DGS
  - Embodied AI
  - Scene Cognition
---

# Video2World

Video2World 把扫描视频转成一组职责明确、可以单独验证的数字资产：PGSR 3DGS 负责视觉，TSDF/GLB/简化 mesh 负责碰撞，SAM3 和多视角投影负责语义实例，EmbodiedGen V2/TRELLIS 提供独立物体补全候选，scene graph 和 caption sidecar 负责问答与场景认知。

## 文档导航

- [项目文档](project-docs/overview.md)
- [Pipeline 设计](project-docs/pipeline.md)
- [调研与架构决策](research-catalog/architecture.md)
- [进展与实测](progress/overview.md)

## 当前完成口径

一个完整 world bundle 必须同时记录视觉层、碰撞层、语义层、独立物体层、坐标变换、来源 hash、质量门禁和场景关系。仅有一个能打开的 PLY、GLB 或网页不视为 pipeline 完成。

