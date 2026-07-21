---
title: Video2World
id: video2world-home
category: Video2World
visibility: public
updated: 2026-07-22
summary: 从扫描视频构建分层、可交互、可查询 3D 世界：场景保留 PGSR/TSDF，独立对象优先使用统一 PBR GLB，并包含可审计的背面、遮挡与 clean-plate 补全。
tags:
  - Video2World
  - 3DGS
  - Embodied AI
  - Scene Cognition
---

# Video2World

Video2World 把扫描视频转成一组职责明确、可以单独验证的数字资产：场景整体由 PGSR 3DGS 负责视觉、TSDF mesh 负责静态碰撞；独立对象默认由一个 PBR GLB 同时承担可见表面、select/drag/spin 逻辑与 MeshBVH character surface collision，不要求额外 object Gaussian/point cloud 或 visual/collider proxy。SAM3 和多视角投影负责语义实例，evidence-based completion 在多视角重建、类别先验、CAD 与生成后端之间路由，scene graph 和 caption sidecar 负责问答与场景认知。

![Video2World production world](assets/pipeline/13-web-production-overview.png "真实 bedroom_4 production world：PGSR 场景视觉、独立对象组件、TSDF/GLB 碰撞、机器人与场景问答在同一 Web runtime 中运行")

上图对应 2026-07-16 已验证的旧多表示 production。当前 direct TRELLIS2 no-clean-plate 本地候选已把三个枕头作为统一 PBR GLB 加入真实 Web runtime：desktop `1440x900` 与 mobile `390x844` 浏览器 QA 均通过，三对象共 293,538 个 object collider faces，机器人分别被 `sam3_pillow_front`、`sam3_pillow_left`、`sam3_pillow_right` 的 unified GLB 表面阻挡，逻辑 `bed` 祖先拒绝 focus 且不加载 collider。这个报告仍只是 `direct_original_uncarved_scene_local_qa_only`，没有证明 clean plate、静态 carve、支撑/穿模最终质量或 production promotion，不能用旧 production 报告替代，也不能把本地候选扩大成通用补全完成。

## 文档导航

- [项目文档](project-docs/overview.md)
- [Pipeline 设计](project-docs/pipeline.md)
- [通用遮挡、背面与背景分层补全](project-docs/completion.md)
- [安装、配置与恢复执行](project-docs/getting-started.md)
- [World Manifest 与质量门禁](project-docs/world-manifest.md)
- [Web Runtime](project-docs/web-runtime.md)
- [调研与架构决策](research-catalog/architecture.md)
- [进展与实测](progress/overview.md)

## 当前完成口径

一个完整 world bundle 必须同时记录场景视觉/碰撞/语义层、独立物体层、坐标变换、来源 hash、质量门禁和场景关系。unified 对象还必须显式记录 `surface_bvh|closed_volume`：non-watertight 只能做表面阻挡，`closed_volume` 必须 watertight。轻微不可见面纹理/材质幻觉可以 passed 并记录 limitation；明显形变、主色错误、缺面/片状、悬空或显著穿模仍会阻断。仅有一个能打开的 PLY、GLB 或网页不视为 pipeline 完成。
