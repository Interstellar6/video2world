---
title: Pipeline：视频到可交互世界
id: video2world-project-pipeline
category: 项目文档
visibility: public
updated: 2026-07-16
summary: Video2World 每个阶段的输入、输出、上下游关系和质量门禁；实现完成后补充真实命令、截图与产物统计。
tags:
  - Pipeline
  - 3DGS
  - Scene QA
---

# Pipeline：视频到可交互世界

本页是实现同步文档。当前先记录经审计确定的阶段边界；代码、真实命令、产物统计、Web QA 截图和 PDF 版将在实现验证后补齐，不把计划写成已完成结果。

```text
video -> cameras/depth -> PGSR scene -> SAM3 semantics -> 3D fusion
      -> captions/relations -> object completion -> alignment/collision
      -> world bundle -> robot interaction + scene cognition QA
```

| 阶段 | 上一步输入 | 本步输出 | 下一步 |
|---|---|---|---|
| ingest | video | frames、camera contract | DA3 / PGSR |
| geometry | frames、cameras | depth、dense point prior | PGSR、lifting |
| scene | frames、cameras、prior | scene Gaussian、render depth、TSDF mesh | semantics、runtime |
| perception | frames、词表 | SAM3 masks/tracks、caption evidence | 3D fusion |
| semantics | masks、depth、cameras、Gaussians | object clouds、bbox、semantic Gaussian | cognition、completion |
| cognition | instances、captions、bbox | relations、QA records | Web query |
| completion | RGBA、description contract | object Gaussian、GLB/OBJ candidate | placement |
| placement | scanned bbox、canonical assets | shared transform、pivot、carved static scene | collision、bundle |
| runtime | validated world bundle | visual/collision/robot/query interaction | browser QA |

