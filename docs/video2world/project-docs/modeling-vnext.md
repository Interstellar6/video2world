---
title: Video2World vNext 物体建模合同
id: video2world-project-modeling-vnext
category: 项目文档
visibility: public
updated: 2026-08-02
summary: vNext 十六阶段可执行图、物体组件语义、生成来源、独立碰撞体和物理参数的事实边界。
tags:
  - Pipeline
  - Object Completion
  - Physics
  - Provenance
---

# Video2World vNext 物体建模合同

vNext 把 Holi-Spatial 场景重建、组件级语义 lifting、背景重建、物体补全、mesh 后处理和物理 sidecar 拆成十六个可恢复 stage。配置入口是：

- `video2world/configs/modeling_vnext_pipeline.yaml`：规范 DAG 和产物路径；
- `video2world/configs/modeling_vnext.provider.example.yaml`：部署端 driver、模型目录和超时合同；
- `video2world/configs/modeling_vnext.site_profile.example.yaml`：`site-init` 使用的现场绑定模板。

```text
ingest
 -> da3
 -> pgsr
 -> object_proposals
 -> qwen_contracts
 -> component_segmentation
 -> semantic_lifting
 -> clean_plate

semantic_lifting + component_segmentation + qwen_contracts
 -> component_assembly
 -> optional image_completion
 -> object_completion
 -> mesh_postprocess
 -> physics_estimation

clean_plate + semantic_lifting + mesh_postprocess + physics_estimation
 -> placement
 -> bundle
 -> web
```

## 模型职责

| Stage | 输入事实 | 输出事实 | 不允许扩大声明 |
|---|---|---|---|
| `ingest` / `da3` / `pgsr` | 视频、统一相机、DA3 prior | RGB/depth、点云 prior、PGSR visual 3DGS、TSDF scene mesh | TSDF 不替代 3DGS 外观；DA3 不带对象语义 |
| `object_proposals` | 代表帧和场景 | GroundingDINO、Qwen-VL 或人工验证的 2D proposal | DINOv3 只可作为 feature encoder，不是 detector backend |
| `qwen_contracts` | proposal 与证据帧 | 结构化描述、正负 prompt、observed/completion-only 组件合同 | Qwen 不拥有 mask、深度或公制真值 |
| `component_segmentation` | proposal、组件 prompt、原始帧 | SAM3-I simple 或 SAM3 fallback 组件 mask | mask 只能声明 `observed_pixels_only`；SAM3-I 必须通过 proposal 几何门控，complex 暂不晋升 |
| `semantic_lifting` | 相机、DA3 depth/point prior、PGSR、组件 mask | 未补全对象点云、可选 observed surface、semantic Gaussian | observed surface 固定为不完整，不可写成 amodal mesh |
| `clean_plate` | 原始场景、对象 mask/lifting | Responses GPT Image 2 去对象 RGB、fresh DA3/PGSR/TSDF clean scene | API 候选固定人工复核且密钥不落盘；2D 通过不等于背景 3D 或 canonical visual QA 通过 |
| `component_assembly` | observed 组件证据 | `observed_union_not_amodal` 条件 | completion-only 部件只保留标签，不伪造成观测像素 |
| `image_completion` | 组件条件和 Qwen prompt | Responses GPT Image 2 可选完整物体参考图 | 产物标记 `generated_reference_not_observation`；API key 不落盘 |
| `object_completion` | observed multiview、assembly、可选生成图 | Stream3D、TRELLIS、TRELLIS2、SAM3D、Hunyuan3D-2.1 或 3D-Fixer 候选 | 3D-Fixer 是 in-place completion backend，不是通用 mesh repair stage |
| `mesh_postprocess` | 候选 mesh 与 observed lifting | repaired/simplified/UV/PBR asset 和显式 CoACD collider | visual mesh、Gaussian/PLY 和 collider 是不同角色；CoACD 要绑定 decomposition provenance |
| `physics_estimation` | mesh、语义和尺度证据 | dimensions/mass/friction/restitution sidecar | Qwen/category prior 保持 `unvalidated_estimate`；只有公制标定或测量证据可升级 |
| `placement` / `bundle` | clean scene、对象 visual/collider、physics | scene transform、碰撞 manifest、WorldManifest | 放回场景不自动证明 support、interpenetration、collision 或 visual QA |

## 对象资产合同

vNext 新增 `layered_visual_and_collider`：对象视觉可以是 render mesh、object Gaussian 或 object point cloud，碰撞代理必须是独立、已验证的 collider。CoACD 输出使用 `collider_topology=convex_decomposition`，并在 collider provenance 中写入 `decomposition=coacd`。这与历史 `separate_render_and_collider` 兼容模式以及统一 PBR GLB 模式都保持区分。

`PhysicalProperties` 保存 `dimensions_m`、`mass_kg`、`density_kg_m3`、静/动摩擦、恢复系数、置信度、尺度依据和 hash-bound evidence。未标定的场景可以保留类别先验，但不能据此宣称真实米制尺度或已验证动力学参数。

## 执行与验证边界

`site-init` 会依据 profile 的 `pipeline_id=modeling_vnext` 生成十六阶段的非空命令；`site-preflight` 只验证 checkout、权重、driver 和静态输入，不启动模型；`site-run` 才能记录 execute/adopt、输入输出 hash、provider receipt 和 stage state。preflight 与 run receipt 都直接记录 `pipeline_id`，避免把 canonical v1 和 vNext 结果混在一起。

截至 2026-08-02，仓库已经实现 DAG、typed semantic adapters、physics/collider schema、provider/site 配置和测试，并在独立 SeetaCloud run 中重新执行了 bedroom_4 的关键 provider。fresh 推理并不等于 canonical promotion：SAM3-I 批量、clean plate 相机保持、对象 PBR、物理校准和 room placement 仍有失败或未闭合门禁，bundle/Web 不会绕过这些失败。

逐阶段执行证据、图片和失败边界见 [Bedroom 4 的 vNext 建模全链实验](../progress/modeling-vnext-20260802.md)。
