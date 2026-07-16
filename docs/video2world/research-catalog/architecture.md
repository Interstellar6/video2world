---
title: Video2World 架构研究与接入决策
id: video2world-research-architecture
category: 调研目录
visibility: public
updated: 2026-07-16
summary: 基于 Holi-Spatial、PGSR、SAM3、Video2Mesh、EmbodiedGen V2 和既有 Web Demo 的实证审计，确定 Video2World 的分层架构、数据合同与实现路线。
tags:
  - Architecture
  - Holi-Spatial
  - PGSR
  - SAM3
  - EmbodiedGen V2
  - Scene QA
---

# Video2World 架构研究与接入决策

## 链接

- Holi-Spatial paper: https://arxiv.org/abs/2603.07660
- Holi-Spatial code: https://github.com/Visionary-Laboratory/Holi-Spatial
- PGSR code and paper: https://github.com/zju3dv/PGSR
- SAM 3 code: https://github.com/facebookresearch/sam3
- EmbodiedGen V2 paper: https://arxiv.org/abs/2607.07459
- EmbodiedGen V2 project: https://horizonrobotics.github.io/EmbodiedGen/
- EmbodiedGen V2 code: https://github.com/HorizonRobotics/EmbodiedGen

## 摘要要点

Video2World 不应成为把四个仓库复制到一起的巨型环境。更稳定的边界是：上游模型保持独立环境和版本，Video2World 用可恢复 stage adapter 调度它们，并用一个强类型 world manifest 连接输出。这样 GPU 训练、语义融合、物体生成、Web 渲染和文档发布可以分别失败、重跑和审计。

正式资产采用四层表示：场景 3DGS 视觉层、场景 mesh 碰撞层、语义/scene graph sidecar、独立物体组件层。视觉 PLY 不参与 raycast；TSDF、GLB 或简化 proxy 负责地面、障碍、选择和碰撞；Gaussian 与 mesh 子节点必须共享一个 scene-space 父变换和 pivot。

## 官方方法边界

| 组件 | 输入 | 输出 | Video2World 中的责任 | 不能误解成 |
|---|---|---|---|---|
| DA3 | 帧、相机上下文 | depth、稠密点云 | PGSR 和 2D-to-3D lifting 的几何 prior | 最终 mesh 或语义模型 |
| PGSR | 多视图 RGB、相机、初始化几何 | Gaussian PLY、rendered depth/normal、TSDF mesh | 场景视觉与连续表面 | 物体识别或 sim-ready collider |
| VLM discovery | 代表帧 | 开放词汇类别、caption 候选 | 发现类别、生成详细描述 | 精细像素分割 |
| SAM3 | 图像/视频、text/visual prompt | boxes、scores、masks、track identity | 2D 实例证据 | 3D bbox 或 scene graph |
| Video2Mesh fusion | masks、depth、cameras、Gaussians | object cloud、概率、semantic splats | 多视角语义融合与实例 lifting | Holi 官方 QA 真值 |
| EmbodiedGen V2/TRELLIS | 单物体 RGBA/文本条件链 | canonical Gaussian、mesh/GLB/OBJ | 物体视觉补全候选 | 自动对齐的真实扫描资产或精确 collider |
| Web runtime | world manifest 与 bundle | 可视、碰撞、查询、交互状态 | 最终用户体验和运行时 QA | 训练/重建后端 |

Holi-Spatial 官方 pipeline 分为几何优化、图像级感知、场景级 lifting/refinement 三段。其公开代码明确列出 DA3、PGSR、VLM 类别/区域发现、SAM3、3D bbox 后处理、instance caption 和 spatial QA generation。PGSR 本身是逐场景优化，不是单次前馈模型。SAM3 输出依然是 2D masks/boxes/scores，需要相机和深度才能变成 3D 实例。

EmbodiedGen V2 的论文目标比本项目当前实测更宽，包括 sim-ready assets、交互 affordance、任务世界和跨仿真器导出。当前本地证据只验证了 TRELLIS 单物体 Gaussian 与 mesh/GLB 导出；尚未验证其官方完整物理恢复、任务布局和策略训练链，文档与代码必须保留这个边界。

## 本地实证

| 证据 | 已验证结果 | 当前边界 |
|---|---|---|
| Holi fresh bedroom_4 | 80 帧；PGSR 871,317 Gaussians；TSDF 694,773 vertices / 1,351,454 faces；13 个 3D records；701,608 个语义选中 Gaussians | 详细 VLM caption、官方 spatial QA 未跑；坐标未公制标定 |
| EmbodiedGen auto completion | 9 个最终 Gaussian PLY；9 组 GLB/OBJ；文件和轻量几何 QA 通过 | 所有 GLB 非 watertight；缺少 scene transform、paired hash 和碰撞 QA；两扇 door 语义拒绝 |
| Web stable baseline `252a85c` | PGSR + TSDF 同帧展示、机器人 mesh 碰撞、相机与性能验证 | 无独立交互物体与 scene QA |
| Web experiment `a5af3ae` | 4 个独立 Gaussian、静态场景 carve、bbox proxy、双击 360 度、机器人跳跃 | collider 仍是 box；无 GLB loader、caption、查询或动态物体碰撞 |

用户点名的 Holi 权威样例保存在 Video2Mesh 的 `tmp_remote_results/holi_spatial_bedroom4_fresh_da3_sam3_pgsr_20260714_184217`。其 fresh 结果只有 bed、ceiling、floor、lamp、nightstand、plant、wall 和 window 类别，没有独立 pillow。旧 GroundingDINO 结果曾把 pillow 合并为 bed 的开放词汇标签，但不能据此生成 pillow bbox；正式示例需要重新分割/lift，或者明确回答“未识别为独立实例”。

## 候选工程方案

| 方案 | 优点 | 风险 | 结论 |
|---|---|---|---|
| 复制 Video2Mesh 单体 CLI | 现有命令最多 | 约 5 万行 CLI、历史分支耦合、硬编码远端路径、部分 `allow-incomplete` 会吞失败 | 不采用 |
| 单一 Conda/Docker 合并所有模型 | 表面上一条命令 | SAM3、PGSR、DA3、TRELLIS 的 CUDA/Python/GLIBC 约束冲突，镜像和权重巨大 | 不采用 |
| 独立 orchestrator + adapter + manifest | 保持上游边界，可逐阶段恢复、远端执行、校验 provenance | 需要先定义合同并写适配器 | 采用 |

## 推荐 Pipeline

```text
video
  -> ingest: frames + camera contract
  -> geometry: DA3 depth/point prior
  -> scene reconstruction: PGSR Gaussian + rendered depth + TSDF mesh
  -> perception: VLM vocabulary + detailed captions + SAM3 masks/tracks
  -> semantics: visibility-aware 2D-to-3D fusion + object clouds + semantic 3DGS
  -> cognition: bbox/OBB + scene relations + QA records
  -> completion: best-view RGBA + TRELLIS Gaussian and mesh
  -> placement: T_scene_from_asset + shared pivot + scanned-region carve
  -> collision: scene TSDF + per-object GLB/convex/box proxy
  -> bundle: hashes, counts, coordinate frames and quality gates
  -> web: Spark visual layer + Three.js collision layer + robot + query UI
```

每个 stage 写入独立 `stage.json`，包含 `status`、输入/输出 hash、命令、环境、开始/结束时间、日志和 issues。只有 required outputs 与 QA 全部通过才更新 world manifest；恢复执行基于内容 hash，而不是仅看目录是否存在。

## World Manifest 核心合同

```json
{
  "schema_version": "1.0",
  "scene": {
    "id": "bedroom_4",
    "coordinate_frame": "pgsr_native",
    "up_axis": "-Y",
    "metric_scale": null,
    "visual": {},
    "collider": {},
    "semantic_visual": {}
  },
  "objects": [
    {
      "id": "sam3_plant_01",
      "source_run_id": "...",
      "name": "plant 1",
      "aliases": ["plant", "植物"],
      "description": {},
      "evidence": {},
      "bbox": {},
      "visual": {},
      "render_mesh": {},
      "collider": {},
      "T_scene_from_asset": [],
      "pivot_scene": [],
      "relations": [],
      "quality_gates": {}
    }
  ]
}
```

对象主键必须是 `scene_id + source_run_id + object_id`。fresh Holi 与旧 EmbodiedGen 的实例集合并不相同，禁止只按裸 object ID 自动绑定。Gaussian 和 GLB 即使来自同一 RGBA/seed，也必须分别记录 hash、canonical bbox 与场景变换；没有对齐报告的资产只能是 candidate。

## 场景认知 QA

第一版采用确定性 scene graph resolver 保证离线可用：中英文 query 归一化后匹配 `id/name/category/aliases`，再分类为 location、appearance 或 unknown intent。location 从审核过的 `OnTopOf/SupportedBy/Near` 等关系生成答案；appearance 从结构化 caption 输出。成功解析后调用同一个 runtime selection API，聚焦 bbox 并高亮；拖拽旋转的是 Gaussian 与 GLB/mesh 共用父组。

可配置 VLM 只负责自然语言解析、消歧和答案润色，不持有几何真值。VLM 请求必须包含候选对象、关系、caption 和 evidence，输出结构化 object ID/intent/answer；provider 不可用或输出越界时回退到本地 resolver。不存在的实例必须 fail closed，不得生成伪 bbox。

## Web 接入判断

以 `252a85c` 保持相机、PGSR/TSDF 对齐和机器人稳定行为；从 `a5af3ae` 选择性迁移静态场景 carve、共享对象父组、bbox、focus、spin 和调试 API。新增 GLTF/OBJ loader、对象 collider BVH、query panel、drag rotation、scene graph resolver 与端到端 debug state。原场景中没有被识别/替换的 Gaussians继续停留在静态视觉层。

## 风险与门禁

- PGSR 长 splat、floaters 和场景边界拉丝只属于视觉质量问题，不能用激进清理破坏墙和地面。
- TSDF mesh 连续不等于 watertight 或可直接作为生产 collider；需要三角形规模、法线、退化面和接触测试。
- TRELLIS 资产必须做语义、尺度、对齐、支撑面、pair provenance 和 collider 简化 QA。
- pillow、被子和植物叶片属于 soft/deformable candidate；第一版只能用静态/kinematic proxy，不声称软体仿真。
- Web 端必须禁用 Gaussian raycast，并分别验证 canvas 非空、实际 asset ID/count、bbox、双击/拖拽、机器人碰撞、桌面和移动端布局。

## 接入结论

推荐路线可以复用现有最强资产，同时把实验性结果限制在显式 candidate/gate 内。Video2World 的核心价值不是再实现一遍 PGSR 或 SAM3，而是提供从视频到可查询交互世界的可追溯编排、跨模型合同、场景坐标对齐、运行时分层与质量闭环。

