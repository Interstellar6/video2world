---
title: Video2World 架构研究与接入决策
id: video2world-research-architecture
category: 调研目录
visibility: public
updated: 2026-07-22
summary: 基于 Holi-Spatial、PGSR、SAM3、Video2Mesh、EmbodiedGen V2 和既有 Web Demo 的实证审计，确定场景 PGSR/TSDF 分层与独立对象统一 PBR GLB 的数据合同。
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

正式资产采用四层表示：场景 3DGS 视觉层、场景 mesh 碰撞层、语义/scene graph sidecar、独立物体组件层。场景 PGSR PLY 不参与 raycast，静态 TSDF 负责地面和结构障碍；新独立对象优先用一个 PBR GLB 同时承担可见表面、选择/旋转逻辑和 MeshBVH character surface collision。对象 Gaussian/point cloud 是可选 evidence，不需要视觉代理与碰撞代理各存一份。

![Layered scene evidence](../assets/pipeline/06-semantic-gaussian.png "语义 Gaussian 是场景语义 sidecar 的可视化，不应和实例 identity、碰撞代理或独立物体生成混为一层")

## 官方方法边界

| 组件 | 输入 | 输出 | Video2World 中的责任 | 不能误解成 |
|---|---|---|---|---|
| DA3 | 帧、相机上下文 | depth、稠密点云 | PGSR 和 2D-to-3D lifting 的几何 prior | 最终 mesh 或语义模型 |
| PGSR | 多视图 RGB、相机、初始化几何 | Gaussian PLY、rendered depth/normal、TSDF mesh | 场景视觉与连续表面 | 物体识别或 sim-ready collider |
| VLM discovery | 代表帧 | 开放词汇类别、caption 候选 | 发现类别、生成详细描述 | 精细像素分割 |
| SAM3 | 图像/视频、text/visual prompt | boxes、scores、masks、track identity | 2D 实例证据 | 3D bbox 或 scene graph |
| Video2Mesh fusion | masks、depth、cameras、Gaussians | object cloud、概率、semantic splats | 多视角语义融合与实例 lifting | Holi 官方 QA 真值 |
| EmbodiedGen V2/TRELLIS | 单物体 RGBA/文本条件链 | canonical PBR GLB/mesh；可选 Gaussian/point export | mesh-first 物体补全候选 | 自动对齐的真实扫描资产、真实不可见面或天然 closed volume |
| Web runtime | world manifest 与 bundle | 可视、碰撞、查询、交互状态 | 最终用户体验和运行时 QA | 训练/重建后端 |

Holi-Spatial 官方 pipeline 分为几何优化、图像级感知、场景级 lifting/refinement 三段。其公开代码明确列出 DA3、PGSR、VLM 类别/区域发现、SAM3、3D bbox 后处理、instance caption 和 spatial QA generation。PGSR 本身是逐场景优化，不是单次前馈模型。SAM3 输出依然是 2D masks/boxes/scores，需要相机和深度才能变成 3D 实例。

EmbodiedGen V2 的论文目标比本项目当前实测更宽，包括 sim-ready assets、交互 affordance、任务世界和跨仿真器导出。当前本地证据验证了 TRELLIS 单物体 PBR GLB/mesh，以及旧 run 的 Gaussian 导出；尚未验证其官方完整物理恢复、任务布局和策略训练链，文档与代码必须保留这个边界。

## 本地实证

| 证据 | 已验证结果 | 当前边界 |
|---|---|---|
| Holi fresh bedroom_4 | 80 帧；PGSR 871,317 Gaussians；TSDF 694,773 vertices / 1,351,454 faces；13 个 3D records；701,608 个语义选中 Gaussians | 基础 fusion 实际 `min_votes=1`；详细 VLM caption、官方 spatial QA 未跑；raw Gaussian 数值健康为 unsafe；本地归档不是自包含 run bundle |
| EmbodiedGen auto completion（旧 production） | 9 个最终 Gaussian PLY；9 组 GLB/OBJ；文件和轻量几何 QA 通过 | 资产实际引用较早 2026-07-13 Holi run，不是 07-14 fresh；三份表示模式只作为历史兼容 |
| TRELLIS2 direct 三枕头 local candidate | front/left/right 三个 PBR GLB 均以 `unified-glb` 进入 Web；object collider faces 共 293,538；desktop/mobile browser QA、PBR material、MeshBVH、pointer focus、yaw 同步、robot object blocking 与逻辑 bed 拒绝 focus 均通过 | `surface_bvh` only；scope 是 `direct_original_uncarved_scene_local_qa_only`，未 clean plate、未静态 carve、未 production promotion |
| Web stable baseline `252a85c` | PGSR + TSDF 同帧展示、机器人 mesh 碰撞、相机与性能验证 | 无独立交互物体与 scene QA |
| Web experiment `a5af3ae` | 4 个独立 Gaussian、静态场景 carve、bbox proxy、双击 360 度、机器人跳跃 | collider 仍是 box；无 GLB loader、caption、查询或动态物体碰撞 |

用户点名的 Holi 权威样例保存在 Video2Mesh 的 `tmp_remote_results/holi_spatial_bedroom4_fresh_da3_sam3_pgsr_20260714_184217`。其 fresh 结果没有独立 pillow。为避免沿用旧 GroundingDINO 中混入 bed 的 pillow 标签，本项目在不覆盖 fresh 的前提下执行了隔离 SAM3 delta；209,479 点的三枕头 ensemble 是 2026-07-16 production 查询/focus 的历史 visual-only 证据。之后 front pillow 已生成独立 TRELLIS2 PBR GLB，source-camera mask IoU `0.709810`、bbox IoU `0.924577`、中心误差 `4.402 px`，但这些指标不替代场景浏览器/支撑/穿模 QA。

fresh 原始 PGSR/semantic PGSR 的 Gaussian 数值健康审计为 unsafe：scale p99 约 `0.6656`、elongation p99 约 `6.25e11`、rotation norm error p99 约 `0.858`。远端曾生成 viewer-safe SuperSplat 派生物，但当前本地 fresh 归档没有同步该文件。当前 Video2World carved raw PGSR 已在 Spark/Chrome 的真实页面与性能门禁中通过，这只证明本运行时可用，不等价于对任意 SuperSplat/Spark 版本都 viewer-safe；raw 与 viewer-safe 派生物必须在 manifest 中分开登记。

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
  -> completion: best-view RGBA + mesh-first PBR GLB
  -> placement: T_scene_from_asset + shared pivot + scanned-region carve
  -> collision: scene TSDF + same object GLB MeshBVH surface
  -> bundle: hashes, counts, coordinate frames and quality gates
  -> web: Spark scene layer + Three.js unified PBR objects + robot + query UI
```

每个 stage 写入独立 `stage.json`，包含 `status`、输入/输出 hash、命令、环境、开始/结束时间、日志和 issues。只有 required outputs 与 QA 全部通过才更新 world manifest；恢复执行基于内容 hash，而不是仅看目录是否存在。

## World Manifest 核心合同

```json
{
  "schema_version": "1.0.0",
  "world_id": "bedroom_4",
  "run_id": "...",
  "scene": {
    "coordinate_system": {
      "frame_id": "pgsr_native",
      "up_axis": "-Y",
      "handedness": "right",
      "units": "scene_scale_not_metric",
      "metric_scale": null
    },
    "visual": {},
    "collider": {},
    "semantic_visual": {}
  },
  "objects": [
    {
      "id": "sam3_plant_01",
      "source_run_id": "...",
      "scoped_id": "bedroom_4::...::sam3_plant_01",
      "name": {"zh": "植物 1", "en": "plant 1"},
      "category": "plant",
      "aliases": ["plant", "植物"],
      "description": {},
      "evidence": {},
      "bbox_scene": {},
      "visual": null,
      "render_mesh": {"uri": "object.scene-fit.glb"},
      "collider": {"uri": "object.scene-fit.glb"},
      "transform_scene_from_asset": {},
      "relations": [],
      "quality_gates": {}
    }
  ]
}
```

对象主键必须是 `scene_id + source_run_id + object_id`。fresh Holi 与旧 EmbodiedGen 的实例集合并不相同，禁止只按裸 object ID 自动绑定。新 unified 模式中 canonical `render_mesh` 与 `collider` 可以引用同一个 PBR GLB/hash，Web 投影显式写 `collision.mode=unified-glb`、`surface_bvh|closed_volume`、face 数与技术 gate；对象 Gaussian/point cloud 可缺省。旧 Gaussian 与 GLB 来自不同 decoder 时仍必须分别记录 hash。

## 场景认知 QA

第一版采用确定性 scene graph resolver 保证离线可用：中英文 query 归一化后匹配 `id/name/category/aliases`，再分类为 location、appearance 或 unknown intent。location 从审核过的 `OnTopOf/SupportedBy/Near` 等关系生成答案；appearance 从结构化 caption 输出。成功解析后调用同一个 runtime selection API，聚焦 bbox 并高亮；unified 模式直接拖拽同一个 PBR GLB 父组。

可配置 VLM 只负责自然语言解析、消歧和答案润色，不持有几何真值。VLM 请求必须包含候选对象、关系、caption 和 evidence，输出结构化 object ID/intent/answer；provider 不可用或输出越界时回退到本地 resolver。不存在的实例必须 fail closed，不得生成伪 bbox。

## Web 接入判断

以 `252a85c` 保持相机、PGSR/TSDF 对齐和机器人稳定行为；从 `a5af3ae` 选择性迁移静态场景 carve、对象父组、bbox、focus、spin 和调试 API。新路径通过 GLTFLoader 一次加载 PBR GLB，同时保留 material 并从同一 mesh 建 BVH；`collision.mode=unified-glb` 禁止单独 visual/renderAsset/colliderProxy，也禁止 box fallback。原场景中没有被识别/替换的 Gaussians 继续停留在静态视觉层。

## 风险与门禁

- PGSR 长 splat、floaters 和场景边界拉丝只属于视觉质量问题，不能用激进清理破坏墙和地面。
- raw PGSR 数值健康为 unsafe 时，不能直接宣称通用 viewer-safe；应同步或重建带独立 hash/报告的安全化派生物，raw 仅保留为训练/审计来源。
- TSDF mesh 连续不等于 watertight 或可直接作为生产 collider；需要三角形规模、法线、退化面和接触测试。
- TRELLIS PBR GLB 必须做语义、尺度、对齐、支撑面、finite/nondegenerate/winding、face budget 与 topology QA；`surface_bvh` 不要求 watertight，但只能声称表面阻挡。
- pillow、被子和植物叶片属于 soft/deformable candidate；第一版只把同一 GLB 当作静态/kinematic 表面，不声称软体仿真或体积恢复。
- 外观门禁按 severity：轻微不可见面纹理/材质幻觉可 `passed + limitation`；明显形变、主色类别错误、缺面/片状、部件断裂、悬空或显著穿模必须 retry/reject，最多三轮。
- Web 端必须禁用 scene Gaussian raycast，并验证 canvas 非空、PBR material、单次 GLB load、实际 asset ID/face count、bbox、双击/拖拽、同 mesh 机器人碰撞、无 fallback、桌面和移动端布局。当前 direct TRELLIS2 三枕头候选已通过 local browser QA，但仍不能引用旧 production 或 direct local 报告替代 clean-scene production promotion、背景补全和最终支撑/穿模 QA。

## 接入结论

推荐路线可以复用现有最强资产，同时把实验性结果限制在显式 candidate/gate 内。Video2World 的核心价值不是再实现一遍 PGSR 或 SAM3，而是提供从视频到可查询交互世界的可追溯编排、跨模型合同、场景坐标对齐、运行时分层与质量闭环。
