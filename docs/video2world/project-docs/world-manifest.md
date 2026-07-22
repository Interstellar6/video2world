---
title: World Manifest 与质量门禁
id: video2world-project-world-manifest
category: 项目文档
visibility: public
updated: 2026-07-22
summary: Video2World world-manifest-1.0.0 与 Web 投影如何表达 mesh-first 对象、统一 PBR GLB、collision topology、hash、provenance 和发布门禁。
tags:
  - Schema
  - Provenance
  - Quality Gates
---

# World Manifest 与质量门禁

World manifest 是 Pipeline 和 Web 的事实边界。代码以 Pydantic 模型定义合同，并生成 `schemas/world-manifest.schema.json`；提交的 JSON Schema 由测试锁定，不能与模型静默漂移。

## 两层 manifest 合同

仓库刻意保留两层合同，避免 Web 为了加载方便而吞掉上游 provenance：

| 合同 | 角色 | 验证入口 |
|---|---|---|
| `world-manifest-1.0.0` | Pipeline 的规范产物；保存完整资产身份、hash、坐标、来源和五项 gate | `video2world validate` 与 JSON Schema |
| `video2world-web-manifest-1.0.0` | 面向浏览器的部署投影；保存 URL、相机、统一 PBR GLB/旧兼容组件、碰撞 topology 和场景问答索引 | `web/web-manifest.js` 在任何网络资产加载前 fail-fast 验证 |

新运行的目标数据流是 `WorldManifest -> web stage -> Web manifest`。Web manifest 不是新的事实源；它至少通过 `sourceWorld.worldId/runId/adoptionMode` 记录来源，canonical-first 发布还应给出可解析的 `manifestUri`。当前 Bedroom4 是历史资产迁移样例：旧版 Web demo 资产先被采用并接受浏览器 QA，再由 `materialize_bedroom4_world.py` 固化为规范 WorldManifest。这个例外由 Web manifest 中的 `sourceWorld.adoptionMode=legacy_web_assets_adopted_then_canonical_manifest_validated` 明示，仅用于迁移，不能作为新视频运行的默认顺序。

因此，看到 Bedroom4 的浏览器 bundle 已通过，不等于任意新视频的上游 provider 已经完成；后者仍必须产生自己的 stage receipt、规范 manifest 和 Web 投影，不能复用 Bedroom4 的通过状态。

![World bundle evidence](../assets/pipeline/12-pillow-projection.png "World manifest 把视觉、语义、几何、证据、坐标、hash 与质量门禁绑定到同一个可审计对象身份")

## Scene layer

`scene` 必须同时包含：

- `visual`：PGSR/GraphDECO Gaussian PLY；
- `collider`：TSDF、GLB 或经验证的 scene mesh；
- `semantic_visual`：保留 Gaussian 字段并增加 `object_id/object_probability` 的 semantic PLY；
- `coordinate_system`：frame ID、up axis、handedness、units；
- 可选 scene bounds。

三类资产都有 URI、SHA-256、字节数、role、status 与 provenance。`manifest_status=validated` 时，三层资产必须全部是 `validated`。

## Object identity

对象使用 `world_id::source_run_id::object_id` 作为 scoped identity。例如：

```text
bedroom_4::holi_fresh_20260714::sam3_plant_01
```

原因是 fresh Holi、旧 EmbodiedGen 和独立 delta 可能重复使用 `sam3_plant_01`，但它们未必指向同一个物理实例。只按裸 ID 自动绑定会把错误资产放进场景。

## Object layer

每个对象可以包含：

- 中英名称、category 与 aliases；
- short/detailed/appearance/location 描述及 provider/model/evidence；
- source image/mask/可选 point cloud 与 2D/3D bounds；
- mesh-first `render_mesh`、可选 object Gaussian `visual`、以及 collision-enabled 时的 `collider`；
- `T_scene_from_asset`、scene pivot 与 scale；
- scene relations；
- interaction policy；
- file/semantic/alignment/collision/visual 五项 gate 与可追溯 limitation。

规范 WorldManifest 保留 `visual`、`render_mesh`、`collider` 三个语义角色，是为了兼容历史资产并保持 provenance。新 unified 对象不需要 object Gaussian，因此 `visual` 可以为空；`render_mesh` 与 `collider` 可以引用同一个 PBR GLB 的同一 URI/hash。Web stage 会把这两个 canonical 角色折叠成一个 `collision.asset`，而不是把文件复制两次。

collision-enabled 的 unified PBR GLB 不能只写 `status=passed`。World manifest 要求 `alignment`、`collision` 与 `visual` 三个 gate 的 passed 状态都带 `report_uri`，分别指向 scene fit / 支撑穿模审核、BVH collision 审核和六视图或浏览器 visual QA。这样对象进入 Web 交互层之前，pivot、摆放、碰撞和视觉审核都有可追溯证据，而不是手写通过。

## Web manifest 的 unified GLB 合同

新独立对象必须显式声明 `collision.mode=unified-glb`。最小形状如下；刻意没有 `visual`、`collision.renderAsset` 和 `colliderProxy`：

```json
{
  "id": "sam3_pillow_front",
  "placement": {
    "pivot": [-0.9132, -0.1425, 14.7607],
    "scale": [1, 1, 1],
    "generatedCenter": [0, 0, 0]
  },
  "collision": {
    "mode": "unified-glb",
    "asset": {
      "url": "./objects/sam3_pillow_front.scene-fit.glb",
      "fileName": "sam3_pillow_front.scene-fit.glb",
      "format": "gltf-binary",
      "faces": 97082,
      "finite": true,
      "nondegenerate": true,
      "windingConsistent": true,
      "watertight": false
    },
    "topology": "surface_bvh",
    "characterCollision": true,
    "gate": {
      "status": "passed",
      "surfaceCollision": "passed"
    }
  },
  "interaction": {
    "kind": "spin",
    "degrees": 360,
    "durationMs": 900,
    "drag": "horizontal_yaw"
  }
}
```

实际 manifest 还必须携带部署资产的 size/hash 等字段；上例只突出统一对象语义。`generatedCenter=[0,0,0]` 表示 scene fit 已把局部中心烘焙好，避免 Web 再做一次隐式 recenter/scale。

### 逻辑层级祖先

当本轮只采用子对象的 TRELLIS2 scene-fit 结果、但其父对象没有被替换时，Web manifest 可以保留一个逻辑祖先节点。例如直接采用三个 pillow 的 PBR GLB、暂不采用 bed mesh 时，`bed` 仍可作为 unrendered parent 保留，保证“移动床会带动枕头”的 scene graph 语义可以被表达；但它不能被渲染、选中、碰撞或作为旧 proxy fallback。

逻辑祖先必须满足：

- `logicalHierarchyOnly=true`；
- `logicalRole="unrendered_unselectable_hierarchy_ancestor"`；
- `independentlyMovable=false`；
- `childObjectIds` 与所有 child 的 `parentObjectId` 精确一致；
- 不允许声明 `placement`、`collision`、`visual`、`renderAsset`、`colliderProxy`、`interaction`、`carve` 或 `sourceAnchor`。

子对象仍使用 `semanticGranularity="independent_child_asset"`、`movesWithParent=true`、`independentlyMovable=true`。运行时会创建不可选中的父组并把 child 挂在其下；选择、focus、drag、双击和机器人碰撞只作用于真实 adopted child。这样可以在 direct TRELLIS2 refit 阶段保留嵌套关系，同时避免把未采用父对象伪装成已验证 runtime asset。

### Collision topology

| topology | 必须满足 | 可以声称 |
|---|---|---|
| `surface_bvh` | binary GLB；1-100,000 faces；finite；nondegenerate；winding consistent；passed gate；`characterCollision=true` | 同一可见 mesh 的表面阻挡；non-watertight 可用，但不声称 volume/inside-outside |
| `closed_volume` | `surface_bvh` 的全部条件，加 `watertight=true` | 闭合拓扑；不自动等于软体/动力学验证 |

unified GLB 加载或 gate 失败时必须 fail closed，不生成 box、不使用 selection proxy 顶替，也不能把对象降级后仍写成 collision passed。

## 交互发布门禁

交互对象分成三种发布历史/状态，不能把门禁混为一谈：

- **unified collision-enabled（新默认）**：同一 PBR GLB 负责 visual/logic/MeshBVH surface，五项 gate 与 topology 条件必须全部通过；
- **separate collision-enabled（旧 production）**：Gaussian/mesh 与 simplified collider 同步运动，继续作为兼容模式；
- **visual-only（旧 production/显式选择）**：允许选择、focus、拖拽或 `spin_360`，但 `collision_enabled=false`、collider 为空且 collision gate 保持 `not_tested`。2026-07-16 production 枕头属于这一类。

collision-enabled 对象的最低要求是：

| Gate | 最低要求 |
|---|---|
| file | PBR GLB 存在、非空、hash 与声明一致；unified 模式不要求 object Gaussian |
| semantic | 生成资产类别与扫描证据一致，没有错误门/柜等替换 |
| alignment | 显式 scene transform 与 pivot；bbox/支撑面检查通过 |
| visual | 场景内无遮挡重影、旋转无漂移、整体形状与主色类别可接受；minor hallucination 有 limitation |
| collision | topology 技术条件、BVH 命中、机器人 probe 与 stale-face 检查通过 |

collision-enabled 对象只要一个 blocking gate 是 failed/not-tested，就不能成为“已验证可碰撞对象”。轻微背面纹理/材质幻觉可表现为 `status=passed` 并在 `reason`、report 或 limitation sidecar 中记录；明显形变、主色类别错误、缺面/片状、部件断裂、悬空或显著穿模仍必须失败。visual-only 对象可以在 file、semantic、alignment、visual 通过后发布视觉交互，但不得携带 collider，也不得把 collision 标为 passed。候选资产仍可记录，但不得伪装成 production-ready。

## 坐标与单位

bedroom_4 当前 up axis 为 `-Y`，handedness 为 right，单位状态是 `scene_scale_not_metric`。该状态禁止填写 `metric_scale`。完成真实尺度标定后才可改成 meters/centimeters，并记录标定方法与误差。

unified PBR GLB 的 PBR visual、逻辑对象和 BVH 必须共享同一个父 transform；runtime 旋转父组而不是复制或逐顶点修改。旧模式中的 Gaussian、render mesh 和 collider 也必须共享父 transform。

## 当前真实候选边界

TRELLIS2 front pillow PBR GLB 有 60,237 vertices / 97,082 faces、PBR material、finite、nondegenerate、winding consistent、non-watertight，因此只适合 `surface_bvh`。source-camera mask IoU 为 `0.709810`、bbox IoU 为 `0.924577`、中心误差为 `4.402 px`。这些 receipt 支持 object-local/source-camera gate。

2026-07-22 的 direct local 三枕头候选进一步通过了真实浏览器合同 QA：`examples/bedroom4/completion/direct-trellis2-refit/candidate/browser-qa-report.json` 状态为 `passed`，manifest SHA-256 为 `c6ebd4820028616597568406683b79d88a5d6434b2883096099ef18337c29623`，desktop `1440x900` 与 mobile `390x844` 都加载原始 TSDF collider `1,351,454` faces、三个 unified PBR GLB colliders `293,538` faces，确认 PBR material、MeshBVH、pointer focus、yaw 后 visual/collision 同步、机器人 object blocking、无 degraded/proxy fallback，并确认逻辑 `sam3_bed_01` 祖先不能 focus、不能提供 screen point、没有 visual/collider。该报告的 scope 仍是 `direct_original_uncarved_scene_local_qa_only`：没有执行 clean plate、没有 carve 原始 PGSR、没有最终支撑/显著穿模质量放行，也没有 promotion。

SDXL clean-plate seed `2026071701` 已由用户以 `pass_with_known_limitation` 放行当前 demo，但 receipt 同时声明没有证明 object-free background、multiview consistency 或 occluded-bed geometry。这个 scope-limited 决定不能被 canonical manifest 扩大成通用背景重建通过。

## 本地验证

```bash
uv run video2world schema --output schemas/world-manifest.schema.json
uv run video2world validate /absolute/world.json
```

本地 URI 相对 manifest 解析并重新计算 hash。HTTP 资产默认不被视为内容已验证；只有显式 `--allow-remote` 才允许跳过本地内容检查，且 remote count 会保留在报告里。
