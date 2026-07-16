---
title: World Manifest 与质量门禁
id: video2world-project-world-manifest
category: 项目文档
visibility: public
updated: 2026-07-16
summary: Video2World world-manifest-1.0.0 的场景层、对象层、坐标、hash、provenance 和交互发布门禁。
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
| `video2world-web-manifest-1.0.0` | 面向浏览器的部署投影；保存 URL、相机、交互组件、碰撞代理和场景问答索引 | `web/web-manifest.js` 在任何网络资产加载前 fail-fast 验证 |

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
- source image/mask/point cloud 与 2D/3D bounds；
- object Gaussian visual、render mesh、collider；
- `T_scene_from_asset`、scene pivot 与 scale；
- scene relations；
- interaction policy；
- file/semantic/alignment/collision/visual 五项 gate。

## 交互发布门禁

交互对象分成两类，不能把它们的门禁混为一谈：

- **collision-enabled**：Gaussian/mesh 与 collider 同步运动，机器人可以碰撞，五项 gate 必须全部通过；
- **visual-only**：允许选择、focus、拖拽或 `spin_360`，但 `collision_enabled=false`、collider 为空且 collision gate 必须保持 `not_tested`。枕头属于这一类。

collision-enabled 对象的最低要求是：

| Gate | 最低要求 |
|---|---|
| file | visual/collider 存在、非空、hash 与声明一致 |
| semantic | 生成资产类别与扫描证据一致，没有错误门/柜等替换 |
| alignment | 显式 scene transform 与 pivot；bbox/支撑面检查通过 |
| visual | 场景内无遮挡重影、旋转无漂移、对象外观可接受 |
| collision | 简化 GLB 可重载、BVH 命中、机器人 probe 与 stale-face 检查通过 |

collision-enabled 对象只要一个 gate 是 failed/not-tested，就不能成为“已验证可碰撞对象”。visual-only 对象可以在 file、semantic、alignment、visual 通过后发布视觉交互，但不得携带 collider，也不得把 collision 标为 passed。候选资产仍可记录，但不得伪装成 production-ready。

## 坐标与单位

bedroom_4 当前 up axis 为 `-Y`，handedness 为 right，单位状态是 `scene_scale_not_metric`。该状态禁止填写 `metric_scale`。完成真实尺度标定后才可改成 meters/centimeters，并记录标定方法与误差。

Gaussian、render mesh 和 collider 必须共享同一个父 transform。runtime 旋转父组而不是逐个修改子节点，保证视觉与碰撞同步。

## 本地验证

```bash
uv run video2world schema --output schemas/world-manifest.schema.json
uv run video2world validate /absolute/world.json
```

本地 URI 相对 manifest 解析并重新计算 hash。HTTP 资产默认不被视为内容已验证；只有显式 `--allow-remote` 才允许跳过本地内容检查，且 remote count 会保留在报告里。
