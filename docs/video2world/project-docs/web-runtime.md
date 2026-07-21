---
title: Web Runtime：场景 3DGS 与统一 PBR GLB 对象
id: video2world-project-web-runtime
category: 项目文档
visibility: public
updated: 2026-07-22
summary: Video2World Web 如何保留 PGSR/TSDF 场景层，并让同一个 PBR GLB 同时承担独立对象的视觉、选择/旋转逻辑和 MeshBVH 表面碰撞。
tags:
  - Web
  - Spark
  - Three.js
  - MeshBVH
---

# Web Runtime：场景 3DGS 与统一 PBR GLB 对象

场景级资产仍然分层：Spark 渲染 PGSR Gaussian，处理后的 TSDF mesh 提供静态地面与障碍。独立对象改为 mesh-first：同一个 PBR GLB 既是 Three.js 可见对象，也是 select/drag/spin 的逻辑主体，并由同一三角网格构建 MeshBVH character surface collision。scene knowledge sidecar 继续回答问题。这样不再要求每个对象同时准备 Gaussian visual、render mesh 和 collider proxy 三份资产。

![Interactive object overlay](../assets/pipeline/16-web-object-overlay.png "2026-07-16 旧 production 的 plant_01：Gaussian、选择框与简化 GLB collider 共用父组；该图是历史模式证据，不是 unified GLB 实测")

## 图层

| 图层 | 表示 | 用途 | 禁止用途 |
|---|---|---|---|
| static visual | carved PGSR Gaussian | 未替换场景的照片级视觉 | raycast/collision |
| static collision | carved TSDF mesh | 地面、墙、床等静态障碍 | 最终可见材质 |
| unified object | 一个 PBR GLB | 可见表面、选择、拖拽、360 旋转、MeshBVH character surface collision | 未通过 topology gate 时继续加载或回退 box |
| knowledge | JSON scene graph | aliases、描述、关系、bbox | 凭空生成几何 |

旧 production 还保留 `object visual + render mesh + simplified collider` 三份资产的兼容路径。它是已经通过浏览器 QA 的历史模式，不是新对象的默认合同；新 unified 对象不声明 `visual`、`collision.renderAsset` 或 `colliderProxy`。

## 对象父组

每个交互对象创建一个 `THREE.Group`，父组原点位于 scene-space pivot。unified 模式下 PBR GLB 与 bbox outline 是子节点；拖拽和双击只修改父组 quaternion，同一 GLB 的可见表面与 BVH world transform 因而保持一致。旧模式仍可把 Gaussian、render mesh 与 collision mesh 作为多个子节点，但不再是必需结构。

unified 模式左键直接 raycast 同一个 PBR GLB；命中对象进入 object-yaw drag，未命中才交给 orbit camera。双击对象播放完整 360 度并精确恢复开始 quaternion，避免多次交互累积漂移。旧 visual-only 模式才可能使用 bbox selection helper，且它不参与 character collision。

### 逻辑层级父节点

direct TRELLIS2 refit 有一种中间状态：只采用某些 child object 的 PBR GLB，而它们的 parent 仍停留在原始静态 PGSR 场景里。此时 Web manifest 可以保留 `logicalHierarchyOnly=true` 的 parent，例如 `bed` 只作为三个 pillow 的逻辑祖先存在。

运行时会为逻辑祖先创建一个空父组，但不会加载视觉资产、不会注册 BVH、不会参与 raycast、不会出现在 focus-next 队列，也不会计入 ready visual/collider 数。它只承担层级关系：child 的 transform 挂在 parent 下，因此将来父对象被正式 adopted 后，可以自然带动 child；当前阶段用户直接选择 child 时，旋转只作用于 child 自己，不会反向旋转 parent 或其它 sibling。

逻辑祖先不能作为 fallback。只要它声明了 `placement`、`collision`、`visual`、`renderAsset`、`colliderProxy`、`interaction`、`carve` 或 `sourceAnchor`，manifest 验证会失败。这样 direct scene candidate 可以保留“床包含枕头”的语义，同时诚实地说明 bed 还没有通过统一 PBR/BVH 浏览器验收。

## 碰撞世界

静态 scene mesh 与所有 ready object colliders 都注册为 collision targets。unified object 通过 `GLTFLoader` 只读取一次二进制 GLB，保留 PBR material，同时遍历同一 mesh 构建 `MeshBVH`；机器人 ground、obstacle、ceiling 与 inspect ray 共享这组 targets。

Web manifest 必须显式使用 `collision.mode=unified-glb`。该模式的 GLB 缺失、格式错误、face 数不一致、技术 gate 不通过或 BVH 构建失败时，runtime fail closed 并阻断对象；没有 `degraded-box`、selection box 或其他 proxy fallback。`degraded-box` 只保留给旧 fixture/历史兼容记录，不能成为 unified 模式的兜底。

### 两种 collision topology

| topology | Web 加载前硬条件 | 运行时语义 |
|---|---|---|
| `surface_bvh` | GLB；1-100,000 faces；finite；nondegenerate；winding consistent；`gate.status=passed`；`gate.surfaceCollision=passed`；`characterCollision=true` | 表面 ray/BVH 阻挡；non-watertight 可以通过，但 `volumePhysics=false`，不做 inside/outside 声明 |
| `closed_volume` | `surface_bvh` 全部条件，加上 `watertight=true` | 可声明闭合体拓扑；更高层动力学仍需独立验证 |

当前 TRELLIS2 pillow PBR GLB 是 60,237 vertices / 97,082 faces、PBR material、winding consistent、non-watertight，因此候选 topology 只能是 `surface_bvh`。整体厚度和主色可接受，背面花纹偏差作为 minor limitation 保留，不阻断对象技术 gate。

显式 `collision.mode=none` 的 visual-only 对象不是 degraded collider。2026-07-16 production 枕头仍是这一旧模式：point visual 可 focus/拖拽/360，但不注册 BVH。它保留为稳定历史基线，不代表新 unified 枕头已经完成浏览器验收。

## 场景问答

输入框支持中英 location/appearance query。resolver 按稳定优先级匹配 scoped ID、ID、name、alias、category；多个植物或床头柜会返回候选按钮。答案带 `focusEntityId` 时，runtime 将相机 target 平滑移动到 bbox center，并显示 outline。

查询输入拦截 `keydown/keyup` 冒泡，避免用户输入 WASD 或空格时机器人移动/跳跃。未知对象、没有 bbox 或没有 reviewed caption 时返回明确不可用状态。

## 稳定基线

`web/baseline-fixture.js` 锁定提交 `252a85c` 的 identity alignment、presentation yaw/pivot、reference camera/FOV 与 robot spawn。Web 改动必须通过 fixture，不能以加入 unified 对象为由改变原有 PGSR/TSDF 对齐。`a5af3ae` 只作为对象选择、父组旋转和机器人行为参考。

## QA hooks

开发环境暴露只读/测试 hook：

```text
__askScene(question)
__focusSceneEntity(objectId)
__inspectInteractiveObject(objectId)
__setInteractiveObjectYaw(objectId, degrees)
__inspectCollisionWorld()
__probeCollisionRay(origin, direction, distance)
```

## QA 状态：不要混合历史与新模式

2026-07-16 旧 production E2E/浏览器 QA 已检查真实 GLB loader + BVH、多表示父组 drag、双击回位、query/focus、输入隔离与 console。报告记录 823,391 个 static Gaussians、480,000 个 object Gaussians、209,479 个 pillow RGB points，共 1,512,870 个视觉 primitives；另有 1,265,671 个 static collider faces、67,660 个 object collider faces、4/4 separate GLB colliders ready。四对象 robot blocking、旧枕头 visual-only drag/360、location/appearance query、`390x844` fresh-load 布局均通过；5 个 FPS 样本为 42/43/40/38/38，平均 40.2、最低 38，runtime errors 为零。完整截图与逐项证据见 [bedroom_4 实测记录](../progress/bedroom4-20260716.md)。

当前 unified GLB 代码合同和 fixture 测试不等于真实 bedroom4 PBR 浏览器验收；新的 direct local QA 已补上第一层真实页面证据，但仍不能扩大成 clean-scene production。`examples/bedroom4/completion/direct-trellis2-refit/candidate/browser-qa-report.json` 在 2026-07-22 通过，覆盖：

1. desktop `1440x900` 与 mobile `390x844` 都能通过 `/@fs` 本地 manifest 加载原始 TSDF scene collider 与三个 pillow unified PBR GLB；
2. 三个 pillow 的 PBR material、MeshBVH、`collision.mode=unified-glb`、`surface_bvh`、face count 与无 proxy/degraded fallback 均通过；
3. pointer focus 命中真实对象，yaw hook 会同步移动 group、visual mesh 与 collision mesh；
4. robot collision harness 对三个 pillow 分别得到 `blocked=true`、`lastCollisionKind=object` 和对应 object ID；
5. 逻辑 `sam3_bed_01` 祖先只保留 hierarchy，拒绝 focus/screen point，且不加载 visual/collider。

仍未完成的是 production promotion 级验收：direct candidate 没有 clean plate、没有静态 carve、没有最终支撑/显著穿模质量放行，也没有证明完整背景恢复。因此本文可以声明 direct local unified runtime QA passed，但不能声明 unified production passed。
