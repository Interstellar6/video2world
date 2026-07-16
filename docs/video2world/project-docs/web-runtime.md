---
title: Web Runtime：3DGS 视觉与 Mesh 碰撞
id: video2world-project-web-runtime
category: 项目文档
visibility: public
updated: 2026-07-16
summary: Video2World Web 的分层渲染、对象父组、GLB/BVH 碰撞、机器人、拖拽/双击交互和场景问答实现。
tags:
  - Web
  - Spark
  - Three.js
  - MeshBVH
---

# Web Runtime：3DGS 视觉与 Mesh 碰撞

Web runtime 的核心原则是视觉与物理解耦、坐标与 object identity 统一：Spark 渲染 Gaussian，Three.js mesh 承担 raycast 与碰撞，scene knowledge sidecar 回答问题。

![Interactive object overlay](../assets/pipeline/16-web-object-overlay.png "真实 plant_01 的 Gaussian、选择框与简化 GLB collider 共用 scene-space 父组，并在生产浏览器中完成叠加复核")

## 图层

| 图层 | 表示 | 用途 | 禁止用途 |
|---|---|---|---|
| static visual | carved PGSR Gaussian | 未替换场景的照片级视觉 | raycast/collision |
| static collision | carved TSDF mesh | 地面、墙、床等静态障碍 | 最终可见材质 |
| object visual | TRELLIS Gaussian 或 accepted RGB point cloud | 独立对象外观 | 物理接触 |
| object render mesh | GLB/OBJ | 可切换检查的表面资产 | 未简化时直接做大规模 BVH |
| object collision | simplified GLB | 选择、ground/obstacle/ceiling probe | 冒充精确 watertight 物理体 |
| knowledge | JSON scene graph | aliases、描述、关系、bbox | 凭空生成几何 |

## 对象父组

每个交互对象创建一个 `THREE.Group`。Gaussian splat、render mesh、collision mesh、bbox outline 都是其子节点；父组原点位于 scene-space pivot。拖拽和双击只修改父组 quaternion，因此所有表示同步旋转。

左键按下先做对象选择代理或 collider raycast；命中对象进入 object-yaw drag，未命中才交给 orbit camera。双击对象播放完整 360 度并精确恢复开始 quaternion，避免多次交互累积漂移。

## 碰撞世界

静态 scene mesh 与所有 ready object colliders 都注册为 collision targets。GLB/OBJ 通过 `GLTFLoader/OBJLoader` 读取，每个 mesh 构建 `MeshBVH`；机器人 ground、obstacle、ceiling 与 inspect ray 共享这组 targets。

如果对象缺少或加载失败 collider，runtime 显式标为 `degraded-box`，UI/debug state 显示 degraded count。production manifest 仍应 fail closed，不能因为 box fallback 可运行就把 collision gate 标为 passed。

显式 `collision.mode=none` 的 visual-only 对象不是 degraded collider。它可以通过独立选择框参与 focus、拖拽和 360 度旋转，但不会注册到 BVH collision targets；当前枕头组件即保持这一边界。

## 场景问答

输入框支持中英 location/appearance query。resolver 按稳定优先级匹配 scoped ID、ID、name、alias、category；多个植物或床头柜会返回候选按钮。答案带 `focusEntityId` 时，runtime 将相机 target 平滑移动到 bbox center，并显示 outline。

查询输入拦截 `keydown/keyup` 冒泡，避免用户输入 WASD 或空格时机器人移动/跳跃。未知对象、没有 bbox 或没有 reviewed caption 时返回明确不可用状态。

## 稳定基线

`web/baseline-fixture.js` 锁定提交 `252a85c` 的 identity alignment、presentation yaw/pivot、reference camera/FOV 与 robot spawn。Web 改动必须通过 fixture，不能以加入对象为由改变原有 PGSR/TSDF 对齐。

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

E2E 会检查真实 GLB loader + BVH、父组 drag、双击回位、query/focus、输入隔离与 console。最终 bedroom_4 production QA 记录 823,391 个 static Gaussians、480,000 个 object Gaussians、209,479 个 pillow RGB points，共 1,512,870 个视觉 primitives；另有 1,265,671 个 static collider faces、67,660 个 object collider faces、4/4 GLB ready。四对象 robot blocking、枕头 visual-only drag/360、location/appearance query、`390x844` fresh-load 布局均通过；最终 portable-manifest 回归的 5 个 FPS 样本为 42/43/40/38/38，平均 40.2、最低 38，runtime errors 为零。完整截图与逐项证据见 [bedroom_4 实测记录](../progress/bedroom4-20260716.md)。
