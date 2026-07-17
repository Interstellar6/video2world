---
title: 通用遮挡、背面与背景分层补全
id: video2world-project-layered-completion
category: 项目文档
visibility: public
updated: 2026-07-18
summary: Video2World 如何按遮挡顺序生成 mesh-first PBR GLB、补全不可见背面和物体移除后的背景，并用六视图、支撑、穿模、clean-plate 与 canonical Web 门禁控制 current-demo-only 发布。
tags:
  - Completion
  - Occlusion
  - Clean Plate
  - Geometry QA
---

# 通用遮挡、背面与背景分层补全

Video2World 的补全对象不是某一种家具，也不是对一张图做一次 inpainting。系统把问题拆成三条必须分别验收的链：独立物体的不可见表面、物体放回场景后的几何关系、移除物体后暴露的背景。只有三条链都通过，才允许用新对象替换静态场景中的原对象。

```text
对象证据 -> mesh-first PBR GLB（对象 Gaussian/point cloud 可选）
         -> scene fit + support + interpenetration
         -> clean plate RGB -> 新 depth/normal -> clean scene Gaussian / mesh
         -> 下一遮挡层重新盘点
```

对象六视图通过，不代表场景通过；scene-space OBB 拟合通过，也不代表背景已经修好；2D inpainting 能输出完整帧，也不代表被遮挡的床面、墙面或地面被真实恢复。这些边界由 `video2world/completion.py`、`video2world/completion_routing.py` 和各阶段 receipt 固化。

## 三种证据状态

每个像素、点、表面和资产都必须能回答“它从哪里来”。

| 状态 | 定义 | 典型内容 | 可以声称什么 |
|---|---|---|---|
| `observed` | 原视频中直接可见，并绑定 frame、mask、相机与 hash | RGB 像素、SAM3 可见 mask、可见轮廓、直接深度证据 | 该视角确实观察到颜色、材质或几何边界 |
| `reconstructed` | 由已标定的 observed 证据确定性求解 | 多视角点云、AABB/OBB、TSDF、重投影 donor、scene fit | 结果由哪些观测和算法推导得到 |
| `generated` | 没有直接观测，由类别、检索或生成先验提出 | 物体背面、长期遮挡区域的纹理、clean-plate residual | 这是受证据约束的候选，不是物理真值 |

相邻视角 donor 像素仍保留原始 frame provenance，但其投影位置和可见性判断属于 reconstructed。类别先验可以提供闭合拓扑，却不能把先验颜色冒充 observed albedo。生成背景必须单独标记，并在生成之后重新估计 depth/normal；不能复用含前景物体的旧深度。

## 为什么必须从前向后逐层处理

场景 inventory 先找主要对象、附属对象、fixture 和结构，再由 SAM3、标定相机与 depth 建立遮挡有向图。边 `A -> B` 表示 A 位于 B 前面，必须先处理 A，才能判断 B 是否完整。置信边存在环时计划直接失败，不能猜一个处理顺序。

Inventory 同时决定 semantic granularity。可独立移动但依附于大对象的实例使用 `independent_child_asset`：父对象移动时 child 跟随，child 自己旋转时不反向带动 parent。床板、床单、床腿等不可独立操作的部分使用 `merge_into_parent`；墙、地面和天花板使用 `structure_background`。这个层级决定 completion target、scene command 的 split/merge/reparent 行为和 Web transform，不只是显示标签。

Qwen scene audit 只负责类别、可见数量、可见颜色/形状/材质和可能的父类。它明确禁止输出 bbox、mask、稳定 instance ID、深度顺序、遮挡边和隐藏几何；这些仍由 SAM3 与标定几何决定。当前 bedroom_4 的 official 7B per-view run 已做到 8/8 严格解析并合并出 7 类 observation；这些结果仍只是 inventory 观察，不会自动升级成 geometry-verified 遮挡边。VLM 输出能“看起来合理”不等于几何合同通过。

默认 `max_parallel_targets_per_round=1`。这是为了避免同一层相邻对象被同时移除后失去 donor、支撑和遮挡归因。一个 round 只消费紧邻的上一轮 clean plate；前一轮不是 `passed`，更深层不能开始。

## 每一轮的输入、输出与依赖

第 0 轮输入是原始 RGB/cameras/depth、scene Gaussian/mesh、SAM3 evidence、对象点云和 scene inventory。之后第 N 轮始终消费第 N-1 轮通过的 clean plate 与重新盘点结果。

| 顺序 | 动作 | 直接输入 | 主要输出 | 阻断门禁 | 下一步 |
|---:|---|---|---|---|---|
| 1 | `reinspect` / 选择最前层 | 上轮 clean plate、scene audit、SAM3、depth | inventory、occlusion graph、target ID | instance identity、图无环 | instance split |
| 2 | `split_instances` / `segment` | 原帧、类别、prompt、上一轮 inventory | 每实例跨帧 masks、证据 frame | mask provenance、同物理实例 | 3D lift |
| 3 | `lift_to_3d` | masks、cameras、depth | object cloud、AABB/OBB、可见面比例 | 多视角投影、depth consistency | appearance contract |
| 4 | appearance contract | 至少两个原视频视角、source RGBA | 颜色/材质/形状/部件布局合同 | 跨帧身份和外观一致 | backend routing |
| 5 | `complete_object` | 合同、可见几何、类别、检索或生成条件 | PBR GLB/mesh、可选 object Gaussian、completion report | 背面非空、六视图、外观、mesh 技术 gate | geometry review |
| 6 | geometry review | source、front render、material、六视图、mesh report | accept/retry/reject、详细 retry prompt | 技术 gate + VLM review | scene fit 或重试 |
| 7 | `place_object` | accepted canonical asset、scene OBB、支撑面 | scene-fit unified GLB 或旧模式多表示、transform、pivot | alignment、support、interpenetration | clean plate |
| 8 | `clean_plate` | 原/上轮 RGB、removal masks、donor views、scene geometry | clean RGB、residual masks、clean-plate manifest | outside unchanged、跨视图、背景语义 | depth/normal rebuild |
| 9 | 重建局部场景 | accepted clean RGB、cameras | 新 depth/normal、clean scene Gaussian/mesh | 背景 depth/normal、static carve | 再盘点 |
| 10 | `validate_round` | 本轮所有 receipt | round quality report、下一轮输入 | 所有 gate 通过 | 下一遮挡层 |

所有前景对象完成后才进入 `final_background`。该轮只重建墙、地面、天花板与剩余结构，不再产出独立对象。计划的停止条件是：没有未建模的主要/次要对象；每个既有 round 的 mask 外区域和跨视图门禁通过；只剩结构背景。

## 后端不是按类别硬编码

`completion_routing.py` 先验证同物理实例和外观合同，再按直接证据强度选择后端。下表分数是当前确定性路由优先级，不是生成质量分数。

| 后端 | 当前优先级 | 进入条件 | 适用边界 |
|---|---:|---|---|
| `support_surface_reconstruction` | 98 | 结构背景且有经过验证的 plane/depth/normal | 先重建床面、墙面、地面等支撑，只生成残余洞 |
| `multi_view_reconstruction` | 96 | 同实例与外观通过；至少 3 个标定视角；baseline `>=0.05`；可见表面 `>=0.35` | 直接几何证据优先 |
| `deformable_category_prior` | 88 | source front RGBA 可用；类别有柔性闭合体 profile | 枕垫、软包等；先验只提供拓扑 |
| `retrieval_cad_prior` | 82 | 刚体类别且 CAD confidence `>=0.82` | 桌、椅、柜、灯等规则物体 |
| `generative_image_to_3d` | 70 | 同实例、外观与 source RGBA 均通过 | 没有更强几何或检索证据时的候选 |
| `hold_for_more_evidence` | fail closed | 身份或外观未验证，或所有候选被审核拒绝 | 不生成、不发布 |

即使类别是已注册的柔性物体，如果跨帧颜色和物理实例不一致，也必须选择 hold。bedroom_4 的第二批三对象 RGBA 就因为“单帧外观合同与用户确认的浅色/白色场景外观存在冲突”在 generation 前被拒绝，没有送入 TRELLIS。

只要 `persistent_occlusion_fraction > 0`，背景残余策略就是 `multi_view_donor_then_constrained_generation`；比例达到 `0.80` 以上时 receipt 还必须显式提醒“多数隐藏像素没有 observed donor”。这不会自动批准生成，只是规定 donor 与 generated residual 的顺序。

## 独立对象默认交付：一个 PBR GLB

独立对象采用 mesh-first 合同。只要 PBR GLB 的整体形状、主色类别、可见面外观和场景放置可接受，它本身就同时承担三种责任：Three.js PBR 可见表面、select/drag/spin 的逻辑主体，以及机器人 MeshBVH 接触表面。对象 Gaussian、对象 point cloud、单独 `renderAsset` 和单独 collider proxy 都不是必需产物；对象点云仍可作为扫描 evidence，场景整体仍保留 PGSR 3DGS 与 TSDF，但不要求把每个生成对象再复制成 Gaussian 视觉代理。

Web 投影用 `collision.mode=unified-glb` 显式声明该模式。同一 GLB 只加载一次，manifest 不得同时声明 `visual`、`collision.renderAsset` 或 `colliderProxy`，加载或技术 gate 失败时直接阻断该对象，不回退到 box。碰撞拓扑分为：

| topology | 必须满足 | 允许声称 |
|---|---|---|
| `surface_bvh` | mesh 非空；vertices/indices finite；无退化三角面；winding consistent；1-100,000 faces；collision gate 与 `surfaceCollision` 均 passed | 机器人和射线的表面阻挡；不声称 watertight、体积或 inside/outside |
| `closed_volume` | `surface_bvh` 全部条件，加上 `watertight=true` | 表面阻挡与闭合体语义；更高层物理仍要单独验证 |

旧 production 中 Gaussian visual、render mesh 和 simplified collider 分开的模式继续作为历史兼容合同保留，但不是新对象的默认输出，也不能据此要求新的 object Gaussian/point cloud。

## 官方 TRELLIS.2 provider 合同

通用 image-to-3D 路径由 `python -m video2world.providers.trellis2_asset` 执行单对象生成。它默认只接受带真实 alpha 的 RGBA，避免对已经由 SAM3/appearance contract 确认的轮廓再次做隐式 RMBG；RGB 或不透明输入只有显式 `--allow-rmbg` 才能进入模型。默认 production 输出是一个必需的 `asset_pbr.glb`，processed input 与 hash receipt 用于审计；raw mesh、点云和 convex 仅为 debug opt-in。

这不是根据文档猜出的 wrapper。2026-07-17 mil8 实际跑通并复核的官方代码是 `https://github.com/microsoft/TRELLIS.2.git` commit `75fbf0183001ed9876c8dbb35de6b68552ee08bd`，模型是 `microsoft/TRELLIS.2-4B` revision `af44b45f2e35a493886929c6d786e563ec68364d`，现场 `pipeline_512.json` SHA-256 为 `308dd782f3eaf1e30e7403ee3837f21beefa664202b0a045ca61308a42c661b3`。provider 直接调用官方 `Trellis2ImageTo3DPipeline.run` 和 `o_voxel.postprocess.to_glb`，decimation target 被限制在 `1..100000` faces。

最终审计不相信导出函数的返回值，而是重新解析 `asset_pbr.glb`：所有 node transform 和 vertices 必须 finite，faces 必须是有效整数三角索引且没有退化面，winding 必须 consistent，每个 mesh 都要有 PBR material，整体 extents 为正且总面数不超过本次 target。每个材质 receipt 记录 normalized base-color factor、`metallicFactor`、`roughnessFactor`、baseColorTexture 是否存在及其 image mode/尺寸/decoded-pixel hash/数值范围、alpha mode/cutoff/texture channel 和 double-sided。factor、cutoff 与解码纹理数值必须 finite 且位于 `[0,1]`，alpha mode 与 double-sided 类型也必须合法。watertight 才能标记 `closed_volume`；否则技术通过的对象只能标记 `surface_bvh`，可以承担表面接触但不能声称体积或 inside/outside。任一技术 gate 失败都会落失败 receipt 并终止；通过也只得到 `technical_passed_visual_pending`。

材质参数的技术合法性与物体语义分开。`metallicFactor=1` 对金属对象可能正确，因此通用 provider 不把它设为失败条件；scene-fit material profile/VLM 再依据对象类别、原视频主色与材质证据决定是否修正。明显 shape mismatch、主色类别错误和显著 interpenetration 仍是 hard gates；轻微不可见面纹理或材质 hallucination 保持 warning，可记录 limitation 后放行。

receipt 固定记录 source commit、model revision、config hash、seed、输入 RGBA hash、最终 GLB hash、debug flags、运行时和完整 mesh audit。随后必须运行 `scripts/capture_object_review.mjs`，从 canonical `object-review` 页面取得 `front/right/back/left/top/bottom` 六个正交 object-local render、3x2 contact sheet 及逐文件 hash receipt。旧 wrapper 的水平 orbit 只能作动画预览，不能替代 top/bottom 与正交侧面证据。

## 六视图与语义 albedo

每个独立对象都必须检查 `front/right/back/left/top/bottom`。只看 front 会漏掉片状点云、空背面、错误厚度、侧面贴图拉伸和断开的部件。

![Object mesh six-view review](../assets/completion/object-mesh-six-view.png "GLB 的六个正交视图；该审核只覆盖 object-local 闭合体、厚度和 neutral-albedo，不覆盖场景放置")

审核分两层：

1. 确定性技术 gate 检查 finite geometry、非空 mesh、正 extents、front aspect、厚度比例、source/front/material 色差和六视图非空；`surface_bvh` 不把 watertight 当作必要条件。
2. VLM 同时读取原始 source、透明背景 front render、material reference、六视图和技术报告，检查 identity drift、缺背面、开洞、颜色/材质/形状变化、部件丢失或合并、漂浮/断开和跨视图风格不一致，并按影响程度分级。

审核是 severity-aware，而不是“出现任何生成痕迹就失败”。轻微背面纹理、不可见面花纹或材质细节幻觉，如果不改变整体形状、主色类别、可见面身份且不造成明显穿模，可以写成 `passed_with_recorded_limitations`；当前 `GateRecord` 的落盘方式是 `status=passed` 并在 `reason`/limitation 或 report 中保存具体偏差。明显形变、主色类别错误、缺面/片状、部件断裂、悬空或显著穿模仍是 blocking issue，必须 retry/reject。所有限制保留 provenance，不能因为人工放行而删除失败观察。

“白色物体”必须在原视频的多个曝光条件下建立浅色语义合同，再在 neutral-albedo mesh 和 Gaussian direct-color 中检查。不能因为场景灯光偏黄就生成棕色背面，也不能仅用平均 RGB 把花纹物体强制涂白。当前 front 对象实验使用的样例阈值是 luminance `>=160`、chroma `<=35`、mean-RGB error `<=45`；这些是本次 receipt 的检查参数，不是适用于所有视频的固定全局阈值。

![Object Gaussian six-view review](../assets/completion/object-gaussian-six-view.png "Spark direct-color 六视图；80,000 个 Gaussian 的 front/back/side/top/bottom 均非空，并独立检查颜色和厚度")

旧 parametric 实验中，浅色 front 对象的 Gaussian mean RGB 为约 `210.1 / 211.7 / 212.5`，六视图均非空，thickness/max-extent 为 `0.28085`；对应 mesh 为 10,226 vertices / 20,448 faces、单连通、watertight。它保留为多表示路线的历史证据，不再构成“必须生成 object Gaussian”的要求。

当前 mesh-first 候选来自真实 TRELLIS2 front pillow seed 42。`asset_pbr.glb` 有 60,237 vertices / 97,082 faces、一个 PBR material，finite、无退化面、winding consistent，但不是 watertight；六视图显示完整厚度，不再是只有正面的片状对象。它因此只能使用 `surface_bvh`，不能声称 closed volume。不可见背面的花纹与扫描正面不完全一致，按用户最新验收口径属于可记录的 minor hallucination：整体枕头形状和浅色主色类别保持可接受，可用 `status=passed + limitation` 放行 object-local gate。

![TRELLIS2 PBR pillow six-view](../assets/completion/object-pbr-six-view.png "真实 TRELLIS2 PBR GLB 的 front/right/back/left/top/bottom 正交视图：完整厚度通过；物体局部 back 是原视频可见的浅色侧，另一侧花纹属于已记录的轻微生成偏差")

早期 image-to-3D 候选因 `front_color_fidelity=false` 和 `missing_back_surface` 被判定 `retry`，没有因为文件能打开而放行。这正是六视图合同要消除的错误。

## VLM 审核与最多三次尝试

当前 production 策略将每个对象的 `max_attempts` 固定为 3：

1. 第一次生成后先跑技术 gate，再跑 VLM geometry review。
2. 只有明显形变、主色类别错误、缺面/片状、部件断裂、悬空、显著穿模或失败的硬技术 gate 才触发 `retry/reject`；轻微背面纹理或材质幻觉写入 limitation 后可以通过。
3. `retry` 必须至少包含一个 blocking issue 或失败的技术 gate，并生成面向下一次的具体 prompt。例如指出缺失哪个面、哪一视图主色错误、需要怎样恢复厚度或连接部件。
4. 下一次必须使用新 seed/prompt，并记录 prompt、输出资产和 review SHA-256。尝试编号严格连续。
5. 第三次仍是 `retry` 时状态变为 `exhausted`，对象保持 candidate/held；不能继续无限抽 seed，也不能进入 placement。

Schema 为研究运行保留更高上限，但 Video2World 当前发布口径不允许通过提高次数绕过三次审核预算。`accept` 后不允许再追加尝试；`reject` 是终止状态。

## Scene fit、支撑与穿模

object-local 资产通过后，`place_object` 才把 canonical bounds 拟合到扫描 OBB。unified 模式把变换直接烘焙到同一个 PBR GLB，并以 scene-space pivot 驱动选择、旋转和 BVH；旧多表示模式才需要把同一个变换分别应用到 Gaussian centers/covariance、render mesh 和 collider。

放置至少检查：

- OBB projected extents 与扫描目标一致，transform finite 且 determinant 为正；
- 对象与 bed/table/floor 等支撑面有合理接触，不能悬空，也不能把柔软对象压进支撑体；
- 对象之间、对象与静态 scene mesh 之间没有不可接受的三角面相交；
- unified GLB 的 surface/volume topology 与 collision gate 明确通过；旧模式 collider 仍需独立审核；
- 场景 overlay 中没有旧静态对象和新对象重影。

单视角或可见表面点云不能直接把 PCA OBB center 当作完整体积中心。`observed_front_surface` placement 必须显式提供原相机方向、厚度轴、类别厚度上限和 source-mask-derived in-plane scale；完整体积只能向遮挡侧扩展，原可见前表面保持锚定。真正有多视角体积证据时才使用 `volume_center`。

完成后还要把 scene-fit mesh 投回原视频相机，与原 SAM3 mask 比较 silhouette、bbox 和中心。PGSR camera rotation 的第二轴在当前归档中是 image-down；该约定必须写入 receipt，不能按常见 camera-up 猜测。

![Source camera silhouette refinement](../assets/completion/source-camera-pbr-silhouette.png "真实 TRELLIS2 PBR GLB 投回原始 000064：黄色为重合，绿色为原 mask 多出部分，红色为候选多出部分")

旧 parametric front 对象经历了三次保留证据的 placement 尝试，Attempt 3 得到 mask IoU `0.8026`、bbox IoU `0.9104`、中心误差 `4.75 px`。当前 TRELLIS2 PBR GLB 复用同一前表面锚定与 source-mask 尺度约束，真实回投结果为 mask IoU `0.709810`、bbox IoU `0.924577`、中心误差 `4.402 px`，三项 source-camera silhouette gate 均通过。

source-camera 回投本身仍不等于 unified browser scene pass。旧 parametric Attempt 3 的 browser receipt 中 `scene <-> sam3_pillow_front` 有一组相交，五个 support offset 全为负，旧静态层从对象左半部和内部明显透出；这段失败证据继续保留。后续严格 clean scene 与四个 TRELLIS2 PBR GLB 已通过 canonical desktop/mobile QA 并完成 `current_demo_only` promotion，详见下文；该结果不会反向把旧 Attempt 3 改写成通过，也不会把 R4 背景升级成高质量补全。

## Clean plate：先找真实 donor，再生成残余洞

clean plate 不是把物体 mask 直接交给视频 inpainting。通用顺序是：

1. 在 donor 视角排除目标和其他前景 mask；
2. 用 donor RGB-D 与标定相机投到 target；
3. 用 z-buffer、depth cluster、遮挡和最小 donor support 拒绝错误颜色；
4. 只把通过的 donor 写入 removal mask，mask 外保持 byte-identical；
5. 对剩余无观测区域使用 geometry-conditioned generation，并标为 generated；
6. 跨标定视角检查纹理、边界和结构连续性；
7. RGB 通过后重新估计 depth/normal，再重建 PGSR/TSDF。

![Calibrated donor prefill](../assets/completion/clean-plate-donor-prefill.png "标定 donor reprojection 的 source/mask/prefill/support/residual 对照；mask 外严格不变，但有效 donor 太少")

bedroom_4 的保守 donor coverage 只有 `0.62% / 0.17% / 0.59%`，也就是三张代表帧分别只覆盖 removal mask 的 113/18,346、40/23,065、127/21,440 像素。残余洞仍为 `99.38% / 99.83% / 99.41%`，因此不能声称“用多视角恢复了床面”。

把 foreground exclusion margin 降到 0 后 coverage 看似升到约 9.86%-11.89%，但 support 只形成原物体边缘的一圈浅色泄漏：

![Foreground edge leakage control](../assets/completion/clean-plate-edge-leakage.png "零 margin 失败对照：增加的 support 是原前景边缘，不是被遮挡背景的观测")

这类数值提升不能冒充 observed donor。Round 1 的 full-mask ProPainter 真实运行虽然 25/25 帧、尺寸、mask 外 RGB 和输出非空等技术 gate 通过，视觉上仍是灰白低频光斑。Round 2 证明相机路径几乎没有观察到隐藏床面。

Round 3 使用固定 revision 的 SDXL inpainting 生成三个单帧 anchor。用户已人工选择 seed `2026071701`，以 `pass_with_known_limitation` 放行当前 bedroom4 demo 和独立对象集成 QA。该 1280x720 候选处理 20,791 个 mask 像素，mask 外 changed pixels 为 0；但它仍在移除区生成了枕头状外观，所以这次人工放行只说明“当前 demo 视觉可以先用”，不证明 object-free clean plate、跨视图背景一致性或被遮挡床面几何已经恢复。通用 pipeline 的 multiview/depth/normal gate 仍保留，不能把 scoped demo exception 写成几何真值。

![User-approved SDXL demo anchor](../assets/completion/clean-plate-sdxl-seed-2026071701.png "seed 2026071701 已由用户按当前 demo 范围人工放行；图中生成残留仍不构成 object-free 或多视图几何证据")

## 真实四轮逐层 Clean Plate：不是一次性全抠除

2026-07-17 的 bedroom_4 已按遮挡关系真正执行四轮，而不是把三个枕头和床合成一个全局 mask 后一次性修背景。顺序固定为 `sam3_pillow_front -> sam3_pillow_left -> sam3_pillow_right -> sam3_bed_01 -> structural background`。**严格输入不变量是：R1 输入原始 RGB，R2 输入 R1 clean plate/composite，R3 输入 R2 clean plate/composite，R4 输入 R3 clean plate/composite。** 这里有两条不能混淆的数据流：R2-R4 每一帧的场景 RGB 输入必须是紧邻上一轮 composite，路径和 SHA-256 都要相等；每轮的 measured RGB-D donor 则始终只能来自原始观测帧和原始 DA3 depth，并用截至本轮的累计 removed masks 排除前景。上一轮生成/PBR/ProPainter 像素和 unresolved residual 永远不能反过来冒充 observed donor。

![Layered clean-plate source views](../assets/completion/layered-source-25view.png "000048-000072 的 25 个原始观测视角；三个枕头、床和房间背景同时存在")

四轮输入和产出是串行闭合的；任何一行失败，下一行都没有合法输入：

| Round | 唯一合法的场景 RGB 输入 | 累计 mask 与 measured RGB-D donor | 本轮独立物体产出 | 本轮场景产出 / 下一步 |
|---|---|---|---|---|
| R1 front pillow | 原始 25 帧 RGB | mask=`front`；只投影原始 RGB-D，排除 front | `sam3_pillow_front` scene-fit PBR GLB | R1 clean plate/composite，唯一允许输入 R2 |
| R2 left pillow | R1 clean plate/composite | mask=`front+left`；仍只投影原始 RGB-D，同时排除 front+left | `sam3_pillow_left` scene-fit PBR GLB | R2 clean plate/composite，唯一允许输入 R3 |
| R3 right pillow | R2 clean plate/composite | mask=`front+left+right`；仍只投影原始 RGB-D，同时排除三个 pillow | `sam3_pillow_right` scene-fit PBR GLB | R3 clean plate/composite，唯一允许输入 R4 |
| R4 bed | R3 clean plate/composite | mask=`front+left+right+bed`；本轮 measured donor 为 0 | `sam3_bed_01` support-adjusted PBR GLB | structural background 覆盖累计 mask，得到最终 R4 clean plate |

这张表也说明了为什么“逐层”不只是顺序跑四次 inpainting：每个 object round 一边交付可独立选择和旋转的 PBR GLB，一边只从场景底图中移除当前最前层；最后才由 structural background 处理床移除后仍不可见的墙面/地面。当前采用的四个对象资产分别位于 `examples/bedroom4/completion/trellis2_pillow_front_seed42/scene_fit_silhouette_refined/`、`round02_left_pillow/trellis2_seed44/scene_fit_silhouette_refined_v6/`、`round03_right_pillow/trellis2_seed43/scene_fit_silhouette_refined_v6/` 与 `round04_bed/component_assembly_v2_support_adjusted/`。它们是各层的独立对象结果，不是 clean plate 像素来源。

R1 只移除最前面的枕头。左枕头、右枕头和床仍以 source-camera PBR render 参与遮挡合成，因此本轮不会把它们错误地当作背景补掉。

![Round 1 front pillow peel](../assets/completion/layered-round01-front-pillow.png "R1：只去除 front pillow；left pillow、right pillow 和 bed 仍属于 remaining layers")

R2 只在 R1 composite 上累计移除左枕头，R1 已移除的 front pillow 不允许重新出现；R3 也只能在 R2 composite 上累计移除右枕头，只留下床。中间图中的 PBR 外观是遮挡层和深度分区证据，不是 measured donor，也不是最终 Web 截图；对象发布质量仍由各自六视图、scene fit、支撑和穿模报告决定。

![Round 2 left pillow peel](../assets/completion/layered-round02-left-pillow.png "R2：front+left pillow 已移除；right pillow 和 bed 保留")

![Round 3 right pillow peel](../assets/completion/layered-round03-right-pillow.png "R3：三个 pillow 均已移除；bed 是唯一 remaining object layer")

R4 以 R3 composite 为 source，累计 mask 覆盖三个枕头和床；本轮没有 measured donor，也没有 remaining object，8,259,257 个 removal pixels 全部由同一套结构背景 RGBA/depth 分区覆盖。背景仍有明显床形低频色块和简化平面，因此 sequence-stage report 记录 `promotion_approved=false`，禁止 R4 RGB 不经重建/QA 就直接发布，也不能把它作为通用背景补全质量证据。后文的 canonical `promoted_current_demo_only` 是在 fresh DA3/PGSR/TSDF、alignment 和 desktop/mobile QA 全部完成后的独立 finalization，不会修改这条 R4 质量判断。

![Round 4 final background](../assets/completion/layered-round04-final-background.png "R4：移除 bed 后只保留结构背景；可见床形低频色块作为 current-demo-only limitation 保留")

真实执行的像素分区如下。每行的 removal pixels 都是**截至本轮的累计 mask**，不是只统计当轮新增对象。`measured` 只允许来自原始观测 RGB-D；R1-R3 report 都显式声明 `generated_pixels=0`、`propainter_pixels=0`，并绑定 `previous_cumulative_manifest_sha256` / `previous_prefill_report_sha256`。PBR 与 structural 只填 measured residual，不能被重标成 measured donor。

| Round | 累计 removed | Remaining | Removal pixels | Measured | PBR render | Structural | Unresolved | Frame-set SHA-256 |
|---|---|---|---:|---:|---:|---:|---:|---|
| R1 | front pillow | left pillow, right pillow, bed | 532,888 | 69 | 532,819 | 0 | 0 | `42435b0a...ddd2` |
| R2 | front + left pillow | right pillow, bed | 1,233,510 | 9,669 | 1,222,217 | 1,624 | 0 | `feac1e67...7d1` |
| R3 | front + left + right pillow | bed | 1,636,824 | 12,443 | 1,622,757 | 1,624 | 0 | `e2230aa4...0f19` |
| R4 | three pillows + bed | none | 8,259,257 | 0 | 0 | 8,259,257 | 0 | `854bd224...dfca` |

Sequence report SHA-256 为 `672ccc38d2d0187a4e99e85a1f54a451039deab5c6d73eb81f71108275483e88`，receipt 为 `cdf2ffc02dbacf184391dafd1cd35c427fd5cf868b084548d4c7e2d7496d4d25`。机器证据同时绑定四轮 manifest/report/receipt、R2-R4 predecessor、25 个 portable R4 masks 和五张 contact sheet；本地 focused evidence tests 会重新计算这些 hash。

此前 `clean-scene-reconstruction-input-v23` 只绑定单独的 R4 背景候选，不具备 R1-R4 predecessor chain。基于它启动的 PGSR 已在 4,210/30,000 主动停止，TSDF 未运行；作废 receipt SHA-256 为 `aaf5837dc931b7d34f0097366297537358821d28a6a274fe8eb5789d4f42ce6d`。新的 reconstruction package 只硬链接严格 R4 最终 RGB 和相机，明确不包含旧 geometry/composite/DA3 depth；其 manifest SHA-256 为 `3645fc895d202d537a9104d91a21c4ff86e87352c60e3c0f118bdea06adc0299`，package receipt SHA-256 为 `33532775d7fe8c3f76724c73b13b68ab01f489d62b5742823a185d52400a468d`。

fresh DA3 的 25 张 `720x1280 float32` depth 共 23,040,000 个值，全部 finite 且为正，范围为 7.7331486-22.4680824。重新生成的 `pointcloud_da3.ply` 有 4,000,000 个 finite/nonzero XYZ，bbox extent 为 `[26.352661, 17.238593, 16.871765]`，PLY SHA-256 为 `196d170d95404976b0a56ba41654cc23c71cc248955f8fc109a14f10b8ac58d3`。DA3 stage receipt SHA-256 为 `287fc496a1bdfc8a44cbd36ab913da8a9263413af8631eccb7c89019352af230`；它只允许进入 current-demo PGSR，不允许直接进入 TSDF、Web 或 general promotion。

fresh PGSR 随后在同一严格 package 上完成 iteration 30,000，状态为 `technical_passed_pgsr_30000_current_demo_only`：L1 `0.0091873651`、PSNR `33.2610672 dB`、454,617 Gaussians、112,746,547 bytes，训练耗时 3,088.6 秒。`point_cloud.ply` SHA-256 为 `4eeb1403194e0248f0ab4cbf206572af5bb1ece635a03775cda6e0358133c9bf`，stage receipt SHA-256 为 `0224cc0868e0fb03822b3f3053f414d909056bc70857e44291b9389b776f349b`。

TSDF 也已在该 PGSR 输出上完成。最终 `tsdf_fusion_post.ply` 为 94,989,275 bytes、1,805,667 vertices / 3,556,615 faces，所有 vertices finite、faces 均为有效三角形，bbox extent 为 `[26.1068731, 18.3792248, 17.0562393]`；文件 SHA-256 为 `ccd48c2376d2cc8d81b979813a09c58feb158b66a441d66512466125b0c8f344`，stage receipt SHA-256 为 `c45d92dcc6810034772d1f8dbb058c34424765cd3d41ad03bdfffe100acad5c2`。alignment receipt `16839c7f0897c0095bf01ada6768d60dd23f636e102e17111fbc505e8fe27d49` 进一步验证 PGSR/TSDF 共坐标、相机子集精确、对象 placement 共用目标坐标系；结论仍限定为 `passed_current_demo_only`，不消除 R4 背景的视觉局限。

## Canonical Web promotion：只在 current demo 范围通过

严格 clean scene 与四个 unified PBR objects 已写入 canonical `web/public/worlds/bedroom4/manifest.json`，最终 manifest SHA-256 为 `58cc2b06ec1a7feb70fc0e0060078e0e0e1db409f3e96843c53948c74d3ab7cb`。final browser QA `qa/clean-unified-scene-browser-qa.final-promoted.json` 的 SHA-256 为 `0dc5f2b312e6bef0e272920169993d2485c9e967dec596db047bedd95aa7b961`，状态 `passed_current_demo_only`、`automatedGate=passed`、`failures=[]`，测试前后 manifest bytes 未变。

QA report 本身保持 `publishingPerformed=false`，不直接执行发布。随后 finalization receipt 以 same-directory atomic rename 把这份**精确 QA 过的 manifest bytes**移到 canonical 路径，状态为 `promoted_current_demo_only`、`promotionAllowed=true`；receipt SHA-256 为 `a7a0691a0effd8898ee2e1a89b3c2a103b966b13005eed51cb52acdb8260c041`。稳定基线别名 `manifest.web-demo-baseline-stable.json` 被原样保留，SHA-256 仍为 `3805f0e5bab09add424b3b78f9349cd2eca6d1262777ef683e13cda07695e82b`。

桌面 `1440x900` fresh load 的 FPS samples 为 `60/58/51/49/56`，平均 `54.8`、最低 `49`；initial runtime FPS 为 `57`。以 bed 为 overview focus 时，投影 coverage 为 width `0.3338`、height `0.4027`，camera-inside object 列表为空。8/8 visual objects 与 8/8 object colliders ready，degraded collider 为 0；console、page error、request failure 与 HTTP error 都为空。

![Canonical strict clean scene desktop QA](../assets/completion/strict-clean-scene-web-desktop.png "最终 canonical desktop QA：严格 clean scene、bed 与三个 pillow 的 unified PBR GLB、bbox 和问答同屏；这是 current-demo-only 浏览器证据，不代表背景已达到高质量补全")

移动端 `390x844` fresh load 的五个 FPS samples 均为 `60`，平均/最低均为 `60`；overview coverage 为 width `0.6840`、height `0.2099`，camera-inside object 列表同样为空。画布与 viewport 都是 `390x844`，8/8 visual objects、8/8 colliders、零 degraded，diagnostics 也没有 console/page/request/HTTP 错误。

![Canonical strict clean scene mobile QA](../assets/completion/strict-clean-scene-web-mobile.png "最终 canonical mobile QA：390x844 HUD 与 canvas 通过布局和非空检查，8/8 visual objects 与 colliders ready；背景低频拉伸仍按已知限制保留")

这里的“promoted”只表示 exact manifest、资产加载、交互/碰撞、相机 framing、问答与桌面/移动运行门禁在当前 Bedroom4 demo 范围内通过。R4 的墙面/地面仍可见低频拉伸、床形色块与简化平面，截图也没有证明这些区域被真实观测恢复；因此不能写成“高质量背景补全完成”，也不能把该结论推广到新视频。

## 当前 bedroom_4 状态

下表只用于说明门禁如何工作，不把某个场景写成算法特例。

| 子问题 | 当前证据 | 状态 |
|---|---|---|
| 三个相触前景的实例拆分 | 单帧 prompt masks + 3D cloud conservation；三个对象 cloud 已单独导出 | 技术通过，跨帧 physical-instance 身份仍需逐对象复核 |
| 初始 image-to-3D 候选 | front color 不匹配，VLM 指出 missing back | `retry`，未发布 |
| 旧 parametric 浅色候选 | watertight mesh、80k Gaussian、mesh/Gaussian 六视图、浅色 direct-color | 历史 object-only passed；非新默认格式 |
| TRELLIS2 PBR GLB | 60,237 vertices / 97,082 faces；完整厚度；PBR；winding consistent；non-watertight | object-local `surface_bvh` passed with recorded backside-texture limitation |
| scene-space fit + 原相机回投 | PBR mask IoU 0.709810、bbox IoU 0.924577、中心误差 4.402 px；canonical desktop/mobile final QA failures 为空 | source-camera 与 canonical browser passed current-demo-only |
| 支撑与穿模 | support-adjusted bed receipt 的 9 项 gate 全通过；三个枕头预测 penetration/gap 都在 0.2 scene-unit policy 内；Web 8/8 colliders ready、degraded=0 | 技术与 canonical browser gate 通过；仅 current demo |
| ProPainter full-mask clean plate | 技术执行成功，但产生灰白模糊光斑 | 历史 rejected |
| 标定 donor clean plate | 有效 coverage 仅 0.17%-0.62%；零 margin 是前景泄漏 | rejected |
| SDXL anchor seed 2026071701 | mask 外逐像素不变；用户确认“当前效果可以先算通过”；移除区仍有枕头状生成 | `accepted_for_current_demo`；不证明 object-free/multiview/occluded geometry |
| 严格 R1-R4 sequence | 25 帧；front -> left -> right -> bed；每轮 predecessor hash 闭合、`unresolved=0`；最终 frame-set `854bd224...dfca` | `technical_passed_complete_sequence_with_limitations`；仅 current demo |
| 新 reconstruction input | 25 个严格 R4 RGB、camera subset、transforms、manifest/receipt；无任何旧 depth/NPY/NPZ | package passed；旧 v23 重建资格已作废 |
| fresh DA3 | 25 张 720x1280 depth 全 finite/positive；4M 点云全 finite/nonzero；严格四轮 lineage 已重哈希 | technical passed；仅允许进入 current-demo PGSR |
| fresh PGSR / TSDF | PGSR 30k：454,617 Gaussians、PSNR 33.2610672 dB；TSDF post：1,805,667 vertices / 3,556,615 faces；sequence/DA3/PGSR/TSDF hash 闭合 | `technical_passed_*_current_demo_only`；alignment passed |
| canonical Web promotion | manifest `58cc2b06...ab7cb`；final QA `passed_current_demo_only` 且 `failures=[]`；finalization receipt 绑定 exact QA bytes；stable alias `3805f0e5...5e82b` 未变 | `promoted_current_demo_only`；旧稳定基线保留 |

因此当前结论必须写成：**TRELLIS2 PBR 枕头的整体形状、完整厚度、浅色主色与 source-camera silhouette 已通过，背面花纹偏差作为 minor limitation 保留；bedroom_4 已真实执行 front -> left -> right -> bed 的四轮 clean plate，每一轮只消费紧邻上一轮 composite，并用累计 mask 约束原始 RGB-D donor。最终背景、fresh DA3、PGSR 30k、TSDF、坐标对齐与 canonical Web finalization receipt 已形成 hash-closed 链，逐轮 `unresolved=0`，desktop/mobile QA failures 为空。canonical 状态是 `promoted_current_demo_only`，不是高质量背景补全完成；R4 的低频拉伸、床形色块和简化平面仍是明确 limitation。**

## Scene commands 如何触发补全

自然语言查询只读取 scene graph；add/delete/update/split/merge/reparent 则先生成 typed preview，不直接改 manifest。服务端重新加载固定 WorldManifest、重建 Pydantic plan，并把确认短语绑定到 request、intent 和 manifest hash。确认后返回 `queued` 只表示写入 immutable async job，不表示 pipeline 已完成。

命令与补全的关系如下：

| 命令 | 触发的补全行为 |
|---|---|
| add | segmentation -> lift -> routing -> object completion -> placement -> clean plate/reinspect |
| delete | tombstone 对象，但必须修复暴露背景并重建相关 scene layers |
| split | 为每个稳定实例建立独立 evidence、completion history 与父子关系 |
| merge | 合并语义单元并重新验证 child、placement、bundle 和 Web |
| update appearance/interaction | 外观证据变化时重跑 completion；启用 collision 时必须验证同一 PBR GLB 的 collider role/topology（旧模式才生成独立 collider） |
| reparent | 保持 world transform，重跑 cognition/placement/bundle/Web 层级 QA |

客户端 preview、`clientPreview.targetIds` 和阶段列表都不是执行真值。歧义目标、过期 manifest、错误确认短语或 invariant 失败均返回 blocked，且不会创建 job。

## 证据与文档图来源

文档图是现有 evidence 的逐字节副本，没有重新渲染或美化：

| 文档资产 | 原始 evidence | SHA-256 |
|---|---|---|
| `assets/completion/object-mesh-six-view.png` | `examples/bedroom4/completion/parametric_pillow_front_attempt1/browser_six_view_contact_sheet.png` | `3d94fa48f8bcc899e6e804a0b6f9df1bd18cb6effea70bb530e4809a058f3ed9` |
| `assets/completion/object-gaussian-six-view.png` | `examples/bedroom4/completion/parametric_pillow_front_attempt1/gaussian_six_view_contact_sheet.png` | `8437bbf5daec603362c78cb6db7407632d23eee38c5ad35c5b45bd4b7a44f2e8` |
| `assets/completion/object-pbr-six-view.png` | `examples/bedroom4/completion/trellis2_pillow_front_seed42/scene_fit_silhouette_refined/canonical_six_view_review/object_six_view_contact_sheet.png` | `5b11f4069bcf66bbe09638fef849c24afc43d14d147648e7e44acd6e0f7a9f5a` |
| `assets/completion/clean-plate-donor-prefill.png` | `examples/bedroom4/completion/clean_plate_round2_multiview/multiview_prefill_contact_sheet.png` | `e0691b29a0df1d21c7fecf1cb62892e64bb1e03b5b2bba7bbd1cbe165787ccf1` |
| `assets/completion/clean-plate-edge-leakage.png` | `examples/bedroom4/completion/clean_plate_round2_multiview/frame64_zero_margin_support.png` | `b8fa37496391f42d6823fc9b7421a7fc75926d504b1fc1fa7bb763a3ed4689b4` |
| `assets/completion/source-camera-silhouette.png` | `examples/bedroom4/completion/parametric_pillow_front_attempt1/scene_fit_silhouette_refined_attempt3/source_camera_review/source_camera_silhouette_overlay.png` | `e8c7648f6b18bcebb217031f34b366e813da25dfcbff928093fc808a6a8263d3` |
| `assets/completion/source-camera-pbr-silhouette.png` | `examples/bedroom4/completion/trellis2_pillow_front_seed42/scene_fit_silhouette_refined/source_camera_review/source_camera_silhouette_overlay.png` | `ade9e0c9632da1ac9f01300d0ca8805754ff82b7188614704913b2ac0d3f6aa0` |
| `assets/completion/clean-plate-sdxl-seed-2026071701.png` | `examples/bedroom4/completion/clean_plate_round3_sdxl_anchor/candidate_seed_2026071701.png` | `26a734c0296d98790ed4daaf5a5a7b45c64f25430ab95e6115e9a26eff41d7b0` |
| `assets/completion/layered-source-25view.png` | `examples/bedroom4/completion/layered-peel/layered-clean-plate-sequence-v1/qa/source_contact_sheet.png` | `c18e229a9b1c9b04bcb044039d9dcdb67d78c9631ce965131b47df67e4869b71` |
| `assets/completion/layered-round01-front-pillow.png` | `examples/bedroom4/completion/layered-peel/layered-clean-plate-sequence-v1/qa/round01_contact_sheet.png` | `fdbdc172137391bf46890ceeed026f3b9a678e7271353db5e44b9133a8749066` |
| `assets/completion/layered-round02-left-pillow.png` | `examples/bedroom4/completion/layered-peel/layered-clean-plate-sequence-v1/qa/round02_contact_sheet.png` | `07403a3fac576710b9fdaec3956091e623051fdb6080ac1115daf01ded4b8aa3` |
| `assets/completion/layered-round03-right-pillow.png` | `examples/bedroom4/completion/layered-peel/layered-clean-plate-sequence-v1/qa/round03_contact_sheet.png` | `f30829672c48ca826c2fce37a6f4f9c01d7504e9f8393567a261cea01e7477a6` |
| `assets/completion/layered-round04-final-background.png` | `examples/bedroom4/completion/layered-peel/layered-clean-plate-sequence-v1/qa/round04_contact_sheet.png` | `4118d32e18c7151dc2013036c07a7dc2229e039730c01098c5e04684767fa7a0` |
| `assets/completion/strict-clean-scene-web-desktop.png` | `web/public/worlds/bedroom4/qa/clean-unified-scene-browser-final-promoted/desktop-1440x900.png` | `c7b6bbbec80aad6fad79db7a71db0413c8af65d2e64e6f95eb760ed12566cc3d` |
| `assets/completion/strict-clean-scene-web-mobile.png` | `web/public/worlds/bedroom4/qa/clean-unified-scene-browser-final-promoted/mobile-390x844.png` | `3ed94cb223727c8397bda9166f40eaab73fefdc2e5f667c8258e06c8d884a692` |

完整机器可读证据还包括 `layered-clean-plate-sequence-v1/execution_receipt.json`、`layered_clean_plate_sequence_report.json`、四轮 `layered_composite_report.json`/receipt、portable R4 mask index 和 `qa/visual_review.json`。累计 mask / donor 合同位于 `examples/bedroom4/completion/layered-peel/cumulative-rgbd-reprojection/summary.json` 及三个 `round0*/manifest/cumulative_removal_manifest.json`；fresh geometry 证据位于 `examples/bedroom4/assets-local/strict-clean-scene-mirror/reconstruction/receipts/`，实际 PGSR/TSDF 产物位于同一 mirror 的 `reconstruction/pgsr_scannetppv2_all/bedroom_4/`。canonical Web 证据为 `web/public/worlds/bedroom4/manifest.json`、`qa/clean-unified-scene-browser-qa.final-promoted.json` 与 `qa/strict-clean-scene-finalization-receipt.json`。文档只解释 receipt，不覆盖 receipt。

## 最低发布条件

一个对象层只有同时满足以下条件才算 passed：

1. stable instance、source frames、masks、camera/depth 和 appearance contract 可追溯；
2. front 保留 observed 的整体形状、主色类别和关键部件，六视图有真实厚度且不存在缺面/片状；轻微不可见面纹理或材质幻觉可以 `status=passed`，但必须记录 limitation；
3. VLM/技术 review 按 severity 在最多三次内 accept；只有明显形变、主色错误、缺面/片状或部件断裂触发 retry/reject；
4. scene fit、支撑、对象间和对象/背景穿模通过；
5. clean plate 的 mask 外不变、跨视图、背景语义和新 depth/normal 通过；
6. clean scene Gaussian/mesh 与 unified object GLB 共坐标；旧多表示模式中的 object visual/collider 也必须共坐标；
7. Web overlay、旋转、碰撞、focus 与 scene QA 通过；
8. 所有输出有 hash、provider、seed/prompt、状态和 limitation。

任一 blocking 项失败，当前层和更深遮挡层保持 blocked。用户可以对指定 demo 做 scope-limited 人工放行，但 receipt 必须同时列出 allowed scope 和 not-proven claims；这种放行不会把单帧生成升级为 object-free/multiview 几何证据。保留失败 evidence 和 minor limitation 是系统能力的一部分，因为它既阻止“片状对象已经补全”这样的明显错误，也避免因轻微不可见面纹理偏差无限重跑。
