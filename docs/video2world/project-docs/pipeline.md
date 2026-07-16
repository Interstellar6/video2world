---
title: Pipeline：从视频到可交互 3D 世界
id: video2world-project-pipeline
category: 项目文档
visibility: public
updated: 2026-07-16
summary: Video2World 十阶段 pipeline 的职责、上下游、输入输出、质量门禁，以及 bedroom_4 的真实产物统计与图像证据。
tags:
  - Pipeline
  - 3DGS
  - SAM3
  - EmbodiedGen V2
  - Scene QA
---

# Pipeline：从视频到可交互 3D 世界

Video2World 接收一段扫描视频，最终输出的不是一个孤立 PLY，而是一个分层 world bundle：PGSR Gaussian 负责视觉，TSDF/GLB 负责碰撞，SAM3 与多视角投影负责语义实例，TRELLIS 提供独立物体补全候选，scene graph 与描述 sidecar 负责查询，Web runtime 把这些层放回同一坐标系。

```text
video
  -> 01 ingest: frames + cameras
  -> 02 DA3: depth + dense point prior
  -> 03 PGSR: scene Gaussian + TSDF mesh
  -> 04 open-vocabulary + SAM3: masks + tracks
  -> 05 fusion: object clouds + semantic Gaussian
  -> 06 cognition: captions + bbox + relations
  -> 07 TRELLIS: object Gaussian + GLB/OBJ
  -> 08 placement: shared transform + carved scene + colliders
  -> 09 bundle: hashes + provenance + quality gates
  -> 10 Web: Spark visual + Three.js collision + robot + scene QA
```

这十个业务阶段有两种进入方式。`default_pipeline.yaml` 是 adoption-first 模板，所有命令保持空值，适合登记既有真实结果；`holi_embodiedgen_upstream.yaml` 在前面增加一个 provider preflight 节点，并把十个业务阶段全部接到 `holi_embodiedgen.provider.example.yaml` 的版本化 argv 合同。现场必须绑定真实 Python、仓库、checkpoint 与十个 driver；preflight 通过只表示静态依赖可用，只有某次 run 的输出角色、非空 hash、stage receipt 和质量门禁都通过，才能称为该场景已验证。Bedroom4 的通过状态不会被继承到新视频。

![PGSR scene Gaussian](../assets/pipeline/01-pgsr-scene.png "bedroom_4 的 PGSR 30k 场景级 Gaussian；视觉层保留真实外观，也能看见边缘拉丝等重建伪影")

## 一览表

| 阶段 | 直接输入 | 主要输出 | 下一步消费者 |
|---|---|---|---|
| 01 Ingest | 扫描视频 | 帧、时间戳、相机内外参合同 | DA3、PGSR、SAM3 |
| 02 DA3 | 帧、相机上下文 | depth、confidence、稠密点云 prior | PGSR 初始化、2D-to-3D lifting |
| 03 PGSR | RGB、相机、DA3 prior | scene Gaussian PLY、render depth/normal、TSDF mesh | Web 视觉、语义投影、碰撞 |
| 04 Open-vocabulary + SAM3 | 代表帧、类别词表、全视频帧 | 2D boxes/scores/masks、跨帧证据 | 3D fusion、caption evidence |
| 05 Fusion | masks、depth、cameras、DA3/PGSR | 对象点云、AABB/OBB、semantic Gaussian | scene graph、completion、Web focus |
| 06 Cognition | 实例、代表视图、几何关系 | 中英描述、relations、query facts | Web scene QA、生成提示 |
| 07 TRELLIS | 单物体 RGBA、描述、seed | object Gaussian、GLB/OBJ 候选 | placement、render mesh、collider |
| 08 Placement | 扫描 bbox、canonical asset、TSDF | `T_scene_from_asset`、pivot、carved scene、简化 collider | world bundle、机器人碰撞 |
| 09 Bundle | 所有通过门禁的层 | 强类型 world manifest、hash、provenance | CLI validate/query、Web adapter |
| 10 Web Runtime | Web manifest 与分片资产 | 3DGS 展示、机器人、选择/旋转、问答 | 浏览器 QA 与最终体验 |

## 01. Ingest：把视频变成可复用相机合同

上一步只有原始视频。本阶段按稳定策略抽帧，保留原始帧号与时间戳，并统一相机字段：`frame_id`、图像尺寸、内参、`world_to_camera`/`camera_to_world`、坐标轴、handedness 与单位状态。任何下游都不能重新猜相机 convention。

输出是 `frames_manifest` 与 `cameras`。bedroom_4 权威 run 使用 80 帧；相机 round-trip 最大绝对误差约 `1.33e-15`。下一步 DA3、PGSR 和 SAM3 共同消费同一组帧身份。

门禁：帧数一致、图片存在、旋转矩阵近似正交、变换互逆。当前场景只有 `scene_scale_not_metric`，所以后续坐标不能写成“米”。

## 02. DA3：深度与稠密几何先验

DA3 读取帧和相机上下文，输出逐帧 depth/confidence 与一个稠密 scene point prior。它为 PGSR 提供初始化，也为 mask lifting 提供逐像素可见深度；它不是最终 mesh，更不携带物体语义。

bedroom_4 的真实输出是 4,000,000 点的 `pointcloud_da3.ply`。下一步 PGSR 使用该几何先验优化 scene Gaussian，Fusion 使用 depth + camera 把 SAM3 像素证据投到三维。

门禁：深度尺寸与 RGB 对齐、有限值比例、相机/点云坐标一致、点数与 hash 固化。

## 03. PGSR：场景视觉层与连续表面

PGSR 对每个场景做多视图优化。输入是 RGB、相机和 DA3 prior，输出 GraphDECO-compatible Gaussian PLY，以及由 rendered depth/normal 融合得到的 TSDF mesh。Gaussian 负责照片级视觉，TSDF 负责连续表面；两者职责不同。

bedroom_4 实测：iteration 30,000、871,317 Gaussians、PSNR `33.1155 dB`、L1 `0.0118202`。`tsdf_fusion_post.ply` 含 694,773 vertices / 1,351,454 faces。

这里的 PSNR/L1 是训练视角指标，不是独立测试集成绩。raw PGSR 的数值健康审计为 unsafe（scale p99 约 `0.6656`、elongation p99 约 `6.25e11`、rotation norm error p99 约 `0.858`）；本地归档未包含远端生成过的 viewer-safe SuperSplat 派生物。当前 carved raw PGSR 已通过本项目 Spark/Chrome 生产门禁，但不能据此宣称它对任意 viewer 都安全。

![TSDF scene mesh](../assets/pipeline/04-tsdf-mesh.png "PGSR render depth 经 TSDF fusion 得到的场景 mesh；它承担连续几何与静态碰撞候选，不替代 3DGS 视觉")

下一步 SAM3 语义可直接投影回 PGSR centers，Web 则以 scene Gaussian 为静态视觉底层、以处理后的 TSDF 为静态 collision world。门禁包括 PLY Gaussian 字段、mesh face、训练 iteration、render 指标和人工视觉检查；边缘拉丝被记录为视觉问题，不用激进裁剪破坏墙地面。

## 04. Open-vocabulary 与 SAM3：从词语到跨帧 mask

先在代表帧发现开放词汇类别，再把类别或 visual prompt 交给 SAM3，在所有帧产生 boxes、scores、masks 与 track evidence。SAM3 只给二维证据，不直接产生 3D bbox。

fresh bedroom_4 使用 GroundingDINO 固定 query bank 作为类别发现代理：9 个候选类别、8 个类别实际产生 mask，957 个 source instance masks 合并成 616 个 class-frame probability masks。这不是 Holi-Spatial 论文里的 Gemini 动态类别记忆，manifest 会保留 provider 边界。

为验证“枕头在哪里”，另行执行了隔离 pillow delta：80/80 帧、raw 293 masks；通过每帧 top-3、score `>=0.90`、5px erosion 后保留 223 个高置信实例证据，且没有覆盖 fresh run。

![Real SAM3 pillow evidence](../assets/pipeline/11-pillow-sam3.png "真实 bedroom_4 帧上的 SAM3 pillow mask 预览；三个相触枕头在本轮作为一个 ensemble")

门禁：每个 mask 绑定 frame ID、prompt、score、像素尺寸与 source run；空类别、低分 mask 和跨 run 裸 ID 不得静默混用。

## 05. 多视角 Fusion：二维 mask 变成三维实例

本阶段把 mask、相机和 depth 联合起来。每个像素沿相机射线回到可见三维点，并用 mask probability、深度一致性和多视角 vote 聚合；随后用 voxel DBSCAN 拆分实例，生成对象点云、AABB/OBB，并把对象概率写回 PGSR Gaussian。

fresh 基础 run 得到 13 条 3D records：10 个前景实例和 3 个结构实例。主 semantic PGSR 保留全部 871,317 个 Gaussian，并追加 `object_id:int` 与 `object_probability:float`；8 个有效类别合计选中 701,608 个 Gaussian。当前 semantic 3DGS 是 class-level，不能把同类 lamp 自动当作稳定 instance identity。

fresh 基础融合的真实命令使用 `--min-votes 1`；旧报告生成器硬编码的“至少 2 票”不是该 run 的执行事实。下面的隔离 pillow delta 才单独使用 `min_votes=2`，两者必须按 source run 分开记录。

![Object-level point cloud](../assets/pipeline/05-instance-cloud.png "多视角投影融合后的独立对象点云示例")

![Semantic Gaussian](../assets/pipeline/06-semantic-gaussian.png "带语义类别颜色的场景 Gaussian；视觉层仍与 PGSR 场景共坐标")

pillow delta 最初混入床头板，三视角最低投影命中率仅 `0.8796`，因此明确失败。最终采用 5px erosion、relative depth tolerance `0.005`、`min_votes=2`，得到 209,479 点的 `sam3_pillow_01`，三视角命中率 `0.9818 / 0.9863 / 0.9734`，通过 `>=0.95` 门禁。

![Pillow projection QA](../assets/pipeline/12-pillow-projection.png "accepted pillow 3D cloud 回投到真实帧后的命中检查")

下一步 Cognition 消费 bbox/证据，TRELLIS 消费最佳视角 RGBA，Web 用 bbox 做 focus/highlight。未被识别或未被替换的 scene Gaussians 始终保留在静态层。

## 06. Scene Cognition：描述、关系和可审计问答

输入是稳定实例 ID、AABB/OBB、代表帧与 mask crop。输出 `object_facts`：中英名称与 aliases、详细外观描述、位置描述、`OnTopOf/SupportedBy/Near/...` 关系、证据帧和置信度。

本次 bedroom_4 使用本地 Hugging Face Transformers 运行 `Qwen/Qwen2.5-VL-3B-Instruct`，精确 revision 为 `66285546d2b821cf421d4f5eb2576359d3770cd3`。模型校验记录为 3,754,622,976 parameters、两片 safetensors 合计 7,509,245,952 bytes；`transformers 4.57.3`、`torch 2.5.1+cu124`、`bfloat16`、SDPA、RTX 3090，且 missing/unexpected/mismatched keys 与 load errors 均为零。6/6 个对象描述通过结构与证据门禁。

每条描述保留 prompt、原始模型输出、输入图、frame/mask hash 与 model revision。最终 provider 标记为 `local_huggingface_transformers+agent_visual_audit`：先由 Qwen 生成，再做显式 schema 规范化和逐图人工审计修正；它不是未经复核的纯模型原文，也不等同于 Holi-Spatial official VLM confidence agent。

几何关系由离线 scene graph 决定；VLM 只描述可见外观或润色语言，不拥有几何真值。查询 resolver 先解析 `id/name/category/alias`，同类多实例要求序号或用户选择，未知对象 fail closed。解析成功后返回稳定 object ID 与 bbox，Web 使用同一 selection API 聚焦相机并显示边界框。

例如 `枕头在哪里？` 现在解析到 `sam3_pillow_01`，回答“枕头组合位于床上”，并聚焦 accepted AABB；`枕头长什么样？` 读取有 evidence 的描述。由于本轮三个枕头相触，答案必须说明这是一个 ensemble，不能伪装成三个稳定实例。

![Pillow location QA](../assets/pipeline/14-web-pillow-location.png "真实 production Web 查询‘枕头在哪里’：resolver 返回 sam3_pillow_01，相机聚焦 accepted AABB 并显示枕头在床上的证据回答")

![Pillow appearance QA](../assets/pipeline/15-web-pillow-appearance.png "真实 production Web 查询枕头外观：复用同一 object ID 与 bbox，展示 Qwen 描述经审计后的三枕头 ensemble 细节与不确定性")

门禁：描述必须记录 provider/model/prompt/evidence；没有描述时返回“未生成”，没有 bbox 时不执行伪 focus，未公制标定时不输出米制距离。

## 07. EmbodiedGen V2 / TRELLIS：独立物体补全

将带 alpha 的单物体 RGBA、类别和描述送入 TRELLIS，分别输出 canonical Gaussian PLY 与 mesh/GLB/OBJ。该步骤补足扫描不可见面，使对象能从场景中独立选择和旋转；它不是扫描几何的自动精确替代。

![Nightstand completion](../assets/pipeline/02-nightstand-completion.png "床头柜补全输入/结果示例")

![Plant completion](../assets/pipeline/03-plant-completion.png "植物补全输入/结果示例")

![Plant Gaussian object](../assets/pipeline/08-plant-gaussian.png "TRELLIS 生成的独立植物 Gaussian")

![Nightstand Gaussian object](../assets/pipeline/09-nightstand-gaussian.png "TRELLIS 生成的独立床头柜 Gaussian")

当前真实归档有 9 个 final Gaussian PLY 和 9 组 GLB/OBJ。该 auto-completion 资产实际引用较早的 2026-07-13 Holi run，不是 07-14 fresh run，因此不能只凭同名 `object_id` 绑定。可接入 bedroom_4 的首批对象是经过单独 source-anchor 与 placement QA 的 `nightstand_01/02` 与 `plant_01/02`；两扇 door 因语义不符被拒绝。现有 GLB 均非 watertight，因此原始 render mesh 不自动等于合格 collider。

门禁：source RGBA/seed、Gaussian 与 GLB 分别记录 hash；类别语义、canonical bounds、非空 mesh、face 数、纹理和重载入均需验证。没有 `T_scene_from_asset` 的资产只能保持 candidate。

## 08. Placement 与碰撞：替换而不是叠加

输入是扫描对象点云 bbox、TRELLIS canonical bounds 和原始 scene layers。本阶段计算显式 `T_scene_from_asset` 与 scene-space pivot；Gaussian、render mesh、collider、outline 统一挂到同一父组。

被替换对象必须从静态 scene Gaussian 中 carve 掉，否则会视觉重影；也必须从静态 TSDF collider 中剔除旧面，否则机器人仍会撞到已经旋转前的位置。未识别对象不 carve，继续保持原样。

原始 GLB 作为可见 mesh 候选，碰撞使用无材质简化 GLB。当前工具用 Blender headless decimate、重新导入并检查 face target、非空 mesh 与 bounds 漂移，再由 Three.js/three-mesh-bvh 建 BVH。柔软枕头当前只有 point cloud 与 bbox，没有 mesh/collider，所以只允许 focus/视觉旋转，不声称机器人精确碰撞。

门禁分为 file、semantic、alignment、visual、collision 五项。collision-enabled 对象只有五项都 passed 才能注册机器人碰撞；visual-only 对象可在前四项通过后标记 `selectable + spin_360`，但必须保持 `collision_enabled=false`、无 collider 且 collision gate 为 `not_tested`。其他对象保持 candidate 或显式 `degraded-box`。

## 09. World Bundle：把来源和质量写进资产

Video2World 用 `world-manifest-1.0.0` 连接各层。scene 记录 visual/collider/semantic visual；object 记录 scoped identity、描述、证据、bbox/OBB、Gaussian、render mesh、collider、变换、关系、交互策略和质量门禁；每个本地资产都有 SHA-256 与字节数。

对象主键是 `world_id::source_run_id::object_id`。fresh Holi 与较早 EmbodiedGen 的同名对象不会只凭裸 ID 自动绑定。CLI 的 `validate` 会重新计算本地资产 hash；`adopt-existing` 只登记真实上游产物，不复制、不伪造外部执行。

每个 stage 同时写内容寻址 state：输入 hash、配置 hash、输出 hash、命令、时间、状态与日志。输入或输出变化后，下游自动变 stale；长任务可以从最近的真实成功阶段恢复。

## 10. Web Runtime：视觉、碰撞、机器人和问答共存

Web 以稳定提交 `252a85c` 的相机、坐标、机器人 spawn 与 PGSR/TSDF 对齐为 fixture，再迁移 `a5af3ae` 的静态 carve、对象父组与双击旋转。Spark 只渲染 scene/object Gaussian；Three.js mesh + MeshBVH 承担选择、ground probe、障碍、天花板和机器人碰撞。

![Production Web overview](../assets/pipeline/13-web-production-overview.png "bedroom_4 production bundle：carved PGSR Gaussian、carved TSDF、四个对象 Gaussian/GLB collider、机器人和场景问答在同一坐标系中运行")

用户左键命中对象后横向拖拽，父组绕 scene-space pivot 旋转，因此 Gaussian 与 GLB/collider 同步；双击执行 360 度并回到起始 quaternion。点击背景仍控制相机。问答输入会阻止 WASD/Space 事件传播，避免打字时机器人移动。

最终 production Chrome QA 已通过：四个 TRELLIS 对象先把 static PGSR 从 871,317 carve 到 850,122，再由 accepted pillow 最近邻保守 carve 26,731 个 Gaussian，保留 823,391 个 static Gaussians；叠加 480,000 个 object Gaussians 与 209,479 个 pillow RGB points，总计 1,512,870 个视觉 primitives。carved static TSDF 为 1,265,671 faces，四个 simplified GLB collider 共 67,660 faces，4/4 加载为真实 GLB、`degradedCount=0`。静态 TSDF carve 从原 mesh 移除 85,783 faces，四个碰撞替换区残留相交面为 0。

![Interactive object overlay](../assets/pipeline/16-web-object-overlay.png "真实 plant_01 的对象 Gaussian、bbox 与 simplified GLB collider 上下文叠加；browser placement review 已通过")

交互门禁实际执行了 `plant_01` 双击 360 回位、`nightstand_01` pointer drag、枕头 point visual + selection bbox 联动拖拽与 360 回位，以及四个 GLB 对象各自的机器人 BVH blocking probe；枕头 location/appearance query 均解析并聚焦 `sam3_pillow_01`。最终 portable-manifest 回归的 5 个 FPS sample 为 `42 / 43 / 40 / 38 / 38`，平均 40.2、最低 38；console messages、page errors 与 request failures 均为 0。生产报告同时通过 asset hash、static visual/collider carve、object collision technical QA、browser placement、cognition 与 pillow visual interaction gate。

![Pillow visual-only drag](../assets/pipeline/17-web-pillow-drag.png "枕头 RGB point visual 与 accepted AABB 选择框绕同一 pivot 联动旋转；该组件不注册机器人碰撞")

运行时仍显式报告 `sceneReady`、每个对象 collider mode/face 数、BVH target、FPS、选中 object ID 与 degraded count。本次 production 大资产报告同时记录 Chrome 1440×900 与 `390x844` fresh-load：移动端长描述换行、canvas/HUD 尺寸、水平溢出和控件不重叠均通过。

## 完成口径

一个 PLY 能打开、一个 GLB 能加载或一个网页能显示，都不等于 pipeline 完成。正式完成至少要求：

1. 场景 visual/collider/semantic 三层 hash 可验证；
2. 交互对象拥有稳定 ID、扫描证据、caption、bbox、visual 与显式 placement；collision-enabled 对象还必须有通过门禁的 collider，visual-only 对象必须明确无碰撞；
3. 静态 visual 与静态 collider 都剔除被替换区域；
4. Gaussian 与 mesh/collider 共用父变换，旋转无漂移；
5. scene query 解析、关系答案、相机 focus 与 bbox 高亮使用同一 object ID；
6. 浏览器桌面/移动、性能、console、碰撞与交互 QA 有真实记录；
7. 所有阶段可由 hash state 恢复，所有未完成项保持 candidate/failed/not-tested。

本页统计对应 2026-07-16 bedroom_4 权威归档；详细门禁和当前限制见 [bedroom_4 实测记录](../progress/bedroom4-20260716.md)。
