---
title: 安装、配置与恢复执行
id: video2world-project-getting-started
category: 项目文档
visibility: public
updated: 2026-07-16
summary: 安装 Video2World、初始化 run、配置上游 argv adapter、登记已有真实产物并按内容 hash 恢复执行。
tags:
  - CLI
  - Recovery
  - Adapters
---

# 安装、配置与恢复执行

Video2World 本体是轻量 orchestration/runtime 项目，不把 PGSR、SAM3、DA3 和 TRELLIS 强行塞进同一个 Python/CUDA 环境。各模型保留独立环境，pipeline 通过 argv adapter 和强类型产物合同连接。

![Pipeline stages](../assets/pipeline/01-pgsr-scene.png "上游模型保持独立环境，Video2World 以阶段合同、内容 hash 和质量门禁连接真实场景产物")

## 安装

```bash
git clone https://github.com/Interstellar6/video2world.git
cd video2world
uv sync --all-groups
npm install

uv run video2world --help
npm test
npm run build
```

Python 要求 3.11+；Web 要求当前 Node LTS 或更新版本。SAM3/PGSR/TRELLIS 的 CUDA 与权重由各自上游环境管理。

## 初始化一个 run

```bash
uv run video2world init runs/my-room \
  --video /absolute/path/to/scan.mp4 \
  --scene-id my_room

uv run video2world plan runs/my-room
uv run video2world run runs/my-room --dry-run
```

默认 adoption 模板的十个 stage 都是 `command: null`。这是有意的安全边界：项目不会猜远端 Conda、checkpoint 或上游仓库路径。`plan` 会把阶段标为 `adopt_or_configure`，直到用户配置 argv 或明确登记已有产物。

## 使用完整上游 provider profile

仓库同时提供 `video2world/configs/holi_embodiedgen_upstream.yaml` 和 `holi_embodiedgen.provider.example.yaml`。前者把 preflight、Holi ingest、DA3、SAM3、PGSR、Video2Mesh fusion、TRELLIS、cognition、placement、bundle 与 Web 都配置成非空 argv；后者定义每个站点 driver 的输入/输出角色、Python、仓库、checkpoint 和超时。

```bash
cp video2world/configs/holi_embodiedgen.provider.example.yaml \
  /secure/site/video2world-provider.yaml

# 编辑 site contract，把各 VIDEO2WORLD_*_DRIVER 和模型环境变量绑定到真实入口；
# 再把 pipeline profile 的 variables.provider_contract 改成该绝对路径。
uv run video2world init runs/my-room \
  --video /absolute/path/to/scan.mp4 \
  --scene-id my_room \
  --template-config video2world/configs/holi_embodiedgen_upstream.yaml

uv run video2world run runs/my-room --stage provider_preflight
uv run video2world plan runs/my-room
```

preflight 会在启动重模型前检查解释器、driver、仓库目录与 checkpoint；阶段执行不经过 shell，输入/输出角色必须精确一致，完成后重新计算非空内容 hash，并写不含 argv 明文和秘密值的 receipt。该 profile 是可执行的通用集成接口，但不是“任意视频零配置已经验证”的声明；Bedroom4 的硬编码历史脚本只作为 lineage evidence，真实通过状态仍只属于 Bedroom4 run。

## 配置 argv adapter

编辑 `runs/my-room/run.yaml`，为阶段写 argv 数组，不使用 shell 字符串：

```yaml
stages:
  pgsr:
    adapter: holi_pgsr
    needs: [ingest, da3]
    command:
      - /absolute/conda/env/bin/python
      - /absolute/upstream/train.py
      - --source_path
      - "{run_dir}/artifacts/ingest"
      - --model_path
      - "{run_dir}/artifacts/pgsr/model"
    cwd: /absolute/upstream/PGSR
    outputs:
      scene_gaussian: "{run_dir}/artifacts/pgsr/scene_gaussian.ply"
      scene_mesh: "{run_dir}/artifacts/pgsr/tsdf_scene.ply"
```

模板字段只允许预定义 key，命令由 `subprocess` 直接执行；敏感环境值必须写 `${ENV_VAR}` 引用，不能落盘到 config。

## 登记已有真实产物

已在 Holi-Spatial、Video2Mesh 或 EmbodiedGen 跑完的结果不必重算。按 stage 的 required roles 显式 adopt：

```bash
uv run video2world adopt-existing runs/my-room pgsr \
  --output scene_gaussian=/absolute/run/point_cloud.ply \
  --output scene_mesh=/absolute/run/tsdf_fusion_post.ply \
  --source-run-id holi-pgsr-30k-20260716 \
  --source-repository /absolute/Holi-Spatial \
  --source-commit <commit>
```

adopt 不会声称执行了外部命令，也不会把文件存在等同于质量通过。它记录 source run/repo/commit、真实内容 hash 和文件大小；原文件变化后 stage 自动失效。

## 恢复与目标执行

```bash
uv run video2world plan runs/my-room
uv run video2world run runs/my-room --stage bundle
uv run video2world run runs/my-room --stage web
```

恢复判断同时比较 stage config、显式输入、依赖 output hash 与本 stage output hash。目录同名但内容不同不会命中 cache；失败或中断阶段会重新执行，已验证且未变化的依赖保持 cached。

## 验证与查询

```bash
uv run video2world validate runs/my-room/bundle/world.json
uv run video2world query runs/my-room/bundle/world.json "枕头在哪里?"
uv run video2world query runs/my-room/bundle/world.json "第二株植物长什么样?"
```

`validate` 重新计算本地资产 SHA-256；`query` 完全离线，未知对象、多实例歧义或缺少 evidence 时 fail closed。坐标标为 `scene_scale_not_metric` 时，回答不会把数值误写成米。

## Web 开发

```bash
npm run dev -- --port 4173
```

打开 `http://127.0.0.1:4173/` 会加载仓库内的微型浏览器 fixture，保证 fresh clone 不依赖本地大资产也能冒烟验证。生产 bedroom bundle 通过 `?manifest=/worlds/bedroom4/manifest.json` 指定；其数百 MB 分片资产由本地 materializer 或发布构建提供，不进入 Git 和 JavaScript bundle。
