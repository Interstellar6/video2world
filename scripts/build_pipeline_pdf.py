#!/usr/bin/env python3
"""Build the illustrated Video2World pipeline handoff PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#172126")
MUTED = colors.HexColor("#59666B")
TEAL = colors.HexColor("#137A76")
TEAL_SOFT = colors.HexColor("#E8F3F1")
AMBER = colors.HexColor("#B46B18")
AMBER_SOFT = colors.HexColor("#FFF1DD")
RED = colors.HexColor("#A54336")
LINE = colors.HexColor("#CAD3D4")
PAPER = colors.HexColor("#FAFBFA")


def register_fonts() -> tuple[str, str]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError("No Chinese-capable system font found")
    pdfmetrics.registerFont(TTFont("Video2WorldCJK", str(font_path)))
    return "Video2WorldCJK", "Video2WorldCJK"


def styles(font: str, bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=bold,
            fontSize=28,
            leading=36,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName=font,
            fontSize=12,
            leading=20,
            textColor=MUTED,
            spaceAfter=6 * mm,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName=bold,
            fontSize=20,
            leading=28,
            textColor=INK,
            spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=14,
            leading=20,
            textColor=TEAL,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9.5,
            leading=16,
            textColor=INK,
            spaceAfter=2.6 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.5,
            leading=11,
            textColor=MUTED,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.8,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=1.5 * mm,
            spaceAfter=3 * mm,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9,
            leading=15,
            textColor=INK,
            backColor=TEAL_SOFT,
            borderColor=TEAL,
            borderWidth=0.7,
            borderPadding=8,
            spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        ),
        "warning": ParagraphStyle(
            "Warning",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8.8,
            leading=14,
            textColor=INK,
            backColor=AMBER_SOFT,
            borderColor=AMBER,
            borderWidth=0.7,
            borderPadding=8,
            spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        ),
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def fit_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def figure(path: Path, caption: str, style: ParagraphStyle, *, height: float = 78 * mm):
    return KeepTogether(
        [
            fit_image(path, 174 * mm, height),
            paragraph(caption, style),
        ]
    )


def figure_cell(path: Path, caption: str, style: ParagraphStyle, *, height: float):
    return [
        fit_image(path, 82 * mm, height),
        paragraph(caption, style),
    ]


def data_table(rows: list[list[str]], font: str, widths: list[float]) -> Table:
    body_style = ParagraphStyle(
        "TableBody",
        fontName=font,
        fontSize=7.6,
        leading=11,
        textColor=INK,
    )
    cells = [[paragraph(cell, body_style) for cell in row] for row in rows]
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PAPER]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def page_decorator(canvas, document) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFont("Video2WorldCJK", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "Video2World · Pipeline & Asset Contract · 2026-07-16")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, str(document.page))
    canvas.restoreState()


def build(output: Path, asset_root: Path) -> None:
    font, bold = register_fonts()
    style = styles(font, bold)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="Video2World：从视频到可交互 3D 世界",
        author="Video2World",
        subject="Pipeline, asset contracts, real bedroom_4 results and Web runtime",
    )
    story = []

    story.extend(
        [
            Spacer(1, 8 * mm),
            paragraph("VIDEO2WORLD", style["small"]),
            paragraph("从视频到可交互 3D 世界", style["title"]),
            paragraph(
                "独立 Pipeline、强类型资产合同、真实 bedroom_4 运行证据，以及 Spark 3DGS + MeshBVH 碰撞 + 场景问答的 Web Runtime。",
                style["subtitle"],
            ),
            figure(
                asset_root / "01-pgsr-scene.png",
                "PGSR 30k 场景 Gaussian：视觉层保留房间外观，碰撞与语义由独立层承担。",
                style["caption"],
                height=104 * mm,
            ),
            paragraph(
                "完成口径：不是“一个 PLY 能打开”，而是视觉、碰撞、语义、对象、关系、hash、坐标、交互与浏览器 QA 同时闭环。",
                style["callout"],
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            paragraph("Pipeline 总览", style["h1"]),
            paragraph(
                "Video2World 不复制四个上游仓库到同一个环境。DA3、PGSR、SAM3、Video2Mesh Fusion 与 TRELLIS 保持独立 provider；本项目拥有十阶段编排、内容哈希恢复、跨模型合同、placement、bundle、query 与 Web runtime。",
                style["body"],
            ),
            data_table(
                [
                    ["阶段", "输入", "输出", "下一步"],
                    ["01 Ingest", "扫描视频", "frames + camera contract", "DA3 / PGSR / SAM3"],
                    ["02 DA3", "frames + cameras", "depth + 4M point prior", "PGSR / lifting"],
                    ["03 PGSR", "RGB + cameras + prior", "871,317 Gaussians + TSDF", "visual / collision"],
                    ["04 SAM3", "vocabulary + frames", "boxes / scores / masks", "3D fusion"],
                    ["05 Fusion", "masks + depth + cameras", "object clouds + semantic GS", "cognition / completion"],
                    ["06 Cognition", "instances + views", "captions + relations + bbox", "query / prompt"],
                    ["07 TRELLIS", "object RGBA + text", "Gaussian + GLB / OBJ", "placement"],
                    ["08 Placement", "scan bbox + canonical asset", "transform + carve + collider", "bundle"],
                    ["09 Bundle", "validated layers", "world manifest + hashes", "CLI / Web"],
                    ["10 Web", "manifest + chunks", "robot + interaction + QA", "browser delivery"],
                ],
                font,
                [23 * mm, 43 * mm, 63 * mm, 43 * mm],
            ),
            Spacer(1, 4 * mm),
            paragraph(
                "恢复原则：stage cache 同时比较 config、显式输入、依赖输出与本阶段输出的内容 hash。adoption-first 模板保持 command=null；完整 provider profile 则以 preflight + 十个非空 argv 连接现场 driver。二者都不会伪造上游执行。",
                style["callout"],
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            paragraph("01-03 · 相机、几何与场景重建", style["h1"]),
            paragraph("01 Ingest", style["h2"]),
            paragraph(
                "从视频抽取稳定帧与时间戳，写出唯一相机 convention。bedroom_4 使用 80 帧；相机变换 round-trip 最大绝对误差约 1.33e-15。下游不得重新猜 world/camera 方向。",
                style["body"],
            ),
            paragraph("02 DA3", style["h2"]),
            paragraph(
                "生成逐帧 depth/confidence 与 4,000,000 点 scene prior。它是 PGSR 初始化和 2D-to-3D 可见性证据，不是最终 mesh，也不携带语义。",
                style["body"],
            ),
            paragraph("03 PGSR", style["h2"]),
            paragraph(
                "官方单场景优化至 30k，输出 871,317 Gaussian，PSNR 33.1155 dB、L1 0.0118202；render depth/normal 经 TSDF 得到 694,773 vertices / 1,351,454 faces。",
                style["body"],
            ),
            paragraph(
                "PSNR/L1 是训练视角指标，不是独立测试集成绩。raw PGSR 的数值健康审计为 unsafe；当前本地 fresh 归档没有远端 viewer-safe SuperSplat 派生物。本项目只声明 carved raw 在本次 Spark/Chrome 门禁中可用。",
                style["warning"],
            ),
            figure(
                asset_root / "04-tsdf-mesh.png",
                "TSDF scene mesh：连续几何和静态碰撞候选；3DGS 仍负责最终视觉。",
                style["caption"],
                height=92 * mm,
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            paragraph("04-05 · 开放词汇分割与三维语义", style["h1"]),
            paragraph(
                "fresh run 以 GroundingDINO query bank 作为类别发现代理，9 个候选类别中 8 类产生 mask；957 个 source instance masks 合并为 616 个 class-frame probability masks。SAM3 只产生二维证据，必须经相机、深度与多视角 vote 才能成为 3D instance。",
                style["body"],
            ),
            Table(
                [[
                    figure_cell(asset_root / "05-instance-cloud.png", "独立对象点云", style["caption"], height=56 * mm),
                    figure_cell(asset_root / "06-semantic-gaussian.png", "semantic Gaussian", style["caption"], height=56 * mm),
                ]],
                colWidths=[86 * mm, 86 * mm],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]),
            ),
            paragraph(
                "基础 run 得到 13 条 3D records（10 foreground + 3 structure）。主 semantic PLY 保留全部 871,317 Gaussian，并追加 object_id/object_probability；8 类共选中 701,608 Gaussian。当前语义是 class-level，同类多实例不能只靠此 PLY 区分。基础 fusion 的真实命令使用 min-votes=1；隔离 pillow delta 才使用 min-votes=2。",
                style["callout"],
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            paragraph("Pillow Delta · 失败证据也保留", style["h1"]),
            paragraph(
                "fresh 没有独立 pillow，旧开放词汇标签又混入 bed。新 run 因此隔离执行，不覆盖权威目录。raw cloud 的最低回投命中率只有 0.8796，明确失败；最终采用 top-3、score>=0.90、5px erosion、relative depth tolerance 0.005、min-votes=2。",
                style["body"],
            ),
            Table(
                [[
                    figure_cell(asset_root / "11-pillow-sam3.png", "真实 SAM3 pillow mask", style["caption"], height=66 * mm),
                    figure_cell(asset_root / "12-pillow-projection.png", "accepted 3D cloud 回投", style["caption"], height=66 * mm),
                ]],
                colWidths=[86 * mm, 86 * mm],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]),
            ),
            paragraph(
                "accepted sam3_pillow_01：209,479 点；三视角命中率 0.9818 / 0.9863 / 0.9734；AABB center [-1.45619, -0.51465, 15.30182]。它是三个相触枕头的 ensemble，可 query/focus/visual rotation；当前没有 GLB/collider，不能声称机器人精确碰撞。",
                style["warning"],
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            paragraph("06 · Scene Cognition", style["h1"]),
            paragraph(
                "object_facts 连接稳定 ID、aliases、中英描述、AABB/OBB、relations 与 evidence。几何关系由离线 scene graph 决定；VLM 只描述可见外观，不拥有几何真值。bedroom_4 使用 Qwen/Qwen2.5-VL-3B-Instruct，精确 revision 66285546d2b821cf421d4f5eb2576359d3770cd3。",
                style["body"],
            ),
            paragraph(
                "模型校验：3,754,622,976 parameters；两片 safetensors 合计 7,509,245,952 bytes；Transformers 4.57.3、Torch 2.5.1+cu124、bfloat16、SDPA、RTX 3090；missing/unexpected/mismatched keys 与 load errors 均为零。6/6 描述通过。provider 为 local_huggingface_transformers+agent_visual_audit：保留 prompt、原始输出、输入图、frame/mask hash，再做显式 schema 规范化与逐图审计。",
                style["callout"],
            ),
            data_table(
                [
                    ["问题", "解析", "动作", "失败策略"],
                    ["枕头在哪里？", "sam3_pillow_01 / location", "回答在床上 + focus AABB", "无 bbox 时不 focus"],
                    ["枕头长什么样？", "sam3_pillow_01 / appearance", "返回 evidence caption", "无 caption 明确 unavailable"],
                    ["植物在哪里？", "category=plant / ambiguous", "显示实例候选", "不自动猜第一个"],
                    ["第二株植物在哪里？", "ordinal=2", "解析稳定 scoped ID", "越界 fail closed"],
                ],
                font,
                [35 * mm, 47 * mm, 48 * mm, 42 * mm],
            ),
            Spacer(1, 3 * mm),
            paragraph(
                "坐标单位为 scene_scale_not_metric：可返回 bbox center，但禁止把数值标成米。枕头是三只相触枕头的 accepted ensemble，location/appearance 共用 sam3_pillow_01 与同一 AABB。",
                style["body"],
            ),
            Table(
                [[
                    figure_cell(asset_root / "14-web-pillow-location.png", "位置问答：回答在床上并 focus AABB", style["caption"], height=54 * mm),
                    figure_cell(asset_root / "15-web-pillow-appearance.png", "外观问答：reviewed Qwen 描述与限制", style["caption"], height=54 * mm),
                ]],
                colWidths=[86 * mm, 86 * mm],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]),
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            paragraph("07 · EmbodiedGen V2 / TRELLIS", style["h1"]),
            paragraph(
                "单物体 RGBA 与文字条件生成 canonical Gaussian 和 GLB/OBJ，补足扫描不可见面。当前归档有 9 个 Gaussian PLY 与 9 组 GLB/OBJ；auto-completion 实际引用较早的 2026-07-13 Holi run，不是 07-14 fresh。首批场景替换对象是经过 source-anchor 与 placement QA 的 nightstand 01/02 与 plant 01/02；door 01/02 因语义不符被拒绝。",
                style["body"],
            ),
            Table(
                [[
                    figure_cell(asset_root / "08-plant-gaussian.png", "独立植物 Gaussian", style["caption"], height=79 * mm),
                    figure_cell(asset_root / "09-nightstand-gaussian.png", "独立床头柜 Gaussian", style["caption"], height=79 * mm),
                ]],
                colWidths=[86 * mm, 86 * mm],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]),
            ),
            paragraph(
                "Gaussian 与 GLB 是同一 RGBA/seed 的独立 decoder 输出，不是互相转换。两者必须分别记录 source hash、canonical bounds 与 scene placement。所有原始 GLB 非 watertight，因此不能自动当精确 collider。",
                style["warning"],
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            paragraph("08-09 · Placement、碰撞与 World Bundle", style["h1"]),
            paragraph(
                "被替换对象必须同时从 static Gaussian 与 static TSDF 中剔除，否则分别产生视觉重影和 stale collision。Gaussian、render mesh、simplified collider 与 bbox outline 统一挂到 scene-space 父组，共用 transform 和 pivot。",
                style["body"],
            ),
            data_table(
                [
                    ["Gate", "要求", "未通过时"],
                    ["file", "非空、hash/size 匹配、可重载", "candidate"],
                    ["semantic", "类别与扫描证据一致", "reject"],
                    ["alignment", "显式 transform/pivot、bbox 与支撑检查", "not interactive"],
                    ["visual", "无重影、旋转无漂移", "candidate"],
                    ["collision", "简化 GLB、bounds、BVH、robot/stale-face QA", "degraded / blocked"],
                ],
                font,
                [30 * mm, 95 * mm, 47 * mm],
            ),
            Spacer(1, 4 * mm),
            paragraph(
                "world-manifest-1.0.0 记录 scene visual/collider/semantic 三层，以及 object scoped ID、caption、evidence、bbox、Gaussian、mesh、collider、transform、relation、interaction 和五项 gate。validated manifest 会重新校验所有本地资产 SHA-256。",
                style["callout"],
            ),
            paragraph(
                "collision-enabled 对象必须通过五项 gate；visual-only 对象可在 file/semantic/alignment/visual 通过后交互，但必须无 collider、collision_enabled=false，且 collision gate 保持 not_tested。枕头属于后者。",
                style["body"],
            ),
            paragraph(
                "对象主键 = world_id::source_run_id::object_id。fresh Holi 与旧 EmbodiedGen 的同名 ID 不会只凭裸字符串自动绑定。",
                style["body"],
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            paragraph("10 · Web Runtime", style["h1"]),
            paragraph(
                "稳定提交 252a85c 锁定相机、PGSR/TSDF 对齐和机器人 spawn；a5af3ae 提供对象 carve 与 360° 交互参考。Spark 只渲染 Gaussian，Three.js + MeshBVH 负责选择、地面、障碍、天花板与机器人碰撞。",
                style["body"],
            ),
            figure(
                asset_root / "13-web-production-overview.png",
                "Production bedroom_4：carved scene 3DGS/TSDF、四个碰撞对象、枕头 visual-only 组件、机器人与 scene QA 同场运行。",
                style["caption"],
                height=78 * mm,
            ),
            paragraph(
                "Production QA passed：823,391 static Gaussians + 480,000 object Gaussians + 209,479 pillow RGB points = 1,512,870 visual primitives；1,265,671 static collider faces；四个 simplified GLB 共 67,660 faces、4/4 ready、degraded 0。pillow carve 仅移除 26,731 个近邻 Gaussian，并保留 accepted AABB 内 4,038 个未匹配点。",
                style["callout"],
            ),
            Table(
                [[
                    figure_cell(asset_root / "16-web-object-overlay.png", "plant_01 Gaussian / bbox / GLB collider 对齐", style["caption"], height=48 * mm),
                    paragraph(
                        "实测交互：nightstand_01 pointer drag；plant_01 双击 360 回位；pillow point visual + bbox drag/360；四对象 robot BVH blocking；pillow location/appearance resolve + focus。portable-manifest 回归 FPS samples 42/43/40/38/38，平均 40.2、最低 38；桌面/390x844 与 runtime error 门禁均通过。",
                        style["body"],
                    ),
                ]],
                colWidths=[86 * mm, 86 * mm],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]),
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            paragraph("验证状态与复现入口", style["h1"]),
            data_table(
                [
                    ["范围", "结果", "口径"],
                    ["Python", "42 pytest + Ruff + lock/build", "DAG/schema/provider/query/recovery passed"],
                    ["Web fixture", "25 Vitest + 2 Playwright E2E", "fresh root + contract + GLB/BVH/drag/spin/query"],
                    ["Holi fresh", "DA3/SAM3/PGSR/Fusion/semantic GS", "真实 run；非论文全量数据工厂"],
                    ["Pillow delta", "209,479 points；3-view >=0.95", "visual-only drag/spin；no collider"],
                    ["Qwen cognition", "6/6 passed；exact revision + evidence hashes", "Qwen output + schema normalization + visual audit"],
                    ["TRELLIS", "9 Gaussian + 9 GLB/OBJ", "placement/collision 逐对象 gate"],
                    ["Production Web", "1,512,870 primitives；4/4 GLB；avg 40.2 FPS", "desktop/mobile；0 runtime errors"],
                ],
                font,
                [34 * mm, 62 * mm, 76 * mm],
            ),
            Spacer(1, 6 * mm),
            paragraph("核心命令", style["h2"]),
            paragraph(
                "uv sync --all-groups<br/>"
                "uv run video2world init runs/&lt;scene&gt; --video /abs/input.mp4 --scene-id &lt;scene&gt;<br/>"
                "uv run video2world plan runs/&lt;scene&gt;<br/>"
                "uv run video2world adopt-existing ...<br/>"
                "uv run video2world validate runs/&lt;scene&gt;/bundle/world.json<br/>"
                "uv run video2world query ... \"枕头在哪里?\"<br/>"
                "npm test &amp;&amp; npm run build<br/>"
                "npm run dev -- --port 4173",
                style["callout"],
            ),
            paragraph("明确限制", style["h2"]),
            paragraph(
                "当前 bedroom_4 未公制标定；semantic PGSR 是 class-level；raw PGSR 数值健康为 unsafe 且本地缺 viewer-safe 派生物；TRELLIS 原始 GLB 非 watertight；pillow 尚无 mesh/collider；Holi official VLM confidence agent 与 official spatial QA 未在 fresh run 执行。所有这些限制都保留在 manifest 与进展记录中。",
                style["warning"],
            ),
            paragraph(
                "文档库：relumeow.top/video2world/  ·  项目：Interstellar6/video2world",
                ParagraphStyle(
                    "FooterLink",
                    parent=style["body"],
                    fontName=bold,
                    fontSize=10,
                    textColor=TEAL,
                    alignment=TA_CENTER,
                    spaceBefore=8 * mm,
                ),
            ),
        ]
    )

    document.build(story, onFirstPage=page_decorator, onLaterPages=page_decorator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/video2world-pipeline.pdf"),
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("docs/video2world/assets/pipeline"),
    )
    args = parser.parse_args()
    build(args.output.resolve(), args.asset_root.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
