import {
  classifySceneQuestion,
  normalizeSceneQuery,
  resolveSceneEntity,
} from "./scene-query.js";

const FULL_RECONSTRUCTION_STAGES = [
  "inventory",
  "sam3",
  "fusion",
  "cognition",
  "completion_plan",
  "layered_completion",
  "placement",
  "bundle",
  "web",
];
const DELETE_STAGES = [
  "inventory",
  "completion_plan",
  "layered_completion",
  "placement",
  "bundle",
  "web",
];
const HIERARCHY_STAGES = ["cognition", "placement", "bundle", "web"];
const METADATA_STAGES = ["cognition", "bundle", "web"];
const CATEGORY_TERMS = [
  ["pillow", ["pillow", "pillows", "枕头", "靠枕", "抱枕"]],
  ["bed", ["bed", "beds", "床"]],
  ["plant", ["plant", "plants", "植物", "盆栽", "绿植"]],
  ["nightstand", ["nightstand", "bedside table", "床头柜"]],
  ["table", ["table", "desk", "桌子", "桌"]],
  ["lamp", ["lamp", "light", "灯", "灯具"]],
  ["chair", ["chair", "椅子", "椅"]],
  ["window", ["window", "窗户", "窗"]],
  ["door", ["door", "门"]],
];
const KIND_LABELS = {
  query_location: "位置查询",
  query_description: "外观查询",
  add_object: "新增物体",
  delete_object: "删除物体",
  update_properties: "更新属性",
  split_object: "拆分物体",
  merge_objects: "合并物体",
  reparent_object: "调整父子关系",
};

function isRecord(value) {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function detectLocale(text) {
  return /[\p{Script=Han}]/u.test(String(text)) ? "zh" : "en";
}

function localizedText(text, locale) {
  return { [locale]: String(text).trim() };
}

function cleanPhrase(value) {
  return String(value ?? "")
    .replace(/^[\s,，。.!！?？:：;；]+|[\s,，。.!！?？:：;；]+$/gu, "")
    .replace(/^(?:请|please)\s*/iu, "")
    .trim();
}

function cleanTargetPhrase(value) {
  return cleanPhrase(value)
    .replace(/^(?:把|将|the)\s*/iu, "")
    .replace(/(?:这个|该|the)\s*(?:物体|对象|object)$/iu, "")
    .replace(/(?:上面|上方|上|下面|下方|下|里面|内)$/u, "")
    .trim();
}

function inferCategory(value, fallback = "object") {
  const normalized = normalizeSceneQuery(value);
  for (const [category, terms] of CATEGORY_TERMS) {
    if (terms.some((term) => normalized.includes(normalizeSceneQuery(term)))) return category;
  }
  return fallback;
}

function proposalId(value, fallback = "object", ordinal = null) {
  const ascii = normalizeSceneQuery(value)
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 36);
  const base = ascii || inferCategory(value, fallback);
  return `new_${base}${ordinal == null ? "" : `_${ordinal}`}`;
}

function splitList(value) {
  return String(value ?? "")
    .split(/\s*(?:、|，|,|；|;|\+|&|\b(?:and|with)\b|和|与)\s*/iu)
    .map(cleanPhrase)
    .filter(Boolean);
}

function resolutionSummary(resolution) {
  return {
    status: resolution.status,
    entityId: resolution.entity?.id || null,
    entityName: resolution.entity?.name || resolution.entity?.label || null,
    candidates: (resolution.candidates || []).map((entity) => ({
      id: entity.id,
      name: entity.name || entity.label || entity.id,
      category: entity.category,
    })),
  };
}

function blockedFromResolution(kind, resolution, rawPrompt, details = {}) {
  const summary = resolutionSummary(resolution);
  const ambiguous = resolution.status === "ambiguous";
  return {
    kind,
    kindLabel: KIND_LABELS[kind],
    mutating: !kind.startsWith("query_"),
    status: ambiguous ? "blocked_ambiguous" : "blocked_not_found",
    previewOnly: true,
    rawPrompt,
    intent: null,
    targetIds: [],
    candidates: summary.candidates,
    affectedStages: [],
    riskLevel: "none",
    requiresConfirmation: false,
    summary: ambiguous
      ? `匹配到多个物体：${summary.candidates.map((item) => item.name).join("、")}。请使用名称、编号或物体 ID 明确指定。`
      : "没有找到可验证的目标物体，未生成可提交操作。",
    ...details,
  };
}

function invalidPreview(kind, rawPrompt, summary, details = {}) {
  return {
    kind,
    kindLabel: KIND_LABELS[kind],
    mutating: !kind.startsWith("query_"),
    status: "blocked_invalid",
    previewOnly: true,
    rawPrompt,
    intent: null,
    targetIds: [],
    candidates: [],
    affectedStages: [],
    riskLevel: "none",
    requiresConfirmation: false,
    summary,
    ...details,
  };
}

function readyMutation({ kind, rawPrompt, intent, targetIds, affectedStages, riskLevel, summary }) {
  return {
    kind,
    kindLabel: KIND_LABELS[kind],
    mutating: true,
    status: "ready",
    previewOnly: true,
    rawPrompt,
    intent,
    targetIds: [...new Set(targetIds)],
    candidates: [],
    affectedStages: affectedStages.map((stage) => ({
      stage,
      invalidatesPreviousOutput: true,
    })),
    riskLevel,
    requiresConfirmation: true,
    summary,
  };
}

function classifyMutation(text) {
  const normalized = normalizeSceneQuery(text);
  if (/\b(?:delete|remove)\b/u.test(normalized) || /(?:删除|移除|去掉)/u.test(text)) return "delete_object";
  if (/\b(?:reparent|parent|child of)\b/u.test(normalized) || /(?:父物体|父对象|子物体|子对象|挂到|归属)/u.test(text)) return "reparent_object";
  if (/\b(?:split|separate)\b/u.test(normalized) || /(?:拆分|拆成|分割成|分成)/u.test(text)) return "split_object";
  if (/\b(?:merge|combine)\b/u.test(normalized) || /(?:合并|组合成)/u.test(text)) return "merge_objects";
  if (/\b(?:add|create|insert)\b/u.test(normalized) || /(?:添加|新增|创建|加入)/u.test(text)) return "add_object";
  if (/\b(?:rename|update|set|change)\b/u.test(normalized) || /(?:改名|重命名|修改|更新|改变|调整|调高|调低|改为|设为|设置为)/u.test(text)) return "update_properties";
  return null;
}

function parseQuery(rawPrompt, index, context) {
  const questionKind = classifySceneQuestion(rawPrompt) === "location"
    ? "query_location"
    : "query_description";
  const resolution = resolveSceneEntity(rawPrompt, index, context);
  if (resolution.status !== "resolved") {
    return blockedFromResolution(questionKind, resolution, rawPrompt);
  }
  const entity = resolution.entity;
  return {
    kind: questionKind,
    kindLabel: KIND_LABELS[questionKind],
    mutating: false,
    status: "ready",
    previewOnly: false,
    rawPrompt,
    intent: {
      kind: questionKind,
      target: { object_id: entity.id },
      language: "auto",
    },
    targetIds: [entity.id],
    candidates: [],
    affectedStages: [],
    riskLevel: "none",
    requiresConfirmation: false,
    summary: `${entity.name || entity.label || entity.id}：${KIND_LABELS[questionKind]}`,
  };
}

function parseDelete(rawPrompt, index, context) {
  const cascadeDescendants = /(?:级联|连同.*子物体|包括.*子物体|及其.*子物体|cascade|with\s+(?:all\s+)?(?:children|descendants))/iu.test(rawPrompt);
  const commandText = rawPrompt.replace(
    /\s*(?:(?:并)?级联(?:删除)?|连同|包括|及其|with|and)\s*(?:所有|全部|all)?\s*(?:子物体|子对象|后代|children|descendants).*$/iu,
    "",
  );
  let target = commandText.match(/^(?:请)?(?:把|将)\s*(.+?)\s*(?:删除|移除|去掉)\s*[。.!！]?$/iu)?.[1];
  if (!target) {
    target = commandText.replace(/^(?:请\s*)?(?:delete|remove|删除|移除|去掉)\s*/iu, "");
  }
  target = cleanTargetPhrase(target);
  if (!target) return invalidPreview("delete_object", rawPrompt, "删除命令缺少目标物体。");
  const resolution = resolveSceneEntity(target, index, context);
  if (resolution.status !== "resolved") return blockedFromResolution("delete_object", resolution, rawPrompt);
  const entity = resolution.entity;
  return readyMutation({
    kind: "delete_object",
    rawPrompt,
    intent: {
      kind: "delete_object",
      target: { object_id: entity.id },
      cascade_descendants: cascadeDescendants,
      repair_exposed_background: true,
      retain_superseded_assets: true,
    },
    targetIds: [entity.id],
    affectedStages: DELETE_STAGES,
    riskLevel: "critical",
    summary: `预览删除 ${entity.name || entity.id}${cascadeDescendants ? " 及其后代" : ""}；对象只会被 tombstone，遮挡区域必须完成 clean-plate 修复后才能发布。`,
  });
}

function parseAdd(rawPrompt, index) {
  const locale = detectLocale(rawPrompt);
  let objectPhrase = "";
  let parentPhrase = "";
  if (locale === "zh") {
    const body = rawPrompt
      .replace(/^(?:请)?(?:在场景中)?\s*(?:添加|新增|创建|加入)\s*/u, "")
      .replace(/^(?:一个|一只|一张|一株|一盏|一把)\s*/u, "");
    const match = body.match(/^(.+?)(?:\s*(?:放到|放在|添加到|加入到|到|至)\s*)(.+)$/u);
    objectPhrase = cleanPhrase(match?.[1] || body);
    parentPhrase = cleanTargetPhrase(match?.[2] || "");
  } else {
    const body = rawPrompt.replace(/^(?:please\s+)?(?:add|create|insert)\s+(?:an?\s+)?/iu, "");
    const match = body.match(/^(.+?)(?:\s+(?:to|onto|under|inside)\s+)(.+)$/iu);
    objectPhrase = cleanPhrase(match?.[1] || body);
    parentPhrase = cleanTargetPhrase(match?.[2] || "");
  }
  if (!objectPhrase) return invalidPreview("add_object", rawPrompt, "新增命令缺少物体描述。");

  let parent = null;
  if (parentPhrase) {
    const resolution = resolveSceneEntity(parentPhrase, index);
    if (resolution.status !== "resolved") {
      return blockedFromResolution("add_object", resolution, rawPrompt, {
        summary: resolution.status === "ambiguous"
          ? `放置或父级目标“${parentPhrase}”不唯一，未生成新增操作。`
          : `没有找到放置或父级目标“${parentPhrase}”，未生成新增操作。`,
      });
    }
    parent = resolution.entity;
  }
  const category = inferCategory(objectPhrase);
  const proposedId = proposalId(objectPhrase, category);
  const object = {
    proposed_id: proposedId,
    name: localizedText(objectPhrase, locale),
    category,
    aliases: [],
    semantic_granularity: parent ? "independent_child_asset" : "independent_root_asset",
    independently_movable: true,
  };
  if (parent) object.parent = { object_id: parent.id };
  return readyMutation({
    kind: "add_object",
    rawPrompt,
    intent: {
      kind: "add_object",
      object,
      segmentation_prompt: objectPhrase,
      reference_asset_uris: [],
      placement_hint: parent ? localizedText(parentPhrase, locale) : null,
    },
    targetIds: [proposedId, ...(parent ? [parent.id] : [])],
    affectedStages: FULL_RECONSTRUCTION_STAGES,
    riskLevel: "high",
    summary: parent
      ? `预览新增“${objectPhrase}”并作为 ${parent.name || parent.id} 的独立子资产；需要分割、六视图补全、碰撞与放置审核。`
      : `预览新增独立物体“${objectPhrase}”；需要完整重建与 Web QA。`,
  });
}

function updateMatch(rawPrompt) {
  const zhProperty = rawPrompt.match(/^(?:请)?(?:把|将)\s*(.+?)(?:的)?\s*(名字|名称|类别|分类|颜色|材质|描述|外观)\s*(?:改为|修改为|更新为|设为|设置为)\s*(.+)$/u);
  if (zhProperty) return { target: zhProperty[1], property: zhProperty[2], value: zhProperty[3] };
  const zhRename = rawPrompt.match(/^(?:请)?(?:把|将)?\s*(.+?)\s*(?:改名为|重命名为)\s*(.+)$/u);
  if (zhRename) return { target: zhRename[1], property: "名字", value: zhRename[2] };
  const zhMovable = rawPrompt.match(/^(?:请)?(?:把|将)?\s*(.+?)\s*(?:设为|设置为)?\s*(不可移动|可移动)$/u);
  if (zhMovable) return { target: zhMovable[1], property: "移动", value: zhMovable[2] };
  const enRename = rawPrompt.match(/^(?:please\s+)?rename\s+(.+?)\s+(?:to|as)\s+(.+)$/iu);
  if (enRename) return { target: enRename[1], property: "name", value: enRename[2] };
  const enProperty = rawPrompt.match(/^(?:please\s+)?(?:set|update|change)\s+(.+?)['’]?s?\s+(name|category|color|material|description|appearance|movable)\s+(?:to|as)\s+(.+)$/iu);
  if (enProperty) return { target: enProperty[1], property: enProperty[2], value: enProperty[3] };
  return null;
}

function parseUpdate(rawPrompt, index, context) {
  const match = updateMatch(rawPrompt);
  if (!match) {
    return invalidPreview(
      "update_properties",
      rawPrompt,
      "无法安全识别要更新的属性。请明确写出名称、类别、颜色、材质、描述或是否可移动。",
    );
  }
  const locale = detectLocale(rawPrompt);
  const target = cleanTargetPhrase(match.target);
  const value = cleanPhrase(match.value);
  const resolution = resolveSceneEntity(target, index, context);
  if (resolution.status !== "resolved") return blockedFromResolution("update_properties", resolution, rawPrompt);
  if (!value) return invalidPreview("update_properties", rawPrompt, "属性更新缺少新值。");

  const property = normalizeSceneQuery(match.property);
  let patch;
  let metadataOnly = true;
  if (["名字", "名称", "name"].includes(property)) {
    patch = { name: localizedText(value, locale) };
  } else if (["类别", "分类", "category"].includes(property)) {
    patch = { category: value };
    metadataOnly = false;
  } else if (["颜色", "color", "材质", "material", "描述", "description", "外观", "appearance"].includes(property)) {
    const label = locale === "zh" && ["颜色", "材质"].includes(property) ? `${match.property}：${value}` : value;
    patch = { description: { appearance: localizedText(label, locale) } };
  } else if (["移动", "movable"].includes(property)) {
    const normalizedValue = normalizeSceneQuery(value);
    const movable = !/(?:不可|不能|false|off|no)/u.test(normalizedValue);
    patch = { independently_movable: movable };
    metadataOnly = false;
  } else {
    return invalidPreview("update_properties", rawPrompt, `不支持安全更新属性“${match.property}”。`);
  }
  const entity = resolution.entity;
  const stages = metadataOnly ? METADATA_STAGES : HIERARCHY_STAGES;
  return readyMutation({
    kind: "update_properties",
    rawPrompt,
    intent: {
      kind: "update_properties",
      target: { object_id: entity.id },
      patch,
    },
    targetIds: [entity.id],
    affectedStages: stages,
    riskLevel: metadataOnly ? "low" : "medium",
    summary: `预览更新 ${entity.name || entity.id} 的“${match.property}”属性；浏览器不会直接改写场景资产。`,
  });
}

function parseSplit(rawPrompt, index, context) {
  const match = rawPrompt.match(/^(?:请)?(?:把|将)?\s*(.+?)\s*(?:拆分为|拆分成|拆成|分割成|分成|split\s+into|separate\s+into)\s*(.+)$/iu)
    || rawPrompt.match(/^(?:please\s+)?(?:split|separate)\s+(.+?)\s+into\s+(.+)$/iu);
  if (!match) return invalidPreview("split_object", rawPrompt, "请使用“把 A 拆分为 B、C”或“split A into B and C”。");
  const target = cleanTargetPhrase(match[1]);
  const partNames = splitList(match[2]);
  if (partNames.length < 2) return invalidPreview("split_object", rawPrompt, "拆分命令至少需要两个明确的结果物体。");
  const resolution = resolveSceneEntity(target, index, context);
  if (resolution.status !== "resolved") return blockedFromResolution("split_object", resolution, rawPrompt);
  const locale = detectLocale(rawPrompt);
  const source = resolution.entity;
  const ids = new Set();
  const parts = partNames.map((name, indexAtPart) => {
    let proposedId = proposalId(name, source.category, indexAtPart + 1);
    while (ids.has(proposedId)) proposedId = `${proposedId}_${ids.size + 1}`;
    ids.add(proposedId);
    return {
      proposed_id: proposedId,
      name: localizedText(name, locale),
      category: inferCategory(name, source.category),
      aliases: [],
      semantic_granularity: "inherit",
    };
  });
  return readyMutation({
    kind: "split_object",
    rawPrompt,
    intent: {
      kind: "split_object",
      target: { object_id: source.id },
      parts,
      child_reassignment: {},
      retain_superseded_assets: true,
    },
    targetIds: [source.id, ...parts.map((part) => part.proposed_id)],
    affectedStages: FULL_RECONSTRUCTION_STAGES,
    riskLevel: "high",
    summary: `预览把 ${source.name || source.id} 拆成 ${partNames.join("、")}；后端还必须核对并重新分配原有子物体。`,
  });
}

function parseMerge(rawPrompt, index, context) {
  let match = rawPrompt.match(/^(?:请)?(?:把|将)?\s*(.+?)\s*(?:合并为|合并成|组合成)\s*(.+)$/u);
  if (!match) match = rawPrompt.match(/^(?:please\s+)?(?:merge|combine)\s+(.+?)\s+(?:into|as)\s+(.+)$/iu);
  if (!match) return invalidPreview("merge_objects", rawPrompt, "请使用“合并 A 和 B 为 C”或“merge A and B into C”。");
  const targetPhrases = splitList(match[1]);
  const resultName = cleanPhrase(match[2]);
  if (targetPhrases.length < 2 || !resultName) return invalidPreview("merge_objects", rawPrompt, "合并命令需要至少两个来源物体和一个结果名称。");

  const resolutions = targetPhrases.map((phrase) => resolveSceneEntity(cleanTargetPhrase(phrase), index, context));
  const unresolved = resolutions.find((resolution) => resolution.status !== "resolved");
  if (unresolved) return blockedFromResolution("merge_objects", unresolved, rawPrompt);
  const entities = resolutions.map((resolution) => resolution.entity);
  if (new Set(entities.map((entity) => entity.id)).size !== entities.length) {
    return invalidPreview("merge_objects", rawPrompt, "多个来源短语解析到了同一个物体，未生成合并操作。");
  }
  const locale = detectLocale(rawPrompt);
  const category = inferCategory(resultName, entities[0].category);
  const result = {
    proposed_id: proposalId(resultName, category),
    name: localizedText(resultName, locale),
    category,
    aliases: [],
    semantic_granularity: "inherit",
  };
  return readyMutation({
    kind: "merge_objects",
    rawPrompt,
    intent: {
      kind: "merge_objects",
      targets: entities.map((entity) => ({ object_id: entity.id })),
      result,
      preserve_descendants: true,
      retain_superseded_assets: true,
    },
    targetIds: [...entities.map((entity) => entity.id), result.proposed_id],
    affectedStages: FULL_RECONSTRUCTION_STAGES,
    riskLevel: "high",
    summary: `预览把 ${entities.map((entity) => entity.name || entity.id).join("、")} 合并为“${resultName}”；来源资产将保留以支持回滚。`,
  });
}

function parseReparent(rawPrompt, index, context) {
  let targetPhrase = "";
  let parentPhrase = "";
  let match = rawPrompt.match(/^(?:请)?(?:把|将)?\s*(.+?)\s*(?:设为|设置为|挂到|挂载到|归到|归属到)\s*(.+?)(?:的)?(?:子物体|子对象|下面|下)?\s*[。.!！]?$/u);
  if (match) {
    targetPhrase = match[1];
    parentPhrase = match[2];
  } else {
    match = rawPrompt.match(/^(?:please\s+)?reparent\s+(.+?)\s+(?:to|under)\s+(.+)$/iu)
      || rawPrompt.match(/^(?:please\s+)?make\s+(.+?)\s+(?:an?\s+)?child\s+of\s+(.+)$/iu);
    if (match) {
      targetPhrase = match[1];
      parentPhrase = match[2];
    }
  }
  if (!targetPhrase || !parentPhrase) return invalidPreview("reparent_object", rawPrompt, "请明确写出子物体和新的父物体。");
  const targetResolution = resolveSceneEntity(cleanTargetPhrase(targetPhrase), index, context);
  if (targetResolution.status !== "resolved") return blockedFromResolution("reparent_object", targetResolution, rawPrompt);

  const rootRequested = /^(?:root|scene root|no parent|无父级|场景根)$/iu.test(cleanTargetPhrase(parentPhrase));
  const parentResolution = rootRequested ? null : resolveSceneEntity(cleanTargetPhrase(parentPhrase), index, context);
  if (parentResolution && parentResolution.status !== "resolved") return blockedFromResolution("reparent_object", parentResolution, rawPrompt);
  const entity = targetResolution.entity;
  const parent = parentResolution?.entity || null;
  if (parent?.id === entity.id) return invalidPreview("reparent_object", rawPrompt, "物体不能成为自己的父物体。");
  return readyMutation({
    kind: "reparent_object",
    rawPrompt,
    intent: {
      kind: "reparent_object",
      target: { object_id: entity.id },
      new_parent: parent ? { object_id: parent.id } : null,
      preserve_world_transform: true,
    },
    targetIds: [entity.id, ...(parent ? [parent.id] : [])],
    affectedStages: HIERARCHY_STAGES,
    riskLevel: "medium",
    summary: parent
      ? `预览把 ${entity.name || entity.id} 设为 ${parent.name || parent.id} 的子物体，并保持世界变换不变。`
      : `预览把 ${entity.name || entity.id} 移到场景根级，并保持世界变换不变。`,
  });
}

export function parseSceneCommand(text, index, context = {}) {
  const rawPrompt = cleanPhrase(text);
  if (!rawPrompt) return invalidPreview("query_description", rawPrompt, "请输入场景问题或修改命令。");
  if (!index?.entities || typeof index.entities.get !== "function") {
    return invalidPreview("query_description", rawPrompt, "场景知识尚未就绪。");
  }
  const kind = classifyMutation(rawPrompt);
  if (kind === "add_object") return parseAdd(rawPrompt, index);
  if (kind === "delete_object") return parseDelete(rawPrompt, index, context);
  if (kind === "update_properties") return parseUpdate(rawPrompt, index, context);
  if (kind === "split_object") return parseSplit(rawPrompt, index, context);
  if (kind === "merge_objects") return parseMerge(rawPrompt, index, context);
  if (kind === "reparent_object") return parseReparent(rawPrompt, index, context);
  return parseQuery(rawPrompt, index, context);
}

export function buildSceneCommandEnvelope(preview, {
  requestId,
  requestedAt = new Date(),
  requester = "video2world-web-demo",
  contextReferences = [],
} = {}) {
  if (!preview?.mutating || preview.status !== "ready" || !isRecord(preview.intent)) {
    throw new Error("Only a ready mutating preview can be submitted");
  }
  if (!/^[A-Za-z0-9_.:-]+$/.test(String(requestId || ""))) {
    throw new Error("requestId must match the SceneCommand contract");
  }
  const timestamp = requestedAt instanceof Date ? requestedAt : new Date(requestedAt);
  if (Number.isNaN(timestamp.getTime())) throw new Error("requestedAt must be a valid timestamp");
  return {
    schema_version: "1.0.0",
    request_id: String(requestId),
    provenance: {
      requested_at: timestamp.toISOString(),
      requester: String(requester),
      raw_prompt: preview.rawPrompt,
      parser_provider: "rule_based",
      context_references: contextReferences.map(String),
    },
    intent: structuredClone(preview.intent),
  };
}

function validateSubmissionEnvelope(command) {
  const envelope = isRecord(command?.command) ? command.command : command;
  if (!isRecord(envelope) || envelope.schema_version !== "1.0.0" || !isRecord(envelope.intent)) {
    throw new Error("Invalid SceneCommand envelope");
  }
  if (envelope !== command) {
    const validExpectedHash = command.expectedManifestSha256 == null
      || /^[a-f0-9]{64}$/u.test(String(command.expectedManifestSha256));
    const validClientPreview = isRecord(command.clientPreview)
      && command.clientPreview.status === "ready"
      && command.clientPreview.previewOnly === true
      && Array.isArray(command.clientPreview.targetIds)
      && Array.isArray(command.clientPreview.affectedStages);
    const validConfirmation = command.confirmationPhrase == null
      || (typeof command.confirmationPhrase === "string" && command.confirmationPhrase.length > 0);
    if (
      command.rawPrompt !== envelope.provenance?.raw_prompt
      || JSON.stringify(command.structuredIntent) !== JSON.stringify(envelope.intent)
      || !validExpectedHash
      || !validClientPreview
      || !validConfirmation
    ) {
      throw new Error("Invalid scene command queue request");
    }
  }
  return envelope;
}

async function postSceneCommandJson(endpoint, payload, envelope, fetchImpl) {
  if (typeof endpoint !== "string" || !endpoint.trim()) throw new Error("Scene command endpoint is unavailable");
  const response = await fetchImpl(endpoint, {
    method: "POST",
    credentials: "same-origin",
    redirect: "error",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "Idempotency-Key": envelope.request_id,
    },
    body: JSON.stringify(payload),
  });
  if (!response?.ok) {
    throw new Error(`Scene command service rejected the request (${response?.status ?? "network"})`);
  }
  const contentType = response.headers?.get?.("content-type") || "";
  if (!contentType.toLowerCase().includes("application/json")) {
    throw new Error("Scene command service returned a non-JSON response");
  }
  return response.json();
}

export async function planSceneCommandEnvelope(endpoint, command, { fetchImpl = fetch } = {}) {
  const envelope = validateSubmissionEnvelope(command);
  const result = await postSceneCommandJson(endpoint, command, envelope, fetchImpl);
  const statuses = new Set([
    "ready",
    "blocked",
    "blocked_ambiguous",
    "blocked_not_found",
    "blocked_invariant",
    "blocked_stale_manifest",
  ]);
  const validPhrase = result?.confirmationRequired === false
    || (result?.confirmationRequired === true
      && typeof result.confirmationPhrase === "string"
      && result.confirmationPhrase.length > 0);
  if (
    !isRecord(result)
    || !statuses.has(result.status)
    || typeof result.confirmationRequired !== "boolean"
    || !validPhrase
    || !isRecord(result.plan)
    || !/^[a-f0-9]{64}$/u.test(String(result.manifestSha256 || ""))
  ) {
    throw new Error("Scene command service returned an invalid plan result");
  }
  return result;
}

export async function submitSceneCommandEnvelope(endpoint, command, { fetchImpl = fetch } = {}) {
  const envelope = validateSubmissionEnvelope(command);
  const result = await postSceneCommandJson(endpoint, command, envelope, fetchImpl);
  const statuses = new Set(["queued", "completed_read", "blocked", "duplicate"]);
  const validBlockedReason = result?.status !== "blocked"
    || (typeof result.reason === "string" && result.reason.length > 0);
  if (
    !isRecord(result)
    || !statuses.has(result.status)
    || !validBlockedReason
    || result.requestId !== envelope.request_id
    || typeof result.planId !== "string"
    || result.planId.length === 0
    || !/^[a-f0-9]{64}$/u.test(String(result.idempotencyKey || ""))
    || !/^[a-f0-9]{64}$/u.test(String(result.manifestSha256 || ""))
  ) {
    throw new Error("Scene command service returned an invalid result");
  }
  return result;
}

export {
  DELETE_STAGES,
  FULL_RECONSTRUCTION_STAGES,
  HIERARCHY_STAGES,
  KIND_LABELS,
  METADATA_STAGES,
};
