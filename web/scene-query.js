const LOCATION_TERMS = [
  "where", "location", "locate", "position", "在哪", "哪里", "位置", "何处",
];
const APPEARANCE_TERMS = [
  "look like", "appearance", "describe", "looks", "长什么样", "什么样", "外观", "描述",
];
const CATEGORY_ALIASES = {
  bed: ["bed", "beds", "床"],
  ceiling: ["ceiling", "ceilings", "天花板"],
  door: ["door", "doors", "门"],
  floor: ["floor", "floors", "地板", "地面"],
  lamp: ["lamp", "lamps", "灯", "灯具"],
  nightstand: ["nightstand", "nightstands", "bedside table", "床头柜"],
  pillow: ["pillow", "pillows", "枕头", "靠枕"],
  plant: ["plant", "plants", "植物", "盆栽", "绿植"],
  table: ["table", "tables", "桌子", "桌"],
  wall: ["wall", "walls", "墙", "墙面"],
  window: ["window", "windows", "窗", "窗户"],
};

export function normalizeSceneQuery(value) {
  return String(value ?? "")
    .normalize("NFKC")
    .toLocaleLowerCase()
    .replace(/[\p{P}\p{S}\s]+/gu, " ")
    .trim();
}

function normalizedTerms(values) {
  return [...new Set((Array.isArray(values) ? values : [])
    .map(normalizeSceneQuery)
    .filter(Boolean))];
}

function descriptionRecord(value) {
  if (typeof value === "string") return { short: value, appearance: value };
  if (!value || typeof value !== "object") return {};
  return { ...value };
}

function localizedValue(value, locale) {
  if (typeof value === "string") return value.trim();
  if (!value || typeof value !== "object") return "";
  const preferred = value[locale] ?? value.en ?? value.zh;
  return typeof preferred === "string" ? preferred.trim() : "";
}

function bboxRecord(value, fallback = null) {
  const bbox = value && typeof value === "object" ? value : fallback;
  if (!bbox || typeof bbox !== "object") return null;
  const min = Array.isArray(bbox.min) ? bbox.min.map(Number) : null;
  const max = Array.isArray(bbox.max) ? bbox.max.map(Number) : null;
  if (!min || !max || min.length < 3 || max.length < 3) return null;
  const center = Array.isArray(bbox.center)
    ? bbox.center.map(Number)
    : min.map((valueAtAxis, axis) => (valueAtAxis + max[axis]) * 0.5);
  const extent = Array.isArray(bbox.extent)
    ? bbox.extent.map(Number)
    : min.map((valueAtAxis, axis) => max[axis] - valueAtAxis);
  if ([...min, ...max, ...center, ...extent].some((item) => !Number.isFinite(item))) return null;
  return {
    coordinateFrame: bbox.coordinateFrame || bbox.frame || "visual_native",
    min: min.slice(0, 3),
    max: max.slice(0, 3),
    center: center.slice(0, 3),
    extent: extent.slice(0, 3),
  };
}

function entityFromRecord(record, interactive = null) {
  const id = String(record?.id || record?.object_id || interactive?.id || "").trim();
  if (!id) return null;
  const name = String(record?.name || record?.label || interactive?.label || id);
  const category = String(record?.category || interactive?.category || "unknown");
  const aliases = normalizedTerms([
    ...(record?.aliases || []),
    ...(record?.openVocabularyLabels || record?.open_vocab_labels || []),
    ...(interactive?.aliases || []),
    ...(interactive?.openVocabularyLabels || interactive?.open_vocab_labels || []),
  ]);
  const sourceBounds = interactive?.sourceAnchor?.robustBounds;
  return {
    ...interactive,
    ...record,
    id,
    name,
    label: record?.label || interactive?.label || name,
    category,
    aliases,
    description: descriptionRecord(record?.description ?? interactive?.description),
    bbox: bboxRecord(record?.bbox, sourceBounds),
    interactiveObjectId: interactive?.id || record?.interactiveObjectId || null,
    independentlyRecognized: record?.independentlyRecognized !== false,
  };
}

export function buildSceneKnowledgeIndex(manifest = {}) {
  const interactive = Array.isArray(manifest.interactiveObjects) ? manifest.interactiveObjects : [];
  const knowledge = manifest.sceneKnowledge && typeof manifest.sceneKnowledge === "object"
    ? manifest.sceneKnowledge
    : {};
  const knowledgeObjects = Array.isArray(knowledge.objects) ? knowledge.objects : [];
  const interactiveById = new Map(interactive.map((item) => [String(item.id), item]));
  const records = new Map();

  for (const item of knowledgeObjects) {
    const id = String(item?.id || item?.object_id || "");
    const entity = entityFromRecord(item, interactiveById.get(id));
    if (entity) records.set(entity.id, entity);
  }
  for (const item of interactive) {
    if (records.has(String(item.id))) continue;
    const entity = entityFromRecord(item, item);
    if (entity) records.set(entity.id, entity);
  }

  const byTerm = new Map();
  const addTerm = (term, id, kind) => {
    const key = normalizeSceneQuery(term);
    if (!key) return;
    const matches = byTerm.get(key) || [];
    if (!matches.some((match) => match.id === id && match.kind === kind)) matches.push({ id, kind });
    byTerm.set(key, matches);
  };
  for (const entity of records.values()) {
    addTerm(entity.id, entity.id, "id");
    addTerm(entity.name, entity.id, "name");
    addTerm(entity.label, entity.id, "name");
    for (const alias of entity.aliases) addTerm(alias, entity.id, "alias");
    addTerm(entity.category, entity.id, "category");
    for (const alias of CATEGORY_ALIASES[normalizeSceneQuery(entity.category)] || []) {
      addTerm(alias, entity.id, "category");
    }
  }

  const hasIndependentPillow = [...records.values()].some((entity) => {
    if (!entity.independentlyRecognized) return false;
    return [entity.id, entity.name, entity.category]
      .map(normalizeSceneQuery)
      .some((term) => term === "pillow" || term === "枕头");
  });
  const missingSource = Array.isArray(knowledge.missingInstances)
    ? knowledge.missingInstances
    : (hasIndependentPillow ? [] : ["pillow", "枕头"]);
  const missingInstances = normalizedTerms(missingSource);
  return {
    coordinateFrame: knowledge.coordinateFrame || "visual_native",
    entities: records,
    byTerm,
    relations: Array.isArray(knowledge.relations) ? knowledge.relations : [],
    missingInstances,
  };
}

export function classifySceneQuestion(text) {
  const normalized = normalizeSceneQuery(text);
  if (APPEARANCE_TERMS.some((term) => normalized.includes(normalizeSceneQuery(term)))) return "appearance";
  if (LOCATION_TERMS.some((term) => normalized.includes(normalizeSceneQuery(term)))) return "location";
  return "summary";
}

function termBoundaryMatch(query, term) {
  if (!term) return false;
  if (/^[\p{Script=Han}]+$/u.test(term)) return query.includes(term);
  return (` ${query} `).includes(` ${term} `);
}

export function resolveSceneEntity(text, index, { selectedEntityId = null } = {}) {
  const query = normalizeSceneQuery(text);
  if (!query) return { status: "empty", query, candidates: [] };

  for (const missing of index.missingInstances || []) {
    if (termBoundaryMatch(query, missing)) {
      return {
        status: "missing-instance",
        query,
        term: missing,
        candidates: [],
      };
    }
  }

  const matches = [];
  const priorities = { id: 0, name: 1, alias: 2, category: 3 };
  for (const [term, termMatches] of index.byTerm.entries()) {
    if (!termBoundaryMatch(query, term)) continue;
    for (const match of termMatches) {
      matches.push({ ...match, term, priority: priorities[match.kind] ?? 9 });
    }
  }
  matches.sort((left, right) => left.priority - right.priority || right.term.length - left.term.length);
  const bestPriority = matches[0]?.priority;
  const bestPriorityMatches = matches.filter((match) => match.priority === bestPriority);
  const longestBestTerm = bestPriorityMatches[0]?.term.length;
  const bestIds = [...new Set(bestPriorityMatches
    .filter((match) => match.term.length === longestBestTerm)
    .map((match) => match.id))];
  if (bestIds.length === 1) {
    return { status: "resolved", query, entity: index.entities.get(bestIds[0]), candidates: [] };
  }
  if (bestIds.length > 1) {
    return { status: "ambiguous", query, candidates: bestIds.map((id) => index.entities.get(id)) };
  }

  const pronoun = ["it", "this object", "它", "这个", "该物体"].some((term) => query.includes(term));
  if (pronoun && selectedEntityId && index.entities.has(selectedEntityId)) {
    return { status: "resolved", query, entity: index.entities.get(selectedEntityId), candidates: [] };
  }
  return { status: "not-found", query, candidates: [] };
}

function relationText(entity, index, locale) {
  const relations = index.relations.filter((relation) => (
    relation?.verified === true
      && (relation.confidence == null || Number(relation.confidence) >= 0.5)
      && (relation.subject === entity.id || relation.object === entity.id)
  ));
  if (!relations.length) return "";
  const first = relations[0];
  const entityIsSubject = first.subject === entity.id;
  const otherId = entityIsSubject ? first.object : first.subject;
  const other = index.entities.get(String(otherId));
  const targetLabel = first.targetLabel || first.target_label;
  const otherName = (entityIsSubject ? localizedValue(targetLabel, locale) : null)
    || other?.name
    || otherId;
  const directPredicates = {
    OnTopOf: locale === "zh" ? "位于其上方" : "on top of",
    SupportedBy: locale === "zh" ? "由其支撑" : "supported by",
    Near: locale === "zh" ? "靠近" : "near",
  };
  const inversePredicates = {
    OnTopOf: locale === "zh" ? "位于其下方" : "under",
    SupportedBy: locale === "zh" ? "支撑" : "supports",
    Near: locale === "zh" ? "靠近" : "near",
  };
  const predicates = entityIsSubject ? directPredicates : inversePredicates;
  return `${predicates[first.predicate] || first.predicate} ${otherName}`;
}

function detectLocale(text) {
  return /[\p{Script=Han}]/u.test(String(text)) ? "zh" : "en";
}

export function answerSceneQuestion(text, index, context = {}) {
  const intent = classifySceneQuestion(text);
  const resolution = resolveSceneEntity(text, index, context);
  const locale = detectLocale(text);
  if (resolution.status === "missing-instance") {
    return {
      ...resolution,
      intent,
      answer: locale === "zh"
        ? `当前场景没有把“${resolution.term}”识别为独立实例，因此没有可验证的独立位置或边界框。`
        : `The scene does not contain an independently recognized “${resolution.term}” instance, so no verified location or bounding box is available.`,
      focusEntityId: null,
    };
  }
  if (resolution.status === "ambiguous") {
    const names = resolution.candidates.map((entity) => entity.name).join(locale === "zh" ? "、" : ", ");
    return {
      ...resolution,
      intent,
      answer: locale === "zh" ? `找到多个匹配实例：${names}。请选择一个。` : `Multiple instances match: ${names}. Choose one.`,
      focusEntityId: null,
    };
  }
  if (resolution.status !== "resolved") {
    return {
      ...resolution,
      intent,
      answer: locale === "zh" ? "当前场景知识中没有找到可验证的对应实例。" : "No verified matching instance exists in the current scene knowledge.",
      focusEntityId: null,
    };
  }

  const entity = resolution.entity;
  const description = entity.description || {};
  let answer;
  if (intent === "appearance") {
    const appearance = localizedValue(description.appearance, locale)
      || localizedValue(description.detailed, locale)
      || localizedValue(description.short, locale);
    const caveat = localizedValue(description.fidelity_caveat, locale);
    answer = appearance
      ? `${entity.name}: ${appearance}${caveat ? ` ${caveat}` : ""}`
      : (locale === "zh" ? `${entity.name} 已识别，但还没有经过验证的详细外观描述。` : `${entity.name} is recognized, but has no verified detailed appearance description.`);
  } else if (intent === "location") {
    const reviewedLocation = localizedValue(description.location, locale);
    const relation = relationText(entity, index, locale);
    if (reviewedLocation) answer = `${entity.name}: ${reviewedLocation}`;
    else if (relation) answer = locale === "zh" ? `${entity.name} ${relation}。` : `${entity.name} is ${relation}.`;
    else if (entity.bbox) {
      const center = entity.bbox.center.map((value) => Number(value.toFixed(2))).join(", ");
      answer = locale === "zh"
        ? `${entity.name} 的场景坐标中心为 [${center}]，已显示其边界框。`
        : `${entity.name} is centered at scene coordinates [${center}]; its bounding box is highlighted.`;
    } else answer = locale === "zh" ? `${entity.name} 已识别，但没有可验证的位置边界框。` : `${entity.name} is recognized but has no verified location bounding box.`;
  } else {
    answer = localizedValue(description.short, locale)
      || localizedValue(description.appearance, locale)
      || (locale === "zh" ? `${entity.name} 是场景中的 ${entity.category} 实例。` : `${entity.name} is a ${entity.category} instance in the scene.`);
  }
  return {
    ...resolution,
    intent,
    answer,
    focusEntityId: entity.bbox ? entity.id : null,
  };
}
