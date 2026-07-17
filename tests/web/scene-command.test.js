import { describe, expect, it, vi } from "vitest";
import { buildSceneKnowledgeIndex } from "../../web/scene-query.js";
import {
  buildSceneCommandEnvelope,
  parseSceneCommand,
  planSceneCommandEnvelope,
  submitSceneCommandEnvelope,
} from "../../web/scene-command.js";

const manifest = {
  sceneKnowledge: {
    missingInstances: [],
    objects: [
      {
        id: "bed01",
        name: "Bed",
        category: "bed",
        aliases: ["床", "大床"],
        bbox: { min: [-2, 0, -3], max: [2, 1, 3] },
      },
      {
        id: "pillow01",
        name: "White pillow",
        category: "pillow",
        aliases: ["白色枕头", "左枕头"],
        bbox: { min: [-1, 1, -1], max: [0, 1.4, 0] },
      },
      {
        id: "plant01",
        name: "Left plant",
        category: "plant",
        aliases: ["左侧盆栽", "left plant"],
        bbox: { min: [-4, 0, 0], max: [-3, 2, 1] },
      },
      {
        id: "plant02",
        name: "Right plant",
        category: "plant",
        aliases: ["右侧盆栽", "right plant"],
        bbox: { min: [3, 0, 0], max: [4, 2, 1] },
      },
    ],
  },
  interactiveObjects: [],
};
const index = buildSceneKnowledgeIndex(manifest);

describe("scene command query routing", () => {
  it("classifies grounded location and description queries without mutation stages", () => {
    const location = parseSceneCommand("白色枕头在哪里？", index);
    const description = parseSceneCommand("白色枕头长什么样？", index);

    expect(location).toMatchObject({
      status: "ready",
      kind: "query_location",
      mutating: false,
      targetIds: ["pillow01"],
      affectedStages: [],
    });
    expect(description.kind).toBe("query_description");
  });

  it("fails closed when a query names an ambiguous category", () => {
    const preview = parseSceneCommand("植物在哪里？", index);
    expect(preview).toMatchObject({
      status: "blocked_ambiguous",
      kind: "query_location",
      intent: null,
    });
    expect(preview.candidates.map((item) => item.id)).toEqual(["plant01", "plant02"]);
  });
});

describe("mutating scene command parsing", () => {
  it("parses add as a full reconstruction with an explicit parent hierarchy", () => {
    const preview = parseSceneCommand("添加一个白色靠枕到床上", index);
    expect(preview).toMatchObject({
      status: "ready",
      kind: "add_object",
      mutating: true,
      riskLevel: "high",
      requiresConfirmation: true,
    });
    expect(preview.intent.object).toMatchObject({
      category: "pillow",
      semantic_granularity: "independent_child_asset",
      parent: { object_id: "bed01" },
    });
    expect(preview.affectedStages.map((item) => item.stage)).toEqual([
      "inventory",
      "sam3",
      "fusion",
      "cognition",
      "completion_plan",
      "layered_completion",
      "placement",
      "bundle",
      "web",
    ]);
  });

  it("resolves a unique delete target and blocks an ambiguous one", () => {
    const resolved = parseSceneCommand("删除左侧盆栽", index);
    expect(resolved.intent).toMatchObject({
      kind: "delete_object",
      target: { object_id: "plant01" },
      repair_exposed_background: true,
      retain_superseded_assets: true,
    });
    expect(resolved.riskLevel).toBe("critical");

    const cascade = parseSceneCommand("删除床及其所有子物体", index);
    expect(cascade.intent).toMatchObject({
      target: { object_id: "bed01" },
      cascade_descendants: true,
    });

    const ambiguous = parseSceneCommand("删除植物", index);
    expect(ambiguous.status).toBe("blocked_ambiguous");
    expect(ambiguous.intent).toBeNull();
    expect(ambiguous.affectedStages).toEqual([]);
  });

  it("parses metadata and movement property updates", () => {
    const rename = parseSceneCommand("把左侧盆栽的名称改为窗边绿植", index);
    expect(rename.intent).toEqual({
      kind: "update_properties",
      target: { object_id: "plant01" },
      patch: { name: { zh: "窗边绿植" } },
    });
    expect(rename.affectedStages.map((item) => item.stage)).toEqual(["cognition", "bundle", "web"]);

    const color = parseSceneCommand("把白色枕头的颜色改为米白色", index);
    expect(color.intent.patch).toEqual({
      description: { appearance: { zh: "颜色：米白色" } },
    });

    const movable = parseSceneCommand("把白色枕头设为不可移动", index);
    expect(movable.intent.patch).toEqual({ independently_movable: false });
    expect(movable.affectedStages.map((item) => item.stage)).toEqual([
      "cognition",
      "placement",
      "bundle",
      "web",
    ]);
  });

  it("parses split, merge, and reparent without mutating the index", () => {
    const split = parseSceneCommand("把白色枕头拆分为枕芯和枕套", index);
    expect(split.status).toBe("ready");
    expect(split.intent.parts).toHaveLength(2);
    expect(split.intent.target.object_id).toBe("pillow01");

    const merge = parseSceneCommand("把左侧盆栽和右侧盆栽合并为植物组合", index);
    expect(merge.status).toBe("ready");
    expect(merge.intent.targets).toEqual([
      { object_id: "plant01" },
      { object_id: "plant02" },
    ]);
    expect(merge.intent.result.name).toEqual({ zh: "植物组合" });

    const reparent = parseSceneCommand("把白色枕头设为床的子物体", index);
    expect(reparent.intent).toEqual({
      kind: "reparent_object",
      target: { object_id: "pillow01" },
      new_parent: { object_id: "bed01" },
      preserve_world_transform: true,
    });
    expect(index.entities.get("pillow01").id).toBe("pillow01");
  });

  it("does not guess unknown targets or unsupported property syntax", () => {
    expect(parseSceneCommand("删除不存在的沙发", index)).toMatchObject({
      status: "blocked_not_found",
      intent: null,
    });
    expect(parseSceneCommand("把白色枕头的硬度调高", index)).toMatchObject({
      status: "blocked_invalid",
      intent: null,
    });
  });
});

describe("SceneCommand service contract", () => {
  it("builds the Python SceneCommand envelope without browser-side mutation", () => {
    const preview = parseSceneCommand("删除左侧盆栽", index);
    const command = buildSceneCommandEnvelope(preview, {
      requestId: "web-test-001",
      requestedAt: new Date("2026-07-17T12:00:00+08:00"),
      contextReferences: ["manifest:fixture-v1"],
    });
    expect(command).toEqual({
      schema_version: "1.0.0",
      request_id: "web-test-001",
      provenance: {
        requested_at: "2026-07-17T04:00:00.000Z",
        requester: "video2world-web-demo",
        raw_prompt: "删除左侧盆栽",
        parser_provider: "rule_based",
        context_references: ["manifest:fixture-v1"],
      },
      intent: preview.intent,
    });
    expect(() => buildSceneCommandEnvelope(parseSceneCommand("植物在哪里？", index), {
      requestId: "bad-query",
    })).toThrow(/ready mutating preview/);
  });

  it("posts JSON with an idempotency key and validates the service response", async () => {
    const command = buildSceneCommandEnvelope(parseSceneCommand("删除左侧盆栽", index), {
      requestId: "web-test-002",
    });
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      status: 202,
      headers: { get: () => "application/json; charset=utf-8" },
      json: async () => ({
        status: "queued",
        requestId: "web-test-002",
        planId: "sceneplan_0123456789abcdef0123",
        idempotencyKey: "c".repeat(64),
        manifestSha256: "d".repeat(64),
      }),
    }));
    const result = await submitSceneCommandEnvelope("/api/scene-command", command, { fetchImpl });
    expect(result.status).toBe("queued");
    expect(fetchImpl).toHaveBeenCalledOnce();
    expect(fetchImpl.mock.calls[0][1]).toMatchObject({
      method: "POST",
      redirect: "error",
      headers: { "Idempotency-Key": "web-test-002" },
    });
    expect(JSON.parse(fetchImpl.mock.calls[0][1].body)).toEqual(command);
  });

  it("accepts a reproducible queue request with structured intent and manifest context", async () => {
    const preview = parseSceneCommand("删除左侧盆栽", index);
    const command = buildSceneCommandEnvelope(preview, { requestId: "web-test-queue-001" });
    const queueRequest = {
      command,
      structuredIntent: command.intent,
      rawPrompt: command.provenance.raw_prompt,
      expectedManifestSha256: "a".repeat(64),
      clientPreview: {
        status: "ready",
        previewOnly: true,
        manifestVersion: "fixture-v1",
        riskLevel: preview.riskLevel,
        summary: preview.summary,
        targetIds: preview.targetIds,
        affectedStages: preview.affectedStages.map((item) => item.stage),
      },
    };
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      headers: { get: () => "application/json" },
      json: async () => ({
        status: "duplicate",
        requestId: command.request_id,
        planId: "sceneplan_0123456789abcdef0123",
        idempotencyKey: "c".repeat(64),
        manifestSha256: "d".repeat(64),
      }),
    }));
    await expect(submitSceneCommandEnvelope("/api/scene-command", queueRequest, { fetchImpl }))
      .resolves.toMatchObject({ status: "duplicate" });
    expect(JSON.parse(fetchImpl.mock.calls[0][1].body).structuredIntent).toEqual(command.intent);
  });

  it("accepts only a server-bound authoritative plan and confirmation phrase", async () => {
    const preview = parseSceneCommand("删除左侧盆栽", index);
    const command = buildSceneCommandEnvelope(preview, { requestId: "web-test-plan-001" });
    const request = {
      command,
      structuredIntent: command.intent,
      rawPrompt: command.provenance.raw_prompt,
      expectedManifestSha256: null,
      clientPreview: {
        status: "ready",
        previewOnly: true,
        manifestVersion: "fixture-v1",
        targetIds: preview.targetIds,
        affectedStages: preview.affectedStages.map((item) => item.stage),
      },
    };
    const result = await planSceneCommandEnvelope("/v1/scene-commands/plan", request, {
      fetchImpl: async () => ({
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({
          status: "ready",
          manifestSha256: "b".repeat(64),
          confirmationRequired: true,
          confirmationPhrase: "confirm sceneplan_123",
          plan: { status: "ready", affected_stages: [] },
        }),
      }),
    });
    expect(result).toMatchObject({
      status: "ready",
      confirmationRequired: true,
      confirmationPhrase: "confirm sceneplan_123",
    });
  });

  it("preserves a manifest-bound blocked submit reason", async () => {
    const command = buildSceneCommandEnvelope(parseSceneCommand("删除左侧盆栽", index), {
      requestId: "web-test-blocked-001",
    });
    const blocked = {
      status: "blocked",
      requestId: command.request_id,
      planId: "sceneplan_0123456789abcdef0123",
      idempotencyKey: "c".repeat(64),
      manifestSha256: "d".repeat(64),
      reason: "expected_manifest_sha256 does not match the canonical manifest",
    };
    const response = await submitSceneCommandEnvelope("/api/scene-command", command, {
      fetchImpl: async () => ({
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => blocked,
      }),
    });
    expect(response).toEqual(blocked);

    await expect(submitSceneCommandEnvelope("/api/scene-command", command, {
      fetchImpl: async () => ({
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ ...blocked, reason: "" }),
      }),
    })).rejects.toThrow(/invalid result/);
  });

  it("rejects non-JSON or malformed backend responses", async () => {
    const command = buildSceneCommandEnvelope(parseSceneCommand("删除左侧盆栽", index), {
      requestId: "web-test-003",
    });
    await expect(submitSceneCommandEnvelope("/api/scene-command", command, {
      fetchImpl: async () => ({
        ok: true,
        headers: { get: () => "text/html" },
      }),
    })).rejects.toThrow(/non-JSON/);
  });
});
