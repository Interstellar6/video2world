import { describe, expect, it } from "vitest";
import {
  answerSceneQuestion,
  buildSceneKnowledgeIndex,
  classifySceneQuestion,
  normalizeSceneQuery,
  resolveSceneEntity,
} from "../../web/scene-query.js";

const manifest = {
  sceneKnowledge: {
    coordinateFrame: "visual_native",
    missingInstances: ["pillow", "枕头"],
    objects: [
      {
        id: "sam3_plant_01",
        name: "Plant 01",
        category: "plant",
        aliases: ["左侧盆栽", "left plant"],
        description: {
          short: "A reconstructed indoor plant.",
          appearance: "绿色阔叶植物，放在深色花盆中。",
        },
        bbox: {
          min: [4, -1, 20],
          max: [6, 0, 23],
          center: [5, -0.5, 21.5],
          extent: [2, 1, 3],
        },
      },
      {
        id: "sam3_plant_02",
        name: "Plant 02",
        category: "plant",
        aliases: ["右侧盆栽", "right plant"],
        description: { appearance: "一株较小的盆栽。" },
        bbox: {
          min: [-11, 0, 14],
          max: [-9, 1, 16],
        },
      },
      {
        id: "sam3_bed_01",
        name: "Bed",
        category: "bed",
        aliases: ["床"],
        open_vocab_labels: ["bedding", "pillow"],
        description: { appearance: "A low bed with light bedding." },
        bbox: { min: [-5, 0, 3], max: [4, 4, 12] },
      },
    ],
    relations: [
      {
        subject: "sam3_plant_01",
        predicate: "Near",
        object: "sam3_bed_01",
        verified: true,
        confidence: 0.9,
      },
    ],
  },
  interactiveObjects: [
    { id: "sam3_plant_01", label: "Plant 01", category: "plant" },
    { id: "sam3_plant_02", label: "Plant 02", category: "plant" },
  ],
};

describe("scene query normalization and intent", () => {
  it("normalizes Chinese and English punctuation", () => {
    expect(normalizeSceneQuery("  Plant-01，在哪里？ ")).toBe("plant 01 在哪里");
  });

  it("classifies location and appearance questions", () => {
    expect(classifySceneQuestion("植物在哪里？")).toBe("location");
    expect(classifySceneQuestion("What does the bed look like?")).toBe("appearance");
  });
});

describe("scene entity resolution", () => {
  const index = buildSceneKnowledgeIndex(manifest);

  it("resolves a Chinese alias to one instance", () => {
    const result = resolveSceneEntity("左侧盆栽长什么样？", index);
    expect(result.status).toBe("resolved");
    expect(result.entity.id).toBe("sam3_plant_01");
  });

  it("requires disambiguation for a category with multiple instances", () => {
    const result = resolveSceneEntity("植物在哪里？", index);
    expect(result.status).toBe("ambiguous");
    expect(result.candidates.map((entity) => entity.id)).toEqual([
      "sam3_plant_01",
      "sam3_plant_02",
    ]);
  });

  it("prefers a directional long alias while keeping the shared alias ambiguous", () => {
    const directionalIndex = buildSceneKnowledgeIndex({
      sceneKnowledge: {
        objects: [
          {
            id: "sam3_lamp_01",
            name: "左侧浅色陶瓷台灯",
            category: "lamp",
            aliases: ["台灯", "左侧台灯"],
          },
          {
            id: "sam3_lamp_02",
            name: "右侧浅色陶瓷台灯",
            category: "lamp",
            aliases: ["台灯", "右侧台灯"],
          },
        ],
      },
    });
    expect(resolveSceneEntity("左侧台灯在哪里？", directionalIndex)).toMatchObject({
      status: "resolved",
      entity: { id: "sam3_lamp_01" },
    });
    expect(resolveSceneEntity("台灯在哪里？", directionalIndex)).toMatchObject({
      status: "ambiguous",
    });
  });

  it("uses selected context only for a pronoun", () => {
    const result = resolveSceneEntity("它长什么样？", index, {
      selectedEntityId: "sam3_plant_02",
    });
    expect(result.status).toBe("resolved");
    expect(result.entity.id).toBe("sam3_plant_02");
  });

  it("fails closed for pillow even when a bed prompt contains that label", () => {
    const result = resolveSceneEntity("枕头在哪里？", index);
    expect(result.status).toBe("missing-instance");
    expect(result.entity).toBeUndefined();
  });

  it("allows a future independently recognized pillow instance", () => {
    const pillowIndex = buildSceneKnowledgeIndex({
      sceneKnowledge: {
        objects: [{
          id: "sam3_pillow_01",
          name: "Pillow",
          category: "pillow",
          bbox: { min: [0, 0, 0], max: [1, 0.3, 0.6] },
        }],
      },
    });
    const result = resolveSceneEntity("枕头在哪里？", pillowIndex);
    expect(result.status).toBe("resolved");
    expect(result.entity.id).toBe("sam3_pillow_01");
  });

  it("keeps reviewed bilingual evidence on an independently recognized pillow", () => {
    const pillowIndex = buildSceneKnowledgeIndex({
      sceneKnowledge: {
        missingInstances: [],
        objects: [{
          id: "sam3_pillow_01",
          name: "Pillow ensemble",
          category: "pillow",
          aliases: ["pillow", "枕头", "抱枕"],
          independentlyRecognized: true,
          bbox: {
            min: [-5.08, -2.1, 13.26],
            max: [2.17, 1.08, 17.34],
          },
          description: {
            appearance: {
              zh: "两只较大的浅绿色拼接枕头位于一只较小的浅色方枕后方。",
              en: "Two larger pale-green patchwork pillows sit behind one smaller light pillow.",
            },
            location: {
              zh: "三只枕头共同放置在床上。",
              en: "The three pillows rest together on the bed.",
            },
          },
        }],
      },
    });
    const location = answerSceneQuestion("枕头在哪里？", pillowIndex);
    const appearance = answerSceneQuestion("What does the pillow look like?", pillowIndex);
    expect(location).toMatchObject({ status: "resolved", focusEntityId: "sam3_pillow_01" });
    expect(location.answer).toContain("共同放置在床上");
    expect(appearance.answer).toContain("Two larger pale-green patchwork pillows");
  });
});

describe("scene answers", () => {
  const index = buildSceneKnowledgeIndex(manifest);

  it("returns grounded appearance text and a focus target", () => {
    const result = answerSceneQuestion("左侧盆栽长什么样？", index);
    expect(result.status).toBe("resolved");
    expect(result.answer).toContain("绿色阔叶植物");
    expect(result.focusEntityId).toBe("sam3_plant_01");
  });

  it("uses a verified relation for a location answer", () => {
    const result = answerSceneQuestion("左侧盆栽在哪里？", index);
    expect(result.answer).toContain("靠近 Bed");
    expect(result.focusEntityId).toBe("sam3_plant_01");
  });

  it("ignores an unverified low-confidence relation", () => {
    const unverified = structuredClone(manifest);
    unverified.sceneKnowledge.relations[0].verified = false;
    unverified.sceneKnowledge.relations[0].confidence = 0.01;
    const result = answerSceneQuestion("左侧盆栽在哪里？", buildSceneKnowledgeIndex(unverified));
    expect(result.answer).not.toContain("靠近 Bed");
  });

  it("uses an inverse predicate when the queried entity is the relation object", () => {
    const directional = structuredClone(manifest);
    directional.sceneKnowledge.relations = [{
      subject: "sam3_plant_01",
      predicate: "OnTopOf",
      object: "sam3_bed_01",
      verified: true,
      confidence: 0.9,
    }];
    const result = answerSceneQuestion("床在哪里？", buildSceneKnowledgeIndex(directional));
    expect(result.answer).toContain("位于其下方 Plant 01");
    expect(result.answer).not.toContain("位于其上方");
  });

  it("does not fabricate a pillow location or focus target", () => {
    const result = answerSceneQuestion("Where is the pillow?", index);
    expect(result.status).toBe("missing-instance");
    expect(result.answer).toContain("does not contain an independently recognized");
    expect(result.focusEntityId).toBeNull();
  });

  it("returns candidates without changing focus for an ambiguous class", () => {
    const result = answerSceneQuestion("Where is the plant?", index);
    expect(result.status).toBe("ambiguous");
    expect(result.candidates).toHaveLength(2);
    expect(result.focusEntityId).toBeNull();
  });
});
