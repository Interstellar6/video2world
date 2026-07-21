import { describe, expect, it } from "vitest";
import {
  REQUIRED_QA_CHECKS,
  failedRequiredQaChecks,
  normalizeOptions,
  parseArgs,
  reportBindingFailures,
} from "../../scripts/simplify_pbr_glb.mjs";

function passingReport() {
  const stats = {
    path: "/tmp/input.glb",
    sha256: "a".repeat(64),
    bytes: 1234,
    faces: 100,
    bounds: { min: [0, 0, 0], max: [1, 1, 1] },
    materials: { pbrMaterialCount: 1, textureImageCount: 1 },
    uvLayers: 1,
    normalLoops: 300,
    glbContract: { allPrimitivesHaveNormals: true },
    degenerateTriangles: 0,
    winding: { consistent: true },
  };
  const report = {
    status: "passed",
    qa: Object.fromEntries(REQUIRED_QA_CHECKS.map((check) => [check, true])),
    source: stats,
    output: { ...stats, path: "/tmp/output.glb", sha256: "b".repeat(64) },
  };
  report.assetBinding = {
    input: {
      path: report.source.path,
      sha256: report.source.sha256,
      bytes: report.source.bytes,
      faces: report.source.faces,
      bounds: structuredClone(report.source.bounds),
      pbrMaterials: 1,
      textureImages: 1,
      uvLayers: 1,
      normalLoops: 300,
      glbHasNormals: true,
      degenerateTriangles: 0,
      windingConsistent: true,
    },
    output: {
      path: report.output.path,
      sha256: report.output.sha256,
      bytes: report.output.bytes,
      faces: report.output.faces,
      bounds: structuredClone(report.output.bounds),
      pbrMaterials: 1,
      textureImages: 1,
      uvLayers: 1,
      normalLoops: 300,
      glbHasNormals: true,
      degenerateTriangles: 0,
      windingConsistent: true,
    },
  };
  return report;
}

describe("PBR GLB browser LOD CLI", () => {
  it("defaults to the 90k face budget and one-percent bounds tolerance", () => {
    const parsed = parseArgs([
      "--input", "source.glb",
      "--output", "browser-lod.glb",
      "--report", "browser-lod.qa.json",
    ]);
    const options = normalizeOptions(parsed, "/tmp/video2world-pbr-test");

    expect(options.targetFaces).toBe(90_000);
    expect(options.boundsToleranceFraction).toBe(0.01);
    expect(options.inputPath).toBe("/tmp/video2world-pbr-test/source.glb");
  });

  it("rejects destructive in-place output and invalid budgets", () => {
    expect(() => normalizeOptions({
      input: "same.glb",
      output: "same.glb",
      report: "qa.json",
    }, "/tmp")).toThrow("must differ");

    expect(() => normalizeOptions({
      input: "source.glb",
      output: "lod.glb",
      report: "qa.json",
      "target-faces": "0",
    }, "/tmp")).toThrow("positive integer");
  });

  it("requires geometry, PBR preservation, and post-export reload gates", () => {
    const report = passingReport();
    expect(failedRequiredQaChecks(report)).toEqual([]);

    report.qa.textureImagesPresent = false;
    report.qa.windingConsistent = false;
    expect(failedRequiredQaChecks(report)).toEqual([
      "windingConsistent",
      "textureImagesPresent",
    ]);
  });

  it("binds input and output hashes plus geometry and PBR evidence", () => {
    const report = passingReport();
    expect(reportBindingFailures(report)).toEqual([]);

    report.assetBinding.output.sha256 = "c".repeat(64);
    report.assetBinding.input.bounds.max[0] = 2;
    expect(reportBindingFailures(report)).toEqual([
      "input:bounds",
      "output:sha256",
    ]);
  });

  it("rejects unknown and duplicate command-line options", () => {
    expect(() => parseArgs(["--wat", "value"])).toThrow("Unknown option");
    expect(() => parseArgs([
      "--input", "one.glb",
      "--input", "two.glb",
    ])).toThrow("Duplicate option");
  });
});
