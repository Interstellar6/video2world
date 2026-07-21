import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, test } from "vitest";

import {
  buildFinalizedManifest,
  commitFinalizedManifest,
  parseArgs,
} from "../../scripts/finalize_bedroom4_strict_clean_scene.mjs";
import { BEDROOM4_STABLE_ALIAS_SHA256 } from
  "../../scripts/promote_bedroom4_strict_clean_scene.mjs";
import { sha256File } from "../../scripts/lib/strict-clean-scene-assets.mjs";

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));
const realWorld = path.join(repoRoot, "web", "public", "worlds", "bedroom4");
const realFinalizationReceiptPath = path.join(
  realWorld,
  "qa",
  "strict-clean-scene-finalization-receipt.json",
);
const localArtifactEvidenceAvailable = fs.existsSync(realFinalizationReceiptPath)
  && fs.existsSync(path.join(realWorld, "manifest.web-demo-baseline-stable.json"));
const realFinalizationReceipt = localArtifactEvidenceAvailable
  ? JSON.parse(fs.readFileSync(realFinalizationReceiptPath, "utf8"))
  : null;
const realPendingManifest = realFinalizationReceipt == null
  ? null
  : path.join(
    realWorld,
    "qa",
    "finalization",
    path.basename(realFinalizationReceipt.manifest.pendingBackupPath),
  );
const roots = [];
const ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE = "archived_current_demo_only";

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function makeFixture() {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "video2world-finalize-")));
  roots.push(root);
  const world = path.join(root, "bedroom4");
  fs.mkdirSync(path.join(world, "qa"), { recursive: true });
  const manifestPath = path.join(world, "manifest.json");
  const manifest = JSON.parse(fs.readFileSync(realPendingManifest, "utf8"));
  writeJson(manifestPath, manifest);
  fs.copyFileSync(
    path.join(realWorld, "manifest.web-demo-baseline-stable.json"),
    path.join(world, "manifest.web-demo-baseline-stable.json"),
  );
  const manifestSha = sha256File(manifestPath);
  const swapReceiptPath = path.join(world, "qa", "swap.json");
  writeJson(swapReceiptPath, {
    kind: "video2world.bedroom4_strict_clean_scene_swap_receipt",
    status: "materialized_pending_promoted_manifest_recheck",
    lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
    correctedFullPipeline: false,
    promotionAllowed: false,
    finalQaClaimed: false,
    promotedManifest: { path: manifestPath, sha256: manifestSha },
    stableAlias: { sha256: BEDROOM4_STABLE_ALIAS_SHA256 },
  });
  const stagedManifestPath = path.join(world, "manifest.promoted.json");
  const finalized = buildFinalizedManifest(manifest, {
    preFinalManifestSha256: manifestSha,
    canonicalPreflightQaReportSha256: "1".repeat(64),
    swapReceiptSha256: sha256File(swapReceiptPath),
  });
  writeJson(stagedManifestPath, finalized);
  const finalQaReport = path.join(world, "qa", "final-qa.json");
  writeJson(finalQaReport, { status: "passed_current_demo_only", automatedGate: "passed" });
  return {
    world,
    manifestPath,
    manifestSha,
    swapReceiptPath,
    stagedManifestPath,
    stagedManifestSha: sha256File(stagedManifestPath),
    finalQaReport,
    outputReceipt: path.join(world, "qa", "finalization.json"),
  };
}

describe.skipIf(!localArtifactEvidenceAvailable)("strict clean-scene finalization", () => {
  test("builds the final metadata without changing runtime assets or hierarchy", () => {
    const manifest = JSON.parse(fs.readFileSync(realPendingManifest, "utf8"));
    const finalized = buildFinalizedManifest(manifest, {
      preFinalManifestSha256: "a".repeat(64),
      canonicalPreflightQaReportSha256: "b".repeat(64),
      swapReceiptSha256: "c".repeat(64),
    });

    expect(finalized.candidateBuild).toMatchObject({
      status: "promoted_current_demo_only",
      lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
      correctedFullPipeline: false,
      canonicalLayeredCompletion: false,
      promotionAllowed: true,
      promotionBlockers: [],
      cleanScene: {
        lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
        correctedFullPipeline: false,
      },
      promotionSwap: {
        lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
        correctedFullPipeline: false,
        finalQaClaimed: true,
        promotionAllowed: true,
      },
    });
    expect(finalized.sourceWorld).toMatchObject({
      lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
      correctedFullPipeline: false,
    });
    expect(finalized.assets).toEqual(manifest.assets);
    expect(finalized.interactiveObjects).toEqual(manifest.interactiveObjects);
    expect(finalized.initialState).toEqual(manifest.initialState);
  });

  test("atomically replaces the pending canonical manifest with the exact QA-tested bytes", () => {
    const fixture = makeFixture();
    let validationCalls = 0;
    const result = commitFinalizedManifest({
      world: fixture.world,
      finalQaReport: fixture.finalQaReport,
      swapReceipt: fixture.swapReceiptPath,
      stagedManifest: fixture.stagedManifestPath,
      outputReceipt: fixture.outputReceipt,
      validateQa: ({ candidateManifestSha, candidateManifestPath }) => {
        validationCalls += 1;
        expect(candidateManifestSha).toBe(fixture.stagedManifestSha);
        expect(candidateManifestPath).toBe(fixture.stagedManifestPath);
      },
    });

    expect(validationCalls).toBe(1);
    expect(result.manifestSha256).toBe(fixture.stagedManifestSha);
    expect(sha256File(fixture.manifestPath)).toBe(fixture.stagedManifestSha);
    expect(fs.existsSync(fixture.stagedManifestPath)).toBe(false);
    expect(result.receipt).toMatchObject({
      status: "promoted_current_demo_only",
      lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
      correctedFullPipeline: false,
      promotionAllowed: true,
      finalQaClaimed: true,
      manifest: { exactQaTestedBytesMovedAtomically: true },
    });
    expect(sha256File(result.receipt.manifest.pendingBackupPath)).toBe(fixture.manifestSha);
  });

  test("rejects a staged manifest that drops the finalized promotion claim", () => {
    const fixture = makeFixture();
    const staged = JSON.parse(fs.readFileSync(fixture.stagedManifestPath, "utf8"));
    staged.candidateBuild.promotionAllowed = false;
    writeJson(fixture.stagedManifestPath, staged);

    expect(() => commitFinalizedManifest({
      world: fixture.world,
      finalQaReport: fixture.finalQaReport,
      swapReceipt: fixture.swapReceiptPath,
      stagedManifest: fixture.stagedManifestPath,
      outputReceipt: fixture.outputReceipt,
      validateQa: () => {},
    })).toThrow(/finalized promotion contract/u);
    expect(sha256File(fixture.manifestPath)).toBe(fixture.manifestSha);
  });

  test("rejects a staged manifest that claims corrected full-pipeline lineage", () => {
    const fixture = makeFixture();
    const staged = JSON.parse(fs.readFileSync(fixture.stagedManifestPath, "utf8"));
    staged.candidateBuild.lineageScope = "corrected_full_pipeline";
    staged.candidateBuild.correctedFullPipeline = true;
    staged.candidateBuild.canonicalLayeredCompletion = true;
    staged.candidateBuild.promotionSwap.lineageScope = "corrected_full_pipeline";
    staged.candidateBuild.promotionSwap.correctedFullPipeline = true;
    writeJson(fixture.stagedManifestPath, staged);

    expect(() => commitFinalizedManifest({
      world: fixture.world,
      finalQaReport: fixture.finalQaReport,
      swapReceipt: fixture.swapReceiptPath,
      stagedManifest: fixture.stagedManifestPath,
      outputReceipt: fixture.outputReceipt,
      validateQa: () => {},
    })).toThrow(/archived current-demo-only lineage/u);
    expect(sha256File(fixture.manifestPath)).toBe(fixture.manifestSha);
  });

  test("parses prepare and commit modes without accepting extra options", () => {
    expect(parseArgs([
      "--prepare",
      "--world", "world",
      "--preflight-qa-report", "preflight.json",
      "--swap-receipt", "swap.json",
      "--staged-manifest", "promoted.json",
    ])).toMatchObject({ mode: "prepare", world: "world" });
    expect(parseArgs([
      "--commit",
      "--world", "world",
      "--final-qa-report", "final.json",
      "--swap-receipt", "swap.json",
      "--staged-manifest", "promoted.json",
      "--output-receipt", "receipt.json",
    ])).toMatchObject({ mode: "commit", world: "world" });
    expect(() => parseArgs(["--prepare", "--world", "world"]))
      .toThrow(/Missing required option/u);
    expect(() => parseArgs(["--commit", "--unknown", "x"]))
      .toThrow(/Unsupported option/u);
  });
});
