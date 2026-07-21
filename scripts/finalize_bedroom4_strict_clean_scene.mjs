#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import {
  BEDROOM4_STABLE_ALIAS_SHA256,
  validateBrowserQa,
} from "./promote_bedroom4_strict_clean_scene.mjs";
import { sha256, sha256File } from "./lib/strict-clean-scene-assets.mjs";
import { validateWebManifest } from "../web/web-manifest.js";

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), "..");
const CANONICAL_WORLD_NAME = "bedroom4";
const STABLE_ALIAS_FILE = "manifest.web-demo-baseline-stable.json";
const ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE = "archived_current_demo_only";

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function readJson(filePath, label) {
  let value;
  try {
    value = JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (error) {
    throw new Error(`${label}: cannot read JSON: ${error.message}`);
  }
  requireCondition(value && typeof value === "object" && !Array.isArray(value),
    `${label}: root must be an object`);
  return value;
}

function fsyncDirectory(directory) {
  const descriptor = fs.openSync(directory, "r");
  try {
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
}

function writeBytesDurable(filePath, bytes) {
  const parent = path.dirname(filePath);
  requireCondition(fs.existsSync(parent) && fs.statSync(parent).isDirectory(),
    `output parent is missing: ${parent}`);
  const temporary = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  let descriptor = null;
  try {
    descriptor = fs.openSync(temporary, "wx", 0o600);
    fs.writeFileSync(descriptor, bytes);
    fs.fsyncSync(descriptor);
    fs.closeSync(descriptor);
    descriptor = null;
    fs.renameSync(temporary, filePath);
    fsyncDirectory(parent);
  } finally {
    if (descriptor != null) fs.closeSync(descriptor);
    if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
  }
}

function writeJsonDurable(filePath, value) {
  writeBytesDurable(filePath, Buffer.from(`${JSON.stringify(value, null, 2)}\n`));
}

function isInside(parent, child) {
  const relative = path.relative(parent, child);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function resolveWorld(worldInput) {
  const world = fs.realpathSync(path.resolve(worldInput));
  requireCondition(fs.statSync(world).isDirectory(), "world must be a directory");
  requireCondition(path.basename(world) === CANONICAL_WORLD_NAME,
    `world basename must be ${CANONICAL_WORLD_NAME}`);
  return world;
}

function resolveInsideWorld(world, input, label, { allowMissing = false } = {}) {
  const absolute = path.resolve(input);
  requireCondition(isInside(world, absolute), `${label} must be inside canonical world`);
  if (!allowMissing) {
    requireCondition(fs.existsSync(absolute) && fs.statSync(absolute).isFile(),
      `${label} must be a file`);
  }
  return absolute;
}

function validateSwapReceipt(receipt, receiptPath, world, canonicalManifestSha) {
  requireCondition(
    receipt.kind === "video2world.bedroom4_strict_clean_scene_swap_receipt"
      && receipt.status === "materialized_pending_promoted_manifest_recheck",
    "swap receipt is not pending promoted-manifest recheck",
  );
  requireCondition(receipt.promotionAllowed === false && receipt.finalQaClaimed === false,
    "swap receipt prematurely claims final QA or promotion");
  requireCondition(
    receipt.lineageScope === ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE
      && receipt.correctedFullPipeline === false,
    "swap receipt lineage must remain archived current-demo-only",
  );
  requireCondition(receipt.promotedManifest?.sha256 === canonicalManifestSha,
    "swap receipt does not bind the current canonical manifest SHA-256");
  requireCondition(path.resolve(receipt.promotedManifest?.path || "") === path.join(world, "manifest.json"),
    "swap receipt promoted manifest path is invalid");
  requireCondition(receipt.stableAlias?.sha256 === BEDROOM4_STABLE_ALIAS_SHA256,
    "swap receipt stable alias SHA-256 is invalid");
  const stableAlias = path.join(world, STABLE_ALIAS_FILE);
  requireCondition(sha256File(stableAlias) === BEDROOM4_STABLE_ALIAS_SHA256,
    "canonical stable alias changed before finalization");
  return {
    path: receiptPath,
    sha256: sha256File(receiptPath),
  };
}

export function buildFinalizedManifest(manifest, evidence) {
  requireCondition(manifest?.candidateBuild?.status === "materialized_pending_promoted_manifest_recheck",
    "canonical manifest is not pending promoted-manifest recheck");
  requireCondition(manifest.candidateBuild.promotionAllowed === false,
    "canonical manifest already allows promotion");
  requireCondition(manifest.candidateBuild.promotionSwap?.finalQaClaimed === false,
    "canonical manifest already claims final QA");
  const finalized = structuredClone(manifest);
  finalized.candidateBuild.status = "promoted_current_demo_only";
  finalized.candidateBuild.promotionAllowed = true;
  finalized.candidateBuild.lineageScope = ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE;
  finalized.candidateBuild.correctedFullPipeline = false;
  finalized.candidateBuild.canonicalLayeredCompletion = false;
  finalized.candidateBuild.promotionBlockers = [];
  finalized.candidateBuild.cleanScene.promotionApproved = true;
  finalized.candidateBuild.cleanScene.lineageScope = ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE;
  finalized.candidateBuild.cleanScene.correctedFullPipeline = false;
  finalized.candidateBuild.unifiedPbrObjects.status =
    "browser_qa_passed_promoted_current_demo_only";
  finalized.candidateBuild.promotionSwap = {
    ...finalized.candidateBuild.promotionSwap,
    status: "promoted_current_demo_only",
    lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
    correctedFullPipeline: false,
    preFinalManifestSha256: evidence.preFinalManifestSha256,
    canonicalPreflightQaReportSha256: evidence.canonicalPreflightQaReportSha256,
    swapReceiptSha256: evidence.swapReceiptSha256,
    finalQaClaimed: true,
    finalQaEvidence: "external_finalization_receipt_bound_to_exact_manifest_bytes",
    promotionAllowed: true,
  };
  if (finalized.productionBuild?.strictCleanSceneCandidate) {
    finalized.productionBuild.strictCleanSceneCandidate.status = "promoted_current_demo_only";
    finalized.productionBuild.strictCleanSceneCandidate.promotionApproved = true;
  }
  finalized.sourceWorld = {
    ...(finalized.sourceWorld || {}),
    adoptionMode: "strict_clean_scene_promoted_current_demo_only",
    lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
    correctedFullPipeline: false,
  };
  validateWebManifest(finalized);
  return finalized;
}

function validateQaReport({
  reportPath,
  manifest,
  manifestPath,
  manifestSha,
  world,
  projectRoot,
  validateQa = validateBrowserQa,
}) {
  const report = readJson(reportPath, "browser QA report");
  validateQa({
    report,
    candidateManifest: manifest,
    candidateManifestPath: manifestPath,
    candidateManifestSha: manifestSha,
    candidateWorld: world,
    projectRoot,
  });
  return { report, sha256: sha256File(reportPath) };
}

export function prepareFinalizedManifest(options) {
  const world = resolveWorld(options.world);
  const canonicalManifestPath = path.join(world, "manifest.json");
  const preflightQaPath = resolveInsideWorld(world, options.preflightQaReport, "preflight QA report");
  const swapReceiptPath = resolveInsideWorld(world, options.swapReceipt, "swap receipt");
  const stagedManifestPath = resolveInsideWorld(world, options.stagedManifest, "staged manifest", {
    allowMissing: true,
  });
  requireCondition(stagedManifestPath !== canonicalManifestPath,
    "staged manifest must differ from canonical manifest");
  const manifest = readJson(canonicalManifestPath, "canonical manifest");
  validateWebManifest(manifest);
  const canonicalManifestSha = sha256File(canonicalManifestPath);
  const swapReceipt = readJson(swapReceiptPath, "swap receipt");
  const swapEvidence = validateSwapReceipt(
    swapReceipt,
    swapReceiptPath,
    world,
    canonicalManifestSha,
  );
  const qaEvidence = validateQaReport({
    reportPath: preflightQaPath,
    manifest,
    manifestPath: canonicalManifestPath,
    manifestSha: canonicalManifestSha,
    world,
    projectRoot: options.projectRoot || repoRoot,
    validateQa: options.validateQa,
  });
  const finalized = buildFinalizedManifest(manifest, {
    preFinalManifestSha256: canonicalManifestSha,
    canonicalPreflightQaReportSha256: qaEvidence.sha256,
    swapReceiptSha256: swapEvidence.sha256,
  });
  const bytes = Buffer.from(`${JSON.stringify(finalized, null, 2)}\n`);
  writeBytesDurable(stagedManifestPath, bytes);
  return {
    status: "staged_final_manifest_browser_qa_pending",
    stagedManifest: stagedManifestPath,
    stagedManifestSha256: sha256(bytes),
    preFinalManifestSha256: canonicalManifestSha,
    preflightQaReportSha256: qaEvidence.sha256,
    promotionAllowed: false,
  };
}

function acquireFinalizationLock(world) {
  const lockPath = path.join(path.dirname(world), `.${path.basename(world)}.strict-clean-finalization.lock.json`);
  const owner = {
    schemaVersion: 1,
    kind: "video2world.bedroom4_strict_clean_scene_finalization_lock",
    pid: process.pid,
    hostname: os.hostname(),
    createdAt: new Date().toISOString(),
    token: crypto.randomUUID(),
  };
  const descriptor = fs.openSync(lockPath, "wx", 0o600);
  try {
    fs.writeFileSync(descriptor, Buffer.from(`${JSON.stringify(owner, null, 2)}\n`));
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
  fsyncDirectory(path.dirname(lockPath));
  return { lockPath, owner };
}

function releaseFinalizationLock(lock) {
  if (!lock) return;
  const current = readJson(lock.lockPath, "finalization lock");
  requireCondition(current.token === lock.owner.token, "finalization lock owner changed");
  fs.unlinkSync(lock.lockPath);
  fsyncDirectory(path.dirname(lock.lockPath));
}

export function commitFinalizedManifest(options) {
  const world = resolveWorld(options.world);
  const canonicalManifestPath = path.join(world, "manifest.json");
  const finalQaPath = resolveInsideWorld(world, options.finalQaReport, "final QA report");
  const swapReceiptPath = resolveInsideWorld(world, options.swapReceipt, "swap receipt");
  const stagedManifestPath = resolveInsideWorld(world, options.stagedManifest, "staged manifest");
  const outputReceiptPath = resolveInsideWorld(world, options.outputReceipt, "output receipt", {
    allowMissing: true,
  });
  requireCondition(!fs.existsSync(outputReceiptPath), "output finalization receipt already exists");
  const canonicalManifest = readJson(canonicalManifestPath, "canonical manifest");
  const canonicalManifestSha = sha256File(canonicalManifestPath);
  const swapReceipt = readJson(swapReceiptPath, "swap receipt");
  const swapEvidence = validateSwapReceipt(
    swapReceipt,
    swapReceiptPath,
    world,
    canonicalManifestSha,
  );
  const stagedManifest = readJson(stagedManifestPath, "staged final manifest");
  validateWebManifest(stagedManifest);
  requireCondition(stagedManifest.candidateBuild?.status === "promoted_current_demo_only"
    && stagedManifest.candidateBuild.promotionAllowed === true
    && stagedManifest.candidateBuild.promotionSwap?.finalQaClaimed === true,
  "staged manifest does not carry the finalized promotion contract");
  requireCondition(
    stagedManifest.candidateBuild.lineageScope === ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE
      && stagedManifest.candidateBuild.correctedFullPipeline === false
      && stagedManifest.candidateBuild.canonicalLayeredCompletion === false
      && stagedManifest.candidateBuild.promotionSwap.lineageScope === ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE
      && stagedManifest.candidateBuild.promotionSwap.correctedFullPipeline === false,
    "staged manifest does not preserve archived current-demo-only lineage",
  );
  requireCondition(
    stagedManifest.candidateBuild.promotionSwap.preFinalManifestSha256 === canonicalManifestSha,
    "staged manifest does not bind the pending canonical manifest",
  );
  const stagedManifestSha = sha256File(stagedManifestPath);
  const finalQa = validateQaReport({
    reportPath: finalQaPath,
    manifest: stagedManifest,
    manifestPath: stagedManifestPath,
    manifestSha: stagedManifestSha,
    world,
    projectRoot: options.projectRoot || repoRoot,
    validateQa: options.validateQa,
  });
  const backupDirectory = path.join(world, "qa", "finalization");
  fs.mkdirSync(backupDirectory, { recursive: true });
  const pendingBackupPath = path.join(
    backupDirectory,
    `manifest.pending-${canonicalManifestSha.slice(0, 16)}.json`,
  );
  requireCondition(!fs.existsSync(pendingBackupPath), "pending manifest backup already exists");
  const lock = acquireFinalizationLock(world);
  let canonicalMoved = false;
  let stagedMoved = false;
  try {
    requireCondition(sha256File(canonicalManifestPath) === canonicalManifestSha,
      "canonical manifest changed before finalization commit");
    requireCondition(sha256File(stagedManifestPath) === stagedManifestSha,
      "staged manifest changed before finalization commit");
    fs.renameSync(canonicalManifestPath, pendingBackupPath);
    canonicalMoved = true;
    fs.renameSync(stagedManifestPath, canonicalManifestPath);
    stagedMoved = true;
    fsyncDirectory(world);
    fsyncDirectory(backupDirectory);
    requireCondition(sha256File(canonicalManifestPath) === stagedManifestSha,
      "final canonical manifest SHA-256 changed during atomic rename");
    const receipt = {
      schemaVersion: 1,
      kind: "video2world.bedroom4_strict_clean_scene_finalization_receipt",
      status: "promoted_current_demo_only",
      acceptanceScope: "current_demo_only",
      lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
      correctedFullPipeline: false,
      promotionAllowed: true,
      finalQaClaimed: true,
      finalizedAt: new Date().toISOString(),
      manifest: {
        path: canonicalManifestPath,
        sha256: stagedManifestSha,
        exactQaTestedBytesMovedAtomically: true,
        pendingBackupPath,
        pendingBackupSha256: canonicalManifestSha,
      },
      browserQa: {
        path: finalQaPath,
        sha256: finalQa.sha256,
        status: finalQa.report.status,
        automatedGate: finalQa.report.automatedGate,
        manifestPathBeforeAtomicRename: stagedManifestPath,
        manifestSha256: stagedManifestSha,
        desktopAndMobileValidated: true,
      },
      swapReceipt: swapEvidence,
      stableAlias: {
        path: path.join(world, STABLE_ALIAS_FILE),
        sha256: BEDROOM4_STABLE_ALIAS_SHA256,
        preserved: true,
      },
      transaction: {
        mode: "same_directory_two_rename_with_rollback",
        canonicalManifestMovedToBackup: true,
        stagedQaTestedManifestMovedToCanonical: true,
      },
    };
    writeJsonDurable(outputReceiptPath, receipt);
    return {
      receipt,
      receiptPath: outputReceiptPath,
      receiptSha256: sha256File(outputReceiptPath),
      manifestSha256: stagedManifestSha,
    };
  } catch (error) {
    if (stagedMoved && fs.existsSync(canonicalManifestPath)) {
      fs.renameSync(canonicalManifestPath, stagedManifestPath);
      stagedMoved = false;
    }
    if (canonicalMoved && fs.existsSync(pendingBackupPath)) {
      fs.renameSync(pendingBackupPath, canonicalManifestPath);
      canonicalMoved = false;
    }
    fsyncDirectory(world);
    throw error;
  } finally {
    releaseFinalizationLock(lock);
  }
}

export function parseArgs(argv) {
  requireCondition(argv[0] === "--prepare" || argv[0] === "--commit",
    "first option must be --prepare or --commit");
  const mode = argv[0].slice(2);
  const required = mode === "prepare"
    ? ["world", "preflight-qa-report", "swap-receipt", "staged-manifest"]
    : ["world", "final-qa-report", "swap-receipt", "staged-manifest", "output-receipt"];
  const options = {};
  for (let index = 1; index < argv.length; index += 2) {
    const token = argv[index];
    const value = argv[index + 1];
    requireCondition(token?.startsWith("--"), `Unexpected argument: ${token}`);
    const key = token.slice(2);
    requireCondition(required.includes(key), `Unsupported option --${key}`);
    requireCondition(options[key] == null, `Duplicate option --${key}`);
    requireCondition(value != null && !value.startsWith("--"), `Missing value for --${key}`);
    options[key] = value;
  }
  for (const key of required) requireCondition(options[key] != null, `Missing required option --${key}`);
  return { mode, ...options };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const result = args.mode === "prepare"
    ? prepareFinalizedManifest({
      world: args.world,
      preflightQaReport: args["preflight-qa-report"],
      swapReceipt: args["swap-receipt"],
      stagedManifest: args["staged-manifest"],
    })
    : commitFinalizedManifest({
      world: args.world,
      finalQaReport: args["final-qa-report"],
      swapReceipt: args["swap-receipt"],
      stagedManifest: args["staged-manifest"],
      outputReceipt: args["output-receipt"],
    });
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exitCode = 1;
  }
}
