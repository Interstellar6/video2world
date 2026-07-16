#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const OBJECT_TARGET_FACES = {
  sam3_nightstand_01: 18_000,
  sam3_nightstand_02: 18_000,
  sam3_plant_01: 10_000,
  sam3_plant_02: 22_000,
};

const DEFAULT_PILLOW_VISUAL_CARVE_DISTANCE = 0.08;
const PILLOW_CARVE_SENSITIVITY_DISTANCES = [0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18];

const CATEGORY_METADATA = {
  nightstand: {
    aliases: ["nightstand", "bedside table", "床头柜"],
    description: {
      short: "A reconstructed bedside storage object.",
      appearance: "A generated nightstand reconstructed from the source instance evidence.",
      evidence: "source_anchor+trellis_generation",
    },
  },
  plant: {
    aliases: ["plant", "potted plant", "植物", "盆栽", "绿植"],
    description: {
      short: "A reconstructed indoor potted plant.",
      appearance: "A generated potted plant used where the scanned instance evidence is sparse.",
      evidence: "source_anchor+category_proxy_trellis_generation",
    },
  },
};

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for --${key}`);
    options[key] = value;
    index += 1;
  }
  return options;
}

function required(options, key) {
  if (!options[key]) throw new Error(`Missing required option --${key}`);
  return options[key];
}

function resolveExecutable(value) {
  if (path.isAbsolute(value) || value.includes(path.sep)) return path.resolve(value);
  for (const directory of String(process.env.PATH || "").split(path.delimiter)) {
    if (!directory) continue;
    const candidate = path.join(directory, value);
    try {
      fs.accessSync(candidate, fs.constants.X_OK);
      if (fs.statSync(candidate).isFile()) return path.resolve(candidate);
    } catch {
      // Continue through PATH until an executable file is found.
    }
  }
  throw new Error(`Executable not found on PATH: ${value}`);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function portableProvenance(value, { tempDir, worldDir, blenderPath }) {
  if (Array.isArray(value)) {
    return value.map((item) => portableProvenance(item, { tempDir, worldDir, blenderPath }));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [
      key,
      portableProvenance(item, { tempDir, worldDir, blenderPath }),
    ]));
  }
  if (typeof value !== "string" || !value.startsWith("/")) return value;
  if (value.startsWith("/worlds/") || value.startsWith("/test-fixtures/")) return value;
  if (value === blenderPath) return "tool://blender";
  for (const root of [tempDir, worldDir]) {
    if (value === root || value.startsWith(`${root}${path.sep}`)) {
      const relative = path.relative(root, value).split(path.sep).join("/");
      return `web-bundle://bedroom4/${relative}`;
    }
  }
  if (value === repoRoot || value.startsWith(`${repoRoot}${path.sep}`)) {
    const relative = path.relative(repoRoot, value).split(path.sep).join("/");
    return `repo://video2world/${relative}`;
  }
  const resultsMarker = `${path.sep}tmp_remote_results${path.sep}`;
  const resultsIndex = value.indexOf(resultsMarker);
  if (resultsIndex >= 0) {
    return `artifact://video2mesh/${value.slice(resultsIndex + resultsMarker.length).split(path.sep).join("/")}`;
  }
  const relumeowMarker = `${path.sep}interstellar6.github.io${path.sep}`;
  const relumeowIndex = value.indexOf(relumeowMarker);
  if (relumeowIndex >= 0) {
    return `repo://relumeow/${value.slice(relumeowIndex + relumeowMarker.length).split(path.sep).join("/")}`;
  }
  return `artifact://external/${path.basename(value)}`;
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function sha256File(filePath) {
  const hash = crypto.createHash("sha256");
  const descriptor = fs.openSync(filePath, "r");
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  try {
    let bytesRead = 0;
    do {
      bytesRead = fs.readSync(descriptor, buffer, 0, buffer.length, null);
      if (bytesRead) hash.update(buffer.subarray(0, bytesRead));
    } while (bytesRead);
  } finally {
    fs.closeSync(descriptor);
  }
  return hash.digest("hex");
}

function verifyFile(filePath, expectedSize, expectedSha, label) {
  if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
    throw new Error(`${label}: missing ${filePath}`);
  }
  const size = fs.statSync(filePath).size;
  if (expectedSize != null && Number.isFinite(Number(expectedSize)) && size !== Number(expectedSize)) {
    throw new Error(`${label}: expected ${expectedSize} bytes, got ${size}`);
  }
  const digest = sha256File(filePath);
  if (expectedSha && digest !== expectedSha) {
    throw new Error(`${label}: expected SHA ${expectedSha}, got ${digest}`);
  }
  return { size, sha256: digest };
}

function loadCognition(filePath) {
  const manifest = readJson(filePath);
  if (manifest.status !== "passed" || !Array.isArray(manifest.objects)) {
    throw new Error(`Cognition manifest has not passed: ${filePath}`);
  }
  const byId = new Map();
  for (const record of manifest.objects) {
    if (record.status !== "passed" || !record.object_id || !record.description_evidence) {
      throw new Error(`Invalid cognition record in ${filePath}: ${record.object_id || "unknown"}`);
    }
    byId.set(record.object_id, record);
  }
  return { manifest, byId };
}

function loadPillowEntity(filePath, cognitionById) {
  const record = readJson(filePath);
  if (record.id !== "sam3_pillow_01" || record.quality?.status !== "accepted_for_visual_focus_and_rotation") {
    throw new Error(`Pillow entity has not passed visual focus QA: ${filePath}`);
  }
  if (!finiteVector(record.aabb?.min) || !finiteVector(record.aabb?.max)) {
    throw new Error(`Pillow entity has invalid AABB: ${filePath}`);
  }
  const cognition = cognitionById.get(record.id);
  if (!cognition) throw new Error(`Cognition manifest is missing ${record.id}`);
  const visualPath = path.resolve(path.dirname(filePath), record.visual?.path || "");
  const visual = verifyFile(
    visualPath,
    null,
    record.visual?.sha256,
    `${record.id} accepted point cloud`
  );
  return {
    id: record.id,
    name: "枕头组合 / Pillow ensemble",
    category: record.category,
    aliases: record.aliases,
    description: cognition.description_evidence,
    bbox: {
      coordinateFrame: "visual_native",
      sourceCoordinateFrame: record.coordinate_frame,
      transform: "identity_holi_da3_to_pgsr_shared_frame",
      min: record.aabb.min,
      max: record.aabb.max,
      center: record.aabb.center,
      extent: record.aabb.extent,
    },
    independentlyRecognized: true,
    interactiveObjectId: record.id,
    focusOnly: false,
    visualOnly: true,
    visualEvidence: {
      type: record.visual.type,
      path: visualPath,
      pointCount: record.visual.point_count,
      size: visual.size,
      sha256: visual.sha256,
      projectionGate: record.quality,
    },
    limitations: record.limitations,
  };
}

function finiteVector(values, length = 3) {
  return Array.isArray(values)
    && values.length === length
    && values.every((value) => Number.isFinite(Number(value)));
}

function rounded(values, digits = 7) {
  return values.map((value) => Number(Number(value).toFixed(digits)));
}

function expandedBounds(bounds, margin) {
  return {
    min: bounds.min.map((value) => Number(value) - margin),
    max: bounds.max.map((value) => Number(value) + margin),
  };
}

function triangleIntersectsBounds(triangleMin, triangleMax, bounds) {
  return [0, 1, 2].every((axis) => (
    triangleMax[axis] >= bounds.min[axis] && triangleMin[axis] <= bounds.max[axis]
  ));
}

const PLY_SCALAR_TYPES = {
  char: { size: 1 }, int8: { size: 1 }, uchar: { size: 1 }, uint8: { size: 1 },
  short: { size: 2 }, int16: { size: 2 }, ushort: { size: 2 }, uint16: { size: 2 },
  int: { size: 4 }, int32: { size: 4 }, uint: { size: 4 }, uint32: { size: 4 },
  float: { size: 4 }, float32: { size: 4 }, double: { size: 8 }, float64: { size: 8 },
};

function readBinaryTriangleMeshPly(filePath) {
  const bytes = fs.readFileSync(filePath);
  const marker = Buffer.from("end_header");
  const markerIndex = bytes.indexOf(marker);
  if (markerIndex < 0) throw new Error(`${filePath}: missing end_header`);
  let dataOffset = markerIndex + marker.length;
  while (bytes[dataOffset] === 10 || bytes[dataOffset] === 13) dataOffset += 1;
  const header = bytes.subarray(0, dataOffset).toString("ascii");
  if (!header.includes("format binary_little_endian 1.0")) {
    throw new Error(`${filePath}: only binary_little_endian PLY is supported`);
  }

  const lines = header.split(/\r?\n/);
  let element = null;
  let vertexCount = 0;
  let faceCount = 0;
  let vertexStride = 0;
  const vertexProperties = new Map();
  let faceList = null;
  for (const line of lines) {
    const fields = line.trim().split(/\s+/);
    if (fields[0] === "element") {
      element = fields[1];
      if (element === "vertex") vertexCount = Number(fields[2]);
      if (element === "face") faceCount = Number(fields[2]);
      continue;
    }
    if (fields[0] !== "property") continue;
    if (element === "vertex") {
      if (fields[1] === "list") throw new Error(`${filePath}: vertex list properties are unsupported`);
      const type = PLY_SCALAR_TYPES[fields[1]];
      if (!type) throw new Error(`${filePath}: unsupported vertex type ${fields[1]}`);
      vertexProperties.set(fields[2], { type: fields[1], offset: vertexStride });
      vertexStride += type.size;
    } else if (element === "face") {
      if (fields[1] !== "list") throw new Error(`${filePath}: scalar face properties are unsupported`);
      faceList = { countType: fields[2], indexType: fields[3], name: fields[4] };
    }
  }
  if (!vertexCount || !faceCount || !faceList) throw new Error(`${filePath}: missing vertex/face declarations`);
  for (const axis of ["x", "y", "z"]) {
    if (vertexProperties.get(axis)?.type !== "double") {
      throw new Error(`${filePath}: expected double ${axis}`);
    }
  }
  if (!(["uchar", "uint8"].includes(faceList.countType) && ["uint", "uint32"].includes(faceList.indexType))) {
    throw new Error(`${filePath}: expected property list uchar uint vertex_indices`);
  }

  const vertexPayloadBytes = vertexCount * vertexStride;
  const faceOffset = dataOffset + vertexPayloadBytes;
  const expectedBytes = faceOffset + faceCount * 13;
  if (expectedBytes !== bytes.length) {
    throw new Error(`${filePath}: expected fixed triangle records (${expectedBytes}), got ${bytes.length}`);
  }
  const positions = new Float64Array(vertexCount * 3);
  const axisOffsets = ["x", "y", "z"].map((axis) => vertexProperties.get(axis).offset);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const offset = dataOffset + vertex * vertexStride;
    positions[vertex * 3] = bytes.readDoubleLE(offset + axisOffsets[0]);
    positions[vertex * 3 + 1] = bytes.readDoubleLE(offset + axisOffsets[1]);
    positions[vertex * 3 + 2] = bytes.readDoubleLE(offset + axisOffsets[2]);
  }
  return {
    filePath,
    bytes,
    header,
    dataOffset,
    faceOffset,
    vertexPayloadBytes,
    vertexCount,
    faceCount,
    positions,
  };
}

function carveTriangleMesh(mesh, objectBounds) {
  const keep = new Uint8Array(mesh.faceCount);
  const removedByObject = Object.fromEntries(objectBounds.map((item) => [item.id, 0]));
  let keptFaces = 0;
  const triangleMin = [0, 0, 0];
  const triangleMax = [0, 0, 0];
  for (let face = 0; face < mesh.faceCount; face += 1) {
    const offset = mesh.faceOffset + face * 13;
    if (mesh.bytes[offset] !== 3) throw new Error(`${mesh.filePath}: non-triangle face ${face}`);
    const indices = [
      mesh.bytes.readUInt32LE(offset + 1),
      mesh.bytes.readUInt32LE(offset + 5),
      mesh.bytes.readUInt32LE(offset + 9),
    ];
    for (let axis = 0; axis < 3; axis += 1) {
      const values = indices.map((index) => mesh.positions[index * 3 + axis]);
      triangleMin[axis] = Math.min(...values);
      triangleMax[axis] = Math.max(...values);
    }
    const matched = objectBounds.find((item) => triangleIntersectsBounds(triangleMin, triangleMax, item.bounds));
    if (matched) {
      removedByObject[matched.id] += 1;
    } else {
      keep[face] = 1;
      keptFaces += 1;
    }
  }
  for (const [objectId, count] of Object.entries(removedByObject)) {
    if (count <= 0) throw new Error(`${objectId}: collider carve removed zero faces`);
  }
  return {
    keep,
    keptFaces,
    removedFaces: mesh.faceCount - keptFaces,
    removedByObject,
    retainedIntersectingFaces: 0,
  };
}

function writeCarvedTriangleMesh(mesh, carve, outputPath) {
  const header = mesh.header.replace(
    `element face ${mesh.faceCount}`,
    `element face ${carve.keptFaces}`
  );
  if (header === mesh.header) throw new Error("Failed to update collider face count in PLY header");
  const headerBytes = Buffer.from(header, "ascii");
  const faceBytes = Buffer.allocUnsafe(carve.keptFaces * 13);
  let cursor = 0;
  for (let face = 0; face < mesh.faceCount; face += 1) {
    if (!carve.keep[face]) continue;
    mesh.bytes.copy(faceBytes, cursor, mesh.faceOffset + face * 13, mesh.faceOffset + (face + 1) * 13);
    cursor += 13;
  }
  const output = Buffer.concat([
    headerBytes,
    mesh.bytes.subarray(mesh.dataOffset, mesh.faceOffset),
    faceBytes,
  ]);
  fs.writeFileSync(outputPath, output);
  return output;
}

function copyAndRewriteChunkedAsset(asset, sourceChunkDir, targetChunkDir, urlPrefix) {
  const loaded = readVerifiedChunkedAsset(asset, sourceChunkDir);
  const parts = loaded.chunks.map(({ part, name, source, verified }) => {
    const target = path.join(targetChunkDir, name);
    fs.copyFileSync(source, target);
    return {
      ...part,
      url: `${urlPrefix}/chunks/${name}`,
      size: verified.size,
      sha256: verified.sha256,
    };
  });
  return { ...asset, parts };
}

function readVerifiedChunkedAsset(asset, sourceChunkDir) {
  if (!Array.isArray(asset.parts) || !asset.parts.length) throw new Error(`${asset.id}: missing chunk parts`);
  const wholeHash = crypto.createHash("sha256");
  let wholeSize = 0;
  const buffers = [];
  const chunks = asset.parts.map((part) => {
    const name = path.basename(part.url);
    const source = path.join(sourceChunkDir, name);
    const verified = verifyFile(source, part.size, part.sha256, `${asset.id}/${name}`);
    const bytes = fs.readFileSync(source);
    buffers.push(bytes);
    wholeHash.update(bytes);
    wholeSize += bytes.length;
    return { part, name, source, verified };
  });
  const digest = wholeHash.digest("hex");
  if (wholeSize !== Number(asset.size) || digest !== asset.sha256) {
    throw new Error(`${asset.id}: whole asset verification failed (${wholeSize}, ${digest})`);
  }
  return { bytes: Buffer.concat(buffers, wholeSize), chunks, size: wholeSize, sha256: digest };
}

function chunkBuffer(bytes, prefix, targetChunkDir, urlPrefix, chunkSize) {
  const parts = [];
  for (let offset = 0, index = 0; offset < bytes.length; offset += chunkSize, index += 1) {
    const chunk = bytes.subarray(offset, Math.min(bytes.length, offset + chunkSize));
    const name = `${prefix}.chunk${String(index).padStart(3, "0")}`;
    fs.writeFileSync(path.join(targetChunkDir, name), chunk);
    parts.push({
      url: `${urlPrefix}/chunks/${name}`,
      size: chunk.length,
      sha256: sha256(chunk),
    });
  }
  return parts;
}

const BINARY_PLY_READERS = {
  char: "readInt8", int8: "readInt8", uchar: "readUInt8", uint8: "readUInt8",
  short: "readInt16LE", int16: "readInt16LE", ushort: "readUInt16LE", uint16: "readUInt16LE",
  int: "readInt32LE", int32: "readInt32LE", uint: "readUInt32LE", uint32: "readUInt32LE",
  float: "readFloatLE", float32: "readFloatLE", double: "readDoubleLE", float64: "readDoubleLE",
};

function readBinaryVertexPly(bytes, label) {
  const marker = Buffer.from("end_header");
  const markerIndex = bytes.indexOf(marker);
  if (markerIndex < 0) throw new Error(`${label}: missing end_header`);
  let dataOffset = markerIndex + marker.length;
  while (bytes[dataOffset] === 10 || bytes[dataOffset] === 13) dataOffset += 1;
  const header = bytes.subarray(0, dataOffset).toString("ascii");
  if (!header.includes("format binary_little_endian 1.0")) {
    throw new Error(`${label}: expected binary_little_endian PLY`);
  }
  let element = null;
  let vertexCount = 0;
  let vertexStride = 0;
  const properties = new Map();
  for (const line of header.split(/\r?\n/)) {
    const fields = line.trim().split(/\s+/);
    if (fields[0] === "element") {
      element = fields[1];
      if (element === "vertex") vertexCount = Number(fields[2]);
      else if (Number(fields[2]) > 0) throw new Error(`${label}: expected a vertex-only PLY`);
      continue;
    }
    if (fields[0] !== "property" || element !== "vertex") continue;
    if (fields[1] === "list") throw new Error(`${label}: vertex list properties are unsupported`);
    const scalar = PLY_SCALAR_TYPES[fields[1]];
    if (!scalar || !BINARY_PLY_READERS[fields[1]]) {
      throw new Error(`${label}: unsupported vertex type ${fields[1]}`);
    }
    properties.set(fields[2], {
      type: fields[1],
      offset: vertexStride,
      size: scalar.size,
      reader: BINARY_PLY_READERS[fields[1]],
    });
    vertexStride += scalar.size;
  }
  if (!(vertexCount > 0)) throw new Error(`${label}: missing vertex count`);
  for (const axis of ["x", "y", "z"]) {
    if (!properties.has(axis)) throw new Error(`${label}: missing ${axis}`);
  }
  if (dataOffset + vertexCount * vertexStride !== bytes.length) {
    throw new Error(`${label}: vertex payload length does not match header`);
  }
  return { bytes, label, header, dataOffset, vertexCount, vertexStride, properties };
}

function readVertexPosition(ply, index, target = [0, 0, 0]) {
  const base = ply.dataOffset + index * ply.vertexStride;
  for (const [axis, name] of ["x", "y", "z"].entries()) {
    const property = ply.properties.get(name);
    target[axis] = ply.bytes[property.reader](base + property.offset);
  }
  return target;
}

function readAsciiPointAnchor(filePath) {
  const bytes = fs.readFileSync(filePath);
  const text = bytes.toString("utf8");
  const lines = text.split(/\r?\n/);
  let vertexCount = 0;
  let dataStart = -1;
  let inVertex = false;
  const properties = [];
  for (let index = 0; index < lines.length; index += 1) {
    const fields = lines[index].trim().split(/\s+/);
    if (fields[0] === "format" && fields[1] !== "ascii") {
      throw new Error(`${filePath}: expected ASCII point PLY`);
    }
    if (fields[0] === "element") {
      inVertex = fields[1] === "vertex";
      if (inVertex) vertexCount = Number(fields[2]);
    } else if (fields[0] === "property" && inVertex) {
      properties.push(fields.at(-1));
    } else if (fields[0] === "end_header") {
      dataStart = index + 1;
      break;
    }
  }
  if (!(vertexCount > 0) || dataStart < 0) throw new Error(`${filePath}: invalid point PLY`);
  const xyz = ["x", "y", "z"].map((axis) => properties.indexOf(axis));
  const rgb = ["red", "green", "blue"].map((channel) => properties.indexOf(channel));
  if (xyz.some((index) => index < 0) || rgb.some((index) => index < 0)) {
    throw new Error(`${filePath}: expected xyz + RGB point properties`);
  }
  const positions = new Float64Array(vertexCount * 3);
  for (let index = 0; index < vertexCount; index += 1) {
    const fields = lines[dataStart + index]?.trim().split(/\s+/) || [];
    for (let axis = 0; axis < 3; axis += 1) {
      const value = Number(fields[xyz[axis]]);
      if (!Number.isFinite(value)) throw new Error(`${filePath}: invalid point ${index}`);
      positions[index * 3 + axis] = value;
    }
  }
  return { bytes, vertexCount, positions, sha256: sha256(bytes) };
}

function buildPointSpatialHash(anchor, cellSize) {
  const cells = new Map();
  for (let index = 0; index < anchor.vertexCount; index += 1) {
    const base = index * 3;
    const key = [0, 1, 2]
      .map((axis) => Math.floor(anchor.positions[base + axis] / cellSize))
      .join(",");
    let values = cells.get(key);
    if (!values) {
      values = [];
      cells.set(key, values);
    }
    values.push(base);
  }
  return { ...anchor, cellSize, cells };
}

function nearestAnchorDistanceSquared(anchor, point) {
  const cell = point.map((value) => Math.floor(value / anchor.cellSize));
  let best = Infinity;
  for (let dx = -1; dx <= 1; dx += 1) {
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dz = -1; dz <= 1; dz += 1) {
        const candidates = anchor.cells.get(`${cell[0] + dx},${cell[1] + dy},${cell[2] + dz}`);
        if (!candidates) continue;
        for (const base of candidates) {
          const x = anchor.positions[base] - point[0];
          const y = anchor.positions[base + 1] - point[1];
          const z = anchor.positions[base + 2] - point[2];
          best = Math.min(best, x * x + y * y + z * z);
        }
      }
    }
  }
  return best;
}

function insideExactBounds(point, bounds) {
  return point.every((value, axis) => value >= bounds.min[axis] && value <= bounds.max[axis]);
}

function carvePillowFromStaticVisual({
  asset,
  sourceChunkDir,
  targetChunkDir,
  urlPrefix,
  chunkSize,
  pillow,
  distance,
}) {
  const loaded = readVerifiedChunkedAsset(asset, sourceChunkDir);
  const scene = readBinaryVertexPly(loaded.bytes, `${asset.id}:pillow-carve-input`);
  const anchor = readAsciiPointAnchor(pillow.visualEvidence.path);
  if (anchor.sha256 !== pillow.visualEvidence.sha256 || anchor.vertexCount !== pillow.visualEvidence.pointCount) {
    throw new Error(`${pillow.id}: accepted point anchor metadata mismatch`);
  }
  let anchorInsideAcceptedAabbCount = 0;
  const anchorPoint = [0, 0, 0];
  const anchorRawMin = [Infinity, Infinity, Infinity];
  const anchorRawMax = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < anchor.vertexCount; index += 1) {
    const base = index * 3;
    anchorPoint[0] = anchor.positions[base];
    anchorPoint[1] = anchor.positions[base + 1];
    anchorPoint[2] = anchor.positions[base + 2];
    for (let axis = 0; axis < 3; axis += 1) {
      anchorRawMin[axis] = Math.min(anchorRawMin[axis], anchorPoint[axis]);
      anchorRawMax[axis] = Math.max(anchorRawMax[axis], anchorPoint[axis]);
    }
    if (insideExactBounds(anchorPoint, pillow.bbox)) anchorInsideAcceptedAabbCount += 1;
  }
  const anchorInsideAcceptedAabbRatio = anchorInsideAcceptedAabbCount / anchor.vertexCount;
  if (anchorInsideAcceptedAabbRatio < 0.97) {
    throw new Error(`${pillow.id}: only ${anchorInsideAcceptedAabbRatio} of accepted points lie in the accepted AABB`);
  }
  const sensitivityDistances = Array.from(new Set([
    ...PILLOW_CARVE_SENSITIVITY_DISTANCES,
    distance,
  ])).sort((left, right) => left - right);
  const searchDistance = Math.max(...sensitivityDistances);
  const spatial = buildPointSpatialHash(anchor, searchDistance);
  const keep = new Uint8Array(scene.vertexCount);
  const sensitivity = Object.fromEntries(sensitivityDistances.map((value) => [String(value), 0]));
  const point = [0, 0, 0];
  const retainedMin = [Infinity, Infinity, Infinity];
  const retainedMax = [-Infinity, -Infinity, -Infinity];
  let insideAabbCount = 0;
  let removedCount = 0;
  for (let index = 0; index < scene.vertexCount; index += 1) {
    readVertexPosition(scene, index, point);
    let remove = false;
    if (insideExactBounds(point, pillow.bbox)) {
      insideAabbCount += 1;
      const nearestSquared = nearestAnchorDistanceSquared(spatial, point);
      for (const threshold of sensitivityDistances) {
        if (nearestSquared <= threshold * threshold) sensitivity[String(threshold)] += 1;
      }
      remove = nearestSquared <= distance * distance;
    }
    if (remove) {
      removedCount += 1;
      continue;
    }
    keep[index] = 1;
    for (let axis = 0; axis < 3; axis += 1) {
      retainedMin[axis] = Math.min(retainedMin[axis], point[axis]);
      retainedMax[axis] = Math.max(retainedMax[axis], point[axis]);
    }
  }
  const retainedInsideAabbCount = insideAabbCount - removedCount;
  if (removedCount < 1_000) throw new Error(`${pillow.id}: visual carve matched only ${removedCount} Gaussians`);
  if (removedCount >= insideAabbCount || retainedInsideAabbCount < 500) {
    throw new Error(`${pillow.id}: visual carve is not conservative inside the accepted AABB`);
  }
  const retainedCount = scene.vertexCount - removedCount;
  const header = scene.header.replace(/element vertex\s+\d+/, `element vertex ${retainedCount}`);
  if (header === scene.header) throw new Error(`${pillow.id}: failed to update static visual vertex count`);
  const headerBytes = Buffer.from(header, "ascii");
  const output = Buffer.allocUnsafe(headerBytes.length + retainedCount * scene.vertexStride);
  headerBytes.copy(output, 0);
  let cursor = headerBytes.length;
  for (let index = 0; index < scene.vertexCount; index += 1) {
    if (!keep[index]) continue;
    const sourceOffset = scene.dataOffset + index * scene.vertexStride;
    scene.bytes.copy(output, cursor, sourceOffset, sourceOffset + scene.vertexStride);
    cursor += scene.vertexStride;
  }
  const outputSha256 = sha256(output);
  const priorRemovedCount = Number(asset.removedVertexCount) || 0;
  const originalVertexCount = scene.vertexCount + priorRemovedCount;
  const parts = chunkBuffer(
    output,
    "visual_bedroom_4_pgsr_static_carved_pillow_ply",
    targetChunkDir,
    urlPrefix,
    chunkSize,
  );
  const carve = {
    status: "passed",
    objectId: pillow.id,
    method: "accepted_mask_projected_point_nearest_neighbor_within_exact_aabb",
    distance,
    units: "scene_scale_not_metric",
    acceptedAabb: pillow.bbox,
    inputStaticGaussianCount: scene.vertexCount,
    insideAcceptedAabbCount: insideAabbCount,
    removedGaussianCount: removedCount,
    retainedInsideAcceptedAabbCount: retainedInsideAabbCount,
    retainedStaticGaussianCount: retainedCount,
    removedOutsideAcceptedAabbCount: 0,
    sensitivityRemovedCounts: sensitivity,
    safetyGate: {
      exactAabbRequired: true,
      nearestAcceptedPointRequired: true,
      wholeAabbDeletionForbidden: true,
      unmatchedInsideAabbRetained: retainedInsideAabbCount,
      note: "Conservative identity-frame carve; unmatched bed/headboard Gaussians inside the broad pillow AABB remain static.",
    },
    inputAssetSha256: loaded.sha256,
    anchorSha256: anchor.sha256,
    anchorPointCount: anchor.vertexCount,
    anchorInsideAcceptedAabbCount,
    anchorInsideAcceptedAabbRatio: Number(anchorInsideAcceptedAabbRatio.toFixed(9)),
    anchorOutsideAcceptedAabbCount: anchor.vertexCount - anchorInsideAcceptedAabbCount,
    anchorRawBounds: { min: anchorRawMin, max: anchorRawMax },
    outputAssetSha256: outputSha256,
  };
  return {
    asset: {
      ...asset,
      id: "bedroom_4_pgsr_static_carved_interactive_objects_and_pillow_ply",
      label: "Bedroom 4 PGSR static Gaussian layer with interactive objects and pillow visual removed",
      vertexCount: retainedCount,
      originalVertexCount,
      inputVertexCount: scene.vertexCount,
      removedVertexCount: priorRemovedCount + removedCount,
      latestRemovedVertexCount: removedCount,
      removedByObject: {
        ...(asset.removedByObject || {}),
        [pillow.id]: removedCount,
      },
      carvedObjectIds: Array.from(new Set([...(asset.carvedObjectIds || []), pillow.id])),
      carveMethod: "prior_object_carve_then_conservative_pillow_anchor_nearest_neighbor",
      pillowCarve: carve,
      bbox: { min: retainedMin, max: retainedMax },
      size: output.length,
      sha256: outputSha256,
      headerByteLength: headerBytes.length,
      parts,
      chunkSize,
    },
    report: carve,
  };
}

function localPlacementMatrix(meshBounds, targetBounds) {
  const sourceMin = meshBounds[0].map(Number);
  const sourceMax = meshBounds[1].map(Number);
  const sourceCenter = sourceMin.map((value, axis) => (value + sourceMax[axis]) * 0.5);
  const sourceExtent = sourceMin.map((value, axis) => sourceMax[axis] - value);
  const targetExtent = targetBounds.extent.map(Number);
  if (![sourceMin, sourceMax, sourceExtent, targetExtent].every((item) => finiteVector(item))) {
    throw new Error("Cannot compute placement from non-finite bounds");
  }
  if (sourceExtent.some((value) => value <= 1e-8) || targetExtent.some((value) => value <= 1e-8)) {
    throw new Error("Cannot compute placement from degenerate bounds");
  }
  const scale = targetExtent.map((value, axis) => value / sourceExtent[axis]);
  const translation = scale.map((value, axis) => -value * sourceCenter[axis]);
  const matrix = [
    scale[0], 0, 0, 0,
    0, scale[1], 0, 0,
    0, 0, scale[2], 0,
    translation[0], translation[1], translation[2], 1,
  ];
  if (!matrix.every(Number.isFinite)) throw new Error("Placement matrix contains non-finite values");
  return {
    matrix: rounded(matrix, 9),
    sourceCenter: rounded(sourceCenter),
    sourceExtent: rounded(sourceExtent),
    targetExtent: rounded(targetExtent),
    scale: rounded(scale),
    fittedLocalCenter: [0, 0, 0],
    fittedLocalExtent: rounded(targetExtent),
  };
}

function blenderBoundsToWeb(bounds) {
  if (!finiteVector(bounds?.min) || !finiteVector(bounds?.max)) {
    throw new Error("Blender QA report contains invalid bounds");
  }
  const minimum = bounds.min.map(Number);
  const maximum = bounds.max.map(Number);
  return [
    [minimum[0], minimum[2], -maximum[1]],
    [maximum[0], maximum[2], -minimum[1]],
  ];
}

function runCollisionSimplifier({ scriptPath, blenderPath, input, output, report, targetFaces }) {
  execFileSync(process.execPath, [
    scriptPath,
    "--input", input,
    "--output", output,
    "--report", report,
    "--target-faces", String(targetFaces),
    "--blender", blenderPath,
  ], { stdio: "inherit" });
  const qa = readJson(report);
  if (qa.status !== "passed") throw new Error(`${input}: collision simplifier reported ${qa.status}`);
  if (qa.qa?.boundsPreservedWithinTolerance !== true) {
    throw new Error(`${input}: collision simplifier did not preserve bounds`);
  }
  return qa;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const sourceManifestPath = path.resolve(required(options, "source-manifest"));
  const sourceChunkDir = path.resolve(required(options, "source-chunks"));
  const tsdfPath = path.resolve(required(options, "tsdf"));
  const meshRoot = path.resolve(required(options, "mesh-root"));
  const worldDir = path.resolve(required(options, "world-dir"));
  const exampleManifest = options["example-manifest"]
    ? path.resolve(options["example-manifest"])
    : path.join(repoRoot, "examples/bedroom4/manifest.production.json");
  const exampleReport = options["example-report"]
    ? path.resolve(options["example-report"])
    : path.join(repoRoot, "examples/bedroom4/production-bundle-report.json");
  const cognitionManifestPath = path.resolve(
    options["cognition-manifest"]
      || path.join(repoRoot, "examples/bedroom4/cognition/outputs/cognition_manifest.json")
  );
  const pillowEntityPath = path.resolve(
    options["pillow-entity"]
      || path.join(repoRoot, "examples/bedroom4/evidence/pillow_delta_20260716/accepted/pillow_scene_entity_candidate.json")
  );
  const blenderPath = resolveExecutable(options.blender || process.env.BLENDER_BIN || "blender");
  const simplifierScript = path.resolve(options.simplifier || path.join(scriptDir, "simplify_collision_glb.mjs"));
  const urlPrefix = options["url-prefix"] || "./worlds/bedroom4";
  const version = options.version || "bedroom4-production-v1";
  const chunkSize = Number(options["chunk-size"] || 1024 * 1024);
  const carveMargin = Number(options["collider-carve-margin"] || 0.18);
  const pillowVisualCarveDistance = Number(
    options["pillow-visual-carve-distance"] || DEFAULT_PILLOW_VISUAL_CARVE_DISTANCE
  );
  if (!(chunkSize > 0 && carveMargin >= 0 && pillowVisualCarveDistance > 0)) {
    throw new Error("Invalid chunk size or carve distance");
  }

  const sourceManifest = readJson(sourceManifestPath);
  const cognition = loadCognition(cognitionManifestPath);
  const pillowEntity = loadPillowEntity(pillowEntityPath, cognition.byId);
  const sourceObjects = Array.isArray(sourceManifest.interactiveObjects)
    ? sourceManifest.interactiveObjects
    : [];
  const objectIds = Object.keys(OBJECT_TARGET_FACES);
  const objects = objectIds.map((objectId) => {
    const definition = sourceObjects.find((item) => item.id === objectId);
    if (!definition) throw new Error(`Source manifest missing ${objectId}`);
    const robustBounds = definition.sourceAnchor?.robustBounds;
    if (!finiteVector(robustBounds?.min) || !finiteVector(robustBounds?.max) || !finiteVector(robustBounds?.extent)) {
      throw new Error(`${objectId}: invalid source anchor bounds`);
    }
    return { definition, robustBounds };
  });

  const parent = path.dirname(worldDir);
  fs.mkdirSync(parent, { recursive: true });
  const tempDir = path.join(parent, `.bedroom4-next-${process.pid}`);
  fs.rmSync(tempDir, { recursive: true, force: true });
  fs.mkdirSync(tempDir, { recursive: true });
  const chunkDir = path.join(tempDir, "chunks");
  const colliderDir = path.join(tempDir, "colliders");
  const renderMeshDir = path.join(tempDir, "render-meshes");
  const evidenceDir = path.join(tempDir, "evidence");
  const qaDir = path.join(tempDir, "qa");
  for (const directory of [chunkDir, colliderDir, renderMeshDir, evidenceDir, qaDir]) fs.mkdirSync(directory, { recursive: true });

  try {
    const pillowVisualCarve = carvePillowFromStaticVisual({
      asset: sourceManifest.assets.visual,
      sourceChunkDir,
      targetChunkDir: chunkDir,
      urlPrefix,
      chunkSize,
      pillow: pillowEntity,
      distance: pillowVisualCarveDistance,
    });
    const visual = pillowVisualCarve.asset;

    const mesh = readBinaryTriangleMeshPly(tsdfPath);
    const objectCarveBounds = objects.map(({ definition, robustBounds }) => ({
      id: definition.id,
      bounds: expandedBounds(robustBounds, carveMargin),
    }));
    const sceneCarve = carveTriangleMesh(mesh, objectCarveBounds);
    const carvedColliderPath = path.join(colliderDir, "bedroom4_tsdf_static_carved.ply");
    const carvedColliderBytes = writeCarvedTriangleMesh(mesh, sceneCarve, carvedColliderPath);
    const carvedColliderParts = chunkBuffer(
      carvedColliderBytes,
      "collider_bedroom4_tsdf_static_carved",
      chunkDir,
      urlPrefix,
      chunkSize
    );
    fs.rmSync(carvedColliderPath);
    const originalCollider = sourceManifest.assets.collider;
    const colliderStaticCarved = {
      ...originalCollider,
      id: "bedroom4_tsdf_static_carved_interactive_objects",
      label: "Bedroom 4 TSDF static collider with interactive objects removed",
      fileName: "bedroom4_tsdf_static_carved.ply",
      sourcePath: tsdfPath,
      sourceSha256: sha256File(tsdfPath),
      vertexCount: mesh.vertexCount,
      faceCount: sceneCarve.keptFaces,
      originalFaceCount: mesh.faceCount,
      removedFaceCount: sceneCarve.removedFaces,
      removedByObject: sceneCarve.removedByObject,
      carveMargin,
      carveMethod: "expanded_source_anchor_aabb_triangle_intersection",
      retainedIntersectingFaces: sceneCarve.retainedIntersectingFaces,
      size: carvedColliderBytes.length,
      sha256: sha256(carvedColliderBytes),
      parts: carvedColliderParts,
      chunkSize,
    };

    const productionObjects = [];
    const collisionReports = [];
    for (const { definition, robustBounds } of objects) {
      const objectId = definition.id;
      const sourceGlb = path.join(meshRoot, objectId, `${objectId}.glb`);
      const renderGlb = path.join(renderMeshDir, `${objectId}.glb`);
      const collisionGlb = path.join(colliderDir, `${objectId}.collision.glb`);
      const collisionQaPath = path.join(qaDir, `${objectId}.collision.json`);
      const sourceVerified = verifyFile(sourceGlb, null, null, `${objectId} render GLB`);
      fs.copyFileSync(sourceGlb, renderGlb);
      const collisionQa = runCollisionSimplifier({
        scriptPath: simplifierScript,
        blenderPath,
        input: sourceGlb,
        output: collisionGlb,
        report: collisionQaPath,
        targetFaces: OBJECT_TARGET_FACES[objectId],
      });
      const outputStats = collisionQa.output;
      const outputWebBounds = blenderBoundsToWeb(outputStats.bounds);
      const sourceWebBounds = blenderBoundsToWeb(collisionQa.source.bounds);
      if (!finiteVector(outputWebBounds[0]) || !finiteVector(outputWebBounds[1])) {
        throw new Error(`${objectId}: simplifier report lacks output bounds`);
      }
      if (!(outputStats.faces > 0 && outputStats.faces <= OBJECT_TARGET_FACES[objectId] * 1.15)) {
        throw new Error(`${objectId}: simplified faces ${outputStats.faces} exceed budget`);
      }
      const placement = localPlacementMatrix(outputWebBounds, robustBounds);
      const collisionVerified = verifyFile(collisionGlb, null, null, `${objectId} collision GLB`);
      const sourceAnchorPath = path.resolve(definition.sourceAnchor.path);
      const sourceAnchorVerified = verifyFile(
        sourceAnchorPath,
        null,
        definition.sourceAnchor.sha256,
        `${objectId} source anchor PLY`,
      );
      const sourceAnchorFileName = `${objectId}.source-anchor.ply`;
      fs.copyFileSync(sourceAnchorPath, path.join(evidenceDir, sourceAnchorFileName));
      const sourceAnchor = {
        ...definition.sourceAnchor,
        path: `${urlPrefix}/evidence/${sourceAnchorFileName}`,
        fileName: sourceAnchorFileName,
        size: sourceAnchorVerified.size,
        sha256: sourceAnchorVerified.sha256,
      };
      const metadata = CATEGORY_METADATA[definition.category] || { aliases: [], description: {} };
      const cognitionRecord = cognition.byId.get(objectId);
      if (!cognitionRecord) throw new Error(`Cognition manifest is missing ${objectId}`);
      const interactiveVisual = copyAndRewriteChunkedAsset(
        definition.visual,
        sourceChunkDir,
        chunkDir,
        urlPrefix
      );
      const gate = {
        status: "candidate",
        boundsFit: "passed",
        collisionTechnicalQa: "passed",
        orientationEvidence: "same_trellis_object_pipeline_requires_browser_overlay_review",
        browserOverlayReview: "pending",
        note: "Promote to passed only after Gaussian/collision overlay and robot collision QA.",
      };
      productionObjects.push({
        ...definition,
        aliases: metadata.aliases,
        description: cognitionRecord.description_evidence,
        bbox: {
          coordinateFrame: "visual_native",
          min: robustBounds.min,
          max: robustBounds.max,
          center: robustBounds.center,
          extent: robustBounds.extent,
        },
        sourceAnchor,
        visual: interactiveVisual,
        collision: {
          mode: "kinematic",
          walkable: true,
          characterCollision: true,
          objectLocalMatrix: placement.matrix,
          placement,
          source: {
            path: sourceGlb,
            size: sourceVerified.size,
            sha256: sourceVerified.sha256,
            vertices: collisionQa.source.vertices,
            faces: collisionQa.source.faces,
            bounds: sourceWebBounds,
            blenderBounds: collisionQa.source.bounds,
          },
          renderAsset: {
            url: `${urlPrefix}/render-meshes/${objectId}.glb`,
            fileName: `${objectId}.glb`,
            fileType: "glb",
            format: "gltf-binary",
            size: sourceVerified.size,
            sha256: sourceVerified.sha256,
          },
          asset: {
            id: `${objectId}_simplified_collision_glb`,
            label: `${definition.label} simplified collision GLB`,
            url: `${urlPrefix}/colliders/${objectId}.collision.glb`,
            fileName: `${objectId}.collision.glb`,
            fileType: "glb",
            format: "gltf-binary",
            size: collisionVerified.size,
            sha256: collisionVerified.sha256,
            vertices: outputStats.vertices,
            faces: outputStats.faces,
            bounds: outputWebBounds,
            blenderBounds: outputStats.bounds,
            simplification: {
              tool: collisionQa.processing?.tool || "blender_headless_decimate",
              targetFaces: OBJECT_TARGET_FACES[objectId],
              originalFaces: collisionQa.source.faces,
              simplifiedFaces: outputStats.faces,
              ratio: Number((outputStats.faces / collisionQa.source.faces).toFixed(6)),
            },
          },
          gate,
        },
      });
      collisionReports.push({
        objectId,
        source: collisionQa.source,
        output: {
          ...collisionQa.output,
          path: path.join(worldDir, "colliders", `${objectId}.collision.glb`),
        },
        qaReport: path.join(worldDir, "qa", `${objectId}.collision.json`),
        placement,
        gate,
        sourceSha256: sourceVerified.sha256,
        collisionSha256: collisionVerified.sha256,
      });
    }

    const pillowVisualBytes = fs.readFileSync(pillowEntity.visualEvidence.path);
    if (sha256(pillowVisualBytes) !== pillowEntity.visualEvidence.sha256) {
      throw new Error(`${pillowEntity.id}: point visual changed after accepted-evidence verification`);
    }
    const pillowCenter = pillowEntity.bbox.center.map(Number);
    const pillowVisual = {
      id: `${pillowEntity.id}_accepted_rgb_point_visual`,
      label: "Accepted SAM3 pillow RGB point-cloud visual",
      fileName: `${pillowEntity.id}.ply`,
      fileType: "ply",
      format: "rgb-point-cloud-ply",
      renderer: "three-points",
      pointSize: 0.03,
      vertexCount: pillowEntity.visualEvidence.pointCount,
      primitiveKind: "rgb-point",
      coordinateFrame: "visual_native",
      bbox: pillowVisualCarve.report.anchorRawBounds,
      acceptedInteractionBounds: pillowEntity.bbox,
      sourcePath: pillowEntity.visualEvidence.path,
      size: pillowVisualBytes.length,
      sha256: pillowEntity.visualEvidence.sha256,
      parts: chunkBuffer(
        pillowVisualBytes,
        `interactive_${pillowEntity.id}_rgb_points`,
        chunkDir,
        urlPrefix,
        chunkSize,
      ),
      chunkSize,
    };
    const pillowComponent = {
      ...pillowEntity,
      label: pillowEntity.name,
      focusOnly: false,
      visualOnly: true,
      interactiveObjectId: pillowEntity.id,
      bbox: pillowEntity.bbox,
      placement: {
        coordinateFrame: "visual_native",
        sourceCoordinateFrame: pillowEntity.bbox.sourceCoordinateFrame,
        transform: pillowEntity.bbox.transform,
        pivot: pillowCenter,
        generatedCenter: pillowCenter,
        scale: [1, 1, 1],
        rotationEulerDeg: [0, 0, 0],
        eulerOrder: "XYZ",
      },
      colliderProxy: {
        type: "selection-box",
        dimensions: pillowEntity.bbox.extent.map(Number),
        center: [0, 0, 0],
        collisionEnabled: false,
      },
      visual: pillowVisual,
      collision: {
        mode: "none",
        walkable: false,
        characterCollision: false,
        asset: null,
        renderAsset: null,
        gate: {
          status: "not_tested",
          reason: "Accepted pillow has no mesh, GLB, OBJ, or robot collision QA.",
        },
      },
      interaction: {
        kind: "spin",
        degrees: 360,
        durationMs: 1250,
        drag: "horizontal_yaw",
      },
    };
    productionObjects.push(pillowComponent);

    const baselineCollider = {
      ...originalCollider,
      parts: [],
      baselineMetadataOnly: true,
      note: "Uncarved baseline bounds/hash retained for camera and presentation fixtures; not loaded for character collision.",
    };
    const manifest = {
      ...sourceManifest,
      schemaVersion: 1,
      contract: "video2world-web-manifest-1.0.0",
      version,
      sourceWorld: {
        schemaVersion: "world-manifest-1.0.0",
        worldId: "bedroom_4",
        runId: version,
        adoptionMode: "legacy_web_assets_adopted_then_canonical_manifest_validated",
      },
      presentation: {
        yawDeg: 180,
        pivot: [2.250915668599896, -3.375, 12.605660413500915],
        referenceBoundsAssetKey: "collider",
        sourceCommit: "252a85c52106838b9bd796f4e70253f7e68af488",
      },
      assets: {
        ...sourceManifest.assets,
        visual,
        collider: baselineCollider,
        colliderStaticCarved,
      },
      collisionWorld: {
        sceneAssetKey: "colliderStaticCarved",
        baselineBoundsAssetKey: "collider",
        replacementMode: "expanded_source_anchor_aabb_triangle_intersection",
        objectFaceRemovalRequired: true,
        gate: {
          status: "passed",
          originalFaces: mesh.faceCount,
          staticFaces: sceneCarve.keptFaces,
          removedFaces: sceneCarve.removedFaces,
          removedByObject: sceneCarve.removedByObject,
          retainedIntersectingFaces: sceneCarve.retainedIntersectingFaces,
          carveMargin,
        },
      },
      interactiveObjectBuild: {
        ...(sourceManifest.interactiveObjectBuild || {}),
        method: "semantic_anchor_carve_plus_affine_baked_trellis_lod_plus_rgb_point_component",
        recognizedObjectCount: productionObjects.length,
        visualObjectCount: productionObjects.length,
        collidableObjectCount: objects.length,
        originalSceneGaussianCount: visual.originalVertexCount,
        staticSceneGaussianCount: visual.vertexCount,
        removedSceneGaussianCount: visual.removedVertexCount,
        removedByObject: visual.removedByObject,
        objectGaussianCount: productionObjects
          .filter((item) => item.visual.primitiveKind !== "rgb-point")
          .reduce((sum, item) => sum + Number(item.visual.vertexCount || 0), 0),
        objectRgbPointCount: pillowVisual.vertexCount,
        pillowVisualCarve: pillowVisualCarve.report,
        objectPlacementBakedIntoPly: true,
      },
      sceneKnowledge: {
        coordinateFrame: "visual_native",
        missingInstances: [],
        objects: productionObjects.map((object) => ({
          id: object.id,
          name: object.label,
          category: object.category,
          aliases: object.aliases,
          description: object.description,
          bbox: object.bbox,
          interactiveObjectId: object.id,
          independentlyRecognized: true,
          visualOnly: object.collision?.mode === "none",
        })),
        relations: [{
          subject: pillowEntity.id,
          predicate: "OnTopOf",
          object: "sam3_bed_01",
          targetLabel: { zh: "床", en: "the bed" },
          confidence: 0.9734102225044038,
          verified: true,
          evidence: "pillow_projection_000020_000040_000064",
        }],
      },
      interactiveObjects: productionObjects,
      productionBuild: {
        schemaVersion: 1,
        createdAt: new Date().toISOString(),
        sourceManifest: sourceManifestPath,
        sourceManifestSha256: sha256File(sourceManifestPath),
        sourceChunkDir,
        tsdfPath,
        tsdfSha256: sha256File(tsdfPath),
        meshRoot,
        collisionSimplifier: simplifierScript,
        blender: blenderPath,
        cognitionManifest: cognitionManifestPath,
        cognitionManifestSha256: sha256File(cognitionManifestPath),
        pillowEntity: pillowEntityPath,
        pillowEntitySha256: sha256File(pillowEntityPath),
        pillowVisualCarveDistance,
        pillowVisualCarve: pillowVisualCarve.report,
        collisionWorldGate: "passed",
        objectPlacementGate: "candidate_pending_browser_overlay_review",
      },
    };

    const manifestPath = path.join(tempDir, "manifest.json");
    const portableManifest = portableProvenance(manifest, { tempDir, worldDir, blenderPath });
    writeJson(manifestPath, portableManifest);
    const report = portableProvenance({
      schemaVersion: 1,
      status: "candidate",
      version,
      worldDir,
      manifest: manifestPath,
      manifestSha256: sha256File(manifestPath),
      visual: {
        staticGaussians: visual.vertexCount,
        objectGaussians: productionObjects
          .filter((item) => item.visual.primitiveKind !== "rgb-point")
          .reduce((sum, item) => sum + Number(item.visual.vertexCount || 0), 0),
        objectRgbPoints: productionObjects
          .filter((item) => item.visual.primitiveKind === "rgb-point")
          .reduce((sum, item) => sum + Number(item.visual.vertexCount || 0), 0),
        objectVisualPrimitives: productionObjects.reduce(
          (sum, item) => sum + Number(item.visual.vertexCount || 0),
          0,
        ),
        visualObjectCount: productionObjects.length,
        chunks: visual.parts.length + productionObjects.reduce((sum, item) => sum + item.visual.parts.length, 0),
        pillowCarve: pillowVisualCarve.report,
      },
      collisionWorld: manifest.collisionWorld,
      collisions: collisionReports,
      gates: {
        assetHashes: "passed",
        staticColliderCarve: "passed",
        objectCollisionTechnicalQa: "passed",
        objectPlacementBrowserQa: "pending",
        cognition: "passed",
        pillowVisualFocus: "passed",
        pillowVisualComponent: "pending_browser_drag_spin",
        pillowStaticVisualCarve: "passed",
        pillowCollisionNone: "passed",
      },
    }, { tempDir, worldDir, blenderPath });
    writeJson(path.join(tempDir, "qa/production-bundle-report.json"), report);

    fs.rmSync(worldDir, { recursive: true, force: true });
    fs.renameSync(tempDir, worldDir);
    for (const objectId of Object.keys(OBJECT_TARGET_FACES)) {
      const collisionQaPath = path.join(worldDir, "qa", `${objectId}.collision.json`);
      const collisionQa = readJson(collisionQaPath);
      collisionQa.output.path = `web-bundle://bedroom4/colliders/${objectId}.collision.glb`;
      writeJson(collisionQaPath, collisionQa);
    }
    const finalManifest = readJson(path.join(worldDir, "manifest.json"));
    const finalReport = readJson(path.join(worldDir, "qa/production-bundle-report.json"));
    finalReport.manifest = "web-bundle://bedroom4/manifest.json";
    finalReport.manifestSha256 = sha256File(path.join(worldDir, "manifest.json"));
    writeJson(path.join(worldDir, "qa/production-bundle-report.json"), finalReport);
    writeJson(exampleManifest, finalManifest);
    writeJson(exampleReport, finalReport);
    console.log(JSON.stringify(finalReport, null, 2));
  } catch (error) {
    fs.rmSync(tempDir, { recursive: true, force: true });
    throw error;
  }
}

main();
