import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const PLY_SCALAR_TYPES = {
  char: { size: 1, reader: "readInt8" },
  int8: { size: 1, reader: "readInt8" },
  uchar: { size: 1, reader: "readUInt8" },
  uint8: { size: 1, reader: "readUInt8" },
  short: { size: 2, reader: "readInt16LE" },
  int16: { size: 2, reader: "readInt16LE" },
  ushort: { size: 2, reader: "readUInt16LE" },
  uint16: { size: 2, reader: "readUInt16LE" },
  int: { size: 4, reader: "readInt32LE" },
  int32: { size: 4, reader: "readInt32LE" },
  uint: { size: 4, reader: "readUInt32LE" },
  uint32: { size: 4, reader: "readUInt32LE" },
  float: { size: 4, reader: "readFloatLE" },
  float32: { size: 4, reader: "readFloatLE" },
  double: { size: 8, reader: "readDoubleLE" },
  float64: { size: 8, reader: "readDoubleLE" },
};

const GRAPHDECO_REQUIRED_PROPERTIES = [
  "x",
  "y",
  "z",
  "f_dc_0",
  "f_dc_1",
  "f_dc_2",
  "opacity",
  "scale_0",
  "scale_1",
  "scale_2",
  "rot_0",
  "rot_1",
  "rot_2",
  "rot_3",
];

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

export function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

export function sha256File(filePath) {
  const hash = crypto.createHash("sha256");
  const descriptor = fs.openSync(filePath, "r");
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  try {
    let bytesRead = 0;
    do {
      bytesRead = fs.readSync(descriptor, buffer, 0, buffer.length, null);
      if (bytesRead > 0) hash.update(buffer.subarray(0, bytesRead));
    } while (bytesRead > 0);
  } finally {
    fs.closeSync(descriptor);
  }
  return hash.digest("hex");
}

function parsePlyHeader(bytes, label) {
  const marker = Buffer.from("end_header");
  const markerIndex = bytes.indexOf(marker);
  requireCondition(markerIndex >= 0, `${label}: missing end_header`);
  let dataOffset = markerIndex + marker.length;
  while (bytes[dataOffset] === 10 || bytes[dataOffset] === 13) dataOffset += 1;
  const header = bytes.subarray(0, dataOffset).toString("ascii");
  requireCondition(
    header.includes("format binary_little_endian 1.0"),
    `${label}: expected binary_little_endian PLY`,
  );
  return { header, dataOffset };
}

function parseVertexElement(header, label, { allowFaces }) {
  let element = null;
  let vertexCount = 0;
  let faceCount = 0;
  let vertexStride = 0;
  const vertexProperties = new Map();
  const vertexPropertyNames = [];
  let faceList = null;
  for (const line of header.split(/\r?\n/u)) {
    const fields = line.trim().split(/\s+/u);
    if (fields[0] === "element") {
      element = fields[1];
      const count = Number(fields[2]);
      requireCondition(Number.isSafeInteger(count) && count >= 0, `${label}: invalid element count`);
      if (element === "vertex") vertexCount = count;
      else if (element === "face") faceCount = count;
      else if (count > 0) throw new Error(`${label}: unsupported non-empty element ${element}`);
      continue;
    }
    if (fields[0] !== "property") continue;
    if (element === "vertex") {
      requireCondition(fields[1] !== "list", `${label}: vertex list properties are unsupported`);
      const scalar = PLY_SCALAR_TYPES[fields[1]];
      requireCondition(scalar != null, `${label}: unsupported vertex type ${fields[1]}`);
      requireCondition(!vertexProperties.has(fields[2]), `${label}: duplicate property ${fields[2]}`);
      vertexProperties.set(fields[2], {
        type: fields[1],
        offset: vertexStride,
        size: scalar.size,
        reader: scalar.reader,
      });
      vertexPropertyNames.push(fields[2]);
      vertexStride += scalar.size;
    } else if (element === "face") {
      requireCondition(allowFaces, `${label}: expected a vertex-only PLY`);
      requireCondition(fields[1] === "list", `${label}: scalar face properties are unsupported`);
      requireCondition(faceList == null, `${label}: multiple face properties are unsupported`);
      faceList = { countType: fields[2], indexType: fields[3], name: fields[4] };
    }
  }
  requireCondition(vertexCount > 0, `${label}: missing vertex count`);
  requireCondition(vertexStride > 0, `${label}: missing vertex properties`);
  if (!allowFaces) requireCondition(faceCount === 0, `${label}: expected a vertex-only PLY`);
  return {
    vertexCount,
    faceCount,
    vertexStride,
    vertexProperties,
    vertexPropertyNames,
    faceList,
  };
}

function readVertexPosition(ply, index, target = [0, 0, 0]) {
  const base = ply.dataOffset + index * ply.vertexStride;
  for (let axis = 0; axis < 3; axis += 1) {
    const property = ply.vertexProperties.get(["x", "y", "z"][axis]);
    target[axis] = ply.bytes[property.reader](base + property.offset);
  }
  return target;
}

function validateBounds(items) {
  requireCondition(Array.isArray(items) && items.length > 0, "carve bounds must not be empty");
  const seen = new Set();
  return items.map((item, index) => {
    requireCondition(item && typeof item === "object", `carve bound ${index} is invalid`);
    requireCondition(typeof item.id === "string" && item.id.length > 0, `carve bound ${index} id`);
    requireCondition(!seen.has(item.id), `duplicate carve object id ${item.id}`);
    seen.add(item.id);
    const bounds = item.bounds;
    requireCondition(bounds && typeof bounds === "object", `${item.id}: bounds are missing`);
    for (const key of ["min", "max"]) {
      requireCondition(
        Array.isArray(bounds[key])
          && bounds[key].length === 3
          && bounds[key].every((value) => Number.isFinite(value)),
        `${item.id}: bounds.${key} must be a finite vec3`,
      );
    }
    requireCondition(
      bounds.min.every((value, axis) => value < bounds.max[axis]),
      `${item.id}: bounds must have positive extent`,
    );
    return { id: item.id, bounds };
  });
}

function pointInsideBounds(position, bounds) {
  return position.every((value, axis) => value >= bounds.min[axis] && value <= bounds.max[axis]);
}

function triangleIntersectsBounds(triangleMin, triangleMax, bounds) {
  return [0, 1, 2].every(
    (axis) => triangleMax[axis] >= bounds.min[axis] && triangleMin[axis] <= bounds.max[axis],
  );
}

function replaceElementCount(header, element, originalCount, outputCount, label) {
  const needle = new RegExp(`(^|\\n)element ${element} ${originalCount}(?=\\r?\\n)`, "u");
  const output = header.replace(needle, `$1element ${element} ${outputCount}`);
  requireCondition(output !== header || originalCount === outputCount, `${label}: cannot update ${element} count`);
  return output;
}

function updateBounds(bounds, position) {
  for (let axis = 0; axis < 3; axis += 1) {
    bounds.min[axis] = Math.min(bounds.min[axis], position[axis]);
    bounds.max[axis] = Math.max(bounds.max[axis], position[axis]);
  }
}

function emptyBounds() {
  return {
    min: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
    max: [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY],
  };
}

export function expandedBounds(bounds, margin) {
  requireCondition(Number.isFinite(margin) && margin >= 0, "carve margin must be non-negative");
  return {
    min: bounds.min.map((value) => Number(value) - margin),
    max: bounds.max.map((value) => Number(value) + margin),
  };
}

// The byte-preserving PLY approach is adapted from build_bedroom4_production_bundle.mjs.
export function carveGraphdecoPly(filePath, objectBounds) {
  const label = path.resolve(filePath);
  const bytes = fs.readFileSync(filePath);
  const headerRecord = parsePlyHeader(bytes, label);
  const element = parseVertexElement(headerRecord.header, label, { allowFaces: false });
  for (const property of GRAPHDECO_REQUIRED_PROPERTIES) {
    requireCondition(element.vertexProperties.has(property), `${label}: missing GraphDECO property ${property}`);
  }
  requireCondition(
    headerRecord.dataOffset + element.vertexCount * element.vertexStride === bytes.length,
    `${label}: vertex payload length does not match header`,
  );
  const bounds = validateBounds(objectBounds);
  const keep = new Uint8Array(element.vertexCount);
  const removedByObject = Object.fromEntries(bounds.map((item) => [item.id, 0]));
  const outputBounds = emptyBounds();
  const position = [0, 0, 0];
  let keptCount = 0;
  const ply = { bytes, dataOffset: headerRecord.dataOffset, ...element };
  for (let index = 0; index < element.vertexCount; index += 1) {
    readVertexPosition(ply, index, position);
    requireCondition(position.every(Number.isFinite), `${label}: non-finite position at vertex ${index}`);
    const matched = bounds.find((item) => pointInsideBounds(position, item.bounds));
    if (matched) {
      removedByObject[matched.id] += 1;
      continue;
    }
    keep[index] = 1;
    keptCount += 1;
    updateBounds(outputBounds, position);
  }
  requireCondition(keptCount > 0, `${label}: carve removed every Gaussian`);
  for (const [objectId, removed] of Object.entries(removedByObject)) {
    requireCondition(removed > 0, `${objectId}: visual carve removed zero Gaussians`);
  }
  const outputHeader = replaceElementCount(
    headerRecord.header,
    "vertex",
    element.vertexCount,
    keptCount,
    label,
  );
  const headerBytes = Buffer.from(outputHeader, "ascii");
  const payload = Buffer.allocUnsafe(keptCount * element.vertexStride);
  let cursor = 0;
  for (let index = 0; index < element.vertexCount; index += 1) {
    if (!keep[index]) continue;
    const start = headerRecord.dataOffset + index * element.vertexStride;
    bytes.copy(payload, cursor, start, start + element.vertexStride);
    cursor += element.vertexStride;
  }
  return {
    bytes: Buffer.concat([headerBytes, payload]),
    inputVertexCount: element.vertexCount,
    vertexCount: keptCount,
    removedVertexCount: element.vertexCount - keptCount,
    removedByObject,
    bbox: outputBounds,
    headerByteLength: headerBytes.length,
    vertexStride: element.vertexStride,
    vertexProperties: element.vertexPropertyNames,
  };
}

export function readBinaryTriangleMeshPly(filePath) {
  const label = path.resolve(filePath);
  const bytes = fs.readFileSync(filePath);
  const headerRecord = parsePlyHeader(bytes, label);
  const element = parseVertexElement(headerRecord.header, label, { allowFaces: true });
  requireCondition(element.faceCount > 0 && element.faceList != null, `${label}: missing faces`);
  for (const axis of ["x", "y", "z"]) {
    requireCondition(element.vertexProperties.has(axis), `${label}: missing ${axis}`);
  }
  requireCondition(
    ["uchar", "uint8"].includes(element.faceList.countType)
      && ["uint", "uint32"].includes(element.faceList.indexType),
    `${label}: expected property list uchar uint vertex_indices`,
  );
  const vertexPayloadBytes = element.vertexCount * element.vertexStride;
  const faceOffset = headerRecord.dataOffset + vertexPayloadBytes;
  const faceStride = 13;
  requireCondition(
    faceOffset + element.faceCount * faceStride === bytes.length,
    `${label}: expected fixed triangle records`,
  );
  const positions = new Float64Array(element.vertexCount * 3);
  const ply = { bytes, dataOffset: headerRecord.dataOffset, ...element };
  const position = [0, 0, 0];
  for (let index = 0; index < element.vertexCount; index += 1) {
    readVertexPosition(ply, index, position);
    requireCondition(position.every(Number.isFinite), `${label}: non-finite position at vertex ${index}`);
    positions.set(position, index * 3);
  }
  return {
    filePath: label,
    bytes,
    header: headerRecord.header,
    dataOffset: headerRecord.dataOffset,
    faceOffset,
    faceStride,
    vertexPayloadBytes,
    positions,
    ...element,
  };
}

export function carveTriangleMeshPly(filePath, objectBounds) {
  const mesh = readBinaryTriangleMeshPly(filePath);
  const bounds = validateBounds(objectBounds);
  const keep = new Uint8Array(mesh.faceCount);
  const removedByObject = Object.fromEntries(bounds.map((item) => [item.id, 0]));
  const referenced = new Uint8Array(mesh.vertexCount);
  let keptFaces = 0;
  const triangleMin = [0, 0, 0];
  const triangleMax = [0, 0, 0];
  for (let face = 0; face < mesh.faceCount; face += 1) {
    const offset = mesh.faceOffset + face * mesh.faceStride;
    requireCondition(mesh.bytes[offset] === 3, `${mesh.filePath}: non-triangle face ${face}`);
    const indices = [
      mesh.bytes.readUInt32LE(offset + 1),
      mesh.bytes.readUInt32LE(offset + 5),
      mesh.bytes.readUInt32LE(offset + 9),
    ];
    requireCondition(
      indices.every((index) => index < mesh.vertexCount),
      `${mesh.filePath}: face ${face} has an invalid vertex index`,
    );
    for (let axis = 0; axis < 3; axis += 1) {
      const values = indices.map((index) => mesh.positions[index * 3 + axis]);
      triangleMin[axis] = Math.min(...values);
      triangleMax[axis] = Math.max(...values);
    }
    const matched = bounds.find((item) =>
      triangleIntersectsBounds(triangleMin, triangleMax, item.bounds));
    if (matched) {
      removedByObject[matched.id] += 1;
      continue;
    }
    keep[face] = 1;
    keptFaces += 1;
    indices.forEach((index) => { referenced[index] = 1; });
  }
  requireCondition(keptFaces > 0, `${mesh.filePath}: carve removed every triangle`);
  for (const [objectId, removed] of Object.entries(removedByObject)) {
    requireCondition(removed > 0, `${objectId}: collider carve removed zero faces`);
  }
  const outputBounds = emptyBounds();
  for (let index = 0; index < mesh.vertexCount; index += 1) {
    if (!referenced[index]) continue;
    updateBounds(outputBounds, [
      mesh.positions[index * 3],
      mesh.positions[index * 3 + 1],
      mesh.positions[index * 3 + 2],
    ]);
  }
  const outputHeader = replaceElementCount(
    mesh.header,
    "face",
    mesh.faceCount,
    keptFaces,
    mesh.filePath,
  );
  const headerBytes = Buffer.from(outputHeader, "ascii");
  const faces = Buffer.allocUnsafe(keptFaces * mesh.faceStride);
  let cursor = 0;
  for (let face = 0; face < mesh.faceCount; face += 1) {
    if (!keep[face]) continue;
    const start = mesh.faceOffset + face * mesh.faceStride;
    mesh.bytes.copy(faces, cursor, start, start + mesh.faceStride);
    cursor += mesh.faceStride;
  }
  return {
    bytes: Buffer.concat([
      headerBytes,
      mesh.bytes.subarray(mesh.dataOffset, mesh.faceOffset),
      faces,
    ]),
    originalFaceCount: mesh.faceCount,
    faceCount: keptFaces,
    removedFaceCount: mesh.faceCount - keptFaces,
    removedByObject,
    bbox: outputBounds,
    headerByteLength: headerBytes.length,
    vertexCount: mesh.vertexCount,
    vertexStride: mesh.vertexStride,
    vertexProperties: mesh.vertexPropertyNames,
    vertexPositionType: mesh.vertexProperties.get("x").type,
    faceIndexType: "uint32",
    faceIndicesValid: true,
    finite: true,
  };
}

export function describeTriangleMeshPly(filePath) {
  const mesh = readBinaryTriangleMeshPly(filePath);
  const bbox = emptyBounds();
  for (let index = 0; index < mesh.vertexCount; index += 1) {
    updateBounds(bbox, [
      mesh.positions[index * 3],
      mesh.positions[index * 3 + 1],
      mesh.positions[index * 3 + 2],
    ]);
  }
  return {
    bytes: mesh.bytes,
    bbox,
    headerByteLength: mesh.dataOffset,
    vertexCount: mesh.vertexCount,
    faceCount: mesh.faceCount,
    vertexStride: mesh.vertexStride,
    vertexProperties: mesh.vertexPropertyNames,
    vertexPositionType: mesh.vertexProperties.get("x").type,
    faceIndexType: "uint32",
  };
}

export function chunkBuffer(bytes, prefix, targetChunkDir, urlPrefix, chunkSize) {
  requireCondition(Number.isSafeInteger(chunkSize) && chunkSize > 0, "chunk size must be positive");
  fs.mkdirSync(targetChunkDir, { recursive: true });
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
  requireCondition(parts.length > 0, `${prefix}: no chunks were written`);
  return parts;
}
