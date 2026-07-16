#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const TYPE_READERS = {
  char: { size: 1, read: "readInt8", write: "writeInt8" },
  int8: { size: 1, read: "readInt8", write: "writeInt8" },
  uchar: { size: 1, read: "readUInt8", write: "writeUInt8" },
  uint8: { size: 1, read: "readUInt8", write: "writeUInt8" },
  short: { size: 2, read: "readInt16LE", write: "writeInt16LE" },
  int16: { size: 2, read: "readInt16LE", write: "writeInt16LE" },
  ushort: { size: 2, read: "readUInt16LE", write: "writeUInt16LE" },
  uint16: { size: 2, read: "readUInt16LE", write: "writeUInt16LE" },
  int: { size: 4, read: "readInt32LE", write: "writeInt32LE" },
  int32: { size: 4, read: "readInt32LE", write: "writeInt32LE" },
  uint: { size: 4, read: "readUInt32LE", write: "writeUInt32LE" },
  uint32: { size: 4, read: "readUInt32LE", write: "writeUInt32LE" },
  float: { size: 4, read: "readFloatLE", write: "writeFloatLE" },
  float32: { size: 4, read: "readFloatLE", write: "writeFloatLE" },
  double: { size: 8, read: "readDoubleLE", write: "writeDoubleLE" },
  float64: { size: 8, read: "readDoubleLE", write: "writeDoubleLE" },
};

const OBJECT_SPECS = [
  { id: "sam3_nightstand_01", label: "Nightstand 01", category: "nightstand" },
  { id: "sam3_nightstand_02", label: "Nightstand 02", category: "nightstand" },
  { id: "sam3_plant_01", label: "Plant 01", category: "plant", fidelity: "category_proxy_from_low_detail_evidence" },
  { id: "sam3_plant_02", label: "Plant 02", category: "plant", fidelity: "category_proxy_from_low_detail_evidence" },
];

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for --${key}`);
    result[key] = value;
    index += 1;
  }
  return result;
}

function required(options, key) {
  if (!options[key]) throw new Error(`Missing required option --${key}`);
  return options[key];
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function readBinaryPly(filePath) {
  const bytes = fs.readFileSync(filePath);
  const marker = Buffer.from("end_header");
  const markerOffset = bytes.indexOf(marker);
  if (markerOffset < 0) throw new Error(`${filePath}: missing PLY end_header`);
  let dataOffset = markerOffset + marker.length;
  while (bytes[dataOffset] === 10 || bytes[dataOffset] === 13) dataOffset += 1;
  const headerText = bytes.subarray(0, dataOffset).toString("ascii");
  if (!headerText.includes("format binary_little_endian 1.0")) {
    throw new Error(`${filePath}: expected binary_little_endian PLY`);
  }

  const elements = [];
  let currentElement = null;
  for (const rawLine of headerText.split(/\r?\n/)) {
    const fields = rawLine.trim().split(/\s+/);
    if (fields[0] === "element") {
      currentElement = { name: fields[1], count: Number(fields[2]), properties: [] };
      elements.push(currentElement);
    } else if (fields[0] === "property" && currentElement) {
      if (fields[1] === "list") {
        currentElement.properties.push({ list: true, countType: fields[2], itemType: fields[3], name: fields[4] });
      } else {
        currentElement.properties.push({ list: false, type: fields[1], name: fields[2] });
      }
    }
  }

  const vertex = elements.find((element) => element.name === "vertex");
  if (!vertex?.count || vertex.properties.some((property) => property.list)) {
    throw new Error(`${filePath}: unsupported or missing vertex element`);
  }
  let vertexStride = 0;
  const properties = new Map();
  for (const property of vertex.properties) {
    const reader = TYPE_READERS[property.type];
    if (!reader) throw new Error(`${filePath}: unsupported PLY type ${property.type}`);
    properties.set(property.name, { ...property, offset: vertexStride, reader });
    vertexStride += reader.size;
  }
  for (const axis of ["x", "y", "z"]) {
    if (!properties.has(axis)) throw new Error(`${filePath}: missing ${axis} property`);
  }
  const vertexPayloadEnd = dataOffset + vertex.count * vertexStride;
  if (vertexPayloadEnd !== bytes.length) {
    throw new Error(`${filePath}: expected a vertex-only PLY; got ${bytes.length - vertexPayloadEnd} trailing bytes`);
  }
  return {
    filePath,
    bytes,
    headerText,
    dataOffset,
    vertexCount: vertex.count,
    vertexStride,
    vertexProperties: vertex.properties.map((property) => property.name),
    properties,
  };
}

function readVertexProperty(ply, vertexIndex, name) {
  const property = ply.properties.get(name);
  if (!property) return null;
  return ply.bytes[property.reader.read](ply.dataOffset + vertexIndex * ply.vertexStride + property.offset);
}

function readPosition(ply, vertexIndex, target = [0, 0, 0]) {
  target[0] = readVertexProperty(ply, vertexIndex, "x");
  target[1] = readVertexProperty(ply, vertexIndex, "y");
  target[2] = readVertexProperty(ply, vertexIndex, "z");
  return target;
}

function readAsciiPointPly(filePath) {
  const text = fs.readFileSync(filePath, "utf8");
  const lines = text.split(/\r?\n/);
  let vertexCount = 0;
  let dataStart = -1;
  const properties = [];
  let inVertex = false;
  for (let index = 0; index < lines.length; index += 1) {
    const fields = lines[index].trim().split(/\s+/);
    if (fields[0] === "format" && fields[1] !== "ascii") {
      throw new Error(`${filePath}: expected ASCII source anchor PLY`);
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
  if (!vertexCount || dataStart < 0) throw new Error(`${filePath}: invalid ASCII PLY header`);
  const axisIndices = ["x", "y", "z"].map((axis) => properties.indexOf(axis));
  if (axisIndices.some((index) => index < 0)) throw new Error(`${filePath}: source anchor lacks xyz`);
  const positions = new Float64Array(vertexCount * 3);
  for (let index = 0; index < vertexCount; index += 1) {
    const fields = lines[dataStart + index].trim().split(/\s+/);
    for (let axis = 0; axis < 3; axis += 1) {
      const value = Number(fields[axisIndices[axis]]);
      if (!Number.isFinite(value)) throw new Error(`${filePath}: non-finite point ${index}`);
      positions[index * 3 + axis] = value;
    }
  }
  return { filePath, vertexCount, positions, sha256: sha256(Buffer.from(text)) };
}

function quantile(sorted, fraction) {
  if (!sorted.length) throw new Error("Cannot compute a quantile of an empty array");
  const index = Math.max(0, Math.min(sorted.length - 1, Math.round((sorted.length - 1) * fraction)));
  return sorted[index];
}

function boundsFromAxisValues(axisValues, low = 0.01, high = 0.99) {
  const min = [];
  const max = [];
  for (const values of axisValues) {
    values.sort();
    min.push(quantile(values, low));
    max.push(quantile(values, high));
  }
  const center = min.map((value, axis) => (value + max[axis]) * 0.5);
  const extent = min.map((value, axis) => Math.max(1e-5, max[axis] - value));
  return { min, max, center, extent, quantileLow: low, quantileHigh: high };
}

function robustBoundsFromPositions(positions, low = 0.01, high = 0.99) {
  const count = positions.length / 3;
  const axes = [new Float64Array(count), new Float64Array(count), new Float64Array(count)];
  for (let index = 0; index < count; index += 1) {
    for (let axis = 0; axis < 3; axis += 1) axes[axis][index] = positions[index * 3 + axis];
  }
  return boundsFromAxisValues(axes, low, high);
}

function robustGaussianBounds(ply, low = 0.01, high = 0.99) {
  const hasOpacity = ply.properties.has("opacity");
  let eligibleCount = 0;
  for (let index = 0; index < ply.vertexCount; index += 1) {
    if (!hasOpacity || readVertexProperty(ply, index, "opacity") > 0) eligibleCount += 1;
  }
  if (!eligibleCount) throw new Error(`${ply.filePath}: no positive-opacity Gaussians`);
  const axes = [new Float64Array(eligibleCount), new Float64Array(eligibleCount), new Float64Array(eligibleCount)];
  const point = [0, 0, 0];
  let cursor = 0;
  for (let index = 0; index < ply.vertexCount; index += 1) {
    if (hasOpacity && readVertexProperty(ply, index, "opacity") <= 0) continue;
    readPosition(ply, index, point);
    for (let axis = 0; axis < 3; axis += 1) axes[axis][cursor] = point[axis];
    cursor += 1;
  }
  return { ...boundsFromAxisValues(axes, low, high), eligibleCount };
}

function selectedGaussianIndices(ply, limit) {
  const hasOpacity = ply.properties.has("opacity");
  const positive = [];
  const remainder = [];
  for (let index = 0; index < ply.vertexCount; index += 1) {
    if (!hasOpacity || readVertexProperty(ply, index, "opacity") > 0) positive.push(index);
    else remainder.push(index);
  }
  const target = Math.min(limit, ply.vertexCount);
  const selected = [];
  const appendEven = (indices, count) => {
    if (count <= 0 || !indices.length) return;
    const actual = Math.min(count, indices.length);
    for (let index = 0; index < actual; index += 1) {
      selected.push(indices[Math.min(indices.length - 1, Math.floor((index + 0.5) * indices.length / actual))]);
    }
  };
  appendEven(positive, Math.min(target, positive.length));
  appendEven(remainder, target - selected.length);
  selected.sort((a, b) => a - b);
  return selected;
}

function rewriteVertexPly(ply, selectedIndices) {
  const headerText = ply.headerText.replace(/element vertex\s+\d+/, `element vertex ${selectedIndices.length}`);
  const header = Buffer.from(headerText, "ascii");
  const output = Buffer.allocUnsafe(header.length + selectedIndices.length * ply.vertexStride);
  header.copy(output, 0);
  for (let outputIndex = 0; outputIndex < selectedIndices.length; outputIndex += 1) {
    const sourceIndex = selectedIndices[outputIndex];
    const sourceStart = ply.dataOffset + sourceIndex * ply.vertexStride;
    const outputStart = header.length + outputIndex * ply.vertexStride;
    ply.bytes.copy(output, outputStart, sourceStart, sourceStart + ply.vertexStride);
  }
  return { bytes: output, headerByteLength: header.length };
}

function writeVertexProperty(output, outputDataOffset, ply, outputIndex, name, value) {
  const property = ply.properties.get(name);
  if (!property) throw new Error(`${ply.filePath}: missing writable property ${name}`);
  output[property.reader.write](value, outputDataOffset + outputIndex * ply.vertexStride + property.offset);
}

function rotationMatrixFromQuaternionWxyz(quaternion) {
  let [w, x, y, z] = quaternion;
  const norm = Math.hypot(w, x, y, z);
  if (norm < 1e-12) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  w /= norm;
  x /= norm;
  y /= norm;
  z /= norm;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

function quaternionWxyzFromRotationMatrix(matrix) {
  const m00 = matrix[0][0];
  const m01 = matrix[0][1];
  const m02 = matrix[0][2];
  const m10 = matrix[1][0];
  const m11 = matrix[1][1];
  const m12 = matrix[1][2];
  const m20 = matrix[2][0];
  const m21 = matrix[2][1];
  const m22 = matrix[2][2];
  let w;
  let x;
  let y;
  let z;
  const trace = m00 + m11 + m22;
  if (trace > 0) {
    const scale = Math.sqrt(trace + 1) * 2;
    w = 0.25 * scale;
    x = (m21 - m12) / scale;
    y = (m02 - m20) / scale;
    z = (m10 - m01) / scale;
  } else if (m00 > m11 && m00 > m22) {
    const scale = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / scale;
    x = 0.25 * scale;
    y = (m01 + m10) / scale;
    z = (m02 + m20) / scale;
  } else if (m11 > m22) {
    const scale = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / scale;
    x = (m01 + m10) / scale;
    y = 0.25 * scale;
    z = (m12 + m21) / scale;
  } else {
    const scale = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / scale;
    x = (m02 + m20) / scale;
    y = (m12 + m21) / scale;
    z = 0.25 * scale;
  }
  const norm = Math.hypot(w, x, y, z) || 1;
  const sign = w < 0 ? -1 : 1;
  return [w, x, y, z].map((value) => (value / norm) * sign);
}

function determinant3(matrix) {
  return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
    - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
    + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

function decomposeSymmetric3(input) {
  const matrix = input.map((row) => [...row]);
  const vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const pairs = [[0, 1], [0, 2], [1, 2]];
  for (let iteration = 0; iteration < 24; iteration += 1) {
    let [p, q] = pairs[0];
    for (const pair of pairs.slice(1)) {
      if (Math.abs(matrix[pair[0]][pair[1]]) > Math.abs(matrix[p][q])) [p, q] = pair;
    }
    const offDiagonal = matrix[p][q];
    if (Math.abs(offDiagonal) < 1e-14) break;
    const tau = (matrix[q][q] - matrix[p][p]) / (2 * offDiagonal);
    const tangent = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const cosine = 1 / Math.sqrt(1 + tangent * tangent);
    const sine = tangent * cosine;
    const app = matrix[p][p];
    const aqq = matrix[q][q];
    matrix[p][p] = app - tangent * offDiagonal;
    matrix[q][q] = aqq + tangent * offDiagonal;
    matrix[p][q] = 0;
    matrix[q][p] = 0;
    for (let axis = 0; axis < 3; axis += 1) {
      if (axis === p || axis === q) continue;
      const aip = matrix[axis][p];
      const aiq = matrix[axis][q];
      matrix[axis][p] = cosine * aip - sine * aiq;
      matrix[p][axis] = matrix[axis][p];
      matrix[axis][q] = sine * aip + cosine * aiq;
      matrix[q][axis] = matrix[axis][q];
    }
    for (let row = 0; row < 3; row += 1) {
      const vip = vectors[row][p];
      const viq = vectors[row][q];
      vectors[row][p] = cosine * vip - sine * viq;
      vectors[row][q] = sine * vip + cosine * viq;
    }
  }
  const order = [0, 1, 2].sort((a, b) => matrix[b][b] - matrix[a][a]);
  const values = order.map((axis) => Math.max(matrix[axis][axis], 1e-18));
  const orderedVectors = vectors.map((row) => order.map((axis) => row[axis]));
  if (determinant3(orderedVectors) < 0) {
    for (let row = 0; row < 3; row += 1) orderedVectors[row][2] *= -1;
  }
  return { values, vectors: orderedVectors };
}

function gaussianShapeAfterAxisScale(ply, vertexIndex, axisScale) {
  const rotation = rotationMatrixFromQuaternionWxyz([
    readVertexProperty(ply, vertexIndex, "rot_0"),
    readVertexProperty(ply, vertexIndex, "rot_1"),
    readVertexProperty(ply, vertexIndex, "rot_2"),
    readVertexProperty(ply, vertexIndex, "rot_3"),
  ]);
  const variance = [0, 1, 2].map((axis) => {
    const logScale = readVertexProperty(ply, vertexIndex, `scale_${axis}`);
    return Math.exp(2 * logScale);
  });
  const covariance = Array.from({ length: 3 }, (_, row) => Array.from({ length: 3 }, (_, column) => {
    let value = 0;
    for (let axis = 0; axis < 3; axis += 1) {
      value += rotation[row][axis] * variance[axis] * rotation[column][axis];
    }
    return value * axisScale[row] * axisScale[column];
  }));
  const decomposed = decomposeSymmetric3(covariance);
  if (vertexIndex % 4096 === 0) {
    let maxExpected = 0;
    let maxError = 0;
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 3; column += 1) {
        let reconstructed = 0;
        for (let axis = 0; axis < 3; axis += 1) {
          reconstructed += decomposed.vectors[row][axis]
            * decomposed.values[axis]
            * decomposed.vectors[column][axis];
        }
        maxExpected = Math.max(maxExpected, Math.abs(covariance[row][column]));
        maxError = Math.max(maxError, Math.abs(reconstructed - covariance[row][column]));
      }
    }
    if (maxError > Math.max(1e-12, maxExpected * 1e-5)) {
      throw new Error(`${ply.filePath}: covariance bake residual ${maxError} at Gaussian ${vertexIndex}`);
    }
  }
  return {
    logScale: decomposed.values.map((value) => Math.log(Math.sqrt(value))),
    quaternion: quaternionWxyzFromRotationMatrix(decomposed.vectors),
  };
}

function rewritePlacedGaussianPly(ply, selectedIndices, generatedCenter, axisScale) {
  for (const name of ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]) {
    if (!ply.properties.has(name)) throw new Error(`${ply.filePath}: cannot bake placement without ${name}`);
  }
  const headerText = ply.headerText.replace(/element vertex\s+\d+/, `element vertex ${selectedIndices.length}`);
  const header = Buffer.from(headerText, "ascii");
  const output = Buffer.allocUnsafe(header.length + selectedIndices.length * ply.vertexStride);
  header.copy(output, 0);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const point = [0, 0, 0];
  for (let outputIndex = 0; outputIndex < selectedIndices.length; outputIndex += 1) {
    const sourceIndex = selectedIndices[outputIndex];
    const sourceStart = ply.dataOffset + sourceIndex * ply.vertexStride;
    const outputStart = header.length + outputIndex * ply.vertexStride;
    ply.bytes.copy(output, outputStart, sourceStart, sourceStart + ply.vertexStride);
    readPosition(ply, sourceIndex, point);
    for (let axis = 0; axis < 3; axis += 1) {
      point[axis] = (point[axis] - generatedCenter[axis]) * axisScale[axis];
      min[axis] = Math.min(min[axis], point[axis]);
      max[axis] = Math.max(max[axis], point[axis]);
      writeVertexProperty(output, header.length, ply, outputIndex, ["x", "y", "z"][axis], point[axis]);
    }
    const shape = gaussianShapeAfterAxisScale(ply, sourceIndex, axisScale);
    for (let axis = 0; axis < 3; axis += 1) {
      writeVertexProperty(output, header.length, ply, outputIndex, `scale_${axis}`, shape.logScale[axis]);
    }
    for (let axis = 0; axis < 4; axis += 1) {
      writeVertexProperty(output, header.length, ply, outputIndex, `rot_${axis}`, shape.quaternion[axis]);
    }
  }
  return { bytes: output, headerByteLength: header.length, bbox: { min, max } };
}

function exactBounds(ply, selectedIndices) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const point = [0, 0, 0];
  for (const index of selectedIndices) {
    readPosition(ply, index, point);
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], point[axis]);
      max[axis] = Math.max(max[axis], point[axis]);
    }
  }
  return { min, max };
}

function buildSpatialHash(source, cellSize) {
  const cells = new Map();
  for (let index = 0; index < source.vertexCount; index += 1) {
    const base = index * 3;
    const key = [0, 1, 2].map((axis) => Math.floor(source.positions[base + axis] / cellSize)).join(",");
    let indices = cells.get(key);
    if (!indices) {
      indices = [];
      cells.set(key, indices);
    }
    indices.push(index);
  }
  return { ...source, cellSize, cells };
}

function nearSourcePoint(anchor, point, threshold) {
  const { cellSize, positions, cells } = anchor;
  const cell = point.map((value) => Math.floor(value / cellSize));
  const thresholdSq = threshold * threshold;
  for (let dx = -1; dx <= 1; dx += 1) {
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dz = -1; dz <= 1; dz += 1) {
        const candidates = cells.get(`${cell[0] + dx},${cell[1] + dy},${cell[2] + dz}`);
        if (!candidates) continue;
        for (const index of candidates) {
          const base = index * 3;
          const px = positions[base] - point[0];
          const py = positions[base + 1] - point[1];
          const pz = positions[base + 2] - point[2];
          if (px * px + py * py + pz * pz <= thresholdSq) return true;
        }
      }
    }
  }
  return false;
}

function insideBounds(point, bounds) {
  return point.every((value, axis) => value >= bounds.min[axis] && value <= bounds.max[axis]);
}

function expandedBounds(bounds, ratio, absoluteMargin) {
  const min = [];
  const max = [];
  for (let axis = 0; axis < 3; axis += 1) {
    const margin = bounds.extent[axis] * ratio + absoluteMargin;
    min.push(bounds.min[axis] - margin);
    max.push(bounds.max[axis] + margin);
  }
  return { min, max };
}

function carveScene(scenePly, objects, threshold) {
  const selected = [];
  const removedByObject = Object.fromEntries(objects.map((object) => [object.id, 0]));
  const point = [0, 0, 0];
  for (let index = 0; index < scenePly.vertexCount; index += 1) {
    readPosition(scenePly, index, point);
    let removed = false;
    for (const object of objects) {
      if (!insideBounds(point, object.carveBounds)) continue;
      if (!nearSourcePoint(object.anchorHash, point, threshold)) continue;
      removedByObject[object.id] += 1;
      removed = true;
      break;
    }
    if (!removed) selected.push(index);
  }
  for (const [objectId, count] of Object.entries(removedByObject)) {
    if (count < 25) throw new Error(`${objectId}: only ${count} scene Gaussians matched; refusing a fake replacement`);
  }
  return { selected, removedByObject, removedCount: scenePly.vertexCount - selected.length };
}

function writeChunks(bytes, prefix, directory, chunkSize) {
  const parts = [];
  for (let offset = 0, index = 0; offset < bytes.length; offset += chunkSize, index += 1) {
    const chunk = bytes.subarray(offset, Math.min(bytes.length, offset + chunkSize));
    const name = `${prefix}.chunk${String(index).padStart(3, "0")}`;
    fs.writeFileSync(path.join(directory, name), chunk);
    parts.push({ url: `./assets/chunks/${name}`, size: chunk.length, sha256: sha256(chunk) });
  }
  return parts;
}

function buildChunkedAsset({ bytes, prefix, directory, chunkSize, fields, headerByteLength, vertexStride, vertexProperties, bbox }) {
  return {
    ...fields,
    bbox,
    size: bytes.length,
    sha256: sha256(bytes),
    headerByteLength,
    vertexStride,
    vertexProperties,
    parts: writeChunks(bytes, prefix, directory, chunkSize),
    chunkSize,
  };
}

function roundedArray(values, digits = 7) {
  return values.map((value) => Number(value.toFixed(digits)));
}

const options = parseArgs(process.argv.slice(2));
const scenePath = path.resolve(required(options, "scene"));
const sourceDir = path.resolve(required(options, "source-dir"));
const generatedDir = path.resolve(required(options, "generated-dir"));
const version = required(options, "version");
const assetDir = path.resolve(options["asset-dir"] || path.join(repoRoot, "static/video2mesh/web-demo/assets"));
const manifestPath = path.join(assetDir, "web-demo-assets.json");
const chunkDir = path.join(assetDir, "chunks");
const chunkSize = Number(options["chunk-size"] || 1024 * 1024);
const objectLimit = Number(options["max-object-gaussians"] || 120000);
const carveDistance = Number(options["carve-distance"] || 0.18);
if (![chunkSize, objectLimit].every((value) => Number.isInteger(value) && value > 0)) {
  throw new Error("chunk size and object Gaussian limit must be positive integers");
}
if (!(carveDistance > 0)) throw new Error("carve distance must be positive");

const existingManifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const scenePly = readBinaryPly(scenePath);
const objects = OBJECT_SPECS.map((spec) => {
  const sourcePath = path.join(sourceDir, `${spec.id}.ply`);
  const generatedPath = path.join(generatedDir, `${spec.id}_trellis_gaussian.ply`);
  const anchor = readAsciiPointPly(sourcePath);
  const generated = readBinaryPly(generatedPath);
  const sourceBounds = robustBoundsFromPositions(anchor.positions, 0.01, 0.99);
  const generatedBounds = robustGaussianBounds(generated, 0.01, 0.99);
  const scale = sourceBounds.extent.map((extent, axis) => extent / generatedBounds.extent[axis]);
  return {
    ...spec,
    sourcePath,
    generatedPath,
    sourceBounds,
    generatedBounds,
    scale,
    anchor,
    anchorHash: buildSpatialHash(anchor, carveDistance),
    carveBounds: expandedBounds(sourceBounds, 0.08, carveDistance),
    generated,
  };
});

const carve = carveScene(scenePly, objects, carveDistance);
const carvedScene = rewriteVertexPly(scenePly, carve.selected);
const carvedSceneBounds = exactBounds(scenePly, carve.selected);
const tempChunkDir = path.join(assetDir, `.interactive-chunks-next-${process.pid}`);
fs.rmSync(tempChunkDir, { recursive: true, force: true });
fs.mkdirSync(tempChunkDir, { recursive: true });

try {
  const visual = buildChunkedAsset({
    bytes: carvedScene.bytes,
    prefix: "visual_bedroom_4_pgsr_static_carved_ply",
    directory: tempChunkDir,
    chunkSize,
    headerByteLength: carvedScene.headerByteLength,
    vertexStride: scenePly.vertexStride,
    vertexProperties: scenePly.vertexProperties,
    bbox: carvedSceneBounds,
    fields: {
      ...existingManifest.assets.visual,
      id: "bedroom_4_pgsr_static_carved_interactive_objects_ply",
      label: "Bedroom 4 PGSR static Gaussian layer with recognized objects carved",
      fileName: "bedroom_4_pgsr_static_carved.ply",
      vertexCount: carve.selected.length,
      sourcePath: scenePath,
      carvedObjectIds: objects.map((object) => object.id),
      removedVertexCount: carve.removedCount,
      removedByObject: carve.removedByObject,
      carveMethod: "semantic_anchor_radius_nearest_neighbor",
      carveDistance,
    },
  });

  const interactiveObjects = [];
  for (const object of objects) {
    const selectedIndices = selectedGaussianIndices(object.generated, objectLimit);
    const lod = rewritePlacedGaussianPly(
      object.generated,
      selectedIndices,
      object.generatedBounds.center,
      object.scale
    );
    const prefix = `interactive_${object.id}_gaussian_lod`;
    const visualAsset = buildChunkedAsset({
      bytes: lod.bytes,
      prefix,
      directory: tempChunkDir,
      chunkSize,
      headerByteLength: lod.headerByteLength,
      vertexStride: object.generated.vertexStride,
      vertexProperties: object.generated.vertexProperties,
      bbox: lod.bbox,
      fields: {
        id: `${object.id}_trellis_gaussian_web_lod`,
        label: `${object.label} TRELLIS Gaussian web LOD`,
        fileName: `${object.id}_trellis_gaussian_web_lod.ply`,
        fileType: "ply",
        format: "graphdeco-gaussian-ply",
        shDegree: 0,
        vertexCount: selectedIndices.length,
        sourceVertexCount: object.generated.vertexCount,
        sourcePath: object.generatedPath,
        sourceSha256: sha256(object.generated.bytes),
        selectionMethod: "deterministic_even_sample_positive_opacity_first_plus_affine_covariance_bake",
      },
    });
    interactiveObjects.push({
      id: object.id,
      label: object.label,
      category: object.category,
      fidelity: object.fidelity || "generated_completion_from_scanned_instance_evidence",
      visual: visualAsset,
      sourceAnchor: {
        path: object.sourcePath,
        sha256: object.anchor.sha256,
        pointCount: object.anchor.vertexCount,
        robustBounds: {
          min: roundedArray(object.sourceBounds.min),
          max: roundedArray(object.sourceBounds.max),
          center: roundedArray(object.sourceBounds.center),
          extent: roundedArray(object.sourceBounds.extent),
          quantileLow: object.sourceBounds.quantileLow,
          quantileHigh: object.sourceBounds.quantileHigh,
        },
      },
      placement: {
        method: "robust_axis_aligned_extent_fit_baked_into_gaussians",
        coordinateFrame: "visual_native",
        pivot: roundedArray(object.sourceBounds.center),
        generatedCenter: [0, 0, 0],
        scale: [1, 1, 1],
        rotationEulerDeg: [0, 0, 0],
        eulerOrder: "XYZ",
        bakedSourceGeneratedCenter: roundedArray(object.generatedBounds.center),
        bakedSourceScale: roundedArray(object.scale),
        bakedCovarianceMethod: "axis_scale_times_covariance_then_symmetric_eigendecomposition",
        note: "The TRELLIS export and PGSR scene both use -Y world-up. Non-uniform extent fitting is baked into Gaussian centers and covariances so Spark uses a runtime identity scale.",
      },
      colliderProxy: {
        type: "box",
        dimensions: roundedArray(object.sourceBounds.extent.map((value) => Math.max(0.08, value))),
        center: [0, 0, 0],
        provenance: "robust source-anchor bounds; proxy mesh, not a reconstructed object surface",
      },
      carve: {
        method: "semantic_anchor_radius_nearest_neighbor",
        distance: carveDistance,
        removedSceneGaussianCount: carve.removedByObject[object.id],
      },
      interaction: { kind: "spin", degrees: 360, durationMs: 1250 },
    });
  }

  const manifest = {
    ...existingManifest,
    version,
    chunkSize,
    assets: { ...existingManifest.assets, visual },
    interactiveObjects,
    interactiveObjectBuild: {
      schemaVersion: 2,
      method: "semantic_anchor_carve_plus_affine_baked_trellis_component_lod",
      staticSceneSource: scenePath,
      recognizedObjectCount: interactiveObjects.length,
      unrecognizedSceneGaussiansRemainStatic: true,
      originalSceneGaussianCount: scenePly.vertexCount,
      staticSceneGaussianCount: carve.selected.length,
      removedSceneGaussianCount: carve.removedCount,
      removedByObject: carve.removedByObject,
      objectGaussianCount: interactiveObjects.reduce((total, object) => total + object.visual.vertexCount, 0),
      carveDistance,
      maxObjectGaussians: objectLimit,
      objectPlacementBakedIntoPly: true,
    },
  };

  fs.mkdirSync(chunkDir, { recursive: true });
  for (const name of fs.readdirSync(chunkDir)) {
    if (name.startsWith("visual_") || name.startsWith("interactive_")) fs.unlinkSync(path.join(chunkDir, name));
  }
  for (const name of fs.readdirSync(tempChunkDir)) {
    fs.renameSync(path.join(tempChunkDir, name), path.join(chunkDir, name));
  }
  const nextManifestPath = `${manifestPath}.next-${process.pid}`;
  fs.writeFileSync(nextManifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  fs.renameSync(nextManifestPath, manifestPath);

  console.log(JSON.stringify({
    version,
    originalSceneGaussians: scenePly.vertexCount,
    staticSceneGaussians: carve.selected.length,
    removedSceneGaussians: carve.removedCount,
    removedByObject: carve.removedByObject,
    interactiveObjects: interactiveObjects.map((object) => ({
      id: object.id,
      gaussians: object.visual.vertexCount,
      scale: object.placement.scale,
      bakedScale: object.placement.bakedSourceScale,
      pivot: object.placement.pivot,
      proxy: object.colliderProxy.dimensions,
    })),
  }, null, 2));
} finally {
  fs.rmSync(tempChunkDir, { recursive: true, force: true });
}
