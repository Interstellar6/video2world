import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "web/public/test-fixtures/tiny-standard-gaussian.ply");
const points = [];
for (let ix = 0; ix < 7; ix += 1) {
  for (let iy = 0; iy < 5; iy += 1) {
    for (let iz = 0; iz < 3; iz += 1) {
      if (ix > 0 && ix < 6 && iy > 0 && iy < 4 && iz > 0 && iz < 2) continue;
      points.push([
        -0.9 + ix * 0.3,
        -0.5 + iy * 0.25,
        -0.25 + iz * 0.25,
      ]);
    }
  }
}

const properties = [
  "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
  ...Array.from({ length: 45 }, (_, index) => `f_rest_${index}`),
  "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
];
const header = [
  "ply",
  "format binary_little_endian 1.0",
  "comment Tiny deterministic standard GraphDECO Gaussian fixture",
  `element vertex ${points.length}`,
  ...properties.map((name) => `property float ${name}`),
  "end_header",
  "",
].join("\n");
const stride = properties.length * 4;
const body = Buffer.alloc(points.length * stride);
const sh0 = (0.86 - 0.5) / 0.28209479177387814;
const opacity = Math.log(0.97 / 0.03);
const scale = Math.log(0.115);
points.forEach((point, pointIndex) => {
  const values = [
    ...point,
    0, 0, 0,
    sh0, sh0, sh0,
    ...Array(45).fill(0),
    opacity,
    scale, scale, scale,
    1, 0, 0, 0,
  ];
  values.forEach((value, propertyIndex) => {
    body.writeFloatLE(value, pointIndex * stride + propertyIndex * 4);
  });
});
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, Buffer.concat([Buffer.from(header, "ascii"), body]));
process.stdout.write(`${output}\n${points.length} Gaussians\n`);
