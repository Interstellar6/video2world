import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { validateWebManifest } from "../../web/web-manifest.js";

const fixturePath = path.resolve("web/public/test-fixtures/manifest.json");

function fixture() {
  return JSON.parse(fs.readFileSync(fixturePath, "utf8"));
}

describe("Web manifest contract", () => {
  it("accepts the committed fresh-clone fixture", () => {
    expect(validateWebManifest(fixture()).version).toBe("browser-fixture-v1");
  });

  it("rejects an unknown schema version", () => {
    const manifest = fixture();
    manifest.schemaVersion = 2;
    expect(() => validateWebManifest(manifest)).toThrow("schemaVersion must equal 1");
  });

  it("rejects duplicate object identities", () => {
    const manifest = fixture();
    manifest.interactiveObjects.push(structuredClone(manifest.interactiveObjects[0]));
    expect(() => validateWebManifest(manifest)).toThrow("duplicates");
  });

  it("rejects a collider on an explicit visual-only object", () => {
    const manifest = fixture();
    const pillow = manifest.interactiveObjects.find((item) => item.id === "sam3_pillow_01");
    pillow.collision.asset = structuredClone(manifest.interactiveObjects[0].collision.asset);
    expect(() => validateWebManifest(manifest)).toThrow("must not declare collision.asset");
  });

  it("accepts only an explicitly marked non-loadable baseline collider", () => {
    const manifest = fixture();
    delete manifest.assets.collider.url;
    manifest.assets.collider.parts = [];
    manifest.assets.collider.baselineMetadataOnly = true;
    expect(validateWebManifest(manifest).assets.collider.baselineMetadataOnly).toBe(true);

    delete manifest.assets.collider.baselineMetadataOnly;
    expect(() => validateWebManifest(manifest)).toThrow("explicit metadata-only record");
  });
});
