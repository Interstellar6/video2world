import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const repoRoot = fileURLToPath(new URL(".", import.meta.url));
const workspaceRoot = fileURLToPath(new URL("..", import.meta.url));

export default defineConfig({
  root: "web",
  publicDir: "public",
  cacheDir: fileURLToPath(new URL("./.vite-cache", import.meta.url)),
  resolve: {
    dedupe: ["three"],
  },
  server: {
    host: "127.0.0.1",
    fs: {
      allow: [repoRoot, workspaceRoot],
    },
  },
  build: {
    outDir: "../dist/web",
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: {
        index: fileURLToPath(new URL("./web/index.html", import.meta.url)),
        objectReview: fileURLToPath(new URL("./web/object-review.html", import.meta.url)),
        gaussianReview: fileURLToPath(new URL("./web/gaussian-review.html", import.meta.url)),
      },
    },
  },
});
