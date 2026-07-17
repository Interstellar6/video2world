import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

export default defineConfig({
  root: "web",
  publicDir: "public",
  server: {
    host: "127.0.0.1",
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
