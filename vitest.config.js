import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/web/**/*.test.js"],
    fileParallelism: false,
    coverage: { enabled: false },
  },
});
