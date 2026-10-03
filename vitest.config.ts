import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
      include: ["src/**/*.ts"],
      exclude: [
        "src/**/*.test.ts",
        "src/**/index.ts",
        "src/**/types.ts",
        "src/**/types/*.ts",
        "src/core/types/**",
        "src/**/env.d.ts",
        "src/**/*.d.ts",
      ],
      thresholds: {
        lines: 94,
        functions: 96,
        branches: 83,
        statements: 93,
      },
    },
  },
});
