import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "core/index": "src/core/index.ts",
    "ndarray/index": "src/ndarray/index.ts",
    "linalg/index": "src/linalg/index.ts",
    "dataframe/index": "src/dataframe/index.ts",
    "stats/index": "src/stats/index.ts",
    "metrics/index": "src/metrics/index.ts",
    "preprocess/index": "src/preprocess/index.ts",
    "ml/index": "src/ml/index.ts",
    "nn/index": "src/nn/index.ts",
    "optim/index": "src/optim/index.ts",
    "random/index": "src/random/index.ts",
    "plot/index": "src/plot/index.ts",
    "datasets/index": "src/datasets/index.ts",
  },
  format: ["cjs", "esm"],
  dts: true,
  splitting: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  target: "es2024",
});
