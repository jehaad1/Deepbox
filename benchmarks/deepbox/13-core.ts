/**
 * Benchmark 13: Core Runtime Utilities
 * Deepbox-only local benchmarks
 */

import {
  check_array,
  check_X_y,
  fromJSON,
  Logger,
  normalizeAxes,
  setLogHandler,
  shapeToSize,
  toJSON,
} from "deepbox/core";
import { tensor } from "deepbox/ndarray";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("core");
header("Benchmark 13: Core Runtime Utilities");

const X1k = tensor(
  Array.from({ length: 1000 }, (_, i) =>
    Array.from({ length: 10 }, (_, j) => ((i * 17 + j * 11) % 97) / 10)
  )
);
const y1k = tensor(Array.from({ length: 1000 }, (_, i) => i % 3));

const tensorPayload = {
  __type: "Tensor" as const,
  data: Array.from({ length: 10000 }, (_, i) => i / 10),
  shape: [100, 100] as const,
  dtype: "float64",
};
const tensorJson = toJSON(tensorPayload);

setLogHandler(() => {});

run(suite, "toJSON (Tensor payload)", "100x100", () => toJSON(tensorPayload), {
  comparable: false,
  tags: ["deepbox-only"],
});

run(suite, "fromJSON (Tensor payload)", "100x100", () => fromJSON(tensorJson), {
  comparable: false,
  tags: ["deepbox-only"],
});

run(suite, "shapeToSize", "32x16x8x4", () => shapeToSize([32, 16, 8, 4]), {
  comparable: false,
  tags: ["deepbox-only"],
});

run(suite, "normalizeAxes", "4D [-1,1]", () => normalizeAxes([-1, 1], 4), {
  comparable: false,
  tags: ["deepbox-only"],
});

run(suite, "check_array", "1Kx10", () => check_array(X1k, { ensureNdim: 2 }), {
  comparable: false,
  tags: ["deepbox-only"],
});

run(suite, "check_X_y", "1Kx10 + 1K", () => check_X_y(X1k, y1k), {
  comparable: false,
  tags: ["deepbox-only"],
});

run(
  suite,
  "Logger.debug",
  "1K entries",
  () => {
    const logger = new Logger(3, "bench");
    for (let i = 0; i < 1000; i++) {
      logger.debug(`iteration ${i}`);
    }
    return logger.getEntries().length;
  },
  { comparable: false, tags: ["deepbox-only"] }
);

footer(suite, "deepbox-core.json");
