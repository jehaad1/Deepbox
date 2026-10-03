/**
 * Benchmark 02: Dataset Loading & Generation
 * Deepbox vs scikit-learn
 */

import {
  DataLoader,
  filterDataset,
  loadBreastCancer,
  loadCustomerSegments,
  loadDiabetes,
  loadDigits,
  loadIris,
  loadLinnerud,
  makeBiclusters,
  makeBlobs,
  makeCheckerboard,
  makeCircles,
  makeClassification,
  makeFriedman1,
  makeFriedman2,
  makeFriedman3,
  makeGaussianQuantiles,
  makeLowRankMatrix,
  makeMoons,
  makeRegression,
  makeSCurve,
  makeSPDMatrix,
  makeSparseUncorrelated,
  makeSwissRoll,
  mapDataset,
  parseCSV,
  randomSplit,
  WeightedRandomSampler,
} from "deepbox/datasets";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("datasets");
header("Benchmark 02: Dataset Loading & Generation");

// ── Built-in Loaders ────────────────────────────────────

run(suite, "loadIris", "150x4", () => loadIris());
run(suite, "loadBreastCancer", "569x30", () => loadBreastCancer());
run(suite, "loadDiabetes", "442x10", () => loadDiabetes());
run(suite, "loadDigits", "1797x64", () => loadDigits(), { iterations: 10 });
run(suite, "loadLinnerud", "20x3", () => loadLinnerud());

// ── makeBlobs ───────────────────────────────────────────

run(suite, "makeBlobs", "100x2 k=3", () => makeBlobs({ nSamples: 100, nFeatures: 2, centers: 3 }));
run(suite, "makeBlobs", "500x2 k=3", () => makeBlobs({ nSamples: 500, nFeatures: 2, centers: 3 }));
run(suite, "makeBlobs", "1Kx5 k=5", () => makeBlobs({ nSamples: 1000, nFeatures: 5, centers: 5 }));
run(suite, "makeBlobs", "2Kx10 k=5", () =>
  makeBlobs({ nSamples: 2000, nFeatures: 10, centers: 5 })
);
run(suite, "makeBlobs", "5Kx2 k=3", () => makeBlobs({ nSamples: 5000, nFeatures: 2, centers: 3 }));
run(suite, "makeBlobs", "10Kx5 k=10", () =>
  makeBlobs({ nSamples: 10000, nFeatures: 5, centers: 10 })
);
run(suite, "makeBlobs", "20Kx20 k=10", () =>
  makeBlobs({ nSamples: 20000, nFeatures: 20, centers: 10 })
);

// ── makeCircles ─────────────────────────────────────────

run(suite, "makeCircles", "100 samples", () => makeCircles({ nSamples: 100 }));
run(suite, "makeCircles", "500 samples", () => makeCircles({ nSamples: 500 }));
run(suite, "makeCircles", "1K samples", () => makeCircles({ nSamples: 1000 }));
run(suite, "makeCircles", "5K samples", () => makeCircles({ nSamples: 5000 }));
run(suite, "makeCircles", "10K noise=0.1", () => makeCircles({ nSamples: 10000, noise: 0.1 }));
run(suite, "makeCircles", "20K noise=0.05", () => makeCircles({ nSamples: 20000, noise: 0.05 }));

// ── makeMoons ───────────────────────────────────────────

run(suite, "makeMoons", "100 samples", () => makeMoons({ nSamples: 100 }));
run(suite, "makeMoons", "500 samples", () => makeMoons({ nSamples: 500 }));
run(suite, "makeMoons", "1K samples", () => makeMoons({ nSamples: 1000 }));
run(suite, "makeMoons", "5K samples", () => makeMoons({ nSamples: 5000 }));
run(suite, "makeMoons", "10K noise=0.1", () => makeMoons({ nSamples: 10000, noise: 0.1 }));
run(suite, "makeMoons", "20K noise=0.05", () => makeMoons({ nSamples: 20000, noise: 0.05 }));

// ── makeClassification ──────────────────────────────────

run(suite, "makeClassification", "100x10", () =>
  makeClassification({ nSamples: 100, nFeatures: 10 })
);
run(suite, "makeClassification", "500x10", () =>
  makeClassification({ nSamples: 500, nFeatures: 10 })
);
run(suite, "makeClassification", "1Kx20", () =>
  makeClassification({ nSamples: 1000, nFeatures: 20 })
);
run(suite, "makeClassification", "5Kx20", () =>
  makeClassification({ nSamples: 5000, nFeatures: 20 })
);
run(suite, "makeClassification", "10Kx50", () =>
  makeClassification({ nSamples: 10000, nFeatures: 50 })
);
run(
  suite,
  "makeClassification",
  "20Kx100",
  () => makeClassification({ nSamples: 20000, nFeatures: 100 }),
  { iterations: 10 }
);

// ── makeRegression ──────────────────────────────────────

run(suite, "makeRegression", "100x10", () => makeRegression({ nSamples: 100, nFeatures: 10 }));
run(suite, "makeRegression", "500x10", () => makeRegression({ nSamples: 500, nFeatures: 10 }));
run(suite, "makeRegression", "1Kx20", () => makeRegression({ nSamples: 1000, nFeatures: 20 }));
run(suite, "makeRegression", "5Kx20", () => makeRegression({ nSamples: 5000, nFeatures: 20 }));
run(suite, "makeRegression", "10Kx50", () => makeRegression({ nSamples: 10000, nFeatures: 50 }));
run(suite, "makeRegression", "20Kx100", () => makeRegression({ nSamples: 20000, nFeatures: 100 }), {
  iterations: 10,
});

// ── makeGaussianQuantiles ───────────────────────────────

run(suite, "makeGaussianQuantiles", "100x2 k=3", () =>
  makeGaussianQuantiles({ nSamples: 100, nFeatures: 2, nClasses: 3 })
);
run(suite, "makeGaussianQuantiles", "500x5 k=3", () =>
  makeGaussianQuantiles({ nSamples: 500, nFeatures: 5, nClasses: 3 })
);
run(suite, "makeGaussianQuantiles", "1Kx5 k=5", () =>
  makeGaussianQuantiles({ nSamples: 1000, nFeatures: 5, nClasses: 5 })
);
run(suite, "makeGaussianQuantiles", "5Kx10 k=5", () =>
  makeGaussianQuantiles({ nSamples: 5000, nFeatures: 10, nClasses: 5 })
);
run(suite, "makeGaussianQuantiles", "10Kx10 k=5", () =>
  makeGaussianQuantiles({ nSamples: 10000, nFeatures: 10, nClasses: 5 })
);

// ── Additional v1.0.0 generators ───────────────────────

run(suite, "makeFriedman1", "1Kx10", () =>
  makeFriedman1({ nSamples: 1000, nFeatures: 10, noise: 1.0, randomState: 42 })
);
run(suite, "makeFriedman2", "1K", () =>
  makeFriedman2({ nSamples: 1000, noise: 1.0, randomState: 42 })
);
run(suite, "makeFriedman3", "1K", () =>
  makeFriedman3({ nSamples: 1000, noise: 1.0, randomState: 42 })
);
run(suite, "makeSwissRoll", "2K", () =>
  makeSwissRoll({ nSamples: 2000, noise: 0.2, randomState: 42 })
);
run(suite, "makeSCurve", "2K", () => makeSCurve({ nSamples: 2000, noise: 0.2, randomState: 42 }));
run(suite, "makeSparseUncorrelated", "2Kx20", () =>
  makeSparseUncorrelated({ nSamples: 2000, nFeatures: 20, randomState: 42 })
);
run(suite, "makeLowRankMatrix", "500x100 rank=10", () =>
  makeLowRankMatrix({ nSamples: 500, nFeatures: 100, effectiveRank: 10, randomState: 42 })
);
run(suite, "makeSPDMatrix", "100x100", () => makeSPDMatrix({ nDim: 100, randomState: 42 }));
run(suite, "makeBiclusters", "200x200 k=4", () =>
  makeBiclusters({ shape: [200, 200], nClusters: 4, noise: 0.1, randomState: 42 })
);
run(suite, "makeCheckerboard", "200x200 k=4x4", () =>
  makeCheckerboard({ shape: [200, 200], nClusters: [4, 4], noise: 0.1, randomState: 42 })
);

// ── Deepbox-only dataset coverage ──────────────────────

run(suite, "loadCustomerSegments", "200x3", () => loadCustomerSegments(), {
  comparable: false,
  tags: ["deepbox-only"],
});

const iris = loadIris();
const irisCsv = ["f1,f2,f3,target", "1,2,3,0", "4,5,6,1", "7,8,9,0", "10,11,12,1"].join("\n");

run(suite, "randomSplit", "150→105/30/15", () => randomSplit(iris, [105, 30, 15], 42), {
  comparable: false,
  tags: ["deepbox-only"],
});
run(
  suite,
  "filterDataset",
  "iris target!=2",
  () => filterDataset(iris, (_row, target) => target !== 2),
  { comparable: false, tags: ["deepbox-only"] }
);
run(
  suite,
  "mapDataset",
  "iris center first 2 cols",
  () =>
    mapDataset(iris, (row, target) => ({
      data: row.map((value, index) => (index < 2 ? value - 5 : value)),
      target,
    })),
  { comparable: false, tags: ["deepbox-only"] }
);
run(
  suite,
  "DataLoader iterate",
  "150 batch=16",
  () => {
    const loader = new DataLoader(iris.data, iris.target, {
      batchSize: 16,
      shuffle: true,
      seed: 42,
    });
    let batches = 0;
    for (const [xBatch] of loader) {
      batches += xBatch.shape[0] ?? 0;
    }
    return batches;
  },
  { comparable: false, tags: ["deepbox-only"] }
);
run(
  suite,
  "WeightedRandomSampler",
  "100 weights→64",
  () => {
    const sampler = new WeightedRandomSampler(
      Array.from({ length: 100 }, (_, i) => (i % 5 === 0 ? 3 : 1)),
      { numSamples: 64, replacement: true, seed: 7 }
    );
    return Array.from(sampler).length;
  },
  { comparable: false, tags: ["deepbox-only"] }
);
run(suite, "parseCSV", "4 rows × 3 cols", () => parseCSV(irisCsv), {
  comparable: false,
  tags: ["deepbox-only"],
});

footer(suite, "deepbox-datasets.json");
