/**
 * Benchmark 02 — Dataset Loading & Generation
 * Deepbox vs scikit-learn
 */

import {
  loadBreastCancer,
  loadDiabetes,
  loadDigits,
  loadIris,
  loadLinnerud,
  makeBlobs,
  makeCircles,
  makeClassification,
  makeGaussianQuantiles,
  makeMoons,
  makeRegression,
} from "deepbox/datasets";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("datasets");
header("Benchmark 02 — Dataset Loading & Generation");

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

footer(suite, "deepbox-datasets.json");
