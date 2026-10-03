/**
 * Example 46: Dataset Transforms & Samplers
 *
 * Dataset helpers around a training loop: randomSplit, Subset, filterDataset,
 * mapDataset, and DataLoader batches drawn by a WeightedRandomSampler or a
 * SubsetRandomSampler. The sampler seeds make every run give the same result.
 */

import {
  DataLoader,
  filterDataset,
  loadIris,
  mapDataset,
  randomSplit,
  Subset,
  SubsetRandomSampler,
  WeightedRandomSampler,
} from "deepbox/datasets";

console.log("=".repeat(60));
console.log("Example 46: Dataset Transforms & Samplers");
console.log("=".repeat(60));

const iris = loadIris();

// ============================================================================
// Part 1: Seeded dataset splitting
// ============================================================================
console.log("\nPart 1: randomSplit");
console.log("-".repeat(60));

const [trainSplit, validationSplit, testSplit] = randomSplit(iris, [105, 30, 15], 42); // sizes, then the seed

console.log(`Dataset description: ${iris.description}`);
console.log(
  `Split sizes: train ${trainSplit.data.shape[0]}, validation ${validationSplit.data.shape[0]}, test ${testSplit.data.shape[0]}`
);
console.log(`First five train indices: ${trainSplit.indices.slice(0, 5).join(", ")}`);

// ============================================================================
// Part 2: Explicit subsets
// ============================================================================
console.log("\nPart 2: Subset");
console.log("-".repeat(60));

const reviewSubset = new Subset(iris, [0, 10, 20, 50, 60, 120]);
console.log(`Review subset rows: ${reviewSubset.data.shape[0]}`);
console.log(`Indices into the original dataset: ${reviewSubset.indices.join(", ")}`);

// ============================================================================
// Part 3: Filtering and mapping
// ============================================================================
console.log("\nPart 3: filterDataset + mapDataset");
console.log("-".repeat(60));

const binaryIris = filterDataset(iris, (_row, target) => target !== 2);
console.log(`Binary subset (classes 0 and 1 only): ${binaryIris.data.shape[0]} samples`);

const centeredBinaryIris = mapDataset(binaryIris, (row, target) => ({
  data: row.map((value, index) => (index < 2 ? value - 5 : value)),
  target,
}));

console.log("First mapped sample (first two features centered around 5):");
console.log(
  `  Original: [${Array.from({ length: 4 }, (_, i) => Number(binaryIris.data.at(0, i)).toFixed(2)).join(", ")}]`
);
console.log(
  `  Mapped:   [${Array.from({ length: 4 }, (_, i) => Number(centeredBinaryIris.data.at(0, i)).toFixed(2)).join(", ")}]`
);

// ============================================================================
// Part 4: WeightedRandomSampler to rebalance classes
// ============================================================================
console.log("\nPart 4: WeightedRandomSampler");
console.log("-".repeat(60));

// Make the classes unequal: all 50 setosa rows (class 0), but only the versicolor
// rows (class 1) with sepal length of at least 6.3
const skewedIris = filterDataset(binaryIris, (row, target) => target === 0 || (row[0] ?? 0) >= 6.3);
const skewedLabels = skewedIris.target.toArray() as number[];
const count0 = skewedLabels.filter((label) => label === 0).length;
const count1 = skewedLabels.length - count0;
console.log(`Skewed dataset: class0=${count0}, class1=${count1}`);

// A sample's weight is the inverse of its class size, so each class is drawn about equally often
const weights = skewedLabels.map((label) => (label === 0 ? 1 / count0 : 1 / count1));

const balancedSampler = new WeightedRandomSampler(weights, {
  numSamples: 24,
  replacement: true,
  seed: 7,
});

const balancedLoader = new DataLoader(skewedIris.data, skewedIris.target, {
  batchSize: 6,
  sampler: balancedSampler,
});

let sampledClass0 = 0;
let sampledClass1 = 0;
let batchNumber = 1;

for (const [xBatch, yBatch] of balancedLoader) {
  const labels = yBatch.toArray() as number[];
  const batchClass0 = labels.filter((label) => label === 0).length;
  const batchClass1 = labels.length - batchClass0;
  sampledClass0 += batchClass0;
  sampledClass1 += batchClass1;

  console.log(
    `  Batch ${batchNumber}: X${JSON.stringify(xBatch.shape)} | class0=${batchClass0}, class1=${batchClass1}`
  );
  batchNumber++;
}

console.log(`Drawn in total: class0=${sampledClass0}, class1=${sampledClass1}`);

// ============================================================================
// Part 5: SubsetRandomSampler (a shuffled pass over chosen rows)
// ============================================================================
console.log("\nPart 5: SubsetRandomSampler");
console.log("-".repeat(60));

const shortlistSampler = new SubsetRandomSampler([0, 25, 50, 75, 100, 125], { seed: 19 });
const shortlistLoader = new DataLoader(iris.data, iris.target, {
  batchSize: 3,
  sampler: shortlistSampler,
});

let shortlistBatch = 1;
for (const [xBatch, yBatch] of shortlistLoader) {
  console.log(
    `  Review batch ${shortlistBatch}: X${JSON.stringify(xBatch.shape)}, labels=${(yBatch.toArray() as number[]).join(", ")}`
  );
  shortlistBatch++;
}

// ============================================================================
// Summary
// ============================================================================
console.log("\nKey Takeaways");
console.log("-".repeat(60));
console.log("• randomSplit: a seeded split into any number of parts");
console.log(
  "• Subset: a fixed list of rows, for example a review queue or a frozen evaluation set"
);
console.log(
  "• filterDataset, mapDataset: keep or rewrite samples without editing the source dataset"
);
console.log("• WeightedRandomSampler: draws rows by weight, which rebalances skewed labels");
console.log("• SubsetRandomSampler: a shuffled pass over chosen rows, repeatable with a seed");

console.log("\nDataset Transforms & Samplers Example Complete!");
console.log("=".repeat(60));
