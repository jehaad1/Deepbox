/**
 * Example 46: Dataset Transforms & Samplers
 *
 * Demonstrates v1.0.0 dataset utilities that sit around model training:
 * Subset, randomSplit, mapDataset, filterDataset, and sampler-driven DataLoader
 * iteration for balancing or curating batches.
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
// Part 1: Deterministic dataset splitting
// ============================================================================
console.log("\n✂️  Part 1: randomSplit");
console.log("-".repeat(60));

const [trainSplit, validationSplit, testSplit] = randomSplit(iris, [105, 30, 15], 42);

console.log(`Dataset description: ${iris.description}`);
console.log(
  `Split sizes -> train: ${trainSplit.data.shape[0]}, validation: ${validationSplit.data.shape[0]}, test: ${testSplit.data.shape[0]}`
);
console.log(`First five train indices: ${trainSplit.indices.slice(0, 5).join(", ")}`);

// ============================================================================
// Part 2: Explicit curated subsets
// ============================================================================
console.log("\n🎯 Part 2: Subset");
console.log("-".repeat(60));

const reviewSubset = new Subset(iris, [0, 10, 20, 50, 60, 120]);
console.log(`Curated review subset rows: ${reviewSubset.data.shape[0]}`);
console.log(`Curated indices: ${reviewSubset.indices.join(", ")}`);

// ============================================================================
// Part 3: Filtering and mapping
// ============================================================================
console.log("\n🧪 Part 3: filterDataset + mapDataset");
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
// Part 4: WeightedRandomSampler for class balancing
// ============================================================================
console.log("\n⚖️  Part 4: WeightedRandomSampler");
console.log("-".repeat(60));

const weights = Array.from({ length: binaryIris.target.shape[0] ?? 0 }, (_, index) => {
  const label = Number(binaryIris.target.at(index));
  return label === 1 ? 4 : 1;
});

const balancedSampler = new WeightedRandomSampler(weights, {
  numSamples: 24,
  replacement: true,
  seed: 7,
});

const balancedLoader = new DataLoader(binaryIris.data, binaryIris.target, {
  batchSize: 6,
  sampler: balancedSampler,
});

let sampledClass0 = 0;
let sampledClass1 = 0;
let batchNumber = 1;

for (const [xBatch, yBatch] of balancedLoader) {
  let batchClass0 = 0;
  let batchClass1 = 0;

  for (let i = 0; i < (yBatch.shape[0] ?? 0); i++) {
    const label = Number(yBatch.at(i));
    if (label === 0) {
      batchClass0++;
      sampledClass0++;
    } else {
      batchClass1++;
      sampledClass1++;
    }
  }

  console.log(
    `  Batch ${batchNumber}: X${JSON.stringify(xBatch.shape)} | class0=${batchClass0}, class1=${batchClass1}`
  );
  batchNumber++;
}

console.log(`Weighted sampling totals -> class0=${sampledClass0}, class1=${sampledClass1}`);

// ============================================================================
// Part 5: Deterministic review queues via SubsetRandomSampler
// ============================================================================
console.log("\n📋 Part 5: SubsetRandomSampler");
console.log("-".repeat(60));

const shortlistSampler = new SubsetRandomSampler([0, 5, 10, 15, 20, 25], { seed: 19 });
const shortlistLoader = new DataLoader(iris.data, iris.target, {
  batchSize: 3,
  sampler: shortlistSampler,
});

let shortlistBatch = 1;
for (const [xBatch, yBatch] of shortlistLoader) {
  console.log(
    `  Review batch ${shortlistBatch}: X${JSON.stringify(xBatch.shape)}, labels=${Array.from({ length: yBatch.shape[0] ?? 0 }, (_, i) => Number(yBatch.at(i))).join(", ")}`
  );
  shortlistBatch++;
}

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• randomSplit gives deterministic multi-way dataset partitioning");
console.log("• Subset is useful for audits, human review queues, or frozen evaluation slices");
console.log("• filterDataset and mapDataset make lightweight data curation easy");
console.log("• WeightedRandomSampler can rebalance skewed labels without copying data");
console.log("• SubsetRandomSampler lets you batch over a curated slice reproducibly");

console.log("\n✅ Dataset Transforms & Samplers Example Complete!");
console.log("=".repeat(60));
