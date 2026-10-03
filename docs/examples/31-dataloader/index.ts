/**
 * Example 31: DataLoader: Batching & Shuffling
 *
 * DataLoader splits a dataset into batches, with optional shuffling, and is the
 * usual way to feed a training loop. This example covers batching, a seeded
 * shuffle, reshuffling every epoch, dropLast and loading without labels.
 */

import { DataLoader } from "deepbox/datasets";
import { type Tensor, tensor } from "deepbox/ndarray";

console.log("=== DataLoader: Batching & Shuffling ===\n");

// The first column of X is 1, 3, 5, ..., so it identifies each sample
const sampleIds = (xBatch: Tensor): number[] =>
  (xBatch.toArray() as number[][]).map((row) => row[0] ?? Number.NaN);

// ---------------------------------------------------------------------------
// Part 1: Basic batching
// ---------------------------------------------------------------------------
console.log("--- Part 1: Basic Batching ---");

const X = tensor([
  [1, 2],
  [3, 4],
  [5, 6],
  [7, 8],
  [9, 10],
  [11, 12],
  [13, 14],
  [15, 16],
  [17, 18],
  [19, 20],
]);
const y = tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);

const loader = new DataLoader(X, y, { batchSize: 3 });
console.log(`Dataset size: ${X.shape[0]} samples`);
console.log("Batch size: 3");
console.log("Expected batches: 4 (the last batch has 1 sample)\n");

let batchIdx = 0;
for (const [xBatch, yBatch] of loader) {
  console.log(
    `  Batch ${batchIdx}: X shape [${xBatch.shape.join(", ")}], y shape [${yBatch.shape.join(", ")}]`
  );
  batchIdx++;
}

// ---------------------------------------------------------------------------
// Part 2: Shuffling with deterministic seed
// ---------------------------------------------------------------------------
console.log("\n--- Part 2: Shuffled Iteration ---");

const shuffledLoader = new DataLoader(X, y, {
  batchSize: 5,
  shuffle: true,
  seed: 42,
});
console.log("DataLoader(batchSize=5, shuffle=true, seed=42)");

console.log("\nFirst iteration (sample ids per batch):");
for (const [xBatch] of shuffledLoader) {
  console.log(`  [${sampleIds(xBatch).join(", ")}]`);
}

console.log("\nSecond iteration (a seed alone repeats the same order):");
for (const [xBatch] of shuffledLoader) {
  console.log(`  [${sampleIds(xBatch).join(", ")}]`);
}

// reshuffleEachIteration continues one seeded random stream, so every epoch has a new
// order and the whole sequence of epochs is still reproducible.
const epochLoader = new DataLoader(X, y, {
  batchSize: 5,
  shuffle: true,
  seed: 42,
  reshuffleEachIteration: true,
});
console.log("\nWith reshuffleEachIteration: true");
for (let epoch = 0; epoch < 2; epoch++) {
  const order: number[] = [];
  for (const [xBatch] of epochLoader) order.push(...sampleIds(xBatch));
  console.log(`  Epoch ${epoch}: [${order.join(", ")}]`);
}

// ---------------------------------------------------------------------------
// Part 3: dropLast (discard incomplete final batch)
// ---------------------------------------------------------------------------
console.log("\n--- Part 3: Drop Last Batch ---");

const dropLoader = new DataLoader(X, y, {
  batchSize: 3,
  dropLast: true,
});
console.log("DataLoader(batchSize=3, dropLast=true)");
console.log(`Dataset: ${X.shape[0]} samples, batch: 3, dropLast: true`);

let dropBatchCount = 0;
for (const [xBatch] of dropLoader) {
  console.log(`  Batch ${dropBatchCount}: shape [${xBatch.shape.join(", ")}]`);
  dropBatchCount++;
}
console.log(`Total batches: ${dropBatchCount} (incomplete last batch dropped)`);

// ---------------------------------------------------------------------------
// Part 4: Inference without labels
// ---------------------------------------------------------------------------
console.log("\n--- Part 4: Inference Without Labels ---");

const testLoader = new DataLoader(X, undefined, {
  batchSize: 4,
  shuffle: false,
});
console.log("DataLoader(X, undefined, { batchSize: 4 })");

let testBatchIdx = 0;
for (const [xBatch] of testLoader) {
  console.log(`  Batch ${testBatchIdx}: X shape [${xBatch.shape.join(", ")}]`);
  testBatchIdx++;
}

console.log("\n=== DataLoader Complete ===");
