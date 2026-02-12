/**
 * Example 31: DataLoader — Batching & Shuffling
 *
 * Demonstrates the DataLoader class for efficient batch iteration over datasets.
 * Essential for training loops where data must be batched and optionally shuffled.
 */

import { DataLoader } from "deepbox/datasets";
import { tensor } from "deepbox/ndarray";

console.log("=== DataLoader: Batching & Shuffling ===\n");

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
console.log(`Batch size: 3`);
console.log(`Expected batches: 4 (last batch has 1 sample)\n`);

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

console.log("\nFirst iteration:");
for (const [xBatch, yBatch] of shuffledLoader) {
  console.log(`  X first row: ${xBatch.toString().split("\n")[0]}, y: ${yBatch.toString()}`);
}

console.log("\nSecond iteration (same seed = same order):");
for (const [xBatch, yBatch] of shuffledLoader) {
  console.log(`  X first row: ${xBatch.toString().split("\n")[0]}, y: ${yBatch.toString()}`);
}

// ---------------------------------------------------------------------------
// Part 3: dropLast — discard incomplete final batch
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
