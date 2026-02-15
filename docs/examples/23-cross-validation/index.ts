/**
 * Example 23: Cross-Validation Strategies
 *
 * Learn different cross-validation techniques for robust model evaluation.
 * Essential for assessing model generalization.
 */

import { tensor } from "deepbox/ndarray";
import { KFold, LeaveOneOut, StratifiedKFold } from "deepbox/preprocess";

// Generate synthetic linear data
// Create training data: y = 2x + 3 + noise
const X_data: number[][] = [];
const y_data: number[] = [];

// Populate data arrays with synthetic data
for (let i = 0; i < 50; i++) {
  const x = i / 5;
  const y = 2 * x + 3 + (Math.random() - 0.5);
  X_data.push([x]);
  y_data.push(y);
}

// Convert data to tensors
const X = tensor(X_data);

// Display dataset size
console.log(`Dataset: ${X.shape[0]} samples\n`);

// 1. K-Fold Cross-Validation
// K-Fold: Split data into k equal folds
console.log("1. K-Fold Cross-Validation (k=5):");
console.log("-".repeat(50));

// Create 5-fold cross-validator with shuffling
const kfold = new KFold({ nSplits: 5, shuffle: true, randomState: 42 });

// Initialize fold counter
let foldNum = 1;

// Iterate through each fold
for (const { trainIndex, testIndex } of kfold.split(X)) {
  // Note: In a real scenario, you'd use gather() to index the data
  // For this example, we'll just count the splits
  console.log(`Fold ${foldNum}: Train=${trainIndex.length}, Test=${testIndex.length}`);
  foldNum++;
}

// Display total number of folds
console.log(`\nTotal folds: ${kfold.getNSplits()}\n`);

// 2. Stratified K-Fold (for classification)
// Stratified K-Fold: Preserves class distribution
console.log("2. Stratified K-Fold:");
console.log("-".repeat(50));

// Create classification data with 3 classes
const y_class = tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
const X_class = tensor(
  Array(12)
    .fill(0)
    .map((_, i) => [i])
);

// Stratified split maintains class proportions in each fold
const stratified = new StratifiedKFold({
  nSplits: 3,
  shuffle: true,
  randomState: 42,
});

// Initialize fold counter
foldNum = 1;

// Iterate through each fold
for (const { trainIndex, testIndex } of stratified.split(X_class, y_class)) {
  console.log(`Fold ${foldNum}: Train=${trainIndex.length}, Test=${testIndex.length}`);
  foldNum++;
}

console.log("\nStratified K-Fold preserves class distribution in each fold\n");

// 3. Leave-One-Out Cross-Validation
// Leave-One-Out: Use n-1 samples for training, 1 for testing
console.log("3. Leave-One-Out Cross-Validation:");
console.log("-".repeat(50));

const X_small = tensor([[1], [2], [3], [4], [5]]);

// LOO creates n folds for n samples
const loo = new LeaveOneOut();
const looFolds = Array.from(loo.split(X_small));

console.log(`Total folds: ${looFolds.length}`);
console.log("Each fold uses n-1 samples for training, 1 for testing");
console.log("Useful for small datasets but computationally expensive\n");

// Summary of when to use each method
console.log("Key Insights:");
console.log("• K-Fold: Good balance between bias and variance");
console.log("• Stratified K-Fold: Maintains class distribution (classification)");
console.log("• Leave-One-Out: Maximum training data, high variance");

console.log("\n✓ Cross-validation complete!");
