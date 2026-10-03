/**
 * Example 23: Cross-Validation
 *
 * Cross-validation scores a model on several different train/test splits, so
 * one lucky or unlucky split does not decide the result. This example shows the
 * splitters (KFold, StratifiedKFold, LeaveOneOut) and crossValScore, which runs
 * the whole loop for you.
 */

import { crossValScore, LinearRegression } from "deepbox/ml";
import { arange, tensor } from "deepbox/ndarray";
import { KFold, LeaveOneOut, StratifiedKFold } from "deepbox/preprocess";
import { rand, setSeed } from "deepbox/random";

// Synthetic linear data: y = 2x + 3 + noise in [-0.5, 0.5). The seed makes it repeatable.
setSeed(42);
const X = arange(0, 50).div(5).reshape([50, 1]);
const y = X.reshape([50])
  .mul(2)
  .add(3)
  .add(rand([50]).sub(0.5));

console.log(`Dataset: ${X.shape[0]} samples\n`);

// 1. K-Fold: split the rows into k folds. Each fold is the test set once.
console.log("1. K-Fold Cross-Validation (k=5):");
console.log("-".repeat(50));

const kfold = new KFold({ nSplits: 5, shuffle: true, randomState: 42 });

let foldNum = 1;
for (const { trainIndex, testIndex } of kfold.split(X)) {
  // The splitter yields row indices. Use them to pick rows, for example with X.gather(...).
  console.log(`Fold ${foldNum}: Train=${trainIndex.length}, Test=${testIndex.length}`);
  foldNum++;
}

console.log(`\nTotal folds: ${kfold.getNSplits()}\n`);

// 2. Stratified K-Fold: every fold keeps the class proportions of y.
console.log("2. Stratified K-Fold:");
console.log("-".repeat(50));

// 12 samples, 3 classes with 4 samples each
const yClass = tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
const XClass = arange(0, 12).reshape([12, 1]);

const stratified = new StratifiedKFold({
  nSplits: 2,
  shuffle: true,
  randomState: 42,
});

const labels = yClass.toArray() as number[];
foldNum = 1;
for (const { trainIndex, testIndex } of stratified.split(XClass, yClass)) {
  // Count how many test rows belong to each class.
  const counts = [0, 1, 2].map((c) => testIndex.filter((i) => labels[i] === c).length);
  console.log(
    `Fold ${foldNum}: Train=${trainIndex.length}, Test=${testIndex.length}, test rows per class=[${counts}]`
  );
  foldNum++;
}

console.log("\nEach test fold holds the same number of rows from every class\n");

// 3. Leave-One-Out: n folds, each tests a single row.
console.log("3. Leave-One-Out Cross-Validation:");
console.log("-".repeat(50));

const XSmall = tensor([[1], [2], [3], [4], [5]]);
const loo = new LeaveOneOut();
const looFolds = Array.from(loo.split(XSmall));

console.log(`Total folds: ${looFolds.length}`);
console.log("Each fold trains on n-1 samples and tests on 1");
console.log("Uses the most training data, but fits the model n times\n");

// 4. crossValScore: fit and score a model on every fold in one call.
console.log("4. crossValScore:");
console.log("-".repeat(50));

// For a regressor the score is R². The folds come from a fixed-seed shuffle.
const scores = crossValScore(new LinearRegression(), X, y, 5);
console.log(`Scores per fold: ${scores.map((s) => s.toFixed(4)).join(", ")}`);
const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
console.log(`Mean R²: ${meanScore.toFixed(4)}\n`);

console.log("Summary:");
console.log("  K-Fold: a good default, 5 or 10 folds");
console.log("  Stratified K-Fold: keeps class proportions, use it for classification");
console.log("  Leave-One-Out: maximum training data per fold, high variance, slow on large data");
