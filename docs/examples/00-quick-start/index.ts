/**
 * Quick Start Guide
 *
 * A rapid introduction to Deepbox's core features.
 * Run this first to get a feel for the library!
 */

import { DataFrame } from "deepbox/dataframe";
import { LinearRegression } from "deepbox/ml";
import { add, mean, tensor } from "deepbox/ndarray";
import { trainTestSplit } from "deepbox/preprocess";

console.log("🚀 Welcome to Deepbox!\n");

// 1. Tensors (N-dimensional arrays)
console.log("1️⃣  Tensors:");
const a = tensor([1, 2, 3, 4, 5]);
const b = tensor([10, 20, 30, 40, 50]);
const c = add(a, b);
console.log("   a + b =", c.toString());
console.log("   mean(a) =", `${mean(a).toString()}\n`);

// 2. DataFrames (tabular data)
console.log("2️⃣  DataFrames:");
const df = new DataFrame({
  name: ["Alice", "Bob", "Charlie"],
  age: [25, 30, 35],
  score: [85, 90, 78],
});
console.log(`${df.toString()}\n`);

// 3. Machine Learning
console.log("3️⃣  Machine Learning:");

// Generate simple data: y = 2x + 1
const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
const y = tensor([3, 5, 7, 9, 11, 13, 15, 17]);

const [X_train, X_test, y_train, y_test] = trainTestSplit(X, y, {
  testSize: 0.25,
  randomState: 42,
});

const model = new LinearRegression();
model.fit(X_train, y_train);
const predictions = model.predict(X_test);

console.log("   Trained linear regression model");
console.log("   Predictions:", predictions.toString());
console.log("   Actual:     ", y_test.toString());

console.log("\n✨ That's Deepbox in a nutshell!");
console.log("📖 Explore the numbered examples (01-32) to learn more.");
console.log("💡 Each example focuses on a specific feature with detailed comments.\n");
