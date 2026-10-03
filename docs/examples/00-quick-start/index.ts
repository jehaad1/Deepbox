/**
 * Quick Start
 *
 * A short tour of tensors, DataFrames and a first machine learning model.
 * Run this first to get a feel for the library.
 */

import { DataFrame } from "deepbox/dataframe";
import { LinearRegression } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { trainTestSplit } from "deepbox/preprocess";

console.log("Deepbox quick start\n");

// 1. Tensors (N-dimensional arrays)
console.log("1. Tensors:");
const a = tensor([1, 2, 3, 4, 5]);
const b = tensor([10, 20, 30, 40, 50]);
// Tensor methods chain, so a.add(b).mul(2) reads left to right.
console.log("   a + b =", a.add(b).toString());
console.log("   (a + b) * 2 =", a.add(b).mul(2).toString());
console.log(`   mean(a) = ${a.mean().item()}\n`);

// 2. DataFrames (tabular data)
console.log("2. DataFrames:");
const df = new DataFrame({
  name: ["Alice", "Bob", "Charlie"],
  age: [25, 30, 35],
  score: [85, 90, 78],
});
console.log(`${df.toString()}\n`);

// 3. Machine learning
console.log("3. Machine learning:");

// Generate simple data: y = 2x + 1
const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
const y = tensor([3, 5, 7, 9, 11, 13, 15, 17]);

const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, {
  testSize: 0.25,
  randomState: 42,
});

const model = new LinearRegression();
model.fit(XTrain, yTrain);
const predictions = model.predict(XTest);

console.log("   Trained a linear regression model");
console.log("   Predictions:", predictions.toString());
console.log("   Actual:     ", yTest.toString());

console.log("\nThe numbered examples (01 to 49) each cover one area in more detail.\n");
