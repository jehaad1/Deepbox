/**
 * Example 07: Linear Regression
 *
 * Fit a linear regression to noisy synthetic data and measure it on a held-out
 * test set. The true relationship is y = 2x + 3, so the fitted coefficient
 * should be close to 2 and the intercept close to 3.
 */

import { mae, mse, r2Score } from "deepbox/metrics";
import { LinearRegression } from "deepbox/ml";
import { arange } from "deepbox/ndarray";
import { trainTestSplit } from "deepbox/preprocess";
import { rand, setSeed } from "deepbox/random";

console.log("=== Linear Regression ===\n");

// Seed the generator so every run produces the same noise.
setSeed(42);

// x runs from 0 to 9.9 in steps of 0.1, shaped [100, 1] as one feature column.
const X = arange(0, 100).div(10).reshape([100, 1]);

// y = 2x + 3 plus uniform noise in [-1, 1).
const noise = rand([100]).sub(0.5).mul(2);
const y = X.reshape([100]).mul(2).add(3).add(noise);

console.log(`Dataset: ${X.shape[0]} samples, ${X.shape[1]} feature column\n`);

// Split data: 80% training, 20% testing
const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, {
  testSize: 0.2,
  randomState: 42,
});

console.log(`Training set: ${XTrain.shape[0]} samples`);
console.log(`Test set: ${XTest.shape[0]} samples\n`);

// Create and train the linear regression model
const model = new LinearRegression();
model.fit(XTrain, yTrain);

console.log("Model trained.");
console.log(`Coefficients: ${model.coef?.toString()}`);
console.log(`Intercept: ${model.intercept}\n`);

// Generate predictions on the test set
const yPred = model.predict(XTest);

// Calculate performance metrics. The metric functions return plain numbers.
const r2 = r2Score(yTest, yPred);
const mseValue = mse(yTest, yPred);
const maeValue = mae(yTest, yPred);

console.log("Model Performance:");
console.log(`R² Score: ${r2.toFixed(4)}`);
console.log(`MSE: ${mseValue.toFixed(4)}`);
console.log(`MAE: ${maeValue.toFixed(4)}`);
