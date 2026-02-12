/**
 * Example 07: Linear Regression
 *
 * Build a simple linear regression model to predict continuous values.
 * Learn the basics of supervised learning with Deepbox.
 */

import { mae, mse, r2Score } from "deepbox/metrics";
import { LinearRegression } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { trainTestSplit } from "deepbox/preprocess";

console.log("=== Linear Regression ===\n");

// Generate synthetic data for demonstration
// True relationship: y = 2x + 3 + noise
const X_data: number[][] = [];
const y_data: number[] = [];

for (let i = 0; i < 100; i++) {
  const x = i / 10;
  // Add random noise to make it realistic
  const y = 2 * x + 3 + (Math.random() - 0.5) * 2;
  X_data.push([x]);
  y_data.push(y);
}

// Convert to tensors
const X = tensor(X_data);
const y = tensor(y_data);

console.log(`Dataset: ${X.shape[0]} samples, ${X.shape[1]} features\n`);

// Split data: 80% training, 20% testing
const [X_train, X_test, y_train, y_test] = trainTestSplit(X, y, {
  testSize: 0.2,
  randomState: 42,
});

console.log(`Training set: ${X_train.shape[0]} samples`);
console.log(`Test set: ${X_test.shape[0]} samples\n`);

// Create and train the linear regression model
const model = new LinearRegression();
model.fit(X_train, y_train);

console.log("Model trained!");
console.log(`Coefficients: ${model.coef?.toString()}`);
console.log(`Intercept: ${model.intercept}\n`);

// Generate predictions on test set
const y_pred = model.predict(X_test);

// Calculate performance metrics
const r2 = r2Score(y_test, y_pred);
const mseValue = mse(y_test, y_pred);
const maeValue = mae(y_test, y_pred);

console.log("Model Performance:");
console.log(`R² Score: ${r2.toFixed(4)}`);
console.log(`MSE: ${mseValue.toFixed(4)}`);
console.log(`MAE: ${maeValue.toFixed(4)}`);

console.log("\n✓ Linear regression complete!");
