/**
 * Example 09: Ridge & Lasso Regression
 *
 * Compare L1 (Lasso) and L2 (Ridge) regularization techniques.
 * Learn when to use each regularization method.
 */

import { loadDiabetes } from "deepbox/datasets";
import { mse, r2Score } from "deepbox/metrics";
import { Lasso, LinearRegression, Ridge } from "deepbox/ml";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";

console.log("=== Ridge & Lasso Regression ===\n");

// Load diabetes dataset for regression
const diabetes = loadDiabetes();
console.log(`Dataset: ${diabetes.data.shape[0]} samples, ${diabetes.data.shape[1]} features\n`);

// Split data into training and testing sets
const [X_train, X_test, y_train, y_test] = trainTestSplit(diabetes.data, diabetes.target, {
  testSize: 0.2,
  randomState: 42,
});

// Scale features
const scaler = new StandardScaler();
scaler.fit(X_train);
const X_train_scaled = scaler.transform(X_train);
const X_test_scaled = scaler.transform(X_test);

console.log("Training models...\n");

// Train different models
const models = [
  { name: "Linear Regression", model: new LinearRegression() },
  { name: "Ridge (α=0.1)", model: new Ridge({ alpha: 0.1 }) },
  // Ridge adds penalty proportional to square of coefficients
  { name: "Ridge (α=1.0)", model: new Ridge({ alpha: 1.0 }) },
  { name: "Ridge (α=10.0)", model: new Ridge({ alpha: 10.0 }) },
  { name: "Lasso (α=0.1)", model: new Lasso({ alpha: 0.1 }) },
  // Lasso adds penalty proportional to absolute value of coefficients
  { name: "Lasso (α=1.0)", model: new Lasso({ alpha: 1.0 }) },
];

// Compare the two models
console.log("\nComparison:");
console.log("-".repeat(50));

for (const { name, model } of models) {
  model.fit(X_train_scaled, y_train);
  const y_pred = model.predict(X_test_scaled);

  const r2 = r2Score(y_test, y_pred);
  const mseValue = mse(y_test, y_pred);

  console.log(`${name.padEnd(25)} R²: ${r2.toFixed(4)}  MSE: ${mseValue.toFixed(2)}`);
}

// Explain when to use each method
console.log("\nKey Differences:");
console.log("• Ridge regression shrinks coefficients smoothly");
console.log("• Lasso can zero out coefficients (feature selection)");

console.log("\n✓ Regularized regression complete!");
