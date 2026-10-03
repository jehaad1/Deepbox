/**
 * Example 09: Ridge & Lasso Regression
 *
 * Compare L2 (Ridge) and L1 (Lasso) regularization against plain linear
 * regression on the diabetes dataset. Ridge shrinks every coefficient toward
 * zero. Lasso can set some coefficients exactly to zero, which removes those
 * features from the model.
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
const [XTrain, XTest, yTrain, yTest] = trainTestSplit(diabetes.data, diabetes.target, {
  testSize: 0.2,
  randomState: 42,
});

// Scale features
const scaler = new StandardScaler();
scaler.fit(XTrain);
const XTrainScaled = scaler.transform(XTrain);
const XTestScaled = scaler.transform(XTest);

console.log("Training models...\n");

// Train different models
const models = [
  { name: "Linear Regression", model: new LinearRegression() },
  { name: "Ridge (α=0.1)", model: new Ridge({ alpha: 0.1 }) },
  { name: "Ridge (α=1.0)", model: new Ridge({ alpha: 1.0 }) },
  { name: "Ridge (α=10.0)", model: new Ridge({ alpha: 10.0 }) },
  { name: "Lasso (α=0.1)", model: new Lasso({ alpha: 0.1 }) },
  { name: "Lasso (α=1.0)", model: new Lasso({ alpha: 1.0 }) },
];

// Ridge penalizes the sum of squared coefficients, Lasso the sum of their absolute values.
// Fit each model and report test R², test MSE and how many coefficients are exactly zero.
console.log("Comparison:");
console.log("-".repeat(50));

for (const { name, model } of models) {
  model.fit(XTrainScaled, yTrain);
  const yPred = model.predict(XTestScaled);

  const r2 = r2Score(yTest, yPred);
  const mseValue = mse(yTest, yPred);

  const zeros = model.coef.eq(0).sum().item();
  console.log(
    `${name.padEnd(20)} R²: ${r2.toFixed(4)}  MSE: ${mseValue.toFixed(2)}  zero coefficients: ${zeros}`
  );
}

console.log("\nRidge keeps all coefficients non-zero and shrinks them smoothly.");
console.log("Lasso with a larger alpha sets more coefficients to exactly zero.");
