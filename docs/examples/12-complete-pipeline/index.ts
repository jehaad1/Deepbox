/**
 * Example 12: Complete ML Pipeline
 *
 * A regression workflow from start to finish on the Housing-Mini dataset:
 * load, explore, split, scale, fit a Ridge model, evaluate and plot.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { loadHousingMini } from "deepbox/datasets";
import { mae, mse, r2Score } from "deepbox/metrics";
import { Ridge } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { Figure } from "deepbox/plot";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";
import { mean, std } from "deepbox/stats";

console.log("=".repeat(60));
console.log("Example 12: Complete Machine Learning Pipeline");
console.log("=".repeat(60));

mkdirSync("docs/examples/12-complete-pipeline/output", { recursive: true });

// Step 1: Load data
console.log("\nStep 1: Loading the dataset");
console.log("-".repeat(60));

const dataset = loadHousingMini();
console.log("Loaded the Housing-Mini dataset");
console.log(`  Samples: ${dataset.data.shape[0]}`);
console.log(`  Features: ${dataset.data.shape[1]}`);

// Step 2: Exploratory data analysis
console.log("\nStep 2: Exploratory data analysis");
console.log("-".repeat(60));

// Take the first feature column. slice({}, 0) keeps every row of axis 0
// and selects index 0 on axis 1.
const feature = dataset.data.slice({}, 0);

console.log("Feature 1 statistics:");
console.log(`  Mean: ${Number(mean(feature).item()).toFixed(2)}`);
console.log(`  Std:  ${Number(std(feature).item()).toFixed(2)}`);

// Step 3: Data preprocessing
console.log("\nStep 3: Data preprocessing");
console.log("-".repeat(60));

const [XTrain, XTest, yTrain, yTest] = trainTestSplit(dataset.data, dataset.target, {
  testSize: 0.2,
  randomState: 42,
  shuffle: true,
});

console.log("Split the data:");
console.log(`  Training: ${XTrain.shape[0]} samples`);
console.log(`  Testing:  ${XTest.shape[0]} samples`);

// Fit the scaler on the training rows only, so no test information leaks in.
const scaler = new StandardScaler();
scaler.fit(XTrain);
const XTrainScaled = scaler.transform(XTrain);
const XTestScaled = scaler.transform(XTest);

console.log("Scaled the features with StandardScaler");

// Step 4: Model training
console.log("\nStep 4: Model training");
console.log("-".repeat(60));

const model = new Ridge({ alpha: 1.0 });
model.fit(XTrainScaled, yTrain);

console.log("Trained Ridge regression (alpha = 1.0)");

// Step 5: Model evaluation
console.log("\nStep 5: Model evaluation");
console.log("-".repeat(60));

const yPred = model.predict(XTestScaled);

const r2 = r2Score(yTest, yPred);
const mseVal = mse(yTest, yPred);
const maeVal = mae(yTest, yPred);

console.log("Performance metrics:");
console.log(`  R² Score: ${r2.toFixed(4)}`);
console.log(`  MSE:      ${mseVal.toFixed(4)}`);
console.log(`  MAE:      ${maeVal.toFixed(4)}`);

// Step 6: Visualization
console.log("\nStep 6: Results visualization");
console.log("-".repeat(60));

// Predictions vs actual values. Tensors go straight into the plot.
const fig = new Figure({ width: 640, height: 480 });
const ax = fig.addAxes();

ax.scatter(yTest, yPred, {
  color: "#1f77b4",
  size: 8,
});

// The red line marks perfect predictions (predicted equals actual).
const lo = Math.min(Number(yTest.min().item()), Number(yPred.min().item()));
const hi = Math.max(Number(yTest.max().item()), Number(yPred.max().item()));
ax.plot(tensor([lo, hi]), tensor([lo, hi]), {
  color: "#ff0000",
  linewidth: 2,
});
ax.setTitle("Predictions vs Actual Values");
ax.setXLabel("Actual");
ax.setYLabel("Predicted");

const svg = fig.renderSVG();
writeFileSync("docs/examples/12-complete-pipeline/output/predictions.svg", svg.svg);
console.log("Saved: output/predictions.svg");

// Step 7: Summary
console.log("\nStep 7: Pipeline summary");
console.log("-".repeat(60));

console.log("Steps run:");
console.log("  1. Load data (Housing-Mini)");
console.log("  2. Exploratory analysis");
console.log("  3. Train/test split (80/20)");
console.log("  4. Feature scaling (StandardScaler)");
console.log("  5. Model training (Ridge)");
console.log(`  6. Model evaluation (R² = ${r2.toFixed(3)})`);
console.log("  7. Results visualization");

console.log("\nNotes:");
console.log("  Split first, then fit the scaler on the training set, to avoid data leakage.");
console.log("  Report more than one metric: R² shows fit quality, MAE is in the target's units.");

console.log(`\n${"=".repeat(60)}`);
console.log("Pipeline finished");
console.log("=".repeat(60));
