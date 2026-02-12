/**
 * Example 12: Complete ML Pipeline
 *
 * Bring everything together in a comprehensive machine learning workflow.
 * From data loading to model evaluation and visualization.
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
console.log("Example 20: Complete Machine Learning Pipeline");
console.log("=".repeat(60));

mkdirSync("docs/examples/12-complete-pipeline/output", { recursive: true });

// Step 1: Load Data
console.log("\n📦 Step 1: Loading Dataset");
console.log("-".repeat(60));

const dataset = loadHousingMini();
console.log(`✓ Loaded Housing-Mini dataset`);
console.log(`  Samples: ${dataset.data.shape[0]}`);
console.log(`  Features: ${dataset.data.shape[1]}`);

// Step 2: Exploratory Data Analysis
console.log("\n📊 Step 2: Exploratory Data Analysis");
console.log("-".repeat(60));

// Extract first feature for analysis
const feature_data: number[] = [];
const numFeatures = dataset.data.shape[1] || 0;
for (let i = 0; i < dataset.data.shape[0]; i++) {
  feature_data.push(Number(dataset.data.data[dataset.data.offset + i * numFeatures]));
}
const feature = tensor(feature_data);

const meanVal = Number(mean(feature).data[0]);
const stdVal = Number(std(feature).data[0]);

console.log(`Feature 1 Statistics:`);
console.log(`  Mean: ${meanVal.toFixed(2)}`);
console.log(`  Std:  ${stdVal.toFixed(2)}`);

// Step 3: Data Preprocessing
console.log("\n🔄 Step 3: Data Preprocessing");
console.log("-".repeat(60));

const [X_train, X_test, y_train, y_test] = trainTestSplit(dataset.data, dataset.target, {
  testSize: 0.2,
  randomState: 42,
  shuffle: true,
});

console.log(`✓ Split data:`);
console.log(`  Training: ${X_train.shape[0]} samples`);
console.log(`  Testing:  ${X_test.shape[0]} samples`);

const scaler = new StandardScaler();
scaler.fit(X_train);
const X_train_scaled = scaler.transform(X_train);
const X_test_scaled = scaler.transform(X_test);

console.log(`✓ Scaled features using StandardScaler`);

// Step 4: Model Training
console.log("\n🤖 Step 4: Model Training");
console.log("-".repeat(60));

const model = new Ridge({ alpha: 1.0 });
model.fit(X_train_scaled, y_train);

console.log(`✓ Trained Ridge Regression (α=1.0)`);

// Step 5: Model Evaluation
console.log("\n📈 Step 5: Model Evaluation");
console.log("-".repeat(60));

const y_pred = model.predict(X_test_scaled);

const r2 = r2Score(y_test, y_pred);
const mseVal = mse(y_test, y_pred);
const maeVal = mae(y_test, y_pred);

console.log(`Performance Metrics:`);
console.log(`  R² Score: ${r2.toFixed(4)}`);
console.log(`  MSE:      ${mseVal.toFixed(4)}`);
console.log(`  MAE:      ${maeVal.toFixed(4)}`);

// Step 6: Visualization
console.log("\n🎨 Step 6: Results Visualization");
console.log("-".repeat(60));

// Extract predictions and actual values
const y_test_array: number[] = [];
const y_pred_array: number[] = [];

for (let i = 0; i < y_test.size; i++) {
  y_test_array.push(Number(y_test.data[y_test.offset + i]));
  y_pred_array.push(Number(y_pred.data[y_pred.offset + i]));
}

// Create predictions vs actual plot
const fig = new Figure({ width: 640, height: 480 });
const ax = fig.addAxes();

ax.scatter(tensor(y_test_array), tensor(y_pred_array), {
  color: "#1f77b4",
  size: 8,
});
ax.plot(tensor([0, 1, 2]), tensor([0, 1, 2]), {
  color: "#ff0000",
  linewidth: 2,
});
ax.setTitle("Predictions vs Actual Values");
ax.setXLabel("Actual");
ax.setYLabel("Predicted");

const svg = fig.renderSVG();
writeFileSync("docs/examples/12-complete-pipeline/output/predictions.svg", svg.svg);
console.log("✓ Saved: output/predictions.svg");

// Step 7: Summary
console.log("\n📋 Step 7: Pipeline Summary");
console.log("-".repeat(60));

console.log(`Complete ML Pipeline Executed:`);
console.log(`  1. ✓ Data Loading (Housing-Mini dataset)`);
console.log(`  2. ✓ Exploratory Analysis`);
console.log(`  3. ✓ Train/Test Split (80/20)`);
console.log(`  4. ✓ Feature Scaling (StandardScaler)`);
console.log(`  5. ✓ Model Training (Ridge Regression)`);
console.log(`  6. ✓ Model Evaluation (R²=${r2.toFixed(3)})`);
console.log(`  7. ✓ Results Visualization`);

console.log("\n💡 Key Takeaways:");
console.log("• Always split data before scaling to prevent data leakage");
console.log("• Feature scaling improves model performance");
console.log("• Use multiple metrics to evaluate models");
console.log("• Visualize results to understand model behavior");
console.log("• Ridge regression adds L2 regularization to prevent overfitting");

console.log(`\n${"=".repeat(60)}`);
console.log("✅ Complete ML Pipeline Finished Successfully!");
console.log("=".repeat(60));
