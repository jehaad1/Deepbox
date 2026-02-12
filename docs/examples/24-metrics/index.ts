/**
 * Example 24: Model Evaluation Metrics
 *
 * Learn to evaluate models using various performance metrics.
 * Different metrics for classification, regression, and clustering.
 */

import {
  // Classification metrics
  accuracy,
  confusionMatrix,
  f1Score,
  mae,
  mape,
  mse,
  precision,
  // Regression metrics
  r2Score,
  recall,
  rmse,
  // Clustering metrics
  silhouetteScore,
} from "deepbox/metrics";
import { tensor } from "deepbox/ndarray";

console.log("=== Model Evaluation Metrics ===\n");

// Classification Metrics
console.log("1. Classification Metrics:");
console.log("-".repeat(50));

const y_true_class = tensor([1, 0, 1, 1, 0, 1, 0, 0, 1, 1]);
const y_pred_class = tensor([1, 0, 1, 0, 0, 1, 0, 1, 1, 1]);

console.log("True labels:", y_true_class.toString());
console.log("Predictions:", `${y_pred_class.toString()}\n`);

const acc = accuracy(y_true_class, y_pred_class);
const prec = precision(y_true_class, y_pred_class);
const rec = recall(y_true_class, y_pred_class);
const f1 = f1Score(y_true_class, y_pred_class);

console.log(`Accuracy:  ${(Number(acc) * 100).toFixed(2)}%`);
console.log(`Precision: ${(Number(prec) * 100).toFixed(2)}%`);
console.log(`Recall:    ${(Number(rec) * 100).toFixed(2)}%`);
console.log(`F1-Score:  ${(Number(f1) * 100).toFixed(2)}%\n`);

const cm = confusionMatrix(y_true_class, y_pred_class);
console.log("Confusion Matrix:");
console.log(cm.toString());
console.log("Format: [[TN, FP], [FN, TP]]\n");

// Regression Metrics
console.log("2. Regression Metrics:");
console.log("-".repeat(50));

const y_true_reg = tensor([3.0, -0.5, 2.0, 7.0, 4.2]);
const y_pred_reg = tensor([2.5, 0.0, 2.1, 7.8, 4.0]);

console.log("True values:", y_true_reg.toString());
console.log("Predictions:", `${y_pred_reg.toString()}\n`);

const r2 = r2Score(y_true_reg, y_pred_reg);
const mseVal = mse(y_true_reg, y_pred_reg);
const rmseVal = rmse(y_true_reg, y_pred_reg);
const maeVal = mae(y_true_reg, y_pred_reg);
const mapeVal = mape(y_true_reg, y_pred_reg);

console.log(`R² Score: ${r2.toFixed(4)}`);
console.log(`MSE:      ${mseVal.toFixed(4)}`);
console.log(`RMSE:     ${rmseVal.toFixed(4)}`);
console.log(`MAE:      ${maeVal.toFixed(4)}`);
console.log(`MAPE:     ${(mapeVal * 100).toFixed(2)}%\n`);

// Clustering Metrics
console.log("3. Clustering Metrics:");
console.log("-".repeat(50));

const X_cluster = tensor([
  [1, 2],
  [1.5, 1.8],
  [5, 8],
  [8, 8],
  [1, 0.6],
  [9, 11],
]);
const labels = tensor([0, 0, 1, 1, 0, 1]);

const silhouette = silhouetteScore(X_cluster, labels);
console.log(`Silhouette Score: ${silhouette.toFixed(4)}`);
console.log("Range: [-1, 1], higher is better");
console.log("Measures how similar points are to their own cluster\n");

console.log("Metric Selection Guide:");
console.log("• Classification: Use F1-score for imbalanced data");
console.log("• Regression: R² for variance explained, MAE for interpretability");
console.log("• Clustering: Silhouette for cluster quality");

console.log("\n✓ Metrics evaluation complete!");
