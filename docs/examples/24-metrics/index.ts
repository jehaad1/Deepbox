/**
 * Example 24: Model Evaluation Metrics
 *
 * Metrics for classification, regression and clustering. Every metric
 * returns a plain number (or an array of numbers when asked for per-class
 * values), so results can be printed or compared directly.
 */

import {
  // Classification metrics
  accuracy,
  confusionMatrix,
  f1Score,
  jaccardScore,
  logLoss,
  mae,
  mape,
  matthewsCorrcoef,
  meanAbsolutePercentageError,
  mse,
  precision,
  // Regression metrics
  r2Score,
  recall,
  rmse,
  rocAucScore,
  // Clustering metrics
  silhouetteScore,
} from "deepbox/metrics";
import { tensor } from "deepbox/ndarray";

console.log("=== Model Evaluation Metrics ===\n");

// Binary classification metrics
console.log("1. Binary Classification Metrics:");
console.log("-".repeat(50));

const yTrueClass = tensor([1, 0, 1, 1, 0, 1, 0, 0, 1, 1]);
const yPredClass = tensor([1, 0, 1, 0, 0, 1, 0, 1, 1, 1]);

console.log("True labels:", yTrueClass.toString());
console.log("Predictions:", `${yPredClass.toString()}\n`);

const acc = accuracy(yTrueClass, yPredClass);
const prec = precision(yTrueClass, yPredClass);
const rec = recall(yTrueClass, yPredClass);
const f1 = f1Score(yTrueClass, yPredClass);

console.log(`Accuracy:  ${(acc * 100).toFixed(2)}%`);
console.log(`Precision: ${(prec * 100).toFixed(2)}%`);
console.log(`Recall:    ${(rec * 100).toFixed(2)}%`);
console.log(`F1-Score:  ${(f1 * 100).toFixed(2)}%`);
console.log(`Jaccard:   ${(jaccardScore(yTrueClass, yPredClass) * 100).toFixed(2)}%`);
console.log(`Matthews correlation: ${matthewsCorrcoef(yTrueClass, yPredClass).toFixed(4)}\n`);

const cm = confusionMatrix(yTrueClass, yPredClass);
console.log("Confusion Matrix:");
console.log(cm.toString());
console.log("Rows are true classes, columns are predicted classes: [[TN, FP], [FN, TP]]\n");

// Probability-based metrics need scores, not labels.
console.log("2. Probability-Based Metrics (binary):");
console.log("-".repeat(50));

const yProba = tensor([0.9, 0.2, 0.8, 0.4, 0.1, 0.7, 0.3, 0.6, 0.85, 0.95]);
console.log(`ROC AUC:  ${rocAucScore(yTrueClass, yProba).toFixed(4)}`);
console.log(`Log loss: ${logLoss(yTrueClass, yProba).toFixed(4)}\n`);

// Multiclass metrics: choose how per-class scores are combined with `average`.
console.log("3. Multiclass Metrics:");
console.log("-".repeat(50));

const yTrueMulti = tensor([0, 1, 2, 2, 1, 0, 2, 1, 0, 2]);
const yPredMulti = tensor([0, 2, 2, 2, 1, 0, 1, 1, 0, 2]);

for (const average of ["macro", "micro", "weighted"] as const) {
  const p = precision(yTrueMulti, yPredMulti, { average });
  const r = recall(yTrueMulti, yPredMulti, { average });
  const f = f1Score(yTrueMulti, yPredMulti, { average });
  console.log(
    `${average.padEnd(9)} precision=${p.toFixed(4)} recall=${r.toFixed(4)} f1=${f.toFixed(4)}`
  );
}

// average: null returns one value per class, in label order.
const perClass = f1Score(yTrueMulti, yPredMulti, { average: null });
console.log(`Per-class F1: ${perClass.map((v) => v.toFixed(4)).join(", ")}`);

// Multiclass ROC AUC takes an [n, classes] matrix of probabilities that sum to 1 per row.
const yProbaMulti = tensor([
  [0.8, 0.1, 0.1],
  [0.2, 0.3, 0.5],
  [0.1, 0.2, 0.7],
  [0.1, 0.1, 0.8],
  [0.2, 0.6, 0.2],
  [0.7, 0.2, 0.1],
  [0.1, 0.5, 0.4],
  [0.3, 0.5, 0.2],
  [0.6, 0.3, 0.1],
  [0.2, 0.2, 0.6],
]);
console.log(`Multiclass ROC AUC (macro): ${rocAucScore(yTrueMulti, yProbaMulti).toFixed(4)}`);
console.log(`Multiclass log loss: ${logLoss(yTrueMulti, yProbaMulti).toFixed(4)}\n`);

// Regression metrics
console.log("4. Regression Metrics:");
console.log("-".repeat(50));

const yTrueReg = tensor([3.0, 0.5, 2.0, 7.0, 4.2]);
const yPredReg = tensor([2.5, 0.6, 2.1, 7.8, 4.0]);

console.log("True values:", yTrueReg.toString());
console.log("Predictions:", `${yPredReg.toString()}\n`);

const r2 = r2Score(yTrueReg, yPredReg);
const mseVal = mse(yTrueReg, yPredReg);
const rmseVal = rmse(yTrueReg, yPredReg);
const maeVal = mae(yTrueReg, yPredReg);

console.log(`R² Score: ${r2.toFixed(4)}`);
console.log(`MSE:      ${mseVal.toFixed(4)}`);
console.log(`RMSE:     ${rmseVal.toFixed(4)}`);
console.log(`MAE:      ${maeVal.toFixed(4)}`);

// mape already returns a percentage. scikit-learn's version returns a fraction,
// which is what meanAbsolutePercentageError returns.
console.log(`MAPE:     ${mape(yTrueReg, yPredReg).toFixed(2)}% (mape returns a percentage)`);
console.log(
  `meanAbsolutePercentageError: ${meanAbsolutePercentageError(yTrueReg, yPredReg).toFixed(4)} (a fraction)\n`
);

// Clustering metrics
console.log("5. Clustering Metrics:");
console.log("-".repeat(50));

const XCluster = tensor([
  [1, 2],
  [1.5, 1.8],
  [5, 8],
  [8, 8],
  [1, 0.6],
  [9, 11],
]);
const labels = tensor([0, 0, 1, 1, 0, 1]);

const silhouette = silhouetteScore(XCluster, labels);
console.log(`Silhouette Score: ${silhouette.toFixed(4)}`);
console.log("Range: [-1, 1], higher is better");
console.log("Measures how close points are to their own cluster compared with the next one\n");

console.log("Metric selection guide:");
console.log(
  "  Classification: F1 for imbalanced data, ROC AUC and log loss when you have probabilities"
);
console.log("  Regression: R² for variance explained, MAE for an error in the target's units");
console.log("  Clustering: silhouette for cluster separation");
