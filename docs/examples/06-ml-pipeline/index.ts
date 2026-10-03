/**
 * Example 06: Machine Learning Pipeline
 *
 * Two small end-to-end workflows:
 * 1. Binary classification on Iris: split, scale, fit, evaluate.
 * 2. Regression on Housing-Mini: compare four linear models, cross-validate
 *    one of them and plot its predictions.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { loadHousingMini, loadIris } from "deepbox/datasets";
import {
  accuracy,
  confusionMatrix,
  f1Score,
  mae,
  mse,
  precision,
  r2Score,
  recall,
  rmse,
} from "deepbox/metrics";
import { crossValidate, Lasso, LinearRegression, LogisticRegression, Ridge } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { Figure } from "deepbox/plot";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";

console.log("=".repeat(60));
console.log("Example 06: Complete Machine Learning Pipeline");
console.log("=".repeat(60));

mkdirSync("docs/examples/06-ml-pipeline/output", { recursive: true });

console.log("\nPart 1: Classification with the Iris dataset");
console.log("-".repeat(60));

const iris = loadIris();
console.log(`Dataset loaded: ${iris.data.shape[0]} samples, ${iris.data.shape[1]} features`);
console.log(`Classes: ${iris.targetNames?.join(", ") || "N/A"}`);
console.log(`Features: ${iris.featureNames?.join(", ") || "N/A"}`);

// Turn the three classes into two: setosa (0) vs the rest (1).
// Clipping labels 0, 1, 2 to the range [0, 1] gives exactly that.
const binaryTarget = iris.target.clip(0, 1);

console.log("\nData preprocessing");
console.log("-".repeat(60));

const [XTrainIris, XTestIris, yTrainIris, yTestIris] = trainTestSplit(iris.data, binaryTarget, {
  testSize: 0.3,
  randomState: 42,
});

console.log(`Training set: ${XTrainIris.shape[0]} samples`);
console.log(`Test set: ${XTestIris.shape[0]} samples`);

const scalerIris = new StandardScaler();
scalerIris.fit(XTrainIris);
const XTrainScaled = scalerIris.transform(XTrainIris);
const XTestScaled = scalerIris.transform(XTestIris);

console.log("Features scaled with StandardScaler (fitted on the training set only)");

console.log("\nTraining logistic regression");
console.log("-".repeat(60));

const logReg = new LogisticRegression({ maxIter: 1000, learningRate: 0.1 });
logReg.fit(XTrainScaled, yTrainIris);

const yPredIris = logReg.predict(XTestScaled);

console.log("\nClassification metrics");
console.log("-".repeat(60));
const acc = accuracy(yTestIris, yPredIris);
const prec = precision(yTestIris, yPredIris);
const rec = recall(yTestIris, yPredIris);
const f1 = f1Score(yTestIris, yPredIris);

console.log(`Accuracy: ${(acc * 100).toFixed(2)}%`);
console.log(`Precision: ${(prec * 100).toFixed(2)}%`);
console.log(`Recall: ${(rec * 100).toFixed(2)}%`);
console.log(`F1-Score: ${(f1 * 100).toFixed(2)}%`);

const confMatrix = confusionMatrix(yTestIris, yPredIris);
console.log("\nConfusion Matrix:");
console.log(confMatrix.toString());

console.log("\nPart 2: Regression with the Housing-Mini dataset");
console.log("-".repeat(60));

const housing = loadHousingMini();
console.log(`Dataset loaded: ${housing.data.shape[0]} samples, ${housing.data.shape[1]} features`);

const [XTrainHousing, XTestHousing, yTrainHousing, yTestHousing] = trainTestSplit(
  housing.data,
  housing.target,
  {
    testSize: 0.25,
    randomState: 42,
  }
);

console.log(`Training set: ${XTrainHousing.shape[0]} samples`);
console.log(`Test set: ${XTestHousing.shape[0]} samples`);

const scalerHousing = new StandardScaler();
scalerHousing.fit(XTrainHousing);
const XTrainHousingScaled = scalerHousing.transform(XTrainHousing);
const XTestHousingScaled = scalerHousing.transform(XTestHousing);

console.log("\nComparing regression models");
console.log("-".repeat(60));

const models = [
  { name: "Linear Regression", model: new LinearRegression() },
  { name: "Ridge Regression (α=1.0)", model: new Ridge({ alpha: 1.0 }) },
  { name: "Ridge Regression (α=10.0)", model: new Ridge({ alpha: 10.0 }) },
  { name: "Lasso Regression (α=0.1)", model: new Lasso({ alpha: 0.1 }) },
];

const results: Array<{
  name: string;
  r2: number;
  mse: number;
  mae: number;
  rmse: number;
}> = [];

for (const { name, model } of models) {
  model.fit(XTrainHousingScaled, yTrainHousing);
  const yPred = model.predict(XTestHousingScaled);

  const r2 = r2Score(yTestHousing, yPred);
  const mseVal = mse(yTestHousing, yPred);
  const maeVal = mae(yTestHousing, yPred);
  const rmseVal = rmse(yTestHousing, yPred);

  results.push({ name, r2, mse: mseVal, mae: maeVal, rmse: rmseVal });

  console.log(`\n${name}:`);
  console.log(`  R² Score: ${r2.toFixed(4)}`);
  console.log(`  MSE: ${mseVal.toFixed(4)}`);
  console.log(`  MAE: ${maeVal.toFixed(4)}`);
  console.log(`  RMSE: ${rmseVal.toFixed(4)}`);
}

console.log("\nCross-validation");
console.log("-".repeat(60));
const cvResult = crossValidate(new Ridge({ alpha: 1.0 }), XTrainHousingScaled, yTrainHousing, {
  cv: 5,
  scoring: {
    r2: (estimator, XFold, yFold) => r2Score(yFold, (estimator as Ridge).predict(XFold)),
    rmse: (estimator, XFold, yFold) => rmse(yFold, (estimator as Ridge).predict(XFold)),
  },
});

const cvR2Scores = cvResult.testScores.r2 ?? [];
const cvRmseScores = cvResult.testScores.rmse ?? [];
const meanCvR2 =
  cvR2Scores.reduce((total, score) => total + score, 0) / Math.max(cvR2Scores.length, 1);
const meanCvRmse =
  cvRmseScores.reduce((total, score) => total + score, 0) / Math.max(cvRmseScores.length, 1);

console.log("5-fold cross-validation for Ridge Regression (α=1.0):");
for (let i = 0; i < cvR2Scores.length; i++) {
  const r2 = cvR2Scores[i];
  const rmseScore = cvRmseScores[i];
  console.log(
    `  Fold ${i + 1}: R²=${r2?.toFixed(4) ?? "n/a"}, RMSE=${rmseScore?.toFixed(4) ?? "n/a"}`
  );
}
console.log(`\nMean CV R²: ${meanCvR2.toFixed(4)}`);
console.log(`Mean CV RMSE: ${meanCvRmse.toFixed(4)}`);

console.log("\nPlotting predictions");
console.log("-".repeat(60));

const bestModel = new Ridge({ alpha: 1.0 });
bestModel.fit(XTrainHousingScaled, yTrainHousing);
const finalPredictions = bestModel.predict(XTestHousingScaled);

// Plain tensors go straight into the plot. The red line is the ideal case
// where every prediction equals the true value.
const lo = Math.min(Number(yTestHousing.min().item()), Number(finalPredictions.min().item()));
const hi = Math.max(Number(yTestHousing.max().item()), Number(finalPredictions.max().item()));

const fig = new Figure();
const ax = fig.addAxes();
ax.scatter(yTestHousing, finalPredictions, {
  color: "#1f77b4",
  size: 6,
});
ax.plot(tensor([lo, hi]), tensor([lo, hi]), {
  color: "#ff0000",
  linewidth: 2,
});
ax.setTitle("Predictions vs Actual");
ax.setXLabel("Actual Values");
ax.setYLabel("Predicted Values");
const svg = fig.renderSVG();
writeFileSync("docs/examples/06-ml-pipeline/output/predictions-vs-actual.svg", svg.svg);
console.log("Saved: output/predictions-vs-actual.svg");

console.log("\nSummary");
console.log("-".repeat(60));
const best = results.reduce((a, b) => (b.r2 > a.r2 ? b : a));
console.log(`Iris accuracy: ${(acc * 100).toFixed(2)}%`);
console.log(`Best housing model by test R²: ${best.name} (${best.r2.toFixed(4)})`);
console.log(`Mean 5-fold CV R² for Ridge (α=1.0): ${meanCvR2.toFixed(4)}`);

console.log("\nPipeline complete.");
console.log("=".repeat(60));
