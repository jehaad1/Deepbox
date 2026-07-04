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
import { cross_validate, Lasso, LinearRegression, LogisticRegression, Ridge } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { Figure } from "deepbox/plot";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";

console.log("=".repeat(60));
console.log("Example 06: Complete Machine Learning Pipeline");
console.log("=".repeat(60));

mkdirSync("docs/examples/06-ml-pipeline/output", { recursive: true });

console.log("\n📦 Part 1: Classification with Iris Dataset");
console.log("-".repeat(60));

const iris = loadIris();
console.log(`Dataset loaded: ${iris.data.shape[0]} samples, ${iris.data.shape[1]} features`);
console.log(`Classes: ${iris.targetNames?.join(", ") || "N/A"}`);
console.log(`Features: ${iris.featureNames?.join(", ") || "N/A"}`);

const binaryIrisTarget = [];
for (let i = 0; i < iris.target.size; i++) {
  const val = Number(iris.target.data[iris.target.offset + i]);
  binaryIrisTarget.push(val === 0 ? 0 : 1);
}
const binaryTarget = tensor(binaryIrisTarget);

console.log("\n🔄 Data Preprocessing");
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

console.log("✓ Features scaled using StandardScaler");

console.log("\n🤖 Training Logistic Regression");
console.log("-".repeat(60));

const logReg = new LogisticRegression({ maxIter: 1000, learningRate: 0.1 });
logReg.fit(XTrainScaled, yTrainIris);

const yPredIris = logReg.predict(XTestScaled);

console.log("\n📊 Classification Metrics");
console.log("-".repeat(60));
const acc = accuracy(yTestIris, yPredIris);
const prec = precision(yTestIris, yPredIris);
const rec = recall(yTestIris, yPredIris);
const f1 = f1Score(yTestIris, yPredIris);

console.log(`Accuracy: ${(Number(acc) * 100).toFixed(2)}%`);
console.log(`Precision: ${(Number(prec) * 100).toFixed(2)}%`);
console.log(`Recall: ${(Number(rec) * 100).toFixed(2)}%`);
console.log(`F1-Score: ${(Number(f1) * 100).toFixed(2)}%`);

const confMatrix = confusionMatrix(yTestIris, yPredIris);
console.log("\nConfusion Matrix:");
console.log(confMatrix.toString());

console.log("\n📦 Part 2: Regression with Housing-Mini Dataset");
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

console.log("\n🔬 Comparing Regression Models");
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

console.log("\n🔄 Cross-Validation");
console.log("-".repeat(60));
const cvResult = cross_validate(new Ridge({ alpha: 1.0 }), XTrainHousingScaled, yTrainHousing, {
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

console.log("\n📈 Visualizing Predictions");
console.log("-".repeat(60));

const bestModel = new Ridge({ alpha: 1.0 });
bestModel.fit(XTrainHousingScaled, yTrainHousing);
const finalPredictions = bestModel.predict(XTestHousingScaled);

const yTestArray: number[] = [];
const yPredArray: number[] = [];

for (let i = 0; i < yTestHousing.size; i++) {
  yTestArray.push(Number(yTestHousing.data[yTestHousing.offset + i]));
  yPredArray.push(Number(finalPredictions.data[finalPredictions.offset + i]));
}

const fig = new Figure();
const ax = fig.addAxes();
ax.scatter(tensor(yTestArray), tensor(yPredArray), {
  color: "#1f77b4",
  size: 6,
});
ax.plot(tensor([0, 1, 2]), tensor([0, 1, 2]), {
  color: "#ff0000",
  linewidth: 2,
});
ax.setTitle("Predictions vs Actual");
ax.setXLabel("Actual Values");
ax.setYLabel("Predicted Values");
const svg = fig.renderSVG();
writeFileSync("docs/examples/06-ml-pipeline/output/predictions-vs-actual.svg", svg.svg);
console.log("✓ Saved: output/predictions-vs-actual.svg");

console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• Logistic Regression achieved high accuracy on binary classification");
console.log("• Ridge Regression with α=1.0 performed best on housing dataset");
console.log("• Cross-validation now reports real fold-by-fold R² and RMSE scores");
console.log("• Feature scaling is crucial for model performance");
console.log("• Regularization helps prevent overfitting");

console.log("\n✅ ML Pipeline Complete!");
console.log("=".repeat(60));
