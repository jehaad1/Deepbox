/**
 * Example 11: Tree-Based & Ensemble Models
 *
 * Decision Trees, Random Forests, Gradient Boosting, and Linear SVM.
 * Covers classification and regression variants.
 */

import { loadIris } from "deepbox/datasets";
import { accuracy, mse, r2Score } from "deepbox/metrics";
import {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  LinearSVC,
  LinearSVR,
  RandomForestClassifier,
  RandomForestRegressor,
} from "deepbox/ml";
import { slice, tensor } from "deepbox/ndarray";
import { trainTestSplit } from "deepbox/preprocess";

console.log("=== Tree-Based & Ensemble Models ===\n");

// ---------------------------------------------------------------------------
// Classification dataset (Iris, all 3 classes for multi-class models)
// ---------------------------------------------------------------------------
const iris = loadIris();
const [XTrain, XTest, yTrain, yTest] = trainTestSplit(iris.data, iris.target, {
  testSize: 0.2,
  randomState: 42,
});

// Binary subset (classes 0 and 1 only) for models that require binary labels
const XBin = slice(iris.data, { start: 0, end: 100 });
const yBin = slice(iris.target, { start: 0, end: 100 });
const [XBinTrain, XBinTest, yBinTrain, yBinTest] = trainTestSplit(XBin, yBin, {
  testSize: 0.2,
  randomState: 42,
});

// ---------------------------------------------------------------------------
// Part 1: Decision Tree Classifier
// ---------------------------------------------------------------------------
console.log("--- Part 1: Decision Tree Classifier ---");

const dtc = new DecisionTreeClassifier({ maxDepth: 5, minSamplesSplit: 2 });
dtc.fit(XTrain, yTrain);
const dtcPred = dtc.predict(XTest);
console.log("  Accuracy:", accuracy(yTest, dtcPred).toFixed(4));

// ---------------------------------------------------------------------------
// Part 2: Random Forest Classifier
// ---------------------------------------------------------------------------
console.log("\n--- Part 2: Random Forest Classifier ---");

const rfc = new RandomForestClassifier({
  nEstimators: 50,
  maxDepth: 5,
  randomState: 42,
});
rfc.fit(XTrain, yTrain);
const rfcPred = rfc.predict(XTest);
console.log("  Accuracy:", accuracy(yTest, rfcPred).toFixed(4));

// ---------------------------------------------------------------------------
// Part 3: Gradient Boosting Classifier
// ---------------------------------------------------------------------------
console.log("\n--- Part 3: Gradient Boosting Classifier ---");

const gbc = new GradientBoostingClassifier({
  nEstimators: 50,
  learningRate: 0.1,
  maxDepth: 3,
});
gbc.fit(XBinTrain, yBinTrain);
const gbcPred = gbc.predict(XBinTest);
console.log("  Accuracy:", accuracy(yBinTest, gbcPred).toFixed(4));

// ---------------------------------------------------------------------------
// Part 4: Linear SVC
// ---------------------------------------------------------------------------
console.log("\n--- Part 4: Linear SVC ---");

const svc = new LinearSVC({ C: 1.0, randomState: 42 });
svc.fit(XBinTrain, yBinTrain);
const svcPred = svc.predict(XBinTest);
console.log("  Accuracy:", accuracy(yBinTest, svcPred).toFixed(4));

// ---------------------------------------------------------------------------
// Regression dataset (synthetic y = x0 + 2*x1 + noise)
// ---------------------------------------------------------------------------
console.log("\n--- Regression Models ---");

const XReg = tensor([
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [8, 9],
  [9, 10],
  [10, 11],
  [1, 3],
  [2, 5],
  [3, 2],
  [4, 1],
  [5, 4],
  [6, 3],
  [7, 6],
  [8, 5],
  [9, 8],
  [10, 7],
]);
const yReg = tensor([5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 7, 12, 7, 6, 13, 12, 19, 18, 25, 24]);

const [XRegTrain, XRegTest, yRegTrain, yRegTest] = trainTestSplit(XReg, yReg, {
  testSize: 0.2,
  randomState: 42,
});

// ---------------------------------------------------------------------------
// Part 5: Decision Tree Regressor
// ---------------------------------------------------------------------------
console.log("\n--- Part 5: Decision Tree Regressor ---");

const dtr = new DecisionTreeRegressor({ maxDepth: 5 });
dtr.fit(XRegTrain, yRegTrain);
const dtrPred = dtr.predict(XRegTest);
console.log("  MSE:", mse(yRegTest, dtrPred).toFixed(4));
console.log("  R²: ", r2Score(yRegTest, dtrPred).toFixed(4));

// ---------------------------------------------------------------------------
// Part 6: Random Forest Regressor
// ---------------------------------------------------------------------------
console.log("\n--- Part 6: Random Forest Regressor ---");

const rfr = new RandomForestRegressor({
  nEstimators: 50,
  maxDepth: 5,
  randomState: 42,
});
rfr.fit(XRegTrain, yRegTrain);
const rfrPred = rfr.predict(XRegTest);
console.log("  MSE:", mse(yRegTest, rfrPred).toFixed(4));
console.log("  R²: ", r2Score(yRegTest, rfrPred).toFixed(4));

// ---------------------------------------------------------------------------
// Part 7: Gradient Boosting Regressor
// ---------------------------------------------------------------------------
console.log("\n--- Part 7: Gradient Boosting Regressor ---");

const gbr = new GradientBoostingRegressor({
  nEstimators: 50,
  learningRate: 0.1,
  maxDepth: 3,
});
gbr.fit(XRegTrain, yRegTrain);
const gbrPred = gbr.predict(XRegTest);
console.log("  MSE:", mse(yRegTest, gbrPred).toFixed(4));
console.log("  R²: ", r2Score(yRegTest, gbrPred).toFixed(4));

// ---------------------------------------------------------------------------
// Part 8: Linear SVR
// ---------------------------------------------------------------------------
console.log("\n--- Part 8: Linear SVR ---");

// The features are unscaled, so the solver needs more passes to converge.
const svr = new LinearSVR({ C: 1.0, maxIter: 20000, randomState: 42 });
svr.fit(XRegTrain, yRegTrain);
const svrPred = svr.predict(XRegTest);
console.log("  MSE:", mse(yRegTest, svrPred).toFixed(4));
console.log("  R²: ", r2Score(yRegTest, svrPred).toFixed(4));

// ---------------------------------------------------------------------------
// Part 9: Tree options for regularization and weighting
// ---------------------------------------------------------------------------
console.log("\n--- Part 9: Tree regularization and weights ---");

// maxLeafNodes grows the tree best-first and stops at the given number of leaves.
// ccpAlpha prunes the finished tree: a larger value removes more branches.
// classWeight and the sampleWeight argument of fit() change how much each row counts.
const variants: Array<[string, DecisionTreeClassifier]> = [
  ["default", new DecisionTreeClassifier({ randomState: 42 })],
  ["maxLeafNodes = 3", new DecisionTreeClassifier({ maxLeafNodes: 3, randomState: 42 })],
  ["ccpAlpha = 0.05", new DecisionTreeClassifier({ ccpAlpha: 0.05, randomState: 42 })],
  [
    "classWeight = balanced",
    new DecisionTreeClassifier({ classWeight: "balanced", randomState: 42 }),
  ],
];
for (const [label, tree] of variants) {
  tree.fit(XTrain, yTrain);
  const acc = accuracy(yTest, tree.predict(XTest)).toFixed(4);
  console.log(
    `  ${label.padEnd(24)} leaves: ${tree.getNLeaves()}  depth: ${tree.getDepth()}  accuracy: ${acc}`
  );
}

// Per-row weights: rows of class 2 count twice as much as the others.
const weights = yTrain.eq(2).astype("float32").add(1);
const weighted = new DecisionTreeClassifier({ maxDepth: 3, randomState: 42 });
weighted.fit(XTrain, yTrain, weights);
console.log(
  `  ${"sampleWeight in fit()".padEnd(24)} leaves: ${weighted.getNLeaves()}  depth: ${weighted.getDepth()}  accuracy: ${accuracy(yTest, weighted.predict(XTest)).toFixed(4)}`
);

// Fitted forests expose how much each feature contributed to the splits.
console.log("  Random forest feature importances:", rfc.featureImportances.toString());

// clone() returns an unfitted copy with the same options.
const rfcCopy = rfc.clone();
rfcCopy.fit(XTrain, yTrain);
console.log("  Cloned forest accuracy:", accuracy(yTest, rfcCopy.predict(XTest)).toFixed(4));

console.log("\n=== Tree-Based & Ensemble Models Complete ===");
