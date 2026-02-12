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
// Classification dataset (Iris — full 3-class for multi-class models)
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

const svc = new LinearSVC({ C: 1.0 });
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

const svr = new LinearSVR({ C: 1.0 });
svr.fit(XRegTrain, yRegTrain);
const svrPred = svr.predict(XRegTest);
console.log("  MSE:", mse(yRegTest, svrPred).toFixed(4));
console.log("  R²: ", r2Score(yRegTest, svrPred).toFixed(4));

console.log("\n=== Tree-Based & Ensemble Models Complete ===");
