/**
 * Example 08: Logistic Regression
 *
 * Binary classification with logistic regression on the Iris dataset:
 * setosa vs the other two species. Features are standardized first, and the
 * model is scored with accuracy, precision, recall, F1 and a confusion matrix.
 */

import { loadIris } from "deepbox/datasets";
import { accuracy, confusionMatrix, f1Score, precision, recall } from "deepbox/metrics";
import { LogisticRegression } from "deepbox/ml";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";

console.log("=== Logistic Regression ===\n");

// Load the famous Iris dataset for classification
const iris = loadIris();
console.log(`Dataset: ${iris.data.shape[0]} samples, ${iris.data.shape[1]} features\n`);

// Simplify to binary classification: setosa (0) vs non-setosa (1).
// Clipping the labels 0, 1, 2 to the range [0, 1] maps setosa to 0 and the others to 1.
const y = iris.target.clip(0, 1);

// Split into train (70%) and test (30%) sets
const [XTrain, XTest, yTrain, yTest] = trainTestSplit(iris.data, y, {
  testSize: 0.3,
  randomState: 42,
});

console.log(`Training set: ${XTrain.shape[0]} samples`);
console.log(`Test set: ${XTest.shape[0]} samples\n`);

// Standardize features (mean 0, std 1) so gradient descent converges faster.
// Fit the scaler on the training set only, then apply it to both sets.
const scaler = new StandardScaler();
scaler.fit(XTrain);
const XTrainScaled = scaler.transform(XTrain);
const XTestScaled = scaler.transform(XTest);

console.log("Features scaled\n");

// Create and train logistic regression classifier
const model = new LogisticRegression({ maxIter: 1000, learningRate: 0.1 });
model.fit(XTrainScaled, yTrain);

console.log("Model trained.\n");

// Make predictions on test data
const yPred = model.predict(XTestScaled);

// Calculate classification metrics
const acc = accuracy(yTest, yPred);
const prec = precision(yTest, yPred);
const rec = recall(yTest, yPred);
const f1 = f1Score(yTest, yPred);

console.log("Model Performance:");
console.log(`Accuracy:  ${(acc * 100).toFixed(2)}%`);
console.log(`Precision: ${(prec * 100).toFixed(2)}%`);
console.log(`Recall:    ${(rec * 100).toFixed(2)}%`);
console.log(`F1-Score:  ${(f1 * 100).toFixed(2)}%\n`);

const cm = confusionMatrix(yTest, yPred);
console.log("Confusion Matrix:");
console.log(cm.toString());
