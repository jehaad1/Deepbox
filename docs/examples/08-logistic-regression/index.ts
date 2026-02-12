/**
 * Example 08: Logistic Regression
 *
 * Build a binary classification model using logistic regression.
 * Learn to classify data into two categories.
 */

import { loadIris } from "deepbox/datasets";
import { accuracy, confusionMatrix, f1Score, precision, recall } from "deepbox/metrics";
import { LogisticRegression } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";

console.log("=== Logistic Regression ===\n");

// Load the famous Iris dataset for classification
const iris = loadIris();
console.log(`Dataset: ${iris.data.shape[0]} samples, ${iris.data.shape[1]} features\n`);

// Simplify to binary classification: setosa (0) vs non-setosa (1)
const y_binary: number[] = [];
// Convert multi-class labels to binary
for (let i = 0; i < iris.target.size; i++) {
  const label = Number(iris.target.data[iris.target.offset + i]);
  // Map setosa to 0 and other classes to 1
  y_binary.push(label === 0 ? 0 : 1);
}
const y = tensor(y_binary);

// Split into train (70%) and test (30%) sets
const [X_train, X_test, y_train, y_test] = trainTestSplit(iris.data, y, {
  testSize: 0.3,
  randomState: 42,
});

console.log(`Training set: ${X_train.shape[0]} samples`);
console.log(`Test set: ${X_test.shape[0]} samples\n`);

// Standardize features (mean=0, std=1) for better convergence
const scaler = new StandardScaler();
scaler.fit(X_train);
const X_train_scaled = scaler.transform(X_train);
const X_test_scaled = scaler.transform(X_test);

console.log("Features scaled\n");

// Create and train logistic regression classifier
const model = new LogisticRegression({ maxIter: 1000, learningRate: 0.1 });
model.fit(X_train_scaled, y_train);

console.log("Model trained!\n");

// Make predictions on test data
const y_pred = model.predict(X_test_scaled);

// Calculate classification metrics
const acc = accuracy(y_test, y_pred);
const prec = precision(y_test, y_pred);
const rec = recall(y_test, y_pred);
const f1 = f1Score(y_test, y_pred);

console.log("Model Performance:");
console.log(`Accuracy:  ${(Number(acc) * 100).toFixed(2)}%`);
console.log(`Precision: ${(Number(prec) * 100).toFixed(2)}%`);
console.log(`Recall:    ${(Number(rec) * 100).toFixed(2)}%`);
console.log(`F1-Score:  ${(Number(f1) * 100).toFixed(2)}%\n`);

const cm = confusionMatrix(y_test, y_pred);
console.log("Confusion Matrix:");
console.log(cm.toString());

console.log("\n✓ Logistic regression complete!");
