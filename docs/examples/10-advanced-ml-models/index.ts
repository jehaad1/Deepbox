/**
 * Advanced ML Models Example
 *
 * Demonstrates the new ML models added to Deepbox:
 * - KMeans clustering
 * - K-Nearest Neighbors (classification and regression)
 * - PCA (dimensionality reduction)
 * - Gaussian Naive Bayes
 */

import { isNumericTypedArray, isTypedArray } from "deepbox/core";
import { accuracy } from "deepbox/metrics";
import { GaussianNB, KMeans, KNeighborsClassifier, KNeighborsRegressor, PCA } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { trainTestSplit } from "deepbox/preprocess";

const expectNumericTypedArray = (
  value: unknown
): Float32Array | Float64Array | Int32Array | Uint8Array => {
  if (!isTypedArray(value) || !isNumericTypedArray(value)) {
    throw new Error("Expected numeric typed array");
  }
  return value;
};

console.log("=".repeat(60));
console.log("Example 21: Advanced ML Models");
console.log("=".repeat(60));

// ============================================================================
// Part 1: KMeans Clustering
// ============================================================================
console.log("\n📦 Part 1: KMeans Clustering");
console.log("-".repeat(60));

const clusterData = tensor([
  [1, 2],
  [1.5, 1.8],
  [5, 8],
  [8, 8],
  [1, 0.6],
  [9, 11],
  [8, 2],
  [10, 2],
  [9, 3],
]);

const kmeans = new KMeans({ nClusters: 3, randomState: 42 });
kmeans.fit(clusterData);

const clusterLabels = kmeans.predict(clusterData);
console.log("Cluster labels:", clusterLabels.toString());
console.log("Cluster centers shape:", kmeans.clusterCenters.shape);
console.log("Inertia:", kmeans.inertia.toFixed(4));
console.log("Number of iterations:", kmeans.nIter);

// ============================================================================
// Part 2: K-Nearest Neighbors Classification
// ============================================================================
console.log("\n📦 Part 2: K-Nearest Neighbors Classification");
console.log("-".repeat(60));

const XClass = tensor([
  [0, 0],
  [1, 1],
  [2, 2],
  [3, 3],
  [4, 4],
  [5, 5],
  [6, 6],
  [7, 7],
]);
const yClass = tensor([0, 0, 0, 0, 1, 1, 1, 1]);

const [XTrainKNN, XTestKNN, yTrainKNN, yTestKNN] = trainTestSplit(XClass, yClass, {
  testSize: 0.25,
  randomState: 42,
});

const knnClassifier = new KNeighborsClassifier({ nNeighbors: 3 });
knnClassifier.fit(XTrainKNN, yTrainKNN);

const yPredKNN = knnClassifier.predict(XTestKNN);
const knnAccuracy = accuracy(yTestKNN, yPredKNN);

console.log("KNN Classifier trained with k=3");
console.log("Test accuracy:", `${(Number(knnAccuracy) * 100).toFixed(2)}%`);

const probabilities = knnClassifier.predictProba(XTestKNN);
console.log("Prediction probabilities shape:", probabilities.shape);

// ============================================================================
// Part 3: K-Nearest Neighbors Regression
// ============================================================================
console.log("\n📦 Part 3: K-Nearest Neighbors Regression");
console.log("-".repeat(60));

const XReg = tensor([[0], [1], [2], [3], [4], [5]]);
const yReg = tensor([0, 1, 4, 9, 16, 25]);

const knnRegressor = new KNeighborsRegressor({ nNeighbors: 2 });
knnRegressor.fit(XReg, yReg);

const yPredReg = knnRegressor.predict(tensor([[2.5], [3.5]]));
console.log("Predictions for [2.5] and [3.5]:", yPredReg.toString());

const r2Score = knnRegressor.score(XReg, yReg);
console.log("R² score:", r2Score.toFixed(4));

// ============================================================================
// Part 4: PCA (Dimensionality Reduction)
// ============================================================================
console.log("\n📦 Part 4: PCA - Dimensionality Reduction");
console.log("-".repeat(60));

const XPca = tensor([
  [2.5, 2.4, 1.1],
  [0.5, 0.7, 0.3],
  [2.2, 2.9, 1.5],
  [1.9, 2.2, 0.9],
  [3.1, 3.0, 1.8],
  [2.3, 2.7, 1.2],
  [2.0, 1.6, 0.8],
  [1.0, 1.1, 0.5],
  [1.5, 1.6, 0.7],
  [1.1, 0.9, 0.4],
]);

const pca = new PCA({ nComponents: 2 });
pca.fit(XPca);

const XTransformed = pca.transform(XPca);
console.log("Original shape:", XPca.shape);
console.log("Transformed shape:", XTransformed.shape);
console.log("Explained variance ratio:", pca.explainedVarianceRatio.toString());

const varianceData = expectNumericTypedArray(pca.explainedVarianceRatio.data);
const totalVariance = Array.from(varianceData).reduce((a, b) => a + b, 0);
console.log("Total variance explained:", `${(totalVariance * 100).toFixed(2)}%`);

// Reconstruct data
const XReconstructed = pca.inverseTransform(XTransformed);
console.log("Reconstructed shape:", XReconstructed.shape);

// ============================================================================
// Part 5: Gaussian Naive Bayes
// ============================================================================
console.log("\n📦 Part 5: Gaussian Naive Bayes");
console.log("-".repeat(60));

const XNB = tensor([
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [8, 9],
]);
const yNB = tensor([0, 0, 0, 0, 1, 1, 1, 1]);

const [XTrainNB, XTestNB, yTrainNB, yTestNB] = trainTestSplit(XNB, yNB, {
  testSize: 0.25,
  randomState: 42,
});

const nb = new GaussianNB();
nb.fit(XTrainNB, yTrainNB);

const yPredNB = nb.predict(XTestNB);
const nbAccuracy = accuracy(yTestNB, yPredNB);

console.log("Gaussian Naive Bayes trained");
console.log("Test accuracy:", `${(Number(nbAccuracy) * 100).toFixed(2)}%`);

const nbProba = nb.predictProba(XTestNB);
console.log("Prediction probabilities shape:", nbProba.shape);

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• KMeans: Unsupervised clustering for grouping similar data points");
console.log("• KNN: Instance-based learning for classification and regression");
console.log("• PCA: Dimensionality reduction while preserving variance");
console.log("• Naive Bayes: Probabilistic classifier based on Bayes' theorem");
console.log("• All models follow the fit/predict/score API (fit/predict/score)");

console.log("\n✅ Advanced ML Models Example Complete!");
console.log("=".repeat(60));
