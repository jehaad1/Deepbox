/**
 * Example 10: Advanced ML Models
 *
 * Five models on tiny datasets, each shown with its own fit/predict/score calls:
 * - KMeans clustering
 * - K-Nearest Neighbors (classification and regression)
 * - PCA (dimensionality reduction)
 * - Gaussian Naive Bayes
 */

import { accuracy } from "deepbox/metrics";
import { GaussianNB, KMeans, KNeighborsClassifier, KNeighborsRegressor, PCA } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { trainTestSplit } from "deepbox/preprocess";

console.log("=".repeat(60));
console.log("Example 10: Advanced ML Models");
console.log("=".repeat(60));

// ============================================================================
// Part 1: KMeans Clustering
// ============================================================================
console.log("\nPart 1: KMeans Clustering");
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
console.log("\nPart 2: K-Nearest Neighbors Classification");
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
console.log("Test accuracy:", `${(knnAccuracy * 100).toFixed(2)}%`);

const probabilities = knnClassifier.predictProba(XTestKNN);
console.log("Prediction probabilities shape:", probabilities.shape);

// clone() returns an unfitted estimator with the same settings.
const knnFresh = knnClassifier.clone();
knnFresh.fit(XClass, yClass);
console.log(
  "Cloned classifier fitted on all rows, predicts:",
  knnFresh
    .predict(
      tensor([
        [0.5, 0.5],
        [6.5, 6.5],
      ])
    )
    .toString()
);

// ============================================================================
// Part 3: K-Nearest Neighbors Regression
// ============================================================================
console.log("\nPart 3: K-Nearest Neighbors Regression");
console.log("-".repeat(60));

const XReg = tensor([[0], [1], [2], [3], [4], [5]]);
const yReg = tensor([0, 1, 4, 9, 16, 25]);

const knnRegressor = new KNeighborsRegressor({ nNeighbors: 2 });
knnRegressor.fit(XReg, yReg);

const yPredReg = knnRegressor.predict(tensor([[2.5], [3.5]]));
console.log("Predictions for [2.5] and [3.5]:", yPredReg.toString());

const knnR2 = knnRegressor.score(XReg, yReg);
console.log("R² score:", knnR2.toFixed(4));

// ============================================================================
// Part 4: PCA (Dimensionality Reduction)
// ============================================================================
console.log("\nPart 4: PCA - Dimensionality Reduction");
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

const totalVariance = Number(pca.explainedVarianceRatio.sum().item());
console.log("Total variance explained:", `${(totalVariance * 100).toFixed(2)}%`);

// Reconstruct data
const XReconstructed = pca.inverseTransform(XTransformed);
console.log("Reconstructed shape:", XReconstructed.shape);

// ============================================================================
// Part 5: Gaussian Naive Bayes
// ============================================================================
console.log("\nPart 5: Gaussian Naive Bayes");
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
console.log("Test accuracy:", `${(nbAccuracy * 100).toFixed(2)}%`);

const nbProba = nb.predictProba(XTestNB);
console.log("Prediction probabilities shape:", nbProba.shape);

// ============================================================================
// Summary
// ============================================================================
console.log("\nSummary");
console.log("-".repeat(60));
console.log("KMeans: unsupervised clustering of similar points");
console.log("KNN: instance-based classification and regression");
console.log("PCA: dimensionality reduction that keeps as much variance as possible");
console.log("Naive Bayes: probabilistic classifier based on Bayes' theorem");
console.log("Supervised models share fit(X, y), predict(X) and score(X, y).");
console.log("=".repeat(60));
