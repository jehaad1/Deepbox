/**
 * Example 36: Kernel SVM & Anomaly Detection
 *
 * Kernel support vector machines (SVC, NuSVC, SVR) and anomaly detectors
 * (IsolationForest, LocalOutlierFactor, OneClassSVM).
 */

import { makeClassification, makeRegression } from "deepbox/datasets";
import { accuracy, r2Score } from "deepbox/metrics";
import { IsolationForest, LocalOutlierFactor, NuSVC, OneClassSVM, SVC, SVR } from "deepbox/ml";
import { type Tensor, tensor } from "deepbox/ndarray";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";

console.log("=".repeat(60));
console.log("Example 36: Kernel SVM & Anomaly Detection");
console.log("=".repeat(60));

// ============================================================================
// Generate datasets
// ============================================================================

const [XClass, yClass] = makeClassification({
  nSamples: 200,
  nFeatures: 6,
  nInformative: 4,
  nClasses: 2,
  randomState: 42,
});

const [XReg, yReg] = makeRegression({
  nSamples: 150,
  nFeatures: 5,
  noise: 0.5,
  randomState: 42,
});

const [XTrainRaw, XTestRaw, yTrain, yTest] = trainTestSplit(XClass, yClass, {
  testSize: 0.25,
  randomState: 42,
});

const [XTrainRegRaw, XTestRegRaw, yTrainR, yTestR] = trainTestSplit(XReg, yReg, {
  testSize: 0.25,
  randomState: 42,
});

// Kernel methods are sensitive to feature scale. Fit the scaler on the training
// rows only, then apply it to the test rows.
const scaler = new StandardScaler();
const XTrain = scaler.fitTransform(XTrainRaw);
const XTest = scaler.transform(XTestRaw);

const scalerReg = new StandardScaler();
const XTrainR = scalerReg.fitTransform(XTrainRegRaw);
const XTestR = scalerReg.transform(XTestRegRaw);

// ============================================================================
// Part 1: SVC with RBF Kernel
// ============================================================================
console.log("\nPart 1: SVC with RBF Kernel");
console.log("-".repeat(60));

// SVC separates the classes with the widest margin in a kernel-induced feature space
const svcRbf = new SVC({
  C: 1.0,
  kernel: "rbf",
  gamma: "scale",
});
svcRbf.fit(XTrain, yTrain);

const svcRbfPred = svcRbf.predict(XTest);
const svcRbfAcc = accuracy(yTest, svcRbfPred);

console.log("SVC (RBF kernel, C=1.0, gamma=scale):");
console.log(`  Accuracy: ${(Number(svcRbfAcc) * 100).toFixed(2)}%`);

// ============================================================================
// Part 2: SVC with Different Kernels
// ============================================================================
console.log("\nPart 2: SVC Kernel Comparison");
console.log("-".repeat(60));

// Compare different kernel functions
for (const kernel of ["linear", "poly", "rbf", "sigmoid"] as const) {
  const svc = new SVC({ C: 1.0, kernel, gamma: "scale" });
  svc.fit(XTrain, yTrain);
  const pred = svc.predict(XTest);
  const acc = accuracy(yTest, pred);
  console.log(`  ${kernel.padEnd(8)} kernel: Accuracy: ${(Number(acc) * 100).toFixed(2)}%`);
}

// ============================================================================
// Part 3: SVC with Regularization Tuning
// ============================================================================
console.log("\nPart 3: SVC Regularization (C parameter)");
console.log("-".repeat(60));

// A small C allows a wide margin with many errors, a large C punishes errors harder.
// At C=0.01 the model underfits, which shows in the accuracy.
for (const C of [0.01, 0.1, 1.0, 10.0, 100.0]) {
  const svc = new SVC({ C, kernel: "rbf", gamma: "scale" });
  svc.fit(XTrain, yTrain);
  const pred = svc.predict(XTest);
  const acc = accuracy(yTest, pred);
  console.log(`  C=${String(C).padEnd(6)}: Accuracy: ${(Number(acc) * 100).toFixed(2)}%`);
}

// ============================================================================
// Part 4: NuSVC
// ============================================================================
console.log("\nPart 4: NuSVC");
console.log("-".repeat(60));

// NuSVC replaces C with nu, a bound on the fraction of margin errors and of support vectors
const nuSvc = new NuSVC({
  nu: 0.5,
  kernel: "rbf",
  gamma: "scale",
});
nuSvc.fit(XTrain, yTrain);

const nuPred = nuSvc.predict(XTest);
const nuAcc = accuracy(yTest, nuPred);

console.log("NuSVC (nu=0.5, RBF kernel):");
console.log(`  Accuracy: ${(Number(nuAcc) * 100).toFixed(2)}%`);

// ============================================================================
// Part 5: SVR (Support Vector Regression)
// ============================================================================
console.log("\nPart 5: SVR (Support Vector Regression)");
console.log("-".repeat(60));

// SVR fits a function within an epsilon tube around the targets
const svr = new SVR({
  C: 1.0,
  kernel: "rbf",
  gamma: "scale",
});
svr.fit(XTrainR, yTrainR);

const svrPred = svr.predict(XTestR);
const svrR2 = r2Score(yTestR, svrPred);

console.log("SVR (RBF kernel, C=1.0):");
console.log(`  R² Score: ${Number(svrR2).toFixed(4)}`);

// Compare SVR kernels
for (const kernel of ["linear", "rbf", "poly"] as const) {
  const sv = new SVR({ C: 1.0, kernel, gamma: "scale" });
  sv.fit(XTrainR, yTrainR);
  const pred = sv.predict(XTestR);
  const r2 = r2Score(yTestR, pred);
  console.log(`  ${kernel.padEnd(8)} kernel: R²: ${Number(r2).toFixed(4)}`);
}

// ============================================================================
// Part 6: Isolation Forest (Anomaly Detection)
// ============================================================================
console.log("\nPart 6: Isolation Forest");
console.log("-".repeat(60));

// Rows 0 to 14 form one cluster. The last three rows (15, 16 and 17) are planted outliers.
// The detectors return 1 for an inlier and -1 for an outlier.
const outlierRows = (labels: Tensor): number[] =>
  (labels.toArray() as number[]).flatMap((label, row) => (label === -1 ? [row] : []));

const normalData = tensor([
  [1, 2],
  [1.5, 1.8],
  [1.2, 2.1],
  [0.8, 1.9],
  [1.3, 2.3],
  [2, 1],
  [1.8, 1.5],
  [2.2, 1.2],
  [1.9, 0.8],
  [2.1, 1.3],
  [1.5, 1.5],
  [1.7, 1.7],
  [1.3, 1.6],
  [1.6, 1.4],
  [1.4, 1.8],
  [10, 10],
  [-5, -5],
  [8, -3],
]);

// IsolationForest isolates points with random splits. Outliers need fewer splits.
const iforest = new IsolationForest({
  nEstimators: 100,
  contamination: 0.15,
  randomState: 42,
});
iforest.fit(normalData);

const ifoLabels = iforest.predict(normalData);
const ifoScores = iforest.scoreSamples(normalData);

console.log("Isolation Forest (100 trees, contamination=0.15):");
console.log(`  Anomaly scores: ${ifoScores.toString()}`);

const ifoOutliers = outlierRows(ifoLabels);
console.log(
  `  Outlier rows: [${ifoOutliers.join(", ")}] (${ifoOutliers.length} of ${normalData.shape[0]})`
);

// ============================================================================
// Part 7: Local Outlier Factor (LOF)
// ============================================================================
console.log("\nPart 7: Local Outlier Factor (LOF)");
console.log("-".repeat(60));

// LOF compares the density around a point with the density around its neighbors
const lof = new LocalOutlierFactor({
  nNeighbors: 5,
  contamination: 0.15,
});
lof.fit(normalData);

const lofLabels = lof.predict(normalData);

console.log("Local Outlier Factor (k=5, contamination=0.15):");

const lofOutliers = outlierRows(lofLabels);
console.log(
  `  Outlier rows: [${lofOutliers.join(", ")}] (${lofOutliers.length} of ${normalData.shape[0]})`
);

// ============================================================================
// Part 8: OneClassSVM
// ============================================================================
console.log("\nPart 8: OneClassSVM");
console.log("-".repeat(60));

// OneClassSVM learns a boundary around the bulk of the data
const ocsvm = new OneClassSVM({
  nu: 0.15,
  kernel: "rbf",
  gamma: "scale",
});
ocsvm.fit(normalData);

const ocLabels = ocsvm.predict(normalData);

console.log("OneClassSVM (nu=0.15, RBF kernel):");

const ocOutliers = outlierRows(ocLabels);
console.log(
  `  Outlier rows: [${ocOutliers.join(", ")}] (${ocOutliers.length} of ${normalData.shape[0]})`
);

// ============================================================================
// Summary
// ============================================================================
console.log("\nKey Takeaways");
console.log("-".repeat(60));
console.log("• SVC: classification with a linear, polynomial, RBF or sigmoid kernel");
console.log("• NuSVC: nu replaces C and bounds the fraction of support vectors");
console.log("• SVR: kernel regression");
console.log(
  "• Scale features before kernel SVMs (StandardScaler), fitting on the training rows only"
);
console.log("• IsolationForest: random splits, so outliers are isolated quickly");
console.log("• LOF: flags points that sit in a sparser region than their neighbors");
console.log("• OneClassSVM: a boundary around normal data in kernel space");
console.log("• Anomaly detectors return 1 for an inlier and -1 for an outlier");

console.log("\nKernel SVM & Anomaly Detection Example Complete!");
console.log("=".repeat(60));
