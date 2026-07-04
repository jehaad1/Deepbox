/**
 * Example 36: Kernel SVM & Anomaly Detection
 *
 * New in v1.0.0: Support Vector Machines with kernel tricks (SVC, SVR, NuSVC,
 * NuSVR, OneClassSVM) and anomaly detection (IsolationForest, LocalOutlierFactor).
 */

import { makeClassification, makeRegression } from "deepbox/datasets";
import { accuracy, r2Score } from "deepbox/metrics";
import { IsolationForest, LocalOutlierFactor, NuSVC, OneClassSVM, SVC, SVR } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
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
  noise: 5,
  randomState: 42,
});

// Scale features for SVM (important for kernel methods)
const scaler = new StandardScaler();
const XClassScaled = scaler.fitTransform(XClass);

const scalerReg = new StandardScaler();
const XRegScaled = scalerReg.fitTransform(XReg);

const [XTrain, XTest, yTrain, yTest] = trainTestSplit(XClassScaled, yClass, {
  testSize: 0.25,
  randomState: 42,
});

const [XTrainR, XTestR, yTrainR, yTestR] = trainTestSplit(XRegScaled, yReg, {
  testSize: 0.25,
  randomState: 42,
});

// ============================================================================
// Part 1: SVC with RBF Kernel
// ============================================================================
console.log("\n🎯 Part 1: SVC with RBF Kernel");
console.log("-".repeat(60));

// SVC uses the kernel trick to find nonlinear decision boundaries
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
console.log("\n🔀 Part 2: SVC Kernel Comparison");
console.log("-".repeat(60));

// Compare different kernel functions
for (const kernel of ["linear", "poly", "rbf", "sigmoid"] as const) {
  const svc = new SVC({ C: 1.0, kernel, gamma: "scale" });
  svc.fit(XTrain, yTrain);
  const pred = svc.predict(XTest);
  const acc = accuracy(yTest, pred);
  console.log(`  ${kernel.padEnd(8)} kernel — Accuracy: ${(Number(acc) * 100).toFixed(2)}%`);
}

// ============================================================================
// Part 3: SVC with Regularization Tuning
// ============================================================================
console.log("\n⚙️  Part 3: SVC Regularization (C parameter)");
console.log("-".repeat(60));

// C controls the trade-off between margin width and classification error
for (const C of [0.01, 0.1, 1.0, 10.0, 100.0]) {
  const svc = new SVC({ C, kernel: "rbf", gamma: "scale" });
  svc.fit(XTrain, yTrain);
  const pred = svc.predict(XTest);
  const acc = accuracy(yTest, pred);
  console.log(`  C=${String(C).padEnd(6)} — Accuracy: ${(Number(acc) * 100).toFixed(2)}%`);
}

// ============================================================================
// Part 4: NuSVC
// ============================================================================
console.log("\n📊 Part 4: NuSVC");
console.log("-".repeat(60));

// NuSVC uses nu parameter instead of C to control the number of support vectors
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
console.log("\n📈 Part 5: SVR (Support Vector Regression)");
console.log("-".repeat(60));

// SVR applies the kernel trick to regression problems
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
  console.log(`  ${kernel.padEnd(8)} kernel — R²: ${Number(r2).toFixed(4)}`);
}

// ============================================================================
// Part 6: Isolation Forest (Anomaly Detection)
// ============================================================================
console.log("\n🌲 Part 6: Isolation Forest");
console.log("-".repeat(60));

// Create normal data with some outliers
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
  [8, -3], // outliers
]);

// IsolationForest detects anomalies by how quickly points are isolated
const iforest = new IsolationForest({
  nEstimators: 100,
  contamination: 0.15,
  randomState: 42,
});
iforest.fit(normalData);

const ifoLabels = iforest.predict(normalData);
const ifoScores = iforest.scoreSamples(normalData);

console.log("Isolation Forest (100 trees, contamination=0.15):");
console.log(`  Labels (1=inlier, -1=outlier): ${ifoLabels.toString()}`);
console.log(`  Anomaly scores: ${ifoScores.toString()}`);

// Count detected outliers
const labelData = ifoLabels.data as Int32Array;
let outlierCount = 0;
for (let i = 0; i < labelData.length; i++) {
  if (labelData[i] === -1) outlierCount++;
}
console.log(`  Detected ${outlierCount} outliers out of ${normalData.shape[0]} samples`);

// ============================================================================
// Part 7: Local Outlier Factor (LOF)
// ============================================================================
console.log("\n🔍 Part 7: Local Outlier Factor (LOF)");
console.log("-".repeat(60));

// LOF measures local deviation of density compared to neighbors
const lof = new LocalOutlierFactor({
  nNeighbors: 5,
  contamination: 0.15,
});
lof.fit(normalData);

const lofLabels = lof.predict(normalData);

console.log("Local Outlier Factor (k=5, contamination=0.15):");
console.log(`  Labels (1=inlier, -1=outlier): ${lofLabels.toString()}`);

const lofLabelData = lofLabels.data as Int32Array;
let lofOutlierCount = 0;
for (let i = 0; i < lofLabelData.length; i++) {
  if (lofLabelData[i] === -1) lofOutlierCount++;
}
console.log(`  Detected ${lofOutlierCount} outliers out of ${normalData.shape[0]} samples`);

// ============================================================================
// Part 8: OneClassSVM
// ============================================================================
console.log("\n🛡️  Part 8: OneClassSVM");
console.log("-".repeat(60));

// OneClassSVM learns a boundary around normal data points
const ocsvm = new OneClassSVM({
  nu: 0.15,
  kernel: "rbf",
  gamma: "scale",
});
ocsvm.fit(normalData);

const ocLabels = ocsvm.predict(normalData);

console.log("OneClassSVM (nu=0.15, RBF kernel):");
console.log(`  Labels (1=inlier, -1=outlier): ${ocLabels.toString()}`);

const ocLabelData = ocLabels.data as Int32Array;
let ocOutlierCount = 0;
for (let i = 0; i < ocLabelData.length; i++) {
  if (ocLabelData[i] === -1) ocOutlierCount++;
}
console.log(`  Detected ${ocOutlierCount} outliers out of ${normalData.shape[0]} samples`);

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• SVC: kernel-based classification with RBF, polynomial, linear, sigmoid kernels");
console.log("• NuSVC: nu parameter controls fraction of support vectors (alternative to C)");
console.log("• SVR: kernel regression for nonlinear relationships");
console.log("• Feature scaling is crucial for kernel SVMs (use StandardScaler)");
console.log("• IsolationForest: tree-based anomaly detection, fast and scalable");
console.log("• LOF: density-based anomaly detection, measures local deviation");
console.log("• OneClassSVM: learns a boundary around normal data in kernel space");
console.log("• Anomaly detectors output +1 (inlier) and -1 (outlier)");

console.log("\n✅ Kernel SVM & Anomaly Detection Example Complete!");
console.log("=".repeat(60));
