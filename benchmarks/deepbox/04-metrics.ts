/**
 * Benchmark 04: Metrics
 * Deepbox vs scikit-learn
 */

import {
  accuracy,
  adjustedMutualInfoScore,
  adjustedR2Score,
  adjustedRandScore,
  averagePrecisionScore,
  balancedAccuracyScore,
  brierScoreLoss,
  calinskiHarabaszScore,
  cohenKappaScore,
  completenessScore,
  confusionMatrix,
  coverageError,
  daviesBouldinScore,
  explainedVarianceScore,
  f1Score,
  fbetaScore,
  fowlkesMallowsScore,
  hammingLoss,
  homogeneityScore,
  jaccardScore,
  logLoss,
  mae,
  mape,
  matthewsCorrcoef,
  maxError,
  meanPinballLoss,
  medianAbsoluteError,
  mse,
  ndcgScore,
  normalizedMutualInfoScore,
  pairwiseCosine,
  pairwiseEuclidean,
  pairwiseManhattan,
  precision,
  r2Score,
  recall,
  rmse,
  rocAucScore,
  silhouetteScore,
  topKAccuracyScore,
  vMeasureScore,
  zeroOneLoss,
} from "deepbox/metrics";
import { tensor } from "deepbox/ndarray";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("metrics");
header("Benchmark 04: Metrics");

// ── Data generators ──────────────────────────────────────

function seededRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function binaryLabels(n: number, seed: number) {
  const rand = seededRng(seed);
  const yt: number[] = [],
    yp: number[] = [];
  for (let i = 0; i < n; i++) {
    yt.push(rand() > 0.5 ? 1 : 0);
    yp.push(rand() > 0.4 ? 1 : 0);
  }
  return { yt: tensor(yt), yp: tensor(yp) };
}

function binaryProbs(n: number, seed: number) {
  const rand = seededRng(seed);
  const yt: number[] = [],
    yp: number[] = [];
  for (let i = 0; i < n; i++) {
    yt.push(rand() > 0.5 ? 1 : 0);
    yp.push(Math.max(0.001, Math.min(0.999, rand())));
  }
  return { yt: tensor(yt), yp: tensor(yp) };
}

function regressionPreds(n: number, seed: number) {
  const rand = seededRng(seed);
  const yt: number[] = [],
    yp: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = rand() * 10;
    yt.push(v);
    yp.push(v + (rand() - 0.5) * 2);
  }
  return { yt: tensor(yt), yp: tensor(yp) };
}

function clusterData(n: number, dims: number, k: number, seed: number) {
  const rand = seededRng(seed);
  const X: number[][] = [],
    labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % k;
    labels.push(c);
    const row: number[] = [];
    for (let d = 0; d < dims; d++) row.push(c * 5 + (rand() * 2 - 1));
    X.push(row);
  }
  return { X: tensor(X), labels: tensor(labels) };
}

function clusterLabels(n: number, k: number, seed: number) {
  const rand = seededRng(seed);
  const a: number[] = [],
    b: number[] = [];
  for (let i = 0; i < n; i++) {
    a.push(i % k);
    b.push(Math.floor(rand() * k));
  }
  return { a: tensor(a), b: tensor(b) };
}

const bin1k = binaryLabels(1000, 42);
const bin10k = binaryLabels(10000, 123);
const prob1k = binaryProbs(1000, 42);
const prob10k = binaryProbs(10000, 123);
const reg1k = regressionPreds(1000, 42);
const reg10k = regressionPreds(10000, 123);
const clust200 = clusterData(200, 5, 3, 42);
const clust500 = clusterData(500, 5, 4, 123);
const clab500 = clusterLabels(500, 4, 42);
const clab1k = clusterLabels(1000, 5, 123);
const rankTrue = tensor([
  [1, 0, 1, 0],
  [0, 1, 0, 1],
  [1, 0, 0, 1],
]);
const rankScore = tensor([
  [0.9, 0.2, 0.8, 0.1],
  [0.3, 0.7, 0.2, 0.6],
  [0.8, 0.1, 0.4, 0.7],
]);
const multiclassTrue = tensor(Array.from({ length: 1000 }, (_, i) => i % 4));
const multiclassScore = tensor(
  Array.from({ length: 1000 }, (_, i) => {
    const label = i % 4;
    return Array.from({ length: 4 }, (_, j) => (j === label ? 0.7 : 0.1));
  })
);
const pairwiseX = tensor(
  Array.from({ length: 300 }, (_, i) =>
    Array.from({ length: 8 }, (_, j) => ((i * 3 + j * 5) % 17) / 17)
  )
);

// ── Classification Metrics ──────────────────────────────

run(suite, "accuracy", "1K", () => accuracy(bin1k.yt, bin1k.yp));
run(suite, "accuracy", "10K", () => accuracy(bin10k.yt, bin10k.yp));
run(suite, "precision", "1K", () => precision(bin1k.yt, bin1k.yp));
run(suite, "precision", "10K", () => precision(bin10k.yt, bin10k.yp));
run(suite, "recall", "1K", () => recall(bin1k.yt, bin1k.yp));
run(suite, "recall", "10K", () => recall(bin10k.yt, bin10k.yp));
run(suite, "f1Score", "1K", () => f1Score(bin1k.yt, bin1k.yp));
run(suite, "f1Score", "10K", () => f1Score(bin10k.yt, bin10k.yp));
run(suite, "fbetaScore (β=0.5)", "1K", () => fbetaScore(bin1k.yt, bin1k.yp, 0.5));
run(suite, "fbetaScore (β=0.5)", "10K", () => fbetaScore(bin10k.yt, bin10k.yp, 0.5));
run(suite, "confusionMatrix", "1K", () => confusionMatrix(bin1k.yt, bin1k.yp));
run(suite, "confusionMatrix", "10K", () => confusionMatrix(bin10k.yt, bin10k.yp));
run(suite, "hammingLoss", "1K", () => hammingLoss(bin1k.yt, bin1k.yp));
run(suite, "hammingLoss", "10K", () => hammingLoss(bin10k.yt, bin10k.yp));
run(suite, "jaccardScore", "1K", () => jaccardScore(bin1k.yt, bin1k.yp));
run(suite, "jaccardScore", "10K", () => jaccardScore(bin10k.yt, bin10k.yp));
run(suite, "cohenKappaScore", "1K", () => cohenKappaScore(bin1k.yt, bin1k.yp));
run(suite, "cohenKappaScore", "10K", () => cohenKappaScore(bin10k.yt, bin10k.yp));
run(suite, "matthewsCorrcoef", "1K", () => matthewsCorrcoef(bin1k.yt, bin1k.yp));
run(suite, "matthewsCorrcoef", "10K", () => matthewsCorrcoef(bin10k.yt, bin10k.yp));
run(suite, "balancedAccuracy", "1K", () => balancedAccuracyScore(bin1k.yt, bin1k.yp));
run(suite, "balancedAccuracy", "10K", () => balancedAccuracyScore(bin10k.yt, bin10k.yp));
run(suite, "logLoss", "1K", () => logLoss(prob1k.yt, prob1k.yp));
run(suite, "logLoss", "10K", () => logLoss(prob10k.yt, prob10k.yp));
run(suite, "rocAucScore", "1K", () => rocAucScore(prob1k.yt, prob1k.yp));
run(suite, "rocAucScore", "10K", () => rocAucScore(prob10k.yt, prob10k.yp));
run(suite, "averagePrecision", "1K", () => averagePrecisionScore(prob1k.yt, prob1k.yp));
run(suite, "averagePrecision", "10K", () => averagePrecisionScore(prob10k.yt, prob10k.yp));
run(suite, "brierScoreLoss", "1K", () => brierScoreLoss(prob1k.yt, prob1k.yp));
run(suite, "zeroOneLoss", "1K", () => zeroOneLoss(bin1k.yt, bin1k.yp));
run(suite, "topKAccuracyScore", "1Kx4", () =>
  topKAccuracyScore(multiclassTrue, multiclassScore, 2)
);
run(suite, "coverageError", "3x4", () => coverageError(rankTrue, rankScore));
run(suite, "ndcgScore", "3x4", () => ndcgScore(rankTrue, rankScore));

// ── Regression Metrics ──────────────────────────────────

run(suite, "mse", "1K", () => mse(reg1k.yt, reg1k.yp));
run(suite, "mse", "10K", () => mse(reg10k.yt, reg10k.yp));
run(suite, "rmse", "1K", () => rmse(reg1k.yt, reg1k.yp));
run(suite, "rmse", "10K", () => rmse(reg10k.yt, reg10k.yp));
run(suite, "mae", "1K", () => mae(reg1k.yt, reg1k.yp));
run(suite, "mae", "10K", () => mae(reg10k.yt, reg10k.yp));
run(suite, "r2Score", "1K", () => r2Score(reg1k.yt, reg1k.yp));
run(suite, "r2Score", "10K", () => r2Score(reg10k.yt, reg10k.yp));
run(suite, "adjustedR2Score", "1K", () => adjustedR2Score(reg1k.yt, reg1k.yp, 5));
run(suite, "adjustedR2Score", "10K", () => adjustedR2Score(reg10k.yt, reg10k.yp, 10));
run(suite, "explainedVariance", "1K", () => explainedVarianceScore(reg1k.yt, reg1k.yp));
run(suite, "explainedVariance", "10K", () => explainedVarianceScore(reg10k.yt, reg10k.yp));
run(suite, "maxError", "1K", () => maxError(reg1k.yt, reg1k.yp));
run(suite, "maxError", "10K", () => maxError(reg10k.yt, reg10k.yp));
run(suite, "medianAbsoluteError", "1K", () => medianAbsoluteError(reg1k.yt, reg1k.yp));
run(suite, "medianAbsoluteError", "10K", () => medianAbsoluteError(reg10k.yt, reg10k.yp));
run(suite, "mape", "1K", () => mape(reg1k.yt, reg1k.yp));
run(suite, "mape", "10K", () => mape(reg10k.yt, reg10k.yp));
run(suite, "meanPinballLoss", "1K", () => meanPinballLoss(reg1k.yt, reg1k.yp, 0.9));

// ── Clustering Metrics ──────────────────────────────────

run(suite, "silhouetteScore", "200x5 k=3", () => silhouetteScore(clust200.X, clust200.labels));
run(suite, "silhouetteScore", "500x5 k=4", () => silhouetteScore(clust500.X, clust500.labels));
run(suite, "calinskiHarabasz", "200x5 k=3", () =>
  calinskiHarabaszScore(clust200.X, clust200.labels)
);
run(suite, "calinskiHarabasz", "500x5 k=4", () =>
  calinskiHarabaszScore(clust500.X, clust500.labels)
);
run(suite, "daviesBouldin", "200x5 k=3", () => daviesBouldinScore(clust200.X, clust200.labels));
run(suite, "daviesBouldin", "500x5 k=4", () => daviesBouldinScore(clust500.X, clust500.labels));
run(suite, "adjustedRandScore", "500", () => adjustedRandScore(clab500.a, clab500.b));
run(suite, "adjustedRandScore", "1K", () => adjustedRandScore(clab1k.a, clab1k.b));
run(suite, "adjustedMutualInfo", "500", () => adjustedMutualInfoScore(clab500.a, clab500.b));
run(suite, "adjustedMutualInfo", "1K", () => adjustedMutualInfoScore(clab1k.a, clab1k.b));
run(suite, "normalizedMutualInfo", "500", () => normalizedMutualInfoScore(clab500.a, clab500.b));
run(suite, "normalizedMutualInfo", "1K", () => normalizedMutualInfoScore(clab1k.a, clab1k.b));
run(suite, "homogeneityScore", "500", () => homogeneityScore(clab500.a, clab500.b));
run(suite, "completenessScore", "500", () => completenessScore(clab500.a, clab500.b));
run(suite, "vMeasureScore", "500", () => vMeasureScore(clab500.a, clab500.b));
run(suite, "fowlkesMallows", "500", () => fowlkesMallowsScore(clab500.a, clab500.b));
run(suite, "fowlkesMallows", "1K", () => fowlkesMallowsScore(clab1k.a, clab1k.b));
run(suite, "pairwiseEuclidean", "300x8", () => pairwiseEuclidean(pairwiseX));
run(suite, "pairwiseCosine", "300x8", () => pairwiseCosine(pairwiseX));
run(suite, "pairwiseManhattan", "300x8", () => pairwiseManhattan(pairwiseX));

// ── Extended coverage (v1.1 benchmark expansion) ────────
// More sizes for classification & regression metrics.
const bin100 = binaryLabels(100, 1);
const bin500 = binaryLabels(500, 2);
const bin5k = binaryLabels(5000, 3);
const prob100 = binaryProbs(100, 1);
const prob500 = binaryProbs(500, 2);
const prob5k = binaryProbs(5000, 3);
const reg100 = regressionPreds(100, 1);
const reg500 = regressionPreds(500, 2);
const reg5k = regressionPreds(5000, 3);
const clsSizes: [string, ReturnType<typeof binaryLabels>][] = [
  ["100", bin100],
  ["500", bin500],
  ["5K", bin5k],
];
const probSizes: [string, ReturnType<typeof binaryProbs>][] = [
  ["100", prob100],
  ["500", prob500],
  ["5K", prob5k],
];
const regSizes: [string, ReturnType<typeof regressionPreds>][] = [
  ["100", reg100],
  ["500", reg500],
  ["5K", reg5k],
];
for (const [sz, d] of clsSizes) {
  run(suite, "accuracy", sz, () => accuracy(d.yt, d.yp));
  run(suite, "precision", sz, () => precision(d.yt, d.yp));
  run(suite, "recall", sz, () => recall(d.yt, d.yp));
  run(suite, "f1Score", sz, () => f1Score(d.yt, d.yp));
  run(suite, "hammingLoss", sz, () => hammingLoss(d.yt, d.yp));
  run(suite, "jaccardScore", sz, () => jaccardScore(d.yt, d.yp));
  run(suite, "cohenKappaScore", sz, () => cohenKappaScore(d.yt, d.yp));
  run(suite, "matthewsCorrcoef", sz, () => matthewsCorrcoef(d.yt, d.yp));
  run(suite, "balancedAccuracy", sz, () => balancedAccuracyScore(d.yt, d.yp));
  run(suite, "zeroOneLoss", sz, () => zeroOneLoss(d.yt, d.yp));
}
for (const [sz, d] of probSizes) {
  run(suite, "logLoss", sz, () => logLoss(d.yt, d.yp));
  run(suite, "brierScoreLoss", sz, () => brierScoreLoss(d.yt, d.yp));
}
for (const [sz, d] of regSizes) {
  run(suite, "mse", sz, () => mse(d.yt, d.yp));
  run(suite, "rmse", sz, () => rmse(d.yt, d.yp));
  run(suite, "mae", sz, () => mae(d.yt, d.yp));
  run(suite, "r2Score", sz, () => r2Score(d.yt, d.yp));
  run(suite, "explainedVariance", sz, () => explainedVarianceScore(d.yt, d.yp));
  run(suite, "maxError", sz, () => maxError(d.yt, d.yp));
  run(suite, "mape", sz, () => mape(d.yt, d.yp));
}

footer(suite, "deepbox-metrics.json");
