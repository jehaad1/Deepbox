import { describe, expect, it } from "vitest";
import {
  accuracy,
  averagePrecisionScore,
  classificationReport,
  explainedVarianceScore,
  f1Score,
  fbetaScore,
  jaccardScore,
  logLoss,
  mape,
  matthewsCorrcoef,
  mse,
  precision,
  precisionRecallCurve,
  r2Score,
  recall,
  silhouetteSamples,
  silhouetteScore,
} from "../src/metrics";
import { tensor } from "../src/ndarray";

describe("metrics regression contracts", () => {
  it("rejects non-binary labels for binary-only classification metrics", () => {
    expect(() => logLoss(tensor([0, 2]), tensor([0.2, 0.8]))).toThrow(/binary/i);
    expect(() => jaccardScore(tensor([2, 2, 3]), tensor([2, 3, 3]))).toThrow(/binary/i);
    expect(() => matthewsCorrcoef(tensor([0, 2]), tensor([0, 1]))).toThrow(/binary/i);
    expect(() => precisionRecallCurve(tensor([0, 2]), tensor([0.1, 0.9]))).toThrow(/binary/i);
    expect(() => averagePrecisionScore(tensor([0, 2]), tensor([0.1, 0.9]))).toThrow(/binary/i);
    expect(() => precision(tensor([0, 2]), tensor([0, 1]), "binary")).toThrow(/binary/i);
    expect(() => recall(tensor([0, 2]), tensor([0, 1]), "binary")).toThrow(/binary/i);
    expect(() => f1Score(tensor([0, 2]), tensor([0, 1]), "binary")).toThrow(/binary/i);
    expect(() => fbetaScore(tensor([0, 2]), tensor([0, 1]), 1, "binary")).toThrow(/binary/i);
    expect(() => classificationReport(tensor([0, 2]), tensor([0, 1]))).toThrow(/binary/i);
  });

  it("uses only non-zero targets in mape denominator", () => {
    expect(mape(tensor([0, 2]), tensor([10, 4]))).toBe(100);
  });

  it("enforces silhouette cluster-count and metric preconditions", () => {
    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
    ]);
    const singleCluster = tensor([0, 0, 0]);
    expect(() => silhouetteScore(X, singleCluster)).toThrow(/n_clusters/i);
    expect(() => silhouetteSamples(X, singleCluster)).toThrow(/n_clusters/i);
    // @ts-expect-error - deliberately passing unsupported metric for runtime validation
    expect(() => silhouetteScore(X, tensor([0, 1, 1]), "manhattan")).toThrow(/euclidean/i);
  });

  it("rejects empty inputs for r2/explained variance", () => {
    const empty = tensor([]);
    expect(() => r2Score(empty, empty)).toThrow(/at least one sample/i);
    expect(() => explainedVarianceScore(empty, empty)).toThrow(/at least one sample/i);
  });

  it("rejects non-vector shapes for classification metrics", () => {
    const yTrue = tensor([
      [0, 1],
      [1, 0],
    ]);
    const yPred = tensor([
      [0, 1],
      [1, 0],
    ]);

    expect(() => accuracy(yTrue, yPred)).toThrow(/column vector|1D/i);
    expect(() => precision(yTrue, yPred, "binary")).toThrow(/column vector|1D/i);
  });

  it("rejects non-vector shapes for regression metrics", () => {
    const yTrue = tensor([
      [1, 2],
      [3, 4],
    ]);
    const yPred = tensor([
      [1, 2],
      [3, 4],
    ]);

    expect(() => mse(yTrue, yPred)).toThrow(/column vector|1D/i);
    expect(() => r2Score(yTrue, yPred)).toThrow(/column vector|1D/i);
  });
});
