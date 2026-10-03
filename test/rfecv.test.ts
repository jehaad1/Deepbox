import { describe, expect, it } from "vitest";
import { type Tensor, tensor } from "../src/ndarray";
import { RFECV } from "../src/preprocess/feature_selection";

/**
 * Mock estimator that exposes featureImportances_ and predict()
 * for use with RFECV cross-validation.
 */
class MockScoringEstimator {
  featureImportances_: number[] | undefined;
  private readonly importances: number[];
  private lastY: Tensor | undefined;

  constructor(importances: number[]) {
    this.importances = importances;
  }

  fit(X: Tensor, y: Tensor): this {
    // Importances follow the original columns, identified by their first-row value.
    const keys = [1, 10, 100, 1000];
    const firstRow = (X.toArray() as number[][])[0] ?? [];
    this.featureImportances_ = firstRow.map((v) => this.importances[keys.indexOf(v)] ?? 0);
    this.lastY = y;
    return this;
  }

  predict(X: Tensor): Tensor {
    // Simple: always predict the majority class from training y
    const n = X.shape[0] ?? 0;
    if (!this.lastY) return tensor(new Array(n).fill(0));
    // Count classes
    const counts = new Map<number, number>();
    for (let i = 0; i < this.lastY.size; i++) {
      const v = Number(this.lastY.data[this.lastY.offset + i]);
      counts.set(v, (counts.get(v) ?? 0) + 1);
    }
    let majorityClass = 0;
    let maxCount = 0;
    for (const [cls, cnt] of counts) {
      if (cnt > maxCount) {
        maxCount = cnt;
        majorityClass = cls;
      }
    }
    return tensor(new Array(n).fill(majorityClass));
  }
}

describe("RFECV", () => {
  it("selects optimal number of features via cross-validation", () => {
    const est = new MockScoringEstimator([0.1, 0.5, 0.3, 0.9]);
    const X = tensor([
      [1, 10, 100, 1000],
      [2, 20, 200, 2000],
      [3, 30, 300, 3000],
      [4, 40, 400, 4000],
      [1, 10, 100, 1000],
      [2, 20, 200, 2000],
    ]);
    const y = tensor([0, 0, 1, 1, 0, 1]);

    const rfecv = new RFECV({
      estimator: est,
      cv: 2,
      step: 1,
      minFeaturesToSelect: 1,
    });

    rfecv.fit(X, y);

    const nOpt = rfecv.nFeatures;
    expect(nOpt).toBeGreaterThanOrEqual(1);
    expect(nOpt).toBeLessThanOrEqual(4);

    const support = rfecv.getSupport();
    expect(support.length).toBe(4);
    expect(support.filter(Boolean).length).toBe(nOpt);

    const ranking = rfecv.ranking;
    expect(ranking.length).toBe(4);

    const scores = rfecv.gridScores;
    expect(scores.size).toBeGreaterThanOrEqual(1);
  });

  it("transforms data to selected features", () => {
    const est = new MockScoringEstimator([0.1, 0.9, 0.5]);
    const X = tensor([
      [1, 10, 100],
      [2, 20, 200],
      [3, 30, 300],
      [4, 40, 400],
      [5, 50, 500],
      [6, 60, 600],
    ]);
    const y = tensor([0, 1, 0, 1, 0, 1]);

    const rfecv = new RFECV({
      estimator: est,
      cv: 2,
      step: 1,
    });

    const Xt = rfecv.fitTransform(X, y);
    const nSelected = rfecv.nFeatures;
    expect(Xt.shape).toEqual([6, nSelected]);
  });

  it("throws on invalid constructor params", () => {
    const est = new MockScoringEstimator([]);
    expect(() => new RFECV({ estimator: est, cv: 1 })).toThrow();

    expect(() => new RFECV({ estimator: est, step: 0 })).toThrow();

    expect(() => new RFECV({ estimator: est, minFeaturesToSelect: 0 })).toThrow();
  });

  it("throws NotFittedError before fitting", () => {
    const est = new MockScoringEstimator([]);
    const rfecv = new RFECV({ estimator: est });
    expect(() => rfecv.transform(tensor([[1, 2]]))).toThrow();
    expect(() => rfecv.getSupport()).toThrow();
    expect(() => rfecv.ranking).toThrow();
    expect(() => rfecv.nFeatures).toThrow();
    expect(() => rfecv.gridScores).toThrow();
  });

  it("getParams returns correct values", () => {
    const est = new MockScoringEstimator([]);
    const rfecv = new RFECV({
      estimator: est,
      cv: 3,
      step: 2,
      minFeaturesToSelect: 2,
    });
    const params = rfecv.getParams();
    expect(params.cv).toBe(3);
    expect(params.step).toBe(2);
    expect(params.minFeaturesToSelect).toBe(2);
  });
});
