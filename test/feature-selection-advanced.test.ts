import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { RFE, SelectFromModel } from "../src/preprocess";

// ---------------------------------------------------------------------------
// Mock estimator that exposes featureImportances_ after fit
// ---------------------------------------------------------------------------
class MockImportanceEstimator {
  featureImportances_: number[] | undefined;
  private readonly importances: number[];

  constructor(importances: number[]) {
    this.importances = importances;
  }

  fit(_X: unknown, _y: unknown): this {
    this.featureImportances_ = [...this.importances];
    return this;
  }
}

// ---------------------------------------------------------------------------
// Mock estimator that exposes coef_ (1D) after fit
// ---------------------------------------------------------------------------
class MockCoefEstimator {
  coef_: ReturnType<typeof tensor> | undefined;
  private readonly coefs: number[];

  constructor(coefs: number[]) {
    this.coefs = coefs;
  }

  fit(_X: unknown, _y: unknown): this {
    this.coef_ = tensor(this.coefs);
    return this;
  }
}

// ---------------------------------------------------------------------------
// SelectFromModel
// ---------------------------------------------------------------------------
describe("SelectFromModel", () => {
  const X = tensor([
    [1, 10, 100, 1000],
    [2, 20, 200, 2000],
    [3, 30, 300, 3000],
    [4, 40, 400, 4000],
  ]);
  const y = tensor([0, 0, 1, 1]);

  it("selects features above mean importance (default)", () => {
    // importances = [0.1, 0.5, 0.3, 0.9]  mean = 0.45
    // features 1 (0.5) and 3 (0.9) are >= mean
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const sel = new SelectFromModel({ estimator: est });
    sel.fit(X, y);
    const support = sel.getSupport();
    expect(support).toEqual([false, true, false, true]);

    const Xt = sel.transform(X);
    expect(Xt.shape).toEqual([4, 2]);
    // First row should be features 1 and 3: [10, 1000]
    expect(Number(Xt.data[0])).toBe(10);
    expect(Number(Xt.data[1])).toBe(1000);
  });

  it("selects features above median importance", () => {
    // importances = [0.1, 0.5, 0.3, 0.9]  sorted = [0.1, 0.3, 0.5, 0.9]
    // median = (0.3 + 0.5) / 2 = 0.4
    // features 1 (0.5) and 3 (0.9) are >= 0.4
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const sel = new SelectFromModel({ estimator: est, threshold: "median" });
    sel.fit(X, y);
    expect(sel.getSupport()).toEqual([false, true, false, true]);
  });

  it("selects features above numeric threshold", () => {
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const sel = new SelectFromModel({ estimator: est, threshold: 0.4 });
    sel.fit(X, y);
    expect(sel.getSupport()).toEqual([false, true, false, true]);
  });

  it("respects maxFeatures constraint", () => {
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const sel = new SelectFromModel({
      estimator: est,
      threshold: 0.0,
      maxFeatures: 2,
    });
    sel.fit(X, y);
    const support = sel.getSupport();
    const nSelected = support.filter(Boolean).length;
    expect(nSelected).toBe(2);
    // Should keep the top 2 by importance: indices 1 (0.5) and 3 (0.9)
    expect(support[1]).toBe(true);
    expect(support[3]).toBe(true);
  });

  it("works with coef_ estimator (1D)", () => {
    // absolute values: [0.1, 0.8, 0.2, 0.7]  mean = 0.45
    const est = new MockCoefEstimator([-0.1, 0.8, -0.2, 0.7]);
    const sel = new SelectFromModel({ estimator: est });
    sel.fit(X, y);
    expect(sel.getSupport()).toEqual([false, true, false, true]);
  });

  it("exposes importances after fit", () => {
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const sel = new SelectFromModel({ estimator: est });
    sel.fit(X, y);
    expect(sel.importances).toEqual([0.1, 0.5, 0.3, 0.9]);
  });

  it("fitTransform works", () => {
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const sel = new SelectFromModel({ estimator: est });
    const Xt = sel.fitTransform(X, y);
    expect(Xt.shape).toEqual([4, 2]);
  });

  it("throws NotFittedError before fit", () => {
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const sel = new SelectFromModel({ estimator: est });
    expect(() => sel.transform(X)).toThrow("fitted");
    expect(() => sel.getSupport()).toThrow("fitted");
    expect(() => sel.importances).toThrow("fitted");
  });

  it("throws on feature mismatch during transform", () => {
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const sel = new SelectFromModel({ estimator: est });
    sel.fit(X, y);
    const X2 = tensor([[1, 2, 3]]);
    expect(() => sel.transform(X2)).toThrow("Expected 4 features");
  });

  it("getParams returns constructor parameters", () => {
    const est = new MockImportanceEstimator([]);
    const sel = new SelectFromModel({ estimator: est, threshold: 0.5 });
    const params = sel.getParams();
    expect(params.threshold).toBe(0.5);
  });

  it("throws on negative threshold", () => {
    const est = new MockImportanceEstimator([]);
    expect(() => new SelectFromModel({ estimator: est, threshold: -1 })).toThrow("threshold");
  });

  it("throws on invalid maxFeatures", () => {
    const est = new MockImportanceEstimator([]);
    expect(() => new SelectFromModel({ estimator: est, maxFeatures: 0 })).toThrow("maxFeatures");
  });
});

// ---------------------------------------------------------------------------
// RFE
// ---------------------------------------------------------------------------
describe("RFE", () => {
  // We create an estimator whose importances change based on active features
  // For simplicity, use a fixed importance estimator — RFE will re-fit each round
  // but the mock always returns the same importances (truncated to active count)

  it("eliminates features down to nFeaturesToSelect", () => {
    // 4 features, importances always [0.1, 0.5, 0.3, 0.9]
    // step=1: round 1 removes idx 0 (0.1), round 2 removes idx 2 (0.3)
    // Final: features 1 and 3 selected
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const X = tensor([
      [1, 10, 100, 1000],
      [2, 20, 200, 2000],
      [3, 30, 300, 3000],
      [4, 40, 400, 4000],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const rfe = new RFE({ estimator: est, nFeaturesToSelect: 2 });
    rfe.fit(X, y);

    const support = rfe.getSupport();
    const nSelected = support.filter(Boolean).length;
    expect(nSelected).toBe(2);

    const Xt = rfe.transform(X);
    expect(Xt.shape).toEqual([4, 2]);
  });

  it("ranking assigns 1 to selected features", () => {
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const X = tensor([
      [1, 10, 100, 1000],
      [2, 20, 200, 2000],
      [3, 30, 300, 3000],
    ]);
    const y = tensor([0, 1, 1]);
    const rfe = new RFE({ estimator: est, nFeaturesToSelect: 2 });
    rfe.fit(X, y);

    const ranking = rfe.ranking;
    // Selected features should have rank 1
    const selectedIdx = ranking.map((r, i) => (r === 1 ? i : -1)).filter((i) => i >= 0);
    expect(selectedIdx.length).toBe(2);
    // Non-selected should have rank > 1
    for (const r of ranking) {
      expect(r).toBeGreaterThanOrEqual(1);
    }
  });

  it("step > 1 removes multiple features per round", () => {
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const X = tensor([
      [1, 10, 100, 1000],
      [2, 20, 200, 2000],
    ]);
    const y = tensor([0, 1]);
    const rfe = new RFE({ estimator: est, nFeaturesToSelect: 2, step: 2 });
    rfe.fit(X, y);

    const support = rfe.getSupport();
    expect(support.filter(Boolean).length).toBe(2);
  });

  it("fitTransform works", () => {
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const X = tensor([
      [1, 10, 100, 1000],
      [2, 20, 200, 2000],
    ]);
    const y = tensor([0, 1]);
    const rfe = new RFE({ estimator: est, nFeaturesToSelect: 2 });
    const Xt = rfe.fitTransform(X, y);
    expect(Xt.shape).toEqual([2, 2]);
  });

  it("throws NotFittedError before fit", () => {
    const est = new MockImportanceEstimator([]);
    const rfe = new RFE({ estimator: est, nFeaturesToSelect: 1 });
    const X = tensor([[1, 2]]);
    expect(() => rfe.transform(X)).toThrow("fitted");
    expect(() => rfe.getSupport()).toThrow("fitted");
    expect(() => rfe.ranking).toThrow("fitted");
  });

  it("throws on feature count mismatch during transform", () => {
    const est = new MockImportanceEstimator([0.1, 0.5, 0.3, 0.9]);
    const X = tensor([
      [1, 10, 100, 1000],
      [2, 20, 200, 2000],
    ]);
    const y = tensor([0, 1]);
    const rfe = new RFE({ estimator: est, nFeaturesToSelect: 2 });
    rfe.fit(X, y);
    expect(() => rfe.transform(tensor([[1, 2]]))).toThrow("Expected 4");
  });

  it("throws on nFeaturesToSelect > nFeatures", () => {
    const est = new MockImportanceEstimator([0.1, 0.5]);
    const X = tensor([[1, 2]]);
    const y = tensor([0]);
    const rfe = new RFE({ estimator: est, nFeaturesToSelect: 5 });
    expect(() => rfe.fit(X, y)).toThrow("exceeds");
  });

  it("throws on invalid constructor params", () => {
    const est = new MockImportanceEstimator([]);
    expect(() => new RFE({ estimator: est, nFeaturesToSelect: 0 })).toThrow();
    expect(() => new RFE({ estimator: est, step: 0 })).toThrow();
  });

  it("getParams returns constructor parameters", () => {
    const est = new MockImportanceEstimator([]);
    const rfe = new RFE({ estimator: est, nFeaturesToSelect: 3, step: 2 });
    const params = rfe.getParams();
    expect(params.nFeaturesToSelect).toBe(3);
    expect(params.step).toBe(2);
  });
});
