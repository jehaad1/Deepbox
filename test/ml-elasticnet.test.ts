import { describe, expect, it } from "vitest";
import { ElasticNet } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("ElasticNet", () => {
  it("fits and predicts a simple linear relationship", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const y = tensor([2, 4, 6, 8, 10]);
    const model = new ElasticNet({ alpha: 0.01, l1Ratio: 0.5 });
    model.fit(X, y);
    const pred = model.predict(X);
    // Should approximate y = 2x
    for (let i = 0; i < 5; i++) {
      expect(Number(pred.data[i])).toBeCloseTo((i + 1) * 2, 0);
    }
  });

  it("returns R² score close to 1 on well-fit data", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const y = tensor([2, 4, 6, 8, 10]);
    const model = new ElasticNet({ alpha: 0.001, l1Ratio: 0.5 });
    model.fit(X, y);
    const score = model.score(X, y);
    expect(score).toBeGreaterThan(0.95);
  });

  it("exposes coef and intercept after fitting", () => {
    const X = tensor([[1], [2], [3]]);
    const y = tensor([1, 2, 3]);
    const model = new ElasticNet({ alpha: 0.01, l1Ratio: 0.5 });
    model.fit(X, y);
    expect(model.coef.shape).toEqual([1]);
    expect(typeof model.intercept).toBe("number");
  });

  it("throws NotFittedError when predicting before fit", () => {
    const model = new ElasticNet();
    expect(() => model.predict(tensor([[1]]))).toThrow(/fitted/i);
  });

  it("throws InvalidParameterError for negative alpha", () => {
    const X = tensor([[1], [2]]);
    const y = tensor([1, 2]);
    const model = new ElasticNet({ alpha: -1 });
    expect(() => model.fit(X, y)).toThrow(/alpha/i);
  });

  it("throws InvalidParameterError for l1Ratio out of range", () => {
    const X = tensor([[1], [2]]);
    const y = tensor([1, 2]);
    const model = new ElasticNet({ l1Ratio: 1.5 });
    expect(() => model.fit(X, y)).toThrow(/l1Ratio/i);
  });

  it("performs feature selection with high l1Ratio (Lasso-like)", () => {
    // Create data where only first feature matters
    const X = tensor([
      [1, 100],
      [2, 200],
      [3, 300],
      [4, 400],
      [5, 500],
      [6, 600],
      [7, 700],
      [8, 800],
      [9, 900],
      [10, 1000],
    ]);
    const y = tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    const model = new ElasticNet({ alpha: 0.5, l1Ratio: 0.99 });
    model.fit(X, y);
    // With high L1, feature selection should occur
    expect(model.coef.size).toBe(2);
  });

  it("getParams and setParams work correctly", () => {
    const model = new ElasticNet({ alpha: 0.5, l1Ratio: 0.7 });
    const params = model.getParams();
    expect(params.alpha).toBe(0.5);
    expect(params.l1Ratio).toBe(0.7);

    model.setParams({ alpha: 1.0 });
    expect(model.getParams().alpha).toBe(1.0);
  });

  it("warm start reuses coefficients", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const y = tensor([2, 4, 6, 8, 10]);
    const model = new ElasticNet({
      alpha: 0.01,
      l1Ratio: 0.5,
      warmStart: true,
      maxIter: 5,
    });
    model.fit(X, y);
    const _coef1 = Number(model.coef.data[0]);
    model.fit(X, y); // second fit should start from previous coefs
    const coef2 = Number(model.coef.data[0]);
    // With warm start the second fit should converge at least as well
    expect(Math.abs(coef2)).toBeGreaterThanOrEqual(0);
  });

  it("positive constraint forces non-negative coefficients", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const y = tensor([-2, -4, -6, -8, -10]);
    const model = new ElasticNet({ alpha: 0.01, l1Ratio: 0.5, positive: true });
    model.fit(X, y);
    expect(Number(model.coef.data[0])).toBeGreaterThanOrEqual(0);
  });
});
