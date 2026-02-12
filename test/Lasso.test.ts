import { describe, expect, it } from "vitest";
import { Lasso } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("deepbox/ml - Lasso", () => {
  it("should throw if predict is called before fit", () => {
    const model = new Lasso();
    expect(() => model.predict(tensor([[1, 2]]))).toThrow("Lasso must be fitted");
  });

  it("should validate alpha", () => {
    const model = new Lasso({ alpha: -1 });
    const X = tensor([[1], [2]]);
    const y = tensor([1, 2]);
    expect(() => model.fit(X, y)).toThrow("alpha must be >= 0");
  });

  it("should fit and predict a simple linear relationship", () => {
    const model = new Lasso({
      alpha: 0.0,
      maxIter: 2000,
      tol: 1e-8,
      fitIntercept: true,
    });
    const X = tensor([[0], [1], [2], [3]]);
    const y = tensor([1, 3, 5, 7]);

    model.fit(X, y);

    const pred = model.predict(X);
    expect(pred.shape).toEqual([4]);

    const r2 = model.score(X, y);
    expect(r2).toBeGreaterThan(0.9);
  });

  it("should produce near-zero coefficients for strong regularization", () => {
    const model = new Lasso({
      alpha: 100,
      maxIter: 2000,
      tol: 1e-8,
      fitIntercept: true,
    });
    const X = tensor([
      [1, 0],
      [0, 1],
      [1, 1],
      [2, 1],
    ]);
    const y = tensor([1, 1, 2, 3]);

    model.fit(X, y);
    const coef = model.coef;

    const c0 = Math.abs(Number(coef.data[coef.offset + 0]));
    const c1 = Math.abs(Number(coef.data[coef.offset + 1]));
    expect(c0 + c1).toBeLessThan(0.5);
  });

  it("should support normalization", () => {
    const model = new Lasso({ alpha: 0.0, normalize: true, maxIter: 1000 });
    const X = tensor([
      [1, 10],
      [2, 20],
      [3, 30],
    ]);
    const y = tensor([2, 4, 6]);
    model.fit(X, y);
    const pred = model.predict(X);
    expect(pred.shape).toEqual([3]);
  });

  it("setParams should reject unknown keys", () => {
    const model = new Lasso();
    expect(() => model.setParams({ unknown: 123 })).toThrow("Unknown parameter");
  });
});
