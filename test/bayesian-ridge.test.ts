import { describe, expect, it } from "vitest";
import { BayesianRidge } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("BayesianRidge", () => {
  // Simple linear data: y = 2*x + 1
  const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
  const y = tensor([3, 5, 7, 9, 11, 13, 15, 17]);

  it("fits linear data with high R²", () => {
    const reg = new BayesianRidge();
    reg.fit(X, y);
    const r2 = reg.score(X, y);
    expect(r2).toBeGreaterThan(0.95);
  });

  it("predict returns correct shape", () => {
    const reg = new BayesianRidge();
    reg.fit(X, y);
    const pred = reg.predict(X);
    expect(pred.size).toBe(8);
  });

  it("coef is close to true value", () => {
    const reg = new BayesianRidge();
    reg.fit(X, y);
    // True slope is 2
    expect(reg.coef[0]).toBeCloseTo(2, 0);
  });

  it("intercept is close to true value", () => {
    const reg = new BayesianRidge();
    reg.fit(X, y);
    // True intercept is 1
    expect(reg.intercept).toBeCloseTo(1, 0);
  });

  it("alpha and lambda are positive after fit", () => {
    const reg = new BayesianRidge();
    reg.fit(X, y);
    expect(reg.alpha).toBeGreaterThan(0);
    expect(reg.lambda).toBeGreaterThan(0);
  });

  it("nIter is positive after fit", () => {
    const reg = new BayesianRidge();
    reg.fit(X, y);
    expect(reg.nIter).toBeGreaterThan(0);
  });

  it("multivariate regression", () => {
    const Xm = tensor([
      [1, 1],
      [2, 1],
      [3, 1],
      [1, 2],
      [2, 2],
      [3, 2],
      [1, 3],
      [2, 3],
      [3, 3],
      [4, 4],
    ]);
    const ym = tensor([6, 8, 10, 9, 11, 13, 12, 14, 16, 20]);
    const reg = new BayesianRidge();
    reg.fit(Xm, ym);
    expect(reg.coef.length).toBe(2);
    const r2 = reg.score(Xm, ym);
    expect(r2).toBeGreaterThan(0.9);
  });

  it("automatic regularization handles near-collinear data", () => {
    // Near-collinear: x2 ≈ x1
    const Xcol = tensor([
      [1, 1.01],
      [2, 2.01],
      [3, 3.01],
      [4, 4.01],
      [5, 5.01],
    ]);
    const ycol = tensor([3, 5, 7, 9, 11]);
    const reg = new BayesianRidge();
    reg.fit(Xcol, ycol);
    // Should still produce reasonable predictions
    const r2 = reg.score(Xcol, ycol);
    expect(r2).toBeGreaterThan(0.5);
  });

  it("throws when not fitted", () => {
    const reg = new BayesianRidge();
    expect(() => reg.predict(X)).toThrow();
    expect(() => reg.coef).toThrow();
    expect(() => reg.intercept).toThrow();
    expect(() => reg.alpha).toThrow();
    expect(() => reg.lambda).toThrow();
    expect(() => reg.nIter).toThrow();
  });

  it("throws for invalid maxIter", () => {
    expect(() => new BayesianRidge({ maxIter: 0 })).toThrow();
  });

  it("throws for invalid alphaInit", () => {
    expect(() => new BayesianRidge({ alphaInit: 0 })).toThrow();
    expect(() => new BayesianRidge({ alphaInit: -1 })).toThrow();
  });

  it("throws for invalid lambdaInit", () => {
    expect(() => new BayesianRidge({ lambdaInit: 0 })).toThrow();
  });

  it("getParams returns options", () => {
    const reg = new BayesianRidge({ maxIter: 200, tol: 1e-4 });
    const params = reg.getParams();
    expect(params.maxIter).toBe(200);
    expect(params.tol).toBe(1e-4);
  });

  it("fitIntercept=false works", () => {
    // y = 2*x (no intercept, data passes through origin)
    const X0 = tensor([[1], [2], [3], [4], [5]]);
    const y0 = tensor([2, 4, 6, 8, 10]);
    const reg = new BayesianRidge({ fitIntercept: false });
    reg.fit(X0, y0);
    expect(reg.intercept).toBe(0);
    expect(reg.coef[0]).toBeCloseTo(2, 0);
  });
});
