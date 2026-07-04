import { describe, expect, it } from "vitest";
import { ElasticNet } from "../src/ml/linear/ElasticNet";
import { tensor } from "../src/ndarray";

const X = tensor([
  [1, 2],
  [3, 4],
  [5, 6],
  [7, 8],
]);
const y = tensor([3, 7, 11, 15]);

describe("ElasticNet", () => {
  // ── Basic fit/predict ──
  it("fits and predicts", () => {
    const model = new ElasticNet({ alpha: 0.01 });
    model.fit(X, y);
    const pred = model.predict(X);
    expect(pred.shape).toEqual([4]);
  });

  it("score returns R²", () => {
    const model = new ElasticNet({ alpha: 0.01 });
    model.fit(X, y);
    const r2 = model.score(X, y);
    expect(r2).toBeGreaterThan(0.5);
  });

  // ── Accessors ──
  it("coef accessor", () => {
    const model = new ElasticNet({ alpha: 0.01 });
    model.fit(X, y);
    expect(model.coef.ndim).toBe(1);
  });

  it("intercept accessor", () => {
    const model = new ElasticNet({ alpha: 0.01 });
    model.fit(X, y);
    expect(typeof model.intercept).toBe("number");
  });

  it("nIter accessor", () => {
    const model = new ElasticNet({ alpha: 0.01 });
    model.fit(X, y);
    expect(typeof model.nIter).toBe("number");
  });

  // ── Not fitted errors ──
  it("predict before fit throws", () => {
    expect(() => new ElasticNet().predict(X)).toThrow(/fitted/i);
  });

  it("score before fit throws", () => {
    expect(() => new ElasticNet().score(X, y)).toThrow(/fitted/i);
  });

  it("coef before fit throws", () => {
    expect(() => new ElasticNet().coef).toThrow(/fitted/i);
  });

  it("intercept before fit throws", () => {
    expect(() => new ElasticNet().intercept).toThrow(/fitted/i);
  });

  it("nIter before fit throws", () => {
    expect(() => new ElasticNet().nIter).toThrow(/fitted/i);
  });

  // ── Options branches ──
  it("fitIntercept=false", () => {
    const model = new ElasticNet({ alpha: 0.01, fitIntercept: false });
    model.fit(X, y);
    expect(model.intercept).toBe(0);
  });

  it("normalize=true", () => {
    const model = new ElasticNet({ alpha: 0.01, normalize: true });
    model.fit(X, y);
    const pred = model.predict(X);
    expect(pred.shape).toEqual([4]);
  });

  it("positive=true forces positive coefs", () => {
    const model = new ElasticNet({ alpha: 0.01, positive: true });
    model.fit(X, y);
    const coefs = Array.from(model.coef.data as Float64Array);
    for (const c of coefs) {
      expect(c).toBeGreaterThanOrEqual(0);
    }
  });

  it("selection=random", () => {
    const model = new ElasticNet({
      alpha: 0.01,
      selection: "random",
      randomState: 42,
    });
    model.fit(X, y);
    const pred = model.predict(X);
    expect(pred.shape).toEqual([4]);
  });

  it("warmStart reuses coefficients", () => {
    const model = new ElasticNet({ alpha: 0.01, warmStart: true });
    model.fit(X, y);
    model.fit(X, y);
    const pred = model.predict(X);
    expect(pred.shape).toEqual([4]);
  });

  it("l1Ratio=1 (pure Lasso)", () => {
    const model = new ElasticNet({ alpha: 0.1, l1Ratio: 1 });
    model.fit(X, y);
    expect(model.predict(X).shape).toEqual([4]);
  });

  it("l1Ratio=0 (pure Ridge)", () => {
    const model = new ElasticNet({ alpha: 0.1, l1Ratio: 0 });
    model.fit(X, y);
    expect(model.predict(X).shape).toEqual([4]);
  });

  // ── Validation errors ──
  it("invalid l1Ratio", () => {
    expect(() => new ElasticNet({ l1Ratio: -0.1 }).fit(X, y)).toThrow(/l1Ratio/);
    expect(() => new ElasticNet({ l1Ratio: 1.5 }).fit(X, y)).toThrow(/l1Ratio/);
  });

  it("invalid alpha", () => {
    expect(() => new ElasticNet({ alpha: -1 }).fit(X, y)).toThrow(/alpha/);
  });

  // ── getParams / setParams ──
  it("getParams returns options", () => {
    const model = new ElasticNet({ alpha: 0.5 });
    const params = model.getParams();
    expect(params.alpha).toBe(0.5);
  });

  it("setParams updates options", () => {
    const model = new ElasticNet();
    model.setParams({ alpha: 0.2, l1Ratio: 0.3, maxIter: 500, tol: 1e-5 });
    const params = model.getParams();
    expect(params.alpha).toBe(0.2);
    expect(params.l1Ratio).toBe(0.3);
    expect(params.maxIter).toBe(500);
    expect(params.tol).toBe(1e-5);
  });

  it("setParams boolean options", () => {
    const model = new ElasticNet();
    model.setParams({
      fitIntercept: false,
      normalize: true,
      warmStart: true,
      positive: true,
    });
    const params = model.getParams();
    expect(params.fitIntercept).toBe(false);
    expect(params.normalize).toBe(true);
    expect(params.warmStart).toBe(true);
    expect(params.positive).toBe(true);
  });

  it("setParams selection", () => {
    const model = new ElasticNet();
    model.setParams({ selection: "random" });
    expect(model.getParams().selection).toBe("random");
  });

  it("setParams randomState", () => {
    const model = new ElasticNet();
    model.setParams({ randomState: 123 });
    expect(model.getParams().randomState).toBe(123);
  });

  it("setParams rejects invalid alpha", () => {
    expect(() => new ElasticNet().setParams({ alpha: "bad" })).toThrow();
  });

  it("setParams rejects invalid l1Ratio", () => {
    expect(() => new ElasticNet().setParams({ l1Ratio: 2 })).toThrow();
  });

  it("setParams rejects invalid maxIter", () => {
    expect(() => new ElasticNet().setParams({ maxIter: NaN })).toThrow();
  });

  it("setParams rejects invalid tol", () => {
    expect(() => new ElasticNet().setParams({ tol: Infinity })).toThrow();
  });

  it("setParams rejects invalid fitIntercept", () => {
    expect(() => new ElasticNet().setParams({ fitIntercept: 1 })).toThrow();
  });

  it("setParams rejects invalid normalize", () => {
    expect(() => new ElasticNet().setParams({ normalize: "yes" })).toThrow();
  });

  it("setParams rejects invalid warmStart", () => {
    expect(() => new ElasticNet().setParams({ warmStart: 0 })).toThrow();
  });

  it("setParams rejects invalid positive", () => {
    expect(() => new ElasticNet().setParams({ positive: null })).toThrow();
  });

  it("setParams rejects invalid selection", () => {
    expect(() => new ElasticNet().setParams({ selection: "invalid" })).toThrow();
  });

  it("setParams rejects invalid randomState", () => {
    expect(() => new ElasticNet().setParams({ randomState: NaN })).toThrow();
  });

  it("setParams rejects unknown param", () => {
    expect(() => new ElasticNet().setParams({ unknown: 1 })).toThrow(/unknown/i);
  });

  // ── Score edge cases ──
  it("score with constant y", () => {
    const constY = tensor([5, 5, 5, 5]);
    const model = new ElasticNet({ alpha: 0.01 });
    model.fit(X, constY);
    const r2 = model.score(X, constY);
    expect(typeof r2).toBe("number");
  });
});
