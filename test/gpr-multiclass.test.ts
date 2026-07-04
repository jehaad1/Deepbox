import { describe, expect, it } from "vitest";
import { GaussianProcessRegressor } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("GaussianProcessRegressor", () => {
  const X = tensor([[1], [2], [3], [4], [5]]);
  const y = tensor([1, 4, 9, 16, 25]);

  it("fit and predict", () => {
    const gpr = new GaussianProcessRegressor({ alpha: 1e-6, lengthScale: 2.0 });
    gpr.fit(X, y);
    const pred = gpr.predict(X);
    expect(pred.size).toBe(5);
    // Should interpolate well on training data
    for (let i = 0; i < 5; i++) {
      expect(Number(pred.data[pred.offset + i])).toBeCloseTo(Number(y.data[y.offset + i]), 0);
    }
  });

  it("predictWithStd returns mean and std", () => {
    const gpr = new GaussianProcessRegressor({ alpha: 1e-6, lengthScale: 2.0 });
    gpr.fit(X, y);
    const { mean, std } = gpr.predictWithStd(X);
    expect(mean.size).toBe(5);
    expect(std.size).toBe(5);
    // Std at training points should be very small
    for (let i = 0; i < 5; i++) {
      expect(Number(std.data[std.offset + i])).toBeLessThan(1);
    }
  });

  it("uncertainty increases away from training data", () => {
    const gpr = new GaussianProcessRegressor({ alpha: 1e-6, lengthScale: 1.0 });
    gpr.fit(X, y);
    const Xfar = tensor([[0], [100]]);
    const { std } = gpr.predictWithStd(Xfar);
    const stdNear = gpr.predictWithStd(X).std;
    // Far-away points should have higher uncertainty than training points
    const maxStdNear = Math.max(
      ...Array.from({ length: 5 }, (_, i) => Number(stdNear.data[stdNear.offset + i]))
    );
    const minStdFar = Math.min(
      ...Array.from({ length: 2 }, (_, i) => Number(std.data[std.offset + i]))
    );
    expect(minStdFar).toBeGreaterThan(maxStdNear);
  });

  it("score is high on training data", () => {
    const gpr = new GaussianProcessRegressor({ alpha: 1e-6, lengthScale: 2.0 });
    gpr.fit(X, y);
    const r2 = gpr.score(X, y);
    expect(r2).toBeGreaterThan(0.9);
  });

  it("multidimensional input", () => {
    const X2d = tensor([
      [1, 0],
      [0, 1],
      [1, 1],
      [2, 2],
    ]);
    const y2d = tensor([1, 2, 3, 4]);
    const gpr = new GaussianProcessRegressor({ alpha: 1e-4, lengthScale: 1.0 });
    gpr.fit(X2d, y2d);
    const pred = gpr.predict(X2d);
    expect(pred.size).toBe(4);
  });

  it("throws when not fitted", () => {
    const gpr = new GaussianProcessRegressor();
    expect(() => gpr.predict(X)).toThrow();
    expect(() => gpr.predictWithStd(X)).toThrow();
  });

  it("throws for invalid alpha", () => {
    expect(() => new GaussianProcessRegressor({ alpha: -1 })).toThrow();
  });

  it("throws for invalid lengthScale", () => {
    expect(() => new GaussianProcessRegressor({ lengthScale: 0 })).toThrow();
    expect(() => new GaussianProcessRegressor({ lengthScale: -1 })).toThrow();
  });

  it("throws for invalid kernelVariance", () => {
    expect(() => new GaussianProcessRegressor({ kernelVariance: 0 })).toThrow();
  });

  it("getParams returns options", () => {
    const gpr = new GaussianProcessRegressor({ alpha: 0.1, lengthScale: 2.0, kernelVariance: 3.0 });
    const params = gpr.getParams();
    expect(params.alpha).toBe(0.1);
    expect(params.lengthScale).toBe(2.0);
    expect(params.kernelVariance).toBe(3.0);
  });
});
