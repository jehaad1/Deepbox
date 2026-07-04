import { describe, expect, it } from "vitest";
import { GaussianProcessClassifier } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("GaussianProcessClassifier", () => {
  const X = tensor([
    [1, 1],
    [2, 2],
    [3, 3],
    [8, 8],
    [9, 9],
    [10, 10],
  ]);
  const y = tensor([0, 0, 0, 1, 1, 1]);

  it("should classify well-separated data", () => {
    const gpc = new GaussianProcessClassifier();
    gpc.fit(X, y);
    const pred = gpc.predict(X);
    expect(pred.shape).toEqual([6]);
    const score = gpc.score(X, y);
    expect(score).toBeGreaterThan(0.5);
  });

  it("should predict probabilities summing to 1", () => {
    const gpc = new GaussianProcessClassifier();
    gpc.fit(X, y);
    const proba = gpc.predictProba(tensor([[5, 5]]));
    expect(proba.ndim).toBe(2);
    expect(proba.shape[1]).toBe(2);
    const p0 = Number(proba.data[0]);
    const p1 = Number(proba.data[1]);
    expect(p0 + p1).toBeCloseTo(1, 5);
    expect(p0).toBeGreaterThanOrEqual(0);
    expect(p1).toBeGreaterThanOrEqual(0);
  });

  it("should expose classes", () => {
    const gpc = new GaussianProcessClassifier();
    gpc.fit(X, y);
    expect(gpc.classes).toBeDefined();
    expect(gpc.classes!.shape).toEqual([2]);
  });

  it("should throw on invalid lengthScale", () => {
    expect(() => new GaussianProcessClassifier({ lengthScale: -1 })).toThrow();
  });

  it("should throw on invalid kernelVariance", () => {
    expect(() => new GaussianProcessClassifier({ kernelVariance: 0 })).toThrow();
  });

  it("should throw when predicting before fitting", () => {
    const gpc = new GaussianProcessClassifier();
    expect(() => gpc.predict(tensor([[1, 2]]))).toThrow();
  });

  it("should get params", () => {
    const gpc = new GaussianProcessClassifier({ lengthScale: 2.0 });
    const params = gpc.getParams();
    expect(params.lengthScale).toBe(2.0);
  });

  it("should throw on unknown param", () => {
    const gpc = new GaussianProcessClassifier();
    expect(() => gpc.setParams({ badParam: 1 })).toThrow();
  });

  it("should handle multi-class", () => {
    const Xmc = tensor([
      [1, 1],
      [1, 2],
      [2, 1],
      [10, 1],
      [10, 2],
      [11, 1],
      [5, 10],
      [5, 11],
      [6, 10],
    ]);
    const ymc = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]);

    const gpc = new GaussianProcessClassifier();
    gpc.fit(Xmc, ymc);
    const pred = gpc.predict(Xmc);
    expect(pred.shape).toEqual([9]);
  });
});
