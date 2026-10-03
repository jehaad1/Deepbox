import { describe, expect, it } from "vitest";
import { MLPClassifier, MLPRegressor } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("MLPClassifier", () => {
  const X = tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ]);

  it("should classify linearly separable data", () => {
    // OR gate, linearly separable
    const y = tensor([0, 1, 1, 1]);
    const clf = new MLPClassifier({ hiddenLayerSizes: [4], maxIter: 500, learningRate: 0.1 });
    clf.fit(X, y);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([4]);
    const score = clf.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0.5);
  });

  it("should predict probabilities summing to 1", () => {
    const y = tensor([0, 1, 1, 1]);
    const clf = new MLPClassifier({ hiddenLayerSizes: [4], maxIter: 200 });
    clf.fit(X, y);
    const proba = clf.predictProba(tensor([[0.5, 0.5]]));
    expect(proba.ndim).toBe(2);
    expect(proba.shape[1]).toBe(2);
    const p0 = Number(proba.data[0]);
    const p1 = Number(proba.data[1]);
    expect(p0 + p1).toBeCloseTo(1, 5);
  });

  it("should handle multi-class", () => {
    const Xmc = tensor([
      [1, 0],
      [2, 0],
      [0, 1],
      [0, 2],
      [1, 1],
      [2, 2],
    ]);
    const ymc = tensor([0, 0, 1, 1, 2, 2]);
    const clf = new MLPClassifier({ hiddenLayerSizes: [8], maxIter: 300, learningRate: 0.05 });
    clf.fit(Xmc, ymc);
    const pred = clf.predict(Xmc);
    expect(pred.shape).toEqual([6]);
  });

  it("should expose classes", () => {
    const y = tensor([0, 1, 1, 0]);
    const clf = new MLPClassifier({ hiddenLayerSizes: [4], maxIter: 10 });
    clf.fit(X, y);
    expect(clf.classes).toBeDefined();
    expect(clf.classes!.shape).toEqual([2]);
  });

  it("should throw on empty hiddenLayerSizes", () => {
    expect(() => new MLPClassifier({ hiddenLayerSizes: [] })).toThrow();
  });

  it("should throw on invalid learningRate", () => {
    expect(() => new MLPClassifier({ learningRate: -1 })).toThrow();
  });

  it("should throw when predicting before fitting", () => {
    const clf = new MLPClassifier();
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow();
  });

  it("should get and set params", () => {
    const clf = new MLPClassifier({ maxIter: 100 });
    expect(clf.getParams().maxIter).toBe(100);
    clf.setParams({ maxIter: 50 });
    expect(clf.getParams().maxIter).toBe(50);
  });

  it("should throw on unknown param", () => {
    const clf = new MLPClassifier();
    expect(() => clf.setParams({ badParam: 1 })).toThrow();
  });

  it("should support tanh activation", () => {
    const y = tensor([0, 1, 1, 1]);
    const clf = new MLPClassifier({ hiddenLayerSizes: [4], activation: "tanh", maxIter: 200 });
    clf.fit(X, y);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([4]);
  });
});

describe("MLPRegressor", () => {
  const X = tensor([[1], [2], [3], [4], [5]]);
  const y = tensor([2, 4, 6, 8, 10]);

  it("should make predictions", () => {
    const reg = new MLPRegressor({ hiddenLayerSizes: [10], maxIter: 500, learningRate: 0.01 });
    reg.fit(X, y);
    const pred = reg.predict(X);
    expect(pred.shape).toEqual([5]);
  });

  it("should compute R^2 score", () => {
    const reg = new MLPRegressor({ hiddenLayerSizes: [10], maxIter: 500, learningRate: 0.01 });
    reg.fit(X, y);
    const score = reg.score(X, y);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should throw on empty hiddenLayerSizes", () => {
    expect(() => new MLPRegressor({ hiddenLayerSizes: [] })).toThrow();
  });

  it("should throw when predicting before fitting", () => {
    const reg = new MLPRegressor();
    expect(() => reg.predict(tensor([[1]]))).toThrow();
  });

  it("should get and set params", () => {
    const reg = new MLPRegressor({ maxIter: 100 });
    expect(reg.getParams().maxIter).toBe(100);
    reg.setParams({ maxIter: 50 });
    expect(reg.getParams().maxIter).toBe(50);
  });

  it("should throw on unknown param", () => {
    const reg = new MLPRegressor();
    expect(() => reg.setParams({ badParam: 1 })).toThrow();
  });
});
